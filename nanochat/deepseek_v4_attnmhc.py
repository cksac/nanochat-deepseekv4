"""
DeepSeek-V4-style nanochat model: mHc + Block Attention Residuals hybrid.

Combines manifold-constrained Hyper-Connections (mHc) from deepseek_v4.py with
Block AttnRes from deepseek_v4_attnres.py. The mHc handles intra-layer multi-slot
residual mixing (A_l/B_l/C_l maps), while block AttnRes adds inter-block depth
aggregation with learned attention over completed block representations.

Architecture:
- Hidden state: [B, T, M, C] (M=hc_mult slots, as in deepseek_v4)
- Layers use HyperConnectionMixer for A/B/C maps (unchanged)
- At block boundaries, snapshot the multi-slot state as a block representation
- Block AttnRes applies learned attention over block reps to produce the
  initial state for the next block, reducing representational drift

Key differences from deepseek_v4.py:
- Added: block_attn_res_mhc() for multi-slot block attention
- Added: attn_res_block_size config parameter
- Added: per-block learned projections for AttnRes queries + sink logits
- The head_mix is unchanged — still uses HyperConnectionMixer to reduce to 1 slot

Key differences from deepseek_v4_attnres.py:
- Keeps full mHc machinery (multi-slot [B,T,M,C] state)
- AttnRes operates over [B,T,M*C] block snapshots rather than [B,T,C]
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import Linear, norm, apply_rotary_emb

# Import shared components from deepseek_v4
from nanochat.deepseek_v4 import (
    DeepSeekV4NanoConfig,
    HyperConnectionMixer,
    hyper_post,
    sinkhorn,
    DeepSeekHybridAttention,
    DeepSeekMoE,
    GroupedOutputProjection,
    _num_hash_layers,
    _default_rank,
    _parse_compress_ratios,
    _attention_kind,
    _rope_dim,
    precompute_yarn_rotary,
)


@dataclass
class DeepSeekV4AttnMhcConfig(DeepSeekV4NanoConfig):
    """Extends DeepSeekV4NanoConfig with block AttnRes parameters."""
    attention_layer_pattern: str = "HCA,HCA,CSA"
    attn_res_block_size: int = 4  # layers per block


# ─── Block Attention Residuals for multi-slot state ────────────────────────────

def block_attn_res_mhc(blocks: list[torch.Tensor], current: torch.Tensor,
                       proj_weight: torch.Tensor, sink_logit: torch.Tensor) -> torch.Tensor:
    """
    Inter-block attention for multi-slot mHc state.

    blocks: list of N tensors of shape [B, T, M, C] — completed block snapshots
    current: [B, T, M, C] — current multi-slot state
    proj_weight: [M*C] — learned pseudo-query vector
    sink_logit: scalar — "attend to nothing" logit
    Returns: [B, T, M, C] — blended state
    """
    B, T, M, C = current.shape
    # Stack all candidates: [N+1, B, T, M*C]
    all_states = [b.flatten(2) for b in blocks] + [current.flatten(2)]
    V = torch.stack(all_states)  # [N+1, B, T, M*C]
    K = norm(V)
    # Compute attention logits
    logits = torch.einsum("d, n b t d -> n b t", proj_weight.to(K.dtype), K)  # [N+1, B, T]
    # Attention sink
    sink = sink_logit.expand(1, B, T)  # [1, B, T]
    logits_with_sink = torch.cat([logits, sink], dim=0)  # [N+2, B, T]
    weights = logits_with_sink.float().softmax(0).to(V.dtype)
    # Weighted sum over block values only (discard sink weight)
    h = torch.einsum("n b t, n b t d -> b t d", weights[:V.shape[0]], V)
    return h.view(B, T, M, C)


# ─── Transformer Block: mHc + Block AttnRes ───────────────────────────────────

class DeepSeekV4AttnMhcBlock(nn.Module):
    """
    Transformer block combining mHc and Block AttnRes.

    - mHc (HyperConnectionMixer) handles per-layer residual mixing in multi-slot space
    - Block AttnRes adds inter-block depth aggregation at block boundaries
    """

    def __init__(self, config: DeepSeekV4AttnMhcConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = config.attn_res_block_size
        self.attn_mix = HyperConnectionMixer(config)
        self.ffn_mix = HyperConnectionMixer(config)
        self.attn = DeepSeekHybridAttention(config, layer_idx)
        self.moe = DeepSeekMoE(config, layer_idx)
        # Block AttnRes projections (operate on flattened M*C dim)
        mc_dim = config.hc_mult * config.n_embd
        self.attn_res_proj = nn.Parameter(torch.zeros(mc_dim))
        self.attn_res_sink = nn.Parameter(torch.zeros(()))

    def _is_block_boundary(self):
        """Whether this layer starts a new block."""
        return self.layer_idx > 0 and self.layer_idx % self.block_size == 0

    def forward(self, blocks: list[torch.Tensor], x: torch.Tensor, input_ids, cos_sin):
        # x: [B, T, M, C]
        # At block boundary: snapshot state and blend via AttnRes
        if self._is_block_boundary():
            blocks = blocks + [x]
            # Apply block AttnRes to produce blended starting state for new block
            x = block_attn_res_mhc(blocks, x, self.attn_res_proj, self.attn_res_sink)

        # Standard mHc attention path
        residual = x
        y, post, comb = self.attn_mix(x)
        y = self.attn(norm(y), cos_sin)
        x = hyper_post(y, residual, post, comb)

        # Standard mHc FFN path
        residual = x
        y, post, comb = self.ffn_mix(x)
        y = self.moe(norm(y), input_ids)
        x = hyper_post(y, residual, post, comb)

        return blocks, x


# ─── Multi-Token Prediction ───────────────────────────────────────────────────

class MTPPredictionHead(nn.Module):
    """Multi-token prediction head."""

    def __init__(self, config: DeepSeekV4AttnMhcConfig, padded_vocab_size: int):
        super().__init__()
        self.e_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.h_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.head = Linear(config.n_embd, padded_vocab_size, bias=False)

    def forward(self, hidden, token_embed):
        x = self.e_proj(norm(token_embed)) + self.h_proj(norm(hidden))
        return self.head(norm(x))


# ─── Top-Level Model ──────────────────────────────────────────────────────────

class DeepSeekV4AttnMhcChat(nn.Module):
    def __init__(self, config: DeepSeekV4AttnMhcConfig, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([DeepSeekV4AttnMhcBlock(config, i) for i in range(config.n_layer)]),
        })
        self.head_mix = HyperConnectionMixer(config)
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList([
            MTPPredictionHead(config, padded_vocab_size)
            for _ in range(config.num_nextn_predict_layers)
        ])
        self.rotary_seq_len = max(config.sequence_len * 2, 4096)
        cos, sin = precompute_yarn_rotary(self.rotary_seq_len, _rope_dim(config), config)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        bound = 3**0.5 * n_embd**-0.5
        output_heads = {self.lm_head, *(head.head for head in self.mtp_heads)}
        for module in self.modules():
            if isinstance(module, Linear) and module not in output_heads:
                torch.nn.init.uniform_(module.weight, -bound, bound)
            elif isinstance(module, GroupedOutputProjection):
                module.init_weights(bound)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for head in self.mtp_heads:
            torch.nn.init.normal_(head.head.weight, mean=0.0, std=0.001)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.o_proj.proj_b.weight)
            # Initialize attn_res projections small for stable start
            torch.nn.init.normal_(block.attn_res_proj, mean=0.0, std=0.02)

        cos, sin = precompute_yarn_rotary(self.rotary_seq_len, _rope_dim(self.config), self.config, device=self.transformer.wte.weight.device)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def _hidden(self, idx):
        B, T = idx.shape
        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        # Initialize multi-slot state [B, T, M, C] with normalized embedding
        x = norm(x).unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)

        # Initialize block list with initial state
        blocks = [x]

        for block in self.transformer.h:
            blocks, x = block(blocks, x, idx, cos_sin)

        # Reduce multi-slot to single via head_mix
        y, _, _ = self.head_mix(x)
        return norm(y)

    def _sequence_balance_loss(self):
        losses = []
        for block in self.transformer.h:
            loss = block.moe.last_sequence_balance_loss
            if loss is not None:
                losses.append(loss)
        if not losses:
            return None
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update_aux_free_balance(self):
        for block in self.transformer.h:
            block.moe.update_aux_free_balance()

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", include_aux_loss=True):
        del kv_cache
        hidden = self._hidden(idx)
        softcap = self.config.logit_softcap
        logits = self.lm_head(hidden)[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is None:
            return logits

        primary_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if not include_aux_loss or not self.mtp_heads or loss_reduction != "mean":
            balance = self._sequence_balance_loss()
            if include_aux_loss and balance is not None:
                return primary_loss + self.config.sequence_balance_loss_weight * balance
            return primary_loss

        aux_losses = []
        for offset, head in enumerate(self.mtp_heads, start=1):
            if hidden.size(1) <= offset:
                continue
            mtp_embed = self.transformer.wte(idx[:, offset:]).to(hidden.dtype)
            aux_logits = head(hidden[:, :-offset], mtp_embed)[..., :self.config.vocab_size].float()
            aux_logits = softcap * torch.tanh(aux_logits / softcap)
            aux_targets = targets[:, offset:]
            aux_losses.append(F.cross_entropy(
                aux_logits.reshape(-1, aux_logits.size(-1)),
                aux_targets.reshape(-1),
                ignore_index=-1,
            ))
        extra_loss = primary_loss.new_zeros(())
        if aux_losses:
            extra_loss = extra_loss + self.config.mtp_loss_weight * torch.stack(aux_losses).mean()
        balance = self._sequence_balance_loss()
        if balance is not None:
            extra_loss = extra_loss + self.config.sequence_balance_loss_weight * balance
        return primary_loss + extra_loss

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        active_expert_frac = self.config.num_experts_per_tok / max(1, self.config.n_routed_experts)
        dense_fraction = 0.35
        return int(6 * nparams * (dense_fraction + active_expert_frac))

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        mtp = sum(p.numel() for p in self.mtp_heads.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "wte": wte,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "mtp": mtp,
            "total": total,
        }
