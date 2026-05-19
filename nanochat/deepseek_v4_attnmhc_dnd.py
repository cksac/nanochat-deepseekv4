"""
DeepSeek-V4-style nanochat model: mHc + Block AttnRes + DND layers.

Combines:
- manifold-constrained Hyper-Connections (mHc) for multi-slot residual mixing
- Block AttnRes for inter-block depth aggregation
- DND (DeltaNet + Diffusion) attention layers that use previous-layer residuals
  as conditioning for a diffusion-based future-state predictor

Layer pattern: CSA,HCA,DND
- CSA: Conventional Self-Attention with Lightning Indexer
- HCA: Hybrid Compressed Attention with simple causal masking
- DND: DeltaNet + Diffusion (linear attention + future-state prediction)

The DND layer collects residuals from previous layers within the same attention
block (via the mHc multi-slot state snapshots) and uses them to condition a
lightweight denoiser that predicts N future states. The final output is a
learned mixture of the DeltaNet output and the predicted future states.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import Linear, norm, apply_rotary_emb
from nanochat.deltanet_diffusion import DeltaNetDiffusionAttention

from nanochat.deepseek_v4 import (
    DeepSeekV4NanoConfig,
    HyperConnectionMixer,
    hyper_post,
    sinkhorn,
    DeepSeekHybridAttention,
    DeepSeekMoE,
    GroupedOutputProjection,
    MTPPredictionHead,
    _num_hash_layers,
    _default_rank,
    _parse_compress_ratios,
    _attention_kind,
    _rope_dim,
    precompute_yarn_rotary,
)


def _attention_kind_dnd(config, layer_idx: int) -> str:
    """Extended attention kind parser that accepts 'DND' for DeltaNet+Diffusion."""
    raw = config.attention_layer_pattern
    if "+" in raw:
        prefix_str, cycle_str = raw.split("+", 1)
        prefix = [p.strip().upper() for p in prefix_str.split(",") if p.strip()]
        cycle = [p.strip().upper() for p in cycle_str.split(",") if p.strip()]
        if layer_idx < len(prefix):
            kind = prefix[layer_idx]
        else:
            kind = cycle[(layer_idx - len(prefix)) % len(cycle)] if cycle else "CSA"
    else:
        pattern = [part.strip().upper() for part in raw.split(",") if part.strip()]
        if not pattern:
            return "CSA"
        kind = pattern[layer_idx % len(pattern)]
    assert kind in {"CSA", "HCA", "SWA", "DN", "DND"}, f"Unknown attention kind: {kind}"
    return kind


@dataclass
class DeepSeekV4AttnMhcDndConfig(DeepSeekV4NanoConfig):
    """mHc + Block AttnRes + DND layers config."""
    attention_layer_pattern: str = "CSA,HCA,DND"
    attn_res_block_size: int = 4
    dnd_n_future: int = 2       # number of future states to predict
    dnd_max_cond: int = 2       # max previous layers used as conditioning
    dnd_conv_size: int = 4


# ─── Block Attention Residuals for multi-slot state ────────────────────────────

def block_attn_res_mhc(blocks: list[torch.Tensor], current: torch.Tensor,
                       proj_weight: torch.Tensor, sink_logit: torch.Tensor) -> torch.Tensor:
    """Inter-block attention for multi-slot mHc state."""
    B, T, M, C = current.shape
    all_states = [b.flatten(2) for b in blocks] + [current.flatten(2)]
    V = torch.stack(all_states)
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", proj_weight.to(K.dtype), K)
    sink = sink_logit.expand(1, B, T)
    logits_with_sink = torch.cat([logits, sink], dim=0)
    weights = logits_with_sink.float().softmax(0).to(V.dtype)
    h = torch.einsum("n b t, n b t d -> b t d", weights[:V.shape[0]], V)
    return h.view(B, T, M, C)


# ─── DND Block ────────────────────────────────────────────────────────────────

class DeepSeekV4AttnMhcDndBlock(nn.Module):
    """
    Transformer block combining mHc, Block AttnRes, and DND attention.

    For DND layers, the block passes residuals from previous layers within
    the same attention block to condition the diffusion future-state predictor.
    """

    def __init__(self, config: DeepSeekV4AttnMhcDndConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = config.attn_res_block_size
        self.attn_mix = HyperConnectionMixer(config)
        self.ffn_mix = HyperConnectionMixer(config)

        kind = _attention_kind_dnd(config, layer_idx)
        self.kind = kind
        if kind == "DND":
            self.attn = DeltaNetDiffusionAttention(
                n_embd=config.n_embd,
                n_head=config.n_head,
                n_future=config.dnd_n_future,
                max_cond_layers=config.dnd_max_cond,
                conv_size=config.dnd_conv_size,
            )
        else:
            self.attn = DeepSeekHybridAttention(config, layer_idx)

        self.moe = DeepSeekMoE(config, layer_idx)
        # Block AttnRes projections
        mc_dim = config.hc_mult * config.n_embd
        self.attn_res_proj = nn.Parameter(torch.zeros(mc_dim))
        self.attn_res_sink = nn.Parameter(torch.zeros(()))

    def _is_block_boundary(self):
        return self.layer_idx > 0 and self.layer_idx % self.block_size == 0

    def forward(self, blocks: list[torch.Tensor], x: torch.Tensor,
                layer_residuals: list[torch.Tensor], input_ids, cos_sin):
        """
        Args:
            blocks: list of completed block snapshots [B, T, M, C]
            x: current multi-slot state [B, T, M, C]
            layer_residuals: list of [B, T, C] from previous layers in current block
            input_ids: token ids for MoE routing
            cos_sin: rotary embeddings

        Returns:
            blocks, x, layer_residuals (updated)
        """
        if self._is_block_boundary():
            blocks = blocks + [x]
            x = block_attn_res_mhc(blocks, x, self.attn_res_proj, self.attn_res_sink)
            layer_residuals = []  # reset for new block

        # mHc attention path
        residual = x
        y, post, comb = self.attn_mix(x)
        normed_y = norm(y)

        if self.kind == "DND":
            # Pass residuals for diffusion conditioning
            attn_out = self.attn(normed_y, cos_sin, residuals=layer_residuals)
        else:
            attn_out = self.attn(normed_y, cos_sin)

        x = hyper_post(attn_out, residual, post, comb)

        # Collect this layer's output as residual for future DND layers
        # Use the reduced single-slot representation for efficiency
        layer_residuals = layer_residuals + [normed_y.detach()]

        # mHc FFN path
        residual = x
        y, post, comb = self.ffn_mix(x)
        y = self.moe(norm(y), input_ids)
        x = hyper_post(y, residual, post, comb)

        return blocks, x, layer_residuals


# ─── Top-Level Model ──────────────────────────────────────────────────────────

class DeepSeekV4AttnMhcDndChat(nn.Module):
    def __init__(self, config: DeepSeekV4AttnMhcDndConfig, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([DeepSeekV4AttnMhcDndBlock(config, i) for i in range(config.n_layer)]),
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
            if isinstance(block.attn, DeepSeekHybridAttention):
                torch.nn.init.zeros_(block.attn.o_proj.proj_b.weight)
            elif isinstance(block.attn, DeltaNetDiffusionAttention):
                torch.nn.init.zeros_(block.attn.o_proj.weight)
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
        x = norm(x).unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)

        blocks = [x]
        layer_residuals = []

        for block in self.transformer.h:
            blocks, x, layer_residuals = block(blocks, x, layer_residuals, idx, cos_sin)

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
            "transformer_h": transformer_matrices,
            "mtp": mtp,
            "total": total,
        }
