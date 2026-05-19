"""
DeepSeek-V4-style nanochat model with DeltaNet linear attention layers.

This variant replaces some softmax attention layers with Gated DeltaNet linear
attention, creating a hybrid architecture:
- CSA: Conventional Self-Attention with Lightning Indexer for compressed memory
- HCA: Hybrid Compressed Attention with simple causal masking
- DN: Gated DeltaNet linear attention (O(T·d²) recurrence, no softmax)

The layer pattern "CSA,HCA,DN" alternates between all three attention types,
testing whether DeltaNet's linear-time recurrence can complement the softmax
attention layers for improved efficiency.

Based on deepseek_v4.py (mHc baseline) with DeltaNet attention injected.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import Linear, norm, apply_rotary_emb
from nanochat.deltanet import DeltaNetAttention

# Import shared components from deepseek_v4
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
    _rope_dim,
    precompute_yarn_rotary,
)


def _attention_kind_dn(config, layer_idx: int) -> str:
    """Extended attention kind parser that also accepts 'DN' for DeltaNet."""
    pattern = [part.strip().upper() for part in config.attention_layer_pattern.split(",") if part.strip()]
    if not pattern:
        return "CSA"
    kind = pattern[layer_idx % len(pattern)]
    assert kind in {"CSA", "HCA", "SWA", "DN"}, f"Unknown attention kind: {kind}"
    return kind


@dataclass
class DeepSeekV4DnConfig(DeepSeekV4NanoConfig):
    """DeepSeekV4 config with DeltaNet layers in the attention pattern."""
    attention_layer_pattern: str = "CSA,HCA,DN"
    dn_conv_size: int = 4  # DeltaNet short conv kernel size


class DeepSeekV4DnBlock(nn.Module):
    """Block that routes to either softmax attention or DeltaNet based on layer pattern."""

    def __init__(self, config: DeepSeekV4DnConfig, layer_idx: int):
        super().__init__()
        self.attn_mix = HyperConnectionMixer(config)
        self.ffn_mix = HyperConnectionMixer(config)

        kind = _attention_kind_dn(config, layer_idx)
        if kind == "DN":
            self.attn = DeltaNetAttention(
                n_embd=config.n_embd,
                n_head=config.n_head,
                conv_size=config.dn_conv_size,
            )
        else:
            self.attn = DeepSeekHybridAttention(config, layer_idx)

        self.moe = DeepSeekMoE(config, layer_idx)

    def forward(self, x, input_ids, cos_sin):
        residual = x
        y, post, comb = self.attn_mix(x)
        y = self.attn(norm(y), cos_sin)
        x = hyper_post(y, residual, post, comb)

        residual = x
        y, post, comb = self.ffn_mix(x)
        y = self.moe(norm(y), input_ids)
        x = hyper_post(y, residual, post, comb)
        return x


class DeepSeekV4DnChat(nn.Module):
    def __init__(self, config: DeepSeekV4DnConfig, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([DeepSeekV4DnBlock(config, i) for i in range(config.n_layer)]),
        })
        self.head_mix = HyperConnectionMixer(config)
        self.lm_norm = nn.Identity()
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
            elif isinstance(block.attn, DeltaNetAttention):
                torch.nn.init.zeros_(block.attn.o_proj.weight)

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
        for block in self.transformer.h:
            x = block(x, idx, cos_sin)
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
