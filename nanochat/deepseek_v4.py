"""
Scaled DeepSeek-V4-style nanochat model.

This is a trainable, small-model port of the DeepSeek V4 architectural ideas:
- hybrid local + compressed sparse attention
- low-rank query and output projections
- partial RoPE over a configured subspace of each head
- YaRN-style rotary scaling
- MoE feed-forward layers with shared experts, hash routing in early layers,
  auxiliary-loss-free bias balancing, and sequence-wise balance loss
- manifold-constrained Hyper-Connections with explicit A_l/B_l/C_l maps
- multi-token prediction module with embedding and hidden projections

It is not a loader for the 1.6T-parameter DeepSeek checkpoint and intentionally
does not depend on custom FP4/FP8 kernels. nanochat's existing FP8 training path
remains the right place for hardware-specific quantized training experiments.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import Linear, norm, apply_rotary_emb


@dataclass
class DeepSeekV4NanoConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 3
    n_embd: int = 768
    window_size: int = 256
    compress_ratios: str = "4,16"
    attention_layer_pattern: str = "CSA,HCA"
    csa_compress_ratio: int = 4
    hca_compress_ratio: int = 16
    rope_head_dim: int = 0
    q_lora_rank: int = 0
    o_lora_rank: int = 0
    o_groups: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 0
    n_hash_layers: int = -1
    n_hash_layers_frac: float = 0.25
    hash_seed: int = 1729
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    aux_free_balance_rate: float = 1e-3
    sequence_balance_loss_weight: float = 1e-2
    logit_softcap: float = 15.0
    swiglu_limit: float = 10.0
    hc_mult: int = 2
    hc_sinkhorn_iters: int = 8
    hc_eps: float = 1e-4
    num_nextn_predict_layers: int = 1
    mtp_loss_weight: float = 0.1
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 4.0
    original_max_position_embeddings: int = 256
    beta_fast: int = 32
    beta_slow: int = 1
    index_topk: int = 8
    index_n_head: int = 0
    index_head_dim: int = 0
    attention_dropout: float = 0.0


def _num_hash_layers(config: DeepSeekV4NanoConfig) -> int:
    if config.n_hash_layers >= 0:
        return min(config.n_hash_layers, config.n_layer)
    frac_layers = math.ceil(config.n_layer * config.n_hash_layers_frac)
    return max(0, min(config.n_layer, frac_layers))


def _default_rank(config: DeepSeekV4NanoConfig, frac: float = 0.5) -> int:
    return max(8, int(config.n_embd * frac))


def _parse_compress_ratios(config: DeepSeekV4NanoConfig) -> list[int]:
    ratios = [int(part.strip()) for part in config.compress_ratios.split(",") if part.strip()]
    return [r for r in ratios if r > 1]


def _attention_kind(config: DeepSeekV4NanoConfig, layer_idx: int) -> str:
    """Parse attention layer pattern with optional prefix layers.

    Supports two formats:
      - Simple cycle: "CSA,HCA,DN" — repeating pattern from layer 0
      - Prefix + cycle: "SWA,SWA+CSA,HCA,DN" — first N layers fixed, then cycle
        The '+' separates prefix layers (left) from the repeating pattern (right).
    """
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
    assert kind in {"CSA", "HCA", "SWA", "DN"}, f"Unknown DeepSeek attention kind: {kind}"
    return kind


def _rope_dim(config: DeepSeekV4NanoConfig) -> int:
    head_dim = config.n_embd // config.n_head
    rope_dim = config.rope_head_dim or max(2, head_dim // 2)
    rope_dim = min(rope_dim, head_dim)
    rope_dim = rope_dim - (rope_dim % 2)
    assert rope_dim > 0
    return rope_dim


def apply_partial_rotary_emb(x, cos, sin, rope_dim):
    """Apply RoPE only to the last `rope_dim` dimensions, leaving the rest unrotated."""
    if rope_dim == x.size(-1):
        return apply_rotary_emb(x, cos[..., :rope_dim // 2], sin[..., :rope_dim // 2])
    x_nope, x_rope = x[..., :-rope_dim], x[..., -rope_dim:]
    x_rope = apply_rotary_emb(x_rope, cos[..., :rope_dim // 2], sin[..., :rope_dim // 2])
    return torch.cat([x_nope, x_rope], dim=-1)


def _find_correction_dim(num_rotations, dim, base, max_seq_len):
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(min_val, max_val, dim, device):
    if min_val == max_val:
        max_val += 0.001
    x = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
    return torch.clamp(x, 0, 1)


def precompute_yarn_rotary(seq_len, head_dim, config: DeepSeekV4NanoConfig, device=None):
    device = torch.device("cpu") if device is None else device
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    if config.original_max_position_embeddings > 0 and config.rope_factor != 1.0:
        low, high = _find_correction_range(
            config.beta_fast,
            config.beta_slow,
            head_dim,
            config.rope_theta,
            config.original_max_position_embeddings,
        )
        smooth = 1 - _linear_ramp_factor(low, high, head_dim // 2, device)
        inv_freq = inv_freq / config.rope_factor * (1 - smooth) + inv_freq * smooth

    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
    return cos[None, :, None, :], sin[None, :, None, :]


def sinkhorn(logits, iters=4, eps=1e-4):
    matrix = logits.float()
    for _ in range(iters):
        matrix = matrix - torch.logsumexp(matrix, dim=-1, keepdim=True)
        matrix = matrix - torch.logsumexp(matrix, dim=-2, keepdim=True)
    return matrix.exp().to(dtype=logits.dtype) + eps


def _cache_key(device, *parts):
    return (str(device),) + tuple(parts)


class GatedCompressor(nn.Module):
    """Learned gated pooling over fixed token blocks for compressed attention memory."""

    def __init__(self, config: DeepSeekV4NanoConfig, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        kv_dim = self.n_kv_head * self.head_dim
        self.k_proj = Linear(config.n_embd, kv_dim, bias=False)
        self.v_proj = Linear(config.n_embd, kv_dim, bias=False)
        self.gate_proj = Linear(config.n_embd, kv_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(ratio, self.n_kv_head, self.head_dim))
        self._group_cache = {}

    def _group_info(self, T, device):
        key = _cache_key(device, T, self.ratio)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached
        groups = math.ceil(T / self.ratio)
        padded = groups * self.ratio
        pad = padded - T
        group_ends = torch.arange(self.ratio - 1, padded, self.ratio, device=device).clamp(max=T - 1)
        valid = None
        if pad:
            valid = torch.ones(padded, device=device, dtype=torch.bool)
            valid[T:] = False
            valid = valid.view(1, groups, self.ratio, 1, 1)
        cached = (groups, pad, group_ends, valid)
        self._group_cache[key] = cached
        return cached

    def forward(self, x):
        B, T, _ = x.shape
        groups, pad, group_ends, valid = self._group_info(T, x.device)

        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        gate = self.gate_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        if pad:
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            gate = F.pad(gate, (0, 0, 0, 0, 0, pad), value=-float("inf"))

        k = k.view(B, groups, self.ratio, self.n_kv_head, self.head_dim)
        v = v.view(B, groups, self.ratio, self.n_kv_head, self.head_dim)
        gate = gate.view(B, groups, self.ratio, self.n_kv_head, self.head_dim)
        gate = gate + self.ape.view(1, 1, self.ratio, self.n_kv_head, self.head_dim).to(gate.dtype)

        if pad:
            gate = gate.masked_fill(~valid, -float("inf"))

        weights = F.softmax(gate.float(), dim=2).to(k.dtype)
        ck = (k * weights).sum(dim=2)
        cv = (v * weights).sum(dim=2)
        return ck, cv, group_ends


class LightningIndexer(nn.Module):
    """Paper-facing compressed-memory selector: indexer queries score compressed indexer keys."""

    def __init__(self, config: DeepSeekV4NanoConfig, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.n_head = config.index_n_head or config.n_head
        self.head_dim = config.index_head_dim or max(8, (config.n_embd // config.n_head) // 2)
        self.topk = config.index_topk
        self.q_proj = Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.gate_proj = Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(ratio, self.n_head, self.head_dim))
        self._group_cache = {}
        self._causal_cache = {}

    def _group_info(self, T, device):
        key = _cache_key(device, T, self.ratio)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached
        groups = math.ceil(T / self.ratio)
        padded = groups * self.ratio
        pad = padded - T
        group_ends = torch.arange(self.ratio - 1, padded, self.ratio, device=device).clamp(max=T - 1)
        valid = None
        if pad:
            valid = torch.ones(padded, device=device, dtype=torch.bool)
            valid[T:] = False
            valid = valid.view(1, groups, self.ratio, 1, 1)
        cached = (groups, pad, group_ends, valid)
        self._group_cache[key] = cached
        return cached

    def _causal_mask(self, T, group_ends, device):
        key = _cache_key(device, T, group_ends.numel(), self.ratio)
        cached = self._causal_cache.get(key)
        if cached is not None:
            return cached
        cached = group_ends.view(1, 1, -1) <= torch.arange(T, device=device).view(1, T, 1)
        self._causal_cache[key] = cached
        return cached

    def _compress_keys(self, x):
        B, T, _ = x.shape
        groups, pad, group_ends, valid = self._group_info(T, x.device)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        gate = self.gate_proj(x).view(B, T, self.n_head, self.head_dim)
        if pad:
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            gate = F.pad(gate, (0, 0, 0, 0, 0, pad), value=-float("inf"))
        k = k.view(B, groups, self.ratio, self.n_head, self.head_dim)
        gate = gate.view(B, groups, self.ratio, self.n_head, self.head_dim)
        gate = gate + self.ape.view(1, 1, self.ratio, self.n_head, self.head_dim).to(gate.dtype)
        if pad:
            gate = gate.masked_fill(~valid, -float("inf"))
        weights = F.softmax(gate.float(), dim=2).to(k.dtype)
        keys = norm((k * weights).sum(dim=2))
        return keys, group_ends

    def forward(self, x):
        B, T, _ = x.shape
        keys, group_ends = self._compress_keys(x)
        queries = norm(self.q_proj(x).view(B, T, self.n_head, self.head_dim))
        index_scores = torch.einsum("bthd,bghd->bhtg", queries, keys) * (self.head_dim ** -0.5)
        index_scores = index_scores.mean(dim=1)
        causal = self._causal_mask(T, group_ends, x.device)
        index_scores = index_scores.masked_fill(~causal, -float("inf"))
        topk = min(self.topk, keys.size(1))
        if topk <= 0:
            return causal.expand(B, T, -1), group_ends
        selected = torch.topk(index_scores, k=topk, dim=-1).indices
        mask = torch.zeros(B, T, keys.size(1), device=x.device, dtype=torch.bool)
        mask.scatter_(2, selected, True)
        return mask & causal, group_ends


class GroupedOutputProjection(nn.Module):
    """Grouped low-rank output projection used after attention heads are concatenated."""

    def __init__(self, config: DeepSeekV4NanoConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.groups = max(1, config.o_groups)
        assert config.n_embd % self.groups == 0
        group_dim = config.n_embd // self.groups
        rank = config.o_lora_rank or _default_rank(config)
        self.rank_per_group = max(1, math.ceil(rank / self.groups))
        self.weight_a = nn.Parameter(torch.empty(self.groups, self.rank_per_group, group_dim))
        self.proj_b = Linear(self.groups * self.rank_per_group, config.n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self, bound):
        torch.nn.init.uniform_(self.weight_a, -bound, bound)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.groups, C // self.groups)
        weight = self.weight_a.to(dtype=x.dtype)
        x = torch.einsum("btgd,grd->btgr", x, weight).flatten(2)
        return self.proj_b(norm(x))


class DeepSeekHybridAttention(nn.Module):
    def __init__(self, config: DeepSeekV4NanoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kind = _attention_kind(config, layer_idx)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rope_dim = _rope_dim(config)
        assert config.n_embd % config.n_head == 0
        assert self.head_dim % 2 == 0
        assert self.n_head % self.n_kv_head == 0

        q_rank = config.q_lora_rank or _default_rank(config)
        kv_dim = self.n_kv_head * self.head_dim

        self.q_a = Linear(config.n_embd, q_rank, bias=False)
        self.q_b = Linear(q_rank, self.n_head * self.head_dim, bias=False)
        self.swa_k_proj = Linear(config.n_embd, kv_dim, bias=False)
        self.swa_v_proj = Linear(config.n_embd, kv_dim, bias=False)
        self.o_proj = GroupedOutputProjection(config)
        self.attn_sink = nn.Parameter(torch.zeros(config.n_head))
        if self.kind == "CSA":
            self.compress_ratio = config.csa_compress_ratio
        elif self.kind == "HCA":
            self.compress_ratio = config.hca_compress_ratio
        else:
            self.compress_ratio = 0
        self.compressor = GatedCompressor(config, self.compress_ratio) if self.compress_ratio else None
        self.indexer = LightningIndexer(config, self.compress_ratio) if self.kind == "CSA" and self.compress_ratio else None
        self.dropout_p = config.attention_dropout
        self._mask_cache = {}

    def _repeat_kv(self, x):
        if self.n_kv_head == self.n_head:
            return x
        repeat = self.n_head // self.n_kv_head
        return x.repeat_interleave(repeat, dim=2)

    def _local_mask(self, T, device):
        key = _cache_key(device, T, self.config.window_size)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        pos = torch.arange(T, device=device)
        qpos = pos.view(T, 1)
        kpos = pos.view(1, T)
        causal = kpos <= qpos
        local = kpos >= (qpos - self.config.window_size + 1)
        cached = causal & local
        self._mask_cache[key] = cached
        return cached

    def _compressed_causal_mask(self, T, group_ends, device):
        key = _cache_key(device, T, group_ends.numel(), self.compress_ratio)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        cached = group_ends.view(1, 1, -1) <= torch.arange(T, device=device).view(1, T, 1)
        self._mask_cache[key] = cached
        return cached

    def forward(self, x, cos_sin):
        B, T, _ = x.shape
        cos, sin = cos_sin

        q = self.q_b(norm(self.q_a(x))).view(B, T, self.n_head, self.head_dim)
        k = self.swa_k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.swa_v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_partial_rotary_emb(q, cos, sin, self.rope_dim)
        k = apply_partial_rotary_emb(k, cos, sin, self.rope_dim)
        q, k = norm(q), norm(k)

        keys = [self._repeat_kv(k)]
        values = [self._repeat_kv(v)]
        masks = [self._local_mask(T, x.device).view(1, T, T).expand(B, -1, -1)]

        if self.compressor is not None:
            ck, cv, group_ends = self.compressor(x)
            ck = apply_partial_rotary_emb(ck, cos[:, group_ends], sin[:, group_ends], self.rope_dim)
            ck = norm(ck)
            keys.append(self._repeat_kv(ck))
            values.append(self._repeat_kv(cv))
            if self.indexer is not None:
                compressed_mask, _ = self.indexer(x)
            else:
                compressed_mask = self._compressed_causal_mask(T, group_ends, x.device).expand(B, -1, -1)
            masks.append(compressed_mask)

        zero_k = q.new_zeros(B, 1, self.n_head, self.head_dim)
        zero_v = q.new_zeros(B, 1, self.n_head, self.head_dim)
        keys.append(zero_k)
        values.append(zero_v)
        masks.append(torch.ones(B, T, 1, device=x.device, dtype=torch.bool))

        k_all = torch.cat(keys, dim=1).transpose(1, 2)
        v_all = torch.cat(values, dim=1).transpose(1, 2)
        q = q.transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k_all.transpose(-2, -1)) * scale
        sink_index = k_all.size(-2) - 1
        scores[..., sink_index] = self.attn_sink.view(1, self.n_head, 1)
        mask = torch.cat(masks, dim=-1)
        scores = scores.masked_fill(~mask.view(B, 1, T, -1), -float("inf"))
        att = F.softmax(scores.float(), dim=-1).to(q.dtype)
        if self.training and self.dropout_p > 0:
            att = F.dropout(att, p=self.dropout_p)
        y = torch.matmul(att, v_all).transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.o_proj(y)


class DeepSeekExpert(nn.Module):
    def __init__(self, config: DeepSeekV4NanoConfig):
        super().__init__()
        inter = config.moe_intermediate_size or 4 * config.n_embd
        self.w1 = Linear(config.n_embd, inter, bias=False)
        self.w2 = Linear(inter, config.n_embd, bias=False)
        self.w3 = Linear(config.n_embd, inter, bias=False)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x):
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.w2((F.silu(gate) * up).to(dtype=x.dtype))


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekV4NanoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hash_routing = layer_idx < _num_hash_layers(config)
        self.topk = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        assert self.topk <= self.n_routed_experts
        self.gate = Linear(config.n_embd, config.n_routed_experts, bias=False)
        self.register_buffer("gate_bias", torch.zeros(config.n_routed_experts), persistent=True)
        self.register_buffer("last_expert_counts", torch.zeros(config.n_routed_experts), persistent=False)
        self.last_sequence_balance_loss = None
        self.experts = nn.ModuleList([DeepSeekExpert(config) for _ in range(config.n_routed_experts)])
        assert config.n_shared_experts in {0, 1}, "This scaled port supports zero or one shared expert."
        self.shared_expert = DeepSeekExpert(config) if config.n_shared_experts else None

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.hash_seed + layer_idx)
        hashed = torch.stack([
            torch.randperm(config.n_routed_experts, generator=generator)[:self.topk]
            for _ in range(config.vocab_size)
        ])
        self.register_buffer("tid2eid", hashed.long(), persistent=False)

    def _expert_weights(self, attr, dtype):
        return torch.stack([getattr(expert, attr).weight for expert in self.experts], dim=0).to(dtype=dtype)

    def _loop_dispatch(self, flat, flat_experts, flat_tokens, flat_weights):
        routed = flat.new_zeros(flat.shape, dtype=torch.float32)
        boundaries = torch.searchsorted(
            flat_experts,
            torch.arange(self.n_routed_experts + 1, device=flat.device),
        )
        for expert_id, expert in enumerate(self.experts):
            start = boundaries[expert_id].item()
            end = boundaries[expert_id + 1].item()
            if start == end:
                continue
            token_idx = flat_tokens[start:end]
            expert_out = expert(flat[token_idx]).float()
            routed.index_add_(0, token_idx, expert_out * flat_weights[start:end, None].float())
        return routed

    def _batched_dispatch(self, flat, flat_experts, flat_tokens, flat_weights):
        routed = flat.new_zeros(flat.shape, dtype=torch.float32)
        counts = torch.bincount(flat_experts, minlength=self.n_routed_experts)
        max_count = int(counts.max().detach().cpu())
        if max_count == 0:
            return routed

        starts = torch.cumsum(counts, dim=0) - counts
        ranks = torch.arange(flat_experts.numel(), device=flat.device) - starts.index_select(0, flat_experts)
        dispatched = flat.new_zeros(self.n_routed_experts, max_count, flat.size(-1))
        dispatched[flat_experts, ranks] = flat.index_select(0, flat_tokens)

        w1 = self._expert_weights("w1", dispatched.dtype)
        w3 = self._expert_weights("w3", dispatched.dtype)
        w2 = self._expert_weights("w2", dispatched.dtype)
        gate = torch.bmm(dispatched, w1.transpose(1, 2)).float()
        up = torch.bmm(dispatched, w3.transpose(1, 2)).float()
        if self.config.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.config.swiglu_limit)
            up = torch.clamp(up, min=-self.config.swiglu_limit, max=self.config.swiglu_limit)
        expert_out = torch.bmm((F.silu(gate) * up).to(dtype=dispatched.dtype), w2.transpose(1, 2)).float()
        selected = expert_out[flat_experts, ranks] * flat_weights[:, None].float()
        routed.index_add_(0, flat_tokens, selected)
        return routed

    def _scores(self, x):
        scores = self.gate(x.float())
        if self.config.scoring_func == "softmax":
            return scores.softmax(dim=-1)
        if self.config.scoring_func == "sigmoid":
            return scores.sigmoid()
        return F.softplus(scores).sqrt()

    def _sequence_balance_loss(self, scores, indices, B, T):
        probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        probs = probs.view(B, T, self.n_routed_experts).mean(dim=1)
        one_hot = F.one_hot(indices, num_classes=self.n_routed_experts).float()
        density = one_hot.view(B, T, self.topk, self.n_routed_experts).sum(dim=(1, 2)) / max(1, T * self.topk)
        return self.n_routed_experts * (density.detach() * probs).sum(dim=-1).mean()

    @torch.no_grad()
    def update_aux_free_balance(self):
        if self.hash_routing or self.config.aux_free_balance_rate <= 0:
            return
        counts = self.last_expert_counts.float()
        if counts.sum() == 0:
            return
        mean = counts.mean()
        direction = torch.sign(counts - mean)
        self.gate_bias.sub_(self.config.aux_free_balance_rate * direction.to(self.gate_bias.device))
        self.gate_bias.sub_(self.gate_bias.mean())

    def forward(self, x, input_ids):
        shape = x.shape
        B, T, _ = shape
        flat = x.view(-1, shape[-1])
        original_scores = self._scores(flat)
        if self.hash_routing:
            indices = self.tid2eid[input_ids.reshape(-1)]
        else:
            routed_scores = original_scores + self.gate_bias
            indices = routed_scores.topk(self.topk, dim=-1).indices

        weights = original_scores.gather(1, indices)
        if self.config.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights * self.config.routed_scaling_factor
        self.last_sequence_balance_loss = self._sequence_balance_loss(original_scores, indices, B, T)
        with torch.no_grad():
            counts = torch.bincount(indices.reshape(-1), minlength=self.n_routed_experts).to(self.last_expert_counts)
            self.last_expert_counts.copy_(counts)

        if self.shared_expert is None:
            y = flat.new_zeros(flat.shape, dtype=torch.float32)
        else:
            y = self.shared_expert(flat).float()
        routed = flat.new_zeros(flat.shape, dtype=torch.float32)
        flat_experts = indices.reshape(-1)
        flat_tokens = torch.arange(flat.size(0), device=flat.device).repeat_interleave(self.topk)
        flat_weights = weights.reshape(-1)
        order = torch.argsort(flat_experts)
        flat_experts = flat_experts[order]
        flat_tokens = flat_tokens[order]
        flat_weights = flat_weights[order]
        if self.training:
            routed = self._batched_dispatch(flat, flat_experts, flat_tokens, flat_weights)
        else:
            routed = self._loop_dispatch(flat, flat_experts, flat_tokens, flat_weights)
        return (y + routed).to(dtype=x.dtype).view(shape)


class HyperConnectionMixer(nn.Module):
    def __init__(self, config: DeepSeekV4NanoConfig):
        super().__init__()
        self.config = config
        in_dim = config.hc_mult * config.n_embd
        self.A_l = Linear(in_dim, config.hc_mult, bias=False)
        self.B_l = Linear(in_dim, config.hc_mult * config.hc_mult, bias=False)
        self.C_l = Linear(in_dim, config.hc_mult, bias=False)
        self.A_base = nn.Parameter(torch.zeros(config.hc_mult))
        self.B_base = nn.Parameter(torch.zeros(config.hc_mult * config.hc_mult))
        self.C_base = nn.Parameter(torch.zeros(config.hc_mult))

    def forward(self, x):
        B, T, M, C = x.shape
        state = norm(x.flatten(2))
        input_mapping = F.softmax((self.A_l(state).float() + self.A_base), dim=-1).to(dtype=x.dtype)
        residual_logits = self.B_l(state).float() + self.B_base
        residual_mapping = sinkhorn(
            residual_logits.view(B, T, M, M),
            iters=self.config.hc_sinkhorn_iters,
            eps=self.config.hc_eps,
        ).to(dtype=x.dtype)
        output_mapping = (torch.sigmoid(self.C_l(state).float() + self.C_base) + self.config.hc_eps).to(dtype=x.dtype)
        reduced = torch.sum(input_mapping.unsqueeze(-1) * x, dim=2)
        return reduced, output_mapping, residual_mapping


def hyper_post(y, residual, post, comb):
    mixed_residual = torch.einsum("btij,btjd->btid", comb, residual)
    return post.unsqueeze(-1) * y.unsqueeze(2) + mixed_residual


class DeepSeekV4Block(nn.Module):
    def __init__(self, config: DeepSeekV4NanoConfig, layer_idx: int):
        super().__init__()
        self.attn_mix = HyperConnectionMixer(config)
        self.ffn_mix = HyperConnectionMixer(config)
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


class MTPPredictionHead(nn.Module):
    """Multi-token prediction head with the paper's embedding/hidden projection shape."""

    def __init__(self, config: DeepSeekV4NanoConfig, padded_vocab_size: int):
        super().__init__()
        self.e_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.h_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.head = Linear(config.n_embd, padded_vocab_size, bias=False)

    def forward(self, hidden, token_embed):
        x = self.e_proj(norm(token_embed)) + self.h_proj(norm(hidden))
        return self.head(norm(x))


class DeepSeekV4NanoChat(nn.Module):
    def __init__(self, config: DeepSeekV4NanoConfig, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([DeepSeekV4Block(config, i) for i in range(config.n_layer)]),
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
            torch.nn.init.zeros_(block.attn.o_proj.proj_b.weight)

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
        dense_fraction = 0.35  # non-expert blocks, embeddings, attention, MTP, and mixing paths.
        return int(6 * nparams * (dense_fraction + active_expert_frac))

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        mtp = sum(p.numel() for p in self.mtp_heads.parameters())
        head = sum(p.numel() for p in self.head_mix.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "wte": wte,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "mtp": mtp,
            "head": head,
            "total": total,
        }
