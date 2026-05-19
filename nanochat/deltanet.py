"""
Gated DeltaNet linear attention block.

A self-contained implementation of the Gated Delta Rule linear attention
mechanism (Yang et al., "Gated Delta Networks", NeurIPS 2024; Hatamizadeh et al.,
"GatedDeltaNet", NVIDIA 2024) without external CUDA kernel dependencies.

The delta rule maintains a key-value state matrix S ∈ R^{d_k × d_v} per head:
    S_t = gate_t * S_{t-1} + beta_t * k_t * (v_t - S_{t-1}^T k_t)^T
    o_t = S_t^T q_t

Where:
- gate_t: per-head decay factor (learned log-sigmoid gate)
- beta_t: per-head step size (learned sigmoid projection)
- k_t, q_t are L2-normalized
- Short 1D convolutions are applied to Q, K, V before normalization
- Output is gated with a SiLU activation

This implementation uses a naive O(T·H·d²) recurrence loop suitable for
short sequences (≤512 tokens) during training. For long sequences, a
chunked or parallel-scan implementation would be needed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Linear, norm


class ShortConv1d(nn.Module):
    """Causal 1D depthwise convolution with SiLU activation."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size - 1,
            groups=dim,
            bias=False,
        )

    def forward(self, x):
        # x: (B, T, D)
        dtype = x.dtype
        x = x.transpose(1, 2).float()  # (B, D, T) — conv1d in float32
        x = self.conv(x)[..., :x.size(-1)]  # causal: trim future
        x = F.silu(x)
        return x.transpose(1, 2).to(dtype)  # (B, T, D)


def l2_norm(x, dim=-1, eps=1e-6):
    """L2 normalize along the last dimension."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def delta_rule_recurrence(q, k, v, beta, gate):
    """
    Naive delta rule recurrence for training on short sequences.

    Args:
        q: (B, H, T, d_k) — L2-normalized queries
        k: (B, H, T, d_k) — L2-normalized keys
        v: (B, H, T, d_v) — values
        beta: (B, H, T) — step sizes in [0, 1]
        gate: (B, H, T) — log decay gates (negative values)

    Returns:
        o: (B, H, T, d_v) — output
    """
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device
    dtype = q.dtype

    # State matrix S: (B, H, d_k, d_v)
    S = torch.zeros(B, H, d_k, d_v, device=device, dtype=torch.float32)
    outputs = []

    # Exponentiate gate for multiplicative decay
    alpha = gate.exp()  # (B, H, T) — values in (0, 1)

    for t in range(T):
        k_t = k[:, :, t, :]  # (B, H, d_k)
        v_t = v[:, :, t, :]  # (B, H, d_v)
        q_t = q[:, :, t, :]  # (B, H, d_k)
        beta_t = beta[:, :, t].unsqueeze(-1)  # (B, H, 1)
        alpha_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        # Retrieved value from current state: S^T k_t
        retrieved = torch.einsum("bhkv,bhk->bhv", S, k_t.float())  # (B, H, d_v)

        # Error signal
        error = v_t.float() - retrieved  # (B, H, d_v)

        # Delta update: S = gate * S + beta * k_t * error^T
        S = alpha_t * S + beta_t.unsqueeze(-1) * torch.einsum("bhk,bhv->bhkv", k_t.float(), error)

        # Output: o_t = S^T q_t
        o_t = torch.einsum("bhkv,bhk->bhv", S, q_t.float())  # (B, H, d_v)
        outputs.append(o_t)

    return torch.stack(outputs, dim=2).to(dtype)  # (B, H, T, d_v)


class DeltaNetAttention(nn.Module):
    """
    Gated DeltaNet attention block matching DeepSeekHybridAttention interface.

    forward(x, cos_sin) -> output tensor of shape (B, T, n_embd)
    """

    def __init__(self, n_embd: int, n_head: int, conv_size: int = 4):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0

        # Q, K, V projections
        self.q_proj = Linear(n_embd, n_embd, bias=False)
        self.k_proj = Linear(n_embd, n_embd, bias=False)
        self.v_proj = Linear(n_embd, n_embd, bias=False)

        # Short causal convolutions
        self.q_conv = ShortConv1d(n_embd, conv_size)
        self.k_conv = ShortConv1d(n_embd, conv_size)
        self.v_conv = ShortConv1d(n_embd, conv_size)

        # Beta (step size) projection: per-head scalar
        self.beta_proj = Linear(n_embd, n_head, bias=True)

        # Gate (decay) projection: per-head scalar
        self.gate_proj = Linear(n_embd, n_head, bias=True)
        self.gate_logit_normalizer = 16

        # Output gating
        self.g_proj = Linear(n_embd, n_embd, bias=False)
        self.g_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

        # Output projection
        self.o_proj = Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        """
        Args:
            x: (B, T, n_embd) — input hidden states (pre-normed)
            cos_sin: tuple of rotary embeddings (unused — DeltaNet uses L2 norm instead)

        Returns:
            output: (B, T, n_embd)
        """
        B, T, C = x.shape

        # Project and apply short convolutions
        q = self.q_conv(self.q_proj(x))  # (B, T, C)
        k = self.k_conv(self.k_proj(x))  # (B, T, C)
        v = self.v_conv(self.v_proj(x))  # (B, T, C)

        # Reshape to heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)

        # L2 normalize Q and K
        q = l2_norm(q)
        k = l2_norm(k)

        # Beta: step size per head per timestep
        beta = self.beta_proj(x).sigmoid().transpose(1, 2)  # (B, H, T)

        # Gate: decay per head per timestep (log-sigmoid for stability)
        gate = F.logsigmoid(self.gate_proj(x)) / self.gate_logit_normalizer  # (B, T, H)
        gate = gate.transpose(1, 2)  # (B, H, T)

        # Delta rule recurrence
        o = delta_rule_recurrence(q, k, v, beta, gate)  # (B, H, T, d_v)

        # Output gating: norm + SiLU gate
        o = o.transpose(1, 2)  # (B, T, H, d)
        o = self.g_norm(o.float()).to(x.dtype)
        o = o.reshape(B, T, C)
        g = F.silu(self.g_proj(x))
        o = o * g

        return self.o_proj(o)
