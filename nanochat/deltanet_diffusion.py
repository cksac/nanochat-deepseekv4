"""
DND: DeltaNet + Diffusion future-state prediction.

Combines Gated DeltaNet linear attention with a block diffusion mechanism
inspired by DFlash (Chen et al., 2026). The DND layer:

1. Runs GatedDeltaNet recurrence to produce the current-state output
2. Uses a lightweight denoising network, conditioned on residuals from
   previous layers within the attention block, to predict N future states
3. Mixes the DeltaNet output with predicted future states via learned weights

The diffusion component operates in a single denoising step (flow matching)
for training efficiency. The conditioning signal is the concatenation of
residual hidden states from earlier layers in the same block, projected
down to the model dimension.

Architecture:
    x → DeltaNet → o_current
    residuals → fc_cond → cond
    noise → denoise_net(noise, cond) → future_states[1..N]
    output = mix_weight[0] * o_current + sum(mix_weight[i] * future_states[i])
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Linear, norm
from nanochat.deltanet import (
    ShortConv1d,
    l2_norm,
    delta_rule_recurrence,
)


class FutureStateDenoiser(nn.Module):
    """
    Single-step denoiser that predicts N future states from noise,
    conditioned on context from previous layer residuals.

    Uses a simple MLP with conditioning via FiLM (Feature-wise Linear Modulation):
        h = act(W1 @ noise + gamma * h + beta)
        future = W2 @ h
    """

    def __init__(self, n_embd: int, n_future: int = 2, cond_dim: int = 0):
        super().__init__()
        self.n_embd = n_embd
        self.n_future = n_future
        # Conditioning projection: map concatenated residuals to n_embd
        self.cond_proj = Linear(cond_dim if cond_dim > 0 else n_embd, n_embd, bias=False)
        # FiLM modulation: produce scale and shift from condition
        self.film = Linear(n_embd, 2 * n_embd, bias=True)
        # Denoising network
        self.noise_proj = Linear(n_embd, n_embd, bias=False)
        self.out_proj = Linear(n_embd, n_future * n_embd, bias=False)

    def forward(self, noise, condition):
        """
        Args:
            noise: (B, T, n_embd) — noise input (current hidden + gaussian)
            condition: (B, T, cond_dim) — conditioning from previous residuals

        Returns:
            futures: (B, T, n_future, n_embd) — predicted future states
        """
        B, T, C = noise.shape
        # Condition projection
        cond = self.cond_proj(condition)  # (B, T, n_embd)
        # FiLM modulation
        film_params = self.film(cond)  # (B, T, 2*n_embd)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, T, n_embd)
        # Denoise
        h = self.noise_proj(noise)
        h = gamma * h + beta
        h = F.silu(h)
        # Predict future states
        out = self.out_proj(h)  # (B, T, n_future * n_embd)
        return out.view(B, T, self.n_future, C)


class DeltaNetDiffusionAttention(nn.Module):
    """
    DND: DeltaNet + Diffusion future-state prediction.

    Combines linear attention with diffusion-based future state forecasting.
    The layer collects residuals from previous layers in the block as
    conditioning signal for the denoiser.

    forward(x, cos_sin, residuals=None) -> output tensor of shape (B, T, n_embd)

    Where residuals is a list of tensors from previous layers in the block.
    """

    def __init__(self, n_embd: int, n_head: int, n_future: int = 2,
                 max_cond_layers: int = 2, conv_size: int = 4):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_future = n_future
        self.max_cond_layers = max_cond_layers
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0

        # === DeltaNet core ===
        self.q_proj = Linear(n_embd, n_embd, bias=False)
        self.k_proj = Linear(n_embd, n_embd, bias=False)
        self.v_proj = Linear(n_embd, n_embd, bias=False)
        self.q_conv = ShortConv1d(n_embd, conv_size)
        self.k_conv = ShortConv1d(n_embd, conv_size)
        self.v_conv = ShortConv1d(n_embd, conv_size)
        self.beta_proj = Linear(n_embd, n_head, bias=True)
        self.gate_proj_dn = Linear(n_embd, n_head, bias=True)
        self.gate_logit_normalizer = 16
        self.g_proj = Linear(n_embd, n_embd, bias=False)
        self.g_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

        # === Diffusion future-state predictor ===
        cond_dim = max_cond_layers * n_embd
        self.denoiser = FutureStateDenoiser(n_embd, n_future=n_future, cond_dim=cond_dim)
        # Noise level embedding (single step — just a learned scale)
        self.noise_scale = nn.Parameter(torch.tensor(0.1))

        # === Mixing weights: softmax over [current, future_1, ..., future_N] ===
        self.mix_logits = nn.Parameter(torch.zeros(1 + n_future))

        # === Output projection ===
        self.o_proj = Linear(n_embd, n_embd, bias=False)

    def _deltanet_forward(self, x):
        """Run the DeltaNet recurrence and return gated output."""
        B, T, C = x.shape

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = l2_norm(q)
        k = l2_norm(k)

        beta = self.beta_proj(x).sigmoid().transpose(1, 2)
        gate = F.logsigmoid(self.gate_proj_dn(x)) / self.gate_logit_normalizer
        gate = gate.transpose(1, 2)

        o = delta_rule_recurrence(q, k, v, beta, gate)

        o = o.transpose(1, 2)  # (B, T, H, d)
        o = self.g_norm(o.float()).to(x.dtype)
        o = o.reshape(B, T, C)
        g = F.silu(self.g_proj(x))
        return o * g  # (B, T, C)

    def _diffusion_forward(self, x, residuals):
        """
        Predict future states using single-step denoising conditioned on residuals.

        Args:
            x: (B, T, C) — current hidden state (used as noise base)
            residuals: list of (B, T, C) tensors from previous layers

        Returns:
            futures: (B, T, n_future, C)
        """
        B, T, C = x.shape

        # Build conditioning from residuals
        if residuals and len(residuals) > 0:
            # Take the last max_cond_layers residuals
            cond_layers = residuals[-self.max_cond_layers:]
            # Pad if fewer residuals available
            while len(cond_layers) < self.max_cond_layers:
                cond_layers = [torch.zeros_like(x)] + cond_layers
            condition = torch.cat(cond_layers, dim=-1)  # (B, T, max_cond_layers * C)
        else:
            condition = torch.zeros(B, T, self.max_cond_layers * C,
                                    device=x.device, dtype=x.dtype)

        # Create noisy input: current state + scaled noise
        if self.training:
            noise_level = self.noise_scale.abs()
            noise = x + noise_level * torch.randn_like(x)
        else:
            noise = x  # deterministic at eval

        # Single-step denoise to predict future states
        futures = self.denoiser(noise, condition)  # (B, T, n_future, C)
        return futures

    def forward(self, x, cos_sin, residuals=None):
        """
        Args:
            x: (B, T, n_embd) — input hidden states (pre-normed)
            cos_sin: rotary embeddings (unused by DeltaNet)
            residuals: list of (B, T, C) from previous layers in block

        Returns:
            output: (B, T, n_embd)
        """
        B, T, C = x.shape

        # 1. DeltaNet current-state output
        o_current = self._deltanet_forward(x)  # (B, T, C)

        # 2. Diffusion future-state predictions
        futures = self._diffusion_forward(x, residuals)  # (B, T, n_future, C)

        # 3. Mix current + future states
        mix_weights = F.softmax(self.mix_logits.float(), dim=0).to(x.dtype)  # (1 + n_future,)
        # Stack: [current, future_1, ..., future_N]
        all_states = torch.cat([
            o_current.unsqueeze(2),  # (B, T, 1, C)
            futures,                  # (B, T, n_future, C)
        ], dim=2)  # (B, T, 1+n_future, C)
        # Weighted sum
        output = torch.einsum("btsc,s->btc", all_states, mix_weights)

        return self.o_proj(output)
