"""
DeepSeek-V4-style nanochat model: mHc baseline with 2×HCA + 1×CSA layer pattern.

This variant is architecturally identical to deepseek_v4.py (mHc with all the
same components) but uses a different attention layer pattern:
- Baseline (deepseek_v4.py): "CSA,HCA" — alternating CSA and HCA layers
- This variant: "HCA,HCA,CSA" — every 3 layers: 2 HCA + 1 CSA

This tests whether increasing the ratio of Hybrid Compressed Attention layers
(which use simple causal masking on compressed KV without the Lightning Indexer)
relative to Conventional Self-Attention layers (which use the full Lightning
Indexer for selective compressed memory) improves efficiency/quality tradeoff.
"""

from dataclasses import dataclass

from nanochat.deepseek_v4 import (
    DeepSeekV4NanoConfig,
    DeepSeekV4NanoChat,
)


@dataclass
class DeepSeekV4MhcConfig(DeepSeekV4NanoConfig):
    """Same as DeepSeekV4NanoConfig but with CSA,HCA,HCA layer pattern."""
    attention_layer_pattern: str = "CSA,HCA,HCA"


# The model class is identical to DeepSeekV4NanoChat — just uses a different config.
class DeepSeekV4MhcChat(DeepSeekV4NanoChat):
    def __init__(self, config: DeepSeekV4MhcConfig, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
