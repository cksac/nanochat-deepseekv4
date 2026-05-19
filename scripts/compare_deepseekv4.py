"""
Compare native nanochat GPT against the scaled DeepSeek-V4 nanochat variant.

Examples:
    python -m scripts.compare_deepseekv4 --steps 40 --device mps
    python -m scripts.compare_deepseekv4 --datasets tiny_shakespeare,wikitext2 --steps 80
"""

import argparse
import csv
import json
import math
import os
import socket
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type
from nanochat.gpt import GPT, GPTConfig
from nanochat.deepseek_v4 import DeepSeekV4NanoChat, DeepSeekV4NanoConfig
from nanochat.deepseek_v4_attnres import DeepSeekV4AttnResChat, DeepSeekV4AttnResConfig
from nanochat.deepseek_v4_attnmhc import DeepSeekV4AttnMhcChat, DeepSeekV4AttnMhcConfig
from nanochat.deepseek_v4_mhc import DeepSeekV4MhcChat, DeepSeekV4MhcConfig
from nanochat.deepseek_v4_dn import DeepSeekV4DnChat, DeepSeekV4DnConfig
from nanochat.deepseek_v4_attnmhc_dnd import DeepSeekV4AttnMhcDndChat, DeepSeekV4AttnMhcDndConfig


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
WIKITEXT_PARQUET_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"
VOCAB_SIZE = 257


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    seq_len: int
    n_layer: int
    n_embd: int
    n_head: int
    batch_size: int
    steps: int
    eval_every: int
    eval_batch_size: int
    train_eval_batches: int
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    deepseek_n_routed_experts: int = 4
    deepseek_num_experts_per_tok: int = 2
    deepseek_moe_intermediate_size: int = 0
    deepseek_index_topk: int = 4


CONFIG_PRESETS = {
    "small": BenchmarkConfig(
        name="small",
        seq_len=64,
        n_layer=2,
        n_embd=128,
        n_head=4,
        batch_size=8,
        steps=100,
        eval_every=10,
        eval_batch_size=256,
        train_eval_batches=32,
        deepseek_n_routed_experts=4,
        deepseek_moe_intermediate_size=128,
        deepseek_index_topk=4,
    ),
    "medium": BenchmarkConfig(
        name="medium",
        seq_len=256,
        n_layer=6,
        n_embd=256,
        n_head=8,
        batch_size=8,
        steps=1000,
        eval_every=100,
        eval_batch_size=128,
        train_eval_batches=16,
        deepseek_n_routed_experts=8,
        deepseek_moe_intermediate_size=256,
        deepseek_index_topk=8,
    ),
}


def fetch_url(url, timeout=120, retries=5):
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 20))


def read_or_fetch_text(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(fetch_url(url))
    return path.read_text(encoding="utf-8", errors="replace")


def load_tiny_shakespeare(cache_dir):
    text = read_or_fetch_text(TINY_SHAKESPEARE_URL, cache_dir / "tiny_shakespeare.txt")
    n = len(text)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)
    return {
        "train": text[:train_end],
        "validation": text[train_end:val_end],
        "test": text[val_end:],
    }


def fetch_wikitext_split(split, cache_dir, max_chars=None):
    suffix = "full" if max_chars is None else str(max_chars)
    path = cache_dir / f"wikitext2_{split}_{suffix}.txt"
    tmp_path = cache_dir / f"wikitext2_{split}_{suffix}.partial.json"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")

    if tmp_path.exists():
        state = json.loads(tmp_path.read_text())
        texts = state["texts"]
        chars = state["chars"]
        offset = state["offset"]
    else:
        texts = []
        chars = 0
        offset = 0
    page_len = 100
    while True:
        params = urllib.parse.urlencode({
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": split,
            "offset": offset,
            "length": page_len,
        })
        payload = json.loads(fetch_url(f"{HF_ROWS_URL}?{params}").decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            text = row["row"].get("text", "")
            if text.strip():
                texts.append(text)
                chars += len(text) + 1
                if max_chars is not None and chars >= max_chars:
                    break
        if max_chars is not None and chars >= max_chars:
            break
        if len(rows) < page_len:
            break
        offset += page_len
        tmp_path.write_text(json.dumps({"texts": texts, "chars": chars, "offset": offset}))

    text = "\n".join(texts)
    if max_chars is not None:
        text = text[:max_chars]
    path.write_text(text, encoding="utf-8")
    if tmp_path.exists():
        tmp_path.unlink()
    return text


def fetch_wikitext_split_parquet(split, cache_dir):
    text_path = cache_dir / f"wikitext2_{split}_full.txt"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="replace")

    import pyarrow.parquet as pq

    parquet_path = cache_dir / f"wikitext2_{split}.parquet"
    if not parquet_path.exists():
        parquet_path.write_bytes(fetch_url(f"{WIKITEXT_PARQUET_URL}/{split}-00000-of-00001.parquet", timeout=300))
    table = pq.read_table(parquet_path, columns=["text"])
    texts = [text for text in table.column("text").to_pylist() if text.strip()]
    text = "\n".join(texts)
    text_path.write_text(text, encoding="utf-8")
    return text


def load_wikitext2(cache_dir, train_chars, eval_chars, full_data=False):
    if full_data:
        return {
            "train": fetch_wikitext_split_parquet("train", cache_dir),
            "validation": fetch_wikitext_split_parquet("validation", cache_dir),
            "test": fetch_wikitext_split_parquet("test", cache_dir),
        }
    return {
        "train": fetch_wikitext_split("train", cache_dir, train_chars),
        "validation": fetch_wikitext_split("validation", cache_dir, eval_chars),
        "test": fetch_wikitext_split("test", cache_dir, eval_chars),
    }


def encode_bytes(text):
    return torch.tensor(list(text.encode("utf-8", errors="replace")), dtype=torch.long)


def get_batch(tokens, batch_size, seq_len, device, rng, positions=None):
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Dataset split is too short for seq_len={seq_len}: {len(tokens)} byte tokens")
    source_device = tokens.device
    if positions is None or positions.device != source_device:
        positions = torch.arange(seq_len + 1, device=source_device)
    starts = torch.randint(0, max_start, (batch_size,), generator=rng, device="cpu")
    if source_device.type != "cpu":
        starts = starts.to(source_device)
    offsets = starts[:, None] + positions[None, :]
    batch = tokens.index_select(0, offsets.reshape(-1)).view(batch_size, seq_len + 1)
    x, y = batch[:, :-1], batch[:, 1:]
    if x.device != device:
        x = x.to(device)
        y = y.to(device)
    return x, y


def native_param_count(n_layer, n_embd, n_head):
    assert n_embd % n_head == 0
    padded_vocab = ((VOCAB_SIZE + 63) // 64) * 64
    return 2 * padded_vocab * n_embd + n_layer * 12 * n_embd * n_embd


def build_native(seq_len, n_layer, n_embd, n_head):
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern="L",
    )
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_config(run_cfg, no_hash_routing=False, no_shared_expert=False):
    config = DeepSeekV4NanoConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=0 if no_shared_expert else 1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=0 if no_hash_routing else -1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseekv4(run_cfg, no_hash_routing=False, no_shared_expert=False):
    config = deepseek_config(run_cfg, no_hash_routing=no_hash_routing, no_shared_expert=no_shared_expert)
    model = DeepSeekV4NanoChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_attnres_config(run_cfg):
    config = DeepSeekV4AttnResConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA,DN",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        attn_res_block_size=max(1, run_cfg.n_layer // 3),
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_attnres(run_cfg):
    config = deepseek_attnres_config(run_cfg)
    model = DeepSeekV4AttnResChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_attnmhc_config(run_cfg):
    config = DeepSeekV4AttnMhcConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA,DN",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        attn_res_block_size=max(1, run_cfg.n_layer // 3),
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_attnmhc(run_cfg):
    config = deepseek_attnmhc_config(run_cfg)
    model = DeepSeekV4AttnMhcChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_attnmhc_ch_config(run_cfg):
    """AttnMhc with simple CSA,HCA cycle (no DN, no extra HCA)."""
    config = DeepSeekV4AttnMhcConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        attn_res_block_size=max(1, run_cfg.n_layer // 3),
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_attnmhc_ch(run_cfg):
    config = deepseek_attnmhc_ch_config(run_cfg)
    model = DeepSeekV4AttnMhcChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_attnmhc_2h_config(run_cfg):
    config = DeepSeekV4AttnMhcConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA,HCA",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        attn_res_block_size=max(1, run_cfg.n_layer // 3),
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_attnmhc_2h(run_cfg):
    config = deepseek_attnmhc_2h_config(run_cfg)
    model = DeepSeekV4AttnMhcChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_attnmhc_swa_config(run_cfg):
    """Flash-style: first 2 layers SWA, then cycle CSA,HCA,DN."""
    config = DeepSeekV4AttnMhcConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="SWA,SWA+CSA,HCA,DN",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        attn_res_block_size=max(1, run_cfg.n_layer // 3),
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_attnmhc_swa(run_cfg):
    config = deepseek_attnmhc_swa_config(run_cfg)
    model = DeepSeekV4AttnMhcChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_attnmhc_hca_config(run_cfg):
    """Pro-style: first 2 layers HCA, then cycle CSA,HCA,DN."""
    config = DeepSeekV4AttnMhcConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="HCA,HCA+CSA,HCA,DN",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        attn_res_block_size=max(1, run_cfg.n_layer // 3),
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_attnmhc_hca(run_cfg):
    config = deepseek_attnmhc_hca_config(run_cfg)
    model = DeepSeekV4AttnMhcChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_mhc_config(run_cfg):
    config = DeepSeekV4MhcConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA,HCA",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_mhc(run_cfg):
    config = deepseek_mhc_config(run_cfg)
    model = DeepSeekV4MhcChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_dn_config(run_cfg):
    config = DeepSeekV4DnConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA,DN",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_dn(run_cfg):
    config = deepseek_dn_config(run_cfg)
    model = DeepSeekV4DnChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def deepseek_attnmhc_dnd_config(run_cfg):
    config = DeepSeekV4AttnMhcDndConfig(
        sequence_len=run_cfg.seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=run_cfg.n_layer,
        n_head=run_cfg.n_head,
        n_kv_head=1,
        n_embd=run_cfg.n_embd,
        window_size=max(16, run_cfg.seq_len // 2),
        attention_layer_pattern="CSA,HCA,DND",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (run_cfg.n_embd // run_cfg.n_head) // 2),
        q_lora_rank=max(16, run_cfg.n_embd // 2),
        o_lora_rank=max(16, run_cfg.n_embd // 2),
        o_groups=min(run_cfg.n_head, 4),
        n_routed_experts=run_cfg.deepseek_n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=run_cfg.deepseek_num_experts_per_tok,
        moe_intermediate_size=run_cfg.deepseek_moe_intermediate_size or max(64, run_cfg.n_embd),
        n_hash_layers=-1,
        n_hash_layers_frac=0.25,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=8,
        attn_res_block_size=max(1, run_cfg.n_layer // 3),
        dnd_n_future=2,
        dnd_max_cond=2,
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=run_cfg.seq_len,
        index_topk=run_cfg.deepseek_index_topk,
    )
    return config


def build_deepseek_attnmhc_dnd(run_cfg):
    config = deepseek_attnmhc_dnd_config(run_cfg)
    model = DeepSeekV4AttnMhcDndChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def choose_param_matched_native(run_cfg, target_active_params):
    best = None
    min_embd = max(run_cfg.n_head * 8, 64)
    max_embd = max(run_cfg.n_embd * 3, run_cfg.n_embd + run_cfg.n_head)
    embd_step = 2 * run_cfg.n_head
    min_embd = ((min_embd + embd_step - 1) // embd_step) * embd_step
    for n_layer in range(1, max(run_cfg.n_layer + 7, 8)):
        for n_embd in range(min_embd, max_embd + 1, embd_step):
            params = native_param_count(n_layer, n_embd, run_cfg.n_head)
            rel_delta = abs(params - target_active_params) / max(1, target_active_params)
            same_depth_penalty = 0 if n_layer == run_cfg.n_layer else 0.02
            score = rel_delta + same_depth_penalty
            candidate = (score, rel_delta, params, n_layer, n_embd)
            if best is None or candidate < best:
                best = candidate
    _, rel_delta, params, n_layer, n_embd = best
    if rel_delta > 0.05:
        print(
            f"Warning: closest parameter-matched GPT is {params:,} params, "
            f"{100 * rel_delta:.1f}% from target {target_active_params:,}",
            flush=True,
        )
    return n_layer, n_embd, params


def primary_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


@torch.inference_mode()
def evaluate(model, tokens, batch_size, seq_len, device, eval_batches, seed):
    model.eval()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    positions = torch.arange(seq_len + 1, device=device)
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(tokens, batch_size, seq_len, device, rng, positions)
        losses.append(primary_loss(model, x, y))
    model.train()
    return sum(losses) / len(losses)


@torch.inference_mode()
def evaluate_full_split(model, tokens, batch_size, seq_len, device):
    model.eval()
    max_start = len(tokens) - seq_len - 1
    num_windows = (max_start + seq_len - 1) // seq_len
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for i in range(0, num_windows, batch_size):
        windows = min(batch_size, num_windows - i)
        start = i * seq_len
        chunk = tokens[start:start + windows * seq_len + 1]
        x = chunk[:-1].view(windows, seq_len)
        y = chunk[1:].view(windows, seq_len)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss = total_loss + loss
        total_tokens += y.numel()
    model.train()
    return float((total_loss / total_tokens).detach().cpu())


def make_fixed_starts(num_tokens, seq_len, count, seed):
    max_start = num_tokens - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Dataset split is too short for seq_len={seq_len}: {num_tokens} byte tokens")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return torch.randint(0, max_start, (count,), generator=rng, device="cpu")


@torch.inference_mode()
def evaluate_fixed_starts(model, tokens, starts, batch_size, seq_len, device):
    model.eval()
    positions = torch.arange(seq_len + 1, device=device)
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for i in range(0, starts.numel(), batch_size):
        batch_starts = starts[i:i + batch_size].to(device)
        offsets = batch_starts[:, None] + positions[None, :]
        batch = tokens.index_select(0, offsets.reshape(-1)).view(batch_starts.numel(), seq_len + 1)
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss = total_loss + loss
        total_tokens += y.numel()
    model.train()
    return float((total_loss / total_tokens).detach().cpu())


def estimate_active_params(model):
    total = sum(p.numel() for p in model.parameters())
    if not isinstance(model, (DeepSeekV4NanoChat, DeepSeekV4AttnResChat, DeepSeekV4AttnMhcChat, DeepSeekV4MhcChat, DeepSeekV4DnChat, DeepSeekV4AttnMhcDndChat)):
        return total
    routed_total = 0
    for block in model.transformer.h:
        routed_total += sum(p.numel() for expert in block.moe.experts for p in expert.parameters())
    mtp_total = sum(p.numel() for p in model.mtp_heads.parameters())
    active_routed = round(routed_total * model.config.num_experts_per_tok / max(1, model.config.n_routed_experts))
    return total - routed_total - mtp_total + active_routed


def perplexity(loss):
    return math.exp(min(loss, 20.0))


def bits_per_byte(loss):
    return loss / math.log(2.0)


def display_dataset_name(dataset):
    if dataset == "wikitext2":
        return "WikiText-2"
    if dataset == "tiny_shakespeare":
        return "Tiny Shakespeare"
    return dataset.replace("_", " ").title()


def record_metric(rows, seed, config_name, model_name, dataset_name, split, tokens_trained, loss):
    rows.append({
        "seed": seed,
        "config": config_name,
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "tokens_trained": tokens_trained,
        "loss": loss,
        "ppl": perplexity(loss),
        "bits_per_byte": bits_per_byte(loss),
    })


def collect_expert_counts(model, seed, config_name, model_name, dataset_name, tokens_trained):
    if not isinstance(model, (DeepSeekV4NanoChat, DeepSeekV4AttnResChat, DeepSeekV4AttnMhcChat, DeepSeekV4MhcChat, DeepSeekV4DnChat, DeepSeekV4AttnMhcDndChat)):
        return []
    rows = []
    for layer_idx, block in enumerate(model.transformer.h):
        counts = block.moe.last_expert_counts.detach().cpu().tolist()
        for expert_idx, count in enumerate(counts):
            rows.append({
                "seed": seed,
                "config": config_name,
                "model": model_name,
                "dataset": dataset_name,
                "tokens_trained": tokens_trained,
                "layer": layer_idx,
                "expert": expert_idx,
                "count": int(count),
            })
    return rows


def train_one(model_name, model, dataset_name, splits, run_cfg, args, seed, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay)
    train_tokens_cpu = encode_bytes(splits["train"])
    train_tokens = train_tokens_cpu.to(device)
    val_tokens = encode_bytes(splits["validation"]).to(device)
    test_tokens = encode_bytes(splits["test"]).to(device)
    train_batch_tokens = train_tokens_cpu if device.type == "mps" else train_tokens

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    positions = torch.arange(run_cfg.seq_len + 1, device=train_batch_tokens.device)
    eval_batch_size = run_cfg.eval_batch_size or run_cfg.batch_size
    train_eval_starts = make_fixed_starts(
        train_tokens.numel(),
        run_cfg.seq_len,
        run_cfg.train_eval_batches * eval_batch_size,
        seed + 3000,
    )
    params = sum(p.numel() for p in model.parameters())
    active_params = estimate_active_params(model)
    metric_rows = []
    curve_rows = []
    expert_rows = []
    start_time = time.time()
    for step in range(1, run_cfg.steps + 1):
        x, y = get_batch(train_batch_tokens, run_cfg.batch_size, run_cfg.seq_len, device, rng, positions)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, (DeepSeekV4NanoChat, DeepSeekV4AttnResChat, DeepSeekV4AttnMhcChat, DeepSeekV4MhcChat, DeepSeekV4DnChat, DeepSeekV4AttnMhcDndChat)):
            loss = model(x, y, include_aux_loss=True)
        else:
            loss = model(x, y)
        loss.backward()
        if run_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), run_cfg.grad_clip)
        optimizer.step()
        if isinstance(model, (DeepSeekV4NanoChat, DeepSeekV4AttnResChat, DeepSeekV4AttnMhcChat, DeepSeekV4MhcChat, DeepSeekV4DnChat, DeepSeekV4AttnMhcDndChat)):
            model.update_aux_free_balance()

        should_eval = (
            (step == 1 and not args.skip_initial_eval)
            or (run_cfg.eval_every > 0 and step % run_cfg.eval_every == 0)
            or step == run_cfg.steps
        )
        if should_eval:
            tokens_trained = step * run_cfg.batch_size * run_cfg.seq_len
            expert_rows.extend(collect_expert_counts(
                model, seed, run_cfg.name, model_name, dataset_name, tokens_trained
            ))
            if args.full_eval:
                val_loss = evaluate_full_split(model, val_tokens, eval_batch_size, run_cfg.seq_len, device)
            else:
                val_loss = evaluate(
                    model, val_tokens, run_cfg.batch_size, run_cfg.seq_len, device, args.eval_batches, seed + 1000 + step
                )
            elapsed = time.time() - start_time
            record_metric(curve_rows, seed, run_cfg.name, model_name, dataset_name, "validation", tokens_trained, val_loss)
            print(
                f"seed {seed:<5d} {dataset_name:18s} {model_name:24s} "
                f"step {step:04d}/{run_cfg.steps:04d} val {val_loss:.4f} "
                f"train_obj {float(loss.detach().cpu()):.4f} elapsed {elapsed:.1f}s",
                flush=True,
            )

    tokens_trained = run_cfg.steps * run_cfg.batch_size * run_cfg.seq_len
    train_split_name = "train" if args.full_eval else "train_eval"
    final_splits = [
        (train_split_name, train_tokens),
        ("validation", val_tokens),
        ("test", test_tokens),
    ]
    for split_name, split_tokens in final_splits:
        if args.full_eval:
            split_loss = evaluate_full_split(model, split_tokens, eval_batch_size, run_cfg.seq_len, device)
        elif split_name == "train":
            split_loss = evaluate_fixed_starts(model, train_tokens, train_eval_starts, eval_batch_size, run_cfg.seq_len, device)
        else:
            split_loss = evaluate(
                model, split_tokens, run_cfg.batch_size, run_cfg.seq_len, device, args.eval_batches, seed + 4000
            )
        record_metric(metric_rows, seed, run_cfg.name, model_name, dataset_name, split_name, tokens_trained, split_loss)

    metadata = {
        "seed": seed,
        "config": run_cfg.name,
        "model": model_name,
        "dataset": dataset_name,
        "params": params,
        "active_params": active_params,
        "n_layer": getattr(model.config, "n_layer", run_cfg.n_layer),
        "n_embd": getattr(model.config, "n_embd", run_cfg.n_embd),
        "n_head": getattr(model.config, "n_head", run_cfg.n_head),
        "seq_len": run_cfg.seq_len,
        "steps": run_cfg.steps,
        "batch_size": run_cfg.batch_size,
    }
    return metric_rows, curve_rows, expert_rows, metadata


def format_token_tick(value, _pos=None):
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def set_loss_axis(ax, values):
    from matplotlib.ticker import MultipleLocator

    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    spread = y_max - y_min
    padding = max(spread * 0.12, 0.02)
    ax.set_ylim(y_min - padding, y_max + padding)
    if spread <= 0.35:
        major, minor = 0.05, 0.01
    elif spread <= 0.8:
        major, minor = 0.10, 0.02
    else:
        major, minor = 0.25, 0.05
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))


def style_token_axis(ax, max_tokens):
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.set_xlim(0, max_tokens)
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.xaxis.set_minor_locator(MultipleLocator(5000))
    ax.xaxis.set_major_formatter(FuncFormatter(format_token_tick))


def plot_results(records, output_path, metric_titles=None, y_label="cross-entropy loss"):
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in records})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.6, 3.7 * len(datasets)), squeeze=False)
    colors = {"native-nanochat": "#2f6fa3", "nanochat-deepseekv4": "#c44e36"}
    if metric_titles is None:
        metric_titles = [("train_loss", "Train eval loss"), ("validation_loss", "Validation loss")]
    for row_idx, dataset in enumerate(datasets):
        subset = [r for r in records if r["dataset"] == dataset]
        display_name = display_dataset_name(dataset)
        for col_idx, (metric, title) in enumerate(metric_titles):
            ax = axes[row_idx, col_idx]
            dataset_tokens = []
            for model_name in ["native-nanochat", "nanochat-deepseekv4"]:
                rows = [r for r in subset if r["model"] == model_name]
                if not rows:
                    continue
                tokens_seen = [r["tokens_seen"] for r in rows]
                dataset_tokens.extend(tokens_seen)
                ax.plot(
                    tokens_seen,
                    [r[metric] for r in rows],
                    label=model_name,
                    color=colors[model_name],
                    linewidth=1.7,
                    marker="o",
                    markersize=6.5,
                )
            metric_values = [r[metric] for r in subset]
            set_loss_axis(ax, metric_values)
            ax.set_title(f"{display_name}: {title}", fontsize=15)
            if dataset_tokens:
                style_token_axis(ax, max(dataset_tokens))
            ax.set_xlabel("training tokens seen")
            ax.set_ylabel(y_label)
            ax.grid(True, which="major", alpha=0.32)
            ax.grid(True, which="minor", alpha=0.12)
            ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Train eval and validation loss: native nanochat vs nanochat-deepseekv4", fontsize=16)
    fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_validation_curves(records, output_path):
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in records})
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.6, 4.0), squeeze=False)
    colors = {"native-nanochat": "#2f6fa3", "nanochat-deepseekv4": "#c44e36"}
    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        subset = [r for r in records if r["dataset"] == dataset]
        dataset_tokens = []
        for model_name in ["native-nanochat", "nanochat-deepseekv4"]:
            rows = [r for r in subset if r["model"] == model_name]
            if not rows:
                continue
            tokens_seen = [r["tokens_seen"] for r in rows]
            dataset_tokens.extend(tokens_seen)
            ax.plot(
                tokens_seen,
                [r["validation_loss"] for r in rows],
                label=model_name,
                color=colors[model_name],
                linewidth=1.8,
                marker="o",
                markersize=6.5,
            )
        set_loss_axis(ax, [r["validation_loss"] for r in subset])
        if dataset_tokens:
            style_token_axis(ax, max(dataset_tokens))
        ax.set_title(display_dataset_name(dataset), fontsize=15)
        ax.set_xlabel("training tokens seen")
        ax.set_ylabel("validation cross-entropy")
        ax.grid(True, which="major", alpha=0.32)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Validation loss: native nanochat vs nanochat-deepseekv4", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def final_records(records):
    finals = []
    datasets = sorted({r["dataset"] for r in records})
    for dataset in datasets:
        for model_name in ["native-nanochat", "nanochat-deepseekv4"]:
            rows = [r for r in records if r["dataset"] == dataset and r["model"] == model_name]
            if rows:
                finals.append(rows[-1])
    return finals


def plot_final_bars(records, output_path):
    import matplotlib.pyplot as plt

    finals = final_records(records)
    datasets = sorted({r["dataset"] for r in finals})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.6, 3.8 * len(datasets)), squeeze=False)
    colors = {"native-nanochat": "#2f6fa3", "nanochat-deepseekv4": "#c44e36"}
    for row_idx, dataset in enumerate(datasets):
        by_model = {r["model"]: r for r in finals if r["dataset"] == dataset}
        for col_idx, (key, title) in enumerate([
            ("test_loss", "Test loss"),
            ("validation_loss", "Validation loss"),
        ]):
            ax = axes[row_idx, col_idx]
            model_names = ["native-nanochat", "nanochat-deepseekv4"]
            values = [by_model[model_name][key] for model_name in model_names]
            bars = ax.bar(
                range(len(values)),
                values,
                color=[colors[model_name] for model_name in model_names],
                width=0.58,
            )
            ax.set_xticks(range(len(values)), ["native-nanochat", "nanochat-deepseekv4"])
            ax.set_title(f"{display_dataset_name(dataset)}: {title}", fontsize=15)
            ax.set_ylabel("cross-entropy loss")
            ax.grid(True, axis="y", alpha=0.25)
            y_min = min(values)
            y_max = max(values)
            padding = max((y_max - y_min) * 0.18, y_max * 0.02)
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    fig.suptitle("Final held-out loss: native nanochat vs nanochat-deepseekv4", fontsize=16)
    fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_rows(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metric_csv(rows, path):
    fields = ["seed", "config", "model", "dataset", "split", "tokens_trained", "loss", "ppl", "bits_per_byte"]
    write_rows(rows, path, fields)


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(var)


def write_summary_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config",
        "model",
        "dataset",
        "split",
        "tokens_trained",
        "n",
        "loss_mean",
        "loss_std",
        "ppl_mean",
        "ppl_std",
        "bits_per_byte_mean",
        "bits_per_byte_std",
    ]
    grouped = {}
    for row in rows:
        key = (row["config"], row["model"], row["dataset"], row["split"], row["tokens_trained"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for key, group in sorted(grouped.items()):
        config_name, model_name, dataset_name, split, tokens_trained = key
        loss_mean, loss_std = mean_std([float(row["loss"]) for row in group])
        ppl_mean, ppl_std = mean_std([float(row["ppl"]) for row in group])
        bpb_mean, bpb_std = mean_std([float(row["bits_per_byte"]) for row in group])
        summary_rows.append({
            "config": config_name,
            "model": model_name,
            "dataset": dataset_name,
            "split": split,
            "tokens_trained": tokens_trained,
            "n": len(group),
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "ppl_mean": ppl_mean,
            "ppl_std": ppl_std,
            "bits_per_byte_mean": bpb_mean,
            "bits_per_byte_std": bpb_std,
        })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)


def write_metadata_csv(rows, path):
    fields = [
        "seed",
        "config",
        "model",
        "dataset",
        "params",
        "active_params",
        "n_layer",
        "n_embd",
        "n_head",
        "seq_len",
        "steps",
        "batch_size",
    ]
    write_rows(rows, path, fields)


def write_expert_counts_csv(rows, path):
    fields = ["seed", "config", "model", "dataset", "tokens_trained", "layer", "expert", "count"]
    write_rows(rows, path, fields)


def parse_csv_arg(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_config(args):
    run_cfg = CONFIG_PRESETS[args.config]
    overrides = {}
    for arg_name, field_name in [
        ("steps", "steps"),
        ("eval_every", "eval_every"),
        ("batch_size", "batch_size"),
        ("eval_batch_size", "eval_batch_size"),
        ("train_eval_batches", "train_eval_batches"),
        ("seq_len", "seq_len"),
        ("n_layer", "n_layer"),
        ("n_embd", "n_embd"),
        ("n_head", "n_head"),
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
        ("grad_clip", "grad_clip"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    return replace(run_cfg, **overrides)


def deepseek_active_params_for_matching(run_cfg):
    model = build_deepseekv4(run_cfg)
    active = estimate_active_params(model)
    del model
    return active


def build_model_specs(selected_models, run_cfg, args):
    specs = []
    if "native" in selected_models:
        specs.append({
            "name": "native",
            "fingerprint": ("gpt", run_cfg.seq_len, run_cfg.n_layer, run_cfg.n_embd, run_cfg.n_head),
            "builder": lambda: build_native(run_cfg.seq_len, run_cfg.n_layer, run_cfg.n_embd, run_cfg.n_head),
        })
    if "param_matched" in selected_models:
        target_active = deepseek_active_params_for_matching(run_cfg)
        matched_layers, matched_embd, matched_params = choose_param_matched_native(run_cfg, target_active)
        print(
            f"param_matched GPT target active {target_active:,}; using "
            f"n_layer={matched_layers}, n_embd={matched_embd}, params={matched_params:,}",
            flush=True,
        )
        specs.append({
            "name": "param_matched",
            "fingerprint": ("gpt", run_cfg.seq_len, matched_layers, matched_embd, run_cfg.n_head),
            "builder": lambda ml=matched_layers, me=matched_embd: build_native(
                run_cfg.seq_len, ml, me, run_cfg.n_head
            ),
        })
    if "deepseekv4" in selected_models:
        variants = parse_csv_arg(args.ablations) if args.ablations else ["full"]
        if args.no_hash_routing or args.no_shared_expert:
            variants = ["custom"]
        for variant in variants:
            if variant not in {"full", "no_hash", "no_shared", "custom"}:
                raise ValueError("DeepSeek ablations must be one of: full,no_hash,no_shared")
            no_hash = args.no_hash_routing or variant == "no_hash"
            no_shared = args.no_shared_expert or variant == "no_shared"
            suffix = []
            if no_hash:
                suffix.append("no_hash")
            if no_shared:
                suffix.append("no_shared")
            model_name = "deepseekv4" if not suffix else "deepseekv4_" + "_".join(suffix)
            specs.append({
                "name": model_name,
                "fingerprint": (
                    "deepseekv4",
                    run_cfg.seq_len,
                    run_cfg.n_layer,
                    run_cfg.n_embd,
                    run_cfg.n_head,
                    no_hash,
                    no_shared,
                ),
                "builder": lambda nh=no_hash, ns=no_shared: build_deepseekv4(
                    run_cfg, no_hash_routing=nh, no_shared_expert=ns
                ),
            })
    if "deepseekv4_attnres" in selected_models:
        specs.append({
            "name": "deepseekv4_attnres",
            "fingerprint": (
                "deepseekv4_attnres",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_attnres(run_cfg),
        })
    if "deepseekv4_attnmhc" in selected_models:
        specs.append({
            "name": "deepseekv4_attnmhc",
            "fingerprint": (
                "deepseekv4_attnmhc",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_attnmhc(run_cfg),
        })
    if "deepseekv4_attnmhc_ch" in selected_models:
        specs.append({
            "name": "deepseekv4_attnmhc_ch",
            "fingerprint": (
                "deepseekv4_attnmhc_ch",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_attnmhc_ch(run_cfg),
        })
    if "deepseekv4_attnmhc_2h" in selected_models:
        specs.append({
            "name": "deepseekv4_attnmhc_2h",
            "fingerprint": (
                "deepseekv4_attnmhc_2h",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_attnmhc_2h(run_cfg),
        })
    if "deepseekv4_attnmhc_swa" in selected_models:
        specs.append({
            "name": "deepseekv4_attnmhc_swa",
            "fingerprint": (
                "deepseekv4_attnmhc_swa",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_attnmhc_swa(run_cfg),
        })
    if "deepseekv4_attnmhc_hca" in selected_models:
        specs.append({
            "name": "deepseekv4_attnmhc_hca",
            "fingerprint": (
                "deepseekv4_attnmhc_hca",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_attnmhc_hca(run_cfg),
        })
    if "deepseekv4_mhc" in selected_models:
        specs.append({
            "name": "deepseekv4_mhc",
            "fingerprint": (
                "deepseekv4_mhc",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_mhc(run_cfg),
        })
    if "deepseekv4_dn" in selected_models:
        specs.append({
            "name": "deepseekv4_dn",
            "fingerprint": (
                "deepseekv4_dn",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_dn(run_cfg),
        })
    if "deepseekv4_attnmhc_dnd" in selected_models:
        specs.append({
            "name": "deepseekv4_attnmhc_dnd",
            "fingerprint": (
                "deepseekv4_attnmhc_dnd",
                run_cfg.seq_len,
                run_cfg.n_layer,
                run_cfg.n_embd,
                run_cfg.n_head,
            ),
            "builder": lambda: build_deepseek_attnmhc_dnd(run_cfg),
        })
    if not specs:
        raise ValueError("No models selected")
    return specs


def clone_rows_for_model(rows, model_name):
    return [{**row, "model": model_name} for row in rows]


def clone_metadata_for_model(metadata, model_name):
    return {**metadata, "model": model_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="tiny_shakespeare,wikitext2", help="comma-separated: tiny_shakespeare,wikitext2")
    parser.add_argument("--config", default="small", choices=sorted(CONFIG_PRESETS), help="benchmark preset")
    parser.add_argument("--models", default="native,param_matched,deepseekv4,deepseekv4_attnres", help="comma-separated: native,param_matched,deepseekv4,deepseekv4_attnres,deepseekv4_attnmhc,deepseekv4_mhc")
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds to run starting at --seed")
    parser.add_argument("--ablations", default="full", help="DeepSeek variants: full,no_hash,no_shared")
    parser.add_argument("--no-hash-routing", action="store_true", help="disable hash routing for the DeepSeek model")
    parser.add_argument("--no-shared-expert", action="store_true", help="disable the DeepSeek shared expert")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None, help="batch size for --full-eval; 0 uses --batch-size")
    parser.add_argument("--train-eval-batches", type=int, default=None, help="fixed train-eval batches for pure next-token cross-entropy")
    parser.add_argument("--full-data", action="store_true", help="use full WikiText-2 train/validation/test splits instead of character caps")
    parser.add_argument("--full-eval", action="store_true", help="evaluate every non-overlapping window in validation/test splits")
    parser.add_argument("--skip-initial-eval", action="store_true", help="do not evaluate after step 1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--train-chars", type=int, default=600_000)
    parser.add_argument("--eval-chars", type=int, default=120_000)
    parser.add_argument("--output-dir", default="artifacts/deepseekv4_compare")
    args = parser.parse_args()

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")
    run_cfg = resolve_config(args)
    device_type = autodetect_device_type() if args.device == "auto" else args.device
    device = torch.device(device_type)
    cache_dir = Path(args.output_dir) / "data_cache"

    loaders = {
        "tiny_shakespeare": lambda: load_tiny_shakespeare(cache_dir),
        "wikitext2": lambda: load_wikitext2(cache_dir, args.train_chars, args.eval_chars, full_data=args.full_data),
    }
    selected_datasets = parse_csv_arg(args.datasets)
    selected_models = parse_csv_arg(args.models)
    unknown_models = sorted(set(selected_models) - {"native", "param_matched", "deepseekv4", "deepseekv4_attnres", "deepseekv4_attnmhc", "deepseekv4_attnmhc_ch", "deepseekv4_attnmhc_2h", "deepseekv4_attnmhc_swa", "deepseekv4_attnmhc_hca", "deepseekv4_mhc", "deepseekv4_dn", "deepseekv4_attnmhc_dnd"})
    if unknown_models:
        raise ValueError(f"Unknown model(s): {unknown_models}")
    model_specs = build_model_specs(selected_models, run_cfg, args)

    all_metric_rows = []
    all_curve_rows = []
    all_expert_rows = []
    all_metadata_rows = []
    for dataset_name in selected_datasets:
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset {dataset_name}. Choose from {sorted(loaders)}")
        print(f"\n=== Dataset: {dataset_name} ===")
        splits = loaders[dataset_name]()

        for seed_idx in range(args.seeds):
            seed = args.seed + seed_idx
            completed_specs = {}
            for spec in model_specs:
                fingerprint = spec["fingerprint"]
                if fingerprint in completed_specs:
                    metric_rows, curve_rows, expert_rows, metadata = completed_specs[fingerprint]
                    print(
                        f"{spec['name']}: reusing identical run from {metadata['model']} seed={seed}",
                        flush=True,
                    )
                    all_metric_rows.extend(clone_rows_for_model(metric_rows, spec["name"]))
                    all_curve_rows.extend(clone_rows_for_model(curve_rows, spec["name"]))
                    all_expert_rows.extend(clone_rows_for_model(expert_rows, spec["name"]))
                    all_metadata_rows.append(clone_metadata_for_model(metadata, spec["name"]))
                    continue

                torch.manual_seed(seed)
                model = spec["builder"]()
                params = sum(p.numel() for p in model.parameters())
                active_params = estimate_active_params(model)
                print(
                    f"{spec['name']}: params={params:,} active={active_params:,} seed={seed}",
                    flush=True,
                )
                metric_rows, curve_rows, expert_rows, metadata = train_one(
                    spec["name"], model, dataset_name, splits, run_cfg, args, seed, device
                )
                all_metric_rows.extend(metric_rows)
                all_curve_rows.extend(curve_rows)
                all_expert_rows.extend(expert_rows)
                all_metadata_rows.append(metadata)
                completed_specs[fingerprint] = (metric_rows, curve_rows, expert_rows, metadata)
                del model
                if device_type == "mps":
                    torch.mps.empty_cache()
                elif device_type == "cuda":
                    torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    run_csv_path = output_dir / "runs.csv"
    curve_csv_path = output_dir / "curves.csv"
    summary_csv_path = output_dir / "summary.csv"
    curve_summary_csv_path = output_dir / "curve_summary.csv"
    metadata_csv_path = output_dir / "model_metadata.csv"
    expert_csv_path = output_dir / "expert_counts.csv"
    write_metric_csv(all_metric_rows, run_csv_path)
    write_metric_csv(all_curve_rows, curve_csv_path)
    write_summary_csv(all_metric_rows, summary_csv_path)
    write_summary_csv(all_curve_rows, curve_summary_csv_path)
    write_metadata_csv(all_metadata_rows, metadata_csv_path)
    write_expert_counts_csv(all_expert_rows, expert_csv_path)

    print("\nFinal losses")
    for row in sorted(all_metric_rows, key=lambda r: (r["config"], r["dataset"], r["model"], r["seed"], r["split"])):
        if row["split"] in {"train", "validation", "test"}:
            print(
                f"{row['config']:7s} seed {row['seed']:<5d} {row['dataset']:18s} "
                f"{row['model']:24s} {row['split']:10s} "
                f"loss {row['loss']:.4f} ppl {row['ppl']:.2f} bpb {row['bits_per_byte']:.3f}"
            )

    print(f"\nWrote {run_csv_path}")
    print(f"Wrote {curve_csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {curve_summary_csv_path}")
    print(f"Wrote {metadata_csv_path}")
    print(f"Wrote {expert_csv_path}")


if __name__ == "__main__":
    main()
