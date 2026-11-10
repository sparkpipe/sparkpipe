#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKER = ROOT / "tools/dsv41_flash_tp16_pack.py"
KV_SOURCES = [2, 8, 14, 20]
INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]


def load_packer():
    specification = importlib.util.spec_from_file_location("sparkpipe_dsv41_flash_tp16_pack", PACKER)
    if specification is None or specification.loader is None:
        raise RuntimeError("failed to load the DeepSeek V4.1 Flash Metal packer")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class Checkpoint:
    def __init__(self):
        self.read = []

    def bf16(self, name, shape):
        self.read.append(name)
        return np.zeros(shape, dtype=np.uint16)

    def f32(self, name, shape):
        self.read.append(name)
        return np.zeros(shape, dtype=np.float32)

    def widened(self, name, shape):
        self.read.append(name)
        return np.zeros(shape, dtype=np.float32)

    def fp8(self, name, rows, columns):
        self.read.append(name)
        return np.zeros((rows, columns), dtype=np.uint8), np.zeros((rows // 32, columns // 32), dtype=np.uint8)

    def mxfp4(self, name, rows, columns):
        self.read.append(name)
        return np.zeros((rows, columns // 2), dtype=np.uint8), np.zeros((rows, columns // 32), dtype=np.uint8)


class Writer:
    def __init__(self):
        self.entries = {}

    def add(self, name, payload, kind, shape):
        self.entries[name] = (kind, tuple(shape), len(payload))

    def add_parts(self, name, parts, kind, shape):
        self.entries[name] = (kind, tuple(shape), sum(len(np.ascontiguousarray(part).tobytes()) for part in parts))


def text_config():
    return {
        "hidden_size": 64, "num_attention_heads": 2, "head_dim": 64, "q_lora_rank": 64, "o_lora_rank": 32, "o_groups": 2,
        "n_routed_experts": 2, "moe_intermediate_size": 64, "hc_mult": 4, "index_n_heads": 2, "index_head_dim": 32,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20, "kv_source_layer_ids": KV_SOURCES, "index_source_layer_ids": INDEX_SOURCES,
    }


def pack(module, layer):
    packs = module.RankPacks.__new__(module.RankPacks)
    packs.tp = 1
    packs.ranks = [0]
    packs.writers = {0: Writer()}
    checkpoint = Checkpoint()
    module.pack_layer(checkpoint, packs, text_config(), {"layers": []}, layer)
    return packs.writers[0].entries, checkpoint.read


def main() -> int:
    module = load_packer()
    for layer in range(40):
        entries, read = pack(module, layer)
        name = f"layers.{layer}."
        source = f"layers.{layer}.attn.indexer."
        keyed = layer in KV_SOURCES
        indexed = layer in INDEX_SOURCES
        if (name + "idx_k" in entries) != keyed or (name + "idx_k_norm" in entries) != keyed:
            raise AssertionError(f"layer {layer}: the index key projection is packed exactly on the compressing layers")
        if (name + "idx_q_b" in entries) != indexed or (name + "idx_q_b_scale" in entries) != indexed or (name + "idx_weights" in entries) != indexed:
            raise AssertionError(f"layer {layer}: the index query and head weights are packed exactly on the index source layers")
        if keyed:
            if entries[name + "idx_k"][:2] != ("bf16", (32, 64)) or entries[name + "idx_k_norm"][:2] != ("f32", (32,)):
                raise AssertionError(f"layer {layer}: idx_k is bf16 [index dim, head dim] and idx_k_norm f32 [index dim]")
            if source + "wk.weight" not in read or source + "k_norm.weight" not in read:
                raise AssertionError(f"layer {layer}: the index key tensors come from the checkpoint's indexer.wk and indexer.k_norm")
        if indexed:
            if entries[name + "idx_q_b"][:2] != ("fp8", (64, 64)) or entries[name + "idx_q_b_scale"][:2] != ("u8", (64, 2)):
                raise AssertionError(f"layer {layer}: idx_q_b holds every index head's FP8 rows with a scale byte per row and 32 columns")
            if entries[name + "idx_weights"][:2] != ("bf16", (2, 64)):
                raise AssertionError(f"layer {layer}: idx_weights is bf16 [index heads, hidden]")
            if source + "wq_b" not in read or source + "weights_proj.weight" not in read:
                raise AssertionError(f"layer {layer}: the index query tensors come from the checkpoint's indexer.wq_b and indexer.weights_proj")
    print("PASS the DeepSeek V4.1 Flash Metal packs carry the indexer on its source layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
