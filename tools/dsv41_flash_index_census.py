#!/usr/bin/env python3
"""Standalone census verifier for the DeepSeek-V4.1-Flash safetensors index.

Re-derives the per-layer tensor census from the real
model.safetensors.index.json weight_map and asserts agreement with the
corrected stagepack kind map: compressor tensors exactly on the kv-source
layers {2,8,14,20} with wkv+wgate+norm members on gate layers {2,8,14} and
wkv+norm only on layer 20 (11 compressor tensors total, no wgate on
layer 20); indexer query tensors (wq_b, weights_proj) exactly on layers
{2,8,14,20,24,28,32,36} with key tensors (wk, k_norm) only on the kv-source
layers (24 indexer tensors total, reference Indexer owns_k); and a uniform
attention member set across all 40 backbone layers.

Feed it the sha256-verified index (pinned oid 74b0686a3d2891980d5e3032
51b075a3bccae2c2ff650747db2620a649b98fa8 at revision dba1be0a). The
verifier never reads shard payloads, so it runs on any host with the
index alone.

Exit 0 = census agrees; non-zero = first disagreement, named loudly.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

LAYER_COUNT = 40
KV_SOURCE_LAYERS = {2, 8, 14, 20}
COMPRESS_GATE_LAYERS = {2, 8, 14}
INDEX_SOURCE_LAYERS = {2, 8, 14, 20, 24, 28, 32, 36}
COMPRESSOR_TENSOR_TOTAL = 11
INDEXER_TENSOR_TOTAL = 24
ATTENTION_MEMBERS = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b", "q_norm", "kv_norm", "attn_sink"}
INDEXER_QUERY_MEMBERS = {"wq_b", "weights_proj"}
INDEXER_KEY_MEMBERS = {"wk", "k_norm"}
COMPRESSOR_MEMBERS_GATE = {"wkv", "wgate", "norm"}
COMPRESSOR_MEMBERS_PLAIN = {"wkv", "norm"}


def fail(message: str):
    print(f"dsv41 index census FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        fail("usage: dsv41_flash_index_census.py INDEX_JSON")
    with open(sys.argv[1], encoding="utf-8") as handle:
        document = json.load(handle)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict):
        fail("weight_map missing")
    compressor = defaultdict(set)
    indexer = defaultdict(set)
    attention = defaultdict(set)
    for name in weight_map:
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "layers" or parts[2] != "attn":
            continue
        layer = int(parts[1])
        if parts[3] == "compressor":
            compressor[layer].add(parts[4])
        elif parts[3] == "indexer":
            indexer[layer].add(parts[4])
        else:
            attention[layer].add(parts[3])
    compressor_layer_ids = set(compressor)
    if compressor_layer_ids != KV_SOURCE_LAYERS:
        fail(f"compressor layers {sorted(compressor_layer_ids)} != kv-source {sorted(KV_SOURCE_LAYERS)}")
    compressor_total = sum(len(members) for members in compressor.values())
    if compressor_total != COMPRESSOR_TENSOR_TOTAL:
        fail(f"compressor tensor count {compressor_total} != {COMPRESSOR_TENSOR_TOTAL}")
    for layer, members in compressor.items():
        expected = COMPRESSOR_MEMBERS_GATE if layer in COMPRESS_GATE_LAYERS else COMPRESSOR_MEMBERS_PLAIN
        if members != expected:
            fail(f"layer {layer} compressor members {sorted(members)} != {sorted(expected)}")
    indexer_layer_ids = set(indexer)
    if indexer_layer_ids != INDEX_SOURCE_LAYERS:
        fail(f"indexer layers {sorted(indexer_layer_ids)} != index-source {sorted(INDEX_SOURCE_LAYERS)}")
    indexer_total = sum(len(members) for members in indexer.values())
    if indexer_total != INDEXER_TENSOR_TOTAL:
        fail(f"indexer tensor count {indexer_total} != {INDEXER_TENSOR_TOTAL}")
    for layer, members in indexer.items():
        expected = INDEXER_QUERY_MEMBERS | (INDEXER_KEY_MEMBERS if layer in KV_SOURCE_LAYERS else set())
        if members != expected:
            fail(f"layer {layer} indexer members {sorted(members)} != {sorted(expected)}")
    if set(attention) != set(range(LAYER_COUNT)):
        fail(f"attention layers {sorted(attention)} != 0..{LAYER_COUNT - 1}")
    for layer, members in attention.items():
        if members != ATTENTION_MEMBERS:
            fail(f"layer {layer} attention members {sorted(members)} != {sorted(ATTENTION_MEMBERS)}")
    total_size = (document.get("metadata") or {}).get("total_size")
    print(
        f"dsv41 index census PASS: {len(weight_map)} entries, "
        f"compressor {compressor_total} (wgate only on {sorted(COMPRESS_GATE_LAYERS)}), "
        f"indexer {indexer_total} (keys only on {sorted(KV_SOURCE_LAYERS)}), total_size {total_size}"
    )


if __name__ == "__main__":
    main()
