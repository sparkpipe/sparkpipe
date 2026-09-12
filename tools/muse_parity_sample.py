#!/usr/bin/env python3
"""Re-derive sampled muse pack payloads in place with the lane packer.

Copies of placed packs are rewritten by muse_glimmer_stagepack's own
write_records over the sampled records (globals + caller-chosen layers).
An unchanged whole-file sha256 after the rewrite proves the lane packer
plus the warm checkpoint reproduce those payload bytes exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import muse_glimmer_stagepack as P
from spark_pack_common import PackFailure, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True,
                        help="writable copy of the placed pack (rewritten in place)")
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--layers", type=str, default="51",
                        help="comma-separated layer indexes to sample")
    parser.add_argument("--expect-sha", type=str, required=True)
    arguments = parser.parse_args()
    before = sha256_file(arguments.pack)
    if before != arguments.expect_sha:
        raise PackFailure(f"copy sha {before} != expected {arguments.expect_sha}")
    records = P.build_records(16, arguments.tp_rank)
    source = P.SafetensorsSource(arguments.source)
    source.check_config({
        "hidden_size": P.HIDDEN,
        "num_hidden_layers": P.LAYER_COUNT,
        "vocab_size": P.VOCAB,
        "num_attention_heads": P.ATTN_QUERY_HEADS,
        "num_key_value_heads": P.ATTN_KV_HEADS,
        "head_dim": P.ATTN_HEAD_DIM,
        "intermediate_size": P.INTERMEDIATE,
    }, section="text_config")
    entries, file_bytes = P.build_directory(records)
    if file_bytes != os.path.getsize(arguments.pack):
        raise PackFailure("pack size != packer layout size")
    layers = {int(x) for x in arguments.layers.split(",") if x}
    subset = [(record, offset, scale) for record, offset, scale in entries
              if record.layer in layers or record.layer == P.GLOBAL_LAYER]
    P.write_records(source, subset, arguments.pack)
    after = sha256_file(arguments.pack)
    result = {
        "family": "muse_glimmer",
        "tp_rank": arguments.tp_rank,
        "sampled_layers": sorted(layers),
        "sampled_records": len(subset),
        "sha_before": before,
        "sha_after": after,
        "expect_sha": arguments.expect_sha,
        "parity": after == before == arguments.expect_sha,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["parity"] else 1


if __name__ == "__main__":
    sys.exit(main())
