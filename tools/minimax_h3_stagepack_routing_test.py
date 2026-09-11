#!/usr/bin/env python3
"""Routing test for minimax H3 stagepack TP16 plan selection (lane minimax).

tp16 tags in tensor_patterns.json must survive match_name into plan_of; r7
caught match_name dropping them, degrading every head plan to plain rows/cols
until the fail-closed extent guard aborted the real emit. Pure name routing,
no warm-copy IO.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import minimax_h3_stagepack as packer


def load():
    spec = json.loads((ROOT / "model-families" / "minimax_h3" / "tensor_patterns.json").read_text())
    patterns = []
    for entry in spec["patterns"]:
        rx = entry["regex"].replace("{N}", "(\\d+)").replace("{M}", "(\\d+)")
        patterns.append(dict(entry, rx=re.compile(rx)))
    codes = packer.assign_kind_codes(spec)
    excluded = [re.compile(rx) for rx in spec.get("excluded", [])]
    return patterns, codes, excluded, spec["tp16"]


def plan_for(patterns, codes, excluded, name, tp_degree):
    item = packer.match_name(name, patterns, codes, excluded)
    if item is None:
        return None
    return packer.plan_of(item, tp_degree)


def main() -> int:
    patterns, codes, excluded, tp16 = load()
    failures = []
    cases = [
        ("transformer_blocks.0.attn.to_q.weight", 16, "heads_rows"),
        ("transformer_blocks.49.attn.to_k.weight", 16, "heads_rows"),
        ("transformer_blocks.49.attn.to_v.weight", 16, "heads_rows"),
        ("transformer_blocks.0.attn.to_out.0.weight", 16, "heads_cols"),
        ("token_refiner.refiner_blocks.2.attn.to_q.weight", 16, "heads_rows"),
        ("token_refiner.refiner_blocks.1.attn.to_out.0.weight", 16, "heads_cols"),
        ("model.language_model.layers.0.self_attn.k_proj.weight", 16, "kv_rows"),
        ("model.language_model.layers.27.self_attn.v_proj.weight", 16, "kv_rows"),
        ("model.language_model.layers.0.self_attn.q_proj.weight", 16, "rows"),
        ("transformer_blocks.0.attn.to_q.weight", 4, "rows"),
        ("transformer_blocks.0.attn.to_out.0.weight", 4, "cols"),
        ("model.language_model.layers.0.self_attn.k_proj.weight", 4, "rows"),
    ]
    for name, tp_degree, expected in cases:
        got = plan_for(patterns, codes, excluded, name, tp_degree)
        state = "PASS" if got == expected else "FAIL"
        print(f"{state} tp{tp_degree} {name} -> {got} (want {expected})")
        if got != expected:
            failures.append(name)
    kv = packer.tp_slice(1024, 5376, "kv_rows", 15, 16, tp16)
    if kv != (7 * 128, 128, 0, 5376):
        failures.append("kv slice rank15")
        print(f"FAIL kv rank15 slice {kv}")
    else:
        print(f"PASS kv rank15 slice rows[{kv[0]},{kv[0] + kv[1]})")
    if failures:
        print(f"ROUTING TEST FAIL ({len(failures)})")
        return 1
    print("ROUTING TEST GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
