#!/usr/bin/env python3
"""Routed-expert parity of one glm52 rank pack against a BF16 reference rank pack.

Both packs are the same TP rank. For each --layers entry the tool decodes the
routed-expert entries of both packs exactly as the pack format defines them
(bf16 verbatim; fp8 e4m3 x one f32 scale per row and 128-column block; nvfp4
e2m1 codes, low nibble first, x UE4M3 per-16 block scale x the expert's f32
global stored ahead of the block scales), then runs the routed MLP of this
rank's shard (up rows then gate rows, SiLU(gate) * up, down) on --tokens
seeded unit-RMS activations routed to --top-k seeded experts with softmax
weights. It reports per layer the relative L2 of the arm's routed output
against the reference, the output norm ratio, and the weight-level relative
L2 and norm ratio of the routed experts it decoded.

A layout error (wrong nibble order, globals read as block scales, up/gate
swapped, a shifted expert stride) shows as a norm ratio far from 1 or a
relative L2 near or above 1; a correct 4-bit decode of the publisher's
weights sits near 0.1 with a norm ratio near 1, a correct fp8 decode near
0.002-0.005.

  python3 tools/glm52_routed_parity.py --pack <arm rank pack> --reference <bf16 rank pack> \\
      --layers 3,40,77 --out parity.json

Exit 0 = parity measured and every layer within --max-rel-l2 and
--max-norm-error; 1 = a layer outside; 2 = unreadable or mismatched packs.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HEADER_FORMAT = "<20I2Q65s32s32s32s"
ENTRY_FORMAT = "<8I4Q"
ENTRY_BYTES = 64
MAGIC = 0x32534C47
KIND_UP_GATE, KIND_DOWN = 22, 23
CODEC_BF16, CODEC_FP8, CODEC_NVFP4 = 1, 5, 6
FP8_BLOCK = 128
NVFP4_BLOCK = 16
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32)


class PackError(Exception):
    pass


def e4m3_table() -> np.ndarray:
    values = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values[code] = np.nan
        elif exponent == 0:
            values[code] = sign * (mantissa / 8.0) * 2.0 ** -6
        else:
            values[code] = sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return values


E4M3 = e4m3_table()


class RankPack:
    def __init__(self, path: Path):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        raw = bytes(self.data[:struct.calcsize(HEADER_FORMAT)])
        fields = struct.unpack(HEADER_FORMAT, raw)
        words = fields[:20]
        if words[0] != MAGIC:
            raise PackError(f"{path}: not a glm52 rank pack")
        self.expert_codec = words[16]
        self.tp = (words[18], words[19])
        self.routed_experts = words[14]
        directory_offset, tensor_count = fields[20], words[6]
        self.entries = {}
        for index in range(tensor_count):
            start = directory_offset + index * ENTRY_BYTES
            values = struct.unpack(ENTRY_FORMAT, bytes(self.data[start:start + ENTRY_BYTES]))
            entry = dict(zip(("kind", "layer", "payload_type", "weight_codec", "scale_encoding", "group_count",
                              "rows", "columns", "payload_offset", "payload_bytes", "scale_offset",
                              "scale_bytes"), values))
            self.entries[(entry["kind"], entry["layer"])] = entry

    def entry(self, kind: int, layer: int) -> dict:
        entry = self.entries.get((kind, layer))
        if entry is None:
            raise PackError(f"{self.path}: no kind={kind} layer={layer} entry")
        return entry

    def expert(self, kind: int, layer: int, expert: int) -> np.ndarray:
        entry = self.entry(kind, layer)
        groups, rows, columns = entry["group_count"], entry["rows"], entry["columns"]
        codec = entry["weight_codec"]
        if expert >= groups:
            raise PackError(f"expert {expert} >= {groups}")
        per_payload = entry["payload_bytes"] // groups
        base = entry["payload_offset"] + expert * per_payload
        payload = self.data[base:base + per_payload]
        if codec == CODEC_BF16:
            if entry["scale_bytes"] != 0 or per_payload != rows * columns * 2:
                raise PackError(f"kind={kind} layer={layer}: bf16 entry geometry")
            return (payload.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, columns)
        if codec == CODEC_FP8:
            blocks = columns // FP8_BLOCK
            per_scale = entry["scale_bytes"] // groups
            if per_payload != rows * columns or per_scale != rows * blocks * 4 or columns % FP8_BLOCK:
                raise PackError(f"kind={kind} layer={layer}: fp8 entry geometry")
            start = entry["scale_offset"] + expert * per_scale
            scale = self.data[start:start + per_scale].view(np.float32).reshape(rows, blocks)
            weight = E4M3[payload].reshape(rows, blocks, FP8_BLOCK)
            return (weight * scale[:, :, None]).reshape(rows, columns)
        if codec == CODEC_NVFP4:
            blocks = columns // NVFP4_BLOCK
            per_block = rows * blocks
            if per_payload != rows * columns // 2 or entry["scale_bytes"] != groups * 4 + groups * per_block:
                raise PackError(f"kind={kind} layer={layer}: nvfp4 entry geometry")
            global_scale = float(self.data[entry["scale_offset"] + expert * 4:
                                           entry["scale_offset"] + expert * 4 + 4].view(np.float32)[0])
            start = entry["scale_offset"] + groups * 4 + expert * per_block
            block = E4M3[self.data[start:start + per_block]].reshape(rows, blocks)
            codes = np.empty(rows * columns, dtype=np.uint8)
            codes[0::2] = payload & 0x0F
            codes[1::2] = payload >> 4
            weight = E2M1[codes].reshape(rows, blocks, NVFP4_BLOCK)
            return (weight * block[:, :, None] * global_scale).reshape(rows, columns)
        raise PackError(f"kind={kind} layer={layer}: codec {codec} is not decoded here")


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(value - reference)) / denominator if denominator > 0 else float("inf")


def routed_output(pack: RankPack, layer: int, x: np.ndarray, experts: np.ndarray, weights: np.ndarray,
                  cache: dict) -> np.ndarray:
    output = np.zeros((x.shape[0], pack.entry(KIND_DOWN, layer)["rows"]), dtype=np.float64)
    for token in range(x.shape[0]):
        for slot in range(experts.shape[1]):
            expert = int(experts[token, slot])
            if (layer, expert) not in cache:
                cache[(layer, expert)] = (pack.expert(KIND_UP_GATE, layer, expert).astype(np.float64),
                                          pack.expert(KIND_DOWN, layer, expert).astype(np.float64))
            up_gate, down = cache[(layer, expert)]
            half = up_gate.shape[0] // 2
            projected = up_gate @ x[token]
            up, gate = projected[:half], projected[half:]
            activated = gate / (1.0 + np.exp(-gate)) * up
            output[token] += weights[token, slot] * (down @ activated)
    return output


def measure(pack: RankPack, reference: RankPack, layer: int, tokens: int, top_k: int, seed: int) -> dict:
    for kind in (KIND_UP_GATE, KIND_DOWN):
        a, b = pack.entry(kind, layer), reference.entry(kind, layer)
        if (a["group_count"], a["rows"], a["columns"]) != (b["group_count"], b["rows"], b["columns"]):
            raise PackError(f"kind={kind} layer={layer}: geometry differs between the packs")
    rng = np.random.default_rng(seed + layer)
    columns = pack.entry(KIND_UP_GATE, layer)["columns"]
    groups = pack.entry(KIND_UP_GATE, layer)["group_count"]
    x = rng.standard_normal((tokens, columns))
    x /= np.sqrt(np.mean(x * x, axis=1, keepdims=True))
    experts = np.stack([rng.choice(groups, size=top_k, replace=False) for _ in range(tokens)])
    logits = rng.standard_normal((tokens, top_k))
    weights = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    arm_cache, reference_cache = {}, {}
    arm = routed_output(pack, layer, x, experts, weights, arm_cache)
    ref = routed_output(reference, layer, x, experts, weights, reference_cache)
    weight_errors, weight_ratios = [], []
    for key in sorted(arm_cache):
        for plane in range(2):
            weight_errors.append(relative_l2(arm_cache[key][plane], reference_cache[key][plane]))
            weight_ratios.append(float(np.linalg.norm(arm_cache[key][plane]) / np.linalg.norm(reference_cache[key][plane])))
    return {"layer": layer, "tokens": tokens, "top_k": top_k, "experts_decoded": len(arm_cache),
            "output_rel_l2": relative_l2(arm, ref),
            "output_norm_ratio": float(np.linalg.norm(arm) / np.linalg.norm(ref)),
            "output_finite": bool(np.isfinite(arm).all()),
            "weight_rel_l2_mean": float(np.mean(weight_errors)), "weight_rel_l2_max": float(np.max(weight_errors)),
            "weight_norm_ratio_min": float(np.min(weight_ratios)), "weight_norm_ratio_max": float(np.max(weight_ratios))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--layers", default="3,40,77")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-rel-l2", type=float, default=0.2)
    parser.add_argument("--max-norm-error", type=float, default=0.05)
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        pack, reference = RankPack(Path(args.pack)), RankPack(Path(args.reference))
        if pack.tp != reference.tp:
            raise PackError(f"tp rank differs: {pack.tp} vs {reference.tp}")
        if reference.expert_codec != CODEC_BF16:
            raise PackError(f"the reference pack's expert codec is {reference.expert_codec}, not bf16")
        layers = [measure(pack, reference, int(layer), args.tokens, args.top_k, args.seed)
                  for layer in args.layers.split(",")]
    except (OSError, ValueError, PackError) as error:
        print(f"ROUTED-PARITY ERROR: {error}", file=sys.stderr)
        return 2
    failing = [row["layer"] for row in layers
               if not row["output_finite"] or row["output_rel_l2"] > args.max_rel_l2
               or abs(row["output_norm_ratio"] - 1.0) > args.max_norm_error]
    report = {"pack": args.pack, "reference": args.reference, "tp": list(pack.tp), "expert_codec": pack.expert_codec,
              "seed": args.seed, "max_rel_l2": args.max_rel_l2, "max_norm_error": args.max_norm_error,
              "layers": layers, "result": "FAIL" if failing else "PASS"}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for row in layers:
        print(f"layer {row['layer']}: output rel_l2 {row['output_rel_l2']:.4g} norm_ratio {row['output_norm_ratio']:.4f} "
              f"weight rel_l2 mean {row['weight_rel_l2_mean']:.4g} max {row['weight_rel_l2_max']:.4g} "
              f"norm_ratio [{row['weight_norm_ratio_min']:.4f}, {row['weight_norm_ratio_max']:.4f}] "
              f"({row['experts_decoded']} experts)")
    print(f"ROUTED-PARITY {report['result']} codec {pack.expert_codec} tp{pack.tp[0]} rank {pack.tp[1]} "
          f"layers {args.layers} against the bf16 reference")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
