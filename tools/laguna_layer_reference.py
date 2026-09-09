#!/usr/bin/env python3
"""Independent fp32 oracle for the laguna layer path.

numpy only, no torch, no driver imports. Written from
model_contracts/laguna_authoritative.json and the pinned modeling_laguna.py;
used by the layer host test and the V0 synth gates. The C side is never
consulted for any formula here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_geometry():
    return json.loads((ROOT / "model_contracts/laguna_authoritative.json").read_text())


def yarn_inv_freq(rotary_dimension, theta, factor, original_positions, beta_fast, beta_slow):
    """HF yarn: low/high correction dims from the stated betas, linear ramp
    blend between the extrapolated base frequency and base/factor."""
    half = rotary_dimension // 2
    low = math.floor(max(min(
        rotary_dimension * math.log(original_positions / (beta_fast * 2.0 * math.pi))
        / (2.0 * math.log(theta)), rotary_dimension - 1.0), 0.0))
    high = math.ceil(max(min(
        rotary_dimension * math.log(original_positions / (beta_slow * 2.0 * math.pi))
        / (2.0 * math.log(theta)), rotary_dimension - 1.0), 0.0))
    index = np.arange(half, dtype=np.float64)
    base = np.power(np.float64(theta), -2.0 * index / np.float64(rotary_dimension))
    ramp = (index - low) / max(high - low, 1e-6)
    blend = np.clip(ramp, 0.0, 1.0)
    return base * (1.0 - blend) + (base / factor) * blend


def attention_factor(factor):
    return 0.1 * math.log(factor) + 1.0


def rope_neox(head, rotary_dimension, inv_freq, position, attention_scale):
    """NeoX half-split rope on the FIRST rotary_dimension elements."""
    half = rotary_dimension // 2
    head = head.copy()
    first = head[:half].astype(np.float64)
    second = head[half:rotary_dimension].astype(np.float64)
    angles = position * inv_freq
    cos = np.cos(angles) * attention_scale
    sin = np.sin(angles) * attention_scale
    head[:half] = first * cos - second * sin
    head[half:rotary_dimension] = first * sin + second * cos
    return head


def rms_norm(values, weight, epsilon):
    squared = np.mean(values.astype(np.float64) ** 2)
    scale = 1.0 / math.sqrt(squared + epsilon)
    return (values.astype(np.float64) * scale * weight.astype(np.float64)).astype(np.float32)


def softplus(x):
    return x if x > 20.0 else math.log1p(math.exp(x))


def sigmoid_router(logits, correction_bias, top_k, routed_scaling):
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    selection = scores + correction_bias.astype(np.float64)
    order = np.argsort(-selection, kind="stable")[:top_k]
    chosen = scores[order]
    weights = chosen / chosen.sum()
    return order, weights * routed_scaling


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-yarn", action="store_true",
                        help="print the 32 yarn frequencies, one per line (V0 anchor)")
    arguments = parser.parse_args()
    geometry = load_geometry()
    yarn = geometry["yarn"]
    table = yarn_inv_freq(yarn["rotary_dimension"], yarn["theta"],
                          yarn["factor"], yarn["original_positions"],
                          yarn["beta_fast"], yarn["beta_slow"])
    if arguments.dump_yarn:
        for value in table:
            print(f"{value:.17g}")
        return 0
    stated = geometry["yarn"]["attention_factor"]
    if abs(attention_factor(yarn["factor"]) - stated) > 1e-15:
        print(f"stated attention factor {stated} disagrees with 0.1*ln(factor)+1")
        return 1
    print(f"oracle: yarn table {len(table)} frequencies, attention factor {stated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
