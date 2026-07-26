"""Measure a weight sample and recommend block-int codec settings.

Bit width and block size are not universal constants. Expert weights differ
enough between model families that the same setting is not the right answer
everywhere, so the packer should measure rather than assume. Sampled expert
gate_proj tensors, error against the bf16 reference:

    family        kurtosis  gamma^2   FP8/128   INT8/128  INT6/32
    GLM-5.2           3.01    1.190    2.572%     0.668%   2.196%
    Kimi-K2           3.16    1.189    2.663%     0.697%   2.281%
    DeepSeek-V3       7.13    1.241    2.480%     0.762%   2.446%
    Qwen3-Next       13.43    1.570    2.548%     0.730%   2.308%

INT8 beats FP8 by 3.3-3.7x on every family, so the ranking is stable. The
magnitudes are not: Qwen3-Next is heavier tailed and less centred, which costs
it accuracy at every width and makes a zero point worth paying for where on
GLM-5.2 it is not.

Two statistics drive the recommendation:

  kurtosis  tail weight. Above roughly 8 the block maximum is set by outliers
            rather than by the bulk, so a smaller block recovers accuracy.

  gamma^2   centering inefficiency, mean over blocks of
            (2*max(|min|,|max|)/(max-min))^2. Symmetric quantisation inflates
            noise variance by this factor. A zero point removes it and costs
            8/block bits, so it pays when 10*log10(gamma^2) exceeds
            6.02*8/block dB.

Usage:
    sparkpipe_quant_calibrate.py WEIGHTS.bf16 [--target-error 0.01]
    sparkpipe_quant_calibrate.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sparkpipe_block_int_codec as codec

BIT_CHOICES = (6, 7, 8)
BLOCK_CHOICES = (32, 64, 128, 256)
HEAVY_TAIL_KURTOSIS = 8.0
DEFAULT_TARGET_ERROR = 0.01


def load_bf16(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size == 0:
        raise ValueError("weight sample is empty")
    # Guard against being handed a tensor that is not bf16. Reinterpreting an
    # fp8 or int8 payload as bf16 produces a plausible-looking but meaningless
    # report, so check the exponent field: real bf16 weights concentrate in a
    # narrow band of live exponents, whereas a byte stream of some other dtype
    # spreads across nearly all 256. Real bf16 samples measured 33 (GLM-5.2) to
    # 105 (Qwen3-Next); misread fp8 payloads measured 254.
    exponent = (raw >> np.uint16(7)) & np.uint16(0xFF)
    live = int(np.bincount(exponent, minlength=256).astype(bool).sum())
    if live > 160:
        raise ValueError(
            "sample does not look like bf16: %d live exponent values. "
            "Convert to bf16 first; an fp8 or int8 payload read as bf16 "
            "yields a meaningless report." % live)
    if int(exponent.max(initial=0)) == 0xFF:
        raise ValueError("weight sample contains inf or nan")
    return raw


def relative_error(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    residual = float(np.linalg.norm(reference - reconstructed))
    magnitude = float(np.linalg.norm(reference))
    return residual / magnitude if magnitude > 0.0 else 0.0


def distribution_statistics(values: np.ndarray, block: int) -> dict[str, float]:
    trimmed = values[: (values.size // block) * block].reshape(-1, block)
    centred = values - values.mean()
    variance = float((centred * centred).mean())
    kurtosis = float((centred ** 4).mean() / variance ** 2) if variance > 0.0 else 0.0
    upper = trimmed.max(axis=1)
    lower = trimmed.min(axis=1)
    span = np.where(upper - lower == 0.0, 1.0, upper - lower)
    gamma = 2.0 * np.maximum(np.abs(lower), np.abs(upper)) / span
    return {
        "kurtosis": kurtosis,
        "gamma_squared": float((gamma * gamma).mean()),
        "zero_fraction": float((values == 0.0).mean()),
    }


def measure(source: np.ndarray, bits: int, block: int) -> dict[str, float]:
    usable = (source.size // block) * block
    reference = codec.bf16_to_f32(source[:usable]).astype(np.float64)
    codes, scales, stats = codec.encode(source[:usable], block, bits)
    reconstructed = codec.bf16_to_f32(codec.decode(codes, scales, block)).astype(np.float64)
    return {
        "bits": bits,
        "block": block,
        "bits_per_weight": stats["bits_per_weight"],
        "error": relative_error(reference, reconstructed),
    }


def zero_point_is_worth_it(gamma_squared: float, block: int) -> bool:
    symmetric_penalty_db = 10.0 * np.log10(max(gamma_squared, 1.0))
    zero_point_cost_db = 6.02 * (8.0 / block)
    return symmetric_penalty_db > zero_point_cost_db


def recommend(source: np.ndarray, target_error: float) -> dict[str, object]:
    statistics = distribution_statistics(
        codec.bf16_to_f32(source).astype(np.float64), 128
    )
    grid = [measure(source, bits, block) for bits in BIT_CHOICES for block in BLOCK_CHOICES]
    admissible = [row for row in grid if row["error"] <= target_error]
    # Cheapest setting that meets the target; if none does, the most accurate.
    if admissible:
        choice = min(admissible, key=lambda row: (row["bits_per_weight"], row["error"]))
    else:
        choice = min(grid, key=lambda row: row["error"])
    return {
        "statistics": statistics,
        "grid": grid,
        "choice": choice,
        "meets_target": bool(admissible),
        "asymmetric_recommended": zero_point_is_worth_it(
            statistics["gamma_squared"], choice["block"]
        ),
    }


def report(result: dict[str, object], target_error: float) -> None:
    statistics = result["statistics"]
    print("distribution")
    print("  kurtosis       %8.2f%s" % (
        statistics["kurtosis"],
        "  (heavy tailed, prefer a smaller block)"
        if statistics["kurtosis"] > HEAVY_TAIL_KURTOSIS else ""))
    print("  gamma^2        %8.3f  (symmetric penalty %+.2f dB)" % (
        statistics["gamma_squared"],
        10.0 * np.log10(max(statistics["gamma_squared"], 1.0))))
    print("  zero fraction  %8.4f" % statistics["zero_fraction"])
    print()
    print("error grid")
    print("  %6s %7s %13s %9s" % ("bits", "block", "bits/weight", "error"))
    for row in sorted(result["grid"], key=lambda item: (item["bits"], item["block"])):
        print("  %6d %7d %13.3f %8.3f%%" % (
            row["bits"], row["block"], row["bits_per_weight"], 100.0 * row["error"]))
    print()
    choice = result["choice"]
    print("recommendation for target error %.3f%%" % (100.0 * target_error))
    if not result["meets_target"]:
        print("  NO SETTING MEETS THE TARGET; reporting the most accurate")
    print("  bits %d  block %d  ->  %.3f bits/weight at %.3f%% error" % (
        choice["bits"], choice["block"], choice["bits_per_weight"],
        100.0 * choice["error"]))
    print("  zero point %s" % (
        "worth its 8/block bits at this gamma"
        if result["asymmetric_recommended"] else
        "not worth its 8/block bits at this gamma"))
    print()
    print("  NOTE: weight-space error is not token-level fidelity. Confirm with an")
    print("  acceptance-rate measurement against the unquantised model before shipping.")


def self_test() -> int:
    generator = np.random.default_rng(0)
    failures = 0

    def check(label: str, condition: bool) -> None:
        nonlocal failures
        print("  %-56s %s" % (label, "PASS" if condition else "FAIL"))
        if not condition:
            failures += 1

    gaussian = codec.f32_to_bf16(generator.standard_normal(128 * 512).astype(np.float32))
    result = recommend(gaussian, DEFAULT_TARGET_ERROR)
    check("gaussian kurtosis is near 3", abs(result["statistics"]["kurtosis"] - 3.0) < 0.4)
    check("gaussian meets a 1 percent target", result["meets_target"])
    check("error falls as bits rise", all(
        measure(gaussian, bits, 128)["error"] > measure(gaussian, bits + 1, 128)["error"]
        for bits in (6, 7)))
    check("error falls as the block shrinks", (
        measure(gaussian, 6, 256)["error"] > measure(gaussian, 6, 32)["error"]))
    heavy = generator.standard_t(3, 128 * 512) / 1.73
    heavy_source = codec.f32_to_bf16(heavy.astype(np.float32))
    heavy_statistics = distribution_statistics(heavy, 128)
    check("heavy tails raise the measured kurtosis",
          heavy_statistics["kurtosis"] > result["statistics"]["kurtosis"])
    check("a smaller block helps heavy tails more", (
        measure(heavy_source, 8, 256)["error"] - measure(heavy_source, 8, 32)["error"] >
        measure(gaussian, 8, 256)["error"] - measure(gaussian, 8, 32)["error"]))
    # Crossover is near block 64: at block 16 a zero point costs 3.01 dB and a
    # centred source only suffers 0.75 dB, so it does not pay; at block 128 it
    # costs 0.38 dB and does.
    check("zero point is rejected at a small block",
          not zero_point_is_worth_it(1.19, 16))
    check("zero point is accepted at a large block",
          zero_point_is_worth_it(1.19, 128))
    check("a heavier gamma only widens the case",
          zero_point_is_worth_it(1.57, 32))
    print("\n%d failures" % failures)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="?", help="raw bf16 weight sample")
    parser.add_argument("--target-error", type=float, default=DEFAULT_TARGET_ERROR)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    if not arguments.weights:
        parser.error("provide a weight sample or --self-test")
    report(recommend(load_bf16(Path(arguments.weights)), arguments.target_error),
           arguments.target_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
