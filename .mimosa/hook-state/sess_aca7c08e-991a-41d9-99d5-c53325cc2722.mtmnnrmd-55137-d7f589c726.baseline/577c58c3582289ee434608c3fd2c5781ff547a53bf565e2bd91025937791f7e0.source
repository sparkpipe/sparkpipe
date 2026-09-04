#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    layer = (repository / "inference" / "llms" / "qwen_3_6" / "layer.cuh").read_text(
        encoding="utf-8"
    )
    bind = (repository / "inference" / "llms" / "qwen_3_6" / "bind.cu").read_text(
        encoding="utf-8"
    )
    targets = json.loads(
        (repository / "model_contracts" / "must_work_targets.json").read_text(
            encoding="utf-8"
        )
    )

    if "Qwen38_27bLaunchSlice<LmBf16Format>" not in bind:
        raise AssertionError("Qwen 3.6 shipping entry point is not pinned to BF16")
    for forbidden in (
        "Qwen38_27bLaunchSlice<LmFp8",
        "Qwen38_27bLaunchSlice<LmMxfp4",
        "Qwen38_27bLaunchSlice<LmNvfp4",
    ):
        if forbidden in bind:
            raise AssertionError(f"Qwen 3.6 shipping entry contains {forbidden}")

    if "if constexpr (Format::kScaleGroup == 0u)" not in layer:
        raise AssertionError("Qwen 3.6 BF16 path does not bypass activation quantization")
    if "*scale_out = LmScaleTensorNone();" not in layer:
        raise AssertionError("Qwen 3.6 BF16 activation scale is not explicitly unscaled")
    if "return LmScaleTensorNone();" not in layer:
        raise AssertionError("Qwen 3.6 BF16 weight scale is not explicitly unscaled")

    qwen_targets = [
        entry for entry in targets["targets"]
        if entry.get("model_family") == "qwen38_27b"
    ]
    if len(qwen_targets) != 1:
        raise AssertionError("must-work manifest must contain exactly one Qwen 3.6 target")
    target = qwen_targets[0]
    expected = {
        "routed_expert_weight_format": "none",
        "routed_expert_activation_format": "none",
        "non_expert_weight_format": "bf16",
        "non_expert_activation_format": "bf16",
        "accumulator_format": "fp32",
    }
    for key, value in expected.items():
        if target.get(key) != value:
            raise AssertionError(
                f"Qwen 3.6 must-work contract {key}={target.get(key)!r}, "
                f"expected {value!r}"
            )

    print("PASS Qwen 3.6 27B BF16 source contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
