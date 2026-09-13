#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cuda_source = ROOT / "modules/glm52_resident_decode_stage/source/cuda"
    unity = (cuda_source / "unity.cu").read_text()
    api = (cuda_source / "api.h").read_text()
    failures = []

    required_entries = (
        "Glm52GemmBf16",
        "Glm52ExpertWeightCodec",
        "Glm52GemmExpertWeightBf16Activation",
        "Glm52LayerAttentionBf16",
        "Glm52LayerDenseMlpBf16",
        "Glm52LayerMoeExpertWeightBf16Activation",
    )
    for name in required_entries:
        if name not in unity:
            failures.append(f"missing GLM entry point {name}")

    for codec_name in ("Fp8", "Int6", "Int7", "Int8", "Nvfp4", "Mxfp4"):
        for prefix in ("Glm52Gemm", "Glm52LayerMoe"):
            token = prefix + codec_name
            if token in unity or token in api:
                failures.append(f"codec-specific public entry point {token}")

    if "typename LmWeightCodec<GLM52_EXPERT_WEIGHT_CODEC>::Format" not in unity:
        failures.append("GLM unity does not resolve the package codec at compile time")
    if "LmGemmWeightOnlyLaunch<\n        Glm52ExpertWeightFormat," not in unity:
        failures.append("GLM expert API does not use the specialized weight-only launch")
    if "if (!grouped)" not in unity:
        failures.append("GLM expert API does not reject a dense call")
    for name in ("Glm52GemmBf16", "Glm52ExpertWeightCodec",
                 "Glm52GemmExpertWeightBf16Activation"):
        if name not in api:
            failures.append(f"GLM public API does not declare {name}")

    if failures:
        print("\n".join(failures))
        return 1
    print("PASS GLM unity exposes BF16 nonexperts and one AOT expert codec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
