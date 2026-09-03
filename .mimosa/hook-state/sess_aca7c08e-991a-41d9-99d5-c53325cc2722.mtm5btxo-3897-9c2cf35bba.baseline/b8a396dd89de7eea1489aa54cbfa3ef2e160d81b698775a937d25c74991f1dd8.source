#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODECS = ("int6", "int7", "int8", "fp8", "nvfp4", "mxfp4")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label} contains forbidden {needle!r}")


def main() -> int:
    cuda_source = ROOT / "modules/glm52_resident_decode_stage/source/cuda"
    layer = (cuda_source / "layer.cuh").read_text()
    unity = (cuda_source / "unity.cu").read_text()
    codec = (ROOT / "inference/kernels/weight_codec.cuh").read_text()
    packer = (ROOT / "tools/glm52_stagepack.py").read_text()
    targets = json.loads(
        (ROOT / "model_contracts/must_work_targets.json").read_text()
    )

    require(layer, "template<uint32_t ExpertCodec>",
            "GLM compile-time expert path")
    require(layer, "typename LmWeightCodec<ExpertCodec>::Format",
            "GLM compile-time expert format")
    assert layer.count("LmWeightCodecScaleTensor<ExpertCodec>(") == 2
    require(layer, "LmGemmWeightOnlyIndirectLaunch<",
            "GLM direct routed W1 activation path")
    require(layer, "gemm.source_row_map = buffers->route_source_token;",
            "GLM routed row map")
    require(layer, "LmGemmWeightOnlyLaunch<",
            "GLM packed W2 path")
    forbid(layer, "LmGatherRowsKernel", "GLM routed expert path")
    forbid(layer, "LmQuantiseRowsKernel", "GLM BF16 activation path")
    require(layer, "LmGemmLaunch<\n        LmBf16Format,",
            "GLM BF16 router and nonexpert path")

    require(unity, "Glm52ExpertWeightCodec(void)",
            "GLM module codec identity")
    require(unity, "Glm52GemmExpertWeightBf16Activation(",
            "GLM generic expert GEMM export")
    require(unity, "Glm52LayerMoeExpertWeightBf16Activation(",
            "GLM generic expert layer export")
    require(unity, "Glm52LayerMoe<GLM52_EXPERT_WEIGHT_CODEC>(",
            "GLM AOT layer specialization")
    for stale in ("GemmFp8Expert", "LayerMoeFp8", "LayerMoeInt8"):
        forbid(unity, stale, "GLM public CUDA surface")

    public_ids = {
        "int6": "SPARK_WEIGHT_CODEC_INT6",
        "int7": "SPARK_WEIGHT_CODEC_INT7",
        "int8": "SPARK_WEIGHT_CODEC_INT8",
        "fp8": "SPARK_WEIGHT_CODEC_FP8_E4M3",
        "nvfp4": "SPARK_WEIGHT_CODEC_NVFP4_E2M1",
        "mxfp4": "SPARK_WEIGHT_CODEC_MXFP4_E2M1",
    }
    for name in CODECS:
        require(codec, f"LM_WEIGHT_CODEC({public_ids[name]},",
                f"generic {name} codec specialization")
        require(packer, f'"{name}": (', f"{name} stage-pack codec")

    glm_target = next(
        target for target in targets["targets"]
        if target["model_family"] == "glm52"
    )
    assert glm_target["routed_expert_weight_formats"] == [
        "int6_block_f32",
        "int7_block_f32",
        "int8_block_f32",
        "fp8_e4m3_block_f32",
        "nvfp4_e2m1_ue4m3_global_f32",
        "mxfp4_e2m1_e8m0",
    ]
    assert glm_target["routed_expert_activation_format"] == "bf16"
    assert glm_target["non_expert_weight_format"] == "bf16"
    assert glm_target["non_expert_activation_format"] == "bf16"
    assert glm_target["accumulator_format"] == "fp32"
    print("PASS GLM BF16-rest compile-time expert codec matrix contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
