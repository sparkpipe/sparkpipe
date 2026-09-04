#!/usr/bin/env python3
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    gemm = (ROOT / "inference/kernels/gemm.cuh").read_text()
    launch = (ROOT / "runtime/gemm.cuh").read_text()
    models = {
        "glm52": (
            "modules/glm52_resident_decode_stage/source/cuda/layer.cuh"
        ),
        "kimi_k3": "inference/llms/kimi_k3/layer.cuh",
        "mimo_2_5": "inference/llms/mimo_2_5/layer.cuh",
    }
    failures = []
    if "void *output_f32;" not in gemm:
        failures.append("GEMM has no FP32 output")
    if "(args->output_bf16 == 0) == (args->output_f32 == 0)" not in launch:
        failures.append("GEMM output selection is not fail-closed")
    for model,path in models.items():
        source = (ROOT / path).read_text()
        if not re.search(r"gemm\.output_f32\s*=\s*\w+->router_logits\s*;", source):
            failures.append(f"{model} router is not FP32")
        if re.search(r"output_bf16\s*=\s*\w+->router_logits\s*;", source):
            failures.append(f"{model} router still writes BF16")
    dsv4 = (
        ROOT
        / "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_cuda.cu"
    ).read_text()
    if not re.search(
        r"SparkDsv4GateScoresKernel\([^)]*float \*scores_f32",dsv4,re.S
    ):
        failures.append("dsv4 router is not FP32")
    if "void *scores_bf16" in dsv4:
        failures.append("dsv4 router still exposes a BF16 score output")
    kimi = (ROOT / "inference/llms/kimi_k3/layer.cuh").read_text()
    sigmoid_topk = re.search(
        r"LmTopkSmallKernel<\s*K3_LAYER_THREADS\s*,\s*K3_TOP_K\s*,"
        r"\s*true\s*,\s*1u\s*,\s*1u\s*,"
        r"\s*(?:true|LM_TOPK_SCORE_SIGMOID)\s*>",
        kimi,
        re.S,
    )
    if sigmoid_topk is None:
        failures.append("K3 FP32 logits do not take sigmoid in top-k")
    if failures:
        print("\n".join(failures))
        return 1
    print("router logits stay FP32 through selection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
