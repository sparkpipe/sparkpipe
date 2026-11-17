#!/usr/bin/env python3
import re
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parent.parent


def main():
    gemm = (ROOT / "inference/kernels/gemm.cuh").read_text()
    launch = (ROOT / "runtime/gemm.cuh").read_text()
    stream = (ROOT / "inference/kernels/stream_gemm.cuh").read_text()
    models = {
        "kimi_k3": "model-families/k3/source/layer.cuh",
    }
    failures = []
    if "void *output_f32;" not in gemm:
        failures.append("GEMM has no FP32 output")
    if "(args->output_bf16 == 0) == (args->output_f32 == 0)" not in launch:
        failures.append("GEMM output selection is not fail-closed")
    if "(args->output_bf16 == 0) == (args->output_f32 == 0)" not in stream:
        failures.append("stream GEMM output selection is not fail-closed")
    arg = r"\s*[\w>-]+\s*,"
    f32_router = {
        "kimi_k3": (r"K3SkinnyRows\(\s*b->router_weight\s*,\s*b->normed_bf16\s*,\s*\(uint16_t \*\)0\s*,\s*b->router_logits\s*,",
                    r"K3SkinnyRows\(\s*slice\s*,\s*b->normed_bf16\s*,\s*\(uint16_t \*\)0\s*,\s*K3_ROUTER_SLICE\(b,rows\)\s*,",
                    r"LmStreamGemmDense<LmBf16Format>\(\s*slice\s*,\s*LmScaleTensorNone\(\)\s*,\s*b->normed_bf16\s*,\s*\(uint16_t \*\)0\s*,"
                    r"\s*K3_ROUTER_SLICE\(b,rows\)\s*,",
                    r"#define K3_ROUTER_SLICE\(b,rows\)\s*\\\s*\(\(float \*\)"),
    }
    bf16_router = {
        "kimi_k3": (r"K3SkinnyRows\([^;]*?,[^;]*?,\s*(?:\([\w\s*]+\)\s*)?(?:\w+->router_logits|K3_ROUTER_SLICE\(b,rows\))\s*,\s*\(float \*\)0",),
    }
    for model,path in models.items():
        source = (ROOT / path).read_text()
        required = f32_router.get(model, (r"gemm\.output_f32\s*=\s*\w+->router_logits\s*;",))
        if not all(re.search(pattern, source) for pattern in required):
            failures.append(f"{model} router is not FP32")
        refused = bf16_router.get(model, ()) + (r"output_bf16\s*=\s*\w+->router_logits\s*;",)
        if any(re.search(pattern, source) for pattern in refused):
            failures.append(f"{model} router still writes BF16")
    kimi = (ROOT / "model-families/k3/source/layer.cuh").read_text()
    sigmoid_topk = re.search(
        r"LmTopkRouteLaunch<\s*K3_LAYER_THREADS\s*,\s*K3_TOP_K\s*,"
        r"\s*true\s*,\s*LM_TOPK_SCORE_SIGMOID\s*>\s*\(\s*rows\s*,"
        r"\s*b->router_logits\s*,",
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
