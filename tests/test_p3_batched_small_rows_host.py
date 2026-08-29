#!/usr/bin/env python3
"""Bit-exactness oracle for the small-batch batched linear kernel.

PERF_PROGRAM P3: the knee-sweep's small-B "r-law" is a dispatch issue. The
shared gate SparkLmHostLaunchBatchedLinear routed every row_count below the
M16 tile to the scalar GEMV whose grid gives each row its own weight-strip
pass, so B2/B4 decode inherited the B1 rate class exactly (the measured
27B FP8 B1==B2==8.31 flat spot). The fix lands SparkLmBatchedLinearKernel -
rows 2..15 stream the weights ONCE - and this oracle proves the new kernel:

  1. bit-exact vs SparkLmLinearKernel (the B1 kernel) per row, on the same
     weights, for all five weight formats the gate dispatches, at B=1..15,
     including odd-K tails, multi-chunk K, partial CTAs, and the head
     shadow's 32-warp geometry;
  2. the dispatch contract: row 1 keeps the scalar GEMV, rows 2..15 route to
     the batched kernel, in the shared gate AND in the head shadow branch
     (source assertions - the launcher bodies are device-only syntax);
  3. host-compilability of the shared kernel header itself: this test cannot
     build unless spark_lm_kernels.cuh compiles under g++.
"""
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "spark_lm_batched_host.cu"
BINARY = Path("/tmp") / "lm_spark_batched_host"
KERNELS = ROOT / "model-families" / "common" / "include" / "sparkpipe" / "spark_lm_kernels.cuh"


def build():
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O2", "-ffp-contract=off",
         f"-I{ROOT}/tests/host_cuda", f"-I{ROOT}",
         f"-I{ROOT}/include",
         f"-I{ROOT}/model-families/common/include",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [result.stderr])[0][:300])
        return False
    return True


def assert_dispatch_contract():
    """The gate must route rows 2..15 to the batched kernel and keep B1 on
    the scalar GEMV; the head shadow branch must do the same."""
    source = KERNELS.read_text()

    def flat(text):
        return " ".join(text.split())

    gate_position = source.index("SparkLmHostLaunchBatchedLinear(cudaStream_t")
    gate = source[gate_position:]
    gate = flat(gate[:gate.index("\ntemplate ")])

    checks = [
        ("batched branch guards rows>1",
         "row_count > 1u" in gate),
        ("batched branch launches the batched kernel",
         "SparkLmBatchedLinearKernel<GROUP_SIZE, ACTIVATION_CODEC,SPARK_LM_TILE, SPARK_LM_CTA_WARPS><<<batched_grid," in gate),
        ("B=1 keeps the scalar GEMV",
         "SparkLmLinearKernel<GROUP_SIZE,ACTIVATION_CODEC, SPARK_LM_CTA_WARPS><<<scalar_grid," in gate),
        ("B=1 scalar grid is still row-indexed",
         "dim3 scalar_grid(row_count," in gate),
    ]
    head_position = source.index("static inline cudaError_t SparkLmHostLaunchHeadScreenedArgmaxWithScore(")
    head = flat(source[head_position:gate_position])
    checks.append(("head shadow branch launches the batched kernel",
                   "SparkLmBatchedLinearKernel<SPARK_LM_HEAD_SHADOW_GROUP, SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_TILE, SPARK_LM_SCALAR_CTA_WARPS><<<" in head))
    checks.append(("head shadow rows stay under the tile gate",
                   "row_count < SPARK_LM_TILE" in head))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    return not failed


def main():
    if not assert_dispatch_contract():
        return 1
    if not build():
        return 1
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        print("FAIL oracle run exited", result.returncode)
        return 1
    passed = result.stdout.count("PASS ")
    print(f"OK spark_lm_batched_host: {passed} PASS lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
