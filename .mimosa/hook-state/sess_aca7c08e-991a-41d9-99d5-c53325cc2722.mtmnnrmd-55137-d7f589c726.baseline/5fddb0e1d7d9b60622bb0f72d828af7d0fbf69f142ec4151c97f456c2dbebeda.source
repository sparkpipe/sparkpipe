#!/usr/bin/env python3
"""Bit-exactness oracle for the small-batch batched linear kernel.

PERF_PROGRAM P3 lane. The knee-sweep's "B1==B2==8.31" premise turned out to
be a misread of the sweep CSV (8.31 is the B1 aggregate row; B2 measured
16.63 = 2.00x): the per-row scalar route's concurrent streams overlap in
GB10's memory system, so small-B already scaled. The device bench
(tools/p3_batched_small_rows_bench.cu) confirmed it kernel-level - scalar
B2/B4 aggregate 2.03x/3.36x vs the one-pass batched kernel's 1.53x/2.58x -
so the measured verdict is: keep the scalar route, record the negative, and
this oracle proves the batched kernel is still exactly right:

  1. bit-exact vs SparkLmLinearKernel (the B1 kernel) per row, on the same
     weights, for all five weight formats the gate dispatches, at B=1..15,
     including odd-K tails, multi-chunk K, partial CTAs, and the head
     shadow's 32-warp geometry;
  2. the dispatch contract: the per-row scalar GEMV stays the routed
     small-B path (the device bench measured its overlapped streams AHEAD
     of the one-pass batched kernel at B2..B4 on the 27B FP8 class), the
     measured-negative verdict is recorded at the gate, and the batched
     kernel stays compiled as the bit-exact one-pass alternative (source
     assertions - the launcher bodies are device-only syntax);
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
    """Pin the measured dispatch: rows below the tile stay on the per-row
    scalar GEMV (the 27B-FP8 bench showed its concurrent row streams overlap
    in the memory system and beat the one-pass batched kernel at B2..B4),
    and the batched kernel remains compiled as the proven one-pass
    alternative. The head shadow branch keeps its scalar geometry."""
    source = KERNELS.read_text()

    def flat(text):
        return " ".join(text.split())

    gate_position = source.index("SparkLmHostLaunchBatchedLinear(cudaStream_t")
    gate = source[gate_position:]
    gate = flat(gate[:gate.index("static inline uint32_t SparkLmSm121B1Bf16LinearPairPolicy")])

    checks = [
        ("scalar GEMV grid is row-indexed (per-row streams, B1..B15)",
         "dim3 scalar_grid(row_count," in gate),
        ("scalar route launches SparkLmLinearKernel",
         "SparkLmLinearKernel<GROUP_SIZE,ACTIVATION_CODEC, SPARK_LM_CTA_WARPS><<<scalar_grid," in gate),
        ("the measured-negative verdict is recorded at the gate",
         "MEASURED" in gate and "1.53x" in gate),
        ("batched kernel stays defined for the overlap-hostile case",
         "SparkLmBatchedLinearKernel" in source),
        ("B=1 keeps the scalar GEMV",
         "dim3 scalar_grid(row_count," in gate),
    ]
    head_position = source.index("static inline cudaError_t SparkLmHostLaunchHeadScreenedArgmaxWithScore(")
    head = flat(source[head_position:gate_position])
    checks.append(("head shadow keeps the scalar geometry",
                   "SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_SCALAR_CTA_WARPS><<<" in head))
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
