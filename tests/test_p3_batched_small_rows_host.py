#!/usr/bin/env python3
"""Small-row dispatch contract for SparkLmHostLaunchBatchedLinear.

Rows below SPARK_LM_TILE stay on the per-row scalar GEMV: its concurrent row
streams overlap in GB10's memory system and measured ahead of a one-pass
batched kernel at B2..B4, so that kernel was removed. The head shadow branch
keeps its scalar geometry and its rows stay under the tile gate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "model-families" / "common" / "include" / "sparkpipe" / "spark_lm_kernels.cuh"


def flat(text):
    return " ".join(text.split())


def assert_dispatch_contract():
    source = KERNELS.read_text()
    gate_position = source.index("SparkLmHostLaunchBatchedLinear(cudaStream_t")
    gate = source[gate_position:]
    gate = flat(gate[:gate.index("static inline uint32_t SparkLmSm121B1Bf16LinearPairPolicy")])

    checks = [
        ("scalar GEMV grid is row-indexed (per-row streams, B1..B15)",
         "dim3 scalar_grid(row_count," in gate),
        ("scalar route launches SparkLmLinearKernel",
         "SparkLmLinearKernel<GROUP_SIZE,ACTIVATION_CODEC, SPARK_LM_CTA_WARPS><<<scalar_grid," in gate),
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
    return 0 if assert_dispatch_contract() else 1


if __name__ == "__main__":
    sys.exit(main())
