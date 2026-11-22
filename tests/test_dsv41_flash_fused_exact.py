#!/usr/bin/env python3
"""GB10 bitwise check of the generated DeepSeek V4.1 Flash fused kernels against the kernels they replace.

Builds build/test_dsv41_flash_fused_exact (generated from model-families/dsv41_flash/fusion/decode.json by
tools/fused_kernels.py) and runs it on an sm_121 device. For 1 to 16 rows, with ordinary and edge-valued inputs,
every output of each fused kernel must be the same bytes as the unfused launches write: for the hyper-connection
block the mix partials, pre, post and comb, the collapsed and normed rows and their FP8 codes; for the attention
output block each group's quantized input and the window ring. A row's outputs must not depend on how many rows
run, a changed input must change them, and a CUDA graph replaying a fused kernel three times must give the same
bytes, so its arrival counters return to zero.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_flash_fused_exact"


def main():
    if gpu_test_runner.built_binary("test_dsv41_flash_fused_exact", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS fused kernels" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
