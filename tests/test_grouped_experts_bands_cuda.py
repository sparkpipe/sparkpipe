#!/usr/bin/env python3
"""GB10 check of the grouped expert GEMV across its route-count variant boundaries.

Builds build/test_grouped_experts_bands_cuda and runs it on an sm_121 device. For every family that reaches the
grouped skinny expert GEMV (Qwen3.8 Max and Flash in NVFP4 and BF16, MiMo 2.6 in MXFP4, HY4 in MXFP8, GLM 5.3 Full
and Flash in FP8 and BF16, Gemma 4, K-EXAONE 2, Laguna, Ling 3, Solar Open 2 and Motif in BF16), with each family's
local experts, top-k and widths, both projections run at the last row count below and the first above each boundary
where the kernel changes: 2, 8 and 16 routes an expert, the last being where LmLocalExpertsUp/Down leave the grouped
skinny GEMV. Every call is timed with its weights out of L2, as in a step. Crossing a boundary must not make a call
more than four times slower; a call may not take longer than its band's variant run on its own; every output of the
grouped GEMV must be the same bytes as a two-row reference variant, since every variant sums each output in the same
order; and no grouped GEMV instantiation in the binary may spill to local memory or keep a stack frame.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_grouped_experts_bands_cuda"
FUNCTION = re.compile(r"\s*Function (\S+):")
RESOURCE = re.compile(r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


def spilling(binary):
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    listing = subprocess.run([cuobjdump, "-res-usage", str(binary)], capture_output=True, text=True, check=True).stdout
    refused, checked, name = [], 0, None
    for line in listing.splitlines():
        match = FUNCTION.match(line)
        if match is not None:
            name = match.group(1)
            continue
        match = RESOURCE.match(line)
        if match is not None and name is not None and "LmSkinnyGroupedKernel" in name:
            checked += 1
            if int(match.group(2)) != 0 or int(match.group(4)) != 0:
                refused.append(f"{name}: stack {match.group(2)} bytes, local {match.group(4)} bytes")
            name = None
    return refused, checked


def main():
    if gpu_test_runner.built_binary("test_grouped_experts_bands_cuda", BINARY) is None:
        return 0
    refused, checked = spilling(BINARY)
    for item in refused:
        print("FAIL a grouped expert GEMV spills: " + item)
    if checked == 0:
        print("FAIL no grouped expert GEMV instantiation found in " + str(BINARY))
        return 1
    if refused:
        return 1
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS grouped expert bands" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
