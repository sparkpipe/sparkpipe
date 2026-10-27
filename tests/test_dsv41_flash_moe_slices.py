#!/usr/bin/env python3
"""GB10 bitwise A/B of the DSV4.1 expert-sliced MoE against the per-rank-expert MoE.

Builds build/test_dsv41_flash_moe_slices and runs it on an sm_121 device: 16 emulated
ranks, the same random MXFP4 experts in both layouts, steps of 1, 3, 16, 17 and 64 rows.
Every output must be the same bits as the per-rank finalize plus the rank-ordered sum.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_flash_moe_slices"


def main():
    if gpu_test_runner.built_binary("test_dsv41_flash_moe_slices", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 expert slices" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
