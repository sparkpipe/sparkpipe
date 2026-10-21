#!/usr/bin/env python3
"""GB10 bitwise A/B of the DSV4.1 expert-sliced MoE against the per-rank-expert MoE.

Builds build/test_dsv41_flash_moe_slices and runs it on an sm_121 device: 16 emulated
ranks, the same random MXFP4 experts in both layouts, steps of 1, 3, 16, 17 and 64 rows.
Every output must be the same bits as the per-rank finalize plus the rank-ordered sum.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_flash_moe_slices"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_dsv41_flash_moe_slices (nvcc unavailable on this host)")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIP test_dsv41_flash_moe_slices (no GPU on this host)")
        return 0
    capabilities = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()
    if "12.1" not in capabilities:
        print(f"SKIP test_dsv41_flash_moe_slices (needs an sm_121 device; this host reports {capabilities or 'none'})")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 expert slices" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
