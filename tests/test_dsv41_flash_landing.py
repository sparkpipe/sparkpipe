#!/usr/bin/env python3
"""GB10 bitwise A/B of DSV4.1 attention over a landed KV copy against the all-to-all records.

Builds build/test_dsv41_flash_landing and runs it on an sm_121 device: 16 emulated ranks, a ratio-2
compressed cache or the full-length ratio-1 cache stored rank-sharded, queries for 64 heads, window
rings per rank. One arm runs
the partial and window-fold kernels per source rank for all heads and exchanges records; the
other packs every rank's keys into a landing and runs the same kernels per source rank for one
rank's own heads. Records and merged outputs must be the same bits, for dense and selected-list
steps of 17 to 256 rows.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_flash_landing"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_dsv41_flash_landing (nvcc unavailable on this host)")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIP test_dsv41_flash_landing (no GPU on this host)")
        return 0
    capabilities = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()
    if "12.1" not in capabilities:
        print(f"SKIP test_dsv41_flash_landing (needs an sm_121 device; this host reports {capabilities or 'none'})")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 landing" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
