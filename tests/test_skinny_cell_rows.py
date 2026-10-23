#!/usr/bin/env python3
"""GB10 check that skinny cell expert launches are row-invariant.

Builds build/test_skinny_cell_rows and runs it on an sm_121 device. For the DSV4.1 expert gate/up and
down shapes and one shape small enough for the shared-offset kernel, with source-token and packed
activations, every routed row of a 2 to 33 row frame must equal, bit for bit, the one-row launch of
its token, both on experts no other row of the frame uses and on experts the frame shares. Verify
frames are exact only if a row computes the same expert output as one-row decode.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_skinny_cell_rows"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_skinny_cell_rows (nvcc unavailable on this host)")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIP test_skinny_cell_rows (no GPU on this host)")
        return 0
    capabilities = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()
    if "12.1" not in capabilities:
        print(f"SKIP test_skinny_cell_rows (needs an sm_121 device; this host reports {capabilities or 'none'})")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS skinny cell rows" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
