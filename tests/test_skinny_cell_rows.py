#!/usr/bin/env python3
"""GB10 check that skinny cell expert launches are row-invariant.

Builds build/test_skinny_cell_rows and runs it on an sm_121 device. For the DSV4.1 expert gate/up and
down shapes and one shape small enough for the shared-offset kernel, with source-token and packed
activations, every routed row of a 2 to 33 row frame must equal, bit for bit, the one-row launch of
its token, both on experts no other row of the frame uses and on experts the frame shares. Verify
frames are exact only if a row computes the same expert output as one-row decode.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_skinny_cell_rows"


def main():
    if gpu_test_runner.built_binary("test_skinny_cell_rows", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS skinny cell rows" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
