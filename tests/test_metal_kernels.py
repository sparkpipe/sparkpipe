#!/usr/bin/env python3
"""Run every Metal kernel against its float64 host reference on the system GPU.

Builds build/test_metal_kernels on macOS where the Metal toolchain exists and
runs it: each operation at the shapes the model uses must sit inside its
derived tolerance, and the same operation with one corrupted input must fall
outside it.
"""

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_metal_kernels"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if platform.system() != "Darwin" or not BINARY.is_file():
        print("SKIP test_metal_kernels (Metal needs macOS and its toolchain)")
        return 0
    arguments = [str(BINARY)] + sys.argv[1:]
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=3600)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS metal kernels" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
