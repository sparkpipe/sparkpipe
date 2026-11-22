#!/usr/bin/env python3
"""A step program recorded on a CUDA stream runs as one graph, patched per run, with the bytes eager launches write."""
import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/device_program_cuda"


def main():
    if gpu_test_runner.built_binary("test_device_program_cuda", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=600)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS device_program_cuda" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
