#!/usr/bin/env python3
"""GB10 bitwise check of the DSV4.1 window fold against the sequential fold it replaced.

Builds build/test_dsv41_window_fold and runs it on an sm_121 device. The fold scores a batch of
window keys before updating the online softmax; every record it writes must be the same bits as
the key-by-key fold, for decode rows, verify frames, prefill waves, local-head landings, early
positions and partial records. Perturbing one key must change the result.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_window_fold"


def main():
    if gpu_test_runner.built_binary("test_dsv41_window_fold", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 window fold" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
