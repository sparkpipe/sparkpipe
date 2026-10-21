#!/usr/bin/env python3
"""GB10 bitwise check of the DSV4.1 window fold against the sequential fold it replaced.

Builds build/test_dsv41_window_fold and runs it on an sm_121 device. The fold scores a batch of
window keys before updating the online softmax; every record it writes must be the same bits as
the key-by-key fold, for decode rows, verify frames, prefill waves, local-head landings, early
positions and partial records. Perturbing one key must change the result.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_window_fold"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_dsv41_window_fold (nvcc unavailable on this host)")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIP test_dsv41_window_fold (no GPU on this host)")
        return 0
    capabilities = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()
    if "12.1" not in capabilities:
        print(f"SKIP test_dsv41_window_fold (needs an sm_121 device; this host reports {capabilities or 'none'})")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 window fold" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
