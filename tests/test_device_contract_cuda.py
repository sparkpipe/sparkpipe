#!/usr/bin/env python3
"""Run the backend-neutral device contract against the CUDA backend.

Builds build/runtime_libs/libsparkpipe_device_cuda and build/test_device_contract_cuda
where the CUDA runtime headers exist, and runs the contract on a CUDA device: the same
suite the host and Metal backends pass, unchanged.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_device_contract_cuda"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_device_contract_cuda (no CUDA runtime headers on this host)")
        return 0
    if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0:
        print("SKIP test_device_contract_cuda (no CUDA device on this host)")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=600)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS device contract (cuda backend)" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
