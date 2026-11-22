#!/usr/bin/env python3
"""Run the device shared-memory test against the CUDA backend.

Builds build/test_device_shared_memory_cuda where the CUDA device library exists and
runs it on a CUDA device: the same cases the host backend passes, with the export,
import and copies in two processes.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_device_shared_memory_cuda"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_device_shared_memory_cuda (no CUDA runtime headers on this host)")
        return 0
    if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0:
        print("SKIP test_device_shared_memory_cuda (no CUDA device on this host)")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=600)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS device shared memory (cuda backend)" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
