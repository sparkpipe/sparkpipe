#!/usr/bin/env python3
"""Run the backend-neutral device contract against the Metal backend.

Builds build/runtime_libs/libsparkpipe_device_metal and build/test_device_contract_metal
on macOS where the Metal toolchain exists, and runs the contract on the system GPU: the
same suite the host and CUDA backends pass, unchanged.
"""

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_device_contract_metal"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_device_contract_metal (no Metal toolchain on this host)")
        return 0
    if platform.system() != "Darwin":
        print("SKIP test_device_contract_metal (Metal needs macOS)")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=600)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS device contract (metal backend)" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
