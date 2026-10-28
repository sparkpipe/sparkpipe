#!/usr/bin/env python3
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_metal_bf16_convert"


def main():
    if platform.system() != "Darwin":
        print("SKIP test_metal_bf16_convert (Metal needs macOS and a Metal 4.0 toolchain)")
        return 0
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print(built.stdout, end="")
        print("SKIP test_metal_bf16_convert (no Metal device or kernel library)")
        return 0
    result = subprocess.run([str(BINARY)], cwd=ROOT, capture_output=True, text=True, timeout=3600)
    print(result.stdout + result.stderr, end="")
    if result.returncode == 0 and result.stdout.startswith("SKIP test_metal_bf16_convert"):
        return 0
    passed = "PASS the shared bf16 conversion" in result.stdout and "PASS the shared FP8 and FP4 decoders" in result.stdout
    return 0 if result.returncode == 0 and passed else 1


if __name__ == "__main__":
    sys.exit(main())
