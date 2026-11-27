#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/test_dsv41_prefill_opt.cu"


def detect_sm121():
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    probe = subprocess.run([smi, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    if probe.returncode != 0:
        return False
    return any(line.strip().startswith("12.") for line in probe.stdout.splitlines())


def main():
    nvcc = shutil.which("nvcc")
    if nvcc is None or not detect_sm121():
        print("SKIP test_dsv41_prefill_opt (no sm_12x CUDA toolchain on this host)")
        return 0
    emb = subprocess.run(["make", "-s", "build/skl2_dsv41_embedded.h"], cwd=ROOT, capture_output=True, text=True)
    if emb.returncode != 0:
        print(emb.stdout + emb.stderr, end="")
        return 1
    with tempfile.TemporaryDirectory(prefix="dsv41_prefill_opt_") as tmp:
        binary = Path(tmp) / "test_dsv41_prefill_opt"
        cmd = [
            nvcc, "-O3", "-std=c++17",
            "-gencode", "arch=compute_121a,code=sm_121a",
            f"-I{ROOT}", f"-I{ROOT / 'build'}", f"-I{ROOT / 'include'}", f"-I{ROOT / 'model-families/dsv41_flash/include'}",
            f"-I{ROOT / 'model-families/dsv41_flash/source'}",
            str(SOURCE), "-lcuda", "-o", str(binary),
        ]
        comp = subprocess.run(cmd, capture_output=True, text=True)
        if comp.returncode != 0:
            print(comp.stdout + comp.stderr, end="")
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        print(run.stdout + run.stderr, end="")
        return 0 if run.returncode == 0 and "PASS dsv41 prefill opt parity" in run.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
