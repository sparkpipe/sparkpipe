#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "k3_head_rank_host.cu"


def main():
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "lm_k3_head_rank_host"
        build = subprocess.run(
            [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}/tests/host_cuda/shim",
             f"-I{ROOT}", f"-I{ROOT}/tests/host_cuda",
             f"-I{ROOT}/model-families/common/include", f"-I{ROOT}/include",
             "-x", "c++", str(SOURCE), "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            errors = [l for l in build.stderr.split("\n") if "error" in l]
            print("FAIL host build:", (errors or [build.stderr])[0][:200])
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    print(run.stdout[-1200:])
    if run.returncode != 0 or "global token id" not in run.stdout:
        print("FAIL a vocabulary-sliced multi-row head returned rank-local token ids")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
