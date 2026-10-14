#!/usr/bin/env python3
"""Plan weight prefetch groups past the plan capacity on the host shim and check a dropped kernel adds nothing to the previous group."""

import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "weight_prefetch_host.cu"
BINARY = Path("/tmp") / "lm_weight_prefetch_host"


def main() -> int:
    build = subprocess.run(
        [
            host_cuda_cxx(),
            "-std=c++17",
            "-O0",
            f"-I{ROOT}",
            f"-I{ROOT / 'tests' / 'host_cuda'}",
            "-x",
            "c++",
            str(SOURCE),
            "-o",
            str(BINARY),
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print("FAIL host build:", build.stderr.strip()[:1000])
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(run.stdout, end="")
    if run.returncode != 0:
        print(run.stderr, end="")
        return run.returncode
    if "(0 failures)" not in run.stdout:
        print("FAIL missing success receipt")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
