#!/usr/bin/env python3
"""Frame-error receipts through the actual kernel bodies (host oracle).

Injects corrupt route maps and a wild sparse position through the real
shared-kernel bodies and asserts the fail-frame contract: the per-frame
error record carries the first failure's code and fields, the kernel
returns a bounded result, and no output is manufactured over a failed
access. The device half of the receipt - the same injections under a live
CUDA context proving the context survives - is
tools/hardware/spark_frame_error_probe.cu, validated on spark5.
"""

import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "frame_error_host.cu"
BINARY = Path("/tmp") / "lm_frame_error_host"


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
    if "PASS (0 failures)" not in run.stdout:
        print("FAIL missing success receipt")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
