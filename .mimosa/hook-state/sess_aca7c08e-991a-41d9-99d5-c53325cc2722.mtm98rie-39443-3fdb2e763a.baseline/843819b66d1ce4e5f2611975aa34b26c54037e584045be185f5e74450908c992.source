#!/usr/bin/env python3
"""Execute required KV access failures through the actual CUDA kernel bodies.

The host shim runs one thread per block, so it cannot emulate the device trap,
but it executes the exact mapping and first-error recording logic. On CUDA the
same failure path records the error and emits PTX `trap`, making the stream
terminal instead of returning an attention result over incomplete history.
"""

import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "kv_failure_host.cu"
BINARY = Path("/tmp") / "lm_kv_failure_host"


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
