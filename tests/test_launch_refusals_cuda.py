#!/usr/bin/env python3
"""A shape the designed kernel refuses fails at once, through SPARK_FAIL at the launch site, with a message that names the cause and
the fix: dense small waves, row-exact dense waves, grouped skinny and cell expert waves, and shard prefill attention by heads, row
positions and selection size. Nothing falls back to a slower kernel."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import NVCC, ROOT, has_device  # noqa: E402

NAME = "test_launch_refusals_cuda"


def main():
    if not has_device(NAME):
        return 0
    if shutil.which(NVCC) is None:
        raise SystemExit(f"FAIL {NAME}: an sm_121 device is present but nvcc is unavailable")
    with tempfile.TemporaryDirectory(prefix=NAME + "-") as directory:
        binary = str(Path(directory) / "probe")
        subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "-I.", "-Iinclude",
                        "-Imodel-families/common/include", "tests/cuda/launch_refusals_cuda.cu", "-lcuda", "-o", binary], cwd=ROOT, check=True)
        subprocess.run([binary], check=True, timeout=300)
    return 0


if __name__ == "__main__":
    sys.exit(main())
