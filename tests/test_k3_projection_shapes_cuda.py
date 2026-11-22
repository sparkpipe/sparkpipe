#!/usr/bin/env python3
"""K3's load check: every projection slice of the production TP16 geometry (K3 and K3.1 Pro) fits the kernel its shape selects,
routed_up's 224-wide slice takes the tiled GEMM for prefill by shape while inputs that are multiples of 64 take the stream GEMM, and
a slice no kernel can run refuses at load through SPARK_FAIL with a message naming the projection and the fix."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import NVCC, ROOT, has_device  # noqa: E402

NAME = "test_k3_projection_shapes_cuda"


def main():
    if not has_device(NAME):
        return 0
    if shutil.which(NVCC) is None:
        raise SystemExit(f"FAIL {NAME}: an sm_121 device is present but nvcc is unavailable")
    with tempfile.TemporaryDirectory(prefix=NAME + "-") as directory:
        binary = str(Path(directory) / "probe")
        subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "-I.", "-Iinclude",
                        "-Imodel-families/common/include", "-Imodel-families/k3/include", "tests/cuda/k3_projection_shapes_cuda.cu", "-lcuda",
                        "-o", binary], cwd=ROOT, check=True)
        subprocess.run([binary], check=True, timeout=300)
    return 0


if __name__ == "__main__":
    sys.exit(main())
