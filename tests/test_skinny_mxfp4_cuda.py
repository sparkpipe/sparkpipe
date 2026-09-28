"""Build and run the MXFP4 skinny expert kernels on an sm_121a device."""
import os
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NVCC = os.environ.get("NVCC", "nvcc")
if shutil.which(NVCC) is None:
    print("SKIP test_skinny_mxfp4_cuda (nvcc unavailable on this host; spark-gated)")
    raise SystemExit(0)
with tempfile.TemporaryDirectory(prefix="skinny-mxfp4-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include",
                    "tests/cuda/skinny_mxfp4_cuda.cu", "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=600)
