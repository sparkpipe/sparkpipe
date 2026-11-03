"""Run the streamed K3 MLA layer (sixteen six-head groups with expanded keys on one device) against sixteen TP16 ranks running latent attention with absorbed queries on an sm_121 device."""
import os
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NVCC = os.environ.get("NVCC", "nvcc")
REQUIRED_COMPUTE_CAPABILITY = "12.1"


def device_capabilities():
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


capabilities = device_capabilities()
if REQUIRED_COMPUTE_CAPABILITY not in capabilities:
    print(f"SKIP test_k3_streamed_mla_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_k3_streamed_mla_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="k3-streamed-mla-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include", "-Imodel-families/k3/include",
                    "tests/cuda/k3_streamed_mla_cuda.cu", "-o", binary, "-lpthread", "-lcuda"], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
