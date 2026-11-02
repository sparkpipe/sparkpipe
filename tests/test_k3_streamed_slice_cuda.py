"""Run the streamed K3 slice (layers 0-1: KDA with dense then routed MLP, AttnRes and the residual bank, all sixteen shards per layer on one device) against sixteen TP16 ranks running K3LaunchSlice with emulated collectives, then the same slice on four emulated nodes over two passes against it run as one node, on an sm_121 device."""
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
    print(f"SKIP test_k3_streamed_slice_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_k3_streamed_slice_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="k3-streamed-slice-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include", "-Imodel-families/k3/include",
                    "tests/cuda/k3_streamed_slice_cuda.cu", "-o", binary, "-lpthread", "-lcuda"], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
