"""Run the glm52 latent KV shard (1/16 per rank at TP16) on an sm_121 device against the replicated-storage oracle and the served kernel."""
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
    print(f"SKIP test_glm52_kv_shard_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_glm52_kv_shard_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="glm52-kv-shard-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include", "-Imodel-families/glm52/include",
                    "tests/cuda/glm52_kv_shard_cuda.cu", "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
