"""Run the streamed K3 KDA layer on emulated nodes (one token slice each, conv tails and chunk transforms exchanged per shard, two passes, the last one full or ragged) against the sequential layer on an sm_121 device."""
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
    print(f"SKIP test_k3_streamed_kda_nodes_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_k3_streamed_kda_nodes_cuda: an sm_121 device is present but nvcc is unavailable")
RAGGED_TOKENS = (None, 203, 194, 130)
with tempfile.TemporaryDirectory(prefix="k3-streamed-kda-nodes-cuda-") as directory:
    for tokens in RAGGED_TOKENS:
        binary = str(pathlib.Path(directory) / f"probe-{tokens or 'full'}")
        defines = [] if tokens is None else [f"-DKDA_TOKENS={tokens}u"]
        subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", *defines,
                        "-I.", "-Iinclude", "-Imodel-families/common/include", "-Imodel-families/k3/include",
                        "tests/cuda/k3_streamed_kda_nodes_cuda.cu", "-o", binary, "-lpthread", "-lcuda"], cwd=ROOT, check=True)
        subprocess.run([binary], check=True, timeout=900)
