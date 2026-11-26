"""Verify 0-bit parity and benchmark shared Blackwell prefill MXFP4 block-scaled MoE (cell_mx), BF16 TMA GEMM (gemm), and expanded latent prefill attention (attn_expanded_prefill) kernels across frontier shapes and M=64..2048 on an sm_121 device."""
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
    print(f"SKIP test_shared_prefill_gemm (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_shared_prefill_gemm: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="shared-prefill-gemm-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include",
                    "tests/cuda/shared_prefill_gemm_bench.cu", "-lcuda", "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
