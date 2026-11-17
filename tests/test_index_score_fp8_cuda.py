"""Run the FP8 index on an sm_121 device: keys, scales, queries and folded weights must equal the publisher's formulas bit for bit, scores must stay within FP32 rounding of FP64, and the run prints FP8 and BF16 times."""
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
    print(f"SKIP test_index_score_fp8_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_index_score_fp8_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="index-score-fp8-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include",
                    "tests/cuda/index_score_fp8_cuda.cu", "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
