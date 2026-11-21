"""Run the index scorer that finds a block's key slots once against the per-row scorer on an sm_121 device; every score must match bit for bit."""
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
    print(f"SKIP test_index_score_slots_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_index_score_slots_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="index-score-slots-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include",
                    "tests/cuda/index_score_slots_cuda.cu", "-o", binary], cwd=ROOT, check=True)
    result = subprocess.run([binary], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    raise SystemExit(0 if result.returncode == 0 and "PASS index score block slots" in result.stdout else 1)
