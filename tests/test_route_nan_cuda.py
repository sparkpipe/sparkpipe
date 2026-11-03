"""Run every routing top-k kernel the stages use (the one- and eight-warp route launches, the grouped small kernel and the
group-limited kernel) on an sm_121 device with NaN, negative-NaN and infinite router rows: every expert index stays in range
and distinct, NaN ranks below every finite score, a row with fewer finite scores than picks raises the fault word the runner
reports, and the route build over the guarded rows gives in-range packed rows."""
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
    print(f"SKIP test_route_nan_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_route_nan_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="route-nan-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Isrc", "-Imodel-families/common/include",
                    "tests/cuda/route_nan_cuda.cu", "-o", binary, "-lcuda"], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
