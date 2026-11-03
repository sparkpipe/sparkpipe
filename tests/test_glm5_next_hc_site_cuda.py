"""Run the GLM 5 Next HC site kernel on an sm_121 device with eight rows per cluster and with one row per cluster,
and check that every output (mixes, gates, comb, collapsed and snapshot rows) is the same bits at every
row count, including partial clusters."""
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
    print(f"SKIP test_glm5_next_hc_site_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_glm5_next_hc_site_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="glm5-next-hc-site-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Isrc", "-Imodel-families/glm5_next/source",
                    "-Imodel-families/glm5_next/include", "-Imodel-families/common/include",
                    "tests/cuda/glm5_next_hc_site_cuda.cu", "-o", binary, "-lcuda"], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
