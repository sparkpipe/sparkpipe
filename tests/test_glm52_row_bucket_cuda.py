"""Run the glm52 graph row bucket on an sm_121 device: padded rows must leave every real row of the dense bf16 linear bit-identical."""
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
    print(f"SKIP test_glm52_row_bucket_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_glm52_row_bucket_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="glm52-row-bucket-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Imodel-families/common/include", "-Imodel-families/glm52/include",
                    "-DGLM_EXPERT_WEIGHT_CODEC=5", "-include", "model-families/glm52/include/sparkpipe/spark_glm52_model.h",
                    "tests/cuda/glm52_row_bucket_cuda.cu", "-o", binary, "-lcuda"], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
