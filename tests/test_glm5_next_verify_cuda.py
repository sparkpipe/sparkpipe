"""Run the GLM 5 Next stage (GLM 5.3 Flash geometry, one TP16 rank with synthetic weights and replicated peers) on an
sm_121 device: verify frames of 1, 2, 7, 16 and the stage's verify-row count must equal the same rows decoded one by
one, bit for bit, in every row's hidden output, in the KDA recurrent state that keep_rows leaves for every kept count,
and in the latent and index cache rows; a rejected draft tail must leave the next decode steps, the recurrent state and
the cache rows equal to a run that never saw the drafts."""
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
    print(f"SKIP test_glm5_next_verify_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_glm5_next_verify_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="glm5-next-verify-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "-Isrc", "-Imodel-families/glm5_next/source", "-Imodel-families/glm5_next/include",
                    "-Imodel-families/common/include", "-Imodules/glm53_flash_tp16_resident_decode_stage/include",
                    "tests/cuda/glm5_next_verify_cuda.cu", "-o", binary, "-lcuda"], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
