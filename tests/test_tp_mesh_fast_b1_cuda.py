"""Verify bitwise-exactness and B1 latency wins of fast TP16 mesh collective kernels on sm_121."""
import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
NVCC = os.environ.get("NVCC", "nvcc")
REQUIRED_COMPUTE_CAPABILITY = "12.1"


def device_capabilities():
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


capabilities = device_capabilities()
if REQUIRED_COMPUTE_CAPABILITY not in capabilities:
    print(f"SKIP test_tp_mesh_fast_b1_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_tp_mesh_fast_b1_cuda: an sm_121 device is present but nvcc is unavailable")

for target in ("tp_mesh_fast_b1_cuda", "tp_mesh_prefill_collective_cuda"):
    binary = ROOT / "build" / target
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            NVCC,
            "-std=c++17",
            "-O3",
            "-arch=sm_121a",
            "-I" + str(ROOT),
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "model-families" / "common" / "include"),
            str(ROOT / "tests" / "cuda" / f"{target}.cu"),
            "-lpthread",
            "-o",
            str(binary),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([str(binary)], check=True, timeout=600)
