"""Drive K3's routed-latent slice round on an sm_121 device against sixteen simulated ranks and check the column reduce-scatter and the exact norm statistics match the all-reduce path."""
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
    print(f"SKIP test_k3_latent_slice_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_k3_latent_slice_cuda: an sm_121 device is present but nvcc is unavailable")
subprocess.run(["make", "-s", "build/k3_latent_slice_cuda"], cwd=ROOT, check=True)
subprocess.run([str(ROOT / "build" / "k3_latent_slice_cuda")], check=True, timeout=600)
