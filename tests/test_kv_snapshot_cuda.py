"""Save a prompt's pages and recurrent states to the KV snapshot store on an sm_121 device, restore them into a fresh engine and compare the next-token logits bit for bit."""
import os
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
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
    print(f"SKIP test_kv_snapshot_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(os.environ.get("NVCC", "nvcc")) is None:
    raise SystemExit("FAIL test_kv_snapshot_cuda: an sm_121 device is present but nvcc is unavailable")
subprocess.run(["make", "build/test_kv_snapshot_cuda"], cwd=ROOT, check=True)
with tempfile.TemporaryDirectory(prefix="kv-snapshot-cuda-") as directory:
    subprocess.run([str(ROOT / "build/test_kv_snapshot_cuda"), directory], cwd=ROOT, check=True, timeout=900)
