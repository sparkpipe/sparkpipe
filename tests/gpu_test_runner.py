import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COMPUTE_CAPABILITY = "12.1"
NVCC = os.environ.get("NVCC", "nvcc")


def device_capabilities():
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def has_device(name):
    capabilities = device_capabilities()
    if REQUIRED_COMPUTE_CAPABILITY in capabilities:
        return True
    print(f"SKIP {name} (needs an sm_121 device; this host reports {capabilities or 'none'})")
    return False


def built_binary(name, binary):
    if not has_device(name):
        return None
    target = str(Path(binary).relative_to(ROOT))
    built = subprocess.run(["make", "-s", target], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0 or not Path(binary).is_file():
        print(built.stdout + built.stderr, end="")
        raise SystemExit(f"FAIL {name}: an sm_121 device is present but {target} did not build (is nvcc on PATH?)")
    return Path(binary)


def run_cuda_source(name, source, timeout=900):
    if not has_device(name):
        return 0
    if shutil.which(NVCC) is None:
        raise SystemExit(f"FAIL {name}: an sm_121 device is present but nvcc is unavailable")
    with tempfile.TemporaryDirectory(prefix=name + "-") as directory:
        binary = str(Path(directory) / "probe")
        subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                        "-I.", "-Iinclude", "-Imodel-families/common/include", source, "-o", binary], cwd=ROOT, check=True)
        subprocess.run([binary], check=True, timeout=timeout)
    return 0
