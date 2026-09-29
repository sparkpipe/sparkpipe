"""Run the head_score kernels and the score-dump writer on an sm_121 device and check the merged dump against numpy.

Two shapes: the 16-shard CPU-shim shape, and a TP1 shape with the Flash head
geometry (hidden 4096, 154880-token vocabulary) on the same hidden rows numpy
scores in float64.
"""
import os
import pathlib
import shutil
import subprocess
import tempfile

import score_dump_oracle

ROOT = pathlib.Path(__file__).resolve().parents[1]
NVCC = os.environ.get("NVCC", "nvcc")
REQUIRED_COMPUTE_CAPABILITY = "12.1"
TP1 = {"tp": 1, "rows": 24, "hidden": 4096, "width": 154880, "document_rows": 8, "tier2": 5}


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
    print(f"SKIP test_score_dump_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_score_dump_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="score-dump-cuda-") as directory:
    work = pathlib.Path(directory)
    binary = str(work / "probe")
    objects = []
    for source in ("src/spark_score_dump.c", "src/spark_sha256.c"):
        objects.append(str(work / (pathlib.Path(source).stem + ".o")))
        subprocess.run(["cc", "-std=c11", "-O2", "-D_POSIX_C_SOURCE=200809L", "-D_DARWIN_C_SOURCE", "-Iinclude", "-c", source, "-o", objects[-1]],
                       cwd=ROOT, check=True)
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "-I.", "-Iinclude",
                    *objects, "tests/host_cuda/head_score_host.cu", "-o", binary], cwd=ROOT, check=True)
    for name, shape in (("tp16_small", score_dump_oracle.SMALL), ("tp1_flash_head", TP1)):
        result = score_dump_oracle.check_pipeline(binary, work / name, shape, lanes=32)
        print(f"PASS {name} {result}")
