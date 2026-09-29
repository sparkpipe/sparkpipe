"""Run the KV quantize-dequantize sim and the BF16-grid KDA state on an sm_121 device."""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_kv_quant_sim_host as host

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


def counters(binary, mode):
    result = subprocess.run([binary, mode], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        host.fail(f"{mode} exited {result.returncode}: {result.stderr[:400]}")
    return {line.split()[0]: int(line.split()[1]) for line in result.stdout.splitlines()}


capabilities = device_capabilities()
if REQUIRED_COMPUTE_CAPABILITY not in capabilities:
    print(f"SKIP test_kv_quant_sim_cuda (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
if shutil.which(NVCC) is None:
    raise SystemExit("FAIL test_kv_quant_sim_cuda: an sm_121 device is present but nvcc is unavailable")
with tempfile.TemporaryDirectory(prefix="kv-quant-sim-cuda-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a",
                    "-I.", "-Iinclude", "tests/cuda/kv_quant_sim_cuda.cu", "-o", binary], cwd=ROOT, check=True)
    hardware = counters(binary, "hardware")
    if hardware["hardware_codec_mismatch"] != 0 or hardware["hardware_codec_compared"] < 30000:
        host.fail(f"sim rounding differs from the hardware cvt.rn.satfinite store codec: {hardware}")
    cases = host.check_rows(binary, np.random.default_rng(20260929))
    host.check_bf16_and_refusals(binary, np.random.default_rng(1))
    state = counters(binary, "state")
    for name in ("grid_state_wave_vs_steps", "grid_out_wave_vs_steps", "grid_state_off_bf16_grid",
                 "fp32_state_wave_vs_steps"):
        if state[name] != 0:
            host.fail(f"{name} = {state[name]}")
    if state["grid_vs_fp32_state_differs"] == 0:
        host.fail("the BF16-grid state never differed from FP32; the test cannot see the rounding")
    print(f"PASS sm_121: sim rounding equals the hardware e4m3/e2m1 cvt on {hardware['hardware_codec_compared']} "
          f"BF16 inputs; {cases} row-sim cases equal the numpy real-codec decode; bf16 no-op and refusals hold; "
          f"BF16-grid KDA state (128x128, 4 heads, 16 tokens): wave equals decode steps bit for bit, "
          f"state on the BF16 grid, {state['grid_vs_fp32_state_differs']} state values differ from FP32")
