#!/usr/bin/env python3
"""Check that tensor maps built on the GPU drive the engine's GEMMs exactly as the CUDA driver's do.

The nvrm backend has no host encoder for tensor maps: SparkDeviceTensorMapEncode builds them on the GPU,
writing every field of a zeroed 128-byte object with tensormap.replace. This test runs the engine's
tensor-map GEMMs on the CUDA flavour three times, in separate processes so the descriptor cache starts
empty: with maps from the device layer's host encoder (cuTensorMapEncodeTiled), with maps from the GPU
builder, and with GPU-built maps whose strides were written one ordinal up. The first two must give
bit-identical outputs for every GEMM, including the identical refusals of formats the map planner does not
take; the control must change every GEMM's output, so the comparison sees the maps. Only outputs are
compared, never map bytes.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_tensor_map_build_cuda"
GEMMS = 32
REFUSALS = 8


def run(mode, output):
    result = subprocess.run([str(BINARY), mode, str(output)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    if result.returncode != 0 or "PASS tensor map build " + mode not in result.stdout:
        return None
    return output.read_text().splitlines()


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_tensor_map_build_cuda (no CUDA toolkit on this host)")
        return 0
    if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0:
        print("SKIP test_tensor_map_build_cuda (no CUDA device on this host)")
        return 0
    lines = {}
    for mode in ("host", "device", "strides-shifted"):
        lines[mode] = run(mode, ROOT / ("build/tensor_map_build.%s.txt" % mode))
        if lines[mode] is None:
            print("FAIL tensor map build: the %s run failed" % mode)
            return 1
    ran = [line for line in lines["host"] if line.endswith(" ok")]
    refused = [line for line in lines["host"] if " refused " in line]
    if len(ran) != GEMMS or len(refused) != REFUSALS:
        print("FAIL tensor map build: %d GEMMs ran and %d were refused, expected %d and %d" % (len(ran), len(refused), GEMMS, REFUSALS))
        return 1
    if lines["host"] != lines["device"]:
        differ = sum(1 for host, device in zip(lines["host"], lines["device"]) if host != device)
        print("FAIL tensor map build: %d of %d outputs differ between driver-encoded and GPU-built maps" % (differ, len(lines["host"])))
        return 1
    unchanged = [device for device, shifted in zip(lines["device"], lines["strides-shifted"]) if device.endswith(" ok") and device == shifted]
    if len(lines["strides-shifted"]) != len(lines["device"]) or unchanged:
        print("FAIL tensor map build: the shifted-stride control left %d GEMM outputs unchanged" % len(unchanged))
        for line in unchanged[:4]:
            print("  " + line)
        return 1
    print("PASS tensor map build: %d GEMMs bit-identical with driver-encoded and GPU-built maps, %d identical refusals, "
          "%d of %d changed by the shifted-stride control" % (GEMMS, REFUSALS, GEMMS, GEMMS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
