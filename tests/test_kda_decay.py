#!/usr/bin/env python3
"""LmBoundedDecayKernel must implement Kimi K3 technical report equation 5.

    g     = g_min * Sigmoid(exp(A_h) * (z + b))   in (g_min, 0)
    alpha = exp(g)                                 in (exp(g_min), 1)

A_h is a learnable per-head log-scale initialised to 0, b the per-channel bias
added to the logit in fp32 before the scale, and g_min is fixed at -5.

The bound keeps every retention factor above exp(-5), so the cumulative
log-decay over a 16-token tile stays inside (-80, 0) and its reciprocal inside
BF16 range. The test runs the real kernel from inference/kernels/linear_attn.cuh
on the CPU through the host CUDA shim and checks that every output matches the
reference, stays inside [exp(g_min), 1] even at saturating logits, and still
moves with the per-head scale and the channel bias, so a clamp cannot pass.
"""
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "kda_decay_host.cu"
G_MIN = -5.0
CASES = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (20.0, 0.0, 0.0),
         (-20.0, 0.0, 0.0), (1000.0, 0.0, 0.0), (-1000.0, 0.0, 0.0),
         (1.0, 1.5, 0.0), (1.0, -1.5, 0.0),
         (0.0, 0.0, 2.0), (0.0, 0.0, -2.0), (1.0, 0.5, -3.0), (0.3, 0.0, 0.0)]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def reference(logit, head_log_scale, bias):
    return math.exp(G_MIN * sigmoid(math.exp(head_log_scale) * (logit + bias)))


def run():
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory) / "kda_decay_host"
        build = subprocess.run(
            [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}", f"-I{ROOT}/tests/host_cuda",
             "-x", "c++", str(SOURCE), "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL host build:", build.stderr.strip()[:400])
            return None
        arguments = ["%.9g" % G_MIN] + ["%.9g" % value for case in CASES for value in case]
        result = subprocess.run([str(binary), *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL host run exited {result.returncode}: {result.stderr.strip()[:200]}")
        return None
    rows = [line.split() for line in result.stdout.splitlines()]
    if len(rows) != len(CASES) or any(len(row) != 2 for row in rows):
        print(f"FAIL expected {len(CASES)} kernel outputs, got {result.stdout!r}")
        return None
    return [(float(row[0]), float(row[1])) for row in rows]


def main():
    got = run()
    if got is None:
        return 1
    failures = 0
    floor = math.exp(G_MIN)
    outputs = {}
    for (logit, scale, bias), (kernel_logit, value) in zip(CASES, got):
        if kernel_logit != bf16(logit):
            print(f"  FAIL z={logit}: kernel read {kernel_logit!r}, bf16 of the input is {bf16(logit)!r}")
            failures += 1
        want = reference(kernel_logit, scale, bias)
        if not math.isfinite(value) or abs(want - value) > max(1e-6, abs(want) * 1e-6):
            print(f"  FAIL z={logit} A={scale} b={bias}: kernel {value:.9g}, reference {want:.9g}")
            failures += 1
        if not (floor * (1.0 - 1e-6) <= value <= 1.0):
            print(f"  FAIL z={logit} A={scale} b={bias}: {value:.9g} outside [exp(-5), 1]")
            failures += 1
        outputs[(logit, scale, bias)] = value
    if abs(outputs[(1.0, 1.5, 0.0)] - outputs[(1.0, -1.5, 0.0)]) < 1e-3:
        print("  FAIL the per-head log-scale does not move the kernel output")
        failures += 1
    if abs(outputs[(0.0, 0.0, 2.0)] - outputs[(0.0, 0.0, -2.0)]) < 1e-3:
        print("  FAIL the channel bias does not move the kernel output")
        failures += 1
    if outputs[(-1000.0, 0.0, 0.0)] != 1.0:
        print("  FAIL a saturating negative logit does not retain the state exactly")
        failures += 1
    print(f"cases {len(CASES)}, g_min {G_MIN}, floor exp(g_min) = {floor:.6g}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nLmBoundedDecayKernel matches report eq. 5 and stays inside [exp(g_min), 1]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
