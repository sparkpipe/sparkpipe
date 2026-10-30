#!/usr/bin/env python3
"""LmSituMulKernel must compute what modeling_kimi_linear.py computes.

Kimi K3 runs SiTU on all 93 layers. The formula is not in the config and was
not published; it came from the released modelling file:

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up     = linear_beta * tanh(up / linear_beta)     when linear_beta is set
    out    = situ_a * up

This gate builds the kernel itself through the host CUDA shim, runs it on
BF16 inputs and compares its BF16 output with that reference evaluated on the
same inputs, at points chosen to separate SiTU from the things it could be
confused with. Checking that it "looks like a tanh and a sigmoid" would pass for
several functions that are not this one.

The distinguishing property is the clamp: both arms saturate at their beta.
beta is 4.0 and linear_beta is 25.0, and swapping them clamps the gate at 25 and
the linear arm at 4 - a model that runs and is wrong. The large-input cases
below catch that swap; small-input cases would not, because near zero
beta*tanh(x/beta) is approximately x for either beta. The kernel is also run
with its betas swapped, and that run must fail the comparison.
"""
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
BETA = 4.0
LINEAR_BETA = 25.0


def reference(gate, up, beta=BETA, linear_beta=LINEAR_BETA):
    situ_a = beta * math.tanh(gate / beta) * (1.0 / (1.0 + math.exp(-gate)))
    if linear_beta > 0.0:
        up = linear_beta * math.tanh(up / linear_beta)
    return situ_a * up


def bf16_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


PROBE = r"""
#include "tests/host_cuda/lm_host_cuda.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES
#include "inference/kernels/norm.cuh"
#include <stdio.h>
#include <stdlib.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
float lm_norm_shared[4104];
static uint16_t rows_bf16[64 * 2];
static uint16_t output_bf16[64];
int main(int argc, char **argv)
{
	uint32_t count;
	float beta = (float)atof(argv[1]), linear_beta = (float)atof(argv[2]);
	if ( fread(&count,4,1,stdin) != 1 || count == 0 || count > 64 || fread(rows_bf16,2,count * 2,stdin) != count * 2 )
		return(1);
	blockDim.x = 32;
	for (blockIdx.x = 0; blockIdx.x < count; blockIdx.x++)
		for (threadIdx.x = 0; threadIdx.x < 32; threadIdx.x++)
			LmSituMulKernel<32>(rows_bf16,output_bf16,1u,1u,beta,linear_beta);
	return(fwrite(output_bf16,2,count,stdout) == count ? 0 : 2);
}
"""


def build_probe(directory):
    source = directory / "situ_probe.cu"
    binary = directory / "situ_probe"
    source.write_text(PROBE)
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O2", "-I" + str(ROOT), "-I" + str(ROOT / "tests/host_cuda"), "-x", "c++",
                    str(source), "-o", str(binary)], check=True)
    return binary


def run_kernel(binary, cases, beta, linear_beta):
    data = struct.pack("<I", len(cases)) + b"".join(struct.pack("<2H", gate, up) for gate, up in cases)
    out = subprocess.run([str(binary), repr(beta), repr(linear_beta)], input=data, capture_output=True, check=True, timeout=30).stdout
    return [bf16_value(bits) for bits in struct.unpack("<%dH" % len(cases), out)]


def mismatches(cases, got):
    bad = []
    for (gate, up), value in zip(cases, got):
        want = reference(bf16_value(gate), bf16_value(up))
        if abs(want - value) > abs(want) * 2.0 ** -7 + 1e-7:
            bad.append((bf16_value(gate), bf16_value(up), value, want))
    return bad


def main():
    points = [
        (0.0, 1.0),
        (1.0, 1.0),
        (-1.0, 2.0),
        (3.9, 10.0),
        (40.0, 1.0),
        (-40.0, 1.0),
        (1.0, 400.0),
        (1.0, -400.0),
        (100.0, 100.0),
        (0.5, 0.5),
    ]
    cases = [(bf16_bits(gate), bf16_bits(up)) for gate, up in points]
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        binary = build_probe(Path(directory))
        got = run_kernel(binary, cases, BETA, LINEAR_BETA)
        swapped_run = run_kernel(binary, cases, LINEAR_BETA, BETA)
    for gate, up, value, want in mismatches(cases, got):
        print(f"  FAIL gate={gate} up={up}: kernel {value:.9g}, reference {want:.9g}")
        failures += 1
    if not mismatches(cases, swapped_run):
        print("  FAIL the kernel run with its betas swapped still matches; the test cannot see a swap")
        failures += 1
    saturated = reference(1e4, 1.0) / (LINEAR_BETA * math.tanh(1.0 / LINEAR_BETA))
    if abs(saturated - BETA) > 1e-3:
        print(f"  FAIL gate arm saturates at {saturated:.6g}, expected {BETA}")
        failures += 1
    print(f"cases {len(cases)}, betas {BETA} and {LINEAR_BETA}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nLmSituMulKernel, run on BF16 inputs, matches modeling_kimi_linear.py; both arms clamp and a beta swap is caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
