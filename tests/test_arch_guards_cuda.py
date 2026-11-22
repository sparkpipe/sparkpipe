#!/usr/bin/env python3
"""An architecture the device code does not support is a build error, never a kernel that compiles to a trap or an empty body:
the block-scaled MMA in inference/kernels/mma.cuh builds for sm_120a and sm_121a and refuses sm_90a, naming the instruction, and
the dependent-launch and L2-prefetch headers refuse sm_80, naming what they need. Needs nvcc, not a GPU."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NVCC = os.environ.get("NVCC", "nvcc")
MMA_PROBE = """#include "inference/kernels/mma.cuh"
__global__ void ProbeMma(float *out, const uint32_t *a, const uint32_t *b)
{
	float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	LmMmaMxf8Mxf8(accumulator, a, b, 0x7f7f7f7fu, 0x7f7f7f7fu);
	out[threadIdx.x] = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];
}
"""
LAUNCH_PROBE = """#include "inference/kernels/dependent_launch.cuh"
__global__ void ProbeLaunch(void) { LmDependentWait(); LmDependentRelease(); }
"""
PREFETCH_PROBE = """#include "inference/kernels/weight_prefetch.cuh"
__global__ void ProbePrefetch(const void *address) { (void)address; }
"""


def compile_probe(directory, name, source, arch):
    path = Path(directory) / (name + ".cu")
    path.write_text(source)
    result = subprocess.run([NVCC, "-std=c++17", "-c", "-gencode", "arch=compute_%s,code=sm_%s" % (arch, arch), "-I", str(ROOT), "-I", str(ROOT / "include"), "-I", str(ROOT / "model-families" / "common" / "include"),
                             str(path), "-o", str(Path(directory) / (name + "." + arch + ".o"))], capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def main():
    if shutil.which(NVCC) is None:
        print("SKIP test_arch_guards_cuda (nvcc unavailable on this host)")
        return 0
    failures = []
    with tempfile.TemporaryDirectory(prefix="arch-guards-") as directory:
        for arch in ("120a", "121a"):
            code, output = compile_probe(directory, "mma", MMA_PROBE, arch)
            if code != 0:
                failures.append("the block-scaled MMA must build for sm_%s: %s" % (arch, output.strip()[-300:]))
        cases = (("mma", MMA_PROBE, "90a", "the block-scaled MMA (mma.sync.aligned.m16n8k32 kind::mxf8f6f4.block_scale"),
                 ("launch", LAUNCH_PROBE, "80", "programmatic dependent launch, cluster barriers and bulk L2 prefetch need sm_90 or newer"),
                 ("prefetch", PREFETCH_PROBE, "80", "the bulk L2 prefetch (cp.async.bulk.prefetch.L2.global) needs sm_90 or newer"))
        for name, source, arch, message in cases:
            code, output = compile_probe(directory, name, source, arch)
            if code == 0 or message not in output:
                failures.append("%s for sm_%s must fail to build naming \"%s\" (exit %d): %s" % (name, arch, message, code, output.strip()[-300:]))
    for failure in failures:
        print("FAIL %s" % failure)
    if failures:
        return 1
    print("PASS arch guards: the block-scaled MMA builds for sm_120a and sm_121a and refuses sm_90a; dependent launch and L2 prefetch refuse sm_80")
    return 0


if __name__ == "__main__":
    sys.exit(main())
