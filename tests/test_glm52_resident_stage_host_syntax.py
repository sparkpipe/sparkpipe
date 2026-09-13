#!/usr/bin/env python3
"""Host syntax gate for the GLM52 resident decode stage CUDA translation unit.

The stage is CUDA source and this host has no nvcc, so the gate rebuilds
spark_glm52_resident_decode_stage_cuda.cu - which includes cuda/unity.cu and
through it the whole layer.cuh kernel stack - as C++: launch configurations are
stripped (the kernels are not executed here), the keyword/intrinsic shims come
from tests/host_cuda, kv.cuh's __CUDACC__-guarded store kernel is scoped in the
same way glm52_layer_host.cu scopes it, and runtime/gemm.cuh resolves to the
recorder shim whose launch signatures mirror the real ones. This catches type
errors, stale references to deleted symbols (the dead-code round's shims), and
argument drift against the shared kernel stack. It is a parse, not an
execution: device semantics stay gated by the sm_121a validator on hardware.

Re-landed (confidence-gating round, from the dead-code round's evidence patch)
with the codec axis made exact: the TU is compiled once per published expert
codec - int6, int7, int8, fp8, nvfp4, mxfp4, the same six the module contract
gate pins - because the expert-weight paths are compile-time selected and a
syntax break in one codec's selection must not hide behind another's.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "modules/glm52_resident_decode_stage/source"
    / "spark_glm52_resident_decode_stage_cuda.cu"
)

# The module Makefile's exact codec set and ids.
CODECS = (
    ("int6", 2),
    ("int7", 3),
    ("int8", 4),
    ("fp8", 5),
    ("nvfp4", 6),
    ("mxfp4", 7),
)

INCLUDES = [
    "-Itests/host_cuda/shim",
    "-Itests/host_cuda",
    "-Iinclude",
    "-I.",
    "-Imodel-families/common/include",
    "-Imodel-families/glm52/include",
    "-Imodules/glm52_resident_decode_stage/include",
    "-Imodules/glm52_resident_decode_stage/source",
    "-Ideployment/include",
]

# Runtime APIs the stage drives beyond lm_host_cuda.cuh's surface, declared
# against its handle types (integer streams). Pointer-to-int casts carry no
# signal under this harness; everything else must be clean.
PRELUDE = """
#include "tests/host_cuda/lm_host_cuda.cuh"
#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__
#include <string.h>
#include <stdlib.h>
typedef int cudaMemcpyKind;
#define cudaMemcpyHostToDevice ((cudaMemcpyKind)1)
#define cudaMemcpyDeviceToHost ((cudaMemcpyKind)2)
#define cudaMemcpyDeviceToDevice ((cudaMemcpyKind)3)
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue 1
#endif
#ifndef cudaErrorUnknown
#define cudaErrorUnknown 999
#endif
struct cudaDeviceProp { int major; int minor; int multiProcessorCount; };
static inline cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, cudaMemcpyKind, cudaStream_t) { memcpy(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaMemsetAsync(void *d, int v, size_t n, cudaStream_t) { memset(d, v, n); return cudaSuccess; }
static inline cudaError_t cudaGetDevice(int *device) { *device = 0; return cudaSuccess; }
static inline cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *properties, int) { properties->major = 12; properties->minor = 1; properties->multiProcessorCount = 1; return cudaSuccess; }
"""


def host_cxx():
    configured = os.environ.get("SPARKPIPE_HOST_CUDA_CXX")
    if configured:
        return configured
    for version in range(20, 9, -1):
        candidate = f"g++-{version}"
        if shutil.which(candidate) is not None:
            return candidate
    raise RuntimeError(
        "the host syntax gate requires GNU g++; install Homebrew gcc or set "
        "SPARKPIPE_HOST_CUDA_CXX")


def check_codec(compiler: str, transformed: str, name: str, codec_id: int) -> int:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cpp", delete=False
    ) as handle:
        handle.write(PRELUDE + transformed)
        tu_path = handle.name
    try:
        result = subprocess.run(
            [compiler, "-std=c++17", "-fpermissive", "-fsyntax-only",
             f"-DGLM52_EXPERT_WEIGHT_CODEC={codec_id}",
             f'-DGLM52_EXPERT_CODEC_NAME="{name}"']
            + INCLUDES + [tu_path],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
    finally:
        os.unlink(tu_path)
    if result.returncode != 0:
        print(result.stderr[-4000:])
        print(f"FAIL GLM52 resident CUDA TU does not compile under the host "
              f"syntax harness for codec {name}")
        return result.returncode or 1
    return 0


def main() -> int:
    source = SOURCE.read_text()
    # Kernel launch configuration is not C++. The kernels themselves keep
    # parsing as plain functions, so their argument lists stay checked.
    transformed = re.sub(r"<<<.*?>>>", "", source, flags=re.S)
    compiler = host_cxx()
    for name, codec_id in CODECS:
        if check_codec(compiler, transformed, name, codec_id) != 0:
            return 1
    print(f"PASS GLM52 resident CUDA TU host syntax x{len(CODECS)} codecs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
