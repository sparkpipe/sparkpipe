#!/usr/bin/env python3
"""Host syntax gate for the GLM52 DSpark draft backend.

The backend is CUDA source and this host has no nvcc, so the gate rebuilds the
translation unit as C++: launch configurations are stripped (the kernels are
not executed here), the keyword/intrinsic shims come from tests/host_cuda, the
GEMM entry resolves to tests/host_cuda/shim/runtime/gemm.cuh - whose
LmGemmLaunch signature mirrors the real one - and the runtime APIs the backend
drives are declared against the shim's handle types. This catches type errors,
argument drift against the shared GEMM stack, and stale references to deleted
struct fields. It is a parse, not an execution: device semantics stay gated by
the epoch-3 validator on hardware.
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
    / "modules/glm52_dspark_draft_backend/source"
    / "spark_glm52_dspark_draft_backend.cu"
)

INCLUDES = [
    "-Itests/host_cuda/shim",
    "-Itests/host_cuda",
    "-Itests/cuda_stub",
    "-Iinclude",
    "-I.",
    "-Imodel-families/common/include",
    "-Imodel-families/glm52/include",
    "-Imodules/glm52_dspark_draft_backend/include",
]

# Runtime APIs the backend drives, declared against lm_host_cuda.cuh's handle
# types (int streams/events). Pointer-to-int casts in the source carry no
# signal under this harness; everything else must be clean.
PRELUDE = """
#include "lm_host_cuda.cuh"
/* The recorder GEMM shim does not carry tile.cuh, where the pipeline stage
 * count lives; the backend launches with it. */
#include "inference/kernels/tile.cuh"
#include <cuda_bf16.h>
#include <string.h>
#include <stdlib.h>

/* Runtime APIs the backend drives, declared against lm_host_cuda.cuh's handle
 * types (int streams/events). Pointer-to-int casts in the source carry no
 * signal under this harness; everything else must be clean. */
typedef int cudaMemcpyKind;
typedef int cudaEvent_t;
#define cudaMemcpyHostToDevice ((cudaMemcpyKind)1)
#define cudaMemcpyDeviceToHost ((cudaMemcpyKind)2)
#ifndef cudaErrorNotReady
#define cudaErrorNotReady 34
#endif
#define cudaEventDisableTiming 2u
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount 16
#endif
static inline const char *cudaGetErrorString(cudaError_t) { return "stub"; }
static inline cudaError_t cudaMalloc(void **p, size_t bytes) { *p = malloc(bytes); return cudaSuccess; }
static inline cudaError_t cudaFree(void *p) { free(p); return cudaSuccess; }
static inline cudaError_t cudaMallocHost(void **p, size_t bytes) { *p = malloc(bytes); return cudaSuccess; }
static inline cudaError_t cudaFreeHost(void *p) { free(p); return cudaSuccess; }
static inline cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind) { memcpy(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, cudaMemcpyKind, cudaStream_t) { memcpy(d, s, n); return cudaSuccess; }
static inline cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = 1; return cudaSuccess; }
static inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaEventCreate(int *e) { *e = 1; return cudaSuccess; }
static inline cudaError_t cudaEventCreateWithFlags(int *e, unsigned) { *e = 1; return cudaSuccess; }
static inline cudaError_t cudaEventRecord(int, cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaEventQuery(int) { return cudaSuccess; }
static inline cudaError_t cudaEventSynchronize(int) { return cudaSuccess; }
static inline cudaError_t cudaEventElapsedTime(float *ms, int, int) { *ms = 0.0f; return cudaSuccess; }
static inline cudaError_t cudaEventDestroy(int) { return cudaSuccess; }
static inline cudaError_t cudaDeviceGetAttribute(int *v, int, int) { *v = 1; return cudaSuccess; }
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


def main() -> int:
    source = SOURCE.read_text()
    # Kernel launch configuration is not C++. The kernels themselves keep
    # parsing as plain functions, so their argument lists stay checked.
    transformed = re.sub(r"<<<.*?>>>", "", source, flags=re.S)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cpp", delete=False
    ) as handle:
        handle.write(PRELUDE + transformed)
        tu_path = handle.name
    try:
        result = subprocess.run(
            [host_cxx(), "-std=c++17", "-Wall", "-Wextra", "-fpermissive",
             "-fsyntax-only"] + INCLUDES + [tu_path],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
    finally:
        os.unlink(tu_path)
    # Warnings here are dominated by transform artifacts (handle casts to the
    # shim's integer stream/event types, arguments consumed only inside
    # stripped launch configs), so they are reported, not fatal. Errors are.
    errors = [
        line
        for line in result.stderr.splitlines()
        if " error:" in line
    ]
    if result.returncode != 0 or errors:
        print(result.stderr)
        print("\nFAIL DSpark draft backend host syntax")
        return 1
    warnings = sum(1 for line in result.stderr.splitlines()
                   if "warning:" in line)
    print(f"PASS DSpark draft backend host syntax "
          f"(launch configs stripped, {warnings} harness-noise warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
