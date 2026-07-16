#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label} contains forbidden {needle!r}")


def section(text: str, begin: str, end: str) -> str:
    start = text.index(begin)
    finish = text.index(end, start)
    return text[start:finish]


def load_aot_tool(repository: Path):
    path = repository / "tools" / "glm52_b12x_aot_compile.py"
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(
        "glm52_b12x_aot_compile",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load B12x AOT compiler")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    nvfp4_path = repository / (
        "modules/glm52_sm121_b12x_compiled_backend/source/"
        "spark_flashinfer_b12x_compiled_moe_backend.cu"
    )
    w8lut_path = repository / (
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_sm121_required_decode_stage.cu"
    )
    aot_path = repository / "tools" / "glm52_b12x_aot_compile.py"
    packed_route_header_path = repository / (
        "modules/glm52_resident_decode_stage/include/sparkpipe/"
        "spark_glm52_resident_decode_stage_required_cuda.h"
    )
    builder_path = repository / (
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_pp13_node_context_builder_cuda.cu"
    )
    makefile_path = repository / "Makefile"
    sentinel_path = repository / (
        "modules/glm52_sm121_b12x_compiled_backend/source/"
        "spark_glm52_sm121_b12x_generated_kernel_table_unavailable.c"
    )
    nvfp4 = nvfp4_path.read_text(encoding="utf-8")
    w8lut = w8lut_path.read_text(encoding="utf-8")
    aot = aot_path.read_text(encoding="utf-8")
    packed_route_header = packed_route_header_path.read_text(encoding="utf-8")
    builder = builder_path.read_text(encoding="utf-8")
    makefile = makefile_path.read_text(encoding="utf-8")
    sentinel = sentinel_path.read_text(encoding="utf-8")

    for forbidden in (
        "SparkGlm52B12xLaunchChunked",
        "SparkGlm52B12xSelectLargestBucketAtMost",
        "SparkGlm52B12xPrepareMicroTopK",
        "SparkGlm52B12xPrepareRouterTopKMicroKernel",
        "SparkGlm52B12xPrepareRouterTopKParallelKernel",
        "SparkGlm52B12xPrepareRouterTopK",
        "SparkGlm52B12xAllocateDynamicWorkspace",
        "SparkGlm52B12xMaybeForceBenchmarkExpertCoverage",
        "SparkGlm52B12xMaybeLogExpertCoverage",
        "GLM52_BENCHMARK_FORCE_EXPERT_COVERAGE",
        "GLM52_LOG_EXPERT_COVERAGE",
        "cudaStreamSynchronize",
        "SPARK_GLM52_SM121_B12X_BACKEND_KIND_MICRO",
        "SPARK_GLM52_SM121_B12X_BACKEND_KIND_DYNAMIC",
    ):
        forbid(nvfp4, forbidden, "NVFP4 production backend")
    require(
        nvfp4,
        "bucket->backend_kind !=\n"
        "                SPARK_GLM52_SM121_B12X_BACKEND_KIND_STATIC",
        "NVFP4 manifest validation",
    )
    require(
        nvfp4,
        "if (bucket == 0)\n"
        "    {\n"
        "        return SPARK_STATUS_CAPACITY_EXCEEDED;",
        "NVFP4 exact bucket dispatch",
    )
    require(
        nvfp4,
        "SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_ROUTER_LOGITS",
        "NVFP4 precomputed top-k rejection",
    )

    w8_kernel = section(
        w8lut,
        "static __global__ void "
        "SparkGlm52ResidentDecodeStageW8lutBuildTilesKernel",
        "static __global__ __launch_bounds__(",
    )
    require(w8_kernel, "atomicAdd(tile_count, expert_tile_count)", "W8LUT tile builder")
    forbid(
        w8_kernel,
        "blockIdx.x != 0u || threadIdx.x != 0u",
        "W8LUT tile builder",
    )
    w8_launch = section(
        w8lut,
        "static SparkStatus "
        "SparkGlm52Sm121RequiredDecodeStageLaunchW8lutMoeTensorCore",
        'extern "C" SparkStatus '
        "SparkGlm52Sm121RequiredDecodeStageBindW8lutMoePlan",
    )
    require(w8_launch, "cudaMemsetAsync(", "W8LUT tile counter reset")
    require(
        w8_launch,
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT",
        "W8LUT parallel tile launch",
    )
    forbid(
        w8_launch,
        "SparkGlm52ResidentDecodeStageMaybeForceBenchmarkExpertCoverage",
        "W8LUT production launch",
    )
    forbid(w8_launch, "cudaStreamSynchronize", "W8LUT production launch")
    forbid(w8_launch, "<<<1u, 1u", "W8LUT production launch")
    require(
        w8lut,
        "__shared__ float shared_reduce_scores",
        "parallel W8LUT and NVFP4 top-k",
    )

    packed_route_prefix = section(
        w8lut,
        "void SparkGlm52ResidentDecodeStageMoePackedRoutePrefixKernel",
        "static __global__ __launch_bounds__("
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)\n"
        "void SparkGlm52ResidentDecodeStageMoePackedRouteFillKernel",
    )
    require(
        packed_route_prefix,
        "__shared__ uint32_t shared_prefix",
        "packed-route parallel prefix",
    )
    forbid(
        packed_route_prefix,
        "threadIdx.x != 0u",
        "packed-route parallel prefix",
    )
    packed_route_fill = section(
        w8lut,
        "void SparkGlm52ResidentDecodeStageMoePackedRouteFillKernel",
        "static __global__ __launch_bounds__("
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 4)\n"
        "void SparkGlm52ResidentDecodeStageMoePackedRouteIndptrKernel",
    )
    require(
        packed_route_fill,
        "atomicAdd(&expert_route_write_cursors[expert_index], 1u)",
        "packed-route parallel fill",
    )
    forbid(
        packed_route_fill,
        "threadIdx.x != 0u",
        "packed-route parallel fill",
    )
    require(
        packed_route_header,
        "MOE_PACKED_ROUTE_VIEW_ABI_VERSION 2u",
        "packed-route workspace ABI",
    )
    require(
        packed_route_header,
        "uint32_t *expert_route_write_cursors;",
        "packed-route workspace cursors",
    )
    require(
        builder,
        "node->phase_clock_mode = "
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED;",
        "production phase-clock default",
    )

    for forbidden in (
        "--disable-micro",
        "--allow-dynamic",
        "_MICRO_KERNEL_CACHE",
        "_DYNAMIC_KERNEL_CACHE",
        "#ifndef kDLFloat4_e2m1fn",
    ):
        forbid(aot, forbidden, "NVFP4 production AOT compiler")
    for required in (
        '"production_backend_policy": "exact_static_buckets_only"',
        '"runtime_bucket_decomposition": "forbidden"',
        '"runtime_diagnostic_routing_mutation": "forbidden"',
        "selected forbidden backend",
        "missing exact static AOT buckets",
    ):
        require(aot, required, "NVFP4 production AOT compiler")
    for dspark_rows in ("7", "14", "28", "56", "112", "224", "448", "672", "896"):
        require(
            makefile,
            dspark_rows,
            "NVFP4 default DSpark exact bucket set",
        )

    forbid(sentinel, "__global__", "inactive B12x sentinel")
    forbid(sentinel, "cuda", "inactive B12x sentinel")

    tool = load_aot_tool(repository)
    try:
        tool.find_export_for_bucket({}, "micro", 1)
    except tool.AotFailure:
        pass
    else:
        raise AssertionError("micro AOT backend unexpectedly accepted")
    try:
        tool.bucket_geometry("dynamic", 1)
    except tool.AotFailure:
        pass
    else:
        raise AssertionError("dynamic AOT backend unexpectedly accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
