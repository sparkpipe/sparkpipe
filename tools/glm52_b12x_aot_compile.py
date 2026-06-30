#!/usr/bin/env python3
"""
Compile the vendored FlashInfer SM121 B12x CuTe DSL MoE kernels once and emit
native SparkPipe runtime glue.

The generated runtime pack contains exported TVM-FFI object files and a C++ CUDA
launch table. Serving links that pack and calls it from C/CUDA only. Python,
Torch, FlashInfer, and CuTe DSL are used only by this one-time build tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUIRED_SHAPE = {
    "hidden_dimension": 6144,
    "intermediate_dimension": 2048,
    "expert_count": 256,
    "top_k": 8,
    "fused_w1_rows": 4096,
}

BACKEND_KIND = {
    "micro": 1,
    "static": 2,
    "dynamic": 3,
}


class AotFailure(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def low64_from_hex(digest: str) -> int:
    return int(digest[:16], 16)


def parse_tokens(value: str) -> List[int]:
    result: List[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        token_count = int(item)
        if token_count <= 0:
            raise AotFailure("token bucket values must be positive")
        result.append(token_count)
    if not result:
        raise AotFailure("at least one token bucket is required")
    return sorted(set(result))


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def mma_scale_shape(m: int, k: int, expert_count: int) -> Tuple[int, int, int, int, int, int]:
    scale_k = ceil_div(k, 16)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(scale_k, 4)
    return (32, 4, m_tiles, 4, k_tiles, expert_count)


def install_vendored_flashinfer(root: Path, workspace: Path) -> None:
    vendored = root / "third_party" / "flashinfer"
    if not (vendored / "flashinfer" / "fused_moe" / "cute_dsl" / "b12x_moe.py").exists():
        raise AotFailure("vendored FlashInfer B12x source is missing")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(workspace))
    sys.path.insert(0, str(vendored))


def require_sm121(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise AotFailure("CUDA is unavailable")
    properties = torch_module.cuda.get_device_properties(0)
    if properties.major != 12 or properties.minor not in (0, 1):
        raise AotFailure(f"expected SM120/SM121, got sm_{properties.major}{properties.minor}")


def make_inputs(torch_module: Any, token_count: int) -> Dict[str, Any]:
    expert_count = REQUIRED_SHAPE["expert_count"]
    top_k = REQUIRED_SHAPE["top_k"]
    hidden_dimension = REQUIRED_SHAPE["hidden_dimension"]
    hidden = torch_module.empty(
        (token_count, hidden_dimension),
        device="cuda",
        dtype=torch_module.bfloat16,
    )
    hidden.normal_(mean=0.0, std=0.01)
    topk_ids = torch_module.arange(
        token_count * top_k,
        device="cuda",
        dtype=torch_module.int32,
    ).reshape(token_count, top_k)
    topk_ids.remainder_(expert_count)
    topk_weights = torch_module.full(
        (token_count, top_k),
        1.0 / float(top_k),
        device="cuda",
        dtype=torch_module.float32,
    )
    output = torch_module.empty(
        (token_count, hidden_dimension),
        device="cuda",
        dtype=torch_module.bfloat16,
    )
    return {
        "hidden": hidden,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "output": output,
    }


def make_weights(torch_module: Any) -> Dict[str, Any]:
    expert_count = REQUIRED_SHAPE["expert_count"]
    hidden_dimension = REQUIRED_SHAPE["hidden_dimension"]
    intermediate_dimension = REQUIRED_SHAPE["intermediate_dimension"]
    fused_w1_rows = REQUIRED_SHAPE["fused_w1_rows"]
    w1_weight = torch_module.empty(
        (expert_count, fused_w1_rows, hidden_dimension // 2),
        device="cuda",
        dtype=torch_module.uint8,
    )
    w2_weight = torch_module.empty(
        (expert_count, hidden_dimension, intermediate_dimension // 2),
        device="cuda",
        dtype=torch_module.uint8,
    )
    w1_weight.random_(0, 16)
    w2_weight.random_(0, 16)
    w1_scale = torch_module.ones(
        mma_scale_shape(fused_w1_rows, hidden_dimension, expert_count),
        device="cuda",
        dtype=torch_module.float8_e4m3fn,
    )
    w2_scale = torch_module.ones(
        mma_scale_shape(hidden_dimension, intermediate_dimension, expert_count),
        device="cuda",
        dtype=torch_module.float8_e4m3fn,
    )
    w1_alpha = torch_module.ones((expert_count,), device="cuda", dtype=torch_module.float32)
    w2_alpha = torch_module.ones((expert_count,), device="cuda", dtype=torch_module.float32)
    fc2_input_scale = torch_module.ones((expert_count,), device="cuda", dtype=torch_module.float32)
    return {
        "w1_weight": w1_weight,
        "w1_scale": w1_scale,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": fc2_input_scale,
        "w2_weight": w2_weight,
        "w2_scale": w2_scale,
        "w2_alpha": w2_alpha,
    }


def warm_bucket(torch_module: Any, wrapper: Any, weights: Dict[str, Any], token_count: int) -> Dict[str, Any]:
    inputs = make_inputs(torch_module, token_count)
    output = wrapper.run(
        x=inputs["hidden"],
        w1_weight=weights["w1_weight"],
        w1_weight_sf=weights["w1_scale"],
        w1_alpha=weights["w1_alpha"],
        fc2_input_scale=weights["fc2_input_scale"],
        w2_weight=weights["w2_weight"],
        w2_weight_sf=weights["w2_scale"],
        w2_alpha=weights["w2_alpha"],
        token_selected_experts=inputs["topk_ids"],
        token_final_scales=inputs["topk_weights"],
    )
    torch_module.cuda.synchronize()
    if output is None:
        output = inputs["output"]
    return inputs


def time_bucket(torch_module: Any, wrapper: Any, weights: Dict[str, Any], token_count: int, warmup: int, iterations: int) -> Dict[str, float]:
    inputs = make_inputs(torch_module, token_count)

    def body() -> None:
        wrapper.run(
            x=inputs["hidden"],
            w1_weight=weights["w1_weight"],
            w1_weight_sf=weights["w1_scale"],
            w1_alpha=weights["w1_alpha"],
            fc2_input_scale=weights["fc2_input_scale"],
            w2_weight=weights["w2_weight"],
            w2_weight_sf=weights["w2_scale"],
            w2_alpha=weights["w2_alpha"],
            token_selected_experts=inputs["topk_ids"],
            token_final_scales=inputs["topk_weights"],
        )

    for _ in range(warmup):
        body()
    torch_module.cuda.synchronize()
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    timings: List[float] = []
    for _ in range(iterations):
        start.record()
        body()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) * 1000.0)
    return {
        "avg_us": statistics.fmean(timings),
        "p50_us": statistics.median(timings),
        "p95_us": sorted(timings)[max(0, int(len(timings) * 0.95) - 1)],
        "min_us": min(timings),
        "max_us": max(timings),
    }


def kernel_function_name(kind: str, token_count: Optional[int]) -> str:
    if kind == "dynamic":
        return "spark_glm52_b12x_dynamic_e256_h6144_i2048_topk8"
    if token_count is None:
        raise AotFailure("static and micro kernels need a token count")
    return f"spark_glm52_b12x_{kind}_t{token_count}_e256_h6144_i2048_topk8"


def export_compiled_objects(dispatch_module: Any, objects_directory: Path) -> Dict[str, Dict[str, Any]]:
    objects_directory.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Dict[str, Any]] = {}
    caches = [
        ("static", dispatch_module._STATIC_KERNEL_CACHE),
        ("micro", dispatch_module._MICRO_KERNEL_CACHE),
        ("dynamic", dispatch_module._DYNAMIC_KERNEL_CACHE),
    ]
    for kind, cache in caches:
        for key, value in sorted(cache.items(), key=lambda item: repr(item[0])):
            compiled = value[0]
            token_count: Optional[int] = None
            if kind == "static":
                token_count = int(key[4])
            elif kind == "micro":
                token_count = int(key[3])
            name = kernel_function_name(kind, token_count)
            object_path = objects_directory / f"{name}.o"
            if not hasattr(compiled, "export_to_c"):
                raise AotFailure(f"compiled {kind} object does not support export_to_c")
            compiled.export_to_c(str(object_path), function_name=name)
            exported[name] = {
                "kind": kind,
                "token_count": token_count,
                "object": object_path.name,
                "cache_key_repr": repr(key),
                "sha256": sha256_file(object_path),
            }
    return exported


def find_export_for_bucket(exported: Dict[str, Dict[str, Any]], kind: str, token_count: int) -> Tuple[str, str]:
    if kind == "dynamic":
        name = kernel_function_name("dynamic", None)
        if name not in exported:
            raise AotFailure(f"dynamic kernel for token bucket {token_count} was not exported")
        return name, "dynamic"
    if kind == "static":
        name = kernel_function_name("micro", token_count)
        if name in exported:
            return name, "micro"
        name = kernel_function_name("static", token_count)
        if name in exported:
            return name, "static"
        raise AotFailure(
            f"neither static nor micro kernel for token bucket {token_count} was exported"
        )
    name = kernel_function_name(kind, token_count)
    if name not in exported:
        raise AotFailure(f"{kind} kernel for token bucket {token_count} was not exported")
    return name, kind


def dynamic_geometry(routed_rows: int) -> Dict[str, int]:
    expert_count = REQUIRED_SHAPE["expert_count"]
    intermediate_dimension = REQUIRED_SHAPE["intermediate_dimension"]
    tile_m = 128
    tile_n = 128
    base_m_tiles = align_up(max(1, routed_rows), tile_m) // tile_m
    active_expert_upper_bound = min(expert_count, max(1, routed_rows))
    physical_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_count = max(1, ceil_div(intermediate_dimension, tile_n))
    slice_groups = gate_tile_count
    task_capacity = physical_tiles * slice_groups
    return {
        "max_rows": physical_tiles * tile_m,
        "physical_tile_capacity": physical_tiles,
        "task_capacity": task_capacity,
    }


def bucket_geometry(kind: str, token_count: int) -> Dict[str, int]:
    routed_rows = token_count * REQUIRED_SHAPE["top_k"]
    if kind == "dynamic":
        geometry = dynamic_geometry(routed_rows)
        return {
            "routed_rows_capacity": routed_rows,
            "max_rows": geometry["max_rows"],
            "physical_tile_capacity": geometry["physical_tile_capacity"],
            "task_capacity": geometry["task_capacity"],
        }
    return {
        "routed_rows_capacity": routed_rows,
        "max_rows": routed_rows,
        "physical_tile_capacity": 0,
        "task_capacity": 0,
    }


def run_config_command(*arguments: str) -> str:
    executable = shutil.which("tvm-ffi-config")
    if executable is None:
        return ""
    completed = subprocess.run(
        [executable, *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def write_tvm_ffi_flags(generated_directory: Path) -> None:
    cflags = run_config_command("--cflags")
    ldflags = run_config_command("--ldflags")
    libs = run_config_command("--libs")
    runtime_libs = ""
    include_flags: List[str] = []

    def add_include(path: Path) -> None:
        if path.exists():
            flag = f"-I{path}"
            if flag not in include_flags:
                include_flags.append(flag)

    try:
        import cutlass.cute as cute

        runtime_libs = " ".join(cute.runtime.find_runtime_libraries(enable_tvm_ffi=True))
    except Exception:
        runtime_libs = ""
    tvm_ffi_spec = importlib.util.find_spec("tvm_ffi")
    if tvm_ffi_spec is not None and tvm_ffi_spec.origin is not None:
        tvm_ffi_directory = Path(tvm_ffi_spec.origin).resolve().parent
        add_include(tvm_ffi_directory / "include")
        add_include(tvm_ffi_directory / "3rdparty" / "dlpack" / "include")
    for item in runtime_libs.split():
        library_path = Path(item)
        if library_path.name == "libtvm_ffi.so":
            tvm_ffi_directory = library_path.resolve().parents[1]
            add_include(tvm_ffi_directory / "include")
            add_include(tvm_ffi_directory / "3rdparty" / "dlpack" / "include")
    cflags = " ".join(item for item in [cflags, *include_flags] if item).strip()
    text = "\n".join(
        [
            f"TVM_FFI_CFLAGS := {cflags}",
            f"TVM_FFI_LDFLAGS := {ldflags}",
            f"TVM_FFI_LIBS := {libs}",
            f"CUTE_TVM_FFI_RUNTIME_LIBS := {runtime_libs}",
            "",
        ]
    )
    (generated_directory / "tvm_ffi_flags.mk").write_text(text)
    (generated_directory / "runtime_link_args.txt").write_text(
        " ".join(item for item in [ldflags, libs, runtime_libs] if item).strip() + "\n"
    )


def generate_launch_table_source(manifest: Dict[str, Any], exported: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    buckets = manifest["buckets"]
    function_names = sorted({bucket["function_name"] for bucket in buckets})
    externs = "\n".join(
        f'extern "C" int __tvm_ffi_{name}(void *, const TVMFFIAny *, int32_t, TVMFFIAny *);'
        for name in function_names
    )
    bucket_initializers = []
    for bucket in buckets:
        bucket_initializers.append(
            "    {"
            f"SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION, {bucket['token_upper_bound']}u, "
            f"SPARK_GLM52_SM121_B12X_BACKEND_KIND_{bucket['backend_kind'].upper()}, {bucket['routed_rows_capacity']}u, "
            f"{bucket['max_rows']}u, {bucket['physical_tile_capacity']}u, {bucket['task_capacity']}u, "
            f"{bucket['max_active_clusters']}u, {bucket['static_mma_tile_m']}u, {bucket['static_mma_tile_n']}u, "
            "0u, 0u, "
            f"{int(bucket.get('avg_us', 0))}u, {int(bucket.get('p95_us', 0))}u"
            "}"
        )
    switch_cases = []
    for bucket in buckets:
        launch_name = (
            f"SparkGlm52B12xLaunch{bucket['backend_kind'].capitalize()}T{bucket['token_upper_bound']}"
        )
        switch_cases.append(
            f"        case {bucket['token_upper_bound']}u:\n"
            f"            return {launch_name}(arguments);"
        )
    helper_source = f'''#include "sparkpipe/spark_glm52_sm121_b12x_generated_kernel_table.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <tvm/ffi/c_api.h>

#ifndef kDLFloat4_e2m1fn
#define kDLFloat4_e2m1fn 17
#endif

#ifndef TVM_FFI_C_API_H_
#error "TVM FFI headers are required for the generated B12x runtime table"
#endif

{externs}

typedef int (*SparkGlm52B12xTvmFunction)(
    void *,
    const TVMFFIAny *,
    int32_t,
    TVMFFIAny *);

static const SparkGlm52Sm121B12xGeneratedKernelBucket
    SparkGlm52B12xGeneratedBuckets[] = {{
{',\n'.join(bucket_initializers)}
}};

const SparkGlm52Sm121B12xGeneratedManifest
    SparkGlm52Sm121B12xGeneratedManifestInstance = {{
        SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION,
        {len(buckets)}u,
        {REQUIRED_SHAPE['hidden_dimension']}u,
        {REQUIRED_SHAPE['intermediate_dimension']}u,
        {REQUIRED_SHAPE['expert_count']}u,
        {REQUIRED_SHAPE['top_k']}u,
        {manifest['maximum_token_count']}u,
        121u,
        UINT64_C(0x{manifest['manifest_hash_low64']:016x}),
        SparkGlm52B12xGeneratedBuckets
    }};

static DLDataType SparkGlm52B12xDataType(
    uint8_t code,
    uint8_t bits,
    uint16_t lanes)
{{
    DLDataType data_type;

    data_type.code = code;
    data_type.bits = bits;
    data_type.lanes = lanes;
    return data_type;
}}

static void SparkGlm52B12xFillTensor(
    DLTensor *tensor,
    void *data,
    DLDataType data_type,
    int32_t dimension_count,
    int64_t *shape,
    int64_t *strides)
{{
    memset(tensor, 0, sizeof(*tensor));
    tensor->data = data;
    tensor->device.device_type = kDLCUDA;
    tensor->device.device_id = 0;
    tensor->ndim = dimension_count;
    tensor->dtype = data_type;
    tensor->shape = shape;
    tensor->strides = strides;
    tensor->byte_offset = 0u;
}}

static TVMFFIAny SparkGlm52B12xTensorArgument(DLTensor *tensor)
{{
    TVMFFIAny argument;

    memset(&argument, 0, sizeof(argument));
    argument.type_index = kTVMFFIDLTensorPtr;
    argument.v_ptr = tensor;
    return argument;
}}

static TVMFFIAny SparkGlm52B12xPointerArgument(const void *pointer)
{{
    TVMFFIAny argument;

    memset(&argument, 0, sizeof(argument));
    argument.type_index = kTVMFFIInt;
    argument.v_int64 = (int64_t)(uintptr_t)pointer;
    return argument;
}}

static TVMFFIAny SparkGlm52B12xIntegerArgument(uint32_t value)
{{
    TVMFFIAny argument;

    memset(&argument, 0, sizeof(argument));
    argument.type_index = kTVMFFIInt;
    argument.v_int64 = (int64_t)value;
    return argument;
}}

static SparkStatus SparkGlm52B12xInvoke(
    SparkGlm52B12xTvmFunction function,
    const char *function_name,
    const TVMFFIAny *arguments,
    int32_t argument_count)
{{
    TVMFFIAny result;
    int status;

    memset(&result, 0, sizeof(result));
    result.type_index = kTVMFFINone;
    status = function(0, arguments, argument_count, &result);
    if (status != 0)
    {{
        fprintf(
            stderr,
            "b12x_tvm_ffi_launch_failed function=%s status=%d argument_count=%d\\n",
            function_name != 0 ? function_name : "unknown",
            status,
            argument_count);
        return SPARK_STATUS_INTERNAL_ERROR;
    }}
    if (result.type_index >= kTVMFFIObject)
    {{
        TVMFFIObjectDecRef(result.v_obj);
    }}
    return SPARK_STATUS_OK;
}}
'''
    functions: List[str] = [helper_source]
    for bucket in buckets:
        functions.append(generate_bucket_launch_function(bucket))
    dispatch_source = f'''
SparkStatus SparkGlm52Sm121B12xGeneratedLaunch(
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    const SparkGlm52Sm121B12xGeneratedLaunchArguments *arguments)
{{
    if (bucket == 0 || arguments == 0)
    {{
        return SPARK_STATUS_INVALID_ARGUMENT;
    }}
    if (arguments->token_count > bucket->token_upper_bound)
    {{
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }}

    switch (bucket->token_upper_bound)
    {{
{chr(10).join(switch_cases)}
        default:
            return SPARK_STATUS_CAPACITY_EXCEEDED;
    }}
}}
'''
    functions.append(dispatch_source)
    output_path.write_text("\n".join(functions))


def generate_bucket_launch_function(bucket: Dict[str, Any]) -> str:
    token_count = int(bucket["token_upper_bound"])
    kind = bucket["backend_kind"]
    function_name = bucket["function_name"]
    c_name = f"SparkGlm52B12xLaunch{kind.capitalize()}T{token_count}"
    if kind in ("static", "micro"):
        return generate_static_launch_function(c_name, function_name, token_count, int(bucket["max_rows"]))
    if kind == "dynamic":
        return generate_dynamic_launch_function(c_name, function_name, token_count, int(bucket["max_rows"]), int(bucket["physical_tile_capacity"]), int(bucket["task_capacity"]))
    raise AotFailure(f"unsupported backend kind {kind}")


def generate_static_launch_function(c_name: str, function_name: str, token_count: int, max_rows: int) -> str:
    hidden = REQUIRED_SHAPE["hidden_dimension"]
    intermediate = REQUIRED_SHAPE["intermediate_dimension"]
    experts = REQUIRED_SHAPE["expert_count"]
    top_k = REQUIRED_SHAPE["top_k"]
    routed_rows = token_count * top_k
    w1_rows = 2 * intermediate
    rows_pad_k = align_up(max_rows, 128)
    cols_pad_k = align_up(hidden // 16, 4)
    return f'''
static SparkStatus {c_name}(
    const SparkGlm52Sm121B12xGeneratedLaunchArguments *arguments)
{{
    DLDataType bf16_type;
    DLDataType fp4_type;
    DLDataType uint8_type;
    DLDataType int32_type;
    DLDataType float32_type;
    DLTensor tensors[20];
    TVMFFIAny call_arguments[25];
    int64_t hidden_shape[2] = {{{token_count}, {hidden}}};
    int64_t hidden_strides[2] = {{{hidden}, 1}};
    int64_t routed_shape[1] = {{{routed_rows}}};
    int64_t routed_strides[1] = {{1}};
    int64_t packed_a_shape[3] = {{{max_rows}, {hidden // 2}, {experts}}};
    int64_t packed_a_strides[3] = {{{hidden // 2}, 1, {max_rows * (hidden // 2)}}};
    int64_t packed_a_flat_shape[1] = {{{experts * max_rows * (hidden // 2)}}};
    int64_t packed_a_flat_strides[1] = {{1}};
    int64_t scale_flat_shape[1] = {{{experts * rows_pad_k * cols_pad_k}}};
    int64_t scale_flat_strides[1] = {{1}};
    int64_t scalar_shape[1] = {{1}};
    int64_t scalar_strides[1] = {{1}};
    int64_t w1_shape[3] = {{{w1_rows}, {hidden // 2}, {experts}}};
    int64_t w1_strides[3] = {{{hidden // 2}, 1, {w1_rows * (hidden // 2)}}};
    int64_t w2_shape[3] = {{{hidden}, {intermediate // 2}, {experts}}};
    int64_t w2_strides[3] = {{{intermediate // 2}, 1, {hidden * (intermediate // 2)}}};
    int64_t expert_shape[1] = {{{experts}}};
    int64_t expert_strides[1] = {{1}};
    int64_t token_map_shape[2] = {{{experts}, {max_rows}}};
    int64_t token_map_strides[2] = {{{max_rows}, 1}};

    bf16_type = SparkGlm52B12xDataType(kDLBfloat, 16, 1);
    fp4_type = SparkGlm52B12xDataType(kDLFloat4_e2m1fn, 4, 2);
    uint8_type = SparkGlm52B12xDataType(kDLUInt, 8, 1);
    int32_type = SparkGlm52B12xDataType(kDLInt, 32, 1);
    float32_type = SparkGlm52B12xDataType(kDLFloat, 32, 1);

    SparkGlm52B12xFillTensor(&tensors[0], (void *)arguments->hidden_bf16, bf16_type, 2, hidden_shape, hidden_strides);
    SparkGlm52B12xFillTensor(&tensors[1], (void *)arguments->topk_ids_i32, int32_type, 1, routed_shape, routed_strides);
    SparkGlm52B12xFillTensor(&tensors[2], (void *)arguments->topk_weights_fp32, float32_type, 1, routed_shape, routed_strides);
    SparkGlm52B12xFillTensor(&tensors[3], arguments->generated_workspace->packed_input_u8, fp4_type, 3, packed_a_shape, packed_a_strides);
    SparkGlm52B12xFillTensor(&tensors[4], arguments->generated_workspace->packed_input_u8, uint8_type, 1, packed_a_flat_shape, packed_a_flat_strides);
    SparkGlm52B12xFillTensor(&tensors[5], arguments->generated_workspace->packed_input_scale_u8, uint8_type, 1, scale_flat_shape, scale_flat_strides);
    SparkGlm52B12xFillTensor(&tensors[6], arguments->generated_workspace->barrier_count_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[7], arguments->generated_workspace->barrier_epoch_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[8], (void *)arguments->w1_weight_fp4_static_view, fp4_type, 3, w1_shape, w1_strides);
    SparkGlm52B12xFillTensor(&tensors[9], (void *)arguments->w2_weight_fp4_static_view, fp4_type, 3, w2_shape, w2_strides);
    SparkGlm52B12xFillTensor(&tensors[10], arguments->generated_workspace->row_counts_i32, int32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[11], arguments->generated_workspace->active_expert_count_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[12], arguments->generated_workspace->weight_expert_ids_i32, int32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[13], arguments->generated_workspace->global_to_local_expert_i32, int32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[14], (void *)arguments->w1_alpha_fp32_by_expert, float32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[15], (void *)arguments->w2_alpha_fp32_by_expert, float32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[16], (void *)arguments->fc2_input_scale_fp32_by_expert, float32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[17], arguments->output_bf16, bf16_type, 2, hidden_shape, hidden_strides);
    SparkGlm52B12xFillTensor(&tensors[18], arguments->generated_workspace->token_map_i32, int32_type, 2, token_map_shape, token_map_strides);
    SparkGlm52B12xFillTensor(&tensors[19], arguments->generated_workspace->token_weights_fp32, float32_type, 2, token_map_shape, token_map_strides);

    call_arguments[0] = SparkGlm52B12xTensorArgument(&tensors[0]);
    call_arguments[1] = SparkGlm52B12xTensorArgument(&tensors[1]);
    call_arguments[2] = SparkGlm52B12xTensorArgument(&tensors[2]);
    call_arguments[3] = SparkGlm52B12xTensorArgument(&tensors[3]);
    call_arguments[4] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->packed_input_scale_u8);
    call_arguments[5] = SparkGlm52B12xTensorArgument(&tensors[4]);
    call_arguments[6] = SparkGlm52B12xTensorArgument(&tensors[5]);
    call_arguments[7] = SparkGlm52B12xTensorArgument(&tensors[6]);
    call_arguments[8] = SparkGlm52B12xTensorArgument(&tensors[7]);
    call_arguments[9] = SparkGlm52B12xTensorArgument(&tensors[8]);
    call_arguments[10] = SparkGlm52B12xPointerArgument(arguments->w1_scale_static_storage_ue4m3);
    call_arguments[11] = SparkGlm52B12xTensorArgument(&tensors[9]);
    call_arguments[12] = SparkGlm52B12xPointerArgument(arguments->w2_scale_static_storage_ue4m3);
    call_arguments[13] = SparkGlm52B12xTensorArgument(&tensors[10]);
    call_arguments[14] = SparkGlm52B12xTensorArgument(&tensors[11]);
    call_arguments[15] = SparkGlm52B12xTensorArgument(&tensors[12]);
    call_arguments[16] = SparkGlm52B12xTensorArgument(&tensors[13]);
    call_arguments[17] = SparkGlm52B12xTensorArgument(&tensors[14]);
    call_arguments[18] = SparkGlm52B12xTensorArgument(&tensors[14]);
    call_arguments[19] = SparkGlm52B12xTensorArgument(&tensors[15]);
    call_arguments[20] = SparkGlm52B12xTensorArgument(&tensors[16]);
    call_arguments[21] = SparkGlm52B12xTensorArgument(&tensors[17]);
    call_arguments[22] = SparkGlm52B12xTensorArgument(&tensors[18]);
    call_arguments[23] = SparkGlm52B12xTensorArgument(&tensors[19]);
    call_arguments[24] = SparkGlm52B12xPointerArgument(arguments->cuda_stream);

    return SparkGlm52B12xInvoke(__tvm_ffi_{function_name}, "{function_name}", call_arguments, 25);
}}
'''


def generate_dynamic_launch_function(c_name: str, function_name: str, token_count: int, max_rows: int, physical_tiles: int, task_capacity: int) -> str:
    hidden = REQUIRED_SHAPE["hidden_dimension"]
    intermediate = REQUIRED_SHAPE["intermediate_dimension"]
    experts = REQUIRED_SHAPE["expert_count"]
    w1_rows = 2 * intermediate
    return f'''
static SparkStatus {c_name}(
    const SparkGlm52Sm121B12xGeneratedLaunchArguments *arguments)
{{
    DLDataType fp4_type;
    DLDataType int32_type;
    DLDataType float32_type;
    DLTensor tensors[16];
    TVMFFIAny call_arguments[41];
    int64_t scalar_shape[1] = {{1}};
    int64_t scalar_strides[1] = {{1}};
    int64_t w1_shape[3] = {{{w1_rows}, {hidden // 2}, {experts}}};
    int64_t w1_strides[3] = {{{hidden // 2}, 1, {w1_rows * (hidden // 2)}}};
    int64_t w2_shape[3] = {{{hidden}, {intermediate // 2}, {experts}}};
    int64_t w2_strides[3] = {{{intermediate // 2}, 1, {hidden * (intermediate // 2)}}};
    int64_t expert_shape[1] = {{{experts}}};
    int64_t expert_strides[1] = {{1}};
    int64_t expert_plus_one_shape[1] = {{{experts + 1}}};
    int64_t expert_plus_one_strides[1] = {{1}};

    fp4_type = SparkGlm52B12xDataType(kDLFloat4_e2m1fn, 4, 2);
    int32_type = SparkGlm52B12xDataType(kDLInt, 32, 1);
    float32_type = SparkGlm52B12xDataType(kDLFloat, 32, 1);

    SparkGlm52B12xFillTensor(&tensors[0], arguments->generated_workspace->barrier_count_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[1], arguments->generated_workspace->barrier_epoch_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[2], arguments->generated_workspace->pair_head_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[3], arguments->generated_workspace->producers_done_count_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[4], arguments->generated_workspace->all_work_published_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[5], arguments->generated_workspace->task_head_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[6], arguments->generated_workspace->task_tail_i32, int32_type, 1, scalar_shape, scalar_strides);
    SparkGlm52B12xFillTensor(&tensors[7], (void *)arguments->w1_weight_fp4_static_view, fp4_type, 3, w1_shape, w1_strides);
    SparkGlm52B12xFillTensor(&tensors[8], (void *)arguments->w2_weight_fp4_static_view, fp4_type, 3, w2_shape, w2_strides);
    SparkGlm52B12xFillTensor(&tensors[9], arguments->generated_workspace->row_counts_i32, int32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[10], arguments->generated_workspace->expert_write_rows_i32, int32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[11], arguments->generated_workspace->expert_tile_base_i32, int32_type, 1, expert_plus_one_shape, expert_plus_one_strides);
    SparkGlm52B12xFillTensor(&tensors[12], (void *)arguments->w1_alpha_fp32_by_expert, float32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[13], (void *)arguments->w1_alpha_fp32_by_expert, float32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[14], (void *)arguments->w2_alpha_fp32_by_expert, float32_type, 1, expert_shape, expert_strides);
    SparkGlm52B12xFillTensor(&tensors[15], (void *)arguments->fc2_input_scale_fp32_by_expert, float32_type, 1, expert_shape, expert_strides);

    call_arguments[0] = SparkGlm52B12xPointerArgument(arguments->hidden_bf16);
    call_arguments[1] = SparkGlm52B12xPointerArgument(arguments->topk_ids_i32);
    call_arguments[2] = SparkGlm52B12xPointerArgument(arguments->topk_weights_fp32);
    call_arguments[3] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->packed_input_u8);
    call_arguments[4] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->packed_input_scale_u8);
    call_arguments[5] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->packed_input_u8);
    call_arguments[6] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->packed_input_scale_u8);
    call_arguments[7] = SparkGlm52B12xTensorArgument(&tensors[0]);
    call_arguments[8] = SparkGlm52B12xTensorArgument(&tensors[1]);
    call_arguments[9] = SparkGlm52B12xTensorArgument(&tensors[2]);
    call_arguments[10] = SparkGlm52B12xTensorArgument(&tensors[3]);
    call_arguments[11] = SparkGlm52B12xTensorArgument(&tensors[4]);
    call_arguments[12] = SparkGlm52B12xTensorArgument(&tensors[5]);
    call_arguments[13] = SparkGlm52B12xTensorArgument(&tensors[6]);
    call_arguments[14] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->task_ready_i32);
    call_arguments[15] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->task_expert_i32);
    call_arguments[16] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->task_m_tile_i32);
    call_arguments[17] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->task_slice_begin_i32);
    call_arguments[18] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->task_slice_count_i32);
    call_arguments[19] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->task_valid_rows_i32);
    call_arguments[20] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->tile_write_count_i32);
    call_arguments[21] = SparkGlm52B12xTensorArgument(&tensors[7]);
    call_arguments[22] = SparkGlm52B12xPointerArgument(arguments->w1_scale_static_storage_ue4m3);
    call_arguments[23] = SparkGlm52B12xTensorArgument(&tensors[8]);
    call_arguments[24] = SparkGlm52B12xPointerArgument(arguments->w2_scale_static_storage_ue4m3);
    call_arguments[25] = SparkGlm52B12xTensorArgument(&tensors[9]);
    call_arguments[26] = SparkGlm52B12xTensorArgument(&tensors[10]);
    call_arguments[27] = SparkGlm52B12xTensorArgument(&tensors[11]);
    call_arguments[28] = SparkGlm52B12xTensorArgument(&tensors[12]);
    call_arguments[29] = SparkGlm52B12xTensorArgument(&tensors[13]);
    call_arguments[30] = SparkGlm52B12xTensorArgument(&tensors[14]);
    call_arguments[31] = SparkGlm52B12xTensorArgument(&tensors[15]);
    call_arguments[32] = SparkGlm52B12xPointerArgument(arguments->output_bf16);
    call_arguments[33] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->token_map_i32);
    call_arguments[34] = SparkGlm52B12xPointerArgument(arguments->generated_workspace->token_weights_fp32);
    call_arguments[35] = SparkGlm52B12xIntegerArgument(arguments->token_count);
    call_arguments[36] = SparkGlm52B12xIntegerArgument({max_rows});
    call_arguments[37] = SparkGlm52B12xIntegerArgument({physical_tiles * 128});
    call_arguments[38] = SparkGlm52B12xIntegerArgument({task_capacity});
    call_arguments[39] = SparkGlm52B12xIntegerArgument({physical_tiles});
    call_arguments[40] = SparkGlm52B12xPointerArgument(arguments->cuda_stream);

    return SparkGlm52B12xInvoke(__tvm_ffi_{function_name}, "{function_name}", call_arguments, 41);
}}
'''


def build_manifest(exported: Dict[str, Dict[str, Any]], bucket_results: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    buckets: List[Dict[str, Any]] = []
    for result in bucket_results:
        token_count = int(result["tokens"])
        kind = str(result["selected_backend"])
        function_name, exported_kind = find_export_for_bucket(exported, kind, token_count)
        geometry = bucket_geometry(exported_kind, token_count)
        bucket = {
            "token_upper_bound": token_count,
            "backend_kind": exported_kind,
            "function_name": function_name,
            "max_active_clusters": int(result.get("max_active_clusters", 0)),
            "static_mma_tile_m": 0,
            "static_mma_tile_n": 0,
            "avg_us": int(round(float(result.get("avg_us", 0.0)))),
            "p95_us": int(round(float(result.get("p95_us", 0.0)))),
        }
        bucket.update(geometry)
        buckets.append(bucket)
    manifest: Dict[str, Any] = {
        "record_schema": "sparkpipe.glm52.sm121.b12x.aot_manifest.v1",
        "required_module": "spark.glm52.sm121.flashinfer_b12x_fused_moe.nvfp4.bf16.v2",
        "required_arch": "sm_121a",
        "runtime_language": "c_cuda_tvm_ffi",
        "compile_time_languages": ["python", "torch", "flashinfer", "cutlass_cute_dsl"],
        "fallback_allowed": False,
        "runtime_backend_selection": "forbidden",
        "shape": REQUIRED_SHAPE,
        "maximum_token_count": max(int(bucket["token_upper_bound"]) for bucket in buckets),
        "buckets": buckets,
        "exported_objects": exported,
    }
    digest_input = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = sha256_text(digest_input)
    manifest["manifest_hash_sha256"] = digest
    manifest["manifest_hash_low64"] = low64_from_hex(digest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,96,128")
    parser.add_argument("--output-dir", default="build/glm52_b12x_aot")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--disable-micro", action="store_true")
    parser.add_argument("--allow-dynamic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    output_dir = (root / args.output_dir).resolve()
    generated_dir = output_dir / "generated"
    objects_dir = generated_dir / "objects"
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)
    objects_dir.mkdir(parents=True, exist_ok=True)
    token_buckets = parse_tokens(args.tokens)

    if args.dry_run:
        dry_manifest = {
            "record_schema": "sparkpipe.glm52.sm121.b12x.aot_manifest.v1.dry_run",
            "tokens": token_buckets,
            "shape": REQUIRED_SHAPE,
            "runtime_python": False,
            "runtime_torch": False,
        }
        (generated_dir / "aot_manifest.json").write_text(json.dumps(dry_manifest, indent=2, sort_keys=True) + "\n")
        return 0

    install_vendored_flashinfer(root, output_dir / "flashinfer_cache")
    if args.disable_micro:
        os.environ.setdefault("FLASHINFER_B12X_MICRO_SHARE_INPUT", "0")
    if not args.allow_dynamic:
        maximum_routed_rows = max(token_buckets) * REQUIRED_SHAPE["top_k"]
        os.environ.setdefault(
            "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
            str(maximum_routed_rows),
        )
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    import torch
    from flashinfer.fused_moe.cute_dsl.b12x_moe import B12xMoEWrapper
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as moe_dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import select_sm120_moe_backend

    if args.disable_micro:
        moe_dispatch._MICRO_COMPACT_CUTOVER_PAIRS = 0
        moe_dispatch._MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK = 0

    require_sm121(torch)
    weights = make_weights(torch)
    wrapper = B12xMoEWrapper(
        num_experts=REQUIRED_SHAPE["expert_count"],
        top_k=REQUIRED_SHAPE["top_k"],
        hidden_size=REQUIRED_SHAPE["hidden_dimension"],
        intermediate_size=REQUIRED_SHAPE["intermediate_dimension"],
        use_cuda_graph=False,
        max_num_tokens=max(token_buckets),
        output_dtype=torch.bfloat16,
        device="cuda",
        activation="silu",
        quant_mode="nvfp4",
        source_format="modelopt",
    )

    bucket_results: List[Dict[str, Any]] = []
    for token_count in token_buckets:
        warm_bucket(torch, wrapper, weights, token_count)
        selected_backend = select_sm120_moe_backend(
            num_tokens=token_count,
            num_topk=REQUIRED_SHAPE["top_k"],
            quant_mode="nvfp4",
        )
        if args.disable_micro and selected_backend == "micro":
            selected_backend = "static"
        result: Dict[str, Any] = {
            "tokens": token_count,
            "selected_backend": selected_backend,
        }
        if args.benchmark:
            result.update(time_bucket(torch, wrapper, weights, token_count, args.warmup, args.iterations))
        bucket_results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    exported = export_compiled_objects(moe_dispatch, objects_dir)
    manifest = build_manifest(exported, bucket_results, output_dir)
    manifest_path = generated_dir / "aot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    generate_launch_table_source(manifest, exported, generated_dir / "spark_glm52_sm121_b12x_generated_kernel_table.cu")
    write_tvm_ffi_flags(generated_dir)
    print(f"wrote {manifest_path}")
    print(f"manifest_hash_low64=0x{manifest['manifest_hash_low64']:016x}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AotFailure as error:
        print(f"glm52_b12x_aot_compile: {error}", file=sys.stderr)
        raise SystemExit(2)
