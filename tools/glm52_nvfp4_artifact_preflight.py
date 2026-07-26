#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import sys
from typing import Any

import glm52_b12x_aot_compile as aot
import glm52_b12x_resident_pack as b12x
import glm52_stagepack_artifact_preflight as stagepack


NVFP4_QUANTIZATION = "nvfp4"
NVFP4_LABEL = "NVFP4"
NVFP4_MANIFEST = "resident_moe_pack_manifest.json"
FOREIGN_MANIFESTS = (
    "fp8_moe_pack_manifest.json",
    "w8lut_moe_pack_manifest.json",
)
AOT_SCHEMA = "sparkpipe.glm52.sm121.b12x.aot_manifest.v1"
PREFLIGHT_SCHEMA = "sparkpipe.glm52.nvfp4.artifact_preflight.v1"
GENERATED_SOURCE = "spark_glm52_sm121_b12x_generated_kernel_table.cu"
GENERATED_FLAGS = "tvm_ffi_flags.mk"
RUNTIME_LINK_ARGS = "runtime_link_args.txt"
MTP_ROWS_PER_LANE = 6
MAX_ROWS_PER_LANE = 8


PreflightFailure = stagepack.PreflightFailure


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PreflightFailure(f"{label} must be a boolean")
    return value


def low64_from_sha256(digest: str) -> int:
    return int(digest[:16], 16)


def hash_aot_manifest(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("manifest_hash_sha256", None)
    payload.pop("manifest_hash_low64", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_aot_shape(value: Any) -> None:
    shape = stagepack.require_object(value, "AOT shape")
    expected = {
        "hidden_dimension": b12x.HIDDEN_DIMENSION,
        "intermediate_dimension": b12x.INTERMEDIATE_DIMENSION,
        "expert_count": b12x.EXPERT_COUNT,
        "top_k": b12x.TOP_K,
        "fused_w1_rows": (
            b12x.W1_COMPONENT_COUNT * b12x.INTERMEDIATE_DIMENSION
        ),
    }
    stagepack.require_equal(shape, expected, "AOT shape")


def validate_aot_buckets(
    document: dict[str, Any],
    maximum_token_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    values = stagepack.require_list(document.get("buckets"), "AOT buckets")
    exported = stagepack.require_object(
        document.get("exported_objects"),
        "AOT exported_objects",
    )
    if not values:
        raise PreflightFailure("AOT buckets must not be empty")
    buckets: list[dict[str, Any]] = []
    observed_bounds: list[int] = []
    for value in values:
        bucket = stagepack.require_object(value, "AOT bucket")
        token_count = stagepack.require_int(
            bucket.get("token_upper_bound"),
            "AOT bucket token_upper_bound",
            1,
        )
        backend_kind = stagepack.require_string(
            bucket.get("backend_kind"),
            "AOT bucket backend_kind",
        )
        if backend_kind != "static":
            raise PreflightFailure(
                "production NVFP4 requires exact static AOT buckets; "
                f"found {backend_kind!r} for {token_count} rows")
        function_name = stagepack.require_string(
            bucket.get("function_name"),
            "AOT bucket function_name",
        )
        if function_name not in exported:
            raise PreflightFailure(
                f"AOT bucket function {function_name} is not exported")
        routed_rows = token_count * b12x.TOP_K
        stagepack.require_equal(
            bucket.get("routed_rows_capacity"),
            routed_rows,
            f"AOT bucket {token_count} routed row capacity",
        )
        stagepack.require_equal(
            bucket.get("max_rows"),
            routed_rows,
            f"AOT bucket {token_count} max_rows",
        )
        stagepack.require_equal(
            bucket.get("physical_tile_capacity"),
            0,
            f"AOT bucket {token_count} physical tile capacity",
        )
        stagepack.require_equal(
            bucket.get("task_capacity"),
            0,
            f"AOT bucket {token_count} task capacity",
        )
        stagepack.require_equal(
            bucket.get("route_output_slice_count"),
            (b12x.INTERMEDIATE_DIMENSION + 127) // 128,
            f"AOT bucket {token_count} route output slices",
        )
        observed_bounds.append(token_count)
        buckets.append(bucket)
    if observed_bounds != sorted(set(observed_bounds)):
        raise PreflightFailure(
            "AOT token buckets must be unique and sorted ascending")
    stagepack.require_equal(
        observed_bounds[-1],
        maximum_token_count,
        "AOT maximum token bucket",
    )
    return buckets, {
        str(name): stagepack.require_object(record, f"AOT object {name}")
        for name, record in exported.items()
    }


def validate_generated_aot_files(
    aot_manifest_path: Path,
    buckets: list[dict[str, Any]],
    exported: dict[str, dict[str, Any]],
    manifest_hash_low64: int,
) -> dict[str, Any]:
    generated_root = aot_manifest_path.parent
    source_path = generated_root / GENERATED_SOURCE
    flags_path = generated_root / GENERATED_FLAGS
    link_args_path = generated_root / RUNTIME_LINK_ARGS
    for path, label in (
        (source_path, "generated B12x launch table"),
        (flags_path, "generated TVM FFI flags"),
        (link_args_path, "generated runtime link arguments"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise PreflightFailure(f"missing or empty {label}: {path}")
    source = source_path.read_text(encoding="utf-8")
    if f"0x{manifest_hash_low64:016x}" not in source:
        raise PreflightFailure(
            "generated B12x launch table does not embed the AOT manifest hash")
    for bucket in buckets:
        function_name = stagepack.require_string(
            bucket.get("function_name"),
            "AOT bucket function_name",
        )
        if function_name not in source:
            raise PreflightFailure(
                f"generated B12x launch table is missing {function_name}")
    flags = flags_path.read_text(encoding="utf-8")
    for variable in (
        "TVM_FFI_CFLAGS :=",
        "TVM_FFI_LDFLAGS :=",
        "TVM_FFI_LIBS :=",
        "CUTE_TVM_FFI_RUNTIME_LIBS :=",
    ):
        if variable not in flags:
            raise PreflightFailure(
                f"generated TVM FFI flags are missing {variable}")
    link_arguments = shlex.split(link_args_path.read_text(encoding="utf-8"))
    if not link_arguments:
        raise PreflightFailure("generated runtime link arguments are empty")
    missing_runtime_paths = [
        argument
        for argument in link_arguments
        if argument.startswith("/") and not Path(argument).exists()
    ]
    if missing_runtime_paths:
        raise PreflightFailure(
            f"runtime link path does not exist: {missing_runtime_paths[0]}")
    object_receipts: list[dict[str, Any]] = []
    for function_name, record in sorted(exported.items()):
        stagepack.require_equal(
            record.get("kind"),
            "static",
            f"AOT object {function_name} backend kind",
        )
        object_name = stagepack.require_string(
            record.get("object"),
            f"AOT object {function_name} file",
        )
        if Path(object_name).name != object_name:
            raise PreflightFailure(
                f"AOT object {function_name} file must be a basename")
        path = generated_root / "objects" / object_name
        if not path.is_file() or path.stat().st_size == 0:
            raise PreflightFailure(f"missing exported AOT object {path}")
        expected_sha256 = stagepack.require_sha256(
            record.get("sha256"),
            f"AOT object {function_name} sha256",
        )
        observed_sha256 = stagepack.sha256_file(path)
        stagepack.require_equal(
            observed_sha256,
            expected_sha256,
            f"AOT object {function_name} sha256",
        )
        object_receipts.append({
            "function_name": function_name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed_sha256,
        })
    return {
        "generated_root": str(generated_root),
        "launch_table": str(source_path),
        "runtime_link_args": str(link_args_path),
        "object_count": len(object_receipts),
        "objects": object_receipts,
    }


def validate_aot_manifest(
    path: Path,
    expected_manifest_hash_low64: int,
    expected_maximum_token_count: int,
) -> tuple[str, dict[str, Any]]:
    document = stagepack.load_json_object(path, "B12x AOT manifest")
    stagepack.require_equal(
        document.get("record_schema"),
        AOT_SCHEMA,
        "AOT manifest schema",
    )
    stagepack.require_equal(
        document.get("required_module"),
        b12x.REQUIRED_MODULE,
        "AOT required module",
    )
    stagepack.require_equal(
        document.get("required_arch"),
        b12x.REQUIRED_ARCH,
        "AOT required arch",
    )
    stagepack.require_equal(
        require_bool(document.get("fallback_allowed"), "AOT fallback_allowed"),
        False,
        "AOT fallback_allowed",
    )
    stagepack.require_equal(
        document.get("runtime_backend_selection"),
        "forbidden",
        "AOT runtime backend selection",
    )
    stagepack.require_equal(
        document.get("production_backend_policy"),
        "exact_static_buckets_only",
        "AOT production backend policy",
    )
    stagepack.require_equal(
        document.get("runtime_bucket_decomposition"),
        "forbidden",
        "AOT runtime bucket decomposition",
    )
    stagepack.require_equal(
        document.get("runtime_diagnostic_routing_mutation"),
        "forbidden",
        "AOT runtime diagnostic routing mutation",
    )
    stagepack.require_equal(
        require_bool(
            document.get("deterministic_fc2_finalize"),
            "AOT deterministic_fc2_finalize",
        ),
        True,
        "AOT deterministic FC2 finalize",
    )
    validate_aot_shape(document.get("shape"))
    maximum_token_count = stagepack.require_int(
        document.get("maximum_token_count"),
        "AOT maximum_token_count",
        1,
    )
    stagepack.require_equal(
        maximum_token_count,
        expected_maximum_token_count,
        "resident/AOT maximum token count",
    )
    manifest_sha256 = stagepack.require_sha256(
        document.get("manifest_hash_sha256"),
        "AOT manifest_hash_sha256",
    )
    stagepack.require_equal(
        hash_aot_manifest(document),
        manifest_sha256,
        "AOT manifest content hash",
    )
    manifest_hash_low64 = stagepack.require_int(
        document.get("manifest_hash_low64"),
        "AOT manifest_hash_low64",
        1,
    )
    stagepack.require_equal(
        manifest_hash_low64,
        low64_from_sha256(manifest_sha256),
        "AOT manifest low64",
    )
    stagepack.require_equal(
        manifest_hash_low64,
        expected_manifest_hash_low64,
        "resident/AOT manifest low64",
    )
    buckets, exported = validate_aot_buckets(
        document,
        maximum_token_count,
    )
    generated = validate_generated_aot_files(
        path,
        buckets,
        exported,
        manifest_hash_low64,
    )
    return stagepack.sha256_file(path), {
        "path": str(path),
        "manifest_hash_sha256": manifest_sha256,
        "manifest_hash_low64": manifest_hash_low64,
        "maximum_token_count": maximum_token_count,
        "bucket_count": len(buckets),
        "buckets": [
            stagepack.require_int(
                bucket.get("token_upper_bound"),
                "AOT bucket token_upper_bound",
                1,
            )
            for bucket in buckets
        ],
        **generated,
    }


def parse_pack_header(payload: bytes) -> tuple[tuple[Any, ...], list[dict[str, int]]]:
    if len(payload) != b12x.HEADER_BYTES:
        raise PreflightFailure("short B12x pack header")
    prefix_bytes = struct.calcsize(b12x.HEADER_PREFIX_FORMAT)
    fields = struct.unpack(
        b12x.HEADER_PREFIX_FORMAT,
        payload[:prefix_bytes],
    )
    regions: list[dict[str, int]] = []
    offset = prefix_bytes
    region_bytes = struct.calcsize(b12x.REGION_FORMAT)
    for _ in range(b12x.REGION_COUNT):
        region_offset, byte_count = struct.unpack(
            b12x.REGION_FORMAT,
            payload[offset:offset + region_bytes],
        )
        regions.append({
            "offset": int(region_offset),
            "bytes": int(byte_count),
        })
        offset += region_bytes
    if any(payload[offset:]):
        raise PreflightFailure("B12x pack header reserved bytes are not zero")
    return fields, regions


def require_nonzero_region_samples(
    file_descriptor: int,
    regions: list[dict[str, int]],
    sample_bytes: int,
    layer: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    sampled = 0
    nonzero = 0
    for region_index in (
        b12x.REGION_W1_WEIGHT,
        b12x.REGION_W1_SCALE,
        b12x.REGION_W2_WEIGHT,
        b12x.REGION_W2_SCALE,
    ):
        region = regions[region_index]
        width = min(sample_bytes, region["bytes"])
        for relative_offset in stagepack.sample_offsets(
            region["bytes"],
            sample_bytes,
        ):
            payload = stagepack.read_exact_at(
                file_descriptor,
                region["offset"] + relative_offset,
                width,
                f"B12x layer {layer} region {region_index}",
            )
            digest.update(struct.pack(
                "<IQQ",
                region_index,
                relative_offset,
                len(payload),
            ))
            digest.update(payload)
            sampled += len(payload)
            nonzero += sum(value != 0 for value in payload)
    if sampled == 0 or nonzero == 0:
        raise PreflightFailure(
            f"B12x layer {layer} weight/scale samples are entirely zero")
    one_values = struct.pack("<" + ("f" * b12x.EXPERT_COUNT),
                             *([1.0] * b12x.EXPERT_COUNT))
    for region_index in (
        b12x.REGION_W1_ALPHA,
        b12x.REGION_FC2_INPUT_SCALE,
        b12x.REGION_W2_ALPHA,
    ):
        region = regions[region_index]
        stagepack.require_equal(
            region["bytes"],
            len(one_values),
            f"B12x layer {layer} scalar region {region_index} bytes",
        )
        stagepack.require_equal(
            stagepack.read_exact_at(
                file_descriptor,
                region["offset"],
                region["bytes"],
                f"B12x layer {layer} scalar region {region_index}",
            ),
            one_values,
            f"B12x layer {layer} scalar region {region_index}",
        )
    return {
        "sample_bytes": sampled,
        "sample_nonzero_bytes": nonzero,
        "sample_sha256": digest.hexdigest(),
    }


def validate_pack(
    root: Path,
    record: dict[str, Any],
    layer: int,
    maximum_token_count: int,
    kernel_manifest_hash_low64: int,
    aot_file_sha256: str,
    sample_bytes: int,
    verify_sha256: bool,
) -> dict[str, Any]:
    stagepack.require_equal(
        record.get("layer_index"),
        layer,
        f"B12x layer {layer} manifest layer",
    )
    expected_name = f"glm52_layer_{layer:04d}_b12x_moe.spb12x"
    recorded_path = Path(stagepack.require_string(
        record.get("path"),
        f"B12x layer {layer} path",
    ))
    stagepack.require_equal(
        recorded_path.name,
        expected_name,
        f"B12x layer {layer} file",
    )
    path = root / expected_name
    if not path.is_file():
        raise PreflightFailure(f"missing B12x layer pack {path}")
    file_bytes = path.stat().st_size
    stagepack.require_equal(
        record.get("bytes"),
        file_bytes,
        f"B12x layer {layer} manifest bytes",
    )
    expected_sha256 = stagepack.require_sha256(
        record.get("sha256"),
        f"B12x layer {layer} sha256",
    )
    expected_regions = b12x.reserve_regions()
    expected_file_bytes = (
        expected_regions[-1]["offset"] + expected_regions[-1]["bytes"]
    )
    stagepack.require_equal(
        file_bytes,
        expected_file_bytes,
        f"B12x layer {layer} file bytes",
    )
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        header = stagepack.read_exact_at(
            file_descriptor,
            0,
            b12x.HEADER_BYTES,
            f"B12x layer {layer} header",
        )
        fields, regions = parse_pack_header(header)
        fixed_fields = (
            b12x.MAGIC,
            b12x.ABI_VERSION,
            b12x.HEADER_BYTES,
            layer,
            maximum_token_count,
            b12x.HIDDEN_DIMENSION,
            b12x.INTERMEDIATE_DIMENSION,
            b12x.EXPERT_COUNT,
            b12x.TOP_K,
            b12x.GATE_UP_ORDER_UP_GATE,
            b12x.WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW,
            b12x.SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE,
            b12x.QUANT_MODE_NVFP4,
            b12x.OUTPUT_DTYPE_BF16,
            b12x.CUDA_ARCHITECTURE_SM121,
            0,
            0,
        )
        stagepack.require_equal(
            fields[:17],
            fixed_fields,
            f"B12x layer {layer} fixed header",
        )
        qualified_us = int(fields[17])
        qualification_hash_low64 = int(fields[18])
        observed_kernel_hash_low64 = int(fields[19])
        pack_hash_low64 = int(fields[20])
        if qualified_us <= 0 or qualification_hash_low64 == 0:
            raise PreflightFailure(
                f"B12x layer {layer} qualification metadata is empty")
        stagepack.require_equal(
            qualification_hash_low64,
            low64_from_sha256(aot_file_sha256),
            f"B12x layer {layer} AOT file hash",
        )
        stagepack.require_equal(
            observed_kernel_hash_low64,
            kernel_manifest_hash_low64,
            f"B12x layer {layer} kernel manifest hash",
        )
        stagepack.require_equal(
            pack_hash_low64,
            b12x.pack_metadata_hash_low64(
                layer,
                expected_regions,
                maximum_token_count,
                qualified_us,
                qualification_hash_low64,
                kernel_manifest_hash_low64,
            ),
            f"B12x layer {layer} pack metadata hash",
        )
        if pack_hash_low64 == 0:
            raise PreflightFailure(
                f"B12x layer {layer} pack metadata hash is zero")
        stagepack.require_equal(
            regions,
            expected_regions,
            f"B12x layer {layer} regions",
        )
        samples = require_nonzero_region_samples(
            file_descriptor,
            regions,
            sample_bytes,
            layer,
        )
    finally:
        os.close(file_descriptor)
    stagepack.require_equal(
        record.get("kernel_manifest_hash_low64"),
        kernel_manifest_hash_low64,
        f"B12x layer {layer} manifest kernel hash",
    )
    stagepack.require_equal(
        record.get("qualification_record_hash_low64"),
        low64_from_sha256(aot_file_sha256),
        f"B12x layer {layer} manifest qualification hash",
    )
    stagepack.require_equal(
        record.get("pack_hash_low64"),
        pack_hash_low64,
        f"B12x layer {layer} manifest pack hash",
    )
    sha256_verified = False
    if verify_sha256:
        stagepack.require_equal(
            stagepack.sha256_file(path),
            expected_sha256,
            f"B12x layer {layer} sha256",
        )
        sha256_verified = True
    return {
        "layer": layer,
        "path": str(path),
        "bytes": file_bytes,
        "sha256": expected_sha256,
        "sha256_verified": sha256_verified,
        "maximum_token_count": maximum_token_count,
        "kernel_manifest_hash_low64": kernel_manifest_hash_low64,
        "pack_hash_low64": pack_hash_low64,
        **samples,
    }


def validate_resident_manifest(
    root: Path,
    aot_manifest_path: Path,
    source_sha256: str,
    rank: int,
    selected_layers: list[int],
    require_complete_stage: bool,
    sample_bytes: int,
    verify_sha256: bool,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    for name in FOREIGN_MANIFESTS:
        if (root / name).exists():
            raise PreflightFailure(f"NVFP4 pack root mixes formats: {root}")
    document = stagepack.load_json_object(
        root / NVFP4_MANIFEST,
        "NVFP4 resident manifest",
    )
    expected_contract = {
        "record_schema": b12x.MANIFEST_SCHEMA,
        "required_module": b12x.REQUIRED_MODULE,
        "required_arch": b12x.REQUIRED_ARCH,
        "runtime_language": "c_cuda",
        "fallback_allowed": False,
        "runtime_backend_selection": "forbidden",
        "production_backend_policy": "exact_static_buckets_only",
        "runtime_bucket_decomposition": "forbidden",
        "runtime_diagnostic_routing_mutation": "forbidden",
        "pack_magic": b12x.MAGIC.rstrip(b"\0").decode("ascii"),
        "pack_extension": b12x.PACK_EXTENSION,
        "pack_abi_version": b12x.ABI_VERSION,
        "gate_up_order": "up_gate",
        "gate_up_order_id": b12x.GATE_UP_ORDER_UP_GATE,
        "weight_layout": "flashinfer_static_view",
        "weight_layout_id": b12x.WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW,
        "scale_layout": "flashinfer_static_storage",
        "scale_layout_id": b12x.SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE,
        "quant_mode": b12x.QUANT_MODE_NVFP4,
        "output_dtype": b12x.OUTPUT_DTYPE_BF16,
        "output_dtype_name": "BF16",
        "cuda_architecture": b12x.CUDA_ARCHITECTURE_SM121,
        "scale2_baked_into_block_scales": True,
        "w1_alpha": "ones_fp32_by_expert",
        "w2_alpha": "ones_fp32_by_expert",
        "fc2_input_scale": "ones_fp32_by_expert",
    }
    for key, expected in expected_contract.items():
        stagepack.require_equal(
            document.get(key),
            expected,
            f"NVFP4 manifest {key}",
        )
    stagepack.require_equal(
        require_bool(
            document.get("fallback_allowed"),
            "NVFP4 fallback_allowed",
        ),
        False,
        "NVFP4 fallback_allowed",
    )
    resident_source_sha256 = stagepack.require_sha256(
        document.get("source_model_index_sha256"),
        "NVFP4 source_model_index_sha256",
    )
    stagepack.require_equal(
        document.get("source_model_index_file"),
        b12x.SOURCE_INDEX_FILE,
        "NVFP4 source model index file",
    )
    recorded_aot_manifest = Path(stagepack.require_string(
        document.get("aot_manifest"),
        "NVFP4 aot_manifest",
    ))
    stagepack.require_equal(
        recorded_aot_manifest.name,
        aot_manifest_path.name,
        "NVFP4 AOT manifest file",
    )
    stagepack.require_equal(
        resident_source_sha256,
        source_sha256,
        "StagePack/NVFP4 source identity",
    )
    shape = stagepack.require_object(document.get("shape"), "NVFP4 shape")
    stagepack.require_equal(
        shape,
        {
            "hidden_dimension": b12x.HIDDEN_DIMENSION,
            "intermediate_dimension": b12x.INTERMEDIATE_DIMENSION,
            "expert_count": b12x.EXPERT_COUNT,
            "top_k": b12x.TOP_K,
        },
        "NVFP4 shape",
    )
    maximum_token_count = stagepack.require_int(
        document.get("maximum_token_count"),
        "NVFP4 maximum_token_count",
        1,
    )
    kernel_manifest_hash_low64 = stagepack.require_int(
        document.get("kernel_manifest_hash_low64"),
        "NVFP4 kernel_manifest_hash_low64",
        1,
    )
    aot_sha256, aot_receipt = validate_aot_manifest(
        aot_manifest_path,
        kernel_manifest_hash_low64,
        maximum_token_count,
    )
    stagepack.require_equal(
        stagepack.require_sha256(
            document.get("aot_manifest_sha256"),
            "NVFP4 aot_manifest_sha256",
        ),
        aot_sha256,
        "resident/AOT file sha256",
    )
    records_value = stagepack.require_list(
        document.get("packs"),
        "NVFP4 packs",
    )
    records: dict[int, dict[str, Any]] = {}
    for value in records_value:
        record = stagepack.require_object(value, "NVFP4 pack record")
        layer = stagepack.require_int(
            record.get("layer_index"),
            "NVFP4 layer",
            stagepack.FIRST_ROUTED_LAYER,
        )
        if layer in records:
            raise PreflightFailure(f"duplicate NVFP4 manifest layer {layer}")
        records[layer] = record
    expected_layers = stagepack.expected_w8lut_layers(rank)
    if require_complete_stage:
        stagepack.require_equal(
            sorted(records),
            expected_layers,
            f"rank {rank} NVFP4 manifest layers",
        )
    for layer in selected_layers:
        if layer not in expected_layers:
            raise PreflightFailure(
                f"layer {layer} does not belong to NVFP4 rank {rank}")
        if layer not in records:
            raise PreflightFailure(
                f"NVFP4 manifest is missing selected layer {layer}")
    layers = [
        validate_pack(
            root,
            records[layer],
            layer,
            maximum_token_count,
            kernel_manifest_hash_low64,
            aot_sha256,
            sample_bytes,
            verify_sha256,
        )
        for layer in selected_layers
    ]
    return maximum_token_count, aot_receipt, layers


def parse_layers(
    values: list[int] | None,
    rank: int,
) -> tuple[list[int], bool]:
    expected = stagepack.expected_w8lut_layers(rank)
    if values is None:
        return expected, True
    selected: list[int] = []
    for layer in values:
        if layer in selected:
            raise PreflightFailure(f"duplicate selected layer {layer}")
        if layer not in expected:
            raise PreflightFailure(
                f"layer {layer} does not belong to rank {rank}")
        selected.append(layer)
    if not selected:
        raise PreflightFailure("at least one NVFP4 layer is required")
    return sorted(selected), False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GLM-5.2 BF16-trunk/NVFP4-expert StagePack, resident "
            "B12x packs, generated CUDA table, and AOT runtime files"
        ),
    )
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--stagepack-root", required=True, type=Path)
    parser.add_argument("--nvfp4-pack-root", required=True, type=Path)
    parser.add_argument("--aot-manifest", required=True, type=Path)
    parser.add_argument("--layer", action="append", type=int)
    parser.add_argument("--sample-bytes", type=int, default=256)
    parser.add_argument("--verify-pack-sha256", action="store_true")
    parser.add_argument("--max-active", type=int, default=1024)
    parser.add_argument("--rows-per-lane", type=int, default=1)
    parser.add_argument("--mtp", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.rank < 0 or arguments.rank >= stagepack.STAGE_COUNT:
            raise PreflightFailure(
                f"rank must be in 0..{stagepack.STAGE_COUNT - 1}")
        if arguments.sample_bytes < 16 or arguments.sample_bytes > 65536:
            raise PreflightFailure("--sample-bytes must be in 16..65536")
        if arguments.max_active <= 0 or arguments.max_active > 1024:
            raise PreflightFailure("--max-active must be in 1..1024")
        if (arguments.rows_per_lane <= 0 or
                arguments.rows_per_lane > MAX_ROWS_PER_LANE):
            raise PreflightFailure(
                f"--rows-per-lane must be in 1..{MAX_ROWS_PER_LANE}")
        rows_per_lane = (
            MTP_ROWS_PER_LANE if arguments.mtp else arguments.rows_per_lane
        )
        if arguments.mtp and arguments.rows_per_lane != 1:
            raise PreflightFailure(
                "--mtp and an explicit --rows-per-lane cannot be combined")
        selected_layers, require_complete_stage = parse_layers(
            arguments.layer,
            arguments.rank,
        )
        source_sha256, stage_receipt = stagepack.validate_stagepack(
            arguments.stagepack_root,
            arguments.rank,
            selected_layers,
            arguments.sample_bytes,
            NVFP4_QUANTIZATION,
            NVFP4_LABEL,
        )
        maximum_token_count, aot_receipt, layers = validate_resident_manifest(
            arguments.nvfp4_pack_root,
            arguments.aot_manifest,
            source_sha256,
            arguments.rank,
            selected_layers,
            require_complete_stage,
            arguments.sample_bytes,
            arguments.verify_pack_sha256,
        )
        required_execution_rows = arguments.max_active * rows_per_lane
        if required_execution_rows > maximum_token_count:
            raise PreflightFailure(
                "NVFP4 execution row capacity is insufficient: "
                f"required={required_execution_rows} "
                f"available={maximum_token_count} "
                f"max_supported_logical_lanes="
                f"{maximum_token_count // rows_per_lane}")
        if required_execution_rows not in aot_receipt["buckets"]:
            raise PreflightFailure(
                "NVFP4 requires an exact static AOT bucket for the requested "
                f"execution rows: required={required_execution_rows} "
                f"available={aot_receipt['buckets']}")
    except PreflightFailure as error:
        print(f"glm52_nvfp4_artifact_preflight: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "format": PREFLIGHT_SCHEMA,
        "status": "ok",
        "rank": arguments.rank,
        "scope": "full-stage" if require_complete_stage else "selected-layers",
        "source_model_index_sha256": source_sha256,
        "logical_lane_capacity": arguments.max_active,
        "rows_per_lane": rows_per_lane,
        "required_execution_rows": required_execution_rows,
        "maximum_token_count": maximum_token_count,
        "maximum_supported_logical_lanes": (
            maximum_token_count // rows_per_lane
        ),
        "stagepack": stage_receipt,
        "aot": aot_receipt,
        "layers": layers,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
