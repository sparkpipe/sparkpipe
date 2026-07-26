#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any

from glm52_model_contract import load_model_contract


STAGEPACK_FORMAT = "sparkpipe.glm52.pp13.stagepack.v1"
STAGEPACK_INDEX = "stagepack_index.json"
STAGEPACK_QUANTIZATION = "w8lut"
STAGEPACK_DTYPE = "BF16"
STAGEPACK_TOPOLOGY = "pp13_fixed6"
W8LUT_FORMAT = "sparkpipe.glm52.w8lut.resident_moe_pack.v1"
W8LUT_MANIFEST = "w8lut_moe_pack_manifest.json"
W8LUT_FOREIGN_MANIFEST = "fp8_moe_pack_manifest.json"
W8LUT_MAGIC = b"SPARKGLM52W8LUT"
W8LUT_WIRE_MAGIC = W8LUT_MAGIC.ljust(16, b"\0")
W8LUT_EXTENSION = ".spw8lut"
W8LUT_ABI_VERSION = 1
W8LUT_HEADER_BYTES = 512
W8LUT_REGION_ALIGNMENT = 4096
W8LUT_QUANT_MODE = 3
W8LUT_OUTPUT_DTYPE_BF16 = 1
W8LUT_CUDA_ARCHITECTURE = 121
W8LUT_GATE_UP_ORDER_UP_GATE = 1
W8LUT_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR = 1
W8LUT_SCALE_LAYOUT_EXPERT_COMPONENT_E0 = 2
MODEL_CONTRACT = load_model_contract(Path(__file__).resolve().parents[1])
STAGE_COUNT = 13
LAYER_COUNT = MODEL_CONTRACT["layer_count"]
LAYERS_PER_STAGE = LAYER_COUNT // STAGE_COUNT
MTP_LAYER_INDEX = LAYER_COUNT
FIRST_ROUTED_LAYER = MODEL_CONTRACT["first_routed_layer"]
HIDDEN_DIMENSION = MODEL_CONTRACT["hidden_dimension"]
QUERY_A_DIMENSION = MODEL_CONTRACT["query_a_dimension"]
QUERY_B_DIMENSION = MODEL_CONTRACT["head_count"] * (
    MODEL_CONTRACT["qk_nope_head_dimension"] + MODEL_CONTRACT["rope_dimension"]
)
KV_A_DIMENSION = (
    MODEL_CONTRACT["latent_dimension"] + MODEL_CONTRACT["rope_dimension"]
)
KV_B_DIMENSION = MODEL_CONTRACT["head_count"] * (
    MODEL_CONTRACT["qk_nope_head_dimension"] + MODEL_CONTRACT["value_head_dimension"]
)
LATENT_DIMENSION = MODEL_CONTRACT["latent_dimension"]
ATTENTION_PROJECTION_DIMENSION = (
    MODEL_CONTRACT["head_count"] * MODEL_CONTRACT["value_head_dimension"]
)
DENSE_INTERMEDIATE_DIMENSION = MODEL_CONTRACT["dense_intermediate_dimension"]
DSA_QUERY_DIMENSION = (
    MODEL_CONTRACT["dsa_index_head_count"] *
    MODEL_CONTRACT["dsa_index_head_dimension"]
)
DSA_KEY_DIMENSION = MODEL_CONTRACT["dsa_index_head_dimension"]
DSA_WEIGHT_DIMENSION = MODEL_CONTRACT["dsa_index_head_count"]
EXPERT_COUNT = MODEL_CONTRACT["moe_expert_count"]
TOP_K = MODEL_CONTRACT["moe_top_k"]
MOE_INTERMEDIATE_DIMENSION = MODEL_CONTRACT["moe_intermediate_dimension"]
W1_COMPONENT_COUNT = MODEL_CONTRACT["moe_w1_component_count"]
VOCABULARY_SIZE = MODEL_CONTRACT["output_vocab_count"]
STAGE_REGION_ALIGNMENT = 4096
HASH_CHUNK_BYTES = 64 * 1024 * 1024
PACK_HEADER_STRUCT = struct.Struct("<16s16I")
PACK_REGION_STRUCT = struct.Struct("<QQ")
PACK_REGION_COUNT = 4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PreflightFailure(RuntimeError):
    pass


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightFailure(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PreflightFailure(f"{label} must be an array")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreflightFailure(f"{label} must be a nonempty string")
    return value


def require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PreflightFailure(f"{label} must be an integer >= {minimum}")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise PreflightFailure(
            f"{label} mismatch: observed={actual!r} expected={expected!r}")


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightFailure(f"could not read {label} {path}: {error}") from error
    return require_object(value, label)


def require_sha256(value: Any, label: str) -> str:
    text = require_string(value, label)
    if SHA256_RE.fullmatch(text) is None:
        raise PreflightFailure(f"{label} must be 64 lowercase hex characters")
    return text


def dsa_source_layer(layer: int) -> int | None:
    if layer < 0 or layer >= LAYER_COUNT:
        return None
    if layer < FIRST_ROUTED_LAYER:
        return layer
    adjusted = layer - 2
    return 2 + adjusted - (adjusted % 4)


def add_spec(
    specs: dict[str, tuple[str, tuple[int, ...]]],
    name: str,
    dtype: str,
    *shape: int,
) -> None:
    specs[name] = (dtype, tuple(shape))


def required_layer_tensors(layer: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    prefix = f"model.layers.{layer}."
    specs: dict[str, tuple[str, tuple[int, ...]]] = {}
    add_spec(specs, prefix + "input_layernorm.weight", "BF16", HIDDEN_DIMENSION)
    add_spec(specs, prefix + "post_attention_layernorm.weight", "BF16", HIDDEN_DIMENSION)
    add_spec(specs, prefix + "self_attn.q_a_layernorm.weight", "BF16", QUERY_A_DIMENSION)
    add_spec(specs, prefix + "self_attn.kv_a_layernorm.weight", "BF16", LATENT_DIMENSION)
    add_spec(specs, prefix + "self_attn.q_a_proj.weight", "BF16", QUERY_A_DIMENSION, HIDDEN_DIMENSION)
    add_spec(specs, prefix + "self_attn.q_b_proj.weight", "BF16", QUERY_B_DIMENSION, QUERY_A_DIMENSION)
    add_spec(specs, prefix + "self_attn.kv_a_proj_with_mqa.weight", "BF16", KV_A_DIMENSION, HIDDEN_DIMENSION)
    add_spec(specs, prefix + "self_attn.kv_b_proj.weight", "BF16", KV_B_DIMENSION, LATENT_DIMENSION)
    add_spec(specs, prefix + "self_attn.o_proj.weight", "BF16", HIDDEN_DIMENSION, ATTENTION_PROJECTION_DIMENSION)
    if dsa_source_layer(layer) == layer:
        add_spec(specs, prefix + "self_attn.indexer.wq_b.weight", "BF16", DSA_QUERY_DIMENSION, QUERY_A_DIMENSION)
        add_spec(specs, prefix + "self_attn.indexer.wk.weight", "BF16", DSA_KEY_DIMENSION, HIDDEN_DIMENSION)
        add_spec(specs, prefix + "self_attn.indexer.weights_proj.weight", "BF16", DSA_WEIGHT_DIMENSION, HIDDEN_DIMENSION)
        add_spec(specs, prefix + "self_attn.indexer.k_norm.weight", "BF16", DSA_KEY_DIMENSION)
        add_spec(specs, prefix + "self_attn.indexer.k_norm.bias", "BF16", DSA_KEY_DIMENSION)
    if layer < FIRST_ROUTED_LAYER:
        add_spec(specs, prefix + "mlp.gate_proj.weight", "BF16", DENSE_INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION)
        add_spec(specs, prefix + "mlp.up_proj.weight", "BF16", DENSE_INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION)
        add_spec(specs, prefix + "mlp.down_proj.weight", "BF16", HIDDEN_DIMENSION, DENSE_INTERMEDIATE_DIMENSION)
    else:
        add_spec(specs, prefix + "mlp.gate.weight", "BF16", EXPERT_COUNT, HIDDEN_DIMENSION)
        add_spec(specs, prefix + "mlp.gate.e_score_correction_bias", "F32", EXPERT_COUNT)
        add_spec(specs, prefix + "mlp.shared_experts.gate_proj.weight", "BF16", MOE_INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION)
        add_spec(specs, prefix + "mlp.shared_experts.up_proj.weight", "BF16", MOE_INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION)
        add_spec(specs, prefix + "mlp.shared_experts.down_proj.weight", "BF16", HIDDEN_DIMENSION, MOE_INTERMEDIATE_DIMENSION)
    return specs


def required_stage_tensors(rank: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    first_layer = rank * LAYERS_PER_STAGE
    specs: dict[str, tuple[str, tuple[int, ...]]] = {}
    for layer in range(first_layer, first_layer + LAYERS_PER_STAGE):
        specs.update(required_layer_tensors(layer))
    if rank == 0:
        add_spec(specs, "model.embed_tokens.weight", "BF16", VOCABULARY_SIZE, HIDDEN_DIMENSION)
    if rank == STAGE_COUNT - 1:
        specs.update(required_layer_tensors(MTP_LAYER_INDEX))
        add_spec(specs, "model.layers.78.enorm.weight", "BF16", HIDDEN_DIMENSION)
        add_spec(specs, "model.layers.78.hnorm.weight", "BF16", HIDDEN_DIMENSION)
        add_spec(specs, "model.layers.78.eh_proj.weight", "BF16", HIDDEN_DIMENSION, HIDDEN_DIMENSION * 2)
        add_spec(specs, "model.layers.78.shared_head.norm.weight", "BF16", HIDDEN_DIMENSION)
        add_spec(specs, "sparkpipe.mtp.embed_tokens.weight", "BF16", VOCABULARY_SIZE, HIDDEN_DIMENSION)
        add_spec(specs, "model.norm.weight", "BF16", HIDDEN_DIMENSION)
        add_spec(specs, "lm_head.weight", "BF16", VOCABULARY_SIZE, HIDDEN_DIMENSION)
    return specs


def expected_pack_layers(rank: int) -> list[int]:
    first_layer = rank * LAYERS_PER_STAGE
    layers = list(range(
        max(first_layer, FIRST_ROUTED_LAYER),
        first_layer + LAYERS_PER_STAGE,
    ))
    if rank == STAGE_COUNT - 1:
        layers.append(MTP_LAYER_INDEX)
    return layers


def dtype_bytes(dtype: str) -> int:
    if dtype == "BF16":
        return 2
    if dtype == "F32":
        return 4
    raise PreflightFailure(f"unsupported StagePack dtype {dtype!r}")


def tensor_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    count = dtype_bytes(dtype)
    for dimension in shape:
        if dimension <= 0:
            raise PreflightFailure("tensor shape dimensions must be positive")
        count *= dimension
    return count


def validate_tensor_record(
    name: str,
    record_value: Any,
    expected: tuple[str, tuple[int, ...]] | None,
    stage_file_name: str,
    stage_file_bytes: int,
    quantization_label: str = "W8LUT",
) -> tuple[int, int]:
    record = require_object(record_value, f"tensor_map[{name}]")
    file_name = require_string(record.get("file"), f"tensor_map[{name}].file")
    require_equal(file_name, stage_file_name, f"tensor_map[{name}].file")
    dtype = require_string(record.get("dtype"), f"tensor_map[{name}].dtype")
    shape_value = require_list(record.get("shape"), f"tensor_map[{name}].shape")
    shape = tuple(
        require_int(value, f"tensor_map[{name}].shape", 1)
        for value in shape_value
    )
    offset = require_int(record.get("offset"), f"tensor_map[{name}].offset")
    byte_count = require_int(record.get("bytes"), f"tensor_map[{name}].bytes", 1)
    if offset % STAGE_REGION_ALIGNMENT != 0:
        raise PreflightFailure(f"tensor {name} offset {offset} is not 4096-byte aligned")
    require_equal(byte_count, tensor_bytes(dtype, shape), f"tensor_map[{name}].bytes")
    if expected is not None:
        require_equal(dtype, expected[0], f"tensor_map[{name}].dtype")
        require_equal(shape, expected[1], f"tensor_map[{name}].shape")
    if offset + byte_count > stage_file_bytes:
        raise PreflightFailure(f"tensor {name} exceeds StagePack file size")
    if ".mlp.experts." in name or name.endswith("weight_scale_inv"):
        raise PreflightFailure(
            f"{quantization_label} StagePack contains forbidden tensor {name}")
    return offset, byte_count


def read_exact_at(file_descriptor: int, offset: int, byte_count: int, label: str) -> bytes:
    data = os.pread(file_descriptor, byte_count, offset)
    if len(data) != byte_count:
        raise PreflightFailure(
            f"short read for {label}: offset={offset} observed={len(data)} expected={byte_count}")
    return data


def sample_offsets(byte_count: int, sample_bytes: int) -> list[int]:
    width = min(byte_count, sample_bytes)
    return sorted({0, (byte_count - width) // 2, byte_count - width})


def sample_stage_tensors(
    stage_file: Path,
    records: dict[str, Any],
    names: list[str],
    sample_bytes: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    sampled = 0
    nonzero = 0
    file_descriptor = os.open(stage_file, os.O_RDONLY)
    try:
        for name in sorted(names):
            record = require_object(records[name], f"tensor_map[{name}]")
            offset = require_int(record.get("offset"), f"tensor_map[{name}].offset")
            byte_count = require_int(record.get("bytes"), f"tensor_map[{name}].bytes", 1)
            width = min(byte_count, sample_bytes)
            for relative_offset in sample_offsets(byte_count, sample_bytes):
                payload = read_exact_at(
                    file_descriptor,
                    offset + relative_offset,
                    width,
                    name,
                )
                digest.update(name.encode("utf-8"))
                digest.update(struct.pack("<QQ", relative_offset, len(payload)))
                digest.update(payload)
                sampled += len(payload)
                nonzero += sum(value != 0 for value in payload)
    finally:
        os.close(file_descriptor)
    if sampled == 0 or nonzero == 0:
        raise PreflightFailure("StagePack tensor samples are empty or entirely zero")
    return {
        "sample_bytes": sampled,
        "sample_nonzero_bytes": nonzero,
        "sample_sha256": digest.hexdigest(),
    }


def validate_stagepack(
    root: Path,
    rank: int,
    selected_layers: list[int],
    sample_bytes: int,
    model_quantization: str = STAGEPACK_QUANTIZATION,
    quantization_label: str = "W8LUT",
) -> tuple[str, dict[str, Any]]:
    index = load_json_object(root / STAGEPACK_INDEX, "StagePack index")
    require_equal(index.get("format"), STAGEPACK_FORMAT, "StagePack format")
    require_equal(index.get("topology"), STAGEPACK_TOPOLOGY, "StagePack topology")
    require_equal(
        index.get("model_quantization"),
        model_quantization,
        "StagePack quantization",
    )
    require_equal(index.get("non_expert_weight_dtype"), STAGEPACK_DTYPE, "StagePack non-expert dtype")
    require_equal(index.get("stage_count"), STAGE_COUNT, "StagePack stage_count")
    require_equal(index.get("layers_per_stage"), LAYERS_PER_STAGE, "StagePack layers_per_stage")
    source_sha256 = require_sha256(
        index.get("source_model_index_sha256"),
        "StagePack source_model_index_sha256",
    )
    stages = require_object(index.get("stages"), "StagePack stages")
    stage = require_object(stages.get(str(rank)), f"StagePack stage {rank}")
    stage_file_name = f"stage_{rank:02d}_non_moe.spstage"
    require_equal(stage.get("file"), stage_file_name, f"StagePack stage {rank} file")
    require_equal(stage.get("first_layer"), rank * LAYERS_PER_STAGE, f"StagePack stage {rank} first_layer")
    require_equal(stage.get("layer_count"), LAYERS_PER_STAGE, f"StagePack stage {rank} layer_count")
    stage_file = root / stage_file_name
    if not stage_file.is_file():
        raise PreflightFailure(f"missing StagePack file {stage_file}")
    stage_file_bytes = stage_file.stat().st_size
    if stage_file_bytes <= 0:
        raise PreflightFailure(f"empty StagePack file {stage_file}")
    records = require_object(index.get("tensor_map"), "StagePack tensor_map")
    local_records = {
        name: value
        for name, value in records.items()
        if isinstance(value, dict) and value.get("file") == stage_file_name
    }
    require_equal(
        len(local_records),
        require_int(stage.get("tensor_count"), f"StagePack stage {rank} tensor_count", 1),
        f"StagePack stage {rank} tensor_count",
    )
    required = required_stage_tensors(rank)
    missing = sorted(set(required) - set(local_records))
    if missing:
        raise PreflightFailure(f"StagePack is missing required tensor {missing[0]}")
    ranges: list[tuple[int, int, str]] = []
    for name, value in local_records.items():
        offset, byte_count = validate_tensor_record(
            name,
            value,
            required.get(name),
            stage_file_name,
            stage_file_bytes,
            quantization_label,
        )
        ranges.append((offset, offset + byte_count, name))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise PreflightFailure(
                f"StagePack tensors overlap: {previous[2]} and {current[2]}")
    require_equal(ranges[-1][1], stage_file_bytes, "StagePack file byte count")
    selected_names: list[str] = []
    for layer in selected_layers:
        selected_names.extend(required_layer_tensors(layer).keys())
    sample = sample_stage_tensors(
        stage_file,
        local_records,
        selected_names,
        sample_bytes,
    )
    return source_sha256, {
        "path": str(stage_file),
        "bytes": stage_file_bytes,
        "tensor_count": len(local_records),
        "required_tensor_count": len(required),
        **sample,
    }


def expected_pack_regions() -> list[tuple[int, int]]:
    byte_counts = [
        EXPERT_COUNT * W1_COMPONENT_COUNT * MOE_INTERMEDIATE_DIMENSION * HIDDEN_DIMENSION,
        EXPERT_COUNT * W1_COMPONENT_COUNT * 2,
        EXPERT_COUNT * HIDDEN_DIMENSION * MOE_INTERMEDIATE_DIMENSION,
        EXPERT_COUNT * 2,
    ]
    regions: list[tuple[int, int]] = []
    offset = W8LUT_HEADER_BYTES
    for byte_count in byte_counts:
        offset = align_up(offset, W8LUT_REGION_ALIGNMENT)
        regions.append((offset, byte_count))
        offset += byte_count
    return regions


def build_pack_header(layer: int, maximum_tokens: int) -> bytes:
    prefix = PACK_HEADER_STRUCT.pack(
        W8LUT_WIRE_MAGIC,
        W8LUT_ABI_VERSION,
        W8LUT_HEADER_BYTES,
        layer,
        maximum_tokens,
        HIDDEN_DIMENSION,
        MOE_INTERMEDIATE_DIMENSION,
        EXPERT_COUNT,
        TOP_K,
        W8LUT_GATE_UP_ORDER_UP_GATE,
        W8LUT_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR,
        W8LUT_SCALE_LAYOUT_EXPERT_COMPONENT_E0,
        W8LUT_QUANT_MODE,
        W8LUT_OUTPUT_DTYPE_BF16,
        W8LUT_CUDA_ARCHITECTURE,
        0,
        0,
    )
    regions = b"".join(
        PACK_REGION_STRUCT.pack(offset, byte_count)
        for offset, byte_count in expected_pack_regions()
    )
    header = prefix + regions
    return header + b"\0" * (W8LUT_HEADER_BYTES - len(header))


def parse_pack_header(payload: bytes) -> tuple[tuple[Any, ...], list[tuple[int, int]]]:
    if len(payload) != W8LUT_HEADER_BYTES:
        raise PreflightFailure("short W8LUT pack header")
    fields = PACK_HEADER_STRUCT.unpack(payload[:PACK_HEADER_STRUCT.size])
    regions: list[tuple[int, int]] = []
    offset = PACK_HEADER_STRUCT.size
    for _ in range(PACK_REGION_COUNT):
        regions.append(PACK_REGION_STRUCT.unpack(
            payload[offset:offset + PACK_REGION_STRUCT.size]))
        offset += PACK_REGION_STRUCT.size
    if any(payload[offset:]):
        raise PreflightFailure("W8LUT pack header reserved bytes are not zero")
    return fields, regions


def validate_pack_header(
    payload: bytes,
    layer: int,
    maximum_tokens: int,
    file_bytes: int,
) -> list[tuple[int, int]]:
    fields, regions = parse_pack_header(payload)
    expected_fields = PACK_HEADER_STRUCT.unpack(
        build_pack_header(layer, maximum_tokens)[:PACK_HEADER_STRUCT.size])
    require_equal(fields, expected_fields, f"W8LUT layer {layer} header")
    require_equal(regions, expected_pack_regions(), f"W8LUT layer {layer} regions")
    expected_bytes = regions[-1][0] + regions[-1][1]
    require_equal(file_bytes, expected_bytes, f"W8LUT layer {layer} file bytes")
    return regions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            payload = file.read(HASH_CHUNK_BYTES)
            if not payload:
                return digest.hexdigest()
            digest.update(payload)


def sample_code_routes(
    file_descriptor: int,
    region_offset: int,
    route_count: int,
    route_bytes: int,
    sample_bytes: int,
    label: str,
) -> bytes:
    samples = bytearray()
    for route in range(route_count):
        route_offset = region_offset + route * route_bytes
        width = min(route_bytes, sample_bytes)
        for relative_offset in sample_offsets(route_bytes, sample_bytes):
            samples.extend(read_exact_at(
                file_descriptor,
                route_offset + relative_offset,
                width,
                f"{label} route {route}",
            ))
    return bytes(samples)


def code_entropy(payload: bytes) -> tuple[float, int, int]:
    if not payload:
        raise PreflightFailure("W8LUT code sample is empty")
    counts = Counter(payload)
    entropy = 0.0
    for count in counts.values():
        probability = count / len(payload)
        entropy -= probability * math.log2(probability)
    return entropy, len(counts), sum(value != 0 for value in payload)


def validate_w8lut_pack(
    root: Path,
    record: dict[str, Any],
    layer: int,
    maximum_tokens: int,
    sample_bytes: int,
    verify_sha256: bool,
) -> dict[str, Any]:
    require_equal(record.get("layer"), layer, f"W8LUT layer {layer} manifest layer")
    file_name = require_string(record.get("file"), f"W8LUT layer {layer} file")
    require_equal(file_name, f"glm52_layer_{layer:04d}_w8lut_moe.spw8lut", f"W8LUT layer {layer} file")
    path = root / file_name
    if not path.is_file():
        raise PreflightFailure(f"missing W8LUT layer pack {path}")
    file_bytes = path.stat().st_size
    require_equal(record.get("bytes"), file_bytes, f"W8LUT layer {layer} manifest bytes")
    require_equal(record.get("maximum_token_count"), maximum_tokens, f"W8LUT layer {layer} maximum token count")
    element_count = require_int(record.get("element_count"), f"W8LUT layer {layer} element_count", 1)
    require_equal(element_count, expected_pack_regions()[0][1] + expected_pack_regions()[2][1], f"W8LUT layer {layer} element_count")
    below_count = require_int(record.get("below_window_count"), f"W8LUT layer {layer} below_window_count")
    below_ppm = require_int(record.get("below_window_ppm"), f"W8LUT layer {layer} below_window_ppm")
    require_equal(below_ppm, below_count * 1_000_000 // element_count, f"W8LUT layer {layer} below_window_ppm")
    expected_sha256 = require_sha256(record.get("sha256"), f"W8LUT layer {layer} sha256")
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        header = read_exact_at(file_descriptor, 0, W8LUT_HEADER_BYTES, f"W8LUT layer {layer} header")
        regions = validate_pack_header(header, layer, maximum_tokens, file_bytes)
        w1_e0 = struct.unpack(
            f"<{EXPERT_COUNT * W1_COMPONENT_COUNT}H",
            read_exact_at(file_descriptor, regions[1][0], regions[1][1], f"W8LUT layer {layer} W1 E0"),
        )
        w2_e0 = struct.unpack(
            f"<{EXPERT_COUNT}H",
            read_exact_at(file_descriptor, regions[3][0], regions[3][1], f"W8LUT layer {layer} W2 E0"),
        )
        if max(w1_e0 + w2_e0) > 247:
            raise PreflightFailure(f"W8LUT layer {layer} exponent base exceeds 247")
        route_bytes = MOE_INTERMEDIATE_DIMENSION * HIDDEN_DIMENSION
        w1_sample = sample_code_routes(
            file_descriptor,
            regions[0][0],
            EXPERT_COUNT * W1_COMPONENT_COUNT,
            route_bytes,
            sample_bytes,
            f"W8LUT layer {layer} W1",
        )
        w2_sample = sample_code_routes(
            file_descriptor,
            regions[2][0],
            EXPERT_COUNT,
            route_bytes,
            sample_bytes,
            f"W8LUT layer {layer} W2",
        )
    finally:
        os.close(file_descriptor)
    sample = w1_sample + w2_sample
    entropy, unique_codes, nonzero_codes = code_entropy(sample)
    if entropy < 2.0 or unique_codes < 16 or nonzero_codes == 0:
        raise PreflightFailure(
            f"W8LUT layer {layer} code sample is degenerate: "
            f"entropy={entropy:.6f} unique={unique_codes} nonzero={nonzero_codes}")
    if verify_sha256:
        require_equal(sha256_file(path), expected_sha256, f"W8LUT layer {layer} sha256")
    return {
        "layer": layer,
        "path": str(path),
        "bytes": file_bytes,
        "sha256": expected_sha256,
        "sha256_verified": verify_sha256,
        "below_window_ppm": below_ppm,
        "sample_bytes": len(sample),
        "sample_sha256": hashlib.sha256(sample).hexdigest(),
        "sample_entropy_bits": round(entropy, 6),
        "sample_unique_codes": unique_codes,
        "w1_e0_min": min(w1_e0),
        "w1_e0_max": max(w1_e0),
        "w2_e0_min": min(w2_e0),
        "w2_e0_max": max(w2_e0),
    }


def validate_w8lut_manifest(
    root: Path,
    source_sha256: str,
    rank: int,
    selected_layers: list[int],
    require_complete_stage: bool,
    sample_bytes: int,
    verify_sha256: bool,
) -> list[dict[str, Any]]:
    if (root / W8LUT_FOREIGN_MANIFEST).exists():
        raise PreflightFailure(f"W8LUT pack root mixes formats: {root}")
    manifest = load_json_object(root / W8LUT_MANIFEST, "W8LUT manifest")
    require_equal(manifest.get("format"), W8LUT_FORMAT, "W8LUT manifest format")
    require_equal(manifest.get("pack_magic"), W8LUT_MAGIC.decode("ascii"), "W8LUT pack magic")
    require_equal(manifest.get("pack_extension"), W8LUT_EXTENSION, "W8LUT pack extension")
    require_equal(manifest.get("source_dtype"), STAGEPACK_DTYPE, "W8LUT source dtype")
    require_equal(manifest.get("quant_mode"), W8LUT_QUANT_MODE, "W8LUT quant mode")
    require_equal(manifest.get("scale_layout"), "expert_component_u16_e0", "W8LUT scale layout")
    require_equal(
        require_sha256(manifest.get("source_model_index_sha256"), "W8LUT source_model_index_sha256"),
        source_sha256,
        "StagePack/W8LUT source identity",
    )
    maximum_tokens = require_int(
        manifest.get("maximum_active_sequence_count"),
        "W8LUT maximum_active_sequence_count",
        1,
    )
    records_value = require_list(manifest.get("layers"), "W8LUT layers")
    records: dict[int, dict[str, Any]] = {}
    for value in records_value:
        record = require_object(value, "W8LUT layer record")
        layer = require_int(record.get("layer"), "W8LUT layer", FIRST_ROUTED_LAYER)
        if layer in records:
            raise PreflightFailure(f"duplicate W8LUT manifest layer {layer}")
        records[layer] = record
    expected = expected_pack_layers(rank)
    if require_complete_stage:
        require_equal(sorted(records), expected, f"rank {rank} W8LUT manifest layers")
    for layer in selected_layers:
        if layer not in expected:
            raise PreflightFailure(f"layer {layer} does not belong to W8LUT rank {rank}")
        if layer not in records:
            raise PreflightFailure(f"W8LUT manifest is missing selected layer {layer}")
    return [
        validate_w8lut_pack(
            root,
            records[layer],
            layer,
            maximum_tokens,
            sample_bytes,
            verify_sha256,
        )
        for layer in selected_layers
    ]


def parse_layers(values: list[int] | None, rank: int) -> tuple[list[int], bool]:
    expected = expected_pack_layers(rank)
    if values is None:
        return expected, True
    layers: list[int] = []
    for layer in values:
        if layer in layers:
            raise PreflightFailure(f"duplicate selected layer {layer}")
        if layer not in expected:
            raise PreflightFailure(f"layer {layer} does not belong to rank {rank}")
        layers.append(layer)
    if not layers:
        raise PreflightFailure("at least one W8LUT layer is required")
    return sorted(layers), False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one real GLM-5.2 W8LUT layer or a complete rank-local artifact set without CUDA",
    )
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--stagepack-root", required=True, type=Path)
    parser.add_argument("--w8lut-pack-root", required=True, type=Path)
    parser.add_argument("--layer", action="append", type=int)
    parser.add_argument("--sample-bytes", type=int, default=256)
    parser.add_argument("--verify-sha256", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.rank < 0 or arguments.rank >= STAGE_COUNT:
            raise PreflightFailure(f"rank must be in 0..{STAGE_COUNT - 1}")
        if arguments.sample_bytes < 16 or arguments.sample_bytes > 65536:
            raise PreflightFailure("--sample-bytes must be in 16..65536")
        selected_layers, require_complete_stage = parse_layers(
            arguments.layer,
            arguments.rank,
        )
        source_sha256, stage = validate_stagepack(
            arguments.stagepack_root,
            arguments.rank,
            selected_layers,
            arguments.sample_bytes,
        )
        layers = validate_w8lut_manifest(
            arguments.w8lut_pack_root,
            source_sha256,
            arguments.rank,
            selected_layers,
            require_complete_stage,
            arguments.sample_bytes,
            arguments.verify_sha256,
        )
    except PreflightFailure as error:
        print(f"glm52_w8lut_artifact_preflight: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "format": "sparkpipe.glm52.w8lut.artifact_preflight.v1",
        "status": "ok",
        "rank": arguments.rank,
        "scope": "full-stage" if require_complete_stage else "selected-layers",
        "source_model_index_sha256": source_sha256,
        "stagepack": stage,
        "layers": layers,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
