#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, BinaryIO, Dict, Iterable, List, Tuple


MAGIC = b"SPARKGLM52B12X\0\0"
ABI_VERSION = 1
HEADER_BYTES = 512
REGION_ALIGNMENT = 4096
REGION_COUNT = 7

REGION_W1_WEIGHT = 0
REGION_W1_SCALE = 1
REGION_W1_ALPHA = 2
REGION_FC2_INPUT_SCALE = 3
REGION_W2_WEIGHT = 4
REGION_W2_SCALE = 5
REGION_W2_ALPHA = 6

HIDDEN_DIMENSION = 6144
INTERMEDIATE_DIMENSION = 2048
EXPERT_COUNT = 256
TOP_K = 8
NVFP4_GROUP_SIZE = 16

GATE_UP_ORDER_UP_GATE = 1
WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW = 2
SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE = 2
QUANT_MODE_NVFP4 = 1
OUTPUT_DTYPE_BF16 = 1
CUDA_ARCHITECTURE_SM121 = 121

HEADER_PREFIX_FORMAT = "<16sIIIIIIIIIIIIIIIIQQQQ"
REGION_FORMAT = "<QQ"


class PackFailure(RuntimeError):
    pass


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


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def parse_layers(text: str) -> List[int]:
    layers: List[int] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise PackFailure("layer indices must be non-negative")
        layers.append(value)
    if not layers:
        raise PackFailure("at least one layer must be selected")
    return sorted(set(layers))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_weight_index(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise PackFailure(f"missing safetensors index: {index_path}")
    document = load_json(index_path)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PackFailure("safetensors index is missing weight_map")
    return {str(key): str(value) for key, value in weight_map.items()}


def tensor_name(layer: int, expert: int, projection: str, suffix: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"


def tensor_file(model_dir: Path, weight_map: Dict[str, str], name: str) -> Path:
    file_name = weight_map.get(name)
    if file_name is None:
        raise PackFailure(f"missing tensor in safetensors index: {name}")
    return model_dir / file_name


class SafetensorReader:
    def __init__(self, model_dir: Path, weight_map: Dict[str, str]) -> None:
        from safetensors import safe_open

        self.model_dir = model_dir
        self.weight_map = weight_map
        self.safe_open = safe_open
        self.handles: Dict[Path, Any] = {}

    def close(self) -> None:
        for handle in self.handles.values():
            close = getattr(handle, "close", None)
            if close is not None:
                close()
        self.handles.clear()

    def tensor(self, name: str):
        path = tensor_file(self.model_dir, self.weight_map, name)
        handle = self.handles.get(path)
        if handle is None:
            handle = self.safe_open(str(path), framework="pt", device="cpu")
            self.handles[path] = handle
        return handle.get_tensor(name)


def require_shape(name: str, tensor: Any, expected: Tuple[int, ...]) -> None:
    observed = tuple(int(item) for item in tensor.shape)
    if observed != expected:
        raise PackFailure(f"{name} has shape {observed}, expected {expected}")


def require_dtype_name(name: str, tensor: Any, expected: str) -> None:
    observed = str(tensor.dtype)
    if observed != expected:
        raise PackFailure(f"{name} has dtype {observed}, expected {expected}")


def tensor_bytes_uint8(name: str, tensor: Any, expected_shape: Tuple[int, ...]) -> bytes:
    require_shape(name, tensor, expected_shape)
    require_dtype_name(name, tensor, "torch.uint8")
    return tensor.contiguous().numpy().tobytes()


def tensor_scalar_f32(name: str, tensor: Any) -> float:
    if tensor.numel() != 1:
        raise PackFailure(f"{name} must be a scalar")
    return float(tensor.reshape(-1)[0].float().item())


def tensor_bytes_f8_scaled(name: str, tensor: Any, scale: float, expected_shape: Tuple[int, ...]) -> bytes:
    import torch

    require_shape(name, tensor, expected_shape)
    if "float8_e4m3" not in str(tensor.dtype):
        raise PackFailure(f"{name} has dtype {tensor.dtype}, expected torch.float8_e4m3fn")
    if not scale > 0.0:
        raise PackFailure(f"{name} scale2 must be positive")
    scaled = (tensor.float() * float(scale)).to(torch.float8_e4m3fn).contiguous()
    return scaled.view(torch.uint8).numpy().tobytes()


def write_padding(file: BinaryIO, target_offset: int) -> None:
    current_offset = file.tell()
    if current_offset > target_offset:
        raise PackFailure("region layout moved backwards")
    if current_offset < target_offset:
        file.write(b"\0" * (target_offset - current_offset))


def reserve_regions() -> List[Dict[str, int]]:
    w1_weight_bytes = EXPERT_COUNT * (2 * INTERMEDIATE_DIMENSION) * (HIDDEN_DIMENSION // 2)
    w1_scale_bytes = EXPERT_COUNT * (2 * INTERMEDIATE_DIMENSION) * (HIDDEN_DIMENSION // NVFP4_GROUP_SIZE)
    w2_weight_bytes = EXPERT_COUNT * HIDDEN_DIMENSION * (INTERMEDIATE_DIMENSION // 2)
    w2_scale_bytes = EXPERT_COUNT * HIDDEN_DIMENSION * (INTERMEDIATE_DIMENSION // NVFP4_GROUP_SIZE)
    alpha_bytes = EXPERT_COUNT * 4
    sizes = [
        w1_weight_bytes,
        w1_scale_bytes,
        alpha_bytes,
        alpha_bytes,
        w2_weight_bytes,
        w2_scale_bytes,
        alpha_bytes,
    ]
    regions: List[Dict[str, int]] = []
    offset = HEADER_BYTES
    for size in sizes:
        offset = align_up(offset, REGION_ALIGNMENT)
        regions.append({"offset": offset, "bytes": size})
        offset += size
    return regions


def pack_header(
    layer: int,
    maximum_token_count: int,
    qualified_maximum_microseconds: int,
    qualification_hash_low64: int,
    kernel_manifest_hash_low64: int,
    pack_hash_low64: int,
    regions: List[Dict[str, int]],
) -> bytes:
    prefix = struct.pack(
        HEADER_PREFIX_FORMAT,
        MAGIC,
        ABI_VERSION,
        HEADER_BYTES,
        layer,
        maximum_token_count,
        HIDDEN_DIMENSION,
        INTERMEDIATE_DIMENSION,
        EXPERT_COUNT,
        TOP_K,
        GATE_UP_ORDER_UP_GATE,
        WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW,
        SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE,
        QUANT_MODE_NVFP4,
        OUTPUT_DTYPE_BF16,
        CUDA_ARCHITECTURE_SM121,
        0,
        0,
        qualified_maximum_microseconds,
        qualification_hash_low64,
        kernel_manifest_hash_low64,
        pack_hash_low64,
    )
    region_bytes = b"".join(
        struct.pack(REGION_FORMAT, int(region["offset"]), int(region["bytes"]))
        for region in regions
    )
    header = prefix + region_bytes
    if len(header) > HEADER_BYTES:
        raise PackFailure("pack header format exceeds fixed header size")
    return header + (b"\0" * (HEADER_BYTES - len(header)))


def write_w1_weight_region(reader: SafetensorReader, file: BinaryIO, layer: int) -> None:
    expected = (INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION // 2)
    for expert in range(EXPERT_COUNT):
        up_name = tensor_name(layer, expert, "up_proj", "weight")
        gate_name = tensor_name(layer, expert, "gate_proj", "weight")
        file.write(tensor_bytes_uint8(up_name, reader.tensor(up_name), expected))
        file.write(tensor_bytes_uint8(gate_name, reader.tensor(gate_name), expected))


def write_w1_scale_region(reader: SafetensorReader, file: BinaryIO, layer: int) -> None:
    expected = (INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION // NVFP4_GROUP_SIZE)
    for expert in range(EXPERT_COUNT):
        up_name = tensor_name(layer, expert, "up_proj", "weight_scale")
        up_scale2_name = tensor_name(layer, expert, "up_proj", "weight_scale_2")
        gate_name = tensor_name(layer, expert, "gate_proj", "weight_scale")
        gate_scale2_name = tensor_name(layer, expert, "gate_proj", "weight_scale_2")
        up_scale2 = tensor_scalar_f32(up_scale2_name, reader.tensor(up_scale2_name))
        gate_scale2 = tensor_scalar_f32(gate_scale2_name, reader.tensor(gate_scale2_name))
        file.write(tensor_bytes_f8_scaled(up_name, reader.tensor(up_name), up_scale2, expected))
        file.write(tensor_bytes_f8_scaled(gate_name, reader.tensor(gate_name), gate_scale2, expected))


def write_w2_weight_region(reader: SafetensorReader, file: BinaryIO, layer: int) -> None:
    expected = (HIDDEN_DIMENSION, INTERMEDIATE_DIMENSION // 2)
    for expert in range(EXPERT_COUNT):
        down_name = tensor_name(layer, expert, "down_proj", "weight")
        file.write(tensor_bytes_uint8(down_name, reader.tensor(down_name), expected))


def write_w2_scale_region(reader: SafetensorReader, file: BinaryIO, layer: int) -> None:
    expected = (HIDDEN_DIMENSION, INTERMEDIATE_DIMENSION // NVFP4_GROUP_SIZE)
    for expert in range(EXPERT_COUNT):
        down_name = tensor_name(layer, expert, "down_proj", "weight_scale")
        down_scale2_name = tensor_name(layer, expert, "down_proj", "weight_scale_2")
        down_scale2 = tensor_scalar_f32(down_scale2_name, reader.tensor(down_scale2_name))
        file.write(tensor_bytes_f8_scaled(down_name, reader.tensor(down_name), down_scale2, expected))


def write_alpha_region(file: BinaryIO) -> None:
    file.write(struct.pack("<" + ("f" * EXPERT_COUNT), *([1.0] * EXPERT_COUNT)))


def read_aot_manifest(path: Path) -> Tuple[int, int, int]:
    document = load_json(path)
    manifest_hash_low64 = int(document.get("manifest_hash_low64", 0))
    if manifest_hash_low64 == 0:
        digest = sha256_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
        manifest_hash_low64 = low64_from_hex(digest)
    maximum_token_count = int(document.get("maximum_token_count", 0))
    if maximum_token_count <= 0:
        raise PackFailure("AOT manifest maximum_token_count is missing")
    qualified = 0
    for bucket in document.get("buckets", []):
        if isinstance(bucket, dict):
            qualified = max(
                qualified,
                int(bucket.get("p95_us", 0) or 0),
                int(bucket.get("avg_us", 0) or 0),
                int(bucket.get("qualified_p95_microseconds", 0) or 0),
            )
    if qualified <= 0:
        qualified = 1
    return maximum_token_count, qualified, manifest_hash_low64


def write_pack(
    model_dir: Path,
    weight_map: Dict[str, str],
    layer: int,
    output_path: Path,
    maximum_token_count: int,
    qualified_maximum_microseconds: int,
    qualification_hash_low64: int,
    kernel_manifest_hash_low64: int,
) -> Dict[str, Any]:
    regions = reserve_regions()
    metadata_digest = sha256_text(json.dumps({
        "layer": layer,
        "regions": regions,
        "maximum_token_count": maximum_token_count,
        "qualified_maximum_microseconds": qualified_maximum_microseconds,
        "qualification_hash_low64": qualification_hash_low64,
        "kernel_manifest_hash_low64": kernel_manifest_hash_low64,
    }, sort_keys=True, separators=(",", ":")))
    pack_hash_low64 = low64_from_hex(metadata_digest)
    header = pack_header(
        layer,
        maximum_token_count,
        qualified_maximum_microseconds,
        qualification_hash_low64,
        kernel_manifest_hash_low64,
        pack_hash_low64,
        regions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = SafetensorReader(model_dir, weight_map)
    try:
        with output_path.open("wb") as file:
            file.write(header)
            write_padding(file, regions[REGION_W1_WEIGHT]["offset"])
            write_w1_weight_region(reader, file, layer)
            write_padding(file, regions[REGION_W1_SCALE]["offset"])
            write_w1_scale_region(reader, file, layer)
            write_padding(file, regions[REGION_W1_ALPHA]["offset"])
            write_alpha_region(file)
            write_padding(file, regions[REGION_FC2_INPUT_SCALE]["offset"])
            write_alpha_region(file)
            write_padding(file, regions[REGION_W2_WEIGHT]["offset"])
            write_w2_weight_region(reader, file, layer)
            write_padding(file, regions[REGION_W2_SCALE]["offset"])
            write_w2_scale_region(reader, file, layer)
            write_padding(file, regions[REGION_W2_ALPHA]["offset"])
            write_alpha_region(file)
    finally:
        reader.close()
    return {
        "path": str(output_path),
        "layer_index": layer,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "pack_hash_low64": pack_hash_low64,
        "kernel_manifest_hash_low64": kernel_manifest_hash_low64,
        "qualification_record_hash_low64": qualification_hash_low64,
        "qualified_maximum_microseconds": qualified_maximum_microseconds,
        "regions": regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--aot-manifest", default="build/glm52_b12x_aot/generated/aot_manifest.json")
    parser.add_argument("--layers", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--output-dir", default="build/glm52_b12x_resident_moe")
    parser.add_argument("--qualified-maximum-microseconds", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    aot_manifest_path = Path(args.aot_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not model_dir.exists():
        raise PackFailure(f"model directory does not exist: {model_dir}")
    if not aot_manifest_path.exists():
        raise PackFailure(f"AOT manifest does not exist: {aot_manifest_path}")

    maximum_token_count, manifest_qualified_us, kernel_manifest_hash_low64 = read_aot_manifest(aot_manifest_path)
    qualified_us = args.qualified_maximum_microseconds or manifest_qualified_us
    if qualified_us <= 0:
        raise PackFailure("qualified maximum microseconds must be positive")
    qualification_hash_low64 = low64_from_hex(sha256_file(aot_manifest_path))
    weight_map = read_weight_index(model_dir)
    layers = parse_layers(args.layers)

    records = []
    for layer in layers:
        output_path = output_dir / f"glm52_layer_{layer:04d}_b12x_moe.spb12x"
        record = write_pack(
            model_dir,
            weight_map,
            layer,
            output_path,
            maximum_token_count,
            qualified_us,
            qualification_hash_low64,
            kernel_manifest_hash_low64,
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    manifest = {
        "record_schema": "sparkpipe.glm52.sm121.b12x.resident_moe_pack.v1",
        "model_dir": str(model_dir),
        "aot_manifest": str(aot_manifest_path),
        "required_module": "spark.glm52.sm121.flashinfer_b12x_fused_moe.nvfp4.bf16.v2",
        "required_arch": "sm_121a",
        "runtime_language": "c_cuda",
        "compile_time_languages": ["python", "torch", "safetensors"],
        "fallback_allowed": False,
        "runtime_backend_selection": "forbidden",
        "gate_up_order": "up_gate",
        "scale2_baked_into_block_scales": True,
        "w1_alpha": "ones_fp32_by_expert",
        "w2_alpha": "ones_fp32_by_expert",
        "fc2_input_scale": "ones_fp32_by_expert",
        "shape": {
            "hidden_dimension": HIDDEN_DIMENSION,
            "intermediate_dimension": INTERMEDIATE_DIMENSION,
            "expert_count": EXPERT_COUNT,
            "top_k": TOP_K,
        },
        "packs": records,
    }
    manifest_path = output_dir / "resident_moe_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackFailure as error:
        print(f"glm52_b12x_resident_pack: {error}", file=sys.stderr)
        raise SystemExit(2)
