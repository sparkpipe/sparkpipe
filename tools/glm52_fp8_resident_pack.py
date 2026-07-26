#!/usr/bin/env python3
"""
Build one-time GLM-5.2 FP8 resident MoE packs from the official FP8 checkpoint.

The generated files are setup artifacts, not serving-path code.  Serving should
load these packs from C/CUDA only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from multiprocessing import Pool
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple

from glm52_resident_pack_common import (
    EXPERT_COUNT,
    HIDDEN_DIMENSION,
    INTERMEDIATE_DIMENSION,
    MODEL_CONTRACT,
    TOP_K,
    W1_COMPONENT_COUNT,
    PackFailure,
    SafetensorReader,
    align_up,
    import_torch,
    parse_layers,
    tensor_name,
    tp_shard_range,
)


MAGIC = b"SPARKGLM52FP8\0\0"
MAGIC_FIELD_BYTES = 16
WIRE_MAGIC = MAGIC.ljust(MAGIC_FIELD_BYTES, b"\0")
ABI_VERSION = 1
HEADER_BYTES = 512
REGION_ALIGNMENT = 4096
HEADER_U32_FIELD_COUNT = 16
HEADER_PREFIX_STRUCT = struct.Struct(
    f"<{MAGIC_FIELD_BYTES}s{HEADER_U32_FIELD_COUNT}I"
)
REGION_STRUCT = struct.Struct("<QQ")
REGION_COUNT = 4
FP8_SCALE_BLOCK = MODEL_CONTRACT["fp8_scale_block"]
FLOAT32_BYTES = struct.calcsize("<f")
CUDA_ARCHITECTURE_SM121 = 121
GATE_UP_ORDER_UP_GATE = 1
WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR = 1
SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR = 1
QUANT_MODE_FP8_E4M3 = 2
OUTPUT_DTYPE_BF16 = 1
DEFAULT_MAX_ACTIVE_SEQUENCE_COUNT = 1024
REGION_W1_WEIGHT = 0
REGION_W1_SCALE_INV = 1
REGION_W2_WEIGHT = 2
REGION_W2_SCALE_INV = 3


@dataclass(frozen=True)
class Fp8MoePackHeader:
    magic: bytes
    abi_version: int
    header_bytes: int
    layer_index: int
    maximum_token_count: int
    hidden_dimension: int
    intermediate_dimension: int
    expert_count: int
    top_k: int
    gate_up_order: int
    weight_layout: int
    scale_layout: int
    quant_mode: int
    output_dtype: int
    cuda_architecture: int
    reserved0: int
    reserved1: int


def fp8_scale_inv_row_block_major_byte_count(rows: int, column_blocks: int) -> int:
    if rows <= 0 or column_blocks <= 0:
        raise PackFailure("FP8 scale dimensions must be positive")
    return rows * column_blocks * FLOAT32_BYTES


def fp8_scale_inv_bytes_to_runtime_row_block_major(
    name: str,
    row_major: bytes,
    rows: int,
    column_blocks: int,
) -> bytes:
    expected_bytes = fp8_scale_inv_row_block_major_byte_count(rows, column_blocks)
    if len(row_major) != expected_bytes:
        raise PackFailure(
            f"{name} scale byte count {len(row_major)}, expected {expected_bytes} "
            f"for runtime shape ({rows}, {column_blocks})"
        )
    return row_major


def transposed_fp8_scale_inv_bytes_to_runtime_row_block_major(
    name: str,
    transposed: bytes,
    rows: int,
    column_blocks: int,
) -> bytes:
    expected_bytes = fp8_scale_inv_row_block_major_byte_count(rows, column_blocks)
    if len(transposed) != expected_bytes:
        raise PackFailure(
            f"{name} transposed scale byte count {len(transposed)}, "
            f"expected {expected_bytes} for checkpoint shape ({column_blocks}, {rows})"
        )
    runtime = bytearray(expected_bytes)
    for row in range(rows):
        for column_block in range(column_blocks):
            source = ((column_block * rows) + row) * FLOAT32_BYTES
            target = ((row * column_blocks) + column_block) * FLOAT32_BYTES
            runtime[target:target + FLOAT32_BYTES] = \
                transposed[source:source + FLOAT32_BYTES]
    return bytes(runtime)


def scale_extent(dimension: int) -> int:
    return (dimension + FP8_SCALE_BLOCK - 1) // FP8_SCALE_BLOCK


def scale_shape(rows: int, columns: int) -> Tuple[int, int]:
    return scale_extent(rows), scale_extent(columns)


def reserve_regions(intermediate_dimension: Optional[int] = None) -> List[Dict[str, int]]:
    if intermediate_dimension is None:
        intermediate_dimension = INTERMEDIATE_DIMENSION
    w1_rows = W1_COMPONENT_COUNT * intermediate_dimension
    w1_weight_bytes = EXPERT_COUNT * w1_rows * HIDDEN_DIMENSION
    w1_scale_bytes = (
        EXPERT_COUNT * scale_extent(w1_rows) * scale_extent(HIDDEN_DIMENSION) *
        FLOAT32_BYTES
    )
    w2_weight_bytes = EXPERT_COUNT * HIDDEN_DIMENSION * intermediate_dimension
    w2_scale_bytes = (
        EXPERT_COUNT * scale_extent(HIDDEN_DIMENSION) *
        scale_extent(intermediate_dimension) * FLOAT32_BYTES
    )
    offset = HEADER_BYTES
    regions: List[Dict[str, int]] = []
    for byte_count in (w1_weight_bytes, w1_scale_bytes, w2_weight_bytes, w2_scale_bytes):
        offset = align_up(offset, REGION_ALIGNMENT)
        regions.append({"offset": offset, "bytes": byte_count})
        offset += byte_count
    return regions


def expected_pack_header(layer: int, max_active: int, tp_degree: int = 1, tp_rank: int = 0) -> Fp8MoePackHeader:
    if max_active <= 0:
        raise PackFailure("maximum active sequence count must be positive")
    if INTERMEDIATE_DIMENSION % tp_degree != 0:
        raise PackFailure("intermediate not divisible by tp degree")
    return Fp8MoePackHeader(
        magic=WIRE_MAGIC,
        abi_version=ABI_VERSION,
        header_bytes=HEADER_BYTES,
        layer_index=layer,
        maximum_token_count=max_active,
        hidden_dimension=HIDDEN_DIMENSION,
        intermediate_dimension=INTERMEDIATE_DIMENSION // tp_degree,
        expert_count=EXPERT_COUNT,
        top_k=TOP_K,
        gate_up_order=GATE_UP_ORDER_UP_GATE,
        weight_layout=WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR,
        scale_layout=SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR,
        quant_mode=QUANT_MODE_FP8_E4M3,
        output_dtype=OUTPUT_DTYPE_BF16,
        cuda_architecture=CUDA_ARCHITECTURE_SM121,
        reserved0=tp_degree,
        reserved1=tp_rank,
    )


def pack_header(layer: int, regions: List[Dict[str, int]], max_active: int, tp_degree: int = 1, tp_rank: int = 0) -> bytes:
    header_fields = expected_pack_header(layer, max_active, tp_degree, tp_rank)
    prefix = HEADER_PREFIX_STRUCT.pack(
        header_fields.magic,
        header_fields.abi_version,
        header_fields.header_bytes,
        header_fields.layer_index,
        header_fields.maximum_token_count,
        header_fields.hidden_dimension,
        header_fields.intermediate_dimension,
        header_fields.expert_count,
        header_fields.top_k,
        header_fields.gate_up_order,
        header_fields.weight_layout,
        header_fields.scale_layout,
        header_fields.quant_mode,
        header_fields.output_dtype,
        header_fields.cuda_architecture,
        header_fields.reserved0,
        header_fields.reserved1,
    )
    region_bytes = b"".join(
        REGION_STRUCT.pack(region["offset"], region["bytes"])
        for region in regions
    )
    header = prefix + region_bytes
    if len(header) > HEADER_BYTES:
        raise PackFailure("FP8 pack header exceeds fixed header size")
    return header + (b"\0" * (HEADER_BYTES - len(header)))


def unpack_pack_header(
    header: bytes,
) -> Tuple[Fp8MoePackHeader, List[Dict[str, int]]]:
    if len(header) != HEADER_BYTES:
        raise PackFailure("short FP8 pack header")
    prefix = HEADER_PREFIX_STRUCT.unpack(header[:HEADER_PREFIX_STRUCT.size])
    fields = Fp8MoePackHeader(*prefix)
    regions = []
    offset = HEADER_PREFIX_STRUCT.size
    for _ in range(REGION_COUNT):
        region_offset, region_bytes = REGION_STRUCT.unpack(
            header[offset:offset + REGION_STRUCT.size]
        )
        regions.append({"offset": region_offset, "bytes": region_bytes})
        offset += REGION_STRUCT.size
    return fields, regions


def existing_pack_header(
    path: Path,
) -> Tuple[Fp8MoePackHeader, List[Dict[str, int]]]:
    with path.open("rb") as file:
        header = file.read(HEADER_BYTES)
    try:
        return unpack_pack_header(header)
    except PackFailure as error:
        raise PackFailure(f"{error}: {path}") from error


def existing_pack_can_reuse(path: Path, layer: int, expected_bytes: int, max_active: int, tp_degree: int = 1, tp_rank: int = 0) -> bool:
    if not path.exists() or path.stat().st_size != expected_bytes:
        return False
    header, regions = existing_pack_header(path)
    expected_regions = reserve_regions(INTERMEDIATE_DIMENSION // tp_degree)
    if regions != expected_regions:
        return False
    if header != expected_pack_header(layer, header.maximum_token_count, tp_degree, tp_rank):
        return False
    if header.maximum_token_count < max_active:
        with path.open("r+b") as file:
            file.write(pack_header(layer, regions, max_active, tp_degree, tp_rank))
    return True


def seek_region(file: BinaryIO, regions: List[Dict[str, int]], region_index: int) -> None:
    file.seek(regions[region_index]["offset"])


def write_tensor_bytes(file: BinaryIO, tensor: Any, expected_shape: Tuple[int, ...], name: str) -> None:
    torch = import_torch()
    if tuple(tensor.shape) != expected_shape:
        raise PackFailure(f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}")
    if "float8_e4m3" not in str(tensor.dtype):
        raise PackFailure(f"{name} has dtype {tensor.dtype}, expected FP8 E4M3")
    file.write(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def fp8_scale_tensor_to_runtime_bytes(
    tensor: Any,
    expected_shape: Tuple[int, int],
    name: str,
    allow_transposed_scales: bool,
) -> bytes:
    shape = tuple(tensor.shape)
    if shape == expected_shape:
        source_tensor = tensor
    elif allow_transposed_scales and shape == (expected_shape[1], expected_shape[0]):
        source_tensor = tensor.transpose(0, 1)
    else:
        raise PackFailure(f"{name} has shape {shape}, expected {expected_shape}")
    if str(source_tensor.dtype) != "torch.float32":
        raise PackFailure(f"{name} has dtype {source_tensor.dtype}, expected float32")
    row_major = source_tensor.contiguous().numpy().astype("<f4", copy=False).tobytes()
    return fp8_scale_inv_bytes_to_runtime_row_block_major(
        name,
        row_major,
        expected_shape[0],
        expected_shape[1],
    )


def write_scale_bytes(
    file: BinaryIO,
    tensor: Any,
    expected_shape: Tuple[int, int],
    name: str,
    allow_transposed_scales: bool,
) -> None:
    file.write(
        fp8_scale_tensor_to_runtime_bytes(
            tensor,
            expected_shape,
            name,
            allow_transposed_scales,
        )
    )


def slice_scale_leading_blocks(tensor: Any, full_shape: Tuple[int, int], block_start: int, block_count: int, name: str) -> Any:
    """Slice the leading (row-block) axis of a scale tensor in whichever
    orientation the checkpoint stored it; downstream orientation handling is
    unchanged. Column slices are made contiguous for byte export."""
    shape = tuple(tensor.shape)
    if shape == full_shape:
        return tensor[block_start:block_start + block_count]
    if shape == (full_shape[1], full_shape[0]):
        return tensor[:, block_start:block_start + block_count].contiguous()
    raise PackFailure(f"{name} has shape {shape}, expected {full_shape} or transposed")


def slice_scale_trailing_blocks(tensor: Any, full_shape: Tuple[int, int], block_start: int, block_count: int, name: str) -> Any:
    shape = tuple(tensor.shape)
    if shape == full_shape:
        return tensor[:, block_start:block_start + block_count].contiguous()
    if shape == (full_shape[1], full_shape[0]):
        return tensor[block_start:block_start + block_count]
    raise PackFailure(f"{name} has shape {shape}, expected {full_shape} or transposed")


def write_layer_pack(
    model_dir: Path,
    output_dir: Path,
    layer: int,
    max_active: int,
    reuse: bool,
    allow_transposed_scales: bool,
    tp_degree: int = 1,
    tp_rank: int = 0,
) -> Dict[str, Any]:
    reader = SafetensorReader(model_dir)
    shard_start, shard_count = tp_shard_range(
        INTERMEDIATE_DIMENSION, tp_degree, tp_rank, FP8_SCALE_BLOCK)
    block_start = shard_start // FP8_SCALE_BLOCK
    block_count = shard_count // FP8_SCALE_BLOCK
    regions = reserve_regions(shard_count)
    shape_tag = "" if tp_degree == 1 else f"_tp{tp_degree}r{tp_rank}"
    output_path = output_dir / f"glm52_layer_{layer:04d}_fp8_moe{shape_tag}.spfp8"
    expected_bytes = regions[-1]["offset"] + regions[-1]["bytes"]
    try:
        if reuse and existing_pack_can_reuse(output_path, layer, expected_bytes, max_active, tp_degree, tp_rank):
            return {"layer": layer, "path": str(output_path), "bytes": expected_bytes, "reused": True, "max_active": max_active, "tp_degree": tp_degree, "tp_rank": tp_rank}
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".glm52_layer_{layer:04d}_",
            suffix=".tmp",
            dir=str(output_dir),
            delete=False,
        ) as file:
            tmp_path = Path(file.name)
            file.write(pack_header(layer, regions, max_active, tp_degree, tp_rank))
            seek_region(file, regions, REGION_W1_WEIGHT)
            for expert in range(EXPERT_COUNT):
                write_tensor_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "up_proj", "weight"))[shard_start:shard_start + shard_count],
                    (shard_count, HIDDEN_DIMENSION),
                    f"layer {layer} expert {expert} up weight",
                )
                write_tensor_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "gate_proj", "weight"))[shard_start:shard_start + shard_count],
                    (shard_count, HIDDEN_DIMENSION),
                    f"layer {layer} expert {expert} gate weight",
                )
            seek_region(file, regions, REGION_W1_SCALE_INV)
            for expert in range(EXPERT_COUNT):
                write_scale_bytes(
                    file,
                    slice_scale_leading_blocks(
                        reader.tensor(tensor_name(layer, expert, "up_proj", "weight_scale_inv")),
                        scale_shape(INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION),
                        block_start, block_count,
                        f"layer {layer} expert {expert} up scale_inv"),
                    scale_shape(shard_count, HIDDEN_DIMENSION),
                    f"layer {layer} expert {expert} up scale_inv",
                    allow_transposed_scales,
                )
                write_scale_bytes(
                    file,
                    slice_scale_leading_blocks(
                        reader.tensor(tensor_name(layer, expert, "gate_proj", "weight_scale_inv")),
                        scale_shape(INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION),
                        block_start, block_count,
                        f"layer {layer} expert {expert} gate scale_inv"),
                    scale_shape(shard_count, HIDDEN_DIMENSION),
                    f"layer {layer} expert {expert} gate scale_inv",
                    allow_transposed_scales,
                )
            seek_region(file, regions, REGION_W2_WEIGHT)
            for expert in range(EXPERT_COUNT):
                write_tensor_bytes(
                    file,
                    reader.tensor(tensor_name(layer, expert, "down_proj", "weight"))[:, shard_start:shard_start + shard_count].contiguous(),
                    (HIDDEN_DIMENSION, shard_count),
                    f"layer {layer} expert {expert} down weight",
                )
            seek_region(file, regions, REGION_W2_SCALE_INV)
            for expert in range(EXPERT_COUNT):
                write_scale_bytes(
                    file,
                    slice_scale_trailing_blocks(
                        reader.tensor(tensor_name(layer, expert, "down_proj", "weight_scale_inv")),
                        scale_shape(HIDDEN_DIMENSION, INTERMEDIATE_DIMENSION),
                        block_start, block_count,
                        f"layer {layer} expert {expert} down scale_inv"),
                    scale_shape(HIDDEN_DIMENSION, shard_count),
                    f"layer {layer} expert {expert} down scale_inv",
                    allow_transposed_scales,
                )
        tmp_path.replace(output_path)
        return {"layer": layer, "path": str(output_path), "bytes": expected_bytes, "reused": False, "max_active": max_active, "tp_degree": tp_degree, "tp_rank": tp_rank}
    finally:
        reader.close()


def worker(argument: Tuple[str, str, int, int, bool, bool, int, int]) -> Dict[str, Any]:
    model_dir, output_dir, layer, max_active, reuse, allow_transposed_scales, tp_degree, tp_rank = argument
    return write_layer_pack(
        Path(model_dir),
        Path(output_dir),
        layer,
        max_active,
        reuse,
        allow_transposed_scales,
        tp_degree,
        tp_rank,
    )


def merge_manifest_records(output_dir: Path, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest_path = output_dir / "fp8_moe_pack_manifest.json"
    merged: Dict[int, Dict[str, Any]] = {}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        for record in prior.get("layers", []):
            merged[int(record["layer"])] = record
    for record in records:
        merged[int(record["layer"])] = record
    return [merged[layer] for layer in sorted(merged)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GLM-5.2 FP8 resident MoE packs")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-active", type=int, default=DEFAULT_MAX_ACTIVE_SEQUENCE_COUNT)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--allow-transposed-scales", action="store_true")
    parser.add_argument("--tp-degree", type=int, default=1,
        help="Tensor-parallel degree: each expert's gate and up rows and down "
        "columns are sliced to this rank's intermediate shard, scale blocks "
        "included, producing a per-node pack for the shape.")
    parser.add_argument("--tp-rank", type=int, default=0)
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    jobs = max(1, int(args.jobs))
    max_active = int(args.max_active)
    if max_active <= 0:
        raise PackFailure("--max-active must be positive")
    output_dir = Path(args.output_dir)
    tasks = [
        (
            args.model_dir,
            args.output_dir,
            layer,
            max_active,
            not args.no_reuse,
            bool(args.allow_transposed_scales),
            int(args.tp_degree),
            int(args.tp_rank),
        )
        for layer in layers
    ]
    if jobs == 1:
        records = [worker(task) for task in tasks]
    else:
        with Pool(processes=jobs) as pool:
            records = list(pool.imap_unordered(worker, tasks))
    records = sorted(records, key=lambda record: int(record["layer"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = merge_manifest_records(output_dir, records)
    (output_dir / "fp8_moe_pack_manifest.json").write_text(
        json.dumps(
            {
                "format": "sparkpipe.glm52.fp8.resident_moe_pack.v1",
                "model_dir": str(Path(args.model_dir)),
                "max_active_sequence_count": max_active,
                "allow_transposed_scales": bool(args.allow_transposed_scales),
                "scale_layout": "expert_major_row_block_major_f32_scale_inv",
                "layers": manifest_records,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    for record in records:
        print(
            "fp8_pack layer={layer} bytes={bytes} reused={reused} path={path}".format(
                **record
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
