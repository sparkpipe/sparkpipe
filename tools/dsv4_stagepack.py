#!/usr/bin/env python3
"""Build DeepSeek V4 Flash stage packs from a safetensors checkpoint.

The CUDA stage consumes a compact wire format, not Hugging Face shards.  This
tool is deliberately setup-time code: it reads safetensors headers and streams
payloads into a temporary pack.  The serving path never opens the checkpoint.

The checkpoint stores non-expert FP8 scales per 128x128 tile.  The resident
linear kernels use one E8M0 byte per output row and 128-column block, so the
packer expands each tile scale across its 128 output rows.  Checkpoint FP4
experts are already packed row-major; their payloads and scales are stacked in
numeric expert order without dequantizing them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import re
import struct
import sys
import tempfile
from typing import BinaryIO, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "model_contracts" / "dsv4_flash.json"
SOURCE_INDEX_NAME = "model.safetensors.index.json"

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_U32 = 2
WEIGHT_FP4 = 3
WEIGHT_FP8 = 4

KIND_ATTN_SINK = 0
KIND_WQ_A = 1
KIND_Q_NORM = 2
KIND_WQ_B = 3
KIND_WKV = 4
KIND_KV_NORM = 5
KIND_WO_A = 6
KIND_WO_B = 7
KIND_ATTN_NORM = 8
KIND_FFN_NORM = 9
KIND_HC_ATTN_FN = 10
KIND_HC_FFN_FN = 11
KIND_HC_ATTN_BASE = 12
KIND_HC_FFN_BASE = 13
KIND_HC_ATTN_SCALE = 14
KIND_HC_FFN_SCALE = 15
KIND_GATE_WEIGHT = 16
KIND_GATE_BIAS = 17
KIND_GATE_TID2EID = 18
KIND_EXPERTS_W1 = 19
KIND_EXPERTS_W2 = 20
KIND_EXPERTS_W3 = 21
KIND_SHARED_W1 = 22
KIND_SHARED_W2 = 23
KIND_SHARED_W3 = 24
KIND_COMPRESS_APE = 25
KIND_COMPRESS_WKV = 26
KIND_COMPRESS_WGATE = 27
KIND_COMPRESS_NORM = 28
KIND_INDEX_WQ_B = 29
KIND_INDEX_WEIGHTS = 30
KIND_INDEX_APE = 31
KIND_INDEX_WKV = 32
KIND_INDEX_WGATE = 33
KIND_INDEX_NORM = 34
KIND_EMBEDDING = 35
KIND_FINAL_NORM = 36
KIND_LM_HEAD = 37
KIND_HC_HEAD_FN = 38
KIND_HC_HEAD_BASE = 39
KIND_HC_HEAD_SCALE = 40
KIND_MTP_E_PROJ = 41
KIND_MTP_H_PROJ = 42
KIND_MTP_ENORM = 43
KIND_MTP_HNORM = 44
KIND_MTP_FINAL_NORM = 45
KIND_MTP_HC_HEAD_FN = 46
KIND_MTP_HC_HEAD_BASE = 47
KIND_MTP_HC_HEAD_SCALE = 48

GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE

HEADER_STRUCT = struct.Struct("<16I2Q")
ENTRY_STRUCT = struct.Struct("<6I2Q")
FORMAT_VERSION = 2
CODEC_ABI_VERSION = 1
CODEC_IDS = {
    "bf16": 1,
    "int6": 2,
    "int7": 3,
    "int8": 4,
    "fp8_e4m3": 5,
    "nvfp4_e2m1": 6,
    "mxfp4_e2m1": 7,
}
FP8_BLOCK = 128
FP4_BLOCK = 32
FP4_EXPERTS = 256
COPY_CHUNK = 16 * 1024 * 1024


class PackFailure(RuntimeError):
    """A source or wire-contract error that must stop pack generation."""


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int
    shard: str

    @property
    def byte_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Record:
    kind: int
    layer: int
    weight_format: int
    rows: int
    columns: int
    source_names: Tuple[str, ...]
    scale_names: Tuple[str, ...]
    source_rows: int
    source_columns: int
    stacked_fp4: bool = False
    i64_to_u32: bool = False

    @property
    def payload_bytes(self) -> int:
        elements = self.rows * self.columns
        if self.weight_format == WEIGHT_FP4:
            return elements // 2
        if self.weight_format in (WEIGHT_F32, WEIGHT_U32):
            return elements * 4
        if self.weight_format == WEIGHT_FP8:
            return elements
        return elements * 2

    @property
    def scale_bytes(self) -> int:
        if self.weight_format == WEIGHT_FP8:
            return self.rows * ((self.columns + FP8_BLOCK - 1) // FP8_BLOCK)
        if self.weight_format == WEIGHT_FP4:
            return self.rows * ((self.columns + FP4_BLOCK - 1) // FP4_BLOCK)
        return 0


@dataclass(frozen=True)
class DirectoryEntry:
    record: Record
    payload_offset: int
    scale_offset: int


def load_contract(path: Path = CONTRACT_PATH) -> Mapping[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PackFailure(f"cannot read DSV4 contract: {path}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(COPY_CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def align_up(value: int, alignment: int = 8) -> int:
    return (value + alignment - 1) // alignment * alignment


def product(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def read_safetensors_header(path: Path) -> Tuple[Mapping[str, object], int]:
    with path.open("rb") as file:
        length_raw = file.read(8)
        if len(length_raw) != 8:
            raise PackFailure(f"short safetensors header length: {path}")
        length = struct.unpack("<Q", length_raw)[0]
        header_raw = file.read(length)
        if len(header_raw) != length:
            raise PackFailure(f"short safetensors header body: {path}")
    header = json.loads(header_raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise PackFailure(f"safetensors header is not an object: {path}")
    return header, 8 + length


class SafetensorSource:
    """Header-indexed range reader that keeps shard handles open for the run."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        index_path = model_dir / SOURCE_INDEX_NAME
        if not index_path.is_file():
            raise PackFailure(f"missing safetensors index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise PackFailure("safetensors index has no weight_map")
        self.weight_map: Mapping[str, str] = {
            str(name): str(shard) for name, shard in weight_map.items()
        }
        self.headers: Dict[str, Mapping[str, object]] = {}
        self.bases: Dict[str, int] = {}
        self.files: Dict[str, BinaryIO] = {}
        for shard in sorted(set(self.weight_map.values())):
            path = model_dir / shard
            if not path.is_file():
                raise PackFailure(f"missing safetensors shard: {path}")
            self.headers[shard], self.bases[shard] = read_safetensors_header(path)

    def close(self) -> None:
        for file in self.files.values():
            file.close()
        self.files.clear()

    def __enter__(self) -> "SafetensorSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def meta(self, name: str) -> TensorMeta:
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        item = self.headers[shard].get(name)
        if not isinstance(item, dict):
            raise PackFailure(f"missing tensor metadata: {name}")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if (not isinstance(dtype, str) or not isinstance(shape, list) or
                not isinstance(offsets, list) or len(offsets) != 2):
            raise PackFailure(f"malformed tensor metadata: {name}")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start:
            raise PackFailure(f"malformed tensor offsets: {name}")
        return TensorMeta(dtype, tuple(int(value) for value in shape), start, end, shard)

    def _file(self, shard: str) -> BinaryIO:
        file = self.files.get(shard)
        if file is None:
            file = (self.model_dir / shard).open("rb")
            self.files[shard] = file
        return file

    def read(self, name: str) -> bytes:
        meta = self.meta(name)
        file = self._file(meta.shard)
        file.seek(self.bases[meta.shard] + meta.start)
        data = file.read(meta.byte_count)
        if len(data) != meta.byte_count:
            raise PackFailure(f"short tensor payload: {name}")
        return data

    def copy(self, name: str, output: BinaryIO) -> None:
        meta = self.meta(name)
        file = self._file(meta.shard)
        file.seek(self.bases[meta.shard] + meta.start)
        remaining = meta.byte_count
        while remaining:
            data = file.read(min(remaining, COPY_CHUNK))
            if not data:
                raise PackFailure(f"short tensor payload: {name}")
            output.write(data)
            remaining -= len(data)


def source_prefix(layer: int) -> str:
    return "mtp.0" if layer == MTP_LAYER else f"layers.{layer}"


def layer_kind(ratios: Sequence[int], layer: int) -> int:
    if layer == MTP_LAYER:
        return 0
    ratio = int(ratios[layer])
    if ratio == 0:
        return 0
    if ratio == 4:
        return 1
    if ratio == 128:
        return 2
    raise PackFailure(f"unsupported DSV4 compression ratio {ratio} at layer {layer}")


def add_record(records: List[Record], kind: int, layer: int, weight_format: int,
               rows: int, columns: int, source_names: Iterable[str],
               scale_names: Iterable[str] = (), source_rows: int | None = None,
               source_columns: int | None = None, stacked_fp4: bool = False,
               i64_to_u32: bool = False) -> None:
    sources = tuple(source_names)
    scales = tuple(scale_names)
    if not sources:
        raise PackFailure(f"record has no source tensor: kind={kind} layer={layer}")
    records.append(Record(
        kind, layer, weight_format, rows, columns, sources, scales,
        source_rows if source_rows is not None else rows,
        source_columns if source_columns is not None else columns,
        stacked_fp4, i64_to_u32,
    ))


def add_linear(records: List[Record], kind: int, prefix: str, layer: int,
               rows: int, columns: int, weight_format: int, name: str) -> None:
    add_record(
        records, kind, layer, weight_format, rows, columns,
        (f"{prefix}.{name}.weight",),
        (f"{prefix}.{name}.scale",) if weight_format in (WEIGHT_FP4, WEIGHT_FP8) else (),
    )


def add_experts(records: List[Record], kind: int, prefix: str, layer: int,
                rows_per_expert: int, columns: int, name: str) -> None:
    add_record(
        records, kind, layer, WEIGHT_FP4, FP4_EXPERTS * rows_per_expert, columns,
        tuple(f"{prefix}.ffn.experts.{expert}.{name}.weight" for expert in range(FP4_EXPERTS)),
        tuple(f"{prefix}.ffn.experts.{expert}.{name}.scale" for expert in range(FP4_EXPERTS)),
        rows_per_expert, columns, True,
    )


def add_layer_records(records: List[Record], ratios: Sequence[int], layer: int) -> None:
    prefix = source_prefix(layer)
    kind = layer_kind(ratios, layer)
    hidden, query_rank, q_dim, head_dim = 4096, 1024, 64 * 512, 512
    expert_rows, expert_columns = 2048, 4096
    add_record(records, KIND_ATTN_SINK, layer, WEIGHT_F32, 1, 64,
               (f"{prefix}.attn.attn_sink",))
    add_linear(records, KIND_WQ_A, f"{prefix}.attn", layer, query_rank, hidden, WEIGHT_FP8, "wq_a")
    add_record(records, KIND_Q_NORM, layer, WEIGHT_BF16, 1, query_rank,
               (f"{prefix}.attn.q_norm.weight",))
    add_linear(records, KIND_WQ_B, f"{prefix}.attn", layer, q_dim, query_rank, WEIGHT_FP8, "wq_b")
    add_linear(records, KIND_WKV, f"{prefix}.attn", layer, head_dim, hidden, WEIGHT_FP8, "wkv")
    add_record(records, KIND_KV_NORM, layer, WEIGHT_BF16, 1, head_dim,
               (f"{prefix}.attn.kv_norm.weight",))
    add_linear(records, KIND_WO_A, f"{prefix}.attn", layer, 8 * 1024, q_dim // 8, WEIGHT_FP8, "wo_a")
    add_linear(records, KIND_WO_B, f"{prefix}.attn", layer, hidden, 8 * 1024, WEIGHT_FP8, "wo_b")
    add_record(records, KIND_ATTN_NORM, layer, WEIGHT_BF16, 1, hidden,
               (f"{prefix}.attn_norm.weight",))
    add_record(records, KIND_FFN_NORM, layer, WEIGHT_BF16, 1, hidden,
               (f"{prefix}.ffn_norm.weight",))
    for kind_id, name in ((KIND_HC_ATTN_FN, "hc_attn_fn"), (KIND_HC_FFN_FN, "hc_ffn_fn")):
        add_record(records, kind_id, layer, WEIGHT_F32, 24, 4 * hidden,
                   (f"{prefix}.{name}",))
    for kind_id, name, columns in ((KIND_HC_ATTN_BASE, "hc_attn_base", 24),
                                   (KIND_HC_FFN_BASE, "hc_ffn_base", 24),
                                   (KIND_HC_ATTN_SCALE, "hc_attn_scale", 3),
                                   (KIND_HC_FFN_SCALE, "hc_ffn_scale", 3)):
        add_record(records, kind_id, layer, WEIGHT_F32, 1, columns,
                   (f"{prefix}.{name}",))
    add_record(records, KIND_GATE_WEIGHT, layer, WEIGHT_BF16, 256, hidden,
               (f"{prefix}.ffn.gate.weight",))
    if layer < 3:
        add_record(records, KIND_GATE_TID2EID, layer, WEIGHT_U32, 129280, 6,
                   (f"{prefix}.ffn.gate.tid2eid",), i64_to_u32=True)
    else:
        add_record(records, KIND_GATE_BIAS, layer, WEIGHT_F32, 1, 256,
                   (f"{prefix}.ffn.gate.bias",))
    add_experts(records, KIND_EXPERTS_W1, prefix, layer, expert_rows, expert_columns, "w1")
    add_experts(records, KIND_EXPERTS_W2, prefix, layer, 4096, 2048, "w2")
    add_experts(records, KIND_EXPERTS_W3, prefix, layer, expert_rows, expert_columns, "w3")
    add_linear(records, KIND_SHARED_W1, f"{prefix}.ffn.shared_experts", layer, expert_rows, expert_columns, WEIGHT_FP8, "w1")
    add_linear(records, KIND_SHARED_W2, f"{prefix}.ffn.shared_experts", layer, hidden, 2048, WEIGHT_FP8, "w2")
    add_linear(records, KIND_SHARED_W3, f"{prefix}.ffn.shared_experts", layer, expert_rows, expert_columns, WEIGHT_FP8, "w3")
    if kind != 0:
        ratio = 4 if kind == 1 else 128
        overlap = 2 if kind == 1 else 1
        channels = overlap * head_dim
        add_record(records, KIND_COMPRESS_APE, layer, WEIGHT_F32, ratio, channels,
                   (f"{prefix}.attn.compressor.ape",))
        add_record(records, KIND_COMPRESS_WKV, layer, WEIGHT_BF16, channels, hidden,
                   (f"{prefix}.attn.compressor.wkv.weight",))
        add_record(records, KIND_COMPRESS_WGATE, layer, WEIGHT_BF16, channels, hidden,
                   (f"{prefix}.attn.compressor.wgate.weight",))
        add_record(records, KIND_COMPRESS_NORM, layer, WEIGHT_BF16, 1, head_dim,
                   (f"{prefix}.attn.compressor.norm.weight",))
    if kind == 1:
        index_prefix = f"{prefix}.attn.indexer"
        add_linear(records, KIND_INDEX_WQ_B, index_prefix, layer, 64 * 128, query_rank, WEIGHT_FP8, "wq_b")
        add_record(records, KIND_INDEX_WEIGHTS, layer, WEIGHT_BF16, 64, hidden,
                   (f"{index_prefix}.weights_proj.weight",))
        add_record(records, KIND_INDEX_APE, layer, WEIGHT_F32, 4, 256,
                   (f"{index_prefix}.compressor.ape",))
        add_record(records, KIND_INDEX_WKV, layer, WEIGHT_BF16, 256, hidden,
                   (f"{index_prefix}.compressor.wkv.weight",))
        add_record(records, KIND_INDEX_WGATE, layer, WEIGHT_BF16, 256, hidden,
                   (f"{index_prefix}.compressor.wgate.weight",))
        add_record(records, KIND_INDEX_NORM, layer, WEIGHT_BF16, 1, 128,
                   (f"{index_prefix}.compressor.norm.weight",))


def build_records(contract: Mapping[str, object], first_layer: int, layer_count: int) -> List[Record]:
    model = contract["model"]
    if not isinstance(model, dict):
        raise PackFailure("contract model section is malformed")
    total_layers = int(model["layer_count"])
    if first_layer < 0 or layer_count <= 0 or first_layer + layer_count > total_layers:
        raise PackFailure(f"invalid layer slice {first_layer}+{layer_count}")
    ratios = contract["attention"]["compression_ratios"]
    if not isinstance(ratios, list) or len(ratios) != total_layers + 1:
        raise PackFailure("contract compression ratio table is malformed")
    records: List[Record] = []
    if first_layer == 0:
        add_record(records, KIND_EMBEDDING, GLOBAL_LAYER, WEIGHT_BF16, 129280, 4096,
                   ("embed.weight",))
    for layer in range(first_layer, first_layer + layer_count):
        add_layer_records(records, ratios, layer)
    if first_layer + layer_count == total_layers:
        if first_layer != 0:
            add_record(records, KIND_EMBEDDING, GLOBAL_LAYER, WEIGHT_BF16, 129280, 4096,
                       ("embed.weight",))
        add_record(records, KIND_FINAL_NORM, GLOBAL_LAYER, WEIGHT_BF16, 1, 4096,
                   ("norm.weight",))
        add_record(records, KIND_LM_HEAD, GLOBAL_LAYER, WEIGHT_BF16, 129280, 4096,
                   ("head.weight",))
        add_record(records, KIND_HC_HEAD_FN, GLOBAL_LAYER, WEIGHT_F32, 4, 16384,
                   ("hc_head_fn",))
        add_record(records, KIND_HC_HEAD_BASE, GLOBAL_LAYER, WEIGHT_F32, 1, 4,
                   ("hc_head_base",))
        add_record(records, KIND_HC_HEAD_SCALE, GLOBAL_LAYER, WEIGHT_F32, 1, 1,
                   ("hc_head_scale",))
        add_record(records, KIND_MTP_E_PROJ, GLOBAL_LAYER, WEIGHT_FP8, 4096, 4096,
                   ("mtp.0.e_proj.weight",), ("mtp.0.e_proj.scale",))
        add_record(records, KIND_MTP_H_PROJ, GLOBAL_LAYER, WEIGHT_FP8, 4096, 4096,
                   ("mtp.0.h_proj.weight",), ("mtp.0.h_proj.scale",))
        add_record(records, KIND_MTP_ENORM, GLOBAL_LAYER, WEIGHT_BF16, 1, 4096,
                   ("mtp.0.enorm.weight",))
        add_record(records, KIND_MTP_HNORM, GLOBAL_LAYER, WEIGHT_BF16, 1, 4096,
                   ("mtp.0.hnorm.weight",))
        add_record(records, KIND_MTP_FINAL_NORM, GLOBAL_LAYER, WEIGHT_BF16, 1, 4096,
                   ("mtp.0.norm.weight",))
        add_record(records, KIND_MTP_HC_HEAD_FN, GLOBAL_LAYER, WEIGHT_F32, 4, 16384,
                   ("mtp.0.hc_head_fn",))
        add_record(records, KIND_MTP_HC_HEAD_BASE, GLOBAL_LAYER, WEIGHT_F32, 1, 4,
                   ("mtp.0.hc_head_base",))
        add_record(records, KIND_MTP_HC_HEAD_SCALE, GLOBAL_LAYER, WEIGHT_F32, 1, 1,
                   ("mtp.0.hc_head_scale",))
        add_layer_records(records, ratios, MTP_LAYER)
    return records


def validate_meta(source: SafetensorSource, record: Record) -> None:
    for name in record.source_names:
        meta = source.meta(name)
        if record.stacked_fp4:
            expected_shape = (record.source_rows, record.source_columns // 2)
            if meta.dtype != "I8" or meta.shape != expected_shape:
                raise PackFailure(
                    f"{name}: expected I8 {expected_shape}, got {meta.dtype} {meta.shape}")
        elif record.i64_to_u32:
            expected_shape = (record.rows, record.columns)
            if meta.dtype != "I64" or meta.shape != expected_shape:
                raise PackFailure(
                    f"{name}: expected I64 {expected_shape}, got {meta.dtype} {meta.shape}")
        else:
            expected_shape = (record.rows, record.columns)
            expected_dtype = {
                WEIGHT_BF16: "BF16", WEIGHT_F32: "F32", WEIGHT_U32: "U32",
                WEIGHT_FP8: "F8_E4M3", WEIGHT_FP4: "I8",
            }[record.weight_format]
            if meta.dtype != expected_dtype or product(meta.shape) != product(expected_shape):
                raise PackFailure(
                    f"{name}: expected {expected_dtype} {expected_shape}, "
                    f"got {meta.dtype} {meta.shape}")
    if record.weight_format in (WEIGHT_FP4, WEIGHT_FP8):
        if len(record.scale_names) != len(record.source_names):
            raise PackFailure(f"scale/source count mismatch for kind={record.kind} layer={record.layer}")
        for name in record.scale_names:
            meta = source.meta(name)
            if meta.dtype != "F8_E8M0":
                raise PackFailure(f"{name}: expected F8_E8M0, got {meta.dtype}")
            if record.stacked_fp4:
                expected_shape = (record.source_rows, record.source_columns // FP4_BLOCK)
            else:
                expected_shape = (
                    (record.rows + FP8_BLOCK - 1) // FP8_BLOCK,
                    (record.columns + FP8_BLOCK - 1) // FP8_BLOCK,
                )
            if meta.shape != expected_shape:
                raise PackFailure(f"{name}: expected scale {expected_shape}, got {meta.shape}")


def validate_records(source: SafetensorSource, records: Sequence[Record]) -> None:
    seen = set()
    for record in records:
        key = (record.layer, record.kind)
        if key in seen:
            raise PackFailure(f"duplicate record layer={record.layer} kind={record.kind}")
        seen.add(key)
        validate_meta(source, record)


def expand_fp8_scale(source: SafetensorSource, name: str, rows: int, columns: int) -> bytes:
    raw = source.read(name)
    column_blocks = (columns + FP8_BLOCK - 1) // FP8_BLOCK
    row_blocks = (rows + FP8_BLOCK - 1) // FP8_BLOCK
    expected = row_blocks * column_blocks
    if len(raw) != expected:
        raise PackFailure(f"{name}: scale byte count {len(raw)}, expected {expected}")
    output = bytearray()
    for row_block in range(row_blocks):
        row = raw[row_block * column_blocks:(row_block + 1) * column_blocks]
        output.extend(row * min(FP8_BLOCK, rows - row_block * FP8_BLOCK))
    return bytes(output)


def copy_record_payload(source: SafetensorSource, record: Record, output: BinaryIO) -> None:
    if record.i64_to_u32:
        raw = source.read(record.source_names[0])
        if len(raw) != record.payload_bytes * 2:
            raise PackFailure(f"{record.source_names[0]}: I64 byte count mismatch")
        values = struct.iter_unpack("<q", raw)
        for (value,) in values:
            if value < 0 or value >= 2**32:
                raise PackFailure(f"{record.source_names[0]}: value outside u32: {value}")
            output.write(struct.pack("<I", value))
    elif record.stacked_fp4:
        for name in record.source_names:
            source.copy(name, output)
    else:
        source.copy(record.source_names[0], output)
    if record.weight_format == WEIGHT_FP8:
        output.write(expand_fp8_scale(source, record.scale_names[0], record.rows, record.columns))
    elif record.weight_format == WEIGHT_FP4:
        for name in record.scale_names:
            source.copy(name, output)


def make_directory(records: Sequence[Record]) -> Tuple[List[DirectoryEntry], int]:
    cursor = HEADER_STRUCT.size + ENTRY_STRUCT.size * len(records)
    entries: List[DirectoryEntry] = []
    for record in records:
        payload_offset = cursor
        cursor += record.payload_bytes
        scale_offset = 0
        if record.scale_bytes:
            scale_offset = cursor
            cursor += record.scale_bytes
        entries.append(DirectoryEntry(record, payload_offset, scale_offset))
    return entries, cursor


def pack_header(records: Sequence[Record], first_layer: int, layer_count: int,
                file_bytes: int, codecs: Tuple[int, int, int]) -> bytes:
    return HEADER_STRUCT.pack(
        0x34565344, FORMAT_VERSION, HEADER_STRUCT.size, ENTRY_STRUCT.size,
        CODEC_ABI_VERSION, *codecs,
        len(records), first_layer, layer_count, 43, 4096, 129280, 256, 1,
        HEADER_STRUCT.size, file_bytes,
    )


def pack_entry(entry: DirectoryEntry) -> bytes:
    record = entry.record
    return ENTRY_STRUCT.pack(
        record.kind, record.layer, record.weight_format, record.rows,
        record.columns, 0, entry.payload_offset, entry.scale_offset,
    )


def build_pack(source: SafetensorSource, output_path: Path, records: Sequence[Record],
               first_layer: int, layer_count: int,
               codecs: Tuple[int, int, int]) -> Dict[str, object]:
    entries, file_bytes = make_directory(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp",
            dir=output_path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        temporary.write(pack_header(records, first_layer, layer_count, file_bytes, codecs))
        for entry in entries:
            temporary.write(pack_entry(entry))
        for entry in entries:
            before = temporary.tell()
            if before != entry.payload_offset:
                raise PackFailure(f"pack cursor mismatch at kind={entry.record.kind} layer={entry.record.layer}")
            copy_record_payload(source, entry.record, temporary)
            if temporary.tell() != entry.payload_offset + entry.record.payload_bytes + entry.record.scale_bytes:
                raise PackFailure(f"payload size mismatch at kind={entry.record.kind} layer={entry.record.layer}")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temp_path, output_path)
    return {
        "file": str(output_path),
        "bytes": file_bytes,
        "sha256": sha256_file(output_path),
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": len(records),
    }


def verify_pack(pack_path: Path, contract: Mapping[str, object],
                codecs: Tuple[int, int, int], include_sha256: bool) -> Dict[str, object]:
    try:
        file_bytes = pack_path.stat().st_size
        with pack_path.open("rb") as file:
            header_raw = file.read(HEADER_STRUCT.size)
            if len(header_raw) != HEADER_STRUCT.size:
                raise PackFailure(f"short stage-pack header: {pack_path}")
            header = HEADER_STRUCT.unpack(header_raw)
            tensor_count = header[8]
            if tensor_count == 0 or tensor_count > 4096:
                raise PackFailure(f"invalid stage-pack tensor count: {tensor_count}")
            directory_raw = file.read(tensor_count * ENTRY_STRUCT.size)
            if len(directory_raw) != tensor_count * ENTRY_STRUCT.size:
                raise PackFailure(f"short stage-pack directory: {pack_path}")
    except OSError as error:
        raise PackFailure(f"cannot read stage pack: {pack_path}") from error
    first_layer = header[9]
    layer_count = header[10]
    records = build_records(contract, first_layer, layer_count)
    entries, expected_file_bytes = make_directory(records)
    expected_header = HEADER_STRUCT.unpack(pack_header(
        records, first_layer, layer_count, expected_file_bytes, codecs))
    if header != expected_header:
        raise PackFailure("stage-pack header does not match the model contract")
    if file_bytes != expected_file_bytes:
        raise PackFailure(
            f"stage-pack size mismatch: header={expected_file_bytes} file={file_bytes}")
    for index, entry in enumerate(entries):
        start = index * ENTRY_STRUCT.size
        actual = directory_raw[start:start + ENTRY_STRUCT.size]
        expected = pack_entry(entry)
        if actual != expected:
            raise PackFailure(f"stage-pack directory mismatch at entry {index}")
    result: Dict[str, object] = {
        "file": str(pack_path),
        "bytes": file_bytes,
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": tensor_count,
        "linear_weight_codec_id": codecs[0],
        "expert_weight_codec_id": codecs[1],
        "kv_cache_codec_id": codecs[2],
        "contract_source_revision": contract.get("source_revision"),
        "validated": True,
    }
    if include_sha256:
        result["sha256"] = sha256_file(pack_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path)
    source.add_argument("--verify-pack", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first-layer", type=int)
    parser.add_argument("--layer-count", type=int)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    arguments = parser.parse_args(argv)
    if arguments.verify_pack is not None:
        if arguments.output is not None or arguments.inspect or arguments.first_layer is not None or arguments.layer_count is not None:
            parser.error("--verify-pack does not accept source-pack arguments")
    elif arguments.first_layer is None or arguments.layer_count is None:
        parser.error("--model-dir requires --first-layer and --layer-count")
    elif arguments.sha256:
        parser.error("--sha256 requires --verify-pack")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_pack is None and not args.inspect and args.output is None:
        print("dsv4_stagepack: --output is required unless --inspect is used", file=sys.stderr)
        return 2
    try:
        contract = load_contract(args.contract)
        precision = contract.get("precision")
        if not isinstance(precision, dict):
            raise PackFailure("contract precision section is malformed")
        codec_names = (
            precision.get("non_expert_linear_weight_codec"),
            precision.get("routed_expert_weight_codec"),
            precision.get("kv_cache_codec"),
        )
        if any(not isinstance(name, str) or name not in CODEC_IDS for name in codec_names):
            raise PackFailure(f"contract contains an unsupported codec tuple: {codec_names}")
        codecs = tuple(CODEC_IDS[name] for name in codec_names)
        if args.verify_pack is not None:
            print(json.dumps(verify_pack(
                args.verify_pack, contract, codecs, args.sha256),
                indent=2, sort_keys=True))
            return 0
        with SafetensorSource(args.model_dir) as source:
            records = build_records(contract, args.first_layer, args.layer_count)
            validate_records(source, records)
            entries, file_bytes = make_directory(records)
            result: Dict[str, object] = {
                "first_layer": args.first_layer,
                "layer_count": args.layer_count,
                "tensor_count": len(records),
                "file_bytes": file_bytes,
                "linear_weight_codec": codec_names[0],
                "expert_weight_codec": codec_names[1],
                "kv_cache_codec": codec_names[2],
                "source_index_sha256": sha256_file(args.model_dir / SOURCE_INDEX_NAME),
                "validated": True,
            }
            if not args.inspect:
                result.update(build_pack(source,args.output,records,args.first_layer,args.layer_count,codecs))
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, PackFailure, json.JSONDecodeError, struct.error) as error:
        print(f"dsv4_stagepack: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
