#!/usr/bin/env python3
"""Shard a validated full DSV4 stage pack for one TP16 rank.

The source pack is the canonical full-model pack produced by
``dsv4_stagepack.py --first-layer 0 --layer-count 43``.  This utility only
changes the dimensions that the TP16 kernels actually shard; replicated
weights remain byte-for-byte identical.  It streams matrix rows, so the
largest expert tensor is never loaded wholesale into host memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from typing import Callable, List, Sequence, Tuple


HEADER = struct.Struct("<16I2Q")
ENTRY = struct.Struct("<6I2Q")
MAGIC = 0x34565344
VERSION = 3
TP_DEGREE = 16
HIDDEN = 4096
QUERY_DIM = 32768
OUTPUT_GROUPS = 8
OUTPUT_LORA = 1024
OUTPUT_GROUP_DIM = QUERY_DIM // OUTPUT_GROUPS
EXPERTS = 256
EXPERT_WIDTH = 2048
VOCAB = 129280

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_U32 = 2
WEIGHT_FP4 = 3
WEIGHT_FP8 = 4

KIND_ATTN_SINK = 0
KIND_WQ_B = 3
KIND_WO_A = 6
KIND_WO_B = 7
KIND_EXPERTS_W1 = 19
KIND_EXPERTS_W2 = 20
KIND_EXPERTS_W3 = 21
KIND_SHARED_W1 = 22
KIND_SHARED_W2 = 23
KIND_SHARED_W3 = 24
KIND_FINAL_NORM = 36
KIND_LM_HEAD = 37
KIND_HC_HEAD_FN = 38
KIND_HC_HEAD_BASE = 39
KIND_HC_HEAD_SCALE = 40

GLOBAL_LAYER = 0xFFFFFFFF


class PackFailure(RuntimeError):
    pass


def payload_bytes(weight: int, rows: int, columns: int) -> int:
    elements = rows * columns
    if weight == WEIGHT_FP4:
        if elements % 2:
            raise PackFailure("FP4 shape is not byte aligned")
        return elements // 2
    if weight in (WEIGHT_F32, WEIGHT_U32):
        return elements * 4
    if weight == WEIGHT_FP8:
        return elements
    return elements * 2


def scale_bytes(weight: int, rows: int, columns: int) -> int:
    if weight == WEIGHT_FP8:
        return rows * ((columns + 127) // 128)
    if weight == WEIGHT_FP4:
        return rows * ((columns + 31) // 32)
    return 0


def element_bytes(weight: int) -> int:
    if weight == WEIGHT_FP4:
        return 0
    if weight == WEIGHT_FP8:
        return 1
    if weight in (WEIGHT_F32, WEIGHT_U32):
        return 4
    return 2


def row_indices(kind: int, rank: int, rows: int) -> List[int]:
    if kind in (KIND_EXPERTS_W1, KIND_EXPERTS_W3):
        per = EXPERT_WIDTH // TP_DEGREE
        return [expert * EXPERT_WIDTH + rank * per + row
                for expert in range(EXPERTS) for row in range(per)]
    if kind == KIND_EXPERTS_W2:
        per = HIDDEN // TP_DEGREE
        return [expert * HIDDEN + rank * per + row
                for expert in range(EXPERTS) for row in range(per)]
    if kind in (KIND_WQ_B, KIND_WO_B, KIND_SHARED_W1, KIND_SHARED_W3,
                KIND_SHARED_W2):
        return list(range(rank * (rows // TP_DEGREE),
                          (rank + 1) * (rows // TP_DEGREE)))
    return list(range(rows))


def column_slice(kind: int, rank: int, columns: int) -> Tuple[int, int]:
    if kind == KIND_WO_A:
        width = OUTPUT_GROUP_DIM // TP_DEGREE
        return rank * width, width
    if kind in (KIND_EXPERTS_W2, KIND_SHARED_W2):
        width = EXPERT_WIDTH // TP_DEGREE
        return rank * width, width
    if kind == KIND_WQ_B:
        return 0, columns
    if kind == KIND_ATTN_SINK:
        width = (columns // TP_DEGREE)
        return rank * width, width
    return 0, columns


def shard_shape(kind: int, rank: int, rows: int, columns: int) -> Tuple[List[int], int]:
    indices = row_indices(kind, rank, rows)
    start, width = column_slice(kind, rank, columns)
    if kind in (KIND_WQ_B, KIND_WO_B, KIND_EXPERTS_W1, KIND_EXPERTS_W2,
                KIND_EXPERTS_W3, KIND_SHARED_W1, KIND_SHARED_W2,
                KIND_SHARED_W3):
        if rows % TP_DEGREE != 0 and kind not in (KIND_EXPERTS_W1,
                                                   KIND_EXPERTS_W2,
                                                   KIND_EXPERTS_W3):
            raise PackFailure(f"kind {kind} rows {rows} not divisible by TP16")
    if kind == KIND_WO_A:
        indices = list(range(rows))
    if start + width > columns:
        raise PackFailure(f"kind {kind} column shard exceeds source shape")
    return indices, width


def selected_global(kind: int, rank: int) -> bool:
    if rank == TP_DEGREE - 1:
        return True
    return kind not in (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_HC_HEAD_FN,
                        KIND_HC_HEAD_BASE, KIND_HC_HEAD_SCALE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            data = file.read(16 * 1024 * 1024)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def copy_payload(source, destination, offset: int, weight: int, rows: int,
                 columns: int, indices: Sequence[int], col_start: int,
                 new_columns: int) -> None:
    source_row_bytes = payload_bytes(weight, 1, columns)
    new_row_bytes = payload_bytes(weight, 1, new_columns)
    if weight == WEIGHT_FP4:
        if col_start % 2 or new_columns % 2:
            raise PackFailure("FP4 column shard is not byte aligned")
        byte_start = col_start // 2
    else:
        byte_start = col_start * element_bytes(weight)
    for source_row in indices:
        if source_row < 0 or source_row >= rows:
            raise PackFailure("row shard exceeds source shape")
        source.seek(offset + source_row * source_row_bytes + byte_start)
        data = source.read(new_row_bytes)
        if len(data) != new_row_bytes:
            raise PackFailure("short stage-pack payload")
        destination.write(data)


def copy_scales(source, destination, offset: int, weight: int, rows: int,
                columns: int, indices: Sequence[int], col_start: int,
                new_columns: int) -> None:
    if weight not in (WEIGHT_FP4, WEIGHT_FP8):
        return
    block = 32 if weight == WEIGHT_FP4 else 128
    source_blocks = (columns + block - 1) // block
    new_blocks = (new_columns + block - 1) // block
    block_start = col_start // block
    for source_row in indices:
        source.seek(offset + source_row * source_blocks + block_start)
        data = source.read(new_blocks)
        if len(data) != new_blocks:
            raise PackFailure("short stage-pack scale data")
        destination.write(data)


def plan_entry(entry: Tuple[int, ...], rank: int) -> Tuple[Tuple[int, ...], List[int], int, int]:
    kind, layer, weight, rows, columns, reserved, payload, scale = entry
    if layer == GLOBAL_LAYER and not selected_global(kind, rank):
        raise PackFailure("filtered")
    indices, new_columns = shard_shape(kind, rank, rows, columns)
    return ((kind, layer, weight, len(indices), new_columns, reserved, 0, 0),
            indices, column_slice(kind, rank, columns)[0],
            scale_bytes(weight, len(indices), new_columns))


def shard_pack(input_path: Path, output_path: Path, rank: int) -> dict:
    if rank < 0 or rank >= TP_DEGREE:
        raise PackFailure("rank must be in [0,15]")
    with input_path.open("rb") as source:
        header_raw = source.read(HEADER.size)
        if len(header_raw) != HEADER.size:
            raise PackFailure("short stage-pack header")
        header = list(HEADER.unpack(header_raw))
        if header[0] != MAGIC or header[1] != VERSION:
            raise PackFailure("input is not a DSV4 stage pack")
        if header[9] != 0 or header[10] != 43:
            raise PackFailure("input must cover the complete 43-layer model")
        source.seek(header[16])
        entries = [ENTRY.unpack(source.read(ENTRY.size)) for _ in range(header[8])]
        plans = []
        for entry in entries:
            try:
                plans.append((plan_entry(entry, rank), entry))
            except PackFailure as error:
                if str(error) == "filtered":
                    continue
                raise
        cursor = HEADER.size + ENTRY.size * len(plans)
        output_entries = []
        for (planned, original) in plans:
            new_entry, indices, col_start, new_scale_bytes = planned
            kind, layer, weight, rows, columns, reserved, payload, scale = original
            payload_offset = cursor
            cursor += payload_bytes(weight, len(indices), new_entry[4])
            scale_offset = 0
            if new_scale_bytes:
                scale_offset = cursor
                cursor += new_scale_bytes
            output_entries.append((new_entry[:6] + (payload_offset, scale_offset),
                                   original, indices, col_start))
        header[8] = len(output_entries)
        header[9] = 0
        header[10] = 43
        header[16] = HEADER.size
        header[17] = cursor
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{output_path.name}.",
                                         suffix=".tmp", dir=output_path.parent,
                                         delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(HEADER.pack(*header))
            for output_entry, _, _, _ in output_entries:
                temporary.write(ENTRY.pack(*output_entry))
            for output_entry, original, indices, col_start in output_entries:
                kind, layer, weight, rows, columns, reserved, payload, scale = original
                copy_payload(source, temporary, payload, weight, rows, columns,
                             indices, col_start, output_entry[4])
                if output_entry[7]:
                    copy_scales(source, temporary, scale, weight, rows, columns,
                                indices, col_start, output_entry[4])
            temporary.flush()
            temporary_path.replace(output_path)
    digest = sha256_file(output_path)
    return {"file": str(output_path), "rank": rank, "tp_degree": TP_DEGREE,
            "bytes": output_path.stat().st_size,
            "tensor_count": len(output_entries), "sha256": digest,
            "validated": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(shard_pack(args.input_pack, args.output, args.rank),
                         indent=2, sort_keys=True))
    except (OSError, PackFailure, struct.error) as error:
        print(f"dsv4_tp16_stagepack: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
