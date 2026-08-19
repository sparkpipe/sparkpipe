#!/usr/bin/env python3
"""Shard a validated full DSV4 stage pack for one TP rank and PP stage.

The source pack is the canonical full-model pack produced by
``dsv4_stagepack.py --first-layer 0 --layer-count N`` (43 for Flash, 61 for
Pro; --model selects the geometry). This utility only
changes the dimensions that the tensor-parallel kernels actually shard and
retains only the requested pipeline stage's balanced layer slice. Replicated
weights remain byte-for-byte identical. It streams matrix rows, so the largest
expert tensor is never loaded wholesale into host memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Callable, List, Sequence, Tuple

# Make the sibling shared packer core importable whether this script runs
# standalone, via `import tools.dsv4_tp16_stagepack`, or via importlib.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from spark_pack_common import (  # noqa: E402
    PackFailure,
    sha256_file,
    spark_pack_replicated_draft_rows,
)


HEADER = struct.Struct("<16I2Q")
ENTRY = struct.Struct("<6I2Q")
MAGIC = 0x34565344

# One parameterized sharder serves both DSV4 variants. The former Pro copy
# (dsv4_pro_tp16_stagepack.py) duplicated 426 of this file's 444 lines; per
# the tree's DRY law the model facts live in this one table instead of in
# parallel machinery. apply_model_geometry() selects them before use.
MODEL_GEOMETRY = {
    "flash": {
        "layers": 43,
        "version": 4,
        "mtp": True,
        "hidden": 4096,
        "query_dim": 32768,
        "output_groups": 8,
        "experts": 256,
        "expert_width": 2048,
    },
    "pro": {
        "layers": 61,
        "version": 4,
        "mtp": True,
        "hidden": 7168,
        "query_dim": 65536,
        "output_groups": 16,
        "experts": 384,
        "expert_width": 3072,
    },
}
MTP_LAYER_FIRST = 0xFFFFFFFB
MTP_LAYER_COUNT_MAX = 3
TP_DEGREE = 16
LAYERS = MODEL_GEOMETRY["flash"]["layers"]
VERSION = MODEL_GEOMETRY["flash"]["version"]
MTP_ENABLED = MODEL_GEOMETRY["flash"]["mtp"]
HIDDEN = MODEL_GEOMETRY["flash"]["hidden"]
QUERY_DIM = MODEL_GEOMETRY["flash"]["query_dim"]
OUTPUT_GROUPS = MODEL_GEOMETRY["flash"]["output_groups"]
OUTPUT_LORA = 1024
OUTPUT_GROUP_DIM = QUERY_DIM // OUTPUT_GROUPS
EXPERTS = MODEL_GEOMETRY["flash"]["experts"]
EXPERT_WIDTH = MODEL_GEOMETRY["flash"]["expert_width"]
VOCAB = 129280
VOCAB_TILE_ROWS = 128


def apply_model_geometry(model: str) -> None:
    global LAYERS, VERSION, MTP_ENABLED, HIDDEN, QUERY_DIM, OUTPUT_GROUPS
    global OUTPUT_GROUP_DIM, EXPERTS, EXPERT_WIDTH
    geometry = MODEL_GEOMETRY[model]
    LAYERS = geometry["layers"]
    VERSION = geometry["version"]
    MTP_ENABLED = geometry["mtp"]
    HIDDEN = geometry["hidden"]
    QUERY_DIM = geometry["query_dim"]
    OUTPUT_GROUPS = geometry["output_groups"]
    OUTPUT_GROUP_DIM = QUERY_DIM // OUTPUT_GROUPS
    EXPERTS = geometry["experts"]
    EXPERT_WIDTH = geometry["expert_width"]

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_U32 = 2
WEIGHT_FP4 = 3
WEIGHT_FP8 = 4

KIND_ATTN_SINK = 0
KIND_WQ_A = 1
KIND_WQ_B = 3
KIND_WKV = 4
KIND_WO_A = 6
KIND_WO_B = 7
KIND_EXPERTS_W1 = 19
KIND_EXPERTS_W2 = 20
KIND_EXPERTS_W3 = 21
KIND_SHARED_W1 = 22
KIND_SHARED_W2 = 23
KIND_SHARED_W3 = 24
KIND_COMPRESS_WKV = 26
KIND_COMPRESS_WGATE = 27
KIND_INDEX_WKV = 32
KIND_INDEX_WGATE = 33
KIND_EMBEDDING = 35
KIND_FINAL_NORM = 36
KIND_LM_HEAD = 37
KIND_HC_HEAD_FN = 38
KIND_HC_HEAD_BASE = 39
KIND_HC_HEAD_SCALE = 40
KIND_MTP_MAIN_PROJ = 41
KIND_MTP_MAIN_NORM = 42
KIND_MTP_FINAL_NORM = 43
KIND_MTP_HC_HEAD_FN = 44
KIND_MTP_HC_HEAD_BASE = 45
KIND_MTP_HC_HEAD_SCALE = 46
KIND_MTP_MARKOV_W1 = 47
KIND_MTP_MARKOV_W2 = 48
KIND_MTP_CONFIDENCE_PROJ = 49
KIND_MTP_SET = frozenset((
    KIND_MTP_MAIN_PROJ, KIND_MTP_MAIN_NORM, KIND_MTP_FINAL_NORM,
    KIND_MTP_HC_HEAD_FN, KIND_MTP_HC_HEAD_BASE, KIND_MTP_HC_HEAD_SCALE,
    KIND_MTP_MARKOV_W1, KIND_MTP_MARKOV_W2, KIND_MTP_CONFIDENCE_PROJ,
))

GLOBAL_LAYER = 0xFFFFFFFF


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


def validate_tp_degree() -> None:
    if TP_DEGREE not in (1, 2, 4, 8, 16):
        raise PackFailure(f"unsupported TP degree {TP_DEGREE}")
    if (QUERY_DIM % TP_DEGREE or EXPERT_WIDTH % TP_DEGREE or
            OUTPUT_GROUPS % min(TP_DEGREE, OUTPUT_GROUPS)):
        raise PackFailure(f"DSV4 dimensions do not support TP{TP_DEGREE}")


def output_group_shard(rank: int) -> Tuple[int, int, int, int]:
    """Return group start/count and the within-group input column shard."""
    validate_tp_degree()
    if rank < 0 or rank >= TP_DEGREE:
        raise PackFailure(f"rank {rank} does not address TP{TP_DEGREE}")
    ranks_per_group = max(1, TP_DEGREE // OUTPUT_GROUPS)
    group_count = max(1, OUTPUT_GROUPS // TP_DEGREE)
    group_start = (rank // ranks_per_group) * group_count
    column_width = OUTPUT_GROUP_DIM // ranks_per_group
    column_start = (rank % ranks_per_group) * column_width
    return group_start, group_count, column_start, column_width


def vocabulary_shard(rank: int) -> Tuple[int, int]:
    validate_tp_degree()
    if rank < 0 or rank >= TP_DEGREE:
        raise PackFailure(f"rank {rank} does not address TP{TP_DEGREE}")
    if VOCAB % VOCAB_TILE_ROWS:
        raise PackFailure("vocabulary is not native-tile aligned")
    base, remainder = divmod(VOCAB // VOCAB_TILE_ROWS, TP_DEGREE)
    start_tile = rank * base + min(rank, remainder)
    tile_count = base + (1 if rank < remainder else 0)
    return start_tile * VOCAB_TILE_ROWS, tile_count * VOCAB_TILE_ROWS


def row_indices(kind: int, rank: int, rows: int) -> List[int]:
    if kind == KIND_LM_HEAD:
        if rows != VOCAB:
            raise PackFailure(f"lm_head rows {rows} do not match vocabulary")
        start, count = vocabulary_shard(rank)
        return list(range(start, start + count))
    if kind in (KIND_EXPERTS_W1, KIND_EXPERTS_W3):
        per = EXPERT_WIDTH // TP_DEGREE
        return [expert * EXPERT_WIDTH + rank * per + row
                for expert in range(EXPERTS) for row in range(per)]
    if kind == KIND_WO_A:
        group_start, group_count, _, _ = output_group_shard(rank)
        return [group * OUTPUT_LORA + row
                for group in range(group_start, group_start + group_count)
                for row in range(OUTPUT_LORA)]
    if kind in (KIND_WQ_A, KIND_WQ_B, KIND_WKV,
                KIND_COMPRESS_WKV, KIND_COMPRESS_WGATE,
                KIND_INDEX_WKV, KIND_INDEX_WGATE,
                KIND_SHARED_W1, KIND_SHARED_W3):
        return list(range(rank * (rows // TP_DEGREE),
                          (rank + 1) * (rows // TP_DEGREE)))
    return list(range(rows))


def column_slice(kind: int, rank: int, columns: int) -> Tuple[int, int]:
    if kind == KIND_WO_A:
        _, _, start, width = output_group_shard(rank)
        return start, width
    if kind == KIND_WO_B:
        group_start, group_count, _, _ = output_group_shard(rank)
        return group_start * OUTPUT_LORA, group_count * OUTPUT_LORA
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
    validate_tp_degree()
    indices = row_indices(kind, rank, rows)
    start, width = column_slice(kind, rank, columns)
    if kind in (KIND_WQ_A, KIND_WQ_B, KIND_WKV,
                KIND_COMPRESS_WKV, KIND_COMPRESS_WGATE,
                KIND_INDEX_WKV, KIND_INDEX_WGATE,
                KIND_EXPERTS_W1, KIND_EXPERTS_W3,
                KIND_SHARED_W1, KIND_SHARED_W3):
        if rows % TP_DEGREE != 0 and kind not in (KIND_EXPERTS_W1,
                                                   KIND_EXPERTS_W3):
            raise PackFailure(f"kind {kind} rows {rows} not divisible by TP{TP_DEGREE}")
    if start + width > columns:
        raise PackFailure(f"kind {kind} column shard exceeds source shape")
    return indices, width


def layer_slice(pp_stages: int, pp_stage: int) -> Tuple[int, int]:
    if pp_stages < 1 or pp_stages > 16 or pp_stage < 0 or pp_stage >= pp_stages:
        raise PackFailure("PP stage must address one of at most sixteen stages")
    base, remainder = divmod(LAYERS, pp_stages)
    if base == 0:
        raise PackFailure("PP degree exceeds the DSV4 layer count")
    count = base + (1 if pp_stage < remainder else 0)
    first = pp_stage * base + min(pp_stage, remainder)
    return first, count


def selected_global(kind: int, rank: int, pp_stages: int = 1,
                    pp_stage: int = 0) -> bool:
    if kind == KIND_EMBEDDING:
        return pp_stage == 0
    if spark_pack_replicated_draft_rows(
            kind, GLOBAL_LAYER, draft_layer_first=MTP_LAYER_FIRST,
            draft_layer_count=MTP_LAYER_COUNT_MAX, global_kinds=KIND_MTP_SET):
        # The DSpark draft extras replicate in full to every rank.
        return True
    if pp_stage + 1 != pp_stages:
        return False
    if kind in (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_HC_HEAD_FN,
                KIND_HC_HEAD_BASE, KIND_HC_HEAD_SCALE):
        return True
    return True


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


def plan_entry(entry: Tuple[int, ...], rank: int, pp_stages: int = 1,
               pp_stage: int = 0) -> Tuple[Tuple[int, ...], List[int], int, int]:
    kind, layer, weight, rows, columns, reserved, payload, scale = entry
    first_layer, layer_count = layer_slice(pp_stages, pp_stage)
    if (MTP_ENABLED and spark_pack_replicated_draft_rows(
            kind, layer, draft_layer_first=MTP_LAYER_FIRST,
            draft_layer_count=MTP_LAYER_COUNT_MAX, global_kinds=KIND_MTP_SET)):
        # DSpark draft layers are REPLICATED full-width on every rank:
        # keep every MTP entry unchanged (the draft runs replicated with
        # zero draft collectives).
        return ((kind, layer, weight, rows, columns, reserved, 0, 0),
                list(range(rows)), 0, scale_bytes(weight, rows, columns))
    if layer == GLOBAL_LAYER:
        if not selected_global(kind, rank, pp_stages, pp_stage):
            raise PackFailure("filtered")
    elif layer < first_layer or layer >= first_layer + layer_count:
        raise PackFailure("filtered")
    indices, new_columns = shard_shape(kind, rank, rows, columns)
    return ((kind, layer, weight, len(indices), new_columns, reserved, 0, 0),
            indices, column_slice(kind, rank, columns)[0],
            scale_bytes(weight, len(indices), new_columns))


def shard_pack(input_path: Path, output_path: Path, rank: int,
               pp_stages: int = 1, pp_stage: int = 0) -> dict:
    if rank < 0 or rank >= TP_DEGREE:
        raise PackFailure("rank must be in [0,15]")
    with input_path.open("rb") as source:
        header_raw = source.read(HEADER.size)
        if len(header_raw) != HEADER.size:
            raise PackFailure("short stage-pack header")
        header = list(HEADER.unpack(header_raw))
        if header[0] != MAGIC or header[1] != VERSION:
            raise PackFailure("input is not a DSV4 stage pack")
        if header[9] != 0 or header[10] != LAYERS:
            raise PackFailure(f"input must cover the complete {LAYERS}-layer model")
        source.seek(header[16])
        entries = [ENTRY.unpack(source.read(ENTRY.size)) for _ in range(header[8])]
        plans = []
        for entry in entries:
            try:
                plans.append((plan_entry(entry, rank, pp_stages, pp_stage), entry))
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
        header[9], header[10] = layer_slice(pp_stages, pp_stage)
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
            "pp_stages": pp_stages, "pp_stage": pp_stage,
            "first_layer": header[9], "layer_count": header[10],
            "bytes": output_path.stat().st_size,
            "tensor_count": len(output_entries), "sha256": digest,
            "validated": True}


def verify_sharded_pack(input_path: Path, output_path: Path, rank: int,
                        pp_stages: int = 1, pp_stage: int = 0) -> dict:
    """Verify output directory geometry against the full source directory."""
    if rank < 0 or rank >= TP_DEGREE:
        raise PackFailure("rank must be in [0,15]")
    with input_path.open("rb") as source:
        source_header = HEADER.unpack(source.read(HEADER.size))
        if source_header[0] != MAGIC or source_header[1] != VERSION:
            raise PackFailure("input is not a DSV4 stage pack")
        source_entries = [ENTRY.unpack(source.read(ENTRY.size))
                          for _ in range(source_header[8])]
    expected = {}
    for entry in source_entries:
        try:
            planned, _, _, _ = plan_entry(entry, rank, pp_stages, pp_stage)
        except PackFailure as error:
            if str(error) == "filtered":
                continue
            raise
        expected[(entry[0], entry[1])] = planned
    file_bytes = output_path.stat().st_size
    with output_path.open("rb") as output:
        output_header = HEADER.unpack(output.read(HEADER.size))
        if output_header[0] != MAGIC or output_header[1] != VERSION:
            raise PackFailure("output is not a DSV4 stage pack")
        output_entries = [ENTRY.unpack(output.read(ENTRY.size))
                          for _ in range(output_header[8])]
    if (output_header[9], output_header[10]) != layer_slice(pp_stages, pp_stage):
        raise PackFailure("output does not cover the requested PP layer slice")
    if output_header[16] != HEADER.size or output_header[17] != file_bytes:
        raise PackFailure("output header size fields are inconsistent")
    if output_header[8] != len(expected) or len(output_entries) != len(expected):
        raise PackFailure("output tensor count does not match source sharding")
    for index, actual in enumerate(output_entries):
        key = (actual[0], actual[1])
        if key not in expected:
            raise PackFailure(f"unexpected output tensor kind={key[0]} layer={key[1]}")
        if actual[:6] != expected[key][:6]:
            raise PackFailure(f"output shape mismatch kind={key[0]} layer={key[1]}")
        if index != 0 and output_entries[index - 1][6] >= actual[6]:
            raise PackFailure("output payload directory is not ordered")
        payload = payload_bytes(actual[2], actual[3], actual[4])
        scales = scale_bytes(actual[2], actual[3], actual[4])
        if actual[6] < HEADER.size + ENTRY.size * output_header[8] or actual[6] + payload > file_bytes:
            raise PackFailure(f"output payload bounds invalid kind={key[0]} layer={key[1]}")
        if scales and (actual[7] != actual[6] + payload or actual[7] + scales > file_bytes):
            raise PackFailure(f"output scale bounds invalid kind={key[0]} layer={key[1]}")
        if not scales and actual[7] != 0:
            raise PackFailure(f"unexpected scale offset kind={key[0]} layer={key[1]}")
    if set((entry[0], entry[1]) for entry in output_entries) != set(expected):
        raise PackFailure("output tensor key set differs from source sharding")
    return {"file": str(output_path), "rank": rank, "tp_degree": TP_DEGREE,
            "pp_stages": pp_stages, "pp_stage": pp_stage,
            "first_layer": output_header[9], "layer_count": output_header[10],
            "bytes": file_bytes, "tensor_count": len(output_entries),
            "validated": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--tp-degree", type=int, default=16,
                        choices=(1, 2, 4, 8, 16))
    parser.add_argument("--pp-stages", type=int, default=1)
    parser.add_argument("--pp-stage", type=int, default=0)
    parser.add_argument("--model", choices=tuple(MODEL_GEOMETRY),
                        default="flash")
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args(argv)
    try:
        global TP_DEGREE
        apply_model_geometry(args.model)
        TP_DEGREE = args.tp_degree
        if args.tp_degree * args.pp_stages > 16:
            raise PackFailure("TP degree times PP stages exceeds sixteen ranks")
        result = (verify_sharded_pack(args.input_pack, args.output, args.rank,
                                      args.pp_stages, args.pp_stage)
                  if args.verify_output else
                  shard_pack(args.input_pack, args.output, args.rank,
                             args.pp_stages, args.pp_stage))
        print(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, PackFailure, struct.error) as error:
        print(f"dsv4_tp16_stagepack: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
