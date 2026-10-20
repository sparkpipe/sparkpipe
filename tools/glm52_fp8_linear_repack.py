#!/usr/bin/env python3
"""Re-emit a GLM TP16 stage pack with the official FP8 linear weights.

extract runs where the blockwise-FP8 checkpoint lives. For one TP rank it
streams the verbatim F8_E4M3 bytes of every spine linear the stage reads as
FP8, each followed by its F32 weight_scale_inv values expanded across each
128-row block and sliced per 128-column block (the routed-expert layout).

merge runs on the node that holds the bf16-linear pack. It proves for every
replaced tensor that bf16(f32(code) * scale) equals the old bf16 payload bit
for bit (the old packer dequantized exactly that way), writes a pack whose
header names linear codec FP8, and copies every other byte unchanged.
No value is requantized: the payload bytes and scales are the checkpoint's.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from glm52_resident_stagepack import (  # noqa: E402
    ALIGNMENT, CODEC_BF16, CODEC_FP8, GLOBAL_LAYER, HEADER_BYTES, ENTRY_BYTES,
    K_ATTN_OUTPUT, K_DENSE_DOWN, K_DENSE_GATE_UP, K_INDEX_K, K_INDEX_Q, K_KV_A,
    K_Q_A, K_Q_B, K_SHARED_DOWN, K_SHARED_GATE_UP, PAYLOAD_BF16,
    PAYLOAD_PACKED_WEIGHT, SCALE_F32, SCALE_NONE, Fp8SourceReader,
    f32_to_bf16_u16, fp8_e4m3_lut)
from spark_pack_common import PackFailure, align_up, tp_shard_range  # noqa: E402

BLOCK = 128
STREAM_MAGIC = b"GLMFP8LINEAR\x00\x00\x00\x01"
HEADER_FORMAT = "<20I2Q65s32s32s32s7x"
ENTRY_FORMAT = "<8I4Q"
LINEAR_CODEC_FIELD = 15
FP8_KINDS = (K_Q_A, K_Q_B, K_KV_A, K_ATTN_OUTPUT, K_INDEX_Q, K_INDEX_K,
             K_DENSE_GATE_UP, K_DENSE_DOWN, K_SHARED_GATE_UP, K_SHARED_DOWN)
COPY_CHUNK = 64 << 20


def sources(kind: int, layer: int) -> Tuple[List[str], str]:
    attn = f"model.layers.{layer}.self_attn"
    mlp = f"model.layers.{layer}.mlp"
    table = {
        K_Q_A: ([f"{attn}.q_a_proj.weight"], ""),
        K_Q_B: ([f"{attn}.q_b_proj.weight"], "rows"),
        K_KV_A: ([f"{attn}.kv_a_proj_with_mqa.weight"], ""),
        K_ATTN_OUTPUT: ([f"{attn}.o_proj.weight"], "cols"),
        K_INDEX_Q: ([f"{attn}.indexer.wq_b.weight"], ""),
        K_INDEX_K: ([f"{attn}.indexer.wk.weight"], ""),
        K_DENSE_GATE_UP: ([f"{mlp}.up_proj.weight", f"{mlp}.gate_proj.weight"], "rows"),
        K_DENSE_DOWN: ([f"{mlp}.down_proj.weight"], "cols"),
        K_SHARED_GATE_UP: ([f"{mlp}.shared_experts.up_proj.weight",
                            f"{mlp}.shared_experts.gate_proj.weight"], "rows"),
        K_SHARED_DOWN: ([f"{mlp}.shared_experts.down_proj.weight"], "cols"),
    }
    return table[kind]


def scale_groups(columns: int) -> int:
    return (columns + BLOCK - 1) // BLOCK


def slice_fp8(reader: Fp8SourceReader, name: str, shard: str, degree: int,
              rank: int) -> Tuple[np.ndarray, np.ndarray]:
    dtype, shape, _ = reader.meta(name)
    if dtype != "F8_E4M3" or len(shape) != 2:
        raise PackFailure(f"{name}: {dtype} {shape}, the stage reads this linear as F8_E4M3")
    rows, cols = shape
    scale_dtype, scale_shape, _ = reader.meta(reader.scale_name(name))
    if scale_dtype != "F32" or tuple(scale_shape) != (scale_groups(rows), scale_groups(cols)):
        raise PackFailure(f"{name}: scale {scale_dtype} {scale_shape} is not a 128x128 F32 tile")
    codes = reader.raw(name).reshape(rows, cols)
    scale = np.repeat(reader.raw(reader.scale_name(name)).view(np.float32).reshape(scale_shape),
                      BLOCK, axis=0)[:rows]
    r0, r1, c0, c1 = 0, rows, 0, cols
    if shard == "rows":
        r0, count = tp_shard_range(rows, degree, rank)
        r1 = r0 + count
    elif shard == "cols":
        c0, count = tp_shard_range(cols, degree, rank)
        c1 = c0 + count
        if c0 % BLOCK != 0 or (count % BLOCK != 0 and c1 != cols):
            raise PackFailure(f"{name}: column shard {c0}+{count} splits a scale block")
    return (np.ascontiguousarray(codes[r0:r1, c0:c1]),
            np.ascontiguousarray(scale[r0:r1, c0 // BLOCK:scale_groups(c1)]))


def layer_kinds(layer: int, first_routed: int, full_indexer) -> List[int]:
    kinds = [K_Q_A, K_Q_B, K_KV_A, K_ATTN_OUTPUT]
    if full_indexer(layer):
        kinds += [K_INDEX_Q, K_INDEX_K]
    kinds += [K_DENSE_GATE_UP, K_DENSE_DOWN] if layer < first_routed else [K_SHARED_GATE_UP, K_SHARED_DOWN]
    return kinds


def full_indexer_of(share: int):
    return lambda layer: layer < 3 or (layer >= 6 and (layer - 6) % share == 0)


def extract(args) -> int:
    reader = Fp8SourceReader(Path(args.model_dir), cache_byte_cap=1 << 30)
    config = reader.config
    layers = config["num_hidden_layers"]
    if layers != args.layers:
        raise PackFailure(f"checkpoint has {layers} decoder layers, the pack has {args.layers}")
    first_routed = config["first_k_dense_replace"]
    share = args.index_share
    out = sys.stdout.buffer
    digest = hashlib.sha256()
    head = STREAM_MAGIC + struct.pack("<4I", args.tp_degree, args.tp_rank, layers, first_routed)
    digest.update(head)
    out.write(head)
    for layer in range(layers):
        for kind in layer_kinds(layer, first_routed, full_indexer_of(share)):
            names, shard = sources(kind, layer)
            parts = [slice_fp8(reader, name, shard, args.tp_degree, args.tp_rank) for name in names]
            codes = np.concatenate([p[0] for p in parts], axis=0)
            scale = np.concatenate([p[1] for p in parts], axis=0)
            head = struct.pack("<4I2Q", kind, layer, codes.shape[0], codes.shape[1], codes.nbytes, scale.nbytes)
            for blob in (head, codes.tobytes(), scale.astype(np.float32).tobytes()):
                digest.update(blob)
                out.write(blob)
    out.write(digest.digest())
    out.flush()
    return 0


class Stream:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, count: int) -> bytes:
        chunks, left = [], count
        while left > 0:
            blob = self.handle.read(left)
            if not blob:
                raise PackFailure(f"stream ended {left} bytes short")
            chunks.append(blob)
            left -= len(blob)
        data = b"".join(chunks)
        self.digest.update(data)
        return data


def drop_cache(handle, offset: int, length: int) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(handle.fileno(), offset, length, os.POSIX_FADV_DONTNEED)


def copy_range(source, target, offset: int, length: int) -> None:
    source.seek(offset)
    done = 0
    while done < length:
        blob = source.read(min(COPY_CHUNK, length - done))
        if not blob:
            raise PackFailure(f"old pack short at {offset + done}")
        target.write(blob)
        done += len(blob)
    drop_cache(source, offset, length)


def merge(args) -> int:
    old_path, new_path = Path(args.old), Path(args.out)
    lut = fp8_e4m3_lut()
    with open(old_path, "rb") as old:
        header = list(struct.unpack(HEADER_FORMAT, old.read(HEADER_BYTES)))
        if header[LINEAR_CODEC_FIELD] != CODEC_BF16:
            raise PackFailure(f"{old_path}: linear codec {header[LINEAR_CODEC_FIELD]}, expected bf16")
        tensor_count, directory_offset = header[6], header[20]
        old.seek(directory_offset)
        entries = [list(struct.unpack(ENTRY_FORMAT, old.read(ENTRY_BYTES))) for _ in range(tensor_count)]
        stream = Stream(sys.stdin.buffer if args.delta == "-" else open(args.delta, "rb"))
        magic = stream.read(len(STREAM_MAGIC) + 16)
        degree, rank, _layers, _first = struct.unpack("<4I", magic[len(STREAM_MAGIC):])
        if magic[:len(STREAM_MAGIC)] != STREAM_MAGIC or degree != header[18] or rank != header[19]:
            raise PackFailure(f"stream is tp {degree} rank {rank}, the pack is tp {header[18]} rank {header[19]}")
        new = [list(e) for e in entries]
        for e in new:
            if e[0] in FP8_KINDS:
                if e[2] != PAYLOAD_BF16 or e[5] != 1:
                    raise PackFailure(f"kind {e[0]} layer {e[1]} is not a single bf16 matrix")
                e[2], e[3], e[4] = PAYLOAD_PACKED_WEIGHT, CODEC_FP8, SCALE_F32
                e[9] = e[6] * e[7]
                e[11] = e[6] * scale_groups(e[7]) * 4
        offset = align_up(directory_offset + tensor_count * ENTRY_BYTES, ALIGNMENT)
        for e in new:
            offset = align_up(offset, ALIGNMENT)
            e[8] = offset
            offset += e[9]
        for e in new:
            if e[11]:
                offset = align_up(offset, ALIGNMENT)
                e[10] = offset
                offset += e[11]
            else:
                e[10] = 0
        file_bytes = offset
        header[LINEAR_CODEC_FIELD] = CODEC_FP8
        header[21] = file_bytes
        pending: Dict[Tuple[int, int], Tuple[bytes, bytes]] = {}
        fp8_scales: Dict[int, bytes] = {}
        replaced = 0
        with open(new_path, "wb") as target:
            target.write(struct.pack(HEADER_FORMAT, *header))
            target.seek(directory_offset)
            for e in new:
                target.write(struct.pack(ENTRY_FORMAT, *e))
            for index, (before, after) in enumerate(zip(entries, new)):
                target.seek(after[8])
                if after[0] not in FP8_KINDS:
                    copy_range(old, target, before[8], before[9])
                    continue
                key = (after[0], after[1])
                while key not in pending:
                    kind, layer, rows, cols, payload_bytes, scale_bytes = struct.unpack("<4I2Q", stream.read(32))
                    if payload_bytes != rows * cols or scale_bytes != rows * scale_groups(cols) * 4:
                        raise PackFailure(f"stream entry kind {kind} layer {layer} has inconsistent sizes")
                    pending[(kind, layer)] = (stream.read(payload_bytes), stream.read(scale_bytes))
                payload, scale = pending.pop(key)
                rows, cols = after[6], after[7]
                if len(payload) != after[9] or len(scale) != after[11]:
                    raise PackFailure(f"kind {key[0]} layer {key[1]}: stream holds {len(payload)}+{len(scale)} bytes, "
                                      f"the pack needs {after[9]}+{after[11]}")
                old.seek(before[8])
                reference = np.frombuffer(old.read(before[9]), dtype=np.uint16).reshape(rows, cols)
                drop_cache(old, before[8], before[9])
                codes = np.frombuffer(payload, dtype=np.uint8).reshape(rows, cols)
                factors = np.repeat(np.frombuffer(scale, dtype=np.float32).reshape(rows, scale_groups(cols)),
                                    BLOCK, axis=1)[:, :cols]
                rebuilt = f32_to_bf16_u16(lut[codes] * factors)
                mismatches = int(np.count_nonzero(rebuilt != reference))
                if mismatches:
                    raise PackFailure(f"kind {key[0]} layer {key[1]}: {mismatches} of {rows * cols} weights differ from "
                                      f"the old bf16 payload after dequantization")
                target.write(payload)
                fp8_scales[index] = scale
                replaced += 1
            for index, (before, after) in enumerate(zip(entries, new)):
                if not after[11]:
                    continue
                target.seek(after[10])
                if index in fp8_scales:
                    target.write(fp8_scales[index])
                else:
                    copy_range(old, target, before[10], before[11])
            target.truncate(file_bytes)
            target.flush()
            os.fsync(target.fileno())
            drop_cache(target, 0, file_bytes)
        trailer = stream.handle.read(32)
        if pending or trailer != stream.digest.digest():
            raise PackFailure(f"stream digest or entry count mismatch ({len(pending)} unconsumed)")
    print(f"merged {replaced} fp8 linears into {new_path} ({file_bytes} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="stream one rank's FP8 linears and scales to stdout")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--tp-degree", type=int, required=True)
    e.add_argument("--tp-rank", type=int, required=True)
    e.add_argument("--layers", type=int, required=True)
    e.add_argument("--index-share", type=int, required=True,
                   help="dsa_index_share_group_layer_count of the model contract")
    m = sub.add_parser("merge", help="write the FP8-linear pack from the old pack and an extract stream")
    m.add_argument("--old", required=True)
    m.add_argument("--delta", required=True, help="extract stream path, or - for stdin")
    m.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return extract(args) if args.command == "extract" else merge(args)
    except PackFailure as failure:
        print(f"glm52_fp8_linear_repack: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
