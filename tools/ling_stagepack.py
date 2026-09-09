#!/usr/bin/env python3
"""Build Ling-3.0-flash (BailingMoeV3) resident-decode stage packs (.lspk).

Reads a warm HF checkpoint (official BF16 release) through header-only
safetensors memmaps and emits one wire-format-v1 .lspk per TP rank. The
tensor vocabulary is the 34-kind table in
modules/ling_resident_decode_stage/source/spark_ling_stagepack_format.h
and every checkpoint->pack transform follows
model-families/ling/name_map.json:

  - kda_qkv_beta = q|k|v|beta rows fused from the checkpoint's separate
    q_proj/k_conv1d(k_proj)/v_proj/b_proj, row order q,k,v,beta to match
    LingSplitFusedProjectionsKernel, sharded PER SECTION.
  - kv_b_proj [8192, 512] split into key-transposed [512, 128] x 32 heads
    and value [128, 512] x 32 heads (both replicated; the module binds
    per-rank head offsets).
  - up|gate row fusions (dense, shared, per-expert) stacked up-first to
    match LmSiluMulKernel's gate_first=false layout.
  - conv1d [channels, kernel] verify-then-pack.
  - routed experts pass through in the source dtype (bf16 for the
    official release; the packer NEVER quantizes). The inter dimension
    shards across ranks (w1 rows, w2 columns); every rank carries all
    512 experts, which is what the indirect grouped GEMM indexes.
  - MTP tensors are OMITTED and the header emits flags = 0 (the
    load-and-ignore contract).

The census must close: every tensor in model.safetensors.index.json is
either packed, omitted MTP, or the run FAILS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_common import PackFailure, sha256_bytes, tp_shard_range  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

MAGIC = 0x33474E4C
FORMAT_VERSION = 1
HEADER_BYTES = 264
ENTRY_BYTES = 64
ALIGNMENT = 256
CODEC_ABI_VERSION = 1
GLOBAL_LAYER = 0xFFFFFFFF

PAYLOAD_BF16 = 1
PAYLOAD_F32 = 2
PAYLOAD_U32 = 3
PAYLOAD_PACKED_WEIGHT = 4

CODEC_NONE = 0
CODEC_BF16 = 1
SCALE_NONE = 0
SCALE_F32 = 1

K_EMBEDDING, K_FINAL_NORM, K_LM_HEAD = 0, 1, 2
K_ATTN_NORM, K_Q, K_KV_A, K_KV_A_NORM = 3, 4, 5, 6
K_KV_B_KEY_T, K_KV_B_VALUE, K_ATTN_GATE, K_ATTN_OUTPUT = 7, 8, 9, 10
K_POST_ATTN_NORM = 11
K_DENSE_GATE_UP, K_DENSE_DOWN = 12, 13
K_ROUTER, K_ROUTER_CORRECTION = 14, 15
K_EXPERT_UP_GATE, K_EXPERT_DOWN = 16, 17
K_SHARED_GATE_UP, K_SHARED_DOWN = 18, 19
K_KDA_QKV_BETA, K_KDA_DECAY_PROJ, K_KDA_GATE_PROJ = 20, 21, 22
K_KDA_Q_CONV, K_KDA_K_CONV, K_KDA_V_CONV = 23, 24, 25
K_KDA_DECAY_BIAS, K_KDA_HEAD_LOG_SCALE, K_KDA_OUT_NORM, K_KDA_OUT = 26, 27, 28, 29
K_MTP_EH_PROJ, K_MTP_ENORM, K_MTP_HNORM, K_MTP_SHARED_NORM = 30, 31, 32, 33

HIDDEN = 2560
LAYERS = 42
MTP_LAYER = 42
KDA_HEADS, KDA_KEY, KDA_CONV = 32, 128, 4
KDA_QK = KDA_HEADS * KDA_KEY
KDA_V = KDA_QK
KDA_FUSED_ROWS = 2 * KDA_QK + KDA_V + KDA_HEADS
MLA_HEADS, LATENT, ROPE, NOPE, VDIM = 32, 512, 64, 128, 128
Q_ROWS = MLA_HEADS * (NOPE + ROPE)
KV_ROW = LATENT + ROPE
EXPERTS, EXPERT_INTER = 512, 768
DENSE_INTER = 6144
VOCAB = 157184
FIRST_ROUTED = 2
CONTRACT = REPO_ROOT / "model_contracts" / "ling_authoritative.json"
NAME_MAP = REPO_ROOT / "model-families" / "ling" / "name_map.json"


def f32_to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
               >> np.uint32(16)).astype(np.uint16)
    return rounded


class SourceReader:
    """Header-only safetensors reader; payloads are memmaps and each
    tensor is materialized at most once per rank pass (small LRU)."""

    def __init__(self, model_dir: Path, cache_byte_cap: int = 512 * 1024 ** 2):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise PackFailure(f"missing safetensors index: {index_path}")
        self.model_dir = model_dir
        self.weight_map: Dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        config_path = model_dir / "config.json"
        self.config: Dict[str, Any] = (json.loads(config_path.read_text())
                                       if config_path.is_file() else {})
        self._mmaps: Dict[str, np.ndarray] = {}
        self._headers: Dict[str, Tuple[dict, int]] = {}
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_bytes = 0
        self._cache_byte_cap = cache_byte_cap

    def _header(self, shard: str) -> Tuple[dict, int]:
        if shard not in self._headers:
            path = self.model_dir / shard
            with path.open("rb") as file:
                header_bytes = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(header_bytes))
            self._headers[shard] = (header, 8 + header_bytes)
        return self._headers[shard]

    def _mmap(self, shard: str) -> np.ndarray:
        if shard not in self._mmaps:
            path = self.model_dir / shard
            size = path.stat().st_size
            self._mmaps[shard] = np.memmap(path, dtype=np.uint8, mode="r", shape=(size,))
        return self._mmaps[shard]

    def meta(self, name: str) -> Tuple[str, Tuple[int, ...]]:
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        header, _ = self._header(shard)
        entry = header.get(name)
        if entry is None:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        return entry["dtype"], tuple(entry["shape"])

    def raw(self, name: str) -> np.ndarray:
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        header, data_start = self._header(shard)
        entry = header.get(name)
        if entry is None:
            raise PackFailure(f"tensor {name} vanished from shard {shard}")
        begin, end = entry["data_offsets"]
        view = self._mmap(shard)[data_start + begin:data_start + end]
        if view.shape[0] != end - begin:
            raise PackFailure(f"short payload for {name}")
        return view

    def matrix2d(self, name: str) -> np.ndarray:
        """[rows, cols] uint16 view of a BF16 checkpoint matrix (conv1d's
        [dim, 1, kernel] squeezes to [dim, kernel])."""
        dtype, shape, = self.meta(name)
        if len(shape) == 3 and shape[1] == 1:
            shape = (shape[0], shape[2])
        if len(shape) == 1:
            shape = (1, shape[0])
        if len(shape) != 2:
            raise PackFailure(f"{name}: expected 2-D, got {shape}")
        rows, cols = shape
        if dtype == "BF16":
            matrix = self.raw(name).view(np.uint16).reshape(rows, cols)
        elif dtype == "F32":
            u32 = self.raw(name).view(np.uint32).reshape(rows, cols)
            rounding = (u32 >> np.uint32(16)) & np.uint32(1)
            return ((u32 + np.uint32(0x7FFF) + rounding)
                    >> np.uint32(16)).astype(np.uint16)
        else:
            raise PackFailure(f"{name}: unsupported spine dtype {dtype}")
        if name not in self._cache:
            self._cache[name] = matrix
            self._cache_bytes += matrix.nbytes
            while self._cache_bytes > self._cache_byte_cap and len(self._cache) > 1:
                victim = next(iter(self._cache))
                self._cache_bytes -= self._cache[victim].nbytes
                del self._cache[victim]
        return matrix

    def vector_f32(self, name: str) -> np.ndarray:
        """Exact f32 vector: F32 verbatim, BF16 upcast (bit-exact)."""
        dtype, _ = self.meta(name)
        if dtype == "F32":
            return self.raw(name).view(np.float32).reshape(-1).copy()
        if dtype == "BF16":
            return (self.raw(name).view(np.uint16).astype(np.uint32)
                    << np.uint32(16)).view(np.float32).astype(np.float32)
        raise PackFailure(f"{name}: f32 upcast from {dtype} unsupported")

    def close(self) -> None:
        self._mmaps.clear()
        self._cache.clear()


def to_bytes(t: np.ndarray) -> bytes:
    return np.ascontiguousarray(t).tobytes()


MASK64 = (1 << 64) - 1


def _rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & MASK64


def _fmix64(k: int) -> int:
    k ^= k >> 33
    k = (k * 0xff51afd7ed558ccd) & MASK64
    k ^= k >> 33
    k = (k * 0xc4ceb9fe1a85ec53) & MASK64
    k ^= k >> 33
    return k


def ck128(data: bytes) -> bytes:
    """Bit-exact port of src/spark_ck128.c (MurmurHash3 x64 128, seed 0)."""
    c1 = 0x87c37b91114253d5
    c2 = 0x4cf5ad432745937f
    h1 = h2 = 0
    whole = len(data) // 16
    for block in range(whole):
        k1, k2 = struct.unpack_from("<QQ", data, block * 16)
        k1 = (k1 * c1) & MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * c2) & MASK64
        h1 ^= k1
        h1 = _rotl64(h1, 27)
        h1 = (h1 + h2) & MASK64
        h1 = (h1 * 5 + 0x52dce729) & MASK64
        k2 = (k2 * c2) & MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * c1) & MASK64
        h2 ^= k2
        h2 = _rotl64(h2, 31)
        h2 = (h2 + h1) & MASK64
        h2 = (h2 * 5 + 0x38495ab5) & MASK64
    tail = data[whole * 16:]
    k1 = k2 = 0
    if len(tail) > 8:
        k2 = struct.unpack("<Q", tail[8:].ljust(8, b"\0"))[0]
        k2 = (k2 * c2) & MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * c1) & MASK64
        h2 ^= k2
    k1 = struct.unpack("<Q", tail[:8].ljust(8, b"\0"))[0] if tail else 0
    k1 = (k1 * c1) & MASK64
    k1 = _rotl64(k1, 31)
    k1 = (k1 * c2) & MASK64
    h1 ^= k1
    total = (len(data) & MASK64)
    h1 ^= total
    h2 ^= total
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    return struct.pack("<QQ", h1, h2)


EXPERT_MANIFEST_MAGIC = 0x58504557
EXPERT_MANIFEST_VERSION = 2
EXPERT_RANGE_BYTES_MAX = 64 * 1024 * 1024


class Entry:
    def __init__(self, kind: int, layer: int, payload_type: int, weight_codec: int,
                 scale_encoding: int, group_count: int, rows: int, columns: int):
        self.kind = kind
        self.layer = layer
        self.payload_type = payload_type
        self.weight_codec = weight_codec
        self.scale_encoding = scale_encoding
        self.group_count = group_count
        self.rows = rows
        self.columns = columns
        self.payload_offset = 0
        self.payload_bytes = 0
        self.scale_offset = 0
        self.scale_bytes = 0


class PlanItem:
    def __init__(self, entry: Entry, produce_payload: Callable[[], Iterator[bytes]]):
        self.entry = entry
        self.produce_payload = produce_payload


class Packer:
    def __init__(self, source: SourceReader, tp_degree: int, tp_rank: int,
                 expert_codec: int):
        self.s = source
        self.tp_degree = tp_degree
        self.tp_rank = tp_rank
        self.expert_codec = expert_codec
        self.plan: List[PlanItem] = []
        self.packed_names: List[str] = []

    def _slice_rows(self, dimension: int) -> Tuple[int, int]:
        return tp_shard_range(dimension, self.tp_degree, self.tp_rank)

    def _track(self, *names: str) -> None:
        self.packed_names.extend(names)

    def add_replicated_bf16(self, kind: int, layer: int, name: str,
                            rows: int, columns: int):
        dtype, shape = self.s.meta(name)
        if len(shape) == 3 and shape[1] == 1:
            shape = (shape[0], shape[2])
        if len(shape) == 1:
            shape = (1, shape[0])
        if tuple(shape) != (rows, columns):
            raise PackFailure(f"{name}: shape {tuple(shape)}, expected {(rows, columns)}")
        if dtype not in ("BF16", "F32"):
            raise PackFailure(f"{name}: spine dtype {dtype}")
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows, columns)
        entry.payload_bytes = rows * columns * 2
        source = self.s
        self._track(name)

        def produce() -> Iterator[bytes]:
            yield to_bytes(source.matrix2d(name))

        self.plan.append(PlanItem(entry, produce))

    def add_rows_sharded_bf16(self, kind: int, layer: int, name: str,
                              rows: int, columns: int):
        dtype, shape = self.s.meta(name)
        if len(shape) == 3 and shape[1] == 1:
            shape = (shape[0], shape[2])
        if tuple(shape) != (rows, columns):
            raise PackFailure(f"{name}: shape {tuple(shape)}, expected {(rows, columns)}")
        if dtype not in ("BF16", "F32"):
            raise PackFailure(f"{name}: spine dtype {dtype}")
        start, count = self._slice_rows(rows)
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, count, columns)
        entry.payload_bytes = count * columns * 2
        source, off = self.s, start
        self._track(name)

        def produce() -> Iterator[bytes]:
            yield to_bytes(source.matrix2d(name)[off:off + count, :])

        self.plan.append(PlanItem(entry, produce))

    def add_cols_sharded_bf16(self, kind: int, layer: int, name: str,
                              rows: int, columns: int):
        dtype, shape = self.s.meta(name)
        if tuple(shape) != (rows, columns):
            raise PackFailure(f"{name}: shape {tuple(shape)}, expected {(rows, columns)}")
        if dtype not in ("BF16", "F32"):
            raise PackFailure(f"{name}: spine dtype {dtype}")
        start, count = self._slice_rows(columns)
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows, count)
        entry.payload_bytes = rows * count * 2
        source, off = self.s, start
        self._track(name)

        def produce() -> Iterator[bytes]:
            yield to_bytes(source.matrix2d(name)[:, off:off + count])

        self.plan.append(PlanItem(entry, produce))

    def add_rows_sharded_f32(self, kind: int, layer: int, name: str, total: int):
        dtype, shape = self.s.meta(name)
        if dtype not in ("F32", "BF16") or (len(shape) != 1 or shape[0] != total):
            raise PackFailure(f"{name}: expected F32[{total}] (or exact BF16), "
                              f"got {dtype} {tuple(shape)}")
        start, count = self._slice_rows(total)
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE, 1, 1, count)
        entry.payload_bytes = count * 4
        source, off = self.s, start
        self._track(name)

        def produce() -> Iterator[bytes]:
            yield to_bytes(source.vector_f32(name)[off:off + count])

        self.plan.append(PlanItem(entry, produce))

    def add_replicated_f32(self, kind: int, layer: int, name: str, total: int):
        dtype, shape = self.s.meta(name)
        if dtype not in ("F32", "BF16") or (len(shape) != 1 and tuple(shape) != (1, total)) or \
                (len(shape) == 1 and shape[0] != total):
            raise PackFailure(f"{name}: expected F32[{total}] (or exact BF16), "
                              f"got {dtype} {tuple(shape)}")
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE, 1, 1, total)
        entry.payload_bytes = total * 4
        source = self.s
        self._track(name)

        def produce() -> Iterator[bytes]:
            yield to_bytes(source.vector_f32(name))

        self.plan.append(PlanItem(entry, produce))

    def add_fused_sections(self, kind: int, layer: int,
                           sections: List[Tuple[str, int]], out_columns: int):
        """Fuse whole tensors as row sections (q|k|v|beta), sharding each
        section by its own rows. A contiguous slice of the concatenation
        would cross section boundaries and hand every rank past the first
        rows of the WRONG projection."""
        total = sum(width for _, width in sections)
        slices = [self._slice_rows(width) for _, width in sections]
        rows_out = sum(count for _, count in slices)
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1,
                      rows_out, out_columns)
        entry.payload_bytes = rows_out * out_columns * 2
        source = self.s
        names = [name for name, _ in sections]
        self._track(*names)

        def produce() -> Iterator[bytes]:
            parts = []
            for name, (start, count) in zip(names, slices):
                matrix = source.matrix2d(name)
                if matrix.shape[0] != 0:
                    parts.append(matrix[start:start + count, :])
            fused = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            blob = to_bytes(fused)
            if len(blob) != entry.payload_bytes:
                raise PackFailure(f"fused {names}: {len(blob)} bytes, planned {entry.payload_bytes}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_up_gate_fused(self, kind: int, layer: int, up_name: str,
                          gate_name: str, rows_per_section: int, columns: int,
                          shard: bool):
        up_slice = self._slice_rows(rows_per_section) if shard else (0, rows_per_section)
        gate_slice = self._slice_rows(rows_per_section) if shard else (0, rows_per_section)
        rows_out = up_slice[1] + gate_slice[1]
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1,
                      rows_out, columns)
        entry.payload_bytes = rows_out * columns * 2
        source = self.s
        self._track(up_name, gate_name)

        def produce() -> Iterator[bytes]:
            up = source.matrix2d(up_name)[up_slice[0]:up_slice[0] + up_slice[1], :]
            gate = source.matrix2d(gate_name)[gate_slice[0]:gate_slice[0] + gate_slice[1], :]
            fused = np.concatenate((up, gate), axis=0)
            blob = to_bytes(fused)
            if len(blob) != entry.payload_bytes:
                raise PackFailure(f"{up_name}|{gate_name}: {len(blob)} bytes, planned {entry.payload_bytes}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_kv_b(self, layer: int, name: str):
        dtype, shape = self.s.meta(name)
        full_rows = MLA_HEADS * (NOPE + VDIM)
        if dtype != "BF16" or tuple(shape) != (full_rows, LATENT):
            raise PackFailure(f"{name}: expected BF16 {(full_rows, LATENT)}, got {dtype} {tuple(shape)}")
        key_entry = Entry(K_KV_B_KEY_T, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                          MLA_HEADS, LATENT, NOPE)
        value_entry = Entry(K_KV_B_VALUE, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                            MLA_HEADS, VDIM, LATENT)
        key_entry.payload_bytes = MLA_HEADS * LATENT * NOPE * 2
        value_entry.payload_bytes = MLA_HEADS * VDIM * LATENT * 2
        source = self.s
        self._track(name)

        def produce_key() -> Iterator[bytes]:
            matrix = source.matrix2d(name).reshape(MLA_HEADS, NOPE + VDIM, LATENT)
            key = np.ascontiguousarray(matrix[:, :NOPE, :].transpose(0, 2, 1))
            blob = to_bytes(key)
            if len(blob) != key_entry.payload_bytes:
                raise PackFailure(f"{name}: key transpose size")
            yield blob

        def produce_value() -> Iterator[bytes]:
            matrix = source.matrix2d(name).reshape(MLA_HEADS, NOPE + VDIM, LATENT)
            blob = to_bytes(np.ascontiguousarray(matrix[:, NOPE:, :]))
            if len(blob) != value_entry.payload_bytes:
                raise PackFailure(f"{name}: value size")
            yield blob

        self.plan.append(PlanItem(key_entry, produce_key))
        self.plan.append(PlanItem(value_entry, produce_value))

    def add_conv(self, kind: int, layer: int, name: str, channels: int):
        dtype, shape = self.s.meta(name)
        if len(shape) == 3 and shape[1] == 1:
            shape = (shape[0], shape[2])
        if dtype not in ("BF16", "F32") or len(shape) != 2 or \
                tuple(shape) != (channels, KDA_CONV):
            raise PackFailure(f"{name}: conv layout {dtype} {tuple(shape)}, expected "
                              f"[{channels}, {KDA_CONV}] (or [channels, 1, kernel])")
        start, count = self._slice_rows(channels)
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1,
                      count, KDA_CONV)
        entry.payload_bytes = count * KDA_CONV * 2
        source, off = self.s, start
        self._track(name)

        def produce() -> Iterator[bytes]:
            yield to_bytes(source.matrix2d(name)[off:off + count, :])

        self.plan.append(PlanItem(entry, produce))

    def add_experts(self, layer: int, prefix: str):
        if self.expert_codec != CODEC_BF16:
            raise PackFailure("only the bf16 expert pass-through arm is wired; "
                              "compressed sources need their own receipt set")
        inter0, inter_count = self._slice_rows(EXPERT_INTER)
        w1_rows = 2 * inter_count
        w1 = Entry(K_EXPERT_UP_GATE, layer, PAYLOAD_PACKED_WEIGHT, CODEC_BF16,
                   SCALE_NONE, EXPERTS, w1_rows, HIDDEN)
        w1.payload_bytes = EXPERTS * w1_rows * HIDDEN * 2
        w2 = Entry(K_EXPERT_DOWN, layer, PAYLOAD_PACKED_WEIGHT, CODEC_BF16,
                   SCALE_NONE, EXPERTS, HIDDEN, inter_count)
        w2.payload_bytes = EXPERTS * HIDDEN * inter_count * 2
        source = self.s

        for expert in range(EXPERTS):
            self._track(f"{prefix}.{expert}.up_proj.weight",
                        f"{prefix}.{expert}.gate_proj.weight",
                        f"{prefix}.{expert}.down_proj.weight")

        def expert_ok(name: str, rows: int, columns: int) -> np.ndarray:
            dtype, shape = source.meta(name)
            if dtype != "BF16" or tuple(shape) != (rows, columns):
                raise PackFailure(f"{name}: expected BF16 {(rows, columns)}, "
                                  f"got {dtype} {tuple(shape)}")
            return source.matrix2d(name)

        def produce_w1() -> Iterator[bytes]:
            for expert in range(EXPERTS):
                up = expert_ok(f"{prefix}.{expert}.up_proj.weight",
                               EXPERT_INTER, HIDDEN)
                gate = expert_ok(f"{prefix}.{expert}.gate_proj.weight",
                                 EXPERT_INTER, HIDDEN)
                fused = np.concatenate(
                    (up[inter0:inter0 + inter_count, :],
                     gate[inter0:inter0 + inter_count, :]), axis=0)
                yield to_bytes(fused)

        def produce_w2() -> Iterator[bytes]:
            for expert in range(EXPERTS):
                down = expert_ok(f"{prefix}.{expert}.down_proj.weight",
                                 HIDDEN, EXPERT_INTER)
                yield to_bytes(down[:, inter0:inter0 + inter_count])

        self.plan.append(PlanItem(w1, produce_w1))
        self.plan.append(PlanItem(w2, produce_w2))

    def build(self, source_revision: str) -> None:
        s = self.s
        if s.config.get("quantization_config"):
            raise PackFailure("source carries quantization_config; the bf16 arm "
                              "requires the official BF16 release")
        torch_dtype = s.config.get("torch_dtype")
        if torch_dtype is not None and str(torch_dtype) not in ("bfloat16", "torch.bfloat16"):
            raise PackFailure(f"torch_dtype {torch_dtype} contradicts the bf16 arm")
        p = "model.layers"
        for layer in range(LAYERS):
            a = f"{p}.{layer}.attention."
            m = f"{p}.{layer}.mlp."
            is_mla = (layer + 1) % 6 == 0
            is_dense = layer < FIRST_ROUTED
            self.add_replicated_bf16(K_ATTN_NORM, layer, f"{p}.{layer}.input_layernorm.weight", 1, HIDDEN)
            self.add_replicated_bf16(K_POST_ATTN_NORM, layer, f"{p}.{layer}.post_attention_layernorm.weight", 1, HIDDEN)
            if is_mla:
                self.add_rows_sharded_bf16(K_Q, layer, f"{a}q_proj.weight", Q_ROWS, HIDDEN)
                self.add_replicated_bf16(K_KV_A, layer, f"{a}kv_a_proj_with_mqa.weight", KV_ROW, HIDDEN)
                self.add_replicated_bf16(K_KV_A_NORM, layer, f"{a}kv_a_layernorm.weight", 1, LATENT)
                self.add_kv_b(layer, f"{a}kv_b_proj.weight")
                self.add_rows_sharded_bf16(K_ATTN_GATE, layer, f"{a}g_proj.weight", MLA_HEADS, HIDDEN)
                self.add_cols_sharded_bf16(K_ATTN_OUTPUT, layer, f"{a}dense.weight", HIDDEN, MLA_HEADS * VDIM)
            else:
                self.add_fused_sections(K_KDA_QKV_BETA, layer, [
                    (f"{a}q_proj.weight", KDA_QK),
                    (f"{a}k_proj.weight", KDA_QK),
                    (f"{a}v_proj.weight", KDA_QK),
                    (f"{a}b_proj.weight", KDA_HEADS),
                ], HIDDEN)
                self.add_conv(K_KDA_Q_CONV, layer, f"{a}q_conv1d.weight", KDA_QK)
                self.add_conv(K_KDA_K_CONV, layer, f"{a}k_conv1d.weight", KDA_QK)
                self.add_conv(K_KDA_V_CONV, layer, f"{a}v_conv1d.weight", KDA_V)
                self.add_rows_sharded_bf16(K_KDA_DECAY_PROJ, layer, f"{a}f_proj.weight", KDA_QK, HIDDEN)
                self.add_rows_sharded_bf16(K_KDA_GATE_PROJ, layer, f"{a}g_proj.weight", KDA_V, HIDDEN)
                self.add_rows_sharded_f32(K_KDA_DECAY_BIAS, layer, f"{a}dt_bias", KDA_QK)
                self.add_rows_sharded_f32(K_KDA_HEAD_LOG_SCALE, layer, f"{a}A_log", KDA_HEADS)
                self.add_replicated_f32(K_KDA_OUT_NORM, layer, f"{a}o_norm.weight", KDA_KEY)
                self.add_cols_sharded_bf16(K_KDA_OUT, layer, f"{a}o_proj.weight", HIDDEN, KDA_V)
            if is_dense:
                self.add_up_gate_fused(K_DENSE_GATE_UP, layer,
                                       f"{m}up_proj.weight", f"{m}gate_proj.weight",
                                       DENSE_INTER, HIDDEN, shard=True)
                self.add_cols_sharded_bf16(K_DENSE_DOWN, layer, f"{m}down_proj.weight",
                                           HIDDEN, DENSE_INTER)
            else:
                self.add_replicated_bf16(K_ROUTER, layer, f"{m}gate.weight", EXPERTS, HIDDEN)
                self.add_replicated_f32(K_ROUTER_CORRECTION, layer,
                                        f"{m}gate.expert_bias", EXPERTS)
                self.add_experts(layer, f"{m}experts")
                self.add_up_gate_fused(K_SHARED_GATE_UP, layer,
                                       f"{m}shared_experts.up_proj.weight",
                                       f"{m}shared_experts.gate_proj.weight",
                                       EXPERT_INTER, HIDDEN, shard=True)
                self.add_cols_sharded_bf16(K_SHARED_DOWN, layer,
                                           f"{m}shared_experts.down_proj.weight",
                                           HIDDEN, EXPERT_INTER)
        self.add_rows_sharded_bf16(K_EMBEDDING, GLOBAL_LAYER,
                                   "model.word_embeddings.weight", VOCAB, HIDDEN)
        self.add_replicated_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight", 1, HIDDEN)
        self.add_rows_sharded_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", VOCAB, HIDDEN)
        self.source_revision = source_revision


def serialize_entry(entry: Entry) -> bytes:
    return struct.pack(
        "<IIIIIII IQQQQ",
        entry.kind, entry.layer, entry.payload_type, entry.weight_codec,
        entry.scale_encoding, entry.group_count, entry.rows, entry.columns,
        entry.payload_offset, entry.payload_bytes,
        entry.scale_offset, entry.scale_bytes,
    )


def assemble_header(plan: List[PlanItem], tp_degree: int, tp_rank: int,
                    expert_codec: int, directory_offset: int, file_bytes: int,
                    revision: str, contract_sha: bytes) -> bytes:
    fields = [
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, CODEC_ABI_VERSION,
        0, len(plan), 1, 0, 0, LAYERS, LAYERS,
        HIDDEN, VOCAB, EXPERTS,
        CODEC_BF16, expert_codec, CODEC_BF16,
        tp_degree, tp_rank,
    ]
    fixed = struct.pack("<20I", *fields)
    tail = struct.pack("<QQ", directory_offset, file_bytes)
    revision_bytes = revision.encode()[:64].ljust(65, b"\0")
    header = fixed + tail + revision_bytes + contract_sha + bytes(32) + bytes(32)
    if len(header) % 8:
        header += b"\0" * (8 - len(header) % 8)
    if len(header) != HEADER_BYTES:
        raise PackFailure(f"header assembled {len(header)} bytes, expected {HEADER_BYTES}")
    return header


def emit_region(out, offset: int, expected: int, chunks: Iterator[bytes]) -> None:
    out.seek(offset)
    written = 0
    for chunk in chunks:
        if len(chunk) > expected - written:
            raise PackFailure(f"region at {offset}: producer exceeds {expected} bytes")
        out.write(chunk)
        written += len(chunk)
    if written != expected:
        raise PackFailure(f"region at {offset}: producer wrote {written}, expected {expected}")


def emit(packer: Packer, path: Path, revision: str, contract_sha: bytes) -> int:
    if path.exists():
        raise PackFailure(f"output already exists; the two-pass proof requires a "
                          f"fresh artifact path: {path}")
    directory_offset = (HEADER_BYTES + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
    cursor = directory_offset + len(packer.plan) * ENTRY_BYTES
    for item in packer.plan:
        e = item.entry
        e.payload_offset = (cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
        cursor = e.payload_offset + e.payload_bytes
    file_bytes = cursor
    header = assemble_header(packer.plan, packer.tp_degree, packer.tp_rank,
                             packer.expert_codec, directory_offset, file_bytes,
                             revision, contract_sha)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial",
                                     dir=path.parent)
    os.close(fd)
    try:
        with Path(temporary).open("wb") as out:
            out.write(header)
            out.seek(directory_offset)
            for item in packer.plan:
                out.write(serialize_entry(item.entry))
            for item in packer.plan:
                emit_region(out, item.entry.payload_offset, item.entry.payload_bytes,
                            item.produce_payload())
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    return file_bytes


def census_check(weight_map: Dict[str, str], packed: List[str],
                 mtp_names: List[str]) -> Dict[str, Any]:
    packed_set = set(packed)
    known = set(weight_map)
    unexplained = sorted(known - packed_set - set(mtp_names))
    missing = sorted(packed_set - known)
    duplicated = len(packed) - len(packed_set)
    if unexplained or missing or duplicated:
        raise PackFailure(
            f"census does not close: {len(unexplained)} unexplained "
            f"(e.g. {unexplained[:5]}), {len(missing)} missing-from-index "
            f"(e.g. {missing[:5]}), {duplicated} double-packed")
    return {
        "checkpoint_tensors": len(known),
        "packed": len(packed_set),
        "omitted_mtp": len(mtp_names),
    }


def write_expert_manifest(pack_path: Path, plan: List[PlanItem]) -> int:
    """Emit <pack>.experts: the weightd lazy-expert manifest v2.

    16-byte header (magic 0x58504557, version 2, range count, reserved 0)
    then one 48-byte record per expert plane: layer, expert, kind
    (tensor kind * 2 + plane, payload plane only - the bf16 arm has no
    scale planes), reserved 0, offset, bytes, ck128 digest. Ranges are
    emitted in offset order; SparkWeightdManifestLoad re-sorts and
    re-checks overlap anyway."""
    ranges = []
    for item in plan:
        entry = item.entry
        if entry.kind not in (K_EXPERT_UP_GATE, K_EXPERT_DOWN):
            continue
        per = entry.payload_bytes // EXPERTS
        if per == 0 or per > EXPERT_RANGE_BYTES_MAX:
            raise PackFailure(f"{entry.kind}: per-expert slab {per} bytes "
                              f"outside the weightd manifest contract")
        for expert in range(EXPERTS):
            ranges.append((entry.payload_offset + expert * per, per,
                           entry.layer_index, expert, entry.kind * 2))
    ranges.sort()
    with pack_path.open("rb") as pack:
        records = bytearray()
        for offset, length, layer, expert, kind in ranges:
            pack.seek(offset)
            digest = ck128(pack.read(length))
            records += struct.pack("<IIIIQQ", layer, expert, kind, 0,
                                   offset, length) + digest
    sidecar = pack_path.parent / (pack_path.name + ".experts")
    with sidecar.open("wb") as out:
        out.write(struct.pack("<IIII", EXPERT_MANIFEST_MAGIC,
                              EXPERT_MANIFEST_VERSION, len(ranges), 0))
        out.write(records)
    return len(ranges)


def mtp_tensor_names(weight_map: Dict[str, str]) -> List[str]:
    return sorted(name for name in weight_map if name.startswith("model.layers.42."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkpoint directory (warm)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-all", type=int, default=0,
                        help="emit all N rank packs in one process")
    parser.add_argument("--expert-codec", default="bf16", choices=["bf16"])
    parser.add_argument("--dry-plan", action="store_true")
    args = parser.parse_args()

    name_map = json.loads(NAME_MAP.read_text())
    source_revision = name_map["source_revision"]
    expected_census = name_map["checkpoint_census"]["tensor_count"]
    contract_sha = sha256_bytes(CONTRACT.read_bytes()) if CONTRACT.is_file() else bytes(32)
    codec_ids = {"bf16": CODEC_BF16}
    expert_codec = codec_ids[args.expert_codec]

    source = SourceReader(Path(args.source))
    if len(source.weight_map) != expected_census:
        raise PackFailure(f"index census {len(source.weight_map)} != pinned "
                          f"{expected_census}; re-pin name_map against this revision")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts = out_dir / "receipts"
    receipts.mkdir(exist_ok=True)
    ranks = range(args.tp_all) if args.tp_all else [args.tp_rank]
    mtp_names = mtp_tensor_names(source.weight_map)
    for rank in ranks:
        packer = Packer(source, args.tp_all or args.tp_degree, rank, expert_codec)
        packer.build(source_revision)
        census = census_check(source.weight_map, packer.packed_names, mtp_names)
        if args.dry_plan:
            print(f"rank {rank}: {len(packer.plan)} pack tensors planned, "
                  f"census {census}")
            continue
        path = out_dir / f"ling_stage.tp{args.tp_all or args.tp_degree}.rank{rank}.lspk"
        file_bytes = emit(packer, path, source_revision, contract_sha)
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        spine_bytes = sum(item.entry.payload_bytes for item in packer.plan
                          if item.entry.kind not in (K_EXPERT_UP_GATE,
                                                     K_EXPERT_DOWN))
        expert_bytes = sum(item.entry.payload_bytes for item in packer.plan
                           if item.entry.kind in (K_EXPERT_UP_GATE,
                                                  K_EXPERT_DOWN))
        manifest_ranges = write_expert_manifest(path, packer.plan)
        receipt = {
            "pack": path.name,
            "sha256": digest.hexdigest(),
            "file_bytes": file_bytes,
            "tensors": len(packer.plan),
            "tp_degree": args.tp_all or args.tp_degree,
            "tp_rank": rank,
            "source": str(args.source),
            "source_revision": source_revision,
            "expert_codec": args.expert_codec,
            "census": census,
            "resident_bytes": {
                "spine": spine_bytes,
                "expert_payload": expert_bytes,
                "note": "spine is always-resident; expert_payload is the "
                        "weightd lazy tier's working set per node",
            },
            "expert_manifest_ranges": manifest_ranges,
        }
        (receipts / f"rank{rank}.json").write_text(json.dumps(receipt, indent=1) + "\n")
        print(f"{path.name}: {len(packer.plan)} tensors, {file_bytes} bytes, "
              f"sha256 {receipt['sha256'][:16]}... (tp{receipt['tp_degree']} "
              f"rank {rank}, census {census['checkpoint_tensors']} = "
              f"{census['packed']} packed + {census['omitted_mtp']} mtp, "
              f"spine {spine_bytes} + expert {expert_bytes} bytes, "
              f"{manifest_ranges} manifest ranges)")
    source.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as failure:
        print(f"FAIL {failure}", file=sys.stderr)
        sys.exit(1)
