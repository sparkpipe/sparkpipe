#!/usr/bin/env python3
"""Build laguna (poolside/Laguna-S-2.1) resident-decode stage packs (.lgsp).

Real-checkpoint packer: reads the warm laguna-s-2.1 checkpoint (bf16
safetensors shards) through header-only safetensors memmaps and emits one
.lgsp per (stage, rank). The census lock fails closed on any checkpoint
tensor outside the 23 locked patterns or any count mismatch (36096 experts
+ 47 router + 47 correction bias + 141 shared + 432 attn/norm + 3 dense +
3 global = 36769); every checkpoint->pack transform follows
model-families/laguna/name_map.json:

  - fused q|k|v with per-section whole-head row slices, o cols by head
    groups.
  - gate-first W1 for dense/shared/routed (silu gate_first=true); expert
    W1/W2 intermediate-sliced per the donor grouped-GEMM.
  - router correction bias stamped f32.

The pack header carries --revision (the warm snapshot's HF tree id) and
--contract-sha256 (sha256 of model_contracts/laguna_authoritative.json);
both are required so a pack is never emitted without its identity.

Run on a spark node with warm ceph (per the fleet notes: NOT sparke - its
client holds a stale negative cache after the metadata incident).

Usage (per rank; or --tp-all for the 16-rank set in one process):
  python3 tools/laguna_stagepack.py \
      --source /mnt/model-warm/laguna-s-2.1 --output-dir build/stagepacks \
      --revision <hf-tree-id> --contract-sha256 <contract-sha> \
      --stage-count 2 --stage-index 0 --tp-degree 8 --tp-rank 0 \
      --owns-embedding
"""
from __future__ import annotations

import argparse
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

CODEC_BF16 = 1
CODEC_NONE = 0
CODEC_FP8 = 5
CODEC_NVFP4 = 6
SCALE_NONE = 0
SCALE_F32 = 1
SCALE_UE4M3_F32_GLOBAL = 4

# Tensor kinds (mirror the format header enum).
K_EMBEDDING, K_FINAL_NORM, K_LM_HEAD = 0, 1, 2
K_ATTN_INPUT_NORM, K_ATTN_POST_NORM = 3, 4
K_FUSED_QKV, K_ATTN_OUTPUT, K_ATTN_GATE = 5, 6, 7
K_Q_NORM, K_K_NORM = 8, 9
K_DENSE_GATE_UP, K_DENSE_DOWN = 10, 11
K_ROUTER, K_ROUTER_CORRECTION = 12, 13
K_EXPERT_GATE_UP, K_EXPERT_DOWN = 14, 15
K_SHARED_GATE_UP, K_SHARED_DOWN = 16, 17

# Geometry (contract-pinned).
HIDDEN = 3072
LAYERS = 48
HEAD_DIM = 128
HEADS_FULL, HEADS_SLIDING, KV_HEADS = 48, 72, 8
EXPERTS, TOP_K, EXPERT_INTER = 256, 10, 1024
DENSE_INTER = 12288
VOCAB = 100352
FIRST_ROUTED = 1
CENSUS_PATTERNS = 23
CENSUS_TENSORS = 36769
CONTRACT_PATH = ROOT_STR = "model_contracts/laguna_authoritative.json"

MAGIC = 0x334C4147  # matches SPARK_LAGUNA_STAGEPACK_MAGIC ("GAL3" LE)


def load_census(repo_root: Path):
    patterns = json.loads(
        (repo_root / "model-families/laguna/tensor_patterns.json").read_text())
    if patterns["tensor_pattern_count"] != CENSUS_PATTERNS or patterns["tensor_count"] != CENSUS_TENSORS:
        raise PackFailure(
            f"tensor_patterns census {patterns['tensor_pattern_count']}/"
            f"{patterns['tensor_count']} != locked 23/36769")
    return patterns


def census_lock(source: "SourceReader"):
    """Fail closed on any checkpoint tensor outside the 23 locked patterns
    and on any count mismatch."""
    import re
    census = load_census(Path(__file__).resolve().parents[1])

    def pattern_regex(pattern: str) -> "re.Pattern":
        parts = []
        for token in re.split(r"(\{layer\}|\{expert\})", pattern):
            if token == "{layer}" or token == "{expert}":
                parts.append(r"(\d+)")
            else:
                parts.append(re.escape(token))
        return re.compile("^" + "".join(parts) + "$")

    compiled = {pattern: pattern_regex(pattern) for pattern in census["patterns"]}
    counts = {pattern: 0 for pattern in census["patterns"]}
    for name in source.weight_map:
        for pattern, regex in compiled.items():
            if regex.match(name):
                counts[pattern] += 1
                break
        else:
            raise PackFailure(f"unknown checkpoint tensor (census lock): {name}")
    for pattern, expected in census["patterns"].items():
        if counts[pattern] != expected["count"]:
            raise PackFailure(
                f"census lock: {pattern} count {counts[pattern]} != {expected['count']}")


def load_name_map(repo_root: Path) -> Dict[str, Any]:
    return json.loads((repo_root / "model-families" / "laguna" / "name_map.json").read_text())



def fp8_e4m3_lut() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint32)
    sign = (codes >> 7) & 1
    exp = (codes >> 3) & 0xF
    man = codes & 0x7
    value = np.where(
        exp == 0,
        man.astype(np.float32) * np.float32(2.0 ** -9),
        (np.float32(1.0) + man.astype(np.float32) / np.float32(8.0))
        * np.power(np.float32(2.0), (exp.astype(np.int32) - 7).astype(np.float32)),
    ).astype(np.float32)
    value[(exp == 15) & (man == 7)] = np.float32("nan")
    return np.where(sign == 1, -value, value).astype(np.float32)


def f32_to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
               >> np.uint32(16)).astype(np.uint16)
    return rounded


class SourceReader:
    """Header-only safetensors reader over the FP8 checkpoint (glm52's
    Fp8SourceReader pattern: memmap payloads, dequant-once LRU for the
    bf16 spine)."""

    def __init__(self, model_dir: Path, cache_byte_cap: int = 8 * 1024 ** 3):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise PackFailure(f"missing safetensors index: {index_path}")
        self.model_dir = model_dir
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.config = json.loads((model_dir / "config.json").read_text())
        self._mmaps: Dict[str, np.ndarray] = {}
        self._headers: Dict[str, dict] = {}
        self._data_start: Dict[str, int] = {}
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_bytes = 0
        self._cache_byte_cap = cache_byte_cap
        self._lut = fp8_e4m3_lut()

    def _header(self, shard: str) -> Tuple[dict, int]:
        if shard not in self._headers:
            path = self.model_dir / shard
            with path.open("rb") as file:
                header_bytes = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(header_bytes))
            self._headers[shard] = header
            self._data_start[shard] = 8 + header_bytes
        return self._headers[shard], self._data_start[shard]

    def _mmap(self, shard: str) -> np.ndarray:
        if shard not in self._mmaps:
            path = self.model_dir / shard
            size = path.stat().st_size
            self._mmaps[shard] = np.memmap(path, dtype=np.uint8, mode="r", shape=(size,))
        return self._mmaps[shard]

    def meta(self, name: str) -> Tuple[str, Tuple[int, ...], str]:
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        header, _ = self._header(shard)
        entry = header.get(name)
        if entry is None:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        return entry["dtype"], tuple(entry["shape"]), shard

    def raw(self, name: str) -> np.ndarray:
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        header, data_start = self._header(shard)
        entry = header.get(name)
        begin, end = entry["data_offsets"]
        view = self._mmap(shard)[data_start + begin:data_start + end]
        if view.shape[0] != end - begin:
            raise PackFailure(f"short payload for {name}")
        return view

    def spine_bf16(self, name: str) -> np.ndarray:
        """Full-width bf16 matrix [rows, cols]; F8_E4M3 dequantizes through
        its [128,128]-block weight_scale_inv (exact to bf16)."""
        if name in self._cache:
            return self._cache[name]
        dtype, shape, _ = self.meta(name)
        if len(shape) == 3 and shape[1] == 1:
            shape = (shape[0], shape[2])   # conv1d [dim, 1, kernel] -> [dim, kernel]
        if len(shape) == 1:
            shape = (1, shape[0])
        rows, cols = shape
        if dtype == "BF16":
            matrix = self.raw(name).view(np.uint16).reshape(rows, cols)
        elif dtype == "F32":
            # the pack format pins BF16 here (the kernels read bf16 convs);
            # the nvfp4 release stores convs F32 - round-to-nearest-even
            # downcast, bit-exact for any value that was bf16-originated
            u32 = self.raw(name).view(np.uint32).reshape(rows, cols)
            rounding = (u32 >> np.uint32(16)) & np.uint32(1)
            return ((u32 + np.uint32(0x7FFF) + rounding)
                    >> np.uint32(16)).astype(np.uint16)
        elif dtype == "F8_E4M3":
            scale_name = name + "_scale_inv"
            _dt, scale_shape, _ = self.meta(scale_name)
            expected = ((rows + 127) // 128, (cols + 127) // 128)
            if scale_shape != expected:
                raise PackFailure(f"{scale_name}: shape {scale_shape}, expected {expected}")
            codes = self.raw(name).reshape(rows, cols)
            scale = self.raw(scale_name).view(np.float32).reshape(scale_shape)
            expanded = np.repeat(scale, 128, axis=0)[:rows]
            expanded = np.repeat(expanded, 128, axis=1)[:, :cols]
            matrix = f32_to_bf16_u16(self._lut[codes] * expanded)
            del expanded
        else:
            raise PackFailure(f"{name}: unexpected spine dtype {dtype}")
        self._cache[name] = matrix
        self._cache_bytes += matrix.nbytes
        while self._cache_bytes > self._cache_byte_cap and len(self._cache) > 1:
            victim = next(iter(self._cache))
            self._cache_bytes -= self._cache[victim].nbytes
            del self._cache[victim]
        return matrix

    def spine_f32(self, name: str) -> np.ndarray:
        """Full-width f32 matrix: BF16 upcasts exactly; F8_E4M3 dequantizes."""
        dtype, shape, _ = self.meta(name)
        if len(shape) == 1:
            shape = (1, shape[0])
        rows, cols = shape
        if dtype == "F32":
            return self.raw(name).view(np.float32).reshape(rows, cols).copy()
        if dtype == "BF16":
            return (self.raw(name).view(np.uint16).reshape(rows, cols)
                    .astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float32)
        raise PackFailure(f"{name}: f32 upcast from {dtype} unsupported here")

    def expert_payload(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        dtype, shape, _ = self.meta(name)
        if dtype == "BF16":
            # bf16-official arm: expert weights are native BF16 — verbatim
            # passthrough (packers repackage, never quantize). No scale plane.
            codes = self.raw(name).view(np.uint16).reshape(shape[0], shape[1])
            return np.ascontiguousarray(codes[r0:r1, c0:c1]).tobytes()
        if dtype != "F8_E4M3":
            raise PackFailure(f"{name}: expected F8_E4M3, got {dtype}")
        codes = self.raw(name).reshape(shape[0], shape[1])
        return np.ascontiguousarray(codes[r0:r1, c0:c1]).tobytes()

    def expert_scale(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        dtype, _, _ = self.meta(name)
        if dtype == "BF16":
            return b""
        scale_name = name + "_scale_inv"
        _dt, scale_shape, _ = self.meta(scale_name)
        scale = self.raw(scale_name).view(np.float32).reshape(scale_shape)
        expanded = np.repeat(scale, 128, axis=0)
        return np.ascontiguousarray(expanded[r0:r1, c0 // 128:c1 // 128]).tobytes()

    # -- nvfp4 (community redhatai/modelopt release; VERBATIM passthrough) --

    def nvfp4_payload(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        """Packed e2m1 bytes for [r0:r1, c0:c1] ELEMENTS from <name>_packed.

        The source stores two 4-bit codes per uint8 ([rows, cols//2], even
        element in the low nibble); column slices must be nibble aligned,
        which the expert sharding guarantees (whole 128-col k-tiles)."""
        packed = name + "_packed"
        dtype, shape, _ = self.meta(packed)
        if dtype != "U8" or len(shape) != 2:
            raise PackFailure(f"{packed}: expected 2-D U8 (packed e2m1), got {dtype} {shape}")
        if c0 % 2 != 0 or c1 % 2 != 0:
            raise PackFailure(f"{packed}: nvfp4 column slice [{c0}:{c1}] not nibble aligned")
        codes = self.raw(packed).reshape(shape[0], shape[1])
        return np.ascontiguousarray(codes[r0:r1, c0 // 2:c1 // 2]).tobytes()

    def nvfp4_block_scale(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        """UE4M3 block-scale bytes from <name>_scale ([rows, cols//16])."""
        plane = name + "_scale"
        dtype, shape, _ = self.meta(plane)
        if dtype != "F8_E4M3" or len(shape) != 2:
            raise PackFailure(f"{plane}: expected 2-D F8_E4M3, got {dtype} {shape}")
        scales = self.raw(plane).reshape(shape[0], shape[1])
        return np.ascontiguousarray(scales[r0:r1, c0 // 16:c1 // 16]).tobytes()

    def nvfp4_weight_global(self, name: str) -> bytes:
        """The per-tensor F32 weight global scale from <name>_global_scale."""
        global_name = name + "_global_scale"
        dtype, shape, _ = self.meta(global_name)
        if dtype != "F32" or tuple(shape) not in ((), (1,)):
            raise PackFailure(f"{global_name}: expected scalar F32, got {dtype} {shape}")
        return to_bytes(self.raw(global_name).view(np.float32))

    def close(self) -> None:
        self._mmaps.clear()
        self._cache.clear()


def to_bytes(t: np.ndarray) -> bytes:
    return np.ascontiguousarray(t).tobytes()


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
    def __init__(self, entry: Entry, produce_payload: Callable[[], Iterator[bytes]],
                 produce_scale: Optional[Callable[[], Iterator[bytes]]] = None):
        self.entry = entry
        self.produce_payload = produce_payload
        self.produce_scale = produce_scale


class Packer:
    def __init__(self, source: SourceReader, tp_degree: int, tp_rank: int,
                 first_layer: int, layer_count: int,
                 owns_embedding: bool, owns_head: bool, expert_codec: int = CODEC_BF16):
        self.s = source
        self.tp_degree = tp_degree
        self.tp_rank = tp_rank
        self.first_layer = first_layer
        self.layer_count = layer_count
        self.owns_embedding = owns_embedding
        self.owns_head = owns_head
        self.expert_codec = expert_codec
        self.plan: List[PlanItem] = []

    # -- helpers -----------------------------------------------------------

    def _rows_slice(self, rows: int) -> Tuple[int, int]:
        start, count = tp_shard_range(rows, self.tp_degree, self.tp_rank)
        return start, count

    def _cols_slice(self, cols: int) -> Tuple[int, int]:
        start, count = tp_shard_range(cols, self.tp_degree, self.tp_rank)
        return start, count

    def add_spine_bf16(self, kind: int, layer: int, name: str, shard: str = ""):
        dtype, shape, _ = self.s.meta(name)
        rows, cols = shape if len(shape) == 2 else (1, shape[0])
        if dtype not in ("BF16", "F8_E4M3"):
            raise PackFailure(f"{name}: spine dtype {dtype}")
        s0 = s1 = 0
        if shard == "rows" and self.tp_degree > 1:
            s0, s1 = self._rows_slice(rows)
            rows = s1
        elif shard == "cols" and self.tp_degree > 1:
            s0, s1 = self._cols_slice(cols)
            cols = s1
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows, cols)
        expected = rows * cols * 2
        entry.payload_bytes = expected
        source, shard_axis, off, count = self.s, shard, s0, s1

        def produce() -> Iterator[bytes]:
            matrix = source.spine_bf16(name)
            if shard_axis == "rows" and self.tp_degree > 1:
                blob = to_bytes(matrix[off:off + count, :])
            elif shard_axis == "cols" and self.tp_degree > 1:
                blob = to_bytes(matrix[:, off:off + count])
            else:
                blob = to_bytes(matrix)
            if len(blob) != expected:
                raise PackFailure(f"{name}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_spine_f32(self, kind: int, layer: int, name: str):
        """f32 in the pack (the kernels read f32; the bf16 checkpoint
        upcasts at pack time)."""
        dtype, shape, _ = self.s.meta(name)
        rows, cols = shape if len(shape) == 2 else (1, shape[0])
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE, 1, rows, cols)
        expected = rows * cols * 4
        entry.payload_bytes = expected
        source = self.s

        def produce() -> Iterator[bytes]:
            blob = to_bytes(source.spine_f32(name))
            if len(blob) != expected:
                raise PackFailure(f"{name}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_f32_slice(self, kind: int, layer: int, name: str, axis: str = "cols"):
        """f32 vector sharded along its one axis (decay bias / A_log),
        or replicated with axis="none"."""
        dtype, shape, _ = self.s.meta(name)
        vector = self.s.spine_f32(name).reshape(-1)
        total = vector.shape[0]
        cols = total
        s0 = s1 = 0
        if self.tp_degree > 1 and axis != "none":
            s0, s1 = (self._cols_slice(total) if axis == "cols" else self._rows_slice(total))
            cols = s1
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE, 1, 1, cols)
        expected = cols * 4
        entry.payload_bytes = expected
        source, off, count, ax = self.s, s0, s1, axis

        def produce() -> Iterator[bytes]:
            v = source.spine_f32(name).reshape(-1)
            if self.tp_degree > 1 and ax != "none":
                v = v[off:off + count]
            blob = to_bytes(v)
            if len(blob) != expected:
                raise PackFailure(f"{name}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_fused_rows_sectioned(self, kind: int, layer: int, names, section_rows, shard=""):
        """Fuse checkpoint tensors' rows into one pack tensor; row sharding
        slices EVERY section by whole rows so section boundaries land on
        whole heads for every rank."""
        total = sum(section_rows)
        section_slices = [(0, width) for width in section_rows]
        if shard == "rows" and self.tp_degree > 1:
            section_slices = [self._rows_slice(width) for width in section_rows]
        rows_out = sum(count for _, count in section_slices)
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows_out, HIDDEN)
        expected = rows_out * HIDDEN * 2
        entry.payload_bytes = expected
        source = self.s

        def produce() -> Iterator[bytes]:
            parts = []
            for name, (off, count) in zip(names, section_slices):
                matrix = source.spine_bf16(name)
                parts.append(matrix[off:off + count, :] if self.tp_degree > 1 and shard == "rows" else matrix)
            blob = to_bytes(np.concatenate(parts, axis=0))
            if len(blob) != expected:
                raise PackFailure(f"fused {names}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_gate_up_fused(self, kind: int, layer: int, gate_name: str, up_name: str,
                          shard: str = ""):
        """GATE rows first, then up rows (LmSiluMulKernel gate_first=true)."""
        gate_rows, cols = self.s.meta(gate_name)[1]
        up_rows = self.s.meta(up_name)[1][0]
        gate_slice = (0, gate_rows)
        up_slice = (0, up_rows)
        rows_out = gate_rows + up_rows
        if shard == "rows" and self.tp_degree > 1:
            gate_slice = self._rows_slice(gate_rows)
            up_slice = self._rows_slice(up_rows)
            rows_out = gate_slice[1] + up_slice[1]
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows_out, cols)
        expected = rows_out * cols * 2
        entry.payload_bytes = expected
        source = self.s

        def produce() -> Iterator[bytes]:
            gate = source.spine_bf16(gate_name)
            up = source.spine_bf16(up_name)
            if self.tp_degree > 1 and shard == "rows":
                gate = gate[gate_slice[0]:gate_slice[0] + gate_slice[1], :]
                up = up[up_slice[0]:up_slice[0] + up_slice[1], :]
            blob = to_bytes(np.concatenate((gate, up), axis=0))
            if len(blob) != expected:
                raise PackFailure(f"{gate_name}|{up_name}: {len(blob)} vs {expected}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_experts(self, layer: int):
        prefix = f"model.layers.{layer}.mlp.experts"
        w1_r0, width = self._rows_slice(EXPERT_INTER)
        w1_r1 = w1_r0 + width
        w1_out_rows = 2 * width
        w2_out_cols = width
        w1 = Entry(K_EXPERT_GATE_UP, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                   EXPERTS, w1_out_rows, HIDDEN)
        w2 = Entry(K_EXPERT_DOWN, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                   EXPERTS, HIDDEN, w2_out_cols)
        w1.payload_bytes = EXPERTS * w1_out_rows * HIDDEN * 2
        w2.payload_bytes = EXPERTS * HIDDEN * w2_out_cols * 2
        source = self.s

        def expert_names_w1():
            for expert in range(EXPERTS):
                yield f"{prefix}.{expert}.gate_proj.weight", w1_r0, w1_r1
                yield f"{prefix}.{expert}.up_proj.weight", w1_r0, w1_r1

        def produce_w1() -> Iterator[bytes]:
            for name, lo, hi in expert_names_w1():
                dtype, shape, _ = source.meta(name)
                if dtype != "BF16":
                    raise PackFailure(f"{name}: expected native BF16, got {dtype}")
                codes = source.raw(name).view(np.uint16).reshape(shape[0], shape[1])
                yield np.ascontiguousarray(codes[lo:hi, :]).tobytes()

        def produce_w2() -> Iterator[bytes]:
            for expert in range(EXPERTS):
                name = f"{prefix}.{expert}.down_proj.weight"
                dtype, shape, _ = source.meta(name)
                if dtype != "BF16":
                    raise PackFailure(f"{name}: expected native BF16, got {dtype}")
                codes = source.raw(name).view(np.uint16).reshape(shape[0], shape[1])
                c0, cwidth = self._cols_slice(shape[1])
                yield np.ascontiguousarray(codes[:, c0:c0 + cwidth]).tobytes()

        self.plan.append(PlanItem(w1, produce_w1))
        self.plan.append(PlanItem(w2, produce_w2))

    def receipt(self) -> dict:
        return {
            "tp_degree": self.tp_degree,
            "tp_rank": self.tp_rank,
            "first_layer": self.first_layer,
            "layer_count": self.layer_count,
            "tensors": len(self.plan),
            "payload_bytes": sum(item.entry.payload_bytes for item in self.plan),
            "sha_feed": [f"{item.entry.kind}:{item.entry.layer}:"
                         f"{item.entry.payload_bytes}" for item in self.plan],
        }
    def build(self) -> None:
        p = "model.layers"
        for layer in range(self.first_layer, self.first_layer + self.layer_count):
            a = f"{p}.{layer}.self_attn."
            m = f"{p}.{layer}.mlp."
            heads = HEADS_SLIDING if layer % 4 != 0 else HEADS_FULL
            self.add_spine_bf16(K_ATTN_INPUT_NORM, layer, f"{p}.{layer}.input_layernorm.weight")
            self.add_spine_bf16(K_ATTN_POST_NORM, layer, f"{p}.{layer}.post_attention_layernorm.weight")
            kv_rows = KV_HEADS * HEAD_DIM
            self.add_fused_rows_sectioned(
                K_FUSED_QKV, layer,
                [f"{a}q_proj.weight", f"{a}k_proj.weight", f"{a}v_proj.weight"],
                [heads * HEAD_DIM, kv_rows, kv_rows], "rows")
            self.add_spine_bf16(K_ATTN_OUTPUT, layer, f"{a}o_proj.weight", shard="cols")
            self.add_spine_bf16(K_ATTN_GATE, layer, f"{a}g_proj.weight", shard="rows")
            self.add_spine_bf16(K_Q_NORM, layer, f"{a}q_norm.weight")
            self.add_spine_bf16(K_K_NORM, layer, f"{a}k_norm.weight")
            if layer < FIRST_ROUTED:
                self.add_gate_up_fused(K_DENSE_GATE_UP, layer,
                    f"{m}gate_proj.weight", f"{m}up_proj.weight", shard="rows")
                self.add_spine_bf16(K_DENSE_DOWN, layer, f"{m}down_proj.weight", shard="cols")
            else:
                self.add_spine_bf16(K_ROUTER, layer, f"{m}gate.weight")
                bias_name = f"{m}experts.e_score_correction_bias"
                dtype = self.s.meta(bias_name)[0]
                if dtype not in ("BF16", "F32"):
                    raise PackFailure(f"{bias_name}: dtype {dtype}, expected BF16 or F32")
                # the module reads the correction as f32 (router_correction_f32)
                # and the resolver models it f32-replicated: BF16 checkpoint
                # biases upcast exactly (bf16 -> f32 is lossless)
                self.add_f32_slice(K_ROUTER_CORRECTION, layer, bias_name, axis="none")
                self.add_experts(layer)
                self.add_gate_up_fused(K_SHARED_GATE_UP, layer,
                    f"{m}shared_expert.gate_proj.weight", f"{m}shared_expert.up_proj.weight",
                    shard="rows")
                self.add_spine_bf16(K_SHARED_DOWN, layer, f"{m}shared_expert.down_proj.weight", shard="cols")
        if self.owns_embedding:
            self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER,
                                "model.embed_tokens.weight", shard="rows")
        if self.owns_head:
            self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight")
            self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", shard="rows")

def assemble_header(packer: Packer, header_extra: Dict[str, Any], file_bytes: int,
                    revision: str, contract_sha256: str) -> bytes:
    """Serialize SparkLagunaStagePackHeader exactly (C layout)."""
    fields = [
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, CODEC_ABI_VERSION,
        header_extra.get("flags", 0),
        len(packer.plan),
        header_extra["stage_count"], header_extra["stage_index"],
        header_extra["first_layer"], header_extra["layer_count"], LAYERS,  # total = weight layers (module expects MODEL_LAYER_COUNT)
        HIDDEN, VOCAB, EXPERTS,
        CODEC_BF16, packer.expert_codec, CODEC_BF16,
        packer.tp_degree, packer.tp_rank,
    ]
    if len(fields) != 20:
        raise PackFailure(f"header field count {len(fields)}, expected 20")
    fixed = struct.pack("<20I", *fields)
    tail = struct.pack("<QQ", header_extra["directory_offset"], file_bytes)
    revision_bytes = revision.encode()[:64].ljust(65, b"\0")
    contract_bytes = bytes.fromhex(contract_sha256[:64].ljust(64, "0"))
    config_bytes = bytes(32)
    recipe_bytes = bytes(32)
    header = fixed + tail + revision_bytes + contract_bytes + config_bytes + recipe_bytes
    # the C struct aligns to 8 (largest member is u64): pad the tail
    if len(header) % 8:
        header += b"\0" * (8 - len(header) % 8)
    if len(header) != HEADER_BYTES:
        raise PackFailure(f"header assembled {len(header)} bytes, expected {HEADER_BYTES}")
    return header


def serialize_entry(entry: Entry) -> bytes:
    return struct.pack(
        "<IIIIIII IQQQQ",
        entry.kind, entry.layer, entry.payload_type, entry.weight_codec,
        entry.scale_encoding, entry.group_count, entry.rows, entry.columns,
        entry.payload_offset, entry.payload_bytes,
        entry.scale_offset, entry.scale_bytes,
    )


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


def _emit(packer: Packer, path: Path, header_extra: Dict[str, Any],
          revision: str, contract_sha256: str) -> int:
    packer.build()
    directory_offset = (HEADER_BYTES + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
    cursor = directory_offset + len(packer.plan) * ENTRY_BYTES
    for item in packer.plan:
        e = item.entry
        e.payload_offset = (cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
        cursor = e.payload_offset + e.payload_bytes
        if e.scale_bytes:
            e.scale_offset = (cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
            cursor = e.scale_offset + e.scale_bytes
    file_bytes = cursor
    header = assemble_header(packer, dict(header_extra, directory_offset=directory_offset),
                             file_bytes, revision, contract_sha256)
    with path.open("wb") as out:
        out.write(header)
        out.seek(directory_offset)
        for item in packer.plan:
            out.write(serialize_entry(item.entry))
        for item in packer.plan:
            emit_region(out, item.entry.payload_offset, item.entry.payload_bytes,
                        item.produce_payload())
            emit_region(out, item.entry.scale_offset, item.entry.scale_bytes,
                        item.produce_scale() if item.produce_scale else iter(()))
    return file_bytes


def emit(packer: Packer, path: Path, header_extra: Dict[str, Any],
         revision: str, contract_sha256: str) -> None:
    if path.exists():
        raise PackFailure(f"output already exists; choose a new artifact path: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        file_bytes = _emit(packer, Path(temporary), header_extra, revision, contract_sha256)
        with open(temporary, "rb") as source:
            os.fsync(source.fileno())
        # An exclusive link preserves an existing artifact even across a race.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    print(f"{path.name}: {len(packer.plan)} tensors, {file_bytes} bytes "
          f"(tp{packer.tp_degree} rank {packer.tp_rank}, layers "
          f"{header_extra['first_layer']}..{header_extra['first_layer'] + header_extra['layer_count'] - 1})")


def validate_stage(stage_count, stage_index, first_layer, layer_count,
                   owns_embedding, owns_head):
    if not 1 <= stage_count <= LAYERS or not 0 <= stage_index < stage_count:
        raise PackFailure("invalid pipeline stage count/index")
    if first_layer < 0 or layer_count <= 0 or first_layer + layer_count > LAYERS:
        raise PackFailure("pipeline layer span is outside the model")
    if owns_embedding and (stage_index != 0 or first_layer != 0):
        raise PackFailure("embedding ownership requires the first stage and layer")
    if owns_head and (stage_index + 1 != stage_count or first_layer + layer_count != LAYERS):
        raise PackFailure("head/MTP ownership requires the final stage and layer")


def stage_pack_name(tp_degree, tp_rank, stage_count, stage_index):
    pipeline = f".pp{stage_count}.stage{stage_index}" if stage_count != 1 else ""
    return f"laguna_stage.tp{tp_degree}{pipeline}.rank{tp_rank}.lgsp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkpoint directory (warm ceph)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=LAYERS)
    parser.add_argument("--stage-count", type=int, default=1)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--owns-embedding", action="store_true")
    parser.add_argument("--owns-head", action="store_true")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=8)
    parser.add_argument("--tp-all", type=int, default=0,
                        help="emit all N rank packs in one process (shared dequant cache)")
    parser.add_argument("--revision", required=True,
                        help="model revision stamped into the pack header (the warm snapshot's HF tree id)")
    parser.add_argument("--contract-sha256", required=True,
                        help="sha256 of model_contracts/laguna_authoritative.json, stamped into the pack header")
    parser.add_argument("--dry-plan", action="store_true",
                        help="plan and print the inventory without writing")
    args = parser.parse_args()
    validate_stage(args.stage_count, args.stage_index, args.first_layer,
                   args.layer_count, args.owns_embedding, args.owns_head)
    if args.stage_count not in (1, 2, 4) or args.tp_degree not in (4, 8):
        raise PackFailure("supported fleet shapes: tp8/pp2 (default), tp4/pp4, single-stage")

    source = SourceReader(Path(args.source))
    census_lock(source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks = range(args.tp_all) if args.tp_all else [args.tp_rank]
    for rank in ranks:
        packer = Packer(source, args.tp_all or args.tp_degree, rank,
                        args.first_layer, args.layer_count,
                        args.owns_embedding, args.owns_head)
        if args.dry_plan:
            packer.build()
            receipt = packer.receipt()
            print(f"rank {rank}: {receipt['tensors']} tensors, "
                  f"{receipt['payload_bytes']} payload bytes")
            continue
        emit(packer, out_dir / stage_pack_name(args.tp_all or args.tp_degree,
                                              rank, args.stage_count, args.stage_index),
             dict(stage_count=args.stage_count, stage_index=args.stage_index, first_layer=args.first_layer,
                  layer_count=args.layer_count,
                  flags=0),
             args.revision, args.contract_sha256)
        print("receipt " + json.dumps(packer.receipt(), sort_keys=True))
    source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
