#!/usr/bin/env python3
"""Build GLM-5.2 resident-decode stage packs (v3 .glm52sp).

Wire format: modules/glm52_resident_decode_stage/source/spark_glm52_stagepack_format.h
Layout: header (264B) | directory (N * 64B) | 256B-aligned payloads | scales.

Precision policy (fp8 spine dequant, verbatim fp8 experts):
  - The only local source is the blockwise-FP8 checkpoint
    zai-org/GLM-5.2-FP8 (cold RAID6, pinned byte-for-byte by
    model_contracts/glm52_authoritative.json). In this checkpoint every
    linear weight (attention, indexer wq_b/wk, dense MLP, shared experts,
    routed experts) is F8_E4M3 with an F32 weight_scale_inv tile
    [rows/128, cols/128]; embeddings, lm_head, all norms, the router and
    the indexer weights_proj/k_norm stay BF16 (correction bias F32).
  - Spine linears are dequantized at pack time:
        bf16(w_f32[i,j]) = bf16(f32(code[i,j]) * scale_inv[i//128, j//128])
    and stored as CODEC_BF16 payloads, so the serving kernel path is
    unchanged. BF16 checkpoint tensors pass through byte-exact.
  - Routed experts: fp8 payload bytes copied as-is (codec 5) with F32
    scales per 128-column block; scale_inv is the dequant multiplier and
    each per-tile value is expanded across its block's 128 rows.
  - Expert up/gate order: [up_proj rows, gate_proj rows] stacked.

Runtime dependencies: python3 + numpy ONLY (no torch, no safetensors) so
the packer runs on any spark node. Payloads are located by safetensors
header parsing (same header-only scheme as spark_pack_common) and read
through np.memmap; spine tensors are dequantized once into a small LRU
cache so the lockstep all-rank writer decodes each tensor a single time.

Streaming: offsets are computed arithmetically from the plan; tensors are
produced lazily per entry and written in one blob per entry, so the full
~8x100 GB rank-pack set is built in bounded memory.

The serving path never opens the checkpoints; this is setup-time code.

History: 95c3e3b built packs from the FP8 checkpoint with code*scale_inv
spine dequant; 6485013 switched the spine to a BF16 master
(zai-org/GLM-5.2 @ b4734de4) that later left local storage; this version
restores the fp8-dequant policy against the pinned FP8 source alone.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Make the sibling shared packer core importable however this tool is loaded.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from spark_pack_common import (  # noqa: E402
    PackFailure,
    align_up,
    sha256_bytes,
    sha256_file,
    tp_shard_range,
    write_receipt,
)

MAGIC = 0x32534C47
FORMAT_VERSION = 3
HEADER_BYTES = 264
ENTRY_BYTES = 64
ALIGNMENT = 256
MODEL_REVISION_BYTES = 65
SHA256_BYTES = 32
CODEC_ABI_VERSION = 1

GLOBAL_LAYER = 0xFFFFFFFF

PAYLOAD_BF16 = 1
PAYLOAD_F32 = 2
PAYLOAD_U32 = 3
PAYLOAD_PACKED_WEIGHT = 4

CODEC_BF16 = 1
CODEC_NONE = 0
CODEC_FP8 = 5
SCALE_NONE = 0
SCALE_F32 = 1

# Tensor kinds (must match SparkGlm52StagePackTensorKind)
K_EMBEDDING = 0
K_FINAL_NORM = 1
K_LM_HEAD = 2
K_ATTN_NORM = 3
K_Q_A = 4
K_Q_A_NORM = 5
K_Q_B = 6
K_KV_A = 7
K_KV_A_NORM = 8
K_KV_B_KEY_T = 9
K_KV_B_VALUE = 10
K_ATTN_OUTPUT = 11
K_POST_ATTN_NORM = 12
K_INDEX_Q = 13
K_INDEX_K = 14
K_INDEX_HEAD = 15
K_INDEX_NORM_W = 16
K_INDEX_NORM_B = 17
K_DENSE_GATE_UP = 18
K_DENSE_DOWN = 19
K_ROUTER = 20
K_ROUTER_CORRECTION = 21
K_EXPERT_UP_GATE = 22
K_EXPERT_DOWN = 23
K_SHARED_GATE_UP = 24
K_SHARED_DOWN = 25

EXPERT_COUNT = 256
# Model snapshot identity carried by the firmware description, the module
# build (MODEL_REVISION) and the deployment config. The exact pack-byte
# provenance (fp8-dequant spine policy, source directory, contract) lives in
# pack_recipe_sha256 / source_config_sha256 / the per-rank receipt.
SPINE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"


def load_contract(repo_root: Path) -> Dict[str, Any]:
    return json.loads((repo_root / "model_contracts" / "glm52.json").read_text())


# -- numeric primitives (numpy; documented in tests/test_glm52_pack_fp8_source.py)

def fp8_e4m3_lut() -> np.ndarray:
    """256-entry float32 table: F8_E4M3 (e4m3fn) code -> value.

    sign(1) | exp(4, bias 7) | mantissa(3); exp=0 denormals are m*2^-9;
    exp=15 mantissa=7 is NaN (no infinities in e4m3fn).
    """
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
    value = np.where(sign == 1, -value, value)
    return value


def f32_to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round-to-nearest-even."""
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
               >> np.uint32(16)).astype(np.uint16)
    source_nan = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & \
                 ((bits & np.uint32(0x007FFFFF)) != 0)
    if source_nan.any():
        quiet = ((bits >> np.uint32(16)) & np.uint32(0xFF80)).astype(np.uint16) | np.uint16(0x0040)
        rounded = np.where(source_nan, quiet, rounded)
    return rounded


def bf16_u16_to_f32(bf16: np.ndarray) -> np.ndarray:
    """bf16 bit patterns (uint16) -> exact float32."""
    return (bf16.astype(np.uint32) << np.uint32(16)).view(np.float32)


# -- source access -----------------------------------------------------------

class Fp8SourceReader:
    """Header-only safetensors reader over one blockwise-FP8 checkpoint.

    Payloads are addressed by header offsets and read through np.memmap
    slices (page-cache friendly for the lockstep all-rank pass). Spine
    tensors decode to full-width bf16 exactly once and live in a small LRU.
    """

    def __init__(self, model_dir: Path, cache_byte_cap: int = 6 * 1024 ** 3):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise PackFailure(f"missing safetensors index: {index_path}")
        self.model_dir = model_dir
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.index_sha256 = sha256_file(index_path)
        self.config_sha256 = sha256_file(model_dir / "config.json")
        self.config = json.loads((model_dir / "config.json").read_text())
        self._mmaps: Dict[str, np.ndarray] = {}
        self._headers: Dict[str, dict] = {}
        self._data_start: Dict[str, int] = {}
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_bytes = 0
        self._cache_byte_cap = cache_byte_cap
        self._lut = fp8_e4m3_lut()

    def _header(self, shard: str) -> Tuple[dict, int]:
        if shard not in self._headers:
            path = self.model_dir / shard
            if not path.is_file():
                raise PackFailure(f"missing shard: {path}")
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
        """(checkpoint dtype string, shape, shard) from header only."""
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        header, _ = self._header(shard)
        entry = header.get(name)
        if entry is None:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        return entry["dtype"], tuple(entry["shape"]), shard

    def raw(self, name: str) -> np.ndarray:
        """This tensor's payload bytes as a uint8 view (zero-copy memmap slice)."""
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        header, data_start = self._header(shard)
        entry = header.get(name)
        if entry is None:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        begin, end = entry["data_offsets"]
        view = self._mmap(shard)[data_start + begin:data_start + end]
        if view.shape[0] != end - begin:
            raise PackFailure(f"short payload for {name}")
        return view

    def scale_name(self, name: str) -> str:
        return name + "_scale_inv"

    def spine_bf16(self, name: str) -> np.ndarray:
        """Full-width spine tensor as a uint16 bf16 matrix [rows, cols].

        BF16 checkpoint tensors pass through byte-exact; F8_E4M3 tensors
        dequantize through their weight_scale_inv F32 tile (128x128 blocks).
        Cached so the lockstep writer decodes each tensor once.
        """
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        dtype, shape, _shard = self.meta(name)
        if len(shape) == 1:
            shape = (1, shape[0])
        rows, cols = shape
        if dtype == "BF16":
            matrix = self.raw(name).view(np.uint16).reshape(rows, cols)
        elif dtype == "F8_E4M3":
            scale_name = self.scale_name(name)
            scale_dtype, scale_shape, _ = self.meta(scale_name)
            if scale_dtype != "F32":
                raise PackFailure(f"{scale_name}: dtype {scale_dtype}, expected F32")
            # partial quantization blocks are padded: the tile is ceil-shaped
            # (e.g. kv_a_proj_with_mqa is [576, 6144] with a [5, 48] tile)
            expected_shape = ((rows + 127) // 128, (cols + 127) // 128)
            if scale_shape != expected_shape:
                raise PackFailure(
                    f"{scale_name}: shape {scale_shape}, expected {expected_shape}")
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
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes
        return matrix

    def expert_payload(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        """fp8 payload bytes for [r0:r1, c0:c1], copied verbatim."""
        dtype, shape, _shard = self.meta(name)
        if dtype != "F8_E4M3" or len(shape) != 2:
            raise PackFailure(f"{name}: expected 2-D F8_E4M3, got {dtype} {shape}")
        codes = self.raw(name).reshape(shape[0], shape[1])
        return np.ascontiguousarray(codes[r0:r1, c0:c1]).tobytes()

    def expert_scale(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        """F32 scale bytes for the sliced region: scale_inv rows expanded
        across each 128-row block, columns sliced per 128-column block."""
        scale_name = self.scale_name(name)
        _dtype, scale_shape, _shard = self.meta(scale_name)
        scale = self.raw(scale_name).view(np.float32).reshape(scale_shape)
        expanded = np.repeat(scale, 128, axis=0)          # [rows, cols//128]
        return np.ascontiguousarray(
            expanded[r0:r1, c0 // 128:c1 // 128]).tobytes()

    def close(self) -> None:
        self._mmaps.clear()
        self._cache.clear()
        self._cache_bytes = 0


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
    """One directory entry plus lazy producers of its payload and scale bytes."""

    def __init__(self, entry: Entry,
                 produce_payload: Optional[Callable[[], Iterator[bytes]]],
                 produce_scale: Optional[Callable[[], Iterator[bytes]]] = None,
                 payload_bytes: int = 0, scale_bytes: int = 0,
                 sources: Optional[List[str]] = None):
        self.entry = entry
        self.produce_payload = produce_payload
        self.produce_scale = produce_scale
        self.entry.payload_bytes = payload_bytes
        self.entry.scale_bytes = scale_bytes
        self.sources = sources or []


class Packer:
    def __init__(self, source: Fp8SourceReader, contract: Dict[str, Any],
                 layer_range: Tuple[int, int], include_embedding: bool = True,
                 include_head: bool = True, tp_degree: int = 1, tp_rank: int = 0):
        self.source = source
        self.c = contract
        self.layer_range = layer_range
        self.include_embedding = include_embedding
        self.include_head = include_head
        self.tp_degree = tp_degree
        self.tp_rank = tp_rank
        self.plan: List[PlanItem] = []

    # -- plan construction -------------------------------------------------

    def add_spine_bf16(self, kind: int, layer: int, name: str, shard: str = ""):
        """Lazily-materialized spine tensor: plan records shape metadata only;
        the payload is produced at write time from the reader's bf16 matrix
        (dequantized once, shared across ranks by the lockstep writer)."""
        dtype, shape, _ = self.source.meta(name)
        if len(shape) == 2:
            rows, cols = shape
        else:
            rows, cols = 1, shape[0]
        if shard == "rows" and self.tp_degree > 1:
            start, count = tp_shard_range(rows, self.tp_degree, self.tp_rank)
            rows = count
        elif shard == "cols" and self.tp_degree > 1:
            start, count = tp_shard_range(cols, self.tp_degree, self.tp_rank)
            cols = count
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows, cols)
        payload_bytes = rows * cols * 2
        source = self.source
        tp_degree, tp_rank = self.tp_degree, self.tp_rank

        def produce() -> Iterator[bytes]:
            matrix = source.spine_bf16(name)      # uint16 [full_rows, full_cols]
            if shard == "rows" and tp_degree > 1:
                s, n = tp_shard_range(matrix.shape[0], tp_degree, tp_rank)
                blob = np.ascontiguousarray(matrix[s:s + n, :]).tobytes()
            elif shard == "cols" and tp_degree > 1:
                s, n = tp_shard_range(matrix.shape[1], tp_degree, tp_rank)
                blob = np.ascontiguousarray(matrix[:, s:s + n]).tobytes()
            else:
                blob = matrix.tobytes()
            if len(blob) != payload_bytes:
                raise PackFailure(
                    f"{name}: produced {len(blob)} payload bytes, planned {payload_bytes}")
            yield blob

        self.plan.append(PlanItem(entry, produce, None, payload_bytes, 0, [name]))

    def add_f32(self, kind: int, layer: int, name: str):
        dtype, shape, _ = self.source.meta(name)
        if dtype != "F32":
            raise PackFailure(f"{name}: dtype {dtype}, expected F32")
        rows, cols = (1, shape[0]) if len(shape) == 1 else shape
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE, 1, rows, cols)
        payload_bytes = rows * cols * 4
        source = self.source

        def produce() -> Iterator[bytes]:
            blob = to_bytes(source.raw(name).view(np.float32))
            if len(blob) != payload_bytes:
                raise PackFailure(f"{name}: produced {len(blob)} bytes, planned {payload_bytes}")
            yield blob

        self.plan.append(PlanItem(entry, produce, None, payload_bytes, 0, [name]))

    def add_experts(self, kind: int, layer: int, projections: List[str],
                    rows: int, columns: int, shard: str = ""):
        """Stack 256 experts' fp8 payloads (up then gate) + F32 scales, streamed.

        With shard='rows'/'cols' the per-expert tensor is sliced to this TP
        rank's range before writing; the entry dims reflect the shard.
        """
        r0, r1 = 0, rows
        c0, c1 = 0, columns
        if shard == "rows" and self.tp_degree > 1:
            if rows % self.tp_degree != 0:
                raise PackFailure(f"expert rows {rows} not divisible by tp {self.tp_degree}")
            count = rows // self.tp_degree
            r0 = self.tp_rank * count
            r1 = r0 + count
        elif shard == "cols" and self.tp_degree > 1:
            if columns % self.tp_degree != 0:
                raise PackFailure(f"expert cols {columns} not divisible by tp {self.tp_degree}")
            count = columns // self.tp_degree
            c0 = self.tp_rank * count
            c1 = c0 + count
        shard_rows = r1 - r0
        shard_cols = c1 - c0
        per_expert_payload = len(projections) * shard_rows * shard_cols   # fp8 bytes
        per_expert_scale = len(projections) * shard_rows * (shard_cols // 128) * 4
        payload_bytes = EXPERT_COUNT * per_expert_payload
        scale_bytes = EXPERT_COUNT * per_expert_scale
        entry = Entry(kind, layer, PAYLOAD_PACKED_WEIGHT, CODEC_FP8, SCALE_F32,
                      EXPERT_COUNT, len(projections) * shard_rows, shard_cols)
        source = self.source

        names = [f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                 for expert in range(EXPERT_COUNT) for proj in projections]

        def produce_payload() -> Iterator[bytes]:
            for name in names:
                _dtype, shape, _s = source.meta(name)
                if tuple(shape) != (rows, columns):
                    raise PackFailure(f"{name} shape {tuple(shape)} != ({rows}, {columns})")
                yield source.expert_payload(name, r0, r1, c0, c1)

        def produce_scale() -> Iterator[bytes]:
            for name in names:
                yield source.expert_scale(name, r0, r1, c0, c1)

        self.plan.append(PlanItem(entry, produce_payload, produce_scale,
                                  payload_bytes, scale_bytes, names))

    def add_kv_b(self, layer: int):
        name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        heads = self.c["head_count"]
        qk_nope = self.c["qk_nope_head_dimension"]
        value_dim = self.c["value_head_dimension"]
        latent = self.c["latent_dimension"]
        expected_rows = heads * (qk_nope + value_dim)
        dtype, shape, _ = self.source.meta(name)
        if tuple(shape) != (expected_rows, latent):
            raise PackFailure(f"kv_b shape {tuple(shape)} != ({expected_rows}, {latent})")
        self._add_kv_b_part(K_KV_B_KEY_T, layer, name, heads, qk_nope, value_dim,
                            latent, key=True)
        self._add_kv_b_part(K_KV_B_VALUE, layer, name, heads, qk_nope, value_dim,
                            latent, key=False)

    def _add_kv_b_part(self, kind: int, layer: int, name: str, heads: int,
                       qk_nope: int, value_dim: int, latent: int, key: bool):
        rows = heads * (latent if key else value_dim)
        cols = qk_nope if key else latent
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                      heads, rows // heads, cols)
        payload_bytes = rows * cols * 2
        source = self.source

        def produce() -> Iterator[bytes]:
            w = source.spine_bf16(name)               # uint16 [heads*(qk_nope+value), latent]
            per_head = qk_nope + value_dim
            parts = []
            for head in range(heads):
                block = w[head * per_head:(head + 1) * per_head, :]
                if key:
                    parts.append(np.ascontiguousarray(block[:qk_nope, :].T))   # [latent, qk_nope]
                else:
                    parts.append(np.ascontiguousarray(block[qk_nope:, :]))     # [value, latent]
            stacked = np.stack(parts, axis=0)
            blob = to_bytes(stacked.reshape(rows, cols))
            if len(blob) != payload_bytes:
                raise PackFailure(f"{name}: produced {len(blob)} bytes, planned {payload_bytes}")
            yield blob

        self.plan.append(PlanItem(entry, produce, None, payload_bytes, 0, [name]))

    def add_dense_gate_up(self, layer: int):
        self._add_fused_gate_up(K_DENSE_GATE_UP, layer,
                                f"model.layers.{layer}.mlp.gate_proj.weight",
                                f"model.layers.{layer}.mlp.up_proj.weight")

    def add_shared_gate_up(self, layer: int):
        self._add_fused_gate_up(K_SHARED_GATE_UP, layer,
                                f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight",
                                f"model.layers.{layer}.mlp.shared_experts.up_proj.weight")

    def _add_fused_gate_up(self, kind: int, layer: int, gate_name: str, up_name: str):
        _gdtype, gate_shape, _ = self.source.meta(gate_name)
        _udtype, up_shape, _ = self.source.meta(up_name)
        if tuple(gate_shape) != tuple(up_shape):
            raise PackFailure(f"{up_name} vs {gate_name} shape drift")
        rows, cols = up_shape
        start = count = None
        if self.tp_degree > 1:
            start, count = tp_shard_range(rows, self.tp_degree, self.tp_rank)
        entry_rows = count if count is not None else rows
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1,
                      2 * entry_rows, cols)
        payload_bytes = 2 * entry_rows * cols * 2
        source = self.source
        tp_degree, tp_rank = self.tp_degree, self.tp_rank

        def produce() -> Iterator[bytes]:
            def shard(name: str) -> bytes:
                matrix = source.spine_bf16(name)
                if tp_degree <= 1:
                    return matrix.tobytes()
                s, n = tp_shard_range(matrix.shape[0], tp_degree, tp_rank)
                return np.ascontiguousarray(matrix[s:s + n, :]).tobytes()
            gate = shard(gate_name)
            up = shard(up_name)
            if len(up) != len(gate):
                raise PackFailure(f"{up_name} vs {gate_name} byte drift after shard")
            yield up + gate

        self.plan.append(PlanItem(entry, produce, None, payload_bytes, 0,
                                  [gate_name, up_name]))

    def has_full_indexer(self, layer: int) -> bool:
        share = self.c["dsa_index_share_group_layer_count"]
        return layer < 3 or (layer >= 6 and (layer - 6) % share == 0)

    # -- main plan ---------------------------------------------------------

    def build_plan(self):
        c = self.c
        first, last = self.layer_range
        first_routed = c["first_routed_layer"]

        if self.include_embedding:
            self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER, "model.embed_tokens.weight", shard="rows")
        if self.include_head:
            self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight")
            self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", shard="rows")

        for layer in range(first, last + 1):
            attn = f"model.layers.{layer}.self_attn"
            self.add_spine_bf16(K_ATTN_NORM, layer,
                                f"model.layers.{layer}.input_layernorm.weight")
            self.add_spine_bf16(K_Q_A, layer, f"{attn}.q_a_proj.weight")
            self.add_spine_bf16(K_Q_A_NORM, layer, f"{attn}.q_a_layernorm.weight")
            self.add_spine_bf16(K_Q_B, layer, f"{attn}.q_b_proj.weight", shard="rows")
            self.add_spine_bf16(K_KV_A, layer, f"{attn}.kv_a_proj_with_mqa.weight")
            self.add_spine_bf16(K_KV_A_NORM, layer, f"{attn}.kv_a_layernorm.weight")
            self.add_kv_b(layer)
            self.add_spine_bf16(K_ATTN_OUTPUT, layer, f"{attn}.o_proj.weight", shard="cols")
            self.add_spine_bf16(K_POST_ATTN_NORM, layer,
                                f"model.layers.{layer}.post_attention_layernorm.weight")
            if self.has_full_indexer(layer):
                self.add_spine_bf16(K_INDEX_Q, layer, f"{attn}.indexer.wq_b.weight")
                self.add_spine_bf16(K_INDEX_K, layer, f"{attn}.indexer.wk.weight")
                self.add_spine_bf16(K_INDEX_HEAD, layer, f"{attn}.indexer.weights_proj.weight")
                self.add_spine_bf16(K_INDEX_NORM_W, layer, f"{attn}.indexer.k_norm.weight")
                self.add_spine_bf16(K_INDEX_NORM_B, layer, f"{attn}.indexer.k_norm.bias")
            if layer < first_routed:
                self.add_dense_gate_up(layer)
                self.add_spine_bf16(K_DENSE_DOWN, layer,
                                    f"model.layers.{layer}.mlp.down_proj.weight", shard="cols")
            else:
                self.add_spine_bf16(K_ROUTER, layer,
                                    f"model.layers.{layer}.mlp.gate.weight")
                self.add_f32(K_ROUTER_CORRECTION, layer,
                             f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
                self.add_experts(K_EXPERT_UP_GATE, layer, ["up_proj", "gate_proj"],
                                 c["moe_intermediate_dimension"], c["hidden_dimension"],
                                 shard="rows")
                self.add_experts(K_EXPERT_DOWN, layer, ["down_proj"],
                                 c["hidden_dimension"], c["moe_intermediate_dimension"],
                                 shard="cols")
                self.add_shared_gate_up(layer)
                self.add_spine_bf16(K_SHARED_DOWN, layer,
                                    f"model.layers.{layer}.mlp.shared_experts.down_proj.weight",
                                    shard="cols")

    # -- writing -----------------------------------------------------------

    def layout(self):
        """Assign payload/scale offsets; returns (entries, file_bytes)."""
        entries = [item.entry for item in self.plan]
        tensor_count = len(entries)
        directory_offset = align_up(HEADER_BYTES, ALIGNMENT)
        directory_bytes = tensor_count * ENTRY_BYTES
        offset = align_up(directory_offset + directory_bytes, ALIGNMENT)
        for item in self.plan:
            item.entry.payload_offset = offset
            offset += item.entry.payload_bytes
        for item in self.plan:
            if item.entry.scale_bytes:
                offset = align_up(offset, ALIGNMENT)
                item.entry.scale_offset = offset
                offset += item.entry.scale_bytes
        return entries, offset

    @staticmethod
    def write_header(f, packer, entries, file_bytes, model_revision, contract_sha256,
                     source_config_sha256, pack_recipe_sha256, stage_count, stage_index,
                     first_layer_index, layer_count, total_layer_count):
        tensor_count = len(entries)
        directory_offset = align_up(HEADER_BYTES, ALIGNMENT)
        header = struct.pack(
            "<20I2Q65s32s32s32s",
            MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES,
            CODEC_ABI_VERSION, 0, tensor_count, stage_count, stage_index,
            first_layer_index, layer_count, total_layer_count,
            packer.c["hidden_dimension"], packer.c["output_vocab_count"],
            packer.c["moe_expert_count"], CODEC_BF16, CODEC_FP8, CODEC_BF16,
            packer.tp_degree, packer.tp_rank, directory_offset, file_bytes,
            model_revision.encode("utf-8")[:MODEL_REVISION_BYTES - 1].ljust(
                MODEL_REVISION_BYTES - 1, b"\0") + b"\0",
            contract_sha256, source_config_sha256, pack_recipe_sha256)
        f.write(header)
        f.seek(directory_offset)
        for entry in entries:
            f.write(struct.pack(
                "<8I4Q",
                entry.kind, entry.layer, entry.payload_type, entry.weight_codec,
                entry.scale_encoding, entry.group_count, entry.rows, entry.columns,
                entry.payload_offset, entry.payload_bytes,
                entry.scale_offset, entry.scale_bytes))


KIND_NAMES = {
    K_EMBEDDING: "EMBEDDING", K_FINAL_NORM: "FINAL_NORM", K_LM_HEAD: "LM_HEAD",
    K_ATTN_NORM: "ATTN_NORM", K_Q_A: "Q_A", K_Q_A_NORM: "Q_A_NORM", K_Q_B: "Q_B",
    K_KV_A: "KV_A", K_KV_A_NORM: "KV_A_NORM", K_KV_B_KEY_T: "KV_B_KEY_T",
    K_KV_B_VALUE: "KV_B_VALUE", K_ATTN_OUTPUT: "ATTN_OUTPUT",
    K_POST_ATTN_NORM: "POST_ATTN_NORM", K_INDEX_Q: "INDEX_Q", K_INDEX_K: "INDEX_K",
    K_INDEX_HEAD: "INDEX_HEAD", K_INDEX_NORM_W: "INDEX_NORM_W",
    K_INDEX_NORM_B: "INDEX_NORM_B", K_DENSE_GATE_UP: "DENSE_GATE_UP",
    K_DENSE_DOWN: "DENSE_DOWN", K_ROUTER: "ROUTER",
    K_ROUTER_CORRECTION: "ROUTER_CORRECTION", K_EXPERT_UP_GATE: "EXPERT_UP_GATE",
    K_EXPERT_DOWN: "EXPERT_DOWN", K_SHARED_GATE_UP: "SHARED_GATE_UP",
    K_SHARED_DOWN: "SHARED_DOWN",
}


def build_receipt(packer: "Packer", output: Path, file_bytes: int,
                  model_revision: str, recipe: Dict[str, Any],
                  source_config: Dict[str, Any], first: int, last: int) -> Path:
    source = packer.source
    receipt = {
        "tool": "glm52_resident_stagepack.py",
        "wire_format": {"magic": MAGIC, "version": FORMAT_VERSION,
                        "header_bytes": HEADER_BYTES, "entry_bytes": ENTRY_BYTES},
        "model_revision": model_revision,
        "recipe": recipe,
        "source": {
            "directory": str(source.model_dir),
            "config_sha256": source.config_sha256,
            "index_sha256": source.index_sha256,
            "architecture": source.config.get("architectures", [None])[0],
            "digest_contract": "model_contracts/glm52_authoritative.json",
            "spine_policy": ("fp8-dequant: bf16(f32(code)*scale_inv) on 128x128 "
                             "blocks; bf16 tensors byte-exact"),
        },
        "tp_degree": packer.tp_degree,
        "tp_rank": packer.tp_rank,
        "layer_range": [first, last],
        "file_bytes": file_bytes,
        "tensor_count": len(packer.plan),
        "tensors": [
            {
                "kind": KIND_NAMES.get(item.entry.kind, item.entry.kind),
                "layer": item.entry.layer if item.entry.layer != GLOBAL_LAYER else "global",
                "rows": item.entry.rows,
                "columns": item.entry.columns,
                "payload_bytes": item.entry.payload_bytes,
                "scale_bytes": item.entry.scale_bytes,
                "sources": item.sources,
            }
            for item in packer.plan
        ],
    }
    return write_receipt(receipt, output)


def write_all_ranks(packers, output_paths, model_revision, contract_sha256,
                    source_config_sha256, pack_recipe_sha256, stage_count, stage_index,
                    first_layer_index, layer_count, total_layer_count) -> List[int]:
    """Lockstep fan-out writer: every packer walks its plan in the same
    tensor order, so driving entry i of all ranks back-to-back serves the
    shared source reads (and the single spine dequant) from the reader LRU -
    one cold pass over the source for the whole TP set instead of one per
    rank."""
    prepared = []
    for packer, path in zip(packers, output_paths):
        entries, file_bytes = packer.layout()
        f = path.open("wb")
        Packer.write_header(f, packer, entries, file_bytes, model_revision,
                            contract_sha256, source_config_sha256, pack_recipe_sha256,
                            stage_count, stage_index, first_layer_index, layer_count,
                            total_layer_count)
        prepared.append((packer, f, file_bytes))
    try:
        for index in range(len(packers[0].plan)):
            for packer, f, _ in prepared:
                item = packer.plan[index]
                f.seek(item.entry.payload_offset)
                if item.produce_payload is not None:
                    for chunk in item.produce_payload():
                        f.write(chunk)
                if item.produce_scale is not None:
                    f.seek(item.entry.scale_offset)
                    for chunk in item.produce_scale():
                        f.write(chunk)
    finally:
        for _, f, _ in prepared:
            f.close()
    return [file_bytes for _, _, file_bytes in prepared]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GLM-5.2 resident stage packs")
    parser.add_argument("--source", required=True,
                        help="blockwise-FP8 checkpoint (spine + routed experts)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-range", default="0-77",
                        help="inclusive layer range, e.g. 0-77 or 0-2 (ignored with --stage)")
    parser.add_argument("--stage", type=int, default=-1,
                        help="PP13 stage 0-12: layers 6*stage..6*stage+5, globals gated")
    parser.add_argument("--model-revision", default=SPINE_REVISION)
    parser.add_argument("--stage-count", type=int, default=1)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=1,
                        help="tensor parallel degree (shards tensors across ranks)")
    parser.add_argument("--tp-rank", type=int, default=0,
                        help="this rank's TP slice index (0..tp_degree-1)")
    parser.add_argument("--all-ranks", action="store_true",
                        help="emit one pack per TP rank in a single lockstep "
                             "pass over the shared source; --output is a "
                             "template containing '{rank}'")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    contract = load_contract(repo_root)
    if args.tp_degree < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_degree:
        raise PackFailure(f"invalid tp rank {args.tp_rank}/{args.tp_degree}")
    stage_count = args.stage_count
    stage_index = args.stage_index
    include_embedding = True
    include_head = True
    if args.stage >= 0:
        if args.stage >= 13:
            raise PackFailure(f"--stage must be 0-12, got {args.stage}")
        first = args.stage * 6
        last = first + 5
        stage_count = 13
        stage_index = args.stage
        include_embedding = args.stage == 0
        include_head = args.stage == 12
    else:
        first, last = (int(v) for v in args.layer_range.split("-"))
        if last < first or last >= contract["layer_count"]:
            raise PackFailure(f"bad layer range {args.layer_range}")

    source = Fp8SourceReader(Path(args.source))
    contract_bytes = (repo_root / "examples" / "model_descriptions" /
                      "glm52_resident_decode_stage_fp8_firmware.json").read_bytes()
    recipe = json.dumps({"tool": "glm52_resident_stagepack.py",
                         "codec": "fp8",
                         "spine": "fp8-dequant-128x128-blockwise",
                         "expert_source_contract":
                             "model_contracts/glm52_authoritative.json"},
                        sort_keys=True)
    single_source_config = json.dumps(
        {"layer_range": f"{first}-{last}", "stage": args.stage,
         "tp_degree": args.tp_degree, "tp_rank": args.tp_rank,
         "source_dir": args.source}, sort_keys=True)
    all_ranks_source_config = json.dumps(
        {"layer_range": f"{first}-{last}", "stage": args.stage,
         "tp_degree": args.tp_degree, "tp_rank": "all",
         "source_dir": args.source}, sort_keys=True)
    selected_source_config = (all_ranks_source_config if args.all_ranks
                              else single_source_config)

    if args.all_ranks:
        packers = []
        for rank in range(args.tp_degree):
            packer = Packer(source, contract, (first, last),
                            include_embedding, include_head, args.tp_degree, rank)
            packer.build_plan()
            packers.append(packer)
        output_paths = [Path(args.output.format(rank=rank))
                        for rank in range(args.tp_degree)]
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        file_sizes = write_all_ranks(
            packers, output_paths, args.model_revision,
            sha256_bytes(contract_bytes), sha256_bytes(selected_source_config.encode()),
            sha256_bytes(recipe.encode()), stage_count, stage_index,
            first, last - first + 1, contract["layer_count"])
        source.close()
        for packer, path, size in zip(packers, output_paths, file_sizes):
            build_receipt(packer, path, size, args.model_revision,
                          json.loads(recipe), json.loads(selected_source_config),
                          first, last)
            print(f"glm52sp written: {path} bytes={size} "
                  f"stage={stage_index}/{stage_count} layers={first}-{last} "
                  f"tensors={len(packer.plan)} tp={args.tp_degree}")
        return 0
    packer = Packer(source, contract, (first, last),
                    include_embedding, include_head, args.tp_degree, args.tp_rank)
    packer.build_plan()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _entries, file_bytes = packer.layout()
    with output_path.open("wb") as f:
        Packer.write_header(f, packer, _entries, file_bytes, args.model_revision,
                            sha256_bytes(contract_bytes),
                            sha256_bytes(single_source_config.encode()),
                            sha256_bytes(recipe.encode()), stage_count, stage_index,
                            first, last - first + 1, contract["layer_count"])
        for item in packer.plan:
            f.seek(item.entry.payload_offset)
            if item.produce_payload is not None:
                for chunk in item.produce_payload():
                    f.write(chunk)
            if item.produce_scale is not None:
                f.seek(item.entry.scale_offset)
                for chunk in item.produce_scale():
                    f.write(chunk)
    build_receipt(packer, output_path, file_bytes, args.model_revision,
                  json.loads(recipe), json.loads(single_source_config), first, last)
    source.close()
    print(f"glm52sp written: {args.output} bytes={file_bytes} "
          f"stage={stage_index}/{stage_count} layers={first}-{last} "
          f"tensors={len(packer.plan)} receipt={output_path}.receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
