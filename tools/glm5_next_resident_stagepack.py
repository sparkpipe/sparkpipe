#!/usr/bin/env python3
"""Build glm5_next (GLM 5.3 Flash) resident-decode stage packs (.g5nsp).

Real-checkpoint packer: reads /mnt/model-warm/glm-5.3-flash (FP8 e4m3
[128,128] checkpoint, 62 shards) through header-only safetensors memmaps
and emits one wire-format-v1 .g5nsp per TP rank. The tensor vocabulary is
the 49-kind table in modules/glm5_next_resident_decode_stage/source/
spark_glm5_next_stagepack_format.h, and every checkpoint->pack transform
follows model-families/glm5_next/name_map.json:

  - the pack-V2 fusions: kda_qkv_beta = q|k|v|beta rows (the checkpoint's
    separate q_proj/k_proj/v_proj/b_proj), kda_decay_gate_down = f_a|g_a.
  - the kv_b split+per-head transpose (glm52's add_kv_b pattern).
  - f32 upcasts where the kernels read f32: hc fn/base/scale, kda o_norm,
    the compressor ape, dt_bias (F32 already), A_log (F32 already).
  - FP8 MLA/dense/shared projections dequantize to bf16 at pack time
    (exact: e4m3 values are representable in bf16) - the module's spine
    path is bf16; routed experts stay packaged fp8 payload + f32 scales.

Run on a spark node with warm ceph (per the fleet notes: NOT sparke - its
client holds a stale negative cache after the metadata incident).

Usage (per rank; or --tp-all for the 16-rank set in one process, sharing
the dequant cache):
  python3 tools/glm5_next_resident_stagepack.py --spark <host-passed-at-runtime> \
      --source /mnt/model-warm/glm-5.3-flash --output-dir build/stagepacks \
      --tp-all 16 --expert-codec fp8
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_common import PackFailure, sha256_bytes, tp_shard_range  # noqa: E402

MAGIC = 0x33584C47  # matches SPARK_GLM5_NEXT_STAGEPACK_MAGIC ("3LXG" LE)
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
SCALE_NONE = 0
SCALE_F32 = 1

# Tensor kinds (mirror the format header enum; asserted at the bottom).
K_EMBEDDING, K_FINAL_NORM, K_LM_HEAD = 0, 1, 2
K_ATTN_NORM, K_Q_A, K_Q_A_NORM, K_Q_B = 3, 4, 5, 6
K_KV_A, K_KV_A_NORM, K_KV_B_KEY_T, K_KV_B_VALUE = 7, 8, 9, 10
K_ATTN_OUTPUT, K_POST_ATTN_NORM = 11, 12
K_INDEX_Q, K_INDEX_K, K_INDEX_HEAD, K_INDEX_NORM_W, K_INDEX_NORM_B = 13, 14, 15, 16, 17
K_DENSE_GATE_UP, K_DENSE_DOWN = 18, 19
K_ROUTER, K_ROUTER_CORRECTION = 20, 21
K_EXPERT_UP_GATE, K_EXPERT_DOWN = 22, 23
K_SHARED_GATE_UP, K_SHARED_DOWN = 24, 25
K_KDA_QKV_BETA, K_KDA_DECAY_GATE_DOWN, K_KDA_DECAY_UP, K_KDA_GATE_UP = 26, 27, 28, 29
K_KDA_Q_CONV, K_KDA_K_CONV, K_KDA_V_CONV = 30, 31, 32
K_KDA_DECAY_BIAS, K_KDA_HEAD_LOG_SCALE, K_KDA_OUT_NORM, K_KDA_OUT = 33, 34, 35, 36
K_HC_ATTN_FN, K_HC_ATTN_BASE, K_HC_ATTN_SCALE = 37, 38, 39
K_HC_FFN_FN, K_HC_FFN_BASE, K_HC_FFN_SCALE = 40, 41, 42
K_INDEX_COMPRESS_APE, K_INDEX_COMPRESS_GATE = 43, 44
K_MTP_EH_PROJ, K_MTP_ENORM, K_MTP_HNORM, K_MTP_SHARED_NORM = 45, 46, 47, 48

# Geometry (contract-pinned).
HIDDEN = 4096
LAYERS = 45
MTP_LAYER = 45
KDA_HEADS, KDA_DIM_PER_HEAD, KDA_LOW_RANK, KDA_CONV = 64, 128, 128, 4
KDA_DIM = KDA_HEADS * KDA_DIM_PER_HEAD
MLA_HEADS, Q_LORA, LATENT, NOPE, VDIM = 64, 1536, 512, 256, 256
IDX_HEADS, IDX_DIM, KPOOL = 32, 128, 4
HC, HC_MIX = 4, 24
EXPERTS, TOP_K, EXPERT_INTER = 288, 8, 2048
DENSE_INTER = 12288
VOCAB = 154880
FIRST_ROUTED = 3
REVISION = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
CONTRACT_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"


def load_name_map(repo_root: Path) -> Dict[str, Any]:
    return json.loads((repo_root / "model-families" / "glm5_next" / "name_map.json").read_text())


def is_kda(layer: int) -> bool:
    return layer % 4 != 3


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
                 first_layer: int, layer_count: int, include_mtp: bool,
                 owns_embedding: bool, owns_head: bool, expert_codec: int = CODEC_FP8):
        self.s = source
        self.tp_degree = tp_degree
        self.tp_rank = tp_rank
        self.first_layer = first_layer
        self.layer_count = layer_count
        self.include_mtp = include_mtp
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

    def add_fused_rows(self, kind: int, layer: int, names: List[str],
                       checkpoint_rows: List[int], shard: str = ""):
        """Fuse several checkpoint tensors' ROWS into one pack tensor
        (the pack-V2 convention). Row sharding slices every section by
        whole rows; the beta/bottleneck sections are narrow but the
        loader prices the fused rows as one tensor.

        THE SLICE IS PER SECTION. Slicing the concatenated tensor
        contiguously puts section boundaries at global row r*width_total/N,
        which for KDA q|k|v|beta hands every rank except rank-0-q sections
        of the WRONG projection (rank 0's "v" was q_proj rows 1024..1535):
        the per-head kernels index local head ids, so rank r's k/v/beta
        must be k/v/b_proj rows [r*w/tp, (r+1)*w/tp). At TP1 the two
        layouts coincide, which is how this passed the M3 gates."""
        total = sum(checkpoint_rows)
        dtype, _, _ = self.s.meta(names[0])
        if dtype not in ("BF16", "F8_E4M3"):
            raise PackFailure(f"{names[0]}: fused dtype {dtype}")
        s0 = s1 = 0
        rows_out = total
        section_slices: List[Tuple[int, int]] = []
        if shard == "rows" and self.tp_degree > 1:
            for width in checkpoint_rows:
                section_slices.append(self._rows_slice(width))
            rows_out = sum(count for _, count in section_slices)
        else:
            section_slices = [(0, width) for width in checkpoint_rows]
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows_out,
                      HIDDEN)
        expected = rows_out * HIDDEN * 2
        entry.payload_bytes = expected
        source = self.s

        def produce() -> Iterator[bytes]:
            matrices = [source.spine_bf16(n) for n in names]
            if self.tp_degree > 1 and shard == "rows":
                # section_slices carry (start, count) — the spine path's
                # convention. m[a:b] treated them as (start, end): rank 0
                # (start=0) was accidentally correct, every rank > 0 sliced
                # m[start:count] with count < start — EMPTY, the 0-byte
                # fused failure that killed the r1-r15 repack.
                parts = [m[a:a + b, :] for m, (a, b) in zip(matrices, section_slices)]
            else:
                parts = matrices
            fused = np.concatenate(parts, axis=0)
            blob = to_bytes(fused)
            if len(blob) != expected:
                raise PackFailure(f"fused {names}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_kda_conv(self, kind: int, layer: int, name: str):
        """[dim, 1, kernel] bf16 -> packed [dim, kernel], rows-sharded."""
        dtype, shape, _ = self.s.meta(name)
        if dtype != "BF16" or shape[1] != 1:
            raise PackFailure(f"{name}: conv shape {shape} dtype {dtype}")
        rows, kernel = shape[0], shape[2]
        s0 = s1 = 0
        if self.tp_degree > 1:
            s0, s1 = self._rows_slice(rows)
            rows = s1
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows, kernel)
        expected = rows * kernel * 2
        entry.payload_bytes = expected
        source, off, count = self.s, s0, s1

        def produce() -> Iterator[bytes]:
            matrix = source.spine_bf16(name).reshape(matrix_shape)
            if self.tp_degree > 1:
                blob = to_bytes(matrix[off:off + count, :])
            else:
                blob = to_bytes(matrix)
            if len(blob) != expected:
                raise PackFailure(f"{name}: {len(blob)} bytes, planned {expected}")
            yield blob

        matrix_shape = (shape[0], shape[2])  # squeezed
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

    def add_kv_b(self, layer: int):
        """kv_b_proj [heads*(nope+v), latent] -> key-transposed per head
        [heads, latent, nope] + value [heads, vdim, latent] (glm52's split;
        the value replicates, the key packs TRANSPOSED slices per head)."""
        name = f"model.language_model.layers.{layer}.self_attn.kv_b_proj.weight"
        dtype, shape, _ = self.s.meta(name)
        if dtype != "BF16":
            raise PackFailure(f"{name}: dtype {dtype}")
        full_rows = MLA_HEADS * (NOPE + VDIM)
        assert shape == (full_rows, LATENT), shape
        # key: [heads, latent, nope] transposed - REPLICATED (glm52 pattern:
        # the kernel indexes per-local-head, bind offsets per rank).
        key_entry = Entry(K_KV_B_KEY_T, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                          MLA_HEADS, LATENT, NOPE)
        value_entry = Entry(K_KV_B_VALUE, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                            MLA_HEADS, VDIM, LATENT)
        source = self.s
        key_expected = MLA_HEADS * LATENT * NOPE * 2
        value_expected = MLA_HEADS * VDIM * LATENT * 2
        key_entry.payload_bytes = key_expected
        value_entry.payload_bytes = value_expected

        def produce_key() -> Iterator[bytes]:
            matrix = source.spine_bf16(name).reshape(MLA_HEADS, NOPE + VDIM, LATENT)
            key = matrix[:, :NOPE, :]                      # [h, nope, latent]
            blob = to_bytes(np.ascontiguousarray(key.transpose(0, 2, 1)))  # [h, latent, nope]
            if len(blob) != key_expected:
                raise PackFailure("kv_b key transpose size")
            yield blob

        def produce_value() -> Iterator[bytes]:
            matrix = source.spine_bf16(name).reshape(MLA_HEADS, NOPE + VDIM, LATENT)
            value = matrix[:, NOPE:, :]                    # [h, vdim, latent]
            blob = to_bytes(value)
            if len(blob) != value_expected:
                raise PackFailure("kv_b value size")
            yield blob

        self.plan.append(PlanItem(key_entry, produce_key))
        self.plan.append(PlanItem(value_entry, produce_value))

    def add_up_gate_fused(self, kind: int, layer: int, up_name: str, gate_name: str,
                          shard: str = ""):
        """up rows then gate rows (glm52's stacked order) from two checkpoint
        tensors; FP8 sources dequantize to bf16."""
        up_rows, cols = self.s.meta(up_name)[1]
        gate_rows = self.s.meta(gate_name)[1][0]
        total = up_rows + gate_rows
        rows_out = total
        up_slice = (0, up_rows)
        gate_slice = (0, gate_rows)
        if shard == "rows" and self.tp_degree > 1:
            # per-section rows (see add_fused_rows): a contiguous slice of
            # [up | gate] crosses the section boundary and every rank past
            # the first reads the wrong tensor in each section.
            up_slice = self._rows_slice(up_rows)
            gate_slice = self._rows_slice(gate_rows)
            rows_out = up_slice[1] + gate_slice[1]
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows_out, cols)
        expected = rows_out * cols * 2
        entry.payload_bytes = expected
        source = self.s

        def produce() -> Iterator[bytes]:
            up = source.spine_bf16(up_name)
            gate = source.spine_bf16(gate_name)
            if self.tp_degree > 1 and shard == "rows":
                up = up[up_slice[0]:up_slice[0] + up_slice[1], :]
                gate = gate[gate_slice[0]:gate_slice[0] + gate_slice[1], :]
            fused = np.concatenate((up, gate), axis=0)
            blob = to_bytes(fused)
            if len(blob) != expected:
                raise PackFailure(f"{up_name}|{gate_name}: {len(blob)} vs {expected}")
            yield blob

        self.plan.append(PlanItem(entry, produce))

    def add_experts(self, layer: int):
        """288 experts' fp8 payloads (up-then-gate stacked, then down) +
        f32 scales, expert-major slabs, per-expert TP slicing exactly as
        glm52 shards: w1 rows-sharded, w2 cols-sharded (the grouped GEMM
        computes partial rows and the TP chain reduces)."""
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        tp = self.tp_degree
        rank = self.tp_rank
        w1_rows, w1_cols = 2 * EXPERT_INTER, HIDDEN      # stacked up|gate
        w2_rows, w2_cols = HIDDEN, EXPERT_INTER
        w1_r0 = w1_r1 = 0; w1_c0 = 0; w1_c1 = w1_cols
        w2_c0 = w2_c1 = 0; w2_r0 = 0; w2_r1 = w2_rows
        if tp > 1:
            if w1_rows % tp or w2_cols % tp:
                raise PackFailure(f"expert dims not divisible by tp{tp}")
            w1_r0 = (w1_rows // tp) * rank; w1_r1 = w1_r0 + w1_rows // tp
            w2_c0 = (w2_cols // tp) * rank; w2_c1 = w2_c0 + w2_cols // tp
        w1_out_rows = w1_r1 - w1_r0
        w2_out_cols = w2_c1 - w2_c0
        # source-driven expert codec: BF16 sources pass through verbatim
        # with no scale plane (the packer never quantizes either direction)
        probe_name = next(
            (n for n in self.s.weight_map
             if ".mlp.experts.0.up_proj.weight" in n), None)
        experts_bf16 = probe_name is not None and self.s.meta(probe_name)[0] == "BF16"
        codec = CODEC_BF16 if experts_bf16 else self.expert_codec
        w1 = Entry(K_EXPERT_UP_GATE, layer, PAYLOAD_PACKED_WEIGHT, codec,
                   SCALE_NONE if experts_bf16 else SCALE_F32, EXPERTS, w1_out_rows, w1_cols)
        w2 = Entry(K_EXPERT_DOWN, layer, PAYLOAD_PACKED_WEIGHT, codec,
                   SCALE_NONE if experts_bf16 else SCALE_F32, EXPERTS, w2_rows, w2_out_cols)
        w1.payload_bytes = EXPERTS * w1_out_rows * w1_cols * (2 if experts_bf16 else 1)
        w1.scale_bytes = 0 if experts_bf16 else EXPERTS * w1_out_rows * (w1_cols // 128) * 4
        w2.payload_bytes = EXPERTS * w2_rows * w2_out_cols * (2 if experts_bf16 else 1)
        w2.scale_bytes = 0 if experts_bf16 else EXPERTS * w2_rows * (w2_out_cols // 128) * 4
        source = self.s

        def produce_w1() -> Iterator[bytes]:
            for expert in range(EXPERTS):
                up = f"{prefix}.{expert}.up_proj.weight"
                gate = f"{prefix}.{expert}.gate_proj.weight"
                # stacked up|gate sliced to [w1_r0:w1_r1] across the STACK:
                up_rows = EXPERT_INTER
                # a rank's 2*inter/tp row slice intersects exactly ONE of
                # the up|gate halves (ranks 0..tp/2-1 in up, the rest in
                # gate) - the partial rows all-reduce before silu-mul.
                for name, base in ((up, 0), (gate, up_rows)):
                    lo = max(w1_r0, base) - base
                    hi = min(w1_r1, base + up_rows) - base
                    if hi > lo:
                        yield source.expert_payload(name, lo, hi, w1_c0, w1_c1)
                for name, base in ((up, 0), (gate, up_rows)):
                    lo = max(w1_r0, base) - base
                    hi = min(w1_r1, base + up_rows) - base
                    if hi > lo:
                        yield source.expert_scale(name, lo, hi, w1_c0, w1_c1)

        def produce_w2() -> Iterator[bytes]:
            for expert in range(EXPERTS):
                down = f"{prefix}.{expert}.down_proj.weight"
                yield source.expert_payload(down, w2_r0, w2_r1, w2_c0, w2_c1)
                yield source.expert_scale(down, w2_r0, w2_r1, w2_c0, w2_c1)

        def empty() -> Iterator[bytes]:
            return iter(())

        self.plan.append(PlanItem(w1, produce_w1, empty))
        self.plan.append(PlanItem(w2, produce_w2, empty))

    # -- the plan -----------------------------------------------------------

    def build(self) -> None:
        s = self.s
        p = f"model.language_model.layers"
        last_layer = self.first_layer + self.layer_count - (0 if self.include_mtp else 1)
        for layer in range(self.first_layer, last_layer + 1):
            a = f"{p}.{layer}.self_attn."
            m = f"{p}.{layer}.mlp."
            self.add_spine_bf16(K_ATTN_NORM, layer, f"{p}.{layer}.input_layernorm.weight")
            self.add_spine_bf16(K_POST_ATTN_NORM, layer, f"{p}.{layer}.post_attention_layernorm.weight")
            if layer < LAYERS and is_kda(layer):
                # the pack-V2 fusions
                self.add_fused_rows(K_KDA_QKV_BETA, layer,
                    [f"{a}q_proj.weight", f"{a}k_proj.weight", f"{a}v_proj.weight", f"{a}b_proj.weight"],
                    [KDA_DIM, KDA_DIM, KDA_DIM, KDA_HEADS], shard="rows")
                self.add_fused_rows(K_KDA_DECAY_GATE_DOWN, layer,
                    [f"{a}f_a_proj.weight", f"{a}g_a_proj.weight"],
                    [KDA_LOW_RANK, KDA_LOW_RANK])
                self.add_spine_bf16(K_KDA_DECAY_UP, layer, f"{a}f_b_proj.weight", shard="rows")
                self.add_spine_bf16(K_KDA_GATE_UP, layer, f"{a}g_b_proj.weight", shard="rows")
                self.add_kda_conv(K_KDA_Q_CONV, layer, f"{a}q_conv1d.weight")
                self.add_kda_conv(K_KDA_K_CONV, layer, f"{a}k_conv1d.weight")
                self.add_kda_conv(K_KDA_V_CONV, layer, f"{a}v_conv1d.weight")
                self.add_f32_slice(K_KDA_DECAY_BIAS, layer, f"{a}dt_bias", axis="cols")
                self.add_f32_slice(K_KDA_HEAD_LOG_SCALE, layer, f"{a}A_log", axis="cols")
                self.add_spine_f32(K_KDA_OUT_NORM, layer, f"{a}o_norm.weight")
                # o_proj is checkpoint [hidden, heads*dim] = out-hidden x
                # in-width, the down-projection family: the rank slice is
                # the INPUT columns (this rank's heads) and the module's
                # out-GEMM lands the full-width rank partial the chain
                # reduces. Row-sharding transposed the block - the pack
                # held [hidden/tp, width] where the consumer reads
                # [hidden, width/tp]: garbage attention partials on every
                # rank from layer 0 (the cold-first-request degeneration;
                # TP1-invariant, so the M3 gates passed). Same for the DSA
                # K_ATTN_OUTPUT below.
                self.add_spine_bf16(K_KDA_OUT, layer, f"{a}o_proj.weight", shard="cols")
            elif layer < LAYERS or layer == MTP_LAYER:
                # DSA (or the MTP layer - same tensor set minus HC)
                self.add_spine_bf16(K_Q_A, layer, f"{a}q_a_proj.weight")
                self.add_spine_bf16(K_Q_A_NORM, layer, f"{a}q_a_layernorm.weight")
                self.add_spine_bf16(K_Q_B, layer, f"{a}q_b_proj.weight", shard="rows")
                self.add_spine_bf16(K_KV_A, layer, f"{a}kv_a_proj_with_mqa.weight")
                self.add_spine_bf16(K_KV_A_NORM, layer, f"{a}kv_a_layernorm.weight")
                self.add_kv_b(layer)
                self.add_spine_bf16(K_ATTN_OUTPUT, layer, f"{a}o_proj.weight", shard="cols")
                i = f"{a}indexer."
                self.add_spine_bf16(K_INDEX_Q, layer, f"{i}wq_b.weight")  # replicated (glm52 pattern)
                self.add_spine_bf16(K_INDEX_K, layer, f"{i}wk.weight")
                self.add_spine_bf16(K_INDEX_HEAD, layer, f"{i}weights_proj.weight")  # replicated: the format table keeps the full 32 head weights per rank
                self.add_spine_bf16(K_INDEX_NORM_W, layer, f"{i}k_norm.weight")
                self.add_spine_bf16(K_INDEX_NORM_B, layer, f"{i}k_norm.bias")
                self.add_spine_f32(K_INDEX_COMPRESS_APE, layer, f"{i}index_kpool_compress_ape")
                self.add_spine_bf16(K_INDEX_COMPRESS_GATE, layer, f"{i}index_kpool_compress_gate")
            if layer < LAYERS:
                # hyper-connections on every weight layer (not MTP)
                for site, kind_fn, kind_base, kind_scale in (
                        ("attn", K_HC_ATTN_FN, K_HC_ATTN_BASE, K_HC_ATTN_SCALE),
                        ("ffn", K_HC_FFN_FN, K_HC_FFN_BASE, K_HC_FFN_SCALE)):
                    self.add_spine_f32(kind_fn, layer, f"{p}.{layer}.hc_{site}_fn")
                    self.add_spine_f32(kind_base, layer, f"{p}.{layer}.hc_{site}_base")
                    self.add_spine_f32(kind_scale, layer, f"{p}.{layer}.hc_{site}_scale")
            if layer < FIRST_ROUTED:
                self.add_up_gate_fused(K_DENSE_GATE_UP, layer,
                    f"{m}up_proj.weight", f"{m}gate_proj.weight", shard="rows")
                self.add_spine_bf16(K_DENSE_DOWN, layer, f"{m}down_proj.weight", shard="cols")
            else:
                self.add_spine_bf16(K_ROUTER, layer, f"{m}gate.weight")
                self.add_f32_slice(K_ROUTER_CORRECTION, layer, f"{m}gate.e_score_correction_bias", axis="none")  # replicated
                self.add_experts(layer)
                self.add_up_gate_fused(K_SHARED_GATE_UP, layer,
                    f"{m}shared_experts.up_proj.weight", f"{m}shared_experts.gate_proj.weight",
                    shard="rows")
                self.add_spine_bf16(K_SHARED_DOWN, layer, f"{m}shared_experts.down_proj.weight", shard="cols")
            if layer == MTP_LAYER:
                self.add_spine_bf16(K_MTP_EH_PROJ, layer, f"{p}.{layer}.eh_proj.weight")
                self.add_spine_bf16(K_MTP_ENORM, layer, f"{p}.{layer}.enorm.weight")
                self.add_spine_bf16(K_MTP_HNORM, layer, f"{p}.{layer}.hnorm.weight")
                self.add_spine_bf16(K_MTP_SHARED_NORM, layer, f"{p}.{layer}.shared_head.norm.weight")
        if self.owns_embedding:
            self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER,
                                "model.language_model.embed_tokens.weight", shard="rows")
        if self.owns_head:
            self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.language_model.norm.weight")
            self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", shard="rows")


def assemble_header(packer: Packer, header_extra: Dict[str, Any], file_bytes: int,
                    revision: str, contract_sha256: str) -> bytes:
    """Serialize SparkGlm5NextStagePackHeader exactly (C layout)."""
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


def emit(packer: Packer, path: Path, header_extra: Dict[str, Any]) -> None:
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
                             file_bytes, REVISION, CONTRACT_SHA256)
    with path.open("wb") as out:
        out.write(header)
        out.seek(directory_offset)
        for item in packer.plan:
            out.write(serialize_entry(item.entry))
        for item in packer.plan:
            out.seek(item.entry.payload_offset)
            for chunk in item.produce_payload():
                out.write(chunk)
            if item.entry.scale_bytes and item.produce_scale:
                out.seek(item.entry.scale_offset)
                for chunk in item.produce_scale():
                    out.write(chunk)
    print(f"{path.name}: {len(packer.plan)} tensors, {file_bytes} bytes "
          f"(tp{packer.tp_degree} rank {packer.tp_rank}, layers "
          f"{header_extra['first_layer']}..{header_extra['first_layer'] + header_extra['layer_count'] - 1})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkpoint directory (warm ceph)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=LAYERS)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--owns-embedding", action="store_true")
    parser.add_argument("--owns-head", action="store_true")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--tp-all", type=int, default=0,
                        help="emit all N rank packs in one process (shared dequant cache)")
    parser.add_argument("--dry-plan", action="store_true",
                        help="plan and print the inventory without writing")
    args = parser.parse_args()

    source = SourceReader(Path(args.source))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks = range(args.tp_all) if args.tp_all else [args.tp_rank]
    for rank in ranks:
        packer = Packer(source, args.tp_all or args.tp_degree, rank,
                        args.first_layer, args.layer_count, args.mtp,
                        args.owns_embedding, args.owns_head)
        if args.dry_plan:
            packer.build()
            print(f"rank {rank}: {len(packer.plan)} tensors planned")
            continue
        emit(packer, out_dir / f"glm5_next_stage.tp{args.tp_all or args.tp_degree}"
                              f".rank{rank}.g5nsp",
             dict(stage_count=1, stage_index=0, first_layer=args.first_layer,
                  layer_count=args.layer_count,
                  flags=1 if args.mtp else 0))
    source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
