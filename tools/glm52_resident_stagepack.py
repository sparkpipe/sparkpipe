#!/usr/bin/env python3
"""Build GLM-5.2 resident-decode stage packs (v3 .glm52sp).

Wire format: modules/glm52_resident_decode_stage/source/spark_glm52_stagepack_format.h
Layout: header (264B) | directory (N * 64B) | 256B-aligned payloads | scales.

Precision policy (full-fidelity spine, quantized experts):
  - spine tensors (embedding, lm_head, norms, attention, indexer, dense MLP,
    router, shared experts): read directly from the BF16 master checkpoint
    zai-org/GLM-5.2 @ b4734de4 and stored losslessly as BF16.
  - routed experts: read from the FP8 checkpoint zai-org/GLM-5.2-FP8; payload
    bytes copied as-is (codec 5) with F32 scales per 128-column block.
    scale_inv is the dequant multiplier; each per-tile value is expanded
    across its block's 128 rows (SparkWeightCodecScaleBytes).
  - expert up/gate order: [up_proj rows, gate_proj rows] stacked.

Streaming: offsets are computed arithmetically from the plan; tensors are
produced lazily per entry and written in chunks, so the full ~1 TB pack is
built in bounded memory.

The serving path never opens the checkpoints; this is setup-time code.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open

# Make the sibling shared packer core importable however this tool is loaded.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from spark_pack_common import (  # noqa: E402
    PackFailure,
    align_up,
    sha256_bytes,
    tp_shard_range,
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
SPINE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"


def load_contract(repo_root: Path) -> Dict[str, Any]:
    return json.loads((repo_root / "model_contracts" / "glm52.json").read_text())


class Reader:
    # Lockstep all-rank builds re-read the same tensor back-to-back for each
    # TP slice; a small LRU keeps one cold read per tensor instead of eight.
    CACHE_BYTE_CAP = 16 * 1024 ** 3

    def __init__(self, model_dir: Path):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise PackFailure(f"missing safetensors index: {index_path}")
        self.model_dir = model_dir
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.handles: Dict[str, Any] = {}
        self.cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.cache_bytes = 0

    def _handle(self, name: str):
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        if shard not in self.handles:
            path = self.model_dir / shard
            if not path.is_file():
                raise PackFailure(f"missing shard: {path}")
            self.handles[shard] = safe_open(str(path), framework="pt", device="cpu")
        handle = self.handles[shard]
        if name not in handle.keys():
            raise PackFailure(f"missing tensor in shard {shard}: {name}")
        return handle

    def shape(self, name: str) -> tuple:
        """Shape from metadata only (no payload read)."""
        handle = self._handle(name)
        try:
            return tuple(handle.get_slice(name).get_shape())
        except AttributeError:
            return tuple(self.tensor(name).shape)

    def tensor(self, name: str) -> torch.Tensor:
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        tensor = self._handle(name).get_tensor(name)
        size = tensor.numel() * tensor.element_size()
        self.cache[name] = tensor
        self.cache_bytes += size
        while self.cache_bytes > self.CACHE_BYTE_CAP and len(self.cache) > 1:
            _, evicted = self.cache.popitem(last=False)
            self.cache_bytes -= evicted.numel() * evicted.element_size()
        return tensor

    def close(self) -> None:
        self.handles.clear()
        self.cache.clear()
        self.cache_bytes = 0


def to_bytes(t: torch.Tensor) -> bytes:
    # numpy lacks bf16/fp8 dtypes; reinterpret as same-width unsigned ints
    if t.dtype == torch.bfloat16:
        t = t.contiguous().view(torch.uint16)
    elif t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = t.contiguous().view(torch.uint8)
    return t.contiguous().numpy().tobytes()


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
                 payload_bytes: int = 0, scale_bytes: int = 0):
        self.entry = entry
        self.produce_payload = produce_payload
        self.produce_scale = produce_scale
        self.entry.payload_bytes = payload_bytes
        self.entry.scale_bytes = scale_bytes


class Packer:
    def __init__(self, spine: Reader, experts: Reader, contract: Dict[str, Any],
                 layer_range: Tuple[int, int], include_embedding: bool = True,
                 include_head: bool = True, tp_degree: int = 1, tp_rank: int = 0):
        self.spine = spine
        self.experts = experts
        self.c = contract
        self.layer_range = layer_range
        self.include_embedding = include_embedding
        self.include_head = include_head
        self.tp_degree = tp_degree
        self.tp_rank = tp_rank
        self.plan: List[PlanItem] = []

    def shard_rows(self, t: torch.Tensor) -> torch.Tensor:
        """This TP rank's row slice; whole tensors when tp_degree == 1."""
        if self.tp_degree <= 1:
            return t
        start, count = tp_shard_range(t.shape[0], self.tp_degree, self.tp_rank)
        return t[start:start + count, :].contiguous()

    def shard_cols(self, t: torch.Tensor) -> torch.Tensor:
        if self.tp_degree <= 1:
            return t
        start, count = tp_shard_range(t.shape[1], self.tp_degree, self.tp_rank)
        return t[:, start:start + count].contiguous()

    # -- plan construction -------------------------------------------------

    def add_bf16(self, kind: int, layer: int, tensor: torch.Tensor, groups: int = 1):
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        rows, cols = tensor.shape
        if groups > 1:
            if rows % groups != 0:
                raise PackFailure(f"rows {rows} not divisible by groups {groups}")
            rows = rows // groups
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                      groups, rows, cols)
        payload_bytes = groups * rows * cols * 2
        blob = to_bytes(tensor)
        self.plan.append(PlanItem(entry, lambda: iter([blob]), None, payload_bytes, 0))

    def add_f32(self, kind: int, layer: int, tensor: torch.Tensor):
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(torch.float32)
        rows, cols = tensor.shape
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE,
                      1, rows, cols)
        payload_bytes = rows * cols * 4
        blob = to_bytes(tensor)
        self.plan.append(PlanItem(entry, lambda: iter([blob]), None, payload_bytes, 0))

    def add_spine_bf16(self, kind: int, layer: int, name: str, shard: str = ""):
        """Lazily-materialized spine tensor: the plan records shape metadata
        only; the payload bytes are produced at write time from the (LRU
        cached) reader, so plans for every TP rank fit in memory together."""
        shape = self.spine.shape(name)
        rows, cols = (shape if len(shape) == 2 else (1, shape[0]))
        if shard == "rows" and self.tp_degree > 1:
            start, count = tp_shard_range(rows, self.tp_degree, self.tp_rank)
            rows = count
        elif shard == "cols" and self.tp_degree > 1:
            start, count = tp_shard_range(cols, self.tp_degree, self.tp_rank)
            cols = count
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows, cols)
        payload_bytes = rows * cols * 2
        spine = self.spine
        tp_degree, tp_rank = self.tp_degree, self.tp_rank

        def produce() -> Iterator[bytes]:
            t = spine.tensor(name)
            if t.dtype != torch.bfloat16:
                raise PackFailure(f"{name}: spine tensor must be BF16, got {t.dtype}")
            if shard == "rows" and tp_degree > 1:
                s, n = tp_shard_range(t.shape[0], tp_degree, tp_rank)
                t = t[s:s + n, :].contiguous()
            elif shard == "cols" and tp_degree > 1:
                s, n = tp_shard_range(t.shape[1], tp_degree, tp_rank)
                t = t[:, s:s + n].contiguous()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            yield to_bytes(t)

        self.plan.append(PlanItem(entry, produce, None, payload_bytes, 0))

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
        experts = self.experts

        def produce_payload() -> Iterator[bytes]:
            for expert in range(EXPERT_COUNT):
                for proj in projections:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                    w = experts.tensor(name + ".weight")
                    if w.shape != (rows, columns):
                        raise PackFailure(
                            f"{name}.weight shape {tuple(w.shape)} != ({rows}, {columns})")
                    yield to_bytes(w[r0:r1, c0:c1])

        def produce_scale() -> Iterator[bytes]:
            for expert in range(EXPERT_COUNT):
                for proj in projections:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                    s_inv = experts.tensor(name + ".weight_scale_inv")
                    expanded = s_inv.to(torch.float32).repeat_interleave(128, dim=0)
                    yield to_bytes(expanded[r0:r1, c0 // 128:c1 // 128])

        self.plan.append(PlanItem(entry, produce_payload, produce_scale,
                                  payload_bytes, scale_bytes))

    def add_kv_b(self, layer: int):
        name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        heads = self.c["head_count"]
        qk_nope = self.c["qk_nope_head_dimension"]
        value_dim = self.c["value_head_dimension"]
        latent = self.c["latent_dimension"]
        expected_rows = heads * (qk_nope + value_dim)
        shape = self.spine.shape(name)
        if shape != (expected_rows, latent):
            raise PackFailure(f"kv_b shape {shape} != ({expected_rows}, {latent})")
        spine = self.spine
        self._add_kv_b_part(K_KV_B_KEY_T, layer, name, heads, qk_nope, value_dim,
                            latent, spine, key=True)
        self._add_kv_b_part(K_KV_B_VALUE, layer, name, heads, qk_nope, value_dim,
                            latent, spine, key=False)

    def _add_kv_b_part(self, kind: int, layer: int, name: str, heads: int,
                       qk_nope: int, value_dim: int, latent: int,
                       spine: "Reader", key: bool):
        rows = heads * (latent if key else value_dim)
        cols = qk_nope if key else latent
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE,
                      heads, rows // heads, cols)
        payload_bytes = rows * cols * 2

        def produce() -> Iterator[bytes]:
            w = spine.tensor(name)
            if w.dtype != torch.bfloat16:
                raise PackFailure(f"kv_b must be BF16, got {w.dtype}")
            parts = []
            per_head = qk_nope + value_dim
            for head in range(heads):
                block = w[head * per_head:(head + 1) * per_head, :]
                if key:
                    parts.append(block[:qk_nope, :].t().contiguous())   # [latent, qk_nope]
                else:
                    parts.append(block[qk_nope:, :])                    # [value, latent]
            stacked = torch.stack(parts, dim=0).contiguous()
            yield to_bytes(stacked.reshape(rows, cols))

        self.plan.append(PlanItem(entry, produce, None, payload_bytes, 0))

    def add_dense_gate_up(self, layer: int):
        self._add_fused_gate_up(K_DENSE_GATE_UP, layer,
                                f"model.layers.{layer}.mlp.gate_proj.weight",
                                f"model.layers.{layer}.mlp.up_proj.weight")

    def add_shared_gate_up(self, layer: int):
        self._add_fused_gate_up(K_SHARED_GATE_UP, layer,
                                f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight",
                                f"model.layers.{layer}.mlp.shared_experts.up_proj.weight")

    def _add_fused_gate_up(self, kind: int, layer: int, gate_name: str, up_name: str):
        gate_shape = self.spine.shape(gate_name)
        up_shape = self.spine.shape(up_name)
        rows, cols = up_shape
        start = count = None
        if self.tp_degree > 1:
            start, count = tp_shard_range(rows, self.tp_degree, self.tp_rank)
        entry_rows = count if count is not None else rows
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1,
                      2 * entry_rows, cols)
        payload_bytes = 2 * entry_rows * cols * 2
        spine = self.spine
        tp_degree, tp_rank = self.tp_degree, self.tp_rank

        def produce() -> Iterator[bytes]:
            def shard(t: torch.Tensor) -> torch.Tensor:
                if tp_degree <= 1:
                    return t
                s, n = tp_shard_range(t.shape[0], tp_degree, tp_rank)
                return t[s:s + n, :].contiguous()
            gate = shard(spine.tensor(gate_name))
            up = shard(spine.tensor(up_name))
            if up.shape != gate.shape:
                raise PackFailure(f"{up_name} vs {gate_name} shape drift after shard")
            yield to_bytes(torch.cat([up, gate], dim=0))

        self.plan.append(PlanItem(entry, produce, None, payload_bytes, 0))

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
                             self.spine.tensor(
                                 f"model.layers.{layer}.mlp.gate.e_score_correction_bias"))
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

    def write(self, output_path: Path, model_revision: str,
              contract_sha256: bytes, source_config_sha256: bytes,
              pack_recipe_sha256: bytes, stage_count: int, stage_index: int,
              first_layer_index: int, layer_count: int, total_layer_count: int):
        entries = [item.entry for item in self.plan]
        tensor_count = len(entries)
        directory_offset = align_up(HEADER_BYTES, ALIGNMENT)
        directory_bytes = tensor_count * ENTRY_BYTES
        offset = align_up(directory_offset + directory_bytes, ALIGNMENT)
        for item in self.plan:
            entry = item.entry
            entry.payload_offset = offset
            offset += entry.payload_bytes
        for item in self.plan:
            entry = item.entry
            if entry.scale_bytes:
                offset = align_up(offset, ALIGNMENT)
                entry.scale_offset = offset
                offset += entry.scale_bytes
        file_bytes = offset

        with output_path.open("wb") as f:
            header = struct.pack(
                "<20I2Q65s32s32s32s",
                MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES,
                CODEC_ABI_VERSION, 0, tensor_count, stage_count, stage_index,
                first_layer_index, layer_count, total_layer_count,
                self.c["hidden_dimension"], self.c["output_vocab_count"],
                self.c["moe_expert_count"], CODEC_BF16, CODEC_FP8, CODEC_BF16,
                self.tp_degree, self.tp_rank, directory_offset, file_bytes,
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
            for item in self.plan:
                f.seek(item.entry.payload_offset)
                if item.produce_payload is not None:
                    for chunk in item.produce_payload():
                        f.write(chunk)
                if item.produce_scale is not None:
                    f.seek(item.entry.scale_offset)
                    for chunk in item.produce_scale():
                        f.write(chunk)
        return file_bytes

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


def write_all_ranks(packers, output_paths, model_revision, contract_sha256,
                    source_config_sha256, pack_recipe_sha256, stage_count, stage_index,
                    first_layer_index, layer_count, total_layer_count):
    """Lockstep fan-out writer: every packer walks its plan in the same
    tensor order, so driving entry i of all ranks back-to-back serves the
    shared source reads from the page cache - one cold pass for the whole
    TP set instead of one per rank."""
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
    parser.add_argument("--spine-dir", required=True,
                        help="BF16 master checkpoint (full-fidelity spine)")
    parser.add_argument("--expert-dir", required=True,
                        help="FP8 checkpoint (routed experts)")
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
                             "pass over the shared sources; --output is a "
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

    spine = Reader(Path(args.spine_dir))
    experts = Reader(Path(args.expert_dir))
    if args.all_ranks:
        packers = []
        for rank in range(args.tp_degree):
            packer = Packer(spine, experts, contract, (first, last),
                            include_embedding, include_head, args.tp_degree, rank)
            packer.build_plan()
            packers.append(packer)
        contract_bytes = (repo_root / "examples" / "model_descriptions" /
                          "glm52_resident_decode_stage_fp8_firmware.json").read_bytes()
        recipe = json.dumps({"tool": "glm52_resident_stagepack.py",
                             "codec": "fp8", "spine": "bf16-master",
                             "expert_source_contract": "model_contracts/glm52_authoritative.json"}, sort_keys=True)
        output_paths = [Path(args.output.format(rank=rank))
                        for rank in range(args.tp_degree)]
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        source_config = json.dumps(
            {"layer_range": f"{first}-{last}", "stage": args.stage,
             "tp_degree": args.tp_degree, "tp_rank": "all",
             "spine_dir": args.spine_dir, "expert_dir": args.expert_dir},
            sort_keys=True)
        file_sizes = write_all_ranks(
            packers, output_paths, args.model_revision,
            sha256_bytes(contract_bytes), sha256_bytes(source_config.encode()),
            sha256_bytes(recipe.encode()), stage_count, stage_index,
            first, last - first + 1, contract["layer_count"])
        spine.close()
        experts.close()
        for path, size in zip(output_paths, file_sizes):
            print(f"glm52sp written: {path} bytes={size} "
                  f"stage={stage_index}/{stage_count} layers={first}-{last} "
                  f"tensors={len(packers[0].plan)} tp={args.tp_degree}")
        return 0
    packer = Packer(spine, experts, contract, (first, last),
                    include_embedding, include_head, args.tp_degree, args.tp_rank)
    packer.build_plan()

    # The "contract" identity is the model DESCRIPTION file's sha256: the module
    # and serving adapter compile it in as GLM52_CONTRACT_SHA256 and the driver
    # embeds it as model_description_sha256, so the pack header must match.
    contract_bytes = (repo_root / "examples" / "model_descriptions" /
                      "glm52_resident_decode_stage_fp8_firmware.json").read_bytes()
    source_config = json.dumps(
        {"layer_range": f"{first}-{last}", "stage": args.stage,
         "tp_degree": args.tp_degree, "tp_rank": args.tp_rank,
         "spine_dir": args.spine_dir, "expert_dir": args.expert_dir},
        sort_keys=True)
    recipe = json.dumps({"tool": "glm52_resident_stagepack.py",
                         "codec": "fp8", "spine": "bf16-master",
                         "expert_source_contract": "model_contracts/glm52_authoritative.json"}, sort_keys=True)
    file_bytes = packer.write(
        Path(args.output), args.model_revision,
        sha256_bytes(contract_bytes), sha256_bytes(source_config.encode()),
        sha256_bytes(recipe.encode()), stage_count, stage_index,
        first, last - first + 1, contract["layer_count"])
    spine.close()
    experts.close()
    print(f"glm52sp written: {args.output} bytes={file_bytes} "
          f"stage={stage_index}/{stage_count} layers={first}-{last} "
          f"tensors={len(packer.plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
