#!/usr/bin/env python3
"""Build one strict GLM 5.2 PP13 stage pack from rank-local BF16 shards."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors import safe_open

TOOLS_DIRECTORY = Path(__file__).resolve().parent
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0,str(TOOLS_DIRECTORY))
from glm52_model_contract import load_model_contract
from spark_pack_common import PackFailure, align_up, sha256_file


MAGIC = 0x32534C47
FORMAT_VERSION = 3
CODEC_ABI_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
MODEL = load_model_contract(ROOT)
LAYERS_PER_STAGE = 6
LAYER_COUNT = MODEL["layer_count"]
STAGE_COUNT = LAYER_COUNT // LAYERS_PER_STAGE
HIDDEN = MODEL["hidden_dimension"]
VOCAB = MODEL["output_vocab_count"]
HEADS = MODEL["head_count"]
Q_A = MODEL["query_a_dimension"]
QK_NOPE = MODEL["qk_nope_head_dimension"]
QK_HEAD = QK_NOPE + MODEL["rope_dimension"]
VALUE_HEAD = MODEL["value_head_dimension"]
LATENT = MODEL["latent_dimension"]
CACHE_TOKEN = LATENT + MODEL["rope_dimension"]
INDEX_HEADS = MODEL["dsa_index_head_count"]
INDEX_HEAD = MODEL["dsa_index_head_dimension"]
INDEX_QUERY = INDEX_HEADS * INDEX_HEAD
DENSE_INTERMEDIATE = MODEL["dense_intermediate_dimension"]
EXPERTS = MODEL["moe_expert_count"]
EXPERT_INTERMEDIATE = MODEL["moe_intermediate_dimension"]
ALIGNMENT = 256
GLOBAL_LAYER = 0xFFFFFFFF
HEADER = struct.Struct("<20I2Q65s32s32s32s7x")
ENTRY = struct.Struct("<8I4Q")
HEADER_BYTES = HEADER.size
ENTRY_BYTES = ENTRY.size
PAYLOAD_BF16 = 1
PAYLOAD_F32 = 2
PAYLOAD_PACKED_WEIGHT = 4
SCALE_NONE = 0
SCALE_F32 = 1
SCALE_E8M0 = 3
SCALE_UE4M3_F32_GLOBAL = 4
CODEC_BF16 = 1
CODECS = {
    "int6": (2, 6, 32, SCALE_F32),
    "int7": (3, 7, 128, SCALE_F32),
    "int8": (4, 8, 128, SCALE_F32),
    "fp8": (5, 8, 128, SCALE_F32),
    "nvfp4": (6, 4, 16, SCALE_UE4M3_F32_GLOBAL),
    "mxfp4": (7, 4, 32, SCALE_E8M0),
}

TENSOR_EMBEDDING = 0
TENSOR_FINAL_NORM = 1
TENSOR_LM_HEAD = 2
TENSOR_ATTN_NORM = 3
TENSOR_Q_A = 4
TENSOR_Q_A_NORM = 5
TENSOR_Q_B = 6
TENSOR_KV_A = 7
TENSOR_KV_A_NORM = 8
TENSOR_KV_B_KEY_TRANSPOSED = 9
TENSOR_KV_B_VALUE = 10
TENSOR_ATTN_OUTPUT = 11
TENSOR_POST_ATTN_NORM = 12
TENSOR_INDEX_Q = 13
TENSOR_INDEX_K = 14
TENSOR_INDEX_HEAD = 15
TENSOR_INDEX_NORM_WEIGHT = 16
TENSOR_INDEX_NORM_BIAS = 17
TENSOR_DENSE_GATE_UP = 18
TENSOR_DENSE_DOWN = 19
TENSOR_ROUTER = 20
TENSOR_ROUTER_CORRECTION = 21
TENSOR_EXPERT_UP_GATE = 22
TENSOR_EXPERT_DOWN = 23
TENSOR_SHARED_GATE_UP = 24
TENSOR_SHARED_DOWN = 25

KIND_NAMES = {
    TENSOR_EMBEDDING: "embedding",
    TENSOR_FINAL_NORM: "final_norm",
    TENSOR_LM_HEAD: "lm_head",
    TENSOR_ATTN_NORM: "attention_norm",
    TENSOR_Q_A: "query_a",
    TENSOR_Q_A_NORM: "query_a_norm",
    TENSOR_Q_B: "query_b",
    TENSOR_KV_A: "kv_a",
    TENSOR_KV_A_NORM: "kv_a_norm",
    TENSOR_KV_B_KEY_TRANSPOSED: "kv_b_key_transposed",
    TENSOR_KV_B_VALUE: "kv_b_value",
    TENSOR_ATTN_OUTPUT: "attention_output",
    TENSOR_POST_ATTN_NORM: "post_attention_norm",
    TENSOR_INDEX_Q: "index_query",
    TENSOR_INDEX_K: "index_key",
    TENSOR_INDEX_HEAD: "index_head",
    TENSOR_INDEX_NORM_WEIGHT: "index_norm_weight",
    TENSOR_INDEX_NORM_BIAS: "index_norm_bias",
    TENSOR_DENSE_GATE_UP: "dense_gate_up",
    TENSOR_DENSE_DOWN: "dense_down",
    TENSOR_ROUTER: "router",
    TENSOR_ROUTER_CORRECTION: "router_correction",
    TENSOR_EXPERT_UP_GATE: "expert_up_gate",
    TENSOR_EXPERT_DOWN: "expert_down",
    TENSOR_SHARED_GATE_UP: "shared_gate_up",
    TENSOR_SHARED_DOWN: "shared_down",
}


@dataclasses.dataclass
class Record:
    kind: int
    layer: int
    payload_type: int
    codec: int
    scale_encoding: int
    groups: int
    rows: int
    columns: int
    payload_offset: int = 0
    payload_bytes: int = 0
    scale_offset: int = 0
    scale_bytes: int = 0


class TensorSource:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.index_path = model_dir / "model.safetensors.index.json"
        self.config_path = model_dir / "config.json"
        if not self.index_path.is_file() or not self.config_path.is_file():
            raise PackFailure("model directory requires config.json and model.safetensors.index.json")
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.weight_map = index.get("weight_map")
        if not isinstance(self.weight_map, dict) or not self.weight_map:
            raise PackFailure("safetensors index has no weight_map")

    def validate_names(self, names: Iterable[str]) -> None:
        missing_names = sorted({name for name in names if name not in self.weight_map})
        if missing_names:
            raise PackFailure(f"source index is missing tensors: {missing_names[:8]}")
        missing_files = sorted({
            self.weight_map[name]
            for name in names
            if not (self.model_dir / self.weight_map[name]).is_file()
        })
        if missing_files:
            raise PackFailure(f"rank-local source is missing shards: {missing_files[:8]}")

    def load(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        shard = self.model_dir / self.weight_map[name]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise PackFailure(
                f"{name}: expected {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}"
            )
        return tensor.contiguous()


def align(value: int) -> int:
    return align_up(value, ALIGNMENT)


def has_full_indexer(layer: int) -> bool:
    return layer < 3 or (layer >= 6 and (layer - 6) % 4 == 0)


def record_bf16(kind: int, layer: int, groups: int, rows: int, columns: int) -> Record:
    return Record(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, groups, rows, columns)


def records_for_stage(stage: int, codec_name: str) -> list[Record]:
    codec, bits, group_size, scale_encoding = CODECS[codec_name]
    del bits, group_size
    first = stage * LAYERS_PER_STAGE
    records: list[Record] = []
    if stage == 0:
        records.append(record_bf16(TENSOR_EMBEDDING, GLOBAL_LAYER, 1, VOCAB, HIDDEN))
    for layer in range(first, first + LAYERS_PER_STAGE):
        records.extend([
            record_bf16(TENSOR_ATTN_NORM, layer, 1, 1, HIDDEN),
            record_bf16(TENSOR_Q_A, layer, 1, Q_A, HIDDEN),
            record_bf16(TENSOR_Q_A_NORM, layer, 1, 1, Q_A),
            record_bf16(TENSOR_Q_B, layer, 1, HEADS * QK_HEAD, Q_A),
            record_bf16(TENSOR_KV_A, layer, 1, CACHE_TOKEN, HIDDEN),
            record_bf16(TENSOR_KV_A_NORM, layer, 1, 1, LATENT),
            record_bf16(TENSOR_KV_B_KEY_TRANSPOSED, layer, HEADS, LATENT, QK_NOPE),
            record_bf16(TENSOR_KV_B_VALUE, layer, HEADS, VALUE_HEAD, LATENT),
            record_bf16(TENSOR_ATTN_OUTPUT, layer, 1, HIDDEN, HEADS * VALUE_HEAD),
            record_bf16(TENSOR_POST_ATTN_NORM, layer, 1, 1, HIDDEN),
        ])
        if has_full_indexer(layer):
            records.extend([
                record_bf16(TENSOR_INDEX_Q, layer, 1, INDEX_QUERY, Q_A),
                record_bf16(TENSOR_INDEX_K, layer, 1, INDEX_HEAD, HIDDEN),
                record_bf16(TENSOR_INDEX_HEAD, layer, 1, INDEX_HEADS, HIDDEN),
                record_bf16(TENSOR_INDEX_NORM_WEIGHT, layer, 1, 1, INDEX_HEAD),
                record_bf16(TENSOR_INDEX_NORM_BIAS, layer, 1, 1, INDEX_HEAD),
            ])
        if layer < 3:
            records.extend([
                record_bf16(TENSOR_DENSE_GATE_UP, layer, 1, DENSE_INTERMEDIATE * 2, HIDDEN),
                record_bf16(TENSOR_DENSE_DOWN, layer, 1, HIDDEN, DENSE_INTERMEDIATE),
            ])
        else:
            records.extend([
                record_bf16(TENSOR_ROUTER, layer, 1, EXPERTS, HIDDEN),
                Record(TENSOR_ROUTER_CORRECTION, layer, PAYLOAD_F32, 0, SCALE_NONE, 1, 1, EXPERTS),
                Record(TENSOR_EXPERT_UP_GATE, layer, PAYLOAD_PACKED_WEIGHT, codec, scale_encoding, EXPERTS, EXPERT_INTERMEDIATE * 2, HIDDEN),
                Record(TENSOR_EXPERT_DOWN, layer, PAYLOAD_PACKED_WEIGHT, codec, scale_encoding, EXPERTS, HIDDEN, EXPERT_INTERMEDIATE),
                record_bf16(TENSOR_SHARED_GATE_UP, layer, 1, EXPERT_INTERMEDIATE * 2, HIDDEN),
                record_bf16(TENSOR_SHARED_DOWN, layer, 1, HIDDEN, EXPERT_INTERMEDIATE),
            ])
    if stage + 1 == STAGE_COUNT:
        records.extend([
            record_bf16(TENSOR_FINAL_NORM, GLOBAL_LAYER, 1, 1, HIDDEN),
            record_bf16(TENSOR_LM_HEAD, GLOBAL_LAYER, 1, VOCAB, HIDDEN),
        ])
    return records


def size_records(records: list[Record], codec_name: str) -> int:
    _, bits, group_size, scale_encoding = CODECS[codec_name]
    cursor = align(align(HEADER_BYTES) + len(records) * ENTRY_BYTES)
    for record in records:
        elements = record.groups * record.rows * record.columns
        if record.payload_type == PAYLOAD_BF16:
            record.payload_bytes = elements * 2
        elif record.payload_type == PAYLOAD_F32:
            record.payload_bytes = elements * 4
        else:
            record.payload_bytes = elements * bits // 8
        record.payload_offset = cursor
        cursor = align(cursor + record.payload_bytes)
        if record.payload_type == PAYLOAD_PACKED_WEIGHT:
            blocks = record.columns // group_size
            block_bytes = record.groups * record.rows * blocks
            record.scale_bytes = block_bytes * (4 if scale_encoding == SCALE_F32 else 1)
            if codec_name == "nvfp4":
                record.scale_bytes += record.groups * 4
            record.scale_offset = cursor
            cursor = align(cursor + record.scale_bytes)
    return cursor


def direct_names(layer: int) -> dict[int, str]:
    prefix = f"model.layers.{layer}"
    return {
        TENSOR_ATTN_NORM: f"{prefix}.input_layernorm.weight",
        TENSOR_Q_A: f"{prefix}.self_attn.q_a_proj.weight",
        TENSOR_Q_A_NORM: f"{prefix}.self_attn.q_a_layernorm.weight",
        TENSOR_Q_B: f"{prefix}.self_attn.q_b_proj.weight",
        TENSOR_KV_A: f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        TENSOR_KV_A_NORM: f"{prefix}.self_attn.kv_a_layernorm.weight",
        TENSOR_ATTN_OUTPUT: f"{prefix}.self_attn.o_proj.weight",
        TENSOR_POST_ATTN_NORM: f"{prefix}.post_attention_layernorm.weight",
        TENSOR_INDEX_Q: f"{prefix}.self_attn.indexer.wq_b.weight",
        TENSOR_INDEX_K: f"{prefix}.self_attn.indexer.wk.weight",
        TENSOR_INDEX_HEAD: f"{prefix}.self_attn.indexer.weights_proj.weight",
        TENSOR_INDEX_NORM_WEIGHT: f"{prefix}.self_attn.indexer.k_norm.weight",
        TENSOR_INDEX_NORM_BIAS: f"{prefix}.self_attn.indexer.k_norm.bias",
        TENSOR_DENSE_DOWN: f"{prefix}.mlp.down_proj.weight",
        TENSOR_ROUTER: f"{prefix}.mlp.gate.weight",
        TENSOR_ROUTER_CORRECTION: f"{prefix}.mlp.gate.e_score_correction_bias",
        TENSOR_SHARED_DOWN: f"{prefix}.mlp.shared_experts.down_proj.weight",
    }


def source_names_for_record(record: Record) -> list[str]:
    if record.layer == GLOBAL_LAYER:
        return {
            TENSOR_EMBEDDING: ["model.embed_tokens.weight"],
            TENSOR_FINAL_NORM: ["model.norm.weight"],
            TENSOR_LM_HEAD: ["lm_head.weight"],
        }[record.kind]
    prefix = f"model.layers.{record.layer}"
    if record.kind in (TENSOR_KV_B_KEY_TRANSPOSED, TENSOR_KV_B_VALUE):
        return [f"{prefix}.self_attn.kv_b_proj.weight"]
    if record.kind == TENSOR_DENSE_GATE_UP:
        return [f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"]
    if record.kind == TENSOR_SHARED_GATE_UP:
        return [f"{prefix}.mlp.shared_experts.gate_proj.weight", f"{prefix}.mlp.shared_experts.up_proj.weight"]
    if record.kind == TENSOR_EXPERT_UP_GATE:
        return [
            f"{prefix}.mlp.experts.{expert}.{projection}_proj.weight"
            for expert in range(EXPERTS)
            for projection in ("up", "gate")
        ]
    if record.kind == TENSOR_EXPERT_DOWN:
        return [f"{prefix}.mlp.experts.{expert}.down_proj.weight" for expert in range(EXPERTS)]
    return [direct_names(record.layer)[record.kind]]


def write_tensor(file, offset: int, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise PackFailure(f"write dtype mismatch: {tensor.dtype} != {dtype}")
    flat = tensor.contiguous().view(torch.uint16 if dtype == torch.bfloat16 else torch.float32).reshape(-1)
    chunk_elements = 16 * 1024 * 1024
    file.seek(offset)
    for first in range(0, flat.numel(), chunk_elements):
        part = flat[first:first + chunk_elements].numpy()
        file.write(part.tobytes(order="C"))


def pack_subbyte(codes: np.ndarray, bits: int) -> bytes:
    rows, columns = codes.shape
    if columns % 8 != 0:
        raise PackFailure("packed rows require a whole eight-code group")
    octets = (codes.astype(np.int16, copy=False) & ((1 << bits) - 1)).astype(np.uint64)
    octets = octets.reshape(rows, columns // 8, 8)
    packed = np.zeros(octets.shape[:2], dtype=np.uint64)
    for index in range(8):
        packed |= octets[:, :, index] << (index * bits)
    return packed.view(np.uint8).reshape(rows, columns // 8, 8)[:, :, :bits].tobytes(order="C")


def e4m3_values() -> np.ndarray:
    values = np.zeros(127, dtype=np.float32)
    for code in range(1, 127):
        exponent = code >> 3
        mantissa = code & 7
        if exponent == 0:
            values[code] = mantissa * (2.0 ** -9)
        else:
            values[code] = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return values


E4M3_VALUES = e4m3_values()


def encode_e4m3_round_up(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = np.searchsorted(E4M3_VALUES, values, side="left")
    codes = np.clip(codes, 0, 126).astype(np.uint8)
    return codes, E4M3_VALUES[codes]


def encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs()
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=values.device)
    codes = torch.bucketize(magnitude, boundaries, right=False)
    for boundary_index in (1, 3, 5):
        equal = magnitude == boundaries[boundary_index]
        codes = torch.where(equal, torch.full_like(codes, boundary_index + 1), codes)
    return(codes.to(torch.uint8) | torch.where(values < 0, 8, 0).to(torch.uint8))


def quantize_matrix(
    matrix: torch.Tensor,
    codec_name: str,
    device: torch.device,
) -> tuple[bytes, bytes, bytes]:
    _, bits, group_size, _ = CODECS[codec_name]
    rows, columns = matrix.shape
    if columns % group_size != 0 or columns * bits % 8 != 0:
        raise PackFailure(f"{codec_name} cannot tile matrix shape {tuple(matrix.shape)}")
    values = matrix.to(device=device, dtype=torch.float32)
    blocks = values.reshape(rows, columns // group_size, group_size)
    block_amax = blocks.abs().amax(dim=2)
    global_bytes = b""
    if codec_name in ("int6", "int7", "int8"):
        maximum = {"int6": 31.0, "int7": 63.0, "int8": 127.0}[codec_name]
        minimum = {"int6": -32, "int7": -64, "int8": -128}[codec_name]
        scales = torch.clamp(block_amax / maximum, min=1.0e-12)
        codes = torch.round(blocks / scales.unsqueeze(2)).clamp(minimum, int(maximum))
        codes = codes.reshape(rows, columns).to(torch.int8).cpu().numpy()
        payload = codes.view(np.uint8).tobytes(order="C") if bits == 8 else pack_subbyte(codes, bits)
        scale_bytes = scales.cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
    elif codec_name == "fp8":
        scales = torch.clamp(block_amax / 448.0, min=1.0e-12)
        normalized = (blocks / scales.unsqueeze(2)).reshape(rows, columns)
        codes = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
        payload = codes.cpu().numpy().tobytes(order="C")
        scale_bytes = scales.cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
    elif codec_name == "nvfp4":
        tensor_amax = float(block_amax.max().item())
        global_scale = max(tensor_amax / (448.0 * 6.0), 1.0e-30)
        raw_block_scales = (block_amax.cpu().numpy() / 6.0) / global_scale
        scale_codes, decoded_blocks = encode_e4m3_round_up(raw_block_scales)
        decoded = torch.from_numpy(decoded_blocks.copy()).to(device=device)
        final_scales = decoded.unsqueeze(2) * global_scale
        normalized = torch.where(final_scales > 0, blocks / final_scales, torch.zeros_like(blocks))
        codes = encode_e2m1(normalized).reshape(rows, columns).cpu().numpy()
        payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).tobytes(order="C")
        global_bytes = struct.pack("<f", global_scale)
        scale_bytes = scale_codes.tobytes(order="C")
    else:
        raw_scales = np.maximum(block_amax.cpu().numpy() / 6.0, np.float32(2.0 ** -127))
        exponents = np.ceil(np.log2(raw_scales)).astype(np.int32)
        scale_codes = np.clip(exponents + 127, 0, 254).astype(np.uint8)
        decoded = torch.from_numpy(np.exp2(scale_codes.astype(np.float32) - 127.0)).to(device=device)
        normalized = blocks / decoded.unsqueeze(2)
        codes = encode_e2m1(normalized).reshape(rows, columns).cpu().numpy()
        payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).tobytes(order="C")
        scale_bytes = scale_codes.tobytes(order="C")
    return(payload,global_bytes,scale_bytes)


def load_nonexpert(source: TensorSource, record: Record) -> torch.Tensor:
    if record.layer == GLOBAL_LAYER:
        name = source_names_for_record(record)[0]
        shape = (record.rows, record.columns) if record.rows != 1 else (record.columns,)
        return source.load(name, shape, torch.bfloat16)
    prefix = f"model.layers.{record.layer}"
    if record.kind in (TENSOR_KV_B_KEY_TRANSPOSED, TENSOR_KV_B_VALUE):
        tensor = source.load(f"{prefix}.self_attn.kv_b_proj.weight", (HEADS * (QK_NOPE + VALUE_HEAD), LATENT), torch.bfloat16)
        tensor = tensor.reshape(HEADS, QK_NOPE + VALUE_HEAD, LATENT)
        return(tensor[:, :QK_NOPE, :].transpose(1, 2).contiguous() if record.kind == TENSOR_KV_B_KEY_TRANSPOSED else tensor[:, QK_NOPE:, :].contiguous())
    if record.kind == TENSOR_DENSE_GATE_UP:
        gate = source.load(f"{prefix}.mlp.gate_proj.weight", (DENSE_INTERMEDIATE, HIDDEN), torch.bfloat16)
        up = source.load(f"{prefix}.mlp.up_proj.weight", (DENSE_INTERMEDIATE, HIDDEN), torch.bfloat16)
        return torch.cat((gate, up), dim=0)
    if record.kind == TENSOR_SHARED_GATE_UP:
        gate = source.load(f"{prefix}.mlp.shared_experts.gate_proj.weight", (EXPERT_INTERMEDIATE, HIDDEN), torch.bfloat16)
        up = source.load(f"{prefix}.mlp.shared_experts.up_proj.weight", (EXPERT_INTERMEDIATE, HIDDEN), torch.bfloat16)
        return torch.cat((gate, up), dim=0)
    name = direct_names(record.layer)[record.kind]
    if record.payload_type == PAYLOAD_F32:
        return source.load(name, (record.columns,), torch.float32)
    shape = (record.rows, record.columns) if record.rows != 1 else (record.columns,)
    return source.load(name, shape, torch.bfloat16)


def write_experts(
    file,
    source: TensorSource,
    record: Record,
    codec_name: str,
    device: torch.device,
) -> None:
    _, bits, group_size, scale_encoding = CODECS[codec_name]
    payload_per_expert = record.rows * record.columns * bits // 8
    blocks_per_expert = record.rows * (record.columns // group_size)
    scale_per_expert = blocks_per_expert * (4 if scale_encoding == SCALE_F32 else 1)
    block_scale_base = record.scale_offset + (EXPERTS * 4 if codec_name == "nvfp4" else 0)
    prefix = f"model.layers.{record.layer}.mlp.experts"
    for expert in range(EXPERTS):
        if record.kind == TENSOR_EXPERT_UP_GATE:
            up = source.load(f"{prefix}.{expert}.up_proj.weight", (EXPERT_INTERMEDIATE, HIDDEN), torch.bfloat16)
            gate = source.load(f"{prefix}.{expert}.gate_proj.weight", (EXPERT_INTERMEDIATE, HIDDEN), torch.bfloat16)
            matrix = torch.cat((up, gate), dim=0)
        else:
            matrix = source.load(f"{prefix}.{expert}.down_proj.weight", (HIDDEN, EXPERT_INTERMEDIATE), torch.bfloat16)
        payload,global_scale,block_scales = quantize_matrix(matrix,codec_name,device)
        if len(payload) != payload_per_expert or len(block_scales) != scale_per_expert:
            raise PackFailure("quantizer emitted a geometry inconsistent with the stage-pack directory")
        file.seek(record.payload_offset + expert * payload_per_expert)
        file.write(payload)
        if global_scale:
            file.seek(record.scale_offset + expert * 4)
            file.write(global_scale)
        file.seek(block_scale_base + expert * scale_per_expert)
        file.write(block_scales)
        if expert % 16 == 15 or expert + 1 == EXPERTS:
            print(json.dumps({"layer":record.layer,"tensor":KIND_NAMES[record.kind],"experts_done":expert + 1,"experts_total":EXPERTS}),flush=True)


def pack_header(
    stage: int,
    codec: int,
    model_revision: str,
    contract_sha: bytes,
    config_sha: bytes,
    recipe_sha: bytes,
    tensor_count: int,
    file_bytes: int,
) -> bytes:
    revision = model_revision.encode("ascii")
    if len(revision) > 64:
        raise PackFailure("model revision exceeds 64 ASCII bytes")
    revision = revision + bytes(65 - len(revision))
    return HEADER.pack(
        MAGIC,FORMAT_VERSION,HEADER_BYTES,ENTRY_BYTES,CODEC_ABI_VERSION,0,
        tensor_count,STAGE_COUNT,stage,stage * LAYERS_PER_STAGE,LAYERS_PER_STAGE,
        LAYER_COUNT,HIDDEN,VOCAB,EXPERTS,CODEC_BF16,codec,CODEC_BF16,0,0,
        align(HEADER_BYTES),file_bytes,revision,contract_sha,config_sha,recipe_sha,
    )


def pack_directory(records: list[Record]) -> bytes:
    return b"".join(ENTRY.pack(
        record.kind,record.layer,record.payload_type,record.codec,
        record.scale_encoding,record.groups,record.rows,record.columns,
        record.payload_offset,record.payload_bytes,record.scale_offset,record.scale_bytes,
    ) for record in records)


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise PackFailure("production stage packing requires an explicit working CUDA device")
    if len(args.contract_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.contract_sha256):
        raise PackFailure("contract sha256 must be 64 lowercase hexadecimal characters")
    source = TensorSource(args.model_dir)
    records = records_for_stage(args.stage_index,args.expert_codec)
    source.validate_names(name for record in records for name in source_names_for_record(record))
    config_bytes = source.config_path.read_bytes()
    config_sha = hashlib.sha256(config_bytes).digest()
    recipe = {
        "schema_version": 1,
        "format_version": FORMAT_VERSION,
        "model_revision": args.model_revision,
        "contract_sha256": args.contract_sha256,
        "stage_index": args.stage_index,
        "expert_codec": args.expert_codec,
        "non_expert_codec": "bf16",
        "kv_cache_codec": "bf16",
        "routed_expert_order": "up_gate",
        "dense_and_shared_order": "gate_up",
        "kv_b_transform": "heads_key_transposed_value_native",
        "nvfp4_global_scale_scope": "one_per_expert_matrix",
    }
    recipe_bytes = json.dumps(recipe,sort_keys=True,separators=(",",":")).encode("utf-8")
    recipe_sha = hashlib.sha256(recipe_bytes).digest()
    file_bytes = size_records(records,args.expert_codec)
    codec = CODECS[args.expert_codec][0]
    output = args.output
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True,exist_ok=True)
    with partial.open("w+b") as file:
        file.truncate(file_bytes)
        file.seek(0)
        file.write(pack_header(args.stage_index,codec,args.model_revision,bytes.fromhex(args.contract_sha256),config_sha,recipe_sha,len(records),file_bytes))
        file.seek(align(HEADER_BYTES))
        file.write(pack_directory(records))
        for record in records:
            print(json.dumps({"layer":record.layer,"tensor":KIND_NAMES[record.kind],"status":"packing"}),flush=True)
            if record.payload_type == PAYLOAD_PACKED_WEIGHT:
                write_experts(file,source,record,args.expert_codec,torch.device("cuda"))
            else:
                tensor = load_nonexpert(source,record)
                expected = record.groups * record.rows * record.columns
                if tensor.numel() != expected:
                    raise PackFailure(f"{KIND_NAMES[record.kind]} produced {tensor.numel()} elements, expected {expected}")
                write_tensor(file,record.payload_offset,tensor,torch.float32 if record.payload_type == PAYLOAD_F32 else torch.bfloat16)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial,output)
    output_sha = sha256_file(output)
    receipt = {
        **recipe,
        "source_config_sha256": config_sha.hex(),
        "pack_recipe_sha256": recipe_sha.hex(),
        "stage_pack_sha256": output_sha,
        "stage_pack_bytes": file_bytes,
        "tensor_count": len(records),
        "output": str(output.resolve()),
    }
    receipt_path = output.with_name(output.name + ".receipt.json")
    receipt_partial = receipt_path.with_name(receipt_path.name + ".partial")
    receipt_partial.write_text(json.dumps(receipt,indent=2,sort_keys=True) + "\n",encoding="utf-8")
    os.replace(receipt_partial,receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage-index",type=int,choices=range(STAGE_COUNT),required=True)
    parser.add_argument("--expert-codec",choices=tuple(CODECS),required=True)
    parser.add_argument("--model-revision",required=True)
    parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--device",choices=("cuda",),required=True)
    return parser.parse_args()


def main() -> int:
    try:
        receipt = build(parse_args())
    except (OSError,ValueError,RuntimeError,KeyError,json.JSONDecodeError) as error:
        print(f"glm52_stagepack: {error}",file=sys.stderr)
        return 1
    print(json.dumps(receipt,sort_keys=True),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
