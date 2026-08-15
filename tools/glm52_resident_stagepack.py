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
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open

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

CODEC_BF16 = 0
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


class PackFailure(RuntimeError):
    pass


def load_contract(repo_root: Path) -> Dict[str, Any]:
    return json.loads((repo_root / "model_contracts" / "glm52.json").read_text())


class Reader:
    def __init__(self, model_dir: Path):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise PackFailure(f"missing safetensors index: {index_path}")
        self.model_dir = model_dir
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.handles: Dict[str, Any] = {}

    def tensor(self, name: str) -> torch.Tensor:
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
        return handle.get_tensor(name)

    def close(self) -> None:
        self.handles.clear()


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


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
                 layer_range: Tuple[int, int]):
        self.spine = spine
        self.experts = experts
        self.c = contract
        self.layer_range = layer_range
        self.plan: List[PlanItem] = []

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
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_BF16, SCALE_NONE,
                      1, rows, cols)
        payload_bytes = rows * cols * 4
        blob = to_bytes(tensor)
        self.plan.append(PlanItem(entry, lambda: iter([blob]), None, payload_bytes, 0))

    def add_spine_bf16(self, kind: int, layer: int, name: str):
        t = self.spine.tensor(name)
        if t.dtype != torch.bfloat16:
            raise PackFailure(f"{name}: spine tensor must be BF16, got {t.dtype}")
        self.add_bf16(kind, layer, t)

    def add_experts(self, kind: int, layer: int, projections: List[str],
                    rows: int, columns: int):
        """Stack 256 experts' fp8 payloads (up then gate) + F32 scales, streamed."""
        per_expert_payload = len(projections) * rows * columns          # fp8 bytes
        per_expert_scale = len(projections) * rows * (columns // 128) * 4  # f32
        payload_bytes = EXPERT_COUNT * per_expert_payload
        scale_bytes = EXPERT_COUNT * per_expert_scale
        entry = Entry(kind, layer, PAYLOAD_PACKED_WEIGHT, CODEC_FP8, SCALE_F32,
                      EXPERT_COUNT, len(projections) * rows, columns)
        experts = self.experts

        def produce_payload() -> Iterator[bytes]:
            for expert in range(EXPERT_COUNT):
                for proj in projections:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                    w = experts.tensor(name + ".weight")
                    if w.shape != (rows, columns):
                        raise PackFailure(
                            f"{name}.weight shape {tuple(w.shape)} != ({rows}, {columns})")
                    yield to_bytes(w)

        def produce_scale() -> Iterator[bytes]:
            for expert in range(EXPERT_COUNT):
                for proj in projections:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                    s_inv = experts.tensor(name + ".weight_scale_inv")
                    expanded = s_inv.to(torch.float32).repeat_interleave(128, dim=0)
                    yield to_bytes(expanded)

        self.plan.append(PlanItem(entry, produce_payload, produce_scale,
                                  payload_bytes, scale_bytes))

    def add_kv_b(self, layer: int):
        w = self.spine.tensor(f"model.layers.{layer}.self_attn.kv_b_proj.weight")
        if w.dtype != torch.bfloat16:
            raise PackFailure(f"kv_b must be BF16, got {w.dtype}")
        heads = self.c["head_count"]
        qk_nope = self.c["qk_nope_head_dimension"]
        value_dim = self.c["value_head_dimension"]
        latent = self.c["latent_dimension"]
        expected_rows = heads * (qk_nope + value_dim)
        if w.shape != (expected_rows, latent):
            raise PackFailure(f"kv_b shape {tuple(w.shape)} != ({expected_rows}, {latent})")
        key_parts = []
        value_parts = []
        per_head = qk_nope + value_dim
        for head in range(heads):
            block = w[head * per_head:(head + 1) * per_head, :]
            key_parts.append(block[:qk_nope, :].t().contiguous())  # [latent, qk_nope]
            value_parts.append(block[qk_nope:, :])                  # [value, latent]
        key_t = torch.stack(key_parts, dim=0).contiguous()
        value = torch.stack(value_parts, dim=0).contiguous()
        self.add_bf16(K_KV_B_KEY_T, layer, key_t.reshape(heads * latent, qk_nope), groups=heads)
        self.add_bf16(K_KV_B_VALUE, layer, value.reshape(heads * value_dim, latent), groups=heads)

    def add_dense_gate_up(self, layer: int):
        gate = self.spine.tensor(f"model.layers.{layer}.mlp.gate_proj.weight")
        up = self.spine.tensor(f"model.layers.{layer}.mlp.up_proj.weight")
        self.add_bf16(K_DENSE_GATE_UP, layer, torch.cat([up, gate], dim=0))

    def add_shared_gate_up(self, layer: int):
        gate = self.spine.tensor(f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight")
        up = self.spine.tensor(f"model.layers.{layer}.mlp.shared_experts.up_proj.weight")
        self.add_bf16(K_SHARED_GATE_UP, layer, torch.cat([up, gate], dim=0))

    def has_full_indexer(self, layer: int) -> bool:
        share = self.c["dsa_index_share_group_layer_count"]
        return layer < 3 or (layer >= 6 and (layer - 6) % share == 0)

    # -- main plan ---------------------------------------------------------

    def build_plan(self):
        c = self.c
        first, last = self.layer_range
        first_routed = c["first_routed_layer"]

        self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER, "model.embed_tokens.weight")
        self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight")
        self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight")

        for layer in range(first, last + 1):
            attn = f"model.layers.{layer}.self_attn"
            self.add_spine_bf16(K_ATTN_NORM, layer,
                                f"model.layers.{layer}.input_layernorm.weight")
            self.add_spine_bf16(K_Q_A, layer, f"{attn}.q_a_proj.weight")
            self.add_spine_bf16(K_Q_A_NORM, layer, f"{attn}.q_a_layernorm.weight")
            self.add_spine_bf16(K_Q_B, layer, f"{attn}.q_b_proj.weight")
            self.add_spine_bf16(K_KV_A, layer, f"{attn}.kv_a_proj_with_mqa.weight")
            self.add_spine_bf16(K_KV_A_NORM, layer, f"{attn}.kv_a_layernorm.weight")
            self.add_kv_b(layer)
            self.add_spine_bf16(K_ATTN_OUTPUT, layer, f"{attn}.o_proj.weight")
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
                                    f"model.layers.{layer}.mlp.down_proj.weight")
            else:
                self.add_spine_bf16(K_ROUTER, layer,
                                    f"model.layers.{layer}.mlp.gate.weight")
                self.add_f32(K_ROUTER_CORRECTION, layer,
                             self.spine.tensor(
                                 f"model.layers.{layer}.mlp.gate.e_score_correction_bias"))
                self.add_experts(K_EXPERT_UP_GATE, layer, ["up_proj", "gate_proj"],
                                 c["moe_intermediate_dimension"], c["hidden_dimension"])
                self.add_experts(K_EXPERT_DOWN, layer, ["down_proj"],
                                 c["hidden_dimension"], c["moe_intermediate_dimension"])
                self.add_shared_gate_up(layer)
                self.add_spine_bf16(K_SHARED_DOWN, layer,
                                    f"model.layers.{layer}.mlp.shared_experts.down_proj.weight")

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
                0, 0, directory_offset, file_bytes,
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


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GLM-5.2 resident stage packs")
    parser.add_argument("--spine-dir", required=True,
                        help="BF16 master checkpoint (full-fidelity spine)")
    parser.add_argument("--expert-dir", required=True,
                        help="FP8 checkpoint (routed experts)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-range", default="0-77",
                        help="inclusive layer range, e.g. 0-77 or 0-2")
    parser.add_argument("--model-revision", default=SPINE_REVISION)
    parser.add_argument("--stage-count", type=int, default=1)
    parser.add_argument("--stage-index", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    contract = load_contract(repo_root)
    first, last = (int(v) for v in args.layer_range.split("-"))
    if last < first or last >= contract["layer_count"]:
        raise PackFailure(f"bad layer range {args.layer_range}")

    spine = Reader(Path(args.spine_dir))
    experts = Reader(Path(args.expert_dir))
    packer = Packer(spine, experts, contract, (first, last))
    packer.build_plan()

    contract_bytes = (repo_root / "model_contracts" / "glm52.json").read_bytes()
    source_config = json.dumps(
        {"layer_range": args.layer_range, "spine_dir": args.spine_dir,
         "expert_dir": args.expert_dir}, sort_keys=True)
    recipe = json.dumps({"tool": "glm52_resident_stagepack.py",
                         "codec": "fp8", "spine": "bf16-master",
                         "expert_revision_pending": True}, sort_keys=True)
    file_bytes = packer.write(
        Path(args.output), args.model_revision,
        sha256_bytes(contract_bytes), sha256_bytes(source_config.encode()),
        sha256_bytes(recipe.encode()), args.stage_count, args.stage_index,
        first, last - first + 1, contract["layer_count"])
    spine.close()
    experts.close()
    print(f"glm52sp written: {args.output} bytes={file_bytes} "
          f"layers={first}-{last} tensors={len(packer.plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
