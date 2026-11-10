#!/usr/bin/env python3
"""Build GLM-5.2 resident-decode stage packs (v3 .glm52sp).

Wire format: model-families/glm52/include/sparkpipe/spark_glm52_stagepack.h
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

NVFP4 sources (--expert-codec nvfp4, 2026-08-29 glm53full lane): the
GLM 5.3-full radixark checkpoint (RadixArk/GLM-5.3-NVFP4, pinned by
model_contracts/glm53_full_authoritative.json) stores routed experts
ONLY as NVFP4 — U8 packed e2m1 [rows, cols/2], F8_E4M3 scales
[rows, cols/16], one F32 weight_scale_2 global per projection — over an
all-BF16 spine. The packer copies every expert byte verbatim (codec 6:
payload, per-16 UE4M3 block scales, one F32 global per expert; the
contract freeze proved up/gate share the expert's global, so the fused
EXPERT_UP_GATE entry is exact) and passes the BF16 spine through. No
quantization happens here; geometry is unchanged (glm52 module family).

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
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import (CODEC_ABI_VERSION, CODEC_BF16, CODEC_FP8, CODEC_NONE, CODEC_NVFP4,  # noqa: E402,F401
                               PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT, PAYLOAD_U32, SCALE_F32, SCALE_NONE,
                               SCALE_UE4M3_F32_GLOBAL, bf16_to_f32, f32_to_bf16, fp8_e4m3_lut)
from spark_pack_common import GLOBAL_LAYER, PackFailure, align_up, run_tool, sha256_bytes, write_receipt  # noqa: E402,F401
from spark_pack_plan import MappedCheckpoint, PlanBuilder, PlanItem  # noqa: E402,F401
from spark_pack_wire import (CODEC_ENTRY_BYTES, CODEC_HEADER_BYTES, MODEL_REVISION_BYTES, PAYLOAD_ALIGNMENT,  # noqa: E402,F401
                             Entry, codec_directory_offset, codec_header, emit_item, place_entries, truncated_revision,
                             write_prefix)

MAGIC = 0x32534C47
FORMAT_VERSION = 3
HEADER_BYTES = CODEC_HEADER_BYTES
ENTRY_BYTES = CODEC_ENTRY_BYTES
ALIGNMENT = PAYLOAD_ALIGNMENT

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

EXPERT_CODECS = {"fp8": CODEC_FP8, "nvfp4": CODEC_NVFP4, "bf16": CODEC_BF16}
EXPERT_POLICIES = {
    CODEC_FP8: "fp8 payload bytes verbatim + F32 dequant scales",
    CODEC_NVFP4: ("nvfp4 VERBATIM passthrough (packed e2m1 bytes, "
                  "UE4M3 per-16 block scales, one F32 global per "
                  "expert) — the packer never quantizes; source is "
                  "an already-NVFP4 publisher/community release"),
    CODEC_BF16: ("bf16 VERBATIM passthrough (native publisher "
                 "precision, no scale plane) — the quant policy's "
                 "serve-the-native-precision arm; expert bytes "
                 "copied exactly as released"),
}


def load_contract(repo_root: Path) -> Dict[str, Any]:
    return json.loads((repo_root / "model_contracts" / "glm52.json").read_text())


def f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    return f32_to_bf16(values, quiet_nan=True)


class Fp8SourceReader(MappedCheckpoint):
    spine_dtypes = ("BF16", "F8_E4M3")
    quiet_nan = True

    def __init__(self, model_dir: Path, cache_byte_cap: int = 6 * 1024 ** 3):
        super().__init__(model_dir, cache_byte_cap)

    def map_name(self, name: str) -> str:
        if name in self.weight_map or not name.startswith("model."):
            return name
        alternative = "model.language_model." + name[len("model."):]
        return alternative if alternative in self.weight_map else name

    def scale_name(self, name: str) -> str:
        return name + "_scale_inv"

    def expert_payload(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        dtype, shape, _ = self.meta(name)
        if dtype != "F8_E4M3" or len(shape) != 2:
            raise PackFailure(f"{name}: expected 2-D F8_E4M3, got {dtype} {shape}")
        return self.window(name, np.uint8, r0, r1, c0, c1)

    def bf16_expert_payload(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        dtype, shape, _ = self.meta(name)
        if dtype != "BF16" or len(shape) != 2:
            raise PackFailure(f"{name}: expected 2-D BF16, got {dtype} {shape}")
        return self.window(name, np.uint16, r0, r1, c0, c1)


class Packer(PlanBuilder):
    def __init__(self, source: Fp8SourceReader, contract: Dict[str, Any],
                 layer_range: Tuple[int, int], include_embedding: bool = True,
                 include_head: bool = True, tp_degree: int = 1, tp_rank: int = 0,
                 expert_codec: int = CODEC_FP8):
        super().__init__(source, tp_degree, tp_rank)
        self.c = contract
        self.layer_range = layer_range
        self.include_embedding = include_embedding
        self.include_head = include_head
        self.expert_codec = expert_codec

    def add_spine_bf16(self, kind: int, layer: int, name: str, shard: str = "") -> PlanItem:
        return self.add_bf16(kind, layer, name, shard)

    def add_f32(self, kind: int, layer: int, name: str, dtypes=("F32",)) -> PlanItem:
        return super().add_f32(kind, layer, name, dtypes)

    def add_experts(self, kind: int, layer: int, projections: List[str], rows: int, columns: int,
                    shard: str = "") -> PlanItem:
        r0, r1, c0, c1 = 0, rows, 0, columns
        if shard == "rows" and self.tp_degree > 1:
            if rows % self.tp_degree != 0:
                raise PackFailure(f"expert rows {rows} not divisible by tp {self.tp_degree}")
            r0 = self.tp_rank * (rows // self.tp_degree)
            r1 = r0 + rows // self.tp_degree
        elif shard == "cols" and self.tp_degree > 1:
            if columns % self.tp_degree != 0:
                raise PackFailure(f"expert cols {columns} not divisible by tp {self.tp_degree}")
            c0 = self.tp_rank * (columns // self.tp_degree)
            c1 = c0 + columns // self.tp_degree
        names = [[f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight" for proj in projections]
                 for expert in range(EXPERT_COUNT)]
        return self.add_routed_experts(kind, layer, names, rows, columns, (r0, r1), (c0, c1), self.expert_codec,
                                       bf16_payload_type=PAYLOAD_BF16, check_shapes=True,
                                       payload_method="bf16_expert_payload" if self.expert_codec == CODEC_BF16
                                       else "expert_payload")

    def add_kv_b(self, layer: int) -> None:
        super().add_kv_b(K_KV_B_KEY_T, K_KV_B_VALUE, layer, f"model.layers.{layer}.self_attn.kv_b_proj.weight",
                         self.c["head_count"], self.c["qk_nope_head_dimension"], self.c["value_head_dimension"],
                         self.c["latent_dimension"])

    def add_fused_gate_up(self, kind: int, layer: int, gate_name: str, up_name: str) -> PlanItem:
        _, gate_shape, _ = self.source.meta(gate_name)
        _, up_shape, _ = self.source.meta(up_name)
        if tuple(gate_shape) != tuple(up_shape):
            raise PackFailure(f"{up_name} vs {gate_name} shape drift")
        rows, columns = up_shape
        return self.add_fused_rows(kind, layer, [up_name, gate_name], [rows, rows], columns, shard=True,
                                   sources=[gate_name, up_name])

    def has_full_indexer(self, layer: int) -> bool:
        share = self.c["dsa_index_share_group_layer_count"]
        return layer < 3 or (layer >= 6 and (layer - 6) % share == 0)

    def build_plan(self):
        c = self.c
        first, last = self.layer_range
        if self.include_embedding:
            self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER, "model.embed_tokens.weight", shard="rows")
        if self.include_head:
            self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight")
            self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", shard="rows")
        for layer in range(first, last + 1):
            attn = f"model.layers.{layer}.self_attn"
            mlp = f"model.layers.{layer}.mlp"
            self.add_spine_bf16(K_ATTN_NORM, layer, f"model.layers.{layer}.input_layernorm.weight")
            self.add_spine_bf16(K_Q_A, layer, f"{attn}.q_a_proj.weight")
            self.add_spine_bf16(K_Q_A_NORM, layer, f"{attn}.q_a_layernorm.weight")
            self.add_spine_bf16(K_Q_B, layer, f"{attn}.q_b_proj.weight", shard="rows")
            self.add_spine_bf16(K_KV_A, layer, f"{attn}.kv_a_proj_with_mqa.weight")
            self.add_spine_bf16(K_KV_A_NORM, layer, f"{attn}.kv_a_layernorm.weight")
            self.add_kv_b(layer)
            self.add_spine_bf16(K_ATTN_OUTPUT, layer, f"{attn}.o_proj.weight", shard="cols")
            self.add_spine_bf16(K_POST_ATTN_NORM, layer, f"model.layers.{layer}.post_attention_layernorm.weight")
            if self.has_full_indexer(layer):
                self.add_spine_bf16(K_INDEX_Q, layer, f"{attn}.indexer.wq_b.weight")
                self.add_spine_bf16(K_INDEX_K, layer, f"{attn}.indexer.wk.weight")
                self.add_spine_bf16(K_INDEX_HEAD, layer, f"{attn}.indexer.weights_proj.weight")
                self.add_spine_bf16(K_INDEX_NORM_W, layer, f"{attn}.indexer.k_norm.weight")
                self.add_spine_bf16(K_INDEX_NORM_B, layer, f"{attn}.indexer.k_norm.bias")
            if layer < c["first_routed_layer"]:
                self.add_fused_gate_up(K_DENSE_GATE_UP, layer, f"{mlp}.gate_proj.weight", f"{mlp}.up_proj.weight")
                self.add_spine_bf16(K_DENSE_DOWN, layer, f"{mlp}.down_proj.weight", shard="cols")
                continue
            self.add_spine_bf16(K_ROUTER, layer, f"{mlp}.gate.weight")
            self.add_f32(K_ROUTER_CORRECTION, layer, f"{mlp}.gate.e_score_correction_bias")
            self.add_experts(K_EXPERT_UP_GATE, layer, ["up_proj", "gate_proj"],
                             c["moe_intermediate_dimension"], c["hidden_dimension"], shard="rows")
            self.add_experts(K_EXPERT_DOWN, layer, ["down_proj"],
                             c["hidden_dimension"], c["moe_intermediate_dimension"], shard="cols")
            self.add_fused_gate_up(K_SHARED_GATE_UP, layer, f"{mlp}.shared_experts.gate_proj.weight",
                                   f"{mlp}.shared_experts.up_proj.weight")
            self.add_spine_bf16(K_SHARED_DOWN, layer, f"{mlp}.shared_experts.down_proj.weight", shard="cols")

    def layout(self):
        entries = [item.entry for item in self.plan]
        start = align_up(codec_directory_offset() + len(entries) * ENTRY_BYTES, ALIGNMENT)
        return entries, place_entries(entries, start, align_payloads=False, scales_last=True)

    def header(self, file_bytes: int, model_revision: str, digests, stage_count: int, stage_index: int,
               first_layer_index: int, layer_count: int, total_layer_count: int) -> bytes:
        fields = (MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, CODEC_ABI_VERSION, 0, len(self.plan),
                  stage_count, stage_index, first_layer_index, layer_count, total_layer_count,
                  self.c["hidden_dimension"], self.c["output_vocab_count"], self.c["moe_expert_count"],
                  CODEC_BF16, self.expert_codec, CODEC_BF16, self.tp_degree, self.tp_rank)
        return codec_header(fields, codec_directory_offset(), file_bytes,
                            model_revision.encode("utf-8")[:MODEL_REVISION_BYTES - 1], digests)


def build_receipt(packer: Packer, output: Path, file_bytes: int, model_revision: str, recipe: Dict[str, Any],
                  first: int, last: int) -> Path:
    source = packer.source
    receipt = {
        "tool": "glm52_resident_stagepack.py",
        "wire_format": {"magic": MAGIC, "version": FORMAT_VERSION,
                        "header_bytes": HEADER_BYTES, "entry_bytes": ENTRY_BYTES},
        "model_revision": model_revision,
        "recipe": recipe,
        "expert_codec": {CODEC_FP8: "fp8", CODEC_NVFP4: "nvfp4", CODEC_BF16: "bf16"}[packer.expert_codec],
        "expert_policy": EXPERT_POLICIES[packer.expert_codec],
        "source": {
            "directory": str(source.root),
            "config_sha256": source.config_sha256,
            "index_sha256": source.index_sha256,
            "architecture": source.config.get("architectures", [None])[0],
            "digest_contract": recipe["expert_source_contract"],
            "spine_policy": ("bf16 tensors byte-exact; F8_E4M3 spine tensors "
                             "dequantized bf16(f32(code)*scale_inv) on 128x128 "
                             "blocks (nvfp4/bf16 sources carry an all-BF16 spine: "
                             "pure passthrough)"),
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


def write_ranks(packers: List[Packer], output_paths: List[Path], model_revision: str, digests,
                stage: Tuple[int, int, int, int, int]) -> List[int]:
    prepared = []
    for packer, path in zip(packers, output_paths):
        entries, file_bytes = packer.layout()
        out = path.open("wb")
        write_prefix(out, packer.header(file_bytes, model_revision, digests, *stage), codec_directory_offset(),
                     entries)
        prepared.append((packer, out, file_bytes))
    try:
        for index in range(len(packers[0].plan)):
            for packer, out, _ in prepared:
                emit_item(out, packer.plan[index])
    finally:
        for _, out, _ in prepared:
            out.close()
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
    parser.add_argument("--expert-codec", choices=("fp8", "nvfp4", "bf16"), default="fp8",
                        help="fp8: blockwise-FP8 checkpoint (5.2 official / 5.3-full official FP8), fp8 expert "
                             "bytes verbatim; nvfp4: NVFP4 checkpoint (5.3-full radixark), e2m1+scale bytes "
                             "verbatim; bf16: native BF16 checkpoint (5.3-full official BF16), expert bytes "
                             "verbatim, no scale plane - never a requant")
    parser.add_argument("--source-contract", default=None,
                        help="digest_contract recorded in the receipt (default: glm53_full_authoritative.json "
                             "for nvfp4, glm52_authoritative.json otherwise); 5.3-full fp8/bf16 builds pass "
                             "model_contracts/glm53_full_authoritative.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    contract = load_contract(repo_root)
    expert_codec = EXPERT_CODECS[args.expert_codec]
    if args.tp_degree < 1 or args.tp_rank < 0 or args.tp_rank >= args.tp_degree:
        raise PackFailure(f"invalid tp rank {args.tp_rank}/{args.tp_degree}")
    stage_count, stage_index = args.stage_count, args.stage_index
    include_embedding = include_head = True
    if args.stage >= 0:
        if args.stage >= 13:
            raise PackFailure(f"--stage must be 0-12, got {args.stage}")
        first, last = args.stage * 6, args.stage * 6 + 5
        stage_count, stage_index = 13, args.stage
        include_embedding, include_head = args.stage == 0, args.stage == 12
    else:
        first, last = (int(value) for value in args.layer_range.split("-"))
        if last < first or last >= contract["layer_count"]:
            raise PackFailure(f"bad layer range {args.layer_range}")

    source = Fp8SourceReader(Path(args.source))
    firmware_name = f"glm52_resident_decode_stage_{args.expert_codec}_firmware.json"
    contract_bytes = (repo_root / "examples" / "model_descriptions" / firmware_name).read_bytes()
    source_contract = args.source_contract or (
        "model_contracts/glm53_full_authoritative.json" if expert_codec == CODEC_NVFP4
        else "model_contracts/glm52_authoritative.json")
    recipe = json.dumps({"tool": "glm52_resident_stagepack.py", "codec": args.expert_codec,
                         "spine": "fp8-dequant-128x128-blockwise" if expert_codec == CODEC_FP8 else "bf16-passthrough",
                         "expert_source_contract": source_contract}, sort_keys=True)
    source_config = json.dumps({"layer_range": f"{first}-{last}", "stage": args.stage,
                                "tp_degree": args.tp_degree, "tp_rank": "all" if args.all_ranks else args.tp_rank,
                                "source_dir": args.source}, sort_keys=True)
    digests = (sha256_bytes(contract_bytes), sha256_bytes(source_config.encode()), sha256_bytes(recipe.encode()))
    stage = (stage_count, stage_index, first, last - first + 1, contract["layer_count"])
    ranks = range(args.tp_degree) if args.all_ranks else [args.tp_rank]
    packers = []
    for rank in ranks:
        packer = Packer(source, contract, (first, last), include_embedding, include_head, args.tp_degree, rank,
                        expert_codec)
        packer.build_plan()
        packers.append(packer)
    paths = [Path(args.output.format(rank=rank)) if args.all_ranks else Path(args.output) for rank in ranks]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    sizes = write_ranks(packers, paths, args.model_revision, digests, stage)
    source.close()
    for packer, path, size in zip(packers, paths, sizes):
        build_receipt(packer, path, size, args.model_revision, json.loads(recipe), first, last)
        print(f"glm52sp written: {path} bytes={size} stage={stage_index}/{stage_count} layers={first}-{last} "
              f"tensors={len(packer.plan)} tp={args.tp_degree}")
    return 0


if __name__ == "__main__":
    run_tool(main, "glm52_resident_stagepack")
