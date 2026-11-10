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
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import (CODEC_ABI_VERSION, CODEC_BF16, CODEC_FP8, CODEC_NONE, CODEC_NVFP4,  # noqa: E402,F401
                               FP8_BLOCK, NVFP4_GROUP, PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT, PAYLOAD_U32,
                               SCALE_F32, SCALE_NONE, SCALE_UE4M3_F32_GLOBAL, fp8_block_scale_rows)
from spark_pack_common import (GLOBAL_LAYER, PackFailure, check_pipeline_stage, publish_link,  # noqa: E402,F401
                               stage_pack_name as named_stage_pack)
from spark_pack_plan import MappedCheckpoint, PlanBuilder, PlanItem, to_bytes  # noqa: E402,F401
from spark_pack_wire import (CODEC_ENTRY_BYTES, CODEC_HEADER_BYTES, PAYLOAD_ALIGNMENT, Entry,  # noqa: E402,F401
                             codec_directory_offset, codec_header, emit_item, place_entries, serialize_entry,
                             truncated_revision, write_prefix)

FORMAT_VERSION = 1
HEADER_BYTES = CODEC_HEADER_BYTES
ENTRY_BYTES = CODEC_ENTRY_BYTES
ALIGNMENT = PAYLOAD_ALIGNMENT

K_EMBEDDING, K_FINAL_NORM, K_LM_HEAD = 0, 1, 2
K_ATTN_INPUT_NORM, K_ATTN_POST_NORM = 3, 4
K_FUSED_QKV, K_ATTN_OUTPUT, K_ATTN_GATE = 5, 6, 7
K_Q_NORM, K_K_NORM = 8, 9
K_DENSE_GATE_UP, K_DENSE_DOWN = 10, 11
K_ROUTER, K_ROUTER_CORRECTION = 12, 13
K_EXPERT_GATE_UP, K_EXPERT_DOWN = 14, 15
K_SHARED_GATE_UP, K_SHARED_DOWN = 16, 17

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
CONTRACT_PATH = "model_contracts/laguna_authoritative.json"

MAGIC = 0x334C4147
SPINE_DTYPES = ("BF16", "F8_E4M3")


def load_census(repo_root: Path, variant: str = "bf16"):
    patterns = json.loads((repo_root / "model-families/laguna/tensor_patterns.json").read_text())
    if variant == "bf16":
        if patterns["tensor_pattern_count"] != CENSUS_PATTERNS or patterns["tensor_count"] != CENSUS_TENSORS:
            raise PackFailure(f"tensor_patterns census {patterns['tensor_pattern_count']}/"
                              f"{patterns['tensor_count']} != locked 23/36769")
        return patterns["patterns"], patterns["tensor_count"]
    variants = patterns.get("variants", {})
    if variant not in variants:
        raise PackFailure(f"no {variant} census variant is pinned in tensor_patterns.json")
    entry = variants[variant]
    return entry["patterns"], entry["tensor_count"]


def pattern_regex(pattern: str) -> "re.Pattern":
    parts = [r"(\d+)" if token in ("{layer}", "{expert}") else re.escape(token)
             for token in re.split(r"(\{layer\}|\{expert\})", pattern)]
    return re.compile("^" + "".join(parts) + "$")


def census_lock(source: MappedCheckpoint, variant: str = "bf16"):
    census, total = load_census(Path(__file__).resolve().parents[1], variant)
    compiled = {pattern: pattern_regex(pattern) for pattern in census}
    counts = {pattern: 0 for pattern in census}
    for name in source.weight_map:
        for pattern, regex in compiled.items():
            if regex.match(name):
                counts[pattern] += 1
                break
        else:
            raise PackFailure(f"unknown checkpoint tensor (census lock): {name}")
    for pattern, expected in census.items():
        if counts[pattern] != expected["count"]:
            raise PackFailure(f"census lock: {pattern} count {counts[pattern]} != {expected['count']}")
    if len(source.weight_map) != total:
        raise PackFailure(f"census lock: {len(source.weight_map)} tensors, variant {variant} pins {total}")


class SourceReader(MappedCheckpoint):
    def __init__(self, model_dir: Path, cache_byte_cap: int = 8 * 1024 ** 3):
        super().__init__(model_dir, cache_byte_cap)


class Packer(PlanBuilder):
    def __init__(self, source: SourceReader, tp_degree: int, tp_rank: int,
                 first_layer: int, layer_count: int,
                 owns_embedding: bool, owns_head: bool, expert_codec: int = CODEC_BF16):
        super().__init__(source, tp_degree, tp_rank)
        self.first_layer = first_layer
        self.layer_count = layer_count
        self.owns_embedding = owns_embedding
        self.owns_head = owns_head
        self.expert_codec = expert_codec

    def add_spine_bf16(self, kind: int, layer: int, name: str, shard: str = "") -> PlanItem:
        return self.add_bf16(kind, layer, name, shard, dtypes=SPINE_DTYPES)

    def add_gate_up_fused(self, kind: int, layer: int, gate_name: str, up_name: str) -> PlanItem:
        gate_rows, columns = self.source.meta(gate_name)[1]
        up_rows = self.source.meta(up_name)[1][0]
        return self.add_fused_rows(kind, layer, [gate_name, up_name], [gate_rows, up_rows], columns, shard=True)

    def add_experts(self, layer: int) -> None:
        prefix = f"model.layers.{layer}.mlp.experts"
        start, width = self.slice_range(EXPERT_INTER)
        if self.expert_codec == CODEC_FP8:
            self._add_experts_fp8(layer, prefix, start, width)
            return
        if self.expert_codec == CODEC_NVFP4:
            self._add_experts_nvfp4(layer, prefix, start, width)
            return
        gate_up = [[f"{prefix}.{expert}.{projection}.weight" for projection in ("gate_proj", "up_proj")]
                   for expert in range(EXPERTS)]
        down = [[f"{prefix}.{expert}.down_proj.weight"] for expert in range(EXPERTS)]
        self.add_routed_experts(K_EXPERT_GATE_UP, layer, gate_up, EXPERT_INTER, HIDDEN, (start, start + width),
                                (0, HIDDEN), CODEC_BF16, bf16_payload_type=PAYLOAD_BF16)
        self.add_routed_experts(K_EXPERT_DOWN, layer, down, HIDDEN, EXPERT_INTER, (0, HIDDEN),
                                (start, start + width), CODEC_BF16, bf16_payload_type=PAYLOAD_BF16)

    def _check_expert_layer_codec(self, layer: int, prefix: str, dtype_want: str, packed: bool):
        for expert in (0, EXPERTS - 1):
            name = f"{prefix}.{expert}.gate_proj.weight" + ("_packed" if packed else "")
            if packed and name not in self.source.weight_map:
                raise PackFailure(
                    f"layer {layer} routed experts are not native to the nvfp4 arm: {name} is absent (this "
                    f"release quantizes a subset of the routed layers; the packer never quantizes and a mixed "
                    f"pack cannot load)")
            dtype = self.source.meta(name)[0]
            if dtype != dtype_want:
                raise PackFailure(
                    f"layer {layer} routed experts are not native to the {self.expert_codec} arm: {name} is "
                    f"{dtype} (this release quantizes a subset of the routed layers; the packer never quantizes "
                    f"and a mixed pack cannot load)")

    def _add_experts_fp8(self, layer: int, prefix: str, start: int, width: int):
        self._check_expert_layer_codec(layer, prefix, "F8_E4M3", packed=False)
        w1_blocks = (HIDDEN + FP8_BLOCK - 1) // FP8_BLOCK
        w2_blocks = (width + FP8_BLOCK - 1) // FP8_BLOCK
        w1 = Entry(K_EXPERT_GATE_UP, layer, PAYLOAD_PACKED_WEIGHT, CODEC_FP8, SCALE_F32, EXPERTS, 2 * width, HIDDEN)
        w2 = Entry(K_EXPERT_DOWN, layer, PAYLOAD_PACKED_WEIGHT, CODEC_FP8, SCALE_F32, EXPERTS, HIDDEN, width)
        source = self.source

        def plane(name: str) -> np.ndarray:
            return source.raw(name).view(np.float32).reshape(source.meta(name)[1])

        def produce_w1():
            for expert in range(EXPERTS):
                for proj in ("gate_proj", "up_proj"):
                    codes = source.raw(f"{prefix}.{expert}.{proj}.weight").reshape(EXPERT_INTER, HIDDEN)
                    yield to_bytes(codes[start:start + width, :])

        def produce_w1_scale():
            for expert in range(EXPERTS):
                yield to_bytes(np.concatenate(
                    [fp8_block_scale_rows(plane(f"{prefix}.{expert}.{proj}.weight_scale_inv"), start, width, 0,
                                          w1_blocks) for proj in ("gate_proj", "up_proj")], axis=0))

        def produce_w2():
            for expert in range(EXPERTS):
                codes = source.raw(f"{prefix}.{expert}.down_proj.weight").reshape(HIDDEN, EXPERT_INTER)
                yield to_bytes(codes[:, start:start + width])

        def produce_w2_scale():
            for expert in range(EXPERTS):
                yield to_bytes(fp8_block_scale_rows(plane(f"{prefix}.{expert}.down_proj.weight_scale_inv"), 0,
                                                    HIDDEN, start // FP8_BLOCK, w2_blocks))

        self.add_item(w1, produce_w1, produce_w1_scale, EXPERTS * 2 * width * HIDDEN,
                      EXPERTS * 2 * width * w1_blocks * 4)
        self.add_item(w2, produce_w2, produce_w2_scale, EXPERTS * HIDDEN * width, EXPERTS * HIDDEN * w2_blocks * 4)

    def _add_experts_nvfp4(self, layer: int, prefix: str, start: int, width: int):
        self._check_expert_layer_codec(layer, prefix, "U8", packed=True)
        w1_blocks = HIDDEN // NVFP4_GROUP
        w2_blocks = width // NVFP4_GROUP
        w1 = Entry(K_EXPERT_GATE_UP, layer, PAYLOAD_PACKED_WEIGHT, CODEC_NVFP4, SCALE_UE4M3_F32_GLOBAL, EXPERTS,
                   2 * width, HIDDEN)
        w2 = Entry(K_EXPERT_DOWN, layer, PAYLOAD_PACKED_WEIGHT, CODEC_NVFP4, SCALE_UE4M3_F32_GLOBAL, EXPERTS,
                   HIDDEN, width)
        source = self.source

        def produce_w1():
            for expert in range(EXPERTS):
                for proj in ("gate_proj", "up_proj"):
                    codes = source.raw(f"{prefix}.{expert}.{proj}.weight_packed").reshape(EXPERT_INTER, HIDDEN // 2)
                    yield to_bytes(codes[start:start + width, :])

        def produce_w1_scale():
            for expert in range(EXPERTS):
                for proj in ("gate_proj", "up_proj"):
                    name = f"{prefix}.{expert}.{proj}.weight_scale"
                    yield to_bytes(source.raw(name).reshape(source.meta(name)[1])[start:start + width, :])
                    yield to_bytes(source.raw(f"{prefix}.{expert}.{proj}.weight_global_scale").view(np.float32))

        def produce_w2():
            for expert in range(EXPERTS):
                codes = source.raw(f"{prefix}.{expert}.down_proj.weight_packed").reshape(HIDDEN, EXPERT_INTER // 2)
                yield to_bytes(codes[:, start:start + width])

        def produce_w2_scale():
            for expert in range(EXPERTS):
                name = f"{prefix}.{expert}.down_proj.weight_scale"
                plane = source.raw(name).reshape(source.meta(name)[1])
                yield to_bytes(plane[:, start // NVFP4_GROUP:start // NVFP4_GROUP + w2_blocks])
                yield to_bytes(source.raw(f"{prefix}.{expert}.down_proj.weight_global_scale").view(np.float32))

        self.add_item(w1, produce_w1, produce_w1_scale, EXPERTS * 2 * width * HIDDEN // 2,
                      EXPERTS * (2 * width * w1_blocks + 4))
        self.add_item(w2, produce_w2, produce_w2_scale, EXPERTS * HIDDEN * width // 2,
                      EXPERTS * (HIDDEN * w2_blocks + 4))

    def receipt(self) -> dict:
        return {
            "tp_degree": self.tp_degree,
            "tp_rank": self.tp_rank,
            "first_layer": self.first_layer,
            "layer_count": self.layer_count,
            "tensors": len(self.plan),
            "payload_bytes": sum(item.entry.payload_bytes for item in self.plan),
            "sha_feed": [f"{item.entry.kind}:{item.entry.layer}:{item.entry.payload_bytes}" for item in self.plan],
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
            self.add_fused_rows(K_FUSED_QKV, layer, [f"{a}q_proj.weight", f"{a}k_proj.weight", f"{a}v_proj.weight"],
                                [heads * HEAD_DIM, kv_rows, kv_rows], HIDDEN, shard=True)
            self.add_spine_bf16(K_ATTN_OUTPUT, layer, f"{a}o_proj.weight", shard="cols")
            self.add_spine_bf16(K_ATTN_GATE, layer, f"{a}g_proj.weight", shard="rows")
            self.add_spine_bf16(K_Q_NORM, layer, f"{a}q_norm.weight")
            self.add_spine_bf16(K_K_NORM, layer, f"{a}k_norm.weight")
            if layer < FIRST_ROUTED:
                self.add_gate_up_fused(K_DENSE_GATE_UP, layer, f"{m}gate_proj.weight", f"{m}up_proj.weight")
                self.add_spine_bf16(K_DENSE_DOWN, layer, f"{m}down_proj.weight", shard="cols")
                continue
            self.add_spine_bf16(K_ROUTER, layer, f"{m}gate.weight")
            bias_name = f"{m}experts.e_score_correction_bias"
            dtype = self.source.meta(bias_name)[0]
            if dtype not in ("BF16", "F32"):
                raise PackFailure(f"{bias_name}: dtype {dtype}, expected BF16 or F32")
            self.add_f32_vector(K_ROUTER_CORRECTION, layer, bias_name)
            self.add_experts(layer)
            self.add_gate_up_fused(K_SHARED_GATE_UP, layer, f"{m}shared_expert.gate_proj.weight",
                                   f"{m}shared_expert.up_proj.weight")
            self.add_spine_bf16(K_SHARED_DOWN, layer, f"{m}shared_expert.down_proj.weight", shard="cols")
        if self.owns_embedding:
            self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER, "model.embed_tokens.weight", shard="rows")
        if self.owns_head:
            self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight")
            self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", shard="rows")


def assemble_header(packer: Packer, header_extra: Dict[str, Any], file_bytes: int, revision: str,
                    contract_sha256: str) -> bytes:
    fields = (MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, CODEC_ABI_VERSION, header_extra.get("flags", 0),
              len(packer.plan), header_extra["stage_count"], header_extra["stage_index"],
              header_extra["first_layer"], header_extra["layer_count"], LAYERS, HIDDEN, VOCAB, EXPERTS,
              CODEC_BF16, packer.expert_codec, CODEC_BF16, packer.tp_degree, packer.tp_rank)
    return codec_header(fields, header_extra["directory_offset"], file_bytes, truncated_revision(revision),
                        [bytes.fromhex(contract_sha256[:64].ljust(64, "0"))])


def _emit(packer: Packer, path: Path, header_extra: Dict[str, Any], revision: str, contract_sha256: str) -> int:
    packer.build()
    directory_offset = codec_directory_offset()
    entries = [item.entry for item in packer.plan]
    file_bytes = place_entries(entries, directory_offset + len(entries) * ENTRY_BYTES)
    header = assemble_header(packer, dict(header_extra, directory_offset=directory_offset), file_bytes, revision,
                             contract_sha256)
    with path.open("wb") as out:
        write_prefix(out, header, directory_offset, entries)
        for item in packer.plan:
            emit_item(out, item)
    return file_bytes


def emit(packer: Packer, path: Path, header_extra: Dict[str, Any], revision: str, contract_sha256: str) -> None:
    if path.exists():
        raise PackFailure(f"output already exists; choose a new artifact path: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        file_bytes = _emit(packer, Path(temporary), header_extra, revision, contract_sha256)
        publish_link(temporary, path)
    finally:
        os.unlink(temporary)
    print(f"{path.name}: {len(packer.plan)} tensors, {file_bytes} bytes "
          f"(tp{packer.tp_degree} rank {packer.tp_rank}, layers "
          f"{header_extra['first_layer']}..{header_extra['first_layer'] + header_extra['layer_count'] - 1})")


def validate_stage(stage_count, stage_index, first_layer, layer_count, owns_embedding, owns_head):
    check_pipeline_stage(stage_count, stage_index, first_layer, layer_count, owns_embedding, owns_head, LAYERS)


def stage_pack_name(tp_degree, tp_rank, stage_count, stage_index):
    return named_stage_pack("laguna_stage", "lgsp", tp_degree, tp_rank, stage_count, stage_index)


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
    parser.add_argument("--expert-codec", default="bf16", choices=["bf16", "fp8", "nvfp4"],
                        help="routed-expert wire codec; fp8/nvfp4 select the matching census variant and fail "
                             "closed unless EVERY routed layer is native to the codec")
    parser.add_argument("--revision", required=True,
                        help="model revision stamped into the pack header (the warm snapshot's HF tree id)")
    parser.add_argument("--contract-sha256", required=True,
                        help="sha256 of model_contracts/laguna_authoritative.json, stamped into the pack header")
    parser.add_argument("--dry-plan", action="store_true", help="plan and print the inventory without writing")
    args = parser.parse_args()
    validate_stage(args.stage_count, args.stage_index, args.first_layer, args.layer_count,
                   args.owns_embedding, args.owns_head)
    if args.stage_count not in (1, 2, 4) or args.tp_degree not in (4, 8):
        raise PackFailure("supported fleet shapes: tp8/pp2 (default), tp4/pp4, single-stage")
    source = SourceReader(Path(args.source))
    expert_codec = {"bf16": CODEC_BF16, "fp8": CODEC_FP8, "nvfp4": CODEC_NVFP4}[args.expert_codec]
    census_lock(source, variant=args.expert_codec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.tp_all) if args.tp_all else [args.tp_rank]:
        packer = Packer(source, args.tp_all or args.tp_degree, rank, args.first_layer, args.layer_count,
                        args.owns_embedding, args.owns_head, expert_codec=expert_codec)
        if args.dry_plan:
            packer.build()
            receipt = packer.receipt()
            print(f"rank {rank}: {receipt['tensors']} tensors, {receipt['payload_bytes']} payload bytes")
            continue
        emit(packer, out_dir / stage_pack_name(args.tp_all or args.tp_degree, rank, args.stage_count,
                                               args.stage_index),
             dict(stage_count=args.stage_count, stage_index=args.stage_index, first_layer=args.first_layer,
                  layer_count=args.layer_count, flags=0),
             args.revision, args.contract_sha256)
        print("receipt " + json.dumps(packer.receipt(), sort_keys=True))
    source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
