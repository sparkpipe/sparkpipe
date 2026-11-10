#!/usr/bin/env python3
"""Build glm5_next (GLM 5.3 Flash) resident-decode stage packs (.g5nsp).

Real-checkpoint packer: reads /mnt/model-warm/glm-5.3-flash (FP8 e4m3
[128,128] checkpoint, 62 shards) through header-only safetensors memmaps
and emits one wire-format-v1 .g5nsp per TP rank. The tensor vocabulary is
the 49-kind table in modules/glm5_next_resident_decode_stage/source/
model-families/glm5_next/include/sparkpipe/spark_glm5_next_stagepack.h, and every checkpoint->pack transform
follows model-families/glm5_next/name_map.json:

  - the pack-V2 fusions: kda_qkv_beta = q|k|v|beta rows (the checkpoint's
    separate q_proj/k_proj/v_proj/b_proj), kda_decay_gate_down = f_a|g_a.
  - the kv_b split+per-head transpose (glm52's add_kv_b pattern).
  - f32 upcasts where the kernels read f32: hc fn/base/scale, kda o_norm,
    the compressor ape, dt_bias (F32 already), A_log (F32 already).
  - FP8 MLA/dense/shared projections dequantize to bf16 at pack time as
    bf16(e4m3 x f32 block scale). The release's block scales are not
    powers of two, so this is the FP8 grid plus a second bf16 rounding,
    not an exact copy of the FP8 values - the module's spine path is bf16;
    routed experts stay packaged fp8 payload + f32 scales.

Run on a spark node with warm ceph (per the fleet notes: NOT sparke - its
client holds a stale negative cache after the metadata incident).

Usage (per rank; or --tp-all for the 16-rank set in one process, sharing
the dequant cache):
  python3 tools/glm5_next_resident_stagepack.py \
      --source /mnt/model-warm/glm-5.3-flash --output-dir build/stagepacks \
      --tp-all 16

The expert codec is source-driven per layer: BF16 sources pass through
verbatim, FP8-native sources package fp8 payload + f32 scales, and nvfp4
releases (redhatai GLM-5.3-Flash-NVFP4: *_packed/*_scale/*_global_scale;
nvidia modelopt: U8 weight + *_scale + *_scale_2) package packed e2m1 +
UE4M3 planes + F32 globals verbatim - never quantized. The header's
expert_weight_codec is the single codec of the pack's routed-expert
entries; a pack whose layers carry different expert codecs is refused,
and a pack with no routed-expert entries needs --expert-codec.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import (CODEC_ABI_VERSION, CODEC_BF16, CODEC_FP8, CODEC_NONE, CODEC_NVFP4,  # noqa: E402,F401
                               PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT, PAYLOAD_U32, SCALE_F32, SCALE_NONE,
                               SCALE_UE4M3_F32_GLOBAL, f32_to_bf16, fp8_e4m3_lut)
from spark_pack_common import (GLOBAL_LAYER, PackFailure, check_pipeline_stage, publish_link,  # noqa: E402,F401
                               stage_pack_name as named_stage_pack, tp_shard_range)
from spark_pack_plan import MappedCheckpoint, PlanBuilder, PlanItem  # noqa: E402,F401
from spark_pack_wire import (CODEC_ENTRY_BYTES, CODEC_HEADER_BYTES, PAYLOAD_ALIGNMENT, Entry,  # noqa: E402,F401
                             codec_directory_offset, codec_header, emit_item, emit_region, place_entries,
                             serialize_entry, truncated_revision, write_prefix)

MAGIC = 0x33584C47
FORMAT_VERSION = 1
HEADER_BYTES = CODEC_HEADER_BYTES
ENTRY_BYTES = CODEC_ENTRY_BYTES
ALIGNMENT = PAYLOAD_ALIGNMENT
EXPERT_CODEC_NAMES = {"bf16": CODEC_BF16, "fp8": CODEC_FP8, "nvfp4": CODEC_NVFP4}

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
SPINE_DTYPES = ("BF16", "F8_E4M3", "U8")


def is_kda(layer: int) -> bool:
    return layer % 4 != 3


class SourceReader(MappedCheckpoint):
    def __init__(self, model_dir: Path, cache_byte_cap: int = 8 * 1024 ** 3):
        super().__init__(model_dir, cache_byte_cap)


class Packer(PlanBuilder):
    def __init__(self, source: SourceReader, tp_degree: int, tp_rank: int,
                 first_layer: int, layer_count: int, include_mtp: bool,
                 owns_embedding: bool, owns_head: bool,
                 expert_codec: Optional[int] = None):
        super().__init__(source, tp_degree, tp_rank)
        self.first_layer = first_layer
        self.layer_count = layer_count
        self.include_mtp = include_mtp
        self.owns_embedding = owns_embedding
        self.owns_head = owns_head
        self.expert_codec = expert_codec

    def add_spine_bf16(self, kind: int, layer: int, name: str, shard: str = "") -> PlanItem:
        return self.add_bf16(kind, layer, name, shard, dtypes=SPINE_DTYPES)

    def add_fused_sections(self, kind: int, layer: int, names: List[str], section_rows: List[int],
                           shard: bool = False) -> PlanItem:
        dtype = self.source.meta(names[0])[0]
        if dtype not in SPINE_DTYPES:
            raise PackFailure(f"{names[0]}: fused dtype {dtype}")
        return self.add_fused_rows(kind, layer, names, section_rows, HIDDEN, shard)

    def add_kda_conv(self, kind: int, layer: int, name: str) -> PlanItem:
        dtype, shape, _ = self.source.meta(name)
        if dtype not in ("BF16", "F32") or len(shape) != 3 or shape[1] != 1:
            raise PackFailure(f"{name}: conv shape {shape} dtype {dtype}")
        return self.add_bf16(kind, layer, name, "rows")

    def add_kv_b(self, layer: int) -> None:
        super().add_kv_b(K_KV_B_KEY_T, K_KV_B_VALUE, layer,
                         f"model.language_model.layers.{layer}.self_attn.kv_b_proj.weight",
                         MLA_HEADS, NOPE, VDIM, LATENT, dtypes=("BF16",))

    def add_up_gate_fused(self, kind: int, layer: int, up_name: str, gate_name: str) -> PlanItem:
        up_dtype, (up_rows, up_columns), _ = self.source.meta(up_name)
        gate_dtype, (gate_rows, gate_columns), _ = self.source.meta(gate_name)
        columns = up_columns * 2 if up_dtype == "U8" else up_columns
        if up_dtype != gate_dtype or (gate_columns * 2 if gate_dtype == "U8" else gate_columns) != columns:
            raise PackFailure(f"{up_name}|{gate_name}: fused section geometry mismatch")
        return self.add_fused_rows(kind, layer, [up_name, gate_name], [up_rows, gate_rows], columns, shard=True)

    def source_expert_codec(self, layer: int) -> int:
        name = f"model.language_model.layers.{layer}.mlp.experts.0.up_proj.weight"
        if name + "_packed" in self.source.weight_map:
            name += "_packed"
        elif name not in self.source.weight_map:
            raise PackFailure(f"layer {layer}: no routed expert 0 up_proj in the source")
        dtype = self.source.meta(name)[0]
        codec = {"BF16": CODEC_BF16, "F8_E4M3": CODEC_FP8, "U8": CODEC_NVFP4}.get(dtype)
        if codec is None:
            raise PackFailure(f"{name}: routed expert source dtype {dtype} has no pack codec")
        return codec

    def header_expert_codec(self) -> int:
        return header_expert_codec(self.plan, self.expert_codec)

    def add_experts(self, layer: int) -> None:
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        start, width = self.slice_range(EXPERT_INTER)
        codec = self.source_expert_codec(layer)
        up_gate = [[f"{prefix}.{expert}.{projection}_proj.weight" for projection in ("up", "gate")]
                   for expert in range(EXPERTS)]
        down = [[f"{prefix}.{expert}.down_proj.weight"] for expert in range(EXPERTS)]
        self.add_routed_experts(K_EXPERT_UP_GATE, layer, up_gate, EXPERT_INTER, HIDDEN, (start, start + width),
                                (0, HIDDEN), codec)
        self.add_routed_experts(K_EXPERT_DOWN, layer, down, HIDDEN, EXPERT_INTER, (0, HIDDEN),
                                (start, start + width), codec)

    def build(self) -> None:
        p = "model.language_model.layers"
        last_layer = self.first_layer + self.layer_count - (0 if self.include_mtp else 1)
        for layer in range(self.first_layer, last_layer + 1):
            a = f"{p}.{layer}.self_attn."
            m = f"{p}.{layer}.mlp."
            self.add_spine_bf16(K_ATTN_NORM, layer, f"{p}.{layer}.input_layernorm.weight")
            self.add_spine_bf16(K_POST_ATTN_NORM, layer, f"{p}.{layer}.post_attention_layernorm.weight")
            if layer < LAYERS and is_kda(layer):
                self.add_fused_sections(K_KDA_QKV_BETA, layer,
                                        [f"{a}q_proj.weight", f"{a}k_proj.weight", f"{a}v_proj.weight",
                                         f"{a}b_proj.weight"],
                                        [KDA_DIM, KDA_DIM, KDA_DIM, KDA_HEADS], shard=True)
                self.add_fused_sections(K_KDA_DECAY_GATE_DOWN, layer, [f"{a}f_a_proj.weight", f"{a}g_a_proj.weight"],
                                        [KDA_LOW_RANK, KDA_LOW_RANK])
                self.add_spine_bf16(K_KDA_DECAY_UP, layer, f"{a}f_b_proj.weight", shard="rows")
                self.add_spine_bf16(K_KDA_GATE_UP, layer, f"{a}g_b_proj.weight", shard="rows")
                self.add_kda_conv(K_KDA_Q_CONV, layer, f"{a}q_conv1d.weight")
                self.add_kda_conv(K_KDA_K_CONV, layer, f"{a}k_conv1d.weight")
                self.add_kda_conv(K_KDA_V_CONV, layer, f"{a}v_conv1d.weight")
                self.add_f32_vector(K_KDA_DECAY_BIAS, layer, f"{a}dt_bias", shard=True)
                self.add_f32_vector(K_KDA_HEAD_LOG_SCALE, layer, f"{a}A_log", shard=True)
                self.add_f32(K_KDA_OUT_NORM, layer, f"{a}o_norm.weight")
                self.add_spine_bf16(K_KDA_OUT, layer, f"{a}o_proj.weight", shard="cols")
            elif layer < LAYERS or layer == MTP_LAYER:
                self.add_spine_bf16(K_Q_A, layer, f"{a}q_a_proj.weight")
                self.add_spine_bf16(K_Q_A_NORM, layer, f"{a}q_a_layernorm.weight")
                self.add_spine_bf16(K_Q_B, layer, f"{a}q_b_proj.weight", shard="rows")
                self.add_spine_bf16(K_KV_A, layer, f"{a}kv_a_proj_with_mqa.weight")
                self.add_spine_bf16(K_KV_A_NORM, layer, f"{a}kv_a_layernorm.weight")
                self.add_kv_b(layer)
                self.add_spine_bf16(K_ATTN_OUTPUT, layer, f"{a}o_proj.weight", shard="cols")
                i = f"{a}indexer."
                self.add_spine_bf16(K_INDEX_Q, layer, f"{i}wq_b.weight")
                self.add_spine_bf16(K_INDEX_K, layer, f"{i}wk.weight")
                self.add_spine_bf16(K_INDEX_HEAD, layer, f"{i}weights_proj.weight")
                self.add_spine_bf16(K_INDEX_NORM_W, layer, f"{i}k_norm.weight")
                self.add_spine_bf16(K_INDEX_NORM_B, layer, f"{i}k_norm.bias")
                self.add_f32(K_INDEX_COMPRESS_APE, layer, f"{i}index_kpool_compress_ape")
                self.add_spine_bf16(K_INDEX_COMPRESS_GATE, layer, f"{i}index_kpool_compress_gate")
            if layer < LAYERS:
                for site, kind_fn, kind_base, kind_scale in (
                        ("attn", K_HC_ATTN_FN, K_HC_ATTN_BASE, K_HC_ATTN_SCALE),
                        ("ffn", K_HC_FFN_FN, K_HC_FFN_BASE, K_HC_FFN_SCALE)):
                    self.add_f32(kind_fn, layer, f"{p}.{layer}.hc_{site}_fn")
                    self.add_f32(kind_base, layer, f"{p}.{layer}.hc_{site}_base")
                    self.add_f32(kind_scale, layer, f"{p}.{layer}.hc_{site}_scale")
            if layer < FIRST_ROUTED:
                self.add_up_gate_fused(K_DENSE_GATE_UP, layer, f"{m}up_proj.weight", f"{m}gate_proj.weight")
                self.add_spine_bf16(K_DENSE_DOWN, layer, f"{m}down_proj.weight", shard="cols")
            else:
                self.add_spine_bf16(K_ROUTER, layer, f"{m}gate.weight")
                self.add_f32_vector(K_ROUTER_CORRECTION, layer, f"{m}gate.e_score_correction_bias")
                self.add_experts(layer)
                self.add_up_gate_fused(K_SHARED_GATE_UP, layer, f"{m}shared_experts.up_proj.weight",
                                       f"{m}shared_experts.gate_proj.weight")
                self.add_spine_bf16(K_SHARED_DOWN, layer, f"{m}shared_experts.down_proj.weight", shard="cols")
            if layer == MTP_LAYER:
                self.add_spine_bf16(K_MTP_EH_PROJ, layer, f"{p}.{layer}.eh_proj.weight")
                self.add_spine_bf16(K_MTP_ENORM, layer, f"{p}.{layer}.enorm.weight")
                self.add_spine_bf16(K_MTP_HNORM, layer, f"{p}.{layer}.hnorm.weight")
                self.add_spine_bf16(K_MTP_SHARED_NORM, layer, f"{p}.{layer}.shared_head.norm.weight")
        if self.owns_embedding:
            self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER, "model.language_model.embed_tokens.weight", shard="rows")
        if self.owns_head:
            self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.language_model.norm.weight")
            self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", shard="rows")


def header_expert_codec(plan: List[PlanItem], requested: Optional[int] = None) -> int:
    codecs = sorted({item.entry.weight_codec for item in plan
                     if item.entry.kind in (K_EXPERT_UP_GATE, K_EXPERT_DOWN)})
    if len(codecs) > 1:
        layers = sorted({(item.entry.layer, item.entry.weight_codec) for item in plan
                         if item.entry.kind in (K_EXPERT_UP_GATE, K_EXPERT_DOWN)})
        raise PackFailure(f"mixed routed-expert codecs {codecs} in one pack (layer, codec): {layers}")
    if not codecs:
        if requested is None:
            raise PackFailure("pack has no routed-expert entries; pass the header "
                              "expert codec explicitly (--expert-codec)")
        return requested
    if requested is not None and requested != codecs[0]:
        raise PackFailure(f"requested expert codec {requested} != source-driven routed-expert codec {codecs[0]}")
    return codecs[0]


def assemble_header(packer: Packer, header_extra: Dict[str, Any], file_bytes: int, revision: str,
                    contract_sha256: str) -> bytes:
    fields = (MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, CODEC_ABI_VERSION, header_extra.get("flags", 0),
              len(packer.plan), header_extra["stage_count"], header_extra["stage_index"],
              header_extra["first_layer"], header_extra["layer_count"], LAYERS, HIDDEN, VOCAB, EXPERTS,
              CODEC_BF16, packer.header_expert_codec(), CODEC_BF16, packer.tp_degree, packer.tp_rank)
    return codec_header(fields, header_extra["directory_offset"], file_bytes, truncated_revision(revision),
                        [bytes.fromhex(contract_sha256[:64].ljust(64, "0"))])


def plan_layout(packer: Packer, header_extra: Dict[str, Any], revision: str) -> Tuple[bytes, int, int]:
    directory_offset = codec_directory_offset()
    entries = [item.entry for item in packer.plan]
    file_bytes = place_entries(entries, directory_offset + len(entries) * ENTRY_BYTES)
    header = assemble_header(packer, dict(header_extra, directory_offset=directory_offset),
                             file_bytes, revision, CONTRACT_SHA256)
    return header, directory_offset, file_bytes


def write_pack_prefix(out, packer: Packer, header: bytes, directory_offset: int) -> None:
    write_prefix(out, header, directory_offset, (item.entry for item in packer.plan))


def _emit(packer: Packer, path: Path, header_extra: Dict[str, Any], revision: str = REVISION) -> int:
    packer.build()
    header, directory_offset, file_bytes = plan_layout(packer, header_extra, revision)
    with path.open("wb") as out:
        write_pack_prefix(out, packer, header, directory_offset)
        for item in packer.plan:
            emit_item(out, item)
    return file_bytes


def emit_ranks(packers: List[Packer], paths: List[Path], header_extra: Dict[str, Any],
               revision: str = REVISION, drop_output_cache: bool = False,
               sync_bytes: int = 2 << 30) -> List[int]:
    if len(packers) != len(paths) or not packers:
        raise PackFailure("emit_ranks needs one output path per packer")
    for path in paths:
        if path.exists():
            raise PackFailure(f"output already exists; choose a new artifact path: {path}")
    if drop_output_cache and not hasattr(os, "posix_fadvise"):
        raise PackFailure("page-cache drop requested but posix_fadvise is unavailable here")
    for packer in packers:
        packer.build()
    identity = [(item.entry.kind, item.entry.layer) for item in packers[0].plan]
    for packer in packers[1:]:
        if [(item.entry.kind, item.entry.layer) for item in packer.plan] != identity:
            raise PackFailure(f"rank {packer.tp_rank} plan order differs from rank {packers[0].tp_rank}")
    layouts = [plan_layout(packer, header_extra, revision) for packer in packers]
    temporaries: List[str] = []
    outs = []
    try:
        for path in paths:
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
            os.close(fd)
            temporaries.append(temporary)
        outs = [open(temporary, "r+b") for temporary in temporaries]
        for out, packer, (header, directory_offset, _) in zip(outs, packers, layouts):
            write_pack_prefix(out, packer, header, directory_offset)
        pending = 0
        for index in range(len(identity)):
            for out, packer in zip(outs, packers):
                item = packer.plan[index]
                emit_item(out, item)
                pending += item.entry.payload_bytes + item.entry.scale_bytes
            if drop_output_cache and pending >= sync_bytes:
                for out in outs:
                    out.flush()
                    os.fdatasync(out.fileno())
                    os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                pending = 0
        for out, (_, _, file_bytes) in zip(outs, layouts):
            out.truncate(file_bytes)
            out.flush()
            os.fsync(out.fileno())
            if drop_output_cache:
                os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            out.close()
        outs = []
        for temporary, path in zip(temporaries, paths):
            publish_link(temporary, path)
    finally:
        for out in outs:
            out.close()
        for temporary in temporaries:
            os.unlink(temporary)
    for packer, path, (_, _, file_bytes) in zip(packers, paths, layouts):
        print(f"{path.name}: {len(packer.plan)} tensors, {file_bytes} bytes "
              f"(tp{packer.tp_degree} rank {packer.tp_rank})")
    return [file_bytes for _, _, file_bytes in layouts]


def emit(packer: Packer, path: Path, header_extra: Dict[str, Any], revision: str = REVISION) -> None:
    if path.exists():
        raise PackFailure(f"output already exists; choose a new artifact path: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        file_bytes = _emit(packer, Path(temporary), header_extra, revision)
        publish_link(temporary, path)
    finally:
        os.unlink(temporary)
    print(f"{path.name}: {len(packer.plan)} tensors, {file_bytes} bytes "
          f"(tp{packer.tp_degree} rank {packer.tp_rank}, layers "
          f"{header_extra['first_layer']}..{header_extra['first_layer'] + header_extra['layer_count'] - 1})")


def validate_stage(stage_count, stage_index, first_layer, layer_count, owns_embedding, owns_head, mtp):
    check_pipeline_stage(stage_count, stage_index, first_layer, layer_count, owns_embedding, owns_head or mtp,
                         LAYERS)


def mtp_pack_name(tp_degree, tp_rank):
    return f"glm5_next_mtp.tp{tp_degree}.rank{tp_rank}.g5nsp"


def validate_mtp_only(args):
    if (args.mtp or args.owns_embedding or args.owns_head or args.first_layer != 0
            or args.layer_count != LAYERS or args.stage_count != 1 or args.stage_index != 0):
        raise PackFailure("--mtp-only builds the layer-45 MTP pack alone; it takes no layer span, stage or "
                          "ownership options")


def stage_pack_name(tp_degree, tp_rank, stage_count, stage_index):
    return named_stage_pack("glm5_next_stage", "g5nsp", tp_degree, tp_rank, stage_count, stage_index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkpoint directory (warm ceph)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=LAYERS)
    parser.add_argument("--stage-count", type=int, default=1)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--mtp-only", action="store_true",
                        help="emit only the layer-45 MTP pack per rank (glm5_next_mtp.tpN.rankR.g5nsp) for the "
                             "resident MTP drafter")
    parser.add_argument("--owns-embedding", action="store_true")
    parser.add_argument("--owns-head", action="store_true")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--tp-all", type=int, default=0,
                        help="emit all N rank packs in one pass: entry i of every rank before entry i+1, so each "
                             "source tensor is read once while it is cached")
    parser.add_argument("--model-revision", default=REVISION,
                        help=f"header model revision (default {REVISION}, the FP8 release)")
    parser.add_argument("--spine-cache-gib", type=float, default=8.0, help="dequantized spine tensor cache cap")
    parser.add_argument("--drop-output-cache", action="store_true",
                        help="--tp-all only: fdatasync and DONTNEED the outputs every 2 GiB")
    parser.add_argument("--expert-codec", choices=sorted(EXPERT_CODEC_NAMES), default=None,
                        help="header expert codec; required only for a pack without routed-expert entries, "
                             "otherwise it must equal the source-driven codec")
    parser.add_argument("--dry-plan", action="store_true", help="plan and print the inventory without writing")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if args.mtp_only:
        validate_mtp_only(args)
    else:
        validate_stage(args.stage_count, args.stage_index, args.first_layer, args.layer_count,
                       args.owns_embedding, args.owns_head, args.mtp)
    if args.drop_output_cache and not args.tp_all:
        parser.error("--drop-output-cache applies to --tp-all")
    source = SourceReader(Path(args.source), int(args.spine_cache_gib * 1024 ** 3))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    degree = args.tp_all or args.tp_degree
    ranks = range(args.tp_all) if args.tp_all else [args.tp_rank]
    requested = None if args.expert_codec is None else EXPERT_CODEC_NAMES[args.expert_codec]
    if args.mtp_only:
        packers = [Packer(source, degree, rank, MTP_LAYER, 0, True, False, False, requested) for rank in ranks]
        header = dict(stage_count=1, stage_index=0, first_layer=MTP_LAYER, layer_count=1, flags=1)
        paths = [out_dir / mtp_pack_name(degree, rank) for rank in ranks]
    else:
        packers = [Packer(source, degree, rank, args.first_layer, args.layer_count, args.mtp,
                          args.owns_embedding, args.owns_head, requested) for rank in ranks]
        header = dict(stage_count=args.stage_count, stage_index=args.stage_index, first_layer=args.first_layer,
                      layer_count=args.layer_count, flags=1 if args.mtp else 0)
        paths = [out_dir / stage_pack_name(degree, rank, args.stage_count, args.stage_index) for rank in ranks]
    if args.dry_plan:
        for packer, path in zip(packers, paths):
            packer.build()
            planned = sum(item.entry.payload_bytes + item.entry.scale_bytes for item in packer.plan)
            print(f"rank {packer.tp_rank}: {len(packer.plan)} tensors planned, {planned} payload bytes, "
                  f"header expert codec {packer.header_expert_codec()} -> {path.name}")
    elif args.tp_all:
        emit_ranks(packers, paths, header, args.model_revision, args.drop_output_cache)
    else:
        emit(packers[0], paths[0], header, args.model_revision)
    source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
