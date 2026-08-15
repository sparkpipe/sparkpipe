#!/usr/bin/env python3
"""Build DeepSeek V4 Pro stage packs from a safetensors checkpoint.

The Pro checkpoint uses the same tensor naming as Flash (layers.N.attn.*,
ffn.experts.E.w{1,2,3}, compressor/indexer, hc_*, mtp.0.*) with different
geometry: hidden 7168, 61 layers + 1 packed MTP layer, 384 routed experts,
128 attention heads, query lora rank 1536, 16 output groups. This wrapper
loads the Flash packer and replaces only the geometry-bearing functions.

The Pro contract carries no authoritative source manifest, so the
source-identity check records the index SHA-256 as provenance instead of
pin-checking a contract hash.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
FLASH_PACKER = ROOT / "tools" / "dsv4_stagepack.py"

spec = importlib.util.spec_from_file_location("dsv4_flash_stagepack", FLASH_PACKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load flash packer: {FLASH_PACKER}")
flash = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = flash
spec.loader.exec_module(flash)

# --- Pro geometry ---------------------------------------------------------

PRO_LAYERS = 61
PRO_HIDDEN = 7168
PRO_VOCAB = 129280
PRO_EXPERTS = 384
PRO_EXPERT_ROWS = 3072
PRO_EXPERT_COLS = 7168
PRO_GATE_ROWS = 384
PRO_QUERY_RANK = 1536
PRO_Q_DIM = 128 * 512      # 128 heads x 512 head dim
PRO_HEAD_DIM = 512
PRO_ATTN_HEADS = 128
PRO_OUTPUT_GROUPS = 16
PRO_OUTPUT_RANK = 1024
PRO_MTP_PACKED = 1

# 62 entries = 61 layers + 1 MTP layer; mirrors SparkDsv4ProCompressionRatios.
PRO_RATIOS = [128, 128] + ([4, 128] * 29) + [4, 0]

flash.CONTRACT_PATH = ROOT / "model_contracts" / "dsv4_pro.json"
flash.FP4_EXPERTS = PRO_EXPERTS


def pro_add_layer_records(records, ratios: Sequence[int], layer: int) -> None:
    prefix = flash.source_prefix(layer)
    kind = flash.layer_kind(ratios, layer)
    hidden = PRO_HIDDEN
    query_rank = PRO_QUERY_RANK
    q_dim = PRO_Q_DIM
    head_dim = PRO_HEAD_DIM
    expert_rows = PRO_EXPERT_ROWS
    expert_columns = PRO_EXPERT_COLS
    add_record = flash.add_record
    add_linear = flash.add_linear
    add_experts = flash.add_experts
    add_record(records, flash.KIND_ATTN_SINK, layer, flash.WEIGHT_F32, 1,
               PRO_ATTN_HEADS, (f"{prefix}.attn.attn_sink",))
    add_linear(records, flash.KIND_WQ_A, f"{prefix}.attn", layer,
               query_rank, hidden, flash.WEIGHT_FP8, "wq_a")
    add_record(records, flash.KIND_Q_NORM, layer, flash.WEIGHT_BF16, 1,
               query_rank, (f"{prefix}.attn.q_norm.weight",))
    add_linear(records, flash.KIND_WQ_B, f"{prefix}.attn", layer,
               q_dim, query_rank, flash.WEIGHT_FP8, "wq_b")
    add_linear(records, flash.KIND_WKV, f"{prefix}.attn", layer,
               head_dim, hidden, flash.WEIGHT_FP8, "wkv")
    add_record(records, flash.KIND_KV_NORM, layer, flash.WEIGHT_BF16, 1,
               head_dim, (f"{prefix}.attn.kv_norm.weight",))
    add_linear(records, flash.KIND_WO_A, f"{prefix}.attn", layer,
               PRO_OUTPUT_GROUPS * PRO_OUTPUT_RANK, q_dim // PRO_OUTPUT_GROUPS,
               flash.WEIGHT_FP8, "wo_a")
    add_linear(records, flash.KIND_WO_B, f"{prefix}.attn", layer,
               hidden, PRO_OUTPUT_GROUPS * PRO_OUTPUT_RANK,
               flash.WEIGHT_FP8, "wo_b")
    add_record(records, flash.KIND_ATTN_NORM, layer, flash.WEIGHT_BF16, 1,
               hidden, (f"{prefix}.attn_norm.weight",))
    add_record(records, flash.KIND_FFN_NORM, layer, flash.WEIGHT_BF16, 1,
               hidden, (f"{prefix}.ffn_norm.weight",))
    for kind_id, name in ((flash.KIND_HC_ATTN_FN, "hc_attn_fn"),
                          (flash.KIND_HC_FFN_FN, "hc_ffn_fn")):
        add_record(records, kind_id, layer, flash.WEIGHT_F32, 24, 4 * hidden,
                   (f"{prefix}.{name}",))
    for kind_id, name, columns in ((flash.KIND_HC_ATTN_BASE, "hc_attn_base", 24),
                                   (flash.KIND_HC_FFN_BASE, "hc_ffn_base", 24),
                                   (flash.KIND_HC_ATTN_SCALE, "hc_attn_scale", 3),
                                   (flash.KIND_HC_FFN_SCALE, "hc_ffn_scale", 3)):
        add_record(records, kind_id, layer, flash.WEIGHT_F32, 1, columns,
                   (f"{prefix}.{name}",))
    add_record(records, flash.KIND_GATE_WEIGHT, layer, flash.WEIGHT_BF16,
               PRO_GATE_ROWS, hidden, (f"{prefix}.ffn.gate.weight",))
    if layer < 3:
        add_record(records, flash.KIND_GATE_TID2EID, layer, flash.WEIGHT_U32,
                   PRO_VOCAB, 6, (f"{prefix}.ffn.gate.tid2eid",),
                   i64_to_u32=True)
    else:
        add_record(records, flash.KIND_GATE_BIAS, layer, flash.WEIGHT_F32, 1,
                   PRO_GATE_ROWS, (f"{prefix}.ffn.gate.bias",))
    add_experts(records, flash.KIND_EXPERTS_W1, prefix, layer,
                expert_rows, expert_columns, "w1")
    add_experts(records, flash.KIND_EXPERTS_W2, prefix, layer,
                hidden, expert_rows, "w2")
    add_experts(records, flash.KIND_EXPERTS_W3, prefix, layer,
                expert_rows, expert_columns, "w3")
    add_linear(records, flash.KIND_SHARED_W1, f"{prefix}.ffn.shared_experts",
               layer, expert_rows, expert_columns, flash.WEIGHT_FP8, "w1")
    add_linear(records, flash.KIND_SHARED_W2, f"{prefix}.ffn.shared_experts",
               layer, hidden, expert_rows, flash.WEIGHT_FP8, "w2")
    add_linear(records, flash.KIND_SHARED_W3, f"{prefix}.ffn.shared_experts",
               layer, expert_rows, expert_columns, flash.WEIGHT_FP8, "w3")
    if kind != 0:
        ratio = 4 if kind == 1 else 128
        overlap = 2 if kind == 1 else 1
        channels = overlap * head_dim
        add_record(records, flash.KIND_COMPRESS_APE, layer, flash.WEIGHT_F32,
                   ratio, channels, (f"{prefix}.attn.compressor.ape",))
        add_record(records, flash.KIND_COMPRESS_WKV, layer, flash.WEIGHT_BF16,
                   channels, hidden,
                   (f"{prefix}.attn.compressor.wkv.weight",))
        add_record(records, flash.KIND_COMPRESS_WGATE, layer, flash.WEIGHT_BF16,
                   channels, hidden,
                   (f"{prefix}.attn.compressor.wgate.weight",))
        add_record(records, flash.KIND_COMPRESS_NORM, layer, flash.WEIGHT_BF16,
                   1, head_dim, (f"{prefix}.attn.compressor.norm.weight",))
    if kind == 1:
        index_prefix = f"{prefix}.attn.indexer"
        add_linear(records, flash.KIND_INDEX_WQ_B, index_prefix, layer,
                   64 * 128, query_rank, flash.WEIGHT_FP8, "wq_b")
        add_record(records, flash.KIND_INDEX_WEIGHTS, layer, flash.WEIGHT_BF16,
                   64, hidden, (f"{index_prefix}.weights_proj.weight",))
        add_record(records, flash.KIND_INDEX_APE, layer, flash.WEIGHT_F32, 4,
                   256, (f"{index_prefix}.compressor.ape",))
        add_record(records, flash.KIND_INDEX_WKV, layer, flash.WEIGHT_BF16,
                   256, hidden, (f"{index_prefix}.compressor.wkv.weight",))
        add_record(records, flash.KIND_INDEX_WGATE, layer, flash.WEIGHT_BF16,
                   256, hidden, (f"{index_prefix}.compressor.wgate.weight",))
        add_record(records, flash.KIND_INDEX_NORM, layer, flash.WEIGHT_BF16,
                   1, 128, (f"{index_prefix}.compressor.norm.weight",))


def pro_build_records(contract: Mapping[str, object], first_layer: int,
                      layer_count: int):
    model = contract["model"]
    if not isinstance(model, dict):
        raise flash.PackFailure("contract model section is malformed")
    total_layers = int(model["layer_count"])
    if first_layer < 0 or layer_count <= 0 or \
            first_layer + layer_count > total_layers:
        raise flash.PackFailure(
            f"invalid layer slice {first_layer}+{layer_count}")
    records = []
    if first_layer == 0:
        flash.add_record(records, flash.KIND_EMBEDDING, flash.GLOBAL_LAYER,
                         flash.WEIGHT_BF16, PRO_VOCAB, PRO_HIDDEN,
                         ("embed.weight",))
    for layer in range(first_layer, first_layer + layer_count):
        pro_add_layer_records(records, PRO_RATIOS, layer)
    if first_layer + layer_count == total_layers:
        if first_layer != 0 and PRO_MTP_PACKED != 0:
            flash.add_record(records, flash.KIND_EMBEDDING,
                             flash.GLOBAL_LAYER, flash.WEIGHT_BF16, PRO_VOCAB,
                             PRO_HIDDEN, ("embed.weight",))
        flash.add_record(records, flash.KIND_FINAL_NORM, flash.GLOBAL_LAYER,
                         flash.WEIGHT_BF16, 1, PRO_HIDDEN, ("norm.weight",))
        flash.add_record(records, flash.KIND_LM_HEAD, flash.GLOBAL_LAYER,
                         flash.WEIGHT_BF16, PRO_VOCAB, PRO_HIDDEN,
                         ("head.weight",))
        flash.add_record(records, flash.KIND_HC_HEAD_FN, flash.GLOBAL_LAYER,
                         flash.WEIGHT_F32, 4, 4 * PRO_HIDDEN, ("hc_head_fn",))
        flash.add_record(records, flash.KIND_HC_HEAD_BASE, flash.GLOBAL_LAYER,
                         flash.WEIGHT_F32, 1, 4, ("hc_head_base",))
        flash.add_record(records, flash.KIND_HC_HEAD_SCALE, flash.GLOBAL_LAYER,
                         flash.WEIGHT_F32, 1, 1, ("hc_head_scale",))
        if PRO_MTP_PACKED != 0:
            flash.add_record(records, flash.KIND_MTP_E_PROJ,
                             flash.GLOBAL_LAYER, flash.WEIGHT_FP8, PRO_HIDDEN,
                             PRO_HIDDEN, ("mtp.0.e_proj.weight",),
                             ("mtp.0.e_proj.scale",))
            flash.add_record(records, flash.KIND_MTP_H_PROJ,
                             flash.GLOBAL_LAYER, flash.WEIGHT_FP8, PRO_HIDDEN,
                             PRO_HIDDEN, ("mtp.0.h_proj.weight",),
                             ("mtp.0.h_proj.scale",))
            flash.add_record(records, flash.KIND_MTP_ENORM,
                             flash.GLOBAL_LAYER, flash.WEIGHT_BF16, 1,
                             PRO_HIDDEN, ("mtp.0.enorm.weight",))
            flash.add_record(records, flash.KIND_MTP_HNORM,
                             flash.GLOBAL_LAYER, flash.WEIGHT_BF16, 1,
                             PRO_HIDDEN, ("mtp.0.hnorm.weight",))
            flash.add_record(records, flash.KIND_MTP_FINAL_NORM,
                             flash.GLOBAL_LAYER, flash.WEIGHT_BF16, 1,
                             PRO_HIDDEN, ("mtp.0.norm.weight",))
            flash.add_record(records, flash.KIND_MTP_HC_HEAD_FN,
                             flash.GLOBAL_LAYER, flash.WEIGHT_F32, 4,
                             4 * PRO_HIDDEN, ("mtp.0.hc_head_fn",))
            flash.add_record(records, flash.KIND_MTP_HC_HEAD_BASE,
                             flash.GLOBAL_LAYER, flash.WEIGHT_F32, 1, 4,
                             ("mtp.0.hc_head_base",))
            flash.add_record(records, flash.KIND_MTP_HC_HEAD_SCALE,
                             flash.GLOBAL_LAYER, flash.WEIGHT_F32, 1, 1,
                             ("mtp.0.hc_head_scale",))
            pro_add_layer_records(records, PRO_RATIOS, flash.MTP_LAYER)
    return records


def pro_pack_header(records, first_layer: int, layer_count: int,
                    file_bytes: int, codecs) -> bytes:
    packed_mtp_layer_count = int(any(
        record.layer == flash.MTP_LAYER for record in records))
    return flash.HEADER_STRUCT.pack(
        0x34565344, flash.FORMAT_VERSION, flash.HEADER_STRUCT.size,
        flash.ENTRY_STRUCT.size, flash.CODEC_ABI_VERSION, *codecs,
        len(records), first_layer, layer_count,
        PRO_LAYERS, PRO_HIDDEN, PRO_VOCAB, PRO_EXPERTS,
        packed_mtp_layer_count,
        flash.HEADER_STRUCT.size, file_bytes,
    )


def pro_validate_source_identity(model_dir: Path, contract, records=None,
                                 first_layer=None, layer_count=None) -> str:
    # The Pro contract carries no pinned index hash; record the observed one.
    return flash.sha256_file(model_dir / flash.SOURCE_INDEX_NAME)


def pro_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path)
    source.add_argument("--verify-pack", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first-layer", type=int)
    parser.add_argument("--layer-count", type=int)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--contract", type=Path,
                        default=flash.CONTRACT_PATH)
    arguments = parser.parse_args(argv)
    if arguments.verify_pack is not None:
        if arguments.output is not None or arguments.inspect or \
                arguments.first_layer is not None or \
                arguments.layer_count is not None:
            parser.error("--verify-pack does not accept source-pack arguments")
    elif arguments.first_layer is None or arguments.layer_count is None:
        parser.error("--model-dir requires --first-layer and --layer-count")
    elif arguments.sha256:
        parser.error("--sha256 requires --verify-pack")
    return arguments


# Patch the flash module's geometry-bearing functions, then reuse its main().
flash.add_layer_records = pro_add_layer_records
flash.build_records = pro_build_records
flash.pack_header = pro_pack_header
flash.validate_source_identity = pro_validate_source_identity
flash.parse_args = pro_parse_args

if __name__ == "__main__":
    raise SystemExit(flash.main())
