#!/usr/bin/env python3
"""Build Ling-3.0-flash (BailingMoeV3) resident-decode stage packs (.lspk).

Reads a warm HF checkpoint (official BF16 release) through header-only
safetensors memmaps and emits one wire-format-v1 .lspk per TP rank. The
tensor vocabulary is the 34-kind table in
modules/ling_resident_decode_stage/source/spark_ling_stagepack_format.h
and every checkpoint->pack transform follows
model-families/ling/name_map.json:

  - kda_qkv_beta = q|k|v|beta rows fused from the checkpoint's separate
    q_proj/k_conv1d(k_proj)/v_proj/b_proj, row order q,k,v,beta to match
    LingSplitFusedProjectionsKernel, sharded PER SECTION.
  - kv_b_proj [8192, 512] split into key-transposed [512, 128] x 32 heads
    and value [128, 512] x 32 heads (both replicated; the module binds
    per-rank head offsets).
  - up|gate row fusions (dense, shared, per-expert) stacked up-first to
    match LmSiluMulKernel's gate_first=false layout.
  - conv1d [channels, kernel] verify-then-pack.
  - routed experts pass through in the source dtype (bf16 for the
    official release; the packer NEVER quantizes). The inter dimension
    shards across ranks (w1 rows, w2 columns); every rank carries all
    512 experts, which is what the indirect grouped GEMM indexes.
  - MTP tensors are OMITTED and the header emits flags = 0 (the
    load-and-ignore contract).

The census must close: every tensor in model.safetensors.index.json is
either packed, omitted MTP, or the run FAILS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import (CODEC_ABI_VERSION, CODEC_BF16, CODEC_FP8, CODEC_NONE, FP8_BLOCK,  # noqa: E402,F401
                               PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT, PAYLOAD_U32, SCALE_F32, SCALE_NONE,
                               fp8_block_scale_rows)
from spark_pack_common import GLOBAL_LAYER, PackFailure, fsync_directory, sha256_bytes  # noqa: E402,F401
from spark_pack_plan import MappedCheckpoint, PlanBuilder, PlanItem, to_bytes  # noqa: E402,F401
from spark_pack_wire import (CODEC_ENTRY_BYTES, CODEC_HEADER_BYTES, EXPERT_RANGE_BYTES_MAX,  # noqa: E402,F401
                             PAYLOAD_ALIGNMENT, Entry, SlabDigestSink, append_journal, ck128, codec_directory_offset,
                             codec_header, emit_region, expert_manifest_bytes, item_region, place_entries,
                             serialize_entry, trusted_journal_prefix, truncated_revision, write_prefix)

REPO_ROOT = Path(__file__).resolve().parent.parent

MAGIC = 0x33474E4C
FORMAT_VERSION = 1
HEADER_BYTES = CODEC_HEADER_BYTES
ENTRY_BYTES = CODEC_ENTRY_BYTES
ALIGNMENT = PAYLOAD_ALIGNMENT

K_EMBEDDING, K_FINAL_NORM, K_LM_HEAD = 0, 1, 2
K_ATTN_NORM, K_Q, K_KV_A, K_KV_A_NORM = 3, 4, 5, 6
K_KV_B_KEY_T, K_KV_B_VALUE, K_ATTN_GATE, K_ATTN_OUTPUT = 7, 8, 9, 10
K_POST_ATTN_NORM = 11
K_DENSE_GATE_UP, K_DENSE_DOWN = 12, 13
K_ROUTER, K_ROUTER_CORRECTION = 14, 15
K_EXPERT_UP_GATE, K_EXPERT_DOWN = 16, 17
K_SHARED_GATE_UP, K_SHARED_DOWN = 18, 19
K_KDA_QKV_BETA, K_KDA_DECAY_PROJ, K_KDA_GATE_PROJ = 20, 21, 22
K_KDA_Q_CONV, K_KDA_K_CONV, K_KDA_V_CONV = 23, 24, 25
K_KDA_DECAY_BIAS, K_KDA_HEAD_LOG_SCALE, K_KDA_OUT_NORM, K_KDA_OUT = 26, 27, 28, 29
K_MTP_EH_PROJ, K_MTP_ENORM, K_MTP_HNORM, K_MTP_SHARED_NORM = 30, 31, 32, 33

HIDDEN = 2560
LAYERS = 42
MTP_LAYER = 42
KDA_HEADS, KDA_KEY, KDA_CONV = 32, 128, 4
KDA_QK = KDA_HEADS * KDA_KEY
KDA_V = KDA_QK
KDA_FUSED_ROWS = 2 * KDA_QK + KDA_V + KDA_HEADS
MLA_HEADS, LATENT, ROPE, NOPE, VDIM = 32, 512, 64, 128, 128
Q_ROWS = MLA_HEADS * (NOPE + ROPE)
KV_ROW = LATENT + ROPE
EXPERTS, EXPERT_INTER = 512, 768
DENSE_INTER = 6144
VOCAB = 157184
FIRST_ROUTED = 2
CONTRACT = REPO_ROOT / "model_contracts" / "ling_authoritative.json"
NAME_MAP = REPO_ROOT / "model-families" / "ling" / "name_map.json"
SPINE_DTYPES = ("BF16", "F32", "F8_E4M3")
VECTOR_DTYPES = ("F32", "BF16")


class SourceReader(MappedCheckpoint):
    spine_dtypes = SPINE_DTYPES

    def __init__(self, model_dir: Path, cache_byte_cap: int = 512 * 1024 ** 2):
        super().__init__(model_dir, cache_byte_cap, config_required=False)


class Packer(PlanBuilder):
    def __init__(self, source: SourceReader, tp_degree: int, tp_rank: int, expert_codec: int):
        super().__init__(source, tp_degree, tp_rank)
        self.expert_codec = expert_codec
        self.packed_names: List[str] = []

    def track(self, *names: str) -> None:
        self.packed_names.extend(names)

    def add_replicated_bf16(self, kind: int, layer: int, name: str, rows: int, columns: int) -> PlanItem:
        return self.add_bf16(kind, layer, name, "", dtypes=SPINE_DTYPES, shape=(rows, columns))

    def add_rows_sharded_bf16(self, kind: int, layer: int, name: str, rows: int, columns: int) -> PlanItem:
        return self.add_bf16(kind, layer, name, "rows", dtypes=SPINE_DTYPES, shape=(rows, columns))

    def add_cols_sharded_bf16(self, kind: int, layer: int, name: str, rows: int, columns: int) -> PlanItem:
        return self.add_bf16(kind, layer, name, "cols", dtypes=SPINE_DTYPES, shape=(rows, columns))

    def add_vector_f32(self, kind: int, layer: int, name: str, total: int, shard: bool) -> PlanItem:
        dtype, shape, _ = self.source.meta(name)
        if dtype not in VECTOR_DTYPES or tuple(shape) not in ((total,),) + (() if shard else ((1, total),)):
            raise PackFailure(f"{name}: expected F32[{total}] (or exact BF16), got {dtype} {tuple(shape)}")
        return self.add_f32_vector(kind, layer, name, shard=shard)

    def add_fused_sections(self, kind: int, layer: int, sections, out_columns: int) -> PlanItem:
        return self.add_fused_rows(kind, layer, [name for name, _ in sections], [width for _, width in sections],
                                   out_columns, shard=True)

    def add_conv(self, kind: int, layer: int, name: str, channels: int) -> PlanItem:
        return self.add_bf16(kind, layer, name, "rows", dtypes=SPINE_DTYPES, shape=(channels, KDA_CONV))

    def add_experts(self, layer: int, prefix: str) -> None:
        if self.expert_codec not in (CODEC_BF16, CODEC_FP8):
            raise PackFailure("only the bf16 and fp8 expert arms are wired; other codecs need their own receipt set")
        start, width = self.slice_range(EXPERT_INTER)
        up_gate = [[f"{prefix}.{expert}.up_proj.weight", f"{prefix}.{expert}.gate_proj.weight"]
                   for expert in range(EXPERTS)]
        down = [[f"{prefix}.{expert}.down_proj.weight"] for expert in range(EXPERTS)]
        for (up, gate), (down_name,) in zip(up_gate, down):
            self.track(up, gate, down_name)
        w1 = self.add_routed_experts(K_EXPERT_UP_GATE, layer, up_gate, EXPERT_INTER, HIDDEN, (start, start + width),
                                     (0, HIDDEN), self.expert_codec, check_shapes=True)
        w2 = self.add_routed_experts(K_EXPERT_DOWN, layer, down, HIDDEN, EXPERT_INTER, (0, HIDDEN),
                                     (start, start + width), self.expert_codec, check_shapes=True)
        if self.expert_codec != CODEC_FP8:
            return
        source = self.source
        w1_blocks = (HIDDEN + FP8_BLOCK - 1) // FP8_BLOCK
        w2_blocks = (width + FP8_BLOCK - 1) // FP8_BLOCK

        def plane(name: str, blocks: Optional[int]) -> np.ndarray:
            scale_name = name + "_scale_inv"
            grid = source.raw(scale_name).view(np.float32).reshape(source.meta(scale_name)[1])
            if blocks is not None and grid.shape[1] != blocks:
                raise PackFailure(f"{scale_name}: {grid.shape[1]} column blocks, expected {blocks}")
            return grid

        def produce_w1_scale():
            for group in up_gate:
                yield to_bytes(np.concatenate([fp8_block_scale_rows(plane(name, w1_blocks), start, width, 0, w1_blocks)
                                               for name in group], axis=0))

        def produce_w2_scale():
            for (name,) in down:
                yield to_bytes(fp8_block_scale_rows(plane(name, None), 0, HIDDEN, start // FP8_BLOCK, w2_blocks))

        w1.produce_scale, w1.entry.scale_bytes = produce_w1_scale, EXPERTS * 2 * width * w1_blocks * 4
        w2.produce_scale, w2.entry.scale_bytes = produce_w2_scale, EXPERTS * HIDDEN * w2_blocks * 4

    def build(self, source_revision: str) -> None:
        quant = self.source.config.get("quantization_config")
        if quant is not None and (self.expert_codec != CODEC_FP8 or quant.get("quant_method") != "fp8"):
            raise PackFailure("source carries quantization_config; the bf16 arm requires the official BF16 release")
        torch_dtype = self.source.config.get("torch_dtype")
        if torch_dtype is not None and str(torch_dtype) not in ("bfloat16", "torch.bfloat16"):
            raise PackFailure(f"torch_dtype {torch_dtype} contradicts the bf16 arm")
        p = "model.layers"
        for layer in range(LAYERS):
            a = f"{p}.{layer}.attention."
            m = f"{p}.{layer}.mlp."
            self.add_replicated_bf16(K_ATTN_NORM, layer, f"{p}.{layer}.input_layernorm.weight", 1, HIDDEN)
            self.add_replicated_bf16(K_POST_ATTN_NORM, layer, f"{p}.{layer}.post_attention_layernorm.weight", 1, HIDDEN)
            if (layer + 1) % 6 == 0:
                self.add_rows_sharded_bf16(K_Q, layer, f"{a}q_proj.weight", Q_ROWS, HIDDEN)
                self.add_replicated_bf16(K_KV_A, layer, f"{a}kv_a_proj_with_mqa.weight", KV_ROW, HIDDEN)
                self.add_replicated_bf16(K_KV_A_NORM, layer, f"{a}kv_a_layernorm.weight", 1, LATENT)
                self.add_kv_b(K_KV_B_KEY_T, K_KV_B_VALUE, layer, f"{a}kv_b_proj.weight", MLA_HEADS, NOPE, VDIM,
                              LATENT, dtypes=("BF16",))
                self.add_rows_sharded_bf16(K_ATTN_GATE, layer, f"{a}g_proj.weight", MLA_HEADS, HIDDEN)
                self.add_cols_sharded_bf16(K_ATTN_OUTPUT, layer, f"{a}dense.weight", HIDDEN, MLA_HEADS * VDIM)
            else:
                self.add_fused_sections(K_KDA_QKV_BETA, layer, [(f"{a}q_proj.weight", KDA_QK),
                                                                (f"{a}k_proj.weight", KDA_QK),
                                                                (f"{a}v_proj.weight", KDA_QK),
                                                                (f"{a}b_proj.weight", KDA_HEADS)], HIDDEN)
                self.add_conv(K_KDA_Q_CONV, layer, f"{a}q_conv1d.weight", KDA_QK)
                self.add_conv(K_KDA_K_CONV, layer, f"{a}k_conv1d.weight", KDA_QK)
                self.add_conv(K_KDA_V_CONV, layer, f"{a}v_conv1d.weight", KDA_V)
                self.add_rows_sharded_bf16(K_KDA_DECAY_PROJ, layer, f"{a}f_proj.weight", KDA_QK, HIDDEN)
                self.add_rows_sharded_bf16(K_KDA_GATE_PROJ, layer, f"{a}g_proj.weight", KDA_V, HIDDEN)
                self.add_vector_f32(K_KDA_DECAY_BIAS, layer, f"{a}dt_bias", KDA_QK, shard=True)
                self.add_vector_f32(K_KDA_HEAD_LOG_SCALE, layer, f"{a}A_log", KDA_HEADS, shard=True)
                self.add_vector_f32(K_KDA_OUT_NORM, layer, f"{a}o_norm.weight", KDA_KEY, shard=False)
                self.add_cols_sharded_bf16(K_KDA_OUT, layer, f"{a}o_proj.weight", HIDDEN, KDA_V)
            if layer < FIRST_ROUTED:
                self.add_fused_rows(K_DENSE_GATE_UP, layer, [f"{m}up_proj.weight", f"{m}gate_proj.weight"],
                                    [DENSE_INTER, DENSE_INTER], HIDDEN, shard=True)
                self.add_cols_sharded_bf16(K_DENSE_DOWN, layer, f"{m}down_proj.weight", HIDDEN, DENSE_INTER)
            else:
                self.add_replicated_bf16(K_ROUTER, layer, f"{m}gate.weight", EXPERTS, HIDDEN)
                self.add_vector_f32(K_ROUTER_CORRECTION, layer, f"{m}gate.expert_bias", EXPERTS, shard=False)
                self.add_experts(layer, f"{m}experts")
                self.add_fused_rows(K_SHARED_GATE_UP, layer, [f"{m}shared_experts.up_proj.weight",
                                                              f"{m}shared_experts.gate_proj.weight"],
                                    [EXPERT_INTER, EXPERT_INTER], HIDDEN, shard=True)
                self.add_cols_sharded_bf16(K_SHARED_DOWN, layer, f"{m}shared_experts.down_proj.weight", HIDDEN,
                                           EXPERT_INTER)
        self.add_rows_sharded_bf16(K_EMBEDDING, GLOBAL_LAYER, "model.word_embeddings.weight", VOCAB, HIDDEN)
        self.add_replicated_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight", 1, HIDDEN)
        self.add_rows_sharded_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", VOCAB, HIDDEN)
        self.source_revision = source_revision


def assemble_header(plan: List[PlanItem], tp_degree: int, tp_rank: int, expert_codec: int, directory_offset: int,
                    file_bytes: int, revision: str, contract_sha: bytes) -> bytes:
    fields = (MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, CODEC_ABI_VERSION, 0, len(plan), 1, 0, 0, LAYERS,
              LAYERS, HIDDEN, VOCAB, EXPERTS, CODEC_BF16, expert_codec, CODEC_BF16, tp_degree, tp_rank)
    return codec_header(fields, directory_offset, file_bytes, truncated_revision(revision), [contract_sha])


def expert_entry(item: PlanItem) -> bool:
    return item.entry.kind in (K_EXPERT_UP_GATE, K_EXPERT_DOWN)


def write_regions(out, packer, skip: int, journal: Optional[Any]) -> None:
    for index, item in enumerate(packer.plan):
        if index < skip:
            continue
        entry = item.entry
        payload_sink = SlabDigestSink(entry.payload_bytes // EXPERTS) if expert_entry(item) else None
        scale_sink = SlabDigestSink(entry.scale_bytes // EXPERTS) if expert_entry(item) and entry.scale_bytes else None
        emit_region(out, entry.payload_offset, entry.payload_bytes, item.produce_payload(), payload_sink)
        if payload_sink is not None:
            if len(payload_sink.digests) != EXPERTS:
                raise PackFailure(f"{entry.kind}: slab digests {len(payload_sink.digests)} != {EXPERTS}")
            item.expert_digests = payload_sink.digests
        if entry.scale_bytes:
            if item.produce_scale is None:
                raise PackFailure(f"{entry.kind}: scale segment has no producer")
            emit_region(out, entry.scale_offset, entry.scale_bytes, item.produce_scale(), scale_sink)
            if scale_sink is not None:
                if len(scale_sink.digests) != EXPERTS:
                    raise PackFailure(f"{entry.kind}: scale slab digests {len(scale_sink.digests)} != {EXPERTS}")
                item.expert_scale_digests = scale_sink.digests
        if journal is not None:
            append_journal(journal, out, index, item)


def emit(packer, path: Path, revision: str, contract_sha: bytes, resume: bool = False) -> int:
    if path.exists():
        raise PackFailure(f"output already exists; the two-pass proof requires a fresh artifact path: {path}")
    directory_offset = codec_directory_offset()
    entries = [item.entry for item in packer.plan]
    file_bytes = place_entries(entries, directory_offset + len(entries) * ENTRY_BYTES)
    header = assemble_header(packer.plan, packer.tp_degree, packer.tp_rank, packer.expert_codec, directory_offset,
                             file_bytes, revision, contract_sha)
    if resume:
        partial = path.parent / (path.name + ".partial")
        journal_file = path.parent / (path.name + ".journal.jsonl")
        done = trusted_journal_prefix(journal_file, partial, packer.plan) if journal_file.exists() else 0
        with partial.open("r+b" if partial.exists() else "w+b") as out:
            write_prefix(out, header, directory_offset, entries)
            with journal_file.open("a", encoding="utf-8") as journal:
                write_regions(out, packer, done, journal)
                out.flush()
                os.fsync(out.fileno())
        os.link(partial, path)
        fsync_directory(path.parent)
        partial.unlink()
        journal_file.unlink()
        return file_bytes
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        with Path(temporary).open("wb") as out:
            write_prefix(out, header, directory_offset, entries)
            write_regions(out, packer, 0, None)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        os.unlink(temporary)
    return file_bytes


def census_check(weight_map: Dict[str, str], packed: List[str], mtp_names: List[str],
                 source: Optional[SourceReader] = None) -> Dict[str, Any]:
    packed_set = set(packed)
    duplicated = len(packed) - len(packed_set)
    if source is not None:
        for name in sorted(packed_set):
            if source.meta(name)[0] == "F8_E4M3":
                packed_set.add(name + "_scale_inv")
    known = set(weight_map)
    unexplained = sorted(known - packed_set - set(mtp_names))
    missing = sorted(packed_set - known)
    if unexplained or missing or duplicated:
        raise PackFailure(f"census does not close: {len(unexplained)} unexplained (e.g. {unexplained[:5]}), "
                          f"{len(missing)} missing-from-index (e.g. {missing[:5]}), {duplicated} double-packed")
    return {"checkpoint_tensors": len(known), "packed": len(packed_set), "omitted_mtp": len(mtp_names)}


def write_expert_manifest(pack_path: Path, plan: List[PlanItem]) -> int:
    ranges = []
    for item in plan:
        if not expert_entry(item):
            continue
        entry = item.entry
        for plane, offset, total, digests in (("payload", entry.payload_offset, entry.payload_bytes,
                                               item.expert_digests),
                                              ("scale", entry.scale_offset, entry.scale_bytes,
                                               item.expert_scale_digests)):
            if plane == "scale" and not total:
                continue
            per = total // EXPERTS
            if per == 0 or per > EXPERT_RANGE_BYTES_MAX:
                raise PackFailure(f"{entry.kind}: per-expert {plane} slab {per} bytes outside the weightd "
                                  f"manifest contract")
            for expert in range(EXPERTS):
                ranges.append((offset + expert * per, per, entry.layer, expert,
                               entry.kind * 2 + (plane == "scale"), digests, expert))
    ranges.sort(key=lambda record: record[:5])
    records = []
    with pack_path.open("rb") as pack:
        for offset, length, layer, expert, kind, digests, index in ranges:
            if digests is not None:
                digest = digests[index]
            else:
                pack.seek(offset)
                digest = ck128(pack.read(length))
            records.append((layer, expert, kind, offset, length, digest))
    (pack_path.parent / (pack_path.name + ".experts")).write_bytes(expert_manifest_bytes(records, len(records)))
    return len(records)


def mtp_tensor_names(weight_map: Dict[str, str]) -> List[str]:
    return sorted(name for name in weight_map if name.startswith("model.layers.42."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkpoint directory (warm)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--tp-all", type=int, default=0, help="emit all N rank packs in one process")
    parser.add_argument("--expert-codec", default="bf16", choices=["bf16", "fp8"])
    parser.add_argument("--model", default="ling", choices=["ling", "lingfin"])
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted emit from the .partial + .journal.jsonl sidecars in the "
                             "output dir")
    args = parser.parse_args()

    family_dir = REPO_ROOT / "model-families" / "ling"
    name_map_path = NAME_MAP if args.model == "ling" else family_dir / f"name_map_{args.model}.json"
    contract_file = CONTRACT if args.model == "ling" else REPO_ROOT / "model_contracts" / f"{args.model}_authoritative.json"
    contract_sha = sha256_bytes(contract_file.read_bytes()) if contract_file.is_file() else bytes(32)
    expert_codec = {"bf16": CODEC_BF16, "fp8": CODEC_FP8}[args.expert_codec]
    if expert_codec == CODEC_FP8:
        name_map_path = family_dir / "name_map_fp8.json"
        if not name_map_path.is_file():
            raise PackFailure(f"the fp8 arm requires {name_map_path} (the fp8 release's census pin)")
    name_map = json.loads(name_map_path.read_text())
    source_revision = name_map["source_revision"]
    expected_census = name_map["checkpoint_census"]["tensor_count"]

    source = SourceReader(Path(args.source))
    if len(source.weight_map) != expected_census:
        raise PackFailure(f"index census {len(source.weight_map)} != pinned {expected_census}; re-pin name_map "
                          f"against this revision")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts = out_dir / "receipts"
    receipts.mkdir(exist_ok=True)
    mtp_names = mtp_tensor_names(source.weight_map)
    tp = args.tp_all or args.tp_degree
    for rank in range(args.tp_all) if args.tp_all else [args.tp_rank]:
        packer = Packer(source, tp, rank, expert_codec)
        packer.build(source_revision)
        census = census_check(source.weight_map, packer.packed_names, mtp_names,
                              source if expert_codec == CODEC_FP8 else None)
        if args.dry_plan:
            print(f"rank {rank}: {len(packer.plan)} pack tensors planned, census {census}")
            continue
        arm = f"{args.model}.{args.expert_codec}.tp{tp}"
        path = out_dir / f"{arm}.rank{rank:x}.sp"
        file_bytes = emit(packer, path, source_revision, contract_sha, args.resume)
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        (out_dir / f"{path.name}.sha256").write_text(f"{digest.hexdigest()}  {path.name}\n")
        spine_bytes = sum(item.entry.payload_bytes for item in packer.plan if not expert_entry(item))
        expert_bytes = sum(item.entry.payload_bytes for item in packer.plan if expert_entry(item))
        manifest_ranges = write_expert_manifest(path, packer.plan)
        receipt = {
            "pack": path.name,
            "arm": arm,
            "sha256": digest.hexdigest(),
            "file_bytes": file_bytes,
            "tensors": len(packer.plan),
            "tp_degree": tp,
            "tp_rank": rank,
            "source": str(args.source),
            "source_revision": source_revision,
            "expert_codec": args.expert_codec,
            "census": census,
            "resident_bytes": {
                "spine": spine_bytes,
                "expert_payload": expert_bytes,
                "note": "spine is always-resident; expert_payload is the weightd lazy tier's working set per node",
            },
            "expert_manifest_ranges": manifest_ranges,
        }
        (receipts / f"rank{rank}.json").write_text(json.dumps(receipt, indent=1) + "\n")
        print(f"{path.name}: {len(packer.plan)} tensors, {file_bytes} bytes, sha256 {receipt['sha256'][:16]}... "
              f"(tp{tp} rank {rank}, census {census['checkpoint_tensors']} = {census['packed']} packed + "
              f"{census['omitted_mtp']} mtp, spine {spine_bytes} + expert {expert_bytes} bytes, "
              f"{manifest_ranges} manifest ranges)")
    source.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as failure:
        print(f"FAIL {failure}", file=sys.stderr)
        sys.exit(1)
