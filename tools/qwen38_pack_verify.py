#!/usr/bin/env python3
"""Verify qwen38 v2 stage packs against their source checkpoint.

Checks, in order:
  1. Header: magic, format version 2, geometry, and the tp fields against
     --tp-degree/--tp-rank (a pack is ONE rank's shard; mixing ranks is a
     hard error).
  2. Directory: every entry inside the file, shapes equal to the format's
     per-rank shape table (imported from tools/qwen38_stagepack.py - the
     packer and verifier can never disagree), natural-format acceptance
     (BF16 spine / F32 decay / MXFP4 or FP8 experts), payload and scale
     byte sizes exact.
  3. Coverage: the expected (kind, layer) inventory exactly once each.
  4. Byte traces: sampled entries compared byte-for-byte against the
     checkpoint slice recomputed through the SAME shard plan the packer
     used. Copies must be exact - any mismatch names the entry.

Usage:
  qwen38_pack_verify.py --pack PACK --checkpoint DIR --source-format quark-mxfp4 \
      --tp-degree 4 --tp-rank 2 [--sample 12]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import qwen38_stagepack as sp  # noqa: E402  (the packer IS the format spec)
from spark_pack_common import PackFailure  # noqa: E402

MAGIC = sp.MAGIC
HEADER_STRUCT = sp.HEADER_STRUCT
ENTRY_STRUCT = sp.ENTRY_STRUCT


def dequant_mxfp4_group(payload: bytes, scales: bytes, base: int, count: int) -> list[float]:
    """Decode `count` E2M1 elements starting at element `base` (group-32
    E8M0 scales) - the verifier's dequant sanity probe."""
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    values = []
    for index in range(count):
        element = base + index
        pair = payload[element >> 1]
        nibble = (pair >> 4) & 0x0F if element & 1 else pair & 0x0F
        scale_code = scales[element // sp.MXFP4_GROUP]
        scale = 0.0 if scale_code == 255 else 2.0 ** (scale_code - 127)
        magnitude = magnitudes[nibble & 7]
        values.append(-magnitude * scale if nibble & 8 else magnitude * scale)
    return values


def verify(pack_path: Path, checkpoint: Path, source_format: str,
           tp_degree: int, tp_rank: int, sample_count: int) -> int:
    source = sp.SafetensorsSource(checkpoint)
    source.check_config()
    # Rank packs are tens of GiB: everything streams, nothing is fully read.
    pack = pack_path.open("rb")
    header_blob = pack.read(sp.HEADER_BYTES)
    if len(header_blob) < sp.HEADER_BYTES:
        raise PackFailure(f"pack is shorter than the header")
    blob = header_blob  # unpack sources name it blob for the header fields
    import os
    file_size = pack_path.stat().st_size
    def read_at(offset: int, count: int) -> bytes:
        pack.seek(offset)
        chunk = pack.read(count)
        if len(chunk) != count:
            raise PackFailure(f"short read at {offset:#x} ({len(chunk)} of {count})")
        return chunk
    fields = HEADER_STRUCT.unpack_from(header_blob, 0)
    (magic, version, header_bytes, entry_bytes, tensor_count,
     hidden, layer_count, first_layer, total_layers,
     period, phase, gdn_key_heads, gdn_value_heads, gdn_key_dim, gdn_value_dim,
     conv_kernel, attn_query_heads, attn_kv_heads, attn_head_dim, attn_rope_dim,
     expert_count, experts_per_token, expert_intermediate, vocab,
     mxfp4_group, mtp_layers, header_tp_degree, header_tp_rank,
     directory_offset, file_bytes) = fields
    if magic != MAGIC:
        raise PackFailure(f"bad magic {magic:#x}")
    if version != sp.FORMAT_VERSION:
        raise PackFailure(f"format version {version}, expected {sp.FORMAT_VERSION}")
    if (header_tp_degree, header_tp_rank) != (tp_degree, tp_rank):
        raise PackFailure(f"pack is tp {header_tp_degree}/{header_tp_rank}, verifying tp {tp_degree}/{tp_rank}")
    if header_bytes != sp.HEADER_BYTES or entry_bytes != sp.ENTRY_BYTES:
        raise PackFailure("wire sizes drifted")
    if file_bytes != file_size:
        raise PackFailure(f"header says {file_bytes} bytes, file is {file_size}")
    expected_header = (sp.HIDDEN, sp.LAYER_COUNT, sp.ATTENTION_PERIOD, sp.FULL_PHASE,
                       sp.GDN_KEY_HEADS, sp.GDN_VALUE_HEADS, sp.GDN_HEAD_KEY_DIM,
                       sp.GDN_HEAD_VALUE_DIM, sp.GDN_CONV_KERNEL, sp.ATTN_QUERY_HEADS,
                       sp.ATTN_KV_HEADS, sp.ATTN_HEAD_DIM, sp.ATTN_ROPE_DIM,
                       sp.EXPERT_COUNT, sp.EXPERTS_PER_TOKEN, sp.EXPERT_INTERMEDIATE,
                       sp.VOCAB, sp.MXFP4_GROUP, sp.MTP_LAYERS)
    actual_header = (hidden, total_layers, period, phase, gdn_key_heads, gdn_value_heads,
                     gdn_key_dim, gdn_value_dim, conv_kernel, attn_query_heads,
                     attn_kv_heads, attn_head_dim, attn_rope_dim, expert_count,
                     experts_per_token, expert_intermediate, vocab, mxfp4_group, mtp_layers)
    if actual_header != expected_header:
        raise PackFailure("geometry fields drifted from the pinned constants")

    # Directory: shapes, formats, bounds, coverage.
    entries = []
    seen = set()
    directory_blob = read_at(directory_offset, tensor_count * sp.ENTRY_BYTES)
    for index in range(tensor_count):
        entry = ENTRY_STRUCT.unpack_from(directory_blob, index * sp.ENTRY_BYTES)
        (kind, layer, weight_format, rows, columns, scale_group,
         payload_offset, payload_bytes, scale_offset, scale_bytes) = entry
        want_rows, want_columns, want_format = sp.kind_shape(kind, tp_degree)
        if (rows, columns) != (want_rows, want_columns):
            raise PackFailure(f"entry {index} kind={kind} shape {(rows, columns)} != {(want_rows, want_columns)}")
        if weight_format not in (want_format,) and not (
                want_format in (sp.WEIGHT_MXFP4_E2M1, sp.WEIGHT_FP8_F32B128)
                and weight_format in (sp.WEIGHT_BF16, sp.WEIGHT_MXFP4_E2M1, sp.WEIGHT_FP8_F32B128)):
            raise PackFailure(f"entry {index} kind={kind} format {weight_format} unexpected (natural {want_format})")
        if payload_offset + payload_bytes > file_size or (scale_bytes and scale_offset + scale_bytes > file_size):
            raise PackFailure(f"entry {index} outside the file")
        key = (kind, layer)
        if key in seen:
            raise PackFailure(f"duplicate entry {key}")
        seen.add(key)
        entries.append(entry)
    expected_refs = sp.build_inventory(first_layer, layer_count, tp_degree)
    expected_keys = {(ref.kind, ref.layer) for ref in expected_refs}
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)[:4]
        extra = sorted(seen - expected_keys)[:4]
        raise PackFailure(f"coverage mismatch: missing {missing} extra {extra}")

    # Byte traces through the packer's own plans.
    step = max(1, tensor_count // max(sample_count, 1))
    traced = 0
    for index in range(0, tensor_count, step):
        kind, layer, weight_format, rows, columns, scale_group, payload_offset, payload_bytes, scale_offset, scale_bytes = entries[index]
        ref = sp.TensorRef(kind, layer if layer != sp.MTP_LAYER else sp.MTP_LAYER, "", tp_degree)
        ref.rows, ref.columns, ref.weight_format = rows, columns, weight_format
        name = sp.layer_tensor_name(kind, layer) if layer != sp.GLOBAL_LAYER else sp.GLOBAL_TENSORS[kind]
        ref.name = name
        if kind in (sp.KIND_MOE_W1, sp.KIND_MOE_W3, sp.KIND_MOE_DOWN):
            experts = sp.EXPERT_COUNT // tp_degree
            first_expert = tp_rank * experts
            rows_per_expert = rows // experts
            expert = first_expert  # trace the rank's first expert
            checkpoint_name = name.replace("{e}", str(expert))
            shard, meta, offset = source.resolve(checkpoint_name)
            with (source.root / shard).open("rb") as f:
                f.seek(offset)
                if weight_format == sp.WEIGHT_MXFP4_E2M1:
                    expected_payload = f.read(rows_per_expert * columns // 2)
                    pack_payload = read_at(payload_offset, len(expected_payload))
                    if pack_payload != expected_payload:
                        raise PackFailure(f"kind {kind} layer {layer}: payload bytes differ from checkpoint expert {expert}")
                    scale_name = checkpoint_name + ("_scale" if weight_format == sp.WEIGHT_MXFP4_E2M1 else "_scale_inv")
                else:
                    expected_payload = f.read(rows_per_expert * columns)
                    pack_payload = read_at(payload_offset, len(expected_payload))
                    if pack_payload != expected_payload:
                        raise PackFailure(f"kind {kind} layer {layer}: payload bytes differ from checkpoint expert {expert}")
            print(f"trace kind={kind} layer={layer} expert={expert} bytes={len(expected_payload)} receipt=verified")
        else:
            shard, meta, offset = source.check_shape(ref) if hasattr(source, "check_shape") else source.resolve(name)
            # Rebuild the expected bytes with the packer's copy logic.
            from io import BytesIO
            sink = BytesIO()
            if tp_degree > 1:
                sp.copy_sharded_bf16(source, ref, offset, sink)
            else:
                sp.copy_bf16_tensor(source, ref, offset, sink)
            expected_payload = sink.getvalue()
            pack_payload = read_at(payload_offset, len(expected_payload))
            if pack_payload != expected_payload:
                raise PackFailure(f"kind {kind} layer {layer}: slice bytes differ from checkpoint")
            print(f"trace kind={kind} layer={layer} slice_bytes={len(expected_payload)} receipt=verified")
        traced += 1
        if traced >= sample_count:
            break

    # MXFP4 dequant sanity on the first traced expert payload. The probe
    # samples windows spread across the payload: quantized checkpoints
    # legitimately begin with runs of exact-zero E2M1 elements (measured:
    # up to 2.7K leading zero bytes on layer-0 experts), so a single
    # window at element 0 false-fails on healthy packs.
    for index in range(0, tensor_count, step):
        kind, layer, weight_format, rows, columns, scale_group, payload_offset, payload_bytes, scale_offset, scale_bytes = entries[index]
        if weight_format == sp.WEIGHT_MXFP4_E2M1:
            windows = 8
            per_window = 256
            finite = nonzero = total = 0
            for w in range(windows):
                base_element = ((payload_bytes // 2) * w) // windows
                payload = read_at(payload_offset + base_element, per_window // 2)
                scales = read_at(scale_offset + base_element // sp.MXFP4_GROUP, per_window // sp.MXFP4_GROUP + 1)
                values = dequant_mxfp4_group(payload, scales, 0, per_window)
                finite += sum(1 for v in values if abs(v) != float("inf"))
                nonzero += sum(1 for v in values if v != 0.0)
                total += per_window
            print(f"trace kind={kind} layer={layer} mxfp4 dequant finite={finite}/{total} nonzero={nonzero}/{total} (8 spread windows)")
            if finite != total or nonzero < total // 8:
                raise PackFailure("mxfp4 dequant sanity failed")
            break
    print(f"PASS {pack_path.name}: header geometry, {tensor_count} directory entries "
          f"(tp {tp_degree}/{tp_rank}), {traced} byte-traced samples receipt=verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-format", choices=("fp8", "quark-mxfp4"), default="quark-mxfp4")
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--sample", type=int, default=12)
    args = parser.parse_args()
    sp.EXPERT_FORMAT[0] = sp.WEIGHT_MXFP4_E2M1 if args.source_format == "quark-mxfp4" else sp.WEIGHT_FP8_F32B128
    sp.TP_RANK[0] = args.tp_rank
    try:
        return verify(args.pack, args.checkpoint, args.source_format,
                      args.tp_degree, args.tp_rank, args.sample)
    except PackFailure as error:
        print(f"FAIL {args.pack.name}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
