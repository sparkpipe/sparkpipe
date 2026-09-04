#!/usr/bin/env python3
"""Verify qwen4_flash stage packs against the checkpoint they claim to hold.

M4 exit gate for the qwen-flash lane: every rank pack must pass this before
it may feed a module. Checks, in order:

  1. Header geometry vs the pinned constants (magic, format version, the
     full model geometry block) and vs the receipt.
  2. Directory: tensor count matches the expected inventory for the slice;
     every entry resolves to a legal (kind, layer); rows/columns match the
     kind table narrowed by this pack's TP plan; payload and scale byte
     counts match the format formulas; offsets are in-file and 256-aligned.
  3. Byte-trace sampling (--sample N, default 8): re-read each sampled
     tensor's checkpoint source slice and compare against the pack bytes.
     BF16/F32 kinds must match bit-exactly (modulo the receipted hc
     stream-0 sections, which compare against their documented section).
     FP8 experts dequantize with the pack's own scale plane and must land
     within --fp8-relative-l2 (default 0.2; E4M3 is ~2^-3 relative) of the
     source block.

Run on a spark node next to the warm checkpoint:
  python3 tools/qwen4_flash_pack_verify.py --pack build/.../tp4-rank0... \
      --checkpoint /mnt/model-warm/qwen3.8-flash-next
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from qwen4_flash_stagepack import (  # noqa: E402
    ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTENTION_PERIOD, ATTN_Q_DIM, ATTN_QUERY_HEADS,
    ATTN_ROPE_DIM, ENTRY_BYTES, ENTRY_STRUCT, EXPERT_COUNT,
    EXPERTS_PER_TOKEN, EXPERT_INTERMEDIATE, FP8_BLOCK, FORMAT_VERSION,
    FULL_PHASE, GDN_CONV_KERNEL, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
    GDN_KEY_HEADS, GDN_QK_DIM, GDN_VALUE_DIM, GDN_VALUE_HEADS, HEADER_BYTES,
    HEADER_STRUCT, HIDDEN, LAYER_COUNT, MAGIC, MTP_LAYERS, MTP_LAYER,
    MXFP4_GROUP, PLE_LAYER, PLE_NGRAM_HEAD_DIM, PLE_NGRAM_ROWS, VOCAB,
    WEIGHT_BF16, WEIGHT_F32, WEIGHT_I64, WEIGHT_FP8_E8M0B128,
    WEIGHT_FP8_F32B128, build_inventory, is_gdn_layer, kind_shape,
    layer_tensor_name, shard_ref, SafetensorsSource,
)
from spark_pack_common import PackFailure, align_up  # noqa: E402

WEIGHT_NVFP4_PACKED = 8

REPLICATED_NOTE = "replicated"


def read_pack_header(pack: Path) -> dict:
    with pack.open("rb") as file:
        raw = file.read(HEADER_BYTES)
    if len(raw) != HEADER_BYTES:
        raise PackFailure(f"pack header truncated: {len(raw)} bytes")
    fields = HEADER_STRUCT.unpack(raw)
    names = ("magic", "format_version", "header_bytes", "directory_entry_bytes",
             "tensor_count", "hidden_dimension", "layer_count", "first_layer_index",
             "total_layer_count", "attention_period", "full_attention_phase",
             "gdn_key_head_count", "gdn_value_head_count", "gdn_head_key_dimension",
             "gdn_head_value_dimension", "gdn_conv_kernel", "attn_query_head_count",
             "attn_kv_head_count", "attn_head_dimension", "attn_rope_dimension",
             "routed_expert_count", "experts_per_token", "expert_intermediate_dimension",
             "output_vocab_count", "mxfp4_group_size", "mtp_layer_count",
             "directory_offset", "file_bytes")
    return dict(zip(names, fields))


def expected_header_geometry(first_layer: int, layer_count: int, tensor_count: int, include_mtp: bool = True) -> dict:
    return {
        "magic": MAGIC, "format_version": FORMAT_VERSION,
        "header_bytes": HEADER_BYTES, "directory_entry_bytes": ENTRY_BYTES,
        "tensor_count": tensor_count,
        "hidden_dimension": HIDDEN, "layer_count": layer_count,
        "first_layer_index": first_layer, "total_layer_count": LAYER_COUNT,
        "attention_period": ATTENTION_PERIOD, "full_attention_phase": FULL_PHASE,
        "gdn_key_head_count": GDN_KEY_HEADS, "gdn_value_head_count": GDN_VALUE_HEADS,
        "gdn_head_key_dimension": GDN_HEAD_KEY_DIM,
        "gdn_head_value_dimension": GDN_HEAD_VALUE_DIM,
        "gdn_conv_kernel": GDN_CONV_KERNEL,
        "attn_query_head_count": ATTN_QUERY_HEADS, "attn_kv_head_count": ATTN_KV_HEADS,
        "attn_head_dimension": ATTN_HEAD_DIM, "attn_rope_dimension": ATTN_ROPE_DIM,
        "routed_expert_count": EXPERT_COUNT, "experts_per_token": EXPERTS_PER_TOKEN,
        "expert_intermediate_dimension": EXPERT_INTERMEDIATE,
        "output_vocab_count": VOCAB, "mxfp4_group_size": MXFP4_GROUP,
        "mtp_layer_count": MTP_LAYERS if include_mtp else 0,
    }


def read_entries(pack: Path, header: dict) -> list[dict]:
    names = ("tensor_kind", "layer_index", "weight_format", "rows", "columns",
             "scale_group_size", "payload_offset", "payload_bytes",
             "scale_offset", "scale_bytes")
    entries = []
    with pack.open("rb") as file:
        file.seek(header["directory_offset"])
        raw = file.read(header["tensor_count"] * ENTRY_BYTES)
    if len(raw) != header["tensor_count"] * ENTRY_BYTES:
        raise PackFailure("pack directory truncated")
    for index in range(header["tensor_count"]):
        entries.append(dict(zip(names, ENTRY_STRUCT.unpack_from(raw, index * ENTRY_BYTES))))
    return entries


def verify_directory(header: dict, entries: list[dict], tp_degree: int, tp_rank: int,
                     pack_bytes: int, include_mtp: bool = True) -> list[str]:
    problems: list[str] = []
    expected_refs = {("kind", r.kind, "layer", r.layer): r
                     for r in (shard_ref(ref, tp_degree, tp_rank)
                               for ref in build_inventory(header["first_layer_index"],
                                                          header["layer_count"],
                                                          include_mtp))}
    seen = {}
    for entry in entries:
        key = ("kind", entry["tensor_kind"], "layer", entry["layer_index"])
        if key in seen:
            problems.append(f"duplicate entry kind={entry['tensor_kind']} layer={entry['layer_index']}")
        seen[key] = entry
        ref = expected_refs.get(key)
        if ref is None:
            problems.append(f"entry kind={entry['tensor_kind']} layer={entry['layer_index']} not in the slice inventory")
            continue
        if entry["rows"] != ref.rows or entry["columns"] != ref.columns:
            problems.append(
                f"kind={entry['tensor_kind']} layer={entry['layer_index']} shape "
                f"[{entry['rows']},{entry['columns']}] != tp-planned [{ref.rows},{ref.columns}]")
        natural = kind_shape(entry["tensor_kind"])[2]
        wire_format = entry["weight_format"]
        if natural in (WEIGHT_FP8_F32B128,) and wire_format not in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128, WEIGHT_BF16, WEIGHT_NVFP4_PACKED):
            problems.append(f"kind={entry['tensor_kind']} illegal expert format {wire_format}")
        if wire_format == WEIGHT_NVFP4_PACKED:
            # NVFP4 wire: U8-packed e2m1 payloads (2 values/byte) +
            # per-expert segment [rows x cols/16 e4m3 plane][input_scale
            # F32][weight_scale_2 F32]. Expert count derives from the
            # fused geometry (gate/up columns == hidden; down columns ==
            # intermediate) — mirror the module's ScaleBytes rule.
            want_payload = entry["rows"] * (entry["columns"] // 2)
            if entry["columns"] == HIDDEN:
                rows_per_expert = EXPERT_INTERMEDIATE
            elif entry["columns"] == EXPERT_INTERMEDIATE:
                rows_per_expert = HIDDEN
            else:
                problems.append(f"kind={entry['tensor_kind']} nvfp4 columns {entry['columns']} matches neither hidden nor intermediate")
                continue
            if rows_per_expert == 0 or entry["rows"] % rows_per_expert != 0:
                problems.append(f"kind={entry['tensor_kind']} nvfp4 rows {entry['rows']} not expert-tiled by {rows_per_expert}")
                continue
            experts = entry["rows"] // rows_per_expert
            want_scale = entry["rows"] * (entry["columns"] // 16) + experts * 8
            if entry["scale_group_size"] != 16:
                problems.append(f"kind={entry['tensor_kind']} scale group {entry['scale_group_size']} != 16")
        elif wire_format in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128):
            want_payload = entry["rows"] * entry["columns"]
            # F32B128: f32 per 128x128 tile; E8M0B128: exponent byte per
            # (row, 128-column block) - the module kernel's per-row MX plane.
            want_scale = ((entry["rows"] // FP8_BLOCK) * (entry["columns"] // FP8_BLOCK) * 4
                          if wire_format == WEIGHT_FP8_F32B128
                          else entry["rows"] * (entry["columns"] // FP8_BLOCK))
            if entry["scale_group_size"] != FP8_BLOCK:
                problems.append(f"kind={entry['tensor_kind']} scale group {entry['scale_group_size']} != {FP8_BLOCK}")
        elif wire_format == WEIGHT_BF16:
            want_payload, want_scale = entry["rows"] * entry["columns"] * 2, 0
        elif wire_format == WEIGHT_F32:
            want_payload, want_scale = entry["rows"] * entry["columns"] * 4, 0
        elif wire_format == WEIGHT_I64:
            want_payload, want_scale = entry["rows"] * entry["columns"] * 8, 0
        else:
            problems.append(f"kind={entry['tensor_kind']} unknown format {wire_format}")
            continue
        if entry["payload_bytes"] != want_payload or entry["scale_bytes"] != want_scale:
            problems.append(
                f"kind={entry['tensor_kind']} layer={entry['layer_index']} bytes "
                f"payload={entry['payload_bytes']}/{want_payload} scale={entry['scale_bytes']}/{want_scale}")
        for offset, length in ((entry["payload_offset"], entry["payload_bytes"]),
                               (entry["scale_offset"], entry["scale_bytes"])):
            if length == 0:
                continue
            if offset % 256 != 0:
                problems.append(f"kind={entry['tensor_kind']} offset {offset} not 256-aligned")
            if offset > pack_bytes or length > pack_bytes - offset:
                problems.append(f"kind={entry['tensor_kind']} range [{offset},{offset + length}) outside file")
    missing = set(expected_refs) - set(seen)
    if missing:
        problems.append(f"{len(missing)} inventory tensors missing (first: {sorted(missing)[:3]})")
    return problems


def dequantize_block(payload: bytes, scales: bytes, wire_format: int):
    import numpy as np
    codes = np.frombuffer(payload, dtype=np.uint8)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    body = (codes & 0x7F).astype(np.int32)
    exponent = (body >> 3).astype(np.int32)
    mantissa = (body & 7).astype(np.float32)
    value = np.where(exponent > 0,
                     (1.0 + mantissa / 8.0) * np.power(2.0, exponent - 7.0),
                     (mantissa / 8.0) * np.power(2.0, -6.0))
    values = (sign * value).astype(np.float32)
    if wire_format == WEIGHT_FP8_F32B128:
        scale = np.frombuffer(scales, dtype="<f4")
    else:
        codes8 = np.frombuffer(scales, dtype=np.uint8).astype(np.float32)
        scale = np.power(2.0, codes8 - 127.0)
    return values, scale


def sample_trace(pack: Path, entries: list[dict], source: SafetensorsSource,
                 tp_degree: int, tp_rank: int, sample_count: int,
                 fp8_relative_l2: float) -> list[str]:
    import numpy as np
    problems: list[str] = []
    candidates = [entry for entry in entries
                  if entry["weight_format"] in (WEIGHT_BF16, WEIGHT_F32, WEIGHT_I64, WEIGHT_NVFP4_PACKED)
                  or entry["weight_format"] in (WEIGHT_FP8_F32B128, WEIGHT_FP8_E8M0B128)]
    # Guarantee wire-8 coverage: a uniform stride can skip the expert
    # entries entirely, and they are exactly the samples that matter.
    wire8 = [entry for entry in candidates if entry["weight_format"] == WEIGHT_NVFP4_PACKED]
    others = [entry for entry in candidates if entry["weight_format"] != WEIGHT_NVFP4_PACKED]
    general_count = max(1, sample_count - min(3, len(wire8)))
    stride = max(1, len(others) // general_count)
    sampled = others[::stride][:general_count]
    if wire8:
        wstride = max(1, len(wire8) // min(3, len(wire8)))
        sampled += wire8[::wstride][:3]
    with pack.open("rb") as file:
        for entry in sampled:
            kind, layer = entry["tensor_kind"], entry["layer_index"]
            # Resolve the full ref for the name, then re-shard for the slice.
            full = [r for r in build_inventory(0, LAYER_COUNT)
                    if r.kind == kind and r.layer == layer]
            if not full:
                problems.append(f"sample kind={kind} layer={layer} unresolvable")
                continue
            ref = shard_ref(full[0], tp_degree, tp_rank)
            file.seek(entry["payload_offset"])
            payload = file.read(entry["payload_bytes"])
            if entry["scale_bytes"]:
                file.seek(entry["scale_offset"])
                scales = file.read(entry["scale_bytes"])
            else:
                scales = b""
            if entry["weight_format"] == WEIGHT_NVFP4_PACKED:
                # Repackage-only arm: the gate is BYTE-EXACT. Spot
                # experts: the pack segment must equal the release's
                # own packed weight bytes, e4m3 plane rows, and the
                # per-expert F32 globals (input, weight_scale_2).
                import numpy as np
                expert_start, expert_count = getattr(ref, "expert_slice", (0, EXPERT_COUNT))
                rows_per_expert = entry["rows"] // expert_count
                plane_cols = entry["columns"] // 16
                proj = {6: "gate_proj", 7: "up_proj", 8: "down_proj"}.get(kind)
                fused = "gate_up_proj" if kind in (6, 7) else "down_proj"
                if proj is None:
                    problems.append(f"kind={kind} unexpected nvfp4 kind")
                    continue
                for probe in (0, expert_count // 2, expert_count - 1):
                    e_abs = expert_start + probe
                    # MoE inventory names carry no .weight suffix; the
                    # split source tensors do.
                    stem = ref.name.replace(
                        "mlp.experts." + fused,
                        f"mlp.experts.{e_abs}." + proj)
                    want_payload_rows = source_bytes(source, stem + ".weight")
                    row_bytes = rows_per_expert * (entry["columns"] // 2)
                    seg_row_base = probe * row_bytes
                    if payload[seg_row_base:seg_row_base + row_bytes] != want_payload_rows:
                        problems.append(f"kind={kind} layer={layer} expert {e_abs} packed payload mismatch")
                        continue
                    want_plane_rows = source_bytes(source, stem + ".weight_scale")
                    plane_row_bytes = rows_per_expert * plane_cols
                    seg_base = probe * (plane_row_bytes + 8)
                    if scales[seg_base:seg_base + plane_row_bytes] != want_plane_rows:
                        problems.append(f"kind={kind} layer={layer} expert {e_abs} e4m3 plane mismatch")
                    want_input = source_bytes(source, stem + ".input_scale")
                    want_weight = source_bytes(source, stem + ".weight_scale_2")
                    if scales[seg_base + plane_row_bytes:seg_base + plane_row_bytes + 4] != want_input or \
                            scales[seg_base + plane_row_bytes + 4:seg_base + plane_row_bytes + 8] != want_weight:
                        problems.append(f"kind={kind} layer={layer} expert {e_abs} global scales mismatch")
                continue
            if entry["weight_format"] in (WEIGHT_BF16, WEIGHT_F32):
                # Expected bytes from the checkpoint, honoring slices.
                row_slice = getattr(ref, "row_slice", None)
                column_slice = getattr(ref, "column_slice", None)
                triple = getattr(ref, "triple_slice", None)
                if triple:  # fused qkv / conv triple slice
                    matrix = source_matrix(source, ref.name)
                    if matrix.ndim == 3:
                        matrix = matrix[:, 0, :]
                    ks, kc, vs, vc = triple
                    q = matrix[ks:ks + kc, :]
                    k = matrix[GDN_QK_DIM + ks:GDN_QK_DIM + ks + kc, :]
                    v = matrix[2 * GDN_QK_DIM + vs:2 * GDN_QK_DIM + vs + vc, :]
                    expected = np.concatenate((q, k, v), axis=0)
                    got = np.frombuffer(payload, dtype="<u2").reshape(expected.shape)
                    if not np.array_equal(got, expected.astype("<u2")):
                        problems.append(f"kind={kind} layer={layer} bf16 triple-slice mismatch")
                    continue
                if entry["weight_format"] == WEIGHT_I64:
                    # Raw little-endian int64 hash constants: byte-exact copy.
                    shard_name = source.weight_map[ref.name]
                    with (source.root / shard_name).open("rb") as sf:
                        sf.seek(source.resolve(ref.name)[2])
                        want_raw = sf.read(entry["columns"] * 8)
                    if payload != want_raw:
                        problems.append(f"kind={kind} layer={layer} i64 byte mismatch")
                    continue
                if kind == 54:  # PLE n-gram table: spot-check rows in this rank's span
                    import numpy as np
                    shard_start = ref.ngram_shard_range[0] if hasattr(ref, "ngram_shard_range") else 0
                    rows_per_shard = PLE_NGRAM_ROWS // 128
                    for probe in (0, entry["rows"] // 2, entry["rows"] - 1):
                        shard_index = shard_start + probe // rows_per_shard
                        in_shard = probe % rows_per_shard
                        shard_name = f"model.language_model.layers.{PLE_LAYER}.ple.ple_embedding.ngram_embedding.shard_{shard_index}.weight"
                        with (source.root / source.weight_map[shard_name]).open("rb") as sf:
                            sf.seek(source.resolve(shard_name)[2] + in_shard * PLE_NGRAM_HEAD_DIM * 2)
                            want_row = sf.read(PLE_NGRAM_HEAD_DIM * 2)
                        got_row = payload[probe * PLE_NGRAM_HEAD_DIM * 2:(probe + 1) * PLE_NGRAM_HEAD_DIM * 2]
                        if got_row != want_row:
                            problems.append(f"kind=54 ngram row {probe} mismatch (shard {shard_index})")
                            break
                    continue
                if kind in (6, 7, 8) and entry["weight_format"] == WEIGHT_BF16:
                    # BF16-expert repackage (policy): byte-exact vs the rank's
                    # expert slab, mirroring the packer's fused read + split.
                    import numpy as np
                    expert_start, expert_count = getattr(ref, "expert_slice", (0, EXPERT_COUNT))
                    if kind == 8:
                        matrix = source_matrix(source, layer_tensor_name(kind, layer))
                        packed = matrix[expert_start:expert_start + expert_count].reshape(-1, EXPERT_INTERMEDIATE)
                    else:
                        gate_up = source_matrix(source, layer_tensor_name(6, layer))
                        section = gate_up[expert_start:expert_start + expert_count]
                        half = section.shape[1] // 2
                        packed = (section[:, :half, :] if kind == 6 else section[:, half:, :]).reshape(-1, HIDDEN)
                    got = np.frombuffer(payload, dtype="<u2")
                    want = np.ascontiguousarray(packed, dtype="<u2").reshape(-1)
                    if not np.array_equal(got, want):
                        problems.append(f"kind={kind} layer={layer} bf16 expert byte mismatch ({len(got)} vs {len(want)})")
                    continue
                if kind in (1, 31) or kind in (3, 4, 30, 47, 48, 49):
                    # Full-width [4H] group-norm vectors (v2: the stream-0
                    # section approximation is retired).
                    vector = source_vector(source, ref.name)
                    expected = vector[:entry["columns"]].reshape(1, -1) if entry["columns"] < len(vector) else vector.reshape(1, -1)
                elif row_slice is None and column_slice is None:
                    expected = source_matrix(source, ref.name)
                    if expected.ndim == 3:
                        expected = expected.reshape(expected.shape[0], -1)
                    if entry["rows"] == 1:
                        expected = expected.reshape(-1)[:entry["columns"]].reshape(1, -1)
                    elif expected.shape[0] != entry["rows"] or expected.shape[1] != entry["columns"]:
                        expected = expected.reshape(entry["rows"], entry["columns"])
                else:
                    matrix = source_matrix(source, ref.name)
                    if matrix.ndim == 3:
                        matrix = matrix[:, 0, :]
                    if matrix.ndim == 1:
                        # 1-D vectors (A_log/dt_bias) shard flat. The row
                        # slice consumes the whole vector; do NOT fall
                        # through to the 2-D row rule below — it would
                        # re-slice the reshaped single row into nothing
                        # for every rank past 0 (rank 0's slice 0:N
                        # masked this until the rank 1-3 builds).
                        if row_slice is not None:
                            matrix = matrix[row_slice[0]:row_slice[0] + row_slice[1]].reshape(1, -1)
                        else:
                            matrix = matrix.reshape(1, -1)
                    elif row_slice is not None:
                        matrix = matrix[row_slice[0]:row_slice[0] + row_slice[1], :]
                        if column_slice is not None:
                            matrix = matrix[:, column_slice[0]:column_slice[0] + column_slice[1]]
                    elif column_slice is not None:
                        matrix = matrix[:, column_slice[0]:column_slice[0] + column_slice[1]]
                    expected = matrix
                if entry["weight_format"] == WEIGHT_BF16:
                    got = np.frombuffer(payload, dtype="<u2")
                    want = np.ascontiguousarray(expected).astype("<u2").reshape(-1)
                    if not np.array_equal(got, want):
                        problems.append(f"kind={kind} layer={layer} bf16 byte mismatch ({len(got)} vs {len(want)})")
                else:  # widened f32
                    got = np.frombuffer(payload, dtype="<f4")
                    want = bf16_widen(np.ascontiguousarray(expected).astype("<u2").reshape(-1))
                    if not np.allclose(got, want, rtol=0, atol=0):
                        problems.append(f"kind={kind} layer={layer} f32 widen mismatch")
            else:
                # FP8 experts: block-dequantize and compare amplitudes.
                matrix = expert_source_matrix(source, ref, kind, layer)
                values, scale = dequantize_block(payload, scales, entry["weight_format"])
                rows, columns = entry["rows"], entry["columns"]
                if entry["weight_format"] == WEIGHT_FP8_E8M0B128:
                    # Per-row MX plane: scale[r, c/128] applies to the 128
                    # columns of row r in that block (matches the module's
                    # SparkLmDotRowFp8E8m0 decode).
                    scale = scale.reshape(rows, columns // FP8_BLOCK, 1)
                    rebuilt = (values.reshape(rows, columns // FP8_BLOCK, FP8_BLOCK) * scale).reshape(rows, columns).astype(np.float32)
                else:
                    values = values.reshape(rows // FP8_BLOCK, FP8_BLOCK, columns // FP8_BLOCK, FP8_BLOCK)
                    scale = scale.reshape(rows // FP8_BLOCK, 1, columns // FP8_BLOCK, 1)
                    rebuilt = (values * scale).reshape(rows, columns).astype(np.float32)
                source_f32 = matrix.astype(np.float32)
                difference = float(np.sqrt(((rebuilt - source_f32) ** 2).sum()))
                norm = float(np.sqrt((source_f32 ** 2).sum()))
                relative = difference / max(norm, 1e-30)
                if relative > fp8_relative_l2:
                    problems.append(
                        f"kind={kind} layer={layer} fp8 dequant relative_l2={relative:.4g} > {fp8_relative_l2}")
                else:
                    print(f"  trace kind={kind} layer={layer} fp8 relative_l2={relative:.4g}")
    return problems


def bf16_widen(packed):
    import numpy as np
    bits = packed.astype(np.uint32) << 16
    return bits.view(np.float32)


def source_matrix(source: SafetensorsSource, name: str):
    import numpy as np
    shard, meta, offset = source.resolve(name)
    elements = 1
    for extent in meta["shape"]:
        elements *= extent
    with (source.root / shard).open("rb") as file:
        file.seek(offset)
        raw = file.read(elements * 2)
    if len(raw) != elements * 2:
        # A transient mount hiccup must fail loudly, not degrade into a
        # misleading byte-comparison failure downstream.
        raise ValueError(f"short read on {name}: {len(raw)} of {elements * 2} bytes")
    return np.frombuffer(raw, dtype="<u2").reshape(meta["shape"])


def source_vector(source: SafetensorsSource, name: str):
    return source_matrix(source, name).reshape(-1)


def source_bytes(source: SafetensorsSource, name: str) -> bytes:
    """The tensor's raw bytes, sized from its own safetensors header."""
    shard, meta, offset = source.resolve(name)
    elements = 1
    for extent in meta["shape"]:
        elements *= extent
    dtype_bytes = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}.get(meta["dtype"])
    if dtype_bytes is None:
        raise PackFailure(f"{name}: unsupported byte-trace dtype {meta['dtype']}")
    want = elements * dtype_bytes
    if meta["shape"] in ([], [1]):
        want = dtype_bytes
    with (source.root / shard).open("rb") as file:
        file.seek(offset)
        raw = file.read(want)
    if len(raw) != want:
        raise ValueError(f"short read on {name}: {len(raw)} of {want} bytes")
    return raw


def expert_source_matrix(source: SafetensorsSource, ref, kind: int, layer: int):
    """The rank's expert slab as f32, widened from bf16 bits (u16 views must
    be widened, never .astype'd - that would reinterpret the bits as ints)."""
    import numpy as np
    expert_start, expert_count = getattr(ref, "expert_slice", (0, EXPERT_COUNT))
    if kind == 8:  # down
        matrix = source_matrix(source, layer_tensor_name(kind, layer))
        packed = matrix[expert_start:expert_start + expert_count].reshape(-1, EXPERT_INTERMEDIATE)
    else:
        gate_up = source_matrix(source, layer_tensor_name(6, layer))
        section = gate_up[expert_start:expert_start + expert_count]
        half = section.shape[1] // 2
        packed = (section[:, :half, :] if kind == 6 else section[:, half:, :]).reshape(-1, HIDDEN)
    return bf16_widen(np.ascontiguousarray(packed).reshape(-1)).reshape(packed.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--fp8-relative-l2", type=float, default=0.2)
    parser.add_argument("--no-mtp", action="store_true",
                        help="the pack was built --no-mtp (MTP tail absent)")
    args = parser.parse_args()

    header = read_pack_header(args.pack)
    pack_bytes = args.pack.stat().st_size
    if header["file_bytes"] != pack_bytes:
        print(f"FAIL file_bytes header={header['file_bytes']} actual={pack_bytes}", file=sys.stderr)
        return 1
    geometry = expected_header_geometry(header["first_layer_index"], header["layer_count"],
                                        header["tensor_count"], include_mtp=not args.no_mtp)
    problems = [f"header {key}={header[key]} expected {expected}"
                for key, expected in geometry.items() if header[key] != expected]
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    entries = read_entries(args.pack, header)
    problems = verify_directory(header, entries, args.tp_degree, args.tp_rank, pack_bytes, include_mtp=not args.no_mtp)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    source = SafetensorsSource(args.checkpoint)
    source.check_config()  # the qwen4_flash subclass pins text_config expectations
    problems = sample_trace(args.pack, entries, source, args.tp_degree, args.tp_rank,
                            args.sample, args.fp8_relative_l2)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    receipt_path = Path(str(args.pack) + ".receipt.json")
    receipt_note = ""
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        digest = hashlib.sha256(args.pack.read_bytes()).hexdigest()
        if receipt.get("output_sha256") != digest:
            print("FAIL receipt output_sha256 mismatch", file=sys.stderr)
            return 1
        receipt_note = " receipt=verified"
    print(f"PASS {args.pack.name}: header geometry, {len(entries)} directory entries "
          f"(tp {args.tp_degree}/{args.tp_rank}), {args.sample} byte-traced samples{receipt_note}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen4_flash_pack_verify: {error}", file=sys.stderr)
        sys.exit(1)
