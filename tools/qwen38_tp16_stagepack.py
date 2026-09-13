#!/usr/bin/env python3
"""Produce the 16 rank-local TP16xPP1 stage packs for Qwen3.8-2.4T-A95B.

Capacity-wall mitigation (docs/QWEN38_MAX_BRINGUP.md R1, audit option 1):
routed experts are requantized vendor-FP8 -> MXFP4-E2M1 (group-32 E8M0) and
SLICED across the TP degree (512/16 = 32 consecutive experts per rank);
every other tensor is emitted verbatim (BF16/F32) and REPLICATED into every
rank pack. Arithmetic footprint per rank: ~74 GiB expert share + the
replicated dense spine (~91 GiB today) => ~165 GiB/rank packs that close to
~99 GiB/rank once the attention/GDN head-slicing increment lands. These are
arithmetic projections, not measurements.

Wire discipline (the qwen38 stage-pack contract in
modules/qwen38_resident_decode_stage/source/
spark_qwen38_resident_decode_stage_internal.h, former
spark_qwen38_stagepack_format.h, is the truth):
  * header geometry stays CANONICAL (routed_expert_count 512, full shapes);
    sharding lives ONLY in the three routed-expert entries' row counts.
  * weight-format codes come from spark_qwen38_resident_decode_stage_firmware.h:
    BF16=0 F32=1 MXFP4_E2M1=3 FP8_E4M3_F32B128=4. NOTE: tools/qwen38_stagepack.py
    carries WEIGHT_MXFP4_E2M1=7 and an undefined copy_mxfp4_tensor() - both dead
    branches; this tool neither uses nor silently fixes them.

Loader dependency (do NOT skip): SparkQwen38ModuleValidateEntry currently
requires entry rows/columns == NATURAL shapes and refuses any routed-expert
format except natural FP8 or the synthesized-BF16 escape. Rank-local packs
therefore need the flagged family-local delta ("the module dispatches the
rank-local view", plan SSState) before S1/S2 can load them: accept, for
KIND_MOE_W1/W3/DOWN only, rows == (EXPERT_COUNT/tp_degree) * per_expert_rows
when SPARK_QWEN38_STAGE_TP_DEGREE/_TP_RANK are set, formats MXFP4_E2M1|FP8.
Until that lands, these artifacts validate structurally (--verify below) but
are not loadable - produce them, land the delta, then run the ladder.

Naming per docs/DATAFILE_NAMING.md rank-local shards:
  qwen38.<placement>.<precision>.rank<NN>.stagepack.<sha256>

Self-test: --self-test runs the vectorized MXFP4 codec against the family's
scalar quantize_mxfp4_e2m1 reference (byte equality) plus E4M3 LUT vectors;
no checkpoint needed. Every conversion also spot-checks sampled groups
against the scalar reference and aborts on mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from spark_pack_common import (  # noqa: E402
    PackFailure,
    align_up,
    sha256_file,
    write_receipt,
)
# Single source of truth for geometry/inventory/scalar-codec: the existing
# family packer. Nothing here re-declares a shape table.
from qwen38_stagepack import (  # noqa: E402
    ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTN_Q_DIM, ATTN_QUERY_HEADS, ATTN_ROPE_DIM,
    ATTENTION_PERIOD, BF16_BYTES, CHUNK_BYTES, ENTRY_BYTES, ENTRY_STRUCT,
    EXPERT_COUNT, EXPERT_INTERMEDIATE, EXPERTS_PER_TOKEN, FULL_PHASE,
    F32_BYTES, FORMAT_VERSION, GLOBAL_LAYER, GLOBAL_TENSORS, GDN_CONV_CHANNELS,
    GDN_CONV_KERNEL, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM, GDN_KEY_HEADS,
    GDN_VALUE_DIM, GDN_VALUE_HEADS, HEADER_BYTES, HEADER_STRUCT, HIDDEN,
    KIND_ATTN_KEY, KIND_ATTN_KEY_NORM, KIND_ATTN_OUTPUT, KIND_ATTN_QUERY,
    KIND_ATTN_QUERY_NORM, KIND_ATTN_VALUE, KIND_EMBEDDING, KIND_FINAL_NORM,
    KIND_GDN_A_LOG, KIND_GDN_BETA, KIND_GDN_CONV_WEIGHT, KIND_GDN_DECAY,
    KIND_GDN_DT_BIAS, KIND_GDN_GATE, KIND_GDN_NORM, KIND_GDN_QKV,
    KIND_GDN_OUTPUT, KIND_LM_HEAD, KIND_MLP_NORM, KIND_MOE_DOWN, KIND_MOE_W1,
    KIND_MOE_W3, KIND_ATTENTION_NORM, KIND_MOE_GATE, KIND_MOE_SHARED_DOWN,
    KIND_MOE_SHARED_GATE, KIND_MOE_SHARED_GATE_WEIGHT, KIND_MOE_SHARED_UP,
    KIND_MTP_EMBED_NORM, KIND_MTP_FC, KIND_MTP_FINAL_NORM, KIND_MTP_HIDDEN_NORM,
    LAYER_COUNT, MAGIC, MTP_LAYERS, MXFP4_GROUP, PAYLOAD_ALIGNMENT, VOCAB,
    SafetensorsSource, TensorRef, build_inventory, copy_bf16_tensor,
    expected_tensor_count,
    is_gdn_layer, quantize_mxfp4_e2m1,
)

DEFAULT_CONTRACT = Path(__file__).resolve().parents[1] / "model_contracts" / "qwen38_authoritative.json"

# Wire codes per spark_qwen38_resident_decode_stage_firmware.h (NOT the
# stale WEIGHT_MXFP4_E2M1=7 constant in qwen38_stagepack.py).
WIRE_BF16 = 0
WIRE_F32 = 1
WIRE_MXFP4_E2M1 = 3
WIRE_FP8_E4M3_F32B128 = 4

CODEC_LABELS = {"mxfp4_e2m1": "mxfp4-e2m1", "fp8_e4m3": "fp8-e4m3-b128"}
GB10_USABLE_GIB = 107.0  # audit section 4: ~119 unified, ~107 usable

MOE_EXPERT_KINDS = (KIND_MOE_W1, KIND_MOE_W3, KIND_MOE_DOWN)


class RankExpertRef:
    """A routed-expert TensorRef sliced to [expert_lo, expert_hi).

    Rows shrink to the LOCAL expert-major extent; columns, layer identity,
    checkpoint tensor-name pattern and the fused gate_up split are inherited
    unchanged. The pack header still declares the canonical 512-expert
    geometry - only these entries' rows carry the rank slice."""

    def __init__(self, ref: TensorRef, expert_lo: int, expert_hi: int):
        if ref.kind not in MOE_EXPERT_KINDS:
            raise PackFailure(f"expert slicing applies to MOE kinds, got kind {ref.kind}")
        self.ref = ref
        self.expert_lo = expert_lo
        self.expert_hi = expert_hi
        self.kind = ref.kind
        self.layer = ref.layer
        self.name = ref.name
        self.columns = ref.columns
        self.slice_start, self.slice_rows = ref.slice_start, ref.slice_rows
        per_expert = ref.rows // EXPERT_COUNT
        if (ref.rows % EXPERT_COUNT) != 0:
            raise PackFailure(f"kind {ref.kind}: rows not expert-major")
        self.per_expert_rows = per_expert
        self.rows = (expert_hi - expert_lo) * per_expert
        self.weight_format = ref.weight_format  # replaced per codec below


def make_rank_inventory(tp_degree: int, tp_rank: int, first_layer: int = 0,
                        layer_count: int = LAYER_COUNT) -> list:
    """Slice inventory over [first_layer, first_layer + layer_count) with the
    routed-expert kinds sliced to this rank's consecutive expert range."""
    if EXPERT_COUNT % tp_degree != 0 or not 1 <= tp_degree <= EXPERT_COUNT:
        raise PackFailure(f"tp_degree {tp_degree} must divide {EXPERT_COUNT}")
    span = EXPERT_COUNT // tp_degree
    if not 0 <= tp_rank < tp_degree:
        raise PackFailure(f"tp_rank {tp_rank} outside 0..{tp_degree - 1}")
    refs = build_inventory(first_layer, layer_count)
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} != {expected}")
    lo, hi = tp_rank * span, (tp_rank + 1) * span
    out: list = []
    expert_entries = 0
    for ref in refs:
        if ref.kind in MOE_EXPERT_KINDS:
            out.append(RankExpertRef(ref, lo, hi))
            expert_entries += 1
        else:
            out.append(ref)
    # Every in-slice layer carries the W1/W3/DOWN trio; a slice reaching the
    # stack end adds one more trio at the MTP marker.
    mtp_trios = 3 if first_layer + layer_count == LAYER_COUNT else 0
    if expert_entries != layer_count * 3 + mtp_trios:
        raise PackFailure(f"expert entries {expert_entries} != "
                          f"{layer_count * 3 + mtp_trios}")
    return out


def entry_byte_sizes(rows: int, columns: int, wire_format: int) -> tuple[int, int]:
    """Mirror SparkHybridStagePackPayloadBytes/ScaleBytes for the classes used."""
    if wire_format == WIRE_MXFP4_E2M1:
        return rows * columns // 2, rows * columns // MXFP4_GROUP
    if wire_format == WIRE_FP8_E4M3_F32B128:
        return rows * columns, (rows // 128) * (columns // 128) * F32_BYTES
    if wire_format == WIRE_F32:
        return rows * columns * F32_BYTES, 0
    return rows * columns * BF16_BYTES, 0


# -- vendor FP8 helpers ----------------------------------------------------------


def read_expert_fp8(source: SafetensorsSource, name: str, rows: int, columns: int):
    """Return (payload bytes, f64 dequantized tile or None) for one expert."""
    shard, meta, off = source.resolve(name)
    if meta["dtype"] != "F8_E4M3":
        raise PackFailure(f"{name}: dtype {meta['dtype']}, expected F8_E4M3")
    want = rows * columns
    with (source.root / shard).open("rb") as f:
        f.seek(off)
        raw = f.read(want)
    if len(raw) != want:
        raise PackFailure(f"short read on {name}")
    return raw


def read_scale_inv_f32(source: SafetensorsSource, name: str, rows: int, columns: int) -> bytes:
    """BF16 scale_inv plane widened to F32 row-major (multiplier semantics),
    exactly like tools/qwen38_stagepack.py::copy_fp8_experts."""
    scale_name = name + "_scale_inv"
    shard, meta, off = source.resolve(scale_name)
    if meta["dtype"] != "BF16":
        raise PackFailure(f"{scale_name}: dtype {meta['dtype']}, expected BF16")
    want = (rows // 128) * (columns // 128) * 2
    with (source.root / shard).open("rb") as f:
        f.seek(off)
        sraw = f.read(want)
    if len(sraw) != want:
        raise PackFailure(f"short read on {scale_name}")
    s16 = np.frombuffer(sraw, dtype="<u2").astype(np.uint32)
    return ((s16 << 16).astype(np.uint32)).view(np.float32).astype("<f4").tobytes()


_E4M3_LUT: np.ndarray | None = None


def e4m3_lut() -> np.ndarray:
    """256-entry F8_E4M3 (fn variant) decode table in float64."""
    global _E4M3_LUT
    if _E4M3_LUT is None:
        lut = np.zeros(256, dtype=np.float64)
        for b in range(256):
            sign = -1.0 if b & 0x80 else 1.0
            exp = (b >> 3) & 0xF
            man = b & 0x7
            if exp == 0:
                value = man * 2.0 ** -9          # denormal: man/8 * 2^-6
            elif exp == 15 and man == 7:
                value = math.nan                 # E4M3fn: no inf, NaN here
            else:
                value = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
            lut[b] = sign * value
        _E4M3_LUT = lut
    return _E4M3_LUT


# -- MXFP4-E2M1 conversion (vectorized twin of quantize_mxfp4_e2m1) --------------

# Midpoints between E2M1 magnitudes {0,.5,1,1.5,2,3}; a magnitude exactly on a
# midpoint rounds AWAY FROM ZERO (up), matching the scalar tie rule.
_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5)


def mxfp4_from_tile(tile_f64: np.ndarray) -> tuple[bytes, bytes]:
    """Quantize one expert tile (rows, columns) to (payload, e8m0 scales).

    Groups run along the contiguous axis in 32-element steps; payload packs
    two 4-bit values per byte, low nibble first. Byte-identical to feeding
    each row through the scalar quantize_mxfp4_e2m1 (spot-checked per call).
    """
    rows, columns = tile_f64.shape
    if columns % (2 * MXFP4_GROUP) != 0:
        raise PackFailure(f"row length {columns} not a multiple of {2 * MXFP4_GROUP}")
    groups = tile_f64.reshape(rows, columns // MXFP4_GROUP, MXFP4_GROUP)
    group_max = np.abs(groups).max(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.ceil(np.log2(group_max / 3.0))
    code = np.where(group_max > 0.0, np.clip(exponent + 127.0, 1.0, 254.0), 0.0)
    code = code.astype(np.int32)
    scale = np.power(2.0, (code - 127).astype(np.float64))
    q = groups / scale[:, :, None]
    nan = np.isnan(q)
    sign = (np.signbit(q) & ~nan).astype(np.uint8)
    mag = np.abs(q)
    mag[nan] = 0.0  # scalar path encodes NaN as +0.0
    index = np.zeros(mag.shape, dtype=np.uint8)
    for mid in _MIDS:
        index += (mag >= mid).astype(np.uint8)  # ties at mid go up: away from zero
    nibble = (sign << 3) | ((index >> 1) << 1) | (index & 1)
    payload = (nibble[:, :, 0::2] | (nibble[:, :, 1::2] << 4)).reshape(-1).tobytes()
    scales = code.astype(np.uint8).reshape(-1).tobytes()
    return payload, scales


def reference_check_tile(tile_f64: np.ndarray, payload: bytes, scales: bytes,
                         name: str, samples: int = 4) -> None:
    """Re-quantize sampled whole rows through the SCALAR family codec and
    demand byte equality with the vectorized output."""
    rows, columns = tile_f64.shape
    step = max(1, rows // max(1, samples))
    for row in range(0, rows, step):
        values = [float(v) for v in tile_f64[row]]
        ref_payload, ref_scales = quantize_mxfp4_e2m1(values)
        got_p = payload[(row * columns) // 2:(row * columns + columns) // 2]
        got_s = scales[(row * columns) // MXFP4_GROUP:
                       (row * columns + columns) // MXFP4_GROUP]
        if bytes(got_p) != bytes(ref_payload) or bytes(got_s) != bytes(ref_scales):
            raise PackFailure(f"{name}: vectorized MXFP4 diverges from scalar "
                              f"reference on row {row} - refusing to pack")


# -- rank writers -----------------------------------------------------------------


def copy_dense_tensor(source: SafetensorsSource, ref, offset: int, out) -> None:
    copy_bf16_tensor(source, ref, offset, out)


def write_expert_slice(source: SafetensorsSource, ref: RankExpertRef, out,
                       codec: str, check_reference: bool) -> None:
    per_rows, columns = ref.per_expert_rows, ref.columns
    payload = bytearray(ref.rows * columns) if codec == "fp8_e4m3" \
        else bytearray(ref.rows * columns // 2)
    scales = bytearray((ref.rows // 128) * (columns // 128) * 4 if codec == "fp8_e4m3"
                       else ref.rows * columns // MXFP4_GROUP)
    for e in range(ref.expert_lo, ref.expert_hi):
        local = e - ref.expert_lo
        name = ref.name.replace("{e}", str(e))
        raw = read_expert_fp8(source, name, per_rows, columns)
        if codec == "fp8_e4m3":
            base = local * per_rows * columns
            payload[base:base + len(raw)] = raw
            s32 = read_scale_inv_f32(source, name, per_rows, columns)
            sbase = local * (per_rows // 128) * (columns // 128) * 4
            scales[sbase:sbase + len(s32)] = s32
        else:
            tile = e4m3_lut()[np.frombuffer(raw, dtype=np.uint8)].reshape(per_rows, columns)
            s16 = np.frombuffer(read_scale_inv_f32(source, name, per_rows, columns),
                                dtype="<f4").reshape(per_rows // 128, columns // 128)
            multiplier = np.repeat(np.repeat(s16, 128, axis=0), 128, axis=1).astype(np.float64)
            tile = tile * multiplier
            part_payload, part_scales = mxfp4_from_tile(tile)
            if check_reference:
                reference_check_tile(tile, part_payload, part_scales, name)
            pbase = local * per_rows * columns // 2
            payload[pbase:pbase + len(part_payload)] = part_payload
            sbase = local * per_rows * columns // MXFP4_GROUP
            scales[sbase:sbase + len(part_scales)] = part_scales
    out.write(payload)
    out.write(scales)


# -- plan / convert ----------------------------------------------------------------


def plan_rank(source: SafetensorsSource | None, refs: list, codec: str) -> dict:
    wire = WIRE_MXFP4_E2M1 if codec == "mxfp4_e2m1" else WIRE_FP8_E4M3_F32B128
    cursor = 0
    plans = []
    for ref in refs:
        rows = ref.rows
        columns = ref.columns
        fmt = wire if isinstance(ref, RankExpertRef) else ref.weight_format
        payload_bytes, scale_bytes = entry_byte_sizes(rows, columns, fmt)
        payload_offset = align_up(cursor, PAYLOAD_ALIGNMENT)
        plans.append((ref, fmt, payload_offset, payload_bytes, scale_bytes))
        cursor = payload_offset + payload_bytes + scale_bytes
    payload_base = align_up(HEADER_BYTES + len(plans) * ENTRY_BYTES, PAYLOAD_ALIGNMENT)
    expert_bytes = sum(p[3] + p[4] for p in plans if isinstance(p[0], RankExpertRef))
    dense_bytes = sum(p[3] + p[4] for p in plans if not isinstance(p[0], RankExpertRef))
    shardable_dense = 0
    for ref, fmt, _, pb, sb in plans:
        if isinstance(ref, RankExpertRef):
            continue
        if ref.kind in (KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY,
                        KIND_GDN_OUTPUT, KIND_ATTN_QUERY, KIND_ATTN_KEY,
                        KIND_ATTN_VALUE, KIND_ATTN_OUTPUT, KIND_LM_HEAD,
                        KIND_EMBEDDING):
            shardable_dense += pb + sb
    return {
        "plans": plans, "payload_base": payload_base, "cursor": cursor,
        "file_bytes": payload_base + cursor, "tensor_count": len(plans),
        "expert_bytes": expert_bytes, "dense_bytes": dense_bytes,
        "shardable_dense_bytes": shardable_dense,
    }


def print_plan(rank: int, tp_degree: int, codec: str, plan: dict) -> None:
    gib = lambda b: b / 2**30
    expert_share_all_ranks = plan["expert_bytes"] * tp_degree
    dense_end_state = plan["dense_bytes"] - plan["shardable_dense_bytes"] \
        + plan["shardable_dense_bytes"] / tp_degree
    print(f"qwen38_tp16_stagepack rank={rank} codec={codec} "
          f"tensors={plan['tensor_count']} file_gib={gib(plan['file_bytes']):.2f}")
    print(f"  expert_share_this_rank_gib={gib(plan['expert_bytes']):.2f} "
          f"(x{tp_degree} ranks = {gib(expert_share_all_ranks):.2f} GiB, must equal "
          f"whole-model expert payload)")
    print(f"  dense_replicated_this_rank_gib={gib(plan['dense_bytes']):.2f} "
          f"(ARITHMETIC: head/GDN channel slicing would cut this to "
          f"~{gib(dense_end_state):.2f}; kernels are a separate increment)")
    today = gib(plan['file_bytes'])
    print(f"  projected_residency_today_gib={today:.2f} vs usable {GB10_USABLE_GIB:.0f} "
          f"-> {'FITS' if today <= GB10_USABLE_GIB else 'DOES NOT FIT'} "
          f"(projection, not measurement)")


def convert_rank(checkpoint: Path, output_dir: Path, tp_degree: int, tp_rank: int,
                 codec: str, contract: Path | None, dry_run: bool,
                 check_reference: bool = True) -> dict:
    source = SafetensorsSource(checkpoint)
    source.check_config()  # subclass adds the qwen38 expectation table
    refs = make_rank_inventory(tp_degree, tp_rank)
    plan = plan_rank(source, refs, codec)
    receipt = {
        "kind": "sparkpipe.qwen38.tp16-stagepack-receipt.v1",
        "tool": "tools/qwen38_tp16_stagepack.py",
        "checkpoint": str(checkpoint),
        "source_index_sha256": source.index_sha256,
        "source_config_sha256": source.config_sha256,
        "contract_sha256": sha256_file(contract) if contract and contract.is_file() else None,
        "topology": {"tp_degree": tp_degree, "pp_stage_count": 1,
                     "world_ranks": tp_degree, "tp_rank": tp_rank},
        "expert_range": [next(r for r in refs if isinstance(r, RankExpertRef)).expert_lo,
                         next(r for r in refs if isinstance(r, RankExpertRef)).expert_hi],
        "experts_per_rank": EXPERT_COUNT // tp_degree,
        "codec": codec,
        "wire_weight_formats": {"routed_experts": CODEC_LABELS[codec],
                                "non_expert": "bf16",
                                "gdn_a_log_dt_bias": "f32"},
        "dense_replication": "full (head/channel slicing is a separate module increment)",
        "module_loader_note": "requires the flagged ValidateEntry rank-local-view delta "
                              "before load; see docs/QWEN38_MAX_BRINGUP.md",
        "layer_count": LAYER_COUNT,
        "first_layer_index": 0,
        "tensor_count": plan["tensor_count"],
    }
    if dry_run:
        print_plan(tp_rank, tp_degree, codec, plan)
        return receipt
    wire = WIRE_MXFP4_E2M1 if codec == "mxfp4_e2m1" else WIRE_FP8_E4M3_F32B128
    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, plan["tensor_count"],
        HIDDEN, LAYER_COUNT, 0, LAYER_COUNT,
        ATTENTION_PERIOD, FULL_PHASE,
        GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
        GDN_CONV_KERNEL, ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM,
        ATTN_ROPE_DIM, EXPERT_COUNT, EXPERTS_PER_TOKEN, EXPERT_INTERMEDIATE,
        VOCAB, MXFP4_GROUP, MTP_LAYERS,
        HEADER_BYTES, plan["payload_base"] + plan["cursor"])
    entries = b"".join(
        ENTRY_STRUCT.pack(
            ref.kind, ref.layer, fmt, ref.rows, ref.columns,
            (MXFP4_GROUP if fmt == WIRE_MXFP4_E2M1 else
             128 if fmt == WIRE_FP8_E4M3_F32B128 else 0),
            plan["payload_base"] + payload_offset, payload_bytes,
            plan["payload_base"] + payload_offset + payload_bytes if scale_bytes else 0,
            scale_bytes)
        for ref, fmt, payload_offset, payload_bytes, scale_bytes in plan["plans"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".tp16.rank{tp_rank}.", suffix=".tmp",
                                     dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as temp:
            temp.write(header)
            temp.write(entries)
            padding = plan["payload_base"] - temp.tell()
            if padding < 0:
                raise PackFailure("directory overruns the payload base")
            temp.write(b"\0" * padding)
            for ref, fmt, source_offset_hint, payload_offset, payload_bytes, scale_bytes in \
                    [(r, f, 0, po, pb, sb) for r, f, po, pb, sb in plan["plans"]]:
                before = temp.tell()
                if temp.tell() != plan["payload_base"] + payload_offset:
                    temp.seek(plan["payload_base"] + payload_offset)
                if isinstance(ref, RankExpertRef):
                    write_expert_slice(source, ref, temp, codec, check_reference)
                else:
                    copy_dense_tensor(source, ref, source_offset_hint, temp)
                wrote = temp.tell() - before
                if wrote != payload_bytes + scale_bytes:
                    raise PackFailure(f"payload size mismatch kind={ref.kind} "
                                      f"layer={ref.layer}: {wrote} != {payload_bytes + scale_bytes}")
                pad = align_up(temp.tell(), PAYLOAD_ALIGNMENT) - temp.tell()
                if pad:
                    temp.write(b"\0" * pad)
            temp.flush()
            os.fsync(temp.fileno())
    except BaseException:
        os.unlink(temp_name)
        raise
    digest = sha256_file(Path(temp_name))
    precision = CODEC_LABELS[codec]
    final = output_dir / f"qwen38.tp16.{precision}.rank{tp_rank:02d}.stagepack.{digest}"
    os.chmod(temp_name, 0o644)
    os.replace(temp_name, final)
    receipt.update({"file": str(final), "output_sha256": digest, "bytes": plan["file_bytes"]})
    write_receipt(receipt, final, suffix=".receipt.json")
    print(f"qwen38_tp16_stagepack rank={tp_rank} wrote {final} "
          f"({plan['file_bytes'] / 2**30:.2f} GiB)")
    return receipt


def aggregate_manifest(output_dir: Path, codec: str, tp_degree: int) -> Path:
    precision = CODEC_LABELS[codec]
    ranks = {}
    for rank in range(tp_degree):
        matches = sorted(output_dir.glob(f"qwen38.tp16.{precision}.rank{rank:02d}.stagepack.*"))
        matches = [m for m in matches if not m.name.endswith(".receipt.json")]
        if len(matches) != 1:
            raise PackFailure(f"rank {rank}: expected exactly one published pack, found {matches}")
        pack = matches[0]
        receipt_path = Path(str(pack) + ".receipt.json")
        if not receipt_path.is_file():
            raise PackFailure(f"missing receipt {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("output_sha256") != pack.name.rsplit(".", 1)[-1]:
            raise PackFailure(f"receipt sha mismatch for {pack}")
        ranks[rank] = {"file": pack.name, "sha256": receipt["output_sha256"],
                       "bytes": receipt["bytes"],
                       "expert_range": receipt["expert_range"]}
    covered = sorted(r["expert_range"][0] for r in ranks.values())
    if covered != list(range(0, EXPERT_COUNT, EXPERT_COUNT // tp_degree)):
        raise PackFailure("expert ranges do not partition 512 exactly once")
    manifest = {
        "kind": "sparkpipe.qwen38.tp16-pack-manifest.v1",
        "model": "qwen38", "placement": "tp16", "precision": precision,
        "codec": codec, "tp_degree": tp_degree,
        "total_bytes": sum(r["bytes"] for r in ranks.values()),
        "ranks": ranks,
        "loader_dependency": "ValidateEntry rank-local-view delta (bring-up runbook R2/R3)",
    }
    blob = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    fd, temp_name = tempfile.mkstemp(prefix=".tp16.manifest.", suffix=".tmp",
                                     dir=output_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    import hashlib
    digest = hashlib.sha256(blob).hexdigest()
    final = output_dir / f"qwen38.tp16.{precision}.manifest.{digest}"
    os.replace(temp_name, final)
    print(f"qwen38_tp16_stagepack manifest {final}")
    return final


def fabricate_rank(output_dir: Path, tp_degree: int, tp_rank: int, codec: str,
                   contract: Path | None, first_layer: int = 0,
                   layer_count: int = LAYER_COUNT) -> Path:
    """Emit a FULL-GEOMETRY rank pack without any checkpoint: header and
    directory come from the SAME inventory/plan/entry code paths as a real
    pack; every payload byte is an implicit zero (one ftruncate - the file
    is sparse, so ~165 GiB apparent costs kilobytes on disk). Zero weights
    are mechanically valid MXFP4 (code 0) and BF16, which is exactly what
    loader-delta validation needs: structure, not numerics. Receipt marks
    the pack synthetic so it can never masquerade as produced weights."""
    refs = make_rank_inventory(tp_degree, tp_rank, first_layer, layer_count)
    plan = plan_rank(None, refs, codec)
    wire = WIRE_MXFP4_E2M1 if codec == "mxfp4_e2m1" else WIRE_FP8_E4M3_F32B128
    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, plan["tensor_count"],
        HIDDEN, layer_count, first_layer, LAYER_COUNT,
        ATTENTION_PERIOD, FULL_PHASE,
        GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
        GDN_CONV_KERNEL, ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM,
        ATTN_ROPE_DIM, EXPERT_COUNT, EXPERTS_PER_TOKEN, EXPERT_INTERMEDIATE,
        VOCAB, MXFP4_GROUP, MTP_LAYERS,
        HEADER_BYTES, plan["payload_base"] + plan["cursor"])
    entries = b"".join(
        ENTRY_STRUCT.pack(
            ref.kind, ref.layer, fmt, ref.rows, ref.columns,
            (MXFP4_GROUP if fmt == WIRE_MXFP4_E2M1 else
             128 if fmt == WIRE_FP8_E4M3_F32B128 else 0),
            plan["payload_base"] + payload_offset, payload_bytes,
            plan["payload_base"] + payload_offset + payload_bytes if scale_bytes else 0,
            scale_bytes)
        for ref, fmt, payload_offset, payload_bytes, scale_bytes in plan["plans"])
    file_bytes = plan["payload_base"] + plan["cursor"]
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".tp16.fabricated.rank{tp_rank}.",
                                     suffix=".tmp", dir=output_dir)
    with os.fdopen(fd, "wb") as temp:
        temp.write(header)
        temp.write(entries)
        padding = plan["payload_base"] - temp.tell()
        if padding < 0:
            raise PackFailure("directory overruns the payload base")
        temp.write(b"\\0" * padding)
        temp.flush()
        os.ftruncate(temp.fileno(), file_bytes)   # everything past dir = holes
        os.fsync(temp.fileno())
    digest = sha256_file(Path(temp_name))
    precision = CODEC_LABELS[codec]
    final = output_dir / f"qwen38.tp16.{precision}.rank{tp_rank:02d}.stagepack.{digest}"
    os.chmod(temp_name, 0o644)
    os.replace(temp_name, final)
    receipt = {
        "kind": "sparkpipe.qwen38.tp16-stagepack-receipt.v1",
        "tool": "tools/qwen38_tp16_stagepack.py",
        "synthetic_zero_filled": True,
        "synthetic_note": "fabricated for loader-delta testing; NOT produced weights",
        "topology": {"tp_degree": tp_degree, "pp_stage_count": 1,
                     "world_ranks": tp_degree, "tp_rank": tp_rank},
        "expert_range": [next(r for r in refs if isinstance(r, RankExpertRef)).expert_lo,
                         next(r for r in refs if isinstance(r, RankExpertRef)).expert_hi],
        "codec": codec,
        "layer_count": layer_count,
        "first_layer_index": first_layer,
        "tensor_count": plan["tensor_count"],
        "bytes": file_bytes,
        "file": str(final),
        "output_sha256": digest,
        "contract_sha256": sha256_file(contract) if contract and contract.is_file() else None,
    }
    write_receipt(receipt, final, suffix=".receipt.json")
    print(f"qwen38_tp16_stagepack fabricated rank={tp_rank} wrote {final} "
          f"({file_bytes / 2**30:.2f} GiB apparent, sparse)")
    return final


def self_test() -> None:
    rng = np.random.default_rng(20260823)
    lut = e4m3_lut()
    vectors = {
        # denormals: man * 2^-9
        0x00: 0.0, 0x80: -0.0, 0x01: 2.0 ** -9, 0x81: -(2.0 ** -9), 0x07: 7 * 2.0 ** -9,
        # normals: (1 + man/8) * 2^(exp-7)
        0x08: 2.0 ** -6, 0x30: 0.5, 0xB0: -0.5, 0x38: 1.0,
        0x34: 0.75, 0xB4: -0.75, 0x70: 128.0,
        0x7E: 448.0, 0xFE: -448.0,
        # E4M3fn NaN slot (exp 15, man 7; there is no inf)
        0x7F: math.nan, 0xFF: math.nan,
    }
    for byte, want in vectors.items():
        got = float(lut[byte])
        if math.isnan(want):
            assert math.isnan(got), f"LUT {byte:#04x}"
        else:
            assert got == want, f"LUT {byte:#04x}: got {got}, want {want}"
    for trial in range(24):
        rows = int(rng.integers(1, 5)) * 2          # multiple of 2 rows x 64 cols
        columns = 64 * int(rng.integers(1, 9))
        tile = rng.normal(0.0, 10.0 ** rng.integers(-3, 4), size=(rows, columns))
        # inject adversarial values: zeros, exact midpoints, huge/tiny
        tile[rng.integers(0, rows), rng.integers(0, columns)] = 0.0
        tile[rng.integers(0, rows), rng.integers(0, columns)] = 0.75
        tile[rng.integers(0, rows), rng.integers(0, columns)] = 1e30
        tile[rng.integers(0, rows), rng.integers(0, columns)] = 1e-30
        payload, scales = mxfp4_from_tile(tile)
        reference_check_tile(tile, payload, scales, f"self-test-{trial}",
                             samples=rows)
    print("qwen38_tp16_stagepack self-test PASS "
          "(vectorized MXFP4 == scalar quantize_mxfp4_e2m1, E4M3 LUT vectors ok)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="vendor FP8 safetensors directory")
    parser.add_argument("--output-dir", type=Path, help="directory for rank packs + receipts")
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--rank", type=int, help="TP rank to pack (0-based)")
    parser.add_argument("--expert-codec", choices=sorted(CODEC_LABELS),
                        help="mxfp4_e2m1 = capacity mitigation (quality-policy override, "
                             "explicit choice); fp8_e4m3 = vendor passthrough, sliced. "
                             "Required for packing and manifest aggregation")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument("--no-reference-check", action="store_true",
                        help="skip per-expert scalar-reference spot checks (not recommended)")
    parser.add_argument("--aggregate-manifest", action="store_true",
                        help="bind all published rank receipts into the placement manifest")
    parser.add_argument("--self-test", action="store_true",
                        help="codec equivalence test against the scalar reference; no I/O")
    parser.add_argument("--fabricate-first-layer", type=int, default=0,
                        help="slice start for --fabricate-rank (default full stack)")
    parser.add_argument("--fabricate-layer-count", type=int, default=None,
                        help="slice length for --fabricate-rank (default all layers)")
    parser.add_argument("--fabricate-rank", type=int, metavar="RANK", default=None,
                        help="emit a full-geometry SYNTHETIC rank pack (zero-filled "
                             "sparse payloads, no checkpoint needed) for loader-delta tests")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.fabricate_rank is not None:
        if not args.output_dir:
            parser.error("--output-dir is required with --fabricate-rank")
        if not args.expert_codec or not (0 <= args.fabricate_rank < args.tp_degree):
            parser.error("--fabricate-rank needs --expert-codec and rank within tp-degree")
        layer_count = args.fabricate_layer_count
        if layer_count is None:
            layer_count = LAYER_COUNT - args.fabricate_first_layer
        if layer_count <= 0 or args.fabricate_first_layer + layer_count > LAYER_COUNT:
            parser.error("fabricated slice escapes the 92-layer stack")
        fabricate_rank(args.output_dir, args.tp_degree, args.fabricate_rank,
                       args.expert_codec, args.contract,
                       args.fabricate_first_layer, layer_count)
        return 0
    if args.aggregate_manifest:
        if not args.output_dir:
            parser.error("--output-dir is required with --aggregate-manifest")
        aggregate_manifest(args.output_dir, args.expert_codec, args.tp_degree)
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.rank is None:
        parser.error("--rank is required (or --self-test / --aggregate-manifest)")
    result = convert_rank(args.checkpoint, args.output_dir or args.checkpoint,
                          args.tp_degree, args.rank, args.expert_codec, args.contract,
                          args.dry_run, not args.no_reference_check)
    if args.dry_run:
        print("qwen38_tp16_stagepack dry run complete (nothing written)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen38_tp16_stagepack: {error}", file=sys.stderr)
        sys.exit(1)
