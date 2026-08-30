#!/usr/bin/env python3
"""Verify a qwen38 stage pack against the wire format AND the live checkpoint.

This is the pack gate for the qwen38_max lane (the module ships no pack
validation harness of its own; per docs/AGENT_LANE_BRIEFS/pack_agent_rules.md
"run the pack verifier. Exit must be PASS before deploy").

Three layers of checking, all from raw bytes:

  1. STRUCTURE - header magic/version/geometry vs the constants the module
     enforces at load (spark_qwen38_max_stagepack_format.h), directory entry
     count vs the format's computed inventory, per-entry shape/format/scale
     rules, payload alignment and bounds, no duplicate or missing
     (kind, layer) pair.
  2. CONTENT - every entry's payload+scale bytes are hashed out of the pack
     file and compared against the byte stream the source checkpoint must
     produce (BF16 pass-through, BF16->F32 widening for A_log/dt_bias,
     per-expert F8_E4M3 stacking + BF16->F32 scale_inv planes for the MoE).
     This covers every payload byte in the file; only inter-entry alignment
     padding is not content-checked.
  3. RECEIPT - tensor/byte counts, slice identity and the packer's
     hash-while-write output_sha256 are cross-checked. --recompute-file-hash
     re-reads the whole file to recompute that digest independently (one
     extra pass over the pack; off by default because warm-storage reads can
     be orders of magnitude slower than the compare pass).

Exit 0 only on PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

# The 27B verifier: expected shapes come from the 27B packer's own
# tables (HIDDEN 5120, GDN 48 value heads — the release geometry). The
# max-family tables (qwen38_stagepack.py, HIDDEN 8192) describe a
# DIFFERENT model; using them here failed every valid 27B entry.
_spec = importlib.util.spec_from_file_location(
    "qwen38_stagepack_tables", str(Path(_TOOLS_DIR) / "qwen38_27b_stagepack.py"))
_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tables)

# The 27B packer owns build_tp_plan/packed_shape (the TP shard shapes);
# the tables module is the shared inventory only.
_packer_spec = importlib.util.spec_from_file_location(
    "qwen38_27b_stagepack_mod", str(Path(_TOOLS_DIR) / "qwen38_27b_stagepack.py"))
_packer = importlib.util.module_from_spec(_packer_spec)
_packer_spec.loader.exec_module(_packer)

from spark_pack_common import sha256_file  # noqa: E402

HASH_CHUNK = 16 * 1024 * 1024

# Wire constants, mirroring spark_qwen38_max_stagepack_format.h (and the
# packer). Any drift here is exactly what this verifier exists to catch.
MAGIC = 0x50533851
FORMAT_VERSION = 1
HEADER_BYTES = 120
ENTRY_BYTES = 56
PAYLOAD_ALIGNMENT = 256
HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_F32B128 = 4

GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE

BF16_BYTES = 2
F32_BYTES = 4


def payload_bytes_for(weight_format: int, rows: int, columns: int) -> int:
    elements = rows * columns
    if weight_format == WEIGHT_FP8_F32B128:
        return elements
    return elements * (BF16_BYTES if weight_format == WEIGHT_BF16 else F32_BYTES)


def scale_bytes_for(weight_format: int, rows: int, columns: int) -> int:
    if weight_format == WEIGHT_FP8_F32B128:
        return (rows // 128) * (columns // 128) * F32_BYTES
    return 0


def natural_format(kind: int) -> int:
    if kind in (_tables.KIND_MOE_W1, _tables.KIND_MOE_W3, _tables.KIND_MOE_DOWN):
        return WEIGHT_FP8_F32B128
    if kind in (_tables.KIND_GDN_A_LOG, _tables.KIND_GDN_DT_BIAS):
        return WEIGHT_F32
    return WEIGHT_BF16


def hash_source_entry(source, ref) -> str:
    """sha256 of the byte stream the checkpoint must yield for this tensor.

    The pack layout is: the full expert-major payload (all 512 experts
    stacked), THEN the whole F32 scale plane (expert 0's plane .. expert
    511's). The expected stream must follow that order, not per-expert
    interleaved payload+scale."""
    digest = hashlib.sha256()
    if ref.weight_format == WEIGHT_FP8_F32B128:
        import numpy as np
        experts = _tables.EXPERT_COUNT
        rows_per_expert = ref.rows // experts
        scale_rows = rows_per_expert // 128
        scale_cols = ref.columns // 128
        names = [ref.name.replace("{e}", str(e)) for e in range(experts)]
        for name in names:
            shard, _, off = source.resolve(name)
            with (source.root / shard).open("rb") as f:
                f.seek(off)
                remaining = rows_per_expert * ref.columns
                while remaining > 0:
                    step = min(remaining, HASH_CHUNK)
                    chunk = f.read(step)
                    if len(chunk) != step:
                        raise RuntimeError(f"short source read on {name}")
                    digest.update(chunk)
                    remaining -= step
        for name in names:
            scale_name = name + "_scale_inv"
            s_shard, _, s_off = source.resolve(scale_name)
            with (source.root / s_shard).open("rb") as f:
                f.seek(s_off)
                sraw = f.read(scale_rows * scale_cols * 2)
            if len(sraw) != scale_rows * scale_cols * 2:
                raise RuntimeError(f"short source read on {scale_name}")
            s16 = np.frombuffer(sraw, dtype="<u2").astype(np.uint32)
            digest.update(((s16 << 16).astype(np.uint32)).view(np.float32)
                          .astype("<f4").tobytes())
        return digest.hexdigest()
    # BF16 pass-through or BF16->F32 widening
    path = source.root / source.weight_map[ref.name]
    elements = ref.rows * ref.columns
    source_bytes = elements * BF16_BYTES
    with path.open("rb") as file:
        file.seek(source.resolve(ref.name)[2])
        remaining = source_bytes
        while remaining > 0:
            step = min(remaining, HASH_CHUNK)
            chunk = file.read(step)
            if len(chunk) != step:
                raise RuntimeError(f"short source read on {ref.name}")
            remaining -= step
            if ref.weight_format == WEIGHT_BF16:
                digest.update(chunk)
            else:
                widened = bytearray(step * 2)
                widened[2::4] = chunk[0::2]
                widened[3::4] = chunk[1::2]
                digest.update(widened)
    return digest.hexdigest()


def verify(pack: Path, checkpoint: Path | None, receipt_path: Path | None,
           recompute_file_hash: bool, tp_degree: int = 1) -> tuple[bool, dict]:
    findings: list[str] = []

    def fail(message: str) -> None:
        findings.append(message)

    file_bytes_actual = pack.stat().st_size
    tensor_count = 0
    first_layer = layer_count = -1
    entries: list[tuple] = []
    file_sha = None
    source = None

    with pack.open("rb") as f:
        raw_header = f.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            return False, {"verdict": "FAIL", "pack": str(pack),
                           "errors": [f"header truncated: {len(raw_header)} bytes"]}
        header = HEADER_STRUCT.unpack(raw_header)
        (magic, version, header_bytes, entry_bytes, tensor_count, hidden,
         layer_count, first_layer, total_layers, period, full_phase,
         gdn_kh, gdn_vh, gdkd, gdvd, conv_k, qh, kvh, hd, rope_d,
         experts, topk, moe_int, vocab, mxfp4_group, mtp_count,
         directory_offset, file_bytes) = header

        def want(field: str, got, expected) -> None:
            if got != expected:
                fail(f"header {field}={got}, expected {expected}")

        want("magic", magic, MAGIC)
        want("format_version", version, FORMAT_VERSION)
        want("header_bytes", header_bytes, HEADER_BYTES)
        want("directory_entry_bytes", entry_bytes, ENTRY_BYTES)
        want("hidden_dimension", hidden, _tables.HIDDEN)
        want("total_layer_count", total_layers, _tables.LAYER_COUNT)
        want("attention_period", period, _tables.ATTENTION_PERIOD)
        want("full_attention_phase", full_phase, _tables.FULL_PHASE)
        want("gdn_key_head_count", gdn_kh, _tables.GDN_KEY_HEADS)
        want("gdn_value_head_count", gdn_vh, _tables.GDN_VALUE_HEADS)
        want("gdn_head_key_dimension", gdkd, _tables.GDN_HEAD_KEY_DIM)
        want("gdn_head_value_dimension", gdvd, _tables.GDN_HEAD_VALUE_DIM)
        want("gdn_conv_kernel", conv_k, _tables.GDN_CONV_KERNEL)
        want("attn_query_head_count", qh, _tables.ATTN_QUERY_HEADS)
        want("attn_kv_head_count", kvh, _tables.ATTN_KV_HEADS)
        want("attn_head_dimension", hd, _tables.ATTN_HEAD_DIM)
        want("attn_rope_dimension", rope_d, _tables.ATTN_ROPE_DIM)
        want("routed_expert_count", experts, _tables.EXPERT_COUNT)
        want("experts_per_token", topk, _tables.EXPERTS_PER_TOKEN)
        want("expert_intermediate_dimension", moe_int, _tables.EXPERT_INTERMEDIATE)
        want("output_vocab_count", vocab, _tables.VOCAB)
        want("mxfp4_group_size", mxfp4_group, _tables.MXFP4_GROUP)
        want("mtp_layer_count", mtp_count, _tables.MTP_LAYERS)
        want("directory_offset", directory_offset, HEADER_BYTES)
        want("file_bytes", file_bytes, file_bytes_actual)
        if layer_count <= 0 or first_layer < 0 or first_layer + layer_count > _tables.LAYER_COUNT:
            fail(f"invalid slice {first_layer}+{layer_count} of {_tables.LAYER_COUNT}")

        expected_count = _tables.expected_tensor_count(first_layer, layer_count)
        if tensor_count != expected_count:
            fail(f"tensor_count={tensor_count}, format inventory expects {expected_count}")

        expected_refs: dict[tuple[int, int], object] = {}
        try:
            for ref in _tables.build_inventory(first_layer, layer_count):
                if tp_degree > 1:
                    plan = _packer.build_tp_plan(ref, tp_degree, 0)
                    if plan is not None:
                        srows, scols = _packer.packed_shape(ref, plan)
                        if (srows, scols) != (ref.rows, ref.columns):
                            # remember both; the entry may carry either the
                            # shard or the replicated source shape
                            ref.alt_shape = (srows, scols)
                expected_refs[(ref.kind, ref.layer)] = ref
        except _tables.PackFailure as error:
            fail(f"inventory build failed: {error}")

        raw_dir = f.read(tensor_count * ENTRY_BYTES)
        if len(raw_dir) != tensor_count * ENTRY_BYTES:
            fail("directory truncated")

        seen: set[tuple[int, int]] = set()
        for index in range(tensor_count):
            entry = ENTRY_STRUCT.unpack_from(raw_dir, index * ENTRY_BYTES)
            (kind, layer, fmt, rows, cols, scale_group, p_off, p_bytes,
             s_off, s_bytes) = entry
            tag = f"entry[{index}] kind={kind} layer={hex(layer)}"
            if not 0 <= kind < 32:
                fail(f"{tag}: kind out of range")
                continue
            key = (kind, layer)
            if key in seen:
                fail(f"{tag}: duplicate (kind, layer)")
            seen.add(key)
            ref = expected_refs.get(key)
            if ref is None:
                fail(f"{tag}: not in the format inventory of slice {first_layer}+{layer_count}")
                continue
            alt = getattr(ref, "alt_shape", None)
            shape_ok = (rows == ref.rows and cols == ref.columns) or                        (alt is not None and (rows, cols) == alt)
            if not shape_ok:
                fail(f"{tag}: shape {rows}x{cols}, expected {ref.rows}x{ref.columns}")
            natural = natural_format(kind)
            if fmt != natural:
                fail(f"{tag}: weight_format={fmt}, natural format is {natural}")
            want_group = 128 if fmt == WEIGHT_FP8_F32B128 else 0
            if scale_group != want_group:
                fail(f"{tag}: scale_group_size={scale_group}, expected {want_group}")
            if p_bytes != payload_bytes_for(fmt, rows, cols):
                fail(f"{tag}: payload_bytes={p_bytes}, format math says {payload_bytes_for(fmt, rows, cols)}")
            if s_bytes != scale_bytes_for(fmt, rows, cols):
                fail(f"{tag}: scale_bytes={s_bytes}, format math says {scale_bytes_for(fmt, rows, cols)}")
            if p_off % PAYLOAD_ALIGNMENT != 0:
                fail(f"{tag}: payload_offset {p_off} not {PAYLOAD_ALIGNMENT}-aligned")
            if s_bytes and s_off % PAYLOAD_ALIGNMENT != 0:
                fail(f"{tag}: scale_offset {s_off} not {PAYLOAD_ALIGNMENT}-aligned")
            if p_off + p_bytes > file_bytes_actual:
                fail(f"{tag}: payload region overruns file")
            if s_bytes and s_off + s_bytes > file_bytes_actual:
                fail(f"{tag}: scale region overruns file")
            entries.append((ref, p_off, p_bytes, s_off, s_bytes))

        for key in sorted(set(expected_refs) - seen):
            fail(f"missing tensor kind={key[0]} layer={hex(key[1])}")

    verdict = {
        "verdict": "FAIL",
        "pack": str(pack),
        "file_bytes": file_bytes_actual,
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": tensor_count,
        "errors": findings,
    }
    if findings:
        return False, verdict

    if checkpoint is not None:
        # -- content: source-vs-pack byte equality per entry ------------------
        source = _tables.SafetensorsSource(checkpoint)
        source.check_config()
        content_failures = 0
        compared = 0
        with pack.open("rb") as f:
            for ref, p_off, p_bytes, s_off, s_bytes in entries:
                pack_digest = hashlib.sha256()

                def stream_region(offset: int, length: int) -> bool:
                    f.seek(offset)
                    remaining = length
                    while remaining > 0:
                        step = min(remaining, HASH_CHUNK)
                        chunk = f.read(step)
                        if len(chunk) != step:
                            return False
                        pack_digest.update(chunk)
                        remaining -= step
                    return True

                if not stream_region(p_off, p_bytes):
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)}: pack payload short read")
                    continue
                if s_bytes and not stream_region(s_off, s_bytes):
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)}: pack scale short read")
                    continue
                try:
                    source_digest = hash_source_entry(source, ref)
                except (RuntimeError, _tables.PackFailure) as error:
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                         f"source read failed: {error}")
                    continue
                compared += 1
                if pack_digest.hexdigest() != source_digest:
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                         f"pack bytes != source bytes")
                if compared % 20 == 0:
                    print(f"  content {compared}/{len(entries)} tensors", flush=True)
        verdict["tensors_compared"] = compared
        verdict["content_errors"] = content_failures
        if content_failures:
            fail(f"{content_failures} content mismatches")

    if recompute_file_hash:
        file_sha = sha256_file(pack)
        verdict["file_sha256_recomputed"] = file_sha

    # -- receipt cross-check ---------------------------------------------------
    if receipt_path is not None and Path(receipt_path).is_file():
        receipt = json.loads(Path(receipt_path).read_text())
        checks = {
            "tensor_count": (receipt.get("tensor_count"), tensor_count),
            "bytes": (receipt.get("bytes"), file_bytes_actual),
            "first_layer_index": (receipt.get("first_layer_index"), first_layer),
            "layer_count": (receipt.get("layer_count"), layer_count),
        }
        if file_sha is not None:
            checks["output_sha256"] = (receipt.get("output_sha256"), file_sha)
        for name, (recorded, recomputed) in checks.items():
            if recorded != recomputed:
                fail(f"receipt {name}={recorded!r}, verifier recomputed {recomputed!r}")
        if source is not None and receipt.get("source_index_sha256") != source.index_sha256:
            fail("receipt source_index_sha256 does not match the live checkpoint index")

    verdict["errors"] = findings
    verdict["verdict"] = "PASS" if not findings else "FAIL"
    return not findings, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tp-degree", type=int, default=1,
        help="the pack's TP degree; entry shapes are compared per-rank")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="checkpoint dir; omit for structure-only pass")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--recompute-file-hash", action="store_true",
                        help="re-read the pack to recompute the whole-file sha256")
    parser.add_argument("--json-out", type=Path, help="verdict JSON path")
    args = parser.parse_args()

    ok, verdict = verify(args.pack, args.checkpoint, args.receipt,
                         args.recompute_file_hash, args.tp_degree)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(f"qwen38_pack_verify pack={args.pack} verdict={verdict['verdict']} "
          f"file_gib={verdict['file_bytes'] / 2**30:.2f} errors={len(verdict['errors'])}")
    for finding in verdict["errors"][:40]:
        print(f"  {finding}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
