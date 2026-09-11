#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

_spec = importlib.util.spec_from_file_location(
    "qwen38_max_stagepack_tables", str(Path(_TOOLS_DIR) / "qwen38_stagepack.py"))
_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tables)
_tables.EXPERT_CODEC = "nvfp4"
_tables.STRIP_MTP = True

HASH_CHUNK = 16 * 1024 * 1024
MAGIC = _tables.MAGIC
FORMAT_VERSION = _tables.FORMAT2_VERSION
HEADER_BYTES = _tables.HEADER2_BYTES
ENTRY_BYTES = _tables.ENTRY_BYTES
PAYLOAD_ALIGNMENT = _tables.PAYLOAD_ALIGNMENT
HEADER_STRUCT = struct.Struct("<28I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
BF16_BYTES = 2
F32_BYTES = 4


class _HashingSink:
    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        self.digest.update(data)
        return len(data)


def entry_bytes_for(weight_format: int, rows: int, cols: int) -> tuple[int, int]:
    if weight_format == _tables.WEIGHT_FP8_F32B128:
        return rows * (cols // 2), rows * (cols // 16)
    element = BF16_BYTES if weight_format == _tables.WEIGHT_BF16 else F32_BYTES
    return rows * cols * element, 0


def verify(pack: Path, tp_degree: int, tp_rank: int, checkpoint: Path | None,
           receipt_path: Path | None, recompute_file_hash: bool) -> tuple[bool, dict]:
    findings: list[str] = []

    def fail(message: str) -> None:
        findings.append(message)

    file_bytes_actual = pack.stat().st_size
    first_layer = layer_count = -1
    tensor_count = 0
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
         expert_count, experts_per_token, moe_int, vocab, mxfp4_group,
         mtp_count, header_tp_degree, header_tp_rank,
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
        want("routed_expert_count", expert_count, _tables.EXPERT_COUNT)
        want("experts_per_token", experts_per_token, _tables.EXPERTS_PER_TOKEN)
        want("expert_intermediate_dimension", moe_int, _tables.EXPERT_INTERMEDIATE)
        want("output_vocab_count", vocab, _tables.VOCAB)
        want("mxfp4_group_size", mxfp4_group, _tables.MXFP4_GROUP)
        want("mtp_layer_count", mtp_count, 0)
        want("header tp_degree", header_tp_degree, tp_degree)
        want("header tp_rank", header_tp_rank, tp_rank)
        want("directory_offset", directory_offset, HEADER_BYTES)
        want("file_bytes", file_bytes, file_bytes_actual)
        if layer_count <= 0 or first_layer < 0 or first_layer + layer_count > _tables.LAYER_COUNT:
            fail(f"invalid slice {first_layer}+{layer_count} of {_tables.LAYER_COUNT}")
        if tp_degree <= 1 or not 0 <= tp_rank < tp_degree:
            fail(f"invalid tp placement degree={tp_degree} rank={tp_rank}")

        expected_refs: dict[tuple[int, int], object] = {}
        try:
            for ref in _tables.build_inventory(first_layer, layer_count):
                expected_refs[(ref.kind, ref.layer)] = ref
        except _tables.PackFailure as error:
            fail(f"inventory build failed: {error}")
        want("tensor_count", tensor_count, len(expected_refs))

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
                continue
            seen.add(key)
            ref = expected_refs.get(key)
            if ref is None:
                fail(f"{tag}: not in the MTP-stripped inventory of slice "
                     f"{first_layer}+{layer_count}")
                continue
            plan = _tables.build_tp_plan(ref, tp_degree, tp_rank)
            shard_rows, shard_cols = _tables.packed_tp_shape(ref, plan)
            if (rows, cols) != (shard_rows, shard_cols):
                fail(f"{tag}: shape {rows}x{cols}, expected rank shard "
                     f"{shard_rows}x{shard_cols}")
            if fmt != ref.weight_format:
                fail(f"{tag}: weight_format={fmt}, packer format ladder says "
                     f"{ref.weight_format}")
            want_group = 128 if fmt == _tables.WEIGHT_FP8_F32B128 else 0
            if scale_group != want_group:
                fail(f"{tag}: scale_group_size={scale_group}, expected {want_group}")
            want_payload, want_scale = entry_bytes_for(fmt, rows, cols)
            if p_bytes != want_payload:
                fail(f"{tag}: payload_bytes={p_bytes}, format math says {want_payload}")
            if s_bytes != want_scale:
                fail(f"{tag}: scale_bytes={s_bytes}, format math says {want_scale}")
            if p_off % PAYLOAD_ALIGNMENT != 0:
                fail(f"{tag}: payload_offset {p_off} not {PAYLOAD_ALIGNMENT}-aligned")
            if s_bytes and s_off % PAYLOAD_ALIGNMENT != 0:
                fail(f"{tag}: scale_offset {s_off} not {PAYLOAD_ALIGNMENT}-aligned")
            if p_off + p_bytes > file_bytes_actual:
                fail(f"{tag}: payload region overruns file")
            if s_bytes and s_off + s_bytes > file_bytes_actual:
                fail(f"{tag}: scale region overruns file")
            entries.append((ref, plan, p_off, p_bytes, s_off, s_bytes))

        for key in sorted(set(expected_refs) - seen):
            fail(f"missing tensor kind={key[0]} layer={hex(key[1])}")

    verdict = {
        "verdict": "FAIL",
        "pack": str(pack),
        "file_bytes": file_bytes_actual,
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": tensor_count,
        "errors": findings,
    }
    if findings:
        return False, verdict

    if checkpoint is not None:
        source = _tables.SafetensorsSource(checkpoint)
        source.check_config()
        content_failures = 0
        compared = 0
        with pack.open("rb") as f:
            for ref, plan, p_off, p_bytes, s_off, s_bytes in entries:
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

                ok = stream_region(p_off, p_bytes)
                if s_bytes:
                    ok = stream_region(s_off, s_bytes) and ok
                if not ok:
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)}: pack region short read")
                    continue
                try:
                    _, _, source_offset = source.check_shape(ref)
                    sink = _HashingSink()
                    if plan is not None:
                        _tables.copy_tp_plan(source, ref, plan, sink)
                    elif ref.weight_format == _tables.WEIGHT_FP8_F32B128:
                        _tables.copy_nvfp4_experts(source, ref, sink)
                    else:
                        _tables.copy_bf16_tensor(source, ref, source_offset, sink)
                except (RuntimeError, _tables.PackFailure) as error:
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                         f"source re-emit failed: {error}")
                    continue
                compared += 1
                if pack_digest.hexdigest() != sink.digest.hexdigest():
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                         f"pack bytes != source bytes for this rank")
                if compared % 50 == 0:
                    print(f"  content {compared}/{len(entries)} tensors", flush=True)
        verdict["tensors_compared"] = compared
        verdict["content_errors"] = content_failures
        if content_failures:
            fail(f"{content_failures} content mismatches")

    if recompute_file_hash:
        file_sha = sha256_file(pack)
        verdict["file_sha256_recomputed"] = file_sha

    if receipt_path is not None and Path(receipt_path).is_file():
        receipt = json.loads(Path(receipt_path).read_text())
        checks = {
            "tensor_count": (receipt.get("tensor_count"), tensor_count),
            "bytes": (receipt.get("bytes"), file_bytes_actual),
            "first_layer_index": (receipt.get("first_layer_index"), first_layer),
            "layer_count": (receipt.get("layer_count"), layer_count),
            "tp_degree": (receipt.get("tp_degree"), tp_degree),
            "tp_rank": (receipt.get("tp_rank"), tp_rank),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="verify one qwen38_max TP16 "
                                     "MTP-stripped rank pack against the v2 wire "
                                     "format and the live checkpoint")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--recompute-file-hash", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    ok, verdict = verify(args.pack, args.tp_degree, args.tp_rank, args.checkpoint,
                         args.receipt, args.recompute_file_hash)
    if args.json_out:
        args.json_out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "errors"},
                     indent=2))
    if verdict["errors"]:
        for message in verdict["errors"][:20]:
            print(f"  {message}", file=sys.stderr)
    print(f"verdict={verdict['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
