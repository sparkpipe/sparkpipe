#!/usr/bin/env python3
"""Standalone CPU verifier for a deployed DSV4 Pro rank pack (TP16 or TP4xPP4).

Re-derives the expected rank directory from the model contract (no full
pack needed on disk), validates the deployed pack's header, geometry,
directory and payload bounds, optionally re-checks the build-receipt
sha256, and byte-compares sampled tensor payloads and scale blocks
against the GA source checkpoint. CPU/disk only - never touches a GPU.

Exit 0 = every check passed; non-zero = first failure, named loudly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

pro = _load("dsv4_pro_stagepack_verify", TOOLS / "dsv4_pro_stagepack.py")
flash = pro.flash
tp16 = _load("dsv4_tp16_stagepack_verify", TOOLS / "dsv4_tp16_stagepack.py")

GLOBAL_LAYER = tp16.GLOBAL_LAYER

PRO_LAYERS = pro.PRO_LAYERS
PRO_HIDDEN = pro.PRO_HIDDEN
PRO_VOCAB = pro.PRO_VOCAB
PRO_EXPERTS = pro.PRO_EXPERTS
PRO_MTP_PACKED = pro.PRO_MTP_PACKED

HEADER = struct.Struct("<16I2Q")
ENTRY = struct.Struct("<6I2Q")
MAGIC = 0x34565344
FP8_BLOCK = 128
FP4_BLOCK = 32


class VerifyFailure(RuntimeError):
    pass


def layout_of(rank: int, tp_degree: int, pp_stages: int) -> Tuple[int, int]:
    """World rank -> (pp_stage, tp_rank) for a tp_degree x pp_stages grid."""
    if not 0 <= rank < tp_degree * pp_stages:
        raise VerifyFailure(f"rank {rank} outside the {tp_stages_desc(tp_degree, pp_stages)} grid")
    return divmod(rank, tp_degree)


def tp_stages_desc(tp_degree: int, pp_stages: int) -> str:
    return f"TP{tp_degree}xPP{pp_stages}"


def build_expected(tp_rank: int, pp_stage: int, tp_degree: int, pp_stages: int,
                   contract: Mapping[str, object]):
    """Plan the expected rank directory from contract records alone."""
    tp16.TP_DEGREE = tp_degree
    tp16.apply_model_geometry("pro")
    records = pro.pro_build_records(contract, 0, PRO_LAYERS)
    expected: Dict[Tuple[int, int], dict] = {}
    for record in records:
        entry = (record.kind, record.layer, record.weight_format,
                 record.rows, record.columns, 0, 0, 0)
        try:
            planned, indices, col_start, scale_total = tp16.plan_entry(
                entry, tp_rank, pp_stages, pp_stage)
        except tp16.PackFailure as error:
            if str(error) == "filtered":
                continue
            raise
        key = (record.kind, record.layer)
        if key in expected:
            raise VerifyFailure(f"duplicate record key {key}")
        expected[key] = {
            "kind": record.kind, "layer": record.layer,
            "weight": record.weight_format,
            "rows": planned[3], "columns": planned[4],
            "indices": indices, "col_start": col_start,
            "scale_bytes": scale_total, "record": record,
        }
    return expected


def verify_directory(header_fields, entries: Sequence[Tuple], expected: Mapping,
                     file_bytes: int, rank: int, tp_degree: int,
                     pp_stages: int) -> None:
    """Header + directory conformance against the planned expectation."""
    pp_stage = rank // tp_degree
    if header_fields[0] != MAGIC:
        raise VerifyFailure("not a DSV4 stage pack (magic mismatch)")
    slice_want = tp16.layer_slice(pp_stages, pp_stage)
    if (header_fields[9], header_fields[10]) != slice_want:
        raise VerifyFailure(
            f"layer slice {header_fields[9]}+{header_fields[10]} does not "
            f"match rank {rank} ({tp_stages_desc(tp_degree, pp_stages)}, "
            f"stage {pp_stage} wants {slice_want[0]}+{slice_want[1]})")
    if header_fields[16] != HEADER.size or header_fields[17] != file_bytes:
        raise VerifyFailure("header size fields inconsistent with the file")
    geometry = (header_fields[11], header_fields[12], header_fields[13],
                header_fields[14], header_fields[15])
    if geometry != (PRO_LAYERS, PRO_HIDDEN, PRO_VOCAB, PRO_EXPERTS,
                    PRO_MTP_PACKED):
        raise VerifyFailure(f"header geometry fields disagree: {geometry}")
    if header_fields[8] != len(expected) or len(entries) != len(expected):
        raise VerifyFailure(
            f"tensor count {header_fields[8]} != expected {len(expected)}")
    previous_end = None
    for index, actual in enumerate(entries):
        key = (actual[0], actual[1])
        plan = expected.get(key)
        if plan is None:
            raise VerifyFailure(f"unexpected tensor kind={key[0]} layer={key[1]}")
        if actual[:5] != (plan["kind"], plan["layer"], plan["weight"],
                          plan["rows"], plan["columns"]):
            raise VerifyFailure(
                f"dims mismatch kind={key[0]} layer={key[1]}: pack "
                f"{actual[3]}x{actual[4]} w{actual[2]}, planned "
                f"{plan['rows']}x{plan['columns']} w{plan['weight']}")
        payload = tp16.payload_bytes(plan["weight"], plan["rows"], plan["columns"])
        scales = plan["scale_bytes"]
        if actual[6] < HEADER.size + ENTRY.size * len(entries):
            raise VerifyFailure(f"payload offset inside directory: {key}")
        if actual[6] + payload > file_bytes:
            raise VerifyFailure(f"payload bounds exceed file: {key}")
        if scales:
            if actual[7] != actual[6] + payload or actual[7] + scales > file_bytes:
                raise VerifyFailure(f"scale bounds invalid: {key}")
        elif actual[7] != 0:
            raise VerifyFailure(f"unexpected scale offset: {key}")
        if previous_end is not None and actual[6] < previous_end:
            raise VerifyFailure("directory payload offsets are not ordered")
        previous_end = actual[6] + payload + scales


def _checkpoint_slice(source, record, indices: Sequence[int], col_start: int,
                      width: int) -> bytes:
    """Expected payload+scale bytes for one entry, read from the checkpoint."""
    weight = record.weight_format
    out = bytearray()
    if record.stacked_fp4:
        rows_per = record.source_rows
        block_width = (record.source_columns + FP4_BLOCK - 1) // FP4_BLOCK
        start_block = col_start // FP4_BLOCK
        width_blocks = (width + FP4_BLOCK - 1) // FP4_BLOCK
        stride = tp16.payload_bytes(weight, 1, record.source_columns)
        # the pack sections are all-payload-then-all-scales (copy_payload
        # runs over every row before copy_scales starts) - never interleave
        for stacked_row in indices:
            expert, row = divmod(stacked_row, rows_per)
            raw = source.read(record.source_names[expert])
            base = row * stride + col_start // 2
            out += raw[base:base + width // 2]
        for stacked_row in indices:
            expert, row = divmod(stacked_row, rows_per)
            scales = source.read(record.scale_names[expert])
            base = row * block_width + start_block
            out += scales[base:base + width_blocks]
        return bytes(out)
    raw = source.read(record.source_names[0])
    stride = tp16.payload_bytes(weight, 1, record.source_columns)
    element = tp16.element_bytes(weight)
    for row in indices:
        if weight == tp16.WEIGHT_FP4:
            base = row * stride + col_start // 2
            count = width // 2
        else:
            base = row * stride + col_start * element
            count = width * element
        out += raw[base:base + count]
    if weight == tp16.WEIGHT_FP8:
        scales = source.read(record.scale_names[0])
        blocks = (record.source_columns + FP8_BLOCK - 1) // FP8_BLOCK
        start_block = col_start // FP8_BLOCK
        width_blocks = (width + FP8_BLOCK - 1) // FP8_BLOCK
        for row in indices:
            base = (row // FP8_BLOCK) * blocks + start_block
            out += scales[base:base + width_blocks]
    return bytes(out)


SAMPLE_PICKS = (
    (tp16.KIND_EMBEDDING, "first"),
    (tp16.KIND_WQ_A, "first"),
    (tp16.KIND_WKV, "first+1"),
    (tp16.KIND_WO_B, "first+2"),
    (tp16.KIND_EXPERTS_W1, "first+3"),
    (tp16.KIND_EXPERTS_W2, "first+4"),
    (tp16.KIND_MTP_MARKOV_W1, "global"),
    (tp16.KIND_HC_HEAD_FN, "global"),
    (tp16.KIND_FINAL_NORM, "global"),
)


def sample_checkpoint(pack_path, expected: Mapping, source, rank: int,
                      tp_degree: int, pp_stages: int,
                      limit_experts: int = 8) -> List[dict]:
    """Byte-compare sampled entries' payload+scale bytes vs the checkpoint."""
    stage_first = tp16.layer_slice(pp_stages, rank // tp_degree)[0]
    picks = []
    for kind, where in SAMPLE_PICKS:
        layer = {"first": stage_first, "first+1": stage_first + 1,
                 "first+2": stage_first + 2, "first+3": stage_first + 3,
                 "first+4": stage_first + 4,
                 "global": GLOBAL_LAYER}[where]
        picks.append((kind, layer))
    results = []
    with open(pack_path, "rb") as pack:
        for kind, layer in picks:
            plan = expected.get((kind, layer))
            if plan is None:
                continue
            record = plan["record"]
            if record.i64_to_u32:
                results.append({"kind": kind, "layer": layer,
                                "status": "skipped",
                                "note": "i64->u32 transform not byte-compared"})
                continue
            payload_size = tp16.payload_bytes(
                record.weight_format, plan["rows"], plan["columns"])
            note = None
            indices = plan["indices"]
            if record.stacked_fp4 and limit_experts:
                rows_per = record.source_rows
                indices = [idx for idx in indices
                           if idx // rows_per < limit_experts]
                note = f"first {limit_experts} experts of {record.source_rows and len(record.source_names)}"
            expected_bytes = _checkpoint_slice(
                source, record, indices, plan["col_start"], plan["columns"])
            pack.seek(plan["entry_offset"])
            actual = pack.read(len(expected_bytes))
            verdict = "OK" if actual == expected_bytes else "MISMATCH"
            entry = {"kind": kind, "layer": layer, "status": verdict,
                     "bytes_compared": len(expected_bytes),
                     "of_payload": payload_size}
            if note:
                entry["note"] = note
            results.append(entry)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--tp-degree", type=int, default=16,
                        choices=(4, 16))
    parser.add_argument("--pp-stages", type=int, default=1,
                        choices=(1, 4))
    parser.add_argument("--expect-sha256")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--contract", type=Path,
                        default=ROOT / "model_contracts" / "dsv4_pro.json")
    parser.add_argument("--limit-experts", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        pp_stage, tp_rank = layout_of(args.rank, args.tp_degree, args.pp_stages)
    except VerifyFailure as error:
        print(error, file=sys.stderr)
        return 2
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = build_expected(tp_rank, pp_stage, args.tp_degree,
                              args.pp_stages, contract)

    file_bytes = args.pack.stat().st_size
    with open(args.pack, "rb") as handle:
        header_fields = HEADER.unpack(handle.read(HEADER.size))
        entries = [ENTRY.unpack(handle.read(ENTRY.size))
                   for _ in range(header_fields[8])]
    by_key = {(entry[0], entry[1]): entry[6] for entry in entries}
    for key, plan in expected.items():
        if key in by_key:
            plan["entry_offset"] = by_key[key]

    verdict = {"pack": str(args.pack), "rank": args.rank,
               "layout": tp_stages_desc(args.tp_degree, args.pp_stages),
               "bytes": file_bytes, "tensor_count": len(entries)}
    try:
        verify_directory(header_fields, entries, expected, file_bytes,
                         args.rank, args.tp_degree, args.pp_stages)
        verdict["directory"] = "OK"
    except VerifyFailure as error:
        verdict["directory"] = f"FAIL: {error}"
        print(json.dumps(verdict) if args.json else verdict["directory"],
              file=sys.stderr)
        return 2
    verdict["tensors_expected"] = len(expected)

    if args.expect_sha256:
        digest = flash.sha256_file(args.pack)
        verdict["sha256"] = digest
        verdict["sha256_match"] = digest == args.expect_sha256.lower()
        if not verdict["sha256_match"]:
            print(f"sha256 mismatch: expected {args.expect_sha256}",
                  file=sys.stderr)
            return 3

    if args.model_dir:
        source = flash.SafetensorSource(args.model_dir)
        try:
            verdict["samples"] = sample_checkpoint(
                args.pack, expected, source, args.rank, args.tp_degree,
                args.pp_stages, args.limit_experts)
        finally:
            source.close()
        bad = [s for s in verdict["samples"] if s["status"] == "MISMATCH"]
        if bad:
            verdict["content"] = f"FAIL: {len(bad)} sampled tensors mismatch"
            print(json.dumps(verdict, indent=2) if args.json
                  else verdict["content"], file=sys.stderr)
            return 4
        verdict["content"] = "OK"

    print(json.dumps(verdict, indent=2) if args.json
          else "PASS " + json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
