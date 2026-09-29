#!/usr/bin/env python3
"""Read SPTD speculation tap dumps written by spark_speculation_tap.c.

Layout (little-endian): a 128-byte header, then records of a 40-byte record
header followed by tap_count rows of row_bytes each, in the order the taps
were configured. docs/SPECULATION_TAPS.md is the contract.
"""
import argparse
import json
import struct
import sys

MAGIC = 0x44545053
VERSION = 1
HEADER_BYTES = 128
RECORD_HEADER_BYTES = 40
FLAG_TRUNCATED = 0x1
FLAG_CLOSED = 0x2
REDUCTIONS = {1: "mean", 2: "all"}
RECORD_FLAGS = {0x1: "prefill", 0x2: "decode", 0x4: "verify"}


class TapDumpError(ValueError):
    pass


def parse_header(raw):
    if len(raw) != HEADER_BYTES:
        raise TapDumpError("short header")
    magic, version, header_bytes, record_header_bytes = struct.unpack_from("<4I", raw, 0)
    if magic != MAGIC:
        raise TapDumpError("not an SPTD tap dump")
    if version != VERSION or header_bytes != HEADER_BYTES or record_header_bytes != RECORD_HEADER_BYTES:
        raise TapDumpError(f"unsupported SPTD version {version}")
    reduction, tap_count, stream_count, hidden, layer_count, row_elements, row_bytes, dtype = struct.unpack_from("<8I", raw, 16)
    if reduction not in REDUCTIONS or not 1 <= tap_count <= 8 or dtype != 1 or row_bytes != row_elements * 2:
        raise TapDumpError("inconsistent tap set in header")
    layers = list(struct.unpack_from("<8I", raw, 48))[:tap_count]
    fingerprint, generation = struct.unpack_from("<2Q", raw, 80)
    tp_rank, flags = struct.unpack_from("<2I", raw, 96)
    (records,) = struct.unpack_from("<Q", raw, 104)
    tag = raw[112:128].split(b"\0", 1)[0].decode("ascii", "replace")
    expected = hidden * (stream_count if reduction == 2 else 1)
    if row_elements != expected:
        raise TapDumpError("row width does not match the reduction")
    return {
        "reduction": REDUCTIONS[reduction], "tap_count": tap_count, "stream_count": stream_count,
        "hidden_dimension": hidden, "layer_count": layer_count, "row_elements": row_elements,
        "row_bytes": row_bytes, "record_bytes": row_bytes * tap_count, "layers": layers,
        "fingerprint": f"{fingerprint:016x}", "engine_generation": generation, "tp_rank": tp_rank,
        "closed": bool(flags & FLAG_CLOSED), "truncated": bool(flags & FLAG_TRUNCATED),
        "records": records, "model_tag": tag,
    }


def iter_records(path):
    """Yield (header, record) where record holds sequence_id, position, token_id, flags, serial, rows."""
    with open(path, "rb") as handle:
        header = parse_header(handle.read(HEADER_BYTES))
        size = RECORD_HEADER_BYTES + header["record_bytes"]
        while True:
            raw = handle.read(size)
            if not raw:
                return
            if len(raw) != size:
                raise TapDumpError("trailing partial record")
            sequence_id, position, token_id, next_token_id, flags, reserved, serial = struct.unpack_from("<QQIIIIQ", raw, 0)
            if flags & ~0x7 or reserved:
                raise TapDumpError(f"unknown record flags {flags:#x} or reserved word {reserved:#x}")
            yield header, {"sequence_id": sequence_id, "position": position, "token_id": token_id,
                           "next_token_id": next_token_id, "flags": flags, "serial": serial,
                           "rows": raw[RECORD_HEADER_BYTES:]}


def tap_rows(header, rows):
    """Split one record's payload into per-tap rows of raw bf16 bytes."""
    width = header["row_bytes"]
    return [rows[index * width:(index + 1) * width] for index in range(header["tap_count"])]


def summarize(path):
    with open(path, "rb") as handle:
        header = parse_header(handle.read(HEADER_BYTES))
    count, flags, sequences, problems, last_serial = 0, {}, {}, [], 0
    for _, record in iter_records(path):
        count += 1
        if record["serial"] <= last_serial:
            problems.append(f"serial {record['serial']} after {last_serial}")
        last_serial = record["serial"]
        for bit, name in RECORD_FLAGS.items():
            if record["flags"] & bit:
                flags[name] = flags.get(name, 0) + 1
        entry = sequences.setdefault(record["sequence_id"], {"records": 0, "first": record["position"], "next": record["position"], "gaps": 0, "repeats": 0})
        if record["position"] < entry["next"] and entry["records"]:
            entry["repeats"] += 1
        elif record["position"] > entry["next"]:
            entry["gaps"] += 1
        entry["next"] = max(entry["next"], record["position"] + 1)
        entry["records"] += 1
    if header["closed"] and count != header["records"]:
        problems.append(f"header says {header['records']} records, file holds {count}")
    if not header["closed"]:
        problems.append("dump was not closed (engine did not shut down cleanly)")
    for sequence, entry in sequences.items():
        if entry["repeats"]:
            problems.append(f"sequence {sequence}: {entry['repeats']} repeated positions")
    return {"path": path, "header": header, "records": count, "flags": flags,
            "sequences": {str(key): value for key, value in sequences.items()}, "problems": problems}


def export_offline(path, out, layer, classes, model, firmware):
    """Write one tap of an SPTD dump as a spark-tapdump-1 directory (tools/spec_offline/tapdump.py)."""
    import hashlib
    import os
    with open(path, "rb") as handle:
        header = parse_header(handle.read(HEADER_BYTES))
    if layer not in header["layers"]:
        raise TapDumpError(f"layer {layer} is not tapped; the dump has {header['layers']}")
    index = header["layers"].index(layer)
    tap_name = f"L{layer}.{header['reduction']}"
    sequences = {}
    for _, record in iter_records(path):
        sequences.setdefault(record["sequence_id"], []).append(record)
    os.makedirs(out, exist_ok=False)
    streams, skipped = [], {}
    for sequence, records in sequences.items():
        records.sort(key=lambda record: record["position"])
        positions = [record["position"] for record in records]
        prompt = sum(1 for record in records if record["flags"] & 0x1)
        if positions != list(range(len(records))):
            skipped[str(sequence)] = "positions are not contiguous from 0 (prefix-cache hit or lost records)"
            continue
        if prompt == 0 or any(record["flags"] & 0x1 for record in records[prompt:]):
            skipped[str(sequence)] = "prefill records are not a prefix of the stream"
            continue
        tokens = [record["token_id"] for record in records] + [records[-1]["next_token_id"]]
        name = f"seq{sequence}"
        with open(os.path.join(out, f"{name}.u32"), "wb") as handle:
            handle.write(struct.pack(f"<II{len(tokens)}I", prompt, len(tokens), *tokens))
        with open(os.path.join(out, f"{name}.{tap_name}.bf16"), "wb") as handle:
            for record in records:
                handle.write(tap_rows(header, record["rows"])[index])
            handle.write(bytes(header["row_bytes"]))
        streams.append({"name": name, "class": classes.get(str(sequence), "unknown"), "tokens": f"{name}.u32",
                        "prompt_tokens": prompt, "total_tokens": len(tokens), "taps": f"{name}.{tap_name}.bf16"})
    manifest = {"format": "spark-tapdump-1", "model": model, "firmware": firmware,
                "hidden_dimension": header["row_elements"],
                "taps": [{"name": tap_name, "layer": layer, "dtype": "bf16", "reduction": header["reduction"],
                          "row_semantics": "row p is the tap after token p was committed"}],
                "streams": streams,
                "notes": {"source": os.path.basename(path), "sptd_fingerprint": header["fingerprint"],
                          "engine_generation": header["engine_generation"], "tp_rank": header["tp_rank"],
                          "final_row": "zero: the engine never computes the row of a stream's final token",
                          "skipped": skipped}}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=1))
    lines = []
    for name in sorted(os.listdir(out)):
        with open(os.path.join(out, name), "rb") as handle:
            lines.append(f"{hashlib.sha256(handle.read()).hexdigest()}  {name}")
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary", help="print header, per-sequence coverage and problems as JSON")
    summary.add_argument("path")
    verify = sub.add_parser("verify", help="exit 1 unless the dump is closed, complete and has no repeated positions")
    verify.add_argument("path")
    export = sub.add_parser("export-offline", help="write one tapped layer as a spark-tapdump-1 directory for tools/spec_offline")
    export.add_argument("path")
    export.add_argument("out")
    export.add_argument("--layer", type=int, required=True)
    export.add_argument("--classes", help="JSON object: sequence id -> content class")
    export.add_argument("--model", required=True)
    export.add_argument("--firmware", required=True)
    args = parser.parse_args(argv)
    if args.command == "export-offline":
        classes = json.loads(open(args.classes, encoding="utf-8").read()) if args.classes else {}
        try:
            manifest = export_offline(args.path, args.out, args.layer, classes, args.model, args.firmware)
        except (OSError, TapDumpError) as error:
            print(json.dumps({"path": args.path, "error": str(error)}))
            return 1
        print(json.dumps({"out": args.out, "streams": len(manifest["streams"]), "skipped": manifest["notes"]["skipped"]}))
        return 0 if manifest["streams"] else 1
    try:
        report = summarize(args.path)
    except (OSError, TapDumpError) as error:
        print(json.dumps({"path": args.path, "error": str(error)}))
        return 1
    print(json.dumps(report, indent=1 if args.command == "summary" else None, sort_keys=True))
    return 1 if args.command == "verify" and report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
