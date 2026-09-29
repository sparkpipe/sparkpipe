#!/usr/bin/env python3
"""Read SPTD speculation tap dumps written by spark_speculation_tap.c.

Layout (little-endian): a 128-byte header, then records of a 32-byte record
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
RECORD_HEADER_BYTES = 32
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
            sequence_id, position, token_id, flags, serial = struct.unpack_from("<QQIIQ", raw, 0)
            if flags & ~0x7:
                raise TapDumpError(f"unknown record flags {flags:#x}")
            yield header, {"sequence_id": sequence_id, "position": position, "token_id": token_id,
                           "flags": flags, "serial": serial, "rows": raw[RECORD_HEADER_BYTES:]}


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary", help="print header, per-sequence coverage and problems as JSON")
    summary.add_argument("path")
    verify = sub.add_parser("verify", help="exit 1 unless the dump is closed, complete and has no repeated positions")
    verify.add_argument("path")
    args = parser.parse_args(argv)
    try:
        report = summarize(args.path)
    except (OSError, TapDumpError) as error:
        print(json.dumps({"path": args.path, "error": str(error)}))
        return 1
    print(json.dumps(report, indent=1 if args.command == "summary" else None, sort_keys=True))
    return 1 if args.command == "verify" and report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
