#!/usr/bin/env python3
"""Reader for tap streams written by a stage adapter with "taps" configured.

Layout (little-endian), from include/sparkpipe/spark_tap_stream.h:
  HELLO, 624 bytes: magic "SPT2", version, kind 1, header_bytes, model tag
  (32), pack identity (32), hidden, total_layers, dtype (1 = bf16),
  entry_count, tap_rank, tp_degree, fingerprint, entries[128] {u16 form,
  u16 layer} (form 1 = layer output, 3 = final hidden before the norm).
  FRAME, 96 bytes per lane and submission: magic, version, kind 2,
  header_bytes, fingerprint, control_generation, request_id,
  request_generation, sequence_id, first_position, serial, row_count, flags
  (1 prefill, 2 decode, 4 verify, 8 chain), tail_token, entry_count,
  payload_bytes; then row_count input tokens (u32) and row_count rows of
  entry_count x hidden bf16.
Row p is the target state computed for input token t_p at position p.
"""
import struct
import sys
from collections import defaultdict
from pathlib import Path

MAGIC = 0x32545053
HELLO = struct.Struct("<4I32s32s6IQ512s")
FRAME = struct.Struct("<4I7Q4IQ")
FORMS = {1: "o", 3: "final"}
FLAGS = {1: "prefill", 2: "decode", 4: "verify", 8: "chain"}


def read_hello(handle):
    raw = handle.read(HELLO.size)
    if len(raw) != HELLO.size:
        raise ValueError("short hello")
    fields = HELLO.unpack(raw)
    magic, version, kind, header_bytes = fields[:4]
    if magic != MAGIC or kind != 1 or header_bytes != HELLO.size:
        raise ValueError(f"not a tap stream hello: magic {magic:#x} kind {kind} header {header_bytes}")
    tag = fields[4].split(b"\0", 1)[0].decode()
    hidden, layers, dtype, count, rank, degree, fingerprint = fields[6:13]
    entries = [struct.unpack_from("<HH", fields[13], 4 * i) for i in range(count)]
    names = [FORMS[form] + (str(layer) if form == 1 else "") for form, layer in entries]
    return {"version": version, "model": tag, "pack_identity": fields[5].hex(), "hidden": hidden, "layers": layers, "dtype": dtype,
            "entries": names, "tap_rank": rank, "tp_degree": degree, "fingerprint": fingerprint}


def frames(path):
    size = Path(path).stat().st_size
    with open(path, "rb") as handle:
        hello = read_hello(handle)
        row_bytes = len(hello["entries"]) * hello["hidden"] * 2
        while True:
            start = handle.tell()
            raw = handle.read(FRAME.size)
            if not raw:
                return
            if len(raw) != FRAME.size:
                print(f"{path}: ends in a truncated frame header at byte {start}; the rest is ignored", file=sys.stderr)
                return
            (magic, version, kind, header_bytes, fingerprint, control, request, request_generation, sequence, first, serial,
             rows, flags, tail, entries, payload) = FRAME.unpack(raw)
            if magic != MAGIC or kind != 2 or fingerprint != hello["fingerprint"] or entries != len(hello["entries"]):
                raise ValueError(f"bad frame at serial {serial}")
            if payload != rows * (4 + row_bytes):
                raise ValueError(f"frame {serial}: payload {payload} for {rows} rows")
            if handle.tell() + payload > size:
                print(f"{path}: frame {serial} at byte {start} is cut short ({size - handle.tell()} of {payload} payload bytes); the rest is ignored", file=sys.stderr)
                return
            tokens = struct.unpack(f"<{rows}I", handle.read(4 * rows))
            offset = handle.tell()
            handle.seek(rows * row_bytes, 1)
            yield hello, {"serial": serial, "request_id": request, "request_generation": request_generation, "sequence_id": sequence,
                          "control_generation": control, "first_position": first, "rows": rows, "flags": flags, "tail_token": tail,
                          "tokens": tokens, "offset": offset, "row_bytes": row_bytes}


def summary(path):
    hello = None
    per = defaultdict(lambda: {"frames": 0, "rows": 0, "kinds": defaultdict(int), "end": None, "gaps": 0, "overlaps": 0})
    for hello, frame in frames(path):
        key = (frame["control_generation"], frame["request_id"], frame["request_generation"], frame["sequence_id"])
        item = per[key]
        item["frames"] += 1
        item["rows"] += frame["rows"]
        for bit, name in FLAGS.items():
            if frame["flags"] & bit:
                item["kinds"][name] += 1
        if item["end"] is not None and frame["first_position"] != item["end"]:
            if frame["first_position"] > item["end"]:
                item["gaps"] += 1
            else:
                item["overlaps"] += 1
        item["end"] = frame["first_position"] + frame["rows"]
    return hello, per


def main():
    for path in sys.argv[1:]:
        hello, per = summary(Path(path))
        print(path, hello)
        for key, item in sorted(per.items()):
            print(" request", key[1], "sequence", key[3], "frames", item["frames"], "rows", item["rows"], "end", item["end"],
                  "gaps", item["gaps"], "overlaps", item["overlaps"], dict(item["kinds"]))


if __name__ == "__main__":
    main()
