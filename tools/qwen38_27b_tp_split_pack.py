#!/usr/bin/env python3
"""Split a whole-stack Qwen3.8-27B stage pack into tensor-parallel rank packs.

Each rank pack keeps the source header, directory order and weight formats;
only tp_degree/tp_rank, file_bytes and the sharded entries change. The row and
column windows come from build_tp_plan in qwen38_27b_stagepack.py, the same
plan the checkpoint packer uses, so a split pack and a packer-built pack carry
the same windows. FP8_E4M3_E8M0B128 scales are per row and per 128 columns, so
a row window slices scale rows and a column window slices whole scale groups.
Any sharded entry in another scaled format is refused.

--verify re-reads every rank pack and checks that each entry's shards
reassemble to the source bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

HEADER_BYTES = 120
ENTRY_BYTES = 56
ALIGNMENT = 256
MAGIC = 0x50533651
FORMAT_VERSION = 3
FMT_BF16 = 0
FMT_F32 = 1
FMT_U32 = 2
FMT_E8M0B128 = 6
ELEMENT_BYTES = {FMT_BF16: 2, FMT_F32: 4, FMT_U32: 4, FMT_E8M0B128: 1}
SCALE_GROUP = 128
TP_DEGREE_OFFSET = 96


def load_packer():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("qwen38_27b_stagepack", here / "qwen38_27b_stagepack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ref:
    def __init__(self, kind, layer, rows, columns):
        self.kind = kind
        self.layer = layer
        self.rows = rows
        self.columns = columns


def read_pack(path: Path):
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
        fields = struct.unpack("<26I2Q", header)
        if fields[0] != MAGIC or fields[1] != FORMAT_VERSION or fields[2] != HEADER_BYTES or fields[3] != ENTRY_BYTES:
            raise SystemExit(f"{path}: not a format-{FORMAT_VERSION} qwen38_27b stage pack")
        count, directory_offset = fields[4], fields[26]
        f.seek(directory_offset)
        raw = f.read(count * ENTRY_BYTES)
    entries = []
    for index in range(count):
        kind, layer, fmt, rows, cols, group, poff, pbytes, soff, sbytes = struct.unpack_from("<6I4Q", raw, index * ENTRY_BYTES)
        entries.append(dict(kind=kind, layer=layer, fmt=fmt, rows=rows, cols=cols, group=group,
                            poff=poff, pbytes=pbytes, soff=soff, sbytes=sbytes))
    return bytearray(header), fields, entries


def row_windows(plan, packer):
    if isinstance(plan, packer.TpFusedSlice):
        return [(off, count) for off, count in plan.segments], None
    return [(plan.row_off, plan.row_count)], (plan.col_off, plan.col_count)


def shard_arrays(entry, plan, packer, source):
    fmt, rows, cols = entry["fmt"], entry["rows"], entry["cols"]
    if fmt not in ELEMENT_BYTES:
        raise SystemExit(f"entry kind={entry['kind']} layer={entry['layer']:#x} format {fmt} cannot be sharded")
    width = ELEMENT_BYTES[fmt]
    payload = np.frombuffer(source, dtype=np.uint8, count=rows * cols * width, offset=entry["poff"]).reshape(rows, cols * width)
    scale = None
    if fmt == FMT_E8M0B128:
        if cols % SCALE_GROUP != 0 or entry["sbytes"] != rows * (cols // SCALE_GROUP):
            raise SystemExit(f"entry kind={entry['kind']} layer={entry['layer']:#x} has an unexpected e8m0 scale plane")
        scale = np.frombuffer(source, dtype=np.uint8, count=entry["sbytes"], offset=entry["soff"]).reshape(rows, cols // SCALE_GROUP)
    elif entry["sbytes"] != 0:
        raise SystemExit(f"entry kind={entry['kind']} layer={entry['layer']:#x} format {fmt} carries scales")
    windows, column = row_windows(plan, packer)
    if column is not None and (column[0] != 0 or column[1] != cols):
        if fmt == FMT_E8M0B128 and (column[0] % SCALE_GROUP != 0 or column[1] % SCALE_GROUP != 0):
            raise SystemExit(f"entry kind={entry['kind']} layer={entry['layer']:#x} column window splits a scale group")
        c0, c1 = column[0], column[0] + column[1]
        payload = payload[:, c0 * width:c1 * width]
        if scale is not None:
            scale = scale[:, c0 // SCALE_GROUP:c1 // SCALE_GROUP]
        out_cols = column[1]
    else:
        out_cols = cols
    parts = [payload[off:off + count] for off, count in windows]
    scale_parts = [scale[off:off + count] for off, count in windows] if scale is not None else []
    out_rows = sum(count for _, count in windows)
    return parts, scale_parts, out_rows, out_cols


def write_rank(source_path: Path, output: Path, degree: int, rank: int, packer) -> dict:
    header, fields, entries = read_pack(source_path)
    if fields[24] != 1 or fields[25] != 0:
        raise SystemExit(f"{source_path}: source must be a TP1 pack (tp_degree={fields[24]} tp_rank={fields[25]})")
    source = np.memmap(source_path, dtype=np.uint8, mode="r")
    struct.pack_into("<2I", header, TP_DEGREE_OFFSET, degree, rank)
    directory_offset = fields[26]
    new_entries = []
    tmp = output.with_name(output.name + ".partial")
    with open(tmp, "wb") as f:
        f.write(b"\x00" * (directory_offset + len(entries) * ENTRY_BYTES))
        cursor = directory_offset + len(entries) * ENTRY_BYTES
        for entry in entries:
            plan = packer.build_tp_plan(Ref(entry["kind"], entry["layer"], entry["rows"], entry["cols"]), degree, rank)
            pad = (-cursor) % ALIGNMENT
            f.write(b"\x00" * pad)
            cursor += pad
            if plan is None:
                f.write(source[entry["poff"]:entry["poff"] + entry["pbytes"]].tobytes())
                if entry["sbytes"]:
                    f.write(source[entry["soff"]:entry["soff"] + entry["sbytes"]].tobytes())
                out_rows, out_cols, pbytes, sbytes = entry["rows"], entry["cols"], entry["pbytes"], entry["sbytes"]
            else:
                parts, scale_parts, out_rows, out_cols = shard_arrays(entry, plan, packer, source)
                pbytes = sbytes = 0
                for part in parts:
                    blob = np.ascontiguousarray(part).tobytes()
                    f.write(blob)
                    pbytes += len(blob)
                for part in scale_parts:
                    blob = np.ascontiguousarray(part).tobytes()
                    f.write(blob)
                    sbytes += len(blob)
            new_entries.append(dict(entry, rows=out_rows, cols=out_cols, poff=cursor, pbytes=pbytes,
                                    soff=cursor + pbytes if sbytes else 0, sbytes=sbytes))
            cursor += pbytes + sbytes
        struct.pack_into("<Q", header, 112, cursor)
        f.seek(0)
        f.write(bytes(header))
        f.seek(directory_offset)
        for e in new_entries:
            f.write(struct.pack("<6I4Q", e["kind"], e["layer"], e["fmt"], e["rows"], e["cols"], e["group"],
                                e["poff"], e["pbytes"], e["soff"], e["sbytes"]))
    tmp.rename(output)
    digest = hashlib.sha256()
    with open(output, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    (output.parent / (output.name + ".sha256")).write_text(f"{digest.hexdigest()}  {output.name}\n")
    return {"rank": rank, "pack": str(output), "bytes": cursor, "sha256": digest.hexdigest()}


def verify(source_path: Path, outputs: list[Path], degree: int, packer) -> dict:
    _, _, entries = read_pack(source_path)
    source = np.memmap(source_path, dtype=np.uint8, mode="r")
    ranks = []
    for path in outputs:
        _, fields, rank_entries = read_pack(path)
        ranks.append((np.memmap(path, dtype=np.uint8, mode="r"), rank_entries, fields))
    for rank, (_, _, fields) in enumerate(ranks):
        if fields[24] != degree or fields[25] != rank or fields[27] != Path(outputs[rank]).stat().st_size:
            raise SystemExit(f"rank {rank}: header tp/file_bytes mismatch")
    sharded = replicated = 0
    for index, entry in enumerate(entries):
        plan0 = packer.build_tp_plan(Ref(entry["kind"], entry["layer"], entry["rows"], entry["cols"]), degree, 0)
        for rank, (data, rank_entries, _) in enumerate(ranks):
            got = rank_entries[index]
            if (got["kind"], got["layer"], got["fmt"]) != (entry["kind"], entry["layer"], entry["fmt"]):
                raise SystemExit(f"rank {rank} entry {index}: identity mismatch")
            plan = packer.build_tp_plan(Ref(entry["kind"], entry["layer"], entry["rows"], entry["cols"]), degree, rank)
            if plan is None:
                same = (data[got["poff"]:got["poff"] + got["pbytes"]] == source[entry["poff"]:entry["poff"] + entry["pbytes"]]).all()
                if entry["sbytes"]:
                    same = same and (data[got["soff"]:got["soff"] + got["sbytes"]] == source[entry["soff"]:entry["soff"] + entry["sbytes"]]).all()
                if not same:
                    raise SystemExit(f"rank {rank} entry {index}: replicated bytes differ")
                continue
            parts, scale_parts, rows, cols = shard_arrays(entry, plan, packer, source)
            want = b"".join(np.ascontiguousarray(p).tobytes() for p in parts)
            want_scale = b"".join(np.ascontiguousarray(p).tobytes() for p in scale_parts)
            if (got["rows"], got["cols"]) != (rows, cols) or data[got["poff"]:got["poff"] + got["pbytes"]].tobytes() != want or data[got["soff"]:got["soff"] + got["sbytes"]].tobytes() != want_scale:
                raise SystemExit(f"rank {rank} entry {index}: shard bytes differ")
        if plan0 is None:
            replicated += 1
        else:
            sharded += 1
    return {"verified_entries": len(entries), "sharded": sharded, "replicated": replicated}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output-directory", type=Path, required=True)
    ap.add_argument("--name", required=True, help="rank pack stem; files are <name>.rank<R>.qwen36sp")
    ap.add_argument("--tp-degree", type=int, required=True)
    ap.add_argument("--ranks", default=None, help="comma list of ranks to write (all when absent)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    packer = load_packer()
    if args.tp_degree < 2:
        raise SystemExit("--tp-degree must be at least 2")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_directory / f"{args.name}.rank{rank}.qwen36sp" for rank in range(args.tp_degree)]
    ranks = range(args.tp_degree) if args.ranks is None else [int(r) for r in args.ranks.split(",")]
    for rank in ranks:
        print(json.dumps(write_rank(args.source, outputs[rank], args.tp_degree, rank, packer)), flush=True)
    if args.verify:
        print(json.dumps(verify(args.source, outputs, args.tp_degree, packer)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
