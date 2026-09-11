#!/usr/bin/env python3
"""hy4 lane: single-pass fan-out packer for the FP8 TP16 rank set.

Reads the FP8 checkpoint ONCE (sequential, per-tensor order) and fans each
tensor's bytes out to all sixteen rank outputs in the same pass — 813 GB of
warm reads total regardless of rank count, versus 16 full passes for the
per-rank packer. Memory stays bounded: 512 KiB read chunks plus one row
buffer for dim-1 gathers; fadvise DONTNEED on source chunks and
sync_file_range + DONTNEED on outputs per tensor.

Emit formats match hy4_fp8_stagepack.py exactly (schema hy4-fp8-tp16-v1,
per-rank safetensors + manifest-rank-XX.json + .sha256 sidecar).

Usage:
  python3 tools/hy4_fp8_stagepack_fanout.py --checkpoint DIR \
      --output-directory DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

from hy4_fp8_stagepack import (CHUNK_BYTES, SCALE_SUFFIX, TP, load_headers,
                               plan_tensors, rank_view)

ALIGN = 8


def out_fd_write(fd, data, digest):
    os.write(fd, data)
    digest.update(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-directory", type=Path, required=True)
    args = ap.parse_args(argv)
    out_dir = args.output_directory
    out_dir.mkdir(parents=True, exist_ok=True)

    index = json.load(open(args.checkpoint / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    headers, data_starts = load_headers(args.checkpoint, weight_map)
    entries = plan_tensors(args.checkpoint, weight_map, headers, data_starts)

    per_rank = []
    for rank in range(TP):
        plan = []
        header = {"__metadata__": {
            "format": "hy4-fp8-tp16-v1", "tp_degree": str(TP),
            "tp_rank": str(rank)}}
        cursor = 0
        for e in entries:
            out_dims, spec = rank_view(e, rank)
            nbytes = e["esize"]
            for d in out_dims:
                nbytes *= d
            if cursor % ALIGN:
                cursor += ALIGN - cursor % ALIGN
            header[e["name"]] = {"dtype": e["dtype"], "shape": out_dims,
                                 "data_offsets": [cursor, cursor + nbytes]}
            plan.append((e, spec, cursor, nbytes))
            cursor += nbytes
        blob = json.dumps(header, separators=(",", ":")).encode()
        per_rank.append({"plan": plan, "blob": blob, "digest":
                         hashlib.sha256(), "bytes": 0, "path":
                         out_dir / f"model-fp8-tp16-rank-{rank:02d}.safetensors"})
    for pr in per_rank:
        pr["fd"] = os.open(pr["path"], os.O_RDWR | os.O_CREAT | os.O_TRUNC,
                           0o644)
        os.write(pr["fd"], struct.pack("<Q", len(pr["blob"])))
        os.write(pr["fd"], pr["blob"])
        pr["digest"].update(struct.pack("<Q", len(pr["blob"])))
        pr["digest"].update(pr["blob"])
        pr["pos"] = 8 + len(pr["blob"])
        del pr["blob"]

    src = {}
    for si, e in enumerate(entries):
        shard = e["shard"]
        if shard not in src:
            src[shard] = open(args.checkpoint / shard, "rb")
        base = e["data_start"] + e["start"]
        nbytes_total = e["end"] - e["start"]
        if si % 50 == 0:
            print(f"tensor {si}/{len(entries)} {e['name']} "
                  f"({nbytes_total / 1e9:.2f} GB)", flush=True)
        dim = e["split"]
        if dim is None:
            done = 0
            while done < nbytes_total:
                step = min(CHUNK_BYTES, nbytes_total - done)
                src[shard].seek(base + done)
                buf = src[shard].read(step)
                if len(buf) != step:
                    raise SystemExit(f"short read {e['name']}")
                for pr in per_rank:
                    out_fd_write(pr["fd"], buf, pr["digest"])
                    pr["pos"] += len(buf)
                try:
                    os.posix_fadvise(src[shard].fileno(), base + done, step,
                                     os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
                done += step
        elif dim == 0:
            row_bytes = e["row_bytes"]
            rows = nbytes_total // row_bytes
            chunk = e["dims"][0] // TP
            done = 0
            for r in range(rows):
                src[shard].seek(base + done)
                buf = src[shard].read(row_bytes)
                if len(buf) != row_bytes:
                    raise SystemExit(f"short read {e['name']} row {r}")
                owner = min(r // chunk, TP - 1)
                pr = per_rank[owner]
                out_fd_write(pr["fd"], buf, pr["digest"])
                pr["pos"] += len(buf)
                done += row_bytes
            try:
                os.posix_fadvise(src[shard].fileno(), base, nbytes_total,
                                 os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        else:
            row_bytes = e["row_bytes"]
            esize = e["esize"]
            total = e["dims"][dim]
            chunk = total // TP
            rows = nbytes_total // row_bytes
            for r in range(rows):
                src[shard].seek(base + r * row_bytes)
                row = src[shard].read(row_bytes)
                if len(row) != row_bytes:
                    raise SystemExit(f"short read {e['name']} row {r}")
                for rank in range(TP):
                    lo = rank * chunk * esize
                    hi = (rank + 1) * chunk * esize
                    pr = per_rank[rank]
                    out_fd_write(pr["fd"], row[lo:hi], pr["digest"])
                    pr["pos"] += hi - lo
            try:
                os.posix_fadvise(src[shard].fileno(), base, nbytes_total,
                                 os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    for rank, pr in enumerate(per_rank):
        os.close(pr["fd"])
        digest = hashlib.sha256()
        with open(pr["path"], "rb") as rf:
            for chunk in iter(lambda: rf.read(1 << 22), b""):
                digest.update(chunk)
        hexd = digest.hexdigest()
        sidecar_name = pr["path"].name + ".sha256"
        (out_dir / sidecar_name).write_text(
            "{}  {}\n".format(hexd, pr["path"].name))
        manifest_tensors = []
        for (e, spec, off, nbytes) in pr["plan"]:
            if spec[0] == "full":
                sl = "replicate"
            else:
                sl = {"dim": e["split"], "kind": spec[0],
                      "start": rank * (e["dims"][e["split"]] // TP)}
            manifest_tensors.append({"name": e["name"],
                                     "dtype": e["dtype"],
                                     "dims": e["dims"], "slice": sl})
        manifest = {
            "schema": "hy4-fp8-tp16-v1",
            "rank": rank,
            "ranks": TP,
            "source_checkpoint": str(args.checkpoint),
            "file": pr["path"].name,
            "file_sha256": hexd,
            "tensors": manifest_tensors,
        }
        (out_dir / f"manifest-rank-{rank:02d}.json").write_text(
            json.dumps(manifest, indent=1))
        print(f"rank{rank:02d} done sha={hexd[:16]} "
              f"bytes={pr['pos']}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
