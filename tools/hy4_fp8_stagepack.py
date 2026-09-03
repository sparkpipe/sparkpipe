#!/usr/bin/env python3
"""hy4 lane: pack the FP8 (MXFP8 modelopt) checkpoint into TP16 rank packs.

Emits one safetensors-per-rank bundle plus manifest.json and a .sha256
sidecar per rank, matching the hy4 rank-pack conventions (schema
hy4-fp8-tp16-v1). Bytes move verbatim: FP8 E4M3 weights, their U8 E8M0
MX scale companions and the BF16/F32 exclude-list tensors are sliced, never
requantized. The MTP layer (model.mtp_layers.0.*) is included and sharded
by the same suffix rules as the main stack.

Memory discipline (stagepack-dev pattern, qwen38_max packer): 512 KiB
streaming chunks, posix_fadvise DONTNEED on every source chunk, and
sync_file_range + fadvise DONTNEED on the output after each tensor, so RSS
stays around a few MB and dirty page cache never approaches the reboot
threshold on 119 GB nodes.

Usage:
  python3 tools/hy4_fp8_stagepack.py --checkpoint DIR --dry-run
  python3 tools/hy4_fp8_stagepack.py --checkpoint DIR --rank 7 \
      --output-directory DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

CHUNK_BYTES = 512 * 1024
TP = 16
VOCAB = 120832

# suffix -> (split_dim on the weight tensor, split the scale identically)
# dim0 = HF output rows, dim1 = input columns; scales carry the split dim
# at the same index (their last dim is the MX group count and is never
# split). None = replicate.
SPLIT_RULES = [
    ("model.embed_tokens.weight", 0),
    ("lm_head.weight", 0),
    (".self_attn.q_b_proj.weight", 0),
    (".self_attn.o_proj.weight", 1),
    (".self_attn.kv_b_proj.weight", 0),
    (".self_attn.linear_gate.weight", 0),
    (".self_attn.learnable_sink_param", 0),
    (".self_attn.indexer.wq_b.weight", 0),
    (".mlp.experts.gate_up_proj", 0),
    (".mlp.experts.down_proj", 0),
]
SCALE_SUFFIX = ".weight_scale"


def split_rule(name: str):
    """Returns (split_dim or None, is_base_weight). Scale companions strip
    their suffix (`.weight_scale` or `_scale`) and inherit the base
    weight's split dim."""
    base = name
    if base.endswith(SCALE_SUFFIX):
        base = base[: -len(SCALE_SUFFIX)]
    elif base.endswith("_scale"):
        base = base[: -len("_scale")]
    for suffix, dim in SPLIT_RULES:
        if base.endswith(suffix) or base == suffix:
            return (dim, base == name)
    return (None, base == name)


def load_headers(checkpoint: Path, weight_map: dict):
    headers = {}
    data_starts = {}
    for shard in sorted(set(weight_map.values())):
        with open(checkpoint / shard, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            headers[shard] = json.loads(f.read(n))
            data_starts[shard] = 8 + n
    return headers, data_starts


def plan_tensors(checkpoint: Path, weight_map: dict, headers: dict,
                 data_starts: dict):
    dtsize = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "U8": 1}
    entries = []
    for name, shard in weight_map.items():
        meta = headers[shard][name]
        dims = list(meta["shape"])
        dim, is_weight = split_rule(name)
        esize = dtsize[meta["dtype"]]
        row_bytes = esize * (dims[1] if len(dims) > 1 else 1)
        if len(dims) >= 3:
            row_bytes = esize * dims[-1]
            for d in dims[1:-1]:
                row_bytes *= d
        entry = {
            "name": name, "dtype": meta["dtype"], "dims": dims,
            "shard": shard, "esize": esize,
            "data_start": data_starts[shard],
            "start": headers[shard][name]["data_offsets"][0],
            "end": headers[shard][name]["data_offsets"][1],
            "row_bytes": row_bytes if len(dims) > 1 else esize * dims[0],
            "split": dim,
        }
        entries.append(entry)
    entries.sort(key=lambda e: (e["shard"], e["start"]))
    return entries


def rank_view(entry: dict, rank: int):
    """Returns (out_dims, copy plan) for this rank: contiguous = (start,
    end) byte range within the tensor; gather = list of (src_off, nbytes)
    per outer row; replicate = full copy."""
    dims = entry["dims"]
    dim = entry["split"]
    if dim is None:
        return (list(dims), ("full", 0, 0))
    total = dims[dim]
    if total % TP:
        raise SystemExit(f"{entry['name']}: dim{dim} {total} not divisible "
                         f"by {TP}")
    chunk = total // TP
    lo, hi = rank * chunk, (rank + 1) * chunk
    esize = entry["esize"]
    if dim == 0:
        out = [chunk] + dims[1:]
        inner = entry["row_bytes"]
        return (out, ("range", lo * inner, hi * inner))
    inner = dims[-1] * esize
    group = dims[1:-1]
    rows = dims[0]
    for g in group:
        rows *= g
    out = list(dims)
    out[dim] = chunk
    c0, c1 = lo * esize, hi * esize
    return (out, ("gather", rows, (c0, c1)))


def stream_copy(src, dst_fd, nbytes, offset, chunk_hash=None):
    src.seek(offset)
    done = 0
    while done < nbytes:
        step = min(CHUNK_BYTES, nbytes - done)
        buf = src.read(step)
        if len(buf) != step:
            raise SystemExit("short source read")
        os.write(dst_fd, buf)
        if chunk_hash is not None:
            chunk_hash.update(buf)
        try:
            os.posix_fadvise(src.fileno(), offset + done, step,
                             os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        done += step


def output_flush(fd, start, end):
    try:
        if hasattr(os, "sync_file_range"):
            os.sync_file_range(fd, start, end - start,
                               os.SYNC_FILE_RANGE_WRITE)
        os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def build_rank(entries, rank: int, checkpoint: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
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
        if cursor % 8:
            cursor += 8 - cursor % 8
        header[e["name"]] = {"dtype": e["dtype"], "shape": out_dims,
                             "data_offsets": [cursor, cursor + nbytes]}
        plan.append((e, spec, cursor, nbytes))
        cursor += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    path = out_dir / f"model-fp8-tp16-rank-{rank:02d}.safetensors"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, struct.pack("<Q", len(blob)))
    os.write(fd, blob)
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(blob)))
    digest.update(blob)
    src = {}
    for (e, spec, off, nbytes) in plan:
        shard = e["shard"]
        if shard not in src:
            src[shard] = open(checkpoint / shard, "rb")
        before = os.lseek(fd, 0, os.SEEK_CUR)
        base = e["data_start"]
        if spec[0] == "full":
            stream_copy(src[shard], fd, nbytes, base + e["start"], digest)
        elif spec[0] == "range":
            stream_copy(src[shard], fd, nbytes,
                        base + e["start"] + spec[1], digest)
        else:
            rows, (c0, c1) = spec[1], spec[2]
            row_bytes = e["row_bytes"]
            for r in range(rows):
                src[shard].seek(base + e["start"] + r * row_bytes)
                got = src[shard].read(row_bytes)
                if len(got) != row_bytes:
                    raise SystemExit(f"short read {e['name']} row {r}")
                os.write(fd, got[c0:c1])
                digest.update(got[c0:c1])
            try:
                os.posix_fadvise(src[shard].fileno(),
                                 base + e["start"], rows * row_bytes,
                                 os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        output_flush(fd, before, os.lseek(fd, 0, os.SEEK_CUR) - before)
        os.lseek(fd, 0, os.SEEK_END)
        if os.environ.get("HY4_PACK_VERIFY_WRITE"):
            pbase_cur = 8 + len(blob)
            got = os.pread(fd, nbytes, pbase_cur + off)
            f2 = open(checkpoint / e["shard"], "rb")
            b2 = e["data_start"] + e["start"]
            if spec[0] == "range":
                b2 += spec[1]
            f2.seek(b2)
            want = f2.read(nbytes)
            f2.close()
            if got != want:
                dif = next((i for i in range(nbytes)
                            if got[i] != want[i]), -1)
                shift = -1
                window = os.pread(fd, nbytes + (1 << 21),
                                  max(0, pbase_cur + off - (1 << 20)))
                at = window.find(want[:64])
                if at >= 0:
                    shift = (max(0, pbase_cur + off - (1 << 20)) + at
                             - (pbase_cur + off))
                raise SystemExit(
                    f"VERIFY-WRITE MISMATCH {e['name']} off={off} "
                    f"nbytes={nbytes} first_diff={dif} shift={shift} "
                    f"src={e['shard']} start={e['start']}")
            print(f"  vw ok {e['name']}", flush=True)
    for s in src.values():
        s.close()
    os.close(fd)
    file_digest = hashlib.sha256()
    with open(path, "rb") as rf:
        for chunk in iter(lambda: rf.read(1 << 22), b""):
            file_digest.update(chunk)
    hexd = file_digest.hexdigest()
    (out_dir / (path.name + ".sha256")).write_text(
        f"{hexd}  {path.name}\n")
    manifest = {
        "schema": "hy4-fp8-tp16-v1",
        "rank": rank,
        "ranks": TP,
        "source_checkpoint": str(checkpoint),
        "file": path.name,
        "file_sha256": hexd,
        "tensors": [
            {"name": e["name"], "dtype": e["dtype"], "dims": e["dims"],
             "slice": ("replicate" if spec[0] == "full" else
                       {"dim": e["split"], "start": rank *
                        (e["dims"][e["split"]] // TP), "kind": spec[0]})}
            for (e, spec, off, nbytes) in plan],
    }
    (out_dir / f"manifest-rank-{rank:02d}.json").write_text(
        json.dumps(manifest, indent=1))
    return {"tensor_count": len(plan), "bytes": cursor + 8 + len(blob),
            "sha256": hexd}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-directory", type=Path)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--manifest-only", action="store_true",
                    help="regenerate the rank manifest + sha sidecar from "
                         "the existing pack file without re-copying data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    index = json.load(open(args.checkpoint / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    headers, data_starts = load_headers(args.checkpoint, weight_map)
    entries = plan_tensors(args.checkpoint, weight_map, headers, data_starts)

    if args.dry_run:
        import collections
        acts = collections.Counter()
        bytes_by = collections.Counter()
        replicated_gb = 0.0
        for e in entries:
            out_dims, spec = rank_view(e, 0)
            nb = e["esize"]
            for d in out_dims:
                nb *= d
            kind = spec[0]
            acts[kind] += 1
            bytes_by[kind] += nb
            if kind == "full":
                replicated_gb += e["end"] - e["start"]
        total_src = sum(e["end"] - e["start"] for e in entries)
        print(f"tensors {len(entries)}; source bytes {total_src/1e9:.1f}G")
        for kind in acts:
            print(f"  {kind:7s}: {acts[kind]:5d} tensors, "
                  f"{bytes_by[kind]/1e9:.1f}G per rank")
        rep_names = [e["name"] for e in entries if e["split"] is None]
        print(f"replicated tensor count: {len(rep_names)}")
        big = sorted(rep_names, key=lambda n: -(headers[weight_map[n]][n]
                                                ["data_offsets"][1] -
                                                headers[weight_map[n]][n]
                                                ["data_offsets"][0]))[:8]
        for n in big:
            v = headers[weight_map[n]][n]
            print(f"  rep {n} {v['dtype']} {v['shape']}")
        return 0

    if args.rank is None:
        ap.error("--rank required without --dry-run")
    if args.manifest_only:
        out_dir = args.output_directory
        path = out_dir / f"model-fp8-tp16-rank-{args.rank:02d}.safetensors"
        file_digest = hashlib.sha256()
        with open(path, "rb") as rf:
            for chunk in iter(lambda: rf.read(1 << 22), b""):
                file_digest.update(chunk)
        hexd = file_digest.hexdigest()
        (out_dir / (path.name + ".sha256")).write_text(
            f"{hexd}  {path.name}\n")
        plan = []
        for e in entries:
            out_dims, sp = rank_view(e, args.rank)
            plan.append((e, sp, 0, 0))
        manifest_tensors = []
        for (e, spec, off, nbytes) in plan:
            if spec[0] == "full":
                sl = "replicate"
            else:
                if e["split"] is None:
                    raise SystemExit(
                        f"manifest: {e['name']} non-full with split None "
                        f"({spec[0]})")
                sl = {"dim": e["split"],
                      "start": args.rank * (e["dims"][e["split"]] // TP),
                      "kind": spec[0]}
            manifest_tensors.append({"name": e["name"],
                                     "dtype": e["dtype"],
                                     "dims": e["dims"], "slice": sl})
        manifest = {
            "schema": "hy4-fp8-tp16-v1",
            "rank": args.rank,
            "ranks": TP,
            "source_checkpoint": str(args.checkpoint),
            "file": path.name,
            "file_sha256": hexd,
            "tensors": manifest_tensors,
        }
        (out_dir / f"manifest-rank-{args.rank:02d}.json").write_text(
            json.dumps(manifest, indent=1))
        print(f"rank{args.rank:02d} manifest+sha regenerated "
              f"({hexd[:16]})", flush=True)
        return 0
    result = build_rank(entries, args.rank, args.checkpoint,
                        args.output_directory)
    print(f"rank{args.rank:02d} ok tensors={result['tensor_count']} "
          f"bytes={result['bytes']} sha={result['sha256'][:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
