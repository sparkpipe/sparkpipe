#!/usr/bin/env python3
"""hy4 lane: pack the FP8 (MXFP8 modelopt) checkpoint into TP16 rank packs.

Emits one safetensors-per-rank bundle plus manifest.json and a .sha256
sidecar per rank, matching the hy4 rank-pack conventions (schema
hy4-fp8-tp16-v1). Bytes move verbatim: FP8 E4M3 weights and the BF16/F32
exclude-list tensors are sliced, never requantized; U8 E8M0 MX scale
companions follow the scale-row-offset contract (SCALE_REPLICATED below).
The MTP layer (model.mtp_layers.0.*) is included and sharded
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
import collections
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_pack_codecs import dtype_bytes  # noqa: E402
from spark_pack_common import read_safetensors_header, sha256_file, window_chunks, write_sha256_sidecar  # noqa: E402

CHUNK_BYTES = 512 * 1024
TP = 16
VOCAB = 120832
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
SCALE_REPLICATED = (
    ".self_attn.q_b_proj.weight_scale",
    ".self_attn.kv_b_proj.weight_scale",
    ".self_attn.o_proj.weight_scale",
    ".self_attn.indexer.wq_b.weight_scale",
)
SCHEMA = "hy4-fp8-tp16-v1"


def split_rule(name: str):
    if name.endswith(SCALE_REPLICATED):
        return (None, False)
    base = name[:-len("_scale")] if name.endswith("_scale") else name
    for suffix, dim in SPLIT_RULES:
        if base.endswith(suffix) or base == suffix:
            return (dim, base == name)
    return (None, base == name)


def load_headers(checkpoint: Path, weight_map: dict):
    headers, data_starts = {}, {}
    for shard in sorted(set(weight_map.values())):
        headers[shard], data_starts[shard], _ = read_safetensors_header(checkpoint / shard)
    return headers, data_starts


def plan_tensors(checkpoint: Path, weight_map: dict, headers: dict, data_starts: dict):
    entries = []
    for name, shard in weight_map.items():
        meta = headers[shard][name]
        dims = list(meta["shape"])
        dim, _ = split_rule(name)
        esize = dtype_bytes(meta["dtype"])
        row_bytes = esize * (dims[1] if len(dims) > 1 else 1)
        if len(dims) >= 3:
            row_bytes = esize * dims[-1]
            for extent in dims[1:-1]:
                row_bytes *= extent
        entries.append({"name": name, "dtype": meta["dtype"], "dims": dims, "shard": shard, "esize": esize,
                        "data_start": data_starts[shard], "start": meta["data_offsets"][0],
                        "end": meta["data_offsets"][1], "row_bytes": row_bytes if len(dims) > 1 else esize * dims[0],
                        "split": dim})
    entries.sort(key=lambda e: (e["shard"], e["start"]))
    return entries


def rank_view(entry: dict, rank: int):
    dims = entry["dims"]
    dim = entry["split"]
    if dim is None:
        return (list(dims), ("full", 0, 0))
    total = dims[dim]
    if total % TP:
        raise SystemExit(f"{entry['name']}: dim{dim} {total} not divisible by {TP}")
    chunk = total // TP
    lo, hi = rank * chunk, (rank + 1) * chunk
    esize = entry["esize"]
    if dim == 0:
        inner = entry["row_bytes"] if len(dims) > 1 else esize
        return ([chunk] + dims[1:], ("range", lo * inner, hi * inner))
    rows = dims[0]
    for extent in dims[1:-1]:
        rows *= extent
    out = list(dims)
    out[dim] = chunk
    return (out, ("gather", rows, (lo * esize, hi * esize)))


def drop_cache(fd: int, offset: int, length: int) -> None:
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def output_flush(fd, start, end):
    try:
        if hasattr(os, "sync_file_range"):
            os.sync_file_range(fd, start, end - start, os.SYNC_FILE_RANGE_WRITE)
        os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def tensor_chunks(source, entry: dict, spec, nbytes: int):
    base = entry["data_start"] + entry["start"]
    read_at = lambda offset, length: os.pread(source.fileno(), length, offset)
    if spec[0] == "gather":
        rows, (c0, c1) = spec[1], spec[2]
        yield from window_chunks(read_at, base, entry["row_bytes"], 0, rows, c0, c1 - c0, entry["row_bytes"],
                                 f"{entry['name']}")
        drop_cache(source.fileno(), base, rows * entry["row_bytes"])
        return
    offset = base + (spec[1] if spec[0] == "range" else 0)
    done = 0
    while done < nbytes:
        step = min(CHUNK_BYTES, nbytes - done)
        chunk = os.pread(source.fileno(), step, offset + done)
        if len(chunk) != step:
            raise SystemExit("short source read")
        yield chunk
        drop_cache(source.fileno(), offset + done, step)
        done += step


def rank_manifest(entries, rank: int, checkpoint: Path, name: str, digest: str) -> dict:
    tensors = []
    for entry in entries:
        _, spec = rank_view(entry, rank)
        slice_ = "replicate" if spec[0] == "full" else {"dim": entry["split"],
                                                         "start": rank * (entry["dims"][entry["split"]] // TP),
                                                         "kind": spec[0]}
        tensors.append({"name": entry["name"], "dtype": entry["dtype"], "dims": entry["dims"], "slice": slice_})
    return {"schema": SCHEMA, "rank": rank, "ranks": TP, "source_checkpoint": str(checkpoint), "file": name,
            "file_sha256": digest, "tensors": tensors}


def rank_file(out_dir: Path, rank: int) -> Path:
    return out_dir / f"model-fp8-tp16-rank-{rank:02d}.safetensors"


def write_manifest(out_dir: Path, entries, rank: int, checkpoint: Path, path: Path, digest: str) -> None:
    write_sha256_sidecar(path, digest)
    (out_dir / f"manifest-rank-{rank:02d}.json").write_text(
        json.dumps(rank_manifest(entries, rank, checkpoint, path.name, digest), indent=1))


def verify_write(fd: int, checkpoint: Path, entry: dict, spec, offset: int, nbytes: int) -> None:
    got = os.pread(fd, nbytes, offset)
    with open(checkpoint / entry["shard"], "rb") as source:
        want = b"".join(bytes(chunk) for chunk in tensor_chunks(source, entry, spec, nbytes))
    if got != want:
        first_diff = next((index for index in range(nbytes) if got[index] != want[index]), -1)
        window_start = max(0, offset - (1 << 20))
        at = os.pread(fd, nbytes + (1 << 21), window_start).find(want[:64])
        shift = window_start + at - offset if at >= 0 else -1
        raise SystemExit(f"VERIFY-WRITE MISMATCH {entry['name']} off={offset} nbytes={nbytes} first_diff={first_diff} "
                         f"shift={shift} src={entry['shard']} start={entry['start']}")
    print(f"  vw ok {entry['name']}", flush=True)


def build_rank(entries, rank: int, checkpoint: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    header = {"__metadata__": {"format": SCHEMA, "tp_degree": str(TP), "tp_rank": str(rank)}}
    cursor = 0
    for entry in entries:
        out_dims, spec = rank_view(entry, rank)
        nbytes = entry["esize"]
        for extent in out_dims:
            nbytes *= extent
        cursor += (-cursor) % 8
        header[entry["name"]] = {"dtype": entry["dtype"], "shape": out_dims, "data_offsets": [cursor, cursor + nbytes]}
        plan.append((entry, spec, cursor, nbytes))
        cursor += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    path = rank_file(out_dir, rank)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, struct.pack("<Q", len(blob)) + blob)
    payload_base = 8 + len(blob)
    sources = {}
    try:
        for entry, spec, offset, nbytes in plan:
            if entry["shard"] not in sources:
                sources[entry["shard"]] = open(checkpoint / entry["shard"], "rb")
            before = os.lseek(fd, 0, os.SEEK_CUR)
            if payload_base + offset > before:
                os.write(fd, bytes(payload_base + offset - before))
            for chunk in tensor_chunks(sources[entry["shard"]], entry, spec, nbytes):
                os.write(fd, chunk)
            output_flush(fd, before, os.lseek(fd, 0, os.SEEK_CUR) - before)
            os.lseek(fd, 0, os.SEEK_END)
            if os.environ.get("HY4_PACK_VERIFY_WRITE"):
                verify_write(fd, checkpoint, entry, spec, payload_base + offset, nbytes)
    finally:
        for source in sources.values():
            source.close()
        os.close(fd)
    digest = sha256_file(path)
    write_manifest(out_dir, entries, rank, checkpoint, path, digest)
    return {"tensor_count": len(plan), "bytes": cursor + payload_base, "sha256": digest}


def dry_run(entries, headers, weight_map) -> int:
    acts, bytes_by = collections.Counter(), collections.Counter()
    for entry in entries:
        out_dims, spec = rank_view(entry, 0)
        nbytes = entry["esize"]
        for extent in out_dims:
            nbytes *= extent
        acts[spec[0]] += 1
        bytes_by[spec[0]] += nbytes
    total_source = sum(entry["end"] - entry["start"] for entry in entries)
    print(f"tensors {len(entries)}; source bytes {total_source/1e9:.1f}G")
    for kind in acts:
        print(f"  {kind:7s}: {acts[kind]:5d} tensors, {bytes_by[kind]/1e9:.1f}G per rank")
    replicated = [entry["name"] for entry in entries if entry["split"] is None]
    print(f"replicated tensor count: {len(replicated)}")
    sizes = {name: headers[weight_map[name]][name]["data_offsets"][1] - headers[weight_map[name]][name]["data_offsets"][0]
             for name in replicated}
    for name in sorted(replicated, key=lambda name: -sizes[name])[:8]:
        meta = headers[weight_map[name]][name]
        print(f"  rep {name} {meta['dtype']} {meta['shape']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--manifest-only", action="store_true",
                        help="regenerate the rank manifest + sha sidecar from the existing pack file without "
                             "re-copying data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    weight_map = json.load(open(args.checkpoint / "model.safetensors.index.json"))["weight_map"]
    headers, data_starts = load_headers(args.checkpoint, weight_map)
    entries = plan_tensors(args.checkpoint, weight_map, headers, data_starts)
    if args.dry_run:
        return dry_run(entries, headers, weight_map)
    if args.rank is None:
        parser.error("--rank required without --dry-run")
    if args.manifest_only:
        path = rank_file(args.output_directory, args.rank)
        digest = sha256_file(path)
        write_manifest(args.output_directory, entries, args.rank, args.checkpoint, path, digest)
        print(f"rank{args.rank:02d} manifest+sha regenerated ({digest[:16]})", flush=True)
        return 0
    result = build_rank(entries, args.rank, args.checkpoint, args.output_directory)
    print(f"rank{args.rank:02d} ok tensors={result['tensor_count']} bytes={result['bytes']} "
          f"sha={result['sha256'][:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
