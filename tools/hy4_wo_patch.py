#!/usr/bin/env python3
"""Repair the dim-0 (o_proj) slices inside built hy4 TP16 rank bundles.

The original sharder wrote split0 tensors as one contiguous span of the
source, but a dim-0 slice of a GGML 2D tensor is a strided gather: every
source row contributes its rank-owned blocks. Built bundles therefore hold
wrong bytes for every blk.<N>.attn_output.weight while passing size, dims
and digest checks. This tool rewrites those regions in place from the
source GGUF (region sizes are unchanged, so the GGUF layout stays valid),
recomputes each rank file digest, and refreshes the manifest + sidecar.

Usage:
  python3 tools/hy4_wo_patch.py --gguf SRC.gguf --out ALLRANKS_DIR [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hy4_tp16_shard import GGML_TYPES, GgufReader  # noqa: E402

TARGET = re.compile(r"^blk\.(\d+)\.attn_output\.weight$")
EXPECTED_BYTES = 18724364736


def inside(base: Path, name: str) -> Path:
    joined = (base / name).resolve()
    if base.resolve() not in joined.parents:
        raise SystemExit(f"path escapes {base}: {joined}")
    return joined


def rank_gguf(rdir: Path) -> Path:
    ggufs = [inside(rdir, p.name) for p in rdir.iterdir()
             if p.is_file() and p.suffix == ".gguf"]
    if len(ggufs) != 1:
        raise SystemExit(f"expected one gguf in {rdir}")
    return ggufs[0]


def gather_geometry(src_dims: list[int], slice_dims: list[int],
                    gtype: int, rank: int) -> dict:
    blck, bpb = GGML_TYPES[gtype]
    in_dim, src_rows = src_dims
    chunk, out_rows = slice_dims
    if in_dim % blck or chunk % blck:
        raise SystemExit(f"dim0 not block aligned: {src_dims} chunk {chunk}")
    if src_rows != out_rows:
        raise SystemExit(f"row mismatch src {src_rows} slice {out_rows}")
    blocks_per_row = in_dim // blck
    row_bytes = blocks_per_row * bpb
    piece_bytes = chunk // blck * bpb
    return {"rows": out_rows,
            "src_row_bytes": row_bytes,
            "piece_bytes": piece_bytes,
            "src_row_offset": rank * chunk // blck * bpb}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", required=True,
                        help="directory containing rank-XX bundles")
    parser.add_argument("--check", action="store_true",
                        help="report differences without writing")
    args = parser.parse_args()

    source = GgufReader(Path(args.gguf))
    src = {t["name"]: t for t in source.tensors}
    out_root = Path(args.out)
    rank_dirs = sorted(out_root.glob("rank-*"))
    if not rank_dirs:
        raise SystemExit(f"no rank dirs under {out_root}")

    for rdir in rank_dirs:
        m = re.search(r"(\d+)$", rdir.name)
        if not m:
            raise SystemExit(f"bad rank dir name {rdir}")
        rank = int(m.group(1))
        gguf_path = next(rdir.glob("*.gguf"))
        rr = GgufReader(gguf_path)
        rr.file.close()
        rr.file = open(gguf_path, "r+b")
        patched = 0
        changed_bytes = 0
        order = sorted(rr.tensors, key=lambda t: t["offset"])
        ends = {t["offset"]: o["offset"] for t, o in zip(order, order[1:])}
        file_end = rr.file.seek(0, 2)
        for rt in rr.tensors:
            tm = TARGET.match(rt["name"])
            if not tm:
                continue
            st = src.get(rt["name"])
            if st is None:
                raise SystemExit(f"source lacks {rt['name']}")
            geo = gather_geometry(st["dims"], rt["dims"], rt["type"], rank)
            expect = geo["rows"] * geo["piece_bytes"]
            end = ends.get(rt["offset"])
            if end is None:
                end = file_end - rr.data_offset
            gap = end - rt["offset"]
            if not expect <= gap < expect + rr.alignment:
                raise SystemExit(
                    f"{rt['name']} rank {rank}: region gap {gap} bytes "
                    f"!= expected {expect}")
            src_base = source.data_offset + st["offset"]
            dst_base = rr.data_offset + rt["offset"]
            for row in range(geo["rows"]):
                s = row * geo["src_row_bytes"] + geo["src_row_offset"]
                source.file.seek(src_base + s)
                want = source.file.read(geo["piece_bytes"])
                rr.file.seek(dst_base + row * geo["piece_bytes"])
                have = rr.file.read(geo["piece_bytes"])
                if have != want:
                    if args.check:
                        patched += 1
                        changed_bytes += geo["piece_bytes"]
                        continue
                    rr.file.seek(dst_base + row * geo["piece_bytes"])
                    rr.file.write(want)
                    patched += 1
                    changed_bytes += geo["piece_bytes"]
        if args.check:
            print(f"rank {rank:02d}: {patched} rows differ "
                  f"({changed_bytes} bytes) -- check only")
            continue
        rr.file.flush()
        rr.file.close()
        digest = hashlib.sha256()
        with open(gguf_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                digest.update(chunk)
        hexd = digest.hexdigest()
        sidecar = gguf_path.with_suffix(".gguf.sha256")
        sidecar.write_text(f"{hexd}  {gguf_path.name}\n")
        manifest_path = rdir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["gguf_sha256"] = hexd
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"rank {rank:02d}: rewrote {patched} rows "
              f"({changed_bytes} bytes), sha256 {hexd[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
