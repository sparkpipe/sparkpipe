import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys

COPY_CHUNK = 64 << 20
SIDE_FILES = ("config.json", "generation_config.json", "ARCHIVE-RECEIPT.json", "LICENSE")


def read_header(path):
    with open(path, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def select(checkpoint, patterns):
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
    else:
        single = os.path.join(checkpoint, "model.safetensors")
        weight_map = {name: "model.safetensors" for name in read_header(single)[0]}
    compiled = [re.compile(p) for p in patterns]
    names = sorted(n for n in weight_map if any(c.fullmatch(n) for c in compiled))
    unmatched = [p for p, c in zip(patterns, compiled) if not any(c.fullmatch(n) for n in weight_map)]
    if unmatched:
        raise SystemExit(f"SUBSET-FAIL patterns matched nothing: {unmatched}")
    rows = []
    headers = {}
    for name in names:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = read_header(os.path.join(checkpoint, shard))
        header, base = headers[shard]
        entry = header[name]
        start, end = entry["data_offsets"]
        rows.append({"name": name, "shard": shard, "dtype": entry["dtype"], "shape": entry["shape"],
                     "source_offset": base + start, "bytes": end - start})
    rows.sort(key=lambda r: (r["shard"], r["source_offset"]))
    return rows


def write_subset(checkpoint, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    header = {}
    offset = 0
    for row in rows:
        header[row["name"]] = {"dtype": row["dtype"], "shape": row["shape"],
                               "data_offsets": [offset, offset + row["bytes"]]}
        offset += row["bytes"]
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    target = os.path.join(out_dir, "model.safetensors")
    partial = target + ".partial"
    file_hash = hashlib.sha256()
    with open(partial, "wb") as out:
        prefix = struct.pack("<Q", len(blob)) + blob
        out.write(prefix)
        file_hash.update(prefix)
        handles = {}
        for row in rows:
            source = handles.get(row["shard"])
            if source is None:
                source = handles[row["shard"]] = open(os.path.join(checkpoint, row["shard"]), "rb")
            source.seek(row["source_offset"])
            tensor_hash = hashlib.sha256()
            left = row["bytes"]
            while left:
                piece = source.read(min(COPY_CHUNK, left))
                if not piece:
                    raise SystemExit(f"SUBSET-FAIL short read in {row['shard']} for {row['name']}")
                out.write(piece)
                tensor_hash.update(piece)
                file_hash.update(piece)
                left -= len(piece)
            row["sha256"] = tensor_hash.hexdigest()
        for handle in handles.values():
            handle.close()
    os.replace(partial, target)
    json.dump({"metadata": {"total_size": offset}, "weight_map": {r["name"]: "model.safetensors" for r in rows}},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1, sort_keys=True)
    for side in SIDE_FILES:
        if os.path.exists(os.path.join(checkpoint, side)):
            shutil.copyfile(os.path.join(checkpoint, side), os.path.join(out_dir, side))
    manifest = {"format": "safetensors-subset-v1", "source": os.path.abspath(checkpoint),
                "file_sha256": file_hash.hexdigest(), "payload_bytes": offset, "tensor_count": len(rows),
                "tensors": [{k: r[k] for k in ("name", "shard", "dtype", "shape", "bytes", "sha256")} for r in rows]}
    json.dump(manifest, open(os.path.join(out_dir, "SUBSET.json"), "w"), indent=1)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="copy named tensors of a safetensors checkpoint into one file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include", action="append", required=True, help="full-match regex on tensor names")
    parser.add_argument("--plan", action="store_true", help="print the selection and bytes, write nothing")
    args = parser.parse_args(argv)
    rows = select(args.checkpoint, args.include)
    total = sum(r["bytes"] for r in rows)
    if args.plan:
        for row in rows:
            print(f"{row['name']} {row['dtype']} {row['shape']} {row['bytes']}")
        print(f"SUBSET-PLAN tensors={len(rows)} bytes={total}")
        return 0
    manifest = write_subset(args.checkpoint, rows, args.out)
    print(f"SUBSET-PASS tensors={manifest['tensor_count']} bytes={manifest['payload_bytes']} "
          f"sha256={manifest['file_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
