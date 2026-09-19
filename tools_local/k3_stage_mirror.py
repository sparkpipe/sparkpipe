import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

SRC = "/mnt/model-warm/kimi-k3"
DST = "/home/spark7/lane-k3-t1/kimi-k3-local"
WORKERS = 10
os.makedirs(DST, exist_ok=True)

index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
weight_map = index["weight_map"]

jobs = {}
entries = {}
for name, fname in sorted(weight_map.items()):
    if ".block_sparse_moe.experts." in name or name.startswith(
            ("vision_tower", "mm_projector")):
        entries[name] = os.path.join(SRC, fname)
        continue
    jobs.setdefault(fname, []).append(name)

print("shards to stage:", len(jobs), flush=True)


def stage_shard(shard):
    out_name = "local-" + shard.replace("/", "_")
    out_path = os.path.join(DST, out_name)
    src_fh = open(os.path.join(SRC, shard), "rb")
    n = struct.unpack("<Q", src_fh.read(8))[0]
    header = json.loads(src_fh.read(n))
    base = 8 + n
    out_header = {}
    offset = 0
    spans = []
    for name in jobs[shard]:
        entry = header[name]
        start, end = entry["data_offsets"]
        length = end - start
        out_header[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                            "data_offsets": [offset, offset + length]}
        spans.append((name, start, length))
        offset += length
    header_bytes = json.dumps(out_header, separators=(",", ":")).encode()
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    with open(out_path, "wb") as out_fh:
        out_fh.write(struct.pack("<Q", len(header_bytes)))
        out_fh.write(header_bytes)
        for name, start, length in spans:
            src_fh.seek(base + start)
            remaining = length
            while remaining > 0:
                chunk = src_fh.read(min(1 << 24, remaining))
                if not chunk:
                    raise RuntimeError("short read " + name)
                out_fh.write(chunk)
                remaining -= len(chunk)
    entries.update({name: out_name for name, _, _ in spans})
    src_fh.close()
    return out_name, offset


with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for out_name, offset in pool.map(stage_shard, sorted(jobs)):
        print("staged", out_name, round(offset / 1e9, 2), "GB", flush=True)

json.dump({"weight_map": entries}, open(
    os.path.join(DST, "model.safetensors.index.json"), "w"))
import shutil
shutil.copyfile(os.path.join(SRC, "config.json"),
                os.path.join(DST, "config.json"))
print("entries:", len(entries), flush=True)
print("DONE", flush=True)
