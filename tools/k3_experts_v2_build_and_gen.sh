#!/usr/bin/env bash
# k3_experts_v2_build_and_gen.sh — per-node: build the ck128+manifest
# shared object from the COMMITTED sources in the node checkout, then
# generate the v2 .experts manifests for the node's k3 TP16 rank packs
# (ctypes calls the shared object for ck128 and the loader self-verify).
# Idempotent: existing .experts files are skipped.
set -euo pipefail
SRC=/home/spark7/k3main-src
CELL=/home/spark7/k3cell
cd "$SRC"
mkdir -p "$CELL"
cc -std=c11 -O2 -shared -fPIC -I. -Iinclude \
  src/spark_ck128.c src/spark_status.c runtime/spark_weightd_manifest.c \
  -o "$CELL/libk3manifest.so"
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PYEOF'
import ctypes
import json
import mmap
import os
import socket
import struct
from pathlib import Path

host = socket.gethostname()
rank_root = "/home/spark7/sparkdata"
if host != "spark7":
    rank_root = "/home/" + host + "/sp" + "arkdata"
packs = sorted(Path(rank_root + "/k3.mxfp4.tp16/packs").glob(
    "k3.stage0.rank*.pack"))
lib = ctypes.CDLL("/home/spark7/k3cell/libk3manifest.so")
lib.SparkCk128Initialize.argtypes = [ctypes.c_void_p]
lib.SparkCk128Update.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                 ctypes.c_size_t]
lib.SparkCk128Finalize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

class Manifest(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in
                ("ranges", "groups", "spine")] + \
               [(n, ctypes.c_uint64) for n in
                ("spine_bytes", "spine_allocation_bytes")] + \
               [(n, ctypes.c_uint32) for n in
                ("range_count", "group_count", "spine_count")]

lib.SparkWeightdManifestLoad.argtypes = [ctypes.c_char_p, ctypes.c_uint64,
                                         ctypes.POINTER(Manifest)]
lib.SparkWeightdManifestLoad.restype = ctypes.c_int
lib.SparkWeightdManifestDestroy.argtypes = [ctypes.POINTER(Manifest)]

def ck128(lib, data):
    context = bytearray(64)
    digest = (ctypes.c_uint8 * 16)()
    lib.SparkCk128Initialize(context)
    lib.SparkCk128Update(context, data, len(data))
    lib.SparkCk128Finalize(context, digest)
    return bytes(digest)

failures = 0
for pack_path in packs:
    out_path = pack_path.with_suffix(pack_path.suffix + ".experts")
    partial = out_path.with_name(out_path.name + ".partial")
    pack_bytes = pack_path.stat().st_size
    with open(pack_path, "rb") as handle:
        raw = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, length = struct.unpack_from("<IIQ", raw, 0)
        assert magic == 0x4B33504B and version == 2
        manifest = json.loads(raw[16:16 + length])
        payload_base = 16 + length
        payload_base += -payload_base % 128
        records = []
        for name, entry in manifest["tensors"].items():
            if not (name.endswith("expert_w1_weight") or
                    name.endswith("expert_w2_weight")):
                continue
            geometry = entry["interleave"]
            expert_bytes = geometry["expert_bytes"]
            experts = geometry["experts"]
            layer = int(name.split(".")[2])
            kind = 0 if name.endswith("w1_weight") else 1
            tensor_base = payload_base + entry["offset"]
            with open(pack_path, "rb") as spans:
                for expert in range(experts):
                    spans.seek(tensor_base + expert * expert_bytes)
                    data = spans.read(expert_bytes)
                    records.append(struct.pack(
                        "<4I2Q16s", layer, expert, kind, 0,
                        tensor_base + expert * expert_bytes, expert_bytes,
                        ck128(lib, data)))
        raw.close()
    count = len(records)
    with open(partial, "wb") as out:
        out.write(struct.pack("<IIII", 0x58504557, 2, count, 0))
        for record in records:
            out.write(record)
        out.flush()
        os.fsync(out.fileno())
    verdict = Manifest()
    if lib.SparkWeightdManifestLoad(str(partial).encode(), pack_bytes,
                                    ctypes.byref(verdict)) != 0:
        print("FAIL " + pack_path.name + ": loader self-verify rejected")
        os.unlink(partial)
        failures += 1
        continue
    lib.SparkWeightdManifestDestroy(ctypes.byref(verdict))
    os.replace(partial, out_path)
    print(f"{out_path.name}: {count} ranges")
if failures:
    print(f"FAILURES: {failures}")
sys.exit(1 if failures else 0)
PYEOF
