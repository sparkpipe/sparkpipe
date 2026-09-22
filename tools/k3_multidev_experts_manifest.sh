#!/usr/bin/env bash
# k3_multidev_experts_manifest.sh PACK_PATH — generate (idempotently) the
# v2 routed-expert manifest PACK_PATH.experts for one k3 rank pack, from
# the COMMITTED sources of this checkout. Port of
# tools/k3_experts_v2_build_and_gen.sh (which is pinned to spark7 and the
# TP16 pack set) to the shared-socket lane workflow: any host, any pack,
# private cell directory. Existing .experts files are left untouched.
# The shared weightd rejects a lazy attach whose pack lacks this sidecar
# (loading must fail closed - see docs/PARALLEL_DRIVER_DEBUG.md).
set -euo pipefail
PACK="${1:?usage: k3_multidev_experts_manifest.sh PACK_PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CELL="${K3_MANIFEST_CELL:-${TMPDIR:-/tmp}/k3-manifest-cell.$$}"
[ -f "$PACK" ] || { echo "pack not found: $PACK" >&2; exit 1; }
# A present sidecar is accepted only when it is a v2 routed-expert
# manifest with records; the stage-3 packs carried stale v1 files
# (1,463 group records) that the lazy attach cannot use - regenerate
# those instead of trusting file existence.
if [ -f "$PACK.experts" ]; then
  if python3 - "$PACK.experts" <<'PYVER'
import struct, sys
with open(sys.argv[1], "rb") as handle:
    head = handle.read(16)
magic, version, count, _ = struct.unpack("<IIII", head)
raise SystemExit(0 if (magic == 0x58504557 and version == 2 and count > 0) else 1)
PYVER
  then
    echo "exists: $PACK.experts"
    exit 0
  fi
  echo "stale sidecar (not v2/empty): regenerating $PACK.experts" >&2
  rm -f "$PACK.experts"
fi
mkdir -p "$CELL"
trap 'rm -rf "$CELL"' EXIT
# shellcheck disable=SC2086
cc -std=c11 -O2 -shared -fPIC -I"$ROOT" -I"$ROOT/include" \
  "$ROOT/src/spark_ck128.c" "$ROOT/src/spark_status.c" \
  "$ROOT/runtime/spark_weightd_manifest.c" \
  -o "$CELL/libk3manifest.so"
PACK="$PACK" CELL="$CELL" PYTHONDONTWRITEBYTECODE=1 python3 - <<'PYEOF'
import ctypes
import json
import mmap
import os
import struct
from pathlib import Path

pack_path = Path(os.environ["PACK"])
lib_path = os.environ["CELL"] + "/libk3manifest.so"
pack_bytes = pack_path.stat().st_size
out_path = pack_path.with_suffix(pack_path.suffix + ".experts")
partial = out_path.with_name(out_path.name + ".partial")
lib = ctypes.CDLL(lib_path)

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


def ck128(data):
    # ctypes portability: strict c_void_p argtypes reject bytearray and
    # array contexts on some fleet Pythons (seen on sparkc, python 3.12);
    # pass explicit ctypes objects with no argtypes instead - the digests
    # match the spark7 v2 generator byte for byte.
    context = (ctypes.c_uint8 * 64)()
    digest = (ctypes.c_uint8 * 16)()
    lib.SparkCk128Initialize(ctypes.byref(context))
    lib.SparkCk128Update(ctypes.byref(context),
                         ctypes.c_char_p(data), ctypes.c_size_t(len(data)))
    lib.SparkCk128Finalize(ctypes.byref(context), digest)
    return bytes(digest)


with open(pack_path, "rb") as handle:
    raw = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    magic, version, length = struct.unpack_from("<IIQ", raw, 0)
    assert magic == 0x4B33504B and version == 2, (hex(magic), version)
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
                    ck128(data)))
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
    raise SystemExit(1)
lib.SparkWeightdManifestDestroy(ctypes.byref(verdict))
os.replace(partial, out_path)
print(f"{out_path.name}: {count} ranges")
PYEOF
