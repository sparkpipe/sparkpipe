#!/usr/bin/env bash
# laguna_multidev_experts_manifest.sh PACK_PATH — generate (idempotently)
# the v2 routed-expert manifest PACK_PATH.experts for one laguna .lgsp
# rank pack, from the COMMITTED sources of this checkout. Laguna port of
# the k3 lane-3 generator (tools/k3_multidev_experts_manifest.sh): any
# host, any pack, private cell directory. Existing .experts files are
# left untouched. The shared weightd rejects a lazy attach whose pack
# lacks this sidecar (loading must fail closed - see
# docs/PARALLEL_DRIVER_DEBUG.md).
#
# Source of truth for the per-expert spans: the .lgsp fixed-layout
# directory itself (modules/laguna_resident_decode_stage/source/
# spark_laguna_stagepack_format.h - 264-byte header, 64-byte entries at
# directory_offset). bf16 expert tensors are uniform per-expert
# interleaves: entry payload_bytes / group_count bytes per expert,
# expert e at payload_offset + e * per_expert_bytes (the packer emits
# [gate|up] per expert for w1 and the column-sliced down projection per
# expert for w2 - tools/laguna_stagepack.py add_experts). The generator
# fails closed on any geometry surprise instead of guessing:
#   - magic/format/header/entry-size mismatch with the format header
#   - expert_weight_codec != bf16 (the placed arm; a future fp8/nvfp4/
#     mixed pack needs its own span arithmetic review - #1076)
#   - payload_bytes not an exact multiple of group_count, or disagreeing
#     with rows*columns*2 (bf16)
#   - nonzero scale planes on expert tensors
# Records carry the GLOBAL layer index (stage packs store global layers:
# the module resolves wave layers as first_layer_index + local and looks
# the manifest up by that number) and the LAGUNA range-kind convention
# kind = tensor_kind * 2 + plane (28 = EXPERT_GATE_UP payload, 30 =
# EXPERT_DOWN payload; the scale kinds 29/31 belong to quantized arms)
# - SparkLagunaManifestCheck walks exactly these kinds. The k3 lane's
# 0/1 kinds are that family's own convention, not a shared one.
set -euo pipefail
PACK="${1:?usage: laguna_multidev_experts_manifest.sh PACK_PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CELL="${LAGUNA_MANIFEST_CELL:-${TMPDIR:-/tmp}/laguna-manifest-cell.$$}"
[ -f "$PACK" ] || { echo "pack not found: $PACK" >&2; exit 1; }
# A present sidecar is accepted only when it is a v2 routed-expert
# manifest with records (the same staleness rule as the k3 generator).
if [ -f "$PACK.experts" ]; then
  if python3 - "$PACK.experts" <<'PYVER'
import struct, sys
with open(sys.argv[1], "rb") as handle:
    head = handle.read(16)
    record = handle.read(48)
magic, version, count, _ = struct.unpack("<IIII", head)
ok = magic == 0x58504557 and version == 2 and count > 0 and len(record) == 48
if ok:
    # the laguna range-kind convention (tensor_kind*2 + payload plane);
    # sidecars carrying foreign kinds (the k3-style 0/1 first attempts)
    # are stale for this module and regenerate below
    kind = struct.unpack_from("<4I2Q", record)[2]
    ok = kind in (28, 30)
raise SystemExit(0 if ok else 1)
PYVER
  then
    echo "exists: $PACK.experts"
    exit 0
  fi
  echo "stale sidecar (not v2/empty/wrong kind convention): regenerating $PACK.experts" >&2
  rm -f "$PACK.experts"
fi
mkdir -p "$CELL"
trap 'rm -rf "$CELL"' EXIT
# shellcheck disable=SC2086
cc -std=c11 -O2 -shared -fPIC -I"$ROOT" -I"$ROOT/include" \
  "$ROOT/src/spark_ck128.c" "$ROOT/src/spark_status.c" \
  "$ROOT/runtime/spark_weightd_manifest.c" \
  -o "$CELL/liblagunamanifest.so"
PACK="$PACK" CELL="$CELL" PYTHONDONTWRITEBYTECODE=1 python3 - <<'PYEOF'
import ctypes
import mmap
import os
import struct
from pathlib import Path

pack_path = Path(os.environ["PACK"])
lib_path = os.environ["CELL"] + "/liblagunamanifest.so"
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

# .lgsp fixed layout (spark_laguna_stagepack_format.h).
MAGIC = 0x334C4147
FORMAT_VERSION = 1
HEADER_BYTES = 264
ENTRY_BYTES = 64
K_EXPERT_GATE_UP, K_EXPERT_DOWN = 14, 15
PAYLOAD_BF16, CODEC_BF16 = 1, 1
SIDE_KIND = {K_EXPERT_GATE_UP: 28, K_EXPERT_DOWN: 30}


def fail(message):
    raise SystemExit(f"laguna experts manifest: FAIL: {message}")


def ck128(data):
    # ctypes portability (the sparkc python 3.12 finding from the k3
    # generator): explicit ctypes objects, no argtypes on the digests.
    context = (ctypes.c_uint8 * 64)()
    digest = (ctypes.c_uint8 * 16)()
    lib.SparkCk128Initialize(ctypes.byref(context))
    lib.SparkCk128Update(ctypes.byref(context),
                         ctypes.c_char_p(data), ctypes.c_size_t(len(data)))
    lib.SparkCk128Finalize(ctypes.byref(context), digest)
    return bytes(digest)


with open(pack_path, "rb") as handle:
    raw = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    if len(raw) < HEADER_BYTES:
        fail("pack shorter than the fixed header")
    header = struct.unpack_from("<20I2Q65s32s32s32s", raw, 0)
    (magic, format_version, header_bytes, entry_bytes, _codec_abi, _flags,
     tensor_count, _stage_count, _stage_index, _first_layer, _layer_count,
     _total_layers, _hidden, _vocab, _routed_experts, _linear_codec,
     expert_codec, _kv_codec, _r0, _r1, directory_offset,
     file_bytes) = header[:22]
    if magic != MAGIC or format_version != FORMAT_VERSION:
        fail(f"not a laguna stage pack (magic {magic:#x} version {format_version})")
    if header_bytes != HEADER_BYTES or entry_bytes != ENTRY_BYTES:
        fail(f"unexpected layout: header {header_bytes} entry {entry_bytes}")
    if file_bytes != pack_bytes:
        fail(f"header file_bytes {file_bytes} != size {pack_bytes}")
    if expert_codec != CODEC_BF16:
        fail(f"expert codec {expert_codec} is not the placed bf16 arm "
             "(#1076 mixed/fp8 packs need their own span arithmetic)")
    if directory_offset + tensor_count * ENTRY_BYTES > pack_bytes:
        fail("directory escapes the pack")

    records = []
    expert_bytes_total = 0
    seen = set()
    with open(pack_path, "rb") as spans:
        for index in range(tensor_count):
            entry = struct.unpack_from(
                "<8I4Q", raw, directory_offset + index * ENTRY_BYTES)
            (tensor_kind, layer_index, payload_type, weight_codec,
             scale_encoding, group_count, rows, columns, payload_offset,
             payload_bytes, _scale_offset, scale_bytes) = entry
            if tensor_kind not in SIDE_KIND:
                continue
            if payload_type != PAYLOAD_BF16 or weight_codec != CODEC_BF16:
                fail(f"layer {layer_index} expert tensor not bf16 payload")
            if scale_bytes != 0 or scale_encoding != 0:
                fail(f"layer {layer_index} expert tensor carries a scale plane")
            if group_count < 2 or payload_bytes % group_count != 0:
                fail(f"layer {layer_index} kind {tensor_kind}: payload "
                     f"{payload_bytes} not a uniform {group_count}-expert interleave")
            per_expert = payload_bytes // group_count
            if rows * columns * 2 != per_expert:
                fail(f"layer {layer_index} kind {tensor_kind}: per-expert "
                     f"{per_expert} != rows*cols*2 = {rows * columns * 2}")
            key = (layer_index, SIDE_KIND[tensor_kind])
            if key in seen:
                fail(f"duplicate expert tensor {key}")
            seen.add(key)
            if payload_offset + payload_bytes > pack_bytes:
                fail(f"layer {layer_index} kind {tensor_kind}: payload escapes the pack")
            for expert in range(group_count):
                base = payload_offset + expert * per_expert
                spans.seek(base)
                data = spans.read(per_expert)
                records.append(struct.pack(
                    "<4I2Q16s", layer_index, expert, SIDE_KIND[tensor_kind], 0,
                    base, per_expert, ck128(data)))
            expert_bytes_total += payload_bytes
    raw.close()

count = len(records)
if count == 0:
    fail("no routed-expert tensors in the directory")
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
print(f"{out_path.name}: {count} ranges, expert_bytes {expert_bytes_total}, "
      f"spine_bytes {pack_bytes - expert_bytes_total}")
PYEOF
