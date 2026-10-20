#!/usr/bin/env python3
"""K3 rank pack format 3: census, convert and verify.

check <v2.pack>    reads a format 2 cell-split rank pack with O_DIRECT and
                   reports, per expert tensor, the E8M0 scale codes of its
                   non-zero 32-element blocks and of its all-zero blocks, and
                   whether one 4-bit window per tensor holds every code.
convert <v2.pack> <v3.pack> <libk3manifest.so>
                   writes the format 3 pack: this rank's router and MLA down
                   rows, and expert tensors in paired 33-row cells whose code
                   row holds both k-tiles' scales as 4-bit offsets from one
                   base per tensor. Every other tensor is copied. The output
                   is re-read from disk with O_DIRECT and compared with the
                   source tensor by tensor before it, its .sha256 and its
                   .experts sidecar are renamed into place.
verify <v2.pack> <v3.pack> <libk3manifest.so>
                   repeats that comparison on an existing pair."""
import copy
import ctypes
import hashlib
import json
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x4B33504B
SOURCE_VERSION = 2
TARGET_VERSION = 3
PAIRED_KIND = "mxfp4_ws_paired_v1"
SOURCE_KIND = "mxfp4_ws_interleaved_v1"
PAIR_ROWS = 33
SYNC_BYTES = 1 << 30
ALIGN = 128
CELL_PAYLOAD_ROWS = 16
SOURCE_CELL_ROWS = 17
ROW_BYTES = 16
TILE_K = 32
SCALE_SPAN = 16
DIRECT = 1 << 20
CHUNK = 64 << 20


class RepackFailure(RuntimeError):
    pass


class DirectReader:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        self.size = os.fstat(self.fd).st_size
        self.buffer = mmap.mmap(-1, CHUNK + 2 * DIRECT)

    def read(self, offset, length):
        start = offset - offset % DIRECT
        end = offset + length
        span = end - start
        span += (-span) % DIRECT
        if span > len(self.buffer):
            raise RepackFailure(f"read of {length} bytes at {offset} exceeds the {CHUNK} byte window")
        view = memoryview(self.buffer)[:span]
        got = os.preadv(self.fd, [view], start)
        if got < min(span, self.size - start):
            raise RepackFailure(f"short read at {start}: {got} of {span} bytes")
        return bytes(view[offset - start:offset - start + length])

    def close(self):
        self.buffer.close()
        os.close(self.fd)


def load_manifest(reader, version_wanted=SOURCE_VERSION):
    magic, version, length = struct.unpack("<IIQ", reader.read(0, 16))
    if magic != MAGIC:
        raise RepackFailure("not a K3 pack")
    if version != version_wanted:
        raise RepackFailure(f"K3 pack format {version}; expected format {version_wanted}")
    manifest = json.loads(reader.read(16, length))
    base = 16 + length
    base += (-base) % ALIGN
    return manifest, base, length


def expert_tensors(manifest):
    out = []
    for name, entry in manifest["tensors"].items():
        if entry.get("kind") != "mxfp4_ws_interleaved_v1":
            continue
        geom = entry.get("interleave") or {}
        if geom.get("tile_k") != TILE_K or geom.get("row_bytes") != ROW_BYTES or geom.get("cell_rows") != SOURCE_CELL_ROWS:
            raise RepackFailure(f"{name}: interleave {geom.get('tile_k')}/{geom.get('row_bytes')}/{geom.get('cell_rows')} is not {TILE_K}/{ROW_BYTES}/{SOURCE_CELL_ROWS}")
        if geom["k_tiles"] % 2 != 0:
            raise RepackFailure(f"{name}: {geom['k_tiles']} k-tiles do not pair")
        out.append((name, entry, geom))
    if not out:
        raise RepackFailure("the pack has no interleaved expert tensors")
    return sorted(out, key=lambda item: item[1]["offset"])


def census_tensor(reader, base, entry, geom):
    experts = entry["shape"][0]
    expert_bytes = geom["expert_bytes"]
    if expert_bytes * experts != entry["bytes"]:
        raise RepackFailure(f"expert bytes {expert_bytes} x {experts} != tensor bytes {entry['bytes']}")
    per_chunk = max(1, CHUNK // expert_bytes)
    live = np.zeros(256, dtype=np.int64)
    zero = np.zeros(256, dtype=np.int64)
    for first in range(0, experts, per_chunk):
        count = min(per_chunk, experts - first)
        raw = reader.read(base + entry["offset"] + first * expert_bytes, count * expert_bytes)
        grid = np.frombuffer(raw, dtype=np.uint8).reshape(count, geom["k_tiles"], geom["cells"], SOURCE_CELL_ROWS, ROW_BYTES)
        scales = grid[:, :, :, CELL_PAYLOAD_ROWS, :]
        empty = np.all((grid[:, :, :, :CELL_PAYLOAD_ROWS, :] & 0x77) == 0, axis=-1)
        live += np.bincount(scales[~empty], minlength=256)
        zero += np.bincount(scales[empty], minlength=256)
    return live, zero


def window(live):
    codes = np.flatnonzero(live)
    if codes.size == 0:
        return None
    return int(codes[0]), int(codes[-1])


def check(path):
    reader = DirectReader(path)
    try:
        manifest, base, _ = load_manifest(reader)
        config = manifest["config"]
        if config.get("expert_split") != "cells":
            raise RepackFailure(f"expert_split {config.get('expert_split')!r} is not cells")
        totals = {}
        ok = True
        for name, entry, geom in expert_tensors(manifest):
            live, zero = census_tensor(reader, base, entry, geom)
            span = window(live)
            fits = span is not None and span[0] >= 1 and span[1] <= 254 and span[1] - span[0] < SCALE_SPAN
            ok &= fits
            kind = "w1" if name.endswith("w1_weight") else "w2"
            totals.setdefault(kind, np.zeros(256, dtype=np.int64))
            totals[kind] += live
            print(json.dumps({"tensor": name, "low": span[0] if span else None, "high": span[1] if span else None,
                              "distinct": int(np.count_nonzero(live)), "zero_blocks": int(zero.sum()),
                              "zero_block_codes": [int(c) for c in np.flatnonzero(zero)], "fits": bool(fits)}), flush=True)
        summary = {kind: {"low": window(h)[0], "high": window(h)[1], "distinct": int(np.count_nonzero(h)),
                          "top": int(np.argmax(h)), "top_share": round(float(h.max() / h.sum()), 4)} for kind, h in totals.items()}
        print(json.dumps({"verdict": "convertible" if ok else "REFUSED", "rank": config.get("tp_rank"),
                          "degree": config.get("tp_degree"), "summary": summary}), flush=True)
        return 0 if ok else 3
    finally:
        reader.close()


ROW_SPLIT = {"router_weight": "expert_rows", "mla_q_down_weight": "output_rows", "mla_kv_a_weight": "output_rows"}


def split_suffix(name):
    for suffix, shard_class in ROW_SPLIT.items():
        if name.endswith("." + suffix):
            return suffix, shard_class
    return None, None


def source_contract(manifest):
    config = manifest["config"]
    if config.get("expert_split") != "cells":
        raise RepackFailure(f"expert_split {config.get('expert_split')!r} is not cells")
    degree, rank = int(config.get("tp_degree", 0)), int(config.get("tp_rank", -1))
    if degree < 2 or not 0 <= rank < degree:
        raise RepackFailure(f"tp_rank {rank} of tp_degree {degree} is not a split rank pack")
    hidden = int(config["hidden"])
    for name, entry in manifest["tensors"].items():
        suffix, _ = split_suffix(name)
        if suffix is None:
            continue
        if entry.get("kind") != "bf16" or len(entry["shape"]) != 2 or entry["shape"][1] != hidden or "rows" in entry:
            raise RepackFailure(f"{name}: expected a whole bf16 [rows, {hidden}] tensor, found {entry.get('kind')} {entry['shape']}")
        if entry["shape"][0] % degree != 0:
            raise RepackFailure(f"{name}: {entry['shape'][0]} rows do not split {degree} ways")
    return degree, rank, hidden


def target_manifest(manifest, bases, source_sha256):
    degree, rank, hidden = source_contract(manifest)
    out = copy.deepcopy(manifest)
    out["format"] = {key: value for key, value in out["format"].items() if key != "mxfp4_interleave"}
    out["format"]["version"] = TARGET_VERSION
    out["format"]["expert_codec"] = PAIRED_KIND
    out["config"]["source_format"] = SOURCE_VERSION
    out["config"]["source_pack_sha256"] = source_sha256
    out["config"]["weights_identity"] = source_sha256
    offset = 0
    for name in sorted(out["tensors"], key=lambda n: manifest["tensors"][n]["offset"]):
        entry = out["tensors"][name]
        suffix, shard_class = split_suffix(name)
        if suffix is not None:
            total = entry["shape"][0]
            count = total // degree
            entry["shape"] = [count, hidden]
            entry["bytes"] = count * hidden * 2
            entry["shard_class"] = shard_class
            entry["rows"] = {"first": rank * count, "count": count, "total": total}
        elif entry.get("kind") == SOURCE_KIND:
            geom = entry["interleave"]
            if geom["k_tiles"] % 2 != 0 or name not in bases:
                raise RepackFailure(f"{name}: cannot pair {geom['k_tiles']} k-tiles")
            pairs = geom["k_tiles"] // 2
            expert_bytes = pairs * geom["cells"] * PAIR_ROWS * ROW_BYTES
            entry["kind"] = PAIRED_KIND
            geom.update({"kind": PAIRED_KIND, "cell_rows": PAIR_ROWS, "tiles_per_cell": 2, "k_pairs": pairs,
                         "scale_bits": 4, "scale_base": bases[name], "scale_span": SCALE_SPAN,
                         "rows_per_expert": pairs * geom["cells"] * PAIR_ROWS, "expert_bytes": expert_bytes,
                         "tensor_bytes": expert_bytes * geom["experts"]})
            geom.pop("scale_bytes_per_neuron_tile", None)
            entry["bytes"] = geom["tensor_bytes"]
        offset += (-offset) % ALIGN
        entry["offset"] = offset
        offset += entry["bytes"]
    return out, offset


def encode_manifest(manifest, reserve):
    text = json.dumps(manifest, separators=(",", ":")).encode()
    if len(text) > reserve:
        raise RepackFailure(f"the format 3 manifest needs {len(text)} bytes, the header reserves {reserve}")
    return text + b" " * (reserve - len(text))


def pair_experts(raw, count, geom, base):
    pairs, cells = geom["k_tiles"] // 2, geom["cells"]
    grid = np.frombuffer(raw, dtype=np.uint8).reshape(count, pairs, 2, cells, SOURCE_CELL_ROWS, ROW_BYTES)
    out = np.empty((count, pairs, cells, PAIR_ROWS, ROW_BYTES), dtype=np.uint8)
    out[:, :, :, :CELL_PAYLOAD_ROWS, :] = grid[:, :, 0, :, :CELL_PAYLOAD_ROWS, :]
    out[:, :, :, CELL_PAYLOAD_ROWS:2 * CELL_PAYLOAD_ROWS, :] = grid[:, :, 1, :, :CELL_PAYLOAD_ROWS, :]
    low = grid[:, :, 0, :, CELL_PAYLOAD_ROWS, :].astype(np.int16) - base
    high = grid[:, :, 1, :, CELL_PAYLOAD_ROWS, :].astype(np.int16) - base
    if low.min() < 0 or high.min() < 0 or low.max() >= SCALE_SPAN or high.max() >= SCALE_SPAN:
        raise RepackFailure(f"a scale code falls outside base {base} + 0..{SCALE_SPAN - 1}")
    out[:, :, :, 2 * CELL_PAYLOAD_ROWS, :] = (low | (high << 4)).astype(np.uint8)
    return out.tobytes()


def unpair_experts(raw, count, geom):
    pairs, cells, base = geom["k_pairs"], geom["cells"], geom["scale_base"]
    grid = np.frombuffer(raw, dtype=np.uint8).reshape(count, pairs, cells, PAIR_ROWS, ROW_BYTES)
    out = np.empty((count, pairs, 2, cells, SOURCE_CELL_ROWS, ROW_BYTES), dtype=np.uint8)
    out[:, :, 0, :, :CELL_PAYLOAD_ROWS, :] = grid[:, :, :, :CELL_PAYLOAD_ROWS, :]
    out[:, :, 1, :, :CELL_PAYLOAD_ROWS, :] = grid[:, :, :, CELL_PAYLOAD_ROWS:2 * CELL_PAYLOAD_ROWS, :]
    codes = grid[:, :, :, 2 * CELL_PAYLOAD_ROWS, :].astype(np.uint16)
    out[:, :, 0, :, CELL_PAYLOAD_ROWS, :] = ((codes & 15) + base).astype(np.uint8)
    out[:, :, 1, :, CELL_PAYLOAD_ROWS, :] = ((codes >> 4) + base).astype(np.uint8)
    return out.tobytes()


def scale_bases(reader, base, manifest):
    bases = {}
    for name, entry, geom in expert_tensors(manifest):
        live, zero = census_tensor(reader, base, entry, geom)
        span = window(live + zero)
        if span is None or span[0] < 1 or span[1] > 254 or span[1] - span[0] >= SCALE_SPAN:
            raise RepackFailure(f"{name}: scale codes {span} do not fit one {SCALE_SPAN}-code window")
        bases[name] = span[0]
    return bases


def file_sha256(path):
    reader = DirectReader(path)
    try:
        digest = hashlib.sha256()
        for offset in range(0, reader.size, CHUNK):
            digest.update(reader.read(offset, min(CHUNK, reader.size - offset)))
        return digest.hexdigest()
    finally:
        reader.close()


class Writer:
    def __init__(self, path):
        self.handle = open(path, "xb", buffering=0)
        self.position = 0
        self.synced = 0

    def write(self, data):
        view = memoryview(data)
        while view:
            done = self.handle.write(view)
            view = view[done:]
            self.position += done
        if self.position - self.synced >= SYNC_BYTES:
            self.flush()

    def pad_to(self, offset):
        if offset < self.position:
            raise RepackFailure(f"layout overlaps at {offset} (written {self.position})")
        self.write(bytes(offset - self.position))

    def flush(self):
        os.fdatasync(self.handle.fileno())
        os.posix_fadvise(self.handle.fileno(), self.synced, self.position - self.synced, os.POSIX_FADV_DONTNEED)
        self.synced = self.position

    def close(self):
        self.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def tensor_chunks(entry, geom):
    if geom is None:
        for first in range(0, entry["bytes"], CHUNK):
            yield first, min(CHUNK, entry["bytes"] - first), None
        return
    per_chunk = max(1, CHUNK // geom["expert_bytes"])
    for first in range(0, geom["experts"], per_chunk):
        count = min(per_chunk, geom["experts"] - first)
        yield first, count, count


def convert(source_path, target_path, library):
    target = Path(target_path)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise RepackFailure(f"{target} or its .partial already exists")
    source_sha256 = file_sha256(source_path)
    reader = DirectReader(source_path)
    try:
        manifest, base, reserve = load_manifest(reader)
        bases = scale_bases(reader, base, manifest)
        out_manifest, payload_bytes = target_manifest(manifest, bases, source_sha256)
        header = struct.pack("<IIQ", MAGIC, TARGET_VERSION, reserve) + encode_manifest(out_manifest, reserve)
        writer = Writer(partial)
        writer.write(header)
        writer.pad_to(base)
        for name in sorted(out_manifest["tensors"], key=lambda n: out_manifest["tensors"][n]["offset"]):
            entry, source_entry = out_manifest["tensors"][name], manifest["tensors"][name]
            writer.pad_to(base + entry["offset"])
            if "rows" in entry:
                rows = entry["rows"]
                row_bytes = source_entry["bytes"] // source_entry["shape"][0]
                writer.write(reader.read(base + source_entry["offset"] + rows["first"] * row_bytes, rows["count"] * row_bytes))
            elif entry.get("kind") == PAIRED_KIND:
                geom = source_entry["interleave"]
                for first, count, _ in tensor_chunks(source_entry, geom):
                    raw = reader.read(base + source_entry["offset"] + first * geom["expert_bytes"], count * geom["expert_bytes"])
                    writer.write(pair_experts(raw, count, geom, entry["interleave"]["scale_base"]))
            else:
                for first, length, _ in tensor_chunks(source_entry, None):
                    writer.write(reader.read(base + source_entry["offset"] + first, length))
            if writer.position != base + entry["offset"] + entry["bytes"]:
                raise RepackFailure(f"{name}: wrote to {writer.position}, expected {base + entry['offset'] + entry['bytes']}")
        writer.pad_to(base + payload_bytes)
        writer.close()
    finally:
        reader.close()
    try:
        verify(source_path, str(partial), library, target.name)
    except RepackFailure:
        partial.unlink()
        raise
    os.replace(str(partial) + ".experts.partial", str(target) + ".experts")
    os.replace(str(partial) + ".sha256.partial", str(target) + ".sha256")
    os.replace(partial, target)
    print(json.dumps({"converted": str(target), "bytes": os.path.getsize(target), "source_sha256": source_sha256}), flush=True)


class WeightdManifest(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("ranges", "groups", "spine")] + \
               [(n, ctypes.c_uint64) for n in ("spine_bytes", "spine_allocation_bytes")] + \
               [(n, ctypes.c_uint32) for n in ("range_count", "group_count", "spine_count")]


def manifest_library(path):
    lib = ctypes.CDLL(str(path))
    lib.SparkCk128Initialize.argtypes = [ctypes.c_void_p]
    lib.SparkCk128Update.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.SparkCk128Finalize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SparkWeightdManifestLoad.restype = ctypes.c_int
    lib.SparkWeightdManifestLoad.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(WeightdManifest)]
    lib.SparkWeightdManifestDestroy.argtypes = [ctypes.POINTER(WeightdManifest)]
    return lib


def ck128(lib, data):
    state = ctypes.create_string_buffer(256)
    digest = ctypes.create_string_buffer(16)
    lib.SparkCk128Initialize(state)
    lib.SparkCk128Update(state, data, len(data))
    lib.SparkCk128Finalize(state, digest)
    return digest.raw


def verify(source_path, target_path, library, final_name=None):
    lib = manifest_library(library)
    source, target = DirectReader(source_path), DirectReader(target_path)
    records = []
    try:
        manifest, base, _ = load_manifest(source)
        out_manifest, out_base, _ = load_manifest(target, TARGET_VERSION)
        if out_base != base or set(out_manifest["tensors"]) != set(manifest["tensors"]):
            raise RepackFailure("the format 3 manifest does not carry the source tensors at the source payload base")
        expected, payload_bytes = target_manifest(manifest, {n: e["interleave"]["scale_base"] for n, e in out_manifest["tensors"].items() if e.get("kind") == PAIRED_KIND},
                                                  out_manifest["config"]["source_pack_sha256"])
        if expected != out_manifest:
            raise RepackFailure("the format 3 manifest differs from the one the source implies")
        if target.size != base + payload_bytes:
            raise RepackFailure(f"size {target.size} != {base + payload_bytes}")
        position = base
        for name in sorted(out_manifest["tensors"], key=lambda n: out_manifest["tensors"][n]["offset"]):
            entry, source_entry = out_manifest["tensors"][name], manifest["tensors"][name]
            start = base + entry["offset"]
            if start > position and any(target.read(position, start - position)):
                raise RepackFailure(f"non-zero padding before {name}")
            if "rows" in entry:
                rows = entry["rows"]
                row_bytes = source_entry["bytes"] // source_entry["shape"][0]
                for first in range(0, entry["bytes"], CHUNK):
                    length = min(CHUNK, entry["bytes"] - first)
                    if target.read(start + first, length) != source.read(base + source_entry["offset"] + rows["first"] * row_bytes + first, length):
                        raise RepackFailure(f"{name}: rows differ from the source")
            elif entry.get("kind") == PAIRED_KIND:
                geom, source_geom = entry["interleave"], source_entry["interleave"]
                layer = int(name.split(".")[2])
                kind = 0 if name.endswith("w1_weight") else 1
                for first, count, _ in tensor_chunks(source_entry, source_geom):
                    raw = target.read(start + first * geom["expert_bytes"], count * geom["expert_bytes"])
                    if unpair_experts(raw, count, geom) != source.read(base + source_entry["offset"] + first * source_geom["expert_bytes"], count * source_geom["expert_bytes"]):
                        raise RepackFailure(f"{name}: experts {first}..{first + count - 1} do not decode to the source")
                    for index in range(count):
                        piece = raw[index * geom["expert_bytes"]:(index + 1) * geom["expert_bytes"]]
                        records.append(struct.pack("<4I2Q16s", layer, first + index, kind, 0, start + (first + index) * geom["expert_bytes"], geom["expert_bytes"], ck128(lib, piece)))
            else:
                for first in range(0, entry["bytes"], CHUNK):
                    length = min(CHUNK, entry["bytes"] - first)
                    if target.read(start + first, length) != source.read(base + source_entry["offset"] + first, length):
                        raise RepackFailure(f"{name}: bytes differ from the source")
            position = start + entry["bytes"]
        if target.size > position and any(target.read(position, target.size - position)):
            raise RepackFailure("non-zero tail padding")
    finally:
        source.close()
        target.close()
    name = final_name or Path(target_path).name
    sha256 = file_sha256(target_path)
    Path(str(target_path) + ".sha256.partial").write_text(f"{sha256}  {name}\n")
    sidecar = str(target_path) + ".experts.partial"
    with open(sidecar, "wb") as out:
        out.write(struct.pack("<IIII", 0x58504557, 2, len(records), 0))
        for record in records:
            out.write(record)
        out.flush()
        os.fsync(out.fileno())
    verdict = WeightdManifest()
    if lib.SparkWeightdManifestLoad(sidecar.encode(), os.path.getsize(target_path), ctypes.byref(verdict)) != 0:
        raise RepackFailure("weightd's manifest loader rejected the format 3 sidecar")
    lib.SparkWeightdManifestDestroy(ctypes.byref(verdict))
    print(json.dumps({"verified": str(target_path), "sha256": sha256, "expert_ranges": len(records)}), flush=True)


def main(argv):
    if len(argv) == 3 and argv[1] == "check":
        return check(argv[2])
    if len(argv) == 5 and argv[1] == "convert":
        convert(argv[2], argv[3], argv[4])
        return 0
    if len(argv) == 5 and argv[1] == "verify":
        verify(argv[2], argv[3], argv[4])
        for suffix in (".sha256", ".experts"):
            os.replace(argv[3] + suffix + ".partial", argv[3] + suffix)
        return 0
    print("usage: k3_repack.py check <v2.pack>\n"
          "       k3_repack.py convert <v2.pack> <v3.pack> <libk3manifest.so>\n"
          "       k3_repack.py verify <v2.pack> <v3.pack> <libk3manifest.so>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except RepackFailure as failure:
        print(f"k3_repack: {failure}", file=sys.stderr)
        sys.exit(1)
