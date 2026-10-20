#!/usr/bin/env python3
import ctypes
import json
import mmap
import os
import struct
import sys
from pathlib import Path

MAGIC = 0x4B33504B
VERSION = 2
ALIGN = 128
CELL_ROWS = 17
W1_CLASS = "expert_cells"
W2_CLASS = "expert_k"


class ReshardFailure(RuntimeError):
    pass


class Pack:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(self.path, "rb")
        self.raw = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, length = struct.unpack_from("<IIQ", self.raw, 0)
        if magic != MAGIC or version != VERSION:
            raise ReshardFailure(f"{path}: not a K3 V2 pack")
        self.length = length
        self.manifest = json.loads(self.raw[16:16 + length])
        self.base = 16 + length
        self.base += (-self.base) % ALIGN
        self.config = self.manifest["config"]
        self.degree = int(self.config.get("tp_degree", 1))
        self.rank = int(self.config.get("tp_rank", 0))

    def entry(self, name):
        return self.manifest["tensors"][name]

    def view(self, name):
        entry = self.entry(name)
        start = self.base + entry["offset"]
        return memoryview(self.raw)[start:start + entry["bytes"]]

    def close(self):
        self.raw.close()
        self.handle.close()


def expert_layers(pack):
    names = [n for n in pack.manifest["tensors"] if n.endswith(".expert_w1_weight")]
    layers = sorted(int(n.split(".")[2]) for n in names)
    for layer in layers:
        for kind in ("w1", "w2"):
            name = f"model.layers.{layer}.expert_{kind}_weight"
            if name not in pack.manifest["tensors"]:
                raise ReshardFailure(f"{name} is missing")
    return layers


def geometry(pack, layer, kind):
    entry = pack.entry(f"model.layers.{layer}.expert_{kind}_weight")
    geom = entry.get("interleave")
    if geom is None or entry.get("kind") != "mxfp4_ws_interleaved_v1":
        raise ReshardFailure(f"layer {layer} {kind} is not an interleaved MXFP4 tensor")
    return entry, geom


def source_plan(pack, layer):
    degree = pack.degree
    w1_entry, w1 = geometry(pack, layer, "w1")
    w2_entry, w2 = geometry(pack, layer, "w2")
    if w1_entry.get("shard_class") != "concat_output" or w2_entry.get("shard_class") != "input_dim":
        raise ReshardFailure(f"layer {layer}: expected a K-split w1 and an output-split w2, found "
                             f"{w1_entry.get('shard_class')} and {w2_entry.get('shard_class')}")
    if w1["cells"] % (2 * degree) != 0 or w2["k_tiles"] % degree != 0:
        raise ReshardFailure(f"layer {layer}: {w1['cells']} w1 cells or {w2['k_tiles']} w2 k-tiles "
                             f"do not split {degree} ways")
    cell_bytes = CELL_ROWS * w1["row_bytes"]
    if CELL_ROWS * w2["row_bytes"] != cell_bytes or w1["experts"] != w2["experts"]:
        raise ReshardFailure(f"layer {layer}: w1 and w2 cell geometry differ")
    return {
        "experts": w1["experts"],
        "cell_bytes": cell_bytes,
        "w1_k_tiles": w1["k_tiles"],
        "w1_cells": w1["cells"],
        "w1_half": w1["cells"] // 2,
        "w1_take": w1["cells"] // (2 * degree),
        "w1_expert_bytes": w1["expert_bytes"],
        "w2_k_tiles": w2["k_tiles"],
        "w2_cells": w2["cells"],
        "w2_take": w2["k_tiles"] // degree,
        "w2_expert_bytes": w2["expert_bytes"],
    }


def piece_bytes_per_expert(plan):
    w1 = plan["w1_k_tiles"] * 2 * plan["w1_take"] * plan["cell_bytes"]
    w2 = plan["w2_take"] * plan["w2_cells"] * plan["cell_bytes"]
    return w1, w2


def write_pieces(pack_path, out_dir):
    pack = Pack(pack_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = expert_layers(pack)
    plans = {layer: source_plan(pack, layer) for layer in layers}
    handles = [open(out / f"to{target:02d}.from{pack.rank:02d}.piece.partial", "wb") for target in range(pack.degree)]
    for layer in layers:
        plan = plans[layer]
        cell = plan["cell_bytes"]
        w1 = pack.view(f"model.layers.{layer}.expert_w1_weight")
        w2 = pack.view(f"model.layers.{layer}.expert_w2_weight")
        take = plan["w1_take"]
        for target in range(pack.degree):
            sink = handles[target]
            for expert in range(plan["experts"]):
                w1_base = expert * plan["w1_expert_bytes"]
                for tile in range(plan["w1_k_tiles"]):
                    row = tile * plan["w1_cells"]
                    gate = w1_base + (row + target * take) * cell
                    up = w1_base + (row + plan["w1_half"] + target * take) * cell
                    sink.write(w1[gate:gate + take * cell])
                    sink.write(w1[up:up + take * cell])
                w2_base = expert * plan["w2_expert_bytes"]
                for tile in range(target * plan["w2_take"], (target + 1) * plan["w2_take"]):
                    start = w2_base + tile * plan["w2_cells"] * cell
                    sink.write(w2[start:start + plan["w2_cells"] * cell])
        del w1, w2
    for target, sink in enumerate(handles):
        sink.close()
        partial = out / f"to{target:02d}.from{pack.rank:02d}.piece.partial"
        os.replace(partial, out / f"to{target:02d}.from{pack.rank:02d}.piece")
    expected = sum(sum(piece_bytes_per_expert(plans[layer])) * plans[layer]["experts"] for layer in layers)
    for target in range(pack.degree):
        size = (out / f"to{target:02d}.from{pack.rank:02d}.piece").stat().st_size
        if size != expected:
            raise ReshardFailure(f"piece for rank {target} holds {size} bytes, expected {expected}")
    print(f"pieces from rank {pack.rank}: {pack.degree} x {expected} bytes")
    pack.close()


def cell_split_manifest(pack, layers, plans):
    manifest = json.loads(json.dumps(pack.manifest))
    degree = pack.degree
    for layer in layers:
        plan = plans[layer]
        w1 = manifest["tensors"][f"model.layers.{layer}.expert_w1_weight"]
        w2 = manifest["tensors"][f"model.layers.{layer}.expert_w2_weight"]
        g1 = w1["interleave"]
        g2 = w2["interleave"]
        w1_cells = 2 * plan["w1_take"]
        w1_k_tiles = plan["w1_k_tiles"] * degree
        w2_cells = plan["w2_cells"] * degree
        w2_k_tiles = plan["w2_take"]
        g1.update({"out_dim": w1_cells * 16, "k_dim": w1_k_tiles * g1["tile_k"], "cells": w1_cells, "k_tiles": w1_k_tiles})
        g2.update({"out_dim": w2_cells * 16, "k_dim": w2_k_tiles * g2["tile_k"], "cells": w2_cells, "k_tiles": w2_k_tiles})
        for geom in (g1, g2):
            if geom["k_tiles"] * geom["cells"] * CELL_ROWS != geom["rows_per_expert"]:
                raise ReshardFailure(f"layer {layer}: the cell split changes the expert size")
        w1["shape"] = [g1["experts"], g1["out_dim"], g1["k_dim"]]
        w2["shape"] = [g2["experts"], g2["out_dim"], g2["k_dim"]]
        w1["shard_class"] = W1_CLASS
        w2["shard_class"] = W2_CLASS
    manifest["config"]["expert_split"] = "cells"
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    if len(encoded) > pack.length:
        raise ReshardFailure(f"the rewritten manifest is {len(encoded)} bytes, over the {pack.length} the pack reserves")
    return encoded + b" " * (pack.length - len(encoded))


def assemble(pack_path, pieces_dir, out_path):
    pack = Pack(pack_path)
    degree = pack.degree
    layers = expert_layers(pack)
    plans = {layer: source_plan(pack, layer) for layer in layers}
    pieces_dir = Path(pieces_dir)
    sources = []
    expected = sum(sum(piece_bytes_per_expert(plans[layer])) * plans[layer]["experts"] for layer in layers)
    for source in range(degree):
        path = pieces_dir / f"to{pack.rank:02d}.from{source:02d}.piece"
        if path.stat().st_size != expected:
            raise ReshardFailure(f"{path.name} holds {path.stat().st_size} bytes, expected {expected}")
        handle = open(path, "rb")
        sources.append(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
        handle.close()
    layer_offset = {}
    cursor = 0
    for layer in layers:
        layer_offset[layer] = cursor
        cursor += sum(piece_bytes_per_expert(plans[layer])) * plans[layer]["experts"]
    header = cell_split_manifest(pack, layers, plans)
    replaced = {}
    for layer in layers:
        replaced[pack.entry(f"model.layers.{layer}.expert_w1_weight")["offset"]] = (layer, "w1")
        replaced[pack.entry(f"model.layers.{layer}.expert_w2_weight")["offset"]] = (layer, "w2")
    out_final = Path(out_path)
    partial = out_final.with_name(out_final.name + ".partial")
    payload_bytes = len(pack.raw) - pack.base
    with open(partial, "wb") as out:
        out.write(struct.pack("<IIQ", MAGIC, VERSION, len(header)))
        out.write(header)
        out.write(b"\0" * (pack.base - 16 - len(header)))
        ordered = sorted(pack.manifest["tensors"].values(), key=lambda entry: entry["offset"])
        position = 0
        for entry in ordered:
            if entry["offset"] > position:
                out.write(pack.raw[pack.base + position:pack.base + entry["offset"]])
            target = replaced.get(entry["offset"])
            if target is None:
                out.write(pack.raw[pack.base + entry["offset"]:pack.base + entry["offset"] + entry["bytes"]])
            else:
                write_expert_tensor(out, target, plans[target[0]], layer_offset[target[0]], sources, degree)
            position = entry["offset"] + entry["bytes"]
        if payload_bytes > position:
            out.write(pack.raw[pack.base + position:pack.base + payload_bytes])
        out.flush()
        os.fsync(out.fileno())
    if partial.stat().st_size != len(pack.raw):
        raise ReshardFailure(f"{partial.name} is {partial.stat().st_size} bytes, the source pack is {len(pack.raw)}")
    os.replace(partial, out_final)
    for source in sources:
        source.close()
    print(f"assembled rank {pack.rank} cell-split pack {out_final} ({len(layers)} expert layers)")
    pack.close()


def write_expert_tensor(out, target, plan, layer_offset, sources, degree):
    layer, kind = target
    cell = plan["cell_bytes"]
    w1_piece, w2_piece = piece_bytes_per_expert(plan)
    per_expert = w1_piece + w2_piece
    for expert in range(plan["experts"]):
        start = layer_offset + expert * per_expert
        if kind == "w1":
            for source in range(degree):
                out.write(sources[source][start:start + w1_piece])
            continue
        row = plan["w2_cells"] * cell
        for tile in range(plan["w2_take"]):
            for source in range(degree):
                begin = start + w1_piece + tile * row
                out.write(sources[source][begin:begin + row])


def write_sidecar(pack_path, library_path, out_path):
    lib = ctypes.CDLL(str(library_path))
    lib.SparkCk128Initialize.argtypes = [ctypes.c_void_p]
    lib.SparkCk128Update.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.SparkCk128Finalize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SparkWeightdManifestLoad.restype = ctypes.c_int

    class Manifest(ctypes.Structure):
        _fields_ = [(n, ctypes.c_void_p) for n in ("ranges", "groups", "spine")] + \
                   [(n, ctypes.c_uint64) for n in ("spine_bytes", "spine_allocation_bytes")] + \
                   [(n, ctypes.c_uint32) for n in ("range_count", "group_count", "spine_count")]

    lib.SparkWeightdManifestLoad.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(Manifest)]

    def ck128(data):
        state = ctypes.create_string_buffer(256)
        digest = ctypes.create_string_buffer(16)
        lib.SparkCk128Initialize(state)
        lib.SparkCk128Update(state, data, len(data))
        lib.SparkCk128Finalize(state, digest)
        return digest.raw

    pack = Pack(pack_path)
    records = []
    for name, entry in pack.manifest["tensors"].items():
        if not (name.endswith("expert_w1_weight") or name.endswith("expert_w2_weight")):
            continue
        geom = entry["interleave"]
        layer = int(name.split(".")[2])
        kind = 0 if name.endswith("w1_weight") else 1
        tensor_base = pack.base + entry["offset"]
        for expert in range(geom["experts"]):
            begin = tensor_base + expert * geom["expert_bytes"]
            data = bytes(pack.raw[begin:begin + geom["expert_bytes"]])
            records.append(struct.pack("<4I2Q16s", layer, expert, kind, 0, begin, geom["expert_bytes"], ck128(data)))
    pack_bytes = len(pack.raw)
    pack.close()
    out_final = Path(out_path)
    partial = out_final.with_name(out_final.name + ".partial")
    with open(partial, "wb") as out:
        out.write(struct.pack("<IIII", 0x58504557, 2, len(records), 0))
        for record in records:
            out.write(record)
        out.flush()
        os.fsync(out.fileno())
    verdict = Manifest()
    if lib.SparkWeightdManifestLoad(str(partial).encode(), pack_bytes, ctypes.byref(verdict)) != 0:
        os.unlink(partial)
        raise ReshardFailure(f"{out_final.name}: weightd's manifest loader rejected the sidecar")
    lib.SparkWeightdManifestDestroy(ctypes.byref(verdict))
    os.replace(partial, out_final)
    print(f"{out_final.name}: {len(records)} expert ranges")


def main(argv):
    try:
        if len(argv) == 4 and argv[1] == "pieces":
            write_pieces(argv[2], argv[3])
            return 0
        if len(argv) == 5 and argv[1] == "assemble":
            assemble(argv[2], argv[3], argv[4])
            return 0
        if len(argv) == 5 and argv[1] == "sidecar":
            write_sidecar(argv[2], argv[3], argv[4])
            return 0
    except ReshardFailure as failure:
        print(f"RESHARD FAILURE: {failure}")
        return 1
    print("usage: k3_reshard_cells.py pieces <rank.pack> <out_dir>\n"
          "       k3_reshard_cells.py assemble <rank.pack> <pieces_dir> <new.pack>\n"
          "       k3_reshard_cells.py sidecar <pack> <libk3manifest.so> <out.experts>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
