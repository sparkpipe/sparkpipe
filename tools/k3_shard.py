#!/usr/bin/env python3
"""Slice a K3 resident pack (format V2) into per-rank TP packs.

The single source of truth for how each K3 tensor splits across a TP group,
in the glm52 shard module's shape: classification is by the manifest's field
name, an unclassified name refuses the whole slice rather than guessing, and
every split is checked against the alignment it owes - output splits land on
whole head blocks, input splits on whole interleave k-tiles when the axis is
packed nibbles with co-tiled scales.

The standard Megatron split for a weight stored [out, in]: output-dimension
shards need no collective, input-dimension shards produce partials the
layer's closing all-reduce combines (SparkTpCollectiveAllReduceSumF32 is that
collective on this ring), replicated tensors load whole. K3's departures
from glm52, each earned by the architecture:

  the low-rank bottlenecks REPLICATE, and V2 fuses the two KDA ones into
  kda_decay_gate_down_weight (docs/K3_PACK_FORMAT_V2.md): their 128-wide
  output is what every rank's up half reads in full, and slicing 128 sixteen
  ways buys nothing but a collective
  the kv_a latent path replicates, which is what keeps the latent KV cache
  identical per rank - TP cannot shard one KV head
  V2 fuses q|k|v|beta into kda_qkv_beta_weight, ONE OUTPUT_DIM_HEADS tensor:
  a rank holding heads [h0, h1) owns one contiguous row range PER SECTION,
  the per-head section widths (128/128/128/1) coming from the manifest's
  section table, beta's single row per head included
  the concatenated gate|up tensors (shared, dense, and every expert's w1)
  slice EACH HALF and re-concatenate per rank, or the SiTU kernel's
  gate-first contract breaks at every rank boundary
  expert w1 is the V2 interleaved grid (64B rows, 16 payload + 1 scale row
  per 16-neuron cell per 128-element k-tile): it output-splits on whole
  16-neuron cells per gate|up half, so each rank's shard is itself a valid
  interleaved tensor with the scales riding along
  expert w2 input-splits on whole 128-element k-tiles - a contiguous row
  range per expert per rank. V1 split K on 32-element groups; the interleave
  coarsens that to the k-tile, so K3's 24 w2 k-tiles admit TP 1/2/4/8 and
  TP16 is REFUSED, not approximated
"""
import json
import mmap
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_pack  # noqa: E402  the interleave geometry is priced once, there

MAGIC = 0x4B33504B
ALIGN = 128


class ShardFailure(RuntimeError):
    pass


REPLICATED = {
    "attn_norm_weight", "mlp_norm_weight", "attnres_attn_weight",
    "attnres_mlp_weight", "router_weight", "router_bias",
    "kda_decay_down_weight", "kda_decay_bias",
    "kda_head_log_scale", "kda_out_norm_weight",
    "mla_q_down_weight", "mla_q_norm_weight", "mla_kv_a_weight",
    "mla_kv_a_norm_weight", "routed_norm_weight",
}
MODEL_REPLICATED = {"model.norm.weight", "model.attnres_out_weight"}
# output-dimension, sliced on whole head blocks: (head elements, row bytes)
OUTPUT_HEADS = {
    "kda_qkv_beta_weight": ("kda_qkvb", 1),
    "kda_q_conv_weight": ("kda", 1), "kda_k_conv_weight": ("kda", 1),
    "kda_v_conv_weight": ("kda", 1),
    "kda_decay_up_weight": ("kda", 1),
    "kda_gate_weight": ("kda", 1),
    "mla_q_up_weight": ("mla_q", 1), "mla_kv_b_value_weight": ("mla_v", 1),
    "mla_gate_weight": ("mla_v", 1),
}
# input-dimension on whole head blocks, partials summed by the all-reduce
INPUT_HEADS = {"kda_out_weight": ("kda", 1), "mla_out_weight": ("mla_v", 1)}
# plain output rows / degree
OUTPUT_DIM = {"routed_down_weight"}
# plain input columns / degree, all-reduce after
INPUT_DIM = {"routed_up_weight"}
# [gate; up] concatenated: each half output-splits, re-concatenated per rank
CONCAT_OUTPUT = {"shared_w1_weight", "dense_gate_up_weight"}
INPUT_DIM_PLAIN = {"shared_w2_weight", "dense_down_weight"}
# the V2 interleaved expert tensors: one weight+scale stream each, no scale
# planes. w1 output-splits on whole 16-neuron cells per gate|up half, w2
# input-splits on whole 128-element k-tiles.
EXPERT_CONCAT = {"expert_w1_weight"}
EXPERT_INPUT = {"expert_w2_weight"}


def head_block(kind, geo):
    return {"kda": geo["kda_head"],
            "mla_q": geo["kv_lora"] + geo["rope"], "mla_v": geo["v_head"]}[kind]


def head_count(kind, geo):
    return geo["kda_heads"] if kind.startswith("kda") else geo["heads"]


def slice_rows(raw, rows, rank, degree):
    if rows % degree != 0:
        raise ShardFailure(f"{rows} rows do not split {degree} ways")
    per = len(raw) // rows
    lo = (rows // degree) * rank
    return raw[lo * per:(lo + rows // degree) * per]


def slice_cols(raw, rows, row_bytes, lo_byte, hi_byte):
    out = bytearray()
    for r in range(rows):
        out += raw[r * row_bytes + lo_byte:r * row_bytes + hi_byte]
    return bytes(out)


class Slicer:
    def __init__(self, pack_path, geo, degree, rank):
        # The stage packs are ~390 GB: the source is MAPPED, never read. A
        # read_bytes() here is the MemoryError this constructor exists to
        # prevent - four ranks x one full copy is five pack-sized buffers
        # on a 128 GB host.
        self.handle = open(pack_path, "rb")
        raw = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, length = struct.unpack_from("<IIQ", raw, 0)
        if magic != MAGIC:
            raise ShardFailure("not a K3 pack")
        if version != 2:
            raise ShardFailure(
                f"pack format version {version}, expected 2; repack with a "
                f"current tools/k3_pack.py (docs/K3_PACK_FORMAT_V2.md)")
        self.manifest = json.loads(raw[16:16 + length])
        base = 16 + length
        base += (-base) % ALIGN
        self.raw, self.base = raw, base
        self.geo, self.degree, self.rank = geo, degree, rank
        self.config = self.manifest["config"]

    def entry_of(self, name):
        return self.manifest["tensors"][name]

    def bytes_of(self, name):
        entry = self.manifest["tensors"][name]
        return self.raw[self.base + entry["offset"]:
                        self.base + entry["offset"] + entry["bytes"]]

    def emit(self, out_path):
        # STREAMED, never assembled: a rank pack is ~97 GB and a bytearray
        # of it is the second MemoryError this file's layout is shaped to
        # avoid. The payload streams to the file first, behind a fixed
        # manifest reserve; the manifest is built from the recorded offsets
        # and written over the reserve at the end, space-padded to the exact
        # reserve so the payload base (align(16 + manifest_len)) is a
        # compile-time constant of this writer.
        manifest_reserve = 65520  # header(16) + manifest = 65536, aligned
        tensors = {}
        offset = 0
        with open(out_path, "wb") as out:
            out.seek(16 + manifest_reserve)
            for name in self.manifest["tensors"]:
                sliced, meta = self.route(name)
                pad = (-offset) % ALIGN
                if pad:
                    out.write(b"\0" * pad)
                    offset += pad
                entry = {"offset": offset, "bytes": len(sliced)}
                entry.update(meta)
                tensors[name] = entry
                out.write(sliced)
                offset += len(sliced)
            echo = dict(self.config)
            echo.update({"tp_degree": self.degree, "tp_rank": self.rank})
            manifest = json.dumps(
                {"format": self.manifest["format"], "config": echo,
                 "tensors": tensors}, separators=(",", ":")).encode()
            if len(manifest) > manifest_reserve:
                raise ShardFailure(
                    f"manifest {len(manifest)} bytes overruns the "
                    f"{manifest_reserve}-byte reserve")
            manifest = manifest + b" " * (manifest_reserve - len(manifest))
            out.seek(0)
            out.write(struct.pack("<IIQ", MAGIC, 2, len(manifest)))
            out.write(manifest)
        return tensors

    def route(self, name):
        """Returns (sliced bytes, per-rank manifest metadata): the source
        entry minus offset/bytes, with shape/sections/interleave adjusted so
        the rank pack stays a self-describing V2 pack."""
        geo, degree, rank, cfg = self.geo, self.degree, self.rank, self.config
        raw = self.bytes_of(name)
        field = name.split(".")[-1]
        meta = {k: v for k, v in self.entry_of(name).items()
                if k not in ("offset", "bytes")}

        def shape(rows=None, cols=None):
            if "shape" in meta and len(meta["shape"]) == 2:
                if rows is not None:
                    meta["shape"] = [rows, meta["shape"][1]]
                if cols is not None:
                    meta["shape"] = [meta["shape"][0], cols]

        if name in MODEL_REPLICATED or field in REPLICATED:
            return raw, meta
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            shape(rows=cfg["vocab"] // degree)
            return slice_rows(raw, cfg["vocab"], rank, degree), meta
        if field == "kda_qkv_beta_weight":
            return self._fused_qkvb(name, raw, meta), meta
        if field in OUTPUT_HEADS:
            kind, _ = OUTPUT_HEADS[field]
            heads = head_count(kind, geo)
            if heads % degree != 0:
                raise ShardFailure(f"{name}: {heads} heads over {degree} ranks")
            rows = heads * head_block(kind, geo)
            shape(rows=rows // degree)
            return slice_rows(raw, rows, rank, degree), meta
        if field in INPUT_HEADS:
            kind, _ = INPUT_HEADS[field]
            heads = head_count(kind, geo)
            if heads % degree != 0:
                raise ShardFailure(f"{name}: {heads} heads over {degree} ranks")
            in_bytes = heads * head_block(kind, geo) * 2
            per = in_bytes // degree
            shape(cols=(heads * head_block(kind, geo)) // degree)
            return slice_cols(raw, len(raw) // in_bytes, in_bytes,
                              rank * per, (rank + 1) * per), meta
        if field in OUTPUT_DIM:
            shape(rows=cfg["latent"] // degree)
            return slice_rows(raw, cfg["latent"], rank, degree), meta
        if field in INPUT_DIM:
            in_bytes = cfg["latent"] * 2
            per = in_bytes // degree
            shape(cols=cfg["latent"] // degree)
            return slice_cols(raw, len(raw) // in_bytes, in_bytes,
                              rank * per, (rank + 1) * per), meta
        if field in CONCAT_OUTPUT:
            half_rows = self._half_rows(name)
            if "shape" in meta and len(meta["shape"]) == 2:
                meta["shape"] = [meta["shape"][0] // degree, meta["shape"][1]]
            half = len(raw) // 2
            return slice_rows(raw[:half], half_rows, rank, degree) + \
                slice_rows(raw[half:], half_rows, rank, degree), meta
        if field in INPUT_DIM_PLAIN:
            rows = cfg["hidden"]
            in_bytes = len(raw) // rows
            per = in_bytes // degree
            if (in_bytes // 2) % degree != 0:
                raise ShardFailure(f"{name}: input does not split {degree} ways")
            if "shape" in meta and len(meta["shape"]) == 2:
                meta["shape"] = [meta["shape"][0],
                                 meta["shape"][1] // degree]
            return slice_cols(raw, rows, in_bytes, rank * per,
                              (rank + 1) * per), meta
        if field in EXPERT_CONCAT:
            return self._expert_gate_up(name, raw, meta), meta
        if field in EXPERT_INPUT:
            return self._expert_down(name, raw, meta), meta
        raise ShardFailure(f"{name}: unclassified tensor, refusing to guess")

    def _half_rows(self, name):
        # shared_w1 is [2 * shared_inter, hidden]; dense_gate_up is
        # [2 * dense_inter, hidden] - rows of one half from the byte length
        raw = self.bytes_of(name)
        return (len(raw) // 2) // (self.config["hidden"] * 2)

    def _fused_qkvb(self, name, raw, meta):
        """One contiguous row range PER SECTION: a rank holding heads
        [h0, h1) owns [off_s + h0*rph_s, off_s + h1*rph_s) of every section,
        per-head widths 128/128/128/1 from the manifest's section table."""
        geo, degree, rank = self.geo, self.degree, self.rank
        heads = geo["kda_heads"]
        if heads % degree != 0:
            raise ShardFailure(f"{name}: {heads} heads over {degree} ranks")
        h0 = (heads // degree) * rank
        h1 = h0 + heads // degree
        row_bytes = self.config["hidden"] * 2
        out = bytearray()
        sections = []
        for section in meta["sections"]:
            lo = (section["row_offset"] + h0 * section["rows_per_head"]) \
                * row_bytes
            hi = (section["row_offset"] + h1 * section["rows_per_head"]) \
                * row_bytes
            out += raw[lo:hi]
            sections.append({**section, "rows": section["rows"] // degree})
        offset = 0
        for section in sections:
            section["row_offset"] = offset
            offset += section["rows"]
        meta["sections"] = sections
        if "shape" in meta and len(meta["shape"]) == 2:
            meta["shape"] = [offset, meta["shape"][1]]
        return bytes(out)

    def _expert_gate_up(self, name, raw, meta):
        """w1 output-splits on whole 16-neuron cells, each gate|up half on
        its own: per (expert, k-tile) the rank takes its gate cell range then
        its up cell range, so the shard is itself a valid interleaved tensor
        whose gate-first order survives every rank boundary."""
        degree, rank = self.degree, self.rank
        geom = self.entry_of(name)["interleave"]
        experts, cells = geom["experts"], geom["cells"]
        k_tiles, cell_rows, row_bytes = (geom["k_tiles"], geom["cell_rows"],
                                         geom["row_bytes"])
        rpe = geom["rows_per_expert"]
        half = cells // 2  # the gate|up boundary is a cell boundary
        if half % degree != 0:
            raise ShardFailure(
                f"{name}: {half} 16-neuron cells per gate|up half do not "
                f"split {degree} ways")
        take = half // degree
        lo = rank * take
        out = bytearray()
        for e in range(experts):
            block = raw[e * rpe * row_bytes:(e + 1) * rpe * row_bytes]
            for t in range(k_tiles):
                for base in (lo, half + lo):
                    row0 = (t * cells + base) * cell_rows
                    out += block[row0 * row_bytes:
                                 (row0 + take * cell_rows) * row_bytes]
        self._reprice_interleave(name, meta, out_dim=geom["out_dim"] // degree)
        return bytes(out)

    def _expert_down(self, name, raw, meta):
        """w2 input-splits on whole 128-element k-tiles: a contiguous row
        range per expert per rank. The interleave coarsened V1's 32-element
        K groups to the k-tile, so a degree that does not divide the tile
        count is refused - for K3's 24 tiles that excludes TP16."""
        degree, rank = self.degree, self.rank
        geom = self.entry_of(name)["interleave"]
        experts, k_tiles = geom["experts"], geom["k_tiles"]
        if k_tiles % degree != 0:
            raise ShardFailure(
                f"{name}: {k_tiles} 128-element k-tiles do not split "
                f"{degree} ways; the interleaved grid coarsened the K split "
                f"from 32-element groups to whole k-tiles, and a partial "
                f"tile would strand its co-tiled scales")
        per = k_tiles // degree
        t0 = rank * per
        tile_rows = geom["cells"] * geom["cell_rows"]
        row_bytes, rpe = geom["row_bytes"], geom["rows_per_expert"]
        out = bytearray()
        for e in range(experts):
            block = raw[e * rpe * row_bytes:(e + 1) * rpe * row_bytes]
            out += block[t0 * tile_rows * row_bytes:
                         (t0 + per) * tile_rows * row_bytes]
        self._reprice_interleave(name, meta, k_dim=geom["k_dim"] // degree)
        return bytes(out)

    def _reprice_interleave(self, name, meta, out_dim=None, k_dim=None):
        geom = self.entry_of(name)["interleave"]
        priced = k3_pack.interleave_geometry(
            out_dim if out_dim is not None else geom["out_dim"],
            k_dim if k_dim is not None else geom["k_dim"],
            geom["experts"])
        meta["interleave"] = priced
        if "shape" in meta and len(meta["shape"]) == 3:
            meta["shape"] = [priced["experts"], priced["out_dim"],
                             priced["k_dim"]]


def main():
    if len(sys.argv) != 4:
        print("usage: k3_shard.py <in.pack> <out_prefix> <tp_degree>")
        return 2
    degree = int(sys.argv[3])
    if degree & (degree - 1) or degree == 0:
        print("SHARD FAILURE: tp_degree must be a power of two")
        return 1
    try:
        probe = Slicer(sys.argv[1], {}, degree, 0)
    except ShardFailure as failure:
        print(f"SHARD FAILURE: {failure}")
        return 1
    cfg = probe.config
    needed = ("kda_heads", "kda_head", "heads", "kv_lora", "rope", "v_head")
    missing = [k for k in needed if k not in cfg]
    if missing:
        print(f"SHARD FAILURE: pack config lacks geometry {missing}; "
              f"repack with a current tools/k3_pack.py")
        return 1
    geo = {k: cfg[k] for k in needed}
    try:
        for rank in range(degree):
            slicer = Slicer(sys.argv[1], geo, degree, rank)
            tensors = slicer.emit(f"{sys.argv[2]}.rank{rank:02d}.pack")
        print(f"sharded {len(tensors)} tensors x {degree} ranks")
    except ShardFailure as failure:
        print(f"SHARD FAILURE: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
