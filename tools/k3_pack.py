#!/usr/bin/env python3
"""Pack a Kimi K3 checkpoint into the sparkpipe resident layout, format V2.

The engines serve the checkpoint as shipped: MXFP4 expert payloads and their
E8M0 scales are moved, never recomputed, and everything on the ignore list
stays BF16. What the packer OWES beyond moving bytes is what the layer's
weight table cannot get from any checkpoint directly. V2 adds three layout
transforms, each removing a defect the V1 format baked in:

  fused KDA     the six projections that read one normed KDA input shipped as
  projections   six tensors, so six GEMM launches read the same activation
                (layer.cuh K3-PERF-003). One GEMM needs one base pointer, and
                the TP shard table splits the six into two classes, so V2
                emits TWO fused tensors per KDA layer, each carrying exactly
                one shard class:
                  kda_qkv_beta_weight       q|k|v|beta rows, one
                                            [sum_out, hidden] BF16 tensor,
                                            OUTPUT_DIM_HEADS
                  kda_decay_gate_down_weight decay_down|gate_down rows,
                                            REPLICATED
                The manifest's per-tensor section table gives each section's
                row offset and rows-per-head, so bind is pointer arithmetic.

  interleaved   V1 shipped expert payload and E8M0 scales as two planes two
  weight+scale  pointers apart; the GEMM staged the payload by TMA and the
                scales by LDG from the far plane. V2 blocks each expert into
                k-tiles of 128 elements and, per k-tile, interleaves each
                16-neuron cell's sixteen 64-byte payload rows with ONE 64-byte
                scale row (16 neurons x 4 group scales - exactly full). The
                ratio is exact, so the tensor is zero padding: payload+scales
                = cells * k_tiles * 17 * 64 bytes per expert. One TMA box of
                17 * (TILE_N/16) rows then fetches a stage's payload AND its
                scales in a single transaction off one descriptor, the 64-byte
                swizzle span intact. The addressing math is
                interleave_geometry / interleave_byte_offset below and the
                consumption contract is docs/K3_PACK_FORMAT_V2.md.

  alignment     every tensor starts 128-byte aligned (TMA base and L2 line),
                tensors are emitted layer by layer in consumption order so a
                sequential prefetcher never re-seeks, and the manifest header
                block carries per-tensor kind/shape/sections/interleave so a
                loader binds with adds, not parsing.

  q-fold        kv_b's k_nope half absorbed into q_b, per head:
                A[h] = kv_b_k[h]^T @ q_b_nope[h], then the unrotated rows -
                heads * (kv_lora + rope) rows over q_lora
  kv_b split    the value half of kv_b as its own [heads * v_head, kv_lora]
                per-head table (the gate lives in v-space and does not
                commute, so o_proj stays as shipped)
  gamma folds   every attention-residual score projection multiplied
                elementwise by its RMSNorm gamma - exact, the kernel norms
                without a weight
  conv flatten  [channels, 1, kernel] to [channels, kernel] (a shape
                assertion; the row-major bytes are identical)

NUMPY IS OPTIONAL. Every layout transform (fusion, interleave, alignment,
manifest) is pure stdlib and bit-exact without numpy; numpy, when importable,
only accelerates the two heavy paths (the q-fold matmul and the expert
interleave relay). The pure-python q-fold fallback emulates float32 per
operation - bit-exact f32 semantics, but the accumulation ORDER is
sequential where numpy's may be pairwise, so the fallback can differ in the
last ulp and is for host verification, not for shipping a pack. It is also
O(n^3) in python: fine for the test mini, days for the real checkpoint.

Failure is loud: a missing tensor, an E8M0 0xff, a group size that is not 32,
an expert count that is not the config's, a dimension the interleave grid
does not divide - each is a PackFailure naming what and where.
"""
import json
import math
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # layout logic below is stdlib-only; see module docstring
    np = None

MAGIC = 0x4B33504B  # 'K3PK'
VERSION = 2
ALIGN = 128            # tensor alignment: TMA base + L2 line
E8M0_NAN = 0xFF
GROUP = 32             # MXFP4 scale group, elements
TILE_K = 128           # interleave k-tile, elements - LmMxfp4::kTileK, the
                       # extent whose 64 payload bytes fill one swizzle span
CELL_PAYLOAD_ROWS = 16 # payload rows per interleave cell; the exact ratio
                       # 16 * 4 scale bytes == one 64-byte row is why 16
A_LOG_SOURCE_HEADS = 128

KIND_BF16 = "bf16"
KIND_F32 = "f32"
KIND_MXFP4_INTERLEAVED = "mxfp4_ws_interleaved_v1"


class PackFailure(RuntimeError):
    pass


# -- pure-python scalar helpers ----------------------------------------------
#
# float32 emulation: a python float is f64, and the product of two f32 values
# is exact in f64 (24 + 24 <= 53 mantissa bits), so rounding the f64 result to
# f32 equals a hardware f32 multiply. Rounding an f32 to BF16 round-to-nearest-
# even is then exact, which is what makes the gamma folds bit-identical to the
# numpy path they replaced.

def f32_round(value):
    # an f64 product past the f32 ceiling rounds to inf, exactly what the
    # numpy f32 multiply this emulates produces
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bf16_raw_to_f32(raw):
    return [struct.unpack("<f", struct.pack("<I", u << 16))[0]
            for (u,) in struct.iter_unpack("<H", raw)]


def f32_list_to_bf16_raw(values):
    out = bytearray()
    for value in values:
        u = struct.unpack("<I", struct.pack("<f", value))[0]
        out += struct.pack("<H", (u + 0x7FFF + ((u >> 16) & 1)) >> 16)
    return bytes(out)


if np is not None:
    def bf16_to_f32(u16):
        return (u16.astype(np.uint32) << 16).view(np.float32)

    def f32_to_bf16(f32):
        u = f32.astype(np.float32).view(np.uint32)
        rounded = u + 0x7FFF + ((u >> 16) & 1)
        return (rounded >> 16).astype(np.uint16)


# -- fused KDA projection section tables --------------------------------------
#
# Both fused tensors are stored [sum_out, hidden] row-major like every other
# projection here; sections are row ranges in the order listed. rows_per_head
# is what a TP OUTPUT_DIM_HEADS slice takes per head from that section - the
# fused tensor carries ONE shard class because each section splits by whole
# heads, beta's single row per head included.

def kda_fused_qkvb_sections(heads, key_dim, value_dim):
    """q|k|v|beta: the OUTPUT_DIM_HEADS half of the six-way KDA fusion."""
    sections = []
    row = 0
    for name, rows, per_head in (
            ("q", heads * key_dim, key_dim),
            ("k", heads * key_dim, key_dim),
            ("v", heads * value_dim, value_dim),
            ("beta", heads, 1)):
        sections.append({"name": name, "row_offset": row, "rows": rows,
                         "rows_per_head": per_head})
        row += rows
    return sections, row


def kda_fused_decay_gate_down_sections(head_dim):
    """decay_down|gate_down: the REPLICATED half. Bottleneck widths are not
    head-split by the TP table, so rows_per_head is 0 - the section exists for
    bind arithmetic, not for slicing."""
    sections = []
    row = 0
    for name in ("decay_down", "gate_down"):
        sections.append({"name": name, "row_offset": row, "rows": head_dim,
                         "rows_per_head": 0})
        row += head_dim
    return sections, row


# -- interleaved weight+scale geometry ----------------------------------------
#
# One expert is a byte grid of 64-byte rows:
#
#   row(expert e, k_tile t, cell c, sub r) =
#       e * rows_per_expert + (t * cells + c) * 17 + r
#
# sub 0..15 hold the payload k-tile of neurons 16c+sub; sub 16 holds the 64
# scale bytes of that cell's sixteen neurons for this k-tile, byte 4n+j the
# E8M0 of neuron 16c+n, k-group 4t+j. Everything a consumer needs is derived
# here once - the packer, the validator, the layout test and the doc all price
# the same arithmetic.

def interleave_geometry(out_dim, k_dim, experts=1, tile_k=TILE_K, group=GROUP,
                        stored_bits=4):
    tile_payload = tile_k * stored_bits // 8      # 64: one swizzle span
    tile_scale = tile_k // group                  # 4 scale bytes/neuron/tile
    if tile_payload != CELL_PAYLOAD_ROWS * tile_scale:
        raise PackFailure(
            f"interleave cell does not close: {CELL_PAYLOAD_ROWS} payload "
            f"rows of {tile_payload}B need a {CELL_PAYLOAD_ROWS * tile_scale}B "
            f"scale row, not {tile_payload}B")
    if k_dim % tile_k != 0:
        raise PackFailure(f"K {k_dim} is not a whole number of {tile_k}-element "
                          f"interleave tiles")
    if k_dim % group != 0:
        raise PackFailure(f"K {k_dim} is not whole MXFP4 groups of {group}")
    if out_dim % CELL_PAYLOAD_ROWS != 0:
        raise PackFailure(f"output {out_dim} is not a whole number of "
                          f"{CELL_PAYLOAD_ROWS}-neuron interleave cells")
    k_tiles = k_dim // tile_k
    cells = out_dim // CELL_PAYLOAD_ROWS
    cell_rows = CELL_PAYLOAD_ROWS + 1
    rows_per_expert = k_tiles * cells * cell_rows
    expert_bytes = rows_per_expert * tile_payload
    payload_bytes = out_dim * (k_dim * stored_bits // 8)
    scale_bytes = out_dim * (k_dim // group)
    if expert_bytes != payload_bytes + scale_bytes:
        raise PackFailure("interleave padding is not zero; the grid is wrong")
    return {"kind": KIND_MXFP4_INTERLEAVED, "out_dim": out_dim,
            "k_dim": k_dim, "experts": experts, "tile_k": tile_k,
            "group": group, "stored_bits": stored_bits,
            "row_bytes": tile_payload,
            "scale_bytes_per_neuron_tile": tile_scale,
            "cell_payload_rows": CELL_PAYLOAD_ROWS, "cell_rows": cell_rows,
            "cells": cells, "k_tiles": k_tiles,
            "rows_per_expert": rows_per_expert,
            "expert_bytes": expert_bytes,
            "tensor_bytes": expert_bytes * experts}


def interleave_byte_offset(geom, expert, k_tile, neuron, kind, lane=0):
    """Byte offset of one payload byte lane (0..63) or one scale group lane
    (0..3) of one neuron inside the interleaved tensor. The published
    addressing contract; the relays below must agree with it and the layout
    test holds them to it."""
    cell, sub = divmod(neuron, geom["cell_payload_rows"])
    row = (expert * geom["rows_per_expert"]
           + (k_tile * geom["cells"] + cell) * geom["cell_rows"])
    if kind == "payload":
        return (row + sub) * geom["row_bytes"] + lane
    if kind == "scale":
        return (row + geom["cell_payload_rows"]) * geom["row_bytes"] \
            + sub * geom["scale_bytes_per_neuron_tile"] + lane
    raise PackFailure(f"unknown interleave lane kind {kind!r}")


def interleave_py(payload, scales, geom):
    """Stdlib relay: payload [experts][out][k/2] and scales [experts][out]
    [k/32] as raw bytes into the interleaved grid. Slice-assignment moves
    whole rows, so this is memory-bound, not loop-bound."""
    experts = geom["experts"]
    row_payload = geom["k_dim"] * geom["stored_bits"] // 8
    k_groups = geom["k_dim"] // geom["group"]
    per_expert_payload = geom["out_dim"] * row_payload
    per_expert_scale = geom["out_dim"] * k_groups
    if len(payload) != experts * per_expert_payload or \
            len(scales) != experts * per_expert_scale:
        raise PackFailure("interleave source plane size does not match geometry")
    out = bytearray(geom["tensor_bytes"])
    for e in range(experts):
        pay_base = e * per_expert_payload
        sc_base = e * per_expert_scale
        for t in range(geom["k_tiles"]):
            for c in range(geom["cells"]):
                dst = (e * geom["rows_per_expert"]
                       + (t * geom["cells"] + c) * geom["cell_rows"]) \
                    * geom["row_bytes"]
                for r in range(geom["cell_payload_rows"]):
                    n = c * geom["cell_payload_rows"] + r
                    src = pay_base + n * row_payload + t * geom["row_bytes"]
                    out[dst + r * geom["row_bytes"]:
                        dst + (r + 1) * geom["row_bytes"]] = \
                        payload[src:src + geom["row_bytes"]]
                srow = dst + geom["cell_payload_rows"] * geom["row_bytes"]
                for r in range(geom["cell_payload_rows"]):
                    n = c * geom["cell_payload_rows"] + r
                    src = sc_base + n * k_groups \
                        + t * geom["scale_bytes_per_neuron_tile"]
                    width = geom["scale_bytes_per_neuron_tile"]
                    out[srow + r * width:srow + (r + 1) * width] = \
                        scales[src:src + width]
    return bytes(out)


def interleave_np(payload, scales, geom):
    """The same grid by reshape+transpose: payload (E, out, k_tiles, 64) ->
    (E, k_tiles, cells, 16, 64); scales (E, out, k_tiles, 4) ->
    (E, k_tiles, cells, 1, 64); concatenate along the sub-row axis."""
    row_bytes = geom["row_bytes"]
    width = geom["scale_bytes_per_neuron_tile"]
    p = payload.reshape(geom["experts"], geom["out_dim"], geom["k_tiles"],
                        row_bytes).transpose(0, 2, 1, 3) \
        .reshape(geom["experts"], geom["k_tiles"], geom["cells"],
                 geom["cell_payload_rows"], row_bytes)
    s = scales.reshape(geom["experts"], geom["out_dim"], geom["k_tiles"],
                       width).transpose(0, 2, 1, 3) \
        .reshape(geom["experts"], geom["k_tiles"], geom["cells"], 1,
                 geom["cell_payload_rows"] * width)
    return np.concatenate([p, s], axis=3).tobytes()


def interleave(payload, scales, geom):
    if np is not None:
        p = np.frombuffer(payload, dtype=np.uint8) \
            .reshape(geom["experts"], geom["out_dim"], -1)
        s = np.frombuffer(scales, dtype=np.uint8) \
            .reshape(geom["experts"], geom["out_dim"], -1)
        return interleave_np(p, s, geom)
    return interleave_py(payload, scales, geom)


# -- checkpoint reading --------------------------------------------------------

class SafetensorDir:
    """Minimal reader: index.json plus shards, or a single model.safetensors.
    Returns raw bytes and the header's shape/dtype; interpretation is the
    caller's."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.model_dir / "model.safetensors"
            if not single.is_file():
                raise PackFailure(f"no safetensors index or file in {model_dir}")
            self.weight_map = None
            self.single = single.name
        self.headers = {}

    def _header(self, shard):
        if shard not in self.headers:
            path = self.model_dir / shard
            with open(path, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
            header.pop("__metadata__", None)
            self.headers[shard] = (header, 8 + length)
        return self.headers[shard]

    def names(self):
        if self.weight_map is not None:
            return set(self.weight_map)
        return set(self._header(self.single)[0])

    def tensor(self, name):
        shard = self.weight_map.get(name) if self.weight_map is not None \
            else (self.single if name in self._header(self.single)[0] else None)
        if shard is None:
            raise PackFailure(f"missing tensor: {name}")
        header, base = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        with open(self.model_dir / shard, "rb") as handle:
            handle.seek(base + begin)
            raw = handle.read(end - begin)
        return entry["dtype"], tuple(entry["shape"]), raw

    def expect(self, name, dtype, shape=None):
        got_dtype, got_shape, raw = self.tensor(name)
        if got_dtype != dtype:
            raise PackFailure(f"{name}: expected {dtype}, checkpoint says "
                              f"{got_dtype}")
        if shape is not None and got_shape != tuple(shape):
            raise PackFailure(f"{name}: shape {got_shape}, expected "
                              f"{tuple(shape)}")
        return raw

    def bf16(self, name, shape=None):
        return self.expect(name, "BF16", shape)

    def u8(self, name, shape=None):
        return self.expect(name, "U8", shape)

    def f32(self, name, shape=None):
        return self.expect(name, "F32", shape)


def quant_pair(reader, base):
    """The routed experts as compressed-tensors serialises them. Two spellings
    exist in the wild; both are probed and anything else is loud."""
    names = reader.names()
    if base + ".weight_packed" in names:
        return base + ".weight_packed", base + ".weight_scale"
    if base + ".weight" in names and base + ".weight_scale" in names:
        return base + ".weight", base + ".weight_scale"
    raise PackFailure(f"{base}: no recognised quantised serialisation "
                      f"(.weight_packed or .weight + .weight_scale)")


def check_scales(name, raw):
    if b"\xff" in raw:
        raise PackFailure(f"{name}: E8M0 0xff (NaN) in the scale plane")
    # Dequantisation is bit-exact in bf16 - power-of-two scales against
    # one-mantissa-bit values - EXCEPT at the exponent ceiling: codes >= 253
    # can push |6 x 2^(code-127)| past bf16's max. Real weight scales sit
    # near 127; a code this large is worth a loud line even though legal.
    high = sum(1 for b in raw if b >= 253)
    if high:
        print(f"ADVISORY {name}: {high} E8M0 codes >= 253; "
              f"bf16 dequant can overflow at this magnitude")


# -- the fold paths -------------------------------------------------------------

def gamma_fold_bf16(proj_raw, gamma_raw, count):
    """proj [1, count] * gamma [count], elementwise, f32 semantics, BF16 RNE
    out. Pure python: count is hidden, the row count is 187, and exactness is
    argued at f32_round - no reason to spend a numpy import here."""
    proj = bf16_raw_to_f32(proj_raw)
    gamma = bf16_raw_to_f32(gamma_raw)
    if len(proj) != count or len(gamma) != count:
        raise PackFailure("gamma fold operand size does not match hidden")
    return f32_list_to_bf16_raw(
        [f32_round(proj[i] * gamma[i]) for i in range(count)])


def q_fold_absorb(q_b_raw, kv_b_raw, heads, nope, rope, v_head, kv_lora,
                  q_lora):
    """The MLA up-projection fold. Returns (q_up_raw, kv_b_value_raw) as BF16
    bytes: kv_b's k_nope half absorbed into q_b per head, and kv_b's value
    half as its own table. numpy when present; the pure-python fallback is
    bit-exact f32 per operation with a sequential accumulation order and is
    for host verification - it is far too slow to ship."""
    q_shape = (heads, nope + rope, q_lora)
    kv_shape = (heads, nope + v_head, kv_lora)
    if np is not None:
        q_b = bf16_to_f32(np.frombuffer(q_b_raw, dtype=np.uint16)
                          .reshape(q_shape))
        kv_b = bf16_to_f32(np.frombuffer(kv_b_raw, dtype=np.uint16)
                           .reshape(kv_shape))
        absorbed = np.einsum("hnl,hnq->hlq", kv_b[:, :nope, :], q_b[:, :nope, :])
        folded = np.concatenate([absorbed, q_b[:, nope:, :]], axis=1)
        return f32_to_bf16(folded.reshape(-1)).tobytes(), \
            f32_to_bf16(kv_b[:, nope:, :].reshape(-1)).tobytes()
    print("ADVISORY numpy not importable: the MLA q-fold is running the "
          "pure-python reference (f32-exact, sequential accumulation; "
          "verification-grade, not ship-grade speed)")
    q_b = bf16_raw_to_f32(q_b_raw)
    kv_b = bf16_raw_to_f32(kv_b_raw)
    q_up = [0.0] * (heads * (kv_lora + rope) * q_lora)
    value = [0.0] * (heads * v_head * kv_lora)
    for h in range(heads):
        q_base = h * (nope + rope) * q_lora
        kv_base = h * (nope + v_head) * kv_lora
        out_base = h * (kv_lora + rope) * q_lora
        for l in range(kv_lora):
            for q in range(q_lora):
                acc = 0.0
                for n in range(nope):
                    acc = f32_round(acc + f32_round(
                        kv_b[kv_base + n * kv_lora + l]
                        * q_b[q_base + n * q_lora + q]))
                q_up[out_base + l * q_lora + q] = acc
        for r in range(rope):
            for q in range(q_lora):
                q_up[out_base + (kv_lora + r) * q_lora + q] = \
                    q_b[q_base + (nope + r) * q_lora + q]
        v_base = h * v_head * kv_lora
        for v in range(v_head):
            for l in range(kv_lora):
                value[v_base + v * kv_lora + l] = \
                    kv_b[kv_base + (nope + v) * kv_lora + l]
    return f32_list_to_bf16_raw(q_up), f32_list_to_bf16_raw(value)


# -- the pack itself ------------------------------------------------------------

class Pack:
    """Sequential payload writer with a side journal: every entry is appended
    to <out>.journal only AFTER its bytes are on disk, so a killed run can
    resume by re-walking the journal, truncating to the last complete tensor,
    and skipping already-emitted tensors (the emission order is deterministic,
    so re-generated entries and offsets are byte-identical)."""
    def __init__(self, out_path, resume=False):
        self.journal_path = str(out_path) + ".journal"
        self.manifest = {}
        self.offset = 0
        if resume and os.path.exists(self.journal_path):
            with open(self.journal_path, "r", encoding="utf-8") as journal:
                for line in journal:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    name = record["name"]
                    self.manifest[name] = record["entry"]
                    self.offset = record["end"]
            self.handle = open(out_path, "r+b")
            self.handle.truncate(self.offset)
            self.handle.seek(0, 2)
        else:
            self.handle = open(out_path, "wb")
            with open(self.journal_path, "w", encoding="utf-8"):
                pass
        self.journal = open(self.journal_path, "a", encoding="utf-8")

    def add(self, name, payload, kind, shape, extra=None):
        if name in self.manifest:
            return
        pad = (-self.offset) % ALIGN
        if pad:
            self.handle.write(b"\0" * pad)
            self.offset += pad
        raw = bytes(payload)
        entry = {"offset": self.offset, "bytes": len(raw), "align": ALIGN,
                 "kind": kind, "shape": list(shape)}
        if extra:
            entry.update(extra)
        self.manifest[name] = entry
        self.handle.write(raw)
        self.offset += len(raw)
        self.journal.write(json.dumps({"name": name, "entry": entry,
                                       "end": self.offset},
                                      separators=(",", ":")) + "\n")
        self.journal.flush()

    def close(self):
        self.journal.close()
        self.handle.close()
        os.unlink(self.journal_path)


def validate_layout(manifest, config):
    """Re-derive every layout identity from the config and hold the emitted
    manifest to it. Runs at the end of pack_model and from the layout test; a
    drift here is a PackFailure, never a pack that binds wrong."""
    hidden = config["hidden"]
    offset = 0
    for name in sorted(manifest, key=lambda n: manifest[n]["offset"]):
        entry = manifest[name]
        if entry["offset"] % ALIGN != 0:
            raise PackFailure(f"{name}: offset {entry['offset']} is not "
                              f"{ALIGN}-byte aligned")
        if entry["offset"] < offset:
            raise PackFailure(f"{name}: overlaps the previous tensor")
        offset = entry["offset"] + entry["bytes"]
        kind = entry["kind"]
        if kind == KIND_MXFP4_INTERLEAVED:
            geom = interleave_geometry(entry["interleave"]["out_dim"],
                                       entry["interleave"]["k_dim"],
                                       entry["interleave"]["experts"])
            if geom["tensor_bytes"] != entry["bytes"]:
                raise PackFailure(f"{name}: {entry['bytes']} bytes, geometry "
                                  f"prices {geom['tensor_bytes']}")
        elif kind == KIND_BF16 and entry["bytes"] % 2 != 0:
            raise PackFailure(f"{name}: odd BF16 byte count")
        if "sections" in entry:
            rows = 0
            for section in entry["sections"]:
                if section["row_offset"] != rows:
                    raise PackFailure(f"{name}: section {section['name']} does "
                                      f"not tile the fused rows")
                rows += section["rows"]
            if rows != entry["shape"][0]:
                raise PackFailure(f"{name}: sections cover {rows} rows, shape "
                                  f"says {entry['shape'][0]}")
            if entry["shape"][1] != hidden:
                raise PackFailure(f"{name}: fused projection is not over "
                                  f"hidden")
            if (hidden * 2) % ALIGN != 0:
                raise PackFailure(f"{name}: a hidden row is {hidden * 2}B, so "
                                  f"section bases cannot stay {ALIGN}B-aligned")


def pack_model(model_dir, out_path, first_layer=0, layer_count=None):
    reader = SafetensorDir(model_dir)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    # The Kimi-K3 checkpoint nests the model config under text_config.
    if isinstance(config.get("text_config"), dict):
        config = config["text_config"]
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    if layer_count is None:
        layer_count = layers
    if first_layer + layer_count > layers or first_layer < 0 or layer_count <= 0:
        raise PackFailure(f"invalid slice {first_layer}+{layer_count} of {layers}")
    experts = config["num_experts"]
    top_k = config.get("num_experts_per_tok", config["num_experts_per_token"])
    latent = config["routed_expert_hidden_size"]
    inter = config["moe_intermediate_size"]
    shared = config.get("num_shared_experts", 1) * inter
    q_lora = config["q_lora_rank"]
    kv_lora = config["kv_lora_rank"]
    rope = config["qk_rope_head_dim"]
    nope = config["qk_nope_head_dim"]
    v_head = config["v_head_dim"]
    heads = config["num_attention_heads"]
    kda_heads = config["linear_attn_config"]["num_heads"] \
        if "linear_attn_config" in config else config["num_attention_heads"]
    kda_head = config["linear_attn_config"]["head_dim"] \
        if "linear_attn_config" in config else config["head_dim"]
    kda_dim = kda_heads * kda_head
    kernel = config["linear_attn_config"]["short_conv_kernel_size"] \
        if "linear_attn_config" in config else 4
    if "layer_types" in config:
        types = config["layer_types"]
    else:
        # Kimi-K3 names the layer map inside linear_attn_config as two
        # one-indexed lists; everything else defaults to the 3:1 period.
        lac = config.get("linear_attn_config", {})
        types = ["full_attention"] * layers
        for i in lac.get("kda_layers", []):
            if 1 <= i <= layers:
                types[i - 1] = "linear_attention"
    if len(types) != layers:
        raise PackFailure("layer_types does not cover num_hidden_layers")
    # The interleave grid prices both expert GEMMs up front; a checkpoint the
    # grid does not divide fails before a byte is written.
    w1_geom = interleave_geometry(2 * inter, latent, experts)
    w2_geom = interleave_geometry(latent, inter, experts)

    payload_path = Path(str(out_path) + ".payload")
    pack = Pack(payload_path, resume=payload_path.exists())
    L = "model.layers.{}."
    SL = "language_model.model.layers.{}."

    def bf(dst, src, shape=None):
        if dst in pack.manifest:
            return
        pack.add(dst, reader.bf16(src, shape), KIND_BF16,
                 shape if shape is not None else [0])

    # model level, in consumption order: the embedding first, the closing
    # norm, output retrieval and head last. The embedding rides stage zero
    # only; the closing globals ride the last stage only.
    if first_layer == 0:
        bf("model.embed_tokens.weight", "language_model.model.embed_tokens.weight",
           (config["vocab_size"], hidden))

    for layer in range(first_layer, first_layer + layer_count):
        p = L.format(layer)
        sp = SL.format(layer)
        linear = types[layer] == "linear_attention"
        bf(p + "attn_norm_weight", sp + "input_layernorm.weight", (hidden,))
        g = reader.bf16(sp + "self_attention_res_norm.weight", (hidden,))
        w = reader.bf16(sp + "self_attention_res_proj.weight", (1, hidden))
        pack.add(p + "attnres_attn_weight", gamma_fold_bf16(w, g, hidden),
                 KIND_BF16, [1, hidden])
        if linear:
            a = sp + "self_attn."
            # THE FUSED WIDE TENSOR, OUTPUT_DIM_HEADS CLASS. q|k|v|beta as
            # one [sum_out, hidden] BF16 tensor: one GEMM over normed_bf16
            # replaces four launches, and the section table in the manifest
            # is the split contract.
            if p + "kda_qkv_beta_weight" in pack.manifest:
                fused = b""
                sections, rows = [], 0
            else:
                sections, rows = kda_fused_qkvb_sections(kda_heads, kda_head,
                                                         kda_head)
                fused = b"".join((
                reader.bf16(a + "q_proj.weight", (kda_dim, hidden)),
                reader.bf16(a + "k_proj.weight", (kda_dim, hidden)),
                reader.bf16(a + "v_proj.weight", (kda_dim, hidden)),
                reader.bf16(a + "b_proj.weight", (kda_heads, hidden))))
            pack.add(p + "kda_qkv_beta_weight", fused, KIND_BF16,
                     [rows, hidden], {"sections": sections,
                                      "shard_class": "output_dim_heads"})
            for conv in "qkv":
                raw = reader.f32(a + f"{conv}_conv1d.weight",
                                 (kda_dim, 1, kernel))
                pack.add(p + f"kda_{conv}_conv_weight", raw, KIND_F32,
                         [kda_dim, kernel])
            # RELEASED CHECKPOINT (full_rank_output_gate): decay_down is the
            # standalone 128-wide replicated bottleneck and the gate is the
            # checkpoint's full-rank g_proj, unchanged. The old low-rank
            # g_a/g_b pair and the decay|gate fusion do not exist in this
            # checkpoint (docs/K3_GATE_RECONCILIATION.md).
            pack.add(p + "kda_decay_down_weight",
                     reader.bf16(a + "f_a_proj.weight", (kda_head, hidden)),
                     KIND_BF16, [kda_head, hidden],
                     {"shard_class": "replicated"})
            bf(p + "kda_decay_up_weight", a + "f_b_proj.weight",
               (kda_dim, kda_head))
            pack.add(p + "kda_decay_bias", reader.f32(a + "dt_bias", (kda_dim,)),
                     KIND_F32, [kda_dim])
            head_log_scale = reader.f32(a + "A_log", (A_LOG_SOURCE_HEADS,))
            pack.add(p + "kda_head_log_scale", head_log_scale[:kda_heads * 4],
                     KIND_F32, [kda_heads])
            pack.add(p + "kda_gate_weight",
                     reader.bf16(a + "g_proj.weight", (kda_dim, hidden)),
                     KIND_BF16, [kda_dim, hidden],
                     {"shard_class": "output_dim_heads"})
            pack.add(p + "kda_out_norm_weight",
                     reader.f32(a + "o_norm.weight", (kda_head,)),
                     KIND_F32, [kda_head])
            bf(p + "kda_out_weight", a + "o_proj.weight", (hidden, kda_dim))
        else:
            a = sp + "self_attn."
            bf(p + "mla_q_down_weight", a + "q_a_proj.weight", (q_lora, hidden))
            bf(p + "mla_q_norm_weight", a + "q_a_layernorm.weight", (q_lora,))
            bf(p + "mla_kv_a_weight", a + "kv_a_proj_with_mqa.weight",
               (kv_lora + rope, hidden))
            bf(p + "mla_kv_a_norm_weight", a + "kv_a_layernorm.weight", (kv_lora,))
            q_up, value = q_fold_absorb(
                reader.bf16(a + "q_b_proj.weight",
                            (heads * (nope + rope), q_lora)),
                reader.bf16(a + "kv_b_proj.weight",
                            (heads * (nope + v_head), kv_lora)),
                heads, nope, rope, v_head, kv_lora, q_lora)
            pack.add(p + "mla_q_up_weight", q_up, KIND_BF16,
                     [heads * (kv_lora + rope), q_lora])
            pack.add(p + "mla_kv_b_value_weight", value, KIND_BF16,
                     [heads * v_head, kv_lora])
            pack.add(p + "mla_gate_weight",
                     reader.bf16(a + "g_proj.weight", (heads * v_head, hidden)),
                     KIND_BF16, [heads * v_head, hidden],
                     {"shard_class": "output_dim_heads"})
            bf(p + "mla_out_weight", a + "o_proj.weight", (hidden, heads * v_head))
        bf(p + "mlp_norm_weight", sp + "post_attention_layernorm.weight", (hidden,))
        g = reader.bf16(sp + "mlp_res_norm.weight", (hidden,))
        w = reader.bf16(sp + "mlp_res_proj.weight", (1, hidden))
        pack.add(p + "attnres_mlp_weight", gamma_fold_bf16(w, g, hidden),
                 KIND_BF16, [1, hidden])
        # Routed layers ship their MoE under block_sparse_moe; the dense
        # replacement layer keeps the mlp.gate_proj naming.
        dense_m = sp + "mlp."
        m = sp + "block_sparse_moe."
        if m + "gate.weight" not in reader.names():
            # the dense layer: one MLP, no router, no experts
            w1 = reader.bf16(dense_m + "gate_proj.weight")
            w3 = reader.bf16(dense_m + "up_proj.weight")
            dense_inter = len(w1) // (hidden * 2)
            pack.add(p + "dense_gate_up_weight", w1 + w3, KIND_BF16,
                     [2 * dense_inter, hidden])
            down = reader.bf16(dense_m + "down_proj.weight")
            pack.add(p + "dense_down_weight", down, KIND_BF16,
                     [hidden, len(down) // (hidden * 2)])
            continue
        bf(p + "router_weight", m + "gate.weight", (experts, hidden))
        dtype, shape, raw = reader.tensor(m + "gate.e_score_correction_bias")
        if dtype != "F32" and dtype != "BF16":
            raise PackFailure(f"{m}gate.e_score_correction_bias: unexpected "
                              f"{dtype}")
        pack.add(p + "router_bias", raw,
                 KIND_F32 if dtype == "F32" else KIND_BF16, [experts])
        # THE INTERLEAVED EXPERT TENSORS. Payload and E8M0 scales become ONE
        # tensor each: the manifest names keep their V1 meaning (w1 = gate|up
        # concat, w2 = down) but there is no separate scale plane to bind,
        # shard or prefetch. Geometry was validated before the layer loop.
        w1_pay, w1_sc, w2_pay, w2_sc = [], [], [], []
        experts_done = p + "expert_w1_weight" in pack.manifest
        for e in range(0 if experts_done else experts):
            base = m + f"experts.{e}."
            g_name, g_scale = quant_pair(reader, base + "w1")
            u_name, u_scale = quant_pair(reader, base + "w3")
            d_name, d_scale = quant_pair(reader, base + "w2")
            g = reader.u8(g_name, (inter, latent // 2))
            u = reader.u8(u_name, (inter, latent // 2))
            d = reader.u8(d_name, (latent, inter // 2))
            gs = reader.u8(g_scale, (inter, latent // GROUP))
            us = reader.u8(u_scale, (inter, latent // GROUP))
            ds = reader.u8(d_scale, (latent, inter // GROUP))
            for name, sc in ((g_scale, gs), (u_scale, us), (d_scale, ds)):
                check_scales(name, sc)
            w1_pay.append(g + u)
            w1_sc.append(gs + us)
            w2_pay.append(d)
            w2_sc.append(ds)
        pack.add(p + "expert_w1_weight",
                 interleave(b"".join(w1_pay), b"".join(w1_sc), w1_geom),
                 KIND_MXFP4_INTERLEAVED, [experts, 2 * inter, latent],
                 {"interleave": w1_geom, "shard_class": "concat_output"})
        pack.add(p + "expert_w2_weight",
                 interleave(b"".join(w2_pay), b"".join(w2_sc), w2_geom),
                 KIND_MXFP4_INTERLEAVED, [experts, latent, inter],
                 {"interleave": w2_geom, "shard_class": "input_dim"})
        s1 = reader.bf16(m + "shared_experts.gate_proj.weight", (shared, hidden))
        s3 = reader.bf16(m + "shared_experts.up_proj.weight", (shared, hidden))
        pack.add(p + "shared_w1_weight", s1 + s3, KIND_BF16, [2 * shared, hidden])
        bf(p + "shared_w2_weight", m + "shared_experts.down_proj.weight",
           (hidden, shared))
        bf(p + "routed_down_weight", m + "routed_expert_down_proj.weight",
           (latent, hidden))
        bf(p + "routed_up_weight", m + "routed_expert_up_proj.weight",
           (hidden, latent))
        bf(p + "routed_norm_weight", m + "routed_expert_norm.weight", (latent,))

    if first_layer + layer_count == layers:
        bf("model.norm.weight", "language_model.model.norm.weight", (hidden,))
        gamma = reader.bf16("language_model.model.output_attn_res_norm.weight", (hidden,))
        proj = reader.bf16("language_model.model.output_attn_res_proj.weight", (1, hidden))
        pack.add("model.attnres_out_weight", gamma_fold_bf16(proj, gamma, hidden),
                 KIND_BF16, [1, hidden])
        bf("lm_head.weight", "language_model.lm_head.weight", (config["vocab_size"], hidden))

    validate_layout(pack.manifest, {"hidden": hidden})

    pack.close()
    echo = {"hidden": hidden, "layers": layer_count, "first_layer": first_layer,
            "total_layers": layers, "experts": experts,
            "top_k": top_k, "latent": latent, "intermediate": inter,
            "group": GROUP, "vocab": config["vocab_size"],
            "kda_heads": kda_heads, "kda_head": kda_head, "heads": heads,
            "kv_lora": kv_lora, "rope": rope, "v_head": v_head,
            "nope": nope, "shared": shared, "q_lora": q_lora}
    fmt = {"version": VERSION, "alignment": ALIGN,
           "mxfp4_interleave": {"tile_k": TILE_K, "group": GROUP,
                                "stored_bits": 4,
                                "cell_payload_rows": CELL_PAYLOAD_ROWS,
                                "cell_rows": CELL_PAYLOAD_ROWS + 1,
                                "row_bytes": TILE_K * 4 // 8,
                                "scale_bytes_per_neuron_tile": TILE_K // GROUP},
           "kda_fused": {"qkvb_sections": ["q", "k", "v", "beta"],
                         "decay_gate_down_sections": ["decay_down",
                                                      "gate_down"]}}
    manifest = json.dumps({"format": fmt, "config": echo,
                           "tensors": pack.manifest},
                          separators=(",", ":")).encode()
    with open(out_path, "wb") as out:
        out.write(struct.pack("<IIQ", MAGIC, VERSION, len(manifest)))
        out.write(manifest)
        pad = (-out.tell()) % ALIGN
        out.write(b"\0" * pad)
        with open(payload_path, "rb") as body:
            while True:
                chunk = body.read(1 << 24)
                if not chunk:
                    break
                out.write(chunk)
    payload_path.unlink()
    return echo, pack.manifest


def main():
    if len(sys.argv) not in (3, 5):
        print("usage: k3_pack.py <checkpoint_dir> <out.pack> [first_layer layer_count]")
        return 2
    try:
        if len(sys.argv) == 5:
            echo, manifest = pack_model(sys.argv[1], sys.argv[2],
                                        int(sys.argv[3]), int(sys.argv[4]))
        else:
            echo, manifest = pack_model(sys.argv[1], sys.argv[2])
    except PackFailure as failure:
        print(f"PACK FAILURE: {failure}")
        return 1
    print(f"packed {len(manifest)} tensors, "
          f"{sum(t['bytes'] for t in manifest.values())} payload bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
