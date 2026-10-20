#!/usr/bin/env python3
"""Decode csf_32x2_v1 compressed FP4 block scales to their exact scale bytes.

The format is the one documented as csf_32x2_v1 by Local Inference Lab
(docs/csf-scale-blocks.md in github.com/local-inference-lab/flashinfer,
Apache-2.0): a logical scale matrix of R rows (a multiple of 32) and C
columns is stored as

  .csf32_bases       uint8  [R]       one palette base per row
  .csf32_codes       uint8  [R*C*b/8] b-bit offsets (b = 4 for NVFP4, 1 for MXFP4)
  .csf32_page_bases  uint32 [P]       byte origin of each page in the exception arena
  .csf32_offsets     uint16 [B+P]     page-relative record prefixes, one terminal per page
  .csf32_exceptions  uint16 [E]       low byte 2*local_row+local_column, high byte the scale byte

Blocks cover 32 rows by two columns (a final odd column forms a one-column
block) and are visited in 128-row by four-column tiles in row-major order,
the first column pair's row blocks before the second's. Offsets inside a
block are row-major, least-significant field first. Pages hold 32 whole tile
groups. A scale byte is its row base plus its offset, unless an exception
replaces it.

subcommands:
  check CHECKPOINT_DIR [--layers L,...] [--experts E,...] [--reference BF16_DIR]
      decode the named experts' gate/up/down scales, validate every structural
      rule, and with --reference report the cosine between the dequantized
      NVFP4 weight and the BF16 reference weight of the same expert
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

BLOCK_ROWS = 32
TILE_BLOCK_ROWS = 4
TILE_COLUMN_PAIRS = 2
PAGE_GROUPS = 32
COMPONENTS = ("bases", "codes", "page_bases", "offsets", "exceptions")


def block_order(rows, columns):
    if rows <= 0 or rows % BLOCK_ROWS or columns <= 0:
        raise ValueError("csf scales need a positive multiple of 32 rows and positive columns")
    row_blocks, pair_blocks = rows // BLOCK_ROWS, (columns + 1) // 2
    order, groups = [], [0]
    for r in range(0, row_blocks, TILE_BLOCK_ROWS):
        for c in range(0, pair_blocks, TILE_COLUMN_PAIRS):
            for pair in range(c, min(c + TILE_COLUMN_PAIRS, pair_blocks)):
                for block_row in range(r, min(r + TILE_BLOCK_ROWS, row_blocks)):
                    order.append((block_row, pair))
            groups.append(len(order))
    return order, groups


def decode(components, rows, columns, bits=4):
    bases = np.asarray(components["bases"], dtype=np.uint8).reshape(-1)
    codes = np.asarray(components["codes"], dtype=np.uint8).reshape(-1)
    page_bases = np.asarray(components["page_bases"]).view(np.uint32).reshape(-1)
    offsets = np.asarray(components["offsets"]).view(np.uint16).reshape(-1)
    exceptions = np.asarray(components["exceptions"]).view(np.uint16).reshape(-1)
    order, groups = block_order(rows, columns)
    blocks = len(order)
    page_starts = groups[::PAGE_GROUPS]
    if page_starts[-1] != blocks:
        page_starts = page_starts + [blocks]
    pages = len(page_starts) - 1
    if bases.shape != (rows,) or codes.shape != (rows * columns * bits // 8,):
        raise ValueError("csf bases or codes have the wrong size")
    if page_bases.shape != (pages,) or offsets.shape != (blocks + pages,):
        raise ValueError("csf page bases or offsets have the wrong size")
    if np.any(bases.astype(np.int32) > 256 - (1 << bits)):
        raise ValueError("csf row base leaves the scale byte range")
    fields = np.unpackbits(codes, bitorder="little").reshape(-1, bits)
    fields = (fields.astype(np.uint16) << np.arange(bits, dtype=np.uint16)).sum(axis=1).astype(np.uint8)
    out = np.empty((rows, columns), dtype=np.uint8)
    cursor = 0
    record = 0
    for page in range(pages):
        if offsets[page_starts[page] + page] != 0 or page_bases[page] != 2 * record:
            raise ValueError("csf page %d does not continue the exception arena" % page)
        for block in range(page_starts[page], page_starts[page + 1]):
            block_row, pair = order[block]
            width = min(2, columns - 2 * pair)
            count = 32 * width
            row0, col0 = block_row * BLOCK_ROWS, 2 * pair
            local = fields[cursor:cursor + count].reshape(32, width)
            cursor += count
            values = bases[row0:row0 + 32, None].astype(np.int32) + local
            first = offsets[block + page]
            last = offsets[block + page + 1]
            if last < first or last - first > count:
                raise ValueError("csf block %d has an impossible exception count" % block)
            previous = -1
            for slot in range(record + first, record + last):
                position, byte = int(exceptions[slot] & 255), int(exceptions[slot] >> 8)
                if position <= previous or position >= 64 or position % 2 >= width:
                    raise ValueError("csf block %d exception positions are not sorted, unique and in bounds" % block)
                previous = position
                values[position // 2, position % 2] = byte
            out[row0:row0 + 32, col0:col0 + width] = values
        record += int(offsets[page_starts[page + 1] + page])
    if cursor != fields.size or record != exceptions.size:
        raise ValueError("csf codes or exceptions were not consumed exactly")
    return out


def safetensors_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


class Checkpoint:
    def __init__(self, directory):
        self.directory = directory
        self.weight_map = json.load(open(os.path.join(directory, "model.safetensors.index.json")))["weight_map"]
        self.headers = {}

    def has(self, name):
        return name in self.weight_map

    def tensor(self, name):
        shard = self.weight_map[name]
        if shard not in self.headers:
            self.headers[shard] = safetensors_header(os.path.join(self.directory, shard))
        header, data_start = self.headers[shard]
        entry = header[name]
        begin, end = entry["data_offsets"]
        dtype = {"U8": np.uint8, "I8": np.int8, "U16": np.uint16, "I16": np.int16, "U32": np.uint32, "I32": np.int32,
                 "F32": np.float32, "BF16": np.uint16, "F8_E4M3": np.uint8, "F16": np.float16}[entry["dtype"]]
        with open(os.path.join(self.directory, shard), "rb") as f:
            f.seek(data_start + begin)
            raw = f.read(end - begin)
        return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"]), entry["dtype"]


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def e4m3_to_float(byte):
    byte = byte.astype(np.int32)
    sign = np.where(byte & 0x80, -1.0, 1.0)
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    normal = (1.0 + mantissa / 8.0) * np.exp2(exponent - 7.0)
    subnormal = (mantissa / 8.0) * np.exp2(-6.0)
    value = np.where(exponent == 0, subnormal, normal) * sign
    return np.where((exponent == 15) & (mantissa == 7), np.nan, value).astype(np.float32)


def bf16_to_float(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def check(arguments):
    checkpoint = Checkpoint(arguments.checkpoint)
    reference = Checkpoint(arguments.reference) if arguments.reference else None
    layers = [int(x) for x in arguments.layers.split(",")]
    experts = [int(x) for x in arguments.experts.split(",")]
    prefix = arguments.prefix
    worst = 1.0
    for layer in layers:
        for expert in experts:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                base = "%s.layers.%d.mlp.experts.%d.%s" % (prefix, layer, expert, projection)
                weight, _ = checkpoint.tensor(base + ".weight")
                rows, columns = weight.shape[0], weight.shape[1] * 2 // 16
                components = {name: checkpoint.tensor(base + ".weight_scale.csf32_" + name)[0] for name in COMPONENTS}
                scales = decode(components, rows, columns, 4)
                line = "layer %d expert %d %s: %d x %d scales decoded, %d exceptions" % (
                    layer, expert, projection, rows, columns, components["exceptions"].size)
                if reference is not None:
                    reference_name = arguments.reference_prefix + ".layers.%d.mlp.experts.%d.%s.weight" % (layer, expert, projection)
                    raw, dtype = reference.tensor(reference_name)
                    truth = bf16_to_float(raw) if dtype == "BF16" else raw.astype(np.float32)
                    global_scale = float(checkpoint.tensor(base + ".weight_scale_2")[0].reshape(-1)[0])
                    nibbles = np.stack((weight & 15, weight >> 4), axis=-1).reshape(rows, -1)
                    values = E2M1[nibbles] * np.repeat(e4m3_to_float(scales), 16, axis=1) * global_scale
                    cosine = float(np.dot(values.ravel(), truth.ravel()) / (np.linalg.norm(values) * np.linalg.norm(truth) + 1e-30))
                    worst = min(worst, cosine)
                    line += ", cosine to the BF16 reference %.5f (global scale %.3g)" % (cosine, global_scale)
                print(line, flush=True)
    if reference is not None:
        print("worst cosine %.5f" % worst)
        if worst < arguments.min_cosine:
            raise SystemExit("FAIL: a decoded expert does not match its BF16 reference")
    print("PASS")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("checkpoint")
    c.add_argument("--layers", default="3,20,44")
    c.add_argument("--experts", default="0,100,287")
    c.add_argument("--prefix", default="model.language_model")
    c.add_argument("--reference")
    c.add_argument("--reference-prefix", default="model.language_model")
    c.add_argument("--min-cosine", type=float, default=0.99)
    arguments = parser.parse_args()
    if arguments.command == "check":
        check(arguments)


if __name__ == "__main__":
    main()
