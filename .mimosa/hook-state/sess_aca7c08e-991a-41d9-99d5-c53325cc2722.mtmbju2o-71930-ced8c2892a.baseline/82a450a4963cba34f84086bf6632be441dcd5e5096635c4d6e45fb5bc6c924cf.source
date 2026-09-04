#!/usr/bin/env python3
"""Convert DSV4 Pro routed-expert weights between quantized codecs at the pack
level. Only mxfp4_e2m1 -> fp8_e4m3 is implemented today; the default pack and
module codec remain MXFP4-E2M1.

The source layout is the pack's FP4 expert records (384 stacked experts,
E2M1 payload pairs, one E8M0 scale per 32 columns). The target is the pack's
FP8-E4M3 convention: one byte per element plus one F32 scale per 128 columns
per row (same convention as the non-expert FP8 linears).

The conversion is lossy by construction (the checkpoint was trained into the
MXFP4 grid), so a converted pack is a VARIANT, never the token-exact baseline.
Running it needs a module built with the FP8 expert kernel variant and an
FP8-expert pack (header expert codec id 5).

Streams per row, so memory stays bounded; the full 865 GB pack converts in
roughly an hour, a 4-layer validation slice in ~15 minutes.

usage: dsv4_pro_expert_requant.py --input PACK --output PACK --to fp8_e4m3
       dsv4_pro_expert_requant.py --input PACK --inspect
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Optional

import numpy as np

MAGIC = 0x34565344
FORMAT_VERSION = 3
HEADER = struct.Struct("<16I2Q")
ENTRY = struct.Struct("<6I2Q")

WEIGHT_FP4 = 3
WEIGHT_FP8 = 4
KIND_EXPERTS_W1 = 19
KIND_EXPERTS_W2 = 20
KIND_EXPERTS_W3 = 21
EXPERT_KINDS = (KIND_EXPERTS_W1, KIND_EXPERTS_W2, KIND_EXPERTS_W3)

CODEC_MXFP4_E2M1 = 7
CODEC_FP8_E4M3 = 5

# E2M1 magnitudes: sign bit, two exponent bits, one mantissa bit; the
# representable magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6 (matching
# inference/kernels/dtype.cuh LmE2m1PairToFloat).
E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           dtype=np.float32)


class PackFailure(RuntimeError):
    pass


def fp4_nibbles(payload: np.ndarray) -> np.ndarray:
    """Unpack FP4 payload bytes into float32 elements (low nibble first).

    Nibble layout: [sign][e1][e0][m]; the magnitude index is the low 3 bits.
    """
    low = payload & 0x0F
    high = (payload >> 4) & 0x0F
    values = np.empty(payload.shape[:-1] + (payload.shape[-1] * 2,),
                      dtype=np.float32)
    values[..., 0::2] = E2M1_MAGNITUDES[low & 0x07]
    values[..., 1::2] = E2M1_MAGNITUDES[high & 0x07]
    values[..., 0::2] = np.where((low & 0x08) != 0,
                                 -values[..., 0::2], values[..., 0::2])
    values[..., 1::2] = np.where((high & 0x08) != 0,
                                 -values[..., 1::2], values[..., 1::2])
    return values


def apply_e8m0_scales(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Multiply FP4-decoded values by their E8M0 block scales (one per 32)."""
    repeats = values.shape[-1] // scales.shape[-1]
    with np.errstate(over="ignore", invalid="ignore"):
        scale_f = np.power(2.0, scales.astype(np.float32) - 127.0)
    # Guard against extreme/NaN scale codes (0xff is the NaN encoding; the
    # checkpoint grid keeps scales in a sane range, this keeps the math finite).
    scale_f = np.minimum(scale_f, np.float32(2.0 ** 64))
    return values * np.repeat(scale_f, repeats, axis=-1)


def quantize_e4m3(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a row block to E4M3 with one F32 scale per 128 columns."""
    count = values.size
    blocks = count // 128
    shaped = values.reshape(blocks, 128)
    maxima = np.max(np.abs(shaped), axis=1)
    scales = np.where(maxima > 0.0, maxima / 448.0, 1.0).astype(np.float32)
    q = shaped / scales[:, None]
    q = np.rint(q)
    q = np.clip(q, -448.0, 448.0)
    codes = encode_e4m3(q)
    return codes.reshape(-1), scales


def encode_e4m3(q: np.ndarray) -> np.ndarray:
    """Encode rounded integers in [-448, 448] to OCP E4M3 bytes."""
    q = q.astype(np.int32)
    sign = np.where(q < 0, 0x80, 0x00).astype(np.uint8)
    mag = np.abs(q)
    code = np.zeros_like(mag, dtype=np.int32)
    # e=15 special: exactly 448; 241..447 round to the nearest of {240, 448}.
    big = mag >= 241
    code[big] = np.where(mag[big] <= 344, (14 << 3) | 7, (15 << 3) | 0)
    small = ~big & (mag > 0)
    m = mag[small]
    e = np.floor(np.log2(m.astype(np.float64))).astype(np.int32) + 7
    base = np.left_shift(1, e - 7)
    k = np.rint((m.astype(np.float64) - base) / (base / 8.0)).astype(np.int32)
    carry = k >= 8
    e[carry] += 1
    k[carry] = 0
    e = np.clip(e, 7, 14)
    k = np.clip(k, 0, 7)
    code[small] = (e << 3) | k
    return (code.astype(np.uint8)) | sign


def convert_record_bytes(payload: bytes, scales: bytes, rows: int,
                         columns: int) -> tuple[bytes, bytes]:
    payload_np = np.frombuffer(payload, dtype=np.uint8)
    scale_np = np.frombuffer(scales, dtype=np.uint8)
    row_payload_bytes = columns // 2
    row_scale_bytes = columns // 32
    blocks_per_row = columns // 128
    chunk_rows = 64
    out_payload = np.empty(rows * columns, dtype=np.uint8)
    out_scales = np.empty(rows * blocks_per_row, dtype=np.float32)
    for start in range(0, rows, chunk_rows):
        chunk = min(chunk_rows, rows - start)
        p = payload_np[start * row_payload_bytes:
                       (start + chunk) * row_payload_bytes]
        s = scale_np[start * row_scale_bytes:
                     (start + chunk) * row_scale_bytes]
        p = p.reshape(chunk, row_payload_bytes)
        s = s.reshape(chunk, row_scale_bytes)
        values = fp4_nibbles(p)
        values = apply_e8m0_scales(values, s)
        codes, scales_out = quantize_e4m3(values)
        out_payload[start * columns:(start + chunk) * columns] = codes
        out_scales[start * blocks_per_row:
                   (start + chunk) * blocks_per_row] = scales_out
    return out_payload.tobytes(), out_scales.tobytes()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--to", choices=("fp8_e4m3",), default="fp8_e4m3")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args(argv)

    source = args.input
    if args.inspect:
        with source.open("rb") as f:
            header = HEADER.unpack(f.read(HEADER.size))
            entries = [ENTRY.unpack(f.read(ENTRY.size))
                       for _ in range(header[8])]
        print("magic=%08x version=%u tensors=%u first=%u+%u codecs=(%u,%u,%u)"
              % (header[0], header[1], header[8], header[9], header[10],
                 header[5], header[6], header[7]))
        for e in entries:
            if e[0] in EXPERT_KINDS:
                print("  expert kind=%u layer=%u weight=%u rows=%u cols=%u "
                      "payload=%u scale=%u" % (e[0], e[1], e[2], e[3], e[4],
                                               e[6], e[7]))
        return 0

    if args.output is None:
        parser.error("--output is required without --inspect")
    with source.open("rb") as f:
        header = list(HEADER.unpack(f.read(HEADER.size)))
        if header[0] != MAGIC or header[1] != FORMAT_VERSION:
            raise PackFailure("input is not a DSV4 stage pack")
        entries = [list(ENTRY.unpack(f.read(ENTRY.size)))
                   for _ in range(header[8])]
        if header[6] != CODEC_MXFP4_E2M1:
            raise PackFailure("input expert codec is not mxfp4_e2m1")
        converted = 0
        new_entries = []
        cursor = HEADER.size + ENTRY.size * len(entries)
        with args.output.open("wb") as out:
            out.seek(cursor)
            for entry in entries:
                kind, layer, weight, rows, cols, reserved, payload, scale = entry
                if kind in EXPERT_KINDS and weight == WEIGHT_FP4:
                    f.seek(payload)
                    payload_bytes = f.read(rows * (cols // 2))
                    f.seek(scale)
                    scale_bytes = f.read(rows * (cols // 32))
                    new_payload, new_scales = convert_record_bytes(
                        payload_bytes, scale_bytes, rows, cols)
                    payload_offset = cursor
                    cursor += len(new_payload)
                    scale_offset = cursor
                    cursor += len(new_scales)
                    out.write(new_payload)
                    out.write(new_scales)
                    new_entries.append((kind, layer, WEIGHT_FP8, rows, cols,
                                        reserved, payload_offset, scale_offset))
                    converted += 1
                else:
                    payload_size, scale_size = passthrough_sizes(weight, rows,
                                                                 cols)
                    f.seek(payload)
                    payload_bytes = f.read(payload_size)
                    scale_bytes = b""
                    if scale:
                        f.seek(scale)
                        scale_bytes = f.read(scale_size)
                    payload_offset = cursor
                    cursor += payload_size
                    scale_offset = cursor if scale_size else 0
                    cursor += scale_size
                    out.write(payload_bytes)
                    out.write(scale_bytes)
                    new_entries.append((kind, layer, weight, rows, cols,
                                        reserved, payload_offset, scale_offset))
            header[6] = CODEC_FP8_E4M3
            header[17] = cursor
            out.seek(0)
            out.write(HEADER.pack(*header))
            for entry in new_entries:
                out.write(ENTRY.pack(*entry))
    print({
        "input": str(source),
        "output": str(args.output),
        "expert_codec_from": "mxfp4_e2m1",
        "expert_codec_to": "fp8_e4m3",
        "expert_records_converted": converted,
        "bytes": cursor,
    })
    return 0


def passthrough_sizes(weight: int, rows: int, columns: int) -> tuple[int, int]:
    """Payload and scale byte sizes for non-converted records."""
    if weight == 0:      # BF16
        return rows * columns * 2, 0
    if weight in (1, 2):  # F32, U32
        return rows * columns * 4, 0
    if weight == 3:      # FP4
        return rows * (columns // 2), rows * ((columns + 31) // 32)
    if weight == 4:      # FP8
        return rows * columns, rows * ((columns + 127) // 128) * 4
    raise PackFailure("unknown weight format %u" % weight)


if __name__ == "__main__":
    raise SystemExit(main())
