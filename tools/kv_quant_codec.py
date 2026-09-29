#!/usr/bin/env python3
"""Reference codec for quantized KV rows: encode to the stored bytes, decode back to BF16.

A stored row holds one element code per value (FP8 E4M3: one byte; MXFP4 E2M1:
two nibbles per byte, low nibble first) and one E8M0 scale byte per group. The
scale is the smallest power of two 2^k, k >= -127, with amax <= format_max * 2^k,
so no element saturates. Element codes round to nearest, ties to the even code.
A group holding a NaN or an infinity stores the E8M0 NaN byte 0xff and decodes
to NaN. Decoding multiplies the element value by 2^k and rounds to BF16
(nearest, ties to even).
"""

import numpy as np

BF16 = "bf16"
FP8_E4M3 = "fp8_e4m3"
MXFP4 = "mxfp4"
CODECS = (BF16, FP8_E4M3, MXFP4)
GROUPS = {BF16: (0,), FP8_E4M3: (64, 128), MXFP4: (32,)}
SCALE_EXPONENT_MIN = -127
SCALE_NAN = 0xFF
BF16_NAN = 0x7FC0


def _e4m3_magnitudes():
    values = []
    for code in range(0x7F):
        exponent, mantissa = code >> 3, code & 7
        if exponent == 0:
            values.append(mantissa * 2.0 ** -9)
        else:
            values.append((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return np.array(values, dtype=np.float64)


E4M3_MAGNITUDES = _e4m3_magnitudes()
E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
FORMAT = {
    FP8_E4M3: (E4M3_MAGNITUDES, 0x80, 448.0),
    MXFP4: (E2M1_MAGNITUDES, 0x8, 6.0),
}


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def check_row(codec, group, width):
    if codec not in CODECS:
        raise ValueError(f"unknown KV codec {codec!r}; choose {CODECS}")
    if group not in GROUPS[codec]:
        raise ValueError(f"codec {codec} takes group {GROUPS[codec]}, got {group}")
    if width <= 0 or (group and width % group):
        raise ValueError(f"row width {width} is not a whole number of {group}-element groups")


def scale_exponent(amax, format_max):
    if amax == 0.0:
        return SCALE_EXPONENT_MIN
    exponent = int(np.ceil(np.log2(amax / format_max)))
    while amax > format_max * 2.0 ** exponent:
        exponent += 1
    while amax <= format_max * 2.0 ** (exponent - 1):
        exponent -= 1
    return max(exponent, SCALE_EXPONENT_MIN)


def _nearest_codes(magnitudes, table):
    upper = np.clip(np.searchsorted(table, magnitudes, side="left"), 0, len(table) - 1)
    lower = np.clip(upper - 1, 0, len(table) - 1)
    below = magnitudes - table[lower]
    above = table[upper] - magnitudes
    choose_upper = (above < below) | ((above == below) & (upper % 2 == 0))
    return np.where(choose_upper, upper, lower)


def encode(rows_bf16, codec, group):
    rows_bf16 = np.asarray(rows_bf16, dtype=np.uint16)
    rows, width = rows_bf16.shape
    check_row(codec, group, width)
    if codec == BF16:
        raise ValueError("bf16 rows are stored as they are; there is nothing to encode")
    table, sign_bit, format_max = FORMAT[codec]
    values = bf16_to_f32(rows_bf16).astype(np.float64).reshape(rows, width // group, group)
    codes = np.zeros(values.shape, dtype=np.uint8)
    scales = np.zeros(values.shape[:2], dtype=np.uint8)
    for row in range(rows):
        for index in range(values.shape[1]):
            block = values[row, index]
            if not np.all(np.isfinite(block)):
                scales[row, index] = SCALE_NAN
                continue
            exponent = scale_exponent(float(np.max(np.abs(block))), format_max)
            scaled = block * 2.0 ** -exponent
            magnitude_codes = _nearest_codes(np.abs(scaled), table).astype(np.uint8)
            codes[row, index] = magnitude_codes | np.where(np.signbit(scaled), sign_bit, 0).astype(np.uint8)
            scales[row, index] = exponent + 127
    return codes.reshape(rows, width), scales


def decode(codes, scales, codec, group):
    codes = np.asarray(codes, dtype=np.uint8)
    rows, width = codes.shape
    check_row(codec, group, width)
    table, sign_bit, _ = FORMAT[codec]
    magnitude = table[codes & (sign_bit - 1)]
    signed = np.where(codes & sign_bit, -magnitude, magnitude)
    exponent = np.repeat(np.asarray(scales, dtype=np.int64), group, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        values = signed * np.exp2((exponent - 127).astype(np.float64))
        result = f32_to_bf16(values.astype(np.float32))
    return np.where(exponent == SCALE_NAN, np.uint16(BF16_NAN), result).astype(np.uint16)


def pack_store(codes, scales, codec):
    codes = np.asarray(codes, dtype=np.uint8)
    if codec == MXFP4:
        payload = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    else:
        payload = codes
    return np.concatenate((payload, np.asarray(scales, dtype=np.uint8)), axis=1)


def unpack_store(stored, codec, group, width):
    stored = np.asarray(stored, dtype=np.uint8)
    payload_bytes = width // 2 if codec == MXFP4 else width
    payload, scales = stored[:, :payload_bytes], stored[:, payload_bytes:]
    if codec == MXFP4:
        codes = np.empty((stored.shape[0], width), dtype=np.uint8)
        codes[:, 0::2] = payload & 0xF
        codes[:, 1::2] = payload >> 4
    else:
        codes = payload
    if scales.shape[1] != width // group:
        raise ValueError("stored row has the wrong number of scale bytes")
    return codes, scales


def store_bytes_per_row(codec, group, width):
    check_row(codec, group, width)
    if codec == BF16:
        return 2 * width
    return (width // 2 if codec == MXFP4 else width) + width // group


def round_trip(rows_bf16, codec, group):
    rows_bf16 = np.asarray(rows_bf16, dtype=np.uint16)
    if codec == BF16:
        check_row(codec, group, rows_bf16.shape[1])
        return rows_bf16.copy()
    codes, scales = encode(rows_bf16, codec, group)
    stored = pack_store(codes, scales, codec)
    if stored.shape[1] != store_bytes_per_row(codec, group, rows_bf16.shape[1]):
        raise ValueError("stored row size disagrees with the codec")
    codes, scales = unpack_store(stored, codec, group, rows_bf16.shape[1])
    return decode(codes, scales, codec, group)
