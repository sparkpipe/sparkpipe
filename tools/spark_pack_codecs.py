"""Weight codecs shared by every stage packer.

The codec ABI numbers mirror include/sparkpipe/spark_weight_codec.h and the
stage-pack weight formats mirror include/sparkpipe/spark_stagepack_format.h.
The conversions are the exact bit recipes the packers ship: bf16 widening and
round-to-nearest-even narrowing, the F8_E4M3 and E2M1 decode tables, the
128 x 128 block FP8 dequantisation, the NVFP4 decode, the E8M0 scale check and
the encoders that requantise BF16 masters to the integer, FP8, NVFP4 and MXFP4
codecs. numpy is optional for the scalar helpers and required for the array
ones; torch is imported only by the requantising encoders.
"""

from __future__ import annotations

import math
import struct

from spark_pack_common import PackFailure

try:
    import numpy as np
except ImportError:
    np = None

PAYLOAD_BF16 = 1
PAYLOAD_F32 = 2
PAYLOAD_U32 = 3
PAYLOAD_PACKED_WEIGHT = 4

CODEC_NONE = 0
CODEC_BF16 = 1
CODEC_INT6 = 2
CODEC_INT7 = 3
CODEC_INT8 = 4
CODEC_FP8 = 5
CODEC_NVFP4 = 6
CODEC_MXFP4 = 7

SCALE_NONE = 0
SCALE_F32 = 1
SCALE_UE4M3 = 2
SCALE_E8M0 = 3
SCALE_UE4M3_F32_GLOBAL = 4

CODEC_ABI_VERSION = 1

WEIGHT_CODECS = {
    "int6": (CODEC_INT6, 6, 32, SCALE_F32),
    "int7": (CODEC_INT7, 7, 128, SCALE_F32),
    "int8": (CODEC_INT8, 8, 128, SCALE_F32),
    "fp8": (CODEC_FP8, 8, 128, SCALE_F32),
    "nvfp4": (CODEC_NVFP4, 4, 16, SCALE_UE4M3_F32_GLOBAL),
    "mxfp4": (CODEC_MXFP4, 4, 32, SCALE_E8M0),
}

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_E4M3_F32B128 = 4
WEIGHT_NVFP4_PACKED = 8
WEIGHT_MXFP4_E2M1_E8M0G32 = 9

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
               "U16": 2, "I16": 2, "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "F32": 4,
               "U64": 8, "I64": 8, "F64": 8}

FP8_BLOCK = 128
NVFP4_GROUP = 16
MXFP4_GROUP = 32
E8M0_NAN = 0xFF


def dtype_bytes(dtype: str) -> int:
    if dtype not in DTYPE_BYTES:
        raise PackFailure(f"unknown safetensors dtype {dtype}")
    return DTYPE_BYTES[dtype]


def bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def float_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def f32_round(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bf16_bytes_to_floats(raw: bytes) -> list:
    return [bf16_bits_to_float(bits) for (bits,) in struct.iter_unpack("<H", raw)]


def floats_to_bf16_bytes(values) -> bytes:
    values = list(values)
    return struct.pack(f"<{len(values)}H", *(float_to_bf16_bits(value) for value in values))


def widen_bf16_bytes(raw) -> bytearray:
    widened = bytearray(len(raw) * 2)
    widened[2::4] = raw[0::2]
    widened[3::4] = raw[1::2]
    return widened


def bf16_to_f32(bits):
    return (np.asarray(bits).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(values, quiet_nan: bool = False):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
               >> np.uint32(16)).astype(np.uint16)
    if quiet_nan:
        source_nan = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & \
                     ((bits & np.uint32(0x007FFFFF)) != 0)
        if source_nan.any():
            quiet = ((bits >> np.uint32(16)) & np.uint32(0xFF80)).astype(np.uint16) | np.uint16(0x0040)
            rounded = np.where(source_nan, quiet, rounded)
    return rounded


def fp8_e4m3_lut(nan: bool = True):
    codes = np.arange(256, dtype=np.uint32)
    sign = (codes >> 7) & 1
    exponent = (codes >> 3) & 0xF
    mantissa = codes & 0x7
    value = np.where(
        exponent == 0,
        mantissa.astype(np.float32) * np.float32(2.0 ** -9),
        (np.float32(1.0) + mantissa.astype(np.float32) / np.float32(8.0))
        * np.power(np.float32(2.0), (exponent.astype(np.int32) - 7).astype(np.float32)),
    ).astype(np.float32)
    if nan:
        value[(exponent == 15) & (mantissa == 7)] = np.float32("nan")
    return np.where(sign == 1, -value, value).astype(np.float32)


def e2m1_lut():
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    return np.concatenate([magnitudes, -magnitudes]).astype(np.float32)


def fp8_block_dequant_bf16(codes, scale, lut, quiet_nan: bool = False):
    rows, columns = codes.shape
    expanded = np.repeat(scale, FP8_BLOCK, axis=0)[:rows]
    expanded = np.repeat(expanded, FP8_BLOCK, axis=1)[:, :columns]
    return f32_to_bf16(lut[codes] * expanded, quiet_nan)


def fp8_block_scale_rows(plane, row_start: int, row_count: int, block_start: int, block_count: int):
    rows = [row // FP8_BLOCK for row in range(row_start, row_start + row_count)]
    return plane[rows, :][:, block_start:block_start + block_count]


E2M1_LUT = e2m1_lut() if np is not None else None
E4M3_LUT = fp8_e4m3_lut() if np is not None else None
E4M3_FINITE_LUT = fp8_e4m3_lut(nan=False) if np is not None else None


def e2m1_unpack(packed):
    values = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float32)
    values[..., 0::2] = E2M1_LUT[packed & np.uint8(0x0F)]
    values[..., 1::2] = E2M1_LUT[packed >> np.uint8(4)]
    return values


def nvfp4_to_bf16(packed, scales, global_scale: float, scale_first: bool = False):
    values = e2m1_unpack(packed)
    width = values.shape[-1]
    expanded = np.repeat(scales, NVFP4_GROUP, axis=-1)[..., :width]
    if scale_first:
        return f32_to_bf16(values * (expanded * np.float32(global_scale)))
    return f32_to_bf16(values * expanded * np.float32(global_scale))


def check_e8m0_scales(name: str, raw: bytes) -> None:
    if bytes([E8M0_NAN]) in raw:
        raise PackFailure(f"{name}: E8M0 0xff (NaN) in the scale plane")
    high = sum(1 for value in raw if value >= 253)
    if high:
        print(f"ADVISORY {name}: {high} E8M0 codes >= 253; "
              f"bf16 dequant can overflow at this magnitude")


def pack_subbyte(codes, bits: int) -> bytes:
    rows, columns = codes.shape
    if columns % 8 != 0:
        raise PackFailure("packed rows require a whole eight-code group")
    octets = (codes.astype(np.int16, copy=False) & ((1 << bits) - 1)).astype(np.uint64)
    octets = octets.reshape(rows, columns // 8, 8)
    packed = np.zeros(octets.shape[:2], dtype=np.uint64)
    for index in range(8):
        packed |= octets[:, :, index] << (index * bits)
    return packed.view(np.uint8).reshape(rows, columns // 8, 8)[:, :, :bits].tobytes(order="C")


def e4m3_positive_values():
    values = np.zeros(127, dtype=np.float32)
    for code in range(1, 127):
        exponent = code >> 3
        mantissa = code & 7
        if exponent == 0:
            values[code] = mantissa * (2.0 ** -9)
        else:
            values[code] = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return values


def encode_e4m3_round_up(values):
    table = e4m3_positive_values()
    codes = np.searchsorted(table, values, side="left")
    codes = np.clip(codes, 0, 126).astype(np.uint8)
    return codes, table[codes]


def encode_e2m1(values):
    import torch
    magnitude = values.abs()
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=values.device)
    codes = torch.bucketize(magnitude, boundaries, right=False)
    for boundary_index in (1, 3, 5):
        equal = magnitude == boundaries[boundary_index]
        codes = torch.where(equal, torch.full_like(codes, boundary_index + 1), codes)
    return codes.to(torch.uint8) | torch.where(values < 0, 8, 0).to(torch.uint8)


def quantize_matrix(matrix, codec_name: str, device):
    import torch
    _, bits, group_size, _ = WEIGHT_CODECS[codec_name]
    rows, columns = matrix.shape
    if columns % group_size != 0 or columns * bits % 8 != 0:
        raise PackFailure(f"{codec_name} cannot tile matrix shape {tuple(matrix.shape)}")
    values = matrix.to(device=device, dtype=torch.float32)
    blocks = values.reshape(rows, columns // group_size, group_size)
    block_amax = blocks.abs().amax(dim=2)
    global_bytes = b""
    if codec_name in ("int6", "int7", "int8"):
        maximum = {"int6": 31.0, "int7": 63.0, "int8": 127.0}[codec_name]
        minimum = {"int6": -32, "int7": -64, "int8": -128}[codec_name]
        scales = torch.clamp(block_amax / maximum, min=1.0e-12)
        codes = torch.round(blocks / scales.unsqueeze(2)).clamp(minimum, int(maximum))
        codes = codes.reshape(rows, columns).to(torch.int8).cpu().numpy()
        payload = codes.view(np.uint8).tobytes(order="C") if bits == 8 else pack_subbyte(codes, bits)
        scale_bytes = scales.cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
    elif codec_name == "fp8":
        scales = torch.clamp(block_amax / 448.0, min=1.0e-12)
        normalized = (blocks / scales.unsqueeze(2)).reshape(rows, columns)
        codes = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
        payload = codes.cpu().numpy().tobytes(order="C")
        scale_bytes = scales.cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
    elif codec_name == "nvfp4":
        tensor_amax = float(block_amax.max().item())
        global_scale = max(tensor_amax / (448.0 * 6.0), 1.0e-30)
        raw_block_scales = (block_amax.cpu().numpy() / 6.0) / global_scale
        scale_codes, decoded_blocks = encode_e4m3_round_up(raw_block_scales)
        decoded = torch.from_numpy(decoded_blocks.copy()).to(device=device)
        final_scales = decoded.unsqueeze(2) * global_scale
        normalized = torch.where(final_scales > 0, blocks / final_scales, torch.zeros_like(blocks))
        codes = encode_e2m1(normalized).reshape(rows, columns).cpu().numpy()
        payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).tobytes(order="C")
        global_bytes = struct.pack("<f", global_scale)
        scale_bytes = scale_codes.tobytes(order="C")
    else:
        raw_scales = np.maximum(block_amax.cpu().numpy() / 6.0, np.float32(2.0 ** -127))
        exponents = np.ceil(np.log2(raw_scales)).astype(np.int32)
        scale_codes = np.clip(exponents + 127, 0, 254).astype(np.uint8)
        decoded = torch.from_numpy(np.exp2(scale_codes.astype(np.float32) - 127.0)).to(device=device)
        normalized = blocks / decoded.unsqueeze(2)
        codes = encode_e2m1(normalized).reshape(rows, columns).cpu().numpy()
        payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).tobytes(order="C")
        scale_bytes = scale_codes.tobytes(order="C")
    return payload, global_bytes, scale_bytes
