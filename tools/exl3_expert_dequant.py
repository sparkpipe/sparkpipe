#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

CODEBOOKS = {"3inst": 0, "mcg": 1, "mul1": 2}
CODEBOOK_NAMES = {value: key for key, value in CODEBOOKS.items()}
MCG_MULT = 0xCBAC1FED
MUL1_MULT = 0x83DCD12D
INST_MULT = 89226354
INST_ADD = 64248484
LOP3_MASK = 0x8FFF8FFF
LOP3_XOR = 0x3B603B60
MUL1_ACC = 0x6400
MUL1_INV_BITS = 0x1EEE
MUL1_BIAS_BITS = 0xC931
TILE = 16
TILE_ELEMS = 256
HAD = 128
BF16_DROP = 45
DTYPES = {"F16": np.float16, "BF16": np.uint16, "F32": np.float32, "F64": np.float64,
          "I16": np.int16, "I32": np.int32, "I64": np.int64, "U8": np.uint8, "U16": np.uint16,
          "I8": np.int8, "U32": np.uint32}
LUT_SHA256 = {
    0: "ad88de2dcff5bb05fff2faec37cceceb43aa1c2e8203f9d81a6fbd6dd3393eea",
    1: "95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9",
    2: "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8",
}


class DequantFailure(Exception):
    pass


def codebook_lut(codebook: int) -> np.ndarray:
    states = np.arange(1 << 16, dtype=np.uint64)
    if codebook == 2:
        x = (states * np.uint64(MUL1_MULT)) & np.uint64(0xFFFFFFFF)
        byte_sum = sum(((x >> np.uint64(shift)) & np.uint64(0xFF)) for shift in (0, 8, 16, 24))
        h = (byte_sum + np.uint64(MUL1_ACC)).astype(np.uint16).view(np.float16).astype(np.float64)
        k_inv = float(np.array([MUL1_INV_BITS], dtype=np.uint16).view(np.float16)[0])
        k_bias = float(np.array([MUL1_BIAS_BITS], dtype=np.uint16).view(np.float16)[0])
        return (h * k_inv + k_bias).astype(np.float16)
    if codebook == 1:
        x = (states * np.uint64(MCG_MULT)) & np.uint64(0xFFFFFFFF)
    elif codebook == 0:
        x = (states * np.uint64(INST_MULT) + np.uint64(INST_ADD)) & np.uint64(0xFFFFFFFF)
    else:
        raise DequantFailure(f"unknown codebook {codebook}")
    x = (x & np.uint64(LOP3_MASK)) ^ np.uint64(LOP3_XOR)
    low = (x & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float64)
    high = (x >> np.uint64(16)).astype(np.uint16).view(np.float16).astype(np.float64)
    return (low + high).astype(np.float16)


def tile_permutation() -> np.ndarray:
    perm = np.zeros(TILE_ELEMS, dtype=np.int64)
    for lane in range(32):
        r0 = (lane % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = lane // 4
        for j, row in enumerate(rows):
            perm[lane * 8 + j] = row * TILE + c0
            perm[lane * 8 + 4 + j] = row * TILE + c0 + 8
    return perm


def state_window(bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    words = bits * TILE_ELEMS // 32
    position = np.arange(TILE_ELEMS, dtype=np.int64)
    b0 = position * bits + bits - 16 + TILE_ELEMS * bits
    b1 = b0 + 16
    i0 = b0 // 32
    i1 = (b1 - 1) // 32
    s0 = (i1 + 1) * 32 - b1
    return i0 % words, i1 % words, s0.astype(np.uint64)


class Decoder:
    def __init__(self):
        self.luts: Dict[int, np.ndarray] = {}
        self.windows: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        perm = tile_permutation()
        self.row_major_source = np.argsort(perm)
        signs = np.array([[1 - 2 * (bin(i & j).count("1") & 1) for j in range(HAD)] for i in range(HAD)],
                         dtype=np.float64)
        self.hadamard = signs

    def lut(self, codebook: int) -> np.ndarray:
        if codebook not in self.luts:
            table = codebook_lut(codebook)
            digest = hashlib.sha256(table.tobytes()).hexdigest()
            if digest != LUT_SHA256[codebook]:
                raise DequantFailure(f"{CODEBOOK_NAMES[codebook]} table sha256 {digest} differs from the table "
                                     "verified against exllamav3 on all 65536 states (numpy float16 rounding?)")
            self.luts[codebook] = table
        return self.luts[codebook]

    def window(self, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bits not in self.windows:
            i0, i1, s0 = state_window(bits)
            order = self.row_major_source
            self.windows[bits] = (i0[order], i1[order], s0[order])
        return self.windows[bits]

    def states(self, trellis: np.ndarray) -> np.ndarray:
        if trellis.dtype != np.int16 or trellis.ndim != 3:
            raise DequantFailure(f"trellis must be int16 [tiles_k, tiles_n, 16*K], got {trellis.dtype} {trellis.shape}")
        width = trellis.shape[2]
        if width % TILE or not 1 <= width // TILE <= 8:
            raise DequantFailure(f"trellis tile width {width} is not 16*K for an integer K in 1..8")
        bits = width // TILE
        tiles = np.ascontiguousarray(trellis).view(np.uint32).reshape(-1, bits * TILE_ELEMS // 32)
        i0, i1, s0 = self.window(bits)
        high = tiles[:, i0].astype(np.uint64)
        low = tiles[:, i1].astype(np.uint64)
        return ((((high << np.uint64(32)) | low) >> s0) & np.uint64(0xFFFF)).astype(np.uint16)

    def inner(self, trellis: np.ndarray, codebook: int) -> np.ndarray:
        tiles_k, tiles_n = trellis.shape[0], trellis.shape[1]
        values = self.lut(codebook)[self.states(trellis)]
        values = values.reshape(tiles_k, tiles_n, TILE, TILE).transpose(0, 2, 1, 3)
        return np.ascontiguousarray(values).reshape(tiles_k * TILE, tiles_n * TILE)

    def rotated(self, inner: np.ndarray) -> np.ndarray:
        k, n = inner.shape
        if k % HAD or n % HAD:
            raise DequantFailure(f"weight {k}x{n} is not a whole number of {HAD}-wide Hadamard blocks")
        w = inner.astype(np.float64)
        blocks = w.reshape(k // HAD, HAD, n).transpose(1, 0, 2).reshape(HAD, -1)
        w = (self.hadamard @ blocks).reshape(HAD, k // HAD, n).transpose(1, 0, 2).reshape(k, n)
        w = (w.reshape(-1, HAD) @ self.hadamard).reshape(k, n)
        return w

    def weight_parts(self, trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray,
                     codebook: int) -> Tuple[np.ndarray, np.ndarray]:
        k, n = trellis.shape[0] * TILE, trellis.shape[1] * TILE
        if suh.shape != (k,) or svh.shape != (n,):
            raise DequantFailure(f"suh {suh.shape} / svh {svh.shape} do not match a {k}x{n} trellis")
        if suh.dtype != np.float16 or svh.dtype != np.float16:
            raise DequantFailure("suh and svh must be float16")
        left = self.rotated(self.inner(trellis, codebook)) * suh.astype(np.float64)[:, None]
        return left, svh.astype(np.float64)

    def weight_f64(self, trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, codebook: int) -> np.ndarray:
        left, right = self.weight_parts(trellis, suh, svh, codebook)
        return left * right[None, :] / float(HAD)

    def weight_bf16(self, trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray,
                    codebook: int) -> Tuple[np.ndarray, int]:
        left, right = self.weight_parts(trellis, suh, svh, codebook)
        product = left * right[None, :] / float(HAD)
        return round_bf16_exact(product, left, right)


def round_bf16_exact(product: np.ndarray, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, int]:
    bits = product.view(np.uint64)
    sign = bits & np.uint64(1 << 63)
    magnitude = bits & np.uint64((1 << 63) - 1)
    if np.any(magnitude >= np.uint64(0x7FF0000000000000)):
        raise DequantFailure("non-finite value in the decoded weight")
    nonzero = magnitude != 0
    exponent = (magnitude >> np.uint64(52)).astype(np.int64)
    if np.any(nonzero & (exponent < 1023 - 126)):
        raise DequantFailure("decoded weight below the bf16 normal range")
    if np.any(exponent > 1023 + 127):
        raise DequantFailure("decoded weight above the bf16 range")
    dropped = magnitude & np.uint64((1 << BF16_DROP) - 1)
    kept = magnitude >> np.uint64(BF16_DROP)
    half = np.uint64(1 << (BF16_DROP - 1))
    up = (dropped > half) | ((dropped == half) & ((kept & np.uint64(1)) == np.uint64(1)))
    ties = np.flatnonzero(dropped == half)
    corrected = 0
    if ties.size:
        flat_left = left.reshape(-1)
        flat_right = np.broadcast_to(right[None, :], left.shape).reshape(-1)
        flat_product = product.reshape(-1)
        flat_up = up.reshape(-1)
        for index in ties:
            exact = abs(Fraction(float(flat_left[index])) * Fraction(float(flat_right[index])) / HAD)
            rounded = abs(Fraction(float(flat_product[index])))
            if exact != rounded:
                flat_up[index] = exact > rounded
                corrected += 1
    kept = kept + up.astype(np.uint64)
    rounded_bits = sign | (kept << np.uint64(BF16_DROP))
    as_f32 = rounded_bits.view(np.float64).astype(np.float32)
    return (as_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16), corrected


def bf16_to_f64(codes: np.ndarray) -> np.ndarray:
    return (codes.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)


class SafeTensors:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                raise DequantFailure(f"{path}: not a safetensors file")
            (length,) = struct.unpack("<Q", head)
            header = json.loads(handle.read(length))
        self.metadata = header.pop("__metadata__", None) or {}
        self.data_start = 8 + length
        self.tensors = header
        self.size = path.stat().st_size
        self.fd: Optional[int] = None

    def meta(self, name: str) -> Tuple[str, List[int], int, int]:
        entry = self.tensors[name]
        start, end = entry["data_offsets"]
        return entry["dtype"], list(entry["shape"]), self.data_start + start, end - start

    def raw(self, name: str) -> bytes:
        _, _, offset, count = self.meta(name)
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        data = os.pread(self.fd, count, offset)
        if len(data) != count:
            raise DequantFailure(f"{self.path}: short read of {name}")
        return data

    def array(self, name: str) -> np.ndarray:
        dtype, shape, _, _ = self.meta(name)
        if dtype not in DTYPES:
            raise DequantFailure(f"{name}: unsupported dtype {dtype}")
        return np.frombuffer(self.raw(name), dtype=DTYPES[dtype]).reshape(shape)


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.files: Dict[str, SafeTensors] = {}
        self.where: Dict[str, str] = {}
        index = root / "model.safetensors.index.json"
        if root.is_file():
            names = [root.name]
            self.root = root.parent
        elif index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            names = sorted(set(weight_map.values()))
        else:
            names = sorted(p.name for p in root.glob("*.safetensors"))
        if not names:
            raise DequantFailure(f"{root}: no safetensors files")
        for name in names:
            reader = SafeTensors(self.root / name)
            self.files[name] = reader
            for tensor in reader.tensors:
                if tensor in self.where:
                    raise DequantFailure(f"{tensor} appears in {self.where[tensor]} and {name}")
                self.where[tensor] = name

    def has(self, name: str) -> bool:
        return name in self.where

    def reader(self, name: str) -> SafeTensors:
        if name not in self.where:
            raise DequantFailure(f"{self.root}: tensor {name} not found")
        return self.files[self.where[name]]

    def array(self, name: str) -> np.ndarray:
        return self.reader(name).array(name)

    def sha256(self, name: str) -> str:
        return hashlib.sha256(self.reader(name).raw(name)).hexdigest()


def linear_codebook(source: Checkpoint, prefix: str) -> int:
    found = [cb for cb in ("mcg", "mul1") if source.has(f"{prefix}.{cb}")]
    if len(found) > 1:
        raise DequantFailure(f"{prefix}: both mcg and mul1 markers present")
    if source.has(f"{prefix}.su") or source.has(f"{prefix}.sv"):
        raise DequantFailure(f"{prefix}: packed sign vectors (su/sv) are not supported; expected suh/svh")
    return CODEBOOKS[found[0]] if found else CODEBOOKS["3inst"]


def load_linear(source: Checkpoint, prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    trellis = source.array(f"{prefix}.trellis")
    suh = source.array(f"{prefix}.suh")
    svh = source.array(f"{prefix}.svh")
    if source.has(f"{prefix}.bias"):
        raise DequantFailure(f"{prefix}: bias is not supported for expert decode")
    return trellis, suh, svh, linear_codebook(source, prefix)


def slice_linear(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, axis: str,
                 start: int, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if start % HAD or count % HAD:
        raise DequantFailure(f"slice [{start}, {start + count}) is not aligned to {HAD}-wide Hadamard blocks")
    t0, t1 = start // TILE, (start + count) // TILE
    if axis == "out":
        if start + count > svh.shape[0]:
            raise DequantFailure(f"output slice past {svh.shape[0]}")
        return np.ascontiguousarray(trellis[:, t0:t1, :]), suh, np.ascontiguousarray(svh[start:start + count])
    if axis == "in":
        if start + count > suh.shape[0]:
            raise DequantFailure(f"input slice past {suh.shape[0]}")
        return np.ascontiguousarray(trellis[t0:t1, :, :]), np.ascontiguousarray(suh[start:start + count]), svh
    raise DequantFailure(f"slice axis {axis} is not in or out")


def write_safetensors(path: Path, tensors: Iterable[Tuple[str, np.ndarray]], metadata: Dict[str, str]) -> str:
    items = list(tensors)
    names = {"float16": "F16", "int16": "I16", "int32": "I32", "uint16": "U16", "float32": "F32"}
    header: Dict[str, object] = {"__metadata__": metadata}
    cursor = 0
    for name, array in items:
        if str(array.dtype) not in names:
            raise DequantFailure(f"{name}: cannot write dtype {array.dtype}")
        header[name] = {"dtype": names[str(array.dtype)], "shape": list(array.shape),
                        "data_offsets": [cursor, cursor + array.nbytes]}
        cursor += array.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    digest = hashlib.sha256()
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as out:
        for chunk in (struct.pack("<Q", len(blob)), blob):
            out.write(chunk)
            digest.update(chunk)
        for _, array in items:
            data = np.ascontiguousarray(array).tobytes()
            out.write(data)
            digest.update(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def lut_report() -> dict:
    report = {}
    for name, codebook in CODEBOOKS.items():
        lut = codebook_lut(codebook)
        report[name] = {"sha256": hashlib.sha256(lut.tobytes()).hexdigest(),
                        "min": float(lut.astype(np.float64).min()), "max": float(lut.astype(np.float64).max())}
    return report


def decode_command(args: argparse.Namespace) -> int:
    source = Checkpoint(Path(args.source))
    decoder = Decoder()
    trellis, suh, svh, codebook = load_linear(source, args.prefix)
    if args.slice_axis:
        trellis, suh, svh = slice_linear(trellis, suh, svh, args.slice_axis, args.slice_start, args.slice_count)
    codes, corrected = decoder.weight_bf16(trellis, suh, svh, codebook)
    out = Path(args.output)
    np.save(out, codes)
    result = {"prefix": args.prefix, "codebook": CODEBOOK_NAMES[codebook], "bits": trellis.shape[2] // TILE,
              "shape_in_out": list(codes.shape), "tie_corrections": corrected,
              "bf16_sha256": hashlib.sha256(codes.tobytes()).hexdigest(), "output": str(out)}
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact offline EXL3 to BF16 decode (weight layout [in, out]).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lut", help="print the sha256 of each 65536-entry codebook table")
    decode = sub.add_parser("decode", help="decode one EXL3 linear to a BF16 .npy in [in, out] layout")
    decode.add_argument("--source", required=True)
    decode.add_argument("--prefix", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--slice-axis", choices=("in", "out"))
    decode.add_argument("--slice-start", type=int, default=0)
    decode.add_argument("--slice-count", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "lut":
            print(json.dumps(lut_report(), indent=1, sort_keys=True))
            return 0
        return decode_command(args)
    except DequantFailure as error:
        print(f"DEQUANT-REFUSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
