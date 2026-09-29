#!/usr/bin/env python3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glm52_routed_parity as parity

GROUPS, HIDDEN, HALF = 4, 128, 128
LAYERS = (3, 4)


def e2m1_codes(rng, shape):
    return rng.integers(0, 16, size=shape, dtype=np.uint8)


def bf16_bytes(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise AssertionError("a fixture weight is not exactly a bf16 value")
    return (bits >> 16).astype(np.uint16).tobytes()


def nvfp4_expert(rng, rows, columns):
    codes = e2m1_codes(rng, (rows, columns))
    block_codes = rng.choice(np.array([0x30, 0x38, 0x3C, 0x40, 0x44, 0x48], dtype=np.uint8), size=(rows, columns // 16))
    global_scale = np.float32(2.0 ** rng.integers(-8, -4))
    weight = parity.E2M1[codes] * np.repeat(parity.E4M3[block_codes], 16, axis=1) * global_scale
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return weight, packed.tobytes(), block_codes.tobytes(), struct.pack("<f", global_scale)


def fp8_expert(rng, rows, columns):
    codes = rng.integers(0, 256, size=(rows, columns), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x38
    scale = (2.0 ** rng.integers(-6, -2, size=(rows, columns // 128))).astype(np.float32)
    weight = parity.E4M3[codes] * np.repeat(scale, 128, axis=1)
    return weight, codes.tobytes(), scale.tobytes()


def build(rng):
    experts = {}
    for layer in LAYERS:
        for kind, rows, columns in ((22, 2 * HALF, HIDDEN), (23, HIDDEN, HALF)):
            experts[(kind, layer)] = [(nvfp4_expert(rng, rows, columns), fp8_expert(rng, rows, columns)) for _ in range(GROUPS)]
    return experts


def write_pack(path, codec, experts, mutate=None):
    keys = sorted(experts)
    directory_offset = 512
    cursor = directory_offset + len(keys) * 64
    placed, blobs = [], []
    for key in keys:
        rows, columns = experts[key][0][0][0].shape
        if codec == parity.CODEC_BF16:
            payload = b"".join(bf16_bytes(e[0][0] if e is not None else 0) for e in experts[key])
            scale = b""
            encoding = 0
        elif codec == parity.CODEC_NVFP4:
            payload = b"".join(e[0][1] for e in experts[key])
            globals_ = b"".join(e[0][3] for e in experts[key])
            blocks = b"".join(e[0][2] for e in experts[key])
            scale = blocks + globals_ if mutate == "globals_last" else globals_ + blocks
            if mutate == "nibbles":
                raw = np.frombuffer(payload, dtype=np.uint8)
                payload = (((raw & 0x0F) << 4) | (raw >> 4)).astype(np.uint8).tobytes()
            encoding = 4
        else:
            payload = b"".join(e[1][1] for e in experts[key])
            scale = b"".join(e[1][2] for e in experts[key])
            encoding = 1
        if mutate == "swap_up_gate" and key[0] == 22:
            per = len(payload) // GROUPS
            parts = []
            for group in range(GROUPS):
                chunk = payload[group * per:(group + 1) * per]
                parts.append(chunk[per // 2:] + chunk[:per // 2])
            payload = b"".join(parts)
        payload_offset = (cursor + 255) & ~255
        cursor = payload_offset + len(payload)
        scale_offset = 0
        if scale:
            scale_offset = (cursor + 255) & ~255
            cursor = scale_offset + len(scale)
        placed.append((key[0], key[1], 4, codec, encoding, GROUPS, rows, columns, payload_offset, len(payload),
                       scale_offset, len(scale)))
        blobs.append((payload_offset, payload, scale_offset, scale))
    header = struct.pack(parity.HEADER_FORMAT, parity.MAGIC, 3, 264, 64, 1, 0, len(keys), 1, 0, 0, 78, 78, HIDDEN,
                         154880, GROUPS, 1, codec, 1, 16, 15, directory_offset, cursor,
                         b"r".ljust(65, b"\0"), bytes(32), bytes(32), bytes(32))
    blob = bytearray(cursor)
    blob[:len(header)] = header
    for index, row in enumerate(placed):
        struct.pack_into(parity.ENTRY_FORMAT, blob, directory_offset + index * 64, *row)
    for payload_offset, payload, scale_offset, scale in blobs:
        blob[payload_offset:payload_offset + len(payload)] = payload
        if scale:
            blob[scale_offset:scale_offset + len(scale)] = scale
    path.write_bytes(bytes(blob))


def run(pack, reference, *extra):
    result = subprocess.run([sys.executable, str(ROOT / "tools/glm52_routed_parity.py"), "--pack", str(pack),
                             "--reference", str(reference), "--layers", "3,4", "--tokens", "4", "--top-k", "2", *extra],
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def rel(text):
    return max(float(part.split()[0]) for part in text.split("output rel_l2 ")[1:])


def main():
    rng = np.random.default_rng(5)
    experts = build(rng)
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        nvfp4_reference = {key: [(e[0], e[1]) for e in value] for key, value in experts.items()}
        fp8_as_reference = {key: [((e[1][0],) + e[0][1:], e[1]) for e in value] for key, value in experts.items()}
        write_pack(root / "ref_nv.sp", parity.CODEC_BF16, nvfp4_reference)
        write_pack(root / "ref_fp8.sp", parity.CODEC_BF16, fp8_as_reference)
        write_pack(root / "nvfp4.sp", parity.CODEC_NVFP4, experts)
        write_pack(root / "fp8.sp", parity.CODEC_FP8, experts)
        code, text = run(root / "nvfp4.sp", root / "ref_nv.sp")
        if code != 0 or rel(text) > 1e-6:
            failures.append(f"an exactly representable nvfp4 pack did not match its bf16 reference: {text}")
        code, text = run(root / "fp8.sp", root / "ref_fp8.sp")
        if code != 0 or rel(text) > 1e-6:
            failures.append(f"an exactly representable fp8 pack did not match its bf16 reference: {text}")
        code, text = run(root / "ref_nv.sp", root / "ref_nv.sp")
        if code != 0 or rel(text) != 0.0:
            failures.append(f"a bf16 pack against itself is not exact: {text}")
        for mutation in ("nibbles", "globals_last", "swap_up_gate"):
            write_pack(root / f"{mutation}.sp", parity.CODEC_NVFP4, experts, mutation)
            code, text = run(root / f"{mutation}.sp", root / "ref_nv.sp")
            if code != 1 or "ROUTED-PARITY FAIL" not in text:
                failures.append(f"the {mutation} layout error passed: {text}")
        code, text = run(root / "nvfp4.sp", root / "nvfp4.sp")
        if code != 2 or "not bf16" not in text:
            failures.append(f"a non-bf16 reference was accepted: {text}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS glm52 routed parity: exact nvfp4 and fp8 decodes match bf16, nibble order, global placement and "
          "up/gate order errors fail, a non-bf16 reference is refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
