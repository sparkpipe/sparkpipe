#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import importlib.machinery
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKER = ROOT / "tools/glm52_stagepack.py"
CODECS = ("int6", "int7", "int8", "fp8", "nvfp4", "mxfp4")


def load_packer():
    if importlib.util.find_spec("torch") is None:
        torch_stub = types.ModuleType("torch")
        torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        torch_stub.Tensor = object
        sys.modules["torch"] = torch_stub
    if importlib.util.find_spec("safetensors") is None:
        safetensors_stub = types.ModuleType("safetensors")
        safetensors_stub.__spec__ = importlib.machinery.ModuleSpec(
            "safetensors", loader=None
        )
        safetensors_stub.safe_open = None
        sys.modules["safetensors"] = safetensors_stub
    specification = importlib.util.spec_from_file_location(
        "sparkpipe_glm52_stagepack", PACKER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("failed to load GLM stage-pack tool")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def unpack_subbyte(payload: bytes, bits: int, rows: int, columns: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(rows, columns // 8, bits)
    words = np.zeros((rows, columns // 8), dtype=np.uint64)
    for byte_index in range(bits):
        words |= packed[:, :, byte_index].astype(np.uint64) << (8 * byte_index)
    codes = np.empty((rows, columns), dtype=np.uint8)
    mask = (1 << bits) - 1
    for code_index in range(8):
        codes[:, code_index::8] = ((words >> (bits * code_index)) & mask).astype(np.uint8)
    return codes


def main() -> int:
    module = load_packer()
    assert tuple(module.CODECS) == CODECS
    for codec in CODECS:
        records = module.records_for_stage(3, codec)
        file_bytes = module.size_records(records, codec)
        expert_records = [
            record for record in records
            if record.payload_type == module.PAYLOAD_PACKED_WEIGHT
        ]
        assert len(expert_records) == 12
        assert all(record.codec == module.CODECS[codec][0] for record in expert_records)
        assert all(record.scale_encoding == module.CODECS[codec][3]
                   for record in expert_records)
        assert all(record.payload_offset % module.ALIGNMENT == 0 for record in records)
        assert all(record.scale_offset % module.ALIGNMENT == 0 for record in expert_records)
        assert file_bytes > max(record.scale_offset + record.scale_bytes
                                for record in expert_records)
        assert all(record.codec == module.CODEC_BF16
                   for record in records if record.payload_type == module.PAYLOAD_BF16)

        header = module.pack_header(
            3,
            module.CODECS[codec][0],
            "test-revision",
            bytes.fromhex("11" * 32),
            bytes.fromhex("22" * 32),
            bytes.fromhex("33" * 32),
            len(records),
            file_bytes,
        )
        fields = module.HEADER.unpack(header)
        assert fields[0:5] == (
            module.MAGIC,
            module.FORMAT_VERSION,
            module.HEADER_BYTES,
            module.ENTRY_BYTES,
            module.CODEC_ABI_VERSION,
        )
        assert fields[8] == 3
        assert fields[16] == module.CODECS[codec][0]
        assert fields[22].split(b"\0", 1)[0] == b"test-revision"
        assert fields[23:26] == (bytes.fromhex("11" * 32),
                                 bytes.fromhex("22" * 32),
                                 bytes.fromhex("33" * 32))

    rows, columns = 3, 256
    for bits in (6, 7):
        mask = (1 << bits) - 1
        codes = (np.arange(rows * columns, dtype=np.uint16) * 29 & mask)
        codes = codes.astype(np.uint8).reshape(rows, columns)
        payload = module.pack_subbyte(codes, bits)
        assert len(payload) == rows * columns * bits // 8
        assert np.array_equal(unpack_subbyte(payload, bits, rows, columns), codes)

    scale_values = np.concatenate((
        np.array((0.0, 2.0 ** -9, 2.0 ** -6), dtype=np.float32),
        np.linspace(0.01, 448.0, 1000, dtype=np.float32),
    ))
    scale_codes, decoded_scales = module.encode_e4m3_round_up(scale_values)
    assert np.all(decoded_scales >= scale_values)
    assert np.all(scale_codes <= 126)
    assert decoded_scales[-1] == 448.0
    print("PASS GLM stage-pack geometry for six codecs and bit-exact packed layouts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
