#!/usr/bin/env python3
"""Hermetic checks for the numpy fp8-source GLM52 resident stage packer.

Covers (no checkpoint, no torch, no safetensors):
  1. the F8_E4M3 decode table (anchor codes against the e4m3fn definition),
  2. float32 -> bf16 round-to-nearest-even (ties, carry, NaN),
  3. Fp8SourceReader over a synthetic blockwise-FP8 safetensors store:
     BF16 passthrough byte-exactness, code*scale_inv dequant, expert fp8
     payload verbatim slicing, row-expanded scale slicing,
  4. every plan entry of tools/glm52_resident_stagepack.py (tp=1 and tp=8,
     all ranks) matches the C-module policy mirror in
     tools/glm52_validate_pack.py: fields, payload/scale byte counts, and
     the global/layer inventory masks.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


packer = load_module(TOOLS / "glm52_resident_stagepack.py", "sparkpipe_glm52_resident_stagepack")
validator = load_module(TOOLS / "glm52_validate_pack.py", "sparkpipe_glm52_validate_pack")


def check_lut():
    lut = packer.fp8_e4m3_lut()
    anchors = {
        0x00: 0.0, 0x80: 0.0,            # +/- zero
        0x38: 1.0, 0xB8: -1.0,           # unity
        0x40: 2.0, 0x30: 0.5,
        0x07: 7.0 * 2.0 ** -9,           # largest denormal
        0x01: 2.0 ** -9,                 # smallest denormal
        0x7E: 448.0, 0xFE: -448.0,       # largest normal
    }
    for code, expected in anchors.items():
        got = lut[code]
        assert got == expected, f"lut[0x{code:02x}]={got}, expected {expected}"
    assert np.isnan(lut[0x7F]) and np.isnan(lut[0xFF]), "e4m3fn NaN codes"
    # every finite code decodes back through the sign/exp/man formula
    codes = np.arange(256, dtype=np.uint32)
    exp = (codes >> 3) & 0xF
    finite = ~((exp == 15) & ((codes & 7) == 7))
    assert np.isfinite(lut[finite]).all() and np.isnan(lut[~finite]).all()
    print("PASS e4m3fn decode table (10 anchors, 254 finite codes, 2 NaN)")


def check_bf16_rounding():
    def f32_bits(bits: int) -> np.ndarray:
        return np.array([bits], dtype=np.uint32).view(np.float32)

    cases = [
        (0x3F800000, 0x3F80),   # 1.0 exact
        (0xBF800000, 0xBF80),   # -2.0 exact
        (0x3F800001, 0x3F80),   # below half-ulp: round down
        (0x3F808000, 0x3F80),   # exact tie, low bit 0: ties to even
        (0x3F818000, 0x3F82),   # exact tie, low bit 1: ties to even (up)
        (0x3F80FFFF, 0x3F81),   # above halfway: round up
        (0x7F800000, 0x7F80),   # +inf preserves
        (0xFF800000, 0xFF80),   # -inf preserves
        (0x00000039, 0x0000),   # tiny denormal: round down to zero
        (0x33800000, 0x3380),   # 2^-23 exponent boundary exact
    ]
    for bits, expected in cases:
        got = packer.f32_to_bf16_u16(f32_bits(bits))[0]
        assert got == expected, f"bf16(0x{bits:08x})=0x{got:04x}, expected 0x{expected:04x}"
    nan = packer.f32_to_bf16_u16(f32_bits(0x7FC12345))[0]
    assert (nan & 0x7FC0) == 0x7FC0, f"NaN does not stay quiet: 0x{nan:04x}"
    # round trip: bf16 -> f32 -> bf16 is identity for every non-NaN pattern
    # (NaN canonicalizes to quiet by design)
    all_u16 = np.arange(65536, dtype=np.uint16)
    non_nan = (all_u16 & 0x7F80) != 0x7F80
    back = packer.f32_to_bf16_u16(packer.bf16_u16_to_f32(all_u16))
    assert np.array_equal(back[non_nan], all_u16[non_nan]), \
        "bf16->f32->bf16 not identity away from NaN"
    print("PASS f32->bf16 round-to-nearest-even (10 bit cases, 65536 round trips)")


def write_synthetic_store(root: Path):
    """One-shard safetensors store: fp8 [256,256] + scale_inv [2,2], a
    partial-block fp8 [320,256] + scale_inv [3,2] (320 rows = 2.5 blocks),
    and bf16 [4,6]."""
    codes = (np.arange(256 * 256, dtype=np.uint32) * 7 + 3).astype(np.uint8).reshape(256, 256)
    scale = np.array([[1.0, 2.0], [0.5, 4.0]], dtype=np.float32)
    partial_codes = (np.arange(320 * 256, dtype=np.uint32) * 11 + 1).astype(np.uint8).reshape(320, 256)
    partial_scale = np.array([[1.0, 0.25], [2.0, 0.5], [4.0, 1.0]], dtype=np.float32)
    bf16 = (np.arange(4 * 6, dtype=np.uint16) * 257 + 1).reshape(4, 6)
    header = {}
    blobs = []
    offset = 0
    for name, dtype, shape, array_bytes in (
        ("model.layers.5.self_attn.q_a_proj.weight", "F8_E4M3", [256, 256], codes.tobytes()),
        ("model.layers.5.self_attn.q_a_proj.weight_scale_inv", "F32", [2, 2], scale.tobytes()),
        ("model.layers.5.self_attn.kv_a_proj_with_mqa.weight", "F8_E4M3",
         [320, 256], partial_codes.tobytes()),
        ("model.layers.5.self_attn.kv_a_proj_with_mqa.weight_scale_inv", "F32",
         [3, 2], partial_scale.tobytes()),
        ("model.embed_tokens.weight", "BF16", [4, 6], bf16.tobytes()),
    ):
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(array_bytes)]}
        blobs.append(array_bytes)
        offset += len(array_bytes)
    header_bytes = json.dumps(header).encode()
    shard = root / "model-00001-of-00001.safetensors"
    with shard.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model-00001-of-00001.safetensors"
                        for name in header}}))
    (root / "config.json").write_text(json.dumps({"architectures": ["TestModel"]}))
    return codes, scale, bf16, partial_codes, partial_scale


def check_reader():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        codes, scale, bf16, partial_codes, partial_scale = write_synthetic_store(root)
        reader = packer.Fp8SourceReader(root)

        # BF16 passthrough: byte-exact uint16 matrix
        matrix = reader.spine_bf16("model.embed_tokens.weight")
        assert matrix.dtype == np.uint16 and matrix.shape == (4, 6)
        assert np.array_equal(matrix, bf16), "BF16 passthrough drift"

        # fp8 dequant: bf16(f32(code) * scale_expanded)
        matrix = reader.spine_bf16("model.layers.5.self_attn.q_a_proj.weight")
        assert matrix.shape == (256, 256)
        expanded = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
        expected = packer.f32_to_bf16_u16(packer.fp8_e4m3_lut()[codes] * expanded)
        assert np.array_equal(matrix, expected), "code*scale_inv dequant drift"

        # partial quantization block: 320 rows = ceil(320/128)=3 tile rows,
        # the last covering only 64 rows (kv_a_proj_with_mqa is [576, 6144])
        matrix = reader.spine_bf16("model.layers.5.self_attn.kv_a_proj_with_mqa.weight")
        expanded = np.repeat(partial_scale, 128, axis=0)[:320]
        expanded = np.repeat(expanded, 128, axis=1)[:, :256]
        expected = packer.f32_to_bf16_u16(
            packer.fp8_e4m3_lut()[partial_codes] * expanded)
        assert matrix.shape == (320, 256)
        assert np.array_equal(matrix, expected), "partial-block dequant drift"

        # expert payload: verbatim byte slices (rows, then cols)
        payload = reader.expert_payload("model.layers.5.self_attn.q_a_proj.weight",
                                        128, 256, 0, 256)
        assert payload == codes[128:256, :].tobytes(), "row payload slice drift"
        payload = reader.expert_payload("model.layers.5.self_attn.q_a_proj.weight",
                                        0, 256, 64, 128)
        assert payload == codes[:, 64:128].tobytes(), "col payload slice drift"

        # expert scale: scale rows expanded across 128-row blocks, sliced
        got = np.frombuffer(
            reader.expert_scale("model.layers.5.self_attn.q_a_proj.weight",
                                128, 256, 0, 256), dtype=np.float32).reshape(128, 2)
        assert np.all(got == scale[1]), "row scale slice drift"
        got = np.frombuffer(
            reader.expert_scale("model.layers.5.self_attn.q_a_proj.weight",
                                0, 256, 128, 256), dtype=np.float32).reshape(256, 1)
        assert np.all(got == np.repeat(scale[:, [1]], 128, axis=0)), "col scale slice drift"
        reader.close()
    print("PASS Fp8SourceReader (bf16 passthrough, fp8 dequant, partial blocks, "
          "verbatim payload, scale expansion)")


class MockSource:
    """Name-driven shape/dtype oracle: enough for plan construction."""

    SHAPES = {
        "model.embed_tokens.weight": ("BF16", (154880, 6144)),
        "lm_head.weight": ("BF16", (154880, 6144)),
        "model.norm.weight": ("BF16", (6144,)),
        "input_layernorm.weight": ("BF16", (6144,)),
        "post_attention_layernorm.weight": ("BF16", (6144,)),
        "q_a_layernorm.weight": ("BF16", (2048,)),
        "kv_a_layernorm.weight": ("BF16", (512,)),
        "q_a_proj.weight": ("F8_E4M3", (2048, 6144)),
        "q_b_proj.weight": ("F8_E4M3", (16384, 2048)),
        "kv_a_proj_with_mqa.weight": ("F8_E4M3", (576, 6144)),
        "kv_b_proj.weight": ("F8_E4M3", (28672, 512)),
        "o_proj.weight": ("F8_E4M3", (6144, 16384)),
        "indexer.wq_b.weight": ("F8_E4M3", (4096, 2048)),
        "indexer.wk.weight": ("F8_E4M3", (128, 6144)),
        "indexer.weights_proj.weight": ("BF16", (32, 6144)),
        "indexer.k_norm.weight": ("BF16", (128,)),
        "indexer.k_norm.bias": ("BF16", (128,)),
        "mlp.gate_proj.weight": ("F8_E4M3", (12288, 6144)),
        "mlp.up_proj.weight": ("F8_E4M3", (12288, 6144)),
        "mlp.down_proj.weight": ("F8_E4M3", (6144, 12288)),
        "mlp.gate.weight": ("BF16", (256, 6144)),
        "mlp.gate.e_score_correction_bias": ("F32", (256,)),
        "shared_experts.gate_proj.weight": ("F8_E4M3", (2048, 6144)),
        "shared_experts.up_proj.weight": ("F8_E4M3", (2048, 6144)),
        "shared_experts.down_proj.weight": ("F8_E4M3", (6144, 2048)),
    }
    EXPERT_SHAPES = {
        "up_proj.weight": ("F8_E4M3", (2048, 6144)),
        "gate_proj.weight": ("F8_E4M3", (2048, 6144)),
        "down_proj.weight": ("F8_E4M3", (6144, 2048)),
    }

    def meta(self, name: str):
        if name.endswith("_scale_inv"):
            base = name[: -len("_scale_inv")]
            dtype, shape, _shard = self.meta(base)
            return "F32", (shape[0] // 128, shape[1] // 128), "mock"
        if ".mlp.experts." in name:
            tail = name.split(".mlp.experts.")[1].split(".", 2)[2]
            dtype, shape = self.EXPERT_SHAPES[tail]
            return dtype, shape, "mock"
        for key, value in self.SHAPES.items():
            if name == key or name.endswith("." + key):
                return value[0], value[1], "mock"
        raise AssertionError(f"mock source has no shape for {name}")

    def close(self):
        pass


CONTRACT = json.loads((ROOT / "model_contracts" / "glm52.json").read_text())


def build_entries(tp_degree: int, tp_rank: int):
    source = MockSource()
    instance = packer.Packer(source, CONTRACT, (0, 77), True, True, tp_degree, tp_rank)
    instance.build_plan()
    return instance


def check_plan_against_validator(tp_degree: int):
    seen_global = 0
    seen_layer: dict = {}
    instance = build_entries(tp_degree, 0)
    for item in instance.plan:
        entry = item.entry
        expected = validator.expected_shape(entry.kind, entry.layer, tp_degree)
        assert expected[0] is not None, \
            f"kind {entry.kind} layer {entry.layer}: validator rejects: {expected[1]}"
        got = (entry.payload_type, entry.weight_codec, entry.scale_encoding,
               entry.group_count, entry.rows, entry.columns)
        assert got == expected[0], \
            f"kind {entry.kind} layer {entry.layer}: {got} != {expected[0]}"
        assert entry.payload_bytes == validator.expected_payload_bytes(expected[0]), \
            f"kind {entry.kind} layer {entry.layer}: payload bytes drift"
        assert entry.scale_bytes == validator.expected_scale_bytes(expected[0]), \
            f"kind {entry.kind} layer {entry.layer}: scale bytes drift"
        if entry.layer == validator.GLOBAL_LAYER:
            seen_global |= 1 << entry.kind
        else:
            seen_layer.setdefault(entry.layer, 0)
            seen_layer[entry.layer] |= 1 << entry.kind
    assert seen_global == ((1 << validator.K_EMBEDDING) |
                           (1 << validator.K_FINAL_NORM) |
                           (1 << validator.K_LM_HEAD)), "global inventory drift"
    for layer in range(78):
        expected = 0
        for kind in range(3, validator.KIND_COUNT):
            shape, _error = validator.expected_shape(kind, layer, tp_degree)
            if shape is not None:
                expected |= 1 << kind
        assert seen_layer.get(layer, 0) == expected, \
            f"layer {layer} inventory drift at tp{tp_degree}"
    total_bytes = sum(item.entry.payload_bytes + item.entry.scale_bytes
                      for item in instance.plan)
    print(f"PASS plan mirrors validator policy at tp{tp_degree} "
          f"({len(instance.plan)} tensors, {total_bytes} payload+scale bytes)")


def check_tp_slices_partition():
    """Sharded kinds' payload+scale bytes must add back to the tp1 totals;
    replicated kinds must be identical on every rank."""
    sharded = validator.TP_SHARDS_ROWS | validator.TP_SHARDS_COLS
    instance_tp1 = build_entries(1, 0)
    for layer in (0, 10):
        kinds = sorted({item.entry.kind for item in instance_tp1.plan
                        if item.entry.layer == layer})
        for kind in kinds:
            base = next(item.entry for item in instance_tp1.plan
                        if item.entry.kind == kind and item.entry.layer == layer)
            total = 0
            identical = True
            for rank in range(8):
                instance = packer.Packer(MockSource(), CONTRACT, (0, 77),
                                         True, True, 8, rank)
                instance.build_plan()
                entry = next(item.entry for item in instance.plan
                             if item.entry.kind == kind and item.entry.layer == layer)
                total += entry.payload_bytes + entry.scale_bytes
                identical &= (entry.payload_bytes == base.payload_bytes and
                              entry.scale_bytes == base.scale_bytes)
            if kind in sharded:
                assert total == base.payload_bytes + base.scale_bytes, \
                    f"layer {layer} kind {kind}: tp8 rank sums {total} != tp1 " \
                    f"{base.payload_bytes + base.scale_bytes}"
            else:
                assert identical, \
                    f"layer {layer} kind {kind}: replicated bytes differ across ranks"
    print("PASS tp8 sharded kinds partition tp1 bytes and replicated kinds are identical "
          "(all kinds, layers 0 and 10)")


def main() -> int:
    check_lut()
    check_bf16_rounding()
    check_reader()
    check_plan_against_validator(1)
    check_plan_against_validator(8)
    check_tp_slices_partition()
    print("PASS glm52 fp8-source packer unit checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
