#!/usr/bin/env python3
"""Hermetic checks for the nvfp4 VERBATIM branch of the GLM52 resident
stage packer (glm53full lane, 2026-08-29).

Covers (no checkpoint, no torch, no safetensors lib):
  1. Fp8SourceReader over a synthetic modelopt-NVFP4 store: packed-e2m1
     payload verbatim slicing (row + nibble-aligned column), UE4M3
     per-16 block-scale slicing, scalar F32 global read, BF16 spine
     passthrough,
  2. the up/gate per-expert global-share invariant the fused wire entry
     depends on: equal globals pass, a disagreeing expert fails the pack,
  3. every plan entry at --expert-codec nvfp4 (tp=1 and tp=16, all
     ranks) matches the C-module policy mirror in
     tools/glm52_validate_pack.py (codec 6, scale encoding
     ue4m3_f32_global, payload 4-bit, global+block scale bytes),
  4. tp16 sharded expert bytes partition the tp1 totals and replicated
     kinds are identical on every rank.
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


packer = load_module(TOOLS / "glm52_resident_stagepack.py",
                     "sparkpipe_glm52_resident_stagepack")
validator = load_module(TOOLS / "glm52_validate_pack.py",
                        "sparkpipe_glm52_validate_pack")


def write_synthetic_nvfp4_store(root: Path, experts: int = 256,
                                globals_mismatch_at: int | None = None):
    """Mini modelopt-NVFP4 style store: 256-entry loops are exercised by
    the plan checks, so the store keeps 8 tiny experts (16-row, 32-col
    experts: U8 [16,16], F8_E4M3 scale [16,2], F32 scalar global)."""
    header = {}
    blobs = []
    offset = 0

    def add(name, dtype, shape, array_bytes):
        nonlocal offset
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(array_bytes)]}
        blobs.append(array_bytes)
        offset += len(array_bytes)

    payloads, scales, globals_ = {}, {}, {}
    for expert in range(experts):
        payload = ((np.arange(16 * 16, dtype=np.uint32) + expert * 37)
                   .astype(np.uint8).reshape(16, 16))
        scale = ((np.arange(16 * 2, dtype=np.uint32) + expert * 5)
                 .astype(np.uint8).reshape(16, 2) % 254)
        scale[0, 0] = 0x38  # 1.0 anchor
        value = np.float32(7.5e-5) * (1 + expert)
        for proj in ("up_proj", "gate_proj", "down_proj"):
            base = f"model.layers.5.mlp.experts.{expert}.{proj}"
            add(f"{base}.weight", "U8", [16, 16], payload.tobytes())
            add(f"{base}.weight_scale", "F8_E4M3", [16, 2], scale.tobytes())
            shared = value if proj in ("up_proj", "gate_proj") else value * 2
            if (proj in ("up_proj", "gate_proj")
                    and globals_mismatch_at == expert and proj == "gate_proj"):
                shared = value * 3
            add(f"{base}.weight_scale_2", "F32", [], struct.pack("<f", shared))
            if proj == "down_proj":
                continue
            payloads[(expert, proj)] = payload
            scales[(expert, proj)] = scale
            globals_[(expert, proj)] = shared
    bf16 = (np.arange(4 * 6, dtype=np.uint16) * 257 + 1).reshape(4, 6)
    add("model.embed_tokens.weight", "BF16", [4, 6], bf16.tobytes())
    header_bytes = json.dumps(header).encode()
    with (root / "model-00001-of-00001.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model-00001-of-00001.safetensors"
                        for name in header}}))
    (root / "config.json").write_text(json.dumps({"architectures": ["TestModel"]}))
    return payloads, scales, globals_, bf16


def check_reader():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payloads, scales, globals_, bf16 = write_synthetic_nvfp4_store(root)
        reader = packer.Fp8SourceReader(root)

        # BF16 spine passthrough stays byte-exact on the nvfp4 path too
        matrix = reader.spine_bf16("model.embed_tokens.weight")
        assert matrix.dtype == np.uint16 and np.array_equal(matrix, bf16), \
            "BF16 spine passthrough drift"

        # payload: verbatim rows, then nibble-aligned columns
        name = "model.layers.5.mlp.experts.3.up_proj.weight"
        assert reader.nvfp4_payload(name, 0, 16, 0, 32) == \
            payloads[(3, "up_proj")].tobytes(), "full payload drift"
        assert reader.nvfp4_payload(name, 8, 16, 0, 32) == \
            payloads[(3, "up_proj")][8:].tobytes(), "row payload slice drift"
        assert reader.nvfp4_payload(name, 0, 16, 16, 32) == \
            payloads[(3, "up_proj")][:, 8:].tobytes(), "col payload slice drift"
        try:
            reader.nvfp4_payload(name, 0, 16, 1, 32)
            raise AssertionError("odd column slice accepted")
        except packer.PackFailure:
            pass

        # block scales: one UE4M3 byte per 16 elements
        got = np.frombuffer(reader.nvfp4_block_scales(name, 0, 16, 0, 32),
                            dtype=np.uint8).reshape(16, 2)
        assert np.array_equal(got, scales[(3, "up_proj")]), "block scale drift"
        got = np.frombuffer(reader.nvfp4_block_scales(name, 4, 8, 16, 32),
                            dtype=np.uint8).reshape(4, 1)
        assert np.array_equal(got, scales[(3, "up_proj")][4:8, 1:]), \
            "block scale slice drift"

        # global: scalar F32, little-endian exact
        value = struct.unpack("<f", reader.nvfp4_global(name))[0]
        assert value == globals_[(3, "up_proj")], "global scale drift"

        # fused-entry globals: 256 experts, one global each (up == gate)
        fused = reader.nvfp4_expert_globals(5, ["up_proj", "gate_proj"])
        assert len(fused) == 256 * 4, "fused globals byte count"
        got = np.frombuffer(fused, dtype=np.float32)
        assert np.array_equal(got, [globals_[(e, "up_proj")] for e in range(256)]), \
            "fused global values drift"

        # a disagreeing expert must FAIL the pack, not serve wrong weights
        with tempfile.TemporaryDirectory() as bad_tmp:
            bad_root = Path(bad_tmp)
            write_synthetic_nvfp4_store(bad_root, globals_mismatch_at=7)
            bad_reader = packer.Fp8SourceReader(bad_root)
            try:
                bad_reader.nvfp4_expert_globals(5, ["up_proj", "gate_proj"])
                raise AssertionError("mismatched up/gate global accepted")
            except packer.PackFailure as error:
                assert "disagree" in str(error), str(error)
        reader.close()
    print("PASS nvfp4 reader (verbatim payload slices, block scales, "
          "scalar global, up/gate share enforcement)")


class MockNvfp4Source:
    """Name-driven shape oracle over the radixark layout (plan only)."""

    SPINE = {
        "model.embed_tokens.weight": ("BF16", (154880, 6144)),
        "lm_head.weight": ("BF16", (154880, 6144)),
        "model.norm.weight": ("BF16", (6144,)),
        "input_layernorm.weight": ("BF16", (6144,)),
        "post_attention_layernorm.weight": ("BF16", (6144,)),
        "q_a_layernorm.weight": ("BF16", (2048,)),
        "kv_a_layernorm.weight": ("BF16", (512,)),
        "q_a_proj.weight": ("BF16", (2048, 6144)),
        "q_b_proj.weight": ("BF16", (16384, 2048)),
        "kv_a_proj_with_mqa.weight": ("BF16", (576, 6144)),
        "kv_b_proj.weight": ("BF16", (28672, 512)),
        "o_proj.weight": ("BF16", (6144, 16384)),
        "indexer.wq_b.weight": ("BF16", (4096, 2048)),
        "indexer.wk.weight": ("BF16", (128, 6144)),
        "indexer.weights_proj.weight": ("BF16", (32, 6144)),
        "indexer.k_norm.weight": ("BF16", (128,)),
        "indexer.k_norm.bias": ("BF16", (128,)),
        "mlp.gate_proj.weight": ("BF16", (12288, 6144)),
        "mlp.up_proj.weight": ("BF16", (12288, 6144)),
        "mlp.down_proj.weight": ("BF16", (6144, 12288)),
        "mlp.gate.weight": ("BF16", (256, 6144)),
        "mlp.gate.e_score_correction_bias": ("F32", (256,)),
        "shared_experts.gate_proj.weight": ("BF16", (2048, 6144)),
        "shared_experts.up_proj.weight": ("BF16", (2048, 6144)),
        "shared_experts.down_proj.weight": ("BF16", (6144, 2048)),
    }
    EXPERT = {
        "up_proj": (2048, 6144),
        "gate_proj": (2048, 6144),
        "down_proj": (6144, 2048),
    }

    def meta(self, name: str):
        if name.endswith("_scale_2"):
            return "F32", (), "mock"
        if name.endswith("_scale"):
            base = name[:-len("_scale")]
            dtype, (rows, cols), _ = self.meta(base)
            return "F8_E4M3", (rows, cols // 16), "mock"
        if ".mlp.experts." in name:
            tail = name.split(".mlp.experts.")[1].split(".", 2)[2]
            expert, proj = tail.split(".")
            rows, cols = self.EXPERT[proj]
            return "U8", (rows, cols // 2), "mock"
        for key, value in self.SPINE.items():
            if name == key or name.endswith("." + key):
                return value[0], value[1], "mock"
        raise AssertionError(f"mock source has no shape for {name}")

    def close(self):
        pass


CONTRACT = json.loads((ROOT / "model_contracts" / "glm52.json").read_text())


def build_entries(tp_degree: int, tp_rank: int):
    instance = packer.Packer(MockNvfp4Source(), CONTRACT, (0, 77), True, True,
                             tp_degree, tp_rank, packer.CODEC_NVFP4)
    instance.build_plan()
    return instance


def check_plan_against_validator(tp_degree: int):
    seen_global = 0
    seen_layer: dict = {}
    instance = build_entries(tp_degree, 0)
    expert_entries = 0
    for item in instance.plan:
        entry = item.entry
        expected = validator.expected_shape(entry.kind, entry.layer, tp_degree,
                                            packer.CODEC_NVFP4)
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
        if entry.weight_codec == packer.CODEC_NVFP4:
            expert_entries += 1
        if entry.layer == validator.GLOBAL_LAYER:
            seen_global |= 1 << entry.kind
        else:
            seen_layer.setdefault(entry.layer, 0)
            seen_layer[entry.layer] |= 1 << entry.kind
    assert expert_entries == 2 * 75, f"expected 150 nvfp4 entries, {expert_entries}"
    assert seen_global == ((1 << validator.K_EMBEDDING) |
                           (1 << validator.K_FINAL_NORM) |
                           (1 << validator.K_LM_HEAD)), "global inventory drift"
    for layer in range(78):
        expected = 0
        for kind in range(3, validator.KIND_COUNT):
            shape, _error = validator.expected_shape(kind, layer, tp_degree,
                                                     packer.CODEC_NVFP4)
            if shape is not None:
                expected |= 1 << kind
        assert seen_layer.get(layer, 0) == expected, \
            f"layer {layer} inventory drift at tp{tp_degree}"
    total_bytes = sum(item.entry.payload_bytes + item.entry.scale_bytes
                      for item in instance.plan)
    print(f"PASS plan mirrors validator policy (nvfp4) at tp{tp_degree} "
          f"({len(instance.plan)} tensors, {total_bytes} payload+scale bytes)")


def check_tp16_slices_partition():
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
            for rank in range(16):
                instance = packer.Packer(MockNvfp4Source(), CONTRACT, (0, 77),
                                         True, True, 16, rank, packer.CODEC_NVFP4)
                instance.build_plan()
                entry = next(item.entry for item in instance.plan
                             if item.entry.kind == kind and item.entry.layer == layer)
                total += entry.payload_bytes + entry.scale_bytes
                identical &= (entry.payload_bytes == base.payload_bytes and
                              entry.scale_bytes == base.scale_bytes)
            if kind in sharded:
                # nvfp4 expert entries carry the per-expert F32 global
                # plane (group_count * 4B) on EVERY rank by design - the
                # module validates it against the unsharded group count.
                replicated = (256 * 4 if base.payload_type ==
                              validator.PAYLOAD_PACKED and
                              base.weight_codec == packer.CODEC_NVFP4 else 0)
                assert total == base.payload_bytes + base.scale_bytes + 15 * replicated, \
                    f"layer {layer} kind {kind}: tp16 rank sums {total} != tp1 " \
                    f"{base.payload_bytes + base.scale_bytes} (+15x replicated {replicated})"
            else:
                assert identical, \
                    f"layer {layer} kind {kind}: replicated bytes differ across ranks"
    print("PASS tp16 sharded kinds partition tp1 bytes and replicated kinds "
          "are identical (all kinds, layers 0 and 10)")


def main() -> int:
    check_reader()
    check_plan_against_validator(1)
    check_plan_against_validator(16)
    check_tp16_slices_partition()
    print("PASS glm52 nvfp4-passthrough packer unit checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
