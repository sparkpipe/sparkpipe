#!/usr/bin/env python3
"""Hermetic checks for the bf16 VERBATIM branch of the GLM52 resident
stage packer (glm53full lane, 2026-08-29 — the native-precision arm).

Covers (no checkpoint, no torch, no safetensors lib):
  1. Fp8SourceReader over a synthetic official-BF16 store: BF16 expert
     payload verbatim slicing (row + column), rejection of a non-BF16
     expert tensor (a wrong-source pack must fail, not serve),
  2. every plan entry at --expert-codec bf16 (tp=1 and tp=16) matches the
     C-module policy mirror in tools/glm52_validate_pack.py (codec 1,
     PAYLOAD_BF16, SCALE_NONE, no scale plane),
  3. tp16 sharded expert bytes partition the tp1 totals exactly (bf16
     experts carry NO replicated scale plane) and replicated kinds are
     identical on every rank,
  4. the bf16 pack header round-trips through the validator's
     unsupported-codec gate BEFORE this lane's validator extension and is
     accepted after it (codec 1 == the spine codec).
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


def write_synthetic_bf16_store(root: Path, experts: int = 256):
    """Mini official-BF16 style store: BF16 expert matrices, NO scale
    tensors at all (16-row, 32-col experts)."""
    header = {}
    blobs = []
    offset = 0

    def add(name, dtype, shape, array_bytes):
        nonlocal offset
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(array_bytes)]}
        blobs.append(array_bytes)
        offset += len(array_bytes)

    payloads = {}
    for expert in range(experts):
        for proj in ("up_proj", "gate_proj", "down_proj"):
            cols = 32 if proj in ("up_proj", "gate_proj") else 16
            matrix = ((np.arange(16 * cols, dtype=np.uint32)
                       + expert * 131 + {"up_proj": 0, "gate_proj": 7,
                                         "down_proj": 13}[proj])
                      .astype(np.uint16).reshape(16, cols))
            base = f"model.layers.5.mlp.experts.{expert}.{proj}"
            add(f"{base}.weight", "BF16", [16, cols], matrix.tobytes())
            if proj != "down_proj":
                payloads[(expert, proj)] = matrix
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
    return payloads, bf16


def check_reader():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payloads, bf16 = write_synthetic_bf16_store(root)
        reader = packer.Fp8SourceReader(root)

        matrix = reader.spine_bf16("model.embed_tokens.weight")
        assert matrix.dtype == np.uint16 and np.array_equal(matrix, bf16), \
            "BF16 spine passthrough drift"

        name = "model.layers.5.mlp.experts.3.up_proj.weight"
        assert reader.bf16_expert_payload(name, 0, 16, 0, 32) == \
            payloads[(3, "up_proj")].tobytes(), "full payload drift"
        assert reader.bf16_expert_payload(name, 8, 16, 0, 32) == \
            payloads[(3, "up_proj")][8:].tobytes(), "row payload slice drift"
        assert reader.bf16_expert_payload(name, 0, 16, 16, 32) == \
            payloads[(3, "up_proj")][:, 16:].tobytes(), "col payload slice drift"

        # wrong source dtype must FAIL the pack (fp8 tensor in a bf16 build)
        fp8_reader = packer.Fp8SourceReader(root)
        assert fp8_reader.meta(name)[0] == "BF16"  # populate the lazy header cache
        fp8_reader._headers[fp8_reader.weight_map[name]][name]["dtype"] = "F8_E4M3"
        try:
            fp8_reader.bf16_expert_payload(name, 0, 16, 0, 32)
            raise AssertionError("non-BF16 expert tensor accepted")
        except packer.PackFailure:
            pass
        reader.close()
        fp8_reader.close()
    print("PASS bf16 reader (verbatim payload slices, dtype gate)")


class MockBf16Source:
    """Name-driven shape oracle over the official-BF16 layout (plan only)."""

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
        if ".mlp.experts." in name:
            tail = name.split(".mlp.experts.")[1].split(".", 2)[2]
            expert, proj = tail.split(".")
            rows, cols = self.EXPERT[proj]
            return "BF16", (rows, cols), "mock"
        for key, value in self.SPINE.items():
            if name == key or name.endswith("." + key):
                return value[0], value[1], "mock"
        raise AssertionError(f"mock source has no shape for {name}")

    def close(self):
        pass


CONTRACT = json.loads((ROOT / "model_contracts" / "glm52.json").read_text())


def build_entries(tp_degree: int, tp_rank: int):
    instance = packer.Packer(MockBf16Source(), CONTRACT, (0, 77), True, True,
                             tp_degree, tp_rank, packer.CODEC_BF16)
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
                                            packer.CODEC_BF16)
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
        if entry.weight_codec == packer.CODEC_BF16 and \
                entry.kind in (validator.K_EXPERT_UP_GATE, validator.K_EXPERT_DOWN):
            expert_entries += 1
            assert entry.payload_type == validator.PAYLOAD_BF16, "expert payload type"
            assert entry.scale_encoding == validator.SCALE_NONE, "bf16 expert scale encoding"
            assert entry.scale_bytes == 0, "bf16 experts must carry no scale plane"
        if entry.layer == validator.GLOBAL_LAYER:
            seen_global |= 1 << entry.kind
        else:
            seen_layer.setdefault(entry.layer, 0)
            seen_layer[entry.layer] |= 1 << entry.kind
    assert expert_entries == 2 * 75, f"expected 150 bf16 expert entries, {expert_entries}"
    assert seen_global == ((1 << validator.K_EMBEDDING) |
                           (1 << validator.K_FINAL_NORM) |
                           (1 << validator.K_LM_HEAD)), "global inventory drift"
    for layer in range(78):
        expected = 0
        for kind in range(3, validator.KIND_COUNT):
            shape, _error = validator.expected_shape(kind, layer, tp_degree,
                                                     packer.CODEC_BF16)
            if shape is not None:
                expected |= 1 << kind
        assert seen_layer.get(layer, 0) == expected, \
            f"layer {layer} inventory drift at tp{tp_degree}"
    total_bytes = sum(item.entry.payload_bytes + item.entry.scale_bytes
                      for item in instance.plan)
    print(f"PASS plan mirrors validator policy (bf16) at tp{tp_degree} "
          f"({len(instance.plan)} tensors, {total_bytes} payload+scale bytes)")
    return instance


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
                instance = packer.Packer(MockBf16Source(), CONTRACT, (0, 77),
                                         True, True, 16, rank, packer.CODEC_BF16)
                instance.build_plan()
                entry = next(item.entry for item in instance.plan
                             if item.entry.kind == kind and item.entry.layer == layer)
                total += entry.payload_bytes + entry.scale_bytes
                identical &= (entry.payload_bytes == base.payload_bytes and
                              entry.scale_bytes == base.scale_bytes)
            if kind in sharded:
                # bf16 experts carry NO scale plane, so the tp16 rank sums
                # partition the tp1 bytes exactly.
                assert total == base.payload_bytes + base.scale_bytes, \
                    f"layer {layer} kind {kind}: tp16 rank sums {total} != tp1 " \
                    f"{base.payload_bytes + base.scale_bytes}"
            else:
                assert identical, \
                    f"layer {layer} kind {kind}: replicated bytes differ across ranks"
    print("PASS tp16 sharded kinds partition tp1 bytes exactly (no replicated "
          "plane) and replicated kinds are identical (layers 0 and 10)")


def main() -> int:
    check_reader()
    check_plan_against_validator(1)
    check_plan_against_validator(16)
    check_tp16_slices_partition()
    print("PASS glm52 bf16-passthrough packer unit checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
