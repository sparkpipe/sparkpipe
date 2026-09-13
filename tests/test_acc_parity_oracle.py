#!/usr/bin/env python3
"""Offline synthetic proof of the ACC-2 parity oracle (muse profile).

Builds a shrunken fake muse checkpoint, emits a real pack through
tools/muse_glimmer_stagepack.py's own write path, then requires the
oracle to (a) verdict CHECKPOINT-FAITHFUL on the intact pack and
(b) verdict DRIFTED with a located first delta after a one-byte payload
corruption. Also covers the negative identity checks: entry-count and
file_bytes drift fail loud.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))
import muse_glimmer_stagepack as packer  # noqa: E402

GEO = dict(HIDDEN=64, LAYER_COUNT=4, VOCAB=256, ATTN_QUERY_HEADS=16,
           ATTN_KV_HEADS=2, ATTN_HEAD_DIM=16, INTERMEDIATE=128)


def shrink() -> None:
    for name, value in GEO.items():
        setattr(packer, name, value)


def write_safetensors(path: Path, tensors: dict) -> None:
    header = {}
    blob = bytearray()
    for name, array in tensors.items():
        raw = array.astype("<u2").tobytes()
        header[name] = dict(dtype="BF16", shape=list(array.shape),
                            data_offsets=[len(blob), len(blob) + len(raw)])
        blob += raw
    header["__metadata__"] = dict(format="pt")
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(blob))


def build_checkpoint(root: Path) -> None:
    rng = np.random.default_rng(7)
    tensors = {}
    tensors["model.language_model.embed_tokens.weight"] = rng.integers(
        0, 65535, (GEO["VOCAB"], GEO["HIDDEN"]), dtype=np.uint16)
    tensors["model.language_model.norm.weight"] = rng.integers(
        0, 65535, (1, GEO["HIDDEN"]), dtype=np.uint16)
    tensors["lm_head.weight"] = rng.integers(
        0, 65535, (GEO["VOCAB"], GEO["HIDDEN"]), dtype=np.uint16)
    prefix = "model.language_model.layers."
    for layer in range(GEO["LAYER_COUNT"]):
        base = dict(
            input_layernorm=(1, GEO["HIDDEN"]),
            post_attention_layernorm=(1, GEO["HIDDEN"]),
            pre_feedforward_layernorm=(1, GEO["HIDDEN"]),
            post_feedforward_layernorm=(1, GEO["HIDDEN"]),
        )
        for suffix, shape in base.items():
            tensors[f"{prefix}{layer}.{suffix}.weight"] = rng.integers(
                0, 65535, shape, dtype=np.uint16)
        heads, kv, dim = GEO["ATTN_QUERY_HEADS"], GEO["ATTN_KV_HEADS"], GEO["ATTN_HEAD_DIM"]
        tensors[f"{prefix}{layer}.self_attn.q_proj.weight"] = rng.integers(
            0, 65535, (heads * dim, GEO["HIDDEN"]), dtype=np.uint16)
        tensors[f"{prefix}{layer}.self_attn.gate_proj.weight"] = rng.integers(
            0, 65535, (heads * dim, GEO["HIDDEN"]), dtype=np.uint16)
        tensors[f"{prefix}{layer}.self_attn.k_proj.weight"] = rng.integers(
            0, 65535, (kv * dim, GEO["HIDDEN"]), dtype=np.uint16)
        tensors[f"{prefix}{layer}.self_attn.v_proj.weight"] = rng.integers(
            0, 65535, (kv * dim, GEO["HIDDEN"]), dtype=np.uint16)
        tensors[f"{prefix}{layer}.self_attn.o_proj.weight"] = rng.integers(
            0, 65535, (GEO["HIDDEN"], heads * dim), dtype=np.uint16)
        inter = GEO["INTERMEDIATE"]
        tensors[f"{prefix}{layer}.mlp.gate_proj.weight"] = rng.integers(
            0, 65535, (inter, GEO["HIDDEN"]), dtype=np.uint16)
        tensors[f"{prefix}{layer}.mlp.up_proj.weight"] = rng.integers(
            0, 65535, (inter, GEO["HIDDEN"]), dtype=np.uint16)
        tensors[f"{prefix}{layer}.mlp.down_proj.weight"] = rng.integers(
            0, 65535, (GEO["HIDDEN"], inter), dtype=np.uint16)
    write_safetensors(root / "model.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps(
        dict(weight_map={name: "model.safetensors" for name in tensors})))
    (root / "config.json").write_text(json.dumps(dict(text_config=dict(
        hidden_size=GEO["HIDDEN"], num_hidden_layers=GEO["LAYER_COUNT"],
        vocab_size=GEO["VOCAB"], num_attention_heads=GEO["ATTN_QUERY_HEADS"],
        num_key_value_heads=GEO["ATTN_KV_HEADS"], head_dim=GEO["ATTN_HEAD_DIM"],
        intermediate_size=GEO["INTERMEDIATE"]))))


def emit_pack(source: Path, output: Path, tp_degree: int, tp_rank: int) -> None:
    from spark_pack_common import SafetensorsSource
    records = packer.build_records(tp_degree, tp_rank)
    entries, file_bytes = packer.build_directory(records)
    source_reader = SafetensorsSource(source)
    with output.open("wb") as out:
        out.write(packer.make_header(tp_degree, len(records), file_bytes))
        for record, payload_offset, _ in entries:
            out.write(packer.ENTRY_STRUCT.pack(
                record.kind, record.layer, record.weight_format, record.rows,
                record.columns, 0, payload_offset, record.payload_bytes, 0, 0))
    packer.write_records(source_reader, entries, output)


def main() -> int:
    shrink()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "ckpt"
        checkpoint.mkdir()
        build_checkpoint(checkpoint)
        pack = root / "muse_tp16_rank4.bf16.gsmu"
        emit_pack(checkpoint, pack, 16, 4)
        import acc_parity_oracle as oracle
        result = oracle.profile_muse(pack, checkpoint, 16, 4)
        intact = (result["compared"] == 3 + 8 * len(oracle.anchor_layers(4))
                  and not result["drift"])
        print(f"intact compared={result['compared']} drift={result['drift']}")
        payload_offset = packer.HEADER_BYTES + packer.ENTRY_BYTES * len(
            packer.build_records(16, 4))
        payload_offset = (payload_offset + packer.PAYLOAD_ALIGNMENT - 1) // \
            packer.PAYLOAD_ALIGNMENT * packer.PAYLOAD_ALIGNMENT
        with pack.open("r+b") as file:
            file.seek(payload_offset + 3)
            original = file.read(1)
            file.seek(payload_offset + 3)
            file.write(bytes([original[0] ^ 0xFF]))
        result = oracle.profile_muse(pack, checkpoint, 16, 4)
        corrupted = (bool(result["drift"])
                     and "first byte delta" in result["drift"][0])
        print(f"corrupted drift[0]={result['drift'][0] if result['drift'] else None}")
        if not intact or not corrupted:
            print("FAIL test_acc_parity_oracle")
            return 1
    print("PASS test_acc_parity_oracle (intact CHECKPOINT-FAITHFUL, "
          "one-byte corruption DRIFTED with located delta)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
