#!/usr/bin/env python3
"""Pack the DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2) into a qwen36 wire pack.

Setup-time code, never the serving path. The drafter is a 5-layer sliding-attention
decoder (no GDN, no MTP) that emits an 8-token block; it SHARES the target's token
embedding and lm_head (the safetensors carries neither). This tool streams the 81
drafter tensors into the same wire format the qwen36 target packer writes, so the
resident module can load both with one header reader.

Drafter geometry (config.json): hidden 5120, 5 layers, 32 Q / 8 KV heads x head_dim
128, FFN intermediate 17408, vocab 248320, block_size 8, target taps {5,19,33,47,61},
mask token 248070, selector_rank 256, selector_top_k 16, conv_kernel_size 2,
conv_group_size 16, sliding_window 2048, is_causal false. No confidence head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
import sys
import tempfile

MAGIC = 0x50533651          # 'Q6SP', same as the target pack
FORMAT_VERSION = 3
HEADER_BYTES = 120
ENTRY_BYTES = 56
PAYLOAD_ALIGNMENT = 256
WEIGHT_BF16 = 0
BF16_BYTES = 2

HIDDEN = 5120
LAYER_COUNT = 5
ATTN_QUERY_HEADS = 32
ATTN_KV_HEADS = 8
ATTN_HEAD_DIM = 128
ATTN_ROPE_DIM = 64
FFN_INTERMEDIATE = 17408
VOCAB = 248320
BLOCK_SIZE = 8
TARGET_TAP_LAYERS = (5, 19, 33, 47, 61)
TAP_COUNT = len(TARGET_TAP_LAYERS)
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
CONV_KERNEL_SIZE = 2
CONV_GROUP_SIZE = 16
SLIDING_WINDOW = 2048
MASK_TOKEN_ID = 248070
CONV_SIDES = 2
CONV_PROJ_ROWS = CONV_SIDES * CONV_KERNEL_SIZE * (HIDDEN // CONV_GROUP_SIZE)  # 1280

HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

# DFlash2 tensor kinds (per-layer + global). Mirror SparkQwen36DsparkTensorKind.
(KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT,
 KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM, KIND_ATTENTION_NORM, KIND_MLP_NORM,
 KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN,
 KIND_PROJECTOR, KIND_SELECTOR_PRED, KIND_SELECTOR_SUCC, KIND_SELECTOR_HIDDEN_PROJ,
 KIND_FINAL_NORM, KIND_HIDDEN_NORM,
 KIND_CONV_ATTN_BASE, KIND_CONV_ATTN_PROJ, KIND_CONV_MLP_BASE, KIND_CONV_MLP_PROJ) = range(21)

PER_LAYER_KINDS = (KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT,
                   KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM, KIND_ATTENTION_NORM,
                   KIND_MLP_NORM, KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN,
                   KIND_CONV_ATTN_BASE, KIND_CONV_ATTN_PROJ,
                   KIND_CONV_MLP_BASE, KIND_CONV_MLP_PROJ)


def kind_shape(kind: int) -> tuple[int, int]:
    table = {
        KIND_ATTN_QUERY: (ATTN_QUERY_HEADS * ATTN_HEAD_DIM, HIDDEN),
        KIND_ATTN_KEY: (ATTN_KV_HEADS * ATTN_HEAD_DIM, HIDDEN),
        KIND_ATTN_VALUE: (ATTN_KV_HEADS * ATTN_HEAD_DIM, HIDDEN),
        KIND_ATTN_OUTPUT: (HIDDEN, ATTN_QUERY_HEADS * ATTN_HEAD_DIM),
        KIND_ATTN_QUERY_NORM: (1, ATTN_HEAD_DIM),
        KIND_ATTN_KEY_NORM: (1, ATTN_HEAD_DIM),
        KIND_ATTENTION_NORM: (1, HIDDEN),
        KIND_MLP_NORM: (1, HIDDEN),
        KIND_FFN_GATE: (FFN_INTERMEDIATE, HIDDEN),
        KIND_FFN_UP: (FFN_INTERMEDIATE, HIDDEN),
        KIND_FFN_DOWN: (HIDDEN, FFN_INTERMEDIATE),
        KIND_PROJECTOR: (HIDDEN, TAP_COUNT * HIDDEN),
        KIND_SELECTOR_PRED: (VOCAB, SELECTOR_RANK),
        KIND_SELECTOR_SUCC: (VOCAB, SELECTOR_RANK),
        KIND_SELECTOR_HIDDEN_PROJ: (SELECTOR_RANK, HIDDEN),
        KIND_FINAL_NORM: (1, HIDDEN),
        KIND_HIDDEN_NORM: (1, HIDDEN),
        # base_kernel [2,2,5120] (sides, taps, channels) -> [sides, taps*channels] = [2, 10240]
        KIND_CONV_ATTN_BASE: (CONV_SIDES, CONV_KERNEL_SIZE * HIDDEN),
        KIND_CONV_ATTN_PROJ: (CONV_PROJ_ROWS, HIDDEN),
        KIND_CONV_MLP_BASE: (CONV_SIDES, CONV_KERNEL_SIZE * HIDDEN),
        KIND_CONV_MLP_PROJ: (CONV_PROJ_ROWS, HIDDEN),
    }
    return table[kind]


_LAYER_NAMES = {
    KIND_ATTN_QUERY: "self_attn.q_proj.weight",
    KIND_ATTN_KEY: "self_attn.k_proj.weight",
    KIND_ATTN_VALUE: "self_attn.v_proj.weight",
    KIND_ATTN_OUTPUT: "self_attn.o_proj.weight",
    KIND_ATTN_QUERY_NORM: "self_attn.q_norm.weight",
    KIND_ATTN_KEY_NORM: "self_attn.k_norm.weight",
    KIND_ATTENTION_NORM: "input_layernorm.weight",
    KIND_MLP_NORM: "post_attention_layernorm.weight",
    KIND_FFN_GATE: "mlp.gate_proj.weight",
    KIND_FFN_UP: "mlp.up_proj.weight",
    KIND_FFN_DOWN: "mlp.down_proj.weight",
    KIND_CONV_ATTN_BASE: "attention_conv.base_kernel",
    KIND_CONV_ATTN_PROJ: "attention_conv.kernel_projection.weight",
    KIND_CONV_MLP_BASE: "mlp_conv.base_kernel",
    KIND_CONV_MLP_PROJ: "mlp_conv.kernel_projection.weight",
}
GLOBAL_NAMES = {
    KIND_PROJECTOR: "fc.weight",
    KIND_SELECTOR_PRED: "candidate_selector.predecessor_codebook",
    KIND_SELECTOR_SUCC: "candidate_selector.successor_codebook",
    KIND_SELECTOR_HIDDEN_PROJ: "candidate_selector.hidden_projection.weight",
    KIND_FINAL_NORM: "norm.weight",
    KIND_HIDDEN_NORM: "hidden_norm.weight",
}


class PackFailure(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SafetensorsHeader:
    """Minimal safetensors reader: the 8-byte length + JSON header + payload map."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(n)
            self.meta = json.loads(raw)
            self.header_len = 8 + n
            if "__metadata__" in self.meta:
                self.meta.pop("__metadata__")
        self.names = list(self.meta)

    def offset(self, name: str) -> tuple[int, int]:
        start, end = self.meta[name]["data_offsets"]
        return self.header_len + start, end - start


def build_inventory() -> list[tuple[int, int, str]]:
    """[(kind, layer, name), ...] in pack order. layer == 0xFFFFFFFF for globals."""
    inv = []
    for layer in range(LAYER_COUNT):
        for kind in PER_LAYER_KINDS:
            inv.append((kind, layer, f"layers.{layer}.{_LAYER_NAMES[kind]}"))
    for kind, name in GLOBAL_NAMES.items():
        inv.append((kind, 0xFFFFFFFF, name))
    return inv


def pack(checkpoint: Path, output: Path, receipt: dict) -> dict:
    src = SafetensorsHeader(checkpoint / "model.safetensors")
    expected = set(inv[2] for inv in build_inventory())
    missing = expected - set(src.names)
    if missing:
        raise PackFailure(f"missing drafter tensors: {sorted(missing)}")

    plans = []
    cursor = 0
    for kind, layer, name in build_inventory():
        offset, nbytes = src.offset(name)
        rows, cols = kind_shape(kind)
        payload_offset = (cursor + PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)
        payload_bytes = rows * cols * BF16_BYTES
        if nbytes != payload_bytes:
            raise PackFailure(f"size mismatch {name}: header {nbytes} vs plan {payload_bytes}")
        plans.append((kind, layer, name, payload_offset, payload_bytes, offset, rows, cols))
        cursor = payload_offset + payload_bytes

    payload_base = (HEADER_BYTES + len(plans) * ENTRY_BYTES + PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)
    file_bytes = payload_base + cursor

    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plans),
        HIDDEN, LAYER_COUNT, 0, LAYER_COUNT,
        1, 0,                       # attention period 1, full phase 0 (no GDN)
        0, 0, 0, 0, 0,              # GDN heads/dims/kernel = 0
        ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM, ATTN_ROPE_DIM,
        FFN_INTERMEDIATE, VOCAB, BLOCK_SIZE, TAP_COUNT,
        1, 0, HEADER_BYTES, file_bytes)
    entries = b"".join(
        ENTRY_STRUCT.pack(kind, layer, WEIGHT_BF16, rows, cols, 0,
                          payload_base + payload_offset, payload_bytes, 0, 0)
        for kind, layer, name, payload_offset, payload_bytes, _, rows, cols in plans)

    receipt.update({
        "layer_count": LAYER_COUNT,
        "tensor_count": len(plans),
        "bytes": file_bytes,
        "tp_degree": 1,
        "tp_rank": 0,
        "block_size": BLOCK_SIZE,
        "target_tap_layers": list(TARGET_TAP_LAYERS),
        "selector_rank": SELECTOR_RANK,
        "selector_top_k": SELECTOR_TOP_K,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "conv_group_size": CONV_GROUP_SIZE,
        "sliding_window": SLIDING_WINDOW,
        "mask_token_id": MASK_TOKEN_ID,
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp",
                                     dir=output.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(header)
        temp.write(entries)
        temp.write(b"\0" * (payload_base - temp.tell()))
        for kind, layer, name, payload_offset, payload_bytes, src_offset, _, _ in plans:
            with open(checkpoint / "model.safetensors", "rb") as f:
                f.seek(src_offset)
                payload = f.read(payload_bytes)
            temp.write(payload)
            pad = ((temp.tell() + PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)) - temp.tell()
            if pad:
                temp.write(b"\0" * pad)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    return receipt


def verify(pack_path: Path) -> dict:
    with open(pack_path, "rb") as f:
        raw = f.read(HEADER_BYTES)
        fields = HEADER_STRUCT.unpack(raw)
        (magic, version, header_bytes, entry_bytes, tensor_count, hidden, layers,
         first, total, period, phase, gk, gv, gkd, gvd, gck, aq, akv, ahd, arope,
         ffn, vocab, block, taps, tp_deg, tp_rank, _, file_bytes) = fields
        assert magic == MAGIC and version == FORMAT_VERSION
        assert hidden == HIDDEN and layers == LAYER_COUNT
        assert aq == ATTN_QUERY_HEADS and akv == ATTN_KV_HEADS and ahd == ATTN_HEAD_DIM
        assert ffn == FFN_INTERMEDIATE and vocab == VOCAB
        assert block == BLOCK_SIZE and taps == TAP_COUNT
        entries = [ENTRY_STRUCT.unpack_from(f.read(ENTRY_BYTES)) for _ in range(tensor_count)]
    return {"ok": True, "tensor_count": tensor_count, "bytes": file_bytes,
            "entries": len(entries)}


def verify_payload(pack_path: Path, checkpoint: Path) -> bool:
    """Byte-for-byte round-trip: every packed payload equals its safetensors tensor."""
    src = SafetensorsHeader(checkpoint / "model.safetensors")
    with open(pack_path, "rb") as f:
        header = f.read(HEADER_BYTES)
        tensor_count = HEADER_STRUCT.unpack(header)[4]
        f.seek(HEADER_BYTES)
        entries = [ENTRY_STRUCT.unpack(f.read(ENTRY_BYTES)) for _ in range(tensor_count)]
    inv = build_inventory()
    ok = True
    for (kind, layer, name), entry in zip(inv, entries):
        e_kind, e_layer, e_fmt, e_rows, e_cols, e_sg, p_off, p_bytes, _, _ = entry
        rows, cols = kind_shape(kind)
        assert e_kind == kind and e_rows == rows and e_cols == cols, name
        src_off, src_bytes = src.offset(name)
        with open(checkpoint / "model.safetensors", "rb") as sf, open(pack_path, "rb") as pf:
            sf.seek(src_off)
            a = sf.read(src_bytes)
            pf.seek(p_off)
            b = pf.read(p_bytes)
        if a != b:
            print(f"payload mismatch {name}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="dir holding the drafter model.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = {"kind": "sparkpipe.qwen36.dflash2-stagepack-receipt.v1",
               "tool": "tools/qwen36_dspark_stagepack.py"}
    try:
        pack(args.checkpoint, args.output, receipt)
    except PackFailure as e:
        print(f"qwen36_dspark_stagepack failed: {e}", file=sys.stderr)
        return 1
    v = verify(args.output)
    if not verify_payload(args.output, args.checkpoint):
        print("qwen36_dspark_stagepack round-trip FAILED", file=sys.stderr)
        return 1
    print(f"qwen36_dspark_stagepack wrote {args.output} tensors={v['tensor_count']} "
          f"file_gib={v['bytes'] / 2**30:.2f} round_trip=ok")
    with open(str(args.output) + ".receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
