#!/usr/bin/env python3
"""Pack a Kimi-K3 DSpark drafter checkpoint into a K3DS wire pack.

Setup-time code, never the serving path. The admitted DSpark drafter sources
(kimi-k3-dspark-{redhatai,inferact,radixark}) are DSparkDraftModel checkpoints:
a 5-layer Qwen3-shaped attention drafter that reads target hidden taps and emits
one block per round, with a low-rank Markov bias head and a confidence head.
This tool REPACKAGES the tensors as shipped (BF16 bytes move, never recomputed
- the quantization policy) into the K3DS wire layout the k3 module's drafter
bind reads (modules/k3_resident_decode_stage/source/spark_k3_dspark_format.h).

Wire shape: the qwen38_27b drafter pack's Q6SP v3 discipline - a fixed little-
endian header, 56-byte entries (6I4Q), 256-byte-aligned payloads - with the
header EXTENDED past the 26I2Q core by 16 more U32 fields, because the DSpark
sources vary per release (block 7 vs 8, different tap layers, mask id) and the
pack must be self-describing. magic 'K3DS', version 3.

Geometry is read from the source's config.json and every planned shape is
checked against the safetensors header before any byte is written: a source
whose layout the format does not carry fails LOUDLY naming the tensor, it does
not get silently repacked into a wrong-shaped slot.

redhatai (verified 2026-08-29, revision in DOWNLOAD-RECEIPT.json): hidden 7168,
5 layers, 96 Q / 16 KV heads x 64, FFN 14336, vocab 163840, block 8, taps
{24,48,72,88,92}, markov_rank 256, mask 163837, sliding_window 2048, rope theta
10000 default, confidence head over hidden+markov (7424), 64 tensors all BF16,
drafter SHIPS its own embed_tokens and lm_head (unlike DFlash2, which shares
the target's).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

MAGIC = 0x5344334B          # 'K3DS'
FORMAT_VERSION = 3          # shared with the Q6SP v3 wire discipline
CORE_HEADER_BYTES = 120     # the 26I2Q core, identical field order to Q6SP
EXT_HEADER_BYTES = 64       # 16 U32 DSpark extension fields
HEADER_BYTES = CORE_HEADER_BYTES + EXT_HEADER_BYTES
ENTRY_BYTES = 56
PAYLOAD_ALIGNMENT = 256
WEIGHT_BF16 = 0
BF16_BYTES = 2

CORE_STRUCT = struct.Struct("<26I2Q")
EXT_STRUCT = struct.Struct("<16I")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert (CORE_STRUCT.size == CORE_HEADER_BYTES and
        EXT_STRUCT.size == EXT_HEADER_BYTES and
        ENTRY_STRUCT.size == ENTRY_BYTES)

# DSpark K3 tensor kinds. 0..16 mirror the qwen38_27b DFlash2 kind numbering
# (SparkQwen38_27bDsparkTensorKind) so the two drafter readers stay legible
# side by side: 11 is the target-tap projector (fc.weight) in both, and the
# DFlash2 selector-codebook slots 12/13 carry the Markov W1/W2 weights whose
# [vocab, rank] shapes are exactly what DFlash2 repurposed those slots FOR.
# 14 is reserved (DFlash2's selector hidden projection has no DSpark twin).
# 17..20 are the tensors a DSpark drafter ships and a DFlash2 drafter does not
# (its own embedding and lm_head, and the confidence head).
(KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT,
 KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM, KIND_ATTENTION_NORM, KIND_MLP_NORM,
 KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN,
 KIND_PROJECTOR, KIND_MARKOV_W1, KIND_MARKOV_W2,
 KIND_RESERVED_SELECTOR_HIDDEN_PROJ,
 KIND_FINAL_NORM, KIND_HIDDEN_NORM,
 KIND_EMBED, KIND_LM_HEAD, KIND_CONFIDENCE_PROJ_WEIGHT, KIND_CONFIDENCE_PROJ_BIAS,
) = range(21)

PER_LAYER_KINDS = (
    KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT,
    KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM, KIND_ATTENTION_NORM,
    KIND_MLP_NORM, KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN,
)

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
}

GLOBAL_NAMES = {
    KIND_PROJECTOR: "fc.weight",
    KIND_MARKOV_W1: "markov_head.markov_w1.weight",
    KIND_MARKOV_W2: "markov_head.markov_w2.weight",
    KIND_FINAL_NORM: "norm.weight",
    KIND_HIDDEN_NORM: "hidden_norm.weight",
    KIND_EMBED: "embed_tokens.weight",
    KIND_LM_HEAD: "lm_head.weight",
    KIND_CONFIDENCE_PROJ_WEIGHT: "confidence_head.proj.weight",
    KIND_CONFIDENCE_PROJ_BIAS: "confidence_head.proj.bias",
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

    def shape(self, name: str) -> list[int]:
        return list(self.meta[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.meta[name]["dtype"]


def read_geometry(checkpoint: Path) -> dict:
    """The DSpark drafter geometry, from the source's own config.json."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    t = cfg.get("transformer_layer_config")
    if not isinstance(t, dict):
        raise PackFailure(
            "config.json carries no transformer_layer_config block; this "
            "source's config schema is not the verified DSpark layout "
            f"(architectures={cfg.get('architectures')}) - record it, never guess")
    geometry = {
        "hidden": t["hidden_size"],
        "layer_count": t["num_hidden_layers"],
        "query_heads": t["num_attention_heads"],
        "kv_heads": t["num_key_value_heads"],
        "head_dim": t["head_dim"],
        "rope_dim": t["head_dim"],
        "ffn": t["intermediate_size"],
        "vocab": cfg.get("draft_vocab_size", t["vocab_size"]),
        "block_size": cfg["block_size"],
        "taps": list(cfg["aux_hidden_state_layer_ids"]),
        "markov_rank": cfg["markov_rank"],
        "mask_token_id": cfg["mask_token_id"],
        "sliding_window": t["sliding_window"],
        "confidence_with_markov": bool(cfg.get("confidence_head_with_markov")),
        "confidence": bool(cfg.get("enable_confidence_head")),
        "markov_type": cfg.get("markov_head_type"),
        "rope_theta": float(t.get("rope_parameters", {}).get("rope_theta", 10000.0)),
        "layer_types": list(t.get("layer_types", [])),
    }
    if geometry["markov_type"] != "vanilla":
        raise PackFailure(
            f"markov_head_type {geometry['markov_type']!r} is not the verified "
            "'vanilla' W1/W2 pair - no pack slot exists for it")
    if len(geometry["taps"]) != 5:
        raise PackFailure(
            f"aux taps {geometry['taps']} - the wire format carries exactly 5")
    return geometry


def kind_shape(kind: int, g: dict) -> tuple[int, int]:
    hidden, ffn = g["hidden"], g["ffn"]
    vocab, rank = g["vocab"], g["markov_rank"]
    table = {
        KIND_ATTN_QUERY: (g["query_heads"] * g["head_dim"], hidden),
        KIND_ATTN_KEY: (g["kv_heads"] * g["head_dim"], hidden),
        KIND_ATTN_VALUE: (g["kv_heads"] * g["head_dim"], hidden),
        KIND_ATTN_OUTPUT: (hidden, g["query_heads"] * g["head_dim"]),
        KIND_ATTN_QUERY_NORM: (1, g["head_dim"]),
        KIND_ATTN_KEY_NORM: (1, g["head_dim"]),
        KIND_ATTENTION_NORM: (1, hidden),
        KIND_MLP_NORM: (1, hidden),
        KIND_FFN_GATE: (ffn, hidden),
        KIND_FFN_UP: (ffn, hidden),
        KIND_FFN_DOWN: (hidden, ffn),
        KIND_PROJECTOR: (hidden, len(g["taps"]) * hidden),
        KIND_MARKOV_W1: (vocab, rank),
        KIND_MARKOV_W2: (vocab, rank),
        KIND_FINAL_NORM: (1, hidden),
        KIND_HIDDEN_NORM: (1, hidden),
        KIND_EMBED: (vocab, hidden),
        KIND_LM_HEAD: (vocab, hidden),
        # confidence head: [1, hidden(+markov)] row vector + its scalar bias
        KIND_CONFIDENCE_PROJ_WEIGHT:
            (1, hidden + (rank if g["confidence_with_markov"] else 0)),
        KIND_CONFIDENCE_PROJ_BIAS: (1, 1),
    }
    return table[kind]


def build_inventory(g: dict) -> list[tuple[int, int, str]]:
    """[(kind, layer, name), ...] in pack order. layer == 0xFFFFFFFF for globals."""
    inv: list[tuple[int, int, str]] = []
    for layer in range(g["layer_count"]):
        for kind in PER_LAYER_KINDS:
            inv.append((kind, layer, f"layers.{layer}.{_LAYER_NAMES[kind]}"))
    for kind, name in GLOBAL_NAMES.items():
        if kind == KIND_CONFIDENCE_PROJ_WEIGHT and not g["confidence"]:
            continue
        if kind == KIND_CONFIDENCE_PROJ_BIAS and not g["confidence"]:
            continue
        inv.append((kind, 0xFFFFFFFF, name))
    return inv


def pack(checkpoint: Path, output: Path, receipt: dict) -> dict:
    g = read_geometry(checkpoint)
    src = SafetensorsHeader(checkpoint / "model.safetensors")

    plans = []
    cursor = 0
    for kind, layer, name in build_inventory(g):
        if name not in src.names:
            raise PackFailure(f"missing drafter tensor: {name}")
        offset, nbytes = src.offset(name)
        rows, cols = kind_shape(kind, g)
        payload_offset = (cursor + PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)
        payload_bytes = rows * cols * BF16_BYTES
        if src.dtype(name) != "BF16":
            raise PackFailure(
                f"{name}: dtype {src.dtype(name)} - the format carries BF16 only; "
                "repackage-never-quantize forbids converting")
        # element-count equality: the wire stores rows x cols, so the source's
        # 1-D norm vectors ([64], [1]) and 2-D weights land identically
        src_numel = 1
        for dim in src.shape(name):
            src_numel *= dim
        if src_numel != rows * cols:
            raise PackFailure(
                f"shape mismatch {name}: safetensors {src.shape(name)} "
                f"({src_numel} elements) vs plan [{rows}, {cols}] (config geometry)")
        if nbytes != payload_bytes:
            raise PackFailure(f"size mismatch {name}: header {nbytes} vs plan {payload_bytes}")
        plans.append((kind, layer, name, payload_offset, payload_bytes, offset, rows, cols))
        cursor = payload_offset + payload_bytes

    payload_base = (HEADER_BYTES + len(plans) * ENTRY_BYTES +
                    PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)
    # the writer pads after EVERY payload, the last one included; the header's
    # file_bytes is the padded end so verify() can hold the actual file to it
    cursor = (cursor + PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)
    file_bytes = payload_base + cursor

    # The Q6SP v3 26I2Q core, field-for-field (first/total carry the drafter's
    # own layer span; GDN fields are zero - the drafter has none).
    core = CORE_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plans),
        g["hidden"], g["layer_count"], 0, g["layer_count"],
        1, 0,                       # attention period 1, full phase 0 (no GDN)
        0, 0, 0, 0, 0,              # GDN heads/dims/kernel = 0
        g["query_heads"], g["kv_heads"], g["head_dim"], g["rope_dim"],
        g["ffn"], g["vocab"], g["block_size"], len(g["taps"]),
        1, 0, HEADER_BYTES, file_bytes)
    ext = EXT_STRUCT.pack(
        *g["taps"],
        g["markov_rank"], g["mask_token_id"], g["sliding_window"],
        (1 if "embed_tokens.weight" in src.names else 0) |
        (2 if "lm_head.weight" in src.names else 0) |
        (4 if g["confidence"] else 0) |
        (8 if g["confidence_with_markov"] else 0) |
        (16 if any(lt == "sliding_attention" for lt in g["layer_types"]) else 0),
        int(g["rope_theta"] * 1000),
        g["hidden"] + (g["markov_rank"] if g["confidence_with_markov"] else 0)
            if g["confidence"] else 0,
        0, 0, 0, 0, 0)
    entries = b"".join(
        ENTRY_STRUCT.pack(kind, layer, WEIGHT_BF16, rows, cols, 0,
                          payload_base + payload_offset, payload_bytes, 0, 0)
        for kind, layer, name, payload_offset, payload_bytes, _, rows, cols in plans)

    receipt.update({
        "kind": "sparkpipe.k3.dspark-stagepack-receipt.v1",
        "tool": "tools/k3_dspark_stagepack.py",
        "layer_count": g["layer_count"],
        "tensor_count": len(plans),
        "bytes": file_bytes,
        "tp_degree": 1,
        "tp_rank": 0,
        "block_size": g["block_size"],
        "target_tap_layers": g["taps"],
        "markov_rank": g["markov_rank"],
        "mask_token_id": g["mask_token_id"],
        "sliding_window": g["sliding_window"],
        "confidence_head": g["confidence"],
        "confidence_with_markov": g["confidence_with_markov"],
        "rope_theta": g["rope_theta"],
        "vocab": g["vocab"],
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp",
                                     dir=output.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(core)
        temp.write(ext)
        temp.write(entries)
        temp.write(b"\0" * (payload_base - temp.tell()))
        temp.flush()
        # payloads stream from the source in FILE-OFFSET order (one
        # sequential pass - random seeks are what make warm-storage reads
        # crawl) into their planned pack slots via pwrite; the safetensors
        # file is opened ONCE
        with open(checkpoint / "model.safetensors", "rb") as src_file:
            for plan in sorted(plans, key=lambda p: p[5]):
                kind, layer, name, payload_offset, payload_bytes, src_offset, _, _ = plan
                src_file.seek(src_offset)
                remaining = payload_bytes
                chunk_off = payload_base + payload_offset
                while remaining > 0:
                    chunk = src_file.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise PackFailure(f"{name}: source ended early")
                    os.pwrite(temp.fileno(), chunk, chunk_off)
                    chunk_off += len(chunk)
                    remaining -= len(chunk)
        # extend to the header's aligned file_bytes so the tail pad exists
        # (holes between payloads are already implicit zeros)
        os.ftruncate(temp.fileno(), file_bytes)
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    return receipt


def read_pack_header(pack_path: Path) -> dict:
    with open(pack_path, "rb") as f:
        core = CORE_STRUCT.unpack(f.read(CORE_HEADER_BYTES))
        ext = EXT_STRUCT.unpack(f.read(EXT_HEADER_BYTES))
    (magic, version, header_bytes, entry_bytes, tensor_count, hidden, layers,
     first, total, period, phase, gk, gv, gkd, gvd, gck, aq, akv, ahd, arope,
     ffn, vocab, block, taps, tp_deg, tp_rank, _, file_bytes) = core
    if magic != MAGIC:
        raise PackFailure(f"magic {magic:#x} is not K3DS")
    if version != FORMAT_VERSION or header_bytes != HEADER_BYTES or entry_bytes != ENTRY_BYTES:
        raise PackFailure("header layout mismatch")
    if hidden == 0 or layers == 0 or aq == 0 or akv == 0 or ahd == 0 or vocab == 0:
        raise PackFailure("degenerate geometry")
    if block < 2 or taps != 5:
        raise PackFailure(
            f"block/tap inconsistency: block={block} tap_count={taps} "
            "(the format carries exactly 5 taps)")
    if tp_deg != 1 or tp_rank != 0:
        raise PackFailure("drafter packs are TP1 rank0 by contract")
    return {
        "tensor_count": tensor_count, "hidden": hidden, "layers": layers,
        "query_heads": aq, "kv_heads": akv, "head_dim": ahd, "rope_dim": arope,
        "ffn": ffn, "vocab": vocab, "block_size": block, "tap_count": taps,
        "taps": list(ext[:5]), "markov_rank": ext[5], "mask_token_id": ext[6],
        "sliding_window": ext[7], "flags": ext[8], "rope_theta_milli": ext[9],
        "confidence_input_dim": ext[10], "file_bytes": file_bytes,
    }


def verify(pack_path: Path) -> dict:
    header = read_pack_header(pack_path)
    actual_bytes = pack_path.stat().st_size
    if actual_bytes != header["file_bytes"]:
        raise PackFailure(
            f"file size {actual_bytes} != header file_bytes {header['file_bytes']} "
            "(truncated or extended pack)")
    with open(pack_path, "rb") as f:
        f.seek(HEADER_BYTES)
        entries = [ENTRY_STRUCT.unpack(f.read(ENTRY_BYTES))
                   for _ in range(header["tensor_count"])]
    seen_offsets = []
    for e_kind, e_layer, e_fmt, e_rows, e_cols, e_sg, p_off, p_bytes, _, _ in entries:
        if e_fmt != WEIGHT_BF16:
            raise PackFailure(f"kind {e_kind}: only BF16 (0) is defined")
        if e_rows == 0 or e_cols == 0 or p_bytes != e_rows * e_cols * BF16_BYTES:
            raise PackFailure(f"kind {e_kind}: entry shape/bytes inconsistent")
        if p_off % PAYLOAD_ALIGNMENT != 0 or p_off + p_bytes > header["file_bytes"]:
            raise PackFailure(f"kind {e_kind}: payload out of file bounds")
        seen_offsets.append(p_off)
    if len(set(seen_offsets)) != len(seen_offsets):
        raise PackFailure("overlapping payloads")
    header["entries"] = len(entries)
    return header


def verify_payload(pack_path: Path, checkpoint: Path) -> bool:
    """Byte-for-byte round-trip: every packed payload equals its safetensors tensor."""
    g = read_geometry(checkpoint)
    src = SafetensorsHeader(checkpoint / "model.safetensors")
    inv = build_inventory(g)
    with open(pack_path, "rb") as f:
        f.seek(HEADER_BYTES)
        entries = [ENTRY_STRUCT.unpack(f.read(ENTRY_BYTES)) for _ in range(len(inv))]
    ok = True
    for (kind, layer, name), entry in zip(inv, entries):
        e_kind, e_layer, e_fmt, e_rows, e_cols, e_sg, p_off, p_bytes, _, _ = entry
        rows, cols = kind_shape(kind, g)
        if e_kind != kind or e_rows != rows or e_cols != cols:
            print(f"plan mismatch {name}", file=sys.stderr)
            ok = False
            continue
        src_off, src_bytes = src.offset(name)
        with open(checkpoint / "model.safetensors", "rb") as sf, \
                open(pack_path, "rb") as pf:
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
                        help="dir holding the DSpark drafter config.json + model.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt: dict = {}
    try:
        pack(args.checkpoint, args.output, receipt)
    except PackFailure as e:
        print(f"k3_dspark_stagepack failed: {e}", file=sys.stderr)
        return 1
    header = verify(args.output)
    if not verify_payload(args.output, args.checkpoint):
        print("k3_dspark_stagepack round-trip FAILED", file=sys.stderr)
        return 1
    print(f"k3_dspark_stagepack wrote {args.output} tensors={header['tensor_count']} "
          f"file_gib={header['file_bytes'] / 2**30:.2f} block={header['block_size']} "
          f"taps={header['taps']} round_trip=ok")
    with open(str(args.output) + ".receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
