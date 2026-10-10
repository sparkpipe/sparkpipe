#!/usr/bin/env python3
"""Qwen3.8-27B official FP8 or BF16 checkpoint -> sixteen TP16 named stage packs.

One pass over the checkpoint writes every rank: each tensor is read once and
its sixteen rank slices go to the sixteen pack writers. Output per rank:
<out>/qwen38_27b.tp16.rank<NN>.pack (tools/spark_named_pack.py container,
magic SPNP), plus <pack>.sha256 and the empty-range <pack>.experts that the
weightd lazy attach reads for a dense pack.

Rank r holds (block = the checkpoint's 128x128 FP8 scale block):
  embed / head        vocab rows [r*15520, +15520), bf16
  norms               (1 + w) as f32 for input, post-attention, final, q and
                      k norms; the gated GDN norm weight raw bf16
  GDN                 head-parallel: q and k key head r (128 rows each), v
                      and z value heads 3r..3r+2 (384 rows each) fused as
                      gdn_qkvz [1024, 5120] fp8; b and a rows 3r..3r+2 as
                      gdn_ba [6, 5120] bf16; conv channels for q|k|v [640, 4];
                      A_log, dt_bias [3] f32; out_proj columns [5120, 384]
  attention           q_proj|k_proj|v_proj fused as 14336 rows, rank rows
                      [r*896, +896) fp8 (7 whole blocks); o_proj columns
                      [r*384, +384) [5120, 384]
  FFN                 136 blocks split 9 (ranks 0-7) / 8 (ranks 8-15):
                      gate|up rows fused [2*rows, 5120], down columns
                      [5120, rows]
With --codec fp8 (the official FP8 checkpoint) every projection is fp8 and
has a sibling "<name>.scale" f32 [rows, k blocks]: the checkpoint's 128x128
block scale repeated over its 128 rows, the per-row 128-K grid the FP8
kernels read (LmWeightCodecScaleTensor). With --codec bf16 (the official
BF16 checkpoint, the quality baseline) the same projections are bf16 with no
scale; every slice is the same, so the two packs differ only in codec.
The MTP decoder and the vision tower are not packed.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble  # noqa: E402

TP = 16
HIDDEN = 5120
LAYERS = 64
VOCAB = 248320
FFN = 17408
BLOCK = 128
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_KEY_DIM = 128
GDN_VALUE_DIM = 128
GDN_QK = GDN_KEY_HEADS * GDN_KEY_DIM
GDN_V = GDN_VALUE_HEADS * GDN_VALUE_DIM
ATTN_HEADS = 24
ATTN_KV_HEADS = 4
HEAD_DIM = 256
ATTN_Q = ATTN_HEADS * HEAD_DIM
ATTN_QG = 2 * ATTN_Q
ATTN_KV = ATTN_KV_HEADS * HEAD_DIM
ATTN_QKV = ATTN_QG + 2 * ATTN_KV
FULL_PERIOD = 4
FULL_PHASE = 3
PREFIX = "model.language_model."
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.headers = {}

    def _header(self, shard):
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        return self.headers[shard]

    def array(self, name):
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"checkpoint has no tensor {name}")
        base, header = self._header(shard)
        meta = header[name]
        dtype = {"BF16": np.uint16, "F8_E4M3": np.uint8, "F32": np.float32}[meta["dtype"]]
        begin, end = meta["data_offsets"]
        count = (end - begin) // np.dtype(dtype).itemsize
        flat = np.memmap(self.directory / shard, dtype=dtype, mode="r", offset=base + begin, shape=(count,))
        return meta["dtype"], flat.reshape(meta["shape"])


def bf16_to_f32(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def window(total, rank):
    count = total // TP
    return rank * count, count


def ffn_window(rank):
    blocks = FFN // BLOCK
    base, extra = divmod(blocks, TP)
    first = rank * base + min(rank, extra)
    return first * BLOCK, (base + (1 if rank < extra else 0)) * BLOCK


class RankPacks:
    def __init__(self, out_dir, resume):
        self.out_dir = Path(out_dir)
        self.writers = [PayloadWriter(self.payload(rank), resume=resume and self.payload(rank).exists())
                        for rank in range(TP)]

    def payload(self, rank):
        return self.out_dir / f"qwen38_27b.tp16.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"qwen38_27b.tp16.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind):
        array = np.ascontiguousarray(array)
        self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def linear(self, rank, name, payload, scale):
        if scale is None:
            self.add(rank, name, payload, "bf16")
        else:
            self.fp8(rank, name, payload, scale)

    def fp8(self, rank, name, payload, scale):
        if payload.shape[0] != scale.shape[0] * BLOCK or payload.shape[1] != scale.shape[1] * BLOCK:
            raise PackFailure(f"{name}: {payload.shape} is not the block grid of scale {scale.shape}")
        self.add(rank, name, payload, "fp8_e4m3")
        self.add(rank, name + ".scale", np.repeat(bf16_to_f32(np.asarray(scale)), BLOCK, axis=0), "f32")

    def manifest(self, rank):
        return self.writers[rank].manifest


def fp8_pair(checkpoint, name):
    dtype, payload = checkpoint.array(name + ".weight")
    scale_dtype, scale = checkpoint.array(name + ".weight_scale_inv")
    if dtype != "F8_E4M3" or scale_dtype != "BF16":
        raise PackFailure(f"{name}: expected F8_E4M3 weight with BF16 scale_inv, found {dtype}/{scale_dtype}")
    rows, cols = payload.shape
    if scale.shape != ((rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK):
        raise PackFailure(f"{name}: scale shape {scale.shape} is not the 128x128 block grid of {payload.shape}")
    return payload, scale


def linear_pair(checkpoint, name, codec):
    if codec == "fp8":
        return fp8_pair(checkpoint, name)
    return bf16(checkpoint, name + ".weight"), None


def concat(parts):
    if any(part is None for part in parts):
        if not all(part is None for part in parts):
            raise PackFailure("a fused projection mixes scaled and unscaled parts")
        return None
    return np.concatenate(parts)


def bf16(checkpoint, name, shape=None):
    dtype, array = checkpoint.array(name)
    if dtype != "BF16":
        raise PackFailure(f"{name}: expected BF16, found {dtype}")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise PackFailure(f"{name}: shape {tuple(array.shape)} != {tuple(shape)}")
    return array


def f32_vector(checkpoint, name, shape):
    dtype, array = checkpoint.array(name)
    if tuple(array.shape) != tuple(shape):
        raise PackFailure(f"{name}: shape {tuple(array.shape)} != {tuple(shape)}")
    if dtype == "BF16":
        return bf16_to_f32(np.asarray(array))
    if dtype == "F32":
        return np.asarray(array, dtype=np.float32)
    raise PackFailure(f"{name}: expected BF16 or F32, found {dtype}")


def folded_norm(checkpoint, name, width):
    return f32_vector(checkpoint, name, (width,)) + np.float32(1.0)


def rows_blocks(payload, scale, segments):
    parts = [payload[first:first + count] for first, count in segments]
    for first, count in segments:
        if first % BLOCK or count % BLOCK:
            raise PackFailure(f"row segment {first}+{count} is not block aligned")
    if scale is None:
        return np.concatenate(parts), None
    return np.concatenate(parts), np.concatenate([scale[first // BLOCK:(first + count) // BLOCK] for first, count in segments])


def cols_blocks(payload, scale, first, count):
    if first % BLOCK or count % BLOCK:
        raise PackFailure(f"column window {first}+{count} is not block aligned")
    return payload[:, first:first + count], None if scale is None else scale[:, first // BLOCK:(first + count) // BLOCK]


def pack_globals(checkpoint, packs):
    embed = bf16(checkpoint, PREFIX + "embed_tokens.weight", (VOCAB, HIDDEN))
    head = bf16(checkpoint, "lm_head.weight", (VOCAB, HIDDEN))
    final = folded_norm(checkpoint, PREFIX + "norm.weight", HIDDEN)
    for rank in range(TP):
        first, count = window(VOCAB, rank)
        packs.add(rank, "embed", embed[first:first + count], "bf16")
        packs.add(rank, "head_norm", final, "f32")
        packs.add(rank, "head", head[first:first + count], "bf16")


def pack_gdn(checkpoint, packs, layer, codec):
    p = f"{PREFIX}layers.{layer}.linear_attn."
    qkv, qkv_scale = linear_pair(checkpoint, p + "in_proj_qkv", codec)
    z, z_scale = linear_pair(checkpoint, p + "in_proj_z", codec)
    out, out_scale = linear_pair(checkpoint, p + "out_proj", codec)
    beta = bf16(checkpoint, p + "in_proj_b.weight", (GDN_VALUE_HEADS, HIDDEN))
    decay = bf16(checkpoint, p + "in_proj_a.weight", (GDN_VALUE_HEADS, HIDDEN))
    conv = bf16(checkpoint, p + "conv1d.weight", (2 * GDN_QK + GDN_V, 1, 4)).reshape(2 * GDN_QK + GDN_V, 4)
    a_log = f32_vector(checkpoint, p + "A_log", (GDN_VALUE_HEADS,))
    dt_bias = f32_vector(checkpoint, p + "dt_bias", (GDN_VALUE_HEADS,))
    norm = bf16(checkpoint, p + "norm.weight", (GDN_VALUE_DIM,))
    if qkv.shape != (2 * GDN_QK + GDN_V, HIDDEN) or z.shape != (GDN_V, HIDDEN) or out.shape != (HIDDEN, GDN_V):
        raise PackFailure(f"layer {layer}: GDN projection shapes {qkv.shape} {z.shape} {out.shape}")
    heads = GDN_VALUE_HEADS // TP
    for rank in range(TP):
        q = (rank * GDN_KEY_DIM, GDN_KEY_DIM)
        k = (GDN_QK + rank * GDN_KEY_DIM, GDN_KEY_DIM)
        v = (2 * GDN_QK + rank * heads * GDN_VALUE_DIM, heads * GDN_VALUE_DIM)
        payload, scale = rows_blocks(qkv, qkv_scale, (q, k, v))
        z_payload, z_scale_rank = rows_blocks(z, z_scale, ((rank * heads * GDN_VALUE_DIM, heads * GDN_VALUE_DIM),))
        name = f"layers.{layer}."
        packs.linear(rank, name + "gdn_qkvz", np.concatenate((payload, z_payload)), concat((scale, z_scale_rank)))
        packs.add(rank, name + "gdn_ba", np.concatenate((beta[rank * heads:(rank + 1) * heads],
                                                         decay[rank * heads:(rank + 1) * heads])), "bf16")
        packs.add(rank, name + "gdn_conv", np.concatenate([conv[first:first + count] for first, count in (q, k, v)]), "bf16")
        packs.add(rank, name + "gdn_a_log", a_log[rank * heads:(rank + 1) * heads], "f32")
        packs.add(rank, name + "gdn_dt_bias", dt_bias[rank * heads:(rank + 1) * heads], "f32")
        packs.add(rank, name + "gdn_norm", norm, "bf16")
        out_payload, out_scale_rank = cols_blocks(out, out_scale, rank * heads * GDN_VALUE_DIM, heads * GDN_VALUE_DIM)
        packs.linear(rank, name + "gdn_out", out_payload, out_scale_rank)


def pack_attention(checkpoint, packs, layer, codec):
    p = f"{PREFIX}layers.{layer}.self_attn."
    q, q_scale = linear_pair(checkpoint, p + "q_proj", codec)
    k, k_scale = linear_pair(checkpoint, p + "k_proj", codec)
    v, v_scale = linear_pair(checkpoint, p + "v_proj", codec)
    o, o_scale = linear_pair(checkpoint, p + "o_proj", codec)
    if q.shape != (ATTN_QG, HIDDEN) or k.shape != (ATTN_KV, HIDDEN) or v.shape != (ATTN_KV, HIDDEN) or o.shape != (HIDDEN, ATTN_Q):
        raise PackFailure(f"layer {layer}: attention shapes {q.shape} {k.shape} {v.shape} {o.shape}")
    fused = np.concatenate((q, k, v))
    fused_scale = concat((q_scale, k_scale, v_scale))
    q_norm = folded_norm(checkpoint, p + "q_norm.weight", HEAD_DIM)
    k_norm = folded_norm(checkpoint, p + "k_norm.weight", HEAD_DIM)
    for rank in range(TP):
        name = f"layers.{layer}."
        first, count = window(ATTN_QKV, rank)
        payload, scale = rows_blocks(fused, fused_scale, ((first, count),))
        packs.linear(rank, name + "attn_qkv", payload, scale)
        packs.add(rank, name + "attn_q_norm", q_norm, "f32")
        packs.add(rank, name + "attn_k_norm", k_norm, "f32")
        first, count = window(ATTN_Q, rank)
        out_payload, out_scale = cols_blocks(o, o_scale, first, count)
        packs.linear(rank, name + "attn_out", out_payload, out_scale)


def pack_ffn(checkpoint, packs, layer, codec):
    p = f"{PREFIX}layers.{layer}."
    gate, gate_scale = linear_pair(checkpoint, p + "mlp.gate_proj", codec)
    up, up_scale = linear_pair(checkpoint, p + "mlp.up_proj", codec)
    down, down_scale = linear_pair(checkpoint, p + "mlp.down_proj", codec)
    input_norm = folded_norm(checkpoint, p + "input_layernorm.weight", HIDDEN)
    post_norm = folded_norm(checkpoint, p + "post_attention_layernorm.weight", HIDDEN)
    for rank in range(TP):
        name = f"layers.{layer}."
        first, count = ffn_window(rank)
        g, gs = rows_blocks(gate, gate_scale, ((first, count),))
        u, us = rows_blocks(up, up_scale, ((first, count),))
        packs.linear(rank, name + "ffn_gate_up", np.concatenate((g, u)), concat((gs, us)))
        d, ds = cols_blocks(down, down_scale, first, count)
        packs.linear(rank, name + "ffn_down", d, ds)
        packs.add(rank, name + "input_norm", input_norm, "f32")
        packs.add(rank, name + "post_norm", post_norm, "f32")


def finish(packs, revision, contract_sha256, codec):
    for rank in range(TP):
        packs.writers[rank].close()
        first, count = ffn_window(rank)
        vocab_first, vocab_count = window(VOCAB, rank)
        config = {"model": "qwen38_27b", "tp_degree": TP, "tp_rank": rank, "hidden": HIDDEN, "layers": LAYERS,
                  "vocab": VOCAB, "vocab_first": vocab_first, "vocab_rows": vocab_count,
                  "ffn_first": first, "ffn_rows": count, "full_period": FULL_PERIOD, "full_phase": FULL_PHASE}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": {"fp8": "fp8_e4m3", "bf16": "bf16"}[codec], "fp8_scale_block": BLOCK,
               "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, config,
                 packs.manifest(rank))
        digest = hashlib.sha256()
        with final.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
        Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
        print(f"rank {rank:02d} {final.stat().st_size} {digest.hexdigest()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--codec", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    text = config["text_config"]
    if (text["hidden_size"], text["num_hidden_layers"], text["vocab_size"], text["intermediate_size"],
            text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"],
            text["linear_num_key_heads"], text["linear_num_value_heads"]) != (
            HIDDEN, LAYERS, VOCAB, FFN, ATTN_HEADS, ATTN_KV_HEADS, HEAD_DIM, GDN_KEY_HEADS, GDN_VALUE_HEADS):
        raise PackFailure("checkpoint geometry differs from the packer's")
    revision = json.loads((args.checkpoint / "DOWNLOAD-RECEIPT.json").read_text()).get("revision", "unknown")
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    packs = RankPacks(args.output_dir, args.resume)
    pack_globals(checkpoint, packs)
    for layer in range(LAYERS):
        if layer % FULL_PERIOD == FULL_PHASE:
            pack_attention(checkpoint, packs, layer, args.codec)
        else:
            pack_gdn(checkpoint, packs, layer, args.codec)
        pack_ffn(checkpoint, packs, layer, args.codec)
        print(f"layer {layer} packed", flush=True)
    finish(packs, revision, contract_sha256, args.codec)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen38_27b_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
