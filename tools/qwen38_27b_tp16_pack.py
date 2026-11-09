#!/usr/bin/env python3
"""Qwen 3.8 and 3.5 dense hybrid checkpoints (official FP8 or BF16) -> TP16 or TP1 named stage packs.

The geometry comes from the checkpoint's config.json, checked against the
dimensions the device and CUDA stages are built for (GDN key and value heads
of 128, convolution of 4, attention heads of 256 with 64 rotary dimensions).
A checkpoint with tied embeddings (the Qwen 3.5 small models) writes no head
entry: the stage reads the embedding rows as the head. The small checkpoints
store the gated GDN norm weight and A_log in F32; the publisher's model runs in
bfloat16 and keeps no module in FP32, so the norm weight is rounded to BF16 to
nearest even as its load does (A_log's F32 values are all BF16 values).

One pass over the checkpoint writes every rank: each tensor is read once and
its rank slices go to the rank pack writers. --tp-degree 16 writes the
sixteen Spark ranks; --tp-degree 1 writes one whole-model pack for a single
device (one Mac Studio). Output per rank:
<out>/qwen38_27b.tp<T>.rank<NN>.pack (tools/spark_named_pack.py container,
magic SPNP), plus <pack>.sha256 and the empty-range <pack>.experts that the
weightd lazy attach reads for a dense pack.

At TP16 rank r holds (block = the checkpoint's 128x128 FP8 scale block; at
TP1 the single rank holds every row and column of the same tensors):
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
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble, source_identity  # noqa: E402

TP = 16
TP_DEGREES = (1, 16)
NAME = "qwen38_27b"
BLOCK = 128
GDN_KEY_DIM = 128
GDN_VALUE_DIM = 128
CONV = 4
HEAD_DIM = 256
ROPE_DIM = 64
ROPE_THETA = 10000000
EPSILON = 1e-6
HIDDEN = LAYERS = VOCAB = FFN = GDN_KEY_HEADS = GDN_VALUE_HEADS = ATTN_HEADS = ATTN_KV_HEADS = 0
GDN_QK = GDN_V = ATTN_Q = ATTN_QG = ATTN_KV = ATTN_QKV = FULL_PERIOD = FULL_PHASE = 0
TIED = False
PREFIX = "model.language_model."
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class PackFailure(RuntimeError):
    pass


def configure(config):
    global HIDDEN, LAYERS, VOCAB, FFN, GDN_KEY_HEADS, GDN_VALUE_HEADS, ATTN_HEADS, ATTN_KV_HEADS, GDN_QK, GDN_V, ATTN_Q, ATTN_QG, ATTN_KV
    global ATTN_QKV, FULL_PERIOD, FULL_PHASE, TIED
    text = config.get("text_config", config)
    rope = text.get("rope_parameters") or {}
    fixed = {"linear_key_head_dim": GDN_KEY_DIM, "linear_value_head_dim": GDN_VALUE_DIM, "linear_conv_kernel_dim": CONV, "head_dim": HEAD_DIM,
             "rms_norm_eps": EPSILON}
    for key, value in fixed.items():
        if text.get(key) != value:
            raise PackFailure(f"checkpoint {key} is {text.get(key)}; the stages are built for {value}")
    if round(text["head_dim"] * rope.get("partial_rotary_factor", 0)) != ROPE_DIM or rope.get("rope_theta") != ROPE_THETA:
        raise PackFailure(f"checkpoint rope {rope} is not {ROPE_DIM} rotary dimensions at theta {ROPE_THETA}")
    HIDDEN, LAYERS, VOCAB, FFN = text["hidden_size"], text["num_hidden_layers"], text["vocab_size"], text["intermediate_size"]
    GDN_KEY_HEADS, GDN_VALUE_HEADS = text["linear_num_key_heads"], text["linear_num_value_heads"]
    ATTN_HEADS, ATTN_KV_HEADS = text["num_attention_heads"], text["num_key_value_heads"]
    GDN_QK, GDN_V = GDN_KEY_HEADS * GDN_KEY_DIM, GDN_VALUE_HEADS * GDN_VALUE_DIM
    ATTN_Q = ATTN_HEADS * HEAD_DIM
    ATTN_QG, ATTN_KV = 2 * ATTN_Q, ATTN_KV_HEADS * HEAD_DIM
    ATTN_QKV = ATTN_QG + 2 * ATTN_KV
    FULL_PERIOD = text["full_attention_interval"]
    FULL_PHASE = FULL_PERIOD - 1
    kinds = text.get("layer_types") or []
    expected = ["full_attention" if layer % FULL_PERIOD == FULL_PHASE else "linear_attention" for layer in range(LAYERS)]
    if kinds != expected:
        raise PackFailure("checkpoint layer_types are not linear attention with full attention every full_attention_interval layers")
    TIED = bool(config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))
    if GDN_VALUE_HEADS % GDN_KEY_HEADS or VOCAB % TP or GDN_KEY_HEADS % TP or GDN_VALUE_HEADS % TP or ATTN_QKV % (TP * BLOCK) or ATTN_Q % (TP * BLOCK):
        raise PackFailure(f"the geometry does not split into {TP} ranks")


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
    def __init__(self, out_dir, resume, identity):
        self.out_dir = Path(out_dir)
        self.writers = [PayloadWriter(self.payload(rank), identity, resume=resume and self.payload(rank).exists())
                        for rank in range(TP)]

    def payload(self, rank):
        return self.out_dir / f"{NAME}.tp{TP}.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"{NAME}.tp{TP}.rank{rank:02d}.pack"

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


def bf16_rounded(checkpoint, name, shape):
    dtype, array = checkpoint.array(name)
    if tuple(array.shape) != tuple(shape):
        raise PackFailure(f"{name}: shape {tuple(array.shape)} != {tuple(shape)}")
    if dtype == "BF16":
        return array
    if dtype != "F32":
        raise PackFailure(f"{name}: expected BF16 or F32, found {dtype}")
    bits = np.asarray(array, dtype=np.float32).view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
    return np.where((bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000), (bits >> 16).astype(np.uint16) | np.uint16(0x40), rounded)


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
    head = None if TIED else bf16(checkpoint, "lm_head.weight", (VOCAB, HIDDEN))
    if TIED and TP != 1:
        raise PackFailure("a tied head is read from the embedding rows of a whole-model pack only")
    final = folded_norm(checkpoint, PREFIX + "norm.weight", HIDDEN)
    for rank in range(TP):
        first, count = window(VOCAB, rank)
        packs.add(rank, "embed", embed[first:first + count], "bf16")
        packs.add(rank, "head_norm", final, "f32")
        if head is not None:
            packs.add(rank, "head", head[first:first + count], "bf16")


def pack_gdn(checkpoint, packs, layer, codec):
    p = f"{PREFIX}layers.{layer}.linear_attn."
    qkv, qkv_scale = linear_pair(checkpoint, p + "in_proj_qkv", codec)
    z, z_scale = linear_pair(checkpoint, p + "in_proj_z", codec)
    out, out_scale = linear_pair(checkpoint, p + "out_proj", codec)
    beta = bf16(checkpoint, p + "in_proj_b.weight", (GDN_VALUE_HEADS, HIDDEN))
    decay = bf16(checkpoint, p + "in_proj_a.weight", (GDN_VALUE_HEADS, HIDDEN))
    conv = bf16(checkpoint, p + "conv1d.weight", (2 * GDN_QK + GDN_V, 1, CONV)).reshape(2 * GDN_QK + GDN_V, CONV)
    a_log = f32_vector(checkpoint, p + "A_log", (GDN_VALUE_HEADS,))
    dt_bias = f32_vector(checkpoint, p + "dt_bias", (GDN_VALUE_HEADS,))
    norm = bf16_rounded(checkpoint, p + "norm.weight", (GDN_VALUE_DIM,))
    if qkv.shape != (2 * GDN_QK + GDN_V, HIDDEN) or z.shape != (GDN_V, HIDDEN) or out.shape != (HIDDEN, GDN_V):
        raise PackFailure(f"layer {layer}: GDN projection shapes {qkv.shape} {z.shape} {out.shape}")
    heads = GDN_VALUE_HEADS // TP
    key_heads = GDN_KEY_HEADS // TP
    for rank in range(TP):
        q = (rank * key_heads * GDN_KEY_DIM, key_heads * GDN_KEY_DIM)
        k = (GDN_QK + rank * key_heads * GDN_KEY_DIM, key_heads * GDN_KEY_DIM)
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
        config = {"model": NAME, "tp_degree": TP, "tp_rank": rank, "hidden": HIDDEN, "layers": LAYERS,
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
    global TP, NAME
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--codec", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--tp-degree", type=int, choices=TP_DEGREES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", default=NAME, help="the pack's model name and file prefix, such as qwen38_27b or qwen35_4b")
    args = parser.parse_args()
    TP, NAME = args.tp_degree, args.name
    configure(json.loads((args.checkpoint / "config.json").read_text()))
    revision = json.loads((args.checkpoint / "DOWNLOAD-RECEIPT.json").read_text()).get("revision", "unknown")
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    packs = RankPacks(args.output_dir, args.resume, source_identity(args.checkpoint, "qwen38_27b_tp16_pack", files=(args.contract,)))
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
