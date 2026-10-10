#!/usr/bin/env python3
"""Qwen3.8 Max SPQ8 v2 TP16 rank pack -> SPNP named stage pack, on the node.

Reads one rank's qwenmax.nvfp4.tp16.rankNN.sp (built from
qwen3.8-max-nvfp4-radixark-bf16-spine; slices checked against the checkpoint
for ranks 0, 1, 5 and 9) and writes qwen38_max.tp16.rankNN.pack beside the
tools/spark_named_pack.py container, plus <pack>.sha256 and the empty-range
<pack>.experts of a dense pack.

Rank r holds:
  embed / head        vocab rows [r*15520, +15520), bf16
  norms               (1 + w) f32 for input, post-attention, final, q and k;
                      the gated GDN norm raw bf16
  GDN                 gdn_qkvz [q key head r | k key head r | v and z value
                      heads 8r..8r+7] = [2304, 8192]; gdn_ba = b and a rows
                      8r..8r+7 [16, 8192]; gdn_conv q|k|v channels [1280, 4];
                      A_log, dt_bias [8] f32; gdn_out [8192, 1024]
  attention           attn_qkv [q and gate of heads 4r..4r+3 (2048) | k rows
                      64r.. (64) | v rows 64r.. (64)] = [2176, 8192], layout
                      "rank_heads": an all-gather in rank order gives every
                      head's query and gate and the whole k and v; attn_out
                      [8192, 1024] (columns of heads 4r..4r+3)
  MoE                 moe_router [512, 8192] bf16; moe_w1 / moe_w3 [32*2048,
                      8192] and moe_w2 [32*8192, 2048] nvfp4 for experts
                      [32r, +32), each with "<name>.scale" = f32
                      weight_scale_2 per expert then the UE4M3 per-16 planes
                      (the order LmWeightCodecScaleTensor<NVFP4> reads);
                      moe_shared_gate_up [256, 8192], moe_shared_down
                      [8192, 128], moe_shared_gate [1, 8192] bf16
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble

SPQ8_MAGIC = 0x50533851
TP = 16
HIDDEN = 8192
LAYERS = 92
VOCAB = 248320
VOCAB_ROWS = VOCAB // TP
FULL_PERIOD = 4
FULL_PHASE = 3
EXPERTS = 512
LOCAL_EXPERTS = EXPERTS // TP
TOP_K = 10
MOE_INTER = 2048
SHARED_ROWS = 2048 // TP
GDN_KEY = 128
GDN_QK = 16 * GDN_KEY
GDN_HEADS = 128 // TP
GDN_V = GDN_HEADS * 128
HEADS = 64
HEAD_DIM = 256
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2

(EMBEDDING, FINAL_NORM, LM_HEAD, ATTENTION_NORM, MLP_NORM, MOE_GATE, MOE_W1, MOE_W3, MOE_DOWN, SHARED_GATE, SHARED_UP,
 SHARED_DOWN, SHARED_GATE_WEIGHT, GDN_QKV, GDN_Z, GDN_BETA, GDN_DECAY, GDN_OUT, GDN_CONV, GDN_A_LOG, GDN_DT_BIAS,
 GDN_NORM, ATTN_Q, ATTN_K, ATTN_V, ATTN_OUT, ATTN_Q_NORM, ATTN_K_NORM) = range(28)
FORMAT_BF16, FORMAT_F32, FORMAT_NVFP4 = 0, 1, 8


class ConvertFailure(RuntimeError):
    pass


class SourcePack:
    def __init__(self, path):
        self.handle = open(path, "rb")
        head = self.handle.read(128)
        fields = struct.unpack_from("<26I", head, 0)
        if fields[0] != SPQ8_MAGIC or fields[1] != 2 or fields[2] != 128 or fields[3] != 56:
            raise ConvertFailure(f"{path} is not an SPQ8 v2 pack")
        self.tp_degree, self.tp_rank, directory, self.file_bytes = struct.unpack_from("<IIQQ", head, 104)
        if self.tp_degree != TP or fields[5] != HIDDEN or fields[6] != LAYERS or fields[20] != EXPERTS or fields[23] != VOCAB:
            raise ConvertFailure(f"{path}: geometry {fields[5:24]} tp {self.tp_degree} is not Qwen3.8 Max TP16")
        self.handle.seek(directory)
        raw = self.handle.read(fields[4] * 56)
        self.entries = {}
        for index in range(fields[4]):
            kind, layer, fmt, rows, cols, group, po, pb, so, sb = struct.unpack_from("<6I4Q", raw, index * 56)
            self.entries[(kind, layer)] = (fmt, rows, cols, po, pb, so, sb)

    def entry(self, kind, layer, fmt, rows, cols):
        found = self.entries.get((kind, layer))
        if found is None or found[0] != fmt or found[1] != rows or found[2] != cols:
            raise ConvertFailure(f"kind {kind} layer {layer}: {found} is not format {fmt} [{rows}, {cols}]")
        return found

    def read(self, offset, count):
        self.handle.seek(offset)
        data = self.handle.read(count)
        if len(data) != count:
            raise ConvertFailure(f"short read at {offset}")
        return data

    def bf16(self, kind, layer, rows, cols, first=0, count=None):
        count = rows if count is None else count
        _, _, _, po, _, _, _ = self.entry(kind, layer, FORMAT_BF16, rows, cols)
        return np.frombuffer(self.read(po + first * cols * 2, count * cols * 2), dtype=np.uint16).reshape(count, cols)

    def f32(self, kind, layer, cols):
        _, _, _, po, _, _, _ = self.entry(kind, layer, FORMAT_F32, 1, cols)
        return np.frombuffer(self.read(po, cols * 4), dtype=np.float32)


def folded(values):
    return ((values.astype(np.uint32) << 16).view(np.float32) + np.float32(1.0)).astype(np.float32)


def experts(source, writer, name, kind, layer, rows, cols):
    fmt, _, _, po, pb, so, sb = source.entry(kind, layer, FORMAT_NVFP4, LOCAL_EXPERTS * rows, cols)
    plane = rows * (cols // 16)
    if pb != LOCAL_EXPERTS * rows * cols // 2 or sb != LOCAL_EXPERTS * (plane + 8):
        raise ConvertFailure(f"{name}: payload {pb} / scale {sb} bytes do not match {LOCAL_EXPERTS} experts of [{rows}, {cols}]")
    writer.add(name, source.read(po, pb), "nvfp4_e2m1", (LOCAL_EXPERTS * rows, cols))
    segments = source.read(so, sb)
    globals_ = np.empty(LOCAL_EXPERTS, dtype=np.float32)
    planes = bytearray()
    for expert in range(LOCAL_EXPERTS):
        base = expert * (plane + 8)
        planes += segments[base:base + plane]
        globals_[expert] = struct.unpack_from("<f", segments, base + plane + 4)[0]
    if not np.all(np.isfinite(globals_)) or np.any(globals_ <= 0):
        raise ConvertFailure(f"{name}: weight_scale_2 {globals_} is not positive and finite")
    writer.add(name + ".scale", globals_.tobytes() + bytes(planes), "ue4m3_f32_global", (LOCAL_EXPERTS, rows, cols // 16),
               {"global_scales": LOCAL_EXPERTS})


def convert(source, writer, rank):
    first = rank * VOCAB_ROWS
    writer.add("embed", source.bf16(EMBEDDING, 0xffffffff, VOCAB, HIDDEN, first, VOCAB_ROWS).tobytes(), "bf16", (VOCAB_ROWS, HIDDEN))
    writer.add("head_norm", folded(source.bf16(FINAL_NORM, 0xffffffff, 1, HIDDEN)[0]).tobytes(), "f32", (HIDDEN,))
    writer.add("head", source.bf16(LM_HEAD, 0xffffffff, VOCAB, HIDDEN, first, VOCAB_ROWS).tobytes(), "bf16", (VOCAB_ROWS, HIDDEN))
    heads = slice(rank * GDN_HEADS, (rank + 1) * GDN_HEADS)
    for layer in range(LAYERS):
        name = f"layers.{layer}."
        writer.add(name + "input_norm", folded(source.bf16(ATTENTION_NORM, layer, 1, HIDDEN)[0]).tobytes(), "f32", (HIDDEN,))
        writer.add(name + "post_norm", folded(source.bf16(MLP_NORM, layer, 1, HIDDEN)[0]).tobytes(), "f32", (HIDDEN,))
        if layer % FULL_PERIOD == FULL_PHASE:
            qkv = np.concatenate((source.bf16(ATTN_Q, layer, 4 * 2 * HEAD_DIM, HIDDEN), source.bf16(ATTN_K, layer, 64, HIDDEN),
                                  source.bf16(ATTN_V, layer, 64, HIDDEN)))
            writer.add(name + "attn_qkv", qkv.tobytes(), "bf16", qkv.shape)
            writer.add(name + "attn_q_norm", folded(source.bf16(ATTN_Q_NORM, layer, 1, HEAD_DIM)[0]).tobytes(), "f32", (HEAD_DIM,))
            writer.add(name + "attn_k_norm", folded(source.bf16(ATTN_K_NORM, layer, 1, HEAD_DIM)[0]).tobytes(), "f32", (HEAD_DIM,))
            writer.add(name + "attn_out", source.bf16(ATTN_OUT, layer, HIDDEN, 4 * HEAD_DIM).tobytes(), "bf16", (HIDDEN, 4 * HEAD_DIM))
        else:
            qkvz = np.concatenate((source.bf16(GDN_QKV, layer, 2 * GDN_KEY + GDN_V, HIDDEN), source.bf16(GDN_Z, layer, GDN_V, HIDDEN)))
            writer.add(name + "gdn_qkvz", qkvz.tobytes(), "bf16", qkvz.shape)
            ba = np.concatenate((source.bf16(GDN_BETA, layer, 128, HIDDEN)[heads], source.bf16(GDN_DECAY, layer, 128, HIDDEN)[heads]))
            writer.add(name + "gdn_ba", ba.tobytes(), "bf16", ba.shape)
            conv = source.bf16(GDN_CONV, layer, 2 * GDN_QK + 128 * 128, 4)
            conv = np.concatenate((conv[rank * GDN_KEY:(rank + 1) * GDN_KEY], conv[GDN_QK + rank * GDN_KEY:GDN_QK + (rank + 1) * GDN_KEY],
                                   conv[2 * GDN_QK + rank * GDN_V:2 * GDN_QK + (rank + 1) * GDN_V]))
            writer.add(name + "gdn_conv", conv.tobytes(), "bf16", conv.shape)
            writer.add(name + "gdn_a_log", np.ascontiguousarray(source.f32(GDN_A_LOG, layer, 128)[heads]).tobytes(), "f32", (GDN_HEADS,))
            writer.add(name + "gdn_dt_bias", np.ascontiguousarray(source.f32(GDN_DT_BIAS, layer, 128)[heads]).tobytes(), "f32", (GDN_HEADS,))
            writer.add(name + "gdn_norm", source.bf16(GDN_NORM, layer, 1, 128).tobytes(), "bf16", (128,))
            writer.add(name + "gdn_out", source.bf16(GDN_OUT, layer, HIDDEN, GDN_V).tobytes(), "bf16", (HIDDEN, GDN_V))
        writer.add(name + "moe_router", source.bf16(MOE_GATE, layer, EXPERTS, HIDDEN).tobytes(), "bf16", (EXPERTS, HIDDEN))
        experts(source, writer, name + "moe_w1", MOE_W1, layer, MOE_INTER, HIDDEN)
        experts(source, writer, name + "moe_w3", MOE_W3, layer, MOE_INTER, HIDDEN)
        experts(source, writer, name + "moe_w2", MOE_DOWN, layer, HIDDEN, MOE_INTER)
        gate_up = np.concatenate((source.bf16(SHARED_GATE, layer, SHARED_ROWS, HIDDEN), source.bf16(SHARED_UP, layer, SHARED_ROWS, HIDDEN)))
        writer.add(name + "moe_shared_gate_up", gate_up.tobytes(), "bf16", gate_up.shape)
        writer.add(name + "moe_shared_down", source.bf16(SHARED_DOWN, layer, HIDDEN, SHARED_ROWS).tobytes(), "bf16", (HIDDEN, SHARED_ROWS))
        writer.add(name + "moe_shared_gate", source.bf16(SHARED_GATE_WEIGHT, layer, 1, HIDDEN).tobytes(), "bf16", (1, HIDDEN))
        print(f"layer {layer} done offset {writer.offset}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = SourcePack(args.source)
    rank = source.tp_rank
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = args.out_dir / f"qwen38_max.tp16.rank{rank:02d}.pack"
    payload = Path(str(final) + ".payload")
    writer = PayloadWriter(payload, resume=args.resume and payload.exists())
    convert(source, writer, rank)
    writer.close()
    config = {"model": "qwen38_max", "tp_degree": TP, "tp_rank": rank, "hidden": HIDDEN, "layers": LAYERS, "vocab": VOCAB,
              "vocab_first": rank * VOCAB_ROWS, "vocab_rows": VOCAB_ROWS, "full_period": FULL_PERIOD, "full_phase": FULL_PHASE,
              "experts": EXPERTS, "local_experts": LOCAL_EXPERTS, "expert_first": rank * LOCAL_EXPERTS, "top_k": TOP_K,
              "moe_inter": MOE_INTER, "shared_rows": SHARED_ROWS, "gdn_heads": GDN_HEADS, "attn_heads_local": HEADS // TP}
    fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN, "linear_codec": "bf16",
           "expert_codec": "nvfp4_e2m1", "attn_qkv_layout": "rank_heads", "source": str(args.source.name)}
    assemble(final, payload, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, config, writer.manifest)
    digest = hashlib.sha256()
    with final.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
    Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
    print(f"rank {rank:02d} {final.stat().st_size} {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
