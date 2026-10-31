#!/usr/bin/env python3
"""GLM 5.3 Flash FP8 checkpoint -> TP16 (or --tp-degree N) named stage packs for the device stage.

Output per rank: <out>/glm53flash.tp<N>.rank<NN>.pack (tools/spark_named_pack.py
container) plus <pack>.sha256 and the empty-range <pack>.experts. The MTP layer
and the sparse-attention indexer are not packed: the device stage refuses rows
that would see more keys than the indexer selects, where attention is dense.

The checkpoint is read through tools/glm5_next_resident_stagepack.py's
SourceReader, so every BF16 projection is the CUDA pack's bf16(e4m3 x block
scale) and the routed experts keep their E4M3 codes with the FP32 block scale
expanded per row (one scale per row and 128 inputs).

Rank r holds, bf16 unless noted (H hidden, R = 1/TP of a sharded dimension):
  embed, head          vocab rows [r*V/TP, +V/TP); final_norm [H]
  every layer          attn_norm, post_norm [H]; attn_hc_fn / ffn_hc_fn f32
                       [24, 4H], *_hc_base f32 [24], *_hc_scale f32 [3]
  KDA layers           kda_qkv: q|k|v rows of the rank's heads [3*W, H];
                       kda_beta [heads, H]; kda_f_a, kda_g_a [128, H];
                       kda_f, kda_g [W, 128]; kda_conv: q|k|v taps [3*W, 4];
                       kda_a_log f32 [heads]; kda_dt_bias f32 [W]; kda_norm
                       f32 [128]; kda_out [H, W] (W = rank heads x 128)
  MLA layers           mla_q_a [1536, H], mla_q_norm [1536], mla_kv_a [512, H],
                       mla_kv_norm [512] (replicated); mla_q_b [heads*256,
                       1536]; mla_uk [heads, 512, 256]; mla_uv [heads, 256,
                       512]; mla_out [H, heads*256]
  dense layers 0-2     ffn_gate_up: gate then up rows [2*12288/TP, H];
                       ffn_down [H, 12288/TP]
  routed layers        router [288, H], router_bias f32 [288] (replicated);
                       all 288 experts split by intermediate width:
                       moe_w1 (gate) and moe_w3 (up) fp8 [288, 2048/TP, H] with
                       *_scale f32 [288, 2048/TP, H/128]; moe_w2 fp8
                       [288, H, 2048/TP] with moe_w2_scale f32
                       [288, H, 2048/TP/128]; shared_gate_up: gate then up
                       [2*2048/TP, H]; shared_down [H, 2048/TP]
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm5_next_resident_stagepack import (DENSE_INTER, EXPERT_INTER, EXPERTS, FIRST_ROUTED, HIDDEN, KDA_DIM_PER_HEAD,  # noqa: E402
                                          KDA_HEADS, LAYERS, LATENT, MLA_HEADS, NOPE, REVISION, TOP_K, VDIM, VOCAB,
                                          SourceReader, is_kda)
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble, source_identity  # noqa: E402
from spark_pack_common import PackFailure  # noqa: E402

TP = 16
PREFIX = "glm53flash"
SCALE_BLOCK = 128
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class RankPacks:
    def __init__(self, out_dir, resume, ranks, identity):
        self.out_dir = Path(out_dir)
        self.ranks = ranks
        self.writers = {rank: PayloadWriter(self.payload(rank), identity, resume=resume and self.payload(rank).exists()) for rank in ranks}

    def payload(self, rank):
        return self.out_dir / f"{PREFIX}.tp{TP}.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"{PREFIX}.tp{TP}.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind):
        if rank in self.writers:
            array = np.ascontiguousarray(array)
            self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def add_bytes(self, rank, name, blob, kind, shape):
        if rank in self.writers:
            self.writers[rank].add(name, blob, kind, shape)

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)


def window(total, rank):
    count = total // TP
    return slice(rank * count, (rank + 1) * count)


def f32_vector(source, name):
    return source.spine_f32(name).reshape(-1)


def pack_kda(source, packs, a, name):
    width = KDA_HEADS * KDA_DIM_PER_HEAD
    projections = {key: source.spine_bf16(f"{a}{key}_proj.weight") for key in ("q", "k", "v", "b", "f_b", "g_b", "o")}
    convs = {key: source.spine_bf16(f"{a}{key}_conv1d.weight") for key in ("q", "k", "v")}
    a_log = f32_vector(source, f"{a}A_log")
    dt_bias = f32_vector(source, f"{a}dt_bias")
    packs.replicated(name + "kda_f_a", source.spine_bf16(f"{a}f_a_proj.weight"), "bf16")
    packs.replicated(name + "kda_g_a", source.spine_bf16(f"{a}g_a_proj.weight"), "bf16")
    packs.replicated(name + "kda_norm", f32_vector(source, f"{a}o_norm.weight"), "f32")
    for rank in packs.ranks:
        rows, heads = window(width, rank), window(KDA_HEADS, rank)
        packs.add(rank, name + "kda_qkv", np.concatenate([projections[key][rows] for key in ("q", "k", "v")]), "bf16")
        packs.add(rank, name + "kda_beta", projections["b"][heads], "bf16")
        packs.add(rank, name + "kda_f", projections["f_b"][rows], "bf16")
        packs.add(rank, name + "kda_g", projections["g_b"][rows], "bf16")
        packs.add(rank, name + "kda_conv", np.concatenate([convs[key][rows] for key in ("q", "k", "v")]), "bf16")
        packs.add(rank, name + "kda_a_log", a_log[heads], "f32")
        packs.add(rank, name + "kda_dt_bias", dt_bias[rows], "f32")
        packs.add(rank, name + "kda_out", projections["o"][:, rows], "bf16")


def pack_mla(source, packs, a, name):
    q_b = source.spine_bf16(f"{a}q_b_proj.weight")
    kv_b = source.spine_bf16(f"{a}kv_b_proj.weight").reshape(MLA_HEADS, NOPE + VDIM, LATENT)
    out = source.spine_bf16(f"{a}o_proj.weight")
    packs.replicated(name + "mla_q_a", source.spine_bf16(f"{a}q_a_proj.weight"), "bf16")
    packs.replicated(name + "mla_q_norm", source.spine_bf16(f"{a}q_a_layernorm.weight").reshape(-1), "bf16")
    packs.replicated(name + "mla_kv_a", source.spine_bf16(f"{a}kv_a_proj_with_mqa.weight"), "bf16")
    packs.replicated(name + "mla_kv_norm", source.spine_bf16(f"{a}kv_a_layernorm.weight").reshape(-1), "bf16")
    for rank in packs.ranks:
        heads = window(MLA_HEADS, rank)
        packs.add(rank, name + "mla_q_b", q_b[window(MLA_HEADS * NOPE, rank)], "bf16")
        packs.add(rank, name + "mla_uk", kv_b[heads, :NOPE, :].transpose(0, 2, 1), "bf16")
        packs.add(rank, name + "mla_uv", kv_b[heads, NOPE:, :], "bf16")
        packs.add(rank, name + "mla_out", out[:, window(MLA_HEADS * VDIM, rank)], "bf16")


def pack_gated(source, packs, m, name, gate_up, down, inter):
    gate = source.spine_bf16(f"{m}gate_proj.weight")
    up = source.spine_bf16(f"{m}up_proj.weight")
    projection = source.spine_bf16(f"{m}down_proj.weight")
    for rank in packs.ranks:
        rows = window(inter, rank)
        packs.add(rank, name + gate_up, np.concatenate((gate[rows], up[rows])), "bf16")
        packs.add(rank, name + down, projection[:, rows], "bf16")


def pack_experts(source, packs, m, name):
    prefix = f"{m}experts"
    rows = EXPERT_INTER // TP
    for rank in packs.ranks:
        first = rank * rows
        for target, projection in (("moe_w1", "gate"), ("moe_w3", "up")):
            names = [f"{prefix}.{expert}.{projection}_proj.weight" for expert in range(EXPERTS)]
            packs.add_bytes(rank, name + target, b"".join(source.expert_payload(n, first, first + rows, 0, HIDDEN) for n in names), "fp8",
                            (EXPERTS, rows, HIDDEN))
            packs.add_bytes(rank, name + target + "_scale", b"".join(source.expert_scale(n, first, first + rows, 0, HIDDEN) for n in names),
                            "f32", (EXPERTS, rows, HIDDEN // SCALE_BLOCK))
        names = [f"{prefix}.{expert}.down_proj.weight" for expert in range(EXPERTS)]
        packs.add_bytes(rank, name + "moe_w2", b"".join(source.expert_payload(n, 0, HIDDEN, first, first + rows) for n in names), "fp8",
                        (EXPERTS, HIDDEN, rows))
        packs.add_bytes(rank, name + "moe_w2_scale", b"".join(source.expert_scale(n, 0, HIDDEN, first, first + rows) for n in names), "f32",
                        (EXPERTS, HIDDEN, rows // SCALE_BLOCK))


def pack_layer(source, packs, layer):
    p = f"model.language_model.layers.{layer}."
    a, m, name = p + "self_attn.", p + "mlp.", f"layers.{layer}."
    packs.replicated(name + "attn_norm", source.spine_bf16(p + "input_layernorm.weight").reshape(-1), "bf16")
    packs.replicated(name + "post_norm", source.spine_bf16(p + "post_attention_layernorm.weight").reshape(-1), "bf16")
    for site in ("attn", "ffn"):
        for part in ("fn", "base", "scale"):
            packs.replicated(f"{name}{site}_hc_{part}", source.spine_f32(f"{p}hc_{site}_{part}"), "f32")
    if is_kda(layer):
        pack_kda(source, packs, a, name)
    else:
        pack_mla(source, packs, a, name)
    if layer < FIRST_ROUTED:
        pack_gated(source, packs, m, name, "ffn_gate_up", "ffn_down", DENSE_INTER)
        return
    packs.replicated(name + "router", source.spine_bf16(m + "gate.weight"), "bf16")
    packs.replicated(name + "router_bias", f32_vector(source, m + "gate.e_score_correction_bias"), "f32")
    pack_experts(source, packs, m, name)
    pack_gated(source, packs, m + "shared_experts.", name, "shared_gate_up", "shared_down", EXPERT_INTER)


def finish(packs, revision, contract_sha256):
    vocab_rows = VOCAB // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        layout = {"model": "glm-5.3-flash", "tp_degree": TP, "tp_rank": rank, "hidden": HIDDEN, "layers": LAYERS, "vocab": VOCAB,
                  "vocab_first": rank * vocab_rows, "vocab_rows": vocab_rows, "ffn_rows": DENSE_INTER // TP, "experts": EXPERTS,
                  "local_experts": EXPERTS, "expert_first": 0, "top_k": TOP_K, "shared_rows": EXPERT_INTER // TP, "moe_rows": EXPERT_INTER // TP}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "expert_codec": "fp8", "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, layout, packs.writers[rank].manifest)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tp-degree", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--ranks")
    args = parser.parse_args()
    global TP
    TP = args.tp_degree
    if args.ranks is None:
        args.ranks = ",".join(str(rank) for rank in range(TP))
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    if (EXPERT_INTER // TP) % SCALE_BLOCK:
        raise PackFailure(f"TP{TP} splits the experts' intermediate width inside a 128-wide scale block")
    source = SourceReader(args.checkpoint)
    text = source.config.get("text_config", source.config)
    if text.get("num_hidden_layers") != LAYERS or text.get("hidden_size") != HIDDEN or text.get("vocab_size") != VOCAB:
        raise PackFailure("the checkpoint is not the GLM 5.3 Flash geometry this packer pins")
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs = RankPacks(args.output_dir, args.resume, ranks, source_identity(args.checkpoint, PREFIX, files=(args.contract,)))
    embed = source.spine_bf16("model.language_model.embed_tokens.weight")
    head = source.spine_bf16("lm_head.weight")
    for rank in ranks:
        rows = window(VOCAB, rank)
        packs.add(rank, "embed", embed[rows], "bf16")
        packs.add(rank, "head", head[rows], "bf16")
    packs.replicated("final_norm", source.spine_bf16("model.language_model.norm.weight").reshape(-1), "bf16")
    for layer in range(LAYERS):
        pack_layer(source, packs, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, REVISION, contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"glm53flash_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
