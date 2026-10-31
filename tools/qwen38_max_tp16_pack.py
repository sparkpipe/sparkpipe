#!/usr/bin/env python3
"""Qwen 3.8 Max (NVFP4 routed experts, BF16 spine) checkpoint -> TP1, TP2, TP4 or TP8 named stage packs for the Metal backend.

One pass over the checkpoint (qwen3.8-max-nvfp4-radixark-bf16-spine) writes
every rank. Output per rank: <out>/qwen38_max.tp<T>.rank<NN>.pack
(tools/spark_named_pack.py container), plus <pack>.sha256 and the
empty-range <pack>.experts. Rank r of T holds:
  embed / head        vocab rows [r*248320/T, +248320/T), bf16; head_norm
                      (1 + w) f32 [8192]
  norms               input_norm and post_norm (1 + w) f32 [8192] per layer
  GDN                 key heads [r*16/T, +16/T) and value heads
                      [r*128/T, +128/T): gdn_qkvz = q | k | v | z rows,
                      gdn_ba = b | a rows, gdn_conv q | k | v channels [., 4],
                      A_log / dt_bias f32, gdn_norm raw bf16 [128], gdn_out
                      columns of the rank's value heads
  attention           q_proj (each head's query, then its gate) | k_proj |
                      v_proj fused (34816 rows), rank rows [r*34816/T, +34816/T)
                      ("fused_windows", config attn_rank_heads 0: the Metal
                      ranks gather the windows before their heads run; the
                      CUDA TP16 packs of tools/qwen38_max_tp16_convert.py hold
                      each rank's own heads instead); q and k norms (1 + w)
                      replicated; o_proj columns [r*16384/T, +16384/T)
  MoE                 router [512, 8192] and the shared expert gate [1, 8192]
                      replicated; local experts [r*512/T, +512/T) as nvfp4
                      moe_w1 (gate) / moe_w3 (up) [E*2048, 8192] and moe_w2
                      (down) [E*8192, 2048], each with a ue4m3 scale plane
                      whose E f32 weight_scale_2 globals come first; the
                      shared expert rows [r*2048/T, +2048/T): gate|up
                      [2*2048/T, 8192] and down columns [8192, 2048/T]
The experts' input_scale (the checkpoint's W4A4 activation scale) is not
packed: CUDA and Metal multiply BF16 activations by the NVFP4 weights. The MTP
layer is not packed. TP16 serving packs come from tools/qwen38_max_tp16_convert.py.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_flash_tp16_pack as flash
from qwen38_flash_tp16_pack import Checkpoint, PackFailure, bf16_to_f32, expert_planes, folded
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, assemble, source_identity

TP = 8
TP_DEGREES = (1, 2, 4, 8)
HIDDEN = 8192
LAYERS = 92
VOCAB = 248320
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 128
GDN_KEY_DIM = 128
GDN_VALUE_DIM = 128
GDN_QK = GDN_KEY_HEADS * GDN_KEY_DIM
GDN_V = GDN_VALUE_HEADS * GDN_VALUE_DIM
GDN_CONV = 4
ATTN_HEADS = 64
ATTN_KV_HEADS = 4
HEAD_DIM = 256
ATTN_Q = ATTN_HEADS * HEAD_DIM
ATTN_QG = 2 * ATTN_Q
ATTN_KV = ATTN_KV_HEADS * HEAD_DIM
ATTN_QKV = ATTN_QG + 2 * ATTN_KV
FULL_PERIOD = 4
FULL_PHASE = 3
EXPERTS = 512
TOP_K = 10
MOE_INTER = 2048
SHARED_INTER = 2048
ROW_TILE = 64
PREFIX = "model."
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class RankPacks(flash.RankPacks):
    def payload(self, rank):
        return self.out_dir / f"qwen38_max.tp{TP}.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"qwen38_max.tp{TP}.rank{rank:02d}.pack"


def rows_of(rank, width):
    return slice(rank * (width // TP), (rank + 1) * (width // TP))


def pack_globals(checkpoint, packs):
    embed = checkpoint.array(PREFIX + "embed_tokens.weight", "BF16", (VOCAB, HIDDEN))
    head = checkpoint.array("lm_head.weight", "BF16", (VOCAB, HIDDEN))
    for rank in packs.ranks:
        packs.add(rank, "embed", embed[rows_of(rank, VOCAB)], "bf16")
        packs.add(rank, "head", head[rows_of(rank, VOCAB)], "bf16")
    packs.replicated("head_norm", folded(checkpoint, PREFIX + "norm.weight", HIDDEN), "f32")


def pack_norms(checkpoint, packs, layer):
    p = f"{PREFIX}layers.{layer}."
    packs.replicated(f"layers.{layer}.input_norm", folded(checkpoint, p + "input_layernorm.weight", HIDDEN), "f32")
    packs.replicated(f"layers.{layer}.post_norm", folded(checkpoint, p + "post_attention_layernorm.weight", HIDDEN), "f32")


def pack_gdn(checkpoint, packs, layer):
    p = f"{PREFIX}layers.{layer}.linear_attn."
    qkv = checkpoint.array(p + "in_proj_qkv.weight", "BF16", (2 * GDN_QK + GDN_V, HIDDEN))
    z = checkpoint.array(p + "in_proj_z.weight", "BF16", (GDN_V, HIDDEN))
    out = checkpoint.array(p + "out_proj.weight", "BF16", (HIDDEN, GDN_V))
    beta = checkpoint.array(p + "in_proj_b.weight", "BF16", (GDN_VALUE_HEADS, HIDDEN))
    decay = checkpoint.array(p + "in_proj_a.weight", "BF16", (GDN_VALUE_HEADS, HIDDEN))
    conv = checkpoint.array(p + "conv1d.weight", "BF16", (2 * GDN_QK + GDN_V, 1, GDN_CONV)).reshape(2 * GDN_QK + GDN_V, GDN_CONV)
    a_log = bf16_to_f32(checkpoint.array(p + "A_log", "BF16", (GDN_VALUE_HEADS,)))
    dt_bias = bf16_to_f32(checkpoint.array(p + "dt_bias", "BF16", (GDN_VALUE_HEADS,)))
    norm = checkpoint.array(p + "norm.weight", "BF16", (GDN_VALUE_DIM,))
    name = f"layers.{layer}."
    for rank in packs.ranks:
        q, k = rows_of(rank, GDN_QK), slice(GDN_QK + rows_of(rank, GDN_QK).start, GDN_QK + rows_of(rank, GDN_QK).stop)
        v = slice(2 * GDN_QK + rows_of(rank, GDN_V).start, 2 * GDN_QK + rows_of(rank, GDN_V).stop)
        heads = rows_of(rank, GDN_VALUE_HEADS)
        packs.add(rank, name + "gdn_qkvz", np.concatenate((qkv[q], qkv[k], qkv[v], z[rows_of(rank, GDN_V)])), "bf16")
        packs.add(rank, name + "gdn_ba", np.concatenate((beta[heads], decay[heads])), "bf16")
        packs.add(rank, name + "gdn_conv", np.concatenate((conv[q], conv[k], conv[v])), "bf16")
        packs.add(rank, name + "gdn_a_log", a_log[heads], "f32")
        packs.add(rank, name + "gdn_dt_bias", dt_bias[heads], "f32")
        packs.add(rank, name + "gdn_norm", norm, "bf16")
        packs.add(rank, name + "gdn_out", out[:, rows_of(rank, GDN_V)], "bf16")


def pack_attention(checkpoint, packs, layer):
    p = f"{PREFIX}layers.{layer}.self_attn."
    fused = np.concatenate((checkpoint.array(p + "q_proj.weight", "BF16", (ATTN_QG, HIDDEN)),
                            checkpoint.array(p + "k_proj.weight", "BF16", (ATTN_KV, HIDDEN)),
                            checkpoint.array(p + "v_proj.weight", "BF16", (ATTN_KV, HIDDEN))))
    o = checkpoint.array(p + "o_proj.weight", "BF16", (HIDDEN, ATTN_Q))
    name = f"layers.{layer}."
    for rank in packs.ranks:
        packs.add(rank, name + "attn_qkv", fused[rows_of(rank, ATTN_QKV)], "bf16")
        packs.add(rank, name + "attn_out", o[:, rows_of(rank, ATTN_Q)], "bf16")
    packs.replicated(name + "attn_q_norm", folded(checkpoint, p + "q_norm.weight", HEAD_DIM), "f32")
    packs.replicated(name + "attn_k_norm", folded(checkpoint, p + "k_norm.weight", HEAD_DIM), "f32")


def pack_moe(checkpoint, packs, layer):
    p = f"{PREFIX}layers.{layer}.mlp."
    name = f"layers.{layer}."
    local = EXPERTS // TP
    packs.replicated(name + "moe_router", checkpoint.array(p + "gate.weight", "BF16", (EXPERTS, HIDDEN)), "bf16")
    packs.replicated(name + "moe_shared_gate", checkpoint.array(p + "shared_expert_gate.weight", "BF16", (1, HIDDEN)), "bf16")
    shared_gate = checkpoint.array(p + "shared_expert.gate_proj.weight", "BF16", (SHARED_INTER, HIDDEN))
    shared_up = checkpoint.array(p + "shared_expert.up_proj.weight", "BF16", (SHARED_INTER, HIDDEN))
    shared_down = checkpoint.array(p + "shared_expert.down_proj.weight", "BF16", (HIDDEN, SHARED_INTER))
    for rank in packs.ranks:
        rows = rows_of(rank, SHARED_INTER)
        packs.add(rank, name + "moe_shared_gate_up", np.concatenate((shared_gate[rows], shared_up[rows])), "bf16")
        packs.add(rank, name + "moe_shared_down", shared_down[:, rows], "bf16")
        experts = range(rank * local, (rank + 1) * local)
        for kind, proj, rows_, cols in (("moe_w1", "gate_proj", MOE_INTER, HIDDEN), ("moe_w3", "up_proj", MOE_INTER, HIDDEN),
                                        ("moe_w2", "down_proj", HIDDEN, MOE_INTER)):
            if name + kind in packs.writers[rank].manifest and name + kind + ".scale" in packs.writers[rank].manifest:
                continue
            payload, globals_, planes = expert_planes(checkpoint, p + "experts.", proj, experts, rows_, cols)
            packs.writers[rank].add(name + kind, payload.tobytes(), "nvfp4_e2m1", (local * rows_, cols))
            packs.writers[rank].add(name + kind + ".scale", globals_.tobytes() + planes.tobytes(), "ue4m3_f32_global",
                                    (local, rows_, cols // 16), {"global_scales": local})


def finish(packs, revision, contract_sha256):
    local = EXPERTS // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        config = {"model": "qwen38_max", "tp_degree": TP, "tp_rank": rank, "hidden": HIDDEN, "layers": LAYERS, "vocab": VOCAB,
                  "vocab_first": rank * (VOCAB // TP), "vocab_rows": VOCAB // TP, "full_period": FULL_PERIOD, "full_phase": FULL_PHASE,
                  "experts": EXPERTS, "local_experts": local, "expert_first": rank * local, "top_k": TOP_K, "moe_inter": MOE_INTER,
                  "shared_rows": SHARED_INTER // TP, "gdn_heads": GDN_VALUE_HEADS // TP, "attn_heads_local": ATTN_HEADS // TP,
                  "attn_rank_heads": 0}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN, "linear_codec": "bf16",
               "expert_codec": "nvfp4_e2m1", "attn_qkv_layout": "fused_windows", "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, config, packs.writers[rank].manifest)
        digest = hashlib.sha256()
        with final.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
        Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
        print(f"rank {rank:02d} {final.stat().st_size} {digest.hexdigest()}", flush=True)


def check_geometry(config):
    rope = config.get("rope_parameters") or {}
    expected = {"hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "vocab_size": VOCAB, "num_attention_heads": ATTN_HEADS,
                "num_key_value_heads": ATTN_KV_HEADS, "head_dim": HEAD_DIM, "linear_num_key_heads": GDN_KEY_HEADS,
                "linear_num_value_heads": GDN_VALUE_HEADS, "linear_key_head_dim": GDN_KEY_DIM, "linear_value_head_dim": GDN_VALUE_DIM,
                "linear_conv_kernel_dim": GDN_CONV, "num_experts": EXPERTS, "num_experts_per_tok": TOP_K, "moe_intermediate_size": MOE_INTER,
                "shared_expert_intermediate_size": SHARED_INTER, "full_attention_interval": FULL_PERIOD, "attn_output_gate": True,
                "output_gate_type": "swish", "tie_word_embeddings": False, "rms_norm_eps": 1e-6}
    differing = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if rope.get("rope_theta") != 10000000 or rope.get("partial_rotary_factor") != 0.25:
        differing["rope_parameters"] = (rope, {"rope_theta": 10000000, "partial_rotary_factor": 0.25})
    kinds = ["full_attention" if layer % FULL_PERIOD == FULL_PHASE else "linear_attention" for layer in range(LAYERS)]
    if config.get("layer_types") != kinds:
        differing["layer_types"] = "not every fourth layer from the fourth full attention"
    if differing:
        raise PackFailure(f"checkpoint geometry differs from the packer's: {differing}")


def main():
    global TP
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, choices=TP_DEGREES, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ranks", help="comma-separated ranks to write (default every rank)")
    args = parser.parse_args()
    TP = args.tp_degree
    if (SHARED_INTER // TP) % ROW_TILE != 0 or GDN_KEY_HEADS % TP != 0 or ATTN_QKV % TP != 0:
        raise PackFailure(f"TP{TP} does not split the shared expert into 64-row tiles, the gated-delta key heads or the fused attention rows")
    ranks = sorted({int(rank) for rank in args.ranks.split(",")}) if args.ranks else list(range(TP))
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    check_geometry(json.loads((args.checkpoint / "config.json").read_text()))
    receipt = args.checkpoint / "DOWNLOAD-RECEIPT.json"
    revision = json.loads(receipt.read_text()).get("revision", "unknown") if receipt.exists() else "unknown"
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    packs = RankPacks(args.output_dir, args.resume, ranks, source_identity(args.checkpoint, "qwen38_max_tp16_pack", TP, files=(args.contract,)))
    pack_globals(checkpoint, packs)
    for layer in range(LAYERS):
        pack_norms(checkpoint, packs, layer)
        if layer % FULL_PERIOD == FULL_PHASE:
            pack_attention(checkpoint, packs, layer)
        else:
            pack_gdn(checkpoint, packs, layer)
        pack_moe(checkpoint, packs, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, revision, contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen38_max_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
