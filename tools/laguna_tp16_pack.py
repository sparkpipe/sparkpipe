#!/usr/bin/env python3
"""Laguna S 2.1 BF16 checkpoint -> TP named stage packs (sixteen by default) for the Gemma-core stage runner.

--tp-degree sets the rank count (16 for the Sparks; 1, 2, 4 or 8 for Metal
ranks); the splits below are of 1/degree each, shown for 16, and the sliding
layers are padded to 80 heads at every degree so the layout matches the
geometry's. Output per rank: <out>/laguna.tp<degree>.rank<NN>.pack (tools/spark_named_pack.py
container), plus <pack>.sha256 and the empty-range <pack>.experts v2
manifest (the rank's experts are plain resident tensors).

Rank r holds, all bf16 unless noted:
  embed, head         vocab rows [r*6272, +6272) of embed_tokens and of the
                      untied lm_head
  head_norm           model.norm as f32 (plain w)
  per layer           input_layernorm as input_norm and
                      post_attention_layernorm as pre_ff_norm, f32 (plain w;
                      Laguna blocks are pre-norm only); q_norm / k_norm f32
  attention           q_proj|k_proj|v_proj fused, rank rows [r*n/16, +n/16).
                      Sliding layers pad 72 query heads to 80: each of the 8
                      GQA groups of 9 heads gets one zero head appended, so
                      padded head 10g+j is real head 9g+j (j < 9) and kv head
                      g serves padded heads [10g, 10g+10). o_proj columns
                      follow the same padding, rank columns [r*q/16, +q/16).
  attn_gate           g_proj rows of the rank's padded heads (5 sliding, 3
                      full) plus one zero row, so the gate GEMM width is even
  ffn_gate_up/down    layer 0: the dense MLP (12288) split in 64-row blocks
                      (768 rows per rank); layers 1..47: the shared expert
                      (1024, 64 rows per rank); gate|up fused [2*rows, H],
                      down columns [H, rows]
  MoE (layers 1..47)  router_proj = mlp.gate.weight [256, H] replicated,
                      expert_bias = mlp.experts.e_score_correction_bias f32
                      [256] replicated, experts [16r, 16r+16) as moe_w1
                      (gate_proj) and moe_w3 (up_proj) [16, 1024, H] and
                      moe_w2 (down_proj) [16, H, 1024]

The packer checks config.json against the frozen contract, every tensor's
dtype and shape, and that the checkpoint holds exactly the tensors it packs.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_tp16_pack  # noqa: E402
from gemma4_tp16_pack import EXPERTS_MAGIC, EXPERTS_VERSION, TP, Checkpoint, PackFailure, RankPacks, ffn_window  # noqa: E402
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, assemble, source_identity  # noqa: E402

PREFIX = "model."
SLIDING_PADDED_HEADS = 80


def gate_rows(heads):
    local = heads // TP
    return local + (local & 1)


def expected_tensors(config):
    names = {"lm_head.weight", PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"}
    for layer in range(config["num_hidden_layers"]):
        p = f"{PREFIX}layers.{layer}."
        names |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        names |= {p + f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "q_norm", "k_norm")}
        if layer in config["mlp_only_layers"]:
            names |= {p + f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
        else:
            names |= {p + "mlp.gate.weight", p + "mlp.experts.e_score_correction_bias"}
            names |= {p + f"mlp.shared_expert.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
            names |= {p + f"mlp.experts.{e}.{n}.weight" for e in range(config["num_experts"]) for n in ("gate_proj", "up_proj", "down_proj")}
    return names


def validate(config, contract):
    source = contract["source"]["config_json"]
    heads = config["num_attention_heads_per_layer"]
    layers = config["num_hidden_layers"]
    full = ["full_attention" if index % 4 == 0 else "sliding_attention" for index in range(layers)]
    rope = config["rope_parameters"]
    checks = [
        (config["model_type"] == "laguna", "model_type is laguna"),
        (config["hidden_size"] == source["hidden_size"] == 3072, "hidden 3072"),
        (layers == source["num_hidden_layers"] == 48, "48 layers"),
        (config["vocab_size"] == source["vocab_size"] == 100352 and config["vocab_size"] % TP == 0, "vocab 100352"),
        (config["head_dim"] == 128 and config["num_key_value_heads"] == 8, "8 KV heads of 128"),
        (config["layer_types"] == full, "full attention at layer % 4 == 0, sliding elsewhere"),
        (heads == [48 if kind == "full_attention" else 72 for kind in full], "48 full / 72 sliding query heads"),
        (config["sliding_window"] == 512, "sliding window 512"),
        (config["gating"] == "per-head" and not config.get("swa_attention_sink_enabled", False), "per-head gate, no sinks"),
        (config.get("hidden_act", "silu") == "silu" and not config.get("attention_bias", False), "silu, no attention bias"),
        (config["mlp_only_layers"] == [0] and config["decoder_sparse_step"] == 1, "layer 0 dense, layers 1..47 MoE"),
        (config["num_experts"] == 256 and config["num_experts_per_tok"] == 10 and config["num_experts"] % TP == 0, "256 experts, top 10"),
        (config["moe_intermediate_size"] == 1024 and config["shared_expert_intermediate_size"] == 1024, "expert and shared intermediate 1024"),
        (config["intermediate_size"] == 12288 and config["intermediate_size"] % 64 == 0, "dense intermediate 12288"),
        (config["norm_topk_prob"] and config["moe_routed_scaling_factor"] == 2.5, "normalized top-k times 2.5"),
        (not config.get("moe_apply_router_weight_on_input", False) and config.get("moe_router_logit_softcapping", 0.0) == 0.0,
         "router weight on the output, no router softcap"),
        (not config["tie_word_embeddings"], "untied lm_head"),
        (rope["full_attention"]["rope_type"] == "yarn" and rope["full_attention"]["partial_rotary_factor"] == 0.5, "full layers YaRN on 64 dims"),
        (rope["sliding_attention"]["rope_type"] == "default" and rope["sliding_attention"]["rope_theta"] == 10000.0 and
         rope["sliding_attention"].get("partial_rotary_factor", 1.0) == 1.0, "sliding layers default rope 1e4 on 128 dims"),
    ]
    failed = [text for ok, text in checks if not ok]
    if failed:
        raise PackFailure("config.json differs from the packer's Laguna TP16 layout: " + "; ".join(failed))


def padded_count(heads):
    return heads if heads % 16 == 0 else SLIDING_PADDED_HEADS


def pad_heads(matrix, heads, groups, axis, width):
    padded = padded_count(heads)
    matrix = np.asarray(matrix)
    if padded == heads:
        return matrix
    if heads % groups or padded % groups:
        raise PackFailure(f"{heads} heads do not pad to {padded} in {groups} whole GQA groups")
    shape = list(matrix.shape)
    grouped = matrix.reshape(shape[:axis] + [groups, heads // groups * width] + shape[axis + 1:])
    zeros = np.zeros(shape[:axis] + [groups, (padded - heads) // groups * width] + shape[axis + 1:], dtype=matrix.dtype)
    return np.concatenate([grouped, zeros], axis=axis + 1).reshape(shape[:axis] + [padded * width] + shape[axis + 1:])


def pack_layer(checkpoint, packs, config, layer):
    hidden = config["hidden_size"]
    p = f"{PREFIX}layers.{layer}."
    name = f"layers.{layer}."
    heads = config["num_attention_heads_per_layer"][layer]
    kv = config["num_key_value_heads"] * config["head_dim"]
    groups = config["num_key_value_heads"]
    padded = padded_count(heads)
    packs.replicated(name + "input_norm", checkpoint.f32(p + "input_layernorm.weight", (hidden,)), "f32")
    packs.replicated(name + "pre_ff_norm", checkpoint.f32(p + "post_attention_layernorm.weight", (hidden,)), "f32")
    packs.replicated(name + "attn_q_norm", checkpoint.f32(p + "self_attn.q_norm.weight", (128,)), "f32")
    packs.replicated(name + "attn_k_norm", checkpoint.f32(p + "self_attn.k_norm.weight", (128,)), "f32")
    q = pad_heads(checkpoint.array(p + "self_attn.q_proj.weight", (heads * 128, hidden)), heads, groups, 0, 128)
    k = checkpoint.array(p + "self_attn.k_proj.weight", (kv, hidden))
    v = checkpoint.array(p + "self_attn.v_proj.weight", (kv, hidden))
    o = pad_heads(checkpoint.array(p + "self_attn.o_proj.weight", (hidden, heads * 128)), heads, groups, 1, 128)
    g = pad_heads(checkpoint.array(p + "self_attn.g_proj.weight", (heads, hidden)), heads, groups, 0, 1)
    fused = np.concatenate([q, k, v])
    if fused.shape[0] != padded * 128 + 2 * kv or fused.shape[0] % TP or o.shape[1] != padded * 128 or g.shape[0] != padded:
        raise PackFailure(f"layer {layer}: padded attention shapes {fused.shape} {o.shape} {g.shape} do not match {padded} heads")
    rows, columns, local_heads = fused.shape[0] // TP, o.shape[1] // TP, padded // TP
    width = gate_rows(padded)
    if layer in config["mlp_only_layers"]:
        inter, mlp = config["intermediate_size"], p + "mlp."
    else:
        inter, mlp = config["shared_expert_intermediate_size"], p + "mlp.shared_expert."
    gate = checkpoint.array(mlp + "gate_proj.weight", (inter, hidden))
    up = checkpoint.array(mlp + "up_proj.weight", (inter, hidden))
    down = checkpoint.array(mlp + "down_proj.weight", (hidden, inter))
    for rank in packs.ranks:
        packs.add(rank, name + "attn_qkv", fused[rank * rows:(rank + 1) * rows], "bf16")
        packs.add(rank, name + "attn_out", o[:, rank * columns:(rank + 1) * columns], "bf16")
        rank_gate = np.zeros((width, hidden), dtype=np.uint16)
        rank_gate[:local_heads] = g[rank * local_heads:(rank + 1) * local_heads]
        packs.add(rank, name + "attn_gate", rank_gate, "bf16")
        window = ffn_window(inter, rank)
        packs.add(rank, name + "ffn_gate_up", np.concatenate((gate[window], up[window])), "bf16")
        packs.add(rank, name + "ffn_down", down[:, window], "bf16")
    if layer in config["mlp_only_layers"]:
        return
    experts, expert_inter = config["num_experts"], config["moe_intermediate_size"]
    local = experts // TP
    packs.replicated(name + "router_proj", checkpoint.array(p + "mlp.gate.weight", (experts, hidden)), "bf16")
    packs.replicated(name + "expert_bias", checkpoint.f32(p + "mlp.experts.e_score_correction_bias", (experts,)), "f32")
    for rank in packs.ranks:
        owned = range(rank * local, (rank + 1) * local)
        for tensor, source, shape in (("moe_w1", "gate_proj", (expert_inter, hidden)), ("moe_w3", "up_proj", (expert_inter, hidden)),
                                      ("moe_w2", "down_proj", (hidden, expert_inter))):
            stacked = np.stack([checkpoint.array(f"{p}mlp.experts.{expert}.{source}.weight", shape) for expert in owned])
            packs.add(rank, name + tensor, stacked, "bf16")


def finish(packs, config, revision, contract_sha256):
    vocab_rows = config["vocab_size"] // TP
    local = config["num_experts"] // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        window = ffn_window(config["intermediate_size"], rank)
        settings = {"model": "laguna_s_2_1", "tp_degree": TP, "tp_rank": rank, "hidden": config["hidden_size"],
                    "layers": config["num_hidden_layers"], "vocab": config["vocab_size"], "vocab_first": rank * vocab_rows,
                    "vocab_rows": vocab_rows, "ffn_rows": window.stop - window.start, "full_period": 4, "full_phase": 0,
                    "sliding_window": config["sliding_window"], "sliding_heads_padded": SLIDING_PADDED_HEADS,
                    "experts": config["num_experts"], "local_experts": local, "expert_first": rank * local,
                    "top_k": config["num_experts_per_tok"]}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "attn_qkv_layout": "fused_windows", "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, settings,
                 packs.writers[rank].manifest)
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
    TP = gemma4_tp16_pack.TP = args.tp_degree
    args.ranks = args.ranks or ",".join(str(rank) for rank in range(TP))
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    config_bytes = (args.checkpoint / "config.json").read_bytes()
    contract_bytes = args.contract.read_bytes()
    contract = json.loads(contract_bytes)
    if hashlib.sha256(config_bytes).hexdigest() != contract["source"]["config_json"]["sha256"]:
        raise PackFailure("config.json sha256 differs from the contract's frozen config")
    config = json.loads(config_bytes)
    validate(config, contract)
    checkpoint = Checkpoint(args.checkpoint)
    missing = expected_tensors(config) ^ set(checkpoint.weight_map)
    if missing:
        raise PackFailure(f"checkpoint tensors differ from the Laguna layout: {sorted(missing)[:8]} ({len(missing)} names)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs = RankPacks(args.output_dir, args.resume, ranks, "laguna", source_identity(args.checkpoint, "laguna", files=(args.contract,)))
    hidden, vocab_rows = config["hidden_size"], config["vocab_size"] // TP
    embed = checkpoint.array(PREFIX + "embed_tokens.weight", (config["vocab_size"], hidden))
    head = checkpoint.array("lm_head.weight", (config["vocab_size"], hidden))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
        packs.add(rank, "head", head[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("head_norm", checkpoint.f32(PREFIX + "norm.weight", (hidden,)), "f32")
    for layer in range(config["num_hidden_layers"]):
        pack_layer(checkpoint, packs, config, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, config, contract["revision"], hashlib.sha256(contract_bytes).hexdigest())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"laguna_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
