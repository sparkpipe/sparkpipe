#!/usr/bin/env python3
"""Solar Open 2 (SolarOpen2ForCausalLM) BF16 checkpoint -> sixteen TP16 named stage packs.

Output per rank: <out>/solar_open2.tp16.rank<NN>.pack (tools/spark_named_pack.py
container) plus <pack>.sha256 and the empty-range <pack>.experts (experts are
plain tensors). Pass --ranks to pack only some ranks, for example one rank per
node.

Rank r holds, bf16 unless noted:
  embed, head         vocab rows [r*V/16, +V/16) of embed_tokens and lm_head
  head_norm           model.norm as f32
  every layer         input / post-attention norms as f32
  KDA layers          kda_qkv: q|k|v rows of the rank's 4 heads [1536, H];
                      kda_f_a, kda_g_a [128, H] (replicated); kda_f, kda_g:
                      f_b / g_b rows [512, 128]; kda_beta [4, H]; kda_conv:
                      the q|k|v depthwise taps [1536, 4]; kda_a_log f32 [4];
                      kda_dt_bias f32 [512]; kda_norm [128]; kda_out: o_proj
                      columns [H, 512]
  GQA layers          gqa_qkv: rows [640r, +640) of q|k|v stacked
                      [10240, H]; gqa_gate: g_proj rows of the rank's 4
                      query heads [512, H]; gqa_out: o_proj columns [H, 512]
  MoE layers          router [E, H] and expert_bias f32 [E] (replicated);
                      every routed expert's rank slice of I/16 = 80 rows as
                      moe_w1 / moe_w3 [E, 80, H] and moe_w2 [E, H, 80]; the
                      shared expert's rank slice of 80 rows padded with
                      zero rows to 128: shared_gate_up [256, H],
                      shared_down [H, 128]
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ling3_tp16_pack import EXPERTS_MAGIC, EXPERTS_VERSION, Checkpoint, PackFailure, RankPacks, pad_rows  # noqa: E402
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, assemble, source_identity  # noqa: E402

TP = 16
SHARED_ROWS = 128
PREFIX = "solar_open2"


def gqa_layers(config):
    layers = config.get("gqa_layers")
    if layers is None:
        raise PackFailure("config.json names no gqa_layers")
    return set(layers)


def pack_layer(checkpoint, packs, config, layer):
    hidden = config["hidden_size"]
    p = f"model.layers.{layer}."
    a = p + "self_attn."
    name = f"layers.{layer}."
    packs.replicated(name + "input_norm", checkpoint.bf16_as_f32(p + "input_layernorm.weight", (hidden,)), "f32")
    packs.replicated(name + "post_norm", checkpoint.bf16_as_f32(p + "post_attention_layernorm.weight", (hidden,)), "f32")
    if layer in gqa_layers(config):
        heads, kv_heads, head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        width, kv_width = heads * head_dim, kv_heads * head_dim
        fused = np.concatenate([np.asarray(checkpoint.bf16(a + "q_proj.weight", (width, hidden))),
                                np.asarray(checkpoint.bf16(a + "k_proj.weight", (kv_width, hidden))),
                                np.asarray(checkpoint.bf16(a + "v_proj.weight", (kv_width, hidden)))])
        gate = checkpoint.bf16(a + "g_proj.weight", (width, hidden))
        out = checkpoint.bf16(a + "o_proj.weight", (hidden, width))
        rank_fused, rank_width = fused.shape[0] // TP, width // TP
        for rank in packs.ranks:
            packs.add(rank, name + "gqa_qkv", fused[rank * rank_fused:(rank + 1) * rank_fused], "bf16")
            packs.add(rank, name + "gqa_gate", gate[rank * rank_width:(rank + 1) * rank_width], "bf16")
            packs.add(rank, name + "gqa_out", out[:, rank * rank_width:(rank + 1) * rank_width], "bf16")
    else:
        linear = config["linear_attn_config"]
        heads, head_dim, kernel = linear["num_heads"], linear["head_dim"], linear["short_conv_kernel_size"]
        width, rank_heads = heads * head_dim, heads // TP
        projections = {key: checkpoint.bf16(a + f"{key}_proj.weight", (width, hidden)) for key in ("q", "k", "v")}
        convs = {key: np.asarray(checkpoint.bf16(a + f"{key}_conv1d.weight", (width, 1, kernel)))[:, 0, :] for key in ("q", "k", "v")}
        f_b = checkpoint.bf16(a + "f_b_proj.weight", (width, head_dim))
        g_b = checkpoint.bf16(a + "g_b_proj.weight", (width, head_dim))
        beta = checkpoint.bf16(a + "b_proj.weight", (heads, hidden))
        a_log = checkpoint.bf16_as_f32(a + "A_log", (1, 1, heads, 1)).reshape(heads)
        dt_bias = checkpoint.bf16_as_f32(a + "dt_bias", (width,))
        out = checkpoint.bf16(a + "o_proj.weight", (hidden, width))
        packs.replicated(name + "kda_f_a", checkpoint.bf16(a + "f_a_proj.weight", (head_dim, hidden)), "bf16")
        packs.replicated(name + "kda_g_a", checkpoint.bf16(a + "g_a_proj.weight", (head_dim, hidden)), "bf16")
        packs.replicated(name + "kda_norm", checkpoint.bf16(a + "o_norm.weight", (head_dim,)), "bf16")
        for rank in packs.ranks:
            rows = slice(rank * rank_heads * head_dim, (rank + 1) * rank_heads * head_dim)
            packs.add(rank, name + "kda_qkv", np.concatenate([projections[key][rows] for key in ("q", "k", "v")]), "bf16")
            packs.add(rank, name + "kda_f", f_b[rows], "bf16")
            packs.add(rank, name + "kda_g", g_b[rows], "bf16")
            packs.add(rank, name + "kda_beta", beta[rank * rank_heads:(rank + 1) * rank_heads], "bf16")
            packs.add(rank, name + "kda_conv", np.concatenate([convs[key][rows] for key in ("q", "k", "v")]), "bf16")
            packs.add(rank, name + "kda_a_log", a_log[rank * rank_heads:(rank + 1) * rank_heads], "f32")
            packs.add(rank, name + "kda_dt_bias", dt_bias[rows], "f32")
            packs.add(rank, name + "kda_out", out[:, rows], "bf16")
    m = p + "mlp."
    experts, inter = config["n_routed_experts"], config["moe_intermediate_size"]
    shared = inter * config["n_shared_experts"]
    expert_rows, shared_rows = inter // TP, shared // TP
    packs.replicated(name + "router", checkpoint.bf16(m + "gate.weight", (experts, hidden)), "bf16")
    packs.replicated(name + "expert_bias", checkpoint.f32(m + "gate.e_score_correction_bias", (experts,)), "f32")
    shared_gate = checkpoint.bf16(m + "shared_experts.gate_proj.weight", (shared, hidden))
    shared_up = checkpoint.bf16(m + "shared_experts.up_proj.weight", (shared, hidden))
    shared_down = checkpoint.bf16(m + "shared_experts.down_proj.weight", (hidden, shared))
    for key, target, shape in (("gate", "moe_w1", (inter, hidden)), ("up", "moe_w3", (inter, hidden)), ("down", "moe_w2", (hidden, inter))):
        matrices = [checkpoint.bf16(m + f"experts.{expert}.{key}_proj.weight", shape) for expert in range(experts)]
        for rank in packs.ranks:
            window = slice(rank * expert_rows, (rank + 1) * expert_rows)
            if key == "down":
                stacked = np.stack([matrix[:, window] for matrix in matrices])
            else:
                stacked = np.stack([matrix[window] for matrix in matrices])
            packs.add(rank, name + target, stacked, "bf16")
    for rank in packs.ranks:
        window = slice(rank * shared_rows, (rank + 1) * shared_rows)
        packs.add(rank, name + "shared_gate_up", np.concatenate((pad_rows(shared_gate[window], SHARED_ROWS), pad_rows(shared_up[window], SHARED_ROWS))), "bf16")
        packs.add(rank, name + "shared_down", pad_rows(np.ascontiguousarray(shared_down[:, window].T), SHARED_ROWS).T, "bf16")


def finish(packs, config, revision, contract_sha256):
    vocab_rows = config["vocab_size"] // TP
    local = config["n_routed_experts"]
    for rank in packs.ranks:
        packs.writers[rank].close()
        layout = {"model": PREFIX, "tp_degree": TP, "tp_rank": rank, "hidden": config["hidden_size"], "layers": config["num_hidden_layers"],
                  "vocab": config["vocab_size"], "vocab_first": rank * vocab_rows, "vocab_rows": vocab_rows,
                  "ffn_rows": 0, "experts": config["n_routed_experts"], "local_experts": local,
                  "expert_first": 0, "expert_rows": config["moe_intermediate_size"] // TP, "top_k": config["num_experts_per_tok"], "shared_rows": SHARED_ROWS}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "expert_codec": "bf16", "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, layout,
                 packs.writers[rank].manifest)
        digest = hashlib.sha256()
        with final.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
        Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
        print(f"rank {rank:02d} {final.stat().st_size} {digest.hexdigest()}", flush=True)


def check_layout(config):
    linear = config["linear_attn_config"]
    expected = {"model_type": "solar_open2", "first_k_dense_replace": 0, "use_rope": False, "use_gqa_gate": True,
                "kda_use_full_proj": False, "kda_allow_neg_eigval": True, "n_shared_experts": 1}
    wrong = {key: config.get(key) for key, value in expected.items() if config.get(key) != value}
    if wrong or linear.get("num_kv_heads") not in (None, linear["num_heads"]) or config.get("kda_gate_lower_bound", -5.0) != -5.0 or \
            config.get("attention_bias") or config.get("use_qk_norm") or config.get("n_group", 1) != 1 or config.get("tie_word_embeddings") or \
            config.get("routed_scaling_factor", 1.0) != 1.0 or not config.get("norm_topk_prob", True):
        raise PackFailure(f"this packer handles the Solar Open 2 layout (NoPE gated GQA, low-rank KDA gates with a -5 bound, one routing group, scale 1, normalized top-k); "
                          f"config differs in {wrong or 'the KDA heads, gate bound, attention bias, qk norm, routing or tied embeddings'}")
    if config["num_attention_heads"] % TP or linear["num_heads"] % TP or config["n_routed_experts"] % TP or config["vocab_size"] % TP or \
            (config["moe_intermediate_size"] * config["n_shared_experts"]) // TP > SHARED_ROWS or \
            config["moe_intermediate_size"] % TP or (config["moe_intermediate_size"] // TP) % 8 or \
            ((config["num_attention_heads"] + 2 * config["num_key_value_heads"]) * config["head_dim"]) % TP:
        raise PackFailure("heads, experts, vocab or the shared expert do not fit the TP16 layout")
    period = sorted(gqa_layers(config))
    if period != list(range(0, config["num_hidden_layers"], 4)):
        raise PackFailure(f"the stage runs GQA on every fourth layer from 0; config has {period}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in range(TP)))
    args = parser.parse_args()
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    config = json.loads((args.checkpoint / "config.json").read_text())
    check_layout(config)
    contract = json.loads(args.contract.read_text())
    receipt = args.checkpoint / "DOWNLOAD-RECEIPT.json"
    revision = json.loads(receipt.read_text()).get("revision") if receipt.exists() else None
    if revision != contract["revision"]:
        raise PackFailure(f"checkpoint revision {revision} is not the contract's {contract['revision']}")
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    packs = RankPacks(args.output_dir, args.resume, ranks, PREFIX, source_identity(args.checkpoint, PREFIX, files=(args.contract,)))
    hidden, vocab_rows = config["hidden_size"], config["vocab_size"] // TP
    embed = checkpoint.bf16("model.embed_tokens.weight", (config["vocab_size"], hidden))
    head = checkpoint.bf16("lm_head.weight", (config["vocab_size"], hidden))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
        packs.add(rank, "head", head[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("head_norm", checkpoint.bf16_as_f32("model.norm.weight", (hidden,)), "f32")
    for layer in range(config["num_hidden_layers"]):
        pack_layer(checkpoint, packs, config, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, config, revision, contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"solar_open2_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
