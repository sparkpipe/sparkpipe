#!/usr/bin/env python3
"""Motif 3 BF16 checkpoint -> TP named stage packs (sixteen by default) for the Motif family schedule.

--tp-degree sets the rank count (16 for the Sparks; 1, 2, 4 or 8 for Metal
ranks). Output per rank: <out>/motif3.tp<degree>.rank<NN>.pack
(tools/spark_named_pack.py container), plus <pack>.sha256 and the empty-range
<pack>.experts v2 manifest (the rank's experts are plain resident tensors).

Rank r of T holds, bf16 unless noted:
  embed, head         vocab rows [r*V/T, +V/T) of embed_tokens and lm_head
  head_norm           model.norm, f32
  per layer           input_layernorm as attn_norm and post_attention_layernorm
                      as ffn_norm, f32
  hyper-connections   per site (hc_attn, hc_ffn): fn = [proj_pre; proj_post;
                      proj_res] x rms_norm.weight per column, f32 [24, 16384];
                      scale = [alpha_pre, alpha_post, alpha_res], f32 [3];
                      base = [bias_pre; bias_post; bias_res], f32 [24]
  attention           head groups [16r/T, 16(r+1)/T) of five query heads (four
                      signal, one noise) and the same KV heads:
                      attn_q_a = wq_a and attn_q_norm (f32), attn_kv_a = wkv_a
                      and attn_kv_norm (f32) replicated; attn_q_b = wq_b rows of
                      the rank's query heads; attn_gate = wq_b_gate rows,
                      attn_lambda = lambda_proj rows and attn_out = wo columns
                      of the rank's signal heads; attn_kv_b = wkv_b rows of the
                      rank's KV heads
  dense MLP (0, 1)    on rank layer % T only: ffn_gate_up = gate_proj|up_proj,
                      ffn_down, ffn_poly = [w0, w1, w2, bias] f32
  MoE (2..52)         router_proj and expert_bias (f32) replicated; experts
                      [384r/T, +384/T) whole: moe_w13 = gate_up_proj
                      [E, 2560, 4096], moe_w2 = down_proj [E, 4096, 1280],
                      moe_poly [E, 4] f32; the shared expert on rank layer % T:
                      shared_gate_up, shared_down, shared_poly f32 [4]

The MTP layer (model.mtp_layers.0) is the drafter's and is not packed here.
The packer checks config.json against the contract, every tensor's dtype and
shape, and that the checkpoint holds exactly the tensors it packs plus the MTP
layer.
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
from gemma4_tp16_pack import EXPERTS_MAGIC, EXPERTS_VERSION, Checkpoint, PackFailure, RankPacks  # noqa: E402
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, assemble, source_identity  # noqa: E402

PREFIX = "model."
TP = 16


def geometry(contract):
    model, attention, moe = contract["model"], contract["attention"], contract["moe"]
    return {
        "hidden": model["hidden_dimension"],
        "layers": model["layer_count"],
        "vocab": model["vocabulary_size"],
        "heads": attention["query_head_count"],
        "groups": attention["noise_head_count"],
        "group": attention["group_size"],
        "signal": attention["group_size"] - 1,
        "kv_heads": attention["key_value_head_count"],
        "head_dim": attention["head_dimension"],
        "value_dim": attention["value_head_dimension"],
        "nope": attention["qk_nope_head_dimension"],
        "q_lora": attention["query_lora_rank"],
        "latent": attention["kv_lora_rank"],
        "rope": attention["qk_rope_head_dimension"],
        "experts": moe["routed_expert_count"],
        "expert_inter": moe["expert_intermediate_dimension"],
        "dense_inter": moe["dense_intermediate_dimension"],
        "first_routed": moe["first_dense_layer_count"],
        "streams": contract["hyper_connections"]["expansion_rate"],
    }


def expected_tensors(g):
    names = {"lm_head.weight", PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"}
    for layer in range(g["layers"]):
        p = f"{PREFIX}layers.{layer}."
        names |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        for site in ("mhc_attn", "mhc_ffn"):
            names |= {p + f"{site}.{n}" for n in ("alpha_pre", "alpha_post", "alpha_res", "bias_pre", "bias_post", "bias_res",
                                                 "proj_pre.weight", "proj_post.weight", "proj_res.weight", "rms_norm.weight")}
        names |= {p + f"self_attn.{n}.weight" for n in ("wq_a", "q_norm", "wq_b", "wq_b_gate", "wkv_a", "kv_norm", "wkv_b", "lambda_proj", "wo")}
        if layer < g["first_routed"]:
            names |= {p + f"mlp.{n}" for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight", "act_fn.weight", "act_fn.bias")}
        else:
            names |= {p + f"moe.{n}" for n in ("expert_bias", "router.gate.weight", "experts.gate_up_proj", "experts.down_proj",
                                               "experts.act_fn.weight", "experts.act_fn.bias")}
            names |= {p + f"moe.shared_experts.{n}" for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight", "act_fn.weight",
                                                              "act_fn.bias")}
    return names


def validate(config, contract, g):
    checks = [
        (config["architectures"] == [contract["architecture"]] and config["model_type"] == "Motif", "MotifForCausalLM"),
        (config["dtype"] == "bfloat16", "bf16 checkpoint"),
        (config["attention_cls"] == "gdla" and config["diff_v2"] is True, "GDLA diff_v2 attention"),
        (config["hidden_act"] == "poly_norm" and config["mhc_enabled"] is True, "PolyNorm MLPs between mHC sites"),
        (g["heads"] == g["groups"] * g["group"] and g["groups"] % TP == 0 and g["kv_heads"] == g["groups"], "whole head groups per rank"),
        (g["experts"] % TP == 0 and g["vocab"] % TP == 0, "experts and vocabulary divide into the ranks"),
        (config["num_shared_experts"] == 1 and config["interleave_moe_layer_step"] == 1, "one shared expert, every layer routed"),
    ]
    failed = [text for ok, text in checks if not ok]
    if failed:
        raise PackFailure("config.json differs from the packer's Motif 3 layout: " + "; ".join(failed))


def mtp_tensors(checkpoint):
    return {name for name in checkpoint.weight_map if name.startswith(PREFIX + "mtp_layers.")}


def hyper_site(checkpoint, packs, g, p, name, site):
    flat = g["streams"] * g["hidden"]
    s = g["streams"]
    norm = checkpoint.f32(p + f"{site}.rms_norm.weight", (flat,))
    proj = np.concatenate([checkpoint.f32(p + f"{site}.proj_pre.weight", (s, flat)), checkpoint.f32(p + f"{site}.proj_post.weight", (s, flat)),
                           checkpoint.f32(p + f"{site}.proj_res.weight", (s * s, flat))])
    packs.replicated(name + "fn", (proj * norm[None, :]).astype(np.float32), "f32")
    scale = [checkpoint.f32(p + f"{site}.alpha_{part}", (1,))[0] for part in ("pre", "post", "res")]
    packs.replicated(name + "scale", np.asarray(scale, dtype=np.float32), "f32")
    base = np.concatenate([checkpoint.f32(p + f"{site}.bias_pre", (s,)), checkpoint.f32(p + f"{site}.bias_post", (s,)),
                           checkpoint.f32(p + f"{site}.bias_res", (s, s)).reshape(-1)])
    packs.replicated(name + "base", base.astype(np.float32), "f32")


def poly(checkpoint, p, shape):
    weight = checkpoint.f32(p + "act_fn.weight", shape + (3,))
    bias = checkpoint.f32(p + "act_fn.bias", shape + (1,))
    return np.concatenate([weight, bias], axis=-1).astype(np.float32)


def pack_layer(checkpoint, packs, g, layer):
    hidden, p, name = g["hidden"], f"{PREFIX}layers.{layer}.", f"layers.{layer}."
    packs.replicated(name + "attn_norm", checkpoint.f32(p + "input_layernorm.weight", (hidden,)), "f32")
    packs.replicated(name + "ffn_norm", checkpoint.f32(p + "post_attention_layernorm.weight", (hidden,)), "f32")
    hyper_site(checkpoint, packs, g, p, name + "hc_attn_", "mhc_attn")
    hyper_site(checkpoint, packs, g, p, name + "hc_ffn_", "mhc_ffn")
    a = p + "self_attn."
    packs.replicated(name + "attn_q_a", checkpoint.array(a + "wq_a.weight", (g["q_lora"], hidden)), "bf16")
    packs.replicated(name + "attn_q_norm", checkpoint.f32(a + "q_norm.weight", (g["q_lora"],)), "f32")
    packs.replicated(name + "attn_kv_a", checkpoint.array(a + "wkv_a.weight", (g["latent"] + g["rope"], hidden)), "bf16")
    packs.replicated(name + "attn_kv_norm", checkpoint.f32(a + "kv_norm.weight", (g["latent"],)), "f32")
    signal_total = g["groups"] * g["signal"]
    q_b = checkpoint.array(a + "wq_b.weight", (g["heads"] * g["head_dim"], g["q_lora"]))
    gate = checkpoint.array(a + "wq_b_gate.weight", (signal_total * g["value_dim"], g["q_lora"]))
    lam = checkpoint.array(a + "lambda_proj.weight", (signal_total, hidden))
    out = checkpoint.array(a + "wo.weight", (hidden, signal_total * g["value_dim"]))
    kv_b = checkpoint.array(a + "wkv_b.weight", (g["kv_heads"] * (g["nope"] + g["value_dim"]), g["latent"]))
    groups = g["groups"] // TP
    for rank in packs.ranks:
        heads = slice(rank * groups * g["group"], (rank + 1) * groups * g["group"])
        signal = slice(rank * groups * g["signal"], (rank + 1) * groups * g["signal"])
        kv = slice(rank * groups, (rank + 1) * groups)
        packs.add(rank, name + "attn_q_b", q_b.reshape(g["heads"], g["head_dim"], -1)[heads].reshape(-1, g["q_lora"]), "bf16")
        packs.add(rank, name + "attn_gate", gate.reshape(signal_total, g["value_dim"], -1)[signal].reshape(-1, g["q_lora"]), "bf16")
        packs.add(rank, name + "attn_lambda", lam[signal], "bf16")
        packs.add(rank, name + "attn_out", out.reshape(hidden, signal_total, g["value_dim"])[:, signal].reshape(hidden, -1), "bf16")
        packs.add(rank, name + "attn_kv_b", kv_b.reshape(g["kv_heads"], g["nope"] + g["value_dim"], -1)[kv].reshape(-1, g["latent"]), "bf16")
    owner = layer % TP
    if layer < g["first_routed"]:
        m, inter = p + "mlp.", g["dense_inter"]
        packs.add(owner, name + "ffn_gate_up", np.concatenate([checkpoint.array(m + "gate_proj.weight", (inter, hidden)),
                                                               checkpoint.array(m + "up_proj.weight", (inter, hidden))]), "bf16")
        packs.add(owner, name + "ffn_down", checkpoint.array(m + "down_proj.weight", (hidden, inter)), "bf16")
        packs.add(owner, name + "ffn_poly", poly(checkpoint, m, ()), "f32")
        return
    e = p + "moe."
    experts, inter = g["experts"], g["expert_inter"]
    packs.replicated(name + "router_proj", checkpoint.array(e + "router.gate.weight", (experts, hidden)), "bf16")
    packs.replicated(name + "expert_bias", checkpoint.f32(e + "expert_bias", (experts,)), "f32")
    w13 = checkpoint.array(e + "experts.gate_up_proj", (experts, 2 * inter, hidden))
    w2 = checkpoint.array(e + "experts.down_proj", (experts, hidden, inter))
    polys = poly(checkpoint, e + "experts.", (experts,))
    local = experts // TP
    for rank in packs.ranks:
        window = slice(rank * local, (rank + 1) * local)
        packs.add(rank, name + "moe_w13", w13[window], "bf16")
        packs.add(rank, name + "moe_w2", w2[window], "bf16")
        packs.add(rank, name + "moe_poly", polys[window], "f32")
    s = e + "shared_experts."
    packs.add(owner, name + "shared_gate_up", np.concatenate([checkpoint.array(s + "gate_proj.weight", (inter, hidden)),
                                                              checkpoint.array(s + "up_proj.weight", (inter, hidden))]), "bf16")
    packs.add(owner, name + "shared_down", checkpoint.array(s + "down_proj.weight", (hidden, inter)), "bf16")
    packs.add(owner, name + "shared_poly", poly(checkpoint, s, ()), "f32")


def seal(final, payload, fmt, settings, manifest):
    assemble(final, payload, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, settings, manifest)
    digest = hashlib.sha256()
    with final.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
    Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
    print(f"{final.name} {final.stat().st_size} {digest.hexdigest()}", flush=True)


def finish(packs, g, revision, contract_sha256, defer):
    vocab_rows, local = g["vocab"] // TP, g["experts"] // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        settings = {"model": "motif_3", "tp_degree": TP, "tp_rank": rank, "hidden": g["hidden"], "layers": g["layers"], "vocab": g["vocab"],
                    "vocab_first": rank * vocab_rows, "vocab_rows": vocab_rows, "head_groups": g["groups"] // TP,
                    "head_group_first": rank * g["groups"] // TP, "experts": g["experts"], "local_experts": local, "expert_first": rank * local,
                    "expert_layout": "whole", "single_owner": "layer % tp_degree"}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN, "linear_codec": "bf16",
               "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        if defer:
            Path(str(final) + ".assembly.json").write_text(json.dumps({"final": final.name, "payload": packs.payload(rank).name, "format": fmt,
                                                                       "settings": settings, "manifest": packs.writers[rank].manifest}))
            print(f"rank {rank:02d} payload written, assembly deferred to {final.name}.assembly.json", flush=True)
        else:
            seal(final, packs.payload(rank), fmt, settings, packs.writers[rank].manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--assemble", type=Path)
    parser.add_argument("--defer-assembly", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tp-degree", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--ranks")
    args = parser.parse_args()
    global TP
    TP = gemma4_tp16_pack.TP = args.tp_degree
    args.ranks = args.ranks or ",".join(str(rank) for rank in range(TP))
    if args.assemble is not None:
        plan = json.loads(args.assemble.read_text())
        directory = args.assemble.parent
        seal(directory / plan["final"], directory / plan["payload"], plan["format"], plan["settings"], plan["manifest"])
        args.assemble.unlink()
        return 0
    if args.checkpoint is None or args.output_dir is None or args.contract is None:
        raise PackFailure("--checkpoint, --output-dir and --contract are required to pack")
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    config_bytes = (args.checkpoint / "config.json").read_bytes()
    contract_bytes = args.contract.read_bytes()
    contract = json.loads(contract_bytes)
    if hashlib.sha256(config_bytes).hexdigest() != contract["source"]["config_json"]["sha256"]:
        raise PackFailure("config.json sha256 differs from the contract's frozen config")
    config = json.loads(config_bytes)
    g = geometry(contract)
    validate(config, contract, g)
    checkpoint = Checkpoint(args.checkpoint)
    names = set(checkpoint.weight_map)
    mtp = mtp_tensors(checkpoint)
    missing = expected_tensors(g) ^ (names - mtp)
    if missing or not mtp:
        raise PackFailure(f"checkpoint tensors differ from the Motif 3 layout: {sorted(missing)[:8]} ({len(missing)} names)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs = RankPacks(args.output_dir, args.resume, ranks, "motif3", source_identity(args.checkpoint, "motif3", files=(args.contract,)))
    vocab_rows = g["vocab"] // TP
    embed = checkpoint.array(PREFIX + "embed_tokens.weight", (g["vocab"], g["hidden"]))
    head = checkpoint.array("lm_head.weight", (g["vocab"], g["hidden"]))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
        packs.add(rank, "head", head[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("head_norm", checkpoint.f32(PREFIX + "norm.weight", (g["hidden"],)), "f32")
    for layer in range(g["layers"]):
        pack_layer(checkpoint, packs, g, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, g, contract["source_revision"], hashlib.sha256(contract_bytes).hexdigest(), args.defer_assembly)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"motif3_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
