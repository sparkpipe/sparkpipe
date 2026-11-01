#!/usr/bin/env python3
"""K-EXAONE 2.0 BF16 checkpoint -> TP named stage packs (sixteen by default) for the Gemma-core stage runner.

--tp-degree sets the rank count (16 for the Sparks; 1, 2, 4 or 8 for Metal
ranks); the splits below are of 1/degree each, shown for 16. Output per rank:
<out>/kexaone2.tp<degree>.rank<NN>.pack (tools/spark_named_pack.py
container), plus <pack>.sha256 and the empty-range <pack>.experts v2
manifest (the rank's experts are plain resident tensors).

Rank r holds, all bf16 unless noted:
  embed, head         vocab rows [r*9600, +9600) of embed_tokens and of the
                      untied lm_head
  head_norm           model.norm as f32
  per layer           input_layernorm as input_norm and
                      post_attention_layernorm as pre_ff_norm, f32 (pre-norm
                      blocks); q_norm / k_norm f32
  attention           q_proj|k_proj|v_proj fused (64 + 8 + 8 heads of 128,
                      10240 rows), rank rows [r*640, +640); o_proj columns
                      [r*512, +512), the rank's four query heads
  ffn_gate_up/down    layers 0 and 1: the dense MLP (18432, 1152 rows per
                      rank); layers 2..77: the shared expert (2048, 128 rows
                      per rank); gate|up fused [2*rows, H], down columns
                      [H, rows]
  MoE (layers 2..77)  router_proj = mlp.gate.weight [256, H] replicated,
                      expert_bias = mlp.e_score_correction_bias f32 [256]
                      replicated, and every expert's intermediate rows
                      [128r, +128) as moe_w1 (gate_proj) and moe_w3
                      (up_proj) [256, 128, H] and moe_w2 (down_proj)
                      [256, H, 128]

The MTP layer (mtp.*) is the drafter's and is not packed here. The packer
checks config.json against the frozen contract, every tensor's dtype and
shape, and that the checkpoint holds exactly the tensors it packs plus the
MTP layer.
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
HEAD_DIM = 128
MTP_TENSORS = {
    "mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight", "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight", "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight", "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight",
}


def layer_window(layer):
    return 0 if layer == 0 else 4096 if layer == 1 else 0 if layer % 4 == 1 else 128


def expected_tensors(config):
    names = {"lm_head.weight", PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"}
    for layer in range(config["num_hidden_layers"]):
        p = f"{PREFIX}layers.{layer}."
        names |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        names |= {p + f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")}
        if config["mlp_layer_types"][layer] == "dense":
            names |= {p + f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
        else:
            names |= {p + "mlp.gate.weight", p + "mlp.e_score_correction_bias"}
            names |= {p + f"mlp.shared_experts.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
            names |= {p + f"mlp.experts.{e}.{n}.weight" for e in range(config["num_experts"]) for n in ("gate_proj", "up_proj", "down_proj")}
    return names


def validate(config, contract):
    source = contract["source"]["config_json"]
    layers = config["num_hidden_layers"]
    kinds = ["full_attention" if layer_window(index) == 0 else "sliding_attention" for index in range(layers)]
    checks = [
        (config["model_type"] == "exaone_moe" and config["architectures"] == ["ExaoneMoeForCausalLM"], "model_type exaone_moe"),
        (config["dtype"] == "bfloat16", "bf16 checkpoint"),
        (config["hidden_size"] == source["hidden_size"] == 6144, "hidden 6144"),
        (layers == source["num_hidden_layers"] == 78, "78 layers"),
        (config["vocab_size"] == source["vocab_size"] == 153600 and config["vocab_size"] % TP == 0, "vocab 153600"),
        (config["head_dim"] == HEAD_DIM and config["num_attention_heads"] == 64 and config["num_key_value_heads"] == 8, "64 query and 8 KV heads of 128"),
        (config["layer_types"] == kinds, "global attention at layers 0 and 4k+1 from 5, sliding elsewhere"),
        (config["sliding_windows"] == [layer_window(index) for index in range(layers)], "windows 4096 at layer 1, 128 on the other sliding layers"),
        (config.get("sliding_window", 4096) is not None, "global layers carry no rope"),
        (config["rope_parameters"] == {"rope_theta": 1000000, "rope_type": "default"}, "default rope 1e6 on 128 dims"),
        (config["hidden_act"] == "silu" and not config.get("attention_bias", False), "silu, no attention bias"),
        (config["mlp_layer_types"] == ["dense", "dense"] + ["sparse"] * (layers - 2), "layers 0 and 1 dense, 2..77 MoE"),
        (config["swiglu_limits"] == [0.0] * 62 + [7.0] * 16, "SwiGLU limit 7 on layers 62..77"),
        (config["num_experts"] == 256 and config["num_experts_per_tok"] == 8 and config["num_experts"] % TP == 0, "256 experts, top 8"),
        (config["num_shared_experts"] == 1 and config["moe_intermediate_size"] == 2048, "one shared expert, intermediate 2048"),
        (config["intermediate_size"] == 18432 and config["intermediate_size"] % 64 == 0, "dense intermediate 18432"),
        (config["scoring_func"] == "sigmoid" and config["topk_method"] == "noaux_tc" and config["n_group"] == 1 and config["topk_group"] == 1,
         "sigmoid router with selection bias, one group"),
        (config["norm_topk_prob"] and config["routed_scaling_factor"] == 2.5, "normalized top-k times 2.5"),
        (config["rms_norm_eps"] == 1e-5, "rms epsilon 1e-5"),
        (not config["tie_word_embeddings"], "untied lm_head"),
    ]
    failed = [text for ok, text in checks if not ok]
    if failed:
        raise PackFailure("config.json differs from the packer's K-EXAONE 2.0 TP16 layout: " + "; ".join(failed))


def float32(checkpoint, name, shape):
    shard = checkpoint.weight_map.get(name)
    if shard is None:
        raise PackFailure(f"checkpoint has no tensor {name}")
    with open(checkpoint.directory / shard, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        meta = json.loads(handle.read(length))[name]
        if meta["dtype"] != "F32" or tuple(meta["shape"]) != tuple(shape):
            raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected F32 {list(shape)}")
        begin, end = meta["data_offsets"]
        handle.seek(8 + length + begin)
        return np.frombuffer(handle.read(end - begin), dtype=np.float32).reshape(shape)


def pack_layer(checkpoint, packs, config, layer):
    hidden = config["hidden_size"]
    p = f"{PREFIX}layers.{layer}."
    name = f"layers.{layer}."
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"] * HEAD_DIM
    packs.replicated(name + "input_norm", checkpoint.f32(p + "input_layernorm.weight", (hidden,)), "f32")
    packs.replicated(name + "pre_ff_norm", checkpoint.f32(p + "post_attention_layernorm.weight", (hidden,)), "f32")
    packs.replicated(name + "attn_q_norm", checkpoint.f32(p + "self_attn.q_norm.weight", (HEAD_DIM,)), "f32")
    packs.replicated(name + "attn_k_norm", checkpoint.f32(p + "self_attn.k_norm.weight", (HEAD_DIM,)), "f32")
    q = checkpoint.array(p + "self_attn.q_proj.weight", (heads * HEAD_DIM, hidden))
    k = checkpoint.array(p + "self_attn.k_proj.weight", (kv, hidden))
    v = checkpoint.array(p + "self_attn.v_proj.weight", (kv, hidden))
    o = checkpoint.array(p + "self_attn.o_proj.weight", (hidden, heads * HEAD_DIM))
    fused = np.concatenate([q, k, v])
    if fused.shape[0] != heads * HEAD_DIM + 2 * kv or fused.shape[0] % TP or (heads * HEAD_DIM) % TP:
        raise PackFailure(f"layer {layer}: attention shapes {fused.shape} {o.shape} do not split over {TP} ranks")
    rows, columns = fused.shape[0] // TP, o.shape[1] // TP
    if config["mlp_layer_types"][layer] == "dense":
        inter, mlp = config["intermediate_size"], p + "mlp."
    else:
        inter, mlp = config["moe_intermediate_size"] * config["num_shared_experts"], p + "mlp.shared_experts."
    gate = checkpoint.array(mlp + "gate_proj.weight", (inter, hidden))
    up = checkpoint.array(mlp + "up_proj.weight", (inter, hidden))
    down = checkpoint.array(mlp + "down_proj.weight", (hidden, inter))
    for rank in packs.ranks:
        packs.add(rank, name + "attn_qkv", fused[rank * rows:(rank + 1) * rows], "bf16")
        packs.add(rank, name + "attn_out", o[:, rank * columns:(rank + 1) * columns], "bf16")
        window = ffn_window(inter, rank)
        packs.add(rank, name + "ffn_gate_up", np.concatenate((gate[window], up[window])), "bf16")
        packs.add(rank, name + "ffn_down", down[:, window], "bf16")
    if config["mlp_layer_types"][layer] == "dense":
        return
    experts, expert_inter = config["num_experts"], config["moe_intermediate_size"]
    packs.replicated(name + "router_proj", checkpoint.array(p + "mlp.gate.weight", (experts, hidden)), "bf16")
    packs.replicated(name + "expert_bias", float32(checkpoint, p + "mlp.e_score_correction_bias", (experts,)), "f32")
    width = expert_inter // TP
    for tensor, source, shape in (("moe_w1", "gate_proj", (expert_inter, hidden)), ("moe_w3", "up_proj", (expert_inter, hidden)),
                                  ("moe_w2", "down_proj", (hidden, expert_inter))):
        matrices = [checkpoint.array(f"{p}mlp.experts.{expert}.{source}.weight", shape) for expert in range(experts)]
        for rank in packs.ranks:
            window = slice(rank * width, (rank + 1) * width)
            packs.add(rank, name + tensor, np.stack([matrix[:, window] if tensor == "moe_w2" else matrix[window] for matrix in matrices]), "bf16")


def seal(final, payload, fmt, settings, manifest):
    assemble(final, payload, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, settings, manifest)
    digest = hashlib.sha256()
    with final.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
    Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
    print(f"{final.name} {final.stat().st_size} {digest.hexdigest()}", flush=True)


def finish(packs, config, revision, contract_sha256, defer):
    vocab_rows = config["vocab_size"] // TP
    local = config["num_experts"]
    for rank in packs.ranks:
        packs.writers[rank].close()
        window = ffn_window(config["intermediate_size"], rank)
        settings = {"model": "k_exaone_2_0", "tp_degree": TP, "tp_rank": rank, "hidden": config["hidden_size"],
                    "layers": config["num_hidden_layers"], "vocab": config["vocab_size"], "vocab_first": rank * vocab_rows,
                    "vocab_rows": vocab_rows, "ffn_rows": window.stop - window.start, "sliding_window": 128, "wide_window_layer": 1,
                    "wide_window": 4096, "experts": config["num_experts"], "local_experts": local, "expert_first": 0, "expert_layout": "rows",
                    "top_k": config["num_experts_per_tok"]}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "attn_qkv_layout": "fused_windows", "revision": revision, "contract_sha256": contract_sha256}
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
    validate(config, contract)
    checkpoint = Checkpoint(args.checkpoint)
    names = set(checkpoint.weight_map)
    missing = expected_tensors(config) ^ (names - MTP_TENSORS)
    if missing or not MTP_TENSORS <= names:
        raise PackFailure(f"checkpoint tensors differ from the K-EXAONE 2.0 layout: {sorted(missing)[:8]} ({len(missing)} names)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs = RankPacks(args.output_dir, args.resume, ranks, "kexaone2", source_identity(args.checkpoint, "kexaone2", files=(args.contract,)))
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
    finish(packs, config, contract["revision"], hashlib.sha256(contract_bytes).hexdigest(), args.defer_assembly)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"kexaone2_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
