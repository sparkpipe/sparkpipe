#!/usr/bin/env python3
"""Generate TP and PP serving recipes from authoritative model contracts.

One recipe per (model, strategy, degree), named by the grammar in
docs/DATAFILE_NAMING.md: <tag>.<strategy><n>.<content-hash>.json, with KV
cache entries under <tag>.<strategy><n>.<geometry-hash>/. The content-hash
covers every recipe input; the geometry-hash covers exactly the contract
fields that change KV CONTENT, so a TP16 recipe and a PP16 recipe of the
same model share a geometry-hash and an NVMe-resident KV entry survives the
strategy switch, while any contract edit that moves KV bytes (layer counts,
head dims, latent widths, kv dtype, rope conventions) mints a new hash and
orphans the old entries instead of misreading them.

Reuse, per the tree's DRY law, instead of parallel machinery:

- this generator owns the deterministic PP placement algorithm used by recipe
  artifacts: contiguous cover, the dense prefix whole in stage zero, bounded
  routed layers per stage, final-token work on the last stage, and a minimax
  balancing DP with a first-best tie-break. An optional measured stage profile
  supplies relative stage capacity so faster ranks can receive more
  contiguous layers; it changes placement artifacts only and never reaches a
  kernel or the model-resident runtime. Its analytic per-layer cost is an
  ESTIMATE for placement only. It is marked NOT_MEASURED in every recipe
  without a profile.
- the k3 TP shard table is built from tools/k3_shard.py's own classification
  sets. tests/test_recipe_generation.py walks those sets against every recipe,
  so the planning table cannot drift from the offline pack slicer.
- the glm52 TP shard table is derived here from the model contract. Routed
  expert tensors remain package-owned, so the planning recipe records them as
  PACKAGE_OWNED instead of inventing a split outside the stage pack.

Shard classes use the recipe vocabulary: REPLICATED, OUTPUT_DIM_HEADS, OUTPUT_DIM,
INPUT_DIM_HEADS, INPUT_DIM, CONCAT_OUTPUT. A class whose split extent does
not divide the degree on whole rows / head blocks / quantization groups is
not failed here (the engine owns refusal) - the recipe marks it replicated
for that degree and records why, which is what makes qwen38_27b.TP16 honest:
24 query heads do not split 16 ways, so attention replicates and the MLP
carries the sharding.

Determinism is the testable contract: same inputs, same bytes, and the
--check flag (generate_k3_contract.py's pattern) fails CI when the
committed examples/recipes/ set is stale for the committed contracts.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_shard  # noqa: E402  the k3 TP classification sets are its table

CONTRACTS = ROOT / "model_contracts"
DEFAULT_OUT_DIR = ROOT / "examples" / "recipes"
DEFAULT_TOPOLOGY = (ROOT / "examples" / "topologies" /
                    "dual_switch_16node_production.json")

HASH_CHARS = 16
# The routed-layer cap is a recipe-placement bound, not a runtime default. The
# model-resident ABI supports at most 16 stages.
MAX_ROUTED_PER_STAGE = 8
ENGINE_MAX_STAGE_COUNT = 16
DEFAULT_DEGREES = (16, 13)
DATAFILE_RE = re.compile(r"^[a-z0-9]+\.(?:TP|PP)\d+\.[0-9a-f]{16}\.json$")


class RecipeFailure(RuntimeError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def short_hash(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:HASH_CHARS]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_hash(family, kv_geometry):
    return short_hash({"family": family, "kv_geometry": kv_geometry})


# ---------------------------------------------------------------------------
# Family adapters: contract JSON -> placement geometry. Each returns
#   layer_count, first_routed_layer, layer_costs (int per layer, active-
#   parameter proxy), final_stage_extra_cost (the vocab projection the last
#   PP stage pays alone), kv_geometry (the hashed KV-content fields) and
#   shard_classes (the TP table, split_extent = extent of the SPLIT
#   dimension of one tensor instance).
# Cost formulas read contract dims only; where a width the formula wants is
# not in the contract (k3's shared-expert intermediate) the nearest
# contract width stands in and the approximation is noted here, not hidden.
# ---------------------------------------------------------------------------

def adapt_k3(c):
    m, hyb, kda, mla = c["model"], c["hybrid_attention"], c["kda"], c["mla"]
    moe, cache = c["moe"], c["cache"]
    hidden, layers, vocab = (m["hidden_dimension"], m["layer_count"],
                             m["vocabulary_size"])
    first_routed = m["first_routed_layer"]
    heads = mla["query_head_count"]
    qk_head = mla["qk_nope_dimension"] + mla["qk_unrotated_dimension"]
    v_head = mla["value_head_dimension"]
    latent, inter = moe["latent_dimension"], moe["expert_intermediate_dimension"]
    mla_attn = (hidden * mla["query_lora_rank"] +
                mla["query_lora_rank"] * heads * qk_head +
                hidden * (mla["kv_lora_rank"] + mla["qk_unrotated_dimension"]) +
                mla["kv_lora_rank"] * heads * (mla["qk_nope_dimension"] + v_head) +
                2 * heads * v_head * hidden)  # out projection + output gate
    kda_attn = (3 * hidden * kda["head_count"] * kda["key_dimension"] +
                kda["head_count"] * kda["value_dimension"] * hidden +
                4 * kda["key_dimension"] * hidden)  # decay/gate down+up pairs
    dense_mlp = 3 * hidden * moe["dense_intermediate_dimension"]
    # shared experts stand at the expert intermediate - the contract does not
    # pin a shared width and k3_shard reads it back from pack bytes.
    routed_mlp = (moe["experts_per_token"] * 3 * latent * inter +
                  moe["shared_expert_count"] * 3 * hidden * inter)

    def is_mla(i):
        return i % hyb["period"] == hyb["global_phase"]

    layer_costs = [
        (mla_attn if is_mla(i) else kda_attn) +
        (routed_mlp if i >= first_routed else dense_mlp)
        for i in range(layers)
    ]
    kv_geometry = {
        "layout": "mla_latent_bf16+kda_recurrent_state_fp32",
        "layer_count": layers,
        "mla_layer_count": hyb["mla_layer_count"],
        "kda_layer_count": hyb["kda_layer_count"],
        "attention_period": hyb["period"],
        "global_phase": hyb["global_phase"],
        "kv_lora_rank": mla["kv_lora_rank"],
        "qk_nope_dimension": mla["qk_nope_dimension"],
        "qk_unrotated_dimension": mla["qk_unrotated_dimension"],
        "value_head_dimension": v_head,
        "uses_nope": mla["uses_nope"],
        "kv_element_bits": cache["kv_element_bits"],
        "kv_page_slots": cache["kv_page_slots"],
        "kda_state": {
            "head_count": kda["head_count"],
            "key_dimension": kda["key_dimension"],
            "value_dimension": kda["value_dimension"],
            "short_conv_kernel": kda["short_conv_kernel"],
            "state_element_type": kda["state_element_type"],
            "state_layout": kda["state_layout"],
        },
    }
    kda_rows = kda["head_count"] * kda["key_dimension"]
    mla_q_rows = heads * qk_head
    mla_v_rows = heads * v_head
    # the V2 fused q|k|v|beta tensor: per-head widths 128/128/128/1, one
    # OUTPUT_DIM_HEADS class whose split extent is the summed section rows
    qkvb_rows = (2 * kda["head_count"] * kda["key_dimension"] +
                 kda["head_count"] * kda["value_dimension"] +
                 kda["head_count"])

    def cls(name, match, shard_class, split, scope, extent, note=None,
            group=None, instances=None, head_count=None):
        return {"name": name, "match": match, "shard_class": shard_class,
                "split": split, "scope": scope, "split_extent": extent,
                "quant_group": group, "note": note,
                "instances_per_model": instances, "head_count": head_count}

    shard_classes = [
        cls("replicated", sorted(k3_shard.REPLICATED | k3_shard.MODEL_REPLICATED),
            "REPLICATED", "none", "mixed", None,
            "norms, router, and the 128-wide bottlenecks every rank reads "
            "whole - replicating them is what keeps the latent KV identical "
            "per rank (tools/k3_shard.py)"),
        cls("embed_and_head", ["model.embed_tokens.weight", "lm_head.weight"],
            "OUTPUT_DIM", "output", "per_model", vocab),
    ]
    for field, (kind, _) in sorted(k3_shard.OUTPUT_HEADS.items()):
        extent = {"kda": kda_rows, "kda_qkvb": qkvb_rows,
                  "mla_q": mla_q_rows, "mla_v": mla_v_rows}[kind]
        shard_classes.append(cls(
            f"heads_out:{field}", [field], "OUTPUT_DIM_HEADS", "output_heads",
            "per_layer", extent, head_count=96))
    for field, (kind, _) in sorted(k3_shard.INPUT_HEADS.items()):
        extent = kda_rows if kind == "kda" else mla_v_rows
        shard_classes.append(cls(
            f"heads_in:{field}", [field], "INPUT_DIM_HEADS", "input_heads",
            "per_layer", extent, head_count=96))
    shard_classes += [
        cls("routed_down", sorted(k3_shard.OUTPUT_DIM), "OUTPUT_DIM",
            "output", "per_layer", latent),
        cls("routed_up", sorted(k3_shard.INPUT_DIM), "INPUT_DIM", "input",
            "per_layer", latent),
        cls("shared_w1", ["shared_w1_weight"], "CONCAT_OUTPUT",
            "concat_output", "per_layer", None,
            "[gate; up] halves split and re-concatenate per rank; the "
            "shared intermediate is pack-defined, not contract-pinned"),
        cls("dense_gate_up", ["dense_gate_up_weight"], "CONCAT_OUTPUT",
            "concat_output", "per_layer",
            2 * moe["dense_intermediate_dimension"]),
        cls("shared_w2", ["shared_w2_weight"], "INPUT_DIM", "input",
            "per_layer", None, "input extent is the pack-defined shared "
            "intermediate"),
        cls("dense_down", ["dense_down_weight"], "INPUT_DIM", "input",
            "per_layer", moe["dense_intermediate_dimension"]),
        cls("expert_w1", sorted(k3_shard.EXPERT_CONCAT),
            "CONCAT_OUTPUT", "concat_output", "per_expert_per_layer",
            2 * inter, "V2: one interleaved weight+scale stream (no scale "
            "plane); each expert's [gate; up] halves split on whole "
            "16-neuron cells, the co-tiled scale rows riding with their "
            "payload",
            instances=moe["routed_expert_count"] * (layers - first_routed)),
        cls("expert_w2", sorted(k3_shard.EXPERT_INPUT),
            "INPUT_DIM", "input", "per_expert_per_layer", inter,
            "V2: per-rank K must be whole 128-element interleave k-tiles "
            "(the grid coarsened V1's 32-element groups), so K3's 24 tiles "
            "admit TP 1/2/4/8 and k3_shard refuses TP16", group=128),
    ]
    return {"layer_count": layers, "first_routed_layer": first_routed,
            "layer_costs": layer_costs,
            "final_stage_extra_cost": hidden * vocab,
            "kv_geometry": kv_geometry, "shard_classes": shard_classes}


def adapt_glm52(c):
    hidden, layers, vocab = (c["hidden_dimension"], c["layer_count"],
                             c["output_vocab_count"])
    first_routed = c["first_routed_layer"]
    heads, latent = c["head_count"], c["latent_dimension"]
    nope, rope, v_head = (c["qk_nope_head_dimension"], c["rope_dimension"],
                          c["value_head_dimension"])
    inter, dense_inter = (c["moe_intermediate_dimension"],
                          c["dense_intermediate_dimension"])
    attn = (hidden * c["query_a_dimension"] +
            c["query_a_dimension"] * heads * (nope + rope) +
            hidden * (latent + rope) +
            latent * heads * (nope + v_head) +
            heads * v_head * hidden)
    routed_mlp = ((c["moe_top_k"] + 1) * 3 * hidden * inter)  # +1 shared
    dense_mlp = 3 * hidden * dense_inter
    layer_costs = [attn + (dense_mlp if i < first_routed else routed_mlp)
                   for i in range(layers)]
    kv_geometry = {
        "layout": "mla_compressed_bf16",
        "layer_count": layers,
        "latent_dimension": latent,
        "qk_nope_head_dimension": nope,
        "rope_dimension": rope,
        "value_head_dimension": v_head,
        "head_count": heads,
        "rope_theta": c["rope_theta"],
        "rope_interleave": c["rope_interleave"],
        "kv_element_bits": 16,
        "dsa_index_head_count": c["dsa_index_head_count"],
        "dsa_index_head_dimension": c["dsa_index_head_dimension"],
        "dsa_rope_interleave": c["dsa_rope_interleave"],
    }

    def cls(name, match, shard_class, split, scope, extent, note=None,
            instances=None, head_count=None):
        return {"name": name, "match": match, "shard_class": shard_class,
                "split": split, "scope": scope, "split_extent": extent,
                "quant_group": None, "note": note,
                "instances_per_model": instances, "head_count": head_count}

    shard_classes = [
        # The first three entries are the architecture's head-parallel forms.
        cls("heads_out:q_b", ["self_attn.q_b_proj.weight"],
            "OUTPUT_DIM_HEADS", "output_heads", "per_layer",
            heads * (nope + rope), head_count=heads),
        cls("heads_out:kv_b", ["self_attn.kv_b_proj.weight"],
            "OUTPUT_DIM_HEADS", "output_heads", "per_layer",
            heads * (nope + v_head), head_count=heads),
        cls("heads_in:o", ["self_attn.o_proj.weight"],
            "INPUT_DIM_HEADS", "input_heads", "per_layer", heads * v_head,
            head_count=heads),
        cls("dense_gate_up", ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "OUTPUT_DIM", "output", "per_dense_layer", dense_inter),
        cls("shared_gate_up", ["mlp.shared_experts.gate_proj.weight",
                               "mlp.shared_experts.up_proj.weight"],
            "OUTPUT_DIM", "output", "per_routed_layer", inter),
        cls("dense_down", ["mlp.down_proj.weight"], "INPUT_DIM", "input",
            "per_dense_layer", dense_inter),
        cls("shared_down", ["mlp.shared_experts.down_proj.weight"],
            "INPUT_DIM", "input", "per_routed_layer", inter),
        cls("replicated", [
            "self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight",
            "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.weights_proj.weight", "self_attn.indexer.wk.weight",
            "self_attn.indexer.wq_b.weight", "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias", "input_layernorm.weight",
            "post_attention_layernorm.weight", "model.norm.weight",
            "model.embed_tokens.weight", "enorm.weight", "hnorm.weight",
            "eh_proj.weight",
        ], "REPLICATED", "none", "mixed", None,
            "the MLA latent paths, router, norms, embedding and MTP "
            "projections - the engine table's own replicated set"),
        cls("routed_experts", ["mlp.experts."], "ENGINE_UNCLASSIFIED",
            "none", "per_expert_per_layer", None,
            "the engine TP classifier returns UNKNOWN for routed expert "
            "tensors today (fail-closed); expert placement is "
            "stagepack/EP-owned, so this recipe shreds nothing it cannot "
            "name",
            instances=c["moe_expert_count"] * (layers - first_routed)),
    ]
    return {"layer_count": layers, "first_routed_layer": first_routed,
            "layer_costs": layer_costs,
            "final_stage_extra_cost": hidden * vocab,
            "kv_geometry": kv_geometry, "shard_classes": shard_classes}


def adapt_dsv4(c, family):
    m, att, moe = c["model"], c["attention"], c["moe"]
    hidden, layers, vocab = (m["hidden_dimension"], m["layer_count"],
                             m["vocabulary_size"])
    heads, head_dim = m["attention_head_count"], m["head_dimension"]
    rope = m["qk_rope_head_dimension"]
    inter = moe["expert_intermediate_dimension"]
    attn = (hidden * m["query_lora_rank"] +
            m["query_lora_rank"] * heads * (head_dim + rope) +
            hidden * (head_dim + rope) * m["kv_head_count"] +
            heads * head_dim * hidden)
    # every layer is MoE; the hash-routed prefix only changes WHO routes
    mlp = ((moe["experts_per_token"] + moe["shared_expert_count"]) *
           3 * hidden * inter)
    layer_costs = [attn + mlp] * layers
    kv_geometry = {
        "layout": "compressed_sparse_history",
        "layer_count": layers,
        "head_dimension": head_dim,
        "qk_rope_head_dimension": rope,
        "kv_head_count": m["kv_head_count"],
        "rope_theta": att["rope_theta"],
        "compressed_rope_theta": att["compressed_rope_theta"],
        "yarn_factor": att["yarn_factor"],
        "yarn_original_context_tokens": att["yarn_original_context_tokens"],
        "sliding_window_tokens": att["sliding_window_tokens"],
        "compression_ratios": att["compression_ratios"],
        "index_head_count": att["index_head_count"],
        "index_head_dimension": att["index_head_dimension"],
        "index_top_k": att["index_top_k"],
        # the cache plan's element bits are a runtime parameter
        # (spark_dsv4_cache_plan.h), not contract-pinned: content width is,
        # and it is all here
    }

    def cls(name, match, shard_class, split, scope, extent, note=None,
            group=None, instances=None, head_count=None):
        return {"name": name, "match": match, "shard_class": shard_class,
                "split": split, "scope": scope, "split_extent": extent,
                "quant_group": group, "note": note,
                "instances_per_model": instances, "head_count": head_count}

    shard_classes = [
        cls("heads_out:q_up", ["self_attn.q_up.weight"], "OUTPUT_DIM_HEADS",
            "output_heads", "per_layer", heads * (head_dim + rope),
            head_count=heads),
        cls("heads_in:o", ["self_attn.o_proj.weight"], "INPUT_DIM_HEADS",
            "input_heads", "per_layer", heads * head_dim,
            head_count=heads),
        cls("compressed_kv", ["self_attn.kv_a.weight", "self_attn.kv_b.weight"],
            "REPLICATED", "none", "per_layer", None,
            "one compressed KV head: replicating keeps the compressed "
            "history identical per rank, the k3/glm52 latent argument"),
        cls("expert_gate_up", ["mlp.experts.gate_proj.weight",
                               "mlp.experts.up_proj.weight"],
            "OUTPUT_DIM", "output", "per_expert_per_layer", inter,
            note="checkpoint_fp4 group is checkpoint-defined, not "
                 "contract-pinned; the slicer must refuse a per-rank K "
                 "that breaks the group (k3_shard's rule) once pinned",
            instances=moe["routed_expert_count"] * layers),
        cls("expert_down", ["mlp.experts.down_proj.weight"], "INPUT_DIM",
            "input", "per_expert_per_layer", inter,
            note="same checkpoint_fp4 group caveat as expert_gate_up",
            instances=moe["routed_expert_count"] * layers),
        cls("shared_gate_up", ["mlp.shared_experts.gate_proj.weight",
                               "mlp.shared_experts.up_proj.weight"],
            "OUTPUT_DIM", "output", "per_layer", inter),
        cls("shared_down", ["mlp.shared_experts.down_proj.weight"],
            "INPUT_DIM", "input", "per_layer", inter),
        cls("embed_and_head", ["model.embed_tokens.weight", "lm_head.weight"],
            "OUTPUT_DIM", "output", "per_model", vocab),
        cls("replicated", ["input_layernorm", "post_attention_layernorm",
                           "model.norm.weight", "mlp.gate", "self_attn.indexer",
                           "hyper_connections", "mtp."],
            "REPLICATED", "none", "mixed", None,
            "norms, routers (learned and hash), the sparse indexer, "
            "hyper-connection tables and the MTP layer"),
    ]
    return {"layer_count": layers, "first_routed_layer": 0,
            "layer_costs": layer_costs,
            "final_stage_extra_cost": hidden * vocab,
            "kv_geometry": kv_geometry, "shard_classes": shard_classes}


def adapt_qwen38_27b(c):
    m, hyb, gdn, att = (c["model"], c["hybrid_attention"], c["gdn"],
                        c["attention"])
    hidden, layers, vocab = (m["hidden_dimension"], m["layer_count"],
                             m["vocabulary_size"])
    q_heads, kv_heads = att["query_head_count"], att["kv_head_count"]
    head_dim = att["head_dimension"]
    # output_gate doubles the query projection's rows
    full_attn = (hidden * q_heads * head_dim * 2 +
                 hidden * kv_heads * head_dim * 2 +
                 q_heads * head_dim * hidden)
    gdn_attn = (hidden * gdn["key_head_count"] * gdn["key_dimension"] * 2 +
                hidden * gdn["value_head_count"] * gdn["value_dimension"] +
                gdn["value_head_count"] * gdn["value_dimension"] * hidden)
    dense_mlp = 3 * hidden * m["dense_intermediate_dimension"]

    def is_full(i):
        return i % hyb["period"] == hyb["full_phase"]

    layer_costs = [(full_attn if is_full(i) else gdn_attn) + dense_mlp
                   for i in range(layers)]
    kv_geometry = {
        "layout": "gqa_full_kv_bf16+gdn_recurrent_state_fp32",
        "layer_count": layers,
        "full_layer_count": hyb["full_layer_count"],
        "gdn_layer_count": hyb["gdn_layer_count"],
        "attention_period": hyb["period"],
        "full_phase": hyb["full_phase"],
        "kv_head_count": kv_heads,
        "head_dimension": head_dim,
        "rope_dimension": att["rope_dimension"],
        "rope_theta": att["rope_theta"],
        "rope_convention": att["rope_convention"],
        "kv_element_bits": c["cache"]["kv_element_bits"],
        "kv_page_slots": c["cache"]["kv_page_slots"],
        "gdn_state": {
            "key_head_count": gdn["key_head_count"],
            "value_head_count": gdn["value_head_count"],
            "key_dimension": gdn["key_dimension"],
            "value_dimension": gdn["value_dimension"],
            "short_conv_kernel": gdn["short_conv_kernel"],
            "state_element_type": gdn["state_element_type"],
        },
    }

    def cls(name, match, shard_class, split, scope, extent, note=None,
            instances=None, head_count=None):
        return {"name": name, "match": match, "shard_class": shard_class,
                "split": split, "scope": scope, "split_extent": extent,
                "quant_group": None, "note": note,
                "instances_per_model": instances, "head_count": head_count}

    shard_classes = [
        cls("heads_out:full_q", ["full_attn.q_proj.weight"],
            "OUTPUT_DIM_HEADS", "output_heads", "per_full_layer",
            q_heads * head_dim * 2,  # gated query projection
            "24 query heads split at degrees 1/2/4/8 only; at 16/13 the "
            "recipe replicates attention and the MLP carries the sharding",
            head_count=q_heads),
        cls("heads_out:full_kv", ["full_attn.k_proj.weight",
                                  "full_attn.v_proj.weight"],
            "OUTPUT_DIM_HEADS", "output_heads", "per_full_layer",
            kv_heads * head_dim, head_count=kv_heads),
        cls("heads_in:full_o", ["full_attn.o_proj.weight"],
            "INPUT_DIM_HEADS", "input_heads", "per_full_layer",
            q_heads * head_dim, head_count=q_heads),
        cls("heads_out:gdn_qk", ["gdn.q_proj.weight", "gdn.k_proj.weight"],
            "OUTPUT_DIM_HEADS", "output_heads", "per_gdn_layer",
            gdn["key_head_count"] * gdn["key_dimension"],
            head_count=gdn["key_head_count"]),
        cls("heads_out:gdn_v", ["gdn.v_proj.weight"], "OUTPUT_DIM_HEADS",
            "output_heads", "per_gdn_layer",
            gdn["value_head_count"] * gdn["value_dimension"],
            head_count=gdn["value_head_count"]),
        cls("heads_in:gdn_out", ["gdn.out_proj.weight"], "INPUT_DIM_HEADS",
            "input_heads", "per_gdn_layer",
            gdn["value_head_count"] * gdn["value_dimension"],
            head_count=gdn["value_head_count"]),
        cls("dense_gate_up", ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "OUTPUT_DIM", "output", "per_layer",
            m["dense_intermediate_dimension"]),
        cls("dense_down", ["mlp.down_proj.weight"], "INPUT_DIM", "input",
            "per_layer", m["dense_intermediate_dimension"]),
        cls("embed_and_head", ["model.embed_tokens.weight", "lm_head.weight"],
            "OUTPUT_DIM", "output", "per_model", vocab),
        cls("replicated", ["layernorm", "norm.weight", "conv", "gate",
                           "model.norm.weight", "mtp."],
            "REPLICATED", "none", "mixed", None,
            "norms, short-conv state, the GDN forget/write gates (which "
            "have no producer yet - see the contract's known_gaps) and the "
            "MTP layer"),
    ]
    # a fully dense model: first_routed == layer_count lifts the PP cut rules
    return {"layer_count": layers, "first_routed_layer": layers,
            "layer_costs": layer_costs,
            "final_stage_extra_cost": hidden * vocab,
            "kv_geometry": kv_geometry, "shard_classes": shard_classes}


def adapt_mimo25(c):
    m, hyb, att, moe = (c["model"], c["hybrid_attention"], c["attention"],
                        c["moe"])
    hidden, layers, vocab = (m["hidden_dimension"], m["layer_count"],
                             m["vocabulary_size"])
    q_heads = att["query_head_count"]
    head_dim, v_dim = att["head_dimension"], att["value_head_dimension"]
    q_rows = q_heads * head_dim
    o_in = att["o_input_dimension"]
    full_attn = (hidden * q_rows +
                 hidden * att["full_kv_head_count"] * (head_dim + v_dim) +
                 o_in * hidden)
    swa_attn = (hidden * q_rows +
                hidden * att["swa_kv_head_count"] * (head_dim + v_dim) +
                o_in * hidden)
    dense_mlp = 3 * hidden * m["dense_intermediate_dimension"]
    routed_mlp = moe["experts_per_token"] * 3 * hidden * moe["expert_intermediate_dimension"]
    full_set = set(hyb["full_layer_indices"])
    layer_costs = [
        (full_attn if i in full_set else swa_attn) +
        (dense_mlp if i < m["first_routed_layer"] else routed_mlp)
        for i in range(layers)
    ]
    kv_geometry = {
        "layout": "gqa_full_kv_bf16_prescaled",
        "layer_count": layers,
        "full_layer_count": hyb["full_layer_count"],
        "swa_layer_count": hyb["swa_layer_count"],
        "full_layer_indices": hyb["full_layer_indices"],
        "query_head_count": q_heads,
        "head_dimension": head_dim,
        "value_head_dimension": v_dim,
        "full_kv_head_count": att["full_kv_head_count"],
        "swa_kv_head_count": att["swa_kv_head_count"],
        "rope_dimension": att["rope_dimension"],
        "rope_convention": att["rope_convention"],
        "full_rope_theta": att["full_rope_theta"],
        "swa_rope_theta": att["swa_rope_theta"],
        "cached_value_scale": att["cached_value_scale"],
        "swa_sink_bias": att["swa_sink_bias"],
        "sliding_window_tokens": hyb["sliding_window_tokens"],
        "kv_element_bits": c["cache"]["kv_element_bits"],
    }

    def cls(name, match, shard_class, split, scope, extent, note=None,
            group=None, instances=None, head_count=None):
        return {"name": name, "match": match, "shard_class": shard_class,
                "split": split, "scope": scope, "split_extent": extent,
                "quant_group": group, "note": note,
                "instances_per_model": instances, "head_count": head_count}

    block = c["precision"]["fp8_scale_block"]
    shard_classes = [
        cls("heads_out:q", ["self_attn.qkv_proj.weight:q"],
            "OUTPUT_DIM_HEADS", "output_heads", "per_layer", q_rows,
            "the fused qkv splits [q | k | v] head-major (family header "
            "REFERENCE-PIN); the q segment is what shards",
            head_count=q_heads),
        cls("kv_segments", ["self_attn.qkv_proj.weight:k",
                            "self_attn.qkv_proj.weight:v"],
            "REPLICATED", "none", "per_layer", None,
            "4 full / 8 SWA KV heads do not divide 16; replicating keeps "
            "the cached pre-scaled values identical per rank"),
        cls("heads_in:o", ["self_attn.o_proj.weight"], "INPUT_DIM_HEADS",
            "input_heads", "per_layer", o_in, head_count=q_heads),
        cls("expert_gate_up", ["mlp.experts.gate_proj.weight",
                               "mlp.experts.up_proj.weight"],
            "OUTPUT_DIM", "output", "per_expert_per_layer",
            moe["expert_intermediate_dimension"], group=block,
            note="per-rank rows stay whole [128,128] fp8 scale blocks",
            instances=moe["routed_expert_count"] * (layers - m["first_routed_layer"])),
        cls("expert_down", ["mlp.experts.down_proj.weight"], "INPUT_DIM",
            "input", "per_expert_per_layer",
            moe["expert_intermediate_dimension"], group=block,
            instances=moe["routed_expert_count"] * (layers - m["first_routed_layer"])),
        cls("dense_gate_up", ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "OUTPUT_DIM", "output", "per_dense_layer",
            m["dense_intermediate_dimension"]),
        cls("dense_down", ["mlp.down_proj.weight"], "INPUT_DIM", "input",
            "per_dense_layer", m["dense_intermediate_dimension"]),
        cls("embed_and_head", ["model.embed_tokens.weight", "lm_head.weight"],
            "OUTPUT_DIM", "output", "per_model", vocab),
        cls("replicated", ["layernorm", "norm.weight", "mlp.gate",
                           "sink", "model.norm.weight", "mtp."],
            "REPLICATED", "none", "mixed", None,
            "norms, the f32 router and its correction bias, SWA sink "
            "biases and the three MTP draft layers"),
    ]
    return {"layer_count": layers, "first_routed_layer": m["first_routed_layer"],
            "layer_costs": layer_costs,
            "final_stage_extra_cost": hidden * vocab,
            "kv_geometry": kv_geometry, "shard_classes": shard_classes}


MODELS = {
    "k3": {"contract": "k3_authoritative.json", "family": "k3",
           "adapter": adapt_k3},
    "dsv4": {"contract": "dsv4_flash_authoritative.json", "family": "dsv4_flash",
             "adapter": lambda c: adapt_dsv4(c, "dsv4_flash")},
    "dsv4pro": {"contract": "dsv4_pro_authoritative.json", "family": "dsv4_pro",
                "adapter": lambda c: adapt_dsv4(c, "dsv4_pro")},
    "glm52": {"contract": "glm52.json", "family": "glm52",
              "adapter": adapt_glm52},
    "qwen38_27b": {"contract": "qwen38_27b_authoritative.json", "family": "qwen38_27b",
               "adapter": adapt_qwen38_27b},
    "mimo25": {"contract": "mimo25_authoritative.json", "family": "mimo25",
               "adapter": adapt_mimo25},
}


# ---------------------------------------------------------------------------
# TP: resolve the shard table at a degree. The engine owns refusal; the
# recipe records the decision and its reason.
# ---------------------------------------------------------------------------

def resolve_shard_classes(shard_classes, degree):
    resolved = []
    for entry in shard_classes:
        cls = dict(entry)
        extent = cls["split_extent"]
        if cls["shard_class"] in ("REPLICATED", "ENGINE_UNCLASSIFIED"):
            cls["degree_resolution"] = {
                "status": "replicated", "rows_per_rank": extent,
                "reason": cls["note"]}
        elif extent is None:
            cls["degree_resolution"] = {
                "status": "replicated", "rows_per_rank": None,
                "reason": "split extent is not contract-derived; replicate "
                          "until the pack pins it"}
        elif cls["split"] in ("output_heads", "input_heads") and (
                cls["head_count"] is None or
                cls["head_count"] % degree != 0):
            cls["degree_resolution"] = {
                "status": "replicated", "rows_per_rank": extent,
                "reason": f"{cls['head_count']} heads do not split "
                          f"{degree} ways on whole head blocks"}
        elif extent % degree != 0:
            cls["degree_resolution"] = {
                "status": "replicated", "rows_per_rank": extent,
                "reason": f"{extent} rows do not split {degree} ways on "
                          f"whole rows/head-blocks"}
        else:
            per = extent // degree
            if (cls["quant_group"] and cls["split"] in ("input",) and
                    per % cls["quant_group"] != 0):
                cls["degree_resolution"] = {
                    "status": "replicated", "rows_per_rank": extent,
                    "reason": f"per-rank K {per} is not whole quantization "
                              f"groups of {cls['quant_group']}; a partial "
                              f"group cannot be sliced"}
            else:
                cls["degree_resolution"] = {
                    "status": "sharded", "rows_per_rank": per,
                    "reason": None}
        resolved.append(cls)
    return resolved


def rank_table(resolved, degree, node_names):
    ranks = []
    for rank in range(degree):
        slices = {}
        for cls in resolved:
            resolution = cls["degree_resolution"]
            if resolution["rows_per_rank"] is None:
                continue
            if resolution["status"] == "sharded":
                slices[cls["name"]] = [rank * resolution["rows_per_rank"],
                                       resolution["rows_per_rank"]]
            else:
                slices[cls["name"]] = [0, resolution["rows_per_rank"]]
        ranks.append({"rank": rank, "node": node_names[rank],
                      "slices": slices})
    return ranks


# ---------------------------------------------------------------------------
# PP recipe placement. Strictly-better candidates replace, so the earliest
# split of equal cost wins deterministically.
# ---------------------------------------------------------------------------

def routed_in_range(first, count, first_routed, layer_count):
    begin = max(first, first_routed)
    end = min(first + count, layer_count)
    return max(0, end - begin)


def range_valid(first, count, first_routed, layer_count):
    if count == 0 or first >= layer_count or count > layer_count - first:
        return False
    if first_routed < layer_count:
        if first != 0 and first < first_routed:
            return False
        if first + count < first_routed:
            return False
    return (routed_in_range(first, count, first_routed, layer_count) <=
            MAX_ROUTED_PER_STAGE)


def _validate_stage_capacity(stage_capacity, stage_count):
    if stage_capacity is None:
        return [1.0] * stage_count
    if len(stage_capacity) != stage_count:
        raise RecipeFailure(
            f"stage capacity profile has {len(stage_capacity)} entries, "
            f"expected {stage_count}")
    normalized = []
    for capacity in stage_capacity:
        if not isinstance(capacity, (int, float)) or capacity <= 0:
            raise RecipeFailure("stage capacity must be positive")
        normalized.append(float(capacity))
    return normalized


def build_balanced_stages(layer_costs, first_routed, stage_count,
                          final_extra, stage_capacity=None):
    layers = len(layer_costs)
    if stage_count == 0 or stage_count > layers:
        raise RecipeFailure(
            f"{stage_count} stages over {layers} layers is not placeable")
    weighted = stage_capacity is not None
    stage_capacity = _validate_stage_capacity(stage_capacity, stage_count)
    prefix = [0]
    for cost in layer_costs:
        if cost <= 0:
            raise RecipeFailure("layer cost cannot be zero")
        prefix.append(prefix[-1] + cost)
    best = [[None] * (layers + 1) for _ in range(stage_count + 1)]
    split = [[None] * (layers + 1) for _ in range(stage_count + 1)]
    best[0][0] = 0
    for stage in range(1, stage_count + 1):
        for layer in range(1, layers + 1):
            for cut in range(layer):
                if best[stage - 1][cut] is None:
                    continue
                if not range_valid(cut, layer - cut, first_routed, layers):
                    continue
                segment = prefix[layer] - prefix[cut]
                if stage == stage_count:
                    segment += final_extra
                candidate_segment = (segment / stage_capacity[stage - 1]
                                     if weighted else segment)
                candidate = max(best[stage - 1][cut], candidate_segment)
                if best[stage][layer] is None or candidate < best[stage][layer]:
                    best[stage][layer] = candidate
                    split[stage][layer] = cut
    if best[stage_count][layers] is None:
        raise RecipeFailure(
            f"{stage_count} stages cannot satisfy the cut rules "
            f"({layers} layers, first routed {first_routed}, at most "
            f"{MAX_ROUTED_PER_STAGE} routed per stage)")
    stages = []
    layer = layers
    for stage in range(stage_count, 0, -1):
        cut = split[stage][layer]
        stages.append({"first_layer_index": cut, "layer_count": layer - cut})
        layer = cut
    stages.reverse()
    return stages, best[stage_count][layers]


def stage_flags(first, count, stage_index, stage_count, first_routed):
    flags = ["INPUT_HIDDEN"]
    if stage_index + 1 < stage_count:
        flags.append("OUTPUT_HIDDEN")
    else:
        flags.append("FINAL_TOKEN")
    if first == 0 and count >= first_routed:
        flags.append("DENSE_PREFIX")
    return flags


# ---------------------------------------------------------------------------
# Recipe assembly
# ---------------------------------------------------------------------------

def load_topology(topology_path, degree):
    if topology_path is not None and Path(topology_path).exists():
        path = Path(topology_path)
        topo = json.loads(path.read_text(encoding="utf-8"))
        nodes = sorted(topo["compute_nodes"], key=lambda n: n.get("rank", 0))
        names = [n["name"] for n in nodes]
        if len(names) < degree:
            names += [None] * (degree - len(names))
        return {"source": str(path.relative_to(ROOT)),
                "name": topo["topology"]["name"],
                "mode": topo["topology"]["mode"],
                "nodes": degree,
                "node_names": names[:degree]}
    return {"source": "--nodes flag", "name": None, "mode": None,
            "nodes": degree, "node_names": [None] * degree}


def build_recipe(tag, contract_rel, contract_text_sha, contract, geometry,
                 strategy, degree, topology, stage_capacity=None,
                 stage_capacity_profile_sha256=None):
    family = MODELS[tag]["family"]
    g_hash = geometry_hash(family, geometry["kv_geometry"])
    body = {
        "schema_version": 1,
        "kind": "sparkpipe.model_recipe",
        "model": tag,
        "family": family,
        "strategy": strategy,
        "degree": degree,
        "contract": {"path": contract_rel, "sha256": contract_text_sha},
        "topology": topology,
        "kv_geometry": geometry["kv_geometry"],
        "geometry_hash": g_hash,
        "kv_entry_prefix": f"{tag}.{strategy}{degree}.{g_hash}/",
    }
    if strategy == "TP":
        resolved = resolve_shard_classes(geometry["shard_classes"], degree)
        body["tp"] = {
            "shard_classes": resolved,
            "collective": "all_reduce_sum_f32 closes every input-dimension "
                          "split (SparkTpCollectiveAllReduceSumF32 on this "
                          "ring)",
            "unclassified_policy": "refuse",
            "ranks": rank_table(resolved, degree, topology["node_names"]),
        }
    else:
        placement_capacity = _validate_stage_capacity(stage_capacity, degree)
        stages, optimum = build_balanced_stages(
            geometry["layer_costs"], geometry["first_routed_layer"], degree,
            geometry["final_stage_extra_cost"], stage_capacity)
        prefix = [0]
        for cost in geometry["layer_costs"]:
            prefix.append(prefix[-1] + cost)
        stage_entries = []
        for index, stage in enumerate(stages):
            first, count = stage["first_layer_index"], stage["layer_count"]
            cost = prefix[first + count] - prefix[first]
            if index + 1 == len(stages):
                cost += geometry["final_stage_extra_cost"]
            stage_entries.append({
                "stage": index,
                "node": topology["node_names"][index],
                "first_layer_index": first,
                "layer_count": count,
                "routed_layer_count": routed_in_range(
                    first, count, geometry["first_routed_layer"],
                    geometry["layer_count"]),
                "flags": stage_flags(first, count, index, len(stages),
                                     geometry["first_routed_layer"]),
                "cost": cost,
            })
            if stage_capacity is not None:
                stage_entries[-1]["normalized_cost"] = (
                    cost / placement_capacity[index])
        costs = [s["cost"] for s in stage_entries]
        effective_costs = [cost / placement_capacity[index]
                           for index, cost in enumerate(costs)]
        body["pp"] = {
            "cost_model": "analytic_active_params_v1",
            "cost_model_status": (
                "MEASURED_STAGE_CAPACITY + analytic placement estimate"
                if stage_capacity is not None else
                "NOT_MEASURED - analytic placement estimate only"),
            "cut_rules": {
                "contiguous_cover": True,
                "dense_prefix_whole_in_stage_zero": True,
                "max_routed_layers_per_stage": MAX_ROUTED_PER_STAGE,
                "final_token_stage": "last",
                "source": "tools/generate_recipe.py",
            },
            "engine_stage_cap": ENGINE_MAX_STAGE_COUNT,
            "engine_cap_note": None if degree <= ENGINE_MAX_STAGE_COUNT else
                f"{degree} stages exceed "
                f"SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT "
                f"({ENGINE_MAX_STAGE_COUNT}); the engine constant must be "
                f"lifted before this topology loads the plan",
            "stages": stage_entries,
            "balance": {"max_stage_cost": max(effective_costs)
                        if stage_capacity is not None else max(costs),
                        "min_stage_cost": min(effective_costs)
                        if stage_capacity is not None else min(costs),
                        "optimum_max_cost": optimum},
        }
        if stage_capacity is not None:
            body["pp"]["balance"]["stage_capacity"] = placement_capacity
        if stage_capacity_profile_sha256 is not None:
            body["pp"]["placement_profile"] = {
                "kind": "stage_capacity_v1",
                "sha256": stage_capacity_profile_sha256,
                "semantics": "capacity relative to one unit; higher is "
                              "faster",
            }
    body["content_hash"] = short_hash(body)
    body["datafile"] = f"{tag}.{strategy}{degree}.{body['content_hash']}.json"
    return body


def render(recipe):
    return json.dumps(recipe, indent=2, sort_keys=True) + "\n"


def managed_files(out_dir, tags):
    files = {}
    if out_dir.exists():
        for path in sorted(out_dir.iterdir()):
            if path.is_file() and DATAFILE_RE.match(path.name) and \
                    path.name.split(".")[0] in tags:
                files[path.name] = path.read_text(encoding="utf-8")
    return files


def load_stage_profile(path, stage_count):
    path = Path(path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise RecipeFailure("stage capacity profile has an unknown schema")
    if profile.get("kind") == "sparkpipe.stage_profile":
        stage_time_ns = profile.get("stage_time_ns")
        if (not isinstance(stage_time_ns, list) or not stage_time_ns or
                any(not isinstance(value, (int, float)) or value <= 0
                    for value in stage_time_ns)):
            raise RecipeFailure("stage profile must contain positive times")
        reference = sum(stage_time_ns) / len(stage_time_ns)
        capacities = [reference / value for value in stage_time_ns]
    elif profile.get("kind") == "sparkpipe.stage_capacity_profile":
        capacities = profile.get("stage_capacity")
        if not isinstance(capacities, list):
            raise RecipeFailure("stage capacity profile must contain a list")
    else:
        raise RecipeFailure("stage capacity profile has an unknown kind")
    return _validate_stage_capacity(capacities, stage_count), file_sha256(path)


def generate_set(tags, strategies, degrees, topology_path,
                 stage_profile_path=None):
    expected = {}
    for tag in tags:
        spec = MODELS[tag]
        contract_path = CONTRACTS / spec["contract"]
        if not contract_path.exists():
            raise RecipeFailure(f"{tag}: contract {contract_path} is missing")
        contract_rel = f"model_contracts/{spec['contract']}"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        geometry = spec["adapter"](contract)
        for degree in degrees:
            topology = load_topology(topology_path, degree)
            stage_capacity = None
            profile_sha256 = None
            if stage_profile_path is not None and "PP" in strategies:
                stage_capacity, profile_sha256 = load_stage_profile(
                    stage_profile_path, degree)
            for strategy in strategies:
                recipe = build_recipe(tag, contract_rel,
                                      file_sha256(contract_path), contract,
                                      geometry, strategy, degree, topology,
                                      stage_capacity, profile_sha256)
                expected[recipe["datafile"]] = render(recipe)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", choices=sorted(MODELS),
                        help="restrict to these tags (default: all)")
    parser.add_argument("--strategy", choices=("TP", "PP", "both"),
                        default="both")
    parser.add_argument("--nodes", type=int, action="append",
                        help="ring size, repeatable (default: 16 and 13)")
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY,
                        help="hardware topology JSON; read when present, "
                             "otherwise node names fall back to null")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed set is current; write "
                             "nothing")
    parser.add_argument("--stage-profile", "--stage-capacity-profile",
                        dest="stage_profile", type=Path,
                        help="optional generic PP placement profile; use "
                             "stage_time_ns or relative stage_capacity")
    args = parser.parse_args()

    tags = args.model or sorted(MODELS)
    strategies = ["TP", "PP"] if args.strategy == "both" else [args.strategy]
    degrees = args.nodes or list(DEFAULT_DEGREES)
    topology_path = args.topology if args.topology.exists() else None

    expected = generate_set(tags, strategies, degrees, topology_path,
                            args.stage_profile)
    if args.check:
        actual = managed_files(args.out_dir, tags)
        stale = sorted(set(actual) ^ set(expected))
        stale += sorted(name for name in set(actual) & set(expected)
                        if actual[name] != expected[name])
        if stale:
            print("stale generated recipe files:")
            for name in stale:
                print(f"{args.out_dir.relative_to(ROOT)}/{name}")
            return 1
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    current = managed_files(args.out_dir, tags)
    for name in sorted(set(current) - set(expected)):
        (args.out_dir / name).unlink()
    for name, text in sorted(expected.items()):
        (args.out_dir / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(expected)} recipes to "
          f"{args.out_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
