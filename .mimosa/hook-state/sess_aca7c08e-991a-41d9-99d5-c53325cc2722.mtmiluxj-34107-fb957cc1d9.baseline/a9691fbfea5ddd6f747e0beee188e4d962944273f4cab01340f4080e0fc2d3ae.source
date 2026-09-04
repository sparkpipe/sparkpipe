#!/usr/bin/env python3
"""Generate the glm5_next checkpoint->pack name mapping (M2).

Consumes the frozen contract (model_contracts/glm53_flash_authoritative.json)
and, when available, the cached tensor manifest (.lane_cache/glm53_source/
glm53_tensor_manifest.tsv, produced by tools/glm53_contract_freeze.py) to emit:

  model-families/glm5_next/name_map.json        - the mapping table
  model-families/glm5_next/tensor_patterns.json - every checkpoint tensor
        pattern with dtype/shape/count evidence (committed so tests are
        hermetic; the raw 76k-row manifest stays out of git)

The mapping names the DONOR MODULE for every packed field: glm52 (MLA +
indexer + MoE + FP8 spine), k3 (KDA), dsv4 (hyper-connections + the kpool
compressor family). Where the checkpoint math differs from the donor the
entry carries a "delta" note; the module build (M3) resolves them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "model_contracts" / "glm53_flash_authoritative.json"
CACHE_MANIFEST = ROOT / ".lane_cache" / "glm53_source" / "glm53_tensor_manifest.tsv"
OUT_MAP = ROOT / "model-families" / "glm5_next" / "name_map.json"
OUT_PATTERNS = ROOT / "model-families" / "glm5_next" / "tensor_patterns.json"

LM = "model.language_model"
L = f"{LM}.layers.{{layer}}"


def pattern_of(name: str) -> str:
    p = re.sub(r"layers\.\d+", "layers.{layer}", name)
    return re.sub(r"experts\.\d+", "experts.{expert}", p)


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    kda_layers = contract["hybrid_attention"]["kda_layers"]
    dsa_layers = contract["hybrid_attention"]["dsa_layers"]
    mtp_layer = contract["mtp"]["layer_index"]
    first_dense = contract["moe"]["first_dense_layer_count"]

    # ---- every mapping entry: (checkpoint pattern, pack field, donor, notes)
    P = lambda s: L + "." + s  # noqa: E731
    entries = []

    def e(checkpoint, pack, donor, layer_class, shard_class="replicated",
          codec="bf16", transform="", delta=""):
        entries.append({
            "checkpoint_pattern": checkpoint,
            "pack_field": pack,
            "donor_module": donor,
            "layer_class": layer_class,
            "shard_class": shard_class,
            "codec": codec,
            "transform": transform,
            "delta": delta,
        })

    # -- globals
    e(f"{LM}.embed_tokens.weight", "embedding_weight", "glm52", "global",
      "vocab_rows", "bf16")
    e(f"{LM}.norm.weight", "final_norm_weight", "glm52", "global")
    e("lm_head.weight", "lm_head_weight", "glm52", "global", "vocab_rows",
      "bf16", transform="",
      delta="code ties lm_head to embed (_tied_weights_keys) but the "
            "checkpoint stores an independent lm_head; pack both, compare "
            "bytes at M4")

    # -- per-layer common
    e(P("input_layernorm.weight"), "attn_norm_weight", "glm52", "all")
    e(P("post_attention_layernorm.weight"), "mlp_norm_weight", "glm52", "all")
    for site in ("attn", "ffn"):
        e(P(f"hc_{site}_fn"), f"hc_{site}_fn", "dsv4", "hc_layers",
          codec="bf16",
          delta="dsv4 stores fn F32; glm53 checkpoint stores BF16 "
                "[24, 4*hidden] - pack as stored, upcast at load")
        e(P(f"hc_{site}_base"), f"hc_{site}_base", "dsv4", "hc_layers",
          codec="f32")
        e(P(f"hc_{site}_scale"), f"hc_{site}_scale", "dsv4", "hc_layers",
          codec="f32")

    # -- KDA layers (k3 donor)
    A = P("self_attn.")
    e(A + "q_proj.weight", "kda_qkv_beta_weight", "k3", "kda_layers",
      "output_dim_heads", "bf16",
      transform="fuse rows q|k|v|beta from q_proj+k_proj+v_proj+b_proj "
                "(k3_fused_qkvb sections: [64*128, 64*128, 64*128, 64] rows)")
    e(A + "k_proj.weight", "kda_qkv_beta_weight", "k3", "kda_layers",
      "output_dim_heads", "bf16", transform="see q_proj (fused)")
    e(A + "v_proj.weight", "kda_qkv_beta_weight", "k3", "kda_layers",
      "output_dim_heads", "bf16", transform="see q_proj (fused)")
    e(A + "b_proj.weight", "kda_qkv_beta_weight", "k3", "kda_layers",
      "output_dim_heads", "bf16",
      transform="beta section: one row per head [64, 4096]",
      delta="per-head scalar beta, sigmoid in kernel (k3-identical)")
    for c in "qkv":
        e(A + f"{c}_conv1d.weight", f"kda_{c}_conv_weight", "k3", "kda_layers",
          "output_dim_heads", "bf16",
          transform="squeeze dim 1: [8192,1,4] -> [8192,4]",
          delta="k3 packs conv as F32; checkpoint is BF16 - pack as stored")
    e(A + "f_a_proj.weight", "kda_decay_gate_down_weight", "k3", "kda_layers",
      "replicated", "bf16",
      transform="fuse rows decay_down|gate_down from f_a_proj+g_a_proj "
                "[128+128, 4096]",
      delta="RESOLVED (k3 kernel audit): LmBoundedDecay is EXACTLY the "
            "glm53 forget gate - exp(-5.0 * sigmoid(exp(A_log_h) * "
            "(W_up(W_down(x)) + dt_bias))) per head per channel - and "
            "RMSNorm+LmOutputGateKernel reproduces RMSNormGated(sigmoid) "
            "exactly. The only delta vs the released K3 checkpoint is FIELD "
            "GEOMETRY: glm53 needs the low-rank fused decay|gate_down "
            "[256,4096] + gate_up [8192,128] pair (k3's own pack V2 "
            "vocabulary) where K3 ships one full-rank kda_gate_weight "
            "[12288,7168]; the module chains the two GEMMs exactly like "
            "the existing decay path. No new kernels.")
    e(A + "g_a_proj.weight", "kda_decay_gate_down_weight", "k3", "kda_layers",
      "replicated", "bf16", transform="see f_a_proj (fused)")
    e(A + "f_b_proj.weight", "kda_decay_up_weight", "k3", "kda_layers",
      "output_dim_heads", "bf16", transform="[8192, 128]")
    e(A + "g_b_proj.weight", "kda_gate_up_weight", "k3", "kda_layers",
      "output_dim_heads", "bf16", transform="[8192, 128]",
      delta="low-rank gate up-projection; donor kernel path is the k3 "
            "low-rank gate (verify kernel still carries it before M3)")
    e(A + "dt_bias", "kda_decay_bias", "k3", "kda_layers",
      "output_dim_heads", "f32",
      delta="reference: g = f_b(f_a(x)) + dt_bias per-head-per-channel "
            "[8192], then -5.0*sigmoid(exp(A_log)*g)")
    e(A + "A_log", "kda_head_log_scale", "k3", "kda_layers", "replicated",
      "f32", transform="[64] = one per head (k3 sliced 96 of 128 source "
                       "heads; glm53 needs no slice)",
      delta="decay_rate = exp(A_log) per head, folded into the safe gate")
    e(A + "o_norm.weight", "kda_out_norm_weight", "k3", "kda_layers",
      "replicated", "f32",
      transform="[128] per-channel, UPCAST to f32 at pack time",
      delta="the gated norm instantiates Weight=float (k3 convention) - a "
            "bf16 store made the kernel read past the tensor and zeroed "
            "channels 64..127 of every head (found on device, fixed)")
    e(A + "o_proj.weight", "kda_out_weight", "k3", "kda_layers",
      "output_dim_heads", "bf16")

    # -- DSA layers (glm52 MLA + indexer donors)
    e(A + "q_a_proj.weight", "mla_q_a_weight", "glm52", "dsa_layers",
      "replicated", "fp8_block", transform="payload + weight_scale_inv")
    e(A + "q_a_layernorm.weight", "mla_q_norm_weight", "glm52", "dsa_layers")
    e(A + "q_b_proj.weight", "mla_q_b_weight", "glm52", "dsa_layers",
      "output_dim_heads", "fp8_block",
      delta="output [64*256, 1536] is NOPE-only (qk_rope_head_dim=0); the "
            "rope slice does not exist - MLA_ROPE_DIM=0 instantiation")
    e(A + "kv_a_proj_with_mqa.weight", "mla_kv_a_weight", "glm52",
      "dsa_layers", "replicated", "fp8_block",
      delta="latent row is 512 with NO rope segment (glm52: 576); the KV "
            "slot is 512 bf16 = 1024 bytes and no cache rope pass runs")
    e(A + "kv_a_layernorm.weight", "mla_kv_a_norm_weight", "glm52", "dsa_layers")
    e(A + "kv_b_proj.weight", "mla_kv_b_key_weight + mla_kv_b_value_weight",
      "glm52", "dsa_layers", "output_dim_heads", "bf16",
      transform="split rows: first 64*256 are the nope key (pack "
                "TRANSPOSED as glm52 K_KV_B_KEY_T), last 64*256 the value",
      delta="no rope key column split needed (glm52 splits nope|rope)")
    e(A + "o_proj.weight", "mla_out_weight", "glm52", "dsa_layers",
      "output_dim_heads", "fp8_block")
    I = A + "indexer."
    e(I + "wq_b.weight", "index_q_weight", "glm52", "dsa_layers",
      "output_dim_heads", "bf16",
      delta="consumes q_a_layernorm output (the MLA q_a path is the "
            "indexer q_a); NO rope - the text stack is NoPE end to end")
    e(I + "wk.weight", "index_k_weight", "glm52", "dsa_layers", "replicated",
      "bf16", delta="single 128-dim k (not per-head); NO rope (NoPE)")
    e(I + "k_norm.weight", "index_norm_weight", "glm52", "dsa_layers",
      "replicated", "bf16",
      delta="LayerNorm WITH bias, eps 1e-6 (glm52 k_norm is also "
            "LayerNorm(w,b) - donor-identical)")
    e(I + "k_norm.bias", "index_norm_bias", "glm52", "dsa_layers",
      "replicated", "bf16")
    e(I + "weights_proj.weight", "index_head_weight", "glm52", "dsa_layers",
      "output_dim_heads", "bf16")
    e(I + "index_kpool_compress_ape", "index_compress_ape", "dsv4",
      "dsa_layers", "replicated", "bf16",
      transform="[4, 128] per-pool-position bias",
      delta="dsv4-0731 compressor carries ape+wkv+wgate+norm at 256 "
            "channels on ratio-4 layers; glm53 has ONLY ape[4,128]+gate "
            "[128,hidden] on every DSA layer - pool key = softmax(gate_j "
            "+ ape_j) weighted sum of the 4 keys")
    e(I + "index_kpool_compress_gate", "index_compress_gate", "dsv4",
      "dsa_layers", "replicated", "bf16",
      transform="[128, 4096] per-token pool-mix logits",
      delta="the pool mixing logits, not a multiplicative gate")

    # -- MoE layers (glm52 donor)
    M = P("mlp.")
    e(M + "gate.weight", "router_weight", "glm52", "moe_layers",
      "replicated", "bf16",
      delta="router math in fp32 (moe_router_dtype float32): sigmoid + "
            "e_score_correction_bias, noaux_tc with n_group=1 topk_group=1, "
            "norm_topk_prob, routed_scaling 2.5 - glm52-identical config")
    e(M + "gate.e_score_correction_bias", "router_correction", "glm52",
      "moe_layers", "replicated", "f32")
    e(M + "experts.{expert}.gate_proj.weight", "expert_up_gate_weight",
      "glm52", "moe_layers", "expert_sharded", "fp8_block",
      transform="up rows then gate rows stacked (glm52 order)")
    e(M + "experts.{expert}.up_proj.weight", "expert_up_gate_weight",
      "glm52", "moe_layers", "expert_sharded", "fp8_block",
      transform="see gate_proj (fused up|gate)")
    e(M + "experts.{expert}.down_proj.weight", "expert_down_weight", "glm52",
      "moe_layers", "expert_sharded", "fp8_block")
    e(M + "shared_experts.gate_proj.weight", "shared_gate_up_weight",
      "glm52", "moe_layers", "output_dim_heads", "fp8_block",
      transform="up|gate stacked")
    e(M + "shared_experts.up_proj.weight", "shared_gate_up_weight", "glm52",
      "moe_layers", "output_dim_heads", "fp8_block",
      transform="see gate_proj (fused)")
    e(M + "shared_experts.down_proj.weight", "shared_down_weight", "glm52",
      "moe_layers", "output_dim_heads", "fp8_block")

    # -- dense layers 0..2
    e(M + "gate_proj.weight", "dense_gate_up_weight", "glm52", "dense_layers",
      "output_dim_heads", "fp8_block", transform="up|gate stacked")
    e(M + "up_proj.weight", "dense_gate_up_weight", "glm52", "dense_layers",
      "output_dim_heads", "fp8_block", transform="see gate_proj (fused)")
    e(M + "down_proj.weight", "dense_down_weight", "glm52", "dense_layers",
      "output_dim_heads", "fp8_block")

    # -- MTP head
    e(P("eh_proj.weight"), "mtp_eh_proj_weight", "glm52", "mtp_layer",
      "replicated", "bf16",
      transform="[4096, 8192] on concat(hidden, embed)")
    e(P("enorm.weight"), "mtp_enorm_weight", "glm52", "mtp_layer")
    e(P("hnorm.weight"), "mtp_hnorm_weight", "glm52", "mtp_layer")
    e(P("shared_head.norm.weight"), "mtp_shared_norm_weight", "glm52",
      "mtp_layer", "replicated", "bf16",
      delta="MTP layer 45 carries no hc_* tensors; its head collapse is the "
            "plain mean + this norm")

    # FP8 companions: every fp8_block entry carries a weight_scale_inv twin
    # in the checkpoint; emit the scale rows so the round-trip closes.
    scale_rows = []
    for entry in entries:
        if entry["codec"] != "fp8_block":
            continue
        if not entry["checkpoint_pattern"].endswith(".weight"):
            continue
        scale_rows.append({
            "checkpoint_pattern": entry["checkpoint_pattern"] + "_scale_inv",
            "pack_field": entry["pack_field"] + "_scale",
            "donor_module": entry["donor_module"],
            "layer_class": entry["layer_class"],
            "shard_class": entry["shard_class"],
            "codec": "f32_scale",
            "transform": "per [128,128] block dequant multipliers "
                         "(scale_inv); expanded across the block at load",
            "delta": "",
        })
    entries.extend(scale_rows)

    mapping = {
        "schema_version": 1,
        "family": "glm5_next",
        "contract": "model_contracts/glm53_flash_authoritative.json",
        "layer_classes": {
            "kda_layers": kda_layers,
            "dsa_layers": dsa_layers,
            "mtp_layer": mtp_layer,
            "dense_layers": list(range(first_dense)),
            "moe_layers": sorted(
                set(kda_layers + dsa_layers) - set(range(first_dense))
            ) + [mtp_layer],
            "hc_layers": sorted(kda_layers + dsa_layers),
        },
        "dispatch": {
            "kda": "layer < 45 and layer % 4 != 3 (34 layers)",
            "dsa": "layer % 4 == 3 and layer < 45 (11 layers)",
            "mtp": "layer == 45 (DSA + MoE + MTP head, no HC)",
            "dense_mlp": "layer < 3",
            "note": "config kda_layers/full_attn_layers agree with the "
                    "checkpoint census (contract hybrid_attention)",
        },
        "reference_semantics": {
            "source": "transformers main: models/glm5_next/modeling_glm5_next.py",
            "kda_forget_gate": "-5.0 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))",
            "kda_beta": "sigmoid(b_proj(x)) per head",
            "kda_core": "kimi delta attention, chunk 64 + recurrent decode, "
                        "qk l2-norm in kernel",
            "kda_output": "RMSNormGated(core, sigmoid(g_b(g_a(x)))) then o_proj",
            "mla": "absorbed nope-only scoring; qk_head_dim = 256, scaling "
                   "256**-0.5; kv latent 512; NO rope anywhere (NoPE)",
            "indexer": "kpool 4 pools; pool key = softmax_j(gate_j + ape_j) "
                       "weighted sum of keys; scores relu(q.k * 128**-0.5) "
                       "per head; head weights softmax-free sum "
                       "weights_proj(x) * 32**-0.5; select 2048/4 = 512 "
                       "pools -> 2048 tokens + tail (max 3), output width "
                       "2051, invalid = -1",
            "hyper_connection": "pre=sigmoid(w*s0+b0)+eps; post=2*sigmoid(...); "
                                "comb=softmax then 20-iter sinkhorn (col "
                                "norm once, then 19x row+col), eps 1e-6; "
                                "streams init to embed expanded; final head "
                                "= UNWEIGHTED MEAN of streams then RMSNorm",
            "moe": "fp32 router, sigmoid+noaux_tc (n_group 1), top-8 of 288, "
                   "norm weights, scale 2.5, + 1 shared expert (int 2048); "
                   "swiglu with limit 10.0 clamp",
        },
        "entries": entries,
    }

    # ---- tensor patterns with evidence (needs the cached manifest)
    if CACHE_MANIFEST.exists():
        manifest = {}
        for line in CACHE_MANIFEST.read_text().splitlines():
            name, dt, shape, shard = line.split("\t")
            manifest[name] = (dt, json.loads(shape))
        pats = {}
        for name, (dt, shape) in sorted(manifest.items()):
            p = pattern_of(name)
            if p not in pats:
                pats[p] = {"dtype": dt, "shape": shape, "count": 0,
                           "example": name}
            pats[p]["count"] += 1
        mapping["tensor_pattern_count"] = len(pats)
        OUT_PATTERNS.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATTERNS.write_text(json.dumps(pats, indent=1, sort_keys=True) + "\n")
        print(f"wrote {OUT_PATTERNS} ({len(pats)} patterns)")
    else:
        print("note: manifest cache absent; tensor_patterns.json not written")

    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAP.write_text(json.dumps(mapping, indent=1, sort_keys=False) + "\n")
    print(f"wrote {OUT_MAP} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
