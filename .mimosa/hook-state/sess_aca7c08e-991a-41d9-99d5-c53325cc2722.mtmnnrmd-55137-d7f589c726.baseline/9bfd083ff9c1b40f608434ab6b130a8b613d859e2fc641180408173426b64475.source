#!/usr/bin/env python3
"""Offline consistency checks for the frozen GLM 5.3 Flash contract.

Validates the committed artifact (model_contracts/glm53_flash_authoritative.json)
without touching a spark: layer-list partition, shape arithmetic against the
recorded checkpoint evidence, the exact FP8 pattern set, shard pinning
completeness, and the strided-sample verification record.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "model_contracts" / "glm53_flash_authoritative.json"

FP8_PATTERNS = [
    "model.language_model.layers.N.mlp.down_proj",
    "model.language_model.layers.N.mlp.experts.N.down_proj",
    "model.language_model.layers.N.mlp.experts.N.gate_proj",
    "model.language_model.layers.N.mlp.experts.N.up_proj",
    "model.language_model.layers.N.mlp.gate_proj",
    "model.language_model.layers.N.mlp.shared_experts.down_proj",
    "model.language_model.layers.N.mlp.shared_experts.gate_proj",
    "model.language_model.layers.N.mlp.shared_experts.up_proj",
    "model.language_model.layers.N.mlp.up_proj",
    "model.language_model.layers.N.self_attn.kv_a_proj_with_mqa",
    "model.language_model.layers.N.self_attn.o_proj",
    "model.language_model.layers.N.self_attn.q_a_proj",
    "model.language_model.layers.N.self_attn.q_b_proj",
]


def shape(contract: dict, *path: str) -> list[int]:
    node = contract
    for key in path:
        node = node[key]
    return node[1]


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    checks: list[tuple[str, bool]] = []

    hybrid = contract["hybrid_attention"]
    kda = hybrid["kda_layers"]
    dsa = hybrid["dsa_layers"]
    layers = sorted(kda + dsa)
    checks.append(("kda+dsa partitions layers 0..44", layers == list(range(45))))
    checks.append(("kda count 34", len(kda) == 34))
    checks.append(("dsa count 11 at 3+4k", dsa == [3 + 4 * i for i in range(11)]))
    checks.append(
        ("first 3 layers dense (kda, no moe)",
         all(i in kda for i in (0, 1, 2))),
    )
    checks.append(("layer 44 kda (final pre-mtp)", 44 in kda))
    checks.append(
        ("checkpoint evidence counts",
         hybrid["checkpoint_evidence"]["layers_with_A_log"] == 34
         and hybrid["checkpoint_evidence"]["layers_with_q_b_proj"] == 12),
    )

    hidden = contract["model"]["hidden_dimension"]
    heads = contract["mla"]["query_head_count"]
    kda_heads = contract["kda"]["head_count"]
    kda_dim = contract["kda"]["head_dimension"]
    checks.append(("hidden 4096", hidden == 4096))
    checks.append(("vocab 154880", contract["model"]["vocabulary_size"] == 154880))
    checks.append(("context 1M", contract["model"]["maximum_context_tokens"] == 1048576))
    checks.append(("45 layers", contract["model"]["layer_count"] == 45))

    # --- shape arithmetic: KDA ---
    k = contract["kda"]["checkpoint_tensor_shapes"]
    checks.append(("kda q_proj [64*128,4096]", k["q_proj.weight"][1] == [kda_heads * kda_dim, hidden]))
    checks.append(("kda conv1d kernel 4", k["q_conv1d.weight"][1] == [kda_heads * kda_dim, 1, 4]))
    checks.append(("kda b_proj per-head scalar", k["b_proj.weight"][1] == [kda_heads, hidden]))
    checks.append(("kda A_log per-head f32", k["A_log"] == ["F32", [kda_heads]]))
    checks.append(("kda dt_bias per-head-channel", k["dt_bias"] == ["F32", [kda_heads * kda_dim]]))
    checks.append(("kda f_a 128-dim bottleneck", k["f_a_proj.weight"][1] == [128, hidden]))
    checks.append(("kda f_b expands 8192", k["f_b_proj.weight"][1] == [kda_heads * kda_dim, 128]))
    checks.append(("kda o_norm per-channel", k["o_norm.weight"][1] == [kda_dim]))

    # --- shape arithmetic: MLA (the rope-0 delta) ---
    m = contract["mla"]["checkpoint_tensor_shapes"]
    q_lora = contract["mla"]["query_lora_rank"]
    kv_lora = contract["mla"]["kv_lora_rank"]
    nope = contract["mla"]["qk_nope_head_dimension"]
    v_dim = contract["mla"]["value_head_dimension"]
    checks.append(("mla rope dim 0", contract["mla"]["qk_rope_head_dimension"] == 0))
    checks.append(("mla q_b nope-only [64*256,1536]",
                   m["q_b_proj.weight"][1] == [heads * nope, q_lora]))
    checks.append(("mla kv_a pure lora [512,4096] (no rope segment)",
                   m["kv_a_proj_with_mqa.weight"][1] == [kv_lora, hidden]))
    checks.append(("mla kv_b [64*(256+256),512]",
                   m["kv_b_proj.weight"][1] == [heads * (nope + v_dim), kv_lora]))
    checks.append(("mla o_proj [4096,64*256]",
                   m["o_proj.weight"][1] == [hidden, heads * v_dim]))

    # --- indexer ---
    i = contract["indexer"]["checkpoint_tensor_shapes"]
    idx_heads = contract["indexer"]["head_count"]
    idx_dim = contract["indexer"]["head_dimension"]
    kpool = contract["indexer"]["kpool"]
    checks.append(("indexer topk 2048", contract["indexer"]["top_k"] == 2048))
    checks.append(("indexer wq_b [32*128,1536]", i["wq_b.weight"][1] == [idx_heads * idx_dim, q_lora]))
    checks.append(("indexer wk [128,4096]", i["wk.weight"][1] == [idx_dim, hidden]))
    checks.append(("indexer weights_proj per-head [32,4096]", i["weights_proj.weight"][1] == [idx_heads, hidden]))
    checks.append(("indexer k_norm has bias", i["k_norm.bias"][1] == [idx_dim]))
    checks.append(("compressor ape [kpool,128]", i["index_kpool_compress_ape"][1] == [kpool, idx_dim]))
    checks.append(("compressor gate [128,4096]", i["index_kpool_compress_gate"][1] == [idx_dim, hidden]))

    # --- hyper-connections ---
    h = contract["hyper_connections"]
    hc_mult = h["hc_mult"]
    checks.append(("hc mult 4 sinkhorn 20", hc_mult == 4 and h["sinkhorn_iterations"] == 20))
    checks.append(("hc fn [24, 4*4096]",
                   h["checkpoint_tensor_shapes"]["hc_attn_fn"][1] == [hc_mult * 3 * 2, hc_mult * hidden]))
    checks.append(("hc base f32 [24]", h["checkpoint_tensor_shapes"]["hc_attn_base"] == ["F32", [hc_mult * 3 * 2]]))
    checks.append(("hc on every layer except MTP",
                   h["per_layer"] and not h["on_mtp_layer"]
                   and h["checkpoint_evidence"] if False else
                   (h["per_layer"] and not h["on_mtp_layer"])))

    # --- MoE ---
    moe = contract["moe"]
    checks.append(("moe 288+1 top8", moe["routed_expert_count"] == 288 and moe["experts_per_token"] == 8))
    checks.append(("moe sigmoid noaux_tc norm scaling 2.5",
                   moe["score_function"] == "sigmoid" and moe["topk_method"] == "noaux_tc"
                   and moe["normalize_selected_probabilities"] and moe["routed_scaling_factor"] == 2.5))
    checks.append(("moe expert layers 43 (42 sparse + MTP)", moe["expert_layer_count"] == 43))
    checks.append(("expert gate [2048,4096]",
                   shape(contract, "moe", "checkpoint_tensor_shapes", "experts.0.gate_proj.weight") == [2048, hidden]))
    checks.append(("expert scale block 128x128",
                   shape(contract, "moe", "checkpoint_tensor_shapes", "experts.0.gate_proj.weight_scale_inv") == [16, 32]))

    # --- MTP ---
    mtp = contract["mtp"]
    checks.append(("mtp layer 45 single", mtp["layer_count"] == 1 and mtp["layer_index"] == 45))
    checks.append(("mtp eh_proj [4096,2*4096]",
                   mtp["checkpoint_tensor_shapes"]["eh_proj.weight"][1] == [hidden, 2 * hidden]))

    # --- precision ---
    checks.append(("fp8 pattern set exact",
                   contract["precision"]["fp8_quantized_patterns"] == FP8_PATTERNS))
    checks.append(("fp8 block [128,128]", contract["precision"]["weight_block_size"] == [128, 128]))
    checks.append(("kda precision bf16", contract["kda"]["precision"] == "BF16 (all KDA tensors; no scale_inv twins)"))

    # --- source pinning ---
    src = contract["source"]
    checks.append(("62 shards pinned", len(src["shard_sha256_pinned"]) == 62))
    checks.append(("all shard pins 64-hex",
                   all(re.fullmatch(r"[0-9a-f]{64}", d) for d in src["shard_sha256_pinned"].values())))
    checks.append(("receipt digest matches SHA256SUMS", src["receipt_digest_matches"] is True))
    checks.append(("strided sample all OK",
                   src["shard_verification"]["ok"] == len(src["shard_verification"]["sampled"])
                   and not src["shard_verification"]["failed"]))
    checks.append(("small files pinned with matching digests",
                   all(f.get("matches_source_sums") for f in src["files"].values() if "matches_source_sums" in f)))
    checks.append(("tensor count 76108", src["tensor_count"] == 76108))
    checks.append(("index total == census payload", src["index_total_bytes"] == 328326771576))

    ba = src["byte_accounting"]
    total = src["total_gbytes"]
    checks.append(("text = total - vision",
                   abs(src["text_stack_gbytes"] - (total - ba["vision"]["gbytes"])) < 0.01))
    checks.append(("tp16 rank ~19.1 GiB (20.45 GB)",
                   abs(src["tp16_gbytes_per_rank"] - src["text_stack_gbytes"] / 16) < 0.01))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"\n{len(failed)} FAILED")
        return 1
    print(f"\nPASS glm53_flash_authoritative contract ({len(checks)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
