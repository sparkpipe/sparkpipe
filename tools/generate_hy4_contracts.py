#!/usr/bin/env python3
"""Generate the hy4 normalized contract and the fp8 firmware description
from model_contracts/hy4_authoritative.json, after
checking that contract against the pinned publisher config. --check fails
when a generated file differs from what the contract renders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "model_contracts" / "hy4_authoritative.json"
COMMUNITY_PATH = ROOT / "model_contracts" / "hy4_ud_iq1m_authoritative.json"
REFERENCE_CONFIG_PATH = ROOT / "model_contracts" / "references" / "hy4_preview_config.json"
NORMALIZED_PATH = ROOT / "model_contracts" / "hy4.json"
DESCRIPTION_PATH = ROOT / "examples" / "model_descriptions" / "hy4_resident_decode_stage_fp8_firmware.json"
DRIVER_MODEL_ID = "tencent.hy4-preview.resident-decode-stage-firmware"
STAGE_NAME = "hy4_resident_decode_stage"
MODULE_ID = "spark.hy4.resident_decode_stage.fp8.tp16.v1"
MODULE_TARGET = "cuda.sm121.hy4.resident_decode_stage.fp8"
TP_RANKS = 16
PROGRAM_SLOTS = 64
PROGRAM_INFLIGHT = 4
CONFIG_FIELDS = (
    ("model", "hidden_dimension", "hidden_size"),
    ("model", "layer_count", "num_hidden_layers"),
    ("model", "mtp_layer_count", "num_nextn_predict_layers"),
    ("model", "vocabulary_size", "vocab_size"),
    ("model", "bos_token_id", "bos_token_id"),
    ("model", "eos_token_id", "eos_token_id"),
    ("model", "pad_token_id", "pad_token_id"),
    ("model", "maximum_context_tokens", "max_position_embeddings"),
    ("model", "rms_norm_epsilon", "rms_norm_eps"),
    ("model", "attention_head_count", "num_attention_heads"),
    ("model", "kv_head_count", "num_key_value_heads"),
    ("model", "head_dimension", "qk_head_dim"),
    ("model", "qk_nope_head_dimension", "qk_nope_head_dim"),
    ("model", "qk_rope_head_dimension", "qk_rope_head_dim"),
    ("model", "v_head_dimension", "v_head_dim"),
    ("model", "kv_lora_rank", "kv_lora_rank"),
    ("model", "query_lora_rank", "q_lora_rank"),
    ("model", "lm_head_fp32", "enable_lm_head_fp32"),
    ("model", "tie_word_embeddings", "tie_word_embeddings"),
    ("attention", "gated_mla", "gated_mla"),
    ("attention", "index_head_count", "index_n_heads"),
    ("attention", "index_head_dimension", "index_head_dim"),
    ("attention", "index_top_k", "index_topk"),
    ("attention", "learnable_sink", "learnable_sink"),
    ("hyper_connections", "stream_count", "hc_mult"),
    ("hyper_connections", "magnitude", "hc_magnitude"),
    ("hyper_connections", "epsilon", "hc_eps"),
    ("hyper_connections", "enable_ihc", "enable_ihc"),
    ("moe", "routed_expert_count", "n_routed_experts"),
    ("moe", "shared_expert_count", "n_shared_experts"),
    ("moe", "experts_per_token", "num_experts_per_tok"),
    ("moe", "expert_intermediate_dimension", "moe_intermediate_size"),
    ("moe", "dense_ffn_intermediate_dimension", "intermediate_size"),
    ("moe", "score_function", "gating_type"),
    ("moe", "swiglu_limit", "swiglu_limit"),
    ("moe", "routed_scaling_factor", "routed_scaling_factor"),
    ("moe", "norm_topk_prob", "norm_topk_prob"),
)


def require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description}: expected {expected!r}, got {actual!r}")


def validate_against_config(contract: dict[str, Any], config_bytes: bytes) -> None:
    config = json.loads(config_bytes)
    require_equal(hashlib.sha256(config_bytes).hexdigest(), contract["source_config_sha256"], "pinned publisher config sha256")
    require_equal(config["architectures"], [contract["architecture"]], "architecture")
    for section, field, key in CONFIG_FIELDS:
        require_equal(contract[section][field], config[key], f"{section}.{field} against config {key}")
    require_equal(contract["attention"]["rope_theta"], config["rope_parameters"]["rope_theta"], "rope theta")
    require_equal(contract["attention"]["rope_type"], config["rope_parameters"]["rope_type"], "rope type")
    full = [index for index, kind in enumerate(config["indexer_types"]) if kind == "full"]
    require_equal(contract["attention"]["indexer_full_layers"], full, "indexer full layers")
    dense = [index for index, kind in enumerate(config["mlp_layer_types"]) if kind == "dense"]
    require_equal(dense, list(range(contract["moe"]["dense_ffn_layer_count"])), "dense ffn layers")
    require_equal(config["quantization_config"]["quantization"]["quant_algo"], "MXFP8", "publisher quantization")


def validate_contract(contract: dict[str, Any]) -> None:
    model = contract["model"]
    attention = contract["attention"]
    shards = contract["shards"]
    require_equal(contract["schema_version"], 1, "schema version")
    require_equal(len(contract["source_revision"]), 40, "source revision length")
    require_equal(model["qk_nope_head_dimension"] + model["qk_rope_head_dimension"], model["head_dimension"], "qk head split")
    for name, count in (("vocabulary", model["vocabulary_size"]), ("query heads", model["attention_head_count"]), ("index heads", attention["index_head_count"]), ("routed experts", contract["moe"]["routed_expert_count"])):
        require_equal(count % TP_RANKS, 0, f"{name} TP16 split")
    require_equal(attention["indexer_full_layers"][0], 0, "first layer computes its own indexer")
    require_equal(attention["indexer_full_layers"], sorted(set(attention["indexer_full_layers"])), "indexer full layers ordered")
    require_equal(attention["indexer_full_layers"][-1] < model["layer_count"], True, "indexer full layers in range")
    require_equal(contract["source_precision"]["scale_group_size"], 32, "E8M0 scale group")
    require_equal(shards["topology"], "tp16", "shard topology")
    for field in ("ranks", "rank_bytes", "experts_manifests"):
        require_equal(sorted(shards[field], key=int), [str(rank) for rank in range(TP_RANKS)], f"{field} rank ids")
    for field in ("ranks", "experts_manifests"):
        require_equal(len(set(shards[field].values())), TP_RANKS, f"{field} digests distinct")
        require_equal({len(digest) for digest in shards[field].values()}, {64}, f"{field} digest length")
    require_equal(contract["qualification"]["cuda_target"], "sm_121a", "cuda target")
    require_equal(contract["qualification"]["production_ready"], False, "readiness")


def validate_community_contract(contract: dict[str, Any]) -> None:
    shards = contract["shards"]
    require_equal(contract["model_id"], "AngelSlim/Hy4-preview-GGUF", "community model id")
    require_equal(contract["source_index_sha256"], "12d325844103bac75bd286d14e0e45f87e35e8e60401877282a30b6f26ba6ac6", "source GGUF sha256")
    require_equal(contract["source_bytes"], 235351974336, "source bytes")
    require_equal(sorted(shards["ranks"], key=int), [str(rank) for rank in range(TP_RANKS)], "GGUF rank ids")
    require_equal(len(set(shards["ranks"].values())), TP_RANKS, "GGUF rank digests distinct")
    require_equal(shards["rank_bytes"] * TP_RANKS, 299589835776, "deployed GGUF rank bytes total")


def render_normalized_contract(contract: dict[str, Any]) -> str:
    keys = ("schema_version", "model_id", "architecture", "license", "source_revision", "model", "attention", "hyper_connections", "moe", "mtp", "source_precision", "source_config_sha256", "source_index_sha256", "source_bytes", "source_file_count", "source_safetensors_count", "runtime", "qualification")
    return json.dumps({key: contract[key] for key in keys}, indent=2, sort_keys=True) + "\n"


def render_geometry(contract: dict[str, Any]) -> dict[str, Any]:
    model = contract["model"]
    attention = contract["attention"]
    moe = contract["moe"]
    return {
        "hidden_dimension": model["hidden_dimension"],
        "layer_count": model["layer_count"],
        "mtp_layer_count": model["mtp_layer_count"],
        "mla_head_count": model["attention_head_count"],
        "mla_query_lora_rank": model["query_lora_rank"],
        "mla_latent_dimension": model["kv_lora_rank"],
        "mla_qk_nope_head_dimension": model["qk_nope_head_dimension"],
        "mla_qk_rope_head_dimension": model["qk_rope_head_dimension"],
        "mla_value_head_dimension": model["v_head_dimension"],
        "index_head_count": attention["index_head_count"],
        "index_head_dimension": attention["index_head_dimension"],
        "index_top_k": attention["index_top_k"],
        "indexer_full_layer_count": len(attention["indexer_full_layers"]),
        "hc_mult": contract["hyper_connections"]["stream_count"],
        "moe_expert_count": moe["routed_expert_count"],
        "moe_intermediate_dimension": moe["expert_intermediate_dimension"],
        "moe_top_k": moe["experts_per_token"],
        "dense_intermediate_dimension": moe["dense_ffn_intermediate_dimension"],
        "output_vocab_count": model["vocabulary_size"],
        "tp_degree": TP_RANKS,
    }


def render_description(contract: dict[str, Any]) -> str:
    precision = contract["source_precision"]
    program = {
        "name": "resident_decode",
        "id": 1,
        "max_inflight": PROGRAM_INFLIGHT,
        "completion": "submit_return",
        "scheduling": {
            "flags": ["stream_ordered", "driver_owns_resident_state", "driver_owns_kv_cache", "fixed_firmware", "requires_hidden_transport", "no_file_transport", "no_shell_transport"],
            "max_active_slots": PROGRAM_SLOTS,
            "max_new_tokens": PROGRAM_SLOTS,
            "max_resident_sequences": PROGRAM_SLOTS,
            "max_sequence_tokens": contract["model"]["maximum_context_tokens"],
        },
        "operations": [{"name": STAGE_NAME, "module": MODULE_ID, "configuration": {"expert_weight_codec": "fp8", "fallback_allowed": False, "runtime_backend_selection": "forbidden"}}],
    }
    description = {
        "schema_version": 1,
        "model": {"id": DRIVER_MODEL_ID, "revision": contract["source_revision"]},
        "metadata": {
            "architecture": STAGE_NAME,
            "purpose": "Hy4 preview resident decode firmware: gated MLA with learnable sinks and a DSA indexer, four hyper-connection streams, 256+1 elementwise-gated MoE with MXFP8 experts, TP16",
            "source_model": {"id": contract["model_id"], "revision": contract["source_revision"], "license": contract["license"]},
            "module_geometry": render_geometry(contract),
            "precision_contract": {"expert_weight_codec": "fp8", "expert_scale_encoding": precision["scale_dtype"].lower(), "expert_scale_group_size": precision["scale_group_size"], "accumulator_codec": "fp32", "fallback_allowed": False, "runtime_precision_selection": "forbidden"},
            "qualification": {"status": contract["qualification"]["status"], "production_ready": contract["qualification"]["production_ready"]},
        },
        "stages": [{"name": STAGE_NAME, "target": MODULE_TARGET, "programs": [program]}],
    }
    return json.dumps(description, indent=2) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        return path.exists() and path.read_text(encoding="utf-8") == content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contract = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    validate_contract(contract)
    validate_against_config(contract, REFERENCE_CONFIG_PATH.read_bytes())
    validate_community_contract(json.loads(COMMUNITY_PATH.read_text(encoding="utf-8")))
    outputs = ((NORMALIZED_PATH, render_normalized_contract(contract)), (DESCRIPTION_PATH, render_description(contract)))
    stale = [str(path.relative_to(ROOT)) for path, content in outputs if not write_or_check(path, content, args.check)]
    for path in stale:
        print(f"stale generated hy4 contract file: {path}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
