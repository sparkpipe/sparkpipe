#!/usr/bin/env python3
"""Generate the hy4 lane's model header and normalized contract from the
authoritative JSON. Follows the generate_dsv4_contracts.py shape: one
editable source (model_contracts/hy4_authoritative.json), generated outputs
must regenerate byte-exact (--check) or the gate fails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "model_contracts" / "hy4_authoritative.json"
HEADER_PATH = ROOT / "model-families" / "hy4" / "include" / "sparkpipe" / "spark_hy4_model.h"
NORMALIZED_PATH = ROOT / "model_contracts" / "hy4.json"

TP_RANKS = 16


def require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description}: expected {expected!r}, got {actual!r}")


def validate_contract(contract: dict[str, Any]) -> None:
    model = contract["model"]
    attention = contract["attention"]
    hyper = contract["hyper_connections"]
    moe = contract["moe"]
    shards = contract["shards"]

    require_equal(contract["schema_version"], 1, "schema version")
    require_equal(contract["architecture"], "HYV4ForCausalLM", "architecture")
    require_equal(model["hidden_dimension"], 6144, "hidden dimension")
    require_equal(model["layer_count"], 78, "layer count")
    require_equal(model["mtp_layer_count"], 0, "shipped MTP layers")
    require_equal(model["vocabulary_size"], 120832, "vocabulary size")
    require_equal(model["attention_head_count"], 64, "query heads")
    require_equal(model["head_dimension"], 256, "qk head dimension")
    require_equal(model["qk_nope_head_dimension"] +
                  model["qk_rope_head_dimension"], 256, "qk head split")
    require_equal(attention["index_head_count"], 32, "indexer heads")
    require_equal(attention["index_top_k"], 2048, "indexer top-k")
    require_equal(moe["routed_expert_count"], 256, "routed experts")
    require_equal(moe["experts_per_token"], 8, "experts per token")
    require_equal(moe["shared_expert_count"], 1, "shared experts")
    require_equal(hyper["stream_count"], 4, "hyper-connection streams")
    require_equal(model["vocabulary_size"] % TP_RANKS, 0, "vocab TP split")
    require_equal(attention["index_head_count"] % TP_RANKS, 0,
                  "indexer head TP split")

    require_equal(contract["source_index_sha256"],
                  "12d325844103bac75bd286d14e0e45f87e35e8e60401877282a30b6f26ba6ac6",
                  "source GGUF sha256")
    require_equal(contract["source_bytes"], 235351974336, "source bytes")
    require_equal(contract["source_tensor_count"], 2134, "source tensors")
    require_equal(sorted(shards["ranks"], key=int),
                  [str(i) for i in range(TP_RANKS)], "rank ids")
    require_equal(len({sha for sha in shards["ranks"].values()}), TP_RANKS,
                  "rank digests distinct")
    for sha in shards["ranks"].values():
        require_equal(len(sha), 64, "rank digest length")
    require_equal(shards["rank_bytes"] * TP_RANKS, 299589835776,
                  "deployed rank bytes total")
    require_equal(contract["qualification"]["cuda_target"], "sm_121a",
                  "cuda target")
    require_equal(contract["qualification"]["production_ready"], False,
                  "readiness")


def c_float(value: float) -> str:
    if value == int(value):
        return f"{int(value)}.0f"
    return f"{value:.10g}f"


def render_header(contract: dict[str, Any]) -> str:
    model = contract["model"]
    attention = contract["attention"]
    hyper = contract["hyper_connections"]
    prefix = "SPARK_HY4_MODEL"

    lines = [
        "#pragma once",
        "",
        '#include "sparkpipe/llm_defines.h"',
        "",
        f"#define {prefix}_ID SPARK_LLM_MODEL_SOURCE_URI",
        f"#define {prefix}_SOURCE_REVISION SPARK_LLM_MODEL_REVISION",
        f"#define {prefix}_SOURCE_SHA256 "
        f"{json.dumps(contract['source_index_sha256'])}",
        f"#define {prefix}_TP_RANKS SPARK_LLM_TP_DEGREE",
        "",
        f"#define {prefix}_HIDDEN_DIMENSION SPARK_LLM_HIDDEN_DIMENSION",
        f"#define {prefix}_LAYER_COUNT SPARK_LLM_LAYER_COUNT",
        f"#define {prefix}_MTP_LAYER_COUNT {model['mtp_layer_count']}u",
        f"#define {prefix}_VOCAB_COUNT SPARK_LLM_VOCAB_COUNT",
        f"#define {prefix}_VOCAB_PER_RANK \\",
        "	(SPARK_LLM_OUTPUT_VOCAB_COUNT / SPARK_LLM_TP_DEGREE)",
        f"#define {prefix}_MAX_POSITIONS SPARK_LLM_MAXIMUM_CONTEXT_TOKENS",
        f"#define {prefix}_ATTN_QUERY_HEAD_COUNT SPARK_LLM_MLA_HEAD_COUNT",
        f"#define {prefix}_ATTN_QUERY_HEADS_PER_RANK \\",
        "	(SPARK_LLM_MLA_HEAD_COUNT / SPARK_LLM_TP_DEGREE)",
        f"#define {prefix}_ATTN_KV_HEAD_COUNT {model['kv_head_count']}u",
        f"#define {prefix}_QK_HEAD_DIMENSION SPARK_LLM_MLA_QK_HEAD_DIMENSION",
        f"#define {prefix}_QK_NOPE_HEAD_DIMENSION "
        f"SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION",
        f"#define {prefix}_QK_ROPE_HEAD_DIMENSION "
        f"SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION",
        f"#define {prefix}_V_HEAD_DIMENSION SPARK_LLM_MLA_V_HEAD_DIMENSION",
        f"#define {prefix}_KV_LORA_RANK SPARK_LLM_MLA_LATENT_DIMENSION",
        f"#define {prefix}_QUERY_LORA_RANK SPARK_LLM_MLA_QUERY_A_DIMENSION",
        f"#define {prefix}_LEARNABLE_SINK "
        f"{1 if attention['learnable_sink'] else 0}u",
        f"#define {prefix}_INDEX_HEAD_COUNT {attention['index_head_count']}u",
        f"#define {prefix}_INDEX_HEADS_PER_RANK \\",
        f"	({prefix}_INDEX_HEAD_COUNT / SPARK_LLM_TP_DEGREE)",
        f"#define {prefix}_INDEX_HEAD_DIMENSION "
        f"{attention['index_head_dimension']}u",
        f"#define {prefix}_INDEX_TOP_K {attention['index_top_k']}u",
        f"#define {prefix}_INDEXER_FULL_PERIOD "
        f"{attention['indexer_full_period']}u",
        f"#define {prefix}_ROUTED_EXPERT_COUNT SPARK_LLM_MOE_EXPERT_COUNT",
        f"#define {prefix}_EXPERTS_PER_RANK \\",
        "	(SPARK_LLM_MOE_EXPERT_COUNT / SPARK_LLM_TP_DEGREE)",
        f"#define {prefix}_SHARED_EXPERT_COUNT SPARK_LLM_MOE_SHARED_EXPERT_COUNT",
        f"#define {prefix}_EXPERTS_PER_TOKEN SPARK_LLM_MOE_TOP_K",
        f"#define {prefix}_EXPERT_INTERMEDIATE_DIMENSION \\",
        "	SPARK_LLM_MOE_INTERMEDIATE_DIMENSION",
        f"#define {prefix}_DENSE_FFN_INTERMEDIATE_DIMENSION \\",
        "	SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION",
        f"#define {prefix}_HC_STREAM_COUNT {hyper['stream_count']}u",
        f"#define {prefix}_ROPE_THETA SPARK_LLM_ROPE_THETA",
        f"#define {prefix}_RMS_NORM_EPSILON SPARK_LLM_RMS_NORM_EPSILON",
        f"#define {prefix}_HC_MAGNITUDE {c_float(hyper['magnitude'])}",
        f"#define {prefix}_HC_EPSILON {c_float(hyper['epsilon'])}",
        f"#define {prefix}_ROUTED_SCALING_FACTOR "
        f"SPARK_LLM_MOE_ROUTED_SCALING_FACTOR",
        f"#define {prefix}_SWIGLU_LIMIT SPARK_LLM_SWIGLU_LIMIT",
        f"#define {prefix}_ATTN_QUERY_DIMENSION SPARK_LLM_MLA_QUERY_DIMENSION",
        f"#define {prefix}_INDEX_DIMENSION \\",
        f"	({prefix}_INDEX_HEAD_COUNT * {prefix}_INDEX_HEAD_DIMENSION)",
        f"#define {prefix}_IS_INDEXER_FULL_LAYER(layer) \\",
        f"	(((layer) % {prefix}_INDEXER_FULL_PERIOD) == 0u)",
        f"#define {prefix}_IS_INDEXER_ACTIVE_LAYER(layer) \\",
        f"	(((layer) < 2u) || {prefix}_IS_INDEXER_FULL_LAYER(layer))",
        f"#define {prefix}_ATTN_KV_HEADS_PER_RANK \\",
        f"	({prefix}_ATTN_KV_HEAD_COUNT / {prefix}_TP_RANKS)",
        f"#define {prefix}_EXPERT_GROUPS_PER_LAYER \\",
        f"	({prefix}_ROUTED_EXPERT_COUNT / {prefix}_EXPERTS_PER_RANK)",
        f"#define {prefix}_HC_FN_OUTPUT_ROWS (2u * {prefix}_HC_STREAM_COUNT)",
        f"#define {prefix}_HC_FLAT_WIDTH \\",
        f"	({prefix}_HC_STREAM_COUNT * {prefix}_HIDDEN_DIMENSION)",
        f"#define {prefix}_EXPERT_SCALE_GROUP_SIZE SPARK_LLM_FP8_SCALE_BLOCK",
        f"#define {prefix}_ROUTE_GROUP_MAX \\",
        f"	({prefix}_EXPERTS_PER_TOKEN * {prefix}_HC_STREAM_COUNT)",
        "",
    ]
    return "\n".join(lines)


def render_normalized_contract(contract: dict[str, Any]) -> str:
    normalized = {
        "schema_version": contract["schema_version"],
        "model_id": contract["model_id"],
        "architecture": contract["architecture"],
        "source_revision": contract["source_revision"],
        "model": contract["model"],
        "attention": contract["attention"],
        "hyper_connections": contract["hyper_connections"],
        "moe": contract["moe"],
        "mtp": contract["mtp"],
        "source_precision": contract["source_precision"],
        "source_index_sha256": contract["source_index_sha256"],
        "source_bytes": contract["source_bytes"],
        "source_tensor_count": contract["source_tensor_count"],
        "runtime": contract["runtime"],
        "qualification": contract["qualification"],
    }
    return json.dumps(normalized, indent=2, sort_keys=True) + "\n"


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
    stale = []
    for path, content in (
        (HEADER_PATH, render_header(contract)),
        (NORMALIZED_PATH, render_normalized_contract(contract)),
    ):
        if not write_or_check(path, content, args.check):
            stale.append(str(path.relative_to(ROOT)))
    if stale:
        print("stale generated hy4 contract files:")
        for path in stale:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
