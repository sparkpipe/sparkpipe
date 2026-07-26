#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


CONTRACT_RELATIVE_PATH = Path("model_contracts/glm52.json")
HEADER_RELATIVE_PATH = Path("model-families/glm52/include/sparkpipe/spark_glm52_model.h")
INTEGER_MACROS = {
    "SPARK_GLM52_MODEL_HIDDEN_DIMENSION": "hidden_dimension",
    "SPARK_GLM52_MODEL_LAYER_COUNT": "layer_count",
    "SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS": "maximum_context_tokens",
    "SPARK_GLM52_MODEL_KV_POOL_TOKENS": "kv_pool_tokens",
    "SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER": "first_routed_layer",
    "SPARK_GLM52_MODEL_HEAD_COUNT": "head_count",
    "SPARK_GLM52_MODEL_LATENT_DIMENSION": "latent_dimension",
    "SPARK_GLM52_MODEL_ROPE_DIMENSION": "rope_dimension",
    "SPARK_GLM52_MODEL_QUERY_A_DIMENSION": "query_a_dimension",
    "SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION": "qk_nope_head_dimension",
    "SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION": "value_head_dimension",
    "SPARK_GLM52_MODEL_MOE_EXPERT_COUNT": "moe_expert_count",
    "SPARK_GLM52_MODEL_MOE_TOP_K": "moe_top_k",
    "SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION": "moe_intermediate_dimension",
    "SPARK_GLM52_MODEL_MOE_W1_COMPONENT_COUNT": "moe_w1_component_count",
    "SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION": "dense_intermediate_dimension",
    "SPARK_GLM52_MODEL_FP8_SCALE_BLOCK": "fp8_scale_block",
    "SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT": "dsa_selected_token_count",
    "SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT": "dsa_index_head_count",
    "SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION": "dsa_index_head_dimension",
    "SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT": "dsa_index_share_group_layer_count",
    "SPARK_GLM52_MODEL_DSA_INDEX_SKIP_TOPK_OFFSET": "dsa_index_skip_topk_offset",
    "SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT": "output_vocab_count",
    "SPARK_GLM52_MODEL_RESTRICTED_VOCAB_COUNT": "restricted_vocab_count",
    "SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH":
        "max_prefill_tokens_per_dispatch",
    "SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT": "mtp_draft_token_count",
    "SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA":
        "mtp_target_hidden_position_delta",
    "SPARK_GLM52_MODEL_MXFP4_GROUP_SIZE": "mxfp4_group_size",
    "SPARK_GLM52_MODEL_NVFP4_GROUP_SIZE": "nvfp4_group_size",
}
NONNEGATIVE_INTEGER_MACROS = {
    "SPARK_GLM52_MODEL_MTP_EXECUTION_POSITION_OFFSET":
        "mtp_execution_position_offset",
}
FLOAT_MACROS = {
    "SPARK_GLM52_MODEL_RMS_NORM_EPSILON": "rms_norm_epsilon",
    "SPARK_GLM52_MODEL_DSA_INDEX_NORM_EPSILON": "dsa_index_norm_epsilon",
    "SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR": "moe_routed_scaling_factor",
    "SPARK_GLM52_MODEL_ROPE_THETA": "rope_theta",
}
BOOLEAN_MACROS = {
    "SPARK_GLM52_MODEL_DSA_ROPE_INTERLEAVE": "dsa_rope_interleave",
    "SPARK_GLM52_MODEL_ROPE_INTERLEAVE": "rope_interleave",
}
DSPARK_INTEGER_MACROS = {
    "SPARK_GLM52_MODEL_DSPARK_AUX_CAPTURE_LAYER_OFFSET":
        "aux_capture_layer_offset",
    "SPARK_GLM52_MODEL_DSPARK_DRAFT_LAYER_COUNT": "draft_layer_count",
    "SPARK_GLM52_MODEL_DSPARK_BLOCK_SIZE": "block_size",
    "SPARK_GLM52_MODEL_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT": "maximum_speculative_token_count",
    "SPARK_GLM52_MODEL_DSPARK_DRAFT_INTERMEDIATE_DIMENSION": "draft_intermediate_dimension",
    "SPARK_GLM52_MODEL_DSPARK_DRAFT_ATTENTION_HEAD_COUNT": "draft_attention_head_count",
    "SPARK_GLM52_MODEL_DSPARK_DRAFT_KV_HEAD_COUNT": "draft_kv_head_count",
    "SPARK_GLM52_MODEL_DSPARK_DRAFT_HEAD_DIMENSION": "draft_head_dimension",
    "SPARK_GLM52_MODEL_DSPARK_MARKOV_RANK": "markov_rank",
    "SPARK_GLM52_MODEL_DSPARK_MASK_TOKEN_ID": "mask_token_id",
    "SPARK_GLM52_MODEL_DSPARK_MAX_ANCHORS": "max_anchors",
}
EOS_TOKEN_MACROS = {
    "SPARK_GLM52_MODEL_END_OF_TEXT_TOKEN_ID": "end_of_text",
    "SPARK_GLM52_MODEL_USER_TOKEN_ID": "user",
    "SPARK_GLM52_MODEL_OBSERVATION_TOKEN_ID": "observation",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_model_contract(root: Path | None = None) -> Dict[str, Any]:
    base = repository_root() if root is None else root
    contract = json.loads((base / CONTRACT_RELATIVE_PATH).read_text())
    for key in INTEGER_MACROS.values():
        if not isinstance(contract.get(key), int) or contract[key] <= 0:
            raise ValueError(f"invalid GLM-5.2 model contract field: {key}")
    for key in NONNEGATIVE_INTEGER_MACROS.values():
        if not isinstance(contract.get(key), int) or contract[key] < 0:
            raise ValueError(f"invalid GLM-5.2 model contract field: {key}")
    for key in FLOAT_MACROS.values():
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"invalid GLM-5.2 model contract field: {key}")
    for key in BOOLEAN_MACROS.values():
        if not isinstance(contract.get(key), bool):
            raise ValueError(f"invalid GLM-5.2 model contract field: {key}")
    if not contract["rope_interleave"] or not contract["dsa_rope_interleave"]:
        raise ValueError("GLM-5.2 requires interleaved MLA and DSA RoPE")
    if contract["rope_dimension"] % 2 != 0:
        raise ValueError("GLM-5.2 RoPE dimension must contain complete pairs")
    dspark = contract.get("dspark")
    if not isinstance(dspark, dict):
        raise ValueError("invalid GLM-5.2 DSpark model contract")
    for key in DSPARK_INTEGER_MACROS.values():
        if not isinstance(dspark.get(key), int) or dspark[key] <= 0:
            raise ValueError(f"invalid GLM-5.2 DSpark contract field: {key}")
    aux_layer_ids = dspark.get("aux_layer_ids")
    if not isinstance(aux_layer_ids, list) or not aux_layer_ids:
        raise ValueError("invalid GLM-5.2 DSpark aux layer ids")
    if any(not isinstance(value, int) or value < 0 for value in aux_layer_ids):
        raise ValueError("invalid GLM-5.2 DSpark aux layer id")
    eos_token_ids = contract.get("eos_token_ids")
    if not isinstance(eos_token_ids, dict):
        raise ValueError("invalid GLM-5.2 EOS token contract")
    for key in EOS_TOKEN_MACROS.values():
        value = eos_token_ids.get(key)
        if not isinstance(value, int) or value < 0 or value >= contract["output_vocab_count"]:
            raise ValueError(f"invalid GLM-5.2 EOS token id: {key}")
    if len(set(eos_token_ids.values())) != len(EOS_TOKEN_MACROS):
        raise ValueError("GLM-5.2 EOS token ids must be unique")
    return contract


def render_float(value: float) -> str:
    text = format(float(value), ".9g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def render_c_header(contract: Dict[str, Any]) -> str:
    lines = ["#pragma once", "", "#include <stdint.h>", ""]
    for name, key in INTEGER_MACROS.items():
        lines.append(f"#define {name} {contract[key]}u")
    for name, key in NONNEGATIVE_INTEGER_MACROS.items():
        lines.append(f"#define {name} {contract[key]}u")
    eos_token_ids = contract["eos_token_ids"]
    for name, key in EOS_TOKEN_MACROS.items():
        lines.append(f"#define {name} {eos_token_ids[key]}u")
    lines.extend([
        "#define SPARK_GLM52_MODEL_EOS_TOKEN_IDS_INITIALIZER \\",
        "\t{ \\",
        "\t\tSPARK_GLM52_MODEL_END_OF_TEXT_TOKEN_ID, \\",
        "\t\tSPARK_GLM52_MODEL_USER_TOKEN_ID, \\",
        "\t\tSPARK_GLM52_MODEL_OBSERVATION_TOKEN_ID \\",
        "\t}",
    ])
    lines.extend([
        "#define SPARK_GLM52_MODEL_MTP_LAYER_INDEX SPARK_GLM52_MODEL_LAYER_COUNT",
        "#define SPARK_GLM52_MODEL_WEIGHT_LAYER_COUNT \\",
        "\t(SPARK_GLM52_MODEL_MTP_LAYER_INDEX + 1u)",
    ])
    for name, key in FLOAT_MACROS.items():
        lines.append(f"#define {name} {render_float(contract[key])}")
    qk_head_dimension = (
        contract["qk_nope_head_dimension"] + contract["rope_dimension"])
    lines.append(
        "#define SPARK_GLM52_MODEL_QK_SCALE " +
        render_float(1.0 / math.sqrt(qk_head_dimension)))
    lines.append(
        "#define SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE " +
        render_float(1.0 / math.sqrt(contract["dsa_index_head_dimension"])))
    for name, key in BOOLEAN_MACROS.items():
        lines.append(f"#define {name} {1 if contract[key] else 0}u")
    dspark = contract["dspark"]
    aux_layer_ids = dspark["aux_layer_ids"]
    lines.append(
        f"#define SPARK_GLM52_MODEL_DSPARK_AUX_LAYER_COUNT {len(aux_layer_ids)}u"
    )
    lines.append(
        "#define SPARK_GLM52_MODEL_DSPARK_AUX_LAYER_IDS_INITIALIZER { "
        + ", ".join(f"{value}u" for value in aux_layer_ids)
        + " }"
    )
    for name, key in DSPARK_INTEGER_MACROS.items():
        lines.append(f"#define {name} {dspark[key]}u")
    lines.extend([
        "#define SPARK_GLM52_MODEL_MAX_SPECULATIVE_ROWS_PER_LANE \\",
        "\t(SPARK_GLM52_MODEL_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u)",
        "#define SPARK_GLM52_MODEL_DSPARK_AUX_CAPTURE_LAYER_INDEX(aux_layer_id) \\",
        "\t((aux_layer_id) - SPARK_GLM52_MODEL_DSPARK_AUX_CAPTURE_LAYER_OFFSET)",
        "#define SPARK_GLM52_MODEL_BF16_ELEMENT_BYTES ((uint32_t)sizeof(uint16_t))",
        "#define SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES \\",
        "\t(SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_GLM52_MODEL_BF16_ELEMENT_BYTES)",
        "#define SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES \\",
        "\t(SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT * ((uint32_t)sizeof(uint32_t)))",
        "#define SPARK_GLM52_MODEL_DSA_INDEX_QUERY_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT * \\",
        "\t SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION)",
        "",
        "#define SPARK_GLM52_MODEL_QK_HEAD_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION + \\",
        "\t SPARK_GLM52_MODEL_ROPE_DIMENSION)",
        "#define SPARK_GLM52_MODEL_QUERY_B_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_QK_HEAD_DIMENSION)",
        "#define SPARK_GLM52_MODEL_KV_A_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_LATENT_DIMENSION + SPARK_GLM52_MODEL_ROPE_DIMENSION)",
        "#define SPARK_GLM52_MODEL_KV_B_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_HEAD_COUNT * \\",
        "\t (SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION + \\",
        "\t  SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION))",
        "#define SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS SPARK_GLM52_MODEL_KV_A_DIMENSION",
        "#define SPARK_GLM52_MODEL_QUERY_LATENT_PROJECTION_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_LATENT_DIMENSION)",
        "#define SPARK_GLM52_MODEL_ATTENTION_PROJECTION_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION)",
        "#define SPARK_GLM52_MODEL_QUERY_ROPE_PROJECTION_DIMENSION \\",
        "\t(SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_ROPE_DIMENSION)",
        "#define SPARK_GLM52_MODEL_FP8_SCALE_EXTENT(dimension) \\",
        "\t(((dimension) + SPARK_GLM52_MODEL_FP8_SCALE_BLOCK - 1u) / \\",
        "\t SPARK_GLM52_MODEL_FP8_SCALE_BLOCK)",
        "#define SPARK_GLM52_MODEL_FP8_SCALE_COUNT(rows, columns) \\",
        "\t(SPARK_GLM52_MODEL_FP8_SCALE_EXTENT(rows) * \\",
        "\t SPARK_GLM52_MODEL_FP8_SCALE_EXTENT(columns))",
        "#define SPARK_GLM52_MODEL_MAX_FP8_LINEAR_SCALE_COUNT \\",
        "\tSPARK_GLM52_MODEL_FP8_SCALE_COUNT( \\",
        "\t\tSPARK_GLM52_MODEL_HIDDEN_DIMENSION, \\",
        "\t\tSPARK_GLM52_MODEL_ATTENTION_PROJECTION_DIMENSION)",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    expected = render_c_header(load_model_contract(root))
    header_path = root / HEADER_RELATIVE_PATH
    if args.check:
        if header_path.read_text() != expected:
            raise SystemExit("GLM-5.2 generated C model contract is stale")
    else:
        header_path.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
