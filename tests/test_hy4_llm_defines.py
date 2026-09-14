#!/usr/bin/env python3
"""hy4 llm_defines gate: contract-derived vectors, sentinel accounting,
negative controls. Values must equal model_contracts/hy4.json; remaining
SET_ME_ rows must be exactly the honest-unknown set; every filled key that
a test flips must fail compilation naming itself."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON_INCLUDE = ROOT / "model-families/common/include"
HY4_INCLUDE = ROOT / "model-families/hy4/include"
HY4_SOURCE = ROOT / "modules/hy4_resident_decode_stage/source"
LLM_DEFINES = HY4_INCLUDE / "sparkpipe/llm_defines.h"

EXPECTED_SENTINELS = {
    "SPARK_LLM_KV_POOL_TOKENS",
    "SPARK_LLM_MAX_PREFILL_TOKENS_PER_DISPATCH",
    "SPARK_LLM_TILE_N",
    "SPARK_LLM_TILE_STAGES",
    "SPARK_LLM_LAYER_THREADS",
    "SPARK_LLM_MISS_PACK_STRIDE",
    "SPARK_LLM_MISS_RING_CAPACITY",
    "SPARK_LLM_ROUTE_UNION_MAX",
    "SPARK_LLM_ROUTE_UNION_TRIM",
    "SPARK_LLM_TP_COLLECTIVE_BACKEND",
    "SPARK_LLM_ADAPTER_DESCRIPTOR",
}


def compile_tu(source: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as workdir:
        tu = Path(workdir) / "tu.c"
        tu.write_text(source)
        command = [
            "cc", "-std=c11", "-Wall", "-Werror", "-fsyntax-only",
            f"-I{HY4_INCLUDE}", f"-I{COMMON_INCLUDE}", f"-I{HY4_SOURCE}",
            f"-I{ROOT / 'include'}",
            str(tu),
        ]
        result = subprocess.run(command, capture_output=True,
                                text=True)
        return result.returncode, result.stderr


def vector_tu(contract: dict) -> str:
    model = contract["model"]
    attention = contract["attention"]
    moe = contract["moe"]
    hidden = model["hidden_dimension"]
    layers = model["layer_count"]
    vocab = model["vocabulary_size"]
    heads = model["attention_head_count"]
    nope = model["qk_nope_head_dimension"]
    rope = model["qk_rope_head_dimension"]
    vdim = model["v_head_dimension"]
    latent = model["kv_lora_rank"]
    query_lora = model["query_lora_rank"]
    experts = moe["routed_expert_count"]
    top_k = moe["experts_per_token"]
    routed_layers = layers - moe["dense_ffn_layer_count"]
    vocab_per_rank = vocab // 16
    return f"""
#include "sparkpipe/llm_defines.h"
#include "sparkpipe/spark_hy4_model.h"
#include "spark_hy4_stagepack_format.h"

_Static_assert(SPARK_LLM_HIDDEN_DIMENSION == {hidden}u, "vector hidden");
_Static_assert(SPARK_LLM_LAYER_COUNT == {layers}u, "vector layers");
_Static_assert(SPARK_LLM_OUTPUT_VOCAB_COUNT == {vocab}u, "vector vocab");
_Static_assert(SPARK_LLM_MAXIMUM_CONTEXT_TOKENS == 1048576u, "vector context");
_Static_assert(SPARK_LLM_MLA_HEAD_COUNT == {heads}u, "vector heads");
_Static_assert(SPARK_LLM_MLA_QK_HEAD_DIMENSION == {nope + rope}u, "vector qk head");
_Static_assert(SPARK_LLM_MLA_QUERY_DIMENSION == {heads * (nope + rope)}u, "vector query dim");
_Static_assert(SPARK_LLM_MLA_KV_A_DIMENSION == {latent + rope}u, "vector kv a dim");
_Static_assert(SPARK_LLM_MLA_KV_B_DIMENSION == {heads * (nope + vdim)}u, "vector kv b dim");
_Static_assert(SPARK_LLM_MLA_QUERY_A_DIMENSION == {query_lora}u, "vector query lora");
_Static_assert(SPARK_LLM_KV_SLOT_BYTES == {(latent + rope) * 4}u, "vector kv slot bytes");
_Static_assert(SPARK_LLM_ROUTED_LAYERS == {routed_layers}u, "vector routed layers");
_Static_assert(SPARK_LLM_MOE_ROUTED_GATE_UP_DIMENSION == {moe["expert_intermediate_dimension"] * 2}u, "vector gate up dim");
_Static_assert(SPARK_LLM_LAYER_IS_FULL(0u) && SPARK_LLM_LAYER_IS_FULL({layers - 1}u), "vector uniform full");
_Static_assert(SPARK_LLM_ATTENTION_KIND == SPARK_LLM_ATTENTION_KIND_MLA, "vector kind");
_Static_assert(SPARK_LLM_TP_DEGREE == 16u, "vector tp degree");
_Static_assert(SPARK_LLM_FP8_SCALE_BLOCK == 32u, "vector fp8 block");
_Static_assert(SPARK_HY4_MODEL_VOCAB_PER_RANK == {vocab_per_rank}u, "vector vocab per rank");
_Static_assert(SPARK_HY4_MODEL_EXPERTS_PER_RANK == {experts // 16}u, "vector experts per rank");
_Static_assert(SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK == {heads // 16}u, "vector heads per rank");
_Static_assert(SPARK_HY4_MODEL_INDEX_HEADS_PER_RANK == 2u, "vector index heads per rank");
_Static_assert(SPARK_HY4_MODEL_ROUTE_GROUP_MAX == {top_k * 4}u, "vector route group max");
_Static_assert(SPARK_HY4_MODEL_IS_INDEXER_ACTIVE_LAYER(0u), "vector indexer layer 0");
_Static_assert(SPARK_HY4_MODEL_IS_INDEXER_ACTIVE_LAYER(4u), "vector indexer layer 4");
_Static_assert(!SPARK_HY4_MODEL_IS_INDEXER_ACTIVE_LAYER(2u), "vector indexer layer 2");
_Static_assert(SPARK_HY4_STAGEPACK_FP8_PLANES_PER_RANK == 832u, "vector fp8 planes");
_Static_assert(SPARK_HY4_STAGEPACK_FP8_ALIGNED_PLANES_PER_RANK == 573u, "vector fp8 aligned planes");

int main(void) {{ return 0; }}
"""


def expected_values(contract: dict) -> dict[str, str]:
    model = contract["model"]
    attention = contract["attention"]
    moe = contract["moe"]
    return {
        "SPARK_LLM_HIDDEN_DIMENSION": f"{model['hidden_dimension']}u",
        "SPARK_LLM_LAYER_COUNT": f"{model['layer_count']}u",
        "SPARK_LLM_OUTPUT_VOCAB_COUNT": f"{model['vocabulary_size']}u",
        "SPARK_LLM_MAXIMUM_CONTEXT_TOKENS":
            f"{model['maximum_context_tokens']}u",
        "SPARK_LLM_RMS_NORM_EPSILON": f"{model['rms_norm_epsilon']:g}f",
        "SPARK_LLM_MLA_HEAD_COUNT": f"{model['attention_head_count']}u",
        "SPARK_LLM_MLA_QUERY_A_DIMENSION": f"{model['query_lora_rank']}u",
        "SPARK_LLM_MLA_LATENT_DIMENSION": f"{model['kv_lora_rank']}u",
        "SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION":
            f"{model['qk_nope_head_dimension']}u",
        "SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION":
            f"{model['qk_rope_head_dimension']}u",
        "SPARK_LLM_MLA_V_HEAD_DIMENSION": f"{model['v_head_dimension']}u",
        "SPARK_LLM_ROPE_THETA": f"{attention['rope_theta']:.1f}f",
        "SPARK_LLM_MOE_EXPERT_COUNT": f"{moe['routed_expert_count']}u",
        "SPARK_LLM_MOE_TOP_K": f"{moe['experts_per_token']}u",
        "SPARK_LLM_MOE_SHARED_EXPERT_COUNT": f"{moe['shared_expert_count']}u",
        "SPARK_LLM_MOE_INTERMEDIATE_DIMENSION":
            f"{moe['expert_intermediate_dimension']}u",
        "SPARK_LLM_MOE_ROUTED_SCALING_FACTOR":
            f"{moe['routed_scaling_factor']:g}f",
        "SPARK_LLM_FIRST_ROUTED_LAYER": f"{moe['dense_ffn_layer_count']}u",
        "SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION":
            f"{moe['dense_ffn_intermediate_dimension']}u",
        "SPARK_LLM_END_OF_TEXT_TOKEN_ID": f"{model['eos_token_id']}u",
        "SPARK_LLM_PAD_TOKEN_ID": f"{model['pad_token_id']}u",
        "SPARK_LLM_SWIGLU_LIMIT": f"{moe['swiglu_limit']:.1f}f",
        "SPARK_LLM_TP_DEGREE": "16u",
        "SPARK_LLM_FP8_SCALE_BLOCK": "32u",
        "SPARK_LLM_KV_BITS": "32u",
        "SPARK_LLM_BF16_ELEMENT_BYTES": "2u",
        "SPARK_LLM_KV_PAGE_SLOTS": "64u",
        "SPARK_LLM_MTP_ENABLED": "0u",
        "SPARK_LLM_MTP_DRAFT_DEPTH": "0u",
        "SPARK_LLM_MODEL_REVISION": f'"{contract["source_revision"]}"',
        "SPARK_LLM_MODEL_SOURCE_URI": f'"{contract["model_id"]}"',
    }


def main() -> int:
    failures: list[str] = []
    print("hy4 llm_defines gate:")
    contract = json.loads(
        (ROOT / "model_contracts" / "hy4.json").read_text(encoding="utf-8"))
    text = LLM_DEFINES.read_text(encoding="utf-8")

    remaining = set(re.findall(r"SET_ME_([A-Z0-9_]+)", text))
    remaining_named = {f"SPARK_LLM_{name}" for name in remaining}
    if remaining_named != EXPECTED_SENTINELS:
        failures.append(
            "sentinel set mismatch:"
            f" unexpected={sorted(remaining_named - EXPECTED_SENTINELS)}"
            f" missing={sorted(EXPECTED_SENTINELS - remaining_named)}")
    else:
        print(f"  PASS sentinels: {len(EXPECTED_SENTINELS)}"
              " honest-unknown rows, exactly the expected set")
    for name, value in expected_values(contract).items():
        pattern = re.compile(
            rf"^#define {re.escape(name)}\s+{re.escape(value)}$", re.M)
        if not pattern.search(text):
            failures.append(f"missing or wrong define {name} = {value}")
    if not failures:
        print("  PASS contract literals: every filled key equals hy4.json")

    code, stderr = compile_tu(vector_tu(contract))
    if code != 0:
        failures.append(f"vector TU failed:\n{stderr}")
    else:
        print("  PASS vectors: 27 derived-key assertions hold")

    sentinel_probe = """
#include "sparkpipe/llm_defines.h"
uint32_t probe = SPARK_LLM_KV_POOL_TOKENS;
int main(void) { return 0; }
"""
    code, stderr = compile_tu(sentinel_probe)
    if code == 0:
        failures.append("sentinel probe compiled — sentinel rows lost their teeth")
    elif "SET_ME_KV_POOL_TOKENS" not in stderr:
        failures.append(f"sentinel probe error does not name the sentinel:\n{stderr}")
    else:
        print("  PASS sentinel probe: consuming an unfilled key fails naming SET_ME_KV_POOL_TOKENS")

    negative_controls = [
        ("hidden_flip",
         (r"#define SPARK_LLM_HIDDEN_DIMENSION\s+\d+u\n",
          "#define SPARK_LLM_HIDDEN_DIMENSION 6148u\n"),
         "vector hidden"),
        ("tp_degree_flip",
         (r"#define SPARK_LLM_TP_DEGREE\s+\d+u\n",
          "#define SPARK_LLM_TP_DEGREE 8u\n"),
         "TP16 stagepack set"),
        ("kv_page_slots_deleted",
         (r"#define SPARK_LLM_KV_PAGE_SLOTS\s+\d+u\n", ""),
         "SET_ME_KV_PAGE_SLOTS"),
        ("router_groups_flip",
         (r"#define SPARK_LLM_MOE_ROUTER_TOP_GROUPS\s+\d+u\n",
          "#define SPARK_LLM_MOE_ROUTER_TOP_GROUPS 9u\n"),
         "cannot exceed"),
        ("latent_flip",
         (r"#define SPARK_LLM_MLA_LATENT_DIMENSION\s+\d+u\n",
          "#define SPARK_LLM_MLA_LATENT_DIMENSION 640u\n"),
         "vector kv slot bytes"),
    ]
    for name, (pattern, replacement), expected_text in negative_controls:
        if not re.search(pattern, text):
            failures.append(f"{name}: fixture pattern absent: {pattern!r}")
            continue
        mutated = re.sub(pattern, replacement, text, count=1)
        with tempfile.TemporaryDirectory() as workdir:
            include_dir = Path(workdir) / "sparkpipe"
            include_dir.mkdir(parents=True)
            (include_dir / "llm_defines.h").write_text(mutated)
            tu = Path(workdir) / "tu.c"
            tu.write_text(vector_tu(contract))
            command = [
                "cc", "-std=c11", "-Wall", "-Werror", "-fsyntax-only",
                f"-I{workdir}", f"-I{HY4_INCLUDE}", f"-I{COMMON_INCLUDE}",
                f"-I{HY4_SOURCE}", f"-I{ROOT / 'include'}", str(tu),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            failures.append(f"{name}: flipped key still compiles — gate has no teeth")
        elif expected_text not in result.stderr:
            failures.append(
                f"{name}: compile failed but error does not name"
                f" '{expected_text}':\n{result.stderr}")
        else:
            print(f"  PASS negative control {name}: rejected, error names '{expected_text}'")

    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"FAIL ({len(failures)}) hy4 llm_defines gate violations")
        return 1
    print("PASS hy4 llm_defines: contract literals hold, sentinels honest, controls bite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
