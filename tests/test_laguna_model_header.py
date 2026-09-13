#!/usr/bin/env python3
"""laguna llm_defines module gate (ADOPT-LAG wave, 2026-09-13).

The single-source law: every laguna driver value lives in the family
llm_defines.h, spark_driver_defines.h derives the shared keys once, and
spark_laguna_model.h is the alias shim. This gate

  1. compiles a vector TU against the shim whose _Static_asserts pin every
     contract value and all 48 hybrid-layer derivations to the values in
     model_contracts/laguna_authoritative.json (vectors FROM the same single
     source the driver builds from), and
  2. runs negative controls: each flipped or deleted parameter must FAIL
     compilation with an error naming that parameter.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON_INCLUDE = ROOT / "model-families/common/include"
LAGUNA_INCLUDE = ROOT / "model-families/laguna/include"
SHIM = LAGUNA_INCLUDE / "sparkpipe/spark_laguna_model.h"
LLM_DEFINES = LAGUNA_INCLUDE / "sparkpipe/llm_defines.h"
CONTRACT = ROOT / "model_contracts/laguna_authoritative.json"


def build_vector_tu(config, yarn):
    period = 4
    asserts = []
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION == {config['hidden_size']}u, \"vector hidden\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_LAYER_COUNT == {config['num_hidden_layers']}u, \"vector layers\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT == {config['vocab_size']}u, \"vector vocab\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_MAXIMUM_CONTEXT_TOKENS == {config['max_position_embeddings']}u, \"vector context\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ATTENTION_KV_HEAD_COUNT == {config['num_key_value_heads']}u, \"vector kv heads\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION == {config['head_dim']}u, \"vector head dim\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_SLIDING_WINDOW == {config['sliding_window']}u, \"vector window\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ATTENTION_Q_HEAD_COUNT_FULL == {config['num_attention_heads_per_layer_full']}u, \"vector q full\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ATTENTION_Q_HEAD_COUNT_SLIDING == {config['num_attention_heads_per_layer_sliding']}u, \"vector q sliding\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT == {config['moe']['num_experts']}u, \"vector experts\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_MOE_TOP_K == {config['moe']['num_experts_per_tok']}u, \"vector top k\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION == {config['moe']['moe_intermediate_size']}u, \"vector moe inter\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_MOE_ROUTED_SCALING_FACTOR == {config['moe']['routed_scaling_factor']}f, \"vector routed scale\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION == {config['intermediate_size']}u, \"vector dense inter\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_END_OF_TEXT_TOKEN_COUNT == {len(config['eos_token_ids'])}u, \"vector eos count\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_END_OF_TEXT_TOKEN_ID == {config['eos_token_ids'][0]}u, \"vector eos0\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_END_OF_TURN_TOKEN_ID == {config['eos_token_ids'][1]}u, \"vector eos1\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_PAD_TOKEN_ID == {config['pad_token_id']}u, \"vector pad\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ROPE_FULL_THETA == {int(yarn['theta'])}.0f, \"vector rope theta\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR == {int(yarn['factor'])}.0f, \"vector rope factor\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS == {int(yarn['original_positions'])}.0f, \"vector rope orig\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST == {int(yarn['beta_fast'])}.0f, \"vector beta fast\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW == {int(yarn['beta_slow'])}.0f, \"vector beta slow\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION == {yarn['rotary_dimension']}u, \"vector rot full\");")
    asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_ROPE_SLIDING_ROTARY_DIMENSION == {int(config['head_dim'] * config['rope_sliding']['partial_rotary_factor'])}u, \"vector rot sliding\");")
    asserts.append("_Static_assert(SPARK_LAGUNA_MODEL_KV_SLOT_BYTES_TP8 == 512u, \"vector kv slot\");")
    asserts.append("_Static_assert(SPARK_LAGUNA_MODEL_TP8_KV_HEAD_COUNT == 1u, \"vector rank kv heads\");")
    asserts.append("_Static_assert(SPARK_LAGUNA_MODEL_TP8_Q_HEAD_COUNT_FULL == 6u, \"vector rank q full\");")
    asserts.append("_Static_assert(SPARK_LAGUNA_MODEL_TP8_Q_HEAD_COUNT_SLIDING == 9u, \"vector rank q sliding\");")
    for index in range(config["num_hidden_layers"]):
        sliding = index % period != 0
        expected = config["num_attention_heads_per_layer_sliding"] if sliding else config["num_attention_heads_per_layer_full"]
        asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_LAYER_IS_SLIDING({index}) == {1 if sliding else 0}, \"vector layer {index} sliding\");")
        asserts.append(f"_Static_assert(SPARK_LAGUNA_MODEL_LAYER_HEAD_COUNT({index}) == {expected}u, \"vector layer {index} heads\");")
    return '#include "sparkpipe/spark_laguna_model.h"\n\n' + "\n".join(asserts) + "\nint main(void) { return 0; }\n"


NEGATIVE_CONTROLS = [
    ("period_flip", r"#define SPARK_LLM_ATTENTION_PERIOD .*\n",
     "#define SPARK_LLM_ATTENTION_PERIOD 5u\n", "tile the stack exactly"),
    ("expert_count_flip", r"#define SPARK_LLM_MOE_EXPERT_COUNT .*\n",
     "#define SPARK_LLM_MOE_EXPERT_COUNT 250u\n", "divide into 8 ranks"),
    ("head_dim_flip", r"#define SPARK_LLM_HEAD_DIMENSION .*\n",
     "#define SPARK_LLM_HEAD_DIMENSION 96u\n", "head_dim is stated 128"),
    ("phase_flip", r"#define SPARK_LLM_GLOBAL_ATTENTION_PHASE .*\n",
     "#define SPARK_LLM_GLOBAL_ATTENTION_PHASE 1u\n", "vector layer 1 sliding"),
    ("window_deleted", r"#define SPARK_LLM_SLIDING_WINDOW_TOKENS .*\n",
     None, "SPARK_LLM_SLIDING_WINDOW_TOKENS"),
    ("rope_mode_deleted", r"#define SPARK_LLM_ROPE_MODE .*\n",
     None, "SPARK_LLM_ROPE_MODE"),
    ("kind_flip_to_mla", r"#define SPARK_LLM_ATTENTION_KIND .*\n",
     "#define SPARK_LLM_ATTENTION_KIND SPARK_LLM_ATTENTION_KIND_MLA\n",
     "SPARK_LLM_MLA_LATENT_DIMENSION"),
]


def compile_tu(source, include_dir):
    with tempfile.TemporaryDirectory() as workdir:
        tu = Path(workdir) / "tu.c"
        tu.write_text(source)
        command = [
            "cc", "-std=c11", "-Wall", "-Werror", "-fsyntax-only",
            f"-I{include_dir}", f"-I{LAGUNA_INCLUDE}", f"-I{COMMON_INCLUDE}",
            "-DSPARK_BATCH_BUCKET=1024u", str(tu),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode, result.stderr


def transformed_defines(pattern, replacement):
    text = LLM_DEFINES.read_text()
    if not re.search(pattern, text):
        return None, f"pattern absent: {pattern!r}"
    substitute = replacement if replacement is not None else ""
    return re.sub(pattern, substitute, text, count=1), None


def main():
    contract = json.loads(CONTRACT.read_text())
    config = contract["source"]["config_json"]
    yarn = contract["yarn"]
    failures = []
    vector_tu = build_vector_tu(config, yarn)
    print("laguna llm_defines module gate:")
    with tempfile.TemporaryDirectory() as workdir:
        include_dir = Path(workdir)
        (include_dir / "sparkpipe").mkdir(parents=True)
        (include_dir / "sparkpipe" / "llm_defines.h").write_text(LLM_DEFINES.read_text())
        (include_dir / "sparkpipe" / "spark_laguna_model.h").write_text(SHIM.read_text())
        code, stderr = compile_tu(vector_tu, include_dir)
        if code != 0:
            failures.append(f"vector TU failed against unmodified headers:\n{stderr}")
        else:
            print("  PASS vectors: contract bindings + 48-layer hybrid derivation hold")
        for name, pattern, replacement, expected_text in NEGATIVE_CONTROLS:
            mutated, error = transformed_defines(pattern, replacement)
            if error:
                failures.append(f"{name}: fixture construction failed: {error}")
                continue
            (include_dir / "sparkpipe" / "llm_defines.h").write_text(mutated)
            code, stderr = compile_tu(vector_tu, include_dir)
            (include_dir / "sparkpipe" / "llm_defines.h").write_text(LLM_DEFINES.read_text())
            if code == 0:
                failures.append(f"{name}: flipped/deleted parameter still compiles — the gate has no teeth")
            elif expected_text not in stderr:
                failures.append(
                    f"{name}: compile failed but error does not name"
                    f" '{expected_text}':\n{stderr}"
                )
            else:
                print(f"  PASS negative control {name}: rejected, error names '{expected_text}'")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"FAIL ({len(failures)}) llm_defines gate violations")
        return 1
    print("PASS laguna single-source defines: vectors hold, negative controls bite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
