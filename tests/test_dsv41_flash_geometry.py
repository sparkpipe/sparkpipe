#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "model-families" / "dsv41_flash" / "include"
CONTRACT = ROOT / "model_contracts" / "dsv41_flash_authoritative.json"

SCALARS = (
    "HIDDEN_DIMENSION", "LAYER_COUNT", "OUTPUT_VOCAB_COUNT", "MAXIMUM_CONTEXT_TOKENS",
    "BOS_TOKEN_ID", "EOS_TOKEN_ID", "PAD_TOKEN_ID", "IMAGE_TOKEN_ID", "DSPARK_NOISE_TOKEN_ID",
    "ENCODER_LAYER_COUNT", "DECODER_LAYER_COUNT", "ATTENTION_HEAD_COUNT", "HEAD_DIMENSION",
    "QK_ROPE_HEAD_DIMENSION", "QUERY_LORA_RANK", "KV_LATENT_DIMENSION", "OUTPUT_LORA_RANK",
    "OUTPUT_GROUP_COUNT", "OUTPUT_A_DIMENSION", "OUTPUT_GROUP_INPUT_DIMENSION", "SLIDING_WINDOW",
    "SWA_LAYERS", "CSA2_LAYER_COUNT", "CSA2_FULL_LAYER_COUNT", "CSA2_REINDEX_LAYER_COUNT",
    "CSA2_REUSE_LAYER_COUNT", "CANDIDATE_SOURCE_LAYER", "INDEX_HEAD_COUNT", "INDEX_HEAD_DIMENSION",
    "INDEX_TOP_K", "CANDIDATE_BLOCK_COUNT", "CANDIDATE_BLOCK_SIZE", "CANDIDATE_POOL_POSITIONS",
    "ROUTED_EXPERT_COUNT", "SHARED_EXPERT_COUNT", "EXPERTS_PER_TOKEN", "MOE_INTERMEDIATE_DIMENSION",
    "HC_MULT", "HC_SINKHORN_ITERATIONS", "ENGRAM_MODULE_COUNT", "ENGRAM_LAYER_0", "ENGRAM_LAYER_1",
    "ENGRAM_HEAD_COUNT", "ENGRAM_HEAD_DIMENSION", "DSPARK_BLOCK_COUNT", "DSPARK_BLOCK_SIZE",
    "DSPARK_ROUTED_EXPERT_COUNT", "DSPARK_EXPERTS_PER_TOKEN", "DSPARK_MARKOV_RANK",
    "KV_GLOBAL_SCALE_CHANNELS", "EXPERT_BYTES_PER_EXPERT", "KV_SOURCE_COUNT", "INDEX_SOURCE_COUNT",
)
FLOATS = ("RMS_NORM_EPSILON", "SWIGLU_LIMIT", "ROUTED_SCALING_FACTOR")
LISTS = {
    "KV_SOURCE_LAYER": 4,
    "INDEX_SOURCE_LAYER": 8,
}


def program():
    lines = ['#include <stdio.h>', '#include "sparkpipe/spark_dsv41_flash_model.h"', '#include "sparkpipe/llm_defines.h"', "int main(void)", "{", "unsigned layer;"]
    for name in SCALARS:
        lines.append(f'printf("{name} %llu\\n",(unsigned long long)(SPARK_DSV41_FLASH_MODEL_{name}));')
    for name in FLOATS:
        lines.append(f'printf("{name} %.9g\\n",(double)(SPARK_DSV41_FLASH_MODEL_{name}));')
    for name, count in LISTS.items():
        for index in range(count):
            lines.append(f'printf("{name}_{index} %u\\n",(unsigned)(SPARK_DSV41_FLASH_MODEL_{name}_{index}));')
    lines.append("for (layer=0u; layer<SPARK_DSV41_FLASH_MODEL_LAYER_COUNT; layer++)")
    lines.append('printf("LAYER %u %u %u %u\\n",layer,(unsigned)SPARK_DSV41_FLASH_MODEL_LAYER_COMPRESSION_RATIO(layer),(unsigned)SPARK_DSV41_FLASH_MODEL_LAYER_IS_SWA_ONLY(layer),(unsigned)SPARK_DSV41_FLASH_MODEL_LAYER_IS_CSA2(layer));')
    lines.append("return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def header_values():
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "geometry.c"
        binary = Path(temp) / "geometry"
        source.write_text(program())
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE), str(source), "-o", str(binary)], check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    values, layers = {}, []
    for line in output.splitlines():
        fields = line.split()
        if fields[0] == "LAYER":
            layers.append(tuple(int(field) for field in fields[1:]))
        else:
            values[fields[0]] = float(fields[1]) if fields[0] in FLOATS else int(fields[1])
    return values, layers


def expected(contract):
    model, csa2, moe, engram, dspark = contract["model"], contract["csa2"], contract["moe"], contract["engram"], contract["dspark"]
    plan, indexer = csa2["layer_plan"], csa2["indexer"]
    census = contract["tensor_census"]["derived_expert_bytes"]
    return {
        "HIDDEN_DIMENSION": model["hidden_dimension"], "LAYER_COUNT": model["layer_count"],
        "OUTPUT_VOCAB_COUNT": model["vocabulary_size"], "MAXIMUM_CONTEXT_TOKENS": model["max_position_embeddings"],
        "BOS_TOKEN_ID": model["bos_token_id"], "EOS_TOKEN_ID": model["eos_token_id"], "PAD_TOKEN_ID": model["pad_token_id"],
        "IMAGE_TOKEN_ID": model["image_token_id"], "DSPARK_NOISE_TOKEN_ID": dspark["noise_token_id"],
        "ENCODER_LAYER_COUNT": contract["ced"]["encoder_layer_count"], "DECODER_LAYER_COUNT": contract["ced"]["decoder_layer_count"],
        "ATTENTION_HEAD_COUNT": model["attention_head_count"], "HEAD_DIMENSION": model["head_dimension"],
        "QK_ROPE_HEAD_DIMENSION": model["qk_rope_head_dim"], "QUERY_LORA_RANK": model["q_lora_rank"],
        "KV_LATENT_DIMENSION": model["kv_latent_dimension"], "OUTPUT_LORA_RANK": model["o_lora_rank"],
        "OUTPUT_GROUP_COUNT": model["o_groups"], "OUTPUT_A_DIMENSION": model["o_groups"] * model["o_lora_rank"],
        "OUTPUT_GROUP_INPUT_DIMENSION": model["o_group_input_dimension"], "SLIDING_WINDOW": model["sliding_window"],
        "SWA_LAYERS": len(plan["swa_only_layers"]), "CSA2_LAYER_COUNT": len(plan["encoder_csa2_m2"]) + len(plan["decoder_csa2_m1"]),
        "CSA2_FULL_LAYER_COUNT": len(plan["full_mode_layers"]), "CSA2_REINDEX_LAYER_COUNT": len(plan["reindex_mode_layers"]),
        "CSA2_REUSE_LAYER_COUNT": len(plan["encoder_csa2_m2"]) + len(plan["decoder_csa2_m1"]) - len(plan["full_mode_layers"]) - len(plan["reindex_mode_layers"]),
        "CANDIDATE_SOURCE_LAYER": plan["candidate_source_layer_id"], "INDEX_HEAD_COUNT": indexer["head_count"],
        "INDEX_HEAD_DIMENSION": indexer["head_dim"], "INDEX_TOP_K": indexer["topk"],
        "CANDIDATE_BLOCK_COUNT": indexer["candidate_topk_blocks"], "CANDIDATE_BLOCK_SIZE": indexer["candidate_block_size"],
        "CANDIDATE_POOL_POSITIONS": indexer["candidate_pool_positions"], "ROUTED_EXPERT_COUNT": moe["routed_expert_count"],
        "SHARED_EXPERT_COUNT": moe["shared_expert_count"], "EXPERTS_PER_TOKEN": moe["experts_per_token"],
        "MOE_INTERMEDIATE_DIMENSION": moe["moe_intermediate_dimension"], "HC_MULT": contract["hyper_connection"]["mult"],
        "HC_SINKHORN_ITERATIONS": contract["hyper_connection"]["sinkhorn_iters"], "ENGRAM_MODULE_COUNT": engram["modules"],
        "ENGRAM_LAYER_0": engram["layer_ids"][0], "ENGRAM_LAYER_1": engram["layer_ids"][1], "ENGRAM_HEAD_COUNT": engram["n_heads"],
        "ENGRAM_HEAD_DIMENSION": engram["head_dim"], "DSPARK_BLOCK_COUNT": model["num_nextn_predict_layers"],
        "DSPARK_BLOCK_SIZE": dspark["dspark_block_size"], "DSPARK_ROUTED_EXPERT_COUNT": dspark["routed_expert_count"],
        "DSPARK_EXPERTS_PER_TOKEN": dspark["experts_per_token"], "DSPARK_MARKOV_RANK": dspark["markov_rank"],
        "KV_GLOBAL_SCALE_CHANNELS": 16, "EXPERT_BYTES_PER_EXPERT": census["per_expert_packed"],
        "KV_SOURCE_COUNT": len(plan["kv_source_layer_ids"]), "INDEX_SOURCE_COUNT": len(plan["index_source_layer_ids"]),
        "RMS_NORM_EPSILON": model["rms_norm_epsilon"], "SWIGLU_LIMIT": contract["precision"]["swiglu_limit"],
        "ROUTED_SCALING_FACTOR": moe["routed_scaling_factor"],
    }


def main():
    contract = json.loads(CONTRACT.read_text())
    values, layers = header_values()
    failures = [f"{name}: header {values[name]} contract {want}" for name, want in expected(contract).items() if abs(values[name] - want) > abs(want) * 1e-6]
    plan = contract["csa2"]["layer_plan"]
    for index, layer in enumerate(plan["kv_source_layer_ids"]):
        if values[f"KV_SOURCE_LAYER_{index}"] != layer:
            failures.append(f"KV_SOURCE_LAYER_{index}")
    for index, layer in enumerate(plan["index_source_layer_ids"]):
        if values[f"INDEX_SOURCE_LAYER_{index}"] != layer:
            failures.append(f"INDEX_SOURCE_LAYER_{index}")
    ratios = contract["csa2"]["compress_ratios_43_entries"]
    swa = set(plan["swa_only_layers"])
    csa2 = set(plan["encoder_csa2_m2"]) | set(plan["decoder_csa2_m1"])
    for layer, ratio, is_swa, is_csa2 in layers:
        if ratio != ratios[layer] or is_swa != (layer in swa) or is_csa2 != (layer in csa2) or is_swa == is_csa2:
            failures.append(f"layer {layer}: ratio {ratio} swa {is_swa} csa2 {is_csa2}")
    if swa != {layer for layer in range(len(layers)) if ratios[layer] == 0} or csa2 != {layer for layer in range(len(layers)) if ratios[layer] > 0}:
        failures.append("layer plan disagrees with the compress ratio table")
    if len(ratios) != contract["model"]["layer_count"] + contract["model"]["num_nextn_predict_layers"]:
        failures.append("compress ratio table does not cover backbone and dspark layers")
    if len(layers) != contract["model"]["layer_count"] or swa | csa2 != set(range(len(layers))):
        failures.append("layer plan does not partition the backbone")
    if values["QK_ROPE_HEAD_DIMENSION"] >= values["HEAD_DIMENSION"] or values["OUTPUT_GROUP_INPUT_DIMENSION"] * values["OUTPUT_GROUP_COUNT"] != values["ATTENTION_HEAD_COUNT"] * values["HEAD_DIMENSION"]:
        failures.append("attention output grouping")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"FAIL dsv41_flash geometry ({len(failures)})")
        return 1
    print(f"PASS dsv41_flash geometry: {len(values)} header values and {len(layers)} layers match {CONTRACT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
