"""The K3 defines layer is the single parameter file; the cache machinery,
the serving headers, and the kernel config must all state no geometry of
their own. This gate reads spark_k3_llm_defines.h and refuses disagreement
with the generated contract (model_contracts/k3.json, produced from
model_contracts/k3_authoritative.json) and the kernel config, and refuses
any shim re-defining a key the defines layer owns."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFINES = ROOT / "model-families/k3/include/sparkpipe/spark_k3_llm_defines.h"
CONFIG = ROOT / "inference/llms/kimi_k3/config.h"
CONTRACT = ROOT / "model_contracts/k3.json"
SHIMS = [
    ROOT / "model-families/k3/include/sparkpipe/spark_k3_model.h",
    ROOT / "model-families/k3/include/sparkpipe/spark_k3_kv_geometry.h",
    ROOT / "modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_pool_sizing.h",
    ROOT / "modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_resident_decode_stage_module.h",
]


def defines(path, pattern):
    text = path.read_text().replace("\\\n", " ")
    values = {}
    for name, value in re.findall(pattern, text):
        expression = re.sub(r"\(u?int\d+_t\)", "", value.strip())
        expression = re.sub(r"(?<=\d)ull|(?<=\d)u", "", expression)
        expression = re.sub(r"SPARK_K3_\w+|K3_\w+",
                            lambda m: str(values.get(m.group(0), 0)),
                            expression)
        try:
            values[name] = int(eval(expression))
        except Exception:
            pass
    return values


def main():
    geometry = defines(DEFINES, r"#define (SPARK_K3_\w+)[ \t]+([^\n]+)")
    config = defines(CONFIG, r"#define (K3_\w+)[ \t]+([^\n]+)")
    contract = json.loads(CONTRACT.read_text())
    failures = 0
    contract_pairs = [
        ("SPARK_K3_MODEL_HIDDEN_DIMENSION", contract["hidden_dimension"]),
        ("SPARK_K3_MODEL_LAYER_COUNT", contract["layer_count"]),
        ("SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT", contract["output_vocab_count"]),
        ("SPARK_K3_MODEL_MAXIMUM_CONTEXT_TOKENS",
         contract["maximum_context_tokens"]),
        ("SPARK_K3_MODEL_FIRST_ROUTED_LAYER", contract["first_routed_layer"]),
        ("SPARK_K3_MODEL_ATTENTION_PERIOD", contract["attention_period"]),
        ("SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE",
         contract["global_attention_phase"]),
        ("SPARK_K3_MODEL_MLA_LAYER_COUNT", contract["mla_layer_count"]),
        ("SPARK_K3_MODEL_KDA_LAYER_COUNT", contract["kda_layer_count"]),
        ("SPARK_K3_MODEL_MLA_HEAD_COUNT", contract["head_count"]),
        ("SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION",
         contract["qk_nope_head_dimension"]),
        ("SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION",
         contract["qk_unrotated_head_dimension"]),
        ("SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION",
         contract["value_head_dimension"]),
        ("SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION",
         contract["query_a_dimension"]),
        ("SPARK_K3_MODEL_KDA_HEAD_COUNT", contract["kda_head_count"]),
        ("SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION",
         contract["kda_head_key_dimension"]),
        ("SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION",
         contract["kda_head_value_dimension"]),
        ("SPARK_K3_MODEL_KDA_CONV_KERNEL", contract["kda_short_conv_kernel"]),
        ("SPARK_K3_MODEL_MOE_EXPERT_COUNT", contract["moe_expert_count"]),
        ("SPARK_K3_MODEL_MOE_TOP_K", contract["moe_top_k"]),
        ("SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT",
         contract["moe_shared_expert_count"]),
        ("SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION",
         contract["moe_intermediate_dimension"]),
        ("SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION",
         contract["latent_dimension"]),
        ("SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION",
         contract["dense_intermediate_dimension"]),
        ("SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS",
         contract["attnres_block_layers"]),
        ("SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS",
         contract["attnres_max_representations"]),
        ("SPARK_K3_MODEL_MXFP4_GROUP_SIZE", contract["mxfp4_group_size"]),
    ]
    kernel_pairs = [
        ("SPARK_K3_MODEL_MLA_LATENT_DIMENSION", "K3_KV_LORA_RANK"),
        ("SPARK_K3_MODEL_MLA_ROPE_DIMENSION", "K3_QK_UNROTATED_DIM"),
        ("SPARK_K3_KV_BITS", "K3_KV_BITS"),
        ("SPARK_K3_KV_PAGE_SLOTS", "K3_KV_PAGE_SLOTS"),
        ("SPARK_K3_MODEL_KDA_STATE_ELEMENT_BYTES", "K3_KDA_STATE_ELEMENT_BYTES"),
        ("SPARK_K3_MODEL_KDA_A_LOG_SOURCE_HEAD_COUNT",
         "K3_KDA_A_LOG_SOURCE_HEADS"),
    ]
    for name, expected in contract_pairs:
        if geometry.get(name) != expected:
            print(f"  FAIL {name}={geometry.get(name)} but the authoritative "
                  f"contract says {expected}")
            failures += 1
    for name, kernel in kernel_pairs:
        if geometry.get(name) != config.get(kernel):
            print(f"  FAIL {name}={geometry.get(name)} but {kernel}="
                  f"{config.get(kernel)}; the tiers disagree")
            failures += 1
    if geometry.get("SPARK_K3_MODEL_KDA_SLOT_BYTES") != \
            69 * (96 * 128 * 128 * 4 + ((2 * 96 * 128) + 96 * 128) * 4 * 2):
        print("  FAIL the kda slot algebra lost the 6586368 bytes/layer receipt")
        failures += 1
    stage_count = geometry.get("SPARK_K3_PP_STAGE_COUNT")
    stage_base = geometry.get("SPARK_K3_PP_STAGE_BASE_LAYERS")
    stage_remainder = geometry.get("SPARK_K3_PP_STAGE_REMAINDER")
    if stage_count and stage_base is not None and stage_remainder is not None:
        layers = [stage_base + (1 if index < stage_remainder else 0)
                  for index in range(stage_count)]
        firsts = [index * stage_base +
                  (index if index < stage_remainder else stage_remainder)
                  for index in range(stage_count)]
        if list(zip(firsts, layers)) != [(0, 24), (24, 23), (47, 23), (70, 23)] \
                or sum(layers) != contract["layer_count"]:
            print(f"  FAIL pp stage tiling {list(zip(firsts, layers))} != "
                  f"0/24 24/23 47/23 70/23 over {contract['layer_count']}")
            failures += 1
    else:
        print("  FAIL pp stage parameters missing from the defines layer")
        failures += 1
    owned = set(geometry)
    for shim in SHIMS:
        text = shim.read_text()
        for name in owned:
            if re.search(rf"#define {name}\b", text):
                print(f"  FAIL {shim.name} re-defines {name}; the defines "
                      f"layer owns it")
                failures += 1
    print(f"checked {len(contract_pairs)} contract pairs, "
          f"{len(kernel_pairs)} kernel pairs, the kda slab, the pp tiling, "
          f"and {len(SHIMS)} shims for re-defined keys")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe defines layer is the one geometry source the tiers repeat "
          "only by derivation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
