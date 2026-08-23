#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "model_contracts" / "k3_authoritative.json"
GENERATED_HEADER = ROOT / "inference" / "llms" / "kimi_k3" / "generated_config.h"
GENERATED_CONTRACT = ROOT / "model_contracts" / "k3.json"
GENERATED_DESCRIPTION = ROOT / "examples" / "model_descriptions" / "k3_resident_decode_stage_firmware.json"


def load_source(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    validate_source(source)
    return source


def require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, got {actual!r}")


def validate_source(source: dict[str, Any]) -> None:
    model = source["model"]
    attention = source["hybrid_attention"]
    kda = source["kda"]
    mla = source["mla"]
    attnres = source["attnres"]
    moe = source["moe"]
    quantization = source["quantization"]

    require_equal(model["layer_count"], 93, "K3 layer count")
    require_equal(model["hidden_dimension"], 7168, "K3 hidden dimension")
    require_equal(attention["kda_layer_count"] + attention["mla_layer_count"], model["layer_count"], "hybrid layer partition")
    require_equal(kda["head_count"], mla["query_head_count"], "KDA/MLA head count")
    require_equal(moe["routed_expert_count"], 896, "routed expert count")
    require_equal(moe["experts_per_token"], 16, "experts per token")
    require_equal(moe["shared_expert_count"], 2, "shared expert count")
    require_equal(attnres["maximum_candidate_representations"], attnres["layer_block_count"] + attnres["embedding_representation_count"], "AttnRes maximum candidates")
    require_equal(quantization["quantized_components"], ["routed_experts"], "K3 quantization scope")
    require_equal(quantization["routed_expert_group_size"], 32, "MXFP4 group size")
    if quantization["non_expert_weight_format"] != "bf16" or quantization["non_expert_activation_format"] != "bf16":
        raise ValueError("K3 non-expert tensors must remain BF16")
    if quantization["accumulator_format"] != "fp32" or kda["state_element_type"] != "fp32":
        raise ValueError("K3 accumulation and recurrent state must remain FP32")

    expected_mla = (model["layer_count"] // attention["period"]) + 1
    require_equal(attention["mla_layer_count"], expected_mla, "MLA layer count")
    require_equal(attention["kda_layer_count"], model["layer_count"] - expected_mla, "KDA layer count")

    # Pack format V2 (tools/k3_pack.py, docs/K3_PACK_FORMAT_V2.md): the
    # geometry must divide the way the layout promises, or the contract is
    # shipping a model the packer has to refuse.
    # - the fused KDA section bases stay 128-aligned only because a hidden
    #   row is a whole number of 128-byte lines
    # - the interleaved expert tensors block K into 128-element tiles
    #   (LmMxfp4::kTileK, whose 64 payload bytes fill one swizzle span) and
    #   outputs into 16-neuron cells; w1's K is the latent, w2's K is the
    #   intermediate, w1's output is gate|up doubled
    # - the scale row closes exactly: 16 neurons x (128/32) bytes == the 64
    #   payload bytes of one k-tile row
    hidden = model["hidden_dimension"]
    latent = moe["latent_dimension"]
    inter = moe["expert_intermediate_dimension"]
    group = quantization["routed_expert_group_size"]
    k_tile = 128  # LmMxfp4::kTileK in inference/kernels/formats/mxfp4.cuh
    require_equal((hidden * 2) % 128, 0, "hidden row 128B alignment")
    require_equal(latent % k_tile, 0, "expert latent whole interleave k-tiles")
    require_equal(inter % k_tile, 0, "expert intermediate whole interleave k-tiles")
    require_equal(latent % group, 0, "expert latent whole scale groups")
    require_equal(inter % group, 0, "expert intermediate whole scale groups")
    require_equal(latent % 16, 0, "expert latent whole 16-neuron cells")
    require_equal((2 * inter) % 16, 0, "expert gate|up whole 16-neuron cells")
    require_equal(16 * (k_tile // group), (k_tile * 4) // 8, "interleave scale row closure")


def c_float(value: float) -> str:
    if value == int(value):
        return f"{int(value)}.0f"
    return f"{value:.10g}f"


def render_header(source: dict[str, Any]) -> str:
    model = source["model"]
    attention = source["hybrid_attention"]
    kda = source["kda"]
    mla = source["mla"]
    attnres = source["attnres"]
    moe = source["moe"]
    quantization = source["quantization"]
    speculation = source["speculation"]
    cache = source["cache"]
    tokens = source["tokens"]
    qk_scale = 1.0 / math.sqrt(mla["qk_nope_dimension"] + mla["qk_unrotated_dimension"])

    values = [
        ("K3_HIDDEN", f"{model['hidden_dimension']}u"),
        ("K3_LAYERS", f"{model['layer_count']}u"),
        ("K3_VOCAB", f"{model['vocabulary_size']}u"),
        ("K3_MAX_CONTEXT", f"{model['maximum_context_tokens']}u"),
        ("K3_RMS_EPSILON", c_float(model["rms_norm_epsilon"])),
        ("K3_LORA_RMS_EPSILON", c_float(model["lora_rms_norm_epsilon"])),
        ("K3_FIRST_ROUTED_LAYER", f"{model['first_routed_layer']}u"),
        ("K3_EXPERTS", f"{moe['routed_expert_count']}u"),
        ("K3_TOP_K", f"{moe['experts_per_token']}u"),
        ("K3_SHARED_EXPERTS", f"{moe['shared_expert_count']}u"),
        ("K3_EXPERT_INTERMEDIATE", f"{moe['expert_intermediate_dimension']}u"),
        ("K3_ROUTED_EXPERT_HIDDEN", f"{moe['latent_dimension']}u"),
        ("K3_DENSE_INTERMEDIATE", f"{moe['dense_intermediate_dimension']}u"),
        ("K3_ROUTED_SCALE", c_float(moe["routed_scaling_factor"])),
        ("K3_MLA_HEADS", f"{mla['query_head_count']}u"),
        ("K3_KV_LORA_RANK", f"{mla['kv_lora_rank']}u"),
        ("K3_Q_LORA_RANK", f"{mla['query_lora_rank']}u"),
        ("K3_QK_NOPE_DIM", f"{mla['qk_nope_dimension']}u"),
        ("K3_QK_UNROTATED_DIM", f"{mla['qk_unrotated_dimension']}u"),
        ("K3_V_HEAD_DIM", f"{mla['value_head_dimension']}u"),
        ("K3_MLA_QK_SCALE", c_float(qk_scale)),
        ("K3_MLA_USE_NOPE", "1u" if mla["uses_nope"] else "0u"),
        ("K3_MLA_OUTPUT_GATE", "1u" if mla["output_gate"] else "0u"),
        ("K3_KDA_HEADS", f"{kda['head_count']}u"),
        ("K3_KDA_KEY_DIM", f"{kda['key_dimension']}u"),
        ("K3_KDA_VALUE_DIM", f"{kda['value_dimension']}u"),
        ("K3_KDA_CONV_KERNEL", f"{kda['short_conv_kernel']}u"),
        ("K3_KDA_GATE_LOWER_BOUND", c_float(kda["minimum_log_decay"])),
        ("K3_KDA_FULL_RANK_GATE", "1u" if kda["full_rank_output_gate"] else "0u"),
        ("K3_KDA_A_LOG_SOURCE_HEADS", f"{kda['a_log_source_head_count']}u"),
        ("K3_KDA_QK_L2NORM", "1u" if kda["qk_l2_normalized"] else "0u"),
        ("K3_ATTNRES_BLOCK_SIZE", f"{attnres['block_size_layers']}u"),
        ("K3_ATTNRES_BLOCK_COUNT", f"{attnres['layer_block_count']}u"),
        ("K3_ATTNRES_MAX_SOURCES", f"{attnres['maximum_candidate_representations']}u"),
        ("K3_ATTNRES_SITES_PER_LAYER", f"{attnres['retrieval_sites_per_layer']}u"),
        ("K3_SITU_BETA", c_float(moe["situ_gate_beta"])),
        ("K3_SITU_LINEAR_BETA", c_float(moe["situ_up_beta"])),
        ("K3_MXFP4_GROUP", f"{quantization['routed_expert_group_size']}u"),
        ("K3_MTP_LAYERS", f"{speculation['base_checkpoint_mtp_layer_count']}u"),
        ("K3_EXTERNAL_DRAFT_LAYERS", f"{speculation['optional_external_draft_layer_count']}u"),
        ("K3_EXTERNAL_DRAFT_UNROLL_STEPS", f"{speculation['training_unroll_steps']}u"),
        ("K3_KV_BITS", f"{cache['kv_element_bits']}u"),
        ("K3_KV_PAGE_SLOTS", f"{cache['kv_page_slots']}u"),
        ("K3_EOS_TOKEN", f"{tokens['end_of_text']}u"),
        ("K3_KDA_LAYER_COUNT", f"{attention['kda_layer_count']}u"),
        ("K3_MLA_LAYER_COUNT", f"{attention['mla_layer_count']}u"),
        ("K3_ATTENTION_PERIOD", f"{attention['period']}u"),
        ("K3_GLOBAL_ATTENTION_PHASE", f"{attention['global_phase']}u"),
        # Pack format V2 (tools/k3_pack.py, docs/K3_PACK_FORMAT_V2.md). The
        # driver binds a pack with these, and validate_source above holds the
        # model geometry to what they assume. K3_PACK_MXFP4_K_TILE is
        # LmMxfp4::kTileK: the k extent whose payload bytes fill exactly one
        # swizzle span; the cell count is the exact 16:1 payload-to-scale
        # byte ratio of group-32 MXFP4 at that tile.
        ("K3_PACK_FORMAT_VERSION", "2u"),
        ("K3_PACK_ALIGNMENT", "128u"),
        ("K3_PACK_MXFP4_K_TILE", "128u"),
        ("K3_PACK_INTERLEAVE_CELL_PAYLOAD_ROWS", "16u"),
        ("K3_PACK_INTERLEAVE_CELL_ROWS", "17u"),
        ("K3_PACK_INTERLEAVE_ROW_BYTES", "64u"),
        ("K3_PACK_INTERLEAVE_SCALE_BYTES", "4u"),
        # The fused KDA projection tensor, as a row count over hidden:
        # q|k|v|beta (OUTPUT_DIM_HEADS). The released checkpoint keeps
        # decay_down standalone and the gate full rank
        # (docs/K3_GATE_RECONCILIATION.md), so there is no second fusion.
        ("K3_KDA_QKVB_FUSED_ROWS",
         "(2u * K3_KDA_HEADS * K3_KDA_KEY_DIM + K3_KDA_HEADS * K3_KDA_VALUE_DIM + K3_KDA_HEADS)")
    ]

    lines = [
        "#pragma once",
        "",
        "/* Generated by tools/generate_k3_contract.py. Do not edit by hand. */",
        ""
    ]
    for name, value in values:
        lines.append(f"#define {name} {value}")
    lines.append("")
    return "\n".join(lines)


def render_contract(source: dict[str, Any]) -> str:
    model = source["model"]
    attention = source["hybrid_attention"]
    kda = source["kda"]
    mla = source["mla"]
    attnres = source["attnres"]
    moe = source["moe"]
    quantization = source["quantization"]
    speculation = source["speculation"]
    cache = source["cache"]
    tokens = source["tokens"]
    contract = {
        "generated_from": "model_contracts/k3_authoritative.json",
        "active_parameters": model["active_parameters"],
        "attention_period": attention["period"],
        "attnres_block_count": attnres["layer_block_count"],
        "attnres_block_layers": attnres["block_size_layers"],
        "attnres_max_representations": attnres["maximum_candidate_representations"],
        "attnres_sites_per_layer": attnres["retrieval_sites_per_layer"],
        "dense_intermediate_dimension": moe["dense_intermediate_dimension"],
        "eos_token_ids": {"end_of_text": tokens["end_of_text"]},
        "first_routed_layer": model["first_routed_layer"],
        "global_attention_phase": attention["global_phase"],
        "head_count": mla["query_head_count"],
        "hidden_dimension": model["hidden_dimension"],
        "kda_head_count": kda["head_count"],
        "kda_head_key_dimension": kda["key_dimension"],
        "kda_head_value_dimension": kda["value_dimension"],
        "kda_layer_count": attention["kda_layer_count"],
        "kda_min_log_decay": kda["minimum_log_decay"],
        "kda_short_conv_kernel": kda["short_conv_kernel"],
        "kda_state_element_type": kda["state_element_type"],
        "layer_count": model["layer_count"],
        "latent_dimension": moe["latent_dimension"],
        "maximum_context_tokens": model["maximum_context_tokens"],
        "mla_layer_count": attention["mla_layer_count"],
        "moe_expert_count": moe["routed_expert_count"],
        "moe_intermediate_dimension": moe["expert_intermediate_dimension"],
        "moe_routed_scaling_factor": moe["routed_scaling_factor"],
        "moe_shared_expert_count": moe["shared_expert_count"],
        "moe_top_k": moe["experts_per_token"],
        "non_expert_activation_format": quantization["non_expert_activation_format"],
        "non_expert_weight_format": quantization["non_expert_weight_format"],
        "output_vocab_count": model["vocabulary_size"],
        "qk_nope_head_dimension": mla["qk_nope_dimension"],
        "qk_unrotated_head_dimension": mla["qk_unrotated_dimension"],
        "query_a_dimension": mla["query_lora_rank"],
        "rms_norm_epsilon": model["rms_norm_epsilon"],
        "routed_expert_activation_format": quantization["routed_expert_activation_format"],
        "routed_expert_scale_format": quantization["routed_expert_scale_format"],
        "routed_expert_weight_format": quantization["routed_expert_weight_format"],
        "mxfp4_group_size": quantization["routed_expert_group_size"],
        "speculation": speculation,
        "total_parameters": model["total_parameters"],
        "value_head_dimension": mla["value_head_dimension"]
    }
    return json.dumps(contract, indent=2, sort_keys=True) + "\n"


def render_description(source: dict[str, Any]) -> str:
    model = source["model"]
    attention = source["hybrid_attention"]
    kda = source["kda"]
    mla = source["mla"]
    attnres = source["attnres"]
    moe = source["moe"]
    quantization = source["quantization"]
    description = {
        "schema_version": 1,
        "model": {
            "id": "moonshot.kimi-k3.resident-decode-stage-firmware",
            "revision": "mxfp4-routed-bf16-rest-h7168-l93-e896-k16-kda69-mla24-v2"
        },
        "metadata": {
            "architecture": "k3_resident_decode_stage",
            "purpose": "Kimi K3 text decode firmware with MXFP4 routed experts, BF16 non-expert tensors, FP32 accumulation and recurrent state",
            "generated_from": "model_contracts/k3_authoritative.json",
            "provenance": {
                "technical_report": source["sources"]["technical_report"],
                "checkpoint_config": source["sources"]["checkpoint_config"],
                "status": "source-integrated, CUDA and target-hardware qualification not measured"
            },
            "module_geometry": {
                "hidden_dimension": model["hidden_dimension"],
                "layer_count": model["layer_count"],
                "kda_layer_count": attention["kda_layer_count"],
                "mla_layer_count": attention["mla_layer_count"],
                "kda_head_count": kda["head_count"],
                "kda_head_key_dimension": kda["key_dimension"],
                "kda_head_value_dimension": kda["value_dimension"],
                "mla_head_count": mla["query_head_count"],
                "mla_query_a_dimension": mla["query_lora_rank"],
                "mla_latent_dimension": mla["kv_lora_rank"],
                "mla_qk_nope_head_dimension": mla["qk_nope_dimension"],
                "mla_unrotated_dimension": mla["qk_unrotated_dimension"],
                "mla_value_head_dimension": mla["value_head_dimension"],
                "moe_expert_count": moe["routed_expert_count"],
                "moe_top_k": moe["experts_per_token"],
                "moe_shared_expert_count": moe["shared_expert_count"],
                "moe_latent_dimension": moe["latent_dimension"],
                "moe_intermediate_dimension": moe["expert_intermediate_dimension"],
                "dense_intermediate_dimension": moe["dense_intermediate_dimension"],
                "attnres_block_layers": attnres["block_size_layers"],
                "attnres_block_count": attnres["layer_block_count"],
                "attnres_max_representations": attnres["maximum_candidate_representations"],
                "vocab_count": model["vocabulary_size"]
            },
            "precision_contract": {
                "routed_expert_weights": quantization["routed_expert_weight_format"],
                "routed_expert_scales": quantization["routed_expert_scale_format"],
                "routed_expert_group_size": quantization["routed_expert_group_size"],
                "routed_expert_activations": quantization["routed_expert_activation_format"],
                "non_expert_weights": quantization["non_expert_weight_format"],
                "non_expert_activations": quantization["non_expert_activation_format"],
                "accumulator": quantization["accumulator_format"],
                "kda_state": kda["state_element_type"]
            },
            "qualification": {
                "status": "NOT_MEASURED",
                "production_ready": False,
                "cuda13_sm121a_compile": "NOT_RUN",
                "gpu_numerical_correctness": "NOT_MEASURED",
                "gpu_latency": "NOT_MEASURED",
                "gpu_throughput": "NOT_MEASURED"
            }
        },
        "stages": [
            {
                "name": "k3_resident_decode_stage",
                "target": "cuda.sm121.k3.resident_decode_stage.mxfp4_routed_bf16_rest",
                "programs": [
                    {
                        "name": "resident_decode",
                        "id": 1,
                        "max_inflight": 1,
                        "completion": "submit_return",
                        "scheduling": {
                            "flags": [
                                "stream_ordered",
                                "driver_owns_resident_state",
                                "driver_owns_hybrid_cache",
                                "fixed_firmware",
                                "requires_hidden_transport",
                                "no_file_transport",
                                "no_shell_transport"
                            ]
                        },
                        "operations": [
                            {
                                "name": "k3_resident_decode_stage",
                                "module": "spark.k3.resident_decode_stage.mxfp4_routed_bf16_rest.h7168.l93.kda69.mla24.v2",
                                "configuration": {
                                    "qualification_status": "NOT_MEASURED",
                                    "production_ready": False,
                                    "fallback_allowed": False,
                                    "runtime_backend_selection": "forbidden"
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return json.dumps(description, indent=2, sort_keys=False) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        return path.exists() and path.read_text(encoding="utf-8") == content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = load_source(args.source)
    outputs = {
        GENERATED_HEADER: render_header(source),
        GENERATED_CONTRACT: render_contract(source),
        GENERATED_DESCRIPTION: render_description(source)
    }
    stale = [str(path.relative_to(ROOT)) for path, content in outputs.items() if not write_or_check(path, content, args.check)]
    if stale:
        print("stale generated K3 contract files:")
        for path in stale:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
