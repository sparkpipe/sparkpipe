#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FLASH_DESCRIPTION_PATH = ROOT / "examples" / "model_descriptions" / "dsv4_resident_decode_stage_firmware.json"
FLASH_B1_DESCRIPTION_PATH = ROOT / "examples" / "model_descriptions" / "dsv4_resident_decode_stage_firmware_b1.json"
FLASH_DRIVER_MODEL_ID = "deepseek.v4.flash.resident-decode-stage-firmware"
FLASH_DRIVER_REVISION = "h4096-l43-dsa-e256k6-hash3-v129280-ga0731-v4"
FLASH_MODULE_ID_PREFIX = "spark.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16.h4096.l43.e256.k6.ga0731"
FLASH_MODULE_ID_SUFFIX = "v4"
FLASH_MODULE_ID = f"{FLASH_MODULE_ID_PREFIX}.{FLASH_MODULE_ID_SUFFIX}"
FLASH_MODULE_TARGET = "cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16"
CONTRACTS = {
    "flash": (
        ROOT / "model_contracts" / "dsv4_flash_authoritative.json",
        ROOT / "model-families" / "dsv4" / "include" / "sparkpipe" / "spark_dsv4_model.h",
        ROOT / "model_contracts" / "dsv4_flash.json",
    ),
    "pro": (
        ROOT / "model_contracts" / "dsv4_pro_authoritative.json",
        ROOT / "model-families" / "dsv4" / "include" / "sparkpipe" / "spark_dsv4_pro_model.h",
        ROOT / "model_contracts" / "dsv4_pro.json",
    ),
}
WEIGHT_CODEC_MACROS = {
    "bf16": "SPARK_WEIGHT_CODEC_BF16",
    "int6": "SPARK_WEIGHT_CODEC_INT6",
    "int7": "SPARK_WEIGHT_CODEC_INT7",
    "int8": "SPARK_WEIGHT_CODEC_INT8",
    "fp8_e4m3": "SPARK_WEIGHT_CODEC_FP8_E4M3",
    "nvfp4_e2m1": "SPARK_WEIGHT_CODEC_NVFP4_E2M1",
    "mxfp4_e2m1": "SPARK_WEIGHT_CODEC_MXFP4_E2M1",
}
ACTIVATION_CODEC_MACROS = {
    "bf16": "SPARK_ACTIVATION_CODEC_NONE",
    "fp8_e4m3": "SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0",
}
OUTPUT_ACTIVATION_CODEC_MACROS = {
    "bf16": "SPARK_ACTIVATION_CODEC_NONE",
    "fp8_e4m3": "SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0",
}


def require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description}: expected {expected!r}, got {actual!r}")


def activation_codec_macro(precision: dict[str, Any], field: str) -> str:
    activation_format = precision[field]
    if activation_format == "fp8_e4m3":
        require_equal(precision["scale_format"], "ue8m0", f"{field} scale format")
    return ACTIVATION_CODEC_MACROS[activation_format]


def activation_codec_name(precision: dict[str, Any], field: str) -> str:
    activation_format = precision[field]
    if activation_format == "fp8_e4m3":
        return f"{activation_format}_{precision['scale_format']}"
    return activation_format


def validate_flash_source_files(contract: dict[str, Any]) -> None:
    source_files = contract["source_files"]
    index_name = "model.safetensors.index.json"
    shard_names = [
        f"model-{index:05d}-of-00048.safetensors" for index in range(1, 49)
    ]
    require_equal(
        sorted(source_files), sorted([index_name] + shard_names),
        "Flash pinned source filenames")
    for name in [index_name] + shard_names:
        entry = source_files[name]
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
            raise ValueError(f"Flash source file byte count is invalid: {name}")
        digest = entry.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Flash source file SHA-256 is invalid: {name}")
    require_equal(source_files[index_name]["bytes"], 5602871, "Flash source index bytes")
    require_equal(
        source_files[index_name]["sha256"], contract["source_index_sha256"],
        "Flash source index file hash")
    require_equal(contract["source_shard_count"], len(shard_names), "Flash source shard count")
    require_equal(
        sum(source_files[name]["bytes"] for name in shard_names),
        contract["source_shard_bytes"], "Flash physical source shard bytes")
    require_equal(contract["source_shard_bytes"], 166886535336, "Flash source shard bytes")
    require_equal(
        contract["source_indexed_payload_bytes"], 166878536440,
        "Flash indexed source payload bytes")


def validate_contract(variant: str, contract: dict[str, Any]) -> None:
    model = contract["model"]
    attention = contract["attention"]
    hyper_connections = contract["hyper_connections"]
    moe = contract["moe"]
    precision = contract["precision"]
    ratios = attention["compression_ratios"]
    source_auxiliary_layer_count = contract.get("dspark", {}).get(
        "layer_count", model["mtp_layer_count"])

    require_equal(contract["schema_version"], 1, f"{variant} schema version")
    require_equal(contract["architecture"], "DeepseekV4ForCausalLM", f"{variant} architecture")
    expected_mtp_layer_count = 3 if variant == "flash" else 1
    packed_mtp_layer_count = contract.get("runtime", {}).get(
        "packed_mtp_layer_count", model["mtp_layer_count"])
    require_equal(model["mtp_layer_count"], expected_mtp_layer_count, f"{variant} MTP layer count")
    require_equal(len(ratios), model["layer_count"] + source_auxiliary_layer_count, f"{variant} compression-ratio count")
    require_equal(moe["experts_per_token"], 6, f"{variant} top-k")
    require_equal(moe["shared_expert_count"], 1, f"{variant} shared experts")
    require_equal(moe["hash_routed_layer_count"], 3, f"{variant} hash-routed layers")
    require_equal(hyper_connections["stream_count"], 4, f"{variant} hyper-connection streams")
    require_equal(precision["routed_expert_weight_codec"], "mxfp4_e2m1", f"{variant} expert codec")
    require_equal(precision["non_expert_linear_weight_codec"], "fp8_e4m3", f"{variant} non-expert codec")
    require_equal(precision["non_expert_linear_weight_format"], "fp8_e4m3_block_128x128", f"{variant} non-expert precision")
    require_equal(precision["kv_cache_codec"], "bf16", f"{variant} KV cache codec")
    expected_non_expert_activation = "bf16"
    require_equal(precision["non_expert_activation_format"], expected_non_expert_activation, f"{variant} non-expert activation format")
    require_equal(precision["routed_expert_activation_format"], "fp8_e4m3", f"{variant} routed-expert activation format")
    require_equal(precision["output_composition_activation_format"], "bf16", f"{variant} output-composition activation format")
    require_equal(precision["scale_format"], "ue8m0", f"{variant} activation scale format")
    require_equal(contract["qualification"]["cuda_target"], "sm_121a", f"{variant} CUDA target")
    require_equal(contract["qualification"]["production_ready"], False, f"{variant} readiness")
    if variant == "flash":
        require_equal(contract["model_id"], "deepseek-ai/DeepSeek-V4-Flash-0731", "Flash source model")
        require_equal(contract["source_revision"], "7872f01b1d1fe23eabc4c98b48bffcef5a386062", "Flash source revision")
        require_equal(packed_mtp_layer_count, 3, "Flash packed MTP layer count")
        require_equal(contract["runtime"]["speculative_decoding"], "enabled", "Flash speculative decoding")
        require_equal(contract["dspark"]["checkpoint_namespace"], "mtp", "Flash DSpark checkpoint namespace")
        require_equal(contract["dspark"]["layer_count"], 3, "Flash DSpark layer count")
        require_equal(contract["dspark"]["block_size"], 5, "Flash DSpark block size")
        require_equal(contract["dspark"]["noise_token_id"], 128799, "Flash DSpark noise token")
        require_equal(contract["dspark"]["target_layer_ids"], [40, 41, 42], "Flash DSpark target layers")
        require_equal(contract["dspark"]["markov_rank"], 256, "Flash DSpark Markov rank")
        require_equal(contract["dspark"]["tensor_counts"], [1568, 1565, 1572], "Flash DSpark tensor counts")
        require_equal(contract["dspark"]["packed"], True, "Flash DSpark packing")
        require_equal(contract["dspark"]["execution_supported"], True, "Flash DSpark execution")
        require_equal(contract["source_index_sha256"], "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b", "Flash source index hash")
        require_equal(contract["source_tensor_count"], 72317, "Flash source tensor count")
        validate_flash_source_files(contract)
        require_equal(model["hidden_dimension"], 4096, "Flash hidden dimension")
        require_equal(model["layer_count"], 43, "Flash layer count")
        require_equal(model["attention_head_count"], 64, "Flash attention heads")
        require_equal(moe["routed_expert_count"], 256, "Flash routed experts")
        require_equal(attention["index_top_k"], 512, "Flash index top-k")
        require_equal(ratios[:2], [0, 0], "Flash bootstrap attention layers")
        require_equal(ratios[-3:], [0, 0, 0], "Flash DSpark attention layers")
    elif variant == "pro":
        require_equal(model["hidden_dimension"], 7168, "Pro hidden dimension")
        require_equal(model["layer_count"], 61, "Pro layer count")
        require_equal(model["attention_head_count"], 128, "Pro attention heads")
        require_equal(moe["routed_expert_count"], 384, "Pro routed experts")
        require_equal(attention["index_top_k"], 1024, "Pro index top-k")
        require_equal(ratios[:2], [128, 128], "Pro bootstrap attention layers")
    else:
        raise ValueError(f"unknown DSV4 variant: {variant}")


def macro_prefix(variant: str) -> str:
    if variant == "flash":
        return "SPARK_DSV4_MODEL"
    return f"SPARK_DSV4_{variant.upper()}"


def c_float(value: float) -> str:
    if value == int(value):
        return f"{int(value)}.0f"
    return f"{value:.10g}f"


def render_header(
        variant: str, contract: dict[str, Any], description_sha256: str = "",
        b1_description_sha256: str = "",
        bucket_shas: dict | None = None) -> str:
    model = contract["model"]
    attention = contract["attention"]
    hyper_connections = contract["hyper_connections"]
    moe = contract["moe"]
    precision = contract["precision"]
    ratios = attention["compression_ratios"]
    packed_mtp_layer_count = contract.get("runtime", {}).get(
        "packed_mtp_layer_count", model["mtp_layer_count"])
    rendered_ratios = ratios[:model["layer_count"] + packed_mtp_layer_count]
    prefix = macro_prefix(variant)
    common_defines = [
        ("HIDDEN_DIMENSION", model["hidden_dimension"]),
        ("LAYER_COUNT", model["layer_count"]),
        ("MTP_LAYER_COUNT", model["mtp_layer_count"]),
        ("VOCAB_COUNT", model["vocabulary_size"]),
        ("QUERY_LORA_RANK", model["query_lora_rank"]),
        ("OUTPUT_LORA_RANK", model["output_lora_rank"]),
        ("OUTPUT_GROUP_COUNT", model["output_group_count"]),
        ("INDEX_HEAD_COUNT", attention["index_head_count"]),
        ("INDEX_HEAD_DIMENSION", attention["index_head_dimension"]),
        ("INDEX_TOP_K", attention["index_top_k"]),
        ("SLIDING_WINDOW_TOKENS", attention["sliding_window_tokens"]),
        ("YARN_FACTOR", attention["yarn_factor"]),
        ("ROUTED_EXPERT_COUNT", moe["routed_expert_count"]),
        ("SHARED_EXPERT_COUNT", moe["shared_expert_count"]),
        ("EXPERTS_PER_TOKEN", moe["experts_per_token"]),
        ("EXPERT_INTERMEDIATE_DIMENSION", moe["expert_intermediate_dimension"]),
        ("HASH_ROUTED_LAYER_COUNT", moe["hash_routed_layer_count"]),
    ]
    if variant == "flash":
        defines = common_defines + [
            ("CHECKPOINT_DSPARK_LAYER_COUNT", contract["dspark"]["layer_count"]),
            ("DSPARK_BLOCK_SIZE", contract["dspark"]["block_size"]),
            ("DSPARK_NOISE_TOKEN_ID", contract["dspark"]["noise_token_id"]),
            ("DSPARK_MARKOV_RANK", contract["dspark"]["markov_rank"]),
            ("DSPARK_TARGET_LAYER_COUNT", len(contract["dspark"]["target_layer_ids"])),
            ("DSPARK_TARGET_LAYER_FIRST", contract["dspark"]["target_layer_ids"][0]),
            ("MAX_POSITIONS", model["maximum_context_tokens"]),
            ("ATTN_QUERY_HEAD_COUNT", model["attention_head_count"]),
            ("ATTN_KV_HEAD_COUNT", model["kv_head_count"]),
            ("HEAD_DIMENSION", model["head_dimension"]),
            ("ATTN_ROPE_DIMENSION", model["qk_rope_head_dimension"]),
            ("ATTN_YARN_ORIGINAL_POSITIONS", attention["yarn_original_context_tokens"]),
            ("HC_STREAM_COUNT", hyper_connections["stream_count"]),
            ("HC_SINKHORN_ITERATIONS", hyper_connections["sinkhorn_iterations"]),
        ]
    else:
        defines = common_defines + [
            ("MAXIMUM_CONTEXT_TOKENS", model["maximum_context_tokens"]),
            ("ATTENTION_HEAD_COUNT", model["attention_head_count"]),
            ("KV_HEAD_COUNT", model["kv_head_count"]),
            ("HEAD_DIMENSION", model["head_dimension"]),
            ("QK_ROPE_HEAD_DIMENSION", model["qk_rope_head_dimension"]),
            ("YARN_ORIGINAL_CONTEXT_TOKENS", attention["yarn_original_context_tokens"]),
            ("HYPER_CONNECTION_STREAM_COUNT", hyper_connections["stream_count"]),
            ("HYPER_CONNECTION_SINKHORN_ITERATIONS", hyper_connections["sinkhorn_iterations"]),
        ]

    lines = ["#pragma once", ""]
    if variant == "flash":
        lines.extend([
            "/*",
            " * Pro builds define SPARK_DSV4_PRO_BUILD and get the Pro geometry through",
            " * the model-generic name space; Flash builds are unchanged by this guard.",
            " */",
            "#if defined(SPARK_DSV4_PRO_BUILD)",
            '#include "sparkpipe/spark_dsv4_pro_model_aliases.h"',
            "#else",
            "",
        ])
    lines.extend([
        "#include <stdint.h>",
        "",
        "#include \"sparkpipe/spark_weight_codec.h\"",
        "",
        "/* Generated from the exact source revision by tools/generate_dsv4_contracts.py. */",
        f"#define {prefix}_ID {json.dumps(contract['model_id'])}",
        f"#define {prefix}_SOURCE_REVISION {json.dumps(contract['source_revision'])}",
        "",
    ])
    for suffix, value in defines:
        if variant == "flash" and suffix == "DSPARK_TARGET_LAYER_FIRST":
            lines.append(f"#define {prefix}_DSPARK_TARGET_LAYER_FIRST \\")
            lines.append(
                f"\t({prefix}_LAYER_COUNT - {prefix}_DSPARK_TARGET_LAYER_COUNT)")
            continue
        lines.append(f"#define {prefix}_{suffix} {value}u")
    if variant == "flash":
        lines.extend([
            f"#define {prefix}_DRIVER_MODEL_ID {json.dumps(FLASH_DRIVER_MODEL_ID)}",
            f"#define {prefix}_DRIVER_REVISION {json.dumps(FLASH_DRIVER_REVISION)}",
            f"#define {prefix}_DESCRIPTION_SHA256 {json.dumps(description_sha256)}",
            f"#define {prefix}_DESCRIPTION_SHA256_B1 {json.dumps(b1_description_sha256)}",
            f"#define {prefix}_MODULE_ID {json.dumps(FLASH_MODULE_ID)}",
            f"#define {prefix}_MODULE_TARGET {json.dumps(FLASH_MODULE_TARGET)}",
            f"#define {prefix}_ATTN_HEAD_DIMENSION {prefix}_HEAD_DIMENSION",
            f"#define {prefix}_ATTN_ROPE_THETA {c_float(attention['rope_theta'])}",
            f"#define {prefix}_COMPRESS_ROPE_THETA {c_float(attention['compressed_rope_theta'])}",
            f"#define {prefix}_ATTN_YARN_FACTOR {attention['yarn_factor']}u",
            f"#define {prefix}_RMS_NORM_EPSILON {c_float(model['rms_norm_epsilon'])}",
            f"#define {prefix}_HC_EPSILON {c_float(hyper_connections['epsilon'])}",
            f"#define {prefix}_ROUTED_SCALING_FACTOR {c_float(moe['routed_scaling_factor'])}",
            f"#define {prefix}_SWIGLU_LIMIT {c_float(moe['swiglu_limit'])}",
            f"#define {prefix}_EXPERT_WEIGHT_CODEC {WEIGHT_CODEC_MACROS[precision['routed_expert_weight_codec']]}",
            f"#define {prefix}_NON_EXPERT_WEIGHT_CODEC {WEIGHT_CODEC_MACROS[precision['non_expert_linear_weight_codec']]}",
            f"#define {prefix}_KV_CACHE_CODEC {WEIGHT_CODEC_MACROS[precision['kv_cache_codec']]}",
            f"#define {prefix}_NON_EXPERT_ACTIVATION_CODEC {activation_codec_macro(precision, 'non_expert_activation_format')}",
            f"#define {prefix}_EXPERT_ACTIVATION_CODEC {activation_codec_macro(precision, 'routed_expert_activation_format')}",
            f"#define {prefix}_OUTPUT_COMPOSITION_ACTIVATION_CODEC {OUTPUT_ACTIVATION_CODEC_MACROS[precision['output_composition_activation_format']]}",
            f"#define {prefix}_CSA_COMPRESS_RATIO 4u",
            f"#define {prefix}_HCA_COMPRESS_RATIO 128u",
            f"#define {prefix}_ROPE_BETA_FAST 32u",
            f"#define {prefix}_ROPE_BETA_SLOW 1u",
            f"#define {prefix}_KV_QUANT_BLOCK 64u",
            f"#define {prefix}_FP4_QUANT_BLOCK 32u",
            f"#define {prefix}_FP8_MAX 448.0f",
            f"#define {prefix}_FP4_MAX 6.0f",
            f"#define {prefix}_CSA_OVERLAP_FACTOR 2u",
            f"#define {prefix}_BF16_ELEMENT_BYTES 2u",
            f"#define {prefix}_HC_MIX_ROWS ((2u + {prefix}_HC_STREAM_COUNT) * {prefix}_HC_STREAM_COUNT)",
            f"#define {prefix}_ATTN_QUERY_DIMENSION ({prefix}_ATTN_QUERY_HEAD_COUNT * {prefix}_ATTN_HEAD_DIMENSION)",
            f"#define {prefix}_OUTPUT_GROUP_DIMENSION ({prefix}_ATTN_QUERY_DIMENSION / {prefix}_OUTPUT_GROUP_COUNT)",
            f"#define {prefix}_INDEX_DIMENSION ({prefix}_INDEX_HEAD_COUNT * {prefix}_INDEX_HEAD_DIMENSION)",
            f"#define {prefix}_BOUNDARY_STREAM_ELEMENTS ({prefix}_HC_STREAM_COUNT * {prefix}_HIDDEN_DIMENSION)",
            f"#define {prefix}_LAYER_KIND_SWA 0u",
            f"#define {prefix}_LAYER_KIND_CSA 1u",
            f"#define {prefix}_LAYER_KIND_HCA 2u",
            f"#define {prefix}_LAYER_KIND_INVALID UINT32_MAX",
            "",
        ])
        if bucket_shas is not None:
            for bucket in sorted(bucket_shas.keys()):
                lines.append(
                    f"#define {prefix}_DESCRIPTION_SHA256_B{bucket} {json.dumps(bucket_shas[bucket])}")
    else:
        lines.extend([
            f"#define {prefix}_RMS_NORM_EPSILON {c_float(model['rms_norm_epsilon'])}",
            f"#define {prefix}_ROPE_THETA {c_float(attention['rope_theta'])}",
            f"#define {prefix}_COMPRESSED_ROPE_THETA {c_float(attention['compressed_rope_theta'])}",
            f"#define {prefix}_HYPER_CONNECTION_EPSILON {c_float(hyper_connections['epsilon'])}",
            f"#define {prefix}_ROUTED_SCALING_FACTOR {c_float(moe['routed_scaling_factor'])}",
            f"#define {prefix}_SWIGLU_LIMIT {c_float(moe['swiglu_limit'])}",
            "#if defined(SPARK_DSV4_PRO_EXPERT_CODEC_FP8_E4M3)",
            "/* Variant builds: FP8-E4M3 expert weights (requires the FP8 expert kernel",
            " * variant and an FP8-expert pack; default remains MXFP4-E2M1). */",
            "#define SPARK_DSV4_PRO_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3",
            "#else",
            "#define SPARK_DSV4_PRO_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_MXFP4_E2M1",
            "#endif",
            f"#define {prefix}_NON_EXPERT_WEIGHT_CODEC {WEIGHT_CODEC_MACROS[precision['non_expert_linear_weight_codec']]}",
            f"#define {prefix}_KV_CACHE_CODEC {WEIGHT_CODEC_MACROS[precision['kv_cache_codec']]}",
            f"#define {prefix}_NON_EXPERT_ACTIVATION_CODEC {activation_codec_macro(precision, 'non_expert_activation_format')}"
            + (" /* first-light: BF16 activations, matching the Flash-validated kernel set */"
               if precision["non_expert_activation_format"] == "bf16" else ""),
            f"#define {prefix}_EXPERT_ACTIVATION_CODEC {activation_codec_macro(precision, 'routed_expert_activation_format')}",
            f"#define {prefix}_OUTPUT_COMPOSITION_ACTIVATION_CODEC {OUTPUT_ACTIVATION_CODEC_MACROS[precision['output_composition_activation_format']]}",
            "",
        ])
    array_name = "SparkDsv4ModelCompressionRatios" if variant == "flash" else f"SparkDsv4{variant.title()}CompressionRatios"
    lines.extend([f"static const uint16_t {array_name}[{len(rendered_ratios)}u] =", "{"])
    for offset in range(0, len(rendered_ratios), 12):
        group = rendered_ratios[offset:offset + 12]
        lines.append("    " + ", ".join(f"{value}u" for value in group) + ("," if offset + 12 < len(rendered_ratios) else ""))
    function_prefix = "SparkDsv4Model" if variant == "flash" else f"SparkDsv4{variant.title()}"
    lines.extend([
        "};",
        "",
        f"static inline uint16_t {function_prefix}BackboneCompressionRatio(uint32_t layer_index)",
        "{",
        f"\tif ( layer_index >= {prefix}_LAYER_COUNT )",
        "\t\treturn(UINT16_MAX);",
        f"\treturn({array_name}[layer_index]);",
        "}",
        "",
    ])
    lines.extend([
        f"static inline uint16_t {function_prefix}MtpCompressionRatio(void)",
        "{",
        ("\treturn(UINT16_MAX);" if packed_mtp_layer_count == 0 else
         f"\treturn({array_name}[{prefix}_LAYER_COUNT]);"),
        "}",
        "",
    ])
    if variant == "flash":
        lines.extend([
            f"#define {prefix}_MTP_LAYER_KIND {prefix}_LAYER_KIND_SWA",
            "",
            "static inline uint32_t SparkDsv4ModelLayerKind(uint32_t layer_index)",
            "{",
            "\tuint16_t ratio;",
            "\tratio = SparkDsv4ModelBackboneCompressionRatio(layer_index);",
            "\tif ( ratio == 0u )",
            f"\t\treturn({prefix}_LAYER_KIND_SWA);",
            f"\tif ( ratio == {prefix}_CSA_COMPRESS_RATIO )",
            f"\t\treturn({prefix}_LAYER_KIND_CSA);",
            f"\tif ( ratio == {prefix}_HCA_COMPRESS_RATIO )",
            f"\t\treturn({prefix}_LAYER_KIND_HCA);",
            f"\treturn({prefix}_LAYER_KIND_INVALID);",
            "}",
            "",
            "#endif /* SPARK_DSV4_PRO_BUILD */",
            "",
        ])
    return "\n".join(lines)


def render_normalized_contract(variant: str, contract: dict[str, Any]) -> str:
    result = dict(contract)
    header_name = "spark_dsv4_model.h" if variant == "flash" else f"spark_dsv4_{variant}_model.h"
    result["generated"] = {
        "variant": variant,
        "header": f"model-families/dsv4/include/sparkpipe/{header_name}",
        "compression_ratio_interpretation": (
            "first layer_count entries are backbone layers; the following dspark.layer_count entries are checkpoint DSpark layers excluded from runtime"
            if variant == "flash" else
            "first layer_count entries are backbone layers; final entry is the one declared MTP layer"
        ),
    }
    if variant == "pro":
        note = result["precision"].pop("_first_light_note")
        body = json.dumps(result, indent=2, sort_keys=True)
        anchor = '    "scale_format": "ue8m0"\n  },\n  "qualification": {'
        replacement = (
            '    "scale_format": "ue8m0",\n'
            f'    "_first_light_note": {json.dumps(note)}\n'
            '  },\n  "qualification": {')
        if anchor not in body:
            raise ValueError("pro normalized contract anchor not found")
        return body.replace(anchor, replacement)
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def flash_module_id(batch_bucket: int) -> str:
    if batch_bucket == 1024:
        return FLASH_MODULE_ID
    if batch_bucket not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        raise ValueError(f"invalid DSV4 batch bucket: {batch_bucket}")
    return f"{FLASH_MODULE_ID_PREFIX}.b{batch_bucket}.{FLASH_MODULE_ID_SUFFIX}"


def render_flash_model_description(
        contract: dict[str, Any], batch_bucket: int = 1024) -> str:
    model = contract["model"]
    attention = contract["attention"]
    moe = contract["moe"]
    precision = contract["precision"]
    description = {
        "schema_version": 1,
        "model": {
            "id": FLASH_DRIVER_MODEL_ID,
            "revision": FLASH_DRIVER_REVISION,
        },
        "metadata": {
            "architecture": "dsv4_resident_decode_stage",
            "source_model": {
                "id": contract["model_id"],
                "revision": contract["source_revision"],
            },
            "purpose": (
                "DeepSeek V4 Flash resident firmware with checkpoint FP8 "
                "non-expert weights, MXFP4 routed-expert weights, BF16 KV "
                "cache and FP32 accumulation"
            ),
            "qualification": {
                "status": "NOT_MEASURED",
                "production_ready": False,
                "gpu_numerical_correctness": "NOT_MEASURED",
                "gpu_latency": "NOT_MEASURED",
                "gpu_throughput": "NOT_MEASURED",
            },
            "precision_contract": {
                "linear_weight_codec": precision["non_expert_linear_weight_codec"],
                "expert_weight_codec": precision["routed_expert_weight_codec"],
                "expert_weight_codec_id": 7,
                "expert_stored_bits": 4,
                "expert_scale_encoding": "e8m0",
                "expert_scale_group_size": 32,
                "activation_codec": activation_codec_name(precision, "non_expert_activation_format"),
                "non_expert_activation_codec": activation_codec_name(precision, "non_expert_activation_format"),
                "output_composition_activation_codec": precision["output_composition_activation_format"],
                "expert_activation_codec": activation_codec_name(precision, "routed_expert_activation_format"),
                "kv_cache_codec": precision["kv_cache_codec"],
                "accumulator_codec": "fp32",
                "aot_codec_specialization": True,
                "runtime_precision_selection": "forbidden",
                "fallback_allowed": False,
            },
            "module_geometry": {
                "hidden_dimension": model["hidden_dimension"],
                "layer_count": model["layer_count"],
                "attention_head_count": model["attention_head_count"],
                "head_dimension": model["head_dimension"],
                "sliding_window_tokens": attention["sliding_window_tokens"],
                "index_head_count": attention["index_head_count"],
                "index_head_dimension": attention["index_head_dimension"],
                "index_top_k": attention["index_top_k"],
                "routed_expert_count": moe["routed_expert_count"],
                "experts_per_token": moe["experts_per_token"],
                "expert_intermediate_dimension": moe["expert_intermediate_dimension"],
                "vocab_count": model["vocabulary_size"],
                "checkpoint_dspark_layer_count": contract["dspark"]["layer_count"],
                "packed_dspark_layer_count": contract["runtime"]["packed_mtp_layer_count"],
            },
            "runtime_contract": {
                "required_environment": [],
                "configuration_source": "typed model-owned serving adapter configuration",
                "runtime_backend_selection": "forbidden",
                "runtime_precision_selection": "forbidden",
                "fallback_allowed": False,
                "completion": "external",
                "speculative_decoding": contract["runtime"]["speculative_decoding"],
                "speculative_decoding_reason": contract["runtime"]["speculative_decoding_reason"],
            },
        },
        "stages": [{
            "name": "dsv4_resident_decode_stage",
            "target": FLASH_MODULE_TARGET,
            "programs": [{
                "name": "resident_decode",
                "id": 1,
                "max_inflight": min(13, batch_bucket),
                "completion": "external",
                "operations": [{
                    "name": "dsv4_resident_decode_stage",
                    "module": flash_module_id(batch_bucket),
                    "configuration": {
                        "linear_weight_codec": precision["non_expert_linear_weight_codec"],
                        "expert_weight_codec": precision["routed_expert_weight_codec"],
                        "activation_codec": activation_codec_name(precision, "non_expert_activation_format"),
                        "expert_activation_codec": activation_codec_name(precision, "routed_expert_activation_format"),
                        "output_composition_activation_codec": precision["output_composition_activation_format"],
                        "kv_cache_codec": precision["kv_cache_codec"],
                        "runtime_backend_selection": "forbidden",
                        "runtime_precision_selection": "forbidden",
                        "fallback_allowed": False,
                    },
                }],
                "scheduling": {
                    "flags": [
                        "stream_ordered",
                        "driver_owns_resident_state",
                        "driver_owns_kv_cache",
                        "fixed_firmware",
                        "requires_hidden_transport",
                        "no_file_transport",
                        "no_shell_transport",
                    ],
                    "max_active_slots": batch_bucket,
                    "max_new_tokens": 65536,
                    "max_resident_sequences": (
                        16384 if batch_bucket == 1024 else batch_bucket),
                    "max_sequence_tokens": model["maximum_context_tokens"],
                    "target_latency_ns": 0,
                    "validated_latency_ns": 0,
                    "resident_weight_bytes": 0,
                    "resident_kv_bytes": 0,
                    "static_workspace_bytes": 0,
                    "device_memcpy_bytes_per_submit_ceiling": 0,
                    "host_staging_bytes_per_submit_ceiling": 0,
                    "private_queue_count": 0,
                },
            }],
        }],
    }
    return json.dumps(description, indent=2, sort_keys=True) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        return path.exists() and path.read_text(encoding="utf-8") == content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale: list[str] = []
    for variant, (source_path, header_path, normalized_path) in CONTRACTS.items():
        contract = json.loads(source_path.read_text(encoding="utf-8"))
        validate_contract(variant, contract)
        description = render_flash_model_description(contract) if variant == "flash" else ""
        b1_description = (
            render_flash_model_description(contract, 1)
            if variant == "flash" else "")
        description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
        b1_description_sha256 = hashlib.sha256(
            b1_description.encode("utf-8")).hexdigest()
        bucket_shas: dict[int, str] = {}
        if variant == "flash":
            for bucket in (8, 16, 32, 64):
                bucket_description = render_flash_model_description(
                    contract, bucket)
                bucket_shas[bucket] = hashlib.sha256(
                    bucket_description.encode("utf-8")).hexdigest()
        outputs = {
            header_path: render_header(
                variant, contract, description_sha256,
                b1_description_sha256, bucket_shas),
            normalized_path: render_normalized_contract(variant, contract),
        }
        if variant == "flash":
            for bucket in (8, 16, 32, 64):
                outputs[ROOT / "examples" / "model_descriptions" /
                    f"dsv4_resident_decode_stage_firmware_b{bucket}.json"] = (
                        render_flash_model_description(contract, bucket))
        if variant == "flash":
            outputs[FLASH_DESCRIPTION_PATH] = description
            outputs[FLASH_B1_DESCRIPTION_PATH] = b1_description
        for path, content in outputs.items():
            if not write_or_check(path, content, args.check):
                stale.append(str(path.relative_to(ROOT)))
    if stale:
        print("stale generated DSV4 contract files:")
        for path in stale:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
