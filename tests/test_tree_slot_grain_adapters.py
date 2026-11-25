#!/usr/bin/env python3
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CC = os.environ.get("CC", "cc")
SHA = "0" * 64

ADAPTERS = (
    (
        "kexaone2",
        "modules/kexaone2_tp16_resident_decode_stage/source/spark_kexaone2_tp16_stage_serving.c",
        ["modules/kexaone2_tp16_resident_decode_stage/include"],
        [f'-DKEXAONE2_TP16_CONTRACT_SHA256="{SHA}"'],
        "SparkKexaone2Tp16StageModel",
        16,
        1,
    ),
    (
        "gemma4_26b",
        "modules/gemma4_26b_resident_decode_stage/source/spark_gemma4_26b_stage_serving.c",
        ["modules/gemma4_26b_resident_decode_stage/include"],
        [f'-DGEMMA4_26B_CONTRACT_SHA256="{SHA}"'],
        "SparkGemma4_26bStageModel",
        16,
        1,
    ),
    (
        "gemma4_31b",
        "modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_serving.c",
        ["modules/gemma4_31b_resident_decode_stage/include"],
        [f'-DGEMMA4_31B_CONTRACT_SHA256="{SHA}"'],
        "SparkGemma4_31bStageModel",
        16,
        1,
    ),
    (
        "muse_glimmer",
        "modules/muse_glimmer_tp16_resident_decode_stage/source/spark_muse_tp16_stage_serving.c",
        ["modules/muse_glimmer_tp16_resident_decode_stage/include"],
        [f'-DMUSE_TP16_CONTRACT_SHA256="{SHA}"'],
        "SparkMuseTp16StageModel",
        16,
        1,
    ),
    (
        "laguna",
        "modules/laguna_tp16_resident_decode_stage/source/spark_laguna_tp16_stage_serving.c",
        ["modules/laguna_tp16_resident_decode_stage/include"],
        [f'-DLAGUNA_TP16_CONTRACT_SHA256="{SHA}"'],
        "SparkLagunaTp16StageModel",
        16,
        1,
    ),
    (
        "hy4",
        "modules/hy4_tp16_resident_decode_stage/source/spark_hy4_tp16_stage_serving.c",
        ["modules/hy4_tp16_resident_decode_stage/include"],
        [f'-DHY4_TP16_CONTRACT_SHA256="{SHA}"'],
        "SparkHy4Tp16StageModel",
        16,
        1,
    ),
    (
        "hy4_fp8",
        "modules/hy4_tp16_resident_decode_stage/source/spark_hy4_tp16_stage_serving.c",
        ["modules/hy4_tp16_resident_decode_stage/include"],
        [f'-DHY4_TP16_CONTRACT_SHA256="{SHA}"', "-DHY4_FP8=1"],
        "SparkHy4Tp16StageModel",
        16,
        1,
    ),
    (
        "mimo26_pro",
        "modules/mimo26_pro_tp16_resident_decode_stage/source/spark_mimo26_pro_tp16_stage_serving.c",
        ["modules/mimo26_pro_tp16_resident_decode_stage/include"],
        [f'-DMIMO26_PRO_TP16_CONTRACT_SHA256="{SHA}"'],
        "SparkMimo26ProTp16StageModel",
        16,
        1,
    ),
    (
        "mimo26_flash",
        "modules/mimo26_flash_tp16_resident_decode_stage/source/spark_mimo26_flash_tp16_stage_serving.c",
        ["modules/mimo26_flash_tp16_resident_decode_stage/include"],
        [f'-DMIMO26_FLASH_TP16_CONTRACT_SHA256="{SHA}"'],
        "SparkMimo26FlashTp16StageModel",
        16,
        1,
    ),
    (
        "glm53_full",
        "modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_serving.c",
        ["modules/glm53_full_tp16_resident_decode_stage/include", "model-families/glm52/include"],
        [
            f'-DGLM53_FULL_CONTRACT_SHA256="{SHA}"',
            '-DGLM53_FULL_MODEL_REVISION="test"',
            "-DGLM53_FULL_LINEAR_CODEC=SPARK_WEIGHT_CODEC_FP8_E4M3",
            "-DGLM53_FULL_EXPERT_CODEC=SPARK_WEIGHT_CODEC_FP8_E4M3",
            '-DGLM53_FULL_ADAPTER_ID="glm53_full-tp16"',
            '-DGLM53_FULL_MODEL_ID="zai-org/GLM-5.3"',
            '-DGLM53_FULL_MODULE_TAG="glm53_full"',
        ],
        "SparkGlm53FullTp16StageModel",
        16,
        1,
    ),
    (
        "axk2",
        "modules/axk2_tp16_resident_decode_stage/source/spark_axk2_tp16_stage_serving.c",
        ["modules/axk2_tp16_resident_decode_stage/include", "model-families/glm52/include"],
        [f'-DAXK2_CONTRACT_SHA256="{SHA}"', '-DAXK2_MODEL_REVISION="test"'],
        "SparkAxk2Tp16StageModel",
        16,
        1,
    ),
    (
        "glm53_flash",
        "modules/glm53_flash_tp16_resident_decode_stage/source/spark_glm53_flash_tp16_stage_serving.c",
        ["modules/glm53_flash_tp16_resident_decode_stage/include", "model-families/glm5_next/include"],
        [
            f'-DGLM53_FLASH_TP16_CONTRACT_SHA256="{SHA}"',
            "-DGLM53_FLASH_EXPERT_CODEC=SPARK_WEIGHT_CODEC_FP8_E4M3",
            '-DGLM53_FLASH_ADAPTER_ID="glm53_flash-tp16"',
            '-DGLM53_FLASH_MODULE_TAG="glm53_flash"',
            '-DGLM53_FLASH_MODEL_ID="zai-org/GLM-5.3-Flash"',
            '-DGLM53_FLASH_MODEL_REVISION="test"',
            '-DGLM53_FLASH_PROGRAM="glm53_flash"',
        ],
        "SparkGlm53FlashTp16StageModel",
        64,
        4,
    ),
    (
        "sage21_flash",
        "modules/glm53_flash_tp16_resident_decode_stage/source/spark_glm53_flash_tp16_stage_serving.c",
        ["modules/glm53_flash_tp16_resident_decode_stage/include", "model-families/glm5_next/include"],
        [
            f'-DGLM53_FLASH_TP16_CONTRACT_SHA256="{SHA}"',
            "-DGLM53_FLASH_EXPERT_CODEC=SPARK_WEIGHT_CODEC_FP8_E4M3",
            '-DGLM53_FLASH_ADAPTER_ID="sage21_flash-tp16"',
            '-DGLM53_FLASH_MODULE_TAG="sage21_flash_stage"',
            '-DGLM53_FLASH_MODEL_ID="internlm/Sage-S2.1-Flash-FP8"',
            '-DGLM53_FLASH_MODEL_REVISION="716e6436905650b7be93e1b7dedfb0bb95afd3a3"',
            "-DSAGE21_FLASH=1",
            '-DGLM53_FLASH_PROGRAM="sage21_flash"',
        ],
        "SparkGlm53FlashTp16StageModel",
        64,
        4,
    ),
    (
        "qwen38_flash",
        "modules/qwen38_flash_resident_decode_stage/source/spark_qwen38_flash_stage_serving.c",
        ["modules/qwen38_flash_resident_decode_stage/include"],
        [
            f'-DQWEN38_FLASH_CONTRACT_SHA256="{SHA}"',
            "-DQWEN38_FLASH_EXPERT_CODEC=SPARK_WEIGHT_CODEC_NVFP4_E2M1",
            '-DQWEN38_FLASH_ADAPTER_ID="qwen38_flash-tp16"',
            '-DQWEN38_FLASH_MODULE_TAG="qwen38_flash"',
        ],
        "SparkQwen38FlashStageModel",
        0,
        0,
    ),
    (
        "ling3",
        "modules/ling3_resident_decode_stage/source/spark_ling3_stage_serving.c",
        ["modules/ling3_resident_decode_stage/include", "model-families/ling/include"],
        [f'-DLING3_CONTRACT_SHA256="{SHA}"'],
        "SparkLing3StageModel",
        0,
        0,
    ),
    (
        "ling31_flash",
        "modules/ling3_resident_decode_stage/source/spark_ling3_stage_serving.c",
        ["modules/ling3_resident_decode_stage/include", "model-families/ling/include"],
        [f'-DLING3_CONTRACT_SHA256="{SHA}"', "-DLING31_FLASH=1"],
        "SparkLing3StageModel",
        0,
        0,
    ),
    (
        "ling32_air",
        "modules/ling3_resident_decode_stage/source/spark_ling3_stage_serving.c",
        ["modules/ling3_resident_decode_stage/include", "model-families/ling/include"],
        [f'-DLING3_CONTRACT_SHA256="{SHA}"', "-DLING32_AIR=1"],
        "SparkLing3StageModel",
        0,
        0,
    ),
    (
        "motif3",
        "modules/motif3_tp16_resident_decode_stage/source/spark_motif3_tp16_stage_serving.c",
        ["modules/motif3_tp16_resident_decode_stage/include", "model-families/motif/include"],
        [f'-DMOTIF3_TP16_CONTRACT_SHA256="{SHA}"'],
        "SparkMotif3Tp16StageModel",
        0,
        0,
    ),
)

HARNESS = """
#include <stdio.h>
#include <string.h>
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void);

const SparkStageRunnerModelInterface *RUNNER_SYMBOL(void)
{
	return 0;
}

static SparkModelServingAdapterDescriptor captured_descriptor;
static SparkModelServingAdapterInterface captured_interface;
static const SparkStageServingModel *captured_model;

static SparkStatus DummyInitialize(const SparkModelServingAdapterConfiguration *c, void **s) { (void)c; (void)s; return SPARK_STATUS_OK; }
static void DummyDestroy(void *s) { (void)s; }
static SparkStatus DummyValidate(void *s, const SparkModelServingSubmission *sub) { (void)s; (void)sub; return SPARK_STATUS_OK; }
static SparkStatus DummySubmit(void *s, const SparkModelServingSubmission *sub) { (void)s; (void)sub; return SPARK_STATUS_OK; }
static SparkStatus DummyPrefetch(void *s, const SparkModelServingSubmission *subs, uint32_t n) { (void)s; (void)subs; (void)n; return SPARK_STATUS_OK; }
static SparkStatus DummyResolve(void *s, const SparkModelServingSubmission *sub, uint32_t r) { (void)s; (void)sub; (void)r; return SPARK_STATUS_OK; }
static SparkStatus DummyProgress(void *s, uint32_t n) { (void)s; (void)n; return SPARK_STATUS_OK; }
static SparkStatus DummyQuiesce(void *s, uint64_t t) { (void)s; (void)t; return SPARK_STATUS_OK; }
static SparkStatus DummySnapshot(void *s, SparkModelServingAdapterSnapshot *st) { (void)s; (void)st; return SPARK_STATUS_OK; }
static SparkStatus DummyReset(void *s, uint64_t g) { (void)s; (void)g; return SPARK_STATUS_OK; }
static SparkStatus DummyCacheHint(void *s, const SparkModelServingCacheIdentity *id, uint32_t n) { (void)s; (void)id; (void)n; return SPARK_STATUS_OK; }
static SparkStatus DummyCacheProbe(void *s, const SparkModelServingCacheIdentity *id, uint32_t n, SparkModelServingCacheProbe *p) { (void)s; (void)id; (void)n; (void)p; return SPARK_STATUS_OK; }

const SparkModelServingAdapterInterface *SparkStageServingAdapterInterface(const SparkStageServingModel *model)
{
	uint32_t stage;
	captured_model = model;
	memset(&captured_descriptor, 0, sizeof(captured_descriptor));
	captured_descriptor.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	captured_descriptor.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	captured_descriptor.adapter_id = model->adapter_id;
	captured_descriptor.model_id = model->model_id;
	captured_descriptor.model_revision = model->model_revision;
	captured_descriptor.driver_program_name = model->program;
	captured_descriptor.artifact_sha256 = model->contract_sha256;
	captured_descriptor.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS |
		(model->speculative_tokens != 0u ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION : 0u) |
		(model->verify_rows != 0u ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATIVE_VERIFY : 0u) |
		(model->tree_slot_grain != 0u ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_TREE_VERIFY : 0u);
	captured_descriptor.stage_count = model->tp_degree;
	captured_descriptor.layer_count = model->layer_count;
	captured_descriptor.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	captured_descriptor.boundary_element_count = model->hidden;
	captured_descriptor.boundary_element_bytes = 2u;
	captured_descriptor.linear_weight_codec = model->linear_weight_codec;
	captured_descriptor.expert_weight_codec = model->expert_weight_codec;
	captured_descriptor.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	captured_descriptor.max_inflight_submission_count = 16u;
	captured_descriptor.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES;
	captured_descriptor.max_input_row_count = model->max_rows;
	captured_descriptor.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES;
	captured_descriptor.max_output_token_count = SPARK_STAGE_SERVING_MAX_LANES * SPARK_STAGE_SERVING_CHAIN_TOKENS;
	captured_descriptor.max_speculative_token_count = model->speculative_tokens;
	for ( stage = 0u; stage < model->tp_degree; ++stage )
		captured_descriptor.stage_layer_counts[stage] = model->layer_count;
	captured_descriptor.minimum_efficient_submission_row_count = 1u;
	captured_descriptor.cache_block_token_count = model->cache_block_token_count;
	captured_descriptor.parallel_group_size = 0u;
	captured_descriptor.cache_checkpoint_token_count = model->cache_checkpoint_token_count;
	captured_descriptor.max_verify_row_count = model->verify_rows;
	captured_descriptor.tree_slot_grain = model->tree_slot_grain;
	memset(&captured_interface, 0, sizeof(captured_interface));
	captured_interface.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	captured_interface.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES;
	captured_interface.descriptor = &captured_descriptor;
	captured_interface.initialize = DummyInitialize;
	captured_interface.destroy = DummyDestroy;
	captured_interface.validate_submission = DummyValidate;
	captured_interface.submit = DummySubmit;
	captured_interface.prefetch = DummyPrefetch;
	captured_interface.resolve_prefetch = DummyResolve;
	captured_interface.progress = DummyProgress;
	captured_interface.quiesce = DummyQuiesce;
	captured_interface.snapshot = DummySnapshot;
	captured_interface.reset = DummyReset;
	captured_interface.cache_hint = DummyCacheHint;
	captured_interface.cache_probe = DummyCacheProbe;
	return &captured_interface;
}

int main(void)
{
	const SparkModelServingAdapterInterface *iface = SparkModelServingAdapterGetInterface();
	SparkStageKvConfiguration kv;
	uint32_t runner_grain;
	if ( iface == 0 || iface->descriptor == 0 || captured_model == 0 )
	{
		fprintf(stderr, "FAIL: null interface\\n");
		return 1;
	}
	if ( SparkModelServingAdapterValidateInterface(iface, 0u) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "FAIL: SparkModelServingAdapterValidateInterface rejected %s\\n", captured_model->adapter_id);
		return 1;
	}
	memset(&kv, 0, sizeof(kv));
	captured_model->kv_layout(0u, captured_model->tp_degree, &kv);
	runner_grain = (uint32_t)(EXPECTED_POOL_MULT) * kv.context_shard.degree * kv.context_shard.grain;
	if ( iface->descriptor->tree_slot_grain != (uint32_t)(EXPECTED_GRAIN) )
	{
		fprintf(stderr, "FAIL: %s tree_slot_grain=%u expected=%u\\n", captured_model->adapter_id, iface->descriptor->tree_slot_grain, (uint32_t)(EXPECTED_GRAIN));
		return 1;
	}
	if ( (uint32_t)(EXPECTED_GRAIN) != 0u && iface->descriptor->tree_slot_grain != runner_grain )
	{
		fprintf(stderr, "FAIL: %s tree_slot_grain=%u != runner_grain=%u\\n", captured_model->adapter_id, iface->descriptor->tree_slot_grain, runner_grain);
		return 1;
	}
	if ( ((iface->descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_TREE_VERIFY) != 0u) != ((uint32_t)(EXPECTED_GRAIN) != 0u) )
	{
		fprintf(stderr, "FAIL: %s TREE_VERIFY capability flag mismatch\\n", captured_model->adapter_id);
		return 1;
	}
	return 0;
}
"""


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="spark-tree-grain-") as directory:
        harness = pathlib.Path(directory) / "harness.c"
        harness.write_text(HARNESS, encoding="utf-8")
        for name, source, includes, defines, symbol, expected_grain, pool_mult in ADAPTERS:
            binary = pathlib.Path(directory) / f"check_{name}"
            cmd = [
                CC,
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Iinclude",
                "-Isrc",
                "-Imodel-families/common/include",
                *[f"-I{inc}" for inc in includes],
                *defines,
                f"-DRUNNER_SYMBOL={symbol}",
                f"-DEXPECTED_GRAIN={expected_grain}u",
                f"-DEXPECTED_POOL_MULT={pool_mult}u",
                str(harness),
                source,
                "runtime/model_serving_adapter.c",
                "src/spark_speculation_tree_frame.c",
                "src/spark_status.c",
                "-o",
                str(binary),
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            subprocess.run([str(binary)], cwd=ROOT, check=True)
    tree_count = sum(1 for entry in ADAPTERS if entry[5] != 0)
    non_tree_count = len(ADAPTERS) - tree_count
    print(f"PASS test_tree_slot_grain_adapters: verified {tree_count} tree-enabled adapters and {non_tree_count} non-tree adapters")


if __name__ == "__main__":
    main()
