#include "sparkpipe/spark_gemma4_31b_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef GEMMA4_31B_CONTRACT_SHA256
#error "GEMMA4_31B_CONTRACT_SHA256 must identify the model contract"
#endif

static void Gemma4_31bServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitFull(kv, tp_rank, tp_degree, (uint64_t)SPARK_GEMMA4_31B_STAGE_PAGE_SLOTS * SPARK_GEMMA4_31B_STAGE_SLIDING_SLOT_BYTES,
		SPARK_GEMMA4_31B_STAGE_SLIDING_LAYERS, SPARK_GEMMA4_31B_STAGE_SLIDING_KV_HEADS, SPARK_GEMMA4_31B_STAGE_SLIDING_DIM, SPARK_GEMMA4_31B_STAGE_SLIDING_DIM);
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_GEMMA4_31B_STAGE_PAGE_SLOTS * SPARK_GEMMA4_31B_STAGE_FULL_SLOT_BYTES, SPARK_GEMMA4_31B_STAGE_FULL_LAYERS);
}

static uint32_t Gemma4_31bServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, SPARK_GEMMA4_31B_STAGE_FULL_FUSED_WIDTH, SPARK_GEMMA4_31B_STAGE_HIDDEN);
}

static const SparkStageServingModel Gemma4_31bStageServing =
{
	.adapter_id = "gemma4_31b-tp16",
	.model_id = "google/gemma-4-31B-it",
	.model_revision = "gemma4_31b-tp16",
	.program = "gemma4_31b",
	.contract_sha256 = GEMMA4_31B_CONTRACT_SHA256,
	.module_tag = "gemma4_31b_stage",
	.runner_model = SparkGemma4_31bStageModel,
	.tp_degree = SPARK_GEMMA4_31B_STAGE_TP_DEGREE,
	.layer_count = SPARK_GEMMA4_31B_STAGE_LAYERS,
	.hidden = SPARK_GEMMA4_31B_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = 2048u,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_GEMMA4_31B_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_GEMMA4_31B_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Gemma4_31bServingKvLayout,
	.collective_sequences = Gemma4_31bServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Gemma4_31bStageServing)
