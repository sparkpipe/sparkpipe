#include "sparkpipe/spark_laguna_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef LAGUNA_TP16_CONTRACT_SHA256
#error "LAGUNA_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

static void LagunaTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	const uint64_t page_bytes = (uint64_t)SPARK_LAGUNA_TP16_STAGE_PAGE_SLOTS * SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES;
	SparkStageKvInitFull(kv, tp_rank, tp_degree, page_bytes, SPARK_LAGUNA_TP16_STAGE_SLIDING_LAYERS,
		SPARK_LAGUNA_TP16_STAGE_KV_HEADS, SPARK_LAGUNA_TP16_STAGE_HEAD_DIM, SPARK_LAGUNA_TP16_STAGE_HEAD_DIM);
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR, page_bytes, SPARK_LAGUNA_TP16_STAGE_FULL_LAYERS);
}

_Static_assert(SPARK_LAGUNA_TP16_STAGE_SLIDING_FUSED_WIDTH >= SPARK_LAGUNA_TP16_STAGE_FULL_FUSED_WIDTH,
	"the sliding q|k|v gather is the widest collective");

static uint32_t LagunaTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, SPARK_LAGUNA_TP16_STAGE_SLIDING_FUSED_WIDTH, SPARK_LAGUNA_TP16_STAGE_HIDDEN);
}

static const SparkStageServingModel LagunaTp16StageServing =
{
	.adapter_id = "laguna-tp16",
	.model_id = "poolside/Laguna-S-2.1",
	.model_revision = "laguna-tp16",
	.program = "laguna",
	.contract_sha256 = LAGUNA_TP16_CONTRACT_SHA256,
	.module_tag = "laguna_stage",
	.runner_model = SparkLagunaTp16StageModel,
	.tp_degree = SPARK_LAGUNA_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_LAGUNA_TP16_STAGE_LAYERS,
	.hidden = SPARK_LAGUNA_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = 2048u,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_LAGUNA_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_LAGUNA_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = LagunaTp16ServingKvLayout,
	.collective_sequences = LagunaTp16ServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(LagunaTp16StageServing)
