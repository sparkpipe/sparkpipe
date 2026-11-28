#include "sparkpipe/spark_muse_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MUSE_TP16_CONTRACT_SHA256
#error "MUSE_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

static void MuseTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitFull(kv, tp_rank, tp_degree, (uint64_t)SPARK_MUSE_TP16_STAGE_PAGE_SLOTS * SPARK_MUSE_TP16_STAGE_SLIDING_SLOT_BYTES,
		SPARK_MUSE_TP16_STAGE_SLIDING_LAYERS, SPARK_MUSE_TP16_STAGE_SLIDING_KV_HEADS, SPARK_MUSE_TP16_STAGE_SLIDING_DIM, SPARK_MUSE_TP16_STAGE_SLIDING_DIM);
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_MUSE_TP16_STAGE_PAGE_SLOTS * SPARK_MUSE_TP16_STAGE_FULL_SLOT_BYTES, SPARK_MUSE_TP16_STAGE_FULL_LAYERS);
}

static uint32_t MuseTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, SPARK_MUSE_TP16_STAGE_FULL_FUSED_WIDTH, SPARK_MUSE_TP16_STAGE_HIDDEN);
}

static const SparkStageServingModel MuseTp16StageServing =
{
	.adapter_id = "muse_glimmer-tp16",
	.model_id = "meta-models/Muse-Glimmer-30B",
	.model_revision = "muse_glimmer-tp16",
	.program = "muse_glimmer",
	.contract_sha256 = MUSE_TP16_CONTRACT_SHA256,
	.module_tag = "muse_glimmer_stage",
	.runner_model = SparkMuseTp16StageModel,
	.tp_degree = SPARK_MUSE_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MUSE_TP16_STAGE_LAYERS,
	.hidden = SPARK_MUSE_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = 2048u,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_MUSE_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_MUSE_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = MuseTp16ServingKvLayout,
	.collective_sequences = MuseTp16ServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(MuseTp16StageServing)
