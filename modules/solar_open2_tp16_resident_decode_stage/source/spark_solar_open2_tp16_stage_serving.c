#include "sparkpipe/spark_solar_open2_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef SOLAR_OPEN2_CONTRACT_SHA256
#error "SOLAR_OPEN2_CONTRACT_SHA256 must identify the model contract"
#endif

static void SolarOpen2ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitFull(kv, tp_rank, tp_degree, (uint64_t)SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS * SPARK_SOLAR_OPEN2_TP16_STAGE_KV_SLOT_BYTES,
		SPARK_SOLAR_OPEN2_TP16_STAGE_ATTN_LAYERS, SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS, SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM, SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM);
}

static uint32_t SolarOpen2ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, SPARK_SOLAR_OPEN2_TP16_STAGE_QKV_WIDTH, SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN);
}

static const SparkStageServingModel SolarOpen2StageServing =
{
	.adapter_id = "solar_open2-tp16",
	.model_id = "upstage/Solar-Open2-250B",
	.model_revision = "solar_open2-tp16",
	.program = "solar_open2",
	.contract_sha256 = SOLAR_OPEN2_CONTRACT_SHA256,
	.module_tag = "solar_open2_stage",
	.runner_model = SparkSolarOpen2StageModel,
	.tp_degree = SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS,
	.hidden = SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = 2048u,
	.verify_rows = SPARK_SOLAR_OPEN2_TP16_VERIFY_ROWS,
	.tree_slot_grain = SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = SolarOpen2ServingKvLayout,
	.collective_sequences = SolarOpen2ServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(SolarOpen2StageServing)
