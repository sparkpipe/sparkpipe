#include "sparkpipe/spark_ling3_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef LING3_CONTRACT_SHA256
#error "LING3_CONTRACT_SHA256 must identify the model contract"
#endif

#ifdef LING31_FLASH
#define SPARK_LING3_TP16_ADAPTER_ID "ling31_flash-tp16"
#define SPARK_LING3_TP16_MODEL_ID "inclusionAI/Ling-3.1-flash-1019"
#define SPARK_LING3_TP16_MODEL_REVISION "2109a6863fb14361eecc346ef98b3db88b8cdaee"
#define SPARK_LING3_TP16_PROGRAM "ling31_flash"
#define SPARK_LING3_TP16_MODULE_TAG "ling31_flash_stage"
#define SPARK_LING3_TP16_WEIGHT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#elif defined(LING32_AIR)
#define SPARK_LING3_TP16_ADAPTER_ID "ling32_air-tp16"
#define SPARK_LING3_TP16_MODEL_ID "inclusionAI/Ling-3.2-air"
#define SPARK_LING3_TP16_MODEL_REVISION "ff099487f6f57165e6fa9386ad61fddfafc78b76"
#define SPARK_LING3_TP16_PROGRAM "ling32_air"
#define SPARK_LING3_TP16_MODULE_TAG "ling32_air_stage"
#define SPARK_LING3_TP16_WEIGHT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#elif defined(LING3_FIN)
#define SPARK_LING3_TP16_ADAPTER_ID "ling3_fin-tp16"
#define SPARK_LING3_TP16_MODEL_ID "inclusionAI/Ling-3.0-flash-Fin"
#define SPARK_LING3_TP16_MODEL_REVISION SPARK_LING3_TP16_ADAPTER_ID
#define SPARK_LING3_TP16_PROGRAM "ling3_fin"
#define SPARK_LING3_TP16_MODULE_TAG "ling3_stage"
#define SPARK_LING3_TP16_WEIGHT_CODEC SPARK_WEIGHT_CODEC_BF16
#else
#define SPARK_LING3_TP16_ADAPTER_ID "ling3_flash-tp16"
#define SPARK_LING3_TP16_MODEL_ID "inclusionAI/Ling-3.0-flash"
#define SPARK_LING3_TP16_MODEL_REVISION SPARK_LING3_TP16_ADAPTER_ID
#define SPARK_LING3_TP16_PROGRAM "ling3_flash"
#define SPARK_LING3_TP16_MODULE_TAG "ling3_stage"
#define SPARK_LING3_TP16_WEIGHT_CODEC SPARK_WEIGHT_CODEC_BF16
#endif

static void Ling3ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitCompressed(kv, tp_rank, tp_degree, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_LING3_STAGE_PAGE_SLOTS * SPARK_LING3_STAGE_KV_SLOT_BYTES,
		SPARK_LING3_STAGE_LATENT + SPARK_LING3_STAGE_ROPE, SPARK_LING3_STAGE_MLA_LAYERS,
		SPARK_LING3_STAGE_LATENT, SPARK_LING3_STAGE_ROPE, 0u);
}

static uint32_t Ling3ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t record_unit = (uint64_t)SPARK_LING3_STAGE_HIDDEN * sizeof(uint16_t) / sizeof(float);
	const uint64_t rank_heads = tp_degree != 0u ? (uint64_t)SPARK_LING3_STAGE_MLA_HEADS / tp_degree : 0u;
	const uint64_t partials = ((uint64_t)rows * rank_heads * (SPARK_LING3_STAGE_LATENT + 2u) + record_unit - 1u) / record_unit;
	return (uint32_t)(partials > rows ? partials : rows) + 1u;
}

static const SparkStageServingModel Ling3StageServing =
{
	.adapter_id = SPARK_LING3_TP16_ADAPTER_ID,
	.model_id = SPARK_LING3_TP16_MODEL_ID,
	.model_revision = SPARK_LING3_TP16_MODEL_REVISION,
	.program = SPARK_LING3_TP16_PROGRAM,
	.contract_sha256 = LING3_CONTRACT_SHA256,
	.module_tag = SPARK_LING3_TP16_MODULE_TAG,
	.runner_model = SparkLing3StageModel,
	.tp_degree = SPARK_LING3_STAGE_TP_DEGREE,
	.layer_count = SPARK_LING3_STAGE_LAYERS,
	.hidden = SPARK_LING3_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_LING3_TP16_WEIGHT_CODEC,
	.expert_weight_codec = SPARK_LING3_TP16_WEIGHT_CODEC,
	.max_rows = 2048u,
	.verify_rows = SPARK_LING3_STAGE_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_LING3_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Ling3ServingKvLayout,
	.collective_sequences = Ling3ServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Ling3StageServing)
