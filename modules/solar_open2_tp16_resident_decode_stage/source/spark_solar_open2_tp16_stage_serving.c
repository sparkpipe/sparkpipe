#include "sparkpipe/spark_solar_open2_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef SOLAR_OPEN2_CONTRACT_SHA256
#error "SOLAR_OPEN2_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_SOLAR_OPEN2_TP16_ADAPTER_ID "solar_open2-tp16"
#define SPARK_SOLAR_OPEN2_TP16_MODEL_ID "upstage/Solar-Open2-250B"
#define SPARK_SOLAR_OPEN2_TP16_PROGRAM "solar_open2"
#define SPARK_SOLAR_OPEN2_TP16_MAX_ROWS 2048u
#define SPARK_SOLAR_OPEN2_TP16_CHECKPOINT_TOKENS 16384u

static void SolarOpen2ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS * SPARK_SOLAR_OPEN2_TP16_STAGE_KV_SLOT_BYTES;
	kv->arena_kv_head_count = SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_SOLAR_OPEN2_TP16_STAGE_ATTN_LAYERS;
	kv->capacity_request.head_count = SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t SolarOpen2ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * SPARK_SOLAR_OPEN2_TP16_STAGE_QKV_WIDTH + SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN - 1u) /
		SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkStageServingModel SolarOpen2StageServing =
{
	.adapter_id = SPARK_SOLAR_OPEN2_TP16_ADAPTER_ID,
	.model_id = SPARK_SOLAR_OPEN2_TP16_MODEL_ID,
	.model_revision = SPARK_SOLAR_OPEN2_TP16_ADAPTER_ID,
	.program = SPARK_SOLAR_OPEN2_TP16_PROGRAM,
	.contract_sha256 = SOLAR_OPEN2_CONTRACT_SHA256,
	.module_tag = "solar_open2_stage",
	.runner_model = SparkSolarOpen2StageModel,
	.tp_degree = SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS,
	.hidden = SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = SPARK_SOLAR_OPEN2_TP16_MAX_ROWS,
	.verify_rows = SPARK_SOLAR_OPEN2_TP16_VERIFY_ROWS,
	.tree_slot_grain = SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_SOLAR_OPEN2_TP16_CHECKPOINT_TOKENS,
	.kv_layout = SolarOpen2ServingKvLayout,
	.collective_sequences = SolarOpen2ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&SolarOpen2StageServing);
}
