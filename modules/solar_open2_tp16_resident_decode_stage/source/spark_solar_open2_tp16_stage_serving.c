#include "sparkpipe/spark_solar_open2_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef SOLAR_OPEN2_CONTRACT_SHA256
#error "SOLAR_OPEN2_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_SOLAR_OPEN2_TP16_ADAPTER_ID "solar_open2-tp16"
#define SPARK_SOLAR_OPEN2_TP16_MODEL_ID "upstage/Solar-Open2-250B"
#define SPARK_SOLAR_OPEN2_TP16_PROGRAM "solar_open2"
#define SPARK_SOLAR_OPEN2_TP16_MAX_ROWS 2048u
#define SPARK_SOLAR_OPEN2_TP16_CHAIN_TOKENS 16u
#define SPARK_SOLAR_OPEN2_TP16_CHECKPOINT_TOKENS 16384u
#define Q SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS

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

static const SparkModelServingAdapterDescriptor SolarOpen2ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_SOLAR_OPEN2_TP16_ADAPTER_ID,
		SPARK_SOLAR_OPEN2_TP16_MODEL_ID,
		SPARK_SOLAR_OPEN2_TP16_ADAPTER_ID,
		SPARK_SOLAR_OPEN2_TP16_PROGRAM,
		SOLAR_OPEN2_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_SOLAR_OPEN2_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_SOLAR_OPEN2_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = SPARK_SOLAR_OPEN2_TP16_CHAIN_TOKENS - 1u,
	.stage_layer_counts = { Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = SPARK_SOLAR_OPEN2_TP16_CHECKPOINT_TOKENS,
};

#undef Q

static const SparkStageServingModel SolarOpen2StageServing =
{
	.descriptor = &SolarOpen2ServingDescriptor,
	.runner_model = SparkSolarOpen2StageModel,
	.module_tag = "solar_open2_stage",
	.tp_degree = SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = SolarOpen2ServingKvLayout,
	.collective_sequences = SolarOpen2ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&SolarOpen2StageServing);
}
