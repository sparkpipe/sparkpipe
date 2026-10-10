#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef DSV41_FLASH_CONTRACT_SHA256
#error "DSV41_FLASH_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_DSV41_FLASH_TP16_ADAPTER_ID "dsv41_flash-tp16"
#define SPARK_DSV41_FLASH_TP16_MODEL_ID "deepseek-ai/DeepSeek-V4.1-Flash"
#define SPARK_DSV41_FLASH_TP16_MAX_ROWS 2048u
#define SPARK_DSV41_FLASH_TP16_LAYERS 40u
#define SPARK_DSV41_FLASH_TP16_CHAIN_TOKENS 16u
#define SPARK_DSV41_FLASH_TP16_CHECKPOINT_TOKENS 16384u
#define SPARK_DSV41_FLASH_TP16_LATENT 512u
#define SPARK_DSV41_FLASH_TP16_INDEX_KEY 128u
#define SPARK_DSV41_FLASH_TP16_RECORD_BYTES ((uint64_t)SPARK_DSV41_FLASH_STAGE_RECORD_FLOATS * sizeof(float))
#define L SPARK_DSV41_FLASH_TP16_LAYERS

static void Dsv41FlashServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)(SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS / 2u) * SPARK_DSV41_FLASH_STAGE_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS * SPARK_DSV41_FLASH_STAGE_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_DSV41_FLASH_STAGE_SLOT_BYTES / 2u;
	kv->arena_bytes_per_scalar = 2u;
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES + SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES;
	kv->capacity_request.compressed_dimension = SPARK_DSV41_FLASH_TP16_LATENT;
	kv->capacity_request.bytes_per_scalar = 2u;
	kv->capacity_request.index_key_layer_count = SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES + SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES;
	kv->capacity_request.index_key_dimension = SPARK_DSV41_FLASH_TP16_INDEX_KEY;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN;
}

static uint32_t Dsv41FlashServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t heads = 64u / tp_degree;
	const uint64_t bytes = (uint64_t)rows * heads * SPARK_DSV41_FLASH_TP16_RECORD_BYTES;
	const uint64_t units = (bytes + SPARK_DSV41_FLASH_STAGE_HIDDEN * 2u - 1u) / (SPARK_DSV41_FLASH_STAGE_HIDDEN * 2u);
	return (uint32_t)(units > rows ? units : rows) + 1u;
}

static const SparkModelServingAdapterDescriptor Dsv41FlashServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_DSV41_FLASH_TP16_ADAPTER_ID,
		SPARK_DSV41_FLASH_TP16_MODEL_ID,
		SPARK_DSV41_FLASH_TP16_ADAPTER_ID,
		"dsv41_flash",
		DSV41_FLASH_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = SPARK_DSV41_FLASH_STAGE_TP_DEGREE,
	.layer_count = SPARK_DSV41_FLASH_TP16_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_DSV41_FLASH_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_DSV41_FLASH_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_DSV41_FLASH_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = 0u,
	.stage_layer_counts = { L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = SPARK_DSV41_FLASH_TP16_CHECKPOINT_TOKENS,
};

#undef L

static const SparkStageServingModel Dsv41FlashStageServing =
{
	.descriptor = &Dsv41FlashServingDescriptor,
	.runner_model = SparkDsv41FlashStageModel,
	.module_tag = "dsv41_flash_stage",
	.tp_degree = SPARK_DSV41_FLASH_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = Dsv41FlashServingKvLayout,
	.collective_sequences = Dsv41FlashServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Dsv41FlashStageServing);
}
