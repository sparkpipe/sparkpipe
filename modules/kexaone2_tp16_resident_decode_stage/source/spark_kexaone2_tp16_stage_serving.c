#include "sparkpipe/spark_kexaone2_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef KEXAONE2_TP16_CONTRACT_SHA256
#error "KEXAONE2_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_KEXAONE2_TP16_ADAPTER_ID "kexaone2-tp16"
#define SPARK_KEXAONE2_TP16_MODEL_ID "LGAI-EXAONE/K-EXAONE-2.0-750B-A37B"
#define SPARK_KEXAONE2_TP16_MAX_ROWS 2048u
#define SPARK_KEXAONE2_TP16_CHAIN_TOKENS 16u
#define SPARK_KEXAONE2_TP16_CHECKPOINT_TOKENS 16384u
#define Q SPARK_KEXAONE2_TP16_STAGE_LAYERS

static void Kexaone2Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_KEXAONE2_TP16_STAGE_PAGE_SLOTS * SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_KEXAONE2_TP16_STAGE_PAGE_SLOTS * SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_KEXAONE2_TP16_STAGE_FULL_LAYERS;
	kv->arena_kv_head_count = SPARK_KEXAONE2_TP16_STAGE_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_KEXAONE2_TP16_STAGE_SLIDING_LAYERS;
	kv->capacity_request.head_count = SPARK_KEXAONE2_TP16_STAGE_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}


static uint32_t Kexaone2Tp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH + SPARK_KEXAONE2_TP16_STAGE_HIDDEN - 1u) /
		SPARK_KEXAONE2_TP16_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkModelServingAdapterDescriptor Kexaone2Tp16ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_KEXAONE2_TP16_ADAPTER_ID,
		SPARK_KEXAONE2_TP16_MODEL_ID,
		SPARK_KEXAONE2_TP16_ADAPTER_ID,
		"kexaone2",
		KEXAONE2_TP16_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATIVE_VERIFY,
	.stage_count = SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_KEXAONE2_TP16_STAGE_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_KEXAONE2_TP16_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_KEXAONE2_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_KEXAONE2_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.stage_layer_counts = { Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_KEXAONE2_TP16_STAGE_PAGE_SLOTS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = SPARK_KEXAONE2_TP16_CHECKPOINT_TOKENS,
	.max_verify_row_count = SPARK_KEXAONE2_TP16_MAX_ROWS,
};

#undef Q

static const SparkStageServingModel Kexaone2Tp16StageServing =
{
	.descriptor = &Kexaone2Tp16ServingDescriptor,
	.runner_model = SparkKexaone2Tp16StageModel,
	.module_tag = "kexaone2_stage",
	.tp_degree = SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = Kexaone2Tp16ServingKvLayout,
	.collective_sequences = Kexaone2Tp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Kexaone2Tp16StageServing);
}
