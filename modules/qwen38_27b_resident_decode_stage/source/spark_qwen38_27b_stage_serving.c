#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef QWEN38_27B_CONTRACT_SHA256
#error "QWEN38_27B_CONTRACT_SHA256 must identify the model contract"
#endif

#if defined(QWEN38_27B_BF16)
#define SPARK_QWEN38_27B_TP16_ADAPTER_ID "qwen38_27b-bf16-tp16"
#define SPARK_QWEN38_27B_TP16_LINEAR_CODEC SPARK_WEIGHT_CODEC_BF16
#else
#define SPARK_QWEN38_27B_TP16_ADAPTER_ID "qwen38_27b-tp16"
#define SPARK_QWEN38_27B_TP16_LINEAR_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#endif
#define SPARK_QWEN38_27B_TP16_MODEL_ID "Qwen/Qwen3.8-27B"
#define SPARK_QWEN38_27B_TP16_MAX_ROWS 2048u
#define SPARK_QWEN38_27B_TP16_LAYERS 64u
#define SPARK_QWEN38_27B_TP16_KV_LAYERS 16u
#define SPARK_QWEN38_27B_TP16_CHAIN_TOKENS 16u
#define SPARK_QWEN38_27B_TP16_CHECKPOINT_TOKENS 16384u
#define Q SPARK_QWEN38_27B_TP16_LAYERS

static void Qwen38_27bServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_QWEN38_27B_STAGE_PAGE_SLOTS * SPARK_QWEN38_27B_STAGE_KV_SLOT_BYTES;
	kv->arena_kv_head_count = SPARK_QWEN38_27B_STAGE_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_QWEN38_27B_STAGE_HEAD_DIM;
	kv->arena_bytes_per_scalar = 2u;
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_QWEN38_27B_TP16_KV_LAYERS;
	kv->capacity_request.head_count = SPARK_QWEN38_27B_STAGE_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_QWEN38_27B_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_QWEN38_27B_STAGE_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = 2u;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Qwen38_27bServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * SPARK_QWEN38_27B_STAGE_QKV_WIDTH + SPARK_QWEN38_27B_STAGE_HIDDEN - 1u) /
		SPARK_QWEN38_27B_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkModelServingAdapterDescriptor Qwen38_27bServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_QWEN38_27B_TP16_ADAPTER_ID,
		SPARK_QWEN38_27B_TP16_MODEL_ID,
		SPARK_QWEN38_27B_TP16_ADAPTER_ID,
		"qwen38_27b",
		QWEN38_27B_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = SPARK_QWEN38_27B_STAGE_TP_DEGREE,
	.layer_count = SPARK_QWEN38_27B_TP16_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_QWEN38_27B_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_QWEN38_27B_TP16_LINEAR_CODEC,
	.expert_weight_codec = SPARK_QWEN38_27B_TP16_LINEAR_CODEC,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_QWEN38_27B_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_QWEN38_27B_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = SPARK_QWEN38_27B_TP16_CHAIN_TOKENS - 1u,
	.stage_layer_counts = { Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_QWEN38_27B_STAGE_PAGE_SLOTS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = SPARK_QWEN38_27B_TP16_CHECKPOINT_TOKENS,
};

#undef Q

static const SparkStageServingModel Qwen38_27bStageServing =
{
	.descriptor = &Qwen38_27bServingDescriptor,
	.runner_model = SparkQwen38_27bStageModel,
	.module_tag = "qwen38_27b_stage",
	.tp_degree = SPARK_QWEN38_27B_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = Qwen38_27bServingKvLayout,
	.collective_sequences = Qwen38_27bServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Qwen38_27bStageServing);
}
