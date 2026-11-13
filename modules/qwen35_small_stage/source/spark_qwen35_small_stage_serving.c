#include "inference/runner/device_stage_model.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef QWEN35_SMALL_CONTRACT_SHA256
#error "QWEN35_SMALL_CONTRACT_SHA256 must identify the model contract"
#endif

#if defined(QWEN35_4B)
#define QWEN35_SMALL_ADAPTER_ID "qwen35_4b-bf16-tp1"
#define QWEN35_SMALL_MODEL_ID "Qwen/Qwen3.5-4B"
#define QWEN35_SMALL_PROGRAM "qwen35_4b"
#define QWEN35_SMALL_RUNNER_MODEL SparkQwen35_4bDeviceStageModel
#define QWEN35_SMALL_HIDDEN 2560u
#define QWEN35_SMALL_LAYERS 32u
#define QWEN35_SMALL_HEADS 16u
#define QWEN35_SMALL_KV_HEADS 4u
#elif defined(QWEN35_2B)
#define QWEN35_SMALL_ADAPTER_ID "qwen35_2b-bf16-tp1"
#define QWEN35_SMALL_MODEL_ID "Qwen/Qwen3.5-2B"
#define QWEN35_SMALL_PROGRAM "qwen35_2b"
#define QWEN35_SMALL_RUNNER_MODEL SparkQwen35_2bDeviceStageModel
#define QWEN35_SMALL_HIDDEN 2048u
#define QWEN35_SMALL_LAYERS 24u
#define QWEN35_SMALL_HEADS 8u
#define QWEN35_SMALL_KV_HEADS 2u
#else
#error "build the Qwen3.5 small-model adapter with QWEN35_4B or QWEN35_2B"
#endif

#define QWEN35_SMALL_HEAD_DIM 256u
#define QWEN35_SMALL_KV_LAYERS (QWEN35_SMALL_LAYERS / 4u)
#define QWEN35_SMALL_KV_SLOT_BYTES (QWEN35_SMALL_KV_HEADS * QWEN35_SMALL_HEAD_DIM * 2u * 2u)
#define QWEN35_SMALL_QKV_WIDTH (2u * QWEN35_SMALL_HEADS * QWEN35_SMALL_HEAD_DIM + 2u * QWEN35_SMALL_KV_HEADS * QWEN35_SMALL_HEAD_DIM)
#define QWEN35_SMALL_MAX_ROWS 2048u
#define QWEN35_SMALL_CHAIN_TOKENS 16u
#define QWEN35_SMALL_CHECKPOINT_TOKENS 16384u

static void Qwen35SmallServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_DEVICE_RUNNER_PAGE_SLOTS * QWEN35_SMALL_KV_SLOT_BYTES;
	kv->arena_kv_head_count = QWEN35_SMALL_KV_HEADS;
	kv->arena_head_dim = 2u * QWEN35_SMALL_HEAD_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = QWEN35_SMALL_KV_LAYERS;
	kv->capacity_request.head_count = QWEN35_SMALL_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = QWEN35_SMALL_HEAD_DIM;
	kv->capacity_request.value_head_dimension = QWEN35_SMALL_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Qwen35SmallServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * QWEN35_SMALL_QKV_WIDTH + QWEN35_SMALL_HIDDEN - 1u) / QWEN35_SMALL_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

#define Q QWEN35_SMALL_LAYERS

static const SparkModelServingAdapterDescriptor Qwen35SmallServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		QWEN35_SMALL_ADAPTER_ID,
		QWEN35_SMALL_MODEL_ID,
		QWEN35_SMALL_ADAPTER_ID,
		QWEN35_SMALL_PROGRAM,
		QWEN35_SMALL_CONTRACT_SHA256),
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
	.stage_count = 1u,
	.layer_count = QWEN35_SMALL_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = QWEN35_SMALL_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = QWEN35_SMALL_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = QWEN35_SMALL_CHAIN_TOKENS,
	.max_speculative_token_count = SPARK_DEVICE_STAGE_VERIFY_ROWS - 1u,
	.stage_layer_counts = { Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_DEVICE_RUNNER_PAGE_SLOTS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = QWEN35_SMALL_CHECKPOINT_TOKENS,
	.max_verify_row_count = SPARK_DEVICE_STAGE_VERIFY_ROWS,
};

#undef Q

static const SparkStageServingModel Qwen35SmallStageServing =
{
	.descriptor = &Qwen35SmallServingDescriptor,
	.runner_model = QWEN35_SMALL_RUNNER_MODEL,
	.module_tag = QWEN35_SMALL_PROGRAM "_stage",
	.tp_degree = 1u,
	.program_id = 1u,
	.kv_layout = Qwen35SmallServingKvLayout,
	.collective_sequences = Qwen35SmallServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Qwen35SmallStageServing);
}
