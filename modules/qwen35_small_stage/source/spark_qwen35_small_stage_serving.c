#include "inference/runner/device_stage_model.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
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

static const SparkStageServingModel Qwen35SmallStageServing =
{
	.adapter_id = QWEN35_SMALL_ADAPTER_ID,
	.model_id = QWEN35_SMALL_MODEL_ID,
	.model_revision = QWEN35_SMALL_ADAPTER_ID,
	.program = QWEN35_SMALL_PROGRAM,
	.contract_sha256 = QWEN35_SMALL_CONTRACT_SHA256,
	.module_tag = QWEN35_SMALL_PROGRAM "_stage",
	.runner_model = QWEN35_SMALL_RUNNER_MODEL,
	.tp_degree = 1u,
	.layer_count = QWEN35_SMALL_LAYERS,
	.hidden = QWEN35_SMALL_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = QWEN35_SMALL_MAX_ROWS,
	.verify_rows = SPARK_DEVICE_STAGE_VERIFY_ROWS,
	.speculative_tokens = SPARK_DEVICE_STAGE_VERIFY_ROWS - 1u,
	.cache_block_token_count = SPARK_DEVICE_RUNNER_PAGE_SLOTS,
	.cache_checkpoint_token_count = QWEN35_SMALL_CHECKPOINT_TOKENS,
	.kv_layout = Qwen35SmallServingKvLayout,
	.collective_sequences = Qwen35SmallServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Qwen35SmallStageServing);
}
