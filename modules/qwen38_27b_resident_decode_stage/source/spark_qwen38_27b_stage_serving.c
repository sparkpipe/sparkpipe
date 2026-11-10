#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef QWEN38_27B_CONTRACT_SHA256
#error "QWEN38_27B_CONTRACT_SHA256 must identify the model contract"
#endif

#if defined(QWEN38_27B_DEVICE_STAGE)
#include "inference/runner/device_stage_model.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "qwen38_27b-bf16-tp1"
#define SPARK_QWEN38_27B_SERVING_LINEAR_CODEC SPARK_WEIGHT_CODEC_BF16
#define SPARK_QWEN38_27B_SERVING_TP 1u
#define SPARK_QWEN38_27B_SERVING_VERIFY_ROWS SPARK_DEVICE_STAGE_VERIFY_ROWS
#define SPARK_QWEN38_27B_SERVING_RUNNER_MODEL SparkQwen38_27bDeviceStageModel
#else
#if defined(QWEN38_27B_BF16)
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "qwen38_27b-bf16-tp16"
#define SPARK_QWEN38_27B_SERVING_LINEAR_CODEC SPARK_WEIGHT_CODEC_BF16
#else
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "qwen38_27b-tp16"
#define SPARK_QWEN38_27B_SERVING_LINEAR_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#endif
#define SPARK_QWEN38_27B_SERVING_TP SPARK_QWEN38_27B_STAGE_TP_DEGREE
#define SPARK_QWEN38_27B_SERVING_VERIFY_ROWS SPARK_QWEN38_27B_STAGE_VERIFY_ROWS
#define SPARK_QWEN38_27B_SERVING_RUNNER_MODEL SparkQwen38_27bStageModel
#endif
#define SPARK_QWEN38_27B_SERVING_MODEL_ID "Qwen/Qwen3.8-27B"
#define SPARK_QWEN38_27B_SERVING_MAX_ROWS 2048u
#define SPARK_QWEN38_27B_SERVING_LAYERS 64u
#define SPARK_QWEN38_27B_SERVING_KV_LAYERS 16u
#define SPARK_QWEN38_27B_SERVING_CHECKPOINT_TOKENS 16384u

static void Qwen38_27bServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_QWEN38_27B_STAGE_PAGE_SLOTS * SPARK_QWEN38_27B_STAGE_KV_SLOT_BYTES;
	kv->arena_kv_head_count = SPARK_QWEN38_27B_STAGE_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_QWEN38_27B_STAGE_HEAD_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_QWEN38_27B_SERVING_KV_LAYERS;
	kv->capacity_request.head_count = SPARK_QWEN38_27B_STAGE_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_QWEN38_27B_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_QWEN38_27B_STAGE_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
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

static const SparkStageServingModel Qwen38_27bStageServing =
{
	.adapter_id = SPARK_QWEN38_27B_SERVING_ADAPTER_ID,
	.model_id = SPARK_QWEN38_27B_SERVING_MODEL_ID,
	.model_revision = SPARK_QWEN38_27B_SERVING_ADAPTER_ID,
	.program = "qwen38_27b",
	.contract_sha256 = QWEN38_27B_CONTRACT_SHA256,
	.module_tag = "qwen38_27b_stage",
	.runner_model = SPARK_QWEN38_27B_SERVING_RUNNER_MODEL,
	.tp_degree = SPARK_QWEN38_27B_SERVING_TP,
	.layer_count = SPARK_QWEN38_27B_SERVING_LAYERS,
	.hidden = SPARK_QWEN38_27B_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_QWEN38_27B_SERVING_LINEAR_CODEC,
	.expert_weight_codec = SPARK_QWEN38_27B_SERVING_LINEAR_CODEC,
	.max_rows = SPARK_QWEN38_27B_SERVING_MAX_ROWS,
	.verify_rows = SPARK_QWEN38_27B_SERVING_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_QWEN38_27B_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_QWEN38_27B_SERVING_CHECKPOINT_TOKENS,
	.kv_layout = Qwen38_27bServingKvLayout,
	.collective_sequences = Qwen38_27bServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Qwen38_27bStageServing);
}
