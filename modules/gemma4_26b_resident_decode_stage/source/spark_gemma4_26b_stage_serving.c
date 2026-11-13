#include "sparkpipe/spark_gemma4_26b_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef GEMMA4_26B_CONTRACT_SHA256
#error "GEMMA4_26B_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_GEMMA4_26B_TP16_ADAPTER_ID "gemma4_26b-tp16"
#define SPARK_GEMMA4_26B_TP16_MODEL_ID "google/gemma-4-26B-A4B-it"
#define SPARK_GEMMA4_26B_TP16_MAX_ROWS 2048u
#define SPARK_GEMMA4_26B_TP16_CHECKPOINT_TOKENS 16384u

static void Gemma4_26bServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_GEMMA4_26B_STAGE_PAGE_SLOTS * SPARK_GEMMA4_26B_STAGE_SLIDING_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_GEMMA4_26B_STAGE_PAGE_SLOTS * SPARK_GEMMA4_26B_STAGE_FULL_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_GEMMA4_26B_STAGE_FULL_LAYERS;
	kv->arena_kv_head_count = SPARK_GEMMA4_26B_STAGE_SLIDING_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_GEMMA4_26B_STAGE_SLIDING_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_GEMMA4_26B_STAGE_SLIDING_LAYERS;
	kv->capacity_request.head_count = SPARK_GEMMA4_26B_STAGE_SLIDING_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_GEMMA4_26B_STAGE_SLIDING_DIM;
	kv->capacity_request.value_head_dimension = SPARK_GEMMA4_26B_STAGE_SLIDING_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Gemma4_26bServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * SPARK_GEMMA4_26B_STAGE_FULL_FUSED_WIDTH + SPARK_GEMMA4_26B_STAGE_HIDDEN - 1u) /
		SPARK_GEMMA4_26B_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkStageServingModel Gemma4_26bStageServing =
{
	.adapter_id = SPARK_GEMMA4_26B_TP16_ADAPTER_ID,
	.model_id = SPARK_GEMMA4_26B_TP16_MODEL_ID,
	.model_revision = SPARK_GEMMA4_26B_TP16_ADAPTER_ID,
	.program = "gemma4_26b",
	.contract_sha256 = GEMMA4_26B_CONTRACT_SHA256,
	.module_tag = "gemma4_26b_stage",
	.runner_model = SparkGemma4_26bStageModel,
	.tp_degree = SPARK_GEMMA4_26B_STAGE_TP_DEGREE,
	.layer_count = SPARK_GEMMA4_26B_STAGE_LAYERS,
	.hidden = SPARK_GEMMA4_26B_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = SPARK_GEMMA4_26B_TP16_MAX_ROWS,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_GEMMA4_26B_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_GEMMA4_26B_TP16_CHECKPOINT_TOKENS,
	.kv_layout = Gemma4_26bServingKvLayout,
	.collective_sequences = Gemma4_26bServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Gemma4_26bStageServing);
}
