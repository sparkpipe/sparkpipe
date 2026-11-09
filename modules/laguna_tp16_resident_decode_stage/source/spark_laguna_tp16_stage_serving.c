#include "sparkpipe/spark_laguna_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef LAGUNA_TP16_CONTRACT_SHA256
#error "LAGUNA_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_LAGUNA_TP16_ADAPTER_ID "laguna-tp16"
#define SPARK_LAGUNA_TP16_MODEL_ID "poolside/Laguna-S-2.1"
#define SPARK_LAGUNA_TP16_MAX_ROWS 2048u
#define SPARK_LAGUNA_TP16_CHECKPOINT_TOKENS 16384u

static void LagunaTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_LAGUNA_TP16_STAGE_PAGE_SLOTS * SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_LAGUNA_TP16_STAGE_PAGE_SLOTS * SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_LAGUNA_TP16_STAGE_FULL_LAYERS;
	kv->arena_kv_head_count = SPARK_LAGUNA_TP16_STAGE_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_LAGUNA_TP16_STAGE_HEAD_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_LAGUNA_TP16_STAGE_SLIDING_LAYERS;
	kv->capacity_request.head_count = SPARK_LAGUNA_TP16_STAGE_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_LAGUNA_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_LAGUNA_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

_Static_assert(SPARK_LAGUNA_TP16_STAGE_SLIDING_FUSED_WIDTH >= SPARK_LAGUNA_TP16_STAGE_FULL_FUSED_WIDTH,
	"the sliding q|k|v gather is the widest collective");

static uint32_t LagunaTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * SPARK_LAGUNA_TP16_STAGE_SLIDING_FUSED_WIDTH + SPARK_LAGUNA_TP16_STAGE_HIDDEN - 1u) /
		SPARK_LAGUNA_TP16_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkStageServingModel LagunaTp16StageServing =
{
	.adapter_id = SPARK_LAGUNA_TP16_ADAPTER_ID,
	.model_id = SPARK_LAGUNA_TP16_MODEL_ID,
	.model_revision = SPARK_LAGUNA_TP16_ADAPTER_ID,
	.program = "laguna",
	.contract_sha256 = LAGUNA_TP16_CONTRACT_SHA256,
	.module_tag = "laguna_stage",
	.runner_model = SparkLagunaTp16StageModel,
	.tp_degree = SPARK_LAGUNA_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_LAGUNA_TP16_STAGE_LAYERS,
	.hidden = SPARK_LAGUNA_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = SPARK_LAGUNA_TP16_MAX_ROWS,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_LAGUNA_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_LAGUNA_TP16_CHECKPOINT_TOKENS,
	.kv_layout = LagunaTp16ServingKvLayout,
	.collective_sequences = LagunaTp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&LagunaTp16StageServing);
}
