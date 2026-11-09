#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen38_flash_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef QWEN38_FLASH_CONTRACT_SHA256
#error "QWEN38_FLASH_CONTRACT_SHA256 must identify the model contract"
#endif
#ifndef QWEN38_FLASH_EXPERT_CODEC
#error "QWEN38_FLASH_EXPERT_CODEC must name the routed-expert weight codec of the packs this build serves"
#endif
#if !defined(QWEN38_FLASH_ADAPTER_ID) || !defined(QWEN38_FLASH_MODULE_TAG)
#error "QWEN38_FLASH_ADAPTER_ID and QWEN38_FLASH_MODULE_TAG must name this build's adapter and KV label"
#endif

#define SPARK_QWEN38_FLASH_TP16_ADAPTER_ID QWEN38_FLASH_ADAPTER_ID
#define SPARK_QWEN38_FLASH_TP16_MODEL_ID "Qwen/Qwen3.8-Flash-Next"
#define SPARK_QWEN38_FLASH_TP16_MAX_ROWS 2048u
#define SPARK_QWEN38_FLASH_TP16_LAYERS SPARK_QWEN38_FLASH_STAGE_LAYERS
#define SPARK_QWEN38_FLASH_TP16_KV_LAYERS SPARK_QWEN38_FLASH_STAGE_KV_LAYERS
#define SPARK_QWEN38_FLASH_TP16_CHECKPOINT_TOKENS 16384u

static void Qwen38FlashServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_QWEN38_FLASH_STAGE_PAGE_SLOTS * SPARK_QWEN38_FLASH_STAGE_KV_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)(SPARK_QWEN38_FLASH_STAGE_PAGE_SLOTS / SPARK_QWEN38_FLASH_STAGE_INDEX_RATIO) *
		SPARK_QWEN38_FLASH_STAGE_INDEX_DIM * 2u;
	kv->regions[1].layer_count = SPARK_QWEN38_FLASH_TP16_KV_LAYERS;
	kv->arena_kv_head_count = SPARK_QWEN38_FLASH_STAGE_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_QWEN38_FLASH_STAGE_HEAD_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_QWEN38_FLASH_TP16_KV_LAYERS;
	kv->capacity_request.head_count = SPARK_QWEN38_FLASH_STAGE_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_QWEN38_FLASH_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_QWEN38_FLASH_STAGE_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Qwen38FlashServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t attention = (uint64_t)rows * SPARK_QWEN38_FLASH_STAGE_QKV_WIDTH;
	const uint64_t candidates = (uint64_t)rows * tp_degree * SPARK_QWEN38_FLASH_STAGE_INDEX_BLOCKS * 4u;
	const uint64_t gathered = ((attention > candidates ? attention : candidates) + SPARK_QWEN38_FLASH_STAGE_HIDDEN - 1u) /
		SPARK_QWEN38_FLASH_STAGE_HIDDEN;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkStageServingModel Qwen38FlashStageServing =
{
	.adapter_id = SPARK_QWEN38_FLASH_TP16_ADAPTER_ID,
	.model_id = SPARK_QWEN38_FLASH_TP16_MODEL_ID,
	.model_revision = SPARK_QWEN38_FLASH_TP16_ADAPTER_ID,
	.program = "qwen38_flash",
	.contract_sha256 = QWEN38_FLASH_CONTRACT_SHA256,
	.module_tag = QWEN38_FLASH_MODULE_TAG,
	.runner_model = SparkQwen38FlashStageModel,
	.tp_degree = SPARK_QWEN38_FLASH_STAGE_TP_DEGREE,
	.layer_count = SPARK_QWEN38_FLASH_TP16_LAYERS,
	.hidden = SPARK_QWEN38_FLASH_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = QWEN38_FLASH_EXPERT_CODEC,
	.max_rows = SPARK_QWEN38_FLASH_TP16_MAX_ROWS,
	.verify_rows = SPARK_QWEN38_FLASH_STAGE_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_QWEN38_FLASH_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_QWEN38_FLASH_TP16_CHECKPOINT_TOKENS,
	.kv_layout = Qwen38FlashServingKvLayout,
	.collective_sequences = Qwen38FlashServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Qwen38FlashStageServing);
}
