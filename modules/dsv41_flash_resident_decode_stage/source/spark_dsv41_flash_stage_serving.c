#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef DSV41_FLASH_CONTRACT_SHA256
#error "DSV41_FLASH_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_DSV41_FLASH_TP16_ADAPTER_ID "dsv41_flash-tp16"
#define SPARK_DSV41_FLASH_TP16_MODEL_ID "deepseek-ai/DeepSeek-V4.1-Flash"
#define SPARK_DSV41_FLASH_TP16_MAX_ROWS 2048u
#define SPARK_DSV41_FLASH_TP16_LAYERS 40u
#define SPARK_DSV41_FLASH_TP16_CHECKPOINT_TOKENS 16384u
#define SPARK_DSV41_FLASH_TP16_LATENT 512u
#define SPARK_DSV41_FLASH_TP16_INDEX_KEY 128u
#define SPARK_DSV41_FLASH_TP16_ENGRAM_WIDTH (24u * 256u)

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
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES + SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES;
	kv->capacity_request.compressed_dimension = SPARK_DSV41_FLASH_TP16_LATENT;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.index_key_layer_count = SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES + SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES;
	kv->capacity_request.index_key_dimension = SPARK_DSV41_FLASH_TP16_INDEX_KEY;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN;
}

static uint32_t Dsv41FlashServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t engram = ((uint64_t)rows * SPARK_DSV41_FLASH_TP16_ENGRAM_WIDTH + SPARK_DSV41_FLASH_STAGE_HIDDEN - 1u) /
		SPARK_DSV41_FLASH_STAGE_HIDDEN;
	const uint64_t split = 2u * (uint64_t)(rows < SPARK_DSV41_FLASH_STAGE_ENGRAM_SPLIT_ROWS ? rows : SPARK_DSV41_FLASH_STAGE_ENGRAM_SPLIT_ROWS);
	uint64_t capacity = engram > rows ? engram : rows;
	(void)tp_degree;
	capacity = split > capacity ? split : capacity;
	capacity = SPARK_DSV41_FLASH_STAGE_LANDING_ROWS(SPARK_DSV41_FLASH_STAGE_MAX_POSITIONS, 1u) > capacity ?
		SPARK_DSV41_FLASH_STAGE_LANDING_ROWS(SPARK_DSV41_FLASH_STAGE_MAX_POSITIONS, 1u) : capacity;
	return (uint32_t)capacity + 1u;
}

static const SparkStageServingModel Dsv41FlashStageServing =
{
	.adapter_id = SPARK_DSV41_FLASH_TP16_ADAPTER_ID,
	.model_id = SPARK_DSV41_FLASH_TP16_MODEL_ID,
	.model_revision = SPARK_DSV41_FLASH_TP16_ADAPTER_ID,
	.program = "dsv41_flash",
	.contract_sha256 = DSV41_FLASH_CONTRACT_SHA256,
	.module_tag = "dsv41_flash_stage",
	.runner_model = SparkDsv41FlashStageModel,
	.tp_degree = SPARK_DSV41_FLASH_STAGE_TP_DEGREE,
	.layer_count = SPARK_DSV41_FLASH_TP16_LAYERS,
	.hidden = SPARK_DSV41_FLASH_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.max_rows = SPARK_DSV41_FLASH_TP16_MAX_ROWS,
	.verify_rows = SPARK_DSV41_FLASH_STAGE_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS,
	.cache_checkpoint_token_count = SPARK_DSV41_FLASH_TP16_CHECKPOINT_TOKENS,
	.kv_layout = Dsv41FlashServingKvLayout,
	.collective_sequences = Dsv41FlashServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Dsv41FlashStageServing);
}
