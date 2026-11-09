#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef GLM53_FLASH_TP16_CONTRACT_SHA256
#error "GLM53_FLASH_TP16_CONTRACT_SHA256 must identify the model contract"
#endif
#ifndef GLM53_FLASH_EXPERT_CODEC
#error "GLM53_FLASH_EXPERT_CODEC must name the routed-expert weight codec of the packs this build serves"
#endif
#if !defined(GLM53_FLASH_ADAPTER_ID) || !defined(GLM53_FLASH_MODULE_TAG)
#error "GLM53_FLASH_ADAPTER_ID and GLM53_FLASH_MODULE_TAG must name this build's adapter and KV label"
#endif
#if !defined(GLM53_FLASH_MODEL_ID) || !defined(GLM53_FLASH_MODEL_REVISION)
#error "GLM53_FLASH_MODEL_ID and GLM53_FLASH_MODEL_REVISION must name the checkpoint this build's packs come from"
#endif

#define SPARK_GLM53_FLASH_TP16_ADAPTER_ID GLM53_FLASH_ADAPTER_ID
#define SPARK_GLM53_FLASH_TP16_MODEL_ID GLM53_FLASH_MODEL_ID
#define SPARK_GLM53_FLASH_TP16_MODEL_REVISION GLM53_FLASH_MODEL_REVISION
#define SPARK_GLM53_FLASH_TP16_MAX_ROWS 4096u
#define SPARK_GLM53_FLASH_TP16_CHECKPOINT_TOKENS 4096u

static void Glm53FlashTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_GLM53_FLASH_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FLASH_TP16_STAGE_KV_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_GLM53_FLASH_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FLASH_TP16_STAGE_INDEX_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_GLM53_FLASH_TP16_STAGE_KV_LAYERS;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_GLM53_FLASH_TP16_STAGE_LATENT;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_GLM53_FLASH_TP16_STAGE_KV_LAYERS;
	kv->capacity_request.compressed_dimension = SPARK_GLM53_FLASH_TP16_STAGE_LATENT;
	kv->capacity_request.position_dimension = 0u;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.fp8_scale_block_size = SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK;
	kv->capacity_request.index_key_layer_count = SPARK_GLM53_FLASH_TP16_STAGE_KV_LAYERS;
	kv->capacity_request.index_key_dimension = SPARK_GLM53_FLASH_TP16_STAGE_INDEX_DIMENSION;
	kv->capacity_request.index_key_bytes_per_scalar = 2u;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Glm53FlashTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return 2u * rows + 1u;
}

static const SparkStageServingModel Glm53FlashTp16StageServing =
{
	.adapter_id = SPARK_GLM53_FLASH_TP16_ADAPTER_ID,
	.model_id = SPARK_GLM53_FLASH_TP16_MODEL_ID,
	.model_revision = SPARK_GLM53_FLASH_TP16_MODEL_REVISION,
	.program = "glm53_flash",
	.contract_sha256 = GLM53_FLASH_TP16_CONTRACT_SHA256,
	.module_tag = GLM53_FLASH_MODULE_TAG,
	.runner_model = SparkGlm53FlashTp16StageModel,
	.tp_degree = SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_GLM53_FLASH_TP16_STAGE_LAYERS,
	.hidden = SPARK_GLM53_FLASH_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = GLM53_FLASH_EXPERT_CODEC,
	.max_rows = SPARK_GLM53_FLASH_TP16_MAX_ROWS,
	.verify_rows = SPARK_GLM53_FLASH_TP16_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_GLM53_FLASH_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_GLM53_FLASH_TP16_CHECKPOINT_TOKENS,
	.kv_layout = Glm53FlashTp16ServingKvLayout,
	.collective_sequences = Glm53FlashTp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Glm53FlashTp16StageServing);
}
