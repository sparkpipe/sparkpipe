#include "sparkpipe/spark_glm53_full_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef GLM53_FULL_CONTRACT_SHA256
#error "GLM53_FULL_CONTRACT_SHA256 must identify the pack contract"
#endif
#ifndef GLM53_FULL_MODEL_REVISION
#error "GLM53_FULL_MODEL_REVISION must name the checkpoint revision"
#endif
#ifndef GLM53_FULL_LINEAR_CODEC
#error "GLM53_FULL_LINEAR_CODEC must name the linear weight codec of the packs this build serves"
#endif
#ifndef GLM53_FULL_EXPERT_CODEC
#error "GLM53_FULL_EXPERT_CODEC must name the routed-expert weight codec of the packs this build serves"
#endif
#if !defined(GLM53_FULL_ADAPTER_ID) || !defined(GLM53_FULL_MODEL_ID) || !defined(GLM53_FULL_MODULE_TAG)
#error "GLM53_FULL_ADAPTER_ID, GLM53_FULL_MODEL_ID and GLM53_FULL_MODULE_TAG must name this build's adapter, served model and KV label"
#endif

#define SPARK_GLM53_FULL_TP16_ADAPTER_ID GLM53_FULL_ADAPTER_ID
#define SPARK_GLM53_FULL_TP16_MODEL_ID GLM53_FULL_MODEL_ID
#define SPARK_GLM53_FULL_TP16_MAX_ROWS 4096u

static void Glm53FullTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_PAGE_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FULL_TP16_STAGE_KV_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FULL_TP16_STAGE_INDEX_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_GLM53_FULL_TP16_STAGE_INDEX_LAYERS;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_GLM53_FULL_TP16_STAGE_LATENT + SPARK_GLM53_FULL_TP16_STAGE_ROPE;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_GLM53_FULL_TP16_STAGE_LAYERS;
	kv->capacity_request.compressed_dimension = SPARK_GLM53_FULL_TP16_STAGE_LATENT;
	kv->capacity_request.position_dimension = SPARK_GLM53_FULL_TP16_STAGE_ROPE;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.fp8_scale_block_size = SPARK_GLM53_FULL_TP16_STAGE_FP8_SCALE_BLOCK;
	kv->capacity_request.index_key_layer_count = SPARK_GLM53_FULL_TP16_STAGE_INDEX_LAYERS;
	kv->capacity_request.index_key_dimension = SPARK_GLM53_FULL_TP16_STAGE_INDEX_DIM;
	kv->capacity_request.index_key_bytes_per_scalar = 2u;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Glm53FullTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return rows;
}

static const SparkStageServingModel Glm53FullTp16StageServing =
{
	.adapter_id = SPARK_GLM53_FULL_TP16_ADAPTER_ID,
	.model_id = SPARK_GLM53_FULL_TP16_MODEL_ID,
	.model_revision = GLM53_FULL_MODEL_REVISION,
	.program = "glm53_full",
	.contract_sha256 = GLM53_FULL_CONTRACT_SHA256,
	.module_tag = GLM53_FULL_MODULE_TAG,
	.runner_model = SparkGlm53FullTp16StageModel,
	.tp_degree = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_GLM53_FULL_TP16_STAGE_LAYERS,
	.hidden = SPARK_GLM53_FULL_TP16_STAGE_HIDDEN,
	.linear_weight_codec = GLM53_FULL_LINEAR_CODEC,
	.expert_weight_codec = GLM53_FULL_EXPERT_CODEC,
	.max_rows = SPARK_GLM53_FULL_TP16_MAX_ROWS,
	.verify_rows = SPARK_GLM53_FULL_TP16_MAX_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 0u,
	.kv_layout = Glm53FullTp16ServingKvLayout,
	.collective_sequences = Glm53FullTp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Glm53FullTp16StageServing);
}
