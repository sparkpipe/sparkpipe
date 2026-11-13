#include "sparkpipe/spark_axk2_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef AXK2_CONTRACT_SHA256
#error "AXK2_CONTRACT_SHA256 must identify the pack contract"
#endif
#ifndef AXK2_MODEL_REVISION
#error "AXK2_MODEL_REVISION must name the checkpoint revision"
#endif

#define SPARK_AXK2_TP16_ADAPTER_ID "axk2-tp16"
#define SPARK_AXK2_TP16_MODEL_ID "skt/A.X-K2"
#define SPARK_AXK2_TP16_MAX_ROWS 4096u

static void Axk2Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_PAGE_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_AXK2_TP16_STAGE_PAGE_SLOTS * SPARK_AXK2_TP16_STAGE_KV_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_AXK2_TP16_STAGE_PAGE_SLOTS * SPARK_AXK2_TP16_STAGE_INDEX_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_AXK2_TP16_STAGE_INDEX_LAYERS;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_AXK2_TP16_STAGE_LATENT + SPARK_AXK2_TP16_STAGE_ROPE;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_AXK2_TP16_STAGE_LAYERS;
	kv->capacity_request.compressed_dimension = SPARK_AXK2_TP16_STAGE_LATENT;
	kv->capacity_request.position_dimension = SPARK_AXK2_TP16_STAGE_ROPE;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.fp8_scale_block_size = SPARK_AXK2_TP16_STAGE_FP8_SCALE_BLOCK;
	kv->capacity_request.index_key_layer_count = SPARK_AXK2_TP16_STAGE_INDEX_LAYERS;
	kv->capacity_request.index_key_dimension = SPARK_AXK2_TP16_STAGE_INDEX_DIM;
	kv->capacity_request.index_key_bytes_per_scalar = 2u;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Axk2Tp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return rows;
}

static const SparkStageServingModel Axk2Tp16StageServing =
{
	.adapter_id = SPARK_AXK2_TP16_ADAPTER_ID,
	.model_id = SPARK_AXK2_TP16_MODEL_ID,
	.model_revision = AXK2_MODEL_REVISION,
	.program = "axk2",
	.contract_sha256 = AXK2_CONTRACT_SHA256,
	.module_tag = "axk2_tp16_stage",
	.runner_model = SparkAxk2Tp16StageModel,
	.tp_degree = SPARK_AXK2_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_AXK2_TP16_STAGE_LAYERS,
	.hidden = SPARK_AXK2_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.max_rows = SPARK_AXK2_TP16_MAX_ROWS,
	.verify_rows = SPARK_AXK2_TP16_MAX_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_AXK2_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 0u,
	.kv_layout = Axk2Tp16ServingKvLayout,
	.collective_sequences = Axk2Tp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Axk2Tp16StageServing);
}
