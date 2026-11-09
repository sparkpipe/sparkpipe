#include "sparkpipe/spark_hy4_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef HY4_TP16_CONTRACT_SHA256
#error "HY4_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_HY4_TP16_ADAPTER_ID "hy4-tp16"
#define SPARK_HY4_TP16_MODEL_ID "tencent/Hy4-preview-FP8"
#define SPARK_HY4_TP16_MAX_ROWS 2048u
#define SPARK_HY4_TP16_CHECKPOINT_TOKENS 16384u

static void Hy4Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_HY4_TP16_STAGE_PAGE_SLOTS * SPARK_HY4_TP16_STAGE_KV_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_HY4_TP16_STAGE_PAGE_SLOTS * SPARK_HY4_TP16_STAGE_INDEX_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_HY4_TP16_STAGE_INDEX_LAYERS;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_HY4_TP16_STAGE_LATENT + SPARK_HY4_TP16_STAGE_ROPE;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_HY4_TP16_STAGE_LAYERS;
	kv->capacity_request.compressed_dimension = SPARK_HY4_TP16_STAGE_LATENT;
	kv->capacity_request.position_dimension = SPARK_HY4_TP16_STAGE_ROPE;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.index_key_layer_count = SPARK_HY4_TP16_STAGE_INDEX_LAYERS;
	kv->capacity_request.index_key_dimension = SPARK_HY4_TP16_STAGE_INDEX_DIM;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Hy4Tp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return 2u * rows + 1u;
}

static const SparkStageServingModel Hy4Tp16StageServing =
{
	.adapter_id = SPARK_HY4_TP16_ADAPTER_ID,
	.model_id = SPARK_HY4_TP16_MODEL_ID,
	.model_revision = SPARK_HY4_TP16_ADAPTER_ID,
	.program = "hy4",
	.contract_sha256 = HY4_TP16_CONTRACT_SHA256,
	.module_tag = "hy4_tp16_stage",
	.runner_model = SparkHy4Tp16StageModel,
	.tp_degree = SPARK_HY4_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_HY4_TP16_STAGE_LAYERS,
	.hidden = SPARK_HY4_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.max_rows = SPARK_HY4_TP16_MAX_ROWS,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_HY4_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_HY4_TP16_CHECKPOINT_TOKENS,
	.kv_layout = Hy4Tp16ServingKvLayout,
	.collective_sequences = Hy4Tp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Hy4Tp16StageServing);
}
