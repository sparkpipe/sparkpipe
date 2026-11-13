#include "sparkpipe/spark_muse_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MUSE_TP16_CONTRACT_SHA256
#error "MUSE_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_MUSE_TP16_ADAPTER_ID "muse_glimmer-tp16"
#define SPARK_MUSE_TP16_MODEL_ID "meta-models/Muse-Glimmer-30B"
#define SPARK_MUSE_TP16_MAX_ROWS 2048u
#define SPARK_MUSE_TP16_CHECKPOINT_TOKENS 16384u

static void MuseTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_MUSE_TP16_STAGE_PAGE_SLOTS * SPARK_MUSE_TP16_STAGE_SLIDING_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_MUSE_TP16_STAGE_PAGE_SLOTS * SPARK_MUSE_TP16_STAGE_FULL_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_MUSE_TP16_STAGE_FULL_LAYERS;
	kv->arena_kv_head_count = SPARK_MUSE_TP16_STAGE_SLIDING_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_MUSE_TP16_STAGE_SLIDING_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_MUSE_TP16_STAGE_SLIDING_LAYERS;
	kv->capacity_request.head_count = SPARK_MUSE_TP16_STAGE_SLIDING_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_MUSE_TP16_STAGE_SLIDING_DIM;
	kv->capacity_request.value_head_dimension = SPARK_MUSE_TP16_STAGE_SLIDING_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t MuseTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * SPARK_MUSE_TP16_STAGE_FULL_FUSED_WIDTH + SPARK_MUSE_TP16_STAGE_HIDDEN - 1u) /
		SPARK_MUSE_TP16_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkStageServingModel MuseTp16StageServing =
{
	.adapter_id = SPARK_MUSE_TP16_ADAPTER_ID,
	.model_id = SPARK_MUSE_TP16_MODEL_ID,
	.model_revision = SPARK_MUSE_TP16_ADAPTER_ID,
	.program = "muse_glimmer",
	.contract_sha256 = MUSE_TP16_CONTRACT_SHA256,
	.module_tag = "muse_glimmer_stage",
	.runner_model = SparkMuseTp16StageModel,
	.tp_degree = SPARK_MUSE_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MUSE_TP16_STAGE_LAYERS,
	.hidden = SPARK_MUSE_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = SPARK_MUSE_TP16_MAX_ROWS,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_MUSE_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_MUSE_TP16_CHECKPOINT_TOKENS,
	.kv_layout = MuseTp16ServingKvLayout,
	.collective_sequences = MuseTp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&MuseTp16StageServing);
}
