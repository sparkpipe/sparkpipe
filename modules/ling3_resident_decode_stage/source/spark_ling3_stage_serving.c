#include "sparkpipe/spark_ling3_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef LING3_CONTRACT_SHA256
#error "LING3_CONTRACT_SHA256 must identify the model contract"
#endif

#ifdef LING3_FIN
#define SPARK_LING3_TP16_ADAPTER_ID "ling3_fin-tp16"
#define SPARK_LING3_TP16_MODEL_ID "inclusionAI/Ling-3.0-flash-Fin"
#define SPARK_LING3_TP16_PROGRAM "ling3_fin"
#else
#define SPARK_LING3_TP16_ADAPTER_ID "ling3_flash-tp16"
#define SPARK_LING3_TP16_MODEL_ID "inclusionAI/Ling-3.0-flash"
#define SPARK_LING3_TP16_PROGRAM "ling3_flash"
#endif
#define SPARK_LING3_TP16_MAX_ROWS 2048u
#define SPARK_LING3_TP16_CHECKPOINT_TOKENS 16384u

static void Ling3ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_LING3_STAGE_PAGE_SLOTS * SPARK_LING3_STAGE_KV_SLOT_BYTES;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_LING3_STAGE_LATENT + SPARK_LING3_STAGE_ROPE;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_LING3_STAGE_MLA_LAYERS;
	kv->capacity_request.compressed_dimension = SPARK_LING3_STAGE_LATENT;
	kv->capacity_request.position_dimension = SPARK_LING3_STAGE_ROPE;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Ling3ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return rows + 1u;
}

static const SparkStageServingModel Ling3StageServing =
{
	.adapter_id = SPARK_LING3_TP16_ADAPTER_ID,
	.model_id = SPARK_LING3_TP16_MODEL_ID,
	.model_revision = SPARK_LING3_TP16_ADAPTER_ID,
	.program = SPARK_LING3_TP16_PROGRAM,
	.contract_sha256 = LING3_CONTRACT_SHA256,
	.module_tag = "ling3_stage",
	.runner_model = SparkLing3StageModel,
	.tp_degree = SPARK_LING3_STAGE_TP_DEGREE,
	.layer_count = SPARK_LING3_STAGE_LAYERS,
	.hidden = SPARK_LING3_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = SPARK_LING3_TP16_MAX_ROWS,
	.verify_rows = SPARK_LING3_STAGE_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_LING3_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_LING3_TP16_CHECKPOINT_TOKENS,
	.kv_layout = Ling3ServingKvLayout,
	.collective_sequences = Ling3ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Ling3StageServing);
}
