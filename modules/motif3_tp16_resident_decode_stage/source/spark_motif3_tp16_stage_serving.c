#include "sparkpipe/spark_motif3_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MOTIF3_TP16_CONTRACT_SHA256
#error "MOTIF3_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_MOTIF3_TP16_ADAPTER_ID "motif3-tp16"
#define SPARK_MOTIF3_TP16_MODEL_ID "Motif-Technologies/Motif-3"
#define SPARK_MOTIF3_TP16_MODEL_REVISION "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"
#define SPARK_MOTIF3_TP16_MAX_ROWS 2048u

static void Motif3Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_MOTIF3_TP16_STAGE_PAGE_SLOTS * SPARK_MOTIF3_TP16_STAGE_SLOT_BYTES;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_MOTIF3_TP16_STAGE_SLOT;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_MOTIF3_TP16_STAGE_LAYERS;
	kv->capacity_request.compressed_dimension = SPARK_MOTIF3_TP16_STAGE_LATENT;
	kv->capacity_request.position_dimension = SPARK_MOTIF3_TP16_STAGE_ROPE;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Motif3Tp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return 2u * rows + 1u;
}

static const SparkStageServingModel Motif3Tp16StageServing =
{
	.adapter_id = SPARK_MOTIF3_TP16_ADAPTER_ID,
	.model_id = SPARK_MOTIF3_TP16_MODEL_ID,
	.model_revision = SPARK_MOTIF3_TP16_MODEL_REVISION,
	.program = "motif3",
	.contract_sha256 = MOTIF3_TP16_CONTRACT_SHA256,
	.module_tag = "motif3_stage",
	.runner_model = SparkMotif3Tp16StageModel,
	.tp_degree = SPARK_MOTIF3_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MOTIF3_TP16_STAGE_LAYERS,
	.hidden = SPARK_MOTIF3_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = SPARK_MOTIF3_TP16_MAX_ROWS,
	.verify_rows = 0u,
	.speculative_tokens = 0u,
	.cache_block_token_count = SPARK_MOTIF3_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 0u,
	.kv_layout = Motif3Tp16ServingKvLayout,
	.collective_sequences = Motif3Tp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Motif3Tp16StageServing);
}
