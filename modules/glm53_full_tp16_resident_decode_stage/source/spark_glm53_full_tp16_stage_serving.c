#include "sparkpipe/spark_glm53_full_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef GLM53_FULL_CONTRACT_SHA256
#error "GLM53_FULL_CONTRACT_SHA256 must identify the pack contract"
#endif
#ifndef GLM53_FULL_MODEL_REVISION
#error "GLM53_FULL_MODEL_REVISION must name the checkpoint revision"
#endif

#define SPARK_GLM53_FULL_TP16_ADAPTER_ID "glm53-full-tp16"
#define SPARK_GLM53_FULL_TP16_MODEL_ID "zai-org/GLM-5.3"
#define SPARK_GLM53_FULL_TP16_MAX_ROWS 1024u
#define SPARK_GLM53_FULL_TP16_CHAIN_TOKENS 16u
#define Q SPARK_GLM53_FULL_TP16_STAGE_LAYERS

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
	kv->arena_bytes_per_scalar = 2u;
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_GLM53_FULL_TP16_STAGE_LAYERS;
	kv->capacity_request.compressed_dimension = SPARK_GLM53_FULL_TP16_STAGE_LATENT;
	kv->capacity_request.position_dimension = SPARK_GLM53_FULL_TP16_STAGE_ROPE;
	kv->capacity_request.bytes_per_scalar = 2u;
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

static const SparkModelServingAdapterDescriptor Glm53FullTp16ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_GLM53_FULL_TP16_ADAPTER_ID,
		SPARK_GLM53_FULL_TP16_MODEL_ID,
		GLM53_FULL_MODEL_REVISION,
		"glm53_full",
		GLM53_FULL_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_GLM53_FULL_TP16_STAGE_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_GLM53_FULL_TP16_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_GLM53_FULL_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_GLM53_FULL_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = 0u,
	.stage_layer_counts = { Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = 0u,
};

#undef Q

static const SparkStageServingModel Glm53FullTp16StageServing =
{
	.descriptor = &Glm53FullTp16ServingDescriptor,
	.runner_model = SparkGlm53FullTp16StageModel,
	.module_tag = "glm53_full_tp16_stage",
	.tp_degree = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = Glm53FullTp16ServingKvLayout,
	.collective_sequences = Glm53FullTp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Glm53FullTp16StageServing);
}
