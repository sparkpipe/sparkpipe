#include "sparkpipe/spark_ling3_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
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
#define SPARK_LING3_TP16_CHAIN_TOKENS 16u
#define SPARK_LING3_TP16_CHECKPOINT_TOKENS 16384u
#define Q SPARK_LING3_STAGE_LAYERS

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

static const SparkModelServingAdapterDescriptor Ling3ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_LING3_TP16_ADAPTER_ID,
		SPARK_LING3_TP16_MODEL_ID,
		SPARK_LING3_TP16_ADAPTER_ID,
		SPARK_LING3_TP16_PROGRAM,
		LING3_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATIVE_VERIFY,
	.stage_count = SPARK_LING3_STAGE_TP_DEGREE,
	.layer_count = SPARK_LING3_STAGE_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_LING3_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_LING3_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_LING3_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.stage_layer_counts = { Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_LING3_STAGE_PAGE_SLOTS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = SPARK_LING3_TP16_CHECKPOINT_TOKENS,
	.max_verify_row_count = SPARK_LING3_STAGE_VERIFY_ROWS,
};

#undef Q

static const SparkStageServingModel Ling3StageServing =
{
	.descriptor = &Ling3ServingDescriptor,
	.runner_model = SparkLing3StageModel,
	.module_tag = "ling3_stage",
	.tp_degree = SPARK_LING3_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = Ling3ServingKvLayout,
	.collective_sequences = Ling3ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Ling3StageServing);
}
