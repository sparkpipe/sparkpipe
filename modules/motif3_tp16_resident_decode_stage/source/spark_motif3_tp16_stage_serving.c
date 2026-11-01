#include "sparkpipe/spark_motif3_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MOTIF3_TP16_CONTRACT_SHA256
#error "MOTIF3_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_MOTIF3_TP16_ADAPTER_ID "motif3-tp16"
#define SPARK_MOTIF3_TP16_MODEL_ID "Motif-Technologies/Motif-3"
#define SPARK_MOTIF3_TP16_MODEL_REVISION "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"
#define SPARK_MOTIF3_TP16_MAX_ROWS 2048u
#define SPARK_MOTIF3_TP16_CHAIN_TOKENS 16u
#define Q SPARK_MOTIF3_TP16_STAGE_LAYERS

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

static const SparkModelServingAdapterDescriptor Motif3Tp16ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_MOTIF3_TP16_ADAPTER_ID,
		SPARK_MOTIF3_TP16_MODEL_ID,
		SPARK_MOTIF3_TP16_MODEL_REVISION,
		"motif3",
		MOTIF3_TP16_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = SPARK_MOTIF3_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MOTIF3_TP16_STAGE_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_MOTIF3_TP16_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_MOTIF3_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_MOTIF3_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = 0u,
	.stage_layer_counts = { Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_MOTIF3_TP16_STAGE_PAGE_SLOTS,
	.parallel_group_size = 0u,
};

#undef Q

static const SparkStageServingModel Motif3Tp16StageServing =
{
	.descriptor = &Motif3Tp16ServingDescriptor,
	.runner_model = SparkMotif3Tp16StageModel,
	.module_tag = "motif3_stage",
	.tp_degree = SPARK_MOTIF3_TP16_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = Motif3Tp16ServingKvLayout,
	.collective_sequences = Motif3Tp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Motif3Tp16StageServing);
}
