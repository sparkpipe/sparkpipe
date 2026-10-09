#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_k3_kv_geometry.h"
#include "sparkpipe/spark_k3_llm_defines.h"
#include "sparkpipe/spark_k3_model.h"
#include "sparkpipe/spark_k3_serving_adapter.h"
#include "sparkpipe/spark_k3_stage_model.h"
#include "sparkpipe/spark_k3_tp_sequences.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#include "spark_k3_dspark_format.h"

#if SPARK_K3_SERVING_TOPOLOGY == 404
#define SPARK_K3_SERVING_ADAPTER_ID "k3-tp4pp4"
#define SPARK_K3_SERVING_TP_DEGREE 4u
#define SPARK_K3_SERVING_PARALLEL_GROUP_SIZE 4u
#define SPARK_K3_SERVING_CAPABILITIES \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP)
#define SPARK_K3_SERVING_STAGE_LAYERS \
	{ \
		SPARK_K3_PP_STAGE_LAYERS(0u), SPARK_K3_PP_STAGE_LAYERS(0u), \
		SPARK_K3_PP_STAGE_LAYERS(0u), SPARK_K3_PP_STAGE_LAYERS(0u), \
		SPARK_K3_PP_STAGE_LAYERS(1u), SPARK_K3_PP_STAGE_LAYERS(1u), \
		SPARK_K3_PP_STAGE_LAYERS(1u), SPARK_K3_PP_STAGE_LAYERS(1u), \
		SPARK_K3_PP_STAGE_LAYERS(2u), SPARK_K3_PP_STAGE_LAYERS(2u), \
		SPARK_K3_PP_STAGE_LAYERS(2u), SPARK_K3_PP_STAGE_LAYERS(2u), \
		SPARK_K3_PP_STAGE_LAYERS(3u), SPARK_K3_PP_STAGE_LAYERS(3u), \
		SPARK_K3_PP_STAGE_LAYERS(3u), SPARK_K3_PP_STAGE_LAYERS(3u) \
	}
#define SPARK_K3_SERVING_SIDEBAND_KIND \
	SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_RESIDUAL_BANK
#define SPARK_K3_SERVING_SIDEBAND_BYTES SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW
#elif SPARK_K3_SERVING_TOPOLOGY == 16
#define SPARK_K3_SERVING_ADAPTER_ID "k3-tp16"
#define SPARK_K3_SERVING_TP_DEGREE 16u
#define SPARK_K3_SERVING_PARALLEL_GROUP_SIZE 0u
#define SPARK_K3_SERVING_CAPABILITIES \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN)
#define L SPARK_K3_MODEL_LAYER_COUNT
#define SPARK_K3_SERVING_STAGE_LAYERS \
	{ L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L }
#define SPARK_K3_SERVING_SIDEBAND_KIND 0u
#define SPARK_K3_SERVING_SIDEBAND_BYTES 0u
#else
#error "SPARK_K3_SERVING_TOPOLOGY must be 404 (TP4xPP4) or 16 (TP16)"
#endif

#define SPARK_K3_SERVING_CONTRACT_SHA256 "318d979200eb3c6784be6f932febe14832b48df53a1520a73af2f03bd39bb217"
#define SPARK_K3_SERVING_MAX_ROWS 2048u

static void K3ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_K3_KV_PAGE_SLOTS *
		SPARK_K3_MODEL_MLA_KV_A_DIMENSION * SPARK_K3_KV_BYTES_PER_SCALAR;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_K3_MODEL_MLA_KV_A_DIMENSION;
	kv->arena_bytes_per_scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	SparkK3KvFillCapacityRequest(&kv->capacity_request);
	if ( tp_degree > 1u )
		kv->context_shard = SparkK3KvShardContext(tp_rank, tp_degree);
}

static uint32_t K3ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	return SparkK3TpSequenceCapacity(rows, tp_degree);
}

static const SparkModelServingAdapterDescriptor K3ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_K3_SERVING_ADAPTER_ID,
		SPARK_K3_MODEL_SOURCE_ID,
		SPARK_K3_SERVING_ADAPTER_ID,
		"k3",
		SPARK_K3_SERVING_CONTRACT_SHA256),
	.capability_flags = SPARK_K3_SERVING_CAPABILITIES |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = 16u,
	.layer_count = SPARK_K3_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_K3_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = 16u,
	.max_input_row_count = SPARK_K3_SERVING_MAX_ROWS,
	.max_resident_sequence_count = 16u,
	.max_output_token_count = 16u,
	.max_speculative_token_count = SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT,
	.stage_layer_counts = SPARK_K3_SERVING_STAGE_LAYERS,
	.boundary_sideband_kinds =
	{
		SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND,
		SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND,
		SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND, SPARK_K3_SERVING_SIDEBAND_KIND
	},
	.boundary_sideband_bytes_per_sequence =
	{
		SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES,
		SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES,
		SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES, SPARK_K3_SERVING_SIDEBAND_BYTES
	},
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_K3_KV_PAGE_SLOTS,
	.parallel_group_size = SPARK_K3_SERVING_PARALLEL_GROUP_SIZE,
	.cache_checkpoint_token_count = SPARK_K3_KV_CHECKPOINT_TOKENS,
};

#if SPARK_K3_SERVING_TOPOLOGY == 16
#undef L
#endif

static const SparkStageServingModel K3StageServing =
{
	.descriptor = &K3ServingDescriptor,
	.runner_model = SparkK3StageModel,
	.module_tag = "k3_stage",
	.tp_degree = SPARK_K3_SERVING_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = K3ServingKvLayout,
	.collective_sequences = K3ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&K3StageServing);
}
