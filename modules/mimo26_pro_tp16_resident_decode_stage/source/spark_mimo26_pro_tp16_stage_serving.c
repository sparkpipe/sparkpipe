#include "sparkpipe/spark_mimo26_pro_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MIMO26_PRO_TP16_CONTRACT_SHA256
#error "MIMO26_PRO_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_MIMO26_PRO_TP16_ADAPTER_ID "mimo26_pro-tp16"
#define SPARK_MIMO26_PRO_TP16_MODEL_ID "XiaomiMiMo/MiMo-V2.6-Pro-RL"
#define SPARK_MIMO26_PRO_TP16_MODEL_REVISION "54b10491b1811c76aa9681a9d0ff872396a4064c"
#define SPARK_MIMO26_PRO_TP16_MAX_ROWS 2048u
#define SPARK_MIMO26_PRO_TP16_CHAIN_TOKENS 16u
#define SPARK_MIMO26_PRO_TP16_CHECKPOINT_TOKENS 16384u
#define Q SPARK_MIMO26_PRO_TP16_STAGE_LAYERS

static void Mimo26ProTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_MIMO26_PRO_TP16_STAGE_PAGE_SLOTS * SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_SLOT_BYTES;
	kv->regions[1].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[1].layer_page_bytes = (uint64_t)SPARK_MIMO26_PRO_TP16_STAGE_PAGE_SLOTS * SPARK_MIMO26_PRO_TP16_STAGE_FULL_SLOT_BYTES;
	kv->regions[1].layer_count = SPARK_MIMO26_PRO_TP16_STAGE_FULL_LAYERS;
	kv->arena_kv_head_count = SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_KV_HEADS;
	kv->arena_head_dim = SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM + SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM;
	kv->arena_bytes_per_scalar = 2u;
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_LAYERS;
	kv->capacity_request.head_count = SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM;
	kv->capacity_request.bytes_per_scalar = 2u;
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Mimo26ProTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint32_t fused = SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_FUSED > SPARK_MIMO26_PRO_TP16_STAGE_FULL_FUSED ? SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_FUSED : SPARK_MIMO26_PRO_TP16_STAGE_FULL_FUSED;
	const uint64_t gathered = ((uint64_t)rows * fused + SPARK_MIMO26_PRO_TP16_STAGE_HIDDEN - 1u) / SPARK_MIMO26_PRO_TP16_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkModelServingAdapterDescriptor Mimo26ProTp16ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_MIMO26_PRO_TP16_ADAPTER_ID,
		SPARK_MIMO26_PRO_TP16_MODEL_ID,
		SPARK_MIMO26_PRO_TP16_MODEL_REVISION,
		"mimo26_pro",
		MIMO26_PRO_TP16_CONTRACT_SHA256),
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = SPARK_MIMO26_PRO_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MIMO26_PRO_TP16_STAGE_LAYERS,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_MIMO26_PRO_TP16_STAGE_HIDDEN,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_input_row_count = SPARK_MIMO26_PRO_TP16_MAX_ROWS,
	.max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES,
	.max_output_token_count = SPARK_MIMO26_PRO_TP16_CHAIN_TOKENS,
	.max_speculative_token_count = SPARK_MIMO26_PRO_TP16_CHAIN_TOKENS - 1u,
	.stage_layer_counts = { Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q },
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = SPARK_MIMO26_PRO_TP16_STAGE_PAGE_SLOTS,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = SPARK_MIMO26_PRO_TP16_CHECKPOINT_TOKENS,
};

#undef Q

static const SparkStageServingModel Mimo26ProTp16StageServing =
{
	.descriptor = &Mimo26ProTp16ServingDescriptor,
	.runner_model = SparkMimo26ProTp16StageModel,
	.module_tag = "mimo26_pro_tp16_stage",
	.tp_degree = SPARK_MIMO26_PRO_TP16_STAGE_TP_DEGREE,
	.program_id = 1u,
	.kv_layout = Mimo26ProTp16ServingKvLayout,
	.collective_sequences = Mimo26ProTp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Mimo26ProTp16StageServing);
}
