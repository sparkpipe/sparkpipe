#include "sparkpipe/spark_mimo26_pro_tp16_stage_model.h"
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MIMO26_PRO_TP16_CONTRACT_SHA256
#error "MIMO26_PRO_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#define SPARK_MIMO26_PRO_TP16_ADAPTER_ID "mimo26_pro-tp16"
#define SPARK_MIMO26_PRO_TP16_MODEL_ID "XiaomiMiMo/MiMo-V2.6-Pro-RL"
#define SPARK_MIMO26_PRO_TP16_MODEL_REVISION "54b10491b1811c76aa9681a9d0ff872396a4064c"
#define SPARK_MIMO26_PRO_TP16_MAX_ROWS 2048u
#define SPARK_MIMO26_PRO_TP16_CHECKPOINT_TOKENS 16384u

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
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_LAYERS;
	kv->capacity_request.head_count = SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
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

static const SparkStageServingModel Mimo26ProTp16StageServing =
{
	.adapter_id = SPARK_MIMO26_PRO_TP16_ADAPTER_ID,
	.model_id = SPARK_MIMO26_PRO_TP16_MODEL_ID,
	.model_revision = SPARK_MIMO26_PRO_TP16_MODEL_REVISION,
	.program = "mimo26_pro",
	.contract_sha256 = MIMO26_PRO_TP16_CONTRACT_SHA256,
	.module_tag = "mimo26_pro_tp16_stage",
	.runner_model = SparkMimo26ProTp16StageModel,
	.tp_degree = SPARK_MIMO26_PRO_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MIMO26_PRO_TP16_STAGE_LAYERS,
	.hidden = SPARK_MIMO26_PRO_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.max_rows = SPARK_MIMO26_PRO_TP16_MAX_ROWS,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_MIMO26_PRO_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_MIMO26_PRO_TP16_CHECKPOINT_TOKENS,
	.kv_layout = Mimo26ProTp16ServingKvLayout,
	.collective_sequences = Mimo26ProTp16ServingCollectiveSequences,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Mimo26ProTp16StageServing);
}
