#include "sparkpipe/spark_hy4_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef HY4_TP16_CONTRACT_SHA256
#error "HY4_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

#ifdef HY4_FP8
#define SPARK_HY4_TP16_ADAPTER_ID "hy4_fp8-tp16"
#define SPARK_HY4_TP16_MODEL_ID "tencent/Hy4-FP8"
#define SPARK_HY4_TP16_MODEL_REVISION "d063b5853ecf56e0776266dab28305fc17a137cc"
#define SPARK_HY4_TP16_PROGRAM "hy4_fp8"
#define SPARK_HY4_TP16_MODULE_TAG "hy4_fp8_tp16_stage"
#else
#define SPARK_HY4_TP16_ADAPTER_ID "hy4-tp16"
#define SPARK_HY4_TP16_MODEL_ID "tencent/Hy4-preview-FP8"
#define SPARK_HY4_TP16_MODEL_REVISION SPARK_HY4_TP16_ADAPTER_ID
#define SPARK_HY4_TP16_PROGRAM "hy4"
#define SPARK_HY4_TP16_MODULE_TAG "hy4_tp16_stage"
#endif

static void Hy4Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitCompressed(kv, tp_rank, tp_degree, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_HY4_TP16_STAGE_PAGE_SLOTS * SPARK_HY4_TP16_STAGE_KV_SLOT_BYTES,
		SPARK_HY4_TP16_STAGE_LATENT + SPARK_HY4_TP16_STAGE_ROPE, SPARK_HY4_TP16_STAGE_LAYERS,
		SPARK_HY4_TP16_STAGE_LATENT, SPARK_HY4_TP16_STAGE_ROPE, 0u);
	SparkStageKvSetIndex(kv, (uint64_t)SPARK_HY4_TP16_STAGE_PAGE_SLOTS * SPARK_HY4_TP16_STAGE_INDEX_SLOT_BYTES,
		SPARK_HY4_TP16_STAGE_INDEX_LAYERS, SPARK_HY4_TP16_STAGE_INDEX_DIM, 0u);
}

static const SparkStageServingModel Hy4Tp16StageServing =
{
	.adapter_id = SPARK_HY4_TP16_ADAPTER_ID,
	.model_id = SPARK_HY4_TP16_MODEL_ID,
	.model_revision = SPARK_HY4_TP16_MODEL_REVISION,
	.program = SPARK_HY4_TP16_PROGRAM,
	.contract_sha256 = HY4_TP16_CONTRACT_SHA256,
	.module_tag = SPARK_HY4_TP16_MODULE_TAG,
	.runner_model = SparkHy4Tp16StageModel,
	.tp_degree = SPARK_HY4_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_HY4_TP16_STAGE_LAYERS,
	.hidden = SPARK_HY4_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.max_rows = 2048u,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_HY4_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_HY4_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Hy4Tp16ServingKvLayout,
	.collective_sequences = SparkStageCollectiveDoubleRows,
};

SPARK_STAGE_SERVING_EXPORT(Hy4Tp16StageServing)
