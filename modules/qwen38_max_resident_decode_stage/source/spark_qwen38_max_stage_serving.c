#include "sparkpipe/spark_qwen38_max_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef QWEN38_MAX_CONTRACT_SHA256
#error "QWEN38_MAX_CONTRACT_SHA256 must identify the model contract"
#endif

static void Qwen38MaxServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitFull(kv, tp_rank, tp_degree, (uint64_t)SPARK_QWEN38_MAX_STAGE_PAGE_SLOTS * SPARK_QWEN38_MAX_STAGE_KV_SLOT_BYTES,
		SPARK_QWEN38_MAX_STAGE_KV_LAYERS, SPARK_QWEN38_MAX_STAGE_KV_HEADS, SPARK_QWEN38_MAX_STAGE_HEAD_DIM, SPARK_QWEN38_MAX_STAGE_HEAD_DIM);
}

static uint32_t Qwen38MaxServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, SPARK_QWEN38_MAX_STAGE_QKV_WIDTH, SPARK_QWEN38_MAX_STAGE_HIDDEN);
}

static const SparkStageServingModel Qwen38MaxStageServing =
{
	.adapter_id = "qwen38_max-tp16",
	.model_id = "Qwen/Qwen3.8-Max",
	.model_revision = "qwen38_max-tp16",
	.program = "qwen38_max",
	.contract_sha256 = QWEN38_MAX_CONTRACT_SHA256,
	.module_tag = "qwen38_max_stage",
	.runner_model = SparkQwen38MaxStageModel,
	.tp_degree = SPARK_QWEN38_MAX_STAGE_TP_DEGREE,
	.layer_count = SPARK_QWEN38_MAX_STAGE_LAYERS,
	.hidden = SPARK_QWEN38_MAX_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_NVFP4_E2M1,
	.max_rows = 2048u,
	.verify_rows = SPARK_QWEN38_MAX_STAGE_VERIFY_ROWS,
	.tree_slot_grain = SPARK_QWEN38_MAX_STAGE_TP_DEGREE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_QWEN38_MAX_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Qwen38MaxServingKvLayout,
	.collective_sequences = Qwen38MaxServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Qwen38MaxStageServing)
