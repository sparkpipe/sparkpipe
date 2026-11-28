#include "sparkpipe/spark_kexaone2_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef KEXAONE2_TP16_CONTRACT_SHA256
#error "KEXAONE2_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

static void Kexaone2Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	const uint64_t page_bytes = (uint64_t)SPARK_KEXAONE2_TP16_STAGE_PAGE_SLOTS * SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES;
	SparkStageKvInitFull(kv, tp_rank, tp_degree, page_bytes, SPARK_KEXAONE2_TP16_STAGE_SLIDING_LAYERS,
		SPARK_KEXAONE2_TP16_STAGE_KV_HEADS, SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM, SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM);
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR, page_bytes, SPARK_KEXAONE2_TP16_STAGE_FULL_LAYERS);
}

static uint32_t Kexaone2Tp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH, SPARK_KEXAONE2_TP16_STAGE_HIDDEN);
}

static const SparkStageServingModel Kexaone2Tp16StageServing =
{
	.adapter_id = "kexaone2-tp16",
	.model_id = "LGAI-EXAONE/K-EXAONE-2.0-750B-A37B",
	.model_revision = "kexaone2-tp16",
	.program = "kexaone2",
	.contract_sha256 = KEXAONE2_TP16_CONTRACT_SHA256,
	.module_tag = "kexaone2_stage",
	.runner_model = SparkKexaone2Tp16StageModel,
	.tp_degree = SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_KEXAONE2_TP16_STAGE_LAYERS,
	.hidden = SPARK_KEXAONE2_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = 2048u,
	.verify_rows = 2048u,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_KEXAONE2_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Kexaone2Tp16ServingKvLayout,
	.collective_sequences = Kexaone2Tp16ServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Kexaone2Tp16StageServing)
