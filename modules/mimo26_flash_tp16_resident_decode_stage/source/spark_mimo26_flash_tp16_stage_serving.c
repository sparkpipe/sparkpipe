#include "sparkpipe/spark_mimo26_flash_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MIMO26_FLASH_TP16_CONTRACT_SHA256
#error "MIMO26_FLASH_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

static void Mimo26FlashTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitFull(kv, tp_rank, tp_degree, (uint64_t)SPARK_MIMO26_FLASH_TP16_STAGE_PAGE_SLOTS * SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_SLOT_BYTES,
		SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_LAYERS, SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_KV_HEADS,
		SPARK_MIMO26_FLASH_TP16_STAGE_HEAD_DIM, SPARK_MIMO26_FLASH_TP16_STAGE_VALUE_DIM);
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_MIMO26_FLASH_TP16_STAGE_PAGE_SLOTS * SPARK_MIMO26_FLASH_TP16_STAGE_FULL_SLOT_BYTES, SPARK_MIMO26_FLASH_TP16_STAGE_FULL_LAYERS);
}

static uint32_t Mimo26FlashTp16ServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint32_t fused = SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_FUSED > SPARK_MIMO26_FLASH_TP16_STAGE_FULL_FUSED ?
		SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_FUSED : SPARK_MIMO26_FLASH_TP16_STAGE_FULL_FUSED;
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, fused, SPARK_MIMO26_FLASH_TP16_STAGE_HIDDEN);
}

static const SparkStageServingModel Mimo26FlashTp16StageServing =
{
	.adapter_id = "mimo26_flash-tp16",
	.model_id = "XiaomiMiMo/MiMo-V2.6-Flash-RL",
	.model_revision = "3b38d063180c3e4aed9691fdc735f3d10b266ee4",
	.program = "mimo26_flash",
	.contract_sha256 = MIMO26_FLASH_TP16_CONTRACT_SHA256,
	.module_tag = "mimo26_flash_tp16_stage",
	.runner_model = SparkMimo26FlashTp16StageModel,
	.tp_degree = SPARK_MIMO26_FLASH_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MIMO26_FLASH_TP16_STAGE_LAYERS,
	.hidden = SPARK_MIMO26_FLASH_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.max_rows = 2048u,
	.verify_rows = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_MIMO26_FLASH_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_MIMO26_FLASH_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Mimo26FlashTp16ServingKvLayout,
	.collective_sequences = Mimo26FlashTp16ServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Mimo26FlashTp16StageServing)
