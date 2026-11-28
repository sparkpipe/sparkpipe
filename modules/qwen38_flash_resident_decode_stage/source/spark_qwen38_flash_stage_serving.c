#include "sparkpipe/spark_qwen38_flash_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef QWEN38_FLASH_CONTRACT_SHA256
#error "QWEN38_FLASH_CONTRACT_SHA256 must identify the model contract"
#endif
#ifndef QWEN38_FLASH_EXPERT_CODEC
#error "QWEN38_FLASH_EXPERT_CODEC must name the routed-expert weight codec of the packs this build serves"
#endif
#if !defined(QWEN38_FLASH_ADAPTER_ID) || !defined(QWEN38_FLASH_MODULE_TAG)
#error "QWEN38_FLASH_ADAPTER_ID and QWEN38_FLASH_MODULE_TAG must name this build's adapter and KV label"
#endif

static void Qwen38FlashServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitFull(kv, tp_rank, tp_degree, (uint64_t)SPARK_QWEN38_FLASH_STAGE_PAGE_SLOTS * SPARK_QWEN38_FLASH_STAGE_KV_SLOT_BYTES,
		SPARK_QWEN38_FLASH_STAGE_KV_LAYERS, SPARK_QWEN38_FLASH_STAGE_KV_HEADS, SPARK_QWEN38_FLASH_STAGE_HEAD_DIM, SPARK_QWEN38_FLASH_STAGE_HEAD_DIM);
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)(SPARK_QWEN38_FLASH_STAGE_PAGE_SLOTS / SPARK_QWEN38_FLASH_STAGE_INDEX_RATIO) * SPARK_QWEN38_FLASH_STAGE_INDEX_DIM * 2u,
		SPARK_QWEN38_FLASH_STAGE_KV_LAYERS);
}

static uint32_t Qwen38FlashServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t attention = (uint64_t)rows * SPARK_QWEN38_FLASH_STAGE_QKV_WIDTH;
	const uint64_t candidates = (uint64_t)rows * tp_degree * SPARK_QWEN38_FLASH_STAGE_INDEX_BLOCKS * 4u;
	const uint64_t gathered = ((attention > candidates ? attention : candidates) + SPARK_QWEN38_FLASH_STAGE_HIDDEN - 1u) /
		SPARK_QWEN38_FLASH_STAGE_HIDDEN;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkStageServingModel Qwen38FlashStageServing =
{
	.adapter_id = QWEN38_FLASH_ADAPTER_ID,
	.model_id = "Qwen/Qwen3.8-Flash-Next",
	.model_revision = QWEN38_FLASH_ADAPTER_ID,
	.program = "qwen38_flash",
	.contract_sha256 = QWEN38_FLASH_CONTRACT_SHA256,
	.module_tag = QWEN38_FLASH_MODULE_TAG,
	.runner_model = SparkQwen38FlashStageModel,
	.tp_degree = SPARK_QWEN38_FLASH_STAGE_TP_DEGREE,
	.layer_count = SPARK_QWEN38_FLASH_STAGE_LAYERS,
	.hidden = SPARK_QWEN38_FLASH_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = QWEN38_FLASH_EXPERT_CODEC,
	.max_rows = 2048u,
	.verify_rows = SPARK_QWEN38_FLASH_STAGE_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_QWEN38_FLASH_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Qwen38FlashServingKvLayout,
	.collective_sequences = Qwen38FlashServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Qwen38FlashStageServing)
