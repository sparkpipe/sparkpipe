#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef GLM53_FLASH_TP16_CONTRACT_SHA256
#error "GLM53_FLASH_TP16_CONTRACT_SHA256 must identify the model contract"
#endif
#ifndef GLM53_FLASH_EXPERT_CODEC
#error "GLM53_FLASH_EXPERT_CODEC must name the routed-expert weight codec of the packs this build serves"
#endif
#if !defined(GLM53_FLASH_ADAPTER_ID) || !defined(GLM53_FLASH_MODULE_TAG)
#error "GLM53_FLASH_ADAPTER_ID and GLM53_FLASH_MODULE_TAG must name this build's adapter and KV label"
#endif
#if !defined(GLM53_FLASH_MODEL_ID) || !defined(GLM53_FLASH_MODEL_REVISION)
#error "GLM53_FLASH_MODEL_ID and GLM53_FLASH_MODEL_REVISION must name the checkpoint this build's packs come from"
#endif
#if !defined(GLM53_FLASH_PROGRAM)
#error "GLM53_FLASH_PROGRAM must be defined by the Makefile"
#endif

static void Glm53FlashTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitCompressed(kv, tp_rank, tp_degree, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_GLM53_FLASH_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FLASH_TP16_STAGE_KV_SLOT_BYTES,
		SPARK_GLM53_FLASH_TP16_STAGE_LATENT, SPARK_GLM53_FLASH_TP16_STAGE_KV_LAYERS,
		SPARK_GLM53_FLASH_TP16_STAGE_LATENT, 0u, SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK);
	SparkStageKvSetIndex(kv, (uint64_t)SPARK_GLM53_FLASH_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FLASH_TP16_STAGE_INDEX_SLOT_BYTES,
		SPARK_GLM53_FLASH_TP16_STAGE_KV_LAYERS, SPARK_GLM53_FLASH_TP16_STAGE_INDEX_DIMENSION, 2u);
}

static const SparkStageServingModel Glm53FlashTp16StageServing =
{
	.adapter_id = GLM53_FLASH_ADAPTER_ID,
	.model_id = GLM53_FLASH_MODEL_ID,
	.model_revision = GLM53_FLASH_MODEL_REVISION,
	.program = GLM53_FLASH_PROGRAM,
	.contract_sha256 = GLM53_FLASH_TP16_CONTRACT_SHA256,
	.module_tag = GLM53_FLASH_MODULE_TAG,
	.runner_model = SparkGlm53FlashTp16StageModel,
	.tp_degree = SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_GLM53_FLASH_TP16_STAGE_LAYERS,
	.hidden = SPARK_GLM53_FLASH_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = GLM53_FLASH_EXPERT_CODEC,
	.max_rows = 4096u,
	.verify_rows = SPARK_GLM53_FLASH_TP16_VERIFY_ROWS,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL * SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_GLM53_FLASH_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 4096u,
	.kv_layout = Glm53FlashTp16ServingKvLayout,
	.collective_sequences = SparkStageCollectiveDoubleRows,
};

SPARK_STAGE_SERVING_EXPORT(Glm53FlashTp16StageServing)
