#include "sparkpipe/spark_glm53_full_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef GLM53_FULL_CONTRACT_SHA256
#error "GLM53_FULL_CONTRACT_SHA256 must identify the pack contract"
#endif
#ifndef GLM53_FULL_MODEL_REVISION
#error "GLM53_FULL_MODEL_REVISION must name the checkpoint revision"
#endif
#ifndef GLM53_FULL_LINEAR_CODEC
#error "GLM53_FULL_LINEAR_CODEC must name the linear weight codec of the packs this build serves"
#endif
#ifndef GLM53_FULL_EXPERT_CODEC
#error "GLM53_FULL_EXPERT_CODEC must name the routed-expert weight codec of the packs this build serves"
#endif
#if !defined(GLM53_FULL_ADAPTER_ID) || !defined(GLM53_FULL_MODEL_ID) || !defined(GLM53_FULL_MODULE_TAG)
#error "GLM53_FULL_ADAPTER_ID, GLM53_FULL_MODEL_ID and GLM53_FULL_MODULE_TAG must name this build's adapter, served model and KV label"
#endif

static void Glm53FullTp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitCompressed(kv, tp_rank, tp_degree, SPARK_STAGE_KV_REGION_PAGE_MAJOR,
		(uint64_t)SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FULL_TP16_STAGE_KV_SLOT_BYTES,
		SPARK_GLM53_FULL_TP16_STAGE_LATENT + SPARK_GLM53_FULL_TP16_STAGE_ROPE, SPARK_GLM53_FULL_TP16_STAGE_LAYERS,
		SPARK_GLM53_FULL_TP16_STAGE_LATENT, SPARK_GLM53_FULL_TP16_STAGE_ROPE, SPARK_GLM53_FULL_TP16_STAGE_FP8_SCALE_BLOCK);
	SparkStageKvSetIndex(kv, (uint64_t)SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS * SPARK_GLM53_FULL_TP16_STAGE_INDEX_SLOT_BYTES,
		SPARK_GLM53_FULL_TP16_STAGE_INDEX_LAYERS, SPARK_GLM53_FULL_TP16_STAGE_INDEX_DIM, 2u);
}

static const SparkStageServingModel Glm53FullTp16StageServing =
{
	.adapter_id = GLM53_FULL_ADAPTER_ID,
	.model_id = GLM53_FULL_MODEL_ID,
	.model_revision = GLM53_FULL_MODEL_REVISION,
	.program = "glm53_full",
	.contract_sha256 = GLM53_FULL_CONTRACT_SHA256,
	.module_tag = GLM53_FULL_MODULE_TAG,
	.runner_model = SparkGlm53FullTp16StageModel,
	.tp_degree = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_GLM53_FULL_TP16_STAGE_LAYERS,
	.hidden = SPARK_GLM53_FULL_TP16_STAGE_HIDDEN,
	.linear_weight_codec = GLM53_FULL_LINEAR_CODEC,
	.expert_weight_codec = GLM53_FULL_EXPERT_CODEC,
	.max_rows = 16384u,
	.verify_rows = 4096u,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 0u,
	.kv_layout = Glm53FullTp16ServingKvLayout,
	.collective_sequences = SparkStageCollectiveSameRows,
};

SPARK_STAGE_SERVING_EXPORT(Glm53FullTp16StageServing)
