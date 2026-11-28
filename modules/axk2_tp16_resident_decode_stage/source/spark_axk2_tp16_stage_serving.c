#include "sparkpipe/spark_axk2_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef AXK2_CONTRACT_SHA256
#error "AXK2_CONTRACT_SHA256 must identify the pack contract"
#endif
#ifndef AXK2_MODEL_REVISION
#error "AXK2_MODEL_REVISION must name the checkpoint revision"
#endif

static void Axk2Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitCompressed(kv, tp_rank, tp_degree, SPARK_STAGE_KV_REGION_PAGE_MAJOR,
		(uint64_t)SPARK_AXK2_TP16_STAGE_PAGE_SLOTS * SPARK_AXK2_TP16_STAGE_KV_SLOT_BYTES,
		SPARK_AXK2_TP16_STAGE_LATENT + SPARK_AXK2_TP16_STAGE_ROPE, SPARK_AXK2_TP16_STAGE_LAYERS,
		SPARK_AXK2_TP16_STAGE_LATENT, SPARK_AXK2_TP16_STAGE_ROPE, SPARK_AXK2_TP16_STAGE_FP8_SCALE_BLOCK);
	SparkStageKvSetIndex(kv, (uint64_t)SPARK_AXK2_TP16_STAGE_PAGE_SLOTS * SPARK_AXK2_TP16_STAGE_INDEX_SLOT_BYTES,
		SPARK_AXK2_TP16_STAGE_INDEX_LAYERS, SPARK_AXK2_TP16_STAGE_INDEX_DIM, 2u);
}

static const SparkStageServingModel Axk2Tp16StageServing =
{
	.adapter_id = "axk2-tp16",
	.model_id = "skt/A.X-K2",
	.model_revision = AXK2_MODEL_REVISION,
	.program = "axk2",
	.contract_sha256 = AXK2_CONTRACT_SHA256,
	.module_tag = "axk2_tp16_stage",
	.runner_model = SparkAxk2Tp16StageModel,
	.tp_degree = SPARK_AXK2_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_AXK2_TP16_STAGE_LAYERS,
	.hidden = SPARK_AXK2_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.max_rows = 4096u,
	.verify_rows = 4096u,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.tree_slot_grain = SPARK_AXK2_TP16_STAGE_TP_DEGREE,
	.cache_block_token_count = SPARK_AXK2_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 0u,
	.kv_layout = Axk2Tp16ServingKvLayout,
	.collective_sequences = SparkStageCollectiveSameRows,
};

SPARK_STAGE_SERVING_EXPORT(Axk2Tp16StageServing)
