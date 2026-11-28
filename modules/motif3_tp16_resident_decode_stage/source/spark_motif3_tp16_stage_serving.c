#include "sparkpipe/spark_motif3_tp16_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef MOTIF3_TP16_CONTRACT_SHA256
#error "MOTIF3_TP16_CONTRACT_SHA256 must identify the model contract"
#endif

static void Motif3Tp16ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitCompressed(kv, tp_rank, tp_degree, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_MOTIF3_TP16_STAGE_PAGE_SLOTS * SPARK_MOTIF3_TP16_STAGE_SLOT_BYTES,
		SPARK_MOTIF3_TP16_STAGE_SLOT, SPARK_MOTIF3_TP16_STAGE_LAYERS,
		SPARK_MOTIF3_TP16_STAGE_LATENT, SPARK_MOTIF3_TP16_STAGE_ROPE, 0u);
}

static const SparkStageServingModel Motif3Tp16StageServing =
{
	.adapter_id = "motif3-tp16",
	.model_id = "Motif-Technologies/Motif-3",
	.model_revision = "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a",
	.program = "motif3",
	.contract_sha256 = MOTIF3_TP16_CONTRACT_SHA256,
	.module_tag = "motif3_stage",
	.runner_model = SparkMotif3Tp16StageModel,
	.tp_degree = SPARK_MOTIF3_TP16_STAGE_TP_DEGREE,
	.layer_count = SPARK_MOTIF3_TP16_STAGE_LAYERS,
	.hidden = SPARK_MOTIF3_TP16_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = 2048u,
	.verify_rows = 0u,
	.speculative_tokens = 0u,
	.cache_block_token_count = SPARK_MOTIF3_TP16_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 0u,
	.kv_layout = Motif3Tp16ServingKvLayout,
	.collective_sequences = SparkStageCollectiveDoubleRows,
};

SPARK_STAGE_SERVING_EXPORT(Motif3Tp16StageServing)
