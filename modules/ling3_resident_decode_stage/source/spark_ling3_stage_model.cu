#include "sparkpipe/spark_ling3_stage_model.h"
#include "ling_stage.cuh"

using Ling3Stage = Ling3Geometry<SPARK_LING3_STAGE_TP_DEGREE>;

static_assert(LingShape<Ling3Stage>::Kv::kSlotBytes == SPARK_LING3_STAGE_KV_SLOT_BYTES, "an MLA slot is the normed latent and its rope part");
static_assert(LingShape<Ling3Stage>::kAttnLayers == SPARK_LING3_STAGE_MLA_LAYERS, "every sixth layer is MLA");

const SparkStageRunnerModelInterface *SparkLing3StageModel(void)
{
	return LingModel<Ling3Stage>::Interface();
}
