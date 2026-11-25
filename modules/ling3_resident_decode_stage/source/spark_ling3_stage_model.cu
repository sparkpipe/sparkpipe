#include "sparkpipe/spark_ling3_stage_model.h"
#include "ling_stage.cuh"

#ifdef LING31_FLASH
using Ling3Stage = Ling31FlashGeometry<SPARK_LING3_STAGE_TP_DEGREE>;
#else
using Ling3Stage = Ling3Geometry<SPARK_LING3_STAGE_TP_DEGREE>;
#endif

static_assert(Ling3Stage::kHidden == SPARK_LING3_STAGE_HIDDEN && Ling3Stage::kLayers == SPARK_LING3_STAGE_LAYERS &&
	Ling3Stage::kMlaHeads == SPARK_LING3_STAGE_MLA_HEADS, "stage header geometry matches runner geometry");
static_assert(LingShape<Ling3Stage>::Kv::kSlotBytes == SPARK_LING3_STAGE_KV_SLOT_BYTES, "an MLA slot is the normed latent and its rope part");
static_assert(LingShape<Ling3Stage>::kAttnLayers == SPARK_LING3_STAGE_MLA_LAYERS, "MLA layer count matches period");

const SparkStageRunnerModelInterface *SparkLing3StageModel(void)
{
	return LingModel<Ling3Stage>::Interface();
}
