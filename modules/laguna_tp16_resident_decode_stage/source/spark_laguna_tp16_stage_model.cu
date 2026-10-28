#include "sparkpipe/spark_laguna_tp16_stage_model.h"
#include "gemma4_stage.cuh"

using LagunaStage = LagunaGeometry<SPARK_LAGUNA_TP16_STAGE_TP_DEGREE>;

static_assert(GemmaShape<LagunaStage>::SlidingKvGeometry::kSlotBytes == SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES &&
	GemmaShape<LagunaStage>::FullKvGeometry::kSlotBytes == SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES, "a KV slot is [K: 8 x 128][V: 8 x 128] bf16");
static_assert(GemmaShape<LagunaStage>::kSlidingFused == SPARK_LAGUNA_TP16_STAGE_SLIDING_FUSED_WIDTH &&
	GemmaShape<LagunaStage>::kFullFused == SPARK_LAGUNA_TP16_STAGE_FULL_FUSED_WIDTH,
	"sliding q|k|v is 80 padded heads + 2 x 8 KV heads of 128, full q|k|v is 48 + 2 x 8 heads of 128");
static_assert(GemmaShape<LagunaStage>::kSlidingFused % SPARK_LAGUNA_TP16_STAGE_TP_DEGREE == 0u &&
	GemmaShape<LagunaStage>::kFullFused % SPARK_LAGUNA_TP16_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");
static_assert(GemmaShape<LagunaStage>::kSlidingGateRows == 6u && GemmaShape<LagunaStage>::kFullGateRows == 4u,
	"a rank gates 5 sliding or 3 full heads; the gate projection is padded to an even row count");
static_assert(SPARK_LAGUNA_TP16_STAGE_SLIDING_HEADS % SPARK_LAGUNA_TP16_STAGE_KV_HEADS == 0u &&
	SPARK_LAGUNA_TP16_STAGE_SLIDING_HEADS / SPARK_LAGUNA_TP16_STAGE_KV_HEADS <= LM_GQA_SHARD_MAX_GROUP,
	"the padded sliding layer keeps whole GQA groups of 10 query heads per KV head");

const SparkStageRunnerModelInterface *SparkLagunaTp16StageModel(void)
{
	return GemmaModel<LagunaStage>::Interface();
}
