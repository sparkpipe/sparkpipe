#include "sparkpipe/spark_gemma4_26b_stage_model.h"
#include "gemma4_stage.cuh"

using Gemma4_26bStage = Gemma4_26bGeometry<SPARK_GEMMA4_26B_STAGE_TP_DEGREE>;

static_assert(GemmaShape<Gemma4_26bStage>::SlidingKvGeometry::kSlotBytes == SPARK_GEMMA4_26B_STAGE_SLIDING_SLOT_BYTES,
	"a sliding KV slot is [K: 8 x 256][V: 8 x 256] bf16");
static_assert(GemmaShape<Gemma4_26bStage>::FullKvGeometry::kSlotBytes == SPARK_GEMMA4_26B_STAGE_FULL_SLOT_BYTES,
	"a full KV slot is [K: 2 x 512][V: 2 x 512] bf16");
static_assert(GemmaShape<Gemma4_26bStage>::kSlidingFused == SPARK_GEMMA4_26B_STAGE_SLIDING_FUSED_WIDTH &&
	GemmaShape<Gemma4_26bStage>::kFullFused == SPARK_GEMMA4_26B_STAGE_FULL_FUSED_WIDTH, "sliding q|k|v is 8192 wide and full q|k is 9216 wide");
static_assert(GemmaShape<Gemma4_26bStage>::kSlidingFused % SPARK_GEMMA4_26B_STAGE_TP_DEGREE == 0u &&
	GemmaShape<Gemma4_26bStage>::kFullFused % SPARK_GEMMA4_26B_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");

const SparkStageRunnerModelInterface *SparkGemma4_26bStageModel(void)
{
	return GemmaModel<Gemma4_26bStage>::Interface();
}
