#include "sparkpipe/spark_gemma4_31b_stage_model.h"
#include "gemma4_stage.cuh"

using Gemma4_31bStage = Gemma4_31bGeometry<SPARK_GEMMA4_31B_STAGE_TP_DEGREE>;

static_assert(GemmaShape<Gemma4_31bStage>::SlidingKvGeometry::kSlotBytes == SPARK_GEMMA4_31B_STAGE_SLIDING_SLOT_BYTES,
	"a sliding KV slot is [K: 16 x 256][V: 16 x 256] bf16");
static_assert(GemmaShape<Gemma4_31bStage>::FullKvGeometry::kSlotBytes == SPARK_GEMMA4_31B_STAGE_FULL_SLOT_BYTES,
	"a full KV slot is [K: 4 x 512][V: 4 x 512] bf16");
static_assert(GemmaShape<Gemma4_31bStage>::kSlidingFused == SPARK_GEMMA4_31B_STAGE_SLIDING_FUSED_WIDTH &&
	GemmaShape<Gemma4_31bStage>::kFullFused == SPARK_GEMMA4_31B_STAGE_FULL_FUSED_WIDTH, "sliding q|k|v is 16384 wide and full q|k is 18432 wide");
static_assert(GemmaShape<Gemma4_31bStage>::kSlidingFused % SPARK_GEMMA4_31B_STAGE_TP_DEGREE == 0u &&
	GemmaShape<Gemma4_31bStage>::kFullFused % SPARK_GEMMA4_31B_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");

const SparkStageRunnerModelInterface *SparkGemma4_31bStageModel(void)
{
	return GemmaModel<Gemma4_31bStage>::Interface();
}
