#include "sparkpipe/spark_muse_tp16_stage_model.h"
#include "gemma4_stage.cuh"

using MuseGlimmerStage = MuseGlimmerGeometry<SPARK_MUSE_TP16_STAGE_TP_DEGREE>;

static_assert(GemmaShape<MuseGlimmerStage>::SlidingKvGeometry::kSlotBytes == SPARK_MUSE_TP16_STAGE_SLIDING_SLOT_BYTES,
	"a KV slot is [K: 2 x 128][V: 2 x 128] bf16");
static_assert(GemmaShape<MuseGlimmerStage>::kSlidingFused == SPARK_MUSE_TP16_STAGE_SLIDING_FUSED_WIDTH &&
	GemmaShape<MuseGlimmerStage>::kFullFused == SPARK_MUSE_TP16_STAGE_FULL_FUSED_WIDTH, "q|k|v is 4608 wide on both layer kinds");
static_assert(GemmaShape<MuseGlimmerStage>::kSlidingFused % SPARK_MUSE_TP16_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");

const SparkStageRunnerModelInterface *SparkMuseTp16StageModel(void)
{
	return GemmaModel<MuseGlimmerStage>::Interface();
}
