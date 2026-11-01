#include "sparkpipe/spark_motif3_tp16_stage_model.h"
#include "motif_stage.cuh"

static_assert(MotifKv::kSlotBytes == SPARK_MOTIF3_TP16_STAGE_SLOT_BYTES, "a latent slot is the normed latent | roped key row");

const SparkStageRunnerModelInterface *SparkMotif3Tp16StageModel(void)
{
	return MotifStage<Motif3Geometry<SPARK_MOTIF3_TP16_STAGE_TP_DEGREE>>::Interface();
}
