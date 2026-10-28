#include "sparkpipe/spark_mimo26_pro_tp16_stage_model.h"
#include "mimo26_stage.cuh"

using Mimo26ProStage = Mimo26ProGeometry<SPARK_MIMO26_PRO_TP16_STAGE_TP_DEGREE>;

static_assert(MimoShape<Mimo26ProStage>::SlidingKv::kSlotBytes == SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_SLOT_BYTES, "a sliding slot is [K: kv x 192][V: kv x 128] bf16");
static_assert(MimoShape<Mimo26ProStage>::FullKv::kSlotBytes == SPARK_MIMO26_PRO_TP16_STAGE_FULL_SLOT_BYTES, "a full slot is [K: kv x 192][V: kv x 128] bf16");
static_assert(MimoShape<Mimo26ProStage>::kSlidingFused == SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_FUSED && MimoShape<Mimo26ProStage>::kFullFused == SPARK_MIMO26_PRO_TP16_STAGE_FULL_FUSED,
	"the fused q|k|v widths match the module header");

const SparkStageRunnerModelInterface *SparkMimo26ProTp16StageModel(void)
{
	return MimoModel<Mimo26ProStage>::Interface();
}
