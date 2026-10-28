#include "sparkpipe/spark_mimo26_flash_tp16_stage_model.h"
#include "mimo26_stage.cuh"

using Mimo26FlashStage = Mimo26FlashGeometry<SPARK_MIMO26_FLASH_TP16_STAGE_TP_DEGREE>;

static_assert(MimoShape<Mimo26FlashStage>::SlidingKv::kSlotBytes == SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_SLOT_BYTES, "a sliding slot is [K: kv x 192][V: kv x 128] bf16");
static_assert(MimoShape<Mimo26FlashStage>::FullKv::kSlotBytes == SPARK_MIMO26_FLASH_TP16_STAGE_FULL_SLOT_BYTES, "a full slot is [K: kv x 192][V: kv x 128] bf16");
static_assert(MimoShape<Mimo26FlashStage>::kSlidingFused == SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_FUSED && MimoShape<Mimo26FlashStage>::kFullFused == SPARK_MIMO26_FLASH_TP16_STAGE_FULL_FUSED,
	"the fused q|k|v widths match the module header");

const SparkStageRunnerModelInterface *SparkMimo26FlashTp16StageModel(void)
{
	return MimoModel<Mimo26FlashStage>::Interface();
}
