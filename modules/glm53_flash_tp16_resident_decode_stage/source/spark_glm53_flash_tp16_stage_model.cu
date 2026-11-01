#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"
#include "glm5_next_stage.cuh"

using Glm53FlashStage = Glm53FlashGeometry<SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE>;

static_assert(Glm5NextKv::kSlotBytes == SPARK_GLM53_FLASH_TP16_STAGE_KV_SLOT_BYTES, "a latent slot is the 512-element normed kv_a row");
static_assert(Glm5NextIndexKv::kSlotBytes == SPARK_GLM53_FLASH_TP16_STAGE_INDEX_SLOT_BYTES, "an index slot is key | compress gate | 1.0");

const SparkStageRunnerModelInterface *SparkGlm53FlashTp16StageModel(void)
{
	return Glm5NextStage<Glm53FlashStage>::Interface();
}
