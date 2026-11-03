#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"
#include "glm5_next_stage.cuh"

#ifndef GLM53_FLASH_EXPERT_CODEC
#error "GLM53_FLASH_EXPERT_CODEC must name the routed-expert weight codec of the packs this build serves"
#endif

struct Glm53FlashGeometry
{
	static constexpr uint32_t kTpDegree = SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kExpertCodec = GLM53_FLASH_EXPERT_CODEC;
	static const char *Tag(void) { return "sparkpipe_glm53_flash"; }
	static const char *WeightdModel(void) { return "glm5_next_stage"; }
	static const char *WeightdRevision(void) { return "84c6a6aa9497188e15a635ba793b0f95a79b1033"; }
	static const char *ReportTag(void) { return "GLM53_FLASH-STATE"; }
};

static_assert(Glm5NextKv::kSlotBytes == SPARK_GLM53_FLASH_TP16_STAGE_KV_SLOT_BYTES, "a latent slot is the 512-element normed kv_a row");
static_assert(Glm5NextIndexKv::kSlotBytes == SPARK_GLM53_FLASH_TP16_STAGE_INDEX_SLOT_BYTES, "an index slot is key | compress gate | 1.0");

const SparkStageRunnerModelInterface *SparkGlm53FlashTp16StageModel(void)
{
	return Glm5NextStage<Glm53FlashGeometry>::Interface();
}
