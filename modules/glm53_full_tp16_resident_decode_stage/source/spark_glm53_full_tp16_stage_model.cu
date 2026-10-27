#include "sparkpipe/spark_glm53_full_tp16_stage_model.h"
#include "glm52_stage.cuh"

#ifndef GLM53_FULL_CONTRACT_SHA256
#error "GLM53_FULL_CONTRACT_SHA256 must identify the pack contract"
#endif
#ifndef GLM53_FULL_MODEL_REVISION
#error "GLM53_FULL_MODEL_REVISION must name the checkpoint revision"
#endif

struct Glm53FullGeometry : Glm52Dimensions
{
	static constexpr uint32_t kTpDegree = SPARK_GLM53_FULL_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kIndexLayers = SPARK_GLM53_FULL_TP16_STAGE_INDEX_LAYERS;
	static const char *Tag(void) { return "sparkpipe_glm53_full"; }
	static const char *ReportTag(void) { return "GLM53-FULL-STATE"; }
	static const char *WeightdModel(void) { return "glm52_stage"; }
	static const char *Revision(void) { return GLM53_FULL_MODEL_REVISION; }
	static const char *Contract(void) { return GLM53_FULL_CONTRACT_SHA256; }
};

typedef GlmShape<Glm53FullGeometry> Glm53FullShape;

static_assert(SPARK_GLM53_FULL_TP16_STAGE_HIDDEN == Glm53FullShape::kHidden && SPARK_GLM53_FULL_TP16_STAGE_LAYERS == Glm53FullShape::kLayers, "stage geometry matches the GLM-5.3 model");
static_assert(SPARK_GLM53_FULL_TP16_STAGE_KV_SLOT_BYTES == GlmKv<Glm53FullShape>::kSlotBytes, "a latent slot is the normed latent and its rope part");
static_assert(SPARK_GLM53_FULL_TP16_STAGE_INDEX_SLOT_BYTES == GlmIndexKv<Glm53FullShape>::kSlotBytes, "an index slot is one indexer key");
static_assert(SPARK_GLM53_FULL_TP16_STAGE_PAGE_SLOTS == GLM_PAGE_SLOTS, "64-token pages");

const SparkStageRunnerModelInterface *SparkGlm53FullTp16StageModel(void)
{
	return GlmModel<Glm53FullGeometry>::Interface();
}
