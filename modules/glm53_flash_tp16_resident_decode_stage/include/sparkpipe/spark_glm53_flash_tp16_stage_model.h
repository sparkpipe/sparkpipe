#ifndef SPARKPIPE_SPARK_GLM53_FLASH_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_GLM53_FLASH_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE 16u
#define SPARK_GLM53_FLASH_TP16_STAGE_PAGE_SLOTS SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS
#define SPARK_GLM53_FLASH_TP16_STAGE_HIDDEN SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM53_FLASH_TP16_STAGE_LAYERS SPARK_GLM5_NEXT_MODEL_LAYER_COUNT
#define SPARK_GLM53_FLASH_TP16_STAGE_KV_LAYERS SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT
#define SPARK_GLM53_FLASH_TP16_STAGE_LATENT SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION
#define SPARK_GLM53_FLASH_TP16_STAGE_KV_SLOT_BYTES SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES
#define SPARK_GLM53_FLASH_TP16_STAGE_INDEX_SLOT_BYTES (SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u)
#define SPARK_GLM53_FLASH_TP16_STAGE_INDEX_DIMENSION SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION

const SparkStageRunnerModelInterface *SparkGlm53FlashTp16StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_glm5_next_family.h"
#include "sparkpipe/spark_weight_codec.h"

template<uint32_t TP>
struct Glm53FlashGeometry
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kExpertCodec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	static const char *Tag(void) { return "sparkpipe_glm53_flash"; }
	static const char *WeightdModel(void) { return "glm5_next_stage"; }
	static const char *WeightdRevision(void) { return "84c6a6aa9497188e15a635ba793b0f95a79b1033"; }
	static const char *ReportTag(void) { return "GLM53_FLASH-STATE"; }
};
#endif

#endif
