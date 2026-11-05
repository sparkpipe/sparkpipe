#ifndef SPARKPIPE_SPARK_MOTIF3_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_MOTIF3_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_motif_model.h"
#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MOTIF3_TP16_STAGE_TP_DEGREE 16u
#define SPARK_MOTIF3_TP16_STAGE_PAGE_SLOTS SPARK_MOTIF_MODEL_KV_PAGE_SLOTS
#define SPARK_MOTIF3_TP16_STAGE_HIDDEN SPARK_MOTIF_MODEL_HIDDEN_DIMENSION
#define SPARK_MOTIF3_TP16_STAGE_LAYERS SPARK_MOTIF_MODEL_LAYER_COUNT
#define SPARK_MOTIF3_TP16_STAGE_LATENT SPARK_MOTIF_MODEL_KV_LORA_RANK
#define SPARK_MOTIF3_TP16_STAGE_ROPE SPARK_MOTIF_MODEL_ROPE_DIMENSION
#define SPARK_MOTIF3_TP16_STAGE_SLOT SPARK_MOTIF_MODEL_LATENT_ROW_DIMENSION
#define SPARK_MOTIF3_TP16_STAGE_SLOT_BYTES SPARK_MOTIF_MODEL_LATENT_ROW_BYTES

const SparkStageRunnerModelInterface *SparkMotif3Tp16StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_motif_family.h"

template<uint32_t TP>
struct Motif3Geometry
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kPackAlignment = 128u;
	static const char *Tag(void) { return "sparkpipe_motif3"; }
	static const char *WeightdModel(void) { return "motif3_stage"; }
	static const char *WeightdRevision(void) { return "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"; }
	static const char *ReportTag(void) { return "MOTIF3-STATE"; }
};
#endif

#endif
