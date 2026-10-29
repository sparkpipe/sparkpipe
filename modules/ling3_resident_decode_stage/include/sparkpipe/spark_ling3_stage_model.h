#ifndef SPARKPIPE_SPARK_LING3_STAGE_MODEL_H
#define SPARKPIPE_SPARK_LING3_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_LING3_STAGE_TP_DEGREE 16u
#define SPARK_LING3_STAGE_PAGE_SLOTS 64u
#define SPARK_LING3_STAGE_HIDDEN 2560u
#define SPARK_LING3_STAGE_LAYERS 42u
#define SPARK_LING3_STAGE_PERIOD 6u
#define SPARK_LING3_STAGE_MLA_LAYERS (SPARK_LING3_STAGE_LAYERS / SPARK_LING3_STAGE_PERIOD)
#define SPARK_LING3_STAGE_LATENT 512u
#define SPARK_LING3_STAGE_ROPE 64u
#define SPARK_LING3_STAGE_KV_SLOT_BYTES ((SPARK_LING3_STAGE_LATENT + SPARK_LING3_STAGE_ROPE) * 2u)

const SparkStageRunnerModelInterface *SparkLing3StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_ling_family.h"

template<uint32_t TP>
struct Ling3Geometry : LingDefaults
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_LING3_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_LING3_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 157184u;
	static constexpr uint32_t kPeriod = SPARK_LING3_STAGE_PERIOD;
	static constexpr uint32_t kPhase = 5u;
	static constexpr uint32_t kDenseLayers = 2u;
	static constexpr uint32_t kKdaHeads = 32u;
	static constexpr uint32_t kKdaDim = 128u;
	static constexpr uint32_t kConv = 4u;
	static constexpr float kKdaLowerBound = -5.0f;
	static constexpr uint32_t kMlaHeads = 32u;
	static constexpr uint32_t kNope = 128u;
	static constexpr uint32_t kRope = SPARK_LING3_STAGE_ROPE;
	static constexpr uint32_t kValueDim = 128u;
	static constexpr uint32_t kLatent = SPARK_LING3_STAGE_LATENT;
	static constexpr float kRopeTheta = 6000000.0f;
	static constexpr uint32_t kExperts = 512u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kGroups = 8u;
	static constexpr uint32_t kTopkGroups = 4u;
	static constexpr uint32_t kMoeInter = 768u;
	static constexpr uint32_t kSharedInter = 768u;
	static constexpr uint32_t kSharedRows = (kSharedInter / TP + 63u) / 64u * 64u;
	static constexpr float kRouteScale = 2.5f;
	static constexpr uint32_t kPageSlots = SPARK_LING3_STAGE_PAGE_SLOTS;
	static float RoutedLimit(uint32_t layer) { return layer >= 35u ? 4.0f : 0.0f; }
	static float SharedLimit(uint32_t layer) { return layer >= 40u ? 7.0f : layer >= 34u ? 5.0f : 0.0f; }
	static const char *Tag(void) { return "sparkpipe_ling3"; }
	static const char *Name(void) { return "ling-3.0-flash"; }
	static const char *ReportTag(void) { return "LING3-STATE"; }
	static constexpr uint32_t kAttention = LING_ATTENTION_MLA;
	static constexpr float kNormEpsilon = LG_EPSILON;
	static constexpr bool kKdaDoubledBeta = false;
};
#endif

#endif
