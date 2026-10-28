#ifndef SPARKPIPE_SPARK_SOLAR_OPEN2_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_SOLAR_OPEN2_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE 16u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS 64u
#define SPARK_SOLAR_OPEN2_TP16_VERIFY_ROWS 256u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN 4096u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS 48u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_PERIOD 4u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_ATTN_LAYERS (SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS / SPARK_SOLAR_OPEN2_TP16_STAGE_PERIOD)
#define SPARK_SOLAR_OPEN2_TP16_STAGE_HEADS 64u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS 8u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM 128u
#define SPARK_SOLAR_OPEN2_TP16_STAGE_QKV_WIDTH ((SPARK_SOLAR_OPEN2_TP16_STAGE_HEADS + 2u * SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS) * SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM)
#define SPARK_SOLAR_OPEN2_TP16_STAGE_KV_SLOT_BYTES (SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS * SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM * 2u * 2u)

const SparkStageRunnerModelInterface *SparkSolarOpen2StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_ling_family.h"

template<uint32_t TP>
struct SolarOpen2Geometry : LingDefaults
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 196608u;
	static constexpr uint32_t kPeriod = SPARK_SOLAR_OPEN2_TP16_STAGE_PERIOD;
	static constexpr uint32_t kPhase = 0u;
	static constexpr uint32_t kDenseLayers = 0u;
	static constexpr float kNormEpsilon = 1e-5f;
	static constexpr uint32_t kKdaHeads = 64u;
	static constexpr uint32_t kKdaDim = 128u;
	static constexpr uint32_t kKdaRank = 128u;
	static constexpr bool kKdaDoubledBeta = true;
	static constexpr uint32_t kConv = 4u;
	static constexpr float kKdaLowerBound = -5.0f;
	static constexpr uint32_t kAttention = LING_ATTENTION_GQA;
	static constexpr uint32_t kGqaHeads = SPARK_SOLAR_OPEN2_TP16_STAGE_HEADS;
	static constexpr uint32_t kGqaKvHeads = SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS;
	static constexpr uint32_t kGqaHeadDim = SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM;
	static constexpr uint32_t kExperts = 320u;
	static constexpr uint32_t kLocalExperts = kExperts;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kGroups = 1u;
	static constexpr uint32_t kTopkGroups = 1u;
	static constexpr uint32_t kMoeInter = 80u;
	static constexpr uint32_t kSharedRows = 128u;
	static constexpr uint32_t kVerifyRows = SPARK_SOLAR_OPEN2_TP16_VERIFY_ROWS;
	static constexpr float kRouteScale = 1.0f;
	static constexpr uint32_t kPageSlots = SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS;
	static const char *Tag(void) { return "sparkpipe_solar_open2"; }
	static const char *Name(void) { return "solar-open-2"; }
	static const char *ReportTag(void) { return "SOLAR-OPEN2-STATE"; }
};
#endif

#endif
