#ifndef SPARKPIPE_SPARK_HY4_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_HY4_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HY4_TP16_STAGE_TP_DEGREE 16u
#define SPARK_HY4_TP16_STAGE_PAGE_SLOTS 64u
#define SPARK_HY4_TP16_STAGE_HIDDEN 6144u
#define SPARK_HY4_TP16_STAGE_LAYERS 78u
#define SPARK_HY4_TP16_STAGE_INDEX_LAYERS 21u
#define SPARK_HY4_TP16_STAGE_LATENT 512u
#define SPARK_HY4_TP16_STAGE_ROPE 64u
#define SPARK_HY4_TP16_STAGE_INDEX_DIM 128u
#define SPARK_HY4_TP16_STAGE_KV_SLOT_BYTES ((SPARK_HY4_TP16_STAGE_LATENT + SPARK_HY4_TP16_STAGE_ROPE) * 2u)
#define SPARK_HY4_TP16_STAGE_INDEX_SLOT_BYTES (SPARK_HY4_TP16_STAGE_INDEX_DIM * 2u)

const SparkStageRunnerModelInterface *SparkHy4Tp16StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_hy4_family.h"

template<uint32_t TP>
struct Hy4Geometry
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_HY4_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_HY4_TP16_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 120832u;
	static constexpr uint32_t kStreams = 4u;
	static constexpr uint32_t kHeads = 64u;
	static constexpr uint32_t kNope = 192u;
	static constexpr uint32_t kRope = SPARK_HY4_TP16_STAGE_ROPE;
	static constexpr uint32_t kValueDim = 256u;
	static constexpr uint32_t kLatent = SPARK_HY4_TP16_STAGE_LATENT;
	static constexpr uint32_t kQLora = 2048u;
	static constexpr float kRopeTheta = 10000000.0f;
	static constexpr uint32_t kIndexHeads = 32u;
	static constexpr uint32_t kIndexDim = SPARK_HY4_TP16_STAGE_INDEX_DIM;
	static constexpr uint32_t kIndexTopk = 2048u;
	static constexpr uint32_t kIndexLayers = SPARK_HY4_TP16_STAGE_INDEX_LAYERS;
	static constexpr uint32_t kDenseLayers = 1u;
	static constexpr uint32_t kExperts = 256u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kMoeInter = 2048u;
	static constexpr float kRouteScale = 2.827f;
	static constexpr float kSwigluLimit = 10.0f;
	static constexpr float kEpsilon = 1e-5f;
	static constexpr float kIndexNormEpsilon = 1e-6f;
	static constexpr float kHcEpsilon = 1e-6f;
	static constexpr float kHcMagnitude = 2.0f;
	static constexpr uint32_t kPageSlots = SPARK_HY4_TP16_STAGE_PAGE_SLOTS;
	static bool IndexerLayer(uint32_t layer) { return layer == 0u || layer % 4u == 1u; }
	static const char *Tag(void) { return "sparkpipe_hy4"; }
	static const char *Name(void) { return "hy4-preview"; }
	static const char *ReportTag(void) { return "HY4-STATE"; }
};
#endif

#endif
