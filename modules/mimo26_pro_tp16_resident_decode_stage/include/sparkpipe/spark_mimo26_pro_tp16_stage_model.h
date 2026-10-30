#ifndef SPARKPIPE_SPARK_MIMO26_PRO_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_MIMO26_PRO_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MIMO26_PRO_TP16_STAGE_TP_DEGREE 16u
#define SPARK_MIMO26_PRO_TP16_STAGE_PAGE_SLOTS 64u
#define SPARK_MIMO26_PRO_TP16_STAGE_HIDDEN 6144u
#define SPARK_MIMO26_PRO_TP16_STAGE_LAYERS 70u
#define SPARK_MIMO26_PRO_TP16_STAGE_FULL_LAYERS 10u
#define SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_LAYERS (SPARK_MIMO26_PRO_TP16_STAGE_LAYERS - SPARK_MIMO26_PRO_TP16_STAGE_FULL_LAYERS)
#define SPARK_MIMO26_PRO_TP16_STAGE_HEADS 128u
#define SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_KV_HEADS 8u
#define SPARK_MIMO26_PRO_TP16_STAGE_FULL_KV_HEADS 8u
#define SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM 192u
#define SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM 128u
#define SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_SLOT_BYTES (SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_KV_HEADS * (SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM + SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM) * 2u)
#define SPARK_MIMO26_PRO_TP16_STAGE_FULL_SLOT_BYTES (SPARK_MIMO26_PRO_TP16_STAGE_FULL_KV_HEADS * (SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM + SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM) * 2u)
#define SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_FUSED (SPARK_MIMO26_PRO_TP16_STAGE_HEADS * SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM + SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_KV_HEADS * (SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM + SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM))
#define SPARK_MIMO26_PRO_TP16_STAGE_FULL_FUSED (SPARK_MIMO26_PRO_TP16_STAGE_HEADS * SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM + SPARK_MIMO26_PRO_TP16_STAGE_FULL_KV_HEADS * (SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM + SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM))

const SparkStageRunnerModelInterface *SparkMimo26ProTp16StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_mimo26_family.h"

template<uint32_t TP>
struct Mimo26ProGeometry
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_MIMO26_PRO_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_MIMO26_PRO_TP16_STAGE_LAYERS;
	static constexpr uint32_t kFullLayers = SPARK_MIMO26_PRO_TP16_STAGE_FULL_LAYERS;
	static constexpr uint32_t kVocab = 152576u;
	static constexpr uint32_t kHeads = SPARK_MIMO26_PRO_TP16_STAGE_HEADS;
	static constexpr uint32_t kSlidingKvHeads = SPARK_MIMO26_PRO_TP16_STAGE_SLIDING_KV_HEADS;
	static constexpr uint32_t kFullKvHeads = SPARK_MIMO26_PRO_TP16_STAGE_FULL_KV_HEADS;
	static constexpr uint32_t kHeadDim = SPARK_MIMO26_PRO_TP16_STAGE_HEAD_DIM;
	static constexpr uint32_t kValueDim = SPARK_MIMO26_PRO_TP16_STAGE_VALUE_DIM;
	static constexpr uint32_t kRopeDim = 64u;
	static constexpr uint32_t kWindow = 128u;
	static constexpr float kSlidingTheta = 10000.0f;
	static constexpr float kFullTheta = 10000000.0f;
	static constexpr float kValueScale = 0.612f;
	static constexpr float kAttentionScale = 0.07216878364870322f;
	static constexpr float kEpsilon = 1e-5f;
	static constexpr uint32_t kDenseLayers = 1u;
	static constexpr uint32_t kDenseInter = 16384u;
	static constexpr uint32_t kExperts = 384u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kMoeInter = 2048u;
	static constexpr uint32_t kPageSlots = SPARK_MIMO26_PRO_TP16_STAGE_PAGE_SLOTS;
	static bool FullLayer(uint32_t layer) { return layer < kLayers && "0111111011111110111111101111111011111110111111101111111011111101111110"[layer] == '0'; }
	static const char *Tag(void) { return "sparkpipe_mimo26_pro"; }
	static const char *Name(void) { return "mimo-v2.6-pro"; }
	static const char *ReportTag(void) { return "MIMO26-PRO-STATE"; }
};
#endif

#endif
