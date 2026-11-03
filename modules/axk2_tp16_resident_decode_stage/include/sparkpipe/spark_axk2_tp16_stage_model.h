#ifndef SPARKPIPE_SPARK_AXK2_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_AXK2_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_AXK2_TP16_STAGE_TP_DEGREE 16u
#define SPARK_AXK2_TP16_STAGE_PAGE_SLOTS 64u
#define SPARK_AXK2_TP16_STAGE_HIDDEN 7168u
#define SPARK_AXK2_TP16_STAGE_LAYERS 61u
#define SPARK_AXK2_TP16_STAGE_INDEX_LAYERS 61u
#define SPARK_AXK2_TP16_STAGE_LATENT 512u
#define SPARK_AXK2_TP16_STAGE_ROPE 64u
#define SPARK_AXK2_TP16_STAGE_INDEX_DIM 128u
#define SPARK_AXK2_TP16_STAGE_FP8_SCALE_BLOCK 128u
#define SPARK_AXK2_TP16_STAGE_KV_SLOT_BYTES ((SPARK_AXK2_TP16_STAGE_LATENT + SPARK_AXK2_TP16_STAGE_ROPE) * 2u)
#define SPARK_AXK2_TP16_STAGE_INDEX_SLOT_BYTES (SPARK_AXK2_TP16_STAGE_INDEX_DIM * 2u)

const SparkStageRunnerModelInterface *SparkAxk2Tp16StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
struct Axk2Dimensions
{
	static constexpr uint32_t kHidden = SPARK_AXK2_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_AXK2_TP16_STAGE_LAYERS;
	static constexpr uint32_t kFirstRouted = 1u;
	static constexpr uint32_t kVocab = 163840u;
	static constexpr uint32_t kHeads = 64u;
	static constexpr uint32_t kLatent = SPARK_AXK2_TP16_STAGE_LATENT;
	static constexpr uint32_t kRope = SPARK_AXK2_TP16_STAGE_ROPE;
	static constexpr uint32_t kNope = 128u;
	static constexpr uint32_t kValue = 128u;
	static constexpr uint32_t kQueryA = 1536u;
	static constexpr uint32_t kSelected = 2048u;
	static constexpr uint32_t kExperts = 256u;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kExpertInter = 2048u;
	static constexpr uint32_t kDenseInter = 18432u;
	static constexpr float kRmsEpsilon = 1e-06f;
	static constexpr float kRopeTheta = 1000000.0f;
	static constexpr float kQkScale = 0.0825202391f;
	static constexpr float kRoutedScale = 2.5f;
	static constexpr uint32_t kQueryGate = 128u;
	static constexpr bool kQueryConcat = true;
	static constexpr uint32_t kNormGateRank = 16u;
	static constexpr uint32_t kExpertGroups = 8u;
	static constexpr uint32_t kTopGroups = 4u;
	static constexpr float kRopeFactor = 2.0f;
	static constexpr float kRopeBetaFast = 32.0f;
	static constexpr float kRopeBetaSlow = 1.0f;
	static constexpr uint32_t kRopeOriginalPositions = 131072u;
};
#endif

#endif
