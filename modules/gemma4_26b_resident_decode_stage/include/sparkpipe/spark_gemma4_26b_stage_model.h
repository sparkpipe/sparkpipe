#ifndef SPARKPIPE_SPARK_GEMMA4_26B_STAGE_MODEL_H
#define SPARKPIPE_SPARK_GEMMA4_26B_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GEMMA4_26B_STAGE_TP_DEGREE 16u
#define SPARK_GEMMA4_26B_STAGE_PAGE_SLOTS 64u
#define SPARK_GEMMA4_26B_STAGE_HIDDEN 2816u
#define SPARK_GEMMA4_26B_STAGE_LAYERS 30u
#define SPARK_GEMMA4_26B_STAGE_PERIOD 6u
#define SPARK_GEMMA4_26B_STAGE_SLIDING_KV_HEADS 8u
#define SPARK_GEMMA4_26B_STAGE_SLIDING_DIM 256u
#define SPARK_GEMMA4_26B_STAGE_FULL_KV_HEADS 2u
#define SPARK_GEMMA4_26B_STAGE_FULL_DIM 512u
#define SPARK_GEMMA4_26B_STAGE_SLIDING_LAYERS (SPARK_GEMMA4_26B_STAGE_LAYERS - SPARK_GEMMA4_26B_STAGE_LAYERS / SPARK_GEMMA4_26B_STAGE_PERIOD)
#define SPARK_GEMMA4_26B_STAGE_FULL_LAYERS (SPARK_GEMMA4_26B_STAGE_LAYERS / SPARK_GEMMA4_26B_STAGE_PERIOD)
#define SPARK_GEMMA4_26B_STAGE_SLIDING_SLOT_BYTES (SPARK_GEMMA4_26B_STAGE_SLIDING_KV_HEADS * SPARK_GEMMA4_26B_STAGE_SLIDING_DIM * 2u * 2u)
#define SPARK_GEMMA4_26B_STAGE_FULL_SLOT_BYTES (SPARK_GEMMA4_26B_STAGE_FULL_KV_HEADS * SPARK_GEMMA4_26B_STAGE_FULL_DIM * 2u * 2u)
#define SPARK_GEMMA4_26B_STAGE_SLIDING_FUSED_WIDTH 8192u
#define SPARK_GEMMA4_26B_STAGE_FULL_FUSED_WIDTH 9216u

const SparkStageRunnerModelInterface *SparkGemma4_26bStageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_gemma4_family.h"

template<uint32_t TP>
struct Gemma4_26bGeometry : GemmaDefaults
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_GEMMA4_26B_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_GEMMA4_26B_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 262144u;
	static constexpr uint32_t kPeriod = SPARK_GEMMA4_26B_STAGE_PERIOD;
	static constexpr uint32_t kPhase = 5u;
	static constexpr uint32_t kSlidingHeads = 16u;
	static constexpr uint32_t kFullHeads = 16u;
	static constexpr uint32_t kSlidingKvHeads = SPARK_GEMMA4_26B_STAGE_SLIDING_KV_HEADS;
	static constexpr uint32_t kSlidingDim = SPARK_GEMMA4_26B_STAGE_SLIDING_DIM;
	static constexpr uint32_t kFullKvHeads = SPARK_GEMMA4_26B_STAGE_FULL_KV_HEADS;
	static constexpr uint32_t kFullDim = SPARK_GEMMA4_26B_STAGE_FULL_DIM;
	static constexpr bool kKeyIsValue = true;
	static constexpr uint32_t kWindow = 1024u;
	static constexpr float kSlidingTheta = 10000.0f;
	static constexpr float kFullTheta = 1000000.0f;
	static constexpr uint32_t kFullRopePairs = 64u;
	static constexpr float kEmbedScale = 53.0f;
	static constexpr float kLogitSoftcap = 30.0f;
	static constexpr uint32_t kPageSlots = SPARK_GEMMA4_26B_STAGE_PAGE_SLOTS;
	static constexpr bool kMoe = true;
	static constexpr uint32_t kExperts = 128u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kMoeInter = 704u;
	static const char *Tag(void) { return "sparkpipe_gemma4_26b"; }
	static const char *Name(void) { return "gemma4-26b-a4b"; }
	static const char *ReportTag(void) { return "GEMMA4_26B-STATE"; }
	static constexpr float kNormEpsilon = 1e-6f;
	static constexpr float kPostEpsilon = 1e-6f;
	static constexpr bool kQkNormWeighted = true;
	static constexpr float kQueryMultiply = 1.0f;
	static constexpr bool kValueNorm = true;
	static constexpr float kAttentionScale = 1.0f;
	static constexpr bool kFullRope = true;
	static constexpr bool kGeluMlp = true;
	static constexpr bool kLayerScalar = true;
	static constexpr bool kPreNorm = false;
	static constexpr bool kSigmoidRouter = false;
	static constexpr float kRoutedScale = 1.0f;
	static constexpr bool kRouterFloat = false;
};
#endif

#endif
