#ifndef SPARKPIPE_SPARK_LAGUNA_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_LAGUNA_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_LAGUNA_TP16_STAGE_TP_DEGREE 16u
#define SPARK_LAGUNA_TP16_STAGE_PAGE_SLOTS 64u
#define SPARK_LAGUNA_TP16_STAGE_HIDDEN 3072u
#define SPARK_LAGUNA_TP16_STAGE_LAYERS 48u
#define SPARK_LAGUNA_TP16_STAGE_PERIOD 4u
#define SPARK_LAGUNA_TP16_STAGE_KV_HEADS 8u
#define SPARK_LAGUNA_TP16_STAGE_HEAD_DIM 128u
#define SPARK_LAGUNA_TP16_STAGE_SLIDING_HEADS 80u
#define SPARK_LAGUNA_TP16_STAGE_FULL_HEADS 48u
#define SPARK_LAGUNA_TP16_STAGE_SLIDING_LAYERS (SPARK_LAGUNA_TP16_STAGE_LAYERS - SPARK_LAGUNA_TP16_STAGE_LAYERS / SPARK_LAGUNA_TP16_STAGE_PERIOD)
#define SPARK_LAGUNA_TP16_STAGE_FULL_LAYERS (SPARK_LAGUNA_TP16_STAGE_LAYERS / SPARK_LAGUNA_TP16_STAGE_PERIOD)
#define SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES (SPARK_LAGUNA_TP16_STAGE_KV_HEADS * SPARK_LAGUNA_TP16_STAGE_HEAD_DIM * 2u * 2u)
#define SPARK_LAGUNA_TP16_STAGE_SLIDING_FUSED_WIDTH 12288u
#define SPARK_LAGUNA_TP16_STAGE_FULL_FUSED_WIDTH 8192u

const SparkStageRunnerModelInterface *SparkLagunaTp16StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_gemma4_family.h"

template<uint32_t TP>
struct LagunaGeometry : GemmaDefaults
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_LAGUNA_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_LAGUNA_TP16_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 100352u;
	static constexpr uint32_t kPeriod = SPARK_LAGUNA_TP16_STAGE_PERIOD;
	static constexpr uint32_t kPhase = 0u;
	static constexpr uint32_t kSlidingHeads = SPARK_LAGUNA_TP16_STAGE_SLIDING_HEADS;
	static constexpr uint32_t kFullHeads = SPARK_LAGUNA_TP16_STAGE_FULL_HEADS;
	static constexpr uint32_t kSlidingKvHeads = SPARK_LAGUNA_TP16_STAGE_KV_HEADS;
	static constexpr uint32_t kSlidingDim = SPARK_LAGUNA_TP16_STAGE_HEAD_DIM;
	static constexpr uint32_t kFullKvHeads = SPARK_LAGUNA_TP16_STAGE_KV_HEADS;
	static constexpr uint32_t kFullDim = SPARK_LAGUNA_TP16_STAGE_HEAD_DIM;
	static constexpr bool kKeyIsValue = false;
	static constexpr uint32_t kWindow = 512u;
	static constexpr float kSlidingTheta = 10000.0f;
	static constexpr float kFullTheta = 500000.0f;
	static constexpr uint32_t kFullRopePairs = 0u;
	static constexpr float kEmbedScale = 1.0f;
	static constexpr float kLogitSoftcap = 0.0f;
	static constexpr uint32_t kPageSlots = SPARK_LAGUNA_TP16_STAGE_PAGE_SLOTS;
	static constexpr float kNormEpsilon = 1e-6f;
	static constexpr bool kValueNorm = false;
	static constexpr float kAttentionScale = 0.08838834764831845f;
	static constexpr bool kGeluMlp = false;
	static constexpr bool kUntiedHead = true;
	static constexpr bool kLayerScalar = false;
	static constexpr bool kMoe = true;
	static constexpr uint32_t kExperts = 256u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 10u;
	static constexpr uint32_t kMoeInter = 1024u;
	static constexpr bool kPreNorm = true;
	static constexpr uint32_t kFullRopeDim = 64u;
	static constexpr float kFullRopeScale = 1.4852030263919618f;
	static constexpr float kYarnFactor = 128.0f;
	static constexpr float kYarnOriginal = 8192.0f;
	static constexpr float kYarnBetaFast = 32.0f;
	static constexpr float kYarnBetaSlow = 1.0f;
	static constexpr bool kHeadGate = true;
	static constexpr bool kSigmoidRouter = true;
	static constexpr uint32_t kFirstMoeLayer = 1u;
	static constexpr float kRoutedScale = 2.5f;
	static const char *Tag(void) { return "sparkpipe_laguna"; }
	static const char *Name(void) { return "laguna-s-2.1"; }
	static const char *ReportTag(void) { return "LAGUNA-STATE"; }
	static constexpr float kPostEpsilon = 1e-6f;
	static constexpr bool kQkNormWeighted = true;
	static constexpr float kQueryMultiply = 1.0f;
	static constexpr bool kFullRope = true;
	static constexpr bool kRouterFloat = false;
};
#endif

#endif
