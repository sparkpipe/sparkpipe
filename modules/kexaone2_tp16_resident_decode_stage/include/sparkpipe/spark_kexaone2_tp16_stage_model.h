#ifndef SPARKPIPE_SPARK_KEXAONE2_TP16_STAGE_MODEL_H
#define SPARKPIPE_SPARK_KEXAONE2_TP16_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE 16u
#define SPARK_KEXAONE2_TP16_STAGE_PAGE_SLOTS 64u
#define SPARK_KEXAONE2_TP16_STAGE_HIDDEN 6144u
#define SPARK_KEXAONE2_TP16_STAGE_LAYERS 78u
#define SPARK_KEXAONE2_TP16_STAGE_FULL_LAYERS 20u
#define SPARK_KEXAONE2_TP16_STAGE_SLIDING_LAYERS (SPARK_KEXAONE2_TP16_STAGE_LAYERS - SPARK_KEXAONE2_TP16_STAGE_FULL_LAYERS)
#define SPARK_KEXAONE2_TP16_STAGE_KV_HEADS 8u
#define SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM 128u
#define SPARK_KEXAONE2_TP16_STAGE_HEADS 64u
#define SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES (SPARK_KEXAONE2_TP16_STAGE_KV_HEADS * SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM * 2u * 2u)
#define SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH 10240u
#define SPARK_KEXAONE2_TP16_STAGE_WINDOW 128u
#define SPARK_KEXAONE2_TP16_STAGE_WIDE_WINDOW 4096u

const SparkStageRunnerModelInterface *SparkKexaone2Tp16StageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_gemma4_family.h"

template<uint32_t TP>
struct Kexaone2Geometry : GemmaDefaults
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_KEXAONE2_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_KEXAONE2_TP16_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 153600u;
	static constexpr uint32_t kPeriod = 4u;
	static constexpr uint32_t kPhase = 1u;
	static constexpr uint32_t kSlidingHeads = SPARK_KEXAONE2_TP16_STAGE_HEADS;
	static constexpr uint32_t kFullHeads = SPARK_KEXAONE2_TP16_STAGE_HEADS;
	static constexpr uint32_t kSlidingKvHeads = SPARK_KEXAONE2_TP16_STAGE_KV_HEADS;
	static constexpr uint32_t kSlidingDim = SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM;
	static constexpr uint32_t kFullKvHeads = SPARK_KEXAONE2_TP16_STAGE_KV_HEADS;
	static constexpr uint32_t kFullDim = SPARK_KEXAONE2_TP16_STAGE_HEAD_DIM;
	static constexpr bool kKeyIsValue = false;
	static constexpr uint32_t kWindow = SPARK_KEXAONE2_TP16_STAGE_WINDOW;
	static constexpr float kSlidingTheta = 1000000.0f;
	static constexpr float kFullTheta = 1000000.0f;
	static constexpr uint32_t kFullRopePairs = 0u;
	static constexpr bool kFullRope = false;
	static constexpr float kEmbedScale = 1.0f;
	static constexpr float kLogitSoftcap = 0.0f;
	static constexpr uint32_t kPageSlots = SPARK_KEXAONE2_TP16_STAGE_PAGE_SLOTS;
	static constexpr float kNormEpsilon = 1e-5f;
	static constexpr bool kValueNorm = false;
	static constexpr float kAttentionScale = 0.08838834764831845f;
	static constexpr bool kGeluMlp = false;
	static constexpr bool kUntiedHead = true;
	static constexpr bool kLayerScalar = false;
	static constexpr bool kMoe = true;
	static constexpr uint32_t kExperts = 256u;
	static constexpr uint32_t kLocalExperts = kExperts;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kMoeInter = 2048u / kTpDegree;
	static constexpr bool kPreNorm = true;
	static constexpr bool kSigmoidRouter = true;
	static constexpr bool kRouterFloat = true;
	static constexpr uint32_t kFirstMoeLayer = 2u;
	static constexpr float kRoutedScale = 2.5f;
	static constexpr bool kLayerTable = true;
	static constexpr uint32_t LayerWindow(uint32_t index)
	{
		return index == 0u ? 0u : index == 1u ? SPARK_KEXAONE2_TP16_STAGE_WIDE_WINDOW : index % kPeriod == kPhase ? 0u : kWindow;
	}
	static constexpr float LayerSwigluLimit(uint32_t index)
	{
		return index >= 62u ? 7.0f : INFINITY;
	}
	static const char *Tag(void) { return "sparkpipe_kexaone2"; }
	static const char *Name(void) { return "k-exaone-2.0"; }
	static const char *ReportTag(void) { return "KEXAONE2-STATE"; }
	static constexpr float kPostEpsilon = 1e-6f;
	static constexpr bool kQkNormWeighted = true;
	static constexpr float kQueryMultiply = 1.0f;
};
#endif

#endif
