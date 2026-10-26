#include "sparkpipe/spark_kexaone2_tp16_stage_model.h"
#include "gemma4_stage.cuh"

struct Kexaone2Geometry : GemmaDefaults
{
	static constexpr uint32_t kTpDegree = SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE;
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
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kMoeInter = 2048u;
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
};

static_assert(GemmaShape<Kexaone2Geometry>::kFullLayers == SPARK_KEXAONE2_TP16_STAGE_FULL_LAYERS &&
	GemmaShape<Kexaone2Geometry>::kSlidingLayers == SPARK_KEXAONE2_TP16_STAGE_SLIDING_LAYERS,
	"layer 0 and every layer 4k+1 from 5 attend globally: 20 full and 58 sliding layers");
static_assert(GemmaShape<Kexaone2Geometry>::SlidingKvGeometry::kSlotBytes == SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES &&
	GemmaShape<Kexaone2Geometry>::FullKvGeometry::kSlotBytes == SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES, "a KV slot is [K: 8 x 128][V: 8 x 128] bf16");
static_assert(GemmaShape<Kexaone2Geometry>::kSlidingFused == SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH &&
	GemmaShape<Kexaone2Geometry>::kFullFused == SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH, "q|k|v is 64 + 2 x 8 heads of 128 on every layer");
static_assert(SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH % SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");
static_assert(SPARK_KEXAONE2_TP16_STAGE_HEADS % SPARK_KEXAONE2_TP16_STAGE_KV_HEADS == 0u &&
	SPARK_KEXAONE2_TP16_STAGE_HEADS / SPARK_KEXAONE2_TP16_STAGE_KV_HEADS <= LM_GQA_SHARD_MAX_GROUP, "whole GQA groups of 8 query heads per KV head");

const SparkStageRunnerModelInterface *SparkKexaone2Tp16StageModel(void)
{
	return GemmaModel<Kexaone2Geometry>::Interface();
}
