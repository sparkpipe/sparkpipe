#include "sparkpipe/spark_laguna_tp16_stage_model.h"
#include "gemma4_stage.cuh"

struct LagunaGeometry : GemmaDefaults
{
	static constexpr uint32_t kTpDegree = SPARK_LAGUNA_TP16_STAGE_TP_DEGREE;
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
};

static_assert(GemmaShape<LagunaGeometry>::SlidingKvGeometry::kSlotBytes == SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES &&
	GemmaShape<LagunaGeometry>::FullKvGeometry::kSlotBytes == SPARK_LAGUNA_TP16_STAGE_SLOT_BYTES, "a KV slot is [K: 8 x 128][V: 8 x 128] bf16");
static_assert(GemmaShape<LagunaGeometry>::kSlidingFused == SPARK_LAGUNA_TP16_STAGE_SLIDING_FUSED_WIDTH &&
	GemmaShape<LagunaGeometry>::kFullFused == SPARK_LAGUNA_TP16_STAGE_FULL_FUSED_WIDTH,
	"sliding q|k|v is 80 padded heads + 2 x 8 KV heads of 128, full q|k|v is 48 + 2 x 8 heads of 128");
static_assert(GemmaShape<LagunaGeometry>::kSlidingFused % SPARK_LAGUNA_TP16_STAGE_TP_DEGREE == 0u &&
	GemmaShape<LagunaGeometry>::kFullFused % SPARK_LAGUNA_TP16_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");
static_assert(GemmaShape<LagunaGeometry>::kSlidingGateRows == 6u && GemmaShape<LagunaGeometry>::kFullGateRows == 4u,
	"a rank gates 5 sliding or 3 full heads; the gate projection is padded to an even row count");
static_assert(SPARK_LAGUNA_TP16_STAGE_SLIDING_HEADS % SPARK_LAGUNA_TP16_STAGE_KV_HEADS == 0u &&
	SPARK_LAGUNA_TP16_STAGE_SLIDING_HEADS / SPARK_LAGUNA_TP16_STAGE_KV_HEADS <= LM_GQA_SHARD_MAX_GROUP,
	"the padded sliding layer keeps whole GQA groups of 10 query heads per KV head");

const SparkStageRunnerModelInterface *SparkLagunaTp16StageModel(void)
{
	return GemmaModel<LagunaGeometry>::Interface();
}
