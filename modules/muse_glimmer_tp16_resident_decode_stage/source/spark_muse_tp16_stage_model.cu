#include "sparkpipe/spark_muse_tp16_stage_model.h"
#include "gemma4_stage.cuh"

struct MuseGlimmerGeometry : GemmaDefaults
{
	static constexpr uint32_t kTpDegree = SPARK_MUSE_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_MUSE_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_MUSE_TP16_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 202048u;
	static constexpr uint32_t kPeriod = SPARK_MUSE_TP16_STAGE_PERIOD;
	static constexpr uint32_t kPhase = 3u;
	static constexpr uint32_t kSlidingHeads = 32u;
	static constexpr uint32_t kFullHeads = 32u;
	static constexpr uint32_t kSlidingKvHeads = SPARK_MUSE_TP16_STAGE_SLIDING_KV_HEADS;
	static constexpr uint32_t kSlidingDim = SPARK_MUSE_TP16_STAGE_SLIDING_DIM;
	static constexpr uint32_t kFullKvHeads = SPARK_MUSE_TP16_STAGE_FULL_KV_HEADS;
	static constexpr uint32_t kFullDim = SPARK_MUSE_TP16_STAGE_FULL_DIM;
	static constexpr bool kKeyIsValue = false;
	static constexpr uint32_t kWindow = 2048u;
	static constexpr float kSlidingTheta = 500000.0f;
	static constexpr float kFullTheta = 500000.0f;
	static constexpr uint32_t kFullRopePairs = 0u;
	static constexpr float kEmbedScale = 1.0f;
	static constexpr float kLogitSoftcap = 20.0f;
	static constexpr float kLogitScale = 0.19611613513818404f;
	static constexpr uint32_t kPageSlots = SPARK_MUSE_TP16_STAGE_PAGE_SLOTS;
	static constexpr bool kEmbedNorm = true;
	static constexpr float kNormEpsilon = 1e-5f;
	static constexpr float kPostEpsilon = 1e-8f;
	static constexpr bool kQkNormWeighted = false;
	static constexpr float kQueryMultiply = 3.87f;
	static constexpr bool kValueNorm = false;
	static constexpr float kAttentionScale = 0.08838834764831845f;
	static constexpr bool kFullRope = false;
	static constexpr bool kOutputGate = true;
	static constexpr bool kGeluMlp = false;
	static constexpr bool kUntiedHead = true;
	static constexpr bool kLayerScalar = false;
	static constexpr uint32_t kEmbedAliasFirst = 200091u;
	static constexpr uint32_t kEmbedAliasCount = 2u;
	static constexpr uint32_t kEmbedAliasRow = 0u;
	static const char *Tag(void) { return "sparkpipe_muse_glimmer"; }
	static const char *Name(void) { return "muse-glimmer-30b"; }
	static const char *ReportTag(void) { return "MUSE-STATE"; }
	static constexpr bool kPreNorm = false;
	static constexpr bool kSigmoidRouter = false;
	static constexpr float kRoutedScale = 1.0f;
	static constexpr bool kRouterFloat = false;
};

static_assert(GemmaShape<MuseGlimmerGeometry>::SlidingKvGeometry::kSlotBytes == SPARK_MUSE_TP16_STAGE_SLIDING_SLOT_BYTES,
	"a KV slot is [K: 2 x 128][V: 2 x 128] bf16");
static_assert(GemmaShape<MuseGlimmerGeometry>::kSlidingFused == SPARK_MUSE_TP16_STAGE_SLIDING_FUSED_WIDTH &&
	GemmaShape<MuseGlimmerGeometry>::kFullFused == SPARK_MUSE_TP16_STAGE_FULL_FUSED_WIDTH, "q|k|v is 4608 wide on both layer kinds");
static_assert(GemmaShape<MuseGlimmerGeometry>::kSlidingFused % SPARK_MUSE_TP16_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");

const SparkStageRunnerModelInterface *SparkMuseTp16StageModel(void)
{
	return GemmaModel<MuseGlimmerGeometry>::Interface();
}
