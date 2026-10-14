#include "sparkpipe/spark_gemma4_26b_stage_model.h"
#include "gemma4_stage.cuh"

struct Gemma4_26bGeometry : GemmaDefaults
{
	static constexpr uint32_t kTpDegree = SPARK_GEMMA4_26B_STAGE_TP_DEGREE;
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
};

static_assert(GemmaShape<Gemma4_26bGeometry>::SlidingKvGeometry::kSlotBytes == SPARK_GEMMA4_26B_STAGE_SLIDING_SLOT_BYTES,
	"a sliding KV slot is [K: 8 x 256][V: 8 x 256] bf16");
static_assert(GemmaShape<Gemma4_26bGeometry>::FullKvGeometry::kSlotBytes == SPARK_GEMMA4_26B_STAGE_FULL_SLOT_BYTES,
	"a full KV slot is [K: 2 x 512][V: 2 x 512] bf16");
static_assert(GemmaShape<Gemma4_26bGeometry>::kSlidingFused == SPARK_GEMMA4_26B_STAGE_SLIDING_FUSED_WIDTH &&
	GemmaShape<Gemma4_26bGeometry>::kFullFused == SPARK_GEMMA4_26B_STAGE_FULL_FUSED_WIDTH, "sliding q|k|v is 8192 wide and full q|k is 9216 wide");
static_assert(GemmaShape<Gemma4_26bGeometry>::kSlidingFused % SPARK_GEMMA4_26B_STAGE_TP_DEGREE == 0u &&
	GemmaShape<Gemma4_26bGeometry>::kFullFused % SPARK_GEMMA4_26B_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");

const SparkStageRunnerModelInterface *SparkGemma4_26bStageModel(void)
{
	return GemmaModel<Gemma4_26bGeometry>::Interface();
}
