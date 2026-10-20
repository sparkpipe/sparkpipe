#include "sparkpipe/spark_qwen38_flash_stage_model.h"
#include "qwen38_hybrid_stage.cuh"

struct Qwen38FlashGeometry : QwenHybridDefaults
{
	static constexpr uint32_t kTpDegree = SPARK_QWEN38_FLASH_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_QWEN38_FLASH_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_QWEN38_FLASH_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 248320u;
	static constexpr uint32_t kPeriod = 4u;
	static constexpr uint32_t kPhase = 3u;
	static constexpr uint32_t kGdnKeyHeads = 16u;
	static constexpr uint32_t kGdnValueHeads = 48u;
	static constexpr uint32_t kHeads = 24u;
	static constexpr uint32_t kKvHeads = SPARK_QWEN38_FLASH_STAGE_KV_HEADS;
	static constexpr uint32_t kHeadDim = SPARK_QWEN38_FLASH_STAGE_HEAD_DIM;
	static constexpr uint32_t kRopeDim = 64u;
	static constexpr float kRopeTheta = 10000000.0f;
	static constexpr uint32_t kPageSlots = SPARK_QWEN38_FLASH_STAGE_PAGE_SLOTS;
	static constexpr bool kRankHeads = false;
	static constexpr bool kMoe = true;
	static constexpr uint32_t kExperts = 512u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 10u;
	static constexpr uint32_t kMoeInter = 640u;
	static constexpr uint32_t kSharedRows = 64u;
	static constexpr uint32_t kStreams = 4u;
	static constexpr uint32_t kHcLowRank = 320u;
	static constexpr uint32_t kPleLayer = 1u;
	static constexpr uint32_t kPleOrders = 2u;
	static constexpr uint32_t kPleHeadsPerOrder = 8u;
	static constexpr uint32_t kPleHeadDim = 160u;
	static constexpr uint32_t kPleConv = 4u;
	static constexpr uint32_t kPleDilation = 3u;
	static constexpr uint32_t kPleEos = 248044u;
	static constexpr bool kGdnSigmoidGate = true;
	static constexpr uint32_t kIndexHeads = 4u;
	static constexpr uint32_t kIndexDim = 128u;
	static constexpr uint32_t kIndexBudget = 2048u;
	static constexpr uint32_t kIndexRatio = 4u;
	static const char *Tag(void) { return "sparkpipe_qwen38_flash"; }
	static const char *Name(void) { return "qwen3.8-flash"; }
	static const char *DumpEnv(void) { return "SPARK_QWEN38_FLASH_LAYER_DUMP"; }
	static const char *ReportTag(void) { return "QWEN38_FLASH-STATE"; }
};

static_assert(QwenHybridShape<Qwen38FlashGeometry>::Kv::kSlotBytes == SPARK_QWEN38_FLASH_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 2 x 256][V: 2 x 256] bf16");
static_assert(QwenHybridShape<Qwen38FlashGeometry>::kQkvWidth == SPARK_QWEN38_FLASH_STAGE_QKV_WIDTH, "the attention projection is query|gate, key, value");
static_assert(QwenHybridShape<Qwen38FlashGeometry>::kAttnLayers == SPARK_QWEN38_FLASH_STAGE_KV_LAYERS, "one KV layer per full-attention layer");
static_assert(QwenHybridShape<Qwen38FlashGeometry>::kPleColumns * Qwen38FlashGeometry::kPleHeadDim == Qwen38FlashGeometry::kHidden,
	"the n-gram heads tile the hidden width");

const SparkStageRunnerModelInterface *SparkQwen38FlashStageModel(void)
{
	return QwenHybridModel<Qwen38FlashGeometry>::Interface();
}
