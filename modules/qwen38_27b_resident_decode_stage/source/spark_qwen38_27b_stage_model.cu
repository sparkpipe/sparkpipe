#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "qwen38_hybrid_stage.cuh"

struct Qwen38_27bGeometry
{
	static constexpr uint32_t kTpDegree = SPARK_QWEN38_27B_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_QWEN38_27B_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = 64u;
	static constexpr uint32_t kVocab = 248320u;
	static constexpr uint32_t kPeriod = 4u;
	static constexpr uint32_t kPhase = 3u;
	static constexpr uint32_t kGdnKeyHeads = 16u;
	static constexpr uint32_t kGdnValueHeads = 48u;
	static constexpr uint32_t kHeads = 24u;
	static constexpr uint32_t kKvHeads = SPARK_QWEN38_27B_STAGE_KV_HEADS;
	static constexpr uint32_t kHeadDim = SPARK_QWEN38_27B_STAGE_HEAD_DIM;
	static constexpr uint32_t kRopeDim = 64u;
	static constexpr float kRopeTheta = 10000000.0f;
	static constexpr uint32_t kPageSlots = SPARK_QWEN38_27B_STAGE_PAGE_SLOTS;
	static constexpr bool kRankHeads = false;
	static constexpr bool kMoe = false;
	static constexpr uint32_t kExperts = 0u;
	static constexpr uint32_t kLocalExperts = 0u;
	static constexpr uint32_t kTopK = 0u;
	static constexpr uint32_t kMoeInter = 0u;
	static constexpr uint32_t kSharedRows = 0u;
	static const char *Tag(void) { return "sparkpipe_qwen38_27b"; }
	static const char *Name(void) { return "qwen3.8-27b"; }
	static const char *DumpEnv(void) { return "SPARK_QWEN38_27B_LAYER_DUMP"; }
	static const char *ReportTag(void) { return "QWEN38_27B-STATE"; }
};

static_assert(QwenHybridShape<Qwen38_27bGeometry>::Kv::kSlotBytes == SPARK_QWEN38_27B_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 4 x 256][V: 4 x 256] bf16");
static_assert(QwenHybridShape<Qwen38_27bGeometry>::kQkvWidth == SPARK_QWEN38_27B_STAGE_QKV_WIDTH, "the fused attention projection is query|gate, key, value");

const SparkStageRunnerModelInterface *SparkQwen38_27bStageModel(void)
{
	return QwenHybridModel<Qwen38_27bGeometry>::Interface();
}
