#include "sparkpipe/spark_qwen38_max_stage_model.h"
#include "qwen38_hybrid_stage.cuh"

struct Qwen38MaxGeometry
{
	static constexpr uint32_t kTpDegree = SPARK_QWEN38_MAX_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_QWEN38_MAX_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_QWEN38_MAX_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 248320u;
	static constexpr uint32_t kPeriod = 4u;
	static constexpr uint32_t kPhase = 3u;
	static constexpr uint32_t kGdnKeyHeads = 16u;
	static constexpr uint32_t kGdnValueHeads = 128u;
	static constexpr uint32_t kHeads = 64u;
	static constexpr uint32_t kKvHeads = SPARK_QWEN38_MAX_STAGE_KV_HEADS;
	static constexpr uint32_t kHeadDim = SPARK_QWEN38_MAX_STAGE_HEAD_DIM;
	static constexpr uint32_t kRopeDim = 64u;
	static constexpr float kRopeTheta = 10000000.0f;
	static constexpr uint32_t kPageSlots = SPARK_QWEN38_MAX_STAGE_PAGE_SLOTS;
	static constexpr bool kRankHeads = true;
	static constexpr bool kMoe = true;
	static constexpr uint32_t kExperts = 512u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 10u;
	static constexpr uint32_t kMoeInter = 2048u;
	static constexpr uint32_t kSharedRows = 2048u / kTpDegree;
	static const char *Tag(void) { return "sparkpipe_qwen38_max"; }
	static const char *Name(void) { return "qwen3.8-max"; }
	static const char *DumpEnv(void) { return "SPARK_QWEN38_MAX_LAYER_DUMP"; }
	static const char *ReportTag(void) { return "QWEN38_MAX-STATE"; }
};

static_assert(QwenHybridShape<Qwen38MaxGeometry>::Kv::kSlotBytes == SPARK_QWEN38_MAX_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 4 x 256][V: 4 x 256] bf16");
static_assert(QwenHybridShape<Qwen38MaxGeometry>::kQkvWidth == SPARK_QWEN38_MAX_STAGE_QKV_WIDTH, "the attention projection is query|gate, key, value");
static_assert(QwenHybridShape<Qwen38MaxGeometry>::kAttnLayers == SPARK_QWEN38_MAX_STAGE_KV_LAYERS, "one KV layer per full-attention layer");

const SparkStageRunnerModelInterface *SparkQwen38MaxStageModel(void)
{
	return QwenHybridModel<Qwen38MaxGeometry>::Interface();
}
