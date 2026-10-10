#include "sparkpipe/spark_hy4_tp16_stage_model.h"
#include "hy4_stage.cuh"

struct Hy4Geometry
{
	static constexpr uint32_t kTpDegree = SPARK_HY4_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_HY4_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_HY4_TP16_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 120832u;
	static constexpr uint32_t kStreams = 4u;
	static constexpr uint32_t kHeads = 64u;
	static constexpr uint32_t kNope = 192u;
	static constexpr uint32_t kRope = SPARK_HY4_TP16_STAGE_ROPE;
	static constexpr uint32_t kValueDim = 256u;
	static constexpr uint32_t kLatent = SPARK_HY4_TP16_STAGE_LATENT;
	static constexpr uint32_t kQLora = 2048u;
	static constexpr float kRopeTheta = 10000000.0f;
	static constexpr uint32_t kIndexHeads = 32u;
	static constexpr uint32_t kIndexDim = SPARK_HY4_TP16_STAGE_INDEX_DIM;
	static constexpr uint32_t kIndexTopk = 2048u;
	static constexpr uint32_t kIndexLayers = SPARK_HY4_TP16_STAGE_INDEX_LAYERS;
	static constexpr uint32_t kDenseLayers = 1u;
	static constexpr uint32_t kExperts = 256u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kMoeInter = 2048u;
	static constexpr float kRouteScale = 2.827f;
	static constexpr float kSwigluLimit = 10.0f;
	static constexpr float kEpsilon = 1e-5f;
	static constexpr float kHcEpsilon = 1e-6f;
	static constexpr float kHcMagnitude = 2.0f;
	static constexpr uint32_t kPageSlots = SPARK_HY4_TP16_STAGE_PAGE_SLOTS;
	static bool IndexerLayer(uint32_t layer) { return layer == 0u || layer % 4u == 1u; }
	static const char *Tag(void) { return "sparkpipe_hy4"; }
	static const char *Name(void) { return "hy4-preview"; }
	static const char *ReportTag(void) { return "HY4-STATE"; }
};

static_assert(HyShape<Hy4Geometry>::Kv::kSlotBytes == SPARK_HY4_TP16_STAGE_KV_SLOT_BYTES, "a latent slot is the normed latent and its rope part");
static_assert(HyShape<Hy4Geometry>::IndexKv::kSlotBytes == SPARK_HY4_TP16_STAGE_INDEX_SLOT_BYTES, "an index slot is one indexer key");

const SparkStageRunnerModelInterface *SparkHy4Tp16StageModel(void)
{
	return HyModel<Hy4Geometry>::Interface();
}
