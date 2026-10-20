#include "sparkpipe/spark_ling3_stage_model.h"
#include "ling_stage.cuh"

struct Ling3Geometry
{
	static constexpr uint32_t kTpDegree = SPARK_LING3_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_LING3_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_LING3_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 157184u;
	static constexpr uint32_t kPeriod = SPARK_LING3_STAGE_PERIOD;
	static constexpr uint32_t kPhase = 5u;
	static constexpr uint32_t kDenseLayers = 2u;
	static constexpr uint32_t kKdaHeads = 32u;
	static constexpr uint32_t kKdaDim = 128u;
	static constexpr uint32_t kConv = 4u;
	static constexpr float kKdaLowerBound = -5.0f;
	static constexpr uint32_t kMlaHeads = 32u;
	static constexpr uint32_t kNope = 128u;
	static constexpr uint32_t kRope = SPARK_LING3_STAGE_ROPE;
	static constexpr uint32_t kValueDim = 128u;
	static constexpr uint32_t kLatent = SPARK_LING3_STAGE_LATENT;
	static constexpr float kRopeTheta = 6000000.0f;
	static constexpr uint32_t kExperts = 512u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kGroups = 8u;
	static constexpr uint32_t kTopkGroups = 4u;
	static constexpr uint32_t kMoeInter = 768u;
	static constexpr uint32_t kSharedRows = 64u;
	static constexpr float kRouteScale = 2.5f;
	static constexpr uint32_t kPageSlots = SPARK_LING3_STAGE_PAGE_SLOTS;
	static float RoutedLimit(uint32_t layer) { return layer >= 35u ? 4.0f : 0.0f; }
	static float SharedLimit(uint32_t layer) { return layer >= 40u ? 7.0f : layer >= 34u ? 5.0f : 0.0f; }
	static const char *Tag(void) { return "sparkpipe_ling3"; }
	static const char *Name(void) { return "ling-3.0-flash"; }
	static const char *ReportTag(void) { return "LING3-STATE"; }
};

static_assert(LingShape<Ling3Geometry>::Kv::kSlotBytes == SPARK_LING3_STAGE_KV_SLOT_BYTES, "an MLA slot is the normed latent and its rope part");
static_assert(LingShape<Ling3Geometry>::kMlaLayers == SPARK_LING3_STAGE_MLA_LAYERS, "every sixth layer is MLA");

const SparkStageRunnerModelInterface *SparkLing3StageModel(void)
{
	return LingModel<Ling3Geometry>::Interface();
}
