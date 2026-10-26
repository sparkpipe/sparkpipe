#include "sparkpipe/spark_solar_open2_tp16_stage_model.h"
#include "ling_stage.cuh"

struct SolarOpen2Geometry : LingDefaults
{
	static constexpr uint32_t kTpDegree = SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_SOLAR_OPEN2_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_SOLAR_OPEN2_TP16_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 196608u;
	static constexpr uint32_t kPeriod = SPARK_SOLAR_OPEN2_TP16_STAGE_PERIOD;
	static constexpr uint32_t kPhase = 0u;
	static constexpr uint32_t kDenseLayers = 0u;
	static constexpr float kNormEpsilon = 1e-5f;
	static constexpr uint32_t kKdaHeads = 64u;
	static constexpr uint32_t kKdaDim = 128u;
	static constexpr uint32_t kKdaRank = 128u;
	static constexpr bool kKdaDoubledBeta = true;
	static constexpr uint32_t kConv = 4u;
	static constexpr float kKdaLowerBound = -5.0f;
	static constexpr uint32_t kAttention = LING_ATTENTION_GQA;
	static constexpr uint32_t kGqaHeads = SPARK_SOLAR_OPEN2_TP16_STAGE_HEADS;
	static constexpr uint32_t kGqaKvHeads = SPARK_SOLAR_OPEN2_TP16_STAGE_KV_HEADS;
	static constexpr uint32_t kGqaHeadDim = SPARK_SOLAR_OPEN2_TP16_STAGE_HEAD_DIM;
	static constexpr uint32_t kExperts = 320u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kGroups = 1u;
	static constexpr uint32_t kTopkGroups = 1u;
	static constexpr uint32_t kMoeInter = 1280u;
	static constexpr uint32_t kSharedRows = 128u;
	static constexpr float kRouteScale = 1.0f;
	static constexpr uint32_t kPageSlots = SPARK_SOLAR_OPEN2_TP16_STAGE_PAGE_SLOTS;
	static const char *Tag(void) { return "sparkpipe_solar_open2"; }
	static const char *Name(void) { return "solar-open-2"; }
	static const char *ReportTag(void) { return "SOLAR-OPEN2-STATE"; }
};

static_assert(LingShape<SolarOpen2Geometry>::Kv::kSlotBytes == SPARK_SOLAR_OPEN2_TP16_STAGE_KV_SLOT_BYTES, "a GQA slot is every KV head's key then value");
static_assert(LingShape<SolarOpen2Geometry>::kAttnLayers == SPARK_SOLAR_OPEN2_TP16_STAGE_ATTN_LAYERS, "every fourth layer is GQA");
static_assert(LingShape<SolarOpen2Geometry>::kGqaFused == SPARK_SOLAR_OPEN2_TP16_STAGE_QKV_WIDTH, "the fused projection is q, then k, then v");

const SparkStageRunnerModelInterface *SparkSolarOpen2StageModel(void)
{
	return LingModel<SolarOpen2Geometry>::Interface();
}
