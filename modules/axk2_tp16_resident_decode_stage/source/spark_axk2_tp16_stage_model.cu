#include "sparkpipe/spark_axk2_tp16_stage_model.h"
#include "glm52_stage.cuh"

#ifndef AXK2_CONTRACT_SHA256
#error "AXK2_CONTRACT_SHA256 must identify the pack contract"
#endif
#ifndef AXK2_MODEL_REVISION
#error "AXK2_MODEL_REVISION must name the checkpoint revision"
#endif

static uint32_t Axk2HasIndexer(uint32_t layer)
{
	return layer < SPARK_AXK2_TP16_STAGE_LAYERS ? 1u : 0u;
}

struct Axk2Geometry : GlmDefaults
{
	static constexpr uint32_t kTpDegree = SPARK_AXK2_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kIndexLayers = SPARK_AXK2_TP16_STAGE_INDEX_LAYERS;
	static constexpr uint32_t kHidden = SPARK_AXK2_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_AXK2_TP16_STAGE_LAYERS;
	static constexpr uint32_t kFirstRouted = 1u;
	static constexpr uint32_t kVocab = 163840u;
	static constexpr uint32_t kHeads = 64u;
	static constexpr uint32_t kLatent = SPARK_AXK2_TP16_STAGE_LATENT;
	static constexpr uint32_t kRope = SPARK_AXK2_TP16_STAGE_ROPE;
	static constexpr uint32_t kNope = 128u;
	static constexpr uint32_t kValue = 128u;
	static constexpr uint32_t kQueryA = 1536u;
	static constexpr uint32_t kSelected = 2048u;
	static constexpr uint32_t kMaxPositions = 262144u;
	static constexpr uint32_t kIndexHeads = 64u;
	static constexpr uint32_t kIndexDim = SPARK_AXK2_TP16_STAGE_INDEX_DIM;
	static constexpr uint32_t kExperts = 256u;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kExpertInter = 2048u;
	static constexpr uint32_t kDenseInter = 18432u;
	static constexpr float kRmsEpsilon = 1e-06f;
	static constexpr float kIndexEpsilon = 1e-06f;
	static constexpr float kIndexScale = 0.0883883476f;
	static constexpr float kRopeTheta = 1000000.0f;
	static constexpr float kQkScale = 0.0825202391f;
	static constexpr float kRoutedScale = 2.5f;
	static constexpr uint32_t kQueryGate = 128u;
	static constexpr bool kQueryConcat = true;
	static constexpr uint32_t kNormGateRank = 16u;
	static constexpr LmRopePairing kIndexPairing = LM_ROPE_HALF_SPLIT;
	static constexpr uint32_t kExpertGroups = 8u;
	static constexpr uint32_t kTopGroups = 4u;
	static constexpr float kRopeFactor = 2.0f;
	static constexpr float kRopeBetaFast = 32.0f;
	static constexpr float kRopeBetaSlow = 1.0f;
	static constexpr uint32_t kRopeOriginalPositions = 131072u;
	static uint32_t FullIndexer(uint32_t layer) { return Axk2HasIndexer(layer); }
	static const SparkGlm52StagePackGeometry *PackGeometry(void)
	{
		static const SparkGlm52StagePackGeometry geometry =
		{
			kLayers, kFirstRouted, kHidden, kVocab, kHeads, kQueryA, 2u * kQueryA, kNope + kRope + kQueryGate, kNope, kValue, kLatent, kRope,
			kIndexHeads, kIndexDim, kDenseInter, kExpertInter, kExperts, kNormGateRank, Axk2HasIndexer
		};
		return &geometry;
	}
	static const char *Tag(void) { return "sparkpipe_axk2"; }
	static const char *ReportTag(void) { return "AXK2-STATE"; }
	static const char *WeightdModel(void) { return "axk2_stage"; }
	static const char *Revision(void) { return AXK2_MODEL_REVISION; }
	static const char *Contract(void) { return AXK2_CONTRACT_SHA256; }
};

typedef GlmShape<Axk2Geometry> Axk2Shape;

static_assert(Axk2Shape::kSlotBytes == SPARK_AXK2_TP16_STAGE_KV_SLOT_BYTES, "a latent slot is the normed latent and its rope part");
static_assert(GlmIndexKv<Axk2Shape>::kSlotBytes == SPARK_AXK2_TP16_STAGE_INDEX_SLOT_BYTES, "an index slot is one indexer key");
static_assert(SPARK_AXK2_TP16_STAGE_PAGE_SLOTS == GLM_PAGE_SLOTS, "64-token pages");
static_assert(Axk2Shape::kQueryHead * (Axk2Geometry::kHeads / Axk2Geometry::kTpDegree) % 128u == 0u, "a rank's query rows cover whole fp8 scale blocks");

const SparkStageRunnerModelInterface *SparkAxk2Tp16StageModel(void)
{
	return GlmModel<Axk2Geometry>::Interface();
}
