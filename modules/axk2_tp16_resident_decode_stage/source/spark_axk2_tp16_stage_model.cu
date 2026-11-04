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
	static constexpr uint32_t kHidden = Axk2Dimensions::kHidden;
	static constexpr uint32_t kLayers = Axk2Dimensions::kLayers;
	static constexpr uint32_t kFirstRouted = Axk2Dimensions::kFirstRouted;
	static constexpr uint32_t kVocab = Axk2Dimensions::kVocab;
	static constexpr uint32_t kHeads = Axk2Dimensions::kHeads;
	static constexpr uint32_t kLatent = Axk2Dimensions::kLatent;
	static constexpr uint32_t kRope = Axk2Dimensions::kRope;
	static constexpr uint32_t kNope = Axk2Dimensions::kNope;
	static constexpr uint32_t kValue = Axk2Dimensions::kValue;
	static constexpr uint32_t kQueryA = Axk2Dimensions::kQueryA;
	static constexpr uint32_t kSelected = Axk2Dimensions::kSelected;
	static constexpr uint32_t kMaxPositions = 262144u;
	static constexpr uint32_t kIndexHeads = 64u;
	static constexpr uint32_t kIndexDim = SPARK_AXK2_TP16_STAGE_INDEX_DIM;
	static constexpr uint32_t kExperts = Axk2Dimensions::kExperts;
	static constexpr uint32_t kTopK = Axk2Dimensions::kTopK;
	static constexpr uint32_t kExpertInter = Axk2Dimensions::kExpertInter;
	static constexpr uint32_t kDenseInter = Axk2Dimensions::kDenseInter;
	static constexpr float kRmsEpsilon = Axk2Dimensions::kRmsEpsilon;
	static constexpr float kIndexEpsilon = 1e-06f;
	static constexpr float kIndexScale = 0.0883883476f;
	static constexpr float kRopeTheta = Axk2Dimensions::kRopeTheta;
	static constexpr float kQkScale = Axk2Dimensions::kQkScale;
	static constexpr float kRoutedScale = Axk2Dimensions::kRoutedScale;
	static constexpr uint32_t kQueryGate = Axk2Dimensions::kQueryGate;
	static constexpr bool kQueryConcat = Axk2Dimensions::kQueryConcat;
	static constexpr uint32_t kNormGateRank = Axk2Dimensions::kNormGateRank;
	static constexpr LmRopePairing kIndexPairing = LM_ROPE_HALF_SPLIT;
	static constexpr uint32_t kExpertGroups = Axk2Dimensions::kExpertGroups;
	static constexpr uint32_t kTopGroups = Axk2Dimensions::kTopGroups;
	static constexpr float kRopeFactor = Axk2Dimensions::kRopeFactor;
	static constexpr float kRopeBetaFast = Axk2Dimensions::kRopeBetaFast;
	static constexpr float kRopeBetaSlow = Axk2Dimensions::kRopeBetaSlow;
	static constexpr uint32_t kRopeOriginalPositions = Axk2Dimensions::kRopeOriginalPositions;
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
	static constexpr uint32_t kExpertCodec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	static constexpr uint32_t kLinearCodec = SPARK_WEIGHT_CODEC_FP8_E4M3;
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
