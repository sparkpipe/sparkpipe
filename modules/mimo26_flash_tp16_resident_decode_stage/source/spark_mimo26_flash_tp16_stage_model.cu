#include "sparkpipe/spark_mimo26_flash_tp16_stage_model.h"
#include "mimo26_stage.cuh"

struct Mimo26FlashGeometry
{
	static constexpr uint32_t kTpDegree = SPARK_MIMO26_FLASH_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kHidden = SPARK_MIMO26_FLASH_TP16_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_MIMO26_FLASH_TP16_STAGE_LAYERS;
	static constexpr uint32_t kFullLayers = SPARK_MIMO26_FLASH_TP16_STAGE_FULL_LAYERS;
	static constexpr uint32_t kVocab = 152576u;
	static constexpr uint32_t kHeads = SPARK_MIMO26_FLASH_TP16_STAGE_HEADS;
	static constexpr uint32_t kSlidingKvHeads = SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_KV_HEADS;
	static constexpr uint32_t kFullKvHeads = SPARK_MIMO26_FLASH_TP16_STAGE_FULL_KV_HEADS;
	static constexpr uint32_t kHeadDim = SPARK_MIMO26_FLASH_TP16_STAGE_HEAD_DIM;
	static constexpr uint32_t kValueDim = SPARK_MIMO26_FLASH_TP16_STAGE_VALUE_DIM;
	static constexpr uint32_t kRopeDim = 64u;
	static constexpr uint32_t kWindow = 128u;
	static constexpr float kSlidingTheta = 10000.0f;
	static constexpr float kFullTheta = 10000000.0f;
	static constexpr float kValueScale = 0.707f;
	static constexpr float kAttentionScale = 0.07216878364870322f;
	static constexpr float kEpsilon = 1e-6f;
	static constexpr uint32_t kDenseLayers = 1u;
	static constexpr uint32_t kDenseInter = 16384u;
	static constexpr uint32_t kExperts = 256u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kMoeInter = 2048u;
	static constexpr uint32_t kPageSlots = SPARK_MIMO26_FLASH_TP16_STAGE_PAGE_SLOTS;
	static bool FullLayer(uint32_t layer) { return layer < kLayers && "011110111110111110111110111110111110111110111110"[layer] == '0'; }
	static const char *Tag(void) { return "sparkpipe_mimo26_flash"; }
	static const char *Name(void) { return "mimo-v2.6-flash"; }
	static const char *ReportTag(void) { return "MIMO26-FLASH-STATE"; }
};

static_assert(MimoShape<Mimo26FlashGeometry>::SlidingKv::kSlotBytes == SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_SLOT_BYTES, "a sliding slot is [K: kv x 192][V: kv x 128] bf16");
static_assert(MimoShape<Mimo26FlashGeometry>::FullKv::kSlotBytes == SPARK_MIMO26_FLASH_TP16_STAGE_FULL_SLOT_BYTES, "a full slot is [K: kv x 192][V: kv x 128] bf16");
static_assert(MimoShape<Mimo26FlashGeometry>::kSlidingFused == SPARK_MIMO26_FLASH_TP16_STAGE_SLIDING_FUSED && MimoShape<Mimo26FlashGeometry>::kFullFused == SPARK_MIMO26_FLASH_TP16_STAGE_FULL_FUSED,
	"the fused q|k|v widths match the module header");

const SparkStageRunnerModelInterface *SparkMimo26FlashTp16StageModel(void)
{
	return MimoModel<Mimo26FlashGeometry>::Interface();
}
