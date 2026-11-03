#include "sparkpipe/spark_qwen38_flash_stage_model.h"
#include "qwen38_hybrid_stage.cuh"

using Qwen38FlashStage = Qwen38FlashGeometry<SPARK_QWEN38_FLASH_STAGE_TP_DEGREE>;

static_assert(QwenHybridShape<Qwen38FlashStage>::Kv::kSlotBytes == SPARK_QWEN38_FLASH_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 2 x 256][V: 2 x 256] bf16");
static_assert(QwenHybridShape<Qwen38FlashStage>::kQkvWidth == SPARK_QWEN38_FLASH_STAGE_QKV_WIDTH, "the attention projection is query|gate, key, value");
static_assert(QwenHybridShape<Qwen38FlashStage>::kAttnLayers == SPARK_QWEN38_FLASH_STAGE_KV_LAYERS, "one KV layer per full-attention layer");
static_assert(QwenHybridShape<Qwen38FlashStage>::kPleColumns * Qwen38FlashStage::kPleHeadDim == Qwen38FlashStage::kHidden,
	"the n-gram heads tile the hidden width");

const SparkStageRunnerModelInterface *SparkQwen38FlashStageModel(void)
{
	return QwenHybridModel<Qwen38FlashStage>::Interface();
}
