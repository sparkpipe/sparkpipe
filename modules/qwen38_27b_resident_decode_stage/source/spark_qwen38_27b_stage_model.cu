#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "qwen38_hybrid_stage.cuh"

using Qwen38_27bStage = Qwen38_27bGeometry<SPARK_QWEN38_27B_STAGE_TP_DEGREE>;

static_assert(QwenHybridShape<Qwen38_27bStage>::Kv::kSlotBytes == SPARK_QWEN38_27B_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 4 x 256][V: 4 x 256] bf16");
static_assert(QwenHybridShape<Qwen38_27bStage>::kQkvWidth == SPARK_QWEN38_27B_STAGE_QKV_WIDTH, "the fused attention projection is query|gate, key, value");

const SparkStageRunnerModelInterface *SparkQwen38_27bStageModel(void)
{
	return QwenHybridModel<Qwen38_27bStage>::Interface();
}
