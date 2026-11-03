#include "sparkpipe/spark_qwen38_max_stage_model.h"
#include "qwen38_hybrid_stage.cuh"

using Qwen38MaxStage = Qwen38MaxGeometry<SPARK_QWEN38_MAX_STAGE_TP_DEGREE>;

static_assert(QwenHybridShape<Qwen38MaxStage>::Kv::kSlotBytes == SPARK_QWEN38_MAX_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 4 x 256][V: 4 x 256] bf16");
static_assert(QwenHybridShape<Qwen38MaxStage>::kQkvWidth == SPARK_QWEN38_MAX_STAGE_QKV_WIDTH, "the attention projection is query|gate, key, value");
static_assert(QwenHybridShape<Qwen38MaxStage>::kAttnLayers == SPARK_QWEN38_MAX_STAGE_KV_LAYERS, "one KV layer per full-attention layer");

const SparkStageRunnerModelInterface *SparkQwen38MaxStageModel(void)
{
	return QwenHybridModel<Qwen38MaxStage>::Interface();
}
