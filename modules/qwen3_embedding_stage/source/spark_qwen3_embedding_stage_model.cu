#include "sparkpipe/spark_qwen3_embedding_stage_model.h"
#include "qwen3_stage.cuh"

#ifndef QWEN3_EMBEDDING_TP_DEGREE
#error "QWEN3_EMBEDDING_TP_DEGREE must name the rank count of this build"
#endif

using Qwen3Embedding8bStage = Qwen3Embedding8bGeometry<QWEN3_EMBEDDING_TP_DEGREE>;

static_assert(Qwen3Shape<Qwen3Embedding8bStage>::Kv::kSlotBytes == SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 8 x 128][V: 8 x 128] bf16");
static_assert(Qwen3Shape<Qwen3Embedding8bStage>::kQkvWidth == SPARK_QWEN3_EMBEDDING_8B_STAGE_QKV_WIDTH, "the fused attention projection is query, key, value");

const SparkStageRunnerModelInterface *SparkQwen3Embedding8bStageModel(void)
{
	return Qwen3Model<Qwen3Embedding8bStage>::Interface();
}
