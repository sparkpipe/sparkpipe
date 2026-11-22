#ifndef SPARKPIPE_SPARK_QWEN3_EMBEDDING_STAGE_MODEL_H
#define SPARKPIPE_SPARK_QWEN3_EMBEDDING_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QWEN3_EMBEDDING_8B_STAGE_PAGE_SLOTS 64u
#define SPARK_QWEN3_EMBEDDING_8B_STAGE_HEADS 32u
#define SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_HEADS 8u
#define SPARK_QWEN3_EMBEDDING_8B_STAGE_HEAD_DIM 128u
#define SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_SLOT_BYTES (SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_HEADS * SPARK_QWEN3_EMBEDDING_8B_STAGE_HEAD_DIM * 2u * 2u)
#define SPARK_QWEN3_EMBEDDING_8B_STAGE_QKV_WIDTH 6144u
#define SPARK_QWEN3_EMBEDDING_8B_STAGE_HIDDEN 4096u
#define SPARK_QWEN3_EMBEDDING_8B_STAGE_LAYERS 36u

const SparkStageRunnerModelInterface *SparkQwen3Embedding8bStageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_qwen3_family.h"

template<uint32_t TP>
struct Qwen3Embedding8bGeometry
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_QWEN3_EMBEDDING_8B_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_QWEN3_EMBEDDING_8B_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 151665u;
	static constexpr uint32_t kFfn = 12288u;
	static constexpr uint32_t kHeads = SPARK_QWEN3_EMBEDDING_8B_STAGE_HEADS;
	static constexpr uint32_t kKvHeads = SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_HEADS;
	static constexpr uint32_t kHeadDim = SPARK_QWEN3_EMBEDDING_8B_STAGE_HEAD_DIM;
	static constexpr float kRopeTheta = 1000000.0f;
	static constexpr uint32_t kPageSlots = SPARK_QWEN3_EMBEDDING_8B_STAGE_PAGE_SLOTS;
	static constexpr bool kHeadless = true;
	static constexpr bool kTiedHead = false;
	static const char *Tag(void) { return "sparkpipe_qwen3_embedding_8b"; }
	static const char *Name(void) { return "qwen3-embedding-8b"; }
	static const char *ReportTag(void) { return "QWEN3_EMBEDDING_8B-STATE"; }
};
#endif

#endif
