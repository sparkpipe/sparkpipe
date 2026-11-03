#ifndef SPARKPIPE_SPARK_QWEN38_MAX_STAGE_MODEL_H
#define SPARKPIPE_SPARK_QWEN38_MAX_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QWEN38_MAX_STAGE_TP_DEGREE 16u
#define SPARK_QWEN38_MAX_STAGE_PAGE_SLOTS 64u
#define SPARK_QWEN38_MAX_STAGE_KV_HEADS 4u
#define SPARK_QWEN38_MAX_STAGE_HEAD_DIM 256u
#define SPARK_QWEN38_MAX_STAGE_KV_SLOT_BYTES (SPARK_QWEN38_MAX_STAGE_KV_HEADS * SPARK_QWEN38_MAX_STAGE_HEAD_DIM * 2u * 2u)
#define SPARK_QWEN38_MAX_STAGE_QKV_WIDTH 34816u
#define SPARK_QWEN38_MAX_STAGE_HIDDEN 8192u
#define SPARK_QWEN38_MAX_STAGE_LAYERS 92u
#define SPARK_QWEN38_MAX_STAGE_KV_LAYERS 23u

const SparkStageRunnerModelInterface *SparkQwen38MaxStageModel(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "sparkpipe/spark_qwen38_family.h"

template<uint32_t TP>
struct Qwen38MaxGeometry : QwenHybridDefaults
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = SPARK_QWEN38_MAX_STAGE_HIDDEN;
	static constexpr uint32_t kLayers = SPARK_QWEN38_MAX_STAGE_LAYERS;
	static constexpr uint32_t kVocab = 248320u;
	static constexpr uint32_t kPeriod = 4u;
	static constexpr uint32_t kPhase = 3u;
	static constexpr uint32_t kGdnKeyHeads = 16u;
	static constexpr uint32_t kGdnValueHeads = 128u;
	static constexpr uint32_t kHeads = 64u;
	static constexpr uint32_t kKvHeads = SPARK_QWEN38_MAX_STAGE_KV_HEADS;
	static constexpr uint32_t kHeadDim = SPARK_QWEN38_MAX_STAGE_HEAD_DIM;
	static constexpr uint32_t kRopeDim = 64u;
	static constexpr float kRopeTheta = 10000000.0f;
	static constexpr uint32_t kPageSlots = SPARK_QWEN38_MAX_STAGE_PAGE_SLOTS;
	static constexpr bool kRankHeads = true;
	static constexpr bool kMoe = true;
	static constexpr uint32_t kExperts = 512u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 10u;
	static constexpr uint32_t kMoeInter = 2048u;
	static constexpr uint32_t kSharedRows = 2048u / kTpDegree;
	static const char *Tag(void) { return "sparkpipe_qwen38_max"; }
	static const char *Name(void) { return "qwen3.8-max"; }
	static const char *DumpEnv(void) { return "SPARK_QWEN38_MAX_LAYER_DUMP"; }
	static const char *ReportTag(void) { return "QWEN38_MAX-STATE"; }
	static constexpr bool kGdnSigmoidGate = false;
};
#endif

#endif
