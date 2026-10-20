#ifndef SPARKPIPE_SPARK_QWEN38_27B_STAGE_MODEL_H
#define SPARKPIPE_SPARK_QWEN38_27B_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QWEN38_27B_STAGE_TP_DEGREE 16u
#define SPARK_QWEN38_27B_STAGE_PAGE_SLOTS 64u
#define SPARK_QWEN38_27B_STAGE_KV_HEADS 4u
#define SPARK_QWEN38_27B_STAGE_HEAD_DIM 256u
#define SPARK_QWEN38_27B_STAGE_KV_SLOT_BYTES (SPARK_QWEN38_27B_STAGE_KV_HEADS * SPARK_QWEN38_27B_STAGE_HEAD_DIM * 2u * 2u)
#define SPARK_QWEN38_27B_STAGE_QKV_WIDTH 14336u
#define SPARK_QWEN38_27B_STAGE_HIDDEN 5120u

const SparkStageRunnerModelInterface *SparkQwen38_27bStageModel(void);

#ifdef __cplusplus
}
#endif

#endif
