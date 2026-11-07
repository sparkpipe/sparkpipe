#ifndef SPARKPIPE_SPARK_QWEN38_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_QWEN38_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkQwen38_27bDeviceStageModel(void);
const SparkStageRunnerModelInterface *SparkQwen38FlashDeviceStageModel(void);
const SparkStageRunnerModelInterface *SparkQwen38MaxDeviceStageModelAt(uint32_t tp_degree);

#ifdef __cplusplus
}
#endif

#endif
