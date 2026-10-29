#ifndef SPARKPIPE_SPARK_QWEN38_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_QWEN38_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkQwen38_27bDeviceStageModel(void);
const SparkStageRunnerModelInterface *SparkQwen38FlashDeviceStageModel(void);

#ifdef __cplusplus
}
#endif

#endif
