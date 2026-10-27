#ifndef SPARKPIPE_SPARK_GEMMA4_26B_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_GEMMA4_26B_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkGemma4_26bDeviceStageModel(void);

#ifdef __cplusplus
}
#endif

#endif
