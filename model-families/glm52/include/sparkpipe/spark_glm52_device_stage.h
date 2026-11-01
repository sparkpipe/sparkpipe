#ifndef SPARKPIPE_SPARK_GLM52_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_GLM52_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkGlm53FullDeviceStageModelAt(uint32_t tp_degree);
const SparkStageRunnerModelInterface *SparkAxk2DeviceStageModelAt(uint32_t tp_degree);

#ifdef __cplusplus
}
#endif

#endif
