#ifndef SPARKPIPE_SPARK_MIMO26_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_MIMO26_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkMimo26FlashDeviceStageModelAt(uint32_t tp_degree);
const SparkStageRunnerModelInterface *SparkMimo26ProDeviceStageModelAt(uint32_t tp_degree);

#ifdef __cplusplus
}
#endif

#endif
