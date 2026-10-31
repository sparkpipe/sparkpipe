#ifndef SPARKPIPE_SPARK_HY4_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_HY4_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkHy4DeviceStageModelAt(uint32_t tp_degree);

#ifdef __cplusplus
}
#endif

#endif
