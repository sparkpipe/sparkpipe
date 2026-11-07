#ifndef SPARKPIPE_SPARK_GLM5_NEXT_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_GLM5_NEXT_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkGlm53FlashDeviceStageModelAt(uint32_t tp_degree);

#ifdef __cplusplus
}
#endif

#endif
