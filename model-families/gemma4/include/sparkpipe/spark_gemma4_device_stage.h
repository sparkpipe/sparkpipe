#ifndef SPARKPIPE_SPARK_GEMMA4_DEVICE_STAGE_H
#define SPARKPIPE_SPARK_GEMMA4_DEVICE_STAGE_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

const SparkStageRunnerModelInterface *SparkGemma4_26bDeviceStageModel(void);
const SparkStageRunnerModelInterface *SparkGemma4_31bDeviceStageModel(void);
const SparkStageRunnerModelInterface *SparkMuseGlimmerDeviceStageModel(void);
const SparkStageRunnerModelInterface *SparkGemma4_26bDeviceStageModelAt(uint32_t tp_degree);
const SparkStageRunnerModelInterface *SparkGemma4_31bDeviceStageModelAt(uint32_t tp_degree);
const SparkStageRunnerModelInterface *SparkMuseGlimmerDeviceStageModelAt(uint32_t tp_degree);
const SparkStageRunnerModelInterface *SparkLagunaDeviceStageModelAt(uint32_t tp_degree);

#ifdef __cplusplus
}
#endif

#endif
