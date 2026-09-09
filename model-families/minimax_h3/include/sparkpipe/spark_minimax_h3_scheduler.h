#pragma once

#include <stdint.h>

#include "sparkpipe/spark_minimax_h3_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MINIMAX_H3_SCHEDULER_VIDEO_SHIFT SPARK_MINIMAX_H3_SCHEDULER_SHIFT
#define SPARK_MINIMAX_H3_SCHEDULER_AUDIO_SHIFT SPARK_MINIMAX_H3_AUDIO_SCHEDULER_SHIFT

int32_t SparkMinimaxH3SchedulerBuildSigmas(float shift, uint32_t sigma_point_count,
	float *sigmas_out, uint32_t *sigma_count_out);

void SparkMinimaxH3SchedulerTimestepsFromSigmas(const float *sigmas, uint32_t sigma_count,
	float *timesteps_out, uint32_t *timestep_count_out);

void SparkMinimaxH3SchedulerStepElement(float timestep, float sigma, float sigma_next,
	float sample, float velocity, float *sample_next_out);

#ifdef __cplusplus
}
#endif
