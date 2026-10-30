#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_COLLECTIVE_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_COLLECTIVE_H

#include <stdint.h>

#include "sparkpipe/spark_device.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DEVICE_COLLECTIVE_MAX_RANKS 16u

typedef struct SparkDeviceCollective
{
	void *context;
	uint32_t degree;
	SparkStatus (*exchange)(void *context, uint32_t rank, SparkDeviceQueue *queue, SparkDeviceRange local, uint64_t source_stride, uint64_t bytes,
		SparkDeviceRange lanes);
	void (*abort)(void *context);
} SparkDeviceCollective;

SparkStatus SparkDeviceCollectiveLocalCreate(SparkDeviceCollective *collective, uint32_t degree);
void SparkDeviceCollectiveLocalDestroy(SparkDeviceCollective *collective);
SparkStatus SparkDeviceCollectiveTcpCreate(SparkDeviceCollective *collective, uint32_t degree, uint32_t rank, const char *const *hosts,
	uint16_t base_port);
void SparkDeviceCollectiveTcpDestroy(SparkDeviceCollective *collective);

#ifdef __cplusplus
}
#endif

#endif
