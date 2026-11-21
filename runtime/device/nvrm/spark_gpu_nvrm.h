#ifndef SPARKPIPE_RUNTIME_DEVICE_NVRM_SPARK_GPU_NVRM_H
#define SPARKPIPE_RUNTIME_DEVICE_NVRM_SPARK_GPU_NVRM_H

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_device.h"

#ifdef __cplusplus
#define SPARK_GPU_DEFAULT(value) = value
extern "C" {
#else
#define SPARK_GPU_DEFAULT(value)
#endif

typedef int SparkGpuError;
typedef SparkDeviceQueue *SparkGpuStream;

#define SPARK_GPU_SUCCESS 0
#define SPARK_GPU_ERROR_INVALID_VALUE 1
#define SPARK_GPU_ERROR_MEMORY_ALLOCATION 2
#define SPARK_GPU_ERROR_INVALID_DEVICE 101
#define SPARK_GPU_ERROR_NOT_READY 600
#define SPARK_GPU_ERROR_NOT_SUPPORTED 801
#define SPARK_GPU_ERROR_UNKNOWN 999

SparkGpuError SparkGpuGetLastError(void);
SparkGpuError SparkGpuPeekAtLastError(void);
const char *SparkGpuGetErrorString(SparkGpuError error);

SparkGpuError SparkNvrmGpuDefaultStream(SparkGpuStream *stream);
SparkGpuError SparkNvrmGpuRelease(void);
SparkGpuError SparkNvrmLaunchPacked(const void *function, const uint32_t grid[3], const uint32_t block[3], size_t shared_bytes, SparkGpuStream stream,
	const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes);
SparkGpuError SparkNvrmLaunchUnsupported(const void *function, const char *what);
SparkGpuError SparkNvrmSharedMemoryOptIn(const void *function, uint32_t shared_bytes);

#ifdef __cplusplus
}
#endif

#endif
