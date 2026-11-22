#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_LOW(range) (0 ? range)
#define SPARK_NVRM_HIGH(range) (1 ? range)
#define SPARK_NVRM_VALUE(range, value) \
	((((uint64_t)(value)) & ((2ull << (SPARK_NVRM_HIGH(range) - SPARK_NVRM_LOW(range))) - 1u)) << SPARK_NVRM_LOW(range))

typedef struct SparkNvrmRm
{
	int control;
	int uvm;
	uint32_t client;
	uint64_t objects;
	char loaded_version[64];
} SparkNvrmRm;

SparkStatus SparkNvrmRmOpen(SparkNvrmRm *rm, const char *owner);
void SparkNvrmRmClose(SparkNvrmRm *rm);
SparkStatus SparkNvrmRmIoctl(int fd, uint32_t number, void *parameters, uint32_t bytes, const char *operation, const char *owner);
SparkStatus SparkNvrmRmAlloc(SparkNvrmRm *rm, uint32_t parent, uint32_t class_number, void *parameters, uint32_t bytes,
	uint32_t *handle, const char *owner);
SparkStatus SparkNvrmRmFree(SparkNvrmRm *rm, uint32_t parent, uint32_t handle, const char *owner);
SparkStatus SparkNvrmRmControl(SparkNvrmRm *rm, uint32_t object, uint32_t command, void *parameters, uint32_t bytes, const char *owner);
SparkStatus SparkNvrmRmUvm(SparkNvrmRm *rm, int fd, uint32_t command, void *parameters, uint32_t bytes, uint32_t *rm_status,
	const char *operation, const char *owner);
SparkStatus SparkNvrmRmStatus(uint32_t rm_status, const char *operation, const char *owner);
const char *SparkNvrmRmStatusText(uint32_t rm_status);
SparkStatus SparkNvrmRmOpenDevice(SparkNvrmRm *rm, uint32_t minor, int *fd, const char *owner);

#ifdef __cplusplus
}
#endif
