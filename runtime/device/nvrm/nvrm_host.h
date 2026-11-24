#pragma once

#include <pthread.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "runtime/device/nvrm/nvrm_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_HOST_ITEMS 256u

typedef void (*SparkNvrmHostFunction)(void *context);

typedef enum SparkNvrmHostKind
{
	SPARK_NVRM_HOST_CALL = 1,
	SPARK_NVRM_HOST_COPY = 2,
	SPARK_NVRM_HOST_FILL = 3
} SparkNvrmHostKind;

typedef struct SparkNvrmHostItem
{
	SparkNvrmHostKind kind;
	uint64_t ticket;
	SparkNvrmHostFunction function;
	void *context;
	uint8_t *destination;
	const uint8_t *source;
	uint64_t bytes;
	uint8_t byte;
} SparkNvrmHostItem;

typedef struct SparkNvrmHostWorker
{
	SparkNvrmMemory gates;
	uint64_t ticket;
	pthread_t thread;
	pthread_mutex_t lock;
	pthread_cond_t changed;
	SparkNvrmHostItem items[SPARK_NVRM_HOST_ITEMS];
	uint32_t head;
	uint32_t count;
	uint32_t started;
	uint32_t stopping;
} SparkNvrmHostWorker;

SparkStatus SparkNvrmHostOpen(SparkNvrmGpu *gpu, SparkNvrmHostWorker *worker, const char *owner);
void SparkNvrmHostClose(SparkNvrmGpu *gpu, SparkNvrmHostWorker *worker);
SparkStatus SparkNvrmHostWork(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmHostWorker *worker, const SparkNvrmHostItem *item, uint64_t *value,
	const char *owner);
void SparkNvrmHostQueue(SparkNvrmHostWorker *worker, const SparkNvrmHostItem *item);
uint64_t SparkNvrmHostGate(const SparkNvrmHostWorker *worker, uint32_t done);

#ifdef __cplusplus
}
#endif
