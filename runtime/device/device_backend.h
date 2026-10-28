#ifndef SPARKPIPE_DEVICE_BACKEND_H
#define SPARKPIPE_DEVICE_BACKEND_H

#include <stdint.h>

#include "sparkpipe/spark_device.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DEVICE_STAMPS 8u
#define SPARK_DEVICE_OWNER_BYTES 48u

typedef enum SparkDeviceDirection
{
	SPARK_DEVICE_HOST_TO_HOST = 1,
	SPARK_DEVICE_HOST_TO_DEVICE = 2,
	SPARK_DEVICE_DEVICE_TO_HOST = 3,
	SPARK_DEVICE_DEVICE_TO_DEVICE = 4
} SparkDeviceDirection;

typedef struct SparkDeviceStamp
{
	uint64_t queue_id;
	uint64_t sequence;
} SparkDeviceStamp;

typedef struct SparkDeviceAllocation
{
	SparkDeviceBuffer view;
	SparkDeviceAddress device;
	void *host;
	uint64_t bytes;
	SparkDeviceSpace space;
	uint32_t access;
	uint32_t registered;
	void *backend;
	SparkDeviceStamp stamps[SPARK_DEVICE_STAMPS];
	char owner[SPARK_DEVICE_OWNER_BYTES];
} SparkDeviceAllocation;

struct SparkDeviceQueue
{
	uint64_t id;
	uint64_t submitted;
	void *backend;
	SparkDeviceQueue *next;
	char owner[SPARK_DEVICE_OWNER_BYTES];
};

struct SparkDeviceEvent
{
	uint64_t id;
	uint64_t queue_id;
	uint64_t sequence;
	uint32_t recorded;
	void *backend;
	SparkDeviceStamp stamps[SPARK_DEVICE_STAMPS];
};

struct SparkDeviceLibrary
{
	uint64_t id;
	uint32_t kernels;
	void *backend;
	SparkDeviceLibrary *next;
	char owner[SPARK_DEVICE_OWNER_BYTES];
};

struct SparkDeviceKernel
{
	uint64_t id;
	SparkDeviceLibrary *library;
	SparkDeviceKernelInfo info;
	void *backend;
	SparkDeviceKernel *next;
	SparkDeviceStamp stamps[SPARK_DEVICE_STAMPS];
	char name[SPARK_DEVICE_KERNEL_NAME_BYTES];
};

SparkStatus SparkDeviceRecordError(SparkStatus status, const char *operation, const char *owner, int32_t backend_code, const char *text);

SparkStatus SparkDeviceBackendInitialize(const SparkDeviceConfiguration *configuration, SparkDeviceProperties *properties);
SparkStatus SparkDeviceBackendShutdown(void);

SparkStatus SparkDeviceBackendAllocate(SparkDeviceAllocation *allocation);
SparkStatus SparkDeviceBackendRegister(SparkDeviceAllocation *allocation, void *host);
SparkStatus SparkDeviceBackendMapFile(SparkDeviceAllocation *allocation, int fd, uint64_t offset);
void SparkDeviceBackendRelease(SparkDeviceAllocation *allocation);

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes);
SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes);

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue);
void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue);
SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns);
uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue);
SparkStatus SparkDeviceBackendQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context);

SparkStatus SparkDeviceBackendEventCreate(SparkDeviceEvent *event);
void SparkDeviceBackendEventDestroy(SparkDeviceEvent *event);
SparkStatus SparkDeviceBackendEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue);
SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event);
SparkStatus SparkDeviceBackendEventWait(SparkDeviceEvent *event, uint64_t timeout_ns);
SparkStatus SparkDeviceBackendEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds);

SparkStatus SparkDeviceBackendLibraryLoad(SparkDeviceLibrary *library, const char *path);
void SparkDeviceBackendLibraryUnload(SparkDeviceLibrary *library);
SparkStatus SparkDeviceBackendKernelCreate(SparkDeviceKernel *kernel, const char *name);
void SparkDeviceBackendKernelDestroy(SparkDeviceKernel *kernel);
SparkStatus SparkDeviceBackendKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch,
	SparkDeviceAllocation *const *allocations);

#ifdef __cplusplus
}
#endif

#endif
