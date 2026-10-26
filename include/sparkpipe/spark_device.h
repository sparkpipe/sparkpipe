#ifndef SPARKPIPE_SPARK_DEVICE_H
#define SPARKPIPE_SPARK_DEVICE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t SparkDeviceAddress;

typedef enum SparkDeviceSpace
{
	SPARK_DEVICE_SPACE_DEVICE_PRIVATE = 1,
	SPARK_DEVICE_SPACE_SHARED_COHERENT = 2,
	SPARK_DEVICE_SPACE_HOST_PINNED = 3,
	SPARK_DEVICE_SPACE_HOST_PAGEABLE = 4,
	SPARK_DEVICE_SPACE_FILE_BACKED = 5
} SparkDeviceSpace;

#define SPARK_DEVICE_ACCESS_HOST_READ 1u
#define SPARK_DEVICE_ACCESS_HOST_WRITE 2u
#define SPARK_DEVICE_ACCESS_DEVICE_READ 4u
#define SPARK_DEVICE_ACCESS_DEVICE_WRITE 8u

#define SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY 1u
#define SPARK_DEVICE_PROPERTY_SHARED_ADDRESS 2u

#define SPARK_DEVICE_WAIT_FOREVER UINT64_MAX

#define SPARK_DEVICE_BIND_READ 1u
#define SPARK_DEVICE_BIND_WRITE 2u
#define SPARK_DEVICE_MAX_BINDINGS 30u
#define SPARK_DEVICE_MAX_CONSTANT_BYTES 4096u
#define SPARK_DEVICE_KERNEL_NAME_BYTES 64u

typedef struct SparkDeviceBuffer
{
	void *host;
	SparkDeviceAddress device;
	uint64_t bytes;
	uint64_t handle;
	SparkDeviceSpace space;
	uint32_t access;
} SparkDeviceBuffer;

typedef struct SparkDeviceRange
{
	const SparkDeviceBuffer *buffer;
	uint64_t offset;
} SparkDeviceRange;

typedef struct SparkDeviceQueue SparkDeviceQueue;
typedef struct SparkDeviceEvent SparkDeviceEvent;
typedef struct SparkDeviceLibrary SparkDeviceLibrary;
typedef struct SparkDeviceKernel SparkDeviceKernel;
typedef void (*SparkDeviceHostFunction)(void *context);

typedef struct SparkDeviceGrid
{
	uint32_t x;
	uint32_t y;
	uint32_t z;
} SparkDeviceGrid;

typedef struct SparkDeviceBinding
{
	SparkDeviceRange range;
	uint64_t bytes;
	uint32_t access;
} SparkDeviceBinding;

typedef struct SparkDeviceLaunch
{
	SparkDeviceGrid groups;
	SparkDeviceGrid threads;
	uint32_t shared_bytes;
	uint32_t constant_bytes;
	const void *constants;
	const SparkDeviceBinding *bindings;
	uint32_t binding_count;
} SparkDeviceLaunch;

typedef struct SparkDeviceKernelInfo
{
	uint32_t max_threads;
	uint32_t simd_width;
	uint32_t static_shared_bytes;
} SparkDeviceKernelInfo;

typedef struct SparkDeviceProperties
{
	char backend[16];
	char target[64];
	char name[64];
	uint64_t memory_budget_bytes;
	uint64_t max_buffer_bytes;
	uint32_t page_bytes;
	uint32_t compute_units;
	uint32_t simd_width;
	uint32_t properties;
	uint32_t max_group_threads;
	uint32_t max_shared_bytes;
} SparkDeviceProperties;

typedef struct SparkDeviceConfiguration
{
	uint32_t ordinal;
	uint64_t memory_budget_bytes;
} SparkDeviceConfiguration;

typedef struct SparkDeviceError
{
	SparkStatus status;
	int32_t backend_code;
	char operation[32];
	char owner[48];
	char text[128];
} SparkDeviceError;

SparkStatus SparkDeviceInitialize(const SparkDeviceConfiguration *configuration);
SparkStatus SparkDeviceShutdown(void);
const SparkDeviceProperties *SparkDeviceGetProperties(void);
const SparkDeviceError *SparkDeviceLastError(void);
uint32_t SparkDeviceFaulted(void);
SparkStatus SparkDeviceSynchronize(uint64_t timeout_ns);

SparkStatus SparkDeviceAllocate(SparkDeviceBuffer *buffer, SparkDeviceSpace space, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceRegister(SparkDeviceBuffer *buffer, void *host, uint64_t bytes, uint32_t access, const char *owner);
SparkStatus SparkDeviceMapFile(SparkDeviceBuffer *buffer, int fd, uint64_t offset, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceFree(SparkDeviceBuffer *buffer);
SparkStatus SparkDeviceResolve(SparkDeviceAddress address, uint64_t bytes, SparkDeviceRange *range);
SparkStatus SparkDeviceAddressOf(SparkDeviceRange range, uint64_t bytes, SparkDeviceAddress *address);
SparkStatus SparkDeviceHostOf(SparkDeviceRange range, uint64_t bytes, void **host);

SparkStatus SparkDeviceCopy(SparkDeviceQueue *queue, SparkDeviceRange destination, SparkDeviceRange source, uint64_t bytes);
SparkStatus SparkDeviceFill(SparkDeviceQueue *queue, SparkDeviceRange destination, uint8_t value, uint64_t bytes);
SparkStatus SparkDeviceCopySync(SparkDeviceRange destination, SparkDeviceRange source, uint64_t bytes);

SparkStatus SparkDeviceQueueCreate(SparkDeviceQueue **queue, const char *owner);
SparkStatus SparkDeviceQueueDestroy(SparkDeviceQueue *queue);
SparkStatus SparkDeviceQueueQuery(SparkDeviceQueue *queue);
SparkStatus SparkDeviceQueueWait(SparkDeviceQueue *queue, uint64_t timeout_ns);
SparkStatus SparkDeviceQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context);

SparkStatus SparkDeviceEventCreate(SparkDeviceEvent **event);
SparkStatus SparkDeviceEventDestroy(SparkDeviceEvent *event);
SparkStatus SparkDeviceEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue);
SparkStatus SparkDeviceQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event);
SparkStatus SparkDeviceEventQuery(SparkDeviceEvent *event);
SparkStatus SparkDeviceEventWait(SparkDeviceEvent *event, uint64_t timeout_ns);
SparkStatus SparkDeviceEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds);

SparkStatus SparkDeviceLibraryLoad(SparkDeviceLibrary **library, const char *path, const char *owner);
SparkStatus SparkDeviceLibraryUnload(SparkDeviceLibrary *library);
SparkStatus SparkDeviceKernelCreate(SparkDeviceKernel **kernel, SparkDeviceLibrary *library, const char *name);
SparkStatus SparkDeviceKernelDestroy(SparkDeviceKernel *kernel);
SparkStatus SparkDeviceKernelGetInfo(const SparkDeviceKernel *kernel, SparkDeviceKernelInfo *info);
SparkStatus SparkDeviceKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch);

#ifdef __cplusplus
}
#endif

#endif
