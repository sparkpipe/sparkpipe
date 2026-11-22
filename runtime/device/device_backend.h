#ifndef SPARKPIPE_DEVICE_BACKEND_H
#define SPARKPIPE_DEVICE_BACKEND_H

#include <pthread.h>
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
	uint32_t programs;
	uint64_t segment_stride;
	void *backend;
	SparkDeviceStamp stamps[SPARK_DEVICE_STAMPS];
	char owner[SPARK_DEVICE_OWNER_BYTES];
} SparkDeviceAllocation;

struct SparkDeviceQueue
{
	uint64_t id;
	uint64_t submitted;
	uint32_t borrows;
	void *backend;
	SparkDeviceProgram *recording;
	SparkDeviceQueue *next;
	char owner[SPARK_DEVICE_OWNER_BYTES];
};

typedef enum SparkDeviceNodeKind
{
	SPARK_DEVICE_NODE_COPY = 1,
	SPARK_DEVICE_NODE_FILL = 2,
	SPARK_DEVICE_NODE_LAUNCH = 3,
	SPARK_DEVICE_NODE_HOST = 4,
	SPARK_DEVICE_NODE_RECORD = 5,
	SPARK_DEVICE_NODE_AWAIT = 6
} SparkDeviceNodeKind;

typedef struct SparkDeviceNode
{
	SparkDeviceNodeKind kind;
	SparkDeviceBuffer buffers[2];
	uint64_t offsets[2];
	uint64_t bytes;
	uint8_t value;
	const SparkDeviceKernel *kernel;
	SparkDeviceLaunch launch;
	SparkDeviceBuffer *binding_buffers;
	SparkDeviceHostFunction function;
	void *context;
	SparkDeviceEvent *event;
} SparkDeviceNode;

typedef struct SparkDevicePlace
{
	uint32_t node;
	uint32_t value;
	uint32_t target;
	uint32_t argument;
	uint32_t offset;
	uint32_t bytes;
	SparkDeviceTransform transform;
	void *backend;
} SparkDevicePlace;

struct SparkDeviceProgram
{
	uint64_t id;
	uint64_t queue_id;
	uint32_t native;
	uint32_t compiled;
	uint32_t finished;
	pthread_t thread;
	uint32_t broken;
	uint32_t kernels;
	uint32_t values;
	uint32_t native_nodes;
	SparkDeviceNode *nodes;
	uint32_t node_count;
	uint32_t node_capacity;
	uint32_t *records;
	uint32_t record_count;
	uint32_t record_capacity;
	SparkDevicePlace *places;
	uint32_t place_count;
	uint32_t place_capacity;
	uint32_t scratch_bytes;
	uint64_t digest;
	void *backend;
	SparkDeviceProgram *next;
	SparkDeviceStamp stamps[SPARK_DEVICE_STAMPS];
	char name[SPARK_DEVICE_OWNER_BYTES];
};

struct SparkDeviceEvent
{
	uint64_t id;
	uint64_t queue_id;
	uint64_t sequence;
	uint32_t recorded;
	uint32_t programs;
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
	uint32_t programs;
	SparkDeviceKernelInfo info;
	void *backend;
	SparkDeviceKernel *next;
	SparkDeviceStamp stamps[SPARK_DEVICE_STAMPS];
	char name[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u];
};

SparkStatus SparkDeviceRecordError(SparkStatus status, const char *operation, const char *owner, int32_t backend_code, const char *text);

SparkStatus SparkDeviceBackendInitialize(const SparkDeviceConfiguration *configuration, SparkDeviceProperties *properties);
SparkStatus SparkDeviceBackendShutdown(void);
SparkStatus SparkDeviceBackendThreadInstance(SparkDeviceInstance *instance);
SparkStatus SparkDeviceBackendThreadBind(SparkDeviceInstance instance);

SparkStatus SparkDeviceBackendAllocate(SparkDeviceAllocation *allocation);
SparkStatus SparkDeviceBackendRegister(SparkDeviceAllocation *allocation, void *host);
SparkStatus SparkDeviceBackendAdopt(SparkDeviceAllocation *allocation);
SparkStatus SparkDeviceBackendMapFile(SparkDeviceAllocation *allocation, int fd, uint64_t offset);
void SparkDeviceBackendRelease(SparkDeviceAllocation *allocation);

SparkStatus SparkDeviceBackendMemoryGranularity(uint32_t granularity, uint64_t *bytes);
SparkStatus SparkDeviceBackendMemoryCreate(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceBackendMemoryCreateHost(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceBackendMemoryRelease(SparkDeviceMemoryHandle memory);
SparkStatus SparkDeviceBackendMemoryExport(SparkDeviceMemoryHandle memory, int *fd);
SparkStatus SparkDeviceBackendMemoryImport(SparkDeviceMemoryHandle *memory, int fd, const char *owner);
SparkStatus SparkDeviceBackendAddressReserve(SparkDeviceAddress *address, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceBackendAddressRelease(SparkDeviceAddress address, uint64_t bytes);
SparkStatus SparkDeviceBackendMemoryMap(SparkDeviceAddress address, uint64_t bytes, SparkDeviceMemoryHandle memory, uint64_t offset, uint32_t access);
SparkStatus SparkDeviceBackendMemoryUnmap(SparkDeviceAddress address, uint64_t bytes);

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes);
SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes);

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue);
SparkStatus SparkDeviceBackendQueueAdopt(SparkDeviceQueue *queue, void *native);
void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue);
SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns);
uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue);
SparkStatus SparkDeviceBackendQueueCapture(SparkDeviceQueue *queue, uint32_t capturing);
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
SparkStatus SparkDeviceBackendKernelShortName(SparkDeviceLibrary *library, const char *name, char short_name[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u]);
void SparkDeviceBackendKernelDestroy(SparkDeviceKernel *kernel);
SparkStatus SparkDeviceBackendKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch,
	SparkDeviceAllocation *const *allocations);

uint32_t SparkDeviceBackendProgramNative(void);
SparkStatus SparkDeviceBackendProgramReplay(SparkDeviceQueue *queue);
SparkStatus SparkDeviceBackendProgramBegin(SparkDeviceQueue *queue, SparkDeviceProgram *program);
SparkStatus SparkDeviceBackendProgramPatch(SparkDeviceQueue *queue, SparkDeviceProgram *program, SparkDevicePlace *place);
SparkStatus SparkDeviceBackendProgramEnd(SparkDeviceQueue *queue, SparkDeviceProgram *program);
SparkStatus SparkDeviceBackendProgramCompile(SparkDeviceQueue *queue, SparkDeviceProgram *program);
SparkStatus SparkDeviceBackendProgramRun(SparkDeviceProgram *program, SparkDeviceQueue *queue, const uint64_t *values);
void SparkDeviceBackendProgramRelease(SparkDeviceProgram *program);

SparkStatus SparkDeviceProgramAppend(SparkDeviceProgram *program, const SparkDeviceNode *node, const char *operation);
SparkDeviceRange SparkDeviceNodeRange(const SparkDeviceNode *node, uint32_t which);
SparkStatus SparkDeviceProgramPlace(SparkDeviceProgram *program, const SparkDevicePatch *patch, SparkDevicePlace *place, const char *owner);
uint64_t SparkDeviceProgramDigest(const SparkDeviceProgram *program);
void SparkDeviceProgramFree(SparkDeviceProgram *program);
SparkStatus SparkDeviceProgramValues(const SparkDeviceProgram *program, const uint64_t *values, uint64_t *placed, const char *owner);
SparkStatus SparkDeviceProgramReplay(SparkDeviceProgram *program, SparkDeviceQueue *queue, const uint64_t *placed);
uint32_t SparkDeviceBackendTensorMapOnDevice(void);
SparkStatus SparkDeviceBackendTensorMapEncode(const SparkDeviceTensorMapRequest *requests, uint32_t count, SparkDeviceTensorMap *maps, const char *owner);
void SparkDeviceProgramPatchLaunch(const SparkDeviceProgram *program, uint32_t node, const uint64_t *placed, SparkDeviceLaunch *launch, uint8_t *scratch);

#ifdef __cplusplus
}
#endif

#endif
