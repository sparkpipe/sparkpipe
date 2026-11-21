#include <stdio.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_version.h"

static SparkStatus NvrmMissing(const char *operation, const char *owner)
{
	char text[96];
	snprintf(text,sizeof(text),"the nvrm backend for module %s does not implement %s",SPARK_NVRM_MODULE_VERSION,operation);
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,operation,owner,0,text));
}

SparkStatus SparkDeviceBackendInitialize(const SparkDeviceConfiguration *configuration, SparkDeviceProperties *properties)
{
	const SparkStatus status = NvrmMissing("initialize","");
	(void)configuration;
	(void)properties;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	const SparkStatus status = NvrmMissing("shutdown","");
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendAllocate(SparkDeviceAllocation *allocation)
{
	const SparkStatus status = NvrmMissing("allocate",allocation->owner);
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendRegister(SparkDeviceAllocation *allocation, void *host)
{
	const SparkStatus status = NvrmMissing("register",allocation->owner);
	(void)host;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendAdopt(SparkDeviceAllocation *allocation)
{
	const SparkStatus status = NvrmMissing("adopt",allocation->owner);
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendMapFile(SparkDeviceAllocation *allocation, int fd, uint64_t offset)
{
	const SparkStatus status = NvrmMissing("map_file",allocation->owner);
	(void)fd;
	(void)offset;
	SPARK_FAIL(status);
}

void SparkDeviceBackendRelease(SparkDeviceAllocation *allocation)
{
	allocation->backend = 0;
}

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	const SparkStatus status = NvrmMissing("copy",queue->owner);
	(void)direction;
	(void)destination;
	(void)destination_offset;
	(void)source;
	(void)source_offset;
	(void)bytes;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	const SparkStatus status = NvrmMissing("fill",queue->owner);
	(void)destination;
	(void)offset;
	(void)value;
	(void)bytes;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue)
{
	const SparkStatus status = NvrmMissing("queue_create",queue->owner);
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendQueueAdopt(SparkDeviceQueue *queue, void *native)
{
	const SparkStatus status = NvrmMissing("queue_adopt",queue->owner);
	(void)native;
	SPARK_FAIL(status);
}

void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue)
{
	queue->backend = 0;
}

SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns)
{
	const SparkStatus status = NvrmMissing("queue_wait",queue->owner);
	(void)sequence;
	(void)timeout_ns;
	SPARK_FAIL(status);
}

uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue)
{
	(void)NvrmMissing("queue_completed",queue->owner);
	return(0u);
}

SparkStatus SparkDeviceBackendQueueCapture(SparkDeviceQueue *queue, uint32_t capturing)
{
	const SparkStatus status = NvrmMissing("queue_capture",queue->owner);
	(void)capturing;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context)
{
	const SparkStatus status = NvrmMissing("host_callback",queue->owner);
	(void)function;
	(void)context;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendEventCreate(SparkDeviceEvent *event)
{
	const SparkStatus status = NvrmMissing("event_create","");
	(void)event;
	SPARK_FAIL(status);
}

void SparkDeviceBackendEventDestroy(SparkDeviceEvent *event)
{
	event->backend = 0;
}

SparkStatus SparkDeviceBackendEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue)
{
	const SparkStatus status = NvrmMissing("event_record",queue->owner);
	(void)event;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	const SparkStatus status = NvrmMissing("queue_wait_event",queue->owner);
	(void)event;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendEventWait(SparkDeviceEvent *event, uint64_t timeout_ns)
{
	const SparkStatus status = NvrmMissing("event_wait","");
	(void)event;
	(void)timeout_ns;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds)
{
	const SparkStatus status = NvrmMissing("event_elapsed","");
	(void)start;
	(void)stop;
	(void)nanoseconds;
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendLibraryLoad(SparkDeviceLibrary *library, const char *path)
{
	const SparkStatus status = NvrmMissing("library_load",library->owner);
	(void)path;
	SPARK_FAIL(status);
}

void SparkDeviceBackendLibraryUnload(SparkDeviceLibrary *library)
{
	library->backend = 0;
}

SparkStatus SparkDeviceBackendKernelCreate(SparkDeviceKernel *kernel, const char *name)
{
	const SparkStatus status = NvrmMissing("kernel_create",name);
	(void)kernel;
	SPARK_FAIL(status);
}

void SparkDeviceBackendKernelDestroy(SparkDeviceKernel *kernel)
{
	kernel->backend = 0;
}

SparkStatus SparkDeviceBackendKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch,
	SparkDeviceAllocation *const *allocations)
{
	const SparkStatus status = NvrmMissing("kernel_launch",kernel->name);
	(void)queue;
	(void)launch;
	(void)allocations;
	SPARK_FAIL(status);
}
