#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "runtime/device/nvrm/nvrm_version.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"

#define NVRM_WARP_THREADS 32u
#define NVRM_GROUP_THREADS 1024u
#define NVRM_GROUP_SHARED_BYTES 101376u
#define NVRM_POLL_NS 20000L

typedef struct NvrmAllocation
{
	SparkNvrmMemory memory;
	void *heap;
	void *mapping;
	uint64_t mapping_bytes;
} NvrmAllocation;

typedef struct NvrmQueue
{
	SparkNvrmChannel channel;
	uint64_t base;
} NvrmQueue;

static SparkNvrmGpu nvrm_gpu;

#define NVRM_TRY(call) \
	do { \
		const SparkStatus nvrm_status = (call); \
		if ( nvrm_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(nvrm_status); \
	} while (0)

static SparkStatus NvrmMissing(const char *operation, const char *owner)
{
	char text[96];
	snprintf(text,sizeof(text),"the nvrm backend for module %s does not implement %s",SPARK_NVRM_MODULE_VERSION,operation);
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,operation,owner,0,text));
}

static void NvrmName(SparkDeviceProperties *properties)
{
	NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS name;
	memset(&name,0,sizeof(name));
	name.gpuNameStringFlags = NV2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE_ASCII;
	if ( SparkNvrmRmControl(&nvrm_gpu.rm,nvrm_gpu.subdevice,NV2080_CTRL_CMD_GPU_GET_NAME_STRING,&name,(uint32_t)sizeof(name),"initialize") == SPARK_STATUS_OK )
		snprintf(properties->name,sizeof(properties->name),"%s",(const char *)name.gpuNameString.ascii);
}

SparkStatus SparkDeviceBackendInitialize(const SparkDeviceConfiguration *configuration, SparkDeviceProperties *properties)
{
	const long pages = sysconf(_SC_PHYS_PAGES),page = sysconf(_SC_PAGESIZE);
	const uint64_t memory = pages > 0 && page > 0 ? (uint64_t)pages * (uint64_t)page : 0u;
	NVRM_TRY(SparkNvrmGpuOpen(&nvrm_gpu,configuration->ordinal,SPARK_NVRM_MEMORY_HOST,"initialize"));
	snprintf(properties->backend,sizeof(properties->backend),"%s","nvrm");
	snprintf(properties->target,sizeof(properties->target),"nvrm.arch%03x.impl%x.sm%03x",nvrm_gpu.architecture,nvrm_gpu.implementation,nvrm_gpu.sm_version);
	NvrmName(properties);
	properties->page_bytes = page > 0 ? (uint32_t)page : SPARK_NVRM_PAGE_BYTES;
	properties->compute_units = nvrm_gpu.gpcs * nvrm_gpu.tpcs_per_gpc * nvrm_gpu.sms_per_tpc;
	properties->simd_width = NVRM_WARP_THREADS;
	properties->memory_budget_bytes = configuration->memory_budget_bytes != 0u ? configuration->memory_budget_bytes : memory;
	properties->max_buffer_bytes = memory;
	properties->max_group_threads = NVRM_GROUP_THREADS;
	properties->max_shared_bytes = NVRM_GROUP_SHARED_BYTES;
	properties->properties = SPARK_DEVICE_PROPERTY_SHARED_ADDRESS | SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	SparkNvrmGpuClose(&nvrm_gpu);
	return(SPARK_STATUS_OK);
}

static NvrmAllocation *NvrmNew(SparkDeviceAllocation *allocation, const char *operation)
{
	NvrmAllocation *nvrm = (NvrmAllocation *)calloc(1u,sizeof(*nvrm));
	if ( nvrm == 0 )
	{
		SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,allocation->owner,ENOMEM,"out of host memory");
		return(0);
	}
	nvrm->memory.map_fd = -1;
	return(nvrm);
}

static SparkStatus NvrmZero(SparkNvrmMemory *memory, const char *owner)
{
	uint64_t value = 0u;
	NVRM_TRY(SparkNvrmFill(&nvrm_gpu,&nvrm_gpu.compute,memory->address,0u,memory->bytes,&value,owner));
	return(SparkNvrmWaitValue((const volatile uint64_t *)(void *)nvrm_gpu.compute.fence.cpu,value,SPARK_NVRM_WAIT_TIMEOUT_NS,"allocate",owner));
}

SparkStatus SparkDeviceBackendAllocate(SparkDeviceAllocation *allocation)
{
	NvrmAllocation *nvrm = NvrmNew(allocation,"allocate");
	const uint32_t private_space = allocation->space == SPARK_DEVICE_SPACE_DEVICE_PRIVATE;
	SparkStatus status;
	if ( nvrm == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( allocation->space == SPARK_DEVICE_SPACE_HOST_PAGEABLE )
	{
		nvrm->heap = calloc(1u,(size_t)allocation->bytes);
		if ( nvrm->heap == 0 )
		{
			free(nvrm);
			return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,ENOMEM,"out of host memory"));
		}
		allocation->host = nvrm->heap;
		allocation->device = 0u;
		allocation->backend = nvrm;
		return(SPARK_STATUS_OK);
	}
	status = SparkNvrmMemoryAllocate(&nvrm_gpu,allocation->bytes,private_space != 0u ? SPARK_NVRM_MEMORY_SYSTEM : SPARK_NVRM_MEMORY_HOST,&nvrm->memory,
		allocation->owner);
	if ( status == SPARK_STATUS_OK && private_space != 0u )
		status = NvrmZero(&nvrm->memory,allocation->owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmMemoryRelease(&nvrm_gpu,&nvrm->memory);
		free(nvrm);
		SPARK_FAIL(status);
	}
	allocation->host = private_space != 0u ? 0 : nvrm->memory.cpu;
	allocation->device = nvrm->memory.address;
	allocation->backend = nvrm;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendRegister(SparkDeviceAllocation *allocation, void *host)
{
	NvrmAllocation *nvrm = NvrmNew(allocation,"register");
	SparkStatus status;
	if ( nvrm == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkNvrmMemoryAttach(&nvrm_gpu,host,allocation->bytes,SPARK_NVRM_MEMORY_USER,&nvrm->memory,allocation->owner);
	if ( status != SPARK_STATUS_OK )
	{
		free(nvrm);
		SPARK_FAIL(status);
	}
	allocation->host = host;
	allocation->device = nvrm->memory.address + (uint64_t)((uint8_t *)host - nvrm->memory.cpu);
	allocation->backend = nvrm;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendAdopt(SparkDeviceAllocation *allocation)
{
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"adopt",allocation->owner,0,
		"the nvrm backend holds only memory it allocated, registered or mapped itself"));
}

SparkStatus SparkDeviceBackendMapFile(SparkDeviceAllocation *allocation, int fd, uint64_t offset)
{
	NvrmAllocation *nvrm = NvrmNew(allocation,"map_file");
	const uint64_t aligned = offset / SPARK_NVRM_PAGE_BYTES * SPARK_NVRM_PAGE_BYTES;
	SparkStatus status;
	if ( nvrm == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	nvrm->mapping_bytes = (offset - aligned + allocation->bytes + SPARK_NVRM_PAGE_BYTES - 1u) / SPARK_NVRM_PAGE_BYTES * SPARK_NVRM_PAGE_BYTES;
	nvrm->mapping = mmap(0,(size_t)nvrm->mapping_bytes,PROT_READ,MAP_SHARED,fd,(off_t)aligned);
	if ( nvrm->mapping == MAP_FAILED )
	{
		const int error = errno;
		free(nvrm);
		return(SparkDeviceRecordError(error == ENOMEM ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_INVALID_ARGUMENT,"map_file",allocation->owner,error,
			"cannot map the file range"));
	}
	status = SparkNvrmMemoryAttach(&nvrm_gpu,nvrm->mapping,nvrm->mapping_bytes,SPARK_NVRM_MEMORY_USER_READ_ONLY,&nvrm->memory,allocation->owner);
	if ( status != SPARK_STATUS_OK )
	{
		munmap(nvrm->mapping,(size_t)nvrm->mapping_bytes);
		free(nvrm);
		SPARK_FAIL(status);
	}
	allocation->host = (uint8_t *)nvrm->mapping + (offset - aligned);
	allocation->device = nvrm->memory.address + (offset - aligned);
	allocation->backend = nvrm;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendRelease(SparkDeviceAllocation *allocation)
{
	NvrmAllocation *nvrm = (NvrmAllocation *)allocation->backend;
	if ( nvrm == 0 )
		return;
	if ( nvrm->memory.bytes != 0u )
		SparkNvrmMemoryRelease(&nvrm_gpu,&nvrm->memory);
	if ( nvrm->mapping != 0 )
		munmap(nvrm->mapping,(size_t)nvrm->mapping_bytes);
	free(nvrm->heap);
	free(nvrm);
	allocation->backend = 0;
}

static SparkStatus NvrmSequence(SparkDeviceQueue *queue, uint64_t value, const char *operation)
{
	const NvrmQueue *nvrm = (const NvrmQueue *)queue->backend;
	char text[128];
	if ( value == nvrm->base + queue->submitted )
		return(SPARK_STATUS_OK);
	snprintf(text,sizeof(text),"the channel released %llu for queue sequence %llu from base %llu",(unsigned long long)value,
		(unsigned long long)queue->submitted,(unsigned long long)nvrm->base);
	return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation,queue->owner,0,text));
}

static SparkStatus NvrmPageable(SparkDeviceQueue *queue, const char *operation)
{
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,operation,queue->owner,0,
		"pageable host memory has no device address; the nvrm backend copies it once queues run host work"));
}

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	uint64_t value = 0u;
	(void)direction;
	if ( destination->device == 0u || source->device == 0u )
		return(NvrmPageable(queue,"copy"));
	NVRM_TRY(SparkNvrmCopy(&nvrm_gpu,&nvrm->channel,destination->device + destination_offset,source->device + source_offset,bytes,&value,queue->owner));
	return(NvrmSequence(queue,value,"copy"));
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	uint64_t released = 0u;
	if ( destination->device == 0u )
		return(NvrmPageable(queue,"fill"));
	NVRM_TRY(SparkNvrmFill(&nvrm_gpu,&nvrm->channel,destination->device + offset,value,bytes,&released,queue->owner));
	return(NvrmSequence(queue,released,"fill"));
}

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue)
{
	NvrmQueue *nvrm = (NvrmQueue *)calloc(1u,sizeof(*nvrm));
	SparkStatus status;
	if ( nvrm == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",queue->owner,ENOMEM,"out of host memory"));
	status = SparkNvrmChannelCreate(&nvrm_gpu,&nvrm->channel,SPARK_NVRM_MEMORY_HOST,queue->owner);
	if ( status != SPARK_STATUS_OK )
	{
		free(nvrm);
		SPARK_FAIL(status);
	}
	nvrm->base = nvrm->channel.fence_value;
	queue->backend = nvrm;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendQueueAdopt(SparkDeviceQueue *queue, void *native)
{
	(void)native;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"queue_adopt",queue->owner,0,"the nvrm backend owns every queue it runs"));
}

void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	if ( nvrm == 0 )
		return;
	SparkNvrmChannelDestroy(&nvrm_gpu,&nvrm->channel);
	free(nvrm);
	queue->backend = 0;
}

static uint64_t NvrmCompleted(const NvrmQueue *nvrm)
{
	const uint64_t reached = SparkNvrmChannelCompleted(&nvrm->channel);
	return(reached > nvrm->base ? reached - nvrm->base : 0u);
}

SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns)
{
	const NvrmQueue *nvrm = (const NvrmQueue *)queue->backend;
	const uint64_t start = SparkMonotonicNs();
	while ( NvrmCompleted(nvrm) < sequence )
	{
		struct timespec pause = {0, NVRM_POLL_NS};
		if ( timeout_ns != SPARK_DEVICE_WAIT_FOREVER && SparkMonotonicNs() - start >= timeout_ns )
			return(SPARK_STATUS_PENDING);
		nanosleep(&pause,0);
	}
	return(SPARK_STATUS_OK);
}

uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue)
{
	return(NvrmCompleted((const NvrmQueue *)queue->backend));
}

SparkStatus SparkDeviceBackendQueueCapture(SparkDeviceQueue *queue, uint32_t capturing)
{
	(void)capturing;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"queue_capture",queue->owner,0,"only the owner of an adopted queue records it into a graph"));
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
