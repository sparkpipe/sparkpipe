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
#include "runtime/device/nvrm/nvrm_host.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_register.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"

#define NVRM_WARP_THREADS 32u
#define NVRM_GROUP_THREADS 1024u
#define NVRM_GROUP_SHARED_BYTES 101376u
#define NVRM_PARAMETER_BYTES 32764u
#define NVRM_POLL_NS 20000L
#define NVRM_SEQUENCES 4096u
#define NVRM_SLOT_RING_BYTES (4ull << 20)
#define NVRM_SLOT_RECORDS 1024u
#define NVRM_DEFAULT_THREADS 1024u
#define NVRM_BOUNCE_BYTES (8ull << 20)
#define NVRM_EVENT_CHUNK_BYTES (64u << 10)
#define NVRM_EVENT_CHUNKS 64u
#define NVRM_EVENT_SLOTS_PER_CHUNK (NVRM_EVENT_CHUNK_BYTES / SPARK_NVRM_MARK_BYTES)

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
	SparkNvrmHostWorker worker;
	SparkNvrmMemory bounce;
	uint64_t ends[NVRM_SEQUENCES];
	uint64_t recorded;
	uint64_t completed;
	SparkNvrmMemory slots;
	uint64_t slot_head;
	SparkNvrmSegment slot_records[NVRM_SLOT_RECORDS];
	uint32_t slot_first;
	uint32_t slot_count;
} NvrmQueue;

typedef struct NvrmLibrary
{
	SparkNvrmProgram program;
} NvrmLibrary;

typedef struct NvrmKernel
{
	const NvrmLibrary *library;
	const SparkNvrmCubinKernel *kernel;
	SparkDeviceParameter *parameters;
} NvrmKernel;

typedef struct NvrmEvent
{
	uint32_t slot;
	uint64_t address;
	uint8_t *cpu;
	uint64_t generation;
} NvrmEvent;

typedef struct NvrmEventPool
{
	SparkNvrmMemory chunks[NVRM_EVENT_CHUNKS];
	uint32_t chunk_count;
	uint32_t free_slots[NVRM_EVENT_CHUNKS * NVRM_EVENT_SLOTS_PER_CHUNK];
	uint32_t free_count;
} NvrmEventPool;

static SparkNvrmGpu nvrm_gpu;
static NvrmEventPool nvrm_events;

#define NVRM_TRY(call) \
	do { \
		const SparkStatus nvrm_status = (call); \
		if ( nvrm_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(nvrm_status); \
	} while (0)

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
	properties->max_constant_bytes = NVRM_PARAMETER_BYTES;
	properties->properties = SPARK_DEVICE_PROPERTY_SHARED_ADDRESS | SPARK_DEVICE_PROPERTY_KERNEL_LIBRARIES | SPARK_DEVICE_PROPERTY_PARAMETER_BLOCKS |
		(nvrm_gpu.integrated != 0u ? SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY : 0u);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	uint32_t chunk;
	for (chunk = 0u; chunk < nvrm_events.chunk_count; chunk++)
		SparkNvrmMemoryRelease(&nvrm_gpu,&nvrm_events.chunks[chunk]);
	memset(&nvrm_events,0,sizeof(nvrm_events));
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

static uint64_t NvrmCompletedSequence(NvrmQueue *nvrm)
{
	const uint64_t reached = SparkNvrmChannelCompleted(&nvrm->channel);
	const uint64_t recorded = __atomic_load_n(&nvrm->recorded,__ATOMIC_ACQUIRE);
	uint64_t done = __atomic_load_n(&nvrm->completed,__ATOMIC_ACQUIRE),known = done;
	while ( done < recorded && __atomic_load_n(&nvrm->ends[(done + 1u) % NVRM_SEQUENCES],__ATOMIC_RELAXED) <= reached )
		done++;
	while ( known < done && !__atomic_compare_exchange_n(&nvrm->completed,&known,done,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE) )
	{
	}
	return(done);
}

static SparkStatus NvrmRoom(SparkDeviceQueue *queue, const char *operation)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	while ( queue->submitted - NvrmCompletedSequence(nvrm) >= NVRM_SEQUENCES )
	{
		const uint64_t oldest = __atomic_load_n(&nvrm->completed,__ATOMIC_ACQUIRE) + 1u;
		NVRM_TRY(SparkNvrmWaitValue((const volatile uint64_t *)(void *)nvrm->channel.fence.cpu,nvrm->ends[oldest % NVRM_SEQUENCES],SPARK_NVRM_WAIT_TIMEOUT_NS,
			operation,queue->owner));
	}
	return(SPARK_STATUS_OK);
}

static void NvrmEnd(SparkDeviceQueue *queue)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	__atomic_store_n(&nvrm->ends[queue->submitted % NVRM_SEQUENCES],nvrm->channel.fence_value,__ATOMIC_RELAXED);
	__atomic_store_n(&nvrm->recorded,queue->submitted,__ATOMIC_RELEASE);
}

static SparkStatus NvrmHost(SparkDeviceQueue *queue, SparkNvrmHostKind kind, uint8_t *destination, const uint8_t *source, uint64_t bytes, uint8_t byte,
	const char *operation)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	SparkNvrmHostItem item;
	memset(&item,0,sizeof(item));
	item.kind = kind;
	item.destination = destination;
	item.source = source;
	item.bytes = bytes;
	item.byte = byte;
	(void)operation;
	return(SparkNvrmHostWork(&nvrm_gpu,&nvrm->channel,&nvrm->worker,&item,0,queue->owner));
}

static SparkStatus NvrmBounce(NvrmQueue *nvrm, const char *owner)
{
	if ( nvrm->bounce.bytes != 0u )
		return(SPARK_STATUS_OK);
	return(SparkNvrmMemoryAllocate(&nvrm_gpu,NVRM_BOUNCE_BYTES,SPARK_NVRM_MEMORY_HOST,&nvrm->bounce,owner));
}

static SparkStatus NvrmStaged(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	const uint32_t upload = source->device == 0u;
	uint64_t done;
	NVRM_TRY(NvrmBounce(nvrm,queue->owner));
	for (done = 0u; done < bytes; done += NVRM_BOUNCE_BYTES)
	{
		const uint64_t chunk = bytes - done < NVRM_BOUNCE_BYTES ? bytes - done : NVRM_BOUNCE_BYTES;
		if ( upload != 0u )
		{
			NVRM_TRY(NvrmHost(queue,SPARK_NVRM_HOST_COPY,nvrm->bounce.cpu,(const uint8_t *)source->host + source_offset + done,chunk,0u,"copy"));
			NVRM_TRY(SparkNvrmCopy(&nvrm_gpu,&nvrm->channel,destination->device + destination_offset + done,nvrm->bounce.address,chunk,0,queue->owner));
		}
		else
		{
			NVRM_TRY(SparkNvrmCopy(&nvrm_gpu,&nvrm->channel,nvrm->bounce.address,source->device + source_offset + done,chunk,0,queue->owner));
			NVRM_TRY(NvrmHost(queue,SPARK_NVRM_HOST_COPY,(uint8_t *)destination->host + destination_offset + done,nvrm->bounce.cpu,chunk,0u,"copy"));
		}
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	SparkStatus status;
	(void)direction;
	NVRM_TRY(NvrmRoom(queue,"copy"));
	if ( destination->device != 0u && source->device != 0u )
		status = SparkNvrmCopy(&nvrm_gpu,&nvrm->channel,destination->device + destination_offset,source->device + source_offset,bytes,0,queue->owner);
	else if ( destination->host != 0 && source->host != 0 )
		status = NvrmHost(queue,SPARK_NVRM_HOST_COPY,(uint8_t *)destination->host + destination_offset,(const uint8_t *)source->host + source_offset,bytes,0u,"copy");
	else
		status = NvrmStaged(queue,destination,destination_offset,source,source_offset,bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	NvrmEnd(queue);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	SparkStatus status;
	NVRM_TRY(NvrmRoom(queue,"fill"));
	if ( destination->device != 0u )
		status = SparkNvrmFill(&nvrm_gpu,&nvrm->channel,destination->device + offset,value,bytes,0,queue->owner);
	else
		status = NvrmHost(queue,SPARK_NVRM_HOST_FILL,(uint8_t *)destination->host + offset,0,bytes,value,"fill");
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	NvrmEnd(queue);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue)
{
	NvrmQueue *nvrm = (NvrmQueue *)calloc(1u,sizeof(*nvrm));
	SparkStatus status;
	if ( nvrm == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",queue->owner,ENOMEM,"out of host memory"));
	nvrm->bounce.map_fd = -1;
	nvrm->slots.map_fd = -1;
	status = SparkNvrmChannelCreate(&nvrm_gpu,&nvrm->channel,SPARK_NVRM_MEMORY_HOST,queue->owner);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkNvrmHostOpen(&nvrm_gpu,&nvrm->worker,queue->owner);
		if ( status != SPARK_STATUS_OK )
			SparkNvrmChannelDestroy(&nvrm_gpu,&nvrm->channel);
	}
	if ( status != SPARK_STATUS_OK )
	{
		free(nvrm);
		SPARK_FAIL(status);
	}
	nvrm->ends[0] = nvrm->channel.fence_value;
	queue->backend = nvrm;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendQueueAdopt(SparkDeviceQueue *queue, void *native)
{
	(void)native;
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"queue_adopt",queue->owner,0,
			"the nvrm backend adopts only its own live queues, and this handle is not one");
		SPARK_FAIL(status);
	}
}

void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	if ( nvrm == 0 )
		return;
	SparkNvrmChannelDestroy(&nvrm_gpu,&nvrm->channel);
	SparkNvrmHostClose(&nvrm_gpu,&nvrm->worker);
	if ( nvrm->bounce.bytes != 0u )
		SparkNvrmMemoryRelease(&nvrm_gpu,&nvrm->bounce);
	if ( nvrm->slots.bytes != 0u )
		SparkNvrmMemoryRelease(&nvrm_gpu,&nvrm->slots);
	free(nvrm);
	queue->backend = 0;
}

SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	const uint64_t start = SparkMonotonicNs();
	while ( NvrmCompletedSequence(nvrm) < sequence )
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
	return(NvrmCompletedSequence((NvrmQueue *)queue->backend));
}

SparkStatus SparkDeviceBackendQueueCapture(SparkDeviceQueue *queue, uint32_t capturing)
{
	(void)capturing;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"queue_capture",queue->owner,0,"only the owner of an adopted queue records it into a graph"));
}

SparkStatus SparkDeviceBackendQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	SparkNvrmHostItem item;
	NVRM_TRY(NvrmRoom(queue,"host_callback"));
	memset(&item,0,sizeof(item));
	item.kind = SPARK_NVRM_HOST_CALL;
	item.function = function;
	item.context = context;
	NVRM_TRY(SparkNvrmHostWork(&nvrm_gpu,&nvrm->channel,&nvrm->worker,&item,0,queue->owner));
	NvrmEnd(queue);
	return(SPARK_STATUS_OK);
}

static SparkStatus NvrmEventGrow(void)
{
	SparkNvrmMemory *chunk;
	uint32_t slot;
	if ( nvrm_events.chunk_count == NVRM_EVENT_CHUNKS )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"event_create","",0,"every event slot the backend keeps is in use"));
	chunk = &nvrm_events.chunks[nvrm_events.chunk_count];
	NVRM_TRY(SparkNvrmMemoryAllocate(&nvrm_gpu,NVRM_EVENT_CHUNK_BYTES,SPARK_NVRM_MEMORY_HOST,chunk,"event"));
	for (slot = NVRM_EVENT_SLOTS_PER_CHUNK; slot > 0u; slot--)
		nvrm_events.free_slots[nvrm_events.free_count++] = nvrm_events.chunk_count * NVRM_EVENT_SLOTS_PER_CHUNK + slot - 1u;
	nvrm_events.chunk_count++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendEventCreate(SparkDeviceEvent *event)
{
	NvrmEvent *nvrm = (NvrmEvent *)calloc(1u,sizeof(*nvrm));
	const SparkNvrmMemory *chunk;
	SparkStatus status = SPARK_STATUS_OK;
	if ( nvrm == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"event_create","",ENOMEM,"out of host memory"));
	if ( nvrm_events.free_count == 0u )
		status = NvrmEventGrow();
	if ( status != SPARK_STATUS_OK )
	{
		free(nvrm);
		SPARK_FAIL(status);
	}
	nvrm->slot = nvrm_events.free_slots[--nvrm_events.free_count];
	chunk = &nvrm_events.chunks[nvrm->slot / NVRM_EVENT_SLOTS_PER_CHUNK];
	nvrm->address = chunk->address + (uint64_t)(nvrm->slot % NVRM_EVENT_SLOTS_PER_CHUNK) * SPARK_NVRM_MARK_BYTES;
	nvrm->cpu = chunk->cpu + (uint64_t)(nvrm->slot % NVRM_EVENT_SLOTS_PER_CHUNK) * SPARK_NVRM_MARK_BYTES;
	memset(nvrm->cpu,0,SPARK_NVRM_MARK_BYTES);
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	event->backend = nvrm;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendEventDestroy(SparkDeviceEvent *event)
{
	NvrmEvent *nvrm = (NvrmEvent *)event->backend;
	if ( nvrm != 0 )
	{
		nvrm_events.free_slots[nvrm_events.free_count++] = nvrm->slot;
		free(nvrm);
	}
	event->backend = 0;
}

SparkStatus SparkDeviceBackendEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue)
{
	NvrmEvent *nvrm = (NvrmEvent *)event->backend;
	NvrmQueue *owner = (NvrmQueue *)queue->backend;
	NVRM_TRY(NvrmRoom(queue,"event_record"));
	NVRM_TRY(SparkNvrmMark(&nvrm_gpu,&owner->channel,nvrm->address,nvrm->generation + 1u,0,queue->owner));
	nvrm->generation++;
	NvrmEnd(queue);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	const NvrmEvent *nvrm = (const NvrmEvent *)event->backend;
	NvrmQueue *waiter = (NvrmQueue *)queue->backend;
	NVRM_TRY(NvrmRoom(queue,"queue_wait_event"));
	NVRM_TRY(SparkNvrmAwait(&nvrm_gpu,&waiter->channel,nvrm->address,nvrm->generation,0,queue->owner));
	NvrmEnd(queue);
	return(SPARK_STATUS_OK);
}

static uint64_t NvrmEventReached(const NvrmEvent *nvrm)
{
	return(__atomic_load_n((const volatile uint64_t *)(const void *)nvrm->cpu,__ATOMIC_ACQUIRE));
}

SparkStatus SparkDeviceBackendEventWait(SparkDeviceEvent *event, uint64_t timeout_ns)
{
	const NvrmEvent *nvrm = (const NvrmEvent *)event->backend;
	const uint64_t generation = nvrm->generation,start = SparkMonotonicNs();
	while ( NvrmEventReached(nvrm) < generation )
	{
		struct timespec pause = {0, NVRM_POLL_NS};
		if ( timeout_ns != SPARK_DEVICE_WAIT_FOREVER && SparkMonotonicNs() - start >= timeout_ns )
			return(SPARK_STATUS_PENDING);
		nanosleep(&pause,0);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds)
{
	const NvrmEvent *first = (const NvrmEvent *)start->backend,*last = (const NvrmEvent *)stop->backend;
	const uint64_t begun = __atomic_load_n((const volatile uint64_t *)(const void *)(first->cpu + SPARK_NVRM_MARK_TIME_OFFSET + sizeof(uint64_t)),__ATOMIC_ACQUIRE);
	const uint64_t ended = __atomic_load_n((const volatile uint64_t *)(const void *)(last->cpu + SPARK_NVRM_MARK_TIME_OFFSET + sizeof(uint64_t)),__ATOMIC_ACQUIRE);
	if ( ended < begun )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"event_elapsed","",0,"the stop event's GPU time is before the start event's"));
	*nanoseconds = ended - begun;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendLibraryLoad(SparkDeviceLibrary *library, const char *path)
{
	NvrmLibrary *nvrm = (NvrmLibrary *)calloc(1u,sizeof(*nvrm));
	SparkStatus status;
	if ( nvrm == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"library_load",library->owner,ENOMEM,"out of host memory"));
	status = SparkNvrmProgramLoadFile(&nvrm_gpu,path,&nvrm->program,library->owner);
	if ( status != SPARK_STATUS_OK )
	{
		free(nvrm);
		SPARK_FAIL(status);
	}
	library->backend = nvrm;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendLibraryUnload(SparkDeviceLibrary *library)
{
	NvrmLibrary *nvrm = (NvrmLibrary *)library->backend;
	if ( nvrm != 0 )
	{
		SparkNvrmProgramRelease(&nvrm_gpu,&nvrm->program);
		free(nvrm);
	}
	library->backend = 0;
}

static SparkStatus NvrmKernelLayout(NvrmKernel *nvrm, SparkDeviceKernel *kernel, const SparkNvrmManifestKernel *reported)
{
	uint32_t index;
	nvrm->parameters = (SparkDeviceParameter *)calloc(reported->parameter_count != 0u ? reported->parameter_count : 1u,sizeof(*nvrm->parameters));
	if ( nvrm->parameters == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_create",kernel->name,ENOMEM,"out of host memory for the parameter table"));
	for (index = 0u; index < reported->parameter_count; index++)
	{
		nvrm->parameters[index].offset = reported->parameters[index].offset;
		nvrm->parameters[index].bytes = reported->parameters[index].bytes;
		nvrm->parameters[index].align = reported->parameters[index].align;
	}
	kernel->info.parameter_count = reported->parameter_count;
	kernel->info.parameter_bytes = nvrm->kernel->parameter_bytes;
	kernel->info.parameters = nvrm->parameters;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendKernelCreate(SparkDeviceKernel *kernel, const char *name)
{
	const NvrmLibrary *library = (const NvrmLibrary *)kernel->library->backend;
	const SparkNvrmCubinKernel *found = SparkNvrmCubinKernelNamed(&library->program.cubin,name);
	const SparkNvrmManifestKernel *reported = SparkNvrmManifestKernelNamed(&library->program.manifest,name);
	NvrmKernel *nvrm;
	SparkStatus status;
	if ( found == 0 || reported == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"kernel_create",name,0,"the kernel library has no kernel by that name"));
	if ( SparkNvrmKernelRunnable(found,name) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	nvrm = (NvrmKernel *)calloc(1u,sizeof(*nvrm));
	if ( nvrm == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_create",name,ENOMEM,"out of host memory"));
	nvrm->library = library;
	nvrm->kernel = found;
	status = NvrmKernelLayout(nvrm,kernel,reported);
	if ( status != SPARK_STATUS_OK )
	{
		free(nvrm);
		SPARK_FAIL(status);
	}
	kernel->info.max_threads = reported->max_threads != 0u ? reported->max_threads : NVRM_DEFAULT_THREADS;
	kernel->info.simd_width = NVRM_WARP_THREADS;
	kernel->info.static_shared_bytes = reported->shared_bytes;
	kernel->backend = nvrm;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmKernelResources(const SparkDeviceKernel *kernel, uint32_t *registers, uint32_t *stack_bytes)
{
	const NvrmKernel *nvrm = kernel != 0 ? (const NvrmKernel *)kernel->backend : 0;
	if ( nvrm == 0 || registers == 0 || stack_bytes == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_info",0,0,"needs a live nvrm kernel and both outputs"));
	*registers = nvrm->kernel->registers;
	*stack_bytes = nvrm->kernel->stack_bytes;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendKernelDestroy(SparkDeviceKernel *kernel)
{
	NvrmKernel *nvrm = (NvrmKernel *)kernel->backend;
	if ( nvrm != 0 )
	{
		free(nvrm->parameters);
		free(nvrm);
	}
	kernel->info.parameters = 0;
	kernel->backend = 0;
}

static uint32_t NvrmSlotFree(const NvrmQueue *nvrm, uint64_t start, uint64_t end)
{
	uint64_t tail;
	if ( nvrm->slot_count == 0u )
		return(1u);
	if ( nvrm->slot_count == NVRM_SLOT_RECORDS )
		return(0u);
	tail = nvrm->slot_records[nvrm->slot_first].start;
	if ( tail < nvrm->slot_head )
		return(start >= nvrm->slot_head || end <= tail ? 1u : 0u);
	if ( tail > nvrm->slot_head )
		return(start >= nvrm->slot_head && end <= tail ? 1u : 0u);
	return(0u);
}

static SparkStatus NvrmSlot(NvrmQueue *nvrm, uint32_t bytes, const char *owner, uint64_t *start)
{
	const uint64_t completed = SparkNvrmChannelCompleted(&nvrm->channel);
	if ( nvrm->slots.bytes == 0u )
		NVRM_TRY(SparkNvrmMemoryAllocate(&nvrm_gpu,NVRM_SLOT_RING_BYTES,SPARK_NVRM_MEMORY_HOST,&nvrm->slots,owner));
	while ( nvrm->slot_count != 0u && nvrm->slot_records[nvrm->slot_first].value <= completed )
	{
		nvrm->slot_first = (nvrm->slot_first + 1u) % NVRM_SLOT_RECORDS;
		nvrm->slot_count--;
	}
	*start = nvrm->slot_head + bytes <= nvrm->slots.bytes ? nvrm->slot_head : 0u;
	while ( NvrmSlotFree(nvrm,*start,*start + bytes) == 0u )
	{
		NVRM_TRY(SparkNvrmWaitValue((const volatile uint64_t *)(void *)nvrm->channel.fence.cpu,nvrm->slot_records[nvrm->slot_first].value,
			SPARK_NVRM_WAIT_TIMEOUT_NS,"kernel_launch",owner));
		nvrm->slot_first = (nvrm->slot_first + 1u) % NVRM_SLOT_RECORDS;
		nvrm->slot_count--;
	}
	return(SPARK_STATUS_OK);
}

static void NvrmSlotTaken(NvrmQueue *nvrm, uint64_t start, uint32_t bytes, uint64_t value)
{
	SparkNvrmSegment *record = &nvrm->slot_records[(nvrm->slot_first + nvrm->slot_count) % NVRM_SLOT_RECORDS];
	record->start = start;
	record->end = start + bytes;
	record->value = value;
	nvrm->slot_count++;
	nvrm->slot_head = start + bytes;
}

SparkStatus SparkDeviceBackendKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch,
	SparkDeviceAllocation *const *allocations)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	const NvrmKernel *compiled = (const NvrmKernel *)kernel->backend;
	SparkNvrmLaunchRequest request;
	const uint32_t bytes = SparkNvrmLaunchSlotBytes(compiled->kernel);
	uint64_t value = 0u,start = 0u;
	(void)allocations;
	if ( launch->binding_count != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"kernel_launch",kernel->name,0,
			"nvrm kernels take every argument in their parameter block; pass it as the launch's constants"));
	NVRM_TRY(NvrmRoom(queue,"kernel_launch"));
	memset(&request,0,sizeof(request));
	NVRM_TRY(NvrmSlot(nvrm,bytes,queue->owner,&start));
	request.slot = nvrm->slots.cpu + start;
	request.slot_address = nvrm->slots.address + start;
	request.slot_bytes = bytes;
	request.kernel = compiled->kernel;
	request.grid[0] = launch->groups.x;
	request.grid[1] = launch->groups.y;
	request.grid[2] = launch->groups.z;
	request.block[0] = launch->threads.x;
	request.block[1] = launch->threads.y;
	request.block[2] = launch->threads.z;
	request.dynamic_shared_bytes = launch->shared_bytes;
	request.arguments = launch->constants;
	request.argument_bytes = launch->constant_bytes;
	NVRM_TRY(SparkNvrmLaunch(&nvrm_gpu,&nvrm->channel,&compiled->library->program,&request,&value,queue->owner));
	NvrmSlotTaken(nvrm,start,bytes,value);
	NvrmEnd(queue);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendThreadInstance(SparkDeviceInstance *instance)
{
	(void)instance;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"thread_instance","",0,"the nvrm backend does not bind threads to an instance yet"));
}

SparkStatus SparkDeviceBackendThreadBind(SparkDeviceInstance instance)
{
	(void)instance;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"thread_bind","",0,"the nvrm backend does not bind threads to an instance yet"));
}

SparkStatus SparkDeviceBackendMemoryGranularity(uint32_t granularity, uint64_t *bytes)
{
	(void)granularity;
	(void)bytes;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"memory_granularity","",0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendMemoryCreate(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner)
{
	(void)memory;
	(void)bytes;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"memory_create",owner,0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendMemoryRelease(SparkDeviceMemoryHandle memory)
{
	(void)memory;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"memory_release","",0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendMemoryExport(SparkDeviceMemoryHandle memory, int *fd)
{
	(void)memory;
	(void)fd;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"memory_export","",0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendMemoryImport(SparkDeviceMemoryHandle *memory, int fd, const char *owner)
{
	(void)memory;
	(void)fd;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"memory_import",owner,0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendAddressReserve(SparkDeviceAddress *address, uint64_t bytes, const char *owner)
{
	(void)address;
	(void)bytes;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"address_reserve",owner,0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendAddressRelease(SparkDeviceAddress address, uint64_t bytes)
{
	(void)address;
	(void)bytes;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"address_release","",0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendMemoryMap(SparkDeviceAddress address, uint64_t bytes, SparkDeviceMemoryHandle memory, uint64_t offset, uint32_t access)
{
	(void)address;
	(void)bytes;
	(void)memory;
	(void)offset;
	(void)access;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"memory_map","",0,"the nvrm backend has no shared device memory yet"));
}

SparkStatus SparkDeviceBackendMemoryUnmap(SparkDeviceAddress address, uint64_t bytes)
{
	(void)address;
	(void)bytes;
	return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"memory_unmap","",0,"the nvrm backend has no shared device memory yet"));
}
