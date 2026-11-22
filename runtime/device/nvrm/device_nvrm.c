#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "runtime/device/nvrm/nvrm_host.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_register.h"
#include "runtime/device/nvrm/nvrm_resources.h"
#include "runtime/device/nvrm/nvrm_occupancy.h"
#include "runtime/device/nvrm/nvrm_wait.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"

#define NVRM_WARP_THREADS 32u
#define NVRM_GROUP_THREADS 1024u
#define NVRM_GROUP_SHARED_BYTES 101376u
#define NVRM_PARAMETER_BYTES 32764u
#define NVRM_SEQUENCES 4096u
#define NVRM_SLOT_RING_BYTES (4ull << 20)
#define NVRM_SLOT_RECORDS 1024u
#define NVRM_DEFAULT_THREADS 1024u
#define NVRM_BOUNCE_BYTES (8ull << 20)
#define NVRM_EVENT_CHUNK_BYTES (64u << 10)
#define NVRM_EVENT_CHUNKS 64u
#define NVRM_EVENT_SLOTS_PER_CHUNK (NVRM_EVENT_CHUNK_BYTES / SPARK_NVRM_MARK_BYTES)
#define NVRM_END_COPY (UINT64_C(1) << 63)
#define NVRM_REPORT_HEAD_BYTES 112u
#define NVRM_REPORT_WHERE_BYTES 160u

typedef enum NvrmOperation
{
	NVRM_OPERATION_NONE = 0,
	NVRM_OPERATION_COPY = 1,
	NVRM_OPERATION_FILL = 2,
	NVRM_OPERATION_HOST = 3,
	NVRM_OPERATION_RECORD = 4,
	NVRM_OPERATION_AWAIT = 5,
	NVRM_OPERATION_LAUNCH = 6
} NvrmOperation;

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
	SparkNvrmChannel copy;
	SparkNvrmChannel *last;
	SparkNvrmHostWorker worker;
	SparkNvrmMemory bounce;
	uint64_t ends[NVRM_SEQUENCES];
	const SparkDeviceKernel *kernels[NVRM_SEQUENCES];
	uint8_t operations[NVRM_SEQUENCES];
	uint64_t recorded;
	uint64_t completed;
	SparkNvrmMemory slots;
	uint64_t slot_head;
	SparkNvrmSegment slot_records[NVRM_SLOT_RECORDS];
	uint32_t slot_first;
	uint32_t slot_count;
	uint64_t kernel_sequence;
	char kernel[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u];
	SparkDeviceQueue *queue;
	struct NvrmQueue *next;
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
	uint64_t queue_id;
} NvrmEvent;

typedef struct NvrmEventPool
{
	SparkNvrmMemory chunks[NVRM_EVENT_CHUNKS];
	uint32_t chunk_count;
	uint32_t free_slots[NVRM_EVENT_CHUNKS * NVRM_EVENT_SLOTS_PER_CHUNK];
	uint32_t free_count;
} NvrmEventPool;

typedef struct NvrmShared
{
	SparkDeviceMemoryHandle id;
	uint64_t bytes;
	uint64_t maps;
	uint32_t handle;
	uint32_t held;
} NvrmShared;

typedef struct NvrmSpan
{
	uint64_t address;
	uint64_t bytes;
	SparkDeviceMemoryHandle memory;
} NvrmSpan;

typedef struct NvrmSpans
{
	NvrmSpan *items;
	uint32_t count;
	uint32_t capacity;
} NvrmSpans;

static SparkNvrmGpu nvrm_gpu;
static NvrmEventPool nvrm_events;
static pthread_mutex_t nvrm_shared_lock = PTHREAD_MUTEX_INITIALIZER;
static NvrmShared *nvrm_shared;
static uint32_t nvrm_shared_count;
static uint32_t nvrm_shared_capacity;
static SparkDeviceMemoryHandle nvrm_shared_serial;
static NvrmSpans nvrm_reservations;
static NvrmSpans nvrm_mappings;
static pthread_mutex_t nvrm_queues_lock = PTHREAD_MUTEX_INITIALIZER;
static NvrmQueue *nvrm_queues;

static uint32_t NvrmMappingsSpan(uint64_t address, uint64_t bytes);
static void NvrmSharedClose(void);

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
	uint32_t copy_engine = 0u;
	NVRM_TRY(SparkNvrmGpuOpen(&nvrm_gpu,configuration->ordinal,SPARK_NVRM_MEMORY_HOST,"initialize"));
	if ( SparkNvrmModuleZeroesMemory(SPARK_NVRM_MODULE_PARAMETERS) == 0 )
	{
		SparkNvrmGpuClose(&nvrm_gpu);
		return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"initialize","",0,
			"the open kernel module does not zero the system memory it allocates (InitializeSystemMemoryAllocations is not 1 in "
			SPARK_NVRM_MODULE_PARAMETERS "); load it with NVreg_InitializeSystemMemoryAllocations=1"));
	}
	{
		SparkStatus copies = SparkNvrmCopyEngineAsync(&nvrm_gpu,&copy_engine,"initialize");
		if ( copies == SPARK_STATUS_OK )
			copies = SparkNvrmCopyGroupOpen(&nvrm_gpu,copy_engine,"initialize");
		if ( copies != SPARK_STATUS_OK )
		{
			SparkNvrmCopyGroupClose(&nvrm_gpu);
			SparkNvrmGpuClose(&nvrm_gpu);
			SPARK_FAIL(copies);
		}
	}
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
		SPARK_DEVICE_PROPERTY_CLUSTERS | SPARK_DEVICE_PROPERTY_NO_PAGE_FALLBACK |
		(nvrm_gpu.integrated != 0u ? SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY : 0u);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	uint32_t chunk;
	for (chunk = 0u; chunk < nvrm_events.chunk_count; chunk++)
		SparkNvrmMemoryRelease(&nvrm_gpu,&nvrm_events.chunks[chunk]);
	memset(&nvrm_events,0,sizeof(nvrm_events));
	SparkNvrmCopyGroupClose(&nvrm_gpu);
	SparkNvrmGpuClose(&nvrm_gpu);
	NvrmSharedClose();
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
	NvrmAllocation *nvrm;
	if ( allocation->host != 0 || allocation->space != SPARK_DEVICE_SPACE_DEVICE_PRIVATE || NvrmMappingsSpan(allocation->device,allocation->bytes) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"adopt",allocation->owner,0,
			"the nvrm backend adopts only device-private ranges of shared memory it mapped"));
	nvrm = NvrmNew(allocation,"adopt");
	if ( nvrm == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	allocation->backend = nvrm;
	return(SPARK_STATUS_OK);
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

static SparkNvrmChannel *NvrmOn(NvrmQueue *nvrm, SparkNvrmChannel *target)
{
	if ( nvrm->last != 0 && nvrm->last != target )
		SparkNvrmChannelAfter(target,nvrm->last);
	nvrm->last = target;
	return(target);
}

static SparkNvrmChannel *NvrmEndChannel(NvrmQueue *nvrm, uint64_t end)
{
	return((end & NVRM_END_COPY) != 0u ? &nvrm->copy : &nvrm->channel);
}

static uint64_t NvrmCompletedSequence(NvrmQueue *nvrm)
{
	const uint64_t reached = SparkNvrmChannelCompleted(&nvrm->channel),copied = SparkNvrmChannelCompleted(&nvrm->copy);
	const uint64_t recorded = __atomic_load_n(&nvrm->recorded,__ATOMIC_ACQUIRE);
	uint64_t done = __atomic_load_n(&nvrm->completed,__ATOMIC_ACQUIRE),known = done;
	while ( done < recorded )
	{
		const uint64_t end = __atomic_load_n(&nvrm->ends[(done + 1u) % NVRM_SEQUENCES],__ATOMIC_RELAXED);
		if ( (end & ~NVRM_END_COPY) > ((end & NVRM_END_COPY) != 0u ? copied : reached) )
			break;
		done++;
	}
	while ( known < done && !__atomic_compare_exchange_n(&nvrm->completed,&known,done,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE) )
	{
	}
	return(done);
}

static const char *NvrmOperationName(uint8_t operation)
{
	switch ( operation )
	{
	case NVRM_OPERATION_COPY:
		return("a copy");
	case NVRM_OPERATION_FILL:
		return("a fill");
	case NVRM_OPERATION_HOST:
		return("host work");
	case NVRM_OPERATION_RECORD:
		return("an event record");
	case NVRM_OPERATION_AWAIT:
		return("an event wait");
	default:
		return("an operation");
	}
}

static uint64_t NvrmProgressLocked(void)
{
	uint64_t progress = SparkNvrmChannelCompleted(&nvrm_gpu.compute) + SparkNvrmChannelCompleted(&nvrm_gpu.copy);
	const NvrmQueue *queue;
	for (queue = nvrm_queues; queue != 0; queue = queue->next)
		progress += SparkNvrmChannelCompleted(&queue->channel) + SparkNvrmChannelCompleted(&queue->copy);
	return(progress);
}

static const SparkNvrmChannel *NvrmFaulted(const NvrmQueue *nvrm, SparkNvrmFault *fault)
{
	const SparkNvrmChannel *channels[4];
	uint32_t count = 0u,index;
	if ( nvrm != 0 )
	{
		channels[count++] = &nvrm->channel;
		channels[count++] = &nvrm->copy;
	}
	channels[count++] = &nvrm_gpu.compute;
	channels[count++] = &nvrm_gpu.copy;
	for (index = 0u; index < count; index++)
		if ( SparkNvrmChannelFault(channels[index],fault) != 0u )
			return(channels[index]);
	return(0);
}

static void NvrmLocateLocked(const NvrmQueue *nvrm, const SparkNvrmChannel *faulted, SparkNvrmFault *fault)
{
	const NvrmQueue *other;
	if ( nvrm != 0 )
	{
		SparkNvrmFaultLocate(&nvrm_gpu,&nvrm->channel,fault);
		SparkNvrmFaultLocate(&nvrm_gpu,&nvrm->copy,fault);
	}
	SparkNvrmFaultLocate(&nvrm_gpu,faulted,fault);
	for (other = nvrm_queues; other != 0; other = other->next)
	{
		SparkNvrmFaultLocate(&nvrm_gpu,&other->channel,fault);
		SparkNvrmFaultLocate(&nvrm_gpu,&other->copy,fault);
	}
}

static void NvrmPosition(NvrmQueue *nvrm, char *text, size_t capacity)
{
	const uint64_t done = NvrmCompletedSequence(nvrm),recorded = __atomic_load_n(&nvrm->recorded,__ATOMIC_ACQUIRE);
	const uint32_t at = (uint32_t)((done + 1u) % NVRM_SEQUENCES);
	const char *last = nvrm->kernel[0] != '\0' ? nvrm->kernel : "none";
	if ( done >= recorded )
		snprintf(text,capacity,"; nothing pending; last kernel launched %s",last);
	else if ( nvrm->operations[at] != NVRM_OPERATION_LAUNCH || nvrm->kernels[at] == 0 )
		snprintf(text,capacity,"; at %s; last kernel launched %s",NvrmOperationName(nvrm->operations[at]),last);
	else if ( done + 1u == nvrm->kernel_sequence )
		snprintf(text,capacity,"; at kernel %s, the last one launched",nvrm->kernels[at]->name);
	else
		snprintf(text,capacity,"; at kernel %s; last launched %s",nvrm->kernels[at]->name,last);
}

static NvrmQueue *NvrmListedLocked(uint64_t queue_id)
{
	NvrmQueue *listed;
	for (listed = nvrm_queues; listed != 0 && queue_id != 0u; listed = listed->next)
		if ( listed->queue->id == queue_id )
			return(listed);
	return(0);
}

static SparkStatus NvrmReportLocked(NvrmQueue *nvrm, const SparkNvrmFault *fault, const char *operation)
{
	char head[NVRM_REPORT_HEAD_BYTES],where[NVRM_REPORT_WHERE_BYTES],text[SPARK_DEVICE_ERROR_TEXT_BYTES],name[SPARK_DEVICE_OWNER_BYTES];
	if ( fault != 0 )
		SparkNvrmFaultText(fault,head,sizeof(head));
	else
		snprintf(head,sizeof(head),"no GPU progress for %llu s",(unsigned long long)(SPARK_NVRM_WAIT_STALL_NS / 1000000000ull));
	where[0] = '\0';
	name[0] = '\0';
	if ( nvrm != 0 )
	{
		NvrmPosition(nvrm,where,sizeof(where));
		snprintf(name,sizeof(name),"%.26s %llu",nvrm->queue->owner,(unsigned long long)nvrm->queue->id);
	}
	snprintf(text,sizeof(text),"%.96s%.159s",head,where);
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation,name,fault != 0 ? (int32_t)fault->code : 0,text);
		SPARK_FAIL(status);
	}
}

static SparkStatus NvrmWatch(uint64_t queue_id, SparkNvrmWait *wait, const char *operation)
{
	SparkNvrmFault fault;
	const SparkNvrmChannel *faulted;
	NvrmQueue *nvrm;
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&nvrm_queues_lock);
	nvrm = NvrmListedLocked(queue_id);
	faulted = NvrmFaulted(nvrm,&fault);
	if ( faulted != 0 )
	{
		NvrmLocateLocked(nvrm,faulted,&fault);
		status = NvrmReportLocked(nvrm,&fault,operation);
	}
	else if ( wait != 0 && SparkNvrmWaitStalled(wait,NvrmProgressLocked(),SPARK_NVRM_WAIT_STALL_NS) != 0 )
		status = NvrmReportLocked(nvrm,0,operation);
	pthread_mutex_unlock(&nvrm_queues_lock);
	return(status);
}

static SparkStatus NvrmWaitChannel(uint64_t queue_id, const SparkNvrmChannel *channel, uint64_t value, const char *operation)
{
	SparkNvrmWait wait;
	SparkStatus status = SPARK_STATUS_OK;
	SparkNvrmWaitBegin(&wait);
	while ( SparkNvrmChannelCompleted(channel) < value && SparkNvrmWaitPace(&wait,SPARK_DEVICE_WAIT_FOREVER) == 0 )
		if ( SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) != 0 && (status = NvrmWatch(queue_id,&wait,operation)) != SPARK_STATUS_OK )
			break;
	SparkNvrmWaitEnd(&wait);
	return(status);
}

static SparkStatus NvrmRoom(SparkDeviceQueue *queue, const char *operation)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	while ( queue->submitted - NvrmCompletedSequence(nvrm) >= NVRM_SEQUENCES )
	{
		const uint64_t oldest = __atomic_load_n(&nvrm->completed,__ATOMIC_ACQUIRE) + 1u;
		const uint64_t end = nvrm->ends[oldest % NVRM_SEQUENCES];
		NVRM_TRY(NvrmWaitChannel(queue->id,NvrmEndChannel(nvrm,end),end & ~NVRM_END_COPY,operation));
	}
	return(SPARK_STATUS_OK);
}

static void NvrmEnd(SparkDeviceQueue *queue, NvrmOperation operation, const SparkDeviceKernel *kernel)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	const uint32_t at = (uint32_t)(queue->submitted % NVRM_SEQUENCES);
	const uint64_t end = nvrm->last == &nvrm->copy ? nvrm->copy.fence_value | NVRM_END_COPY : nvrm->channel.fence_value;
	nvrm->operations[at] = (uint8_t)operation;
	nvrm->kernels[at] = kernel;
	__atomic_store_n(&nvrm->ends[at],end,__ATOMIC_RELAXED);
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
	return(SparkNvrmHostWork(&nvrm_gpu,NvrmOn(nvrm,&nvrm->channel),&nvrm->worker,&item,0,queue->owner));
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
			NVRM_TRY(SparkNvrmCopy(&nvrm_gpu,NvrmOn(nvrm,&nvrm->copy),destination->device + destination_offset + done,nvrm->bounce.address,chunk,0,queue->owner));
		}
		else
		{
			NVRM_TRY(SparkNvrmCopy(&nvrm_gpu,NvrmOn(nvrm,&nvrm->copy),nvrm->bounce.address,source->device + source_offset + done,chunk,0,queue->owner));
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
		status = SparkNvrmCopy(&nvrm_gpu,NvrmOn(nvrm,&nvrm->copy),destination->device + destination_offset,source->device + source_offset,bytes,0,queue->owner);
	else if ( destination->host != 0 && source->host != 0 )
		status = NvrmHost(queue,SPARK_NVRM_HOST_COPY,(uint8_t *)destination->host + destination_offset,(const uint8_t *)source->host + source_offset,bytes,0u,"copy");
	else
		status = NvrmStaged(queue,destination,destination_offset,source,source_offset,bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	NvrmEnd(queue,NVRM_OPERATION_COPY,0);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	NvrmQueue *nvrm = (NvrmQueue *)queue->backend;
	SparkStatus status;
	NVRM_TRY(NvrmRoom(queue,"fill"));
	if ( destination->device != 0u )
		status = SparkNvrmFill(&nvrm_gpu,NvrmOn(nvrm,&nvrm->copy),destination->device + offset,value,bytes,0,queue->owner);
	else
		status = NvrmHost(queue,SPARK_NVRM_HOST_FILL,(uint8_t *)destination->host + offset,0,bytes,value,"fill");
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	NvrmEnd(queue,NVRM_OPERATION_FILL,0);
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
		status = SparkNvrmCopyChannelCreate(&nvrm_gpu,&nvrm->copy,queue->owner);
		if ( status != SPARK_STATUS_OK )
			SparkNvrmChannelDestroy(&nvrm_gpu,&nvrm->channel);
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkNvrmHostOpen(&nvrm_gpu,&nvrm->worker,queue->owner);
		if ( status != SPARK_STATUS_OK )
		{
			SparkNvrmChannelDestroy(&nvrm_gpu,&nvrm->copy);
			SparkNvrmChannelDestroy(&nvrm_gpu,&nvrm->channel);
		}
	}
	if ( status != SPARK_STATUS_OK )
	{
		free(nvrm);
		SPARK_FAIL(status);
	}
	nvrm->ends[0] = nvrm->channel.fence_value;
	nvrm->queue = queue;
	queue->backend = nvrm;
	pthread_mutex_lock(&nvrm_queues_lock);
	nvrm->next = nvrm_queues;
	nvrm_queues = nvrm;
	pthread_mutex_unlock(&nvrm_queues_lock);
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
	NvrmQueue **link;
	if ( nvrm == 0 )
		return;
	pthread_mutex_lock(&nvrm_queues_lock);
	for (link = &nvrm_queues; *link != 0; link = &(*link)->next)
		if ( *link == nvrm )
		{
			*link = nvrm->next;
			break;
		}
	pthread_mutex_unlock(&nvrm_queues_lock);
	SparkNvrmChannelDestroy(&nvrm_gpu,&nvrm->copy);
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
	SparkNvrmWait wait;
	SparkStatus status = SPARK_STATUS_OK;
	int expired = 0;
	SparkNvrmWaitBegin(&wait);
	while ( NvrmCompletedSequence(nvrm) < sequence && (expired = SparkNvrmWaitPace(&wait,timeout_ns)) == 0 )
		if ( SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) != 0 && (status = NvrmWatch(queue->id,&wait,"queue_wait")) != SPARK_STATUS_OK )
			break;
	SparkNvrmWaitEnd(&wait);
	if ( status == SPARK_STATUS_OK && expired != 0 )
		status = NvrmWatch(queue->id,0,"queue_wait");
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(expired != 0 ? SPARK_STATUS_PENDING : SPARK_STATUS_OK);
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
	NVRM_TRY(SparkNvrmHostWork(&nvrm_gpu,NvrmOn(nvrm,&nvrm->channel),&nvrm->worker,&item,0,queue->owner));
	NvrmEnd(queue,NVRM_OPERATION_HOST,0);
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
	NVRM_TRY(SparkNvrmMark(&nvrm_gpu,NvrmOn(owner,&owner->channel),nvrm->address,nvrm->generation + 1u,0,queue->owner));
	nvrm->generation++;
	nvrm->queue_id = queue->id;
	NvrmEnd(queue,NVRM_OPERATION_RECORD,0);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	const NvrmEvent *nvrm = (const NvrmEvent *)event->backend;
	NvrmQueue *waiter = (NvrmQueue *)queue->backend;
	NVRM_TRY(NvrmRoom(queue,"queue_wait_event"));
	NVRM_TRY(SparkNvrmAwait(&nvrm_gpu,NvrmOn(waiter,&waiter->channel),nvrm->address,nvrm->generation,0,queue->owner));
	NvrmEnd(queue,NVRM_OPERATION_AWAIT,0);
	return(SPARK_STATUS_OK);
}

static uint64_t NvrmEventReached(const NvrmEvent *nvrm)
{
	return(__atomic_load_n((const volatile uint64_t *)(const void *)nvrm->cpu,__ATOMIC_ACQUIRE));
}

SparkStatus SparkDeviceBackendEventWait(SparkDeviceEvent *event, uint64_t timeout_ns)
{
	const NvrmEvent *nvrm = (const NvrmEvent *)event->backend;
	const uint64_t generation = nvrm->generation;
	SparkNvrmWait wait;
	SparkStatus status = SPARK_STATUS_OK;
	int expired = 0;
	SparkNvrmWaitBegin(&wait);
	while ( NvrmEventReached(nvrm) < generation && (expired = SparkNvrmWaitPace(&wait,timeout_ns)) == 0 )
		if ( SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) != 0 && (status = NvrmWatch(nvrm->queue_id,&wait,"event_wait")) != SPARK_STATUS_OK )
			break;
	SparkNvrmWaitEnd(&wait);
	if ( status == SPARK_STATUS_OK && expired != 0 )
		status = NvrmWatch(nvrm->queue_id,0,"event_wait");
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(expired != 0 ? SPARK_STATUS_PENDING : SPARK_STATUS_OK);
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

SparkStatus SparkDeviceBackendKernelShortName(SparkDeviceLibrary *library, const char *name, char short_name[SPARK_DEVICE_KERNEL_NAME_LIMIT + 1u])
{
	const NvrmLibrary *nvrm = (const NvrmLibrary *)library->backend;
	SparkNvrmShortName(name,short_name);
	if ( nvrm == 0 || SparkNvrmCubinKernelNamed(&nvrm->program.cubin,short_name) == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"kernel_create",short_name,0,"the kernel library has no kernel with that symbol"));
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

SparkStatus SparkNvrmKernelResources(const SparkDeviceKernel *kernel, uint32_t *registers, uint32_t *stack_bytes, uint32_t *shared_bytes)
{
	const NvrmKernel *nvrm = kernel != 0 ? (const NvrmKernel *)kernel->backend : 0;
	if ( nvrm == 0 || registers == 0 || stack_bytes == 0 || shared_bytes == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_info",0,0,"needs a live nvrm kernel and every output"));
	*registers = nvrm->kernel->registers;
	*stack_bytes = nvrm->kernel->stack_bytes;
	*shared_bytes = nvrm->kernel->shared_bytes;
	return(SPARK_STATUS_OK);
}

int SparkNvrmOccupancyLimitsOf(SparkNvrmOccupancyLimits *limits)
{
	if ( limits == 0 || nvrm_gpu.warps_per_sm == 0u )
		return(0);
	limits->warps_per_sm = nvrm_gpu.warps_per_sm;
	limits->registers_per_sm = SPARK_NVRM_OCCUPANCY_REGISTERS_PER_SM;
	limits->blocks_per_sm = SPARK_NVRM_OCCUPANCY_BLOCKS_PER_SM;
	limits->shared_per_sm = SPARK_NVRM_OCCUPANCY_SHARED_PER_SM;
	limits->register_unit = SPARK_NVRM_OCCUPANCY_REGISTER_UNIT;
	limits->shared_unit = SPARK_NVRM_OCCUPANCY_SHARED_UNIT;
	return(1);
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
		NVRM_TRY(NvrmWaitChannel(nvrm->queue->id,&nvrm->channel,nvrm->slot_records[nvrm->slot_first].value,"kernel_launch"));
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
	request.cluster[0] = launch->cluster.x;
	request.cluster[1] = launch->cluster.y;
	request.cluster[2] = launch->cluster.z;
	request.dependent = launch->dependent;
	NVRM_TRY(SparkNvrmLaunch(&nvrm_gpu,NvrmOn(nvrm,&nvrm->channel),&compiled->library->program,&request,&value,queue->owner));
	NvrmSlotTaken(nvrm,start,bytes,value);
	memcpy(nvrm->kernel,kernel->name,sizeof(nvrm->kernel));
	nvrm->kernel_sequence = queue->submitted;
	NvrmEnd(queue,NVRM_OPERATION_LAUNCH,kernel);
	return(SPARK_STATUS_OK);
}

#define NVRM_INSTANCE UINT64_C(0x6e76726d00000001)
#define NVRM_SHARED_GRANULARITY SPARK_NVRM_HUGE_PAGE_BYTES

SparkStatus SparkDeviceBackendThreadInstance(SparkDeviceInstance *instance)
{
	*instance = NVRM_INSTANCE;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendThreadBind(SparkDeviceInstance instance)
{
	if ( instance != NVRM_INSTANCE )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"thread_bind","",0,"not the instance this device layer returned"));
	return(SPARK_STATUS_OK);
}

static uint32_t NvrmSharedAligned(uint64_t value)
{
	return((value & (NVRM_SHARED_GRANULARITY - 1u)) == 0u ? 1u : 0u);
}

static SparkStatus NvrmSpansRoom(NvrmSpans *spans, uint32_t more, const char *operation)
{
	NvrmSpan *grown;
	uint32_t capacity = spans->capacity != 0u ? spans->capacity : 64u;
	if ( spans->count + more <= spans->capacity )
		return(SPARK_STATUS_OK);
	while ( capacity < spans->count + more )
		capacity *= 2u;
	grown = (NvrmSpan *)realloc(spans->items,(size_t)capacity * sizeof(*grown));
	if ( grown == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,"",ENOMEM,"out of host memory for the shared memory table"));
	spans->items = grown;
	spans->capacity = capacity;
	return(SPARK_STATUS_OK);
}

static uint32_t NvrmSpansFirstEnding(const NvrmSpans *spans, uint64_t address)
{
	uint32_t index = 0u;
	while ( index < spans->count && spans->items[index].address + spans->items[index].bytes <= address )
		index++;
	return(index);
}

static void NvrmSpansInsert(NvrmSpans *spans, uint32_t index, uint64_t address, uint64_t bytes, SparkDeviceMemoryHandle memory)
{
	memmove(spans->items + index + 1u,spans->items + index,(size_t)(spans->count - index) * sizeof(spans->items[0]));
	spans->items[index].address = address;
	spans->items[index].bytes = bytes;
	spans->items[index].memory = memory;
	spans->count++;
}

static void NvrmSpansRemove(NvrmSpans *spans, uint32_t index, uint32_t count)
{
	memmove(spans->items + index,spans->items + index + count,(size_t)(spans->count - index - count) * sizeof(spans->items[0]));
	spans->count -= count;
}

static NvrmShared *NvrmSharedFind(SparkDeviceMemoryHandle memory)
{
	uint32_t index;
	for (index = 0u; index < nvrm_shared_count; index++)
		if ( nvrm_shared[index].id == memory )
			return(&nvrm_shared[index]);
	return(0);
}

static SparkStatus NvrmSharedAdd(uint32_t handle, uint64_t bytes, SparkDeviceMemoryHandle *memory, const char *operation)
{
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&nvrm_shared_lock);
	if ( nvrm_shared_count == nvrm_shared_capacity )
	{
		const uint32_t capacity = nvrm_shared_capacity != 0u ? nvrm_shared_capacity * 2u : 64u;
		NvrmShared *grown = (NvrmShared *)realloc(nvrm_shared,(size_t)capacity * sizeof(*grown));
		if ( grown == 0 )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,"",ENOMEM,"out of host memory for the shared memory table");
		else
		{
			nvrm_shared = grown;
			nvrm_shared_capacity = capacity;
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		NvrmShared *shared = &nvrm_shared[nvrm_shared_count++];
		memset(shared,0,sizeof(*shared));
		shared->id = ++nvrm_shared_serial;
		shared->handle = handle;
		shared->bytes = bytes;
		shared->held = 1u;
		*memory = shared->id;
	}
	pthread_mutex_unlock(&nvrm_shared_lock);
	return(status);
}

static void NvrmSharedSettle(NvrmShared *shared)
{
	if ( shared->held != 0u || shared->maps != 0u )
		return;
	(void)SparkNvrmRmFree(&nvrm_gpu.rm,nvrm_gpu.device,shared->handle,"free");
	*shared = nvrm_shared[--nvrm_shared_count];
}

static void NvrmSharedClose(void)
{
	free(nvrm_shared);
	free(nvrm_reservations.items);
	free(nvrm_mappings.items);
	nvrm_shared = 0;
	nvrm_shared_count = 0u;
	nvrm_shared_capacity = 0u;
	memset(&nvrm_reservations,0,sizeof(nvrm_reservations));
	memset(&nvrm_mappings,0,sizeof(nvrm_mappings));
}

static uint32_t NvrmMappingsSpan(uint64_t address, uint64_t bytes)
{
	uint32_t index,spanned = 0u;
	uint64_t reached = address;
	pthread_mutex_lock(&nvrm_shared_lock);
	index = NvrmSpansFirstEnding(&nvrm_mappings,address);
	while ( index < nvrm_mappings.count && nvrm_mappings.items[index].address <= reached && reached < address + bytes )
	{
		reached = nvrm_mappings.items[index].address + nvrm_mappings.items[index].bytes;
		index++;
	}
	spanned = bytes != 0u && address + bytes > address && reached >= address + bytes ? 1u : 0u;
	pthread_mutex_unlock(&nvrm_shared_lock);
	return(spanned);
}

SparkStatus SparkNvrmResourceCountsRead(SparkNvrmResourceCounts *counts)
{
	if ( counts == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"resource_counts","",0,"needs an output"));
	memset(counts,0,sizeof(*counts));
	pthread_mutex_lock(&nvrm_shared_lock);
	counts->shared_memory = nvrm_shared_count;
	counts->reservations = nvrm_reservations.count;
	counts->mappings = nvrm_mappings.count;
	pthread_mutex_unlock(&nvrm_shared_lock);
	counts->rm_objects = __atomic_load_n(&nvrm_gpu.rm.objects,__ATOMIC_ACQUIRE);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendMemoryGranularity(uint32_t granularity, uint64_t *bytes)
{
	(void)granularity;
	*bytes = NVRM_SHARED_GRANULARITY;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendMemoryCreate(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner)
{
	uint32_t handle = 0u;
	SparkStatus status;
	if ( NvrmSharedAligned(bytes) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_create",owner,0,"a shared allocation is a multiple of 2 MiB"));
	NVRM_TRY(SparkNvrmSystemAllocate(&nvrm_gpu,bytes,SparkNvrmSystemPageBytes(bytes),0u,&handle,owner));
	status = NvrmSharedAdd(handle,bytes,memory,"memory_create");
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkNvrmRmFree(&nvrm_gpu.rm,nvrm_gpu.device,handle,"free");
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendMemoryRelease(SparkDeviceMemoryHandle memory)
{
	NvrmShared *shared;
	pthread_mutex_lock(&nvrm_shared_lock);
	shared = NvrmSharedFind(memory);
	if ( shared == 0 || shared->held == 0u )
	{
		pthread_mutex_unlock(&nvrm_shared_lock);
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_release","",0,"not a live shared memory handle of this process"));
	}
	shared->held = 0u;
	NvrmSharedSettle(shared);
	pthread_mutex_unlock(&nvrm_shared_lock);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendMemoryExport(SparkDeviceMemoryHandle memory, int *fd)
{
	NvrmShared *shared;
	SparkStatus status;
	pthread_mutex_lock(&nvrm_shared_lock);
	shared = NvrmSharedFind(memory);
	status = shared != 0 && shared->held != 0u ? SparkNvrmShareExport(&nvrm_gpu,shared->handle,fd,"") :
		SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_export","",0,"not a live shared memory handle of this process");
	pthread_mutex_unlock(&nvrm_shared_lock);
	return(status);
}

SparkStatus SparkDeviceBackendMemoryImport(SparkDeviceMemoryHandle *memory, int fd, const char *owner)
{
	uint32_t handle = 0u;
	uint64_t bytes = 0u;
	SparkStatus status;
	NVRM_TRY(SparkNvrmShareImport(&nvrm_gpu,fd,&handle,&bytes,owner));
	status = NvrmSharedAligned(bytes) != 0u && bytes != 0u ? NvrmSharedAdd(handle,bytes,memory,"memory_import") :
		SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_import",owner,0,"the descriptor's memory is not a whole number of 2 MiB pages");
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkNvrmRmFree(&nvrm_gpu.rm,nvrm_gpu.device,handle,"free");
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendAddressReserve(SparkDeviceAddress *address, uint64_t bytes, const char *owner)
{
	uint64_t reserved = 0u;
	SparkStatus status;
	uint32_t index;
	if ( NvrmSharedAligned(bytes) == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"address_reserve",owner,0,"a reservation is a multiple of 2 MiB"));
	NVRM_TRY(SparkNvrmVirtual(&nvrm_gpu,bytes,NVRM_SHARED_GRANULARITY,&reserved,owner));
	pthread_mutex_lock(&nvrm_shared_lock);
	status = NvrmSpansRoom(&nvrm_reservations,1u,"address_reserve");
	if ( status == SPARK_STATUS_OK )
	{
		index = NvrmSpansFirstEnding(&nvrm_reservations,reserved);
		NvrmSpansInsert(&nvrm_reservations,index,reserved,bytes,0u);
	}
	pthread_mutex_unlock(&nvrm_shared_lock);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmVirtualRelease(&nvrm_gpu,reserved,bytes);
		SPARK_FAIL(status);
	}
	*address = reserved;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendAddressRelease(SparkDeviceAddress address, uint64_t bytes)
{
	uint32_t index,mapping;
	pthread_mutex_lock(&nvrm_shared_lock);
	index = NvrmSpansFirstEnding(&nvrm_reservations,address);
	mapping = NvrmSpansFirstEnding(&nvrm_mappings,address);
	if ( index >= nvrm_reservations.count || nvrm_reservations.items[index].address != address || nvrm_reservations.items[index].bytes != bytes )
	{
		pthread_mutex_unlock(&nvrm_shared_lock);
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"address_release","",0,"not a whole reservation of this process"));
	}
	if ( mapping < nvrm_mappings.count && nvrm_mappings.items[mapping].address < address + bytes )
	{
		pthread_mutex_unlock(&nvrm_shared_lock);
		return(SparkDeviceRecordError(SPARK_STATUS_BUSY,"address_release","",0,"the reservation still holds a mapping"));
	}
	NvrmSpansRemove(&nvrm_reservations,index,1u);
	pthread_mutex_unlock(&nvrm_shared_lock);
	SparkNvrmVirtualRelease(&nvrm_gpu,address,bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus NvrmMapCheck(SparkDeviceAddress address, uint64_t bytes, const NvrmShared *shared, uint64_t offset)
{
	const uint32_t reservation = NvrmSpansFirstEnding(&nvrm_reservations,address);
	const uint32_t mapping = NvrmSpansFirstEnding(&nvrm_mappings,address);
	if ( NvrmSharedAligned(address) == 0u || NvrmSharedAligned(bytes) == 0u || NvrmSharedAligned(offset) == 0u || address + bytes < address )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"a mapping's address, size and offset are multiples of 2 MiB"));
	if ( reservation >= nvrm_reservations.count || nvrm_reservations.items[reservation].address > address ||
		nvrm_reservations.items[reservation].address + nvrm_reservations.items[reservation].bytes < address + bytes )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"a mapping lies inside one reservation of this process"));
	if ( mapping < nvrm_mappings.count && nvrm_mappings.items[mapping].address < address + bytes )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"the range already holds a mapping"));
	if ( shared == 0 || shared->held == 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"not a live shared memory handle of this process"));
	if ( offset > shared->bytes || bytes > shared->bytes - offset )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_map","",0,"the mapping runs past the end of the memory"));
	return(NvrmSpansRoom(&nvrm_mappings,1u,"memory_map"));
}

SparkStatus SparkDeviceBackendMemoryMap(SparkDeviceAddress address, uint64_t bytes, SparkDeviceMemoryHandle memory, uint64_t offset, uint32_t access)
{
	NvrmShared *shared;
	SparkStatus status;
	pthread_mutex_lock(&nvrm_shared_lock);
	shared = NvrmSharedFind(memory);
	status = NvrmMapCheck(address,bytes,shared,offset);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmGpuMap(&nvrm_gpu,address,bytes,shared->handle,offset,(access & SPARK_DEVICE_ACCESS_DEVICE_WRITE) == 0u ? 1u : 0u,"memory_map");
	if ( status == SPARK_STATUS_OK )
	{
		NvrmSpansInsert(&nvrm_mappings,NvrmSpansFirstEnding(&nvrm_mappings,address),address,bytes,memory);
		shared->maps++;
	}
	pthread_mutex_unlock(&nvrm_shared_lock);
	return(status);
}

SparkStatus SparkDeviceBackendMemoryUnmap(SparkDeviceAddress address, uint64_t bytes)
{
	const uint64_t end = address + bytes;
	uint32_t first,last;
	uint64_t reached = address;
	pthread_mutex_lock(&nvrm_shared_lock);
	first = NvrmSpansFirstEnding(&nvrm_mappings,address);
	last = first;
	while ( last < nvrm_mappings.count && nvrm_mappings.items[last].address == reached && reached < end )
		reached += nvrm_mappings.items[last++].bytes;
	if ( end < address || first >= nvrm_mappings.count || nvrm_mappings.items[first].address != address || reached != end )
	{
		pthread_mutex_unlock(&nvrm_shared_lock);
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"memory_unmap","",0,"an unmap takes one or more whole, adjacent mappings"));
	}
	for (uint32_t index = first; index < last; index++)
	{
		NvrmShared *shared = NvrmSharedFind(nvrm_mappings.items[index].memory);
		if ( shared == 0 )
			continue;
		SparkNvrmGpuUnmap(&nvrm_gpu,nvrm_mappings.items[index].address,nvrm_mappings.items[index].bytes,shared->handle);
		shared->maps--;
		NvrmSharedSettle(shared);
	}
	NvrmSpansRemove(&nvrm_mappings,first,last - first);
	pthread_mutex_unlock(&nvrm_shared_lock);
	return(SPARK_STATUS_OK);
}

uint32_t SparkDeviceBackendProgramNative(void)
{
	return(0u);
}

SparkStatus SparkDeviceBackendProgramReplay(SparkDeviceQueue *queue)
{
	const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_UNSUPPORTED,"program_run",queue->owner,0,
		"the nvrm backend does not run programs yet, and replaying their operations one by one would be a slow path");
	SPARK_FAIL(status);
}

SparkStatus SparkDeviceBackendProgramBegin(SparkDeviceQueue *queue, SparkDeviceProgram *program)
{
	(void)queue;
	(void)program;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendProgramPatch(SparkDeviceQueue *queue, SparkDeviceProgram *program, SparkDevicePlace *place)
{
	(void)queue;
	(void)place;
	return(SparkDeviceRecordError(SPARK_STATUS_INTERNAL_ERROR,"program_patch",program->name,0,"the nvrm backend records programs in the device layer"));
}

SparkStatus SparkDeviceBackendProgramEnd(SparkDeviceQueue *queue, SparkDeviceProgram *program)
{
	(void)queue;
	return(SparkDeviceRecordError(SPARK_STATUS_INTERNAL_ERROR,"program_finish",program->name,0,"the nvrm backend records programs in the device layer"));
}

SparkStatus SparkDeviceBackendProgramRun(SparkDeviceProgram *program, SparkDeviceQueue *queue, const uint64_t *values)
{
	(void)queue;
	(void)values;
	return(SparkDeviceRecordError(SPARK_STATUS_INTERNAL_ERROR,"program_run",program->name,0,"the nvrm backend does not run programs yet"));
}

void SparkDeviceBackendProgramRelease(SparkDeviceProgram *program)
{
	(void)program;
}
