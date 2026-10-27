#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include "runtime/device/device_backend.h"

typedef enum MetalMemoryKind
{
	METAL_MEMORY_PRIVATE = 1,
	METAL_MEMORY_SHARED = 2,
	METAL_MEMORY_HEAP = 3,
	METAL_MEMORY_BORROWED = 4,
	METAL_MEMORY_FILE = 5
} MetalMemoryKind;

typedef struct MetalAllocation
{
	MetalMemoryKind kind;
	void *buffer;
	uint64_t offset;
	void *memory;
	uint64_t memory_bytes;
} MetalAllocation;

#define METAL_COMMIT_OPERATIONS 64u
#define METAL_ERROR_BYTES 128u

typedef struct MetalEvent
{
	pthread_mutex_t mutex;
	pthread_cond_t done;
	uint64_t recorded;
	uint64_t completed;
	uint64_t timestamp_ns;
	void *shared;
} MetalEvent;

typedef enum MetalMarkKind
{
	METAL_MARK_NONE = 0,
	METAL_MARK_EVENT = 1,
	METAL_MARK_CALLBACK = 2
} MetalMarkKind;

typedef struct MetalMark
{
	MetalMarkKind kind;
	MetalEvent *event;
	uint64_t generation;
	SparkDeviceHostFunction function;
	void *context;
} MetalMark;

typedef struct MetalBatch
{
	uint64_t sequence;
	MetalMark mark;
	uint32_t done;
	int32_t code;
	uint64_t end_ns;
	char error[METAL_ERROR_BYTES];
	struct MetalBatch *next;
} MetalBatch;

typedef struct MetalQueue
{
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t wake;
	pthread_cond_t progress;
	pthread_mutex_t encode;
	MetalBatch *head;
	MetalBatch *tail;
	uint64_t completed;
	uint32_t stop;
	int32_t error_code;
	char error[METAL_ERROR_BYTES];
	void *commands;
	void *open;
	void *compute;
	uint32_t open_operations;
	uint64_t open_sequence;
	void *gate;
	uint64_t gate_target;
	char owner[SPARK_DEVICE_OWNER_BYTES];
} MetalQueue;

static id<MTLDevice> metal_device;
static id<MTLCommandQueue> metal_setup_queue;
static uint64_t metal_page_bytes;

static uint64_t MetalNowNs(void)
{
	return(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
}

static void MetalDeadline(uint64_t timeout_ns, struct timespec *deadline)
{
	uint64_t nanoseconds;
	clock_gettime(CLOCK_REALTIME,deadline);
	if ( timeout_ns > 3153600000000000000ull )
		timeout_ns = 3153600000000000000ull;
	nanoseconds = (uint64_t)deadline->tv_nsec + timeout_ns % 1000000000ull;
	deadline->tv_sec += (time_t)(timeout_ns / 1000000000ull + nanoseconds / 1000000000ull);
	deadline->tv_nsec = (long)(nanoseconds % 1000000000ull);
}

static uint64_t MetalRound(uint64_t bytes)
{
	return((bytes + metal_page_bytes - 1u) / metal_page_bytes * metal_page_bytes);
}

static uint32_t MetalGpuCores(void)
{
	io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault,IOServiceMatching("AGXAccelerator"));
	CFTypeRef property;
	uint32_t cores = 0u;
	if ( service == IO_OBJECT_NULL )
		return(0u);
	property = IORegistryEntrySearchCFProperty(service,kIOServicePlane,CFSTR("gpu-core-count"),kCFAllocatorDefault,kIORegistryIterateRecursively);
	if ( property != 0 && CFGetTypeID(property) == CFNumberGetTypeID() )
		CFNumberGetValue((CFNumberRef)property,kCFNumberSInt32Type,&cores);
	if ( property != 0 )
		CFRelease(property);
	IOObjectRelease(service);
	return(cores);
}

SparkStatus SparkDeviceBackendInitialize(const SparkDeviceConfiguration *configuration, SparkDeviceProperties *properties)
{
	long page = sysconf(_SC_PAGESIZE);
	@autoreleasepool
	{
		if ( configuration->ordinal != 0u )
			return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"initialize","",0,"the Metal backend drives the system default GPU, ordinal 0"));
		metal_device = MTLCreateSystemDefaultDevice();
		if ( metal_device == nil )
			return(SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"initialize","",0,"no Metal device"));
		metal_setup_queue = [metal_device newCommandQueue];
		if ( metal_setup_queue == nil )
		{
			metal_device = nil;
			return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"initialize","",0,"cannot create a Metal command queue"));
		}
		metal_page_bytes = page > 0 ? (uint64_t)page : 16384u;
		snprintf(properties->backend,sizeof(properties->backend),"%s","metal");
		snprintf(properties->target,sizeof(properties->target),"metal.%s",[metal_device supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple9 + 1)] ? "apple10" :
			[metal_device supportsFamily:MTLGPUFamilyApple9] ? "apple9" : "apple");
		snprintf(properties->name,sizeof(properties->name),"%s",metal_device.name.UTF8String);
		properties->page_bytes = (uint32_t)metal_page_bytes;
		properties->compute_units = MetalGpuCores();
		properties->simd_width = 32u;
		properties->memory_budget_bytes = configuration->memory_budget_bytes != 0u ? configuration->memory_budget_bytes :
			(uint64_t)metal_device.recommendedMaxWorkingSetSize;
		properties->max_buffer_bytes = (uint64_t)metal_device.maxBufferLength;
		properties->properties = metal_device.hasUnifiedMemory ? SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY : 0u;
		properties->max_group_threads = (uint32_t)metal_device.maxThreadsPerThreadgroup.width;
		properties->max_shared_bytes = (uint32_t)metal_device.maxThreadgroupMemoryLength;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	metal_setup_queue = nil;
	metal_device = nil;
	return(SPARK_STATUS_OK);
}

static id<MTLBuffer> MetalBuffer(const MetalAllocation *metal)
{
	return((__bridge id<MTLBuffer>)metal->buffer);
}

static void MetalReleaseRecord(MetalAllocation *metal)
{
	if ( metal->buffer != 0 )
		CFRelease(metal->buffer);
	if ( metal->kind == METAL_MEMORY_HEAP )
		free(metal->memory);
	else if ( metal->kind == METAL_MEMORY_FILE && metal->memory != 0 )
		munmap(metal->memory,(size_t)metal->memory_bytes);
	free(metal);
}

static SparkStatus MetalWrap(SparkDeviceAllocation *allocation, MetalAllocation *metal, void *base, uint64_t bytes, const char *operation)
{
	id<MTLBuffer> buffer = [metal_device newBufferWithBytesNoCopy:base length:(NSUInteger)bytes options:MTLResourceStorageModeShared deallocator:nil];
	if ( buffer == nil )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,allocation->owner,0,"Metal cannot wrap the host pages"));
	metal->buffer = (__bridge_retained void *)buffer;
	return(SPARK_STATUS_OK);
}

static SparkStatus MetalZero(SparkDeviceAllocation *allocation, id<MTLBuffer> buffer)
{
	@autoreleasepool
	{
		id<MTLCommandBuffer> commands = [metal_setup_queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
		[blit fillBuffer:buffer range:NSMakeRange(0u,(NSUInteger)buffer.length) value:0u];
		[blit endEncoding];
		[commands commit];
		[commands waitUntilCompleted];
		if ( commands.status != MTLCommandBufferStatusCompleted )
			return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"allocate",allocation->owner,(int32_t)commands.status,"the Metal zero fill failed"));
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendAllocate(SparkDeviceAllocation *allocation)
{
	MetalAllocation *metal = (MetalAllocation *)calloc(1u,sizeof(*metal));
	SparkStatus status = SPARK_STATUS_OK;
	if ( metal == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,ENOMEM,"out of host memory"));
	@autoreleasepool
	{
		if ( allocation->space == SPARK_DEVICE_SPACE_HOST_PAGEABLE )
		{
			metal->kind = METAL_MEMORY_HEAP;
			metal->memory_bytes = MetalRound(allocation->bytes);
			if ( posix_memalign(&metal->memory,(size_t)metal_page_bytes,(size_t)metal->memory_bytes) != 0 )
			{
				metal->memory = 0;
				MetalReleaseRecord(metal);
				return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,ENOMEM,"out of host memory"));
			}
			memset(metal->memory,0,(size_t)metal->memory_bytes);
			status = MetalWrap(allocation,metal,metal->memory,metal->memory_bytes,"allocate");
			if ( status != SPARK_STATUS_OK )
			{
				MetalReleaseRecord(metal);
				return(status);
			}
			allocation->host = metal->memory;
			allocation->device = 0u;
			allocation->backend = metal;
			return(SPARK_STATUS_OK);
		}
		{
			const uint32_t private_space = allocation->space == SPARK_DEVICE_SPACE_DEVICE_PRIVATE ? 1u : 0u;
			id<MTLBuffer> buffer = [metal_device newBufferWithLength:(NSUInteger)allocation->bytes
				options:private_space != 0u ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared];
			if ( buffer == nil )
			{
				MetalReleaseRecord(metal);
				return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,0,"Metal cannot allocate the buffer"));
			}
			metal->kind = private_space != 0u ? METAL_MEMORY_PRIVATE : METAL_MEMORY_SHARED;
			metal->buffer = (__bridge_retained void *)buffer;
			if ( private_space != 0u )
				status = MetalZero(allocation,buffer);
			else
				memset(buffer.contents,0,(size_t)allocation->bytes);
			if ( status != SPARK_STATUS_OK )
			{
				MetalReleaseRecord(metal);
				return(status);
			}
			allocation->host = private_space != 0u ? 0 : buffer.contents;
			allocation->device = (SparkDeviceAddress)buffer.gpuAddress;
			allocation->backend = metal;
		}
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendRegister(SparkDeviceAllocation *allocation, void *host)
{
	MetalAllocation *metal = (MetalAllocation *)calloc(1u,sizeof(*metal));
	uintptr_t base = (uintptr_t)host / metal_page_bytes * metal_page_bytes;
	SparkStatus status;
	if ( metal == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"register",allocation->owner,ENOMEM,"out of host memory"));
	metal->kind = METAL_MEMORY_BORROWED;
	metal->offset = (uint64_t)((uintptr_t)host - base);
	@autoreleasepool
	{
		status = MetalWrap(allocation,metal,(void *)base,MetalRound(metal->offset + allocation->bytes),"register");
	}
	if ( status != SPARK_STATUS_OK )
	{
		MetalReleaseRecord(metal);
		return(status);
	}
	allocation->host = host;
	allocation->device = (SparkDeviceAddress)MetalBuffer(metal).gpuAddress + metal->offset;
	allocation->backend = metal;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendMapFile(SparkDeviceAllocation *allocation, int fd, uint64_t offset)
{
	MetalAllocation *metal = (MetalAllocation *)calloc(1u,sizeof(*metal));
	uint64_t aligned = offset / metal_page_bytes * metal_page_bytes;
	SparkStatus status;
	if ( metal == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"map_file",allocation->owner,ENOMEM,"out of host memory"));
	metal->kind = METAL_MEMORY_FILE;
	metal->offset = offset - aligned;
	metal->memory_bytes = MetalRound(metal->offset + allocation->bytes);
	metal->memory = mmap(0,(size_t)metal->memory_bytes,PROT_READ,MAP_SHARED,fd,(off_t)aligned);
	if ( metal->memory == MAP_FAILED )
	{
		int error = errno;
		metal->memory = 0;
		MetalReleaseRecord(metal);
		return(SparkDeviceRecordError(error == ENOMEM ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_INVALID_ARGUMENT,
			"map_file",allocation->owner,error,"cannot map the file range"));
	}
	@autoreleasepool
	{
		status = MetalWrap(allocation,metal,metal->memory,metal->memory_bytes,"map_file");
	}
	if ( status != SPARK_STATUS_OK )
	{
		MetalReleaseRecord(metal);
		return(status);
	}
	allocation->host = (uint8_t *)metal->memory + metal->offset;
	allocation->device = (SparkDeviceAddress)MetalBuffer(metal).gpuAddress + metal->offset;
	allocation->backend = metal;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendRelease(SparkDeviceAllocation *allocation)
{
	if ( allocation->backend != 0 )
		MetalReleaseRecord((MetalAllocation *)allocation->backend);
	allocation->backend = 0;
}

static void MetalReleaseObject(void **object)
{
	if ( *object != 0 )
		CFRelease(*object);
	*object = 0;
}

static void MetalBatchDone(MetalQueue *queue, MetalBatch *batch, id<MTLCommandBuffer> buffer)
{
	pthread_mutex_lock(&queue->mutex);
	batch->end_ns = buffer.GPUEndTime > 0.0 ? (uint64_t)(buffer.GPUEndTime * 1e9) : MetalNowNs();
	if ( buffer.status != MTLCommandBufferStatusCompleted )
	{
		batch->code = buffer.error != nil ? (int32_t)buffer.error.code : -1;
		snprintf(batch->error,sizeof(batch->error),"%s",buffer.error != nil ? buffer.error.localizedDescription.UTF8String : "the Metal command buffer failed");
	}
	batch->done = 1u;
	pthread_cond_broadcast(&queue->wake);
	pthread_mutex_unlock(&queue->mutex);
}

static void MetalFinishMark(MetalQueue *queue, const MetalBatch *batch)
{
	const MetalMark *mark = &batch->mark;
	if ( mark->kind == METAL_MARK_EVENT )
	{
		pthread_mutex_lock(&mark->event->mutex);
		if ( mark->event->completed < mark->generation )
		{
			mark->event->completed = mark->generation;
			mark->event->timestamp_ns = batch->end_ns;
		}
		pthread_cond_broadcast(&mark->event->done);
		pthread_mutex_unlock(&mark->event->mutex);
	}
	else if ( mark->kind == METAL_MARK_CALLBACK )
	{
		mark->function(mark->context);
		((__bridge id<MTLSharedEvent>)queue->gate).signaledValue = mark->generation;
	}
}

static void *MetalWorker(void *argument)
{
	MetalQueue *queue = (MetalQueue *)argument;
	pthread_mutex_lock(&queue->mutex);
	for ( ;; )
	{
		MetalBatch *batch;
		while ( (queue->head == 0 && queue->stop == 0u) || (queue->head != 0 && queue->head->done == 0u) )
			pthread_cond_wait(&queue->wake,&queue->mutex);
		if ( queue->head == 0 )
			break;
		batch = queue->head;
		queue->head = batch->next;
		if ( queue->head == 0 )
			queue->tail = 0;
		if ( batch->code != 0 && queue->error_code == 0 )
		{
			queue->error_code = batch->code;
			memcpy(queue->error,batch->error,sizeof(queue->error));
		}
		pthread_mutex_unlock(&queue->mutex);
		if ( batch->code != 0 )
			SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"queue_worker",queue->owner,batch->code,batch->error);
		@autoreleasepool
		{
			MetalFinishMark(queue,batch);
		}
		pthread_mutex_lock(&queue->mutex);
		queue->completed = batch->sequence;
		pthread_cond_broadcast(&queue->progress);
		free(batch);
	}
	pthread_mutex_unlock(&queue->mutex);
	return(0);
}

static id<MTLCommandBuffer> MetalOpen(MetalQueue *queue)
{
	if ( queue->open == 0 )
	{
		id<MTLCommandBuffer> buffer = [(__bridge id<MTLCommandQueue>)queue->commands commandBufferWithUnretainedReferences];
		if ( buffer == nil )
			return(nil);
		if ( queue->gate_target != 0u )
			[buffer encodeWaitForEvent:(__bridge id<MTLSharedEvent>)queue->gate value:queue->gate_target];
		queue->open = (__bridge_retained void *)buffer;
	}
	return((__bridge id<MTLCommandBuffer>)queue->open);
}

static void MetalEndCompute(MetalQueue *queue)
{
	if ( queue->compute != 0 )
	{
		[(__bridge id<MTLComputeCommandEncoder>)queue->compute endEncoding];
		MetalReleaseObject(&queue->compute);
	}
}

static SparkStatus MetalCommit(MetalQueue *queue, const MetalMark *mark, const char *operation)
{
	MetalBatch *batch;
	id<MTLCommandBuffer> buffer;
	if ( queue->open == 0 && mark == 0 )
		return(SPARK_STATUS_OK);
	buffer = MetalOpen(queue);
	batch = (MetalBatch *)calloc(1u,sizeof(*batch));
	if ( buffer == nil || batch == 0 )
	{
		free(batch);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,queue->owner,0,"cannot create a Metal command buffer"));
	}
	MetalEndCompute(queue);
	batch->sequence = queue->open_sequence;
	if ( mark != 0 )
		batch->mark = *mark;
	if ( mark != 0 && mark->kind == METAL_MARK_EVENT )
		[buffer encodeSignalEvent:(__bridge id<MTLSharedEvent>)mark->event->shared value:mark->generation];
	pthread_mutex_lock(&queue->mutex);
	if ( queue->tail != 0 )
		queue->tail->next = batch;
	else
		queue->head = batch;
	queue->tail = batch;
	pthread_mutex_unlock(&queue->mutex);
	[buffer addCompletedHandler:^(id<MTLCommandBuffer> done) { MetalBatchDone(queue,batch,done); }];
	[buffer commit];
	MetalReleaseObject(&queue->open);
	queue->open_operations = 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus MetalCounted(MetalQueue *queue, uint64_t sequence, const char *operation)
{
	queue->open_sequence = sequence;
	if ( ++queue->open_operations < METAL_COMMIT_OPERATIONS )
		return(SPARK_STATUS_OK);
	return(MetalCommit(queue,0,operation));
}

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue)
{
	MetalQueue *metal = (MetalQueue *)calloc(1u,sizeof(*metal));
	int error;
	if ( metal == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",queue->owner,ENOMEM,"out of host memory"));
	@autoreleasepool
	{
		id<MTLCommandQueue> commands = [metal_device newCommandQueue];
		id<MTLSharedEvent> gate = [metal_device newSharedEvent];
		if ( commands == nil || gate == nil )
		{
			free(metal);
			return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"queue_create",queue->owner,0,"cannot create a Metal command queue"));
		}
		metal->commands = (__bridge_retained void *)commands;
		metal->gate = (__bridge_retained void *)gate;
	}
	memcpy(metal->owner,queue->owner,sizeof(metal->owner));
	pthread_mutex_init(&metal->mutex,0);
	pthread_mutex_init(&metal->encode,0);
	pthread_cond_init(&metal->wake,0);
	pthread_cond_init(&metal->progress,0);
	error = pthread_create(&metal->thread,0,MetalWorker,metal);
	if ( error != 0 )
	{
		pthread_cond_destroy(&metal->progress);
		pthread_cond_destroy(&metal->wake);
		pthread_mutex_destroy(&metal->encode);
		pthread_mutex_destroy(&metal->mutex);
		MetalReleaseObject(&metal->gate);
		MetalReleaseObject(&metal->commands);
		free(metal);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",queue->owner,error,"cannot start the queue worker"));
	}
	queue->backend = metal;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	if ( metal == 0 )
		return;
	pthread_mutex_lock(&metal->mutex);
	metal->stop = 1u;
	pthread_cond_broadcast(&metal->wake);
	pthread_mutex_unlock(&metal->mutex);
	pthread_join(metal->thread,0);
	MetalEndCompute(metal);
	MetalReleaseObject(&metal->open);
	pthread_cond_destroy(&metal->progress);
	pthread_cond_destroy(&metal->wake);
	pthread_mutex_destroy(&metal->encode);
	pthread_mutex_destroy(&metal->mutex);
	MetalReleaseObject(&metal->gate);
	MetalReleaseObject(&metal->commands);
	free(metal);
	queue->backend = 0;
}

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	const MetalAllocation *target = (const MetalAllocation *)destination->backend, *origin = (const MetalAllocation *)source->backend;
	SparkStatus status = SPARK_STATUS_OK;
	(void)direction;
	pthread_mutex_lock(&metal->encode);
	@autoreleasepool
	{
		id<MTLCommandBuffer> buffer = MetalOpen(metal);
		id<MTLBlitCommandEncoder> blit;
		MetalEndCompute(metal);
		blit = buffer != nil ? [buffer blitCommandEncoder] : nil;
		if ( blit == nil )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"copy",queue->owner,0,"cannot create a Metal blit encoder");
		else
		{
			[blit copyFromBuffer:(__bridge id<MTLBuffer>)origin->buffer sourceOffset:(NSUInteger)(origin->offset + source_offset)
				toBuffer:(__bridge id<MTLBuffer>)target->buffer destinationOffset:(NSUInteger)(target->offset + destination_offset) size:(NSUInteger)bytes];
			[blit endEncoding];
			status = MetalCounted(metal,queue->submitted,"copy");
		}
	}
	pthread_mutex_unlock(&metal->encode);
	return(status);
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	const MetalAllocation *target = (const MetalAllocation *)destination->backend;
	SparkStatus status = SPARK_STATUS_OK;
	pthread_mutex_lock(&metal->encode);
	@autoreleasepool
	{
		id<MTLCommandBuffer> buffer = MetalOpen(metal);
		id<MTLBlitCommandEncoder> blit;
		MetalEndCompute(metal);
		blit = buffer != nil ? [buffer blitCommandEncoder] : nil;
		if ( blit == nil )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"fill",queue->owner,0,"cannot create a Metal blit encoder");
		else
		{
			[blit fillBuffer:(__bridge id<MTLBuffer>)target->buffer range:NSMakeRange((NSUInteger)(target->offset + offset),(NSUInteger)bytes) value:value];
			[blit endEncoding];
			status = MetalCounted(metal,queue->submitted,"fill");
		}
	}
	pthread_mutex_unlock(&metal->encode);
	return(status);
}

SparkStatus SparkDeviceBackendQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	MetalMark mark;
	SparkStatus status;
	memset(&mark,0,sizeof(mark));
	mark.kind = METAL_MARK_CALLBACK;
	mark.function = function;
	mark.context = context;
	pthread_mutex_lock(&metal->encode);
	@autoreleasepool
	{
		if ( MetalOpen(metal) == nil )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"host_callback",queue->owner,0,"cannot create a Metal command buffer");
		else
		{
			mark.generation = ++metal->gate_target;
			metal->open_sequence = queue->submitted;
			status = MetalCommit(metal,&mark,"host_callback");
		}
	}
	pthread_mutex_unlock(&metal->encode);
	return(status);
}

SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	struct timespec deadline;
	SparkStatus status;
	int32_t code;
	pthread_mutex_lock(&metal->encode);
	@autoreleasepool
	{
		status = MetalCommit(metal,0,"queue_wait");
	}
	pthread_mutex_unlock(&metal->encode);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( timeout_ns != 0u && timeout_ns != SPARK_DEVICE_WAIT_FOREVER )
		MetalDeadline(timeout_ns,&deadline);
	pthread_mutex_lock(&metal->mutex);
	while ( metal->completed < sequence && metal->error_code == 0 )
	{
		if ( timeout_ns == 0u )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
		if ( timeout_ns == SPARK_DEVICE_WAIT_FOREVER )
			pthread_cond_wait(&metal->progress,&metal->mutex);
		else if ( pthread_cond_timedwait(&metal->progress,&metal->mutex,&deadline) == ETIMEDOUT && metal->completed < sequence )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
	}
	code = metal->error_code;
	pthread_mutex_unlock(&metal->mutex);
	if ( code != 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"queue_wait",queue->owner,code,metal->error));
	return(status);
}

uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	uint64_t completed;
	pthread_mutex_lock(&metal->mutex);
	completed = metal->completed;
	pthread_mutex_unlock(&metal->mutex);
	return(completed);
}

SparkStatus SparkDeviceBackendEventCreate(SparkDeviceEvent *event)
{
	MetalEvent *metal = (MetalEvent *)calloc(1u,sizeof(*metal));
	if ( metal == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"event_create","",ENOMEM,"out of host memory"));
	@autoreleasepool
	{
		id<MTLSharedEvent> shared = [metal_device newSharedEvent];
		if ( shared == nil )
		{
			free(metal);
			return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"event_create","",0,"cannot create a Metal shared event"));
		}
		metal->shared = (__bridge_retained void *)shared;
	}
	pthread_mutex_init(&metal->mutex,0);
	pthread_cond_init(&metal->done,0);
	event->backend = metal;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendEventDestroy(SparkDeviceEvent *event)
{
	MetalEvent *metal = (MetalEvent *)event->backend;
	if ( metal == 0 )
		return;
	pthread_cond_destroy(&metal->done);
	pthread_mutex_destroy(&metal->mutex);
	MetalReleaseObject(&metal->shared);
	free(metal);
	event->backend = 0;
}

SparkStatus SparkDeviceBackendEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue)
{
	MetalEvent *metal_event = (MetalEvent *)event->backend;
	MetalQueue *metal = (MetalQueue *)queue->backend;
	MetalMark mark;
	SparkStatus status;
	memset(&mark,0,sizeof(mark));
	mark.kind = METAL_MARK_EVENT;
	mark.event = metal_event;
	pthread_mutex_lock(&metal->encode);
	pthread_mutex_lock(&metal_event->mutex);
	mark.generation = ++metal_event->recorded;
	pthread_mutex_unlock(&metal_event->mutex);
	metal->open_sequence = queue->submitted;
	@autoreleasepool
	{
		status = MetalCommit(metal,&mark,"event_record");
	}
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&metal_event->mutex);
		metal_event->recorded--;
		pthread_mutex_unlock(&metal_event->mutex);
	}
	pthread_mutex_unlock(&metal->encode);
	return(status);
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	MetalEvent *metal_event = (MetalEvent *)event->backend;
	MetalQueue *metal = (MetalQueue *)queue->backend;
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t generation;
	pthread_mutex_lock(&metal_event->mutex);
	generation = metal_event->recorded;
	pthread_mutex_unlock(&metal_event->mutex);
	pthread_mutex_lock(&metal->encode);
	@autoreleasepool
	{
		id<MTLCommandBuffer> buffer = MetalOpen(metal);
		if ( buffer == nil )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_wait_event",queue->owner,0,"cannot create a Metal command buffer");
		else
		{
			MetalEndCompute(metal);
			[buffer encodeWaitForEvent:(__bridge id<MTLSharedEvent>)metal_event->shared value:generation];
			status = MetalCounted(metal,queue->submitted,"queue_wait_event");
		}
	}
	pthread_mutex_unlock(&metal->encode);
	return(status);
}

SparkStatus SparkDeviceBackendEventWait(SparkDeviceEvent *event, uint64_t timeout_ns)
{
	MetalEvent *metal = (MetalEvent *)event->backend;
	struct timespec deadline;
	SparkStatus status = SPARK_STATUS_OK;
	if ( timeout_ns != 0u && timeout_ns != SPARK_DEVICE_WAIT_FOREVER )
		MetalDeadline(timeout_ns,&deadline);
	pthread_mutex_lock(&metal->mutex);
	while ( metal->completed < metal->recorded )
	{
		if ( timeout_ns == 0u )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
		if ( timeout_ns == SPARK_DEVICE_WAIT_FOREVER )
			pthread_cond_wait(&metal->done,&metal->mutex);
		else if ( pthread_cond_timedwait(&metal->done,&metal->mutex,&deadline) == ETIMEDOUT && metal->completed < metal->recorded )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
	}
	pthread_mutex_unlock(&metal->mutex);
	return(status);
}

SparkStatus SparkDeviceBackendLibraryLoad(SparkDeviceLibrary *library, const char *path)
{
	@autoreleasepool
	{
		NSError *error = nil;
		NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
		id<MTLLibrary> loaded = url != nil ? [metal_device newLibraryWithURL:url error:&error] : nil;
		if ( loaded == nil )
			return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"library_load",library->owner,error != nil ? (int32_t)error.code : 0,
				error != nil ? error.localizedDescription.UTF8String : "cannot read the Metal library"));
		library->backend = (__bridge_retained void *)loaded;
	}
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendLibraryUnload(SparkDeviceLibrary *library)
{
	MetalReleaseObject(&library->backend);
}

SparkStatus SparkDeviceBackendKernelCreate(SparkDeviceKernel *kernel, const char *name)
{
	@autoreleasepool
	{
		NSError *error = nil;
		id<MTLLibrary> library = (__bridge id<MTLLibrary>)kernel->library->backend;
		id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
		id<MTLComputePipelineState> pipeline;
		if ( function == nil )
			return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"kernel_create",name,0,"the Metal library has no function of that name"));
		pipeline = [metal_device newComputePipelineStateWithFunction:function error:&error];
		if ( pipeline == nil )
			return(SparkDeviceRecordError(SPARK_STATUS_TARGET_MISMATCH,"kernel_create",name,error != nil ? (int32_t)error.code : 0,
				error != nil ? error.localizedDescription.UTF8String : "Metal cannot build the compute pipeline"));
		kernel->info.max_threads = (uint32_t)pipeline.maxTotalThreadsPerThreadgroup;
		kernel->info.simd_width = (uint32_t)pipeline.threadExecutionWidth;
		kernel->info.static_shared_bytes = (uint32_t)pipeline.staticThreadgroupMemoryLength;
		kernel->backend = (__bridge_retained void *)pipeline;
	}
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendKernelDestroy(SparkDeviceKernel *kernel)
{
	MetalReleaseObject(&kernel->backend);
}

SparkStatus SparkDeviceBackendKernelLaunch(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch,
	SparkDeviceAllocation *const *allocations)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	pthread_mutex_lock(&metal->encode);
	@autoreleasepool
	{
		id<MTLCommandBuffer> buffer = MetalOpen(metal);
		id<MTLComputeCommandEncoder> encoder = nil;
		if ( buffer != nil && metal->compute == 0 )
		{
			encoder = [buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
			if ( encoder != nil )
				metal->compute = (__bridge_retained void *)encoder;
		}
		else if ( buffer != nil )
			encoder = (__bridge id<MTLComputeCommandEncoder>)metal->compute;
		if ( encoder == nil )
			status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"kernel_launch",kernel->name,0,"cannot create a Metal compute encoder");
		else
		{
			[encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)kernel->backend];
			if ( launch->constant_bytes != 0u )
				[encoder setBytes:launch->constants length:launch->constant_bytes atIndex:0];
			for ( index = 0u; index < launch->binding_count; index++ )
			{
				const MetalAllocation *bound = (const MetalAllocation *)allocations[index]->backend;
				[encoder setBuffer:(__bridge id<MTLBuffer>)bound->buffer offset:(NSUInteger)(bound->offset + launch->bindings[index].range.offset)
					atIndex:index + 1u];
			}
			if ( launch->shared_bytes != 0u )
				[encoder setThreadgroupMemoryLength:(launch->shared_bytes + 15u) / 16u * 16u atIndex:0];
			[encoder dispatchThreadgroups:MTLSizeMake(launch->groups.x,launch->groups.y,launch->groups.z)
				threadsPerThreadgroup:MTLSizeMake(launch->threads.x,launch->threads.y,launch->threads.z)];
			status = MetalCounted(metal,queue->submitted,"kernel_launch");
		}
	}
	pthread_mutex_unlock(&metal->encode);
	return(status);
}

SparkStatus SparkDeviceBackendEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds)
{
	MetalEvent *first = (MetalEvent *)start->backend, *second = (MetalEvent *)stop->backend;
	uint64_t begin, end;
	pthread_mutex_lock(&first->mutex);
	begin = first->timestamp_ns;
	pthread_mutex_unlock(&first->mutex);
	pthread_mutex_lock(&second->mutex);
	end = second->timestamp_ns;
	pthread_mutex_unlock(&second->mutex);
	if ( end < begin )
		return(SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"event_elapsed","",0,"the stop event completed before the start event"));
	*nanoseconds = end - begin;
	return(SPARK_STATUS_OK);
}
