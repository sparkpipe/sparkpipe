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

typedef enum MetalOperationKind
{
	METAL_OPERATION_COPY = 1,
	METAL_OPERATION_FILL = 2,
	METAL_OPERATION_CALLBACK = 3,
	METAL_OPERATION_RECORD = 4,
	METAL_OPERATION_WAIT = 5
} MetalOperationKind;

typedef struct MetalEvent
{
	pthread_mutex_t mutex;
	pthread_cond_t done;
	uint64_t recorded;
	uint64_t completed;
	uint64_t timestamp_ns;
} MetalEvent;

typedef struct MetalOperation
{
	MetalOperationKind kind;
	uint64_t sequence;
	void *destination;
	uint64_t destination_offset;
	void *source;
	uint64_t source_offset;
	uint64_t bytes;
	uint8_t value;
	SparkDeviceHostFunction function;
	void *context;
	MetalEvent *event;
	uint64_t generation;
	struct MetalOperation *next;
} MetalOperation;

typedef struct MetalQueue
{
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t wake;
	pthread_cond_t progress;
	MetalOperation *head;
	MetalOperation *tail;
	uint64_t completed;
	uint32_t stop;
	void *commands;
	char owner[SPARK_DEVICE_OWNER_BYTES];
} MetalQueue;

static id<MTLDevice> metal_device;
static id<MTLCommandQueue> metal_setup_queue;
static uint64_t metal_page_bytes;

static uint64_t MetalNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
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
	uint32_t cores = 0u;
	size_t size = sizeof(cores);
	if ( sysctlbyname("hw.perflevel0.gpu_cores",&cores,&size,0,0) == 0 && cores != 0u )
		return(cores);
	return(0u);
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
		memcpy(properties->backend,"metal",6u);
		snprintf(properties->target,sizeof(properties->target),"metal.%s",[metal_device supportsFamily:MTLGPUFamilyApple9] ? "apple9" : "apple");
		snprintf(properties->name,sizeof(properties->name),"%s",metal_device.name.UTF8String);
		properties->page_bytes = (uint32_t)metal_page_bytes;
		properties->compute_units = MetalGpuCores();
		properties->simd_width = 32u;
		properties->memory_budget_bytes = configuration->memory_budget_bytes != 0u ? configuration->memory_budget_bytes :
			(uint64_t)metal_device.recommendedMaxWorkingSetSize;
		properties->max_buffer_bytes = (uint64_t)metal_device.maxBufferLength;
		properties->properties = metal_device.hasUnifiedMemory ? SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY : 0u;
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

static void MetalRunBlit(MetalQueue *queue, MetalOperation *operation)
{
	@autoreleasepool
	{
		id<MTLCommandQueue> commands_queue = (__bridge id<MTLCommandQueue>)queue->commands;
		id<MTLCommandBuffer> commands = [commands_queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
		id<MTLBuffer> destination = (__bridge id<MTLBuffer>)operation->destination;
		if ( operation->kind == METAL_OPERATION_COPY )
			[blit copyFromBuffer:(__bridge id<MTLBuffer>)operation->source sourceOffset:(NSUInteger)operation->source_offset
				toBuffer:destination destinationOffset:(NSUInteger)operation->destination_offset size:(NSUInteger)operation->bytes];
		else
			[blit fillBuffer:destination range:NSMakeRange((NSUInteger)operation->destination_offset,(NSUInteger)operation->bytes) value:operation->value];
		[blit endEncoding];
		[commands commit];
		[commands waitUntilCompleted];
		if ( commands.status != MTLCommandBufferStatusCompleted )
			SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,operation->kind == METAL_OPERATION_COPY ? "copy" : "fill",queue->owner,(int32_t)commands.status,
				commands.error != nil ? commands.error.localizedDescription.UTF8String : "the Metal command buffer failed");
	}
}

static void MetalRun(MetalQueue *queue, MetalOperation *operation)
{
	if ( operation->kind == METAL_OPERATION_COPY || operation->kind == METAL_OPERATION_FILL )
		MetalRunBlit(queue,operation);
	else if ( operation->kind == METAL_OPERATION_CALLBACK )
		operation->function(operation->context);
	else if ( operation->kind == METAL_OPERATION_RECORD )
	{
		pthread_mutex_lock(&operation->event->mutex);
		if ( operation->event->completed < operation->generation )
		{
			operation->event->completed = operation->generation;
			operation->event->timestamp_ns = MetalNowNs();
		}
		pthread_cond_broadcast(&operation->event->done);
		pthread_mutex_unlock(&operation->event->mutex);
	}
	else if ( operation->kind == METAL_OPERATION_WAIT )
	{
		pthread_mutex_lock(&operation->event->mutex);
		while ( operation->event->completed < operation->generation )
			pthread_cond_wait(&operation->event->done,&operation->event->mutex);
		pthread_mutex_unlock(&operation->event->mutex);
	}
}

static void MetalReleaseOperation(MetalOperation *operation)
{
	if ( operation->destination != 0 )
		CFRelease(operation->destination);
	if ( operation->source != 0 )
		CFRelease(operation->source);
	free(operation);
}

static void *MetalWorker(void *argument)
{
	MetalQueue *queue = (MetalQueue *)argument;
	pthread_mutex_lock(&queue->mutex);
	for ( ;; )
	{
		MetalOperation *operation;
		while ( queue->head == 0 && queue->stop == 0u )
			pthread_cond_wait(&queue->wake,&queue->mutex);
		if ( queue->head == 0 )
			break;
		operation = queue->head;
		queue->head = operation->next;
		if ( queue->head == 0 )
			queue->tail = 0;
		pthread_mutex_unlock(&queue->mutex);
		MetalRun(queue,operation);
		pthread_mutex_lock(&queue->mutex);
		queue->completed = operation->sequence;
		pthread_cond_broadcast(&queue->progress);
		MetalReleaseOperation(operation);
	}
	pthread_mutex_unlock(&queue->mutex);
	return(0);
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
		if ( commands == nil )
		{
			free(metal);
			return(SparkDeviceRecordError(SPARK_STATUS_IO_ERROR,"queue_create",queue->owner,0,"cannot create a Metal command queue"));
		}
		metal->commands = (__bridge_retained void *)commands;
	}
	memcpy(metal->owner,queue->owner,sizeof(metal->owner));
	pthread_mutex_init(&metal->mutex,0);
	pthread_cond_init(&metal->wake,0);
	pthread_cond_init(&metal->progress,0);
	error = pthread_create(&metal->thread,0,MetalWorker,metal);
	if ( error != 0 )
	{
		pthread_cond_destroy(&metal->progress);
		pthread_cond_destroy(&metal->wake);
		pthread_mutex_destroy(&metal->mutex);
		CFRelease(metal->commands);
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
	pthread_cond_destroy(&metal->progress);
	pthread_cond_destroy(&metal->wake);
	pthread_mutex_destroy(&metal->mutex);
	CFRelease(metal->commands);
	free(metal);
	queue->backend = 0;
}

static SparkStatus MetalEnqueue(SparkDeviceQueue *queue, MetalOperation *operation, const char *name)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	MetalOperation *copy = (MetalOperation *)malloc(sizeof(*copy));
	if ( copy == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,name,queue->owner,ENOMEM,"out of host memory"));
	*copy = *operation;
	if ( copy->destination != 0 )
		CFRetain(copy->destination);
	if ( copy->source != 0 )
		CFRetain(copy->source);
	copy->sequence = queue->submitted;
	copy->next = 0;
	pthread_mutex_lock(&metal->mutex);
	if ( metal->tail != 0 )
		metal->tail->next = copy;
	else
		metal->head = copy;
	metal->tail = copy;
	pthread_cond_signal(&metal->wake);
	pthread_mutex_unlock(&metal->mutex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	const MetalAllocation *target = (const MetalAllocation *)destination->backend, *origin = (const MetalAllocation *)source->backend;
	MetalOperation operation;
	(void)direction;
	memset(&operation,0,sizeof(operation));
	operation.kind = METAL_OPERATION_COPY;
	operation.destination = target->buffer;
	operation.destination_offset = target->offset + destination_offset;
	operation.source = origin->buffer;
	operation.source_offset = origin->offset + source_offset;
	operation.bytes = bytes;
	return(MetalEnqueue(queue,&operation,"copy"));
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	const MetalAllocation *target = (const MetalAllocation *)destination->backend;
	MetalOperation operation;
	memset(&operation,0,sizeof(operation));
	operation.kind = METAL_OPERATION_FILL;
	operation.destination = target->buffer;
	operation.destination_offset = target->offset + offset;
	operation.value = value;
	operation.bytes = bytes;
	return(MetalEnqueue(queue,&operation,"fill"));
}

SparkStatus SparkDeviceBackendQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context)
{
	MetalOperation operation;
	memset(&operation,0,sizeof(operation));
	operation.kind = METAL_OPERATION_CALLBACK;
	operation.function = function;
	operation.context = context;
	return(MetalEnqueue(queue,&operation,"host_callback"));
}

SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns)
{
	MetalQueue *metal = (MetalQueue *)queue->backend;
	struct timespec deadline;
	SparkStatus status = SPARK_STATUS_OK;
	if ( timeout_ns != 0u && timeout_ns != SPARK_DEVICE_WAIT_FOREVER )
		MetalDeadline(timeout_ns,&deadline);
	pthread_mutex_lock(&metal->mutex);
	while ( metal->completed < sequence )
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
	pthread_mutex_unlock(&metal->mutex);
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
	free(metal);
	event->backend = 0;
}

SparkStatus SparkDeviceBackendEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue)
{
	MetalEvent *metal = (MetalEvent *)event->backend;
	MetalOperation operation;
	SparkStatus status;
	memset(&operation,0,sizeof(operation));
	operation.kind = METAL_OPERATION_RECORD;
	operation.event = metal;
	pthread_mutex_lock(&metal->mutex);
	operation.generation = ++metal->recorded;
	pthread_mutex_unlock(&metal->mutex);
	status = MetalEnqueue(queue,&operation,"event_record");
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&metal->mutex);
		metal->recorded--;
		pthread_mutex_unlock(&metal->mutex);
	}
	return(status);
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	MetalEvent *metal = (MetalEvent *)event->backend;
	MetalOperation operation;
	memset(&operation,0,sizeof(operation));
	operation.kind = METAL_OPERATION_WAIT;
	operation.event = metal;
	pthread_mutex_lock(&metal->mutex);
	operation.generation = metal->recorded;
	pthread_mutex_unlock(&metal->mutex);
	return(MetalEnqueue(queue,&operation,"queue_wait_event"));
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
