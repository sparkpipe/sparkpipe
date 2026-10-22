#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "runtime/device/device_backend.h"

typedef enum HostMemoryKind
{
	HOST_MEMORY_ANONYMOUS = 1,
	HOST_MEMORY_HEAP = 2,
	HOST_MEMORY_BORROWED = 3,
	HOST_MEMORY_FILE = 4
} HostMemoryKind;

typedef struct HostAllocation
{
	HostMemoryKind kind;
	uint8_t *data;
	void *mapping;
	uint64_t mapping_bytes;
	void *reservation;
	uint64_t reservation_bytes;
} HostAllocation;

typedef enum HostOperationKind
{
	HOST_OPERATION_COPY = 1,
	HOST_OPERATION_FILL = 2,
	HOST_OPERATION_CALLBACK = 3,
	HOST_OPERATION_RECORD = 4,
	HOST_OPERATION_WAIT = 5
} HostOperationKind;

typedef struct HostEvent
{
	pthread_mutex_t mutex;
	pthread_cond_t done;
	uint64_t recorded;
	uint64_t completed;
	uint64_t timestamp_ns;
} HostEvent;

typedef struct HostOperation
{
	HostOperationKind kind;
	uint64_t sequence;
	uint8_t *destination;
	const uint8_t *source;
	uint64_t bytes;
	uint8_t value;
	SparkDeviceHostFunction function;
	void *context;
	HostEvent *event;
	uint64_t generation;
	struct HostOperation *next;
} HostOperation;

typedef struct HostQueue
{
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t wake;
	pthread_cond_t progress;
	HostOperation *head;
	HostOperation *tail;
	uint64_t completed;
	uint32_t stop;
} HostQueue;

static uint64_t host_page_bytes;

static uint64_t HostNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static void HostDeadline(uint64_t timeout_ns, struct timespec *deadline)
{
	uint64_t nanoseconds;
	clock_gettime(CLOCK_REALTIME,deadline);
	if ( timeout_ns > 3153600000000000000ull )
		timeout_ns = 3153600000000000000ull;
	nanoseconds = (uint64_t)deadline->tv_nsec + timeout_ns % 1000000000ull;
	deadline->tv_sec += (time_t)(timeout_ns / 1000000000ull + nanoseconds / 1000000000ull);
	deadline->tv_nsec = (long)(nanoseconds % 1000000000ull);
}

static uint64_t HostRound(uint64_t bytes)
{
	return((bytes + host_page_bytes - 1u) / host_page_bytes * host_page_bytes);
}

SparkStatus SparkDeviceBackendInitialize(const SparkDeviceConfiguration *configuration, SparkDeviceProperties *properties)
{
	long page = sysconf(_SC_PAGESIZE);
	long processors = sysconf(_SC_NPROCESSORS_ONLN);
	long pages = sysconf(_SC_PHYS_PAGES);
	if ( configuration->ordinal != 0u )
		return(SparkDeviceRecordError(SPARK_STATUS_NOT_FOUND,"initialize","",0,"the host backend has one device, ordinal 0"));
	host_page_bytes = page > 0 ? (uint64_t)page : 4096u;
	snprintf(properties->backend,sizeof(properties->backend),"%s","host");
	snprintf(properties->target,sizeof(properties->target),"%s","host");
	snprintf(properties->name,sizeof(properties->name),"%s","host queues");
	properties->page_bytes = (uint32_t)host_page_bytes;
	properties->compute_units = processors > 0 ? (uint32_t)processors : 1u;
	properties->simd_width = 32u;
	properties->memory_budget_bytes = configuration->memory_budget_bytes != 0u ? configuration->memory_budget_bytes :
		(pages > 0 ? (uint64_t)pages * host_page_bytes : 0u);
	properties->max_buffer_bytes = 1ull << 44;
	properties->properties = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendShutdown(void)
{
	return(SPARK_STATUS_OK);
}

static SparkStatus HostReserve(SparkDeviceAllocation *allocation, HostAllocation *host, const char *operation)
{
	host->reservation_bytes = HostRound(allocation->bytes);
	host->reservation = mmap(0,(size_t)host->reservation_bytes,PROT_NONE,MAP_PRIVATE | MAP_ANON,-1,0);
	if ( host->reservation == MAP_FAILED )
	{
		host->reservation = 0;
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,operation,allocation->owner,errno,"cannot reserve the device address range"));
	}
	allocation->device = (SparkDeviceAddress)(uintptr_t)host->reservation;
	return(SPARK_STATUS_OK);
}

static void HostReleaseRecord(HostAllocation *host)
{
	if ( host->reservation != 0 )
		munmap(host->reservation,(size_t)host->reservation_bytes);
	if ( host->kind == HOST_MEMORY_ANONYMOUS || host->kind == HOST_MEMORY_FILE )
	{
		if ( host->mapping != 0 )
			munmap(host->mapping,(size_t)host->mapping_bytes);
	}
	else if ( host->kind == HOST_MEMORY_HEAP )
		free(host->data);
	free(host);
}

SparkStatus SparkDeviceBackendAllocate(SparkDeviceAllocation *allocation)
{
	HostAllocation *host = (HostAllocation *)calloc(1u,sizeof(*host));
	SparkStatus status;
	if ( host == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,ENOMEM,"out of host memory"));
	if ( allocation->space == SPARK_DEVICE_SPACE_HOST_PAGEABLE )
	{
		host->kind = HOST_MEMORY_HEAP;
		host->data = (uint8_t *)calloc(1u,(size_t)allocation->bytes);
		if ( host->data == 0 )
		{
			HostReleaseRecord(host);
			return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,ENOMEM,"out of host memory"));
		}
		allocation->host = host->data;
		allocation->device = 0u;
		allocation->backend = host;
		return(SPARK_STATUS_OK);
	}
	host->kind = HOST_MEMORY_ANONYMOUS;
	host->mapping_bytes = HostRound(allocation->bytes);
	host->mapping = mmap(0,(size_t)host->mapping_bytes,PROT_READ | PROT_WRITE,MAP_PRIVATE | MAP_ANON,-1,0);
	if ( host->mapping == MAP_FAILED )
	{
		host->mapping = 0;
		HostReleaseRecord(host);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",allocation->owner,errno,"out of host memory"));
	}
	host->data = (uint8_t *)host->mapping;
	status = HostReserve(allocation,host,"allocate");
	if ( status != SPARK_STATUS_OK )
	{
		HostReleaseRecord(host);
		return(status);
	}
	allocation->host = allocation->space == SPARK_DEVICE_SPACE_DEVICE_PRIVATE ? 0 : host->data;
	allocation->backend = host;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendRegister(SparkDeviceAllocation *allocation, void *memory)
{
	HostAllocation *host = (HostAllocation *)calloc(1u,sizeof(*host));
	SparkStatus status;
	if ( host == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"register",allocation->owner,ENOMEM,"out of host memory"));
	host->kind = HOST_MEMORY_BORROWED;
	host->data = (uint8_t *)memory;
	status = HostReserve(allocation,host,"register");
	if ( status != SPARK_STATUS_OK )
	{
		HostReleaseRecord(host);
		return(status);
	}
	allocation->host = memory;
	allocation->backend = host;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendMapFile(SparkDeviceAllocation *allocation, int fd, uint64_t offset)
{
	HostAllocation *host = (HostAllocation *)calloc(1u,sizeof(*host));
	uint64_t aligned = offset / host_page_bytes * host_page_bytes;
	SparkStatus status;
	if ( host == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"map_file",allocation->owner,ENOMEM,"out of host memory"));
	host->kind = HOST_MEMORY_FILE;
	host->mapping_bytes = HostRound(offset - aligned + allocation->bytes);
	host->mapping = mmap(0,(size_t)host->mapping_bytes,PROT_READ,MAP_SHARED,fd,(off_t)aligned);
	if ( host->mapping == MAP_FAILED )
	{
		int error = errno;
		host->mapping = 0;
		HostReleaseRecord(host);
		return(SparkDeviceRecordError(error == ENOMEM ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_INVALID_ARGUMENT,
			"map_file",allocation->owner,error,"cannot map the file range"));
	}
	host->data = (uint8_t *)host->mapping + (offset - aligned);
	status = HostReserve(allocation,host,"map_file");
	if ( status != SPARK_STATUS_OK )
	{
		HostReleaseRecord(host);
		return(status);
	}
	allocation->host = host->data;
	allocation->backend = host;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendRelease(SparkDeviceAllocation *allocation)
{
	if ( allocation->backend != 0 )
		HostReleaseRecord((HostAllocation *)allocation->backend);
	allocation->backend = 0;
}

static void HostRun(HostOperation *operation)
{
	if ( operation->kind == HOST_OPERATION_COPY )
		memcpy(operation->destination,operation->source,(size_t)operation->bytes);
	else if ( operation->kind == HOST_OPERATION_FILL )
		memset(operation->destination,operation->value,(size_t)operation->bytes);
	else if ( operation->kind == HOST_OPERATION_CALLBACK )
		operation->function(operation->context);
	else if ( operation->kind == HOST_OPERATION_RECORD )
	{
		pthread_mutex_lock(&operation->event->mutex);
		if ( operation->event->completed < operation->generation )
		{
			operation->event->completed = operation->generation;
			operation->event->timestamp_ns = HostNowNs();
		}
		pthread_cond_broadcast(&operation->event->done);
		pthread_mutex_unlock(&operation->event->mutex);
	}
	else if ( operation->kind == HOST_OPERATION_WAIT )
	{
		pthread_mutex_lock(&operation->event->mutex);
		while ( operation->event->completed < operation->generation )
			pthread_cond_wait(&operation->event->done,&operation->event->mutex);
		pthread_mutex_unlock(&operation->event->mutex);
	}
}

static void *HostWorker(void *argument)
{
	HostQueue *queue = (HostQueue *)argument;
	pthread_mutex_lock(&queue->mutex);
	for ( ;; )
	{
		HostOperation *operation;
		while ( queue->head == 0 && queue->stop == 0u )
			pthread_cond_wait(&queue->wake,&queue->mutex);
		if ( queue->head == 0 )
			break;
		operation = queue->head;
		queue->head = operation->next;
		if ( queue->head == 0 )
			queue->tail = 0;
		pthread_mutex_unlock(&queue->mutex);
		HostRun(operation);
		pthread_mutex_lock(&queue->mutex);
		queue->completed = operation->sequence;
		pthread_cond_broadcast(&queue->progress);
		free(operation);
	}
	pthread_mutex_unlock(&queue->mutex);
	return(0);
}

SparkStatus SparkDeviceBackendQueueCreate(SparkDeviceQueue *queue)
{
	HostQueue *host = (HostQueue *)calloc(1u,sizeof(*host));
	int error;
	if ( host == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",queue->owner,ENOMEM,"out of host memory"));
	pthread_mutex_init(&host->mutex,0);
	pthread_cond_init(&host->wake,0);
	pthread_cond_init(&host->progress,0);
	error = pthread_create(&host->thread,0,HostWorker,host);
	if ( error != 0 )
	{
		pthread_cond_destroy(&host->progress);
		pthread_cond_destroy(&host->wake);
		pthread_mutex_destroy(&host->mutex);
		free(host);
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",queue->owner,error,"cannot start the queue worker"));
	}
	queue->backend = host;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendQueueDestroy(SparkDeviceQueue *queue)
{
	HostQueue *host = (HostQueue *)queue->backend;
	if ( host == 0 )
		return;
	pthread_mutex_lock(&host->mutex);
	host->stop = 1u;
	pthread_cond_broadcast(&host->wake);
	pthread_mutex_unlock(&host->mutex);
	pthread_join(host->thread,0);
	pthread_cond_destroy(&host->progress);
	pthread_cond_destroy(&host->wake);
	pthread_mutex_destroy(&host->mutex);
	free(host);
	queue->backend = 0;
}

static SparkStatus HostEnqueue(SparkDeviceQueue *queue, HostOperation *operation, const char *name)
{
	HostQueue *host = (HostQueue *)queue->backend;
	HostOperation *copy = (HostOperation *)malloc(sizeof(*copy));
	if ( copy == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,name,queue->owner,ENOMEM,"out of host memory"));
	*copy = *operation;
	copy->sequence = queue->submitted;
	copy->next = 0;
	pthread_mutex_lock(&host->mutex);
	if ( host->tail != 0 )
		host->tail->next = copy;
	else
		host->head = copy;
	host->tail = copy;
	pthread_cond_signal(&host->wake);
	pthread_mutex_unlock(&host->mutex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceBackendCopy(SparkDeviceQueue *queue, SparkDeviceDirection direction,
	const SparkDeviceAllocation *destination, uint64_t destination_offset,
	const SparkDeviceAllocation *source, uint64_t source_offset, uint64_t bytes)
{
	HostOperation operation;
	(void)direction;
	memset(&operation,0,sizeof(operation));
	operation.kind = HOST_OPERATION_COPY;
	operation.destination = ((HostAllocation *)destination->backend)->data + destination_offset;
	operation.source = ((HostAllocation *)source->backend)->data + source_offset;
	operation.bytes = bytes;
	return(HostEnqueue(queue,&operation,"copy"));
}

SparkStatus SparkDeviceBackendFill(SparkDeviceQueue *queue, const SparkDeviceAllocation *destination, uint64_t offset, uint8_t value, uint64_t bytes)
{
	HostOperation operation;
	memset(&operation,0,sizeof(operation));
	operation.kind = HOST_OPERATION_FILL;
	operation.destination = ((HostAllocation *)destination->backend)->data + offset;
	operation.value = value;
	operation.bytes = bytes;
	return(HostEnqueue(queue,&operation,"fill"));
}

SparkStatus SparkDeviceBackendQueueHostCallback(SparkDeviceQueue *queue, SparkDeviceHostFunction function, void *context)
{
	HostOperation operation;
	memset(&operation,0,sizeof(operation));
	operation.kind = HOST_OPERATION_CALLBACK;
	operation.function = function;
	operation.context = context;
	return(HostEnqueue(queue,&operation,"host_callback"));
}

SparkStatus SparkDeviceBackendQueueWait(SparkDeviceQueue *queue, uint64_t sequence, uint64_t timeout_ns)
{
	HostQueue *host = (HostQueue *)queue->backend;
	struct timespec deadline;
	SparkStatus status = SPARK_STATUS_OK;
	if ( timeout_ns != 0u && timeout_ns != SPARK_DEVICE_WAIT_FOREVER )
		HostDeadline(timeout_ns,&deadline);
	pthread_mutex_lock(&host->mutex);
	while ( host->completed < sequence )
	{
		if ( timeout_ns == 0u )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
		if ( timeout_ns == SPARK_DEVICE_WAIT_FOREVER )
			pthread_cond_wait(&host->progress,&host->mutex);
		else if ( pthread_cond_timedwait(&host->progress,&host->mutex,&deadline) == ETIMEDOUT && host->completed < sequence )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
	}
	pthread_mutex_unlock(&host->mutex);
	return(status);
}

uint64_t SparkDeviceBackendQueueCompleted(SparkDeviceQueue *queue)
{
	HostQueue *host = (HostQueue *)queue->backend;
	uint64_t completed;
	pthread_mutex_lock(&host->mutex);
	completed = host->completed;
	pthread_mutex_unlock(&host->mutex);
	return(completed);
}

SparkStatus SparkDeviceBackendEventCreate(SparkDeviceEvent *event)
{
	HostEvent *host = (HostEvent *)calloc(1u,sizeof(*host));
	if ( host == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"event_create","",ENOMEM,"out of host memory"));
	pthread_mutex_init(&host->mutex,0);
	pthread_cond_init(&host->done,0);
	event->backend = host;
	return(SPARK_STATUS_OK);
}

void SparkDeviceBackendEventDestroy(SparkDeviceEvent *event)
{
	HostEvent *host = (HostEvent *)event->backend;
	if ( host == 0 )
		return;
	pthread_cond_destroy(&host->done);
	pthread_mutex_destroy(&host->mutex);
	free(host);
	event->backend = 0;
}

SparkStatus SparkDeviceBackendEventRecord(SparkDeviceEvent *event, SparkDeviceQueue *queue)
{
	HostEvent *host = (HostEvent *)event->backend;
	HostOperation operation;
	SparkStatus status;
	memset(&operation,0,sizeof(operation));
	operation.kind = HOST_OPERATION_RECORD;
	operation.event = host;
	pthread_mutex_lock(&host->mutex);
	operation.generation = ++host->recorded;
	pthread_mutex_unlock(&host->mutex);
	status = HostEnqueue(queue,&operation,"event_record");
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&host->mutex);
		host->recorded--;
		pthread_mutex_unlock(&host->mutex);
	}
	return(status);
}

SparkStatus SparkDeviceBackendQueueWaitEvent(SparkDeviceQueue *queue, SparkDeviceEvent *event)
{
	HostEvent *host = (HostEvent *)event->backend;
	HostOperation operation;
	memset(&operation,0,sizeof(operation));
	operation.kind = HOST_OPERATION_WAIT;
	operation.event = host;
	pthread_mutex_lock(&host->mutex);
	operation.generation = host->recorded;
	pthread_mutex_unlock(&host->mutex);
	return(HostEnqueue(queue,&operation,"queue_wait_event"));
}

SparkStatus SparkDeviceBackendEventWait(SparkDeviceEvent *event, uint64_t timeout_ns)
{
	HostEvent *host = (HostEvent *)event->backend;
	struct timespec deadline;
	SparkStatus status = SPARK_STATUS_OK;
	if ( timeout_ns != 0u && timeout_ns != SPARK_DEVICE_WAIT_FOREVER )
		HostDeadline(timeout_ns,&deadline);
	pthread_mutex_lock(&host->mutex);
	while ( host->completed < host->recorded )
	{
		if ( timeout_ns == 0u )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
		if ( timeout_ns == SPARK_DEVICE_WAIT_FOREVER )
			pthread_cond_wait(&host->done,&host->mutex);
		else if ( pthread_cond_timedwait(&host->done,&host->mutex,&deadline) == ETIMEDOUT && host->completed < host->recorded )
		{
			status = SPARK_STATUS_PENDING;
			break;
		}
	}
	pthread_mutex_unlock(&host->mutex);
	return(status);
}

SparkStatus SparkDeviceBackendEventElapsed(SparkDeviceEvent *start, SparkDeviceEvent *stop, uint64_t *nanoseconds)
{
	HostEvent *first = (HostEvent *)start->backend, *second = (HostEvent *)stop->backend;
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
