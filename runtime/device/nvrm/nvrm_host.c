#include <errno.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_host.h"
#include "runtime/device/nvrm/nvrm_wait.h"

#define HOST_REACHED_OFFSET 0u
#define HOST_DONE_OFFSET 64u

#define HOST_TRY(call) \
	do { \
		const SparkStatus host_status = (call); \
		if ( host_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(host_status); \
	} while (0)

static volatile uint64_t *HostGate(const SparkNvrmHostWorker *worker, uint32_t offset)
{
	return((volatile uint64_t *)(void *)(worker->gates.cpu + offset));
}

static void HostRunItem(const SparkNvrmHostItem *item)
{
	if ( item->kind == SPARK_NVRM_HOST_COPY )
		memcpy(item->destination,item->source,(size_t)item->bytes);
	else if ( item->kind == SPARK_NVRM_HOST_FILL )
		memset(item->destination,item->byte,(size_t)item->bytes);
	else
		item->function(item->context);
}

static uint32_t HostReached(SparkNvrmHostWorker *worker, uint64_t ticket)
{
	SparkNvrmWait wait;
	uint32_t reached = 1u;
	SparkNvrmWaitBegin(&wait);
	while ( __atomic_load_n(HostGate(worker,HOST_REACHED_OFFSET),__ATOMIC_ACQUIRE) < ticket )
	{
		if ( __atomic_load_n(&worker->stopping,__ATOMIC_ACQUIRE) != 0u )
		{
			reached = 0u;
			break;
		}
		(void)SparkNvrmWaitPace(&wait,SPARK_DEVICE_WAIT_FOREVER);
	}
	SparkNvrmWaitEnd(&wait);
	return(reached);
}

static void *HostRun(void *argument)
{
	SparkNvrmHostWorker *worker = (SparkNvrmHostWorker *)argument;
	for ( ;; )
	{
		SparkNvrmHostItem item;
		pthread_mutex_lock(&worker->lock);
		while ( worker->count == 0u && worker->stopping == 0u )
			pthread_cond_wait(&worker->changed,&worker->lock);
		if ( worker->count == 0u )
		{
			pthread_mutex_unlock(&worker->lock);
			return(0);
		}
		item = worker->items[worker->head];
		pthread_mutex_unlock(&worker->lock);
		if ( HostReached(worker,item.ticket) == 0u )
			return(0);
		HostRunItem(&item);
		__atomic_store_n(HostGate(worker,HOST_DONE_OFFSET),item.ticket,__ATOMIC_RELEASE);
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		pthread_mutex_lock(&worker->lock);
		worker->head = (worker->head + 1u) % SPARK_NVRM_HOST_ITEMS;
		worker->count--;
		pthread_cond_broadcast(&worker->changed);
		pthread_mutex_unlock(&worker->lock);
	}
}

SparkStatus SparkNvrmHostOpen(SparkNvrmGpu *gpu, SparkNvrmHostWorker *worker, const char *owner)
{
	int error;
	memset(worker,0,sizeof(*worker));
	worker->gates.map_fd = -1;
	HOST_TRY(SparkNvrmMemoryAllocate(gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&worker->gates,owner));
	pthread_mutex_init(&worker->lock,0);
	pthread_cond_init(&worker->changed,0);
	error = pthread_create(&worker->thread,0,HostRun,worker);
	if ( error != 0 )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"queue_create",owner,error,"cannot start the queue's host worker");
		pthread_cond_destroy(&worker->changed);
		pthread_mutex_destroy(&worker->lock);
		SparkNvrmMemoryRelease(gpu,&worker->gates);
		SPARK_FAIL(status);
	}
	worker->started = 1u;
	return(SPARK_STATUS_OK);
}

void SparkNvrmHostClose(SparkNvrmGpu *gpu, SparkNvrmHostWorker *worker)
{
	if ( worker->started != 0u )
	{
		pthread_mutex_lock(&worker->lock);
		__atomic_store_n(&worker->stopping,1u,__ATOMIC_RELEASE);
		pthread_cond_broadcast(&worker->changed);
		pthread_mutex_unlock(&worker->lock);
		pthread_join(worker->thread,0);
		pthread_cond_destroy(&worker->changed);
		pthread_mutex_destroy(&worker->lock);
		worker->started = 0u;
	}
	if ( worker->gates.bytes != 0u )
		SparkNvrmMemoryRelease(gpu,&worker->gates);
}

static SparkStatus HostSubmit(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmHostWorker *worker, uint64_t ticket, uint64_t *value,
	const char *owner)
{
	SparkStatus status;
	HOST_TRY(SparkNvrmOpBegin(channel,SPARK_NVRM_ENGINE_HOST,owner));
	status = SparkNvrmSemaphoreRelease(channel,worker->gates.address + HOST_REACHED_OFFSET,ticket,0u,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmSemaphoreAcquire(channel,worker->gates.address + HOST_DONE_OFFSET,ticket,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmOpAbandon(channel);
		SPARK_FAIL(status);
	}
	return(SparkNvrmOpFinishHost(gpu,channel,value,owner));
}

static void HostRoom(SparkNvrmHostWorker *worker)
{
	pthread_mutex_lock(&worker->lock);
	while ( worker->count == SPARK_NVRM_HOST_ITEMS )
		pthread_cond_wait(&worker->changed,&worker->lock);
	pthread_mutex_unlock(&worker->lock);
}

void SparkNvrmHostQueue(SparkNvrmHostWorker *worker, const SparkNvrmHostItem *item)
{
	HostRoom(worker);
	pthread_mutex_lock(&worker->lock);
	worker->ticket++;
	worker->items[(worker->head + worker->count) % SPARK_NVRM_HOST_ITEMS] = *item;
	worker->items[(worker->head + worker->count) % SPARK_NVRM_HOST_ITEMS].ticket = worker->ticket;
	worker->count++;
	pthread_cond_broadcast(&worker->changed);
	pthread_mutex_unlock(&worker->lock);
}

uint64_t SparkNvrmHostGate(const SparkNvrmHostWorker *worker, uint32_t done)
{
	return(worker->gates.address + (done != 0u ? HOST_DONE_OFFSET : HOST_REACHED_OFFSET));
}

SparkStatus SparkNvrmHostWork(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkNvrmHostWorker *worker, const SparkNvrmHostItem *item, uint64_t *value,
	const char *owner)
{
	SparkStatus status;
	HostRoom(worker);
	status = HostSubmit(gpu,channel,worker,worker->ticket + 1u,value,owner);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	SparkNvrmHostQueue(worker,item);
	return(SPARK_STATUS_OK);
}
