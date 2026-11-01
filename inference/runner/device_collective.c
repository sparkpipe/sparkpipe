#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_collective.h"

typedef struct LocalCollective
{
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	uint32_t degree;
	uint32_t arrived;
	uint64_t generation;
	uint32_t aborted;
	SparkDeviceRange published[SPARK_DEVICE_COLLECTIVE_MAX_RANKS];
} LocalCollective;

static SparkStatus LocalBarrier(LocalCollective *local)
{
	SparkStatus status;
	uint64_t generation;
	pthread_mutex_lock(&local->mutex);
	generation = local->generation;
	if ( local->aborted == 0u && ++local->arrived == local->degree )
	{
		local->arrived = 0u;
		local->generation++;
		pthread_cond_broadcast(&local->cond);
	}
	else
		while ( local->aborted == 0u && local->generation == generation )
			pthread_cond_wait(&local->cond,&local->mutex);
	status = local->aborted != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
	pthread_mutex_unlock(&local->mutex);
	return(status);
}

static void LocalAbort(void *context)
{
	LocalCollective *local = (LocalCollective *)context;
	pthread_mutex_lock(&local->mutex);
	local->aborted = 1u;
	pthread_cond_broadcast(&local->cond);
	pthread_mutex_unlock(&local->mutex);
}

static SparkStatus LocalExchange(void *context, uint32_t rank, SparkDeviceQueue *queue, SparkDeviceRange source, uint64_t source_stride,
	uint64_t bytes, SparkDeviceRange lanes)
{
	LocalCollective *local = (LocalCollective *)context;
	SparkStatus status = rank < local->degree && bytes != 0u ? SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) : SPARK_STATUS_INVALID_ARGUMENT;
	uint32_t peer;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"device_collective: rank %u cannot enter the exchange (status %d)\n",rank,(int)status);
		LocalAbort(local);
		return(status);
	}
	local->published[rank] = source;
	status = LocalBarrier(local);
	for ( peer = 0u; status == SPARK_STATUS_OK && peer < local->degree; peer++ )
	{
		SparkDeviceRange from = local->published[peer], to = lanes;
		from.offset += (uint64_t)rank * source_stride;
		to.offset += (uint64_t)peer * bytes;
		status = SparkDeviceCopy(queue,to,from,bytes);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status != SPARK_STATUS_OK )
	{
		LocalAbort(local);
		return(status);
	}
	return(LocalBarrier(local));
}

SparkStatus SparkDeviceCollectiveLocalCreate(SparkDeviceCollective *collective, uint32_t degree)
{
	LocalCollective *local;
	memset(collective,0,sizeof(*collective));
	if ( degree < 2u || degree > SPARK_DEVICE_COLLECTIVE_MAX_RANKS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	local = (LocalCollective *)calloc(1u,sizeof(*local));
	if ( local == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	pthread_mutex_init(&local->mutex,0);
	pthread_cond_init(&local->cond,0);
	local->degree = degree;
	collective->context = local;
	collective->degree = degree;
	collective->exchange = LocalExchange;
	collective->abort = LocalAbort;
	return(SPARK_STATUS_OK);
}

void SparkDeviceCollectiveLocalDestroy(SparkDeviceCollective *collective)
{
	LocalCollective *local = (LocalCollective *)collective->context;
	if ( local != 0 )
	{
		pthread_cond_destroy(&local->cond);
		pthread_mutex_destroy(&local->mutex);
		free(local);
	}
	memset(collective,0,sizeof(*collective));
}
