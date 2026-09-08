#include "sparkpipe/spark_weightd_worker.h"
#include <cuda.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct SparkWeightdWork
{
	SparkWeightdWorkFunction function;
	void *context;
} SparkWeightdWork;

struct SparkWeightdWorker
{
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t changed;
	CUcontext cuda_context;
	SparkStatus startup;
	uint32_t ready,stop,active,head,count;
	SparkWeightdWork jobs[SPARK_WEIGHTD_WORK_QUEUE_CAPACITY];
};

static void *worker_main(void *argument)
{
	SparkWeightdWorker *worker = argument;
	SparkWeightdWork job;
	SparkStatus status;
	status = cuCtxSetCurrent(worker->cuda_context) == CUDA_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_TARGET_MISMATCH;
	pthread_mutex_lock(&worker->mutex);
	worker->startup = status;
	worker->ready = 1u;
	pthread_cond_broadcast(&worker->changed);
	while ( status == SPARK_STATUS_OK && worker->stop == 0u )
	{
		while ( worker->count == 0u && worker->stop == 0u )
			pthread_cond_wait(&worker->changed,&worker->mutex);
		if ( worker->stop != 0u )
			break;
		job = worker->jobs[worker->head];
		worker->head = ((worker->head + 1u) % SPARK_WEIGHTD_WORK_QUEUE_CAPACITY);
		worker->count--;
		worker->active = 1u;
		pthread_mutex_unlock(&worker->mutex);
		job.function(job.context);
		pthread_mutex_lock(&worker->mutex);
		worker->active = 0u;
	}
	pthread_mutex_unlock(&worker->mutex);
	return(0);
}

static void worker_free(SparkWeightdWorker *worker)
{
	pthread_cond_destroy(&worker->changed);
	pthread_mutex_destroy(&worker->mutex);
	free(worker);
}

SparkStatus SparkWeightdWorkerCreate(SparkWeightdWorker **out)
{
	SparkWeightdWorker *worker;
	SparkStatus status;
	CUcontext context;
	if ( out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	worker = calloc(1u,sizeof(*worker));
	if ( worker == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	worker->cuda_context = context;
	if ( pthread_mutex_init(&worker->mutex,0) != 0 )
	{
		free(worker);
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( pthread_cond_init(&worker->changed,0) != 0 )
	{
		pthread_mutex_destroy(&worker->mutex);
		free(worker);
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( pthread_create(&worker->thread,0,worker_main,worker) != 0 )
	{
		worker_free(worker);
		return(SPARK_STATUS_IO_ERROR);
	}
	pthread_mutex_lock(&worker->mutex);
	while ( worker->ready == 0u )
		pthread_cond_wait(&worker->changed,&worker->mutex);
	status = worker->startup;
	pthread_mutex_unlock(&worker->mutex);
	if ( status != SPARK_STATUS_OK )
	{
		pthread_join(worker->thread,0);
		worker_free(worker);
		return(status);
	}
	*out = worker;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdWorkerSubmit(SparkWeightdWorker *worker,SparkWeightdWorkFunction function,void *context)
{
	uint32_t tail;
	if ( worker == 0 || function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pthread_mutex_lock(&worker->mutex);
	if ( worker->stop != 0u || worker->count == SPARK_WEIGHTD_WORK_QUEUE_CAPACITY )
	{
		pthread_mutex_unlock(&worker->mutex);
		return(SPARK_STATUS_BUSY);
	}
	tail = ((worker->head + worker->count) % SPARK_WEIGHTD_WORK_QUEUE_CAPACITY);
	worker->jobs[tail].function = function;
	worker->jobs[tail].context = context;
	worker->count++;
	pthread_cond_signal(&worker->changed);
	pthread_mutex_unlock(&worker->mutex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdWorkerDestroy(SparkWeightdWorker *worker)
{
	if ( worker == 0 || pthread_equal(pthread_self(),worker->thread) != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pthread_mutex_lock(&worker->mutex);
	if ( worker->count != 0u || worker->active != 0u )
	{
		pthread_mutex_unlock(&worker->mutex);
		return(SPARK_STATUS_BUSY);
	}
	worker->stop = 1u;
	pthread_cond_signal(&worker->changed);
	pthread_mutex_unlock(&worker->mutex);
	if ( pthread_join(worker->thread,0) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	worker_free(worker);
	return(SPARK_STATUS_OK);
}
