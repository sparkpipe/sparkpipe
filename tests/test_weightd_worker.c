#include "sparkpipe/spark_weightd_worker.h"
#include "sparkpipe/spark_device.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

typedef struct State
{
	pthread_mutex_t mutex;
	pthread_cond_t changed;
	pthread_t caller;
	SparkDeviceInstance instance;
	SparkWeightdWorker *worker;
	uint32_t started,release,completed;
} State;

typedef struct Job
{
	State *state;
	uint32_t index;
} Job;

static void execute(void *context)
{
	Job *job = context;
	State *state = job->state;
	SparkDeviceInstance current = 0u;
	assert(pthread_equal(pthread_self(),state->caller) == 0);
	assert(SparkDeviceThreadInstance(&current) == SPARK_STATUS_OK && current == state->instance);
	assert(SparkWeightdWorkerDestroy(state->worker) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkWeightdWorkerWaitIdle(state->worker,1u) == SPARK_STATUS_INVALID_ARGUMENT);
	pthread_mutex_lock(&state->mutex);
	if ( job->index == 0u )
	{
		state->started = 1u;
		pthread_cond_signal(&state->changed);
		while ( state->release == 0u )
			pthread_cond_wait(&state->changed,&state->mutex);
	}
	assert(job->index == state->completed);
	state->completed++;
	pthread_cond_signal(&state->changed);
	pthread_mutex_unlock(&state->mutex);
}

int main(void)
{
	State state = {0};
	Job jobs[SPARK_WEIGHTD_WORK_QUEUE_CAPACITY + 1u];
	uint32_t i;
	assert(pthread_mutex_init(&state.mutex,0) == 0);
	assert(pthread_cond_init(&state.changed,0) == 0);
	state.caller = pthread_self();
	assert(SparkDeviceInitialize(0) == SPARK_STATUS_OK);
	assert(SparkDeviceThreadInstance(&state.instance) == SPARK_STATUS_OK && state.instance != 0u);
	assert(SparkWeightdWorkerCreate(&state.worker) == SPARK_STATUS_OK);
	for (i=0u; i<=SPARK_WEIGHTD_WORK_QUEUE_CAPACITY; i++)
		jobs[i] = (Job){&state,i};
	assert(SparkWeightdWorkerSubmit(state.worker,execute,&jobs[0]) == SPARK_STATUS_OK);
	pthread_mutex_lock(&state.mutex);
	while ( state.started == 0u )
		pthread_cond_wait(&state.changed,&state.mutex);
	pthread_mutex_unlock(&state.mutex);
	for (i=1u; i<=SPARK_WEIGHTD_WORK_QUEUE_CAPACITY; i++)
		assert(SparkWeightdWorkerSubmit(state.worker,execute,&jobs[i]) == SPARK_STATUS_OK);
	assert(SparkWeightdWorkerSubmit(state.worker,execute,0) == SPARK_STATUS_BUSY);
	assert(SparkWeightdWorkerDestroy(state.worker) == SPARK_STATUS_BUSY);
	assert(SparkWeightdWorkerWaitIdle(state.worker,0u) == SPARK_STATUS_BUSY);
	assert(SparkWeightdWorkerWaitIdle(state.worker,1000000u) == SPARK_STATUS_BUSY);
	pthread_mutex_lock(&state.mutex);
	state.release = 1u;
	pthread_cond_signal(&state.changed);
	while ( state.completed != (SPARK_WEIGHTD_WORK_QUEUE_CAPACITY + 1u) )
		pthread_cond_wait(&state.changed,&state.mutex);
	pthread_mutex_unlock(&state.mutex);
	assert(SparkWeightdWorkerWaitIdle(state.worker,1000000000u) == SPARK_STATUS_OK);
	{
		struct timespec before,after;
		uint64_t elapsed_ns;
		state.completed = 0u;
		jobs[1] = (Job){&state,0u};
		assert(clock_gettime(CLOCK_MONOTONIC,&before) == 0);
		assert(SparkWeightdWorkerSubmit(state.worker,execute,&jobs[1]) == SPARK_STATUS_OK);
		assert(SparkWeightdWorkerWaitIdle(state.worker,1000000000u) == SPARK_STATUS_OK);
		assert(clock_gettime(CLOCK_MONOTONIC,&after) == 0);
		elapsed_ns = ((uint64_t)(after.tv_sec - before.tv_sec) * UINT64_C(1000000000)) + (uint64_t)after.tv_nsec - (uint64_t)before.tv_nsec;
		assert(state.completed == 1u && elapsed_ns < UINT64_C(50000000));
	}
	assert(SparkWeightdWorkerDestroy(state.worker) == SPARK_STATUS_OK);
	pthread_cond_destroy(&state.changed);
	pthread_mutex_destroy(&state.mutex);
	puts("PASS weightd worker: separate thread, FIFO, bounded admission and idle teardown");
	return(0);
}
