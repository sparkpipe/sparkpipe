#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_device.h"

static int failures;

#define CHECK(condition, what) do { if ( !(condition) ) { const SparkDeviceError *last = SparkDeviceLastError(); \
	printf("FAIL %s:%d %s (last device error: %s %s)\n", __FILE__, __LINE__, what, last->operation, last->text); failures++; } } while ( 0 )

typedef enum CaptureAction
{
	CAPTURE_ACTION_COPY = 1,
	CAPTURE_ACTION_FREE = 2,
	CAPTURE_ACTION_RECORD = 3,
	CAPTURE_ACTION_QUERY_STREAM = 4
} CaptureAction;

typedef struct CaptureWork
{
	CaptureAction action;
	SparkDeviceQueue *queue;
	SparkDeviceRange destination;
	SparkDeviceRange source;
	uint64_t bytes;
	SparkDeviceBuffer *buffer;
	SparkDeviceEvent *event;
	cudaStream_t stream;
	SparkStatus status;
} CaptureWork;

typedef struct Gate
{
	pthread_mutex_t mutex;
	pthread_cond_t changed;
	int open;
} Gate;

__global__ static void CountKernel(uint32_t *counter)
{
	*counter += 1u;
}

static void GateInit(Gate *gate)
{
	pthread_mutex_init(&gate->mutex, 0);
	pthread_cond_init(&gate->changed, 0);
	gate->open = 0;
}

static void GateOpen(Gate *gate)
{
	pthread_mutex_lock(&gate->mutex);
	gate->open = 1;
	pthread_cond_broadcast(&gate->changed);
	pthread_mutex_unlock(&gate->mutex);
}

static void GateWaitCallback(void *context)
{
	Gate *gate = (Gate *)context;
	pthread_mutex_lock(&gate->mutex);
	while ( gate->open == 0 )
		pthread_cond_wait(&gate->changed, &gate->mutex);
	pthread_mutex_unlock(&gate->mutex);
}

static SparkDeviceRange At(const SparkDeviceBuffer *buffer, uint64_t offset)
{
	SparkDeviceRange range;
	range.buffer = buffer;
	range.offset = offset;
	return range;
}

static void *CaptureWorkMain(void *argument)
{
	CaptureWork *work = (CaptureWork *)argument;
	switch ( work->action )
	{
	case CAPTURE_ACTION_COPY:
		work->status = SparkDeviceCopy(work->queue, work->destination, work->source, work->bytes);
		if ( work->status == SPARK_STATUS_OK )
			work->status = SparkDeviceQueueWait(work->queue, SPARK_DEVICE_WAIT_FOREVER);
		break;
	case CAPTURE_ACTION_FREE:
		work->status = SparkDeviceFree(work->buffer);
		break;
	case CAPTURE_ACTION_RECORD:
		work->status = SparkDeviceEventRecord(work->event, work->queue);
		if ( work->status == SPARK_STATUS_OK )
			work->status = SparkDeviceEventWait(work->event, SPARK_DEVICE_WAIT_FOREVER);
		break;
	case CAPTURE_ACTION_QUERY_STREAM:
		(void)cudaStreamQuery(work->stream);
		work->status = SPARK_STATUS_OK;
		break;
	}
	return 0;
}

static cudaError_t CaptureAround(cudaStream_t stream, uint32_t *counter, CaptureWork *work)
{
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec = 0;
	pthread_t thread;
	uint32_t before = 0u, after = 0u;
	cudaError_t error = cudaMemcpy(&before, counter, sizeof(before), cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess )
		error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
	if ( error != cudaSuccess )
		return error;
	CountKernel<<<1, 1, 0, stream>>>(counter);
	if ( pthread_create(&thread, 0, CaptureWorkMain, work) == 0 )
		(void)pthread_join(thread, 0);
	else
		work->status = SPARK_STATUS_INTERNAL_ERROR;
	CountKernel<<<1, 1, 0, stream>>>(counter);
	error = cudaStreamEndCapture(stream, &graph);
	if ( error == cudaSuccess )
		error = cudaGraphInstantiate(&exec, graph, 0);
	if ( error == cudaSuccess )
		error = cudaGraphLaunch(exec, stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy(&after, counter, sizeof(after), cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess && after != before + 2u )
		error = cudaErrorUnknown;
	if ( exec != 0 )
		(void)cudaGraphExecDestroy(exec);
	if ( graph != 0 )
		(void)cudaGraphDestroy(graph);
	(void)cudaGetLastError();
	return error;
}

static void ReportCapture(cudaError_t error, const char *what)
{
	if ( error != cudaSuccess )
	{
		printf("FAIL %s: the capture on the adopted stream ended with %s\n", what, cudaGetErrorString(error));
		failures++;
	}
}

static void TestSaverCopy(SparkDeviceQueue *owner, cudaStream_t stream, uint32_t *counter)
{
	enum { BYTES = 65536 };
	SparkDeviceQueue *copier = 0;
	SparkDeviceBuffer pool, table, snapshot;
	CaptureWork work;
	CHECK(SparkDeviceQueueCreate(&copier, "capture copier") == SPARK_STATUS_OK, "create the copy queue");
	CHECK(SparkDeviceAllocate(&pool, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, BYTES, "capture pool") == SPARK_STATUS_OK, "allocate the pool");
	CHECK(SparkDeviceAllocate(&table, SPARK_DEVICE_SPACE_HOST_PINNED, BYTES, "capture table") == SPARK_STATUS_OK, "allocate the table");
	CHECK(SparkDeviceAllocate(&snapshot, SPARK_DEVICE_SPACE_HOST_PINNED, BYTES, "capture snapshot") == SPARK_STATUS_OK, "allocate the snapshot page");
	memset(table.host, 0x5a, BYTES);
	CHECK(SparkDeviceFill(copier, At(&pool, 0u), 0xffu, BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(copier, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK, "the copy queue fills the pool");
	CHECK(SparkDeviceCopy(owner, At(&pool, 0u), At(&table, 0u), BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(owner, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK, "the adopted queue uploads a table into the pool");
	CHECK(SparkDeviceCopy(owner, At(&pool, 0u), At(&table, 0u), BYTES) == SPARK_STATUS_OK, "the adopted queue uploads the table again before its step");
	memset(&work, 0, sizeof(work));
	work.action = CAPTURE_ACTION_COPY;
	work.queue = copier;
	work.destination = At(&snapshot, 0u);
	work.source = At(&pool, 0u);
	work.bytes = BYTES;
	ReportCapture(CaptureAround(stream, counter, &work), "a copy out of the pool on another queue during the capture");
	CHECK(work.status == SPARK_STATUS_OK, "the copy out of the pool completes");
	CHECK(SparkDeviceQueueWait(owner, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK, "the adopted queue drains after the capture");
	CHECK(SparkDeviceFree(&pool) == SPARK_STATUS_OK && SparkDeviceFree(&table) == SPARK_STATUS_OK && SparkDeviceFree(&snapshot) == SPARK_STATUS_OK,
		"free the pool, the table and the snapshot page");
	CHECK(SparkDeviceQueueDestroy(copier) == SPARK_STATUS_OK, "destroy the copy queue");
}

static void TestEveryStampTaken(SparkDeviceQueue *owner, cudaStream_t stream, uint32_t *counter)
{
	enum { BYTES = 4096, OTHERS = 7 };
	SparkDeviceQueue *others[OTHERS], *late = 0;
	SparkDeviceBuffer shared, source, sink;
	CaptureWork work;
	Gate gate;
	uint32_t index;
	memset(others, 0, sizeof(others));
	GateInit(&gate);
	CHECK(SparkDeviceAllocate(&shared, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, BYTES, "capture shared") == SPARK_STATUS_OK, "allocate the shared buffer");
	CHECK(SparkDeviceAllocate(&source, SPARK_DEVICE_SPACE_HOST_PINNED, BYTES, "capture source") == SPARK_STATUS_OK, "allocate the source");
	CHECK(SparkDeviceAllocate(&sink, SPARK_DEVICE_SPACE_HOST_PINNED, BYTES, "capture sink") == SPARK_STATUS_OK, "allocate the sink");
	CHECK(SparkDeviceQueueHostCallback(owner, GateWaitCallback, &gate) == SPARK_STATUS_OK, "hold the adopted queue");
	CHECK(SparkDeviceCopy(owner, At(&shared, 0u), At(&source, 0u), BYTES) == SPARK_STATUS_OK, "the adopted queue stamps the shared buffer first");
	for ( index = 0u; index < OTHERS; index++ )
		CHECK(SparkDeviceQueueCreate(&others[index], "capture other") == SPARK_STATUS_OK &&
			SparkDeviceCopy(others[index], At(&shared, 0u), At(&source, 0u), BYTES) == SPARK_STATUS_OK &&
			SparkDeviceQueueWait(others[index], SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK, "another queue stamps the shared buffer");
	GateOpen(&gate);
	CHECK(SparkDeviceQueueWait(owner, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK, "the adopted queue's owner waits for it");
	CHECK(SparkDeviceQueueCreate(&late, "capture late") == SPARK_STATUS_OK, "create a ninth queue");
	memset(&work, 0, sizeof(work));
	work.action = CAPTURE_ACTION_COPY;
	work.queue = late;
	work.destination = At(&sink, 0u);
	work.source = At(&shared, 0u);
	work.bytes = BYTES;
	ReportCapture(CaptureAround(stream, counter, &work), "a ninth queue copying a buffer whose every stamp slot is taken during the capture");
	CHECK(work.status == SPARK_STATUS_OK, "the ninth queue reuses a settled stamp slot");
	CHECK(SparkDeviceFree(&shared) == SPARK_STATUS_OK && SparkDeviceFree(&source) == SPARK_STATUS_OK && SparkDeviceFree(&sink) == SPARK_STATUS_OK,
		"free the shared buffer, the source and the sink");
	for ( index = 0u; index < OTHERS; index++ )
		CHECK(others[index] != 0 && SparkDeviceQueueDestroy(others[index]) == SPARK_STATUS_OK, "destroy another queue");
	CHECK(SparkDeviceQueueDestroy(late) == SPARK_STATUS_OK, "destroy the ninth queue");
}

static void TestFreeDuringCapture(SparkDeviceQueue *owner, cudaStream_t stream, uint32_t *counter)
{
	enum { BYTES = 4096 };
	SparkDeviceBuffer staged, source;
	CaptureWork work;
	CHECK(SparkDeviceAllocate(&staged, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, BYTES, "capture staged") == SPARK_STATUS_OK, "allocate the staged buffer");
	CHECK(SparkDeviceAllocate(&source, SPARK_DEVICE_SPACE_HOST_PINNED, BYTES, "capture staged source") == SPARK_STATUS_OK, "allocate its source");
	CHECK(SparkDeviceCopy(owner, At(&staged, 0u), At(&source, 0u), BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(owner, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK, "the adopted queue writes the staged buffer and its owner waits");
	memset(&work, 0, sizeof(work));
	work.action = CAPTURE_ACTION_FREE;
	work.buffer = &staged;
	ReportCapture(CaptureAround(stream, counter, &work), "freeing a buffer the adopted queue used during the capture");
	CHECK(work.status == SPARK_STATUS_OK, "a buffer whose adopted-queue work its owner waited for frees during the capture");
	CHECK(SparkDeviceFree(&source) == SPARK_STATUS_OK, "free the staged source");
}

static void TestFenceRecord(SparkDeviceQueue *owner, cudaStream_t stream, uint32_t *counter)
{
	SparkDeviceQueue *copier = 0;
	SparkDeviceEvent *fence = 0;
	CaptureWork work;
	CHECK(SparkDeviceQueueCreate(&copier, "capture fence copier") == SPARK_STATUS_OK && SparkDeviceEventCreate(&fence) == SPARK_STATUS_OK,
		"create the copy queue and its fence");
	CHECK(SparkDeviceEventRecord(fence, copier) == SPARK_STATUS_OK && SparkDeviceQueueWait(copier, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"the copy queue records the fence");
	CHECK(SparkDeviceQueueWaitEvent(owner, fence) == SPARK_STATUS_OK && SparkDeviceQueueWait(owner, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"the adopted queue waits on the fence");
	memset(&work, 0, sizeof(work));
	work.action = CAPTURE_ACTION_RECORD;
	work.queue = copier;
	work.event = fence;
	ReportCapture(CaptureAround(stream, counter, &work), "the copy queue recording the fence again during the capture");
	CHECK(work.status == SPARK_STATUS_OK, "the fence records and completes");
	CHECK(SparkDeviceEventDestroy(fence) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(copier) == SPARK_STATUS_OK, "destroy the fence and its queue");
}

int main(void)
{
	cudaStream_t stream = 0;
	SparkDeviceQueue *owner = 0;
	uint32_t *counter = 0;
	CaptureWork premise;
	cudaError_t premise_error;
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK, "initialize the device");
	if ( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess || cudaMalloc((void **)&counter, sizeof(*counter)) != cudaSuccess ||
		cudaMemset(counter, 0, sizeof(*counter)) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
	{
		printf("FAIL create the capture stream and its counter\n");
		return 1;
	}
	CHECK(SparkDeviceQueueAdopt(&owner, (void *)stream, "capture owner") == SPARK_STATUS_OK, "adopt the capture stream");
	memset(&premise, 0, sizeof(premise));
	premise.action = CAPTURE_ACTION_QUERY_STREAM;
	premise.stream = stream;
	premise_error = CaptureAround(stream, counter, &premise);
	printf("NOTE a query of the capturing stream from another thread %s the capture (%s)\n",
		premise_error != cudaSuccess ? "invalidates" : "does not invalidate", cudaGetErrorString(premise_error));
	TestSaverCopy(owner, stream, counter);
	TestEveryStampTaken(owner, stream, counter);
	TestFreeDuringCapture(owner, stream, counter);
	TestFenceRecord(owner, stream, counter);
	CHECK(SparkDeviceQueueDestroy(owner) == SPARK_STATUS_OK, "release the adopted queue");
	(void)cudaFree(counter);
	(void)cudaStreamDestroy(stream);
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK, "shut the device down");
	if ( failures != 0 )
	{
		printf("device_adopted_capture_cuda: %d failure(s)\n", failures);
		return 1;
	}
	printf("PASS device_adopted_capture_cuda: work on other queues leaves a capture on an adopted stream intact (saver copy, every stamp slot taken, free, fence record)\n");
	return 0;
}
