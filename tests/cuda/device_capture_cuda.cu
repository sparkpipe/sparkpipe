#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"

static int failures;

#define CHECK(condition, text) do { if ( !(condition) ) { printf("FAIL %s\n", text); failures++; } } while ( 0 )

__global__ void Bump(uint32_t *value)
{
	value[threadIdx.x] += 1u;
}

typedef struct Prober
{
	volatile uint32_t go, stop;
	uint32_t raw;
	cudaStream_t stream;
	SparkDeviceQueue *adopted, *own;
	SparkDeviceBuffer *buffer;
	uint64_t polls, pending;
} Prober;

static void *Probe(void *argument)
{
	Prober *prober = (Prober *)argument;
	while ( prober->go == 0u )
	{
	}
	while ( prober->stop == 0u )
	{
		if ( prober->raw != 0u )
			(void)cudaStreamQuery(prober->stream);
		else
		{
			SparkDeviceRange range;
			range.buffer = prober->buffer;
			range.offset = 0u;
			prober->pending += SparkDeviceQueueQuery(prober->adopted) == SPARK_STATUS_PENDING ? 1u : 0u;
			prober->pending += SparkDeviceQueueWait(prober->adopted, 200000u) == SPARK_STATUS_PENDING ? 1u : 0u;
			(void)SparkDeviceFill(prober->own, range, (uint8_t)prober->polls, 4096u);
			(void)SparkDeviceQueueWait(prober->own, SPARK_DEVICE_WAIT_FOREVER);
		}
		prober->polls++;
	}
	return 0;
}

static cudaError_t CaptureWithProber(cudaStream_t stream, uint32_t *value, Prober *prober, cudaGraph_t *graph)
{
	pthread_t thread;
	cudaError_t error;
	uint64_t until;
	prober->go = 0u;
	prober->stop = 0u;
	prober->polls = 0u;
	prober->pending = 0u;
	pthread_create(&thread, 0, Probe, prober);
	error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
	prober->go = 1u;
	until = SparkMonotonicNs() + 50000000ull;
	while ( error == cudaSuccess && SparkMonotonicNs() < until )
	{
		Bump<<<1, 32, 0, stream>>>(value);
		error = cudaPeekAtLastError();
	}
	while ( prober->polls < 4u )
	{
	}
	{
		const cudaError_t ended = cudaStreamEndCapture(stream, graph);
		error = error != cudaSuccess ? error : ended;
	}
	prober->stop = 1u;
	pthread_join(thread, 0);
	(void)cudaGetLastError();
	return error;
}

int main(void)
{
	cudaStream_t stream = 0;
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec = 0;
	uint32_t *value = 0;
	SparkDeviceQueue *adopted = 0, *own = 0;
	SparkDeviceBuffer buffer;
	SparkDeviceRange range;
	Prober prober;
	cudaError_t error;
	memset(&prober, 0, sizeof(prober));
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK, "initialize the device API");
	CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc(&value, 32u * sizeof(uint32_t)) == cudaSuccess,
		"create the stream the owner captures");
	CHECK(SparkDeviceQueueAdopt(&adopted, (void *)stream, "capture owner") == SPARK_STATUS_OK, "adopt the owner's stream");
	CHECK(SparkDeviceQueueCreate(&own, "prober") == SPARK_STATUS_OK, "create the prober's own queue");
	CHECK(SparkDeviceAllocate(&buffer, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, 1u << 20, "capture stamps") == SPARK_STATUS_OK, "allocate a shared buffer");
	range.buffer = &buffer;
	range.offset = 0u;
	CHECK(SparkDeviceFill(adopted, range, 0u, 1u << 20) == SPARK_STATUS_OK, "the owner's queue stamps the buffer");
	CHECK(SparkDeviceQueueCapture(own, 1u) == SPARK_STATUS_UNSUPPORTED, "a queue the device API created is never recorded into a graph");
	prober.stream = stream;
	prober.adopted = adopted;
	prober.own = own;
	prober.buffer = &buffer;
	prober.raw = 1u;
	error = CaptureWithProber(stream, value, &prober, &graph);
	CHECK(error == cudaErrorStreamCaptureInvalidated || error == cudaErrorStreamCaptureUnsupported,
		"negative control: another thread's raw stream query invalidates the owner's thread-local capture");
	if ( graph != 0 )
		(void)cudaGraphDestroy(graph);
	graph = 0;
	CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "the stream leaves the failed capture usable");
	CHECK(SparkDeviceQueueCapture(adopted, 1u) == SPARK_STATUS_OK, "the owner marks its adopted queue before capturing it");
	prober.raw = 0u;
	error = CaptureWithProber(stream, value, &prober, &graph);
	CHECK(SparkDeviceQueueCapture(adopted, 0u) == SPARK_STATUS_OK, "the owner clears the mark after the capture");
	CHECK(error == cudaSuccess && graph != 0, "queue queries, waits and stamped fills from another thread leave a marked capture whole");
	CHECK(prober.polls != 0u && prober.pending == 2u * prober.polls, "a marked queue reads as busy without its stream being queried");
	CHECK(graph != 0 && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess && cudaGraphLaunch(exec, stream) == cudaSuccess &&
		cudaStreamSynchronize(stream) == cudaSuccess, "the captured graph runs");
	CHECK(SparkDeviceQueueQuery(adopted) == SPARK_STATUS_OK && SparkDeviceQueueWait(adopted, SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"after the mark clears the adopted queue answers from its stream again");
	if ( exec != 0 )
		(void)cudaGraphExecDestroy(exec);
	if ( graph != 0 )
		(void)cudaGraphDestroy(graph);
	CHECK(SparkDeviceFree(&buffer) == SPARK_STATUS_OK, "free the shared buffer");
	CHECK(SparkDeviceQueueDestroy(own) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(adopted) == SPARK_STATUS_OK, "destroy both queues");
	(void)cudaFree(value);
	(void)cudaStreamDestroy(stream);
	if ( failures != 0 )
	{
		printf("FAIL device capture (cuda backend): %d failure(s)\n", failures);
		return 1;
	}
	printf("PASS device capture (cuda backend): another thread's raw stream query invalidates a thread-local capture, while device API queries, waits "
		"and stamped fills against a marked adopted queue read it as busy and leave the capture whole; created queues refuse the mark\n");
	return 0;
}
