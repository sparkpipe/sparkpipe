#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

extern "C" cudaError_t SparkDsv4GraphStreamWriteSystemU32(
	cudaStream_t stream,uint32_t *word,uint32_t value);
extern "C" cudaError_t SparkDsv4GraphStreamWaitSystemU32(
	cudaStream_t stream,uint32_t *word,uint32_t value);

static __global__ void SparkDsv4GraphSemaphoreProbeKernel(uint32_t *output)
{
	*output = 42u;
}

static uint64_t SparkDsv4GraphSemaphoreNowNs(void)
{
	struct timespec value;
	if ( clock_gettime(CLOCK_MONOTONIC,&value) != 0 )
		return(0u);
	return(((uint64_t)value.tv_sec * UINT64_C(1000000000)) +
		(uint64_t)value.tv_nsec);
}

static int32_t SparkDsv4GraphSemaphoreWaitProducer(
	volatile uint32_t *producer)
{
	uint64_t deadline;
	deadline = SparkDsv4GraphSemaphoreNowNs() + UINT64_C(1000000000);
	while ( __atomic_load_n(producer,__ATOMIC_ACQUIRE) == 0u )
		if ( SparkDsv4GraphSemaphoreNowNs() >= deadline )
			return(-1);
	return(0);
}

int32_t main(void)
{
	cudaGraph_t graph;
	cudaGraphExec_t executable;
	cudaStream_t graph_stream,operation_stream;
	volatile uint32_t *words_host;
	uint32_t *words_device,*output_device,output_host;
	int greatest_priority,least_priority;
	cudaError_t error;
	if ( cudaSetDevice(0) != cudaSuccess ||
		cudaDeviceGetStreamPriorityRange(&least_priority,&greatest_priority) !=
		cudaSuccess ||
		cudaStreamCreateWithFlags(&graph_stream,cudaStreamNonBlocking) !=
		cudaSuccess ||
		cudaStreamCreateWithPriority(&operation_stream,cudaStreamNonBlocking,
			greatest_priority) != cudaSuccess )
		return(-1);
	if ( cudaHostAlloc((void **)&words_host,2u * sizeof(uint32_t),
		cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
		cudaHostGetDevicePointer((void **)&words_device,(void *)words_host,0u) !=
		cudaSuccess || cudaMalloc((void **)&output_device,sizeof(uint32_t)) !=
		cudaSuccess )
		return(-2);
	words_host[0] = 0u;
	words_host[1] = 0u;
	if ( cudaStreamBeginCapture(graph_stream,cudaStreamCaptureModeRelaxed) !=
		cudaSuccess )
		return(-3);
	error = SparkDsv4GraphStreamWriteSystemU32(graph_stream,words_device,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4GraphStreamWaitSystemU32(graph_stream,
			words_device + 1,1u);
	if ( error == cudaSuccess )
		SparkDsv4GraphSemaphoreProbeKernel<<<1,1,0,graph_stream>>>(output_device);
	if ( error != cudaSuccess || cudaStreamEndCapture(graph_stream,&graph) !=
		cudaSuccess || cudaGraphInstantiate(&executable,graph,0ull) !=
		cudaSuccess || cudaGraphLaunch(executable,graph_stream) != cudaSuccess )
		return(-4);
	if ( SparkDsv4GraphSemaphoreWaitProducer(words_host) < 0 )
		return(-5);
	if ( SparkDsv4GraphStreamWriteSystemU32(operation_stream,words_device + 1,
		1u) != cudaSuccess || cudaStreamSynchronize(graph_stream) != cudaSuccess )
		return(-6);
	output_host = 0u;
	if ( cudaMemcpy(&output_host,output_device,sizeof(uint32_t),
		cudaMemcpyDeviceToHost) != cudaSuccess || output_host != 42u )
		return(-7);
	printf("dsv4_graph_semaphore_probe=pass priority=%d\n",greatest_priority);
	return(0);
}
