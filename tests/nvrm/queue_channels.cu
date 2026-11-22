#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define CHANNELS_BYTES (64ull << 20)
#define CHANNELS_WORDS (CHANNELS_BYTES / sizeof(uint32_t))
#define CHANNELS_ROUNDS 32u
#define CHANNELS_HOLD_NS 1000000ull
#define CHANNELS_BLOCKS 384u
#define CHANNELS_THREADS 256u

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s (%s)\n",what,SparkGpuGetErrorString(SparkGpuGetLastError())); \
			failures++; \
		} \
	} while (0)

__device__ static void ChannelsHold(void)
{
	unsigned long long start,now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < CHANNELS_HOLD_NS );
}

__global__ void ChannelsWrite(uint32_t *output, uint32_t seed)
{
	uint64_t index;
	ChannelsHold();
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < CHANNELS_WORDS; index += (uint64_t)gridDim.x * blockDim.x)
		output[index] = seed * 2654435761u + (uint32_t)index;
}

__global__ void ChannelsCopy(const uint32_t *input, uint32_t *output)
{
	uint64_t index;
	ChannelsHold();
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < CHANNELS_WORDS; index += (uint64_t)gridDim.x * blockDim.x)
		output[index] = input[index] + 1u;
}

static uint64_t Misses(const uint32_t *host, uint32_t seed, uint32_t offset, int bytes_fill)
{
	uint64_t index,misses = 0u;
	const uint32_t filled = (uint32_t)bytes_fill * 0x01010101u;
	for (index = 0u; index < CHANNELS_WORDS; index++)
		misses += host[index] != (bytes_fill >= 0 ? filled + offset : seed * 2654435761u + (uint32_t)index + offset);
	return(misses);
}

int main(void)
{
	SparkGpuStream stream = 0;
	uint32_t *first = 0,*second = 0,*host = 0,*source = 0;
	uint64_t kernel_then_copy = 0u,fill_then_kernel = 0u,upload_then_kernel = 0u;
	uint32_t round;
	uint64_t index;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_queue_channels (no open kernel module on this host)\n");
		return(0);
	}
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&first,CHANNELS_BYTES) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&second,CHANNELS_BYTES) != SPARK_GPU_SUCCESS || SparkGpuMallocHost(&host,CHANNELS_BYTES) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost(&source,CHANNELS_BYTES) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL set up the stream and buffers: %s\n",SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	for (round = 1u; round <= CHANNELS_ROUNDS; round++)
	{
		memset(host,0xee,CHANNELS_BYTES);
		LM_LAUNCH((ChannelsWrite),CHANNELS_BLOCKS,CHANNELS_THREADS,0,stream,first,round);
		CHECK(SparkGpuMemcpyAsync(host,first,CHANNELS_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST,stream) == SPARK_GPU_SUCCESS &&
			SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,"a kernel then a copy of its output");
		kernel_then_copy += Misses(host,round,0u,-1);
		memset(host,0xee,CHANNELS_BYTES);
		CHECK(SparkGpuMemsetAsync(first,(int)(round & 0x7fu),CHANNELS_BYTES,stream) == SPARK_GPU_SUCCESS,"a copy-engine fill");
		LM_LAUNCH((ChannelsCopy),CHANNELS_BLOCKS,CHANNELS_THREADS,0,stream,first,second);
		CHECK(SparkGpuMemcpyAsync(host,second,CHANNELS_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST,stream) == SPARK_GPU_SUCCESS &&
			SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,"a fill, a kernel reading it and a copy of the kernel's output");
		fill_then_kernel += Misses(host,0u,1u,(int)(round & 0x7fu));
		for (index = 0u; index < CHANNELS_WORDS; index++)
			source[index] = (round + 100u) * 2654435761u + (uint32_t)index;
		memset(host,0xee,CHANNELS_BYTES);
		CHECK(SparkGpuMemcpyAsync(first,source,CHANNELS_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) == SPARK_GPU_SUCCESS,"an upload");
		LM_LAUNCH((ChannelsCopy),CHANNELS_BLOCKS,CHANNELS_THREADS,0,stream,first,second);
		CHECK(SparkGpuMemcpyAsync(host,second,CHANNELS_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST,stream) == SPARK_GPU_SUCCESS &&
			SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,"an upload, a kernel reading it and a copy of the kernel's output");
		upload_then_kernel += Misses(host,round + 100u,1u,-1);
	}
	printf("queue channels: %u rounds of 64 MiB, words out of order: kernel then copy %llu, fill then kernel %llu, upload then kernel %llu\n",CHANNELS_ROUNDS,
		(unsigned long long)kernel_then_copy,(unsigned long long)fill_then_kernel,(unsigned long long)upload_then_kernel);
	CHECK(kernel_then_copy == 0u,"a copy after a kernel on one queue reads what the kernel wrote");
	CHECK(fill_then_kernel == 0u,"a kernel after a copy-engine fill on one queue reads the fill");
	CHECK(upload_then_kernel == 0u,"a kernel after an upload on one queue reads the uploaded bytes");
	CHECK(SparkGpuFree(first) == SPARK_GPU_SUCCESS && SparkGpuFree(second) == SPARK_GPU_SUCCESS && SparkGpuFreeHost(host) == SPARK_GPU_SUCCESS &&
		SparkGpuFreeHost(source) == SPARK_GPU_SUCCESS && SparkGpuStreamDestroy(stream) == SPARK_GPU_SUCCESS && SparkNvrmGpuRelease() == SPARK_GPU_SUCCESS,
		"free the buffers and the stream");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS one queue orders kernels and copy-engine work across its two channels, with no host wait between them\n");
	return(0);
}
