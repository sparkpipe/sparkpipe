#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

#include <cuda_runtime.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" cudaError_t SparkDsv4LaunchExactBf16ProducerMirror(
	cudaStream_t stream,const void *source,uint64_t source_bytes,
	const SparkDsv4ExactBf16MirrorTarget *target);

extern "C" cudaError_t SparkDsv4LaunchAccumU64MaxTp4(
	cudaStream_t stream,uint64_t *destination,
	const uint64_t *const rank_devices[4],uint32_t tp_rank,
	uint32_t element_count);

static void TestU64MaxTp4(cudaStream_t stream)
{
	const uint64_t rank_values[4][4] = {
		{UINT64_C(1),UINT64_C(90),UINT64_C(3),UINT64_C(4)},
		{UINT64_C(20),UINT64_C(8),UINT64_C(30),UINT64_C(2)},
		{UINT64_C(7),UINT64_C(9),UINT64_C(300),UINT64_C(40)},
		{UINT64_C(6),UINT64_C(80),UINT64_C(5),UINT64_C(400)}
	};
	const uint64_t expected[4] = {
		UINT64_C(20),UINT64_C(90),UINT64_C(300),UINT64_C(400)
	};
	const uint64_t *rank_devices[4];
	uint64_t actual[4];
	uint64_t *device_values[4];
	uint32_t rank;

	for (rank=0u; rank<4u; rank++)
	{
		assert(cudaMalloc(&device_values[rank],sizeof(rank_values[rank])) ==
			cudaSuccess);
		assert(cudaMemcpyAsync(device_values[rank],rank_values[rank],
			sizeof(rank_values[rank]),cudaMemcpyHostToDevice,stream) ==
			cudaSuccess);
		rank_devices[rank] = device_values[rank];
	}
	assert(SparkDsv4LaunchAccumU64MaxTp4(stream,device_values[1],rank_devices,
		1u,4u) == cudaSuccess);
	assert(cudaMemcpyAsync(actual,device_values[1],sizeof(actual),
		cudaMemcpyDeviceToHost,stream) == cudaSuccess);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(memcmp(actual,expected,sizeof(expected)) == 0);
	for (rank=0u; rank<4u; rank++)
		assert(cudaFree(device_values[rank]) == cudaSuccess);
}

int main(void)
{
	SparkDsv4ExactBf16MirrorTarget target;
	cudaStream_t stream;
	uint8_t host_first[16],host_second[16],host_source[64];
	void *first,*second,*source;
	uint32_t index;
	for (index=0u; index<sizeof(host_source); index++)
		host_source[index] = (uint8_t)((index * 37u) ^ (index >> 1u));
	assert(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) ==
		cudaSuccess);
	assert(cudaMalloc(&source,sizeof(host_source)) == cudaSuccess);
	assert(cudaMalloc(&first,sizeof(host_first)) == cudaSuccess);
	assert(cudaMalloc(&second,sizeof(host_second)) == cudaSuccess);
	assert(cudaMemcpyAsync(source,host_source,sizeof(host_source),
		cudaMemcpyHostToDevice,stream) == cudaSuccess);
	memset(&target,0,sizeof(target));
	target.span_count = 2u;
	target.spans[0].destination_device = first;
	target.spans[0].source_offset_bytes = 16u;
	target.spans[0].byte_count = sizeof(host_first);
	target.spans[1].destination_device = second;
	target.spans[1].source_offset_bytes = 48u;
	target.spans[1].byte_count = sizeof(host_second);
	assert(SparkDsv4LaunchExactBf16ProducerMirror(stream,source,
		sizeof(host_source),&target) == cudaSuccess);
	assert(cudaMemcpyAsync(host_first,first,sizeof(host_first),
		cudaMemcpyDeviceToHost,stream) == cudaSuccess);
	assert(cudaMemcpyAsync(host_second,second,sizeof(host_second),
		cudaMemcpyDeviceToHost,stream) == cudaSuccess);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(memcmp(host_first,host_source + 16u,sizeof(host_first)) == 0);
	assert(memcmp(host_second,host_source + 48u,sizeof(host_second)) == 0);
	TestU64MaxTp4(stream);
	target.spans[1].source_offset_bytes = 16u;
	assert(SparkDsv4LaunchExactBf16ProducerMirror(stream,source,
		sizeof(host_source),&target) == cudaErrorInvalidValue);
	assert(cudaFree(second) == cudaSuccess);
	assert(cudaFree(first) == cudaSuccess);
	assert(cudaFree(source) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	puts("dsv4_exact_bf16_mirror: ok");
	return(0);
}
