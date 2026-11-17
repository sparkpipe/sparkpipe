#include "sparkpipe/spark_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define WORDS 64u

static int failures;

static void Check(int ok, const char *what)
{
	printf("%s %s\n", ok ? "PASS" : "FAIL", what);
	if ( !ok )
		failures++;
}

static int ReadOn(SparkGpuStream stream, uint64_t *seen, const void *device)
{
	return SparkGpuMemcpyAsync(seen, device, WORDS * sizeof(uint64_t), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS;
}

static void Fill(uint64_t *words, uint64_t base)
{
	for ( uint32_t index = 0u; index < WORDS; index++ )
		words[index] = base + index;
}

int main(void)
{
	uint64_t pageable[WORDS], seen[WORDS], *pinned = 0;
	void *device = 0, *other = 0;
	SparkGpuStream stream = 0;
	spark_stub_cuda_discrete_ordering(1u);
	if ( SparkGpuMalloc(&device, sizeof(pageable)) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&other, sizeof(pageable)) != SPARK_GPU_SUCCESS ||
		SparkGpuHostAlloc((void **)&pinned, sizeof(pageable), SPARK_GPU_HOST_ALLOC_DEFAULT) != SPARK_GPU_SUCCESS ||
		SparkGpuStreamCreateWithFlags(&stream, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL setup\n");
		return 1;
	}

	Fill(pageable, 0x1000u);
	Check(cudaMemcpy(device, pageable, sizeof(pageable), cudaMemcpyHostToDevice) == cudaSuccess && ReadOn(stream, seen, device) &&
		memcmp(seen, pageable, sizeof(seen)) != 0,
		"control: a raw pageable cudaMemcpy has not landed when a non-blocking stream reads, as on a discrete GPU");
	Check(SparkGpuDeviceSynchronize() == SPARK_GPU_SUCCESS && ReadOn(stream, seen, device) && memcmp(seen, pageable, sizeof(seen)) == 0,
		"control: it lands once the device is synchronized");

	Fill(pinned, 0x2000u);
	Check(cudaMemcpy(device, pinned, sizeof(pageable), cudaMemcpyHostToDevice) == cudaSuccess && ReadOn(stream, seen, device) &&
		memcmp(seen, pinned, sizeof(seen)) == 0,
		"control: a raw cudaMemcpy from pinned memory is host-synchronous and has landed");

	Fill(pageable, 0x3000u);
	Check(SparkGpuMemcpy(device, pageable, sizeof(pageable), SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS && spark_stub_cuda_pending_ops() == 0u &&
		ReadOn(stream, seen, device) && memcmp(seen, pageable, sizeof(seen)) == 0,
		"SparkGpuMemcpy from pageable memory has landed when it returns, so a non-blocking stream reads the new words");

	Check(SparkGpuMemcpy(other, device, sizeof(pageable), SPARK_GPU_COPY_DEVICE_TO_DEVICE) == SPARK_GPU_SUCCESS && spark_stub_cuda_pending_ops() == 0u &&
		ReadOn(stream, seen, other) && memcmp(seen, pageable, sizeof(seen)) == 0,
		"SparkGpuMemcpy device to device has landed when it returns");

	Check(SparkGpuMemset(device, 0x5a, sizeof(pageable)) == SPARK_GPU_SUCCESS && spark_stub_cuda_pending_ops() == 0u && ReadOn(stream, seen, device) &&
		seen[0] == UINT64_C(0x5a5a5a5a5a5a5a5a) && seen[WORDS - 1u] == UINT64_C(0x5a5a5a5a5a5a5a5a),
		"SparkGpuMemset has landed when it returns");

	Fill(pageable, 0x4000u);
	Check(SparkGpuMemcpyAsync(device, pageable, sizeof(pageable), SPARK_GPU_COPY_HOST_TO_DEVICE, stream) == SPARK_GPU_SUCCESS && ReadOn(stream, seen, device) &&
		memcmp(seen, pageable, sizeof(seen)) == 0,
		"an upload ordered in the consumer stream is seen by that stream");

	(void)SparkGpuStreamDestroy(stream);
	(void)SparkGpuFreeHost(pinned);
	(void)SparkGpuFree(other);
	(void)SparkGpuFree(device);
	if ( failures != 0 )
	{
		printf("FAIL upload ordering: %d checks\n", failures);
		return 1;
	}
	printf("PASS upload ordering: on a stub that keeps pageable uploads, device copies and fills pending until the legacy stream synchronizes, as a "
		"discrete GPU may, SparkGpuMemcpy and SparkGpuMemset have landed when they return\n");
	return 0;
}
