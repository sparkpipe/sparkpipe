#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

__global__ void SingleEvalDirectKernel(uint32_t *out, uint32_t value)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		out[0] = value;
}

template<uint32_t Tag>
__global__ void SingleEvalDefaultKernel(uint32_t *out, uint32_t value, uint32_t extra = 77u)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
	{
		out[0] = value;
		out[1] = extra;
	}
}

template<uint32_t Tag>
__global__ void SingleEvalOverloadedKernel(uint32_t *out, uint32_t a)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		out[0] = a;
}

template<uint32_t Tag>
__global__ void SingleEvalOverloadedKernel(uint32_t *out, uint32_t a, uint32_t b)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
	{
		out[0] = a;
		out[1] = b;
	}
}

static uint32_t g_failures = 0u;

static void Check(int cond, const char *msg)
{
	if ( cond )
		return;
	fprintf(stderr, "FAIL %s\n", msg);
	g_failures++;
}

#if defined(SPARKPIPE_DEVICE_NVRM)
static uint32_t g_nvrm_out[2] = {0u, 0u};
static uint32_t g_nvrm_launches = 0u;
static uint32_t g_nvrm_dependent = 0u;

extern "C" SparkGpuError SparkNvrmLaunchPacked(const void *function, const uint32_t grid[3], const uint32_t block[3], size_t shared_bytes,
	SparkGpuStream stream, const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes)
{
	(void)function; (void)grid; (void)block; (void)shared_bytes; (void)stream; (void)bytes;
	g_nvrm_out[0] = 0u;
	g_nvrm_out[1] = 0u;
	g_nvrm_dependent = 0u;
	if ( count >= 2u )
		memcpy(&g_nvrm_out[0], (const uint8_t *)parameters + layout[1].offset, sizeof(uint32_t));
	if ( count >= 3u )
		memcpy(&g_nvrm_out[1], (const uint8_t *)parameters + layout[2].offset, sizeof(uint32_t));
	g_nvrm_launches++;
	return SPARK_GPU_SUCCESS;
}

extern "C" SparkGpuError SparkNvrmLaunchDependent(const void *function, const uint32_t grid[3], const uint32_t block[3], size_t shared_bytes,
	SparkGpuStream stream, const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes)
{
	const SparkGpuError err = SparkNvrmLaunchPacked(function, grid, block, shared_bytes, stream, parameters, layout, count, bytes);
	g_nvrm_dependent = 1u;
	return err;
}

extern "C" SparkGpuError SparkNvrmLaunchPointers(const void *function, const uint32_t grid[3], const uint32_t block[3], uint32_t dependent, size_t shared_bytes,
	SparkGpuStream stream, void **arguments)
{
	SparkDeviceParameter layout[3] = {{0u, 8u, 8u}, {8u, 4u, 4u}, {12u, 4u, 4u}};
	uint8_t packed[16];
	memset(packed, 0, sizeof(packed));
	memcpy(packed + 0u, arguments[0], 8u);
	memcpy(packed + 8u, arguments[1], 4u);
	memcpy(packed + 12u, arguments[2], 4u);
	return dependent != 0u
		? SparkNvrmLaunchDependent(function, grid, block, shared_bytes, stream, packed, layout, 3u, 16u)
		: SparkNvrmLaunchPacked(function, grid, block, shared_bytes, stream, packed, layout, 3u, 16u);
}
#endif

static void FetchOut(uint32_t *device_out, uint32_t host_out[2], const char *tag)
{
#if defined(SPARKPIPE_DEVICE_NVRM)
	(void)device_out; (void)tag;
	host_out[0] = g_nvrm_out[0];
	host_out[1] = g_nvrm_out[1];
#else
	Check(SparkGpuDeviceSynchronize() == SPARK_GPU_SUCCESS, tag);
	Check(SparkGpuMemcpy(host_out, device_out, 2u * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS, tag);
#endif
}

int main(void)
{
	uint32_t *device_out = 0;
	uint32_t host_out[2] = {0u, 0u};
	uint32_t i = 10u, g = 1u, b = 32u, s = 0u;
	SparkGpuStream stream = 0;
#if !defined(SPARKPIPE_DEVICE_NVRM)
	if ( SparkGpuMalloc((void **)&device_out, sizeof(host_out)) != SPARK_GPU_SUCCESS )
	{
		fprintf(stderr, "FAIL SparkGpuMalloc\n");
		return 1;
	}
#endif
	LM_LAUNCH((SingleEvalDirectKernel), g++, b++, s++, stream, device_out, i++);
	FetchOut(device_out, host_out, "direct");
	Check(i == 11u && g == 2u && b == 33u && s == 1u && host_out[0] == 10u, "direct launch evaluates args once");

	g = 1u; b = 32u; s = 0u;
	LM_LAUNCH((SingleEvalDefaultKernel<1u>), g++, b++, s++, stream, device_out, i++);
	FetchOut(device_out, host_out, "default");
	Check(i == 12u && g == 2u && b == 33u && s == 1u && host_out[0] == 11u && host_out[1] == 77u, "default launch evaluates args once");

	g = 1u; b = 32u; s = 0u;
	LM_LAUNCH((SingleEvalOverloadedKernel<2u>), g++, b++, s++, stream, device_out, i++, 99u);
	FetchOut(device_out, host_out, "overloaded");
	Check(i == 13u && g == 2u && b == 33u && s == 1u && host_out[0] == 12u && host_out[1] == 99u, "overloaded launch evaluates args once");

	g = 1u; b = 32u; s = 0u;
	Check(LM_LAUNCH_DEPENDENT((SingleEvalDirectKernel), g++, b++, s++, stream, device_out, i++) == SPARK_GPU_SUCCESS, "dependent direct launch");
	FetchOut(device_out, host_out, "dependent direct");
	Check(i == 14u && g == 2u && b == 33u && s == 1u && host_out[0] == 13u, "dependent direct launch evaluates args once");

#if defined(SPARKPIPE_DEVICE_NVRM)
	g = 1u; b = 32u; s = 0u;
	Check(LM_LAUNCH_DEPENDENT((SingleEvalDefaultKernel<1u>), g++, b++, s++, stream, device_out, i++) == SPARK_GPU_SUCCESS && g_nvrm_dependent == 1u,
		"dependent default launch");
	FetchOut(device_out, host_out, "dependent default");
	Check(i == 15u && g == 2u && b == 33u && s == 1u && host_out[0] == 14u && host_out[1] == 77u, "dependent default launch evaluates args once");
#else
	(void)SparkGpuFree(device_out);
#endif
	if ( g_failures != 0u )
		return 1;
	printf("PASS launch_single_eval (%s): every launch argument evaluated strictly once\n",
#if defined(SPARKPIPE_DEVICE_NVRM)
		"nvrm"
#else
		"cuda"
#endif
	);
	return 0;
}
