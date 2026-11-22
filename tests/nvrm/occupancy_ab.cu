#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define AB_STATIC_WORDS 3072u
#define AB_DYNAMIC_GRANT (96u * 1024u)

__global__ void AbLight(uint32_t *output)
{
	output[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x;
}

__global__ void AbHeavy(const float *input, float *output, uint32_t count)
{
	float accumulators[48];
	uint32_t index,step;
	for (index = 0u; index < 48u; index++)
		accumulators[index] = input[(threadIdx.x + index) % count];
	for (step = 0u; step < count; step++)
		for (index = 0u; index < 48u; index++)
			accumulators[index] = accumulators[index] * accumulators[(index + step) % 48u] + input[(step + index) % count];
	for (index = 1u; index < 48u; index++)
		accumulators[0] += accumulators[index];
	output[blockIdx.x * blockDim.x + threadIdx.x] = accumulators[0];
}

__global__ void AbStatic(uint32_t *output)
{
	__shared__ uint32_t tile[AB_STATIC_WORDS];
	uint32_t index;
	for (index = threadIdx.x; index < AB_STATIC_WORDS; index += blockDim.x)
		tile[index] = index;
	__syncthreads();
	output[blockIdx.x * blockDim.x + threadIdx.x] = tile[(threadIdx.x * 7u) % AB_STATIC_WORDS];
}

__global__ void AbDynamic(uint32_t *output, uint32_t words)
{
	extern __shared__ uint32_t buffer[];
	uint32_t index;
	for (index = threadIdx.x; index < words; index += blockDim.x)
		buffer[index] = index;
	__syncthreads();
	output[blockIdx.x * blockDim.x + threadIdx.x] = words != 0u ? buffer[threadIdx.x % words] : 0u;
}

__global__ void AbResident(uint32_t *active, uint32_t *peak, unsigned long long spin)
{
	extern __shared__ uint32_t held[];
	uint32_t sm;
	long long start;
	asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
	if ( threadIdx.x == 0u )
		atomicMax(&peak[sm],atomicAdd(&active[sm],1u) + 1u);
	__syncthreads();
	start = clock64();
	while ( (unsigned long long)(clock64() - start) < spin )
		;
	held[threadIdx.x] = sm;
	__syncthreads();
	if ( threadIdx.x == 0u )
		atomicSub(&active[sm],1u);
}

#define AB_MAX_SMS 256u
#define AB_SPIN_CYCLES 200000ull

static int Residency(int sms, int threads, uint32_t dynamic)
{
	uint32_t *active = 0,*peak = 0,host[AB_MAX_SMS];
	int predicted = 0,most = 0,index;
	const uint32_t bytes = AB_MAX_SMS * sizeof(uint32_t);
	if ( SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&predicted,AbResident,threads,dynamic + (uint32_t)threads * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&active,bytes) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&peak,bytes) != SPARK_GPU_SUCCESS ||
		SparkGpuMemset(active,0,bytes) != SPARK_GPU_SUCCESS || SparkGpuMemset(peak,0,bytes) != SPARK_GPU_SUCCESS )
		return(1);
	LM_LAUNCH((AbResident),dim3((unsigned)(sms * (predicted > 0 ? predicted : 1) * 2)),dim3((unsigned)threads),dynamic + (uint32_t)threads * sizeof(uint32_t),0,
		active,peak,AB_SPIN_CYCLES);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS || SparkGpuDeviceSynchronize() != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(host,peak,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		return(1);
	for (index = 0; index < sms && index < (int)AB_MAX_SMS; index++)
		most = host[index] > (uint32_t)most ? (int)host[index] : most;
	printf("residency threads %d dynamic %u predicted %d peak %d\n",threads,dynamic,predicted,most);
	return(SparkGpuFree(active) != SPARK_GPU_SUCCESS || SparkGpuFree(peak) != SPARK_GPU_SUCCESS);
}

typedef struct AbKernel
{
	const char *name;
	const void *function;
	uint32_t dynamic;
} AbKernel;

int main(void)
{
	static const int threads[] = {32, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024};
	static const uint32_t shared[] = {0u, 1024u, 4096u, 9000u, 12288u, 24576u, 32768u, 49152u, 65536u, 98304u};
	const AbKernel kernels[] = {
		{"light", (const void *)AbLight, 0u},
		{"heavy", (const void *)AbHeavy, 0u},
		{"static", (const void *)AbStatic, 0u},
		{"dynamic", (const void *)AbDynamic, 1u},
	};
	uint32_t kernel,thread,size;
	if ( SparkGpuFuncSetAttribute(AbDynamic,SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)AB_DYNAMIC_GRANT) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL grant dynamic shared memory: %s\n",SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	static const int resident_threads[] = {32, 128, 256, 512};
	static const uint32_t resident_shared[] = {0u, 8192u, 20480u, 40960u};
	int sms = 0;
	if ( SparkGpuFuncSetAttribute(AbResident,SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)AB_DYNAMIC_GRANT) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&sms,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,0) != SPARK_GPU_SUCCESS || sms <= 0 )
	{
		printf("FAIL set up the residency kernel\n");
		return(1);
	}
	for (thread = 0u; thread < sizeof(resident_threads) / sizeof(resident_threads[0]); thread++)
		for (size = 0u; size < sizeof(resident_shared) / sizeof(resident_shared[0]); size++)
			if ( Residency(sms,resident_threads[thread],resident_shared[size]) != 0 )
			{
				printf("FAIL residency at %d threads and %u bytes\n",resident_threads[thread],resident_shared[size]);
				return(1);
			}
	for (kernel = 0u; kernel < sizeof(kernels) / sizeof(kernels[0]); kernel++)
	{
		SparkGpuFunctionAttributes attributes;
		if ( SparkGpuFuncGetAttributes(&attributes,kernels[kernel].function) != SPARK_GPU_SUCCESS )
		{
			printf("FAIL attributes of %s\n",kernels[kernel].name);
			return(1);
		}
		printf("kernel %s registers %d static %zu\n",kernels[kernel].name,attributes.numRegs,attributes.sharedSizeBytes);
		for (thread = 0u; thread < sizeof(threads) / sizeof(threads[0]); thread++)
			for (size = 0u; size < (kernels[kernel].dynamic != 0u ? sizeof(shared) / sizeof(shared[0]) : 1u); size++)
			{
				int blocks = -1;
				const SparkGpuError error = SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernels[kernel].function,threads[thread],shared[size]);
				printf("occupancy %s threads %d dynamic %u blocks %d error %d\n",kernels[kernel].name,threads[thread],shared[size],
					error == SPARK_GPU_SUCCESS ? blocks : -1,(int)error);
			}
	}
	return(0);
}
