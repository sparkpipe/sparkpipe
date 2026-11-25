#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define HV_GROUPS 48u
#define HV_THREADS 256u
#define HV_WORDS 16384u
#define HV_BYTES (HV_WORDS * sizeof(uint32_t))
#define HV_HOST_GRANULE (2ull << 20)
#define HV_ROUNDS 100u
#define HV_SPIN_NS 20000ull
#define HV_MUTANT_SPIN_NS 50000ull

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s: %s / %s\n",what,SparkGpuGetErrorString(SparkGpuGetLastError()),SparkDeviceLastError()->text); \
			failures++; \
		} \
	} while (0)

static __host__ __device__ uint32_t HvExpected(uint32_t seed, uint32_t index)
{
	uint32_t value = (seed * 2654435761u) ^ (index * 2246822519u) ^ 0x9e3779b9u;
	value ^= value >> 16;
	value *= 0x85ebca6bu;
	value ^= value >> 13;
	return(value);
}

static __host__ __device__ uint32_t HvTransformA(uint32_t value, uint32_t salt)
{
	return(value * 3u + salt);
}

static __host__ __device__ uint32_t HvTransformB(uint32_t value, uint32_t salt)
{
	return((value ^ (value >> 7)) * 5u + salt);
}

__global__ void HvWritePayload(uint32_t *payload, uint32_t seed, uint64_t spin_ns)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t stride = gridDim.x * blockDim.x;
	uint32_t index;
	if ( blockIdx.x == 0u && threadIdx.x == 0u && spin_ns != 0u )
	{
		uint64_t start,now;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
		do
			asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
		while ( now - start < spin_ns );
	}
	__syncthreads();
	for ( index = tid; index < HV_WORDS; index += stride )
		payload[index] = HvExpected(seed,index);
}

__global__ void HvMutantWritePayload(uint32_t *payload, volatile uint32_t *flag, uint32_t seed, uint32_t round, uint64_t spin_ns)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t stride = gridDim.x * blockDim.x;
	uint32_t index;
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
	{
		__threadfence_system();
		*flag = round;
	}
	if ( threadIdx.x == 0u && spin_ns != 0u )
	{
		uint64_t start,now;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
		do
			asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
		while ( now - start < spin_ns );
	}
	__syncthreads();
	for ( index = tid; index < HV_WORDS; index += stride )
		payload[index] = HvExpected(seed,index);
}

__global__ void HvTransformPlain(uint32_t *payload, uint32_t salt)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t stride = gridDim.x * blockDim.x;
	uint32_t index;
	for ( index = tid; index < HV_WORDS; index += stride )
		payload[index] = HvTransformA(payload[index],salt);
}

__global__ void HvTransformDependent(uint32_t *payload, uint32_t salt)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t stride = gridDim.x * blockDim.x;
	uint32_t index;
	asm volatile("griddepcontrol.wait;" ::: "memory");
	for ( index = tid; index < HV_WORDS; index += stride )
		payload[index] = HvTransformB(payload[index],salt);
}

__global__ void HvTouchDevice(uint32_t *scratch, uint32_t value)
{
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
		scratch[0] = scratch[0] + value;
}

__global__ void HvSignalPlain(volatile uint32_t *flag, uint32_t round, uint32_t sys_fence)
{
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
	{
		if ( sys_fence != 0u )
			__threadfence_system();
		*flag = round;
	}
}

__global__ void HvSignalDependent(volatile uint32_t *flag, uint32_t round, uint32_t sys_fence)
{
	asm volatile("griddepcontrol.wait;" ::: "memory");
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
	{
		if ( sys_fence != 0u )
			__threadfence_system();
		*flag = round;
	}
}

typedef struct HvRegion
{
	const char *name;
	uint32_t kind;
	SparkDeviceBuffer buffer;
	SparkDeviceMemoryHandle memory;
	SparkDeviceAddress address;
	void *raw_host;
	uint32_t *host;
	uint32_t *device;
} HvRegion;

static int HvRegionOpen(HvRegion *region, uint32_t kind)
{
	memset(region,0,sizeof(*region));
	region->kind = kind;
	if ( kind == 0u )
	{
		region->name = "host-pinned";
		if ( SparkDeviceAllocate(&region->buffer,SPARK_DEVICE_SPACE_HOST_PINNED,HV_BYTES + 64u,region->name) != SPARK_STATUS_OK )
			return(0);
		region->host = (uint32_t *)region->buffer.host;
		region->device = (uint32_t *)(uintptr_t)region->buffer.device;
		return(1);
	}
	if ( kind == 1u )
	{
		region->name = "shared-coherent";
		if ( SparkDeviceAllocate(&region->buffer,SPARK_DEVICE_SPACE_SHARED_COHERENT,HV_BYTES + 64u,region->name) != SPARK_STATUS_OK )
			return(0);
		region->host = (uint32_t *)region->buffer.host;
		region->device = (uint32_t *)(uintptr_t)region->buffer.device;
		return(1);
	}
	if ( kind == 2u )
	{
		region->name = "registered-host";
		if ( posix_memalign(&region->raw_host,65536u,HV_BYTES + 64u) != 0 )
			return(0);
		memset(region->raw_host,0,HV_BYTES + 64u);
		if ( SparkDeviceRegister(&region->buffer,region->raw_host,HV_BYTES + 64u,
			SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE | SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,region->name) != SPARK_STATUS_OK )
		{
			free(region->raw_host);
			region->raw_host = 0;
			return(0);
		}
		region->host = (uint32_t *)region->buffer.host;
		region->device = (uint32_t *)(uintptr_t)region->buffer.device;
		return(1);
	}
	region->name = "created-host";
	if ( SparkDeviceMemoryCreateHost(&region->memory,HV_HOST_GRANULE,region->name) != SPARK_STATUS_OK )
		return(0);
	if ( SparkDeviceAddressReserve(&region->address,HV_HOST_GRANULE,region->name) != SPARK_STATUS_OK )
	{
		SparkDeviceMemoryRelease(region->memory);
		return(0);
	}
	if ( SparkDeviceMemoryMap(region->address,HV_HOST_GRANULE,region->memory,0u,
		SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE | SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE) != SPARK_STATUS_OK )
	{
		SparkDeviceAddressRelease(region->address,HV_HOST_GRANULE);
		SparkDeviceMemoryRelease(region->memory);
		return(0);
	}
	region->host = (uint32_t *)(uintptr_t)region->address;
	region->device = (uint32_t *)(uintptr_t)region->address;
	return(1);
}

static void HvRegionClose(HvRegion *region)
{
	if ( region->kind < 3u )
	{
		SparkDeviceFree(&region->buffer);
		free(region->raw_host);
	}
	else
	{
		SparkDeviceMemoryUnmap(region->address,HV_HOST_GRANULE);
		SparkDeviceAddressRelease(region->address,HV_HOST_GRANULE);
		SparkDeviceMemoryRelease(region->memory);
	}
}

typedef struct HvCallbackState
{
	const uint32_t *host;
	uint32_t seed;
	uint32_t salt_a;
	uint32_t salt_b;
	uint64_t bad_words;
	uint32_t calls;
} HvCallbackState;

static void HvCallbackCheck(void *context)
{
	HvCallbackState *state = (HvCallbackState *)context;
	uint32_t index;
	uint64_t bad = 0u;
	for ( index = 0u; index < HV_WORDS; index++ )
	{
		const uint32_t want = HvTransformB(HvTransformA(HvExpected(state->seed,index),state->salt_a),state->salt_b);
		if ( state->host[index] != want )
			bad++;
	}
	state->bad_words += bad;
	state->calls++;
}

typedef struct HvPoller
{
	const volatile uint32_t *flag;
	const uint32_t *payload;
	uint32_t rounds;
	uint32_t seed_base;
	uint64_t stale_words;
	uint64_t stale_rounds;
	volatile uint32_t ready_round;
	volatile uint32_t done_round;
} HvPoller;

static void *HvPollThread(void *argument)
{
	HvPoller *poller = (HvPoller *)argument;
	uint32_t round;
	for ( round = 1u; round <= poller->rounds; round++ )
	{
		uint32_t index,sink = 0u;
		uint64_t bad = 0u;
		while ( __atomic_load_n(&poller->ready_round,__ATOMIC_ACQUIRE) < round )
			;
		for ( index = 0u; index < HV_WORDS; index += 16u )
			sink ^= poller->payload[index];
		(void)sink;
		while ( __atomic_load_n(poller->flag,__ATOMIC_ACQUIRE) < round )
			;
		for ( index = 0u; index < HV_WORDS; index++ )
			if ( poller->payload[index] != HvExpected(poller->seed_base + round,index) )
				bad++;
		poller->stale_words += bad;
		if ( bad != 0u )
			poller->stale_rounds++;
		__atomic_store_n(&poller->done_round,round,__ATOMIC_RELEASE);
	}
	return(0);
}

static void TestQueueWaitAndCallback(SparkGpuStream stream, SparkDeviceQueue *queue, HvRegion *region, uint32_t *scratch)
{
	SparkDeviceProgram *program = 0;
	HvCallbackState cb;
	uint32_t round,index;
	uint64_t wait_bad = 0u;
	memset(&cb,0,sizeof(cb));
	cb.host = region->host;
	cb.seed = 777u;
	cb.salt_a = 19u;
	cb.salt_b = 43u;
	CHECK(SparkDeviceProgramRecord(queue,"hv chain") == SPARK_STATUS_OK,"begin recording hv chain");
	LM_LAUNCH((HvWritePayload),HV_GROUPS,HV_THREADS,0u,stream,region->device,cb.seed,(uint64_t)HV_SPIN_NS);
	LM_LAUNCH((HvTransformPlain),HV_GROUPS,HV_THREADS,0u,stream,region->device,cb.salt_a);
	CHECK(LM_LAUNCH_DEPENDENT((HvTransformDependent),HV_GROUPS,HV_THREADS,0u,stream,region->device,cb.salt_b) == SPARK_GPU_SUCCESS,"record dependent transform");
	LM_LAUNCH((HvTouchDevice),1,32,0u,stream,scratch,1u);
	CHECK(SparkGpuLaunchHostFunc(stream,HvCallbackCheck,&cb) == SPARK_GPU_SUCCESS,"record host callback");
	LM_LAUNCH((HvTouchDevice),1,32,0u,stream,scratch,2u);
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK,"finish hv chain program");
	for ( round = 0u; round < HV_ROUNDS; round++ )
	{
		for ( index = 0u; index < HV_WORDS; index++ )
			region->host[index] = 0xdead0000u | round;
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		CHECK(SparkDeviceProgramRun(program,queue,0,0u) == SPARK_STATUS_OK,"run hv chain program");
		CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait hv chain program");
		for ( index = 0u; index < HV_WORDS; index++ )
		{
			const uint32_t want = HvTransformB(HvTransformA(HvExpected(cb.seed,index),cb.salt_a),cb.salt_b);
			if ( region->host[index] != want )
				wait_bad++;
		}
	}
	printf("  %-16s program chain+callback (%u rounds x %u words): queue-wait stale %llu, callback stale %llu (calls %u)\n",
		region->name,HV_ROUNDS,HV_WORDS,(unsigned long long)wait_bad,(unsigned long long)cb.bad_words,cb.calls);
	CHECK(wait_bad == 0u && cb.bad_words == 0u && cb.calls == HV_ROUNDS,"queue wait and host callback see every chained write");
	CHECK(SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK,"destroy hv chain program");
}

static void TestHostPoll(SparkGpuStream stream, SparkDeviceQueue *queue, HvRegion *region, uint32_t dependent, uint32_t sys_fence, uint32_t mutant)
{
	volatile uint32_t *flag_host = region->host + HV_WORDS;
	volatile uint32_t *flag_dev = region->device + HV_WORDS;
	HvPoller poller;
	pthread_t thread;
	uint32_t round,index;
	const uint32_t seed_base = 10000u + dependent * 1000u + sys_fence * 100u + mutant * 5000u;
	memset(&poller,0,sizeof(poller));
	poller.flag = flag_host;
	poller.payload = region->host;
	poller.rounds = HV_ROUNDS;
	poller.seed_base = seed_base;
	*flag_host = 0u;
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	CHECK(pthread_create(&thread,0,HvPollThread,&poller) == 0,"spawn CPU poller");
	for ( round = 1u; round <= HV_ROUNDS; round++ )
	{
		SparkDeviceProgram *program = 0;
		for ( index = 0u; index < HV_WORDS; index++ )
			region->host[index] = 0xbad00000u | round;
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		CHECK(SparkDeviceProgramRecord(queue,"hv poll") == SPARK_STATUS_OK,"record poll program");
		if ( mutant != 0u )
			LM_LAUNCH((HvMutantWritePayload),HV_GROUPS,HV_THREADS,0u,stream,region->device,flag_dev,seed_base + round,round,(uint64_t)HV_MUTANT_SPIN_NS);
		else
		{
			LM_LAUNCH((HvWritePayload),HV_GROUPS,HV_THREADS,0u,stream,region->device,seed_base + round,(uint64_t)HV_SPIN_NS);
			if ( dependent != 0u )
				CHECK(LM_LAUNCH_DEPENDENT((HvSignalDependent),1,32,0u,stream,flag_dev,round,sys_fence) == SPARK_GPU_SUCCESS,"record dependent signal");
			else
				LM_LAUNCH((HvSignalPlain),1,32,0u,stream,flag_dev,round,sys_fence);
		}
		CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK,"finish poll program");
		__atomic_store_n(&poller.ready_round,round,__ATOMIC_RELEASE);
		CHECK(SparkDeviceProgramRun(program,queue,0,0u) == SPARK_STATUS_OK,"run poll program");
		while ( __atomic_load_n(&poller.done_round,__ATOMIC_ACQUIRE) < round )
			;
		CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait poll program");
		CHECK(SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK,"destroy poll program");
	}
	pthread_join(thread,0);
	printf("  %-16s poll %-10s %-10s: %llu stale words in %llu/%u rounds\n",
		region->name,mutant != 0u ? "MUTANT" : dependent != 0u ? "latch-dep" : "qmd-plain",
		mutant != 0u ? "early-flag" : sys_fence != 0u ? "sys-fence" : "no-fence",
		(unsigned long long)poller.stale_words,(unsigned long long)poller.stale_rounds,HV_ROUNDS);
	if ( mutant != 0u )
		CHECK(poller.stale_words > 0u,"mutant early-flag triggers stale reads");
	else
		CHECK(poller.stale_words == 0u,"CPU poll after chained flag sees zero stale payload words");
}

int main(int argc, char **argv)
{
	const char *flavour = argc > 1 ? argv[1] : "this";
	SparkGpuStream stream = 0;
	SparkDeviceQueue *queue = 0;
	SparkStatus initialized = SparkDeviceInitialize(0);
	uint32_t *scratch = 0,kind;
	CHECK((initialized == SPARK_STATUS_OK || initialized == SPARK_STATUS_DUPLICATE) &&
		SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS &&
		SparkGpuMalloc(&scratch,sizeof(uint32_t)) == SPARK_GPU_SUCCESS &&
		SparkGpuMemset(scratch,0,sizeof(uint32_t)) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS &&
		SparkDeviceQueueAdopt(&queue,(void *)stream,"host visibility") == SPARK_STATUS_OK,"initialize host visibility test");
	if ( failures != 0 )
		return(1);
	printf("host visibility on %s:\n",flavour);
	for ( kind = 0u; kind < 4u; kind++ )
	{
		HvRegion region;
		if ( HvRegionOpen(&region,kind) == 0 )
		{
			printf("  SKIP kind %u: %s\n",kind,SparkDeviceLastError()->text);
			if ( kind < 3u )
				failures++;
			continue;
		}
		TestQueueWaitAndCallback(stream,queue,&region,scratch);
		TestHostPoll(stream,queue,&region,0u,0u,0u);
		TestHostPoll(stream,queue,&region,0u,1u,0u);
		TestHostPoll(stream,queue,&region,1u,0u,0u);
		TestHostPoll(stream,queue,&region,1u,1u,0u);
		if ( kind == 0u )
			TestHostPoll(stream,queue,&region,0u,0u,1u);
		HvRegionClose(&region);
	}
	SparkDeviceQueueDestroy(queue);
	SparkGpuFree(scratch);
	SparkGpuStreamDestroy(stream);
	if ( failures != 0 )
	{
		printf("FAIL host visibility on %s (%d failures)\n",flavour,failures);
		return(1);
	}
	printf("PASS host visibility on %s: all host-visible spaces and chained kernels show zero stale CPU reads\n",flavour);
	return(0);
}
