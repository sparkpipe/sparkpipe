#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define FAULTS_UNMAPPED 0x7f0000001000ull
#define FAULTS_PROMPT_NS 1000000000ull
#define FAULTS_STALL_NS 10000000000ull
#define FAULTS_STALL_LATE_NS 3000000000ull
#define FAULTS_RELEASE_NS 2000000000ull
#define FAULTS_WORDS 16u
#define FAULTS_PLAIN_VALUE 42ull

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

extern "C" __global__ void FaultsTrap(unsigned long long *out)
{
	out[0] = 1ull;
	asm volatile("trap;");
	out[1] = 1ull;
}

extern "C" __global__ void FaultsStore(unsigned long long *out, unsigned long long *target)
{
	out[0] = 2ull;
	*(volatile unsigned long long *)target = 7ull;
}

extern "C" __global__ void FaultsSpin(unsigned long long *out, const unsigned long long *flag)
{
	unsigned long long seen;
	out[0] = 3ull;
	do
		asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(seen) : "l"(flag) : "memory");
	while ( seen == 0ull );
	out[1] = 3ull;
}

extern "C" __global__ void FaultsPlain(unsigned long long *out, unsigned long long value)
{
	out[0] = value;
}

static uint64_t FaultsNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static void FaultsShow(const char *what, const SparkDeviceError *report)
{
	printf("%s: status %d, operation %s, queue '%s', code %d: %s\n",what,(int)report->status,report->operation,report->owner,(int)report->backend_code,report->text);
}

static int FaultsReported(const SparkDeviceError *report, const char *operation, int32_t code, const char *head, const char *position)
{
	return(report->status == SPARK_STATUS_IO_ERROR && strcmp(report->operation,operation) == 0 && report->backend_code == code &&
		strncmp(report->owner,"gpu stream ",11) == 0 && strncmp(report->text,head,strlen(head)) == 0 && strstr(report->text,position) != 0);
}

static void FaultsLater(void)
{
	void *more = 0;
	CHECK(SparkGpuMalloc(&more,64u) != SPARK_GPU_SUCCESS && SparkDeviceLastError()->status == SPARK_STATUS_IO_ERROR &&
		strstr(SparkDeviceLastError()->text,"faulted earlier") != 0,"after a fault every later operation fails");
}

static int FaultsPlainRun(SparkGpuStream stream, unsigned long long *out)
{
	LM_LAUNCH((FaultsPlain),1,1,0,stream,out,FAULTS_PLAIN_VALUE);
	CHECK(SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS && out[0] == FAULTS_PLAIN_VALUE,"a plain kernel runs and its stream synchronizes");
	CHECK(SparkGpuStreamDestroy(stream) == SPARK_GPU_SUCCESS && SparkGpuFreeHost(out) == SPARK_GPU_SUCCESS && SparkNvrmGpuRelease() == SPARK_GPU_SUCCESS,
		"release the stream and the device");
	return(failures == 0 ? 0 : 1);
}

static void FaultsTrapRun(SparkGpuStream stream, unsigned long long *out)
{
	uint64_t start,took;
	uint32_t index;
	LM_LAUNCH((FaultsTrap),1,1,0,stream,out);
	for (index = 0u; index < 3u; index++)
		LM_LAUNCH((FaultsPlain),1,1,0,stream,out + 2u + index,FAULTS_PLAIN_VALUE);
	start = FaultsNow();
	CHECK(SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS,"synchronizing a queue whose kernel trapped fails");
	took = FaultsNow() - start;
	FaultsShow("trap",SparkDeviceLastError());
	CHECK(FaultsReported(SparkDeviceLastError(),"queue_wait",13,"GPU fault 13 (graphics engine exception)","; at kernel FaultsTrap; last launched FaultsPlain"),
		"the report names the queue, the trapping kernel, the last kernel launched and the error notifier's code");
	CHECK(took < FAULTS_PROMPT_NS,"the trap is reported within a second, not at the stall deadline");
	CHECK(out[2] == 0ull,"no kernel after the trap ran");
	FaultsLater();
}

static void FaultsAddressRun(SparkGpuStream stream, unsigned long long *out)
{
	SparkGpuEvent event = 0;
	char position[64];
	CHECK(SparkGpuEventCreate(&event) == SPARK_GPU_SUCCESS,"create an event");
	LM_LAUNCH((FaultsStore),1,1,0,stream,out,(unsigned long long *)FAULTS_UNMAPPED);
	CHECK(SparkGpuEventRecord(event,stream) == SPARK_GPU_SUCCESS,"record an event after the bad store");
	CHECK(SparkGpuEventSynchronize(event) != SPARK_GPU_SUCCESS,"waiting for an event behind a bad address fails");
	FaultsShow("address",SparkDeviceLastError());
	snprintf(position,sizeof(position),"at %#llx)",FAULTS_UNMAPPED);
	CHECK(FaultsReported(SparkDeviceLastError(),"event_wait",31,"GPU fault 31 (MMU fault, ",position),
		"an event wait reports the MMU fault with the module's fault type and address");
	CHECK(strstr(SparkDeviceLastError()->text,"; at kernel FaultsStore, the last one launched") != 0,"and names the faulting kernel of the event's queue");
	CHECK(SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS && SparkDeviceLastError()->status == SPARK_STATUS_IO_ERROR,
		"the queue's own wait fails after the fault");
	FaultsLater();
}

static void FaultsStallRun(SparkGpuStream stream, unsigned long long *out)
{
	unsigned long long *flag = 0;
	uint64_t start,took;
	CHECK(SparkGpuMallocHost(&flag,64u) == SPARK_GPU_SUCCESS,"allocate the release flag");
	if ( flag == 0 )
		return;
	__atomic_store_n(flag,0ull,__ATOMIC_RELEASE);
	LM_LAUNCH((FaultsSpin),1,1,0,stream,out,flag);
	start = FaultsNow();
	CHECK(SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS,"synchronizing a queue that makes no progress fails at the deadline");
	took = FaultsNow() - start;
	FaultsShow("stall",SparkDeviceLastError());
	printf("stall: reported after %.2f s\n",(double)took / 1e9);
	CHECK(FaultsReported(SparkDeviceLastError(),"queue_wait",0,"no GPU progress for 10 s","; at kernel FaultsSpin, the last one launched"),
		"the stall report names the queue and the kernel it stopped at");
	CHECK(took >= FAULTS_STALL_NS && took < FAULTS_STALL_NS + FAULTS_STALL_LATE_NS,"the stall is reported once the deadline has passed with no progress");
	__atomic_store_n(flag,1ull,__ATOMIC_RELEASE);
	start = FaultsNow();
	while ( __atomic_load_n(&out[1],__ATOMIC_ACQUIRE) != 3ull && FaultsNow() - start < FAULTS_RELEASE_NS )
		;
	CHECK(__atomic_load_n(&out[1],__ATOMIC_ACQUIRE) == 3ull,"the stalled kernel finishes once released");
	FaultsLater();
}

int main(int argc, char **argv)
{
	const char *mode = argc > 1 ? argv[1] : "";
	SparkGpuStream stream = 0;
	unsigned long long *out = 0;
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuMallocHost(&out,FAULTS_WORDS * sizeof(*out)) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL set up the stream and the output: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	memset(out,0,FAULTS_WORDS * sizeof(*out));
	if ( strcmp(mode,"plain") == 0 )
		return(FaultsPlainRun(stream,out));
	if ( strcmp(mode,"trap") == 0 )
		FaultsTrapRun(stream,out);
	else if ( strcmp(mode,"address") == 0 )
		FaultsAddressRun(stream,out);
	else if ( strcmp(mode,"stall") == 0 )
		FaultsStallRun(stream,out);
	else
	{
		printf("usage: %s plain | trap | address | stall\n",argv[0]);
		return(2);
	}
	if ( failures != 0 )
	{
		printf("\nFAIL %s (%d)\n",mode,failures);
		return(1);
	}
	printf("PASS %s: the wait failed with SPARK_STATUS_IO_ERROR naming the queue, the kernel and the code\n",mode);
	return(0);
}
