#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_wait.h"

#define WAIT_TIMEOUT_NS 150000u
#define WAIT_HANG_NS 10000000000ull
#define WAIT_LATE_STORE_NS 2000000u

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

static int Slack(void)
{
#if defined(__linux__)
	return(prctl(PR_GET_TIMERSLACK,0,0,0,0));
#else
	return(-1);
#endif
}

static void TestZeroTimeout(void)
{
	SparkNvrmWait wait;
	SparkNvrmWaitBegin(&wait);
	CHECK(SparkNvrmWaitPace(&wait,0u) == 1 && wait.sleeps == 0u,"a zero timeout expires at the first pace, without sleeping");
	SparkNvrmWaitEnd(&wait);
}

static void TestSpinThenBackoff(void)
{
	static const uint32_t expected[] = {2000u, 4000u, 8000u, 8000u, 8000u};
	SparkNvrmWait wait;
	const int before = Slack();
	uint32_t index = 0u,slept_early = 0u,wrong_step = 0u;
	int lowered = 0;
	SparkNvrmWaitBegin(&wait);
	while ( index < sizeof(expected) / sizeof(expected[0]) )
	{
		const uint32_t sleeps = wait.sleeps,step = wait.sleep_ns;
		(void)SparkNvrmWaitPace(&wait,SPARK_DEVICE_WAIT_FOREVER);
		if ( wait.sleeps == sleeps )
			continue;
		if ( wait.elapsed < SPARK_NVRM_WAIT_SPIN_NS )
			slept_early++;
		if ( (sleeps == 0u ? SPARK_NVRM_WAIT_SLEEP_FIRST_NS : step) != expected[index] )
			wrong_step++;
		if ( index == 0u )
			lowered = Slack();
		index++;
	}
	SparkNvrmWaitEnd(&wait);
	CHECK(slept_early == 0u,"the waiter never sleeps before its spin budget has passed");
	CHECK(wrong_step == 0u,"after the spin budget the sleeps double from 2 us to an 8 us ceiling");
	if ( before >= 0 )
	{
		CHECK(before <= (int)SPARK_NVRM_WAIT_SLACK_NS || lowered == (int)SPARK_NVRM_WAIT_SLACK_NS,"the thread sleeps with a 1 us timer slack");
		CHECK(Slack() == before,"the thread's timer slack is restored when the wait ends");
	}
}

typedef struct LateStore
{
	volatile uint64_t value;
} LateStore;

static void *StoreLate(void *argument)
{
	LateStore *store = (LateStore *)argument;
	const struct timespec pause = {0, WAIT_LATE_STORE_NS};
	nanosleep(&pause,0);
	__atomic_store_n(&store->value,1u,__ATOMIC_RELEASE);
	return(0);
}

static void TestTimeout(void)
{
	LateStore store = {0u};
	SparkNvrmWait wait;
	pthread_t storer;
	uint64_t start,took;
	int expired = 0;
	CHECK(pthread_create(&storer,0,StoreLate,&store) == 0,"start the late storer");
	start = SparkMonotonicNs();
	SparkNvrmWaitBegin(&wait);
	while ( (expired = SparkNvrmWaitPace(&wait,WAIT_TIMEOUT_NS)) == 0 )
		;
	SparkNvrmWaitEnd(&wait);
	took = SparkMonotonicNs() - start;
	CHECK(expired == 1 && took >= WAIT_TIMEOUT_NS,"a finite timeout reports that it expired, never before its time");
	CHECK(took < WAIT_HANG_NS,"a finite timeout returns even on a loaded host");
	expired = 0;
	SparkNvrmWaitBegin(&wait);
	while ( __atomic_load_n(&store.value,__ATOMIC_ACQUIRE) == 0u && (expired = SparkNvrmWaitPace(&wait,SPARK_DEVICE_WAIT_FOREVER)) == 0 )
		;
	SparkNvrmWaitEnd(&wait);
	pthread_join(storer,0);
	CHECK(expired == 0 && store.value == 1u,"a completion after the timed-out wait is still seen by the next wait");
}

static void TestWakesOnValue(void)
{
	LateStore store = {0u};
	SparkNvrmWait wait;
	pthread_t storer;
	int expired = 0;
	CHECK(pthread_create(&storer,0,StoreLate,&store) == 0,"start the late storer");
	SparkNvrmWaitBegin(&wait);
	while ( __atomic_load_n(&store.value,__ATOMIC_ACQUIRE) == 0u && (expired = SparkNvrmWaitPace(&wait,SPARK_DEVICE_WAIT_FOREVER)) == 0 )
		;
	SparkNvrmWaitEnd(&wait);
	pthread_join(storer,0);
	CHECK(expired == 0 && store.value == 1u,"a wait that outlives the spin budget still sees the value");
}

int main(void)
{
	TestZeroTimeout();
	TestSpinThenBackoff();
	TestTimeout();
	TestWakesOnValue();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS the nvrm waiter spins for its budget, then sleeps with low timer slack and backs off, and keeps its timeout\n");
	return(0);
}
