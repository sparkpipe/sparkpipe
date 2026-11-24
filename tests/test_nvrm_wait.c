#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
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
#define WAIT_PREDICTED_NS 5000000u

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

static void *StorePredicted(void *argument)
{
	LateStore *store = (LateStore *)argument;
	const struct timespec pause = {0, WAIT_PREDICTED_NS};
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

static void TestPredict(void)
{
	SparkNvrmWaitHistory history;
	SparkNvrmWait wait;
	uint32_t index;
	memset(&history,0,sizeof(history));
	SparkNvrmWaitBegin(&wait);
	CHECK(SparkNvrmWaitPredict(&history) == 0u,"no prediction without a history");
	for (index = 0u; index < SPARK_NVRM_WAIT_HISTORY; index++)
		history.waited[history.next++ % SPARK_NVRM_WAIT_HISTORY] = 3000000u + 10000u * index, history.count++;
	CHECK(SparkNvrmWaitPredict(&history) == 3000000u,"four steady waits predict the shortest of them");
	history.waited[2] = 4000000u;
	CHECK(SparkNvrmWaitPredict(&history) == 0u,"waits that spread by more than an eighth predict nothing");
	for (index = 0u; index < SPARK_NVRM_WAIT_HISTORY; index++)
		history.waited[index] = 120000u;
	CHECK(SparkNvrmWaitPredict(&history) == 0u,"waits inside the spin budget and the early margin predict nothing");
	history.count = 0u;
	history.next = 0u;
	wait.start = SparkMonotonicNs() - 2000000u;
	SparkNvrmWaitLearn(&history,&wait);
	CHECK(history.count == 1u && history.waited[0] >= 2000000u,"a finished wait is learned from its start");
	SparkNvrmWaitEnd(&wait);
}

static void TestPredictedNaps(void)
{
	LateStore store = {0u};
	SparkNvrmWait wait;
	pthread_t storer;
	uint64_t start,woke;
	uint32_t watched = 0u;
	int expired = 0;
	CHECK(pthread_create(&storer,0,StorePredicted,&store) == 0,"start the predicted storer");
	start = SparkMonotonicNs();
	SparkNvrmWaitBegin(&wait);
	wait.start = start;
	wait.predicted_ns = WAIT_PREDICTED_NS;
	while ( __atomic_load_n(&store.value,__ATOMIC_ACQUIRE) == 0u && (expired = SparkNvrmWaitPace(&wait,SPARK_DEVICE_WAIT_FOREVER)) == 0 )
		if ( SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) != 0 )
			watched++;
	woke = SparkMonotonicNs() - start;
	SparkNvrmWaitEnd(&wait);
	pthread_join(storer,0);
	CHECK(expired == 0 && store.value == 1u,"a predicted wait still sees the value");
	CHECK(wait.sleeps >= 2u,"a predicted 5 ms wait naps more than once, at most 1 ms a nap");
	CHECK(woke >= WAIT_LATE_STORE_NS && woke < WAIT_HANG_NS,"the predicted wait ends after the value lands");
	CHECK(watched >= 3u,"the fault watch stays due about every millisecond while a predicted wait naps");
}

static void TestPredictedSpin(void)
{
	SparkNvrmWait wait;
	SparkNvrmWaitBegin(&wait);
	wait.predicted_ns = WAIT_PREDICTED_NS;
	wait.start = SparkMonotonicNs() - WAIT_PREDICTED_NS;
	CHECK(SparkNvrmWaitPace(&wait,SPARK_DEVICE_WAIT_FOREVER) == 0 && wait.sleeps == 0u && SparkNvrmWaitDue(&wait,SPARK_NVRM_WAIT_WATCH_NS) == 0,
		"at its predicted end a wait spins without a nap and without looking for faults");
	SparkNvrmWaitEnd(&wait);
}

int main(void)
{
	TestZeroTimeout();
	TestSpinThenBackoff();
	TestTimeout();
	TestWakesOnValue();
	TestPredict();
	TestPredictedNaps();
	TestPredictedSpin();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS the nvrm waiter spins for its budget, then sleeps with low timer slack and backs off, keeps its timeout, and naps at most 1 ms at a time toward a learned wait\n");
	return(0);
}
