#include <string.h>
#include <time.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include "sparkpipe/spark_clock.h"
#include "runtime/device/nvrm/nvrm_wait.h"

void SparkNvrmWaitBegin(SparkNvrmWait *wait)
{
	memset(wait,0,sizeof(*wait));
}

static void WaitLowerSlack(SparkNvrmWait *wait)
{
#if defined(__linux__)
	const int slack = prctl(PR_GET_TIMERSLACK,0,0,0,0);
	if ( slack > (int)SPARK_NVRM_WAIT_SLACK_NS && prctl(PR_SET_TIMERSLACK,(unsigned long)SPARK_NVRM_WAIT_SLACK_NS,0,0,0) == 0 )
		wait->slack = slack;
#else
	(void)wait;
#endif
}

int SparkNvrmWaitPace(SparkNvrmWait *wait, uint64_t timeout_ns)
{
	struct timespec pause;
	uint64_t now,sleep_ns;
	wait->napping = 0u;
	if ( wait->sleep_ns == 0u && (wait->polls++ % SPARK_NVRM_WAIT_CLOCK_POLLS) != 0u )
		return(0);
	now = SparkMonotonicNs();
	if ( wait->start == 0u )
		wait->start = now;
	wait->elapsed = now - wait->start;
	if ( wait->elapsed >= timeout_ns )
		return(1);
	if ( wait->predicted_ns != 0u && wait->sleep_ns == 0u )
	{
		const uint64_t early = SPARK_NVRM_WAIT_EARLY_NS + wait->predicted_ns / SPARK_NVRM_WAIT_EARLY_SHARE;
		if ( wait->elapsed + early < wait->predicted_ns )
		{
			uint64_t nap = wait->predicted_ns - early - wait->elapsed;
			nap = nap < SPARK_NVRM_WAIT_NAP_NS ? nap : SPARK_NVRM_WAIT_NAP_NS;
			nap = nap < timeout_ns - wait->elapsed ? nap : timeout_ns - wait->elapsed;
			if ( wait->sleeps == 0u )
				WaitLowerSlack(wait);
			pause.tv_sec = (time_t)(nap / 1000000000ull);
			pause.tv_nsec = (long)(nap % 1000000000ull);
			nanosleep(&pause,0);
			wait->sleeps++;
			wait->napping = 1u;
			return(0);
		}
		if ( wait->elapsed < wait->predicted_ns + (wait->predicted_ns >> 3) + early )
			return(0);
	}
	else if ( wait->elapsed < SPARK_NVRM_WAIT_SPIN_NS )
		return(0);
	if ( wait->sleep_ns == 0u )
	{
		wait->sleep_ns = SPARK_NVRM_WAIT_SLEEP_FIRST_NS;
		WaitLowerSlack(wait);
	}
	sleep_ns = timeout_ns - wait->elapsed < wait->sleep_ns ? timeout_ns - wait->elapsed : wait->sleep_ns;
	pause.tv_sec = 0;
	pause.tv_nsec = (long)sleep_ns;
	nanosleep(&pause,0);
	wait->sleeps++;
	wait->sleep_ns = wait->sleep_ns * 2u < SPARK_NVRM_WAIT_SLEEP_MAX_NS ? wait->sleep_ns * 2u : SPARK_NVRM_WAIT_SLEEP_MAX_NS;
	return(0);
}

int SparkNvrmWaitDue(SparkNvrmWait *wait, uint64_t every_ns)
{
	if ( (wait->sleep_ns == 0u && wait->napping == 0u) || (wait->watched != 0u && wait->elapsed - wait->watched < every_ns) )
		return(0);
	wait->watched = wait->elapsed;
	return(1);
}

int SparkNvrmWaitStalled(SparkNvrmWait *wait, uint64_t progress, uint64_t stall_ns)
{
	if ( wait->tracking == 0u || progress != wait->progress )
	{
		wait->tracking = 1u;
		wait->progress = progress;
		wait->progress_at = wait->elapsed;
		return(0);
	}
	return(wait->elapsed - wait->progress_at >= stall_ns ? 1 : 0);
}

void SparkNvrmWaitEnd(SparkNvrmWait *wait)
{
#if defined(__linux__)
	if ( wait->slack > 0 )
		(void)prctl(PR_SET_TIMERSLACK,(unsigned long)wait->slack,0,0,0);
#endif
	wait->slack = 0;
}

uint64_t SparkNvrmWaitPredict(const SparkNvrmWaitHistory *history)
{
	uint64_t low = UINT64_MAX,high = 0u;
	uint32_t index;
	if ( history->count < SPARK_NVRM_WAIT_HISTORY )
		return(0u);
	for ( index = 0u; index < SPARK_NVRM_WAIT_HISTORY; index++ )
	{
		low = history->waited[index] < low ? history->waited[index] : low;
		high = history->waited[index] > high ? history->waited[index] : high;
	}
	if ( low < SPARK_NVRM_WAIT_SPIN_NS + SPARK_NVRM_WAIT_EARLY_NS || high - low > (low >> 3) + 20000u )
		return(0u);
	return(low);
}

void SparkNvrmWaitLearn(SparkNvrmWaitHistory *history, const SparkNvrmWait *wait)
{
	if ( wait->start == 0u )
		return;
	history->waited[history->next] = SparkMonotonicNs() - wait->start;
	history->next = (history->next + 1u) % SPARK_NVRM_WAIT_HISTORY;
	if ( history->count < SPARK_NVRM_WAIT_HISTORY )
		history->count++;
}
