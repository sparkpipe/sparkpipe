#ifndef SPARKPIPE_RUNTIME_DEVICE_NVRM_NVRM_WAIT_H
#define SPARKPIPE_RUNTIME_DEVICE_NVRM_NVRM_WAIT_H

#include <stdint.h>

#define SPARK_NVRM_WAIT_SPIN_NS 100000u
#define SPARK_NVRM_WAIT_CLOCK_POLLS 64u
#define SPARK_NVRM_WAIT_SLEEP_FIRST_NS 2000u
#define SPARK_NVRM_WAIT_SLEEP_MAX_NS 8000u
#define SPARK_NVRM_WAIT_SLACK_NS 1000u
#define SPARK_NVRM_WAIT_WATCH_NS 1000000u
#define SPARK_NVRM_WAIT_STALL_NS 10000000000ull
#define SPARK_NVRM_WAIT_EARLY_NS 50000u
#define SPARK_NVRM_WAIT_EARLY_SHARE 50u
#define SPARK_NVRM_WAIT_NAP_NS 1000000u
#define SPARK_NVRM_WAIT_HISTORY 4u

typedef struct SparkNvrmWait
{
	uint64_t start;
	uint64_t elapsed;
	uint64_t watched;
	uint64_t progress;
	uint64_t progress_at;
	uint32_t polls;
	uint32_t sleep_ns;
	uint32_t sleeps;
	uint32_t tracking;
	int slack;
	uint64_t predicted_ns;
	uint32_t napping;
} SparkNvrmWait;

typedef struct SparkNvrmWaitHistory
{
	uint64_t waited[SPARK_NVRM_WAIT_HISTORY];
	uint32_t count;
	uint32_t next;
} SparkNvrmWaitHistory;

void SparkNvrmWaitBegin(SparkNvrmWait *wait);
int SparkNvrmWaitPace(SparkNvrmWait *wait, uint64_t timeout_ns);
int SparkNvrmWaitDue(SparkNvrmWait *wait, uint64_t every_ns);
int SparkNvrmWaitStalled(SparkNvrmWait *wait, uint64_t progress, uint64_t stall_ns);
void SparkNvrmWaitEnd(SparkNvrmWait *wait);
uint64_t SparkNvrmWaitStallNs(void);
void SparkNvrmWaitStallSet(uint64_t stall_ns);
uint64_t SparkNvrmWaitPredict(const SparkNvrmWaitHistory *history);
void SparkNvrmWaitLearn(SparkNvrmWaitHistory *history, const SparkNvrmWait *wait);

#endif
