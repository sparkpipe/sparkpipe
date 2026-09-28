#pragma once

#include <stdint.h>

#define SPARK_STEP_POISON_TOKEN UINT32_MAX
#define SPARK_STEP_MISS_FLAG 0u
#define SPARK_STEP_MISS_COUNT 1u
#define SPARK_STEP_MISS_ENTRIES 2u

typedef enum SparkStepVerdict
{
	SPARK_STEP_VERDICT_COMMIT = 0,
	SPARK_STEP_VERDICT_ROLLBACK_LOCAL,
	SPARK_STEP_VERDICT_ROLLBACK_REMOTE,
	SPARK_STEP_VERDICT_MIXED,
	SPARK_STEP_VERDICT_POISON_LOST,
	SPARK_STEP_VERDICT_COUNT
} SparkStepVerdict;

static inline SparkStepVerdict SparkStepVerdictClassify(const uint32_t *tokens,uint32_t rows,uint32_t local_miss)
{
	uint32_t row,poisoned = 0u;
	for (row=0u; row<rows; row++)
		poisoned += tokens[row] == SPARK_STEP_POISON_TOKEN ? 1u : 0u;
	if ( rows == 0u || (poisoned != 0u && poisoned != rows) )
		return(SPARK_STEP_VERDICT_MIXED);
	if ( poisoned == 0u )
		return(local_miss != 0u ? SPARK_STEP_VERDICT_POISON_LOST : SPARK_STEP_VERDICT_COMMIT);
	return(local_miss != 0u ? SPARK_STEP_VERDICT_ROLLBACK_LOCAL : SPARK_STEP_VERDICT_ROLLBACK_REMOTE);
}

static inline const char *SparkStepVerdictName(SparkStepVerdict verdict)
{
	static const char *const names[SPARK_STEP_VERDICT_COUNT] = {"commit","rollback-local","rollback-remote","mixed","poison-lost"};
	return((uint32_t)verdict < SPARK_STEP_VERDICT_COUNT ? names[verdict] : "invalid");
}

typedef enum SparkStepAction
{
	SPARK_STEP_ACTION_COMMIT = 0,
	SPARK_STEP_ACTION_REPLAY,
	SPARK_STEP_ACTION_EXHAUSTED,
	SPARK_STEP_ACTION_FAIL
} SparkStepAction;

typedef struct SparkStepReplay
{
	uint32_t attempts;
	uint32_t limit;
	uint64_t commits;
	uint64_t local;
	uint64_t remote;
	uint64_t replays;
	uint64_t exhausted;
} SparkStepReplay;

static inline SparkStepAction SparkStepReplayNext(SparkStepReplay *replay,SparkStepVerdict verdict)
{
	if ( verdict == SPARK_STEP_VERDICT_COMMIT )
	{
		replay->attempts = 0u;
		replay->commits++;
		return(SPARK_STEP_ACTION_COMMIT);
	}
	if ( verdict != SPARK_STEP_VERDICT_ROLLBACK_LOCAL && verdict != SPARK_STEP_VERDICT_ROLLBACK_REMOTE )
		return(SPARK_STEP_ACTION_FAIL);
	replay->local += verdict == SPARK_STEP_VERDICT_ROLLBACK_LOCAL ? 1u : 0u;
	replay->remote += verdict == SPARK_STEP_VERDICT_ROLLBACK_REMOTE ? 1u : 0u;
	if ( replay->attempts < replay->limit )
	{
		replay->attempts++;
		replay->replays++;
		return(SPARK_STEP_ACTION_REPLAY);
	}
	replay->attempts = 0u;
	replay->exhausted++;
	return(SPARK_STEP_ACTION_EXHAUSTED);
}
