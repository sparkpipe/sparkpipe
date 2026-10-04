#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_WRITE_BUDGET_WINDOW_NS (86400ull * 1000000000ull)

typedef struct SparkKvWriteBudget
{
	uint64_t bytes_per_day;
	uint64_t available_bytes;
	uint64_t refilled_ns;
	uint64_t charged_bytes;
	uint64_t overrun_bytes;
	uint64_t refused_saves;
	uint64_t discarded_pages;
} SparkKvWriteBudget;

static inline void SparkKvWriteBudgetRefill(SparkKvWriteBudget *budget,uint64_t now_ns)
{
	unsigned __int128 earned;
	if ( now_ns <= budget->refilled_ns )
		return;
	earned = (unsigned __int128)(now_ns - budget->refilled_ns) * budget->bytes_per_day / SPARK_KV_WRITE_BUDGET_WINDOW_NS;
	if ( earned == 0u )
		return;
	budget->refilled_ns = now_ns;
	budget->available_bytes = earned >= (unsigned __int128)(budget->bytes_per_day - budget->available_bytes) ?
		budget->bytes_per_day : budget->available_bytes + (uint64_t)earned;
}

static inline SparkStatus SparkKvWriteBudgetInitialize(SparkKvWriteBudget *budget,uint64_t bytes_per_day,uint64_t now_ns)
{
	if ( budget == 0 || bytes_per_day == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(budget,0,sizeof(*budget));
	budget->bytes_per_day = bytes_per_day;
	budget->available_bytes = bytes_per_day;
	budget->refilled_ns = now_ns;
	return(SPARK_STATUS_OK);
}

static inline uint32_t SparkKvWriteBudgetAllows(SparkKvWriteBudget *budget,uint64_t bytes,uint64_t now_ns)
{
	SparkKvWriteBudgetRefill(budget,now_ns);
	return(budget->available_bytes >= bytes ? 1u : 0u);
}

static inline void SparkKvWriteBudgetCharge(SparkKvWriteBudget *budget,uint64_t bytes,uint64_t now_ns)
{
	SparkKvWriteBudgetRefill(budget,now_ns);
	budget->charged_bytes += bytes;
	if ( budget->available_bytes >= bytes )
		budget->available_bytes -= bytes;
	else
	{
		budget->overrun_bytes += bytes - budget->available_bytes;
		budget->available_bytes = 0u;
	}
}

#ifdef __cplusplus
}
#endif
