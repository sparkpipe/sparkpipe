#pragma once

#include <stdint.h>

#define SPARK_LATENCY_BUCKETS 24u

typedef struct SparkLatencyHistogram
{
	uint64_t count[SPARK_LATENCY_BUCKETS];
	uint64_t total_ns;
	uint64_t max_ns;
} SparkLatencyHistogram;

static inline uint32_t SparkLatencyBucket(uint64_t ns)
{
	uint64_t us = ns / 1000u;
	uint32_t bucket = us == 0u ? 0u : 64u - (uint32_t)__builtin_clzll(us);
	return(bucket < SPARK_LATENCY_BUCKETS ? bucket : SPARK_LATENCY_BUCKETS - 1u);
}

static inline void SparkLatencyAdd(SparkLatencyHistogram *histogram,uint64_t start_ns,uint64_t end_ns)
{
	uint64_t elapsed = end_ns - start_ns;
	if ( start_ns == 0u || end_ns < start_ns )
		return;
	histogram->count[SparkLatencyBucket(elapsed)]++;
	histogram->total_ns += elapsed;
	histogram->max_ns = elapsed > histogram->max_ns ? elapsed : histogram->max_ns;
}

static inline uint64_t SparkLatencyCount(const SparkLatencyHistogram *histogram)
{
	uint64_t total = 0u;
	uint32_t bucket;
	for (bucket=0u; bucket<SPARK_LATENCY_BUCKETS; bucket++)
		total += histogram->count[bucket];
	return(total);
}

static inline uint64_t SparkLatencyPercentileUs(const SparkLatencyHistogram *histogram,uint32_t percent)
{
	uint64_t total = SparkLatencyCount(histogram),seen = 0u;
	uint32_t bucket;
	for (bucket=0u; bucket<SPARK_LATENCY_BUCKETS && total != 0u; bucket++)
	{
		seen += histogram->count[bucket];
		if ( seen * 100u >= total * percent )
			return(UINT64_C(1) << bucket);
	}
	return(0u);
}
