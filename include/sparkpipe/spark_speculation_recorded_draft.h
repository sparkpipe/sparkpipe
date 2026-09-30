#pragma once

#include <stdint.h>

#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_RECORDED_MAGIC 0x44525053u
#define SPARK_SPECULATION_RECORDED_VERSION 1u
#define SPARK_SPECULATION_RECORDED_HEADER_BYTES 32u
#define SPARK_SPECULATION_RECORDED_ENTRY_FIXED_BYTES 24u

typedef struct SparkSpeculationRecordedDraft
{
	const uint8_t *entries;
	uint64_t entry_count;
	uint32_t depth;
	uint32_t entry_bytes;
	uint32_t vocab_size;
	uint32_t reserved;
	uint64_t hits;
	uint64_t misses;
} SparkSpeculationRecordedDraft;

SparkStatus SparkSpeculationRecordedDraftInitialize(SparkSpeculationRecordedDraft *draft,const uint8_t *bytes,uint64_t byte_count,uint32_t vocab_size);
SparkStatus SparkSpeculationRecordedDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result);

#ifdef __cplusplus
}
#endif
