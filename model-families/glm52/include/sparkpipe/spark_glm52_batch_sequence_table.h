#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

// Batch-plane sequence table: admission and exchange lifecycle for longmem
// style workloads, sequences of B8 shared-prefix exchanges. This is the batch
// plane's own population structure, separate from the 1024-slot frame API:
// the expert-queue plane schedules per-sequence waves, so admission scales to
// sixteen thousand sequences without inflating frame dispatch shapes. The
// table is the authority for the active count that drives the expert queue
// firing threshold (active x lanes x topk / experts), and pausing a sequence
// for a tool call is the signal that its KV fragments become demotable in the
// JIT pool.

#define SPARK_GLM52_BATCH_SEQUENCE_ABI_VERSION 2u
#define SPARK_GLM52_BATCH_SEQUENCE_MAX_SEQUENCES 16384u
// Handles returned by Admit tag the slot index with a generation that
// increments when the slot is recycled, so a stale handle held past Complete is
// refused instead of silently acting on the slot's next occupant.
#define SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_BITS 14u
#define SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK ((1u << SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_BITS) - 1u)

#define SPARK_GLM52_BATCH_SEQUENCE_STATE_FREE 0u
#define SPARK_GLM52_BATCH_SEQUENCE_STATE_ACTIVE 1u
#define SPARK_GLM52_BATCH_SEQUENCE_STATE_AWAITING_TOOL 2u
#define SPARK_GLM52_BATCH_SEQUENCE_STATE_COMPLETE 3u

typedef struct SparkGlm52BatchSequence
{
	uint64_t sequence_id;
	uint32_t state;
	uint32_t lane_count;
	uint32_t exchange_number;
	uint32_t context_tokens;
	uint32_t fragment_base;
	uint32_t fragment_count;
	uint32_t free_next;
	uint32_t generation;
} SparkGlm52BatchSequence;

typedef struct SparkGlm52BatchSequenceTableConfiguration
{
	uint32_t abi_version;
	uint32_t sequence_capacity;
	uint32_t lane_count;
} SparkGlm52BatchSequenceTableConfiguration;

typedef struct SparkGlm52BatchSequenceTable
{
	uint32_t abi_version;
	uint32_t sequence_capacity;
	uint32_t lane_count;
	uint32_t active_count;
	uint32_t awaiting_tool_count;
	uint32_t complete_count;
	uint32_t free_head;
	uint32_t free_high_water;
	uint64_t exchange_count;
	SparkGlm52BatchSequence sequences[SPARK_GLM52_BATCH_SEQUENCE_MAX_SEQUENCES];
} SparkGlm52BatchSequenceTable;

SparkStatus SparkGlm52BatchSequenceTableInitialize(SparkGlm52BatchSequenceTable *table,const SparkGlm52BatchSequenceTableConfiguration *configuration);
SparkStatus SparkGlm52BatchSequenceTableAdmit(SparkGlm52BatchSequenceTable *table,uint64_t sequence_id,uint32_t context_tokens,uint32_t fragment_base,uint32_t fragment_count,uint32_t *sequence_handle_out);
SparkStatus SparkGlm52BatchSequenceTableBeginExchange(SparkGlm52BatchSequenceTable *table,uint32_t sequence_handle,uint32_t appended_context_tokens);
SparkStatus SparkGlm52BatchSequenceTablePauseForTool(SparkGlm52BatchSequenceTable *table,uint32_t sequence_handle);
SparkStatus SparkGlm52BatchSequenceTableComplete(SparkGlm52BatchSequenceTable *table,uint32_t sequence_handle);
uint32_t SparkGlm52BatchSequenceTableFiringThreshold(const SparkGlm52BatchSequenceTable *table,uint32_t topk,uint32_t expert_count,uint32_t threshold_cap);
