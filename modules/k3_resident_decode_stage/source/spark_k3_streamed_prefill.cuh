#pragma once

#include <cstdint>
#include <cstring>

#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_pack_window.h"
#include "sparkpipe/spark_stage_runner_model.h"

#define K3_STREAMED_RANGE_GAP 4096u

typedef struct K3StreamedRoundContext
{
	const SparkStageRunnerServices *services;
	SparkStatus status;
} K3StreamedRoundContext;

static void K3StreamedRound(void *context, void *stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *send, void *receive)
{
	K3StreamedRoundContext *round = (K3StreamedRoundContext *)context;
	SparkStatus status;
	if ( round == 0 || round->services == 0 || round->services->round == 0 )
		return;
	status = round->services->round(round->services->context, stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL, rows, row_elements, send, receive);
	if ( status != SPARK_STATUS_OK && round->status == SPARK_STATUS_OK )
		round->status = status;
}

static int32_t K3StreamedLayerRanges(uint64_t payload_base, const SparkK3BoundLayer *bound, SparkPackWindowLayer *ranges)
{
	uint64_t start[SPARK_K3_BIND_MAX_NAMES], end[SPARK_K3_BIND_MAX_NAMES], expert_start = UINT64_MAX, expert_end = 0u;
	uint32_t count = 0u, index, other, dense;
	if ( bound == 0 || ranges == 0 || bound->tensor_count > SPARK_K3_BIND_MAX_NAMES )
		return -1;
	memset(ranges, 0, sizeof(*ranges));
	for ( index = 0u; index < bound->tensor_count; index++ )
	{
		const SparkK3PackEntry *entry = &bound->tensors[index].entry;
		const uint64_t first = payload_base + entry->payload_offset, last = first + entry->bytes;
		const size_t length = strlen(bound->tensors[index].name);
		if ( length >= 16u && (strcmp(bound->tensors[index].name + length - 16u, "expert_w1_weight") == 0 ||
			strcmp(bound->tensors[index].name + length - 16u, "expert_w2_weight") == 0) )
		{
			expert_start = first < expert_start ? first : expert_start;
			expert_end = last > expert_end ? last : expert_end;
			continue;
		}
		start[count] = first;
		end[count] = last;
		count++;
	}
	for ( index = 1u; index < count; index++ )
		for ( other = index; other > 0u && start[other - 1u] > start[other]; other-- )
		{
			const uint64_t swap_start = start[other], swap_end = end[other];
			start[other] = start[other - 1u];
			end[other] = end[other - 1u];
			start[other - 1u] = swap_start;
			end[other - 1u] = swap_end;
		}
	if ( expert_end != 0u )
	{
		ranges->experts.offset = expert_start;
		ranges->experts.bytes = expert_end - expert_start;
	}
	for ( index = 0u, dense = 0u; index < count; index++ )
	{
		const uint32_t crosses = expert_end != 0u && dense != 0u &&
			ranges->dense[dense - 1u].offset < expert_start && start[index] >= expert_end ? 1u : 0u;
		if ( expert_end != 0u && start[index] < expert_end && end[index] > expert_start )
			return -1;
		if ( dense != 0u && crosses == 0u && start[index] <= ranges->dense[dense - 1u].offset + ranges->dense[dense - 1u].bytes + K3_STREAMED_RANGE_GAP )
		{
			const uint64_t range_end = ranges->dense[dense - 1u].offset + ranges->dense[dense - 1u].bytes;
			if ( end[index] > range_end )
				ranges->dense[dense - 1u].bytes = end[index] - ranges->dense[dense - 1u].offset;
			continue;
		}
		if ( dense == SPARK_PACK_WINDOW_MAX_DENSE_RANGES )
			return -1;
		ranges->dense[dense].offset = start[index];
		ranges->dense[dense].bytes = end[index] - start[index];
		dense++;
	}
	ranges->dense_count = dense;
	return dense != 0u ? 0 : -1;
}
