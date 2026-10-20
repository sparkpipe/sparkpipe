#pragma once

#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define LM_WEIGHT_PREFETCH_SPANS 8u
#define LM_WEIGHT_PREFETCH_GROUPS 1024u
#define LM_WEIGHT_PREFETCH_PIECE_BYTES 16384u
#define LM_WEIGHT_PREFETCH_ISSUERS 8u
#define LM_WEIGHT_PREFETCH_BUDGET_BYTES (12u << 20)

typedef struct LmWeightPrefetchGroup
{
	uint64_t address[LM_WEIGHT_PREFETCH_SPANS];
	uint32_t bytes[LM_WEIGHT_PREFETCH_SPANS];
	uint32_t count;
	uint32_t total;
	uint64_t progress;
}
LmWeightPrefetchGroup;

typedef struct LmWeightPrefetchPlan
{
	cudaStream_t stream;
	LmWeightPrefetchGroup *host;
	LmWeightPrefetchGroup *device;
	uint32_t capacity;
	uint32_t groups;
	uint32_t dropped;
	uint32_t budget;
	uint32_t current_dropped;
}
LmWeightPrefetchPlan;

inline LmWeightPrefetchPlan *&LmWeightPrefetchActive(void)
{
	static thread_local LmWeightPrefetchPlan *active = 0;
	return(active);
}

static inline void LmWeightPrefetchArm(LmWeightPrefetchPlan *plan, cudaStream_t stream)
{
	if ( plan == 0 || plan->host == 0 || plan->device == 0 || plan->capacity == 0u )
		return;
	plan->stream = stream;
	plan->groups = 0u;
	plan->dropped = 0u;
	plan->current_dropped = 0u;
	plan->budget = LM_WEIGHT_PREFETCH_BUDGET_BYTES;
	memset(plan->host,0,(size_t)plan->capacity * sizeof(*plan->host));
	LmWeightPrefetchActive() = plan;
}

static inline cudaError_t LmWeightPrefetchDisarm(LmWeightPrefetchPlan *plan, uint32_t publish)
{
	if ( plan == 0 || LmWeightPrefetchActive() != plan )
		return(cudaSuccess);
	LmWeightPrefetchActive() = 0;
	if ( publish == 0u || plan->groups == 0u )
		return(cudaSuccess);
	return(cudaMemcpy(plan->device,plan->host,(size_t)plan->groups * sizeof(*plan->host),cudaMemcpyHostToDevice));
}

static inline LmWeightPrefetchGroup *LmWeightPrefetchSlot(cudaStream_t stream, uint32_t *resume)
{
	LmWeightPrefetchPlan *plan = LmWeightPrefetchActive();
	*resume = 0u;
	if ( plan == 0 || plan->stream != stream )
		return(0);
	if ( plan->groups != 0u && plan->host[plan->groups - 1u].count == 0u )
	{
		*resume = 1u;
		plan->current_dropped = 0u;
		return(plan->device + plan->groups - 1u);
	}
	if ( plan->groups == plan->capacity )
	{
		plan->dropped++;
		plan->current_dropped = 1u;
		return(0);
	}
	plan->current_dropped = 0u;
	plan->groups++;
	return(plan->device + plan->groups - 1u);
}

static inline void LmWeightPrefetchNote(cudaStream_t stream, const void *address, uint64_t bytes)
{
	LmWeightPrefetchPlan *plan = LmWeightPrefetchActive();
	LmWeightPrefetchGroup *group;
	uint64_t start = (uint64_t)(uintptr_t)address,room;
	if ( plan == 0 || plan->stream != stream || plan->groups == 0u || plan->current_dropped != 0u || address == 0 || (start & 15u) != 0u )
		return;
	group = &plan->host[plan->groups - 1u];
	room = plan->budget > group->total ? plan->budget - group->total : 0u;
	bytes = (bytes < room ? bytes : room) & ~(uint64_t)15u;
	if ( bytes == 0u )
		return;
	if ( group->count != 0u && group->address[group->count - 1u] + group->bytes[group->count - 1u] == start )
		group->bytes[group->count - 1u] += (uint32_t)bytes;
	else if ( group->count == LM_WEIGHT_PREFETCH_SPANS )
		return;
	else
	{
		group->address[group->count] = start;
		group->bytes[group->count] = (uint32_t)bytes;
		group->count++;
	}
	group->total += (uint32_t)bytes;
}

#ifdef __CUDACC__
static __device__ __forceinline__ uint64_t LmWeightPrefetchExtent(const LmWeightPrefetchGroup *group, uint32_t count)
{
	uint64_t extent = 0u;
	for ( uint32_t span = 0u; span < count; span++ )
		extent += (group->bytes[span] + LM_WEIGHT_PREFETCH_PIECE_BYTES - 1u) & ~(LM_WEIGHT_PREFETCH_PIECE_BYTES - 1u);
	return(extent);
}

static __device__ __forceinline__ void LmWeightPrefetchPiece(const LmWeightPrefetchGroup *group, uint32_t count, uint64_t position)
{
	uint32_t span,length,padded;
	for ( span = 0u; span < count; span++ )
	{
		length = group->bytes[span];
		padded = (length + LM_WEIGHT_PREFETCH_PIECE_BYTES - 1u) & ~(LM_WEIGHT_PREFETCH_PIECE_BYTES - 1u);
		if ( position < padded )
			break;
		position -= padded;
	}
	if ( span == count || position >= length )
		return;
	length -= (uint32_t)position;
	length = length < LM_WEIGHT_PREFETCH_PIECE_BYTES ? length : LM_WEIGHT_PREFETCH_PIECE_BYTES;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(group->address[span] + position), "r"(length) : "memory");
#endif
}
#endif
