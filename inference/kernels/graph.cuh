#pragma once

// CUDA graph capture. Record a step's launches once, replay them thereafter.
//
// A decode layer issues about twenty launches and a model has seventy-eight
// layers, so a token costs roughly 1,560 launches. Each is a few microseconds of
// driver work on the host, and at four tokens per second that host work is not
// hidden by anything - the GPU finishes a 30-microsecond kernel and waits for the
// CPU to ask for the next one.
//
// A captured graph replaces all of it with one submission. The driver walks a
// recorded dependency tree it already validated, and the host cost of a step
// becomes one call instead of 1,560.
//
// WHY THIS IS SMALL WHERE THE OLD ONE WAS 1,501 LINES. The old capture had to
// enumerate every launch it was recording, because the launches were spread
// across a 27,000-line file with no single function that issued them in order.
// Here the sequence IS a function - GlmLayerAttention followed by
// GlmLayerMoe - so capturing it is capturing one call, and the graph knows
// nothing about what is inside.
//
// THE CONSTRAINT THAT MAKES IT WORK. A graph records pointers, not values, so
// every buffer a captured step touches must live at the same address on replay.
// That is why the workspace is allocated once per bucket and never reallocated -
// and it is also why a graph is keyed by bucket: a different row count means a
// different grid, which is baked into the recording.
//
// WHAT MUST NOT BE CAPTURED. Anything whose control flow depends on device data.
// The sparse path branches on whether the context exceeds the selection budget,
// and that is a host-side decision made before capture; capturing one branch and
// replaying it for the other silently attends to the wrong positions. So the key
// includes the branch, and a step whose branch differs from its key rebuilds.

#include <cuda_runtime.h>
#include <stdint.h>

#define LM_GRAPH_OK 0
#define LM_GRAPH_ERR_CAPTURE (-81)
#define LM_GRAPH_ERR_FULL (-82)
#define LM_GRAPH_ERR_SHAPE (-83)

// What makes two steps interchangeable. Two steps with the same key issue the
// same launches against the same addresses, so one recording serves both.
//
// It is a struct rather than a hash because a hash collision here does not
// produce a wrong answer, it produces a wrong SEQUENCE - replaying a graph
// recorded for a different shape, which reads and writes whatever those pointers
// meant last time.
typedef struct LmGraphKey
{
	uint32_t rows;
	uint32_t layer_kind;
	uint32_t format;
	uint32_t sparse;               /* the branch, not a hint */
	uint32_t context_bucket;       /* selection budget class, not the length */
}
LmGraphKey;

typedef struct LmGraphEntry
{
	LmGraphKey key;
	cudaGraphExec_t executable;
	uint64_t replays;
	int32_t live;
}
LmGraphEntry;

typedef struct LmGraphCache
{
	LmGraphEntry *entries;
	uint32_t capacity;
	uint32_t count;
	uint64_t captures;
	uint64_t replays;
	uint64_t misses;
}
LmGraphCache;

static int32_t LmGraphKeyEqual(const LmGraphKey *a, const LmGraphKey *b)
{
	return(a->rows == b->rows && a->layer_kind == b->layer_kind
		&& a->format == b->format && a->sparse == b->sparse
		&& a->context_bucket == b->context_bucket);
}

static void LmGraphCacheInitialise(LmGraphCache *cache, LmGraphEntry *storage, uint32_t capacity)
{
	uint32_t index;
	cache->entries = storage;
	cache->capacity = capacity;
	cache->count = 0u;
	cache->captures = 0u;
	cache->replays = 0u;
	cache->misses = 0u;
	for (index = 0u; index < capacity; ++index)
		cache->entries[index].live = 0;
}

static LmGraphEntry *LmGraphFind(LmGraphCache *cache, const LmGraphKey *key)
{
	uint32_t index;
	for (index = 0u; index < cache->capacity; ++index)
		if ( cache->entries[index].live && LmGraphKeyEqual(&cache->entries[index].key,key) )
			return(&cache->entries[index]);
	return(0);
}

// Replay if this shape has been seen, otherwise report a miss so the caller
// captures. Split rather than combined because capture needs the caller's launch
// sequence and this file must not know what that is - a graph module that knows
// which kernels it records is a graph module that has to change when they do.
static int32_t LmGraphReplay(LmGraphCache *cache, const LmGraphKey *key, cudaStream_t stream)
{
	LmGraphEntry *entry = LmGraphFind(cache,key);
	if ( entry == 0 )
	{
		cache->misses++;
		return(LM_GRAPH_ERR_SHAPE);
	}
	if ( cudaGraphLaunch(entry->executable,stream) != cudaSuccess )
		return(LM_GRAPH_ERR_CAPTURE);
	entry->replays++;
	cache->replays++;
	return(LM_GRAPH_OK);
}

static int32_t LmGraphBeginCapture(cudaStream_t stream)
{
	// Relaxed rather than global: a global capture forbids any other stream from
	// launching for the duration, which stalls the transport posting the previous
	// layer's hidden state. Relaxed captures this stream's work and leaves the
	// ring alone.
	return(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed) == cudaSuccess
		? LM_GRAPH_OK : LM_GRAPH_ERR_CAPTURE);
}

static int32_t LmGraphEndCapture(LmGraphCache *cache, const LmGraphKey *key, cudaStream_t stream)
{
	cudaGraph_t graph;
	LmGraphEntry *entry;
	uint32_t index;
	if ( cudaStreamEndCapture(stream,&graph) != cudaSuccess )
		return(LM_GRAPH_ERR_CAPTURE);
	entry = 0;
	for (index = 0u; index < cache->capacity; ++index)
		if ( cache->entries[index].live == 0 )
		{
			entry = &cache->entries[index];
			break;
		}
	if ( entry == 0 )
	{
		// No slot. Destroying a live graph mid-serving to make room would stall
		// whichever request is replaying it, so the miss is reported and the
		// caller runs un-captured. Slow is better than stalled.
		cudaGraphDestroy(graph);
		return(LM_GRAPH_ERR_FULL);
	}
	if ( cudaGraphInstantiate(&entry->executable,graph,0ull) != cudaSuccess )
	{
		cudaGraphDestroy(graph);
		return(LM_GRAPH_ERR_CAPTURE);
	}
	cudaGraphDestroy(graph);
	entry->key = *key;
	entry->replays = 0u;
	entry->live = 1;
	cache->count++;
	cache->captures++;
	return(LM_GRAPH_OK);
}

// Context is bucketed rather than exact, because a graph recorded at 8,192
// tokens is valid at 8,193: the grid depends on the SELECTION BUDGET, which is
// fixed, not on the context length. Only the sparse-versus-dense branch changes
// with context, and that is its own key field.
//
// Without bucketing every token would be a new key and the cache would capture
// once per step, which costs more than it saves - the failure mode the old
// implementation's comments warn about.
static uint32_t LmGraphContextBucket(uint32_t context, uint32_t selection_budget)
{
	return(context <= selection_budget ? 0u : 1u);
}
