#pragma once


#include <cuda_runtime.h>
#include <stdint.h>

#define LM_GRAPH_OK 0
#define LM_GRAPH_ERR_CAPTURE (-81)
#define LM_GRAPH_ERR_FULL (-82)
#define LM_GRAPH_ERR_SHAPE (-83)

typedef struct LmGraphKey
{
	uint32_t rows;
	uint32_t layer_kind;
	uint32_t format;
	uint32_t sparse;
	uint32_t context_bucket;
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

static uint32_t LmGraphContextBucket(uint32_t context, uint32_t selection_budget)
{
	return(context <= selection_budget ? 0u : 1u);
}
