#pragma once


#include "runtime/tensor_map.h"
#include <mutex>
#include <stdint.h>
#include <string.h>

#define LM_GEMM_DESCRIPTOR_CACHE_SETS 1024u
#define LM_GEMM_DESCRIPTOR_CACHE_WAYS 4u

typedef struct LmGemmDescriptorSlot
{
	uint64_t last_used;
	uint32_t valid;
	LmTensorMapRequest key;
	alignas(64) CUtensorMap map;
}
LmGemmDescriptorSlot;

typedef struct LmGemmDescriptorCache
{
	std::mutex mutex;
	LmGemmDescriptorSlot
	    slots[LM_GEMM_DESCRIPTOR_CACHE_SETS][LM_GEMM_DESCRIPTOR_CACHE_WAYS];
	uint64_t tick;
}
LmGemmDescriptorCache;

typedef int32_t (*LmGemmDescriptorEncodeFn)(
	CUtensorMap *map,
	const LmTensorMapRequest *request);

static LmTensorMapRequest LmGemmDescriptorKey(const LmTensorMapRequest *request)
{
	LmTensorMapRequest key;

	memset(&key,0,sizeof(key));
	key.global_address = request->global_address;
	key.rows = request->rows;
	key.columns = request->columns;
	key.groups = request->groups;
	key.box_rows = request->box_rows;
	key.box_columns = request->box_columns;
	key.element_bits = request->element_bits;
	return(key);
}

static uint64_t LmGemmDescriptorKeyMix(uint64_t value, uint64_t hash)
{
	value += UINT64_C(0x9e3779b97f4a7c15);
	value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
	value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
	value = value ^ (value >> 31u);
	return(hash ^ value);
}

static uint64_t LmGemmDescriptorKeyHash(const LmTensorMapRequest *key)
{
	uint64_t hash = 0u;

	hash = LmGemmDescriptorKeyMix((uint64_t)(uintptr_t)key->global_address,hash);
	hash = LmGemmDescriptorKeyMix(key->rows,hash);
	hash = LmGemmDescriptorKeyMix(key->columns,hash);
	hash = LmGemmDescriptorKeyMix(key->groups,hash);
	hash = LmGemmDescriptorKeyMix(
		((uint64_t)key->box_rows << 32u) | key->box_columns,hash);
	return(LmGemmDescriptorKeyMix(key->element_bits,hash));
}

static int32_t LmGemmDescriptorCacheFetch(
	LmGemmDescriptorCache *cache,
	const LmTensorMapRequest *request,
	LmGemmDescriptorEncodeFn encode,
	CUtensorMap *out)
{
	LmTensorMapRequest key;
	LmGemmDescriptorSlot *set;
	LmGemmDescriptorSlot *victim;
	alignas(64) CUtensorMap encoded;
	uint32_t way;
	int32_t status;

	if ( cache == 0 || request == 0 || encode == 0 || out == 0 )
		return(LM_TM_ENCODE_ERR_NULL);
	key = LmGemmDescriptorKey(request);
	set = cache->slots[
		LmGemmDescriptorKeyHash(&key) % LM_GEMM_DESCRIPTOR_CACHE_SETS];
	{
		std::lock_guard<std::mutex> lock(cache->mutex);
		for ( way = 0u; way < LM_GEMM_DESCRIPTOR_CACHE_WAYS; way++ )
		{
			if ( set[way].valid != 0u
				&& memcmp(&set[way].key,&key,sizeof(key)) == 0 )
			{
				memcpy(out,&set[way].map,sizeof(*out));
				set[way].last_used = ++cache->tick;
				return(LM_TM_ENCODE_OK);
			}
		}
	}
	status = encode(&encoded,&key);
	if ( status != LM_TM_ENCODE_OK )
		return(status);
	{
		std::lock_guard<std::mutex> lock(cache->mutex);
		victim = &set[0];
		for ( way = 0u; way < LM_GEMM_DESCRIPTOR_CACHE_WAYS; way++ )
		{
			if ( set[way].valid == 0u )
			{
				victim = &set[way];
				break;
			}
			if ( set[way].last_used < victim->last_used )
				victim = &set[way];
		}
		victim->key = key;
		memcpy(&victim->map,&encoded,sizeof(encoded));
		victim->valid = 1u;
		victim->last_used = ++cache->tick;
	}
	memcpy(out,&encoded,sizeof(*out));
	return(LM_TM_ENCODE_OK);
}

static int32_t LmGemmTensorMapCached(
	CUtensorMap *map,
	const LmTensorMapRequest *request)
{
	static LmGemmDescriptorCache cache;

	return(LmGemmDescriptorCacheFetch(
		&cache,request,LmTensorMapPrepare,map));
}
