#pragma once

// GEMM-007: the GEMM host path called cuTensorMapEncodeTiled twice on EVERY
// launch. The encode is a driver call in the microseconds, and on the decode
// hot path it runs once per operand per GEMM per layer per token - hundreds
// of driver round-trips of pure host latency spent re-deriving bytes that do
// not change.
//
// They do not change because the descriptor is a pure function of the encode
// REQUEST. LmTensorMapPlanBuild is arithmetic over (address, rows, columns,
// groups, box, element width) and cuTensorMapEncodeTiled encodes exactly that
// plan against exactly that address. Identical request therefore means
// byte-identical descriptor, which is the correctness rule of this cache: a
// key hit CANNOT return a stale map, and anything that is not a key hit
// re-encodes. Weight maps change only at layer bind; activation maps change
// per distinct (pointer, rows), which at decode is a handful. Steady-state
// decode performs zero encodes per token.
//
// The table is fixed storage - no allocation, ever. It is hashed on the full
// request rather than scanned, because the working set is NOT small across a
// whole model: one translation unit launches every layer's GEMMs, so the
// cache holds (layers x projections) weight keys plus the activation keys,
// several hundred entries whose access pattern is a strict cycle. A linear
// scan pays that cycle microseconds per token and a small LRU evicts the next
// layer's entry just before it is needed; sets of LM_GEMM_DESCRIPTOR_CACHE_WAYS
// with per-set LRU eviction degrade to one re-encode only when five LIVE keys
// collide on one set, which 1024 sets makes a sizing choice rather than a
// risk.

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

// Signature matches LmTensorMapPrepare, so production passes it directly and
// tests pass a counting mock.
typedef int32_t (*LmGemmDescriptorEncodeFn)(
	CUtensorMap *map,
	const LmTensorMapRequest *request);

// Canonicalise the key. The request is 44 bytes of fields in a 48-byte struct
// and no caller is required to zero the tail padding, so a bytewise compare
// of a raw request could miss a logically-identical key - only ever a
// redundant encode, never a wrong map, but a miss is the cost this cache
// exists to remove. Field-by-field into a zeroed struct makes memcmp sound.
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

// splitmix64 per field. Device pointers are at least 16-byte aligned, so the
// low bits of the address carry no entropy and an unmixed modulo would pile
// same-page tensors onto one set. Exported (header-static) so the host test
// can FIND colliding keys instead of hoping for them.
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
				// Byte-identical by construction: every input to the
				// encode is a key field, so this copy is the map a
				// re-encode would have produced.
				memcpy(out,&set[way].map,sizeof(*out));
				set[way].last_used = ++cache->tick;
				return(LM_TM_ENCODE_OK);
			}
		}
	}
	// Miss. Encode OUTSIDE the lock: this driver call is the slow path the
	// cache exists to make rare, and holding the mutex across it would park
	// every other thread's cache HIT behind one layer bind. Two threads
	// racing the same miss both encode and both store the same bytes - the
	// second store is redundant, never wrong.
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

// The cache the launcher uses. Function-local static: construction is
// thread-safe, there is no init-order dependence between translation units,
// and steady state never touches the heap.
static int32_t LmGemmTensorMapCached(
	CUtensorMap *map,
	const LmTensorMapRequest *request)
{
	static LmGemmDescriptorCache cache;

	return(LmGemmDescriptorCacheFetch(
		&cache,request,LmTensorMapPrepare,map));
}
