#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_weightd_lease_sets.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_error_site.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum
{
	LEASE_SET_EMPTY,
	LEASE_SET_ACQUIRED,
	LEASE_SET_BEGUN,
	LEASE_SET_RECORDED
};

struct SparkWeightdLeaseSets
{
	SparkWeightdLeaseRecord record;
	void *context;
	uint32_t set_count;
	uint32_t slots;
	uint64_t *identifier;
	uint32_t *phase;
};

static uint64_t lease_sets_now(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return(((uint64_t)now.tv_sec * UINT64_C(1000000000)) + (uint64_t)now.tv_nsec);
}

SparkStatus SparkWeightdLeaseSetsCreate(uint32_t set_count,uint32_t keys_max,SparkWeightdLeaseRecord record,void *context,SparkWeightdLeaseSets **out)
{
	SparkWeightdLeaseSets *sets;
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( record == 0 || set_count == 0u || keys_max == 0u || keys_max > (UINT32_MAX - SPARK_WEIGHTD_LEASE_GROUPS_MAX) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sets = calloc(1u,sizeof(*sets));
	if ( sets == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	sets->record = record;
	sets->context = context;
	sets->set_count = set_count;
	sets->slots = ((keys_max + SPARK_WEIGHTD_LEASE_GROUPS_MAX - 1u) / SPARK_WEIGHTD_LEASE_GROUPS_MAX);
	sets->identifier = calloc((size_t)set_count * sets->slots,sizeof(*sets->identifier));
	sets->phase = calloc((size_t)set_count * sets->slots,sizeof(*sets->phase));
	if ( sets->identifier == 0 || sets->phase == 0 )
	{
		free(sets->identifier);
		free(sets->phase);
		free(sets);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	*out = sets;
	return(SPARK_STATUS_OK);
}

uint32_t SparkWeightdLeaseSetsCount(const SparkWeightdLeaseSets *sets)
{
	return(sets != 0 ? sets->set_count : 0u);
}

uint32_t SparkWeightdLeaseSetsHeld(const SparkWeightdLeaseSets *sets,uint32_t set)
{
	uint32_t slot,held = 0u;
	if ( sets == 0 || set >= sets->set_count )
		return(0u);
	for (slot=0u; slot<sets->slots; slot++)
		if ( sets->identifier[(set * sets->slots) + slot] != 0u )
			held++;
	return(held);
}

SparkStatus SparkWeightdLeaseSetsDestroy(SparkWeightdLeaseSets *sets)
{
	uint32_t set;
	if ( sets == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (set=0u; set<sets->set_count; set++)
		if ( SparkWeightdLeaseSetsHeld(sets,set) != 0u )
			SPARK_FAIL(SPARK_STATUS_BUSY);
	free(sets->identifier);
	free(sets->phase);
	free(sets);
	return(SPARK_STATUS_OK);
}

static SparkStatus lease_sets_release_one(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint32_t index,uint64_t deadline)
{
	uint64_t now;
	SparkStatus status;
	if ( sets->identifier[index] == 0u )
		return(sets->phase[index] == LEASE_SET_EMPTY ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED);
	if ( sets->phase[index] == LEASE_SET_BEGUN )
	{
		status = sets->record(sets->context,map,sets->identifier[index]);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		sets->phase[index] = LEASE_SET_RECORDED;
	}
	now = lease_sets_now();
	if ( now == 0u || now >= deadline )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	status = sets->phase[index] == LEASE_SET_RECORDED
		? SparkWeightdMapReleaseCompleted(map,sets->identifier[index],deadline - now)
		: SparkWeightdMapRelease(map,sets->identifier[index],deadline - now);
	if ( status == SPARK_STATUS_OK )
	{
		sets->identifier[index] = 0u;
		sets->phase[index] = LEASE_SET_EMPTY;
	}
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdLeaseSetsRelease(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint32_t set,uint64_t timeout)
{
	SparkStatus status,result = SPARK_STATUS_OK;
	uint64_t now = lease_sets_now();
	uint32_t slot;
	if ( sets == 0 || set >= sets->set_count || timeout == 0u || now == 0u || timeout > (UINT64_MAX - now) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightdLeaseSetsHeld(sets,set) == 0u )
		return(SPARK_STATUS_OK);
	if ( map == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (slot=0u; slot<sets->slots; slot++)
	{
		status = lease_sets_release_one(sets,map,(set * sets->slots) + slot,now + timeout);
		if ( status != SPARK_STATUS_OK && result == SPARK_STATUS_OK )
			result = status;
	}
	SPARK_RETURN(result);
}

SparkStatus SparkWeightdLeaseSetsReleaseAll(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint64_t timeout)
{
	SparkStatus status,result = SPARK_STATUS_OK;
	uint32_t set;
	if ( sets == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (set=0u; set<sets->set_count; set++)
	{
		status = SparkWeightdLeaseSetsRelease(sets,map,set,timeout);
		if ( status != SPARK_STATUS_OK && result == SPARK_STATUS_OK )
			result = status;
	}
	SPARK_RETURN(result);
}

SparkStatus SparkWeightdLeaseSetsRecord(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint32_t set)
{
	SparkStatus status;
	uint32_t slot,index;
	if ( sets == 0 || set >= sets->set_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (slot=0u; slot<sets->slots; slot++)
	{
		index = (set * sets->slots) + slot;
		if ( sets->phase[index] != LEASE_SET_BEGUN )
			continue;
		if ( map == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		status = sets->record(sets->context,map,sets->identifier[index]);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		sets->phase[index] = LEASE_SET_RECORDED;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLeaseSetsAcquire(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint32_t set,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t timeout,void **base)
{
	uint32_t first,chunk,slot,index;
	SparkStatus status = SPARK_STATUS_OK;
	if ( base == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*base = 0;
	if ( sets == 0 || map == 0 || set >= sets->set_count || keys == 0 || count == 0u || timeout == 0u ||
		count > (sets->slots * SPARK_WEIGHTD_LEASE_GROUPS_MAX) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightdLeaseSetsHeld(sets,set) != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (first=0u,slot=0u; first<count && status == SPARK_STATUS_OK; first+=chunk,slot++)
	{
		chunk = (count - first) < SPARK_WEIGHTD_LEASE_GROUPS_MAX ? (count - first) : SPARK_WEIGHTD_LEASE_GROUPS_MAX;
		index = (set * sets->slots) + slot;
		status = SparkWeightdMapAcquire(map,keys + first,chunk,&sets->identifier[index],timeout);
		if ( sets->identifier[index] != 0u )
			sets->phase[index] = LEASE_SET_ACQUIRED;
		if ( status == SPARK_STATUS_OK && sets->identifier[index] == 0u )
			status = SPARK_STATUS_VALIDATION_FAILED;
		if ( status == SPARK_STATUS_OK )
			status = SparkWeightdMapBeginUse(map,sets->identifier[index],base);
		if ( status == SPARK_STATUS_OK )
			sets->phase[index] = LEASE_SET_BEGUN;
	}
	if ( status == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	*base = 0;
	(void)SparkWeightdLeaseSetsRelease(sets,map,set,timeout);
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdLeaseSetsFit(const SparkWeightdManifest *manifest,const SparkWeightdLazyAttachResult *attached,uint32_t layers,uint32_t set_count,uint64_t *widest,uint64_t *pool_chunks)
{
	uint64_t chunks;
	uint32_t layer;
	SparkStatus status;
	if ( widest == 0 || pool_chunks == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*widest = 0u;
	*pool_chunks = 0u;
	if ( manifest == 0 || attached == 0 || attached->chunk_bytes == 0u || layers == 0u || set_count == 0u || set_count > layers )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (layer=0u; layer + set_count <= layers; layer++)
	{
		status = SparkWeightdManifestLayerChunks(manifest,attached->chunk_bytes,layer,set_count,&chunks);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( chunks > *widest )
			*widest = chunks;
	}
	*pool_chunks = attached->expert_pool_bytes > attached->arena_bytes ? UINT64_MAX : (attached->expert_pool_bytes / attached->chunk_bytes);
	return(SPARK_STATUS_OK);
}
