#include "sparkpipe/spark_expert_working_set.h"
#include "sparkpipe/spark_step_verdict.h"
#include "sparkpipe/spark_error_site.h"
#include <stdlib.h>
#include <string.h>

SparkStatus SparkExpertWorkingSetCreate(SparkExpertWorkingSet *set,uint32_t layers,uint32_t experts,uint32_t pack_stride,uint32_t cap_keys,SparkExpertWorkingSetAcquire acquire,void *acquire_context)
{
	uint64_t words;
	if ( set == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(set,0,sizeof(*set));
	if ( layers == 0u || experts == 0u || pack_stride <= experts || acquire == 0 || cap_keys == 0u || (uint64_t)layers * pack_stride > UINT32_MAX || cap_keys > layers * experts )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	set->stride = (experts + 31u) / 32u;
	words = (uint64_t)layers * set->stride;
	set->cover = (uint32_t *)calloc((size_t)words,sizeof(uint32_t));
	if ( set->cover == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	set->layers = layers;
	set->experts = experts;
	set->pack_stride = pack_stride;
	set->cap_keys = cap_keys;
	set->acquire = acquire;
	set->acquire_context = acquire_context;
	return(SPARK_STATUS_OK);
}

void SparkExpertWorkingSetDestroy(SparkExpertWorkingSet *set)
{
	if ( set == 0 )
		return;
	free(set->cover);
	memset(set,0,sizeof(*set));
}

uint32_t SparkExpertWorkingSetCovered(const SparkExpertWorkingSet *set,uint32_t layer,uint32_t expert)
{
	if ( set == 0 || set->cover == 0 || layer >= set->layers || expert >= set->experts )
		return(0u);
	return((set->cover[(uint64_t)layer * set->stride + (expert >> 5u)] >> (expert & 31u)) & 1u);
}

uint64_t SparkExpertWorkingSetCoverBytes(const SparkExpertWorkingSet *set)
{
	return(set != 0 ? (uint64_t)set->layers * set->stride * sizeof(uint32_t) : 0u);
}

static int SparkExpertWorkingSetCompareKey(const void *left,const void *right)
{
	uint32_t a = *(const uint32_t *)left,b = *(const uint32_t *)right;
	return(a < b ? -1 : (a > b ? 1 : 0));
}

static uint32_t SparkExpertWorkingSetUnique(uint32_t *keys,uint32_t count)
{
	uint32_t index,unique = 0u;
	qsort(keys,count,sizeof(*keys),SparkExpertWorkingSetCompareKey);
	for (index=0u; index<count; index++)
		if ( unique == 0u || keys[unique - 1u] != keys[index] )
			keys[unique++] = keys[index];
	return(unique);
}

static SparkStatus SparkExpertWorkingSetFresh(const SparkExpertWorkingSet *set,const uint32_t *keys,uint32_t count,uint32_t *fresh,uint32_t *fresh_count)
{
	uint32_t index,layer,expert,unique = 0u;
	for (index=0u; index<count; index++)
	{
		layer = keys[index] / set->pack_stride;
		expert = keys[index] % set->pack_stride;
		if ( layer >= set->layers || expert >= set->experts )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( SparkExpertWorkingSetCovered(set,layer,expert) == 0u )
			fresh[unique++] = keys[index];
	}
	*fresh_count = SparkExpertWorkingSetUnique(fresh,unique);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkExpertWorkingSetAdd(SparkExpertWorkingSet *set,const uint32_t *keys,uint32_t count)
{
	uint32_t *fresh,fresh_count = 0u,index,layer,expert;
	SparkStatus status;
	if ( set == 0 || set->cover == 0 || (count != 0u && keys == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( count == 0u )
		return(SPARK_STATUS_OK);
	fresh = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
	if ( fresh == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkExpertWorkingSetFresh(set,keys,count,fresh,&fresh_count);
	if ( status == SPARK_STATUS_OK && fresh_count > set->cap_keys - set->key_count )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK && fresh_count != 0u )
		status = set->acquire(set->acquire_context,fresh,fresh_count);
	if ( status != SPARK_STATUS_OK )
	{
		set->grow_denied += status != SPARK_STATUS_INVALID_ARGUMENT ? 1u : 0u;
		free(fresh);
		SPARK_FAIL(status);
	}
	for (index=0u; index<fresh_count; index++)
	{
		layer = fresh[index] / set->pack_stride;
		expert = fresh[index] % set->pack_stride;
		set->cover[(uint64_t)layer * set->stride + (expert >> 5u)] |= UINT32_C(1) << (expert & 31u);
	}
	set->key_count += fresh_count;
	set->grown += fresh_count;
	set->generation += fresh_count != 0u ? 1u : 0u;
	free(fresh);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkExpertWorkingSetCheckAnchors(const SparkExpertWorkingSet *set,uint32_t first_layer,uint32_t layer_count,uint32_t *missing_layer)
{
	uint32_t layer,word,held;
	if ( set == 0 || set->cover == 0 || missing_layer == 0 || first_layer > set->layers || layer_count > set->layers - first_layer )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*missing_layer = UINT32_MAX;
	for (layer=first_layer; layer<first_layer + layer_count; layer++)
	{
		held = 0u;
		for (word=0u; word<set->stride && held == 0u; word++)
			held = set->cover[(uint64_t)layer * set->stride + word] != 0u ? 1u : 0u;
		if ( held == 0u )
		{
			*missing_layer = layer;
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkExpertWorkingSetFirstLayer(const SparkExpertWorkingSet *set,const volatile uint32_t *miss,uint32_t entries,uint32_t *first_layer,uint32_t *last_layer)
{
	uint32_t index,entry,layer,expert;
	*first_layer = UINT32_MAX;
	*last_layer = 0u;
	for (index=0u; index<entries; index++)
	{
		entry = miss[SPARK_STEP_MISS_ENTRIES + index];
		layer = entry / set->pack_stride;
		expert = entry % set->pack_stride;
		if ( layer >= set->layers || expert > set->experts )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		*first_layer = layer < *first_layer ? layer : *first_layer;
		*last_layer = layer > *last_layer ? layer : *last_layer;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkExpertWorkingSetHarvest(SparkExpertWorkingSet *set,const volatile uint32_t *miss,uint32_t ring_capacity,uint32_t *keys,uint32_t key_capacity,SparkExpertMissHarvest *harvest)
{
	uint32_t index,entry,count = 0u,first_layer,last_layer;
	SparkStatus status;
	if ( set == 0 || miss == 0 || keys == 0 || harvest == 0 || ring_capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(harvest,0,sizeof(*harvest));
	harvest->first_layer = UINT32_MAX;
	harvest->recorded = miss[SPARK_STEP_MISS_COUNT];
	harvest->entries = harvest->recorded < ring_capacity ? harvest->recorded : ring_capacity;
	set->harvests++;
	if ( (miss[SPARK_STEP_MISS_FLAG] != 0u) != (harvest->recorded != 0u) )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( harvest->recorded == 0u )
		return(SPARK_STATUS_OK);
	status = SparkExpertWorkingSetFirstLayer(set,miss,harvest->entries,&first_layer,&last_layer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	harvest->first_layer = first_layer;
	if ( harvest->recorded > ring_capacity )
		set->harvest_overflow++;
	if ( harvest->recorded > ring_capacity && last_layer == first_layer )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<harvest->entries; index++)
	{
		entry = miss[SPARK_STEP_MISS_ENTRIES + index];
		if ( entry / set->pack_stride != first_layer )
			continue;
		if ( entry % set->pack_stride >= set->experts )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		if ( count >= key_capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		keys[count++] = entry;
	}
	harvest->key_count = SparkExpertWorkingSetUnique(keys,count);
	return(SPARK_STATUS_OK);
}
