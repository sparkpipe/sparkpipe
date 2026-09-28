#pragma once

#include <stdint.h>
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef SparkStatus (*SparkExpertWorkingSetAcquire)(void *context,const uint32_t *keys,uint32_t count);

typedef struct SparkExpertWorkingSet
{
	uint32_t *cover;
	uint32_t layers;
	uint32_t experts;
	uint32_t stride;
	uint32_t pack_stride;
	uint32_t cap_keys;
	uint32_t key_count;
	uint64_t generation;
	uint64_t grown;
	uint64_t grow_denied;
	uint64_t harvests;
	uint64_t harvest_overflow;
	SparkExpertWorkingSetAcquire acquire;
	void *acquire_context;
} SparkExpertWorkingSet;

typedef struct SparkExpertMissHarvest
{
	uint32_t first_layer;
	uint32_t entries;
	uint32_t recorded;
	uint32_t key_count;
} SparkExpertMissHarvest;

SparkStatus SparkExpertWorkingSetCreate(SparkExpertWorkingSet *set,uint32_t layers,uint32_t experts,uint32_t pack_stride,uint32_t cap_keys,SparkExpertWorkingSetAcquire acquire,void *acquire_context);
void SparkExpertWorkingSetDestroy(SparkExpertWorkingSet *set);
uint32_t SparkExpertWorkingSetCovered(const SparkExpertWorkingSet *set,uint32_t layer,uint32_t expert);
SparkStatus SparkExpertWorkingSetAdd(SparkExpertWorkingSet *set,const uint32_t *keys,uint32_t count);
SparkStatus SparkExpertWorkingSetCheckAnchors(const SparkExpertWorkingSet *set,uint32_t first_layer,uint32_t layer_count,uint32_t *missing_layer);
SparkStatus SparkExpertWorkingSetHarvest(SparkExpertWorkingSet *set,const volatile uint32_t *miss,uint32_t ring_capacity,uint32_t *keys,uint32_t key_capacity,SparkExpertMissHarvest *harvest);
uint64_t SparkExpertWorkingSetCoverBytes(const SparkExpertWorkingSet *set);

#ifdef __cplusplus
}
#endif
