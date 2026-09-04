#ifndef SPARKPIPE_SPARK_STATE_POOL_H
#define SPARKPIPE_SPARK_STATE_POOL_H

#include <stdint.h>


#define SPARK_STATE_POOL_NO_SLOT 0xffffffffu
#define SPARK_STATE_POOL_IN_USE 0xfffffffeu

typedef struct SparkStatePool
{
	uint8_t *base;
	uint32_t *next_free;
	uint64_t slot_bytes;
	uint32_t slot_count;
	uint32_t free_head;
	uint32_t free_count;
	uint32_t reserved0;
} SparkStatePool;

static inline int32_t SparkStatePoolInitialize(SparkStatePool *pool, uint8_t *base, uint32_t *next_free, uint32_t slot_count, uint64_t slot_bytes)
{
	uint32_t slot;
	if (pool == 0 || base == 0 || next_free == 0 || slot_count == 0u ||
		slot_bytes == 0u)
		return(-30801);
	pool->base = base;
	pool->next_free = next_free;
	pool->slot_bytes = slot_bytes;
	pool->slot_count = slot_count;
	pool->free_head = 0u;
	pool->free_count = slot_count;
	for (slot = 0u; slot < slot_count; ++slot)
		next_free[slot] = slot + 1u;
	next_free[slot_count - 1u] = SPARK_STATE_POOL_NO_SLOT;
	return(0);
}

static inline uint32_t SparkStatePoolAcquire(SparkStatePool *pool)
{
	uint32_t slot;
	if (pool == 0 || pool->free_count == 0u)
		return(SPARK_STATE_POOL_NO_SLOT);
	slot = pool->free_head;
	pool->free_head = pool->next_free[slot];
	pool->next_free[slot] = SPARK_STATE_POOL_IN_USE;
	pool->free_count -= 1u;
	return(slot);
}

static inline int32_t SparkStatePoolRelease(SparkStatePool *pool, uint32_t slot)
{
	if (pool == 0 || slot >= pool->slot_count ||
		pool->next_free[slot] != SPARK_STATE_POOL_IN_USE)
		return(-30802);
	pool->next_free[slot] = pool->free_head;
	pool->free_head = slot;
	pool->free_count += 1u;
	return(0);
}

static inline uint8_t *SparkStatePoolSlot(const SparkStatePool *pool, uint32_t slot)
{
	if (pool == 0 || slot >= pool->slot_count)
		return(0);
	return(pool->base + ((uint64_t)slot * pool->slot_bytes));
}

#endif
