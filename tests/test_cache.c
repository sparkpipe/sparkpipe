#include "cache/cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if (!condition)
		++failures;
}

static uint32_t g_unresolvable = 0xffffffffu;
static int32_t resolve_source(void *context, uint32_t block, uint32_t *source, uint64_t *offset)
{
	(void)context;
	if (block == g_unresolvable)
		return LM_CACHE_ERR_NOT_FOUND;
	*source = LM_CACHE_SOURCE_MEMORY;
	*offset = (uint64_t)block * 4096u;
	return LM_CACHE_OK;
}

static LmCache *make_cache(uint32_t blocks, uint32_t reserve)
{
	static LmCache cache;
	static LmCacheBlock block_table[4096];
	static LmCacheIndexEntry index_table[4096];
	static uint32_t buckets[4096];
	LmCachePartition partition;
	memset(&partition, 0, sizeof(partition));
	partition.block_tokens = 64u;
	partition.slot_bytes = 1152u;
	partition.index_entries = blocks;
	partition.jit_reserve_blocks = reserve;
	partition.resident_limit = blocks / 4u ? blocks / 4u : 1u;
	partition.total_bytes = ((uint64_t)blocks * (64u * 1152u + sizeof(LmCacheBlock)))
		+ ((uint64_t)blocks * (sizeof(LmCacheIndexEntry) + sizeof(uint32_t)));
	if (LmCacheInitialise(&cache, &partition, block_table, index_table, buckets) != LM_CACHE_OK)
		return NULL;
	return &cache;
}

int main(void)
{
	printf("kv cache\n\npartitioning\n");
	{
		LmCachePartition partition;
		memset(&partition, 0, sizeof(partition));
		partition.total_bytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
		partition.block_tokens = 64u;
		partition.slot_bytes = 1152u;
		partition.index_entries = 1u << 20;
		uint64_t blocks = LmCacheBlocksAvailable(&partition);
		printf("    1 TB, 64-token blocks, GLM 5.2 latent slots\n");
		printf("      %llu blocks = %llu tokens = %llu sequences at 8k context\n",
			(unsigned long long)blocks,
			(unsigned long long)(blocks * 64u),
			(unsigned long long)(blocks * 64u / 8192u));
		expect(blocks > 14000000ULL, "1 TB holds over 14 million blocks");
		partition.slot_bytes = 6144u;
		printf("      the same TB at MiMo 2.5's 6144-byte slot: %llu blocks\n",
			(unsigned long long)LmCacheBlocksAvailable(&partition));
	}

	printf("\nsharing is a reference count, not a copy\n");
	{
		LmCache *cache = make_cache(64u, 8u);
		uint32_t first, second;
		int32_t hit;
		uint64_t hash = LmCacheHashBlock(0u, (const uint32_t[]){ 1u, 2u, 3u }, 3u);
		LmCacheAcquire(cache, hash, 0u, &first, &hit);
		expect(hit == 0, "first acquire of a hash misses");
		LmCachePublish(cache, first, hash);
		LmCacheAcquire(cache, hash, 1u, &second, &hit);
		expect(hit == 1 && second == first, "second acquire hits the same block");
		expect(cache->blocks[first].reference_count == 2u, "two references, one block");
		expect(cache->live_blocks == 1u, "and one live block, not two");
	}

	printf("\nthe hash chains, so a block matches only when the whole prefix does\n");
	{
		uint32_t a[] = { 10u, 11u };
		uint32_t b[] = { 99u, 11u };
		uint64_t chain_a = LmCacheHashBlock(LmCacheHashBlock(0u, a, 2u), a, 2u);
		uint64_t chain_b = LmCacheHashBlock(LmCacheHashBlock(0u, b, 2u), a, 2u);
		expect(chain_a != chain_b,
			"same block, different preceding context, different hash");
	}

	printf("\neviction keeps what is reused\n");
	{
		LmCache *cache = make_cache(4u, 0u);
		uint32_t block[4], fresh;
		int32_t hit, index;
		for (index = 0; index < 4; ++index)
		{
			uint64_t hash = LmCacheHashBlock(0u, (const uint32_t *)&index, 1u);
			LmCacheAcquire(cache, hash, (uint32_t)index, &block[index], &hit);
			LmCachePublish(cache, block[index], hash);
			LmCacheRelease(cache, block[index]);
		}
		{
			uint64_t hash = cache->blocks[block[0]].content_hash;
			uint32_t got;
			for (index = 0; index < 20; ++index)
			{
				LmCacheAcquire(cache, hash, 0u, &got, &hit);
				LmCacheRelease(cache, got);
			}
		}
		LmCacheAcquire(cache, 0u, 9u, &fresh, &hit);
		expect(fresh != block[0], "a heavily reused block survives eviction despite age");
	}

	printf("\ncopy-on-write forks only the diverging block\n");
	{
		LmCache *cache = make_cache(64u, 0u);
		uint32_t shared, second, forked;
		int32_t hit;
		uint64_t hash = LmCacheHashBlock(0u, (const uint32_t[]){ 7u }, 1u);
		LmCacheAcquire(cache, hash, 0u, &shared, &hit);
		LmCachePublish(cache, shared, hash);
		LmCacheAcquire(cache, hash, 1u, &second, &hit);
		LmCacheFork(cache, second, 1u, &forked);
		expect(forked != shared, "the writer gets its own block");
		expect(cache->blocks[shared].reference_count == 1u, "the reader keeps the original");
	}

	printf("\nthe JIT reserve protects admitted requests\n");
	{
		LmCache *cache = make_cache(16u, 4u);
		uint32_t block;
		int32_t hit, index, status;
		expect(LmCacheReserve(cache, 4u) == LM_CACHE_OK, "reserve the whole allowance");
		expect(LmCacheReserve(cache, 1u) == LM_CACHE_ERR_BUSY, "and no more than that");
		for (index = 0; index < 12; ++index)
			LmCacheAcquire(cache, 0u, (uint32_t)index, &block, &hit);
		status = LmCacheAcquire(cache, 0u, 99u, &block, &hit);
		expect(status == LM_CACHE_ERR_FULL,
			"a fresh miss cannot consume what an admitted request reserved");
		LmCacheReleaseReservation(cache, 4u);
		expect(LmCacheAcquire(cache, 0u, 99u, &block, &hit) == LM_CACHE_OK,
			"and can once the reservation is released");
	}

	printf("\nresidency is a second question and the pool cannot answer it\n");
	{
		LmCache *cache = make_cache(64u, 0u);
		uint32_t block[20], index;
		int32_t hit, fetch;
		for (index = 0; index < 20u; ++index)
		{
			LmCacheAcquire(cache, 0u, index, &block[index], &hit);
			LmCacheMakeResident(cache, block[index], &fetch);
		}
		expect(cache->resident_blocks == 16u, "residency is bounded by the device, not the pool");
		expect(cache->live_blocks == 20u, "while the pool holds more than fits");
		expect(cache->resident_evictions == 4u, "and paged out the difference");
		expect(LmCacheBlocksAreResident(cache, block, 20u) == 0,
			"a referenced block can be non-resident, which admission must ask about");
	}

	printf("\nprotection survives residency pressure\n");
	{
		LmCache *cache = make_cache(64u, 0u);
		uint32_t keep, block, index;
		int32_t hit, fetch;
		LmCacheAcquire(cache, 0u, 0u, &keep, &hit);
		LmCacheMakeResident(cache, keep, &fetch);
		LmCacheProtect(cache, keep, 1);
		for (index = 0; index < 40u; ++index)
		{
			LmCacheAcquire(cache, 0u, index + 1u, &block, &hit);
			LmCacheMakeResident(cache, block, &fetch);
		}
		expect(cache->blocks[keep].resident == 1u,
			"a protected block is still readable after 40 competing fetches");
		LmCacheProtect(cache, keep, 0);
		expect(cache->blocks[keep].protected_from_eviction == 0u, "and unprotects");
	}

	printf("\n1 TB is mostly not resident, which is the point\n");
	{
		LmCachePartition partition;
		memset(&partition, 0, sizeof(partition));
		partition.total_bytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
		partition.block_tokens = 64u;
		partition.slot_bytes = 1152u;
		partition.index_entries = 1u << 20;
		uint64_t pool = LmCacheBlocksAvailable(&partition);
		uint64_t resident = (40ULL * 1024 * 1024 * 1024) / (64u * 1152u);
		printf("    pool %llu blocks, 40 GB of device holds %llu (%.1f%%)\n",
			(unsigned long long)pool, (unsigned long long)resident,
			100.0 * resident / pool);
		expect(resident * 20u < pool,
			"the resident set is a small fraction, so which blocks are resident "
			"is the decision that matters");
	}

	printf("\nthe JIT sequence: plan, protect, fetch, complete\n");
	{
		LmCache *cache = make_cache(64u, 8u);
		LmCachePrefetchPlan plan;
		uint32_t want[6], index;
		int32_t hit, fetch;
		for (index = 0; index < 4u; ++index)
			LmCacheAcquire(cache, 0u, index, &want[index], &hit);
		LmCacheMakeResident(cache, want[0], &fetch);
		LmCacheMakeResident(cache, want[1], &fetch);
		want[4] = want[2];
		want[5] = want[3];
		g_unresolvable = want[3];
		expect(LmCachePlanPrefetch(cache, want, 6u, resolve_source, cache, &plan) == LM_CACHE_OK,
			"a plan is built");
		printf("      %u to fetch, %u resident, %u duplicate, %u missing\n",
			plan.prefetch_count, plan.resident_count, plan.duplicate_count,
			plan.missing_count);
		expect(plan.resident_count == 2u, "resident blocks are not fetched again");
		expect(plan.duplicate_count == 1u,
			"a block named twice costs one fetch, not two - which is what makes "
			"a shared prefix cost its length rather than its length times the "
			"number of sequences on it");
		expect(plan.prefetch_count > 0u && plan.blocks[0].lane < LM_CACHE_PREFETCH_LANES,
			"and fetches are spread across lanes");

		LmCacheProtectPlan(cache, &plan, 1);
		expect(cache->blocks[plan.blocks[0].block].protected_from_eviction == 1u,
			"the plan's blocks are protected BEFORE the fetches are issued");
		for (index = 0; index < 40u; ++index)
		{
			uint32_t other;
			LmCacheAcquire(cache, 0u, 100u + index, &other, &hit);
			LmCacheMakeResident(cache, other, &fetch);
		}
		LmCacheCompletePlan(cache, &plan);
		expect(cache->blocks[plan.blocks[0].block].resident == 1u,
			"so a fetch that lands after 40 competing ones still finds its block");
		LmCacheProtectPlan(cache, &plan, 0);
	}

	printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
