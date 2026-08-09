#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_stage_module_common.h"
#include "spark_dsv4_paged_cache.h"

static void SparkTestDsv4Identity(
	SparkModelDriverCacheIdentity *identity,
	uint8_t seed)
{
	uint32_t index;
	memset(identity,0,sizeof(*identity));
	for (index=0u; index<sizeof(identity->sha256); index++)
		identity->sha256[index] = (uint8_t)(seed + index);
}

static void SparkTestDsv4Lane(
	SparkModelDriverCacheLane *lane,
	uint64_t sequence_id,
	uint32_t sequence_position,
	uint32_t context_token_count)
{
	memset(lane,0,sizeof(*lane));
	lane->sequence_id = sequence_id;
	lane->sequence_position = sequence_position;
	lane->resident_sequence_slot = 0u;
	lane->context_token_count = context_token_count;
}

int main(void)
{
	SparkDsv4PagedCacheConfiguration configuration;
	SparkDsv4PagedCacheLane prepared;
	SparkDsv4PagedCache cache;
	SparkModelDriverCacheLane lane;
	SparkStageModuleLedger ledger;
	uint32_t logical_pages[2u],physical_pages[2u],page_count;
	memset(&ledger,0,sizeof(ledger));
	ledger.module_tag = "test.dsv4.paged_cache";
	memset(&configuration,0,sizeof(configuration));
	configuration.first_layer_index = 0u;
	configuration.layer_count = 1u;
	configuration.resident_sequence_capacity = 4u;
	configuration.maximum_sequence_positions = 256u;
	configuration.logical_page_capacity = 8u;
	configuration.physical_page_capacity = 2u;
	assert(SparkDsv4PagedCacheInitialize(&cache,&configuration,&ledger) ==
		SPARK_STATUS_OK);
	SparkTestDsv4Lane(&lane,1u,0u,128u);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	lane.publish_token_count = 128u;
	SparkTestDsv4Identity(&lane.publish_identity,17u);
	assert(SparkDsv4PagedCachePrefetchLane(&cache,&lane,logical_pages,2u) ==
		SPARK_STATUS_OK);
	assert(cache.page_cache.live_sequence_count == 0u);
	assert(SparkDsv4PagedCachePrepareLane(&cache,&lane,logical_pages,
		physical_pages,2u,&prepared) == SPARK_STATUS_OK);
	assert(prepared.logical_page_count == 1u);
	assert(prepared.requires_initialization == 1u);
	assert(prepared.mutation_flags ==
		(SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE |
		 SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE));
	assert(cache.blocks[logical_pages[0u]].residency_reference_count == 1u);
	assert(SparkDsv4PagedCacheUnpinLane(&cache,logical_pages,1u) ==
		SPARK_STATUS_OK);
	assert(SparkDsv4PagedCacheRollbackLane(&cache,&lane,&prepared) ==
		SPARK_STATUS_OK);
	assert(cache.page_cache.live_sequence_count == 0u);
	assert(SparkDsv4PagedCachePrepareLane(&cache,&lane,logical_pages,
		physical_pages,2u,&prepared) == SPARK_STATUS_OK);
	assert(SparkDsv4PagedCacheUnpinLane(&cache,logical_pages,1u) ==
		SPARK_STATUS_OK);
	assert(SparkDsv4PagedCacheCompleteLane(&cache,&lane,&prepared) ==
		SPARK_STATUS_OK);
	assert(cache.page_cache.live_sequence_count == 1u);
	assert(cache.physical_page_content_logical_pages[
		prepared.mutable_physical_page] == prepared.mutable_logical_page);
	cache.physical_page_content_logical_pages[1u] =
		prepared.mutable_logical_page;
	cache.physical_page_generations[1u] = prepared.mutable_generation;
	SparkTestDsv4Lane(&lane,1u,128u,129u);
	assert(SparkDsv4PagedCachePrepareLane(&cache,&lane,logical_pages,
		physical_pages,2u,&prepared) == SPARK_STATUS_OK);
	assert(prepared.logical_page_count == 2u);
	assert(prepared.mutable_physical_page == 1u);
	assert(prepared.requires_initialization == 1u);
	assert(prepared.mutation_flags ==
		SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE);
	assert(SparkDsv4PagedCacheUnpinLane(&cache,logical_pages,2u) ==
		SPARK_STATUS_OK);
	assert(SparkDsv4PagedCacheRollbackLane(&cache,&lane,&prepared) ==
		SPARK_STATUS_OK);
	assert(SparkKvPageCacheBuildLaneTable(&cache.page_cache,0u,1u,
		logical_pages,2u,&page_count) == SPARK_STATUS_OK);
	assert(page_count == 1u);
	SparkDsv4PagedCacheDestroyHost(&cache);
	SparkStageModuleLedgerRelease(&ledger);
	return(0);
}
