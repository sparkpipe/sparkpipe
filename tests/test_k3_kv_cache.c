#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_k3_kv_geometry.h"
#include "sparkpipe/spark_state_pool.h"

#define K3_TEST_CONTEXT_TOKENS 4096u
#define K3_TEST_BLOCK_TOKENS 64u
#define K3_TEST_RESIDENT_SEQUENCES 16u

static uint32_t K3TestFailures;

static void K3TestCheck(int condition, const char *what)
{
	if ( condition )
		return;
	printf("  FAIL %s\n", what);
	K3TestFailures++;
}

static void K3TestFillerPinsTheAbiContract(void)
{
	SparkKvCacheCapacityRequest request;
	memset(&request,0,sizeof(request));
	SparkK3KvFillCapacityRequest(&request);
	K3TestCheck(request.abi_version == SPARK_KV_CACHE_ABI_VERSION,
		"request.abi_version == SPARK_KV_CACHE_ABI_VERSION");
	K3TestCheck(request.descriptor_bytes ==
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES,
		"request.descriptor_bytes == SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES");
	K3TestCheck(request.layout == SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE,
		"request.layout == SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE");
	K3TestCheck(request.layer_count == SPARK_K3_KV_MLA_LAYER_COUNT,
		"request.layer_count == SPARK_K3_KV_MLA_LAYER_COUNT");
	K3TestCheck(request.compressed_dimension == SPARK_K3_KV_LATENT_DIMENSION,
		"request.compressed_dimension == SPARK_K3_KV_LATENT_DIMENSION");
	K3TestCheck(request.position_dimension == SPARK_K3_KV_ROPE_DIMENSION,
		"request.position_dimension == SPARK_K3_KV_ROPE_DIMENSION");
	K3TestCheck(request.bytes_per_scalar == SPARK_K3_KV_BYTES_PER_SCALAR,
		"request.bytes_per_scalar == SPARK_K3_KV_BYTES_PER_SCALAR");
	printf("  filler: abi %u, %u latent layers, %u+%u latent token\n",
		(unsigned)request.abi_version,(unsigned)request.layer_count,
		(unsigned)request.compressed_dimension,
		(unsigned)request.position_dimension);
}

static void K3TestEstimatorRejectsStaleContracts(void)
{
	SparkKvCacheCapacityRequest request;
	SparkKvCacheCapacityEstimate estimate;
	memset(&request,0,sizeof(request));
	SparkK3KvFillCapacityRequest(&request);
	request.context_token_count = K3_TEST_CONTEXT_TOKENS;
	request.block_token_count = K3_TEST_BLOCK_TOKENS;
	request.cache_bytes_per_rank = 8ull * 1024ull * 1024ull * 1024ull;
	memset(&estimate,0,sizeof(estimate));
	estimate.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	estimate.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_ESTIMATE_DESCRIPTOR_BYTES;
	request.abi_version = SPARK_KV_CACHE_ABI_VERSION + 1u;
	K3TestCheck(SparkKvCacheEstimateCapacity(&request,&estimate) ==
		SPARK_STATUS_INVALID_ARGUMENT,
		"SparkKvCacheEstimateCapacity(&request,&estimate) == SPARK_STATUS_INVALID_ARGUMENT");
	request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request.descriptor_bytes =
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES + 1u;
	K3TestCheck(SparkKvCacheEstimateCapacity(&request,&estimate) ==
		SPARK_STATUS_INVALID_ARGUMENT,
		"SparkKvCacheEstimateCapacity(&request,&estimate) == SPARK_STATUS_INVALID_ARGUMENT");
	request.descriptor_bytes =
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	K3TestCheck(SparkKvCacheEstimateCapacity(&request,&estimate) ==
		SPARK_STATUS_OK,
		"SparkKvCacheEstimateCapacity(&request,&estimate) == SPARK_STATUS_OK");
	printf("  estimator: stale abi and descriptor refused, filled pass\n");
}

static void K3TestEstimatorPricesTheLatentArena(void)
{
	SparkKvCacheCapacityRequest request;
	SparkKvCacheCapacityEstimate estimate;
	uint64_t expected_token_layer_bytes;
	memset(&request,0,sizeof(request));
	SparkK3KvFillCapacityRequest(&request);
	request.context_token_count = K3_TEST_CONTEXT_TOKENS;
	request.block_token_count = K3_TEST_BLOCK_TOKENS;
	request.cache_bytes_per_rank = 8ull * 1024ull * 1024ull * 1024ull;
	memset(&estimate,0,sizeof(estimate));
	estimate.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	estimate.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_ESTIMATE_DESCRIPTOR_BYTES;
	K3TestCheck(SparkKvCacheEstimateCapacity(&request,&estimate) == SPARK_STATUS_OK,
		"SparkKvCacheEstimateCapacity(&request,&estimate) == SPARK_STATUS_OK");
	expected_token_layer_bytes =
		(uint64_t)(SPARK_K3_KV_LATENT_DIMENSION + SPARK_K3_KV_ROPE_DIMENSION) *
		SPARK_K3_KV_BYTES_PER_SCALAR;
	K3TestCheck(estimate.attention_bytes_per_token_per_layer ==
		expected_token_layer_bytes,
		"estimate.attention_bytes_per_token_per_layer == expected_token_layer_bytes");
	K3TestCheck(estimate.index_key_bytes_per_token == 0u,
		"estimate.index_key_bytes_per_token == 0u");
	K3TestCheck(estimate.block_count_per_context ==
		K3_TEST_CONTEXT_TOKENS / K3_TEST_BLOCK_TOKENS,
		"estimate.block_count_per_context == K3_TEST_CONTEXT_TOKENS / K3_TEST_BLOCK_TOKENS");
	K3TestCheck(estimate.contexts_per_rank > 0u,
		"estimate.contexts_per_rank > 0u");
	printf("  estimator: %llu bytes/token/layer over %u MLA layers, no dsa\n",
		(unsigned long long)estimate.attention_bytes_per_token_per_layer,
		(unsigned)request.layer_count);
}

static void K3TestArenaPagesLatentTokens(void)
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[8u];
	SparkKvCacheConfiguration configuration;
	SparkKvCacheBlockView view;
	uint32_t resident_owners[8u];
	uint32_t block_index;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.logical_block_count = 8u;
	configuration.block_token_count = K3_TEST_BLOCK_TOKENS;
	configuration.resident_block_capacity = 8u;
	configuration.layer_count = SPARK_K3_KV_MLA_LAYER_COUNT;
	configuration.kv_head_count = 1u;
	configuration.head_dim =
		SPARK_K3_KV_LATENT_DIMENSION + SPARK_K3_KV_ROPE_DIMENSION;
	configuration.bytes_per_scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
	configuration.value_device_base = (void *)(uintptr_t)0x100000000ull;
	configuration.blocks = blocks;
	configuration.resident_slot_logical_block_indices = resident_owners;
	K3TestCheck(SparkKvCacheArenaInitialize(&arena,&configuration) ==
		SPARK_STATUS_OK,
		"SparkKvCacheArenaInitialize(&arena,&configuration) == SPARK_STATUS_OK");
	K3TestCheck(SparkKvCacheArenaAcquireBlock(&arena,&block_index) == SPARK_STATUS_OK,
		"SparkKvCacheArenaAcquireBlock(&arena,&block_index) == SPARK_STATUS_OK");
	K3TestCheck(SparkKvCacheArenaMarkBlockResident(&arena,block_index) ==
		SPARK_STATUS_OK,
		"SparkKvCacheArenaMarkBlockResident(&arena,block_index) == SPARK_STATUS_OK");
	memset(&view,0,sizeof(view));
	K3TestCheck(SparkKvCacheArenaResolveBlock(&arena,block_index,&view) ==
		SPARK_STATUS_OK,
		"SparkKvCacheArenaResolveBlock(&arena,block_index,&view) == SPARK_STATUS_OK");
	K3TestCheck(view.key_device_address != 0u,
		"view.key_device_address != 0u");
	K3TestCheck(view.head_dim ==
		SPARK_K3_KV_LATENT_DIMENSION + SPARK_K3_KV_ROPE_DIMENSION,
		"view.head_dim == SPARK_K3_KV_LATENT_DIMENSION + SPARK_K3_KV_ROPE_DIMENSION");
	K3TestCheck(view.key_block_stride_bytes ==
		(uint64_t)K3_TEST_BLOCK_TOKENS * view.head_dim *
		SPARK_K3_KV_BYTES_PER_SCALAR * SPARK_K3_KV_MLA_LAYER_COUNT,
		"view.key_block_stride_bytes == (uint64_t)K3_TEST_BLOCK_TOKENS * view.head_dim * SPARK_K3_KV_BYTES_PER_SCALAR * SPARK_K3_KV_MLA_LAYER_COUNT");
	printf("  arena: block %u resolved, %u-wide latent tokens paged\n",
		block_index,(unsigned)view.head_dim);
}

static void K3TestSlotPoolBudgetsKdaState(void)
{
	static uint32_t next_free[K3_TEST_RESIDENT_SEQUENCES];
	uint8_t *const base = (uint8_t *)(uintptr_t)0x100000000ull;
	SparkStatePool pool;
	uint32_t slot, index, released;
	K3TestCheck(SparkStatePoolInitialize(&pool,base,next_free,
		K3_TEST_RESIDENT_SEQUENCES,SPARK_K3_KV_KDA_SLOT_BYTES) == 0,
		"the state pool accepts the resident capacity of KDA slots");
	for (index = 0u; index < K3_TEST_RESIDENT_SEQUENCES; index++)
	{
		slot = SparkStatePoolAcquire(&pool);
		K3TestCheck(slot == index,"the pool hands out every resident slot once");
		K3TestCheck((uintptr_t)SparkStatePoolSlot(&pool,slot) ==
			(uintptr_t)base + ((uint64_t)slot * SPARK_K3_KV_KDA_SLOT_BYTES),
			"each slot is a whole KDA state of every KDA layer, 64-bit strided");
	}
	K3TestCheck(SparkStatePoolAcquire(&pool) == SPARK_STATE_POOL_NO_SLOT,
		"a full pool refuses another sequence");
	released = K3_TEST_RESIDENT_SEQUENCES - 1u;
	K3TestCheck(SparkStatePoolRelease(&pool,released) == 0 &&
		SparkStatePoolAcquire(&pool) == released,
		"a released slot is the next one reused");
	K3TestCheck((uint64_t)((uintptr_t)SparkStatePoolSlot(&pool,released) - (uintptr_t)base) ==
		(uint64_t)released * SPARK_K3_KV_KDA_SLOT_BYTES &&
		(uint64_t)released * SPARK_K3_KV_KDA_SLOT_BYTES > 0xffffffffull,
		"the last slot sits past 4 GiB without wrapping");
	printf("  kda: %llu bytes a sequence across %u layers, %u slots in %llu bytes\n",
		(unsigned long long)SPARK_K3_KV_KDA_SLOT_BYTES,
		(unsigned)SPARK_K3_KV_KDA_LAYER_COUNT,K3_TEST_RESIDENT_SEQUENCES,
		(unsigned long long)SPARK_K3_KV_KDA_SLOT_BYTES * K3_TEST_RESIDENT_SEQUENCES);
}

int main(void)
{
	K3TestFillerPinsTheAbiContract();
	K3TestEstimatorRejectsStaleContracts();
	K3TestEstimatorPricesTheLatentArena();
	K3TestArenaPagesLatentTokens();
	K3TestSlotPoolBudgetsKdaState();
	if ( K3TestFailures != 0u )
	{
		printf("\nFAIL (%u)\n",K3TestFailures);
		return 1;
	}
	printf("\nk3 crosses the kv seam on the common machinery alone\n");
	return 0;
}
