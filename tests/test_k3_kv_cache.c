// The seam, crossed: K3 builds its cache from the COMMON machinery using
// only its geometry header. The estimator prices the MLA latent arena from
// the request - 576 elements a token, 24 layers, no DSA rows - and the
// arena pages 576-wide tokens exactly as it pages glm's. The KDA side
// budgets one slab per sequence through the slot pool, at the byte sizes
// the kernel config defines. No glm symbol appears in this file: that
// absence is the test.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_k3_kv_geometry.h"

#define K3_TEST_CONTEXT_TOKENS 4096u
#define K3_TEST_BLOCK_TOKENS 64u
#define K3_TEST_SEQUENCES 4u

static void K3TestEstimatorPricesTheLatentArena(void)
{
	SparkKvCacheCapacityRequest request;
	SparkKvCacheCapacityEstimate estimate;
	uint64_t expected_token_layer_bytes;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	SparkK3KvFillCapacityRequest(&request);
	request.context_token_count = K3_TEST_CONTEXT_TOKENS;
	request.block_token_count = K3_TEST_BLOCK_TOKENS;
	request.cache_bytes_per_rank = 8ull * 1024ull * 1024ull * 1024ull;
	memset(&estimate,0,sizeof(estimate));
	estimate.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	estimate.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_ESTIMATE_DESCRIPTOR_BYTES;
	assert(SparkKvCacheEstimateCapacity(&request,&estimate) == SPARK_STATUS_OK);
	expected_token_layer_bytes =
		(uint64_t)(SPARK_K3_KV_LATENT_DIMENSION + SPARK_K3_KV_ROPE_DIMENSION) *
		SPARK_K3_KV_BYTES_PER_SCALAR;
	assert(estimate.attention_bytes_per_token_per_layer ==
		expected_token_layer_bytes);
	assert(estimate.index_key_bytes_per_token == 0u);
	assert(estimate.block_count_per_context ==
		K3_TEST_CONTEXT_TOKENS / K3_TEST_BLOCK_TOKENS);
	assert(estimate.contexts_per_rank > 0u);
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
	assert(SparkKvCacheArenaInitialize(&arena,&configuration) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaAcquireBlock(&arena,&block_index) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&arena,block_index) ==
		SPARK_STATUS_OK);
	memset(&view,0,sizeof(view));
	assert(SparkKvCacheArenaResolveBlock(&arena,block_index,&view) ==
		SPARK_STATUS_OK);
	assert(view.key_device_address != 0u);
	assert(view.head_dim ==
		SPARK_K3_KV_LATENT_DIMENSION + SPARK_K3_KV_ROPE_DIMENSION);
	assert(view.key_block_stride_bytes ==
		(uint64_t)K3_TEST_BLOCK_TOKENS * view.head_dim *
		SPARK_K3_KV_BYTES_PER_SCALAR * SPARK_K3_KV_MLA_LAYER_COUNT);
	printf("  arena: block %u resolved, %u-wide latent tokens paged\n",
		block_index,(unsigned)view.head_dim);
}

static void K3TestSlotPoolBudgetsKdaState(void)
{
	static uint8_t storage[K3_TEST_SEQUENCES * 64u];
	static uint32_t next_free[K3_TEST_SEQUENCES];
	SparkStatePool pool;
	uint32_t slot;
	// the true slab is ~6.5 MB a sequence; the pool math is what is under
	// test here, so the backing shrinks and the SIZES stay symbolic
	assert(SPARK_K3_KV_KDA_SLOT_BYTES ==
		SPARK_K3_KV_KDA_LAYER_COUNT *
		(SPARK_K3_KV_KDA_STATE_BYTES_PER_LAYER +
		 SPARK_K3_KV_KDA_CONV_BYTES_PER_LAYER));
	assert(SparkStatePoolInitialize(&pool,storage,next_free,
		K3_TEST_SEQUENCES,64u) == 0);
	slot = SparkStatePoolAcquire(&pool);
	assert(slot != SPARK_STATE_POOL_NO_SLOT);
	assert(SparkStatePoolSlot(&pool,slot) == storage + ((uint64_t)slot * 64u));
	assert(SparkStatePoolRelease(&pool,slot) == 0);
	printf("  kda: %llu bytes a sequence across %u layers, slot pool holds\n",
		(unsigned long long)SPARK_K3_KV_KDA_SLOT_BYTES,
		(unsigned)SPARK_K3_KV_KDA_LAYER_COUNT);
}

int main(void)
{
	K3TestEstimatorPricesTheLatentArena();
	K3TestArenaPagesLatentTokens();
	K3TestSlotPoolBudgetsKdaState();
	printf("\nk3 crosses the kv seam on the common machinery alone\n");
	return 0;
}
