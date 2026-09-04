#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_model_table.h"

#define TEST_BLOCK_TOKENS 64u
#define TEST_LOGICAL_BLOCKS 4u
#define TEST_LAYER_COUNT 1u
#define TEST_HEAD_DIM 128u
#define TEST_BYTES_PER_SCALAR 2u
#define TEST_BLOCK_BYTES \
	((uint64_t)TEST_BLOCK_TOKENS * TEST_HEAD_DIM * TEST_BYTES_PER_SCALAR * \
	 TEST_LAYER_COUNT)

static void SparkTestFillTable(
	SparkKvModelTable *table,
	SparkKvCacheBlock *blocks,
	uint32_t *resident_slots,
	SparkKvPageCacheEntry *entries,
	SparkKvPageCacheSequence *sequences,
	uint32_t *hash_heads,
	uint32_t *entry_indices,
	const char *backing_path,
	uint8_t *staging)
{
	memset(table,0,sizeof(*table));
	table->abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table->descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;

	table->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table->capacity_request.descriptor_bytes =
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	table->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	table->capacity_request.layer_count = TEST_LAYER_COUNT;
	table->capacity_request.head_count = 1u;
	table->capacity_request.query_key_head_dimension = TEST_HEAD_DIM;
	table->capacity_request.value_head_dimension = TEST_HEAD_DIM;

	table->arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table->arena_configuration.descriptor_bytes =
		SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table->arena_configuration.logical_block_count = TEST_LOGICAL_BLOCKS;
	table->arena_configuration.block_token_count = TEST_BLOCK_TOKENS;
	table->arena_configuration.resident_block_capacity = TEST_LOGICAL_BLOCKS;
	table->arena_configuration.layer_count = TEST_LAYER_COUNT;
	table->arena_configuration.kv_head_count = 1u;
	table->arena_configuration.head_dim = TEST_HEAD_DIM;
	table->arena_configuration.bytes_per_scalar = TEST_BYTES_PER_SCALAR;
	table->arena_configuration.key_device_base =
		(void *)(uintptr_t)0x100000000ull;
	table->arena_configuration.blocks = blocks;
	table->arena_configuration.resident_slot_logical_block_indices =
		resident_slots;

	table->page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table->page_store_config.descriptor_bytes =
		SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table->page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	table->page_store_config.logical_page_capacity = TEST_LOGICAL_BLOCKS;
	table->page_store_config.transfer_capacity = 2u;
	table->page_store_config.page_bytes = TEST_BLOCK_BYTES;
	table->page_store_config.maximum_backing_bytes = 2u * TEST_BLOCK_BYTES;
	table->page_store_config.backing_path = backing_path;
	table->page_store_config.staging_address = staging;
	table->page_store_config.staging_bytes = TEST_BLOCK_BYTES;

	table->sequence_capacity = TEST_LOGICAL_BLOCKS;
	table->entry_capacity = TEST_LOGICAL_BLOCKS;
	table->hash_bucket_count = TEST_LOGICAL_BLOCKS;
	table->entries = entries;
	table->sequences = sequences;
	table->hash_bucket_heads = hash_heads;
	table->entry_indices_by_logical_page = entry_indices;

	table->model_id = "test-model";
	table->model_revision = "test-revision";
	table->cache_layout_fingerprint = "test-layout";
}

static void SparkTestBackendInitializesFromTable(void)
{
	SparkKvModelTable table;
	SparkKvCacheBlock blocks[TEST_LOGICAL_BLOCKS];
	uint32_t resident_slots[TEST_LOGICAL_BLOCKS];
	SparkKvPageCacheEntry entries[TEST_LOGICAL_BLOCKS];
	SparkKvPageCacheSequence sequences[TEST_LOGICAL_BLOCKS];
	uint32_t hash_heads[TEST_LOGICAL_BLOCKS];
	uint32_t entry_indices[TEST_LOGICAL_BLOCKS];
	uint8_t staging[TEST_BLOCK_BYTES];
	SparkKvCacheArena arena;
	SparkKvPageCache page_cache;
	SparkKvPageStore page_store;
	SparkKvCacheBlockView view;
	uint32_t block;
	char path[] = "/tmp/sparkpipe-kv-model-table-XXXXXX";
	int32_t descriptor = mkstemp(path);
	assert(descriptor >= 0);
	assert(close(descriptor) == 0);
	assert(unlink(path) == 0);

	SparkTestFillTable(&table,blocks,resident_slots,entries,sequences,
		hash_heads,entry_indices,path,staging);
	assert(SparkKvModelTableValidate(&table) == SPARK_STATUS_OK);
	assert(SparkKvBackendInitialize(&table,&arena,&page_cache,&page_store) ==
		SPARK_STATUS_OK);
	assert(arena.abi_version == SPARK_KV_CACHE_ABI_VERSION);
	assert(page_cache.abi_version == SPARK_KV_PAGE_CACHE_ABI_VERSION);
	assert(page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION);
	assert(SparkKvCacheArenaAcquireBlock(&arena,&block) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&arena,block) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaResolveBlock(&arena,block,&view) == SPARK_STATUS_OK);
	assert(view.key_device_address == (uintptr_t)0x100000000ull);
	assert((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
	SparkKvPageStoreDestroy(&page_store);
	printf("  backend: arena + page store + page directory from one table\n");
}

static void SparkTestValidationRejectsBadTables(void)
{
	SparkKvModelTable table;
	SparkKvCacheBlock blocks[TEST_LOGICAL_BLOCKS];
	uint32_t resident_slots[TEST_LOGICAL_BLOCKS];
	SparkKvPageCacheEntry entries[TEST_LOGICAL_BLOCKS];
	SparkKvPageCacheSequence sequences[TEST_LOGICAL_BLOCKS];
	uint32_t hash_heads[TEST_LOGICAL_BLOCKS];
	uint32_t entry_indices[TEST_LOGICAL_BLOCKS];
	uint8_t staging[TEST_BLOCK_BYTES];
	char path[] = "/tmp/sparkpipe-kv-model-table-XXXXXX";
	int32_t descriptor = mkstemp(path);
	assert(descriptor >= 0);
	assert(close(descriptor) == 0);
	assert(unlink(path) == 0);

	SparkTestFillTable(&table,blocks,resident_slots,entries,sequences,
		hash_heads,entry_indices,path,staging);

	table.abi_version = 999u;
	assert(SparkKvModelTableValidate(&table) == SPARK_STATUS_ABI_MISMATCH);
	table.abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;

	table.sequence_capacity = 0u;
	assert(SparkKvModelTableValidate(&table) == SPARK_STATUS_INVALID_ARGUMENT);
	table.sequence_capacity = TEST_LOGICAL_BLOCKS;

	table.entry_capacity = TEST_LOGICAL_BLOCKS + 1u;
	assert(SparkKvModelTableValidate(&table) == SPARK_STATUS_CAPACITY_EXCEEDED);
	table.entry_capacity = TEST_LOGICAL_BLOCKS;

	assert(SparkKvModelTableValidate(&table) == SPARK_STATUS_OK);
	printf("  validate: ABI, capacity, and entry-vs-arena bounds fail closed\n");
}

int main(void)
{
	SparkTestBackendInitializesFromTable();
	SparkTestValidationRejectsBadTables();
	printf("\nkv model table: token-free seam validates and initializes\n");
	return 0;
}
