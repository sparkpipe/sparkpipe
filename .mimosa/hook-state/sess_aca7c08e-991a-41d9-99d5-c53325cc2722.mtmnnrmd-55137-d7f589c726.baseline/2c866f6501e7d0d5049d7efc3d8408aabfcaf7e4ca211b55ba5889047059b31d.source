#include "sparkpipe/spark_kv_store.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static SparkStatus SparkTestKvStoreInitialize(
	const SparkKvStoreConfiguration *configuration,
	void **store_state_out)
{
	if (SparkKvStoreValidateConfiguration(configuration) != SPARK_STATUS_OK ||
		store_state_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*store_state_out = (void *)configuration;
	return SPARK_STATUS_OK;
}

static void SparkTestKvStoreDestroy(void *store_state)
{
	(void)store_state;
}

static SparkStatus SparkTestKvStoreSubmit(
	void *store_state,
	const SparkKvStoreBatch *batch)
{
	if (store_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkKvStoreValidateBatch(batch);
}

static SparkStatus SparkTestKvStorePoll(
	void *store_state,
	uint64_t batch_id,
	SparkKvStoreCompletion *completion)
{
	if (store_state == 0 || batch_id == 0u || completion == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(completion,0,sizeof(*completion));
	completion->abi_version = SPARK_KV_STORE_ABI_VERSION;
	completion->descriptor_bytes = SPARK_KV_STORE_COMPLETION_BYTES;
	completion->status = SPARK_STATUS_OK;
	completion->batch_id = batch_id;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTestKvStoreAllocateBuffer(
	void *store_state,
	uint64_t buffer_bytes,
	void **buffer_out)
{
	if (store_state == 0 || buffer_bytes == 0u || buffer_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*buffer_out = store_state;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTestKvStoreReleaseBuffer(
	void *store_state,
	void *buffer)
{
	return store_state != 0 && buffer != 0
		? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
}

static void SparkTestKvStoreLookahead(void)
{
	uint32_t cumulative_counts[3u] = {4u,7u,11u};
	assert(SparkKvStoreNormalizeLookaheadPacketCount(0u,8u) == 3u);
	assert(SparkKvStoreNormalizeLookaheadPacketCount(0u,2u) == 2u);
	assert(SparkKvStoreNormalizeLookaheadPacketCount(7u,4u) == 4u);
	assert(SparkKvStoreNormalizeLookaheadPacketCount(99u,99u) == 8u);
	assert(SparkKvStoreNormalizeLookaheadPacketCount(3u,0u) == 0u);
	assert(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
		3u,3u,32u,16u,16u,cumulative_counts) == 3u);
	assert(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
		3u,3u,20u,12u,16u,cumulative_counts) == 2u);
	assert(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
		3u,3u,20u,16u,16u,cumulative_counts) == 1u);
	assert(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
		3u,3u,20u,20u,16u,cumulative_counts) == 1u);
	assert(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
		3u,3u,32u,16u,6u,cumulative_counts) == 1u);
	assert(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
		3u,0u,32u,16u,16u,cumulative_counts) == 0u);
}

static void SparkTestKvStoreContracts(void)
{
	SparkKvStoreConfiguration configuration;
	SparkKvStoreInterface store_interface;
	SparkKvStoreBatch batch;
	uint8_t payload[64u];

	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_STORE_CONFIGURATION_BYTES;
	configuration.rank_index = 2u;
	configuration.first_layer_index = 12u;
	configuration.layer_count = 6u;
	configuration.worker_count = 3u;
	configuration.maximum_inflight_batch_count = 8u;
	configuration.maximum_batch_block_count = 128u;
	configuration.model_fingerprint = 0x11223344u;
	configuration.cache_layout_fingerprint = 0x55667788u;
	configuration.service_address = "127.0.0.1:50052";
	assert(SparkKvStoreValidateConfiguration(&configuration) ==
		SPARK_STATUS_OK);

	memset(&store_interface,0,sizeof(store_interface));
	store_interface.abi_version = SPARK_KV_STORE_ABI_VERSION;
	store_interface.descriptor_bytes = SPARK_KV_STORE_INTERFACE_BYTES;
	store_interface.capability_flags = SPARK_KV_STORE_REQUIRED_CAPS;
	store_interface.initialize = SparkTestKvStoreInitialize;
	store_interface.destroy = SparkTestKvStoreDestroy;
	store_interface.submit = SparkTestKvStoreSubmit;
	store_interface.poll = SparkTestKvStorePoll;
	store_interface.allocate_buffer = SparkTestKvStoreAllocateBuffer;
	store_interface.release_buffer = SparkTestKvStoreReleaseBuffer;
	assert(SparkKvStoreValidateInterface(
		&store_interface,SPARK_KV_STORE_REQUIRED_CAPS) == SPARK_STATUS_OK);

	memset(&batch,0,sizeof(batch));
	batch.abi_version = SPARK_KV_STORE_ABI_VERSION;
	batch.descriptor_bytes = SPARK_KV_STORE_BATCH_BYTES;
	batch.block_count = 1u;
	batch.batch_id = 9u;
	batch.blocks[0u].operation = SPARK_KV_STORE_OPERATION_GET;
	batch.blocks[0u].key_bytes = 7u;
	batch.blocks[0u].payload_bytes = sizeof(payload);
	batch.blocks[0u].payload = payload;
	memcpy(batch.blocks[0u].key,"kv/test",sizeof("kv/test"));
	assert(SparkKvStoreValidateBatch(&batch) == SPARK_STATUS_OK);
	batch.blocks[0u].key[7u] = 'x';
	assert(SparkKvStoreValidateBatch(&batch) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
	SparkTestKvStoreLookahead();
	SparkTestKvStoreContracts();
	printf("test_kv_store: ok\n");
	return 0;
}
