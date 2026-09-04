#include "sparkpipe/spark_kv_store.h"

#include <dlfcn.h>
#include <string.h>

uint32_t SparkKvStoreNormalizeLookaheadPacketCount(
	uint32_t lookahead_packet_count,
	uint32_t queue_depth)
{
	if (lookahead_packet_count == 0u)
		lookahead_packet_count = SPARK_KV_STORE_DEFAULT_LOOKAHEAD_PACKETS;
	if (lookahead_packet_count > SPARK_KV_STORE_MAX_LOOKAHEAD_PACKETS)
		lookahead_packet_count = SPARK_KV_STORE_MAX_LOOKAHEAD_PACKETS;
	if (lookahead_packet_count > queue_depth)
		lookahead_packet_count = queue_depth;
	return lookahead_packet_count;
}

uint32_t SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
	uint32_t lookahead_packet_count,
	uint32_t queue_depth,
	uint32_t physical_block_capacity,
	uint32_t allocated_physical_block_count,
	uint32_t staging_block_capacity,
	const uint32_t *cumulative_nonresident_block_counts)
{
	uint32_t candidate_count,free_physical_block_count,selected_count;
	lookahead_packet_count = SparkKvStoreNormalizeLookaheadPacketCount(
		lookahead_packet_count,queue_depth);
	if (lookahead_packet_count == 0u ||
		cumulative_nonresident_block_counts == 0)
		return 0u;
	selected_count = 1u;
	if (physical_block_capacity == 0u ||
		allocated_physical_block_count >= physical_block_capacity ||
		staging_block_capacity == 0u)
		return selected_count;
	free_physical_block_count =
		physical_block_capacity - allocated_physical_block_count;
	for (candidate_count = 2u;
		 candidate_count <= lookahead_packet_count;
		 ++candidate_count)
	{
		uint32_t cumulative_block_count;
		cumulative_block_count =
			cumulative_nonresident_block_counts[candidate_count - 1u];
		if (cumulative_block_count > free_physical_block_count ||
			cumulative_block_count > staging_block_capacity)
			break;
		selected_count = candidate_count;
	}
	return selected_count;
}

SparkStatus SparkKvStoreValidateConfiguration(
	const SparkKvStoreConfiguration *configuration)
{
	if (configuration == 0 ||
		configuration->abi_version != SPARK_KV_STORE_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_KV_STORE_CONFIGURATION_BYTES ||
		configuration->layer_count == 0u ||
		configuration->worker_count == 0u ||
		configuration->maximum_inflight_batch_count == 0u ||
		configuration->maximum_inflight_batch_count >
			SPARK_KV_STORE_MAX_INFLIGHT_BATCHES ||
		configuration->maximum_batch_block_count == 0u ||
		configuration->maximum_batch_block_count >
			SPARK_KV_STORE_MAX_BATCH_BLOCKS ||
		configuration->model_fingerprint == 0u ||
		configuration->cache_layout_fingerprint == 0u ||
		configuration->service_address == 0 ||
		configuration->service_address[0] == '\0')
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkKvStoreValidateBatch(
	const SparkKvStoreBatch *batch)
{
	uint32_t block_index;
	if (batch == 0 || batch->abi_version != SPARK_KV_STORE_ABI_VERSION ||
		batch->descriptor_bytes != SPARK_KV_STORE_BATCH_BYTES ||
		batch->batch_id == 0u || batch->block_count == 0u ||
		batch->block_count > SPARK_KV_STORE_MAX_BATCH_BLOCKS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (block_index = 0u; block_index < batch->block_count; ++block_index)
	{
		const SparkKvStoreBlock *block;
		block = &batch->blocks[block_index];
		if ((block->operation != SPARK_KV_STORE_OPERATION_GET &&
			 block->operation != SPARK_KV_STORE_OPERATION_PUT) ||
			block->key_bytes == 0u ||
			block->key_bytes >= SPARK_KV_STORE_MAX_KEY_BYTES ||
			block->key[block->key_bytes] != '\0' ||
			block->payload_bytes == 0u || block->payload == 0)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkKvStoreValidateInterface(
	const SparkKvStoreInterface *store_interface,
	uint32_t required_capability_flags)
{
	if (store_interface == 0 ||
		store_interface->abi_version != SPARK_KV_STORE_ABI_VERSION ||
		store_interface->descriptor_bytes != SPARK_KV_STORE_INTERFACE_BYTES ||
		(store_interface->capability_flags & required_capability_flags) !=
			required_capability_flags ||
		store_interface->initialize == 0 || store_interface->destroy == 0 ||
		store_interface->submit == 0 || store_interface->poll == 0 ||
		store_interface->allocate_buffer == 0 ||
		store_interface->release_buffer == 0)
		return SPARK_STATUS_ABI_MISMATCH;
	return SPARK_STATUS_OK;
}

SparkStatus SparkKvStoreLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkKvStoreDynamicLibrary *library)
{
	SparkKvStoreGetInterfaceFunction get_interface;
	const SparkKvStoreInterface *store_interface;
	void *dynamic_library;
	SparkStatus status;
	if (shared_object_path == 0 || shared_object_path[0] == '\0' || library == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(library,0,sizeof(*library));
	dynamic_library = dlopen(shared_object_path,RTLD_NOW | RTLD_LOCAL);
	if (dynamic_library == 0)
		return SPARK_STATUS_NOT_FOUND;
	*(void **)(&get_interface) = dlsym(
		dynamic_library,SPARK_KV_STORE_INTERFACE_SYMBOL);
	if (get_interface == 0)
	{
		dlclose(dynamic_library);
		return SPARK_STATUS_ABI_MISMATCH;
	}
	store_interface = get_interface();
	status = SparkKvStoreValidateInterface(
		store_interface,required_capability_flags);
	if (status != SPARK_STATUS_OK)
	{
		dlclose(dynamic_library);
		return status;
	}
	library->dynamic_library = dynamic_library;
	library->store_interface = *store_interface;
	return SPARK_STATUS_OK;
}

void SparkKvStoreUnloadInterface(
	SparkKvStoreDynamicLibrary *library)
{
	if (library == 0)
		return;
	if (library->dynamic_library != 0)
		dlclose(library->dynamic_library);
	memset(library,0,sizeof(*library));
}
