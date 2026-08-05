#include "sparkpipe/spark_stage_kv_client.h"

#include <stdio.h>
#include <string.h>

SparkStatus SparkStageKvClientOpen(SparkStageKvClient *client, const char *module_tag, const char *provider_path_or_none, uint32_t rank_index, uint32_t first_layer_index, uint32_t layer_count, uint64_t model_fingerprint, uint64_t cache_layout_fingerprint, const char *service_address, const char *ipc_socket_path, uint64_t client_memory_pool_bytes, uint32_t worker_count)
{
	SparkKvStoreConfiguration configuration;
	SparkStatus status;
	if ( client == 0 || module_tag == 0 || provider_path_or_none == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(client,0,sizeof(*client));
	client->module_tag = module_tag;
	client->next_batch_id = 1u;
	if ( strcmp(provider_path_or_none,"none") == 0 )
		return(SPARK_STATUS_OK);
	status = SparkKvStoreLoadInterfaceFromSharedObject(provider_path_or_none,SPARK_KV_STORE_REQUIRED_CAPS,&client->library);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv_provider_load_failed path=%s\n",module_tag,provider_path_or_none);
		return(status);
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_STORE_CONFIGURATION_BYTES;
	configuration.rank_index = rank_index;
	configuration.first_layer_index = first_layer_index;
	configuration.layer_count = layer_count;
	configuration.worker_count = worker_count;
	configuration.maximum_inflight_batch_count = SPARK_KV_STORE_MAX_INFLIGHT_BATCHES;
	configuration.maximum_batch_block_count = SPARK_KV_STORE_MAX_BATCH_BLOCKS;
	configuration.model_fingerprint = model_fingerprint;
	configuration.cache_layout_fingerprint = cache_layout_fingerprint;
	configuration.client_memory_pool_bytes = client_memory_pool_bytes;
	configuration.local_buffer_bytes = client_memory_pool_bytes;
	configuration.service_address = service_address;
	configuration.ipc_socket_path = ipc_socket_path;
	status = client->library.store_interface.initialize(&configuration,&client->store_state);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv_provider_initialize_failed\n",module_tag);
		SparkKvStoreUnloadInterface(&client->library);
		return(status);
	}
	client->enabled = 1u;
	fprintf(stderr,"%s kv_tier_enabled rank=%u slice=%u+%u\n",module_tag,rank_index,first_layer_index,layer_count);
	return(SPARK_STATUS_OK);
}

// <family>/<model_fp>/<layout_fp>/r<rank>/s<sequence>/b<block> - the same
// binding scheme used by every tier: a model or layout change can never
// consume old KV, and rank scoping keeps stages from crossing streams.
int32_t SparkStageKvClientFormatKey(char *key, uint32_t key_capacity, uint64_t model_fingerprint, uint64_t cache_layout_fingerprint, uint32_t rank_index, uint64_t sequence_id, uint32_t logical_block)
{
	int written = snprintf(key,key_capacity,"kv/%016llx/%016llx/r%u/s%llu/b%u",(unsigned long long)model_fingerprint,(unsigned long long)cache_layout_fingerprint,rank_index,(unsigned long long)sequence_id,logical_block);
	if ( written <= 0 || (uint32_t)written >= key_capacity || (uint32_t)written >= SPARK_KV_STORE_MAX_KEY_BYTES )
		return(-1);
	return(written);
}

SparkStatus SparkStageKvClientSubmit(SparkStageKvClient *client, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority, uint64_t *batch_id)
{
	SparkKvStoreBatch batch;
	SparkStatus status;
	uint32_t index;
	if ( client == 0 || blocks == 0 || batch_id == 0 || block_count == 0u || block_count > SPARK_KV_STORE_MAX_BATCH_BLOCKS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->enabled == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&batch,0,sizeof(batch));
	batch.abi_version = SPARK_KV_STORE_ABI_VERSION;
	batch.descriptor_bytes = SPARK_KV_STORE_BATCH_BYTES;
	batch.block_count = block_count;
	batch.priority = priority;
	batch.batch_id = client->next_batch_id;
	for (index = 0; index < block_count; index++)
	{
		batch.blocks[index] = blocks[index];
		batch.blocks[index].operation = operation;
	}
	status = client->library.store_interface.submit(client->store_state,&batch);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*batch_id = client->next_batch_id;
	client->next_batch_id++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageKvClientPoll(SparkStageKvClient *client, uint64_t batch_id, SparkKvStoreCompletion *completion)
{
	if ( client == 0 || completion == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->enabled == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	return(client->library.store_interface.poll(client->store_state,batch_id,completion));
}

SparkStatus SparkStageKvClientAllocateBuffer(SparkStageKvClient *client, uint64_t buffer_bytes, void **buffer)
{
	if ( client == 0 || buffer == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->enabled == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(client->library.store_interface.allocate_buffer(client->store_state,buffer_bytes,buffer));
}

void SparkStageKvClientClose(SparkStageKvClient *client)
{
	if ( client == 0 || client->enabled == 0u )
		return;
	client->library.store_interface.destroy(client->store_state);
	SparkKvStoreUnloadInterface(&client->library);
	client->enabled = 0u;
	client->store_state = 0;
}
