#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_STORE_ABI_VERSION 2u
#define SPARK_KV_STORE_INTERFACE_SYMBOL \
	"SparkKvStoreGetInterface"
#define SPARK_KV_STORE_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkKvStoreInterface))
#define SPARK_KV_STORE_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkKvStoreConfiguration))
#define SPARK_KV_STORE_BATCH_BYTES \
	((uint32_t)sizeof(SparkKvStoreBatch))
#define SPARK_KV_STORE_COMPLETION_BYTES \
	((uint32_t)sizeof(SparkKvStoreCompletion))
#define SPARK_KV_STORE_MAX_KEY_BYTES 192u
#define SPARK_KV_STORE_MAX_BATCH_BLOCKS 128u
#define SPARK_KV_STORE_MAX_INFLIGHT_BATCHES 8u
#define SPARK_KV_STORE_DEFAULT_LOOKAHEAD_PACKETS 3u
#define SPARK_KV_STORE_MAX_LOOKAHEAD_PACKETS 8u

#define SPARK_KV_STORE_CAP_BATCH_GET 0x00000001u
#define SPARK_KV_STORE_CAP_BATCH_PUT 0x00000002u
#define SPARK_KV_STORE_CAP_PERSISTENT_SERVICE 0x00000004u
#define SPARK_KV_STORE_CAP_PROVIDER_BUFFERS 0x00000008u
#define SPARK_KV_STORE_REQUIRED_CAPS \
	(SPARK_KV_STORE_CAP_BATCH_GET | \
	 SPARK_KV_STORE_CAP_BATCH_PUT | \
	 SPARK_KV_STORE_CAP_PERSISTENT_SERVICE | \
	 SPARK_KV_STORE_CAP_PROVIDER_BUFFERS)

#define SPARK_KV_STORE_OPERATION_GET 1u
#define SPARK_KV_STORE_OPERATION_PUT 2u

typedef struct SparkKvStoreConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t worker_count;
	uint32_t maximum_inflight_batch_count;
	uint32_t maximum_batch_block_count;
	uint64_t model_fingerprint;
	uint64_t cache_layout_fingerprint;
	uint64_t client_memory_pool_bytes;
	uint64_t local_buffer_bytes;
	const char *service_address;
	const char *ipc_socket_path;
} SparkKvStoreConfiguration;

typedef struct SparkKvStoreBlock
{
	uint32_t operation;
	uint32_t key_bytes;
	uint32_t payload_bytes;
	uint32_t reserved0;
	void *payload;
	char key[SPARK_KV_STORE_MAX_KEY_BYTES];
} SparkKvStoreBlock;

typedef struct SparkKvStoreBatch
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_count;
	uint32_t priority;
	uint64_t batch_id;
	SparkKvStoreBlock blocks[SPARK_KV_STORE_MAX_BATCH_BLOCKS];
} SparkKvStoreBatch;

typedef struct SparkKvStoreCompletion
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	SparkStatus status;
	uint32_t completed_block_count;
	uint64_t batch_id;
} SparkKvStoreCompletion;

typedef SparkStatus (*SparkKvStoreInitializeFunction)(
	const SparkKvStoreConfiguration *configuration,
	void **store_state_out);
typedef void (*SparkKvStoreDestroyFunction)(void *store_state);
typedef SparkStatus (*SparkKvStoreSubmitFunction)(
	void *store_state,
	const SparkKvStoreBatch *batch);
typedef SparkStatus (*SparkKvStorePollFunction)(
	void *store_state,
	uint64_t batch_id,
	SparkKvStoreCompletion *completion);
typedef SparkStatus (*SparkKvStoreAllocateBufferFunction)(
	void *store_state,
	uint64_t buffer_bytes,
	void **buffer_out);
typedef SparkStatus (*SparkKvStoreReleaseBufferFunction)(
	void *store_state,
	void *buffer);

typedef struct SparkKvStoreInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t reserved0;
	SparkKvStoreInitializeFunction initialize;
	SparkKvStoreDestroyFunction destroy;
	SparkKvStoreSubmitFunction submit;
	SparkKvStorePollFunction poll;
	SparkKvStoreAllocateBufferFunction allocate_buffer;
	SparkKvStoreReleaseBufferFunction release_buffer;
} SparkKvStoreInterface;

typedef const SparkKvStoreInterface *(
	*SparkKvStoreGetInterfaceFunction)(void);

typedef struct SparkKvStoreDynamicLibrary
{
	void *dynamic_library;
	SparkKvStoreInterface store_interface;
} SparkKvStoreDynamicLibrary;

uint32_t SparkKvStoreNormalizeLookaheadPacketCount(
	uint32_t lookahead_packet_count,
	uint32_t queue_depth);
uint32_t SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
	uint32_t lookahead_packet_count,
	uint32_t queue_depth,
	uint32_t physical_block_capacity,
	uint32_t allocated_physical_block_count,
	uint32_t staging_block_capacity,
	const uint32_t *cumulative_nonresident_block_counts);
SparkStatus SparkKvStoreValidateConfiguration(
	const SparkKvStoreConfiguration *configuration);
SparkStatus SparkKvStoreValidateBatch(
	const SparkKvStoreBatch *batch);
SparkStatus SparkKvStoreValidateInterface(
	const SparkKvStoreInterface *store_interface,
	uint32_t required_capability_flags);
SparkStatus SparkKvStoreLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkKvStoreDynamicLibrary *library);
void SparkKvStoreUnloadInterface(
	SparkKvStoreDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
