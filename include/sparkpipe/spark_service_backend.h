#pragma once

#include <stdint.h>

#include "sparkpipe/spark_service.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SERVICE_BACKEND_ABI_VERSION 10u
#define SPARK_SERVICE_BACKEND_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkServiceBackendInterface))
#define SPARK_SERVICE_BACKEND_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkServiceBackendConfiguration))
#define SPARK_SERVICE_BACKEND_VIEW_BYTES \
	((uint32_t)sizeof(SparkServiceBackendView))
#define SPARK_SERVICE_BACKEND_DYNAMIC_LIBRARY_BYTES \
	((uint32_t)sizeof(SparkServiceBackendDynamicLibrary))
#define SPARK_SERVICE_BACKEND_INTERFACE_SYMBOL \
	"SparkServiceBackendGetInterface"
#define SPARK_SERVICE_BACKEND_BLOCKER_BYTES 256u

#define SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK 0x00000001u
#define SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP 0x00000002u
#define SPARK_SERVICE_BACKEND_CONFIGURATION_KNOWN_FLAGS \
	(SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK | \
	 SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP)

#define SPARK_SERVICE_BACKEND_CAPABILITY_SERVICE_RUNTIME 0x00000001u
#define SPARK_SERVICE_BACKEND_CAPABILITY_RING_RUNTIME 0x00000002u
#define SPARK_SERVICE_BACKEND_CAPABILITY_TOKENIZER 0x00000004u
#define SPARK_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS 0x00000008u
#define SPARK_SERVICE_BACKEND_POLL_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkServiceBackendPollDescriptor))
#define SPARK_SERVICE_BACKEND_POLL_READ 0x00000001u
#define SPARK_SERVICE_BACKEND_POLL_WRITE 0x00000002u
#define SPARK_SERVICE_BACKEND_REQUIRED_PRODUCTION_CAPS \
	(SPARK_SERVICE_BACKEND_CAPABILITY_SERVICE_RUNTIME | \
	 SPARK_SERVICE_BACKEND_CAPABILITY_RING_RUNTIME | \
	 SPARK_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS)

typedef struct SparkServiceBackendConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t max_active_sequence_count;
	uint32_t port_base;
	uint32_t kv_logical_block_capacity;
	uint32_t model_quantization_mode;
	const char *moe_pack_root;
	const char *stagepack_root;
	const char *transport_shared_object_path;
	const char *driver_shared_object_path;
	const char *node_context_builder_shared_object_path;
	const char *embedding_pack_path;
	const char *driver_program_name;
	const char *node_target;
	const char *tokenizer_path;
	const char *final_event_bind_address;
	const char *final_event_return_host;
	const char *cuda_resident_socket_path;
} SparkServiceBackendConfiguration;

typedef struct SparkServiceBackendView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t runtime_initialized;
	uint32_t local_control_ready;
	uint32_t configured_kv_context_limit_tokens;
	uint32_t configured_max_active_sequences;
	uint32_t transport_capability_flags;
	uint32_t speculation_configuration_flags;
	uint32_t request_api_configuration_flags;
	uint32_t adaptive_decode_batch_width;
	uint32_t decode_batch_capacity;
	uint32_t prefill_wave_token_count;
	uint64_t release_generation;
	SparkServiceRuntime *service;
	const SparkTokenizer *tokenizer;
	const char *first_blocker;
	const char *release_id;
	const char *release_git_commit;
	const char *transport_shared_object_path;
} SparkServiceBackendView;

typedef struct SparkServiceBackendPollDescriptor
{
	uint32_t descriptor_bytes;
	uint32_t events;
	int32_t fd;
	uint32_t reserved0;
} SparkServiceBackendPollDescriptor;

typedef SparkStatus (*SparkServiceBackendInitializeFunction)(
	const SparkServiceBackendConfiguration *configuration,
	void **backend_state);
typedef void (*SparkServiceBackendDestroyFunction)(
	void *backend_state);
typedef SparkStatus (*SparkServiceBackendGetViewFunction)(
	void *backend_state,
	SparkServiceBackendView *view);
typedef SparkStatus (*SparkServiceBackendPumpFunction)(
	void *backend_state,
	uint32_t max_dispatch_steps,
	SparkServiceStats *stats_out);
typedef SparkStatus (*SparkServiceBackendGetPollDescriptorsFunction)(
	void *backend_state,
	SparkServiceBackendPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out);

typedef struct SparkServiceBackendInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t reserved0;
	SparkServiceBackendInitializeFunction initialize;
	SparkServiceBackendDestroyFunction destroy;
	SparkServiceBackendGetViewFunction get_view;
	SparkServiceBackendPumpFunction pump;
	SparkServiceBackendGetPollDescriptorsFunction get_poll_descriptors;
} SparkServiceBackendInterface;

typedef const SparkServiceBackendInterface *(*SparkServiceBackendGetInterfaceFunction)(
	void);

typedef struct SparkServiceBackendDynamicLibrary
{
	void *dynamic_library;
	SparkServiceBackendInterface backend_interface;
} SparkServiceBackendDynamicLibrary;

SparkStatus SparkServiceBackendValidateInterface(
	const SparkServiceBackendInterface *backend_interface,
	uint32_t required_capability_flags);
SparkStatus SparkServiceBackendLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkServiceBackendDynamicLibrary *library);
void SparkServiceBackendUnloadInterface(
	SparkServiceBackendDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
