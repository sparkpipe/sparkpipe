#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_service.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION 10u
#define SPARK_GLM52_SERVICE_BACKEND_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkGlm52ServiceBackendInterface))
#define SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkGlm52ServiceBackendConfiguration))
#define SPARK_GLM52_SERVICE_BACKEND_VIEW_BYTES \
	((uint32_t)sizeof(SparkGlm52ServiceBackendView))
#define SPARK_GLM52_SERVICE_BACKEND_DYNAMIC_LIBRARY_BYTES \
	((uint32_t)sizeof(SparkGlm52ServiceBackendDynamicLibrary))
#define SPARK_GLM52_SERVICE_BACKEND_INTERFACE_SYMBOL \
	"SparkGlm52ServiceBackendGetInterface"
#define SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES 256u

#define SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK 0x00000001u
#define SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP 0x00000002u
#define SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_KNOWN_FLAGS \
	(SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK | \
	 SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP)

#define SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_SERVICE_RUNTIME 0x00000001u
#define SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_RING_RUNTIME 0x00000002u
#define SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_TOKENIZER 0x00000004u
#define SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS 0x00000008u
#define SPARK_GLM52_SERVICE_BACKEND_POLL_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkGlm52ServiceBackendPollDescriptor))
#define SPARK_GLM52_SERVICE_BACKEND_POLL_READ 0x00000001u
#define SPARK_GLM52_SERVICE_BACKEND_POLL_WRITE 0x00000002u
#define SPARK_GLM52_SERVICE_BACKEND_REQUIRED_PRODUCTION_CAPS \
	(SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_SERVICE_RUNTIME | \
	 SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_RING_RUNTIME | \
	 SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS)

typedef struct SparkGlm52ServiceBackendConfiguration
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
} SparkGlm52ServiceBackendConfiguration;

typedef struct SparkGlm52ServiceBackendView
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
	SparkGlm52ServiceRuntime *service;
	const SparkTokenizer *tokenizer;
	const char *first_blocker;
	const char *release_id;
	const char *release_git_commit;
	const char *transport_shared_object_path;
} SparkGlm52ServiceBackendView;

typedef struct SparkGlm52ServiceBackendPollDescriptor
{
	uint32_t descriptor_bytes;
	uint32_t events;
	int32_t fd;
	uint32_t reserved0;
} SparkGlm52ServiceBackendPollDescriptor;

typedef SparkStatus (*SparkGlm52ServiceBackendInitializeFunction)(
	const SparkGlm52ServiceBackendConfiguration *configuration,
	void **backend_state);
typedef void (*SparkGlm52ServiceBackendDestroyFunction)(
	void *backend_state);
typedef SparkStatus (*SparkGlm52ServiceBackendGetViewFunction)(
	void *backend_state,
	SparkGlm52ServiceBackendView *view);
typedef SparkStatus (*SparkGlm52ServiceBackendPumpFunction)(
	void *backend_state,
	uint32_t max_dispatch_steps,
	SparkGlm52ServiceStats *stats_out);
typedef SparkStatus (*SparkGlm52ServiceBackendGetPollDescriptorsFunction)(
	void *backend_state,
	SparkGlm52ServiceBackendPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out);

typedef struct SparkGlm52ServiceBackendInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t reserved0;
	SparkGlm52ServiceBackendInitializeFunction initialize;
	SparkGlm52ServiceBackendDestroyFunction destroy;
	SparkGlm52ServiceBackendGetViewFunction get_view;
	SparkGlm52ServiceBackendPumpFunction pump;
	SparkGlm52ServiceBackendGetPollDescriptorsFunction get_poll_descriptors;
} SparkGlm52ServiceBackendInterface;

typedef const SparkGlm52ServiceBackendInterface *(*SparkGlm52ServiceBackendGetInterfaceFunction)(
	void);

typedef struct SparkGlm52ServiceBackendDynamicLibrary
{
	void *dynamic_library;
	SparkGlm52ServiceBackendInterface backend_interface;
} SparkGlm52ServiceBackendDynamicLibrary;

SparkStatus SparkGlm52ServiceBackendValidateInterface(
	const SparkGlm52ServiceBackendInterface *backend_interface,
	uint32_t required_capability_flags);
SparkStatus SparkGlm52ServiceBackendLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkGlm52ServiceBackendDynamicLibrary *library);
void SparkGlm52ServiceBackendUnloadInterface(
	SparkGlm52ServiceBackendDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
