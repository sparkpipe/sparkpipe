#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_ring_runtime.h"
#include "sparkpipe/spark_glm52_ring_work_control.h"
#include "sparkpipe/spark_glm52_serving_engine.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_ABI_VERSION 18u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_RESIDENT_SEQUENCE_COUNT \
	16384u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS \
	SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BLOCK_CAPACITY \
	1048576u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BATCH_BLOCK_COUNT \
	32u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MAX_NVME_BATCH_BLOCK_COUNT \
	128u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_INTERFACE_SYMBOL \
	"SparkGlm52RingNodeContextBuilderGetInterface"
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkGlm52RingNodeContextBuilderConfiguration))
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_RESULT_BYTES \
	((uint32_t)sizeof(SparkGlm52RingNodeContextBuilderResult))
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkGlm52RingNodeContextBuilderInterface))
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_KV_STATS_BYTES \
	((uint32_t)sizeof(SparkGlm52RingNodeContextBuilderKvStats))
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MOE_BACKEND_NONE \
	SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_NONE
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MOE_BACKEND_FP8_FLASHINFER_GROUPED \
	SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MOE_BACKEND_W8LUT_BF16_WMMA \
	SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_W8LUT_BF16_WMMA
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_MOE_BACKEND_NVFP4_B12X \
	SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_NVFP4_B12X
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_DISABLED 0u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_SYNCHRONOUS_FULL_HISTORY 1u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT 2u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT 3u

#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_RESIDENT_MOE_PACKS 0x00000001u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_STAGE_SLICE 0x00000002u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_RANK0_TOKEN_INPUT 0x00000004u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_RANK_WORK_DISPATCH 0x00000008u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_DSPARK_DRAFT 0x00000010u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_MTP_DRAFT 0x00000020u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_NVME_KV 0x00000040u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_ASYNC_WORK 0x00000080u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_LAYER_MAJOR_MTP_VERIFY 0x00000100u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_CONTROL_GENERATION_RESET 0x00000200u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_KV_LOOKAHEAD_PREFETCH 0x00000400u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_MOONCAKE_KV 0x00000800u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS \
	(SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_RESIDENT_MOE_PACKS | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_STAGE_SLICE | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_RANK0_TOKEN_INPUT | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_RANK_WORK_DISPATCH | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_DSPARK_DRAFT | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_MTP_DRAFT | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_NVME_KV | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_ASYNC_WORK | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_LAYER_MAJOR_MTP_VERIFY | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_CONTROL_GENERATION_RESET | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_KV_LOOKAHEAD_PREFETCH | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CAP_MOONCAKE_KV)

#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK \
	0x00000001u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP \
	0x00000002u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV \
	0x00000004u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MOONCAKE_KV \
	0x00000008u
#define SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_KNOWN_FLAGS \
	(SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV | \
	 SPARK_GLM52_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MOONCAKE_KV)

typedef struct SparkGlm52RingNodeContextBuilderConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t rank_index;
	uint32_t max_active_sequence_count;
	uint32_t port_base;
	uint32_t kv_pool_token_capacity;
	uint32_t maximum_resident_sequence_count;
	const char *moe_pack_root;
	const char *stagepack_root;
	const char *embedding_pack_path;
	const char *node_target;
	const char *dspark_manifest_path;
	const char *dspark_config_path;
	const char *dspark_safetensors_path;
	const char *kv_nvme_path;
	const char *kv_store_module_path;
	const char *kv_store_service_address;
	const char *kv_store_ipc_socket_path;
	uint32_t dspark_maximum_lane_count;
	uint32_t dspark_maximum_context_token_count;
	uint32_t kv_nvme_block_capacity;
	uint32_t kv_nvme_batch_block_count;
	uint32_t kv_store_block_capacity;
	uint32_t kv_store_batch_block_count;
	uint32_t kv_store_worker_count;
	uint32_t kv_store_lookahead_packet_count;
	uint64_t kv_store_model_fingerprint;
	uint64_t kv_store_layout_fingerprint;
	uint64_t kv_store_client_memory_pool_bytes;
	uint64_t kv_store_local_buffer_bytes;
	const SparkGlm52RingRuntimeRankPlan *rank_plan;
} SparkGlm52RingNodeContextBuilderConfiguration;

typedef struct SparkGlm52RingNodeContextBuilderResult
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t rank_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t hidden_dimension;
	uint32_t vocabulary_size;
	void *node_context;
	const void *embedding_weight_bf16;
	void *private_state;
} SparkGlm52RingNodeContextBuilderResult;

typedef struct SparkGlm52RingNodeContextBuilderKvStats
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t nvme_enabled;
	uint32_t nvme_mode;
	uint32_t physical_block_capacity;
	uint32_t logical_block_capacity;
	uint32_t logical_block_count;
	uint32_t resident_block_count;
	uint32_t swapped_block_count;
	uint64_t nvme_record_bytes;
	uint64_t nvme_store_count;
	uint64_t nvme_load_count;
	uint64_t nvme_write_bytes;
	uint64_t nvme_read_bytes;
	uint64_t nvme_synchronous_wait_count;
	uint64_t nvme_batch_flush_count;
	uint64_t nvme_maximum_batch_operation_count;
	uint64_t resident_bytes_per_token;
	uint64_t resident_pool_bytes;
	uint64_t nvme_capacity_bytes;
	uint64_t compact_selected_mla_working_set_bytes;
	uint32_t nvme_batch_block_capacity;
	uint32_t nvme_pending_store_count;
	uint32_t nvme_pending_load_count;
	uint32_t nvme_clean_evict_count;
	uint32_t pending_work_active;
	uint32_t logical_lane_capacity;
	uint32_t execution_row_capacity;
	uint32_t last_layer_major_logical_lane_count;
	uint32_t last_layer_major_rows_per_lane;
	uint32_t last_layer_major_execution_row_count;
	uint32_t moe_backend_kind;
	uint32_t moe_bound_layer_count;
	uint32_t moe_expected_layer_count;
	uint32_t fp8_scaled_gemm_bound_plan_count;
	uint32_t fp8_scaled_gemm_expected_plan_count;
	uint32_t model_quantization_mode;
	uint64_t asynchronous_submit_count;
	uint64_t asynchronous_completion_count;
	uint64_t asynchronous_failure_count;
	uint64_t layer_major_submit_count;
	uint64_t layer_major_completion_count;
	uint64_t layer_major_failure_count;
	uint64_t cuda_total_bytes;
	uint64_t cuda_initial_free_bytes;
	uint64_t cuda_current_free_bytes;
	uint64_t cuda_consumed_bytes;
	uint64_t cuda_builder_allocation_bytes;
	uint64_t cuda_largest_allocation_bytes;
	uint64_t host_mapped_allocation_bytes;
} SparkGlm52RingNodeContextBuilderKvStats;

typedef SparkStatus (*SparkGlm52RingNodeContextBuilderInitializeFunction)(
	const SparkGlm52RingNodeContextBuilderConfiguration *configuration,
	void **builder_state);
typedef void (*SparkGlm52RingNodeContextBuilderDestroyFunction)(
	void *builder_state);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderBuildFunction)(
	void *builder_state,
	SparkGlm52RingNodeContextBuilderResult *result);
typedef void (*SparkGlm52RingNodeContextBuilderDestroyResultFunction)(
	void *builder_state,
	SparkGlm52RingNodeContextBuilderResult *result);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderAttachDriverFunction)(
	void *builder_state,
	const SparkModelDriverInterface *driver_interface,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	SparkHiddenTransportSession *output_transport_session);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderIdlePumpFunction)(
	void *idle_pump_context);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderPrefillFunction)(
	void *builder_state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	SparkGlm52RingNodeContextBuilderIdlePumpFunction idle_pump_function,
	void *idle_pump_context);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderDecodeFunction)(
	void *builder_state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderSubmitWorkFunction)(
	void *builder_state,
	const SparkGlm52RingWorkControlPacket *work_packet,
	SparkHiddenTransportSession *input_transport_session,
	SparkHiddenTransportSession *output_transport_session,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderPrefetchWorkFunction)(
	void *builder_state,
	const SparkGlm52RingWorkControlPacket *work_packets,
	uint32_t work_packet_count);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderProgressFunction)(
	void *builder_state);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderTakeDsparkDraftFunction)(
	void *builder_state,
	SparkGlm52DsparkDraftResult *draft_result);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderGetKvStatsFunction)(
	void *builder_state,
	SparkGlm52RingNodeContextBuilderKvStats *stats);
typedef SparkStatus (*SparkGlm52RingNodeContextBuilderResetControlGenerationFunction)(
	void *builder_state,
	uint64_t control_generation);

typedef struct SparkGlm52RingNodeContextBuilderInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t reserved0;
	SparkGlm52RingNodeContextBuilderInitializeFunction initialize;
	SparkGlm52RingNodeContextBuilderDestroyFunction destroy;
	SparkGlm52RingNodeContextBuilderBuildFunction build;
	SparkGlm52RingNodeContextBuilderDestroyResultFunction destroy_result;
	SparkGlm52RingNodeContextBuilderAttachDriverFunction attach_driver;
	SparkGlm52RingNodeContextBuilderPrefillFunction prefill;
	SparkGlm52RingNodeContextBuilderDecodeFunction decode;
	SparkGlm52RingNodeContextBuilderSubmitWorkFunction submit_work;
	SparkGlm52RingNodeContextBuilderPrefetchWorkFunction prefetch_work;
	SparkGlm52RingNodeContextBuilderProgressFunction progress;
	SparkGlm52RingNodeContextBuilderTakeDsparkDraftFunction take_dspark_draft;
	SparkGlm52RingNodeContextBuilderGetKvStatsFunction get_kv_stats;
	SparkGlm52RingNodeContextBuilderResetControlGenerationFunction
		reset_control_generation;
} SparkGlm52RingNodeContextBuilderInterface;

typedef const SparkGlm52RingNodeContextBuilderInterface *(
	*SparkGlm52RingNodeContextBuilderGetInterfaceFunction)(void);

typedef struct SparkGlm52RingNodeContextBuilderDynamicLibrary
{
	void *dynamic_library;
	SparkGlm52RingNodeContextBuilderInterface builder_interface;
} SparkGlm52RingNodeContextBuilderDynamicLibrary;

SparkStatus SparkGlm52RingNodeContextBuilderValidateInterface(
	const SparkGlm52RingNodeContextBuilderInterface *builder_interface,
	uint32_t required_capability_flags);
SparkStatus SparkGlm52RingNodeContextBuilderValidateResult(
	const SparkGlm52RingNodeContextBuilderResult *result,
	const SparkGlm52RingRuntimeRankPlan *rank_plan);
SparkStatus SparkGlm52RingNodeContextBuilderLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkGlm52RingNodeContextBuilderDynamicLibrary *library);
void SparkGlm52RingNodeContextBuilderUnloadInterface(
	SparkGlm52RingNodeContextBuilderDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
