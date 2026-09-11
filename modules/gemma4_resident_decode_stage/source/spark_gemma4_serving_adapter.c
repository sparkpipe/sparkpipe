
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_gemma4_model.h"
#include "sparkpipe/spark_gemma4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_gemma4_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_serving_cache_admission.h"

#ifndef GEMMA4_MODEL_REVISION
#error "GEMMA4_MODEL_REVISION must name the exact source snapshot revision"
#endif
#ifndef GEMMA4_CONTRACT_SHA256
#error "GEMMA4_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_GEMMA4_SERVING_DRIVER_MODEL_ID SPARK_GEMMA4_MODEL_DRIVER_MODEL_ID
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
#define SPARK_GEMMA4_SERVING_ADAPTER_ID "spark.gemma4.serving-adapter.tp4pp4.v1"
#define SPARK_GEMMA4_SERVING_MODEL_ID "google/gemma-4-26B-A4B-it"
#define SPARK_GEMMA4_SERVING_TARGET "cuda.sm121.gemma4.26b-a4b.resident_decode_stage.bf16"
#define SPARK_GEMMA4_SERVING_STAGE_COUNT 16u
#define SPARK_GEMMA4_SERVING_DEFAULT_TP_DEGREE 4u
#define SPARK_GEMMA4_SERVING_PARALLEL_GROUP_SIZE 4u
#define SPARK_GEMMA4_SERVING_PP_STAGE_COUNT 4u
#define SPARK_GEMMA4_SERVING_STAGE_LAYER_LIST \
	8u,8u,8u,8u,8u,8u,8u,8u,7u,7u,7u,7u,7u,7u,7u,7u
#else
#define SPARK_GEMMA4_SERVING_ADAPTER_ID "spark.gemma4.serving-adapter.tp16.v1"
#define SPARK_GEMMA4_SERVING_MODEL_ID "google/gemma-4-31B-it"
#define SPARK_GEMMA4_SERVING_TARGET "cuda.sm121.gemma4.31b.resident_decode_stage.bf16"
#define SPARK_GEMMA4_SERVING_STAGE_COUNT 16u
#define SPARK_GEMMA4_SERVING_DEFAULT_TP_DEGREE 16u
#define SPARK_GEMMA4_SERVING_PARALLEL_GROUP_SIZE 16u
#define SPARK_GEMMA4_SERVING_PP_STAGE_COUNT 1u
#define SPARK_GEMMA4_SERVING_STAGE_LAYER_LIST \
	60u,60u,60u,60u,60u,60u,60u,60u,60u,60u,60u,60u,60u,60u,60u,60u
#endif
#define SPARK_GEMMA4_SERVING_MAX_PP_STAGE_COUNT SPARK_GEMMA4_SERVING_PP_STAGE_COUNT
#define SPARK_GEMMA4_SERVING_STAGE_NAME "gemma4_resident_decode_stage"
#define SPARK_GEMMA4_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_GEMMA4_SERVING_MAX_SEQUENCE_POSITIONS_CAP \
	SPARK_GEMMA4_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_GEMMA4_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

#define SPARK_QWEN38_SERVING_ADAPTER_FN(name) SparkGemma4##name
#define SPARK_QWEN38_SERVING_ADAPTER_TYPE(name) SparkGemma4##name
#define SPARK_QWEN38_SERVING_ADAPTER_CONST(name) SPARK_GEMMA4_##name
#define SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION GEMMA4_MODEL_REVISION
#define SPARK_QWEN38_SERVING_ADAPTER_CONTRACT_SHA256 GEMMA4_CONTRACT_SHA256
#define SPARK_QWEN38_SERVING_ADAPTER_TP_DEGREE_VALID(tp_degree) \
	((tp_degree) == SPARK_GEMMA4_SERVING_PARALLEL_GROUP_SIZE)
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_COUNT(state) \
	(state)->pp_stage_count
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_INDEX(state) \
	SparkGemma4ServingPpStageIndex(state,(state)->stage_index)
#define SPARK_QWEN38_SERVING_ADAPTER_BIND_FAMILY(state) \
	SparkGemma4ServingInitializeFamilyState(state)
#define SPARK_QWEN38_SERVING_ADAPTER_PREFETCH SparkGemma4ServingPrefetch
#define SPARK_QWEN38_SERVING_ADAPTER_RESOLVE_PREFETCH \
	SparkGemma4ServingResolvePrefetch
#define SPARK_QWEN38_SERVING_ADAPTER_RESET SparkGemma4ServingReset
#define SPARK_QWEN38_SERVING_ADAPTER_SUBMISSION_STALE(state,submission) \
	SparkGemma4ServingSubmissionStale((state),(submission))

typedef struct SparkGemma4ServingPending
{
	struct SparkGemma4ServingState *owner;
	SparkServingAdapterPendingCommon common;
	uint64_t frame_sequence_id;
	uint64_t frame_sequence_position;
	SparkStatus frame_status;
	SparkModelDriverResidencyToken residency;
	uint64_t accepted_token_count;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint32_t last_row_by_lane[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_slots[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_flats[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t output_token_ids[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_output_ids[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_token_ids[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkGemma4ServingPending;

typedef struct SparkGemma4ServingTransportShim
{
	const void *input_base;
	const uint32_t *input_row_map;
	uint32_t input_rows;
	void *input_scratch;
	void *output_base;
	const uint32_t *output_row_map;
	void *execution_stream;
} SparkGemma4ServingTransportShim;

typedef struct SparkGemma4ServingState
{
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkModelServingWakeFunction wake_function;
	void *wake_context;
	void *execution_stream;
	char stage_pack_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t stage_index;
	uint32_t tp_degree;
	uint32_t pp_stage_count;
	uint32_t stage_layer_counts[SPARK_GEMMA4_SERVING_PP_STAGE_COUNT];
	uint32_t first_layer_index;
	uint32_t stage_layer_count;
	uint32_t stage_attn_layer_count;
	uint32_t pipeline_slot_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t max_sequence_positions;
	uint32_t blocks_per_lane;
	uint32_t kv_block_count;
	uint32_t quiescing;
	uint64_t orphan_completion_count;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkGemma4KvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	uint32_t *free_blocks;
	uint32_t free_block_count;
	uint32_t lane_block_counts[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	void *gather_scratch;
	SparkGemma4ServingTransportShim shim;
	SparkGemma4ServingPending pending[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint reset_active;
	atomic_uint_fast64_t reset_generation;
} SparkGemma4ServingState;

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateSubmission)(
	void *adapter_state,const SparkModelServingSubmission *submission);
static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingQuiesce)(
	void *adapter_state,uint64_t deadline_time_ns);

static _Thread_local SparkModelDriverCacheLane SparkGemma4ServingCacheScratch[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];

static uint32_t SparkGemma4ServingSubmissionStale(
	const SparkGemma4ServingState *state,
	const SparkModelServingSubmission *submission)
{
	if ( submission == 0 )
		return(0u);
	return(submission->control_generation <
		atomic_load_explicit(&state->reset_generation,memory_order_acquire) ?
		1u : 0u);
}

static SparkStatus SparkGemma4ServingInitializeFamilyState(
	SparkGemma4ServingState *state)
{
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	atomic_init(&state->reset_active,0u);
	atomic_init(&state->reset_generation,0u);
	return(SPARK_STATUS_OK);
}

static SparkServingCacheAdmission SparkGemma4ServingCacheContext(
	SparkGemma4ServingState *state,SparkModelDriverCacheLane *lanes)
{
	SparkServingCacheAdmission cache;
	cache.program_id = state->program->program_id;
	cache.lane_capacity = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	cache.lanes = lanes;
	cache.driver = state->driver.interface;
	cache.driver_instance = state->driver_instance;
	cache.validate = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateSubmission);
	cache.adapter_state = state;
	return(cache);
}

static SparkStatus SparkGemma4ServingPrefetch(void *adapter_state,
	const SparkModelServingSubmission *submissions,uint32_t count)
{
	SparkGemma4ServingState *state;
	SparkServingCacheAdmission cache;
	state = (SparkGemma4ServingState *)adapter_state;
	if ( state == 0 || state->program == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	cache = SparkGemma4ServingCacheContext(state,SparkGemma4ServingCacheScratch);
	return(SparkServingCacheAdmissionRun(&cache,submissions,count,
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE));
}

static SparkStatus SparkGemma4ServingResolvePrefetch(void *adapter_state,
	const SparkModelServingSubmission *submission,uint32_t resolution)
{
	SparkGemma4ServingState *state;
	SparkServingCacheAdmission cache;
	uint32_t flags;
	state = (SparkGemma4ServingState *)adapter_state;
	if ( state == 0 || state->program == 0 ||
		(resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		 resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	flags = resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ?
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT :
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	cache = SparkGemma4ServingCacheContext(state,SparkGemma4ServingCacheScratch);
	return(SparkServingCacheAdmissionRun(&cache,submission,1u,flags));
}

static SparkStatus SparkGemma4ServingResetControl(void *adapter_state,
	uint64_t control_generation)
{
	SparkGemma4ServingState *state = (SparkGemma4ServingState *)adapter_state;
	SparkModelDriverAdmissionRequest request = {0};
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	if ( state == 0 || control_generation == 0u ||
		control_generation <= atomic_load_explicit(&state->reset_generation,memory_order_acquire) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingQuiesce)(state,UINT64_MAX);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request.descriptor_bytes = (uint32_t)sizeof(request);
	request.program_id = state->program->program_id;
	request.control_generation = control_generation;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	SparkModelDriverInitializeAdmissionDecision(&decision);
	status = state->driver.interface->admit(state->driver_instance,&request,&decision);
	if ( status == SPARK_STATUS_OK && decision.accepted == 0u )
		status = SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
	{
		atomic_store_explicit(&state->reset_generation,control_generation,memory_order_release);
		state->quiescing = 0u;
	}
	return(status);
}

static SparkStatus SparkGemma4ServingReset(void *adapter_state,
	uint64_t control_generation)
{
	SparkGemma4ServingState *state = (SparkGemma4ServingState *)adapter_state;
	uint32_t expected = 0u;
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( atomic_compare_exchange_strong_explicit(&state->reset_active,&expected,1u,
		memory_order_acquire,memory_order_relaxed) == 0 )
		return(SPARK_STATUS_BUSY);
	status = SparkGemma4ServingResetControl(state,control_generation);
	atomic_store_explicit(&state->reset_active,0u,memory_order_release);
	return(status);
}

static const SparkModelServingAdapterDescriptor SparkGemma4ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_GEMMA4_SERVING_ADAPTER_ID,
		SPARK_GEMMA4_SERVING_MODEL_ID,
		GEMMA4_MODEL_REVISION,
		SPARK_GEMMA4_SERVING_PROGRAM_NAME,
		GEMMA4_CONTRACT_SHA256),
	.capability_flags = SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP),
	.parallel_group_size = SPARK_GEMMA4_SERVING_PARALLEL_GROUP_SIZE,
	.stage_count = SPARK_GEMMA4_SERVING_STAGE_COUNT,
	.layer_count = SPARK_GEMMA4_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 1u,
	.max_active_sequence_count = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequence_count = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.stage_layer_counts = {SPARK_GEMMA4_SERVING_STAGE_LAYER_LIST},
	.minimum_efficient_submission_row_count = 0u,
	.cache_block_token_count = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS
};

#include "sparkpipe/spark_qwen38_pp_serving_adapter_common.h"
