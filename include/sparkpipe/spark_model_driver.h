#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_DRIVER_ABI_VERSION 10u
#define SPARK_MODEL_DRIVER_INTERFACE_SYMBOL "SparkModelDriverGetInterface"
#define SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY 8u
#define SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY 8u
#define SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS 0x00000001u
#define SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS 0x00000002u

#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION 0x00000001u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED 0x00000002u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE 0x00000004u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE 0x00000008u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_JIT_KV_CACHE 0x00000010u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_ZERO_COPY_NODE_CONTEXT 0x00000020u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_PRIVATE_QUEUE_PRESSURE 0x00000040u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_HOST_STAGING 0x00000080u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE 0x00000100u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_VALIDATED_LATENCY 0x00000200u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_CAPTURED_CUDA_GRAPH 0x00000400u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_DEVICE_MEMCPY 0x00000800u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_PRIVATE_EXPERT_QUEUES 0x00001000u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_EVENT_DEPENDENCIES 0x00002000u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_RESIDENCY_AFFINITY_REQUIRED 0x00004000u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_BATCH_SHAPE_FIXED 0x00008000u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT 0x00010000u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT 0x00020000u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT 0x00040000u
#define SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL 0x00080000u

#define SPARK_MODEL_DRIVER_BUFFER_FLAG_READ 0x00000001u
#define SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE 0x00000002u
#define SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID 0x00000001u
#define SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL 0x00000002u
#define SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE 0x00000004u
#define SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE 0x00000001u
#define SPARK_MODEL_DRIVER_ADMISSION_KNOWN_FLAGS \
    SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE
#define SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX 0x00000001u
#define SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH 0x00000002u
#define SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE 0x00000004u
#define SPARK_MODEL_DRIVER_CACHE_LANE_KNOWN_FLAGS \
    (SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX | \
     SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH | \
     SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE)
#define SPARK_MODEL_DRIVER_SCALAR_COUNT 8u
#define SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT UINT32_MAX

#if defined(_WIN32)
#define SPARK_MODEL_DRIVER_EXPORT __declspec(dllexport)
#else
#define SPARK_MODEL_DRIVER_EXPORT __attribute__((visibility("default")))
#endif

typedef struct SparkModelDriverBuffer
{
    uint32_t slot;
    uint32_t flags;
    void *address;
    uint64_t bytes;
} SparkModelDriverBuffer;

typedef struct SparkModelDriverResidencyToken
{
    uint64_t word0;
    uint64_t word1;
    uint64_t generation;
    uint64_t owner;
} SparkModelDriverResidencyToken;

typedef struct SparkModelDriverCacheIdentity
{
    uint8_t sha256[32];
} SparkModelDriverCacheIdentity;

/*
 * Model-neutral cache intent. Admission prepares or verifies the requested
 * mappings; model drivers alone translate them into physical page layouts.
 * A prepare admission is idempotent and must not publish or release state.
 */
typedef struct SparkModelDriverCacheLane
{
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint32_t resident_sequence_slot;
    uint32_t context_token_count;
    uint32_t prefix_token_count;
    uint32_t publish_token_count;
    uint32_t flags;
    uint32_t reserved;
    SparkModelDriverCacheIdentity prefix_identity;
    SparkModelDriverCacheIdentity publish_identity;
} SparkModelDriverCacheLane;

typedef struct SparkModelDriverCompletion
{
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint32_t program_id;
    uint32_t driver_dispatch_slot;
    uint32_t accepted_token_count;
    uint32_t completion_flags;
    uint32_t token_count;
    uint32_t token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
    uint32_t draft_token_count;
    uint32_t draft_token_ids[SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY];
    SparkStatus status;
    SparkModelDriverResidencyToken residency;
    uint64_t queue_delay_ns;
    uint64_t service_time_ns;
    uint64_t device_memcpy_bytes;
    uint64_t host_staging_bytes;
} SparkModelDriverCompletion;

typedef void (*SparkModelDriverCompletionFunction)(void *completion_context, const SparkModelDriverCompletion *completion);
typedef void (*SparkModelDriverWakeFunction)(void *wake_context);

typedef struct SparkModelDriverFrame
{
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t deadline_time_ns;
    uint32_t active_slot_count;
    uint32_t new_token_count;
    uint32_t priority;
    uint32_t flags;
    uint32_t driver_dispatch_slot;
    uint32_t program_id;
    uint64_t driver_dispatch_generation;
    uint64_t driver_dispatch_cookie0;
    uint64_t driver_dispatch_cookie1;
    void *execution_stream;
    SparkModelDriverBuffer *buffers;
    uint32_t buffer_count;
    uint32_t reserved;
    uint32_t cache_lane_count;
    uint32_t reserved1;
    const SparkModelDriverCacheLane *cache_lanes;
    uint64_t scalar[SPARK_MODEL_DRIVER_SCALAR_COUNT];
    SparkModelDriverResidencyToken residency;
    void *user_context;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
} SparkModelDriverFrame;

typedef struct SparkModelDriverProgramProfile
{
    uint32_t descriptor_bytes;
    uint32_t profile_flags;
    uint32_t max_inflight;
    uint32_t max_active_slots;
    uint32_t max_new_tokens;
    uint32_t max_resident_sequences;
    uint64_t max_sequence_tokens;
    uint64_t target_latency_ns;
    uint64_t validated_latency_ns;
    uint64_t resident_weight_bytes;
    uint64_t resident_kv_bytes;
    uint64_t static_workspace_bytes;
    uint64_t device_memcpy_bytes_per_submit_ceiling;
    uint64_t host_staging_bytes_per_submit_ceiling;
    uint32_t private_queue_count;
    uint32_t reserved;
} SparkModelDriverProgramProfile;

typedef enum SparkModelDriverAdmissionRejection
{
    SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED = 0,
    SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY = 1,
    SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY = 2,
    SPARK_MODEL_DRIVER_ADMISSION_REJECTED_DEADLINE = 3,
    SPARK_MODEL_DRIVER_ADMISSION_REJECTED_HOST_STAGING_REQUIRED = 4,
    SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE = 5
} SparkModelDriverAdmissionRejection;

typedef struct SparkModelDriverAdmissionRequest
{
    uint32_t descriptor_bytes;
    uint32_t program_id;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t deadline_time_ns;
    uint32_t active_slot_count;
    uint32_t new_token_count;
    uint32_t priority;
    uint32_t frame_flags;
    uint32_t admission_flags;
    uint32_t cache_lane_count;
    const SparkModelDriverCacheLane *cache_lanes;
    SparkModelDriverResidencyToken residency;
} SparkModelDriverAdmissionRequest;

typedef struct SparkModelDriverAdmissionDecision
{
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t rejection_reason;
    uint32_t driver_dispatch_slot;
    uint64_t estimated_queue_delay_ns;
    uint64_t estimated_service_time_ns;
    uint64_t endpoint_cost;
    uint64_t residency_match_score;
    uint64_t device_memcpy_bytes;
    uint64_t host_staging_bytes;
    uint32_t private_queue_pressure;
    uint32_t available_dispatch_slot_count;
    uint64_t driver_dispatch_generation;
    uint64_t driver_dispatch_cookie0;
    uint64_t driver_dispatch_cookie1;
} SparkModelDriverAdmissionDecision;

typedef struct SparkModelDriverRuntimeSnapshot
{
    uint32_t descriptor_bytes;
    uint32_t program_id;
    uint32_t active_submission_count;
    uint32_t available_dispatch_slot_count;
    uint64_t submitted_count;
    uint64_t completed_count;
    uint64_t rejected_count;
    uint64_t resident_sequence_count;
    uint64_t resident_token_count;
    uint64_t kv_token_capacity;
    uint64_t device_memcpy_bytes_per_submit;
    uint64_t host_staging_bytes_per_submit;
    uint64_t cuda_graph_capture_count;
    uint64_t cuda_graph_replay_count;
    uint64_t host_callback_completion_count;
    uint64_t stale_admission_count;
    uint32_t private_queue_pressure;
    uint32_t reserved;
} SparkModelDriverRuntimeSnapshot;

typedef struct SparkModelDriverCreateRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t reserved0;
    const char *node_id;
    const char *node_target;
    void *node_context;
    uint32_t kv_logical_page_capacity;
    uint32_t kv_physical_page_capacity;
    const char *kv_backing_directory;
    uint64_t kv_backing_maximum_bytes;
    void *execution_stream;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
    SparkModelDriverWakeFunction wake_function;
    void *wake_context;
    uint64_t reserved[1];
} SparkModelDriverCreateRequest;

typedef SparkStatus (*SparkModelDriverProgramSubmitFunction)(void *driver_instance, SparkModelDriverFrame *frame);

typedef struct SparkModelDriverProgramDescriptor
{
    uint32_t program_id;
    uint32_t flags;
    uint32_t max_inflight;
    uint32_t reserved;
    const char *name;
    const SparkModelDriverProgramProfile *profile;
    SparkModelDriverProgramSubmitFunction submit;
} SparkModelDriverProgramDescriptor;

typedef struct SparkModelDriverDescriptor
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *model_id;
    const char *model_revision;
    const char *stage_name;
    const char *target;
    const char *model_description_sha256;
    const char *compiled_program_sha256;
    uint32_t program_count;
    uint32_t module_instance_count;
    const SparkModelDriverProgramDescriptor *programs;
} SparkModelDriverDescriptor;

typedef struct SparkModelDriverInterface
{
    uint32_t abi_version;
    uint32_t interface_bytes;
    const SparkModelDriverDescriptor *descriptor;
    SparkStatus (*create)(const SparkModelDriverCreateRequest *request, void **driver_instance);
    void (*destroy)(void *driver_instance);
    SparkStatus (*admit)(void *driver_instance, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
    SparkStatus (*snapshot)(void *driver_instance, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
} SparkModelDriverInterface;

typedef const SparkModelDriverInterface *(*SparkModelDriverGetInterfaceFunction)(void);

#define SPARK_MODEL_DRIVER_CREATE_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkModelDriverCreateRequest))

#ifdef __cplusplus
}
#endif
