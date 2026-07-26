#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_serving_engine.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SERVICE_ABI_VERSION 1u
#define SPARK_GLM52_SERVICE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceConfiguration))
#define SPARK_GLM52_SERVICE_RUNTIME_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceRuntime))
#define SPARK_GLM52_SERVICE_CLIENT_SESSION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceClientSession))
#define SPARK_GLM52_SERVICE_REQUEST_MAP_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceRequestMap))
#define SPARK_GLM52_SERVICE_SUBMIT_TEXT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceSubmitTextRequest))
#define SPARK_GLM52_SERVICE_SUBMIT_TOKENS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceSubmitTokenIdsRequest))
#define SPARK_GLM52_SERVICE_SUBMIT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceSubmitResult))
#define SPARK_GLM52_SERVICE_EVENT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceEvent))
#define SPARK_GLM52_SERVICE_STATS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceStats))
#define SPARK_GLM52_SERVICE_FRAME_HEADER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceFrameHeader))
#define SPARK_GLM52_SERVICE_FRAME_SUBMIT_TEXT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceSubmitTextFrameBody))
#define SPARK_GLM52_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceSubmitTokenIdsFrameBody))
#define SPARK_GLM52_SERVICE_FRAME_CANCEL_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServiceCancelFrameBody))

#define SPARK_GLM52_SERVICE_DEFAULT_PUMP_DISPATCH_STEPS 256u
#define SPARK_GLM52_SERVICE_DEFAULT_REQUEST_ID_BASE 5000000000ull
#define SPARK_GLM52_SERVICE_FRAME_MAGIC 0x35504B53u
#define SPARK_GLM52_SERVICE_MAX_FRAME_BODY_BYTES (128u * 1024u * 1024u)
#define SPARK_GLM52_SERVICE_MAX_TEXT_BYTES (64u * 1024u * 1024u)
#define SPARK_GLM52_SERVICE_MAX_TOKEN_FRAME_COUNT \
    (SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS)
#define SPARK_GLM52_SERVICE_CLIENT_HASH_SLOTS 1024u
#define SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS 4096u
#define SPARK_GLM52_SERVICE_NO_HASH_SLOT UINT32_MAX

#define SPARK_GLM52_SERVICE_CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS \
    0x00000001u
#define SPARK_GLM52_SERVICE_CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP \
    0x00000002u
#define SPARK_GLM52_SERVICE_CONFIGURATION_DEFAULT_FLAGS \
    (SPARK_GLM52_SERVICE_CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS | \
     SPARK_GLM52_SERVICE_CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP)
#define SPARK_GLM52_SERVICE_CONFIGURATION_KNOWN_FLAGS \
    SPARK_GLM52_SERVICE_CONFIGURATION_DEFAULT_FLAGS

#define SPARK_GLM52_SERVICE_CLIENT_STATE_FREE 0u
#define SPARK_GLM52_SERVICE_CLIENT_STATE_CONNECTED 1u
#define SPARK_GLM52_SERVICE_CLIENT_STATE_DRAINING 2u

#define SPARK_GLM52_SERVICE_REQUEST_STATE_FREE 0u
#define SPARK_GLM52_SERVICE_REQUEST_STATE_LIVE 1u
#define SPARK_GLM52_SERVICE_REQUEST_STATE_COMPLETED 2u
#define SPARK_GLM52_SERVICE_REQUEST_STATE_CANCELLED 3u

#define SPARK_GLM52_SERVICE_EVENT_KIND_NONE 0u
#define SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_ACCEPTED 1u
#define SPARK_GLM52_SERVICE_EVENT_KIND_PREFILL_PROGRESS 2u
#define SPARK_GLM52_SERVICE_EVENT_KIND_TOKEN 3u
#define SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_COMPLETED 4u
#define SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_CANCELLED 5u
#define SPARK_GLM52_SERVICE_EVENT_KIND_ERROR 6u
#define SPARK_GLM52_SERVICE_EVENT_KIND_BACKPRESSURE 7u
#define SPARK_GLM52_SERVICE_EVENT_KIND_CLIENT_CONNECTED 8u
#define SPARK_GLM52_SERVICE_EVENT_KIND_CLIENT_DISCONNECTED 9u
#define SPARK_GLM52_SERVICE_EVENT_KIND_STATS 10u

#define SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_TEXT 1u
#define SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_TOKEN_IDS 2u
#define SPARK_GLM52_SERVICE_FRAME_KIND_CANCEL_REQUEST 3u
#define SPARK_GLM52_SERVICE_FRAME_KIND_PING 4u
#define SPARK_GLM52_SERVICE_FRAME_KIND_EVENT 100u
#define SPARK_GLM52_SERVICE_FRAME_KIND_SUBMIT_ACK 101u
#define SPARK_GLM52_SERVICE_FRAME_KIND_ERROR 102u
#define SPARK_GLM52_SERVICE_FRAME_KIND_PONG 103u

#define SPARK_GLM52_SERVICE_FRAME_FLAG_REALTIME \
    SPARK_GLM52_SERVING_SUBMIT_FLAG_REALTIME
#define SPARK_GLM52_SERVICE_FRAME_FLAG_DISABLE_SPECULATION \
    SPARK_GLM52_SERVING_SUBMIT_FLAG_DISABLE_SPECULATION
#define SPARK_GLM52_SERVICE_FRAME_KNOWN_SUBMIT_FLAGS \
    SPARK_GLM52_SERVING_SUBMIT_KNOWN_FLAGS

typedef uint64_t SparkGlm52ServiceClientId;
typedef uint64_t SparkGlm52ServiceRequestId;

typedef struct SparkGlm52ServiceClientSession
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t flags;
    SparkGlm52ServiceClientId client_id;
    uint64_t user_cookie;
    uint64_t accepted_request_count;
    uint64_t completed_request_count;
    uint32_t client_hash_next;
    uint32_t reserved0;
} SparkGlm52ServiceClientSession;

typedef struct SparkGlm52ServiceRequestMap
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t flags;
    SparkGlm52ServiceClientId client_id;
    SparkGlm52ServiceRequestId client_request_id;
    uint64_t serving_request_id;
    uint64_t sequence_id;
    SparkGlm52ServingRequestHandle serving_request_handle;
    uint32_t client_request_hash_next;
    uint32_t serving_handle_hash_next;
} SparkGlm52ServiceRequestMap;

typedef struct SparkGlm52ServiceEvent
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t flags;
    uint32_t status;
    uint32_t token_id;
    uint32_t token_index;
    uint32_t token_count;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t dispatch_kind;
    uint32_t dispatch_flags;
    SparkGlm52ServiceClientId client_id;
    SparkGlm52ServiceRequestId client_request_id;
    uint64_t serving_request_id;
    uint64_t sequence_id;
    SparkGlm52ServingRequestHandle serving_request_handle;
} SparkGlm52ServiceEvent;

typedef struct SparkGlm52ServiceSubmitTextRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t tokenizer_encode_flags;
    SparkGlm52ServiceClientId client_id;
    SparkGlm52ServiceRequestId client_request_id;
    uint64_t sequence_id;
    const char *text;
    uint32_t text_bytes;
    uint32_t reserved0;
} SparkGlm52ServiceSubmitTextRequest;

typedef struct SparkGlm52ServiceSubmitTokenIdsRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t token_count;
    SparkGlm52ServiceClientId client_id;
    SparkGlm52ServiceRequestId client_request_id;
    uint64_t sequence_id;
    const uint32_t *token_ids;
} SparkGlm52ServiceSubmitTokenIdsRequest;

typedef struct SparkGlm52ServiceSubmitResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t prompt_token_count;
    uint32_t required_token_capacity;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    SparkGlm52ServiceClientId client_id;
    SparkGlm52ServiceRequestId client_request_id;
    uint64_t serving_request_id;
    uint64_t sequence_id;
    SparkGlm52ServingRequestHandle serving_request_handle;
} SparkGlm52ServiceSubmitResult;

typedef struct SparkGlm52ServiceStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t connected_client_count;
    uint32_t live_request_count;
    uint32_t completed_request_mapping_count;
    uint32_t event_count;
    uint32_t event_capacity;
    uint32_t dropped_event_count;
    uint32_t last_status;
    uint32_t reserved0;
    uint64_t submitted_request_count;
    uint64_t accepted_request_count;
    uint64_t forwarded_event_count;
    uint64_t engine_pump_count;
    SparkGlm52ServingStats serving_stats;
} SparkGlm52ServiceStats;

typedef struct SparkGlm52ServiceConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t default_pump_dispatch_steps;
    uint64_t request_id_base;
    SparkGlm52ServingEngine *serving_engine;
    SparkGlm52ServiceClientSession *client_sessions;
    uint32_t client_session_capacity;
    SparkGlm52ServiceRequestMap *request_maps;
    uint32_t request_map_capacity;
    SparkGlm52ServiceEvent *event_ring;
    uint32_t event_ring_capacity;
} SparkGlm52ServiceConfiguration;

typedef struct SparkGlm52ServiceRuntime
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t default_pump_dispatch_steps;
    uint64_t next_generated_request_id;
    uint64_t next_generated_client_id;
    SparkGlm52ServingEngine *serving_engine;
    SparkGlm52ServiceClientSession *client_sessions;
    uint32_t client_session_capacity;
    SparkGlm52ServiceRequestMap *request_maps;
    uint32_t request_map_capacity;
    SparkGlm52ServiceEvent *event_ring;
    uint32_t event_ring_capacity;
    uint32_t event_read_index;
    uint32_t event_write_index;
    uint32_t event_count;
    uint32_t dropped_event_count;
    uint32_t client_hash_heads[SPARK_GLM52_SERVICE_CLIENT_HASH_SLOTS];
    uint32_t client_request_hash_heads[
        SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS];
    uint32_t serving_handle_hash_heads[
        SPARK_GLM52_SERVICE_REQUEST_MAP_HASH_SLOTS];
    SparkGlm52ServiceStats stats;
} SparkGlm52ServiceRuntime;

typedef struct SparkGlm52ServiceFrameHeader
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t flags;
    uint32_t body_bytes;
    SparkGlm52ServiceClientId client_id;
    SparkGlm52ServiceRequestId client_request_id;
} SparkGlm52ServiceFrameHeader;

typedef struct SparkGlm52ServiceSubmitTextFrameBody
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t tokenizer_encode_flags;
    uint32_t text_bytes;
    uint64_t sequence_id;
} SparkGlm52ServiceSubmitTextFrameBody;

typedef struct SparkGlm52ServiceSubmitTokenIdsFrameBody
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t token_count;
    uint32_t reserved0;
    uint64_t sequence_id;
} SparkGlm52ServiceSubmitTokenIdsFrameBody;

typedef struct SparkGlm52ServiceCancelFrameBody
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkGlm52ServingRequestHandle serving_request_handle;
    SparkGlm52ServiceRequestId client_request_id;
} SparkGlm52ServiceCancelFrameBody;

void SparkGlm52ServiceInitializeSubmitTextRequest(
    SparkGlm52ServiceSubmitTextRequest *request);

void SparkGlm52ServiceInitializeSubmitTokenIdsRequest(
    SparkGlm52ServiceSubmitTokenIdsRequest *request);

void SparkGlm52ServiceInitializeFrameHeader(
    SparkGlm52ServiceFrameHeader *frame_header,
    uint32_t frame_kind);

SparkStatus SparkGlm52ServiceValidateFrameHeader(
    const SparkGlm52ServiceFrameHeader *frame_header,
    uint32_t maximum_body_bytes);

SparkStatus SparkGlm52ServiceInitialize(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceConfiguration *configuration);

SparkStatus SparkGlm52ServiceRegisterClient(
    SparkGlm52ServiceRuntime *service,
    uint64_t user_cookie,
    SparkGlm52ServiceClientId *client_id_out);

SparkStatus SparkGlm52ServiceDisconnectClient(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id);

SparkStatus SparkGlm52ServiceSubmitTokenIds(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceSubmitTokenIdsRequest *request,
    SparkGlm52ServiceSubmitResult *result);

SparkStatus SparkGlm52ServiceSubmitText(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52ServiceSubmitTextRequest *request,
    SparkGlm52ServiceSubmitResult *result);

SparkStatus SparkGlm52ServiceHandleSubmitTokenIdsFrame(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    const SparkGlm52ServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes,
    SparkGlm52ServiceSubmitResult *result);

SparkStatus SparkGlm52ServiceHandleSubmitTextFrame(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    const SparkGlm52ServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes,
    SparkGlm52ServiceSubmitResult *result);

SparkStatus SparkGlm52ServiceHandleCancelFrame(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    const SparkGlm52ServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes);

SparkStatus SparkGlm52ServiceBuildEventFrame(
    const SparkGlm52ServiceEvent *event,
    SparkGlm52ServiceFrameHeader *frame_header,
    SparkGlm52ServiceEvent *frame_body);

SparkStatus SparkGlm52ServiceCancelRequest(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceClientId client_id,
    SparkGlm52ServiceRequestId client_request_id);

SparkStatus SparkGlm52ServicePump(
    SparkGlm52ServiceRuntime *service,
    uint32_t max_dispatch_steps,
    SparkGlm52ServiceStats *stats_out);

SparkStatus SparkGlm52ServicePopEvent(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceEvent *event_out);

SparkStatus SparkGlm52ServiceGetStats(
    SparkGlm52ServiceRuntime *service,
    SparkGlm52ServiceStats *stats_out);

#ifdef __cplusplus
}
#endif
