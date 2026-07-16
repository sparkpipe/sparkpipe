#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_http_gateway.h"
#include "sparkpipe/spark_glm52_model.h"

#define main SparkTestHttpGatewayToolMain
#include "../tools/sparkpipe_glm52_http_gateway.c"
#undef main

static void SparkTestHttpGatewayQueuesBeyondActiveLanes(void)
{
	static SparkGlm52GatewayRuntime runtime;
	SparkGlm52GatewayPendingStream *stream;
	uint64_t client_request_id;
	uint32_t slot_index;
	uint32_t index;

	SparkGlm52GatewayInitializeConfig(&runtime.configuration);
	runtime.configuration.max_active_sequence_count = 4u;
	assert(SparkGlm52GatewayInitializePendingStreams(&runtime) == 0);
	assert(runtime.pending_stream_capacity ==
		SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY);
	for (index = 0u; index < 16u; ++index)
	{
		stream = SparkGlm52GatewayAllocatePendingStream(
			&runtime,
			(int32_t)(100u + index),
			&slot_index,
			&client_request_id);
		assert(stream != 0);
		assert(slot_index == index);
		assert(client_request_id != 0u);
	}
	assert(runtime.pending_stream_count == 16u);
	assert(SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY ==
		SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT *
		SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET);
	assert(SPARK_GLM52_GATEWAY_PENDING_STREAM_CAPACITY >
		runtime.configuration.max_active_sequence_count);
}

static void SparkTestHttpGatewayCoalescesOnlyIdlePartialBatch(void)
{
	static SparkGlm52GatewayRuntime runtime;

	SparkGlm52GatewayInitializeConfig(&runtime.configuration);
	runtime.configuration.max_active_sequence_count = 64u;
	runtime.pending_stream_count = 1u;
	assert(SparkGlm52GatewayShouldCoalesceBatch(&runtime) == 1u);
	runtime.last_live_request_count = 1u;
	assert(SparkGlm52GatewayShouldCoalesceBatch(&runtime) == 0u);
	runtime.last_live_request_count = 0u;
	runtime.pending_stream_count = 64u;
	assert(SparkGlm52GatewayShouldCoalesceBatch(&runtime) == 0u);
	runtime.pending_stream_count = 0u;
	assert(SparkGlm52GatewayShouldCoalesceBatch(&runtime) == 0u);
}

static void SparkTestHttpGatewayPollsBetweenDispatches(void)
{
	static SparkGlm52GatewayRuntime runtime;

	SparkGlm52GatewayInitializeConfig(&runtime.configuration);
	assert(runtime.configuration.pump_steps == 1u);
	assert(SparkGlm52GatewayPollTimeout(&runtime) == -1);
	runtime.pump_log_valid = 1u;
	runtime.last_pump_status = SPARK_STATUS_OK;
	assert(SparkGlm52GatewayPollTimeout(&runtime) == -1);
	runtime.last_live_request_count = 1u;
	assert(SparkGlm52GatewayPollTimeout(&runtime) == 1);
	runtime.last_live_request_count = 0u;
	runtime.last_queued_request_count = 1u;
	assert(SparkGlm52GatewayPollTimeout(&runtime) == 1);
	runtime.last_queued_request_count = 0u;
	runtime.last_pump_status = SPARK_STATUS_BUSY;
	assert(SparkGlm52GatewayPollTimeout(&runtime) == -1);
}

static void SparkTestHttpGatewayCancelsDisconnectedStream(void)
{
	static SparkGlm52GatewayRuntime runtime;
	SparkGlm52GatewayPendingStream *stream;
	struct pollfd poll_fds[2u];
	uint32_t poll_stream_slots[2u];
	uint64_t client_request_id;
	uint32_t fd_count;
	uint32_t slot_index;
	int32_t sockets[2u];

	SparkGlm52GatewayInitializeConfig(&runtime.configuration);
	runtime.configuration.max_active_sequence_count = 4u;
	assert(SparkGlm52GatewayInitializePendingStreams(&runtime) == 0);
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	stream = SparkGlm52GatewayAllocatePendingStream(
		&runtime,sockets[0u],&slot_index,&client_request_id);
	assert(stream != 0);
	memset(poll_fds,0,sizeof(poll_fds));
	fd_count = SparkGlm52GatewayAppendPendingStreamPollFds(
		&runtime,poll_fds,poll_stream_slots,2u,0u);
	assert(fd_count == 1u);
	assert((poll_fds[0u].events & POLLIN) != 0);
	assert(close(sockets[1u]) == 0);
	assert(poll(poll_fds,fd_count,1000) == 1);
	SparkGlm52GatewayHandlePendingStreamPollFds(
		&runtime,poll_fds,poll_stream_slots,0u,fd_count);
	assert(runtime.pending_streams[slot_index].active == 0u);
	assert(runtime.pending_stream_count == 0u);
	(void)client_request_id;
}

static void SparkTestHttpGatewayRoutes(void)
{
	SparkGlm52HttpGatewayRequest request;

	SparkGlm52HttpGatewayInitializeRequest(&request);
	request.method = "OPTIONS";
	request.path = "/v1/chat/completions";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_CORS_PREFLIGHT);
	request.method = "GET";
	request.path = "/";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI);
	request.path = "/health";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH);
	request.method = "POST";
	request.path = "/v1/chat/completions";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_CHAT);
	request.path = "/v1/completions";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS);
	request.path = "/v1/messages";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES);
	request.path = "/bad";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE);
}


static void SparkTestHttpGatewayBuildsCorsPreflight(void)
{
    SparkGlm52HttpGatewayResponse response;
    char body[16];

    SparkGlm52HttpGatewayInitializeResponse(&response, body, sizeof(body));
    assert(SparkGlm52HttpGatewayBuildCorsPreflight(&response) ==
        SPARK_STATUS_OK);
    assert(response.status_code == 200u);
    assert(response.body_bytes == 0u);
}

static void SparkTestHttpGatewayBuildsHealth(void)
{
	SparkGlm52HttpGatewayResponse response;
	char body[512];

	SparkGlm52HttpGatewayInitializeResponse(&response,body,sizeof(body));
	assert(SparkGlm52HttpGatewayBuildHealth(&response,1u,0u) ==
		SPARK_STATUS_OK);
	assert(response.status_code == 200u);
	assert(strcmp(response.content_type,"application/json") == 0);
	assert(strstr(body,"\"runtime_initialized\":1") != 0);
	assert(strstr(body,"\"local_control_ready\":0") != 0);
	assert(strstr(body,"\"performance_status\":\"NOT_MEASURED\"") != 0);
}

static void SparkTestHttpGatewayBuildsSseUnavailable(void)
{
	SparkGlm52HttpGatewayResponse response;
	char body[512];

	SparkGlm52HttpGatewayInitializeResponse(&response,body,sizeof(body));
	assert(SparkGlm52HttpGatewayBuildBackendUnavailable(&response,1u) ==
		SPARK_STATUS_OK);
	assert(response.status_code == 503u);
	assert((response.flags & SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM) != 0u);
	assert(strcmp(response.content_type,"text/event-stream") == 0);
	assert(strstr(body,"event: error") != 0);
}


static void SparkTestHttpGatewayBuildsServiceHealth(void)
{
    SparkGlm52HttpGatewayResponse response;
    SparkGlm52ServiceBackendView backend_view;
    SparkGlm52ServiceStats stats;
    char body[4096];

    memset(&stats, 0, sizeof(stats));
    memset(&backend_view, 0, sizeof(backend_view));
    stats.connected_client_count = 2u;
    stats.live_request_count = 3u;
    stats.serving_stats.queued_request_count = 4u;
    stats.serving_stats.jit_prefetch_dispatch_count = 5u;
    stats.serving_stats.mtp_draft_token_count = 6u;
    stats.serving_stats.mtp_verify_dispatch_count = 7u;
    stats.serving_stats.mtp_accepted_draft_token_count = 8u;
    stats.serving_stats.mtp_committed_token_count = 8u;
    stats.serving_stats.completed_stream_count = 1u;
    stats.serving_stats.decoded_token_count = 9u;
    stats.serving_stats.maximum_decode_lane_count = 1u;
    backend_view.runtime_initialized = 1u;
    backend_view.local_control_ready = 1u;
    backend_view.configured_kv_context_limit_tokens =
        SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS;
    backend_view.configured_max_active_sequences = 1u;
	backend_view.adaptive_decode_batch_width = 1u;
	backend_view.decode_batch_capacity = 256u;
	backend_view.prefill_wave_token_count = 16u;
    backend_view.speculation_configuration_flags =
        SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP;
    backend_view.release_id = "release-1";
    backend_view.release_git_commit = "abcdef";
    backend_view.release_generation = 42u;
    backend_view.transport_shared_object_path = "tcp-host-staged.so";
    backend_view.first_blocker = "none";
    SparkGlm52HttpGatewayInitializeResponse(&response, body, sizeof(body));
    assert(SparkGlm52HttpGatewayBuildServiceHealth(
        &response,
        &stats,
        &backend_view) ==
        SPARK_STATUS_OK);
    assert(response.status_code == 200u);
    assert(strstr(
        body,
        "\"configured_kv_context_limit_tokens\":1048576") != 0);
	assert(strstr(body,"\"adaptive_decode_batch_width\":1") != 0);
	assert(strstr(body,"\"decode_batch_capacity\":256") != 0);
	assert(strstr(body,"\"prefill_wave_token_count\":16") != 0);
    assert(strstr(body, "\"release_git_commit\":\"abcdef\"") != 0);
    assert(strstr(
        body,
        "\"end_to_end_observation_status\":\"OBSERVED\"") != 0);
    assert(strstr(body, "\"performance_status\":\"NOT_MEASURED\"") != 0);
    assert(strstr(body, "\"multi_sequence_batching_status\":\"NOT_WORKING\"") != 0);
    assert(strstr(body, "production_contract_flags") == 0);
    assert(strstr(body, "\"connected_clients\":2") != 0);
    assert(strstr(body, "\"jit_prefetch_dispatches\":5") != 0);
    assert(strstr(body, "\"mtp_draft_tokens\":6") != 0);
    assert(strstr(body, "\"mtp_verify_dispatches\":7") != 0);
    assert(strstr(body, "\"mtp_accepted_draft_tokens\":8") != 0);
    assert(strstr(body, "\"first_blocker\":\"none\"") != 0);
}

static void SparkTestHttpGatewayFormatsTokenEvent(void)
{
    SparkGlm52HttpGatewayResponse response;
    SparkGlm52ServiceEvent event;
    char body[1024];

    memset(&event, 0, sizeof(event));
    event.abi_version = SPARK_GLM52_SERVICE_ABI_VERSION;
    event.descriptor_bytes = SPARK_GLM52_SERVICE_EVENT_DESCRIPTOR_BYTES;
    event.kind = SPARK_GLM52_SERVICE_EVENT_KIND_TOKEN;
    event.client_id = 11u;
    event.client_request_id = 22u;
    event.token_id = 333u;
    event.token_index = 4u;
    SparkGlm52HttpGatewayInitializeResponse(&response, body, sizeof(body));
    assert(SparkGlm52HttpGatewayBuildServiceEventStream(
        &response,
        &event,
        0) == SPARK_STATUS_OK);
    assert(response.status_code == 200u);
    assert((response.flags & SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM) != 0u);
    assert(strstr(body, "event: token") != 0);
    assert(strstr(body, "\"token_id\":333") != 0);
}

static void SparkTestHttpGatewayAuth(void)
{
	SparkGlm52HttpGatewayRequest request;
	static const char Auth[] = "Bearer secret";

	SparkGlm52HttpGatewayInitializeRequest(&request);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,0) == 1u);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,"") == 1u);
	request.authorization = Auth;
	request.authorization_bytes = (uint32_t)strlen(Auth);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,"secret") == 1u);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,"bad") == 0u);
}

static void SparkTestHttpGatewayStreamFlag(void)
{
	static const char StreamBody[] = "{\"model\":\"glm-5.2\",\"stream\" : true}";
	static const char PlainBody[] = "{\"model\":\"glm-5.2\"}";

	assert(SparkGlm52HttpGatewayBodyRequestsStream(
		StreamBody,
		(uint32_t)strlen(StreamBody)) == 1u);
	assert(SparkGlm52HttpGatewayBodyRequestsStream(
		PlainBody,
		(uint32_t)strlen(PlainBody)) == 0u);
}

int main(void)
{
	SparkTestHttpGatewayQueuesBeyondActiveLanes();
	SparkTestHttpGatewayCoalescesOnlyIdlePartialBatch();
	SparkTestHttpGatewayPollsBetweenDispatches();
	SparkTestHttpGatewayCancelsDisconnectedStream();
	SparkTestHttpGatewayRoutes();
	SparkTestHttpGatewayBuildsCorsPreflight();
	SparkTestHttpGatewayBuildsHealth();
	SparkTestHttpGatewayBuildsServiceHealth();
	SparkTestHttpGatewayBuildsSseUnavailable();
	SparkTestHttpGatewayFormatsTokenEvent();
	SparkTestHttpGatewayAuth();
	SparkTestHttpGatewayStreamFlag();
	return 0;
}
