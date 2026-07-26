#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_compat_api.h"
#include "sparkpipe/spark_glm52_service_backend.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_HTTP_GATEWAY_ABI_VERSION 1u
#define SPARK_GLM52_HTTP_GATEWAY_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkGlm52HttpGatewayRequest))
#define SPARK_GLM52_HTTP_GATEWAY_RESPONSE_BYTES \
    ((uint32_t)sizeof(SparkGlm52HttpGatewayResponse))

#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE 0u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI 1u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH 2u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_CHAT 3u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS 4u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES 5u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_CORS_PREFLIGHT 6u

#define SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM 0x00000001u

#define SPARK_GLM52_HTTP_GATEWAY_DEFAULT_MAX_CONTEXT_TOKENS \
    SPARK_GLM52_SERVICE_MAX_TOKEN_FRAME_COUNT
#define SPARK_GLM52_HTTP_GATEWAY_DEFAULT_MAX_UPLOAD_BYTES \
    SPARK_GLM52_SERVICE_MAX_TEXT_BYTES

typedef struct SparkGlm52HttpGatewayRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *method;
    const char *path;
    const char *body;
    uint32_t body_bytes;
    const char *authorization;
    uint32_t authorization_bytes;
} SparkGlm52HttpGatewayRequest;

typedef struct SparkGlm52HttpGatewayResponse
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t status_code;
    uint32_t flags;
    const char *content_type;
    char *body;
    uint32_t body_capacity;
    uint32_t body_bytes;
} SparkGlm52HttpGatewayResponse;

void SparkGlm52HttpGatewayInitializeRequest(
    SparkGlm52HttpGatewayRequest *request);

void SparkGlm52HttpGatewayInitializeResponse(
    SparkGlm52HttpGatewayResponse *response,
    char *body,
    uint32_t body_capacity);

uint32_t SparkGlm52HttpGatewayRoute(
    const SparkGlm52HttpGatewayRequest *request);

SparkStatus SparkGlm52HttpGatewayBuildDemoUi(
    SparkGlm52HttpGatewayResponse *response);

SparkStatus SparkGlm52HttpGatewayBuildHealth(
    SparkGlm52HttpGatewayResponse *response,
    uint32_t runtime_initialized,
    uint32_t local_control_ready);

SparkStatus SparkGlm52HttpGatewayBuildBackendUnavailable(
    SparkGlm52HttpGatewayResponse *response,
    uint32_t stream);

SparkStatus SparkGlm52HttpGatewayBuildRequestTimeout(
    SparkGlm52HttpGatewayResponse *response,
    uint32_t stream);

SparkStatus SparkGlm52HttpGatewayBuildUnauthorized(
    SparkGlm52HttpGatewayResponse *response);

SparkStatus SparkGlm52HttpGatewayBuildNotFound(
    SparkGlm52HttpGatewayResponse *response);

SparkStatus SparkGlm52HttpGatewayBuildCorsPreflight(
    SparkGlm52HttpGatewayResponse *response);

uint32_t SparkGlm52HttpGatewayBodyRequestsStream(
    const char *body,
    uint32_t body_bytes);

uint32_t SparkGlm52HttpGatewayAuthorizationMatches(
    const SparkGlm52HttpGatewayRequest *request,
    const char *api_key);

SparkStatus SparkGlm52HttpGatewayBuildServiceHealth(
    SparkGlm52HttpGatewayResponse *response,
    const SparkGlm52ServiceStats *stats,
    const SparkGlm52ServiceBackendView *backend_view);

SparkStatus SparkGlm52HttpGatewaySubmitJsonToService(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52HttpGatewayRequest *request,
    SparkGlm52ServiceClientId client_id,
    SparkGlm52ServiceRequestId client_request_id,
    SparkGlm52CompatTextRequest *compat_request,
    SparkGlm52ServiceSubmitResult *submit_result,
    SparkGlm52HttpGatewayResponse *response);

SparkStatus SparkGlm52HttpGatewayBuildSubmitAccepted(
    SparkGlm52HttpGatewayResponse *response,
    const SparkGlm52ServiceSubmitResult *submit_result,
    uint32_t stream);

SparkStatus SparkGlm52HttpGatewayBuildServiceEventStream(
    SparkGlm52HttpGatewayResponse *response,
    const SparkGlm52ServiceEvent *service_event,
    const SparkTokenizer *tokenizer);

#ifdef __cplusplus
}
#endif
