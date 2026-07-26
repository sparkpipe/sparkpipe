#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_service.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_COMPAT_API_ABI_VERSION 2u
#define SPARK_GLM52_COMPAT_TEXT_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkGlm52CompatTextRequest))

typedef struct SparkGlm52CompatTextRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t chat_template_flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t tokenizer_encode_flags;
    SparkGlm52ServiceClientId client_id;
    SparkGlm52ServiceRequestId client_request_id;
    uint64_t sequence_id;
    char *text;
    uint32_t text_capacity;
    uint32_t text_bytes;
} SparkGlm52CompatTextRequest;

void SparkGlm52CompatInitializeTextRequest(
    SparkGlm52CompatTextRequest *request,
    char *text,
    uint32_t text_capacity);

SparkStatus SparkGlm52CompatPrepareOpenAiJson(
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request);

SparkStatus SparkGlm52CompatPrepareAnthropicJson(
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request);

SparkStatus SparkGlm52CompatSubmitOpenAiJson(
    SparkGlm52ServiceRuntime *service,
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request,
    SparkGlm52ServiceSubmitResult *result);

SparkStatus SparkGlm52CompatSubmitAnthropicJson(
    SparkGlm52ServiceRuntime *service,
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request,
    SparkGlm52ServiceSubmitResult *result);

#ifdef __cplusplus
}
#endif
