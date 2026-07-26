#ifndef SPARKPIPE_SPARK_GLM52_TEXT_PROMPT_H
#define SPARKPIPE_SPARK_GLM52_TEXT_PROMPT_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_request_api.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_TEXT_PROMPT_ABI_VERSION 1u
#define SPARK_GLM52_TEXT_PROMPT_SUBMIT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52TextPromptSubmitRequest))
#define SPARK_GLM52_TEXT_PROMPT_SUBMIT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52TextPromptSubmitResult))

typedef struct SparkGlm52TextPromptSubmitRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t prompt_token_storage_capacity;
    uint64_t request_id;
    uint64_t sequence_id;
    const char *prompt_text;
    uint32_t prompt_text_bytes;
    uint32_t tokenizer_encode_flags;
    SparkTokenizerWorkspace *tokenizer_workspace;
    uint32_t *prompt_token_storage;
} SparkGlm52TextPromptSubmitRequest;

typedef struct SparkGlm52TextPromptSubmitResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t prompt_token_count;
    uint32_t required_prompt_token_count;
    uint64_t request_handle;
} SparkGlm52TextPromptSubmitResult;

void SparkGlm52TextPromptGetDefaultSubmitRequest(
    SparkGlm52TextPromptSubmitRequest *request);

SparkStatus SparkGlm52RequestApiSubmitTextPrompt(
    SparkGlm52RequestApi *api,
    const SparkTokenizer *tokenizer,
    const SparkGlm52TextPromptSubmitRequest *request,
    SparkGlm52TextPromptSubmitResult *result);

#ifdef __cplusplus
}
#endif

#endif
