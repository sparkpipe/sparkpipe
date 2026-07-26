#include "sparkpipe/spark_glm52_text_prompt.h"

#include <string.h>

void SparkGlm52TextPromptGetDefaultSubmitRequest(
    SparkGlm52TextPromptSubmitRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_TEXT_PROMPT_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_TEXT_PROMPT_SUBMIT_DESCRIPTOR_BYTES;
}

static SparkStatus SparkGlm52TextPromptValidateSubmitRequest(
    const SparkGlm52TextPromptSubmitRequest *request)
{
    if (request == 0 ||
        request->abi_version != SPARK_GLM52_TEXT_PROMPT_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_TEXT_PROMPT_SUBMIT_DESCRIPTOR_BYTES ||
        (request->prompt_text == 0 && request->prompt_text_bytes != 0u) ||
        request->prompt_token_storage == 0 ||
        request->prompt_token_storage_capacity == 0u ||
        (request->tokenizer_encode_flags & ~SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS) != 0u ||
        (request->tokenizer_workspace != 0 &&
            (request->tokenizer_workspace->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
                request->tokenizer_workspace->descriptor_bytes !=
                    SPARK_TOKENIZER_WORKSPACE_DESCRIPTOR_BYTES ||
                request->tokenizer_workspace->symbol_capacity == 0u)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiSubmitTextPrompt(
    SparkGlm52RequestApi *api,
    const SparkTokenizer *tokenizer,
    const SparkGlm52TextPromptSubmitRequest *request,
    SparkGlm52TextPromptSubmitResult *result)
{
    SparkTokenizerEncoding encoding;
    SparkGlm52RequestApiSubmitRequest submit_request;
    SparkStatus status;

    if (result != 0)
    {
        memset(result, 0, sizeof(*result));
        result->abi_version = SPARK_GLM52_TEXT_PROMPT_ABI_VERSION;
        result->descriptor_bytes = SPARK_GLM52_TEXT_PROMPT_SUBMIT_RESULT_DESCRIPTOR_BYTES;
    }
    if (api == 0 || tokenizer == 0 || result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52TextPromptValidateSubmitRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&encoding, 0, sizeof(encoding));
    encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    encoding.descriptor_bytes = SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES;
    encoding.token_capacity = request->prompt_token_storage_capacity;
    encoding.token_ids = request->prompt_token_storage;
    if (request->tokenizer_workspace != 0)
    {
        status = SparkTokenizerEncodeUtf8WithWorkspace(
            tokenizer,
            request->prompt_text,
            request->prompt_text_bytes,
            request->tokenizer_encode_flags,
            request->tokenizer_workspace,
            &encoding);
    }
    else
    {
        status = SparkTokenizerEncodeUtf8(
            tokenizer,
            request->prompt_text,
            request->prompt_text_bytes,
            request->tokenizer_encode_flags,
            &encoding);
    }
    result->prompt_token_count = encoding.token_count;
    result->required_prompt_token_count = encoding.token_count +
        encoding.overflow_token_count;
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (result->required_prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(&submit_request, 0, sizeof(submit_request));
    submit_request.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    submit_request.descriptor_bytes = SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES;
    submit_request.flags = request->request_flags;
    submit_request.priority = request->priority;
    submit_request.prompt_token_count = result->required_prompt_token_count;
    submit_request.thinking_token_budget = request->thinking_token_budget;
    submit_request.output_token_budget = request->output_token_budget;
    submit_request.max_prefill_tokens_per_step = request->max_prefill_tokens_per_step;
    submit_request.request_id = request->request_id;
    submit_request.sequence_id = request->sequence_id;
    submit_request.prompt_token_ids = request->prompt_token_storage;

    status = SparkGlm52RequestApiSubmit(api, &submit_request, &result->request_handle);
    if (status != SPARK_STATUS_OK)
    {
        result->request_handle = SPARK_GLM52_REQUEST_API_INVALID_HANDLE;
        return status;
    }
    return SPARK_STATUS_OK;
}
