#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tokenizer.h"

#include "sparkpipe/spark_glm52_model.h"
#include "tools/sparkpipe_tool_file.h"

#define SPARK_TOKENIZE_PROMPT_DEFAULT_TOKEN_CAPACITY \
    SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_TOKENIZE_PROMPT_MAX_TEXT_BYTES 0xffffffffull

static int SparkTokenizePromptUsage(
    const char *program)
{
    fprintf(
        stderr,
        "usage: %s (--tokenizer-json path | --tokenizer-compiled path) (--prompt text | --prompt-file path | --text text | --text-file path) [--output path] [--save-compiled-tokenizer path] [--disable-special | --disable-special-token-match] [--add-prefix-space] [--disable-regex-pretokenization]\n",
        program);
    return 2;
}

static int SparkTokenizePromptWriteTokens(
    const char *path,
    const uint32_t *token_ids,
    uint32_t token_count)
{
    FILE *file;
    uint32_t token_index;

    if (token_ids == 0 && token_count != 0u)
    {
        return -1;
    }

    file = stdout;
    if (path != 0)
    {
        file = fopen(path, "wb");
        if (file == 0)
        {
            return -1;
        }
    }

    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        if (fprintf(file, "%u\n", token_ids[token_index]) < 0)
        {
            if (path != 0)
            {
                fclose(file);
            }
            return -1;
        }
    }

    if (path != 0 && fclose(file) != 0)
    {
        return -1;
    }
    return 0;
}

static SparkStatus SparkTokenizePromptParseArguments(
    int argc,
    char **argv,
    const char **tokenizer_json_path_out,
    const char **tokenizer_compiled_path_out,
    const char **save_compiled_path_out,
    const char **prompt_text_out,
    const char **prompt_file_path_out,
    const char **output_path_out,
    uint32_t *encode_flags_out)
{
    int argument_index;

    if (tokenizer_json_path_out == 0 || tokenizer_compiled_path_out == 0 ||
        save_compiled_path_out == 0 || prompt_text_out == 0 ||
        prompt_file_path_out == 0 || output_path_out == 0 ||
        encode_flags_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *tokenizer_json_path_out = 0;
    *tokenizer_compiled_path_out = 0;
    *save_compiled_path_out = 0;
    *prompt_text_out = 0;
    *prompt_file_path_out = 0;
    *output_path_out = 0;
    *encode_flags_out = 0u;

    for (argument_index = 1; argument_index < argc; ++argument_index)
    {
        if (strcmp(argv[argument_index], "--tokenizer-json") == 0 &&
            argument_index + 1 < argc)
        {
            *tokenizer_json_path_out = argv[++argument_index];
        }
        else if ((strcmp(argv[argument_index], "--tokenizer-compiled") == 0 ||
                  strcmp(argv[argument_index], "--compiled-tokenizer") == 0) &&
            argument_index + 1 < argc)
        {
            *tokenizer_compiled_path_out = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--save-compiled-tokenizer") == 0 &&
            argument_index + 1 < argc)
        {
            *save_compiled_path_out = argv[++argument_index];
        }
        else if ((strcmp(argv[argument_index], "--prompt") == 0 ||
                  strcmp(argv[argument_index], "--text") == 0) &&
            argument_index + 1 < argc)
        {
            *prompt_text_out = argv[++argument_index];
        }
        else if ((strcmp(argv[argument_index], "--prompt-file") == 0 ||
                  strcmp(argv[argument_index], "--text-file") == 0) &&
            argument_index + 1 < argc)
        {
            *prompt_file_path_out = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--output") == 0 &&
            argument_index + 1 < argc)
        {
            *output_path_out = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--disable-special") == 0 ||
            strcmp(argv[argument_index], "--disable-special-token-match") == 0)
        {
            *encode_flags_out |= SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH;
        }
        else if (strcmp(argv[argument_index], "--add-prefix-space") == 0)
        {
            *encode_flags_out |= SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE;
        }
        else if (strcmp(argv[argument_index], "--disable-regex-pretokenization") == 0)
        {
            *encode_flags_out |= SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION;
        }
        else
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    if (((*tokenizer_json_path_out == 0) == (*tokenizer_compiled_path_out == 0)) ||
        ((*prompt_text_out == 0) == (*prompt_file_path_out == 0)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

int main(
    int argc,
    char **argv)
{
    const char *tokenizer_json_path;
    const char *tokenizer_compiled_path;
    const char *save_compiled_path;
    const char *prompt_text;
    const char *prompt_file_path;
    const char *output_path;
    char *owned_prompt_text;
    uint32_t prompt_text_bytes;
    uint32_t encode_flags;
    uint32_t *token_ids;
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration tokenizer_configuration;
    SparkTokenizerCompiledFileConfiguration compiled_configuration;
    SparkTokenizerEncoding encoding;
    SparkStatus status;

    status = SparkTokenizePromptParseArguments(
        argc,
        argv,
        &tokenizer_json_path,
        &tokenizer_compiled_path,
        &save_compiled_path,
        &prompt_text,
        &prompt_file_path,
        &output_path,
        &encode_flags);
    if (status != SPARK_STATUS_OK)
    {
        return SparkTokenizePromptUsage(argv[0]);
    }

    owned_prompt_text = 0;
    if (prompt_file_path != 0)
    {
        owned_prompt_text = SparkToolReadWholeFile(
            prompt_file_path,
            &prompt_text_bytes);
        if (owned_prompt_text == 0)
        {
            fprintf(stderr, "sparkpipe_tokenize_prompt_error=failed_to_read_prompt_file\n");
            return 1;
        }
        prompt_text = owned_prompt_text;
    }
    else
    {
        size_t prompt_text_length;

        prompt_text_length = strlen(prompt_text);
        if (prompt_text_length > SPARK_TOKENIZE_PROMPT_MAX_TEXT_BYTES)
        {
            fprintf(stderr, "sparkpipe_tokenize_prompt_error=prompt_too_large\n");
            return 1;
        }
        prompt_text_bytes = (uint32_t)prompt_text_length;
    }

    token_ids = (uint32_t *)malloc(
        (uint64_t)SPARK_TOKENIZE_PROMPT_DEFAULT_TOKEN_CAPACITY * sizeof(*token_ids));
    if (token_ids == 0)
    {
        free(owned_prompt_text);
        fprintf(stderr, "sparkpipe_tokenize_prompt_error=allocation_failed\n");
        return 1;
    }

    SparkTokenizerReset(&tokenizer);
    if (tokenizer_compiled_path != 0)
    {
        memset(&compiled_configuration, 0, sizeof(compiled_configuration));
        compiled_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        compiled_configuration.descriptor_bytes =
            SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
        compiled_configuration.compiled_tokenizer_path = tokenizer_compiled_path;
        status = SparkTokenizerLoadCompiledFile(&tokenizer, &compiled_configuration);
    }
    else
    {
        memset(&tokenizer_configuration, 0, sizeof(tokenizer_configuration));
        tokenizer_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        tokenizer_configuration.descriptor_bytes =
            SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
        tokenizer_configuration.tokenizer_json_path = tokenizer_json_path;
        status = SparkTokenizerLoadHuggingFaceJson(&tokenizer, &tokenizer_configuration);
        if (status == SPARK_STATUS_OK && save_compiled_path != 0)
        {
            memset(&compiled_configuration, 0, sizeof(compiled_configuration));
            compiled_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
            compiled_configuration.descriptor_bytes =
                SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
            compiled_configuration.compiled_tokenizer_path = save_compiled_path;
            status = SparkTokenizerSaveCompiledFile(&tokenizer, &compiled_configuration);
        }
    }
    if (status != SPARK_STATUS_OK)
    {
        free(token_ids);
        free(owned_prompt_text);
        fprintf(
            stderr,
            "sparkpipe_tokenize_prompt_error=tokenizer_load_failed status=%s\n",
            SparkStatusToString(status));
        return 1;
    }

    memset(&encoding, 0, sizeof(encoding));
    encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    encoding.descriptor_bytes = SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES;
    encoding.token_capacity = SPARK_TOKENIZE_PROMPT_DEFAULT_TOKEN_CAPACITY;
    encoding.token_ids = token_ids;
    status = SparkTokenizerEncodeUtf8(
        &tokenizer,
        prompt_text,
        prompt_text_bytes,
        encode_flags,
        &encoding);
    if (status != SPARK_STATUS_OK || encoding.overflow_token_count != 0u)
    {
        SparkTokenizerDestroy(&tokenizer);
        free(token_ids);
        free(owned_prompt_text);
        fprintf(
            stderr,
            "sparkpipe_tokenize_prompt_error=encode_failed status=%s overflow=%u invalid_segments=%u\n",
            SparkStatusToString(status),
            encoding.overflow_token_count,
            encoding.invalid_segment_count);
        return 1;
    }

    if (SparkTokenizePromptWriteTokens(
            output_path,
            token_ids,
            encoding.token_count) != 0)
    {
        SparkTokenizerDestroy(&tokenizer);
        free(token_ids);
        free(owned_prompt_text);
        fprintf(stderr, "sparkpipe_tokenize_prompt_error=write_failed\n");
        return 1;
    }

    SparkTokenizerDestroy(&tokenizer);
    free(token_ids);
    free(owned_prompt_text);
    return 0;
}
