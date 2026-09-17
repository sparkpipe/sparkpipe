#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_chat_template.h"
#include "sparkpipe/spark_tokenizer.h"

#include "sparkpipe/spark_glm52_model.h"
#include "tools/sparkpipe_tool_file.h"
#include "runtime/net.h"

#define SPARK_GLM52_TOKENIZE_DEFAULT_TOKEN_CAPACITY \
    SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS

static const char SparkGlm52TokenizerFileName[] = "tokenizer.json";

static uint32_t SparkGlm52TokenizeStringLengthU32(
    const char *text)
{
    size_t text_bytes;

    if (text == 0)
    {
        return 0u;
    }
    text_bytes = strlen(text);
    if (text_bytes > 0xffffffffull)
    {
        return 0xffffffffu;
    }
    return (uint32_t)text_bytes;
}

static int32_t SparkGlm52TokenizeBuildTokenizerPath(
    const char *model_dir,
    char **tokenizer_json_path_out)
{
    uint32_t model_dir_bytes;
    uint32_t needs_separator;
    uint32_t path_bytes;
    char *path;

    if (model_dir == 0 || tokenizer_json_path_out == 0)
    {
        return -1;
    }
    *tokenizer_json_path_out = 0;
    model_dir_bytes = SparkGlm52TokenizeStringLengthU32(model_dir);
    needs_separator = model_dir_bytes != 0u && model_dir[model_dir_bytes - 1u] != '/' ? 1u : 0u;
    if (model_dir_bytes >
        0xffffffffu - sizeof(SparkGlm52TokenizerFileName) - needs_separator)
    {
        return -2;
    }
    path_bytes = model_dir_bytes + needs_separator +
        sizeof(SparkGlm52TokenizerFileName) - 1u;
    path = (char *)malloc((size_t)path_bytes + 1u);
    if (path == 0)
    {
        return -3;
    }
    memcpy(path, model_dir, model_dir_bytes);
    if (needs_separator != 0u)
    {
        path[model_dir_bytes] = '/';
    }
    memcpy(
        path + model_dir_bytes + needs_separator,
        SparkGlm52TokenizerFileName,
        sizeof(SparkGlm52TokenizerFileName));
    *tokenizer_json_path_out = path;
    return 0;
}

static int32_t SparkGlm52TokenizeBuildSimpleChatPrompt(
    const char *prompt,
    uint32_t prompt_bytes,
    const char *system_prompt,
    uint32_t system_prompt_bytes,
    const char *reasoning_effort,
    uint32_t chat_flags,
    char **chat_text_out,
    uint32_t *chat_text_bytes_out)
{
    SparkGlm52ChatTemplateWriter writer;
    SparkStatus status;
    char *chat_text;

    if (prompt == 0 || chat_text_out == 0 || chat_text_bytes_out == 0)
        return -1;
    *chat_text_out = 0;
    *chat_text_bytes_out = 0u;
    status = SparkGlm52ChatTemplateInitializeWriter(
        &writer,
        0,
        UINT32_MAX,
        0u);
    if (status != SPARK_STATUS_OK)
        return -2;
    status = SparkGlm52ChatTemplateRenderSimple(
        &writer,
        prompt,
        prompt_bytes,
        system_prompt,
        system_prompt_bytes,
        reasoning_effort,
        chat_flags);
    if (status != SPARK_STATUS_OK || writer.text_bytes == UINT32_MAX)
        return -5;
    chat_text = (char *)malloc((size_t)writer.text_bytes + 1u);
    if (chat_text == 0)
        return -6;
    status = SparkGlm52ChatTemplateInitializeWriter(
        &writer,
        chat_text,
        writer.text_bytes + 1u,
        0u);
    if (status == SPARK_STATUS_OK)
        status = SparkGlm52ChatTemplateRenderSimple(
            &writer,
            prompt,
            prompt_bytes,
            system_prompt,
            system_prompt_bytes,
            reasoning_effort,
            chat_flags);
    if (status != SPARK_STATUS_OK)
    {
        free(chat_text);
        return -7;
    }
    *chat_text_out = chat_text;
    *chat_text_bytes_out = writer.text_bytes;
    return 0;
}

static int32_t SparkGlm52TokenizeWriteTokens(
    FILE *file,
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint32_t token_index;

    if (file == 0 || token_ids == 0)
    {
        return -1;
    }
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        if (fprintf(file, "%u\n", token_ids[token_index]) < 0)
        {
            return -2;
        }
    }
    return 0;
}

static int32_t SparkGlm52TokenizeUsage(
    const char *program)
{
    fprintf(stderr,
        "usage: %s (--model-dir dir | --tokenizer-json path) (--prompt text | --prompt-file path) [--tokens-out path] [--chat] [--system-prompt text] [--system-prompt-file path] [--no-add-generation-prompt] [--disable-thinking] [--reasoning-effort Max|High] [--capacity n]\n",
        program);
    return 2;
}

int main(
    int argc,
    char **argv)
{
    const char *model_dir;
    const char *tokenizer_json_path;
    const char *prompt_argument;
    const char *prompt_file;
    const char *tokens_out_path;
    const char *system_prompt_argument;
    const char *system_prompt_file;
    const char *reasoning_effort;
    char *owned_tokenizer_json_path;
    char *prompt_text;
    char *system_prompt_text;
    char *chat_text;
    uint32_t prompt_bytes;
    uint32_t system_prompt_bytes;
    uint32_t chat_text_bytes;
    uint32_t token_capacity;
    uint32_t chat_mode;
    uint32_t chat_flags;
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    SparkTokenizerEncoding encoding;
    FILE *tokens_out;
    SparkStatus status;
    int32_t arg_index;
    int32_t parse_status;

    model_dir = 0;
    tokenizer_json_path = 0;
    prompt_argument = 0;
    prompt_file = 0;
    tokens_out_path = 0;
    system_prompt_argument = 0;
    system_prompt_file = 0;
    reasoning_effort = "Max";
    owned_tokenizer_json_path = 0;
    prompt_text = 0;
    system_prompt_text = 0;
    chat_text = 0;
    prompt_bytes = 0u;
    system_prompt_bytes = 0u;
    chat_text_bytes = 0u;
    token_capacity = SPARK_GLM52_TOKENIZE_DEFAULT_TOKEN_CAPACITY;
    chat_mode = 0u;
    chat_flags = SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT |
        SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING;

    for (arg_index = 1; arg_index < argc; ++arg_index)
    {
        if (strcmp(argv[arg_index], "--model-dir") == 0 && arg_index + 1 < argc)
        {
            model_dir = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--tokenizer-json") == 0 && arg_index + 1 < argc)
        {
            tokenizer_json_path = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--prompt") == 0 && arg_index + 1 < argc)
        {
            prompt_argument = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--prompt-file") == 0 && arg_index + 1 < argc)
        {
            prompt_file = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--tokens-out") == 0 && arg_index + 1 < argc)
        {
            tokens_out_path = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--system-prompt") == 0 && arg_index + 1 < argc)
        {
            system_prompt_argument = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--system-prompt-file") == 0 && arg_index + 1 < argc)
        {
            system_prompt_file = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--reasoning-effort") == 0 && arg_index + 1 < argc)
        {
            reasoning_effort = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--capacity") == 0 && arg_index + 1 < argc)
        {
            parse_status = SparkNetParseU32(argv[++arg_index], &token_capacity);
            if (parse_status != 0 || token_capacity == 0u)
            {
                return SparkGlm52TokenizeUsage(argv[0]);
            }
            continue;
        }
        if (strcmp(argv[arg_index], "--chat") == 0)
        {
            chat_mode = 1u;
            continue;
        }
        if (strcmp(argv[arg_index], "--no-add-generation-prompt") == 0)
        {
            chat_flags &= ~SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT;
            continue;
        }
        if (strcmp(argv[arg_index], "--disable-thinking") == 0)
        {
            chat_flags &= ~SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING;
            continue;
        }
        return SparkGlm52TokenizeUsage(argv[0]);
    }

    if ((model_dir == 0 && tokenizer_json_path == 0) ||
        (model_dir != 0 && tokenizer_json_path != 0) ||
        (prompt_argument == 0 && prompt_file == 0) ||
        (prompt_argument != 0 && prompt_file != 0) ||
        (system_prompt_argument != 0 && system_prompt_file != 0))
    {
        return SparkGlm52TokenizeUsage(argv[0]);
    }

    if (tokenizer_json_path == 0)
    {
        if (SparkGlm52TokenizeBuildTokenizerPath(
            model_dir,
            &owned_tokenizer_json_path) != 0)
        {
            fprintf(stderr, "failed to build tokenizer path\n");
            return 1;
        }
        tokenizer_json_path = owned_tokenizer_json_path;
    }

    if (prompt_argument != 0)
    {
        prompt_bytes = SparkGlm52TokenizeStringLengthU32(prompt_argument);
        prompt_text = (char *)malloc((size_t)prompt_bytes + 1u);
        if (prompt_text == 0)
        {
            free(owned_tokenizer_json_path);
            return 1;
        }
        memcpy(prompt_text, prompt_argument, prompt_bytes + 1u);
    }
    else if (((prompt_text = SparkToolReadWholeFile(prompt_file, &prompt_bytes)) == 0))
    {
        fprintf(stderr, "failed to read prompt file\n");
        free(owned_tokenizer_json_path);
        return 1;
    }

    if (system_prompt_argument != 0)
    {
        system_prompt_bytes = SparkGlm52TokenizeStringLengthU32(system_prompt_argument);
        system_prompt_text = (char *)malloc((size_t)system_prompt_bytes + 1u);
        if (system_prompt_text == 0)
        {
            free(prompt_text);
            free(owned_tokenizer_json_path);
            return 1;
        }
        memcpy(system_prompt_text, system_prompt_argument, system_prompt_bytes + 1u);
    }
    else if (system_prompt_file != 0 &&
        ((system_prompt_text = SparkToolReadWholeFile(system_prompt_file, &system_prompt_bytes)) == 0))
    {
        fprintf(stderr, "failed to read system prompt file\n");
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }

    if (chat_mode != 0u)
    {
        if (SparkGlm52TokenizeBuildSimpleChatPrompt(
            prompt_text,
            prompt_bytes,
            system_prompt_text,
            system_prompt_bytes,
            reasoning_effort,
            chat_flags,
            &chat_text,
            &chat_text_bytes) != 0)
        {
            fprintf(stderr, "failed to render GLM52 chat prompt\n");
            free(system_prompt_text);
            free(prompt_text);
            free(owned_tokenizer_json_path);
            return 1;
        }
    }
    else
    {
        chat_text = prompt_text;
        chat_text_bytes = prompt_bytes;
        prompt_text = 0;
    }

    SparkTokenizerReset(&tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = tokenizer_json_path;
    status = SparkTokenizerLoadHuggingFaceJson(&tokenizer, &configuration);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "failed to load tokenizer: %s\n", SparkStatusToString(status));
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }

    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = token_capacity;
    encoding.token_ids = (uint32_t *)malloc((size_t)token_capacity * sizeof(*encoding.token_ids));
    if (encoding.token_ids == 0)
    {
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }
    status = SparkTokenizerEncodeUtf8(
        &tokenizer,
        chat_text,
        chat_text_bytes,
        0u,
        &encoding);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "failed to encode prompt: %s\n", SparkStatusToString(status));
        free(encoding.token_ids);
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }
    if (encoding.overflow_token_count != 0u)
    {
        fprintf(stderr, "token output capacity exceeded: %u overflow tokens\n", encoding.overflow_token_count);
        free(encoding.token_ids);
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }

    tokens_out = stdout;
    if (tokens_out_path != 0)
    {
        tokens_out = fopen(tokens_out_path, "wb");
        if (tokens_out == 0)
        {
            fprintf(stderr, "failed to open token output\n");
            free(encoding.token_ids);
            SparkTokenizerDestroy(&tokenizer);
            free(chat_text);
            free(system_prompt_text);
            free(prompt_text);
            free(owned_tokenizer_json_path);
            return 1;
        }
    }
    if (SparkGlm52TokenizeWriteTokens(
        tokens_out,
        encoding.token_ids,
        encoding.token_count) != 0)
    {
        fprintf(stderr, "failed to write token output\n");
        if (tokens_out_path != 0)
        {
            fclose(tokens_out);
        }
        free(encoding.token_ids);
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }
    if (tokens_out_path != 0)
    {
        fclose(tokens_out);
    }
    fprintf(stderr,
        "sparkpipe_glm52_tokenize_token_count=%u\n",
        encoding.token_count);
    fprintf(stderr,
        "sparkpipe_glm52_tokenize_backend=c_bytelevel_bpe\n");

    free(encoding.token_ids);
    SparkTokenizerDestroy(&tokenizer);
    free(chat_text);
    free(system_prompt_text);
    free(prompt_text);
    free(owned_tokenizer_json_path);
    return 0;
}
