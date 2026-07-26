#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_tokenizer.h"
#include "sparkpipe_tool_file.h"

static double SparkTokenizerBenchmarkSeconds(void)
{
    struct timespec timestamp;

#if defined(TIME_UTC)
    if (timespec_get(&timestamp, TIME_UTC) != TIME_UTC)
    {
        return 0.0;
    }
#else
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0)
    {
        return 0.0;
    }
#endif
    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 0.000000001;
}

static int SparkTokenizerBenchmarkParseUint32(
    const char *text,
    uint32_t *value_out)
{
    char *end;
    unsigned long value;

    if (text == 0 || value_out == 0)
    {
        return 1;
    }
    value = strtoul(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value > UINT32_MAX)
    {
        return 1;
    }
    *value_out = (uint32_t)value;
    return 0;
}

static void SparkTokenizerBenchmarkPrintUsage(
    const char *program_name)
{
    fprintf(stderr,
        "usage: %s (--tokenizer-json path | --tokenizer-compiled path) [options]\n"
        "options:\n"
        "  --save-compiled-tokenizer path\n"
        "  --prompt text\n"
        "  --prompt-file path\n"
        "  --batch count              default 16\n"
        "  --iterations count         default 16\n"
        "  --workers count            default 16, capped by tokenizer API\n"
        "  --capacity tokens          default prompt_bytes + 32\n"
        "  --disable-regex-pretokenization\n",
        program_name);
}

int main(
    int argc,
    char **argv)
{
    const char *tokenizer_json_path;
    const char *compiled_tokenizer_path;
    const char *save_compiled_tokenizer_path;
    const char *prompt_literal;
    const char *prompt_file_path;
    char *owned_prompt_text;
    const char *prompt_text;
    uint32_t prompt_bytes;
    uint32_t batch_count;
    uint32_t iteration_count;
    uint32_t worker_count;
    uint32_t token_capacity;
    uint32_t encode_flags;
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration json_configuration;
    SparkTokenizerCompiledFileConfiguration compiled_configuration;
    SparkTokenizerBatchEncodeConfiguration batch_configuration;
    const char **texts;
    uint32_t *text_bytes;
    uint32_t *token_ids;
    uint32_t *token_counts;
    uint32_t *overflow_counts;
    uint32_t *invalid_counts;
    uint64_t total_input_bytes;
    uint64_t total_output_tokens;
    uint64_t total_overflow_tokens;
    uint64_t total_invalid_segments;
    double load_start_seconds;
    double load_seconds;
    double encode_start_seconds;
    double encode_seconds;
    uint32_t argument_index;
    uint32_t iteration_index;
    uint32_t batch_index;
    SparkStatus status;

    tokenizer_json_path = 0;
    compiled_tokenizer_path = 0;
    save_compiled_tokenizer_path = 0;
    prompt_literal = "The quick brown fox explains SparkPipe tokenizer throughput.\n";
    prompt_file_path = 0;
    owned_prompt_text = 0;
    prompt_text = 0;
    prompt_bytes = 0u;
    batch_count = 16u;
    iteration_count = 16u;
    worker_count = SPARK_TOKENIZER_MAX_PARALLEL_WORKER_COUNT;
    token_capacity = 0u;
    encode_flags = 0u;

    for (argument_index = 1u; argument_index < (uint32_t)argc; ++argument_index)
    {
        if (strcmp(argv[argument_index], "--tokenizer-json") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            tokenizer_json_path = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--tokenizer-compiled") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            compiled_tokenizer_path = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--save-compiled-tokenizer") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            save_compiled_tokenizer_path = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--prompt") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            prompt_literal = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--prompt-file") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            prompt_file_path = argv[++argument_index];
        }
        else if (strcmp(argv[argument_index], "--batch") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            if (SparkTokenizerBenchmarkParseUint32(argv[++argument_index], &batch_count) != 0 || batch_count == 0u)
            {
                SparkTokenizerBenchmarkPrintUsage(argv[0]);
                return 2;
            }
        }
        else if (strcmp(argv[argument_index], "--iterations") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            if (SparkTokenizerBenchmarkParseUint32(argv[++argument_index], &iteration_count) != 0 || iteration_count == 0u)
            {
                SparkTokenizerBenchmarkPrintUsage(argv[0]);
                return 2;
            }
        }
        else if (strcmp(argv[argument_index], "--workers") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            if (SparkTokenizerBenchmarkParseUint32(argv[++argument_index], &worker_count) != 0)
            {
                SparkTokenizerBenchmarkPrintUsage(argv[0]);
                return 2;
            }
        }
        else if (strcmp(argv[argument_index], "--capacity") == 0 && argument_index + 1u < (uint32_t)argc)
        {
            if (SparkTokenizerBenchmarkParseUint32(argv[++argument_index], &token_capacity) != 0 || token_capacity == 0u)
            {
                SparkTokenizerBenchmarkPrintUsage(argv[0]);
                return 2;
            }
        }
        else if (strcmp(argv[argument_index], "--disable-regex-pretokenization") == 0)
        {
            encode_flags |= SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION;
        }
        else
        {
            SparkTokenizerBenchmarkPrintUsage(argv[0]);
            return 2;
        }
    }

    if ((tokenizer_json_path == 0 && compiled_tokenizer_path == 0) ||
        (tokenizer_json_path != 0 && compiled_tokenizer_path != 0))
    {
        SparkTokenizerBenchmarkPrintUsage(argv[0]);
        return 2;
    }

    if (prompt_file_path != 0)
    {
        if (((owned_prompt_text = SparkToolReadWholeFile(prompt_file_path, &prompt_bytes)) == 0))
        {
            return 1;
        }
        prompt_text = owned_prompt_text;
    }
    else
    {
        prompt_text = prompt_literal;
        prompt_bytes = (uint32_t)strlen(prompt_text);
    }
    if (token_capacity == 0u)
    {
        token_capacity = prompt_bytes + 32u;
        if (token_capacity < 64u)
        {
            token_capacity = 64u;
        }
    }

    SparkTokenizerReset(&tokenizer);
    load_start_seconds = SparkTokenizerBenchmarkSeconds();
    if (compiled_tokenizer_path != 0)
    {
        memset(&compiled_configuration, 0, sizeof(compiled_configuration));
        compiled_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        compiled_configuration.descriptor_bytes =
            SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
        compiled_configuration.compiled_tokenizer_path = compiled_tokenizer_path;
        status = SparkTokenizerLoadCompiledFile(&tokenizer, &compiled_configuration);
    }
    else
    {
        memset(&json_configuration, 0, sizeof(json_configuration));
        json_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        json_configuration.descriptor_bytes =
            SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
        json_configuration.tokenizer_json_path = tokenizer_json_path;
        status = SparkTokenizerLoadHuggingFaceJson(&tokenizer, &json_configuration);
    }
    load_seconds = SparkTokenizerBenchmarkSeconds() - load_start_seconds;
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "tokenizer load failed: %d\n", (int)status);
        free(owned_prompt_text);
        return 1;
    }

    if (save_compiled_tokenizer_path != 0)
    {
        memset(&compiled_configuration, 0, sizeof(compiled_configuration));
        compiled_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        compiled_configuration.descriptor_bytes =
            SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
        compiled_configuration.compiled_tokenizer_path = save_compiled_tokenizer_path;
        status = SparkTokenizerSaveCompiledFile(&tokenizer, &compiled_configuration);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "compiled tokenizer save failed: %d\n", (int)status);
            SparkTokenizerDestroy(&tokenizer);
            free(owned_prompt_text);
            return 1;
        }
    }

    texts = (const char **)calloc(batch_count, sizeof(*texts));
    text_bytes = (uint32_t *)calloc(batch_count, sizeof(*text_bytes));
    token_ids = (uint32_t *)calloc((uint64_t)batch_count * token_capacity, sizeof(*token_ids));
    token_counts = (uint32_t *)calloc(batch_count, sizeof(*token_counts));
    overflow_counts = (uint32_t *)calloc(batch_count, sizeof(*overflow_counts));
    invalid_counts = (uint32_t *)calloc(batch_count, sizeof(*invalid_counts));
    if (texts == 0 || text_bytes == 0 || token_ids == 0 ||
        token_counts == 0 || overflow_counts == 0 || invalid_counts == 0)
    {
        fprintf(stderr, "benchmark allocation failed\n");
        free(texts);
        free(text_bytes);
        free(token_ids);
        free(token_counts);
        free(overflow_counts);
        free(invalid_counts);
        SparkTokenizerDestroy(&tokenizer);
        free(owned_prompt_text);
        return 1;
    }
    for (batch_index = 0u; batch_index < batch_count; ++batch_index)
    {
        texts[batch_index] = prompt_text;
        text_bytes[batch_index] = prompt_bytes;
    }

    memset(&batch_configuration, 0, sizeof(batch_configuration));
    batch_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    batch_configuration.descriptor_bytes =
        SPARK_TOKENIZER_BATCH_ENCODE_CONFIGURATION_DESCRIPTOR_BYTES;
    batch_configuration.texts = texts;
    batch_configuration.text_bytes = text_bytes;
    batch_configuration.text_count = batch_count;
    batch_configuration.encode_flags = encode_flags;
    batch_configuration.token_ids = token_ids;
    batch_configuration.token_stride = token_capacity;
    batch_configuration.token_counts = token_counts;
    batch_configuration.overflow_token_counts = overflow_counts;
    batch_configuration.invalid_segment_counts = invalid_counts;
    batch_configuration.worker_count = worker_count;

    total_input_bytes = 0u;
    total_output_tokens = 0u;
    total_overflow_tokens = 0u;
    total_invalid_segments = 0u;
    encode_start_seconds = SparkTokenizerBenchmarkSeconds();
    for (iteration_index = 0u; iteration_index < iteration_count; ++iteration_index)
    {
        memset(token_counts, 0, (size_t)batch_count * sizeof(*token_counts));
        memset(overflow_counts, 0, (size_t)batch_count * sizeof(*overflow_counts));
        memset(invalid_counts, 0, (size_t)batch_count * sizeof(*invalid_counts));
        status = SparkTokenizerEncodeBatchUtf8Configured(&tokenizer, &batch_configuration);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_CAPACITY_EXCEEDED)
        {
            fprintf(stderr, "tokenizer encode failed: %d\n", (int)status);
            break;
        }
        for (batch_index = 0u; batch_index < batch_count; ++batch_index)
        {
            total_input_bytes += prompt_bytes;
            total_output_tokens += token_counts[batch_index];
            total_overflow_tokens += overflow_counts[batch_index];
            total_invalid_segments += invalid_counts[batch_index];
        }
    }
    encode_seconds = SparkTokenizerBenchmarkSeconds() - encode_start_seconds;

    printf("load_seconds %.6f\n", load_seconds);
    printf("encode_seconds %.6f\n", encode_seconds);
    printf("batch_count %u\n", batch_count);
    printf("iteration_count %u\n", iteration_count);
    printf("worker_count %u\n", worker_count);
    printf("prompt_bytes %u\n", prompt_bytes);
    printf("total_input_bytes %llu\n", (unsigned long long)total_input_bytes);
    printf("total_output_tokens %llu\n", (unsigned long long)total_output_tokens);
    printf("total_overflow_tokens %llu\n", (unsigned long long)total_overflow_tokens);
    printf("total_invalid_segments %llu\n", (unsigned long long)total_invalid_segments);
    if (encode_seconds > 0.0)
    {
        printf("input_bytes_per_second %.2f\n", (double)total_input_bytes / encode_seconds);
        printf("output_tokens_per_second %.2f\n", (double)total_output_tokens / encode_seconds);
        printf("prompts_per_second %.2f\n", (double)((uint64_t)batch_count * iteration_count) / encode_seconds);
    }

    free(texts);
    free(text_bytes);
    free(token_ids);
    free(token_counts);
    free(overflow_counts);
    free(invalid_counts);
    SparkTokenizerDestroy(&tokenizer);
    free(owned_prompt_text);
    return status == SPARK_STATUS_OK || status == SPARK_STATUS_CAPACITY_EXCEEDED ? 0 : 1;
}
