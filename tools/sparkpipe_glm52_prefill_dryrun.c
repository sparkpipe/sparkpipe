#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_request_api.h"
#include "sparkpipe/spark_tokenizer.h"
#include "sparkpipe_tool_file.h"

#define SPARK_PREFILL_DRYRUN_REQUEST_SLOT_COUNT 4u
#define SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_PREFILL_DRYRUN_PREFIX_ENTRY_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_PREFILL_DRYRUN_PREFIX_BINDING_COUNT \
    (SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY + 4u)
#define SPARK_PREFILL_DRYRUN_MAX_STEPS 1024u

typedef struct SparkPrefillDryrun
{
    SparkGlm52KvCacheArena kv_arena;
    SparkGlm52KvCacheBlock kv_blocks[
        SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT];
    SparkGlm52PrefixCache prefix_cache;
    SparkGlm52PrefixCacheEntry prefix_entries[
        SPARK_PREFILL_DRYRUN_PREFIX_ENTRY_COUNT];
    SparkGlm52PrefixCacheSequenceBinding prefix_bindings[
        SPARK_PREFILL_DRYRUN_PREFIX_BINDING_COUNT];
    SparkGlm52Scheduler scheduler;
    SparkGlm52RequestApiSlot request_slots[
        SPARK_PREFILL_DRYRUN_REQUEST_SLOT_COUNT];
    SparkGlm52RequestApi api;
    uint32_t physical_block_indices[
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
    uint32_t lane_physical_block_counts[1u];
} SparkPrefillDryrun;

static SparkPrefillDryrun Dryrun;
static uint32_t PromptTokens[SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS];

static int32_t SparkPrefillDryrunParseU32(
    const char *text,
    uint32_t *value_out)
{
    uint64_t value;
    uint32_t index;

    if (text == 0 || text[0u] == '\0' || value_out == 0)
    {
        return -1;
    }
    value = 0u;
    for (index = 0u; text[index] != '\0'; ++index)
    {
        if (text[index] < '0' || text[index] > '9')
        {
            return -2;
        }
        value = value * 10u + (uint32_t)(text[index] - '0');
        if (value > 0xffffffffull)
        {
            return -3;
        }
    }
    *value_out = (uint32_t)value;
    return 0;
}

static uint32_t SparkPrefillDryrunCharacterIsSeparator(
    int32_t character)
{
    return character == ',' || character == ' ' || character == '\n' ||
        character == '\r' || character == '\t';
}

static int32_t SparkPrefillDryrunReadTokenFile(
    const char *path,
    uint32_t *tokens,
    uint32_t token_capacity,
    uint32_t *token_count_out)
{
    FILE *file;
    uint64_t value;
    uint32_t token_count;
    uint32_t have_value;
    int32_t character;

    if (path == 0 || tokens == 0 || token_capacity == 0u ||
        token_count_out == 0)
    {
        return -1;
    }
    *token_count_out = 0u;
    file = fopen(path, "rb");
    if (file == 0)
    {
        return -2;
    }
    value = 0u;
    token_count = 0u;
    have_value = 0u;
    while ((character = fgetc(file)) != EOF)
    {
        if (character >= '0' && character <= '9')
        {
            have_value = 1u;
            value = value * 10u + (uint32_t)(character - '0');
            if (value > 0xffffffffull)
            {
                fclose(file);
                return -3;
            }
            continue;
        }
        if (!SparkPrefillDryrunCharacterIsSeparator(character))
        {
            fclose(file);
            return -4;
        }
        if (have_value != 0u)
        {
            if (token_count >= token_capacity)
            {
                fclose(file);
                return -5;
            }
            tokens[token_count++] = (uint32_t)value;
            value = 0u;
            have_value = 0u;
        }
    }
    if (have_value != 0u)
    {
        if (token_count >= token_capacity)
        {
            fclose(file);
            return -6;
        }
        tokens[token_count++] = (uint32_t)value;
    }
    fclose(file);
    if (token_count == 0u)
    {
        return -7;
    }
    *token_count_out = token_count;
    return 0;
}

static int32_t SparkPrefillDryrunWriteTokenFile(
    const char *path,
    const uint32_t *tokens,
    uint32_t token_count)
{
    FILE *file;
    uint32_t token_index;

    if (path == 0 || tokens == 0)
    {
        return -1;
    }
    file = fopen(path, "wb");
    if (file == 0)
    {
        return -2;
    }
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        if (fprintf(file, "%u\n", tokens[token_index]) < 0)
        {
            fclose(file);
            return -3;
        }
    }
    if (fclose(file) != 0)
    {
        return -4;
    }
    return 0;
}

static int32_t SparkPrefillDryrunTokenizePrompt(
    const char *tokenizer_json_path,
    const char *prompt_text,
    uint32_t prompt_text_bytes,
    uint32_t encode_flags,
    uint32_t *tokens,
    uint32_t token_capacity,
    uint32_t *token_count_out)
{
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    SparkTokenizerEncoding encoding;
    SparkStatus status;

    if (tokenizer_json_path == 0 ||
        (prompt_text == 0 && prompt_text_bytes != 0u) ||
        tokens == 0 || token_capacity == 0u || token_count_out == 0)
    {
        return -1;
    }
    *token_count_out = 0u;
    SparkTokenizerReset(&tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = tokenizer_json_path;
    status = SparkTokenizerLoadHuggingFaceJson(&tokenizer, &configuration);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "tokenizer load failed: %s\n",
            SparkStatusToString(status));
        return -2;
    }

    memset(&encoding, 0, sizeof(encoding));
    encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    encoding.descriptor_bytes = SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES;
    encoding.token_capacity = token_capacity;
    encoding.token_ids = tokens;
    status = SparkTokenizerEncodeUtf8(
        &tokenizer,
        prompt_text,
        prompt_text_bytes,
        encode_flags,
        &encoding);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "tokenizer encode failed: %s stored_tokens=%u overflow_tokens=%u invalid_segments=%u\n",
            SparkStatusToString(status),
            encoding.token_count,
            encoding.overflow_token_count,
            encoding.invalid_segment_count);
        SparkTokenizerDestroy(&tokenizer);
        return -3;
    }
    if (encoding.invalid_segment_count != 0u || encoding.token_count == 0u)
    {
        fprintf(stderr,
            "tokenizer encode rejected prompt: stored_tokens=%u invalid_segments=%u\n",
            encoding.token_count,
            encoding.invalid_segment_count);
        SparkTokenizerDestroy(&tokenizer);
        return -4;
    }
    *token_count_out = encoding.token_count;
    SparkTokenizerDestroy(&tokenizer);
    return 0;
}

static SparkStatus SparkPrefillDryrunKvPrefetch(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    (void)context;
    (void)prefetch_plan;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkPrefillDryrunInitialize(
    SparkPrefillDryrun *dryrun,
    uint32_t max_prefill_tokens_per_step)
{
    SparkGlm52KvCacheConfiguration kv_configuration;
    SparkGlm52PrefixCacheConfiguration prefix_configuration;
    SparkGlm52SchedulerConfiguration scheduler_configuration;
    SparkGlm52RequestApiConfiguration api_configuration;
    SparkStatus status;

    memset(dryrun, 0, sizeof(*dryrun));
    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count =
        SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT;
    kv_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    kv_configuration.layer_count = 78u;
    kv_configuration.kv_head_count = 8u;
    kv_configuration.head_dim = 128u;
    kv_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
    kv_configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    kv_configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    kv_configuration.blocks = dryrun->kv_blocks;
    status = SparkGlm52KvCacheArenaInitialize(
        &dryrun->kv_arena,
        &kv_configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&prefix_configuration, 0, sizeof(prefix_configuration));
    prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    prefix_configuration.descriptor_bytes =
        SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    prefix_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    prefix_configuration.entry_count =
        SPARK_PREFILL_DRYRUN_PREFIX_ENTRY_COUNT;
    prefix_configuration.physical_block_count =
        SPARK_PREFILL_DRYRUN_KV_BLOCK_COUNT;
    prefix_configuration.sequence_binding_count =
        SPARK_PREFILL_DRYRUN_PREFIX_BINDING_COUNT;
    prefix_configuration.entries = dryrun->prefix_entries;
    prefix_configuration.sequence_bindings = dryrun->prefix_bindings;
    prefix_configuration.kv_cache_arena = &dryrun->kv_arena;
    status = SparkGlm52PrefixCacheInitialize(
        &dryrun->prefix_cache,
        &prefix_configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&scheduler_configuration, 0, sizeof(scheduler_configuration));
    scheduler_configuration.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler_configuration.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
    scheduler_configuration.spark_count =
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    scheduler_configuration.queue_depth_per_spark = 2u;
    scheduler_configuration.measured_profile_id =
        SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    scheduler_configuration.quantization_mode =
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    scheduler_configuration.max_prefill_tokens_per_step =
        max_prefill_tokens_per_step;
    scheduler_configuration.prefix_cache_block_tokens =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    scheduler_configuration.configuration_flags =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
    scheduler_configuration.prefix_cache = &dryrun->prefix_cache;
    status = SparkGlm52SchedulerInitialize(
        &dryrun->scheduler,
        &scheduler_configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&api_configuration, 0, sizeof(api_configuration));
    api_configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    api_configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    api_configuration.request_capacity =
        SPARK_PREFILL_DRYRUN_REQUEST_SLOT_COUNT;
    api_configuration.prefetch_lane_count =
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    api_configuration.decode_batch_target = 1u;
    api_configuration.scheduler = &dryrun->scheduler;
    api_configuration.request_slots = dryrun->request_slots;
    api_configuration.kv_prefetch_function = SparkPrefillDryrunKvPrefetch;
    return SparkGlm52RequestApiInitialize(
        &dryrun->api,
        &api_configuration);
}

static SparkStatus SparkPrefillDryrunSubmit(
    SparkPrefillDryrun *dryrun,
    const uint32_t *tokens,
    uint32_t token_count,
    uint32_t max_prefill_tokens_per_step,
    SparkGlm52RequestApiHandle *handle_out)
{
    SparkGlm52RequestApiSubmitRequest request;

    memset(&request, 0, sizeof(request));
    request.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    request.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES;
    request.prompt_token_count = token_count;
    request.output_token_budget = 1u;
    request.max_prefill_tokens_per_step = max_prefill_tokens_per_step;
    request.request_id = 1u;
    request.sequence_id = 1u;
    request.prompt_token_ids = tokens;
    return SparkGlm52RequestApiSubmit(&dryrun->api, &request, handle_out);
}

static void SparkPrefillDryrunPrintPrefill(
    uint32_t step_index,
    const SparkGlm52RequestApiDispatch *dispatch,
    const SparkGlm52SchedulerDecision *decision,
    uint32_t block_count)
{
    printf("%u\tprefill\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n",
        step_index,
        decision->scheduled_prompt_token_offset,
        decision->scheduled_prompt_token_count,
        decision->remaining_prompt_token_count_after_step,
        decision->cache_commit_token_count_after_step,
        decision->prefill_block_count,
        block_count,
        dispatch->flags);
}

static SparkStatus SparkPrefillDryrunPrintDispatch(
    SparkPrefillDryrun *dryrun,
    uint32_t step_index,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    SparkStatus status;

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        memset(dryrun->physical_block_indices, 0,
            sizeof(dryrun->physical_block_indices));
        memset(dryrun->lane_physical_block_counts, 0,
            sizeof(dryrun->lane_physical_block_counts));
        status = SparkGlm52RequestApiBuildDispatchKvBlockTables(
            &dryrun->api,
            dispatch,
            dryrun->physical_block_indices,
            SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY,
            SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY,
            dryrun->lane_physical_block_counts,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkPrefillDryrunPrintPrefill(
            step_index,
            dispatch,
            &dispatch->prefill_decision,
            dryrun->lane_physical_block_counts[0u]);
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        printf("%u\tdecode_ready\t0\t0\t0\t0\t0\t0\t%u\n",
            step_index,
            dispatch->flags);
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkPrefillDryrunRun(
    SparkPrefillDryrun *dryrun)
{
    SparkGlm52RequestApiDispatch dispatch;
    SparkStatus status;
    uint32_t step_index;

    printf("step\tkind\ttoken_offset\ttoken_count\tremaining\tcommit_after\tprefill_blocks\tkv_blocks\tflags\n");
    for (step_index = 0u; step_index < SPARK_PREFILL_DRYRUN_MAX_STEPS; ++step_index)
    {
        memset(&dispatch, 0, sizeof(dispatch));
        status = SparkGlm52RequestApiScheduleNext(&dryrun->api, &dispatch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (dispatch.accepted == 0u)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        status = SparkPrefillDryrunPrintDispatch(
            dryrun,
            step_index,
            &dispatch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52RequestApiCompleteDispatch(
            &dryrun->api,
            &dispatch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
        {
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}

static int32_t SparkPrefillDryrunUsage(
    const char *program)
{
    fprintf(stderr,
        "usage: %s (--tokens path | --tokenizer-json path (--prompt text | --prompt-file path)) "
        "[--max-prefill-tokens n] [--write-tokens path] [--disable-special] [--add-prefix-space]\n",
        program);
    return 2;
}

int main(
    int argc,
    char **argv)
{
    const char *token_path;
    const char *tokenizer_json_path;
    const char *prompt_text;
    const char *prompt_file_path;
    const char *write_tokens_path;
    char *owned_prompt_text;
    uint32_t prompt_text_bytes;
    uint32_t token_count;
    uint32_t max_prefill_tokens_per_step;
    uint32_t encode_flags;
    SparkGlm52RequestApiHandle handle;
    SparkStatus status;
    int32_t argument_index;
    int32_t parse_status;

    token_path = 0;
    tokenizer_json_path = 0;
    prompt_text = 0;
    prompt_file_path = 0;
    write_tokens_path = 0;
    owned_prompt_text = 0;
    prompt_text_bytes = 0u;
    token_count = 0u;
    encode_flags = 0u;
    max_prefill_tokens_per_step =
        SPARK_GLM52_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP;

    for (argument_index = 1; argument_index < argc; ++argument_index)
    {
        if (strcmp(argv[argument_index], "--tokens") == 0 &&
            argument_index + 1 < argc)
        {
            token_path = argv[++argument_index];
            continue;
        }
        if (strcmp(argv[argument_index], "--tokenizer-json") == 0 &&
            argument_index + 1 < argc)
        {
            tokenizer_json_path = argv[++argument_index];
            continue;
        }
        if (strcmp(argv[argument_index], "--prompt") == 0 &&
            argument_index + 1 < argc)
        {
            prompt_text = argv[++argument_index];
            prompt_text_bytes = (uint32_t)strlen(prompt_text);
            continue;
        }
        if (strcmp(argv[argument_index], "--prompt-file") == 0 &&
            argument_index + 1 < argc)
        {
            prompt_file_path = argv[++argument_index];
            continue;
        }
        if (strcmp(argv[argument_index], "--write-tokens") == 0 &&
            argument_index + 1 < argc)
        {
            write_tokens_path = argv[++argument_index];
            continue;
        }
        if (strcmp(argv[argument_index], "--max-prefill-tokens") == 0 &&
            argument_index + 1 < argc)
        {
            parse_status = SparkPrefillDryrunParseU32(
                argv[++argument_index],
                &max_prefill_tokens_per_step);
            if (parse_status < 0)
            {
                return SparkPrefillDryrunUsage(argv[0]);
            }
            continue;
        }
        if (strcmp(argv[argument_index], "--disable-special") == 0)
        {
            encode_flags |= SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH;
            continue;
        }
        if (strcmp(argv[argument_index], "--add-prefix-space") == 0)
        {
            encode_flags |= SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE;
            continue;
        }
        return SparkPrefillDryrunUsage(argv[0]);
    }

    if ((token_path == 0 && tokenizer_json_path == 0) ||
        (token_path != 0 && tokenizer_json_path != 0) ||
        (tokenizer_json_path != 0 && ((prompt_text == 0) == (prompt_file_path == 0))))
    {
        return SparkPrefillDryrunUsage(argv[0]);
    }

    if (token_path != 0)
    {
        parse_status = SparkPrefillDryrunReadTokenFile(
            token_path,
            PromptTokens,
            SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS,
            &token_count);
        if (parse_status < 0)
        {
            fprintf(stderr, "failed to read token file: %d\n", parse_status);
            return 1;
        }
    }
    else
    {
        if (prompt_file_path != 0)
        {
            owned_prompt_text = SparkToolReadWholeFile(
                prompt_file_path,
                &prompt_text_bytes);
            if (owned_prompt_text == 0)
            {
                fprintf(stderr, "failed to read prompt file\n");
                return 1;
            }
            prompt_text = owned_prompt_text;
        }
        parse_status = SparkPrefillDryrunTokenizePrompt(
            tokenizer_json_path,
            prompt_text,
            prompt_text_bytes,
            encode_flags,
            PromptTokens,
            SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS,
            &token_count);
        if (parse_status < 0)
        {
            free(owned_prompt_text);
            fprintf(stderr, "failed to tokenize prompt: %d\n", parse_status);
            return 1;
        }
    }

    if (write_tokens_path != 0 &&
        SparkPrefillDryrunWriteTokenFile(
            write_tokens_path,
            PromptTokens,
            token_count) != 0)
    {
        free(owned_prompt_text);
        fprintf(stderr, "failed to write token file\n");
        return 1;
    }

    status = SparkPrefillDryrunInitialize(
        &Dryrun,
        max_prefill_tokens_per_step);
    if (status != SPARK_STATUS_OK)
    {
        free(owned_prompt_text);
        fprintf(stderr, "initialize failed: %s\n", SparkStatusToString(status));
        return 1;
    }
    status = SparkPrefillDryrunSubmit(
        &Dryrun,
        PromptTokens,
        token_count,
        max_prefill_tokens_per_step,
        &handle);
    if (status != SPARK_STATUS_OK || handle == 0u)
    {
        free(owned_prompt_text);
        fprintf(stderr, "submit failed: %s\n", SparkStatusToString(status));
        return 1;
    }
    status = SparkPrefillDryrunRun(&Dryrun);
    free(owned_prompt_text);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "dryrun failed: %s\n", SparkStatusToString(status));
        return 1;
    }
    return 0;
}
