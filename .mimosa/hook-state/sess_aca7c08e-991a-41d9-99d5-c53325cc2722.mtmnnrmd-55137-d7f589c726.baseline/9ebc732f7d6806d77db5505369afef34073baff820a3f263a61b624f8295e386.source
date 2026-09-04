#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_hardware_topology_probe.h"

typedef struct SparkTopologyOptions
{
    const char *provider_path;
    const char *question_id;
    const char *source_package_sha256;
    const char *run_id;
    const char *topology;
    const char *node;
    const char *model_id;
    const char *candidate;
    const char *output_path;
    uint32_t batch_size;
    uint32_t context_tokens;
    uint32_t pipeline_degree;
    uint32_t window_depth;
    uint32_t iterations;
} SparkTopologyOptions;

static int SparkTopologyHexIsValid(const char *text)
{
    uint32_t index;

    if (text == 0 || strlen(text) != 64u)
    {
        return 0;
    }
    for (index = 0u; index < 64u; ++index)
    {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f')))
        {
            return 0;
        }
    }
    return 1;
}

static int SparkTopologyIdentifierIsValid(const char *text)
{
    const unsigned char *cursor;

    if (text == 0 || text[0] == '\0' ||
        strlen(text) >= SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES)
    {
        return 0;
    }
    cursor = (const unsigned char *)text;
    while (*cursor != 0u)
    {
        if (!(isalnum(*cursor) || *cursor == '_' || *cursor == '-' ||
              *cursor == '.' || *cursor == ':' || *cursor == '/'))
        {
            return 0;
        }
        cursor += 1u;
    }
    return 1;
}

static int SparkTopologyParseU32(
    const char *text,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t *value_out)
{
    char *end;
    unsigned long value;

    if (text == 0 || value_out == 0 || text[0] == '\0')
    {
        return 0;
    }
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < minimum || value > maximum)
    {
        return 0;
    }
    *value_out = (uint32_t)value;
    return 1;
}

static int SparkTopologyQuestionIsValid(const char *question_id)
{
    return strcmp(question_id, "NET-RING-001") == 0 ||
        strcmp(question_id, "NET-SWITCH-001") == 0 ||
        strcmp(question_id, "TOPO-PP-001") == 0 ||
        strcmp(question_id, "TOPO-WINDOW-001") == 0 ||
        strcmp(question_id, "TOPO-PLACEMENT-001") == 0;
}

static int SparkTopologyReservedU64IsZero(const uint64_t *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
    {
        if (values[index] != 0u)
        {
            return 0;
        }
    }
    return 1;
}

static int SparkTopologyReservedU32IsZero(const uint32_t *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
    {
        if (values[index] != 0u)
        {
            return 0;
        }
    }
    return 1;
}

static void SparkTopologyCopyIdentifier(
    char destination[SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES],
    const char *source)
{
    (void)snprintf(destination, SPARK_HARDWARE_TOPOLOGY_PROBE_MAX_IDENTIFIER_BYTES,
        "%s", source == 0 ? "" : source);
}

static void SparkTopologyWriteJsonString(FILE *output, const char *text)
{
    const unsigned char *cursor;
    fputc('"', output);
    cursor = (const unsigned char *)(text == 0 ? "" : text);
    while (*cursor != 0u)
    {
        if (*cursor == '"' || *cursor == '\\')
        {
            fputc('\\', output);
            fputc((int)*cursor, output);
        }
        else if (*cursor == '\n')
        {
            fputs("\\n", output);
        }
        else if (*cursor < 0x20u)
        {
            fprintf(output, "\\u%04x", (unsigned int)*cursor);
        }
        else
        {
            fputc((int)*cursor, output);
        }
        cursor += 1u;
    }
    fputc('"', output);
}

static const char *SparkTopologyStatusName(SparkStatus status)
{
    if (status == SPARK_STATUS_OK)
    {
        return "measured";
    }
    if (status == SPARK_STATUS_UNSUPPORTED || status == SPARK_STATUS_NOT_FOUND)
    {
        return "unsupported";
    }
    return "failed";
}

static int SparkTopologyResultIsValid(
    SparkStatus run_status,
    const SparkHardwareTopologyProbeResult *result,
    const SparkTopologyOptions *options)
{
    const uint32_t required_flags =
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_EXACT_PRODUCTION_PATH |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_ALL_RANKS_PARTICIPATED |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_GLOBAL_COMMIT_VERIFIED |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_FINAL_EVENT_ACK_VERIFIED |
        SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_INTEGRITY_VERIFIED;

    if (result->abi_version != SPARK_HARDWARE_TOPOLOGY_PROBE_ABI_VERSION ||
        result->descriptor_bytes != SPARK_HARDWARE_TOPOLOGY_PROBE_RESULT_BYTES ||
        result->status != run_status ||
        (result->result_flags & ~SPARK_HARDWARE_TOPOLOGY_RESULT_KNOWN_FLAGS) != 0u ||
        !SparkTopologyReservedU32IsZero(result->reserved_u32,
            sizeof(result->reserved_u32) / sizeof(result->reserved_u32[0])))
    {
        return 0;
    }
    if (run_status != SPARK_STATUS_OK)
    {
        return result->result_flags == 0u;
    }
    if ((result->result_flags & required_flags) != required_flags ||
        !SparkTopologyHexIsValid(result->topology_artifact_sha256) ||
        result->sample_count < options->iterations ||
        result->latency_min_ns == 0u || result->latency_p50_ns == 0u ||
        result->latency_min_ns > result->latency_p50_ns ||
        result->latency_p50_ns > result->latency_p95_ns ||
        result->latency_p95_ns > result->latency_p99_ns ||
        result->latency_p99_ns > result->latency_max_ns ||
        result->rank_count < options->pipeline_degree ||
        result->completion_count < result->sample_count ||
        result->numerical_pass == 0u || result->integrity_pass == 0u ||
        result->integrity_error_count != 0u ||
        !isfinite(result->tokens_per_second) || result->tokens_per_second <= 0.0 ||
        !isfinite(result->rows_per_second) || result->rows_per_second <= 0.0 ||
        !isfinite(result->network_gb_s) || result->network_gb_s < 0.0 ||
        !isfinite(result->cpu_percent) || result->cpu_percent < 0.0)
    {
        return 0;
    }
    return 1;
}

static int SparkTopologyWriteReceipt(
    const SparkTopologyOptions *options,
    const SparkHardwareTopologyProbeInterface *provider,
    const SparkHardwareTopologyProbeResult *result)
{
    FILE *output;

    output = options->output_path == 0 ? stdout : fopen(options->output_path, "wb");
    if (output == 0)
    {
        return 0;
    }
    fputs("{\n  \"schema_version\": 1,\n  \"receipt_kind\": \"spark_hardware_probe\",\n  \"run_id\": ", output);
    SparkTopologyWriteJsonString(output, options->run_id);
    fputs(",\n  \"probe_id\": \"topology_characterize\",\n  \"source_identity\": {\"source_package_sha256\": ", output);
    SparkTopologyWriteJsonString(output, options->source_package_sha256);
    fputs("},\n  \"scope\": {\"topology\": ", output);
    SparkTopologyWriteJsonString(output, options->topology);
    fputs(", \"node\": ", output);
    SparkTopologyWriteJsonString(output, options->node);
    fputs("},\n  \"answers\": [\n    {\"question_id\": ", output);
    SparkTopologyWriteJsonString(output, options->question_id);
    fputs(", \"status\": ", output);
    SparkTopologyWriteJsonString(output, SparkTopologyStatusName(result->status));
    fputs(", \"summary\": {\"provider_id\": ", output);
    SparkTopologyWriteJsonString(output, provider->provider_id);
    fputs(", \"provider_build_identity\": ", output);
    SparkTopologyWriteJsonString(output, provider->provider_build_identity);
    fputs(", \"topology_artifact_sha256\": ", output);
    SparkTopologyWriteJsonString(output, result->topology_artifact_sha256);
    fprintf(output,
        ", \"exact_production_path\": %s, \"all_ranks_participated\": %s, "
        "\"global_commit_verified\": %s, \"final_event_ack_verified\": %s, "
        "\"integrity_verified\": %s}, \"observations\": [",
        (result->result_flags & SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_EXACT_PRODUCTION_PATH) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_ALL_RANKS_PARTICIPATED) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_GLOBAL_COMMIT_VERIFIED) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_FINAL_EVENT_ACK_VERIFIED) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TOPOLOGY_RESULT_FLAG_INTEGRITY_VERIFIED) != 0u ? "true" : "false");
    if (result->status == SPARK_STATUS_OK)
    {
        fputs("{\"parameters\": {\"model_id\": ", output);
        SparkTopologyWriteJsonString(output, options->model_id);
        fputs(", \"candidate\": ", output);
        SparkTopologyWriteJsonString(output, options->candidate);
        fprintf(output,
            ", \"batch_size\": %u, \"context_tokens\": %u, \"pipeline_degree\": %u, "
            "\"window_depth\": %u}, \"metrics\": {\"sample_count\": %" PRIu64 ", "
            "\"latency_min_ns\": %" PRIu64 ", \"latency_p50_ns\": %" PRIu64 ", "
            "\"latency_p95_ns\": %" PRIu64 ", \"latency_p99_ns\": %" PRIu64 ", "
            "\"latency_max_ns\": %" PRIu64 ", \"tokens_per_second\": %.17g, "
            "\"rows_per_second\": %.17g, \"network_gb_s\": %.17g, "
            "\"cpu_percent\": %.17g, \"retry_count\": %" PRIu64 ", "
            "\"duplicate_count\": %" PRIu64 ", \"integrity_error_count\": %" PRIu64 ", "
            "\"rank_count\": %u, \"completion_count\": %u, "
            "\"numerical_pass\": true, \"integrity_pass\": true}}",
            options->batch_size, options->context_tokens, options->pipeline_degree,
            options->window_depth, result->sample_count, result->latency_min_ns,
            result->latency_p50_ns, result->latency_p95_ns, result->latency_p99_ns,
            result->latency_max_ns, result->tokens_per_second, result->rows_per_second,
            result->network_gb_s, result->cpu_percent, result->retry_count,
            result->duplicate_count, result->integrity_error_count, result->rank_count,
            result->completion_count);
    }
    fputs("]", output);
    if (result->status != SPARK_STATUS_OK)
    {
        fprintf(output, ", \"error\": \"provider returned SparkStatus %d\"", result->status);
    }
    fputs("}\n  ]\n}\n", output);
    if (output != stdout)
    {
        fclose(output);
    }
    return 1;
}

static int SparkTopologyParseOptions(int argument_count, char **arguments, SparkTopologyOptions *options)
{
    int index;

    memset(options, 0, sizeof(*options));
    options->iterations = 32u;
    options->window_depth = 1u;
    options->pipeline_degree = 1u;
    options->context_tokens = 2048u;
    for (index = 1; index < argument_count; ++index)
    {
        const char *argument = arguments[index];
        const char **text_destination = 0;
        if (strcmp(argument, "--provider") == 0) text_destination = &options->provider_path;
        else if (strcmp(argument, "--question") == 0) text_destination = &options->question_id;
        else if (strcmp(argument, "--source-package-sha256") == 0) text_destination = &options->source_package_sha256;
        else if (strcmp(argument, "--run-id") == 0) text_destination = &options->run_id;
        else if (strcmp(argument, "--topology") == 0) text_destination = &options->topology;
        else if (strcmp(argument, "--node") == 0) text_destination = &options->node;
        else if (strcmp(argument, "--model") == 0) text_destination = &options->model_id;
        else if (strcmp(argument, "--candidate") == 0) text_destination = &options->candidate;
        else if (strcmp(argument, "--output") == 0) text_destination = &options->output_path;
        if (text_destination != 0)
        {
            if (++index >= argument_count) return 0;
            *text_destination = arguments[index];
            continue;
        }
#define SPARK_TOPOLOGY_PARSE(NAME, MINIMUM, MAXIMUM, FIELD) \
        if (strcmp(argument, NAME) == 0) \
        { \
            if (++index >= argument_count || !SparkTopologyParseU32(arguments[index], MINIMUM, MAXIMUM, &options->FIELD)) return 0; \
        }
        SPARK_TOPOLOGY_PARSE("--batch", 1u, 1048576u, batch_size)
        else SPARK_TOPOLOGY_PARSE("--context", 1u, 1048576u, context_tokens)
        else SPARK_TOPOLOGY_PARSE("--pipeline-degree", 1u, 1024u, pipeline_degree)
        else SPARK_TOPOLOGY_PARSE("--window-depth", 1u, 65536u, window_depth)
        else SPARK_TOPOLOGY_PARSE("--iterations", 1u, 1000000u, iterations)
        else return 0;
#undef SPARK_TOPOLOGY_PARSE
    }
    return options->provider_path != 0 && SparkTopologyQuestionIsValid(options->question_id) &&
        SparkTopologyHexIsValid(options->source_package_sha256) &&
        SparkTopologyIdentifierIsValid(options->run_id) &&
        SparkTopologyIdentifierIsValid(options->topology) &&
        SparkTopologyIdentifierIsValid(options->node) &&
        SparkTopologyIdentifierIsValid(options->model_id) &&
        SparkTopologyIdentifierIsValid(options->candidate) && options->batch_size != 0u;
}

static void SparkTopologyUsage(const char *program_name)
{
    fprintf(stderr,
        "usage: %s --provider LIB --question ID --source-package-sha256 HASH "
        "--run-id ID --topology NAME --node NAME --model ID --candidate ID "
        "--batch N [--context N] [--pipeline-degree N] [--window-depth N] "
        "[--iterations N] [--output FILE]\n", program_name);
}

int main(int argument_count, char **arguments)
{
    SparkTopologyOptions options;
    SparkHardwareTopologyProbeRequest request;
    SparkHardwareTopologyProbeResult result;
    SparkHardwareTopologyProbeGetInterfaceFunction get_interface;
    const SparkHardwareTopologyProbeInterface *provider;
    uint32_t required_flags;
    SparkStatus status;
    void *library;
    void *symbol;
    const char *symbol_error;
    int write_ok;

    if (!SparkTopologyParseOptions(argument_count, arguments, &options))
    {
        SparkTopologyUsage(arguments[0]);
        return 2;
    }
    library = dlopen(options.provider_path, RTLD_NOW | RTLD_LOCAL);
    if (library == 0)
    {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    dlerror();
    symbol = dlsym(library, SPARK_HARDWARE_TOPOLOGY_PROBE_INTERFACE_SYMBOL);
    symbol_error = dlerror();
    if (symbol == 0 || symbol_error != 0)
    {
        fprintf(stderr, "provider lacks %s: %s\n", SPARK_HARDWARE_TOPOLOGY_PROBE_INTERFACE_SYMBOL,
            symbol_error == 0 ? "not found" : symbol_error);
        dlclose(library);
        return 1;
    }
    memcpy(&get_interface, &symbol, sizeof(get_interface));
    provider = get_interface();
    required_flags = SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_EXACT_PRODUCTION_PATH |
        SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_ALL_RANK_COMMIT |
        SPARK_HARDWARE_TOPOLOGY_PROVIDER_FLAG_FINAL_EVENT_ACK;
    if (provider == 0 || provider->abi_version != SPARK_HARDWARE_TOPOLOGY_PROBE_ABI_VERSION ||
        provider->descriptor_bytes != SPARK_HARDWARE_TOPOLOGY_PROBE_INTERFACE_BYTES ||
        (provider->capability_flags & ~SPARK_HARDWARE_TOPOLOGY_PROVIDER_KNOWN_FLAGS) != 0u ||
        (provider->capability_flags & required_flags) != required_flags ||
        provider->reserved_u32 != 0u ||
        !SparkTopologyReservedU64IsZero(provider->reserved_u64,
            sizeof(provider->reserved_u64) / sizeof(provider->reserved_u64[0])) ||
        !SparkTopologyIdentifierIsValid(provider->provider_id) ||
        !SparkTopologyHexIsValid(provider->provider_build_identity) || provider->run == 0)
    {
        fprintf(stderr, "invalid hardware topology provider interface\n");
        dlclose(library);
        return 1;
    }
    memset(&request, 0, sizeof(request));
    request.abi_version = SPARK_HARDWARE_TOPOLOGY_PROBE_ABI_VERSION;
    request.descriptor_bytes = SPARK_HARDWARE_TOPOLOGY_PROBE_REQUEST_BYTES;
    request.batch_size = options.batch_size;
    request.context_tokens = options.context_tokens;
    request.pipeline_degree = options.pipeline_degree;
    request.window_depth = options.window_depth;
    request.iterations = options.iterations;
    SparkTopologyCopyIdentifier(request.question_id, options.question_id);
    SparkTopologyCopyIdentifier(request.topology, options.topology);
    SparkTopologyCopyIdentifier(request.node, options.node);
    SparkTopologyCopyIdentifier(request.model_id, options.model_id);
    SparkTopologyCopyIdentifier(request.candidate, options.candidate);
    memset(&result, 0, sizeof(result));
    result.abi_version = SPARK_HARDWARE_TOPOLOGY_PROBE_ABI_VERSION;
    result.descriptor_bytes = SPARK_HARDWARE_TOPOLOGY_PROBE_RESULT_BYTES;
    status = provider->run(&request, &result);
    if (!SparkTopologyResultIsValid(status, &result, &options))
    {
        fprintf(stderr, "provider failed to prove exact production topology semantics\n");
        dlclose(library);
        return 1;
    }
    write_ok = SparkTopologyWriteReceipt(&options, provider, &result);
    dlclose(library);
    return write_ok ? 0 : 1;
}
