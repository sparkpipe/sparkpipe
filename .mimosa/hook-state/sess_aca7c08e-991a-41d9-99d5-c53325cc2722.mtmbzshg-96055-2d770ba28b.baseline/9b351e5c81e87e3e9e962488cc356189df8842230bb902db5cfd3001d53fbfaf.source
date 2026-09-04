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

#include "sparkpipe/spark_hardware_transport_probe.h"

#define SPARK_TRANSPORT_DEFAULT_ITERATIONS 256u

typedef struct SparkTransportOptions
{
    const char *provider_path;
    const char *question_id;
    const char *source_package_sha256;
    const char *run_id;
    const char *topology;
    const char *node;
    const char *peer;
    const char *candidate;
    const char *progress_mode;
    const char *output_path;
    uint32_t payload_bytes;
    uint32_t lane_count;
    uint32_t window_depth;
    uint32_t cq_batch;
    uint32_t registered_region_count;
    uint32_t iterations;
    uint32_t local_rank;
    uint32_t peer_rank;
} SparkTransportOptions;

static int SparkTransportHexIsValid(const char *text)
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

static int SparkTransportIdentifierIsValid(const char *text)
{
    const unsigned char *cursor;

    if (text == 0 || text[0] == '\0' ||
        strlen(text) >= SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES)
    {
        return 0;
    }
    cursor = (const unsigned char *)text;
    while (*cursor != 0u)
    {
        if (!(isalnum(*cursor) || *cursor == '_' || *cursor == '-' ||
              *cursor == '.' || *cursor == ':'))
        {
            return 0;
        }
        cursor += 1u;
    }
    return 1;
}

static int SparkTransportParseU32(
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
    end = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < minimum || value > maximum)
    {
        return 0;
    }
    *value_out = (uint32_t)value;
    return 1;
}

static int SparkTransportQuestionIsSupported(const char *question_id)
{
    return strcmp(question_id, "NET-TCP-001") == 0 ||
        strcmp(question_id, "NET-RDMA-001") == 0 ||
        strcmp(question_id, "NET-MR-001") == 0 ||
        strcmp(question_id, "NET-LANES-001") == 0 ||
        strcmp(question_id, "NET-CQ-001") == 0 ||
        strcmp(question_id, "NET-PROGRESS-001") == 0;
}

static int SparkTransportCandidateIsValidForQuestion(
    const char *question_id,
    const char *candidate)
{
    if (strcmp(question_id, "NET-TCP-001") == 0)
    {
        return strcmp(candidate, "tcp") == 0;
    }
    return strcmp(candidate, "mapped_host") == 0 ||
        strcmp(candidate, "gpudirect") == 0;
}

static int SparkTransportProgressModeIsValid(const char *progress_mode)
{
    return strcmp(progress_mode, "event_loop") == 0 ||
        strcmp(progress_mode, "autonomous") == 0;
}

static int SparkTransportU64Multiply(
    uint64_t left,
    uint64_t right,
    uint64_t *product_out)
{
    if (product_out == 0 || (left != 0u && right > UINT64_MAX / left))
    {
        return 0;
    }
    *product_out = left * right;
    return 1;
}

static int SparkTransportRequiresPeer(const char *question_id)
{
    return SparkTransportQuestionIsSupported(question_id);
}

static void SparkTransportCopyIdentifier(
    char destination[SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES],
    const char *source)
{
    (void)snprintf(
        destination,
        SPARK_HARDWARE_TRANSPORT_PROBE_MAX_IDENTIFIER_BYTES,
        "%s",
        source == 0 ? "" : source);
}

static int SparkTransportReservedU32IsZero(const uint32_t *values, size_t count)
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

static int SparkTransportReservedU64IsZero(const uint64_t *values, size_t count)
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

static void SparkTransportWriteJsonString(FILE *output, const char *text)
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

static void SparkTransportWriteParameters(
    FILE *output,
    const SparkTransportOptions *options)
{
    int comma;

    comma = 0;
#define SPARK_TRANSPORT_PARAMETER_SEPARATOR() \
    do \
    { \
        if (comma != 0) \
        { \
            fputs(", ", output); \
        } \
        comma = 1; \
    } while (0)
    if (options->payload_bytes != 0u)
    {
        SPARK_TRANSPORT_PARAMETER_SEPARATOR();
        fprintf(output, "\"payload_bytes\": %u", options->payload_bytes);
    }
    if (options->lane_count != 0u)
    {
        SPARK_TRANSPORT_PARAMETER_SEPARATOR();
        fprintf(output, "\"lane_count\": %u", options->lane_count);
    }
    if (options->window_depth != 0u)
    {
        SPARK_TRANSPORT_PARAMETER_SEPARATOR();
        fprintf(output, "\"window_depth\": %u", options->window_depth);
    }
    if (options->cq_batch != 0u)
    {
        SPARK_TRANSPORT_PARAMETER_SEPARATOR();
        fprintf(output, "\"cq_batch\": %u", options->cq_batch);
    }
    if (options->registered_region_count != 0u)
    {
        SPARK_TRANSPORT_PARAMETER_SEPARATOR();
        fprintf(output, "\"registered_region_count\": %u", options->registered_region_count);
    }
    if (options->progress_mode != 0)
    {
        SPARK_TRANSPORT_PARAMETER_SEPARATOR();
        fputs("\"progress_mode\": ", output);
        SparkTransportWriteJsonString(output, options->progress_mode);
    }
    SPARK_TRANSPORT_PARAMETER_SEPARATOR();
    fputs("\"candidate\": ", output);
    SparkTransportWriteJsonString(output, options->candidate);
    SPARK_TRANSPORT_PARAMETER_SEPARATOR();
    fprintf(output, "\"iterations\": %u", options->iterations);
#undef SPARK_TRANSPORT_PARAMETER_SEPARATOR
}

static int SparkTransportResultIsValid(
    const SparkTransportOptions *options,
    SparkStatus run_status,
    const SparkHardwareTransportProbeResult *result)
{
    uint32_t required_flags;
    uint32_t forbidden_flags;
    uint64_t minimum_transferred_bytes;

    if (result->abi_version != SPARK_HARDWARE_TRANSPORT_PROBE_ABI_VERSION ||
        result->descriptor_bytes != SPARK_HARDWARE_TRANSPORT_PROBE_RESULT_BYTES ||
        result->status != run_status ||
        (result->result_flags & ~SPARK_HARDWARE_TRANSPORT_RESULT_KNOWN_FLAGS) != 0u ||
        !SparkTransportReservedU32IsZero(
            result->reserved_u32,
            sizeof(result->reserved_u32) / sizeof(result->reserved_u32[0])))
    {
        return 0;
    }
    if (run_status != SPARK_STATUS_OK)
    {
        return result->result_flags == 0u;
    }
    if (!SparkTransportHexIsValid(result->local_transport_artifact_sha256) ||
        !SparkTransportHexIsValid(result->peer_transport_artifact_sha256) ||
        strcmp(
            result->local_transport_artifact_sha256,
            result->peer_transport_artifact_sha256) != 0)
    {
        return 0;
    }
    required_flags =
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_INTEGRITY_VERIFIED |
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REMOTE_COMPLETION_VERIFIED |
        SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_EXACT_PRODUCTION_TRANSPORT;
    forbidden_flags = 0u;
    if (strcmp(options->candidate, "tcp") == 0)
    {
        required_flags |= SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_TCP_PATH;
        forbidden_flags |=
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_VERBS_PATH |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_MAPPED_HOST_DIRECT_MR |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_NO_CPU_STAGING_COPY |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_DEVICE_DIRECT_MR;
    }
    else if (strcmp(options->candidate, "mapped_host") == 0)
    {
        required_flags |=
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_VERBS_PATH |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_MAPPED_HOST_DIRECT_MR |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_NO_CPU_STAGING_COPY;
        forbidden_flags |=
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_TCP_PATH |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_DEVICE_DIRECT_MR;
    }
    else if (strcmp(options->candidate, "gpudirect") == 0)
    {
        required_flags |=
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_VERBS_PATH |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_NO_CPU_STAGING_COPY |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_DEVICE_DIRECT_MR;
        forbidden_flags |=
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_TCP_PATH |
            SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_MAPPED_HOST_DIRECT_MR;
    }
    else
    {
        return 0;
    }
    if ((result->result_flags & required_flags) != required_flags ||
        (result->result_flags & forbidden_flags) != 0u ||
        result->sample_count < options->iterations ||
        result->numerical_pass == 0u ||
        result->integrity_pass == 0u ||
        result->integrity_error_count != 0u ||
        !isfinite(result->throughput_gb_s) || result->throughput_gb_s < 0.0 ||
        !isfinite(result->message_rate_per_second) || result->message_rate_per_second < 0.0 ||
        !isfinite(result->cpu_percent) || result->cpu_percent < 0.0 ||
        result->source_fingerprint == 0u ||
        result->source_fingerprint != result->destination_fingerprint)
    {
        return 0;
    }
    if (strcmp(options->question_id, "NET-MR-001") == 0)
    {
        return result->mr_registration_count != 0u &&
            result->registration_time_ns != 0u;
    }
    if (!SparkTransportU64Multiply(
            options->payload_bytes,
            result->sample_count,
            &minimum_transferred_bytes) ||
        result->transferred_bytes < minimum_transferred_bytes ||
        result->latency_min_ns == 0u ||
        result->latency_p50_ns == 0u ||
        result->latency_p50_ns > result->latency_p95_ns ||
        result->latency_p95_ns > result->latency_p99_ns ||
        result->latency_p99_ns > result->latency_max_ns ||
        result->latency_min_ns > result->latency_p50_ns ||
        result->throughput_gb_s <= 0.0 ||
        result->message_rate_per_second <= 0.0)
    {
        return 0;
    }
    return 1;
}

static const char *SparkTransportStatusName(SparkStatus status)
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

static int SparkTransportWriteReceipt(
    const SparkTransportOptions *options,
    const SparkHardwareTransportProbeInterface *provider,
    const SparkHardwareTransportProbeResult *result)
{
    FILE *output;

    output = options->output_path == 0 ? stdout : fopen(options->output_path, "wb");
    if (output == 0)
    {
        return 0;
    }
    fputs("{\n  \"schema_version\": 1,\n  \"receipt_kind\": \"spark_hardware_probe\",\n  \"run_id\": ", output);
    SparkTransportWriteJsonString(output, options->run_id);
    fputs(",\n  \"probe_id\": \"transport_characterize\",\n  \"source_identity\": {\"source_package_sha256\": ", output);
    SparkTransportWriteJsonString(output, options->source_package_sha256);
    fputs("},\n  \"scope\": {\"topology\": ", output);
    SparkTransportWriteJsonString(output, options->topology);
    fputs(", \"node\": ", output);
    SparkTransportWriteJsonString(output, options->node);
    if (options->peer != 0)
    {
        fputs(", \"peer\": ", output);
        SparkTransportWriteJsonString(output, options->peer);
    }
    fputs("},\n  \"answers\": [\n    {\"question_id\": ", output);
    SparkTransportWriteJsonString(output, options->question_id);
    fputs(", \"status\": ", output);
    SparkTransportWriteJsonString(output, SparkTransportStatusName(result->status));
    fputs(", \"summary\": {\"provider_id\": ", output);
    SparkTransportWriteJsonString(output, provider->provider_id);
    fputs(", \"provider_build_identity\": ", output);
    SparkTransportWriteJsonString(output, provider->provider_build_identity);
    fputs(", \"local_transport_artifact_sha256\": ", output);
    SparkTransportWriteJsonString(output, result->local_transport_artifact_sha256);
    fputs(", \"peer_transport_artifact_sha256\": ", output);
    SparkTransportWriteJsonString(output, result->peer_transport_artifact_sha256);
    fprintf(output,
        ", \"real_verbs_path\": %s, \"real_tcp_path\": %s, \"integrity_verified\": %s, "
        "\"remote_completion_verified\": %s, \"mapped_host_direct_mr\": %s, "
        "\"no_cpu_staging_copy\": %s, \"device_direct_mr\": %s, "
        "\"exact_production_transport\": %s}, \"observations\": [",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_VERBS_PATH) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REAL_TCP_PATH) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_INTEGRITY_VERIFIED) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_REMOTE_COMPLETION_VERIFIED) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_MAPPED_HOST_DIRECT_MR) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_NO_CPU_STAGING_COPY) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_DEVICE_DIRECT_MR) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_TRANSPORT_RESULT_FLAG_EXACT_PRODUCTION_TRANSPORT) != 0u ? "true" : "false");
    if (result->status == SPARK_STATUS_OK)
    {
        fputs("{\"parameters\": {", output);
        SparkTransportWriteParameters(output, options);
        fprintf(output,
            "}, \"metrics\": {\"sample_count\": %" PRIu64 ", "
            "\"transferred_bytes\": %" PRIu64 ", \"latency_min_ns\": %" PRIu64 ", "
            "\"latency_p50_ns\": %" PRIu64 ", \"latency_p95_ns\": %" PRIu64 ", "
            "\"latency_p99_ns\": %" PRIu64 ", \"latency_max_ns\": %" PRIu64 ", "
            "\"throughput_gb_s\": %.17g, \"message_rate_per_second\": %.17g, "
            "\"cpu_percent\": %.17g, \"setup_time_ns\": %" PRIu64 ", "
            "\"registration_time_ns\": %" PRIu64 ", \"deregistration_time_ns\": %" PRIu64 ", "
            "\"cq_poll_count\": %" PRIu64 ", \"cq_wakeup_count\": %" PRIu64 ", "
            "\"mr_registration_count\": %" PRIu64 ", \"mr_cache_hit_count\": %" PRIu64 ", "
            "\"mr_eviction_count\": %" PRIu64 ", \"retry_count\": %" PRIu64 ", "
            "\"integrity_error_count\": %" PRIu64 ", \"numerical_pass\": true, "
            "\"integrity_pass\": true}}",
            result->sample_count,
            result->transferred_bytes,
            result->latency_min_ns,
            result->latency_p50_ns,
            result->latency_p95_ns,
            result->latency_p99_ns,
            result->latency_max_ns,
            result->throughput_gb_s,
            result->message_rate_per_second,
            result->cpu_percent,
            result->setup_time_ns,
            result->registration_time_ns,
            result->deregistration_time_ns,
            result->cq_poll_count,
            result->cq_wakeup_count,
            result->mr_registration_count,
            result->mr_cache_hit_count,
            result->mr_eviction_count,
            result->retry_count,
            result->integrity_error_count);
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

static int SparkTransportParseOptions(
    int argument_count,
    char **arguments,
    SparkTransportOptions *options)
{
    int index;

    memset(options, 0, sizeof(*options));
    options->progress_mode = "event_loop";
    options->cq_batch = 1u;
    options->registered_region_count = 128u;
    options->iterations = SPARK_TRANSPORT_DEFAULT_ITERATIONS;
    for (index = 1; index < argument_count; ++index)
    {
        const char *argument;
        const char **text_destination;

        argument = arguments[index];
        text_destination = 0;
        if (strcmp(argument, "--provider") == 0)
        {
            text_destination = &options->provider_path;
        }
        else if (strcmp(argument, "--question") == 0)
        {
            text_destination = &options->question_id;
        }
        else if (strcmp(argument, "--source-package-sha256") == 0)
        {
            text_destination = &options->source_package_sha256;
        }
        else if (strcmp(argument, "--run-id") == 0)
        {
            text_destination = &options->run_id;
        }
        else if (strcmp(argument, "--topology") == 0)
        {
            text_destination = &options->topology;
        }
        else if (strcmp(argument, "--node") == 0)
        {
            text_destination = &options->node;
        }
        else if (strcmp(argument, "--peer") == 0)
        {
            text_destination = &options->peer;
        }
        else if (strcmp(argument, "--candidate") == 0)
        {
            text_destination = &options->candidate;
        }
        else if (strcmp(argument, "--progress-mode") == 0)
        {
            text_destination = &options->progress_mode;
        }
        else if (strcmp(argument, "--output") == 0)
        {
            text_destination = &options->output_path;
        }
        if (text_destination != 0)
        {
            if (index + 1 >= argument_count)
            {
                return 0;
            }
            *text_destination = arguments[++index];
            continue;
        }
#define SPARK_TRANSPORT_PARSE_OPTION(NAME, MINIMUM, MAXIMUM, FIELD) \
        if (strcmp(argument, NAME) == 0) \
        { \
            if (index + 1 >= argument_count || \
                !SparkTransportParseU32(arguments[++index], MINIMUM, MAXIMUM, &options->FIELD)) \
            { \
                return 0; \
            } \
        }
        SPARK_TRANSPORT_PARSE_OPTION("--payload-bytes", 1u, 64u * 1024u * 1024u, payload_bytes)
        else SPARK_TRANSPORT_PARSE_OPTION("--lane-count", 1u, 64u, lane_count)
        else SPARK_TRANSPORT_PARSE_OPTION("--window-depth", 1u, 4096u, window_depth)
        else SPARK_TRANSPORT_PARSE_OPTION("--cq-batch", 1u, 4096u, cq_batch)
        else SPARK_TRANSPORT_PARSE_OPTION("--registered-region-count", 1u, 65536u, registered_region_count)
        else SPARK_TRANSPORT_PARSE_OPTION("--iterations", 1u, 1000000u, iterations)
        else SPARK_TRANSPORT_PARSE_OPTION("--local-rank", 0u, 65535u, local_rank)
        else SPARK_TRANSPORT_PARSE_OPTION("--peer-rank", 0u, 65535u, peer_rank)
        else
        {
            return 0;
        }
#undef SPARK_TRANSPORT_PARSE_OPTION
    }
    if (options->provider_path == 0 ||
        !SparkTransportIdentifierIsValid(options->question_id) ||
        !SparkTransportQuestionIsSupported(options->question_id) ||
        !SparkTransportIdentifierIsValid(options->candidate) ||
        !SparkTransportCandidateIsValidForQuestion(
            options->question_id,
            options->candidate) ||
        !SparkTransportIdentifierIsValid(options->progress_mode) ||
        !SparkTransportProgressModeIsValid(options->progress_mode) ||
        !SparkTransportHexIsValid(options->source_package_sha256) ||
        options->run_id == 0 || options->topology == 0 ||
        !SparkTransportIdentifierIsValid(options->node))
    {
        return 0;
    }
    if (SparkTransportRequiresPeer(options->question_id) &&
        !SparkTransportIdentifierIsValid(options->peer))
    {
        return 0;
    }
    return options->payload_bytes != 0u && options->lane_count != 0u &&
        options->window_depth != 0u && options->cq_batch != 0u &&
        options->registered_region_count != 0u;
}

static void SparkTransportUsage(const char *program_name)
{
    fprintf(stderr,
        "usage: %s --provider LIB --question ID --candidate ID "
        "--source-package-sha256 HASH --run-id ID --topology NAME --node NAME "
        "[--peer NAME] [--payload-bytes N] [--lane-count N] [--window-depth N] "
        "[--cq-batch N] [--registered-region-count N] [--progress-mode ID] "
        "[--iterations N] [--local-rank N] [--peer-rank N] [--output FILE]\n",
        program_name);
}

int main(int argument_count, char **arguments)
{
    SparkTransportOptions options;
    SparkHardwareTransportProbeRequest request;
    SparkHardwareTransportProbeResult result;
    const SparkHardwareTransportProbeInterface *provider;
    SparkHardwareTransportProbeGetInterfaceFunction get_interface;
    SparkStatus run_status;
    uint32_t required_provider_flags;
    void *library;
    void *symbol;
    const char *symbol_error;
    int write_ok;

    if (!SparkTransportParseOptions(argument_count, arguments, &options))
    {
        SparkTransportUsage(arguments[0]);
        return 2;
    }
    library = dlopen(options.provider_path, RTLD_NOW | RTLD_LOCAL);
    if (library == 0)
    {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    dlerror();
    symbol = dlsym(library, SPARK_HARDWARE_TRANSPORT_PROBE_INTERFACE_SYMBOL);
    symbol_error = dlerror();
    if (symbol == 0 || symbol_error != 0)
    {
        fprintf(stderr, "provider lacks %s: %s\n",
            SPARK_HARDWARE_TRANSPORT_PROBE_INTERFACE_SYMBOL,
            symbol_error == 0 ? "not found" : symbol_error);
        dlclose(library);
        return 1;
    }
    memcpy(&get_interface, &symbol, sizeof(get_interface));
    provider = get_interface();
    required_provider_flags =
        SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_EXACT_PRODUCTION_TRANSPORT |
        SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REMOTE_COMPLETION;
    if (strcmp(options.candidate, "tcp") == 0)
    {
        required_provider_flags |= SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REAL_TCP_PATH;
    }
    else
    {
        required_provider_flags |= SPARK_HARDWARE_TRANSPORT_PROVIDER_FLAG_REAL_VERBS_PATH;
    }
    if (provider == 0 ||
        provider->abi_version != SPARK_HARDWARE_TRANSPORT_PROBE_ABI_VERSION ||
        provider->descriptor_bytes != SPARK_HARDWARE_TRANSPORT_PROBE_INTERFACE_BYTES ||
        (provider->capability_flags & ~SPARK_HARDWARE_TRANSPORT_PROVIDER_KNOWN_FLAGS) != 0u ||
        (provider->capability_flags & required_provider_flags) != required_provider_flags ||
        provider->reserved != 0u ||
        !SparkTransportReservedU64IsZero(
            provider->reserved_u64,
            sizeof(provider->reserved_u64) / sizeof(provider->reserved_u64[0])) ||
        !SparkTransportIdentifierIsValid(provider->provider_id) ||
        !SparkTransportHexIsValid(provider->provider_build_identity) ||
        provider->run == 0)
    {
        fprintf(stderr, "invalid hardware transport provider interface\n");
        dlclose(library);
        return 1;
    }
    memset(&request, 0, sizeof(request));
    request.abi_version = SPARK_HARDWARE_TRANSPORT_PROBE_ABI_VERSION;
    request.descriptor_bytes = SPARK_HARDWARE_TRANSPORT_PROBE_REQUEST_BYTES;
    request.iterations = options.iterations;
    request.payload_bytes = options.payload_bytes;
    request.lane_count = options.lane_count;
    request.window_depth = options.window_depth;
    request.cq_batch = options.cq_batch;
    request.registered_region_count = options.registered_region_count;
    request.local_rank = options.local_rank;
    request.peer_rank = options.peer_rank;
    SparkTransportCopyIdentifier(request.question_id, options.question_id);
    SparkTransportCopyIdentifier(request.candidate, options.candidate);
    SparkTransportCopyIdentifier(request.progress_mode, options.progress_mode);
    SparkTransportCopyIdentifier(request.node, options.node);
    SparkTransportCopyIdentifier(request.peer, options.peer == 0 ? options.node : options.peer);
    SparkTransportCopyIdentifier(request.topology, options.topology);

    memset(&result, 0, sizeof(result));
    result.abi_version = SPARK_HARDWARE_TRANSPORT_PROBE_ABI_VERSION;
    result.descriptor_bytes = SPARK_HARDWARE_TRANSPORT_PROBE_RESULT_BYTES;
    run_status = provider->run(&request, &result);
    if (!SparkTransportResultIsValid(&options, run_status, &result))
    {
        fprintf(stderr, "provider failed to prove exact production transport semantics\n");
        dlclose(library);
        return 1;
    }
    write_ok = SparkTransportWriteReceipt(&options, provider, &result);
    dlclose(library);
    return write_ok ? 0 : 1;
}
