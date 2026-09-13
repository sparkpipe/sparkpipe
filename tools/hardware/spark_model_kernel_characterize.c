#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_hardware_kernel_probe.h"

#define SPARK_MODEL_KERNEL_DEFAULT_ITERATIONS 100u

typedef struct SparkModelKernelProbeOptions
{
    const char *provider_path;
    const char *question_id;
    const char *model_id;
    const char *role;
    const char *candidate;
    const char *kernel_class;
    const char *route_distribution;
    const char *source_package_sha256;
    const char *run_id;
    const char *topology;
    const char *node;
    const char *output_path;
    uint32_t batch_size;
    uint32_t context_tokens;
    uint32_t iterations;
} SparkModelKernelProbeOptions;

static int SparkModelKernelProbeHexIsValid(const char *text)
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

static int SparkModelKernelProbeParseU32(
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

static int SparkModelKernelProbeIdentifierIsValid(const char *text)
{
    const unsigned char *cursor;
    size_t bytes;

    if (text == 0 || text[0] == '\0')
    {
        return 0;
    }
    bytes = strlen(text);
    if (bytes >= SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES)
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

static int SparkModelKernelProbeQuestionIsSupported(const char *question_id)
{
    return strcmp(question_id, "GB10-REG-001") == 0 ||
        strcmp(question_id, "GB10-NATIVE-MMA-001") == 0 ||
        strcmp(question_id, "MODEL-GQA-001") == 0 ||
        strcmp(question_id, "MODEL-MOE-001") == 0 ||
        strcmp(question_id, "MODEL-KDA-001") == 0;
}

static void SparkModelKernelProbeCopyIdentifier(
    char destination[SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES],
    const char *source)
{
    (void)snprintf(
        destination,
        SPARK_HARDWARE_KERNEL_PROBE_MAX_IDENTIFIER_BYTES,
        "%s",
        source == 0 ? "" : source);
}

static int SparkModelKernelProbeReservedU32IsZero(const uint32_t *values, size_t count)
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

static int SparkModelKernelProbeReservedU64IsZero(const uint64_t *values, size_t count)
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

static int SparkModelKernelProbeResultIsValid(
    const SparkModelKernelProbeOptions *options,
    SparkStatus run_status,
    const SparkHardwareKernelProbeResult *result)
{
    uint32_t required_flags;

    if (result->abi_version != SPARK_HARDWARE_KERNEL_PROBE_ABI_VERSION ||
        result->descriptor_bytes != SPARK_HARDWARE_KERNEL_PROBE_RESULT_BYTES ||
        result->status != run_status ||
        (result->result_flags & ~SPARK_HARDWARE_KERNEL_PROBE_RESULT_KNOWN_FLAGS) != 0u ||
        !SparkModelKernelProbeReservedU32IsZero(
            result->reserved_u32,
            sizeof(result->reserved_u32) / sizeof(result->reserved_u32[0])))
    {
        return 0;
    }
    if (run_status != SPARK_STATUS_OK)
    {
        return 1;
    }
    if (!SparkModelKernelProbeHexIsValid(result->kernel_artifact_sha256) ||
        !SparkModelKernelProbeHexIsValid(result->ptxas_receipt_sha256) ||
        !SparkModelKernelProbeHexIsValid(result->reference_artifact_sha256))
    {
        return 0;
    }
    required_flags =
        SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_EXACT_PRODUCTION_KERNEL |
        SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_PTXAS_RECEIPT_BOUND |
        SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_REFERENCE_INDEPENDENT;
    if ((result->result_flags & required_flags) != required_flags ||
        result->sample_count == 0u ||
        result->latency_p50_ns == 0u ||
        result->latency_p50_ns > result->latency_p95_ns ||
        result->latency_p95_ns > result->latency_p99_ns ||
        result->numerical_pass == 0u || result->integrity_pass == 0u ||
        !isfinite(result->throughput_gb_s) || result->throughput_gb_s < 0.0 ||
        !isfinite(result->tokens_per_second) || result->tokens_per_second < 0.0 ||
        !isfinite(result->achieved_tflops) || result->achieved_tflops < 0.0 ||
        !isfinite(result->maximum_absolute_error) || result->maximum_absolute_error < 0.0 ||
        !isfinite(result->maximum_relative_error) || result->maximum_relative_error < 0.0 ||
        !isfinite(result->theoretical_occupancy) || result->theoretical_occupancy < 0.0 ||
        !isfinite(result->achieved_occupancy) || result->achieved_occupancy < 0.0)
    {
        return 0;
    }
    if (strcmp(options->question_id, "GB10-NATIVE-MMA-001") == 0 &&
        strcmp(options->candidate, "native_block_scaled") == 0 &&
        (result->result_flags & SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_NATIVE_TENSOR_CORE) == 0u)
    {
        return 0;
    }
    return 1;
}

static void SparkModelKernelProbeWriteJsonString(FILE *output, const char *text)
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

static const char *SparkModelKernelProbeStatusName(SparkStatus status)
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

static int SparkModelKernelProbeWriteReceipt(
    const SparkModelKernelProbeOptions *options,
    const SparkHardwareKernelProbeInterface *provider,
    const SparkHardwareKernelProbeResult *result)
{
    FILE *output;
    const char *status_name;

    output = options->output_path == 0 ? stdout : fopen(options->output_path, "wb");
    if (output == 0)
    {
        return 0;
    }
    status_name = SparkModelKernelProbeStatusName(result->status);
    fputs("{\n  \"schema_version\": 1,\n  \"receipt_kind\": \"spark_hardware_probe\",\n  \"run_id\": ", output);
    SparkModelKernelProbeWriteJsonString(output, options->run_id);
    fputs(",\n  \"probe_id\": \"model_kernel_characterize\",\n  \"source_identity\": {\"source_package_sha256\": ", output);
    SparkModelKernelProbeWriteJsonString(output, options->source_package_sha256);
    fputs("},\n  \"scope\": {\"topology\": ", output);
    SparkModelKernelProbeWriteJsonString(output, options->topology);
    fputs(", \"node\": ", output);
    SparkModelKernelProbeWriteJsonString(output, options->node);
    fputs("},\n  \"answers\": [\n    {\"question_id\": ", output);
    SparkModelKernelProbeWriteJsonString(output, options->question_id);
    fputs(", \"status\": ", output);
    SparkModelKernelProbeWriteJsonString(output, status_name);
    fputs(", \"summary\": {\"provider_id\": ", output);
    SparkModelKernelProbeWriteJsonString(output, provider->provider_id);
    fputs(", \"provider_build_identity\": ", output);
    SparkModelKernelProbeWriteJsonString(output, provider->provider_build_identity);
    fputs(", \"kernel_artifact_sha256\": ", output);
    SparkModelKernelProbeWriteJsonString(output, result->kernel_artifact_sha256);
    fputs(", \"ptxas_receipt_sha256\": ", output);
    SparkModelKernelProbeWriteJsonString(output, result->ptxas_receipt_sha256);
    fputs(", \"reference_artifact_sha256\": ", output);
    SparkModelKernelProbeWriteJsonString(output, result->reference_artifact_sha256);
    fprintf(output,
        ", \"exact_production_kernel\": %s, \"native_tensor_core\": %s, "
        "\"ptxas_receipt_bound\": %s, \"independent_reference\": %s}, "
        "\"observations\": [",
        (result->result_flags & SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_EXACT_PRODUCTION_KERNEL) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_NATIVE_TENSOR_CORE) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_PTXAS_RECEIPT_BOUND) != 0u ? "true" : "false",
        (result->result_flags & SPARK_HARDWARE_KERNEL_PROBE_RESULT_FLAG_REFERENCE_INDEPENDENT) != 0u ? "true" : "false");
    if (result->status == SPARK_STATUS_OK)
    {
        fputs("{\"parameters\": {\"model_id\": ", output);
        SparkModelKernelProbeWriteJsonString(output, options->model_id);
        fputs(", \"role\": ", output);
        SparkModelKernelProbeWriteJsonString(output, options->role);
        fputs(", \"candidate\": ", output);
        SparkModelKernelProbeWriteJsonString(output, options->candidate);
        fputs(", \"kernel_class\": ", output);
        SparkModelKernelProbeWriteJsonString(output, options->kernel_class);
        fputs(", \"route_distribution\": ", output);
        SparkModelKernelProbeWriteJsonString(output, options->route_distribution);
        fprintf(output,
            ", \"batch_size\": %u, \"context_tokens\": %u, \"iterations\": %u}, \"metrics\": {"
            "\"latency_p50_ns\": %" PRIu64 ", \"latency_p95_ns\": %" PRIu64 ", "
            "\"latency_p99_ns\": %" PRIu64 ", \"throughput_gb_s\": %.17g, "
            "\"tokens_per_second\": %.17g, \"achieved_tflops\": %.17g, "
            "\"maximum_absolute_error\": %.17g, \"maximum_relative_error\": %.17g, "
            "\"register_count\": %u, \"static_shared_bytes\": %u, "
            "\"dynamic_shared_bytes\": %u, \"local_bytes_per_thread\": %u, "
            "\"spill_load_bytes_per_thread\": %u, \"spill_store_bytes_per_thread\": %u, "
            "\"active_blocks_per_sm\": %u, \"active_warps_per_sm\": %u, "
            "\"theoretical_occupancy\": %.17g, \"achieved_occupancy\": %.17g, "
            "\"sample_count\": %" PRIu64 ", \"numerical_pass\": true, "
            "\"integrity_pass\": true}}",
            options->batch_size,
            options->context_tokens,
            options->iterations,
            result->latency_p50_ns,
            result->latency_p95_ns,
            result->latency_p99_ns,
            result->throughput_gb_s,
            result->tokens_per_second,
            result->achieved_tflops,
            result->maximum_absolute_error,
            result->maximum_relative_error,
            result->register_count,
            result->static_shared_bytes,
            result->dynamic_shared_bytes,
            result->local_bytes_per_thread,
            result->spill_load_bytes_per_thread,
            result->spill_store_bytes_per_thread,
            result->active_blocks_per_sm,
            result->active_warps_per_sm,
            result->theoretical_occupancy,
            result->achieved_occupancy,
            result->sample_count);
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

static void SparkModelKernelProbeUsage(const char *program_name)
{
    fprintf(stderr,
        "usage: %s --provider LIB --question ID --model ID --candidate ID "
        "[--role ID] [--kernel-class ID] [--route-distribution ID] "
        "--batch N [--context N] [--iterations N] --source-package-sha256 HASH "
        "--run-id ID --topology NAME --node NAME [--output FILE]\n",
        program_name);
}

static int SparkModelKernelProbeParseOptions(
    int argument_count,
    char **arguments,
    SparkModelKernelProbeOptions *options)
{
    int index;

    memset(options, 0, sizeof(*options));
    options->role = "none";
    options->kernel_class = "none";
    options->route_distribution = "none";
    options->iterations = SPARK_MODEL_KERNEL_DEFAULT_ITERATIONS;
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
        else if (strcmp(argument, "--model") == 0)
        {
            text_destination = &options->model_id;
        }
        else if (strcmp(argument, "--role") == 0)
        {
            text_destination = &options->role;
        }
        else if (strcmp(argument, "--candidate") == 0)
        {
            text_destination = &options->candidate;
        }
        else if (strcmp(argument, "--kernel-class") == 0)
        {
            text_destination = &options->kernel_class;
        }
        else if (strcmp(argument, "--route-distribution") == 0)
        {
            text_destination = &options->route_distribution;
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
        if (strcmp(argument, "--batch") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkModelKernelProbeParseU32(
                    arguments[++index], 1u, 1048576u, &options->batch_size))
            {
                return 0;
            }
        }
        else if (strcmp(argument, "--context") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkModelKernelProbeParseU32(
                    arguments[++index], 0u, 16777216u, &options->context_tokens))
            {
                return 0;
            }
        }
        else if (strcmp(argument, "--iterations") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkModelKernelProbeParseU32(
                    arguments[++index], 1u, 1000000u, &options->iterations))
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }
    return options->provider_path != 0 &&
        SparkModelKernelProbeIdentifierIsValid(options->question_id) &&
        SparkModelKernelProbeQuestionIsSupported(options->question_id) &&
        SparkModelKernelProbeIdentifierIsValid(options->model_id) &&
        SparkModelKernelProbeIdentifierIsValid(options->role) &&
        SparkModelKernelProbeIdentifierIsValid(options->candidate) &&
        SparkModelKernelProbeIdentifierIsValid(options->kernel_class) &&
        SparkModelKernelProbeIdentifierIsValid(options->route_distribution) &&
        options->batch_size != 0u &&
        SparkModelKernelProbeHexIsValid(options->source_package_sha256) &&
        options->run_id != 0 && options->topology != 0 && options->node != 0;
}

int main(int argument_count, char **arguments)
{
    SparkModelKernelProbeOptions options;
    SparkHardwareKernelProbeRequest request;
    SparkHardwareKernelProbeResult result;
    const SparkHardwareKernelProbeInterface *provider;
    SparkHardwareKernelProbeGetInterfaceFunction get_interface;
    SparkStatus run_status;
    uint32_t required_provider_flags;
    void *library;
    void *symbol;
    const char *symbol_error;
    int write_ok;

    if (!SparkModelKernelProbeParseOptions(argument_count, arguments, &options))
    {
        SparkModelKernelProbeUsage(arguments[0]);
        return 2;
    }
    library = dlopen(options.provider_path, RTLD_NOW | RTLD_LOCAL);
    if (library == 0)
    {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    dlerror();
    symbol = dlsym(library, SPARK_HARDWARE_KERNEL_PROBE_INTERFACE_SYMBOL);
    symbol_error = dlerror();
    if (symbol == 0 || symbol_error != 0)
    {
        fprintf(stderr, "provider lacks %s: %s\n",
            SPARK_HARDWARE_KERNEL_PROBE_INTERFACE_SYMBOL,
            symbol_error == 0 ? "not found" : symbol_error);
        dlclose(library);
        return 1;
    }
    memcpy(&get_interface, &symbol, sizeof(get_interface));
    provider = get_interface();
    required_provider_flags =
        SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_EXACT_PRODUCTION_PATH |
        SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_INDEPENDENT_REFERENCE |
        SPARK_HARDWARE_KERNEL_PROVIDER_FLAG_RESOURCE_RECEIPTS;
    if (provider == 0 ||
        provider->abi_version != SPARK_HARDWARE_KERNEL_PROBE_ABI_VERSION ||
        provider->descriptor_bytes != SPARK_HARDWARE_KERNEL_PROBE_INTERFACE_BYTES ||
        (provider->capability_flags & ~SPARK_HARDWARE_KERNEL_PROVIDER_KNOWN_FLAGS) != 0u ||
        (provider->capability_flags & required_provider_flags) != required_provider_flags ||
        provider->reserved != 0u ||
        !SparkModelKernelProbeReservedU64IsZero(
            provider->reserved_u64,
            sizeof(provider->reserved_u64) / sizeof(provider->reserved_u64[0])) ||
        !SparkModelKernelProbeIdentifierIsValid(provider->provider_id) ||
        !SparkModelKernelProbeHexIsValid(provider->provider_build_identity) ||
        provider->run == 0)
    {
        fprintf(stderr, "invalid hardware kernel provider interface\n");
        dlclose(library);
        return 1;
    }
    memset(&request, 0, sizeof(request));
    request.abi_version = SPARK_HARDWARE_KERNEL_PROBE_ABI_VERSION;
    request.descriptor_bytes = SPARK_HARDWARE_KERNEL_PROBE_REQUEST_BYTES;
    request.flags = SPARK_HARDWARE_KERNEL_PROBE_FLAG_REQUIRE_NUMERICAL_REFERENCE |
        SPARK_HARDWARE_KERNEL_PROBE_FLAG_REQUIRE_EXACT_PRODUCTION_KERNEL |
        SPARK_HARDWARE_KERNEL_PROBE_FLAG_COLLECT_RESOURCE_METRICS;
    request.iterations = options.iterations;
    request.batch_size = options.batch_size;
    request.context_tokens = options.context_tokens;
    SparkModelKernelProbeCopyIdentifier(request.question_id, options.question_id);
    SparkModelKernelProbeCopyIdentifier(request.model_id, options.model_id);
    SparkModelKernelProbeCopyIdentifier(request.role, options.role);
    SparkModelKernelProbeCopyIdentifier(request.candidate, options.candidate);
    SparkModelKernelProbeCopyIdentifier(request.kernel_class, options.kernel_class);
    SparkModelKernelProbeCopyIdentifier(request.route_distribution, options.route_distribution);

    memset(&result, 0, sizeof(result));
    result.abi_version = SPARK_HARDWARE_KERNEL_PROBE_ABI_VERSION;
    result.descriptor_bytes = SPARK_HARDWARE_KERNEL_PROBE_RESULT_BYTES;
    run_status = provider->run(&request, &result);
    if (!SparkModelKernelProbeResultIsValid(&options, run_status, &result))
    {
        fprintf(stderr, "provider returned invalid or incomplete qualification evidence\n");
        dlclose(library);
        return 1;
    }
    write_ok = SparkModelKernelProbeWriteReceipt(&options, provider, &result);
    dlclose(library);
    return write_ok ? 0 : 1;
}
