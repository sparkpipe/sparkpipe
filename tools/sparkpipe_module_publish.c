#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_module_library.h"

#define SPARK_MODULE_PUBLISH_MAX_VALIDATOR_ARGUMENTS 64u

static void SparkPrintUsage(const char *program_name)
{
    fprintf(
        stderr,
        "usage: %s --library DIR --module ID --target TARGET --link-unit FILE --recipe ID "
        "--execute SYMBOL [--initialize SYMBOL] [--admit SYMBOL] [--snapshot SYMBOL] [--destroy SYMBOL] "
        "[--validator PROGRAM] [--validator-arg ARG ...]\n",
        program_name);
}

int main(int argument_count, char **arguments)
{
    SparkModulePublishRequest request;
    SparkModulePublishReport report;
    const char *validator_arguments[SPARK_MODULE_PUBLISH_MAX_VALIDATOR_ARGUMENTS];
    char error_buffer[1024];
    uint32_t validator_argument_count;
    int argument_index;
    SparkStatus status;

    memset(&request, 0, sizeof(request));
    memset(&report, 0, sizeof(report));
    validator_argument_count = 0u;
    request.module_abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    request.initialize_symbol = "";
    request.admit_symbol = "";
    request.snapshot_symbol = "";
    request.destroy_symbol = "";

    for (argument_index = 1; argument_index < argument_count; ++argument_index)
    {
        const char *argument;

        argument = arguments[argument_index];
#define SPARK_PARSE_OPTION(option_name, destination) \
        if (strcmp(argument, option_name) == 0 && argument_index + 1 < argument_count) \
        { \
            destination = arguments[++argument_index]; \
            continue; \
        }
        SPARK_PARSE_OPTION("--library", request.library_root)
        SPARK_PARSE_OPTION("--module", request.module_id)
        SPARK_PARSE_OPTION("--target", request.target)
        SPARK_PARSE_OPTION("--link-unit", request.link_unit_path)
        SPARK_PARSE_OPTION("--recipe", request.validation_recipe)
        SPARK_PARSE_OPTION("--initialize", request.initialize_symbol)
        SPARK_PARSE_OPTION("--execute", request.execute_symbol)
        SPARK_PARSE_OPTION("--admit", request.admit_symbol)
        SPARK_PARSE_OPTION("--snapshot", request.snapshot_symbol)
        SPARK_PARSE_OPTION("--destroy", request.destroy_symbol)
        SPARK_PARSE_OPTION("--validator", request.validator_path)
#undef SPARK_PARSE_OPTION
        if (strcmp(argument, "--validator-arg") == 0 && argument_index + 1 < argument_count)
        {
            if (validator_argument_count >= SPARK_MODULE_PUBLISH_MAX_VALIDATOR_ARGUMENTS)
            {
                fprintf(stderr, "too many validator arguments\n");
                return 2;
            }
            validator_arguments[validator_argument_count++] = arguments[++argument_index];
            continue;
        }
        SparkPrintUsage(arguments[0]);
        return 2;
    }

    request.validator_arguments = validator_arguments;
    request.validator_argument_count = validator_argument_count;
    status = SparkPublishValidatedModule(&request, &report, error_buffer, sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "module publish failed: %s: %s\n", SparkStatusToString(status), error_buffer);
        return 1;
    }
    printf(
        "module=%s target=%s artifact=%s kind=%s validation=%s link_unit=%s record=%s\n",
        request.module_id,
        request.target,
        report.artifact_sha256,
        SparkModuleLinkUnitKindToString(report.link_unit_kind),
        report.validation_reused ? "reused" : "executed",
        report.stored_link_unit_path,
        report.active_record_path);
    return 0;
}
