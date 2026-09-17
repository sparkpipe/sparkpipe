#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_driver_compiler.h"

#define SPARK_MODEL_COMPILE_MAX_EXTRA_ARGUMENTS 64u

static void SparkPrintUsage(const char *program_name)
{
    fprintf(
        stderr,
        "usage: %s --model FILE --library DIR --output DIR "
        "[--stage NAME] [--cc PROGRAM] [--include DIR] [--cc-arg ARG ...]\n"
        "       omit --stage to compile every stage into one model package\n",
        program_name);
}

int main(int argument_count, char **arguments)
{
    const char *model_description_path;
    const char *stage_name;
    const char *module_library_root;
    const char *output_directory;
    const char *compiler_path;
    const char *sparkpipe_include_directory;
    const char *extra_arguments[SPARK_MODEL_COMPILE_MAX_EXTRA_ARGUMENTS];
    uint32_t extra_argument_count;
    char error_buffer[1024];
    int argument_index;
    SparkStatus status;

    model_description_path = 0;
    stage_name = 0;
    module_library_root = 0;
    output_directory = 0;
    compiler_path = "cc";
    sparkpipe_include_directory = "include";
    extra_argument_count = 0u;

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
        SPARK_PARSE_OPTION("--model", model_description_path)
        SPARK_PARSE_OPTION("--stage", stage_name)
        SPARK_PARSE_OPTION("--library", module_library_root)
        SPARK_PARSE_OPTION("--output", output_directory)
        SPARK_PARSE_OPTION("--cc", compiler_path)
        SPARK_PARSE_OPTION("--include", sparkpipe_include_directory)
#undef SPARK_PARSE_OPTION
        if (strcmp(argument, "--cc-arg") == 0 && argument_index + 1 < argument_count)
        {
            if (extra_argument_count >= SPARK_MODEL_COMPILE_MAX_EXTRA_ARGUMENTS)
            {
                fprintf(stderr, "too many compiler arguments\n");
                return 2;
            }
            extra_arguments[extra_argument_count++] = arguments[++argument_index];
            continue;
        }
        SparkPrintUsage(arguments[0]);
        return 2;
    }

    if (stage_name != 0)
    {
        SparkDriverCompileRequest request;
        SparkDriverCompileReport report;

        memset(&request, 0, sizeof(request));
        memset(&report, 0, sizeof(report));
        request.model_description_path = model_description_path;
        request.stage_name = stage_name;
        request.module_library_root = module_library_root;
        request.output_directory = output_directory;
        request.compiler_path = compiler_path;
        request.sparkpipe_include_directory = sparkpipe_include_directory;
        request.extra_compiler_arguments = extra_arguments;
        request.extra_compiler_argument_count = extra_argument_count;

        status = SparkCompileModelDriver(&request, &report, error_buffer, sizeof(error_buffer));
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "model driver compile failed: %s: %s\n", SparkStatusToString(status), error_buffer);
            return 1;
        }
        printf(
            "driver=%s manifest=%s programs=%u operations=%u unique_link_units=%u "
            "model_sha256=%s compiled_program_sha256=%s driver_sha256=%s\n",
            report.driver_path,
            report.manifest_path,
            report.program_count,
            report.operation_count,
            report.unique_link_unit_count,
            report.model_description_sha256,
            report.compiled_program_sha256,
            report.driver_sha256);
        return 0;
    }
    else
    {
        SparkModelPackageCompileRequest request;
        SparkModelPackageCompileReport report;

        memset(&request, 0, sizeof(request));
        memset(&report, 0, sizeof(report));
        request.model_description_path = model_description_path;
        request.module_library_root = module_library_root;
        request.output_directory = output_directory;
        request.compiler_path = compiler_path;
        request.sparkpipe_include_directory = sparkpipe_include_directory;
        request.extra_compiler_arguments = extra_arguments;
        request.extra_compiler_argument_count = extra_argument_count;

        status = SparkCompileModelPackage(&request, &report, error_buffer, sizeof(error_buffer));
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "model package compile failed: %s: %s\n", SparkStatusToString(status), error_buffer);
            return 1;
        }
        printf(
            "package_manifest=%s stages=%u programs=%u operations=%u collected_link_units=%u "
            "model_sha256=%s package_manifest_sha256=%s\n",
            report.package_manifest_path,
            report.stage_count,
            report.total_program_count,
            report.total_operation_count,
            report.total_collected_link_unit_count,
            report.model_description_sha256,
            report.package_manifest_sha256);
        return 0;
    }
}
