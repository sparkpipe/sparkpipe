#include "test_support.h"

#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"

uint64_t SparkTestReadCounter(const char *path)
{
    FILE *file;
    unsigned long long value;

    file = fopen(path, "r");
    if (file == 0)
    {
        return 0u;
    }
    value = 0u;
    if (fscanf(file, "%llu", &value) != 1)
    {
        fclose(file);
        return 0u;
    }
    fclose(file);
    return (uint64_t)value;
}

static SparkStatus SparkTestPublishModule(
    const char *library_root,
    const char *target,
    const char *counter_path,
    const char *module_id,
    const char *link_unit_path,
    const char *initialize_symbol,
    const char *execute_symbol,
    const char *destroy_symbol,
    SparkModulePublishReport *report)
{
    SparkModulePublishRequest request;
    const char *validator_arguments[1];
    char error_buffer[1024];
    SparkStatus status;

    memset(&request, 0, sizeof(request));
    validator_arguments[0] = counter_path;
    request.library_root = library_root;
    request.module_id = module_id;
    request.target = target;
    request.module_abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    request.link_unit_path = link_unit_path;
    request.validation_recipe = "test.module.validator.v1";
    request.initialize_symbol = initialize_symbol;
    request.execute_symbol = execute_symbol;
    request.destroy_symbol = destroy_symbol;
    request.validator_path = "build/test_module_validator";
    request.validator_arguments = validator_arguments;
    request.validator_argument_count = 1u;
    status = SparkPublishValidatedModule(&request, report, error_buffer, sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "test module publish failed: %s: %s\n", SparkStatusToString(status), error_buffer);
    }
    return status;
}

SparkStatus SparkTestPublishAddOneModule(
    const char *library_root,
    const char *target,
    const char *counter_path,
    SparkModulePublishReport *report)
{
    return SparkTestPublishModule(
        library_root,
        target,
        counter_path,
        "spark.test.add_one.v1",
        "build/test_modules/module_add_one.o",
        "SparkTestAddOneInitialize",
        "SparkTestAddOneExecute",
        "SparkTestAddOneDestroy",
        report);
}

SparkStatus SparkTestPublishAddTwoAsAddOneModule(
    const char *library_root,
    const char *target,
    const char *counter_path,
    SparkModulePublishReport *report)
{
    return SparkTestPublishModule(
        library_root,
        target,
        counter_path,
        "spark.test.add_one.v1",
        "build/test_modules/module_add_two.o",
        "SparkTestAddOneInitialize",
        "SparkTestAddOneExecute",
        "SparkTestAddOneDestroy",
        report);
}

SparkStatus SparkTestPublishDoubleModule(
    const char *library_root,
    const char *target,
    const char *counter_path,
    SparkModulePublishReport *report)
{
    return SparkTestPublishModule(
        library_root,
        target,
        counter_path,
        "spark.test.double.v1",
        "build/test_modules/module_double.o",
        "",
        "SparkTestDoubleExecute",
        "",
        report);
}

SparkStatus SparkTestPublishAffineArchiveModule(
    const char *library_root,
    const char *target,
    const char *counter_path,
    SparkModulePublishReport *report)
{
    return SparkTestPublishModule(
        library_root,
        target,
        counter_path,
        "spark.test.affine.archive.v1",
        "build/test_modules/module_affine.a",
        "",
        "SparkTestAffineArchiveExecute",
        "",
        report);
}

SparkStatus SparkTestCompileDemoStage(
    const char *library_root,
    const char *stage_name,
    const char *output_directory,
    SparkDriverCompileReport *report)
{
    SparkDriverCompileRequest request;
    char error_buffer[1024];
    SparkStatus status;

    memset(&request, 0, sizeof(request));
    request.model_description_path = "examples/model_descriptions/firmware_demo.json";
    request.stage_name = stage_name;
    request.module_library_root = library_root;
    request.output_directory = output_directory;
    request.compiler_path = "cc";
    request.sparkpipe_include_directory = "include";
    status = SparkCompileModelDriver(&request, report, error_buffer, sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "test driver compile failed: %s: %s\n", SparkStatusToString(status), error_buffer);
    }
    return status;
}


SparkStatus SparkTestCompileDemoPackage(
    const char *library_root,
    const char *output_directory,
    SparkModelPackageCompileReport *report)
{
    SparkModelPackageCompileRequest request;
    char error_buffer[1024];
    SparkStatus status;

    memset(&request, 0, sizeof(request));
    request.model_description_path = "examples/model_descriptions/firmware_demo.json";
    request.module_library_root = library_root;
    request.output_directory = output_directory;
    request.compiler_path = "cc";
    request.sparkpipe_include_directory = "include";
    status = SparkCompileModelPackage(&request, report, error_buffer, sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "test model package compile failed: %s: %s\n", SparkStatusToString(status), error_buffer);
    }
    return status;
}

SparkStatus SparkTestCompileLinkUnitDemoStage(
    const char *library_root,
    const char *output_directory,
    SparkDriverCompileReport *report)
{
    SparkDriverCompileRequest request;
    char error_buffer[1024];
    SparkStatus status;

    memset(&request, 0, sizeof(request));
    request.model_description_path = "examples/model_descriptions/link_unit_demo.json";
    request.stage_name = "archive_stage";
    request.module_library_root = library_root;
    request.output_directory = output_directory;
    request.compiler_path = "cc";
    request.sparkpipe_include_directory = "include";
    status = SparkCompileModelDriver(&request, report, error_buffer, sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "test archive driver compile failed: %s: %s\n", SparkStatusToString(status), error_buffer);
    }
    return status;
}
