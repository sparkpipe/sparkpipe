#include <assert.h>
#include <dirent.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_compiler.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_model_driver.h"
#include "test_support.h"

typedef struct SparkTestCompletionState
{
    uint32_t completion_count;
    SparkModelDriverCompletion completion;
} SparkTestCompletionState;

static void SparkTestCompletion(void *completion_context, const SparkModelDriverCompletion *completion)
{
    SparkTestCompletionState *state;

    state = (SparkTestCompletionState *)completion_context;
    assert(state != 0);
    assert(completion != 0);
    state->completion_count += 1u;
    state->completion = *completion;
}

static uint32_t SparkTestCountDirectoryEntries(const char *path)
{
    DIR *directory;
    struct dirent *directory_entry;
    uint32_t entry_count;

    directory = opendir(path);
    assert(directory != 0);
    entry_count = 0u;
    while ((directory_entry = readdir(directory)) != 0)
    {
        if (strcmp(directory_entry->d_name, ".") != 0 && strcmp(directory_entry->d_name, "..") != 0)
        {
            entry_count += 1u;
        }
    }
    assert(closedir(directory) == 0);
    return entry_count;
}

static char *SparkTestReadFile(const char *path)
{
    FILE *file;
    long file_bytes;
    char *text;

    file = fopen(path, "rb");
    assert(file != 0);
    assert(fseek(file, 0, SEEK_END) == 0);
    file_bytes = ftell(file);
    assert(file_bytes >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    text = (char *)malloc((size_t)file_bytes + 1u);
    assert(text != 0);
    assert(fread(text, 1u, (size_t)file_bytes, file) == (size_t)file_bytes);
    text[file_bytes] = '\0';
    assert(fclose(file) == 0);
    return text;
}

int main(void)
{
    static const char LibraryRoot[] = "build/test_driver_library";
    static const char CounterPath[] = "build/test_driver_validator_count.txt";
    SparkModulePublishReport publish_report;
    SparkModulePublishReport archive_publish_report;
    SparkDriverCompileReport compile_report;
    SparkDriverCompileReport second_compile_report;
    SparkDriverCompileReport reused_output_report;
    SparkDriverCompileReport variant_compile_report;
    SparkDriverCompileReport archive_compile_report;
    SparkDriverCompileRequest variant_request;
    SparkDriverCompileRequest missing_request;
    SparkDriverCompileRequest invalid_request;
    SparkModelPackageCompileReport package_report;
    SparkModelPackageCompileReport stale_package_report;
    SparkModelPackageCompileRequest missing_package_request;
    SparkLoadedModelDriver loaded_driver;
    SparkModelDriverCreateRequest create_request;
    const SparkModelDriverProgramDescriptor *program;
    SparkModelDriverFrame frame;
    SparkTestCompletionState completion_state;
    void *driver_instance;
    const char *variant_compiler_arguments[1];
    struct stat library_link_unit_status;
    struct stat gathered_link_unit_status;
    char gathered_link_unit_path[SPARK_DRIVER_COMPILER_PATH_BYTES];
    char error_buffer[1024];
    char *generated_source;
    char *compiled_manifest;

    assert(system("rm -rf build/test_driver_library build/test_driver_cpu build/test_driver_cpu_second build/test_driver_variant build/test_driver_reused build/test_driver_package build/test_driver_package_missing build/test_driver_missing build/test_driver_invalid build/test_driver_archive build/test_driver_validator_count.txt") == 0);
    assert(SparkTestPublishAddOneModule(LibraryRoot, "host.cpu", CounterPath, &publish_report) == SPARK_STATUS_OK);
    assert(SparkTestPublishDoubleModule(LibraryRoot, "host.cpu", CounterPath, &publish_report) == SPARK_STATUS_OK);
    assert(SparkTestReadCounter(CounterPath) == 2u);

    assert(SparkTestCompileDemoStage(LibraryRoot, "cpu_stage", "build/test_driver_cpu", &compile_report) == SPARK_STATUS_OK);
    assert(compile_report.program_count == 1u);
    assert(compile_report.operation_count == 2u);
    assert(compile_report.unique_link_unit_count == 2u);
    assert(SparkSha256HexIsValid(compile_report.driver_sha256));
    assert(SparkTestReadCounter(CounterPath) == 2u);
    assert(snprintf(
               gathered_link_unit_path,
               sizeof(gathered_link_unit_path),
               "build/test_driver_cpu/link_units/%s.o",
               publish_report.artifact_sha256) > 0);
    assert(stat(publish_report.stored_link_unit_path, &library_link_unit_status) == 0);
    assert(stat(gathered_link_unit_path, &gathered_link_unit_status) == 0);
    assert(library_link_unit_status.st_dev != gathered_link_unit_status.st_dev ||
           library_link_unit_status.st_ino != gathered_link_unit_status.st_ino);
    assert((gathered_link_unit_status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0);

    generated_source = SparkTestReadFile(compile_report.generated_source_path);
    assert(strstr(generated_source, "SparkTestAddOneExecute(instance->operation_0_state, frame)") != 0);
    assert(strstr(generated_source, "SparkTestDoubleExecute(instance->operation_1_state, frame)") != 0);
    assert(strstr(generated_source, "SparkResolveValidatedModule") == 0);
    assert(strstr(generated_source, "for (") == 0);
    free(generated_source);

    compiled_manifest = SparkTestReadFile(compile_report.manifest_path);
    assert(strstr(compiled_manifest, "\"validation_recipe\": \"test.module.validator.v1\"") != 0);
    assert(strstr(compiled_manifest, "\"module_abi_version\": 2") != 0);
    assert(strstr(compiled_manifest, "\"execute_symbol\": \"SparkTestAddOneExecute\"") != 0);
    free(compiled_manifest);

    SparkLoadedModelDriverReset(&loaded_driver);
    assert(SparkLoadModelDriver(compile_report.driver_path, "host.cpu", &loaded_driver, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(dlsym(loaded_driver.dynamic_library, "SparkTestAddOneExecute") == 0);
    assert(dlsym(loaded_driver.dynamic_library, "SparkTestDoubleExecute") == 0);
    assert(strcmp(loaded_driver.interface->descriptor->model_id, "sparkpipe.firmware.demo") == 0);
    assert(strcmp(loaded_driver.interface->descriptor->stage_name, "cpu_stage") == 0);
    program = SparkFindLoadedModelDriverProgram(&loaded_driver, "decode");
    assert(program != 0);

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&create_request, 0, sizeof(create_request));
    create_request.node_id = "cpu0";
    create_request.node_target = "host.cpu";
    create_request.completion_function = SparkTestCompletion;
    create_request.completion_context = &completion_state;
    driver_instance = 0;
    assert(loaded_driver.interface->create(&create_request, &driver_instance) == SPARK_STATUS_OK);
    assert(driver_instance != 0);

    memset(&frame, 0, sizeof(frame));
    frame.request_id = 77u;
    frame.scalar[0] = 3u;
    assert(program->submit(driver_instance, &frame) == SPARK_STATUS_OK);
    assert(frame.scalar[0] == 8u);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.completion.request_id == 77u);
    assert(completion_state.completion.status == SPARK_STATUS_OK);
    loaded_driver.interface->destroy(driver_instance);
    SparkUnloadModelDriver(&loaded_driver);

    assert(SparkTestCompileDemoStage(LibraryRoot, "cpu_stage", "build/test_driver_cpu_second", &second_compile_report) == SPARK_STATUS_OK);
    assert(SparkTestReadCounter(CounterPath) == 2u);
    assert(strcmp(compile_report.compiled_program_sha256, second_compile_report.compiled_program_sha256) == 0);

    memset(&variant_request, 0, sizeof(variant_request));
    variant_compiler_arguments[0] = "-DSPARK_TEST_FIRMWARE_VARIANT=1";
    variant_request.model_description_path = "examples/model_descriptions/firmware_demo.json";
    variant_request.stage_name = "cpu_stage";
    variant_request.module_library_root = LibraryRoot;
    variant_request.output_directory = "build/test_driver_variant";
    variant_request.compiler_path = "cc";
    variant_request.sparkpipe_include_directory = "include";
    variant_request.extra_compiler_arguments = variant_compiler_arguments;
    variant_request.extra_compiler_argument_count = 1u;
    assert(SparkCompileModelDriver(&variant_request, &variant_compile_report, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(strcmp(compile_report.compiled_program_sha256, variant_compile_report.compiled_program_sha256) != 0);
    assert(SparkTestReadCounter(CounterPath) == 2u);

    assert(SparkTestCompileDemoStage(LibraryRoot, "cpu_stage", "build/test_driver_reused", &reused_output_report) == SPARK_STATUS_OK);
    assert(SparkTestCountDirectoryEntries("build/test_driver_reused/link_units") == 2u);
    assert(SparkTestPublishDoubleModule(LibraryRoot, "host.accelerator", CounterPath, &publish_report) == SPARK_STATUS_OK);
    assert(SparkTestReadCounter(CounterPath) == 3u);
    assert(SparkTestCompileDemoStage(LibraryRoot, "accelerator_stage", "build/test_driver_reused", &reused_output_report) == SPARK_STATUS_OK);
    assert(reused_output_report.unique_link_unit_count == 1u);
    assert(SparkTestCountDirectoryEntries("build/test_driver_reused/link_units") == 1u);
    assert(SparkTestReadCounter(CounterPath) == 3u);

    assert(SparkTestCompileDemoPackage(LibraryRoot, "build/test_driver_package", &package_report) == SPARK_STATUS_OK);
    assert(package_report.stage_count == 2u);
    assert(package_report.total_program_count == 2u);
    assert(package_report.total_operation_count == 3u);
    assert(package_report.total_collected_link_unit_count == 3u);
    assert(SparkSha256HexIsValid(package_report.package_manifest_sha256));
    assert(SparkTestReadCounter(CounterPath) == 3u);
    assert(access("build/test_driver_package/stages/stage_000/model_driver.so", F_OK) == 0);
    assert(access("build/test_driver_package/stages/stage_001/model_driver.so", F_OK) == 0);
    assert(access("build/test_driver_package/stages/stage_000/compiled_manifest.json", F_OK) != 0);
    assert(access("build/test_driver_package/stages/stage_001/compiled_manifest.json", F_OK) != 0);
    compiled_manifest = SparkTestReadFile(package_report.package_manifest_path);
    assert(strstr(compiled_manifest, "\"model_description\": {") != 0);
    assert(strstr(compiled_manifest, "\"stage_name\": \"cpu_stage\"") != 0);
    assert(strstr(compiled_manifest, "\"stage_name\": \"accelerator_stage\"") != 0);
    assert(strstr(compiled_manifest, "stages/stage_000/model_driver.so") != 0);
    assert(strstr(compiled_manifest, "stages/stage_001/model_driver.so") != 0);
    free(compiled_manifest);

    SparkLoadedModelDriverReset(&loaded_driver);
    assert(SparkLoadModelDriver("build/test_driver_package/stages/stage_000/model_driver.so", "host.cpu", &loaded_driver, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(strcmp(loaded_driver.interface->descriptor->stage_name, "cpu_stage") == 0);
    SparkUnloadModelDriver(&loaded_driver);
    assert(SparkLoadModelDriver("build/test_driver_package/stages/stage_001/model_driver.so", "host.accelerator", &loaded_driver, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(strcmp(loaded_driver.interface->descriptor->stage_name, "accelerator_stage") == 0);
    SparkUnloadModelDriver(&loaded_driver);

    assert(SparkTestCompileDemoPackage(LibraryRoot, "build/test_driver_package_missing", &stale_package_report) == SPARK_STATUS_OK);
    memset(&missing_package_request, 0, sizeof(missing_package_request));
    missing_package_request.model_description_path = "examples/model_descriptions/missing_module.json";
    missing_package_request.module_library_root = LibraryRoot;
    missing_package_request.output_directory = "build/test_driver_package_missing";
    missing_package_request.compiler_path = "cc";
    missing_package_request.sparkpipe_include_directory = "include";
    assert(SparkCompileModelPackage(&missing_package_request, &stale_package_report, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(access("build/test_driver_package_missing/model_package.json", F_OK) != 0);
    assert(access("build/test_driver_package_missing/stages", F_OK) != 0);
    assert(SparkTestReadCounter(CounterPath) == 3u);

    assert(SparkTestCompileDemoStage(LibraryRoot, "cpu_stage", "build/test_driver_missing", &second_compile_report) == SPARK_STATUS_OK);
    assert(access("build/test_driver_missing/model_driver.so", F_OK) == 0);
    assert(access("build/test_driver_missing/compiled_manifest.json", F_OK) == 0);
    assert(SparkTestCountDirectoryEntries("build/test_driver_missing/link_units") == 2u);

    memset(&missing_request, 0, sizeof(missing_request));
    missing_request.model_description_path = "examples/model_descriptions/missing_module.json";
    missing_request.stage_name = "stage0";
    missing_request.module_library_root = LibraryRoot;
    missing_request.output_directory = "build/test_driver_missing";
    missing_request.compiler_path = "cc";
    missing_request.sparkpipe_include_directory = "include";
    assert(SparkCompileModelDriver(&missing_request, &second_compile_report, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(strstr(error_buffer, "not in the library") != 0);
    assert(access("build/test_driver_missing/model_driver.so", F_OK) != 0);
    assert(access("build/test_driver_missing/compiled_manifest.json", F_OK) != 0);
    assert(access("build/test_driver_missing/spark_model_driver_generated.c", F_OK) != 0);
    assert(SparkTestCountDirectoryEntries("build/test_driver_missing/link_units") == 0u);
    assert(SparkTestReadCounter(CounterPath) == 3u);

    assert(SparkTestPublishAffineArchiveModule(
               LibraryRoot,
               "host.cpu",
               CounterPath,
               &archive_publish_report) == SPARK_STATUS_OK);
    assert(!archive_publish_report.validation_reused);
    assert(archive_publish_report.link_unit_kind == SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE);
    assert(SparkTestReadCounter(CounterPath) == 4u);
    assert(SparkTestCompileLinkUnitDemoStage(
               LibraryRoot,
               "build/test_driver_archive",
               &archive_compile_report) == SPARK_STATUS_OK);
    assert(archive_compile_report.program_count == 1u);
    assert(archive_compile_report.operation_count == 1u);
    assert(archive_compile_report.unique_link_unit_count == 1u);
    assert(snprintf(
               gathered_link_unit_path,
               sizeof(gathered_link_unit_path),
               "build/test_driver_archive/link_units/%s.a",
               archive_publish_report.artifact_sha256) > 0);
    assert(access(gathered_link_unit_path, F_OK) == 0);
    assert(SparkTestCountDirectoryEntries("build/test_driver_archive/link_units") == 1u);

    generated_source = SparkTestReadFile(archive_compile_report.generated_source_path);
    assert(strstr(
               generated_source,
               "SparkTestAffineArchiveExecute(instance->operation_0_state, frame)") != 0);
    assert(strstr(generated_source, "SparkResolveValidatedModule") == 0);
    assert(strstr(generated_source, "for (") == 0);
    free(generated_source);

    compiled_manifest = SparkTestReadFile(archive_compile_report.manifest_path);
    assert(strstr(compiled_manifest, "\"link_unit_kind\": \"static_archive\"") != 0);
    assert(strstr(compiled_manifest, ".a\"") != 0);
    free(compiled_manifest);

    SparkLoadedModelDriverReset(&loaded_driver);
    assert(SparkLoadModelDriver(
               archive_compile_report.driver_path,
               "host.cpu",
               &loaded_driver,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(dlsym(loaded_driver.dynamic_library, "SparkTestAffineArchiveExecute") == 0);
    assert(dlsym(loaded_driver.dynamic_library, "SparkTestAffineApply") == 0);
    program = SparkFindLoadedModelDriverProgram(&loaded_driver, "decode");
    assert(program != 0);
    memset(&completion_state, 0, sizeof(completion_state));
    memset(&create_request, 0, sizeof(create_request));
    create_request.node_id = "cpu0";
    create_request.node_target = "host.cpu";
    create_request.completion_function = SparkTestCompletion;
    create_request.completion_context = &completion_state;
    driver_instance = 0;
    assert(loaded_driver.interface->create(&create_request, &driver_instance) == SPARK_STATUS_OK);
    memset(&frame, 0, sizeof(frame));
    frame.request_id = 91u;
    frame.scalar[0] = 4u;
    assert(program->submit(driver_instance, &frame) == SPARK_STATUS_OK);
    assert(frame.scalar[0] == 17u);
    assert(completion_state.completion_count == 1u);
    assert(completion_state.completion.request_id == 91u);
    assert(completion_state.completion.status == SPARK_STATUS_OK);
    loaded_driver.interface->destroy(driver_instance);
    SparkUnloadModelDriver(&loaded_driver);

    assert(SparkTestPublishAffineArchiveModule(
               LibraryRoot,
               "host.cpu",
               CounterPath,
               &archive_publish_report) == SPARK_STATUS_OK);
    assert(archive_publish_report.validation_reused);
    assert(SparkTestReadCounter(CounterPath) == 4u);

    invalid_request = missing_request;
    invalid_request.output_directory = "build/test_driver_invalid";
    invalid_request.extra_compiler_argument_count = 1u;
    invalid_request.extra_compiler_arguments = 0;
    assert(SparkCompileModelDriver(&invalid_request, &second_compile_report, error_buffer, sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(strstr(error_buffer, "arguments are missing") != 0);
    assert(access("build/test_driver_invalid", F_OK) != 0);
    return 0;
}
