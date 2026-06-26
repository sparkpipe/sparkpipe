#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_module_library.h"
#include "test_support.h"

static void SparkTestCreateModifiedLinkUnit(const char *source_path, const char *destination_path)
{
    FILE *source;
    FILE *destination;
    unsigned char buffer[4096];
    size_t bytes_read;

    source = fopen(source_path, "rb");
    assert(source != 0);
    destination = fopen(destination_path, "wb");
    assert(destination != 0);
    while ((bytes_read = fread(buffer, 1u, sizeof(buffer), source)) != 0u)
    {
        assert(fwrite(buffer, 1u, bytes_read, destination) == bytes_read);
    }
    assert(ferror(source) == 0);
    assert(fputc(0, destination) != EOF);
    assert(fclose(source) == 0);
    assert(fclose(destination) == 0);
}

static void SparkTestCreateThinArchiveMarker(const char *path)
{
    static const char ThinArchiveMagic[] = "!<thin>\n";
    FILE *file;

    file = fopen(path, "wb");
    assert(file != 0);
    assert(fwrite(ThinArchiveMagic, 1u, sizeof(ThinArchiveMagic) - 1u, file) == sizeof(ThinArchiveMagic) - 1u);
    assert(fclose(file) == 0);
}

int main(void)
{
    static const char LibraryRoot[] = "build/test_module_library_store";
    static const char CounterPath[] = "build/test_module_library_validator_count.txt";
    static const char ValidationArgumentCounterPath[] =
        "build/test_module_library_validator_argument_count.txt";
    static const char ModifiedLinkUnitPath[] = "build/test_modules/module_add_one_modified.o";
    static const char ThinArchivePath[] = "build/test_modules/module_thin_archive.a";
    SparkModulePublishReport first_report;
    SparkModulePublishReport second_report;
    SparkModulePublishReport modified_report;
    SparkModulePublishReport archive_first_report;
    SparkModulePublishReport archive_second_report;
    SparkModulePublishReport validation_argument_first_report;
    SparkModulePublishReport validation_argument_second_report;
    SparkModulePublishReport validation_argument_changed_report;
    SparkModulePublishRequest modified_request;
    SparkModuleArtifact artifact;
    const char *validator_arguments[1];
    const char *validation_contract_arguments[2];
    char oversized_module_id[SPARK_MODULE_ID_BYTES + 1u];
    struct stat link_unit_status;
    char error_buffer[1024];

    assert(system(
               "rm -rf build/test_module_library_store "
               "build/test_module_library_validator_count.txt "
               "build/test_module_library_validator_argument_count.txt "
               "build/test_modules/module_add_one_modified.o "
               "build/test_modules/module_thin_archive.a") == 0);

    assert(SparkTestPublishAddOneModule(LibraryRoot, "host.cpu", CounterPath, &first_report) == SPARK_STATUS_OK);
    assert(!first_report.validation_reused);
    assert(first_report.link_unit_kind == SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT);
    assert(SparkTestReadCounter(CounterPath) == 1u);
    assert(stat(first_report.stored_link_unit_path, &link_unit_status) == 0);
    assert((link_unit_status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0);

    assert(SparkTestPublishAddOneModule(LibraryRoot, "host.cpu", CounterPath, &second_report) == SPARK_STATUS_OK);
    assert(second_report.validation_reused);
    assert(second_report.link_unit_kind == SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT);
    assert(strcmp(first_report.artifact_sha256, second_report.artifact_sha256) == 0);
    assert(SparkTestReadCounter(CounterPath) == 1u);

    SparkModuleArtifactReset(&artifact);
    assert(SparkResolveValidatedModule(
               LibraryRoot,
               "spark.test.add_one.v1",
               "host.cpu",
               &artifact,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(artifact.validated);
    assert(artifact.link_unit_kind == SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT);
    assert(strcmp(artifact.artifact_sha256, first_report.artifact_sha256) == 0);
    assert(strcmp(artifact.execute_symbol, "SparkTestAddOneExecute") == 0);

    SparkTestCreateModifiedLinkUnit("build/test_modules/module_add_one.o", ModifiedLinkUnitPath);
    memset(&modified_request, 0, sizeof(modified_request));
    validator_arguments[0] = CounterPath;
    modified_request.library_root = LibraryRoot;
    modified_request.module_id = "spark.test.add_one.v1";
    modified_request.target = "host.cpu";
    modified_request.module_abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    modified_request.link_unit_path = ModifiedLinkUnitPath;
    modified_request.validation_recipe = "test.module.validator.v1";
    modified_request.initialize_symbol = "SparkTestAddOneInitialize";
    modified_request.execute_symbol = "SparkTestAddOneExecute";
    modified_request.destroy_symbol = "SparkTestAddOneDestroy";
    modified_request.validator_path = "build/test_module_validator";
    modified_request.validator_arguments = validator_arguments;
    modified_request.validator_argument_count = 1u;
    assert(SparkPublishValidatedModule(
               &modified_request,
               &modified_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(!modified_report.validation_reused);
    assert(modified_report.link_unit_kind == SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT);
    assert(strcmp(modified_report.artifact_sha256, first_report.artifact_sha256) != 0);
    assert(SparkTestReadCounter(CounterPath) == 2u);

    assert(SparkTestPublishAffineArchiveModule(
               LibraryRoot,
               "host.cpu",
               CounterPath,
               &archive_first_report) == SPARK_STATUS_OK);
    assert(!archive_first_report.validation_reused);
    assert(archive_first_report.link_unit_kind == SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE);
    assert(strstr(archive_first_report.stored_link_unit_path, ".a") != 0);
    assert(SparkTestReadCounter(CounterPath) == 3u);
    assert(stat(archive_first_report.stored_link_unit_path, &link_unit_status) == 0);
    assert((link_unit_status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0);

    assert(SparkTestPublishAffineArchiveModule(
               LibraryRoot,
               "host.cpu",
               CounterPath,
               &archive_second_report) == SPARK_STATUS_OK);
    assert(archive_second_report.validation_reused);
    assert(archive_second_report.link_unit_kind == SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE);
    assert(strcmp(archive_first_report.artifact_sha256, archive_second_report.artifact_sha256) == 0);
    assert(SparkTestReadCounter(CounterPath) == 3u);

    SparkModuleArtifactReset(&artifact);
    assert(SparkResolveValidatedModule(
               LibraryRoot,
               "spark.test.affine.archive.v1",
               "host.cpu",
               &artifact,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(artifact.validated);
    assert(artifact.link_unit_kind == SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE);
    assert(strcmp(artifact.artifact_sha256, archive_first_report.artifact_sha256) == 0);
    assert(strcmp(artifact.execute_symbol, "SparkTestAffineArchiveExecute") == 0);

    memset(&modified_request, 0, sizeof(modified_request));
    validation_contract_arguments[0] = ValidationArgumentCounterPath;
    validation_contract_arguments[1] = "contract-a";
    modified_request.library_root = LibraryRoot;
    modified_request.module_id = "spark.test.validator.arguments.v1";
    modified_request.target = "host.cpu";
    modified_request.module_abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    modified_request.link_unit_path = "build/test_modules/module_add_one.o";
    modified_request.validation_recipe = "test.module.validator.arguments.v1";
    modified_request.initialize_symbol = "SparkTestAddOneInitialize";
    modified_request.execute_symbol = "SparkTestAddOneExecute";
    modified_request.destroy_symbol = "SparkTestAddOneDestroy";
    modified_request.validator_path = "build/test_module_validator";
    modified_request.validator_arguments = validation_contract_arguments;
    modified_request.validator_argument_count = 2u;
    assert(SparkPublishValidatedModule(
               &modified_request,
               &validation_argument_first_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(!validation_argument_first_report.validation_reused);
    assert(SparkTestReadCounter(ValidationArgumentCounterPath) == 1u);

    assert(SparkPublishValidatedModule(
               &modified_request,
               &validation_argument_second_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(validation_argument_second_report.validation_reused);
    assert(SparkTestReadCounter(ValidationArgumentCounterPath) == 1u);

    validation_contract_arguments[1] = "contract-b";
    assert(SparkPublishValidatedModule(
               &modified_request,
               &validation_argument_changed_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(!validation_argument_changed_report.validation_reused);
    assert(SparkTestReadCounter(ValidationArgumentCounterPath) == 2u);

    SparkTestCreateThinArchiveMarker(ThinArchivePath);
    modified_request.module_id = "spark.test.thin.archive.v1";
    modified_request.link_unit_path = ThinArchivePath;
    modified_request.initialize_symbol = "";
    modified_request.execute_symbol = "SparkTestAffineArchiveExecute";
    modified_request.destroy_symbol = "";
    assert(SparkPublishValidatedModule(
               &modified_request,
               &modified_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(strstr(error_buffer, "thin archive") != 0);
    assert(SparkTestReadCounter(CounterPath) == 3u);

    modified_request.module_id = "spark.test.invalid.arguments.v1";
    modified_request.link_unit_path = ModifiedLinkUnitPath;
    modified_request.initialize_symbol = "SparkTestAddOneInitialize";
    modified_request.execute_symbol = "SparkTestAddOneExecute";
    modified_request.destroy_symbol = "SparkTestAddOneDestroy";
    modified_request.validator_arguments = 0;
    modified_request.validator_argument_count = 1u;
    assert(SparkPublishValidatedModule(
               &modified_request,
               &modified_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);

    memset(oversized_module_id, 'm', sizeof(oversized_module_id) - 1u);
    oversized_module_id[sizeof(oversized_module_id) - 1u] = '\0';
    modified_request.module_id = oversized_module_id;
    modified_request.validator_argument_count = 0u;
    assert(SparkPublishValidatedModule(
               &modified_request,
               &modified_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_CAPACITY_EXCEEDED);

    modified_request.module_id = "spark.test.validation.failure.v1";
    modified_request.validator_path = "/bin/false";
    assert(SparkPublishValidatedModule(
               &modified_request,
               &modified_report,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_VALIDATION_FAILED);
    assert(SparkResolveValidatedModule(
               LibraryRoot,
               "spark.test.validation.failure.v1",
               "host.cpu",
               &artifact,
               error_buffer,
               sizeof(error_buffer)) == SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(SparkTestReadCounter(CounterPath) == 3u);
    return 0;
}
