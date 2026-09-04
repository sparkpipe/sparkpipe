#ifndef SPARKPIPE_SPARK_DRIVER_COMPILER_H
#define SPARKPIPE_SPARK_DRIVER_COMPILER_H

#include <stdint.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DRIVER_COMPILER_PATH_BYTES 1024u

typedef struct SparkDriverCompileRequest
{
    const char *model_description_path;
    const char *stage_name;
    const char *module_library_root;
    const char *output_directory;
    const char *compiler_path;
    const char *sparkpipe_include_directory;
    const char *const *extra_compiler_arguments;
    uint32_t extra_compiler_argument_count;
} SparkDriverCompileRequest;

typedef struct SparkDriverCompileReport
{
    uint32_t program_count;
    uint32_t operation_count;
    uint32_t unique_link_unit_count;
    char model_description_sha256[SPARK_SHA256_HEX_BYTES];
    char compiled_program_sha256[SPARK_SHA256_HEX_BYTES];
    char driver_sha256[SPARK_SHA256_HEX_BYTES];
    char generated_source_path[SPARK_DRIVER_COMPILER_PATH_BYTES];
    char driver_path[SPARK_DRIVER_COMPILER_PATH_BYTES];
    char manifest_path[SPARK_DRIVER_COMPILER_PATH_BYTES];
} SparkDriverCompileReport;

typedef struct SparkModelPackageCompileRequest
{
    const char *model_description_path;
    const char *module_library_root;
    const char *output_directory;
    const char *compiler_path;
    const char *sparkpipe_include_directory;
    const char *const *extra_compiler_arguments;
    uint32_t extra_compiler_argument_count;
} SparkModelPackageCompileRequest;

typedef struct SparkModelPackageCompileReport
{
    uint32_t stage_count;
    uint32_t total_program_count;
    uint32_t total_operation_count;
    uint32_t total_collected_link_unit_count;
    char model_description_sha256[SPARK_SHA256_HEX_BYTES];
    char package_manifest_sha256[SPARK_SHA256_HEX_BYTES];
    char package_manifest_path[SPARK_DRIVER_COMPILER_PATH_BYTES];
} SparkModelPackageCompileReport;

SparkStatus SparkCompileModelDriver(const SparkDriverCompileRequest *request, SparkDriverCompileReport *report, char *error_buffer, uint32_t error_buffer_bytes);
SparkStatus SparkCompileModelPackage(const SparkModelPackageCompileRequest *request, SparkModelPackageCompileReport *report, char *error_buffer, uint32_t error_buffer_bytes);

#ifdef __cplusplus
}
#endif

#endif
