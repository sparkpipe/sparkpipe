#ifndef SPARKPIPE_SPARK_MODULE_LIBRARY_H
#define SPARKPIPE_SPARK_MODULE_LIBRARY_H

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODULE_ARTIFACT_SCHEMA_VERSION 3u
#define SPARK_MODULE_SYMBOL_BYTES 192u
#define SPARK_MODULE_ID_BYTES 256u
#define SPARK_MODULE_TARGET_BYTES 128u
#define SPARK_MODULE_PATH_BYTES 1024u
#define SPARK_MODULE_RECIPE_BYTES 256u

typedef enum SparkModuleLinkUnitKind
{
    SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT = 1,
    SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE = 2
} SparkModuleLinkUnitKind;

typedef struct SparkModuleArtifact
{
    uint32_t schema_version;
    uint32_t module_abi_version;
    SparkModuleLinkUnitKind link_unit_kind;
    char module_id[SPARK_MODULE_ID_BYTES];
    char target[SPARK_MODULE_TARGET_BYTES];
    char artifact_sha256[SPARK_SHA256_HEX_BYTES];
    char link_unit_path[SPARK_MODULE_PATH_BYTES];
    char validation_recipe[SPARK_MODULE_RECIPE_BYTES];
    char initialize_symbol[SPARK_MODULE_SYMBOL_BYTES];
    char execute_symbol[SPARK_MODULE_SYMBOL_BYTES];
    char admit_symbol[SPARK_MODULE_SYMBOL_BYTES];
    char snapshot_symbol[SPARK_MODULE_SYMBOL_BYTES];
    char destroy_symbol[SPARK_MODULE_SYMBOL_BYTES];
    bool validated;
} SparkModuleArtifact;

typedef struct SparkModulePublishRequest
{
    const char *library_root;
    const char *module_id;
    const char *target;
    uint32_t module_abi_version;
    const char *link_unit_path;
    const char *validation_recipe;
    const char *initialize_symbol;
    const char *execute_symbol;
    const char *admit_symbol;
    const char *snapshot_symbol;
    const char *destroy_symbol;
    const char *validator_path;
    const char *const *validator_arguments;
    uint32_t validator_argument_count;
} SparkModulePublishRequest;

typedef struct SparkModulePublishReport
{
    bool validation_reused;
    SparkModuleLinkUnitKind link_unit_kind;
    char artifact_sha256[SPARK_SHA256_HEX_BYTES];
    char immutable_record_path[SPARK_MODULE_PATH_BYTES];
    char active_record_path[SPARK_MODULE_PATH_BYTES];
    char stored_link_unit_path[SPARK_MODULE_PATH_BYTES];
} SparkModulePublishReport;

const char *SparkModuleLinkUnitKindToString(SparkModuleLinkUnitKind link_unit_kind);
void SparkModuleArtifactReset(SparkModuleArtifact *artifact);
SparkStatus SparkPublishValidatedModule(const SparkModulePublishRequest *request, SparkModulePublishReport *report, char *error_buffer, uint32_t error_buffer_bytes);
SparkStatus SparkResolveValidatedModule(const char *library_root, const char *module_id, const char *target, SparkModuleArtifact *artifact, char *error_buffer, uint32_t error_buffer_bytes);
SparkStatus SparkLoadModuleArtifactRecord(const char *record_path, SparkModuleArtifact *artifact, char *error_buffer, uint32_t error_buffer_bytes);

#ifdef __cplusplus
}
#endif

#endif
