#include "sparkpipe/spark_driver_compiler.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_model_description.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_module_library.h"

#define SPARK_DRIVER_GENERATED_SOURCE_NAME "spark_model_driver_generated.c"
#define SPARK_DRIVER_SHARED_OBJECT_NAME "model_driver.so"
#define SPARK_DRIVER_MANIFEST_NAME "compiled_manifest.json"
#define SPARK_DRIVER_LINK_UNIT_DIRECTORY_NAME "link_units"
#define SPARK_MODEL_PACKAGE_MANIFEST_NAME "model_package.json"
#define SPARK_MODEL_PACKAGE_STAGE_DIRECTORY_NAME "stages"
#define SPARK_MODEL_PACKAGE_SCHEMA_VERSION 3u
#define SPARK_DRIVER_GENERATOR_ID "sparkpipe.driver.generator.v3"
#if defined(__APPLE__)
#define SPARK_DRIVER_FIXED_LINK_CONTRACT "-std=c11;-O3;-fPIC;-fvisibility=hidden;-fno-semantic-interposition;-dynamiclib;-Wl,-undefined,error;-Wl,-exported_symbol,_SparkModelDriverGetInterface"
#else
#define SPARK_DRIVER_FIXED_LINK_CONTRACT "-std=c11;-O3;-fPIC;-fvisibility=hidden;-fno-semantic-interposition;-shared;-Wl,-z,defs;-Wl,-O1;-Wl,--exclude-libs,ALL"
#endif

typedef struct SparkDriverBuildOperation
{
    const SparkModelProgramDescription *program;
    const SparkModelOperationDescription *operation;
    SparkModuleArtifact artifact;
    uint32_t operation_index;
} SparkDriverBuildOperation;

typedef struct SparkDriverBuildImage
{
    const SparkModelStageDescription *stage;
    SparkDriverBuildOperation *operations;
    uint32_t operation_count;
    char **unique_link_unit_paths;
    char **unique_link_unit_hashes;
    SparkModuleLinkUnitKind *unique_link_unit_kinds;
    uint32_t unique_link_unit_count;
    char compiled_program_sha256[SPARK_SHA256_HEX_BYTES];
} SparkDriverBuildImage;

static void SparkDriverBuildImageReset(SparkDriverBuildImage *driver_image)
{
    if (driver_image == 0)
    {
        return;
    }
    memset(driver_image, 0, sizeof(*driver_image));
}

static void SparkDriverBuildImageDestroy(SparkDriverBuildImage *driver_image)
{
    uint32_t link_unit_index;

    if (driver_image == 0)
    {
        return;
    }
    for (link_unit_index = 0u; link_unit_index < driver_image->unique_link_unit_count; ++link_unit_index)
    {
        free(driver_image->unique_link_unit_paths[link_unit_index]);
        free(driver_image->unique_link_unit_hashes[link_unit_index]);
    }
    free(driver_image->unique_link_unit_paths);
    free(driver_image->unique_link_unit_hashes);
    free(driver_image->unique_link_unit_kinds);
    free(driver_image->operations);
    SparkDriverBuildImageReset(driver_image);
}

static char *SparkDuplicateText(const char *text)
{
    size_t text_bytes;
    char *copy;

    if (text == 0)
    {
        return 0;
    }
    text_bytes = strlen(text);
    copy = (char *)malloc(text_bytes + 1u);
    if (copy == 0)
    {
        return 0;
    }
    memcpy(copy, text, text_bytes + 1u);
    return copy;
}

static uint32_t SparkCountStageOperations(const SparkModelStageDescription *stage)
{
    uint32_t program_index;
    uint32_t operation_count;

    operation_count = 0u;
    for (program_index = 0u; program_index < stage->program_count; ++program_index)
    {
        if (UINT32_MAX - operation_count < stage->programs[program_index].operation_count)
        {
            return UINT32_MAX;
        }
        operation_count += stage->programs[program_index].operation_count;
    }
    return operation_count;
}

static SparkStatus SparkDriverBuildImageAddUniqueLinkUnit(
    SparkDriverBuildImage *driver_image,
    const SparkModuleArtifact *artifact)
{
    uint32_t link_unit_index;

    for (link_unit_index = 0u; link_unit_index < driver_image->unique_link_unit_count; ++link_unit_index)
    {
        if (strcmp(driver_image->unique_link_unit_hashes[link_unit_index], artifact->artifact_sha256) == 0)
        {
            if (strcmp(driver_image->unique_link_unit_paths[link_unit_index], artifact->link_unit_path) != 0 ||
                driver_image->unique_link_unit_kinds[link_unit_index] != artifact->link_unit_kind)
            {
                return SPARK_STATUS_HASH_MISMATCH;
            }
            return SPARK_STATUS_OK;
        }
    }

    driver_image->unique_link_unit_paths[driver_image->unique_link_unit_count] = SparkDuplicateText(artifact->link_unit_path);
    driver_image->unique_link_unit_hashes[driver_image->unique_link_unit_count] = SparkDuplicateText(artifact->artifact_sha256);
    driver_image->unique_link_unit_kinds[driver_image->unique_link_unit_count] = artifact->link_unit_kind;
    if (driver_image->unique_link_unit_paths[driver_image->unique_link_unit_count] == 0 || driver_image->unique_link_unit_hashes[driver_image->unique_link_unit_count] == 0)
    {
        free(driver_image->unique_link_unit_paths[driver_image->unique_link_unit_count]);
        free(driver_image->unique_link_unit_hashes[driver_image->unique_link_unit_count]);
        driver_image->unique_link_unit_paths[driver_image->unique_link_unit_count] = 0;
        driver_image->unique_link_unit_hashes[driver_image->unique_link_unit_count] = 0;
        driver_image->unique_link_unit_kinds[driver_image->unique_link_unit_count] = 0;
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    driver_image->unique_link_unit_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkHashTextField(SparkSha256Context *hash_context, const char *text)
{
    const char *value;

    value = text != 0 ? text : "";
    SparkSha256Update(hash_context, value, strlen(value));
    SparkSha256Update(hash_context, "\0", 1u);
}

static void SparkHashUInt32Field(SparkSha256Context *hash_context, uint32_t value)
{
    uint8_t encoded[4];

    encoded[0] = (uint8_t)(value >> 24u);
    encoded[1] = (uint8_t)(value >> 16u);
    encoded[2] = (uint8_t)(value >> 8u);
    encoded[3] = (uint8_t)value;
    SparkSha256Update(hash_context, encoded, sizeof(encoded));
}

static void SparkHashUInt64Field(SparkSha256Context *hash_context, uint64_t value)
{
    uint8_t encoded[8];
    uint32_t byte_index;

    for (byte_index = 0u; byte_index < 8u; ++byte_index)
    {
        encoded[byte_index] = (uint8_t)(value >> ((7u - byte_index) * 8u));
    }
    SparkSha256Update(hash_context, encoded, sizeof(encoded));
}

static SparkStatus SparkComputeCompiledProgramHash(
    const SparkDriverCompileRequest *request,
    const SparkModelDescription *description,
    const SparkDriverBuildImage *driver_image,
    char compiled_program_sha256[SPARK_SHA256_HEX_BYTES])
{
    SparkSha256Context hash_context;
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
    uint32_t extra_argument_index;
    uint32_t operation_index;

    if (request == 0 || description == 0 || driver_image == 0 || compiled_program_sha256 == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkSha256Initialize(&hash_context);
    SparkHashTextField(&hash_context, SPARK_DRIVER_GENERATOR_ID);
    SparkHashTextField(&hash_context, SPARK_DRIVER_FIXED_LINK_CONTRACT);
    SparkHashUInt32Field(&hash_context, SPARK_MODEL_DRIVER_ABI_VERSION);
    SparkHashUInt32Field(&hash_context, SPARK_FIRMWARE_MODULE_ABI_VERSION);
    SparkHashTextField(&hash_context, request->compiler_path);
    SparkHashTextField(&hash_context, request->sparkpipe_include_directory);
    SparkHashUInt32Field(&hash_context, request->extra_compiler_argument_count);
    for (extra_argument_index = 0u; extra_argument_index < request->extra_compiler_argument_count;
         ++extra_argument_index)
    {
        SparkHashTextField(&hash_context, request->extra_compiler_arguments[extra_argument_index]);
    }
    SparkHashTextField(&hash_context, description->source_sha256);
    SparkHashTextField(&hash_context, description->model_id);
    SparkHashTextField(&hash_context, description->model_revision);
    SparkHashTextField(&hash_context, driver_image->stage->name);
    SparkHashTextField(&hash_context, driver_image->stage->target);
    for (operation_index = 0u; operation_index < driver_image->operation_count; ++operation_index)
    {
        const SparkDriverBuildOperation *resolved_operation;

        resolved_operation = &driver_image->operations[operation_index];
        SparkHashTextField(&hash_context, resolved_operation->program->name);
        SparkHashUInt32Field(&hash_context, resolved_operation->program->program_id);
        SparkHashUInt32Field(&hash_context, resolved_operation->program->max_inflight);
        SparkHashUInt32Field(&hash_context, (uint32_t)resolved_operation->program->completion_mode);
        SparkHashUInt32Field(&hash_context, resolved_operation->program->scheduling.flags);
        SparkHashUInt32Field(&hash_context, resolved_operation->program->scheduling.max_active_slots);
        SparkHashUInt32Field(&hash_context, resolved_operation->program->scheduling.max_new_tokens);
        SparkHashUInt32Field(&hash_context, resolved_operation->program->scheduling.max_resident_sequences);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.max_sequence_tokens);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.target_latency_ns);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.validated_latency_ns);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.resident_weight_bytes);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.resident_kv_bytes);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.static_workspace_bytes);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.device_memcpy_bytes_per_submit_ceiling);
        SparkHashUInt64Field(&hash_context, resolved_operation->program->scheduling.host_staging_bytes_per_submit_ceiling);
        SparkHashUInt32Field(&hash_context, resolved_operation->program->scheduling.private_queue_count);
        SparkHashTextField(&hash_context, resolved_operation->operation->name);
        SparkHashTextField(&hash_context, resolved_operation->operation->module_id);
        SparkHashTextField(&hash_context, resolved_operation->artifact.target);
        SparkHashTextField(&hash_context, resolved_operation->artifact.artifact_sha256);
        SparkHashUInt32Field(&hash_context, (uint32_t)resolved_operation->artifact.link_unit_kind);
        SparkHashUInt32Field(&hash_context, resolved_operation->artifact.module_abi_version);
        SparkHashTextField(&hash_context, resolved_operation->artifact.validation_recipe);
        SparkHashTextField(&hash_context, resolved_operation->artifact.initialize_symbol);
        SparkHashTextField(&hash_context, resolved_operation->artifact.execute_symbol);
        SparkHashTextField(&hash_context, resolved_operation->artifact.admit_symbol);
        SparkHashTextField(&hash_context, resolved_operation->artifact.snapshot_symbol);
        SparkHashTextField(&hash_context, resolved_operation->artifact.destroy_symbol);
        SparkHashUInt32Field(&hash_context, resolved_operation->operation->configuration_json_bytes);
        SparkSha256Update(
            &hash_context,
            resolved_operation->operation->configuration_json,
            resolved_operation->operation->configuration_json_bytes);
    }
    SparkSha256Finalize(&hash_context, digest);
    SparkSha256DigestToHex(digest, compiled_program_sha256);
    return SPARK_STATUS_OK;
}

static const SparkModuleArtifact *SparkFindPreviouslyResolvedModule(
    const SparkDriverBuildImage *driver_image,
    uint32_t resolved_operation_count,
    const char *module_id)
{
    uint32_t operation_index;

    for (operation_index = 0u; operation_index < resolved_operation_count; ++operation_index)
    {
        const SparkDriverBuildOperation *resolved_operation;

        resolved_operation = &driver_image->operations[operation_index];
        if (strcmp(resolved_operation->operation->module_id, module_id) == 0)
        {
            return &resolved_operation->artifact;
        }
    }
    return 0;
}

static SparkStatus SparkResolveDriverBuildImage(
    const SparkModelStageDescription *stage,
    const char *module_library_root,
    SparkDriverBuildImage *driver_image,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t program_index;
    uint32_t operation_index;
    uint32_t flattened_operation_index;
    SparkStatus status;

    SparkDriverBuildImageReset(driver_image);
    driver_image->stage = stage;
    driver_image->operation_count = SparkCountStageOperations(stage);
    if (driver_image->operation_count == 0u || driver_image->operation_count == UINT32_MAX)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "stage '%s' has an invalid operation count", stage->name);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    driver_image->operations = (SparkDriverBuildOperation *)calloc(driver_image->operation_count, sizeof(*driver_image->operations));
    driver_image->unique_link_unit_paths = (char **)calloc(driver_image->operation_count, sizeof(*driver_image->unique_link_unit_paths));
    driver_image->unique_link_unit_hashes = (char **)calloc(driver_image->operation_count, sizeof(*driver_image->unique_link_unit_hashes));
    driver_image->unique_link_unit_kinds = (SparkModuleLinkUnitKind *)calloc(driver_image->operation_count, sizeof(*driver_image->unique_link_unit_kinds));
    if (driver_image->operations == 0 || driver_image->unique_link_unit_paths == 0 ||
        driver_image->unique_link_unit_hashes == 0 || driver_image->unique_link_unit_kinds == 0)
    {
        SparkDriverBuildImageDestroy(driver_image);
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    flattened_operation_index = 0u;
    for (program_index = 0u; program_index < stage->program_count; ++program_index)
    {
        const SparkModelProgramDescription *program;

        program = &stage->programs[program_index];
        for (operation_index = 0u; operation_index < program->operation_count; ++operation_index)
        {
            SparkDriverBuildOperation *resolved_operation;
            const SparkModuleArtifact *previously_resolved_artifact;

            resolved_operation = &driver_image->operations[flattened_operation_index];
            resolved_operation->program = program;
            resolved_operation->operation = &program->operations[operation_index];
            resolved_operation->operation_index = flattened_operation_index;
            previously_resolved_artifact = SparkFindPreviouslyResolvedModule(
                driver_image,
                flattened_operation_index,
                resolved_operation->operation->module_id);
            if (previously_resolved_artifact != 0)
            {
                resolved_operation->artifact = *previously_resolved_artifact;
            }
            else
            {
                status = SparkResolveValidatedModule(
                    module_library_root,
                    resolved_operation->operation->module_id,
                    stage->target,
                    &resolved_operation->artifact,
                    error_buffer,
                    error_buffer_bytes);
                if (status != SPARK_STATUS_OK)
                {
                    SparkDriverBuildImageDestroy(driver_image);
                    return status;
                }
            }
            if (resolved_operation->artifact.module_abi_version != SPARK_FIRMWARE_MODULE_ABI_VERSION)
            {
                SparkSetError(
                    error_buffer,
                    error_buffer_bytes,
                    "module '%s' ABI %u does not match firmware module ABI %u",
                    resolved_operation->operation->module_id,
                    resolved_operation->artifact.module_abi_version,
                    SPARK_FIRMWARE_MODULE_ABI_VERSION);
                SparkDriverBuildImageDestroy(driver_image);
                return SPARK_STATUS_ABI_MISMATCH;
            }
            status = SparkDriverBuildImageAddUniqueLinkUnit(driver_image, &resolved_operation->artifact);
            if (status != SPARK_STATUS_OK)
            {
                SparkSetError(error_buffer, error_buffer_bytes, "cannot collect module link unit '%s'", resolved_operation->operation->module_id);
                SparkDriverBuildImageDestroy(driver_image);
                return status;
            }
            flattened_operation_index += 1u;
        }
    }

    return SPARK_STATUS_OK;
}

static void SparkWriteCStringLiteral(FILE *file, const char *text, size_t text_bytes)
{
    size_t character_index;

    fputc('"', file);
    for (character_index = 0u; character_index < text_bytes; ++character_index)
    {
        unsigned char character;

        character = (unsigned char)text[character_index];
        switch (character)
        {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            case '\b': fputs("\\b", file); break;
            case '\f': fputs("\\f", file); break;
            default:
            {
                if (character >= 0x20u && character <= 0x7eu)
                {
                    fputc((int)character, file);
                }
                else
                {
                    fprintf(file, "\\%03o", character);
                }
                break;
            }
        }
    }
    fputc('"', file);
}

static void SparkWriteGeneratedExterns(FILE *file, const SparkDriverBuildImage *driver_image)
{
    uint32_t operation_index;

    fputs(
        "#if defined(_WIN32)\n"
        "#define SPARK_GENERATED_MODULE_HIDDEN\n"
        "#else\n"
        "#define SPARK_GENERATED_MODULE_HIDDEN __attribute__((visibility(\"hidden\")))\n"
        "#endif\n\n",
        file);
    for (operation_index = 0u; operation_index < driver_image->operation_count; ++operation_index)
    {
        const SparkModuleArtifact *artifact;

        artifact = &driver_image->operations[operation_index].artifact;
        if (artifact->initialize_symbol[0] != '\0')
        {
            fprintf(
                file,
                "extern SPARK_GENERATED_MODULE_HIDDEN SparkStatus %s(const SparkFirmwareModuleConfiguration *, const SparkFirmwareModuleHostServices *, void **);\n",
                artifact->initialize_symbol);
        }
        fprintf(file, "extern SPARK_GENERATED_MODULE_HIDDEN SparkStatus %s(void *, SparkModelDriverFrame *);\n", artifact->execute_symbol);
        if (artifact->admit_symbol[0] != '\0')
        {
            fprintf(file, "extern SPARK_GENERATED_MODULE_HIDDEN SparkStatus %s(void *, const SparkModelDriverAdmissionRequest *, SparkModelDriverAdmissionDecision *);\n", artifact->admit_symbol);
        }
        if (artifact->snapshot_symbol[0] != '\0')
        {
            fprintf(file, "extern SPARK_GENERATED_MODULE_HIDDEN SparkStatus %s(void *, uint32_t, SparkModelDriverRuntimeSnapshot *);\n", artifact->snapshot_symbol);
        }
        if (artifact->destroy_symbol[0] != '\0')
        {
            fprintf(file, "extern SPARK_GENERATED_MODULE_HIDDEN void %s(void *);\n", artifact->destroy_symbol);
        }
    }
    fputs("\n#undef SPARK_GENERATED_MODULE_HIDDEN\n\n", file);
}

static void SparkWriteGeneratedConfigurations(
    FILE *file,
    const SparkModelDescription *description,
    const SparkDriverBuildImage *driver_image)
{
    uint32_t operation_index;

    for (operation_index = 0u; operation_index < driver_image->operation_count; ++operation_index)
    {
        const SparkDriverBuildOperation *resolved_operation;

        resolved_operation = &driver_image->operations[operation_index];
        fprintf(file, "static const char SparkGeneratedConfigurationJson_%u[] = ", operation_index);
        SparkWriteCStringLiteral(
            file,
            resolved_operation->operation->configuration_json,
            resolved_operation->operation->configuration_json_bytes);
        fputs(";\n", file);
        fprintf(file, "static const SparkFirmwareModuleConfiguration SparkGeneratedConfiguration_%u =\n{\n", operation_index);
        fprintf(file, "    SPARK_FIRMWARE_MODULE_ABI_VERSION,\n    %uu,\n    ", operation_index);
        SparkWriteCStringLiteral(file, description->model_id, strlen(description->model_id));
        fputs(",\n    ", file);
        SparkWriteCStringLiteral(file, description->model_revision, strlen(description->model_revision));
        fputs(",\n    ", file);
        SparkWriteCStringLiteral(file, driver_image->stage->name, strlen(driver_image->stage->name));
        fputs(",\n    ", file);
        SparkWriteCStringLiteral(file, resolved_operation->program->name, strlen(resolved_operation->program->name));
        fputs(",\n    ", file);
        SparkWriteCStringLiteral(file, resolved_operation->operation->name, strlen(resolved_operation->operation->name));
        fprintf(
            file,
            ",\n    SparkGeneratedConfigurationJson_%u,\n    %uu,\n    0u\n};\n\n",
            operation_index,
            resolved_operation->operation->configuration_json_bytes);
    }
}

static void SparkWriteGeneratedInstance(FILE *file, const SparkDriverBuildImage *driver_image)
{
    uint32_t operation_index;

    fputs("typedef struct SparkGeneratedDriverInstance\n{\n", file);
    fputs("    SparkModelDriverCompletionFunction completion_function;\n", file);
    fputs("    void *completion_context;\n", file);
    fputs("    SparkFirmwareModuleHostServices host_services;\n", file);
    for (operation_index = 0u; operation_index < driver_image->operation_count; ++operation_index)
    {
        fprintf(file, "    void *operation_%u_state;\n", operation_index);
    }
    fputs("} SparkGeneratedDriverInstance;\n\n", file);
}

static void SparkWriteGeneratedDestroy(FILE *file, const SparkDriverBuildImage *driver_image)
{
    uint32_t reverse_index;

    fputs("static void SparkGeneratedDriverDestroy(void *driver_instance)\n{\n", file);
    fputs("    SparkGeneratedDriverInstance *instance;\n\n", file);
    fputs("    instance = (SparkGeneratedDriverInstance *)driver_instance;\n", file);
    fputs("    if (instance == 0)\n    {\n        return;\n    }\n", file);
    reverse_index = driver_image->operation_count;
    while (reverse_index != 0u)
    {
        const SparkModuleArtifact *artifact;
        uint32_t operation_index;

        operation_index = reverse_index - 1u;
        artifact = &driver_image->operations[operation_index].artifact;
        if (artifact->destroy_symbol[0] != '\0')
        {
            fprintf(file, "    if (instance->operation_%u_state != 0)\n    {\n", operation_index);
            fprintf(file, "        %s(instance->operation_%u_state);\n", artifact->destroy_symbol, operation_index);
            fputs("    }\n", file);
        }
        reverse_index -= 1u;
    }
    fputs("    free(instance);\n}\n\n", file);
}

static void SparkWriteGeneratedCreate(FILE *file, const SparkDriverBuildImage *driver_image)
{
    uint32_t operation_index;

    fputs("static SparkStatus SparkGeneratedDriverCreate(const SparkModelDriverCreateRequest *request, void **driver_instance)\n{\n", file);
    fputs("    SparkGeneratedDriverInstance *instance;\n    SparkStatus status;\n\n", file);
    fputs("    if (request == 0 || driver_instance == 0)\n    {\n        return SPARK_STATUS_INVALID_ARGUMENT;\n    }\n", file);
    fputs("    *driver_instance = 0;\n", file);
    fputs("    instance = (SparkGeneratedDriverInstance *)calloc(1u, sizeof(*instance));\n", file);
    fputs("    if (instance == 0)\n    {\n        return SPARK_STATUS_INTERNAL_ERROR;\n    }\n", file);
    fputs("    instance->completion_function = request->completion_function;\n", file);
    fputs("    instance->completion_context = request->completion_context;\n", file);
    fputs("    instance->host_services.completion_function = request->completion_function;\n", file);
    fputs("    instance->host_services.completion_context = request->completion_context;\n", file);
    fputs("    instance->host_services.node_id = request->node_id;\n", file);
    fputs("    instance->host_services.node_target = request->node_target;\n", file);
    fputs("    instance->host_services.node_context = request->node_context;\n", file);
    fputs("    status = SPARK_STATUS_OK;\n", file);

    for (operation_index = 0u; operation_index < driver_image->operation_count; ++operation_index)
    {
        const SparkModuleArtifact *artifact;

        artifact = &driver_image->operations[operation_index].artifact;
        if (artifact->initialize_symbol[0] != '\0')
        {
            fprintf(
                file,
                "    status = %s(&SparkGeneratedConfiguration_%u, &instance->host_services, &instance->operation_%u_state);\n",
                artifact->initialize_symbol,
                operation_index,
                operation_index);
            fputs("    if (status != SPARK_STATUS_OK)\n    {\n        SparkGeneratedDriverDestroy(instance);\n        return status;\n    }\n", file);
        }
    }
    fputs("    *driver_instance = instance;\n    return SPARK_STATUS_OK;\n}\n\n", file);
}

static uint32_t SparkFindFlattenedProgramStart(
    const SparkDriverBuildImage *driver_image,
    const SparkModelProgramDescription *program)
{
    uint32_t operation_index;

    for (operation_index = 0u; operation_index < driver_image->operation_count; ++operation_index)
    {
        if (driver_image->operations[operation_index].program == program)
        {
            return operation_index;
        }
    }
    return driver_image->operation_count;
}

static void SparkWriteGeneratedProgramFunction(
    FILE *file,
    const SparkDriverBuildImage *driver_image,
    uint32_t program_index)
{
    const SparkModelProgramDescription *program;
    uint32_t flattened_operation_index;
    uint32_t program_operation_index;

    program = &driver_image->stage->programs[program_index];
    flattened_operation_index = SparkFindFlattenedProgramStart(driver_image, program);
    fprintf(file, "static SparkStatus SparkGeneratedSubmitProgram_%u(void *driver_instance, SparkModelDriverFrame *frame)\n{\n", program_index);
    fputs("    SparkGeneratedDriverInstance *instance;\n    SparkStatus execution_status;\n\n", file);
    fputs("    if (driver_instance == 0 || frame == 0)\n    {\n        return SPARK_STATUS_INVALID_ARGUMENT;\n    }\n", file);
    fputs("    instance = (SparkGeneratedDriverInstance *)driver_instance;\n", file);
    fprintf(file, "    frame->program_id = %uu;\n", program->program_id);
    fputs("    frame->completion_function = instance->completion_function;\n", file);
    fputs("    frame->completion_context = instance->completion_context;\n", file);
    fputs("    execution_status = SPARK_STATUS_OK;\n", file);

    for (program_operation_index = 0u; program_operation_index < program->operation_count; ++program_operation_index)
    {
        const SparkDriverBuildOperation *resolved_operation;
        uint32_t operation_index;

        operation_index = flattened_operation_index + program_operation_index;
        resolved_operation = &driver_image->operations[operation_index];
        fprintf(
            file,
            "    execution_status = %s(instance->operation_%u_state, frame);\n",
            resolved_operation->artifact.execute_symbol,
            operation_index);
        fputs("    if (execution_status != SPARK_STATUS_OK)\n    {\n        goto complete;\n    }\n", file);
    }
    fputs("\ncomplete:\n", file);
    if (program->completion_mode == SPARK_MODEL_PROGRAM_COMPLETION_SUBMIT_RETURN)
    {
        fputs("    if (instance->completion_function != 0)\n    {\n", file);
        fputs("        SparkModelDriverCompletion completion;\n\n", file);
        fputs("        memset(&completion, 0, sizeof(completion));\n", file);
        fputs("        completion.request_id = frame->request_id;\n", file);
        fputs("        completion.sequence_id = frame->sequence_id;\n", file);
        fputs("        completion.sequence_position = frame->sequence_position;\n", file);
        fputs("        completion.driver_dispatch_slot = frame->driver_dispatch_slot;\n", file);
        fputs("        completion.accepted_token_count = frame->new_token_count;\n", file);
        fputs("        completion.residency = frame->residency;\n", file);
        fprintf(file, "        completion.program_id = %uu;\n", program->program_id);
        fputs("        completion.status = execution_status;\n", file);
        fputs("        instance->completion_function(instance->completion_context, &completion);\n", file);
        fputs("        return SPARK_STATUS_OK;\n    }\n", file);
        fputs("    return execution_status;\n}\n\n", file);
    }
    else
    {
        fputs("    return execution_status;\n}\n\n", file);
    }
}


static void SparkWriteGeneratedAdmissionHelpers(FILE *file)
{
    fputs("static void SparkGeneratedInitializeAdmissionDecision(SparkModelDriverAdmissionDecision *decision)\n{\n", file);
    fputs("    memset(decision, 0, sizeof(*decision));\n", file);
    fputs("    decision->descriptor_bytes = sizeof(*decision);\n", file);
    fputs("    decision->accepted = 1u;\n", file);
    fputs("    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;\n", file);
    fputs("    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;\n", file);
    fputs("}\n\n", file);

    fputs("static SparkStatus SparkGeneratedRejectAdmission(SparkModelDriverAdmissionDecision *decision, uint32_t reason)\n{\n", file);
    fputs("    SparkGeneratedInitializeAdmissionDecision(decision);\n", file);
    fputs("    decision->accepted = 0u;\n", file);
    fputs("    decision->rejection_reason = reason;\n", file);
    fputs("    return SPARK_STATUS_OK;\n", file);
    fputs("}\n\n", file);

    fputs("static void SparkGeneratedMergeAdmissionDecision(SparkModelDriverAdmissionDecision *destination, const SparkModelDriverAdmissionDecision *source)\n{\n", file);
    fputs("    if (source->driver_dispatch_slot != SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT &&\n", file);
    fputs("        destination->driver_dispatch_slot == SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)\n    {\n", file);
    fputs("        destination->driver_dispatch_slot = source->driver_dispatch_slot;\n", file);
    fputs("        destination->driver_dispatch_generation = source->driver_dispatch_generation;\n", file);
    fputs("        destination->driver_dispatch_cookie0 = source->driver_dispatch_cookie0;\n", file);
    fputs("        destination->driver_dispatch_cookie1 = source->driver_dispatch_cookie1;\n    }\n", file);
    fputs("    if (source->estimated_queue_delay_ns > destination->estimated_queue_delay_ns)\n    {\n", file);
    fputs("        destination->estimated_queue_delay_ns = source->estimated_queue_delay_ns;\n    }\n", file);
    fputs("    if (source->estimated_service_time_ns > destination->estimated_service_time_ns)\n    {\n", file);
    fputs("        destination->estimated_service_time_ns = source->estimated_service_time_ns;\n    }\n", file);
    fputs("    destination->endpoint_cost += source->endpoint_cost;\n", file);
    fputs("    destination->residency_match_score += source->residency_match_score;\n", file);
    fputs("    destination->device_memcpy_bytes += source->device_memcpy_bytes;\n", file);
    fputs("    destination->host_staging_bytes += source->host_staging_bytes;\n", file);
    fputs("    if (source->private_queue_pressure > destination->private_queue_pressure)\n    {\n", file);
    fputs("        destination->private_queue_pressure = source->private_queue_pressure;\n    }\n", file);
    fputs("    if (source->available_dispatch_slot_count > destination->available_dispatch_slot_count)\n    {\n", file);
    fputs("        destination->available_dispatch_slot_count = source->available_dispatch_slot_count;\n    }\n", file);
    fputs("}\n\n", file);

    fputs("static void SparkGeneratedMergeRuntimeSnapshot(SparkModelDriverRuntimeSnapshot *destination, const SparkModelDriverRuntimeSnapshot *source)\n{\n", file);
    fputs("    destination->active_submission_count += source->active_submission_count;\n", file);
    fputs("    destination->available_dispatch_slot_count += source->available_dispatch_slot_count;\n", file);
    fputs("    destination->submitted_count += source->submitted_count;\n", file);
    fputs("    destination->completed_count += source->completed_count;\n", file);
    fputs("    destination->rejected_count += source->rejected_count;\n", file);
    fputs("    destination->resident_sequence_count += source->resident_sequence_count;\n", file);
    fputs("    destination->resident_token_count += source->resident_token_count;\n", file);
    fputs("    destination->kv_token_capacity += source->kv_token_capacity;\n", file);
    fputs("    destination->device_memcpy_bytes_per_submit += source->device_memcpy_bytes_per_submit;\n", file);
    fputs("    destination->host_staging_bytes_per_submit += source->host_staging_bytes_per_submit;\n", file);
    fputs("    destination->cuda_graph_capture_count += source->cuda_graph_capture_count;\n", file);
    fputs("    destination->cuda_graph_replay_count += source->cuda_graph_replay_count;\n", file);
    fputs("    destination->host_callback_completion_count += source->host_callback_completion_count;\n", file);
    fputs("    destination->stale_admission_count += source->stale_admission_count;\n", file);
    fputs("    if (source->private_queue_pressure > destination->private_queue_pressure)\n    {\n", file);
    fputs("        destination->private_queue_pressure = source->private_queue_pressure;\n    }\n", file);
    fputs("}\n\n", file);
}

static void SparkWriteGeneratedAdmitFunction(
    FILE *file,
    const SparkDriverBuildImage *driver_image)
{
    uint32_t program_index;

    SparkWriteGeneratedAdmissionHelpers(file);
    fputs("static SparkStatus SparkGeneratedDriverAdmit(void *driver_instance, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision)\n{\n", file);
    fputs("    SparkGeneratedDriverInstance *instance;\n\n", file);
    fputs("    if (driver_instance == 0 || request == 0 || decision == 0 || request->descriptor_bytes < sizeof(*request))\n    {\n", file);
    fputs("        return SPARK_STATUS_INVALID_ARGUMENT;\n    }\n", file);
    fputs("    instance = (SparkGeneratedDriverInstance *)driver_instance;\n", file);
    fputs("    SparkGeneratedInitializeAdmissionDecision(decision);\n", file);
    fputs("    switch (request->program_id)\n    {\n", file);
    for (program_index = 0u; program_index < driver_image->stage->program_count; ++program_index)
    {
        const SparkModelProgramDescription *program;
        uint32_t flattened_operation_index;
        uint32_t program_operation_index;

        program = &driver_image->stage->programs[program_index];
        flattened_operation_index = SparkFindFlattenedProgramStart(driver_image, program);
        fprintf(file, "        case %uu:\n        {\n", program->program_id);
        fputs("            SparkStatus status;\n", file);
        if (program->scheduling.max_active_slots != 0u)
        {
            fputs("            if (request->active_slot_count == 0u)\n            {\n", file);
            fputs("                return SparkGeneratedRejectAdmission(decision, SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);\n            }\n", file);
            fprintf(file, "            if (request->active_slot_count > %uu)\n            {\n", program->scheduling.max_active_slots);
            fputs("                return SparkGeneratedRejectAdmission(decision, SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);\n            }\n", file);
        }
        if (program->scheduling.max_new_tokens != 0u)
        {
            fprintf(file, "            if (request->new_token_count > %uu)\n            {\n", program->scheduling.max_new_tokens);
            fputs("                return SparkGeneratedRejectAdmission(decision, SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);\n            }\n", file);
        }
        fprintf(file, "            decision->estimated_service_time_ns = %lluull;\n", (unsigned long long)program->scheduling.target_latency_ns);
        fprintf(file, "            decision->device_memcpy_bytes = %lluull;\n", (unsigned long long)program->scheduling.device_memcpy_bytes_per_submit_ceiling);
        fprintf(file, "            decision->host_staging_bytes = %lluull;\n", (unsigned long long)program->scheduling.host_staging_bytes_per_submit_ceiling);
        for (program_operation_index = 0u; program_operation_index < program->operation_count; ++program_operation_index)
        {
            const SparkDriverBuildOperation *resolved_operation;
            uint32_t operation_index;

            operation_index = flattened_operation_index + program_operation_index;
            resolved_operation = &driver_image->operations[operation_index];
            if (resolved_operation->artifact.admit_symbol[0] != '\0')
            {
                fputs("            {\n", file);
                fputs("                SparkModelDriverAdmissionDecision module_decision;\n", file);
                fprintf(file, "                status = %s(instance->operation_%u_state, request, &module_decision);\n", resolved_operation->artifact.admit_symbol, operation_index);
                fputs("                if (status != SPARK_STATUS_OK)\n                {\n                    return status;\n                }\n", file);
                fputs("                if (module_decision.descriptor_bytes < sizeof(module_decision))\n                {\n                    return SPARK_STATUS_ABI_MISMATCH;\n                }\n", file);
                fputs("                if (module_decision.accepted == 0u)\n                {\n                    *decision = module_decision;\n                    return SPARK_STATUS_OK;\n                }\n", file);
                fputs("                SparkGeneratedMergeAdmissionDecision(decision, &module_decision);\n", file);
                fputs("            }\n", file);
            }
        }
        fputs("            return SPARK_STATUS_OK;\n        }\n", file);
    }
    fputs("        default:\n        {\n            return SPARK_STATUS_INVALID_ARGUMENT;\n        }\n", file);
    fputs("    }\n}\n\n", file);
}

static void SparkWriteGeneratedSnapshotFunction(
    FILE *file,
    const SparkDriverBuildImage *driver_image)
{
    uint32_t program_index;

    fputs("static SparkStatus SparkGeneratedDriverSnapshot(void *driver_instance, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot)\n{\n", file);
    fputs("    SparkGeneratedDriverInstance *instance;\n\n", file);
    fputs("    if (driver_instance == 0 || snapshot == 0)\n    {\n        return SPARK_STATUS_INVALID_ARGUMENT;\n    }\n", file);
    fputs("    instance = (SparkGeneratedDriverInstance *)driver_instance;\n", file);
    fputs("    memset(snapshot, 0, sizeof(*snapshot));\n", file);
    fputs("    snapshot->descriptor_bytes = sizeof(*snapshot);\n", file);
    fputs("    snapshot->program_id = program_id;\n", file);
    fputs("    switch (program_id)\n    {\n", file);
    for (program_index = 0u; program_index < driver_image->stage->program_count; ++program_index)
    {
        const SparkModelProgramDescription *program;
        uint32_t flattened_operation_index;
        uint32_t program_operation_index;

        program = &driver_image->stage->programs[program_index];
        flattened_operation_index = SparkFindFlattenedProgramStart(driver_image, program);
        fprintf(file, "        case %uu:\n        {\n", program->program_id);
        fputs("            SparkStatus status;\n", file);
        for (program_operation_index = 0u; program_operation_index < program->operation_count; ++program_operation_index)
        {
            const SparkDriverBuildOperation *resolved_operation;
            uint32_t operation_index;

            operation_index = flattened_operation_index + program_operation_index;
            resolved_operation = &driver_image->operations[operation_index];
            if (resolved_operation->artifact.snapshot_symbol[0] != '\0')
            {
                fputs("            {\n", file);
                fputs("                SparkModelDriverRuntimeSnapshot module_snapshot;\n", file);
                fprintf(file, "                status = %s(instance->operation_%u_state, program_id, &module_snapshot);\n", resolved_operation->artifact.snapshot_symbol, operation_index);
                fputs("                if (status != SPARK_STATUS_OK)\n                {\n                    return status;\n                }\n", file);
                fputs("                if (module_snapshot.descriptor_bytes < sizeof(module_snapshot))\n                {\n                    return SPARK_STATUS_ABI_MISMATCH;\n                }\n", file);
                fputs("                SparkGeneratedMergeRuntimeSnapshot(snapshot, &module_snapshot);\n", file);
                fputs("            }\n", file);
            }
        }
        fputs("            return SPARK_STATUS_OK;\n        }\n", file);
    }
    fputs("        default:\n        {\n            return SPARK_STATUS_INVALID_ARGUMENT;\n        }\n", file);
    fputs("    }\n}\n\n", file);
}
static void SparkWriteGeneratedDescriptors(
    FILE *file,
    const SparkModelDescription *description,
    const SparkDriverBuildImage *driver_image)
{
    uint32_t program_index;

    fputs("static const SparkModelDriverProgramProfile SparkGeneratedProgramProfiles[] =\n{\n", file);
    for (program_index = 0u; program_index < driver_image->stage->program_count; ++program_index)
    {
        const SparkModelProgramDescription *program;
        program = &driver_image->stage->programs[program_index];
        fprintf(
            file,
            "    { sizeof(SparkModelDriverProgramProfile), %uu, %uu, %uu, %uu, %uu, %lluull, %lluull, %lluull, %lluull, %lluull, %lluull, %lluull, %lluull, %uu, 0u }%s\n",
            program->scheduling.flags,
            program->max_inflight,
            program->scheduling.max_active_slots,
            program->scheduling.max_new_tokens,
            program->scheduling.max_resident_sequences,
            (unsigned long long)program->scheduling.max_sequence_tokens,
            (unsigned long long)program->scheduling.target_latency_ns,
            (unsigned long long)program->scheduling.validated_latency_ns,
            (unsigned long long)program->scheduling.resident_weight_bytes,
            (unsigned long long)program->scheduling.resident_kv_bytes,
            (unsigned long long)program->scheduling.static_workspace_bytes,
            (unsigned long long)program->scheduling.device_memcpy_bytes_per_submit_ceiling,
            (unsigned long long)program->scheduling.host_staging_bytes_per_submit_ceiling,
            program->scheduling.private_queue_count,
            program_index + 1u == driver_image->stage->program_count ? "" : ",");
    }
    fputs("};\n\n", file);

    fputs("static const SparkModelDriverProgramDescriptor SparkGeneratedPrograms[] =\n{\n", file);
    for (program_index = 0u; program_index < driver_image->stage->program_count; ++program_index)
    {
        const SparkModelProgramDescription *program;
        uint32_t flags;

        program = &driver_image->stage->programs[program_index];
        flags = program->scheduling.flags;
        if (program->completion_mode == SPARK_MODEL_PROGRAM_COMPLETION_EXTERNAL)
        {
            flags |= SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION;
        }
        fprintf(file, "    { %uu, %uu, %uu, 0u, ", program->program_id, flags, program->max_inflight);
        SparkWriteCStringLiteral(file, program->name, strlen(program->name));
        fprintf(file, ", &SparkGeneratedProgramProfiles[%u], SparkGeneratedSubmitProgram_%u }%s\n", program_index, program_index, program_index + 1u == driver_image->stage->program_count ? "" : ",");
    }
    fputs("};\n\n", file);

    fputs("static const SparkModelDriverDescriptor SparkGeneratedDescriptor =\n{\n", file);
    fputs("    SPARK_MODEL_DRIVER_ABI_VERSION,\n    sizeof(SparkModelDriverDescriptor),\n    ", file);
    SparkWriteCStringLiteral(file, description->model_id, strlen(description->model_id));
    fputs(",\n    ", file);
    SparkWriteCStringLiteral(file, description->model_revision, strlen(description->model_revision));
    fputs(",\n    ", file);
    SparkWriteCStringLiteral(file, driver_image->stage->name, strlen(driver_image->stage->name));
    fputs(",\n    ", file);
    SparkWriteCStringLiteral(file, driver_image->stage->target, strlen(driver_image->stage->target));
    fputs(",\n    ", file);
    SparkWriteCStringLiteral(file, description->source_sha256, strlen(description->source_sha256));
    fputs(",\n    ", file);
    SparkWriteCStringLiteral(file, driver_image->compiled_program_sha256, strlen(driver_image->compiled_program_sha256));
    fprintf(
        file,
        ",\n    %uu,\n    %uu,\n    SparkGeneratedPrograms\n};\n\n",
        driver_image->stage->program_count,
        driver_image->operation_count);

    fputs("static const SparkModelDriverInterface SparkGeneratedInterface =\n{\n", file);
    fputs("    SPARK_MODEL_DRIVER_ABI_VERSION,\n", file);
    fputs("    sizeof(SparkModelDriverInterface),\n", file);
    fputs("    &SparkGeneratedDescriptor,\n", file);
    fputs("    SparkGeneratedDriverCreate,\n", file);
    fputs("    SparkGeneratedDriverDestroy,\n", file);
    fputs("    SparkGeneratedDriverAdmit,\n", file);
    fputs("    SparkGeneratedDriverSnapshot\n};\n\n", file);
    fputs("SPARK_MODEL_DRIVER_EXPORT const SparkModelDriverInterface *SparkModelDriverGetInterface(void)\n{\n", file);
    fputs("    return &SparkGeneratedInterface;\n}\n", file);
}

static SparkStatus SparkGenerateDriverSource(
    const char *source_path,
    const SparkModelDescription *description,
    const SparkDriverBuildImage *driver_image,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    FILE *file;
    uint32_t program_index;

    file = fopen(source_path, "wb");
    if (file == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot create generated driver source '%s'", source_path);
        return SPARK_STATUS_IO_ERROR;
    }
    fputs("#include <stdlib.h>\n", file);
    fputs("#include <string.h>\n\n", file);
    fputs("#include \"sparkpipe/spark_model_driver.h\"\n", file);
    fputs("#include \"sparkpipe/spark_module_abi.h\"\n\n", file);
    SparkWriteGeneratedExterns(file, driver_image);
    SparkWriteGeneratedConfigurations(file, description, driver_image);
    SparkWriteGeneratedInstance(file, driver_image);
    fputs("static void SparkGeneratedDriverDestroy(void *driver_instance);\n\n", file);
    SparkWriteGeneratedCreate(file, driver_image);
    SparkWriteGeneratedDestroy(file, driver_image);
    for (program_index = 0u; program_index < driver_image->stage->program_count; ++program_index)
    {
        SparkWriteGeneratedProgramFunction(file, driver_image, program_index);
    }
    SparkWriteGeneratedAdmitFunction(file, driver_image);
    SparkWriteGeneratedSnapshotFunction(file, driver_image);
    SparkWriteGeneratedDescriptors(file, description, driver_image);

    {
        int flush_result;
        int close_result;

        flush_result = fflush(file);
        close_result = fclose(file);
        if (flush_result != 0 || close_result != 0)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "cannot finalize generated driver source '%s'", source_path);
            return SPARK_STATUS_IO_ERROR;
        }
    }
    return SPARK_STATUS_OK;
}

static const char *SparkDriverLinkUnitExtension(SparkModuleLinkUnitKind link_unit_kind)
{
    if (link_unit_kind == SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT)
    {
        return "o";
    }
    if (link_unit_kind == SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE)
    {
        return "a";
    }
    return 0;
}

static bool SparkDriverLinkUnitFilenameIsContentAddressed(const char *filename)
{
    size_t character_index;
    char extension;

    if (filename == 0 || strlen(filename) != (SPARK_SHA256_HEX_BYTES - 1u) + 2u)
    {
        return false;
    }
    for (character_index = 0u; character_index < SPARK_SHA256_HEX_BYTES - 1u; ++character_index)
    {
        char character;

        character = filename[character_index];
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f')))
        {
            return false;
        }
    }
    extension = filename[SPARK_SHA256_HEX_BYTES];
    return filename[SPARK_SHA256_HEX_BYTES - 1u] == '.' &&
           (extension == 'o' || extension == 'a');
}

static SparkStatus SparkClearDriverLinkUnitDirectory(
    const char *link_unit_directory,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    DIR *directory;
    struct dirent *directory_entry;
    SparkStatus status;

    directory = opendir(link_unit_directory);
    if (directory == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot open driver link unit directory '%s'", link_unit_directory);
        return SPARK_STATUS_IO_ERROR;
    }

    status = SPARK_STATUS_OK;
    errno = 0;
    while ((directory_entry = readdir(directory)) != 0)
    {
        char link_unit_path[SPARK_DRIVER_COMPILER_PATH_BYTES];

        if (strcmp(directory_entry->d_name, ".") == 0 || strcmp(directory_entry->d_name, "..") == 0)
        {
            continue;
        }
        if (!SparkDriverLinkUnitFilenameIsContentAddressed(directory_entry->d_name))
        {
            SparkSetError(
                error_buffer,
                error_buffer_bytes,
                "driver link unit directory '%s' contains unexpected entry '%s'",
                link_unit_directory,
                directory_entry->d_name);
            status = SPARK_STATUS_IO_ERROR;
            break;
        }
        status = SparkJoinPath(link_unit_directory, directory_entry->d_name, link_unit_path, sizeof(link_unit_path));
        if (status != SPARK_STATUS_OK)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "driver link unit path exceeds capacity");
            break;
        }
        if (unlink(link_unit_path) != 0)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "cannot remove stale driver link unit '%s'", link_unit_path);
            status = SPARK_STATUS_IO_ERROR;
            break;
        }
        errno = 0;
    }
    if (status == SPARK_STATUS_OK && errno != 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot enumerate driver link unit directory '%s'", link_unit_directory);
        status = SPARK_STATUS_IO_ERROR;
    }
    if (closedir(directory) != 0 && status == SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot close driver link unit directory '%s'", link_unit_directory);
        status = SPARK_STATUS_IO_ERROR;
    }
    return status;
}

static SparkStatus SparkCollectValidatedLinkUnits(
    const SparkDriverBuildImage *driver_image,
    const char *output_directory,
    char ***collected_link_unit_paths,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char link_unit_directory[SPARK_DRIVER_COMPILER_PATH_BYTES];
    char **paths;
    uint32_t link_unit_index;
    SparkStatus status;

    if (SparkJoinPath(output_directory, SPARK_DRIVER_LINK_UNIT_DIRECTORY_NAME, link_unit_directory, sizeof(link_unit_directory)) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkCreateDirectories(link_unit_directory);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot create driver link unit directory '%s'", link_unit_directory);
        return status;
    }
    status = SparkClearDriverLinkUnitDirectory(link_unit_directory, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    paths = (char **)calloc(driver_image->unique_link_unit_count, sizeof(*paths));
    if (paths == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    for (link_unit_index = 0u; link_unit_index < driver_image->unique_link_unit_count; ++link_unit_index)
    {
        char link_unit_name[SPARK_SHA256_HEX_BYTES + 3u];
        char destination_path[SPARK_DRIVER_COMPILER_PATH_BYTES];
        char copied_hash[SPARK_SHA256_HEX_BYTES];
        const char *link_unit_extension;

        link_unit_extension = SparkDriverLinkUnitExtension(driver_image->unique_link_unit_kinds[link_unit_index]);
        if (link_unit_extension == 0 ||
            snprintf(
                link_unit_name,
                sizeof(link_unit_name),
                "%s.%s",
                driver_image->unique_link_unit_hashes[link_unit_index],
                link_unit_extension) >= (int)sizeof(link_unit_name))
        {
            status = SPARK_STATUS_INTERNAL_ERROR;
            goto fail;
        }
        status = SparkJoinPath(link_unit_directory, link_unit_name, destination_path, sizeof(destination_path));
        if (status != SPARK_STATUS_OK)
        {
            goto fail;
        }
        status = SparkCopyFile(driver_image->unique_link_unit_paths[link_unit_index], destination_path);
        if (status != SPARK_STATUS_OK)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "cannot collect validated link unit '%s'", driver_image->unique_link_unit_paths[link_unit_index]);
            goto fail;
        }
        status = SparkSha256File(destination_path, copied_hash);
        if (status != SPARK_STATUS_OK || strcmp(copied_hash, driver_image->unique_link_unit_hashes[link_unit_index]) != 0)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "collected link unit '%s' failed its content-address check", destination_path);
            status = SPARK_STATUS_HASH_MISMATCH;
            goto fail;
        }
        if (chmod(destination_path, S_IRUSR | S_IRGRP | S_IROTH) != 0)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "cannot make collected link unit '%s' read-only", destination_path);
            status = SPARK_STATUS_IO_ERROR;
            goto fail;
        }
        paths[link_unit_index] = SparkDuplicateText(destination_path);
        if (paths[link_unit_index] == 0)
        {
            status = SPARK_STATUS_INTERNAL_ERROR;
            goto fail;
        }
    }

    *collected_link_unit_paths = paths;
    return SPARK_STATUS_OK;

fail:
    for (link_unit_index = 0u; link_unit_index < driver_image->unique_link_unit_count; ++link_unit_index)
    {
        free(paths[link_unit_index]);
    }
    free(paths);
    return status;
}

static void SparkDestroyCollectedLinkUnitPaths(char **paths, uint32_t path_count)
{
    uint32_t path_index;

    if (paths == 0)
    {
        return;
    }
    for (path_index = 0u; path_index < path_count; ++path_index)
    {
        free(paths[path_index]);
    }
    free(paths);
}

static SparkStatus SparkLinkGeneratedDriver(
    const SparkDriverCompileRequest *request,
    const SparkDriverBuildImage *driver_image,
    char **collected_link_unit_paths,
    const char *generated_source_path,
    const char *driver_path,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t argument_capacity;
    char **arguments;
    uint32_t argument_index;
    uint32_t extra_argument_index;
    uint32_t link_unit_index;
    char include_argument[SPARK_DRIVER_COMPILER_PATH_BYTES + 3u];
    int exit_code;
    SparkStatus status;

    if (snprintf(include_argument, sizeof(include_argument), "-I%s", request->sparkpipe_include_directory) >= (int)sizeof(include_argument))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    argument_capacity = 16u + request->extra_compiler_argument_count + driver_image->unique_link_unit_count;
    arguments = (char **)calloc(argument_capacity, sizeof(*arguments));
    if (arguments == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    argument_index = 0u;
    arguments[argument_index++] = (char *)request->compiler_path;
    arguments[argument_index++] = "-std=c11";
    arguments[argument_index++] = "-O3";
    arguments[argument_index++] = "-fPIC";
    arguments[argument_index++] = "-fvisibility=hidden";
    arguments[argument_index++] = "-fno-semantic-interposition";
#if defined(__APPLE__)
    arguments[argument_index++] = "-dynamiclib";
    arguments[argument_index++] = "-Wl,-undefined,error";
    arguments[argument_index++] = "-Wl,-exported_symbol,_SparkModelDriverGetInterface";
#else
    arguments[argument_index++] = "-shared";
    arguments[argument_index++] = "-Wl,-z,defs";
    arguments[argument_index++] = "-Wl,-O1";
    arguments[argument_index++] = "-Wl,--exclude-libs,ALL";
#endif
    arguments[argument_index++] = include_argument;
    arguments[argument_index++] = (char *)generated_source_path;
    for (link_unit_index = 0u; link_unit_index < driver_image->unique_link_unit_count; ++link_unit_index)
    {
        arguments[argument_index++] = collected_link_unit_paths[link_unit_index];
    }
    for (extra_argument_index = 0u; extra_argument_index < request->extra_compiler_argument_count; ++extra_argument_index)
    {
        arguments[argument_index++] = (char *)request->extra_compiler_arguments[extra_argument_index];
    }
    arguments[argument_index++] = "-o";
    arguments[argument_index++] = (char *)driver_path;
    arguments[argument_index] = 0;

    status = SparkRunProcess(request->compiler_path, arguments, &exit_code);
    free(arguments);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot execute compiler '%s'", request->compiler_path);
        return status;
    }
    if (exit_code != 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "driver compiler failed with exit code %d", exit_code);
        return SPARK_STATUS_COMPILER_ERROR;
    }
    return SPARK_STATUS_OK;
}

static char *SparkEscapeManifestString(const char *text)
{
    size_t source_bytes;
    size_t source_index;
    size_t destination_index;
    char *escaped_text;

    if (text == 0)
    {
        return 0;
    }
    source_bytes = strlen(text);
    if (source_bytes > (SIZE_MAX - 1u) / 6u)
    {
        return 0;
    }
    escaped_text = (char *)malloc((source_bytes * 6u) + 1u);
    if (escaped_text == 0)
    {
        return 0;
    }
    destination_index = 0u;
    for (source_index = 0u; source_index < source_bytes; ++source_index)
    {
        unsigned char character;

        character = (unsigned char)text[source_index];
        if (character == '"' || character == '\\')
        {
            escaped_text[destination_index++] = '\\';
            escaped_text[destination_index++] = (char)character;
        }
        else if (character == '\n')
        {
            escaped_text[destination_index++] = '\\';
            escaped_text[destination_index++] = 'n';
        }
        else if (character == '\r')
        {
            escaped_text[destination_index++] = '\\';
            escaped_text[destination_index++] = 'r';
        }
        else if (character == '\t')
        {
            escaped_text[destination_index++] = '\\';
            escaped_text[destination_index++] = 't';
        }
        else if (character < 0x20u)
        {
            static const char HexDigits[] = "0123456789abcdef";

            escaped_text[destination_index++] = '\\';
            escaped_text[destination_index++] = 'u';
            escaped_text[destination_index++] = '0';
            escaped_text[destination_index++] = '0';
            escaped_text[destination_index++] = HexDigits[character >> 4u];
            escaped_text[destination_index++] = HexDigits[character & 0x0fu];
        }
        else
        {
            escaped_text[destination_index++] = (char)character;
        }
    }
    escaped_text[destination_index] = '\0';
    return escaped_text;
}

static SparkStatus SparkWriteCompiledModuleEntries(
    FILE *file,
    const SparkDriverBuildImage *driver_image)
{
    char *target;
    uint32_t operation_index;
    SparkStatus status;

    if (file == 0 || driver_image == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    target = SparkEscapeManifestString(driver_image->stage->target);
    if (target == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    fputs("[\n", file);
    status = SPARK_STATUS_OK;
    for (operation_index = 0u; operation_index < driver_image->operation_count; ++operation_index)
    {
        const SparkDriverBuildOperation *resolved_operation;
        char *program_name;
        char *operation_name;
        char *module_id;
        char *validation_recipe;

        resolved_operation = &driver_image->operations[operation_index];
        program_name = SparkEscapeManifestString(resolved_operation->program->name);
        operation_name = SparkEscapeManifestString(resolved_operation->operation->name);
        module_id = SparkEscapeManifestString(resolved_operation->operation->module_id);
        validation_recipe = SparkEscapeManifestString(resolved_operation->artifact.validation_recipe);
        if (program_name == 0 || operation_name == 0 || module_id == 0 || validation_recipe == 0)
        {
            free(program_name);
            free(operation_name);
            free(module_id);
            free(validation_recipe);
            status = SPARK_STATUS_INTERNAL_ERROR;
            break;
        }
        fprintf(
            file,
            "    { \"operation_index\": %u, \"program\": \"%s\", \"program_id\": %u, "
            "\"program_max_inflight\": %u, \"program_completion\": \"%s\", "
            "\"program_flags\": %u, \"max_active_slots\": %u, "
            "\"max_new_tokens\": %u, \"target_latency_ns\": %llu, "
            "\"host_staging_bytes_per_submit_ceiling\": %llu, "
            "\"device_memcpy_bytes_per_submit_ceiling\": %llu, "
            "\"operation\": \"%s\", \"module_id\": \"%s\", \"configuration\": ",
            operation_index,
            program_name,
            resolved_operation->program->program_id,
            resolved_operation->program->max_inflight,
            SparkModelProgramCompletionModeToString(resolved_operation->program->completion_mode),
            resolved_operation->program->scheduling.flags,
            resolved_operation->program->scheduling.max_active_slots,
            resolved_operation->program->scheduling.max_new_tokens,
            (unsigned long long)resolved_operation->program->scheduling.target_latency_ns,
            (unsigned long long)resolved_operation->program->scheduling.host_staging_bytes_per_submit_ceiling,
            (unsigned long long)resolved_operation->program->scheduling.device_memcpy_bytes_per_submit_ceiling,
            operation_name,
            module_id);
        if (fwrite(
                resolved_operation->operation->configuration_json,
                1u,
                resolved_operation->operation->configuration_json_bytes,
                file) != resolved_operation->operation->configuration_json_bytes)
        {
            status = SPARK_STATUS_IO_ERROR;
        }
        else
        {
            const char *link_unit_extension;

            link_unit_extension = SparkDriverLinkUnitExtension(resolved_operation->artifact.link_unit_kind);
            if (link_unit_extension == 0)
            {
                status = SPARK_STATUS_INTERNAL_ERROR;
            }
            else
            {
                fprintf(
                    file,
                    ", \"target\": \"%s\", \"module_abi_version\": %u, "
                    "\"validation_recipe\": \"%s\", \"artifact_sha256\": \"%s\", "
                    "\"link_unit_kind\": \"%s\", "
                    "\"initialize_symbol\": \"%s\", \"execute_symbol\": \"%s\", "
                    "\"admit_symbol\": \"%s\", \"snapshot_symbol\": \"%s\", "
                    "\"destroy_symbol\": \"%s\", \"link_unit\": \"link_units/%s.%s\" }%s\n",
                    target,
                    resolved_operation->artifact.module_abi_version,
                    validation_recipe,
                    resolved_operation->artifact.artifact_sha256,
                    SparkModuleLinkUnitKindToString(resolved_operation->artifact.link_unit_kind),
                    resolved_operation->artifact.initialize_symbol,
                    resolved_operation->artifact.execute_symbol,
                    resolved_operation->artifact.admit_symbol,
                    resolved_operation->artifact.snapshot_symbol,
                    resolved_operation->artifact.destroy_symbol,
                    resolved_operation->artifact.artifact_sha256,
                    link_unit_extension,
                    operation_index + 1u == driver_image->operation_count ? "" : ",");
            }
        }
        free(program_name);
        free(operation_name);
        free(module_id);
        free(validation_recipe);
        if (status != SPARK_STATUS_OK)
        {
            break;
        }
    }
    if (status == SPARK_STATUS_OK)
    {
        fputs("  ]", file);
        if (ferror(file) != 0)
        {
            status = SPARK_STATUS_IO_ERROR;
        }
    }
    free(target);
    return status;
}

static SparkStatus SparkWriteCompiledManifest(
    const char *manifest_path,
    const char *driver_path,
    const SparkModelDescription *description,
    const SparkDriverBuildImage *driver_image,
    const char *driver_sha256,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    FILE *file;
    char *model_id;
    char *model_revision;
    char *stage_name;
    char *target;
    SparkStatus status;

    model_id = SparkEscapeManifestString(description->model_id);
    model_revision = SparkEscapeManifestString(description->model_revision);
    stage_name = SparkEscapeManifestString(driver_image->stage->name);
    target = SparkEscapeManifestString(driver_image->stage->target);
    if (model_id == 0 || model_revision == 0 || stage_name == 0 || target == 0)
    {
        free(model_id);
        free(model_revision);
        free(stage_name);
        free(target);
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    file = fopen(manifest_path, "wb");
    if (file == 0)
    {
        free(model_id);
        free(model_revision);
        free(stage_name);
        free(target);
        SparkSetError(error_buffer, error_buffer_bytes, "cannot create compiled manifest '%s'", manifest_path);
        return SPARK_STATUS_IO_ERROR;
    }
    fprintf(file, "{\n  \"schema_version\": 3,\n  \"driver_abi_version\": %u,\n", SPARK_MODEL_DRIVER_ABI_VERSION);
    fprintf(file, "  \"model_id\": \"%s\",\n  \"model_revision\": \"%s\",\n", model_id, model_revision);
    fprintf(file, "  \"stage\": \"%s\",\n  \"target\": \"%s\",\n", stage_name, target);
    fprintf(file, "  \"model_description_sha256\": \"%s\",\n", description->source_sha256);
    fprintf(file, "  \"compiled_program_sha256\": \"%s\",\n", driver_image->compiled_program_sha256);
    fprintf(file, "  \"driver_sha256\": \"%s\",\n", driver_sha256);
    fprintf(file, "  \"driver\": \"%s\",\n", strrchr(driver_path, '/') != 0 ? strrchr(driver_path, '/') + 1 : driver_path);
    fputs("  \"modules\": ", file);
    status = SparkWriteCompiledModuleEntries(file, driver_image);
    if (status == SPARK_STATUS_OK)
    {
        fputs("\n}\n", file);
    }

    free(model_id);
    free(model_revision);
    free(stage_name);
    free(target);
    {
        int flush_result;
        int close_result;

        flush_result = fflush(file);
        close_result = fclose(file);
        if (status == SPARK_STATUS_OK && (flush_result != 0 || close_result != 0))
        {
            status = SPARK_STATUS_IO_ERROR;
        }
    }
    if (status != SPARK_STATUS_OK)
    {
        unlink(manifest_path);
        SparkSetError(error_buffer, error_buffer_bytes, "cannot finalize compiled manifest '%s'", manifest_path);
    }
    return status;
}

static SparkStatus SparkRemoveDriverOutputFile(
    const char *path,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (unlink(path) == 0 || errno == ENOENT)
    {
        return SPARK_STATUS_OK;
    }
    SparkSetError(error_buffer, error_buffer_bytes, "cannot invalidate previous driver output '%s'", path);
    return SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkInvalidatePreviousDriverOutput(
    const char *output_directory,
    const SparkDriverCompileReport *report,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char link_unit_directory[SPARK_DRIVER_COMPILER_PATH_BYTES];
    SparkStatus status;

    status = SparkRemoveDriverOutputFile(report->driver_path, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRemoveDriverOutputFile(report->manifest_path, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRemoveDriverOutputFile(report->generated_source_path, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkJoinPath(output_directory, SPARK_DRIVER_LINK_UNIT_DIRECTORY_NAME, link_unit_directory, sizeof(link_unit_directory));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkPathExists(link_unit_directory))
    {
        return SPARK_STATUS_OK;
    }
    return SparkClearDriverLinkUnitDirectory(link_unit_directory, error_buffer, error_buffer_bytes);
}

static SparkStatus SparkValidateCompilerArguments(
    const char *const *extra_compiler_arguments,
    uint32_t extra_compiler_argument_count,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t extra_argument_index;

    if (extra_compiler_argument_count != 0u && extra_compiler_arguments == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "driver compiler arguments are missing");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (extra_argument_index = 0u; extra_argument_index < extra_compiler_argument_count;
         ++extra_argument_index)
    {
        if (extra_compiler_arguments[extra_argument_index] == 0)
        {
            SparkSetError(error_buffer, error_buffer_bytes, "driver compiler argument %u is null", extra_argument_index);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkValidateDriverCompileRequest(
    const SparkDriverCompileRequest *request,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (request == 0 || request->model_description_path == 0 || request->stage_name == 0 ||
        request->module_library_root == 0 || request->output_directory == 0 ||
        request->compiler_path == 0 || request->sparkpipe_include_directory == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->model_description_path[0] == '\0' || request->stage_name[0] == '\0' ||
        request->module_library_root[0] == '\0' || request->output_directory[0] == '\0' ||
        request->compiler_path[0] == '\0' || request->sparkpipe_include_directory[0] == '\0')
    {
        SparkSetError(error_buffer, error_buffer_bytes, "driver compile request contains an empty required field");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkValidateCompilerArguments(
        request->extra_compiler_arguments,
        request->extra_compiler_argument_count,
        error_buffer,
        error_buffer_bytes);
}

static SparkStatus SparkValidateModelPackageCompileRequest(
    const SparkModelPackageCompileRequest *request,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (request == 0 || request->model_description_path == 0 ||
        request->module_library_root == 0 || request->output_directory == 0 ||
        request->compiler_path == 0 || request->sparkpipe_include_directory == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->model_description_path[0] == '\0' ||
        request->module_library_root[0] == '\0' || request->output_directory[0] == '\0' ||
        request->compiler_path[0] == '\0' || request->sparkpipe_include_directory[0] == '\0')
    {
        SparkSetError(error_buffer, error_buffer_bytes, "model package compile request contains an empty required field");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkValidateCompilerArguments(
        request->extra_compiler_arguments,
        request->extra_compiler_argument_count,
        error_buffer,
        error_buffer_bytes);
}

static SparkStatus SparkPrepareDriverOutput(
    const char *output_directory,
    SparkDriverCompileReport *report,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;

    status = SparkCreateDirectories(output_directory);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot create driver output directory '%s'", output_directory);
        return status;
    }
    if (SparkJoinPath(output_directory, SPARK_DRIVER_GENERATED_SOURCE_NAME, report->generated_source_path, sizeof(report->generated_source_path)) != SPARK_STATUS_OK ||
        SparkJoinPath(output_directory, SPARK_DRIVER_SHARED_OBJECT_NAME, report->driver_path, sizeof(report->driver_path)) != SPARK_STATUS_OK ||
        SparkJoinPath(output_directory, SPARK_DRIVER_MANIFEST_NAME, report->manifest_path, sizeof(report->manifest_path)) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SparkInvalidatePreviousDriverOutput(output_directory, report, error_buffer, error_buffer_bytes);
}

static SparkStatus SparkBuildLoadedModelStage(
    const SparkDriverCompileRequest *request,
    const SparkModelDescription *description,
    const SparkModelStageDescription *stage,
    bool write_compiled_manifest,
    SparkDriverBuildImage *driver_image,
    SparkDriverCompileReport *report,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char **collected_link_unit_paths;
    SparkStatus status;

    memset(report, 0, sizeof(*report));
    SparkDriverBuildImageReset(driver_image);
    collected_link_unit_paths = 0;

    status = SparkPrepareDriverOutput(request->output_directory, report, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkResolveDriverBuildImage(stage, request->module_library_root, driver_image, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkComputeCompiledProgramHash(
        request,
        description,
        driver_image,
        driver_image->compiled_program_sha256);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkGenerateDriverSource(report->generated_source_path, description, driver_image, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkCollectValidatedLinkUnits(driver_image, request->output_directory, &collected_link_unit_paths, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkLinkGeneratedDriver(request, driver_image, collected_link_unit_paths, report->generated_source_path, report->driver_path, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkSha256File(report->driver_path, report->driver_sha256);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot hash compiled driver '%s'", report->driver_path);
        goto cleanup;
    }
    if (write_compiled_manifest)
    {
        status = SparkWriteCompiledManifest(report->manifest_path, report->driver_path, description, driver_image, report->driver_sha256, error_buffer, error_buffer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            goto cleanup;
        }
    }

    report->program_count = stage->program_count;
    report->operation_count = driver_image->operation_count;
    report->unique_link_unit_count = driver_image->unique_link_unit_count;
    memcpy(report->model_description_sha256, description->source_sha256, sizeof(report->model_description_sha256));
    memcpy(report->compiled_program_sha256, driver_image->compiled_program_sha256, sizeof(report->compiled_program_sha256));

cleanup:
    SparkDestroyCollectedLinkUnitPaths(collected_link_unit_paths, driver_image->unique_link_unit_count);
    if (status != SPARK_STATUS_OK)
    {
        SparkDriverBuildImageDestroy(driver_image);
        if (report->driver_path[0] != '\0')
        {
            char cleanup_error[256];
            SparkStatus cleanup_status;

            cleanup_error[0] = '\0';
            cleanup_status = SparkInvalidatePreviousDriverOutput(
                request->output_directory,
                report,
                cleanup_error,
                sizeof(cleanup_error));
            if (cleanup_status != SPARK_STATUS_OK)
            {
                SparkSetError(
                    error_buffer,
                    error_buffer_bytes,
                    "driver compilation failed and stale output cleanup also failed: %s",
                    cleanup_error);
                status = cleanup_status;
            }
        }
    }
    return status;
}

SparkStatus SparkCompileModelDriver(
    const SparkDriverCompileRequest *request,
    SparkDriverCompileReport *report,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkModelDescription description;
    const SparkModelStageDescription *stage;
    SparkDriverBuildImage driver_image;
    SparkStatus status;

    if (report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(report, 0, sizeof(*report));
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        error_buffer[0] = '\0';
    }
    status = SparkValidateDriverCompileRequest(request, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkModelDescriptionReset(&description);
    status = SparkLoadModelDescription(request->model_description_path, &description, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    stage = SparkFindModelStage(&description, request->stage_name);
    if (stage == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "model description does not contain stage '%s'", request->stage_name);
        SparkModelDescriptionDestroy(&description);
        return SPARK_STATUS_NOT_FOUND;
    }

    SparkDriverBuildImageReset(&driver_image);
    status = SparkBuildLoadedModelStage(
        request,
        &description,
        stage,
        true,
        &driver_image,
        report,
        error_buffer,
        error_buffer_bytes);
    SparkDriverBuildImageDestroy(&driver_image);
    SparkModelDescriptionDestroy(&description);
    return status;
}

static SparkStatus SparkWriteModelPackageManifest(
    const char *manifest_path,
    const char *model_description_path,
    const SparkModelDescription *description,
    const SparkDriverBuildImage *stage_images,
    const SparkDriverCompileReport *stage_reports,
    const SparkModelPackageCompileReport *package_report,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char temporary_path[SPARK_DRIVER_COMPILER_PATH_BYTES];
    FILE *file;
    char *model_id;
    char *model_revision;
    char *model_description_json;
    size_t model_description_json_bytes;
    uint32_t stage_index;
    SparkStatus status;

    status = SparkReadEntireFile(model_description_path, &model_description_json, &model_description_json_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot read model description '%s' for package manifest", model_description_path);
        return status;
    }
    if (snprintf(temporary_path, sizeof(temporary_path), "%s.tmp.%ld", manifest_path, (long)getpid()) >= (int)sizeof(temporary_path))
    {
        free(model_description_json);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    file = fopen(temporary_path, "wb");
    if (file == 0)
    {
        free(model_description_json);
        SparkSetError(error_buffer, error_buffer_bytes, "cannot create model package manifest '%s'", manifest_path);
        return SPARK_STATUS_IO_ERROR;
    }

    model_id = SparkEscapeManifestString(description->model_id);
    model_revision = SparkEscapeManifestString(description->model_revision);
    if (model_id == 0 || model_revision == 0)
    {
        free(model_id);
        free(model_revision);
        free(model_description_json);
        fclose(file);
        unlink(temporary_path);
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    fprintf(
        file,
        "{\n"
        "  \"schema_version\": %u,\n"
        "  \"model_id\": \"%s\",\n"
        "  \"model_revision\": \"%s\",\n"
        "  \"model_description_sha256\": \"%s\",\n"
        "  \"stage_count\": %u,\n"
        "  \"total_program_count\": %u,\n"
        "  \"total_operation_count\": %u,\n"
        "  \"total_collected_link_unit_count\": %u,\n"
        "  \"model_description\": ",
        SPARK_MODEL_PACKAGE_SCHEMA_VERSION,
        model_id,
        model_revision,
        description->source_sha256,
        package_report->stage_count,
        package_report->total_program_count,
        package_report->total_operation_count,
        package_report->total_collected_link_unit_count);
    if (fwrite(model_description_json, 1u, model_description_json_bytes, file) != model_description_json_bytes)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    else
    {
        fputs(",\n  \"stages\": [\n", file);
        status = SPARK_STATUS_OK;
    }
    free(model_id);
    free(model_revision);
    free(model_description_json);

    for (stage_index = 0u; stage_index < description->stage_count && status == SPARK_STATUS_OK; ++stage_index)
    {
        const SparkModelStageDescription *stage;
        const SparkDriverCompileReport *stage_report;
        char *stage_name;
        char *target;

        stage = &description->stages[stage_index];
        stage_report = &stage_reports[stage_index];
        stage_name = SparkEscapeManifestString(stage->name);
        target = SparkEscapeManifestString(stage->target);
        if (stage_name == 0 || target == 0)
        {
            free(stage_name);
            free(target);
            status = SPARK_STATUS_INTERNAL_ERROR;
            break;
        }
        fprintf(
            file,
            "    { \"stage_index\": %u, \"stage_name\": \"%s\", \"target\": \"%s\", "
            "\"directory\": \"stages/stage_%03u\", \"driver\": \"stages/stage_%03u/%s\", "
            "\"link_units\": \"stages/stage_%03u/link_units\", \"program_count\": %u, "
            "\"operation_count\": %u, \"collected_link_unit_count\": %u, "
            "\"compiled_program_sha256\": \"%s\", \"driver_sha256\": \"%s\", \"modules\": ",
            stage_index,
            stage_name,
            target,
            stage_index,
            stage_index,
            SPARK_DRIVER_SHARED_OBJECT_NAME,
            stage_index,
            stage_report->program_count,
            stage_report->operation_count,
            stage_report->unique_link_unit_count,
            stage_report->compiled_program_sha256,
            stage_report->driver_sha256);
        status = SparkWriteCompiledModuleEntries(file, &stage_images[stage_index]);
        if (status == SPARK_STATUS_OK)
        {
            fprintf(file, " }%s\n", stage_index + 1u == description->stage_count ? "" : ",");
        }
        free(stage_name);
        free(target);
    }
    if (status == SPARK_STATUS_OK)
    {
        fputs("  ]\n}\n", file);
    }

    {
        int flush_result;
        int close_result;

        flush_result = fflush(file);
        close_result = fclose(file);
        if (status == SPARK_STATUS_OK && (flush_result != 0 || close_result != 0))
        {
            status = SPARK_STATUS_IO_ERROR;
        }
    }
    if (status != SPARK_STATUS_OK)
    {
        unlink(temporary_path);
        SparkSetError(error_buffer, error_buffer_bytes, "cannot finalize model package manifest '%s'", manifest_path);
        return status;
    }
    if (rename(temporary_path, manifest_path) != 0)
    {
        unlink(temporary_path);
        SparkSetError(error_buffer, error_buffer_bytes, "cannot activate model package manifest '%s'", manifest_path);
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkAddPackageCount(
    uint32_t current_count,
    uint32_t added_count,
    uint32_t *result)
{
    if (result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (UINT32_MAX - current_count < added_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *result = current_count + added_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkCompileModelPackage(
    const SparkModelPackageCompileRequest *request,
    SparkModelPackageCompileReport *report,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkModelDescription description;
    SparkDriverBuildImage *stage_images;
    SparkDriverCompileReport *stage_reports;
    char stages_directory[SPARK_DRIVER_COMPILER_PATH_BYTES];
    uint32_t stage_index;
    SparkStatus status;

    if (report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(report, 0, sizeof(*report));
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        error_buffer[0] = '\0';
    }
    status = SparkValidateModelPackageCompileRequest(request, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCreateDirectories(request->output_directory);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot create model package output directory '%s'", request->output_directory);
        return status;
    }
    if (SparkJoinPath(request->output_directory, SPARK_MODEL_PACKAGE_MANIFEST_NAME, report->package_manifest_path, sizeof(report->package_manifest_path)) != SPARK_STATUS_OK ||
        SparkJoinPath(request->output_directory, SPARK_MODEL_PACKAGE_STAGE_DIRECTORY_NAME, stages_directory, sizeof(stages_directory)) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkRemoveDriverOutputFile(report->package_manifest_path, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRemoveDirectoryTree(stages_directory);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot clear previous model package stages '%s'", stages_directory);
        return status;
    }

    SparkModelDescriptionReset(&description);
    stage_images = 0;
    stage_reports = 0;
    status = SparkLoadModelDescription(request->model_description_path, &description, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkCreateDirectories(stages_directory);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot create model package stages directory '%s'", stages_directory);
        goto cleanup;
    }
    stage_images = (SparkDriverBuildImage *)calloc(description.stage_count, sizeof(*stage_images));
    stage_reports = (SparkDriverCompileReport *)calloc(description.stage_count, sizeof(*stage_reports));
    if (stage_images == 0 || stage_reports == 0)
    {
        status = SPARK_STATUS_INTERNAL_ERROR;
        goto cleanup;
    }

    report->stage_count = description.stage_count;
    for (stage_index = 0u; stage_index < description.stage_count; ++stage_index)
    {
        SparkDriverCompileRequest stage_request;
        char stage_directory_name[32];
        char stage_output_directory[SPARK_DRIVER_COMPILER_PATH_BYTES];

        if (snprintf(stage_directory_name, sizeof(stage_directory_name), "stage_%03u", stage_index) >= (int)sizeof(stage_directory_name) ||
            SparkJoinPath(stages_directory, stage_directory_name, stage_output_directory, sizeof(stage_output_directory)) != SPARK_STATUS_OK)
        {
            status = SPARK_STATUS_CAPACITY_EXCEEDED;
            goto cleanup;
        }
        memset(&stage_request, 0, sizeof(stage_request));
        stage_request.model_description_path = request->model_description_path;
        stage_request.stage_name = description.stages[stage_index].name;
        stage_request.module_library_root = request->module_library_root;
        stage_request.output_directory = stage_output_directory;
        stage_request.compiler_path = request->compiler_path;
        stage_request.sparkpipe_include_directory = request->sparkpipe_include_directory;
        stage_request.extra_compiler_arguments = request->extra_compiler_arguments;
        stage_request.extra_compiler_argument_count = request->extra_compiler_argument_count;

        status = SparkBuildLoadedModelStage(
            &stage_request,
            &description,
            &description.stages[stage_index],
            false,
            &stage_images[stage_index],
            &stage_reports[stage_index],
            error_buffer,
            error_buffer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            goto cleanup;
        }
        status = SparkAddPackageCount(report->total_program_count, stage_reports[stage_index].program_count, &report->total_program_count);
        if (status != SPARK_STATUS_OK)
        {
            goto cleanup;
        }
        status = SparkAddPackageCount(report->total_operation_count, stage_reports[stage_index].operation_count, &report->total_operation_count);
        if (status != SPARK_STATUS_OK)
        {
            goto cleanup;
        }
        status = SparkAddPackageCount(report->total_collected_link_unit_count, stage_reports[stage_index].unique_link_unit_count, &report->total_collected_link_unit_count);
        if (status != SPARK_STATUS_OK)
        {
            goto cleanup;
        }
    }

    memcpy(report->model_description_sha256, description.source_sha256, sizeof(report->model_description_sha256));
    status = SparkWriteModelPackageManifest(
        report->package_manifest_path,
        request->model_description_path,
        &description,
        stage_images,
        stage_reports,
        report,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkSha256File(report->package_manifest_path, report->package_manifest_sha256);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot hash model package manifest '%s'", report->package_manifest_path);
        goto cleanup;
    }

cleanup:
    if (stage_images != 0)
    {
        for (stage_index = 0u; stage_index < description.stage_count; ++stage_index)
        {
            SparkDriverBuildImageDestroy(&stage_images[stage_index]);
        }
    }
    free(stage_images);
    free(stage_reports);
    SparkModelDescriptionDestroy(&description);
    if (status != SPARK_STATUS_OK)
    {
        SparkRemoveDriverOutputFile(report->package_manifest_path, 0, 0u);
        SparkRemoveDirectoryTree(stages_directory);
    }
    return status;
}
