#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_description.h"

typedef struct SparkGlm52ArtifactGeometry
{
    uint32_t hidden_size;
    uint32_t head_count;
    uint32_t latent_dimension;
    uint32_t rope_dimension;
    uint32_t selected_token_count;
    uint32_t layer_count;
    uint32_t vocab_size;
    uint32_t routed_expert_count;
    uint32_t shared_expert_count;
    uint32_t experts_per_token;
    uint32_t qk_nope_head_dimension;
    uint32_t qk_head_dimension;
    uint32_t value_head_dimension;
} SparkGlm52ArtifactGeometry;

static SparkStatus SparkSetToolError(
    SparkStatus status,
    char *error_buffer,
    uint32_t error_buffer_bytes,
    const char *format,
    ...)
{
    va_list arguments;

    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        va_start(arguments, format);
        vsnprintf(error_buffer, (size_t)error_buffer_bytes, format, arguments);
        va_end(arguments);
    }
    return status;
}

static void SparkPrintUsage(const char *program_name)
{
    fprintf(
        stderr,
        "usage: %s (--config FILE | --model-dir DIR) [--model FILE]\n",
        program_name);
}

static SparkStatus SparkReadRequiredUInt32(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    uint32_t *value,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t member_token_index;
    SparkStatus status;

    member_token_index = SparkJsonFindObjectMember(document, object_token_index, member_name);
    if (member_token_index < 0)
    {
        return SparkSetToolError(
            SPARK_STATUS_NOT_FOUND,
            error_buffer,
            error_buffer_bytes,
            "missing required GLM config field %s",
            member_name);
    }
    status = SparkJsonGetUInt32(document, member_token_index, value);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSetToolError(
            status,
            error_buffer,
            error_buffer_bytes,
            "GLM config field %s is not a uint32",
            member_name);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkExpectUInt32(
    const char *field_name,
    uint32_t actual_value,
    uint32_t expected_value,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (actual_value != expected_value)
    {
        return SparkSetToolError(
            SPARK_STATUS_SCHEMA_ERROR,
            error_buffer,
            error_buffer_bytes,
            "%s mismatch: actual=%u expected=%u",
            field_name,
            actual_value,
            expected_value);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReadAndExpectUInt32(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    uint32_t expected_value,
    uint32_t *actual_value,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;

    status = SparkReadRequiredUInt32(
        document,
        object_token_index,
        member_name,
        actual_value,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkExpectUInt32(
        member_name,
        *actual_value,
        expected_value,
        error_buffer,
        error_buffer_bytes);
}

static SparkStatus SparkReadHfConfigGeometry(
    const char *config_path,
    const SparkGlm52ArtifactGeometry *expected,
    SparkGlm52ArtifactGeometry *actual,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    SparkStatus status;

    SparkJsonDocumentReset(&document);
    status = SparkJsonLoadFile(config_path, &document);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSetToolError(status, error_buffer, error_buffer_bytes, "could not load %s", config_path);
    }
    root_token_index = SparkJsonGetRootToken(&document);
    if (!SparkJsonTokenIsType(&document, root_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkJsonDocumentDestroy(&document);
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "config root must be an object");
    }
#define SPARK_CHECK_CONFIG_U32(member_name, expected_value, destination) \
    do \
    { \
        status = SparkReadAndExpectUInt32(&document, root_token_index, member_name, expected_value, destination, error_buffer, error_buffer_bytes); \
        if (status != SPARK_STATUS_OK) \
        { \
            SparkJsonDocumentDestroy(&document); \
            return status; \
        } \
    } while (0)
    SPARK_CHECK_CONFIG_U32("hidden_size", expected->hidden_size, &actual->hidden_size);
    SPARK_CHECK_CONFIG_U32("num_attention_heads", expected->head_count, &actual->head_count);
    SPARK_CHECK_CONFIG_U32("kv_lora_rank", expected->latent_dimension, &actual->latent_dimension);
    SPARK_CHECK_CONFIG_U32("qk_rope_head_dim", expected->rope_dimension, &actual->rope_dimension);
    SPARK_CHECK_CONFIG_U32("index_topk", expected->selected_token_count, &actual->selected_token_count);
    SPARK_CHECK_CONFIG_U32("num_hidden_layers", expected->layer_count, &actual->layer_count);
    SPARK_CHECK_CONFIG_U32("vocab_size", expected->vocab_size, &actual->vocab_size);
    SPARK_CHECK_CONFIG_U32("n_routed_experts", expected->routed_expert_count, &actual->routed_expert_count);
    SPARK_CHECK_CONFIG_U32("n_shared_experts", expected->shared_expert_count, &actual->shared_expert_count);
    SPARK_CHECK_CONFIG_U32("num_experts_per_tok", expected->experts_per_token, &actual->experts_per_token);
    SPARK_CHECK_CONFIG_U32("qk_nope_head_dim", expected->qk_nope_head_dimension, &actual->qk_nope_head_dimension);
    SPARK_CHECK_CONFIG_U32("qk_head_dim", expected->qk_head_dimension, &actual->qk_head_dimension);
    SPARK_CHECK_CONFIG_U32("v_head_dim", expected->value_head_dimension, &actual->value_head_dimension);
#undef SPARK_CHECK_CONFIG_U32
    SparkJsonDocumentDestroy(&document);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkExpectString(
    const char *field_name,
    const char *actual_value,
    const char *expected_value,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (actual_value == 0 || strcmp(actual_value, expected_value) != 0)
    {
        return SparkSetToolError(
            SPARK_STATUS_SCHEMA_ERROR,
            error_buffer,
            error_buffer_bytes,
            "%s mismatch: actual=%s expected=%s",
            field_name,
            actual_value != 0 ? actual_value : "(null)",
            expected_value);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCheckMetadataGeometry(
    const SparkModelDescription *description,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t geometry_token_index;
    uint32_t actual_value;
    SparkStatus status;

    if (description->metadata_json == 0)
    {
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "model description metadata is missing");
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(description->metadata_json, description->metadata_json_bytes, &document);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSetToolError(status, error_buffer, error_buffer_bytes, "could not parse model description metadata");
    }
    root_token_index = SparkJsonGetRootToken(&document);
    geometry_token_index = SparkJsonFindObjectMember(&document, root_token_index, "module_geometry");
    if (geometry_token_index < 0)
    {
        SparkJsonDocumentDestroy(&document);
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "model description metadata.module_geometry is missing");
    }
#define SPARK_CHECK_METADATA_U32(member_name, expected_value) \
    do \
    { \
        status = SparkReadAndExpectUInt32(&document, geometry_token_index, member_name, expected_value, &actual_value, error_buffer, error_buffer_bytes); \
        if (status != SPARK_STATUS_OK) \
        { \
            SparkJsonDocumentDestroy(&document); \
            return status; \
        } \
    } while (0)
    SPARK_CHECK_METADATA_U32("hidden_dimension", SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
    SPARK_CHECK_METADATA_U32("heads", SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT);
    SPARK_CHECK_METADATA_U32("latent_dimension", SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
    SPARK_CHECK_METADATA_U32("rope_dimension", SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION);
    SPARK_CHECK_METADATA_U32("selected_tokens", SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT);
    SPARK_CHECK_METADATA_U32("kv_block_tokens", SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS);
    SPARK_CHECK_METADATA_U32("restricted_vocab_count", SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT);
    SPARK_CHECK_METADATA_U32("mtp_draft_tokens", SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT);
#undef SPARK_CHECK_METADATA_U32
    SparkJsonDocumentDestroy(&document);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReadArtifactGeometryFromMetadata(
    const SparkModelDescription *description,
    SparkGlm52ArtifactGeometry *expected,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t geometry_token_index;
    SparkStatus status;

    if (description->metadata_json == 0)
    {
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "model description metadata is missing");
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(description->metadata_json, description->metadata_json_bytes, &document);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSetToolError(status, error_buffer, error_buffer_bytes, "could not parse model description metadata");
    }
    root_token_index = SparkJsonGetRootToken(&document);
    geometry_token_index = SparkJsonFindObjectMember(&document, root_token_index, "hf_config_geometry");
    if (geometry_token_index < 0)
    {
        SparkJsonDocumentDestroy(&document);
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "model description metadata.hf_config_geometry is missing");
    }
#define SPARK_READ_HF_GEOMETRY(member_name, destination) \
    do \
    { \
        status = SparkReadRequiredUInt32(&document, geometry_token_index, member_name, destination, error_buffer, error_buffer_bytes); \
        if (status != SPARK_STATUS_OK) \
        { \
            SparkJsonDocumentDestroy(&document); \
            return status; \
        } \
    } while (0)
    SPARK_READ_HF_GEOMETRY("hidden_size", &expected->hidden_size);
    SPARK_READ_HF_GEOMETRY("num_attention_heads", &expected->head_count);
    SPARK_READ_HF_GEOMETRY("kv_lora_rank", &expected->latent_dimension);
    SPARK_READ_HF_GEOMETRY("qk_rope_head_dim", &expected->rope_dimension);
    SPARK_READ_HF_GEOMETRY("index_topk", &expected->selected_token_count);
    SPARK_READ_HF_GEOMETRY("num_hidden_layers", &expected->layer_count);
    SPARK_READ_HF_GEOMETRY("vocab_size", &expected->vocab_size);
    SPARK_READ_HF_GEOMETRY("n_routed_experts", &expected->routed_expert_count);
    SPARK_READ_HF_GEOMETRY("n_shared_experts", &expected->shared_expert_count);
    SPARK_READ_HF_GEOMETRY("num_experts_per_tok", &expected->experts_per_token);
    SPARK_READ_HF_GEOMETRY("qk_nope_head_dim", &expected->qk_nope_head_dimension);
    SPARK_READ_HF_GEOMETRY("qk_head_dim", &expected->qk_head_dimension);
    SPARK_READ_HF_GEOMETRY("v_head_dim", &expected->value_head_dimension);
#undef SPARK_READ_HF_GEOMETRY
    SparkJsonDocumentDestroy(&document);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCheckArtifactGeometryAgainstFirmware(
    const SparkGlm52ArtifactGeometry *expected,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;

    status = SparkExpectUInt32("hf_config_geometry.hidden_size", expected->hidden_size, SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkExpectUInt32("hf_config_geometry.num_attention_heads", expected->head_count, SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkExpectUInt32("hf_config_geometry.kv_lora_rank", expected->latent_dimension, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkExpectUInt32("hf_config_geometry.qk_rope_head_dim", expected->rope_dimension, SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkExpectUInt32("hf_config_geometry.index_topk", expected->selected_token_count, SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT, error_buffer, error_buffer_bytes);
}

static SparkStatus SparkCheckModelDescription(
    const char *model_description_path,
    SparkGlm52ArtifactGeometry *expected_geometry,
    char *revision_buffer,
    uint32_t revision_buffer_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkModelDescription description;
    const SparkModelStageDescription *stage;
    const SparkModelProgramDescription *program;
    SparkStatus status;

    SparkModelDescriptionReset(&description);
    status = SparkLoadModelDescription(model_description_path, &description, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (description.model_revision == 0 || description.model_revision[0] == '\0')
    {
        status = SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "model.revision is missing");
    }
    else if (snprintf(revision_buffer, (size_t)revision_buffer_bytes, "%s", description.model_revision) >= (int)revision_buffer_bytes)
    {
        status = SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "model.revision is too long");
    }
    if (status == SPARK_STATUS_OK)
    {
        stage = SparkFindModelStage(&description, "resident_decode");
        if (stage == 0)
        {
            status = SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "resident_decode stage is missing");
        }
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkExpectString("resident_decode.target", stage->target, SPARK_GLM52_RESIDENT_DECODE_STAGE_TARGET, error_buffer, error_buffer_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        program = SparkFindModelProgram(stage, "decode");
        if (program == 0 || program->operation_count != 1u)
        {
            status = SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "decode program must have exactly one resident operation");
        }
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkExpectString("decode.operation.module", program->operations[0].module_id, SPARK_GLM52_RESIDENT_DECODE_STAGE_MODULE_ID, error_buffer, error_buffer_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkCheckMetadataGeometry(&description, error_buffer, error_buffer_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkReadArtifactGeometryFromMetadata(&description, expected_geometry, error_buffer, error_buffer_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkCheckArtifactGeometryAgainstFirmware(expected_geometry, error_buffer, error_buffer_bytes);
    }
    SparkModelDescriptionDestroy(&description);
    return status;
}

static int SparkBuildConfigPath(
    const char *model_directory,
    char *config_path,
    uint32_t config_path_bytes)
{
    int written_bytes;

    written_bytes = snprintf(config_path, (size_t)config_path_bytes, "%s/config.json", model_directory);
    if (written_bytes < 0 || (uint32_t)written_bytes >= config_path_bytes)
    {
        return -1;
    }
    return 0;
}

int main(int argument_count, char **arguments)
{
    SparkGlm52ArtifactGeometry expected_geometry;
    SparkGlm52ArtifactGeometry actual_geometry;
    const char *config_path;
    const char *model_directory;
    const char *model_description_path;
    char config_path_buffer[4096];
    char revision_buffer[256];
    char error_buffer[1024];
    int argument_index;
    SparkStatus status;

    memset(&expected_geometry, 0, sizeof(expected_geometry));
    memset(&actual_geometry, 0, sizeof(actual_geometry));
    memset(revision_buffer, 0, sizeof(revision_buffer));
    config_path = 0;
    model_directory = 0;
    model_description_path = "examples/model_descriptions/glm52_resident_decode_stage_firmware.json";
    for (argument_index = 1; argument_index < argument_count; ++argument_index)
    {
        const char *argument;

        argument = arguments[argument_index];
        if (strcmp(argument, "--config") == 0 && argument_index + 1 < argument_count)
        {
            config_path = arguments[++argument_index];
            continue;
        }
        if (strcmp(argument, "--model-dir") == 0 && argument_index + 1 < argument_count)
        {
            model_directory = arguments[++argument_index];
            continue;
        }
        if (strcmp(argument, "--model") == 0 && argument_index + 1 < argument_count)
        {
            model_description_path = arguments[++argument_index];
            continue;
        }
        SparkPrintUsage(arguments[0]);
        return 2;
    }
    if ((config_path == 0 && model_directory == 0) || (config_path != 0 && model_directory != 0))
    {
        SparkPrintUsage(arguments[0]);
        return 2;
    }
    if (model_directory != 0)
    {
        if (SparkBuildConfigPath(model_directory, config_path_buffer, sizeof(config_path_buffer)) != 0)
        {
            fprintf(stderr, "model-dir path is too long\n");
            return 2;
        }
        config_path = config_path_buffer;
    }
    status = SparkCheckModelDescription(
        model_description_path,
        &expected_geometry,
        revision_buffer,
        sizeof(revision_buffer),
        error_buffer,
        sizeof(error_buffer));
    if (status == SPARK_STATUS_OK)
    {
        status = SparkReadHfConfigGeometry(config_path, &expected_geometry, &actual_geometry, error_buffer, sizeof(error_buffer));
    }
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "glm52 artifact check failed: %s: %s\n", SparkStatusToString(status), error_buffer);
        return 1;
    }
    printf(
        "ready=1 config=%s model=%s revision=%s module=%s hidden_size=%u heads=%u kv_lora_rank=%u qk_rope_head_dim=%u index_topk=%u layers=%u vocab_size=%u n_routed_experts=%u n_shared_experts=%u experts_per_tok=%u qk_nope_head_dim=%u qk_head_dim=%u v_head_dim=%u\n",
        config_path,
        model_description_path,
        revision_buffer,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MODULE_ID,
        actual_geometry.hidden_size,
        actual_geometry.head_count,
        actual_geometry.latent_dimension,
        actual_geometry.rope_dimension,
        actual_geometry.selected_token_count,
        actual_geometry.layer_count,
        actual_geometry.vocab_size,
        actual_geometry.routed_expert_count,
        actual_geometry.shared_expert_count,
        actual_geometry.experts_per_token,
        actual_geometry.qk_nope_head_dimension,
        actual_geometry.qk_head_dimension,
        actual_geometry.value_head_dimension);
    return 0;
}
