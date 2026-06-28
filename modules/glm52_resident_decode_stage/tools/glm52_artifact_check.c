#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_description.h"

#define SPARK_GLM52_TENSOR_CONTRACT_MAX_TENSORS 32u
#define SPARK_GLM52_TENSOR_CONTRACT_MAX_RANK 4u
#define SPARK_GLM52_SAFETENSORS_HEADER_MAX_BYTES (128ull * 1024ull * 1024ull)

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

typedef struct SparkGlm52TensorContract
{
    char *name;
    char *dtype;
    uint64_t shape[SPARK_GLM52_TENSOR_CONTRACT_MAX_RANK];
    uint32_t rank;
} SparkGlm52TensorContract;

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

static SparkStatus SparkCopyRequiredStringMember(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    char **value,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t member_token_index;
    SparkStatus status;

    member_token_index = SparkJsonFindObjectMember(document, object_token_index, member_name);
    if (member_token_index < 0)
    {
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "missing required string field %s", member_name);
    }
    status = SparkJsonCopyString(document, member_token_index, value);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSetToolError(status, error_buffer, error_buffer_bytes, "field %s is not a string", member_name);
    }
    return SPARK_STATUS_OK;
}

static void SparkDestroyTensorContracts(
    SparkGlm52TensorContract *contracts,
    uint32_t contract_count)
{
    uint32_t contract_index;

    if (contracts == 0)
    {
        return;
    }
    for (contract_index = 0u; contract_index < contract_count; ++contract_index)
    {
        free(contracts[contract_index].name);
        free(contracts[contract_index].dtype);
        memset(&contracts[contract_index], 0, sizeof(contracts[contract_index]));
    }
}

static uint64_t SparkDtypeElementBytes(const char *dtype)
{
    if (dtype != 0 && strcmp(dtype, "BF16") == 0)
    {
        return 2u;
    }
    if (dtype != 0 && strcmp(dtype, "F8_E4M3") == 0)
    {
        return 1u;
    }
    if (dtype != 0 && strcmp(dtype, "F32") == 0)
    {
        return 4u;
    }
    return 0u;
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

static SparkStatus SparkReadTensorContractShape(
    const SparkJsonDocument *document,
    int32_t tensor_token_index,
    SparkGlm52TensorContract *contract,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t shape_token_index;
    uint32_t dimension_index;
    SparkStatus status;

    shape_token_index = SparkJsonFindObjectMember(document, tensor_token_index, "shape");
    if (shape_token_index < 0 || !SparkJsonTokenIsType(document, shape_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "tensor contract shape is missing or invalid");
    }
    contract->rank = SparkJsonGetArrayElementCount(document, shape_token_index);
    if (contract->rank == 0u || contract->rank > SPARK_GLM52_TENSOR_CONTRACT_MAX_RANK)
    {
        return SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "tensor contract rank is unsupported");
    }
    for (dimension_index = 0u; dimension_index < contract->rank; ++dimension_index)
    {
        int32_t dimension_token_index;

        dimension_token_index = SparkJsonGetArrayElement(document, shape_token_index, dimension_index);
        status = SparkJsonGetUInt64(document, dimension_token_index, &contract->shape[dimension_index]);
        if (status != SPARK_STATUS_OK || contract->shape[dimension_index] == 0u)
        {
            return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "tensor contract shape dimension is invalid");
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReadTensorContractsFromMetadata(
    const SparkModelDescription *description,
    const char *contract_member_name,
    SparkGlm52TensorContract *contracts,
    uint32_t *contract_count,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t contracts_token_index;
    uint32_t tensor_index;
    SparkStatus status;

    *contract_count = 0u;
    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(description->metadata_json, description->metadata_json_bytes, &document);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSetToolError(status, error_buffer, error_buffer_bytes, "could not parse model description metadata");
    }
    root_token_index = SparkJsonGetRootToken(&document);
    contracts_token_index = SparkJsonFindObjectMember(&document, root_token_index, contract_member_name);
    if (contracts_token_index < 0 || !SparkJsonTokenIsType(&document, contracts_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        SparkJsonDocumentDestroy(&document);
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "model description metadata.%s is missing", contract_member_name);
    }
    *contract_count = SparkJsonGetArrayElementCount(&document, contracts_token_index);
    if (*contract_count == 0u || *contract_count > SPARK_GLM52_TENSOR_CONTRACT_MAX_TENSORS)
    {
        SparkJsonDocumentDestroy(&document);
        return SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "tensor contract count is unsupported");
    }
    for (tensor_index = 0u; tensor_index < *contract_count; ++tensor_index)
    {
        int32_t tensor_token_index;

        tensor_token_index = SparkJsonGetArrayElement(&document, contracts_token_index, tensor_index);
        if (!SparkJsonTokenIsType(&document, tensor_token_index, SPARK_JSON_TOKEN_OBJECT))
        {
            SparkJsonDocumentDestroy(&document);
            return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "tensor contract entry must be an object");
        }
        status = SparkCopyRequiredStringMember(&document, tensor_token_index, "name", &contracts[tensor_index].name, error_buffer, error_buffer_bytes);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkCopyRequiredStringMember(&document, tensor_token_index, "dtype", &contracts[tensor_index].dtype, error_buffer, error_buffer_bytes);
        }
        if (status == SPARK_STATUS_OK)
        {
            status = SparkReadTensorContractShape(&document, tensor_token_index, &contracts[tensor_index], error_buffer, error_buffer_bytes);
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkJsonDocumentDestroy(&document);
            SparkDestroyTensorContracts(contracts, tensor_index + 1u);
            *contract_count = 0u;
            return status;
        }
    }
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

static int SparkBuildModelPath(
    const char *model_directory,
    const char *leaf_name,
    char *path,
    uint32_t path_bytes)
{
    int written_bytes;

    written_bytes = snprintf(path, (size_t)path_bytes, "%s/%s", model_directory, leaf_name);
    if (written_bytes < 0 || (uint32_t)written_bytes >= path_bytes)
    {
        return -1;
    }
    return 0;
}

static int SparkBuildConfigPath(
    const char *model_directory,
    char *config_path,
    uint32_t config_path_bytes)
{
    return SparkBuildModelPath(model_directory, "config.json", config_path, config_path_bytes);
}

static SparkStatus SparkBuildTensorPath(
    const char *model_directory,
    const char *file_name,
    char *path,
    uint32_t path_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (SparkBuildModelPath(model_directory, file_name, path, path_bytes) != 0)
    {
        return SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "tensor path is too long");
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReadSafetensorsHeader(
    const char *path,
    char **header_text,
    uint32_t *header_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    FILE *file;
    uint8_t header_length_bytes[8];
    uint64_t header_length;
    uint32_t index;

    *header_text = 0;
    *header_bytes = 0u;
    file = fopen(path, "rb");
    if (file == 0)
    {
        return SparkSetToolError(SPARK_STATUS_IO_ERROR, error_buffer, error_buffer_bytes, "could not open safetensors file %s", path);
    }
    if (fread(header_length_bytes, 1u, sizeof(header_length_bytes), file) != sizeof(header_length_bytes))
    {
        fclose(file);
        return SparkSetToolError(SPARK_STATUS_IO_ERROR, error_buffer, error_buffer_bytes, "could not read safetensors header length from %s", path);
    }
    header_length = 0u;
    for (index = 0u; index < 8u; ++index)
    {
        header_length |= ((uint64_t)header_length_bytes[index]) << (8u * index);
    }
    if (header_length == 0u || header_length > SPARK_GLM52_SAFETENSORS_HEADER_MAX_BYTES)
    {
        fclose(file);
        return SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "safetensors header size is unsupported");
    }
    *header_text = (char *)malloc((size_t)header_length + 1u);
    if (*header_text == 0)
    {
        fclose(file);
        return SparkSetToolError(SPARK_STATUS_INTERNAL_ERROR, error_buffer, error_buffer_bytes, "could not allocate safetensors header buffer");
    }
    if (fread(*header_text, 1u, (size_t)header_length, file) != (size_t)header_length)
    {
        free(*header_text);
        *header_text = 0;
        fclose(file);
        return SparkSetToolError(SPARK_STATUS_IO_ERROR, error_buffer, error_buffer_bytes, "could not read safetensors header from %s", path);
    }
    fclose(file);
    (*header_text)[header_length] = '\0';
    *header_bytes = (uint32_t)header_length;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkExpectedTensorBytes(
    const SparkGlm52TensorContract *contract,
    uint64_t *expected_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint64_t element_count;
    uint64_t element_bytes;
    uint32_t dimension_index;

    element_bytes = SparkDtypeElementBytes(contract->dtype);
    if (element_bytes == 0u)
    {
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "unsupported tensor dtype %s", contract->dtype);
    }
    element_count = 1u;
    for (dimension_index = 0u; dimension_index < contract->rank; ++dimension_index)
    {
        if (contract->shape[dimension_index] > UINT64_MAX / element_count)
        {
            return SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "tensor shape byte count overflow");
        }
        element_count *= contract->shape[dimension_index];
    }
    if (element_count > UINT64_MAX / element_bytes)
    {
        return SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "tensor byte count overflow");
    }
    *expected_bytes = element_count * element_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCheckSafetensorsShape(
    const SparkJsonDocument *document,
    int32_t tensor_token_index,
    const SparkGlm52TensorContract *contract,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t shape_token_index;
    uint32_t dimension_index;
    uint32_t rank;
    SparkStatus status;

    shape_token_index = SparkJsonFindObjectMember(document, tensor_token_index, "shape");
    if (shape_token_index < 0 || !SparkJsonTokenIsType(document, shape_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "safetensors shape is missing for %s", contract->name);
    }
    rank = SparkJsonGetArrayElementCount(document, shape_token_index);
    if (rank != contract->rank)
    {
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "safetensors rank mismatch for %s", contract->name);
    }
    for (dimension_index = 0u; dimension_index < rank; ++dimension_index)
    {
        int32_t dimension_token_index;
        uint64_t dimension_value;

        dimension_token_index = SparkJsonGetArrayElement(document, shape_token_index, dimension_index);
        status = SparkJsonGetUInt64(document, dimension_token_index, &dimension_value);
        if (status != SPARK_STATUS_OK || dimension_value != contract->shape[dimension_index])
        {
            return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "safetensors shape mismatch for %s", contract->name);
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCheckSafetensorsOffsets(
    const SparkJsonDocument *document,
    int32_t tensor_token_index,
    const SparkGlm52TensorContract *contract,
    uint64_t *actual_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t offsets_token_index;
    int32_t start_token_index;
    int32_t end_token_index;
    uint64_t start_offset;
    uint64_t end_offset;
    uint64_t expected_bytes;
    SparkStatus status;

    expected_bytes = 0u;
    offsets_token_index = SparkJsonFindObjectMember(document, tensor_token_index, "data_offsets");
    if (offsets_token_index < 0 || SparkJsonGetArrayElementCount(document, offsets_token_index) != 2u)
    {
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "safetensors offsets are missing for %s", contract->name);
    }
    start_token_index = SparkJsonGetArrayElement(document, offsets_token_index, 0u);
    end_token_index = SparkJsonGetArrayElement(document, offsets_token_index, 1u);
    status = SparkJsonGetUInt64(document, start_token_index, &start_offset);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkJsonGetUInt64(document, end_token_index, &end_offset);
    }
    if (status != SPARK_STATUS_OK || end_offset < start_offset)
    {
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "safetensors offsets are invalid for %s", contract->name);
    }
    status = SparkExpectedTensorBytes(contract, &expected_bytes, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *actual_bytes = end_offset - start_offset;
    if (*actual_bytes != expected_bytes)
    {
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "safetensors byte span mismatch for %s", contract->name);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCheckSafetensorsTensor(
    const char *path,
    const SparkGlm52TensorContract *contract,
    uint64_t *tensor_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkJsonDocument document;
    char *header_text;
    uint32_t header_bytes;
    int32_t root_token_index;
    int32_t tensor_token_index;
    int32_t dtype_token_index;
    SparkStatus status;

    status = SparkReadSafetensorsHeader(path, &header_text, &header_bytes, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(header_text, header_bytes, &document);
    free(header_text);
    if (status != SPARK_STATUS_OK)
    {
        return SparkSetToolError(status, error_buffer, error_buffer_bytes, "could not parse safetensors header %s", path);
    }
    root_token_index = SparkJsonGetRootToken(&document);
    tensor_token_index = SparkJsonFindObjectMember(&document, root_token_index, contract->name);
    if (tensor_token_index < 0)
    {
        SparkJsonDocumentDestroy(&document);
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "tensor %s is missing from %s", contract->name, path);
    }
    dtype_token_index = SparkJsonFindObjectMember(&document, tensor_token_index, "dtype");
    if (dtype_token_index < 0 || !SparkJsonStringEquals(&document, dtype_token_index, contract->dtype))
    {
        SparkJsonDocumentDestroy(&document);
        return SparkSetToolError(SPARK_STATUS_SCHEMA_ERROR, error_buffer, error_buffer_bytes, "safetensors dtype mismatch for %s", contract->name);
    }
    status = SparkCheckSafetensorsShape(&document, tensor_token_index, contract, error_buffer, error_buffer_bytes);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkCheckSafetensorsOffsets(&document, tensor_token_index, contract, tensor_bytes, error_buffer, error_buffer_bytes);
    }
    SparkJsonDocumentDestroy(&document);
    return status;
}

static SparkStatus SparkCheckTensorContractMember(
    const char *model_directory,
    const SparkModelDescription *description,
    const char *contract_member_name,
    uint32_t *checked_tensor_count,
    uint64_t *checked_tensor_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkGlm52TensorContract contracts[SPARK_GLM52_TENSOR_CONTRACT_MAX_TENSORS];
    SparkJsonDocument index_document;
    char index_path[4096];
    uint32_t contract_count;
    uint32_t contract_index;
    int32_t root_token_index;
    int32_t weight_map_token_index;
    SparkStatus status;

    memset(contracts, 0, sizeof(contracts));
    *checked_tensor_count = 0u;
    *checked_tensor_bytes = 0u;
    status = SparkReadTensorContractsFromMetadata(description, contract_member_name, contracts, &contract_count, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkBuildModelPath(model_directory, "model.safetensors.index.json", index_path, sizeof(index_path)) != 0)
    {
        SparkDestroyTensorContracts(contracts, contract_count);
        return SparkSetToolError(SPARK_STATUS_CAPACITY_EXCEEDED, error_buffer, error_buffer_bytes, "safetensors index path is too long");
    }
    SparkJsonDocumentReset(&index_document);
    status = SparkJsonLoadFile(index_path, &index_document);
    if (status != SPARK_STATUS_OK)
    {
        SparkDestroyTensorContracts(contracts, contract_count);
        return SparkSetToolError(status, error_buffer, error_buffer_bytes, "could not load safetensors index %s", index_path);
    }
    root_token_index = SparkJsonGetRootToken(&index_document);
    weight_map_token_index = SparkJsonFindObjectMember(&index_document, root_token_index, "weight_map");
    if (weight_map_token_index < 0)
    {
        SparkJsonDocumentDestroy(&index_document);
        SparkDestroyTensorContracts(contracts, contract_count);
        return SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "safetensors index weight_map is missing");
    }
    for (contract_index = 0u; contract_index < contract_count; ++contract_index)
    {
        char *file_name;
        char tensor_path[4096];
        uint64_t tensor_bytes;
        int32_t file_token_index;

        file_name = 0;
        tensor_bytes = 0u;
        file_token_index = SparkJsonFindObjectMember(&index_document, weight_map_token_index, contracts[contract_index].name);
        if (file_token_index < 0)
        {
            status = SparkSetToolError(SPARK_STATUS_NOT_FOUND, error_buffer, error_buffer_bytes, "tensor %s is missing from safetensors index", contracts[contract_index].name);
        }
        else
        {
            status = SparkJsonCopyString(&index_document, file_token_index, &file_name);
        }
        if (status == SPARK_STATUS_OK)
        {
            status = SparkBuildTensorPath(model_directory, file_name, tensor_path, sizeof(tensor_path), error_buffer, error_buffer_bytes);
        }
        if (status == SPARK_STATUS_OK)
        {
            status = SparkCheckSafetensorsTensor(tensor_path, &contracts[contract_index], &tensor_bytes, error_buffer, error_buffer_bytes);
        }
        free(file_name);
        if (status != SPARK_STATUS_OK)
        {
            SparkJsonDocumentDestroy(&index_document);
            SparkDestroyTensorContracts(contracts, contract_count);
            return status;
        }
        *checked_tensor_count += 1u;
        *checked_tensor_bytes += tensor_bytes;
    }
    SparkJsonDocumentDestroy(&index_document);
    SparkDestroyTensorContracts(contracts, contract_count);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkCheckTensorContract(
    const char *model_directory,
    const SparkModelDescription *description,
    uint32_t *checked_tensor_count,
    uint64_t *checked_tensor_bytes,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char first_error_buffer[1024];
    char second_error_buffer[1024];
    SparkStatus first_status;
    SparkStatus second_status;

    memset(first_error_buffer, 0, sizeof(first_error_buffer));
    memset(second_error_buffer, 0, sizeof(second_error_buffer));
    first_status = SparkCheckTensorContractMember(
        model_directory,
        description,
        "hf_tensor_contract",
        checked_tensor_count,
        checked_tensor_bytes,
        first_error_buffer,
        sizeof(first_error_buffer));
    if (first_status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    second_status = SparkCheckTensorContractMember(
        model_directory,
        description,
        "hf_tensor_contract_fp8_e4m3",
        checked_tensor_count,
        checked_tensor_bytes,
        second_error_buffer,
        sizeof(second_error_buffer));
    if (second_status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    snprintf(
        error_buffer,
        (size_t)error_buffer_bytes,
        "no tensor contract matched: bf16=%s; fp8=%s",
        first_error_buffer,
        second_error_buffer);
    return second_status;
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
    uint32_t checked_tensor_count;
    uint64_t checked_tensor_bytes;
    int argument_index;
    SparkStatus status;

    memset(&expected_geometry, 0, sizeof(expected_geometry));
    memset(&actual_geometry, 0, sizeof(actual_geometry));
    memset(revision_buffer, 0, sizeof(revision_buffer));
    checked_tensor_count = 0u;
    checked_tensor_bytes = 0u;
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
    if (status == SPARK_STATUS_OK && model_directory != 0)
    {
        SparkModelDescription description;

        SparkModelDescriptionReset(&description);
        status = SparkLoadModelDescription(model_description_path, &description, error_buffer, sizeof(error_buffer));
        if (status == SPARK_STATUS_OK)
        {
            status = SparkCheckTensorContract(
                model_directory,
                &description,
                &checked_tensor_count,
                &checked_tensor_bytes,
                error_buffer,
                sizeof(error_buffer));
        }
        SparkModelDescriptionDestroy(&description);
    }
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "glm52 artifact check failed: %s: %s\n", SparkStatusToString(status), error_buffer);
        return 1;
    }
    printf(
        "ready=1 config=%s model=%s revision=%s module=%s hidden_size=%u heads=%u kv_lora_rank=%u qk_rope_head_dim=%u index_topk=%u layers=%u vocab_size=%u n_routed_experts=%u n_shared_experts=%u experts_per_tok=%u qk_nope_head_dim=%u qk_head_dim=%u v_head_dim=%u tensor_contract_ready=%u tensor_count=%u tensor_bytes=%llu\n",
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
        actual_geometry.value_head_dimension,
        model_directory != 0 ? 1u : 0u,
        checked_tensor_count,
        (unsigned long long)checked_tensor_bytes);
    return 0;
}
