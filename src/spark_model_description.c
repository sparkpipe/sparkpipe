#include "sparkpipe/spark_model_description.h"

#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_driver.h"

static SparkStatus SparkModelDescriptionCopyRequiredString(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    char **destination,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t member_token_index;
    SparkStatus status;

    member_token_index = SparkJsonFindObjectMember(document, object_token_index, member_name);
    if (member_token_index < 0 || !SparkJsonTokenIsType(document, member_token_index, SPARK_JSON_TOKEN_STRING))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "required string '%s' is missing", member_name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkJsonCopyString(document, member_token_index, destination);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot decode string '%s'", member_name);
        return status;
    }
    if ((*destination)[0] == '\0')
    {
        SparkSetError(error_buffer, error_buffer_bytes, "required string '%s' is empty", member_name);
        free(*destination);
        *destination = 0;
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkModelDescriptionGetRequiredObject(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    int32_t *member_token_index,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    *member_token_index = SparkJsonFindObjectMember(document, object_token_index, member_name);
    if (*member_token_index < 0 || !SparkJsonTokenIsType(document, *member_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "required object '%s' is missing", member_name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkModelDescriptionGetRequiredArray(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    int32_t *member_token_index,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    *member_token_index = SparkJsonFindObjectMember(document, object_token_index, member_name);
    if (*member_token_index < 0 || !SparkJsonTokenIsType(document, *member_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "required array '%s' is missing", member_name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    return SPARK_STATUS_OK;
}


static SparkStatus SparkModelDescriptionGetOptionalUInt32(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    uint32_t *destination,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t member_token_index;

    member_token_index = SparkJsonFindObjectMember(document, object_token_index, member_name);
    if (member_token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    if (SparkJsonGetUInt32(document, member_token_index, destination) != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "optional integer '%s' is invalid", member_name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkModelDescriptionGetOptionalUInt64(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    uint64_t *destination,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t member_token_index;

    member_token_index = SparkJsonFindObjectMember(document, object_token_index, member_name);
    if (member_token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    if (SparkJsonGetUInt64(document, member_token_index, destination) != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "optional integer '%s' is invalid", member_name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkParseModelProgramSchedulingFlag(
    const char *flag_name,
    uint32_t *flag)
{
    if (flag_name == 0 || flag == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (strcmp(flag_name, "stream_ordered") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "driver_owns_resident_state") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "driver_owns_kv_cache") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "jit_kv_cache") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_JIT_KV_CACHE;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "zero_copy_node_context") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_ZERO_COPY_NODE_CONTEXT;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "private_queue_pressure") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_PRIVATE_QUEUE_PRESSURE;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "no_host_staging") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_HOST_STAGING;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "fixed_firmware") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "validated_latency") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_VALIDATED_LATENCY;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "captured_cuda_graph") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_CAPTURED_CUDA_GRAPH;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "no_device_memcpy") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_DEVICE_MEMCPY;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "driver_private_expert_queues") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_PRIVATE_EXPERT_QUEUES;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "stream_event_dependencies") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_EVENT_DEPENDENCIES;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "residency_affinity_required") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_RESIDENCY_AFFINITY_REQUIRED;
        return SPARK_STATUS_OK;
    }
    if (strcmp(flag_name, "batch_shape_fixed") == 0)
    {
        *flag = SPARK_MODEL_DRIVER_PROGRAM_FLAG_BATCH_SHAPE_FIXED;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_SCHEMA_ERROR;
}

static SparkStatus SparkParseModelProgramScheduling(
    const SparkJsonDocument *document,
    int32_t program_token_index,
    SparkModelProgramDescription *program,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t scheduling_token_index;
    int32_t flags_token_index;
    uint32_t flag_index;
    SparkStatus status;

    scheduling_token_index = SparkJsonFindObjectMember(document, program_token_index, "scheduling");
    if (scheduling_token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    if (!SparkJsonTokenIsType(document, scheduling_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "program '%s' scheduling must be an object", program->name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }

    flags_token_index = SparkJsonFindObjectMember(document, scheduling_token_index, "flags");
    if (flags_token_index >= 0)
    {
        uint32_t flag_count;

        if (!SparkJsonTokenIsType(document, flags_token_index, SPARK_JSON_TOKEN_ARRAY))
        {
            SparkSetError(error_buffer, error_buffer_bytes, "program '%s' scheduling flags must be an array", program->name);
            return SPARK_STATUS_SCHEMA_ERROR;
        }
        flag_count = SparkJsonGetArrayElementCount(document, flags_token_index);
        for (flag_index = 0u; flag_index < flag_count; ++flag_index)
        {
            int32_t flag_token_index;
            char *flag_name;
            uint32_t parsed_flag;

            flag_token_index = SparkJsonGetArrayElement(document, flags_token_index, flag_index);
            if (!SparkJsonTokenIsType(document, flag_token_index, SPARK_JSON_TOKEN_STRING))
            {
                SparkSetError(error_buffer, error_buffer_bytes, "program '%s' scheduling flag must be a string", program->name);
                return SPARK_STATUS_SCHEMA_ERROR;
            }
            status = SparkJsonCopyString(document, flag_token_index, &flag_name);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            status = SparkParseModelProgramSchedulingFlag(flag_name, &parsed_flag);
            if (status != SPARK_STATUS_OK)
            {
                SparkSetError(error_buffer, error_buffer_bytes, "program '%s' has unknown scheduling flag '%s'", program->name, flag_name);
                free(flag_name);
                return SPARK_STATUS_SCHEMA_ERROR;
            }
            free(flag_name);
            program->scheduling.flags |= parsed_flag;
        }
    }

    status = SparkModelDescriptionGetOptionalUInt32(document, scheduling_token_index, "max_active_slots", &program->scheduling.max_active_slots, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt32(document, scheduling_token_index, "max_new_tokens", &program->scheduling.max_new_tokens, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt32(document, scheduling_token_index, "max_resident_sequences", &program->scheduling.max_resident_sequences, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "max_sequence_tokens", &program->scheduling.max_sequence_tokens, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "target_latency_ns", &program->scheduling.target_latency_ns, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "validated_latency_ns", &program->scheduling.validated_latency_ns, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "resident_weight_bytes", &program->scheduling.resident_weight_bytes, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "resident_kv_bytes", &program->scheduling.resident_kv_bytes, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "static_workspace_bytes", &program->scheduling.static_workspace_bytes, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "device_memcpy_bytes_per_submit_ceiling", &program->scheduling.device_memcpy_bytes_per_submit_ceiling, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetOptionalUInt64(document, scheduling_token_index, "host_staging_bytes_per_submit_ceiling", &program->scheduling.host_staging_bytes_per_submit_ceiling, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkModelDescriptionGetOptionalUInt32(document, scheduling_token_index, "private_queue_count", &program->scheduling.private_queue_count, error_buffer, error_buffer_bytes);
}

static void SparkModelOperationDescriptionDestroy(SparkModelOperationDescription *operation)
{
    if (operation == 0)
    {
        return;
    }
    free(operation->name);
    free(operation->module_id);
    free(operation->configuration_json);
    memset(operation, 0, sizeof(*operation));
}

static void SparkModelProgramDescriptionDestroy(SparkModelProgramDescription *program)
{
    uint32_t operation_index;

    if (program == 0)
    {
        return;
    }
    for (operation_index = 0u; operation_index < program->operation_count; ++operation_index)
    {
        SparkModelOperationDescriptionDestroy(&program->operations[operation_index]);
    }
    free(program->operations);
    free(program->name);
    memset(program, 0, sizeof(*program));
}

static void SparkModelStageDescriptionDestroy(SparkModelStageDescription *stage)
{
    uint32_t program_index;

    if (stage == 0)
    {
        return;
    }
    for (program_index = 0u; program_index < stage->program_count; ++program_index)
    {
        SparkModelProgramDescriptionDestroy(&stage->programs[program_index]);
    }
    free(stage->programs);
    free(stage->name);
    free(stage->target);
    memset(stage, 0, sizeof(*stage));
}

void SparkModelDescriptionReset(SparkModelDescription *description)
{
    if (description == 0)
    {
        return;
    }
    memset(description, 0, sizeof(*description));
}

void SparkModelDescriptionDestroy(SparkModelDescription *description)
{
    uint32_t stage_index;

    if (description == 0)
    {
        return;
    }
    for (stage_index = 0u; stage_index < description->stage_count; ++stage_index)
    {
        SparkModelStageDescriptionDestroy(&description->stages[stage_index]);
    }
    free(description->stages);
    free(description->model_id);
    free(description->model_revision);
    free(description->metadata_json);
    SparkModelDescriptionReset(description);
}

const char *SparkModelProgramCompletionModeToString(SparkModelProgramCompletionMode completion_mode)
{
    switch (completion_mode)
    {
        case SPARK_MODEL_PROGRAM_COMPLETION_SUBMIT_RETURN:
        {
            return "submit_return";
        }
        case SPARK_MODEL_PROGRAM_COMPLETION_EXTERNAL:
        {
            return "external";
        }
        default:
        {
            return "unknown";
        }
    }
}

static SparkStatus SparkParseModelProgramCompletionMode(
    const SparkJsonDocument *document,
    int32_t program_token_index,
    SparkModelProgramCompletionMode *completion_mode,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t completion_token_index;
    char *completion_text;
    SparkStatus status;

    completion_token_index = SparkJsonFindObjectMember(document, program_token_index, "completion");
    if (completion_token_index < 0)
    {
        *completion_mode = SPARK_MODEL_PROGRAM_COMPLETION_SUBMIT_RETURN;
        return SPARK_STATUS_OK;
    }
    if (!SparkJsonTokenIsType(document, completion_token_index, SPARK_JSON_TOKEN_STRING))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "program completion must be a string");
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkJsonCopyString(document, completion_token_index, &completion_text);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (strcmp(completion_text, "submit_return") == 0)
    {
        *completion_mode = SPARK_MODEL_PROGRAM_COMPLETION_SUBMIT_RETURN;
    }
    else if (strcmp(completion_text, "external") == 0)
    {
        *completion_mode = SPARK_MODEL_PROGRAM_COMPLETION_EXTERNAL;
    }
    else
    {
        SparkSetError(error_buffer, error_buffer_bytes, "unknown program completion mode '%s'", completion_text);
        free(completion_text);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    free(completion_text);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkParseModelOperation(
    const SparkJsonDocument *document,
    int32_t operation_token_index,
    SparkModelOperationDescription *operation,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t configuration_token_index;
    SparkStatus status;

    if (!SparkJsonTokenIsType(document, operation_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "operation entry must be an object");
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkModelDescriptionCopyRequiredString(document, operation_token_index, "name", &operation->name, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionCopyRequiredString(document, operation_token_index, "module", &operation->module_id, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    configuration_token_index = SparkJsonFindObjectMember(document, operation_token_index, "configuration");
    if (configuration_token_index < 0)
    {
        operation->configuration_json = (char *)malloc(3u);
        if (operation->configuration_json == 0)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        memcpy(operation->configuration_json, "{}", 3u);
        operation->configuration_json_bytes = 2u;
        return SPARK_STATUS_OK;
    }
    if (!SparkJsonTokenIsType(document, configuration_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "operation '%s' configuration must be an object", operation->name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkJsonCopyRawValue(document, configuration_token_index, &operation->configuration_json, &operation->configuration_json_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot copy configuration for operation '%s'", operation->name);
    }
    return status;
}

static SparkStatus SparkValidateUniqueOperationNames(
    const SparkModelProgramDescription *program,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t left_index;
    uint32_t right_index;

    for (left_index = 0u; left_index < program->operation_count; ++left_index)
    {
        for (right_index = left_index + 1u; right_index < program->operation_count; ++right_index)
        {
            if (strcmp(program->operations[left_index].name, program->operations[right_index].name) == 0)
            {
                SparkSetError(error_buffer, error_buffer_bytes, "program '%s' contains duplicate operation '%s'", program->name, program->operations[left_index].name);
                return SPARK_STATUS_DUPLICATE;
            }
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkParseModelProgram(
    const SparkJsonDocument *document,
    int32_t program_token_index,
    SparkModelProgramDescription *program,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t operations_token_index;
    int32_t member_token_index;
    uint32_t operation_index;
    SparkStatus status;

    if (!SparkJsonTokenIsType(document, program_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "program entry must be an object");
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkModelDescriptionCopyRequiredString(document, program_token_index, "name", &program->name, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    member_token_index = SparkJsonFindObjectMember(document, program_token_index, "id");
    if (member_token_index < 0 || SparkJsonGetUInt32(document, member_token_index, &program->program_id) != SPARK_STATUS_OK || program->program_id == 0u)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "program '%s' requires a nonzero integer id", program->name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    member_token_index = SparkJsonFindObjectMember(document, program_token_index, "max_inflight");
    if (member_token_index < 0)
    {
        program->max_inflight = 1u;
    }
    else if (SparkJsonGetUInt32(document, member_token_index, &program->max_inflight) != SPARK_STATUS_OK || program->max_inflight == 0u)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "program '%s' max_inflight must be nonzero", program->name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkParseModelProgramCompletionMode(document, program_token_index, &program->completion_mode, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkParseModelProgramScheduling(document, program_token_index, program, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetRequiredArray(document, program_token_index, "operations", &operations_token_index, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    program->operation_count = SparkJsonGetArrayElementCount(document, operations_token_index);
    if (program->operation_count == 0u || program->operation_count > SPARK_MODEL_DESCRIPTION_MAX_OPERATIONS_PER_PROGRAM)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "program '%s' operation count is outside the supported range", program->name);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    program->operations = (SparkModelOperationDescription *)calloc(program->operation_count, sizeof(*program->operations));
    if (program->operations == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    for (operation_index = 0u; operation_index < program->operation_count; ++operation_index)
    {
        int32_t operation_token_index;

        operation_token_index = SparkJsonGetArrayElement(document, operations_token_index, operation_index);
        status = SparkParseModelOperation(document, operation_token_index, &program->operations[operation_index], error_buffer, error_buffer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SparkValidateUniqueOperationNames(program, error_buffer, error_buffer_bytes);
}

static SparkStatus SparkValidateUniquePrograms(
    const SparkModelStageDescription *stage,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t left_index;
    uint32_t right_index;

    for (left_index = 0u; left_index < stage->program_count; ++left_index)
    {
        for (right_index = left_index + 1u; right_index < stage->program_count; ++right_index)
        {
            if (strcmp(stage->programs[left_index].name, stage->programs[right_index].name) == 0 ||
                stage->programs[left_index].program_id == stage->programs[right_index].program_id)
            {
                SparkSetError(error_buffer, error_buffer_bytes, "stage '%s' contains duplicate program name or id", stage->name);
                return SPARK_STATUS_DUPLICATE;
            }
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkParseModelStage(
    const SparkJsonDocument *document,
    int32_t stage_token_index,
    SparkModelStageDescription *stage,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t programs_token_index;
    uint32_t program_index;
    SparkStatus status;

    if (!SparkJsonTokenIsType(document, stage_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "stage entry must be an object");
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkModelDescriptionCopyRequiredString(document, stage_token_index, "name", &stage->name, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionCopyRequiredString(document, stage_token_index, "target", &stage->target, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModelDescriptionGetRequiredArray(document, stage_token_index, "programs", &programs_token_index, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    stage->program_count = SparkJsonGetArrayElementCount(document, programs_token_index);
    if (stage->program_count == 0u || stage->program_count > SPARK_MODEL_DESCRIPTION_MAX_PROGRAMS_PER_STAGE)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "stage '%s' program count is outside the supported range", stage->name);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    stage->programs = (SparkModelProgramDescription *)calloc(stage->program_count, sizeof(*stage->programs));
    if (stage->programs == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    for (program_index = 0u; program_index < stage->program_count; ++program_index)
    {
        int32_t program_token_index;

        program_token_index = SparkJsonGetArrayElement(document, programs_token_index, program_index);
        status = SparkParseModelProgram(document, program_token_index, &stage->programs[program_index], error_buffer, error_buffer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SparkValidateUniquePrograms(stage, error_buffer, error_buffer_bytes);
}

static SparkStatus SparkValidateUniqueStages(
    const SparkModelDescription *description,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t left_index;
    uint32_t right_index;

    for (left_index = 0u; left_index < description->stage_count; ++left_index)
    {
        for (right_index = left_index + 1u; right_index < description->stage_count; ++right_index)
        {
            if (strcmp(description->stages[left_index].name, description->stages[right_index].name) == 0)
            {
                SparkSetError(error_buffer, error_buffer_bytes, "duplicate stage '%s'", description->stages[left_index].name);
                return SPARK_STATUS_DUPLICATE;
            }
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkLoadModelDescription(
    const char *path,
    SparkModelDescription *description,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t schema_token_index;
    int32_t model_token_index;
    int32_t stages_token_index;
    int32_t metadata_token_index;
    uint32_t stage_index;
    SparkStatus status;

    if (path == 0 || description == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        error_buffer[0] = '\0';
    }
    SparkModelDescriptionDestroy(description);
    SparkJsonDocumentReset(&document);

    status = SparkJsonLoadFile(path, &document);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot parse model description '%s'", path);
        return status;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    if (!SparkJsonTokenIsType(&document, root_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "model description root must be an object");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto fail;
    }

    schema_token_index = SparkJsonFindObjectMember(&document, root_token_index, "schema_version");
    if (schema_token_index < 0 || SparkJsonGetUInt32(&document, schema_token_index, &description->schema_version) != SPARK_STATUS_OK ||
        description->schema_version != SPARK_MODEL_DESCRIPTION_SCHEMA_VERSION)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "unsupported model description schema_version");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto fail;
    }
    status = SparkModelDescriptionGetRequiredObject(&document, root_token_index, "model", &model_token_index, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto fail;
    }
    status = SparkModelDescriptionCopyRequiredString(&document, model_token_index, "id", &description->model_id, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto fail;
    }
    status = SparkModelDescriptionCopyRequiredString(&document, model_token_index, "revision", &description->model_revision, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto fail;
    }

    metadata_token_index = SparkJsonFindObjectMember(&document, root_token_index, "metadata");
    if (metadata_token_index >= 0)
    {
        if (!SparkJsonTokenIsType(&document, metadata_token_index, SPARK_JSON_TOKEN_OBJECT))
        {
            SparkSetError(error_buffer, error_buffer_bytes, "metadata must be an object");
            status = SPARK_STATUS_SCHEMA_ERROR;
            goto fail;
        }
        status = SparkJsonCopyRawValue(&document, metadata_token_index, &description->metadata_json, &description->metadata_json_bytes);
        if (status != SPARK_STATUS_OK)
        {
            goto fail;
        }
    }

    status = SparkModelDescriptionGetRequiredArray(&document, root_token_index, "stages", &stages_token_index, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto fail;
    }
    description->stage_count = SparkJsonGetArrayElementCount(&document, stages_token_index);
    if (description->stage_count == 0u || description->stage_count > SPARK_MODEL_DESCRIPTION_MAX_STAGES)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "stage count is outside the supported range");
        status = SPARK_STATUS_CAPACITY_EXCEEDED;
        goto fail;
    }
    description->stages = (SparkModelStageDescription *)calloc(description->stage_count, sizeof(*description->stages));
    if (description->stages == 0)
    {
        status = SPARK_STATUS_INTERNAL_ERROR;
        goto fail;
    }
    for (stage_index = 0u; stage_index < description->stage_count; ++stage_index)
    {
        int32_t stage_token_index;

        stage_token_index = SparkJsonGetArrayElement(&document, stages_token_index, stage_index);
        status = SparkParseModelStage(&document, stage_token_index, &description->stages[stage_index], error_buffer, error_buffer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            goto fail;
        }
    }
    status = SparkValidateUniqueStages(description, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto fail;
    }
    status = SparkSha256File(path, description->source_sha256);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot hash model description '%s'", path);
        goto fail;
    }

    SparkJsonDocumentDestroy(&document);
    return SPARK_STATUS_OK;

fail:
    SparkJsonDocumentDestroy(&document);
    SparkModelDescriptionDestroy(description);
    return status;
}

const SparkModelStageDescription *SparkFindModelStage(const SparkModelDescription *description, const char *stage_name)
{
    uint32_t stage_index;

    if (description == 0 || stage_name == 0)
    {
        return 0;
    }
    for (stage_index = 0u; stage_index < description->stage_count; ++stage_index)
    {
        if (strcmp(description->stages[stage_index].name, stage_name) == 0)
        {
            return &description->stages[stage_index];
        }
    }
    return 0;
}

const SparkModelProgramDescription *SparkFindModelProgram(const SparkModelStageDescription *stage, const char *program_name)
{
    uint32_t program_index;

    if (stage == 0 || program_name == 0)
    {
        return 0;
    }
    for (program_index = 0u; program_index < stage->program_count; ++program_index)
    {
        if (strcmp(stage->programs[program_index].name, program_name) == 0)
        {
            return &stage->programs[program_index];
        }
    }
    return 0;
}
