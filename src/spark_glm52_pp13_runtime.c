#include "sparkpipe/spark_glm52_pp13_runtime.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static const uint32_t SparkGlm52Pp13RuntimeDefaultLayerCounts[
    SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT] =
{
    6u, 6u, 6u, 6u, 6u, 6u, 6u,
    6u, 6u, 6u, 6u, 6u, 6u
};

static SparkStatus SparkGlm52Pp13RuntimeReport(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    SparkStatus status,
    const char *message)
{
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        if (message == 0)
        {
            error_buffer[0] = '\0';
        }
        else
        {
            (void)snprintf(error_buffer, error_buffer_bytes, "%s", message);
        }
    }
    return status;
}

uint32_t SparkGlm52Pp13RuntimeDsaCandidateBucket(
    uint32_t context_token_count)
{
    uint32_t candidate_count;
    if (context_token_count == 0u ||
        context_token_count > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS)
        return 0u;
    candidate_count = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
    while (candidate_count < context_token_count &&
        candidate_count < SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS)
        candidate_count <<= 1u;
    if (candidate_count > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS)
        candidate_count = SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS;
    return candidate_count;
}

SparkStatus SparkGlm52Pp13RuntimeParseQuantizationMode(
    const char *name,
    uint32_t *quantization_mode_out)
{
    if (name == 0 || quantization_mode_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (strcmp(name,"fp8") == 0)
        *quantization_mode_out =
            SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
    else if (strcmp(name,"nvfp4") == 0)
        *quantization_mode_out =
            SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    else if (strcmp(name,"w8lut") == 0)
        *quantization_mode_out =
            SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT;
    else
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

const char *SparkGlm52Pp13RuntimeQuantizationModeName(
    uint32_t quantization_mode)
{
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
        return "fp8";
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
        return "nvfp4";
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
        return "w8lut";
    return 0;
}

SparkStatus SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
    uint32_t quantization_mode,
    uint32_t bound_plan_count,
    uint32_t expected_plan_count)
{
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
    {
        return expected_plan_count != 0u &&
            bound_plan_count == expected_plan_count
            ? SPARK_STATUS_OK : SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (quantization_mode ==
            SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT ||
        quantization_mode ==
            SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
    {
        return bound_plan_count == 0u && expected_plan_count == 0u
            ? SPARK_STATUS_OK : SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkGlm52Pp13RuntimeExpectedMoeBackendKind(
    uint32_t quantization_mode,
    uint32_t *backend_kind_out)
{
    if (backend_kind_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
        *backend_kind_out =
            SPARK_GLM52_PP13_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED;
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
        *backend_kind_out =
            SPARK_GLM52_PP13_RUNTIME_MOE_BACKEND_NVFP4_B12X;
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
        *backend_kind_out =
            SPARK_GLM52_PP13_RUNTIME_MOE_BACKEND_W8LUT_BF16_WMMA;
    else
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52Pp13RuntimePathIsPresent(const char *path)
{
    struct stat path_status;

    if (path == 0 || path[0] == '\0')
    {
        return 0u;
    }
    return stat(path, &path_status) == 0 && path_status.st_size > 0 ? 1u : 0u;
}

static SparkStatus SparkGlm52Pp13RuntimeFormatRoute(
    const char *left,
    const char *right,
    char *route_name,
    uint32_t route_name_bytes)
{
    int written;

    if (left == 0 || right == 0 || route_name == 0 || route_name_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    written = snprintf(route_name, route_name_bytes, "%s_to_%s_hidden", left, right);
    if (written < 0 || (uint32_t)written >= route_name_bytes)
    {
        route_name[0] = '\0';
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13RuntimeInitializeEndpoint(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t max_active_sequence_count,
    const char *route_name)
{
    memset(endpoint, 0, sizeof(*endpoint));
    endpoint->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    endpoint->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
    endpoint->capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS;
    endpoint->hidden_dimension = SPARK_GLM52_PP13_RUNTIME_HIDDEN_DIMENSION;
    endpoint->bytes_per_sequence =
        SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
    endpoint->max_active_sequence_count = max_active_sequence_count;
    endpoint->max_packet_bytes =
        (uint64_t)SPARK_GLM52_PP13_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW *
        (uint64_t)endpoint->max_active_sequence_count;
    endpoint->transport_module_id =
        SPARK_HIDDEN_TRANSPORT_TCP_CUDA_HOST_MODULE_ID;
    endpoint->route_name = route_name;
}

SparkStatus SparkGlm52Pp13RuntimeBuildFixedStagePlan(
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (stage_plan == 0)
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan is null");
    }
    return SparkGlm52StagePlanBuildFromLayerCounts(
        SparkGlm52Pp13RuntimeDefaultLayerCounts,
        SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52Pp13RuntimeRankHostName(
    uint32_t rank_index,
    char *host_name,
    uint32_t host_name_bytes)
{
    int written;

    if (host_name == 0 || host_name_bytes == 0u ||
        rank_index >= SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    written = snprintf(host_name, host_name_bytes, "10.10.100.%u",
        10u + rank_index);
    if (written < 0 || (uint32_t)written >= host_name_bytes)
    {
        host_name[0] = '\0';
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13RuntimeBuildRankPlan(
    uint32_t rank_index,
    uint32_t logical_lane_capacity,
    uint32_t port_base,
    uint32_t quantization_mode,
    SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkGlm52StagePlan stage_plan;
    uint64_t maximum_execution_row_count;
    SparkStatus status;

    if (rank_plan == 0 || logical_lane_capacity == 0u ||
        logical_lane_capacity > SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
        rank_index >= SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT ||
        (quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT &&
         quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT &&
         quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT) ||
        port_base > (65535u - SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT))
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "PP13 rank-plan arguments are invalid");
    }
    status = SparkGlm52Pp13RuntimeBuildFixedStagePlan(
        &stage_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    memset(rank_plan, 0, sizeof(*rank_plan));
    rank_plan->abi_version = SPARK_GLM52_PP13_RUNTIME_ABI_VERSION;
    rank_plan->descriptor_bytes =
        SPARK_GLM52_PP13_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES;
    rank_plan->rank_index = rank_index;
    rank_plan->first_layer_index =
        stage_plan.stages[rank_index].first_layer_index;
    rank_plan->layer_count = stage_plan.stages[rank_index].layer_count;
    rank_plan->previous_rank_index = UINT32_MAX;
    rank_plan->next_rank_index = UINT32_MAX;
    rank_plan->listen_port = port_base + rank_index;
    rank_plan->next_port = 0u;
    maximum_execution_row_count =
        (uint64_t)logical_lane_capacity *
        (uint64_t)SPARK_GLM52_PP13_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE;
    if (maximum_execution_row_count > UINT32_MAX)
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_CAPACITY_EXCEEDED,
            "PP13 execution-row capacity overflows u32");
    }
    rank_plan->logical_lane_capacity = logical_lane_capacity;
    rank_plan->maximum_speculative_rows_per_lane =
        SPARK_GLM52_PP13_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE;
    rank_plan->execution_row_capacity =
        (uint32_t)maximum_execution_row_count;
    rank_plan->hidden_dimension = SPARK_GLM52_PP13_RUNTIME_HIDDEN_DIMENSION;
    rank_plan->bytes_per_sequence =
        SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
    rank_plan->quantization_mode = quantization_mode;
    rank_plan->max_packet_bytes =
        (uint64_t)SPARK_GLM52_PP13_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW *
        (uint64_t)rank_plan->execution_row_capacity;
    status = SparkGlm52Pp13RuntimeRankHostName(
        rank_index,
        rank_plan->host_name,
        sizeof(rank_plan->host_name));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (rank_index > 0u)
    {
        rank_plan->flags |= SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS;
        rank_plan->previous_rank_index = rank_index - 1u;
        status = SparkGlm52Pp13RuntimeRankHostName(
            rank_plan->previous_rank_index,
            rank_plan->previous_host_name,
            sizeof(rank_plan->previous_host_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52Pp13RuntimeFormatRoute(
            rank_plan->previous_host_name,
            rank_plan->host_name,
            rank_plan->input_route_name,
            sizeof(rank_plan->input_route_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52Pp13RuntimeInitializeEndpoint(
            &rank_plan->input_endpoint,
            rank_plan->execution_row_capacity,
            rank_plan->input_route_name);
    }
    if (rank_index + 1u < SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT)
    {
        rank_plan->flags |= SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT;
        rank_plan->next_rank_index = rank_index + 1u;
        rank_plan->next_port = port_base + rank_plan->next_rank_index;
        status = SparkGlm52Pp13RuntimeRankHostName(
            rank_plan->next_rank_index,
            rank_plan->next_host_name,
            sizeof(rank_plan->next_host_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52Pp13RuntimeFormatRoute(
            rank_plan->host_name,
            rank_plan->next_host_name,
            rank_plan->output_route_name,
            sizeof(rank_plan->output_route_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52Pp13RuntimeInitializeEndpoint(
            &rank_plan->output_endpoint,
            rank_plan->execution_row_capacity,
            rank_plan->output_route_name);
    }
    if ((stage_plan.stages[rank_index].flags &
            SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u)
    {
        rank_plan->flags |= SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE;
    }
    if ((stage_plan.stages[rank_index].flags &
            SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX) != 0u)
    {
        rank_plan->flags |= SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_DENSE_PREFIX;
    }
    return SparkGlm52Pp13RuntimeValidateRankPlan(
        rank_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52Pp13RuntimeValidateRankPlan(
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkGlm52StagePlan stage_plan;
    SparkStatus status;

    status = SparkGlm52Pp13RuntimeBuildFixedStagePlan(
        &stage_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (rank_plan == 0 ||
        rank_plan->abi_version != SPARK_GLM52_PP13_RUNTIME_ABI_VERSION ||
        rank_plan->descriptor_bytes !=
            SPARK_GLM52_PP13_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES ||
        rank_plan->rank_index >= SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT ||
        (rank_plan->flags & ~SPARK_GLM52_PP13_RUNTIME_RANK_KNOWN_FLAGS) != 0u ||
        rank_plan->first_layer_index !=
            stage_plan.stages[rank_plan->rank_index].first_layer_index ||
        rank_plan->layer_count !=
            stage_plan.stages[rank_plan->rank_index].layer_count ||
        rank_plan->host_name[0] == '\0' ||
        rank_plan->logical_lane_capacity == 0u ||
        rank_plan->logical_lane_capacity >
            SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
        rank_plan->maximum_speculative_rows_per_lane !=
            SPARK_GLM52_PP13_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE ||
        rank_plan->execution_row_capacity !=
            rank_plan->logical_lane_capacity *
                rank_plan->maximum_speculative_rows_per_lane ||
        rank_plan->hidden_dimension != SPARK_GLM52_PP13_RUNTIME_HIDDEN_DIMENSION ||
        rank_plan->bytes_per_sequence !=
            SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE ||
        (rank_plan->quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT &&
         rank_plan->quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT &&
         rank_plan->quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT) ||
        rank_plan->max_packet_bytes !=
            ((uint64_t)SPARK_GLM52_PP13_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW *
             (uint64_t)rank_plan->execution_row_capacity))
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "PP13 rank plan is invalid");
    }
    if ((rank_plan->flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
    {
        status = SparkHiddenTransportValidateEndpoint(&rank_plan->input_endpoint);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52Pp13RuntimeReport(
                error_buffer,
                error_buffer_bytes,
                status,
                "PP13 input transport endpoint is invalid");
        }
    }
    if ((rank_plan->flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
    {
        status = SparkHiddenTransportValidateEndpoint(&rank_plan->output_endpoint);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52Pp13RuntimeReport(
                error_buffer,
                error_buffer_bytes,
                status,
                "PP13 output transport endpoint is invalid");
        }
    }
    if (rank_plan->rank_index == 0u &&
        (rank_plan->flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (rank_plan->rank_index + 1u == SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT &&
        (rank_plan->flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52Pp13RuntimeReport(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}

SparkStatus SparkGlm52Pp13RuntimeBuildMoePackPath(
    const char *pack_root,
    uint32_t quantization_mode,
    uint32_t layer_index,
    char *pack_path,
    uint32_t pack_path_bytes)
{
    int written;

    if (pack_root == 0 || pack_root[0] == '\0' || pack_path == 0 ||
        pack_path_bytes == 0u ||
        layer_index >= SPARK_GLM52_MODEL_WEIGHT_LAYER_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
    {
        written = snprintf(
            pack_path,
            pack_path_bytes,
            "%s/glm52_layer_%04u_fp8_moe.spfp8",
            pack_root,
            layer_index);
    }
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
    {
        written = snprintf(
            pack_path,
            pack_path_bytes,
            "%s/glm52_layer_%04u_b12x_moe.spb12x",
            pack_root,
            layer_index);
    }
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
    {
        written = snprintf(
            pack_path,
            pack_path_bytes,
            "%s/glm52_layer_%04u_w8lut_moe.spw8lut",
            pack_root,
            layer_index);
    }
    else
    {
        pack_path[0] = '\0';
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (written < 0 || (uint32_t)written >= pack_path_bytes)
    {
        pack_path[0] = '\0';
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    const char *pack_root,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char pack_path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
    char manifest_path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
    char foreign_manifest_path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
    const char *manifest_names[3];
    SparkStatus status;
    uint32_t layer_index;
    uint32_t manifest_index;
    uint32_t selected_manifest_index;
    const char *manifest_name;
    int written;

    status = SparkGlm52Pp13RuntimeValidateRankPlan(
        rank_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (pack_root == 0 || pack_root[0] == '\0')
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "MoE pack root is empty");
    }
    manifest_names[0] = SPARK_GLM52_PP13_RUNTIME_FP8_PACK_MANIFEST;
    manifest_names[1] = SPARK_GLM52_PP13_RUNTIME_B12X_PACK_MANIFEST;
    manifest_names[2] = SPARK_GLM52_PP13_RUNTIME_W8LUT_PACK_MANIFEST;
    if (rank_plan->quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
        selected_manifest_index = 0u;
    else if (rank_plan->quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
        selected_manifest_index = 1u;
    else
        selected_manifest_index = 2u;
    manifest_name = manifest_names[selected_manifest_index];
    written = snprintf(
        manifest_path,
        sizeof(manifest_path),
        "%s/%s",
        pack_root,
        manifest_name);
    if (written < 0 || (uint32_t)written >= sizeof(manifest_path))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (!SparkGlm52Pp13RuntimePathIsPresent(manifest_path))
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_NOT_FOUND,
            "resident MoE pack manifest is missing");
    }
    for (manifest_index = 0u; manifest_index < 3u; ++manifest_index)
    {
        if (manifest_index == selected_manifest_index)
            continue;
        written = snprintf(
            foreign_manifest_path,
            sizeof(foreign_manifest_path),
            "%s/%s",
            pack_root,
            manifest_names[manifest_index]);
        if (written < 0 || (uint32_t)written >= sizeof(foreign_manifest_path))
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        if (SparkGlm52Pp13RuntimePathIsPresent(foreign_manifest_path))
        {
            return SparkGlm52Pp13RuntimeReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_MODULE_NOT_VALIDATED,
                "resident MoE pack root mixes quantization formats");
        }
    }
    for (layer_index = rank_plan->first_layer_index;
         layer_index < rank_plan->first_layer_index + rank_plan->layer_count;
         ++layer_index)
    {
        if (layer_index < SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER)
        {
            continue;
        }
        status = SparkGlm52Pp13RuntimeBuildMoePackPath(
            pack_root,
            rank_plan->quantization_mode,
            layer_index,
            pack_path,
            sizeof(pack_path));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (!SparkGlm52Pp13RuntimePathIsPresent(pack_path))
        {
            return SparkGlm52Pp13RuntimeReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_NOT_FOUND,
                "resident MoE layer pack is missing");
        }
    }
    return SparkGlm52Pp13RuntimeReport(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}

SparkStatus SparkGlm52Pp13RuntimeBuildFinalEventRoute(
    uint32_t port_base,
    SparkGlm52Pp13RuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int written;
    SparkStatus status;

    if (route == 0 ||
        port_base > (65535u - SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_PORT_OFFSET))
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "PP13 final event route arguments are invalid");
    }
    memset(route,0,sizeof(*route));
    route->abi_version = SPARK_GLM52_PP13_RUNTIME_ABI_VERSION;
    route->descriptor_bytes =
        SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES;
    route->source_rank_index = SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT - 1u;
    route->sink_rank_index = 0u;
    route->listen_port =
        port_base + SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_PORT_OFFSET;
    route->connect_port = route->listen_port;
    status = SparkGlm52Pp13RuntimeRankHostName(
        route->source_rank_index,
        route->source_host_name,
        sizeof(route->source_host_name));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52Pp13RuntimeRankHostName(
        route->sink_rank_index,
        route->sink_host_name,
        sizeof(route->sink_host_name));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    written = snprintf(
        route->route_name,
        sizeof(route->route_name),
        "%s_to_%s_final_events",
        route->source_host_name,
        route->sink_host_name);
    if (written < 0 || (uint32_t)written >= sizeof(route->route_name))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SparkGlm52Pp13RuntimeValidateFinalEventRoute(
        route,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52Pp13RuntimeValidateFinalEventRoute(
    const SparkGlm52Pp13RuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (route == 0 ||
        route->abi_version != SPARK_GLM52_PP13_RUNTIME_ABI_VERSION ||
        route->descriptor_bytes !=
            SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES ||
        route->source_rank_index !=
            SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT - 1u ||
        route->sink_rank_index != 0u ||
        route->listen_port == 0u ||
        route->connect_port != route->listen_port ||
        strcmp(route->source_host_name,"10.10.100.22") != 0 ||
        strcmp(route->sink_host_name,"10.10.100.10") != 0 ||
        strcmp(route->route_name,
            "10.10.100.22_to_10.10.100.10_final_events") != 0)
    {
        return SparkGlm52Pp13RuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "PP13 final event route is invalid");
    }
    return SparkGlm52Pp13RuntimeReport(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}
