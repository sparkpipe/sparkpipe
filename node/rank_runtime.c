#include "sparkpipe/spark_glm52_ring_runtime.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static const uint32_t SparkGlm52RingRuntimeDefaultLayerCounts[
    SPARK_GLM52_RING_RUNTIME_STAGE_COUNT] =
{
    6u, 6u, 6u, 6u, 6u, 6u, 6u,
    6u, 6u, 6u, 6u, 6u, 6u
};

static SparkStatus SparkGlm52RingRuntimeReport(
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

uint32_t SparkGlm52RingRuntimeDsaCandidateBucket(
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

uint32_t SparkGlm52RingRuntimeExecutionRowCapacity(
    uint32_t logical_lane_capacity)
{
    uint64_t execution_row_capacity;
    if (logical_lane_capacity == 0u ||
        logical_lane_capacity > SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET)
        return 0u;
    execution_row_capacity =
        (uint64_t)logical_lane_capacity *
        (uint64_t)SPARK_GLM52_RING_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE;
    if (execution_row_capacity > SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET)
        execution_row_capacity = SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET;
    return (uint32_t)execution_row_capacity;
}

SparkStatus SparkGlm52RingRuntimeParseQuantizationMode(
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

const char *SparkGlm52RingRuntimeQuantizationModeName(
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

SparkStatus SparkGlm52RingRuntimeValidateFp8PlanCounts(
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

SparkStatus SparkGlm52RingRuntimeExpectedMoeBackendKind(
    uint32_t quantization_mode,
    uint32_t *backend_kind_out)
{
    if (backend_kind_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
        *backend_kind_out =
            SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED;
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
        *backend_kind_out =
            SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_NVFP4_B12X;
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
        *backend_kind_out =
            SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_W8LUT_BF16_WMMA;
    else
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RingRuntimePathIsPresent(const char *path)
{
    struct stat path_status;

    if (path == 0 || path[0] == '\0')
    {
        return 0u;
    }
    return stat(path, &path_status) == 0 && path_status.st_size > 0 ? 1u : 0u;
}

static SparkStatus SparkGlm52RingRuntimeFormatRoute(
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

static void SparkGlm52RingRuntimeInitializeEndpoint(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t max_active_sequence_count,
    const char *route_name)
{
    memset(endpoint, 0, sizeof(*endpoint));
    endpoint->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    endpoint->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
    endpoint->capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS;
    endpoint->hidden_dimension = SPARK_GLM52_RING_RUNTIME_HIDDEN_DIMENSION;
    endpoint->bytes_per_sequence =
        SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
    endpoint->max_active_sequence_count = max_active_sequence_count;
    endpoint->max_packet_bytes =
        (uint64_t)SPARK_GLM52_RING_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW *
        (uint64_t)endpoint->max_active_sequence_count;
    endpoint->transport_module_id =
        SPARK_HIDDEN_TRANSPORT_TCP_CUDA_HOST_MODULE_ID;
    endpoint->route_name = route_name;
}

// Shape derivation over the authoritative model constants: one call site for
// the geometry and inputs so the plan, the sharder, and the packs cannot
// disagree. The latent KV element is one byte, the FP8 latent cache.
static SparkStatus SparkGlm52RingRuntimeShapeNodeConfig(
    const SparkGlm52TpShapeDescriptor *shape,
    SparkGlm52ShapeNodeConfig *config)
{
    SparkGlm52TpModelGeometry geometry;
    SparkGlm52ShapeModelInputs inputs;

    SparkGlm52TpModelGeometryFromModel(&geometry);
    memset(&inputs, 0, sizeof(inputs));
    inputs.abi_version = SPARK_GLM52_SHAPE_CONFIG_ABI_VERSION;
    inputs.total_layer_count = SPARK_GLM52_MODEL_LAYER_COUNT;
    inputs.hidden_dimension = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
    inputs.moe_intermediate_dimension =
        SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION;
    inputs.dense_intermediate_dimension =
        SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION;
    inputs.kv_latent_plus_rope_dimension =
        SPARK_GLM52_MODEL_LATENT_DIMENSION + SPARK_GLM52_MODEL_ROPE_DIMENSION;
    inputs.kv_bytes_per_element = 1u;
    return SparkGlm52ShapeDeriveNodeConfig(shape, &geometry, &inputs, config);
}

SparkStatus SparkGlm52RingRuntimeBuildFixedStagePlan(
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (stage_plan == 0)
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan is null");
    }
    return SparkGlm52StagePlanBuildFromLayerCounts(
        SparkGlm52RingRuntimeDefaultLayerCounts,
        SPARK_GLM52_RING_RUNTIME_STAGE_COUNT,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52RingRuntimeRankHostName(
    uint32_t rank_index,
    char *host_name,
    uint32_t host_name_bytes)
{
    int written;

    if (host_name == 0 || host_name_bytes == 0u ||
        rank_index >= SPARK_GLM52_RING_RUNTIME_STAGE_COUNT)
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

SparkStatus SparkGlm52RingRuntimeBuildRankPlan(
    uint32_t rank_index,
    uint32_t logical_lane_capacity,
    uint32_t port_base,
    uint32_t quantization_mode,
    SparkGlm52RingRuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkGlm52StagePlan stage_plan;
    SparkStatus status;

    if (rank_plan == 0 || logical_lane_capacity == 0u ||
        logical_lane_capacity > SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
        rank_index >= SPARK_GLM52_RING_RUNTIME_STAGE_COUNT ||
        (quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT &&
         quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT &&
         quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT) ||
        port_base > (65535u - SPARK_GLM52_RING_RUNTIME_STAGE_COUNT))
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "RING rank-plan arguments are invalid");
    }
    status = SparkGlm52RingRuntimeBuildFixedStagePlan(
        &stage_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    memset(rank_plan, 0, sizeof(*rank_plan));
    rank_plan->abi_version = SPARK_GLM52_RING_RUNTIME_ABI_VERSION;
    rank_plan->descriptor_bytes =
        SPARK_GLM52_RING_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES;
    rank_plan->rank_index = rank_index;
    rank_plan->first_layer_index =
        stage_plan.stages[rank_index].first_layer_index;
    rank_plan->layer_count = stage_plan.stages[rank_index].layer_count;
    rank_plan->previous_rank_index = UINT32_MAX;
    rank_plan->next_rank_index = UINT32_MAX;
    rank_plan->listen_port = port_base + rank_index;
    rank_plan->next_port = 0u;
    rank_plan->logical_lane_capacity = logical_lane_capacity;
    rank_plan->maximum_speculative_rows_per_lane =
        SPARK_GLM52_RING_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE;
    rank_plan->execution_row_capacity =
        SparkGlm52RingRuntimeExecutionRowCapacity(logical_lane_capacity);
    rank_plan->hidden_dimension = SPARK_GLM52_RING_RUNTIME_HIDDEN_DIMENSION;
    rank_plan->bytes_per_sequence =
        SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
    rank_plan->quantization_mode = quantization_mode;
    rank_plan->tp_degree = 1u;
    rank_plan->tp_rank = 0u;
    rank_plan->pp_stage_count = SPARK_GLM52_RING_RUNTIME_STAGE_COUNT;
    rank_plan->pp_stage_index = rank_index;
    rank_plan->tp_collective_listen_port = 0u;
    {
        SparkGlm52TpShapeDescriptor shape;
        SparkGlm52ShapeNodeConfig shape_config;
        memset(&shape, 0, sizeof(shape));
        shape.abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
        shape.tp_degree = 1u;
        shape.tp_rank = 0u;
        shape.pp_stage_count = SPARK_GLM52_RING_RUNTIME_STAGE_COUNT;
        shape.pp_stage_index = rank_index;
        status = SparkGlm52RingRuntimeShapeNodeConfig(&shape, &shape_config);
        if (status != SPARK_STATUS_OK ||
            shape_config.first_layer_index != rank_plan->first_layer_index ||
            shape_config.layer_count != rank_plan->layer_count)
        {
            return SparkGlm52RingRuntimeReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_VALIDATION_FAILED,
                "RING fixed stage plan disagrees with shape derivation");
        }
        rank_plan->shape_configuration_hash =
            shape_config.configuration_hash;
    }
    rank_plan->max_packet_bytes =
        (uint64_t)SPARK_GLM52_RING_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW *
        (uint64_t)rank_plan->execution_row_capacity;
    status = SparkGlm52RingRuntimeRankHostName(
        rank_index,
        rank_plan->host_name,
        sizeof(rank_plan->host_name));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (rank_index > 0u)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS;
        rank_plan->previous_rank_index = rank_index - 1u;
        status = SparkGlm52RingRuntimeRankHostName(
            rank_plan->previous_rank_index,
            rank_plan->previous_host_name,
            sizeof(rank_plan->previous_host_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52RingRuntimeFormatRoute(
            rank_plan->previous_host_name,
            rank_plan->host_name,
            rank_plan->input_route_name,
            sizeof(rank_plan->input_route_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52RingRuntimeInitializeEndpoint(
            &rank_plan->input_endpoint,
            rank_plan->execution_row_capacity,
            rank_plan->input_route_name);
    }
    if (rank_index + 1u < SPARK_GLM52_RING_RUNTIME_STAGE_COUNT)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT;
        rank_plan->next_rank_index = rank_index + 1u;
        rank_plan->next_port = port_base + rank_plan->next_rank_index;
        status = SparkGlm52RingRuntimeRankHostName(
            rank_plan->next_rank_index,
            rank_plan->next_host_name,
            sizeof(rank_plan->next_host_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52RingRuntimeFormatRoute(
            rank_plan->host_name,
            rank_plan->next_host_name,
            rank_plan->output_route_name,
            sizeof(rank_plan->output_route_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52RingRuntimeInitializeEndpoint(
            &rank_plan->output_endpoint,
            rank_plan->execution_row_capacity,
            rank_plan->output_route_name);
    }
    if ((stage_plan.stages[rank_index].flags &
            SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE;
    }
    if ((stage_plan.stages[rank_index].flags &
            SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX) != 0u)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_DENSE_PREFIX;
    }
    return SparkGlm52RingRuntimeValidateRankPlan(
        rank_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52RingRuntimeBuildShapeRankPlan(
    const SparkGlm52TpShapeDescriptor *shape,
    uint32_t logical_lane_capacity,
    uint32_t port_base,
    uint32_t tp_port_base,
    uint32_t quantization_mode,
    SparkGlm52RingRuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkGlm52ShapeNodeConfig shape_config;
    SparkStatus status;
    uint32_t node_index,step_index,step_count;

    if (shape == 0 || rank_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52RingRuntimeShapeNodeConfig(shape, &shape_config);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            status,
            "RING shape does not derive a node configuration");
    }
    node_index = shape->pp_stage_index * shape->tp_degree + shape->tp_rank;
    memset(rank_plan, 0, sizeof(*rank_plan));
    rank_plan->abi_version = SPARK_GLM52_RING_RUNTIME_ABI_VERSION;
    rank_plan->descriptor_bytes =
        SPARK_GLM52_RING_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES;
    rank_plan->rank_index = node_index;
    rank_plan->first_layer_index = shape_config.first_layer_index;
    rank_plan->layer_count = shape_config.layer_count;
    rank_plan->previous_rank_index = UINT32_MAX;
    rank_plan->next_rank_index = UINT32_MAX;
    rank_plan->listen_port = port_base + node_index;
    rank_plan->next_port = 0u;
    rank_plan->logical_lane_capacity = logical_lane_capacity;
    rank_plan->maximum_speculative_rows_per_lane =
        SPARK_GLM52_RING_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE;
    rank_plan->execution_row_capacity =
        SparkGlm52RingRuntimeExecutionRowCapacity(logical_lane_capacity);
    rank_plan->hidden_dimension = SPARK_GLM52_RING_RUNTIME_HIDDEN_DIMENSION;
    rank_plan->bytes_per_sequence =
        SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
    rank_plan->quantization_mode = quantization_mode;
    rank_plan->tp_degree = shape->tp_degree;
    rank_plan->tp_rank = shape->tp_rank;
    rank_plan->pp_stage_count = shape->pp_stage_count;
    rank_plan->pp_stage_index = shape->pp_stage_index;
    rank_plan->tp_collective_listen_port = tp_port_base + node_index;
    rank_plan->shape_configuration_hash = shape_config.configuration_hash;
    rank_plan->max_packet_bytes =
        (uint64_t)SPARK_GLM52_RING_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW *
        (uint64_t)rank_plan->execution_row_capacity;
    status = SparkGlm52RingRuntimeRankHostName(
        node_index,
        rank_plan->host_name,
        sizeof(rank_plan->host_name));
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            status,
            "RING shape node index exceeds the host table");
    }
    if (shape->pp_stage_index > 0u)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS;
        rank_plan->previous_rank_index = node_index - shape->tp_degree;
        status = SparkGlm52RingRuntimeRankHostName(
            rank_plan->previous_rank_index,
            rank_plan->previous_host_name,
            sizeof(rank_plan->previous_host_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52RingRuntimeFormatRoute(
            rank_plan->previous_host_name,
            rank_plan->host_name,
            rank_plan->input_route_name,
            sizeof(rank_plan->input_route_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52RingRuntimeInitializeEndpoint(
            &rank_plan->input_endpoint,
            rank_plan->execution_row_capacity,
            rank_plan->input_route_name);
    }
    if (shape->pp_stage_index + 1u < shape->pp_stage_count)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT;
        rank_plan->next_rank_index = node_index + shape->tp_degree;
        rank_plan->next_port = port_base + rank_plan->next_rank_index;
        status = SparkGlm52RingRuntimeRankHostName(
            rank_plan->next_rank_index,
            rank_plan->next_host_name,
            sizeof(rank_plan->next_host_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52RingRuntimeFormatRoute(
            rank_plan->host_name,
            rank_plan->next_host_name,
            rank_plan->output_route_name,
            sizeof(rank_plan->output_route_name));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52RingRuntimeInitializeEndpoint(
            &rank_plan->output_endpoint,
            rank_plan->execution_row_capacity,
            rank_plan->output_route_name);
    }
    step_count = 0u;
    while ((shape->tp_degree >> (step_count + 1u)) != 0u)
    {
        step_count += 1u;
    }
    for (step_index = 0u; step_index < step_count; ++step_index)
    {
        uint32_t partner_rank = shape->tp_rank ^ (1u << step_index);
        uint32_t partner_node =
            shape->pp_stage_index * shape->tp_degree + partner_rank;
        status = SparkGlm52RingRuntimeRankHostName(
            partner_node,
            rank_plan->tp_peer_host_names[step_index],
            sizeof(rank_plan->tp_peer_host_names[step_index]));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        rank_plan->tp_peer_ports[step_index] = tp_port_base + partner_node;
    }
    if (shape->pp_stage_index == 0u)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_DENSE_PREFIX;
    }
    if (shape->pp_stage_index + 1u == shape->pp_stage_count)
    {
        rank_plan->flags |= SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE;
    }
    return SparkGlm52RingRuntimeValidateRankPlan(
        rank_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52RingRuntimeValidateRankPlan(
    const SparkGlm52RingRuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkGlm52TpShapeDescriptor shape;
    SparkGlm52ShapeNodeConfig shape_config;
    SparkStatus status;

    if (rank_plan == 0)
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "RING rank plan is null");
    }
    memset(&shape, 0, sizeof(shape));
    shape.abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
    shape.tp_degree = rank_plan->tp_degree;
    shape.tp_rank = rank_plan->tp_rank;
    shape.pp_stage_count = rank_plan->pp_stage_count;
    shape.pp_stage_index = rank_plan->pp_stage_index;
    status = SparkGlm52RingRuntimeShapeNodeConfig(&shape, &shape_config);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            status,
            "RING rank plan shape does not derive");
    }

    if (rank_plan == 0 ||
        rank_plan->abi_version != SPARK_GLM52_RING_RUNTIME_ABI_VERSION ||
        rank_plan->descriptor_bytes !=
            SPARK_GLM52_RING_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES ||
        rank_plan->rank_index !=
            rank_plan->pp_stage_index * rank_plan->tp_degree +
                rank_plan->tp_rank ||
        rank_plan->rank_index >=
            rank_plan->pp_stage_count * rank_plan->tp_degree ||
        (rank_plan->flags & ~SPARK_GLM52_RING_RUNTIME_RANK_KNOWN_FLAGS) != 0u ||
        rank_plan->first_layer_index != shape_config.first_layer_index ||
        rank_plan->layer_count != shape_config.layer_count ||
        rank_plan->shape_configuration_hash !=
            shape_config.configuration_hash ||
        rank_plan->host_name[0] == '\0' ||
        rank_plan->logical_lane_capacity == 0u ||
        rank_plan->logical_lane_capacity >
            SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
        rank_plan->maximum_speculative_rows_per_lane !=
            SPARK_GLM52_RING_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE ||
        rank_plan->execution_row_capacity !=
            SparkGlm52RingRuntimeExecutionRowCapacity(
                rank_plan->logical_lane_capacity) ||
        rank_plan->hidden_dimension != SPARK_GLM52_RING_RUNTIME_HIDDEN_DIMENSION ||
        rank_plan->bytes_per_sequence !=
            SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE ||
        (rank_plan->quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT &&
         rank_plan->quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT &&
         rank_plan->quantization_mode !=
             SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT) ||
        rank_plan->max_packet_bytes !=
            ((uint64_t)SPARK_GLM52_RING_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW *
             (uint64_t)rank_plan->execution_row_capacity))
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "RING rank plan is invalid");
    }
    if ((rank_plan->flags & SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
    {
        status = SparkHiddenTransportValidateEndpoint(&rank_plan->input_endpoint);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52RingRuntimeReport(
                error_buffer,
                error_buffer_bytes,
                status,
                "RING input transport endpoint is invalid");
        }
    }
    if ((rank_plan->flags & SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
    {
        status = SparkHiddenTransportValidateEndpoint(&rank_plan->output_endpoint);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52RingRuntimeReport(
                error_buffer,
                error_buffer_bytes,
                status,
                "RING output transport endpoint is invalid");
        }
    }
    if (rank_plan->pp_stage_index == 0u &&
        (rank_plan->flags & SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (rank_plan->pp_stage_index + 1u == rank_plan->pp_stage_count &&
        (rank_plan->flags & SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52RingRuntimeReport(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}

SparkStatus SparkGlm52RingRuntimeBuildMoePackPath(
    const char *pack_root,
    uint32_t quantization_mode,
    uint32_t layer_index,
    uint32_t tp_degree,
    uint32_t tp_rank,
    char *pack_path,
    uint32_t pack_path_bytes)
{
    char shape_tag[24];
    int written;

    if (tp_degree == 0u || tp_rank >= tp_degree)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (tp_degree == 1u)
    {
        shape_tag[0] = '\0';
    }
    else if (snprintf(shape_tag, sizeof(shape_tag), "_tp%ur%u",
            tp_degree, tp_rank) <= 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

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
            "%s/glm52_layer_%04u_fp8_moe%s.spfp8",
            pack_root,
            layer_index,
            shape_tag);
    }
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
    {
        written = snprintf(
            pack_path,
            pack_path_bytes,
            "%s/glm52_layer_%04u_b12x_moe%s.spb12x",
            pack_root,
            layer_index,
            shape_tag);
    }
    else if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
    {
        written = snprintf(
            pack_path,
            pack_path_bytes,
            "%s/glm52_layer_%04u_w8lut_moe%s.spw8lut",
            pack_root,
            layer_index,
            shape_tag);
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

SparkStatus SparkGlm52RingRuntimeValidateStageMoePackFiles(
    const SparkGlm52RingRuntimeRankPlan *rank_plan,
    const char *pack_root,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char pack_path[SPARK_GLM52_RING_RUNTIME_PACK_PATH_BYTES];
    char manifest_path[SPARK_GLM52_RING_RUNTIME_PACK_PATH_BYTES];
    char foreign_manifest_path[SPARK_GLM52_RING_RUNTIME_PACK_PATH_BYTES];
    const char *manifest_names[3];
    SparkStatus status;
    uint32_t layer_index;
    uint32_t manifest_index;
    uint32_t selected_manifest_index;
    const char *manifest_name;
    int written;

    status = SparkGlm52RingRuntimeValidateRankPlan(
        rank_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (pack_root == 0 || pack_root[0] == '\0')
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "MoE pack root is empty");
    }
    manifest_names[0] = SPARK_GLM52_RING_RUNTIME_FP8_PACK_MANIFEST;
    manifest_names[1] = SPARK_GLM52_RING_RUNTIME_B12X_PACK_MANIFEST;
    manifest_names[2] = SPARK_GLM52_RING_RUNTIME_W8LUT_PACK_MANIFEST;
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
    if (!SparkGlm52RingRuntimePathIsPresent(manifest_path))
    {
        return SparkGlm52RingRuntimeReport(
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
        if (SparkGlm52RingRuntimePathIsPresent(foreign_manifest_path))
        {
            return SparkGlm52RingRuntimeReport(
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
        status = SparkGlm52RingRuntimeBuildMoePackPath(
            pack_root,
            rank_plan->quantization_mode,
            layer_index,
            rank_plan->tp_degree,
            rank_plan->tp_rank,
            pack_path,
            sizeof(pack_path));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (!SparkGlm52RingRuntimePathIsPresent(pack_path))
        {
            return SparkGlm52RingRuntimeReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_NOT_FOUND,
                "resident MoE layer pack is missing");
        }
    }
    return SparkGlm52RingRuntimeReport(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}

SparkStatus SparkGlm52RingRuntimeBuildFinalEventRoute(
    uint32_t port_base,
    SparkGlm52RingRuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int written;
    SparkStatus status;

    if (route == 0 ||
        port_base > (65535u - SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_PORT_OFFSET))
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "RING final event route arguments are invalid");
    }
    memset(route,0,sizeof(*route));
    route->abi_version = SPARK_GLM52_RING_RUNTIME_ABI_VERSION;
    route->descriptor_bytes =
        SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES;
    route->source_rank_index = SPARK_GLM52_RING_RUNTIME_STAGE_COUNT - 1u;
    route->sink_rank_index = 0u;
    route->listen_port =
        port_base + SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_PORT_OFFSET;
    route->connect_port = route->listen_port;
    status = SparkGlm52RingRuntimeRankHostName(
        route->source_rank_index,
        route->source_host_name,
        sizeof(route->source_host_name));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52RingRuntimeRankHostName(
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
    return SparkGlm52RingRuntimeValidateFinalEventRoute(
        route,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52RingRuntimeValidateFinalEventRoute(
    const SparkGlm52RingRuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    if (route == 0 ||
        route->abi_version != SPARK_GLM52_RING_RUNTIME_ABI_VERSION ||
        route->descriptor_bytes !=
            SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES ||
        route->source_rank_index !=
            SPARK_GLM52_RING_RUNTIME_STAGE_COUNT - 1u ||
        route->sink_rank_index != 0u ||
        route->listen_port == 0u ||
        route->connect_port != route->listen_port ||
        strcmp(route->source_host_name,"10.10.100.22") != 0 ||
        strcmp(route->sink_host_name,"10.10.100.10") != 0 ||
        strcmp(route->route_name,
            "10.10.100.22_to_10.10.100.10_final_events") != 0)
    {
        return SparkGlm52RingRuntimeReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "RING final event route is invalid");
    }
    return SparkGlm52RingRuntimeReport(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}
