#include "sparkpipe/spark_glm52_resident_decode_stage_linear_plan.h"

#include <cuda_runtime_api.h>
#include <cublasLt.h>

#include <float.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_GLM52_LINEAR_PLAN_MAX_HEURISTIC_RESULTS 32u

typedef struct SparkGlm52ResidentDecodeStageLinearPlanStorage
{
    uint32_t initialized;
    cublasLtMatmulDesc_t matmul_descriptor;
    cublasLtMatrixLayout_t input_layout;
    cublasLtMatrixLayout_t weight_layout;
    cublasLtMatrixLayout_t output_layout;
    cublasLtMatmulAlgo_t algorithm;
    void *workspace;
    uint64_t workspace_bytes;
} SparkGlm52ResidentDecodeStageLinearPlanStorage;

struct SparkGlm52ResidentDecodeStageLinearPlanResidentBinding
{
    uint32_t abi_version;
    uint32_t plan_count;
    cublasLtHandle_t cublaslt_handle;
    SparkGlm52ResidentDecodeStageLinearPlan plans[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT];
    SparkGlm52ResidentDecodeStageLinearPlanStorage storage[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT];
};

static SparkStatus SparkGlm52LinearPlanCudaToSparkStatus(
    cudaError_t cuda_status)
{
    if (cuda_status == cudaSuccess)
    {
        return SPARK_STATUS_OK;
    }
    if (cuda_status == cudaErrorMemoryAllocation)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52LinearPlanCublasToSparkStatus(
    cublasStatus_t cublas_status)
{
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        return SPARK_STATUS_OK;
    }
    if (cublas_status == CUBLAS_STATUS_ALLOC_FAILED)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (cublas_status == CUBLAS_STATUS_INVALID_VALUE ||
        cublas_status == CUBLAS_STATUS_NOT_SUPPORTED)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52LinearPlanCheckedMultiplyU64(
    uint64_t left,
    uint64_t right,
    uint64_t *product_out)
{
    if (product_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (left != 0u && right > UINT64_MAX / left)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *product_out = left * right;
    return SPARK_STATUS_OK;
}

static void SparkGlm52LinearPlanDestroyStorage(
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage)
{
    if (storage == 0)
    {
        return;
    }
    if (storage->workspace != 0)
    {
        cudaFree(storage->workspace);
    }
    if (storage->output_layout != 0)
    {
        cublasLtMatrixLayoutDestroy(storage->output_layout);
    }
    if (storage->weight_layout != 0)
    {
        cublasLtMatrixLayoutDestroy(storage->weight_layout);
    }
    if (storage->input_layout != 0)
    {
        cublasLtMatrixLayoutDestroy(storage->input_layout);
    }
    if (storage->matmul_descriptor != 0)
    {
        cublasLtMatmulDescDestroy(storage->matmul_descriptor);
    }
    memset(storage, 0, sizeof(*storage));
}

static SparkStatus SparkGlm52LinearPlanSetRowMajorLayout(
    cublasLtMatrixLayout_t layout)
{
    cublasLtOrder_t order;
    cublasStatus_t cublas_status;

    order = CUBLASLT_ORDER_ROW;
    cublas_status = cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order));
    return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
}

static SparkStatus SparkGlm52LinearPlanCreateDescriptors(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_is_f32)
{
    cublasOperation_t transa;
    cublasOperation_t transb;
    cudaDataType_t output_type;
    cublasComputeType_t compute_type;
    cublasStatus_t cublas_status;
    SparkStatus status;

    if (binding == 0 || storage == 0 ||
        active_sequence_count == 0u ||
        input_dimension == 0u ||
        output_dimension == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    output_type = output_is_f32 != 0u ? CUDA_R_32F : CUDA_R_16BF;
    compute_type = CUBLAS_COMPUTE_32F;
    cublas_status = cublasLtMatmulDescCreate(
        &storage->matmul_descriptor,
        compute_type,
        CUDA_R_32F);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }

    transa = CUBLAS_OP_N;
    transb = CUBLAS_OP_T;
    cublas_status = cublasLtMatmulDescSetAttribute(
        storage->matmul_descriptor,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transa,
        sizeof(transa));
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }
    cublas_status = cublasLtMatmulDescSetAttribute(
        storage->matmul_descriptor,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transb,
        sizeof(transb));
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }

    cublas_status = cublasLtMatrixLayoutCreate(
        &storage->input_layout,
        CUDA_R_16BF,
        active_sequence_count,
        input_dimension,
        input_dimension);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }
    status = SparkGlm52LinearPlanSetRowMajorLayout(storage->input_layout);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cublas_status = cublasLtMatrixLayoutCreate(
        &storage->weight_layout,
        CUDA_R_16BF,
        output_dimension,
        input_dimension,
        input_dimension);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }
    status = SparkGlm52LinearPlanSetRowMajorLayout(storage->weight_layout);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cublas_status = cublasLtMatrixLayoutCreate(
        &storage->output_layout,
        output_type,
        active_sequence_count,
        output_dimension,
        output_dimension);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }
    return SparkGlm52LinearPlanSetRowMajorLayout(storage->output_layout);
}

static SparkStatus SparkGlm52LinearPlanLaunchCandidate(
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    const SparkGlm52ResidentDecodeStageLinearPlanStorage *storage,
    const cublasLtMatmulAlgo_t *algorithm,
    const void *input,
    const void *weight,
    void *output,
    void *workspace,
    uint64_t workspace_bytes,
    cudaStream_t cuda_stream)
{
    cublasStatus_t cublas_status;
    float alpha;
    float beta;

    if (binding == 0 || storage == 0 || algorithm == 0 ||
        input == 0 || weight == 0 || output == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    alpha = 1.0f;
    beta = 0.0f;
    cublas_status = cublasLtMatmul(
        binding->cublaslt_handle,
        storage->matmul_descriptor,
        &alpha,
        input,
        storage->input_layout,
        weight,
        storage->weight_layout,
        &beta,
        output,
        storage->output_layout,
        output,
        storage->output_layout,
        algorithm,
        workspace,
        (size_t)workspace_bytes,
        cuda_stream);
    return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
}

static SparkStatus SparkGlm52LinearPlanMeasureCandidate(
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    const SparkGlm52ResidentDecodeStageLinearPlanStorage *storage,
    const cublasLtMatmulAlgo_t *algorithm,
    const void *input,
    const void *weight,
    void *output,
    void *workspace,
    uint64_t workspace_bytes,
    cudaStream_t cuda_stream,
    uint32_t warmup_iterations,
    uint32_t measurement_iterations,
    float *average_milliseconds_out)
{
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    SparkStatus status;
    cudaError_t cuda_status;
    float elapsed_milliseconds;
    uint32_t iteration;

    if (average_milliseconds_out == 0 || measurement_iterations == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *average_milliseconds_out = FLT_MAX;

    for (iteration = 0u; iteration < warmup_iterations; ++iteration)
    {
        status = SparkGlm52LinearPlanLaunchCandidate(
            binding,
            storage,
            algorithm,
            input,
            weight,
            output,
            workspace,
            workspace_bytes,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    cuda_status = cudaStreamSynchronize(cuda_stream);
    if (cuda_status != cudaSuccess)
    {
        return SparkGlm52LinearPlanCudaToSparkStatus(cuda_status);
    }

    start_event = 0;
    stop_event = 0;
    cuda_status = cudaEventCreate(&start_event);
    if (cuda_status != cudaSuccess)
    {
        return SparkGlm52LinearPlanCudaToSparkStatus(cuda_status);
    }
    cuda_status = cudaEventCreate(&stop_event);
    if (cuda_status != cudaSuccess)
    {
        cudaEventDestroy(start_event);
        return SparkGlm52LinearPlanCudaToSparkStatus(cuda_status);
    }

    cuda_status = cudaEventRecord(start_event, cuda_stream);
    if (cuda_status == cudaSuccess)
    {
        for (iteration = 0u; iteration < measurement_iterations; ++iteration)
        {
            status = SparkGlm52LinearPlanLaunchCandidate(
                binding,
                storage,
                algorithm,
                input,
                weight,
                output,
                workspace,
                workspace_bytes,
                cuda_stream);
            if (status != SPARK_STATUS_OK)
            {
                cudaEventDestroy(stop_event);
                cudaEventDestroy(start_event);
                return status;
            }
        }
        cuda_status = cudaEventRecord(stop_event, cuda_stream);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaEventSynchronize(stop_event);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaEventElapsedTime(
            &elapsed_milliseconds,
            start_event,
            stop_event);
    }

    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    if (cuda_status != cudaSuccess)
    {
        return SparkGlm52LinearPlanCudaToSparkStatus(cuda_status);
    }

    *average_milliseconds_out = elapsed_milliseconds /
        (float)measurement_iterations;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52LinearPlanSelectAlgorithm(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage,
    const void *input,
    const void *weight,
    void *output,
    uint64_t workspace_limit_bytes,
    cudaStream_t cuda_stream,
    uint32_t warmup_iterations,
    uint32_t measurement_iterations,
    uint64_t *selected_workspace_bytes_out)
{
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic_results[
        SPARK_GLM52_LINEAR_PLAN_MAX_HEURISTIC_RESULTS];
    cublasStatus_t cublas_status;
    SparkStatus status;
    void *tuning_workspace;
    int returned_results;
    int result_index;
    int best_result_index;
    float best_average_milliseconds;

    if (binding == 0 || storage == 0 || selected_workspace_bytes_out == 0 ||
        input == 0 || weight == 0 || output == 0 ||
        measurement_iterations == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *selected_workspace_bytes_out = 0u;
    preference = 0;
    tuning_workspace = 0;
    returned_results = 0;
    best_result_index = -1;
    best_average_milliseconds = FLT_MAX;

    cublas_status = cublasLtMatmulPreferenceCreate(&preference);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }
    cublas_status = cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit_bytes,
        sizeof(workspace_limit_bytes));
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        cublasLtMatmulPreferenceDestroy(preference);
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }

    cublas_status = cublasLtMatmulAlgoGetHeuristic(
        binding->cublaslt_handle,
        storage->matmul_descriptor,
        storage->input_layout,
        storage->weight_layout,
        storage->output_layout,
        storage->output_layout,
        preference,
        (int)SPARK_GLM52_LINEAR_PLAN_MAX_HEURISTIC_RESULTS,
        heuristic_results,
        &returned_results);
    cublasLtMatmulPreferenceDestroy(preference);
    if (cublas_status != CUBLAS_STATUS_SUCCESS || returned_results <= 0)
    {
        return cublas_status == CUBLAS_STATUS_SUCCESS
            ? SPARK_STATUS_INVALID_ARGUMENT
            : SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }

    if (workspace_limit_bytes != 0u)
    {
        status = SparkGlm52LinearPlanCudaToSparkStatus(
            cudaMalloc(&tuning_workspace, (size_t)workspace_limit_bytes));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    for (result_index = 0; result_index < returned_results; ++result_index)
    {
        float average_milliseconds;

        if (heuristic_results[result_index].state != CUBLAS_STATUS_SUCCESS ||
            (uint64_t)heuristic_results[result_index].workspaceSize >
                workspace_limit_bytes)
        {
            continue;
        }

        status = SparkGlm52LinearPlanMeasureCandidate(
            binding,
            storage,
            &heuristic_results[result_index].algo,
            input,
            weight,
            output,
            tuning_workspace,
            (uint64_t)heuristic_results[result_index].workspaceSize,
            cuda_stream,
            warmup_iterations,
            measurement_iterations,
            &average_milliseconds);
        if (status == SPARK_STATUS_OK &&
            average_milliseconds < best_average_milliseconds)
        {
            best_average_milliseconds = average_milliseconds;
            best_result_index = result_index;
        }
    }

    if (tuning_workspace != 0)
    {
        cudaFree(tuning_workspace);
    }
    if (best_result_index < 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    storage->algorithm = heuristic_results[best_result_index].algo;
    *selected_workspace_bytes_out =
        (uint64_t)heuristic_results[best_result_index].workspaceSize;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52LinearPlanCreateOne(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_is_f32,
    const void *input,
    const void *weight,
    void *output,
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo *create_info)
{
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage;
    SparkGlm52ResidentDecodeStageLinearPlan *plan;
    uint64_t selected_workspace_bytes;
    SparkStatus status;

    if (binding == 0 || create_info == 0 ||
        plan_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT ||
        input == 0 || weight == 0 || output == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    storage = &binding->storage[plan_index];
    plan = &binding->plans[plan_index];
    status = SparkGlm52LinearPlanCreateDescriptors(
        binding,
        storage,
        create_info->maximum_active_sequence_count,
        input_dimension,
        output_dimension,
        output_is_f32);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52LinearPlanSelectAlgorithm(
        binding,
        storage,
        input,
        weight,
        output,
        create_info->workspace_limit_bytes,
        (cudaStream_t)create_info->cuda_stream,
        create_info->autotune_warmup_iterations,
        create_info->autotune_measurement_iterations,
        &selected_workspace_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (selected_workspace_bytes != 0u)
    {
        status = SparkGlm52LinearPlanCudaToSparkStatus(
            cudaMalloc(&storage->workspace, (size_t)selected_workspace_bytes));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    storage->workspace_bytes = selected_workspace_bytes;
    storage->initialized = 1u;

    memset(plan, 0, sizeof(*plan));
    plan->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION;
    plan->plan_kind = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR;
    plan->input_dimension = input_dimension;
    plan->output_dimension = output_dimension;
    plan->maximum_active_sequence_count =
        create_info->maximum_active_sequence_count;
    plan->output_is_f32 = output_is_f32;
    plan->cublaslt_handle = (void *)binding->cublaslt_handle;
    plan->matmul_descriptor = (void *)storage->matmul_descriptor;
    plan->input_layout = (void *)storage->input_layout;
    plan->weight_layout = (void *)storage->weight_layout;
    plan->output_layout = (void *)storage->output_layout;
    plan->algorithm = (const void *)&storage->algorithm;
    plan->workspace = storage->workspace;
    plan->workspace_bytes = storage->workspace_bytes;
    plan->alpha = 1.0f;
    plan->beta = 0.0f;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52LinearPlanValidateCreateInfo(
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo *create_info)
{
    uint64_t workspace_limit_bytes;

    if (create_info == 0 ||
        create_info->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION ||
        create_info->maximum_active_sequence_count == 0u ||
        create_info->dense_intermediate_dimension == 0u ||
        create_info->expert_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
        create_info->required_plan_mask == 0u ||
        (create_info->required_plan_mask &
         ~SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_REQUIRED_GLM52_PREFIX) != 0u ||
        create_info->autotune_measurement_iterations == 0u ||
        create_info->reserved0 != 0u ||
        create_info->cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    workspace_limit_bytes = create_info->workspace_limit_bytes;
    if (workspace_limit_bytes == 0u)
    {
        workspace_limit_bytes =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_DEFAULT_WORKSPACE_BYTES;
    }
    if (workspace_limit_bytes > (uint64_t)SIZE_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreate(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding **binding_out,
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo *create_info)
{
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding;
    SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo normalized_create_info;
    SparkStatus status;
    cublasStatus_t cublas_status;
    uint64_t workspace_limit_bytes;

    if (binding_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *binding_out = 0;
    status = SparkGlm52LinearPlanValidateCreateInfo(create_info);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    binding = (SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *)
        calloc(1u, sizeof(*binding));
    if (binding == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    binding->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION;
    binding->plan_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT;

    cublas_status = cublasLtCreate(&binding->cublaslt_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        free(binding);
        return SparkGlm52LinearPlanCublasToSparkStatus(cublas_status);
    }

    workspace_limit_bytes = create_info->workspace_limit_bytes != 0u
        ? create_info->workspace_limit_bytes
        : SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_DEFAULT_WORKSPACE_BYTES;
    normalized_create_info = *create_info;
    normalized_create_info.workspace_limit_bytes = workspace_limit_bytes;

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            normalized_create_info.dense_intermediate_dimension,
            0u,
            normalized_create_info.dense_input_bf16,
            normalized_create_info.dense_gate_weight_bf16,
            normalized_create_info.dense_gate_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            normalized_create_info.dense_intermediate_dimension,
            0u,
            normalized_create_info.dense_input_bf16,
            normalized_create_info.dense_up_weight_bf16,
            normalized_create_info.dense_up_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
            normalized_create_info.dense_intermediate_dimension,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            0u,
            normalized_create_info.dense_intermediate_bf16,
            normalized_create_info.dense_down_weight_bf16,
            normalized_create_info.dense_down_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ROUTER_LOGITS) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            normalized_create_info.expert_count,
            1u,
            normalized_create_info.router_input_bf16,
            normalized_create_info.router_weight_bf16,
            normalized_create_info.router_logits_f32,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_QUERY_A) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
            0u,
            normalized_create_info.raw_projection_input_bf16,
            normalized_create_info.raw_query_a_weight_bf16,
            normalized_create_info.raw_query_a_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_QUERY_B) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,
            0u,
            normalized_create_info.raw_query_b_input_bf16,
            normalized_create_info.raw_query_b_weight_bf16,
            normalized_create_info.raw_query_b_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_KV_A) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,
            0u,
            normalized_create_info.raw_projection_input_bf16,
            normalized_create_info.raw_kv_a_weight_bf16,
            normalized_create_info.raw_kv_a_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_KV_B) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,
            0u,
            normalized_create_info.raw_kv_b_input_bf16,
            normalized_create_info.raw_kv_b_weight_bf16,
            normalized_create_info.raw_kv_b_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ATTENTION_OUTPUT) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            0u,
            normalized_create_info.attention_output_input_bf16,
            normalized_create_info.attention_output_weight_bf16,
            normalized_create_info.attention_output_bf16,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RESTRICTED_LOGITS) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESTRICTED_LOGITS,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,
            1u,
            normalized_create_info.restricted_logits_input_bf16,
            normalized_create_info.restricted_lm_head_weight_bf16,
            normalized_create_info.restricted_logits_f32,
            &normalized_create_info);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    *binding_out = binding;
    return SPARK_STATUS_OK;
}

void SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding)
{
    uint32_t plan_index;

    if (binding == 0)
    {
        return;
    }
    for (plan_index = 0u;
         plan_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT;
         ++plan_index)
    {
        SparkGlm52LinearPlanDestroyStorage(&binding->storage[plan_index]);
    }
    if (binding->cublaslt_handle != 0)
    {
        cublasLtDestroy(binding->cublaslt_handle);
    }
    memset(binding, 0, sizeof(*binding));
    free(binding);
}

const SparkGlm52ResidentDecodeStageLinearPlan *
SparkGlm52ResidentDecodeStageLinearPlanResidentBindingPlans(
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    uint32_t *plan_count_out)
{
    if (plan_count_out != 0)
    {
        *plan_count_out = 0u;
    }
    if (binding == 0 ||
        binding->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION)
    {
        return 0;
    }
    if (plan_count_out != 0)
    {
        *plan_count_out = binding->plan_count;
    }
    return binding->plans;
}
