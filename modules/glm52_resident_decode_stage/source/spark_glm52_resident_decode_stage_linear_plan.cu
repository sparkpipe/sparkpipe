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
#define SPARK_GLM52_LINEAR_PLAN_FP8_ACTIVATION_WORKSPACE_ALIGNMENT_BYTES 256ull

typedef struct SparkGlm52ResidentDecodeStageLinearPlanStorage
{
    uint32_t initialized;
    uint32_t prepared_active_sequence_count;
    uint32_t input_dimension;
    uint32_t output_dimension;
    uint32_t output_is_f32;
    uint32_t autotune_warmup_iterations;
    uint32_t autotune_measurement_iterations;
    uint32_t reserved0;
    cublasLtMatmulDesc_t matmul_descriptor;
    cublasLtMatrixLayout_t input_layout;
    cublasLtMatrixLayout_t weight_layout;
    cublasLtMatrixLayout_t output_layout;
    cublasLtMatmulAlgo_t algorithm;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t workspace_limit_bytes;
    cudaStream_t cuda_stream;
    const void *selection_input;
    const void *selection_weight;
    void *selection_output;
    void *quantized_weight_workspace;
    void *quantized_output_workspace;
    uint64_t quantized_output_workspace_bytes;
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
    SparkGlm52ResidentDecodeStageQuantizedLinearView quantized_views[
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

static void SparkGlm52LinearPlanDestroyCublasStorage(
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage)
{
    if (storage == 0)
    {
        return;
    }
    if (storage->workspace != 0)
    {
        cudaFree(storage->workspace);
        storage->workspace = 0;
    }
    if (storage->output_layout != 0)
    {
        cublasLtMatrixLayoutDestroy(storage->output_layout);
        storage->output_layout = 0;
    }
    if (storage->weight_layout != 0)
    {
        cublasLtMatrixLayoutDestroy(storage->weight_layout);
        storage->weight_layout = 0;
    }
    if (storage->input_layout != 0)
    {
        cublasLtMatrixLayoutDestroy(storage->input_layout);
        storage->input_layout = 0;
    }
    if (storage->matmul_descriptor != 0)
    {
        cublasLtMatmulDescDestroy(storage->matmul_descriptor);
        storage->matmul_descriptor = 0;
    }
    memset(&storage->algorithm, 0, sizeof(storage->algorithm));
    storage->workspace_bytes = 0u;
    storage->prepared_active_sequence_count = 0u;
}

static void SparkGlm52LinearPlanDestroyStorage(
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage)
{
    if (storage == 0)
    {
        return;
    }
    SparkGlm52LinearPlanDestroyCublasStorage(storage);
    if (storage->quantized_output_workspace != 0)
    {
        cudaFree(storage->quantized_output_workspace);
    }
    if (storage->quantized_weight_workspace != 0)
    {
        cudaFree(storage->quantized_weight_workspace);
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

static bool SparkGlm52LinearPlanAlgorithmIsDeterministic(
    const cublasLtMatmulAlgo_t *algorithm)
{
    cublasStatus_t cublas_status;
    uint32_t reduction_scheme;
    uint32_t splitk_count;
    size_t written_size;

    if (algorithm == 0)
    {
        return false;
    }
    reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
    splitk_count = 0u;
    written_size = 0u;
    cublas_status = cublasLtMatmulAlgoConfigGetAttribute(
        algorithm,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &reduction_scheme,
        sizeof(reduction_scheme),
        &written_size);
    if (cublas_status != CUBLAS_STATUS_SUCCESS ||
        reduction_scheme != CUBLASLT_REDUCTION_SCHEME_NONE)
    {
        return false;
    }
    written_size = 0u;
    cublas_status = cublasLtMatmulAlgoConfigGetAttribute(
        algorithm,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        &splitk_count,
        sizeof(splitk_count),
        &written_size);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return false;
    }
    return splitk_count <= 1u;
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
    uint32_t reduction_scheme_mask;
    int returned_results;
    int result_index;

    if (binding == 0 || storage == 0 || selected_workspace_bytes_out == 0 ||
        input == 0 || weight == 0 || output == 0 ||
        measurement_iterations == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *selected_workspace_bytes_out = 0u;
    preference = 0;
    returned_results = 0;

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
    reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_NONE;
    cublas_status = cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
        &reduction_scheme_mask,
        sizeof(reduction_scheme_mask));
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

    for (result_index = 0; result_index < returned_results; ++result_index)
    {
        if (heuristic_results[result_index].state != CUBLAS_STATUS_SUCCESS ||
            (uint64_t)heuristic_results[result_index].workspaceSize >
                workspace_limit_bytes ||
            !SparkGlm52LinearPlanAlgorithmIsDeterministic(
                &heuristic_results[result_index].algo))
        {
            continue;
        }

        storage->algorithm = heuristic_results[result_index].algo;
        *selected_workspace_bytes_out =
            (uint64_t)heuristic_results[result_index].workspaceSize;
        return SPARK_STATUS_OK;
    }

    return SPARK_STATUS_INVALID_ARGUMENT;
}

static void SparkGlm52LinearPlanPublishPreparedStorage(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage,
    SparkGlm52ResidentDecodeStageLinearPlan *plan)
{
    plan->cublaslt_handle = (void *)binding->cublaslt_handle;
    plan->matmul_descriptor = (void *)storage->matmul_descriptor;
    plan->input_layout = (void *)storage->input_layout;
    plan->weight_layout = (void *)storage->weight_layout;
    plan->output_layout = (void *)storage->output_layout;
    plan->algorithm = (const void *)&storage->algorithm;
    plan->workspace = storage->workspace;
    plan->workspace_bytes = storage->workspace_bytes;
    plan->custom_state = storage;
}

static SparkStatus SparkGlm52LinearPlanPrepareOne(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage,
    SparkGlm52ResidentDecodeStageLinearPlan *plan,
    uint32_t active_sequence_count)
{
    SparkGlm52ResidentDecodeStageLinearPlanStorage prepared;
    uint64_t selected_workspace_bytes;
    SparkStatus status;

    if (binding == 0 || storage == 0 || plan == 0 ||
        active_sequence_count == 0u ||
        active_sequence_count > plan->maximum_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (storage->prepared_active_sequence_count == active_sequence_count)
    {
        SparkGlm52LinearPlanPublishPreparedStorage(binding, storage, plan);
        return SPARK_STATUS_OK;
    }
    memset(&prepared, 0, sizeof(prepared));
    status = SparkGlm52LinearPlanCreateDescriptors(
        binding,
        &prepared,
        active_sequence_count,
        storage->input_dimension,
        storage->output_dimension,
        storage->output_is_f32);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52LinearPlanSelectAlgorithm(
            binding,
            &prepared,
            storage->selection_input,
            storage->selection_weight,
            storage->selection_output,
            storage->workspace_limit_bytes,
            storage->cuda_stream,
            storage->autotune_warmup_iterations,
            storage->autotune_measurement_iterations,
            &selected_workspace_bytes);
    }
    if (status == SPARK_STATUS_OK && selected_workspace_bytes != 0u)
    {
        status = SparkGlm52LinearPlanCudaToSparkStatus(
            cudaMalloc(&prepared.workspace, (size_t)selected_workspace_bytes));
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52LinearPlanDestroyCublasStorage(&prepared);
        return status;
    }
    prepared.workspace_bytes = selected_workspace_bytes;
    SparkGlm52LinearPlanDestroyCublasStorage(storage);
    storage->matmul_descriptor = prepared.matmul_descriptor;
    storage->input_layout = prepared.input_layout;
    storage->weight_layout = prepared.weight_layout;
    storage->output_layout = prepared.output_layout;
    storage->algorithm = prepared.algorithm;
    storage->workspace = prepared.workspace;
    storage->workspace_bytes = prepared.workspace_bytes;
    storage->prepared_active_sequence_count = active_sequence_count;
    storage->initialized = 1u;
    SparkGlm52LinearPlanPublishPreparedStorage(binding, storage, plan);
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

    if (binding == 0 || create_info == 0 ||
        plan_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT ||
        input == 0 || weight == 0 || output == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    storage = &binding->storage[plan_index];
    plan = &binding->plans[plan_index];
    storage->input_dimension = input_dimension;
    storage->output_dimension = output_dimension;
    storage->output_is_f32 = output_is_f32;
    storage->autotune_warmup_iterations =
        create_info->autotune_warmup_iterations;
    storage->autotune_measurement_iterations =
        create_info->autotune_measurement_iterations;
    storage->workspace_limit_bytes = create_info->workspace_limit_bytes;
    storage->cuda_stream = (cudaStream_t)create_info->cuda_stream;
    storage->selection_input = input;
    storage->selection_weight = weight;
    storage->selection_output = output;
    memset(plan, 0, sizeof(*plan));
    plan->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION;
    plan->plan_kind = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR;
    plan->input_dimension = input_dimension;
    plan->output_dimension = output_dimension;
    plan->maximum_active_sequence_count =
        create_info->maximum_active_sequence_count;
    plan->output_is_f32 = output_is_f32;
    plan->alpha = 1.0f;
    plan->beta = 0.0f;
    return SparkGlm52LinearPlanPrepareOne(binding, storage, plan, 1u);
}

static uint64_t SparkGlm52LinearPlanDivideRoundUpU64(
    uint64_t value,
    uint64_t divisor)
{
    if (divisor == 0u)
    {
        return 0u;
    }
    return (value + divisor - 1u) / divisor;
}

static uint64_t SparkGlm52LinearPlanAlignUpU64(
    uint64_t value,
    uint64_t alignment)
{
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u)
    {
        return 0u;
    }
    if (value > UINT64_MAX - (alignment - 1u))
    {
        return 0u;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint64_t SparkGlm52LinearPlanFp8ActivationLinearWorkspaceBytes(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size)
{
    uint64_t payload_bytes;
    uint64_t scale_block_count;
    uint64_t scale_element_count;
    uint64_t scale_bytes;
    uint64_t aligned_payload_bytes;
    uint64_t aligned_scale_bytes;

    if (maximum_active_sequence_count == 0u || input_dimension == 0u ||
        scale_block_size == 0u)
    {
        return 0u;
    }
    if ((uint64_t)maximum_active_sequence_count >
        UINT64_MAX / (uint64_t)input_dimension)
    {
        return 0u;
    }

    payload_bytes =
        (uint64_t)maximum_active_sequence_count * (uint64_t)input_dimension;
    scale_block_count = SparkGlm52LinearPlanDivideRoundUpU64(
        input_dimension,
        scale_block_size);
    if (scale_block_count == 0u ||
        (uint64_t)maximum_active_sequence_count >
            UINT64_MAX / scale_block_count)
    {
        return 0u;
    }

    scale_element_count =
        (uint64_t)maximum_active_sequence_count * scale_block_count;
    if (scale_element_count > UINT64_MAX / (uint64_t)sizeof(float))
    {
        return 0u;
    }
    scale_bytes = scale_element_count * (uint64_t)sizeof(float);

    aligned_payload_bytes = SparkGlm52LinearPlanAlignUpU64(
        payload_bytes,
        SPARK_GLM52_LINEAR_PLAN_FP8_ACTIVATION_WORKSPACE_ALIGNMENT_BYTES);
    aligned_scale_bytes = SparkGlm52LinearPlanAlignUpU64(
        scale_bytes,
        SPARK_GLM52_LINEAR_PLAN_FP8_ACTIVATION_WORKSPACE_ALIGNMENT_BYTES);
    if (aligned_payload_bytes == 0u || aligned_scale_bytes == 0u ||
        aligned_payload_bytes > UINT64_MAX - aligned_scale_bytes ||
        aligned_payload_bytes + aligned_scale_bytes >
            UINT64_MAX - aligned_scale_bytes)
    {
        return 0u;
    }
    return aligned_payload_bytes + aligned_scale_bytes + aligned_scale_bytes;
}

static uint64_t SparkGlm52LinearPlanFp8PayloadBytes(
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    return (uint64_t)input_dimension * (uint64_t)output_dimension;
}

static uint64_t SparkGlm52LinearPlanFp8ScaleBytes(
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    uint64_t input_scale_block_count;
    uint64_t output_scale_block_count;

    input_scale_block_count = SparkGlm52LinearPlanDivideRoundUpU64(
        input_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    output_scale_block_count = SparkGlm52LinearPlanDivideRoundUpU64(
        output_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    return input_scale_block_count * output_scale_block_count *
        (uint64_t)sizeof(float);
}

static SparkStatus SparkGlm52LinearPlanInitializeFp8TailStorage(
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage,
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_is_f32,
    const void *weight_payload,
    cudaStream_t cuda_stream,
    uint32_t *storage_output_dimension_out,
    const void **storage_weight_payload_out)
{
    uint32_t alignment;

    if (storage == 0 || weight_payload == 0 || cuda_stream == 0 ||
        storage_output_dimension_out == 0 || storage_weight_payload_out == 0 ||
        maximum_active_sequence_count == 0u || input_dimension == 0u ||
        output_dimension == 0u || output_is_f32 > 1u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    alignment =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALED_GEMM_OUTPUT_ALIGNMENT;
    if ((output_dimension % alignment) != 0u)
    {
        /*
         * The production GLM geometry is alignment-clean. Padding an
         * unsupported shape here creates a second output tensor followed by
         * a full device-to-device row trim on every decode. Fail closed
         * instead: a future unaligned shape needs a backend with a genuine
         * strided/direct epilogue and its own retained qualification receipt.
         */
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    *storage_output_dimension_out = output_dimension;
    *storage_weight_payload_out = weight_payload;
    storage->quantized_output_workspace = 0;
    storage->quantized_output_workspace_bytes = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52LinearPlanCreateQuantizedFp8One(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_is_f32,
    const void *weight_payload,
    const void *weight_scale,
    const SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo *create_info)
{
    SparkGlm52ResidentDecodeStageLinearPlanStorage *storage;
    SparkGlm52ResidentDecodeStageLinearPlan *plan;
    SparkGlm52ResidentDecodeStageQuantizedLinearView *view;
    const void *storage_weight_payload;
    uint64_t required_workspace_bytes;
    uint32_t storage_output_dimension;
    SparkStatus status;

    if (binding == 0 || create_info == 0 ||
        plan_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT ||
        input_dimension == 0u ||
        output_dimension == 0u ||
        weight_payload == 0 ||
        weight_scale == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    required_workspace_bytes =
        SparkGlm52LinearPlanFp8ActivationLinearWorkspaceBytes(
            create_info->maximum_active_sequence_count,
            input_dimension,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    if (required_workspace_bytes == 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (required_workspace_bytes > create_info->workspace_limit_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    storage = &binding->storage[plan_index];
    plan = &binding->plans[plan_index];
    view = &binding->quantized_views[plan_index];
    memset(plan, 0, sizeof(*plan));
    memset(view, 0, sizeof(*view));
    SparkGlm52LinearPlanDestroyStorage(storage);

    status = SparkGlm52LinearPlanInitializeFp8TailStorage(
        storage,
        create_info->maximum_active_sequence_count,
        input_dimension,
        output_dimension,
        output_is_f32,
        weight_payload,
        (cudaStream_t)create_info->cuda_stream,
        &storage_output_dimension,
        &storage_weight_payload);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52LinearPlanCudaToSparkStatus(
        cudaMalloc(&storage->workspace, (size_t)required_workspace_bytes));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    storage->workspace_bytes = required_workspace_bytes;
    storage->initialized = 1u;

    view->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUANTIZED_LINEAR_VIEW_ABI_VERSION;
    view->weight_format =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
    view->input_dimension = input_dimension;
    view->output_dimension = output_dimension;
    view->storage_output_dimension = storage_output_dimension;
    view->scale_block_size = SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK;
    view->output_is_f32 = output_is_f32;
    view->weight_payload = storage_weight_payload;
    view->weight_scale = weight_scale;
    view->weight_payload_bytes = SparkGlm52LinearPlanFp8PayloadBytes(
        input_dimension,
        storage_output_dimension);
    view->weight_scale_bytes = SparkGlm52LinearPlanFp8ScaleBytes(
        input_dimension,
        storage_output_dimension);
    view->output_workspace = storage->quantized_output_workspace;
    view->output_workspace_bytes = storage->quantized_output_workspace_bytes;

    plan->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION;
    plan->plan_kind =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR;
    plan->input_dimension = input_dimension;
    plan->output_dimension = output_dimension;
    plan->maximum_active_sequence_count =
        create_info->maximum_active_sequence_count;
    plan->output_is_f32 = output_is_f32;
    plan->custom_state = view;
    plan->workspace = storage->workspace;
    plan->workspace_bytes = storage->workspace_bytes;
    plan->alpha = 1.0f;
    plan->beta = 0.0f;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52LinearPlanValidateQuantizedFp8Pair(
    const void *weight_payload,
    const void *weight_scale)
{
    if ((weight_payload == 0) != (weight_scale == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52LinearPlanUseQuantizedFp8(
    const void *weight_payload,
    const void *weight_scale)
{
    return weight_payload != 0 && weight_scale != 0;
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

    if (SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->dense_gate_weight_fp8_e4m3,
            create_info->dense_gate_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->dense_up_weight_fp8_e4m3,
            create_info->dense_up_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->dense_down_weight_fp8_e4m3,
            create_info->dense_down_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->raw_query_a_weight_fp8_e4m3,
            create_info->raw_query_a_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->raw_query_b_weight_fp8_e4m3,
            create_info->raw_query_b_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->raw_kv_a_weight_fp8_e4m3,
            create_info->raw_kv_a_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->raw_kv_b_weight_fp8_e4m3,
            create_info->raw_kv_b_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->attention_output_weight_fp8_e4m3,
            create_info->attention_output_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->dsa_query_weight_fp8_e4m3,
            create_info->dsa_query_weight_scale_inv_f32) != SPARK_STATUS_OK ||
        SparkGlm52LinearPlanValidateQuantizedFp8Pair(
            create_info->dsa_key_weight_fp8_e4m3,
            create_info->dsa_key_weight_scale_inv_f32) != SPARK_STATUS_OK)
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
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.dense_gate_weight_fp8_e4m3,
                normalized_create_info.dense_gate_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                normalized_create_info.dense_intermediate_dimension,
                0u,
                normalized_create_info.dense_gate_weight_fp8_e4m3,
                normalized_create_info.dense_gate_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.dense_up_weight_fp8_e4m3,
                normalized_create_info.dense_up_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                normalized_create_info.dense_intermediate_dimension,
                0u,
                normalized_create_info.dense_up_weight_fp8_e4m3,
                normalized_create_info.dense_up_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.dense_down_weight_fp8_e4m3,
                normalized_create_info.dense_down_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
                normalized_create_info.dense_intermediate_dimension,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                0u,
                normalized_create_info.dense_down_weight_fp8_e4m3,
                normalized_create_info.dense_down_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
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
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.raw_query_a_weight_fp8_e4m3,
                normalized_create_info.raw_query_a_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                0u,
                normalized_create_info.raw_query_a_weight_fp8_e4m3,
                normalized_create_info.raw_query_a_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_QUERY_B) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.raw_query_b_weight_fp8_e4m3,
                normalized_create_info.raw_query_b_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,
                0u,
                normalized_create_info.raw_query_b_weight_fp8_e4m3,
                normalized_create_info.raw_query_b_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_KV_A) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.raw_kv_a_weight_fp8_e4m3,
                normalized_create_info.raw_kv_a_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,
                0u,
                normalized_create_info.raw_kv_a_weight_fp8_e4m3,
                normalized_create_info.raw_kv_a_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_KV_B) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.raw_kv_b_weight_fp8_e4m3,
                normalized_create_info.raw_kv_b_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,
                0u,
                normalized_create_info.raw_kv_b_weight_fp8_e4m3,
                normalized_create_info.raw_kv_b_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ATTENTION_OUTPUT) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.attention_output_weight_fp8_e4m3,
                normalized_create_info.attention_output_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                0u,
                normalized_create_info.attention_output_weight_fp8_e4m3,
                normalized_create_info.attention_output_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
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
        }
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

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DSA_QUERY) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.dsa_query_weight_fp8_e4m3,
                normalized_create_info.dsa_query_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_QUERY,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION,
                0u,
                normalized_create_info.dsa_query_weight_fp8_e4m3,
                normalized_create_info.dsa_query_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
        {
            status = SparkGlm52LinearPlanCreateOne(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_QUERY,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION,
                0u,
                normalized_create_info.dsa_query_input_bf16,
                normalized_create_info.dsa_query_weight_bf16,
                normalized_create_info.dsa_query_output_bf16,
                &normalized_create_info);
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DSA_KEY) != 0u)
    {
        if (SparkGlm52LinearPlanUseQuantizedFp8(
                normalized_create_info.dsa_key_weight_fp8_e4m3,
                normalized_create_info.dsa_key_weight_scale_inv_f32) != 0u)
        {
            status = SparkGlm52LinearPlanCreateQuantizedFp8One(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_KEY,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,
                0u,
                normalized_create_info.dsa_key_weight_fp8_e4m3,
                normalized_create_info.dsa_key_weight_scale_inv_f32,
                &normalized_create_info);
        }
        else
        {
            status = SparkGlm52LinearPlanCreateOne(
                binding,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_KEY,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,
                0u,
                normalized_create_info.dsa_key_input_bf16,
                normalized_create_info.dsa_key_weight_bf16,
                normalized_create_info.dsa_key_output_bf16,
                &normalized_create_info);
        }
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(binding);
            return status;
        }
    }

    if ((normalized_create_info.required_plan_mask &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DSA_WEIGHTS) != 0u)
    {
        status = SparkGlm52LinearPlanCreateOne(
            binding,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DSA_WEIGHTS,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION,
            0u,
            normalized_create_info.dsa_weights_input_bf16,
            normalized_create_info.dsa_weights_proj_weight_bf16,
            normalized_create_info.dsa_weights_output_bf16,
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

SparkStatus SparkGlm52ResidentDecodeStageLinearPlanResidentBindingPrepareActiveRows(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
    uint32_t active_sequence_count)
{
    uint32_t plan_index;
    uint32_t prepared_active_sequence_count;
    SparkStatus status;

    if (binding == 0 ||
        binding->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION ||
        active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (plan_index = 0u; plan_index < binding->plan_count; ++plan_index)
    {
        if (binding->plans[plan_index].plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR)
        {
            continue;
        }
        prepared_active_sequence_count =
            SparkGlm52ResidentDecodeStageLinearPlanRequiredPreparedActiveRows(
                &binding->plans[plan_index], active_sequence_count);
        if (prepared_active_sequence_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (binding->storage[plan_index].prepared_active_sequence_count >=
            prepared_active_sequence_count)
        {
            continue;
        }
        status = SparkGlm52LinearPlanPrepareOne(
            binding,
            &binding->storage[plan_index],
            &binding->plans[plan_index],
            prepared_active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

uint32_t SparkGlm52ResidentDecodeStageLinearPlanPreparedActiveRows(
    const SparkGlm52ResidentDecodeStageLinearPlan *plan)
{
    const SparkGlm52ResidentDecodeStageLinearPlanStorage *storage;

    if (plan == 0 ||
        plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR ||
        plan->custom_state == 0)
    {
        return 0u;
    }
    storage = (const SparkGlm52ResidentDecodeStageLinearPlanStorage *)
        plan->custom_state;
    return storage->prepared_active_sequence_count;
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

SparkGlm52ResidentDecodeStageLinearPlan *
SparkGlm52ResidentDecodeStageLinearPlanResidentBindingMutablePlans(
    SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *binding,
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
