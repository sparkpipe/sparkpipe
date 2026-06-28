#include "sparkpipe/spark_glm52_resident_decode_stage_cublaslt_plans.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdlib.h>
#include <string.h>

typedef struct SparkGlm52ResidentDecodeStageCublasLtLinearPlanStorage
{
    cublasLtMatmulDesc_t matmul_descriptor;
    cublasLtMatrixLayout_t input_layout;
    cublasLtMatrixLayout_t weight_layout;
    cublasLtMatrixLayout_t output_layout;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulAlgo_t algorithm;
} SparkGlm52ResidentDecodeStageCublasLtLinearPlanStorage;

static SparkStatus SparkGlm52ResidentDecodeStageTranslateCublasStatus(
    cublasStatus_t cublas_status)
{
    return cublas_status == CUBLAS_STATUS_SUCCESS
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

static cudaDataType SparkGlm52ResidentDecodeStageResolveCublasWeightType(
    uint32_t plan_kind)
{
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR)
    {
        return CUDA_R_16BF;
    }
#if defined(CUDA_R_8F_E4M3)
    if (plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR)
    {
        return CUDA_R_8F_E4M3;
    }
#endif
    return CUDA_R_16BF;
}

static SparkStatus SparkGlm52ResidentDecodeStageSetRowMajorLayout(
    cublasLtMatrixLayout_t layout)
{
    cublasLtOrder_t row_order;
    cublasStatus_t cublas_status;

    row_order = CUBLASLT_ORDER_ROW;
    cublas_status = cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order,
        sizeof(row_order));
    return SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
}

static void SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(
    SparkGlm52ResidentDecodeStageCublasLtLinearPlanStorage *storage)
{
    if (storage == 0)
    {
        return;
    }
    if (storage->preference != 0)
    {
        cublasLtMatmulPreferenceDestroy(storage->preference);
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
    free(storage);
}

extern "C" SparkStatus SparkGlm52ResidentDecodeStageBuildCublasLtLinearPlan(
    const SparkGlm52ResidentDecodeStageCublasLtLinearPlanBuildRequest *request,
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan)
{
    SparkGlm52ResidentDecodeStageCublasLtLinearPlanStorage *storage;
    cublasOperation_t transa;
    cublasOperation_t transb;
    cudaDataType weight_type;
    cudaDataType output_type;
    uint64_t workspace_bytes;
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned_result_count;
    cublasStatus_t cublas_status;
    SparkStatus status;

    if (request == 0 || linear_plan == 0 || request->cublaslt_handle == 0 ||
        request->input_dimension == 0u || request->output_dimension == 0u ||
        request->maximum_active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR &&
        request->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
#if !defined(CUDA_R_8F_E4M3)
    if (request->plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
#endif

    storage = (SparkGlm52ResidentDecodeStageCublasLtLinearPlanStorage *)calloc(
        1u,
        sizeof(*storage));
    if (storage == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    transa = CUBLAS_OP_N;
    transb = CUBLAS_OP_T;
    weight_type = SparkGlm52ResidentDecodeStageResolveCublasWeightType(
        request->plan_kind);
    output_type = request->output_is_f32 != 0u ? CUDA_R_32F : CUDA_R_16BF;
    workspace_bytes = request->workspace_bytes;

    cublas_status = cublasLtMatmulDescCreate(
        &storage->matmul_descriptor,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F);
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }
    cublas_status = cublasLtMatmulDescSetAttribute(
        storage->matmul_descriptor,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transa,
        sizeof(transa));
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }
    cublas_status = cublasLtMatmulDescSetAttribute(
        storage->matmul_descriptor,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transb,
        sizeof(transb));
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }

    cublas_status = cublasLtMatrixLayoutCreate(
        &storage->input_layout,
        CUDA_R_16BF,
        request->maximum_active_sequence_count,
        request->input_dimension,
        request->input_dimension);
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }
    status = SparkGlm52ResidentDecodeStageSetRowMajorLayout(storage->input_layout);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }

    cublas_status = cublasLtMatrixLayoutCreate(
        &storage->weight_layout,
        weight_type,
        request->output_dimension,
        request->input_dimension,
        request->input_dimension);
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }
    status = SparkGlm52ResidentDecodeStageSetRowMajorLayout(storage->weight_layout);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }

    cublas_status = cublasLtMatrixLayoutCreate(
        &storage->output_layout,
        output_type,
        request->maximum_active_sequence_count,
        request->output_dimension,
        request->output_dimension);
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }
    status = SparkGlm52ResidentDecodeStageSetRowMajorLayout(storage->output_layout);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }

    cublas_status = cublasLtMatmulPreferenceCreate(&storage->preference);
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }
    cublas_status = cublasLtMatmulPreferenceSetAttribute(
        storage->preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes));
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status;
    }

    memset(&heuristic, 0, sizeof(heuristic));
    returned_result_count = 0;
    cublas_status = cublasLtMatmulAlgoGetHeuristic(
        (cublasLtHandle_t)request->cublaslt_handle,
        storage->matmul_descriptor,
        storage->input_layout,
        storage->weight_layout,
        storage->output_layout,
        storage->output_layout,
        storage->preference,
        1,
        &heuristic,
        &returned_result_count);
    status = SparkGlm52ResidentDecodeStageTranslateCublasStatus(cublas_status);
    if (status != SPARK_STATUS_OK || returned_result_count == 0)
    {
        SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
        return status != SPARK_STATUS_OK ? status : SPARK_STATUS_NOT_FOUND;
    }

    storage->algorithm = heuristic.algo;
    memset(linear_plan, 0, sizeof(*linear_plan));
    linear_plan->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION;
    linear_plan->plan_kind = request->plan_kind;
    linear_plan->input_dimension = request->input_dimension;
    linear_plan->output_dimension = request->output_dimension;
    linear_plan->maximum_active_sequence_count = request->maximum_active_sequence_count;
    linear_plan->output_is_f32 = request->output_is_f32;
    linear_plan->cublaslt_handle = request->cublaslt_handle;
    linear_plan->matmul_descriptor = storage->matmul_descriptor;
    linear_plan->input_layout = storage->input_layout;
    linear_plan->weight_layout = storage->weight_layout;
    linear_plan->output_layout = storage->output_layout;
    linear_plan->algorithm = &storage->algorithm;
    linear_plan->workspace = request->workspace;
    linear_plan->workspace_bytes = request->workspace_bytes;
    linear_plan->alpha = 1.0f;
    linear_plan->beta = 0.0f;
    linear_plan->custom_state = storage;
    return SPARK_STATUS_OK;
}

extern "C" void SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlan(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan)
{
    SparkGlm52ResidentDecodeStageCublasLtLinearPlanStorage *storage;

    if (linear_plan == 0)
    {
        return;
    }
    storage = (SparkGlm52ResidentDecodeStageCublasLtLinearPlanStorage *)
        linear_plan->custom_state;
    SparkGlm52ResidentDecodeStageDestroyCublasLtLinearPlanStorage(storage);
    memset(linear_plan, 0, sizeof(*linear_plan));
}
