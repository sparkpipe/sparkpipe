#ifndef SPARK_GLM52_SOTA_CUBLASLT_LINEAR_PLANS_SM121_CUH
#define SPARK_GLM52_SOTA_CUBLASLT_LINEAR_PLANS_SM121_CUH

#include "spark_glm52_sota_decode_common.cuh"
#include <cublasLt.h>

struct SparkGlm52SotaCublasLtLinearPlan
{
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation_desc;
    cublasLtMatrixLayout_t input_layout;
    cublasLtMatrixLayout_t weight_layout;
    cublasLtMatrixLayout_t output_layout;
    cublasLtMatrixLayout_t bias_layout;
    cublasLtMatmulAlgo_t algorithm;
    const void *input;
    const void *weight;
    const void *bias;
    void *output;
    void *workspace;
    size_t workspace_bytes;
    float alpha;
    float beta;
};

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t SparkGlm52SotaCublasLtLinearLaunchSm121(
    void *plan_context,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
