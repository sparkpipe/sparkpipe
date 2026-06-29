#include "spark_glm52_sota_cublaslt_linear_plans_sm121.cuh"

extern "C" cudaError_t SparkGlm52SotaCublasLtLinearLaunchSm121(
    void *plan_context,
    cudaStream_t stream)
{
    SparkGlm52SotaCublasLtLinearPlan *plan;
    cublasStatus_t status;

    if (plan_context == 0)
    {
        return cudaErrorInvalidValue;
    }
    plan = reinterpret_cast<SparkGlm52SotaCublasLtLinearPlan *>(plan_context);
    if (plan->handle == 0 || plan->operation_desc == 0 || plan->input_layout == 0 ||
        plan->weight_layout == 0 || plan->output_layout == 0 || plan->input == 0 ||
        plan->weight == 0 || plan->output == 0)
    {
        return cudaErrorInvalidValue;
    }
    status = cublasLtMatmul(
        plan->handle,
        plan->operation_desc,
        &plan->alpha,
        plan->input,
        plan->input_layout,
        plan->weight,
        plan->weight_layout,
        &plan->beta,
        plan->output,
        plan->output_layout,
        plan->output,
        plan->output_layout,
        &plan->algorithm,
        plan->workspace,
        plan->workspace_bytes,
        stream);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
