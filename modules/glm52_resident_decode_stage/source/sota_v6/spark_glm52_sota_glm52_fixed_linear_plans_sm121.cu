#include "spark_glm52_sota_glm52_fixed_linear_plans_sm121.cuh"

extern "C" cudaError_t SparkGlm52SotaRunLinearPlanSm121(
    const SparkGlm52SotaLinearPlan *plan,
    cudaStream_t stream);

extern "C" cudaError_t SparkGlm52SotaRunFixedLinearPlanSm121(
    const SparkGlm52SotaFixedLinearPlan *fixed_plan,
    cudaStream_t stream)
{
    uint32_t expected_input;
    uint32_t expected_output;

    if (fixed_plan == 0)
    {
        return cudaErrorInvalidValue;
    }
    expected_input = SparkGlm52SotaFixedLinearExpectedInput(fixed_plan->kind);
    expected_output = SparkGlm52SotaFixedLinearExpectedOutput(fixed_plan->kind);
    if (expected_input == 0u || expected_output == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (fixed_plan->plan.input_columns != expected_input ||
        fixed_plan->plan.output_columns != expected_output ||
        fixed_plan->plan.launch == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (fixed_plan->expected_alignment_bytes != 0u && fixed_plan->plan.required_alignment_bytes < fixed_plan->expected_alignment_bytes)
    {
        return cudaErrorInvalidValue;
    }
    return SparkGlm52SotaRunLinearPlanSm121(&fixed_plan->plan, stream);
}
