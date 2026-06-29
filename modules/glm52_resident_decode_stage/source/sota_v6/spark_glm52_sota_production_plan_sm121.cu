#include "spark_glm52_sota_production_plan_sm121.cuh"

static cudaError_t SparkGlm52SotaCheckPointerAlignment(const void *pointer, uint32_t alignment_bytes)
{
    uintptr_t value = (uintptr_t)pointer;
    if (pointer == 0)
    {
        return cudaErrorInvalidValue;
    }
    if ((value & ((uintptr_t)alignment_bytes - 1u)) != 0u)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

static cudaError_t SparkGlm52SotaValidateBf16LinearPlan(const SparkGlm52SotaBf16LinearPlanSm121 *plan, uint32_t expected_k, uint32_t expected_n)
{
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->m == 0u || plan->m > 128u || plan->k != expected_k || plan->n != expected_n)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->handle == 0 || plan->operation_desc == 0 || plan->a_layout == 0 || plan->b_layout == 0 || plan->c_layout == 0 || plan->d_layout == 0)
    {
        return cudaErrorInvalidValue;
    }
    if ((plan->capability_flags & SPARK_GLM52_SOTA_FAST_CAP_CUBLASLT_LINEAR) == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (SparkGlm52SotaCheckPointerAlignment(plan->a, 16u) != cudaSuccess || SparkGlm52SotaCheckPointerAlignment(plan->b, 16u) != cudaSuccess || SparkGlm52SotaCheckPointerAlignment(plan->d, 16u) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

static cudaError_t SparkGlm52SotaValidateNvfp4GemmPlan(const SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan, uint32_t expected_k, uint32_t expected_n, uint32_t output_is_nvfp4)
{
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->maximum_problem_count == 0u || plan->maximum_problem_count > 256u || plan->problem_count > plan->maximum_problem_count || plan->expected_k != expected_k || plan->expected_n != expected_n || plan->expected_group_size != SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE || plan->output_is_nvfp4 != output_is_nvfp4)
    {
        return cudaErrorInvalidValue;
    }
    if ((plan->capability_flags & SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT) == 0u || plan->launch == 0 || plan->workspace == 0 || plan->workspace_bytes == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->cutlass_or_cublas_state != 0)
    {
        if (plan->cutlass_m != expected_n || plan->cutlass_k != expected_k || plan->cutlass_group_count != plan->problem_count || plan->cutlass_n_capacity == 0u)
        {
            return cudaErrorInvalidValue;
        }
        if (plan->tokens_per_expert_device == 0 || plan->cutlass_a_payload_u8 == 0 || plan->cutlass_a_scale_ue4m3_u8 == 0 || plan->cutlass_b_payload_u8 == 0 || plan->cutlass_b_scale_ue4m3_u8 == 0 || plan->cutlass_c_bf16 == 0 || plan->cutlass_d_bf16 == 0)
        {
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }
    if (plan->problems_device == 0)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

static cudaError_t SparkGlm52SotaValidateDenseAlphaMoePlan(const SparkGlm52SotaProductionMoePlanSm121 *plan)
{
    uint64_t required_flags;

    required_flags = SPARK_GLM52_SOTA_FAST_CAP_TOKEN_QUANT_ONCE |
        SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROUTER_TOPK |
        SPARK_GLM52_SOTA_FAST_CAP_DENSE_ALPHA_MOE |
        SPARK_GLM52_SOTA_FAST_CAP_DENSE_ALPHA_STRICT |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP |
        SPARK_GLM52_SOTA_FAST_CAP_FUSED_SILU_REQUANT |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN |
        SPARK_GLM52_SOTA_FAST_CAP_WEIGHTED_COMBINE |
        SPARK_GLM52_SOTA_FAST_CAP_NO_HOST_STAGING |
        SPARK_GLM52_SOTA_FAST_CAP_NO_DEVICE_MEMCPY |
        SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_B_BROADCAST |
        SPARK_GLM52_SOTA_FAST_CAP_FIXED_GLM52_SHAPES;
    if ((plan->capability_flags & required_flags) != required_flags)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->maximum_bound_experts != SPARK_GLM52_SOTA_EXPERT_COUNT ||
        plan->dense_alpha_token_capacity == 0u ||
        plan->dense_alpha_token_capacity > plan->maximum_tokens ||
        plan->dense_alpha_minimum_tokens == 0u ||
        plan->dense_alpha_maximum_tokens < plan->dense_alpha_minimum_tokens ||
        plan->dense_alpha_maximum_tokens > plan->dense_alpha_token_capacity ||
        plan->dense_alpha_require_exact_token_count == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->dense_hidden_nvfp4_by_token_sm1xx.payload_u8 == 0 ||
        plan->dense_hidden_nvfp4_by_token_sm1xx.scale_e4m3_u8 == 0 ||
        plan->dense_gate_up_bf16_by_expert_token == 0 ||
        plan->dense_intermediate_nvfp4_by_expert_token.payload_u8 == 0 ||
        plan->dense_intermediate_nvfp4_by_expert_token.scale_e4m3_u8 == 0 ||
        plan->dense_down_bf16_by_expert_token == 0 ||
        plan->overflow_flag == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (SparkGlm52SotaValidateNvfp4GemmPlan(&plan->dense_gate_up_gemm, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u, 0u) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    if (SparkGlm52SotaValidateNvfp4GemmPlan(&plan->dense_down_gemm, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, 0u) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->dense_gate_up_gemm.cutlass_b_broadcast == 0u ||
        plan->dense_gate_up_gemm.cutlass_sfb_broadcast == 0u ||
        (plan->dense_gate_up_gemm.capability_flags & SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_B_BROADCAST) == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->dense_down_gemm.cutlass_b_broadcast != 0u ||
        plan->dense_down_gemm.cutlass_sfb_broadcast != 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->dense_gate_up_gemm.cutlass_n_capacity != plan->dense_alpha_token_capacity ||
        plan->dense_down_gemm.cutlass_n_capacity != plan->dense_alpha_token_capacity ||
        plan->dense_gate_up_gemm.cutlass_group_count != plan->maximum_bound_experts ||
        plan->dense_down_gemm.cutlass_group_count != plan->maximum_bound_experts)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

static cudaError_t SparkGlm52SotaValidateSparseGroupedMoePlan(const SparkGlm52SotaProductionMoePlanSm121 *plan)
{
    uint64_t required_flags;

    required_flags = SPARK_GLM52_SOTA_FAST_CAP_TOKEN_QUANT_ONCE |
        SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROUTER_TOPK |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP |
        SPARK_GLM52_SOTA_FAST_CAP_FUSED_SILU_REQUANT |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN |
        SPARK_GLM52_SOTA_FAST_CAP_WEIGHTED_COMBINE |
        SPARK_GLM52_SOTA_FAST_CAP_NO_HOST_STAGING |
        SPARK_GLM52_SOTA_FAST_CAP_NO_DEVICE_MEMCPY |
        SPARK_GLM52_SOTA_FAST_CAP_FIXED_GLM52_SHAPES;
    if ((plan->capability_flags & required_flags) != required_flags)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->expert_id_to_bound_slot == 0 ||
        plan->expert_route_counts == 0 ||
        plan->expert_route_offsets == 0 ||
        plan->expert_route_cursors == 0 ||
        plan->route_indices_by_expert == 0 ||
        plan->grouped_row_by_route == 0 ||
        plan->overflow_flag == 0 ||
        plan->grouped_hidden_nvfp4_by_route.payload_u8 == 0 ||
        plan->grouped_hidden_nvfp4_by_route.scale_e4m3_u8 == 0 ||
        plan->gate_up_bf16_by_grouped_route == 0 ||
        plan->intermediate_nvfp4_by_grouped_route.payload_u8 == 0 ||
        plan->intermediate_nvfp4_by_grouped_route.scale_e4m3_u8 == 0 ||
        plan->down_bf16_by_grouped_route == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (SparkGlm52SotaValidateNvfp4GemmPlan(&plan->gate_up_gemm, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u, 0u) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    if (SparkGlm52SotaValidateNvfp4GemmPlan(&plan->down_gemm, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, 0u) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

static cudaError_t SparkGlm52SotaValidateProductionMoePlan(const SparkGlm52SotaProductionMoePlanSm121 *plan)
{
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->maximum_tokens == 0u ||
        plan->maximum_tokens > 128u ||
        plan->maximum_routes < plan->maximum_tokens * SPARK_GLM52_SOTA_TOP_K ||
        plan->maximum_bound_experts == 0u ||
        plan->maximum_bound_experts > SPARK_GLM52_SOTA_EXPERT_COUNT ||
        plan->validated_latency_ns == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if ((plan->capability_flags & SPARK_GLM52_SOTA_FAST_CAP_DENSE_ALPHA_MOE) != 0u)
    {
        return SparkGlm52SotaValidateDenseAlphaMoePlan(plan);
    }
    return SparkGlm52SotaValidateSparseGroupedMoePlan(plan);
}


extern "C" cudaError_t SparkGlm52SotaValidateProductionDecodePlanSm121(const SparkGlm52SotaProductionDecodePlanSm121 *plan)
{
    uint64_t missing_flags;
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI)
    {
        return cudaErrorInvalidValue;
    }
    missing_flags = SPARK_GLM52_SOTA_REQUIRED_FAST_CAPS & ~plan->capability_flags;
    if (missing_flags != 0u || plan->validated_stage_latency_ns == 0u || plan->active_token_count == 0u || plan->active_token_count > plan->maximum_active_token_count || plan->maximum_active_token_count > 128u)
    {
        return cudaErrorInvalidValue;
    }
    if (SparkGlm52SotaValidateProductionMoePlan(plan->moe_plan) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    if (SparkGlm52SotaValidateBf16LinearPlan(plan->router_plan, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_EXPERT_COUNT) != cudaSuccess || SparkGlm52SotaValidateBf16LinearPlan(plan->q_a_plan, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, 2048u) != cudaSuccess || SparkGlm52SotaValidateBf16LinearPlan(plan->q_b_plan, 2048u, SPARK_GLM52_SOTA_HEAD_COUNT * 256u) != cudaSuccess || SparkGlm52SotaValidateBf16LinearPlan(plan->kv_a_plan, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_LATENT_DIMENSION + SPARK_GLM52_SOTA_ROPE_DIMENSION) != cudaSuccess || SparkGlm52SotaValidateBf16LinearPlan(plan->kv_b_plan, SPARK_GLM52_SOTA_LATENT_DIMENSION, SPARK_GLM52_SOTA_HEAD_COUNT * (192u + SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION)) != cudaSuccess || SparkGlm52SotaValidateBf16LinearPlan(plan->o_proj_plan, SPARK_GLM52_SOTA_HEAD_COUNT * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION, SPARK_GLM52_SOTA_HIDDEN_DIMENSION) != cudaSuccess || SparkGlm52SotaValidateBf16LinearPlan(plan->restricted_logits_plan, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, plan->restricted_logits_plan->n) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->flash_mla_plan == 0 || plan->mtp_plan == 0 || plan->graph_plan == 0 || plan->graph_plan->graph_exec == 0)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

extern "C" cudaError_t SparkGlm52SotaLaunchBf16CublasLtLinearSm121(SparkGlm52SotaBf16LinearPlanSm121 *plan, cudaStream_t stream)
{
    cublasStatus_t status;
    if (plan == 0 || SparkGlm52SotaValidateBf16LinearPlan(plan, plan->k, plan->n) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    status = cublasLtMatmul(plan->handle, plan->operation_desc, &plan->alpha, plan->a, plan->a_layout, plan->b, plan->b_layout, &plan->beta, plan->c == 0 ? plan->d : plan->c, plan->c_layout, plan->d, plan->d_layout, &plan->algo, plan->workspace, (size_t)plan->workspace_bytes, stream);
    return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}
