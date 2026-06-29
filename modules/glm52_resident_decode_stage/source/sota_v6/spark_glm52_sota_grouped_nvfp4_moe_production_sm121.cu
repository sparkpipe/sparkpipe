#include "spark_glm52_sota_production_plan_sm121.cuh"

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaClearU32Kernel(uint32_t *values, uint32_t count)
{
    uint32_t index = threadIdx.x + (blockIdx.x * blockDim.x);
    while (index < count)
    {
        values[index] = 0u;
        index += blockDim.x * gridDim.x;
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaCountRoutesByExpertKernel(const SparkGlm52SotaMoeArguments arguments, SparkGlm52SotaProductionMoePlanSm121 plan)
{
    uint32_t route_index = threadIdx.x + (blockIdx.x * blockDim.x);
    while (route_index < arguments.active_route_count)
    {
        int32_t bound_slot = arguments.route_workspace.topk_bound_slots[route_index];
        if (bound_slot >= 0 && (uint32_t)bound_slot < plan.maximum_bound_experts)
        {
            atomicAdd(&plan.expert_route_counts[(uint32_t)bound_slot], 1u);
        }
        else
        {
            atomicExch(plan.overflow_flag, 1u);
        }
        route_index += blockDim.x * gridDim.x;
    }
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52SotaExclusiveScan256Kernel(uint32_t *offsets, const uint32_t *counts, uint32_t *overflow_flag, uint32_t maximum_bound_experts, uint32_t maximum_routes)
{
    __shared__ uint32_t scan[SPARK_GLM52_SOTA_EXPERT_COUNT];
    uint32_t lane = threadIdx.x;
    uint32_t value = lane < maximum_bound_experts ? counts[lane] : 0u;
    uint32_t total;

    scan[lane] = value;
    __syncthreads();
    #pragma unroll
    for (uint32_t stride = 1u; stride < SPARK_GLM52_SOTA_EXPERT_COUNT; stride <<= 1u)
    {
        uint32_t addend = lane >= stride ? scan[lane - stride] : 0u;
        __syncthreads();
        scan[lane] += addend;
        __syncthreads();
    }
    if (lane < maximum_bound_experts)
    {
        offsets[lane] = lane == 0u ? 0u : scan[lane - 1u];
    }
    total = maximum_bound_experts == 0u ? 0u : scan[maximum_bound_experts - 1u];
    if (lane == 0u && total > maximum_routes)
    {
        *overflow_flag = 1u;
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaScatterRoutesByExpertKernel(const SparkGlm52SotaMoeArguments arguments, SparkGlm52SotaProductionMoePlanSm121 plan)
{
    uint32_t route_index = threadIdx.x + (blockIdx.x * blockDim.x);
    while (route_index < arguments.active_route_count)
    {
        int32_t bound_slot = arguments.route_workspace.topk_bound_slots[route_index];
        if (bound_slot >= 0 && (uint32_t)bound_slot < plan.maximum_bound_experts)
        {
            uint32_t slot = (uint32_t)bound_slot;
            uint32_t write_index = atomicAdd(&plan.expert_route_cursors[slot], 1u);
            uint32_t grouped_row = plan.expert_route_offsets[slot] + write_index;
            if (grouped_row < plan.maximum_routes)
            {
                plan.route_indices_by_expert[grouped_row] = route_index;
                plan.grouped_row_by_route[route_index] = grouped_row;
            }
            else
            {
                atomicExch(plan.overflow_flag, 1u);
            }
        }
        route_index += blockDim.x * gridDim.x;
    }
}

static __global__ __launch_bounds__(128, 4)
void SparkGlm52SotaQuantizeHiddenOnceNvfp4Kernel(const SparkGlm52SotaMoeArguments arguments)
{
    uint32_t token_index = blockIdx.y;
    uint32_t group_index = blockIdx.x * 4u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t column_base = group_index * SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE;
    float first = 0.0f;
    float second = 0.0f;
    float maximum_value;
    float scale;
    uint8_t scale_code;

    if (token_index >= arguments.active_token_count || group_index >= SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_HIDDEN)
    {
        return;
    }
    if (lane < 8u)
    {
        uint32_t column = column_base + lane * 2u;
        first = SparkGlm52SotaBf16ToFloat(arguments.hidden_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + column]);
        second = SparkGlm52SotaBf16ToFloat(arguments.hidden_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + column + 1u]);
    }
    maximum_value = lane < 8u ? fmaxf(fabsf(first), fabsf(second)) : 0.0f;
    maximum_value = SparkGlm52SotaWarpReduceMax(maximum_value);
    maximum_value = __shfl_sync(0xffffffffu, maximum_value, 0);
    scale = fmaxf(maximum_value * 0.1666666667f, 1.0e-8f);
    scale_code = SparkGlm52SotaEncodePositiveE4m3(scale);
    scale = SparkGlm52SotaDecodeE4m3(scale_code);
    if (lane == 0u)
    {
        arguments.hidden_nvfp4_by_token.scale_e4m3_u8[((uint64_t)token_index * arguments.hidden_nvfp4_by_token.scale_row_stride_bytes) + group_index] = scale_code;
    }
    if (lane < 8u)
    {
        uint64_t packed_index = ((uint64_t)token_index * arguments.hidden_nvfp4_by_token.packed_row_stride_bytes) + ((uint64_t)(column_base + lane * 2u) >> 1u);
        SparkGlm52SotaStoreNvfp4Pair(arguments.hidden_nvfp4_by_token.payload_u8, packed_index, SparkGlm52SotaEncodeE2m1(first / scale), SparkGlm52SotaEncodeE2m1(second / scale));
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaPackGroupedHiddenRowsKernel(const SparkGlm52SotaMoeArguments arguments, SparkGlm52SotaProductionMoePlanSm121 plan)
{
    uint32_t grouped_row = blockIdx.y;
    uint32_t byte_index = threadIdx.x + (blockIdx.x * blockDim.x);
    uint32_t route_index;
    uint32_t token_index;

    if (grouped_row >= arguments.active_route_count)
    {
        return;
    }
    route_index = plan.route_indices_by_expert[grouped_row];
    token_index = route_index / SPARK_GLM52_SOTA_TOP_K;
    while (byte_index < arguments.hidden_nvfp4_by_token.packed_row_stride_bytes)
    {
        plan.grouped_hidden_nvfp4_by_route.payload_u8[((uint64_t)grouped_row * plan.grouped_hidden_nvfp4_by_route.packed_row_stride_bytes) + byte_index] = arguments.hidden_nvfp4_by_token.payload_u8[((uint64_t)token_index * arguments.hidden_nvfp4_by_token.packed_row_stride_bytes) + byte_index];
        byte_index += blockDim.x * gridDim.x;
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaPackGroupedHiddenScalesKernel(const SparkGlm52SotaMoeArguments arguments, SparkGlm52SotaProductionMoePlanSm121 plan)
{
    uint32_t grouped_row = blockIdx.y;
    uint32_t scale_index = threadIdx.x + (blockIdx.x * blockDim.x);
    uint32_t route_index;
    uint32_t token_index;

    if (grouped_row >= arguments.active_route_count)
    {
        return;
    }
    route_index = plan.route_indices_by_expert[grouped_row];
    token_index = route_index / SPARK_GLM52_SOTA_TOP_K;
    while (scale_index < SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_HIDDEN)
    {
        plan.grouped_hidden_nvfp4_by_route.scale_e4m3_u8[((uint64_t)grouped_row * plan.grouped_hidden_nvfp4_by_route.scale_row_stride_bytes) + scale_index] = arguments.hidden_nvfp4_by_token.scale_e4m3_u8[((uint64_t)token_index * arguments.hidden_nvfp4_by_token.scale_row_stride_bytes) + scale_index];
        scale_index += blockDim.x * gridDim.x;
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaBuildGroupedGemmProblemsKernel(const SparkGlm52SotaMoeArguments arguments, SparkGlm52SotaProductionMoePlanSm121 plan)
{
    uint32_t expert_slot = threadIdx.x + (blockIdx.x * blockDim.x);
    while (expert_slot < plan.maximum_bound_experts)
    {
        uint32_t route_count = plan.expert_route_counts[expert_slot];
        uint32_t route_offset = plan.expert_route_offsets[expert_slot];
        SparkGlm52SotaNvfp4GroupedGemmProblemSm121 gate_up_problem;
        SparkGlm52SotaNvfp4GroupedGemmProblemSm121 down_problem;
        uint64_t gate_up_weight_row = (uint64_t)expert_slot * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u);
        uint64_t down_weight_row = (uint64_t)expert_slot * SPARK_GLM52_SOTA_HIDDEN_DIMENSION;

        gate_up_problem.a_payload_u8 = plan.grouped_hidden_nvfp4_by_route.payload_u8 + ((uint64_t)route_offset * plan.grouped_hidden_nvfp4_by_route.packed_row_stride_bytes);
        gate_up_problem.a_scale_ue4m3_u8 = plan.grouped_hidden_nvfp4_by_route.scale_e4m3_u8 + ((uint64_t)route_offset * plan.grouped_hidden_nvfp4_by_route.scale_row_stride_bytes);
        gate_up_problem.b_payload_u8 = arguments.gate_weight_nvfp4.payload_u8 + (gate_up_weight_row * arguments.gate_weight_nvfp4.packed_row_stride_bytes);
        gate_up_problem.b_scale_ue4m3_u8 = arguments.gate_weight_nvfp4.scale_e4m3_u8 + (gate_up_weight_row * arguments.gate_weight_nvfp4.scale_row_stride_bytes);
        gate_up_problem.c = 0;
        gate_up_problem.d = plan.gate_up_bf16_by_grouped_route + ((uint64_t)route_offset * SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u);
        gate_up_problem.m = route_count;
        gate_up_problem.n = SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u;
        gate_up_problem.k = SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
        gate_up_problem.expert_slot = expert_slot;
        gate_up_problem.a_packed_row_stride_bytes = plan.grouped_hidden_nvfp4_by_route.packed_row_stride_bytes;
        gate_up_problem.a_scale_row_stride_bytes = plan.grouped_hidden_nvfp4_by_route.scale_row_stride_bytes;
        gate_up_problem.b_packed_row_stride_bytes = arguments.gate_weight_nvfp4.packed_row_stride_bytes;
        gate_up_problem.b_scale_row_stride_bytes = arguments.gate_weight_nvfp4.scale_row_stride_bytes;
        gate_up_problem.c_row_stride_elements = 0u;
        gate_up_problem.d_row_stride_elements_or_bytes = SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u;
        gate_up_problem.alpha = 1.0f;
        gate_up_problem.beta = 0.0f;

        down_problem = gate_up_problem;
        down_problem.a_payload_u8 = plan.intermediate_nvfp4_by_grouped_route.payload_u8 + ((uint64_t)route_offset * plan.intermediate_nvfp4_by_grouped_route.packed_row_stride_bytes);
        down_problem.a_scale_ue4m3_u8 = plan.intermediate_nvfp4_by_grouped_route.scale_e4m3_u8 + ((uint64_t)route_offset * plan.intermediate_nvfp4_by_grouped_route.scale_row_stride_bytes);
        down_problem.b_payload_u8 = arguments.down_weight_nvfp4.payload_u8 + (down_weight_row * arguments.down_weight_nvfp4.packed_row_stride_bytes);
        down_problem.b_scale_ue4m3_u8 = arguments.down_weight_nvfp4.scale_e4m3_u8 + (down_weight_row * arguments.down_weight_nvfp4.scale_row_stride_bytes);
        down_problem.d = plan.down_bf16_by_grouped_route + ((uint64_t)route_offset * SPARK_GLM52_SOTA_HIDDEN_DIMENSION);
        down_problem.m = route_count;
        down_problem.n = SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
        down_problem.k = SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION;
        down_problem.a_packed_row_stride_bytes = plan.intermediate_nvfp4_by_grouped_route.packed_row_stride_bytes;
        down_problem.a_scale_row_stride_bytes = plan.intermediate_nvfp4_by_grouped_route.scale_row_stride_bytes;
        down_problem.b_packed_row_stride_bytes = arguments.down_weight_nvfp4.packed_row_stride_bytes;
        down_problem.b_scale_row_stride_bytes = arguments.down_weight_nvfp4.scale_row_stride_bytes;
        down_problem.d_row_stride_elements_or_bytes = SPARK_GLM52_SOTA_HIDDEN_DIMENSION;

        plan.gate_up_gemm.problems_device[expert_slot] = gate_up_problem;
        plan.down_gemm.problems_device[expert_slot] = down_problem;
        expert_slot += blockDim.x * gridDim.x;
    }
}

static __global__ __launch_bounds__(128, 4)
void SparkGlm52SotaSiluMulRequantGroupedKernel(SparkGlm52SotaProductionMoePlanSm121 plan, const SparkGlm52SotaMoeArguments arguments)
{
    uint32_t grouped_route = blockIdx.y;
    uint32_t group_index = blockIdx.x;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t column_base = group_index * SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE;
    float first = 0.0f;
    float second = 0.0f;
    float maximum_value;
    float scale;
    uint8_t scale_code;

    if (grouped_route >= arguments.active_route_count || group_index >= SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_MOE)
    {
        return;
    }
    if (lane < 8u)
    {
        uint32_t column = column_base + lane * 2u;
        uint64_t base = ((uint64_t)grouped_route * SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) + column;
        float gate0 = SparkGlm52SotaBf16ToFloat(plan.gate_up_bf16_by_grouped_route[base]);
        float gate1 = SparkGlm52SotaBf16ToFloat(plan.gate_up_bf16_by_grouped_route[base + 1u]);
        float up0 = SparkGlm52SotaBf16ToFloat(plan.gate_up_bf16_by_grouped_route[base + SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION]);
        float up1 = SparkGlm52SotaBf16ToFloat(plan.gate_up_bf16_by_grouped_route[base + SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION + 1u]);
        first = SparkGlm52SotaSilu(gate0) * up0;
        second = SparkGlm52SotaSilu(gate1) * up1;
    }
    maximum_value = lane < 8u ? fmaxf(fabsf(first), fabsf(second)) : 0.0f;
    maximum_value = SparkGlm52SotaWarpReduceMax(maximum_value);
    maximum_value = __shfl_sync(0xffffffffu, maximum_value, 0);
    scale = fmaxf(maximum_value * 0.1666666667f, 1.0e-8f);
    scale_code = SparkGlm52SotaEncodePositiveE4m3(scale);
    scale = SparkGlm52SotaDecodeE4m3(scale_code);
    if (lane == 0u)
    {
        plan.intermediate_nvfp4_by_grouped_route.scale_e4m3_u8[((uint64_t)grouped_route * plan.intermediate_nvfp4_by_grouped_route.scale_row_stride_bytes) + group_index] = scale_code;
    }
    if (lane < 8u)
    {
        uint64_t packed_index = ((uint64_t)grouped_route * plan.intermediate_nvfp4_by_grouped_route.packed_row_stride_bytes) + ((uint64_t)(column_base + lane * 2u) >> 1u);
        SparkGlm52SotaStoreNvfp4Pair(plan.intermediate_nvfp4_by_grouped_route.payload_u8, packed_index, SparkGlm52SotaEncodeE2m1(first / scale), SparkGlm52SotaEncodeE2m1(second / scale));
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaWeightedCombineGroupedKernel(SparkGlm52SotaProductionMoePlanSm121 plan, const SparkGlm52SotaMoeArguments arguments)
{
    uint32_t token_index = blockIdx.y;
    uint32_t hidden_index = blockIdx.x * 256u + threadIdx.x;
    float accumulator = 0.0f;

    if (token_index >= arguments.active_token_count || hidden_index >= SPARK_GLM52_SOTA_HIDDEN_DIMENSION)
    {
        return;
    }
    #pragma unroll
    for (uint32_t topk_index = 0u; topk_index < SPARK_GLM52_SOTA_TOP_K; ++topk_index)
    {
        uint32_t route_index = token_index * SPARK_GLM52_SOTA_TOP_K + topk_index;
        uint32_t grouped_row = plan.grouped_row_by_route[route_index];
        float route_weight = arguments.route_workspace.topk_weights[route_index];
        float value = SparkGlm52SotaBf16ToFloat(plan.down_bf16_by_grouped_route[((uint64_t)grouped_row * SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + hidden_index]);
        accumulator = fmaf(value, route_weight, accumulator);
    }
    if (arguments.combined_output_f32 != 0)
    {
        arguments.combined_output_f32[((uint64_t)token_index * SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + hidden_index] = accumulator;
    }
    if (arguments.combined_output_bf16 != 0)
    {
        arguments.combined_output_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + hidden_index] = SparkGlm52SotaFloatToBf16(accumulator);
    }
}

static cudaError_t SparkGlm52SotaValidateProductionMoeLaunch(SparkGlm52SotaProductionMoePlanSm121 *plan, const SparkGlm52SotaMoeArguments *arguments)
{
    if (plan == 0 || arguments == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI)
    {
        return cudaErrorInvalidValue;
    }
    if (arguments->active_token_count == 0u || arguments->active_token_count > plan->maximum_tokens || arguments->active_route_count != arguments->active_token_count * SPARK_GLM52_SOTA_TOP_K || arguments->active_route_count > plan->maximum_routes)
    {
        return cudaErrorInvalidValue;
    }
    if ((plan->capability_flags & (SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP | SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN | SPARK_GLM52_SOTA_FAST_CAP_FUSED_SILU_REQUANT | SPARK_GLM52_SOTA_FAST_CAP_TOKEN_QUANT_ONCE)) == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (arguments->hidden_bf16 == 0 || arguments->hidden_nvfp4_by_token.payload_u8 == 0 || arguments->hidden_nvfp4_by_token.scale_e4m3_u8 == 0 || arguments->route_workspace.topk_bound_slots == 0 || arguments->route_workspace.topk_weights == 0 || arguments->gate_weight_nvfp4.payload_u8 == 0 || arguments->gate_weight_nvfp4.scale_e4m3_u8 == 0 || arguments->down_weight_nvfp4.payload_u8 == 0 || arguments->down_weight_nvfp4.scale_e4m3_u8 == 0)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

extern "C" cudaError_t SparkGlm52SotaLaunchProductionGroupedMoeSm121(SparkGlm52SotaProductionMoePlanSm121 *plan, const SparkGlm52SotaMoeArguments *arguments, cudaStream_t stream)
{
    cudaError_t error;
    uint32_t clear_grid;
    uint32_t route_grid;
    uint32_t group_grid;

    error = SparkGlm52SotaValidateProductionMoeLaunch(plan, arguments);
    if (error != cudaSuccess)
    {
        return error;
    }

    clear_grid = SparkGlm52SotaCeilDivU32(plan->maximum_bound_experts, 256u);
    SparkGlm52SotaClearU32Kernel<<<clear_grid, 256u, 0u, stream>>>(plan->expert_route_counts, plan->maximum_bound_experts);
    SparkGlm52SotaClearU32Kernel<<<clear_grid, 256u, 0u, stream>>>(plan->expert_route_cursors, plan->maximum_bound_experts);
    SparkGlm52SotaClearU32Kernel<<<1u, 256u, 0u, stream>>>(plan->overflow_flag, 1u);

    route_grid = SparkGlm52SotaCeilDivU32(arguments->active_route_count, 256u);
    group_grid = SparkGlm52SotaCeilDivU32(SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_HIDDEN, 4u);
    SparkGlm52SotaQuantizeHiddenOnceNvfp4Kernel<<<dim3(group_grid, arguments->active_token_count, 1u), 128u, 0u, stream>>>(*arguments);
    SparkGlm52SotaCountRoutesByExpertKernel<<<route_grid, 256u, 0u, stream>>>(*arguments, *plan);
    SparkGlm52SotaExclusiveScan256Kernel<<<1u, 256u, 0u, stream>>>(plan->expert_route_offsets, plan->expert_route_counts, plan->overflow_flag, plan->maximum_bound_experts, plan->maximum_routes);
    SparkGlm52SotaScatterRoutesByExpertKernel<<<route_grid, 256u, 0u, stream>>>(*arguments, *plan);
    SparkGlm52SotaPackGroupedHiddenRowsKernel<<<dim3(SparkGlm52SotaCeilDivU32(arguments->hidden_nvfp4_by_token.packed_row_stride_bytes, 256u), arguments->active_route_count, 1u), 256u, 0u, stream>>>(*arguments, *plan);
    SparkGlm52SotaPackGroupedHiddenScalesKernel<<<dim3(1u, arguments->active_route_count, 1u), 256u, 0u, stream>>>(*arguments, *plan);
    SparkGlm52SotaBuildGroupedGemmProblemsKernel<<<clear_grid, 256u, 0u, stream>>>(*arguments, *plan);

    plan->gate_up_gemm.problem_count = plan->maximum_bound_experts;
    error = plan->gate_up_gemm.launch(&plan->gate_up_gemm, stream);
    if (error != cudaSuccess)
    {
        return error;
    }
    SparkGlm52SotaSiluMulRequantGroupedKernel<<<dim3(SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_MOE, arguments->active_route_count, 1u), 128u, 0u, stream>>>(*plan, *arguments);

    plan->down_gemm.problem_count = plan->maximum_bound_experts;
    error = plan->down_gemm.launch(&plan->down_gemm, stream);
    if (error != cudaSuccess)
    {
        return error;
    }
    SparkGlm52SotaWeightedCombineGroupedKernel<<<dim3(SparkGlm52SotaCeilDivU32(SPARK_GLM52_SOTA_HIDDEN_DIMENSION, 256u), arguments->active_token_count, 1u), 256u, 0u, stream>>>(*plan, *arguments);
    return cudaGetLastError();
}
