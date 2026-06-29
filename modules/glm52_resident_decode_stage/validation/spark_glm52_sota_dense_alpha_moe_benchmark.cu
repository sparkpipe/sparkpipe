#include "spark_glm52_sota_production_plan_sm121.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static uint64_t SparkBenchCeilDivU64(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

static uint64_t SparkBenchSm1xxScaleBytes(uint32_t groups, uint32_t rows, uint32_t columns)
{
    return (uint64_t)groups * SparkBenchCeilDivU64(rows, 128u) * SparkBenchCeilDivU64(columns, 64u) * 512u;
}

static int32_t SparkBenchCuda(cudaError_t error, const char *label)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "%s failed: %s\n", label, cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

static int32_t SparkBenchParseU32(int argc, char **argv, const char *name, uint32_t *value)
{
    int32_t argument_index;
    char *endptr;
    unsigned long parsed;

    for (argument_index = 1; argument_index < argc - 1; ++argument_index)
    {
        if (strcmp(argv[argument_index], name) == 0)
        {
            parsed = strtoul(argv[argument_index + 1], &endptr, 10);
            if (*endptr != 0 || parsed > 0xfffffffful)
            {
                return -1;
            }
            *value = (uint32_t)parsed;
            return 1;
        }
    }
    return 0;
}

static int32_t SparkBenchDeviceAlloc(void **pointer, uint64_t bytes, uint8_t fill, const char *label)
{
    if (bytes == 0u)
    {
        fprintf(stderr, "%s requested zero bytes\n", label);
        return -1;
    }
    if (SparkBenchCuda(cudaMalloc(pointer, bytes), label) < 0)
    {
        return -2;
    }
    if (SparkBenchCuda(cudaMemset(*pointer, fill, bytes), "cudaMemset") < 0)
    {
        return -3;
    }
    return 0;
}

static int32_t SparkBenchInitRoutes(int32_t *slots_device, float *weights_device, uint32_t tokens, uint32_t groups)
{
    uint32_t routes = tokens * SPARK_GLM52_SOTA_TOP_K;
    std::vector<int32_t> slots(routes);
    std::vector<float> weights(routes);
    uint32_t route_index;

    for (route_index = 0u; route_index < routes; ++route_index)
    {
        slots[route_index] = (int32_t)(route_index % groups);
        weights[route_index] = 1.0f / (float)SPARK_GLM52_SOTA_TOP_K;
    }
    if (SparkBenchCuda(cudaMemcpy(slots_device, slots.data(), routes * sizeof(int32_t), cudaMemcpyHostToDevice), "cudaMemcpy topk slots") < 0)
    {
        return -1;
    }
    if (SparkBenchCuda(cudaMemcpy(weights_device, weights.data(), routes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy topk weights") < 0)
    {
        return -2;
    }
    return 0;
}

static int32_t SparkBenchInitCutlassGemm(
    SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *gemm,
    uint32_t kind,
    uint32_t m,
    uint32_t n_capacity,
    uint32_t k,
    uint32_t groups,
    uint64_t capability,
    const uint8_t *a,
    const uint8_t *sfa,
    const uint8_t *b,
    const uint8_t *sfb,
    const void *c,
    void *d,
    int32_t *tokens_device,
    int32_t *tokens_host,
    void *workspace,
    uint64_t workspace_bytes,
    void *state,
    uint64_t state_bytes,
    uint32_t broadcast_b,
    uint32_t broadcast_sfb,
    cudaStream_t stream)
{
    memset(gemm, 0, sizeof(*gemm));
    gemm->abi_version = SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI;
    gemm->gemm_kind = kind;
    gemm->maximum_problem_count = groups;
    gemm->problem_count = groups;
    gemm->expected_k = k;
    gemm->expected_n = m;
    gemm->expected_group_size = SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE;
    gemm->output_is_nvfp4 = 0u;
    gemm->capability_flags = capability | SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT;
    if (broadcast_b != 0u || broadcast_sfb != 0u)
    {
        gemm->capability_flags |= SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_B_BROADCAST;
    }
    gemm->workspace = workspace;
    gemm->workspace_bytes = workspace_bytes;
    gemm->cutlass_m = m;
    gemm->cutlass_n_capacity = n_capacity;
    gemm->cutlass_k = k;
    gemm->cutlass_group_count = groups;
    gemm->tokens_per_expert_device = tokens_device;
    gemm->tokens_per_expert_host = tokens_host;
    gemm->cutlass_a_payload_u8 = a;
    gemm->cutlass_a_scale_ue4m3_u8 = sfa;
    gemm->cutlass_b_payload_u8 = b;
    gemm->cutlass_b_scale_ue4m3_u8 = sfb;
    gemm->cutlass_c_bf16 = c;
    gemm->cutlass_d_bf16 = d;
    gemm->cutlass_b_broadcast = broadcast_b;
    gemm->cutlass_sfb_broadcast = broadcast_sfb;
    return SparkBenchCuda(SparkGlm52SotaInitializeCutlassNvfp4GroupedGemmSm121(gemm, state, state_bytes, stream), "SparkGlm52SotaInitializeCutlassNvfp4GroupedGemmSm121");
}

static int32_t SparkBenchRunDenseAlpha(SparkGlm52SotaProductionMoePlanSm121 *plan, SparkGlm52SotaMoeArguments *arguments, uint32_t warmup, uint32_t iterations, cudaStream_t stream)
{
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float elapsed_ms;
    double gate_flops;
    double down_flops;
    double total_flops;
    double tflops;
    double payload_bytes;
    double scale_bytes;
    double effective_gbps;
    uint32_t iteration_index;
    uint32_t overflow = 0u;

    if (SparkBenchCuda(cudaEventCreate(&start_event), "cudaEventCreate start") < 0)
    {
        return -1;
    }
    if (SparkBenchCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop") < 0)
    {
        return -2;
    }
    for (iteration_index = 0u; iteration_index < warmup; ++iteration_index)
    {
        if (SparkBenchCuda(SparkGlm52SotaLaunchProductionGroupedMoeSm121(plan, arguments, stream), "warmup launch") < 0)
        {
            return -3;
        }
    }
    if (SparkBenchCuda(cudaStreamSynchronize(stream), "warmup sync") < 0)
    {
        return -4;
    }
    if (SparkBenchCuda(cudaEventRecord(start_event, stream), "cudaEventRecord start") < 0)
    {
        return -5;
    }
    for (iteration_index = 0u; iteration_index < iterations; ++iteration_index)
    {
        if (SparkBenchCuda(SparkGlm52SotaLaunchProductionGroupedMoeSm121(plan, arguments, stream), "bench launch") < 0)
        {
            return -6;
        }
    }
    if (SparkBenchCuda(cudaEventRecord(stop_event, stream), "cudaEventRecord stop") < 0)
    {
        return -7;
    }
    if (SparkBenchCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop") < 0)
    {
        return -8;
    }
    if (SparkBenchCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime") < 0)
    {
        return -9;
    }
    if (SparkBenchCuda(cudaMemcpy(&overflow, plan->overflow_flag, sizeof(overflow), cudaMemcpyDeviceToHost), "cudaMemcpy overflow") < 0)
    {
        return -10;
    }
    gate_flops = 2.0 * (double)plan->maximum_bound_experts * (double)plan->dense_alpha_token_capacity * (double)(SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * (double)SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
    down_flops = 2.0 * (double)plan->maximum_bound_experts * (double)plan->dense_alpha_token_capacity * (double)SPARK_GLM52_SOTA_HIDDEN_DIMENSION * (double)SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION;
    total_flops = gate_flops + down_flops;
    tflops = (total_flops * (double)iterations) / ((double)elapsed_ms * 1.0e9);
    payload_bytes = (double)plan->maximum_bound_experts * ((double)(SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * (double)SPARK_GLM52_SOTA_HIDDEN_DIMENSION * 0.5 + (double)SPARK_GLM52_SOTA_HIDDEN_DIMENSION * (double)SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 0.5);
    scale_bytes = (double)SparkBenchSm1xxScaleBytes(plan->maximum_bound_experts, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u, SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + (double)SparkBenchSm1xxScaleBytes(plan->maximum_bound_experts, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION);
    effective_gbps = ((payload_bytes + scale_bytes) * (double)iterations) / ((double)elapsed_ms * 1.0e6);
    printf("dense_alpha_moe_ready=1 active_tokens=%u active_routes=%u bound_experts=%u dense_token_capacity=%u dense_rows=%llu iterations=%u total_ms=%.3f avg_us=%.3f dense_estimated_tflops=%.3f expert_payload_scale_gbps=%.3f overflow=%u\n",
        arguments->active_token_count,
        arguments->active_route_count,
        plan->maximum_bound_experts,
        plan->dense_alpha_token_capacity,
        (unsigned long long)((uint64_t)plan->maximum_bound_experts * plan->dense_alpha_token_capacity),
        iterations,
        elapsed_ms,
        (elapsed_ms * 1000.0f) / (float)iterations,
        tflops,
        effective_gbps,
        overflow);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    return overflow == 0u ? 0 : -11;
}

int main(int argc, char **argv)
{
    uint32_t tokens = 96u;
    uint32_t groups = SPARK_GLM52_SOTA_EXPERT_COUNT;
    uint32_t capacity = 96u;
    uint32_t warmup = 5u;
    uint32_t iterations = 20u;
    uint32_t workspace_mb = 1024u;
    uint32_t routes;
    uint64_t dense_rows;
    uint64_t state_bytes;
    uint64_t workspace_bytes;
    cudaStream_t stream;
    SparkGlm52SotaProductionMoePlanSm121 plan;
    SparkGlm52SotaMoeArguments arguments;
    SparkGlm52SotaGroupedRouteWorkspace route_workspace;
    void *workspace = 0;
    void *gate_state = 0;
    void *down_state = 0;
    int32_t *tokens_device = 0;
    std::vector<int32_t> tokens_host;
    __nv_bfloat16 *hidden = 0;
    uint8_t *hidden_payload = 0;
    uint8_t *hidden_scales_linear = 0;
    uint8_t *hidden_scales_sm1xx = 0;
    int32_t *topk_slots = 0;
    float *topk_weights = 0;
    uint32_t *overflow = 0;
    uint8_t *gate_a = 0;
    uint8_t *gate_sfa = 0;
    __nv_bfloat16 *gate_c = 0;
    __nv_bfloat16 *gate_d = 0;
    uint8_t *down_a = 0;
    uint8_t *down_sfa = 0;
    uint8_t *intermediate = 0;
    uint8_t *intermediate_sfb = 0;
    __nv_bfloat16 *down_c = 0;
    __nv_bfloat16 *down_d = 0;
    __nv_bfloat16 *combined = 0;

    SparkBenchParseU32(argc, argv, "--tokens", &tokens);
    SparkBenchParseU32(argc, argv, "--groups", &groups);
    SparkBenchParseU32(argc, argv, "--capacity", &capacity);
    SparkBenchParseU32(argc, argv, "--warmup", &warmup);
    SparkBenchParseU32(argc, argv, "--iterations", &iterations);
    SparkBenchParseU32(argc, argv, "--workspace-mb", &workspace_mb);
    if (tokens == 0u || capacity == 0u || tokens != capacity || capacity > 128u || groups != SPARK_GLM52_SOTA_EXPERT_COUNT || iterations == 0u)
    {
        fprintf(stderr, "invalid dense-alpha benchmark shape: exact GLM52 dense-alpha requires --groups 256 and --capacity equal to --tokens\n");
        return 2;
    }
    routes = tokens * SPARK_GLM52_SOTA_TOP_K;
    dense_rows = (uint64_t)groups * capacity;
    state_bytes = SparkGlm52SotaCutlassNvfp4GroupedGemmStateBytesSm121();
    if (state_bytes == 0u)
    {
        fprintf(stderr, "CUTLASS NVFP4 support is not compiled in\n");
        return 3;
    }
    workspace_bytes = (uint64_t)workspace_mb * 1024u * 1024u;
    tokens_host.assign(groups, (int32_t)capacity);
    if (SparkBenchCuda(cudaStreamCreate(&stream), "cudaStreamCreate") < 0)
    {
        return 4;
    }
    if (SparkBenchDeviceAlloc(&workspace, workspace_bytes, 0, "cudaMalloc workspace") < 0)
    {
        return 5;
    }
    gate_state = malloc(state_bytes);
    down_state = malloc(state_bytes);
    if (gate_state == 0 || down_state == 0)
    {
        return 6;
    }
    if (SparkBenchCuda(cudaMalloc((void **)&tokens_device, groups * sizeof(int32_t)), "cudaMalloc tokens") < 0)
    {
        return 7;
    }
    if (SparkBenchCuda(cudaMemcpy(tokens_device, tokens_host.data(), groups * sizeof(int32_t), cudaMemcpyHostToDevice), "cudaMemcpy tokens") < 0)
    {
        return 8;
    }
    if (SparkBenchDeviceAlloc((void **)&hidden, (uint64_t)capacity * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * sizeof(__nv_bfloat16), 0, "cudaMalloc hidden") < 0)
    {
        return 9;
    }
    if (SparkBenchDeviceAlloc((void **)&hidden_payload, (uint64_t)capacity * (SPARK_GLM52_SOTA_HIDDEN_DIMENSION / 2u), 0, "cudaMalloc hidden payload") < 0)
    {
        return 10;
    }
    if (SparkBenchDeviceAlloc((void **)&hidden_scales_linear, (uint64_t)capacity * SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_HIDDEN, 0x38, "cudaMalloc hidden scales linear") < 0)
    {
        return 11;
    }
    if (SparkBenchDeviceAlloc((void **)&hidden_scales_sm1xx, SparkBenchSm1xxScaleBytes(1u, capacity, SPARK_GLM52_SOTA_HIDDEN_DIMENSION), 0x38, "cudaMalloc hidden scales sm1xx") < 0)
    {
        return 12;
    }
    if (SparkBenchDeviceAlloc((void **)&topk_slots, (uint64_t)routes * sizeof(int32_t), 0, "cudaMalloc topk slots") < 0)
    {
        return 13;
    }
    if (SparkBenchDeviceAlloc((void **)&topk_weights, (uint64_t)routes * sizeof(float), 0, "cudaMalloc topk weights") < 0)
    {
        return 14;
    }
    if (SparkBenchInitRoutes(topk_slots, topk_weights, tokens, groups) < 0)
    {
        return 15;
    }
    if (SparkBenchDeviceAlloc((void **)&overflow, sizeof(uint32_t), 0, "cudaMalloc overflow") < 0)
    {
        return 16;
    }
    if (SparkBenchDeviceAlloc((void **)&gate_a, (uint64_t)groups * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * SPARK_GLM52_SOTA_HIDDEN_DIMENSION / 2u, 0, "cudaMalloc gate A") < 0)
    {
        return 17;
    }
    if (SparkBenchDeviceAlloc((void **)&gate_sfa, SparkBenchSm1xxScaleBytes(groups, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u, SPARK_GLM52_SOTA_HIDDEN_DIMENSION), 0x38, "cudaMalloc gate SFA") < 0)
    {
        return 18;
    }
    if (SparkBenchDeviceAlloc((void **)&gate_c, dense_rows * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * sizeof(__nv_bfloat16), 0, "cudaMalloc gate C") < 0)
    {
        return 19;
    }
    if (SparkBenchDeviceAlloc((void **)&gate_d, dense_rows * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * sizeof(__nv_bfloat16), 0, "cudaMalloc gate D") < 0)
    {
        return 20;
    }
    if (SparkBenchDeviceAlloc((void **)&down_a, (uint64_t)groups * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION / 2u, 0, "cudaMalloc down A") < 0)
    {
        return 21;
    }
    if (SparkBenchDeviceAlloc((void **)&down_sfa, SparkBenchSm1xxScaleBytes(groups, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION), 0x38, "cudaMalloc down SFA") < 0)
    {
        return 22;
    }
    if (SparkBenchDeviceAlloc((void **)&intermediate, dense_rows * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION / 2u), 0, "cudaMalloc intermediate") < 0)
    {
        return 23;
    }
    if (SparkBenchDeviceAlloc((void **)&intermediate_sfb, SparkBenchSm1xxScaleBytes(groups, capacity, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION), 0x38, "cudaMalloc intermediate SFB") < 0)
    {
        return 24;
    }
    if (SparkBenchDeviceAlloc((void **)&down_c, dense_rows * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * sizeof(__nv_bfloat16), 0, "cudaMalloc down C") < 0)
    {
        return 25;
    }
    if (SparkBenchDeviceAlloc((void **)&down_d, dense_rows * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * sizeof(__nv_bfloat16), 0, "cudaMalloc down D") < 0)
    {
        return 26;
    }
    if (SparkBenchDeviceAlloc((void **)&combined, (uint64_t)tokens * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * sizeof(__nv_bfloat16), 0, "cudaMalloc combined") < 0)
    {
        return 27;
    }

    memset(&plan, 0, sizeof(plan));
    memset(&arguments, 0, sizeof(arguments));
    memset(&route_workspace, 0, sizeof(route_workspace));
    plan.abi_version = SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI;
    plan.maximum_tokens = capacity;
    plan.maximum_routes = capacity * SPARK_GLM52_SOTA_TOP_K;
    plan.maximum_bound_experts = groups;
    plan.validated_latency_ns = 1u;
    plan.capability_flags = SPARK_GLM52_SOTA_FAST_CAP_TOKEN_QUANT_ONCE |
        SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROUTER_TOPK |
        SPARK_GLM52_SOTA_FAST_CAP_DENSE_ALPHA_MOE |
        SPARK_GLM52_SOTA_FAST_CAP_DENSE_ALPHA_STRICT |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_B_BROADCAST |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP |
        SPARK_GLM52_SOTA_FAST_CAP_FUSED_SILU_REQUANT |
        SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN |
        SPARK_GLM52_SOTA_FAST_CAP_WEIGHTED_COMBINE |
        SPARK_GLM52_SOTA_FAST_CAP_NO_HOST_STAGING |
        SPARK_GLM52_SOTA_FAST_CAP_NO_DEVICE_MEMCPY |
        SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT |
        SPARK_GLM52_SOTA_FAST_CAP_FIXED_GLM52_SHAPES;
    plan.overflow_flag = overflow;
    plan.dense_alpha_token_capacity = capacity;
    plan.dense_alpha_minimum_tokens = tokens;
    plan.dense_alpha_maximum_tokens = capacity;
    plan.dense_alpha_require_exact_token_count = 1u;
    plan.dense_hidden_nvfp4_by_token_sm1xx.payload_u8 = hidden_payload;
    plan.dense_hidden_nvfp4_by_token_sm1xx.scale_e4m3_u8 = hidden_scales_sm1xx;
    plan.dense_hidden_nvfp4_by_token_sm1xx.row_count = capacity;
    plan.dense_hidden_nvfp4_by_token_sm1xx.column_count = SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
    plan.dense_hidden_nvfp4_by_token_sm1xx.packed_row_stride_bytes = SPARK_GLM52_SOTA_HIDDEN_DIMENSION / 2u;
    plan.dense_hidden_nvfp4_by_token_sm1xx.scale_row_stride_bytes = (uint32_t)SparkBenchSm1xxScaleBytes(1u, capacity, SPARK_GLM52_SOTA_HIDDEN_DIMENSION);
    plan.dense_gate_up_bf16_by_expert_token = gate_d;
    plan.dense_intermediate_nvfp4_by_expert_token.payload_u8 = intermediate;
    plan.dense_intermediate_nvfp4_by_expert_token.scale_e4m3_u8 = intermediate_sfb;
    plan.dense_intermediate_nvfp4_by_expert_token.row_count = (uint32_t)dense_rows;
    plan.dense_intermediate_nvfp4_by_expert_token.column_count = SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION;
    plan.dense_intermediate_nvfp4_by_expert_token.packed_row_stride_bytes = SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION / 2u;
    plan.dense_intermediate_nvfp4_by_expert_token.scale_row_stride_bytes = (uint32_t)SparkBenchSm1xxScaleBytes(1u, capacity, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION);
    plan.dense_down_bf16_by_expert_token = down_d;
    if (SparkBenchInitCutlassGemm(&plan.dense_gate_up_gemm, SPARK_GLM52_SOTA_GEMM_KIND_GATE_UP, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u, capacity, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, groups, SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP, gate_a, gate_sfa, hidden_payload, hidden_scales_sm1xx, gate_c, gate_d, tokens_device, tokens_host.data(), workspace, workspace_bytes, gate_state, state_bytes, 1u, 1u, stream) < 0)
    {
        return 28;
    }
    if (SparkBenchInitCutlassGemm(&plan.dense_down_gemm, SPARK_GLM52_SOTA_GEMM_KIND_DOWN, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, capacity, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION, groups, SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN, down_a, down_sfa, intermediate, intermediate_sfb, down_c, down_d, tokens_device, tokens_host.data(), workspace, workspace_bytes, down_state, state_bytes, 0u, 0u, stream) < 0)
    {
        return 29;
    }
    route_workspace.topk_bound_slots = topk_slots;
    route_workspace.topk_weights = topk_weights;
    arguments.active_token_count = tokens;
    arguments.active_route_count = routes;
    arguments.bound_expert_count = groups;
    arguments.hidden_bf16 = hidden;
    arguments.hidden_nvfp4_by_token.payload_u8 = hidden_payload;
    arguments.hidden_nvfp4_by_token.scale_e4m3_u8 = hidden_scales_linear;
    arguments.hidden_nvfp4_by_token.row_count = capacity;
    arguments.hidden_nvfp4_by_token.column_count = SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
    arguments.hidden_nvfp4_by_token.packed_row_stride_bytes = SPARK_GLM52_SOTA_HIDDEN_DIMENSION / 2u;
    arguments.hidden_nvfp4_by_token.scale_row_stride_bytes = SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_HIDDEN;
    arguments.route_workspace = route_workspace;
    arguments.gate_weight_nvfp4.payload_u8 = gate_a;
    arguments.gate_weight_nvfp4.scale_e4m3_u8 = gate_sfa;
    arguments.down_weight_nvfp4.payload_u8 = down_a;
    arguments.down_weight_nvfp4.scale_e4m3_u8 = down_sfa;
    arguments.combined_output_bf16 = combined;
    return SparkBenchRunDenseAlpha(&plan, &arguments, warmup, iterations, stream) == 0 ? 0 : 30;
}
