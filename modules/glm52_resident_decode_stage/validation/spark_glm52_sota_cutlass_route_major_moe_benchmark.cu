#include "spark_glm52_sota_production_plan_sm121.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <vector>

#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#endif

static uint64_t SparkBenchCeilDivU64(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

static uint64_t SparkBenchScaleBytesOne(uint32_t rows, uint32_t columns)
{
    return SparkBenchCeilDivU64(rows, 128u) * SparkBenchCeilDivU64(columns, 64u) * 512u;
}

static uint64_t SparkBenchScaleBytes(uint32_t groups, uint32_t rows, uint32_t columns)
{
    return (uint64_t)groups * SparkBenchScaleBytesOne(rows, columns);
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
    int32_t i;
    char *endptr;
    unsigned long parsed;

    for (i=1; i<(argc - 1); i++)
    {
        if (strcmp(argv[i], name) == 0)
        {
            parsed = strtoul(argv[i + 1], &endptr, 10);
            if (*endptr != 0 || parsed > 0xfffffffful)
                return -1;
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
        return -2;
    if (SparkBenchCuda(cudaMemset(*pointer, fill, bytes), "cudaMemset") < 0)
        return -3;
    return 0;
}

static __device__ __forceinline__ uint64_t SparkBenchSm1xxBlockScaleOffset(uint32_t expert_slot, uint32_t row, uint32_t k_group, uint32_t row_extent, uint32_t k_extent)
{
    uint32_t row_block_count = SparkGlm52SotaCeilDivU32(row_extent, 128u);
    uint32_t k_block_count = SparkGlm52SotaCeilDivU32(k_extent, 64u);
    uint32_t row_block = row >> 7u;
    uint32_t row_lane = row & 31u;
    uint32_t row_quad = (row & 127u) >> 5u;
    uint32_t k_block = k_group >> 2u;
    uint32_t k_in_block = k_group & 3u;

    return (((((uint64_t)expert_slot * k_block_count + k_block) * row_block_count + row_block) * 512u) + (row_lane * 16u) + (row_quad * 4u) + k_in_block);
}

static __global__ __launch_bounds__(32, 8)
void SparkBenchSiluMulRequantRouteMajorKernel(
    const __nv_bfloat16 *gate_up_bf16,
    uint8_t *intermediate_payload,
    uint8_t *intermediate_scales,
    uint32_t groups,
    uint32_t route_capacity)
{
    uint32_t group_index = blockIdx.x;
    uint32_t route_index = blockIdx.y;
    uint32_t expert_slot = blockIdx.z;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t column_base = group_index * SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE;
    uint64_t row = ((uint64_t)expert_slot * route_capacity) + route_index;
    float first = 0.0f;
    float second = 0.0f;
    float maximum_value;
    float scale;
    uint8_t scale_code;

    if (expert_slot >= groups || route_index >= route_capacity || group_index >= SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_MOE)
        return;
    if (lane < 8u)
    {
        uint32_t column = column_base + (lane * 2u);
        uint64_t base = (row * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u)) + column;
        float gate0 = SparkGlm52SotaBf16ToFloat(gate_up_bf16[base]);
        float gate1 = SparkGlm52SotaBf16ToFloat(gate_up_bf16[base + 1u]);
        float up0 = SparkGlm52SotaBf16ToFloat(gate_up_bf16[base + SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION]);
        float up1 = SparkGlm52SotaBf16ToFloat(gate_up_bf16[base + SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION + 1u]);
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
        uint64_t scale_offset = SparkBenchSm1xxBlockScaleOffset(expert_slot, route_index, group_index, route_capacity, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION);
        intermediate_scales[scale_offset] = scale_code;
    }
    if (lane < 8u)
    {
        uint64_t packed_index = (row * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION / 2u)) + ((uint64_t)(column_base + lane * 2u) >> 1u);
        SparkGlm52SotaStoreNvfp4Pair(intermediate_payload, packed_index, SparkGlm52SotaEncodeE2m1(first / scale), SparkGlm52SotaEncodeE2m1(second / scale));
    }
}

static __global__ __launch_bounds__(256, 2)
void SparkBenchWeightedCombineRouteMajorKernel(
    const __nv_bfloat16 *down_bf16,
    __nv_bfloat16 *combined_bf16,
    uint32_t tokens,
    uint32_t groups,
    uint32_t route_capacity)
{
    uint32_t token_index = blockIdx.y;
    uint32_t hidden_index = blockIdx.x * 256u + threadIdx.x;
    float accumulator = 0.0f;

    if (token_index >= tokens || hidden_index >= SPARK_GLM52_SOTA_HIDDEN_DIMENSION)
        return;
    #pragma unroll
    for (uint32_t topk_index=0u; topk_index<SPARK_GLM52_SOTA_TOP_K; topk_index++)
    {
        uint32_t route_index = (token_index * SPARK_GLM52_SOTA_TOP_K) + topk_index;
        uint32_t expert_slot = route_index % groups;
        uint32_t local_slot = route_index / groups;
        if (local_slot < route_capacity)
        {
            uint64_t row = ((uint64_t)expert_slot * route_capacity) + local_slot;
            float value = SparkGlm52SotaBf16ToFloat(down_bf16[(row * SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + hidden_index]);
            accumulator = fmaf(value, 1.0f / (float)SPARK_GLM52_SOTA_TOP_K, accumulator);
        }
    }
    combined_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_HIDDEN_DIMENSION) + hidden_index] = SparkGlm52SotaFloatToBf16(accumulator);
}

#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
namespace spark_glm52_route_major_cutlass
{
using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag *, AlignmentC, ElementD, LayoutDTag *, AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA, LayoutATag *, AlignmentA, ElementB, LayoutBTag *, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using GemmElementA = typename Gemm::ElementA;
using GemmElementB = typename Gemm::ElementB;
using GemmElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

struct RouteMajorGemmState
{
    Gemm gemm;
    typename Gemm::Arguments arguments;
    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
    typename ProblemShape::UnderlyingProblemShape *problem_sizes_device;
    const GemmElementA **ptr_a_device;
    const GemmElementB **ptr_b_device;
    const GemmElementSF **ptr_sfa_device;
    const GemmElementSF **ptr_sfb_device;
    const ElementC **ptr_c_device;
    ElementD **ptr_d_device;
    StrideA *stride_a_device;
    StrideB *stride_b_device;
    StrideC *stride_c_device;
    StrideD *stride_d_device;
    LayoutSFA *layout_sfa_device;
    LayoutSFB *layout_sfb_device;
};

template <class T>
static cudaError_t SparkBenchCopyVector(T **device_pointer, const std::vector<T> &host_values)
{
    cudaError_t error;

    *device_pointer = 0;
    if (host_values.empty())
        return cudaErrorInvalidValue;
    error = cudaMalloc(reinterpret_cast<void **>(device_pointer), host_values.size() * sizeof(T));
    if (error != cudaSuccess)
        return error;
    return cudaMemcpy(*device_pointer, host_values.data(), host_values.size() * sizeof(T), cudaMemcpyHostToDevice);
}

static cudaError_t SparkBenchInitRouteMajorGemm(
    RouteMajorGemmState *state,
    uint32_t groups,
    uint32_t m_routes,
    uint32_t n_output,
    uint32_t k_input,
    const uint8_t *a_payload,
    const uint8_t *a_scales,
    const uint8_t *b_payload,
    const uint8_t *b_scales,
    const __nv_bfloat16 *c_bf16,
    __nv_bfloat16 *d_bf16,
    void *workspace,
    uint64_t workspace_bytes)
{
    std::vector<const GemmElementA *> ptr_a_host;
    std::vector<const GemmElementB *> ptr_b_host;
    std::vector<const GemmElementSF *> ptr_sfa_host;
    std::vector<const GemmElementSF *> ptr_sfb_host;
    std::vector<const ElementC *> ptr_c_host;
    std::vector<ElementD *> ptr_d_host;
    std::vector<StrideA> stride_a_host;
    std::vector<StrideB> stride_b_host;
    std::vector<StrideC> stride_c_host;
    std::vector<StrideD> stride_d_host;
    std::vector<LayoutSFA> layout_sfa_host;
    std::vector<LayoutSFB> layout_sfb_host;
    cutlass::KernelHardwareInfo hardware_info;
    cutlass::Status status;
    ProblemShape problem_shape;
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    decltype(state->arguments.epilogue.thread) fusion_args;
    cudaError_t error;
    uint64_t a_payload_bytes = ((uint64_t)m_routes * k_input) >> 1u;
    uint64_t b_payload_bytes = ((uint64_t)n_output * k_input) >> 1u;
    uint64_t c_elements = (uint64_t)m_routes * n_output;
    uint64_t sfa_bytes = SparkBenchScaleBytesOne(m_routes, k_input);
    uint64_t sfb_bytes = SparkBenchScaleBytesOne(n_output, k_input);
    uint32_t group_index;

    state = new (state) RouteMajorGemmState();
    state->problem_sizes_host.resize(groups);
    ptr_a_host.resize(groups);
    ptr_b_host.resize(groups);
    ptr_sfa_host.resize(groups);
    ptr_sfb_host.resize(groups);
    ptr_c_host.resize(groups);
    ptr_d_host.resize(groups);
    stride_a_host.resize(groups);
    stride_b_host.resize(groups);
    stride_c_host.resize(groups);
    stride_d_host.resize(groups);
    layout_sfa_host.resize(groups);
    layout_sfb_host.resize(groups);
    for (group_index=0u; group_index<groups; group_index++)
    {
        state->problem_sizes_host[group_index] = {static_cast<int>(m_routes), static_cast<int>(n_output), static_cast<int>(k_input)};
        ptr_a_host[group_index] = reinterpret_cast<const GemmElementA *>(a_payload + ((uint64_t)group_index * a_payload_bytes));
        ptr_b_host[group_index] = reinterpret_cast<const GemmElementB *>(b_payload + ((uint64_t)group_index * b_payload_bytes));
        ptr_sfa_host[group_index] = reinterpret_cast<const GemmElementSF *>(a_scales + ((uint64_t)group_index * sfa_bytes));
        ptr_sfb_host[group_index] = reinterpret_cast<const GemmElementSF *>(b_scales + ((uint64_t)group_index * sfb_bytes));
        ptr_c_host[group_index] = reinterpret_cast<const ElementC *>(c_bf16 + ((uint64_t)group_index * c_elements));
        ptr_d_host[group_index] = reinterpret_cast<ElementD *>(d_bf16 + ((uint64_t)group_index * c_elements));
        stride_a_host[group_index] = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(m_routes), static_cast<int>(k_input), 1});
        stride_b_host[group_index] = cutlass::make_cute_packed_stride(StrideB{}, {static_cast<int>(n_output), static_cast<int>(k_input), 1});
        stride_c_host[group_index] = cutlass::make_cute_packed_stride(StrideC{}, {static_cast<int>(m_routes), static_cast<int>(n_output), 1});
        stride_d_host[group_index] = cutlass::make_cute_packed_stride(StrideD{}, {static_cast<int>(m_routes), static_cast<int>(n_output), 1});
        layout_sfa_host[group_index] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(static_cast<int>(m_routes), static_cast<int>(n_output), static_cast<int>(k_input), 1));
        layout_sfb_host[group_index] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(static_cast<int>(m_routes), static_cast<int>(n_output), static_cast<int>(k_input), 1));
    }
    error = SparkBenchCopyVector(&state->problem_sizes_device, state->problem_sizes_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->ptr_a_device, ptr_a_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->ptr_b_device, ptr_b_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->ptr_sfa_device, ptr_sfa_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->ptr_sfb_device, ptr_sfb_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->ptr_c_device, ptr_c_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->ptr_d_device, ptr_d_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->stride_a_device, stride_a_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->stride_b_device, stride_b_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->stride_c_device, stride_c_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->stride_d_device, stride_d_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->layout_sfa_device, layout_sfa_host);
    if (error != cudaSuccess)
        return error;
    error = SparkBenchCopyVector(&state->layout_sfb_device, layout_sfb_host);
    if (error != cudaSuccess)
        return error;
    hardware_info.device_id = 0;
    hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hardware_info.device_id);
    hardware_info.cluster_shape = dim3(1u, 1u, 1u);
    hardware_info.cluster_shape_fallback = dim3(1u, 1u, 1u);
    problem_shape = ProblemShape{static_cast<int>(groups), state->problem_sizes_device, state->problem_sizes_host.data()};
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    state->arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        problem_shape,
        {
            state->ptr_a_device,
            state->stride_a_device,
            state->ptr_b_device,
            state->stride_b_device,
            state->ptr_sfa_device,
            state->layout_sfa_device,
            state->ptr_sfb_device,
            state->layout_sfb_device
        },
        {
            fusion_args,
            state->ptr_c_device,
            state->stride_c_device,
            state->ptr_d_device,
            state->stride_d_device
        },
        hardware_info,
        scheduler
    };
    status = state->gemm.can_implement(state->arguments);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;
    if (Gemm::get_workspace_size(state->arguments) > workspace_bytes)
        return cudaErrorMemoryAllocation;
    status = state->gemm.initialize(state->arguments, workspace);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorInvalidValue;
}

static cudaError_t SparkBenchRunRouteMajorGemm(RouteMajorGemmState *state, cudaStream_t stream)
{
    cutlass::Status status = state->gemm.run(stream, nullptr, false);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}
}
#endif

int main(int argc, char **argv)
{
#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
    using namespace spark_glm52_route_major_cutlass;
    uint32_t tokens = 96u;
    uint32_t groups = SPARK_GLM52_SOTA_EXPERT_COUNT;
    uint32_t route_capacity = 3u;
    uint32_t warmup = 5u;
    uint32_t iterations = 20u;
    uint32_t workspace_mb = 1024u;
    uint32_t routes;
    uint64_t route_rows;
    uint64_t workspace_bytes;
    cudaStream_t stream;
    RouteMajorGemmState *gate_state;
    RouteMajorGemmState *down_state;
    void *workspace = 0;
    uint8_t *gate_a = 0;
    uint8_t *gate_sfa = 0;
    uint8_t *gate_b = 0;
    uint8_t *gate_sfb = 0;
    __nv_bfloat16 *gate_c = 0;
    __nv_bfloat16 *gate_d = 0;
    uint8_t *intermediate = 0;
    uint8_t *intermediate_sfa = 0;
    uint8_t *down_b = 0;
    uint8_t *down_sfb = 0;
    __nv_bfloat16 *down_c = 0;
    __nv_bfloat16 *down_d = 0;
    __nv_bfloat16 *combined = 0;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float elapsed_ms;
    double gate_flops;
    double down_flops;
    double total_flops;
    double tflops;
    uint32_t i;

    SparkBenchParseU32(argc, argv, "--tokens", &tokens);
    SparkBenchParseU32(argc, argv, "--groups", &groups);
    SparkBenchParseU32(argc, argv, "--capacity", &route_capacity);
    SparkBenchParseU32(argc, argv, "--warmup", &warmup);
    SparkBenchParseU32(argc, argv, "--iterations", &iterations);
    SparkBenchParseU32(argc, argv, "--workspace-mb", &workspace_mb);
    routes = tokens * SPARK_GLM52_SOTA_TOP_K;
    route_rows = (uint64_t)groups * route_capacity;
    if (tokens == 0u || groups == 0u || groups > SPARK_GLM52_SOTA_EXPERT_COUNT || route_capacity == 0u || iterations == 0u)
    {
        fprintf(stderr, "invalid route-major benchmark shape\n");
        return 2;
    }
    if (routes > route_rows)
    {
        fprintf(stderr, "route capacity too small: routes=%u capacity=%llu\n", routes, (unsigned long long)route_rows);
        return 3;
    }
    workspace_bytes = (uint64_t)workspace_mb * 1024u * 1024u;
    gate_state = static_cast<RouteMajorGemmState *>(malloc(sizeof(RouteMajorGemmState)));
    down_state = static_cast<RouteMajorGemmState *>(malloc(sizeof(RouteMajorGemmState)));
    if (gate_state == 0 || down_state == 0)
        return 4;
    if (SparkBenchCuda(cudaStreamCreate(&stream), "cudaStreamCreate") < 0)
        return 5;
    if (SparkBenchDeviceAlloc(&workspace, workspace_bytes, 0, "cudaMalloc workspace") < 0)
        return 6;
    if (SparkBenchDeviceAlloc((void **)&gate_a, route_rows * (SPARK_GLM52_SOTA_HIDDEN_DIMENSION / 2u), 0x11, "cudaMalloc gate A activations") < 0)
        return 7;
    if (SparkBenchDeviceAlloc((void **)&gate_sfa, SparkBenchScaleBytes(groups, route_capacity, SPARK_GLM52_SOTA_HIDDEN_DIMENSION), 0x38, "cudaMalloc gate A scales") < 0)
        return 8;
    if (SparkBenchDeviceAlloc((void **)&gate_b, (uint64_t)groups * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * SPARK_GLM52_SOTA_HIDDEN_DIMENSION / 2u, 0x11, "cudaMalloc gate B weights") < 0)
        return 9;
    if (SparkBenchDeviceAlloc((void **)&gate_sfb, SparkBenchScaleBytes(groups, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u, SPARK_GLM52_SOTA_HIDDEN_DIMENSION), 0x38, "cudaMalloc gate B scales") < 0)
        return 10;
    if (SparkBenchDeviceAlloc((void **)&gate_c, route_rows * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * sizeof(__nv_bfloat16), 0, "cudaMalloc gate C") < 0)
        return 11;
    if (SparkBenchDeviceAlloc((void **)&gate_d, route_rows * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * sizeof(__nv_bfloat16), 0, "cudaMalloc gate D") < 0)
        return 12;
    if (SparkBenchDeviceAlloc((void **)&intermediate, route_rows * (SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION / 2u), 0, "cudaMalloc intermediate") < 0)
        return 13;
    if (SparkBenchDeviceAlloc((void **)&intermediate_sfa, SparkBenchScaleBytes(groups, route_capacity, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION), 0x38, "cudaMalloc intermediate scales") < 0)
        return 14;
    if (SparkBenchDeviceAlloc((void **)&down_b, (uint64_t)groups * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION / 2u, 0x11, "cudaMalloc down B weights") < 0)
        return 15;
    if (SparkBenchDeviceAlloc((void **)&down_sfb, SparkBenchScaleBytes(groups, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION), 0x38, "cudaMalloc down B scales") < 0)
        return 16;
    if (SparkBenchDeviceAlloc((void **)&down_c, route_rows * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * sizeof(__nv_bfloat16), 0, "cudaMalloc down C") < 0)
        return 17;
    if (SparkBenchDeviceAlloc((void **)&down_d, route_rows * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * sizeof(__nv_bfloat16), 0, "cudaMalloc down D") < 0)
        return 18;
    if (SparkBenchDeviceAlloc((void **)&combined, (uint64_t)tokens * SPARK_GLM52_SOTA_HIDDEN_DIMENSION * sizeof(__nv_bfloat16), 0, "cudaMalloc combined") < 0)
        return 19;
    if (SparkBenchCuda(SparkBenchInitRouteMajorGemm(gate_state, groups, route_capacity, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, gate_a, gate_sfa, gate_b, gate_sfb, gate_c, gate_d, workspace, workspace_bytes), "init route-major gate/up CUTLASS") < 0)
        return 20;
    if (SparkBenchCuda(SparkBenchInitRouteMajorGemm(down_state, groups, route_capacity, SPARK_GLM52_SOTA_HIDDEN_DIMENSION, SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION, intermediate, intermediate_sfa, down_b, down_sfb, down_c, down_d, workspace, workspace_bytes), "init route-major down CUTLASS") < 0)
        return 21;
    if (SparkBenchCuda(cudaEventCreate(&start_event), "cudaEventCreate start") < 0)
        return 22;
    if (SparkBenchCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop") < 0)
        return 23;
    for (i=0u; i<warmup; i++)
    {
        if (SparkBenchCuda(SparkBenchRunRouteMajorGemm(gate_state, stream), "warmup gate/up") < 0)
            return 24;
        SparkBenchSiluMulRequantRouteMajorKernel<<<dim3(SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_MOE, route_capacity, groups), 32u, 0u, stream>>>(gate_d, intermediate, intermediate_sfa, groups, route_capacity);
        if (SparkBenchCuda(SparkBenchRunRouteMajorGemm(down_state, stream), "warmup down") < 0)
            return 25;
        SparkBenchWeightedCombineRouteMajorKernel<<<dim3(SparkGlm52SotaCeilDivU32(SPARK_GLM52_SOTA_HIDDEN_DIMENSION, 256u), tokens, 1u), 256u, 0u, stream>>>(down_d, combined, tokens, groups, route_capacity);
    }
    if (SparkBenchCuda(cudaStreamSynchronize(stream), "warmup sync") < 0)
        return 26;
    if (SparkBenchCuda(cudaEventRecord(start_event, stream), "cudaEventRecord start") < 0)
        return 27;
    for (i=0u; i<iterations; i++)
    {
        if (SparkBenchCuda(SparkBenchRunRouteMajorGemm(gate_state, stream), "bench gate/up") < 0)
            return 28;
        SparkBenchSiluMulRequantRouteMajorKernel<<<dim3(SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_MOE, route_capacity, groups), 32u, 0u, stream>>>(gate_d, intermediate, intermediate_sfa, groups, route_capacity);
        if (SparkBenchCuda(SparkBenchRunRouteMajorGemm(down_state, stream), "bench down") < 0)
            return 29;
        SparkBenchWeightedCombineRouteMajorKernel<<<dim3(SparkGlm52SotaCeilDivU32(SPARK_GLM52_SOTA_HIDDEN_DIMENSION, 256u), tokens, 1u), 256u, 0u, stream>>>(down_d, combined, tokens, groups, route_capacity);
    }
    if (SparkBenchCuda(cudaEventRecord(stop_event, stream), "cudaEventRecord stop") < 0)
        return 30;
    if (SparkBenchCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop") < 0)
        return 31;
    if (SparkBenchCuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime") < 0)
        return 32;
    gate_flops = 2.0 * (double)groups * (double)route_capacity * (double)(SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION * 2u) * (double)SPARK_GLM52_SOTA_HIDDEN_DIMENSION;
    down_flops = 2.0 * (double)groups * (double)route_capacity * (double)SPARK_GLM52_SOTA_HIDDEN_DIMENSION * (double)SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION;
    total_flops = gate_flops + down_flops;
    tflops = (total_flops * (double)iterations) / ((double)elapsed_ms * 1.0e9);
    printf("cutlass_route_major_moe_ready=1 active_tokens=%u active_routes=%u bound_experts=%u route_capacity_per_expert=%u capacity_routes=%llu iterations=%u total_ms=%.3f avg_us=%.3f estimated_tflops=%.3f orientation=activations_as_a_weights_as_b output=row_major\n",
        tokens,
        routes,
        groups,
        route_capacity,
        (unsigned long long)route_rows,
        iterations,
        elapsed_ms,
        (elapsed_ms * 1000.0f) / (float)iterations,
        tflops);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    return 0;
#else
    (void)argc;
    (void)argv;
    fprintf(stderr, "CUTLASS NVFP4 support is not compiled in\n");
    return 4;
#endif
}
