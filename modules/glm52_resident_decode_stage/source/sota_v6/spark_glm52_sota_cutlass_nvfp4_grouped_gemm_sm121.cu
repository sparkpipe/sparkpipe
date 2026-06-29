#include "spark_glm52_sota_production_plan_sm121.cuh"

#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#include <new>
#include <vector>
#endif

#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
namespace spark_glm52_sota_cutlass_nvfp4_sm121
{
using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementCompute = float;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::ColumnMajor;
using LayoutDTag = cutlass::layout::ColumnMajor;
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

struct CachedArguments
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
static cudaError_t SparkGlm52SotaCudaCopyVector(T **device_pointer, const std::vector<T> &host_values)
{
    cudaError_t error;

    *device_pointer = 0;
    if (host_values.empty())
    {
        return cudaErrorInvalidValue;
    }
    error = cudaMalloc(reinterpret_cast<void **>(device_pointer), host_values.size() * sizeof(T));
    if (error != cudaSuccess)
    {
        return error;
    }
    return cudaMemcpy(*device_pointer, host_values.data(), host_values.size() * sizeof(T), cudaMemcpyHostToDevice);
}

static uint64_t SparkGlm52SotaScaleBytes(uint32_t row_extent, uint32_t k_extent)
{
    return ((uint64_t)SparkGlm52SotaCeilDivU32(row_extent, 128u) * SparkGlm52SotaCeilDivU32(k_extent, 64u) * 512u);
}

static cudaError_t SparkGlm52SotaBuildSm121GroupedState(SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan, CachedArguments *cached_arguments)
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
    cudaError_t error;
    uint32_t group_index;
    uint32_t m = plan->cutlass_m;
    uint32_t n = plan->cutlass_n_capacity;
    uint32_t k = plan->cutlass_k;
    uint64_t a_payload_bytes = ((uint64_t)m * k) >> 1u;
    uint64_t b_payload_bytes = ((uint64_t)n * k) >> 1u;
    uint64_t c_elements = (uint64_t)m * n;
    uint64_t sfa_bytes = SparkGlm52SotaScaleBytes(m, k);
    uint64_t sfb_bytes = SparkGlm52SotaScaleBytes(n, k);

    cached_arguments->problem_sizes_host.resize(plan->cutlass_group_count);
    ptr_a_host.resize(plan->cutlass_group_count);
    ptr_b_host.resize(plan->cutlass_group_count);
    ptr_sfa_host.resize(plan->cutlass_group_count);
    ptr_sfb_host.resize(plan->cutlass_group_count);
    ptr_c_host.resize(plan->cutlass_group_count);
    ptr_d_host.resize(plan->cutlass_group_count);
    stride_a_host.resize(plan->cutlass_group_count);
    stride_b_host.resize(plan->cutlass_group_count);
    stride_c_host.resize(plan->cutlass_group_count);
    stride_d_host.resize(plan->cutlass_group_count);
    layout_sfa_host.resize(plan->cutlass_group_count);
    layout_sfb_host.resize(plan->cutlass_group_count);
    for (group_index = 0u; group_index < plan->cutlass_group_count; ++group_index)
    {
        cached_arguments->problem_sizes_host[group_index] = {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
        ptr_a_host[group_index] = reinterpret_cast<const GemmElementA *>(plan->cutlass_a_payload_u8 + ((uint64_t)group_index * a_payload_bytes));
        ptr_b_host[group_index] = reinterpret_cast<const GemmElementB *>(plan->cutlass_b_payload_u8 + ((uint64_t)group_index * b_payload_bytes));
        ptr_sfa_host[group_index] = reinterpret_cast<const GemmElementSF *>(plan->cutlass_a_scale_ue4m3_u8 + ((uint64_t)group_index * sfa_bytes));
        ptr_sfb_host[group_index] = reinterpret_cast<const GemmElementSF *>(plan->cutlass_b_scale_ue4m3_u8 + ((uint64_t)group_index * sfb_bytes));
        ptr_c_host[group_index] = reinterpret_cast<const ElementC *>(static_cast<const uint8_t *>(plan->cutlass_c_bf16) + ((uint64_t)group_index * c_elements * sizeof(ElementC)));
        ptr_d_host[group_index] = reinterpret_cast<ElementD *>(static_cast<uint8_t *>(plan->cutlass_d_bf16) + ((uint64_t)group_index * c_elements * sizeof(ElementD)));
        stride_a_host[group_index] = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(m), static_cast<int>(k), 1});
        stride_b_host[group_index] = cutlass::make_cute_packed_stride(StrideB{}, {static_cast<int>(n), static_cast<int>(k), 1});
        stride_c_host[group_index] = cutlass::make_cute_packed_stride(StrideC{}, {static_cast<int>(m), static_cast<int>(n), 1});
        stride_d_host[group_index] = cutlass::make_cute_packed_stride(StrideD{}, {static_cast<int>(m), static_cast<int>(n), 1});
        layout_sfa_host[group_index] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
        layout_sfb_host[group_index] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
    }
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->problem_sizes_device, cached_arguments->problem_sizes_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->ptr_a_device, ptr_a_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->ptr_b_device, ptr_b_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->ptr_sfa_device, ptr_sfa_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->ptr_sfb_device, ptr_sfb_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->ptr_c_device, ptr_c_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->ptr_d_device, ptr_d_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->stride_a_device, stride_a_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->stride_b_device, stride_b_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->stride_c_device, stride_c_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->stride_d_device, stride_d_host);
    if (error != cudaSuccess)
        return error;
    error = SparkGlm52SotaCudaCopyVector(&cached_arguments->layout_sfa_device, layout_sfa_host);
    if (error != cudaSuccess)
        return error;
    return SparkGlm52SotaCudaCopyVector(&cached_arguments->layout_sfb_device, layout_sfb_host);
}
}
#endif

static cudaError_t SparkGlm52SotaValidateCutlassGroupedPlan(const SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan)
{
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI || plan->workspace == 0 || plan->workspace_bytes == 0u || plan->cutlass_or_cublas_state == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->expected_group_size != SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE || (plan->capability_flags & SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT) == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->cutlass_m == 0u || plan->cutlass_n_capacity == 0u || plan->cutlass_k == 0u || plan->cutlass_group_count == 0u || plan->tokens_per_expert_device == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->cutlass_a_payload_u8 == 0 || plan->cutlass_a_scale_ue4m3_u8 == 0 || plan->cutlass_b_payload_u8 == 0 || plan->cutlass_b_scale_ue4m3_u8 == 0 || plan->cutlass_c_bf16 == 0 || plan->cutlass_d_bf16 == 0)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->expected_k != plan->cutlass_k || plan->expected_n != plan->cutlass_m || plan->problem_count != plan->cutlass_group_count || plan->problem_count > plan->maximum_problem_count)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

extern "C" uint64_t SparkGlm52SotaCutlassNvfp4GroupedGemmStateBytesSm121(void)
{
#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
    return sizeof(spark_glm52_sota_cutlass_nvfp4_sm121::CachedArguments);
#else
    return 0u;
#endif
}

extern "C" cudaError_t SparkGlm52SotaInitializeCutlassNvfp4GroupedGemmSm121(SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan, void *state_memory, uint64_t state_memory_bytes, cudaStream_t stream)
{
#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
    using namespace spark_glm52_sota_cutlass_nvfp4_sm121;
    CachedArguments *cached_arguments;
    cutlass::KernelHardwareInfo hardware_info;
    cutlass::Status status;
    ProblemShape problem_shape;
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    decltype(cached_arguments->arguments.epilogue.thread) fusion_args;

    if (state_memory == 0 || state_memory_bytes < sizeof(CachedArguments))
    {
        return cudaErrorInvalidValue;
    }
    if (plan->problem_count == 0u)
    {
        plan->problem_count = plan->cutlass_group_count;
    }
    if (plan->maximum_problem_count == 0u)
    {
        plan->maximum_problem_count = plan->cutlass_group_count;
    }
    plan->cutlass_or_cublas_state = state_memory;
    cached_arguments = new (state_memory) CachedArguments();
    if (SparkGlm52SotaValidateCutlassGroupedPlan(plan) != cudaSuccess)
    {
        plan->cutlass_or_cublas_state = 0;
        return cudaErrorInvalidValue;
    }
    hardware_info.device_id = 0;
    hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hardware_info.device_id);
    hardware_info.cluster_shape = dim3(1u, 1u, 1u);
    hardware_info.cluster_shape_fallback = dim3(1u, 1u, 1u);
    if (SparkGlm52SotaBuildSm121GroupedState(plan, cached_arguments) != cudaSuccess)
    {
        plan->cutlass_or_cublas_state = 0;
        return cudaErrorMemoryAllocation;
    }
    problem_shape = ProblemShape{
        static_cast<int>(plan->cutlass_group_count),
        cached_arguments->problem_sizes_device,
        cached_arguments->problem_sizes_host.data()};
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    cached_arguments->arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        problem_shape,
        {
            cached_arguments->ptr_a_device,
            cached_arguments->stride_a_device,
            cached_arguments->ptr_b_device,
            cached_arguments->stride_b_device,
            cached_arguments->ptr_sfa_device,
            cached_arguments->layout_sfa_device,
            cached_arguments->ptr_sfb_device,
            cached_arguments->layout_sfb_device
        },
        {
            fusion_args,
            cached_arguments->ptr_c_device,
            cached_arguments->stride_c_device,
            cached_arguments->ptr_d_device,
            cached_arguments->stride_d_device
        },
        hardware_info,
        scheduler
    };
    status = cached_arguments->gemm.can_implement(cached_arguments->arguments);
    if (status != cutlass::Status::kSuccess)
    {
        plan->cutlass_or_cublas_state = 0;
        return cudaErrorNotSupported;
    }
    if (Gemm::get_workspace_size(cached_arguments->arguments) > plan->workspace_bytes)
    {
        plan->cutlass_or_cublas_state = 0;
        return cudaErrorMemoryAllocation;
    }
    status = cached_arguments->gemm.initialize(cached_arguments->arguments, plan->workspace);
    if (status != cutlass::Status::kSuccess)
    {
        plan->cutlass_or_cublas_state = 0;
        return cudaErrorInvalidValue;
    }
    plan->launch = SparkGlm52SotaLaunchCutlassNvfp4GroupedGemmSm121;
    (void)stream;
    return cudaSuccess;
#else
    (void)plan;
    (void)state_memory;
    (void)state_memory_bytes;
    return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t SparkGlm52SotaLaunchCutlassNvfp4GroupedGemmSm121(SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan, cudaStream_t stream)
{
#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
    using namespace spark_glm52_sota_cutlass_nvfp4_sm121;
    CachedArguments *cached_arguments;
    cutlass::Status status;
    if (SparkGlm52SotaValidateCutlassGroupedPlan(plan) != cudaSuccess)
    {
        return cudaErrorInvalidValue;
    }
    cached_arguments = static_cast<CachedArguments *>(plan->cutlass_or_cublas_state);
    if (cached_arguments == 0)
    {
        return cudaErrorInvalidValue;
    }
    status = cached_arguments->gemm.run(stream, nullptr, false);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
#else
    (void)plan;
    (void)stream;
    return cudaErrorNotSupported;
#endif
}
