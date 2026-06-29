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
#include "cute/tensor.hpp"
#include <new>
#endif

#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
namespace spark_glm52_sota_cutlass_nvfp4_sm121
{
using namespace cute;
using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;
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
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _64, _256>;
using ClusterShape = Shape<_1, _1, _1>;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto, cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

struct CachedArguments
{
    Gemm gemm;
    typename Gemm::Arguments arguments;
};

template <class StrideIntT>
static cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> SparkGlm52SotaMakePackedStride(cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> stride, cute::Shape<int, int, int> shape_mkl)
{
    cute::get<0>(stride) = static_cast<StrideIntT>(cute::get<1>(shape_mkl));
    return stride;
}

template <class StrideIntT>
static cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> SparkGlm52SotaMakePackedStride(cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> stride, cute::Shape<int, int, int> shape_mkl)
{
    cute::get<1>(stride) = static_cast<StrideIntT>(cute::get<0>(shape_mkl));
    return stride;
}

template <class StrideIntT>
static cute::Stride<StrideIntT, cute::Int<1>, int64_t> SparkGlm52SotaMakePackedStride(cute::Stride<StrideIntT, cute::Int<1>, int64_t> stride, cute::Shape<int, int, int> shape_mkl)
{
    cute::get<0>(stride) = static_cast<StrideIntT>(cute::get<1>(shape_mkl));
    cute::get<2>(stride) = cute::get<2>(shape_mkl) > 1 ? static_cast<int64_t>(cute::get<0>(shape_mkl) * cute::get<1>(shape_mkl)) : 0;
    return stride;
}

template <class StrideIntT>
static cute::Stride<cute::Int<1>, StrideIntT, int64_t> SparkGlm52SotaMakePackedStride(cute::Stride<cute::Int<1>, StrideIntT, int64_t> stride, cute::Shape<int, int, int> shape_mkl)
{
    cute::get<1>(stride) = static_cast<StrideIntT>(cute::get<0>(shape_mkl));
    cute::get<2>(stride) = cute::get<2>(shape_mkl) > 1 ? static_cast<int64_t>(cute::get<0>(shape_mkl) * cute::get<1>(shape_mkl)) : 0;
    return stride;
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
    StrideC stride_c;
    StrideD stride_d;

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
    problem_shape = ProblemShape{
        static_cast<int>(plan->cutlass_m),
        static_cast<int>(plan->cutlass_n_capacity),
        static_cast<int>(plan->cutlass_k),
        static_cast<int>(plan->cutlass_group_count),
        plan->tokens_per_expert_device,
        plan->tokens_per_expert_host};
    stride_c = SparkGlm52SotaMakePackedStride(StrideC{}, {static_cast<int>(plan->cutlass_m), static_cast<int>(plan->cutlass_n_capacity), static_cast<int>(plan->cutlass_group_count)});
    stride_d = SparkGlm52SotaMakePackedStride(StrideD{}, {static_cast<int>(plan->cutlass_m), static_cast<int>(plan->cutlass_n_capacity), static_cast<int>(plan->cutlass_group_count)});
    cached_arguments->arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        problem_shape,
        {
            reinterpret_cast<ElementA::DataType *>(const_cast<uint8_t *>(plan->cutlass_a_payload_u8)),
            reinterpret_cast<ElementB::DataType *>(const_cast<uint8_t *>(plan->cutlass_b_payload_u8)),
            reinterpret_cast<ElementA::ScaleFactorType *>(const_cast<uint8_t *>(plan->cutlass_a_scale_ue4m3_u8)),
            reinterpret_cast<ElementB::ScaleFactorType *>(const_cast<uint8_t *>(plan->cutlass_b_scale_ue4m3_u8))
        },
        {
            {},
            reinterpret_cast<ElementC *>(const_cast<void *>(plan->cutlass_c_bf16)),
            stride_c,
            reinterpret_cast<ElementD *>(plan->cutlass_d_bf16),
            stride_d
        },
        hardware_info
    };
    cached_arguments->arguments.epilogue.thread.alpha = 1.0f;
    cached_arguments->arguments.epilogue.thread.beta = 0.0f;
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
    status = cached_arguments->gemm.initialize(cached_arguments->arguments, plan->workspace, stream);
    if (status != cutlass::Status::kSuccess)
    {
        plan->cutlass_or_cublas_state = 0;
        return cudaErrorInvalidValue;
    }
    plan->launch = SparkGlm52SotaLaunchCutlassNvfp4GroupedGemmSm121;
    return cudaSuccess;
#else
    (void)plan;
    (void)state_memory;
    (void)state_memory_bytes;
    (void)stream;
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
    status = cached_arguments->gemm.run(stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
#else
    (void)plan;
    (void)stream;
    return cudaErrorNotSupported;
#endif
}
