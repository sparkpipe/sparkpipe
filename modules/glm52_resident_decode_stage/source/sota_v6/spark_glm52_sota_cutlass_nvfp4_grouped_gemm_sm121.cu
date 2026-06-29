#include "spark_glm52_sota_production_plan_sm121.cuh"

#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#endif

#if defined(SPARK_GLM52_ENABLE_CUTLASS_NVFP4_SM121)
namespace spark_glm52_sota_cutlass_nvfp4_sm121
{
using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::nv_float4_t;
using ElementB = cutlass::nv_float4_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, ElementC, LayoutC *, AlignmentC, ElementD, LayoutD *, AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

struct CachedArguments
{
    Gemm gemm;
    typename Gemm::Arguments arguments;
};
}
#endif

static cudaError_t SparkGlm52SotaValidateCutlassGroupedPlan(const SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan)
{
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI || plan->problem_count == 0u || plan->problem_count > plan->maximum_problem_count || plan->problems_device == 0 || plan->workspace == 0 || plan->workspace_bytes == 0u)
    {
        return cudaErrorInvalidValue;
    }
    if (plan->expected_group_size != SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE || (plan->capability_flags & SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT) == 0u)
    {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
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
    cached_arguments->arguments.problem_shape.problem_count = static_cast<int>(plan->problem_count);
    status = cached_arguments->gemm.run(cached_arguments->arguments, stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
#else
    (void)plan;
    (void)stream;
    return cudaErrorNotSupported;
#endif
}
