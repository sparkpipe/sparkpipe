#ifndef SPARK_GLM52_SOTA_PRODUCTION_PLAN_SM121_CUH
#define SPARK_GLM52_SOTA_PRODUCTION_PLAN_SM121_CUH

#include "spark_glm52_sota_cuda_common.cuh"
#include <cublasLt.h>

#define SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI 6u

#define SPARK_GLM52_SOTA_FAST_CAP_TOKEN_QUANT_ONCE         0x0000000000000001ull
#define SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROUTER_TOPK        0x0000000000000002ull
#define SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP    0x0000000000000004ull
#define SPARK_GLM52_SOTA_FAST_CAP_FUSED_SILU_REQUANT       0x0000000000000008ull
#define SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN       0x0000000000000010ull
#define SPARK_GLM52_SOTA_FAST_CAP_WEIGHTED_COMBINE         0x0000000000000020ull
#define SPARK_GLM52_SOTA_FAST_CAP_FLASH_MLA                0x0000000000000040ull
#define SPARK_GLM52_SOTA_FAST_CAP_CUBLASLT_LINEAR          0x0000000000000080ull
#define SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROPE_KV            0x0000000000000100ull
#define SPARK_GLM52_SOTA_FAST_CAP_RESTRICTED_LOGITS_PLAN   0x0000000000000200ull
#define SPARK_GLM52_SOTA_FAST_CAP_MXFP4_MTP_PLAN           0x0000000000000400ull
#define SPARK_GLM52_SOTA_FAST_CAP_GRAPH_REPLAY             0x0000000000000800ull
#define SPARK_GLM52_SOTA_FAST_CAP_NO_HOST_STAGING          0x0000000000001000ull
#define SPARK_GLM52_SOTA_FAST_CAP_NO_DEVICE_MEMCPY         0x0000000000002000ull
#define SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT 0x0000000000004000ull
#define SPARK_GLM52_SOTA_FAST_CAP_FIXED_GLM52_SHAPES       0x0000000000008000ull

#define SPARK_GLM52_SOTA_REQUIRED_FAST_CAPS \
    (SPARK_GLM52_SOTA_FAST_CAP_TOKEN_QUANT_ONCE | \
     SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROUTER_TOPK | \
     SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP | \
     SPARK_GLM52_SOTA_FAST_CAP_FUSED_SILU_REQUANT | \
     SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN | \
     SPARK_GLM52_SOTA_FAST_CAP_WEIGHTED_COMBINE | \
     SPARK_GLM52_SOTA_FAST_CAP_FLASH_MLA | \
     SPARK_GLM52_SOTA_FAST_CAP_CUBLASLT_LINEAR | \
     SPARK_GLM52_SOTA_FAST_CAP_FUSED_ROPE_KV | \
     SPARK_GLM52_SOTA_FAST_CAP_RESTRICTED_LOGITS_PLAN | \
     SPARK_GLM52_SOTA_FAST_CAP_MXFP4_MTP_PLAN | \
     SPARK_GLM52_SOTA_FAST_CAP_GRAPH_REPLAY | \
     SPARK_GLM52_SOTA_FAST_CAP_NO_HOST_STAGING | \
     SPARK_GLM52_SOTA_FAST_CAP_NO_DEVICE_MEMCPY | \
     SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT | \
     SPARK_GLM52_SOTA_FAST_CAP_FIXED_GLM52_SHAPES)

#define SPARK_GLM52_SOTA_GEMM_KIND_GATE_UP 1u
#define SPARK_GLM52_SOTA_GEMM_KIND_DOWN    2u
#define SPARK_GLM52_SOTA_GEMM_KIND_ROUTER  3u
#define SPARK_GLM52_SOTA_GEMM_KIND_LINEAR  4u
#define SPARK_GLM52_SOTA_GEMM_KIND_LOGITS  5u

struct SparkGlm52SotaNvfp4GroupedGemmProblemSm121
{
    const uint8_t *a_payload_u8;
    const uint8_t *a_scale_ue4m3_u8;
    const uint8_t *b_payload_u8;
    const uint8_t *b_scale_ue4m3_u8;
    const void *c;
    void *d;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t expert_slot;
    uint32_t a_packed_row_stride_bytes;
    uint32_t a_scale_row_stride_bytes;
    uint32_t b_packed_row_stride_bytes;
    uint32_t b_scale_row_stride_bytes;
    uint32_t c_row_stride_elements;
    uint32_t d_row_stride_elements_or_bytes;
    float alpha;
    float beta;
};

struct SparkGlm52SotaNvfp4GroupedGemmPlanSm121
{
    uint32_t abi_version;
    uint32_t gemm_kind;
    uint32_t maximum_problem_count;
    uint32_t problem_count;
    uint32_t expected_k;
    uint32_t expected_n;
    uint32_t expected_group_size;
    uint32_t output_is_nvfp4;
    uint64_t capability_flags;
    uint64_t workspace_bytes;
    void *workspace;
    void *cutlass_or_cublas_state;
    uint32_t cutlass_m;
    uint32_t cutlass_n_capacity;
    uint32_t cutlass_k;
    uint32_t cutlass_group_count;
    int32_t *tokens_per_expert_device;
    int32_t *tokens_per_expert_host;
    const uint8_t *cutlass_a_payload_u8;
    const uint8_t *cutlass_a_scale_ue4m3_u8;
    const uint8_t *cutlass_b_payload_u8;
    const uint8_t *cutlass_b_scale_ue4m3_u8;
    const void *cutlass_c_bf16;
    void *cutlass_d_bf16;
    SparkGlm52SotaNvfp4GroupedGemmProblemSm121 *problems_device;
    SparkGlm52SotaNvfp4GroupedGemmProblemSm121 *problems_host_mapped;
    cudaError_t (*launch)(SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan, cudaStream_t stream);
};

struct SparkGlm52SotaBf16LinearPlanSm121
{
    uint32_t abi_version;
    uint32_t gemm_kind;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t ldd;
    uint32_t trans_b;
    uint64_t capability_flags;
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation_desc;
    cublasLtMatrixLayout_t a_layout;
    cublasLtMatrixLayout_t b_layout;
    cublasLtMatrixLayout_t c_layout;
    cublasLtMatrixLayout_t d_layout;
    cublasLtMatmulAlgo_t algo;
    void *workspace;
    uint64_t workspace_bytes;
    const void *a;
    const void *b;
    const void *bias;
    const void *c;
    void *d;
    float alpha;
    float beta;
};

struct SparkGlm52SotaProductionMoePlanSm121
{
    uint32_t abi_version;
    uint32_t maximum_tokens;
    uint32_t maximum_routes;
    uint32_t maximum_bound_experts;
    uint32_t route_tile_size;
    uint64_t capability_flags;
    uint64_t validated_latency_ns;
    SparkGlm52SotaNvfp4GroupedGemmPlanSm121 gate_up_gemm;
    SparkGlm52SotaNvfp4GroupedGemmPlanSm121 down_gemm;
    int32_t *expert_id_to_bound_slot;
    uint32_t *expert_route_counts;
    uint32_t *expert_route_offsets;
    uint32_t *expert_route_cursors;
    uint32_t *route_indices_by_expert;
    uint32_t *grouped_row_by_route;
    uint32_t *problem_count_device;
    uint32_t *overflow_flag;
    SparkGlm52SotaMutableNvfp4MatrixView grouped_hidden_nvfp4_by_route;
    __nv_bfloat16 *gate_up_bf16_by_grouped_route;
    SparkGlm52SotaMutableNvfp4MatrixView intermediate_nvfp4_by_grouped_route;
    __nv_bfloat16 *down_bf16_by_grouped_route;
};

struct SparkGlm52SotaProductionDecodePlanSm121
{
    uint32_t abi_version;
    uint32_t active_token_count;
    uint32_t maximum_active_token_count;
    uint64_t capability_flags;
    uint64_t validated_stage_latency_ns;
    SparkGlm52SotaProductionMoePlanSm121 *moe_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *router_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *q_a_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *q_b_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *kv_a_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *kv_b_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *o_proj_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *dense_gate_up_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *dense_down_plan;
    SparkGlm52SotaBf16LinearPlanSm121 *restricted_logits_plan;
    void *flash_mla_plan;
    void *mtp_plan;
    SparkGlm52SotaDecodeGraphPlan *graph_plan;
};

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t SparkGlm52SotaValidateProductionDecodePlanSm121(
    const SparkGlm52SotaProductionDecodePlanSm121 *plan);

cudaError_t SparkGlm52SotaLaunchBf16CublasLtLinearSm121(
    SparkGlm52SotaBf16LinearPlanSm121 *plan,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaLaunchCutlassNvfp4GroupedGemmSm121(
    SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan,
    cudaStream_t stream);

uint64_t SparkGlm52SotaCutlassNvfp4GroupedGemmStateBytesSm121(void);

cudaError_t SparkGlm52SotaInitializeCutlassNvfp4GroupedGemmSm121(
    SparkGlm52SotaNvfp4GroupedGemmPlanSm121 *plan,
    void *state_memory,
    uint64_t state_memory_bytes,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaLaunchProductionGroupedMoeSm121(
    SparkGlm52SotaProductionMoePlanSm121 *plan,
    const SparkGlm52SotaMoeArguments *arguments,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
