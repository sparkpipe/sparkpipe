#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_FP4_GEMM_SENTINEL 0x5350435546503447ull
#define SPARKPIPE_CUDA_FP4_GROUPED_EXPERT_SENTINEL 0x53504634474D4F45ull
#define SPARKPIPE_CUDA_FP4_GEMM_HEURISTIC_BYTES 8192u
#define SPARKPIPE_CUDA_FP4_GEMM_SCALE_BLOCK 16u
#define SPARKPIPE_CUDA_FP4_GROUPED_MAX_EXPERTS 256u
#define SPARKPIPE_CUDA_FP4_GROUPED_STREAM_LIMIT 8u
#define SPARKPIPE_CUDA_FP4_GROUPED_STREAM_AUTO 0u
#define SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_CUBLASLT_PER_EXPERT 1u
#define SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_CUTLASS_BLACKWELL_MOE 2u
#define SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_DEEPGEMM_MEGA_MOE 3u

typedef struct SparkCudaFp4GemmPlan
{
	uint32_t m;
	uint32_t n;
	uint32_t k;
	uint32_t lda;
	uint32_t ldb;
	uint32_t ldd;
	uint64_t workspace_bytes;
	void *workspace;
	uint64_t stream;
	uint64_t lt_handle;
	uint64_t operation_desc;
	uint64_t a_desc;
	uint64_t b_desc;
	uint64_t d_desc;
	uint64_t preference;
	uint64_t a_scale_pointer;
	uint64_t b_scale_pointer;
	uint8_t heuristic[SPARKPIPE_CUDA_FP4_GEMM_HEURISTIC_BYTES];
	uint32_t initialized;
	uint64_t sentinel;
} SparkCudaFp4GemmPlan;

typedef struct SparkCudaFp4GemmReport
{
	uint64_t operation_count;
	uint64_t output_element_count;
	uint64_t flops_per_run;
	uint64_t packed_a_bytes;
	uint64_t packed_b_bytes;
	uint64_t a_scale_bytes;
	uint64_t b_scale_bytes;
	uint64_t workspace_bytes;
	uint32_t plan_initialized;
	uint32_t heuristic_result_count;
	uint32_t tensor_core_candidate;
	uint32_t run_count;
	uint32_t hot_path_allocation_count;
	uint32_t sentinel_violation_count;
	uint32_t unsupported_shape_count;
} SparkCudaFp4GemmReport;

typedef struct SparkCudaFp4GroupedExpertGemmPlan
{
	uint32_t expert_count;
	uint32_t active_expert_count;
	uint32_t m_per_expert;
	uint32_t n;
	uint32_t k;
	uint64_t a_stride_bytes;
	uint64_t b_stride_bytes;
	uint64_t d_stride_bytes;
	uint64_t a_scale_base;
	uint64_t b_scale_base;
	uint64_t a_scale_stride_bytes;
	uint64_t b_scale_stride_bytes;
	uint64_t workspace_bytes_per_expert;
	uint64_t workspace_total_bytes;
	uint64_t stream;
	uint64_t expert_streams[SPARKPIPE_CUDA_FP4_GROUPED_MAX_EXPERTS];
	uint32_t active_expert_ids[SPARKPIPE_CUDA_FP4_GROUPED_MAX_EXPERTS];
	SparkCudaFp4GemmPlan *expert_plans;
	uint32_t expert_plan_capacity;
	uint32_t expert_stream_count;
	uint32_t requested_stream_count;
	uint32_t workspace_stream_capacity;
	uint32_t backend_kind;
	uint32_t owns_expert_streams;
	uint32_t initialized;
	uint64_t sentinel;
} SparkCudaFp4GroupedExpertGemmPlan;

typedef struct SparkCudaFp4GroupedExpertGemmReport
{
	uint64_t operation_count;
	uint64_t output_element_count;
	uint64_t flops_per_run;
	uint64_t packed_a_bytes;
	uint64_t packed_b_bytes;
	uint64_t output_bytes;
	uint64_t a_scale_bytes;
	uint64_t b_scale_bytes;
	uint64_t workspace_bytes_per_expert;
	uint64_t workspace_total_bytes;
	uint32_t expert_count;
	uint32_t active_expert_count;
	uint32_t m_per_expert;
	uint32_t n;
	uint32_t k;
	uint32_t backend_kind;
	uint32_t expert_stream_count;
	uint32_t workspace_stream_capacity;
	uint32_t plan_initialized_count;
	uint32_t run_count;
	uint32_t hot_path_allocation_count;
	uint32_t sentinel_violation_count;
	uint32_t unsupported_shape_count;
	uint32_t failed_expert_count;
} SparkCudaFp4GroupedExpertGemmReport;

SparkStatus SparkValidateCudaFp4GemmShape(uint32_t m, uint32_t n, uint32_t k);
SparkStatus SparkValidateCudaFp4GroupedExpertGemmShape(uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k);
uint64_t SparkCudaFp4GemmPackedBytes(uint32_t rows, uint32_t cols);
uint64_t SparkCudaFp4GemmScaleBytes(uint32_t rows, uint32_t cols);
SparkStatus SparkInitCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan, uint32_t m, uint32_t n, uint32_t k, const void *device_a_scale_ue4m3, const void *device_b_scale_ue4m3, void *workspace, uint64_t workspace_bytes, uint64_t stream);
SparkStatus SparkDestroyCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan);
SparkStatus SparkRunCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan, const void *device_a_fp4, const void *device_b_fp4, float alpha, void *device_d_bf16, SparkCudaFp4GemmReport *report);
SparkStatus SparkInitCudaFp4GroupedExpertGemmPlanEx(SparkCudaFp4GroupedExpertGemmPlan *plan, SparkCudaFp4GemmPlan *expert_plans, uint32_t expert_plan_capacity, uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k, const uint32_t *active_expert_ids, const void *device_a_scale_base_ue4m3, uint64_t a_scale_stride_bytes, const void *device_b_scale_base_ue4m3, uint64_t b_scale_stride_bytes, void *workspace, uint64_t workspace_bytes_per_expert, uint64_t workspace_total_bytes, uint32_t requested_stream_count, uint32_t backend_kind, uint64_t stream);
SparkStatus SparkInitCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan, SparkCudaFp4GemmPlan *expert_plans, uint32_t expert_plan_capacity, uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k, const uint32_t *active_expert_ids, const void *device_a_scale_base_ue4m3, uint64_t a_scale_stride_bytes, const void *device_b_scale_base_ue4m3, uint64_t b_scale_stride_bytes, void *workspace, uint64_t workspace_bytes_per_expert, uint64_t stream);
SparkStatus SparkDestroyCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan);
SparkStatus SparkRunCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan, const void *device_a_fp4_base, uint64_t a_stride_bytes, const void *device_b_fp4_base, uint64_t b_stride_bytes, float alpha, void *device_d_bf16_base, uint64_t d_stride_bytes, SparkCudaFp4GroupedExpertGemmReport *report);

#ifdef __cplusplus
}
#endif
