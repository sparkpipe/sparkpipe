#include <string.h>

#include "sparkpipe/spark_cuda_fp4_gemm.h"

SparkStatus SparkValidateCudaFp4GemmShape(uint32_t m, uint32_t n, uint32_t k)
{
	if (m == 0u || n == 0u || k == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((k % SPARKPIPE_CUDA_FP4_GEMM_SCALE_BLOCK) != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((m & 15u) != 0u || (n & 15u) != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkValidateCudaFp4GroupedExpertGemmShape(uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k)
{
	if (expert_count == 0u || active_expert_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (expert_count > SPARKPIPE_CUDA_FP4_GROUPED_MAX_EXPERTS || active_expert_count > expert_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (m_per_expert != 16u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkValidateCudaFp4GemmShape(m_per_expert, n, k);
}

uint64_t SparkCudaFp4GemmPackedBytes(uint32_t rows, uint32_t cols)
{
	if (rows == 0u || cols == 0u || (cols & 1u) != 0u)
		return 0u;
	return ((uint64_t)rows * (uint64_t)cols) >> 1u;
}

uint64_t SparkCudaFp4GemmScaleBytes(uint32_t rows, uint32_t cols)
{
	uint64_t scale_bytes;
	uint32_t scale_rows;

	if (rows == 0u || cols == 0u || (cols % SPARKPIPE_CUDA_FP4_GEMM_SCALE_BLOCK) != 0u)
		return 0u;
	scale_rows = rows < 32u ? 32u : rows;
	scale_bytes = (uint64_t)scale_rows * ((uint64_t)cols / SPARKPIPE_CUDA_FP4_GEMM_SCALE_BLOCK);
	return (scale_bytes + 511ull) & ~511ull;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static uint32_t SparkFp4GroupedMinU32(uint32_t a, uint32_t b)
{
	return a < b ? a : b;
}

static uint32_t SparkFp4GroupedWorkspaceStreamCapacity(uint64_t workspace_total_bytes, uint64_t workspace_bytes_per_expert)
{
	uint64_t capacity;

	if (workspace_total_bytes == 0u || workspace_bytes_per_expert == 0u)
		return 0u;
	capacity = workspace_total_bytes / workspace_bytes_per_expert;
	if (capacity > SPARKPIPE_CUDA_FP4_GROUPED_MAX_EXPERTS)
		return SPARKPIPE_CUDA_FP4_GROUPED_MAX_EXPERTS;
	return (uint32_t)capacity;
}

static SparkStatus SparkFp4GroupedDefaultWorkspaceTotal(uint32_t active_expert_count, uint64_t workspace_bytes_per_expert, uint64_t *workspace_total_bytes)
{
	uint32_t stream_count;

	if (workspace_total_bytes == 0 || workspace_bytes_per_expert == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	stream_count = SparkFp4GroupedMinU32(active_expert_count, SPARKPIPE_CUDA_FP4_GROUPED_STREAM_LIMIT);
	if (workspace_bytes_per_expert > (UINT64_MAX / (uint64_t)stream_count))
		return SPARK_STATUS_INVALID_ARGUMENT;
	*workspace_total_bytes = workspace_bytes_per_expert * (uint64_t)stream_count;
	return SPARK_STATUS_OK;
}

static uint32_t SparkFp4GroupedResolveStreamCount(uint32_t active_expert_count, uint32_t requested_stream_count, uint32_t workspace_stream_capacity, uint64_t stream)
{
	uint32_t stream_count;

	if (stream != 0u)
		return workspace_stream_capacity == 0u ? 0u : 1u;
	stream_count = SparkFp4GroupedMinU32(active_expert_count, SPARKPIPE_CUDA_FP4_GROUPED_STREAM_LIMIT);
	stream_count = SparkFp4GroupedMinU32(stream_count, workspace_stream_capacity);
	if (requested_stream_count != SPARKPIPE_CUDA_FP4_GROUPED_STREAM_AUTO)
	{
		if (requested_stream_count == 0u || requested_stream_count > stream_count)
			return 0u;
		stream_count = requested_stream_count;
	}
	return stream_count;
}

static SparkStatus SparkValidateFp4GroupedBackend(uint32_t backend_kind)
{
	if (backend_kind == SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_CUBLASLT_PER_EXPERT)
		return SPARK_STATUS_OK;
	if (backend_kind == SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_CUTLASS_BLACKWELL_MOE || backend_kind == SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_DEEPGEMM_MEGA_MOE)
		return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
	return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkInitCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan, uint32_t m, uint32_t n, uint32_t k, const void *device_a_scale_ue4m3, const void *device_b_scale_ue4m3, void *workspace, uint64_t workspace_bytes, uint64_t stream)
{
	(void)device_a_scale_ue4m3;
	(void)device_b_scale_ue4m3;
	(void)workspace;
	(void)workspace_bytes;
	(void)stream;
	if (plan == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(plan, 0, sizeof(*plan));
	if (SparkValidateCudaFp4GemmShape(m, n, k) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkDestroyCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan)
{
	if (plan == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(plan, 0, sizeof(*plan));
	return SPARK_STATUS_OK;
}

SparkStatus SparkRunCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan, const void *device_a_fp4, const void *device_b_fp4, float alpha, void *device_d_bf16, SparkCudaFp4GemmReport *report)
{
	(void)plan;
	(void)device_a_fp4;
	(void)device_b_fp4;
	(void)alpha;
	(void)device_d_bf16;
	if (report != 0)
		memset(report, 0, sizeof(*report));
	return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkInitCudaFp4GroupedExpertGemmPlanEx(SparkCudaFp4GroupedExpertGemmPlan *plan, SparkCudaFp4GemmPlan *expert_plans, uint32_t expert_plan_capacity, uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k, const uint32_t *active_expert_ids, const void *device_a_scale_base_ue4m3, uint64_t a_scale_stride_bytes, const void *device_b_scale_base_ue4m3, uint64_t b_scale_stride_bytes, void *workspace, uint64_t workspace_bytes_per_expert, uint64_t workspace_total_bytes, uint32_t requested_stream_count, uint32_t backend_kind, uint64_t stream)
{
	SparkStatus status;

	(void)active_expert_ids;
	(void)device_a_scale_base_ue4m3;
	(void)a_scale_stride_bytes;
	(void)device_b_scale_base_ue4m3;
	(void)b_scale_stride_bytes;
	(void)workspace;
	if (expert_plans == 0 || workspace_bytes_per_expert == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (plan == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(plan, 0, sizeof(*plan));
	status = SparkValidateFp4GroupedBackend(backend_kind);
	if (status != SPARK_STATUS_OK)
		return status;
	if (SparkValidateCudaFp4GroupedExpertGemmShape(expert_count, active_expert_count, m_per_expert, n, k) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (expert_plan_capacity < active_expert_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (SparkFp4GroupedResolveStreamCount(active_expert_count, requested_stream_count, SparkFp4GroupedWorkspaceStreamCapacity(workspace_total_bytes, workspace_bytes_per_expert), stream) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkInitCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan, SparkCudaFp4GemmPlan *expert_plans, uint32_t expert_plan_capacity, uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k, const uint32_t *active_expert_ids, const void *device_a_scale_base_ue4m3, uint64_t a_scale_stride_bytes, const void *device_b_scale_base_ue4m3, uint64_t b_scale_stride_bytes, void *workspace, uint64_t workspace_bytes_per_expert, uint64_t stream)
{
	SparkStatus status;
	uint64_t workspace_total_bytes;

	status = SparkFp4GroupedDefaultWorkspaceTotal(active_expert_count, workspace_bytes_per_expert, &workspace_total_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkInitCudaFp4GroupedExpertGemmPlanEx(plan, expert_plans, expert_plan_capacity, expert_count, active_expert_count, m_per_expert, n, k, active_expert_ids, device_a_scale_base_ue4m3, a_scale_stride_bytes, device_b_scale_base_ue4m3, b_scale_stride_bytes, workspace, workspace_bytes_per_expert, workspace_total_bytes, SPARKPIPE_CUDA_FP4_GROUPED_STREAM_AUTO, SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_CUBLASLT_PER_EXPERT, stream);
}

SparkStatus SparkDestroyCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan)
{
	if (plan == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(plan, 0, sizeof(*plan));
	return SPARK_STATUS_OK;
}

SparkStatus SparkRunCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan, const void *device_a_fp4_base, uint64_t a_stride_bytes, const void *device_b_fp4_base, uint64_t b_stride_bytes, float alpha, void *device_d_bf16_base, uint64_t d_stride_bytes, SparkCudaFp4GroupedExpertGemmReport *report)
{
	(void)plan;
	(void)device_a_fp4_base;
	(void)a_stride_bytes;
	(void)device_b_fp4_base;
	(void)b_stride_bytes;
	(void)alpha;
	(void)device_d_bf16_base;
	(void)d_stride_bytes;
	if (report != 0)
		memset(report, 0, sizeof(*report));
	return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
