#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_fp4_gemm.h"

#define SPARKPIPE_CUDA_FP4_GEMM_HEURISTIC_LIMIT 8

static SparkStatus SparkStatusFromCublasLtFp4(cublasStatus_t status)
{
	if (status == CUBLAS_STATUS_SUCCESS)
		return SPARK_STATUS_OK;
	if (status == CUBLAS_STATUS_NOT_SUPPORTED)
		return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
	if (status == CUBLAS_STATUS_INVALID_VALUE || status == CUBLAS_STATUS_NOT_INITIALIZED)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkStatusFromCudaFp4(cudaError_t status)
{
	if (status == cudaSuccess)
		return SPARK_STATUS_OK;
	if (status == cudaErrorInvalidValue)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkSetFp4RowMajorLayout(cublasLtMatrixLayout_t layout)
{
	cublasLtOrder_t order;
	cublasStatus_t status;

	order = CUBLASLT_ORDER_ROW;
	status = cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
	return SparkStatusFromCublasLtFp4(status);
}

static void SparkDestroyFp4PlanObjects(SparkCudaFp4GemmPlan *plan)
{
	if (plan->preference != 0u)
		cublasLtMatmulPreferenceDestroy((cublasLtMatmulPreference_t)(uintptr_t)plan->preference);
	if (plan->d_desc != 0u)
		cublasLtMatrixLayoutDestroy((cublasLtMatrixLayout_t)(uintptr_t)plan->d_desc);
	if (plan->b_desc != 0u)
		cublasLtMatrixLayoutDestroy((cublasLtMatrixLayout_t)(uintptr_t)plan->b_desc);
	if (plan->a_desc != 0u)
		cublasLtMatrixLayoutDestroy((cublasLtMatrixLayout_t)(uintptr_t)plan->a_desc);
	if (plan->operation_desc != 0u)
		cublasLtMatmulDescDestroy((cublasLtMatmulDesc_t)(uintptr_t)plan->operation_desc);
	if (plan->lt_handle != 0u)
		cublasLtDestroy((cublasLtHandle_t)(uintptr_t)plan->lt_handle);
	plan->preference = 0u;
	plan->d_desc = 0u;
	plan->b_desc = 0u;
	plan->a_desc = 0u;
	plan->operation_desc = 0u;
	plan->lt_handle = 0u;
}

static void SparkFillFp4ReportShape(const SparkCudaFp4GemmPlan *plan, SparkCudaFp4GemmReport *report)
{
	report->output_element_count = (uint64_t)plan->m * (uint64_t)plan->n;
	report->flops_per_run = 2ull * (uint64_t)plan->m * (uint64_t)plan->n * (uint64_t)plan->k;
	report->packed_a_bytes = SparkCudaFp4GemmPackedBytes(plan->m, plan->k);
	report->packed_b_bytes = SparkCudaFp4GemmPackedBytes(plan->n, plan->k);
	report->a_scale_bytes = SparkCudaFp4GemmScaleBytes(plan->m, plan->k);
	report->b_scale_bytes = SparkCudaFp4GemmScaleBytes(plan->n, plan->k);
	report->workspace_bytes = plan->workspace_bytes;
}

static uint64_t SparkFp4GroupedDefaultAStrideBytes(const SparkCudaFp4GroupedExpertGemmPlan *plan)
{
	return SparkCudaFp4GemmPackedBytes(plan->m_per_expert, plan->k);
}

static uint64_t SparkFp4GroupedDefaultBStrideBytes(const SparkCudaFp4GroupedExpertGemmPlan *plan)
{
	return SparkCudaFp4GemmPackedBytes(plan->n, plan->k);
}

static uint64_t SparkFp4GroupedDefaultDStrideBytes(const SparkCudaFp4GroupedExpertGemmPlan *plan)
{
	return (uint64_t)plan->m_per_expert * (uint64_t)plan->n * sizeof(uint16_t);
}

static void SparkFillFp4GroupedReportShape(const SparkCudaFp4GroupedExpertGemmPlan *plan, SparkCudaFp4GroupedExpertGemmReport *report)
{
	report->operation_count = plan->active_expert_count;
	report->output_element_count = (uint64_t)plan->active_expert_count * (uint64_t)plan->m_per_expert * (uint64_t)plan->n;
	report->flops_per_run = 2ull * (uint64_t)plan->active_expert_count * (uint64_t)plan->m_per_expert * (uint64_t)plan->n * (uint64_t)plan->k;
	report->packed_a_bytes = (uint64_t)plan->active_expert_count * SparkCudaFp4GemmPackedBytes(plan->m_per_expert, plan->k);
	report->packed_b_bytes = (uint64_t)plan->active_expert_count * SparkCudaFp4GemmPackedBytes(plan->n, plan->k);
	report->output_bytes = (uint64_t)plan->active_expert_count * SparkFp4GroupedDefaultDStrideBytes(plan);
	report->a_scale_bytes = (uint64_t)plan->active_expert_count * SparkCudaFp4GemmScaleBytes(plan->m_per_expert, plan->k);
	report->b_scale_bytes = (uint64_t)plan->active_expert_count * SparkCudaFp4GemmScaleBytes(plan->n, plan->k);
	report->workspace_bytes_per_expert = plan->workspace_bytes_per_expert;
	report->workspace_total_bytes = plan->workspace_total_bytes;
	report->expert_count = plan->expert_count;
	report->active_expert_count = plan->active_expert_count;
	report->m_per_expert = plan->m_per_expert;
	report->n = plan->n;
	report->k = plan->k;
	report->backend_kind = plan->backend_kind;
	report->expert_stream_count = plan->expert_stream_count;
	report->workspace_stream_capacity = plan->workspace_stream_capacity;
}

static SparkStatus SparkValidateFp4GroupedExpertIds(uint32_t expert_count, uint32_t active_expert_count, const uint32_t *active_expert_ids, uint32_t *stored_expert_ids)
{
	uint32_t seen_words[SPARKPIPE_CUDA_FP4_GROUPED_MAX_EXPERTS / 32u];
	uint32_t active_index;

	memset(seen_words, 0, sizeof(seen_words));
	for (active_index = 0u; active_index < active_expert_count; ++active_index)
	{
		uint32_t expert_id;
		uint32_t word_index;
		uint32_t bit_mask;

		expert_id = active_expert_ids != 0 ? active_expert_ids[active_index] : active_index;
		if (expert_id >= expert_count)
			return SPARK_STATUS_INVALID_ARGUMENT;
		word_index = expert_id >> 5u;
		bit_mask = 1u << (expert_id & 31u);
		if ((seen_words[word_index] & bit_mask) != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		seen_words[word_index] |= bit_mask;
		stored_expert_ids[active_index] = expert_id;
	}
	return SPARK_STATUS_OK;
}

static void SparkDestroyFp4GroupedExpertStreams(SparkCudaFp4GroupedExpertGemmPlan *plan)
{
	uint32_t active_index;

	if (plan->owns_expert_streams == 0u)
		return;
	for (active_index = 0u; active_index < plan->expert_stream_count; ++active_index)
	{
		if (plan->expert_streams[active_index] != 0u)
			(void)cudaStreamDestroy((cudaStream_t)(uintptr_t)plan->expert_streams[active_index]);
		plan->expert_streams[active_index] = 0u;
	}
	plan->owns_expert_streams = 0u;
}

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

extern "C" SparkStatus SparkInitCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan, uint32_t m, uint32_t n, uint32_t k, const void *device_a_scale_ue4m3, const void *device_b_scale_ue4m3, void *workspace, uint64_t workspace_bytes, uint64_t stream)
{
	cublasLtMatmulHeuristicResult_t heuristics[SPARKPIPE_CUDA_FP4_GEMM_HEURISTIC_LIMIT];
	cublasLtMatmulPreference_t preference;
	cublasLtMatrixLayout_t a_desc;
	cublasLtMatrixLayout_t b_desc;
	cublasLtMatrixLayout_t d_desc;
	cublasLtMatmulDesc_t operation_desc;
	cublasLtHandle_t lt_handle;
	cublasLtMatmulMatrixScale_t scale_mode;
	cublasOperation_t transa;
	cublasOperation_t transb;
	cublasStatus_t cublas_status;
	SparkStatus status;
	int32_t returned_results;

	if (plan == 0 || device_a_scale_ue4m3 == 0 || device_b_scale_ue4m3 == 0 || workspace == 0 || workspace_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkValidateCudaFp4GemmShape(m, n, k);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(plan, 0, sizeof(*plan));
	memset(heuristics, 0, sizeof(heuristics));
	lt_handle = 0;
	operation_desc = 0;
	a_desc = 0;
	b_desc = 0;
	d_desc = 0;
	preference = 0;
	returned_results = 0;
	cublas_status = cublasLtCreate(&lt_handle);
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
	transa = CUBLAS_OP_N;
	transb = CUBLAS_OP_T;
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
	scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &device_a_scale_ue4m3, sizeof(device_a_scale_ue4m3));
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &device_b_scale_ue4m3, sizeof(device_b_scale_ue4m3));
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t)m, (uint64_t)k, (int64_t)k);
	if (cublas_status == CUBLAS_STATUS_SUCCESS && SparkSetFp4RowMajorLayout(a_desc) != SPARK_STATUS_OK)
		cublas_status = CUBLAS_STATUS_INVALID_VALUE;
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t)n, (uint64_t)k, (int64_t)k);
	if (cublas_status == CUBLAS_STATUS_SUCCESS && SparkSetFp4RowMajorLayout(b_desc) != SPARK_STATUS_OK)
		cublas_status = CUBLAS_STATUS_INVALID_VALUE;
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, (uint64_t)m, (uint64_t)n, (int64_t)n);
	if (cublas_status == CUBLAS_STATUS_SUCCESS && SparkSetFp4RowMajorLayout(d_desc) != SPARK_STATUS_OK)
		cublas_status = CUBLAS_STATUS_INVALID_VALUE;
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulPreferenceCreate(&preference);
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
	if (cublas_status == CUBLAS_STATUS_SUCCESS)
		cublas_status = cublasLtMatmulAlgoGetHeuristic(lt_handle, operation_desc, a_desc, b_desc, d_desc, d_desc, preference, SPARKPIPE_CUDA_FP4_GEMM_HEURISTIC_LIMIT, heuristics, &returned_results);
	if (cublas_status != CUBLAS_STATUS_SUCCESS || returned_results <= 0)
	{
		plan->lt_handle = (uint64_t)(uintptr_t)lt_handle;
		plan->operation_desc = (uint64_t)(uintptr_t)operation_desc;
		plan->a_desc = (uint64_t)(uintptr_t)a_desc;
		plan->b_desc = (uint64_t)(uintptr_t)b_desc;
		plan->d_desc = (uint64_t)(uintptr_t)d_desc;
		plan->preference = (uint64_t)(uintptr_t)preference;
		SparkDestroyFp4PlanObjects(plan);
		memset(plan, 0, sizeof(*plan));
		return cublas_status == CUBLAS_STATUS_SUCCESS ? SPARK_STATUS_GRAPH_NOT_AVAILABLE : SparkStatusFromCublasLtFp4(cublas_status);
	}
	plan->m = m;
	plan->n = n;
	plan->k = k;
	plan->lda = k;
	plan->ldb = k;
	plan->ldd = n;
	plan->workspace = workspace;
	plan->workspace_bytes = workspace_bytes;
	plan->stream = stream;
	plan->lt_handle = (uint64_t)(uintptr_t)lt_handle;
	plan->operation_desc = (uint64_t)(uintptr_t)operation_desc;
	plan->a_desc = (uint64_t)(uintptr_t)a_desc;
	plan->b_desc = (uint64_t)(uintptr_t)b_desc;
	plan->d_desc = (uint64_t)(uintptr_t)d_desc;
	plan->preference = (uint64_t)(uintptr_t)preference;
	plan->a_scale_pointer = (uint64_t)(uintptr_t)device_a_scale_ue4m3;
	plan->b_scale_pointer = (uint64_t)(uintptr_t)device_b_scale_ue4m3;
	memcpy(plan->heuristic, &heuristics[0], sizeof(heuristics[0]));
	plan->initialized = 1u;
	plan->sentinel = SPARKPIPE_CUDA_FP4_GEMM_SENTINEL;
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkDestroyCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan)
{
	if (plan == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkDestroyFp4PlanObjects(plan);
	memset(plan, 0, sizeof(*plan));
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaFp4GemmPlan(SparkCudaFp4GemmPlan *plan, const void *device_a_fp4, const void *device_b_fp4, float alpha, void *device_d_bf16, SparkCudaFp4GemmReport *report)
{
	cublasLtMatmulHeuristicResult_t heuristic;
	cublasStatus_t cublas_status;
	cudaError_t cuda_status;
	float beta;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	if (plan == 0 || device_a_fp4 == 0 || device_b_fp4 == 0 || device_d_bf16 == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (plan->initialized == 0u || plan->sentinel != SPARKPIPE_CUDA_FP4_GEMM_SENTINEL)
	{
		report->sentinel_violation_count = 1u;
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	memcpy(&heuristic, plan->heuristic, sizeof(heuristic));
	beta = 0.0f;
	cublas_status = cublasLtMatmul((cublasLtHandle_t)(uintptr_t)plan->lt_handle, (cublasLtMatmulDesc_t)(uintptr_t)plan->operation_desc, &alpha, device_a_fp4, (cublasLtMatrixLayout_t)(uintptr_t)plan->a_desc, device_b_fp4, (cublasLtMatrixLayout_t)(uintptr_t)plan->b_desc, &beta, device_d_bf16, (cublasLtMatrixLayout_t)(uintptr_t)plan->d_desc, device_d_bf16, (cublasLtMatrixLayout_t)(uintptr_t)plan->d_desc, &heuristic.algo, plan->workspace, (size_t)plan->workspace_bytes, (cudaStream_t)(uintptr_t)plan->stream);
	if (cublas_status != CUBLAS_STATUS_SUCCESS)
		return SparkStatusFromCublasLtFp4(cublas_status);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
		return SparkStatusFromCudaFp4(cuda_status);
	SparkFillFp4ReportShape(plan, report);
	report->operation_count = 1u;
	report->plan_initialized = plan->initialized;
	report->heuristic_result_count = 1u;
	report->tensor_core_candidate = 1u;
	report->run_count = 1u;
	report->hot_path_allocation_count = 0u;
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkInitCudaFp4GroupedExpertGemmPlanEx(SparkCudaFp4GroupedExpertGemmPlan *plan, SparkCudaFp4GemmPlan *expert_plans, uint32_t expert_plan_capacity, uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k, const uint32_t *active_expert_ids, const void *device_a_scale_base_ue4m3, uint64_t a_scale_stride_bytes, const void *device_b_scale_base_ue4m3, uint64_t b_scale_stride_bytes, void *workspace, uint64_t workspace_bytes_per_expert, uint64_t workspace_total_bytes, uint32_t requested_stream_count, uint32_t backend_kind, uint64_t stream)
{
	const uint8_t *a_scale_base;
	const uint8_t *b_scale_base;
	uint8_t *workspace_base;
	uint64_t default_a_scale_stride;
	uint64_t default_b_scale_stride;
	SparkStatus status;
	uint32_t stream_count;
	uint32_t active_index;
	uint32_t workspace_stream_capacity;

	if (plan == 0 || expert_plans == 0 || device_a_scale_base_ue4m3 == 0 || device_b_scale_base_ue4m3 == 0 || workspace == 0 || workspace_bytes_per_expert == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkValidateFp4GroupedBackend(backend_kind);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkValidateCudaFp4GroupedExpertGemmShape(expert_count, active_expert_count, m_per_expert, n, k);
	if (status != SPARK_STATUS_OK)
		return status;
	if (expert_plan_capacity < active_expert_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	workspace_stream_capacity = SparkFp4GroupedWorkspaceStreamCapacity(workspace_total_bytes, workspace_bytes_per_expert);
	stream_count = SparkFp4GroupedResolveStreamCount(active_expert_count, requested_stream_count, workspace_stream_capacity, stream);
	if (stream_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(plan, 0, sizeof(*plan));
	memset(expert_plans, 0, (size_t)expert_plan_capacity * sizeof(*expert_plans));
	status = SparkValidateFp4GroupedExpertIds(expert_count, active_expert_count, active_expert_ids, plan->active_expert_ids);
	if (status != SPARK_STATUS_OK)
		return status;
	plan->expert_count = expert_count;
	plan->active_expert_count = active_expert_count;
	plan->m_per_expert = m_per_expert;
	plan->n = n;
	plan->k = k;
	plan->a_scale_base = (uint64_t)(uintptr_t)device_a_scale_base_ue4m3;
	plan->b_scale_base = (uint64_t)(uintptr_t)device_b_scale_base_ue4m3;
	plan->workspace_bytes_per_expert = workspace_bytes_per_expert;
	plan->workspace_total_bytes = workspace_total_bytes;
	plan->stream = stream;
	plan->expert_plans = expert_plans;
	plan->expert_plan_capacity = expert_plan_capacity;
	plan->requested_stream_count = requested_stream_count;
	plan->workspace_stream_capacity = workspace_stream_capacity;
	plan->backend_kind = backend_kind;
	default_a_scale_stride = SparkCudaFp4GemmScaleBytes(m_per_expert, k);
	default_b_scale_stride = SparkCudaFp4GemmScaleBytes(n, k);
	if (a_scale_stride_bytes == 0u)
		a_scale_stride_bytes = default_a_scale_stride;
	if (b_scale_stride_bytes == 0u)
		b_scale_stride_bytes = default_b_scale_stride;
	plan->a_scale_stride_bytes = a_scale_stride_bytes;
	plan->b_scale_stride_bytes = b_scale_stride_bytes;
	workspace_base = (uint8_t *)workspace;
	a_scale_base = (const uint8_t *)device_a_scale_base_ue4m3;
	b_scale_base = (const uint8_t *)device_b_scale_base_ue4m3;
	if (stream == 0u && active_expert_count > 1u)
	{
		for (active_index = 0u; active_index < stream_count; ++active_index)
		{
			cudaStream_t expert_stream;

			expert_stream = 0;
			if (cudaStreamCreateWithFlags(&expert_stream, cudaStreamNonBlocking) != cudaSuccess)
			{
				SparkDestroyFp4GroupedExpertStreams(plan);
				memset(plan, 0, sizeof(*plan));
				return SPARK_STATUS_INTERNAL_ERROR;
			}
			plan->expert_streams[active_index] = (uint64_t)(uintptr_t)expert_stream;
		}
		plan->expert_stream_count = stream_count;
		plan->owns_expert_streams = 1u;
	}
	for (active_index = 0u; active_index < active_expert_count; ++active_index)
	{
		uint64_t expert_stream;
		uint64_t workspace_index;

		workspace_index = plan->owns_expert_streams != 0u ? (uint64_t)(active_index % plan->expert_stream_count) : 0u;
		expert_stream = plan->owns_expert_streams != 0u ? plan->expert_streams[workspace_index] : stream;
		status = SparkInitCudaFp4GemmPlan(&expert_plans[active_index], m_per_expert, n, k, a_scale_base + ((uint64_t)active_index * a_scale_stride_bytes), b_scale_base + ((uint64_t)active_index * b_scale_stride_bytes), workspace_base + (workspace_index * workspace_bytes_per_expert), workspace_bytes_per_expert, expert_stream);
		if (status != SPARK_STATUS_OK)
		{
			while (active_index > 0u)
			{
				--active_index;
				(void)SparkDestroyCudaFp4GemmPlan(&expert_plans[active_index]);
			}
			SparkDestroyFp4GroupedExpertStreams(plan);
			memset(plan, 0, sizeof(*plan));
			return status;
		}
	}
	plan->initialized = 1u;
	plan->sentinel = SPARKPIPE_CUDA_FP4_GROUPED_EXPERT_SENTINEL;
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkInitCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan, SparkCudaFp4GemmPlan *expert_plans, uint32_t expert_plan_capacity, uint32_t expert_count, uint32_t active_expert_count, uint32_t m_per_expert, uint32_t n, uint32_t k, const uint32_t *active_expert_ids, const void *device_a_scale_base_ue4m3, uint64_t a_scale_stride_bytes, const void *device_b_scale_base_ue4m3, uint64_t b_scale_stride_bytes, void *workspace, uint64_t workspace_bytes_per_expert, uint64_t stream)
{
	SparkStatus status;
	uint64_t workspace_total_bytes;

	status = SparkFp4GroupedDefaultWorkspaceTotal(active_expert_count, workspace_bytes_per_expert, &workspace_total_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkInitCudaFp4GroupedExpertGemmPlanEx(plan, expert_plans, expert_plan_capacity, expert_count, active_expert_count, m_per_expert, n, k, active_expert_ids, device_a_scale_base_ue4m3, a_scale_stride_bytes, device_b_scale_base_ue4m3, b_scale_stride_bytes, workspace, workspace_bytes_per_expert, workspace_total_bytes, SPARKPIPE_CUDA_FP4_GROUPED_STREAM_AUTO, SPARKPIPE_CUDA_FP4_GROUPED_BACKEND_CUBLASLT_PER_EXPERT, stream);
}

extern "C" SparkStatus SparkDestroyCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan)
{
	if (plan == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (plan->expert_plans != 0 && plan->expert_plan_capacity > 0u)
	{
		uint32_t active_index;

		for (active_index = 0u; active_index < plan->active_expert_count; ++active_index)
			(void)SparkDestroyCudaFp4GemmPlan(&plan->expert_plans[active_index]);
	}
	SparkDestroyFp4GroupedExpertStreams(plan);
	memset(plan, 0, sizeof(*plan));
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaFp4GroupedExpertGemmPlan(SparkCudaFp4GroupedExpertGemmPlan *plan, const void *device_a_fp4_base, uint64_t a_stride_bytes, const void *device_b_fp4_base, uint64_t b_stride_bytes, float alpha, void *device_d_bf16_base, uint64_t d_stride_bytes, SparkCudaFp4GroupedExpertGemmReport *report)
{
	const uint8_t *a_base;
	const uint8_t *b_base;
	uint8_t *d_base;
	SparkStatus status;
	uint32_t active_index;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	if (plan == 0 || device_a_fp4_base == 0 || device_b_fp4_base == 0 || device_d_bf16_base == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (plan->initialized == 0u || plan->sentinel != SPARKPIPE_CUDA_FP4_GROUPED_EXPERT_SENTINEL || plan->expert_plans == 0)
	{
		report->sentinel_violation_count = 1u;
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if (a_stride_bytes == 0u)
		a_stride_bytes = SparkFp4GroupedDefaultAStrideBytes(plan);
	if (b_stride_bytes == 0u)
		b_stride_bytes = SparkFp4GroupedDefaultBStrideBytes(plan);
	if (d_stride_bytes == 0u)
		d_stride_bytes = SparkFp4GroupedDefaultDStrideBytes(plan);
	a_base = (const uint8_t *)device_a_fp4_base;
	b_base = (const uint8_t *)device_b_fp4_base;
	d_base = (uint8_t *)device_d_bf16_base;
	SparkFillFp4GroupedReportShape(plan, report);
	report->plan_initialized_count = plan->active_expert_count;
	for (active_index = 0u; active_index < plan->active_expert_count; ++active_index)
	{
		SparkCudaFp4GemmReport expert_report;
		SparkCudaFp4GemmPlan *expert_plan;
		const void *a_ptr;
		const void *b_ptr;
		void *d_ptr;

		memset(&expert_report, 0, sizeof(expert_report));
		expert_plan = &plan->expert_plans[active_index];
		a_ptr = (const void *)(a_base + ((uint64_t)active_index * a_stride_bytes));
		b_ptr = (const void *)(b_base + ((uint64_t)active_index * b_stride_bytes));
		d_ptr = (void *)(d_base + ((uint64_t)active_index * d_stride_bytes));
		status = SparkRunCudaFp4GemmPlan(expert_plan, a_ptr, b_ptr, alpha, d_ptr, &expert_report);
		if (status != SPARK_STATUS_OK)
		{
			report->failed_expert_count += 1u;
			return status;
		}
		report->run_count += expert_report.run_count;
		report->hot_path_allocation_count += expert_report.hot_path_allocation_count;
	}
	if (plan->owns_expert_streams != 0u)
	{
		for (active_index = 0u; active_index < plan->expert_stream_count; ++active_index)
		{
			if (cudaStreamSynchronize((cudaStream_t)(uintptr_t)plan->expert_streams[active_index]) != cudaSuccess)
			{
				report->failed_expert_count += 1u;
				return SPARK_STATUS_INTERNAL_ERROR;
			}
		}
	}
	else if (cudaStreamSynchronize((cudaStream_t)(uintptr_t)plan->stream) != cudaSuccess)
	{
		report->failed_expert_count += 1u;
		return SPARK_STATUS_INTERNAL_ERROR;
	}
	return SPARK_STATUS_OK;
}
