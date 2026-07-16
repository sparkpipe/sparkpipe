#include <assert.h>
#include <string.h>

#define main SparkTestGlm52CudaResidentGateToolMain
#include "../tools/sparkpipe_glm52_cuda_resident_gate.c"
#undef main

static void SparkTestGlm52CudaResidentGateAcceptsMeasuredB1024(void)
{
	SparkGlm52CudaResidentIpcStats stats;

	memset(&stats,0,sizeof(stats));
	stats.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_STATS_BYTES;
	stats.state = SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_READY;
	stats.capability_flags =
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_DRIVER_RESIDENT |
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_BUILDER_RESIDENT |
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_TRANSPORT_RESIDENT |
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_CUDA_STATE_RESIDENT;
	stats.rank_index = 0u;
	stats.max_active_sequence_count = 1024u;
	stats.logical_lane_capacity = 1024u;
	stats.execution_row_capacity = 8192u;
	stats.model_quantization_mode =
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
	stats.moe_backend_kind =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MOE_BACKEND_FP8_FLASHINFER_GROUPED;
	stats.moe_bound_layer_count = 3u;
	stats.moe_expected_layer_count = 3u;
	stats.fp8_scaled_gemm_bound_plan_count = 42u;
	stats.fp8_scaled_gemm_expected_plan_count = 42u;
	stats.kv_nvme_enabled = 1u;
	stats.kv_nvme_mode =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT;
	stats.kv_physical_block_capacity = 65536u;
	stats.kv_logical_block_capacity = 1048576u;
	stats.kv_resident_bytes_per_token = 7680u;
	stats.kv_resident_pool_bytes = UINT64_C(32313114624);
	stats.kv_nvme_record_bytes = 499712u;
	stats.kv_nvme_capacity_bytes = UINT64_C(523986010112);
	stats.kv_nvme_batch_block_capacity = 32u;
	stats.work_queue_accepted_count = 2u;
	stats.work_queue_submit_count = 2u;
	stats.asynchronous_submit_count = 2u;
	stats.asynchronous_completion_count = 2u;
	stats.layer_major_submit_count = 1u;
	stats.layer_major_completion_count = 1u;
	stats.last_layer_major_logical_lane_count = 1024u;
	stats.last_layer_major_rows_per_lane =
		SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
	stats.last_layer_major_execution_row_count =
		1024u * SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
		SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_WORK |
		SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_LAYER_MAJOR) ==
		SPARK_STATUS_OK);
	stats.fp8_scaled_gemm_bound_plan_count -= 1u;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,0u) == SPARK_STATUS_MODULE_NOT_VALIDATED);
	stats.fp8_scaled_gemm_bound_plan_count += 1u;
	stats.kv_nvme_pending_load_count = 1u;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,0u) == SPARK_STATUS_MODULE_NOT_VALIDATED);
	stats.kv_nvme_pending_load_count = 0u;
	stats.model_quantization_mode =
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT;
	stats.moe_backend_kind =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MOE_BACKEND_W8LUT_BF16_WMMA;
	stats.fp8_scaled_gemm_bound_plan_count = 0u;
	stats.fp8_scaled_gemm_expected_plan_count = 0u;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,0u) ==
		SPARK_STATUS_OK);
	stats.fp8_scaled_gemm_bound_plan_count = 1u;
	stats.fp8_scaled_gemm_expected_plan_count = 1u;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,0u) ==
		SPARK_STATUS_MODULE_NOT_VALIDATED);
	stats.fp8_scaled_gemm_bound_plan_count = 0u;
	stats.fp8_scaled_gemm_expected_plan_count = 0u;
	stats.model_quantization_mode =
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
	stats.moe_backend_kind =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MOE_BACKEND_NVFP4_B12X;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,0u) ==
		SPARK_STATUS_OK);
	stats.moe_backend_kind =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MOE_BACKEND_FP8_FLASHINFER_GROUPED;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,0u) ==
		SPARK_STATUS_MODULE_NOT_VALIDATED);
	stats.model_quantization_mode =
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT;
	stats.moe_backend_kind =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MOE_BACKEND_W8LUT_BF16_WMMA;
	assert(SparkGlm52CudaResidentGateValidateStats(
		&stats,0u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,0u) ==
		SPARK_STATUS_MODULE_NOT_VALIDATED);
}

int main(void)
{
	SparkTestGlm52CudaResidentGateAcceptsMeasuredB1024();
	return 0;
}
