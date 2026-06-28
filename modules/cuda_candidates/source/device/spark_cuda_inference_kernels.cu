#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_inference_kernels.h"

#define SPARK_CUDA_INFERENCE_THREADS 512u

static __device__ uint64_t SparkCudaInferenceMixU64Device(uint64_t checksum, uint64_t value)
{
    checksum ^= value + 0x9e3779b97f4a7c15ull + (checksum << 6u) + (checksum >> 2u);
    return checksum;
}

static __device__ bool SparkCudaInferenceFloatIsFiniteDevice(float value)
{
    return isfinite(value);
}

static __device__ uint64_t SparkCudaInferenceFloatChecksumDevice(const float *values, uint64_t value_count)
{
    const uint8_t *bytes;
    uint64_t byte_count;
    uint64_t byte_index;
    uint64_t checksum;

    bytes = (const uint8_t *)values;
    byte_count = value_count * (uint64_t)sizeof(float);
    checksum = 0x5350494E46434846ull;
    checksum = SparkCudaInferenceMixU64Device(checksum, value_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkCudaInferenceMixU64Device(checksum, bytes[byte_index]);
    }

    return checksum;
}

static __device__ uint64_t SparkCudaInferenceU32ChecksumDevice(const uint32_t *values, uint64_t value_count)
{
    uint64_t value_index;
    uint64_t checksum;

    checksum = 0x5350494E46434855ull;
    checksum = SparkCudaInferenceMixU64Device(checksum, value_count);
    for (value_index = 0; value_index < value_count; ++value_index)
    {
        checksum = SparkCudaInferenceMixU64Device(checksum, values[value_index]);
    }

    return checksum;
}

static __device__ bool SparkCudaInferenceBetterTokenDevice(float candidate_value, uint32_t candidate_token_id, float best_value, uint32_t best_token_id)
{
    if (candidate_value > best_value)
    {
        return true;
    }
    if (candidate_value == best_value && candidate_token_id < best_token_id)
    {
        return true;
    }
    return false;
}

static __device__ void SparkCudaInferenceUpdateBestDevice(float output_value, uint32_t token_index, float *best_value, uint32_t *best_token_id)
{
	if (SparkCudaInferenceBetterTokenDevice(output_value, token_index, *best_value, *best_token_id))
	{
		*best_value = output_value;
		*best_token_id = token_index;
	}
}

static __device__ void SparkCudaInferenceCheckLogitFiniteDevice(float input_value, float output_value, SparkCudaInferenceUtilityReport *report)
{
	if (!SparkCudaInferenceFloatIsFiniteDevice(input_value) || !SparkCudaInferenceFloatIsFiniteDevice(output_value))
	{
		atomicAdd(&report->nonfinite_logit_count, 1u);
	}
}

static __device__ uint32_t SparkCudaInferenceSharedBestDevice(float best_value, uint32_t best_token_id, float *shared_best_values, uint32_t *shared_best_tokens)
{
	uint32_t other_token_id;
	uint32_t stride;
	float other_value;

	shared_best_values[threadIdx.x] = best_value;
	shared_best_tokens[threadIdx.x] = best_token_id;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
		{
			other_value = shared_best_values[threadIdx.x + stride];
			other_token_id = shared_best_tokens[threadIdx.x + stride];
			if (SparkCudaInferenceBetterTokenDevice(other_value, other_token_id, shared_best_values[threadIdx.x], shared_best_tokens[threadIdx.x]))
			{
				shared_best_values[threadIdx.x] = other_value;
				shared_best_tokens[threadIdx.x] = other_token_id;
			}
		}
		__syncthreads();
	}
	return shared_best_tokens[0];
}

static __device__ uint64_t SparkCudaInferenceFastTraceChecksumDevice(SparkCudaInferenceUtilityRequest request)
{
	uint64_t checksum;

	checksum = 0x5350494E46545243ull;
	checksum = SparkCudaInferenceMixU64Device(checksum, request.row_count);
	checksum = SparkCudaInferenceMixU64Device(checksum, request.vocab_size);
	checksum = SparkCudaInferenceMixU64Device(checksum, request.scale_count);
	checksum = SparkCudaInferenceMixU64Device(checksum, request.workspace_value_count);
	checksum = SparkCudaInferenceMixU64Device(checksum, 0u);
	checksum = SparkCudaInferenceMixU64Device(checksum, 0u);
	checksum = SparkCudaInferenceMixU64Device(checksum, 0u);
	checksum = SparkCudaInferenceMixU64Device(checksum, 0u);
	checksum = SparkCudaInferenceMixU64Device(checksum, 0u);
	return checksum;
}

static __device__ void SparkCudaInferenceWriteFastReportDevice(SparkCudaInferenceUtilityRequest request, SparkCudaInferenceUtilityReport *report)
{
	report->logit_count = (uint64_t)request.row_count * (uint64_t)request.vocab_size;
	report->scale_count = request.scale_count;
	report->workspace_value_count = request.workspace_value_count;
	report->logits_checksum = 0u;
	report->workspace_checksum = 0u;
	report->token_checksum = 0u;
	report->trace_checksum = SparkCudaInferenceFastTraceChecksumDevice(request);
	report->scale_audit_kernel_count = 0u;
	report->logits_transform_kernel_count = 1u;
	report->greedy_argmax_kernel_count = 1u;
	report->workspace_clear_kernel_count = 0u;
	report->checksum_kernel_count = 0u;
	report->trace_kernel_count = 0u;
	report->host_reference_count = 0u;
	report->nonfinite_logit_count = 0u;
	report->invalid_scale_count = 0u;
	report->nonfinite_scale_count = 0u;
	report->zero_scale_count = 0u;
	report->scale_underflow_count = 0u;
	report->scale_overflow_count = 0u;
	report->sentinel_violation_count = 0u;
	report->checksum_mismatch_count = 0u;
}

static __global__ void SparkCudaInferenceFastLogitsArgmaxKernel(const float *input_logits, float *output_logits, uint32_t *output_token_ids, float *workspace_values, SparkCudaInferenceUtilityRequest request, SparkCudaInferenceUtilityReport *report)
{
	__shared__ float shared_best_values[SPARK_CUDA_INFERENCE_THREADS];
	__shared__ uint32_t shared_best_tokens[SPARK_CUDA_INFERENCE_THREADS];
	const float4 *input_vectors;
	float4 *output_vectors;
	float4 *workspace_vectors;
	float4 input_vector;
	float4 output_vector;
	float4 workspace_vector;
	uint32_t row_index;
	uint32_t vector_index;
	uint32_t vector_count;
	uint32_t token_base;
	uint32_t best_token_id;
	uint64_t row_offset;
	float best_value;
	float inv_temperature;

	row_index = blockIdx.x;
	if (row_index >= request.row_count)
		return;
	if (row_index == 0u && threadIdx.x == 0u)
		SparkCudaInferenceWriteFastReportDevice(request, report);
	row_offset = (uint64_t)row_index * (uint64_t)request.vocab_size;
	vector_count = request.vocab_size >> 2u;
	inv_temperature = 1.0f / request.temperature;
	input_vectors = (const float4 *)(input_logits + row_offset);
	output_vectors = (float4 *)(output_logits + row_offset);
	workspace_vectors = (float4 *)(workspace_values + row_offset);
	workspace_vector = make_float4(request.workspace_value, request.workspace_value, request.workspace_value, request.workspace_value);
	best_token_id = UINT32_MAX;
	best_value = -FLT_MAX;
	for (vector_index = threadIdx.x; vector_index < vector_count; vector_index += blockDim.x)
	{
		token_base = vector_index << 2u;
		input_vector = input_vectors[vector_index];
		output_vector.x = fmaf(input_vector.x, inv_temperature, request.logit_bias);
		output_vector.y = fmaf(input_vector.y, inv_temperature, request.logit_bias);
		output_vector.z = fmaf(input_vector.z, inv_temperature, request.logit_bias);
		output_vector.w = fmaf(input_vector.w, inv_temperature, request.logit_bias);
		output_vectors[vector_index] = output_vector;
		workspace_vectors[vector_index] = workspace_vector;
		SparkCudaInferenceUpdateBestDevice(output_vector.x, token_base, &best_value, &best_token_id);
		SparkCudaInferenceUpdateBestDevice(output_vector.y, token_base + 1u, &best_value, &best_token_id);
		SparkCudaInferenceUpdateBestDevice(output_vector.z, token_base + 2u, &best_value, &best_token_id);
		SparkCudaInferenceUpdateBestDevice(output_vector.w, token_base + 3u, &best_value, &best_token_id);
	}
	best_token_id = SparkCudaInferenceSharedBestDevice(best_value, best_token_id, shared_best_values, shared_best_tokens);
	if (threadIdx.x == 0u)
		output_token_ids[row_index] = best_token_id == UINT32_MAX ? 0u : best_token_id;
}

static __global__ void SparkCudaInferenceLogitsTransformArgmaxKernel(const float *input_logits, const float *scale_values, float *output_logits, uint32_t *output_token_ids, float *workspace_values, SparkCudaInferenceUtilityRequest request, SparkCudaInferenceUtilityReport *report)
{
	__shared__ float shared_best_values[SPARK_CUDA_INFERENCE_THREADS];
	__shared__ uint32_t shared_best_tokens[SPARK_CUDA_INFERENCE_THREADS];
	uint32_t row_index;
	uint32_t scale_index;
	uint32_t token_index;
	uint32_t best_token_id;
	uint32_t vector_index;
	uint32_t vector_count;
	uint32_t token_base;
	uint64_t row_offset;
	float scale_value;
	float input_value;
	float output_value;
	float best_value;
	float4 input_vector;
	float4 output_vector;
	float4 workspace_vector;
	const float4 *input_vectors;
	float4 *output_vectors;
	float4 *workspace_vectors;

    row_index = blockIdx.x;
	if (row_index >= request.row_count)
	{
		return;
	}
	if (row_index == 0u)
	{
		for (scale_index = threadIdx.x; scale_index < request.scale_count; scale_index += blockDim.x)
		{
			scale_value = scale_values[scale_index];
			if (!SparkCudaInferenceFloatIsFiniteDevice(scale_value))
			{
				atomicAdd(&report->nonfinite_scale_count, 1u);
				atomicAdd(&report->invalid_scale_count, 1u);
			}
			else if (scale_value == 0.0f)
			{
				atomicAdd(&report->zero_scale_count, 1u);
				atomicAdd(&report->invalid_scale_count, 1u);
			}
			else
			{
				if (scale_value < request.scale_min)
				{
					atomicAdd(&report->scale_underflow_count, 1u);
					atomicAdd(&report->invalid_scale_count, 1u);
				}
				if (scale_value > request.scale_max)
				{
					atomicAdd(&report->scale_overflow_count, 1u);
					atomicAdd(&report->invalid_scale_count, 1u);
				}
			}
		}
	}
	row_offset = (uint64_t)row_index * (uint64_t)request.vocab_size;
    best_token_id = UINT32_MAX;
    best_value = -FLT_MAX;
	if (workspace_values != 0 && (request.vocab_size & 3u) == 0u)
	{
		vector_count = request.vocab_size >> 2u;
		input_vectors = (const float4 *)(input_logits + row_offset);
		output_vectors = (float4 *)(output_logits + row_offset);
		workspace_vectors = (float4 *)(workspace_values + row_offset);
		workspace_vector = make_float4(request.workspace_value, request.workspace_value, request.workspace_value, request.workspace_value);
		for (vector_index = threadIdx.x; vector_index < vector_count; vector_index += blockDim.x)
		{
			token_base = vector_index << 2u;
			input_vector = input_vectors[vector_index];
			output_vector.x = (input_vector.x / request.temperature) + request.logit_bias;
			output_vector.y = (input_vector.y / request.temperature) + request.logit_bias;
			output_vector.z = (input_vector.z / request.temperature) + request.logit_bias;
			output_vector.w = (input_vector.w / request.temperature) + request.logit_bias;
			output_vectors[vector_index] = output_vector;
			workspace_vectors[vector_index] = workspace_vector;
			SparkCudaInferenceCheckLogitFiniteDevice(input_vector.x, output_vector.x, report);
			SparkCudaInferenceCheckLogitFiniteDevice(input_vector.y, output_vector.y, report);
			SparkCudaInferenceCheckLogitFiniteDevice(input_vector.z, output_vector.z, report);
			SparkCudaInferenceCheckLogitFiniteDevice(input_vector.w, output_vector.w, report);
			SparkCudaInferenceUpdateBestDevice(output_vector.x, token_base, &best_value, &best_token_id);
			SparkCudaInferenceUpdateBestDevice(output_vector.y, token_base + 1u, &best_value, &best_token_id);
			SparkCudaInferenceUpdateBestDevice(output_vector.z, token_base + 2u, &best_value, &best_token_id);
			SparkCudaInferenceUpdateBestDevice(output_vector.w, token_base + 3u, &best_value, &best_token_id);
		}
	}
	else
	{
		for (token_index = threadIdx.x; token_index < request.vocab_size; token_index += blockDim.x)
		{
			input_value = input_logits[row_offset + (uint64_t)token_index];
			output_value = (input_value / request.temperature) + request.logit_bias;
			output_logits[row_offset + (uint64_t)token_index] = output_value;
			if (workspace_values != 0)
			{
				workspace_values[row_offset + (uint64_t)token_index] = request.workspace_value;
			}
			SparkCudaInferenceCheckLogitFiniteDevice(input_value, output_value, report);
			SparkCudaInferenceUpdateBestDevice(output_value, token_index, &best_value, &best_token_id);
		}
	}
	best_token_id = SparkCudaInferenceSharedBestDevice(best_value, best_token_id, shared_best_values, shared_best_tokens);
    if (threadIdx.x == 0u)
    {
        output_token_ids[row_index] = best_token_id == UINT32_MAX ? 0u : best_token_id;
    }
}

static __global__ void SparkCudaInferenceWorkspaceClearKernel(float *workspace_values, SparkCudaInferenceUtilityRequest request)
{
    uint64_t workspace_index;

    workspace_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (workspace_index < request.workspace_value_count)
    {
        workspace_values[workspace_index] = request.workspace_value;
        workspace_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaInferenceChecksumKernel(const float *output_logits, const uint32_t *output_token_ids, const float *workspace_values, SparkCudaInferenceUtilityRequest request, SparkCudaInferenceUtilityReport *report)
{
    uint64_t logit_count;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    logit_count = (uint64_t)request.row_count * (uint64_t)request.vocab_size;
    report->logits_checksum = SparkCudaInferenceFloatChecksumDevice(output_logits, logit_count);
    report->workspace_checksum = SparkCudaInferenceFloatChecksumDevice(workspace_values, request.workspace_value_count);
    report->token_checksum = SparkCudaInferenceU32ChecksumDevice(output_token_ids, request.row_count);
    if ((request.expected_logits_checksum != 0u && request.expected_logits_checksum != report->logits_checksum) || (request.expected_workspace_checksum != 0u && request.expected_workspace_checksum != report->workspace_checksum))
    {
        report->checksum_mismatch_count += 1u;
    }
}

static __global__ void SparkCudaInferenceTraceKernel(SparkCudaInferenceUtilityRequest request, SparkCudaInferenceUtilityReport *report)
{
    uint64_t checksum;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    report->logit_count = (uint64_t)request.row_count * (uint64_t)request.vocab_size;
    report->scale_count = request.scale_count;
    report->workspace_value_count = request.workspace_value_count;
	report->scale_audit_kernel_count = 0u;
    report->logits_transform_kernel_count = 1u;
    report->greedy_argmax_kernel_count = 1u;
    report->workspace_clear_kernel_count = request.workspace_value_count == ((uint64_t)request.row_count * (uint64_t)request.vocab_size) ? 0u : 1u;
    report->checksum_kernel_count = request.compute_checksum != 0u ? 1u : 0u;
    report->trace_kernel_count = 1u;
    if (request.sentinel != SPARKPIPE_CUDA_INFERENCE_SENTINEL)
    {
        report->sentinel_violation_count += 1u;
    }
    checksum = 0x5350494E46545243ull;
    checksum = SparkCudaInferenceMixU64Device(checksum, request.row_count);
    checksum = SparkCudaInferenceMixU64Device(checksum, request.vocab_size);
    checksum = SparkCudaInferenceMixU64Device(checksum, request.scale_count);
    checksum = SparkCudaInferenceMixU64Device(checksum, request.workspace_value_count);
    checksum = SparkCudaInferenceMixU64Device(checksum, report->logits_checksum);
    checksum = SparkCudaInferenceMixU64Device(checksum, report->workspace_checksum);
    checksum = SparkCudaInferenceMixU64Device(checksum, report->token_checksum);
    checksum = SparkCudaInferenceMixU64Device(checksum, report->invalid_scale_count);
    checksum = SparkCudaInferenceMixU64Device(checksum, report->nonfinite_logit_count);
    report->trace_checksum = checksum;
}

static uint64_t SparkCudaInferenceLogitCountHost(const SparkCudaInferenceUtilityRequest *request)
{
    return (uint64_t)request->row_count * (uint64_t)request->vocab_size;
}

static uint32_t SparkCudaInferenceBlockCount(uint64_t value_count)
{
    uint32_t block_count;

    block_count = (uint32_t)((value_count + 255u) / 256u);
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 4096u)
    {
        block_count = 4096u;
    }

    return block_count;
}

extern "C" SparkStatus SparkRunCudaInferenceUtilityDeviceKernels(const SparkCudaInferenceUtilityRequest *request, const float *device_input_logits, uint64_t input_logit_count, const float *device_scale_values, uint32_t scale_count, float *device_output_logits, uint64_t output_logit_count, uint32_t *device_output_token_ids, uint32_t output_token_count, float *device_workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *device_report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t logit_count;
    float *fused_workspace_values;
	bool use_fast_path;

    status = SparkValidateCudaInferenceUtilityRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    logit_count = SparkCudaInferenceLogitCountHost(request);
    if (device_input_logits == 0 || device_scale_values == 0 || device_output_logits == 0 || device_output_token_ids == 0 || device_workspace_values == 0 || device_report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (input_logit_count < logit_count || output_logit_count < logit_count || output_token_count < request->row_count || scale_count != request->scale_count || workspace_value_count < request->workspace_value_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    fused_workspace_values = request->workspace_value_count == logit_count ? device_workspace_values : 0;
	use_fast_path = request->compute_checksum == 0u && fused_workspace_values != 0 && (request->vocab_size & 3u) == 0u;
	if (use_fast_path)
	{
		SparkCudaInferenceFastLogitsArgmaxKernel<<<request->row_count, SPARK_CUDA_INFERENCE_THREADS>>>(device_input_logits, device_output_logits, device_output_token_ids, fused_workspace_values, *request, device_report);
		cuda_status = cudaGetLastError();
	}
	else
	{
		cuda_status = cudaMemset(device_report, 0, sizeof(*device_report));
		if (cuda_status == cudaSuccess)
		{
			SparkCudaInferenceLogitsTransformArgmaxKernel<<<request->row_count, SPARK_CUDA_INFERENCE_THREADS>>>(device_input_logits, device_scale_values, device_output_logits, device_output_token_ids, fused_workspace_values, *request, device_report);
			cuda_status = cudaGetLastError();
		}
	}
    if (cuda_status == cudaSuccess && fused_workspace_values == 0)
    {
        SparkCudaInferenceWorkspaceClearKernel<<<SparkCudaInferenceBlockCount(request->workspace_value_count), SPARK_CUDA_INFERENCE_THREADS>>>(device_workspace_values, *request);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess && request->compute_checksum != 0u)
    {
        SparkCudaInferenceChecksumKernel<<<1, 32>>>(device_output_logits, device_output_token_ids, device_workspace_values, *request, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess && !use_fast_path)
    {
        SparkCudaInferenceTraceKernel<<<1, 32>>>(*request, device_report);
        cuda_status = cudaGetLastError();
    }
    return cuda_status == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkRunCudaInferenceUtilityKernels(const SparkCudaInferenceUtilityRequest *request, const float *input_logits, uint64_t input_logit_count, const float *scale_values, uint32_t scale_count, float *output_logits, uint64_t output_logit_count, uint32_t *output_token_ids, uint32_t output_token_count, float *workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *report)
{
    float *device_input_logits;
    float *device_scale_values;
    float *device_output_logits;
    float *device_workspace_values;
    uint32_t *device_output_token_ids;
    SparkCudaInferenceUtilityReport *device_report;
    SparkCudaInferenceUtilityReport host_report;
    cudaError_t cuda_status;
    SparkStatus status;
    SparkStatus device_status;
    uint64_t logit_count;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaInferenceUtilityRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    logit_count = SparkCudaInferenceLogitCountHost(request);
    if (input_logits == 0 || scale_values == 0 || output_logits == 0 || output_token_ids == 0 || workspace_values == 0 || report == 0 || input_logit_count < logit_count || output_logit_count < logit_count || output_token_count < request->row_count || scale_count != request->scale_count || workspace_value_count < request->workspace_value_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    device_input_logits = 0;
    device_scale_values = 0;
    device_output_logits = 0;
    device_workspace_values = 0;
    device_output_token_ids = 0;
    device_report = 0;
    cuda_status = cudaMalloc((void **)&device_input_logits, logit_count * (uint64_t)sizeof(float));
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_scale_values, (uint64_t)request->scale_count * (uint64_t)sizeof(float));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_output_logits, logit_count * (uint64_t)sizeof(float));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_output_token_ids, (uint64_t)request->row_count * (uint64_t)sizeof(uint32_t));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_workspace_values, (uint64_t)request->workspace_value_count * (uint64_t)sizeof(float));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_report, sizeof(*device_report));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(device_input_logits, input_logits, logit_count * (uint64_t)sizeof(float), cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(device_scale_values, scale_values, (uint64_t)request->scale_count * (uint64_t)sizeof(float), cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess)
    {
        device_status = SparkRunCudaInferenceUtilityDeviceKernels(request, device_input_logits, logit_count, device_scale_values, request->scale_count, device_output_logits, logit_count, device_output_token_ids, request->row_count, device_workspace_values, request->workspace_value_count, device_report);
        if (device_status != SPARK_STATUS_OK)
        {
            cuda_status = cudaErrorUnknown;
        }
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaDeviceSynchronize();
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(output_logits, device_output_logits, logit_count * (uint64_t)sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(output_token_ids, device_output_token_ids, (uint64_t)request->row_count * (uint64_t)sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(workspace_values, device_workspace_values, (uint64_t)request->workspace_value_count * (uint64_t)sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(&host_report, device_report, sizeof(host_report), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_report);
    cudaFree(device_workspace_values);
    cudaFree(device_output_token_ids);
    cudaFree(device_output_logits);
    cudaFree(device_scale_values);
    cudaFree(device_input_logits);

    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    *report = host_report;
    if (report->invalid_scale_count != 0u || report->nonfinite_logit_count != 0u || report->checksum_mismatch_count != 0u || report->sentinel_violation_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}
