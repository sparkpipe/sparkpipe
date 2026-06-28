#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_structured_mask.h"

static __device__ uint64_t SparkCudaStructuredMaskMixU64Device(uint64_t checksum, uint64_t value)
{
    checksum ^= value + 0x9e3779b97f4a7c15ull + (checksum << 6u) + (checksum >> 2u);
    return checksum;
}

static __device__ bool SparkCudaStructuredMaskTokenIsDuplicateDevice(const uint32_t *allowed_tokens, uint32_t start_index, uint32_t token_index, uint32_t token_id)
{
    uint32_t compare_index;

    for (compare_index = start_index; compare_index < token_index; ++compare_index)
    {
        if (allowed_tokens[compare_index] == token_id)
        {
            return true;
        }
    }

    return false;
}

static __device__ uint64_t SparkCudaStructuredMaskChecksumDevice(const float *values, uint64_t value_count)
{
    const uint8_t *bytes;
    uint64_t byte_count;
    uint64_t byte_index;
    uint64_t checksum;

    bytes = (const uint8_t *)values;
    byte_count = value_count * (uint64_t)sizeof(float);
    checksum = 0x53504B4D43484B38ull;
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, value_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkCudaStructuredMaskMixU64Device(checksum, bytes[byte_index]);
    }

    return checksum;
}

static __global__ void SparkCudaStructuredMaskFillKernel(float *masked_logits, uint64_t logit_count, float mask_value)
{
    uint64_t logit_index;

    logit_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (logit_index < logit_count)
    {
        masked_logits[logit_index] = mask_value;
        logit_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaStructuredMaskScatterKernel(const float *input_logits, float *masked_logits, const uint32_t *allowed_offsets, const uint32_t *allowed_tokens, SparkCudaStructuredMaskRequest request, SparkCudaStructuredMaskReport *report)
{
    uint32_t row_index;
    uint32_t token_index;
    uint32_t token_id;
    uint32_t row_start;
    uint32_t row_end;

    row_index = blockIdx.x;
    if (row_index >= request.row_count)
    {
        return;
    }

    row_start = allowed_offsets[row_index];
    row_end = allowed_offsets[row_index + 1u];
    if (threadIdx.x == 0 && row_start == row_end)
    {
        atomicAdd(&report->empty_row_count, 1u);
    }
    for (token_index = row_start + threadIdx.x; token_index < row_end; token_index += blockDim.x)
    {
        token_id = allowed_tokens[token_index];
        if (token_id >= request.vocab_size)
        {
            atomicAdd(&report->invalid_token_count, 1u);
            continue;
        }
        if (SparkCudaStructuredMaskTokenIsDuplicateDevice(allowed_tokens, row_start, token_index, token_id))
        {
            atomicAdd(&report->duplicate_token_count, 1u);
            continue;
        }
        masked_logits[((uint64_t)row_index * (uint64_t)request.vocab_size) + token_id] = input_logits[((uint64_t)row_index * (uint64_t)request.vocab_size) + token_id];
        atomicAdd((unsigned long long *)&report->unique_allowed_token_count, 1ull);
	}
}

static __global__ void SparkCudaStructuredMaskCandidateKernel(const float *input_logits, float *candidate_logits, const uint32_t *allowed_offsets, const uint32_t *allowed_tokens, SparkCudaStructuredMaskRequest request, SparkCudaStructuredMaskReport *report)
{
	uint32_t row_index;
	uint32_t token_index;
	uint32_t token_id;
	uint32_t row_start;
	uint32_t row_end;

	row_index = blockIdx.x;
	if (row_index >= request.row_count)
	{
		return;
	}
	row_start = allowed_offsets[row_index];
	row_end = allowed_offsets[row_index + 1u];
	if (threadIdx.x == 0u && row_start == row_end)
	{
		atomicAdd(&report->empty_row_count, 1u);
	}
	for (token_index = row_start + threadIdx.x; token_index < row_end; token_index += blockDim.x)
	{
		token_id = allowed_tokens[token_index];
		if (token_id >= request.vocab_size)
		{
			candidate_logits[token_index] = request.mask_value;
			atomicAdd(&report->invalid_token_count, 1u);
			continue;
		}
		if (SparkCudaStructuredMaskTokenIsDuplicateDevice(allowed_tokens, row_start, token_index, token_id))
		{
			candidate_logits[token_index] = request.mask_value;
			atomicAdd(&report->duplicate_token_count, 1u);
			continue;
		}
		candidate_logits[token_index] = input_logits[((uint64_t)row_index * (uint64_t)request.vocab_size) + token_id];
		atomicAdd((unsigned long long *)&report->unique_allowed_token_count, 1ull);
	}
}

static __global__ void SparkCudaStructuredMaskChecksumKernel(const float *masked_logits, SparkCudaStructuredMaskRequest request, SparkCudaStructuredMaskReport *report)
{
    uint64_t checksum;
    uint64_t logit_count;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    logit_count = (uint64_t)request.row_count * (uint64_t)request.vocab_size;
    checksum = SparkCudaStructuredMaskChecksumDevice(masked_logits, logit_count);
    report->output_checksum = checksum;
    if (request.expected_output_checksum != 0u && request.expected_output_checksum != checksum)
    {
        report->checksum_mismatch_count += 1u;
    }
}

static __global__ void SparkCudaStructuredMaskTraceKernel(SparkCudaStructuredMaskRequest request, SparkCudaStructuredMaskReport *report)
{
    uint64_t checksum;
    uint64_t logit_count;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    logit_count = (uint64_t)request.row_count * (uint64_t)request.vocab_size;
    report->logit_count = logit_count;
    report->allowed_token_count = request.allowed_token_count;
    report->masked_token_count = logit_count - report->unique_allowed_token_count;
    report->fill_kernel_count = 1u;
    report->scatter_kernel_count = 1u;
    report->checksum_kernel_count = request.compute_checksum != 0u ? 1u : 0u;
    report->trace_kernel_count = 1u;
    if (request.sentinel != SPARKPIPE_CUDA_STRUCTURED_MASK_SENTINEL)
    {
        report->sentinel_violation_count += 1u;
    }
    checksum = 0x53504B4D41534B54ull;
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.row_count);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.vocab_size);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.allowed_token_count);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.max_allowed_tokens_per_row);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->unique_allowed_token_count);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->masked_token_count);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->output_checksum);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->invalid_token_count);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->duplicate_token_count);
    checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->empty_row_count);
	report->trace_checksum = checksum;
}

static __global__ void SparkCudaStructuredMaskCandidateTraceKernel(SparkCudaStructuredMaskRequest request, SparkCudaStructuredMaskReport *report)
{
	uint64_t checksum;
	uint64_t logit_count;

	if (blockIdx.x != 0 || threadIdx.x != 0)
	{
		return;
	}
	logit_count = (uint64_t)request.row_count * (uint64_t)request.vocab_size;
	report->logit_count = logit_count;
	report->allowed_token_count = request.allowed_token_count;
	report->masked_token_count = logit_count - report->unique_allowed_token_count;
	report->fill_kernel_count = 0u;
	report->scatter_kernel_count = 1u;
	report->checksum_kernel_count = request.compute_checksum != 0u ? 1u : 0u;
	report->trace_kernel_count = 1u;
	if (request.sentinel != SPARKPIPE_CUDA_STRUCTURED_MASK_SENTINEL)
	{
		report->sentinel_violation_count += 1u;
	}
	checksum = 0x53504B4D41534B54ull;
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.row_count);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.vocab_size);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.allowed_token_count);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, request.max_allowed_tokens_per_row);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->unique_allowed_token_count);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->masked_token_count);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->output_checksum);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->invalid_token_count);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->duplicate_token_count);
	checksum = SparkCudaStructuredMaskMixU64Device(checksum, report->empty_row_count);
	report->trace_checksum = checksum;
}

static uint64_t SparkCudaStructuredMaskLogitCountHost(const SparkCudaStructuredMaskRequest *request)
{
    return (uint64_t)request->row_count * (uint64_t)request->vocab_size;
}

static uint32_t SparkCudaStructuredMaskBlockCount(uint64_t logit_count)
{
    uint32_t block_count;

    block_count = (uint32_t)((logit_count + 255u) / 256u);
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

static SparkStatus SparkCudaStructuredMaskValidateDeviceRequestHost(const SparkCudaStructuredMaskRequest *request, const float *device_input_logits, uint64_t input_value_count, const uint32_t *device_allowed_offsets, uint32_t allowed_offset_count, const uint32_t *device_allowed_tokens, uint32_t allowed_token_count, float *device_candidate_logits, uint32_t candidate_value_count, SparkCudaStructuredMaskReport *device_report)
{
	uint64_t logit_count;

	if (request == 0 || device_input_logits == 0 || device_allowed_offsets == 0 || device_allowed_tokens == 0 || device_candidate_logits == 0 || device_report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->row_count == 0 || request->row_count > SPARKPIPE_MAX_PHYSICAL_SLOTS || request->vocab_size == 0 || request->allowed_token_count == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->max_allowed_tokens_per_row == 0 || request->max_allowed_tokens_per_row > SPARKPIPE_MAX_CONSTRAINT_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->mask_value >= 0.0f || request->mask_value != request->mask_value)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->sentinel != SPARKPIPE_CUDA_STRUCTURED_MASK_SENTINEL || allowed_offset_count < request->row_count + 1u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->compute_checksum != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	logit_count = SparkCudaStructuredMaskLogitCountHost(request);
	if (input_value_count < logit_count || allowed_token_count != request->allowed_token_count || candidate_value_count < request->allowed_token_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaStructuredMaskCandidateDeviceKernels(const SparkCudaStructuredMaskRequest *request, const float *device_input_logits, uint64_t input_value_count, const uint32_t *device_allowed_offsets, uint32_t allowed_offset_count, const uint32_t *device_allowed_tokens, uint32_t allowed_token_count, float *device_candidate_logits, uint32_t candidate_value_count, SparkCudaStructuredMaskReport *device_report)
{
	cudaError_t cuda_status;
	SparkStatus status;

	status = SparkCudaStructuredMaskValidateDeviceRequestHost(request, device_input_logits, input_value_count, device_allowed_offsets, allowed_offset_count, device_allowed_tokens, allowed_token_count, device_candidate_logits, candidate_value_count, device_report);
	if (status != SPARK_STATUS_OK)
		return status;
	cuda_status = cudaMemset(device_report, 0, sizeof(*device_report));
	if (cuda_status == cudaSuccess)
	{
		SparkCudaStructuredMaskCandidateKernel<<<request->row_count, 128>>>(device_input_logits, device_candidate_logits, device_allowed_offsets, device_allowed_tokens, *request, device_report);
		cuda_status = cudaGetLastError();
	}
	if (cuda_status == cudaSuccess)
	{
		SparkCudaStructuredMaskCandidateTraceKernel<<<1, 32>>>(*request, device_report);
		cuda_status = cudaGetLastError();
	}
	return cuda_status == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkRunCudaStructuredMaskKernels(const SparkCudaStructuredMaskRequest *request, const float *input_host, uint64_t input_host_values, const uint32_t *allowed_offsets, uint32_t allowed_offset_count, const uint32_t *allowed_tokens, uint32_t allowed_token_count, float *masked_host, uint64_t masked_host_values, SparkCudaStructuredMaskReport *report)
{
    float *device_input;
    float *device_masked;
    uint32_t *device_offsets;
    uint32_t *device_tokens;
    SparkCudaStructuredMaskReport *device_report;
    SparkCudaStructuredMaskReport host_report;
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t logit_count;
    uint32_t block_count;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaStructuredMaskRequest(request, allowed_offsets, allowed_offset_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    logit_count = SparkCudaStructuredMaskLogitCountHost(request);
    if (input_host == 0 || allowed_tokens == 0 || masked_host == 0 || report == 0 || input_host_values < logit_count || masked_host_values < logit_count || allowed_token_count != request->allowed_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    device_input = 0;
    device_masked = 0;
    device_offsets = 0;
    device_tokens = 0;
    device_report = 0;
    cuda_status = cudaMalloc((void **)&device_input, logit_count * (uint64_t)sizeof(float));
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_masked, logit_count * (uint64_t)sizeof(float));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_offsets, (uint64_t)allowed_offset_count * (uint64_t)sizeof(uint32_t));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_tokens, (uint64_t)allowed_token_count * (uint64_t)sizeof(uint32_t));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_report, sizeof(*device_report));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(device_input, input_host, logit_count * (uint64_t)sizeof(float), cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(device_offsets, allowed_offsets, (uint64_t)allowed_offset_count * (uint64_t)sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(device_tokens, allowed_tokens, (uint64_t)allowed_token_count * (uint64_t)sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemset(device_report, 0, sizeof(*device_report));
    }

    block_count = SparkCudaStructuredMaskBlockCount(logit_count);
    if (cuda_status == cudaSuccess)
    {
        SparkCudaStructuredMaskFillKernel<<<block_count, 256>>>(device_masked, logit_count, request->mask_value);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkCudaStructuredMaskScatterKernel<<<request->row_count, 128>>>(device_input, device_masked, device_offsets, device_tokens, *request, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess && request->compute_checksum != 0u)
    {
        SparkCudaStructuredMaskChecksumKernel<<<1, 32>>>(device_masked, *request, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkCudaStructuredMaskTraceKernel<<<1, 32>>>(*request, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaDeviceSynchronize();
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(masked_host, device_masked, logit_count * (uint64_t)sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(&host_report, device_report, sizeof(host_report), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_report);
    cudaFree(device_tokens);
    cudaFree(device_offsets);
    cudaFree(device_masked);
    cudaFree(device_input);

    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    *report = host_report;
    if (report->invalid_token_count != 0u || report->duplicate_token_count != 0u || report->empty_row_count != 0u || report->checksum_mismatch_count != 0u || report->sentinel_violation_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}
