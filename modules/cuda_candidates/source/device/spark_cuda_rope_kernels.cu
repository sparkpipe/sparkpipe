#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_rope_kernels.h"

#define SPARK_CUDA_ROPE_THREADS 256u

static __device__ float SparkCudaRopeBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ uint16_t SparkCudaRopeFloatToBf16(float value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;
    uint32_t rounding_bias;

    bits.f = value;
    rounding_bias = 0x7fffu + ((bits.u >> 16u) & 1u);
    return (uint16_t)((bits.u + rounding_bias) >> 16u);
}

static uint64_t SparkCudaRopeElementCountHost(const SparkCudaRopeRequest *request)
{
    return (uint64_t)request->token_count * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaRopePairCountHost(const SparkCudaRopeRequest *request)
{
    return (uint64_t)request->token_count * (uint64_t)request->head_count * ((uint64_t)request->rotary_dim / 2u);
}

static uint32_t SparkCudaRopeBlockCountHost(uint64_t element_count)
{
    uint32_t block_count;

    block_count = (uint32_t)((element_count + (uint64_t)SPARK_CUDA_ROPE_THREADS - 1u) / (uint64_t)SPARK_CUDA_ROPE_THREADS);
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 65535u)
    {
        block_count = 65535u;
    }
    return block_count;
}

static void SparkCudaRopeFillReportShape(const SparkCudaRopeRequest *request, SparkCudaRopeReport *report)
{
    uint64_t rotated_element_count;

    report->element_count = SparkCudaRopeElementCountHost(request);
    report->rotary_pair_count = SparkCudaRopePairCountHost(request);
    rotated_element_count = report->rotary_pair_count * 2u;
    report->copy_element_count = report->element_count >= rotated_element_count ? report->element_count - rotated_element_count : 0u;
}

static __global__ void SparkCudaRopeBf16PairKernel(SparkCudaRopeRequest request, const uint16_t *input_values, const uint32_t *positions, const float *cos_table, const float *sin_table, uint16_t *output_values)
{
    uint64_t pair_global_index;
    uint64_t total_pairs;
    uint64_t row_index;
    uint64_t pair_index;
    uint64_t row_offset;
    uint64_t table_offset;
    uint32_t pair_count;
    uint32_t position;
    float x0;
    float x1;
    float cos_value;
    float sin_value;

    pair_count = request.rotary_dim >> 1u;
    total_pairs = (uint64_t)request.token_count * (uint64_t)request.head_count * (uint64_t)pair_count;
    pair_global_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (pair_global_index < total_pairs)
    {
        row_index = pair_global_index / (uint64_t)pair_count;
        pair_index = pair_global_index - (row_index * (uint64_t)pair_count);
        row_offset = row_index * (uint64_t)request.head_size;
        position = positions[row_index / (uint64_t)request.head_count];
        if (position < request.position_count)
        {
            table_offset = ((uint64_t)position * (uint64_t)pair_count) + pair_index;
            x0 = SparkCudaRopeBf16ToFloat(input_values[row_offset + (pair_index * 2u)]);
            x1 = SparkCudaRopeBf16ToFloat(input_values[row_offset + (pair_index * 2u) + 1u]);
            cos_value = cos_table[table_offset];
            sin_value = sin_table[table_offset];
            output_values[row_offset + (pair_index * 2u)] = SparkCudaRopeFloatToBf16((x0 * cos_value) - (x1 * sin_value));
            output_values[row_offset + (pair_index * 2u) + 1u] = SparkCudaRopeFloatToBf16((x0 * sin_value) + (x1 * cos_value));
        }
        pair_global_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaRopeBf16TailCopyKernel(SparkCudaRopeRequest request, const uint16_t *input_values, uint16_t *output_values)
{
    uint64_t copy_index;
    uint64_t copy_count;
    uint64_t row_index;
    uint64_t tail_dim;
    uint64_t dim_index;

    tail_dim = (uint64_t)(request.head_size - request.rotary_dim);
    copy_count = (uint64_t)request.token_count * (uint64_t)request.head_count * tail_dim;
    copy_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (copy_index < copy_count)
    {
        row_index = copy_index / tail_dim;
        dim_index = request.rotary_dim + (copy_index - (row_index * tail_dim));
        output_values[(row_index * (uint64_t)request.head_size) + dim_index] = input_values[(row_index * (uint64_t)request.head_size) + dim_index];
        copy_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

extern "C" SparkStatus SparkRunCudaRopeBf16(const SparkCudaRopeRequest *request, const void *device_input_bf16, const uint32_t *device_positions, const float *device_cos_table, const float *device_sin_table, void *device_output_bf16, SparkCudaRopeReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRopeRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_positions == 0 || device_cos_table == 0 || device_sin_table == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaRopeFillReportShape(request, report);
    SparkCudaRopeBf16PairKernel<<<SparkCudaRopeBlockCountHost(report->rotary_pair_count), SPARK_CUDA_ROPE_THREADS>>>(*request, (const uint16_t *)device_input_bf16, device_positions, device_cos_table, device_sin_table, (uint16_t *)device_output_bf16);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->standard_rope_kernel_count = 1u;
    if (report->copy_element_count != 0u)
    {
        SparkCudaRopeBf16TailCopyKernel<<<SparkCudaRopeBlockCountHost(report->copy_element_count), SPARK_CUDA_ROPE_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (uint16_t *)device_output_bf16);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        report->standard_rope_kernel_count += 1u;
    }
    return SPARK_STATUS_OK;
}
