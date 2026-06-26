#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_activation_kernels.h"

#define SPARK_CUDA_ACTIVATION_THREADS 256u
#define SPARK_CUDA_ACTIVATION_FAST_GRID_X_MAX 65535u

static __device__ float SparkCudaActivationBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ uint16_t SparkCudaActivationFloatToBf16(float value)
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

static __device__ uint32_t SparkCudaActivationPackBf16Pair(float low_value, float high_value)
{
    uint32_t low_bits;
    uint32_t high_bits;

    low_bits = (uint32_t)SparkCudaActivationFloatToBf16(low_value);
    high_bits = (uint32_t)SparkCudaActivationFloatToBf16(high_value);
    return low_bits | (high_bits << 16u);
}

static __device__ float SparkCudaActivationSilu(float value)
{
    return value / (1.0f + __expf(-value));
}

static __device__ float SparkCudaActivationApply(float value, SparkCudaGatedActivationKind activation_kind)
{
    float cubed;
    float inner;

    if (activation_kind == SPARK_CUDA_GATED_ACTIVATION_SILU)
    {
        return SparkCudaActivationSilu(value);
    }
    if (activation_kind == SPARK_CUDA_GATED_ACTIVATION_GELU)
    {
        return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    }
    cubed = value * value * value;
    inner = 0.7978845608028654f * (value + (0.044715f * cubed));
    return 0.5f * value * (1.0f + tanhf(inner));
}

static __global__ void SparkCudaGatedActivationSiluContiguousBf16x2Kernel(SparkCudaActivationRequest request, const uint32_t *gate_up_pairs, uint32_t *output_pairs)
{
    uint32_t row_index;
    uint32_t pair_index;
    uint32_t hidden_pairs;
    uint64_t input_pair_offset;
    uint64_t output_pair_offset;
    uint32_t gate_pair;
    uint32_t up_pair;
    float gate_low;
    float gate_high;
    float up_low;
    float up_high;

    row_index = blockIdx.y;
    pair_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    hidden_pairs = request.hidden_size >> 1u;
    if (row_index >= request.row_count || pair_index >= hidden_pairs)
    {
        return;
    }
    input_pair_offset = ((uint64_t)row_index * (uint64_t)(request.input_stride >> 1u)) + (uint64_t)pair_index;
    output_pair_offset = ((uint64_t)row_index * (uint64_t)(request.output_stride >> 1u)) + (uint64_t)pair_index;
    gate_pair = gate_up_pairs[input_pair_offset];
    up_pair = gate_up_pairs[input_pair_offset + (uint64_t)hidden_pairs];
    gate_low = SparkCudaActivationBf16ToFloat((uint16_t)(gate_pair & 0xffffu));
    gate_high = SparkCudaActivationBf16ToFloat((uint16_t)(gate_pair >> 16u));
    up_low = SparkCudaActivationBf16ToFloat((uint16_t)(up_pair & 0xffffu));
    up_high = SparkCudaActivationBf16ToFloat((uint16_t)(up_pair >> 16u));
    output_pairs[output_pair_offset] = SparkCudaActivationPackBf16Pair(SparkCudaActivationSilu(gate_low) * up_low, SparkCudaActivationSilu(gate_high) * up_high);
}

static __global__ void SparkCudaGatedActivationBf16Kernel(SparkCudaActivationRequest request, const uint16_t *gate_up_values, uint16_t *output_values)
{
    uint64_t linear_index;
    uint32_t row_index;
    uint32_t hidden_index;
    uint64_t input_offset;
    uint64_t output_offset;
    float gate_value;
    float up_value;
    float output_value;

    linear_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (linear_index < (uint64_t)request.row_count * (uint64_t)request.hidden_size)
    {
        row_index = (uint32_t)(linear_index / request.hidden_size);
        hidden_index = (uint32_t)(linear_index - ((uint64_t)row_index * (uint64_t)request.hidden_size));
        input_offset = ((uint64_t)row_index * (uint64_t)request.input_stride) + hidden_index;
        output_offset = ((uint64_t)row_index * (uint64_t)request.output_stride) + hidden_index;
        gate_value = SparkCudaActivationBf16ToFloat(gate_up_values[input_offset]);
        up_value = SparkCudaActivationBf16ToFloat(gate_up_values[input_offset + request.hidden_size]);
        output_value = SparkCudaActivationApply(gate_value, request.activation_kind) * up_value;
        output_values[output_offset] = SparkCudaActivationFloatToBf16(output_value);
        linear_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static uint32_t SparkCudaActivationBlockCount(uint64_t element_count)
{
    uint32_t block_count;

    block_count = (uint32_t)((element_count + (SPARK_CUDA_ACTIVATION_THREADS - 1u)) / SPARK_CUDA_ACTIVATION_THREADS);
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

static uint32_t SparkCudaActivationTileCount(uint32_t pair_count)
{
    uint32_t tile_count;

    tile_count = (pair_count + (SPARK_CUDA_ACTIVATION_THREADS - 1u)) / SPARK_CUDA_ACTIVATION_THREADS;
    if (tile_count == 0u)
    {
        tile_count = 1u;
    }
    if (tile_count > SPARK_CUDA_ACTIVATION_FAST_GRID_X_MAX)
    {
        tile_count = SPARK_CUDA_ACTIVATION_FAST_GRID_X_MAX;
    }
    return tile_count;
}

static bool SparkCudaActivationCanUseContiguousSiluBf16x2(const SparkCudaActivationRequest *request)
{
    if (request->activation_kind != SPARK_CUDA_GATED_ACTIVATION_SILU)
    {
        return false;
    }
    if ((request->hidden_size & 1u) != 0u)
    {
        return false;
    }
    if (request->input_stride != (request->hidden_size * 2u))
    {
        return false;
    }
    if (request->output_stride != request->hidden_size)
    {
        return false;
    }
    return true;
}

extern "C" SparkStatus SparkRunCudaGatedActivationBf16(const SparkCudaActivationRequest *request, const void *device_gate_up_bf16, void *device_output_bf16, SparkCudaActivationReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t element_count;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaActivationRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_gate_up_bf16 == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->element_count = element_count;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    report->activation_kind = (uint32_t)request->activation_kind;
    if (request->sentinel != SPARKPIPE_CUDA_ACTIVATION_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaActivationCanUseContiguousSiluBf16x2(request))
    {
        dim3 grid(SparkCudaActivationTileCount(request->hidden_size >> 1u), request->row_count, 1u);
        SparkCudaGatedActivationSiluContiguousBf16x2Kernel<<<grid, SPARK_CUDA_ACTIVATION_THREADS>>>(*request, (const uint32_t *)device_gate_up_bf16, (uint32_t *)device_output_bf16);
    }
    else
    {
        SparkCudaGatedActivationBf16Kernel<<<SparkCudaActivationBlockCount(element_count), SPARK_CUDA_ACTIVATION_THREADS>>>(*request, (const uint16_t *)device_gate_up_bf16, (uint16_t *)device_output_bf16);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->gated_activation_kernel_count = 1u;
    return SPARK_STATUS_OK;
}
