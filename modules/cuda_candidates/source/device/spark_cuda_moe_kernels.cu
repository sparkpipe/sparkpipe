#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_moe_kernels.h"

#define SPARK_CUDA_MOE_THREADS 512u
#define SPARK_CUDA_MOE_DISPATCH_FAST_THREADS 512u
#define SPARK_CUDA_MOE_FAST_THREADS 256u

static __device__ float SparkCudaMoeBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ uint16_t SparkCudaMoeFloatToBf16(float value)
{
    union
    {
        float f;
        uint32_t u;
    } bits;
    uint32_t rounding_bias;

    bits.f = value;
    rounding_bias = 0x7fffu + ((bits.u >> 16u) & 1u);
    return (uint16_t)((bits.u + rounding_bias) >> 16u);
}

static uint64_t SparkCudaMoeRouteCountHost(const SparkCudaMoeDispatchRequest *request)
{
    return (uint64_t)request->token_count * (uint64_t)request->top_k;
}

static uint64_t SparkCudaMoeHiddenValueCountHost(const SparkCudaMoeDispatchRequest *request)
{
    return (uint64_t)request->token_count * (uint64_t)request->hidden_size;
}

static uint64_t SparkCudaMoeExpertCapacityHost(const SparkCudaMoeDispatchRequest *request)
{
    return (uint64_t)request->expert_count * (uint64_t)request->capacity_per_expert;
}

static uint32_t SparkCudaMoeBlockCount(uint64_t value_count)
{
    uint64_t block_count;

    block_count = (value_count + (uint64_t)SPARK_CUDA_MOE_THREADS - 1u) / (uint64_t)SPARK_CUDA_MOE_THREADS;
    if (block_count == 0u)
    {
        return 1u;
    }
    if (block_count > 65535u)
    {
        return 65535u;
    }
    return (uint32_t)block_count;
}

static void SparkCudaMoeFillReportShape(const SparkCudaMoeDispatchRequest *request, SparkCudaMoeDispatchReport *report)
{
    report->route_count = SparkCudaMoeRouteCountHost(request);
    report->hidden_value_count = SparkCudaMoeHiddenValueCountHost(request);
    report->expert_capacity = SparkCudaMoeExpertCapacityHost(request);
    report->error_counter_count = SPARKPIPE_CUDA_MOE_ERROR_COUNTERS;
}

static __global__ void SparkCudaMoeClearDispatchKernel(SparkCudaMoeDispatchRequest request, uint32_t *expert_counts, uint32_t *expert_offsets, uint32_t *expert_cursors, uint32_t *route_to_permuted_index, uint32_t *permuted_token_ids, uint32_t *permuted_route_ids, uint32_t *error_counters)
{
    uint64_t linear_index;
    uint64_t route_count;
    uint64_t expert_capacity;

    linear_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    route_count = (uint64_t)request.token_count * (uint64_t)request.top_k;
    expert_capacity = (uint64_t)request.expert_count * (uint64_t)request.capacity_per_expert;
    while (linear_index < expert_capacity)
    {
        permuted_token_ids[linear_index] = SPARKPIPE_CUDA_MOE_ROUTE_INVALID;
        permuted_route_ids[linear_index] = SPARKPIPE_CUDA_MOE_ROUTE_INVALID;
        linear_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
    linear_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (linear_index < route_count)
    {
        route_to_permuted_index[linear_index] = SPARKPIPE_CUDA_MOE_ROUTE_INVALID;
        linear_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
    linear_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (linear_index < request.expert_count)
    {
        expert_counts[linear_index] = 0u;
        expert_offsets[linear_index] = (uint32_t)(linear_index * (uint64_t)request.capacity_per_expert);
        expert_cursors[linear_index] = 0u;
        linear_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
    if (blockIdx.x == 0u && threadIdx.x < SPARKPIPE_CUDA_MOE_ERROR_COUNTERS)
    {
        error_counters[threadIdx.x] = 0u;
    }
    if (blockIdx.x == 0u && threadIdx.x == 0u)
    {
        expert_offsets[request.expert_count] = (uint32_t)expert_capacity;
        if (request.sentinel != SPARKPIPE_CUDA_MOE_SENTINEL)
        {
            error_counters[SPARK_CUDA_MOE_ERROR_SENTINEL] = 1u;
        }
    }
}

static __global__ void SparkCudaMoePermuteKernel(SparkCudaMoeDispatchRequest request, const uint16_t *hidden, const uint32_t *topk_ids, uint32_t *expert_counts, const uint32_t *expert_offsets, uint32_t *expert_cursors, uint32_t *route_to_permuted_index, uint32_t *permuted_token_ids, uint32_t *permuted_route_ids, uint16_t *permuted_hidden, uint32_t *error_counters)
{
    __shared__ uint32_t shared_destination;
    uint64_t route_index;
    uint32_t token_index;
    uint32_t expert_index;
    uint32_t local_slot;
    uint32_t destination;
    uint32_t hidden_index;
    uint32_t chunk_index;
    uint32_t chunk_count;

    route_index = blockIdx.x;
    if (route_index >= (uint64_t)request.token_count * (uint64_t)request.top_k)
    {
        return;
    }
    expert_index = topk_ids[route_index];
    if (threadIdx.x == 0u)
    {
        shared_destination = SPARKPIPE_CUDA_MOE_ROUTE_INVALID;
        if (expert_index < request.expert_count)
        {
            local_slot = atomicAdd(&expert_cursors[expert_index], 1u);
            atomicAdd(&expert_counts[expert_index], 1u);
            if (local_slot < request.capacity_per_expert)
            {
                shared_destination = expert_offsets[expert_index] + local_slot;
                route_to_permuted_index[route_index] = shared_destination;
            }
            else
            {
                atomicAdd(&error_counters[SPARK_CUDA_MOE_ERROR_CAPACITY], 1u);
            }
        }
        else
        {
            atomicAdd(&error_counters[SPARK_CUDA_MOE_ERROR_INVALID_EXPERT], 1u);
        }
    }
    __syncthreads();
    destination = shared_destination;
    if (destination == SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
    {
        return;
    }
    token_index = (uint32_t)(route_index / request.top_k);
    if ((request.hidden_size & 7u) == 0u)
    {
        const uint4 *hidden4;
        uint4 *permuted4;
        uint64_t source_chunk_base;
        uint64_t destination_chunk_base;

        hidden4 = (const uint4 *)hidden;
        permuted4 = (uint4 *)permuted_hidden;
        chunk_count = request.hidden_size >> 3u;
        source_chunk_base = (uint64_t)token_index * (uint64_t)chunk_count;
        destination_chunk_base = (uint64_t)destination * (uint64_t)chunk_count;
        for (chunk_index = threadIdx.x; chunk_index < chunk_count; chunk_index += blockDim.x)
        {
            permuted4[destination_chunk_base + chunk_index] = hidden4[source_chunk_base + chunk_index];
        }
    }
    else
    {
        for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
        {
            permuted_hidden[(uint64_t)destination * (uint64_t)request.hidden_size + hidden_index] = hidden[(uint64_t)token_index * (uint64_t)request.hidden_size + hidden_index];
        }
    }
    if (threadIdx.x == 0u)
    {
        permuted_token_ids[destination] = token_index;
        permuted_route_ids[destination] = (uint32_t)route_index;
    }
}

static __global__ void SparkCudaMoePermuteTop8TokenKernel(SparkCudaMoeDispatchRequest request, const uint16_t *hidden, const uint32_t *topk_ids, uint32_t *expert_counts, const uint32_t *expert_offsets, uint32_t *expert_cursors, uint32_t *route_to_permuted_index, uint32_t *permuted_token_ids, uint32_t *permuted_route_ids, uint16_t *permuted_hidden, uint32_t *error_counters)
{
    __shared__ uint32_t shared_destination[8];
    uint32_t token_index;
    uint32_t route_slot;
    uint32_t expert_index;
    uint32_t local_slot;
    uint32_t chunk_index;
    uint32_t chunk_count;
    uint64_t route_index;
    uint64_t source_chunk_base;
    uint4 value;
    const uint4 *hidden4;
    uint4 *permuted4;

    token_index = blockIdx.x;
    if (token_index >= request.token_count)
    {
        return;
    }
    route_slot = threadIdx.x;
    if (route_slot < 8u)
    {
        route_index = ((uint64_t)token_index * 8u) + (uint64_t)route_slot;
        shared_destination[route_slot] = SPARKPIPE_CUDA_MOE_ROUTE_INVALID;
        expert_index = topk_ids[route_index];
        if (expert_index < request.expert_count)
        {
            local_slot = atomicAdd(&expert_cursors[expert_index], 1u);
            atomicAdd(&expert_counts[expert_index], 1u);
            if (local_slot < request.capacity_per_expert)
            {
                shared_destination[route_slot] = expert_offsets[expert_index] + local_slot;
                route_to_permuted_index[route_index] = shared_destination[route_slot];
                permuted_token_ids[shared_destination[route_slot]] = token_index;
                permuted_route_ids[shared_destination[route_slot]] = (uint32_t)route_index;
            }
            else
            {
                atomicAdd(&error_counters[SPARK_CUDA_MOE_ERROR_CAPACITY], 1u);
            }
        }
        else
        {
            atomicAdd(&error_counters[SPARK_CUDA_MOE_ERROR_INVALID_EXPERT], 1u);
        }
    }
    __syncthreads();
    hidden4 = (const uint4 *)hidden;
    permuted4 = (uint4 *)permuted_hidden;
    chunk_count = request.hidden_size >> 3u;
    source_chunk_base = (uint64_t)token_index * (uint64_t)chunk_count;
    for (chunk_index = threadIdx.x; chunk_index < chunk_count; chunk_index += blockDim.x)
    {
        value = hidden4[source_chunk_base + chunk_index];
        if (shared_destination[0] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[0] * (uint64_t)chunk_count) + chunk_index] = value;
        if (shared_destination[1] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[1] * (uint64_t)chunk_count) + chunk_index] = value;
        if (shared_destination[2] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[2] * (uint64_t)chunk_count) + chunk_index] = value;
        if (shared_destination[3] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[3] * (uint64_t)chunk_count) + chunk_index] = value;
        if (shared_destination[4] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[4] * (uint64_t)chunk_count) + chunk_index] = value;
        if (shared_destination[5] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[5] * (uint64_t)chunk_count) + chunk_index] = value;
        if (shared_destination[6] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[6] * (uint64_t)chunk_count) + chunk_index] = value;
        if (shared_destination[7] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            permuted4[((uint64_t)shared_destination[7] * (uint64_t)chunk_count) + chunk_index] = value;
    }
}

static __global__ void SparkCudaMoePreparedPermuteKernel(SparkCudaMoeDispatchRequest request, const uint16_t *hidden, const uint32_t *assignment_token_ids, const uint32_t *assignment_route_ids, const uint32_t *assignment_permuted_indices, uint32_t *route_to_permuted_index, uint32_t *permuted_token_ids, uint32_t *permuted_route_ids, uint16_t *permuted_hidden)
{
    uint64_t assignment_index;
    uint64_t route_count;
    uint64_t expert_capacity;
    uint32_t token_index;
    uint32_t route_index;
    uint32_t destination;
    uint32_t hidden_index;
    uint32_t chunk_index;
    uint32_t chunk_count;

    assignment_index = blockIdx.x;
    route_count = (uint64_t)request.token_count * (uint64_t)request.top_k;
    expert_capacity = (uint64_t)request.expert_count * (uint64_t)request.capacity_per_expert;
    if (assignment_index >= route_count)
    {
        return;
    }
    token_index = assignment_token_ids[assignment_index];
    route_index = assignment_route_ids[assignment_index];
    destination = assignment_permuted_indices[assignment_index];
    if (token_index >= request.token_count || route_index >= route_count || destination >= expert_capacity)
    {
        return;
    }
    if (threadIdx.x == 0u)
    {
        route_to_permuted_index[route_index] = destination;
        permuted_token_ids[destination] = token_index;
        permuted_route_ids[destination] = route_index;
    }
    if ((request.hidden_size & 7u) == 0u)
    {
        const uint4 *hidden4;
        uint4 *permuted4;
        uint64_t source_chunk_base;
        uint64_t destination_chunk_base;

        hidden4 = (const uint4 *)hidden;
        permuted4 = (uint4 *)permuted_hidden;
        chunk_count = request.hidden_size >> 3u;
        source_chunk_base = (uint64_t)token_index * (uint64_t)chunk_count;
        destination_chunk_base = (uint64_t)destination * (uint64_t)chunk_count;
        for (chunk_index = threadIdx.x; chunk_index < chunk_count; chunk_index += blockDim.x)
        {
            permuted4[destination_chunk_base + chunk_index] = hidden4[source_chunk_base + chunk_index];
        }
    }
    else
    {
        for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
        {
            permuted_hidden[(uint64_t)destination * (uint64_t)request.hidden_size + hidden_index] = hidden[(uint64_t)token_index * (uint64_t)request.hidden_size + hidden_index];
        }
    }
}

static __global__ void SparkCudaMoePreparedRouteMajorPermuteKernel(SparkCudaMoeDispatchRequest request, const uint16_t *hidden, const uint32_t *assignment_permuted_indices, uint32_t *route_to_permuted_index, uint32_t *permuted_token_ids, uint32_t *permuted_route_ids, uint16_t *permuted_hidden)
{
    uint64_t route_index;
    uint64_t route_count;
    uint64_t expert_capacity;
    uint32_t token_index;
    uint32_t destination;
    uint32_t hidden_index;
    uint32_t chunk_index;
    uint32_t chunk_count;

    route_index = blockIdx.x;
    route_count = (uint64_t)request.token_count * (uint64_t)request.top_k;
    expert_capacity = (uint64_t)request.expert_count * (uint64_t)request.capacity_per_expert;
    if (route_index >= route_count)
    {
        return;
    }
    token_index = (uint32_t)(route_index / request.top_k);
    destination = assignment_permuted_indices[route_index];
    if (destination >= expert_capacity)
    {
        return;
    }
    if (threadIdx.x == 0u)
    {
        route_to_permuted_index[route_index] = destination;
        permuted_token_ids[destination] = token_index;
        permuted_route_ids[destination] = (uint32_t)route_index;
    }
    if ((request.hidden_size & 7u) == 0u)
    {
        const uint4 *hidden4;
        uint4 *permuted4;
        uint64_t source_chunk_base;
        uint64_t destination_chunk_base;

        hidden4 = (const uint4 *)hidden;
        permuted4 = (uint4 *)permuted_hidden;
        chunk_count = request.hidden_size >> 3u;
        source_chunk_base = (uint64_t)token_index * (uint64_t)chunk_count;
        destination_chunk_base = (uint64_t)destination * (uint64_t)chunk_count;
        for (chunk_index = threadIdx.x; chunk_index < chunk_count; chunk_index += blockDim.x)
        {
            permuted4[destination_chunk_base + chunk_index] = hidden4[source_chunk_base + chunk_index];
        }
    }
    else
    {
        for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
        {
            permuted_hidden[(uint64_t)destination * (uint64_t)request.hidden_size + hidden_index] = hidden[(uint64_t)token_index * (uint64_t)request.hidden_size + hidden_index];
        }
    }
}

static __global__ void SparkCudaMoePreparedTop8WarpPermuteKernel(SparkCudaMoeDispatchRequest request, const uint16_t *hidden, const uint32_t *assignment_permuted_indices, uint32_t *route_to_permuted_index, uint32_t *permuted_token_ids, uint32_t *permuted_route_ids, uint16_t *permuted_hidden)
{
    uint32_t token_index;
    uint32_t route_slot;
    uint32_t lane_index;
    uint64_t route_index;
    uint64_t expert_capacity;
    uint32_t destination;
    uint32_t chunk_index;
    uint32_t chunk_count;
    uint64_t source_chunk_base;
    uint64_t destination_chunk_base;
    const uint4 *hidden4;
    uint4 *permuted4;

    token_index = blockIdx.x;
    route_slot = threadIdx.x >> 5u;
    lane_index = threadIdx.x & 31u;
    if (token_index >= request.token_count || route_slot >= 8u)
    {
        return;
    }
    route_index = ((uint64_t)token_index * 8u) + (uint64_t)route_slot;
    expert_capacity = (uint64_t)request.expert_count * (uint64_t)request.capacity_per_expert;
    destination = assignment_permuted_indices[route_index];
    if (destination >= expert_capacity)
    {
        return;
    }
    if (lane_index == 0u)
    {
        route_to_permuted_index[route_index] = destination;
        permuted_token_ids[destination] = token_index;
        permuted_route_ids[destination] = (uint32_t)route_index;
    }
    hidden4 = (const uint4 *)hidden;
    permuted4 = (uint4 *)permuted_hidden;
    chunk_count = request.hidden_size >> 3u;
    source_chunk_base = (uint64_t)token_index * (uint64_t)chunk_count;
    destination_chunk_base = (uint64_t)destination * (uint64_t)chunk_count;
    for (chunk_index = lane_index; chunk_index < chunk_count; chunk_index += 32u)
    {
        permuted4[destination_chunk_base + chunk_index] = hidden4[source_chunk_base + chunk_index];
    }
}

static __global__ void SparkCudaMoeCombineTokenKernel(SparkCudaMoeDispatchRequest request, const uint16_t *expert_output, const float *topk_weights, const uint32_t *route_to_permuted_index, uint16_t *output, uint32_t *error_counters)
{
    __shared__ uint32_t shared_permuted[SPARKPIPE_CUDA_MOE_MAX_TOP_K];
    __shared__ float shared_weight[SPARKPIPE_CUDA_MOE_MAX_TOP_K];
    uint32_t token_index;
    uint32_t hidden_index;
    uint32_t topk_index;
    uint64_t route_index;
    float sum;

    token_index = blockIdx.x;
    if (token_index >= request.token_count)
    {
        return;
    }
    if (threadIdx.x < request.top_k)
    {
        route_index = ((uint64_t)token_index * (uint64_t)request.top_k) + (uint64_t)threadIdx.x;
        shared_permuted[threadIdx.x] = route_to_permuted_index[route_index];
        shared_weight[threadIdx.x] = topk_weights[route_index];
    }
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        for (topk_index = 0u; topk_index < request.top_k; ++topk_index)
        {
            if (shared_permuted[topk_index] == SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            {
                atomicAdd(&error_counters[SPARK_CUDA_MOE_ERROR_INVALID_ROUTE], 1u);
            }
        }
    }
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        sum = 0.0f;
        for (topk_index = 0u; topk_index < request.top_k; ++topk_index)
        {
            if (shared_permuted[topk_index] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            {
                sum += shared_weight[topk_index] * SparkCudaMoeBf16ToFloat(expert_output[(uint64_t)shared_permuted[topk_index] * (uint64_t)request.hidden_size + hidden_index]);
            }
        }
        output[(uint64_t)token_index * (uint64_t)request.hidden_size + hidden_index] = SparkCudaMoeFloatToBf16(sum);
    }
}

static __global__ void SparkCudaMoeCombineTokenBf16x2Kernel(SparkCudaMoeDispatchRequest request, const __nv_bfloat162 *expert_output, const float *topk_weights, const uint32_t *route_to_permuted_index, __nv_bfloat162 *output, uint32_t *error_counters)
{
    __shared__ uint32_t shared_permuted[SPARKPIPE_CUDA_MOE_MAX_TOP_K];
    __shared__ float shared_weight[SPARKPIPE_CUDA_MOE_MAX_TOP_K];
    uint32_t token_index;
    uint32_t pair_index;
    uint32_t pair_count;
    uint32_t topk_index;
    uint64_t route_index;
    float2 sum;
    float2 value;

    token_index = blockIdx.x;
    if (token_index >= request.token_count)
    {
        return;
    }
    if (threadIdx.x < request.top_k)
    {
        route_index = ((uint64_t)token_index * (uint64_t)request.top_k) + (uint64_t)threadIdx.x;
        shared_permuted[threadIdx.x] = route_to_permuted_index[route_index];
        shared_weight[threadIdx.x] = topk_weights[route_index];
    }
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        for (topk_index = 0u; topk_index < request.top_k; ++topk_index)
        {
            if (shared_permuted[topk_index] == SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            {
                atomicAdd(&error_counters[SPARK_CUDA_MOE_ERROR_INVALID_ROUTE], 1u);
            }
        }
    }
    pair_count = request.hidden_size >> 1u;
    for (pair_index = threadIdx.x; pair_index < pair_count; pair_index += blockDim.x)
    {
        sum.x = 0.0f;
        sum.y = 0.0f;
        for (topk_index = 0u; topk_index < request.top_k; ++topk_index)
        {
            if (shared_permuted[topk_index] != SPARKPIPE_CUDA_MOE_ROUTE_INVALID)
            {
                value = __bfloat1622float2(expert_output[(uint64_t)shared_permuted[topk_index] * (uint64_t)pair_count + pair_index]);
                sum.x += shared_weight[topk_index] * value.x;
                sum.y += shared_weight[topk_index] * value.y;
            }
        }
        output[(uint64_t)token_index * (uint64_t)pair_count + pair_index] = __float22bfloat162_rn(sum);
    }
}

static __global__ void SparkCudaMoeCombineTop8TrustedBf16x2Kernel(SparkCudaMoeDispatchRequest request, const __nv_bfloat162 *expert_output, const float *topk_weights, const uint32_t *route_to_permuted_index, __nv_bfloat162 *output)
{
    __shared__ uint32_t shared_permuted[8];
    __shared__ float shared_weight[8];
    uint32_t token_index;
    uint32_t pair_index;
    uint32_t pair_count;
    uint64_t route_base;
    float2 sum;
    float2 value;
    uint32_t slot0,slot1,slot2,slot3,slot4,slot5,slot6,slot7;
    float weight0,weight1,weight2,weight3,weight4,weight5,weight6,weight7;

    token_index = blockIdx.x;
    if (token_index >= request.token_count)
    {
        return;
    }
    route_base = (uint64_t)token_index * 8u;
    if (threadIdx.x < 8u)
    {
        shared_permuted[threadIdx.x] = route_to_permuted_index[route_base + threadIdx.x];
        shared_weight[threadIdx.x] = topk_weights[route_base + threadIdx.x];
    }
    __syncthreads();
    slot0 = shared_permuted[0];
    slot1 = shared_permuted[1];
    slot2 = shared_permuted[2];
    slot3 = shared_permuted[3];
    slot4 = shared_permuted[4];
    slot5 = shared_permuted[5];
    slot6 = shared_permuted[6];
    slot7 = shared_permuted[7];
    weight0 = shared_weight[0];
    weight1 = shared_weight[1];
    weight2 = shared_weight[2];
    weight3 = shared_weight[3];
    weight4 = shared_weight[4];
    weight5 = shared_weight[5];
    weight6 = shared_weight[6];
    weight7 = shared_weight[7];
    pair_count = request.hidden_size >> 1u;
    for (pair_index = threadIdx.x; pair_index < pair_count; pair_index += blockDim.x)
    {
        sum.x = 0.0f;
        sum.y = 0.0f;
        value = __bfloat1622float2(expert_output[((uint64_t)slot0 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight0 * value.x;
        sum.y += weight0 * value.y;
        value = __bfloat1622float2(expert_output[((uint64_t)slot1 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight1 * value.x;
        sum.y += weight1 * value.y;
        value = __bfloat1622float2(expert_output[((uint64_t)slot2 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight2 * value.x;
        sum.y += weight2 * value.y;
        value = __bfloat1622float2(expert_output[((uint64_t)slot3 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight3 * value.x;
        sum.y += weight3 * value.y;
        value = __bfloat1622float2(expert_output[((uint64_t)slot4 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight4 * value.x;
        sum.y += weight4 * value.y;
        value = __bfloat1622float2(expert_output[((uint64_t)slot5 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight5 * value.x;
        sum.y += weight5 * value.y;
        value = __bfloat1622float2(expert_output[((uint64_t)slot6 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight6 * value.x;
        sum.y += weight6 * value.y;
        value = __bfloat1622float2(expert_output[((uint64_t)slot7 * (uint64_t)pair_count) + pair_index]);
        sum.x += weight7 * value.x;
        sum.y += weight7 * value.y;
        output[((uint64_t)token_index * (uint64_t)pair_count) + pair_index] = __float22bfloat162_rn(sum);
    }
}

static __global__ void SparkCudaMoeCombineRouteMajorTokenKernel(SparkCudaMoeDispatchRequest request, const uint16_t *expert_output, const float *topk_weights, uint16_t *output)
{
    uint32_t token_index;
    uint32_t hidden_index;
    uint32_t topk_index;
    uint64_t route_base;
    float sum;

    token_index = blockIdx.x;
    if (token_index >= request.token_count)
    {
        return;
    }
    route_base = (uint64_t)token_index * (uint64_t)request.top_k;
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        sum = 0.0f;
        for (topk_index = 0u; topk_index < request.top_k; ++topk_index)
        {
            sum += topk_weights[route_base + topk_index] * SparkCudaMoeBf16ToFloat(expert_output[((route_base + topk_index) * (uint64_t)request.hidden_size) + hidden_index]);
        }
        output[((uint64_t)token_index * (uint64_t)request.hidden_size) + hidden_index] = SparkCudaMoeFloatToBf16(sum);
    }
}

static __global__ void SparkCudaMoeCombineRouteMajorTokenBf16x2Kernel(SparkCudaMoeDispatchRequest request, const __nv_bfloat162 *expert_output, const float *topk_weights, __nv_bfloat162 *output)
{
    uint32_t token_index;
    uint32_t pair_index;
    uint32_t pair_count;
    uint32_t topk_index;
    uint64_t route_base;
    float2 sum;
    float2 value;

    token_index = blockIdx.x;
    if (token_index >= request.token_count)
    {
        return;
    }
    route_base = (uint64_t)token_index * (uint64_t)request.top_k;
    pair_count = request.hidden_size >> 1u;
    for (pair_index = threadIdx.x; pair_index < pair_count; pair_index += blockDim.x)
    {
        sum.x = 0.0f;
        sum.y = 0.0f;
        for (topk_index = 0u; topk_index < request.top_k; ++topk_index)
        {
            value = __bfloat1622float2(expert_output[((route_base + topk_index) * (uint64_t)pair_count) + pair_index]);
            sum.x += topk_weights[route_base + topk_index] * value.x;
            sum.y += topk_weights[route_base + topk_index] * value.y;
        }
        output[((uint64_t)token_index * (uint64_t)pair_count) + pair_index] = __float22bfloat162_rn(sum);
    }
}

static __global__ void SparkCudaMoeCombineRouteMajorTop8TrustedBf16x2Kernel(SparkCudaMoeDispatchRequest request, const __nv_bfloat162 *expert_output, const float *topk_weights, __nv_bfloat162 *output)
{
    uint32_t token_index;
    uint32_t pair_index;
    uint32_t pair_count;
    uint64_t route_base;
    float2 sum;
    float2 value;
    float weight0,weight1,weight2,weight3,weight4,weight5,weight6,weight7;

    token_index = blockIdx.x;
    if (token_index >= request.token_count)
    {
        return;
    }
    route_base = (uint64_t)token_index * 8u;
    weight0 = topk_weights[route_base + 0u];
    weight1 = topk_weights[route_base + 1u];
    weight2 = topk_weights[route_base + 2u];
    weight3 = topk_weights[route_base + 3u];
    weight4 = topk_weights[route_base + 4u];
    weight5 = topk_weights[route_base + 5u];
    weight6 = topk_weights[route_base + 6u];
    weight7 = topk_weights[route_base + 7u];
    pair_count = request.hidden_size >> 1u;
    for (pair_index = threadIdx.x; pair_index < pair_count; pair_index += blockDim.x)
    {
        sum.x = 0.0f;
        sum.y = 0.0f;
        value = __bfloat1622float2(expert_output[((route_base + 0u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight0 * value.x;
        sum.y += weight0 * value.y;
        value = __bfloat1622float2(expert_output[((route_base + 1u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight1 * value.x;
        sum.y += weight1 * value.y;
        value = __bfloat1622float2(expert_output[((route_base + 2u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight2 * value.x;
        sum.y += weight2 * value.y;
        value = __bfloat1622float2(expert_output[((route_base + 3u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight3 * value.x;
        sum.y += weight3 * value.y;
        value = __bfloat1622float2(expert_output[((route_base + 4u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight4 * value.x;
        sum.y += weight4 * value.y;
        value = __bfloat1622float2(expert_output[((route_base + 5u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight5 * value.x;
        sum.y += weight5 * value.y;
        value = __bfloat1622float2(expert_output[((route_base + 6u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight6 * value.x;
        sum.y += weight6 * value.y;
        value = __bfloat1622float2(expert_output[((route_base + 7u) * (uint64_t)pair_count) + pair_index]);
        sum.x += weight7 * value.x;
        sum.y += weight7 * value.y;
        output[((uint64_t)token_index * (uint64_t)pair_count) + pair_index] = __float22bfloat162_rn(sum);
    }
}

extern "C" SparkStatus SparkRunCudaMoeDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_topk_ids, uint32_t *device_expert_counts, uint32_t *device_expert_offsets, uint32_t *device_expert_cursors, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, uint32_t *device_error_counters, SparkCudaMoeDispatchReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t route_count;
    uint64_t expert_capacity;
    uint32_t clear_blocks;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_hidden_bf16 == 0 || device_topk_ids == 0 || device_expert_counts == 0 || device_expert_offsets == 0 || device_expert_cursors == 0 || device_route_to_permuted_index == 0 || device_permuted_token_ids == 0 || device_permuted_route_ids == 0 || device_permuted_hidden_bf16 == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route_count = SparkCudaMoeRouteCountHost(request);
    expert_capacity = SparkCudaMoeExpertCapacityHost(request);
    clear_blocks = SparkCudaMoeBlockCount(expert_capacity > route_count ? expert_capacity : route_count);
    SparkCudaMoeFillReportShape(request, report);
    SparkCudaMoeClearDispatchKernel<<<clear_blocks, SPARK_CUDA_MOE_THREADS>>>(*request, device_expert_counts, device_expert_offsets, device_expert_cursors, device_route_to_permuted_index, device_permuted_token_ids, device_permuted_route_ids, device_error_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status == cudaSuccess && request->top_k == 8u && (request->hidden_size & 7u) == 0u && (request->flags & SPARKPIPE_CUDA_MOE_FLAG_TOKEN_GROUPED_DISPATCH) != 0u)
    {
        SparkCudaMoePermuteTop8TokenKernel<<<request->token_count, SPARK_CUDA_MOE_DISPATCH_FAST_THREADS>>>(*request, (const uint16_t *)device_hidden_bf16, device_topk_ids, device_expert_counts, device_expert_offsets, device_expert_cursors, device_route_to_permuted_index, device_permuted_token_ids, device_permuted_route_ids, (uint16_t *)device_permuted_hidden_bf16, device_error_counters);
        cuda_status = cudaGetLastError();
        report->permute_fast_kernel_count = 1u;
    }
    else if (cuda_status == cudaSuccess)
    {
        SparkCudaMoePermuteKernel<<<(uint32_t)route_count, SPARK_CUDA_MOE_THREADS>>>(*request, (const uint16_t *)device_hidden_bf16, device_topk_ids, device_expert_counts, device_expert_offsets, device_expert_cursors, device_route_to_permuted_index, device_permuted_token_ids, device_permuted_route_ids, (uint16_t *)device_permuted_hidden_bf16, device_error_counters);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->clear_kernel_count = 1u;
    report->count_kernel_count = 0u;
    report->prefix_kernel_count = 0u;
    report->permute_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaMoePreparedDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_assignment_token_ids, const uint32_t *device_assignment_route_ids, const uint32_t *device_assignment_permuted_indices, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, SparkCudaMoeDispatchReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t route_count;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_hidden_bf16 == 0 || device_assignment_token_ids == 0 || device_assignment_route_ids == 0 || device_assignment_permuted_indices == 0 || device_route_to_permuted_index == 0 || device_permuted_token_ids == 0 || device_permuted_route_ids == 0 || device_permuted_hidden_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->flags & SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route_count = SparkCudaMoeRouteCountHost(request);
    SparkCudaMoeFillReportShape(request, report);
    SparkCudaMoePreparedPermuteKernel<<<(uint32_t)route_count, SPARK_CUDA_MOE_DISPATCH_FAST_THREADS>>>(*request, (const uint16_t *)device_hidden_bf16, device_assignment_token_ids, device_assignment_route_ids, device_assignment_permuted_indices, device_route_to_permuted_index, device_permuted_token_ids, device_permuted_route_ids, (uint16_t *)device_permuted_hidden_bf16);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->clear_kernel_count = 0u;
    report->count_kernel_count = 0u;
    report->prefix_kernel_count = 0u;
    report->permute_kernel_count = 1u;
    report->permute_fast_kernel_count = 0u;
    report->prepared_dispatch_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaMoePreparedRouteMajorDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_assignment_permuted_indices, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, SparkCudaMoeDispatchReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t route_count;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_hidden_bf16 == 0 || device_assignment_permuted_indices == 0 || device_route_to_permuted_index == 0 || device_permuted_token_ids == 0 || device_permuted_route_ids == 0 || device_permuted_hidden_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->flags & SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route_count = SparkCudaMoeRouteCountHost(request);
    SparkCudaMoeFillReportShape(request, report);
    if (request->top_k == 8u && (request->hidden_size & 7u) == 0u && (request->flags & SPARKPIPE_CUDA_MOE_FLAG_TOP8_WARP_DISPATCH) != 0u)
    {
        SparkCudaMoePreparedTop8WarpPermuteKernel<<<request->token_count, 256u>>>(*request, (const uint16_t *)device_hidden_bf16, device_assignment_permuted_indices, device_route_to_permuted_index, device_permuted_token_ids, device_permuted_route_ids, (uint16_t *)device_permuted_hidden_bf16);
        report->permute_fast_kernel_count = 1u;
    }
    else
    {
        SparkCudaMoePreparedRouteMajorPermuteKernel<<<(uint32_t)route_count, SPARK_CUDA_MOE_DISPATCH_FAST_THREADS>>>(*request, (const uint16_t *)device_hidden_bf16, device_assignment_permuted_indices, device_route_to_permuted_index, device_permuted_token_ids, device_permuted_route_ids, (uint16_t *)device_permuted_hidden_bf16);
        report->permute_fast_kernel_count = 0u;
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->clear_kernel_count = 0u;
    report->count_kernel_count = 0u;
    report->prefix_kernel_count = 0u;
    report->permute_kernel_count = 1u;
    report->prepared_dispatch_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaMoeCombine(const SparkCudaMoeDispatchRequest *request, const void *device_expert_output_bf16, const float *device_topk_weights, const uint32_t *device_route_to_permuted_index, void *device_output_bf16, uint32_t *device_error_counters, SparkCudaMoeDispatchReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_expert_output_bf16 == 0 || device_topk_weights == 0 || device_route_to_permuted_index == 0 || device_output_bf16 == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMoeFillReportShape(request, report);
    if (request->top_k == 8u && (request->hidden_size & 1u) == 0u && (request->flags & SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES) != 0u)
    {
        SparkCudaMoeCombineTop8TrustedBf16x2Kernel<<<request->token_count, SPARK_CUDA_MOE_FAST_THREADS>>>(*request, (const __nv_bfloat162 *)device_expert_output_bf16, device_topk_weights, device_route_to_permuted_index, (__nv_bfloat162 *)device_output_bf16);
        report->combine_fast_kernel_count = 1u;
    }
    else if ((request->hidden_size & 1u) == 0u)
    {
        SparkCudaMoeCombineTokenBf16x2Kernel<<<request->token_count, SPARK_CUDA_MOE_THREADS>>>(*request, (const __nv_bfloat162 *)device_expert_output_bf16, device_topk_weights, device_route_to_permuted_index, (__nv_bfloat162 *)device_output_bf16, device_error_counters);
    }
    else
    {
        SparkCudaMoeCombineTokenKernel<<<request->token_count, SPARK_CUDA_MOE_THREADS>>>(*request, (const uint16_t *)device_expert_output_bf16, device_topk_weights, device_route_to_permuted_index, (uint16_t *)device_output_bf16, device_error_counters);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->combine_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaMoeCombineRouteMajor(const SparkCudaMoeDispatchRequest *request, const void *device_route_major_expert_output_bf16, const float *device_topk_weights, void *device_output_bf16, SparkCudaMoeDispatchReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_route_major_expert_output_bf16 == 0 || device_topk_weights == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->flags & SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMoeFillReportShape(request, report);
    if (request->top_k == 8u && (request->hidden_size & 1u) == 0u)
    {
        SparkCudaMoeCombineRouteMajorTop8TrustedBf16x2Kernel<<<request->token_count, SPARK_CUDA_MOE_FAST_THREADS>>>(*request, (const __nv_bfloat162 *)device_route_major_expert_output_bf16, device_topk_weights, (__nv_bfloat162 *)device_output_bf16);
        report->combine_fast_kernel_count = 1u;
    }
    else if ((request->hidden_size & 1u) == 0u)
    {
        SparkCudaMoeCombineRouteMajorTokenBf16x2Kernel<<<request->token_count, SPARK_CUDA_MOE_THREADS>>>(*request, (const __nv_bfloat162 *)device_route_major_expert_output_bf16, device_topk_weights, (__nv_bfloat162 *)device_output_bf16);
    }
    else
    {
        SparkCudaMoeCombineRouteMajorTokenKernel<<<request->token_count, SPARK_CUDA_MOE_THREADS>>>(*request, (const uint16_t *)device_route_major_expert_output_bf16, device_topk_weights, (uint16_t *)device_output_bf16);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->combine_kernel_count = 1u;
    report->combine_route_major_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}
