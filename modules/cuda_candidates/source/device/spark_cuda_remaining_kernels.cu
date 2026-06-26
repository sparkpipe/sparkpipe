#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_remaining_kernels.h"

#define SPARK_REMAINING_REDUCE_THREADS 256u
#define SPARK_REMAINING_TILE_M 8u
#define SPARK_REMAINING_TILE_N 16u
#define SPARK_REMAINING_TILE_K 32u

static __device__ uint64_t SparkRemainingMixU64Device(uint64_t checksum, uint64_t value)
{
    checksum ^= value + 0x9e3779b97f4a7c15ull + (checksum << 6u) + (checksum >> 2u);
    return checksum;
}

static __device__ int32_t SparkDecodeFp4NibbleDevice(uint8_t packed_value, uint64_t value_index)
{
    uint8_t nibble;

    nibble = (value_index & 1u) == 0u ? (packed_value & 0x0fu) : ((packed_value >> 4u) & 0x0fu);
    return nibble < 8u ? (int32_t)nibble : (int32_t)nibble - 16;
}

static __device__ float SparkReadFp4ValueDevice(const uint8_t *packed_values, uint64_t value_index, float scale)
{
    return (float)SparkDecodeFp4NibbleDevice(packed_values[value_index >> 1u], value_index) * scale;
}

static __device__ bool SparkRemainingBetterScoreDevice(float candidate_score, uint32_t candidate_index, float best_score, uint32_t best_index)
{
    if (candidate_score > best_score)
    {
        return true;
    }
    if (candidate_score == best_score && candidate_index < best_index)
    {
        return true;
    }
    return false;
}

static __device__ uint64_t SparkRemainingFloatChecksumDevice(const float *values, uint64_t value_count)
{
    const uint8_t *bytes;
    uint64_t byte_count;
    uint64_t byte_index;
    uint64_t checksum;

    bytes = (const uint8_t *)values;
    byte_count = value_count * (uint64_t)sizeof(float);
    checksum = 0x535052454D434846ull;
    checksum = SparkRemainingMixU64Device(checksum, value_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkRemainingMixU64Device(checksum, bytes[byte_index]);
    }

    return checksum;
}

static __device__ uint64_t SparkRemainingU32ChecksumDevice(const uint32_t *values, uint64_t value_count)
{
    uint64_t value_index;
    uint64_t checksum;

    checksum = 0x535052454D434855ull;
    checksum = SparkRemainingMixU64Device(checksum, value_count);
    for (value_index = 0; value_index < value_count; ++value_index)
    {
        checksum = SparkRemainingMixU64Device(checksum, values[value_index]);
    }

    return checksum;
}

static __global__ void SparkRemainingFp4LinearTiledKernel(SparkCudaRemainingInferenceRequest request, const float *token_input, const uint8_t *fp4_weight, float *output)
{
    __shared__ float tile_input[SPARK_REMAINING_TILE_M][SPARK_REMAINING_TILE_K];
    __shared__ float tile_weight[SPARK_REMAINING_TILE_K][SPARK_REMAINING_TILE_N];
    uint32_t local_row;
    uint32_t local_col;
    uint32_t linear_thread;
    uint32_t load_index;
    uint32_t tile_index;
    uint32_t token_index;
    uint32_t output_index;
    uint32_t hidden_index;
    float sum;

    local_col = threadIdx.x;
    local_row = threadIdx.y;
    linear_thread = (local_row * SPARK_REMAINING_TILE_N) + local_col;
    token_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + local_row;
    output_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + local_col;
    sum = 0.0f;
    for (tile_index = 0u; tile_index < request.hidden_size; tile_index += SPARK_REMAINING_TILE_K)
    {
        for (load_index = linear_thread; load_index < (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_K); load_index += (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_N))
        {
            hidden_index = tile_index + (load_index % SPARK_REMAINING_TILE_K);
            token_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + (load_index / SPARK_REMAINING_TILE_K);
            tile_input[load_index / SPARK_REMAINING_TILE_K][load_index % SPARK_REMAINING_TILE_K] = (token_index < request.token_count && hidden_index < request.hidden_size) ? token_input[(uint64_t)token_index * request.hidden_size + hidden_index] : 0.0f;
        }
        for (load_index = linear_thread; load_index < (SPARK_REMAINING_TILE_K * SPARK_REMAINING_TILE_N); load_index += (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_N))
        {
            hidden_index = tile_index + (load_index / SPARK_REMAINING_TILE_N);
            output_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + (load_index % SPARK_REMAINING_TILE_N);
            tile_weight[load_index / SPARK_REMAINING_TILE_N][load_index % SPARK_REMAINING_TILE_N] = (hidden_index < request.hidden_size && output_index < request.output_size) ? SparkReadFp4ValueDevice(fp4_weight, ((uint64_t)hidden_index * request.output_size) + output_index, request.fp4_scale) : 0.0f;
        }
        __syncthreads();
        for (hidden_index = 0u; hidden_index < SPARK_REMAINING_TILE_K; ++hidden_index)
        {
            sum += tile_input[local_row][hidden_index] * tile_weight[hidden_index][local_col];
        }
        __syncthreads();
    }
    token_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + local_row;
    output_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + local_col;
    if (token_index < request.token_count && output_index < request.output_size)
    {
        output[(uint64_t)token_index * request.output_size + output_index] = sum;
    }
}

static __global__ void SparkRemainingExpertKernel(SparkCudaRemainingInferenceRequest request, const float *token_input, const float *expert_weight, const uint32_t *expert_ids, float *output, SparkCudaRemainingInferenceReport *report)
{
    uint64_t output_linear_index;
    uint32_t token_index;
    uint32_t output_index;
    uint32_t hidden_index;
    uint32_t expert_index;
    float sum;

    output_linear_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (output_linear_index < (uint64_t)request.token_count * request.output_size)
    {
        token_index = (uint32_t)(output_linear_index / request.output_size);
        output_index = (uint32_t)(output_linear_index - ((uint64_t)token_index * request.output_size));
        expert_index = expert_ids[token_index];
        if (expert_index >= request.expert_count)
        {
            if (output_index == 0u)
            {
                atomicAdd(&report->invalid_expert_count, 1u);
            }
            expert_index = 0u;
        }
        sum = 0.0f;
        for (hidden_index = 0; hidden_index < request.hidden_size; ++hidden_index)
        {
            sum += token_input[(uint64_t)token_index * request.hidden_size + hidden_index] * expert_weight[((uint64_t)expert_index * request.hidden_size * request.output_size) + ((uint64_t)hidden_index * request.output_size) + output_index];
        }
        output[output_linear_index] = sum;
        output_linear_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkRemainingExpertFp4Kernel(SparkCudaRemainingInferenceRequest request, const float *token_input, const uint8_t *expert_weight, const uint32_t *expert_ids, float *output)
{
    uint64_t output_linear_index;
    uint32_t token_index;
    uint32_t output_index;
    uint32_t hidden_index;
    uint32_t expert_index;
    float sum;

    output_linear_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (output_linear_index < (uint64_t)request.token_count * request.output_size)
    {
        token_index = (uint32_t)(output_linear_index / request.output_size);
        output_index = (uint32_t)(output_linear_index - ((uint64_t)token_index * request.output_size));
        expert_index = expert_ids[token_index] < request.expert_count ? expert_ids[token_index] : 0u;
        sum = 0.0f;
        for (hidden_index = 0; hidden_index < request.hidden_size; ++hidden_index)
        {
            sum += token_input[(uint64_t)token_index * request.hidden_size + hidden_index] * SparkReadFp4ValueDevice(expert_weight, ((uint64_t)expert_index * request.hidden_size * request.output_size) + ((uint64_t)hidden_index * request.output_size) + output_index, request.expert_fp4_scale);
        }
        output[output_linear_index] = sum;
        output_linear_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkRemainingSharedTiledKernel(SparkCudaRemainingInferenceRequest request, const float *token_input, const float *expert_output, const float *shared_weight, float *shared_output)
{
    __shared__ float tile_input[SPARK_REMAINING_TILE_M][SPARK_REMAINING_TILE_K];
    __shared__ float tile_weight[SPARK_REMAINING_TILE_K][SPARK_REMAINING_TILE_N];
    uint32_t local_row;
    uint32_t local_col;
    uint32_t linear_thread;
    uint32_t load_index;
    uint32_t tile_index;
    uint32_t token_index;
    uint32_t output_index;
    uint32_t hidden_index;
    float sum;

    local_col = threadIdx.x;
    local_row = threadIdx.y;
    linear_thread = (local_row * SPARK_REMAINING_TILE_N) + local_col;
    token_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + local_row;
    output_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + local_col;
    sum = 0.0f;
    for (tile_index = 0u; tile_index < request.hidden_size; tile_index += SPARK_REMAINING_TILE_K)
    {
        for (load_index = linear_thread; load_index < (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_K); load_index += (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_N))
        {
            hidden_index = tile_index + (load_index % SPARK_REMAINING_TILE_K);
            token_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + (load_index / SPARK_REMAINING_TILE_K);
            tile_input[load_index / SPARK_REMAINING_TILE_K][load_index % SPARK_REMAINING_TILE_K] = (token_index < request.token_count && hidden_index < request.hidden_size) ? token_input[(uint64_t)token_index * request.hidden_size + hidden_index] : 0.0f;
        }
        for (load_index = linear_thread; load_index < (SPARK_REMAINING_TILE_K * SPARK_REMAINING_TILE_N); load_index += (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_N))
        {
            hidden_index = tile_index + (load_index / SPARK_REMAINING_TILE_N);
            output_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + (load_index % SPARK_REMAINING_TILE_N);
            tile_weight[load_index / SPARK_REMAINING_TILE_N][load_index % SPARK_REMAINING_TILE_N] = (hidden_index < request.hidden_size && output_index < request.output_size) ? shared_weight[(uint64_t)hidden_index * request.output_size + output_index] : 0.0f;
        }
        __syncthreads();
        for (hidden_index = 0u; hidden_index < SPARK_REMAINING_TILE_K; ++hidden_index)
        {
            sum += tile_input[local_row][hidden_index] * tile_weight[hidden_index][local_col];
        }
        __syncthreads();
    }
    token_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + local_row;
    output_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + local_col;
    if (token_index < request.token_count && output_index < request.output_size)
    {
        shared_output[(uint64_t)token_index * request.output_size + output_index] = expert_output[(uint64_t)token_index * request.output_size + output_index] + sum;
    }
}

static __global__ void SparkRemainingDenseMlaKernel(SparkCudaRemainingInferenceRequest request, const float *query, const float *key, const float *value, float *output)
{
    __shared__ float shared_scores[SPARK_REMAINING_REDUCE_THREADS];
    __shared__ uint32_t shared_indices[SPARK_REMAINING_REDUCE_THREADS];
    uint32_t query_index;
    uint32_t kv_index;
    uint32_t hidden_index;
    uint32_t best_index;
    uint32_t other_index;
    uint32_t stride;
    float score;
    float best_score;
    float other_score;

    query_index = blockIdx.x;
    if (query_index >= request.query_count)
    {
        return;
    }
    best_index = UINT32_MAX;
    best_score = -FLT_MAX;
    for (kv_index = threadIdx.x; kv_index < request.kv_count; kv_index += blockDim.x)
    {
        score = 0.0f;
        for (hidden_index = 0; hidden_index < request.head_dim; ++hidden_index)
        {
            score += query[(uint64_t)query_index * request.head_dim + hidden_index] * key[(uint64_t)kv_index * request.head_dim + hidden_index];
        }
        if (SparkRemainingBetterScoreDevice(score, kv_index, best_score, best_index))
        {
            best_score = score;
            best_index = kv_index;
        }
    }
    shared_scores[threadIdx.x] = best_score;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            other_score = shared_scores[threadIdx.x + stride];
            other_index = shared_indices[threadIdx.x + stride];
            if (SparkRemainingBetterScoreDevice(other_score, other_index, shared_scores[threadIdx.x], shared_indices[threadIdx.x]))
            {
                shared_scores[threadIdx.x] = other_score;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    best_index = shared_indices[0] == UINT32_MAX ? 0u : shared_indices[0];
    for (hidden_index = threadIdx.x; hidden_index < request.head_dim; hidden_index += blockDim.x)
    {
        output[(uint64_t)query_index * request.head_dim + hidden_index] = value[(uint64_t)best_index * request.head_dim + hidden_index];
    }
}

static __global__ void SparkRemainingSparseMapMlaKernel(SparkCudaRemainingInferenceRequest request, const float *query, const float *key, const float *value, const uint32_t *sparse_indices, const uint32_t *physical_block_map, uint32_t *mapped_indices, float *output, SparkCudaRemainingInferenceReport *report)
{
    __shared__ float shared_scores[SPARK_REMAINING_REDUCE_THREADS];
    __shared__ uint32_t shared_indices[SPARK_REMAINING_REDUCE_THREADS];
    uint32_t query_index;
    uint32_t sparse_index;
    uint32_t kv_index;
    uint32_t mapped_index;
    uint32_t hidden_index;
    uint32_t best_index;
    uint32_t other_index;
    uint32_t stride;
    float score;
    float best_score;
    float other_score;

    query_index = blockIdx.x;
    if (query_index >= request.query_count)
    {
        return;
    }
    best_index = UINT32_MAX;
    best_score = -FLT_MAX;
    for (sparse_index = threadIdx.x; sparse_index < request.sparse_top_k; sparse_index += blockDim.x)
    {
        kv_index = sparse_indices[(uint64_t)query_index * request.sparse_top_k + sparse_index];
        if (kv_index >= request.kv_count || physical_block_map[kv_index] >= request.kv_count)
        {
            atomicAdd(&report->invalid_sparse_index_count, 1u);
            mapped_indices[(uint64_t)query_index * request.sparse_top_k + sparse_index] = UINT32_MAX;
            continue;
        }
        mapped_index = physical_block_map[kv_index];
        mapped_indices[(uint64_t)query_index * request.sparse_top_k + sparse_index] = mapped_index;
        score = 0.0f;
        for (hidden_index = 0; hidden_index < request.head_dim; ++hidden_index)
        {
            score += query[(uint64_t)query_index * request.head_dim + hidden_index] * key[(uint64_t)mapped_index * request.head_dim + hidden_index];
        }
        if (SparkRemainingBetterScoreDevice(score, mapped_index, best_score, best_index))
        {
            best_score = score;
            best_index = mapped_index;
        }
    }
    shared_scores[threadIdx.x] = best_score;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            other_score = shared_scores[threadIdx.x + stride];
            other_index = shared_indices[threadIdx.x + stride];
            if (SparkRemainingBetterScoreDevice(other_score, other_index, shared_scores[threadIdx.x], shared_indices[threadIdx.x]))
            {
                shared_scores[threadIdx.x] = other_score;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    best_index = shared_indices[0] == UINT32_MAX ? 0u : shared_indices[0];
    for (hidden_index = threadIdx.x; hidden_index < request.head_dim; hidden_index += blockDim.x)
    {
        output[(uint64_t)query_index * request.head_dim + hidden_index] = value[(uint64_t)best_index * request.head_dim + hidden_index];
    }
}

static __global__ void SparkRemainingDraftProjectionTiledKernel(SparkCudaRemainingInferenceRequest request, const float *draft_hidden, const float *draft_weight, float *draft_logits)
{
    __shared__ float tile_input[SPARK_REMAINING_TILE_M][SPARK_REMAINING_TILE_K];
    __shared__ float tile_weight[SPARK_REMAINING_TILE_K][SPARK_REMAINING_TILE_N];
    uint32_t local_row;
    uint32_t local_col;
    uint32_t linear_thread;
    uint32_t load_index;
    uint32_t tile_index;
    uint32_t draft_index;
    uint32_t vocab_index;
    uint32_t hidden_index;
    float sum;

    local_col = threadIdx.x;
    local_row = threadIdx.y;
    linear_thread = (local_row * SPARK_REMAINING_TILE_N) + local_col;
    draft_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + local_row;
    vocab_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + local_col;
    sum = 0.0f;
    for (tile_index = 0u; tile_index < request.hidden_size; tile_index += SPARK_REMAINING_TILE_K)
    {
        for (load_index = linear_thread; load_index < (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_K); load_index += (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_N))
        {
            hidden_index = tile_index + (load_index % SPARK_REMAINING_TILE_K);
            draft_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + (load_index / SPARK_REMAINING_TILE_K);
            tile_input[load_index / SPARK_REMAINING_TILE_K][load_index % SPARK_REMAINING_TILE_K] = (draft_index < request.draft_count && hidden_index < request.hidden_size) ? draft_hidden[(uint64_t)draft_index * request.hidden_size + hidden_index] : 0.0f;
        }
        for (load_index = linear_thread; load_index < (SPARK_REMAINING_TILE_K * SPARK_REMAINING_TILE_N); load_index += (SPARK_REMAINING_TILE_M * SPARK_REMAINING_TILE_N))
        {
            hidden_index = tile_index + (load_index / SPARK_REMAINING_TILE_N);
            vocab_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + (load_index % SPARK_REMAINING_TILE_N);
            tile_weight[load_index / SPARK_REMAINING_TILE_N][load_index % SPARK_REMAINING_TILE_N] = (hidden_index < request.hidden_size && vocab_index < request.vocab_size) ? draft_weight[(uint64_t)hidden_index * request.vocab_size + vocab_index] : 0.0f;
        }
        __syncthreads();
        for (hidden_index = 0u; hidden_index < SPARK_REMAINING_TILE_K; ++hidden_index)
        {
            sum += tile_input[local_row][hidden_index] * tile_weight[hidden_index][local_col];
        }
        __syncthreads();
    }
    draft_index = (blockIdx.y * SPARK_REMAINING_TILE_M) + local_row;
    vocab_index = (blockIdx.x * SPARK_REMAINING_TILE_N) + local_col;
    if (draft_index < request.draft_count && vocab_index < request.vocab_size)
    {
        draft_logits[(uint64_t)draft_index * request.vocab_size + vocab_index] = sum;
    }
}

static __global__ void SparkRemainingDraftArgmaxKernel(SparkCudaRemainingInferenceRequest request, const float *draft_logits, uint32_t *draft_token_ids)
{
    __shared__ float shared_scores[SPARK_REMAINING_REDUCE_THREADS];
    __shared__ uint32_t shared_tokens[SPARK_REMAINING_REDUCE_THREADS];
    uint32_t draft_index;
    uint32_t vocab_index;
    uint32_t best_index;
    uint32_t other_index;
    uint32_t stride;
    float best_score;
    float score;
    float other_score;

    draft_index = blockIdx.x;
    if (draft_index >= request.draft_count)
    {
        return;
    }
    best_index = UINT32_MAX;
    best_score = -FLT_MAX;
    for (vocab_index = threadIdx.x; vocab_index < request.vocab_size; vocab_index += blockDim.x)
    {
        score = draft_logits[(uint64_t)draft_index * request.vocab_size + vocab_index];
        if (SparkRemainingBetterScoreDevice(score, vocab_index, best_score, best_index))
        {
            best_score = score;
            best_index = vocab_index;
        }
    }
    shared_scores[threadIdx.x] = best_score;
    shared_tokens[threadIdx.x] = best_index;
    __syncthreads();
    for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            other_score = shared_scores[threadIdx.x + stride];
            other_index = shared_tokens[threadIdx.x + stride];
            if (SparkRemainingBetterScoreDevice(other_score, other_index, shared_scores[threadIdx.x], shared_tokens[threadIdx.x]))
            {
                shared_scores[threadIdx.x] = other_score;
                shared_tokens[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        draft_token_ids[draft_index] = shared_tokens[0] == UINT32_MAX ? 0u : shared_tokens[0];
    }
}

static __global__ void SparkRemainingSpecVerifyKernel(SparkCudaRemainingInferenceRequest request, const uint32_t *expected_tokens, const uint32_t *draft_token_ids, uint32_t *accepted_mask, SparkCudaRemainingInferenceReport *report)
{
    uint32_t draft_index;
    uint32_t accepted_prefix;
    uint32_t accepted_count;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    accepted_prefix = 1u;
    accepted_count = 0u;
    for (draft_index = 0; draft_index < request.draft_count; ++draft_index)
    {
        if (accepted_prefix != 0u && draft_token_ids[draft_index] == expected_tokens[draft_index])
        {
            accepted_mask[draft_index] = 1u;
            accepted_count += 1u;
            continue;
        }
        accepted_mask[draft_index] = 0u;
        accepted_prefix = 0u;
    }
    report->accepted_token_count += accepted_count;
    report->rejected_token_count += request.draft_count - accepted_count;
}

static __global__ void SparkRemainingChecksumTraceKernel(SparkCudaRemainingInferenceRequest request, const SparkCudaRemainingInferenceOutputs outputs, SparkCudaRemainingInferenceReport *report)
{
    uint64_t checksum;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    if (request.compute_checksum != 0u)
    {
        report->fp4_linear_checksum = SparkRemainingFloatChecksumDevice(outputs.fp4_linear_output, (uint64_t)request.token_count * request.output_size);
        report->expert_checksum = SparkRemainingFloatChecksumDevice(outputs.expert_output, (uint64_t)request.token_count * request.output_size);
        report->expert_fp4_checksum = SparkRemainingFloatChecksumDevice(outputs.expert_fp4_output, (uint64_t)request.token_count * request.output_size);
        report->shared_checksum = SparkRemainingFloatChecksumDevice(outputs.shared_output, (uint64_t)request.token_count * request.output_size);
        report->map_checksum = SparkRemainingU32ChecksumDevice(outputs.mapped_sparse_indices, (uint64_t)request.query_count * request.sparse_top_k);
        report->dense_mla_checksum = SparkRemainingFloatChecksumDevice(outputs.dense_mla_output, (uint64_t)request.query_count * request.head_dim);
        report->sparse_mla_checksum = SparkRemainingFloatChecksumDevice(outputs.sparse_mla_output, (uint64_t)request.query_count * request.head_dim);
        report->draft_checksum = SparkRemainingFloatChecksumDevice(outputs.draft_logits, (uint64_t)request.draft_count * request.vocab_size);
        report->token_checksum = SparkRemainingU32ChecksumDevice(outputs.draft_token_ids, request.draft_count);
        report->accept_checksum = SparkRemainingU32ChecksumDevice(outputs.accepted_mask, request.draft_count);
    }
    report->fp4_linear_kernel_count = 1u;
    report->expert_kernel_count = 1u;
    report->expert_fp4_kernel_count = 1u;
    report->shared_kernel_count = 1u;
    report->sparse_map_kernel_count = 1u;
    report->dense_mla_kernel_count = 1u;
    report->sparse_mla_kernel_count = 1u;
    report->draft_projection_kernel_count = 1u;
    report->spec_verify_kernel_count = 1u;
    report->checksum_kernel_count = request.compute_checksum != 0u ? 1u : 0u;
    report->trace_kernel_count = 1u;
    if (request.sentinel != SPARKPIPE_CUDA_REMAINING_SENTINEL)
    {
        report->sentinel_violation_count += 1u;
    }
    checksum = 0x535052454D545243ull;
    checksum = SparkRemainingMixU64Device(checksum, request.token_count);
    checksum = SparkRemainingMixU64Device(checksum, request.hidden_size);
    checksum = SparkRemainingMixU64Device(checksum, request.output_size);
    checksum = SparkRemainingMixU64Device(checksum, request.expert_count);
    checksum = SparkRemainingMixU64Device(checksum, request.query_count);
    checksum = SparkRemainingMixU64Device(checksum, request.kv_count);
    checksum = SparkRemainingMixU64Device(checksum, request.head_dim);
    checksum = SparkRemainingMixU64Device(checksum, report->fp4_linear_checksum);
    checksum = SparkRemainingMixU64Device(checksum, report->sparse_mla_checksum);
    checksum = SparkRemainingMixU64Device(checksum, report->draft_checksum);
    checksum = SparkRemainingMixU64Device(checksum, report->accepted_token_count);
    checksum = SparkRemainingMixU64Device(checksum, report->rejected_token_count);
    report->trace_checksum = checksum;
}

static uint32_t SparkRemainingBlockCount(uint64_t value_count)
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

static SparkStatus SparkValidateRemainingCudaHostPointers(const SparkCudaRemainingInferenceInputs *inputs, const SparkCudaRemainingInferenceOutputs *outputs, SparkCudaRemainingInferenceReport *report)
{
    if (inputs == 0 || outputs == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (inputs->token_input == 0 || inputs->fp4_linear_weight == 0 || inputs->expert_weight == 0 || inputs->expert_fp4_weight == 0 || inputs->shared_weight == 0 || inputs->expert_ids == 0 || inputs->query == 0 || inputs->key == 0 || inputs->value == 0 || inputs->sparse_indices == 0 || inputs->physical_block_map == 0 || inputs->draft_hidden == 0 || inputs->draft_weight == 0 || inputs->expected_tokens == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (outputs->fp4_linear_output == 0 || outputs->expert_output == 0 || outputs->expert_fp4_output == 0 || outputs->shared_output == 0 || outputs->mapped_sparse_indices == 0 || outputs->dense_mla_output == 0 || outputs->sparse_mla_output == 0 || outputs->draft_logits == 0 || outputs->draft_token_ids == 0 || outputs->accepted_mask == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaRemainingInferenceDeviceKernels(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *device_inputs, SparkCudaRemainingInferenceOutputs *device_outputs, SparkCudaRemainingInferenceReport *device_report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t output_values;
    dim3 projection_block;
    dim3 token_output_grid;
    dim3 draft_grid;

    status = SparkValidateCudaRemainingInferenceRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateRemainingCudaHostPointers(device_inputs, device_outputs, device_report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    output_values = (uint64_t)request->token_count * request->output_size;
    projection_block = dim3(SPARK_REMAINING_TILE_N, SPARK_REMAINING_TILE_M, 1u);
    token_output_grid = dim3((request->output_size + SPARK_REMAINING_TILE_N - 1u) / SPARK_REMAINING_TILE_N, (request->token_count + SPARK_REMAINING_TILE_M - 1u) / SPARK_REMAINING_TILE_M, 1u);
    draft_grid = dim3((request->vocab_size + SPARK_REMAINING_TILE_N - 1u) / SPARK_REMAINING_TILE_N, (request->draft_count + SPARK_REMAINING_TILE_M - 1u) / SPARK_REMAINING_TILE_M, 1u);
    cuda_status = cudaMemset(device_report, 0, sizeof(*device_report));
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingFp4LinearTiledKernel<<<token_output_grid, projection_block>>>(*request, device_inputs->token_input, device_inputs->fp4_linear_weight, device_outputs->fp4_linear_output);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingExpertKernel<<<SparkRemainingBlockCount(output_values), 256>>>(*request, device_inputs->token_input, device_inputs->expert_weight, device_inputs->expert_ids, device_outputs->expert_output, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingExpertFp4Kernel<<<SparkRemainingBlockCount(output_values), 256>>>(*request, device_inputs->token_input, device_inputs->expert_fp4_weight, device_inputs->expert_ids, device_outputs->expert_fp4_output);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingSharedTiledKernel<<<token_output_grid, projection_block>>>(*request, device_inputs->token_input, device_outputs->expert_output, device_inputs->shared_weight, device_outputs->shared_output);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingDenseMlaKernel<<<request->query_count, SPARK_REMAINING_REDUCE_THREADS>>>(*request, device_inputs->query, device_inputs->key, device_inputs->value, device_outputs->dense_mla_output);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingSparseMapMlaKernel<<<request->query_count, SPARK_REMAINING_REDUCE_THREADS>>>(*request, device_inputs->query, device_inputs->key, device_inputs->value, device_inputs->sparse_indices, device_inputs->physical_block_map, device_outputs->mapped_sparse_indices, device_outputs->sparse_mla_output, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingDraftProjectionTiledKernel<<<draft_grid, projection_block>>>(*request, device_inputs->draft_hidden, device_inputs->draft_weight, device_outputs->draft_logits);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingDraftArgmaxKernel<<<request->draft_count, SPARK_REMAINING_REDUCE_THREADS>>>(*request, device_outputs->draft_logits, device_outputs->draft_token_ids);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingSpecVerifyKernel<<<1, 32>>>(*request, device_inputs->expected_tokens, device_outputs->draft_token_ids, device_outputs->accepted_mask, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkRemainingChecksumTraceKernel<<<1, 32>>>(*request, *device_outputs, device_report);
        cuda_status = cudaGetLastError();
    }

    return cuda_status == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkRunCudaRemainingInferenceKernels(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *inputs, SparkCudaRemainingInferenceOutputs *outputs, SparkCudaRemainingInferenceReport *report)
{
    SparkCudaRemainingInferenceInputs device_inputs;
    SparkCudaRemainingInferenceOutputs device_outputs;
    SparkCudaRemainingInferenceReport *device_report;
    SparkCudaRemainingInferenceReport host_report;
    cudaError_t cuda_status;
    SparkStatus status;
    SparkStatus device_status;
    uint64_t token_values;
    uint64_t output_values;
    uint64_t expert_values;
    uint64_t query_values;
    uint64_t kv_values;
    uint64_t sparse_values;
    uint64_t draft_values;
    uint64_t draft_logits_values;
    float *device_token_input;
    uint8_t *device_fp4_linear_weight;
    float *device_expert_weight;
    uint8_t *device_expert_fp4_weight;
    float *device_shared_weight;
    uint32_t *device_expert_ids;
    float *device_query;
    float *device_key;
    float *device_value;
    uint32_t *device_sparse_indices;
    uint32_t *device_physical_block_map;
    float *device_draft_hidden;
    float *device_draft_weight;
    uint32_t *device_expected_tokens;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRemainingInferenceRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateRemainingCudaHostPointers(inputs, outputs, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    token_values = (uint64_t)request->token_count * request->hidden_size;
    output_values = (uint64_t)request->token_count * request->output_size;
    expert_values = (uint64_t)request->expert_count * request->hidden_size * request->output_size;
    query_values = (uint64_t)request->query_count * request->head_dim;
    kv_values = (uint64_t)request->kv_count * request->head_dim;
    sparse_values = (uint64_t)request->query_count * request->sparse_top_k;
    draft_values = (uint64_t)request->draft_count * request->hidden_size;
    draft_logits_values = (uint64_t)request->draft_count * request->vocab_size;
    memset(&device_inputs, 0, sizeof(device_inputs));
    memset(&device_outputs, 0, sizeof(device_outputs));
    device_report = 0;
    device_token_input = 0;
    device_fp4_linear_weight = 0;
    device_expert_weight = 0;
    device_expert_fp4_weight = 0;
    device_shared_weight = 0;
    device_expert_ids = 0;
    device_query = 0;
    device_key = 0;
    device_value = 0;
    device_sparse_indices = 0;
    device_physical_block_map = 0;
    device_draft_hidden = 0;
    device_draft_weight = 0;
    device_expected_tokens = 0;
    cuda_status = cudaMalloc((void **)&device_token_input, token_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_fp4_linear_weight, ((uint64_t)request->hidden_size * request->output_size + 1u) / 2u);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_expert_weight, expert_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_expert_fp4_weight, (expert_values + 1u) / 2u);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_shared_weight, (uint64_t)request->hidden_size * request->output_size * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_expert_ids, (uint64_t)request->token_count * sizeof(uint32_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_query, query_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_key, kv_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_value, kv_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_sparse_indices, sparse_values * sizeof(uint32_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_physical_block_map, (uint64_t)request->kv_count * sizeof(uint32_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_draft_hidden, draft_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_draft_weight, (uint64_t)request->hidden_size * request->vocab_size * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_expected_tokens, (uint64_t)request->draft_count * sizeof(uint32_t));
    device_inputs.token_input = device_token_input;
    device_inputs.fp4_linear_weight = device_fp4_linear_weight;
    device_inputs.expert_weight = device_expert_weight;
    device_inputs.expert_fp4_weight = device_expert_fp4_weight;
    device_inputs.shared_weight = device_shared_weight;
    device_inputs.expert_ids = device_expert_ids;
    device_inputs.query = device_query;
    device_inputs.key = device_key;
    device_inputs.value = device_value;
    device_inputs.sparse_indices = device_sparse_indices;
    device_inputs.physical_block_map = device_physical_block_map;
    device_inputs.draft_hidden = device_draft_hidden;
    device_inputs.draft_weight = device_draft_weight;
    device_inputs.expected_tokens = device_expected_tokens;
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.fp4_linear_output, output_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.expert_output, output_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.expert_fp4_output, output_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.shared_output, output_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.mapped_sparse_indices, sparse_values * sizeof(uint32_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.dense_mla_output, query_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.sparse_mla_output, query_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.draft_logits, draft_logits_values * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.draft_token_ids, (uint64_t)request->draft_count * sizeof(uint32_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_outputs.accepted_mask, (uint64_t)request->draft_count * sizeof(uint32_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc((void **)&device_report, sizeof(*device_report));
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.token_input, inputs->token_input, token_values * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.fp4_linear_weight, inputs->fp4_linear_weight, ((uint64_t)request->hidden_size * request->output_size + 1u) / 2u, cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.expert_weight, inputs->expert_weight, expert_values * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.expert_fp4_weight, inputs->expert_fp4_weight, (expert_values + 1u) / 2u, cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.shared_weight, inputs->shared_weight, (uint64_t)request->hidden_size * request->output_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.expert_ids, inputs->expert_ids, (uint64_t)request->token_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.query, inputs->query, query_values * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.key, inputs->key, kv_values * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.value, inputs->value, kv_values * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.sparse_indices, inputs->sparse_indices, sparse_values * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.physical_block_map, inputs->physical_block_map, (uint64_t)request->kv_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.draft_hidden, inputs->draft_hidden, draft_values * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.draft_weight, inputs->draft_weight, (uint64_t)request->hidden_size * request->vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy((void *)device_inputs.expected_tokens, inputs->expected_tokens, (uint64_t)request->draft_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemset(device_report, 0, sizeof(*device_report));
    if (cuda_status == cudaSuccess)
    {
        device_status = SparkRunCudaRemainingInferenceDeviceKernels(request, &device_inputs, &device_outputs, device_report);
        if (device_status != SPARK_STATUS_OK)
        {
            cuda_status = cudaErrorUnknown;
        }
    }
    if (cuda_status == cudaSuccess) cuda_status = cudaDeviceSynchronize();
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->fp4_linear_output, device_outputs.fp4_linear_output, output_values * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->expert_output, device_outputs.expert_output, output_values * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->expert_fp4_output, device_outputs.expert_fp4_output, output_values * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->shared_output, device_outputs.shared_output, output_values * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->mapped_sparse_indices, device_outputs.mapped_sparse_indices, sparse_values * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->dense_mla_output, device_outputs.dense_mla_output, query_values * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->sparse_mla_output, device_outputs.sparse_mla_output, query_values * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->draft_logits, device_outputs.draft_logits, draft_logits_values * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->draft_token_ids, device_outputs.draft_token_ids, (uint64_t)request->draft_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(outputs->accepted_mask, device_outputs.accepted_mask, (uint64_t)request->draft_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(&host_report, device_report, sizeof(host_report), cudaMemcpyDeviceToHost);
    cudaFree((void *)device_inputs.token_input);
    cudaFree((void *)device_inputs.fp4_linear_weight);
    cudaFree((void *)device_inputs.expert_weight);
    cudaFree((void *)device_inputs.expert_fp4_weight);
    cudaFree((void *)device_inputs.shared_weight);
    cudaFree((void *)device_inputs.expert_ids);
    cudaFree((void *)device_inputs.query);
    cudaFree((void *)device_inputs.key);
    cudaFree((void *)device_inputs.value);
    cudaFree((void *)device_inputs.sparse_indices);
    cudaFree((void *)device_inputs.physical_block_map);
    cudaFree((void *)device_inputs.draft_hidden);
    cudaFree((void *)device_inputs.draft_weight);
    cudaFree((void *)device_inputs.expected_tokens);
    cudaFree(device_outputs.fp4_linear_output);
    cudaFree(device_outputs.expert_output);
    cudaFree(device_outputs.expert_fp4_output);
    cudaFree(device_outputs.shared_output);
    cudaFree(device_outputs.mapped_sparse_indices);
    cudaFree(device_outputs.dense_mla_output);
    cudaFree(device_outputs.sparse_mla_output);
    cudaFree(device_outputs.draft_logits);
    cudaFree(device_outputs.draft_token_ids);
    cudaFree(device_outputs.accepted_mask);
    cudaFree(device_report);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    *report = host_report;
    if (report->invalid_expert_count != 0u || report->invalid_sparse_index_count != 0u || report->sentinel_violation_count != 0u || report->checksum_mismatch_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}
