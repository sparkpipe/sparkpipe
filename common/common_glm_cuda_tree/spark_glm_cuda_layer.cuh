#pragma once

#include "runtime/gemm.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "common/common_glm_cuda_tree/spark_glm_cuda_config.h"

struct GlmKv
{
    static constexpr uint32_t kSlotBytes = GLM_KV_SLOT_BYTES;
    static constexpr uint32_t kPageSlots = GLM_KV_PAGE_SLOTS;
    static constexpr uint32_t kPageBytes =
        GLM_KV_SLOT_BYTES * GLM_KV_PAGE_SLOTS * GLM_LAYERS;
    static constexpr bool kGrows = true;
    static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
    { return position / GLM_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
    { return position % GLM_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
    { return (tokens + GLM_KV_PAGE_SLOTS - 1u) / GLM_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
    { return pages * (uint64_t)kPageBytes; }
};
using GlmIndexKv = LmKvLatent<
    GLM_KV_BITS,
    GLM_DSA_INDEX_DIM,
    0u,
    GLM_KV_PAGE_SLOTS>;

#include "common/common_glm_cuda_tree/spark_glm_cuda_launch_shape.h"

#define GLM_LAYER_TILE_N SPARK_LLM_TILE_N
#define GLM_LAYER_STAGES SPARK_LLM_TILE_STAGES
#define GLM_LAYER_WARPS SPARK_LLM_TILE_WARPS
#define GLM_LAYER_STREAM_EXPERT_ROWS 256u
#define GLM_HEAD_TILE SPARK_LLM_HEAD_TILE
#define GLM_HEAD_ROWS 16u

static_assert(
    GLM_HIDDEN % LmBf16Format::kTileK == 0u,
    "GLM hidden projections must cover every BF16 K tile");
static_assert(
    GLM_QUERY_A_DIM % LmBf16Format::kTileK == 0u,
    "GLM low-rank query projections must cover every BF16 K tile");
static_assert(
    SPARK_LLM_ROPE_INTERLEAVE == 1u,
    "GLM query and key RoPE must use checkpoint interleaved pairing");
static_assert(
    GLM_DSA_QUERY_DIM % LmBf16Format::kTileK == 0u,
    "GLM DSA index queries must cover every BF16 K tile");
static_assert(
    (GLM_ATTN_HEADS * GLM_LATENT) % LmBf16Format::kTileK == 0u,
    "GLM latent attention output must cover every BF16 K tile");
static_assert(
    GLM_DENSE_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "GLM dense FFN down projection must cover every BF16 K tile");
static_assert(
    GLM_EXPERT_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "GLM expert down projection must cover every BF16 K tile");
static_assert(
    GLM_QUERY_A_DIM % GLM_PROJECTION_GRANULE == 0u &&
        GLM_LATENT_ROW % GLM_PROJECTION_GRANULE == 0u,
    "split q_a/kv_a slices start on 16-byte output columns");
struct GlmLayerBuffers
{
    const uint32_t *dense_row_offset;
    uint32_t *dense_tile_prefix;

    const void *attn_norm_weight;
    const void *q_a_weight;
    const void *q_a_norm_weight;
    const void *q_b_weight;
    const void *kv_a_weight;
    const void *kv_a_norm_weight;
    const void *kv_b_key_transposed_weight;
    const void *kv_b_value_weight;
    const void *index_q_weight;
    const void *index_k_weight;
    const void *index_head_weight;
    const void *index_norm_weight;
    const void *index_norm_bias;
    float qk_scale;
    const void *output_weight;
    const void *mlp_norm_weight;
    const void *router_weight;
    const float *router_correction_bias;
    const void *dense_gate_weight;
    const void *dense_up_weight;
    const void *dense_down_weight;
    uint32_t dense_gate_up_fused;

    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t attn_heads;
    uint32_t q_b_rows;
    uint32_t attn_output_columns;
    uint32_t dense_gate_up_rows;
    uint32_t dense_intermediate;
    uint32_t expert_w1_rows;
    uint32_t expert_intermediate;
    uint32_t shared_gate_up_rows;
    uint32_t shared_intermediate;
    uint32_t head_vocabulary;
    const void *expert_w1_weight;
    const void *expert_w1_scale;
    const void *expert_w2_weight;
    const void *expert_w2_scale;
    const void *shared_gate_up_weight;
    const void *shared_down_weight;

    uint16_t *hidden_bf16;
    uint16_t *residual_bf16;
    uint16_t *normed_bf16;
    uint16_t *q_compressed_bf16;
    uint16_t *q_bf16;
    uint16_t *query_latent_bf16;
    uint16_t *query_rope_bf16;
    uint16_t *index_query_bf16;
    uint16_t *index_key_bf16;
    uint16_t *index_head_weight_bf16;
    uint16_t *kv_slot_bf16;
    uint16_t *attention_latent_bf16;
    uint16_t *attention_value_bf16;
    uint16_t *attention_out_bf16;
    uint16_t *gate_up_bf16;
    uint16_t *intermediate_bf16;
    uint16_t *expert_out_bf16;
    uint16_t *shared_out_bf16;
    float *router_logits;
    float *selection_scores;
    uint32_t *route_expert;
    float *route_weight;
    uint32_t *route_source_token;
    uint32_t *route_packed_row;
    float *head_candidate_score;
    uint32_t *head_candidate_token;
    uint32_t *output_token;
    float *output_score;
    uint32_t *group_row_offset;
    uint32_t *group_tile_prefix_w1;
    uint32_t *group_tile_prefix_w2;

    LmKvView cache;
    LmKvView index_cache;
    const uint32_t *sequence_of_row;
    const uint32_t *context_length;
    const uint32_t *positions;
    const uint32_t *row_positions;
    uint32_t *selected_positions;
    uint32_t selected_position_count;
    float *attention_split_partials;
    uint64_t attention_split_partial_blocks;
    uint32_t decode_split_context_threshold;
    uint32_t single_sequence_rows;
    uint16_t *projection_gather_bf16;
    uint16_t *projection_local_bf16;
};

static inline void GlmProjectionSlice(
    uint32_t total,
    uint32_t tp_degree,
    uint32_t tp_rank,
    uint32_t *first,
    uint32_t *count)
{
    uint32_t granules;
    uint32_t base;
    uint32_t extra;

    granules = total / GLM_PROJECTION_GRANULE;
    base = granules / tp_degree;
    extra = granules % tp_degree;
    *count = GLM_PROJECTION_GRANULE * (base + (tp_rank < extra ? 1u : 0u));
    *first = GLM_PROJECTION_GRANULE *
        (tp_rank * base + (tp_rank < extra ? tp_rank : extra));
}

static inline uint32_t GlmProjectionSliceMax(uint32_t total, uint32_t tp_degree)
{
    return GLM_PROJECTION_GRANULE *
        ((total / GLM_PROJECTION_GRANULE + tp_degree - 1u) / tp_degree);
}

static inline uint32_t GlmProjectionSliceWidth(uint32_t tp_degree)
{
    return GlmProjectionSliceMax(GLM_QUERY_A_DIM, tp_degree) +
        GlmProjectionSliceMax(GLM_LATENT_ROW, tp_degree);
}

static __device__ __forceinline__ void GlmProjectionSource(
    uint32_t column,
    uint32_t total,
    uint32_t tp_degree,
    uint32_t *rank,
    uint32_t *offset)
{
    uint32_t granules = total / GLM_PROJECTION_GRANULE;
    uint32_t base = granules / tp_degree;
    uint32_t extra = granules % tp_degree;
    uint32_t granule = column / GLM_PROJECTION_GRANULE;
    uint32_t boundary = extra * (base + 1u);
    uint32_t local;

    if (granule < boundary)
    {
        *rank = granule / (base + 1u);
        local = granule % (base + 1u);
    }
    else
    {
        *rank = extra + (granule - boundary) / base;
        local = (granule - boundary) % base;
    }
    *offset = local * GLM_PROJECTION_GRANULE + column % GLM_PROJECTION_GRANULE;
}

static __global__ void GlmProjectionUnpackKernel(
    const uint16_t *gathered,
    uint16_t *query_out,
    uint16_t *latent_out,
    uint32_t rows,
    uint32_t tp_degree,
    uint32_t width,
    uint32_t query_max)
{
    uint32_t row = blockIdx.x;
    uint32_t rank;
    uint32_t offset;

    for (uint32_t column = threadIdx.x; column < GLM_PROJECTION_GATHER_WIDTH;
         column += blockDim.x)
    {
        if (column < GLM_QUERY_A_DIM)
        {
            GlmProjectionSource(column, GLM_QUERY_A_DIM, tp_degree, &rank, &offset);
            query_out[(uint64_t)row * GLM_QUERY_A_DIM + column] =
                gathered[((uint64_t)rank * rows + row) * width + offset];
        }
        else
        {
            GlmProjectionSource(column - GLM_QUERY_A_DIM, GLM_LATENT_ROW, tp_degree, &rank, &offset);
            latent_out[(uint64_t)row * GLM_LATENT_ROW + column - GLM_QUERY_A_DIM] =
                gathered[((uint64_t)rank * rows + row) * width + query_max + offset];
        }
    }
}

static_assert(
    LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS ==
        SPARK_LLM_STAGE_ATTN_SPLIT_PARTITIONS,
    "the firmware's split-partials sizing must match the kernel's "
    "partition cap");

static int32_t GlmLaunchBf16Linear(
    const uint16_t *activation_bf16,
    const void *weight_bf16,
    uint16_t *output_bf16,
    uint32_t rows,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_row_stride,
    uint32_t output_column_offset,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    if (activation_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 ||
        rows == 0u || input_dimension == 0u || output_dimension == 0u ||
        multiprocessors == 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    if (rows <= LM_SKINNY_ROWS_WIDE)
        return LmSkinnyDense<LmBf16Format>(weight_bf16, activation_bf16, output_bf16, 0, rows, input_dimension, output_dimension, output_row_stride, output_column_offset, stream);
    return LmStreamGemmDense<LmBf16Format>(weight_bf16, LmScaleTensorNone(), activation_bf16, output_bf16, 0, rows, input_dimension, output_dimension, output_row_stride, output_column_offset, multiprocessors, stream);
}

static int32_t GlmLayerIndexer(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    if (GlmLayerHasFullIndexer(layer_index) == 0u)
    {
        return LM_LAUNCH_OK;
    }
    if (buffers == 0 || rows == 0u || context == 0u ||
        buffers->positions == 0 || buffers->sequence_of_row == 0 ||
        buffers->context_length == 0 ||
        !LmKvViewIsConfigured(buffers->index_cache) ||
        buffers->index_q_weight == 0 || buffers->index_k_weight == 0 ||
        buffers->index_head_weight == 0 || buffers->index_norm_weight == 0 ||
        buffers->index_norm_bias == 0 || buffers->index_query_bf16 == 0 ||
        buffers->index_key_bf16 == 0 ||
        buffers->index_head_weight_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    status = GlmLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_k_weight,
        buffers->index_key_bf16,
        rows,
        GLM_HIDDEN,
        GLM_DSA_INDEX_DIM,
        GLM_DSA_INDEX_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmLayerNormKernel<GLM_LAYER_THREADS,uint16_t>),
        rows,
        GLM_LAYER_THREADS,
        (GLM_DSA_INDEX_DIM + 8u) * sizeof(float),
        stream,
        buffers->index_key_bf16,
        (const uint16_t *)buffers->index_norm_weight,
        (const uint16_t *)buffers->index_norm_bias,
        buffers->index_key_bf16,
        GLM_DSA_INDEX_DIM,
        GLM_DSA_INDEX_DIM,
        GLM_DSA_INDEX_EPSILON);
    LM_LAUNCH(
        (LmRopeKernel<GLM_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->index_key_bf16,
        buffers->positions,
        GLM_DSA_INDEX_DIM,
        0u,
        GLM_ROPE_DIM,
        GLM_ROPE_THETA);
    LM_LAUNCH(
        (LmKvStoreKernel<GlmIndexKv,GLM_LAYER_THREADS>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->index_cache,
        buffers->index_key_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        GLM_DSA_INDEX_DIM);
    if (context <= GLM_DSA_SELECTED)
    {
        return cudaPeekAtLastError() == cudaSuccess
            ? LM_LAUNCH_OK
            : LM_LAUNCH_ERR_LAUNCH;
    }
    if (buffers->selection_scores == 0 || buffers->selected_positions == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    status = GlmLaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->index_q_weight,
        buffers->index_query_bf16,
        rows,
        GLM_QUERY_A_DIM,
        GLM_DSA_QUERY_DIM,
        GLM_DSA_QUERY_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = GlmLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_head_weight,
        buffers->index_head_weight_bf16,
        rows,
        GLM_HIDDEN,
        GLM_DSA_INDEX_HEADS,
        GLM_DSA_INDEX_HEADS,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmRopePerHeadKernel<GLM_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        dim3(rows,GLM_DSA_INDEX_HEADS),
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->index_query_bf16,
        buffers->positions,
        GLM_DSA_INDEX_HEADS,
        GLM_DSA_INDEX_DIM,
        0u,
        GLM_ROPE_DIM,
        GLM_ROPE_THETA);
    LM_LAUNCH(
        (LmWeightedSparseScoreKernel<
            GlmIndexKv,GLM_LAYER_THREADS,GLM_DSA_INDEX_DIM>),
        dim3(context,rows),
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->index_query_bf16,
        buffers->index_head_weight_bf16,
        buffers->index_cache,
        buffers->sequence_of_row,
        buffers->context_length,
        buffers->row_positions,
        GLM_DSA_INDEX_HEADS,
        GLM_DSA_INDEX_SCALE / sqrtf((float)GLM_DSA_INDEX_HEADS),
        buffers->selection_scores);
    LM_LAUNCH(
        (LmTopkExactKernel<GLM_LAYER_THREADS>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->selection_scores,
        context,
        GLM_DSA_SELECTED,
        buffers->selected_positions);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmLayerAttentionProject(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    uint32_t q_first;
    uint32_t q_count;
    uint32_t kv_first;
    uint32_t kv_count;
    uint32_t width;
    uint32_t query_max;
    int32_t status;

    if (buffers == 0 || rows == 0u || buffers->hidden_bf16 == 0 ||
        buffers->residual_bf16 == 0 || buffers->normed_bf16 == 0 ||
        buffers->attn_norm_weight == 0 ||
        (buffers->projection_local_bf16 != 0 &&
         (buffers->tp_degree == 0u || buffers->tp_rank >= buffers->tp_degree ||
          buffers->q_a_weight == 0 || buffers->kv_a_weight == 0)))
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM_LAYER_THREADS, uint16_t>),
        rows,
        GLM_LAYER_THREADS,
        (GLM_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->attn_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM_HIDDEN,
        GLM_HIDDEN,
        GLM_RMS_EPSILON);
    if (buffers->projection_local_bf16 == 0)
    {
        return cudaPeekAtLastError() == cudaSuccess
            ? LM_LAUNCH_OK
            : LM_LAUNCH_ERR_LAUNCH;
    }
    GlmProjectionSlice(GLM_QUERY_A_DIM, buffers->tp_degree, buffers->tp_rank,
        &q_first, &q_count);
    GlmProjectionSlice(GLM_LATENT_ROW, buffers->tp_degree, buffers->tp_rank,
        &kv_first, &kv_count);
    width = GlmProjectionSliceWidth(buffers->tp_degree);
    query_max = GlmProjectionSliceMax(GLM_QUERY_A_DIM, buffers->tp_degree);
    if (q_count != 0u && kv_count != 0u)
    {
        const LmSkinnyDenseTarget targets[2] = {
            {(const uint16_t *)buffers->q_a_weight + (uint64_t)q_first * GLM_HIDDEN, buffers->projection_local_bf16, 0, q_count, width, 0u},
            {(const uint16_t *)buffers->kv_a_weight + (uint64_t)kv_first * GLM_HIDDEN, buffers->projection_local_bf16, 0, kv_count, width, query_max}};
        status = LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, buffers->normed_bf16, rows, GLM_HIDDEN, stream);
        if (status != LM_LAUNCH_ERR_SHAPE)
            return status;
    }
    status = LM_LAUNCH_OK;
    if (q_count != 0u)
    {
        status = GlmLaunchBf16Linear(
            buffers->normed_bf16,
            (const uint16_t *)buffers->q_a_weight + (uint64_t)q_first * GLM_HIDDEN,
            buffers->projection_local_bf16,
            rows,
            GLM_HIDDEN,
            q_count,
            width,
            0u,
            multiprocessors,
            stream);
    }
    if (status == LM_LAUNCH_OK && kv_count != 0u)
    {
        status = GlmLaunchBf16Linear(
            buffers->normed_bf16,
            (const uint16_t *)buffers->kv_a_weight + (uint64_t)kv_first * GLM_HIDDEN,
            buffers->projection_local_bf16,
            rows,
            GLM_HIDDEN,
            kv_count,
            width,
            query_max,
            multiprocessors,
            stream);
    }
    return status;
}

static int32_t GlmLayerAttentionCore(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    const uint32_t *selected_positions;
    uint32_t selected_position_count;
    uint32_t gathered;
    int32_t status;

    if (buffers == 0 || rows == 0u || context == 0u ||
        buffers->qk_scale <= 0.0f || buffers->hidden_bf16 == 0 ||
        buffers->residual_bf16 == 0 || buffers->normed_bf16 == 0 ||
        buffers->attn_norm_weight == 0 || buffers->kv_slot_bf16 == 0 ||
        buffers->attention_latent_bf16 == 0 ||
        buffers->attention_value_bf16 == 0 ||
        buffers->attention_out_bf16 == 0 || buffers->output_weight == 0 ||
        buffers->sequence_of_row == 0 || buffers->context_length == 0 ||
        buffers->positions == 0 ||
        buffers->q_a_weight == 0 || buffers->q_a_norm_weight == 0 ||
        buffers->q_b_weight == 0 || buffers->kv_a_weight == 0 ||
        buffers->kv_a_norm_weight == 0 ||
        buffers->kv_b_key_transposed_weight == 0 ||
        buffers->kv_b_value_weight == 0 ||
        buffers->q_compressed_bf16 == 0 || buffers->q_bf16 == 0 ||
        buffers->query_latent_bf16 == 0 ||
        buffers->query_rope_bf16 == 0 ||
        (context > GLM_DSA_SELECTED &&
         (buffers->selected_positions == 0 ||
          buffers->selected_position_count != GLM_DSA_SELECTED)))
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    selected_positions = context > GLM_DSA_SELECTED
        ? buffers->selected_positions : 0;
    selected_position_count = context > GLM_DSA_SELECTED
        ? buffers->selected_position_count : 0u;
    gathered = buffers->projection_gather_bf16 != 0 ? 1u : 0u;

    if (gathered != 0u)
    {
        if (buffers->tp_degree == 0u)
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
        GlmProjectionUnpackKernel<<<rows, GLM_LAYER_THREADS, 0, stream>>>(
            buffers->projection_gather_bf16,
            buffers->q_compressed_bf16,
            buffers->kv_slot_bf16,
            rows,
            buffers->tp_degree,
            GlmProjectionSliceWidth(buffers->tp_degree),
            GlmProjectionSliceMax(GLM_QUERY_A_DIM, buffers->tp_degree));
        if (cudaPeekAtLastError() != cudaSuccess)
        {
            return LM_LAUNCH_ERR_LAUNCH;
        }
    }
    else
    {
        status = GlmLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->q_a_weight,
            buffers->q_compressed_bf16,
            rows,
            GLM_HIDDEN,
            GLM_QUERY_A_DIM,
            GLM_QUERY_A_DIM,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM_LAYER_THREADS,uint16_t>),
        rows,
        GLM_LAYER_THREADS,
        (GLM_QUERY_A_DIM + 8u) * sizeof(float),
        stream,
        buffers->q_compressed_bf16,
        0,
        (const uint16_t *)buffers->q_a_norm_weight,
        0,
        buffers->q_compressed_bf16,
        GLM_QUERY_A_DIM,
        GLM_QUERY_A_DIM,
        GLM_RMS_EPSILON);
    status = GlmLayerIndexer(
        buffers,
        rows,
        context,
        layer_index,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = GlmLaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->q_b_weight,
        buffers->q_bf16,
        rows,
        GLM_QUERY_A_DIM,
        buffers->q_b_rows,
        buffers->q_b_rows,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    if (gathered == 0u)
    {
        status = GlmLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->kv_a_weight,
            buffers->kv_slot_bf16,
            rows,
            GLM_HIDDEN,
            GLM_LATENT_ROW,
            GLM_LATENT_ROW,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM_LAYER_THREADS,uint16_t>),
        rows,
        GLM_LAYER_THREADS,
        (GLM_LATENT + 8u) * sizeof(float),
        stream,
        buffers->kv_slot_bf16,
        0,
        (const uint16_t *)buffers->kv_a_norm_weight,
        0,
        buffers->kv_slot_bf16,
        GLM_LATENT,
        GLM_LATENT_ROW,
        GLM_RMS_EPSILON);

    LM_LAUNCH(
        (LmExtractRopePerHeadKernel<
            GLM_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        dim3(rows, buffers->attn_heads),
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        buffers->query_rope_bf16,
        buffers->positions,
        buffers->attn_heads,
        GLM_QK_NOPE_DIM + GLM_ROPE_DIM,
        GLM_QK_NOPE_DIM,
        GLM_ROPE_DIM,
        GLM_ROPE_THETA);
    LM_LAUNCH(
        (LmRopeKernel<GLM_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->kv_slot_bf16,
        buffers->positions,
        GLM_LATENT_ROW,
        GLM_LATENT,
        GLM_ROPE_DIM,
        GLM_ROPE_THETA);
    if (LmPerHeadProjectChainLaunch<
            GLM_LAYER_THREADS,
            GLM_QK_NOPE_DIM,
            GLM_LATENT,
            GLM_QK_NOPE_DIM + GLM_ROPE_DIM,
            0u>(
            buffers->q_bf16,
            (const uint16_t *)buffers->kv_b_key_transposed_weight,
            buffers->query_latent_bf16,
            buffers->attn_heads,
            rows,
            stream) != cudaSuccess)
    {
        return LM_LAUNCH_ERR_LAUNCH;
    }
    LM_LAUNCH(
        (LmKvStoreKernel<GlmKv, GLM_LAYER_THREADS>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->cache,
        buffers->kv_slot_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        GLM_LATENT_ROW);
    if (buffers->single_sequence_rows != 0u && rows > LM_SKINNY_ROWS && selected_positions == 0)
    {
        if (LmLatentAttentionPrefillLaunch<GlmKv, GLM_LATENT, GLM_ROPE_DIM>(
                buffers->query_latent_bf16,
                buffers->query_rope_bf16,
                buffers->cache,
                buffers->sequence_of_row,
                buffers->row_positions,
                buffers->attn_heads,
                buffers->qk_scale,
                buffers->attention_latent_bf16,
                rows,
                stream) != cudaSuccess)
        {
            return LM_LAUNCH_ERR_LAUNCH;
        }
    }
    else if (LmLatentRopeHeadsSplitLaunch<
            GlmKv, GLM_ATTN_THREADS, GLM_LATENT, GLM_ROPE_DIM, true>(
            buffers->query_latent_bf16,
            buffers->query_rope_bf16,
            buffers->cache,
            buffers->sequence_of_row,
            buffers->context_length,
            selected_positions,
            selected_position_count,
            buffers->attn_heads,
            buffers->qk_scale,
            buffers->attention_latent_bf16,
            buffers->row_positions,
            rows,
            context > GLM_DSA_SELECTED ? GLM_DSA_SELECTED : context,
            buffers->decode_split_context_threshold,
            buffers->attention_split_partials,
            (uint32_t)buffers->attention_split_partial_blocks,
            multiprocessors,
            stream) != cudaSuccess)
    {
        return LM_LAUNCH_ERR_LAUNCH;
    }

    if (LmPerHeadProjectChainLaunch<
            GLM_LAYER_THREADS,GLM_LATENT,GLM_VALUE_DIM>(
            buffers->attention_latent_bf16,
            (const uint16_t *)buffers->kv_b_value_weight,
            buffers->attention_value_bf16,
            buffers->attn_heads,
            rows,
            stream) != cudaSuccess)
    {
        return LM_LAUNCH_ERR_LAUNCH;
    }

    return GlmLaunchBf16Linear(
        buffers->attention_value_bf16,
        buffers->output_weight,
        buffers->attention_out_bf16,
        rows,
        buffers->attn_output_columns,
        GLM_HIDDEN,
        GLM_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

static int32_t GlmLayerAttention(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    if (buffers == 0 || buffers->projection_gather_bf16 != 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    status = GlmLayerAttentionProject(buffers, rows, multiprocessors, stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    return GlmLayerAttentionCore(buffers, rows, context, layer_index,
        multiprocessors, stream);
}

static int32_t GlmLayerDenseMlp(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    if (buffers == 0 || rows == 0u || buffers->attention_out_bf16 == 0 ||
        buffers->residual_bf16 == 0 || buffers->mlp_norm_weight == 0 ||
        buffers->normed_bf16 == 0 || buffers->dense_gate_weight == 0 ||
        buffers->dense_up_weight == 0 || buffers->dense_down_weight == 0 ||
        buffers->gate_up_bf16 == 0 || buffers->intermediate_bf16 == 0 ||
        buffers->hidden_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM_LAYER_THREADS, uint16_t>),
        rows,
        GLM_LAYER_THREADS,
        (GLM_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM_HIDDEN,
        GLM_HIDDEN,
        GLM_RMS_EPSILON);

    if (buffers->dense_gate_up_fused != 0u)
    {
        status = GlmLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            rows,
            GLM_HIDDEN,
            buffers->dense_gate_up_rows,
            buffers->dense_gate_up_rows,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }
    else
    {
        status = GlmLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            rows,
            GLM_HIDDEN,
            buffers->dense_intermediate,
            buffers->dense_gate_up_rows,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
        status = GlmLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_up_weight,
            buffers->gate_up_bf16,
            rows,
            GLM_HIDDEN,
            buffers->dense_intermediate,
            buffers->dense_gate_up_rows,
            buffers->dense_intermediate,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }

    LM_LAUNCH(
        (LmSiluMulKernel<GLM_LAYER_THREADS>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->dense_intermediate,
        false);

    return GlmLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->dense_down_weight,
        buffers->hidden_bf16,
        rows,
        buffers->dense_intermediate,
        GLM_HIDDEN,
        GLM_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

template<uint32_t ExpertCodec>
static int32_t GlmLayerMoeValidate(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t require_expert_weights)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
    int32_t status;

    static_assert(ExpertFormat::kScaleGroup == 0u ||
        (GLM_HIDDEN % ExpertFormat::kScaleGroup == 0u &&
        GLM_EXPERT_INTERMEDIATE % ExpertFormat::kScaleGroup == 0u),
        "GLM expert dimensions must contain complete codec scale groups");

    if (buffers == 0 || rows == 0u ||
        packed_rows != rows * GLM_TOP_K ||
        buffers->attention_out_bf16 == 0 || buffers->residual_bf16 == 0 ||
        buffers->mlp_norm_weight == 0 || buffers->normed_bf16 == 0 ||
        buffers->router_weight == 0 || buffers->router_logits == 0 ||
        buffers->router_correction_bias == 0 ||
        buffers->route_expert == 0 || buffers->route_weight == 0 ||
        buffers->route_source_token == 0 || buffers->route_packed_row == 0 ||
        buffers->group_row_offset == 0 ||
        buffers->group_tile_prefix_w1 == 0 ||
        buffers->group_tile_prefix_w2 == 0 ||
        buffers->expert_out_bf16 == 0 || buffers->gate_up_bf16 == 0 ||
        buffers->intermediate_bf16 == 0 || buffers->hidden_bf16 == 0 ||
        buffers->shared_gate_up_weight == 0 ||
        buffers->shared_down_weight == 0 || buffers->shared_out_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    if (require_expert_weights != 0u)
    {
        if (buffers->expert_w1_weight == 0 ||
            (ExpertCodec != SPARK_WEIGHT_CODEC_BF16 && buffers->expert_w1_scale == 0) ||
            buffers->expert_w2_weight == 0 ||
            (ExpertCodec != SPARK_WEIGHT_CODEC_BF16 && buffers->expert_w2_scale == 0))
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
    }
    (void)status;
    return LM_LAUNCH_OK;
}
static int32_t GlmLayerMoeRouterLogits(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    status = rows <= LM_SKINNY_ROWS_WIDE
        ? LmSkinnyDense<LmBf16Format>(buffers->router_weight, buffers->normed_bf16, 0, buffers->router_logits, rows, GLM_HIDDEN, GLM_EXPERTS, 0u, 0u, stream)
        : LmStreamGemmDense<LmBf16Format>(buffers->router_weight, LmScaleTensorNone(), buffers->normed_bf16, 0, buffers->router_logits, rows, GLM_HIDDEN, GLM_EXPERTS, 0u, 0u, multiprocessors, stream);
    if (status != LM_LAUNCH_OK)
        return status;
    return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmLayerMoeRouteSelect(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    cudaStream_t stream)
{
    if (LmTopkRouteLaunch<GLM_LAYER_THREADS, GLM_TOP_K, true, LM_TOPK_SCORE_SIGMOID>(
            rows,
            buffers->router_logits,
            GLM_EXPERTS,
            buffers->route_expert,
            buffers->route_weight,
            buffers->router_correction_bias,
            0,
            GLM_ROUTED_SCALE,
            stream) != cudaSuccess)
    {
        return LM_LAUNCH_ERR_LAUNCH;
    }
    return LM_LAUNCH_OK;
}

static int32_t GlmLayerMoeRoutePack(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    cudaStream_t stream)
{
    int32_t status;

    status = LmRouteBuild<GLM_LAYER_THREADS, GLM_EXPERTS>(
        buffers->route_expert,
        rows,
        packed_rows,
        GLM_TOP_K,
        buffers->group_row_offset,
        buffers->route_packed_row,
        buffers->route_source_token,
        buffers->expert_w1_rows,
        GLM_HIDDEN,
        GLM_LAYER_TILE_N,
        buffers->group_tile_prefix_w1,
        buffers->group_tile_prefix_w2,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t ExpertCodec>
static int32_t GlmLayerMoeRoute(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status = GlmLayerMoeValidate<ExpertCodec>(buffers,rows,packed_rows,0u);
    if (status != LM_LAUNCH_OK)
        return status;

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM_LAYER_THREADS, uint16_t>),
        rows,
        GLM_LAYER_THREADS,
        (GLM_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM_HIDDEN,
        GLM_HIDDEN,
        GLM_RMS_EPSILON);

    status = GlmLayerMoeRouterLogits(buffers,rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = GlmLayerMoeRouteSelect(buffers,rows,stream);
    if (status != LM_LAUNCH_OK)
        return status;
    return GlmLayerMoeRoutePack(buffers,rows,packed_rows,stream);
}

template<uint32_t ExpertCodec>
static int32_t GlmLayerMoeExpertsGateUp(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
    LmGemmArguments gemm;
    int32_t status;

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
        buffers->expert_w1_scale,
        GLM_EXPERTS,
        buffers->expert_w1_rows,
        GLM_HIDDEN);
    if constexpr ( LmStreamWeight<ExpertFormat>::kSupported )
    {
        if (rows >= GLM_LAYER_STREAM_EXPERT_ROWS)
        {
            status = LmStreamGemmGrouped<ExpertFormat>(buffers->expert_w1_weight, gemm.scale_b, buffers->normed_bf16, buffers->gate_up_bf16, buffers->group_row_offset, buffers->route_source_token, GLM_EXPERTS, packed_rows, GLM_HIDDEN, buffers->expert_w1_rows, multiprocessors, stream);
            return status == LM_LAUNCH_OK && cudaPeekAtLastError() != cudaSuccess ? LM_LAUNCH_ERR_LAUNCH : status;
        }
    }
    status = rows == 1u ? LmSkinnyExperts<ExpertFormat>(buffers->expert_w1_weight, gemm.scale_b, buffers->normed_bf16, buffers->gate_up_bf16, buffers->route_expert, buffers->route_packed_row, packed_rows, GLM_TOP_K, 0u, GLM_HIDDEN, buffers->expert_w1_rows, stream) : LM_LAUNCH_ERR_SHAPE;
    if (status == LM_LAUNCH_ERR_SHAPE)
        status = LmSkinnyGroupedExperts<ExpertFormat>(buffers->expert_w1_weight, gemm.scale_b, buffers->normed_bf16, buffers->gate_up_bf16, buffers->group_row_offset, buffers->route_source_token, GLM_EXPERTS, packed_rows, 0u, GLM_HIDDEN, buffers->expert_w1_rows, stream);
    if (status != LM_LAUNCH_ERR_SHAPE)
        return status == LM_LAUNCH_OK && cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : status == LM_LAUNCH_OK ? LM_LAUNCH_ERR_LAUNCH : status;
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
	gemm.source_row_map = buffers->route_source_token;
	gemm.source_row_count = rows;
    gemm.output_bf16 = buffers->gate_up_bf16;
    status = LmGemmWeightOnlyIndirectLaunch<
        ExpertFormat,
        GLM_LAYER_TILE_N,
        GLM_LAYER_STAGES,
        GLM_LAYER_WARPS>(
            &gemm,
			buffers->normed_bf16,
            buffers->expert_w1_weight,
            packed_rows,
            rows,
            GLM_TOP_K,
            GLM_EXPERTS,
            GLM_HIDDEN,
            buffers->expert_w1_rows,
            multiprocessors,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t ExpertCodec>
static int32_t GlmLayerMoeExpertsDown(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
    LmGemmArguments gemm;
    int32_t status;

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
        buffers->expert_w2_scale,
        GLM_EXPERTS,
        GLM_HIDDEN,
        buffers->expert_intermediate);
    if constexpr ( LmStreamWeight<ExpertFormat>::kSupported )
    {
        if (rows >= GLM_LAYER_STREAM_EXPERT_ROWS)
        {
            status = LmStreamGemmGrouped<ExpertFormat>(buffers->expert_w2_weight, gemm.scale_b, buffers->intermediate_bf16, buffers->expert_out_bf16, buffers->group_row_offset, 0, GLM_EXPERTS, packed_rows, buffers->expert_intermediate, GLM_HIDDEN, multiprocessors, stream);
            return status == LM_LAUNCH_OK && cudaPeekAtLastError() != cudaSuccess ? LM_LAUNCH_ERR_LAUNCH : status;
        }
    }
    status = LmSkinnyExperts<ExpertFormat>(buffers->expert_w2_weight, gemm.scale_b, buffers->intermediate_bf16, buffers->expert_out_bf16, buffers->route_expert, buffers->route_packed_row, packed_rows, GLM_TOP_K, 1u, buffers->expert_intermediate, GLM_HIDDEN, stream);
    if (status == LM_LAUNCH_ERR_SHAPE)
        status = LmSkinnyGroupedExperts<ExpertFormat>(buffers->expert_w2_weight, gemm.scale_b, buffers->intermediate_bf16, buffers->expert_out_bf16, buffers->group_row_offset, buffers->route_source_token, GLM_EXPERTS, packed_rows, 1u, buffers->expert_intermediate, GLM_HIDDEN, stream);
    if (status != LM_LAUNCH_ERR_SHAPE)
        return status == LM_LAUNCH_OK && cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : status == LM_LAUNCH_OK ? LM_LAUNCH_ERR_LAUNCH : status;
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
    gemm.output_bf16 = buffers->expert_out_bf16;
    status = LmGemmWeightOnlyLaunch<
        ExpertFormat,
        GLM_LAYER_TILE_N,
        GLM_LAYER_STAGES,
        GLM_LAYER_WARPS>(
            &gemm,
            buffers->intermediate_bf16,
            buffers->expert_w2_weight,
            packed_rows,
            rows,
            GLM_TOP_K,
            GLM_EXPERTS,
        buffers->expert_intermediate,
        GLM_HIDDEN,
        multiprocessors,
        true,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t ExpertCodec>
static int32_t GlmLayerMoeRoutedExperts(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    status = GlmLayerMoeExpertsGateUp<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;

    LM_LAUNCH(
        (LmSiluMulKernel<GLM_LAYER_THREADS>),
        packed_rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->expert_intermediate,
        false);

    status = GlmLayerMoeExpertsDown<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;

    LM_LAUNCH(
        (LmMoeFinalizeKernel<GLM_LAYER_THREADS>),
        dim3(
            (GLM_HIDDEN + GLM_LAYER_THREADS - 1u) /
                GLM_LAYER_THREADS,
            rows),
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->expert_out_bf16,
        buffers->route_packed_row,
        buffers->route_weight,
        buffers->hidden_bf16,
        rows,
        GLM_TOP_K,
        GLM_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmLayerMoeSharedCombine(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    status = GlmLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->shared_gate_up_weight,
        buffers->gate_up_bf16,
        rows,
        GLM_HIDDEN,
        buffers->shared_gate_up_rows,
        buffers->shared_gate_up_rows,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmSiluMulKernel<GLM_LAYER_THREADS>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->shared_intermediate,
        false);
    return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmLayerMoeSharedExperts(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    status = GlmLayerMoeSharedCombine(buffers,rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = GlmLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->shared_down_weight,
        buffers->shared_out_bf16,
        rows,
        buffers->shared_intermediate,
        GLM_HIDDEN,
        GLM_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmAddRowsKernel<GLM_LAYER_THREADS>),
        dim3(
            (GLM_HIDDEN + GLM_LAYER_THREADS - 1u) /
                GLM_LAYER_THREADS,
            rows),
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->hidden_bf16,
        buffers->shared_out_bf16,
        buffers->hidden_bf16,
        rows,
        GLM_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t ExpertCodec>
static int32_t GlmLayerMoeExperts(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status = GlmLayerMoeValidate<ExpertCodec>(buffers,rows,packed_rows,1u);
    if (status != LM_LAUNCH_OK)
        return status;
    status = GlmLayerMoeRoutedExperts<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;
    return GlmLayerMoeSharedExperts(buffers,rows,multiprocessors,stream);
}
template<uint32_t ExpertCodec>
static int32_t GlmLayerMoe(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status = GlmLayerMoeRoute<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;
    return GlmLayerMoeExperts<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
}

static int32_t GlmHead(
    const GlmLayerBuffers *buffers,
    const void *head_norm_weight,
    const void *head_weight,
    const uint32_t *token_ids,
    uint32_t vocabulary,
    uint32_t rows,
    cudaStream_t stream)
{
    uint32_t tiles;

    if (buffers == 0 || head_norm_weight == 0 || head_weight == 0 ||
        rows == 0u || vocabulary == 0u || buffers->hidden_bf16 == 0 ||
        buffers->residual_bf16 == 0 || buffers->normed_bf16 == 0 ||
        buffers->head_candidate_score == 0 ||
        buffers->head_candidate_token == 0 || buffers->output_token == 0 ||
        buffers->output_score == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    tiles = (vocabulary + GLM_HEAD_TILE - 1u) / GLM_HEAD_TILE;
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM_LAYER_THREADS, uint16_t>),
        rows,
        GLM_LAYER_THREADS,
        (GLM_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)head_norm_weight,
        0,
        buffers->normed_bf16,
        GLM_HIDDEN,
        GLM_HIDDEN,
        GLM_RMS_EPSILON);
    if (rows > 1u)
        LM_LAUNCH(
            (LmHeadCandidateRowsKernel<GLM_LAYER_THREADS, GLM_HEAD_TILE, GLM_HEAD_ROWS>),
            dim3(tiles, (rows + GLM_HEAD_ROWS - 1u) / GLM_HEAD_ROWS),
            GLM_LAYER_THREADS,
            0,
            stream,
            buffers->normed_bf16,
            (const uint16_t *)head_weight,
            token_ids,
            buffers->head_candidate_score,
            buffers->head_candidate_token,
            rows,
            GLM_HIDDEN,
            vocabulary);
    else
        LM_LAUNCH(
            (LmHeadCandidateKernel<GLM_LAYER_THREADS, GLM_HEAD_TILE>),
            dim3(tiles, rows),
            GLM_LAYER_THREADS,
            0,
            stream,
            buffers->normed_bf16,
            (const uint16_t *)head_weight,
            token_ids,
            buffers->head_candidate_score,
            buffers->head_candidate_token,
            rows,
            GLM_HIDDEN,
            vocabulary);
    LM_LAUNCH(
        (LmHeadCommitKernel<GLM_LAYER_THREADS>),
        rows,
        GLM_LAYER_THREADS,
        0,
        stream,
        buffers->head_candidate_score,
        buffers->head_candidate_token,
        tiles,
        buffers->output_token,
        buffers->output_score,
        rows);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmHeadCertifiedB1(
    const GlmLayerBuffers *buffers,
    const void *head_norm_weight,
    const void *head_weight,
    const uint8_t *certified_payload,
    const float *certified_scale,
    const float *certified_norm,
    void *certified_scratch,
    uint32_t *candidate_ids,
    uint32_t *screened_count,
    uint32_t rank_offset,
    uint32_t vocabulary,
    cudaStream_t stream)
{
    cudaError_t status;
    if (buffers == 0 || head_norm_weight == 0 || head_weight == 0 ||
        certified_payload == 0 || certified_scale == 0 ||
        certified_norm == 0 || certified_scratch == 0 ||
        candidate_ids == 0 || screened_count == 0 ||
        buffers->hidden_bf16 == 0 || buffers->residual_bf16 == 0 ||
        buffers->normed_bf16 == 0 || buffers->output_token == 0 ||
        buffers->output_score == 0)
        return LM_LAUNCH_ERR_SHAPE;
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM_LAYER_THREADS, uint16_t>),
        1u,
        GLM_LAYER_THREADS,
        (GLM_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)head_norm_weight,
        0,
        buffers->normed_bf16,
        GLM_HIDDEN,
        GLM_HIDDEN,
        GLM_RMS_EPSILON);
    status = SparkLmHostLaunchHeadCertifiedFp8B1WithScore(
        stream, buffers->normed_bf16, head_weight, certified_payload,
        certified_scale, certified_norm, certified_scratch, candidate_ids,
        screened_count, buffers->output_token, buffers->output_score,
        rank_offset, 1u, vocabulary, GLM_HIDDEN);
    return status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
