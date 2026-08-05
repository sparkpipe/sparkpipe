#pragma once

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "modules/glm52_resident_decode_stage/source/cuda/config.h"

using Glm52Kv = LmKvLatent<
    GLM52_KV_BITS,
    GLM52_LATENT,
    GLM52_ROPE_DIM,
    GLM52_KV_PAGE_SLOTS>;
using Glm52IndexKv = LmKvLatent<
    GLM52_KV_BITS,
    GLM52_DSA_INDEX_DIM,
    0u,
    GLM52_KV_PAGE_SLOTS>;

#include "modules/glm52_resident_decode_stage/source/cuda/launch_shape.h"

#define GLM52_LAYER_TILE_N 128u
#define GLM52_LAYER_STAGES 2u
#define GLM52_LAYER_WARPS 8u
#define GLM52_HEAD_TILE 1024u

static_assert(
    GLM52_HIDDEN % LmBf16Format::kTileK == 0u,
    "GLM 5.2 hidden projections must cover every BF16 K tile");
static_assert(
    GLM52_QUERY_A_DIM % LmBf16Format::kTileK == 0u,
    "GLM 5.2 low-rank query projections must cover every BF16 K tile");
static_assert(
    SPARK_GLM52_MODEL_ROPE_INTERLEAVE == 1u,
    "GLM 5.2 query and key RoPE must use checkpoint interleaved pairing");
static_assert(
    GLM52_DSA_QUERY_DIM % LmBf16Format::kTileK == 0u,
    "GLM 5.2 DSA index queries must cover every BF16 K tile");
static_assert(
    (GLM52_ATTN_HEADS * GLM52_LATENT) % LmBf16Format::kTileK == 0u,
    "GLM 5.2 latent attention output must cover every BF16 K tile");
static_assert(
    GLM52_DENSE_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "GLM 5.2 dense FFN down projection must cover every BF16 K tile");
static_assert(
    GLM52_EXPERT_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "GLM 5.2 expert down projection must cover every BF16 K tile");
struct Glm52LayerBuffers
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
    // Bind-time fact, not a launch-time decision: the pack laid the up rows
    // immediately behind the gate rows, so one GEMM over the concatenated
    // tensor produces the [gate | up] layout two launches would. Zero means
    // unknown or non-contiguous, which takes the two-launch path.
    uint32_t dense_gate_up_fused;
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
};

static int32_t Glm52LaunchBf16Linear(
    const uint16_t *activation_bf16,
    const void *weight_bf16,
    uint16_t *output_bf16,
    const uint32_t *row_offset,
    uint32_t *tile_prefix,
    uint32_t rows,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_row_stride,
    uint32_t output_column_offset,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGemmArguments gemm;

    if (activation_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 ||
        row_offset == 0 || tile_prefix == 0 || rows == 0u ||
        input_dimension == 0u || output_dimension == 0u ||
        multiprocessors == 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = row_offset;
    gemm.group_tile_prefix = tile_prefix;
    gemm.output_bf16 = output_bf16;
    gemm.output_row_stride = output_row_stride;
    gemm.output_column_offset = output_column_offset;
    return LmGemmLaunch<
        LmBf16Format,
        GLM52_LAYER_TILE_N,
        LmBf16Format::kTileK,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
            activation_bf16,
            weight_bf16,
            rows,
            rows,
            1u,
            1u,
            input_dimension,
            output_dimension,
            multiprocessors,
            false,
            stream);
}

static int32_t Glm52LayerIndexer(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    if (Glm52LayerHasFullIndexer(layer_index) == 0u)
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
    status = Glm52LaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->index_q_weight,
        buffers->index_query_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_QUERY_A_DIM,
        GLM52_DSA_QUERY_DIM,
        GLM52_DSA_QUERY_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = Glm52LaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_k_weight,
        buffers->index_key_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_HIDDEN,
        GLM52_DSA_INDEX_DIM,
        GLM52_DSA_INDEX_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmLayerNormKernel<GLM52_LAYER_THREADS,uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_DSA_INDEX_DIM + 8u) * sizeof(float),
        stream,
        buffers->index_key_bf16,
        (const uint16_t *)buffers->index_norm_weight,
        (const uint16_t *)buffers->index_norm_bias,
        buffers->index_key_bf16,
        GLM52_DSA_INDEX_DIM,
        GLM52_DSA_INDEX_DIM,
        GLM52_DSA_INDEX_EPSILON);
    status = Glm52LaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_head_weight,
        buffers->index_head_weight_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_HIDDEN,
        GLM52_DSA_INDEX_HEADS,
        GLM52_DSA_INDEX_HEADS,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmRopePerHeadKernel<GLM52_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        dim3(rows,GLM52_DSA_INDEX_HEADS),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->index_query_bf16,
        buffers->positions,
        GLM52_DSA_INDEX_HEADS,
        GLM52_DSA_INDEX_DIM,
        0u,
        GLM52_ROPE_DIM,
        GLM52_ROPE_THETA);
    LM_LAUNCH(
        (LmRopeKernel<GLM52_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->index_key_bf16,
        buffers->positions,
        GLM52_DSA_INDEX_DIM,
        0u,
        GLM52_ROPE_DIM,
        GLM52_ROPE_THETA);
    LM_LAUNCH(
        (LmKvStoreKernel<Glm52IndexKv,GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->index_cache,
        buffers->index_key_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        GLM52_DSA_INDEX_DIM);
    if (context <= GLM52_DSA_SELECTED)
    {
        return cudaPeekAtLastError() == cudaSuccess
            ? LM_LAUNCH_OK
            : LM_LAUNCH_ERR_LAUNCH;
    }
    if (buffers->selection_scores == 0 || buffers->selected_positions == 0 ||
        buffers->head_candidate_token == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    LM_LAUNCH(
        (LmWeightedSparseScoreKernel<
            Glm52IndexKv,GLM52_LAYER_THREADS,GLM52_DSA_INDEX_DIM>),
        dim3(context,rows),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->index_query_bf16,
        buffers->index_head_weight_bf16,
        buffers->index_cache,
        buffers->sequence_of_row,
        buffers->context_length,
        buffers->row_positions,
        GLM52_DSA_INDEX_HEADS,
        GLM52_DSA_INDEX_SCALE / sqrtf((float)GLM52_DSA_INDEX_HEADS),
        buffers->selection_scores);
    LM_LAUNCH(
        (LmTopkHistogramKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->selection_scores,
        context,
        GLM52_DSA_SELECTED,
        buffers->head_candidate_token);
    LM_LAUNCH(
        (LmTopkGatherKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->selection_scores,
        context,
        GLM52_DSA_SELECTED,
        buffers->head_candidate_token,
        buffers->selected_positions,
        0);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t Glm52LayerAttention(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    const uint32_t *selected_positions;
    uint32_t selected_position_count;
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
        (context > GLM52_DSA_SELECTED &&
         (buffers->selected_positions == 0 ||
          buffers->selected_position_count != GLM52_DSA_SELECTED)))
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    selected_positions = context > GLM52_DSA_SELECTED
        ? buffers->selected_positions : 0;
    selected_position_count = context > GLM52_DSA_SELECTED
        ? buffers->selected_position_count : 0u;

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->attn_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);

    // The two low-rank norms are nonlinear and cannot be folded into a static
    // hidden-to-latent matrix. Run the checkpoint sequence exactly.
    status = Glm52LaunchBf16Linear(
        buffers->normed_bf16,
        buffers->q_a_weight,
        buffers->q_compressed_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_HIDDEN,
        GLM52_QUERY_A_DIM,
        GLM52_QUERY_A_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS,uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_QUERY_A_DIM + 8u) * sizeof(float),
        stream,
        buffers->q_compressed_bf16,
        0,
        (const uint16_t *)buffers->q_a_norm_weight,
        0,
        buffers->q_compressed_bf16,
        GLM52_QUERY_A_DIM,
        GLM52_QUERY_A_DIM,
        GLM52_RMS_EPSILON);
    status = Glm52LayerIndexer(
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
    status = Glm52LaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->q_b_weight,
        buffers->q_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_QUERY_A_DIM,
        GLM52_ATTN_HEADS * (GLM52_QK_NOPE_DIM + GLM52_ROPE_DIM),
        GLM52_ATTN_HEADS * (GLM52_QK_NOPE_DIM + GLM52_ROPE_DIM),
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = Glm52LaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kv_a_weight,
        buffers->kv_slot_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_HIDDEN,
        GLM52_LATENT_ROW,
        GLM52_LATENT_ROW,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS,uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_LATENT + 8u) * sizeof(float),
        stream,
        buffers->kv_slot_bf16,
        0,
        (const uint16_t *)buffers->kv_a_norm_weight,
        0,
        buffers->kv_slot_bf16,
        GLM52_LATENT,
        GLM52_LATENT_ROW,
        GLM52_RMS_EPSILON);

    LM_LAUNCH(
        (LmExtractRopePerHeadKernel<
            GLM52_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        dim3(rows, GLM52_ATTN_HEADS),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        buffers->query_rope_bf16,
        buffers->positions,
        GLM52_ATTN_HEADS,
        GLM52_QK_NOPE_DIM + GLM52_ROPE_DIM,
        GLM52_QK_NOPE_DIM,
        GLM52_ROPE_DIM,
        GLM52_ROPE_THETA);
    LM_LAUNCH(
        (LmRopeKernel<GLM52_LAYER_THREADS,LM_ROPE_INTERLEAVED>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->kv_slot_bf16,
        buffers->positions,
        GLM52_LATENT_ROW,
        GLM52_LATENT,
        GLM52_ROPE_DIM,
        GLM52_ROPE_THETA);
    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            GLM52_LAYER_THREADS,
            GLM52_QK_NOPE_DIM,
            GLM52_LATENT,
            GLM52_QK_NOPE_DIM + GLM52_ROPE_DIM,
            0u>),
        dim3(rows, GLM52_ATTN_HEADS),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        (const uint16_t *)buffers->kv_b_key_transposed_weight,
        buffers->query_latent_bf16,
        GLM52_ATTN_HEADS,
        rows);
    LM_LAUNCH(
        (LmKvStoreKernel<Glm52Kv, GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->cache,
        buffers->kv_slot_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        GLM52_LATENT_ROW);
    LM_LAUNCH(
        (LmLatentAttentionDecodeKernel<
            Glm52Kv,
            GLM52_ATTN_THREADS,
            GLM52_LATENT,
            GLM52_ROPE_DIM>),
        dim3(rows, GLM52_ATTN_HEADS),
        GLM52_ATTN_THREADS,
        0,
        stream,
        buffers->query_latent_bf16,
        buffers->query_rope_bf16,
        buffers->cache,
        buffers->sequence_of_row,
        buffers->context_length,
        selected_positions,
        selected_position_count,
        GLM52_ATTN_HEADS,
        buffers->qk_scale,
        buffers->attention_latent_bf16,
        buffers->row_positions);

    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            GLM52_LAYER_THREADS,GLM52_LATENT,GLM52_VALUE_DIM>),
        dim3(rows, GLM52_ATTN_HEADS),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->attention_latent_bf16,
        (const uint16_t *)buffers->kv_b_value_weight,
        buffers->attention_value_bf16,
        GLM52_ATTN_HEADS,
        rows);

    return Glm52LaunchBf16Linear(
        buffers->attention_value_bf16,
        buffers->output_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_ATTN_HEADS * GLM52_VALUE_DIM,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

static int32_t Glm52LayerDenseMlp(
    const Glm52LayerBuffers *buffers,
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
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);

    if (buffers->dense_gate_up_fused != 0u)
    {
        // Bind proved the up rows sit immediately behind the gate rows, so one
        // GEMM over the concatenated tensor writes the [gate | up] layout
        // directly - the two-launch form below re-reads the normed activation
        // and pays a second launch for the same weight bytes. Per-element math
        // is identical either way; only the launch count differs.
        status = Glm52LaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            GLM52_HIDDEN,
            GLM52_DENSE_INTERMEDIATE * 2u,
            GLM52_DENSE_INTERMEDIATE * 2u,
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
        status = Glm52LaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            GLM52_HIDDEN,
            GLM52_DENSE_INTERMEDIATE,
            GLM52_DENSE_INTERMEDIATE * 2u,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
        status = Glm52LaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_up_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            GLM52_HIDDEN,
            GLM52_DENSE_INTERMEDIATE,
            GLM52_DENSE_INTERMEDIATE * 2u,
            GLM52_DENSE_INTERMEDIATE,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }

    LM_LAUNCH(
        (LmSiluMulKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        GLM52_DENSE_INTERMEDIATE,
        true);

    return Glm52LaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->dense_down_weight,
        buffers->hidden_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_DENSE_INTERMEDIATE,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

template<uint32_t ExpertCodec>
static int32_t Glm52LayerMoe(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
    LmGemmArguments gemm;
    int32_t status;

    static_assert(ExpertCodec != SPARK_WEIGHT_CODEC_BF16,
        "GLM 5.2 routed experts require an explicit compressed codec");
    static_assert(GLM52_HIDDEN % ExpertFormat::kScaleGroup == 0u &&
        GLM52_EXPERT_INTERMEDIATE % ExpertFormat::kScaleGroup == 0u,
        "GLM 5.2 expert dimensions must contain complete codec scale groups");

    if (buffers == 0 || rows == 0u ||
        packed_rows != rows * GLM52_TOP_K ||
        buffers->attention_out_bf16 == 0 || buffers->residual_bf16 == 0 ||
        buffers->mlp_norm_weight == 0 || buffers->normed_bf16 == 0 ||
        buffers->router_weight == 0 || buffers->router_logits == 0 ||
        buffers->router_correction_bias == 0 ||
        buffers->route_expert == 0 || buffers->route_weight == 0 ||
        buffers->route_source_token == 0 || buffers->route_packed_row == 0 ||
        buffers->group_row_offset == 0 ||
        buffers->group_tile_prefix_w1 == 0 ||
        buffers->group_tile_prefix_w2 == 0 ||
        buffers->expert_w1_weight == 0 || buffers->expert_w1_scale == 0 ||
        buffers->expert_w2_weight == 0 || buffers->expert_w2_scale == 0 ||
        buffers->expert_out_bf16 == 0 || buffers->gate_up_bf16 == 0 ||
        buffers->intermediate_bf16 == 0 || buffers->hidden_bf16 == 0 ||
        buffers->shared_gate_up_weight == 0 ||
        buffers->shared_down_weight == 0 || buffers->shared_out_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->residual_bf16,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = buffers->dense_row_offset;
    gemm.group_tile_prefix = buffers->dense_tile_prefix;
    gemm.output_f32 = buffers->router_logits;
    status = LmGemmLaunch<
        LmBf16Format,
        GLM52_LAYER_TILE_N,
        LmBf16Format::kTileK,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
            buffers->normed_bf16,
            buffers->router_weight,
            rows,
            rows,
            1u,
            1u,
            GLM52_HIDDEN,
            GLM52_EXPERTS,
            multiprocessors,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmTopkSmallKernel<
            GLM52_LAYER_THREADS,
            GLM52_TOP_K,
            true,
            1u,
            1u,
            LM_TOPK_SCORE_SIGMOID>),
        rows,
        GLM52_LAYER_THREADS,
        2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),
        stream,
        buffers->router_logits,
        GLM52_EXPERTS,
        buffers->route_expert,
        buffers->route_weight,
        buffers->router_correction_bias,
        0,
        GLM52_ROUTED_SCALE);
    status = LmRouteBuild<GLM52_LAYER_THREADS, GLM52_EXPERTS>(
        buffers->route_expert,
        rows,
        packed_rows,
        GLM52_TOP_K,
        buffers->group_row_offset,
        buffers->route_packed_row,
        buffers->route_source_token,
        GLM52_GATE_UP_DIM,
        GLM52_HIDDEN,
        GLM52_LAYER_TILE_N,
        buffers->group_tile_prefix_w1,
        buffers->group_tile_prefix_w2,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
        buffers->expert_w1_scale,
        GLM52_EXPERTS,
        GLM52_GATE_UP_DIM,
        GLM52_HIDDEN);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
	gemm.source_row_map = buffers->route_source_token;
	gemm.source_row_count = rows;
    gemm.output_bf16 = buffers->gate_up_bf16;
    status = LmGemmWeightOnlyIndirectLaunch<
        ExpertFormat,
        GLM52_LAYER_TILE_N,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
			buffers->normed_bf16,
            buffers->expert_w1_weight,
            packed_rows,
            rows,
            GLM52_TOP_K,
            GLM52_EXPERTS,
            GLM52_HIDDEN,
            GLM52_GATE_UP_DIM,
            multiprocessors,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmSiluMulKernel<GLM52_LAYER_THREADS>),
        packed_rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        GLM52_EXPERT_INTERMEDIATE,
        false);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
        buffers->expert_w2_scale,
        GLM52_EXPERTS,
        GLM52_HIDDEN,
        GLM52_EXPERT_INTERMEDIATE);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
    gemm.output_bf16 = buffers->expert_out_bf16;
    status = LmGemmWeightOnlyLaunch<
        ExpertFormat,
        GLM52_LAYER_TILE_N,
        GLM52_LAYER_STAGES,
        GLM52_LAYER_WARPS>(
            &gemm,
            buffers->intermediate_bf16,
            buffers->expert_w2_weight,
            packed_rows,
            rows,
            GLM52_TOP_K,
            GLM52_EXPERTS,
            GLM52_EXPERT_INTERMEDIATE,
            GLM52_HIDDEN,
            multiprocessors,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmMoeFinalizeKernel<GLM52_LAYER_THREADS>),
        dim3(
            (GLM52_HIDDEN + GLM52_LAYER_THREADS - 1u) /
                GLM52_LAYER_THREADS,
            rows),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->expert_out_bf16,
        buffers->route_packed_row,
        buffers->route_weight,
        buffers->hidden_bf16,
        rows,
        GLM52_TOP_K,
        GLM52_HIDDEN);
    status = Glm52LaunchBf16Linear(
        buffers->normed_bf16,
        buffers->shared_gate_up_weight,
        buffers->gate_up_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_HIDDEN,
        GLM52_EXPERT_INTERMEDIATE * 2u,
        GLM52_EXPERT_INTERMEDIATE * 2u,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmSiluMulKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        GLM52_EXPERT_INTERMEDIATE,
        true);
    status = Glm52LaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->shared_down_weight,
        buffers->shared_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM52_EXPERT_INTERMEDIATE,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmAddRowsKernel<GLM52_LAYER_THREADS>),
        dim3(
            (GLM52_HIDDEN + GLM52_LAYER_THREADS - 1u) /
                GLM52_LAYER_THREADS,
            rows),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->hidden_bf16,
        buffers->shared_out_bf16,
        buffers->hidden_bf16,
        rows,
        GLM52_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t Glm52Head(
    const Glm52LayerBuffers *buffers,
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

    tiles = (vocabulary + GLM52_HEAD_TILE - 1u) / GLM52_HEAD_TILE;
    // The stream arrives SPLIT: the layer loop leaves the last MLP output in
    // hidden_bf16 and everything before it in residual_bf16, and the true
    // final stream is their sum. Norming hidden alone - as this did - drops
    // the whole residual stream at the one norm the model cannot afford to
    // lose. The fold is free: the kernel adds the residual and writes no
    // residual back when residual_out is null.
    // CONTRACT: a caller that flushes residual into hidden itself must zero
    // residual_bf16 afterwards, or the stream is counted twice here.
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM52_LAYER_THREADS, uint16_t>),
        rows,
        GLM52_LAYER_THREADS,
        (GLM52_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)head_norm_weight,
        0,
        buffers->normed_bf16,
        GLM52_HIDDEN,
        GLM52_HIDDEN,
        GLM52_RMS_EPSILON);
    LM_LAUNCH(
        (LmHeadCandidateKernel<GLM52_LAYER_THREADS, GLM52_HEAD_TILE>),
        dim3(tiles, rows),
        GLM52_LAYER_THREADS,
        0,
        stream,
        buffers->normed_bf16,
        (const uint16_t *)head_weight,
        token_ids,
        buffers->head_candidate_score,
        buffers->head_candidate_token,
        rows,
        GLM52_HIDDEN,
        vocabulary);
    LM_LAUNCH(
        (LmHeadCommitKernel<GLM52_LAYER_THREADS>),
        rows,
        GLM52_LAYER_THREADS,
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
