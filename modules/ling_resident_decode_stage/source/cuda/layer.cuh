#pragma once

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "sparkpipe/spark_ling_resident_decode_stage_firmware.h"
#include "modules/ling_resident_decode_stage/source/cuda/config.h"

struct LingKv
{
    static constexpr uint32_t kSlotBytes = LING_KV_SLOT_BYTES;
    static constexpr uint32_t kPageSlots = LING_KV_PAGE_SLOTS;
    static constexpr uint32_t kPageBytes =
        LING_KV_SLOT_BYTES * LING_KV_PAGE_SLOTS;
    static constexpr bool kGrows = true;
    static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
    { return position / LING_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
    { return position % LING_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
    { return (tokens + LING_KV_PAGE_SLOTS - 1u) / LING_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
    { return pages * (uint64_t)kPageBytes; }
};

#include "modules/ling_resident_decode_stage/source/cuda/launch_shape.h"

#define LING_LAYER_TILE_N 128u
#define LING_LAYER_STAGES 2u
#define LING_LAYER_WARPS 8u
#define LING_HEAD_TILE 1024u

static_assert(
    LING_HIDDEN % LmBf16Format::kTileK == 0u,
    "ling hidden projections must cover every BF16 K tile");
static_assert(
    LING_QUERY_DIM % LmBf16Format::kTileK == 0u,
    "ling query projection must cover every BF16 K tile");
static_assert(
    (LING_ATTN_HEADS * LING_LATENT) % LmBf16Format::kTileK == 0u,
    "ling latent attention output must cover every BF16 K tile");
static_assert(
    LING_DENSE_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "ling dense FFN down projection must cover every BF16 K tile");
static_assert(
    LING_EXPERT_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "ling expert down projection must cover every BF16 K tile");
static_assert(
    LING_KDA_QK_DIM % LmBf16Format::kTileK == 0u,
    "ling fused KDA q|k|v|beta rows must cover every BF16 K tile");
struct LingLayerBuffers
{
    const uint32_t *dense_row_offset;
    uint32_t *dense_tile_prefix;

    const void *attn_norm_weight;
    const void *q_a_weight;
    const void *kv_a_weight;
    const void *kv_a_norm_weight;
    const void *kv_b_key_transposed_weight;
    const void *kv_b_value_weight;
    const void *attn_gate_weight;
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
    uint32_t layer_index;
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
    uint16_t *q_bf16;
    uint16_t *query_latent_bf16;
    uint16_t *query_rope_bf16;
    uint16_t *attn_gate_bf16;
    uint16_t *kv_slot_bf16;
    uint16_t *attention_latent_bf16;
    uint16_t *attention_value_bf16;
    uint16_t *attention_out_bf16;
    uint16_t *gate_up_bf16;
    uint16_t *intermediate_bf16;
    uint16_t *expert_out_bf16;
    uint16_t *shared_out_bf16;
    float *router_logits;
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

    const void *kda_qkv_beta_weight;
    const void *kda_decay_weight;
    const void *kda_gate_weight;
    const void *kda_q_conv_weight;
    const void *kda_k_conv_weight;
    const void *kda_v_conv_weight;
    const float *kda_decay_bias;
    const float *kda_head_log_scale;
    const void *kda_out_norm_weight;
    const void *kda_out_weight;
    uint32_t kda_heads;
    uint8_t *kda_state_pool;
    uint32_t kda_state_slot_bytes;
    const uint32_t *kda_state_index;
    const uint32_t *sequence_row_begin;
    uint16_t *kda_q_window;
    uint16_t *kda_k_window;
    uint16_t *kda_v_window;
    uint16_t *fused_qkvb_bf16;
    uint16_t *kda_output_bf16;
    uint16_t *kda_beta_logit;
    uint16_t *kda_gate_bf16;
    uint16_t *kda_decay_logit_bf16;
    float *kda_retention;
    float *kda_write_gate;

    LmKvView cache;
    const uint32_t *sequence_of_row;
    const uint32_t *context_length;
    const uint32_t *positions;
    const uint32_t *row_positions;
    float *attention_split_partials;
    uint64_t attention_split_partial_blocks;
    uint32_t decode_split_context_threshold;
};

static_assert(
    LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS ==
        SPARK_LING_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTITIONS,
    "the firmware's split-partials sizing must match the kernel's "
    "partition cap");

static int32_t LingLaunchBf16Linear(
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
        LING_LAYER_TILE_N,
        LmBf16Format::kTileK,
        LING_LAYER_STAGES,
        LING_LAYER_WARPS>(
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

static int32_t LingLayerAttention(
    const LingLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;

    (void)layer_index;
    (void)context;
    if (buffers == 0 || rows == 0u ||
        buffers->qk_scale <= 0.0f || buffers->hidden_bf16 == 0 ||
        buffers->normed_bf16 == 0 ||
        buffers->attn_norm_weight == 0 || buffers->kv_slot_bf16 == 0 ||
        buffers->attention_latent_bf16 == 0 ||
        buffers->attention_value_bf16 == 0 ||
        buffers->attention_out_bf16 == 0 || buffers->output_weight == 0 ||
        buffers->attn_gate_weight == 0 ||
        buffers->sequence_of_row == 0 || buffers->context_length == 0 ||
        buffers->positions == 0 ||
        buffers->q_a_weight == 0 || buffers->kv_a_weight == 0 ||
        buffers->kv_a_norm_weight == 0 ||
        buffers->kv_b_key_transposed_weight == 0 ||
        buffers->kv_b_value_weight == 0 ||
        buffers->q_bf16 == 0 ||
        buffers->query_latent_bf16 == 0 ||
        buffers->query_rope_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<LING_LAYER_THREADS, uint16_t>),
        rows,
        LING_LAYER_THREADS,
        (LING_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->attn_norm_weight,
        buffers->hidden_bf16,
        buffers->normed_bf16,
        LING_HIDDEN,
        LING_HIDDEN,
        LING_RMS_EPSILON);
    status = LingLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->q_a_weight,
        buffers->q_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LING_HIDDEN,
        LING_QUERY_DIM,
        LING_QUERY_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmRopePerHeadKernel<LING_LAYER_THREADS, LM_ROPE_INTERLEAVED>),
        dim3(rows, buffers->attn_heads),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        buffers->positions,
        buffers->attn_heads,
        LING_QK_HEAD_DIM,
        LING_ROPE_DIM,
        LING_ROPE_THETA);
    status = LingLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kv_a_weight,
        buffers->kv_slot_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LING_HIDDEN,
        LING_LATENT_ROW,
        LING_LATENT_ROW,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<LING_LAYER_THREADS,uint16_t>),
        rows,
        LING_LAYER_THREADS,
        (LING_LATENT + 8u) * sizeof(float),
        stream,
        buffers->kv_slot_bf16,
        0,
        (const uint16_t *)buffers->kv_a_norm_weight,
        0,
        buffers->kv_slot_bf16,
        LING_LATENT,
        LING_LATENT_ROW,
        LING_RMS_EPSILON);
    LM_LAUNCH(
        (LmRopeKernel<LING_LAYER_THREADS, LM_ROPE_INTERLEAVED>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->kv_slot_bf16,
        buffers->positions,
        LING_LATENT_ROW,
        LING_LATENT,
        LING_ROPE_DIM,
        LING_ROPE_THETA);

    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            LING_LAYER_THREADS,
            LING_QK_NOPE_DIM,
            LING_LATENT,
            LING_QK_HEAD_DIM,
            0u>),
        dim3(rows, buffers->attn_heads),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        (const uint16_t *)buffers->kv_b_key_transposed_weight,
        buffers->query_latent_bf16,
        buffers->attn_heads,
        rows);
    LM_LAUNCH(
        (LmExtractRopePerHeadKernel<LING_LAYER_THREADS, LM_ROPE_INTERLEAVED>),
        dim3(rows, buffers->attn_heads),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        buffers->query_rope_bf16,
        buffers->positions,
        buffers->attn_heads,
        LING_QK_HEAD_DIM,
        LING_QK_NOPE_DIM,
        LING_ROPE_DIM,
        LING_ROPE_THETA);
    LM_LAUNCH(
        (LmKvStoreKernel<LingKv, LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->cache,
        buffers->kv_slot_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        LING_LATENT_ROW);
    if (LmLatentAttentionDecodeSplitLaunch<
            LingKv, LING_ATTN_THREADS, LING_LATENT,
            LING_ROPE_DIM>(
            buffers->query_latent_bf16,
            buffers->query_rope_bf16,
            buffers->cache,
            buffers->sequence_of_row,
            buffers->context_length,
            0,
            0u,
            buffers->attn_heads,
            buffers->qk_scale,
            buffers->attention_latent_bf16,
            buffers->row_positions,
            rows,
            context,
            buffers->decode_split_context_threshold,
            buffers->attention_split_partials,
            (uint32_t)buffers->attention_split_partial_blocks,
            multiprocessors,
            stream) != cudaSuccess)
    {
        return LM_LAUNCH_ERR_LAUNCH;
    }

    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            LING_LAYER_THREADS,LING_LATENT,LING_VALUE_DIM>),
        dim3(rows, buffers->attn_heads),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->attention_latent_bf16,
        (const uint16_t *)buffers->kv_b_value_weight,
        buffers->attention_value_bf16,
        buffers->attn_heads,
        rows);
    status = LingLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->attn_gate_weight,
        buffers->attn_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LING_HIDDEN,
        buffers->attn_heads,
        buffers->attn_heads,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmHeadWiseGateKernel<LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->attention_value_bf16,
        buffers->attn_gate_bf16,
        buffers->attn_heads,
        LING_VALUE_DIM);

    return LingLaunchBf16Linear(
        buffers->attention_value_bf16,
        buffers->output_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->attn_output_columns,
        LING_HIDDEN,
        LING_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LingSplitFusedProjectionsKernel(
    const uint16_t *__restrict__ qkvb_bf16,
    uint16_t *__restrict__ query_bf16,
    uint16_t *__restrict__ key_bf16,
    uint16_t *__restrict__ value_bf16,
    uint16_t *__restrict__ beta_bf16,
    uint32_t rows,
    uint32_t qk_dim,
    uint32_t v_dim,
    uint32_t heads,
    uint32_t fused_rows)
{
    uint32_t row = blockIdx.x, index;
    const uint32_t k_offset = qk_dim;
    const uint32_t v_offset = 2u * qk_dim;
    const uint32_t beta_offset = v_offset + v_dim;
    uint64_t fused = (uint64_t)row * fused_rows;
    uint64_t dense = (uint64_t)row * qk_dim;
    if (row >= rows)
        return;
    for (index = threadIdx.x; index < qk_dim; index += THREADS)
        query_bf16[dense + index] = qkvb_bf16[fused + index];
    for (index = threadIdx.x; index < qk_dim; index += THREADS)
        key_bf16[dense + index] = qkvb_bf16[fused + k_offset + index];
    for (index = threadIdx.x; index < v_dim; index += THREADS)
        value_bf16[((uint64_t)row * v_dim) + index] =
            qkvb_bf16[fused + v_offset + index];
    for (index = threadIdx.x; index < heads; index += THREADS)
        beta_bf16[((uint64_t)row * heads) + index] =
            qkvb_bf16[fused + beta_offset + index];
}

static int32_t LingDeltaRuleOptIn(uint32_t shared_bytes)
{
    return(LmKernelSharedMemoryOptIn(
        (const void *)LmDeltaRuleKernel<LING_LAYER_THREADS,
                                        LING_KDA_KEY_DIM,
                                        LING_KDA_VALUE_DIM>,
        shared_bytes));
}


static int32_t LingLayerKda(
    const LingLayerBuffers *buffers,
    uint32_t rows,
    uint32_t sequences,
    uint32_t commit,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    const uint32_t rank_heads = buffers->kda_heads;
    const uint32_t rank_qk = rank_heads * LING_KDA_KEY_DIM;
    const uint32_t rank_v = rank_heads * LING_KDA_VALUE_DIM;
    int32_t status;

    if (buffers == 0 || rows == 0u || sequences == 0u ||
        buffers->kda_state_pool == 0 ||
        buffers->kda_state_slot_bytes != LING_KDA_STATE_BYTES_PER_LAYER ||
        buffers->kda_qkv_beta_weight == 0 ||
        buffers->kda_decay_weight == 0 ||
        buffers->kda_gate_weight == 0 ||
        buffers->kda_q_conv_weight == 0 || buffers->kda_k_conv_weight == 0 ||
        buffers->kda_v_conv_weight == 0 || buffers->kda_decay_bias == 0 ||
        buffers->kda_head_log_scale == 0 ||
        buffers->kda_out_norm_weight == 0 ||
        buffers->kda_out_weight == 0 ||
        buffers->fused_qkvb_bf16 == 0 ||
        buffers->kda_beta_logit == 0 ||
        buffers->kda_gate_bf16 == 0 ||
        buffers->kda_decay_logit_bf16 == 0 ||
        buffers->kda_retention == 0 || buffers->kda_write_gate == 0 ||
        buffers->normed_bf16 == 0 || buffers->attention_out_bf16 == 0 ||
        buffers->q_bf16 == 0 || buffers->kv_slot_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<LING_LAYER_THREADS,uint16_t>),
        rows,
        LING_LAYER_THREADS,
        (LING_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->attn_norm_weight,
        buffers->hidden_bf16,
        buffers->normed_bf16,
        LING_HIDDEN,
        LING_HIDDEN,
        LING_RMS_EPSILON);
    status = LingLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kda_qkv_beta_weight,
        buffers->fused_qkvb_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LING_HIDDEN,
        rank_qk * 2u + rank_v + rank_heads,
        rank_qk * 2u + rank_v + rank_heads,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = LingLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kda_decay_weight,
        buffers->kda_decay_logit_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LING_HIDDEN,
        rank_qk,
        rank_qk,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = LingLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kda_gate_weight,
        buffers->kda_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LING_HIDDEN,
        rank_v,
        rank_v,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    LM_LAUNCH(
        (LingSplitFusedProjectionsKernel<LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->fused_qkvb_bf16,
        buffers->q_bf16,
        buffers->kv_slot_bf16,
        buffers->gate_up_bf16,
        buffers->kda_beta_logit,
        rows,
        rank_qk,
        rank_v,
        rank_heads,
        rank_qk * 2u + rank_v + rank_heads);
    LM_LAUNCH(
        (LmCausalConvKernel<LING_LAYER_THREADS,LING_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_qk + LING_LAYER_THREADS - 1u) / LING_LAYER_THREADS),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->kda_q_window,
        buffers->kda_state_index,
        buffers->sequence_row_begin,
        0,
        buffers->q_bf16,
        (const uint16_t *)buffers->kda_q_conv_weight,
        buffers->q_bf16,
        rank_qk,
        sequences,
        commit);
    LM_LAUNCH(
        (LmCausalConvKernel<LING_LAYER_THREADS,LING_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_qk + LING_LAYER_THREADS - 1u) / LING_LAYER_THREADS),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->kda_k_window,
        buffers->kda_state_index,
        buffers->sequence_row_begin,
        0,
        buffers->kv_slot_bf16,
        (const uint16_t *)buffers->kda_k_conv_weight,
        buffers->kv_slot_bf16,
        rank_qk,
        sequences,
        commit);
    LM_LAUNCH(
        (LmCausalConvKernel<LING_LAYER_THREADS,LING_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_v + LING_LAYER_THREADS - 1u) / LING_LAYER_THREADS),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->kda_v_window,
        buffers->kda_state_index,
        buffers->sequence_row_begin,
        0,
        buffers->gate_up_bf16,
        (const uint16_t *)buffers->kda_v_conv_weight,
        buffers->gate_up_bf16,
        rank_v,
        sequences,
        commit);
    LM_LAUNCH(
        (LmL2NormalisePerHeadKernel<LING_LAYER_THREADS,LING_KDA_KEY_DIM>),
        dim3(rows,rank_heads),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        rank_heads,
        rows,
        LING_RMS_EPSILON);
    LM_LAUNCH(
        (LmL2NormalisePerHeadKernel<LING_LAYER_THREADS,LING_KDA_KEY_DIM>),
        dim3(rows,rank_heads),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->kv_slot_bf16,
        rank_heads,
        rows,
        LING_RMS_EPSILON);
    LM_LAUNCH(
        (LmBoundedDecayKernel<LING_LAYER_THREADS,LING_KDA_KEY_DIM>),
        dim3(rows,rank_heads),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->kda_decay_logit_bf16,
        buffers->kda_decay_bias,
        buffers->kda_head_log_scale,
        buffers->kda_retention,
        rank_heads,
        LING_KDA_GATE_LOWER_BOUND,
        rows);
    LM_LAUNCH(
        (LmSigmoidRowsKernel<LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->kda_beta_logit,
        buffers->kda_write_gate,
        rank_heads);    status = LingDeltaRuleOptIn(
        LING_KDA_KEY_DIM * LING_KDA_VALUE_DIM * sizeof(float));
    if (status != LM_LAUNCH_OK)
        return(status);
    LM_LAUNCH(
        (LmDeltaRuleKernel<LING_LAYER_THREADS,LING_KDA_KEY_DIM,LING_KDA_VALUE_DIM>),
        dim3(sequences,rank_heads),
        LING_LAYER_THREADS,
        LING_KDA_KEY_DIM * LING_KDA_VALUE_DIM * sizeof(float),
        stream,
        buffers->kda_state_pool,
        buffers->kda_state_slot_bytes,
        buffers->kda_state_index,
        buffers->sequence_row_begin,
        0,
        buffers->q_bf16,
        buffers->kv_slot_bf16,
        buffers->gate_up_bf16,
        buffers->kda_retention,
        buffers->kda_write_gate,
        buffers->attention_out_bf16,
        rank_heads,
        1u,
        sequences,
        commit);    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<LING_LAYER_THREADS,float>),
        dim3((uint64_t)rows * rank_heads),
        LING_LAYER_THREADS,
        (LING_KDA_VALUE_DIM + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        0,
        (const float *)buffers->kda_out_norm_weight,
        0,
        buffers->attention_out_bf16,
        LING_KDA_VALUE_DIM,
        LING_KDA_VALUE_DIM,
        LING_RMS_EPSILON);
    LM_LAUNCH(
        (LmOutputGateKernel<LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->kda_gate_bf16,
        rank_v);    LM_LAUNCH(
        (LmCopyRowsKernel<LING_LAYER_THREADS>),
        dim3((rank_v + LING_LAYER_THREADS - 1u) / LING_LAYER_THREADS,rows),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->kv_slot_bf16,
        rows,
        rank_v);
    status = LingLaunchBf16Linear(
        buffers->kv_slot_bf16,
        buffers->kda_out_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        rank_v,
        LING_HIDDEN,
        LING_HIDDEN,
        0u,
        multiprocessors,
        stream);    return(status);
}

static int32_t LingLayerDenseMlp(
    const LingLayerBuffers *buffers,
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
        (LmFusedResidualRmsNormKernel<LING_LAYER_THREADS, uint16_t>),
        rows,
        LING_LAYER_THREADS,
        (LING_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->hidden_bf16,
        buffers->normed_bf16,
        LING_HIDDEN,
        LING_HIDDEN,
        LING_RMS_EPSILON);

    if (buffers->dense_gate_up_fused != 0u)
    {
        status = LingLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            LING_HIDDEN,
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
        status = LingLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            LING_HIDDEN,
            buffers->dense_intermediate,
            buffers->dense_gate_up_rows,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
        status = LingLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_up_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            LING_HIDDEN,
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
        (LmSiluMulKernel<LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->dense_intermediate,
        false);

    return LingLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->dense_down_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->dense_intermediate,
        LING_HIDDEN,
        LING_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

template<uint32_t ExpertCodec>
static int32_t LingLayerMoe(
    const LingLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
    LmGemmArguments gemm;
    int32_t status;

    static_assert(ExpertFormat::kScaleGroup == 0u ||
        (LING_HIDDEN % ExpertFormat::kScaleGroup == 0u &&
        LING_EXPERT_INTERMEDIATE % ExpertFormat::kScaleGroup == 0u),
        "ling expert dimensions must contain complete codec scale groups");

    if (buffers == 0 || rows == 0u ||
        packed_rows != rows * LING_TOP_K ||
        buffers->attention_out_bf16 == 0 || buffers->residual_bf16 == 0 ||
        buffers->mlp_norm_weight == 0 || buffers->normed_bf16 == 0 ||
        buffers->router_weight == 0 || buffers->router_logits == 0 ||
        buffers->router_correction_bias == 0 ||
        buffers->route_expert == 0 || buffers->route_weight == 0 ||
        buffers->route_source_token == 0 || buffers->route_packed_row == 0 ||
        buffers->group_row_offset == 0 ||
        buffers->group_tile_prefix_w1 == 0 ||
        buffers->group_tile_prefix_w2 == 0 ||
        buffers->expert_w1_weight == 0 ||
        (ExpertCodec != SPARK_WEIGHT_CODEC_BF16 && buffers->expert_w1_scale == 0) ||
        buffers->expert_w2_weight == 0 ||
        (ExpertCodec != SPARK_WEIGHT_CODEC_BF16 && buffers->expert_w2_scale == 0) ||
        buffers->expert_out_bf16 == 0 || buffers->gate_up_bf16 == 0 ||
        buffers->intermediate_bf16 == 0 || buffers->hidden_bf16 == 0 ||
        buffers->shared_gate_up_weight == 0 ||
        buffers->shared_down_weight == 0 || buffers->shared_out_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<LING_LAYER_THREADS, uint16_t>),
        rows,
        LING_LAYER_THREADS,
        (LING_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->hidden_bf16,
        buffers->normed_bf16,
        LING_HIDDEN,
        LING_HIDDEN,
        LING_RMS_EPSILON);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = buffers->dense_row_offset;
    gemm.group_tile_prefix = buffers->dense_tile_prefix;
    gemm.output_f32 = buffers->router_logits;
    status = LmGemmLaunch<
        LmBf16Format,
        LING_LAYER_TILE_N,
        LmBf16Format::kTileK,
        LING_LAYER_STAGES,
        LING_LAYER_WARPS>(
            &gemm,
            buffers->normed_bf16,
            buffers->router_weight,
            rows,
            rows,
            1u,
            1u,
            LING_HIDDEN,
            LING_EXPERTS,
            multiprocessors,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmTopkSmallKernel<
            LING_LAYER_THREADS,
            LING_TOP_K,
            true,
            LING_ROUTER_GROUPS,
            LING_ROUTER_TOP_GROUPS,
            LM_TOPK_SCORE_SIGMOID>),
        rows,
        LING_LAYER_THREADS,
        2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),
        stream,
        buffers->router_logits,
        LING_EXPERTS,
        buffers->route_expert,
        buffers->route_weight,
        buffers->router_correction_bias,
        0,
        LING_ROUTED_SCALE);
    status = LmRouteBuild<LING_LAYER_THREADS, LING_EXPERTS>(
        buffers->route_expert,
        rows,
        packed_rows,
        LING_TOP_K,
        buffers->group_row_offset,
        buffers->route_packed_row,
        buffers->route_source_token,
        buffers->expert_w1_rows,
        LING_HIDDEN,
        LING_LAYER_TILE_N,
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
        LING_EXPERTS,
        buffers->expert_w1_rows,
        LING_HIDDEN);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
	gemm.source_row_map = buffers->route_source_token;
	gemm.source_row_count = rows;
    gemm.output_bf16 = buffers->gate_up_bf16;
    status = LmGemmWeightOnlyIndirectLaunch<
        ExpertFormat,
        LING_LAYER_TILE_N,
        LING_LAYER_STAGES,
        LING_LAYER_WARPS>(
            &gemm,
			buffers->normed_bf16,
            buffers->expert_w1_weight,
            packed_rows,
            rows,
            LING_TOP_K,
            LING_EXPERTS,
            LING_HIDDEN,
            buffers->expert_w1_rows,
            multiprocessors,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmSiluMulKernel<LING_LAYER_THREADS>),
        packed_rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->expert_intermediate,
        false);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
        buffers->expert_w2_scale,
        LING_EXPERTS,
        LING_HIDDEN,
        buffers->expert_intermediate);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
    gemm.output_bf16 = buffers->expert_out_bf16;
    status = LmGemmWeightOnlyLaunch<
        ExpertFormat,
        LING_LAYER_TILE_N,
        LING_LAYER_STAGES,
        LING_LAYER_WARPS>(
            &gemm,
            buffers->intermediate_bf16,
            buffers->expert_w2_weight,
            packed_rows,
            rows,
            LING_TOP_K,
            LING_EXPERTS,
            buffers->expert_intermediate,
            LING_HIDDEN,
            multiprocessors,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmMoeFinalizeKernel<LING_LAYER_THREADS>),
        dim3(
            (LING_HIDDEN + LING_LAYER_THREADS - 1u) /
                LING_LAYER_THREADS,
            rows),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->expert_out_bf16,
        buffers->route_packed_row,
        buffers->route_weight,
        buffers->attention_out_bf16,
        rows,
        LING_TOP_K,
        LING_HIDDEN);
    status = LingLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->shared_gate_up_weight,
        buffers->gate_up_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LING_HIDDEN,
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
        (LmSiluMulKernel<LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->shared_intermediate,
        false);
    status = LingLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->shared_down_weight,
        buffers->shared_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->shared_intermediate,
        LING_HIDDEN,
        LING_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmAddRowsKernel<LING_LAYER_THREADS>),
        dim3(
            (LING_HIDDEN + LING_LAYER_THREADS - 1u) /
                LING_LAYER_THREADS,
            rows),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->shared_out_bf16,
        buffers->attention_out_bf16,
        rows,
        LING_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t LingHead(
    const LingLayerBuffers *buffers,
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

    tiles = (vocabulary + LING_HEAD_TILE - 1u) / LING_HEAD_TILE;
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<LING_LAYER_THREADS, uint16_t>),
        rows,
        LING_LAYER_THREADS,
        (LING_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hidden_bf16,
        buffers->residual_bf16,
        (const uint16_t *)head_norm_weight,
        0,
        buffers->normed_bf16,
        LING_HIDDEN,
        LING_HIDDEN,
        LING_RMS_EPSILON);
    LM_LAUNCH(
        (LmHeadCandidateKernel<LING_LAYER_THREADS, LING_HEAD_TILE>),
        dim3(tiles, rows),
        LING_LAYER_THREADS,
        0,
        stream,
        buffers->normed_bf16,
        (const uint16_t *)head_weight,
        token_ids,
        buffers->head_candidate_score,
        buffers->head_candidate_token,
        rows,
        LING_HIDDEN,
        vocabulary);
    LM_LAUNCH(
        (LmHeadCommitKernel<LING_LAYER_THREADS>),
        rows,
        LING_LAYER_THREADS,
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
