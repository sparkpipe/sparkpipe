#pragma once

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "sparkpipe/spark_laguna_resident_decode_stage_firmware.h"
#include "modules/laguna_resident_decode_stage/source/cuda/config.h"
#include "modules/laguna_resident_decode_stage/source/cuda/index_kv.cuh"

struct LagunaKv
{
    static constexpr uint32_t kSlotBytes = LAGUNA_KV_SLOT_BYTES;
    static constexpr uint32_t kPageSlots = LAGUNA_KV_PAGE_SLOTS;
    static constexpr uint32_t kPageBytes =
        LAGUNA_KV_SLOT_BYTES * LAGUNA_KV_PAGE_SLOTS;
    static constexpr bool kGrows = true;
    static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
    { return position / LAGUNA_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
    { return position % LAGUNA_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
    { return (tokens + LAGUNA_KV_PAGE_SLOTS - 1u) / LAGUNA_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
    { return pages * (uint64_t)kPageBytes; }
};

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LagunaIndexPackKernel(
    const uint16_t *__restrict__ key_bf16,
    const uint16_t *__restrict__ gate_bf16,
    uint16_t *__restrict__ packed_bf16,
    uint32_t dimension)
{
    uint32_t row = blockIdx.x;
    uint64_t base = (uint64_t)row * (2u * dimension + 1u);
    uint32_t index;
    for (index = threadIdx.x; index < dimension; index += THREADS)
    {
        packed_bf16[base + index] = key_bf16[(uint64_t)row * dimension + index];
        packed_bf16[base + dimension + index] =
            gate_bf16[(uint64_t)row * dimension + index];
    }
    if (threadIdx.x == 0u)
        packed_bf16[base + 2u * dimension] = 0x3F80u;
}

template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS>
__global__ __launch_bounds__(THREADS, 1)
void LagunaPoolScoreKernel(
    const uint16_t *__restrict__ query_bf16,
    const uint16_t *__restrict__ head_weight_bf16,
    LmKvView index_cache,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    const float *__restrict__ compress_ape_f32,
    uint32_t pools,
    float softmax_scale,
    float head_scale,
    float *__restrict__ pool_scores)
{
    __shared__ float pool_key[DIM];
    __shared__ float reduction[THREADS / LM_WARP_LANES];
    uint32_t pool = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t sequence = sequence_of_row[row];
    uint32_t context = context_length[sequence];
    uint32_t first = pool * KPOOL;
    uint32_t index, slot_in_pool, head;
    if (pool >= pools)
        return;

    for (index = threadIdx.x; index < DIM; index += THREADS)
    {
        float maximum = -INFINITY;
        float sum = 0.0f;
        float logits[KPOOL];
        for (slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool)
        {
            uint32_t position = first + slot_in_pool;
            if (position >= context)
            {
                logits[slot_in_pool] = -INFINITY;
                continue;
            }
            const uint16_t *slot = (const uint16_t *)LmKvSlotRequired<LagunaIndexKv>(
                index_cache, sequence, position, row, LM_KV_ACCESS_READ);
            if (slot == 0)
                return;
            logits[slot_in_pool] =
                LmBf16ToFloat(slot[DIM + index]) +
                compress_ape_f32[slot_in_pool * DIM + index];
            maximum = fmaxf(maximum, logits[slot_in_pool]);
        }
        for (slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool)
            if (logits[slot_in_pool] != -INFINITY)
                sum += __expf(logits[slot_in_pool] - maximum);
        float mix[KPOOL];
        for (slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool)
            mix[slot_in_pool] = logits[slot_in_pool] == -INFINITY
                ? 0.0f
                : __expf(logits[slot_in_pool] - maximum) / fmaxf(sum, 1.0e-20f);
        float key = 0.0f;
        for (slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool)
        {
            if (mix[slot_in_pool] == 0.0f)
                continue;
            uint32_t position = first + slot_in_pool;
            const uint16_t *slot = (const uint16_t *)LmKvSlotRequired<LagunaIndexKv>(
                index_cache, sequence, position, row, LM_KV_ACCESS_READ);
            if (slot == 0)
                return;
            key += mix[slot_in_pool] * LmBf16ToFloat(slot[index]);
        }
        pool_key[index] = key;
    }
    __syncthreads();

    float total = 0.0f;
    for (head = 0u; head < HEADS; ++head)
    {
        float score = 0.0f;
        for (index = threadIdx.x; index < DIM; index += THREADS)
            score += LmBf16ToFloat(
                query_bf16[((uint64_t)row * HEADS + head) * DIM + index]) *
                pool_key[index];
        score = LmBlockSum<THREADS>(score, reduction) * softmax_scale;
        __syncthreads();
        if (score < 0.0f)
            score = 0.0f;
        if (threadIdx.x == 0u)
            reduction[0] = score * LmBf16ToFloat(
                head_weight_bf16[(uint64_t)row * HEADS + head]) * head_scale;
        __syncthreads();
        total += reduction[0];
    }
    if (threadIdx.x == 0u)
        pool_scores[(uint64_t)row * pools + pool] = total;
}

template<uint32_t THREADS, uint32_t KPOOL, uint32_t TOPK, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 1)
void LagunaPoolExpandKernel(
    const uint32_t *__restrict__ selected_pools,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    uint32_t *__restrict__ selected_positions,
    uint32_t rows)
{
    uint32_t row = blockIdx.x;
    uint32_t index;
    if (row >= rows)
        return;
    uint32_t sequence = sequence_of_row[row];
    uint32_t context = context_length[sequence];
    uint32_t select = TOPK / KPOOL;
    for (index = threadIdx.x; index < WIDTH; index += THREADS)
    {
        uint32_t position = 0xFFFFFFFFu;
        if (index < TOPK)
        {
            uint32_t pool = selected_pools[(uint64_t)row * select +
                                           index / KPOOL];
            uint32_t within = index % KPOOL;
            position = pool * KPOOL + within;
            if (position >= context)
                position = 0xFFFFFFFFu;
        }
        else
        {
            uint32_t tail_count = context % KPOOL;
            uint32_t tail_index = index - TOPK;
            if (tail_index < tail_count)
                position = context - tail_count + tail_index;
        }
        selected_positions[(uint64_t)row * WIDTH + index] = position;
    }
}

#include "modules/laguna_resident_decode_stage/source/cuda/launch_shape.h"

#define LAGUNA_LAYER_TILE_N 128u
#define LAGUNA_LAYER_STAGES 2u
#define LAGUNA_LAYER_WARPS 8u
#define LAGUNA_HEAD_TILE 1024u

static_assert(
    LAGUNA_HIDDEN % LmBf16Format::kTileK == 0u,
    "laguna hidden projections must cover every BF16 K tile");
static_assert(
    LAGUNA_QUERY_A_DIM % LmBf16Format::kTileK == 0u,
    "laguna low-rank query projections must cover every BF16 K tile");
static_assert(
    LAGUNA_DSA_QUERY_DIM % LmBf16Format::kTileK == 0u,
    "laguna DSA index queries must cover every BF16 K tile");
static_assert(
    (LAGUNA_ATTN_HEADS * LAGUNA_LATENT) % LmBf16Format::kTileK == 0u,
    "laguna latent attention output must cover every BF16 K tile");
static_assert(
    LAGUNA_DENSE_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "laguna dense FFN down projection must cover every BF16 K tile");
static_assert(
    LAGUNA_EXPERT_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "laguna expert down projection must cover every BF16 K tile");
static_assert(
    LAGUNA_KDA_QK_DIM % LmBf16Format::kTileK == 0u,
    "laguna fused KDA q|k|v|beta rows must cover every BF16 K tile");
struct LagunaLayerBuffers
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

    const void *kda_qkv_beta_weight;
    const void *kda_decay_gate_down_weight;
    const void *kda_decay_up_weight;
    const void *kda_gate_up_weight;
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
    uint16_t *fused_decay_gate_bf16;
    uint16_t *kda_output_bf16;
    uint16_t *kda_decay_latent_bf16;
    uint16_t *kda_beta_logit;
    uint16_t *kda_gate_bf16;
    uint16_t *kda_gate_latent_bf16;
    uint16_t *kda_decay_logit_bf16;
    float *kda_retention;
    float *kda_write_gate;
    uint8_t *kda_replay_layer;
    uint32_t kda_replay_steps;

    const void *hc_attn_fn;
    const void *hc_attn_base;
    const void *hc_attn_scale;
    const void *hc_ffn_fn;
    const void *hc_ffn_base;
    const void *hc_ffn_scale;
    float *hc_mixes_f32;
    float *hc_pre_f32;
    float *hc_post_f32;
    float *hc_comb_f32;
    uint16_t *hc_collapsed_bf16;
    uint16_t *hc_snapshot_bf16;
    uint16_t *hc_mean_bf16;

    const void *index_compress_ape;
    const void *index_compress_gate;
    uint16_t *index_gate_bf16;
    uint16_t *index_packed_bf16;
    uint32_t *selected_pools;

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
};

static_assert(
    LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS ==
        SPARK_LAGUNA_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTITIONS,
    "the firmware's split-partials sizing must match the kernel's "
    "partition cap");

static int32_t LagunaLaunchBf16Linear(
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
        LAGUNA_LAYER_TILE_N,
        LmBf16Format::kTileK,
        LAGUNA_LAYER_STAGES,
        LAGUNA_LAYER_WARPS>(
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

static int32_t LagunaLayerIndexer(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;
    uint32_t pools;

    if (buffers == 0 || rows == 0u || context == 0u ||
        buffers->positions == 0 || buffers->sequence_of_row == 0 ||
        buffers->context_length == 0 ||
        !LmKvViewIsConfigured(buffers->index_cache) ||
        buffers->index_q_weight == 0 || buffers->index_k_weight == 0 ||
        buffers->index_head_weight == 0 || buffers->index_norm_weight == 0 ||
        buffers->index_norm_bias == 0 || buffers->index_query_bf16 == 0 ||
        buffers->index_key_bf16 == 0 ||
        buffers->index_head_weight_bf16 == 0 ||
        buffers->index_compress_gate == 0 ||
        buffers->index_compress_ape == 0 ||
        buffers->index_gate_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    status = LagunaLaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->index_q_weight,
        buffers->index_query_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_QUERY_A_DIM,
        LAGUNA_DSA_QUERY_DIM,
        LAGUNA_DSA_QUERY_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_k_weight,
        buffers->index_key_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
        LAGUNA_DSA_INDEX_DIM,
        LAGUNA_DSA_INDEX_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_compress_gate,
        buffers->index_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
        LAGUNA_DSA_INDEX_DIM,
        LAGUNA_DSA_INDEX_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmLayerNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_DSA_INDEX_DIM + 8u) * sizeof(float),
        stream,
        buffers->index_key_bf16,
        (const uint16_t *)buffers->index_norm_weight,
        (const uint16_t *)buffers->index_norm_bias,
        buffers->index_key_bf16,
        LAGUNA_DSA_INDEX_DIM,
        LAGUNA_DSA_INDEX_DIM,
        LAGUNA_DSA_INDEX_EPSILON);
    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_head_weight,
        buffers->index_head_weight_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
        LAGUNA_DSA_INDEX_HEADS,
        LAGUNA_DSA_INDEX_HEADS,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LagunaIndexPackKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->index_key_bf16,
        buffers->index_gate_bf16,
        buffers->index_packed_bf16,
        LAGUNA_DSA_INDEX_DIM);
    LM_LAUNCH(
        (LmKvStoreKernel<LagunaIndexKv,LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->index_cache,
        buffers->index_packed_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        SPARK_LAGUNA_MODEL_INDEX_PACKED_TOKEN_DIMENSION);
    if (context <= LAGUNA_DSA_SELECTED)
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
    pools = context / LAGUNA_DSA_KPOOL;
    LM_LAUNCH(
        (LagunaPoolScoreKernel<
            LAGUNA_LAYER_THREADS,
            LAGUNA_DSA_INDEX_DIM,
            LAGUNA_DSA_KPOOL,
            LAGUNA_DSA_INDEX_HEADS>),
        dim3(pools,rows),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->index_query_bf16,
        buffers->index_head_weight_bf16,
        buffers->index_cache,
        buffers->sequence_of_row,
        buffers->context_length,
        (const float *)buffers->index_compress_ape,
        pools,
        LAGUNA_DSA_INDEX_SCALE,
        LAGUNA_DSA_INDEX_HEAD_WEIGHT_SCALE,
        buffers->selection_scores);
    LM_LAUNCH(
        (LmTopkHistogramKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->selection_scores,
        pools,
        LAGUNA_DSA_SELECTED / LAGUNA_DSA_KPOOL,
        buffers->head_candidate_token);
    LM_LAUNCH(
        (LmTopkGatherKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->selection_scores,
        pools,
        LAGUNA_DSA_SELECTED / LAGUNA_DSA_KPOOL,
        buffers->head_candidate_token,
        buffers->selected_pools,
        0);
    LM_LAUNCH(
        (LagunaPoolExpandKernel<
            LAGUNA_LAYER_THREADS,
            LAGUNA_DSA_KPOOL,
            LAGUNA_DSA_SELECTED,
            SPARK_LAGUNA_MODEL_INDEX_OUTPUT_WIDTH>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->selected_pools,
        buffers->sequence_of_row,
        buffers->context_length,
        buffers->selected_positions,
        rows);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}


static int LagunaDsaProbeVecLayer(void)
{
    static int vec_layer = -1;
    if ( vec_layer < 0 )
    {
        const char *layer_env = getenv("SPARK_LAGUNA_PROBE_VEC_LAYER");
        vec_layer = (layer_env != 0 && *layer_env != 0)
            ? (int)atoi(layer_env) : 3;
    }
    return(vec_layer);
}

static int LagunaDsaProbeVecPass(const LagunaLayerBuffers *buffers)
{
    static int vec_enabled = -1;
    static uint32_t vec_pass = 0u;
    uint32_t cap;
    if ( vec_enabled < 0 )
        vec_enabled = (getenv("SPARK_LAGUNA_PROBE_VEC") != 0 &&
            getenv("SPARK_LAGUNA_PROBE_VEC_DSA") != 0) ? 1 : 0;
    if ( vec_enabled == 0 || buffers == 0 || buffers->tp_rank != 0u ||
        (int)buffers->layer_index != LagunaDsaProbeVecLayer() )
        return(0);
    cap = 30u;
    {
        const char *cap_env = getenv("SPARK_LAGUNA_PROBE_VEC_PASSES");
        if ( cap_env != 0 && *cap_env != 0 )
            cap = (uint32_t)atoi(cap_env);
    }
    if ( vec_pass >= cap )
        return(0);
    vec_pass += 1u;
    return((int)vec_pass);
}

static void LagunaProbeVecU16(cudaStream_t stream,const uint16_t *device,uint32_t count,uint32_t layer,uint32_t pass,const char *label)
{
    static uint16_t vec_buf[16384];
    uint32_t i;
    if ( count > 16384u || cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaMemcpy(vec_buf,device,count * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
        return;
    fprintf(stderr,"G5N-VEC L%u P%u %s %u",layer,pass,label,count);
    for ( i = 0u; i < count; i++ )
        fprintf(stderr," %04x",vec_buf[i]);
    fputc('\n',stderr);
}

static void LagunaProbeVecF32(cudaStream_t stream,const float *device,uint32_t count,uint32_t layer,uint32_t pass,const char *label)
{
    static uint32_t vec_buf[16384];
    uint32_t i;
    if ( count > 16384u || cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaMemcpy(vec_buf,device,count * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess )
        return;
    fprintf(stderr,"G5N-VEC L%u P%u %s %u",layer,pass,label,count);
    for ( i = 0u; i < count; i++ )
        fprintf(stderr," %08x",vec_buf[i]);
    fputc('\n',stderr);
}

static int32_t LagunaLayerAttention(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    const uint32_t *selected_positions;
    uint32_t selected_position_count;
    int32_t status;
    const int32_t vec_pass = (int32_t)LagunaDsaProbeVecPass(buffers);
    uint32_t vec_rank_heads = 0u;

    if (buffers == 0 || rows == 0u || context == 0u ||
        buffers->qk_scale <= 0.0f || buffers->hidden_bf16 == 0 ||
        buffers->normed_bf16 == 0 ||
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
        (context > LAGUNA_DSA_SELECTED &&
         (buffers->selected_positions == 0 ||
          buffers->selected_position_count != LAGUNA_DSA_SELECTED)))
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    selected_positions = context > LAGUNA_DSA_SELECTED
        ? buffers->selected_positions : 0;
    selected_position_count = context > LAGUNA_DSA_SELECTED
        ? buffers->selected_position_count : 0u;

    if ( vec_pass != 0 )
    {
        vec_rank_heads = buffers->attn_heads;
        LagunaProbeVecU16(stream,buffers->hidden_bf16,
            LAGUNA_HC * LAGUNA_HIDDEN,layer_index,(uint32_t)vec_pass,"hc_streams");
        LagunaProbeVecU16(stream,buffers->hc_collapsed_bf16,
            LAGUNA_HIDDEN,layer_index,(uint32_t)vec_pass,"hc_collapsed");
        LagunaProbeVecF32(stream,buffers->hc_mixes_f32,
            (2u + LAGUNA_HC) * LAGUNA_HC,layer_index,(uint32_t)vec_pass,"hc_mixes");
        LagunaProbeVecF32(stream,buffers->hc_pre_f32,
            LAGUNA_HC,layer_index,(uint32_t)vec_pass,"hc_pre");
        LagunaProbeVecF32(stream,buffers->hc_post_f32,
            LAGUNA_HC,layer_index,(uint32_t)vec_pass,"hc_post");
        LagunaProbeVecF32(stream,buffers->hc_comb_f32,
            LAGUNA_HC * LAGUNA_HC,layer_index,(uint32_t)vec_pass,"hc_comb");
    }

    LM_LAUNCH(
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        (const uint16_t *)buffers->attn_norm_weight,
        buffers->normed_bf16,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        LAGUNA_RMS_EPSILON);
    if ( vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->normed_bf16,LAGUNA_HIDDEN,
            layer_index,(uint32_t)vec_pass,"attn_normed");

    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->q_a_weight,
        buffers->q_compressed_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
        LAGUNA_QUERY_A_DIM,
        LAGUNA_QUERY_A_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_QUERY_A_DIM + 8u) * sizeof(float),
        stream,
        buffers->q_compressed_bf16,
        (const uint16_t *)buffers->q_a_norm_weight,
        buffers->q_compressed_bf16,
        LAGUNA_QUERY_A_DIM,
        LAGUNA_QUERY_A_DIM,
        LAGUNA_RMS_EPSILON);
    if ( vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->q_compressed_bf16,
            LAGUNA_QUERY_A_DIM,layer_index,(uint32_t)vec_pass,"q_compressed");
    status = LagunaLayerIndexer(
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
    status = LagunaLaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->q_b_weight,
        buffers->q_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_QUERY_A_DIM,
        buffers->q_b_rows,
        buffers->q_b_rows,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    if ( vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->q_bf16,
            vec_rank_heads * LAGUNA_QK_NOPE_DIM,layer_index,(uint32_t)vec_pass,"q");
    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kv_a_weight,
        buffers->kv_slot_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
        LAGUNA_LATENT_ROW,
        LAGUNA_LATENT_ROW,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_LATENT + 8u) * sizeof(float),
        stream,
        buffers->kv_slot_bf16,
        (const uint16_t *)buffers->kv_a_norm_weight,
        buffers->kv_slot_bf16,
        LAGUNA_LATENT,
        LAGUNA_LATENT_ROW,
        LAGUNA_RMS_EPSILON);
    if ( vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->kv_slot_bf16,LAGUNA_LATENT_ROW,
            layer_index,(uint32_t)vec_pass,"kv_slot");

    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            LAGUNA_LAYER_THREADS,
            LAGUNA_QK_NOPE_DIM,
            LAGUNA_LATENT,
            LAGUNA_QK_NOPE_DIM + LAGUNA_ROPE_DIM,
            0u>),
        dim3(rows, buffers->attn_heads),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        (const uint16_t *)buffers->kv_b_key_transposed_weight,
        buffers->query_latent_bf16,
        buffers->attn_heads,
        rows);
    if ( vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->query_latent_bf16,
            vec_rank_heads * LAGUNA_LATENT,layer_index,(uint32_t)vec_pass,"query_latent");
    LM_LAUNCH(
        (LmKvStoreKernel<LagunaKv, LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->cache,
        buffers->kv_slot_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        LAGUNA_LATENT_ROW);
    if (LmLatentAttentionDecodeSplitLaunch<
            LagunaKv, LAGUNA_ATTN_THREADS, LAGUNA_LATENT,
            LAGUNA_ROPE_DIM>(
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
            context > LAGUNA_DSA_SELECTED ? LAGUNA_DSA_SELECTED
                                             : context,
            buffers->decode_split_context_threshold,
            buffers->attention_split_partials,
            (uint32_t)buffers->attention_split_partial_blocks,
            multiprocessors,
            stream) != cudaSuccess)
    {
        return LM_LAUNCH_ERR_LAUNCH;
    }
    if ( vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->attention_latent_bf16,
            vec_rank_heads * LAGUNA_LATENT,layer_index,(uint32_t)vec_pass,"attn_latent");

    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            LAGUNA_LAYER_THREADS,LAGUNA_LATENT,LAGUNA_VALUE_DIM>),
        dim3(rows, buffers->attn_heads),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->attention_latent_bf16,
        (const uint16_t *)buffers->kv_b_value_weight,
        buffers->attention_value_bf16,
        buffers->attn_heads,
        rows);
    if ( vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->attention_value_bf16,
            vec_rank_heads * LAGUNA_VALUE_DIM,layer_index,(uint32_t)vec_pass,"attn_value");

    status = LagunaLaunchBf16Linear(
        buffers->attention_value_bf16,
        buffers->output_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->attn_output_columns,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if ( status == LM_LAUNCH_OK && vec_pass != 0 )
        LagunaProbeVecU16(stream,buffers->attention_out_bf16,LAGUNA_HIDDEN,
            layer_index,(uint32_t)vec_pass,"attn_out_partial");
    return status;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LagunaSplitFusedProjectionsKernel(
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

template<uint32_t THREADS, uint32_t LOW_RANK>
__global__ __launch_bounds__(THREADS, 1)
void LagunaSplitDecayGateDownKernel(
    const uint16_t *__restrict__ fused_bf16,
    uint16_t *__restrict__ decay_latent_bf16,
    uint16_t *__restrict__ gate_latent_bf16,
    uint32_t rows)
{
    uint32_t row = blockIdx.x, index;
    if (row >= rows)
        return;
    for (index = threadIdx.x; index < LOW_RANK; index += THREADS)
    {
        decay_latent_bf16[((uint64_t)row * LOW_RANK) + index] =
            fused_bf16[((uint64_t)row * 2u * LOW_RANK) + index];
        gate_latent_bf16[((uint64_t)row * LOW_RANK) + index] =
            fused_bf16[((uint64_t)row * 2u * LOW_RANK) + LOW_RANK + index];
    }
}

static int32_t LagunaDeltaRuleOptIn(uint32_t shared_bytes)
{
    return(LmKernelSharedMemoryOptIn(
        (const void *)LmDeltaRuleKernel<LAGUNA_LAYER_THREADS,
                                        LAGUNA_KDA_KEY_DIM,
                                        LAGUNA_KDA_VALUE_DIM>,
        shared_bytes));
}


#include <stdlib.h>
#include <stdio.h>

static int LagunaKdaProbeActive(const LagunaLayerBuffers *buffers)
{
    static int probe_enabled = -1;
    if ( probe_enabled < 0 )
        probe_enabled = getenv("SPARK_LAGUNA_PROBE") != 0 ? 1 : 0;
    return(probe_enabled != 0 && buffers != 0 && buffers->tp_rank == 0u);
}

static int LagunaKdaProbeDeep(const LagunaLayerBuffers *buffers)
{
    uint32_t layer = buffers != 0 ? buffers->layer_index : 0u;
    return(LagunaKdaProbeActive(buffers) &&
        (layer == 0u || (layer >= 16u && layer <= 20u) || layer >= 43u));
}

#define LAGUNA_KDA_PROBE_RAW(stream,lyr,label,dev) \
    do { \
        uint16_t probe_h[256]; float probe_f[8]; uint32_t probe_i; \
        if ( cudaStreamSynchronize((stream)) == cudaSuccess && \
             cudaMemcpy(probe_h,(dev),256 * sizeof(uint16_t),cudaMemcpyDeviceToHost) == cudaSuccess ) \
        { \
            for ( probe_i = 0u; probe_i < 8u; probe_i++ ) \
            { \
                uint32_t probe_bits = ((uint32_t)probe_h[probe_i]) << 16; \
                (void)memcpy(&probe_f[probe_i],&probe_bits,sizeof(float)); \
            } \
            fprintf(stderr,"G5N-PROBE kda L%u %s raw %u %u %u %u %u %u %u %u f %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n", \
                (unsigned)(lyr),(label),probe_h[0],probe_h[1],probe_h[2],probe_h[3], \
                probe_h[4],probe_h[5],probe_h[6],probe_h[7], \
                (double)probe_f[0],(double)probe_f[1],(double)probe_f[2],(double)probe_f[3], \
                (double)probe_f[4],(double)probe_f[5],(double)probe_f[6],(double)probe_f[7]); \
        } \
    } while (0)

static uint64_t LagunaProbeBf16Sum(cudaStream_t stream,const uint16_t *device,uint32_t count)
{
    uint16_t host[256];
    uint64_t total;
    uint32_t i,taken;
    total = 0u;
    if ( cudaStreamSynchronize(stream) != cudaSuccess )
        return(0xDEADDEADu);
    while ( count != 0u )
    {
        taken = count < 256u ? count : 256u;
        if ( cudaMemcpy(host,device,taken * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
            return(0xDEADDEADu);
        for ( i = 0u; i < taken; i++ )
            total += host[i];
        count -= taken;
        device += taken;
    }
    return(total);
}

static void LagunaProbeFloats(cudaStream_t stream,const float *device,uint32_t count,float *host)
{
    if ( cudaStreamSynchronize(stream) != cudaSuccess )
        return;
    (void)cudaMemcpy(host,device,count * sizeof(float),cudaMemcpyDeviceToHost);
}

static void LagunaProbeBf16Floats(cudaStream_t stream,const uint16_t *device,uint32_t count,float *host)
{
    uint16_t probe_b[8];
    uint32_t probe_i;
    if ( count > 8u || cudaStreamSynchronize(stream) != cudaSuccess ||
         cudaMemcpy(probe_b,device,count * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
        return;
    for ( probe_i = 0u; probe_i < count; probe_i++ )
    {
        uint32_t probe_bits = ((uint32_t)probe_b[probe_i]) << 16;
        (void)memcpy(&host[probe_i],&probe_bits,sizeof(float));
    }
}

#define LAGUNA_KDA_PROBE(stream,label,dev,cnt) \
    do { fprintf(stderr,"G5N-PROBE kda L%u %s bf16sum %llu\n", \
        buffers->layer_index,(label), \
        (unsigned long long)LagunaProbeBf16Sum((stream),(const uint16_t *)(dev),(cnt))); } while (0)

#define LAGUNA_KDA_PROBE_STATE(stream,label,pool) \
    do { \
        float probe_s[4]; uint32_t probe_w; uint64_t probe_bits = 0u; \
        uint32_t probe_words[1024]; \
        LagunaProbeFloats((stream),(const float *)(pool),4u,probe_s); \
        if ( cudaStreamSynchronize((stream)) == cudaSuccess && \
             cudaMemcpy(probe_words,(pool),sizeof(probe_words),cudaMemcpyDeviceToHost) == cudaSuccess ) \
            for ( probe_w = 0u; probe_w < 1024u; probe_w++ ) \
                probe_bits += probe_words[probe_w]; \
        else \
            probe_bits = 0xDEADDEADu; \
        fprintf(stderr,"G5N-PROBE kda L%u %s f %.6g %.6g %.6g %.6g bits4096B %llu\n", \
            buffers->layer_index,(label),(double)probe_s[0],(double)probe_s[1], \
            (double)probe_s[2],(double)probe_s[3],(unsigned long long)probe_bits); \
    } while (0)


static int LagunaKdaProbeVecLayer(const LagunaLayerBuffers *buffers)
{
    static int vec_layer = -1;
    if ( vec_layer < 0 )
    {
        const char *layer_env = getenv("SPARK_LAGUNA_PROBE_VEC_KDA_LAYER");
        vec_layer = (layer_env != 0 && *layer_env != 0)
            ? (int)atoi(layer_env) : 0;
    }
    return(vec_layer);
}

static int32_t LagunaLayerProbeVecPass(const LagunaLayerBuffers *buffers,uint32_t *vec_pass,uint32_t calls_per_layer)
{
    static int vec_enabled = -1;
    uint32_t cap;
    if ( vec_enabled < 0 )
        vec_enabled = getenv("SPARK_LAGUNA_PROBE_VEC") != 0 ? 1 : 0;
    if ( vec_enabled == 0 || buffers == 0 || buffers->tp_rank != 0u ||
        (int)buffers->layer_index != LagunaKdaProbeVecLayer(buffers) )
        return(0);
    cap = 30u;
    {
        const char *cap_env = getenv("SPARK_LAGUNA_PROBE_VEC_PASSES");
        if ( cap_env != 0 && *cap_env != 0 )
            cap = (uint32_t)atoi(cap_env);
    }
    if ( *vec_pass / calls_per_layer >= cap )
        return(0);
    *vec_pass += 1u;
    return((int32_t)*vec_pass);
}

static int32_t LagunaKdaReplayRecord(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t rank_heads,
    uint32_t committed_inputs,
    cudaStream_t stream)
{
    SparkLagunaKdaReplayLayout layout;
    uint32_t rank_qk = rank_heads * LAGUNA_KDA_KEY_DIM;
    uint32_t rank_v = rank_heads * LAGUNA_KDA_VALUE_DIM;
    cudaError_t error;
    if (buffers->kda_replay_layer == 0)
        return LM_LAUNCH_OK;
    if (rows == 0u || rows > buffers->kda_replay_steps)
        return LM_LAUNCH_ERR_SHAPE;
    layout = SparkLagunaKdaReplayLayoutFor(rank_heads, buffers->kda_replay_steps);
    if (committed_inputs == 0u)
    {
        error = cudaMemcpyAsync(buffers->kda_replay_layer + layout.pre_q_offset, buffers->q_bf16, (uint64_t)rows * rank_qk * 2u, cudaMemcpyDeviceToDevice, stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(buffers->kda_replay_layer + layout.pre_k_offset, buffers->kv_slot_bf16, (uint64_t)rows * rank_qk * 2u, cudaMemcpyDeviceToDevice, stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(buffers->kda_replay_layer + layout.pre_v_offset, buffers->gate_up_bf16, (uint64_t)rows * rank_v * 2u, cudaMemcpyDeviceToDevice, stream);
    }
    else
    {
        error = cudaMemcpyAsync(buffers->kda_replay_layer + layout.key_offset, buffers->kv_slot_bf16, (uint64_t)rows * rank_qk * 2u, cudaMemcpyDeviceToDevice, stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(buffers->kda_replay_layer + layout.value_offset, buffers->gate_up_bf16, (uint64_t)rows * rank_v * 2u, cudaMemcpyDeviceToDevice, stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(buffers->kda_replay_layer + layout.retention_offset, buffers->kda_retention, (uint64_t)rows * rank_qk * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(buffers->kda_replay_layer + layout.write_gate_offset, buffers->kda_write_gate, (uint64_t)rows * rank_heads * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    return error == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t LagunaLayerKda(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t sequences,
    uint32_t commit,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    const uint32_t rank_heads = buffers->kda_heads;
    const uint32_t rank_qk = rank_heads * LAGUNA_KDA_KEY_DIM;
    const uint32_t rank_v = rank_heads * LAGUNA_KDA_VALUE_DIM;
    static uint32_t probe_count = 0u;
    const int32_t vec_pass = LagunaLayerProbeVecPass(buffers,&probe_count,1u);
    int32_t status;

    if (buffers == 0 || rows == 0u || sequences == 0u ||
        buffers->kda_state_pool == 0 ||
        buffers->kda_state_slot_bytes != rank_heads * LAGUNA_KDA_KEY_DIM * LAGUNA_KDA_VALUE_DIM * sizeof(float) ||
        buffers->kda_qkv_beta_weight == 0 ||
        buffers->kda_decay_gate_down_weight == 0 ||
        buffers->kda_decay_up_weight == 0 ||
        buffers->kda_gate_up_weight == 0 ||
        buffers->kda_q_conv_weight == 0 || buffers->kda_k_conv_weight == 0 ||
        buffers->kda_v_conv_weight == 0 || buffers->kda_decay_bias == 0 ||
        buffers->kda_head_log_scale == 0 ||
        buffers->kda_out_norm_weight == 0 ||
        buffers->kda_out_weight == 0 ||
        buffers->fused_qkvb_bf16 == 0 ||
        buffers->fused_decay_gate_bf16 == 0 ||
        buffers->kda_decay_latent_bf16 == 0 ||
        buffers->kda_beta_logit == 0 ||
        buffers->kda_gate_bf16 == 0 ||
        buffers->kda_gate_latent_bf16 == 0 ||
        buffers->kda_decay_logit_bf16 == 0 ||
        buffers->kda_retention == 0 || buffers->kda_write_gate == 0 ||
        buffers->normed_bf16 == 0 || buffers->attention_out_bf16 == 0 ||
        buffers->q_bf16 == 0 || buffers->kv_slot_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    LM_LAUNCH(
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        (const uint16_t *)buffers->attn_norm_weight,
        buffers->normed_bf16,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        LAGUNA_RMS_EPSILON);

    if ( LagunaKdaProbeActive(buffers) )
    {
        LAGUNA_KDA_PROBE(stream,"collapsed",buffers->hc_collapsed_bf16,256u);
        LAGUNA_KDA_PROBE(stream,"normed",buffers->normed_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->hc_collapsed_bf16,LAGUNA_HIDDEN,buffers->layer_index,(uint32_t)vec_pass,"collapsed");
        LagunaProbeVecU16(stream,buffers->normed_bf16,LAGUNA_HIDDEN,buffers->layer_index,(uint32_t)vec_pass,"normed");
        LagunaProbeVecU16(stream,(const uint16_t *)buffers->attn_norm_weight,256u,buffers->layer_index,(uint32_t)vec_pass,"w_attn_norm");
        LagunaProbeVecU16(stream,(const uint16_t *)buffers->kda_qkv_beta_weight,256u,buffers->layer_index,(uint32_t)vec_pass,"w_qkvb_row0");
        LagunaProbeVecU16(stream,(const uint16_t *)buffers->kda_out_weight,256u,buffers->layer_index,(uint32_t)vec_pass,"w_kda_out_row0");
        LagunaProbeVecF32(stream,buffers->hc_pre_f32,LAGUNA_HC,buffers->layer_index,(uint32_t)vec_pass,"k_pre");
        LagunaProbeVecF32(stream,buffers->hc_post_f32,LAGUNA_HC,buffers->layer_index,(uint32_t)vec_pass,"k_post");
        LagunaProbeVecF32(stream,buffers->hc_mixes_f32,(2u + LAGUNA_HC) * LAGUNA_HC,buffers->layer_index,(uint32_t)vec_pass,"k_mixes");
        LagunaProbeVecU16(stream,(const uint16_t *)buffers->hidden_bf16,256u,buffers->layer_index,(uint32_t)vec_pass,"k_streams_head");
        LagunaProbeVecU16(stream,(const uint16_t *)buffers->hc_snapshot_bf16,256u,buffers->layer_index,(uint32_t)vec_pass,"k_snapshot_head");
    }
    if ( LagunaKdaProbeDeep(buffers) )
    {
        float probe_pre[4],probe_post[4],probe_comb[4];
        LagunaProbeFloats(stream,buffers->hc_pre_f32,4u,probe_pre);
        LagunaProbeFloats(stream,buffers->hc_post_f32,4u,probe_post);
        LagunaProbeFloats(stream,buffers->hc_comb_f32,4u,probe_comb);
        fprintf(stderr,"G5N-PROBE kda L%u hc pre %.6g %.6g %.6g %.6g post %.6g %.6g %.6g %.6g comb %.6g %.6g %.6g %.6g\n",
            buffers->layer_index,(double)probe_pre[0],(double)probe_pre[1],(double)probe_pre[2],(double)probe_pre[3],
            (double)probe_post[0],(double)probe_post[1],(double)probe_post[2],(double)probe_post[3],
            (double)probe_comb[0],(double)probe_comb[1],(double)probe_comb[2],(double)probe_comb[3]);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"collapsed",buffers->hc_collapsed_bf16);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"normed",buffers->normed_bf16);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"attn_norm_weight",buffers->attn_norm_weight);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"qkv_beta_weight_row0",buffers->kda_qkv_beta_weight);
    }
    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kda_qkv_beta_weight,
        buffers->fused_qkvb_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
        rank_qk * 2u + rank_v + rank_heads,
        rank_qk * 2u + rank_v + rank_heads,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kda_decay_gate_down_weight,
        buffers->fused_decay_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
        2u * LAGUNA_KDA_LOW_RANK,
        2u * LAGUNA_KDA_LOW_RANK,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    LM_LAUNCH(
        (LagunaSplitFusedProjectionsKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->fused_qkvb_bf16,
        buffers->q_bf16,
        buffers->kv_slot_bf16   ,
        buffers->gate_up_bf16   ,
        buffers->kda_beta_logit,
        rows,
        rank_qk,
        rank_v,
        rank_heads,
        rank_qk * 2u + rank_v + rank_heads);
    LM_LAUNCH(
        (LagunaSplitDecayGateDownKernel<
            LAGUNA_LAYER_THREADS,LAGUNA_KDA_LOW_RANK>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->fused_decay_gate_bf16,
        buffers->kda_decay_latent_bf16,
        buffers->kda_gate_latent_bf16,
        rows);
    status = LagunaKdaReplayRecord(buffers, rows, rank_heads, 0u, stream);
    if (status != LM_LAUNCH_OK)
        return status;

    if ( LagunaKdaProbeActive(buffers) )
    {
        LAGUNA_KDA_PROBE(stream,"fused_qkvb",buffers->fused_qkvb_bf16,256u);
        LAGUNA_KDA_PROBE(stream,"decay_latent",buffers->kda_decay_latent_bf16,64u);
        LAGUNA_KDA_PROBE(stream,"gate_latent",buffers->kda_gate_latent_bf16,64u);
    }
    if ( vec_pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->fused_qkvb_bf16,rank_qk * 2u + rank_v + rank_heads,buffers->layer_index,(uint32_t)vec_pass,"fused_qkvb");
        LagunaProbeVecU16(stream,buffers->kda_decay_latent_bf16,LAGUNA_KDA_LOW_RANK,buffers->layer_index,(uint32_t)vec_pass,"decay_latent");
        LagunaProbeVecU16(stream,buffers->kda_gate_latent_bf16,LAGUNA_KDA_LOW_RANK,buffers->layer_index,(uint32_t)vec_pass,"gate_latent");
    }
    if ( LagunaKdaProbeDeep(buffers) )
    {
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"fused_qkvb",buffers->fused_qkvb_bf16);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"decay_latent",buffers->kda_decay_latent_bf16);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"gate_latent",buffers->kda_gate_latent_bf16);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"beta_logit",buffers->kda_beta_logit);
    }
    LM_LAUNCH(
        (LmCausalConvKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_qk + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS),
        LAGUNA_LAYER_THREADS,
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
        (LmCausalConvKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_qk + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS),
        LAGUNA_LAYER_THREADS,
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
        (LmCausalConvKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_v + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS),
        LAGUNA_LAYER_THREADS,
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
    if ( LagunaKdaProbeActive(buffers) )
    {
        LAGUNA_KDA_PROBE(stream,"q_postconv",buffers->q_bf16,256u);
        LAGUNA_KDA_PROBE(stream,"k_postconv",buffers->kv_slot_bf16,256u);
        LAGUNA_KDA_PROBE(stream,"v_postconv",buffers->gate_up_bf16,256u);
        LAGUNA_KDA_PROBE(stream,"beta_logit",buffers->kda_beta_logit,64u);
    }
    if ( vec_pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->q_bf16,rank_qk,buffers->layer_index,(uint32_t)vec_pass,"q_postconv");
        LagunaProbeVecU16(stream,buffers->kv_slot_bf16,rank_qk,buffers->layer_index,(uint32_t)vec_pass,"k_postconv");
        LagunaProbeVecU16(stream,buffers->gate_up_bf16,rank_v,buffers->layer_index,(uint32_t)vec_pass,"v_postconv");
    }
    if ( LagunaKdaProbeDeep(buffers) )
    {
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"q_postconv",buffers->q_bf16);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"k_postconv",buffers->kv_slot_bf16);
        LAGUNA_KDA_PROBE_RAW(stream,buffers->layer_index,"v_postconv",buffers->gate_up_bf16);
    }
    status = LagunaLaunchBf16Linear(
        buffers->kda_decay_latent_bf16,
        buffers->kda_decay_up_weight,
        buffers->kda_decay_logit_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_KDA_LOW_RANK,
        rank_qk,
        rank_qk,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = LagunaLaunchBf16Linear(
        buffers->kda_gate_latent_bf16,
        buffers->kda_gate_up_weight,
        buffers->kda_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_KDA_LOW_RANK,
        rank_v,
        rank_v,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    if ( LagunaKdaProbeActive(buffers) )
    {
        LAGUNA_KDA_PROBE(stream,"decay_logit",buffers->kda_decay_logit_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->kda_decay_logit_bf16,rank_qk,buffers->layer_index,(uint32_t)vec_pass,"decay_logit");
    }
    LM_LAUNCH(
        (LmBoundedDecayKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_KEY_DIM>),
        dim3(rows,rank_heads),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->kda_decay_logit_bf16,
        buffers->kda_decay_bias,
        buffers->kda_head_log_scale,
        buffers->kda_retention,
        rank_heads,
        LAGUNA_KDA_GATE_LOWER_BOUND,
        rows);
    LM_LAUNCH(
        (LmBf16SigmoidRowsKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->kda_beta_logit,
        buffers->kda_write_gate,
        rank_heads);
    if ( LagunaKdaProbeActive(buffers) )
    {
        float probe_r[4],probe_b[4];
        LagunaProbeFloats(stream,buffers->kda_retention,4u,probe_r);
        LagunaProbeFloats(stream,buffers->kda_write_gate,4u,probe_b);
        fprintf(stderr,"G5N-PROBE kda L%u retention %.6g %.6g %.6g %.6g write_gate %.6g %.6g %.6g %.6g\n",
            buffers->layer_index,(double)probe_r[0],(double)probe_r[1],(double)probe_r[2],
            (double)probe_r[3],(double)probe_b[0],(double)probe_b[1],(double)probe_b[2],(double)probe_b[3]);
        LAGUNA_KDA_PROBE_STATE(stream,"state_pre",buffers->kda_state_pool);
    }
    if ( vec_pass != 0 )
    {
        LagunaProbeVecF32(stream,buffers->kda_retention,rank_qk,buffers->layer_index,(uint32_t)vec_pass,"retention");
        LagunaProbeVecF32(stream,buffers->kda_write_gate,rank_heads,buffers->layer_index,(uint32_t)vec_pass,"write_gate");
    }
    status = LagunaDeltaRuleOptIn(
        LAGUNA_KDA_KEY_DIM * LAGUNA_KDA_VALUE_DIM * sizeof(float));
    if (status != LM_LAUNCH_OK)
        return(status);
#ifdef LAGUNA_KDA_DEBUG_LAUNCHES
    fprintf(stderr,"kda delta: grid(%u,%u) threads %u shared %u heads %u vhp %u seqs %u slot_bytes %u q=%p k=%p v=%p out=%p\n",
        sequences,rank_heads,LAGUNA_LAYER_THREADS,
        (unsigned)(LAGUNA_KDA_KEY_DIM * LAGUNA_KDA_VALUE_DIM * sizeof(float)),
        rank_heads,1u,sequences,buffers->kda_state_slot_bytes,
        (void*)buffers->q_bf16,(void*)buffers->kv_slot_bf16,(void*)buffers->gate_up_bf16,
        (void*)buffers->attention_out_bf16);
#endif
    LM_LAUNCH(
        (LmDeltaRuleKernel<LAGUNA_LAYER_THREADS,LAGUNA_KDA_KEY_DIM,LAGUNA_KDA_VALUE_DIM>),
        dim3(sequences,rank_heads),
        LAGUNA_LAYER_THREADS,
        LAGUNA_KDA_KEY_DIM * LAGUNA_KDA_VALUE_DIM * sizeof(float),
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
        commit);
    status = LagunaKdaReplayRecord(buffers, rows, rank_heads, 1u, stream);
    if (status != LM_LAUNCH_OK)
        return status;
    if ( LagunaKdaProbeActive(buffers) )
    {
        LAGUNA_KDA_PROBE(stream,"delta_out_raw",buffers->attention_out_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->attention_out_bf16,rank_v,buffers->layer_index,(uint32_t)vec_pass,"delta_out");
    }
#ifdef LAGUNA_KDA_DEBUG_LAUNCHES
    fprintf(stderr,"kda norm: grid %llu threads %u shared %u dim %u\n",
        (unsigned long long)((uint64_t)rows * rank_heads),LAGUNA_LAYER_THREADS,
        (unsigned)((LAGUNA_KDA_VALUE_DIM + 8u) * sizeof(float)),
        LAGUNA_KDA_VALUE_DIM);
#endif
    LM_LAUNCH(
        (LmRmsNormSigmoidGateKernel<LAGUNA_LAYER_THREADS>),
        dim3((uint64_t)rows * rank_heads),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->kda_gate_bf16,
        (const float *)buffers->kda_out_norm_weight,
        buffers->attention_out_bf16,
        LAGUNA_KDA_VALUE_DIM,
        LAGUNA_RMS_EPSILON);
    if ( LagunaKdaProbeActive(buffers) )
    {
        LAGUNA_KDA_PROBE(stream,"delta_out_gated",buffers->attention_out_bf16,256u);
        LAGUNA_KDA_PROBE(stream,"kda_gate",buffers->kda_gate_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->attention_out_bf16,rank_v,buffers->layer_index,(uint32_t)vec_pass,"delta_gated");
        LagunaProbeVecU16(stream,buffers->kda_gate_bf16,rank_v,buffers->layer_index,(uint32_t)vec_pass,"kda_gate");
    }
    LM_LAUNCH(
        (LmCopyRowsKernel<LAGUNA_LAYER_THREADS>),
        dim3((rank_v + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS,rows),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->kv_slot_bf16,
        rows,
        rank_v);
    status = LagunaLaunchBf16Linear(
        buffers->kv_slot_bf16,
        buffers->kda_out_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        rank_v,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if ( status == LM_LAUNCH_OK && LagunaKdaProbeActive(buffers) )
    {
        float probe_o[4];
        LAGUNA_KDA_PROBE(stream,"kda_out_partial",buffers->attention_out_bf16,256u);
        LagunaProbeBf16Floats(stream,buffers->attention_out_bf16,4u,probe_o);
        fprintf(stderr,"G5N-PROBE kda L%u out_partial_f %.6g %.6g %.6g %.6g\n",
            buffers->layer_index,(double)probe_o[0],(double)probe_o[1],(double)probe_o[2],(double)probe_o[3]);
        LAGUNA_KDA_PROBE_STATE(stream,"state_post",buffers->kda_state_pool);
    }
    if ( status == LM_LAUNCH_OK && vec_pass != 0 )
    {
        uint32_t vec_head;
        LagunaProbeVecU16(stream,buffers->attention_out_bf16,LAGUNA_HIDDEN,buffers->layer_index,(uint32_t)vec_pass,"out_partial");
        for ( vec_head = 0u; vec_head < rank_heads; vec_head++ )
            LagunaProbeVecF32(stream,(const float *)buffers->kda_state_pool +
                ((uint64_t)vec_head * LAGUNA_KDA_KEY_DIM * LAGUNA_KDA_VALUE_DIM),
                LAGUNA_KDA_KEY_DIM * LAGUNA_KDA_VALUE_DIM,buffers->layer_index,(uint32_t)vec_pass,
                vec_head == 0u ? "state_h0" : (vec_head == 1u ? "state_h1" :
                (vec_head == 2u ? "state_h2" : "state_h3")));
    }
    return(status);
}

__global__ void LagunaHcSplitSinkhornKernel(
    const float *__restrict__ mixes_f32,
    const float *__restrict__ scale3_f32,
    const float *__restrict__ base_f32,
    uint32_t row_count,
    uint32_t hc,
    uint32_t iterations,
    float epsilon,
    float *__restrict__ pre_f32,
    float *__restrict__ post_f32,
    float *__restrict__ comb_f32)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i, j, iteration;
    const uint32_t mix_rows = (2u + hc) * hc;
    const float *mixes;
    float comb[16], maximum, total;
    if (row >= row_count || hc > 4u)
        return;
    mixes = mixes_f32 + (uint64_t)row * mix_rows;
    for (i = 0u; i < hc; ++i)
    {
        pre_f32[(uint64_t)row * hc + i] =
            1.0f / (1.0f + __expf(-(mixes[i] * scale3_f32[0] + base_f32[i]))) +
            epsilon;
        post_f32[(uint64_t)row * hc + i] =
            2.0f / (1.0f + __expf(-(mixes[hc + i] * scale3_f32[1] +
                                    base_f32[hc + i])));
    }
    for (i = 0u; i < hc; ++i)
    {
        maximum = -3.0e38f;
        for (j = 0u; j < hc; ++j)
        {
            comb[i * hc + j] =
                mixes[2u * hc + i * hc + j] * scale3_f32[2] +
                base_f32[2u * hc + i * hc + j];
            maximum = fmaxf(maximum, comb[i * hc + j]);
        }
        total = 0.0f;
        for (j = 0u; j < hc; ++j)
            total += (comb[i * hc + j] = __expf(comb[i * hc + j] - maximum));
        for (j = 0u; j < hc; ++j)
            comb[i * hc + j] = comb[i * hc + j] / total + epsilon;
    }
    for (iteration = 0u; iteration < iterations; ++iteration)
    {
        if (iteration != 0u)
            for (i = 0u; i < hc; ++i)
            {
                total = 0.0f;
                for (j = 0u; j < hc; ++j)
                    total += comb[i * hc + j];
                for (j = 0u; j < hc; ++j)
                    comb[i * hc + j] /= total + epsilon;
            }
        for (j = 0u; j < hc; ++j)
        {
            total = 0.0f;
            for (i = 0u; i < hc; ++i)
                total += comb[i * hc + j];
            for (i = 0u; i < hc; ++i)
                comb[i * hc + j] /= total + epsilon;
        }
    }
    for (i = 0u; i < hc * hc; ++i)
        comb_f32[(uint64_t)row * hc * hc + i] = comb[i];
}

#define LAGUNA_HC_MIX_TILE 4096u
__global__ void LagunaHcMixKernel(
    const uint16_t *__restrict__ streams_bf16,
    const float *__restrict__ fn_f32,
    float *__restrict__ mixes_f32,
    uint32_t row_count,
    uint32_t flat_dimension,
    uint32_t mix_rows,
    float rms_epsilon)
{
    extern __shared__ float staged[];
    __shared__ float reduction[LAGUNA_LAYER_THREADS / LM_WARP_LANES];
    uint32_t row = blockIdx.x;
    uint32_t warp = threadIdx.x / LM_WARP_LANES;
    uint32_t lane = threadIdx.x % LM_WARP_LANES;
    uint32_t mix, element, tile, tile_end, tile_elements;
    const uint32_t warps = LAGUNA_LAYER_THREADS / LM_WARP_LANES;
    float value, total = 0.0f, accumulator;
    float accum[4];
    if (row >= row_count)
        return;
    for (mix = 0u; mix < 3u; mix++)
        accum[mix] = 0.0f;
    for (tile = 0u; tile < flat_dimension; tile += LAGUNA_HC_MIX_TILE)
    {
        tile_end = tile + LAGUNA_HC_MIX_TILE < flat_dimension
            ? tile + LAGUNA_HC_MIX_TILE
            : flat_dimension;
        tile_elements = tile_end - tile;
        __syncthreads();
        for (element = threadIdx.x; element < tile_elements;
             element += LAGUNA_LAYER_THREADS)
        {
            value = LmBf16ToFloat(
                streams_bf16[((uint64_t)row * flat_dimension) + tile + element]);
            staged[element] = value;
            total += value * value;
        }
        __syncthreads();
        for (mix = warp; mix < mix_rows; mix += warps)
        {
            accumulator = 0.0f;
            for (element = lane; element < tile_elements; element += LM_WARP_LANES)
                accumulator += staged[element] *
                    fn_f32[((uint64_t)mix * flat_dimension) + tile + element];
            for (uint32_t lane_step = LM_WARP_LANES / 2u; lane_step > 0u;
                 lane_step >>= 1)
                accumulator +=
                    __shfl_down_sync(0xFFFFFFFFu, accumulator, lane_step);
            if (lane == 0u)
                accum[mix / warps] += accumulator;
        }
    }
    total = LmBlockSum<LAGUNA_LAYER_THREADS>(total, reduction);
    __shared__ float inverse_shared[1];
    if (threadIdx.x == 0u)
        inverse_shared[0] =
            rsqrtf(total / (float)flat_dimension + rms_epsilon);
    __syncthreads();
    for (mix = warp; mix < mix_rows; mix += warps)
        if (lane == 0u)
            mixes_f32[((uint64_t)row * mix_rows) + mix] =
                accum[mix / warps] * inverse_shared[0];
}

__global__ void LagunaHcPreReduceKernel(
    const uint16_t *__restrict__ streams_bf16,
    const float *__restrict__ pre_f32,
    uint16_t *__restrict__ collapsed_bf16,
    uint16_t *__restrict__ snapshot_bf16,
    uint32_t row_count,
    uint32_t hc,
    uint32_t dimension)
{
    uint32_t row = blockIdx.x;
    uint32_t element, stream;
    float value;
    if (row >= row_count)
        return;
    for (element = threadIdx.x; element < dimension;
         element += blockDim.x)
    {
        value = 0.0f;
        for (stream = 0u; stream < hc; ++stream)
        {
            uint64_t index =
                (((uint64_t)row * hc) + stream) * dimension + element;
            uint16_t raw = streams_bf16[index];
            snapshot_bf16[index] = raw;
            value += pre_f32[((uint64_t)row * hc) + stream] *
                LmBf16ToFloat(raw);
        }
        collapsed_bf16[(uint64_t)row * dimension + element] =
            LmFloatToBf16(value);
    }
}

__global__ void LagunaHcHeadMeanKernel(
    const uint16_t *__restrict__ streams_bf16,
    uint16_t *__restrict__ reduced_bf16,
    uint32_t row_count,
    uint32_t hc,
    uint32_t dimension)
{
    uint32_t row = blockIdx.x;
    uint32_t element, stream;
    float value;
    if (row >= row_count)
        return;
    for (element = threadIdx.x; element < dimension; element += blockDim.x)
    {
        value = 0.0f;
        for (stream = 0u; stream < hc; ++stream)
            value += LmBf16ToFloat(
                streams_bf16[((uint64_t)row * hc + stream) * dimension +
                             element]);
        reduced_bf16[(uint64_t)row * dimension + element] =
            LmFloatToBf16(value / (float)hc);
    }
}

static int32_t LagunaHcSite(
    const LagunaLayerBuffers *buffers,
    const void *fn_weight,
    const void *base_weight,
    const void *scale_weight,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    uint32_t mix_rows = LAGUNA_HC_MIX;
    uint32_t flat = LAGUNA_HC_FLAT;
    if (buffers->hc_mixes_f32 == 0 || buffers->hc_pre_f32 == 0 ||
        buffers->hc_post_f32 == 0 || buffers->hc_comb_f32 == 0 ||
        buffers->hc_collapsed_bf16 == 0 || buffers->hc_snapshot_bf16 == 0 ||
        fn_weight == 0 || base_weight == 0 || scale_weight == 0)
        return LM_LAUNCH_ERR_SHAPE;
    LM_LAUNCH(
        (LagunaHcMixKernel),
        rows,
        LAGUNA_LAYER_THREADS,
        LAGUNA_HC_MIX_TILE * sizeof(float),
        stream,
        buffers->hidden_bf16   ,
        (const float *)fn_weight,
        buffers->hc_mixes_f32,
        rows,
        flat,
        mix_rows,
        LAGUNA_RMS_EPSILON);
    LM_LAUNCH(
        (LagunaHcSplitSinkhornKernel),
        (rows + 63u) / 64u,
        64u,
        0,
        stream,
        buffers->hc_mixes_f32,
        (const float *)scale_weight,
        (const float *)base_weight,
        rows,
        LAGUNA_HC,
        LAGUNA_HC_SINKHORN_ITERATIONS,
        LAGUNA_HC_EPSILON,
        buffers->hc_pre_f32,
        buffers->hc_post_f32,
        buffers->hc_comb_f32);
    LM_LAUNCH(
        (LagunaHcPreReduceKernel),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->hidden_bf16,
        buffers->hc_pre_f32,
        buffers->hc_collapsed_bf16,
        buffers->hc_snapshot_bf16,
        rows,
        LAGUNA_HC,
        LAGUNA_HIDDEN);
    return LM_LAUNCH_OK;
}

static int32_t LagunaHcPost(
    const LagunaLayerBuffers *buffers,
    const uint16_t *sublayer_out,
    uint32_t rows,
    cudaStream_t stream)
{
    static uint32_t probe_count = 0u;
    int32_t pass = LagunaLayerProbeVecPass(buffers,&probe_count,2u);
    if ( pass != 0 )
    {
        LagunaProbeVecU16(stream,sublayer_out,LAGUNA_HIDDEN,buffers->layer_index,(uint32_t)pass,"hc_sublayer_out");
        LagunaProbeVecU16(stream,buffers->hc_snapshot_bf16,LAGUNA_HC_FLAT,buffers->layer_index,(uint32_t)pass,"hc_snapshot");
        LagunaProbeVecF32(stream,buffers->hc_post_f32,LAGUNA_HC,buffers->layer_index,(uint32_t)pass,"hc_post_weights");
        LagunaProbeVecF32(stream,buffers->hc_comb_f32,LAGUNA_HC * LAGUNA_HC,buffers->layer_index,(uint32_t)pass,"hc_comb_weights");
    }
    LM_LAUNCH(
        (LmHcPostBf16Kernel),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        sublayer_out,
        buffers->hc_snapshot_bf16,
        buffers->hc_post_f32,
        buffers->hc_comb_f32,
        buffers->hidden_bf16,
        rows,
        LAGUNA_HC,
        LAGUNA_HIDDEN);
    if ( pass != 0 )
        LagunaProbeVecU16(stream,buffers->hidden_bf16,LAGUNA_HC_FLAT,buffers->layer_index,(uint32_t)pass,"hc_result");
    return LM_LAUNCH_OK;
}


static int32_t LagunaLayerDenseMlp(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    static uint32_t probe_count = 0u;
    int32_t status,pass = LagunaLayerProbeVecPass(buffers,&probe_count,1u);

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
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->normed_bf16,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        LAGUNA_RMS_EPSILON);

    if ( pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->hc_collapsed_bf16,LAGUNA_HIDDEN,buffers->layer_index,(uint32_t)pass,"dense_collapsed");
        LagunaProbeVecU16(stream,buffers->normed_bf16,LAGUNA_HIDDEN,buffers->layer_index,(uint32_t)pass,"dense_normed");
    }
    if (buffers->dense_gate_up_fused != 0u)
    {
        status = LagunaLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            LAGUNA_HIDDEN,
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
        status = LagunaLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            LAGUNA_HIDDEN,
            buffers->dense_intermediate,
            buffers->dense_gate_up_rows,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
        status = LagunaLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_up_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            LAGUNA_HIDDEN,
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
        (LmClampedUpGateKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->dense_intermediate,
        SPARK_LAGUNA_MODEL_SWIGLU_LIMIT);

    if ( pass != 0 )
    {
        LagunaProbeVecU16(stream,buffers->gate_up_bf16,buffers->dense_gate_up_rows,buffers->layer_index,(uint32_t)pass,"dense_gate_up");
        LagunaProbeVecU16(stream,buffers->intermediate_bf16,buffers->dense_intermediate,buffers->layer_index,(uint32_t)pass,"dense_intermediate");
    }
    status = LagunaLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->dense_down_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->dense_intermediate,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if ( status == LM_LAUNCH_OK && pass != 0 )
        LagunaProbeVecU16(stream,buffers->attention_out_bf16,LAGUNA_HIDDEN,buffers->layer_index,(uint32_t)pass,"dense_down_partial");
    return(status);
}

template<uint32_t ExpertCodec>
static int32_t LagunaLayerMoeValidate(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;

    static_assert(ExpertCodec != SPARK_WEIGHT_CODEC_BF16,
        "GLM 5.2 routed experts require an explicit compressed codec");
    static_assert(LAGUNA_HIDDEN % ExpertFormat::kScaleGroup == 0u &&
        LAGUNA_EXPERT_INTERMEDIATE % ExpertFormat::kScaleGroup == 0u,
        "GLM 5.2 expert dimensions must contain complete codec scale groups");

    if (buffers == 0 || rows == 0u ||
        packed_rows != rows * LAGUNA_TOP_K ||
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

    return LM_LAUNCH_OK;
}

template<uint32_t ExpertCodec>
static int32_t LagunaLayerMoeRoute(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGemmArguments gemm;
    int32_t status = LagunaLayerMoeValidate<ExpertCodec>(buffers,rows,packed_rows);
    if (status != LM_LAUNCH_OK)
        return status;
    LM_LAUNCH(
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        (const uint16_t *)buffers->mlp_norm_weight,
        buffers->normed_bf16,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        LAGUNA_RMS_EPSILON);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = buffers->dense_row_offset;
    gemm.group_tile_prefix = buffers->dense_tile_prefix;
    gemm.output_f32 = buffers->router_logits;
    status = LmGemmLaunch<
        LmBf16Format,
        LAGUNA_LAYER_TILE_N,
        LmBf16Format::kTileK,
        LAGUNA_LAYER_STAGES,
        LAGUNA_LAYER_WARPS>(
            &gemm,
            buffers->normed_bf16,
            buffers->router_weight,
            rows,
            rows,
            1u,
            1u,
            LAGUNA_HIDDEN,
            LAGUNA_EXPERTS,
            multiprocessors,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmTopkSmallKernel<
            LAGUNA_LAYER_THREADS,
            LAGUNA_TOP_K,
            true,
            1u,
            1u,
            LM_TOPK_SCORE_SIGMOID>),
        rows,
        LAGUNA_LAYER_THREADS,
        2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),
        stream,
        buffers->router_logits,
        LAGUNA_EXPERTS,
        buffers->route_expert,
        buffers->route_weight,
        buffers->router_correction_bias,
        0,
        LAGUNA_ROUTED_SCALE);
    status = LmRouteBuild<LAGUNA_LAYER_THREADS, LAGUNA_EXPERTS>(
        buffers->route_expert,
        rows,
        packed_rows,
        LAGUNA_TOP_K,
        buffers->group_row_offset,
        buffers->route_packed_row,
        buffers->route_source_token,
        buffers->expert_w1_rows,
        LAGUNA_HIDDEN,
        LAGUNA_LAYER_TILE_N,
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
static int32_t LagunaLayerMoeExperts(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
    LmGemmArguments gemm;
    int32_t status = LagunaLayerMoeValidate<ExpertCodec>(buffers,rows,packed_rows);
    if (status != LM_LAUNCH_OK)
        return status;
    if ( buffers->expert_w1_weight == 0 || buffers->expert_w1_scale == 0 ||
        buffers->expert_w2_weight == 0 || buffers->expert_w2_scale == 0 )
        return(LM_LAUNCH_ERR_SHAPE);
    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
        buffers->expert_w1_scale,
        LAGUNA_EXPERTS,
        buffers->expert_w1_rows,
        LAGUNA_HIDDEN);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
	gemm.source_row_map = buffers->route_source_token;
	gemm.source_row_count = rows;
    gemm.output_bf16 = buffers->gate_up_bf16;
    status = LmGemmWeightOnlyIndirectLaunch<
        ExpertFormat,
        LAGUNA_LAYER_TILE_N,
        LAGUNA_LAYER_STAGES,
        LAGUNA_LAYER_WARPS>(
            &gemm,
			buffers->normed_bf16,
            buffers->expert_w1_weight,
            packed_rows,
            rows,
            LAGUNA_TOP_K,
            LAGUNA_EXPERTS,
            LAGUNA_HIDDEN,
            buffers->expert_w1_rows,
            multiprocessors,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmClampedUpGateKernel<LAGUNA_LAYER_THREADS>),
        packed_rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->expert_intermediate,
        SPARK_LAGUNA_MODEL_SWIGLU_LIMIT);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
        buffers->expert_w2_scale,
        LAGUNA_EXPERTS,
        LAGUNA_HIDDEN,
        buffers->expert_intermediate);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
    gemm.output_bf16 = buffers->expert_out_bf16;
    status = LmGemmWeightOnlyLaunch<
        ExpertFormat,
        LAGUNA_LAYER_TILE_N,
        LAGUNA_LAYER_STAGES,
        LAGUNA_LAYER_WARPS>(
            &gemm,
            buffers->intermediate_bf16,
            buffers->expert_w2_weight,
            packed_rows,
            rows,
            LAGUNA_TOP_K,
            LAGUNA_EXPERTS,
            buffers->expert_intermediate,
            LAGUNA_HIDDEN,
            multiprocessors,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmMoeFinalizeKernel<LAGUNA_LAYER_THREADS>),
        dim3(
            (LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) /
                LAGUNA_LAYER_THREADS,
            rows),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->expert_out_bf16,
        buffers->route_packed_row,
        buffers->route_weight,
        buffers->attention_out_bf16,
        rows,
        LAGUNA_TOP_K,
        LAGUNA_HIDDEN);
    status = LagunaLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->shared_gate_up_weight,
        buffers->gate_up_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        LAGUNA_HIDDEN,
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
        (LmClampedUpGateKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->shared_intermediate,
        SPARK_LAGUNA_MODEL_SWIGLU_LIMIT);
    status = LagunaLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->shared_down_weight,
        buffers->shared_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->shared_intermediate,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmAddRowsKernel<LAGUNA_LAYER_THREADS>),
        dim3(
            (LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) /
                LAGUNA_LAYER_THREADS,
            rows),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->shared_out_bf16,
        buffers->attention_out_bf16,
        rows,
        LAGUNA_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

// Resident execution retains the same submission order. Lazy execution can
// acquire/import the routed working set between these two calls on this stream.
template<uint32_t ExpertCodec>
static int32_t LagunaLayerMoe(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status = LagunaLayerMoeRoute<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;
    return LagunaLayerMoeExperts<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
}

static int32_t LagunaHead(
    const LagunaLayerBuffers *buffers,
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

    tiles = (vocabulary + LAGUNA_HEAD_TILE - 1u) / LAGUNA_HEAD_TILE;
    LM_LAUNCH(
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_mean_bf16,
        (const uint16_t *)head_norm_weight,
        buffers->normed_bf16,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        LAGUNA_RMS_EPSILON);
    LM_LAUNCH(
        (LmHeadCandidateKernel<LAGUNA_LAYER_THREADS, LAGUNA_HEAD_TILE>),
        dim3(tiles, rows),
        LAGUNA_LAYER_THREADS,
        0,
        stream,
        buffers->normed_bf16,
        (const uint16_t *)head_weight,
        token_ids,
        buffers->head_candidate_score,
        buffers->head_candidate_token,
        rows,
        LAGUNA_HIDDEN,
        vocabulary);
    LM_LAUNCH(
        (LmHeadCommitKernel<LAGUNA_LAYER_THREADS>),
        rows,
        LAGUNA_LAYER_THREADS,
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

static int32_t LagunaHeadCertifiedB1(
    const LagunaLayerBuffers *buffers,
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
        buffers->hc_mean_bf16 == 0 || buffers->normed_bf16 == 0 ||
        buffers->output_token == 0 || buffers->output_score == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    LM_LAUNCH(
        (LmBf16RmsNormKernel<LAGUNA_LAYER_THREADS>),
        1u,
        LAGUNA_LAYER_THREADS,
        (LAGUNA_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_mean_bf16,
        (const uint16_t *)head_norm_weight,
        buffers->normed_bf16,
        LAGUNA_HIDDEN,
        LAGUNA_HIDDEN,
        LAGUNA_RMS_EPSILON);
    status = SparkLmHostLaunchHeadCertifiedFp8B1WithScore(
        stream, buffers->normed_bf16, head_weight, certified_payload,
        certified_scale, certified_norm, certified_scratch, candidate_ids,
        screened_count, buffers->output_token, buffers->output_score,
        rank_offset, 1u, vocabulary, LAGUNA_HIDDEN);
    return status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
