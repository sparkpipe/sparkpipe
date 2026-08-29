#pragma once

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "modules/glm5_next_resident_decode_stage/source/cuda/config.h"

// Block-major KV geometry: one logical block = GLM5_NEXT_KV_PAGE_SLOTS
// tokens across the DSA layers (only the 11 sparse layers carry an MLA KV
// row; KDA layers are O(1)-state and need none), laid out contiguously in
// the common SparkKvCacheArena block-major layout. kPageBytes is the block
// stride; the per-layer pool base the driver passes already carries the
// DSA-layer offset, so SlotInPage stays the within-block token index.
struct Glm5NextKv
{
    /* PER-LAYER view: one DSA layer's sub-pool. The module's pool is
     * layer-major - layer l's sub-pool sits at pool + l * (pages * 64 *
     * slot) - and the kernel addresses pages within its layer's sub-pool,
     * so this page stride covers ONE layer's tokens, not eleven. (The
     * first bring-up sized it x DSA_COUNT and the pool came out 65 GB.)
     */
    static constexpr uint32_t kSlotBytes = GLM5_NEXT_KV_SLOT_BYTES;
    static constexpr uint32_t kPageSlots = GLM5_NEXT_KV_PAGE_SLOTS;
    static constexpr uint32_t kPageBytes =
        GLM5_NEXT_KV_SLOT_BYTES * GLM5_NEXT_KV_PAGE_SLOTS;
    static constexpr bool kGrows = true;
    static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
    { return position / GLM5_NEXT_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
    { return position % GLM5_NEXT_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
    { return (tokens + GLM5_NEXT_KV_PAGE_SLOTS - 1u) / GLM5_NEXT_KV_PAGE_SLOTS; }
    static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
    { return pages * (uint64_t)kPageBytes; }
};
/* The packed indexer cache row: [ k(128) | gate(128) | valid(1) ] bf16.
 * The gate half carries the per-token pool-mix logits; valid is 1.0 for
 * every stored row (only real tokens are stored). */
struct Glm5NextIndexKv
{
    static constexpr uint32_t kSlotBytes =
        SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
    static constexpr uint32_t kPageSlots = GLM5_NEXT_KV_PAGE_SLOTS;
    static constexpr uint32_t kPageBytes =
        kSlotBytes * kPageSlots * SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT;
    static constexpr bool kGrows = true;
    static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
    { return position / kPageSlots; }
    static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
    { return position % kPageSlots; }
    static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
    { return (tokens + kPageSlots - 1u) / kPageSlots; }
    static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
    { return pages * (uint64_t)kPageBytes; }
};

/* Fuse [ k | gate | 1.0 ] into the packed 257-wide indexer cache row. */
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void Glm5NextIndexPackKernel(
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
        packed_bf16[base + 2u * dimension] = 0x3F80u; /* bf16 1.0 */
}

/* Score one 4-token k-pool for every query row: one block per (pool, row).
 *
 * pool_key = sum_j softmax_j(gate_j + ape_j) * k_j over the pool's KPOOL
 * positions (the reference's get_pooled_states; the softmax is per-channel
 * over pool positions); score_h = relu(q_h . pool_key * 128**-0.5); the
 * row's pool score is the head-weighted sum weights_proj(x)_h * 32**-0.5.
 * Positions past the sequence's context contribute nothing (masked out of
 * the softmax with -inf, exactly as the reference's grouped_valid_keys).
 */
template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS>
__global__ __launch_bounds__(THREADS, 1)
void Glm5NextPoolScoreKernel(
    const uint16_t *__restrict__ query_bf16,      /* [rows, HEADS*DIM] */
    const uint16_t *__restrict__ head_weight_bf16,/* [rows, HEADS] */
    LmKvView index_cache,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    const float *__restrict__ compress_ape_f32,   /* [KPOOL, DIM] */
    uint32_t pools,
    float softmax_scale,
    float head_scale,
    float *__restrict__ pool_scores)              /* [rows, pools] */
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

    /* Per-channel softmax over the pool's gate logits ( ape bias included )
     * then the weighted key sum, both over DIM channels in parallel. */
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
            const uint16_t *slot = (const uint16_t *)LmKvSlotRequired<Glm5NextIndexKv>(
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
            const uint16_t *slot = (const uint16_t *)LmKvSlotRequired<Glm5NextIndexKv>(
                index_cache, sequence, position, row, LM_KV_ACCESS_READ);
            if (slot == 0)
                return;
            key += mix[slot_in_pool] * LmBf16ToFloat(slot[index]);
        }
        pool_key[index] = key;
    }
    __syncthreads();

    /* Head-major dot products with per-head relu, then the weighted head
     * sum. One thread per (head, channel-slice); block reduction per head
     * would serialise, so each thread owns whole heads when HEADS <=
     * THREADS and accumulates across its heads' channels serially. */
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

/* Expand selected pools into raw token positions and append the tail.
 * Output width is TOPK + KPOOL - 1 (2051): 512 pools x 4 tokens, then the
 * incomplete tail (up to 3). Slots beyond the tail count carry UINT32_MAX,
 * which the latent decode kernel's causal check (position > row_position)
 * skips without a cache read. */
template<uint32_t THREADS, uint32_t KPOOL, uint32_t TOPK, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 1)
void Glm5NextPoolExpandKernel(
    const uint32_t *__restrict__ selected_pools,  /* [rows, TOPK/KPOOL] */
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    uint32_t *__restrict__ selected_positions,    /* [rows, WIDTH] */
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
            /* A pool is only selectable complete: clamp incomplete tails
             * out to the sentinel. */
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

#include "modules/glm5_next_resident_decode_stage/source/cuda/launch_shape.h"

#define GLM5_NEXT_LAYER_TILE_N 128u
#define GLM5_NEXT_LAYER_STAGES 2u
#define GLM5_NEXT_LAYER_WARPS 8u
#define GLM5_NEXT_HEAD_TILE 1024u

static_assert(
    GLM5_NEXT_HIDDEN % LmBf16Format::kTileK == 0u,
    "glm5_next hidden projections must cover every BF16 K tile");
static_assert(
    GLM5_NEXT_QUERY_A_DIM % LmBf16Format::kTileK == 0u,
    "glm5_next low-rank query projections must cover every BF16 K tile");
static_assert(
    GLM5_NEXT_DSA_QUERY_DIM % LmBf16Format::kTileK == 0u,
    "glm5_next DSA index queries must cover every BF16 K tile");
static_assert(
    (GLM5_NEXT_ATTN_HEADS * GLM5_NEXT_LATENT) % LmBf16Format::kTileK == 0u,
    "glm5_next latent attention output must cover every BF16 K tile");
static_assert(
    GLM5_NEXT_DENSE_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "glm5_next dense FFN down projection must cover every BF16 K tile");
static_assert(
    GLM5_NEXT_EXPERT_INTERMEDIATE % LmBf16Format::kTileK == 0u,
    "glm5_next expert down projection must cover every BF16 K tile");
static_assert(
    GLM5_NEXT_KDA_QK_DIM % LmBf16Format::kTileK == 0u,
    "glm5_next fused KDA q|k|v|beta rows must cover every BF16 K tile");
struct Glm5NextLayerBuffers
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

    // Tensor-parallel sharding. The pack stores per-rank shards of the row- or
    // column-sharded projections, so every projection dimension the kernels
    // price must come from here rather than the full-model constant; tp_degree
    // == 1 recovers the single-rank geometry. Replicated tensors (norms, q_a,
    // kv_a, kv_b, indexer, router, correction) keep their full dims and the KV
    // cache keeps all heads on every rank.
    uint32_t tp_degree;
    uint32_t tp_rank;
    /* G5N-PROBE only: the layer's global index, for ordinal-gated dumps. */
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

    /* -- KDA (k3 donor): weights, per-layer state, and scratch. Every
     * per-head dimension comes from kda_heads (the rank's shard), never
     * the full-model constant. */
    const void *kda_qkv_beta_weight;
    const void *kda_decay_gate_down_weight;
    const void *kda_decay_up_weight;
    const void *kda_gate_up_weight;
    /* BF16 conv weights, unconverted from the checkpoint (k3 packs them
     * F32; glm5_next's LmCausalConvKernel<uint16_t> reads bf16 taps). */
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

    /* -- hyper-connections (dsv4 donor): weights and per-site scratch.
     * hidden_bf16 carries hc_mult streams of GLM5_NEXT_HIDDEN (the HC
     * "streams_bf16" surface); residual semantics live in the HC launch
     * pair, not the residual pointer. */
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

    /* -- indexer kpool compressor (dsv4 mechanism, glm53 geometry) */
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
    /* R3 flash-decode: the split path engages only above the deployment
     * threshold and only with the partials workspace bound. */
    float *attention_split_partials;
    uint64_t attention_split_partial_blocks;
    uint32_t decode_split_context_threshold;
};

// The firmware header sizes the partials workspace; the shared kernel policy
// owns the partition cap. One definition of the cap, checked not duplicated.
static_assert(
    LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS ==
        SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTITIONS,
    "the firmware's split-partials sizing must match the kernel's "
    "partition cap");

static int32_t Glm5NextLaunchBf16Linear(
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
        GLM5_NEXT_LAYER_TILE_N,
        LmBf16Format::kTileK,
        GLM5_NEXT_LAYER_STAGES,
        GLM5_NEXT_LAYER_WARPS>(
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

static int32_t Glm5NextLayerIndexer(
    const Glm5NextLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status;
    uint32_t pools;

    /* Every DSA layer is a FULL indexer in glm5_next (indexer_types is 45x
     * "full"); the caller gates on layer kind. The kpool selection is
     * bypassed while the context cannot fill the pool budget (context at
     * or below TOPK selects everything): the projections and cache stores
     * below still run because the packed cache row must exist for future
     * contexts, but the pool scoring pass is pure overhead there. */
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
    status = Glm5NextLaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->index_q_weight,
        buffers->index_query_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_QUERY_A_DIM,
        GLM5_NEXT_DSA_QUERY_DIM,
        GLM5_NEXT_DSA_QUERY_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_k_weight,
        buffers->index_key_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_DSA_INDEX_DIM,
        GLM5_NEXT_DSA_INDEX_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    /* The compressor's per-token gate logits: hidden -> 128, packed beside
     * the key. NO rope on either half - NoPE model, and
     * indexer_rope_interleave is not a porting dependency. */
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_compress_gate,
        buffers->index_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_DSA_INDEX_DIM,
        GLM5_NEXT_DSA_INDEX_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmLayerNormKernel<GLM5_NEXT_LAYER_THREADS,uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_DSA_INDEX_DIM + 8u) * sizeof(float),
        stream,
        buffers->index_key_bf16,
        (const uint16_t *)buffers->index_norm_weight,
        (const uint16_t *)buffers->index_norm_bias,
        buffers->index_key_bf16,
        GLM5_NEXT_DSA_INDEX_DIM,
        GLM5_NEXT_DSA_INDEX_DIM,
        GLM5_NEXT_DSA_INDEX_EPSILON);
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->index_head_weight,
        buffers->index_head_weight_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_DSA_INDEX_HEADS,
        GLM5_NEXT_DSA_INDEX_HEADS,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (Glm5NextIndexPackKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->index_key_bf16,
        buffers->index_gate_bf16,
        buffers->index_packed_bf16,
        GLM5_NEXT_DSA_INDEX_DIM);
    LM_LAUNCH(
        (LmKvStoreKernel<Glm5NextIndexKv,GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->index_cache,
        buffers->index_packed_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION);
    if (context <= GLM5_NEXT_DSA_SELECTED)
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
    /* Complete pools only: context > TOPK guarantees at least TOPK/KPOOL
     * of them, so every selected pool expands to four real positions. */
    pools = context / GLM5_NEXT_DSA_KPOOL;
    LM_LAUNCH(
        (Glm5NextPoolScoreKernel<
            GLM5_NEXT_LAYER_THREADS,
            GLM5_NEXT_DSA_INDEX_DIM,
            GLM5_NEXT_DSA_KPOOL,
            GLM5_NEXT_DSA_INDEX_HEADS>),
        dim3(pools,rows),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->index_query_bf16,
        buffers->index_head_weight_bf16,
        buffers->index_cache,
        buffers->sequence_of_row,
        buffers->context_length,
        (const float *)buffers->index_compress_ape,
        pools,
        GLM5_NEXT_DSA_INDEX_SCALE,
        GLM5_NEXT_DSA_INDEX_HEAD_WEIGHT_SCALE,
        buffers->selection_scores);
    LM_LAUNCH(
        (LmTopkHistogramKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->selection_scores,
        pools,
        GLM5_NEXT_DSA_SELECTED / GLM5_NEXT_DSA_KPOOL,
        buffers->head_candidate_token);
    LM_LAUNCH(
        (LmTopkGatherKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->selection_scores,
        pools,
        GLM5_NEXT_DSA_SELECTED / GLM5_NEXT_DSA_KPOOL,
        buffers->head_candidate_token,
        buffers->selected_pools,
        0);
    LM_LAUNCH(
        (Glm5NextPoolExpandKernel<
            GLM5_NEXT_LAYER_THREADS,
            GLM5_NEXT_DSA_KPOOL,
            GLM5_NEXT_DSA_SELECTED,
            SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
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

/* DSA-site full-vector probe (glm5-dsa lane): same discipline as the KDA
 * pass counter, but counting Glm5NextLayerAttention calls on the DSA layer
 * named by SPARK_GLM5_NEXT_PROBE_VEC_LAYER (default 3 - the first DSA
 * layer). Armed by the same SPARK_GLM5_NEXT_PROBE_VEC env as the KDA probe
 * plus SPARK_GLM5_NEXT_PROBE_VEC_DSA=1, so the L0 KDA dump behaviour is
 * byte-identical when the new env is unset. Waves are single-row, so pass
 * p of the first request IS position p and the host oracle can rebuild the
 * latent cache from the per-pass kv_slot dumps. */

static int Glm5NextDsaProbeVecLayer(void)
{
    static int vec_layer = -1;
    if ( vec_layer < 0 )
    {
        const char *layer_env = getenv("SPARK_GLM5_NEXT_PROBE_VEC_LAYER");
        vec_layer = (layer_env != 0 && *layer_env != 0)
            ? (int)atoi(layer_env) : 3;
    }
    return(vec_layer);
}

static int Glm5NextDsaProbeVecPass(const Glm5NextLayerBuffers *buffers)
{
    static int vec_enabled = -1;
    static uint32_t vec_pass = 0u;
    uint32_t cap;
    if ( vec_enabled < 0 )
        vec_enabled = (getenv("SPARK_GLM5_NEXT_PROBE_VEC") != 0 &&
            getenv("SPARK_GLM5_NEXT_PROBE_VEC_DSA") != 0) ? 1 : 0;
    if ( vec_enabled == 0 || buffers == 0 || buffers->tp_rank != 0u ||
        (int)buffers->layer_index != Glm5NextDsaProbeVecLayer() )
        return(0);
    cap = 30u;
    {
        const char *cap_env = getenv("SPARK_GLM5_NEXT_PROBE_VEC_PASSES");
        if ( cap_env != 0 && *cap_env != 0 )
            cap = (uint32_t)atoi(cap_env);
    }
    if ( vec_pass >= cap )
        return(0);
    vec_pass += 1u;
    return((int)vec_pass);
}

static void Glm5NextProbeVecU16(cudaStream_t stream,const uint16_t *device,uint32_t count,uint32_t layer,uint32_t pass,const char *label)
{
    /* 16384: the HC streams surface (4 x 4096) must fit in one dump for the
     * DSA-site oracle's mix-dot recompute (glm5-dsa lane). The KDA path
     * never exceeds 8192, so widening the cap changes nothing there. */
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

static void Glm5NextProbeVecF32(cudaStream_t stream,const float *device,uint32_t count,uint32_t layer,uint32_t pass,const char *label)
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

static int32_t Glm5NextLayerAttention(
    const Glm5NextLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_index,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    const uint32_t *selected_positions;
    uint32_t selected_position_count;
    int32_t status;
    const int32_t vec_pass = (int32_t)Glm5NextDsaProbeVecPass(buffers);
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
        /* query_rope_bf16 stays null at ROPE_DIM == 0: the latent decode
         * template never dereferences it (compile-time guarantee). */
        (context > GLM5_NEXT_DSA_SELECTED &&
         (buffers->selected_positions == 0 ||
          buffers->selected_position_count != GLM5_NEXT_DSA_SELECTED)))
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    selected_positions = context > GLM5_NEXT_DSA_SELECTED
        ? buffers->selected_positions : 0;
    selected_position_count = context > GLM5_NEXT_DSA_SELECTED
        ? buffers->selected_position_count : 0u;

    if ( vec_pass != 0 )
    {
        /* HC-site stages of THIS attention site: the scratch still holds
         * this site's values (the MLP HC site runs later). The streams
         * surface (hidden_bf16) is the mix input; collapsed is the
         * sublayer input the norm below consumes. */
        vec_rank_heads = buffers->attn_heads;
        Glm5NextProbeVecU16(stream,buffers->hidden_bf16,
            GLM5_NEXT_HC * GLM5_NEXT_HIDDEN,layer_index,(uint32_t)vec_pass,"hc_streams");
        Glm5NextProbeVecU16(stream,buffers->hc_collapsed_bf16,
            GLM5_NEXT_HIDDEN,layer_index,(uint32_t)vec_pass,"hc_collapsed");
        Glm5NextProbeVecF32(stream,buffers->hc_mixes_f32,
            (2u + GLM5_NEXT_HC) * GLM5_NEXT_HC,layer_index,(uint32_t)vec_pass,"hc_mixes");
        Glm5NextProbeVecF32(stream,buffers->hc_pre_f32,
            GLM5_NEXT_HC,layer_index,(uint32_t)vec_pass,"hc_pre");
        Glm5NextProbeVecF32(stream,buffers->hc_post_f32,
            GLM5_NEXT_HC,layer_index,(uint32_t)vec_pass,"hc_post");
        Glm5NextProbeVecF32(stream,buffers->hc_comb_f32,
            GLM5_NEXT_HC * GLM5_NEXT_HC,layer_index,(uint32_t)vec_pass,"hc_comb");
    }

    /* PLAIN norm of the HC-collapsed input: the residual bookkeeping
     * belongs to the HC post step, so the sublayer never adds one. */
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS, uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        0,
        (const uint16_t *)buffers->attn_norm_weight,
        0,
        buffers->normed_bf16,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_RMS_EPSILON);
    if ( vec_pass != 0 )
        Glm5NextProbeVecU16(stream,buffers->normed_bf16,GLM5_NEXT_HIDDEN,
            layer_index,(uint32_t)vec_pass,"attn_normed");

    // The two low-rank norms are nonlinear and cannot be folded into a static
    // hidden-to-latent matrix. Run the checkpoint sequence exactly.
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->q_a_weight,
        buffers->q_compressed_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_QUERY_A_DIM,
        GLM5_NEXT_QUERY_A_DIM,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS,uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_QUERY_A_DIM + 8u) * sizeof(float),
        stream,
        buffers->q_compressed_bf16,
        0,
        (const uint16_t *)buffers->q_a_norm_weight,
        0,
        buffers->q_compressed_bf16,
        GLM5_NEXT_QUERY_A_DIM,
        GLM5_NEXT_QUERY_A_DIM,
        GLM5_NEXT_RMS_EPSILON);
    if ( vec_pass != 0 )
        Glm5NextProbeVecU16(stream,buffers->q_compressed_bf16,
            GLM5_NEXT_QUERY_A_DIM,layer_index,(uint32_t)vec_pass,"q_compressed");
    status = Glm5NextLayerIndexer(
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
    status = Glm5NextLaunchBf16Linear(
        buffers->q_compressed_bf16,
        buffers->q_b_weight,
        buffers->q_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_QUERY_A_DIM,
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
        Glm5NextProbeVecU16(stream,buffers->q_bf16,
            vec_rank_heads * GLM5_NEXT_QK_NOPE_DIM,layer_index,(uint32_t)vec_pass,"q");
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kv_a_weight,
        buffers->kv_slot_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_LATENT_ROW,
        GLM5_NEXT_LATENT_ROW,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS,uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_LATENT + 8u) * sizeof(float),
        stream,
        buffers->kv_slot_bf16,
        0,
        (const uint16_t *)buffers->kv_a_norm_weight,
        0,
        buffers->kv_slot_bf16,
        GLM5_NEXT_LATENT,
        GLM5_NEXT_LATENT_ROW,
        GLM5_NEXT_RMS_EPSILON);
    if ( vec_pass != 0 )
        Glm5NextProbeVecU16(stream,buffers->kv_slot_bf16,GLM5_NEXT_LATENT_ROW,
            layer_index,(uint32_t)vec_pass,"kv_slot");

    /* ROPE-0 MLA: the query carries nope only and the latent row is the
     * pure 512 lora, so both rope launches are GONE - not skipped at
     * runtime, absent at compile time (NoPE: no rope exists anywhere in
     * the glm5_next text stack). query_rope_bf16 stays null; the latent
     * attention template below reads it only when ROPE != 0. */
    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            GLM5_NEXT_LAYER_THREADS,
            GLM5_NEXT_QK_NOPE_DIM,
            GLM5_NEXT_LATENT,
            GLM5_NEXT_QK_NOPE_DIM + GLM5_NEXT_ROPE_DIM,
            0u>),
        dim3(rows, buffers->attn_heads),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        (const uint16_t *)buffers->kv_b_key_transposed_weight,
        buffers->query_latent_bf16,
        buffers->attn_heads,
        rows);
    if ( vec_pass != 0 )
        Glm5NextProbeVecU16(stream,buffers->query_latent_bf16,
            vec_rank_heads * GLM5_NEXT_LATENT,layer_index,(uint32_t)vec_pass,"query_latent");
    LM_LAUNCH(
        (LmKvStoreKernel<Glm5NextKv, GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->cache,
        buffers->kv_slot_bf16,
        buffers->sequence_of_row,
        buffers->positions,
        rows,
        GLM5_NEXT_LATENT_ROW);
    // R3 flash-decode: below the deployment threshold (or with no workspace,
    // or when the grid already fills the machine) this is the SAME
    // single-pass launch as before, byte for byte. Above it, the position
    // range splits across CTAs and a combine pass merges the partial
    // softmax states deterministically. position_bound is the host-side walk
    // bound: the selection width when the DSA selection is active, else the
    // context.
    if (LmLatentAttentionDecodeSplitLaunch<
            Glm5NextKv, GLM5_NEXT_ATTN_THREADS, GLM5_NEXT_LATENT,
            GLM5_NEXT_ROPE_DIM>(
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
            context > GLM5_NEXT_DSA_SELECTED ? GLM5_NEXT_DSA_SELECTED
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
        Glm5NextProbeVecU16(stream,buffers->attention_latent_bf16,
            vec_rank_heads * GLM5_NEXT_LATENT,layer_index,(uint32_t)vec_pass,"attn_latent");

    LM_LAUNCH(
        (LmPerHeadProjectKernel<
            GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_LATENT,GLM5_NEXT_VALUE_DIM>),
        dim3(rows, buffers->attn_heads),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->attention_latent_bf16,
        (const uint16_t *)buffers->kv_b_value_weight,
        buffers->attention_value_bf16,
        buffers->attn_heads,
        rows);
    if ( vec_pass != 0 )
        Glm5NextProbeVecU16(stream,buffers->attention_value_bf16,
            vec_rank_heads * GLM5_NEXT_VALUE_DIM,layer_index,(uint32_t)vec_pass,"attn_value");

    status = Glm5NextLaunchBf16Linear(
        buffers->attention_value_bf16,
        buffers->output_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->attn_output_columns,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if ( status == LM_LAUNCH_OK && vec_pass != 0 )
        Glm5NextProbeVecU16(stream,buffers->attention_out_bf16,GLM5_NEXT_HIDDEN,
            layer_index,(uint32_t)vec_pass,"attn_out_partial");
    return status;
}

/* Split the fused q|k|v|beta GEMM output into the four per-row buffers
 * (k3's pack-V2 split, verbatim: sections are row ranges head-major). */
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void Glm5NextSplitFusedProjectionsKernel(
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

/* Split the fused decay_down|gate_down bottleneck output: the first
 * GLM5_NEXT_KDA_LOW_RANK rows are the decay latent, the second half the
 * gate latent (pack-V2's replicated fusion, resurrected for glm5_next's
 * LOW-RANK output gate). */
template<uint32_t THREADS, uint32_t LOW_RANK>
__global__ __launch_bounds__(THREADS, 1)
void Glm5NextSplitDecayGateDownKernel(
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

/* THE DELTA RULE'S 64 KiB OPT-IN (runtime/launch.h's shared grant table;
 * k3 names its own instantiation the same way). */
static int32_t Glm5NextDeltaRuleOptIn(uint32_t shared_bytes)
{
    return(LmKernelSharedMemoryOptIn(
        (const void *)LmDeltaRuleKernel<GLM5_NEXT_LAYER_THREADS,
                                        GLM5_NEXT_KDA_KEY_DIM,
                                        GLM5_NEXT_KDA_VALUE_DIM>,
        shared_bytes));
}

/* -- G5N-PROBE (kda lane, diag only, env SPARK_GLM5_NEXT_PROBE) ---------------
 * Rank-0 checksums of the KDA stage buffers between launches. Every dump
 * synchronizes the stream and copies device->host: probe builds stall the
 * chain BY DESIGN and must never ship in a serving binary. */

#include <stdlib.h>
#include <stdio.h>

static int Glm5NextKdaProbeActive(const Glm5NextLayerBuffers *buffers)
{
    static int probe_enabled = -1;
    if ( probe_enabled < 0 )
        probe_enabled = getenv("SPARK_GLM5_NEXT_PROBE") != 0 ? 1 : 0;
    return(probe_enabled != 0 && buffers != 0 && buffers->tp_rank == 0u);
}

/* Deep-dive gate: healthy reference layer 0 + the first-zero neighbourhood
 * 16..20 + the last weight layer 44 before the head. */
static int Glm5NextKdaProbeDeep(const Glm5NextLayerBuffers *buffers)
{
    uint32_t layer = buffers != 0 ? buffers->layer_index : 0u;
    return(Glm5NextKdaProbeActive(buffers) &&
        (layer == 0u || (layer >= 16u && layer <= 20u) || layer >= 43u));
}

/* first8 raw u16 + their float values, to separate zeros from garbage from
 * huge from NaN - the bit-pattern sum cannot. LYR is the layer index for the
 * tag; dev is a device pointer. */
#define GLM5_NEXT_KDA_PROBE_RAW(stream,lyr,label,dev) \
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

/* Sum of the RAW uint16 bit patterns of the first count elements (the same
 * quantity the module-level probes print as bf16sum, so numbers compare). */
static uint64_t Glm5NextProbeBf16Sum(cudaStream_t stream,const uint16_t *device,uint32_t count)
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

static void Glm5NextProbeFloats(cudaStream_t stream,const float *device,uint32_t count,float *host)
{
    if ( cudaStreamSynchronize(stream) != cudaSuccess )
        return;
    (void)cudaMemcpy(host,device,count * sizeof(float),cudaMemcpyDeviceToHost);
}

/* Same, for a bf16 device buffer: decode to floats host-side. */
static void Glm5NextProbeBf16Floats(cudaStream_t stream,const uint16_t *device,uint32_t count,float *host)
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

#define GLM5_NEXT_KDA_PROBE(stream,label,dev,cnt) \
    do { fprintf(stderr,"G5N-PROBE kda L%u %s bf16sum %llu\n", \
        buffers->layer_index,(label), \
        (unsigned long long)Glm5NextProbeBf16Sum((stream),(const uint16_t *)(dev),(cnt))); } while (0)

#define GLM5_NEXT_KDA_PROBE_STATE(stream,label,pool) \
    do { \
        float probe_s[4]; uint32_t probe_w; uint64_t probe_bits = 0u; \
        uint32_t probe_words[1024]; \
        Glm5NextProbeFloats((stream),(const float *)(pool),4u,probe_s); \
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

/* -- G5N-VEC (glm5-attractor lane, diag only, env SPARK_GLM5_NEXT_PROBE_VEC) --
 * FULL-vector hex dumps of the LAYER-0 KDA stage buffers: the raw material
 * for the independent host oracle (a checkpoint-semantics reimplementation
 * of the KDA cell that shares no math with this module). Same discipline as
 * the checksum probes: rank 0 only, one stream sync per dump, never in a
 * serving binary. The pass id counts Glm5NextLayerKda calls on layer 0
 * (waves are single-row, so pass p of the first request IS position p). */

static int Glm5NextKdaProbeVecPass(const Glm5NextLayerBuffers *buffers)
{
    static int vec_enabled = -1;
    static uint32_t vec_pass = 0u;
    uint32_t cap;
    if ( vec_enabled < 0 )
        vec_enabled = getenv("SPARK_GLM5_NEXT_PROBE_VEC") != 0 ? 1 : 0;
    if ( vec_enabled == 0 || buffers == 0 || buffers->tp_rank != 0u ||
        buffers->layer_index != 0u )
        return(0);
    cap = 30u;
    {
        const char *cap_env = getenv("SPARK_GLM5_NEXT_PROBE_VEC_PASSES");
        if ( cap_env != 0 && *cap_env != 0 )
            cap = (uint32_t)atoi(cap_env);
    }
    if ( vec_pass >= cap )
        return(0);
    vec_pass += 1u;
    return((int)vec_pass);
}

/* KDA linear attention, 34 of 45 layers - k3's launch chain with the
 * glm5_next deltas:
 *
 *     q, k = L2Norm(Swish(ShortConv(W x)))        (checkpoint convs are BF16;
 *     v    = Swish(ShortConv(W x))                 LmScalarToFloat reads them)
 *     a    = exp(-5.0 * sigmoid(exp(A_log_h) * (Wf_up(Wf_down(x)) + dt_bias)))
 *     S    = (I - beta k k^T) Diag(a) S + beta k v^T,  beta = sigmoid(b_proj x)
 *     y    = W_o[ sigmoid(Wg_up(Wg_down(x))) * RMSNorm(S^T q) ]
 *
 * The decay mapping is LmBoundedDecay EXACTLY (verified against the
 * reference forget gate); the gated norm is the shared RMSNorm followed by
 * LmOutputGateKernel, which multiplies by sigmoid(gate) - RMSNormGated's
 * "sigmoid" activation, same order.
 */
static int32_t Glm5NextLayerKda(
    const Glm5NextLayerBuffers *buffers,
    uint32_t rows,
    uint32_t sequences,
    uint32_t commit,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    const uint32_t rank_heads = buffers->kda_heads;
    const uint32_t rank_qk = rank_heads * GLM5_NEXT_KDA_KEY_DIM;
    const uint32_t rank_v = rank_heads * GLM5_NEXT_KDA_VALUE_DIM;
    const int32_t vec_pass = (int32_t)Glm5NextKdaProbeVecPass(buffers);
    int32_t status;

    if (buffers == 0 || rows == 0u || sequences == 0u ||
        buffers->kda_state_pool == 0 ||
        buffers->kda_state_slot_bytes != GLM5_NEXT_KDA_STATE_BYTES_PER_LAYER ||
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
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS,uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        0,
        (const uint16_t *)buffers->attn_norm_weight,
        0,
        buffers->normed_bf16,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_RMS_EPSILON);

    if ( Glm5NextKdaProbeActive(buffers) )
    {
        GLM5_NEXT_KDA_PROBE(stream,"collapsed",buffers->hc_collapsed_bf16,256u);
        GLM5_NEXT_KDA_PROBE(stream,"normed",buffers->normed_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        Glm5NextProbeVecU16(stream,buffers->hc_collapsed_bf16,GLM5_NEXT_HIDDEN,0u,(uint32_t)vec_pass,"collapsed");
        Glm5NextProbeVecU16(stream,buffers->normed_bf16,GLM5_NEXT_HIDDEN,0u,(uint32_t)vec_pass,"normed");
    }
    if ( Glm5NextKdaProbeDeep(buffers) )
    {
        float probe_pre[4],probe_post[4],probe_comb[4];
        Glm5NextProbeFloats(stream,buffers->hc_pre_f32,4u,probe_pre);
        Glm5NextProbeFloats(stream,buffers->hc_post_f32,4u,probe_post);
        Glm5NextProbeFloats(stream,buffers->hc_comb_f32,4u,probe_comb);
        fprintf(stderr,"G5N-PROBE kda L%u hc pre %.6g %.6g %.6g %.6g post %.6g %.6g %.6g %.6g comb %.6g %.6g %.6g %.6g\n",
            buffers->layer_index,(double)probe_pre[0],(double)probe_pre[1],(double)probe_pre[2],(double)probe_pre[3],
            (double)probe_post[0],(double)probe_post[1],(double)probe_post[2],(double)probe_post[3],
            (double)probe_comb[0],(double)probe_comb[1],(double)probe_comb[2],(double)probe_comb[3]);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"collapsed",buffers->hc_collapsed_bf16);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"normed",buffers->normed_bf16);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"attn_norm_weight",buffers->attn_norm_weight);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"qkv_beta_weight_row0",buffers->kda_qkv_beta_weight);
    }
    /* TWO WIDE GEMMs over one activation read: the fused q|k|v|beta tensor
     * (OUTPUT_DIM_HEADS class) and the fused decay|gate-down bottleneck
     * (replicated). Both read normed_bf16 back to back. */
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kda_qkv_beta_weight,
        buffers->fused_qkvb_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
        rank_qk * 2u + rank_v + rank_heads,
        rank_qk * 2u + rank_v + rank_heads,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->kda_decay_gate_down_weight,
        buffers->fused_decay_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
        2u * GLM5_NEXT_KDA_LOW_RANK,
        2u * GLM5_NEXT_KDA_LOW_RANK,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    LM_LAUNCH(
        (Glm5NextSplitFusedProjectionsKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->fused_qkvb_bf16,
        buffers->q_bf16,
        buffers->kv_slot_bf16 /* key rows */,
        buffers->gate_up_bf16 /* value rows: rank_v wide */,
        buffers->kda_beta_logit,
        rows,
        rank_qk,
        rank_v,
        rank_heads,
        rank_qk * 2u + rank_v + rank_heads);
    LM_LAUNCH(
        (Glm5NextSplitDecayGateDownKernel<
            GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_LOW_RANK>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->fused_decay_gate_bf16,
        buffers->kda_decay_latent_bf16,
        buffers->kda_gate_latent_bf16,
        rows);

    if ( Glm5NextKdaProbeActive(buffers) )
    {
        GLM5_NEXT_KDA_PROBE(stream,"fused_qkvb",buffers->fused_qkvb_bf16,256u);
        GLM5_NEXT_KDA_PROBE(stream,"decay_latent",buffers->kda_decay_latent_bf16,64u);
        GLM5_NEXT_KDA_PROBE(stream,"gate_latent",buffers->kda_gate_latent_bf16,64u);
    }
    if ( vec_pass != 0 )
    {
        Glm5NextProbeVecU16(stream,buffers->fused_qkvb_bf16,rank_qk * 2u + rank_v + rank_heads,0u,(uint32_t)vec_pass,"fused_qkvb");
        Glm5NextProbeVecU16(stream,buffers->kda_decay_latent_bf16,GLM5_NEXT_KDA_LOW_RANK,0u,(uint32_t)vec_pass,"decay_latent");
        Glm5NextProbeVecU16(stream,buffers->kda_gate_latent_bf16,GLM5_NEXT_KDA_LOW_RANK,0u,(uint32_t)vec_pass,"gate_latent");
    }
    if ( Glm5NextKdaProbeDeep(buffers) )
    {
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"fused_qkvb",buffers->fused_qkvb_bf16);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"decay_latent",buffers->kda_decay_latent_bf16);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"gate_latent",buffers->kda_gate_latent_bf16);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"beta_logit",buffers->kda_beta_logit);
    }
    /* THREE CONVOLUTIONS, each with its own window; the weights are the
     * checkpoint's BF16 tensors unconverted. */
    LM_LAUNCH(
        (LmCausalConvKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_qk + GLM5_NEXT_LAYER_THREADS - 1u) / GLM5_NEXT_LAYER_THREADS),
        GLM5_NEXT_LAYER_THREADS,
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
        (LmCausalConvKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_qk + GLM5_NEXT_LAYER_THREADS - 1u) / GLM5_NEXT_LAYER_THREADS),
        GLM5_NEXT_LAYER_THREADS,
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
        (LmCausalConvKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_CONV_KERNEL,LM_CONV_SWISH,uint16_t>),
        dim3(sequences,(rank_v + GLM5_NEXT_LAYER_THREADS - 1u) / GLM5_NEXT_LAYER_THREADS),
        GLM5_NEXT_LAYER_THREADS,
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
    /* q and k only; the value is not normalised. */
    LM_LAUNCH(
        (LmL2NormalisePerHeadKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_KEY_DIM>),
        dim3(rows,rank_heads),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->q_bf16,
        rank_heads,
        rows,
        GLM5_NEXT_RMS_EPSILON);
    LM_LAUNCH(
        (LmL2NormalisePerHeadKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_KEY_DIM>),
        dim3(rows,rank_heads),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->kv_slot_bf16,
        rank_heads,
        rows,
        GLM5_NEXT_RMS_EPSILON);

    if ( Glm5NextKdaProbeActive(buffers) )
    {
        GLM5_NEXT_KDA_PROBE(stream,"q_postconv",buffers->q_bf16,256u);
        GLM5_NEXT_KDA_PROBE(stream,"k_postconv",buffers->kv_slot_bf16,256u);
        GLM5_NEXT_KDA_PROBE(stream,"v_postconv",buffers->gate_up_bf16,256u);
        GLM5_NEXT_KDA_PROBE(stream,"beta_logit",buffers->kda_beta_logit,64u);
    }
    if ( vec_pass != 0 )
    {
        Glm5NextProbeVecU16(stream,buffers->q_bf16,rank_qk,0u,(uint32_t)vec_pass,"q_postconv");
        Glm5NextProbeVecU16(stream,buffers->kv_slot_bf16,rank_qk,0u,(uint32_t)vec_pass,"k_postconv");
        Glm5NextProbeVecU16(stream,buffers->gate_up_bf16,rank_v,0u,(uint32_t)vec_pass,"v_postconv");
    }
    if ( Glm5NextKdaProbeDeep(buffers) )
    {
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"q_postconv",buffers->q_bf16);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"k_postconv",buffers->kv_slot_bf16);
        GLM5_NEXT_KDA_PROBE_RAW(stream,buffers->layer_index,"v_postconv",buffers->gate_up_bf16);
    }
    /* The two up-projections: decay logits (bounded-decay input) and the
     * gate (waits for the delta rule to finish). */
    status = Glm5NextLaunchBf16Linear(
        buffers->kda_decay_latent_bf16,
        buffers->kda_decay_up_weight,
        buffers->kda_decay_logit_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_KDA_LOW_RANK,
        rank_qk,
        rank_qk,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    status = Glm5NextLaunchBf16Linear(
        buffers->kda_gate_latent_bf16,
        buffers->kda_gate_up_weight,
        buffers->kda_gate_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_KDA_LOW_RANK,
        rank_v,
        rank_v,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
        return status;
    if ( Glm5NextKdaProbeActive(buffers) )
    {
        GLM5_NEXT_KDA_PROBE(stream,"decay_logit",buffers->kda_decay_logit_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        Glm5NextProbeVecU16(stream,buffers->kda_decay_logit_bf16,rank_qk,0u,(uint32_t)vec_pass,"decay_logit");
    }
    LM_LAUNCH(
        (LmBoundedDecayKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_KEY_DIM>),
        dim3(rows,rank_heads),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->kda_decay_logit_bf16,
        buffers->kda_decay_bias,
        buffers->kda_head_log_scale,
        buffers->kda_retention,
        rank_heads,
        GLM5_NEXT_KDA_GATE_LOWER_BOUND,
        rows);
    LM_LAUNCH(
        (LmSigmoidRowsKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->kda_beta_logit,
        buffers->kda_write_gate,
        rank_heads);
    if ( Glm5NextKdaProbeActive(buffers) )
    {
        float probe_r[4],probe_b[4];
        Glm5NextProbeFloats(stream,buffers->kda_retention,4u,probe_r);
        Glm5NextProbeFloats(stream,buffers->kda_write_gate,4u,probe_b);
        fprintf(stderr,"G5N-PROBE kda L%u retention %.6g %.6g %.6g %.6g write_gate %.6g %.6g %.6g %.6g\n",
            buffers->layer_index,(double)probe_r[0],(double)probe_r[1],(double)probe_r[2],
            (double)probe_r[3],(double)probe_b[0],(double)probe_b[1],(double)probe_b[2],(double)probe_b[3]);
        GLM5_NEXT_KDA_PROBE_STATE(stream,"state_pre",buffers->kda_state_pool);
    }
    if ( vec_pass != 0 )
    {
        Glm5NextProbeVecF32(stream,buffers->kda_retention,rank_qk,0u,(uint32_t)vec_pass,"retention");
        Glm5NextProbeVecF32(stream,buffers->kda_write_gate,rank_heads,0u,(uint32_t)vec_pass,"write_gate");
    }
    /* THE DELTA RULE'S 64 KiB OF DYNAMIC SHARED IS PAST THE 48 KiB DEFAULT
     * (k3's identical lesson): the launch fails with invalid argument on
     * device unless cudaFuncSetAttribute opts this instantiation in. */
    status = Glm5NextDeltaRuleOptIn(
        GLM5_NEXT_KDA_KEY_DIM * GLM5_NEXT_KDA_VALUE_DIM * sizeof(float));
    if (status != LM_LAUNCH_OK)
        return(status);
#ifdef GLM5_NEXT_KDA_DEBUG_LAUNCHES
    fprintf(stderr,"kda delta: grid(%u,%u) threads %u shared %u heads %u vhp %u seqs %u slot_bytes %u q=%p k=%p v=%p out=%p\n",
        sequences,rank_heads,GLM5_NEXT_LAYER_THREADS,
        (unsigned)(GLM5_NEXT_KDA_KEY_DIM * GLM5_NEXT_KDA_VALUE_DIM * sizeof(float)),
        rank_heads,1u,sequences,buffers->kda_state_slot_bytes,
        (void*)buffers->q_bf16,(void*)buffers->kv_slot_bf16,(void*)buffers->gate_up_bf16,
        (void*)buffers->attention_out_bf16);
#endif
    LM_LAUNCH(
        (LmDeltaRuleKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_KDA_KEY_DIM,GLM5_NEXT_KDA_VALUE_DIM>),
        dim3(sequences,rank_heads),
        GLM5_NEXT_LAYER_THREADS,
        GLM5_NEXT_KDA_KEY_DIM * GLM5_NEXT_KDA_VALUE_DIM * sizeof(float),
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
    if ( Glm5NextKdaProbeActive(buffers) )
    {
        GLM5_NEXT_KDA_PROBE(stream,"delta_out_raw",buffers->attention_out_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        Glm5NextProbeVecU16(stream,buffers->attention_out_bf16,rank_v,0u,(uint32_t)vec_pass,"delta_out");
    }
    /* RMSNorm before the gate (head-wise, fp32 strict), then the sigmoid
     * gate that has been waiting in kda_gate_bf16. */
#ifdef GLM5_NEXT_KDA_DEBUG_LAUNCHES
    fprintf(stderr,"kda norm: grid %llu threads %u shared %u dim %u\n",
        (unsigned long long)((uint64_t)rows * rank_heads),GLM5_NEXT_LAYER_THREADS,
        (unsigned)((GLM5_NEXT_KDA_VALUE_DIM + 8u) * sizeof(float)),
        GLM5_NEXT_KDA_VALUE_DIM);
#endif
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS,float>),
        dim3((uint64_t)rows * rank_heads),
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_KDA_VALUE_DIM + 8u) * sizeof(float),
        stream,
        buffers->attention_out_bf16,
        0,
        (const float *)buffers->kda_out_norm_weight,
        0,
        buffers->attention_out_bf16,
        GLM5_NEXT_KDA_VALUE_DIM,
        GLM5_NEXT_KDA_VALUE_DIM,
        GLM5_NEXT_RMS_EPSILON);
    LM_LAUNCH(
        (LmOutputGateKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->kda_gate_bf16,
        rank_v);
    if ( Glm5NextKdaProbeActive(buffers) )
    {
        GLM5_NEXT_KDA_PROBE(stream,"delta_out_gated",buffers->attention_out_bf16,256u);
        GLM5_NEXT_KDA_PROBE(stream,"kda_gate",buffers->kda_gate_bf16,256u);
    }
    if ( vec_pass != 0 )
    {
        Glm5NextProbeVecU16(stream,buffers->attention_out_bf16,rank_v,0u,(uint32_t)vec_pass,"delta_gated");
        Glm5NextProbeVecU16(stream,buffers->kda_gate_bf16,rank_v,0u,(uint32_t)vec_pass,"kda_gate");
    }
    /* Stage the gated y in kv_slot (the k scratch is dead once the delta
     * rule has consumed it) so the out-GEMM writes the FULL-WIDTH rank
     * partial straight into attention_out_bf16 - the buffer the chain
     * reduces, shared with the MLA path and the MLP finalize. */
    LM_LAUNCH(
        (LmCopyRowsKernel<GLM5_NEXT_LAYER_THREADS>),
        dim3((rank_v + GLM5_NEXT_LAYER_THREADS - 1u) / GLM5_NEXT_LAYER_THREADS,rows),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->kv_slot_bf16,
        rows,
        rank_v);
    status = Glm5NextLaunchBf16Linear(
        buffers->kv_slot_bf16,
        buffers->kda_out_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        rank_v,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if ( status == LM_LAUNCH_OK && Glm5NextKdaProbeActive(buffers) )
    {
        float probe_o[4];
        GLM5_NEXT_KDA_PROBE(stream,"kda_out_partial",buffers->attention_out_bf16,256u);
        Glm5NextProbeBf16Floats(stream,buffers->attention_out_bf16,4u,probe_o);
        fprintf(stderr,"G5N-PROBE kda L%u out_partial_f %.6g %.6g %.6g %.6g\n",
            buffers->layer_index,(double)probe_o[0],(double)probe_o[1],(double)probe_o[2],(double)probe_o[3]);
        GLM5_NEXT_KDA_PROBE_STATE(stream,"state_post",buffers->kda_state_pool);
    }
    if ( status == LM_LAUNCH_OK && vec_pass != 0 )
    {
        uint32_t vec_head;
        Glm5NextProbeVecU16(stream,buffers->attention_out_bf16,GLM5_NEXT_HIDDEN,0u,(uint32_t)vec_pass,"out_partial");
        for ( vec_head = 0u; vec_head < rank_heads; vec_head++ )
            Glm5NextProbeVecF32(stream,(const float *)buffers->kda_state_pool +
                ((uint64_t)vec_head * GLM5_NEXT_KDA_KEY_DIM * GLM5_NEXT_KDA_VALUE_DIM),
                GLM5_NEXT_KDA_KEY_DIM * GLM5_NEXT_KDA_VALUE_DIM,0u,(uint32_t)vec_pass,
                vec_head == 0u ? "state_h0" : (vec_head == 1u ? "state_h1" :
                (vec_head == 2u ? "state_h2" : "state_h3")));
    }
    return(status);
}

/* -- hyper-connections (dsv4 donor, glm5_next constants) ---------------------
 *
 * mHC per the reference: mixes are fn-dots over the UNWEIGHTED-RMSNorm'd
 * flattened streams; pre = sigmoid(w*s0 + b0) + eps; post = 2*sigmoid(...);
 * comb = softmax(+eps) then ONE column normalisation and 19 alternating
 * row/column pairs (sinkhorn 20) - dsv4's iteration structure exactly,
 * with the first row pass elided by the softmax. The final head collapse
 * is an UNWEIGHTED MEAN (dsv4 weights it), then the model norm.
 *
 * One thread per row for the split: hc is 4, comb fits in 16 registers. */
__global__ void Glm5NextHcSplitSinkhornKernel(
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

/* Mix projection: 24 fp32 dots of fn against the flattened streams,
 * scaled by the rsqrt of the flattened mean square (the reference's
 * UnweightedRMSNorm applied to the mix, exactly). The staging is TILED at
 * 4096 floats (16 KB shared) like dsv4's pattern: the flat row is 4 x
 * 4096 = 64 KB, over the default dynamic-shared limit, so a single-pass
 * stage would fail launch with invalid argument. The fn weights are F32
 * in the pack (the checkpoint stores BF16; the packer upcasts once). */
#define GLM5_NEXT_HC_MIX_TILE 4096u
__global__ void Glm5NextHcMixKernel(
    const uint16_t *__restrict__ streams_bf16,
    const float *__restrict__ fn_f32,
    float *__restrict__ mixes_f32,
    uint32_t row_count,
    uint32_t flat_dimension,
    uint32_t mix_rows,
    float rms_epsilon)
{
    extern __shared__ float staged[];
    __shared__ float reduction[GLM5_NEXT_LAYER_THREADS / LM_WARP_LANES];
    uint32_t row = blockIdx.x;
    uint32_t warp = threadIdx.x / LM_WARP_LANES;
    uint32_t lane = threadIdx.x % LM_WARP_LANES;
    uint32_t mix, element, tile, tile_end, tile_elements;
    const uint32_t warps = GLM5_NEXT_LAYER_THREADS / LM_WARP_LANES;
    float value, total = 0.0f, accumulator;
    float accum[4]; /* warps (8) >= mix rows per warp step of 3 used below */
    if (row >= row_count)
        return;
    /* Accumulate per-mix partials over tiles: mix_rows (24) exceeds the
     * warp count (8), so each warp carries ceil(24/8) = 3 mixes. */
    for (mix = 0u; mix < 3u; mix++)
        accum[mix] = 0.0f;
    for (tile = 0u; tile < flat_dimension; tile += GLM5_NEXT_HC_MIX_TILE)
    {
        tile_end = tile + GLM5_NEXT_HC_MIX_TILE < flat_dimension
            ? tile + GLM5_NEXT_HC_MIX_TILE
            : flat_dimension;
        tile_elements = tile_end - tile;
        __syncthreads();
        for (element = threadIdx.x; element < tile_elements;
             element += GLM5_NEXT_LAYER_THREADS)
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
    total = LmBlockSum<GLM5_NEXT_LAYER_THREADS>(total, reduction);
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

/* Stream collapse for the sublayer input, and the residual snapshot the
 * post step mixes back in: one block tile per (row, element range). */
__global__ void Glm5NextHcPreReduceKernel(
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

/* Sublayer output placement: streams_new[s] = post_s * out + sum_r
 * comb[r][s] * snapshot_r. */
__global__ void Glm5NextHcPostKernel(
    const uint16_t *__restrict__ out_bf16,
    const uint16_t *__restrict__ snapshot_bf16,
    const float *__restrict__ post_f32,
    const float *__restrict__ comb_f32,
    uint16_t *__restrict__ streams_bf16,
    uint32_t row_count,
    uint32_t hc,
    uint32_t dimension)
{
    __shared__ float post[4];
    __shared__ float comb[16];
    uint32_t row = blockIdx.x;
    uint32_t element, stream, source;
    float residual[4], out, value;
    if (row >= row_count || hc > 4u)
        return;
    if (threadIdx.x < hc)
        post[threadIdx.x] = post_f32[(uint64_t)row * hc + threadIdx.x];
    if (threadIdx.x < hc * hc)
        comb[threadIdx.x] = comb_f32[(uint64_t)row * hc * hc + threadIdx.x];
    __syncthreads();
    for (element = threadIdx.x; element < dimension; element += blockDim.x)
    {
        out = LmBf16ToFloat(out_bf16[(uint64_t)row * dimension + element]);
        for (source = 0u; source < hc; ++source)
            residual[source] = LmBf16ToFloat(
                snapshot_bf16[((uint64_t)row * hc + source) * dimension +
                              element]);
        for (stream = 0u; stream < hc; ++stream)
        {
            value = post[stream] * out;
            for (source = 0u; source < hc; ++source)
                value = __fmaf_rn(comb[source * hc + stream],
                                  residual[source], value);
            streams_bf16[((uint64_t)row * hc + stream) * dimension + element] =
                LmFloatToBf16(value);
        }
    }
}

/* Final head collapse: the UNWEIGHTED MEAN of the streams (dsv4 weights
 * its hc_head; glm5_next does not), ready for the model norm + lm_head. */
__global__ void Glm5NextHcHeadMeanKernel(
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

/* One attention site under HC: mix -> sinkhorn -> collapse -> sublayer ->
 * place. The sublayer (KDA or MLA) receives `collapsed` as its hidden
 * input and writes `sublayer_out`; every sublayer norm runs in the PLAIN
 * form (residual bookkeeping belongs to the HC post step, not to the
 * sublayer). */
static int32_t Glm5NextHcSite(
    const Glm5NextLayerBuffers *buffers,
    const void *fn_weight,
    const void *base_weight,
    const void *scale_weight,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    uint32_t mix_rows = GLM5_NEXT_HC_MIX;
    uint32_t flat = GLM5_NEXT_HC_FLAT;
    if (buffers->hc_mixes_f32 == 0 || buffers->hc_pre_f32 == 0 ||
        buffers->hc_post_f32 == 0 || buffers->hc_comb_f32 == 0 ||
        buffers->hc_collapsed_bf16 == 0 || buffers->hc_snapshot_bf16 == 0 ||
        fn_weight == 0 || base_weight == 0 || scale_weight == 0)
        return LM_LAUNCH_ERR_SHAPE;
    LM_LAUNCH(
        (Glm5NextHcMixKernel),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        GLM5_NEXT_HC_MIX_TILE * sizeof(float),
        stream,
        buffers->hidden_bf16 /* the streams surface */,
        (const float *)fn_weight,
        buffers->hc_mixes_f32,
        rows,
        flat,
        mix_rows,
        GLM5_NEXT_RMS_EPSILON);
    LM_LAUNCH(
        (Glm5NextHcSplitSinkhornKernel),
        (rows + 63u) / 64u,
        64u,
        0,
        stream,
        buffers->hc_mixes_f32,
        (const float *)scale_weight,
        (const float *)base_weight,
        rows,
        GLM5_NEXT_HC,
        GLM5_NEXT_HC_SINKHORN_ITERATIONS,
        GLM5_NEXT_HC_EPSILON,
        buffers->hc_pre_f32,
        buffers->hc_post_f32,
        buffers->hc_comb_f32);
    LM_LAUNCH(
        (Glm5NextHcPreReduceKernel),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->hidden_bf16,
        buffers->hc_pre_f32,
        buffers->hc_collapsed_bf16,
        buffers->hc_snapshot_bf16,
        rows,
        GLM5_NEXT_HC,
        GLM5_NEXT_HIDDEN);
    return LM_LAUNCH_OK;
}

static int32_t Glm5NextHcPost(
    const Glm5NextLayerBuffers *buffers,
    const uint16_t *sublayer_out,
    uint32_t rows,
    cudaStream_t stream)
{
    LM_LAUNCH(
        (Glm5NextHcPostKernel),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        sublayer_out,
        buffers->hc_snapshot_bf16,
        buffers->hc_post_f32,
        buffers->hc_comb_f32,
        buffers->hidden_bf16,
        rows,
        GLM5_NEXT_HC,
        GLM5_NEXT_HIDDEN);
    return LM_LAUNCH_OK;
}

/* SiLU-mul with the reference's swiglu_limit clamp (glm5-dsa lane; the
 * shared LmSiluMulKernel clamps nothing and every other family runs it
 * unclamped, so this stays module-local - the dsv4 donor precedent,
 * SparkDsv4SwigluClampKernel, is module-local the same way). The
 * checkpoint reference: gate.clamp(max=limit), up.clamp(-limit, limit),
 * then silu(gate)*up. Inert whenever |gate|,|up| < 10 (fixture traffic
 * measured max gate 2.59, max |up| 2.27 at L0); live at depth. */
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void Glm5NextSwigluLimitKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first)
{
    uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
    uint64_t out_base = (uint64_t)blockIdx.x * dimension;
    uint32_t index;
    for (index = threadIdx.x; index < dimension; index += THREADS)
    {
        float gate = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? index : dimension + index)]);
        float up = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? dimension + index : index)]);
        gate = gate > GLM5_NEXT_SWIGLU_LIMIT ? GLM5_NEXT_SWIGLU_LIMIT : gate;
        up = up > GLM5_NEXT_SWIGLU_LIMIT ? GLM5_NEXT_SWIGLU_LIMIT
            : (up < -GLM5_NEXT_SWIGLU_LIMIT ? -GLM5_NEXT_SWIGLU_LIMIT : up);
        output_bf16[out_base + index] =
            LmFloatToBf16((gate / (1.0f + __expf(-gate))) * up);
    }
}

static int32_t Glm5NextLayerDenseMlp(
    const Glm5NextLayerBuffers *buffers,
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

    /* PLAIN norm of the HC-collapsed input (see the attention note). */
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS, uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        0,
        (const uint16_t *)buffers->mlp_norm_weight,
        0,
        buffers->normed_bf16,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_RMS_EPSILON);

    if (buffers->dense_gate_up_fused != 0u)
    {
        // The pack stores the dense gate-up stack as [up | gate] (matching the
        // routed-expert convention), so one GEMM over the concatenated tensor
        // writes that [up | gate] layout directly and LmSiluMulKernel runs with
        // gate_first=false - the two-launch form below re-reads the normed
        // activation and pays a second launch for the same weight bytes.
        // Per-element math is identical either way; only the launch count differs.
        status = Glm5NextLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            GLM5_NEXT_HIDDEN,
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
        status = Glm5NextLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_gate_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            GLM5_NEXT_HIDDEN,
            buffers->dense_intermediate,
            buffers->dense_gate_up_rows,
            0u,
            multiprocessors,
            stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
        status = Glm5NextLaunchBf16Linear(
            buffers->normed_bf16,
            buffers->dense_up_weight,
            buffers->gate_up_bf16,
            buffers->dense_row_offset,
            buffers->dense_tile_prefix,
            rows,
            GLM5_NEXT_HIDDEN,
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
        (Glm5NextSwigluLimitKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->dense_intermediate,
        false);

    return Glm5NextLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->dense_down_weight,
        buffers->attention_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->dense_intermediate,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        0u,
        multiprocessors,
        stream);
}

template<uint32_t ExpertCodec>
static int32_t Glm5NextLayerMoe(
    const Glm5NextLayerBuffers *buffers,
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
    static_assert(GLM5_NEXT_HIDDEN % ExpertFormat::kScaleGroup == 0u &&
        GLM5_NEXT_EXPERT_INTERMEDIATE % ExpertFormat::kScaleGroup == 0u,
        "GLM 5.2 expert dimensions must contain complete codec scale groups");

    if (buffers == 0 || rows == 0u ||
        packed_rows != rows * GLM5_NEXT_TOP_K ||
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
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS, uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_collapsed_bf16,
        0,
        (const uint16_t *)buffers->mlp_norm_weight,
        0,
        buffers->normed_bf16,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_RMS_EPSILON);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = buffers->dense_row_offset;
    gemm.group_tile_prefix = buffers->dense_tile_prefix;
    gemm.output_f32 = buffers->router_logits;
    status = LmGemmLaunch<
        LmBf16Format,
        GLM5_NEXT_LAYER_TILE_N,
        LmBf16Format::kTileK,
        GLM5_NEXT_LAYER_STAGES,
        GLM5_NEXT_LAYER_WARPS>(
            &gemm,
            buffers->normed_bf16,
            buffers->router_weight,
            rows,
            rows,
            1u,
            1u,
            GLM5_NEXT_HIDDEN,
            GLM5_NEXT_EXPERTS,
            multiprocessors,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmTopkSmallKernel<
            GLM5_NEXT_LAYER_THREADS,
            GLM5_NEXT_TOP_K,
            true,
            1u,
            1u,
            LM_TOPK_SCORE_SIGMOID>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),
        stream,
        buffers->router_logits,
        GLM5_NEXT_EXPERTS,
        buffers->route_expert,
        buffers->route_weight,
        buffers->router_correction_bias,
        0,
        GLM5_NEXT_ROUTED_SCALE);
    status = LmRouteBuild<GLM5_NEXT_LAYER_THREADS, GLM5_NEXT_EXPERTS>(
        buffers->route_expert,
        rows,
        packed_rows,
        GLM5_NEXT_TOP_K,
        buffers->group_row_offset,
        buffers->route_packed_row,
        buffers->route_source_token,
        buffers->expert_w1_rows,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_LAYER_TILE_N,
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
        GLM5_NEXT_EXPERTS,
        buffers->expert_w1_rows,
        GLM5_NEXT_HIDDEN);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
	gemm.source_row_map = buffers->route_source_token;
	gemm.source_row_count = rows;
    gemm.output_bf16 = buffers->gate_up_bf16;
    status = LmGemmWeightOnlyIndirectLaunch<
        ExpertFormat,
        GLM5_NEXT_LAYER_TILE_N,
        GLM5_NEXT_LAYER_STAGES,
        GLM5_NEXT_LAYER_WARPS>(
            &gemm,
			buffers->normed_bf16,
            buffers->expert_w1_weight,
            packed_rows,
            rows,
            GLM5_NEXT_TOP_K,
            GLM5_NEXT_EXPERTS,
            GLM5_NEXT_HIDDEN,
            buffers->expert_w1_rows,
            multiprocessors,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (Glm5NextSwigluLimitKernel<GLM5_NEXT_LAYER_THREADS>),
        packed_rows,
        GLM5_NEXT_LAYER_THREADS,
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
        GLM5_NEXT_EXPERTS,
        GLM5_NEXT_HIDDEN,
        buffers->expert_intermediate);
    gemm.prefix_built = 1u;
    gemm.group_row_offset = buffers->group_row_offset;
    gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
    gemm.output_bf16 = buffers->expert_out_bf16;
    status = LmGemmWeightOnlyLaunch<
        ExpertFormat,
        GLM5_NEXT_LAYER_TILE_N,
        GLM5_NEXT_LAYER_STAGES,
        GLM5_NEXT_LAYER_WARPS>(
            &gemm,
            buffers->intermediate_bf16,
            buffers->expert_w2_weight,
            packed_rows,
            rows,
            GLM5_NEXT_TOP_K,
            GLM5_NEXT_EXPERTS,
            buffers->expert_intermediate,
            GLM5_NEXT_HIDDEN,
            multiprocessors,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmMoeFinalizeKernel<GLM5_NEXT_LAYER_THREADS>),
        dim3(
            (GLM5_NEXT_HIDDEN + GLM5_NEXT_LAYER_THREADS - 1u) /
                GLM5_NEXT_LAYER_THREADS,
            rows),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->expert_out_bf16,
        buffers->route_packed_row,
        buffers->route_weight,
        buffers->attention_out_bf16,
        rows,
        GLM5_NEXT_TOP_K,
        GLM5_NEXT_HIDDEN);
    status = Glm5NextLaunchBf16Linear(
        buffers->normed_bf16,
        buffers->shared_gate_up_weight,
        buffers->gate_up_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        GLM5_NEXT_HIDDEN,
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
        (Glm5NextSwigluLimitKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->gate_up_bf16,
        buffers->intermediate_bf16,
        buffers->shared_intermediate,
        false);
    status = Glm5NextLaunchBf16Linear(
        buffers->intermediate_bf16,
        buffers->shared_down_weight,
        buffers->shared_out_bf16,
        buffers->dense_row_offset,
        buffers->dense_tile_prefix,
        rows,
        buffers->shared_intermediate,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        0u,
        multiprocessors,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    /* THE MOE PARTIAL LANDS IN attention_out_bf16 - the buffer the chain
     * reduces - as the routed sum, then the replicated-compute shared
     * expert's rank partial adds IN PLACE. The finalize previously wrote
     * the routed sum into hidden_bf16 (the HC streams surface) and the
     * add overwrote it with attention_out + shared_out: the routed experts
     * never reached the residual, and REDUCE_MLP summed sixteen identical
     * copies of the already-reduced attention output - every MoE layer
     * placed 16x the attention sublayer output as its "FFN" contribution
     * (receipt: second r0 post == 16.000000x first r0 post on L42/L44).
     * Rank-invariant, TP1-invisible (the reduce no-ops), and the TP1
     * oracle mirrored the chain, so every gate passed. The dense tail
     * (layers 0-2) always wrote attention_out directly and was correct. */
    LM_LAUNCH(
        (LmAddRowsKernel<GLM5_NEXT_LAYER_THREADS>),
        dim3(
            (GLM5_NEXT_HIDDEN + GLM5_NEXT_LAYER_THREADS - 1u) /
                GLM5_NEXT_LAYER_THREADS,
            rows),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->attention_out_bf16,
        buffers->shared_out_bf16,
        buffers->attention_out_bf16,
        rows,
        GLM5_NEXT_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t Glm5NextHead(
    const Glm5NextLayerBuffers *buffers,
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

    tiles = (vocabulary + GLM5_NEXT_HEAD_TILE - 1u) / GLM5_NEXT_HEAD_TILE;
    /* glm5_next: the layer loop leaves the HC STREAMS in hidden_bf16; the
     * head collapse is the UNWEIGHTED MEAN (hc_mean_bf16, filled by
     * Glm5NextHcHeadMeanKernel in the wave runner) followed by this plain
     * norm - residual_bf16 carries nothing at this point. */
    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<GLM5_NEXT_LAYER_THREADS, uint16_t>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
        (GLM5_NEXT_HIDDEN + 8u) * sizeof(float),
        stream,
        buffers->hc_mean_bf16,
        0,
        (const uint16_t *)head_norm_weight,
        0,
        buffers->normed_bf16,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_HIDDEN,
        GLM5_NEXT_RMS_EPSILON);
    LM_LAUNCH(
        (LmHeadCandidateKernel<GLM5_NEXT_LAYER_THREADS, GLM5_NEXT_HEAD_TILE>),
        dim3(tiles, rows),
        GLM5_NEXT_LAYER_THREADS,
        0,
        stream,
        buffers->normed_bf16,
        (const uint16_t *)head_weight,
        token_ids,
        buffers->head_candidate_score,
        buffers->head_candidate_token,
        rows,
        GLM5_NEXT_HIDDEN,
        vocabulary);
    LM_LAUNCH(
        (LmHeadCommitKernel<GLM5_NEXT_LAYER_THREADS>),
        rows,
        GLM5_NEXT_LAYER_THREADS,
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
