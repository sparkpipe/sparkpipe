// Run the real GLM 5.2 layer on a CPU: one attention, one MoE, one dense MLP
// in both gate/up forms and the head - the code that ships, end to end, two
// rows.
//
// The glm5_2 driver had no harness at all: the per-kernel tests cover kernels,
// and the python contracts read the launch sites, but nothing EXECUTED the
// layer. That is how the join-assembled KV slot and the head's dropped
// residual could sit in the tree unobserved - both live between the kernels,
// which is exactly the span only a layer harness sees. k3_slice_host found six
// defects in that span for K3; this is the same instrument for GLM.
//
// The GEMM is the recorder - output 0.125 * call index, constant across rows,
// honouring the output row stride and column offset - so the whole stream is
// closed-form arithmetic the checker recomputes: the norm chain, the rope
// rotations, the slot layout written directly by the kv projections, the
// attention softmax over four positions, the indirect routed-row read, and
// the final norm over hidden + residual. Everything else is the shipping
// kernel.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/tile.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "runtime/gemm.cuh"
std::vector<LmRecordedGemm> lm_recorded_gemms;

// kv.cuh guards its store kernel on __CUDACC__; scope the macro to that one
// include exactly as k3_slice_host.cu does, so dtype.cuh never sees it.
#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__

#include "modules/glm52_resident_decode_stage/source/cuda/layer.cuh"

#ifndef EXPERT_CODEC
#error "EXPERT_CODEC must name the exact GLM 5.2 routed-expert codec"
#endif

#define ROWS 2u
#define PACKED (ROWS * GLM52_TOP_K)
#define CONTEXT 4u
#define POSITION 3u
#define SEQUENCES 2u
#define HEAD_VOCAB 128u
#define QK_SCALE 0.05f

// The dead outputs of the retired join path stay null on purpose: a regression
// that routes the kv projections through them again crashes the recorder here
// instead of silently re-adding the launch on hardware.
static uint16_t hidden[ROWS * GLM52_HIDDEN];
static uint16_t residual[ROWS * GLM52_HIDDEN];
static uint16_t normed[ROWS * GLM52_HIDDEN];
static uint16_t q_compressed[ROWS * GLM52_QUERY_A_DIM];
static uint16_t q_b[ROWS * GLM52_ATTN_HEADS *
    (GLM52_QK_NOPE_DIM + GLM52_ROPE_DIM)];
static uint16_t query_latent[ROWS * GLM52_ATTN_HEADS * GLM52_LATENT];
static uint16_t query_rope[ROWS * GLM52_ATTN_HEADS * GLM52_ROPE_DIM];
static uint16_t kv_slot[ROWS * GLM52_LATENT_ROW];
static uint16_t attention_latent[ROWS * GLM52_ATTN_HEADS * GLM52_LATENT];
static uint16_t attention_value[ROWS * GLM52_ATTN_HEADS * GLM52_VALUE_DIM];
static uint16_t attention_out[ROWS * GLM52_HIDDEN];
static uint16_t gate_up[PACKED * GLM52_GATE_UP_DIM];
static uint16_t intermediate[PACKED * GLM52_EXPERT_INTERMEDIATE];
static uint16_t expert_out[PACKED * GLM52_HIDDEN];
static uint16_t shared_out[ROWS * GLM52_HIDDEN];
static float router_logits[ROWS * GLM52_EXPERTS];
static uint32_t route_expert[PACKED];
static float route_weight[PACKED];
static uint32_t route_source_token[PACKED];
static uint32_t route_packed_row[PACKED];
static float head_candidate_score[ROWS];
static uint32_t head_candidate_token[ROWS];
static uint32_t output_token[ROWS];
static float output_score[ROWS];
static uint32_t group_row_offset[GLM52_EXPERTS + 1u];
static uint32_t group_tile_prefix_w1[GLM52_EXPERTS + 1u];
static uint32_t group_tile_prefix_w2[GLM52_EXPERTS + 1u];
static uint32_t dense_row_offset[ROWS + 1u];
static uint32_t dense_tile_prefix[ROWS + 1u];

static uint8_t kv_pool[SEQUENCES * Glm52Kv::kPageBytes];
static uint32_t page_table[SEQUENCES];
static uint32_t sequence_of_row[ROWS];
static uint32_t context_length[SEQUENCES];
static uint32_t positions[ROWS];
static LmKvAccessError kv_access_error;

static uint16_t attn_norm_w[GLM52_HIDDEN];
static uint16_t q_a_norm_w[GLM52_QUERY_A_DIM];
static uint16_t kv_a_norm_w[GLM52_LATENT];
static uint16_t mlp_norm_w[GLM52_HIDDEN];
static uint16_t head_norm_w[GLM52_HIDDEN];
static uint16_t head_weight[HEAD_VOCAB * GLM52_HIDDEN];

// Distinct addresses, never read: the recorder logs the pointer, so each
// weight must be its own buffer for the gemm log to name it.
static uint16_t w_q_a[8], w_q_b[8], w_kv_a[8];
static uint16_t w_kv_b_key[GLM52_ATTN_HEADS * GLM52_LATENT * GLM52_QK_NOPE_DIM];
static uint16_t w_kv_b_value[GLM52_ATTN_HEADS * GLM52_VALUE_DIM * GLM52_LATENT];
static uint16_t w_o[8], w_router[8], w_down[8], w_e1[8], w_e2[8];
static uint16_t w_shared_gate_up[8], w_shared_down[8];
static float s_e1[8], s_e2[8];
static float router_correction_bias[GLM52_EXPERTS];

static uint32_t seed = 13579u;
static float NextRandom(void)
{
    seed = (seed * 1664525u) + 1013904223u;
    return (float)((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static void FillRandom(uint16_t *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        values[index] = LmFloatToBf16(NextRandom());
}

static void FillNormWeight(uint16_t *values, uint32_t count)
{
    uint32_t index;
    for (index = 0u; index < count; ++index)
        values[index] = LmFloatToBf16(1.0f + 0.2f * NextRandom());
}

static void Emit(const char *tag, const uint16_t *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        printf("%s %.9g\n", tag, (double)LmBf16ToFloat(values[index]));
}

static void EmitSample(const char *tag, const uint16_t *values, uint64_t rows,
    uint64_t row_width, uint64_t stride)
{
    uint64_t row, index;
    for (row = 0u; row < rows; ++row)
        for (index = 0u; index < row_width; index += stride)
            printf("%s %.9g\n", tag,
                (double)LmBf16ToFloat(values[(row * row_width) + index]));
}

static void EmitU32(const char *tag, const uint32_t *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        printf("%s %u\n", tag, values[index]);
}

static void EmitF32(const char *tag, const float *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        printf("%s %.9g\n", tag, (double)values[index]);
}

static const char *PointerName(const void *pointer)
{
    if (pointer == normed) return "normed";
    if (pointer == q_compressed) return "q_compressed";
    if (pointer == q_b) return "q_b";
    if (pointer == query_latent) return "query_latent";
    if (pointer == query_rope) return "query_rope";
    if (pointer == kv_slot) return "kv_slot";
    if (pointer == attention_latent) return "attention_latent";
    if (pointer == attention_value) return "attention_value";
    if (pointer == attention_out) return "attention_out";
    if (pointer == gate_up) return "gate_up";
    if (pointer == intermediate) return "intermediate";
    if (pointer == expert_out) return "expert_out";
    if (pointer == shared_out) return "shared_out";
    if (pointer == hidden) return "hidden";
    if (pointer == router_logits) return "router_logits";
    if (pointer == w_q_a) return "w_q_a";
    if (pointer == w_q_b) return "w_q_b";
    if (pointer == w_kv_a) return "w_kv_a";
    if (pointer == w_o) return "w_o";
    if (pointer == w_router) return "w_router";
    if (pointer == w_down) return "w_down";
    if (pointer == w_e1) return "w_e1";
    if (pointer == w_e2) return "w_e2";
    if (pointer == w_shared_gate_up) return "w_shared_gate_up";
    if (pointer == w_shared_down) return "w_shared_down";
    return "unknown";
}

static void EmitGemmLog(uint32_t begin, const char *phase)
{
    uint32_t index;
    for (index = begin; index < lm_recorded_gemms.size(); ++index)
    {
        const LmRecordedGemm *record = &lm_recorded_gemms[index];
        printf("gemm %s %u K %u N %u rows %u grouped %u indirect %u act %s w %s dst %s\n",
            phase, index + 1u, record->input_dimension,
            record->output_dimension, record->packed_rows,
            record->grouped ? 1u : 0u, record->indirect ? 1u : 0u,
            PointerName(record->activation),
            PointerName(record->weight), PointerName(record->output));
    }
}

static uint32_t CountPoison(const uint16_t *values, uint64_t count, uint16_t poison)
{
    uint64_t index;
    uint32_t found = 0u;
    for (index = 0u; index < count; ++index)
        if (values[index] == poison)
            ++found;
    return found;
}

int main(void)
{
    Glm52LayerBuffers buffers;
    uint32_t row, index;
    int32_t status;
    uint16_t poison;

    memset(&buffers, 0, sizeof(buffers));
    LmKvAccessErrorReset(&kv_access_error);
    FillRandom(hidden, ROWS * GLM52_HIDDEN);
    FillRandom(residual, ROWS * GLM52_HIDDEN);
    FillNormWeight(attn_norm_w, GLM52_HIDDEN);
    for (index = 0u; index < GLM52_QUERY_A_DIM; ++index)
        q_a_norm_w[index] = LmFloatToBf16(1.0f);
    for (index = 0u; index < GLM52_LATENT; ++index)
        kv_a_norm_w[index] = LmFloatToBf16(1.0f);
    FillNormWeight(mlp_norm_w, GLM52_HIDDEN);
    FillNormWeight(head_norm_w, GLM52_HIDDEN);
    FillRandom(head_weight, (uint64_t)HEAD_VOCAB * GLM52_HIDDEN);

    // Three cached positions per sequence, filled before the step; the step
    // stores position 3 and attends over all four.
    for (row = 0u; row < SEQUENCES; ++row)
    {
        uint32_t position, element;
        page_table[row] = row;
        context_length[row] = CONTEXT;
        for (position = 0u; position < POSITION; ++position)
        {
            uint16_t *slot = (uint16_t *)(kv_pool +
                (row * Glm52Kv::kPageBytes) +
                (position * Glm52Kv::kSlotBytes));
            for (element = 0u; element < GLM52_LATENT_ROW; ++element)
                slot[element] = LmFloatToBf16(NextRandom());
        }
    }
    for (row = 0u; row < ROWS; ++row)
    {
        sequence_of_row[row] = row;
        positions[row] = POSITION;
    }
    dense_row_offset[0] = 0u;
    dense_row_offset[1] = ROWS;
    dense_row_offset[2] = ROWS;

    buffers.dense_row_offset = dense_row_offset;
    buffers.dense_tile_prefix = dense_tile_prefix;
    buffers.attn_norm_weight = attn_norm_w;
    buffers.q_a_weight = w_q_a;
    buffers.q_a_norm_weight = q_a_norm_w;
    buffers.q_b_weight = w_q_b;
    buffers.kv_a_weight = w_kv_a;
    buffers.kv_a_norm_weight = kv_a_norm_w;
    buffers.kv_b_key_transposed_weight = w_kv_b_key;
    buffers.kv_b_value_weight = w_kv_b_value;
    buffers.qk_scale = QK_SCALE;
    buffers.output_weight = w_o;
    buffers.mlp_norm_weight = mlp_norm_w;
    buffers.router_weight = w_router;
    buffers.router_correction_bias = router_correction_bias;
    buffers.expert_w1_weight = w_e1;
    buffers.expert_w1_scale = s_e1;
    buffers.expert_w2_weight = w_e2;
    buffers.expert_w2_scale = s_e2;
    buffers.shared_gate_up_weight = w_shared_gate_up;
    buffers.shared_down_weight = w_shared_down;
    buffers.hidden_bf16 = hidden;
    buffers.residual_bf16 = residual;
    buffers.normed_bf16 = normed;
    buffers.q_compressed_bf16 = q_compressed;
    buffers.q_bf16 = q_b;
    buffers.query_latent_bf16 = query_latent;
    buffers.query_rope_bf16 = query_rope;
    buffers.kv_slot_bf16 = kv_slot;
    buffers.attention_latent_bf16 = attention_latent;
    buffers.attention_value_bf16 = attention_value;
    buffers.attention_out_bf16 = attention_out;
    buffers.gate_up_bf16 = gate_up;
    buffers.intermediate_bf16 = intermediate;
    buffers.expert_out_bf16 = expert_out;
    buffers.shared_out_bf16 = shared_out;
    buffers.router_logits = router_logits;
    buffers.route_expert = route_expert;
    buffers.route_weight = route_weight;
    buffers.route_source_token = route_source_token;
    buffers.route_packed_row = route_packed_row;
    buffers.head_candidate_score = head_candidate_score;
    buffers.head_candidate_token = head_candidate_token;
    buffers.output_token = output_token;
    buffers.output_score = output_score;
    buffers.group_row_offset = group_row_offset;
    buffers.group_tile_prefix_w1 = group_tile_prefix_w1;
    buffers.group_tile_prefix_w2 = group_tile_prefix_w2;
    buffers.cache.pool = kv_pool;
    buffers.cache.page_table = page_table;
    buffers.cache.page_table_stride = 1u;
    buffers.cache.sequence_count = SEQUENCES;
    buffers.cache.pool_page_count = SEQUENCES;
    buffers.cache.access_error = &kv_access_error;
    buffers.sequence_of_row = sequence_of_row;
    buffers.context_length = context_length;
    buffers.positions = positions;
    buffers.row_positions = 0;

    printf("qkscale %.9g\n", (double)QK_SCALE);
    printf("theta %.9g\n", (double)GLM52_ROPE_THETA);
    printf("routedscale %.9g\n", (double)GLM52_ROUTED_SCALE);
    printf("eps %.9g\n", (double)GLM52_RMS_EPSILON);

    status = Glm52LayerAttention(&buffers, ROWS, CONTEXT, 3u, 48u, 0);
    printf("status attention %d\n", (int)status);
    EmitGemmLog(0u, "attention");
    Emit("normed1", normed, ROWS * GLM52_HIDDEN);
    Emit("kvslot", kv_slot, ROWS * GLM52_LATENT_ROW);
    for (row = 0u; row < SEQUENCES; ++row)
    {
        const uint16_t *slot = (const uint16_t *)(kv_pool +
            (row * Glm52Kv::kPageBytes) + (POSITION * Glm52Kv::kSlotBytes));
        Emit("slot", slot, GLM52_LATENT_ROW);
    }
    // The layer's attention decode runs at GLM52_ATTN_THREADS, which the shim
    // floors at 64: on the host it writes only every sixty-fourth output
    // element from a partially-staged query. The recorder erases that output
    // before anything reads it, so the harness checks the kernel's math at a
    // small geometry below instead, where one thread covers everything.
    {
        using TestKv = LmKvLatent<16u, 8u, 8u, 64u>;
        static uint8_t test_pool[TestKv::kPageBytes];
        static uint32_t test_pages[1];
        static uint16_t test_query_latent[2 * 8];
        static uint16_t test_query_rope[2 * 8];
        static uint16_t test_out[2 * 8];
        static uint32_t test_sequence[1];
        static uint32_t test_context[1];
        LmKvView test_cache;
        LmKvAccessError test_access_error;
        uint32_t head, position, element;

        test_pages[0] = 0u;
        test_sequence[0] = 0u;
        test_context[0] = 3u;
        LmKvAccessErrorReset(&test_access_error);
        test_cache.pool = test_pool;
        test_cache.page_table = test_pages;
        test_cache.page_table_stride = 1u;
        test_cache.sequence_count = 1u;
        test_cache.pool_page_count = 1u;
        test_cache.access_error = &test_access_error;
        for (position = 0u; position < 3u; ++position)
            for (element = 0u; element < 16u; ++element)
                ((uint16_t *)(test_pool + (position * TestKv::kSlotBytes)))
                    [element] = LmFloatToBf16(
                        0.1f * (float)(position + 1u) +
                        0.01f * (float)element);
        for (head = 0u; head < 2u; ++head)
            for (element = 0u; element < 8u; ++element)
            {
                test_query_latent[(head * 8u) + element] = LmFloatToBf16(
                    0.3f - (0.02f * (float)element));
                test_query_rope[(head * 8u) + element] = LmFloatToBf16(
                    -0.2f + (0.03f * (float)element) +
                    (0.05f * (float)head));
            }
        LM_HOST_LAUNCH(
            dim3(1u, 2u),
            (LmLatentAttentionDecodeKernel<TestKv, 1u, 8u, 8u>(
                test_query_latent, test_query_rope, test_cache,
                test_sequence, test_context, 0, 0u, 2u, 0.5f, test_out, 0)));
        Emit("smallattn", test_out, 2u * 8u);
    }

    status = Glm52LayerMoe<EXPERT_CODEC>(
        &buffers, ROWS, PACKED, 48u, 0);
    printf("status moe %d\n", (int)status);
    EmitGemmLog(4u, "moe");
    Emit("normed2", normed, ROWS * GLM52_HIDDEN);
    EmitU32("routeexpert", route_expert, PACKED);
    EmitF32("routeweight", route_weight, PACKED);
    EmitU32("routepacked", route_packed_row, PACKED);
    EmitU32("routesource", route_source_token, PACKED);
    EmitU32("groupoffset", group_row_offset, GLM52_EXPERTS + 1u);
    EmitU32("tileup", group_tile_prefix_w1, GLM52_EXPERTS + 1u);
    EmitU32("tiledown", group_tile_prefix_w2, GLM52_EXPERTS + 1u);
    EmitSample("intershared", intermediate, ROWS,
        GLM52_EXPERT_INTERMEDIATE, 173u);
    EmitSample("interrouted", intermediate +
        (ROWS * GLM52_EXPERT_INTERMEDIATE), PACKED - ROWS,
        GLM52_EXPERT_INTERMEDIATE, 173u);
    Emit("hidden2", hidden, ROWS * GLM52_HIDDEN);

    // The dense MLP twice: once with one contiguous gate/up tensor, once with
    // two tensors. The poison fill turns a half-written gate_up into a failure
    // rather than a stale read.
    buffers.dense_gate_weight = w_q_a;
    buffers.dense_up_weight = w_q_b;
    buffers.dense_down_weight = w_down;
    buffers.dense_gate_up_fused = 1u;
    poison = LmFloatToBf16(-7.0f);
    memset(gate_up, 0, sizeof(gate_up));
    for (index = 0u; index < ROWS * GLM52_DENSE_INTERMEDIATE * 2u; ++index)
        gate_up[index] = poison;
    status = Glm52LayerDenseMlp(&buffers, ROWS, 48u, 0);
    printf("status densefused %d\n", (int)status);
    EmitGemmLog(9u, "densefused");
    printf("poison %u\n",
        CountPoison(gate_up, ROWS * GLM52_DENSE_INTERMEDIATE * 2u, poison));
    EmitSample("gateup", gate_up, ROWS, GLM52_DENSE_INTERMEDIATE * 2u, 997u);

    buffers.dense_gate_up_fused = 0u;
    for (index = 0u; index < ROWS * GLM52_DENSE_INTERMEDIATE * 2u; ++index)
        gate_up[index] = poison;
    status = Glm52LayerDenseMlp(&buffers, ROWS, 48u, 0);
    printf("status densetwo %d\n", (int)status);
    EmitGemmLog(11u, "densetwo");
    printf("poison %u\n",
        CountPoison(gate_up, ROWS * GLM52_DENSE_INTERMEDIATE * 2u, poison));
    EmitSample("gateup2", gate_up, ROWS, GLM52_DENSE_INTERMEDIATE * 2u, 997u);

    status = Glm52Head(&buffers, head_norm_w, head_weight, 0, HEAD_VOCAB,
        ROWS, 0);
    printf("status head %d\n", (int)status);
    Emit("normed3", normed, ROWS * GLM52_HIDDEN);
    EmitU32("token", output_token, ROWS);
    EmitF32("score", output_score, ROWS);

    printf("done\n");
    return 0;
}
