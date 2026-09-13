#ifndef SPARKPIPE_DSV4_ROCM_L4_ROUTE_SYNTHESIS_H
#define SPARKPIPE_DSV4_ROCM_L4_ROUTE_SYNTHESIS_H

/* l4_route_synthesis.h - deterministic synthetic sealed-route batches and
 * the scalar REFERENCE MODEL for the L4 island's kernels, shared verbatim
 * by the host fixture generator (l4_route_driver.c) and the on-hardware
 * device checker (l4_route_device_check.hip). One source of truth so the
 * two sides cannot drift.
 *
 * The reference mirrors the device kernels' documented fixed trees exactly:
 *   - dot products accumulate per-lane strided partials (lane l sums k = l,
 *     l+64, ... ascending) then one shuffle-down butterfly (offsets
 *     32,16,8,4,2,1); lane 0 holds the total;
 *   - every bf16 store is round-to-nearest-even, infinities/NaN bypassing
 *     the rounding add;
 *   - swiglu clamps up two-sided and gate max-only when limit > 0;
 *   - the pair reduce folds routing weights in ascending k starting FROM
 *     the existing accumulator value (CUDA MoePairReduce association);
 *   - the route group publishes canonical ascending-expert exclusive
 *     prefix offsets and packs pairs within an expert in ascending flat
 *     route-element order.
 *
 * Host fp32 add/mul are IEEE-754 like device fp32; exp2f/expf may differ by
 * ulps between host libm and the device math library, which is why C3 is
 * compared under tolerances while C2 integer fields compare bit-exact.
 */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>

/* Weight-format codes mirrored from spark_dsv4_rocm_islands.h without
 * including it, so this header stays pure C usable from any TU. */
#define SLR4_FORMAT_BF16      0u
#define SLR4_FORMAT_MXFP4     3u
#define SLAR4_FORMAT_FP8_E4M3 4u
#define SLR4_FORMAT_FP8       4u

#define SLR4_WAVE_LANES 64u

#define SLR4_MAGIC 0x53524c34u /* "SLR4" little-endian */
#define SLR4_VERSION 1u

/* Deterministic 64-bit LCG (Knuth MMIX constants); identical sequence on
 * both sides of the comparison. */
typedef struct Slr4Random
{
    uint64_t state;
} Slr4Random;

static inline void Slr4Seed(Slr4Random *random, uint64_t seed)
{
    random->state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    random->state ^= random->state >> 31u;
}

static inline uint32_t Slr4NextU32(Slr4Random *random)
{
    random->state =
        random->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(random->state >> 33u);
}

static inline float Slr4NextUnitFloat(Slr4Random *random)
{
    /* [-1, 1), magnitude-biased small for stable accumulation. */
    float value = ((float)(Slr4NextU32(random) & 0xFFFFu) / 32768.0f) - 1.0f;
    return value * 0.25f;
}

/* ---- bf16 / packed-format scalar helpers (mirror of the island TU) ---- */

static inline float Slr4Bf16ToFloat(uint16_t raw)
{
    union { uint32_t bits; float value; } converter;
    converter.bits = ((uint32_t)raw) << 16u;
    return converter.value;
}

static inline uint16_t Slr4FloatToBf16Rne(float value)
{
    union { float value; uint32_t bits; } converter;
    uint32_t lsb, rounded;
    converter.value = value;
    if ((converter.bits & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)(converter.bits >> 16u);
    lsb = (converter.bits >> 16u) & 1u;
    rounded = converter.bits + 0x7fffu + lsb;
    return (uint16_t)(rounded >> 16u);
}

static inline float Slr4DecodeE8m0(uint32_t byte_value)
{
    union { uint32_t bits; float value; } converter;
    if (byte_value == 0xffu)
        return 0.0f;
    if (byte_value == 0u)
        converter.bits = 0x00400000u;
    else
        converter.bits = byte_value << 23u;
    return converter.value;
}

static inline float Slr4DecodeE4m3(uint32_t byte_value)
{
    uint32_t exponent = (byte_value >> 3u) & 0x0fu;
    uint32_t mantissa = byte_value & 7u;
    float sign = (byte_value & 0x80u) != 0u ? -1.0f : 1.0f;
    float magnitude;
    if ((byte_value & 0x7fu) == 0x7fu)
        return 0.0f;
    if (exponent == 0u)
        magnitude = (float)mantissa * 0.001953125f;
    else
        magnitude = (1.0f + ((float)mantissa * 0.125f)) *
                    exp2f((float)(int32_t)exponent - 7.0f);
    return sign * magnitude;
}

static inline float Slr4DecodeE2m1(uint32_t nibble)
{
    static const float magnitude_table[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float value = magnitude_table[nibble & 7u];
    return (nibble & 8u) != 0u ? -value : value;
}

static inline float Slr4Swish(float value)
{
    return value / (1.0f + expf(-value));
}

/* One weight element, row-major [rows x input_dimension], stacked-expert
 * tensors addressed by the already-stacked row index. Mirrors
 * SparkDsv4RocmWeightElement including scale indexing. */
static inline float Slr4WeightElement(uint32_t format,
                                      const void *payload,
                                      const void *scale_data,
                                      uint32_t scale_group, uint32_t row,
                                      uint32_t element,
                                      uint32_t input_dimension)
{
    if (format == SLR4_FORMAT_BF16)
    {
        const uint16_t *words = (const uint16_t *)payload;
        return Slr4Bf16ToFloat(words[((uint64_t)row * input_dimension) +
                                     element]);
    }
    {
        const uint8_t *bytes = (const uint8_t *)payload;
        uint32_t groups_per_row = input_dimension / scale_group;
        float value;
        if (format == SLR4_FORMAT_MXFP4)
        {
            uint64_t flat = ((uint64_t)row * input_dimension) + element;
            uint32_t nibble = bytes[flat >> 1u];
            nibble = (element & 1u) == 0u ? (nibble & 0x0fu)
                                          : (nibble >> 4u);
            value = Slr4DecodeE2m1(nibble);
        }
        else /* FP8_E4M3 */
        {
            value = Slr4DecodeE4m3(
                bytes[((uint64_t)row * input_dimension) + element]);
        }
        if (scale_data != 0 && scale_group != 0u)
        {
            const uint8_t *scales = (const uint8_t *)scale_data;
            value *= Slr4DecodeE8m0(
                scales[((uint64_t)row * groups_per_row) +
                       (element / scale_group)]);
        }
        return value;
    }
}

/* Fixed-tree wavefront sum over 64 per-lane partials: butterfly with
 * shuffle-down semantics (lane i adds lane i+offset). */
static inline float Slr4WaveReduce(const float lane_partial[SLR4_WAVE_LANES])
{
    float lane_value[SLR4_WAVE_LANES];
    uint32_t lane, offset;
    for (lane = 0u; lane < SLR4_WAVE_LANES; ++lane)
        lane_value[lane] = lane_partial[lane];
    for (offset = SLR4_WAVE_LANES / 2u; offset != 0u; offset >>= 1u)
    {
        for (lane = 0u; lane < offset; ++lane)
            lane_value[lane] += lane_value[lane + offset];
    }
    return lane_value[0];
}

/* Per-lane strided dot: lane l accumulates k = l, l+64, ... in fp32. */
static inline void Slr4LaneStridedDot(
    const float *input_values, uint32_t input_dimension,
    uint32_t format, const void *payload, const void *scale_data,
    uint32_t scale_group, uint32_t row, float lane_partial[SLR4_WAVE_LANES])
{
    uint32_t lane, k;
    for (lane = 0u; lane < SLR4_WAVE_LANES; ++lane)
        lane_partial[lane] = 0.0f;
    for (lane = 0u; lane < SLR4_WAVE_LANES; ++lane)
    {
        for (k = lane; k < input_dimension; k += SLR4_WAVE_LANES)
            lane_partial[lane] +=
                input_values[k] *
                Slr4WeightElement(format, payload, scale_data, scale_group,
                                  row, k, input_dimension);
    }
}

/* ---- Synthetic sealed-route batch descriptor ---- */

typedef struct Slr4CaseShape
{
    uint32_t row_count;
    uint32_t experts_per_token;
    uint32_t expert_count;
    uint32_t hidden_dimension;
    uint32_t moe_intermediate_dimension;
    uint32_t weight_format;    /* applied to w13_gate, w13_up, w2_down */
    uint32_t scale_group;      /* 0 for BF16 */
    float swiglu_limit;        /* <= 0 disables clamping */
    uint64_t seed;
} Slr4CaseShape;

static inline uint32_t Slr4PairCount(const Slr4CaseShape *shape)
{
    return shape->row_count * shape->experts_per_token;
}

static inline uint64_t Slr4Bf16ElementsPerCase(const Slr4CaseShape *shape)
{
    /* hidden states + initial accumulator only; weights are synthesized
     * straight into typed payloads below. */
    return (uint64_t)shape->row_count * shape->hidden_dimension +
           (uint64_t)shape->row_count * shape->hidden_dimension;
}

static inline size_t Slr4ViewPayloadBytes(const Slr4CaseShape *shape,
                                          uint32_t view_rows,
                                          uint32_t view_columns)
{
    size_t payload_bytes, scale_bytes;
    if (shape->weight_format == SLR4_FORMAT_BF16)
        return (size_t)view_rows * view_columns * sizeof(uint16_t);
    if (shape->weight_format == SLR4_FORMAT_MXFP4)
        payload_bytes = (size_t)view_rows * view_columns / 2u;
    else
        payload_bytes = (size_t)view_rows * view_columns;
    scale_bytes = (size_t)view_rows * (view_columns / shape->scale_group);
    return payload_bytes + scale_bytes;
}

/* Synthesize a stacked weight view (payload immediately followed by its
 * scale region inside one caller-owned buffer). Deterministic per seed and
 * view role byte. Payload bytes avoid the E4M3 NaN encoding 0x7f/0xff. */
static inline void Slr4SynthesizeView(const Slr4CaseShape *shape,
                                      uint64_t view_seed,
                                      uint32_t view_rows,
                                      uint32_t view_columns,
                                      uint8_t *buffer)
{
    Slr4Random random;
    size_t index, payload_bytes;
    uint32_t row, column;

    Slr4Seed(&random, view_seed);
    if (shape->weight_format == SLR4_FORMAT_BF16)
    {
        uint16_t *words = (uint16_t *)buffer;
        size_t word_count = (size_t)view_rows * view_columns;
        for (index = 0; index < word_count; ++index)
            words[index] = Slr4FloatToBf16Rne(Slr4NextUnitFloat(&random));
        return;
    }
    payload_bytes = (size_t)view_rows * view_columns /
                    (shape->weight_format == SLR4_FORMAT_MXFP4 ? 2u : 1u);
    for (index = 0; index < payload_bytes; ++index)
    {
        uint8_t byte = (uint8_t)(Slr4NextU32(&random) & 0xFFu);
        if (shape->weight_format == SLR4_FORMAT_FP8 &&
            (byte & 0x7fu) == 0x7fu)
            byte = 0x7eu;
        buffer[index] = byte;
    }
    index = payload_bytes;
    for (row = 0u; row < view_rows; ++row)
    {
        for (column = 0u; column < view_columns / shape->scale_group;
             ++column)
        {
            /* Small power-of-two neighborhood around 2^0, never 0x00 or
             * 0xff, so magnitudes stay sane for bring-up comparisons. */
            buffer[index++] =
                (uint8_t)(124u + (Slr4NextU32(&random) % 4u));
        }
    }
}

/* Hidden activation rows (bf16) and the initial FFN accumulator pattern. */
static inline void Slr4SynthesizeHidden(const Slr4CaseShape *shape,
                                        uint16_t *hidden_bf16)
{
    Slr4Random random;
    uint32_t row, column;
    size_t index = 0;

    Slr4Seed(&random, shape->seed ^ 0x48494444ULL); /* "HIDD" */
    for (row = 0u; row < shape->row_count; ++row)
    {
        for (column = 0u; column < shape->hidden_dimension; ++column)
            hidden_bf16[index++] =
                Slr4FloatToBf16Rne(Slr4NextUnitFloat(&random));
    }
}

static inline void Slr4SynthesizeAccumulator(const Slr4CaseShape *shape,
                                             uint16_t *accum_bf16)
{
    Slr4Random random;
    uint32_t row, column;
    size_t index = 0;

    Slr4Seed(&random, shape->seed ^ 0x41434355ULL); /* "ACCU" */
    for (row = 0u; row < shape->row_count; ++row)
    {
        for (column = 0u; column < shape->hidden_dimension; ++column)
            accum_bf16[index++] =
                Slr4FloatToBf16Rne(Slr4NextUnitFloat(&random));
    }
}

/* Sealed logical route: top-k DISTINCT experts per row (the router contract
 * guarantees distinctness; route.cuh documents the kernel neither checks
 * nor needs it) plus normalized descending weights with an occasional
 * deliberate zero. Selection order is NOT sorted so the flat-element rank
 * ordering is genuinely exercised. */
static inline void Slr4SynthesizeRoute(const Slr4CaseShape *shape,
                                       uint32_t *indices_u32,
                                       float *weights_f32)
{
    Slr4Random random;
    uint32_t row, rank;
    uint32_t scratch_index;
    uint32_t *scratch;

    Slr4Seed(&random, shape->seed ^ 0x524f5554ULL); /* "ROUT" */
    scratch = (uint32_t *)malloc(sizeof(uint32_t) * shape->expert_count);
    if (scratch == 0)
        return;
    for (scratch_index = 0u; scratch_index < shape->expert_count;
         ++scratch_index)
        scratch[scratch_index] = scratch_index;

    for (row = 0u; row < shape->row_count; ++row)
    {
        uint32_t remaining = shape->expert_count;
        float weight_total = 0.0f;
        for (rank = 0u; rank < shape->experts_per_token; ++rank)
        {
            uint32_t pick = Slr4NextU32(&random) % remaining;
            uint32_t expert = scratch[pick];
            scratch[pick] = scratch[remaining - 1u];
            remaining -= 1u;
            indices_u32[(uint64_t)row * shape->experts_per_token + rank] =
                expert;
            /* Deliberate zero weight on the last rank of every 5th row. */
            float weight =
                ((row % 5u) == 4u && rank + 1u == shape->experts_per_token)
                    ? 0.0f
                    : (1.0f / (float)(rank + 1u));
            weights_f32[(uint64_t)row * shape->experts_per_token + rank] =
                weight;
            weight_total += weight;
        }
        if (weight_total > 0.0f)
        {
            for (rank = 0u; rank < shape->experts_per_token; ++rank)
                weights_f32[(uint64_t)row * shape->experts_per_token +
                            rank] /= weight_total;
        }
    }
    free(scratch);
}

/* ---- Reference model ---- */

/* Route realization: canonical expert_offsets (exclusive prefix + total at
 * [expert_count]), inverse map, grouped source tokens and grouped experts,
 * pairs packed ascending within each expert. Mirrors RouteGroupKernel
 * phases; the histogram is order-free and phase 4 is a pure function of
 * the sealed element order. */
static inline void Slr4ReferenceRouteGroup(
    const uint32_t *indices_u32, uint32_t pair_count, uint32_t expert_count,
    uint32_t experts_per_token, uint32_t *expert_offsets_u32,
    uint32_t *inverse_pair_u32, uint32_t *grouped_source_token_u32,
    uint32_t *grouped_expert_u32)
{
    uint32_t expert, i, j, total = 0u;

    for (expert = 0u; expert < expert_count; ++expert)
        expert_offsets_u32[expert] = 0u;
    for (i = 0u; i < pair_count; ++i)
        expert_offsets_u32[indices_u32[i]] += 1u;
    total = 0u;
    for (expert = 0u; expert < expert_count; ++expert)
        total += expert_offsets_u32[expert];
    expert_offsets_u32[expert_count] = total;
    for (expert = expert_count; expert != 0u; --expert)
    {
        total -= expert_offsets_u32[expert - 1u];
        expert_offsets_u32[expert - 1u] = total;
    }
    for (i = 0u; i < pair_count; ++i)
    {
        uint32_t owned_expert = indices_u32[i];
        uint32_t rank = 0u;
        for (j = 0u; j < i; ++j)
            rank += indices_u32[j] == owned_expert ? 1u : 0u;
        {
            uint32_t slot = expert_offsets_u32[owned_expert] + rank;
            grouped_source_token_u32[slot] = i / experts_per_token;
            grouped_expert_u32[slot] = owned_expert;
            inverse_pair_u32[i] = slot;
        }
    }
}

/* Scale region of a synthesized view buffer sits directly behind its
 * payload bytes. */
static inline const uint8_t *Slr4ViewScale(const Slr4CaseShape *shape,
                                           const uint8_t *view_buffer,
                                           uint32_t view_rows,
                                           uint32_t view_columns)
{
    if (shape->weight_format == SLR4_FORMAT_BF16)
        return 0;
    return view_buffer + ((size_t)view_rows * view_columns /
                          (shape->weight_format == SLR4_FORMAT_MXFP4 ? 2u
                                                                     : 1u));
}

/* Fixed-tree dot of one stacked row against the staged input values. */
static inline float Slr4ViewDot(const Slr4CaseShape *shape,
                                const uint8_t *view_buffer,
                                uint32_t view_rows, uint32_t view_columns,
                                uint32_t input_dimension,
                                const float *input_values, uint32_t row)
{
    float lane_partial[SLR4_WAVE_LANES];
    const uint8_t *scale_data =
        Slr4ViewScale(shape, view_buffer, view_rows, view_columns);
    Slr4LaneStridedDot(input_values, input_dimension, shape->weight_format,
                       view_buffer, scale_data, shape->scale_group, row,
                       lane_partial);
    return Slr4WaveReduce(lane_partial);
}

/* Full C3 chain: grouped W13 + swiglu -> grouped W2 -> weighted reduce into
 * the accumulator. accum_bf16 is updated in place and becomes the expected
 * post-island buffer. Mirrors the four device kernels' trees exactly. */
static inline void Slr4ReferenceMoeChain(
    const Slr4CaseShape *shape, const uint16_t *hidden_bf16,
    const uint32_t *grouped_source_token_u32,
    const uint32_t *grouped_expert_u32, const uint32_t *inverse_pair_u32,
    const float *weights_f32, const uint8_t *w13_gate_buffer,
    const uint8_t *w13_up_buffer, const uint8_t *w2_down_buffer,
    uint16_t *up_scratch_bf16, uint16_t *pair_out_scratch_bf16,
    uint16_t *accum_bf16)
{
    float input_values[4096];
    uint32_t pair_count = Slr4PairCount(shape);
    uint32_t pair, neuron, row, column, k;

    if (shape->hidden_dimension > 4096u)
        return; /* reference bound; fixture shapes stay far below */

    for (pair = 0u; pair < pair_count; ++pair)
    {
        uint32_t source_row = grouped_source_token_u32[pair];
        uint32_t expert = grouped_expert_u32[pair];

        for (k = 0u; k < shape->hidden_dimension; ++k)
            input_values[k] = Slr4Bf16ToFloat(
                hidden_bf16[(uint64_t)source_row * shape->hidden_dimension +
                            k]);

        /* Grouped W13 + clamped swiglu (weight NOT folded here). */
        for (neuron = 0u; neuron < shape->moe_intermediate_dimension;
             ++neuron)
        {
            uint32_t stacked_neuron =
                expert * shape->moe_intermediate_dimension + neuron;
            float gate_value = Slr4ViewDot(
                shape, w13_gate_buffer, shape->expert_count *
                                            shape->moe_intermediate_dimension,
                shape->hidden_dimension, shape->hidden_dimension,
                input_values, stacked_neuron);
            float up_value = Slr4ViewDot(
                shape, w13_up_buffer,
                shape->expert_count * shape->moe_intermediate_dimension,
                shape->hidden_dimension, shape->hidden_dimension,
                input_values, stacked_neuron);
            if (shape->swiglu_limit > 0.0f)
            {
                up_value =
                    up_value > shape->swiglu_limit
                        ? shape->swiglu_limit
                        : (up_value < -shape->swiglu_limit
                               ? -shape->swiglu_limit
                               : up_value);
                gate_value = gate_value > shape->swiglu_limit
                                 ? shape->swiglu_limit
                                 : gate_value;
            }
            up_scratch_bf16[(uint64_t)pair *
                                shape->moe_intermediate_dimension +
                            neuron] =
                Slr4FloatToBf16Rne(Slr4Swish(gate_value) * up_value);
        }

        /* Grouped W2 down projection, unweighted rows. */
        for (neuron = 0u; neuron < shape->hidden_dimension; ++neuron)
        {
            uint32_t stacked_neuron =
                expert * shape->hidden_dimension + neuron;
            float down_lanes[SLR4_WAVE_LANES];
            uint32_t lane;
            for (lane = 0u; lane < SLR4_WAVE_LANES; ++lane)
                down_lanes[lane] = 0.0f;
            for (lane = 0u; lane < SLR4_WAVE_LANES; ++lane)
            {
                for (k = lane; k < shape->moe_intermediate_dimension;
                     k += SLR4_WAVE_LANES)
                {
                    float activated = Slr4Bf16ToFloat(
                        up_scratch_bf16[(uint64_t)pair *
                                            shape->moe_intermediate_dimension +
                                        k]);
                    down_lanes[lane] +=
                        activated *
                        Slr4WeightElement(
                            shape->weight_format, w2_down_buffer,
                            Slr4ViewScale(shape, w2_down_buffer,
                                          shape->expert_count *
                                              shape->hidden_dimension,
                                          shape->moe_intermediate_dimension),
                            shape->scale_group, stacked_neuron, k,
                            shape->moe_intermediate_dimension);
                }
            }
            pair_out_scratch_bf16[(uint64_t)pair * shape->hidden_dimension +
                                  neuron] =
                Slr4FloatToBf16Rne(Slr4WaveReduce(down_lanes));
        }
    }

    /* Weighted pair reduce: accumulator-first, ascending k, one RNE store. */
    for (row = 0u; row < shape->row_count; ++row)
    {
        for (column = 0u; column < shape->hidden_dimension; ++column)
        {
            uint32_t flat_base = row * shape->experts_per_token;
            uint64_t output_index =
                (uint64_t)row * shape->hidden_dimension + column;
            float total = Slr4Bf16ToFloat(accum_bf16[output_index]);
            for (k = 0u; k < shape->experts_per_token; ++k)
            {
                uint32_t flat = flat_base + k;
                total += weights_f32[flat] *
                         Slr4Bf16ToFloat(
                             pair_out_scratch_bf16[(uint64_t)
                                                       inverse_pair_u32[flat] *
                                                       shape->hidden_dimension +
                                                   column]);
            }
            accum_bf16[output_index] = Slr4FloatToBf16Rne(total);
        }
    }
}

#endif /* SPARKPIPE_DSV4_ROCM_L4_ROUTE_SYNTHESIS_H */
