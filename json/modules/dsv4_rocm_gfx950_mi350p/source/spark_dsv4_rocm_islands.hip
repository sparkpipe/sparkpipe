// spark_dsv4_rocm_islands.hip - island kernels + entries for the S7
// bring-up order steps 1 and 3 (docs/coord/plan_amd_gfx950_mi350p.md
// section 5.2): E0 `prologue.embed` (C2, integer exact), L5
// `layer.moe_shared` (C3, simplest accumulation island), and L4
// `layer.moe_routed` - synthetic sealed-route batches, route realization
// C2 integer-exact through the single-CTA grouping kernel, grouped-GEMM
// halves C3 (W13 gate/up + clamped swiglu, W2 down projection, weighted
// pair reduce).
//
// Declared surface lives in spark_dsv4_rocm_islands.h (freeze F1 table
// rows E0/L5). Everything below the seam is target-private (R1): these are
// deliberately simple bring-up kernels - correctness and a FIXED reduction
// tree first, tiling/fusion/perf is explicitly out of S7 scope (plan 5.3).
//
// Numerics mirrored from the shared machinery's documented formulas (the
// landmine rule forbids including spark_lm_kernels.cuh from any ROCm TU,
// so the algorithms are restated here from their frozen definitions):
//   - E4M3fn decode, finite-only (lm_kernels.cuh:157-170)
//   - E8M0 power-of-two scale byte (lm_kernels.cuh:172-180)
//   - swiglu clamp: up two-sided, gate max-only, swish(gate)*up in fp32
//     (spark_dsv4_resident_decode_stage_cuda.cu:1437-1461)
//   - bf16 stores round-to-nearest-even, matching __float2bfloat16
//
// L5 reduction tree (fixed for every shape bucket, B3/C3): each lane
// accumulates its k-stride subset sequentially in fp32, then one
// log2(64)-step shuffle-down butterfly reduces the wavefront, then exactly
// one RNE bf16 store. Never re-tuned per bucket.

#include "spark_hw_rocm_internal.h"

#include <math.h>

#include "spark_dsv4_rocm_islands.h"
#include <sparkpipe/spark_hardware_topology.h>

// ---------------------------------------------------------------------------
// Host-surface proof mode. When parsed by a PLAIN host compiler (no HIP
// language mode - the authoring workstation has no ROCm toolchain, plan
// section 2 box limitation), the vendor headers render __global__/__device__
// inert but provide no device intrinsics. These shims complete the surface
// so the whole TU parses as ordinary C++. hipcc NEVER compiles this branch
// (__HIP__ defined there); on-gfx950 verification happens on hardware.
// ---------------------------------------------------------------------------
#if !defined(__HIP__)

/* Implicit kernel coordinates exist only under the HIP compiler. The shims
 * are defined AFTER every include above, so they cannot shadow the
 * gridDim/blockDim struct members the API headers declare; kernel bodies
 * below are the only consumers. All zeros: proof-mode kernels would no-op,
 * but a host parse never executes them. */
typedef struct SparkDsv4RocmProbeCoordTag
{
    uint32_t x, y, z;
} SparkDsv4RocmProbeCoord;

static SparkDsv4RocmProbeCoord spark_dsv4_rocm_probe_coord[1] = {{0u, 0u, 0u}};

#define gridDim spark_dsv4_rocm_probe_coord[0]
#define blockIdx spark_dsv4_rocm_probe_coord[0]
#define blockDim spark_dsv4_rocm_probe_coord[0]
#define threadIdx spark_dsv4_rocm_probe_coord[0]

static inline float SparkDsv4RocmProbeShflDown(float value, int delta,
                                               int width)
{
    (void)delta;
    (void)width;
    /* Host proof: the caller's sequential partial IS the result; the real
     * butterfly runs only under the HIP compiler. */
    return value;
}

#define __shfl_down_sync(mask, var, src_lane, width)                          \
    SparkDsv4RocmProbeShflDown((var), (src_lane), (width))

/* Barrier exists only under the HIP compiler; a host parse never executes
 * device code, so it renders as a no-op. */
#define __syncthreads() do { } while (0)

static inline uint32_t SparkDsv4RocmProbeAtomicAdd(uint32_t *address,
                                                   uint32_t value)
{
    uint32_t previous = *address;
    *address = previous + value;
    return previous;
}

#define atomicAdd(address, value)                                             \
    SparkDsv4RocmProbeAtomicAdd((address), (uint32_t)(value))

static inline float SparkHwRocmBitsAsFloat(uint32_t bits)
{
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

#else /* !defined(__HIP__) */

static inline float SparkHwRocmBitsAsFloat(uint32_t bits)
{
    return __uint_as_float(bits);
}

#endif /* !defined(__HIP__) */

// Archive-local constants (target-owned; GB10 tile/thread constants are
// never imported - plan section 8 risk table). One CTA is four 64-lane
// wavefronts.
#define SPARK_DSV4_ROCM_CTA_THREADS 256u
#define SPARK_DSV4_ROCM_WARPS_PER_CTA                                              \
    (SPARK_DSV4_ROCM_CTA_THREADS / SPARK_HW_WAVEFRONT_LANES)
#define SPARK_DSV4_ROCM_WAVE_LANES ((uint32_t)SPARK_HW_WAVEFRONT_LANES)

#if defined(__cplusplus)
static_assert(SPARK_DSV4_ROCM_WAVE_LANES == 64u,
              "gfx950 island kernels are written for 64-lane wavefronts; "
              "compile with -DSPARK_HW_TARGET_ID=SPARK_HW_TARGET_ROCM_GFX950_MI350P");
#endif

// ---------------------------------------------------------------------------
// Scalar helpers (annotated for the device pass; inert annotations keep them
// host-parsable in proof mode).
// ---------------------------------------------------------------------------

#define SPARK_DSV4_ROCM_DEV static inline

// bf16 -> fp32: pure exponent-width shift, exact.
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmBf16ToFloat(uint16_t raw)
{
    return SparkHwRocmBitsAsFloat(((uint32_t)raw) << 16u);
}

// fp32 -> bf16 round-to-nearest-even, the __float2bfloat16 semantic.
// Infinity/NaN bypass the rounding add so a NaN payload can never carry
// into an infinity.
SPARK_DSV4_ROCM_DEV uint16_t SparkDsv4RocmFloatToBf16Rne(float value)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)(bits >> 16u);
    {
        uint32_t lsb = (bits >> 16u) & 1u;
        uint32_t rounded = bits + 0x7fffu + lsb;
        return (uint16_t)(rounded >> 16u);
    }
}

// E4M3fn byte -> float, finite only (0x7f/0xff are NaN -> zero),
// subnormals at 2^-9 granularity. Mirrors the shared definition verbatim.
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmDecodeE4m3(uint32_t byte_value)
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
        magnitude =
            (1.0f + ((float)mantissa * 0.125f)) *
            exp2f((float)(int32_t)exponent - 7.0f);
    return sign * magnitude;
}

// E8M0 scale byte: pure power of two, bias 127, 0xff reserved -> zero,
// 0x00 -> 2^-127. Mirrors the shared definition verbatim.
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmDecodeE8m0(uint32_t byte_value)
{
    if (byte_value == 0xffu)
        return 0.0f;
    if (byte_value == 0u)
        return SparkHwRocmBitsAsFloat(0x00400000u);
    return SparkHwRocmBitsAsFloat(byte_value << 23u);
}

// E2M1 nibble -> float: sign bit 3, 2 exponent bits (bias 1), 1 mantissa
// bit. Table decode exactly as the shared definition; every table value is
// representable, so decoding straight to fp32 matches the shared machinery's
// fp16-round-tripped values bit for bit.
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmDecodeE2m1(uint32_t nibble)
{
    /* Indexing constant folded per call; no static state on device. */
    float magnitude;
    switch (nibble & 7u)
    {
    case 0u: magnitude = 0.0f; break;
    case 1u: magnitude = 0.5f; break;
    case 2u: magnitude = 1.0f; break;
    case 3u: magnitude = 1.5f; break;
    case 4u: magnitude = 2.0f; break;
    case 5u: magnitude = 3.0f; break;
    case 6u: magnitude = 4.0f; break;
    default: magnitude = 6.0f; break;
    }
    return (nibble & 8u) != 0u ? -magnitude : magnitude;
}

SPARK_DSV4_ROCM_DEV float SparkDsv4RocmSigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

SPARK_DSV4_ROCM_DEV float SparkDsv4RocmSwish(float value)
{
    return value * SparkDsv4RocmSigmoid(value);
}

// Fixed-tree wavefront sum: shuffle-down butterfly, offsets 32..1, width 64.
// THE tree for every L5 dot product in every shape bucket.
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmWaveReduceSum(float value)
{
    /* One spelling, two renderings: the HIP intrinsic on device passes,
     * the proof-mode macro above under a plain host parse. */
    for (uint32_t offset = SPARK_DSV4_ROCM_WAVE_LANES / 2u; offset != 0u;
         offset >>= 1u)
        value += __shfl_down_sync(0xffffffffffffffffull, value, (int)offset,
                                  (int)SPARK_DSV4_ROCM_WAVE_LANES);
    return value;
}

// One weight element decoded to fp32 through the same arithmetic for every
// supported format. Row-major packed views: element (neuron, k) of a
// rows x input_dimension matrix; stacked-expert tensors pass the stacked
// row index expert*per_expert_rows + row as `neuron`. Scale indexing
// assumes scale_group divides input_dimension exactly (validated at the
// entry).
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmWeightElement(
    uint32_t weight_format, const void *weight_payload,
    const void *weight_scale, uint32_t scale_group, uint32_t neuron,
    uint32_t element, uint32_t input_dimension)
{
    if (weight_format == SPARK_DSV4_ROCM_WEIGHT_FORMAT_BF16)
    {
        const uint16_t *words = (const uint16_t *)weight_payload;
        return SparkDsv4RocmBf16ToFloat(
            words[((uint64_t)neuron * input_dimension) + element]);
    }
    if (weight_format == SPARK_DSV4_ROCM_WEIGHT_FORMAT_MXFP4_E2M1)
    {
        /* E2M1 nibbles, low nibble first within each little-endian byte -
         * the byte view of the shared machinery's 8-per-word decode.
         * Byte addressing is exact because element pairs never straddle a
         * word in that layout. One E8M0 scale byte per scale_group block. */
        const uint8_t *payload = (const uint8_t *)weight_payload;
        uint64_t flat = ((uint64_t)neuron * input_dimension) + element;
        uint32_t nibble = payload[flat >> 1u];
        nibble = (element & 1u) == 0u ? (nibble & 0x0fu) : (nibble >> 4u);
        float value = SparkDsv4RocmDecodeE2m1(nibble);
        if (weight_scale != 0 && scale_group != 0u)
        {
            const uint8_t *scale = (const uint8_t *)weight_scale;
            uint32_t groups_per_row = input_dimension / scale_group;
            value *= SparkDsv4RocmDecodeE8m0(
                scale[((uint64_t)neuron * groups_per_row) +
                      (element / scale_group)]);
        }
        return value;
    }
    {
        /* SPARK_DSV4_ROCM_WEIGHT_FORMAT_FP8_E4M3 */
        const uint8_t *payload = (const uint8_t *)weight_payload;
        float value = SparkDsv4RocmDecodeE4m3(
            payload[((uint64_t)neuron * input_dimension) + element]);
        if (weight_scale != 0 && scale_group != 0u)
        {
            const uint8_t *scale = (const uint8_t *)weight_scale;
            uint32_t groups_per_row = input_dimension / scale_group;
            value *= SparkDsv4RocmDecodeE8m0(
                scale[((uint64_t)neuron * groups_per_row) +
                      (element / scale_group)]);
        }
        return value;
    }
}

// ---------------------------------------------------------------------------
// E0 `prologue.embed` kernel - C2. One CTA per boundary packet row; bytes
// move verbatim (16-byte vectors when hidden allows, raw 16-bit tail
// otherwise). No arithmetic touches the payload, so cross-target identity
// holds by construction.
// ---------------------------------------------------------------------------

typedef struct __attribute__((aligned(16))) SparkDsv4RocmVec16Tag
{
    uint32_t word[4];
} SparkDsv4RocmVec16;

__global__ void SparkDsv4RocmEmbedGatherExpandKernel(
    const uint32_t *token_ids, const void *embedding_bf16,
    void *boundary_packet_bf16, uint32_t hidden_dimension, uint32_t hc_mult)
{
    uint32_t row = blockIdx.x;
    uint32_t vector_count = hidden_dimension >> 3u;
    uint32_t stream, element;
    uint64_t source_base, destination_base;
    const uint16_t *source_words = (const uint16_t *)embedding_bf16;
    uint16_t *destination_words = (uint16_t *)boundary_packet_bf16;

    source_base = (uint64_t)token_ids[row] * hidden_dimension;
    destination_base = (uint64_t)row * hc_mult * hidden_dimension;

    for (stream = 0u; stream < hc_mult; ++stream)
    {
        uint64_t source_offset = source_base;
        uint64_t destination_offset = destination_base +
            (uint64_t)stream * hidden_dimension;
        if (vector_count != 0u)
        {
            const SparkDsv4RocmVec16 *source_vectors =
                (const SparkDsv4RocmVec16 *)((const uint8_t *)embedding_bf16);
            SparkDsv4RocmVec16 *destination_vectors =
                (SparkDsv4RocmVec16 *)boundary_packet_bf16;
            uint64_t source_vector = source_offset >> 3u;
            uint64_t destination_vector = destination_offset >> 3u;
            for (element = threadIdx.x; element < vector_count;
                 element += blockDim.x)
                destination_vectors[destination_vector + element] =
                    source_vectors[source_vector + element];
        }
        for (element = (vector_count << 3u) + threadIdx.x;
             element < hidden_dimension; element += blockDim.x)
            destination_words[destination_offset + element] =
                source_words[source_offset + element];
    }
}

// ---------------------------------------------------------------------------
// L5 `layer.moe_shared` kernels - C3. Three uncaptured launches (fusion and
// graph capture come later in S7, plan 5.2 step 5):
//   1. gate/up projections (one wavefront per output neuron, w1/w3 halves),
//   2. clamped swiglu elementwise,
//   3. down projection accumulated into the FFN accumulator.
// ---------------------------------------------------------------------------

__global__ void SparkDsv4RocmSharedGateUpKernel(
    uint32_t weight_format, const void *w1_payload, const void *w1_scale,
    const void *w3_payload, const void *w3_scale, uint32_t scale_group,
    const void *input_bf16, void *gate_bf16, void *up_bf16,
    uint32_t input_dimension, uint32_t output_dimension)
{
    uint32_t warp = threadIdx.x / SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t lane = threadIdx.x % SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t combined_neuron = blockIdx.y * SPARK_DSV4_ROCM_WARPS_PER_CTA +
                               warp;
    uint32_t neuron, side, k;
    const void *weight_payload;
    const void *weight_scale;
    void *output_bf16;
    const uint16_t *input_words = (const uint16_t *)input_bf16;
    float accumulator;

    if (combined_neuron >= output_dimension * 2u)
        return;
    if (combined_neuron < output_dimension)
    {
        side = 0u;
        neuron = combined_neuron;
        weight_payload = w1_payload;
        weight_scale = w1_scale;
        output_bf16 = gate_bf16;
    }
    else
    {
        side = 1u;
        neuron = combined_neuron - output_dimension;
        weight_payload = w3_payload;
        weight_scale = w3_scale;
        output_bf16 = up_bf16;
    }
    (void)side;

    accumulator = 0.0f;
    for (k = lane; k < input_dimension; k += SPARK_DSV4_ROCM_WAVE_LANES)
    {
        float input_value = SparkDsv4RocmBf16ToFloat(
            input_words[((uint64_t)blockIdx.x * input_dimension) + k]);
        accumulator += input_value *
            SparkDsv4RocmWeightElement(weight_format, weight_payload,
                                       weight_scale, scale_group, neuron, k,
                                       input_dimension);
    }
    accumulator = SparkDsv4RocmWaveReduceSum(accumulator);
    if (lane == 0u)
    {
        uint16_t *output_words = (uint16_t *)output_bf16;
        output_words[((uint64_t)blockIdx.x * output_dimension) + neuron] =
            SparkDsv4RocmFloatToBf16Rne(accumulator);
    }
}

__global__ void SparkDsv4RocmSharedSwigluClampKernel(
    void *gate_and_activated_bf16, const void *up_bf16, uint32_t row_count,
    uint32_t width, float limit)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = row_count * width;
    if (index >= total)
        return;
    {
        uint16_t *gate_words = (uint16_t *)gate_and_activated_bf16;
        const uint16_t *up_words = (const uint16_t *)up_bf16;
        float gate = SparkDsv4RocmBf16ToFloat(gate_words[index]);
        float up = SparkDsv4RocmBf16ToFloat(up_words[index]);
        if (limit > 0.0f)
        {
            up = up > limit ? limit : (up < -limit ? -limit : up);
            gate = gate > limit ? limit : gate;
        }
        /* In place over the gate half: this thread owns exactly index. */
        gate_words[index] =
            SparkDsv4RocmFloatToBf16Rne(SparkDsv4RocmSwish(gate) * up);
    }
}

__global__ void SparkDsv4RocmSharedDownAccumKernel(
    uint32_t weight_format, const void *w2_payload, const void *w2_scale,
    uint32_t scale_group, const void *activated_bf16,
    void *ffn_accum_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
    uint32_t warp = threadIdx.x / SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t lane = threadIdx.x % SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t neuron = blockIdx.y * SPARK_DSV4_ROCM_WARPS_PER_CTA + warp;
    uint32_t k;
    const uint16_t *activated_words = (const uint16_t *)activated_bf16;
    uint16_t *accum_words = (uint16_t *)ffn_accum_bf16;
    float accumulator;

    if (neuron >= output_dimension)
        return;

    accumulator = 0.0f;
    for (k = lane; k < input_dimension; k += SPARK_DSV4_ROCM_WAVE_LANES)
    {
        float activated = SparkDsv4RocmBf16ToFloat(
            activated_words[((uint64_t)blockIdx.x * input_dimension) + k]);
        accumulator += activated *
            SparkDsv4RocmWeightElement(weight_format, w2_payload, w2_scale,
                                       scale_group, neuron, k,
                                       input_dimension);
    }
    accumulator = SparkDsv4RocmWaveReduceSum(accumulator);
    if (lane == 0u)
    {
        uint64_t offset = ((uint64_t)blockIdx.x * output_dimension) + neuron;
        /* FFN accumulator (partial): add into the joined accumulator in
         * fp32, one final RNE bf16 store - the AccumAdd semantic. */
        accumulator += SparkDsv4RocmBf16ToFloat(accum_words[offset]);
        accum_words[offset] = SparkDsv4RocmFloatToBf16Rne(accumulator);
    }
}

// ---------------------------------------------------------------------------
// L4 `layer.moe_routed` kernels - route realization C2, grouped GEMM C3.
//
// Grouping discipline (the C2 half): ONE cooperative CTA owns the whole
// sealed batch. Phase 1 zeroes the offset table; phase 2 histograms expert
// occurrence counts with device atomics (order-free - only totals matter);
// phase 3 is a thread-0 in-place descending exclusive scan publishing the
// canonical offsets plus total; phase 4 assigns each flat route element a
// slot by counting rank - #{j < i : expert[j] == expert[i]} - which is a
// pure function of the core-sealed element order, so grouped assignment is
// run-invariant, not merely usually-stable. No shared memory, no cross-CTA
// coupling; pairs are packed ascending within each expert.
// ---------------------------------------------------------------------------

__global__ void SparkDsv4RocmRouteGroupKernel(
    const uint32_t *route_indices_u32, uint32_t *grouped_source_token_u32,
    uint32_t *grouped_expert_u32, uint32_t *expert_offsets_u32,
    uint32_t *inverse_pair_u32, uint32_t pair_count,
    uint32_t experts_per_token, uint32_t expert_count)
{
    uint32_t i, j, e;
    /* Phase 1: zero the count region (offset[0..expert_count) double as
     * counters until the scan rewrites them canonically). */
    for (e = threadIdx.x; e < expert_count; e += blockDim.x)
        expert_offsets_u32[e] = 0u;
    __syncthreads();
    /* Phase 2: histogram. Atomics make the sum order-free; only totals
     * feed the C2 observable. Core-sealed indices are < expert_count by
     * contract (entry documents the precondition). */
    for (i = threadIdx.x; i < pair_count; i += blockDim.x)
        atomicAdd(&expert_offsets_u32[route_indices_u32[i]], 1u);
    __syncthreads();
    /* Phase 3: canonical exclusive scan + total, one writer, in place:
     * ascending pass reads every count exactly once into the running total,
     * then the descending rewrite consumes cells before overwriting them. */
    if (threadIdx.x == 0u)
    {
        uint32_t total = 0u;
        for (e = 0u; e < expert_count; ++e)
            total += expert_offsets_u32[e];
        expert_offsets_u32[expert_count] = total;
        for (e = expert_count; e != 0u; --e)
        {
            total -= expert_offsets_u32[e - 1u];
            expert_offsets_u32[e - 1u] = total;
        }
    }
    __syncthreads();
    /* Phase 4: deterministic scatter. Rank by explicit counting over the
     * flat element order (bring-up O(n^2), correctness first; S7 excludes
     * perf). Writes within an expert target distinct slots, so no races. */
    for (i = threadIdx.x; i < pair_count; i += blockDim.x)
    {
        uint32_t expert = route_indices_u32[i];
        uint32_t rank = 0u;
        for (j = 0u; j < i; ++j)
            rank += route_indices_u32[j] == expert ? 1u : 0u;
        {
            uint32_t slot = expert_offsets_u32[expert] + rank;
            grouped_source_token_u32[slot] = i / experts_per_token;
            grouped_expert_u32[slot] = expert;
            inverse_pair_u32[i] = slot;
        }
    }
}

// Grouped W13 gate/up projections with clamped swiglu folded at the store:
// one warp per (packed pair, intermediate neuron); lanes stride hidden in
// the fixed tree (sequential fp32 per-lane accumulation, one wavefront
// butterfly, exactly one RNE bf16 store). Routing weight NOT folded here -
// it is applied once at the pair reduce (CUDA MoePairReduce parity).
__global__ void SparkDsv4RocmExpertW13SwigluKernel(
    uint32_t weight_format, const void *w13_gate_payload,
    const void *w13_gate_scale, const void *w13_up_payload,
    const void *w13_up_scale, uint32_t scale_group,
    const void *normalized_hidden_bf16,
    const uint32_t *grouped_source_token_u32,
    const uint32_t *grouped_expert_u32, void *up_scratch_bf16,
    float swiglu_limit, uint32_t hidden_dimension,
    uint32_t moe_intermediate_dimension)
{
    const uint16_t *hidden_words = (const uint16_t *)normalized_hidden_bf16;
    uint16_t *up_words = (uint16_t *)up_scratch_bf16;
    uint32_t pair = blockIdx.x;
    uint32_t lane = threadIdx.x % SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t neuron = blockIdx.y * SPARK_DSV4_ROCM_WARPS_PER_CTA +
                      (threadIdx.x / SPARK_DSV4_ROCM_WAVE_LANES);
    uint32_t source_row, stacked_neuron, k;
    float gate_accumulator, up_accumulator, gate_value, up_value;

    if (neuron >= moe_intermediate_dimension)
        return;
    source_row = grouped_source_token_u32[pair];
    stacked_neuron = grouped_expert_u32[pair] * moe_intermediate_dimension +
                     neuron;
    gate_accumulator = 0.0f;
    up_accumulator = 0.0f;
    for (k = lane; k < hidden_dimension; k += SPARK_DSV4_ROCM_WAVE_LANES)
    {
        float x = SparkDsv4RocmBf16ToFloat(
            hidden_words[((uint64_t)source_row * hidden_dimension) + k]);
        gate_accumulator += x * SparkDsv4RocmWeightElement(
            weight_format, w13_gate_payload, w13_gate_scale, scale_group,
            stacked_neuron, k, hidden_dimension);
        up_accumulator += x * SparkDsv4RocmWeightElement(
            weight_format, w13_up_payload, w13_up_scale, scale_group,
            stacked_neuron, k, hidden_dimension);
    }
    gate_value = SparkDsv4RocmWaveReduceSum(gate_accumulator);
    up_value = SparkDsv4RocmWaveReduceSum(up_accumulator);
    if (lane == 0u)
    {
        /* Identical clamp semantics to L5: up two-sided, gate max-only,
         * limit <= 0 disables. swish(gate)*up computed in fp32, one RNE
         * bf16 store. */
        if (swiglu_limit > 0.0f)
        {
            up_value = up_value > swiglu_limit ? swiglu_limit :
                       (up_value < -swiglu_limit ? -swiglu_limit : up_value);
            gate_value = gate_value > swiglu_limit ? swiglu_limit : gate_value;
        }
        up_words[((uint64_t)pair * moe_intermediate_dimension) + neuron] =
            SparkDsv4RocmFloatToBf16Rne(SparkDsv4RocmSwish(gate_value) *
                                        up_value);
    }
}

// The FFN down projection of the routed leg: grouped W2, one warp per
// (packed pair, hidden neuron), unweighted output rows so the pair reduce
// folds each routing weight exactly once.
__global__ void SparkDsv4RocmExpertDownKernel(
    uint32_t weight_format, const void *w2_down_payload,
    const void *w2_down_scale, uint32_t scale_group,
    const void *up_scratch_bf16, const uint32_t *grouped_expert_u32,
    void *pair_out_scratch_bf16, uint32_t hidden_dimension,
    uint32_t moe_intermediate_dimension)
{
    const uint16_t *activated_words = (const uint16_t *)up_scratch_bf16;
    uint16_t *out_words = (uint16_t *)pair_out_scratch_bf16;
    uint32_t pair = blockIdx.x;
    uint32_t lane = threadIdx.x % SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t neuron = blockIdx.y * SPARK_DSV4_ROCM_WARPS_PER_CTA +
                      (threadIdx.x / SPARK_DSV4_ROCM_WAVE_LANES);
    uint32_t stacked_neuron, i;
    float accumulator;

    if (neuron >= hidden_dimension)
        return;
    stacked_neuron = grouped_expert_u32[pair] * hidden_dimension + neuron;
    accumulator = 0.0f;
    for (i = lane; i < moe_intermediate_dimension;
         i += SPARK_DSV4_ROCM_WAVE_LANES)
    {
        float activated = SparkDsv4RocmBf16ToFloat(
            activated_words[((uint64_t)pair * moe_intermediate_dimension) +
                            i]);
        /* W2 row layout: rows = expert_count x hidden_dimension, columns =
         * moe_intermediate_dimension - element (stacked_neuron, i). */
        accumulator += activated * SparkDsv4RocmWeightElement(
            weight_format, w2_down_payload, w2_down_scale, scale_group,
            stacked_neuron, i, moe_intermediate_dimension);
    }
    accumulator = SparkDsv4RocmWaveReduceSum(accumulator);
    if (lane == 0u)
        out_words[((uint64_t)pair * hidden_dimension) + neuron] =
            SparkDsv4RocmFloatToBf16Rne(accumulator);
}

// Weighted pair reduce over the unweighted W2 outputs, AccumAdd semantics
// like L5 and fold order exactly like the CUDA MoePairReduce: start from
// the existing accumulator value, then fold the token's top-k pair results
// in ascending k order in fp32, each routing weight applied exactly once;
// one final RNE bf16 store. Each (row, element) is written by exactly one
// thread - race-free through the inverse map, no atomics.
__global__ void SparkDsv4RocmMoePairReduceKernel(
    const void *pair_out_scratch_bf16, const uint32_t *inverse_pair_u32,
    const float *route_weights_f32, void *ffn_accum_bf16,
    uint32_t row_count, uint32_t experts_per_token,
    uint32_t hidden_dimension)
{
    const uint16_t *out_words = (const uint16_t *)pair_out_scratch_bf16;
    uint16_t *accum_words = (uint16_t *)ffn_accum_bf16;
    uint32_t global = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t column, k, row;
    float total;

    if (global >= row_count * hidden_dimension)
        return;
    row = global / hidden_dimension;
    column = global % hidden_dimension;
    total = SparkDsv4RocmBf16ToFloat(accum_words[global]);
    for (k = 0u; k < experts_per_token; ++k)
    {
        uint32_t flat = (row * experts_per_token) + k;
        uint32_t pair = inverse_pair_u32[flat];
        total += route_weights_f32[flat] *
                 SparkDsv4RocmBf16ToFloat(
                     out_words[((uint64_t)pair * hidden_dimension) + column]);
    }
    accum_words[global] = SparkDsv4RocmFloatToBf16Rne(total);
}

// ---------------------------------------------------------------------------
// Launch plumbing
// ---------------------------------------------------------------------------

static SparkHwStatus SparkDsv4RocmLaunch(
    const void *kernel_function, unsigned int grid_x, unsigned int grid_y,
    unsigned int block_x, unsigned int shared_bytes, hipStream_t stream,
    void **kernel_arguments)
{
    dim3 grid(grid_x, grid_y, 1u);
    dim3 block(block_x, 1u, 1u);
    hipError_t error = hipLaunchKernel(kernel_function, grid, block,
                                       kernel_arguments, shared_bytes, stream);
    return spark_hw_rocm_map_status(error);
}

/* HIP grid-Y ceiling. Exceeding it is a capacity refusal (EXHAUSTED feeds
 * core admission/backpressure), never an argument error - same reasoning
 * as the uint32 flattening guards. */
#define SPARK_DSV4_ROCM_MAX_GRID_Y 65535u

static SparkHwStatus SparkDsv4RocmCheckGridY(uint32_t blocks_y)
{
    return blocks_y <= SPARK_DSV4_ROCM_MAX_GRID_Y
               ? SPARK_HW_OK
               : SPARK_HW_EXHAUSTED;
}

// View validation shared by the island entries. Returns SPARK_HW_OK or
// INVALID. Accepts every format SparkDsv4RocmWeightElement decodes; a
// format added there without a rule here must fail closed, not fall
// through to the FP8 path.
static SparkHwStatus SparkDsv4RocmCheckLinearView(
    const SparkDsv4RocmLinearView *view, uint32_t expected_rows,
    uint32_t expected_columns)
{
    if (view == 0)
        return SPARK_HW_INVALID;
    if (view->weight_format != SPARK_DSV4_ROCM_WEIGHT_FORMAT_BF16 &&
        view->weight_format != SPARK_DSV4_ROCM_WEIGHT_FORMAT_FP8_E4M3 &&
        view->weight_format != SPARK_DSV4_ROCM_WEIGHT_FORMAT_MXFP4_E2M1)
        return SPARK_HW_INVALID;
    if (view->payload == 0 || view->rows != expected_rows ||
        view->columns != expected_columns)
        return SPARK_HW_INVALID;
    if (view->weight_format == SPARK_DSV4_ROCM_WEIGHT_FORMAT_BF16)
    {
        /* Stagepack bf16 tensors are scale-free. */
        if (view->scale_data != 0 || view->scale_group != 0u)
            return SPARK_HW_INVALID;
    }
    else
    {
        /* Packed formats carry one E8M0 byte per scale_group elements per
         * row (stagepack rules: fp8 block 128 columns, fp4 block 32);
         * scales divide evenly, and the byte-packed E2M1 decoder walks
         * 8-element runs, so a scale block must never split a run. */
        if (view->scale_data == 0 || view->scale_group == 0u ||
            (expected_columns % view->scale_group) != 0u)
            return SPARK_HW_INVALID;
        if (view->weight_format == SPARK_DSV4_ROCM_WEIGHT_FORMAT_MXFP4_E2M1 &&
            (view->scale_group % 8u) != 0u)
            return SPARK_HW_INVALID;
    }
    return SPARK_HW_OK;
}

// ---------------------------------------------------------------------------
// Island entries (C linkage, declared in spark_dsv4_rocm_islands.h)
// ---------------------------------------------------------------------------

SparkHwStatus spark_dsv4_rocm_e0_prologue_embed(
    SparkHwQueue queue, SparkHwMemory memory, const uint32_t *token_ids_device,
    const void *embedding_bf16_device, void *boundary_packet_bf16_device,
    uint32_t row_count, uint32_t hidden_dimension, uint32_t hc_mult,
    uint32_t vocab_size, void *read_ahead_sink_u32,
    uint32_t read_ahead_word_capacity, void *read_ahead_context)
{
    hipStream_t stream;
    SparkHwStatus status;
    void *arguments[5];

    if (queue == 0 || token_ids_device == 0 || embedding_bf16_device == 0 ||
        boundary_packet_bf16_device == 0 || hidden_dimension == 0u ||
        hc_mult == 0u || vocab_size == 0u)
        return SPARK_HW_INVALID;
    /* The 16-byte gather path indexes rows in hidden_dimension units, so
     * every row base is vector-aligned exactly when hidden divides by 8 -
     * a pack geometry invariant (model dimensions are multiples of 64).
     * Refuse rather than fault on a misaligned vector. */
    if (((uintptr_t)boundary_packet_bf16_device % 8u) != 0u ||
        (hidden_dimension % 8u) != 0u)
        return SPARK_HW_INVALID;
    if (memory != 0)
    {
        /* Declared handle; the v1 gather consumes no workspace, but a bad
         * handle still fails closed rather than being ignored. */
        status = spark_hw_rocm_check_memory(&memory);
        if (status != SPARK_HW_OK)
            return status;
    }
    /* Zero rows is a legal empty frame: no launch, stream untouched. */
    if (row_count == 0u)
    {
        if (read_ahead_sink_u32 != 0)
            return spark_hw_read_ahead(queue, read_ahead_sink_u32,
                                       read_ahead_word_capacity,
                                       read_ahead_context);
        return SPARK_HW_OK;
    }

    stream = (hipStream_t)(void *)queue;
    arguments[0] = (void *)&token_ids_device;
    arguments[1] = (void *)&embedding_bf16_device;
    arguments[2] = (void *)&boundary_packet_bf16_device;
    arguments[3] = (void *)&hidden_dimension;
    arguments[4] = (void *)&hc_mult;

    status = SparkDsv4RocmLaunch((const void *)&SparkDsv4RocmEmbedGatherExpandKernel,
                                 row_count, 1u, SPARK_DSV4_ROCM_CTA_THREADS, 0u,
                                 stream, arguments);
    if (status != SPARK_HW_OK)
        return status;

    if (read_ahead_sink_u32 != 0)
    {
        status = spark_hw_read_ahead(queue, read_ahead_sink_u32,
                                     read_ahead_word_capacity,
                                     read_ahead_context);
        if (status != SPARK_HW_OK)
            return status;
    }
    return SPARK_HW_OK;
}

SparkHwStatus spark_dsv4_rocm_l5_moe_shared(
    SparkHwQueue aux_queue, SparkHwEvent fork_event, SparkHwEvent join_event,
    SparkHwMemory memory, const void *normalized_hidden_bf16_device,
    const SparkDsv4RocmLinearView *w1_gate,
    const SparkDsv4RocmLinearView *w3_up,
    const SparkDsv4RocmLinearView *w2_down, float swiglu_limit,
    void *gate_up_scratch_bf16_device, void *ffn_accum_bf16_device,
    uint32_t row_count, uint32_t hidden_dimension,
    uint32_t moe_intermediate_dimension)
{
    hipStream_t stream;
    SparkHwStatus status;
    uint32_t format, scale_group;
    uint32_t gate_up_grid_y, down_grid_y;
    void *gate_half, *up_half;
    void *arguments[11];

    if (memory != 0)
    {
        /* Declared handle; activations live in caller-owned slot workspace
         * (pool-backed allocation is core policy), but a bad handle still
         * fails closed rather than being ignored - same as E0. */
        status = spark_hw_rocm_check_memory(&memory);
        if (status != SPARK_HW_OK)
            return status;
    }

    if (aux_queue == 0 || normalized_hidden_bf16_device == 0 ||
        gate_up_scratch_bf16_device == 0 || ffn_accum_bf16_device == 0 ||
        hidden_dimension == 0u || moe_intermediate_dimension == 0u)
        return SPARK_HW_INVALID;
    if (swiglu_limit != swiglu_limit) /* NaN limit refuses */
        return SPARK_HW_INVALID;

    status = SparkDsv4RocmCheckLinearView(w1_gate,
                                          moe_intermediate_dimension,
                                          hidden_dimension);
    if (status != SPARK_HW_OK)
        return status;
    status = SparkDsv4RocmCheckLinearView(w3_up,
                                          moe_intermediate_dimension,
                                          hidden_dimension);
    if (status != SPARK_HW_OK)
        return status;
    status = SparkDsv4RocmCheckLinearView(w2_down, hidden_dimension,
                                          moe_intermediate_dimension);
    if (status != SPARK_HW_OK)
        return status;
    if (w1_gate->weight_format != w3_up->weight_format ||
        w1_gate->weight_format != w2_down->weight_format)
        return SPARK_HW_INVALID;
    if (w1_gate->scale_group != w3_up->scale_group ||
        w2_down->scale_group != w1_gate->scale_group)
        return SPARK_HW_INVALID;

    /* The swiglu kernel flattens row_count x width into one uint32_t
     * index, and the gate/up kernel doubles output_dimension in its bound
     * check; refuse anything that could wrap either. */
    if ((uint64_t)row_count * moe_intermediate_dimension >
            (uint64_t)UINT32_MAX ||
        moe_intermediate_dimension > UINT32_MAX / 2u)
        return SPARK_HW_INVALID;

    if (row_count == 0u)
        return SPARK_HW_OK; /* empty frame: nothing to fork or join */

    format = w1_gate->weight_format;
    scale_group = w1_gate->scale_group;
    stream = (hipStream_t)(void *)aux_queue;

    gate_half = gate_up_scratch_bf16_device;
    up_half = (void *)((uint16_t *)gate_up_scratch_bf16_device +
                       (uint64_t)row_count * moe_intermediate_dimension);

    /* Fork leg: order after the producer event before any launch. */
    if (fork_event != 0)
    {
        hipError_t error = hipStreamWaitEvent(stream, (hipEvent_t)(void *)fork_event, 0u);
        status = spark_hw_rocm_map_status(error);
        if (status != SPARK_HW_OK)
            return status;
    }

    gate_up_grid_y = (moe_intermediate_dimension +
                      (SPARK_DSV4_ROCM_WARPS_PER_CTA - 1u)) /
                     SPARK_DSV4_ROCM_WARPS_PER_CTA;
    down_grid_y = (hidden_dimension + (SPARK_DSV4_ROCM_WARPS_PER_CTA - 1u)) /
                  SPARK_DSV4_ROCM_WARPS_PER_CTA;
    status = SparkDsv4RocmCheckGridY(gate_up_grid_y);
    if (status != SPARK_HW_OK)
        return status;
    status = SparkDsv4RocmCheckGridY(down_grid_y);
    if (status != SPARK_HW_OK)
        return status;

    /* 1. gate/up projections into the scratch halves. */
    arguments[0] = (void *)&format;
    arguments[1] = (void *)&w1_gate->payload;
    arguments[2] = (void *)&w1_gate->scale_data;
    arguments[3] = (void *)&w3_up->payload;
    arguments[4] = (void *)&w3_up->scale_data;
    arguments[5] = (void *)&scale_group;
    arguments[6] = (void *)&normalized_hidden_bf16_device;
    arguments[7] = (void *)&gate_half;
    arguments[8] = (void *)&up_half;
    arguments[9] = (void *)&hidden_dimension;
    arguments[10] = (void *)&moe_intermediate_dimension;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmSharedGateUpKernel, row_count,
        gate_up_grid_y, SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
    if (status != SPARK_HW_OK)
        return status;

    /* 2. clamped swiglu, activated overwrites the gate half in place. */
    {
        uint32_t total_elements = row_count * moe_intermediate_dimension;
        uint32_t blocks = (total_elements + SPARK_DSV4_ROCM_CTA_THREADS - 1u) /
                          SPARK_DSV4_ROCM_CTA_THREADS;
        void *swiglu_arguments[5];
        swiglu_arguments[0] = (void *)&gate_half;
        swiglu_arguments[1] = (void *)&up_half;
        swiglu_arguments[2] = (void *)&row_count;
        swiglu_arguments[3] = (void *)&moe_intermediate_dimension;
        swiglu_arguments[4] = (void *)&swiglu_limit;
        status = SparkDsv4RocmLaunch(
            (const void *)&SparkDsv4RocmSharedSwigluClampKernel, blocks, 1u,
            SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, swiglu_arguments);
        if (status != SPARK_HW_OK)
            return status;
    }

    /* 3. down projection added into the FFN accumulator. */
    arguments[0] = (void *)&format;
    arguments[1] = (void *)&w2_down->payload;
    arguments[2] = (void *)&w2_down->scale_data;
    arguments[3] = (void *)&scale_group;
    arguments[4] = (void *)&gate_half;
    arguments[5] = (void *)&ffn_accum_bf16_device;
    arguments[6] = (void *)&moe_intermediate_dimension;
    arguments[7] = (void *)&hidden_dimension;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmSharedDownAccumKernel, row_count,
        down_grid_y, SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
    if (status != SPARK_HW_OK)
        return status;

    /* Join leg: the pre-hcPost consumer waits this event. */
    if (join_event != 0)
    {
        hipError_t error = hipEventRecord((hipEvent_t)(void *)join_event, stream);
        status = spark_hw_rocm_map_status(error);
        if (status != SPARK_HW_OK)
            return status;
    }
    return SPARK_HW_OK;
}

SparkHwStatus spark_dsv4_rocm_l4_moe_routed(
    SparkHwQueue queue, SparkHwEvent fork_event, SparkHwEvent join_event,
    SparkHwMemory memory, const void *normalized_hidden_bf16_device,
    const uint32_t *route_indices_u32_device,
    const float *route_weights_f32_device,
    const SparkDsv4RocmLinearView *w13_gate,
    const SparkDsv4RocmLinearView *w13_up,
    const SparkDsv4RocmLinearView *w2_down, float swiglu_limit,
    void *grouped_source_token_u32_device,
    void *grouped_expert_u32_device, void *expert_offsets_u32_device,
    void *inverse_pair_u32_device, void *up_scratch_bf16_device,
    void *pair_out_scratch_bf16_device, void *ffn_accum_bf16_device,
    uint32_t row_count, uint32_t expert_count, uint32_t experts_per_token,
    uint32_t hidden_dimension, uint32_t moe_intermediate_dimension)
{
    hipStream_t stream;
    SparkHwStatus status;
    uint32_t format, scale_group, pair_count;
    uint32_t w13_grid_y, down_grid_y, reduce_blocks;
    void *arguments[13];

    if (memory != 0)
    {
        /* Declared handle; bad handles fail closed exactly like E0/L5. */
        status = spark_hw_rocm_check_memory(&memory);
        if (status != SPARK_HW_OK)
            return status;
    }

    if (queue == 0 || normalized_hidden_bf16_device == 0 ||
        route_indices_u32_device == 0 || grouped_source_token_u32_device == 0 ||
        grouped_expert_u32_device == 0 || expert_offsets_u32_device == 0 ||
        inverse_pair_u32_device == 0 || up_scratch_bf16_device == 0 ||
        pair_out_scratch_bf16_device == 0 || ffn_accum_bf16_device == 0 ||
        expert_count == 0u || experts_per_token == 0u ||
        hidden_dimension == 0u || moe_intermediate_dimension == 0u)
        return SPARK_HW_INVALID;
    if (swiglu_limit != swiglu_limit) /* NaN limit refuses */
        return SPARK_HW_INVALID;
    /* Flat route element indexing and every grid dimension stay inside
     * uint32; a larger sealed batch is a capacity refusal (EXHAUSTED), not
     * an argument error. */
    if ((uint64_t)row_count * experts_per_token > (uint64_t)UINT32_MAX ||
        row_count > (uint32_t)0x7fffffffu /
                        (experts_per_token != 0u ? experts_per_token : 1u))
        return SPARK_HW_EXHAUSTED;
    /* The pair reduce flattens row x hidden into one uint32 lane index. */
    if ((uint64_t)row_count * hidden_dimension > (uint64_t)UINT32_MAX)
        return SPARK_HW_EXHAUSTED;
    pair_count = row_count * experts_per_token;
    /* MXFP4 word runs index hidden in 8-element steps; the pack geometry
     * invariant (dimensions are multiples of 64) keeps every row base
     * word-aligned. Refuse rather than fault on a misaligned run. */
    if ((hidden_dimension % 8u) != 0u)
        return SPARK_HW_INVALID;

    /* Stacked-expert row counts must fit the view's uint32 rows field. */
    if ((uint64_t)expert_count * moe_intermediate_dimension >
            (uint64_t)UINT32_MAX ||
        (uint64_t)expert_count * hidden_dimension > (uint64_t)UINT32_MAX)
        return SPARK_HW_INVALID;

    status = SparkDsv4RocmCheckLinearView(
        w13_gate, (uint32_t)((uint64_t)expert_count *
                             moe_intermediate_dimension),
        hidden_dimension);
    if (status != SPARK_HW_OK)
        return status;
    status = SparkDsv4RocmCheckLinearView(
        w13_up,
        (uint32_t)((uint64_t)expert_count * moe_intermediate_dimension),
        hidden_dimension);
    if (status != SPARK_HW_OK)
        return status;
    status = SparkDsv4RocmCheckLinearView(
        w2_down, (uint32_t)((uint64_t)expert_count * hidden_dimension),
        moe_intermediate_dimension);
    if (status != SPARK_HW_OK)
        return status;
    format = w13_gate->weight_format;
    scale_group = w13_gate->scale_group;
    if (w13_up->weight_format != format || w2_down->weight_format != format)
        return SPARK_HW_INVALID;
    if (w13_up->scale_group != scale_group ||
        w2_down->scale_group != scale_group)
        return SPARK_HW_INVALID;
    /* MXFP4 scale-block straddle rule lives in SparkDsv4RocmCheckLinearView
     * so every island entry enforces it identically. */

    stream = (hipStream_t)(void *)queue;

    /* Fork leg: order after the producer fork point before any launch. */
    if (fork_event != 0)
    {
        hipError_t error = hipStreamWaitEvent(
            stream, (hipEvent_t)(void *)fork_event, 0u);
        status = spark_hw_rocm_map_status(error);
        if (status != SPARK_HW_OK)
            return status;
    }

    /* 1. Route realization. Runs even on an empty frame so the declared
     * C2 output buffer always holds canonical offsets (+ total = 0). */
    arguments[0] = (void *)&route_indices_u32_device;
    arguments[1] = (void *)&grouped_source_token_u32_device;
    arguments[2] = (void *)&grouped_expert_u32_device;
    arguments[3] = (void *)&expert_offsets_u32_device;
    arguments[4] = (void *)&inverse_pair_u32_device;
    arguments[5] = (void *)&pair_count;
    arguments[6] = (void *)&experts_per_token;
    arguments[7] = (void *)&expert_count;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmRouteGroupKernel, 1u, 1u,
        SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
    if (status != SPARK_HW_OK)
        return status;

    if (pair_count == 0u)
    {
        /* Empty frame: offsets published, no GEMM/reduce work exists. The
         * join event still records so the consumer's wait is well-formed. */
        if (join_event != 0)
        {
            hipError_t error = hipEventRecord(
                (hipEvent_t)(void *)join_event, stream);
            status = spark_hw_rocm_map_status(error);
            if (status != SPARK_HW_OK)
                return status;
        }
        return SPARK_HW_OK;
    }

    /* route_weights is consumed by the reduce only when pairs exist. */
    if (route_weights_f32_device == 0)
        return SPARK_HW_INVALID;

    /* 2. Grouped W13 gate/up + clamped swiglu into the up scratch. */
    w13_grid_y = (moe_intermediate_dimension +
                  (SPARK_DSV4_ROCM_WARPS_PER_CTA - 1u)) /
                 SPARK_DSV4_ROCM_WARPS_PER_CTA;
    status = SparkDsv4RocmCheckGridY(w13_grid_y);
    if (status != SPARK_HW_OK)
        return status;
    arguments[0] = (void *)&format;
    arguments[1] = (void *)&w13_gate->payload;
    arguments[2] = (void *)&w13_gate->scale_data;
    arguments[3] = (void *)&w13_up->payload;
    arguments[4] = (void *)&w13_up->scale_data;
    arguments[5] = (void *)&scale_group;
    arguments[6] = (void *)&normalized_hidden_bf16_device;
    arguments[7] = (void *)&grouped_source_token_u32_device;
    arguments[8] = (void *)&grouped_expert_u32_device;
    arguments[9] = (void *)&up_scratch_bf16_device;
    arguments[10] = (void *)&swiglu_limit;
    arguments[11] = (void *)&hidden_dimension;
    arguments[12] = (void *)&moe_intermediate_dimension;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmExpertW13SwigluKernel, pair_count,
        w13_grid_y, SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
    if (status != SPARK_HW_OK)
        return status;

    /* 3. The routed FFN down projection: grouped W2, unweighted rows. */
    down_grid_y = (hidden_dimension + (SPARK_DSV4_ROCM_WARPS_PER_CTA - 1u)) /
                  SPARK_DSV4_ROCM_WARPS_PER_CTA;
    status = SparkDsv4RocmCheckGridY(down_grid_y);
    if (status != SPARK_HW_OK)
        return status;
    arguments[0] = (void *)&format;
    arguments[1] = (void *)&w2_down->payload;
    arguments[2] = (void *)&w2_down->scale_data;
    arguments[3] = (void *)&scale_group;
    arguments[4] = (void *)&up_scratch_bf16_device;
    arguments[5] = (void *)&grouped_expert_u32_device;
    arguments[6] = (void *)&pair_out_scratch_bf16_device;
    arguments[7] = (void *)&hidden_dimension;
    arguments[8] = (void *)&moe_intermediate_dimension;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmExpertDownKernel, pair_count,
        down_grid_y, SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
    if (status != SPARK_HW_OK)
        return status;

    /* 4. Weighted pair reduce accumulating into the FFN accumulator. */
    reduce_blocks =
        (uint32_t)(((uint64_t)row_count * hidden_dimension +
                    SPARK_DSV4_ROCM_CTA_THREADS - 1u) /
                   SPARK_DSV4_ROCM_CTA_THREADS);
    arguments[0] = (void *)&pair_out_scratch_bf16_device;
    arguments[1] = (void *)&inverse_pair_u32_device;
    arguments[2] = (void *)&route_weights_f32_device;
    arguments[3] = (void *)&ffn_accum_bf16_device;
    arguments[4] = (void *)&row_count;
    arguments[5] = (void *)&experts_per_token;
    arguments[6] = (void *)&hidden_dimension;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmMoePairReduceKernel, reduce_blocks, 1u,
        SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
    if (status != SPARK_HW_OK)
        return status;

    /* Join leg: the pre-hcPost consumer waits this event. */
    if (join_event != 0)
    {
        hipError_t error = hipEventRecord((hipEvent_t)(void *)join_event, stream);
        status = spark_hw_rocm_map_status(error);
        if (status != SPARK_HW_OK)
            return status;
    }
    return SPARK_HW_OK;
}
