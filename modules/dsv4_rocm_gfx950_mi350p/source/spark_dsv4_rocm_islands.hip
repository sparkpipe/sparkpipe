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
 * below are the only consumers.
 *
 * Two host renderings:
 *   - plain proof parse (default): all four coordinates alias one all-zero
 *     object; a parse never executes kernels.
 *   - SPARK_DSV4_ROCM_HOST_EXECUTOR: three DISTINCT globals the executor
 *     sets per launch/thread. Every kernel in this TU whose per-element work
 *     is thread-strided (each element wholly computed by one thread) then
 *     produces IDENTICAL final state under a single-thread walk with the
 *     real blockDim, which is exactly what tools/l3_c2_host_probe.cpp
 *     exercises for the C2 integer surfaces. Kernels whose results depend
 *     on cross-lane reductions are executor-excluded by design and say so
 *     at their dispatch site. */
typedef struct SparkDsv4RocmProbeCoordTag
{
    uint32_t x, y, z;
} SparkDsv4RocmProbeCoord;

#if !defined(SPARK_DSV4_ROCM_HOST_EXECUTOR)

static SparkDsv4RocmProbeCoord spark_dsv4_rocm_probe_coord[1] = {{0u, 0u, 0u}};

#define gridDim spark_dsv4_rocm_probe_coord[0]
#define blockIdx spark_dsv4_rocm_probe_coord[0]
#define blockDim spark_dsv4_rocm_probe_coord[0]
#define threadIdx spark_dsv4_rocm_probe_coord[0]

#else /* SPARK_DSV4_ROCM_HOST_EXECUTOR */

extern SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_grid;
extern SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_block;
extern SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_threadblock;
extern SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_thread;

#define gridDim spark_dsv4_rocm_exec_grid
#define blockIdx spark_dsv4_rocm_exec_block
#define blockDim spark_dsv4_rocm_exec_threadblock
#define threadIdx spark_dsv4_rocm_exec_thread

#endif /* SPARK_DSV4_ROCM_HOST_EXECUTOR */

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

static inline float SparkDsv4RocmProbeShflBroadcast(float value, int src_lane,
                                                    int width)
{
    (void)src_lane;
    (void)width;
    /* Host proof: every thread already holds the same value; the real
     * broadcast runs only under the HIP compiler. */
    return value;
}

#define __shfl_sync(mask, var, src_lane, width)                               \
    SparkDsv4RocmProbeShflBroadcast((var), (src_lane), (width))

/* Reciprocal square root is a device intrinsic; math.h only has sqrtf. The
 * two differ in ULPs but a host parse never executes kernels. */
#define rsqrtf(x) (1.0f / sqrtf((float)(x)))

/* Shared-memory declarations. Under hipcc these are the real storage classes
 * (extern dynamic array sized by the launch, fixed __shared__ arrays); a
 * host parse gets function-scoped statics sized to the largest dynamic
 * request any kernel in this TU makes, so every body type-checks unchanged.
 * Proof mode never executes device code, so the sizing is inert there. */
#define SPARK_DSV4_ROCM_SHARED_DYNAMIC(name, element_type)                    \
    static element_type name[8192]
#define SPARK_DSV4_ROCM_SHARED_FIXED(name, element_type, count)               \
    static element_type name[count]

static inline float SparkHwRocmBitsAsFloat(uint32_t bits)
{
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

#else /* !defined(__HIP__) */

#define SPARK_DSV4_ROCM_SHARED_DYNAMIC(name, element_type)                    \
    extern __shared__ element_type name[]
#define SPARK_DSV4_ROCM_SHARED_FIXED(name, element_type, count)               \
    __shared__ element_type name[count]

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

// fp32 -> E2M1 nibble value (as float), round-to-nearest-even at every
// midpoint with the mantissa-zero neighbour winning ties - the shared
// encoder's exact table walk.
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmEncodeE2m1(float value)
{
    float points[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float ties[7] = {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 4.0f, 4.0f};
    float magnitude = fabsf(value);
    float sign = value < 0.0f ? -1.0f : 1.0f;
    uint32_t index;
    if (magnitude >= 6.0f)
        return sign * 6.0f;
    for (index = 0u; index < 7u; ++index)
    {
        float midpoint = (points[index] + points[index + 1u]) * 0.5f;
        if (magnitude < midpoint)
            return sign * points[index];
        if (magnitude == midpoint)
            return sign * ties[index];
    }
    return sign * 6.0f;
}

// fp32 -> E4M3fn byte, round-to-nearest-even, saturating-finite: the software
// mirror of the PTX cvt.rn.satfinite.e4m3x2.f32 conversion the CUDA target's
// sim-quant uses (dtype.cuh LmFloatToE4m3). 0x7f stays NaN-only; saturation
// lands on 0x7e (=448). Decode side is SparkDsv4RocmDecodeE4m3 above.
SPARK_DSV4_ROCM_DEV uint8_t SparkDsv4RocmEncodeE4m3Satfinite(float value)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    uint32_t sign_byte = (bits >> 24u) & 0x80u;
    uint32_t magnitude = bits & 0x7fffffffu;
    int32_t unbiased, exponent;
    uint32_t mantissa, keep, round_bit, sticky, q;
    if (magnitude > 0x7f800000u)
        return (uint8_t)(sign_byte | 0x7fu); /* NaN encodings propagate */
    if (magnitude >= 0x7f800000u)
        return (uint8_t)(sign_byte | 0x7eu); /* infinity saturates */
    if (magnitude == 0u || ((magnitude >> 23u) & 0xffu) == 0u)
        return (uint8_t)sign_byte; /* zero and float subnormals -> zero */
    unbiased = (int32_t)((magnitude >> 23u) & 0xffu) - 127;
    mantissa = (magnitude & 0x007fffffu) | 0x00800000u;
    if (unbiased < -6)
    {
        /* Subnormal grid first, from the ORIGINAL exponent: the 3-bit
         * rounding below carries into the exponent and would corrupt the
         * grid selection here (found by exhaustive boundary sweep). */
        uint32_t base, rb, st, s;
        int32_t shift = 14 - unbiased;
        if (shift > 31)
            return (uint8_t)sign_byte;
        base = mantissa >> shift;
        rb = (mantissa >> (shift - 1)) & 1u;
        st = (mantissa & ((1u << (shift - 1)) - 1u)) != 0u ? 1u : 0u;
        s = base + ((rb != 0u && (st != 0u || (base & 1u) != 0u)) ? 1u : 0u);
        if (s == 16u) /* rounded up into the smallest normal 2^-6 */
            return (uint8_t)(sign_byte | (1u << 3));
        return (uint8_t)(sign_byte | s);
    }
    exponent = unbiased;
    /* Round the implicit-1 mantissa to 3 fractional bits, ties to even. */
    keep = mantissa >> 20u;
    round_bit = (mantissa >> 19u) & 1u;
    sticky = (mantissa & 0x0007ffffu) != 0u ? 1u : 0u;
    q = keep + ((round_bit != 0u && (sticky != 0u || (keep & 1u) != 0u))
                    ? 1u
                    : 0u);
    if (q == 16u)
    {
        q = 8u;
        exponent += 1;
    }
    if (exponent > 8 || (exponent == 8 && q > 14))
        return (uint8_t)(sign_byte | 0x7eu); /* beyond 448 saturates */
    return (uint8_t)(sign_byte |
                     (((uint32_t)(exponent + 7)) << 3u) | (q - 8u));
    {
        /* Subnormal grid: multiples of 2^-9. The shift is always rightward
         * here but can exceed 31 bits for vanishing inputs - those are zero
         * under any rounding. */
        int32_t shift = -(exponent - 23 + 9);
        uint32_t base, rb, st, s;
        if (shift > 31)
            return (uint8_t)sign_byte;
        base = mantissa >> shift;
        rb = shift != 0 ? (mantissa >> (shift - 1)) & 1u : 0u;
        st = shift != 0 ? ((mantissa & ((1u << (shift - 1)) - 1u)) != 0u
                              ? 1u
                              : 0u)
                        : 0u;
        s = base + ((rb != 0u && (st != 0u || (base & 1u) != 0u)) ? 1u : 0u);
        if (s == 16u) /* rounded up into the smallest normal 2^-6 */
            return (uint8_t)(sign_byte | (1u << 3));
        return (uint8_t)(sign_byte | s);
    }
}

// Per-block power-of-two scale: exp2(ceil(log2(amax / format_max))), the
// shared sim-quant scale rule.
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmPow2CeilScale(float amax,
                                                      float format_max)
{
    return exp2f(ceilf(log2f(amax / format_max)));
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

// Block sum over blockDim.x threads; scratch holds WARPS_PER_CTA floats.
// Wave64 rendering of the shared SparkLmBlockReduceSum: same two-level
// structure, butterfly width 64. The reduction ORDER is part of this
// archive's fixed C3 tree (documented once, never tuned per bucket).
SPARK_DSV4_ROCM_DEV float SparkDsv4RocmBlockReduceSum(float value,
                                                      float *scratch)
{
    uint32_t lane = threadIdx.x % SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t warp = threadIdx.x / SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t warp_count =
        (blockDim.x + SPARK_DSV4_ROCM_WAVE_LANES - 1u) /
        SPARK_DSV4_ROCM_WAVE_LANES;
    value = SparkDsv4RocmWaveReduceSum(value);
    if (lane == 0u)
        scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < warp_count ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0u)
        value = SparkDsv4RocmWaveReduceSum(value);
    if (threadIdx.x == 0u)
        scratch[0] = value;
    __syncthreads();
    value = scratch[0];
    __syncthreads();
    return value;
}

// In-place block quantize-dequantize sim over one block of a row: per-block
// amax floored at 1e-4, power-of-two scale, clamp to +/-format_max, snap
// (E2M1 table walk for fp4 / E4M3 RNE round trip for fp8), rescale, RNE bf16
// store. One warp per (row, block); lanes stride block elements. amax is a
// max reduction - order-free and exact - so the wider gfx950 wavefront does
// not change a single output bit against the CUDA target here.
SPARK_DSV4_ROCM_DEV void SparkDsv4RocmQuantSimGroup(
    void *data_bf16, uint32_t row, uint32_t group, uint32_t lane,
    uint32_t row_stride, uint32_t width, uint32_t block, float format_max,
    uint32_t fp4)
{
    uint16_t *words = (uint16_t *)data_bf16;
    uint32_t base = group * block;
    uint32_t limit = base + block < width ? base + block : width;
    uint64_t offset = ((uint64_t)row * row_stride);
    uint32_t element;
    float value, amax = 1e-4f, scale;
    if (base >= width)
        return;
    for (element = base + lane; element < limit;
         element += SPARK_DSV4_ROCM_WAVE_LANES)
    {
        value = fabsf(SparkDsv4RocmBf16ToFloat(words[offset + element]));
        if (value > amax)
            amax = value;
    }
    for (element = SPARK_DSV4_ROCM_WAVE_LANES / 2u; element != 0u;
         element >>= 1u)
    {
        value = __shfl_down_sync(0xffffffffffffffffull, amax, (int)element,
                                 (int)SPARK_DSV4_ROCM_WAVE_LANES);
        if (value > amax)
            amax = value;
    }
    amax = __shfl_sync(0xffffffffffffffffull, amax, 0,
                       (int)SPARK_DSV4_ROCM_WAVE_LANES);
    scale = SparkDsv4RocmPow2CeilScale(amax, format_max);
    for (element = base + lane; element < limit;
         element += SPARK_DSV4_ROCM_WAVE_LANES)
    {
        value = SparkDsv4RocmBf16ToFloat(words[offset + element]) / scale;
        if (value > format_max)
            value = format_max;
        if (value < -format_max)
            value = -format_max;
        value = (fp4 != 0u
                     ? SparkDsv4RocmEncodeE2m1(value)
                     : SparkDsv4RocmDecodeE4m3(
                           SparkDsv4RocmEncodeE4m3Satfinite(value))) *
                scale;
        words[offset + element] = SparkDsv4RocmFloatToBf16Rne(value);
    }
}

// RMS norm in place over one bf16 row with a separate gain vector: sum of
// squares staged in dynamic shared memory, block-reduced on the fixed tree,
// rsqrt(mean + epsilon), scaled gain product stored back as bf16 pairs.
// Mirrors the shared SparkLmRmsNormRow including its odd-dimension tail.
SPARK_DSV4_ROCM_DEV void SparkDsv4RocmRmsNormRowInPlace(
    void *data_bf16, const void *gain_bf16, uint32_t row, uint32_t dimension,
    float epsilon, float *staged_input, float *reduce_scratch)
{
    const uint16_t *input_words = (const uint16_t *)data_bf16;
    const uint16_t *gain_words = (const uint16_t *)gain_bf16;
    uint16_t *output_words = (uint16_t *)data_bf16;
    uint64_t row_offset = ((uint64_t)row * (uint64_t)dimension);
    uint32_t element;
    float sum_squares = 0.0f, inverse_rms;
    for (element = threadIdx.x; element < dimension; element += blockDim.x)
    {
        /* fmaf, not *+: the shared norm never rounds the products before
         * accumulating - part of the mirrored arithmetic contract. */
        float value = SparkDsv4RocmBf16ToFloat(input_words[row_offset + element]);
        staged_input[element] = value;
        sum_squares = fmaf(value, value, sum_squares);
    }
    sum_squares = SparkDsv4RocmBlockReduceSum(sum_squares, reduce_scratch);
    inverse_rms = rsqrtf((sum_squares / (float)dimension) + epsilon);
    __syncthreads();
    for (element = threadIdx.x; element < dimension; element += blockDim.x)
    {
        float gained = (staged_input[element] * inverse_rms) *
                       SparkDsv4RocmBf16ToFloat(gain_words[element]);
        output_words[row_offset + element] =
            SparkDsv4RocmFloatToBf16Rne(gained);
    }
}

// Adjacent-pair rotation on the LAST rope_dimension entries of a single-head
// row; inverse conjugates the sine. Pairs are adjacent elements, so both
// halves stay inside one lane's iteration - fully deterministic.
SPARK_DSV4_ROCM_DEV void SparkDsv4RocmRopePairs(
    void *data_bf16, const float *freqs_f32, const uint64_t *row_positions,
    uint32_t row, uint32_t position_slot, uint32_t width, uint32_t rope_dim,
    uint32_t inverse)
{
    uint16_t *words = (uint16_t *)data_bf16;
    uint32_t pair;
    uint64_t base;
    float angle, cosine, sine, real, imaginary;
    pair = threadIdx.x;
    if (pair >= rope_dim / 2u)
        return;
    base = ((uint64_t)row * width) + (width - rope_dim) + (2u * pair);
    angle = (float)row_positions[position_slot] * freqs_f32[pair];
    cosine = cosf(angle);
    sine = inverse != 0u ? -sinf(angle) : sinf(angle);
    real = SparkDsv4RocmBf16ToFloat(words[base]);
    imaginary = SparkDsv4RocmBf16ToFloat(words[base + 1u]);
    words[base] = SparkDsv4RocmFloatToBf16Rne(real * cosine - imaginary * sine);
    words[base + 1u] =
        SparkDsv4RocmFloatToBf16Rne(real * sine + imaginary * cosine);
}

// In-place Hadamard rotation scaled n^-0.5 on power-of-two rows; the whole
// row is staged in shared memory. Every stage touches disjoint pairs with
// pure +/- arithmetic, so the transform is bit-exact regardless of wave
// width or scheduling.
SPARK_DSV4_ROCM_DEV void SparkDsv4RocmHadamardRow(
    void *data_bf16, uint32_t vector_index, uint32_t width,
    float *hadamard_shared)
{
    const uint16_t *words = (const uint16_t *)data_bf16;
    uint16_t *out_words = (uint16_t *)data_bf16;
    uint64_t base = (uint64_t)vector_index * width;
    float scale = rsqrtf((float)width);
    uint32_t element, half, partner;
    float a, b;
    for (element = threadIdx.x; element < width; element += blockDim.x)
        hadamard_shared[element] = SparkDsv4RocmBf16ToFloat(words[base + element]);
    __syncthreads();
    for (half = 1u; half < width; half <<= 1u)
    {
        for (element = threadIdx.x; element < width / 2u; element += blockDim.x)
        {
            partner = ((element / half) * half * 2u) + (element % half);
            a = hadamard_shared[partner];
            b = hadamard_shared[partner + half];
            hadamard_shared[partner] = a + b;
            hadamard_shared[partner + half] = a - b;
        }
        __syncthreads();
    }
    for (element = threadIdx.x; element < width; element += blockDim.x)
        out_words[base + element] =
            SparkDsv4RocmFloatToBf16Rne(hadamard_shared[element] * scale);
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
// L3 `layer.cache_transition` kernels - the ring-page cache transition.
//
// Split verdict (freeze F1): page addresses, ring indices and emit counters
// are C2 integer-pure (they depend only on positions, lane indices, base
// slots, ratios - never on an L2 float score). Emitted payload bytes are a
// verbatim bf16 copy on the direct path (C2-eligible: copied unrounded) or
// quantized sim output on the emission path (C3, recipe-time field table).
// ---------------------------------------------------------------------------

// One boundary-predicated CTA per row scatters verbatim bf16 rows into the
// lane's ring: slot = ratio != 0 ? base + ((position % ring_slots)/ratio)
// : position % ring_slots. Pure integer addressing + byte moves; when
// ring_slot_out is non-NULL thread 0 also publishes the computed slot as an
// explicit C2 observable.
__global__ void SparkDsv4RocmCacheScatterKernel(
    const void *source_bf16, const uint32_t *emitted_flags_u32,
    void *cache_pages_bf16, uint64_t cache_lane_stride_elements,
    const uint32_t *row_lane_indices_u32, const uint64_t *row_positions_u64,
    uint32_t row_count, uint32_t width, uint64_t base_slot, uint32_t ratio,
    uint32_t ring_slots, void *ring_slot_out_u32)
{
    const uint16_t *source_words = (const uint16_t *)source_bf16;
    uint16_t *cache_words = (uint16_t *)cache_pages_bf16;
    uint32_t row = blockIdx.x, element;
    uint64_t position, slot, destination;
    if (row >= row_count || (ratio != 0u && emitted_flags_u32[row] == 0u))
        return;
    position = row_positions_u64[row];
    slot = ratio != 0u ? base_slot + ((position % ring_slots) / ratio)
                       : position % ring_slots;
    destination = ((uint64_t)row_lane_indices_u32[row] *
                   cache_lane_stride_elements) +
                  (slot * width);
    for (element = threadIdx.x; element < width; element += blockDim.x)
        cache_words[destination + element] =
            source_words[((uint64_t)row * width) + element];
    if (ring_slot_out_u32 != 0 && threadIdx.x == 0u)
        ((uint32_t *)ring_slot_out_u32)[row] = (uint32_t)slot;
}

// One CTA owns each live lane and advances its rows in packet order, so
// lanes parallelize without racing a lane's recurrent compressor state.
// Per row: ring slot = position % ratio, boundary = (position+1) % ratio
// == 0 (the INTEGER emission decision), state write with APE add, softmax
// pool over the group window on boundaries (plain or 2x overlapped), bf16
// emit row, emitted flag, optional per-lane emit counter increment, and the
// overlap window shift. Pool arithmetic mirrors the CUDA helpers exactly:
// same slot order, expf, and -3.0e38f sentinel.
__global__ void SparkDsv4RocmCompressStepKernel(
    const void *kv_bf16, const void *score_bf16, const float *ape_f32,
    float *kv_state_f32, float *score_state_f32,
    uint64_t state_lane_stride_elements, const uint32_t *row_lane_indices_u32,
    const uint64_t *row_positions_u64, uint32_t row_count, uint32_t ratio,
    uint32_t overlapped, uint32_t width, void *emit_bf16,
    uint32_t *emitted_flags_u32, uint32_t *emit_counters_u32)
{
    const uint16_t *kv_words = (const uint16_t *)kv_bf16;
    const uint16_t *score_words = (const uint16_t *)score_bf16;
    uint16_t *emit_words = (uint16_t *)emit_bf16;
    uint32_t row = blockIdx.x;
    uint32_t coff = overlapped != 0u ? 2u : 1u;
    uint32_t channels = coff * width;
    uint32_t channel, slot, boundary, lane, previous;
    uint64_t ape_base, source, state_base;
    float *kv_state, *score_state;
    if (row >= row_count)
        return;
    lane = row_lane_indices_u32[row];
    /* Only the FIRST packet row of each lane runs the walk (CUDA parity). */
    for (previous = 0u; previous < row; ++previous)
        if (row_lane_indices_u32[previous] == lane)
            return;
    state_base = (uint64_t)lane * state_lane_stride_elements;
    kv_state = kv_state_f32 + state_base;
    score_state = score_state_f32 + state_base;
    for (; row < row_count; ++row)
    {
        if (row_lane_indices_u32[row] != lane)
            continue;
        slot = (uint32_t)(row_positions_u64[row] % ratio);
        boundary = (row_positions_u64[row] + 1u) % ratio == 0u ? 1u : 0u;
        ape_base = (row_positions_u64[row] % ratio) * channels;
        for (channel = threadIdx.x; channel < channels; channel += blockDim.x)
        {
            source = (uint64_t)row * channels + channel;
            kv_state[((overlapped != 0u ? ratio : 0u) + slot) * channels +
                     channel] =
                SparkDsv4RocmBf16ToFloat(kv_words[source]);
            score_state[((overlapped != 0u ? ratio : 0u) + slot) * channels +
                        channel] =
                SparkDsv4RocmBf16ToFloat(score_words[source]) +
                ape_f32[ape_base + channel];
        }
        __syncthreads();
        if (threadIdx.x == 0u)
            emitted_flags_u32[row] = boundary;
        for (channel = threadIdx.x; channel < width; channel += blockDim.x)
        {
            /* Softmax pool of one channel over the group's slots: -inf
             * scores drop out; overlap reads both windows. */
            float maximum = -3.0e38f, total = 0.0f, value = 0.0f;
            float pooled = 0.0f, weight, score;
            uint32_t pool_slot, source_index, pool_slots;
            if (boundary != 0u)
            {
                pool_slots = overlapped != 0u ? 2u * ratio : ratio;
                for (pool_slot = 0u; pool_slot < pool_slots; ++pool_slot)
                {
                    source_index =
                        overlapped != 0u
                            ? (pool_slot < ratio
                                   ? pool_slot * channels + channel
                                   : pool_slot * channels + width + channel)
                            : pool_slot * channels + channel;
                    if (score_state[source_index] > maximum)
                        maximum = score_state[source_index];
                }
                for (pool_slot = 0u; pool_slot < pool_slots; ++pool_slot)
                {
                    source_index =
                        overlapped != 0u
                            ? (pool_slot < ratio
                                   ? pool_slot * channels + channel
                                   : pool_slot * channels + width + channel)
                            : pool_slot * channels + channel;
                    score = score_state[source_index];
                    if (score <= -3.0e38f)
                        continue;
                    weight = expf(score - maximum);
                    total += weight;
                    value += weight * kv_state[source_index];
                }
                pooled = value / total;
            }
            emit_words[(uint64_t)row * width + channel] =
                SparkDsv4RocmFloatToBf16Rne(pooled);
        }
        __syncthreads();
        if (threadIdx.x == 0u && boundary != 0u)
        {
            /* Emit counters are integer atomics: the TOTAL is exact and
             * run-invariant regardless of arrival order (C2). */
            if (emit_counters_u32 != 0)
                atomicAdd(&emit_counters_u32[lane], 1u);
        }
        if (overlapped != 0u && boundary != 0u)
        {
            /* Slide the overlap window forward by one group. */
            for (channel = threadIdx.x; channel < ratio * channels;
                 channel += blockDim.x)
            {
                kv_state[channel] = kv_state[ratio * channels + channel];
                score_state[channel] = score_state[ratio * channels + channel];
            }
        }
        __syncthreads();
    }
}

// One boundary-predicated CTA per emitting row replaces five post-compressor
// launches: RMS norm -> RoPE -> optional Hadamard rotate -> fp8/fp4 sim-quant
// groups -> ring scatter. Every bf16 materialization boundary of the CUDA
// kernel is retained in order, so quantized payloads stay inside the shared
// C3 tolerance set by construction.
__global__ void SparkDsv4RocmKvEmissionKernel(
    void *emit_bf16, const uint32_t *emitted_flags_u32,
    const void *norm_weight_bf16, const float *freqs_f32,
    const uint64_t *row_emit_positions_u64, void *cache_pages_bf16,
    uint64_t cache_lane_stride_elements, const uint32_t *row_lane_indices_u32,
    const uint64_t *row_positions_u64, uint32_t row_count, uint32_t width,
    uint64_t base_slot, uint32_t ratio, uint32_t ring_slots,
    uint32_t rotate_enable, uint32_t rope_dimension, float rms_epsilon,
    uint32_t quant_block, float quant_format_max, void *ring_slot_out_u32)
{
    SPARK_DSV4_ROCM_SHARED_DYNAMIC(emission_shared, float);
    SPARK_DSV4_ROCM_SHARED_FIXED(reduce_scratch, float,
                                 SPARK_DSV4_ROCM_WARPS_PER_CTA);
    uint32_t row = blockIdx.x;
    uint32_t warp = threadIdx.x / SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t lane = threadIdx.x % SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t group, group_count, quant_width;
    float format_max;
    if (row >= row_count || (ratio != 0u && emitted_flags_u32[row] == 0u))
        return;
    SparkDsv4RocmRmsNormRowInPlace(emit_bf16, norm_weight_bf16, row, width,
                                   rms_epsilon, emission_shared,
                                   reduce_scratch);
    __syncthreads();
    SparkDsv4RocmRopePairs(emit_bf16, freqs_f32, row_emit_positions_u64, row,
                           row, width, rope_dimension, 0u);
    __syncthreads();
    if (rotate_enable != 0u)
        SparkDsv4RocmHadamardRow(emit_bf16, row, width, emission_shared);
    __syncthreads();
    /* CUDA parity: rotated (fp4-class) path quantizes the full width;
     * unrotated path leaves the rope tail unquantized. Format max comes in
     * per call (448 fp8 / 6 fp4 class), snap selects via rotate. */
    quant_width = rotate_enable != 0u ? width : width - rope_dimension;
    format_max = quant_format_max;
    group_count = (quant_width + quant_block - 1u) / quant_block;
    /* The shared kernel strides groups across its 32-lane warps; here the
     * four wave64 warps take the same role. Group coverage is identical and
     * each group is self-contained, so results are unchanged. */
    for (group = warp; group < group_count;
         group += SPARK_DSV4_ROCM_WARPS_PER_CTA)
        SparkDsv4RocmQuantSimGroup(emit_bf16, row, group, lane, width, width,
                                   quant_block, format_max, rotate_enable);
    __syncthreads();
    {
        /* Ring scatter of the quantized emit row (verbatim bytes at this
         * stage - rounding already happened in the sim). */
        const uint16_t *source_words = (const uint16_t *)emit_bf16;
        uint16_t *cache_words = (uint16_t *)cache_pages_bf16;
        uint64_t position, slot, destination;
        uint32_t element;
        position = row_positions_u64[row];
        slot = ratio != 0u ? base_slot + ((position % ring_slots) / ratio)
                           : position % ring_slots;
        destination = ((uint64_t)row_lane_indices_u32[row] *
                       cache_lane_stride_elements) +
                      (slot * width);
        for (element = threadIdx.x; element < width; element += blockDim.x)
            cache_words[destination + element] =
                source_words[((uint64_t)row * width) + element];
        if (ring_slot_out_u32 != 0 && threadIdx.x == 0u)
            ((uint32_t *)ring_slot_out_u32)[row] = (uint32_t)slot;
    }
}

// New-page initialization: copy the parent page's bytes (root pages zero),
// then poison the declared score spans with -inf so the first attention read
// never trusts uninitialized pool words. Byte semantics - C2.
__global__ void SparkDsv4RocmInitializePagesKernel(
    uint32_t *page_pool_u32, uint64_t page_stride_words,
    const uint32_t *page_indices_u32, const uint32_t *parent_page_indices_u32,
    uint32_t page_count, const uint64_t *score_span_offset_words_u64,
    const uint64_t *score_span_element_count_u64, uint32_t score_span_count)
{
    float *score_base;
    uint32_t page, parent, span_index;
    uint64_t destination, source, word;
    if (blockIdx.x >= page_count)
        return;
    page = page_indices_u32[blockIdx.x];
    if (page == 0xffffffffu)
        return;
    parent = parent_page_indices_u32[blockIdx.x];
    destination = (uint64_t)page * page_stride_words;
    source = (uint64_t)parent * page_stride_words;
    for (word = threadIdx.x; word < page_stride_words; word += blockDim.x)
        page_pool_u32[destination + word] =
            parent == 0xffffffffu ? 0u : page_pool_u32[source + word];
    __syncthreads();
    if (parent != 0xffffffffu)
        return;
    score_base = (float *)(page_pool_u32 + destination);
    for (span_index = 0u; span_index < score_span_count; ++span_index)
        for (word = threadIdx.x;
             word < score_span_element_count_u64[span_index];
             word += blockDim.x)
            score_base[score_span_offset_words_u64[span_index] + word] =
                -INFINITY;
}

// Scattered physical-page-table updates: table[index[i]] = value[i]. Pure
// integer writes - the C2 page-address surface.
__global__ void SparkDsv4RocmUpdatePageTableKernel(
    uint32_t *page_table_u32, const uint32_t *update_indices_u32,
    const uint32_t *update_values_u32, uint32_t update_count)
{
    uint32_t update = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (update < update_count)
        page_table_u32[update_indices_u32[update]] =
            update_values_u32[update];
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

// ---------------------------------------------------------------------------
// L3 entries (C linkage, declared in spark_dsv4_rocm_islands.h)
// ---------------------------------------------------------------------------

SparkHwStatus spark_dsv4_rocm_l3_cache_transition(
    SparkHwQueue queue, SparkHwMemory memory,
    const void *compressor_kv_bf16_device,
    const void *compressor_score_bf16_device, const float *ape_f32_device,
    void *compress_kv_state_f32_device, void *compress_score_state_f32_device,
    uint64_t state_lane_stride_elements, uint32_t compress_ratio,
    uint32_t overlap_factor, uint32_t compress_width,
    const void *emission_norm_weight_bf16_device,
    const float *emission_freqs_f32_device,
    const uint64_t *row_emit_positions_u64_device, float rms_epsilon,
    uint32_t rope_dimension, uint32_t quant_block, float quant_format_max,
    uint32_t rotate_enable, void *emission_scratch_bf16_device,
    const uint32_t *row_lane_indices_u32_device,
    const uint64_t *row_positions_u64_device, uint32_t row_count,
    void *cache_pages_bf16_device, uint64_t cache_lane_stride_elements,
    uint64_t compressed_base_slot, uint32_t ring_slots,
    const void *direct_kv_bf16_device, uint32_t direct_width,
    void *emitted_flags_u32_device, void *emit_counters_u32_device,
    void *ring_slots_out_u32_device)
{
    hipStream_t stream;
    SparkHwStatus status;
    uint32_t compressor_active = compress_ratio != 0u ? 1u : 0u;
    uint32_t overlapped = overlap_factor > 1u ? 1u : 0u;

    if (memory != 0)
    {
        status = spark_hw_rocm_check_memory(&memory);
        if (status != SPARK_HW_OK)
            return status;
    }
    if (queue == 0 || row_lane_indices_u32_device == 0 ||
        row_positions_u64_device == 0 || cache_pages_bf16_device == 0 ||
        ring_slots == 0u || cache_lane_stride_elements == 0u)
        return SPARK_HW_INVALID;
    if (rms_epsilon != rms_epsilon || quant_format_max != quant_format_max)
        return SPARK_HW_INVALID; /* NaN geometry refuses */
    /* CUDA launcher parity, geometry-free renderings: rotate needs a
     * power-of-two width strictly above the rope span (the unquantized tail
     * rule); ratio addressing requires ring_slots to divide by ratio. */
    if (rotate_enable > 1u)
        return SPARK_HW_INVALID;
    if (compressor_active != 0u)
    {
        if (compressor_kv_bf16_device == 0 ||
            compressor_score_bf16_device == 0 || ape_f32_device == 0 ||
            compress_kv_state_f32_device == 0 ||
            compress_score_state_f32_device == 0 ||
            emission_norm_weight_bf16_device == 0 ||
            emission_freqs_f32_device == 0 ||
            row_emit_positions_u64_device == 0 ||
            emission_scratch_bf16_device == 0 ||
            emitted_flags_u32_device == 0 ||
            state_lane_stride_elements == 0u || compress_width == 0u ||
            quant_block == 0u || quant_format_max <= 0.0f ||
            overlap_factor == 0u || ring_slots % compress_ratio != 0u ||
            compress_width < rope_dimension ||
            (rotate_enable == 0u && compress_width == rope_dimension) ||
            (rotate_enable != 0u &&
             (compress_width & (compress_width - 1u)) != 0u))
            return SPARK_HW_INVALID;
    }
    else if (direct_kv_bf16_device == 0)
    {
        /* Nothing to transition. */
        return SPARK_HW_INVALID;
    }
    if (direct_kv_bf16_device != 0 && direct_width == 0u)
        return SPARK_HW_INVALID;
    if (row_count == 0u)
        return SPARK_HW_OK; /* empty frame: nothing to scatter or emit */

    stream = (hipStream_t)(void *)queue;

    /* Phase A: direct verbatim kv scatter (ratio-0 ring addressing). */
    if (direct_kv_bf16_device != 0)
    {
        void *scatter_arguments[12];
        uint64_t base_slot_zero = 0u;
        uint32_t ratio_zero = 0u;
        scatter_arguments[0] = (void *)&direct_kv_bf16_device;
        scatter_arguments[1] = (void *)&emitted_flags_u32_device;
        scatter_arguments[2] = (void *)&cache_pages_bf16_device;
        scatter_arguments[3] = (void *)&cache_lane_stride_elements;
        scatter_arguments[4] = (void *)&row_lane_indices_u32_device;
        scatter_arguments[5] = (void *)&row_positions_u64_device;
        scatter_arguments[6] = (void *)&row_count;
        scatter_arguments[7] = (void *)&direct_width;
        scatter_arguments[8] = (void *)&base_slot_zero;
        scatter_arguments[9] = (void *)&ratio_zero;
        scatter_arguments[10] = (void *)&ring_slots;
        scatter_arguments[11] = (void *)&ring_slots_out_u32_device;
        status = SparkDsv4RocmLaunch(
            (const void *)&SparkDsv4RocmCacheScatterKernel, row_count, 1u,
            SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, scatter_arguments);
        if (status != SPARK_HW_OK)
            return status;
    }

    if (compressor_active == 0u)
        return SPARK_HW_OK;

    /* Phase B: compressor state advance + integer emission decision +
     * pooled bf16 emit rows into the scratch. */
    {
        void *compress_arguments[15];
        compress_arguments[0] = (void *)&compressor_kv_bf16_device;
        compress_arguments[1] = (void *)&compressor_score_bf16_device;
        compress_arguments[2] = (void *)&ape_f32_device;
        compress_arguments[3] = (void *)&compress_kv_state_f32_device;
        compress_arguments[4] = (void *)&compress_score_state_f32_device;
        compress_arguments[5] = (void *)&state_lane_stride_elements;
        compress_arguments[6] = (void *)&row_lane_indices_u32_device;
        compress_arguments[7] = (void *)&row_positions_u64_device;
        compress_arguments[8] = (void *)&row_count;
        compress_arguments[9] = (void *)&compress_ratio;
        compress_arguments[10] = (void *)&overlapped;
        compress_arguments[11] = (void *)&compress_width;
        compress_arguments[12] = (void *)&emission_scratch_bf16_device;
        compress_arguments[13] = (void *)&emitted_flags_u32_device;
        compress_arguments[14] = (void *)&emit_counters_u32_device;
        status = SparkDsv4RocmLaunch(
            (const void *)&SparkDsv4RocmCompressStepKernel, row_count, 1u,
            SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, compress_arguments);
        if (status != SPARK_HW_OK)
            return status;
    }

    /* Phase C: emission post - RMS norm, RoPE, optional Hadamard rotate,
     * fp8/fp4 sim-quant groups, ring scatter of quantized rows. */
    {
        void *emission_arguments[20];
        uint32_t shared_bytes = compress_width * (uint32_t)sizeof(float);
        emission_arguments[0] = (void *)&emission_scratch_bf16_device;
        emission_arguments[1] = (void *)&emitted_flags_u32_device;
        emission_arguments[2] = (void *)&emission_norm_weight_bf16_device;
        emission_arguments[3] = (void *)&emission_freqs_f32_device;
        emission_arguments[4] = (void *)&row_emit_positions_u64_device;
        emission_arguments[5] = (void *)&cache_pages_bf16_device;
        emission_arguments[6] = (void *)&cache_lane_stride_elements;
        emission_arguments[7] = (void *)&row_lane_indices_u32_device;
        emission_arguments[8] = (void *)&row_positions_u64_device;
        emission_arguments[9] = (void *)&row_count;
        emission_arguments[10] = (void *)&compress_width;
        emission_arguments[11] = (void *)&compressed_base_slot;
        emission_arguments[12] = (void *)&compress_ratio;
        emission_arguments[13] = (void *)&ring_slots;
        emission_arguments[14] = (void *)&rotate_enable;
        emission_arguments[15] = (void *)&rope_dimension;
        emission_arguments[16] = (void *)&rms_epsilon;
        emission_arguments[17] = (void *)&quant_block;
        emission_arguments[18] = (void *)&quant_format_max;
        emission_arguments[19] = (void *)&ring_slots_out_u32_device;
        status = SparkDsv4RocmLaunch(
            (const void *)&SparkDsv4RocmKvEmissionKernel, row_count, 1u,
            SPARK_DSV4_ROCM_CTA_THREADS, shared_bytes, stream,
            emission_arguments);
        if (status != SPARK_HW_OK)
            return status;
    }
    return SPARK_HW_OK;
}

SparkHwStatus spark_dsv4_rocm_l3_initialize_pages(
    SparkHwQueue queue, SparkHwMemory memory, void *page_pool_u32_device,
    uint64_t page_stride_words, const uint32_t *page_indices_u32_device,
    const uint32_t *parent_page_indices_u32_device, uint32_t page_count,
    const uint64_t *score_span_offset_words_u64_device,
    const uint64_t *score_span_element_count_u64_device,
    uint32_t score_span_count)
{
    hipStream_t stream;
    SparkHwStatus status;
    void *arguments[8];

    if (memory != 0)
    {
        status = spark_hw_rocm_check_memory(&memory);
        if (status != SPARK_HW_OK)
            return status;
    }
    if (queue == 0 || page_pool_u32_device == 0 ||
        page_indices_u32_device == 0 || parent_page_indices_u32_device == 0 ||
        page_count == 0u || page_stride_words == 0u ||
        (score_span_count != 0u &&
         (score_span_offset_words_u64_device == 0 ||
          score_span_element_count_u64_device == 0)))
        return SPARK_HW_INVALID;

    stream = (hipStream_t)(void *)queue;
    arguments[0] = (void *)&page_pool_u32_device;
    arguments[1] = (void *)&page_stride_words;
    arguments[2] = (void *)&page_indices_u32_device;
    arguments[3] = (void *)&parent_page_indices_u32_device;
    arguments[4] = (void *)&page_count;
    arguments[5] = (void *)&score_span_offset_words_u64_device;
    arguments[6] = (void *)&score_span_element_count_u64_device;
    arguments[7] = (void *)&score_span_count;
    return SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmInitializePagesKernel, page_count, 1u,
        SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
}

SparkHwStatus spark_dsv4_rocm_l3_update_page_table(
    SparkHwQueue queue, SparkHwMemory memory, void *page_table_u32_device,
    const uint32_t *update_indices_u32_device,
    const uint32_t *update_values_u32_device, uint32_t update_count)
{
    hipStream_t stream;
    SparkHwStatus status;
    uint32_t blocks;
    void *arguments[4];

    if (memory != 0)
    {
        status = spark_hw_rocm_check_memory(&memory);
        if (status != SPARK_HW_OK)
            return status;
    }
    if (queue == 0 || page_table_u32_device == 0 ||
        update_indices_u32_device == 0 || update_values_u32_device == 0 ||
        update_count == 0u)
        return SPARK_HW_INVALID;

    stream = (hipStream_t)(void *)queue;
    blocks = (update_count + SPARK_DSV4_ROCM_CTA_THREADS - 1u) /
             SPARK_DSV4_ROCM_CTA_THREADS;
    arguments[0] = (void *)&page_table_u32_device;
    arguments[1] = (void *)&update_indices_u32_device;
    arguments[2] = (void *)&update_values_u32_device;
    arguments[3] = (void *)&update_count;
    return SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmUpdatePageTableKernel, blocks, 1u,
        SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream, arguments);
}

// ---------------------------------------------------------------------------
// L1 `layer.boundary_norm_project` - hcEnter half (C3 with fixed trees).
//
// The frozen L1 contents row names "hcPost fold, hcEnter, RMSNorm,
// attention input projections (+ TP shard pack/unpack)"; the hcPost fold
// landed with the earlier L1 work and consumes post/comb produced here.
// The sequence mirrors the CUDA module's SparkDsv4ModuleHcEnter exactly
// (spark_dsv4_resident_decode_stage_module.c :3226):
//   1. split-K mix dots + sum-of-squares partials over the flattened
//      boundary packet (SparkDsv4RocmHcMixSplitKKernel), then one finalize
//      kernel folding the partials on a fixed tree, applying the flat
//      rsqrt(mean-square) norm, and running the inference Sinkhorn
//      (SparkDsv4RocmHcMixFinalizeSinkhornKernel - port of
//      HcMixSplitKFinalizeKernel + HcParallelSinkhorn).
//   2. pre-reduce into the normalized hidden row plus the verbatim
//      residual copy (SparkDsv4RocmHcPreReduceKernel - port of
//      HcPreReduceKernel).
//
// Fixed trees (documented once, B3/C3 discipline - never re-tuned per
// bucket):
//   - MixSplitK staging: one element per thread over a 256-element split;
//     sum-of-squares through the SAME two-level block tree as every other
//     C3 island here (per-lane sequential fp32, wave64 butterfly, wave
//     partials folded by thread 0 in ascending wave order).
//   - Mix dots: lanes stride the 256-element split at wave64 granularity
//     (lane e, e+64, e+128, e+192), one wave64 butterfly per mix row;
//     partials fold across splits in ascending split order in the
//     finalize pass below. The wave64 lane subsets differ from the CUDA
//     target's wave32 walk - the recorded C3 delta, inside shared
//     tolerances like the archive's wave64 RMS reduce.
//   - Sinkhorn: serial per-row arithmetic, no cross-lane reduction at
//     all, so it is order-free by construction.
//   - PreReduce: strict ascending stream order per element, one RNE bf16
//     store; the residual copy moves raw bytes unrounded.
//
// Deliberate target deltas vs CUDA, both inside the shared C3 tolerance
// set and recorded honestly:
//   1. wave64 lane subsets for the mix/sumsq reductions (as above).
//   2. expf where the reference uses fast-math __expf inside its sigmoid
//      and softmax helpers (the archive-wide spelling rule; the archive's
//      SparkDsv4RocmSigmoid is reused verbatim).
// ---------------------------------------------------------------------------

#define SPARK_DSV4_ROCM_HC_SPLIT_K_ELEMENTS 256u
#define SPARK_DSV4_ROCM_HC_STREAM_MAX 4u
/* (2 + stream_count) * stream_count peaks at stream_count = 4 -> 24 rows. */
#define SPARK_DSV4_ROCM_HC_MIX_ROWS_MAX                                          \
    ((2u + SPARK_DSV4_ROCM_HC_STREAM_MAX) * SPARK_DSV4_ROCM_HC_STREAM_MAX)

#if defined(__cplusplus)
static_assert(SPARK_DSV4_ROCM_CTA_THREADS ==
                  SPARK_DSV4_ROCM_HC_SPLIT_K_ELEMENTS,
              "hcEnter split-K stages exactly one element per thread; the "
              "launch pins blockDim to the split width");
static_assert(SPARK_DSV4_ROCM_WARPS_PER_CTA <= SPARK_DSV4_ROCM_HC_MIX_ROWS_MAX,
              "every wave of the hcEnter CTA must own at least one mix row");
#endif

__global__ void SparkDsv4RocmHcMixSplitKKernel(
    const void *streams_bf16, const float *fn_f32, float *partials_f32,
    uint32_t combined_blocks, uint32_t split_count, uint32_t flat_dimension,
    uint32_t mix_rows)
{
    /* One thread per staged element: the launch pins blockDim to the split
     * width (see the static_assert above), so no participation guard may
     * exist here without breaking the block reduce - the entry refuses any
     * flat dimension that does not divide into whole splits instead. */
    SPARK_DSV4_ROCM_SHARED_FIXED(staged_values, float,
                                 SPARK_DSV4_ROCM_HC_SPLIT_K_ELEMENTS);
    SPARK_DSV4_ROCM_SHARED_FIXED(reduce_scratch, float,
                                 SPARK_DSV4_ROCM_WARPS_PER_CTA);
    uint32_t row = blockIdx.x / split_count;
    uint32_t split = blockIdx.x % split_count;
    uint32_t base = split * SPARK_DSV4_ROCM_HC_SPLIT_K_ELEMENTS;
    uint32_t warp = threadIdx.x / SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t lane = threadIdx.x % SPARK_DSV4_ROCM_WAVE_LANES;
    uint32_t mix, element;
    uint64_t partial_base;
    float value;
    if (blockIdx.x >= combined_blocks)
        return;
    value = SparkDsv4RocmBf16ToFloat(
        ((const uint16_t *)streams_bf16)[(uint64_t)row * flat_dimension +
                                         base + threadIdx.x]);
    staged_values[threadIdx.x] = value;
    value = SparkDsv4RocmBlockReduceSum(value * value, reduce_scratch);
    /* Explicit ordering for the staged reads below: the mix dots consume
     * EVERY thread's staged element, so visibility must not depend on
     * whatever barriers SparkDsv4RocmBlockReduceSum happens to contain
     * internally. */
    __syncthreads();
    partial_base = ((uint64_t)row * split_count + split) * ((uint64_t)mix_rows + 1u);
    if (threadIdx.x == 0u)
        partials_f32[partial_base] = value;
    for (mix = warp; mix < mix_rows; mix += SPARK_DSV4_ROCM_WARPS_PER_CTA)
    {
        float accumulator = 0.0f;
        for (element = lane; element < SPARK_DSV4_ROCM_HC_SPLIT_K_ELEMENTS;
             element += SPARK_DSV4_ROCM_WAVE_LANES)
            accumulator += staged_values[element] *
                fn_f32[(uint64_t)mix * flat_dimension + base + element];
        accumulator = SparkDsv4RocmWaveReduceSum(accumulator);
        if (lane == 0u)
            partials_f32[partial_base + 1u + mix] = accumulator;
    }
}

__global__ void SparkDsv4RocmHcMixFinalizeSinkhornKernel(
    const float *partials_f32, const float *scale3_f32, const float *base_f32,
    uint32_t row_count, uint32_t sinkhorn_iterations, uint32_t split_count,
    uint32_t flat_dimension, uint32_t mix_rows, uint32_t stream_count,
    float rms_epsilon, float hc_epsilon, float *mixes_f32, float *pre_f32,
    float *post_f32, float *comb_f32)
{
    /* One wave owns the whole finalize: threads [0, mix_rows] hold the
     * per-index partial totals, threads [0, stream_count^2) walk the
     * Sinkhorn comb. Launched at one 64-lane wavefront; the guards keep
     * every stage well inside that shape. Early returns are uniform across
     * the block (row/stream_count only), so every barrier stays legal. */
    SPARK_DSV4_ROCM_SHARED_FIXED(mixes_shared, float,
                                 SPARK_DSV4_ROCM_HC_MIX_ROWS_MAX);
    SPARK_DSV4_ROCM_SHARED_FIXED(comb_shared, float,
                                 SPARK_DSV4_ROCM_HC_STREAM_MAX *
                                     SPARK_DSV4_ROCM_HC_STREAM_MAX);
    SPARK_DSV4_ROCM_SHARED_FIXED(sums_shared, float,
                                 SPARK_DSV4_ROCM_HC_STREAM_MAX);
    uint32_t row = blockIdx.x;
    uint32_t lane_id = threadIdx.x;
    uint32_t split, iteration, index;
    uint64_t partial_base;
    float total = 0.0f, inverse, maximum, running;
    if (row >= row_count || stream_count == 0u ||
        stream_count > SPARK_DSV4_ROCM_HC_STREAM_MAX)
        return;
    /* Fixed cross-split fold: ascending split order, one addend chain per
     * lane; lane 0 carries the sum-of-squares total, lanes 1..mix_rows
     * carry their mix dot totals. */
    if (lane_id < mix_rows + 1u)
    {
        for (split = 0u; split < split_count; ++split)
        {
            partial_base = ((uint64_t)row * split_count + split) *
                           ((uint64_t)mix_rows + 1u);
            total += partials_f32[partial_base + lane_id];
        }
    }
    if (lane_id == 0u)
        sums_shared[0] = rsqrtf(total / (float)flat_dimension + rms_epsilon);
    __syncthreads();
    if (lane_id > 0u && lane_id < mix_rows + 1u)
    {
        inverse = sums_shared[0];
        mixes_shared[lane_id - 1u] = total * inverse;
        mixes_f32[(uint64_t)row * mix_rows + lane_id - 1u] = total * inverse;
    }
    __syncthreads();
    /* Inference Sinkhorn, ported step-for-step from HcParallelSinkhorn at
     * hc <= 4: sigmoid pre (+epsilon), doubled sigmoid post, comb built as
     * mixes*scale3[2] + base[...] then row-softmaxed with +epsilon added
     * after the division (so iteration 0's first row normalization IS
     * that softmax), then sinkhorn_iterations alternating row/column
     * normalizations with +epsilon inside every divisor. */
    if (lane_id < stream_count)
    {
        pre_f32[(uint64_t)row * stream_count + lane_id] =
            SparkDsv4RocmSigmoid(mixes_shared[lane_id] * scale3_f32[0] +
                                 base_f32[lane_id]) + hc_epsilon;
        post_f32[(uint64_t)row * stream_count + lane_id] =
            2.0f * SparkDsv4RocmSigmoid(
                       mixes_shared[stream_count + lane_id] * scale3_f32[1] +
                       base_f32[stream_count + lane_id]);
        maximum = -3.0e38f;
        for (index = 0u; index < stream_count; ++index)
        {
            comb_shared[lane_id * stream_count + index] =
                mixes_shared[2u * stream_count + lane_id * stream_count +
                             index] *
                    scale3_f32[2] +
                base_f32[2u * stream_count + lane_id * stream_count + index];
            maximum = fmaxf(maximum,
                            comb_shared[lane_id * stream_count + index]);
        }
        running = 0.0f;
        for (index = 0u; index < stream_count; ++index)
        {
            comb_shared[lane_id * stream_count + index] =
                expf(comb_shared[lane_id * stream_count + index] - maximum);
            running += comb_shared[lane_id * stream_count + index];
        }
        for (index = 0u; index < stream_count; ++index)
            comb_shared[lane_id * stream_count + index] =
                comb_shared[lane_id * stream_count + index] / running +
                hc_epsilon;
    }
    __syncthreads();
    for (iteration = 0u; iteration < sinkhorn_iterations; ++iteration)
    {
        if (iteration != 0u && lane_id < stream_count)
        {
            running = 0.0f;
            for (index = 0u; index < stream_count; ++index)
                running += comb_shared[lane_id * stream_count + index];
            sums_shared[lane_id] = running;
        }
        __syncthreads();
        if (iteration != 0u && lane_id < stream_count * stream_count)
            comb_shared[lane_id] /=
                sums_shared[lane_id / stream_count] + hc_epsilon;
        __syncthreads();
        if (lane_id < stream_count)
        {
            running = 0.0f;
            for (index = 0u; index < stream_count; ++index)
                running += comb_shared[index * stream_count + lane_id];
            sums_shared[lane_id] = running;
        }
        __syncthreads();
        if (lane_id < stream_count * stream_count)
            comb_shared[lane_id] /= sums_shared[lane_id % stream_count] + hc_epsilon;
        __syncthreads();
    }
    if (lane_id < stream_count * stream_count)
        comb_f32[(uint64_t)row * stream_count * stream_count + lane_id] =
            comb_shared[lane_id];
}

__global__ void SparkDsv4RocmHcPreReduceKernel(
    const void *streams_bf16, const float *pre_f32, void *reduced_bf16,
    void *residual_bf16, uint32_t row_count, uint32_t stream_count,
    uint32_t dimension, uint32_t tiles_per_row)
{
    const uint16_t *stream_words = (const uint16_t *)streams_bf16;
    uint16_t *residual_words = (uint16_t *)residual_bf16;
    uint16_t *reduced_words = (uint16_t *)reduced_bf16;
    uint32_t row = blockIdx.y;
    uint32_t tile = blockIdx.x;
    uint32_t element, stream;
    uint64_t index;
    uint16_t raw;
    float value;
    if (row >= row_count)
        return;
    for (element = tile * blockDim.x + threadIdx.x; element < dimension;
         element += tiles_per_row * blockDim.x)
    {
        value = 0.0f;
        for (stream = 0u; stream < stream_count; ++stream)
        {
            index = (((uint64_t)row * stream_count) + stream) * dimension +
                    element;
            raw = stream_words[index]; /* verbatim byte move - unrounded */
            residual_words[index] = raw;
            value += pre_f32[(uint64_t)row * stream_count + stream] *
                     SparkDsv4RocmBf16ToFloat(raw);
        }
        reduced_words[(uint64_t)row * dimension + element] =
            SparkDsv4RocmFloatToBf16Rne(value);
    }
}

SparkHwStatus spark_dsv4_rocm_l1_hc_enter(
    SparkHwQueue queue, SparkHwMemory memory, const void *streams_bf16_device,
    const float *fn_f32_device, const float *scale3_f32_device,
    const float *base_f32_device, void *partials_f32_device,
    void *mixes_f32_device, void *pre_f32_device, void *post_f32_device,
    void *comb_f32_device, void *reduced_hidden_bf16_device,
    void *residual_bf16_device, uint32_t row_count, uint32_t stream_count,
    uint32_t hidden_dimension, uint32_t sinkhorn_iterations, float rms_epsilon,
    float hc_epsilon)
{
    hipStream_t stream;
    SparkHwStatus status;
    uint32_t mix_rows, flat_dimension, split_count, tiles_per_row;
    uint32_t dimension_tiles, minimum_tiles, combined_blocks_u32;
    uint64_t combined_blocks;
    void *mix_arguments[7];
    void *finalize_arguments[15];
    void *pre_arguments[8];

    if (memory != 0)
    {
        status = spark_hw_rocm_check_memory(&memory);
        if (status != SPARK_HW_OK)
            return status;
    }
    if (queue == 0 || streams_bf16_device == 0 || fn_f32_device == 0 ||
        scale3_f32_device == 0 || base_f32_device == 0 ||
        partials_f32_device == 0 || mixes_f32_device == 0 ||
        pre_f32_device == 0 || post_f32_device == 0 ||
        comb_f32_device == 0 || reduced_hidden_bf16_device == 0 ||
        residual_bf16_device == 0)
        return SPARK_HW_INVALID;
    if (rms_epsilon != rms_epsilon || hc_epsilon != hc_epsilon)
        return SPARK_HW_INVALID; /* NaN geometry refuses */
    if (stream_count == 0u || stream_count > SPARK_DSV4_ROCM_HC_STREAM_MAX ||
        hidden_dimension == 0u)
        return SPARK_HW_INVALID;
    mix_rows = (2u + stream_count) * stream_count;
    flat_dimension = stream_count * hidden_dimension;
    /* Split-K covers the flat dimension only in whole 256-element splits
     * (the staging step owns exactly one element per thread). */
    if ((flat_dimension % SPARK_DSV4_ROCM_HC_SPLIT_K_ELEMENTS) != 0u)
        return SPARK_HW_INVALID;
    split_count = flat_dimension / SPARK_DSV4_ROCM_HC_SPLIT_K_ELEMENTS;
    /* Grid-x capacity for the split-K launch; a larger sealed frame is a
     * capacity refusal, not an argument error (same reasoning as the
     * uint32 flattening guards in L4/L5). */
    combined_blocks = (uint64_t)row_count * split_count;
    if (combined_blocks > (uint64_t)UINT32_MAX)
        return SPARK_HW_EXHAUSTED;
    if (row_count > SPARK_DSV4_ROCM_MAX_GRID_Y)
        return SPARK_HW_EXHAUSTED; /* PreReduce addresses rows via grid-y */
    if (row_count == 0u)
        return SPARK_HW_OK; /* empty frame: nothing to mix or reduce */
    combined_blocks_u32 = (uint32_t)combined_blocks;

    stream = (hipStream_t)(void *)queue;

    /* 1. Split-K mix dots + sum-of-squares partials. */
    mix_arguments[0] = (void *)&streams_bf16_device;
    mix_arguments[1] = (void *)&fn_f32_device;
    mix_arguments[2] = (void *)&partials_f32_device;
    mix_arguments[3] = (void *)&combined_blocks_u32;
    mix_arguments[4] = (void *)&split_count;
    mix_arguments[5] = (void *)&flat_dimension;
    mix_arguments[6] = (void *)&mix_rows;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmHcMixSplitKKernel,
        combined_blocks_u32, 1u, SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream,
        mix_arguments);
    if (status != SPARK_HW_OK)
        return status;

    /* 2. Partial fold + flat norm + inference Sinkhorn (one wave). */
    finalize_arguments[0] = (void *)&partials_f32_device;
    finalize_arguments[1] = (void *)&scale3_f32_device;
    finalize_arguments[2] = (void *)&base_f32_device;
    finalize_arguments[3] = (void *)&row_count;
    finalize_arguments[4] = (void *)&sinkhorn_iterations;
    finalize_arguments[5] = (void *)&split_count;
    finalize_arguments[6] = (void *)&flat_dimension;
    finalize_arguments[7] = (void *)&mix_rows;
    finalize_arguments[8] = (void *)&stream_count;
    finalize_arguments[9] = (void *)&rms_epsilon;
    finalize_arguments[10] = (void *)&hc_epsilon;
    finalize_arguments[11] = (void *)&mixes_f32_device;
    finalize_arguments[12] = (void *)&pre_f32_device;
    finalize_arguments[13] = (void *)&post_f32_device;
    finalize_arguments[14] = (void *)&comb_f32_device;
    status = SparkDsv4RocmLaunch(
        (const void *)&SparkDsv4RocmHcMixFinalizeSinkhornKernel, row_count,
        1u, SPARK_DSV4_ROCM_WAVE_LANES, 0u, stream, finalize_arguments);
    if (status != SPARK_HW_OK)
        return status;

    /* 3. Pre-reduce into normalized hidden + verbatim residual copy.
     * Tile policy mirrored from the CUDA launcher: at least sixteen
     * blocks overall when rows allow, capped by the dimension's own
     * tiling. */
    minimum_tiles = (16u + row_count - 1u) / row_count;
    dimension_tiles = (hidden_dimension + SPARK_DSV4_ROCM_CTA_THREADS - 1u) /
                      SPARK_DSV4_ROCM_CTA_THREADS;
    tiles_per_row = minimum_tiles < dimension_tiles ? minimum_tiles
                                                    : dimension_tiles;
    if (tiles_per_row == 0u)
        tiles_per_row = 1u;
    pre_arguments[0] = (void *)&streams_bf16_device;
    pre_arguments[1] = (void *)&pre_f32_device;
    pre_arguments[2] = (void *)&reduced_hidden_bf16_device;
    pre_arguments[3] = (void *)&residual_bf16_device;
    pre_arguments[4] = (void *)&row_count;
    pre_arguments[5] = (void *)&stream_count;
    pre_arguments[6] = (void *)&hidden_dimension;
    pre_arguments[7] = (void *)&tiles_per_row;
    return SparkDsv4RocmLaunch((const void *)&SparkDsv4RocmHcPreReduceKernel,
                               tiles_per_row, row_count,
                               SPARK_DSV4_ROCM_CTA_THREADS, 0u, stream,
                               pre_arguments);
}

