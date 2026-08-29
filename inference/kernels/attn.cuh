#pragma once

// Decode attention. RoPE, latent-absorbed attention, and sparse selection.
//
// None of this is model-specific, and the version it replaces was named as
// though it were: SparkGlmResidentDecodeStageAbsorbedAttentionKernel, 299
// lines carrying 52 SPARK_GLM_* references. Every one of the 52 is a
// dimension, a head count or a cache stride - none changes what is computed. The
// model belongs in the arguments.
//
// WHY LATENT-ABSORBED. Storing per-head keys and values costs
// heads * (qk_dim + v_dim) * 2 bytes per position. Storing one shared latent row
// costs (latent + rope) * 2. For a 64-head model that is 57 KB against 1152
// bytes - a 50x reduction on the only tensor that grows with context, bought by
// folding the up-projections into the query and output weights so the attention
// happens in latent space and per-head K and V are never materialised.
//
// The compute that buys it is free here. Decode attention is bound by cache
// bytes, not arithmetic: the calibration found three structurally different
// attention kernels producing identical time, which is the signature of a path
// limited only by the bytes all three share.
//
// THE CACHE FORMAT IS A TRAIT, INDEPENDENTLY OF THE WEIGHTS. A slot is read once
// per (sequence, position) and shared with nothing, where a weight tile is read
// once and shared across every row in its tile - different points on the
// precision-versus-bytes curve, so BF16 weights with an FP8 cache is a normal
// combination and either can change without the other.

#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include <stdint.h>

// -- RoPE ---------------------------------------------------------------------
//
// Rotary position embedding over the trailing rope_dim elements of a row.
// theta and the dimension are arguments because they are the only things that
// differ between models - four families in the old tree had their own copy of
// this and the difference between them was two constants.
//
// The half-split pairing, not the interleaved one: element i pairs with
// i + rope_dim/2. Getting that wrong produces output that is fluent and subtly
// positionally wrong, which is the hardest kind of bug to attribute.
static __device__ __forceinline__ void LmRopePair(float *low, float *high, float angle)
{
	float c = __cosf(angle),s = __sinf(angle);
	float a = *low,b = *high;
	*low = (a * c) - (b * s);
	*high = (a * s) + (b * c);
}

// WHICH TWO ELEMENTS FORM A PAIR. Both conventions are called "rope" and they
// are different rotations, so a checkpoint trained under one and served under
// the other is fluent and positionally wrong - the failure the comment above
// describes, now selectable instead of assumed.
//
// Half-split pairs i with i + rope_dim/2. The GDN model and MiMo 2.5 use this
// convention and it stays the default, so existing call sites remain explicit
// only when their checkpoint differs.
//
// Interleaved pairs 2i with 2i+1, the view_as_complex layout. The V4 model and
// GLM 5.2 encode their released checkpoints this way; GLM declares the same
// convention separately for its MLA and DSA indexer.
enum LmRopePairing
{
	LM_ROPE_HALF_SPLIT = 0,
	LM_ROPE_INTERLEAVED = 1
};

// The three rope kernels below had this body character for character. It is one
// function now, because the pairing is the thing that varies and a convention
// duplicated three times is a convention that will only be fixed twice.
template<LmRopePairing PAIRING>
static __device__ __forceinline__ void LmRopeRotate(uint16_t *rows_bf16, uint64_t base, uint32_t index, uint32_t half, float angle)
{
	uint32_t low_offset,high_offset;
	float low,high;
	low_offset = (PAIRING == LM_ROPE_INTERLEAVED) ? (index * 2u) : index;
	high_offset = (PAIRING == LM_ROPE_INTERLEAVED) ? ((index * 2u) + 1u) : (half + index);
	low = LmBf16ToFloat(rows_bf16[base + low_offset]);
	high = LmBf16ToFloat(rows_bf16[base + high_offset]);
	LmRopePair(&low,&high,angle);
	rows_bf16[base + low_offset] = LmFloatToBf16(low);
	rows_bf16[base + high_offset] = LmFloatToBf16(high);
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopeKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t row_stride, uint32_t rope_offset, uint32_t rope_dim, float theta)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + rope_offset;
	uint32_t half = rope_dim / 2u,index;
	float position = (float)positions[blockIdx.x];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dim));
}

// RoPE over one slice of every head. A single-row RoPE kernel cannot express
// an index query laid out as [row, head, head_dim] without either duplicating
// positions or launching once per head from the host.
template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopePerHeadKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t head_count, uint32_t head_dimension, uint32_t rope_offset, uint32_t rope_dimension, float theta)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,half = rope_dimension / 2u,index;
	uint64_t base;
	float position;
	if ( head >= head_count )
		return;
	base = (((uint64_t)row * head_count) + head) * head_dimension + rope_offset;
	position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dimension));
}

// -- latent-absorbed decode attention ------------------------------------------
//
// One block per (sequence, head group). Online softmax in a single pass over the
// cache: running maximum and denominator are rescaled as a larger score appears,
// so the cache is read once. Reading it twice - once to find the maximum, once
// to accumulate - would double the only traffic that matters.
//
// SPARSE SELECTION IS AN INDEX ARRAY, NOT A SEPARATE KERNEL. Passing a list of
// selected positions makes this the sparse path; passing null makes it dense.
// The old tree had these as different kernels, and the difference was which
// positions the loop visited.
// PREFILL IS THIS KERNEL WITH MORE ROWS AND A MASK.
//
// The old tree had DsaPrefillRowSetupKernel, DsaSparsePrefillAttentionKernel,
// DsaPrefillIndexerPass and PagedChunkPrefill as separate implementations. They
// compute the same thing decode does: a row attends over cached positions. What
// differs is that a prefill row is not the last position, so it must not see
// positions after its own, and that there are many rows rather than one.
//
// Both are one argument. row_position, when non-null, gives each row its own
// position and the loop stops there; when null every row attends to everything
// cached, which is decode. That is the whole difference, and it is a comparison
// rather than a kernel.
//
// Chunking is the caller's business and stays there. A prefill of 8,000 tokens
// runs as chunks of a few hundred rows because the intermediate buffers are
// sized for that, and which chunk size is a scheduling decision that depends on
// what else is resident - not something a kernel should decide.
template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmAttentionDecodeKernel(const uint16_t *__restrict__ query_latent_bf16, const uint16_t *__restrict__ query_rope_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position)
{
	// THE ACCUMULATOR IS EXACTLY SIZED. The first version held eight slots and
	// guarded element < LATENT, which silently drops the tail of any latent
	// wider than 8 * THREADS - a static_assert said so. LATENT is a template
	// argument here, so the array is the exact stride count instead: one slot
	// per thread on device (576 latents at 256 threads fits in three), and the
	// whole latent on the host harness's one-thread schedule, which is the only
	// reason the full layer can be emulated at model width.
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float shared_query[LATENT + ROPE];
	float accumulator[(LATENT + THREADS - 1u) / THREADS];
	uint32_t row = blockIdx.x,head = blockIdx.y,index,step,positions;
	uint32_t sequence = sequence_of_row[row];
	uint64_t query_base = ((uint64_t)row * heads + head) * (LATENT + ROPE);
	float running_max = -INFINITY,running_sum = 0.0f;
	if ( !LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count )
	{
		LmKvReportRequiredAccessFailure(
			cache,
			!LmKvViewIsConfigured(cache)
				? LM_KV_ACCESS_ERROR_INVALID_VIEW
				: LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			0xffffffffu,
			0xffffffffu);
		return;
	}
	for (index = 0u; index < (LATENT + THREADS - 1u) / THREADS; ++index)
		accumulator[index] = 0.0f;
	for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
		shared_query[index] = LmBf16ToFloat(query_latent_bf16[query_base + index]);
	__syncthreads();
	positions = selected_positions != 0 ? selected_count : context_length[sequence];
	for (step = 0u; step < positions; ++step)
	{
		uint32_t position = selected_positions != 0
			? selected_positions[(row * selected_count) + step] : step;
		// Causal: a prefill row must not see past itself. Skipping rather than
		// masking the score keeps the online softmax's running maximum honest -
		// a masked-to-negative-infinity score still participates in the rescale
		// and costs the cache read that made it worthless.
		if ( row_position != 0 && position > row_position[row] )
			continue;
		const uint8_t *slot = LmKvSlotRequired<Geometry>(
			cache,sequence,position,row,LM_KV_ACCESS_READ);
		float score = 0.0f,scaled,previous;
		if ( slot == 0 )
			return;
		for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
			score += shared_query[index] * LmBf16ToFloat(((const uint16_t *)slot)[index]);
		score = LmBlockSum<THREADS>(score,reduction) * qk_scale;
		// Online softmax: rescale what is already accumulated rather than
		// revisit the cache once the maximum is known.
		previous = running_max;
		running_max = fmaxf(running_max,score);
		scaled = __expf(previous - running_max);
		running_sum = (running_sum * scaled) + __expf(score - running_max);
		for (index = 0u; index < (LATENT + THREADS - 1u) / THREADS; ++index)
		{
			uint32_t element = (index * THREADS) + threadIdx.x;
			if ( element < LATENT )
				accumulator[index] = (accumulator[index] * scaled)
					+ (__expf(score - running_max)
						* LmBf16ToFloat(((const uint16_t *)slot)[element]));
		}
	}
	for (index = 0u; index < (LATENT + THREADS - 1u) / THREADS; ++index)
	{
		uint32_t element = (index * THREADS) + threadIdx.x;
		if ( element < LATENT )
			output_bf16[(((uint64_t)row * heads) + head) * LATENT + element] =
				LmFloatToBf16(accumulator[index] / fmaxf(running_sum,1.0e-20f));
	}
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentAttentionDecodeKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ query_rope_bf16,
    LmKvView cache,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    const uint32_t *__restrict__ selected_positions,
    uint32_t selected_count,
    uint32_t heads,
    float qk_scale,
    uint16_t *__restrict__ output_bf16,
    const uint32_t *__restrict__ row_position)
{
    static_assert(
        LATENT <= 8u * THREADS,
        "the latent must fit the per-thread accumulator");
    __shared__ float reduction[THREADS / LM_WARP_LANES];
    __shared__ float shared_query[LATENT + ROPE];
    float accumulator[8];
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t index;
    uint32_t step;
    uint32_t position_count;
    uint32_t sequence = sequence_of_row[row];
    uint64_t latent_base = ((uint64_t)row * heads + head) * LATENT;
    uint64_t rope_base = ((uint64_t)row * heads + head) * ROPE;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
    {
        LmKvReportRequiredAccessFailure(
            cache,
            !LmKvViewIsConfigured(cache)
                ? LM_KV_ACCESS_ERROR_INVALID_VIEW
                : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
            LM_KV_ACCESS_READ,
            row,
            sequence,
            0xffffffffu,
            0xffffffffu);
        return;
    }

    for (index = 0u; index < 8u; ++index)
    {
        accumulator[index] = 0.0f;
    }
    for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
    {
        shared_query[index] = index < LATENT
            ? LmBf16ToFloat(query_latent_bf16[latent_base + index])
            : LmBf16ToFloat(query_rope_bf16[rope_base + index - LATENT]);
    }
    __syncthreads();

    position_count = selected_positions != 0
        ? selected_count
        : context_length[sequence];
    for (step = 0u; step < position_count; ++step)
    {
        uint32_t position = selected_positions != 0
            ? selected_positions[(row * selected_count) + step]
            : step;
        const uint8_t *slot;
        float score = 0.0f;
        float scaled_previous;
        float scaled_current;
        float previous_max;

        if (row_position != 0 && position > row_position[row])
        {
            continue;
        }
        slot = LmKvSlotRequired<Geometry>(
            cache, sequence, position, row, LM_KV_ACCESS_READ);
        if (slot == 0)
        {
            return;
        }
        for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
        {
            score += shared_query[index] *
                LmBf16ToFloat(((const uint16_t *)slot)[index]);
        }
        score = LmBlockSum<THREADS>(score, reduction) * qk_scale;
        previous_max = running_max;
        running_max = fmaxf(running_max, score);
        scaled_previous = __expf(previous_max - running_max);
        scaled_current = __expf(score - running_max);
        running_sum = (running_sum * scaled_previous) + scaled_current;
        for (index = 0u; index < 8u; ++index)
        {
            uint32_t element = (index * THREADS) + threadIdx.x;

            if (element < LATENT)
            {
                accumulator[index] =
                    (accumulator[index] * scaled_previous) +
                    (scaled_current *
                        LmBf16ToFloat(((const uint16_t *)slot)[element]));
            }
        }
    }
    for (index = 0u; index < 8u; ++index)
    {
        uint32_t element = (index * THREADS) + threadIdx.x;

        if (element < LATENT)
        {
            output_bf16[
                (((uint64_t)row * heads) + head) * LATENT + element] =
                LmFloatToBf16(
                    accumulator[index] / fmaxf(running_sum, 1.0e-20f));
        }
    }
}

// -- split-K decode attention (flash-decode) -----------------------------------
//
// One (row, head) CTA walks its whole context serially - at decode the grid is
// rows x heads (24-64 CTAs against 48 SMs at one row), every CTA holds eight
// warps at a twelfth of an SM, and each position costs a full-block reduction
// plus two reads of the same slot. Long contexts leave the machine idle inside
// a full grid. The split form partitions the position range across a third
// grid axis, runs the IDENTICAL per-position body per partition (same block
// reduction width, same online-softmax update - a partition is the base
// kernel restricted to a contiguous range), and a combine pass merges the
// per-partition (max, sum, accumulator) states.
//
// The merge is a fixed-order deterministic combine, not a bit-exact
// reproduction: the base kernel's running rescale chains through every
// position, and re-partitioning the chain changes where the multiplies round.
// The formula is the same softmax (sum exp(s-M) v over sum exp(s-M)), the
// order is fixed (ascending partition), and the extremes are exact:
//  - one active partition (splits == 1, or every other partition empty):
//    the combine multiplies by exp(0) = 1 and adds zeros, so the output is
//    bit-identical to the base kernel;
//  - an empty context: every partial max stays -INFINITY, the combine scales
//    to zero, and 0 / 1e-20 reproduces the base kernel's 0.
// The engagement threshold is deployment policy, not kernel policy: below it
// the caller launches the base kernel byte-for-byte.
#define LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS 16u
#define LM_LATENT_ATTN_SPLIT_CTAS_PER_SM 4u
#define LM_LATENT_ATTN_SPLIT_BLOCK_FLOATS(l) ((l) + 2u)

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentAttentionDecodeSplitKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ query_rope_bf16,
    LmKvView cache,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    const uint32_t *__restrict__ selected_positions,
    uint32_t selected_count,
    uint32_t heads,
    uint32_t partitions,
    float qk_scale,
    float *__restrict__ partials,
    const uint32_t *__restrict__ row_position)
{
    static_assert(
        LATENT <= 8u * THREADS,
        "the latent must fit the per-thread accumulator");
    __shared__ float reduction[THREADS / LM_WARP_LANES];
    __shared__ float shared_query[LATENT + ROPE];
    float accumulator[8];
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t partition = blockIdx.z;
    uint32_t index;
    uint32_t step;
    uint32_t position_count;
    uint32_t first_position;
    uint32_t last_position;
    uint32_t partition_span;
    uint32_t sequence = sequence_of_row[row];
    uint64_t latent_base = ((uint64_t)row * heads + head) * LATENT;
    uint64_t rope_base = ((uint64_t)row * heads + head) * ROPE;
    uint64_t partial_base;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
    {
        LmKvReportRequiredAccessFailure(
            cache,
            !LmKvViewIsConfigured(cache)
                ? LM_KV_ACCESS_ERROR_INVALID_VIEW
                : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
            LM_KV_ACCESS_READ,
            row,
            sequence,
            0xffffffffu,
            0xffffffffu);
        return;
    }
    if (partition >= partitions)
    {
        return;
    }

    for (index = 0u; index < 8u; ++index)
    {
        accumulator[index] = 0.0f;
    }
    for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
    {
        shared_query[index] = index < LATENT
            ? LmBf16ToFloat(query_latent_bf16[latent_base + index])
            : LmBf16ToFloat(query_rope_bf16[rope_base + index - LATENT]);
    }
    __syncthreads();

    // Contiguous partitions of the position range: a partition is the base
    // kernel's walk restricted to [first, last). The span comes from the
    // DEVICE-side position count, so the boundaries are data, not policy.
    position_count = selected_positions != 0
        ? selected_count
        : context_length[sequence];
    partition_span = (position_count + partitions - 1u) / partitions;
    first_position = partition * partition_span;
    last_position = first_position + partition_span;
    if (last_position > position_count)
    {
        last_position = position_count;
    }
    partial_base = (((uint64_t)row * heads + head) * partitions + partition) *
                   (LATENT + 2u);
    if (first_position >= last_position)
    {
        // An over-partitioned tail: a -INFINITY max contributes nothing to
        // the combine, so over-partitioning stays bit-identical to one
        // partition covering the same range.
        if (threadIdx.x == 0u)
        {
            partials[partial_base] = -INFINITY;
            partials[partial_base + 1u] = 0.0f;
        }
        for (index = threadIdx.x; index < LATENT; index += THREADS)
        {
            partials[partial_base + 2u + index] = 0.0f;
        }
        return;
    }
    for (step = first_position; step < last_position; ++step)
    {
        uint32_t position = selected_positions != 0
            ? selected_positions[(row * selected_count) + step]
            : step;
        const uint8_t *slot;
        float score = 0.0f;
        float scaled_previous;
        float scaled_current;
        float previous_max;

        if (row_position != 0 && position > row_position[row])
        {
            continue;
        }
        slot = LmKvSlotRequired<Geometry>(
            cache, sequence, position, row, LM_KV_ACCESS_READ);
        if (slot == 0)
        {
            // The base kernel returns without writing output; the frame is
            // already failed through the access-error channel. Publish the
            // partial as it stands so the combine reads no stale bytes, and
            // let the failed frame discard the result.
            if (threadIdx.x == 0u)
            {
                partials[partial_base] = running_max;
                partials[partial_base + 1u] = running_sum;
            }
            for (index = 0u; index < 8u; ++index)
            {
                uint32_t element = (index * THREADS) + threadIdx.x;

                if (element < LATENT)
                {
                    partials[partial_base + 2u + element] = accumulator[index];
                }
            }
            return;
        }
        for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
        {
            score += shared_query[index] *
                LmBf16ToFloat(((const uint16_t *)slot)[index]);
        }
        score = LmBlockSum<THREADS>(score, reduction) * qk_scale;
        previous_max = running_max;
        running_max = fmaxf(running_max, score);
        scaled_previous = __expf(previous_max - running_max);
        scaled_current = __expf(score - running_max);
        running_sum = (running_sum * scaled_previous) + scaled_current;
        for (index = 0u; index < 8u; ++index)
        {
            uint32_t element = (index * THREADS) + threadIdx.x;

            if (element < LATENT)
            {
                accumulator[index] =
                    (accumulator[index] * scaled_previous) +
                    (scaled_current *
                        LmBf16ToFloat(((const uint16_t *)slot)[element]));
            }
        }
    }
    if (threadIdx.x == 0u)
    {
        partials[partial_base] = running_max;
        partials[partial_base + 1u] = running_sum;
    }
    for (index = 0u; index < 8u; ++index)
    {
        uint32_t element = (index * THREADS) + threadIdx.x;

        if (element < LATENT)
        {
            partials[partial_base + 2u + element] = accumulator[index];
        }
    }
}

// The deterministic combine: global max over the partition maxes, each
// partition rescaled by exp(max - global max), denominator and accumulator
// summed in ASCENDING partition order - one fixed evaluation order, so the
// same inputs give the same bits on every run, on every GPU. One non-empty
// partition (or none) reproduces the single-pass kernel bit for bit; the
// general case is the same softmax up to where the multiplies round.
template<uint32_t THREADS, uint32_t LATENT>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentAttentionDecodeSplitCombineKernel(
    const float *__restrict__ partials,
    uint16_t *__restrict__ output_bf16,
    uint32_t heads,
    uint32_t partitions)
{
    __shared__ float scales[LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS];
    __shared__ float denominator_shared;
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t partition;
    uint32_t element;
    uint64_t block_base;
    float global_max;
    float denominator;

    block_base = ((uint64_t)row * heads + head) * partitions *
                 (LATENT + 2u);
    if (threadIdx.x == 0u)
    {
        global_max = -INFINITY;
        for (partition = 0u; partition < partitions; ++partition)
        {
            float candidate = partials[
                block_base + (uint64_t)partition * (LATENT + 2u)];

            global_max = candidate > global_max ? candidate : global_max;
        }
        denominator = 0.0f;
        for (partition = 0u; partition < partitions; ++partition)
        {
            float partition_max = partials[
                block_base + (uint64_t)partition * (LATENT + 2u)];
            float scale = (global_max == -INFINITY ||
                           partition_max == -INFINITY)
                ? 0.0f
                : __expf(partition_max - global_max);

            scales[partition] = scale;
            denominator = fmaf(
                partials[block_base + (uint64_t)partition * (LATENT + 2u) +
                         1u],
                scale,
                denominator);
        }
        // A division per element, matching the single-pass kernel's epilogue
        // (x / max(den, 1e-20)) - a precomputed reciprocal would round
        // differently and break the one-partition bit-exactness.
        denominator_shared = denominator;
    }
    __syncthreads();
    for (element = threadIdx.x; element < LATENT; element += THREADS)
    {
        float merged = 0.0f;

        for (partition = 0u; partition < partitions; ++partition)
        {
            merged = fmaf(
                partials[block_base + (uint64_t)partition * (LATENT + 2u) +
                         2u + element],
                scales[partition],
                merged);
        }
        output_bf16[((uint64_t)row * heads + head) * LATENT + element] =
            LmFloatToBf16(merged / fmaxf(denominator_shared, 1.0e-20f));
    }
}

// THE FLASH-DECODE ENTRY: below the deployment threshold (or with no
// workspace, or when the grid already fills the machine) this launches the
// single-pass kernel byte-for-byte; above it, one split launch plus one
// combine launch. split_context_threshold == 0 disables the split path
// entirely - the shipped deployments keep it off until the GPU cell
// qualifies it. position_bound is the caller's host-side upper bound on the
// walk length (the selected count when selecting, else the context bound);
// the partition boundaries themselves come from the device-side count.
template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentAttentionDecodeSplitLaunch(
    const uint16_t *query_latent_bf16,
    const uint16_t *query_rope_bf16,
    LmKvView cache,
    const uint32_t *sequence_of_row,
    const uint32_t *context_length,
    const uint32_t *selected_positions,
    uint32_t selected_count,
    uint32_t heads,
    float qk_scale,
    uint16_t *output_bf16,
    const uint32_t *row_position,
    uint32_t rows,
    uint32_t position_bound,
    uint32_t split_context_threshold,
    float *split_partials,
    uint32_t split_partial_blocks,
    uint32_t multiprocessor_count,
    cudaStream_t stream)
{
    uint32_t blocks;
    uint32_t wanted;
    uint32_t partitions;

    blocks = rows * heads;
    wanted = multiprocessor_count == 0u || blocks == 0u
        ? 1u
        : (multiprocessor_count * LM_LATENT_ATTN_SPLIT_CTAS_PER_SM +
           blocks - 1u) / blocks;
    partitions = wanted < 1u ? 1u : wanted;
    if (partitions > LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS)
    {
        partitions = LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS;
    }
    if (split_context_threshold == 0u || split_partials == 0 ||
        position_bound < split_context_threshold || partitions < 2u ||
        blocks == 0u ||
        (uint64_t)blocks * partitions > split_partial_blocks)
    {
        partitions = 1u;
    }
    if (partitions == 1u)
    {
        LM_LAUNCH(
            (LmLatentAttentionDecodeKernel<Geometry, THREADS, LATENT, ROPE>),
            dim3(rows, heads),
            THREADS,
            0,
            stream,
            query_latent_bf16,
            query_rope_bf16,
            cache,
            sequence_of_row,
            context_length,
            selected_positions,
            selected_count,
            heads,
            qk_scale,
            output_bf16,
            row_position);
        return cudaPeekAtLastError();
    }
    LM_LAUNCH(
        (LmLatentAttentionDecodeSplitKernel<
            Geometry, THREADS, LATENT, ROPE>),
        dim3(rows, heads, partitions),
        THREADS,
        0,
        stream,
        query_latent_bf16,
        query_rope_bf16,
        cache,
        sequence_of_row,
        context_length,
        selected_positions,
        selected_count,
        heads,
        partitions,
        qk_scale,
        split_partials,
        row_position);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return cudaPeekAtLastError();
    }
    LM_LAUNCH(
        (LmLatentAttentionDecodeSplitCombineKernel<THREADS, LATENT>),
        dim3(rows, heads),
        THREADS,
        0,
        stream,
        split_partials,
        output_bf16,
        heads,
        partitions);
    return cudaPeekAtLastError();
}

// -- sparse selection ----------------------------------------------------------
//
// Score every cached position with a low-rank index head and keep the top K.
// The scoring is a dot product against a narrower query, so it costs
// index_dim/latent of a full attention pass - about a fifth here - and the full
// pass then reads a fixed number of positions instead of the whole context.
//
// That is the entire point: it turns attention from linear in context length
// into constant, which for a growing cache is the difference between a model
// that degrades with conversation length and one that does not.
//
// The selection is shared across a group of layers, so the score is computed
// once per group rather than once per layer.
template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseScoreKernel(const uint16_t *__restrict__ index_query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, uint32_t index_heads, float *__restrict__ scores)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,position = blockIdx.y;
	uint32_t sequence = sequence_of_row[row],head,index;
	const uint8_t *slot;
	float total = 0.0f;
	if ( !LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count )
	{
		LmKvReportRequiredAccessFailure(
			cache,
			!LmKvViewIsConfigured(cache)
				? LM_KV_ACCESS_ERROR_INVALID_VIEW
				: LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			position,
			0xffffffffu);
		return;
	}
	if ( position >= context_length[sequence] )
		return;
	slot = LmKvSlotRequired<Geometry>(
		cache,sequence,position,row,LM_KV_ACCESS_READ);
	if ( slot == 0 )
		return;
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[(((uint64_t)row * index_heads) + head) * INDEX_DIM + index])
				* LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
		scores[((uint64_t)row * gridDim.y) + position] = total;
}

// DSA score with a separate index-key cache and a learned per-head mixture.
// Position is blockIdx.x so million-token contexts do not hit CUDA's 65,535
// blockIdx.y ceiling. row_position makes the same launch causal for prefill.
template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmWeightedSparseScoreKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight_bf16, LmKvView index_cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t index_heads, float qk_scale, float *__restrict__ scores)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t position = blockIdx.x,row = blockIdx.y,sequence,head,index;
	const uint8_t *slot;
	float total = 0.0f,partial;
	if ( scores == 0 )
		return;
	if ( threadIdx.x == 0u )
		scores[((uint64_t)row * gridDim.x) + position] = -INFINITY;
	if ( sequence_of_row == 0 || context_length == 0 || index_query_bf16 == 0
		|| head_weight_bf16 == 0 || !LmKvViewIsConfigured(index_cache) )
	{
		LmKvReportRequiredAccessFailure(
			index_cache,
			LM_KV_ACCESS_ERROR_INVALID_VIEW,
			LM_KV_ACCESS_READ,
			row,
			0xffffffffu,
			position,
			0xffffffffu);
		return;
	}
	sequence = sequence_of_row[row];
	if ( sequence >= index_cache.sequence_count )
	{
		LmKvReportRequiredAccessFailure(
			index_cache,
			LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			position,
			0xffffffffu);
		return;
	}
	if ( position >= context_length[sequence] || (row_position != 0 && position > row_position[row]) )
		return;
	slot = LmKvSlotRequired<Geometry>(index_cache,sequence,position,row,LM_KV_ACCESS_READ);
	if ( slot == 0 )
		return;
	for (head = 0u; head < index_heads; head++)
	{
		partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[(((uint64_t)row * index_heads) + head) * INDEX_DIM + index]) * LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction) * LmBf16ToFloat(head_weight_bf16[((uint64_t)row * index_heads) + head]);
	}
	if ( threadIdx.x == 0u )
		scores[((uint64_t)row * gridDim.x) + position] = total * qk_scale;
}

// -- YaRN rope ------------------------------------------------------------------
//
// Rope for a model whose context was extended past its training length.
//
// Plain rope extrapolates: a position beyond anything seen in training gets a
// frequency the model has no calibration for, and quality degrades with
// distance. YaRN interpolates instead, and does it PER FREQUENCY BAND - high
// frequencies encode local order and are left alone, low frequencies encode
// long-range position and are scaled by the extension factor, with a ramp
// between.
//
// The band edges come from how many full rotations a wavelength completes within
// the original context. A dimension whose wavelength is shorter than the
// original length has seen every phase and needs no correction; one longer than
// it has not, and is interpolated.
//
// Getting this wrong is quiet: applying plain rope to a YaRN model produces text
// that is fine for a few thousand tokens and drifts after, which reads as the
// model being bad at long context rather than as a missing transform.
static __device__ __forceinline__ float LmYarnFrequency(uint32_t index, uint32_t rope_dimension, float theta, float scale_factor, float original_positions, float low_band, float high_band)
{
	float exponent = 2.0f * (float)index / (float)rope_dimension;
	float inverse = __powf(theta,-exponent);
	// Wavelength in positions, and where it falls between the two band edges.
	float wavelength = 6.2831853f / inverse;
	float rotations = original_positions / wavelength;
	float ramp = (rotations - low_band) / fmaxf(high_band - low_band,1e-3f);
	float blend = fminf(fmaxf(ramp,0.0f),1.0f);
	// blend 1 is a short wavelength, left at the extrapolated frequency;
	// blend 0 is a long one, fully interpolated by the scale factor.
	return((inverse * blend) + ((inverse / scale_factor) * (1.0f - blend)));
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopeYarnKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t row_stride, uint32_t rope_offset, uint32_t rope_dim, float theta, float scale_factor, float original_positions, float low_band, float high_band)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + rope_offset;
	uint32_t half = rope_dim / 2u,index;
	float position = (float)positions[blockIdx.x];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,position *
			LmYarnFrequency(index,rope_dim,theta,scale_factor,
				original_positions,low_band,high_band));
}

// -- hierarchical sparse selection ------------------------------------------------
//
// Harvested from the old decode stage's DsaKeyIndexBlockSummaryBuildKernel and
// DsaScoreSelectHierarchicalKernel, which my first version did not have and
// badly needed.
//
// LmSparseScoreKernel above scores every cached position, so the selection pass
// is linear in context - which defeats the point of sparse attention, whose
// whole job is to make attention constant. The hierarchical form scores one
// SUMMARY per block first, keeps the promising blocks, and scores positions only
// inside those:
//
//     context    flat dots   hierarchical   speedup
//        8,192       8,192          2,176      3.8x
//      131,072     131,072          4,096     32.0x
//    1,048,576   1,048,576         18,432     56.9x
//
// At a million tokens the flat version spends more on deciding what to attend to
// than on attending.
//
// THE SUMMARY IS A MAX, NOT A MEAN. A block is worth visiting if it contains ANY
// position that scores well, and a mean over 64 positions buries one high score
// under sixty-three low ones. Max over the element-wise absolute value bounds
// the best possible dot product in the block, so a block the summary rejects
// cannot contain a winner - which is what makes the pruning safe rather than
// heuristic.
//
// Summaries are maintained incrementally: only the block a new token lands in is
// dirty, so a decode step rebuilds one summary rather than all of them.
template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseSummaryBuildKernel(LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ dirty_block, uint32_t block_positions, uint16_t *__restrict__ summary_bf16, uint32_t blocks_per_sequence)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,block = dirty_block != 0 ? dirty_block[row] : blockIdx.y;
	uint32_t sequence = sequence_of_row[row],position,index;
	for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
	{
		float best = 0.0f;
		for (position = 0u; position < block_positions; ++position)
		{
			uint32_t absolute_position = (block * block_positions) + position;
			const uint8_t *slot;
			if ( absolute_position >= context_length[sequence] )
				continue;
			slot = LmKvSlotRequired<Geometry>(
				cache,sequence,absolute_position,row,LM_KV_ACCESS_READ);
			if ( slot == 0 )
				return;
			best = fmaxf(best,fabsf(LmBf16ToFloat(((const uint16_t *)slot)[index])));
		}
		summary_bf16[(((uint64_t)sequence * blocks_per_sequence) + block) * INDEX_DIM + index] =
			LmFloatToBf16(best);
	}
	(void)reduction;
}

// Score block summaries. One block per (row, summary block), so the grid is the
// context divided by the block size rather than the context.
//
// The score is an upper bound on any position's score within the block, because
// the summary is an element-wise max of absolute values and the query is dotted
// against it with absolute values too. A block whose bound is below the current
// threshold cannot contain a better position than one already selected.
template<uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseSummaryScoreKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ summary_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ block_count, uint32_t index_heads, uint32_t blocks_per_sequence, float *__restrict__ block_score)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,block = blockIdx.y;
	uint32_t sequence = sequence_of_row[row],head,index;
	float total = 0.0f;
	if ( block >= block_count[sequence] )
	{
		if ( threadIdx.x == 0u )
			block_score[(row * blocks_per_sequence) + block] = -INFINITY;
		return;
	}
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += fabsf(LmBf16ToFloat(index_query_bf16[
				(((uint64_t)row * index_heads) + head) * INDEX_DIM + index]))
				* LmBf16ToFloat(summary_bf16[
				(((uint64_t)sequence * blocks_per_sequence) + block) * INDEX_DIM + index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
		block_score[(row * blocks_per_sequence) + block] = total;
}

// Score positions inside the selected blocks only.
//
// The grid is (row, selected block, position-in-block), so its size is the
// selection budget rather than the context. This is where the speedup lands: the
// expensive per-position dot runs on a fixed number of positions no matter how
// long the conversation is.
template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseRefineKernel(const uint16_t *__restrict__ index_query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_block, uint32_t block_positions, uint32_t index_heads, float *__restrict__ scores, uint32_t *__restrict__ positions_out)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,slot_index = blockIdx.y;
	uint32_t block = selected_block[(row * gridDim.y) + (slot_index / block_positions)];
	uint32_t position = (block * block_positions) + (slot_index % block_positions);
	uint32_t sequence = sequence_of_row[row],head,index;
	const uint8_t *slot;
	float total = 0.0f;
	if ( position >= context_length[sequence] )
	{
		if ( threadIdx.x == 0u )
		{
			scores[(row * gridDim.y) + slot_index] = -INFINITY;
			positions_out[(row * gridDim.y) + slot_index] = position;
		}
		return;
	}
	slot = LmKvSlotRequired<Geometry>(
		cache,sequence,position,row,LM_KV_ACCESS_READ);
	if ( slot == 0 )
		return;
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[
				(((uint64_t)row * index_heads) + head) * INDEX_DIM + index])
				* LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
	{
		scores[(row * gridDim.y) + slot_index] = total;
		positions_out[(row * gridDim.y) + slot_index] = position;
	}
}
