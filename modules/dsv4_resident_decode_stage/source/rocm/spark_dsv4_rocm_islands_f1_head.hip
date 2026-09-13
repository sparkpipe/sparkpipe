/*
 * F1 head.final — ROCm island implementation (.hip)
 *
 * Frozen declared surface (hwiface_v1_freeze.md F1):
 *   inputs   : final boundary packet, head weights
 *   outputs  : output token ids (+scores), resident token/lane feedback
 *   handles  : queue, event, memory
 *   class    : split verdict — C2: token ids + maxloc pack/unpack + feedback
 *              integers; C3: scores + quantized head outputs
 *
 * SCOPE (honest classification):
 *   LANDED HERE — hc head reduce, MXFP4 head shadow quantize (E2M1 payload +
 *   E8M0 scale + certified error norm), screen two-pass bound, exact bf16
 *   rescore argmax fused with ordered maxloc packing, maxloc unpack,
 *   resident token/lane feedback, accum-u64-max.
 *   NOT HERE    — markov bias term and the DSpark drafter heads (freeze Q3
 *   keeps them INSIDE F1; they remain unported). requested_caps makes that
 *   checkable exactly like the L2 partial-island guard.
 *
 * Screened-pipeline deviation (target-internal, R1): CUDA splits the exact
 * pass into chunked fallback partials + rescore-argmax when the survivor
 * count exceeds SPARK_LM_HEAD_SCREEN_CAP. Here the exact rescore runs the
 * survivors directly in one block-per-row kernel. Identical result set
 * (winner + score + packed maxloc), different work shape; the chunked
 * machinery is S8 perf work.
 */
#include <string.h>

#include "spark_dsv4_rocm_kernels_common.hiph"

#define SPARK_DSV4_ROCM_F1_CAP_HEAD_REDUCE 0x1u
#define SPARK_DSV4_ROCM_F1_CAP_SCREENED_ARGMAX 0x2u
#define SPARK_DSV4_ROCM_F1_CAP_FEEDBACK 0x4u
#define SPARK_DSV4_ROCM_F1_CAP_KNOWN \
	(SPARK_DSV4_ROCM_F1_CAP_HEAD_REDUCE | \
	 SPARK_DSV4_ROCM_F1_CAP_SCREENED_ARGMAX | \
	 SPARK_DSV4_ROCM_F1_CAP_FEEDBACK)

/* Measured shared-kernel facts (spark_lm_kernels.cuh :1192-1204). */
#define ROCM_HEAD_SHADOW_GROUP 32u
#define ROCM_HEAD_SCREEN_SLACK_COARSE 0.0078125f
#define ROCM_HEAD_NEGATIVE_INFINITY (-3.0e38f)

/* ------------------------------------------------------------------ */
/* Ordered score packing (exact port of SparkDsv4OrderedHeadScore +    */
/* HeadMaxlocPack/Unpack, .cu:52-86): total order over f32 with NaN    */
/* first-forced-lowest, ties resolved to LOWEST candidate id via       */
/* UINT32_MAX - id. The packed u64 comparison key is reduction-order   */
/* INDEPENDENT, which is what makes F1's C2 half portable across      */
/* wave64 trees.                                                       */
/* ------------------------------------------------------------------ */

__device__ static inline uint32_t RocmOrderedHeadScore(float score)
{
	uint32_t bits;
	if (isnan(score))
		return(0u);
	if (score == 0.0f)
		score = 0.0f;
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ?
		UINT32_MAX : UINT32_C(0x80000000)));
}

/* E2M1 / E8M0 codecs (exact port of cuh :1207-1216 semantics). */

__device__ static inline uint32_t RocmEncodeE2m1(float value)
{
	const float edges[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
	uint32_t sign = value < 0.0f ? 8u : 0u, code = 0u;
	float magnitude = fabsf(value);
	for (uint32_t edge = 0u; edge < 7u; edge++)
		if (magnitude >= edges[edge])
			code = edge + 1u;
	return(sign | code);
}

__device__ static inline float RocmDecodeE2m1(uint32_t code)
{
	/* Magnitude set {0,.5,1,1.5,2,3,4,6}, bit 3 = sign. */
	static const float magnitudes[8] =
		{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
	float magnitude = magnitudes[code & 7u];
	return((code & 8u) != 0u ? -magnitude : magnitude);
}

__device__ static inline float RocmDecodeE8m0(uint32_t exponent)
{
	return(__uint_as_float(exponent << 23u));
}

/* Block-wide scalar max with lowest-index tie semantics NOT needed here:
 * only the numeric bound is maximized (screen pass one). Fixed tree. */

__device__ static inline float RocmBlockReduceMax(float value, float *scratch)
{
	value = fmaxf(value, __shfl_down(value, SPARK_ROCM_WAVE_LANES / 2u));
	value = fmaxf(value, __shfl_down(value, SPARK_ROCM_WAVE_LANES / 4u));
	value = fmaxf(value, __shfl_down(value, SPARK_ROCM_WAVE_LANES / 8u));
	value = fmaxf(value, __shfl_down(value, SPARK_ROCM_WAVE_LANES / 16u));
	value = fmaxf(value, __shfl_down(value, SPARK_ROCM_WAVE_LANES / 32u));
	value = fmaxf(value, __shfl_down(value, 1u));
	if ((threadIdx.x & (SPARK_ROCM_WAVE_LANES - 1u)) == 0u)
		scratch[threadIdx.x / SPARK_ROCM_WAVE_LANES] = value;
	__syncthreads();
	if (threadIdx.x == 0u)
	{
		float best = scratch[0];
		for (uint32_t wave = 1u; wave < SPARK_ROCM_CTA_WARPS; wave++)
			best = fmaxf(best, scratch[wave]);
		scratch[0] = best;
	}
	__syncthreads();
	return(scratch[0]);
}

/* ------------------------------------------------------------------ */
/* Stage A — hc head reduce (exact port of HcHeadReduceKernel .cu:2223)*/
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmHcHeadReduceKernel(
	const void *streams_bf16, const float *mixes_f32, float scale,
	const float *base_f32, float epsilon, void *reduced_bf16,
	uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	uint32_t row = blockIdx.x;
	float value, pre;
	if (row >= row_count)
		return;
	for (uint32_t element = threadIdx.x; element < dimension;
		element += SPARK_ROCM_CTA_THREADS)
	{
		value = 0.0f;
		for (uint32_t stream = 0u; stream < hc; stream++)
		{
			pre = 1.0f / (1.0f + expf(-(mixes_f32[(uint64_t)row * hc + stream] *
				scale + base_f32[stream]))) + epsilon;
			value += pre * RocmBf16ToFloat(streams_bf16,
				(((uint64_t)row * hc) + stream) * dimension + element);
		}
		RocmFloatToBf16(reduced_bf16, (uint64_t)row * dimension + element, value);
	}
}

/* ------------------------------------------------------------------ */
/* Stage B — head shadow quantize (port of cuh :1221; wave per neuron) */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmHeadShadowQuantizeKernel(
	const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale_e8m0,
	float *error_norm_f32, uint32_t candidate_count, uint32_t hidden_dimension)
{
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	uint32_t neuron = blockIdx.x * SPARK_ROCM_CTA_WARPS +
		(threadIdx.x / SPARK_ROCM_WAVE_LANES);
	uint64_t row_base = (uint64_t)neuron * hidden_dimension;
	float error_squares = 0.0f;
	if (neuron >= candidate_count)
		return;
	for (uint32_t group = lane;
		group < (hidden_dimension / ROCM_HEAD_SHADOW_GROUP);
		group += SPARK_ROCM_WAVE_LANES)
	{
		float absmax = 0.0f, scale_value, exact, coded;
		uint32_t exponent = 127u;
		int32_t clamped;
		for (uint32_t element = 0u; element < ROCM_HEAD_SHADOW_GROUP; element++)
			absmax = fmaxf(absmax, fabsf(RocmBf16ToFloat(head_bf16,
				row_base + (group * ROCM_HEAD_SHADOW_GROUP) + element)));
		if (absmax > 0.0f)
		{
			clamped = (int32_t)ceilf(log2f(absmax / 6.0f));
			clamped = clamped < -127 ? -127 : (clamped > 127 ? 127 : clamped);
			exponent = (uint32_t)(clamped + 127);
		}
		shadow_scale_e8m0[((uint64_t)neuron *
			(hidden_dimension / ROCM_HEAD_SHADOW_GROUP)) + group] =
			(uint8_t)exponent;
		scale_value = RocmDecodeE8m0(exponent);
		for (uint32_t element = 0u; element < ROCM_HEAD_SHADOW_GROUP;
			element += 8u)
		{
			uint32_t packed = 0u;
			for (uint32_t nibble = 0u; nibble < 8u; nibble++)
			{
				exact = RocmBf16ToFloat(head_bf16, row_base +
					(group * ROCM_HEAD_SHADOW_GROUP) + element + nibble);
				packed |= RocmEncodeE2m1(scale_value > 0.0f ?
					exact / scale_value : 0.0f) << (nibble << 2u);
				coded = RocmDecodeE2m1((packed >> (nibble << 2u)) & 0x0fu) *
					scale_value;
				error_squares = fmaf(exact - coded, exact - coded,
					error_squares);
			}
			((uint32_t *)shadow_payload)[((row_base +
				(group * ROCM_HEAD_SHADOW_GROUP) + element) >> 3u)] = packed;
		}
	}
	error_squares = RocmWaveReduceSum(error_squares);
	if (lane == 0u)
		error_norm_f32[neuron] = sqrtf(error_squares);
}

/* ------------------------------------------------------------------ */
/* Stage B2 — coarse logits: MXFP4 shadow x hidden (bring-up GEMV;     */
/* perf shape is S8 work, results are bound-checked by the screen).    */
/* ------------------------------------------------------------------ */

__device__ static inline float RocmMxfp4ShadowDot(
	const uint8_t *shadow_payload, const uint8_t *shadow_scale_e8m0,
	const float *hidden_shared, uint32_t neuron, uint32_t hidden_dimension)
{
	float accumulator = 0.0f;
	uint32_t group_count = hidden_dimension / ROCM_HEAD_SHADOW_GROUP;
	for (uint32_t group = 0u; group < group_count; group++)
	{
		float scale_value = RocmDecodeE8m0(shadow_scale_e8m0[
			(uint64_t)neuron * group_count + group]);
		uint32_t base_element = group * ROCM_HEAD_SHADOW_GROUP;
		for (uint32_t element = 0u; element < ROCM_HEAD_SHADOW_GROUP;
			element += 8u)
		{
			uint32_t packed = ((const uint32_t *)shadow_payload)[
				(((uint64_t)neuron * hidden_dimension) +
				base_element + element) >> 3u];
			for (uint32_t nibble = 0u; nibble < 8u; nibble++)
			{
				float coded = RocmDecodeE2m1((packed >> (nibble << 2u)) & 0x0fu) *
					scale_value;
				accumulator += coded *
					hidden_shared[base_element + element + nibble];
			}
		}
	}
	return(accumulator);
}

__global__ void SparkDsv4RocmHeadCoarseLogitsKernel(
	const void *hidden_bf16, const uint8_t *shadow_payload,
	const uint8_t *shadow_scale_e8m0, void *coarse_logits_bf16,
	uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	extern __shared__ float hidden_shared[];
	uint32_t row = blockIdx.x;
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	uint32_t wave = threadIdx.x / SPARK_ROCM_WAVE_LANES;
	__shared__ float wave_scores[SPARK_ROCM_CTA_WARPS];
	if (row >= row_count)
		return;
	for (uint32_t element = threadIdx.x; element < hidden_dimension;
		element += SPARK_ROCM_CTA_THREADS)
		hidden_shared[element] = RocmBf16ToFloat(hidden_bf16,
			(uint64_t)row * hidden_dimension + element);
	__syncthreads();
	for (uint32_t candidate = wave; candidate < candidate_count;
		candidate += SPARK_ROCM_CTA_WARPS)
	{
		float total = RocmMxfp4ShadowDot(shadow_payload, shadow_scale_e8m0,
			hidden_shared, candidate, hidden_dimension);
		total = RocmWaveReduceSum(total);
		if (lane == 0u)
			RocmFloatToBf16(coarse_logits_bf16,
				(uint64_t)row * candidate_count + candidate, total);
	}
	(void)wave_scores;
}

/* ------------------------------------------------------------------ */
/* Stage C — screen (exact port of cuh :1266 two-pass bound).          */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmHeadScreenKernel(
	const void *hidden_bf16, const void *coarse_logits_bf16,
	const float *error_norm_f32, uint32_t *candidate_ids,
	uint32_t *candidate_counts, uint32_t row_count, uint32_t candidate_count,
	uint32_t hidden_dimension)
{
	__shared__ float reduce_scratch[SPARK_ROCM_CTA_WARPS];
	__shared__ float shared_norm, shared_bound;
	__shared__ uint32_t shared_cursor;
	uint32_t row = blockIdx.x;
	uint64_t row_base = (uint64_t)row * candidate_count;
	float sum_squares = 0.0f, lower = ROCM_HEAD_NEGATIVE_INFINITY, coarse,
		priced;
	if (row >= row_count)
		return;
	for (uint32_t element = threadIdx.x; element < (hidden_dimension >> 1u);
		element += SPARK_ROCM_CTA_THREADS)
	{
		float a = RocmBf16ToFloat(hidden_bf16,
			((uint64_t)row * hidden_dimension) + (element << 1u));
		float b = RocmBf16ToFloat(hidden_bf16,
			((uint64_t)row * hidden_dimension) + (element << 1u) + 1u);
		sum_squares = fmaf(a, a, fmaf(b, b, sum_squares));
	}
	sum_squares = RocmBlockReduceSum(sum_squares, reduce_scratch);
	if (threadIdx.x == 0u)
	{
		shared_norm = sqrtf(sum_squares);
		shared_cursor = 0u;
	}
	__syncthreads();
	for (uint32_t candidate = threadIdx.x; candidate < candidate_count;
		candidate += SPARK_ROCM_CTA_THREADS)
	{
		coarse = RocmBf16ToFloat(coarse_logits_bf16, row_base + candidate);
		priced = coarse - (shared_norm * error_norm_f32[candidate]) -
			(1.0f + fabsf(coarse) * ROCM_HEAD_SCREEN_SLACK_COARSE);
		lower = priced > lower ? priced : lower;
	}
	shared_bound = RocmBlockReduceMax(lower, reduce_scratch);
	for (uint32_t candidate = threadIdx.x; candidate < candidate_count;
		candidate += SPARK_ROCM_CTA_THREADS)
	{
		coarse = RocmBf16ToFloat(coarse_logits_bf16, row_base + candidate);
		if (coarse + (shared_norm * error_norm_f32[candidate]) +
			(1.0f + fabsf(coarse) * ROCM_HEAD_SCREEN_SLACK_COARSE) <
			shared_bound)
			continue;
		uint32_t slot = atomicAdd(&shared_cursor, 1u);
		if (slot < 4096u /* SPARK_LM_HEAD_SCREEN_CAP */)
			candidate_ids[(uint64_t)row * 4096u + slot] = candidate;
	}
	__syncthreads();
	if (threadIdx.x == 0u)
		candidate_counts[row] = shared_cursor;
}

/* ------------------------------------------------------------------ */
/* Stage D — exact bf16 rescore over survivors, fused with ordered     */
/* maxloc packing (replaces CUDA's fallback-partials + RescoreArgmax   */
/* pair with one kernel; same results, see header deviation note).     */
/* Winner selection compares the PACKED u64 key, so equal scores       */
/* resolve to the lowest candidate id regardless of reduction shape.   */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmHeadExactRescoreArgmaxKernel(
	const void *hidden_bf16, const void *head_weight_bf16,
	const uint32_t *candidate_ids, const uint32_t *candidate_counts,
	void *logits_bf16, float *output_scores, uint32_t *output_token_ids,
	uint64_t *maxloc, uint32_t row_count, uint32_t hidden_dimension)
{
	extern __shared__ float hidden_shared[];
	__shared__ uint64_t wave_best[SPARK_ROCM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	uint32_t wave = threadIdx.x / SPARK_ROCM_WAVE_LANES;
	uint32_t survivors = candidate_counts[row];
	uint64_t best_key = 0ull; /* ordered-score 0 => NaN => loses everything */
	if (row >= row_count)
		return;
	/* Every wave slot must hold a defined key even if that wave sees no
	 * survivor (survivors < wave count); 0 loses to any real key. */
	if (threadIdx.x < SPARK_ROCM_CTA_WARPS)
		wave_best[threadIdx.x] = 0ull;
	__syncthreads();
	if (survivors > 4096u)
		survivors = 4096u;
	for (uint32_t element = threadIdx.x; element < hidden_dimension;
		element += SPARK_ROCM_CTA_THREADS)
		hidden_shared[element] = RocmBf16ToFloat(hidden_bf16,
			(uint64_t)row * hidden_dimension + element);
	__syncthreads();
	for (uint32_t slot = wave; slot < survivors;
		slot += SPARK_ROCM_CTA_WARPS)
	{
		uint32_t candidate = candidate_ids[(uint64_t)row * 4096u + slot];
		float accumulator = 0.0f;
		uint64_t weight_base = (uint64_t)candidate * hidden_dimension;
		for (uint32_t element = lane; element < hidden_dimension;
			element += SPARK_ROCM_WAVE_LANES)
			accumulator += hidden_shared[element] *
				RocmBf16ToFloat(head_weight_bf16, weight_base + element);
		accumulator = RocmWaveReduceSum(accumulator);
		if (lane == 0u)
		{
			uint64_t key = ((uint64_t)RocmOrderedHeadScore(accumulator) << 32u) |
				(UINT32_MAX - candidate);
			RocmFloatToBf16(logits_bf16, (uint64_t)row * 4096u + slot,
				accumulator);
			if (key > best_key)
				best_key = key;
		}
	}
	if (lane == 0u)
		wave_best[wave] = best_key;
	__syncthreads();
	if (threadIdx.x == 0u)
	{
		uint64_t row_best = (survivors != 0u) ? wave_best[0] : 0ull;
		for (uint32_t w = 1u; w < SPARK_ROCM_CTA_WARPS; w++)
			if (wave_best[w] > row_best)
				row_best = wave_best[w];
		maxloc[row] = row_best;
		output_token_ids[row] = UINT32_MAX - (uint32_t)row_best;
		if (output_scores != 0)
		{
			uint32_t ordered = (uint32_t)(row_best >> 32u);
			uint32_t bits = ordered ^ ((ordered & UINT32_C(0x80000000)) != 0u ?
				UINT32_C(0x80000000) : UINT32_MAX);
			output_scores[row] = __uint_as_float(bits);
		}
	}
}

/* ------------------------------------------------------------------ */
/* Stage E/F — maxloc unpack + resident feedback + accum u64 max       */
/* (exact ports, .cu:77-129).                                          */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmHeadMaxlocUnpackKernel(
	const uint64_t *maxloc, uint32_t *token_ids, uint32_t row_count)
{
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < row_count)
		token_ids[row] = UINT32_MAX - (uint32_t)maxloc[row];
}

__global__ void SparkDsv4RocmResidentTokenFeedbackKernel(
	const uint32_t *output_token_ids, uint32_t *resident_token_ids,
	uint32_t *input_token_ids, uint64_t *row_positions,
	uint64_t *row_emit_positions, uint64_t *row_emit_positions_hca,
	uint32_t row_count, uint32_t tokens_per_sequence, uint32_t step_index,
	uint32_t advance)
{
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t position;
	if (row >= row_count)
		return;
	uint32_t token = output_token_ids[row];
	resident_token_ids[row * tokens_per_sequence + step_index] = token;
	if (advance == 0u)
		return;
	position = row_positions[row] + 1u;
	input_token_ids[row] = token;
	row_positions[row] = position;
	row_emit_positions[row] = position + 1u >= 4u /* CSA ratio */ ?
		position + 1u - 4u : 0u;
	row_emit_positions_hca[row] = position + 1u >= 128u /* HCA ratio */ ?
		position + 1u - 128u : 0u;
}

__global__ void SparkDsv4RocmAccumU64MaxKernel(
	uint64_t *destination, const uint64_t *source, uint32_t element_count)
{
	uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	if (element < element_count && source[element] > destination[element])
		destination[element] = source[element];
}

/* ------------------------------------------------------------------ */
/* Island io + entry                                                   */
/* ------------------------------------------------------------------ */

typedef struct SparkDsv4RocmF1Io
{
	uint32_t row_count;
	uint32_t requested_caps;          /* SPARK_DSV4_ROCM_F1_CAP_* mask */
	uint32_t candidate_count;
	/* Head reduce inputs. */
	const void *streams_bf16;         /* [rows][hc*hidden] boundary packet */
	const float *mixes_f32;           /* [rows][hc] */
	float scale;
	const float *base_f32;            /* [hc] */
	float epsilon;
	/* Screened head inputs (bf16 head weights; MXFP4 shadow derived here). */
	const void *head_weight_bf16;     /* [candidate_count][hidden] */
	/* Scratch sinks (slot workspace views; never dynamically allocated). */
	void *reduced_hidden_bf16;        /* [rows][hidden] */
	uint8_t *shadow_payload;          /* candidate_count*hidden/2 bytes */
	uint8_t *shadow_scale_e8m0;       /* candidate_count*hidden/32 bytes */
	float *error_norm_f32;            /* [candidate_count] */
	void *coarse_logits_bf16;         /* [rows][candidate_count] */
	void *rescored_logits_bf16;       /* [rows][4096] survivor logits */
	uint32_t *candidate_ids;          /* [rows][4096] */
	uint32_t *candidate_counts;       /* [rows] */
	float *output_scores_f32;         /* [rows], may be null */
	uint32_t *output_token_ids;       /* [rows] */
	uint64_t *maxloc_u64;             /* [rows] */
	/* Feedback lanes. */
	uint32_t *resident_token_ids;
	uint32_t *input_token_ids;
	uint64_t *row_positions_u64;
	uint64_t *row_emit_positions_u64;
	uint64_t *row_emit_positions_hca_u64;
	uint32_t tokens_per_sequence;
	uint32_t step_index;
	uint32_t advance;
	uint64_t *accum_dest_u64;         /* optional; null skips stage F */
	const uint64_t *accum_src_u64;
	uint32_t accum_element_count;
} SparkDsv4RocmF1Io;

static SparkHwStatus SparkDsv4RocmF1Validate(const SparkDsv4RocmF1Io *io)
{
	if (io == 0 || io->row_count == 0u || io->row_count > 1024u ||
		io->candidate_count == 0u ||
		(io->requested_caps & ~SPARK_DSV4_ROCM_F1_CAP_KNOWN) != 0u)
		return(SPARK_HW_INVALID);
	if ((io->requested_caps & SPARK_DSV4_ROCM_F1_CAP_HEAD_REDUCE) != 0u &&
		(io->streams_bf16 == 0 || io->mixes_f32 == 0 || io->base_f32 == 0 ||
		 io->reduced_hidden_bf16 == 0))
		return(SPARK_HW_INVALID);
	if ((io->requested_caps & SPARK_DSV4_ROCM_F1_CAP_SCREENED_ARGMAX) != 0u &&
		(io->head_weight_bf16 == 0 || io->shadow_payload == 0 ||
		 io->shadow_scale_e8m0 == 0 || io->error_norm_f32 == 0 ||
		 io->coarse_logits_bf16 == 0 || io->rescored_logits_bf16 == 0 ||
		 io->candidate_ids == 0 || io->candidate_counts == 0 ||
		 io->output_token_ids == 0 || io->maxloc_u64 == 0 ||
		 (ROCM_MODEL_HIDDEN_DIMENSION %
			ROCM_HEAD_SHADOW_GROUP) != 0u))
		return(SPARK_HW_INVALID);
	if ((io->requested_caps & SPARK_DSV4_ROCM_F1_CAP_FEEDBACK) != 0u &&
		(io->resident_token_ids == 0 || io->input_token_ids == 0 ||
		 io->row_positions_u64 == 0 || io->row_emit_positions_u64 == 0 ||
		 io->row_emit_positions_hca_u64 == 0 ||
		 io->tokens_per_sequence == 0u ||
		 io->step_index >= io->tokens_per_sequence || io->advance > 1u))
		return(SPARK_HW_INVALID);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandF1(SparkHwQueue queue,
	const SparkDsv4RocmF1Io *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 grid;
	SparkHwStatus status;
	status = SparkDsv4RocmF1Validate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	if ((io->requested_caps & SPARK_DSV4_ROCM_F1_CAP_HEAD_REDUCE) != 0u)
	{
		grid.x = io->row_count;
		grid.y = 1u;
		grid.z = 1u;
		SparkDsv4RocmHcHeadReduceKernel<<<grid, SPARK_ROCM_CTA_THREADS, 0,
			stream>>>(io->streams_bf16, io->mixes_f32, io->scale,
			io->base_f32, io->epsilon, io->reduced_hidden_bf16,
			io->row_count, ROCM_MODEL_HC_STREAM_COUNT,
			ROCM_MODEL_HIDDEN_DIMENSION);
	}
	if ((io->requested_caps & SPARK_DSV4_ROCM_F1_CAP_SCREENED_ARGMAX) != 0u)
	{
		/* B: shadow quantize — one wave per neuron stripe. */
		grid.x = (io->candidate_count + SPARK_ROCM_CTA_WARPS - 1u) /
			SPARK_ROCM_CTA_WARPS;
		grid.y = 1u;
		grid.z = 1u;
		SparkDsv4RocmHeadShadowQuantizeKernel<<<grid, SPARK_ROCM_CTA_THREADS,
			0, stream>>>(io->head_weight_bf16, io->shadow_payload,
			io->shadow_scale_e8m0, io->error_norm_f32, io->candidate_count,
			ROCM_MODEL_HIDDEN_DIMENSION);
		/* B2: coarse logits from the shadow. */
		grid.x = io->row_count;
		SparkDsv4RocmHeadCoarseLogitsKernel<<<grid, SPARK_ROCM_CTA_THREADS,
			ROCM_MODEL_HIDDEN_DIMENSION * sizeof(float), stream>>>(
			io->reduced_hidden_bf16 != 0 ? io->reduced_hidden_bf16 :
			io->streams_bf16, io->shadow_payload, io->shadow_scale_e8m0,
			io->coarse_logits_bf16, io->row_count, io->candidate_count,
			ROCM_MODEL_HIDDEN_DIMENSION);
		/* C: screen. */
		SparkDsv4RocmHeadScreenKernel<<<grid, SPARK_ROCM_CTA_THREADS, 0,
			stream>>>(io->reduced_hidden_bf16 != 0 ?
			io->reduced_hidden_bf16 : io->streams_bf16,
			io->coarse_logits_bf16, io->error_norm_f32, io->candidate_ids,
			io->candidate_counts, io->row_count, io->candidate_count,
			ROCM_MODEL_HIDDEN_DIMENSION);
		/* D: exact rescore + fused ordered-maxloc (C2 key). */
		SparkDsv4RocmHeadExactRescoreArgmaxKernel<<<grid,
			SPARK_ROCM_CTA_THREADS,
			ROCM_MODEL_HIDDEN_DIMENSION * sizeof(float), stream>>>(
			io->reduced_hidden_bf16 != 0 ? io->reduced_hidden_bf16 :
			io->streams_bf16, io->head_weight_bf16, io->candidate_ids,
			io->candidate_counts, io->rescored_logits_bf16,
			io->output_scores_f32, io->output_token_ids, io->maxloc_u64,
			io->row_count, ROCM_MODEL_HIDDEN_DIMENSION);
	}
	if ((io->requested_caps & SPARK_DSV4_ROCM_F1_CAP_FEEDBACK) != 0u)
	{
		grid.x = (io->row_count + SPARK_ROCM_CTA_THREADS - 1u) /
			SPARK_ROCM_CTA_THREADS;
		grid.y = 1u;
		grid.z = 1u;
		SparkDsv4RocmResidentTokenFeedbackKernel<<<grid,
			SPARK_ROCM_CTA_THREADS, 0, stream>>>(io->output_token_ids,
			io->resident_token_ids, io->input_token_ids,
			io->row_positions_u64, io->row_emit_positions_u64,
			io->row_emit_positions_hca_u64, io->row_count,
			io->tokens_per_sequence, io->step_index, io->advance);
		if (io->accum_dest_u64 != 0 && io->accum_src_u64 != 0 &&
			io->accum_element_count != 0u)
			SparkDsv4RocmAccumU64MaxKernel<<<grid, SPARK_ROCM_CTA_THREADS, 0,
				stream>>>(io->accum_dest_u64, io->accum_src_u64,
				io->accum_element_count);
	}
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}
