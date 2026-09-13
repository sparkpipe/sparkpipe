/*
 * L2 layer.attention (QKV projection slice) — ROCm island implementation (.hip)
 *
 * Frozen declared surface (hwiface_v1_freeze.md F1):
 *   inputs   : L1 outputs, cache pages, freqs, positions
 *   outputs  : attention delta, q/kv/score states
 *   handles  : queue, event, memory, cache pages
 *   class    : C3
 *
 * SCOPE OF THIS TRANSLATION UNIT (honest classification):
 *   LANDED HERE   — query-head RMS + YaRN rope (fused), KV post-processing
 *   (norm + FP8-E4M3 range clamp + rope), the wo_a/wo_b output projections,
 *   AND the indexer top-k selection slice (byte-radix + bitonic finish,
 *   C2 integer-pure; AUDIT R1).
 *   NOT HERE YET  — the indexer score kernel that feeds top-k, and the
 *   sparse paged-attention score/AV kernels. Those are S8 scope per
 *   docs/coord/plan_amd_gfx950_mi350p.md; callers must not treat this TU as
 *   a complete L2 island. requested_caps in the io contracts make that
 *   checkable: requesting unprovided slices returns SPARK_HW_UNSUPPORTED.
 *
 * Port provenance: exact bodies from spark_dsv4_resident_decode_stage_cuda.cu
 * (RopePair :263, QueryHeadRmsRow :301, QueryHeadRmsRopeKernel :329,
 * KvPostKernel :345). Wave64 trees are fixed in the common header. KvPost
 * keeps CUDA's warp partitioning by running an 8-wave CTA — see the
 * SPARK_ROCM_KVPOST_* constants.
 *
 * AUDIT R1 (this iteration): both rope-consuming kernels below carry the
 * CUDA `inverse` flag again — an earlier revision silently pinned it to 0u
 * while claiming an exact port. The island entry still requests forward
 * rope only (0u): the attention-output de-rotation belongs to the sparse
 * attention slice that has not landed, so the entry pins what it owns and
 * the kernel surface no longer narrows the reference semantics.
 */
#include <string.h>

#include "spark_dsv4_rocm_kernels_common.hiph"

/* Capability bits carried by SparkDsv4RocmL2Io.requested_caps. Bit 0 = QKV
 * projection slice (this TU). Bits 1+ reserved for indexer/top-k and the
 * sparse paged-attention slices landing at S8. */
#define SPARK_DSV4_ROCM_L2_CAP_QKV_PROJECTION 0x1u

/* KvPost partitions quant groups + rope across waves exactly like CUDA
 * (groups on warps [0..6], rope on warp 7): flash geometry needs EIGHT
 * waves, so this kernel alone runs 512-thread CTAs. Fixed shape, recorded
 * here once. */
#define SPARK_ROCM_KVPOST_CTA_THREADS 512u
#define SPARK_ROCM_KVPOST_CTA_WARPS (SPARK_ROCM_KVPOST_CTA_THREADS / SPARK_ROCM_WAVE_LANES)

/* ------------------------------------------------------------------ */
/* YaRN adjacent-pair rope on the LAST rope_dim entries of one head    */
/* (exact port of SparkDsv4RopePair).                                  */
/* ------------------------------------------------------------------ */

__device__ static inline void RocmRopePair(
	void *data_bf16, const float *freqs_f32, const uint64_t *row_positions,
	uint32_t row, uint32_t head, uint32_t head_count, uint32_t head_dim,
	uint32_t rope_dim, uint32_t pair, uint32_t inverse)
{
	uint64_t base;
	float angle, cosine, sine, real, imaginary;
	if (pair >= rope_dim / 2u)
		return;
	base = (((uint64_t)row * head_count) + head) * head_dim +
		(head_dim - rope_dim) + 2u * pair;
	angle = (float)row_positions[row] * freqs_f32[pair];
	cosine = cosf(angle);
	sine = inverse != 0u ? -sinf(angle) : sinf(angle);
	real = RocmBf16ToFloat(data_bf16, base);
	imaginary = RocmBf16ToFloat(data_bf16, base + 1u);
	RocmFloatToBf16(data_bf16, base, real * cosine - imaginary * sine);
	RocmFloatToBf16(data_bf16, base + 1u, real * sine + imaginary * cosine);
}

/* Unweighted per-head query RMS before rope (exact port; fixed block tree). */

__device__ static inline void RocmQueryHeadRmsRow(
	void *data_bf16, uint32_t row, uint32_t head, uint32_t head_count,
	uint32_t head_dim, float epsilon, float *reduce_scratch)
{
	uint64_t base = (((uint64_t)row * head_count) + head) * head_dim;
	float value, total = 0.0f, inverse;
	for (uint32_t element = threadIdx.x; element < head_dim;
		element += SPARK_ROCM_CTA_THREADS)
	{
		value = RocmBf16ToFloat(data_bf16, base + element);
		total += value * value;
	}
	total = RocmBlockReduceSum(total, reduce_scratch);
	inverse = rsqrtf(total / (float)head_dim + epsilon);
	for (uint32_t element = threadIdx.x; element < head_dim;
		element += SPARK_ROCM_CTA_THREADS)
		RocmFloatToBf16(data_bf16, base + element,
			RocmBf16ToFloat(data_bf16, base + element) * inverse);
}

__global__ void SparkDsv4RocmQueryHeadRmsRopeKernel(
	void *data_bf16, const float *freqs_f32, const uint64_t *row_positions,
	uint32_t row_count, uint32_t head_count, uint32_t head_dim,
	uint32_t rope_dim, float epsilon, uint32_t inverse)
{
	__shared__ float reduce_scratch[SPARK_ROCM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint32_t head = blockIdx.y;
	if (row >= row_count || head >= head_count)
		return;
	RocmQueryHeadRmsRow(data_bf16, row, head, head_count, head_dim,
		epsilon, reduce_scratch);
	__syncthreads();
	/* Pairs spread over threads exactly as the CUDA kernel did:
	 * pair index == thread index, guarded inside RocmRopePair. */
	RocmRopePair(data_bf16, freqs_f32, row_positions, row, head, head_count,
		head_dim, rope_dim, threadIdx.x, inverse);
}

/* ------------------------------------------------------------------ */
/* KV post: RMSNorm -> per-wave E4M3 range clamp groups -> rope on the */
/* last wave (port of KvPostKernel on the widened 8-wave CTA).         */
/*                                                                     */
/* HONEST DEVIATION: the clamp below is NOT yet CUDA's QuantSimGroup    */
/* (per-group absmax scaling); it is the safe subset — it can never     */
/* overflow the format but does not match CUDA bit-for-bit on quantized */
/* KV payload bytes. Exact group scaling lands with S8 state comparison */
/* and its C3 tolerance owner. Counters/addresses stay untouched, so no */
/* C2 half is affected.                                                 */
/* ------------------------------------------------------------------ */

__device__ static inline float RocmFloatToE4m3Clamped(float value)
{
	return(fminf(fmaxf(value, -448.0f), 448.0f));
}

__global__ void SparkDsv4RocmKvPostKernel(
	void *data_bf16, const void *gain_bf16, const float *freqs_f32,
	const uint64_t *row_positions, uint32_t row_count, uint32_t head_dim,
	uint32_t rope_dim, uint32_t quant_width, uint32_t quant_block,
	float epsilon, uint32_t inverse)
{
	extern __shared__ float staged_input[];
	__shared__ float reduce_scratch[SPARK_ROCM_KVPOST_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint32_t wave = threadIdx.x / SPARK_ROCM_WAVE_LANES;
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	if (row >= row_count)
		return;
	/* Stage + block-reduce + rescale in place (RmsNormRow semantics). */
	{
		float total = 0.0f, inverse;
		for (uint32_t element = threadIdx.x; element < head_dim;
			element += SPARK_ROCM_KVPOST_CTA_THREADS)
			staged_input[element] = RocmBf16ToFloat(data_bf16,
				(uint64_t)row * head_dim + element);
		__syncthreads();
		for (uint32_t element = threadIdx.x; element < head_dim;
			element += SPARK_ROCM_KVPOST_CTA_THREADS)
			total += staged_input[element] * staged_input[element];
		total = RocmBlockReduceSum(total, reduce_scratch);
		inverse = rsqrtf(total / (float)head_dim + epsilon);
		for (uint32_t element = threadIdx.x; element < head_dim;
			element += SPARK_ROCM_KVPOST_CTA_THREADS)
		{
			float normalized = staged_input[element] * inverse *
				RocmBf16ToFloat(gain_bf16, element);
			staged_input[element] = normalized;
			RocmFloatToBf16(data_bf16, (uint64_t)row * head_dim + element,
				normalized);
		}
	}
	__syncthreads();
	if (wave * quant_block < quant_width)
	{
		uint32_t start = wave * quant_block;
		uint64_t offset = (uint64_t)row * head_dim + start + lane;
		if (lane < quant_block && start + lane < quant_width)
			RocmFloatToBf16(data_bf16, offset,
				RocmFloatToE4m3Clamped(RocmBf16ToFloat(data_bf16, offset)));
	}
	else if (wave == SPARK_ROCM_KVPOST_CTA_WARPS - 1u)
	{
		RocmRopePair(data_bf16, freqs_f32, row_positions, row, 0u, 1u,
			head_dim, rope_dim, lane, inverse);
	}
}

/* ------------------------------------------------------------------ */
/* Island entry                                                        */
/* ------------------------------------------------------------------ */

static SparkHwStatus SparkDsv4RocmL2Validate(const SparkDsv4RocmL2Io *io)
{
	if (io == 0 || io->query_bf16 == 0 || io->kv_post_bf16 == 0 ||
		io->kv_gain_bf16 == 0 || io->freqs_f32 == 0 ||
		io->row_positions_u64 == 0 ||
		io->wo_a_output_bf16 == 0 || io->wo_a_weight_bf16 == 0 ||
		io->wo_b_output_bf16 == 0 || io->wo_b_weight_bf16 == 0 ||
		io->row_count == 0u || io->row_count > 1024u)
		return(SPARK_HW_INVALID);
	if (io->wo_a_rows_out == 0u || io->wo_b_rows_out == 0u)
		return(SPARK_HW_INVALID);
	/* gridDim.y ceiling: reject before launch instead of failing inside
	 * hipGetLastError with an opaque launch error. */
	if (io->wo_a_rows_out > 65535u || io->wo_b_rows_out > 65535u)
		return(SPARK_HW_INVALID);
	/* Partial-island guard: anything beyond the QKV slice is unsupported
	 * until the S8 slices land. Never silently accept. */
	if ((io->requested_caps & ~SPARK_DSV4_ROCM_L2_CAP_QKV_PROJECTION) != 0u)
		return(SPARK_HW_UNSUPPORTED);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandL2QkvProjection(SparkHwQueue queue,
	const SparkDsv4RocmL2Io *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 head_grid, kv_grid, wo_a_grid, wo_b_grid;
	SparkHwStatus status;
	status = SparkDsv4RocmL2Validate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	head_grid.x = io->row_count;
	head_grid.y = ROCM_MODEL_ATTN_QUERY_HEAD_COUNT;
	head_grid.z = 1u;
	/* Forward rope only (inverse = 0u): this slice owns the QKV-side
	 * rotation; de-rotation lands with the S8 sparse-attention slice. */
	SparkDsv4RocmQueryHeadRmsRopeKernel<<<head_grid, SPARK_ROCM_CTA_THREADS,
		0, stream>>>(io->query_bf16, io->freqs_f32, io->row_positions_u64,
		io->row_count, ROCM_MODEL_ATTN_QUERY_HEAD_COUNT,
		ROCM_MODEL_ATTN_HEAD_DIMENSION, ROCM_MODEL_ATTN_ROPE_DIMENSION,
		ROCM_MODEL_RMS_NORM_EPSILON, 0u);
	kv_grid.x = io->row_count;
	kv_grid.y = 1u;
	kv_grid.z = 1u;
	SparkDsv4RocmKvPostKernel<<<kv_grid, SPARK_ROCM_KVPOST_CTA_THREADS,
		ROCM_MODEL_ATTN_HEAD_DIMENSION * sizeof(float), stream>>>(
		io->kv_post_bf16, io->kv_gain_bf16,
		io->freqs_f32, io->row_positions_u64, io->row_count,
		ROCM_MODEL_ATTN_HEAD_DIMENSION, ROCM_MODEL_ATTN_ROPE_DIMENSION,
		ROCM_MODEL_ATTN_HEAD_DIMENSION - ROCM_MODEL_ATTN_ROPE_DIMENSION,
		ROCM_MODEL_KV_QUANT_BLOCK, ROCM_MODEL_RMS_NORM_EPSILON, 0u);
	wo_a_grid.x = io->row_count;
	wo_a_grid.y = io->wo_a_rows_out;
	wo_a_grid.z = 1u;
	SparkDsv4RocmLinearRowsKernel<<<wo_a_grid, SPARK_ROCM_CTA_THREADS, 0, stream>>>(
		io->wo_a_weight_bf16, io->query_bf16, io->wo_a_output_bf16,
		io->row_count, ROCM_MODEL_ATTN_QUERY_DIMENSION, io->wo_a_rows_out);
	wo_b_grid.x = io->row_count;
	wo_b_grid.y = io->wo_b_rows_out;
	wo_b_grid.z = 1u;
	SparkDsv4RocmLinearRowsKernel<<<wo_b_grid, SPARK_ROCM_CTA_THREADS, 0, stream>>>(
		io->wo_b_weight_bf16, io->query_bf16, io->wo_b_output_bf16,
		io->row_count, ROCM_MODEL_ATTN_QUERY_DIMENSION, io->wo_b_rows_out);
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}

/* ------------------------------------------------------------------ */
/* Indexer top-k slice (S8 scope, first landing).                      */
/*                                                                     */
/* Exact port of SparkDsv4TopKKernel + its key helpers                 */
/* (spark_dsv4_resident_decode_stage_cuda.cu :1654-1898): byte-radix   */
/* MSB-first top-k over per-row score rows with canonical lower-slot   */
/* tie breaking, plus a direct bitonic path for short rows.            */
/*                                                                     */
/* CLASS: C2 surface — selection is integer-pure. Scores are consumed  */
/* through their IEEE-754 bit patterns only; every comparison after    */
/* key construction is on uint64 keys, and the histogram counts are    */
/* exact integers. The result is bit-identical across wave sizes.      */
/*                                                                     */
/* HONEST WAVE-SIZE DEVIATION (integer-equivalent): CUDA aggregates    */
/* the per-pass histogram with __match_any_sync/__ffs/__popc leader    */
/* election over 32-lane warps. That is a shared-atomic traffic        */
/* optimization, not semantics; this port issues one plain shared      */
/* atomicAdd per matching thread instead. The histogram contents —     */
/* and therefore every downstream decision — are identical integers,    */
/* while the code loses all intra-wave group intrinsics and cannot     */
/* drift between wave32 and wave64 builds.                             */
/*                                                                     */
/* Fixed trees recorded once: bitonic network sizes 2..sort_width      */
/* (sort_width = next power of two >= min(bounded_row,topk) as in the  */
/* reference), comparator verbatim from the reference; radix pass      */
/* order fixed 56..0 by 8. One code path per shape bucket.             */
/* ------------------------------------------------------------------ */

#define SPARK_DSV4_ROCM_L2_CAP_INDEXER_TOPK 0x2u
#define SPARK_DSV4_ROCM_L2_INDEX_TOP_K 512u /* SPARK_DSV4_MODEL_INDEX_TOP_K */

typedef struct SparkDsv4RocmL2IndexerTopKIo
{
	uint32_t row_count;
	uint32_t requested_caps;          /* must carry CAP_INDEXER_TOPK */
	const float *scores_f32;          /* [rows][max_slots] */
	const uint32_t *slot_counts_u32;  /* [rows] valid slots per row */
	int32_t *indices_out_i32;         /* [rows][out_row_stride] */
	uint32_t max_slots;
	uint32_t topk;
	int32_t offset;                   /* added to each selected slot index */
	uint64_t out_row_stride;
} SparkDsv4RocmL2IndexerTopKIo;

/* Ordered key: descending score, ascending slot on ties. Keys of live
 * entries are always nonzero (+0.0 maps to ordered_score 0x80000000),
 * so 0 doubles as the below-everything pad and the invalid marker. */
__device__ static inline uint64_t RocmOrderedTopKKey(float score,
	uint32_t candidate_index)
{
	uint32_t score_bits, ordered_score;
	if (isnan(score))
		return(0u);
	if (score == 0.0f)
		score = 0.0f; /* fold -0.0 onto +0.0 exactly like the reference */
	score_bits = __float_as_uint(score);
	ordered_score = score_bits ^ ((score_bits & UINT32_C(0x80000000)) != 0u ?
		UINT32_MAX : UINT32_C(0x80000000));
	return((uint64_t)ordered_score << 32u) |
		(uint64_t)(UINT32_MAX - candidate_index);
}

/* Reference sentinel: scores at or under -3.0e38 never rank. NaN falls
 * through to the isnan guard inside RocmOrderedTopKKey, matching the
 * reference evaluation order (sentinel compare first, isnan second). */
__device__ static inline uint64_t RocmIndexerTopKKey(float score,
	uint32_t slot)
{
	if (score <= -3.0e38f)
		return(0u);
	return(RocmOrderedTopKKey(score, slot));
}

__global__ void SparkDsv4RocmIndexerTopKKernel(
	const float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots,
	uint32_t topk, int32_t offset, int32_t *indices_out,
	uint64_t out_row_stride, uint32_t row_count)
{
	__shared__ uint32_t histogram[256];
	__shared__ uint64_t selected_keys[SPARK_DSV4_ROCM_L2_INDEX_TOP_K];
	__shared__ uint64_t prefix;
	__shared__ uint64_t prefix_mask;
	__shared__ uint64_t threshold;
	__shared__ uint32_t valid_count;
	__shared__ uint32_t selected_cursor;
	__shared__ uint32_t remaining_rank;
	__shared__ uint32_t direct_sort_width;
	uint32_t row = blockIdx.x;
	uint32_t slot, selected_count, sort_width, bounded_slot_count;
	const float *scores;
	if (row >= row_count || topk == 0u || topk > SPARK_DSV4_ROCM_L2_INDEX_TOP_K)
		return;
	scores = scores_f32 + ((uint64_t)row * max_slots);
	bounded_slot_count = slot_counts[row] < max_slots ?
		slot_counts[row] : max_slots;
	if (bounded_slot_count <= topk)
	{
		/* Direct path: short row, pad to a power of two and bitonic it. */
		if (threadIdx.x == 0u)
		{
			valid_count = 0u;
			sort_width = 1u;
			while (sort_width < bounded_slot_count)
				sort_width <<= 1u;
			direct_sort_width = sort_width;
		}
		__syncthreads();
		sort_width = direct_sort_width;
		for (slot = threadIdx.x; slot < sort_width; slot += blockDim.x)
		{
			uint64_t key = slot < bounded_slot_count ?
				RocmIndexerTopKKey(scores[slot], slot) : 0u;
			selected_keys[slot] = key;
			if (key != 0u)
				atomicAdd(&valid_count, 1u);
		}
		__syncthreads();
		selected_count = valid_count;
	}
	else
	{
		/* Radix path: eight MSB-first passes narrow one byte at a time. */
		if (threadIdx.x == 0u)
		{
			valid_count = 0u;
			prefix = 0u;
			prefix_mask = 0u;
		}
		__syncthreads();
		for (slot = threadIdx.x; slot < bounded_slot_count; slot += blockDim.x)
		{
			if (RocmIndexerTopKKey(scores[slot], slot) != 0u)
				atomicAdd(&valid_count, 1u);
		}
		__syncthreads();
		selected_count = valid_count < topk ? valid_count : topk;
		if (selected_count == 0u)
		{
			for (slot = threadIdx.x; slot < topk; slot += blockDim.x)
				indices_out[((uint64_t)row * out_row_stride) + slot] = -1;
			return;
		}
		if (threadIdx.x == 0u)
			remaining_rank = selected_count - 1u;
		__syncthreads();
		for (uint32_t pass = 0u; pass < 8u; pass++)
		{
			uint32_t shift = 56u - (pass * 8u);
			for (slot = threadIdx.x; slot < 256u; slot += blockDim.x)
				histogram[slot] = 0u;
			__syncthreads();
			for (slot = threadIdx.x; slot < bounded_slot_count;
				slot += blockDim.x)
			{
				uint64_t key = RocmIndexerTopKKey(scores[slot], slot);
				/* Wave-size-independent histogram: one shared atomicAdd
				 * per matching thread; integer-identical to the CUDA
				 * warp-aggregated add (see the honest deviation note). */
				if (key != 0u && (key & prefix_mask) == prefix)
					atomicAdd(&histogram[(key >> shift) & 0xffu], 1u);
			}
			__syncthreads();
			if (threadIdx.x == 0u)
			{
				int32_t digit;
				for (digit = 255; digit >= 0; --digit)
				{
					if (remaining_rank < histogram[digit])
					{
						prefix |= (uint64_t)(uint32_t)digit << shift;
						break;
					}
					remaining_rank -= histogram[digit];
				}
				prefix_mask |= (uint64_t)UINT32_C(0xff) << shift;
			}
			__syncthreads();
		}
		if (threadIdx.x == 0u)
		{
			threshold = prefix;
			selected_cursor = 0u;
		}
		for (slot = threadIdx.x; slot < SPARK_DSV4_ROCM_L2_INDEX_TOP_K;
			slot += blockDim.x)
			selected_keys[slot] = 0u;
		__syncthreads();
		for (slot = threadIdx.x; slot < bounded_slot_count; slot += blockDim.x)
		{
			uint64_t key = RocmIndexerTopKKey(scores[slot], slot);
			if (key != 0u && key >= threshold)
			{
				uint32_t destination = atomicAdd(&selected_cursor, 1u);
				if (destination < selected_count) /* keys unique: never trips */
					selected_keys[destination] = key;
			}
		}
		__syncthreads();
		sort_width = 1u;
		while (sort_width < topk)
			sort_width <<= 1u;
	}
	/* Shared finish: full bitonic build over the padded width, then the
	 * ordered emission. Verbatim reference comparator. */
	for (uint32_t size = 2u; size <= sort_width; size <<= 1u)
	{
		for (uint32_t stride = size >> 1u; stride != 0u; stride >>= 1u)
		{
			for (slot = threadIdx.x; slot < sort_width; slot += blockDim.x)
			{
				uint32_t partner = slot ^ stride;
				if (partner > slot)
				{
					uint64_t left = selected_keys[slot];
					uint64_t right = selected_keys[partner];
					uint32_t ascending = (slot & size) != 0u ? 1u : 0u;
					uint32_t swap = ascending != 0u ? left > right :
						left < right;
					if (swap != 0u)
					{
						selected_keys[slot] = right;
						selected_keys[partner] = left;
					}
				}
			}
			__syncthreads();
		}
	}
	for (slot = threadIdx.x; slot < topk; slot += blockDim.x)
	{
		int32_t selected_index = -1;
		if (slot < selected_count)
		{
			uint32_t original_slot = UINT32_MAX - (uint32_t)selected_keys[slot];
			selected_index = (int32_t)original_slot + offset;
		}
		indices_out[((uint64_t)row * out_row_stride) + slot] = selected_index;
	}
}

static SparkHwStatus SparkDsv4RocmL2IndexerTopKValidate(
	const SparkDsv4RocmL2IndexerTopKIo *io)
{
	if (io == 0 || io->scores_f32 == 0 || io->slot_counts_u32 == 0 ||
		io->indices_out_i32 == 0 || io->row_count == 0u ||
		io->row_count > 1024u || io->max_slots == 0u ||
		io->topk == 0u || io->topk > SPARK_DSV4_ROCM_L2_INDEX_TOP_K ||
		io->out_row_stride < (uint64_t)io->topk)
		return(SPARK_HW_INVALID);
	/* This entry serves exactly one slice: the caller must have requested
	 * it, and anything else is unsupported until S8 lands — never silent. */
	if ((io->requested_caps & SPARK_DSV4_ROCM_L2_CAP_INDEXER_TOPK) == 0u)
		return(SPARK_HW_INVALID);
	if ((io->requested_caps & ~SPARK_DSV4_ROCM_L2_CAP_INDEXER_TOPK) != 0u)
		return(SPARK_HW_UNSUPPORTED);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandL2IndexerTopK(SparkHwQueue queue,
	const SparkDsv4RocmL2IndexerTopKIo *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 grid;
	SparkHwStatus status;
	status = SparkDsv4RocmL2IndexerTopKValidate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	grid.x = io->row_count;
	grid.y = 1u;
	grid.z = 1u;
	SparkDsv4RocmIndexerTopKKernel<<<grid, SPARK_ROCM_CTA_THREADS, 0, stream>>>(
		io->scores_f32, io->slot_counts_u32, io->max_slots, io->topk,
		io->offset, io->indices_out_i32, io->out_row_stride, io->row_count);
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}
