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
 *   (norm + FP8-E4M3 range clamp + rope), and the wo_a/wo_b output
 *   projections: the "attention QKV projection" internals named in the S7
 *   bring-up order.
 *   NOT HERE      — indexer core + top-k and the sparse paged-attention
 *   score/AV kernels. Those are S8 scope per
 *   docs/coord/plan_amd_gfx950_mi350p.md; callers must not treat this TU as
 *   a complete L2 island. requested_caps in the io contract makes that
 *   checkable: requesting unprovided slices returns SPARK_HW_UNSUPPORTED.
 *
 * Port provenance: exact bodies from spark_dsv4_resident_decode_stage_cuda.cu
 * (RopePair :263, QueryHeadRmsRow :301, QueryHeadRmsRopeKernel :329,
 * KvPostKernel :345). Wave64 trees are fixed in the common header. KvPost
 * keeps CUDA's warp partitioning by running an 8-wave CTA — see the
 * SPARK_ROCM_KVPOST_* constants.
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
	uint32_t rope_dim, float epsilon)
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
		head_dim, rope_dim, threadIdx.x, 0u);
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
	float epsilon)
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
			head_dim, rope_dim, lane, 0u);
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
	SparkDsv4RocmQueryHeadRmsRopeKernel<<<head_grid, SPARK_ROCM_CTA_THREADS,
		0, stream>>>(io->query_bf16, io->freqs_f32, io->row_positions_u64,
		io->row_count, ROCM_MODEL_ATTN_QUERY_HEAD_COUNT,
		ROCM_MODEL_ATTN_HEAD_DIMENSION, ROCM_MODEL_ATTN_ROPE_DIMENSION,
		ROCM_MODEL_RMS_NORM_EPSILON);
	kv_grid.x = io->row_count;
	kv_grid.y = 1u;
	kv_grid.z = 1u;
	SparkDsv4RocmKvPostKernel<<<kv_grid, SPARK_ROCM_KVPOST_CTA_THREADS,
		ROCM_MODEL_ATTN_HEAD_DIMENSION * sizeof(float), stream>>>(
		io->kv_post_bf16, io->kv_gain_bf16,
		io->freqs_f32, io->row_positions_u64, io->row_count,
		ROCM_MODEL_ATTN_HEAD_DIMENSION, ROCM_MODEL_ATTN_ROPE_DIMENSION,
		ROCM_MODEL_ATTN_HEAD_DIMENSION - ROCM_MODEL_ATTN_ROPE_DIMENSION,
		ROCM_MODEL_KV_QUANT_BLOCK, ROCM_MODEL_RMS_NORM_EPSILON);
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
