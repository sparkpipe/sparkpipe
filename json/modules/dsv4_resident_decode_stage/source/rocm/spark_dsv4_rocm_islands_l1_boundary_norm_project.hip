/*
 * L1 layer.boundary_norm_project — ROCm island implementation (.hip)
 *
 * Frozen declared surface (hwiface_v1_freeze.md F1):
 *   inputs   : boundary packet, layer weights
 *   outputs  : normalized hidden, packed Q/KV/indexer shards
 *   handles  : queue, memory
 *   class    : C3 (accumulation; shard permutation verified downstream via C2)
 *
 * Port provenance: kernel bodies are line-faithful ports of
 * spark_dsv4_resident_decode_stage_cuda.cu (HcPostKernel :2190,
 * ProjectionPack/Unpack :2343/:2376) with wave32->wave64 reduction trees
 * re-derived and FIXED here. RMSNorm follows SparkLmRmsNorm semantics
 * (staged f32, gain bf16) under the same fixed tree.
 *
 * S7 scope note: hcPost fold + boundary norm + Q/KV/indexer shard packing
 * land now; the TP>1 unpack path compiles but is exercised only from S8.
 * Entry spelling is provisional pending interface-side pinning; see common
 * header.
 */
#include <string.h>

#include "spark_dsv4_rocm_kernels_common.hiph"

/* ------------------------------------------------------------------ */
/* hcPost fold: out + comb-weighted residual streams -> new packet     */
/* (exact port of HcPostKernel arithmetic order: __fmul_rn then        */
/*  __fmaf_rn chain in source order — C3 tree fixed per bucket).       */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmHcPostKernel(
	const void *out_bf16, const void *residual_bf16,
	const float *post_f32, const float *comb_f32,
	void *streams_bf16, uint32_t row_count,
	uint32_t dimension, uint32_t tiles_per_row)
{
	__shared__ float post[ROCM_MODEL_HC_STREAM_COUNT];
	__shared__ float comb[ROCM_MODEL_HC_STREAM_COUNT * ROCM_MODEL_HC_STREAM_COUNT];
	uint32_t row = blockIdx.y;
	uint32_t tile = blockIdx.x;
	float residual[ROCM_MODEL_HC_STREAM_COUNT], out, value;
	if (row >= row_count || threadIdx.x >= blockDim.x)
		return;
	if (threadIdx.x < ROCM_MODEL_HC_STREAM_COUNT)
		post[threadIdx.x] = post_f32[(uint64_t)row * ROCM_MODEL_HC_STREAM_COUNT + threadIdx.x];
	if (threadIdx.x < ROCM_MODEL_HC_STREAM_COUNT * ROCM_MODEL_HC_STREAM_COUNT)
		comb[threadIdx.x] = comb_f32[(uint64_t)row * (ROCM_MODEL_HC_STREAM_COUNT * ROCM_MODEL_HC_STREAM_COUNT) + threadIdx.x];
	__syncthreads();
	for (uint32_t element = tile * blockDim.x + threadIdx.x; element < dimension;
		element += tiles_per_row * blockDim.x)
	{
		out = RocmBf16ToFloat(out_bf16, (uint64_t)row * dimension + element);
		#pragma unroll
		for (uint32_t source = 0u; source < ROCM_MODEL_HC_STREAM_COUNT; source++)
			residual[source] = RocmBf16ToFloat(residual_bf16,
				((uint64_t)row * ROCM_MODEL_HC_STREAM_COUNT + source) * dimension + element);
		#pragma unroll
		for (uint32_t stream = 0u; stream < ROCM_MODEL_HC_STREAM_COUNT; stream++)
		{
			value = __fmul_rn(post[stream], out);
			#pragma unroll
			for (uint32_t source = 0u; source < ROCM_MODEL_HC_STREAM_COUNT; source++)
				value = __fmaf_rn(comb[source * ROCM_MODEL_HC_STREAM_COUNT + stream],
					residual[source], value);
			RocmFloatToBf16(streams_bf16,
				((uint64_t)row * ROCM_MODEL_HC_STREAM_COUNT + stream) * dimension + element,
				value);
		}
	}
}

/* ------------------------------------------------------------------ */
/* RMSNorm over one hidden row: staged f32, bf16 gain, fixed tree      */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmRmsNormKernel(
	const void *input_bf16, const void *gain_bf16, void *output_bf16,
	uint32_t dimension, float epsilon)
{
	extern __shared__ float staged_input[];
	__shared__ float reduce_scratch[SPARK_ROCM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint64_t base = (uint64_t)row * dimension;
	float value, total;
	for (uint32_t element = threadIdx.x; element < dimension;
		element += SPARK_ROCM_CTA_THREADS)
		staged_input[element] = RocmBf16ToFloat(input_bf16, base + element);
	__syncthreads();
	total = 0.0f;
	for (uint32_t element = threadIdx.x; element < dimension;
		element += SPARK_ROCM_CTA_THREADS)
		total += staged_input[element] * staged_input[element];
	total = RocmBlockReduceSum(total, reduce_scratch);
	float inverse = rsqrtf(total / (float)dimension + epsilon);
	for (uint32_t element = threadIdx.x; element < dimension;
		element += SPARK_ROCM_CTA_THREADS)
		RocmFloatToBf16(output_bf16, base + element,
			staged_input[element] * inverse *
			RocmBf16ToFloat(gain_bf16, element));
}

/* ------------------------------------------------------------------ */
/* Shard pack/unpack: exact port of the CUDA interleave layout         */
/* (wq | wkv | compress_kv,score interleaved by tp_rank | index_*)      */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmProjectionPackKernel(
	const uint16_t *wq, const uint16_t *wkv,
	const uint16_t *compress_kv, const uint16_t *compress_score,
	const uint16_t *index_kv, const uint16_t *index_score,
	uint16_t *packed, uint32_t row_count,
	uint32_t tp_rank, uint32_t tp_degree,
	uint32_t compress_channels, uint32_t index_channels)
{
	uint32_t row = blockIdx.x;
	uint32_t column = blockIdx.y * blockDim.x + threadIdx.x;
	uint32_t wq_width = ROCM_MODEL_QUERY_LORA_RANK / tp_degree;
	uint32_t wkv_width = ROCM_MODEL_ATTN_HEAD_DIMENSION / tp_degree;
	uint32_t compress_width = compress_channels / tp_degree;
	uint32_t index_width = index_channels / tp_degree;
	uint32_t base, local;
	uint16_t value = 0u;
	if (row >= row_count || column >= ROCM_MODEL_HIDDEN_DIMENSION)
		return;
	base = tp_rank * wq_width;
	if (column >= base && column < base + wq_width)
		value = wq[(uint64_t)row * wq_width + column - base];
	base = ROCM_MODEL_QUERY_LORA_RANK + tp_rank * wkv_width;
	if (column >= base && column < base + wkv_width)
		value = wkv[(uint64_t)row * wkv_width + column - base];
	base = ROCM_MODEL_QUERY_LORA_RANK + ROCM_MODEL_ATTN_HEAD_DIMENSION;
	local = column >= base ? column - base : UINT32_MAX;
	if (local < compress_channels * 2u &&
		local % compress_channels >= tp_rank * compress_width &&
		local % compress_channels < (tp_rank + 1u) * compress_width)
		value = local < compress_channels ?
			compress_kv[(uint64_t)row * compress_width + local - tp_rank * compress_width] :
			compress_score[(uint64_t)row * compress_width + local - compress_channels - tp_rank * compress_width];
	base += compress_channels * 2u;
	local = column >= base ? column - base : UINT32_MAX;
	if (local < index_channels * 2u &&
		local % index_channels >= tp_rank * index_width &&
		local % index_channels < (tp_rank + 1u) * index_width)
		value = local < index_channels ?
			index_kv[(uint64_t)row * index_width + local - tp_rank * index_width] :
			index_score[(uint64_t)row * index_width + local - index_channels - tp_rank * index_width];
	packed[(uint64_t)row * ROCM_MODEL_HIDDEN_DIMENSION + column] = value;
}

/* ------------------------------------------------------------------ */
/* Island entry                                                        */
/* ------------------------------------------------------------------ */

static SparkHwStatus SparkDsv4RocmL1Validate(const SparkDsv4RocmL1Io *io)
{
	if (io == 0 || io->boundary_in_bf16 == 0 || io->boundary_out_bf16 == 0 ||
		io->post_f32 == 0 || io->comb_f32 == 0 || io->norm_gain_bf16 == 0 ||
		io->q_lora_weight_bf16 == 0 || io->kv_weight_bf16 == 0 ||
		io->packed_shards_bf16 == 0 || io->row_count == 0u ||
		io->row_count > 1024u /* MAX_INPUT_ROW_COUNT */)
		return(SPARK_HW_INVALID);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandL1(SparkHwQueue queue,
	const SparkDsv4RocmL1Io *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 post_grid, norm_grid, pack_grid;
	SparkHwStatus status;
	status = SparkDsv4RocmL1Validate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	/* Stage 1 — hcPost fold over all four streams. */
	post_grid.x = (ROCM_MODEL_HIDDEN_DIMENSION + SPARK_ROCM_CTA_THREADS - 1u) /
		SPARK_ROCM_CTA_THREADS;
	post_grid.y = io->row_count;
	post_grid.z = 1u;
	SparkDsv4RocmHcPostKernel<<<post_grid, SPARK_ROCM_CTA_THREADS, 0, stream>>>(
		io->boundary_in_bf16, io->boundary_in_bf16, io->post_f32, io->comb_f32,
		io->boundary_out_bf16, io->row_count, ROCM_MODEL_HIDDEN_DIMENSION,
		post_grid.x);
	/* Stage 2 — boundary norm of stream 0 into the normalized hidden view
	 * consumed by projections (flash geometry: norm over hidden slice). */
	norm_grid.x = io->row_count;
	norm_grid.y = 1u;
	norm_grid.z = 1u;
	SparkDsv4RocmRmsNormKernel<<<norm_grid, SPARK_ROCM_CTA_THREADS,
		ROCM_MODEL_HIDDEN_DIMENSION * sizeof(float), stream>>>(
		io->boundary_out_bf16, io->norm_gain_bf16, io->boundary_out_bf16,
		ROCM_MODEL_HIDDEN_DIMENSION, ROCM_MODEL_RMS_NORM_EPSILON);
	/* Stage 3 — projection GEMV rows: q lora rank + kv head dim outputs per
	 * input row, then shard-pack into the frozen interleave layout. */
	dim3 q_grid, kv_grid;
	/* Stage 3a — projections into their own sinks (no aliasing). */
	q_grid.x = io->row_count;
	q_grid.y = ROCM_MODEL_QUERY_LORA_RANK;
	q_grid.z = 1u;
	SparkDsv4RocmLinearRowsKernel<<<q_grid, SPARK_ROCM_CTA_THREADS, 0, stream>>>(
		io->q_lora_weight_bf16, io->boundary_out_bf16, io->q_proj_bf16,
		io->row_count, ROCM_MODEL_HIDDEN_DIMENSION, ROCM_MODEL_QUERY_LORA_RANK);
	kv_grid.x = io->row_count;
	kv_grid.y = ROCM_MODEL_ATTN_HEAD_DIMENSION;
	kv_grid.z = 1u;
	SparkDsv4RocmLinearRowsKernel<<<kv_grid, SPARK_ROCM_CTA_THREADS, 0, stream>>>(
		io->kv_weight_bf16, io->boundary_out_bf16, io->kv_proj_bf16,
		io->row_count, ROCM_MODEL_HIDDEN_DIMENSION, ROCM_MODEL_ATTN_HEAD_DIMENSION);
	/* Stage 3b — pack the frozen TP interleave. TP1 bring-up passes rank 0
	 * degree 1 with zero compress/index channels; wq and wkv land at their
	 * packed offsets from distinct sinks, so no input/output aliasing. */
	pack_grid.x = io->row_count;
	pack_grid.y = (ROCM_MODEL_HIDDEN_DIMENSION + SPARK_ROCM_CTA_THREADS - 1u) /
		SPARK_ROCM_CTA_THREADS;
	pack_grid.z = 1u;
	SparkDsv4RocmProjectionPackKernel<<<pack_grid, SPARK_ROCM_CTA_THREADS, 0, stream>>>(
		(const uint16_t *)io->q_proj_bf16,
		(const uint16_t *)io->kv_proj_bf16, (const uint16_t *)0,
		(const uint16_t *)0, (const uint16_t *)0, (const uint16_t *)0,
		(uint16_t *)io->packed_shards_bf16, io->row_count, 0u, 1u, 0u, 0u);
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}
