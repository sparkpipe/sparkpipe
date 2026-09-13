/*
 * E0 prologue.embed — ROCm island implementation (.hip)
 *
 * Frozen declared surface (hwiface_v1_freeze.md F1):
 *   inputs   : token ids, embedding view
 *   outputs   : boundary packet (hc_mult x hidden bf16)
 *   handles  : queue, memory
 *   class    : C2 — pure integer gather/copy; bit-exact vs CUDA required.
 *
 * Port provenance: SparkLmEmbeddingGatherKernel body (cuh :806-818) is a
 * pure uint16-row copy — ported verbatim modulo accessor spelling. Stage-0
 * stream expansion follows the firmware contract ("stage 0 expands the
 * embedding into four identical streams", firmware.h header comment): the
 * gathered hidden row is broadcast across all HC_STREAM_COUNT streams of
 * the boundary packet. Weight read-ahead kick satisfies freeze F2
 * ("non-blocking enqueue that satisfies kick semantics"); the CUDA side
 * warms L2 through a dedicated kernel — here it is a strided touch with a
 * sink word write, classified honestly as simplified until S8 perf work.
 */
#include <string.h>

#include "spark_dsv4_rocm_kernels_common.hiph"

/* ------------------------------------------------------------------ */
/* Embedding gather: one block per row, uint4-style vector moves over   */
/* 16-bit elements, scalar tail (exact cuh :806 semantics).             */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmEmbeddingGatherKernel(
	const uint32_t *token_ids, const void *embedding_bf16,
	void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	uint32_t row = blockIdx.x;
	if (row >= row_count)
		return;
	uint64_t source_offset = (uint64_t)token_ids[row] * hidden_dimension;
	uint64_t destination_offset = (uint64_t)row * hidden_dimension;
	uint32_t vector_count = hidden_dimension >> 3u;
	const uint16_t *source = (const uint16_t *)embedding_bf16;
	uint16_t *destination = (uint16_t *)hidden_bf16;
	for (uint32_t vector = threadIdx.x; vector < vector_count;
		vector += SPARK_ROCM_CTA_THREADS)
	{
		uint64_t base = destination_offset + (vector << 3u);
		uint64_t src = source_offset + (vector << 3u);
		#pragma unroll
		for (uint32_t lane_copy = 0u; lane_copy < 8u; lane_copy++)
			destination[base + lane_copy] = source[src + lane_copy];
	}
	for (uint32_t element = (vector_count << 3u) + threadIdx.x;
		element < hidden_dimension; element += SPARK_ROCM_CTA_THREADS)
		destination[destination_offset + element] =
			source[source_offset + element];
}

/* ------------------------------------------------------------------ */
/* Stage-0 stream expand: broadcast the reduced hidden row into every   */
/* hyper-connection stream slot of the boundary packet (integer copy,   */
/* stays inside the C2 class).                                          */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmStreamExpandKernel(
	const void *reduced_bf16, void *boundary_packet_bf16,
	uint32_t row_count, uint32_t dimension, uint32_t hc_streams)
{
	uint32_t row = blockIdx.x;
	uint64_t span = (uint64_t)hc_streams * dimension;
	for (uint64_t offset = blockIdx.y * blockDim.x + threadIdx.x;
		offset < span;
		offset += gridDim.y * blockDim.x)
	{
		uint64_t element = offset % dimension;
		/* AUDIT R1 FIX: this write previously subscripted the void* output
		 * directly with a float right-hand side — a TU that could never have
		 * passed hipcc. The bf16->f32->bf16 roundtrip through the shared
		 * helpers is bit-exact (widening is exact; RNE of an exactly
		 * representable value returns the original bits, payload included),
		 * so the copy stays inside the C2 integer-pure class. */
		RocmFloatToBf16(boundary_packet_bf16, (uint64_t)row * span + offset,
			RocmBf16ToFloat(reduced_bf16,
				(uint64_t)row * dimension + element));
	}
}

/* ------------------------------------------------------------------ */
/* Read-ahead kick: strided 128B-granule touch over payload bytes with  */
/* progress words into the sink (SIMPLIFIED vs CUDA L2-warm kernel —    */
/* declared honest deviation; contents below the seam are target-free). */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmReadAheadKickKernel(
	const uint8_t *payload, uint64_t bytes, uint32_t *sink_u32,
	uint32_t block_capacity)
{
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x * 16u;
	volatile uint32_t accumulator = 0u;
	for (uint64_t offset = ((uint64_t)blockIdx.x * blockDim.x +
		threadIdx.x) * 16u; offset < bytes; offset += stride)
		accumulator += payload[offset];
	if (sink_u32 != 0 && threadIdx.x == 0u && blockIdx.x == 0u)
		*sink_u32 = block_capacity;
	(void)accumulator;
}

/* ------------------------------------------------------------------ */
/* Island io + entry                                                    */
/* ------------------------------------------------------------------ */

typedef struct SparkDsv4RocmE0Io
{
	uint32_t row_count;
	const uint32_t *token_ids_u32;    /* [rows] */
	const void *token_embedding_bf16; /* [vocab][hidden] */
	void *reduced_hidden_bf16;        /* [rows][hidden] gather sink */
	void *boundary_packet_bf16;       /* [rows][hc*hidden] island output */
	/* Read-ahead kick (optional; null skips). */
	const uint8_t *read_ahead_payload;
	uint64_t read_ahead_bytes;
	uint32_t *read_ahead_sink_u32;
	uint32_t read_ahead_block_capacity;
} SparkDsv4RocmE0Io;

static SparkHwStatus SparkDsv4RocmE0Validate(const SparkDsv4RocmE0Io *io)
{
	if (io == 0 || io->row_count == 0u || io->row_count > 1024u ||
		io->token_ids_u32 == 0 || io->token_embedding_bf16 == 0 ||
		io->reduced_hidden_bf16 == 0 || io->boundary_packet_bf16 == 0)
		return(SPARK_HW_INVALID);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandE0(SparkHwQueue queue,
	const SparkDsv4RocmE0Io *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 gather_grid, expand_grid;
	SparkHwStatus status = SparkDsv4RocmE0Validate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	gather_grid.x = io->row_count;
	gather_grid.y = 1u;
	gather_grid.z = 1u;
	SparkDsv4RocmEmbeddingGatherKernel<<<gather_grid, SPARK_ROCM_CTA_THREADS,
		0, stream>>>(io->token_ids_u32, io->token_embedding_bf16,
		io->reduced_hidden_bf16, io->row_count, ROCM_MODEL_HIDDEN_DIMENSION);
	expand_grid.x = io->row_count;
	expand_grid.y = (ROCM_MODEL_BOUNDARY_ELEMENTS +
		SPARK_ROCM_CTA_THREADS - 1u) / SPARK_ROCM_CTA_THREADS;
	expand_grid.z = 1u;
	SparkDsv4RocmStreamExpandKernel<<<expand_grid, SPARK_ROCM_CTA_THREADS,
		0, stream>>>(io->reduced_hidden_bf16, io->boundary_packet_bf16,
		io->row_count, ROCM_MODEL_HIDDEN_DIMENSION,
		ROCM_MODEL_HC_STREAM_COUNT);
	if (io->read_ahead_payload != 0 && io->read_ahead_bytes != 0u)
	{
		dim3 kick_grid;
		kick_grid.x = 64u;
		kick_grid.y = 1u;
		kick_grid.z = 1u;
		SparkDsv4RocmReadAheadKickKernel<<<kick_grid, SPARK_ROCM_CTA_THREADS,
			0, stream>>>(io->read_ahead_payload, io->read_ahead_bytes,
			io->read_ahead_sink_u32, io->read_ahead_block_capacity);
	}
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}
