/*
 * L3 layer.cache_transition — ROCm island implementation (.hip)
 *
 * Frozen declared surface (hwiface_v1_freeze.md F1):
 *   inputs   : q/kv/score states, positions, page tables
 *   outputs  : updated pages, emitted entries, emit counters
 *   handles  : queue, memory, cache pages
 *   class    : split verdict — C2: page addresses, emit counters, ring
 *              indices (+ any payload field the recipe's emitted-field
 *              table marks unrounded-copy); C3: quantized payloads
 *
 * THE SHARPEST EDGE OF THE PORT (impl guide §4 L3): the ring-addressing
 * and emit-counter math below is a line-faithful port and must be
 * bit-exact against CUDA. Divergence here is an integration failure, not
 * a tolerance question.
 *
 * SCOPE (honest classification):
 *   LANDED HERE — compressor step (state staging, boundary detection,
 *   softmax pooling incl. the overlapped-window form, emitted[] flags),
 *   cache scatter (exact ring-slot arithmetic), accumulators.
 *   NOT HERE    — the fused KvEmission boundary-predicated pipeline
 *   (.cu :1055+: emit-row RMSNorm -> rope -> FP8 quantize -> scatter in
 *   one CTA). Its pieces exist across this archive but the fused form is
 *   S8 scope; requested_caps gates it exactly like F1/L2.
 *
 * Port provenance: exact bodies from spark_dsv4_resident_decode_stage_cuda.cu
 * (:899-1047). No wave-size-sensitive reduction exists on this path (all
 * pooling walks slots serially per channel), so C2 parity is tree-free.
 */
#include <string.h>

#include "spark_dsv4_rocm_kernels_common.hiph"

#define SPARK_DSV4_ROCM_L3_CAP_COMPRESS_SCATTER 0x1u
#define SPARK_DSV4_ROCM_L3_CAP_FUSED_KV_EMISSION 0x2u /* S8 */
#define SPARK_DSV4_ROCM_L3_CAP_KNOWN \
	(SPARK_DSV4_ROCM_L3_CAP_COMPRESS_SCATTER | \
	 SPARK_DSV4_ROCM_L3_CAP_FUSED_KV_EMISSION)

#define ROCM_NEGATIVE_INFINITY (-3.0e38f)

/* ------------------------------------------------------------------ */
/* Softmax pooling over pool slots (exact port of .cu :899-942).        */
/* ------------------------------------------------------------------ */

__device__ static inline float RocmPoolChannel(const float *kv,
	const float *score, uint32_t slots, uint32_t stride, uint32_t channel)
{
	float maximum = ROCM_NEGATIVE_INFINITY, total = 0.0f, value = 0.0f;
	for (uint32_t slot = 0u; slot < slots; slot++)
		if (score[slot * stride + channel] > maximum)
			maximum = score[slot * stride + channel];
	for (uint32_t slot = 0u; slot < slots; slot++)
	{
		float weight;
		if (score[slot * stride + channel] <= ROCM_NEGATIVE_INFINITY)
			continue;
		weight = __expf(score[slot * stride + channel] - maximum);
		total += weight;
		value += weight * kv[slot * stride + channel];
	}
	return(value / total);
}

/* Overlap gather pool: 2*ratio slots; i < ratio reads the previous group's
 * FIRST half, i >= ratio the current group's SECOND half (exact :921). */

__device__ static inline float RocmPoolOverlapChannel(const float *kv_state,
	const float *score_state, uint32_t ratio, uint32_t channels,
	uint32_t width, uint32_t channel)
{
	float maximum = ROCM_NEGATIVE_INFINITY, total = 0.0f, value = 0.0f;
	for (uint32_t slot = 0u; slot < 2u * ratio; slot++)
	{
		uint32_t source = slot < ratio ?
			slot * channels + channel :
			slot * channels + width + channel;
		if (score_state[source] > maximum)
			maximum = score_state[source];
	}
	for (uint32_t slot = 0u; slot < 2u * ratio; slot++)
	{
		uint32_t source = slot < ratio ?
			slot * channels + channel :
			slot * channels + width + channel;
		float score = score_state[source];
		float weight;
		if (score <= ROCM_NEGATIVE_INFINITY)
			continue;
		weight = __expf(score - maximum);
		total += weight;
		value += weight * kv_state[source];
	}
	return(value / total);
}

/* ------------------------------------------------------------------ */
/* Compressor step (exact port of .cu :949-1005): one CTA per live      */
/* lane, rows advanced in packet order; slot = pos % ratio; boundary    */
/* when (pos+1) % ratio == 0; overlapped form stages 2*ratio windows    */
/* and copies the second half forward after each emission.              */
/* C2 artifacts written here: emitted[row] flags + state layout.        */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmCompressStepKernel(
	const void *kv_bf16, const void *score_bf16, const float *ape_f32,
	float *kv_state_f32, float *score_state_f32, uint64_t state_lane_stride,
	const uint32_t *row_lane_indices, const uint64_t *row_positions,
	uint32_t row_count, uint32_t ratio, uint32_t overlapped, uint32_t width,
	void *emit_bf16, uint32_t *emitted)
{
	uint32_t coff = overlapped != 0u ? 2u : 1u;
	uint32_t channels = coff * width;
	uint32_t first_row = blockIdx.x;
	uint32_t lane, previous;
	if (first_row >= row_count)
		return;
	lane = row_lane_indices[first_row];
	for (previous = 0u; previous < first_row; previous++)
		if (row_lane_indices[previous] == lane)
			return; /* an earlier row already owns this lane */
	float *kv_state = kv_state_f32 + (uint64_t)lane * state_lane_stride;
	float *score_state = score_state_f32 + (uint64_t)lane * state_lane_stride;
	for (uint32_t row = first_row; row < row_count; row++)
	{
		uint32_t slot, boundary;
		if (row_lane_indices[row] != lane)
			continue;
		slot = (uint32_t)(row_positions[row] % ratio);
		boundary = (row_positions[row] + 1u) % ratio == 0u ? 1u : 0u;
		uint64_t ape_base = (row_positions[row] % ratio) * channels;
		for (uint32_t channel = threadIdx.x; channel < channels;
			channel += SPARK_ROCM_CTA_THREADS)
		{
			uint64_t source = (uint64_t)row * channels + channel;
			kv_state[((overlapped != 0u ? ratio : 0u) + slot) * channels +
				channel] = RocmBf16ToFloat(kv_bf16, source);
			score_state[((overlapped != 0u ? ratio : 0u) + slot) * channels +
				channel] = RocmBf16ToFloat(score_bf16, source) +
				ape_f32[ape_base + channel];
		}
		__syncthreads();
		if (threadIdx.x == 0u)
			emitted[row] = boundary;
		for (uint32_t channel = threadIdx.x; channel < width;
			channel += SPARK_ROCM_CTA_THREADS)
		{
			float pooled = 0.0f;
			if (boundary != 0u)
				pooled = overlapped != 0u ?
					RocmPoolOverlapChannel(kv_state, score_state, ratio,
						channels, width, channel) :
					RocmPoolChannel(kv_state, score_state, ratio,
						channels, channel);
			RocmFloatToBf16(emit_bf16, (uint64_t)row * width + channel,
				pooled);
		}
		__syncthreads();
		if (overlapped != 0u && boundary != 0u)
			for (uint32_t channel = threadIdx.x;
				channel < ratio * channels;
				channel += SPARK_ROCM_CTA_THREADS)
			{
				kv_state[channel] = kv_state[ratio * channels + channel];
				score_state[channel] = score_state[ratio * channels + channel];
			}
		__syncthreads();
	}
}

/* ------------------------------------------------------------------ */
/* Cache scatter (exact port of .cu :1007-1047). Ring arithmetic is the */
/* C2 core: slot = base_slot + (position % ring_slots) / ratio;         */
/* destination = lane * cache_lane_stride + slot * width.               */
/* ------------------------------------------------------------------ */

__device__ static inline void RocmCacheScatterRow(
	const uint16_t *source_bf16, uint16_t *cache_bf16,
	uint64_t cache_lane_stride, const uint32_t *row_lane_indices,
	const uint64_t *row_positions, uint32_t row, uint32_t width,
	uint64_t base_slot, uint32_t ratio, uint32_t ring_slots)
{
	uint64_t position = row_positions[row];
	uint64_t slot = ratio != 0u ?
		base_slot + ((position % ring_slots) / ratio) :
		position % ring_slots;
	uint64_t destination = (uint64_t)row_lane_indices[row] *
		cache_lane_stride + slot * width;
	uint64_t source = (uint64_t)row * width;
	for (uint32_t element = threadIdx.x; element < width;
		element += SPARK_ROCM_CTA_THREADS)
		cache_bf16[destination + element] = source_bf16[source + element];
}

__global__ void SparkDsv4RocmCacheScatterKernel(
	const uint16_t *source_bf16, const uint32_t *emitted,
	uint16_t *cache_bf16, uint64_t cache_lane_stride,
	const uint32_t *row_lane_indices, const uint64_t *row_positions,
	uint32_t row_count, uint32_t width, uint64_t base_slot, uint32_t ratio,
	uint32_t ring_slots)
{
	uint32_t row = blockIdx.x;
	if (row >= row_count || (ratio != 0u && emitted[row] == 0u))
		return;
	RocmCacheScatterRow(source_bf16, cache_bf16, cache_lane_stride,
		row_lane_indices, row_positions, row, width, base_slot, ratio,
		ring_slots);
}

/* ------------------------------------------------------------------ */
/* Island io + entry                                                    */
/* ------------------------------------------------------------------ */

typedef struct SparkDsv4RocmL3Io
{
	uint32_t row_count;
	uint32_t requested_caps;          /* SPARK_DSV4_ROCM_L3_CAP_* mask */
	uint32_t ratio;                   /* CSA compress ratio (flash: 4) */
	uint32_t overlapped;              /* 0 plain | 1 overlap-gather pool */
	uint32_t width;                   /* output channel width */
	const void *kv_bf16;              /* [rows][coff*width] */
	const void *score_bf16;           /* [rows][coff*width] */
	const float *ape_f32;             /* [ratio*channels] additive pos enc */
	float *kv_state_f32;              /* per-lane compressor state */
	float *score_state_f32;
	uint64_t state_lane_stride;       /* f32 elements per lane */
	const uint32_t *row_lane_indices_u32;
	const uint64_t *row_positions_u64;
	void *emit_bf16;                  /* [rows][width] pooled entries */
	uint32_t *emitted_u32;            /* [rows] C2 emit flags/counters */
	/* Cache scatter. */
	uint16_t *cache_bf16;             /* page backing store */
	uint64_t cache_lane_stride;       /* u16 elements per lane */
	uint64_t base_slot;
	uint32_t ring_slots;
} SparkDsv4RocmL3Io;

static SparkHwStatus SparkDsv4RocmL3Validate(const SparkDsv4RocmL3Io *io)
{
	if (io == 0 || io->row_count == 0u || io->row_count > 1024u ||
		io->ratio == 0u || io->width == 0u ||
		(io->requested_caps & ~SPARK_DSV4_ROCM_L3_CAP_KNOWN) != 0u)
		return(SPARK_HW_INVALID);
	if ((io->requested_caps & SPARK_DSV4_ROCM_L3_CAP_COMPRESS_SCATTER) != 0u &&
		(io->kv_bf16 == 0 || io->score_bf16 == 0 || io->ape_f32 == 0 ||
		 io->kv_state_f32 == 0 || io->score_state_f32 == 0 ||
		 io->state_lane_stride == 0u || io->row_lane_indices_u32 == 0 ||
		 io->row_positions_u64 == 0 || io->emit_bf16 == 0 ||
		 io->emitted_u32 == 0 || io->cache_bf16 == 0 ||
		 io->cache_lane_stride == 0u || io->ring_slots == 0u))
		return(SPARK_HW_INVALID);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandL3(SparkHwQueue queue,
	const SparkDsv4RocmL3Io *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 grid;
	SparkHwStatus status = SparkDsv4RocmL3Validate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	grid.x = io->row_count;
	grid.y = 1u;
	grid.z = 1u;
	SparkDsv4RocmCompressStepKernel<<<grid, SPARK_ROCM_CTA_THREADS, 0,
		stream>>>(io->kv_bf16, io->score_bf16, io->ape_f32,
		io->kv_state_f32, io->score_state_f32, io->state_lane_stride,
		io->row_lane_indices_u32, io->row_positions_u64, io->row_count,
		io->ratio, io->overlapped, io->width, io->emit_bf16,
		io->emitted_u32);
	SparkDsv4RocmCacheScatterKernel<<<grid, SPARK_ROCM_CTA_THREADS, 0,
		stream>>>((const uint16_t *)io->emit_bf16, io->emitted_u32,
		io->cache_bf16, io->cache_lane_stride, io->row_lane_indices_u32,
		io->row_positions_u64, io->row_count, io->width, io->base_slot,
		io->ratio, io->ring_slots);
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}
