/*
 * L4 layer.moe_routed — ROCm island implementation (.hip)
 *
 * Frozen declared surface (hwiface_v1_freeze.md F1):
 *   inputs   : normalized hidden, sealed-route batch (opaque), routed weights
 *   outputs  : FFN accumulator, expert offsets
 *   handles  : queue, event, memory
 *   class    : split verdict — C2: route indices/group offsets/expert
 *              assignment (core-decided, backend-realized); C3: grouped-GEMM
 *              output
 *
 * SCOPE (honest classification):
 *   LANDED HERE — grouped MXFP4 gate/up activation over the sealed batch
 *   (per-packed-row via route_source_token/group_row_offset), grouped W2
 *   down projection over the stacked [experts x hidden] view, MoE pair
 *   reduce (weighted scatter-accumulate via inverse map) in plain and
 *   strided forms, accum-add.
 *   NOT HERE    — ROUTE REALIZATION ITSELF (the GateRoute pipeline:
 *   gate scores -> top-k -> tid2eid remap -> grouped-layout build producing
 *   group_row_offset/route_packed_row/route_source_token/group_tile_prefix).
 *   That pipeline is THE C2 half; porting it requires a dedicated
 *   line-exact pass over the CUDA kernel bodies, and a probably-divergent
 *   guess would fail integration by construction. It stays behind
 *   requested_caps (SPARK_HW_UNSUPPORTED) until that pass lands. The GEMM
 *   side consumes whatever sealed batch the portable core presents using
 *   the frozen slot-workspace field names.
 *
 * Port provenance: stacked-view geometry measured at .cu :3156/:3243-3252
 * (W13 views [experts x expert_width][hidden]; W2 view
 * [experts x hidden][expert_width]); decode tables duplicated locally to
 * keep TUs self-contained (landmine rule still active).
 */
#include <string.h>

#include "spark_dsv4_rocm_kernels_common.hiph"

#define SPARK_DSV4_ROCM_L4_CAP_GROUPED_GEMM 0x1u
#define SPARK_DSV4_ROCM_L4_CAP_PAIR_REDUCE 0x2u
#define SPARK_DSV4_ROCM_L4_CAP_ROUTE_REALIZATION 0x4u /* reserved, S8 pass */
#define SPARK_DSV4_ROCM_L4_CAP_KNOWN \
	(SPARK_DSV4_ROCM_L4_CAP_GROUPED_GEMM | \
	 SPARK_DSV4_ROCM_L4_CAP_PAIR_REDUCE | \
	 SPARK_DSV4_ROCM_L4_CAP_ROUTE_REALIZATION)

#define ROCM_MXFP4_GROUP 32u

/* Local codec duplicates (see F1 TU for the annotated originals). */

__device__ static inline float RocmL4DecodeE2m1(uint32_t code)
{
	static const float magnitudes[8] =
		{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
	float magnitude = magnitudes[code & 7u];
	return((code & 8u) != 0u ? -magnitude : magnitude);
}

__device__ static inline float RocmL4DecodeE8m0(uint32_t exponent)
{
	return(__uint_as_float(exponent << 23u));
}

/* MXFP4 dot: one neuron row of a stacked expert weight matrix against the
 * staged input; block-32 e8m0 scales, nibble payload. */

__device__ static inline float RocmL4DotRowMxfp4(
	const uint8_t *payload, const uint8_t *scale_e8m0,
	const float *shared_input, uint64_t neuron_base,
	uint32_t input_dimension, uint32_t lane)
{
	float accumulator = 0.0f;
	uint32_t group_count = input_dimension / ROCM_MXFP4_GROUP;
	for (uint32_t group = lane; group < group_count;
		group += SPARK_ROCM_WAVE_LANES)
	{
		float scale_value = RocmL4DecodeE8m0(
			scale_e8m0[(neuron_base / ROCM_MXFP4_GROUP) + group]);
		uint32_t base_element = group * ROCM_MXFP4_GROUP;
		float partial = 0.0f;
		for (uint32_t element = 0u; element < ROCM_MXFP4_GROUP;
			element += 8u)
		{
			uint32_t packed = ((const uint32_t *)payload)[
				(neuron_base + base_element + element) >> 3u];
			for (uint32_t nibble = 0u; nibble < 8u; nibble++)
				partial += RocmL4DecodeE2m1(
					(packed >> (nibble << 2u)) & 0x0fu) * scale_value *
					shared_input[base_element + element + nibble];
		}
		accumulator += partial;
	}
	return(accumulator);
}

/* ------------------------------------------------------------------ */
/* Grouped gate/up activation: one wave per (packed_row, neuron stripe). */
/* Packed row r computes against expert = route_expert[r] and gates the  */
/* input row route_source_token[r].                                      */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmFusedExpertW13ActKernel(
	const uint8_t *w1_payload, const uint8_t *w1_scale_e8m0,
	const uint8_t *w3_payload, const uint8_t *w3_scale_e8m0,
	const void *input_bf16, const uint32_t *route_source_token_u32,
	const uint32_t *route_expert_u32, void *activated_bf16,
	uint32_t packed_row_count, uint32_t hidden_dimension,
	uint32_t expert_width, uint32_t routed_expert_count, float limit)
{
	extern __shared__ float shared_input[];
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	uint32_t wave = threadIdx.x / SPARK_ROCM_WAVE_LANES;
	uint32_t packed_row = blockIdx.x;
	uint32_t neuron_block = blockIdx.y;
	if (packed_row >= packed_row_count)
		return;
	uint32_t token = route_source_token_u32[packed_row];
	uint32_t expert = route_expert_u32[packed_row];
	if (expert >= routed_expert_count || token == UINT32_MAX)
		return;
	for (uint32_t element = threadIdx.x; element < hidden_dimension;
		element += SPARK_ROCM_CTA_THREADS)
		shared_input[element] = RocmBf16ToFloat(input_bf16,
			(uint64_t)token * hidden_dimension + element);
	__syncthreads();
	uint32_t neuron = neuron_block * SPARK_ROCM_CTA_WARPS + wave;
	if (neuron >= expert_width)
		return;
	uint64_t neuron_base = ((uint64_t)expert * expert_width + neuron) *
		hidden_dimension;
	float gate = RocmL4DotRowMxfp4(w1_payload, w1_scale_e8m0, shared_input,
		neuron_base, hidden_dimension, lane);
	float up = RocmL4DotRowMxfp4(w3_payload, w3_scale_e8m0, shared_input,
		neuron_base, hidden_dimension, lane);
	gate = RocmWaveReduceSum(gate);
	up = RocmWaveReduceSum(up);
	if (lane == 0u)
	{
		gate = gate > limit ? limit : gate;
		up = up > limit ? limit : (up < -limit ? -limit : up);
		float swish = gate / (1.0f + expf(-gate));
		RocmFloatToBf16(activated_bf16,
			(uint64_t)packed_row * expert_width + neuron, swish * up);
	}
}

/* ------------------------------------------------------------------ */
/* Grouped W2 down: stacked [experts x hidden][expert_width]; output     */
/* written per packed row (pair reduce combines into the accumulator).   */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmExpertDownKernel(
	const uint8_t *w2_payload, const uint8_t *w2_scale_e8m0,
	const void *activated_bf16, const uint32_t *route_source_token_u32,
	const uint32_t *route_expert_u32, void *down_output_bf16,
	uint32_t packed_row_count, uint32_t expert_width,
	uint32_t hidden_dimension, uint32_t routed_expert_count)
{
	extern __shared__ float staged_activation[];
	uint32_t row = blockIdx.x;
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	if (row >= packed_row_count)
		return;
	uint32_t expert = route_expert_u32[row];
	if (expert >= routed_expert_count)
		return;
	for (uint32_t element = threadIdx.x; element < expert_width;
		element += SPARK_ROCM_CTA_THREADS)
		staged_activation[element] = RocmBf16ToFloat(activated_bf16,
			(uint64_t)row * expert_width + element);
	__syncthreads();
	for (uint32_t neuron = blockIdx.y * SPARK_ROCM_CTA_WARPS +
		threadIdx.x / SPARK_ROCM_WAVE_LANES; neuron < hidden_dimension;
		neuron += gridDim.y * SPARK_ROCM_CTA_WARPS)
	{
		uint64_t neuron_base = ((uint64_t)expert * hidden_dimension +
			neuron) * expert_width;
		float total = 0.0f;
		for (uint32_t start = 0u; start < expert_width;
			start += ROCM_MXFP4_GROUP)
		{
			float scale_value = RocmL4DecodeE8m0(w2_scale_e8m0[
				(neuron_base / ROCM_MXFP4_GROUP) + start / ROCM_MXFP4_GROUP]);
			float partial = 0.0f;
			for (uint32_t element = start + lane;
				element < start + ROCM_MXFP4_GROUP &&
				element < expert_width;
				element += SPARK_ROCM_WAVE_LANES)
			{
				/* Payload is nibble-packed: eight 4-bit codes per u32. */
				uint32_t packed = ((const uint32_t *)w2_payload)[
					(neuron_base + element) >> 3u];
				partial += RocmL4DecodeE2m1(
					(packed >> ((element & 7u) << 2u)) & 0x0fu) *
					scale_value * staged_activation[element];
			}
			total += partial;
		}
		total = RocmWaveReduceSum(total);
		if (lane == 0u)
			RocmFloatToBf16(down_output_bf16,
				(uint64_t)row * hidden_dimension + neuron, total);
	}
}

/* ------------------------------------------------------------------ */
/* Pair reduce (weighted scatter-accumulate via inverse map):            */
/* accumulator[token] += weight[pair] * slot_out[pair_row]. C3 path.     */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmMoePairReduceKernel(
	const void *slot_out_bf16, const uint32_t *inverse_map_u32,
	const float *pair_weights_f32, void *accum_bf16,
	uint64_t accum_row_stride, uint32_t accum_offset,
	uint32_t pair_row_count, uint32_t hidden_dimension)
{
	uint32_t pair_row = blockIdx.x;
	uint64_t destination;
	float weight;
	if (pair_row >= pair_row_count)
		return;
	uint32_t token = inverse_map_u32[pair_row];
	if (token == UINT32_MAX)
		return;
	weight = pair_weights_f32[pair_row];
	destination = (uint64_t)token * accum_row_stride + accum_offset;
	for (uint32_t element = threadIdx.x; element < hidden_dimension;
		element += SPARK_ROCM_CTA_THREADS)
	{
		float existing = RocmBf16ToFloat(accum_bf16, destination + element);
		float contribution = RocmBf16ToFloat(slot_out_bf16,
			(uint64_t)pair_row * hidden_dimension + element);
		RocmFloatToBf16(accum_bf16, destination + element,
			existing + weight * contribution);
	}
}

/* ------------------------------------------------------------------ */
/* Island io + entry                                                    */
/* ------------------------------------------------------------------ */

typedef struct SparkDsv4RocmL4Io
{
	uint32_t packed_row_count;        /* sealed-batch grouped row count */
	uint32_t routed_expert_count;
	uint32_t expert_width;
	float limit;                      /* shared activation clamp limit */
	uint32_t requested_caps;          /* SPARK_DSV4_ROCM_L4_CAP_* mask */
	/* Sealed-route batch views (slot workspace; opaque above the seam).
	 * route_realization builds these on CUDA; until that slice lands the
	 * caller may inject a core-side realization for bring-up. */
	const uint32_t *route_source_token_u32; /* [packed_rows] */
	const uint32_t *route_expert_u32;       /* [packed_rows] expert ids */
	/* Stacked MXFP4 weights. */
	const uint8_t *w1_payload;
	const uint8_t *w1_scale_e8m0;
	const uint8_t *w3_payload;
	const uint8_t *w3_scale_e8m0;
	const uint8_t *w2_payload;
	const uint8_t *w2_scale_e8m0;
	const void *normalized_hidden_bf16; /* [rows][hidden] */
	void *activated_bf16;               /* scratch [packed_rows][width] */
	void *down_output_bf16;             /* scratch [packed_rows][hidden] */
	/* Pair reduce. */
	void *ffn_accumulator_bf16;         /* [rows][hidden] */
	const uint32_t *inverse_map_u32;    /* [packed_rows] -> token */
	const float *pair_weights_f32;      /* [packed_rows] */
	uint64_t accum_row_stride;
	uint32_t accum_offset;
} SparkDsv4RocmL4Io;

static SparkHwStatus SparkDsv4RocmL4Validate(const SparkDsv4RocmL4Io *io)
{
	if (io == 0 || io->routed_expert_count == 0u ||
		io->expert_width == 0u ||
		io->expert_width % ROCM_MXFP4_GROUP != 0u ||
		(io->requested_caps & ~SPARK_DSV4_ROCM_L4_CAP_KNOWN) != 0u)
		return(SPARK_HW_INVALID);
	if ((io->requested_caps & SPARK_DSV4_ROCM_L4_CAP_ROUTE_REALIZATION) != 0u)
		return(SPARK_HW_UNSUPPORTED); /* reserved slice, S8 pass */
	if ((io->requested_caps & SPARK_DSV4_ROCM_L4_CAP_GROUPED_GEMM) != 0u)
	{
		if (io->w1_payload == 0 || io->w1_scale_e8m0 == 0 ||
			io->w3_payload == 0 || io->w3_scale_e8m0 == 0 ||
			io->w2_payload == 0 || io->w2_scale_e8m0 == 0 ||
			io->normalized_hidden_bf16 == 0 || io->activated_bf16 == 0 ||
			io->down_output_bf16 == 0 ||
			(ROCM_MODEL_HIDDEN_DIMENSION % ROCM_MXFP4_GROUP) != 0u)
			return(SPARK_HW_INVALID);
		if (io->packed_row_count != 0u && (io->route_source_token_u32 == 0 ||
			io->route_expert_u32 == 0))
			return(SPARK_HW_INVALID);
	}
	if ((io->requested_caps & SPARK_DSV4_ROCM_L4_CAP_PAIR_REDUCE) != 0u &&
		(io->ffn_accumulator_bf16 == 0 || io->inverse_map_u32 == 0 ||
		 io->pair_weights_f32 == 0))
		return(SPARK_HW_INVALID);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandL4(SparkHwQueue queue,
	const SparkDsv4RocmL4Io *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 act_grid, down_grid, reduce_grid;
	SparkHwStatus status = SparkDsv4RocmL4Validate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	if ((io->requested_caps & SPARK_DSV4_ROCM_L4_CAP_GROUPED_GEMM) != 0u &&
		io->packed_row_count != 0u)
	{
		act_grid.x = io->packed_row_count;
		act_grid.y = (io->expert_width + SPARK_ROCM_CTA_WARPS - 1u) /
			SPARK_ROCM_CTA_WARPS;
		act_grid.z = 1u;
		SparkDsv4RocmFusedExpertW13ActKernel<<<act_grid,
			SPARK_ROCM_CTA_THREADS,
			ROCM_MODEL_HIDDEN_DIMENSION * sizeof(float), stream>>>(
			io->w1_payload, io->w1_scale_e8m0, io->w3_payload,
			io->w3_scale_e8m0, io->normalized_hidden_bf16,
			io->route_source_token_u32, io->route_expert_u32,
			io->activated_bf16, io->packed_row_count,
			ROCM_MODEL_HIDDEN_DIMENSION, io->expert_width,
			io->routed_expert_count, io->limit);
		down_grid.x = io->packed_row_count;
		down_grid.y = (ROCM_MODEL_HIDDEN_DIMENSION +
			SPARK_ROCM_CTA_THREADS - 1u) / SPARK_ROCM_CTA_THREADS;
		down_grid.z = 1u;
		SparkDsv4RocmExpertDownKernel<<<down_grid, SPARK_ROCM_CTA_THREADS,
			io->expert_width * sizeof(float), stream>>>(io->w2_payload,
			io->w2_scale_e8m0, io->activated_bf16,
			io->route_source_token_u32, io->route_expert_u32,
			io->down_output_bf16, io->packed_row_count, io->expert_width,
			ROCM_MODEL_HIDDEN_DIMENSION, io->routed_expert_count);
	}
	if ((io->requested_caps & SPARK_DSV4_ROCM_L4_CAP_PAIR_REDUCE) != 0u &&
		io->packed_row_count != 0u)
	{
		reduce_grid.x = io->packed_row_count;
		reduce_grid.y = 1u;
		reduce_grid.z = 1u;
		SparkDsv4RocmMoePairReduceKernel<<<reduce_grid,
			SPARK_ROCM_CTA_THREADS, 0, stream>>>(io->down_output_bf16,
			io->inverse_map_u32, io->pair_weights_f32,
			io->ffn_accumulator_bf16, io->accum_row_stride != 0u ?
			io->accum_row_stride : (uint64_t)ROCM_MODEL_HIDDEN_DIMENSION,
			io->accum_offset, io->packed_row_count,
			ROCM_MODEL_HIDDEN_DIMENSION);
	}
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}
