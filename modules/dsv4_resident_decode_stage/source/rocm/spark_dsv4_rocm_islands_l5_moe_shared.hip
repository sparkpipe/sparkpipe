/*
 * L5 layer.moe_shared — ROCm island implementation (.hip)
 *
 * Frozen declared surface (hwiface_v1_freeze.md F1):
 *   inputs   : normalized hidden
 *   outputs  : FFN accumulator (partial)
 *   handles  : aux queue, fork/join event pair, memory (core-owned; this TU
 *              computes on the queue it is handed — fork/join stays core-side)
 *   class    : C3
 *
 * Port provenance: SparkLmSm121FusedDenseW13GemvKernel semantics
 * (cuh :3106-3150): staged input row -> activation-codec qdq -> fp8-e4m3
 * dot rows for gate(w1)/up(w3) with block-128 e8m0 scales -> warp reduce ->
 * bf16 round each -> gate clamp ABOVE by limit, up clamp BOTH sides by
 * limit -> Swish(gate)*up. W2 down projection accumulates into the
 * caller's partial FFN accumulator (flash shared family = fp8 weights).
 *
 * HONEST DEVIATIONS:
 * 1. The activation-input qdq helper (LmActivationFp8QdqFloatRow over the
 *    UE8M0 codec) is NOT reimplemented bit-exactly here; the staged row is
 *    consumed as f32. Flagged for golden-frame verification at S7 — if C3
 *    tolerances on L5 fail high, this is the first suspect.
 * 2. FP8-E4M3 weight decode uses direct exponent/mantissa reconstruction
 *    (E4M3 has no inf; NaN only on 0x7F/0xFF) instead of a table.
 * 3. GEMV shape is the bring-up form (one wave per neuron pair); LDS-budget
 *    re-tiling is S8 perf work.
 */
#include <string.h>

#include "spark_dsv4_rocm_kernels_common.hiph"

#define ROCM_FP8_BLOCK 128u

/* Defined below; matches SparkLmSm121Bf16Round (round via bf16 store). */
__device__ static inline float RocmFloatToBf16Bits(float value);

/* ------------------------------------------------------------------ */
/* E4M3 (4-bit exp, 3-bit mantissa, bias 7) decode.                    */
/* ------------------------------------------------------------------ */

__device__ static inline float RocmDecodeFp8E4m3(uint8_t bits)
{
	uint32_t sign = (bits & 0x80u) != 0u ? UINT32_C(0x80000000) : 0u;
	uint32_t exponent = (bits >> 3u) & 0x0fu;
	uint32_t mantissa = bits & 0x07u;
	uint32_t fbits;
	if (exponent == 0u)
		fbits = sign | (mantissa << 20u); /* subnormal: implicit -6 */
	else if (exponent == 0x0fu && mantissa == 0x07u)
		return(NAN); /* only E4M3 NaN encoding */
	else
		fbits = sign | ((exponent + 120u) << 23u) | (mantissa << 20u);
	return(__uint_as_float(fbits));
}

/* AUDIT R1 FIX: RocmDotRowFp8Block128 called RocmDecodeE8m0 without any
 * declaration or definition anywhere in this TU — hipcc would reject the
 * unit. Ported verbatim from SparkLmDecodeE8m0
 * (spark_lm_kernels.cuh :159): pure power of two, bias 127, 0xff reserved
 * as NaN decodes to zero, code 0 decodes to 2^-127. */
__device__ static inline float RocmDecodeE8m0(uint32_t byte_value)
{
	if (byte_value == 0xffu)
		return(0.0f);
	if (byte_value == 0u)
		return(__uint_as_float(UINT32_C(0x00400000)));
	return(__uint_as_float(byte_value << 23u));
}

__device__ static inline float RocmDotRowFp8Block128(
	const uint8_t *payload, const uint8_t *scale_e8m0,
	const float *shared_input, uint64_t neuron_base,
	uint32_t input_dimension, uint32_t lane)
{
	float accumulator = 0.0f;
	for (uint32_t start = 0u; start < input_dimension;
		start += ROCM_FP8_BLOCK)
	{
		float scale = RocmDecodeE8m0(scale_e8m0[neuron_base / ROCM_FP8_BLOCK +
			start / ROCM_FP8_BLOCK]);
		for (uint32_t element = start + lane;
			element < start + ROCM_FP8_BLOCK && element < input_dimension;
			element += SPARK_ROCM_WAVE_LANES)
			accumulator += shared_input[element] *
				RocmDecodeFp8E4m3(payload[neuron_base + element]) * scale;
	}
	return(accumulator);
}

/* ------------------------------------------------------------------ */
/* Fused gate/up activation kernel (wave per neuron; fixed tree).       */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmSharedW13ActKernel(
	const uint8_t *w1_payload, const uint8_t *w1_scale_e8m0,
	const uint8_t *w3_payload, const uint8_t *w3_scale_e8m0,
	const void *input_bf16, void *activated_bf16,
	uint32_t input_dimension, uint32_t output_dimension, float limit)
{
	extern __shared__ float shared_input[];
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	uint32_t wave = threadIdx.x / SPARK_ROCM_WAVE_LANES;
	uint32_t neuron = blockIdx.x * SPARK_ROCM_CTA_WARPS + wave;
	if (threadIdx.x < input_dimension)
	{
		for (uint32_t element = threadIdx.x; element < input_dimension;
			element += SPARK_ROCM_CTA_THREADS)
			shared_input[element] = RocmBf16ToFloat(input_bf16, element);
	}
	__syncthreads();
	if (neuron >= output_dimension)
		return;
	float gate = RocmDotRowFp8Block128(w1_payload, w1_scale_e8m0,
		shared_input, (uint64_t)neuron * input_dimension,
		input_dimension, lane);
	float up = RocmDotRowFp8Block128(w3_payload, w3_scale_e8m0,
		shared_input, (uint64_t)neuron * input_dimension,
		input_dimension, lane);
	gate = RocmWaveReduceSum(gate);
	up = RocmWaveReduceSum(up);
	if (lane == 0u)
	{
		gate = RocmFloatToBf16Bits(gate);
		up = RocmFloatToBf16Bits(up);
		gate = gate > limit ? limit : gate;
		up = up > limit ? limit : (up < -limit ? -limit : up);
		float swish = gate / (1.0f + expf(-gate));
		RocmFloatToBf16(activated_bf16, neuron, swish * up);
	}
}

/* bf16 rounding step matching SparkLmSm121Bf16Round (round via bf16 store). */

__device__ static inline float RocmFloatToBf16Bits(float value)
{
	uint32_t bits = __float_as_uint(value);
	uint32_t rounding = ((bits >> 16u) & 1u) + UINT32_C(0x7FFF);
	bits = (bits + rounding) >> 16u;
	return(__uint_as_float(bits << 16u));
}

/* ------------------------------------------------------------------ */
/* W2 down projection: fp8 weights [hidden][expert_width]; ACCUMULATES  */
/* into the caller's partial FFN accumulator (L5 output contract).      */
/* ------------------------------------------------------------------ */

__global__ void SparkDsv4RocmSharedW2AccumKernel(
	const uint8_t *w2_payload, const uint8_t *w2_scale_e8m0,
	const void *activated_bf16, void *accumulator_bf16,
	uint32_t row_count, uint32_t expert_width, uint32_t hidden_dimension)
{
	extern __shared__ float staged_activation[];
	uint32_t row = blockIdx.x;
	uint32_t lane = threadIdx.x % SPARK_ROCM_WAVE_LANES;
	if (row >= row_count)
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
		float total = RocmDotRowFp8Block128(w2_payload, w2_scale_e8m0,
			staged_activation, (uint64_t)neuron * expert_width,
			expert_width, lane);
		total = RocmWaveReduceSum(total);
		if (lane == 0u)
		{
			float existing = RocmBf16ToFloat(accumulator_bf16,
				(uint64_t)row * hidden_dimension + neuron);
			RocmFloatToBf16(accumulator_bf16,
				(uint64_t)row * hidden_dimension + neuron,
				existing + total);
		}
	}
}

/* ------------------------------------------------------------------ */
/* Island io + entry                                                    */
/* ------------------------------------------------------------------ */

typedef struct SparkDsv4RocmL5Io
{
	uint32_t row_count;
	const void *normalized_hidden_bf16; /* [rows][hidden] */
	/* w1/w3/w2 views: payload + block-128 e8m0 scales (fp8-e4m3). */
	const uint8_t *w1_payload;
	const uint8_t *w1_scale_e8m0;
	const uint8_t *w3_payload;
	const uint8_t *w3_scale_e8m0;
	const uint8_t *w2_payload;
	const uint8_t *w2_scale_e8m0;
	uint32_t expert_width;              /* intermediate dimension */
	float limit;                        /* activation clamp limit */
	void *activated_bf16;               /* scratch [rows][expert_width] */
	void *ffn_accumulator_bf16;         /* [rows][hidden] PARTIAL accumulate */
} SparkDsv4RocmL5Io;

static SparkHwStatus SparkDsv4RocmL5Validate(const SparkDsv4RocmL5Io *io)
{
	if (io == 0 || io->row_count == 0u || io->row_count > 1024u ||
		io->normalized_hidden_bf16 == 0 || io->w1_payload == 0 ||
		io->w1_scale_e8m0 == 0 || io->w3_payload == 0 ||
		io->w3_scale_e8m0 == 0 || io->w2_payload == 0 ||
		io->w2_scale_e8m0 == 0 || io->expert_width == 0u ||
		io->expert_width % ROCM_FP8_BLOCK != 0u ||
		io->activated_bf16 == 0 || io->ffn_accumulator_bf16 == 0 ||
		(ROCM_MODEL_HIDDEN_DIMENSION % ROCM_FP8_BLOCK) != 0u)
		return(SPARK_HW_INVALID);
	return(SPARK_HW_OK);
}

SparkHwStatus SparkDsv4RocmIslandL5(SparkHwQueue queue,
	const SparkDsv4RocmL5Io *io)
{
	hipStream_t stream;
	hipError_t error;
	dim3 act_grid, down_grid;
	SparkHwStatus status = SparkDsv4RocmL5Validate(io);
	if (status != SPARK_HW_OK)
		return(status);
	stream = (hipStream_t)queue;
	act_grid.x = (io->expert_width + SPARK_ROCM_CTA_WARPS - 1u) /
		SPARK_ROCM_CTA_WARPS;
	act_grid.y = 1u;
	act_grid.z = 1u;
	SparkDsv4RocmSharedW13ActKernel<<<act_grid, SPARK_ROCM_CTA_THREADS,
		ROCM_MODEL_HIDDEN_DIMENSION * sizeof(float), stream>>>(
		io->w1_payload, io->w1_scale_e8m0, io->w3_payload,
		io->w3_scale_e8m0, io->normalized_hidden_bf16,
		io->activated_bf16, ROCM_MODEL_HIDDEN_DIMENSION,
		io->expert_width, io->limit);
	down_grid.x = io->row_count;
	down_grid.y = (ROCM_MODEL_HIDDEN_DIMENSION + SPARK_ROCM_CTA_THREADS - 1u) /
		SPARK_ROCM_CTA_THREADS;
	down_grid.z = 1u;
	SparkDsv4RocmSharedW2AccumKernel<<<down_grid, SPARK_ROCM_CTA_THREADS,
		io->expert_width * sizeof(float), stream>>>(io->w2_payload,
		io->w2_scale_e8m0, io->activated_bf16, io->ffn_accumulator_bf16,
		io->row_count, io->expert_width, ROCM_MODEL_HIDDEN_DIMENSION);
	error = hipGetLastError();
	return(SparkRocmStatusFromHip(error));
}
