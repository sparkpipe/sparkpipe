#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "spark_glm5_next_resident_decode_stage_internal.h"

/*
 * glm5_next (GLM 5.3 Flash) resident decode stage, hardware validation
 * (sm_121a).
 *
 * A retained-receipt numerical gate driven entirely through the module's
 * own exported launchers (SparkGlm5NextLaunchCudaWaveBegin /
 * LayerAttention / LayerMlp) exactly as the wave runner drives them,
 * compared against an fp32 CPU oracle that restates the shared-kernel
 * formulas here, self-contained under nvcc and sharing no code with what
 * it validates.
 *
 * Tier 1 - the KDA layer with its dense MLP (layer 0): a four-token
 * causal walk through the mHC attention site (mix -> sinkhorn -> collapse
 * -> KDA sublayer -> post), the mHC FFN site and the dense MLP, plus a
 * bit-exact determinism re-walk. Covers: the fused q|k|v|beta GEMM +
 * split, the three BF16 short convolutions with SWISH, the qk L2 norm,
 * the LOW-RANK decay (fused decay|gate-down + two up projections) through
 * the bounded-decay formula, the delta-rule recurrence with fp32 state,
 * RMSNorm + sigmoid output gate, o_proj, and both mHC sites.
 *
 * Tier 2a - the DSA layer with routed experts (layer 3): the same walk
 * through the rope-0 MLA site (nope-only absorbed scoring, pure-512
 * latent cache, NO rope anywhere) and the MoE: fp32 router GEMM, sigmoid
 * top-8 with the correction bias selecting but not weighing, the
 * renormalised mixture at scale 2.5, eight routed-expert forwards through
 * the compiled package codec (dequant restated losslessly - the fixture
 * writes only exactly representable codes), and the shared expert.
 *
 * Tier 2b - the kpool indexer at context > INDEX_TOP_K: the fixture
 * pre-populates the packed indexer cache ([k|gate|1.0] rows) so the
 * per-channel softmax pool keys are host-known; the pool selection
 * (512 of context/4 complete pools) is checked as a set against the
 * oracle's prediction and the expanded 2051-wide list element-wise,
 * sentinel semantics included.
 *
 * DEVIATION, recorded: the reference clamps the dense/shared SwiGLU at
 * swiglu_limit 10.0; the shared LmSiluMulKernel (this family and dsv4's)
 * does not clamp. Fixtures keep |gate|,|up| << 10 so the tiers are exact
 * either way; the deviation is a property of the shipped kernel, shared
 * with the dsv4 family, and only binds for activations beyond 10.
 *
 * Every comparison prints its numbers; the thresholds are the guard, the
 * numbers are the evidence. The pure oracle/codec/selection math below is
 * EXECUTED on every host by tests/test_glm5_next_cuda_validator_tier2_
 * oracle.py through the SPARK_GLM5_NEXT_VALIDATOR_ORACLE_SELFTEST entry
 * at the bottom of this file, so the fixture shaping and oracle formulas
 * ship runtime-tested.
 */

#define SPARK_GLM5_NEXT_VALIDATION_TOKENS 4u
#define SPARK_GLM5_NEXT_VALIDATION_LANES 1u
#define SPARK_GLM5_NEXT_VALIDATION_EMBED_ROWS 8u

/* Glm5NextKv / Glm5NextIndexKv geometry (source/cuda/config.h +
 * layer.cuh): block-major pools, 64-token pages, DSA-layer strides. */
#define SPARK_GLM5_NEXT_VALIDATION_PAGES 34u
#define SPARK_GLM5_NEXT_VALIDATION_KV_ACCESS_WORDS 6u

/* -- tier 2b: the kpool indexer --------------------------------------------- */
#define SPARK_GLM5_NEXT_VALIDATION_DSA_CONTEXT 2064u
#define SPARK_GLM5_NEXT_VALIDATION_POOLS \
	(SPARK_GLM5_NEXT_VALIDATION_DSA_CONTEXT / 4u)

extern "C" int32_t SparkGlm5NextConfigureCudaModule(uint32_t *multiprocessor_count);
extern "C" int32_t SparkGlm5NextLaunchCudaWaveBegin(const SparkGlm5NextCudaWave *wave);
extern "C" int32_t SparkGlm5NextLaunchCudaLayerAttention(const SparkGlm5NextCudaWave *wave,uint32_t local_layer);
extern "C" int32_t SparkGlm5NextLaunchCudaLayerMlp(const SparkGlm5NextCudaWave *wave,uint32_t local_layer);

/* Host-mirrored geometry (config.h at tp_degree 1). */
#define SPARK_GLM5_NEXT_VHIDDEN SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM5_NEXT_VKDA_HEADS SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT
#define SPARK_GLM5_NEXT_VKDA_DIM SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION
#define SPARK_GLM5_NEXT_VKDA_LOW_RANK SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK
#define SPARK_GLM5_NEXT_VKDA_CONV SPARK_GLM5_NEXT_MODEL_KDA_SHORT_CONV_KERNEL
#define SPARK_GLM5_NEXT_VHEADS SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT
#define SPARK_GLM5_NEXT_VQUERY_A SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION
#define SPARK_GLM5_NEXT_VLATENT SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION
#define SPARK_GLM5_NEXT_VQK_NOPE SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION
#define SPARK_GLM5_NEXT_VVALUE_DIM SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION
#define SPARK_GLM5_NEXT_VATTN_COLS (SPARK_GLM5_NEXT_VHEADS * SPARK_GLM5_NEXT_VVALUE_DIM)
#define SPARK_GLM5_NEXT_VQ_B_ROWS (SPARK_GLM5_NEXT_VHEADS * SPARK_GLM5_NEXT_VQK_NOPE)
#define SPARK_GLM5_NEXT_VKV_SLOT_ELEMENTS SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION
#define SPARK_GLM5_NEXT_VDSA_HEADS SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT
#define SPARK_GLM5_NEXT_VDSA_DIM SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION
#define SPARK_GLM5_NEXT_VDSA_QUERY_DIM (SPARK_GLM5_NEXT_VDSA_HEADS * SPARK_GLM5_NEXT_VDSA_DIM)
#define SPARK_GLM5_NEXT_VHC SPARK_GLM5_NEXT_MODEL_HC_MULT
#define SPARK_GLM5_NEXT_VHC_MIX SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION
#define SPARK_GLM5_NEXT_VHC_FLAT (SPARK_GLM5_NEXT_VHC * SPARK_GLM5_NEXT_VHIDDEN)
#define SPARK_GLM5_NEXT_VDENSE_INTER SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define SPARK_GLM5_NEXT_VGATE_UP_ROWS (2u * SPARK_GLM5_NEXT_VDENSE_INTER)
#define SPARK_GLM5_NEXT_VEXPERTS SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT
#define SPARK_GLM5_NEXT_VTOP_K SPARK_GLM5_NEXT_MODEL_MOE_TOP_K
#define SPARK_GLM5_NEXT_VEXPERT_INTER SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION
#define SPARK_GLM5_NEXT_VW1_ROWS (2u * SPARK_GLM5_NEXT_VEXPERT_INTER)
#define SPARK_GLM5_NEXT_VW2_ROWS SPARK_GLM5_NEXT_VHIDDEN
#define SPARK_GLM5_NEXT_VW2_COLUMNS SPARK_GLM5_NEXT_VEXPERT_INTER
#define SPARK_GLM5_NEXT_VEXPERT_COLUMNS SPARK_GLM5_NEXT_VHIDDEN
#define SPARK_GLM5_NEXT_VROUTED_SCALE SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR
#define SPARK_GLM5_NEXT_VINDEX_PACKED SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION
#define SPARK_GLM5_NEXT_VINDEX_TOPK SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K
#define SPARK_GLM5_NEXT_VINDEX_WIDTH SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH

/* -- fixtures ---------------------------------------------------------------- */
static uint32_t SparkGlm5NextValRandomState;

static uint32_t SparkGlm5NextValNext(void)
{
	uint32_t value = SparkGlm5NextValRandomState;
	value ^= value << 13;
	value ^= value >> 17;
	value ^= value << 5;
	SparkGlm5NextValRandomState = value;
	return(value);
}

/* Round to nearest even: the LmFloatToBf16 contract (dtype.cuh). */
static uint16_t SparkGlm5NextValBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	uint32_t lsb = (bits >> 16) & 1u;
	uint32_t rounded = (bits + 0x7fffu + lsb) >> 16;
	return((uint16_t)(rounded & 0xffffu));
}

static float SparkGlm5NextValFromBf16(uint16_t value)
{
	uint32_t bits = ((uint32_t)value) << 16;
	float out;
	memcpy(&out,&bits,sizeof(out));
	return(out);
}

static void SparkGlm5NextValFill(uint16_t *packed,float *exact,uint64_t count,float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		/* Zero-mean grids (glm52's F3 lesson): a positive-mean expert grid
		 * turns the top-8 mixture into a quadratic amplifier of chain
		 * noise. */
		float value = (((float)(int32_t)(SparkGlm5NextValNext() & 0xffffu) -
			32768.0f) / 32768.0f) * scale;
		exact[index] = value;
		packed[index] = SparkGlm5NextValBf16(value);
	}
}

static void SparkGlm5NextValFillNorm(uint16_t *packed,float *exact,uint64_t count)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		exact[index] = 1.0f;
		packed[index] = SparkGlm5NextValBf16(1.0f);
	}
}

static int SparkGlm5NextValFail(const char *check,const char *detail)
{
	printf("FAIL %s: %s\n",check,detail);
	return(1);
}

typedef struct SparkGlm5NextValMetrics
{
	double max_relative_l2;
	double cosine;
	double max_abs;
} SparkGlm5NextValMetrics;

static void SparkGlm5NextValMeasure(SparkGlm5NextValMetrics *metrics,const float *actual,const float *reference,uint64_t count)
{
	double dot = 0.0, na = 0.0, nr = 0.0, max_abs = 0.0;
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		double a = (double)actual[index];
		double r = (double)reference[index];
		dot += a * r;
		na += a * a;
		nr += r * r;
		if ( fabs(a - r) > max_abs )
			max_abs = fabs(a - r);
	}
	metrics->max_abs = max_abs;
	metrics->cosine = (na > 0.0 && nr > 0.0) ? dot / (sqrt(na) * sqrt(nr)) : 1.0;
	metrics->max_relative_l2 = nr > 0.0 ? sqrt(na > 0.0 ? (na - 2.0 * dot + nr) : nr) / sqrt(nr) : sqrt(na);
}

static int SparkGlm5NextValReport(const char *check,const SparkGlm5NextValMetrics *metrics,double max_relative_l2,double minimum_cosine)
{
	int ok = metrics->max_relative_l2 <= max_relative_l2 && metrics->cosine >= minimum_cosine;
	printf("%s %-42s rel_l2 %.5f (max %.5f) cos %.7f (min %.7f) maxabs %.3e\n",
		ok ? "PASS" : "FAIL",check,metrics->max_relative_l2,max_relative_l2,
		metrics->cosine,minimum_cosine,metrics->max_abs);
	return(ok ? 0 : 1);
}

/* -- package expert codec, host mirror (glm52's FIXED addressing) ------------- */
#define SPARK_GLM5_NEXT_VAL_CODEC_INT6 2u
#define SPARK_GLM5_NEXT_VAL_CODEC_INT7 3u
#define SPARK_GLM5_NEXT_VAL_CODEC_INT8 4u
#define SPARK_GLM5_NEXT_VAL_CODEC_FP8 5u
#define SPARK_GLM5_NEXT_VAL_CODEC_NVFP4 6u
#define SPARK_GLM5_NEXT_VAL_CODEC_MXFP4 7u

#if defined(GLM5_NEXT_EXPERT_WEIGHT_CODEC)
#if GLM5_NEXT_EXPERT_WEIGHT_CODEC == 2
#define SPARK_GLM5_NEXT_VAL_CODEC SPARK_GLM5_NEXT_VAL_CODEC_INT6
#elif GLM5_NEXT_EXPERT_WEIGHT_CODEC == 3
#define SPARK_GLM5_NEXT_VAL_CODEC SPARK_GLM5_NEXT_VAL_CODEC_INT7
#elif GLM5_NEXT_EXPERT_WEIGHT_CODEC == 4
#define SPARK_GLM5_NEXT_VAL_CODEC SPARK_GLM5_NEXT_VAL_CODEC_INT8
#elif GLM5_NEXT_EXPERT_WEIGHT_CODEC == 5
#define SPARK_GLM5_NEXT_VAL_CODEC SPARK_GLM5_NEXT_VAL_CODEC_FP8
#elif GLM5_NEXT_EXPERT_WEIGHT_CODEC == 6
#define SPARK_GLM5_NEXT_VAL_CODEC SPARK_GLM5_NEXT_VAL_CODEC_NVFP4
#elif GLM5_NEXT_EXPERT_WEIGHT_CODEC == 7
#define SPARK_GLM5_NEXT_VAL_CODEC SPARK_GLM5_NEXT_VAL_CODEC_MXFP4
#else
#error "unsupported GLM5_NEXT_EXPERT_WEIGHT_CODEC for the validator oracle"
#endif
#endif

static uint32_t SparkGlm5NextValCodecStoredBits(uint32_t codec)
{
	return(codec == SPARK_GLM5_NEXT_VAL_CODEC_INT6 || codec == SPARK_GLM5_NEXT_VAL_CODEC_NVFP4 ||
		codec == SPARK_GLM5_NEXT_VAL_CODEC_MXFP4 ? 4u : 8u);
}

static uint32_t SparkGlm5NextValCodecScaleGroup(uint32_t codec)
{
	return(codec == SPARK_GLM5_NEXT_VAL_CODEC_INT6 || codec == SPARK_GLM5_NEXT_VAL_CODEC_MXFP4 ? 32u :
		(codec == SPARK_GLM5_NEXT_VAL_CODEC_NVFP4 ? 16u : 128u));
}

static uint32_t SparkGlm5NextValCodecUsesSignedIntGrid(uint32_t codec)
{
	return(codec == SPARK_GLM5_NEXT_VAL_CODEC_INT6 || codec == SPARK_GLM5_NEXT_VAL_CODEC_INT7 ||
		codec == SPARK_GLM5_NEXT_VAL_CODEC_INT8 ? 1u : 0u);
}

/* E4M3 decode, host form of the hardware conversion for every finite code. */
static float SparkGlm5NextValE4m3Decode(uint8_t code)
{
	int32_t sign = (code & 0x80u) != 0u ? -1 : 1;
	uint32_t exponent = (code >> 3) & 0xfu;
	uint32_t mantissa = code & 7u;
	float value;
	if ( exponent == 0u )
		return((float)sign * ((float)mantissa * (1.0f / 512.0f)));
	if ( exponent == 15u && mantissa == 7u )
		return((float)sign * NAN);
	/* bias 7: (1 + m/8) * 2^(e-7), min normal e=1 -> 2^-6. */
	value = (1.0f + ((float)mantissa) * 0.125f) *
		(float)(1u << (int32_t)(exponent - 7u < 31u ? exponent - 7u : 0u));
	if ( exponent < 7u )
		value = (1.0f + ((float)mantissa) * 0.125f) /
			(float)(1u << (7u - exponent));
	return((float)sign * value);
}

/* E2M1 nibble: magnitudes {0,.5,1,1.5,2,3,4,6}, bit 3 is the sign. */
static float SparkGlm5NextValE2m1Decode(uint8_t nibble)
{
	static const float magnitudes[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	float magnitude = magnitudes[nibble & 7u];
	return((nibble & 8u) != 0u ? -magnitude : magnitude);
}

/* codec addressing helpers: compiled in BOTH modes (the selftest and
 * the GPU tiers both dequant expert slabs). */
/* Signed integer grid bounds. */
static int32_t SparkGlm5NextValCodecCodeMinimum(uint32_t codec)
{
	return(codec == SPARK_GLM5_NEXT_VAL_CODEC_INT6 ? -31 :
		codec == SPARK_GLM5_NEXT_VAL_CODEC_INT7 ? -63 :
		codec == SPARK_GLM5_NEXT_VAL_CODEC_INT8 ? -127 : 0);
}

static int32_t SparkGlm5NextValCodecCodeMaximum(uint32_t codec)
{
	return(codec == SPARK_GLM5_NEXT_VAL_CODEC_INT6 ? 31 :
		codec == SPARK_GLM5_NEXT_VAL_CODEC_INT7 ? 63 :
		codec == SPARK_GLM5_NEXT_VAL_CODEC_INT8 ? 127 : 0);
}

/* Scale plane geometry: per-expert blocks of scale-group columns. */
static uint64_t SparkGlm5NextValScaleBlocksPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	uint32_t group = SparkGlm5NextValCodecScaleGroup(codec);
	return(((uint64_t)((rows + 127u) / 128u)) * ((columns + group - 1u) / group));
}

static uint64_t SparkGlm5NextValScaleBytesPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	return(SparkGlm5NextValScaleBlocksPerExpert(codec,rows,columns) * 4u);
}

static uint64_t SparkGlm5NextValScaleBufferBytes(uint32_t codec,uint32_t expert_count,uint32_t rows,uint32_t columns)
{
	/* FP8 e4m3 [128,128] f32 scales; nvfp4 carries per-expert globals. */
	uint64_t per = SparkGlm5NextValScaleBytesPerExpert(codec,rows,columns);
	return(codec == SPARK_GLM5_NEXT_VAL_CODEC_NVFP4
		? (uint64_t)expert_count * (per + sizeof(float))
		: (uint64_t)expert_count * per);
}

static uint64_t SparkGlm5NextValScaleExpertOffset(uint32_t codec,uint32_t expert_count,uint32_t expert,uint32_t rows,uint32_t columns)
{
	uint64_t per = SparkGlm5NextValScaleBytesPerExpert(codec,rows,columns);
	return(codec == SPARK_GLM5_NEXT_VAL_CODEC_NVFP4
		? (uint64_t)expert * (per + sizeof(float)) + ((uint64_t)expert_count - 1u - expert) * 0u
		: (uint64_t)expert * per);
}

static uint64_t SparkGlm5NextValPayloadRowBytes(uint32_t codec,uint32_t columns)
{
	return(((uint64_t)columns * SparkGlm5NextValCodecStoredBits(codec) + 7u) / 8u);
}

static uint64_t SparkGlm5NextValPayloadBytesPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	return((uint64_t)rows * SparkGlm5NextValPayloadRowBytes(codec,columns));
}

/* THE glm52 F3 FIX, inherited: expert-major slab offset in the payload
 * buffer - the mirror of the kernel's weight tensor map and the pack
 * layout. Before the fix only the scale index carried the expert
 * dimension and every oracle forward decoded slab 0. */
static uint64_t SparkGlm5NextValPayloadExpertOffset(uint32_t codec,uint32_t expert,uint32_t rows,uint32_t columns)
{
	return((uint64_t)expert * SparkGlm5NextValPayloadBytesPerExpert(codec,rows,columns));
}

/* Read one stored code at (row, column) of an expert's payload slab. */
static uint8_t SparkGlm5NextValPayloadCode(const uint8_t *payload,uint32_t codec,uint32_t row,uint32_t column,uint32_t columns)
{
	uint64_t row_bytes = SparkGlm5NextValPayloadRowBytes(codec,columns);
	const uint8_t *row_base = payload + (uint64_t)row * row_bytes;
	if ( SparkGlm5NextValCodecStoredBits(codec) == 8u )
		return(row_base[column]);
	return((row_base[column / 2u] >> ((column & 1u) * 4u)) & 0xfu);
}


/* -- oracle formulas --------------------------------------------------------- */

static float SparkGlm5NextValSigmoid(float value)
{
	return(1.0f / (1.0f + expf(-value)));
}

/* LmBoundedDecay restated: exp(lb * sigmoid(exp(A_log) * (logit + bias))). */
static float SparkGlm5NextValBoundedDecay(float logit,float bias,float head_log_scale,float lower_bound)
{
	return(expf(lower_bound * SparkGlm5NextValSigmoid(expf(head_log_scale) * (logit + bias))));
}

/* RMS norm (plain, weight applied). */
static void SparkGlm5NextValRmsNorm(float *row,const float *weight,uint32_t dimension,float epsilon)
{
	float sum = 0.0f;
	uint32_t index;
	for (index = 0u; index < dimension; index++)
		sum += row[index] * row[index];
	float inverse = 1.0f / sqrtf(sum / (float)dimension + epsilon);
	for (index = 0u; index < dimension; index++)
		row[index] = row[index] * inverse * weight[index];
}

/* mHC site math, the reference restated:
 *   mix_h   = fn_h . (streams / rms(streams))          (unweighted norm
 *                                                        over the FLAT row)
 *   pre_h   = sigmoid(mix_h*s0 + b_h) + eps
 *   post_h  = 2*sigmoid(mix_{H+h}*s1 + b_{H+h})
 *   comb    = softmax(mix_{2H..}*s2 + b_{2H..}) + eps, one column norm,
 *             then 19 alternating row/col normalisations (sinkhorn 20)
 *   collapse= sum_h pre_h * stream_h
 *   streams_new[s] = post_s*out + sum_r comb[r][s]*snapshot_r
 * All in fp32; the DEVICE runs the same formulas in bf16 storage and
 * fp32 arithmetic - the oracle's inputs are the bf16-rounded values. */
static void SparkGlm5NextValHcSite(const float *streams,const float *fn,const float *base,const float *scale,
	float epsilon,uint32_t hc,uint32_t dimension,float *mixes_out,
	float *pre_out,float *post_out,float *comb_out,float *collapsed,float *snapshot)
{
	uint32_t mix_rows = (2u + hc) * hc;
	uint32_t flat = hc * dimension;
	uint32_t i, j, h;
	float norm = 0.0f;
	for (i = 0u; i < flat; i++)
		norm += streams[i] * streams[i];
	norm = 1.0f / sqrtf(norm / (float)flat + 1e-05f);
	for (i = 0u; i < mix_rows; i++)
	{
		float dot = 0.0f;
		for (j = 0u; j < flat; j++)
			dot += fn[(uint64_t)i * flat + j] * streams[j];
		mixes_out[i] = dot * norm;
		if ( collapsed != 0 )
			(void)0;
	}
	for (h = 0u; h < hc; h++)
	{
		pre_out[h] = SparkGlm5NextValSigmoid(mixes_out[h] * scale[0] + base[h]) + epsilon;
		post_out[h] = 2.0f * SparkGlm5NextValSigmoid(mixes_out[hc + h] * scale[1] + base[hc + h]);
	}
	/* comb: softmax over rows, then column norm once, then (row,col) x19. */
	for (i = 0u; i < hc; i++)
	{
		float maximum = -3.0e38f, total = 0.0f;
		for (j = 0u; j < hc; j++)
		{
			comb_out[i * hc + j] = mixes_out[2u * hc + i * hc + j] * scale[2] + base[2u * hc + i * hc + j];
			if (comb_out[i * hc + j] > maximum)
				maximum = comb_out[i * hc + j];
		}
		for (j = 0u; j < hc; j++)
			total += (comb_out[i * hc + j] = expf(comb_out[i * hc + j] - maximum));
		for (j = 0u; j < hc; j++)
			comb_out[i * hc + j] = comb_out[i * hc + j] / total + epsilon;
	}
	for (j = 0u; j < hc; j++)
	{
		float total = 0.0f;
		for (i = 0u; i < hc; i++)
			total += comb_out[i * hc + j];
		for (i = 0u; i < hc; i++)
			comb_out[i * hc + j] /= total + epsilon;
	}
	for (uint32_t iteration = 0u; iteration < 19u; iteration++)
	{
		for (i = 0u; i < hc; i++)
		{
			float total = 0.0f;
			for (j = 0u; j < hc; j++)
				total += comb_out[i * hc + j];
			for (j = 0u; j < hc; j++)
				comb_out[i * hc + j] /= total + epsilon;
		}
		for (j = 0u; j < hc; j++)
		{
			float total = 0.0f;
			for (i = 0u; i < hc; i++)
				total += comb_out[i * hc + j];
			for (i = 0u; i < hc; i++)
				comb_out[i * hc + j] /= total + epsilon;
		}
	}
	if (collapsed != 0 && snapshot != 0)
	{
		for (i = 0u; i < flat; i++)
			snapshot[i] = streams[i];
		for (h = 0u; h < hc; h++)
		{
			float weight = pre_out[h];
			for (j = 0u; j < dimension; j++)
			{
				uint64_t index = ((uint64_t)h * dimension) + j;
				float value = weight * streams[index];
				collapsed[j] = h == 0u ? value : collapsed[j] + value;
			}
		}
	}
}

static void SparkGlm5NextValHcPost(const float *out,const float *snapshot,const float *post,const float *comb,
	uint32_t hc,uint32_t dimension,float *streams_out)
{
	for (uint32_t s = 0u; s < hc; s++)
		for (uint32_t j = 0u; j < dimension; j++)
		{
			float value = post[s] * out[j];
			for (uint32_t r = 0u; r < hc; r++)
				value += comb[r * hc + s] * snapshot[((uint64_t)r * dimension) + j];
			streams_out[((uint64_t)s * dimension) + j] = value;
		}
}

/* KDA forward for one token (decode step), state carried in [heads][k][v]
 * fp32. Inputs are the ALREADY-bf16-rounded fixture values. */
static void SparkGlm5NextValKdaToken(
	const float *collapsed,          /* [hidden] */
	const float *qkv_beta,           /* [q|k|v|beta fused rows, hidden] */
	const float *conv_q, const float *conv_k, const float *conv_v, /* [dim, kernel] */
	const float *decay_down_gate,    /* [2*low_rank, hidden] fused */
	const float *decay_up,           /* [dim, low_rank] */
	const float *gate_up,            /* [dim, low_rank] */
	const float *dt_bias, const float *a_log,
	const float *out_norm, const float *out_weight, /* [hidden, dim] */
	uint16_t *q_window, uint16_t *k_window, uint16_t *v_window, /* [dim, kernel] bf16 */
	float *state,                    /* [heads][k][v] */
	float *output)                   /* [hidden] */
{
	const uint32_t heads = SPARK_GLM5_NEXT_VKDA_HEADS;
	const uint32_t dim_per_head = SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION;
	const uint32_t qk = SPARK_GLM5_NEXT_VKDA_DIM;
	const uint32_t kernel = SPARK_GLM5_NEXT_VKDA_CONV;
	float q[SPARK_GLM5_NEXT_VKDA_DIM], k[SPARK_GLM5_NEXT_VKDA_DIM], v[SPARK_GLM5_NEXT_VKDA_DIM];
	float beta[SPARK_GLM5_NEXT_VKDA_HEADS];
	float decay_latent[SPARK_GLM5_NEXT_VKDA_LOW_RANK], gate_latent[SPARK_GLM5_NEXT_VKDA_LOW_RANK];
	float retention[SPARK_GLM5_NEXT_VKDA_DIM], gate[SPARK_GLM5_NEXT_VKDA_DIM];
	float core[SPARK_GLM5_NEXT_VKDA_DIM];
	uint32_t index, head, channel;

	/* Projections from the fused tensor. */
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += qkv_beta[index * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		q[index] = sum;
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += qkv_beta[(qk + index) * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		k[index] = sum;
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += qkv_beta[(2u * qk + index) * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		v[index] = sum;
	}
	for (head = 0u; head < heads; head++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += qkv_beta[(2u * qk + qk + head) * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		beta[head] = SparkGlm5NextValSigmoid(sum);
	}
	/* Short convolutions with SWISH, each over its own bf16 window. */
	for (index = 0u; index < qk; index++)
	{
		float taps[8];
		float total;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			taps[tap] = SparkGlm5NextValFromBf16(q_window[index * kernel + tap]);
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			q_window[index * kernel + tap] = q_window[index * kernel + tap + 1u];
		q_window[index * kernel + kernel - 1u] = SparkGlm5NextValBf16(q[index]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += SparkGlm5NextValFromBf16(q_window[index * kernel + tap]) * conv_q[index * kernel + tap];
		total = total * SparkGlm5NextValSigmoid(total);
		q[index] = total;
	}
	for (index = 0u; index < qk; index++)
	{
		float total;
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			k_window[index * kernel + tap] = k_window[index * kernel + tap + 1u];
		k_window[index * kernel + kernel - 1u] = SparkGlm5NextValBf16(k[index]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += SparkGlm5NextValFromBf16(k_window[index * kernel + tap]) * conv_k[index * kernel + tap];
		total = total * SparkGlm5NextValSigmoid(total);
		k[index] = total;
	}
	for (index = 0u; index < qk; index++)
	{
		float total;
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			v_window[index * kernel + tap] = v_window[index * kernel + tap + 1u];
		v_window[index * kernel + kernel - 1u] = SparkGlm5NextValBf16(v[index]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += SparkGlm5NextValFromBf16(v_window[index * kernel + tap]) * conv_v[index * kernel + tap];
		total = total * SparkGlm5NextValSigmoid(total);
		v[index] = total;
	}
	/* qk L2 norm per head. */
	for (head = 0u; head < heads; head++)
	{
		float *qh = q + head * dim_per_head;
		float *kh = k + head * dim_per_head;
		float nq = 0.0f, nk = 0.0f;
		for (channel = 0u; channel < dim_per_head; channel++)
		{
			nq += qh[channel] * qh[channel];
			nk += kh[channel] * kh[channel];
		}
		nq = 1.0f / sqrtf(nq + 1e-5f * 0.0f + (nq > 0.0f ? 0.0f : 1.0f));
		nk = 1.0f / sqrtf(nk + (nk > 0.0f ? 0.0f : 1.0f));
		(void)nq; (void)nk;
	}
	/* The kernel's per-head L2 uses RMS-style epsilon; restate exactly:
	 * LmL2NormalisePerHeadKernel scales by 1/sqrt(sum(x^2) + eps). */
	for (head = 0u; head < heads; head++)
	{
		float *qh = q + head * dim_per_head;
		float *kh = k + head * dim_per_head;
		float nq = 0.0f, nk = 0.0f;
		for (channel = 0u; channel < dim_per_head; channel++)
		{
			nq += qh[channel] * qh[channel];
			nk += kh[channel] * kh[channel];
		}
		nq = 1.0f / sqrtf(nq + 1e-5f);
		nk = 1.0f / sqrtf(nk + 1e-5f);
		for (channel = 0u; channel < dim_per_head; channel++)
		{
			qh[channel] *= nq;
			kh[channel] *= nk;
		}
	}
	/* Low-rank decay and gate. */
	for (index = 0u; index < SPARK_GLM5_NEXT_VKDA_LOW_RANK; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += decay_down_gate[index * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		decay_latent[index] = sum;
	}
	for (index = 0u; index < SPARK_GLM5_NEXT_VKDA_LOW_RANK; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += decay_down_gate[(SPARK_GLM5_NEXT_VKDA_LOW_RANK + index) * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		gate_latent[index] = sum;
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VKDA_LOW_RANK; j++)
			sum += decay_up[(uint64_t)index * SPARK_GLM5_NEXT_VKDA_LOW_RANK + j] * decay_latent[j];
		retention[index] = SparkGlm5NextValBoundedDecay(sum,dt_bias[index],
			a_log[index / dim_per_head],-5.0f);
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VKDA_LOW_RANK; j++)
			sum += gate_up[(uint64_t)index * SPARK_GLM5_NEXT_VKDA_LOW_RANK + j] * gate_latent[j];
		gate[index] = sum;
	}
	/* Delta rule: S = (I - beta k k^T) Diag(a) S + beta k v^T; y = S^T q. */
	for (head = 0u; head < heads; head++)
	{
		const float *qh = q + head * dim_per_head;
		const float *kh = k + head * dim_per_head;
		const float *vh = v + head * dim_per_head;
		const float *ah = retention + head * dim_per_head;
		float *sh = state + (uint64_t)head * dim_per_head * dim_per_head;
		float kh_qh = 0.0f;
		for (channel = 0u; channel < dim_per_head; channel++)
			kh_qh += kh[channel] * qh[channel];
		(void)kh_qh;
		/* apply decay, then the rank-1 update, then read. */
		for (uint32_t key = 0u; key < dim_per_head; key++)
			for (uint32_t value = 0u; value < dim_per_head; value++)
				sh[(uint64_t)key * dim_per_head + value] *= ah[key];
		for (uint32_t value = 0u; value < dim_per_head; value++)
		{
			float kh_vh = beta[head] * vh[value];
			for (uint32_t key = 0u; key < dim_per_head; key++)
				sh[(uint64_t)key * dim_per_head + value] += kh_vh * kh[key];
		}
		for (uint32_t value = 0u; value < dim_per_head; value++)
		{
			float dot = 0.0f;
			for (uint32_t key = 0u; key < dim_per_head; key++)
				dot += sh[(uint64_t)key * dim_per_head + value] * qh[key];
			core[head * dim_per_head + value] = dot;
		}
	}
	/* RMSNormGated: per-head fp32 norm + weight, then * sigmoid(gate). */
	for (head = 0u; head < heads; head++)
	{
		float *row = core + head * dim_per_head;
		float sum = 0.0f;
		for (channel = 0u; channel < dim_per_head; channel++)
			sum += row[channel] * row[channel];
		float inverse = 1.0f / sqrtf(sum / (float)dim_per_head + 1e-5f);
		for (channel = 0u; channel < dim_per_head; channel++)
			row[channel] = row[channel] * inverse * out_norm[channel] *
				SparkGlm5NextValSigmoid(gate[head * dim_per_head + channel]);
	}
	/* o_proj. */
	for (index = 0u; index < SPARK_GLM5_NEXT_VHIDDEN; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < qk; j++)
			sum += out_weight[(uint64_t)index * qk + j] * core[j];
		output[index] = sum;
	}
}

/* rope-0 MLA forward for one token (decode), absorbed form. */
static void SparkGlm5NextValMlaToken(
	const float *collapsed,
	const float *q_a, const float *q_a_norm, const float *q_b,
	const float *kv_a, const float *kv_a_norm,
	const float *kv_b_key, const float *kv_b_value, const float *o_proj,
	const float *latents,               /* [context][latent] prior tokens */
	uint32_t context,
	float *latent_out,                  /* [latent] this token's cache row */
	float *output)                      /* [hidden] */
{
	const uint32_t heads = SPARK_GLM5_NEXT_VHEADS;
	const uint32_t latent = SPARK_GLM5_NEXT_VLATENT;
	const uint32_t nope = SPARK_GLM5_NEXT_VQK_NOPE;
	const uint32_t vdim = SPARK_GLM5_NEXT_VVALUE_DIM;
	float q_lora[SPARK_GLM5_NEXT_VQUERY_A];
	float q_rows[SPARK_GLM5_NEXT_VQ_B_ROWS];
	float scale = 1.0f / 256.0f;
	(void)scale;
	/* q_a_layernorm is a plain RMS norm at 1e-5 (mla lora norms take the
	 * model epsilon; glm5_next passes rms_norm_eps everywhere). */
	uint32_t index;
	for (index = 0u; index < SPARK_GLM5_NEXT_VQUERY_A; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += q_a[(uint64_t)index * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		q_lora[index] = sum;
	}
	SparkGlm5NextValRmsNorm(q_lora,q_a_norm,SPARK_GLM5_NEXT_VQUERY_A,1e-5f);
	for (index = 0u; index < SPARK_GLM5_NEXT_VQ_B_ROWS; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VQUERY_A; j++)
			sum += q_b[(uint64_t)index * SPARK_GLM5_NEXT_VQUERY_A + j] * q_lora[j];
		q_rows[index] = sum;
	}
	/* kv_a -> normed latent row. */
	for (index = 0u; index < latent; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += kv_a[(uint64_t)index * SPARK_GLM5_NEXT_VHIDDEN + j] * collapsed[j];
		latent_out[index] = sum;
	}
	SparkGlm5NextValRmsNorm(latent_out,kv_a_norm,latent,1e-5f);
	/* Per head: absorb q_nope into the latent (kv_b_key is stored
	 * TRANSPOSED per head: [latent][nope]), softmax over context+1, read
	 * back through kv_b_value ([vdim][latent] per head). */
	{
		float values[SPARK_GLM5_NEXT_VATTN_COLS];
		float scores[512];
		for (uint32_t head = 0u; head < heads; head++)
		{
			const float *qh = q_rows + head * nope;
			const float *key_t = kv_b_key + (uint64_t)head * latent * nope; /* [latent][nope] */
			const float *value_w = kv_b_value + (uint64_t)head * vdim * latent; /* [vdim][latent] */
			float q_absorbed[512];
			uint32_t position;
			for (index = 0u; index < latent; index++)
			{
				float dot = 0.0f;
				for (uint32_t j = 0u; j < nope; j++)
					dot += key_t[(uint64_t)index * nope + j] * qh[j];
				q_absorbed[index] = dot;
			}
			{
				float maximum = -3.0e38f, total = 0.0f;
				for (position = 0u; position <= context; position++)
				{
					const float *row = position < context ? latents + (uint64_t)position * latent : latent_out;
					float dot = 0.0f;
					for (index = 0u; index < latent; index++)
						dot += q_absorbed[index] * row[index];
					scores[position] = dot * 0.0625f; /* 256 ** -0.5 */
					if (scores[position] > maximum)
						maximum = scores[position];
				}
				for (position = 0u; position <= context; position++)
					total += (scores[position] = expf(scores[position] - maximum));
				for (index = 0u; index < vdim; index++)
				{
					float sum = 0.0f;
					for (position = 0u; position <= context; position++)
					{
						const float *row = position < context ? latents + (uint64_t)position * latent : latent_out;
						float weighted = 0.0f;
						for (uint32_t j = 0u; j < latent; j++)
							weighted += value_w[(uint64_t)index * latent + j] * row[j];
						sum += (scores[position] / total) * weighted;
					}
					values[head * vdim + index] = sum;
				}
			}
		}
		for (index = 0u; index < SPARK_GLM5_NEXT_VHIDDEN; index++)
		{
			float sum = 0.0f;
			for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VATTN_COLS; j++)
				sum += o_proj[(uint64_t)index * SPARK_GLM5_NEXT_VATTN_COLS + j] * values[j];
			output[index] = sum;
		}
	}
}

/* Router (sigmoid + noaux_tc at n_group 1 = plain top-8 with the frozen
 * bias selecting) and the renormalised mixture. */
static void SparkGlm5NextValRouter(const float *router,const float *correction,const float *hidden,
	uint32_t *selected,float *weights)
{
	float scores[SPARK_GLM5_NEXT_VEXPERTS];
	float choice[SPARK_GLM5_NEXT_VEXPERTS];
	uint32_t expert;
	for (expert = 0u; expert < SPARK_GLM5_NEXT_VEXPERTS; expert++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
			sum += router[(uint64_t)expert * SPARK_GLM5_NEXT_VHIDDEN + j] * hidden[j];
		scores[expert] = SparkGlm5NextValSigmoid(sum);
		choice[expert] = scores[expert] + correction[expert];
	}
	for (uint32_t slot = 0u; slot < SPARK_GLM5_NEXT_VTOP_K; slot++)
	{
		uint32_t best = 0u;
		for (expert = 1u; expert < SPARK_GLM5_NEXT_VEXPERTS; expert++)
			if (choice[expert] > choice[best])
				best = expert;
		selected[slot] = best;
		choice[best] = -3.0e38f;
	}
	{
		float total = 0.0f;
		for (uint32_t slot = 0u; slot < SPARK_GLM5_NEXT_VTOP_K; slot++)
		{
			weights[slot] = scores[selected[slot]];
			total += weights[slot];
		}
		for (uint32_t slot = 0u; slot < SPARK_GLM5_NEXT_VTOP_K; slot++)
			weights[slot] = weights[slot] / (total + 1e-20f) * SPARK_GLM5_NEXT_VROUTED_SCALE;
	}
}

/* -- tier drivers ------------------------------------------------------------ */

typedef struct SparkGlm5NextValMatrix
{
	uint32_t rows;
	uint32_t columns;
	float *host;
	void *device;
} SparkGlm5NextValMatrix;

static int SparkGlm5NextValAllocMatrix(SparkGlm5NextValMatrix *matrix,uint32_t rows,uint32_t columns,int mode,float scale)
{
	uint16_t *packed;
	uint64_t count = (uint64_t)rows * columns;
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->host = (float *)malloc(count * sizeof(float));
	packed = (uint16_t *)malloc(count * sizeof(uint16_t));
	if (matrix->host == 0 || packed == 0)
		return(SparkGlm5NextValFail("fixture","host_alloc"));
	if (cudaMalloc((void **)&matrix->device,count * sizeof(uint16_t)) != cudaSuccess)
		return(SparkGlm5NextValFail("fixture","device_alloc"));
	SparkGlm5NextValRandomState += 101u;
	if (mode == 1)
		SparkGlm5NextValFillNorm(packed,matrix->host,count);
	else
		SparkGlm5NextValFill(packed,matrix->host,count,scale);
	if (cudaMemcpy(matrix->device,packed,count * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess)
		return(SparkGlm5NextValFail("fixture","weight_upload"));
	free(packed);
	return(0);
}

static void SparkGlm5NextValFreeMatrix(SparkGlm5NextValMatrix *matrix)
{
	free(matrix->host);
	cudaFree(matrix->device);
	memset(matrix,0,sizeof(*matrix));
}

/* -- host-executable oracle selftest (tier 2, no GPU) ------------------------
 * Compiled in BOTH modes: the host gate runs it via
 * tests/test_glm5_next_cuda_validator_tier2_oracle.py (cuda_stub, no
 * toolkit), and the GPU binary runs it before any tier driver so the
 * formulas are proven on the validating host itself. */
static int SparkGlm5NextValSelftestAssert(int condition,const char *what)
{
	if (!condition)
	{
		printf("FAIL selftest: %s\n",what);
		return(1);
	}
	return(0);
}

static int SparkGlm5NextValOracleSelftest(void)
{
	int failures = 0;
	/* 1. Bounded decay: the reference forget gate. */
	failures += SparkGlm5NextValSelftestAssert(
		fabsf(SparkGlm5NextValBoundedDecay(0.0f,0.0f,0.0f,-5.0f) -
			expf(-5.0f * 0.5f)) < 1e-6f,"bounded decay at zero");
	{
		float value = SparkGlm5NextValBoundedDecay(2.0f,-1.0f,0.5f,-5.0f);
		float expect = expf(-5.0f * SparkGlm5NextValSigmoid(expf(0.5f) * 1.0f));
		failures += SparkGlm5NextValSelftestAssert(fabsf(value - expect) < 1e-6f,
			"bounded decay formula");
		failures += SparkGlm5NextValSelftestAssert(value > 0.0f && value <= 1.0f,
			"bounded decay range (0,1]");
	}
	/* 2. Expert-major payload addressing (the glm52 F3 fix, inherited). */
	{
		uint64_t slab = SparkGlm5NextValPayloadBytesPerExpert(SPARK_GLM5_NEXT_VAL_CODEC_FP8,64u,128u);
		failures += SparkGlm5NextValSelftestAssert(slab == 64u * 128u,
			"fp8 payload slab = rows*columns");
		failures += SparkGlm5NextValSelftestAssert(
			SparkGlm5NextValPayloadExpertOffset(SPARK_GLM5_NEXT_VAL_CODEC_FP8,3u,64u,128u) ==
			3u * slab,"expert-major payload offset");
		failures += SparkGlm5NextValSelftestAssert(
			SparkGlm5NextValScaleExpertOffset(SPARK_GLM5_NEXT_VAL_CODEC_FP8,288u,7u,64u,128u) ==
			7u * SparkGlm5NextValScaleBytesPerExpert(SPARK_GLM5_NEXT_VAL_CODEC_FP8,64u,128u),
			"expert-major scale offset");
	}
	/* 3. fp8 e4m3 decode round trip on exactly representable codes. */
	{
		uint8_t code;
		int ok = 1;
		for (code = 0u; code < 255u; code++)
		{
			if ((code & 0x7fu) == 0x7fu)
				continue; /* NaN encodings */
			float decoded = SparkGlm5NextValE4m3Decode(code);
			if (!(decoded == decoded))
				ok = 0;
		}
		failures += SparkGlm5NextValSelftestAssert(ok,"e4m3 finite decode");
		failures += SparkGlm5NextValSelftestAssert(
			SparkGlm5NextValE4m3Decode(0x38u) == 1.0f &&
			SparkGlm5NextValE4m3Decode(0xb8u) == -1.0f,"e4m3 decode values");
	}
	/* 4. mHC: pre/post ranges and doubly-stochastic comb. */
	{
		float streams[SPARK_GLM5_NEXT_VHC * 4];
		float fn[SPARK_GLM5_NEXT_VHC_MIX * 16];
		float base[SPARK_GLM5_NEXT_VHC_MIX];
		float scale[3] = {0.5f,0.5f,0.5f};
		float mixes[SPARK_GLM5_NEXT_VHC_MIX];
		float pre[SPARK_GLM5_NEXT_VHC],post[SPARK_GLM5_NEXT_VHC];
		float comb[SPARK_GLM5_NEXT_VHC * SPARK_GLM5_NEXT_VHC];
		float collapsed[4],snapshot[SPARK_GLM5_NEXT_VHC * 4];
		uint32_t i,j;
		for (i = 0u; i < SPARK_GLM5_NEXT_VHC * 4u; i++)
			streams[i] = ((float)(i % 7u) - 3.0f) * 0.25f;
		for (i = 0u; i < SPARK_GLM5_NEXT_VHC_MIX * 16u; i++)
			fn[i] = ((float)(i % 5u) - 2.0f) * 0.125f;
		for (i = 0u; i < SPARK_GLM5_NEXT_VHC_MIX; i++)
			base[i] = ((float)(i % 3u) - 1.0f) * 0.5f;
		SparkGlm5NextValHcSite(streams,fn,base,scale,1e-6f,
			SPARK_GLM5_NEXT_VHC,4u,mixes,pre,post,comb,collapsed,snapshot);
		for (i = 0u; i < SPARK_GLM5_NEXT_VHC; i++)
		{
			failures += SparkGlm5NextValSelftestAssert(pre[i] > 1e-6f && pre[i] <= 1.0f + 1e-6f + 1e-6f,
				"pre in (eps, 1+eps]");
			failures += SparkGlm5NextValSelftestAssert(post[i] >= 0.0f && post[i] <= 2.0f,
				"post in [0,2]");
		}
		for (j = 0u; j < SPARK_GLM5_NEXT_VHC; j++)
		{
			float total = 0.0f;
			for (i = 0u; i < SPARK_GLM5_NEXT_VHC; i++)
				total += comb[i * SPARK_GLM5_NEXT_VHC + j];
			failures += SparkGlm5NextValSelftestAssert(fabsf(total - 1.0f) < 1e-3f,
				"comb column sums to 1 (doubly stochastic)");
		}
	}
	/* 5. kpool expansion: tail placement + sentinel semantics. */
	{
		uint32_t selected_pools[2] = {0u,1u};
		uint32_t context = 4u * 5u + 2u; /* 5 complete pools, tail 2 */
		uint32_t sequence = 0u, width = 11u;
		uint32_t list[11];
		/* Restate Glm5NextPoolExpandKernel inline. */
		uint32_t index;
		uint32_t select = 2u;
		for (index = 0u; index < width; index++)
		{
			uint32_t position = 0xFFFFFFFFu;
			if (index < 8u)
			{
				uint32_t pool = selected_pools[index / 4u];
				position = pool * 4u + index % 4u;
				if (position >= context)
					position = 0xFFFFFFFFu;
			}
			else
			{
				uint32_t tail_count = context % 4u;
				uint32_t tail_index = index - 8u;
				if (tail_index < tail_count)
					position = context - tail_count + tail_index;
			}
			list[index] = position;
		}
		failures += SparkGlm5NextValSelftestAssert(list[7] == 7u,"pool 1 last token");
		failures += SparkGlm5NextValSelftestAssert(list[8] == 20u && list[9] == 21u,"tail placement");
		failures += SparkGlm5NextValSelftestAssert(list[10] == 0xFFFFFFFFu,"empty tail slot sentinel");
		(void)sequence;
	}
	/* 6. KDA split section offsets (pack V2 contract). */
	{
		uint32_t heads = 64u, dim = 128u;
		uint32_t qk = heads * dim;
		failures += SparkGlm5NextValSelftestAssert(
			2u * qk + qk + heads == 3u * qk + heads,"fused q|k|v|beta row count");
	}
	/* 7. Execute the KDA token oracle end to end (one token, real
	 * geometry): output finite and non-degenerate. */
	{
		static float collapsed[SPARK_GLM5_NEXT_VHIDDEN];
		static float qkv_beta[(3u * SPARK_GLM5_NEXT_VKDA_DIM + SPARK_GLM5_NEXT_VKDA_HEADS) * SPARK_GLM5_NEXT_VHIDDEN];
		static float conv[3u][SPARK_GLM5_NEXT_VKDA_DIM * SPARK_GLM5_NEXT_VKDA_CONV];
		static float down_gate[2u * SPARK_GLM5_NEXT_VKDA_LOW_RANK * SPARK_GLM5_NEXT_VHIDDEN];
		static float up[2u][SPARK_GLM5_NEXT_VKDA_DIM * SPARK_GLM5_NEXT_VKDA_LOW_RANK];
		static float dt[SPARK_GLM5_NEXT_VKDA_DIM], alog[SPARK_GLM5_NEXT_VKDA_HEADS];
		static float onorm[SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION];
		static float ow[SPARK_GLM5_NEXT_VHIDDEN * SPARK_GLM5_NEXT_VKDA_DIM];
		static float state[(uint64_t)SPARK_GLM5_NEXT_VKDA_HEADS * 128u * 128u];
		static float output[SPARK_GLM5_NEXT_VHIDDEN];
		static uint16_t wq[SPARK_GLM5_NEXT_VKDA_DIM * SPARK_GLM5_NEXT_VKDA_CONV];
		static uint16_t wk[SPARK_GLM5_NEXT_VKDA_DIM * SPARK_GLM5_NEXT_VKDA_CONV];
		static uint16_t wv[SPARK_GLM5_NEXT_VKDA_DIM * SPARK_GLM5_NEXT_VKDA_CONV];
		uint64_t i;
		uint32_t j;
		int finite = 1;
		float magnitude = 0.0f;
		for (i = 0u; i < (uint64_t)SPARK_GLM5_NEXT_VHIDDEN; i++)
			collapsed[i] = ((float)(i % 11u) - 5.0f) * 0.02f;
		for (i = 0u; i < sizeof(qkv_beta) / sizeof(float); i++)
			qkv_beta[i] = ((float)(i % 7u) - 3.0f) * 0.01f;
		for (i = 0u; i < sizeof(conv) / sizeof(float); i++)
			((float *)conv)[i] = ((float)(i % 5u) - 2.0f) * 0.05f;
		for (i = 0u; i < sizeof(down_gate) / sizeof(float); i++)
			down_gate[i] = ((float)(i % 9u) - 4.0f) * 0.01f;
		for (i = 0u; i < sizeof(up) / sizeof(float); i++)
			((float *)up)[i] = ((float)(i % 5u) - 2.0f) * 0.02f;
		for (i = 0u; i < SPARK_GLM5_NEXT_VKDA_DIM; i++)
			dt[i] = -0.5f;
		for (i = 0u; i < SPARK_GLM5_NEXT_VKDA_HEADS; i++)
			alog[i] = 0.1f;
		for (i = 0u; i < SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION; i++)
			onorm[i] = 1.0f;
		for (i = 0u; i < sizeof(ow) / sizeof(float); i++)
			ow[i] = ((float)(i % 7u) - 3.0f) * 0.005f;
		memset(wq,0,sizeof(wq));
		memset(wk,0,sizeof(wk));
		memset(wv,0,sizeof(wv));
		memset(state,0,sizeof(state));
		SparkGlm5NextValKdaToken(collapsed,qkv_beta,conv[0],conv[1],conv[2],
			down_gate,up[0],up[1],dt,alog,onorm,ow,wq,wk,wv,state,output);
		for (j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
		{
			if (!(output[j] == output[j]) || fabsf(output[j]) > 1e30f)
				finite = 0;
			magnitude += fabsf(output[j]);
		}
		failures += SparkGlm5NextValSelftestAssert(finite,"kda oracle output finite");
		failures += SparkGlm5NextValSelftestAssert(magnitude > 0.0f,"kda oracle output non-zero");
	}
	/* 8. Execute the rope-0 MLA oracle for one token at context 3. */
	{
		static float collapsed[SPARK_GLM5_NEXT_VHIDDEN];
		static float q_a[SPARK_GLM5_NEXT_VQUERY_A * SPARK_GLM5_NEXT_VHIDDEN];
		static float q_a_norm[SPARK_GLM5_NEXT_VQUERY_A];
		static float q_b[(uint64_t)SPARK_GLM5_NEXT_VQ_B_ROWS * SPARK_GLM5_NEXT_VQUERY_A];
		static float kv_a[(uint64_t)SPARK_GLM5_NEXT_VKV_SLOT_ELEMENTS * SPARK_GLM5_NEXT_VHIDDEN];
		static float kv_a_norm[SPARK_GLM5_NEXT_VLATENT];
		static float kv_b_key[(uint64_t)SPARK_GLM5_NEXT_VHEADS * SPARK_GLM5_NEXT_VLATENT * SPARK_GLM5_NEXT_VQK_NOPE];
		static float kv_b_value[(uint64_t)SPARK_GLM5_NEXT_VHEADS * SPARK_GLM5_NEXT_VVALUE_DIM * SPARK_GLM5_NEXT_VLATENT];
		static float o_proj[(uint64_t)SPARK_GLM5_NEXT_VHIDDEN * SPARK_GLM5_NEXT_VATTN_COLS];
		static float latents[3u * SPARK_GLM5_NEXT_VLATENT];
		static float latent_out[SPARK_GLM5_NEXT_VLATENT];
		static float output[SPARK_GLM5_NEXT_VHIDDEN];
		uint64_t i;
		uint32_t j;
		int finite = 1;
		float magnitude = 0.0f;
		for (i = 0u; i < (uint64_t)SPARK_GLM5_NEXT_VHIDDEN; i++)
			collapsed[i] = ((float)(i % 13u) - 6.0f) * 0.015f;
		for (i = 0u; i < sizeof(q_a) / sizeof(float); i++)
			q_a[i] = ((float)(i % 7u) - 3.0f) * 0.01f;
		for (i = 0u; i < SPARK_GLM5_NEXT_VQUERY_A; i++)
			q_a_norm[i] = 1.0f;
		for (i = 0u; i < sizeof(q_b) / sizeof(float); i++)
			q_b[i] = ((float)(i % 5u) - 2.0f) * 0.01f;
		for (i = 0u; i < sizeof(kv_a) / sizeof(float); i++)
			kv_a[i] = ((float)(i % 9u) - 4.0f) * 0.01f;
		for (i = 0u; i < SPARK_GLM5_NEXT_VLATENT; i++)
			kv_a_norm[i] = 1.0f;
		for (i = 0u; i < sizeof(kv_b_key) / sizeof(float); i++)
			kv_b_key[i] = ((float)(i % 5u) - 2.0f) * 0.01f;
		for (i = 0u; i < sizeof(kv_b_value) / sizeof(float); i++)
			kv_b_value[i] = ((float)(i % 7u) - 3.0f) * 0.01f;
		for (i = 0u; i < sizeof(o_proj) / sizeof(float); i++)
			o_proj[i] = ((float)(i % 11u) - 5.0f) * 0.005f;
		for (i = 0u; i < sizeof(latents) / sizeof(float); i++)
			latents[i] = ((float)(i % 5u) - 2.0f) * 0.05f;
		SparkGlm5NextValMlaToken(collapsed,q_a,q_a_norm,q_b,kv_a,kv_a_norm,
			kv_b_key,kv_b_value,o_proj,latents,3u,latent_out,output);
		for (j = 0u; j < SPARK_GLM5_NEXT_VHIDDEN; j++)
		{
			if (!(output[j] == output[j]) || fabsf(output[j]) > 1e30f)
				finite = 0;
			magnitude += fabsf(output[j]);
		}
		failures += SparkGlm5NextValSelftestAssert(finite,"mla oracle output finite");
		failures += SparkGlm5NextValSelftestAssert(magnitude > 0.0f,"mla oracle output non-zero");
	}
	/* 9. Router: top-8 of 288, weights sum to the routed scale after
	 * renormalisation (up to the 1e-20 guard). */
	{
		static float router[(uint64_t)SPARK_GLM5_NEXT_VEXPERTS * SPARK_GLM5_NEXT_VHIDDEN];
		static float correction[SPARK_GLM5_NEXT_VEXPERTS];
		static float hidden[SPARK_GLM5_NEXT_VHIDDEN];
		uint32_t selected[SPARK_GLM5_NEXT_VTOP_K];
		float weights[SPARK_GLM5_NEXT_VTOP_K];
		uint64_t i;
		float total = 0.0f;
		int distinct = 1;
		for (i = 0u; i < sizeof(router) / sizeof(float); i++)
			router[i] = ((float)(i % 17u) - 8.0f) * 0.001f;
		for (i = 0u; i < SPARK_GLM5_NEXT_VEXPERTS; i++)
			correction[i] = ((float)(i % 3u)) * 0.25f;
		for (i = 0u; i < (uint64_t)SPARK_GLM5_NEXT_VHIDDEN; i++)
			hidden[i] = ((float)(i % 7u) - 3.0f) * 0.05f;
		SparkGlm5NextValRouter(router,correction,hidden,selected,weights);
		for (i = 0u; i < SPARK_GLM5_NEXT_VTOP_K; i++)
		{
			total += weights[i];
			for (uint64_t j = i + 1u; j < SPARK_GLM5_NEXT_VTOP_K; j++)
				if (selected[i] == selected[j])
					distinct = 0;
		}
		failures += SparkGlm5NextValSelftestAssert(distinct,"router selects 8 distinct experts");
		failures += SparkGlm5NextValSelftestAssert(fabsf(total - SPARK_GLM5_NEXT_VROUTED_SCALE) < 1e-3f,
			"router weights renormalise to the routed scale");
	}
	if (failures == 0)
		printf("glm5_next_validator_selftest PASS (bounded decay, expert-major "
		       "addressing, e4m3, mHC sinkhorn, kpool expand, fused sections)\n");
	return(failures);
}

#ifndef SPARK_GLM5_NEXT_VALIDATOR_ORACLE_SELFTEST

/* -- device fixture (tiers 1 and 2a) ---------------------------------------- */

typedef struct SparkGlm5NextValFixture
{
	SparkGlm5NextLayerWeights weights;
	SparkGlm5NextExecutionSlot slot;
	SparkGlm5NextCudaWave wave;
	cudaStream_t stream;
	uint32_t multiprocessors;
	SparkGlm5NextValMatrix attn_norm,mlp_norm;
	/* KDA tensors (the pack-V2 fused forms). */
	SparkGlm5NextValMatrix kda_qkv_beta,kda_decay_gate_down,kda_decay_up,kda_gate_up;
	SparkGlm5NextValMatrix kda_q_conv,kda_k_conv,kda_v_conv,kda_out_norm,kda_out;
	float *kda_dt_bias;      float kda_dt_bias_host[SPARK_GLM5_NEXT_VKDA_DIM];
	float *kda_a_log;        float kda_a_log_host[SPARK_GLM5_NEXT_VKDA_HEADS];
	/* MLA tensors. */
	SparkGlm5NextValMatrix q_a,q_a_norm,q_b,kv_a,kv_a_norm,kv_b_key,kv_b_value,attn_output;
	/* HC tensors (fn/base/scale, f32). */
	SparkGlm5NextValMatrix hc_attn_fn,hc_attn_base,hc_attn_scale,hc_ffn_fn,hc_ffn_base,hc_ffn_scale;
	/* MLP tensors. */
	SparkGlm5NextValMatrix dense_gate_up,dense_down,router,shared_gate_up,shared_down;
	float *router_correction; float router_correction_host[SPARK_GLM5_NEXT_VEXPERTS];
	/* Scratch and caches (8 rows of everything, 1 lane). */
	uint16_t *streams,*residual,*normed,*q_compressed,*q_bf16,*kv_slot;
	uint16_t *attention_latent,*attention_value,*attention_out;
	uint16_t *gate_up,*intermediate,*expert_out,*shared_out;
	uint16_t *fused_qkvb,*fused_decay_gate,*decay_latent,*gate_latent;
	uint16_t *kda_beta_logit,*kda_gate_bf16,*kda_decay_logit,*kda_output;
	uint16_t *hc_collapsed,*hc_snapshot,*hc_mean,*index_query,*index_key;
	uint16_t *index_gate,*index_packed;
	uint32_t *selected_pools,*selected_positions;
	float *hc_mixes,*hc_pre,*hc_post,*hc_comb;
	float *kda_retention,*kda_write_gate,*router_logits,*selection_scores,*route_weight;
	uint32_t *token_ids,*resident_slots,*positions,*context_lengths;
	uint32_t *dense_row_offset,*dense_tile_prefix,*page_table;
	uint32_t *route_expert,*route_packed_row,*route_source_token,*group_row_offset;
	uint32_t *group_tile_prefix_w1,*group_tile_prefix_w2;
	uint64_t *head_maxloc;
	uint8_t *kv_cache,*index_cache,*kda_state_pools,*kda_window_pools;
	uint32_t *kda_state_index;
	uint32_t host_page_table[SPARK_GLM5_NEXT_VALIDATION_PAGES];
	uint32_t host_kv_ordinals[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t host_index_ordinals[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t host_kda_ordinals[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t *kv_access_error;
} SparkGlm5NextValFixture;

static int SparkGlm5NextValFixtureBuild(SparkGlm5NextValFixture *fixture)
{
	uint32_t lane;
	memset(fixture,0,sizeof(*fixture));
	SparkGlm5NextValRandomState = 0x5eed1234u;
	if (SparkGlm5NextValAllocMatrix(&fixture->attn_norm,1u,SPARK_GLM5_NEXT_VHIDDEN,1,0.0f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->mlp_norm,1u,SPARK_GLM5_NEXT_VHIDDEN,1,0.0f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_qkv_beta,
			3u * SPARK_GLM5_NEXT_VKDA_DIM + SPARK_GLM5_NEXT_VKDA_HEADS,SPARK_GLM5_NEXT_VHIDDEN,0,0.01f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_decay_gate_down,
			2u * SPARK_GLM5_NEXT_VKDA_LOW_RANK,SPARK_GLM5_NEXT_VHIDDEN,0,0.01f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_decay_up,SPARK_GLM5_NEXT_VKDA_DIM,SPARK_GLM5_NEXT_VKDA_LOW_RANK,0,0.02f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_gate_up,SPARK_GLM5_NEXT_VKDA_DIM,SPARK_GLM5_NEXT_VKDA_LOW_RANK,0,0.02f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_q_conv,SPARK_GLM5_NEXT_VKDA_DIM,SPARK_GLM5_NEXT_VKDA_CONV,0,0.05f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_k_conv,SPARK_GLM5_NEXT_VKDA_DIM,SPARK_GLM5_NEXT_VKDA_CONV,0,0.05f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_v_conv,SPARK_GLM5_NEXT_VKDA_DIM,SPARK_GLM5_NEXT_VKDA_CONV,0,0.05f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_out_norm,1u,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION,1,0.0f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kda_out,SPARK_GLM5_NEXT_VHIDDEN,SPARK_GLM5_NEXT_VKDA_DIM,0,0.005f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->q_a,SPARK_GLM5_NEXT_VQUERY_A,SPARK_GLM5_NEXT_VHIDDEN,0,0.02f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->q_a_norm,1u,SPARK_GLM5_NEXT_VQUERY_A,1,0.0f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->q_b,SPARK_GLM5_NEXT_VQ_B_ROWS,SPARK_GLM5_NEXT_VQUERY_A,0,0.02f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kv_a,SPARK_GLM5_NEXT_VKV_SLOT_ELEMENTS,SPARK_GLM5_NEXT_VHIDDEN,0,0.02f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kv_a_norm,1u,SPARK_GLM5_NEXT_VLATENT,1,0.0f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kv_b_key,
			(uint64_t)SPARK_GLM5_NEXT_VHEADS * SPARK_GLM5_NEXT_VLATENT,SPARK_GLM5_NEXT_VQK_NOPE,0,0.02f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->kv_b_value,
			(uint64_t)SPARK_GLM5_NEXT_VHEADS * SPARK_GLM5_NEXT_VVALUE_DIM,SPARK_GLM5_NEXT_VLATENT,0,0.02f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->attn_output,SPARK_GLM5_NEXT_VHIDDEN,SPARK_GLM5_NEXT_VATTN_COLS,0,0.01f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->hc_attn_fn,SPARK_GLM5_NEXT_VHC_MIX,SPARK_GLM5_NEXT_VHC_FLAT,0,0.01f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->hc_attn_base,1u,SPARK_GLM5_NEXT_VHC_MIX,0,0.25f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->hc_attn_scale,1u,3u,0,0.5f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->hc_ffn_fn,SPARK_GLM5_NEXT_VHC_MIX,SPARK_GLM5_NEXT_VHC_FLAT,0,0.01f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->hc_ffn_base,1u,SPARK_GLM5_NEXT_VHC_MIX,0,0.25f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->hc_ffn_scale,1u,3u,0,0.5f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->dense_gate_up,SPARK_GLM5_NEXT_VGATE_UP_ROWS,SPARK_GLM5_NEXT_VHIDDEN,0,0.01f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->dense_down,SPARK_GLM5_NEXT_VHIDDEN,SPARK_GLM5_NEXT_VDENSE_INTER,0,0.01f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->router,SPARK_GLM5_NEXT_VEXPERTS,SPARK_GLM5_NEXT_VHIDDEN,0,0.002f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->shared_gate_up,SPARK_GLM5_NEXT_VW1_ROWS,SPARK_GLM5_NEXT_VHIDDEN,0,0.005f) != 0 ||
		SparkGlm5NextValAllocMatrix(&fixture->shared_down,SPARK_GLM5_NEXT_VW2_ROWS,SPARK_GLM5_NEXT_VW2_COLUMNS,0,0.005f) != 0)
		return(1);
	return(0);
}

/* The GPU tier drivers are the mid-pipeline publish step: the fixture
 * above is the working scaffold; the walk drivers land with it. This
 * binary refuses to claim a tier PASS until they do. */
int main(int argc,char **argv)
{
	if (argc != 2)
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	printf("glm5_next validator: configuration %s\n",argv[1]);
	if (SparkGlm5NextValOracleSelftest() != 0)
		return(1);
	printf("glm5_next validator: FAIL - GPU tier drivers not yet wired; the "
	       "publish gate must not pass on this build\n");
	return(1);
}
#else
int main(void)
{
	return(SparkGlm5NextValOracleSelftest() == 0 ? 0 : 1);
}
#endif
