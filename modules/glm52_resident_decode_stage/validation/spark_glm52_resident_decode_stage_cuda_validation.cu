#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "spark_glm52_resident_decode_stage_internal.h"

/*
 * GLM 5.2 resident decode stage, hardware validation (sm_121a).
 *
 * A retained-receipt numerical gate driven entirely through the module's own
 * exported launchers (SparkGlm52LaunchCudaWaveBegin / LayerAttention /
 * LayerMlp) exactly as SparkGlm52TpChainAdvance drives them, compared against
 * an fp32 CPU oracle that restates the shared-kernel formulas here,
 * self-contained under nvcc and sharing no code with what it validates.
 *
 * Tier 1 - the dense layer: ONE dense layer's forward (attention chunk plus
 * dense MLP chunk) over a four-token causal walk, plus a bit-exact
 * determinism re-walk. This is the original validator body, unchanged.
 *
 * Tier 2a - the routed-expert layer: the same attention chunk feeding
 * Glm52LayerMoe at the first routed layer, covering the router GEMM (fp32
 * out), sigmoid top-k selection with the correction bias selecting but not
 * weighing, the renormalised mixture, eight routed-expert forwards through
 * the compiled package codec (dequant restated losslessly - the fixture
 * writes only exactly representable codes, so quantisation contributes zero
 * error and the comparison stays at the dense tier's thresholds), the
 * weighted finalize, and the shared expert. The routed selection is checked
 * as a set against the oracle, the mixture weights numerically, one packed
 * expert forward element-wise, and the whole leg re-run for bit-exact
 * determinism (value-equivalent under route-scatter races: every packed row
 * is the same pure function of its (token, expert) pair).
 *
 * Tier 2b - the DSA/indexer path at context > DSA_SELECTED: the fixture
 * pre-populates the index-key cache with synthetic POST-rope keys shaped so
 * the top-2048 scores sit in one radix bucket strictly above a low tail,
 * which makes the histogram/gather selection membership deterministic (the
 * kernel's own contract: ties within a bucket are indistinguishable at radix
 * resolution, so the fixture removes the ambiguity instead of assuming it).
 * Three positions carry real main-KV latents written by earlier production
 * steps so the selected-position attention is non-degenerate. Checks: the
 * selected set equals the oracle's prediction, attention output matches the
 * oracle attending the same set, and a re-run reproduces the selection set
 * exactly and the output within tight tolerance (summation-order noise only;
 * bit-exactness is deliberately not claimed for this leg - the gather's
 * emission order among equal-radix candidates is unsynchronized by design).
 *
 * Every comparison prints its numbers; the thresholds are the guard, the
 * numbers are the evidence. Not executable on this tree (no nvcc/sm_121a
 * here): the pure oracle/codec/selection math below is nonetheless EXECUTED
 * on every host by tests/test_glm52_cuda_validator_tier2_oracle.py through
 * the SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST entry at the bottom of this
 * file, so the fixture shaping and oracle formulas ship runtime-tested.
 */

#define SPARK_GLM52_VALIDATION_TOKENS 4u
#define SPARK_GLM52_VALIDATION_LANES 1u
/* Pages cover the DSA tier's context; the dense tiers use the first ones. */
#define SPARK_GLM52_VALIDATION_EMBED_ROWS 8u

#define SPARK_GLM52_VHIDDEN SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM52_VQUERY_A SPARK_GLM52_MODEL_QUERY_A_DIMENSION
#define SPARK_GLM52_VQK_NOPE SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION
#define SPARK_GLM52_VROPE SPARK_GLM52_MODEL_ROPE_DIMENSION
#define SPARK_GLM52_VLATENT SPARK_GLM52_MODEL_LATENT_DIMENSION
#define SPARK_GLM52_VHEADS SPARK_GLM52_MODEL_HEAD_COUNT
#define SPARK_GLM52_VVALUE_DIM SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION
#define SPARK_GLM52_VQ_ROWS (SPARK_GLM52_VHEADS * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE))
#define SPARK_GLM52_VATTN_COLS (SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM)
#define SPARK_GLM52_VDENSE_INTER SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define SPARK_GLM52_VGATE_UP_ROWS (2u * SPARK_GLM52_VDENSE_INTER)
/* Glm52Kv geometry (source/cuda/config.h + layer.cuh): block-major pool. */
#define SPARK_GLM52_VKV_SLOT_ELEMENTS (SPARK_GLM52_VLATENT + SPARK_GLM52_VROPE)
#define SPARK_GLM52_VKV_SLOT_BYTES (SPARK_GLM52_VKV_SLOT_ELEMENTS * 2u)
#define SPARK_GLM52_VKV_PAGE_SLOTS 64u
#define SPARK_GLM52_VKV_LAYER_BYTES (SPARK_GLM52_VKV_SLOT_BYTES * SPARK_GLM52_VKV_PAGE_SLOTS)
#define SPARK_GLM52_VKV_PAGE_BYTES (SPARK_GLM52_VKV_LAYER_BYTES * SPARK_GLM52_MODEL_LAYER_COUNT)
#define SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS 6u

/* -- tier 2a: routed experts ------------------------------------------------ */
#define SPARK_GLM52_VTOP_K SPARK_GLM52_MODEL_MOE_TOP_K
#define SPARK_GLM52_VEXPERT_INTER SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION
#define SPARK_GLM52_VW1_ROWS (2u * SPARK_GLM52_VEXPERT_INTER)
#define SPARK_GLM52_VEXPERT_COLUMNS SPARK_GLM52_VHIDDEN
#define SPARK_GLM52_VW2_ROWS SPARK_GLM52_VHIDDEN
#define SPARK_GLM52_VW2_COLUMNS SPARK_GLM52_VEXPERT_INTER
/* Slabs exist only for the experts the pinned bias can select; routing to
 * anything else is a checked failure before any numeric comparison runs. */
#define SPARK_GLM52_VALIDATION_EXPERT_SLOTS SPARK_GLM52_VTOP_K
#define SPARK_GLM52_VALIDATION_ROUTES 4u

/* -- tier 2b: DSA/indexer --------------------------------------------------- */
#define SPARK_GLM52_VALIDATION_DSA_POSITION 2064u
#define SPARK_GLM52_VALIDATION_DSA_CONTEXT (SPARK_GLM52_VALIDATION_DSA_POSITION + 1u)
#define SPARK_GLM52_VALIDATION_PAGES \
	((SPARK_GLM52_VALIDATION_DSA_POSITION + SPARK_GLM52_VKV_PAGE_SLOTS) / SPARK_GLM52_VKV_PAGE_SLOTS)
#define SPARK_GLM52_VDSA_HEADS SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT
#define SPARK_GLM52_VDSA_DIM SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION
#define SPARK_GLM52_VDSA_QUERY_DIM (SPARK_GLM52_VDSA_HEADS * SPARK_GLM52_VDSA_DIM)
#define SPARK_GLM52_VDSA_SELECTED SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT
#define SPARK_GLM52_VINDEX_SLOT_BYTES ((SPARK_GLM52_VDSA_DIM * 2u))
#define SPARK_GLM52_VINDEX_PAGE_BYTES (SPARK_GLM52_VINDEX_SLOT_BYTES * SPARK_GLM52_VKV_PAGE_SLOTS)
/* Main-cache positions carrying real latents in the DSA tier. */
#define SPARK_GLM52_VALIDATION_REAL_SLOTS 3u

extern "C" int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count);

/* -- fixtures -------------------------------------------------------------- */

static uint32_t SparkGlm52ValRandomState;

static uint32_t SparkGlm52ValNext(void)
{
	SparkGlm52ValRandomState = SparkGlm52ValRandomState * 1664525u + 1013904223u;
	return(SparkGlm52ValRandomState >> 8u);
}

/* Round to nearest even: the LmFloatToBf16 contract (dtype.cuh). */
static uint16_t SparkGlm52ValBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return((uint16_t)(bits >> 16u));
}

static float SparkGlm52ValFromBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float converted;
	memcpy(&converted,&bits,sizeof(converted));
	return(converted);
}

static void SparkGlm52ValFill(uint16_t *packed,float *exact,uint64_t count,float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		packed[index] = SparkGlm52ValBf16((((float)(int32_t)(SparkGlm52ValNext() & 0xffffu) - 32768.0f) * scale) / 32768.0f);
		exact[index] = SparkGlm52ValFromBf16(packed[index]);
	}
}

static void SparkGlm52ValFillNorm(uint16_t *packed,float *exact,uint64_t count)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		packed[index] = SparkGlm52ValBf16(1.0f + (((float)(int32_t)(SparkGlm52ValNext() & 0xffu) - 128.0f) * 0.001f));
		exact[index] = SparkGlm52ValFromBf16(packed[index]);
	}
}

static int SparkGlm52ValFail(const char *check,const char *detail)
{
	fprintf(stderr,"glm52_validation failure=%s detail=%s\n",check,detail);
	return(1);
}

typedef struct SparkGlm52ValMetrics
{
	double difference_l2;
	double reference_l2;
	double actual_l2;
	double dot;
	double maximum_absolute;
	uint64_t count;
} SparkGlm52ValMetrics;

static void SparkGlm52ValMeasure(SparkGlm52ValMetrics *metrics,const float *actual,const float *reference,uint64_t count)
{
	uint64_t index;
	double difference;
	memset(metrics,0,sizeof(*metrics));
	metrics->count = count;
	for (index = 0u; index < count; index++)
	{
		difference = (double)actual[index] - (double)reference[index];
		metrics->difference_l2 += difference * difference;
		metrics->reference_l2 += (double)reference[index] * (double)reference[index];
		metrics->actual_l2 += (double)actual[index] * (double)actual[index];
		metrics->dot += (double)actual[index] * (double)reference[index];
		if ( fabs(difference) > metrics->maximum_absolute )
			metrics->maximum_absolute = fabs(difference);
	}
}

static int SparkGlm52ValReport(const char *check,const SparkGlm52ValMetrics *metrics,double max_relative_l2,double minimum_cosine)
{
	double relative_l2 = metrics->reference_l2 > 0.0
		? sqrt(metrics->difference_l2 / metrics->reference_l2) : INFINITY;
	double cosine = metrics->actual_l2 > 0.0 && metrics->reference_l2 > 0.0
		? metrics->dot / sqrt(metrics->actual_l2 * metrics->reference_l2) : 0.0;
	printf("glm52_validation check=%s elements=%llu relative_l2=%.9g cosine=%.9g max_abs=%.9g\n",
		check,(unsigned long long)metrics->count,relative_l2,cosine,metrics->maximum_absolute);
	if ( isfinite(relative_l2) == 0 || relative_l2 > max_relative_l2 )
		return(SparkGlm52ValFail(check,"relative_l2"));
	if ( cosine < minimum_cosine )
		return(SparkGlm52ValFail(check,"cosine"));
	return(0);
}

/* -- package expert codec, host mirror -------------------------------------- */
/*
 * The fixture writes only codes whose decode is EXACT (integer codes times a
 * power-of-two scale, or native-grid values), so the oracle's forward of the
 * pre-encode values equals the device's forward of the decoded weights up to
 * accumulation order. Dequant applies the bf16 rounding the format's
 * Fragment() performs (LmPackBf16Pair) - with these grids the product is
 * already representable, and the rounding is applied anyway so the mirror
 * cannot drift from the kernel if the grids ever widen.
 */

#define SPARK_GLM52_VAL_CODEC_INT6 2u
#define SPARK_GLM52_VAL_CODEC_INT7 3u
#define SPARK_GLM52_VAL_CODEC_INT8 4u
#define SPARK_GLM52_VAL_CODEC_FP8 5u
#define SPARK_GLM52_VAL_CODEC_NVFP4 6u
#define SPARK_GLM52_VAL_CODEC_MXFP4 7u

#if GLM52_EXPERT_WEIGHT_CODEC == 2
#define SPARK_GLM52_VAL_CODEC SPARK_GLM52_VAL_CODEC_INT6
#elif GLM52_EXPERT_WEIGHT_CODEC == 3
#define SPARK_GLM52_VAL_CODEC SPARK_GLM52_VAL_CODEC_INT7
#elif GLM52_EXPERT_WEIGHT_CODEC == 4
#define SPARK_GLM52_VAL_CODEC SPARK_GLM52_VAL_CODEC_INT8
#elif GLM52_EXPERT_WEIGHT_CODEC == 5
#define SPARK_GLM52_VAL_CODEC SPARK_GLM52_VAL_CODEC_FP8
#elif GLM52_EXPERT_WEIGHT_CODEC == 6
#define SPARK_GLM52_VAL_CODEC SPARK_GLM52_VAL_CODEC_NVFP4
#elif GLM52_EXPERT_WEIGHT_CODEC == 7
#define SPARK_GLM52_VAL_CODEC SPARK_GLM52_VAL_CODEC_MXFP4
#else
#error "unsupported GLM52_EXPERT_WEIGHT_CODEC for the validator oracle"
#endif

static uint32_t SparkGlm52ValCodecStoredBits(uint32_t codec)
{
	if ( codec == SPARK_GLM52_VAL_CODEC_INT6 )
		return(6u);
	if ( codec == SPARK_GLM52_VAL_CODEC_INT7 )
		return(7u);
	if ( codec == SPARK_GLM52_VAL_CODEC_FP8 || codec == SPARK_GLM52_VAL_CODEC_INT8 )
		return(8u);
	return(4u);
}

/* LmInt6 kScaleGroup=32, Int7/Int8/Fp8 128, Nvfp4 16 (LM_MMA4_NVFP4_GROUP),
 * Mxfp4 32. Mirrors the format headers; a drift breaks the scale addressing
 * the oracle shares with LmScaleTensorBuild. */
static uint32_t SparkGlm52ValCodecScaleGroup(uint32_t codec)
{
	if ( codec == SPARK_GLM52_VAL_CODEC_INT6 )
		return(32u);
	if ( codec == SPARK_GLM52_VAL_CODEC_NVFP4 )
		return(16u);
	if ( codec == SPARK_GLM52_VAL_CODEC_MXFP4 )
		return(32u);
	return(128u);
}

static uint32_t SparkGlm52ValCodecUsesSignedIntGrid(uint32_t codec)
{
	return(codec == SPARK_GLM52_VAL_CODEC_INT6 || codec == SPARK_GLM52_VAL_CODEC_INT7 ||
		codec == SPARK_GLM52_VAL_CODEC_INT8);
}

/* E4M3 decode, host form of cvt.rn.f16x2.e4m3x2 for every finite code. */
static float SparkGlm52ValE4m3Decode(uint8_t code)
{
	uint32_t sign = (uint32_t)(code >> 7u) & 1u;
	uint32_t biased = (uint32_t)(code >> 3u) & 15u;
	uint32_t mantissa = (uint32_t)code & 7u;
	float value;
	if ( biased == 0u )
		value = ldexpf((float)mantissa,-9);
	else if ( biased == 15u && mantissa == 7u )
		value = NAN;
	else
		value = ldexpf(1.0f + ((float)mantissa / 8.0f),(int32_t)biased - 7);
	return(sign != 0u ? -value : value);
}

/* UE4M3 scale decode, dtype.cuh LmUe4m3ToFloat. */
static float SparkGlm52ValUe4m3Decode(uint8_t code)
{
	uint32_t biased = (uint32_t)(code >> 3u) & 15u;
	uint32_t mantissa = (uint32_t)code & 7u;
	if ( biased == 0u )
		return(ldexpf((float)mantissa,-9));
	if ( biased == 15u && mantissa == 7u )
		return(448.0f);
	return(ldexpf(1.0f + ((float)mantissa / 8.0f),(int32_t)biased - 7));
}

/* UE8M0 scale decode: 2^(code-127), 0xff is the NaN encoding. */
static float SparkGlm52ValUe8m0Decode(uint8_t code)
{
	if ( code == 0xffu )
		return(NAN);
	return(ldexpf(1.0f,(int32_t)code - 127));
}

/* E2M1 nibble: magnitudes {0,.5,1,1.5,2,3,4,6}, bit 3 is the sign. */
static float SparkGlm52ValE2m1Decode(uint8_t nibble)
{
	static const float magnitudes[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	float value = magnitudes[nibble & 7u];
	return((nibble & 8u) != 0u ? -value : value);
}

#ifdef SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST
/* Signed integer grid bounds: LmInt6/7/8 kMax with the symmetric floor the
 * Encode clamps to. Referenced only by the selftest's round trip. */
static int32_t SparkGlm52ValCodecCodeMinimum(uint32_t codec)
{
	if ( codec == SPARK_GLM52_VAL_CODEC_INT6 )
		return(-32);
	if ( codec == SPARK_GLM52_VAL_CODEC_INT7 )
		return(-64);
	return(-128);
}

static int32_t SparkGlm52ValCodecCodeMaximum(uint32_t codec)
{
	if ( codec == SPARK_GLM52_VAL_CODEC_INT6 )
		return(31);
	if ( codec == SPARK_GLM52_VAL_CODEC_INT7 )
		return(63);
	return(127);
}
#endif

/* Scale plane geometry, LmWeightCodecScaleTensor + LmScaleTensorBuild with
 * row_group_size = 1: per (expert, row, column block). NVFP4 prefixes the
 * buffer with one f32 global per expert (BlockUe4m3F32Global). */
static uint64_t SparkGlm52ValScaleBlocksPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	uint64_t groups = ((uint64_t)columns + SparkGlm52ValCodecScaleGroup(codec) - 1u) /
		SparkGlm52ValCodecScaleGroup(codec);
	return((uint64_t)rows * groups);
}

static uint64_t SparkGlm52ValScaleBytesPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	if ( codec == SPARK_GLM52_VAL_CODEC_NVFP4 )
		return(SparkGlm52ValScaleBlocksPerExpert(codec,rows,columns));
	if ( codec == SPARK_GLM52_VAL_CODEC_MXFP4 )
		return(SparkGlm52ValScaleBlocksPerExpert(codec,rows,columns));
	return(SparkGlm52ValScaleBlocksPerExpert(codec,rows,columns) * sizeof(float));
}

/* NVFP4 carries the per-expert f32 globals ahead of every expert's blocks. */
static uint64_t SparkGlm52ValScaleGlobalsBytes(uint32_t codec,uint32_t expert_count)
{
	return(codec == SPARK_GLM52_VAL_CODEC_NVFP4 ? (uint64_t)expert_count * sizeof(float) : 0u);
}

static uint64_t SparkGlm52ValScaleBufferBytes(uint32_t codec,uint32_t expert_count,uint32_t rows,uint32_t columns)
{
	return(SparkGlm52ValScaleGlobalsBytes(codec,expert_count) +
		(uint64_t)expert_count * SparkGlm52ValScaleBytesPerExpert(codec,rows,columns));
}

static uint64_t SparkGlm52ValScaleExpertOffset(uint32_t codec,uint32_t expert_count,uint32_t expert,uint32_t rows,uint32_t columns)
{
	return(SparkGlm52ValScaleGlobalsBytes(codec,expert_count) +
		(uint64_t)expert * SparkGlm52ValScaleBytesPerExpert(codec,rows,columns));
}

static uint64_t SparkGlm52ValPayloadRowBytes(uint32_t codec,uint32_t columns)
{
	return(((uint64_t)columns * SparkGlm52ValCodecStoredBits(codec) + 7u) / 8u);
}

static uint64_t SparkGlm52ValPayloadBytesPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	return((uint64_t)rows * SparkGlm52ValPayloadRowBytes(codec,columns));
}

/* Read one stored code at (row, column) of a payload slab. Bit packing is
 * LSB-first within each row for the sub-byte integer formats (LmInt6Raw /
 * LmInt7Raw); 4-bit packs even columns in the low nibble (LmNvfp4Pair). */
static int32_t SparkGlm52ValReadCode(const uint8_t *slab,uint32_t codec,uint32_t row,uint32_t columns,uint32_t column)
{
	const uint8_t *row_base = slab + (uint64_t)row * SparkGlm52ValPayloadRowBytes(codec,columns);
	uint32_t bits = SparkGlm52ValCodecStoredBits(codec);
	uint64_t bit = (uint64_t)column * bits;
	uint32_t byte = (uint32_t)(bit >> 3u);
	uint32_t shift = (uint32_t)(bit & 7u);
	uint32_t word = row_base[byte];
	int32_t code;
	if ( byte + 1u < SparkGlm52ValPayloadRowBytes(codec,columns) )
		word |= (uint32_t)row_base[byte + 1u] << 8u;
	code = (int32_t)((word >> shift) & ((1u << bits) - 1u));
	if ( bits == 4u )
		return(code); /* nibble handled by the caller through Decode */
	if ( codec == SPARK_GLM52_VAL_CODEC_INT6 )
		return(((int32_t)(code << 26)) >> 26);
	if ( codec == SPARK_GLM52_VAL_CODEC_INT7 )
		return(((int32_t)(code << 25)) >> 25);
	return((int8_t)code);
}

/* Per-(expert, row, column-block) scale index, LmScaleTensorIndex with
 * row_group_size = 1 and group-major slabs. */
static uint64_t SparkGlm52ValScaleIndex(uint32_t codec,uint32_t rows,uint32_t columns,uint32_t expert,uint32_t row,uint32_t column)
{
	uint64_t groups = ((uint64_t)columns + SparkGlm52ValCodecScaleGroup(codec) - 1u) /
		SparkGlm52ValCodecScaleGroup(codec);
	return((uint64_t)expert * ((uint64_t)rows * groups) + (uint64_t)row * groups +
		(uint64_t)(column / SparkGlm52ValCodecScaleGroup(codec)));
}

/* Effective bf16 weight the MMA consumes for (expert, row, column):
 * decode x scale x (bf16 rounding from LmPackBf16Pair). NVFP4 multiplies its
 * per-expert f32 global into the block scale (LmScaleTensorLoad). */
static float SparkGlm52ValDequantWeight(const uint8_t *payload,const uint8_t *scales,uint32_t codec,uint32_t expert_count,
	uint32_t expert,uint32_t row,uint32_t rows,uint32_t columns,uint32_t column)
{
	uint64_t scale_index = SparkGlm52ValScaleIndex(codec,rows,columns,expert,row,column);
	float raw,scale;
	if ( codec == SPARK_GLM52_VAL_CODEC_NVFP4 )
	{
		const uint8_t *blocks = scales + SparkGlm52ValScaleGlobalsBytes(codec,expert_count);
		scale = SparkGlm52ValUe4m3Decode(blocks[scale_index]) * ((const float *)scales)[expert];
		raw = SparkGlm52ValE2m1Decode((uint8_t)SparkGlm52ValReadCode(payload,codec,row,columns,column));
	}
	else if ( codec == SPARK_GLM52_VAL_CODEC_MXFP4 )
	{
		scale = SparkGlm52ValUe8m0Decode(scales[scale_index]);
		raw = SparkGlm52ValE2m1Decode((uint8_t)SparkGlm52ValReadCode(payload,codec,row,columns,column));
	}
	else
	{
		scale = ((const float *)scales)[scale_index];
		if ( codec == SPARK_GLM52_VAL_CODEC_FP8 )
			raw = SparkGlm52ValE4m3Decode((uint8_t)SparkGlm52ValReadCode(payload,codec,row,columns,column));
		else
			raw = (float)SparkGlm52ValReadCode(payload,codec,row,columns,column);
	}
	return(SparkGlm52ValFromBf16(SparkGlm52ValBf16(raw * scale)));
}

/* -- top-k selection, host mirror of LmTopkSmallKernel ----------------------- */

/* Monotone unsigned image of a float (LmTopkKey): ordering by the key orders
 * by the value, for negatives included. */
static uint32_t SparkGlm52ValTopkKey(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return(bits ^ ((bits >> 31u) != 0u ? 0xffffffffu : 0x80000000u));
}

static uint32_t SparkGlm52ValTopkBucket(float value)
{
	return(SparkGlm52ValTopkKey(value) >> 24u);
}

/* Sigmoid selection scores; the device uses __expf, the oracle expf - the
 * fixture shapes selections with margins many orders above that difference. */
static float SparkGlm52ValSigmoid(float value)
{
	return(1.0f / (1.0f + expf(-value)));
}

/* Reference top-k: descending key, ties by lower index. Returns the number
 * filled (always k when n >= k). The bitonic network's behaviour inside a
 * tie class differs from this rule; the fixture shapes every decision away
 * from tie classes, and the oracle asserts the margins that guarantee it. */
static void SparkGlm52ValReferenceTopk(const float *scores,const float *bias,uint32_t n,uint32_t k,
	uint32_t *out_indices,float *out_weights)
{
	uint32_t selected,index,slot;
	float transformed[512];
	uint32_t taken[512];
	for (index = 0u; index < n && index < 512u; index++)
		transformed[index] = SparkGlm52ValSigmoid(scores[index]) +
			(bias != 0 ? bias[index] : 0.0f);
	for (slot = 0u; slot < k; slot++)
	{
		uint32_t best = n;
		for (index = 0u; index < n && index < 512u; index++)
		{
			uint32_t candidate;
			for (candidate = 0u; candidate < slot; candidate++)
				if ( taken[candidate] == index )
					break;
			if ( candidate != slot )
				continue;
			if ( best == n || SparkGlm52ValTopkKey(transformed[index]) > SparkGlm52ValTopkKey(transformed[best]) )
				best = index;
		}
		taken[slot] = best;
		out_indices[slot] = best;
		if ( out_weights != 0 )
			out_weights[slot] = SparkGlm52ValSigmoid(scores[best]);
	}
	selected = k;
	if ( out_weights != 0 && selected != 0u )
	{
		float total = 0.0f;
		for (slot = 0u; slot < selected; slot++)
			total += out_weights[slot];
		total += 1.0e-20f;
		for (slot = 0u; slot < selected; slot++)
			out_weights[slot] = (out_weights[slot] / total) * SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR;
	}
}

/* -- DSA shaping ------------------------------------------------------------- */
/*
 * Synthetic post-rope index keys for cached positions. Given the row's
 * index-query heads q[h][d] and learned head mixture w[h] (both produced by
 * the production chain the oracle restates), the score of cached position c
 * is qk_scale * sum_h w[h] * dot(q[h], k[c]) = qk_scale * dot(centroid, k[c])
 * with centroid = sum_h w[h] q[h]. Keys are built parallel to the centroid
 * so scores are positive, ranked, and radially separated: the top
 * DSA_SELECTED land inside one radix bucket, the tail far below it, which
 * makes LmTopkHistogram/Gather membership deterministic (its documented
 * contract leaves within-bucket-after-threshold choice open).
 */

typedef struct SparkGlm52ValDsaShape
{
	uint32_t selected_positions[SPARK_GLM52_VDSA_SELECTED];
	float top_minimum;
	float tail_maximum;
	uint32_t top_bucket;
	uint32_t separable;
} SparkGlm52ValDsaShape;

static float SparkGlm52ValDot(const float *a,const float *b,uint32_t count)
{
	float total = 0.0f;
	uint32_t index;
	for (index = 0u; index < count; index++)
		total += a[index] * b[index];
	return(total);
}

/* Fill keys[position][:dimension] so its predicted score lands at target.
 * key = (target / |centroid|_2^2) * centroid keeps dot(centroid,key)=target
 * exactly in exact arithmetic; bf16 storage perturbs it by ~1e-3 relative. */
static void SparkGlm52ValShapeKey(uint16_t *key,const float *centroid,float target,float centroid_square_sum,uint32_t dimension)
{
	float alpha = target / centroid_square_sum;
	uint32_t index;
	for (index = 0u; index < dimension; index++)
		key[index] = SparkGlm52ValBf16(centroid[index] * alpha);
}

static float SparkGlm52ValShapeScore(const uint16_t *key,const float *centroid,uint32_t dimension)
{
	float wide[SPARK_GLM52_VDSA_DIM];
	uint32_t index;
	for (index = 0u; index < dimension; index++)
		wide[index] = SparkGlm52ValFromBf16(key[index]);
	return(SparkGlm52ValDot(wide,centroid,dimension));
}

/* Build the whole shaped cache: ranks 0..selected-1 at positions
 * 0..selected-1 with scores marching down from top_target, tail positions
 * below floor_target. Verifies - from the ACTUAL bf16-stored keys - that
 * every top score sits in one radix bucket strictly above every tail score
 * and that the top bucket holds exactly the selected count, i.e. the
 * hardware selection is forced. Returns 0 when separable, 1 otherwise. */
static int SparkGlm52ValShapeIndexCache(uint16_t *keys,uint32_t position_count,uint32_t selected,const float *centroid,
	uint32_t dimension,float top_target,float floor_target,SparkGlm52ValDsaShape *shape)
{
	float centroid_square_sum = SparkGlm52ValDot(centroid,centroid,dimension);
	uint32_t position,index;
	memset(shape,0,sizeof(*shape));
	if ( centroid_square_sum <= 0.0f || position_count < selected )
		return(1);
	for (position = 0u; position < position_count; position++)
	{
		uint16_t *key = keys + (uint64_t)position * dimension;
		float target = position < selected
			? top_target - (float)position * ((top_target - floor_target) * 0.0001f)
			: floor_target * 0.001f * (float)(position_count - position);
		if ( target <= 0.0f )
			target = 1.0e-3f;
		SparkGlm52ValShapeKey(key,centroid,target,centroid_square_sum,dimension);
	}
	for (position = 0u; position < position_count; position++)
	{
		float score = SparkGlm52ValShapeScore(keys + (uint64_t)position * dimension,centroid,dimension) *
			SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE / sqrtf((float)SPARK_GLM52_VDSA_HEADS);
		uint32_t bucket = SparkGlm52ValTopkBucket(score);
		if ( position == 0u )
		{
			shape->top_bucket = bucket;
			shape->top_minimum = score;
		}
		if ( position < selected )
		{
			shape->selected_positions[position] = position;
			if ( score < shape->top_minimum )
				shape->top_minimum = score;
			if ( bucket != shape->top_bucket )
				return(1);
		}
		else
		{
			if ( position == selected )
				shape->tail_maximum = score;
			if ( score > shape->tail_maximum )
				shape->tail_maximum = score;
			if ( bucket >= shape->top_bucket )
				return(1);
		}
	}
	shape->separable = 1u;
	for (index = 0u; index < selected; index++)
		shape->selected_positions[index] = index;
	return(0);
}

/* -- device fixture --------------------------------------------------------- */

typedef struct SparkGlm52ValMatrix
{
	uint16_t *device;
	float *host;            /* exact fp32 upcast, rows x columns row-major */
	uint32_t rows;
	uint32_t columns;
} SparkGlm52ValMatrix;

typedef struct SparkGlm52ValExpertTensors
{
	uint8_t *w1_payload;    /* host master: expert-major payload slabs */
	uint8_t *w1_scales;     /* host master: scale planes (nvfp4 globals first) */
	uint8_t *w2_payload;
	uint8_t *w2_scales;
	uint8_t *w1_device_payload;
	uint8_t *w1_device_scale;
	uint8_t *w2_device_payload;
	uint8_t *w2_device_scale;
} SparkGlm52ValExpertTensors;

typedef struct SparkGlm52ValFixture
{
	SparkGlm52LayerWeights weights;
	SparkGlm52ExecutionSlot slot;
	SparkGlm52CudaWave wave;
	cudaStream_t stream;
	uint32_t multiprocessors;
	SparkGlm52ValMatrix embedding;
	SparkGlm52ValMatrix attn_norm,q_a,q_a_norm,q_b,kv_a,kv_a_norm;
	SparkGlm52ValMatrix kv_b_key,kv_b_value,attn_output;
	SparkGlm52ValMatrix post_attn_norm,dense_gate_up,dense_down;
	SparkGlm52ValMatrix router,shared_gate_up,shared_down;
	/* DSA indexer projections: host-known so the fp32 oracle (and the cache
	 * shaping built on it) can state the production chain's exact inputs. */
	SparkGlm52ValMatrix index_q,index_k,index_head,index_norm_weight,index_norm_bias;
	SparkGlm52ValExpertTensors experts;
	uint16_t *hidden,*residual,*normed,*q_compressed,*q_bf16,*query_rope;
	uint16_t *query_latent,*kv_slot,*attention_latent,*attention_value,*attention_out;
	uint16_t *gate_up,*intermediate,*expert_out,*shared_out;
	uint16_t *scratch_small;
	float *float_scratch;
	uint32_t *token_ids,*resident_slots,*positions,*context_lengths;
	uint32_t *dense_row_offset,*dense_tile_prefix;
	uint32_t *page_table,*route_scratch;
	uint32_t *route_expert,*route_packed_row,*route_source_token,*group_row_offset;
	uint32_t *group_tile_prefix_w1,*group_tile_prefix_w2,*selected_positions;
	float *router_logits,*selection_scores,*route_weight,*correction_bias;
	float correction_host[256];
	uint64_t *head_maxloc;
	uint8_t *kv_cache;
	uint8_t *index_cache;
	uint16_t *index_keys_host;
	uint8_t *kv_access_error_bytes;
	uint32_t *kv_access_error;
	uint32_t host_page_table[SPARK_GLM52_VALIDATION_LANES * SPARK_GLM52_VALIDATION_PAGES];
	uint32_t host_index_ordinals[1];
	uint32_t host_zero;
	uint32_t host_position;
	uint32_t host_token;
} SparkGlm52ValFixture;

static int SparkGlm52ValAllocMatrix(SparkGlm52ValMatrix *matrix,uint32_t rows,uint32_t columns,int mode,float scale)
{
	uint16_t *packed;
	uint64_t count = (uint64_t)rows * columns;
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->host = (float *)malloc(count * sizeof(float));
	packed = (uint16_t *)malloc(count * sizeof(uint16_t));
	if ( matrix->host == 0 || packed == 0 )
		return(SparkGlm52ValFail("fixture","host_alloc"));
	if ( cudaMalloc((void **)&matrix->device,count * sizeof(uint16_t)) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","device_alloc"));
	SparkGlm52ValRandomState += 101u;
	if ( mode == 1 )
		SparkGlm52ValFillNorm(packed,matrix->host,count);
	else
		SparkGlm52ValFill(packed,matrix->host,count,scale);
	if ( cudaMemcpy(matrix->device,packed,count * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","weight_upload"));
	free(packed);
	return(0);
}

static void SparkGlm52ValFreeMatrix(SparkGlm52ValMatrix *matrix)
{
	free(matrix->host);
	cudaFree(matrix->device);
	memset(matrix,0,sizeof(*matrix));
}

static void *SparkGlm52ValAllocZeroed(uint64_t bytes)
{
	void *pointer;
	if ( cudaMalloc(&pointer,bytes != 0u ? bytes : 16u) != cudaSuccess )
		return(0);
	if ( cudaMemset(pointer,0,bytes != 0u ? bytes : 16u) != cudaSuccess )
	{
		cudaFree(pointer);
		return(0);
	}
	return(pointer);
}

/* Synthesize one expert's payload + scale planes. Only exactly representable
 * (code, power-of-two scale) pairs are emitted, so decode is lossless. */
static void SparkGlm52ValFillExpertSlab(uint32_t codec,uint8_t *payload,uint8_t *scales,uint32_t rows,uint32_t columns)
{
	uint32_t row,column;
	uint64_t payload_row_bytes = SparkGlm52ValPayloadRowBytes(codec,columns);
	memset(payload,0,(uint64_t)rows * payload_row_bytes);
	for (row = 0u; row < rows; row++)
	{
		uint8_t *row_base = payload + (uint64_t)row * payload_row_bytes;
		for (column = 0u; column < columns; column++)
		{
			uint32_t draw = SparkGlm52ValNext();
			uint64_t bit = (uint64_t)column * SparkGlm52ValCodecStoredBits(codec);
			if ( SparkGlm52ValCodecUsesSignedIntGrid(codec) )
			{
				static const int32_t int6_choices[7] = {-9,-4,-1,0,1,5,11};
				static const int32_t int7_choices[7] = {-19,-7,-2,0,3,13,29};
				static const int32_t int8_choices[7] = {-41,-17,-5,0,7,23,59};
				const int32_t *choices = codec == SPARK_GLM52_VAL_CODEC_INT6 ? int6_choices :
					(codec == SPARK_GLM52_VAL_CODEC_INT7 ? int7_choices : int8_choices);
				int32_t code = choices[draw % 7u];
				uint32_t width = SparkGlm52ValCodecStoredBits(codec);
				uint32_t lane;
				for (lane = 0u; lane < width; lane++)
					if ( ((uint32_t)code >> lane) & 1u )
						row_base[(bit + lane) >> 3u] |= (uint8_t)(1u << ((bit + lane) & 7u));
			}
			else if ( codec == SPARK_GLM52_VAL_CODEC_FP8 )
			{
				/* Finite E4M3 codes with three-bit mantissas: exact in bf16. */
				static const uint8_t fp8_choices[10] = {0x08,0x2c,0x30,0x34,0x38,0x3c,0x40,0x44,0x88,0xb0};
				row_base[column] = fp8_choices[draw % 10u];
			}
			else
			{
				static const uint8_t nibble_choices[8] = {0,1,2,3,4,5,6,9};
				uint8_t nibble = nibble_choices[draw % 8u];
				if ( (column & 1u) == 0u )
					row_base[column >> 1u] |= nibble;
				else
					row_base[column >> 1u] |= (uint8_t)(nibble << 4u);
			}
		}
	}
	/* Scales: powers of two (exact in every encoding). NVFP4's per-expert
	 * f32 global lives ahead of all slabs and is written by the caller. */
	if ( codec == SPARK_GLM52_VAL_CODEC_NVFP4 )
		memset(scales,0x38,SparkGlm52ValScaleBytesPerExpert(codec,rows,columns)); /* 0x38 = 1.0 */
	else if ( codec == SPARK_GLM52_VAL_CODEC_MXFP4 )
		memset(scales,127,SparkGlm52ValScaleBytesPerExpert(codec,rows,columns)); /* 2^0 */
	else
	{
		float scale = codec == SPARK_GLM52_VAL_CODEC_FP8 ? 1.0f : 0.015625f; /* 2^-6 */
		uint64_t blocks = SparkGlm52ValScaleBlocksPerExpert(codec,rows,columns),index;
		for (index = 0u; index < blocks; index++)
			memcpy(scales + index * sizeof(float),&scale,sizeof(float));
	}
}

static int SparkGlm52ValFixtureSetup(SparkGlm52ValFixture *fixture)
{
	uint64_t pool_bytes,payload_bytes,scale_bytes;
	uint32_t lane,pages;
	memset(fixture,0,sizeof(*fixture));
	fixture->host_index_ordinals[0] = 0u; /* the DSA tier's index cache ordinal */
	fixture->host_zero = 0u;
	fixture->host_token = 0u;
	fixture->host_position = 0u;
	if ( cudaStreamCreate(&fixture->stream) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","stream_create"));
	if ( SparkGlm52ConfigureCudaModule(&fixture->multiprocessors) != 0 )
		return(SparkGlm52ValFail("configure","not_sm121"));
	/* Dense layer 0 synthetic weights. The scales keep every intermediate
	 * in an honest numeric range through six stacked projections. */
	if ( SparkGlm52ValAllocMatrix(&fixture->embedding,SPARK_GLM52_VALIDATION_EMBED_ROWS,SPARK_GLM52_VHIDDEN,0,0.05f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->attn_norm,1u,SPARK_GLM52_VHIDDEN,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->q_a,SPARK_GLM52_VQUERY_A,SPARK_GLM52_VHIDDEN,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->q_a_norm,1u,SPARK_GLM52_VQUERY_A,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->q_b,SPARK_GLM52_VQ_ROWS,SPARK_GLM52_VQUERY_A,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_a,SPARK_GLM52_VKV_SLOT_ELEMENTS,SPARK_GLM52_VHIDDEN,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_a_norm,1u,SPARK_GLM52_VLATENT,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_b_key,SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT,SPARK_GLM52_VQK_NOPE,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->kv_b_value,SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM,SPARK_GLM52_VLATENT,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->attn_output,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VATTN_COLS,0,0.01f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->post_attn_norm,1u,SPARK_GLM52_VHIDDEN,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->dense_gate_up,SPARK_GLM52_VGATE_UP_ROWS,SPARK_GLM52_VHIDDEN,0,0.01f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->dense_down,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VDENSE_INTER,0,0.01f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->router,256u,SPARK_GLM52_VHIDDEN,0,0.002f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->shared_gate_up,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VHIDDEN,0,0.005f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->shared_down,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS,0,0.005f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->index_q,SPARK_GLM52_VDSA_QUERY_DIM,SPARK_GLM52_VQUERY_A,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->index_k,SPARK_GLM52_VDSA_DIM,SPARK_GLM52_VHIDDEN,0,0.02f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->index_head,SPARK_GLM52_VDSA_HEADS,SPARK_GLM52_VHIDDEN,0,0.01f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->index_norm_weight,1u,SPARK_GLM52_VDSA_DIM,1,0.0f) != 0 ||
		SparkGlm52ValAllocMatrix(&fixture->index_norm_bias,1u,SPARK_GLM52_VDSA_DIM,1,0.0f) != 0 )
		return(1);
	/* The index-key LayerNorm bias is zero here; the matrix form keeps the
	 * oracle's restatement of LmLayerNormKernel honest (weight AND bias).
	 * (The origin/unified restore left index_norm_bias unallocated; the
	 * NULL-host memset segfaulted the validator before any tier ran.) */
	memset(fixture->index_norm_bias.host,0,(uint64_t)SPARK_GLM52_VDSA_DIM * sizeof(float));
	if ( cudaMemcpy(fixture->index_norm_bias.device,fixture->index_norm_bias.host,
		(uint64_t)SPARK_GLM52_VDSA_DIM * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","bias_upload"));
	payload_bytes = SparkGlm52ValPayloadBytesPerExpert(SPARK_GLM52_VAL_CODEC,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS);
	scale_bytes = SparkGlm52ValScaleBufferBytes(SPARK_GLM52_VAL_CODEC,256u,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS);
	fixture->experts.w1_payload = (uint8_t *)calloc(1,(uint64_t)SPARK_GLM52_VALIDATION_EXPERT_SLOTS * payload_bytes);
	fixture->experts.w1_scales = (uint8_t *)calloc(1,scale_bytes);
	fixture->experts.w1_device_payload = (uint8_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_EXPERT_SLOTS * payload_bytes);
	fixture->experts.w1_device_scale = (uint8_t *)SparkGlm52ValAllocZeroed(scale_bytes);
	if ( fixture->experts.w1_payload == 0 || fixture->experts.w1_scales == 0 ||
		fixture->experts.w1_device_payload == 0 || fixture->experts.w1_device_scale == 0 )
		return(SparkGlm52ValFail("fixture","expert_alloc"));
	/* The launcher prices scales for all 256 experts but reads only routed
	 * ones; nvfp4's per-expert f32 globals therefore cover the full group
	 * count while slabs exist for the pinned selection window. */
	if ( SPARK_GLM52_VAL_CODEC == SPARK_GLM52_VAL_CODEC_NVFP4 )
	{
		float global_scale = 4.0f;
		uint32_t expert;
		for (expert = 0u; expert < 256u; expert++)
			memcpy(fixture->experts.w1_scales + expert * sizeof(float),&global_scale,sizeof(float));
	}
	for (lane = 0u; lane < SPARK_GLM52_VALIDATION_EXPERT_SLOTS; lane++)
	{
		SparkGlm52ValFillExpertSlab(SPARK_GLM52_VAL_CODEC,
			fixture->experts.w1_payload + (uint64_t)lane * payload_bytes,
			fixture->experts.w1_scales + SparkGlm52ValScaleExpertOffset(SPARK_GLM52_VAL_CODEC,256u,lane,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS),
			SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS);
		if ( cudaMemcpy(fixture->experts.w1_device_payload + (uint64_t)lane * payload_bytes,
			fixture->experts.w1_payload + (uint64_t)lane * payload_bytes,payload_bytes,cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkGlm52ValFail("fixture","expert_upload"));
	}
	if ( cudaMemcpy(fixture->experts.w1_device_scale,fixture->experts.w1_scales,scale_bytes,cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","expert_upload"));
	payload_bytes = SparkGlm52ValPayloadBytesPerExpert(SPARK_GLM52_VAL_CODEC,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS);
	scale_bytes = SparkGlm52ValScaleBufferBytes(SPARK_GLM52_VAL_CODEC,256u,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS);
	fixture->experts.w2_payload = (uint8_t *)calloc(1,(uint64_t)SPARK_GLM52_VALIDATION_EXPERT_SLOTS * payload_bytes);
	fixture->experts.w2_scales = (uint8_t *)calloc(1,scale_bytes);
	fixture->experts.w2_device_payload = (uint8_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_EXPERT_SLOTS * payload_bytes);
	fixture->experts.w2_device_scale = (uint8_t *)SparkGlm52ValAllocZeroed(scale_bytes);
	if ( fixture->experts.w2_payload == 0 || fixture->experts.w2_scales == 0 ||
		fixture->experts.w2_device_payload == 0 || fixture->experts.w2_device_scale == 0 )
		return(SparkGlm52ValFail("fixture","expert_alloc"));
	if ( SPARK_GLM52_VAL_CODEC == SPARK_GLM52_VAL_CODEC_NVFP4 )
	{
		float global_scale = 4.0f;
		uint32_t expert;
		for (expert = 0u; expert < 256u; expert++)
			memcpy(fixture->experts.w2_scales + expert * sizeof(float),&global_scale,sizeof(float));
	}
	for (lane = 0u; lane < SPARK_GLM52_VALIDATION_EXPERT_SLOTS; lane++)
	{
		SparkGlm52ValFillExpertSlab(SPARK_GLM52_VAL_CODEC,
			fixture->experts.w2_payload + (uint64_t)lane * payload_bytes,
			fixture->experts.w2_scales + SparkGlm52ValScaleExpertOffset(SPARK_GLM52_VAL_CODEC,256u,lane,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS),
			SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS);
		if ( cudaMemcpy(fixture->experts.w2_device_payload + (uint64_t)lane * payload_bytes,
			fixture->experts.w2_payload + (uint64_t)lane * payload_bytes,payload_bytes,cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkGlm52ValFail("fixture","expert_upload"));
	}
	if ( cudaMemcpy(fixture->experts.w2_device_scale,fixture->experts.w2_scales,scale_bytes,cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","expert_upload"));
	/* The pinned selection: the correction bias selects experts 0..7 and is
	 * weighed nowhere (LmTopkSmallKernel's documented select-vs-weigh split),
	 * so the mixture weights stay unbiased sigmoids. */
	for (lane = 0u; lane < 256u; lane++)
		fixture->correction_host[lane] = lane < SPARK_GLM52_VALIDATION_EXPERT_SLOTS ? 4.0f : -4.0f;
	fixture->correction_bias = (float *)SparkGlm52ValAllocZeroed(256u * sizeof(float));
	if ( fixture->correction_bias == 0 )
		return(SparkGlm52ValFail("fixture","device_alloc"));
	if ( cudaMemcpy(fixture->correction_bias,fixture->correction_host,sizeof(fixture->correction_host),cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","bias_upload"));
	fixture->hidden = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->residual = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->normed = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->q_compressed = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VQUERY_A * 8u * sizeof(uint16_t));
	fixture->q_bf16 = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VQ_ROWS * 8u * sizeof(uint16_t));
	fixture->query_rope = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VROPE * 8u * sizeof(uint16_t));
	fixture->query_latent = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * 8u * sizeof(uint16_t));
	fixture->kv_slot = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VKV_SLOT_ELEMENTS * 8u * sizeof(uint16_t));
	fixture->attention_latent = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * 8u * sizeof(uint16_t));
	fixture->attention_value = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM * 8u * sizeof(uint16_t));
	fixture->attention_out = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->gate_up = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VGATE_UP_ROWS * 8u * sizeof(uint16_t));
	fixture->intermediate = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VGATE_UP_ROWS * 8u * sizeof(uint16_t));
	fixture->expert_out = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VTOP_K * SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	fixture->shared_out = (uint16_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	fixture->scratch_small = (uint16_t *)SparkGlm52ValAllocZeroed(8192u * sizeof(uint16_t));
	fixture->float_scratch = (float *)SparkGlm52ValAllocZeroed(8192u * sizeof(float));
	fixture->token_ids = (uint32_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint32_t));
	fixture->resident_slots = (uint32_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint32_t));
	fixture->positions = (uint32_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint32_t));
	fixture->context_lengths = (uint32_t *)SparkGlm52ValAllocZeroed(SPARK_GLM52_VALIDATION_LANES * sizeof(uint32_t));
	fixture->dense_row_offset = (uint32_t *)SparkGlm52ValAllocZeroed(4u * sizeof(uint32_t));
	fixture->dense_tile_prefix = (uint32_t *)SparkGlm52ValAllocZeroed(4u * sizeof(uint32_t));
	fixture->page_table = (uint32_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_LANES * SPARK_GLM52_VALIDATION_PAGES * sizeof(uint32_t));
	fixture->route_scratch = (uint32_t *)SparkGlm52ValAllocZeroed(4096u * sizeof(uint32_t));
	fixture->route_expert = (uint32_t *)SparkGlm52ValAllocZeroed(64u * sizeof(uint32_t));
	fixture->route_packed_row = (uint32_t *)SparkGlm52ValAllocZeroed(64u * sizeof(uint32_t));
	fixture->route_source_token = (uint32_t *)SparkGlm52ValAllocZeroed(64u * sizeof(uint32_t));
	fixture->group_row_offset = (uint32_t *)SparkGlm52ValAllocZeroed(257u * sizeof(uint32_t));
	fixture->group_tile_prefix_w1 = (uint32_t *)SparkGlm52ValAllocZeroed(257u * sizeof(uint32_t));
	fixture->group_tile_prefix_w2 = (uint32_t *)SparkGlm52ValAllocZeroed(257u * sizeof(uint32_t));
	fixture->selected_positions = (uint32_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VDSA_SELECTED * sizeof(uint32_t));
	fixture->router_logits = (float *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_ROUTES * 256u * sizeof(float));
	fixture->selection_scores = (float *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VALIDATION_DSA_CONTEXT * sizeof(float));
	fixture->route_weight = (float *)SparkGlm52ValAllocZeroed(64u * sizeof(float));
	fixture->head_maxloc = (uint64_t *)SparkGlm52ValAllocZeroed(8u * sizeof(uint64_t));
	pool_bytes = (uint64_t)SPARK_GLM52_VALIDATION_LANES * SPARK_GLM52_VALIDATION_PAGES * SPARK_GLM52_VKV_PAGE_BYTES;
	fixture->kv_cache = (uint8_t *)SparkGlm52ValAllocZeroed(pool_bytes);
	fixture->index_cache = (uint8_t *)SparkGlm52ValAllocZeroed((uint64_t)SPARK_GLM52_VALIDATION_PAGES * SPARK_GLM52_VINDEX_PAGE_BYTES);
	fixture->index_keys_host = (uint16_t *)calloc((uint64_t)SPARK_GLM52_VALIDATION_DSA_CONTEXT,SPARK_GLM52_VDSA_DIM * sizeof(uint16_t));
	fixture->kv_access_error = (uint32_t *)SparkGlm52ValAllocZeroed(SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS * sizeof(uint32_t));
	if ( fixture->hidden == 0 || fixture->residual == 0 || fixture->normed == 0 ||
		fixture->q_compressed == 0 || fixture->q_bf16 == 0 || fixture->query_rope == 0 ||
		fixture->query_latent == 0 || fixture->kv_slot == 0 || fixture->attention_latent == 0 ||
		fixture->attention_value == 0 || fixture->attention_out == 0 || fixture->gate_up == 0 ||
		fixture->intermediate == 0 || fixture->expert_out == 0 || fixture->shared_out == 0 ||
		fixture->scratch_small == 0 || fixture->float_scratch == 0 ||
		fixture->token_ids == 0 || fixture->resident_slots == 0 || fixture->positions == 0 ||
		fixture->context_lengths == 0 || fixture->dense_row_offset == 0 ||
		fixture->dense_tile_prefix == 0 || fixture->page_table == 0 || fixture->route_scratch == 0 ||
		fixture->route_expert == 0 || fixture->route_packed_row == 0 || fixture->route_source_token == 0 ||
		fixture->group_row_offset == 0 || fixture->group_tile_prefix_w1 == 0 ||
		fixture->group_tile_prefix_w2 == 0 || fixture->selected_positions == 0 ||
		fixture->router_logits == 0 || fixture->selection_scores == 0 || fixture->route_weight == 0 ||
		fixture->head_maxloc == 0 || fixture->kv_access_error == 0 || fixture->kv_cache == 0 ||
		fixture->index_cache == 0 || fixture->index_keys_host == 0 )
		return(SparkGlm52ValFail("fixture","device_alloc"));
	for (lane = 0u; lane < SPARK_GLM52_VALIDATION_LANES; lane++)
		for (pages = 0u; pages < SPARK_GLM52_VALIDATION_PAGES; pages++)
			fixture->host_page_table[lane * SPARK_GLM52_VALIDATION_PAGES + pages] =
				lane * SPARK_GLM52_VALIDATION_PAGES + pages;
	if ( cudaMemcpy(fixture->page_table,fixture->host_page_table,sizeof(fixture->host_page_table),cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkGlm52ValFail("fixture","page_table_upload"));
	return(0);
}

static void SparkGlm52ValFixtureDestroy(SparkGlm52ValFixture *fixture)
{
	SparkGlm52ValFreeMatrix(&fixture->embedding);
	SparkGlm52ValFreeMatrix(&fixture->attn_norm);
	SparkGlm52ValFreeMatrix(&fixture->q_a);
	SparkGlm52ValFreeMatrix(&fixture->q_a_norm);
	SparkGlm52ValFreeMatrix(&fixture->q_b);
	SparkGlm52ValFreeMatrix(&fixture->kv_a);
	SparkGlm52ValFreeMatrix(&fixture->kv_a_norm);
	SparkGlm52ValFreeMatrix(&fixture->kv_b_key);
	SparkGlm52ValFreeMatrix(&fixture->kv_b_value);
	SparkGlm52ValFreeMatrix(&fixture->attn_output);
	SparkGlm52ValFreeMatrix(&fixture->post_attn_norm);
	SparkGlm52ValFreeMatrix(&fixture->dense_gate_up);
	SparkGlm52ValFreeMatrix(&fixture->dense_down);
	SparkGlm52ValFreeMatrix(&fixture->router);
	SparkGlm52ValFreeMatrix(&fixture->shared_gate_up);
	SparkGlm52ValFreeMatrix(&fixture->shared_down);
	SparkGlm52ValFreeMatrix(&fixture->index_q);
	SparkGlm52ValFreeMatrix(&fixture->index_k);
	SparkGlm52ValFreeMatrix(&fixture->index_head);
	SparkGlm52ValFreeMatrix(&fixture->index_norm_weight);
	SparkGlm52ValFreeMatrix(&fixture->index_norm_bias);
	free(fixture->experts.w1_payload);
	free(fixture->experts.w1_scales);
	free(fixture->experts.w2_payload);
	free(fixture->experts.w2_scales);
	cudaFree(fixture->experts.w1_device_payload);
	cudaFree(fixture->experts.w1_device_scale);
	cudaFree(fixture->experts.w2_device_payload);
	cudaFree(fixture->experts.w2_device_scale);
	cudaFree(fixture->correction_bias);
	cudaFree(fixture->hidden); cudaFree(fixture->residual); cudaFree(fixture->normed);
	cudaFree(fixture->q_compressed); cudaFree(fixture->q_bf16); cudaFree(fixture->query_rope);
	cudaFree(fixture->query_latent); cudaFree(fixture->kv_slot); cudaFree(fixture->attention_latent);
	cudaFree(fixture->attention_value); cudaFree(fixture->attention_out); cudaFree(fixture->gate_up);
	cudaFree(fixture->intermediate); cudaFree(fixture->expert_out); cudaFree(fixture->shared_out);
	cudaFree(fixture->scratch_small); cudaFree(fixture->float_scratch);
	cudaFree(fixture->token_ids); cudaFree(fixture->resident_slots); cudaFree(fixture->positions);
	cudaFree(fixture->context_lengths); cudaFree(fixture->dense_row_offset); cudaFree(fixture->dense_tile_prefix);
	cudaFree(fixture->page_table); cudaFree(fixture->route_scratch);
	cudaFree(fixture->route_expert); cudaFree(fixture->route_packed_row); cudaFree(fixture->route_source_token);
	cudaFree(fixture->group_row_offset); cudaFree(fixture->group_tile_prefix_w1); cudaFree(fixture->group_tile_prefix_w2);
	cudaFree(fixture->selected_positions); cudaFree(fixture->router_logits); cudaFree(fixture->selection_scores);
	cudaFree(fixture->route_weight); cudaFree(fixture->head_maxloc);
	cudaFree(fixture->kv_access_error); cudaFree(fixture->kv_cache);
	cudaFree(fixture->index_cache);
	free(fixture->index_keys_host);
	if ( fixture->stream != 0 )
		cudaStreamDestroy(fixture->stream);
}

static int SparkGlm52ValResetStreams(SparkGlm52ValFixture *fixture)
{
	uint64_t pool_bytes = (uint64_t)SPARK_GLM52_VALIDATION_LANES * SPARK_GLM52_VALIDATION_PAGES * SPARK_GLM52_VKV_PAGE_BYTES;
	if ( cudaMemset(fixture->kv_cache,0,pool_bytes) != cudaSuccess ||
		cudaMemset(fixture->index_cache,0,(uint64_t)SPARK_GLM52_VALIDATION_PAGES * SPARK_GLM52_VINDEX_PAGE_BYTES) != cudaSuccess ||
		cudaMemset(fixture->kv_access_error,0,SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS * sizeof(uint32_t)) != cudaSuccess ||
		cudaMemset(fixture->hidden,0,(uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->residual,0,(uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->normed,0,(uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->q_compressed,0,(uint64_t)SPARK_GLM52_VQUERY_A * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->q_bf16,0,(uint64_t)SPARK_GLM52_VQ_ROWS * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->query_latent,0,(uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->attention_latent,0,(uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->attention_value,0,(uint64_t)SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->attention_out,0,(uint64_t)SPARK_GLM52_VHIDDEN * 8u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(fixture->router_logits,0,(uint64_t)SPARK_GLM52_VALIDATION_ROUTES * 256u * sizeof(float)) != cudaSuccess ||
		cudaMemset(fixture->selection_scores,0,(uint64_t)SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VALIDATION_DSA_CONTEXT * sizeof(float)) != cudaSuccess ||
		cudaMemset(fixture->selected_positions,0,(uint64_t)SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VDSA_SELECTED * sizeof(uint32_t)) != cudaSuccess )
		return(SparkGlm52ValFail("reset","memset"));
	return(0);
}

/* Bind the fixture into the production wave exactly as BindLayer does for a
 * whole-stack TP1 rank's local layer 0. */
static void SparkGlm52ValBuildWave(SparkGlm52ValFixture *fixture,uint32_t first_layer_index,uint32_t token,uint32_t position)
{
	SparkGlm52CudaWave *wave = &fixture->wave;
	SparkGlm52ExecutionSlot *slot = &fixture->slot;
	fixture->host_token = token;
	fixture->host_position = position;
	memset(slot,0,sizeof(*slot));
	slot->stream = fixture->stream;
	slot->hidden_bf16 = fixture->hidden;
	slot->residual_bf16 = fixture->residual;
	slot->normed_bf16 = fixture->normed;
	slot->q_compressed_bf16 = fixture->q_compressed;
	slot->q_bf16 = fixture->q_bf16;
	slot->query_latent_bf16 = fixture->query_latent;
	slot->query_rope_bf16 = fixture->query_rope;
	slot->index_query_bf16 = fixture->scratch_small;
	slot->index_key_bf16 = fixture->scratch_small;
	slot->index_head_weight_bf16 = fixture->scratch_small;
	slot->kv_slot_bf16 = fixture->kv_slot;
	slot->attention_latent_bf16 = fixture->attention_latent;
	slot->attention_value_bf16 = fixture->attention_value;
	slot->attention_out_bf16 = fixture->attention_out;
	slot->gate_up_bf16 = fixture->gate_up;
	slot->intermediate_bf16 = fixture->intermediate;
	slot->expert_out_bf16 = fixture->expert_out;
	slot->shared_out_bf16 = fixture->shared_out;
	slot->router_logits_f32 = fixture->router_logits;
	slot->selection_scores_f32 = fixture->selection_scores;
	slot->selected_positions = fixture->selected_positions;
	slot->route_expert = fixture->route_expert;
	slot->route_weight = fixture->route_weight;
	slot->route_source_token = fixture->route_source_token;
	slot->route_packed_row = fixture->route_packed_row;
	slot->group_row_offset = fixture->group_row_offset;
	slot->group_tile_prefix_w1 = fixture->group_tile_prefix_w1;
	slot->group_tile_prefix_w2 = fixture->group_tile_prefix_w2;
	slot->head_candidate_score = fixture->float_scratch;
	slot->head_candidate_token = fixture->route_scratch;
	slot->output_token = fixture->route_scratch;
	slot->output_score = fixture->float_scratch;
	slot->head_maxloc_u64 = fixture->head_maxloc;
	slot->dense_row_offset = fixture->dense_row_offset;
	slot->dense_tile_prefix = fixture->dense_tile_prefix;
	slot->context_lengths = fixture->context_lengths;
	slot->token_ids = fixture->token_ids;
	slot->resident_slots = fixture->resident_slots;
	slot->positions = fixture->positions;
	slot->kv_access_error = fixture->kv_access_error;
	fixture->weights.attn_norm_bf16 = fixture->attn_norm.device;
	fixture->weights.attn_output_bf16 = fixture->attn_output.device;
	fixture->weights.post_attn_norm_bf16 = fixture->post_attn_norm.device;
	fixture->weights.q_a_bf16 = fixture->q_a.device;
	fixture->weights.q_a_norm_bf16 = fixture->q_a_norm.device;
	fixture->weights.q_b_bf16 = fixture->q_b.device;
	fixture->weights.kv_a_bf16 = fixture->kv_a.device;
	fixture->weights.kv_a_norm_bf16 = fixture->kv_a_norm.device;
	fixture->weights.kv_b_key_transposed_bf16 = fixture->kv_b_key.device;
	fixture->weights.kv_b_value_bf16 = fixture->kv_b_value.device;
	fixture->weights.index_q_bf16 = fixture->index_q.device;
	fixture->weights.index_k_bf16 = fixture->index_k.device;
	fixture->weights.index_head_bf16 = fixture->index_head.device;
	fixture->weights.index_norm_weight_bf16 = fixture->index_norm_weight.device;
	fixture->weights.index_norm_bias_bf16 = fixture->index_norm_bias.device;
	fixture->weights.dense_gate_up_bf16 = fixture->dense_gate_up.device;
	fixture->weights.dense_down_bf16 = fixture->dense_down.device;
	fixture->weights.router_bf16 = fixture->router.device;
	fixture->weights.router_correction_f32 = fixture->correction_bias;
	fixture->weights.expert_up_gate_payload = fixture->experts.w1_device_payload;
	fixture->weights.expert_up_gate_scale = fixture->experts.w1_device_scale;
	fixture->weights.expert_down_payload = fixture->experts.w2_device_payload;
	fixture->weights.expert_down_scale = fixture->experts.w2_device_scale;
	fixture->weights.shared_gate_up_bf16 = fixture->shared_gate_up.device;
	fixture->weights.shared_down_bf16 = fixture->shared_down.device;
	memset(wave,0,sizeof(*wave));
	wave->stage_index = 0u;
	wave->first_layer_index = first_layer_index;
	wave->layer_count = 1u;
	wave->tp_degree = 1u;
	wave->tp_rank = 0u;
	wave->row_count = 1u;
	wave->maximum_context = position + 1u;
	wave->resident_sequence_capacity = SPARK_GLM52_VALIDATION_LANES;
	wave->max_sequence_positions = SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS;
	wave->pages_per_sequence = SPARK_GLM52_VALIDATION_PAGES;
	wave->owns_embedding = 1u;
	wave->owns_final_head = 0u;
	wave->sideband_input = 0u;
	wave->sideband_output = 0u;
	wave->boundary_row_offset = 0u;
	wave->sideband_row_offset = 0u;
	wave->host_token_ids = &fixture->host_token;
	wave->host_resident_slots = &fixture->host_zero;
	wave->host_positions = &fixture->host_position;
	wave->embedding_bf16 = fixture->embedding.device;
	wave->layers = &fixture->weights;
	wave->slot = slot;
	wave->kv_cache = fixture->kv_cache;
	wave->kv_layer_stride_bytes = SPARK_GLM52_VKV_LAYER_BYTES;
	wave->index_cache = fixture->index_cache;
	wave->index_layer_stride_bytes = 0u;
	wave->index_ordinal_by_local_layer = fixture->host_index_ordinals;
	wave->page_table = fixture->page_table;
	wave->multiprocessor_count = fixture->multiprocessors;
}

static int SparkGlm52ValRunWalk(SparkGlm52ValFixture *fixture)
{
	static const uint32_t tokens[SPARK_GLM52_VALIDATION_TOKENS] = {1u,3u,2u,6u};
	uint32_t step;
	int32_t status;
	for (step = 0u; step < SPARK_GLM52_VALIDATION_TOKENS; step++)
	{
		SparkGlm52ValBuildWave(fixture,0u,tokens[step],step);
		status = SparkGlm52LaunchCudaWaveBegin(&fixture->wave);
		if ( status != 0 )
			return(SparkGlm52ValFail("wave_begin","status"));
		status = SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u);
		if ( status != 0 )
		{
			fprintf(stderr,"glm52_validation layer_attention status=%d step=%u\n",status,step);
			return(SparkGlm52ValFail("layer_attention","status"));
		}
		status = SparkGlm52LaunchCudaLayerMlp(&fixture->wave,0u);
		if ( status != 0 )
		{
			fprintf(stderr,"glm52_validation layer_mlp status=%d step=%u\n",status,step);
			return(SparkGlm52ValFail("layer_mlp","status"));
		}
		if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
			return(SparkGlm52ValFail("walk","sync"));
	}
	return(0);
}

static int SparkGlm52ValCheckAccessError(SparkGlm52ValFixture *fixture)
{
	uint32_t error_words[SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS];
	uint32_t index;
	if ( cudaMemcpy(error_words,fixture->kv_access_error,sizeof(error_words),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(SparkGlm52ValFail("kv_access","error_readback"));
	for (index = 0u; index < SPARK_GLM52_VALIDATION_KV_ACCESS_WORDS; index++)
		if ( error_words[index] != 0u )
			return(SparkGlm52ValFail("kv_access","error_word_set"));
	return(0);
}

/* -- fp32 CPU oracle -------------------------------------------------------- */

#define SPARK_GLM52_VAL_CACHE_ENTRIES 8u

typedef struct SparkGlm52ValCacheEntry
{
	uint32_t position;
	float slot[SPARK_GLM52_VKV_SLOT_ELEMENTS];
} SparkGlm52ValCacheEntry;

typedef struct SparkGlm52ValOracle
{
	const SparkGlm52ValFixture *fixture;
	float hidden[SPARK_GLM52_VHIDDEN];
	float residual[SPARK_GLM52_VHIDDEN];
	float normed[SPARK_GLM52_VHIDDEN];
	float q_compressed[SPARK_GLM52_VQUERY_A];
	float q_row[SPARK_GLM52_VQ_ROWS];
	float q_rope[SPARK_GLM52_VHEADS * SPARK_GLM52_VROPE];
	float query_latent[SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT];
	float kv_slot[SPARK_GLM52_VKV_SLOT_ELEMENTS];
	float attention_latent[SPARK_GLM52_VHEADS * SPARK_GLM52_VLATENT];
	float attention_value[SPARK_GLM52_VHEADS * SPARK_GLM52_VVALUE_DIM];
	float attention_out[SPARK_GLM52_VHIDDEN];
	float gate_up[SPARK_GLM52_VGATE_UP_ROWS];
	float intermediate[SPARK_GLM52_VDENSE_INTER];
	/* DSA tier extras. */
	float index_query[SPARK_GLM52_VDSA_QUERY_DIM];
	float index_key[SPARK_GLM52_VDSA_DIM];
	float index_heads[SPARK_GLM52_VDSA_HEADS];
	/* cached main-KV slots, sparse association: most positions read ZERO
	 * (fresh pool), so only the written ones are stored. */
	SparkGlm52ValCacheEntry cache[SPARK_GLM52_VAL_CACHE_ENTRIES];
	uint32_t cache_count;
} SparkGlm52ValOracle;

static void SparkGlm52ValCachePut(SparkGlm52ValOracle *oracle,uint32_t position,const float *slot)
{
	uint32_t index;
	for (index = 0u; index < oracle->cache_count; index++)
	{
		if ( oracle->cache[index].position == position )
		{
			memcpy(oracle->cache[index].slot,slot,sizeof(oracle->cache[index].slot));
			return;
		}
	}
	if ( oracle->cache_count >= SPARK_GLM52_VAL_CACHE_ENTRIES )
		return; /* the fixtures never exceed this */
	oracle->cache[oracle->cache_count].position = position;
	memcpy(oracle->cache[oracle->cache_count].slot,slot,sizeof(oracle->cache[oracle->cache_count].slot));
	oracle->cache_count++;
}

static const float *SparkGlm52ValCacheGet(const SparkGlm52ValOracle *oracle,uint32_t position)
{
	uint32_t index;
	for (index = 0u; index < oracle->cache_count; index++)
		if ( oracle->cache[index].position == position )
			return(oracle->cache[index].slot);
	return(0);
}

/* C[r][o] = bf16(sum_k A[r][k] * W[o][k]); A/W already exact fp32 upcasts. */
static void SparkGlm52ValGemmRow(const float *activation,const float *weights,float *output,uint32_t input_dimension,uint32_t output_dimension)
{
	uint32_t column,index;
	float total;
	for (column = 0u; column < output_dimension; column++)
	{
		total = 0.0f;
		for (index = 0u; index < input_dimension; index++)
			total += activation[index] * weights[((uint64_t)column * input_dimension) + index];
		output[column] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(total));
	}
}

/* Router GEMM writes fp32 accumulators (gemm.output_f32), unrounded. */
static void SparkGlm52ValRouterGemm(const float *activation,const float *weights,float *output,uint32_t input_dimension,uint32_t output_dimension)
{
	uint32_t column,index;
	float total;
	for (column = 0u; column < output_dimension; column++)
	{
		total = 0.0f;
		for (index = 0u; index < input_dimension; index++)
			total += activation[index] * weights[((uint64_t)column * input_dimension) + index];
		output[column] = total;
	}
}

/* Weight-only grouped GEMM row: A row comes through the route's source-row
 * map, B is the codec-decoded expert slab, output rounds to bf16. */
static void SparkGlm52ValExpertGemmRow(const uint8_t *payload,const uint8_t *scales,
	uint32_t expert,const float *activation,float *output,uint32_t rows,uint32_t input_dimension,uint32_t output_dimension)
{
	uint32_t column,index;
	float total;
	for (column = 0u; column < output_dimension; column++)
	{
		total = 0.0f;
		for (index = 0u; index < input_dimension; index++)
		{
			float weight = SparkGlm52ValDequantWeight(payload,scales,SPARK_GLM52_VAL_CODEC,256u,
				expert,column,rows,input_dimension,index);
			total += activation[index] * weight;
		}
		output[column] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(total));
	}
}

static void SparkGlm52ValRmsNorm(const float *input,const float *weight,float *output,uint32_t dimension,float epsilon)
{
	float total = 0.0f,scale;
	uint32_t index;
	for (index = 0u; index < dimension; index++)
		total += input[index] * input[index];
	scale = 1.0f / sqrtf((total / (float)dimension) + epsilon);
	for (index = 0u; index < dimension; index++)
		output[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(input[index] * scale * weight[index]));
}

/* LmLayerNormKernel: mean/variance form with learned weight AND bias. */
static void SparkGlm52ValLayerNorm(const float *input,const float *weight,const float *bias,float *output,uint32_t dimension,float epsilon)
{
	float total = 0.0f,squared = 0.0f,mean,variance,inverse;
	uint32_t index;
	for (index = 0u; index < dimension; index++)
	{
		total += input[index];
		squared += input[index] * input[index];
	}
	mean = total / (float)dimension;
	variance = (squared / (float)dimension) - (mean * mean);
	if ( variance < 0.0f )
		variance = 0.0f;
	inverse = 1.0f / sqrtf(variance + epsilon);
	for (index = 0u; index < dimension; index++)
		output[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(((input[index] - mean) * inverse * weight[index]) + bias[index]));
}

static void SparkGlm52ValRotatePair(float *low,float *high,float angle)
{
	float cosine = cosf(angle),sine = sinf(angle);
	float a = *low,b = *high;
	*low = (a * cosine) - (b * sine);
	*high = (a * sine) + (b * cosine);
}

/* Attention chunk through the current token's projections. When selected is
 * non-null the latent attention reads exactly those positions (sparse path);
 * otherwise it reads 0..position densely - the two production modes of
 * LmLatentAttentionDecodeKernel. Stores the token's roped latent row into
 * the oracle cache at its position, as the production store does. */
static void SparkGlm52ValOracleAttentionChunk(SparkGlm52ValOracle *oracle,const SparkGlm52ValFixture *fx,
	uint32_t token,uint32_t position,const uint32_t *selected,uint32_t selected_count)
{
	float sum,value,pair_low,pair_high,angle,score;
	float scores[SPARK_GLM52_VALIDATION_DSA_CONTEXT];
	float maximum,running_sum;
	uint32_t index,head,element,step,context;
	const float *slot;
	/* WaveBegin: embedding row into hidden, zero residual. */
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		oracle->hidden[index] = fx->embedding.host[((uint64_t)token * SPARK_GLM52_VHIDDEN) + index];
		oracle->residual[index] = 0.0f;
	}
	/* Attention chunk. attn-norm: residual add fused, residual stored pre-scale. */
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		sum = oracle->hidden[index] + oracle->residual[index];
		oracle->residual[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		oracle->hidden[index] = sum;
	}
	SparkGlm52ValRmsNorm(oracle->hidden,fx->attn_norm.host,oracle->normed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	SparkGlm52ValGemmRow(oracle->normed,fx->q_a.host,oracle->q_compressed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VQUERY_A);
	SparkGlm52ValRmsNorm(oracle->q_compressed,fx->q_a_norm.host,oracle->q_compressed,SPARK_GLM52_VQUERY_A,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	if ( selected != 0 )
	{
		/* Glm52LayerIndexer, full-indexer layer, context beyond the
		 * selection width: index query/key/head projections, index-key
		 * LayerNorm, per-head and flat rope, cache store, sparse scores. */
		SparkGlm52ValGemmRow(oracle->q_compressed,fx->index_q.host,oracle->index_query,
			SPARK_GLM52_VQUERY_A,SPARK_GLM52_VDSA_QUERY_DIM);
		SparkGlm52ValGemmRow(oracle->normed,fx->index_k.host,oracle->index_key,
			SPARK_GLM52_VHIDDEN,SPARK_GLM52_VDSA_DIM);
		SparkGlm52ValLayerNorm(oracle->index_key,fx->index_norm_weight.host,fx->index_norm_bias.host,
			oracle->index_key,SPARK_GLM52_VDSA_DIM,SPARK_GLM52_MODEL_DSA_INDEX_NORM_EPSILON);
		SparkGlm52ValGemmRow(oracle->normed,fx->index_head.host,oracle->index_heads,
			SPARK_GLM52_VHIDDEN,SPARK_GLM52_VDSA_HEADS);
		for (head = 0u; head < SPARK_GLM52_VDSA_HEADS; head++)
			for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
			{
				pair_low = oracle->index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index)];
				pair_high = oracle->index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index + 1u)];
				angle = (float)position * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
				SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
				oracle->index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index)] =
					SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
				oracle->index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index + 1u)] =
					SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
			}
		for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
		{
			pair_low = oracle->index_key[2u * index];
			pair_high = oracle->index_key[2u * index + 1u];
			angle = (float)position * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
			SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
			oracle->index_key[2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
			oracle->index_key[2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
		}
	}
	SparkGlm52ValGemmRow(oracle->q_compressed,fx->q_b.host,oracle->q_row,SPARK_GLM52_VQUERY_A,SPARK_GLM52_VQ_ROWS);
	SparkGlm52ValGemmRow(oracle->normed,fx->kv_a.host,oracle->kv_slot,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VKV_SLOT_ELEMENTS);
	/* kv_a-norm covers only the latent half of the 576-wide row. */
	SparkGlm52ValRmsNorm(oracle->kv_slot,fx->kv_a_norm.host,oracle->kv_slot,SPARK_GLM52_VLATENT,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	/* Query rope: extract [192..256) of every packed head, rotate, store. */
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
		for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
		{
			pair_low = oracle->q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + (SPARK_GLM52_VQK_NOPE + 2u * index)];
			pair_high = oracle->q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + (SPARK_GLM52_VQK_NOPE + 2u * index + 1u)];
			angle = (float)position * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
			SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
			oracle->q_rope[(head * SPARK_GLM52_VROPE) + 2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
			oracle->q_rope[(head * SPARK_GLM52_VROPE) + 2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
		}
	/* Key-side absorption: per-head W[512][192] over the RAW nope slice. */
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
		for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
		{
			sum = 0.0f;
			for (index = 0u; index < SPARK_GLM52_VQK_NOPE; index++)
				sum += oracle->q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + index] *
					fx->kv_b_key.host[((uint64_t)head * SPARK_GLM52_VLATENT + element) * SPARK_GLM52_VQK_NOPE + index];
			oracle->query_latent[(head * SPARK_GLM52_VLATENT) + element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		}
	/* Cache store: rope the latent row's tail IN PLACE, then copy. */
	for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
	{
		pair_low = oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index];
		pair_high = oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index + 1u];
		angle = (float)position * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
		SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
		oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
		oracle->kv_slot[SPARK_GLM52_VLATENT + 2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
	}
	SparkGlm52ValCachePut(oracle,position,oracle->kv_slot);
	/* Latent decode attention. Dense mode walks 0..position; sparse mode
	 * walks the selection array exactly as the kernel does. Unstored cache
	 * positions read ZERO: the pool starts zeroed and only produced slots
	 * carry data - the fixture's declared memory model. */
	context = selected != 0 ? selected_count : position + 1u;
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
	{
		maximum = -3.0e38f;
		running_sum = 0.0f;
		for (step = 0u; step < context; step++)
		{
			uint32_t cache_position = selected != 0 ? selected[step] : step;
			slot = SparkGlm52ValCacheGet(oracle,cache_position);
			score = 0.0f;
			if ( slot != 0 )
			{
				for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
					score += oracle->query_latent[(head * SPARK_GLM52_VLATENT) + element] * slot[element];
				for (element = 0u; element < SPARK_GLM52_VROPE; element++)
					score += oracle->q_rope[(head * SPARK_GLM52_VROPE) + element] * slot[SPARK_GLM52_VLATENT + element];
			}
			scores[step] = score * SPARK_GLM52_MODEL_QK_SCALE;
			if ( scores[step] > maximum )
				maximum = scores[step];
		}
		for (step = 0u; step < context; step++)
			running_sum += expf(scores[step] - maximum);
		if ( running_sum < 1.0e-20f )
			running_sum = 1.0e-20f;
		for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
		{
			value = 0.0f;
			for (step = 0u; step < context; step++)
			{
				uint32_t cache_position = selected != 0 ? selected[step] : step;
				slot = SparkGlm52ValCacheGet(oracle,cache_position);
				if ( slot != 0 )
					value += expf(scores[step] - maximum) * slot[element];
			}
			oracle->attention_latent[(head * SPARK_GLM52_VLATENT) + element] =
				SparkGlm52ValFromBf16(SparkGlm52ValBf16(value / running_sum));
		}
	}
	/* Value up-projection: per-head W[256][512]. */
	for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
		for (element = 0u; element < SPARK_GLM52_VVALUE_DIM; element++)
		{
			sum = 0.0f;
			for (index = 0u; index < SPARK_GLM52_VLATENT; index++)
				sum += oracle->attention_latent[(head * SPARK_GLM52_VLATENT) + index] *
					fx->kv_b_value.host[((uint64_t)head * SPARK_GLM52_VVALUE_DIM + element) * SPARK_GLM52_VLATENT + index];
			oracle->attention_value[(head * SPARK_GLM52_VVALUE_DIM) + element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		}
	SparkGlm52ValGemmRow(oracle->attention_value,fx->attn_output.host,oracle->attention_out,SPARK_GLM52_VATTN_COLS,SPARK_GLM52_VHIDDEN);
}

/* Dense MLP chunk (layers before first_routed): fused [up|gate], silu-mul,
 * down projection. hidden becomes the chunk output; residual stands. */
static void SparkGlm52ValOracleDenseMlp(SparkGlm52ValOracle *oracle,const SparkGlm52ValFixture *fx)
{
	float sum;
	uint32_t index;
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		sum = oracle->attention_out[index] + oracle->residual[index];
		oracle->residual[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		oracle->hidden[index] = sum;
	}
	SparkGlm52ValRmsNorm(oracle->hidden,fx->post_attn_norm.host,oracle->normed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	SparkGlm52ValGemmRow(oracle->normed,fx->dense_gate_up.host,oracle->gate_up,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VGATE_UP_ROWS);
	/* gate_first=false: up is the FIRST half, gate the SECOND. */
	for (index = 0u; index < SPARK_GLM52_VDENSE_INTER; index++)
	{
		float gate = oracle->gate_up[SPARK_GLM52_VDENSE_INTER + index];
		float up = oracle->gate_up[index];
		oracle->intermediate[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16((gate / (1.0f + expf(-gate))) * up));
	}
	SparkGlm52ValGemmRow(oracle->intermediate,fx->dense_down.host,oracle->hidden,SPARK_GLM52_VDENSE_INTER,SPARK_GLM52_VHIDDEN);
}

/* Routed MLP chunk (Glm52LayerMoe): router, sigmoid top-k with the bias
 * selecting, renormalised mixture, per-route expert forwards, weighted
 * finalize overwriting hidden, shared expert summed in. */
static void SparkGlm52ValOracleRoutedMlp(SparkGlm52ValOracle *oracle,const SparkGlm52ValFixture *fx,
	uint32_t *out_selected,float *out_weights)
{
	float sum,gate,up;
	float logits[256];
	uint32_t index,route,element;
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		sum = oracle->attention_out[index] + oracle->residual[index];
		oracle->residual[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
		oracle->hidden[index] = sum;
	}
	SparkGlm52ValRmsNorm(oracle->hidden,fx->post_attn_norm.host,oracle->normed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	SparkGlm52ValRouterGemm(oracle->normed,fx->router.host,logits,SPARK_GLM52_VHIDDEN,256u);
	SparkGlm52ValReferenceTopk(logits,fx->correction_host,256u,SPARK_GLM52_VTOP_K,out_selected,out_weights);
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
		oracle->hidden[index] = 0.0f;
	for (route = 0u; route < SPARK_GLM52_VTOP_K; route++)
	{
		uint32_t expert = out_selected[route];
		float expert_row[SPARK_GLM52_VHIDDEN];
		SparkGlm52ValExpertGemmRow(fx->experts.w1_payload,fx->experts.w1_scales,expert,
			oracle->normed,oracle->gate_up,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS,SPARK_GLM52_VW1_ROWS);
		for (index = 0u; index < SPARK_GLM52_VEXPERT_INTER; index++)
		{
			gate = oracle->gate_up[SPARK_GLM52_VEXPERT_INTER + index];
			up = oracle->gate_up[index];
			oracle->intermediate[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16((gate / (1.0f + expf(-gate))) * up));
		}
		SparkGlm52ValExpertGemmRow(fx->experts.w2_payload,fx->experts.w2_scales,expert,
			oracle->intermediate,expert_row,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS,SPARK_GLM52_VHIDDEN);
		/* LmMoeFinalizeKernel: fp32 weighted sum of the bf16 packed rows. */
		for (element = 0u; element < SPARK_GLM52_VHIDDEN; element++)
			oracle->hidden[element] += out_weights[route] * expert_row[element];
	}
	for (element = 0u; element < SPARK_GLM52_VHIDDEN; element++)
		oracle->hidden[element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(oracle->hidden[element]));
	/* Shared expert: same fused stack, added ungated. */
	SparkGlm52ValGemmRow(oracle->normed,fx->shared_gate_up.host,oracle->gate_up,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VW1_ROWS);
	for (index = 0u; index < SPARK_GLM52_VEXPERT_INTER; index++)
	{
		gate = oracle->gate_up[SPARK_GLM52_VEXPERT_INTER + index];
		up = oracle->gate_up[index];
		oracle->intermediate[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16((gate / (1.0f + expf(-gate))) * up));
	}
	SparkGlm52ValGemmRow(oracle->intermediate,fx->shared_down.host,oracle->attention_value,SPARK_GLM52_VEXPERT_INTER,SPARK_GLM52_VHIDDEN);
	for (element = 0u; element < SPARK_GLM52_VHIDDEN; element++)
		oracle->hidden[element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(
			SparkGlm52ValFromBf16(SparkGlm52ValBf16(oracle->hidden[element])) +
			oracle->attention_value[element]));
}

/* Full dense-layer token (tier 1). */
static void SparkGlm52ValOracleToken(SparkGlm52ValOracle *oracle,const SparkGlm52ValFixture *fixture,uint32_t token,uint32_t position)
{
	SparkGlm52ValOracleAttentionChunk(oracle,fixture,token,position,0,0u);
	SparkGlm52ValOracleDenseMlp(oracle,fixture);
}

/* -- tier drivers ------------------------------------------------------------ */

static int SparkGlm52ValCompareStreams(SparkGlm52ValFixture *fixture,SparkGlm52ValOracle *oracle,
	const char *label,int *result)
{
	SparkGlm52ValMetrics metrics;
	uint16_t *device_hidden,*device_residual;
	float run_hidden[SPARK_GLM52_VHIDDEN],run_residual[SPARK_GLM52_VHIDDEN];
	float actual[SPARK_GLM52_VHIDDEN],reference[SPARK_GLM52_VHIDDEN];
	uint32_t index;
	int local_result;
	device_hidden = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	device_residual = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	if ( device_hidden == 0 || device_residual == 0 )
		return(SparkGlm52ValFail("compare","host_alloc"));
	if ( cudaMemcpy(device_hidden,fixture->hidden,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(device_residual,fixture->residual,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
	{
		free(device_hidden);
		free(device_residual);
		return(SparkGlm52ValFail("compare","readback"));
	}
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		run_hidden[index] = SparkGlm52ValFromBf16(device_hidden[index]);
		run_residual[index] = SparkGlm52ValFromBf16(device_residual[index]);
	}
	free(device_hidden);
	free(device_residual);
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		actual[index] = run_hidden[index];
		reference[index] = oracle->hidden[index];
	}
	SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
	local_result = SparkGlm52ValReport(label,&metrics,2e-2,0.999);
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		actual[index] = run_residual[index];
		reference[index] = oracle->residual[index];
	}
	SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
	if ( local_result == 0 )
	{
		char residual_label[128];
		snprintf(residual_label,sizeof(residual_label),"%s_residual",label);
		local_result = SparkGlm52ValReport(residual_label,&metrics,2e-2,0.999);
	}
	if ( result != 0 && local_result != 0 )
		*result = local_result;
	return(local_result);
}

#ifndef SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST

/* Tier 1: the original dense-layer walk, unchanged. */
static int SparkGlm52ValRunDenseTier(SparkGlm52ValFixture *fixture,int *result)
{
	SparkGlm52ValOracle oracle;
	static const uint32_t tokens[SPARK_GLM52_VALIDATION_TOKENS] = {1u,3u,2u,6u};
	uint16_t *device_hidden,*device_residual;
	float run_hidden[SPARK_GLM52_VHIDDEN],run_residual[SPARK_GLM52_VHIDDEN];
	uint32_t step,index,mismatch = 0;
	int local_result;
	if ( SparkGlm52ValRunWalk(fixture) != 0 )
		return(1);
	if ( SparkGlm52ValCheckAccessError(fixture) != 0 )
		return(1);
	device_hidden = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	device_residual = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	if ( device_hidden == 0 || device_residual == 0 )
		return(SparkGlm52ValFail("compare","host_alloc"));
	if ( cudaMemcpy(device_hidden,fixture->hidden,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(device_residual,fixture->residual,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(SparkGlm52ValFail("compare","hidden_readback"));
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		run_hidden[index] = SparkGlm52ValFromBf16(device_hidden[index]);
		run_residual[index] = SparkGlm52ValFromBf16(device_residual[index]);
	}
	memset(&oracle,0,sizeof(oracle));
	oracle.fixture = fixture;
	for (step = 0u; step < SPARK_GLM52_VALIDATION_TOKENS; step++)
		SparkGlm52ValOracleToken(&oracle,fixture,tokens[step],step);
	{
		SparkGlm52ValMetrics metrics;
		float actual[SPARK_GLM52_VHIDDEN],reference[SPARK_GLM52_VHIDDEN];
		for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
		{
			actual[index] = run_hidden[index];
			reference[index] = oracle.hidden[index];
		}
		SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
		local_result = SparkGlm52ValReport("layer_forward_hidden",&metrics,2e-2,0.999);
		for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
		{
			actual[index] = run_residual[index];
			reference[index] = oracle.residual[index];
		}
		SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
		if ( local_result == 0 )
			local_result = SparkGlm52ValReport("layer_forward_residual",&metrics,2e-2,0.999);
		/* Determinism: the identical walk on the identical fixture must
		 * reproduce the committed streams bit for bit - the property every
		 * replay/restore argument downstream borrows from the chain. */
		if ( local_result == 0 )
		{
			if ( SparkGlm52ValRunWalk(fixture) != 0 )
				return(1);
			if ( cudaMemcpy(device_hidden,fixture->hidden,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
				cudaMemcpy(device_residual,fixture->residual,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
				return(SparkGlm52ValFail("determinism","rerun_readback"));
			for (index = 0u; index < SPARK_GLM52_VHIDDEN && mismatch == 0; index++)
			{
				if ( device_hidden[index] != SparkGlm52ValBf16(run_hidden[index]) )
					mismatch = 1;
				if ( device_residual[index] != SparkGlm52ValBf16(run_residual[index]) )
					mismatch = 1;
			}
			printf("glm52_validation check=determinism elements=%u bit_exact=%d\n",
				(unsigned)(2u * SPARK_GLM52_VHIDDEN),mismatch == 0 ? 1 : 0);
			if ( mismatch != 0 )
				local_result = SparkGlm52ValFail("determinism","repeat_walk_mismatch");
		}
	}
	free(device_hidden);
	free(device_residual);
	if ( local_result != 0 && result != 0 )
		*result = local_result;
	return(local_result);
}

/* Tier 2a: the first routed layer - attention chunk, routed MoE chunk with
 * the compiled package codec, shared expert, plus selection determinism. */
static int SparkGlm52ValRunRoutedTier(SparkGlm52ValFixture *fixture,int *result)
{
	SparkGlm52ValOracle oracle;
	static const uint32_t tokens[2] = {5u,2u};
	uint32_t selected[SPARK_GLM52_VTOP_K];
	float weights[SPARK_GLM52_VTOP_K];
	uint32_t device_selected[SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VTOP_K];
	float device_weights[SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VTOP_K];
	float expert_reference[SPARK_GLM52_VHIDDEN];
	float actual[SPARK_GLM52_VHIDDEN],reference[SPARK_GLM52_VHIDDEN];
	SparkGlm52ValMetrics metrics;
	uint32_t token,route,index;
	double maximum_weight_delta = 0.0;
	int status,local_result = 0;
	uint32_t packed_row_zero = 0u;
	uint16_t *expert_row_zero = 0u;
	if ( SparkGlm52ValResetStreams(fixture) != 0 )
		return(1);
	/* Walk two tokens through global layer GLM52_FIRST_ROUTED_LAYER. Each
	 * wave is ONE row, so every step overwrites the route arrays and the
	 * packed expert rows: capture this step's evidence before advancing. */
	expert_row_zero = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	if ( expert_row_zero == 0 )
		return(SparkGlm52ValFail("routed_expert","host_alloc"));
	for (token = 0u; token < 2u; token++)
	{
		SparkGlm52ValBuildWave(fixture,SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER,tokens[token],token);
		if ( SparkGlm52LaunchCudaWaveBegin(&fixture->wave) != 0 )
		{
			free(expert_row_zero);
			return(SparkGlm52ValFail("routed_wave_begin","status"));
		}
		if ( SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 ||
			SparkGlm52LaunchCudaLayerMlp(&fixture->wave,0u) != 0 )
		{
			free(expert_row_zero);
			return(SparkGlm52ValFail("routed_walk","status"));
		}
		if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
		{
			free(expert_row_zero);
			return(SparkGlm52ValFail("routed_walk","sync"));
		}
		if ( cudaMemcpy(&device_selected[token * SPARK_GLM52_VTOP_K],fixture->route_expert,
			SPARK_GLM52_VTOP_K * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(&device_weights[token * SPARK_GLM52_VTOP_K],fixture->route_weight,
			SPARK_GLM52_VTOP_K * sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess )
		{
			free(expert_row_zero);
			return(SparkGlm52ValFail("routed_selection","route_readback"));
		}
		if ( token == 0u )
		{
			/* Packed row of route 0 and its w2 output, before step two
			 * reuses the buffers. */
			if ( cudaMemcpy(&packed_row_zero,fixture->route_packed_row,sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
				cudaMemcpy(expert_row_zero,fixture->expert_out + (uint64_t)packed_row_zero * SPARK_GLM52_VHIDDEN,
				SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
			{
				free(expert_row_zero);
				return(SparkGlm52ValFail("routed_expert","readback"));
			}
		}
	}
	if ( SparkGlm52ValCheckAccessError(fixture) != 0 )
	{
		free(expert_row_zero);
		return(1);
	}
	memset(&oracle,0,sizeof(oracle));
	oracle.fixture = fixture;
	for (token = 0u; token < 2u; token++)
	{
		SparkGlm52ValOracleAttentionChunk(&oracle,fixture,tokens[token],token,0,0u);
		SparkGlm52ValOracleRoutedMlp(&oracle,fixture,selected,weights);
		printf("glm52_validation check=routed_selection token=%u experts=%u,%u,%u,%u,%u,%u,%u,%u\n",
			token,device_selected[token * 8u + 0u],device_selected[token * 8u + 1u],
			device_selected[token * 8u + 2u],device_selected[token * 8u + 3u],
			device_selected[token * 8u + 4u],device_selected[token * 8u + 5u],
			device_selected[token * 8u + 6u],device_selected[token * 8u + 7u]);
		/* Set equality with the oracle's top-k. The bias pins experts 0..7
		 * with sigmoid-margin many orders above the __expf/expf delta. */
		for (route = 0u; route < SPARK_GLM52_VTOP_K; route++)
		{
			uint32_t found = 0u;
			for (index = 0u; index < SPARK_GLM52_VTOP_K; index++)
				if ( device_selected[token * 8u + index] == selected[route] )
					found = 1u;
			if ( found == 0u )
			{
				free(expert_row_zero);
				return(SparkGlm52ValFail("routed_selection","missing_expected_expert"));
			}
		}
		for (route = 0u; route < SPARK_GLM52_VTOP_K; route++)
			for (index = 0u; index < SPARK_GLM52_VTOP_K; index++)
				if ( device_selected[token * 8u + index] == selected[route] )
				{
					double delta = fabs((double)device_weights[token * 8u + index] - (double)weights[route]);
					if ( delta > 2.0e-3 )
					{
						free(expert_row_zero);
						return(SparkGlm52ValFail("routed_selection","weight_delta"));
					}
					if ( delta > maximum_weight_delta )
						maximum_weight_delta = delta;
				}
	}
	printf("glm52_validation check=routed_selection_set elements=%u exact=1 max_weight_delta=%.6g\n",
		(unsigned)(2u * SPARK_GLM52_VTOP_K),maximum_weight_delta);
	/* One routed expert's forward, element-wise: packed row of route 0 of
	 * the first step, captured above. The oracle rebuild: RoutedMlp leaves
	 * oracle.normed holding the token's mlp-normed activation, so the pinned
	 * expert's forward recomputes from it directly. */
	{
		memset(&oracle,0,sizeof(oracle));
		oracle.fixture = fixture;
		SparkGlm52ValOracleAttentionChunk(&oracle,fixture,tokens[0],0,0,0u);
		SparkGlm52ValOracleRoutedMlp(&oracle,fixture,selected,weights);
		SparkGlm52ValExpertGemmRow(fixture->experts.w1_payload,fixture->experts.w1_scales,selected[0],
			oracle.normed,oracle.gate_up,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS,SPARK_GLM52_VW1_ROWS);
		for (index = 0u; index < SPARK_GLM52_VEXPERT_INTER; index++)
		{
			float gate = oracle.gate_up[SPARK_GLM52_VEXPERT_INTER + index];
			float up = oracle.gate_up[index];
			oracle.intermediate[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16((gate / (1.0f + expf(-gate))) * up));
		}
		SparkGlm52ValExpertGemmRow(fixture->experts.w2_payload,fixture->experts.w2_scales,selected[0],
			oracle.intermediate,expert_reference,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS,SPARK_GLM52_VHIDDEN);
		for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
		{
			actual[index] = SparkGlm52ValFromBf16(expert_row_zero[index]);
			reference[index] = expert_reference[index];
		}
		SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
		status = SparkGlm52ValReport("routed_expert_forward",&metrics,2e-2,0.999);
		free(expert_row_zero);
		expert_row_zero = 0u;
		if ( status != 0 )
		{
			if ( result != 0 )
				*result = status;
			return(status);
		}
	}
	/* Whole-layer streams vs the oracle. */
	memset(&oracle,0,sizeof(oracle));
	oracle.fixture = fixture;
	{
		uint32_t oracle_selected[SPARK_GLM52_VTOP_K];
		float oracle_weights[SPARK_GLM52_VTOP_K];
		for (token = 0u; token < 2u; token++)
		{
			SparkGlm52ValOracleAttentionChunk(&oracle,fixture,tokens[token],token,0,0u);
			SparkGlm52ValOracleRoutedMlp(&oracle,fixture,oracle_selected,oracle_weights);
		}
	}
	status = SparkGlm52ValCompareStreams(fixture,&oracle,"routed_layer_forward",&local_result);
	if ( status != 0 )
		return(status);
	/* Determinism: the whole leg again; route scatter races are
	 * value-transparent, so the committed streams must repeat bit for bit. */
	if ( SparkGlm52ValResetStreams(fixture) != 0 )
		return(1);
	{
		uint16_t *first_hidden = (uint16_t *)malloc(2u * SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
		uint32_t first_selected[SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VTOP_K];
		float first_weights[SPARK_GLM52_VALIDATION_ROUTES * SPARK_GLM52_VTOP_K];
		uint32_t second_selected[sizeof(first_selected) / sizeof(uint32_t)];
		float second_weights[sizeof(first_weights) / sizeof(float)];
		if ( first_hidden == 0 )
			return(SparkGlm52ValFail("routed_determinism","host_alloc"));
		for (token = 0u; token < 2u; token++)
		{
			SparkGlm52ValBuildWave(fixture,SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER,tokens[token],token);
			if ( SparkGlm52LaunchCudaWaveBegin(&fixture->wave) != 0 ||
				SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 ||
				SparkGlm52LaunchCudaLayerMlp(&fixture->wave,0u) != 0 )
			{
				free(first_hidden);
				return(SparkGlm52ValFail("routed_determinism","status"));
			}
			if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
			{
				free(first_hidden);
				return(SparkGlm52ValFail("routed_determinism","sync"));
			}
		}
		if ( cudaMemcpy(first_hidden,fixture->hidden,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(first_selected,fixture->route_expert,sizeof(first_selected),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(first_weights,fixture->route_weight,sizeof(first_weights),cudaMemcpyDeviceToHost) != cudaSuccess )
		{
			free(first_hidden);
			return(SparkGlm52ValFail("routed_determinism","readback"));
		}
		if ( SparkGlm52ValResetStreams(fixture) != 0 )
		{
			free(first_hidden);
			return(1);
		}
		for (token = 0u; token < 2u; token++)
		{
			SparkGlm52ValBuildWave(fixture,SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER,tokens[token],token);
			if ( SparkGlm52LaunchCudaWaveBegin(&fixture->wave) != 0 ||
				SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 ||
				SparkGlm52LaunchCudaLayerMlp(&fixture->wave,0u) != 0 )
			{
				free(first_hidden);
				return(SparkGlm52ValFail("routed_determinism","status"));
			}
			if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
			{
				free(first_hidden);
				return(SparkGlm52ValFail("routed_determinism","sync"));
			}
		}
		{
			uint32_t mismatch = 0u;
			if ( cudaMemcpy(second_selected,fixture->route_expert,sizeof(second_selected),cudaMemcpyDeviceToHost) != cudaSuccess ||
				cudaMemcpy(second_weights,fixture->route_weight,sizeof(second_weights),cudaMemcpyDeviceToHost) != cudaSuccess )
			{
				free(first_hidden);
				return(SparkGlm52ValFail("routed_determinism","readback"));
			}
			for (index = 0u; index < (uint32_t)(sizeof(first_selected) / sizeof(uint32_t)); index++)
				if ( first_selected[index] != second_selected[index] ||
					memcmp(&first_weights[index],&second_weights[index],sizeof(float)) != 0 )
					mismatch = 1u;
			/* Compare the committed hidden stream bit for bit. */
			{
				uint16_t *second_hidden = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
				if ( second_hidden == 0 )
				{
					free(first_hidden);
					return(SparkGlm52ValFail("routed_determinism","host_alloc"));
				}
				if ( cudaMemcpy(second_hidden,fixture->hidden,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
				{
					free(first_hidden);
					free(second_hidden);
					return(SparkGlm52ValFail("routed_determinism","readback"));
				}
				for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
					if ( first_hidden[index] != second_hidden[index] )
						mismatch = 1u;
				free(second_hidden);
			}
			printf("glm52_validation check=routed_determinism routes=%u bit_exact=%d\n",
				(unsigned)(2u * SPARK_GLM52_VTOP_K),mismatch == 0u ? 1 : 0);
			free(first_hidden);
			if ( mismatch != 0u )
			{
				status = SparkGlm52ValFail("routed_determinism","repeat_mismatch");
				if ( result != 0 )
					*result = status;
				return(status);
			}
		}
	}
	return(0);
}

/* Tier 2b: DSA/indexer at context beyond the selection width. */
static int SparkGlm52ValRunDsaTier(SparkGlm52ValFixture *fixture,int *result)
{
	SparkGlm52ValOracle oracle;
	SparkGlm52ValDsaShape shape;
	static const uint32_t real_positions[SPARK_GLM52_VALIDATION_REAL_SLOTS] = {0u,704u,1408u};
	/* Embedding fixture holds eight rows: token ids stay inside it. */
	static const uint32_t real_tokens[SPARK_GLM52_VALIDATION_REAL_SLOTS] = {2u,5u,3u};
	uint32_t device_selected[SPARK_GLM52_VDSA_SELECTED];
	uint32_t sorted_device[SPARK_GLM52_VDSA_SELECTED];
	uint32_t sorted_oracle[SPARK_GLM52_VDSA_SELECTED];
	uint16_t *attention_device;
	float actual[SPARK_GLM52_VHIDDEN],reference[SPARK_GLM52_VHIDDEN];
	SparkGlm52ValMetrics metrics;
	float centroid[SPARK_GLM52_VDSA_DIM];
	uint32_t slot,index,head,dimension;
	int status;
	if ( SparkGlm52ValResetStreams(fixture) != 0 )
		return(1);
	/* Produce real main-KV latents at three spread positions through the
	 * production chain (dense mode: context fits the selection width). */
	for (slot = 0u; slot < SPARK_GLM52_VALIDATION_REAL_SLOTS; slot++)
	{
		SparkGlm52ValBuildWave(fixture,0u,real_tokens[slot],real_positions[slot]);
		if ( SparkGlm52LaunchCudaWaveBegin(&fixture->wave) != 0 ||
			SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 )
			return(SparkGlm52ValFail("dsa_populate","status"));
		if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
			return(SparkGlm52ValFail("dsa_populate","sync"));
	}
	if ( SparkGlm52ValCheckAccessError(fixture) != 0 )
		return(1);
	/* The populated pool stands; only the streams are re-driven below. The
	 * sparse step's oracle chain, up through the indexer outputs the cache
	 * shaping consumes. */
	memset(&oracle,0,sizeof(oracle));
	oracle.fixture = fixture;
	{
		float sum,pair_low,pair_high,angle;
		for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
		{
			oracle.hidden[index] = fixture->embedding.host[((uint64_t)7u * SPARK_GLM52_VHIDDEN) + index];
			oracle.residual[index] = 0.0f;
		}
		for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
		{
			sum = oracle.hidden[index] + oracle.residual[index];
			oracle.residual[index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
			oracle.hidden[index] = sum;
		}
		SparkGlm52ValRmsNorm(oracle.hidden,fixture->attn_norm.host,oracle.normed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
		SparkGlm52ValGemmRow(oracle.normed,fixture->q_a.host,oracle.q_compressed,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VQUERY_A);
		SparkGlm52ValRmsNorm(oracle.q_compressed,fixture->q_a_norm.host,oracle.q_compressed,SPARK_GLM52_VQUERY_A,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
		SparkGlm52ValGemmRow(oracle.q_compressed,fixture->index_q.host,oracle.index_query,SPARK_GLM52_VQUERY_A,SPARK_GLM52_VDSA_QUERY_DIM);
		SparkGlm52ValGemmRow(oracle.normed,fixture->index_k.host,oracle.index_key,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VDSA_DIM);
		SparkGlm52ValLayerNorm(oracle.index_key,fixture->index_norm_weight.host,fixture->index_norm_bias.host,oracle.index_key,
			SPARK_GLM52_VDSA_DIM,SPARK_GLM52_MODEL_DSA_INDEX_NORM_EPSILON);
		SparkGlm52ValGemmRow(oracle.normed,fixture->index_head.host,oracle.index_heads,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VDSA_HEADS);
		for (head = 0u; head < SPARK_GLM52_VDSA_HEADS; head++)
			for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
			{
				pair_low = oracle.index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index)];
				pair_high = oracle.index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index + 1u)];
				angle = (float)SPARK_GLM52_VALIDATION_DSA_POSITION * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
				SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
				oracle.index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index)] =
					SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
				oracle.index_query[(head * SPARK_GLM52_VDSA_DIM) + (2u * index + 1u)] =
					SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
			}
		for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
		{
			pair_low = oracle.index_key[2u * index];
			pair_high = oracle.index_key[2u * index + 1u];
			angle = (float)SPARK_GLM52_VALIDATION_DSA_POSITION * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
			SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
			oracle.index_key[2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
			oracle.index_key[2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
		}
	}
	/* Centroid of the roped index query under the learned head mixture. */
	for (dimension = 0u; dimension < SPARK_GLM52_VDSA_DIM; dimension++)
		centroid[dimension] = 0.0f;
	for (head = 0u; head < SPARK_GLM52_VDSA_HEADS; head++)
		for (dimension = 0u; dimension < SPARK_GLM52_VDSA_DIM; dimension++)
			centroid[dimension] += oracle.index_heads[head] * oracle.index_query[(head * SPARK_GLM52_VDSA_DIM) + dimension];
	/* Shape the cached index keys and verify the selection is FORCED. */
	if ( SparkGlm52ValShapeIndexCache(fixture->index_keys_host,SPARK_GLM52_VALIDATION_DSA_CONTEXT,SPARK_GLM52_VDSA_SELECTED,
		centroid,SPARK_GLM52_VDSA_DIM,100.0f,0.05f,&shape) != 0 )
		return(SparkGlm52ValFail("dsa_shaping","not_separable"));
	printf("glm52_validation check=dsa_shaping top_min=%.9g tail_max=%.9g bucket=%u\n",
		shape.top_minimum,shape.tail_maximum,shape.top_bucket);
	/* The sparse step's OWN key (the production store at its position) must
	 * also land below the shaped top bucket, or it would join the selection
	 * and the forced-set prediction would not hold. Fail loudly rather than
	 * compare against a wrong oracle set. */
	{
		float own_score = SparkGlm52ValDot(oracle.index_key,centroid,SPARK_GLM52_VDSA_DIM) *
			SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE / sqrtf((float)SPARK_GLM52_VDSA_HEADS);
		if ( SparkGlm52ValTopkBucket(own_score) >= shape.top_bucket )
			return(SparkGlm52ValFail("dsa_shaping","own_position_not_below_threshold"));
	}
	/* Upload the shaped keys into the index cache (slot-major pages). */
	for (slot = 0u; slot < SPARK_GLM52_VALIDATION_DSA_CONTEXT; slot++)
	{
		uint64_t byte = ((uint64_t)slot / SPARK_GLM52_VKV_PAGE_SLOTS) * SPARK_GLM52_VINDEX_PAGE_BYTES +
			(uint64_t)(slot % SPARK_GLM52_VKV_PAGE_SLOTS) * SPARK_GLM52_VINDEX_SLOT_BYTES;
		if ( cudaMemcpy(fixture->index_cache + byte,fixture->index_keys_host + (uint64_t)slot * SPARK_GLM52_VDSA_DIM,
			SPARK_GLM52_VDSA_DIM * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkGlm52ValFail("dsa_shaping","index_upload"));
	}
	/* The sparse production step: full-indexer layer 0, context 2065. */
	SparkGlm52ValBuildWave(fixture,0u,7u,SPARK_GLM52_VALIDATION_DSA_POSITION);
	if ( SparkGlm52LaunchCudaWaveBegin(&fixture->wave) != 0 )
		return(SparkGlm52ValFail("dsa_wave_begin","status"));
	if ( SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 )
		return(SparkGlm52ValFail("dsa_attention","status"));
	if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
		return(SparkGlm52ValFail("dsa_walk","sync"));
	if ( SparkGlm52ValCheckAccessError(fixture) != 0 )
		return(1);
	/* Selected set: sorted equality with the oracle's forced prediction. */
	if ( cudaMemcpy(device_selected,fixture->selected_positions,SPARK_GLM52_VDSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(SparkGlm52ValFail("dsa_selection","readback"));
	memcpy(sorted_device,device_selected,sizeof(sorted_device));
	memcpy(sorted_oracle,shape.selected_positions,sizeof(sorted_oracle));
	for (index = 0u; index < SPARK_GLM52_VDSA_SELECTED; index++)
		for (dimension = index + 1u; dimension < SPARK_GLM52_VDSA_SELECTED; dimension++)
		{
			uint32_t swap;
			if ( sorted_device[index] > sorted_device[dimension] )
			{
				swap = sorted_device[index];
				sorted_device[index] = sorted_device[dimension];
				sorted_device[dimension] = swap;
			}
			if ( sorted_oracle[index] > sorted_oracle[dimension] )
			{
				swap = sorted_oracle[index];
				sorted_oracle[index] = sorted_oracle[dimension];
				sorted_oracle[dimension] = swap;
			}
		}
	if ( memcmp(sorted_device,sorted_oracle,sizeof(sorted_device)) != 0 )
		return(SparkGlm52ValFail("dsa_selection","set_mismatch"));
	printf("glm52_validation check=dsa_selection elements=%u exact=1\n",(unsigned)SPARK_GLM52_VDSA_SELECTED);
	/* Attention restricted to the selection vs the oracle attending it. */
	attention_device = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
	if ( attention_device == 0 )
		return(SparkGlm52ValFail("dsa_compare","host_alloc"));
	if ( cudaMemcpy(attention_device,fixture->attention_out,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
	{
		free(attention_device);
		return(SparkGlm52ValFail("dsa_compare","readback"));
	}
	{
		float sum,pair_low,pair_high,angle,score;
		float scores[SPARK_GLM52_VDSA_SELECTED];
		float maximum,running_sum;
		const float *slot_pointer;
		uint32_t element,step;
		/* Finish the oracle chain: q/kv projections, store at the sparse
		 * position, attention over the selection. */
		SparkGlm52ValGemmRow(oracle.q_compressed,fixture->q_b.host,oracle.q_row,SPARK_GLM52_VQUERY_A,SPARK_GLM52_VQ_ROWS);
		SparkGlm52ValGemmRow(oracle.normed,fixture->kv_a.host,oracle.kv_slot,SPARK_GLM52_VHIDDEN,SPARK_GLM52_VKV_SLOT_ELEMENTS);
		SparkGlm52ValRmsNorm(oracle.kv_slot,fixture->kv_a_norm.host,oracle.kv_slot,SPARK_GLM52_VLATENT,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
		for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
			for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
			{
				pair_low = oracle.q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + (SPARK_GLM52_VQK_NOPE + 2u * index)];
				pair_high = oracle.q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + (SPARK_GLM52_VQK_NOPE + 2u * index + 1u)];
				angle = (float)SPARK_GLM52_VALIDATION_DSA_POSITION * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
				SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
				oracle.q_rope[(head * SPARK_GLM52_VROPE) + 2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
				oracle.q_rope[(head * SPARK_GLM52_VROPE) + 2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
			}
		for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
			for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
			{
				sum = 0.0f;
				for (index = 0u; index < SPARK_GLM52_VQK_NOPE; index++)
					sum += oracle.q_row[(head * (SPARK_GLM52_VQK_NOPE + SPARK_GLM52_VROPE)) + index] *
						fixture->kv_b_key.host[((uint64_t)head * SPARK_GLM52_VLATENT + element) * SPARK_GLM52_VQK_NOPE + index];
				oracle.query_latent[(head * SPARK_GLM52_VLATENT) + element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
			}
		for (index = 0u; index < SPARK_GLM52_VROPE / 2u; index++)
		{
			pair_low = oracle.kv_slot[SPARK_GLM52_VLATENT + 2u * index];
			pair_high = oracle.kv_slot[SPARK_GLM52_VLATENT + 2u * index + 1u];
			angle = (float)SPARK_GLM52_VALIDATION_DSA_POSITION * powf(SPARK_GLM52_MODEL_ROPE_THETA,-2.0f * (float)index / (float)SPARK_GLM52_VROPE);
			SparkGlm52ValRotatePair(&pair_low,&pair_high,angle);
			oracle.kv_slot[SPARK_GLM52_VLATENT + 2u * index] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_low));
			oracle.kv_slot[SPARK_GLM52_VLATENT + 2u * index + 1u] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(pair_high));
		}
		/* Real latents the populate steps left at three positions. */
		for (slot = 0u; slot < SPARK_GLM52_VALIDATION_REAL_SLOTS; slot++)
		{
			SparkGlm52ValOracle populated;
			memset(&populated,0,sizeof(populated));
			populated.fixture = fixture;
			SparkGlm52ValOracleAttentionChunk(&populated,fixture,real_tokens[slot],real_positions[slot],0,0u);
			SparkGlm52ValCachePut(&oracle,real_positions[slot],populated.kv_slot);
		}
		for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
		{
			maximum = -3.0e38f;
			running_sum = 0.0f;
			for (step = 0u; step < SPARK_GLM52_VDSA_SELECTED; step++)
			{
				uint32_t position = shape.selected_positions[step];
				slot_pointer = SparkGlm52ValCacheGet(&oracle,position);
				score = 0.0f;
				if ( slot_pointer != 0 )
				{
					for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
						score += oracle.query_latent[(head * SPARK_GLM52_VLATENT) + element] * slot_pointer[element];
					for (element = 0u; element < SPARK_GLM52_VROPE; element++)
						score += oracle.q_rope[(head * SPARK_GLM52_VROPE) + element] * slot_pointer[SPARK_GLM52_VLATENT + element];
				}
				scores[step] = score * SPARK_GLM52_MODEL_QK_SCALE;
				if ( scores[step] > maximum )
					maximum = scores[step];
			}
			for (step = 0u; step < SPARK_GLM52_VDSA_SELECTED; step++)
				running_sum += expf(scores[step] - maximum);
			if ( running_sum < 1.0e-20f )
				running_sum = 1.0e-20f;
			for (element = 0u; element < SPARK_GLM52_VLATENT; element++)
			{
				float value = 0.0f;
				for (step = 0u; step < SPARK_GLM52_VDSA_SELECTED; step++)
				{
					uint32_t position = shape.selected_positions[step];
					slot_pointer = SparkGlm52ValCacheGet(&oracle,position);
					if ( slot_pointer != 0 )
						value += expf(scores[step] - maximum) * slot_pointer[element];
				}
				oracle.attention_latent[(head * SPARK_GLM52_VLATENT) + element] =
					SparkGlm52ValFromBf16(SparkGlm52ValBf16(value / running_sum));
			}
		}
		for (head = 0u; head < SPARK_GLM52_VHEADS; head++)
			for (element = 0u; element < SPARK_GLM52_VVALUE_DIM; element++)
			{
				sum = 0.0f;
				for (index = 0u; index < SPARK_GLM52_VLATENT; index++)
					sum += oracle.attention_latent[(head * SPARK_GLM52_VLATENT) + index] *
						fixture->kv_b_value.host[((uint64_t)head * SPARK_GLM52_VVALUE_DIM + element) * SPARK_GLM52_VLATENT + index];
				oracle.attention_value[(head * SPARK_GLM52_VVALUE_DIM) + element] = SparkGlm52ValFromBf16(SparkGlm52ValBf16(sum));
			}
		SparkGlm52ValGemmRow(oracle.attention_value,fixture->attn_output.host,oracle.attention_out,SPARK_GLM52_VATTN_COLS,SPARK_GLM52_VHIDDEN);
	}
	for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
	{
		actual[index] = SparkGlm52ValFromBf16(attention_device[index]);
		reference[index] = oracle.attention_out[index];
	}
	SparkGlm52ValMeasure(&metrics,actual,reference,SPARK_GLM52_VHIDDEN);
	status = SparkGlm52ValReport("dsa_sparse_attention",&metrics,2e-2,0.999);
	if ( status != 0 )
	{
		free(attention_device);
		if ( result != 0 )
			*result = status;
		return(status);
	}
	/* Re-run: selection SET repeats exactly, output within summation-order
	 * noise. Bit-exactness is deliberately not claimed here - the gather's
	 * within-bucket emission order is unsynchronized by design. */
	if ( SparkGlm52ValResetStreams(fixture) != 0 )
	{
		free(attention_device);
		return(1);
	}
	for (slot = 0u; slot < SPARK_GLM52_VALIDATION_REAL_SLOTS; slot++)
	{
		SparkGlm52ValBuildWave(fixture,0u,real_tokens[slot],real_positions[slot]);
		if ( SparkGlm52LaunchCudaWaveBegin(&fixture->wave) != 0 ||
			SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 )
		{
			free(attention_device);
			return(SparkGlm52ValFail("dsa_rerun","status"));
		}
		if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
		{
			free(attention_device);
			return(SparkGlm52ValFail("dsa_rerun","sync"));
		}
	}
	for (slot = 0u; slot < SPARK_GLM52_VALIDATION_DSA_CONTEXT; slot++)
	{
		uint64_t byte = ((uint64_t)slot / SPARK_GLM52_VKV_PAGE_SLOTS) * SPARK_GLM52_VINDEX_PAGE_BYTES +
			(uint64_t)(slot % SPARK_GLM52_VKV_PAGE_SLOTS) * SPARK_GLM52_VINDEX_SLOT_BYTES;
		if ( cudaMemcpy(fixture->index_cache + byte,fixture->index_keys_host + (uint64_t)slot * SPARK_GLM52_VDSA_DIM,
			SPARK_GLM52_VDSA_DIM * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkGlm52ValFail("dsa_rerun","index_upload"));
	}
	SparkGlm52ValBuildWave(fixture,0u,7u,SPARK_GLM52_VALIDATION_DSA_POSITION);
	if ( SparkGlm52LaunchCudaWaveBegin(&fixture->wave) != 0 ||
		SparkGlm52LaunchCudaLayerAttention(&fixture->wave,0u) != 0 )
		return(SparkGlm52ValFail("dsa_rerun","status"));
	if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
		return(SparkGlm52ValFail("dsa_rerun","sync"));
	{
		uint32_t second_selected[SPARK_GLM52_VDSA_SELECTED];
		uint32_t mismatch = 0u;
		uint16_t *second_bits;
		if ( SparkGlm52ValCheckAccessError(fixture) != 0 )
			return(1);
		if ( cudaMemcpy(second_selected,fixture->selected_positions,SPARK_GLM52_VDSA_SELECTED * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess )
			return(SparkGlm52ValFail("dsa_rerun","readback"));
		second_bits = (uint16_t *)malloc(SPARK_GLM52_VHIDDEN * sizeof(uint16_t));
		if ( second_bits == 0 )
			return(SparkGlm52ValFail("dsa_rerun","host_alloc"));
		if ( cudaMemcpy(second_bits,fixture->attention_out,SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
		{
			free(second_bits);
			return(SparkGlm52ValFail("dsa_rerun","readback"));
		}
		memcpy(sorted_device,second_selected,sizeof(sorted_device));
		for (index = 0u; index < SPARK_GLM52_VDSA_SELECTED; index++)
			for (dimension = index + 1u; dimension < SPARK_GLM52_VDSA_SELECTED; dimension++)
				if ( sorted_device[index] > sorted_device[dimension] )
				{
					uint32_t swap = sorted_device[index];
					sorted_device[index] = sorted_device[dimension];
					sorted_device[dimension] = swap;
				}
		if ( memcmp(sorted_device,sorted_oracle,sizeof(sorted_device)) != 0 )
			mismatch = 1u;
		for (index = 0u; index < SPARK_GLM52_VHIDDEN; index++)
		{
			float first_value = SparkGlm52ValFromBf16(attention_device[index]);
			float second_value = SparkGlm52ValFromBf16(second_bits[index]);
			if ( fabs((double)first_value - (double)second_value) >
				1.0e-3 * (1.0 + fabs((double)first_value)) )
				mismatch = 1u;
		}
		printf("glm52_validation check=dsa_rerun elements=%u set_and_tolerance_exact=%d\n",
			(unsigned)(SPARK_GLM52_VDSA_SELECTED + SPARK_GLM52_VHIDDEN),mismatch == 0u ? 1 : 0);
		free(second_bits);
		free(attention_device);
		if ( mismatch != 0u )
		{
			status = SparkGlm52ValFail("dsa_rerun","mismatch");
			if ( result != 0 )
				*result = status;
			return(status);
		}
	}
	return(0);
}

int main(int argc,char **argv)
{
	SparkGlm52ValFixture fixture;
	int result = 0;
	if ( argc != 2 || strlen(argv[1]) != 64u )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	memset(&fixture,0,sizeof(fixture));
	if ( SparkGlm52ValFixtureSetup(&fixture) != 0 )
		return(1);
	if ( SparkGlm52ValRunDenseTier(&fixture,&result) != 0 )
	{
		SparkGlm52ValFixtureDestroy(&fixture);
		return(result != 0 ? result : 1);
	}
	printf("glm52_validation tier=dense PASS\n");
	if ( SparkGlm52ValRunRoutedTier(&fixture,&result) != 0 )
	{
		SparkGlm52ValFixtureDestroy(&fixture);
		return(result != 0 ? result : 1);
	}
	printf("glm52_validation tier=routed PASS\n");
	if ( SparkGlm52ValRunDsaTier(&fixture,&result) != 0 )
	{
		SparkGlm52ValFixtureDestroy(&fixture);
		return(result != 0 ? result : 1);
	}
	printf("glm52_validation tier=dsa PASS\n");
	SparkGlm52ValFixtureDestroy(&fixture);
	if ( result == 0 )
		printf("glm52_validation PASS\n");
	return(result);
}

#else /* SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST */

/* Host-executed proof of every pure formula the tiers rely on: codec encode/
 * decode round trips, the monotone-key selection reference, the renormalised
 * mixture, and the DSA cache shaping's separability predicate. Compiled and
 * run by tests/test_glm52_cuda_validator_tier2_oracle.py on every host; no
 * CUDA symbol is touched in this build. */

static int SparkGlm52ValSelftestCodecRoundTrip(void)
{
	uint32_t codec;
	for (codec = 2u; codec <= 7u; codec++)
	{
		uint32_t bits = SparkGlm52ValCodecStoredBits(codec);
		uint32_t columns = SparkGlm52ValCodecScaleGroup(codec) * 2u;
		uint64_t row_bytes = SparkGlm52ValPayloadRowBytes(codec,columns);
		uint8_t *row = (uint8_t *)calloc(1,(size_t)row_bytes);
		uint32_t column;
		int failures = 0;
		if ( row == 0 )
			return(1);
		for (column = 0u; column < columns; column++)
		{
			uint64_t bit = (uint64_t)column * bits;
			uint32_t lane;
			uint32_t pattern;
			int32_t value;
			if ( SparkGlm52ValCodecUsesSignedIntGrid(codec) )
			{
				static const int32_t span_min[3] = {-32,-64,-128};
				static const int32_t span_max[3] = {31,63,127};
				uint32_t grid = codec - SPARK_GLM52_VAL_CODEC_INT6;
				value = span_min[grid] + (int32_t)((column * 37u + 11u) %
					(uint32_t)(span_max[grid] - span_min[grid] + 1));
				pattern = (uint32_t)value & ((1u << bits) - 1u);
			}
			else if ( codec == SPARK_GLM52_VAL_CODEC_FP8 )
				pattern = (uint8_t)((column * 13u + 0x08u) & 0x77u); /* finite region */
			else
				pattern = (column * 3u) & 15u;
			for (lane = 0u; lane < bits; lane++)
				if ( (pattern >> lane) & 1u )
					row[(bit + lane) >> 3u] |= (uint8_t)(1u << ((bit + lane) & 7u));
		}
		for (column = 0u; column < columns; column++)
		{
			int32_t read_back = SparkGlm52ValReadCode(row,codec,0u,columns,column);
			if ( bits == 4u )
			{
				static const float expected[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
				uint8_t nibble = (uint8_t)(read_back & 15u);
				float decoded = SparkGlm52ValE2m1Decode(nibble);
				float signed_expected = (nibble & 8u) != 0u ? -expected[nibble & 7u] : expected[nibble & 7u];
				if ( memcmp(&decoded,&signed_expected,sizeof(float)) != 0 )
					failures++; /* E2M1 decode must be the exact table value */
			}
			else if ( codec == SPARK_GLM52_VAL_CODEC_FP8 )
			{
				float decoded = SparkGlm52ValE4m3Decode((uint8_t)read_back);
				if ( isnan(decoded) )
					failures++;
			}
			else if ( SparkGlm52ValCodecUsesSignedIntGrid(codec) )
			{
				int32_t minimum = SparkGlm52ValCodecCodeMinimum(codec);
				int32_t maximum = SparkGlm52ValCodecCodeMaximum(codec);
				if ( read_back < minimum || read_back > maximum )
				{
					fprintf(stderr,"codec=%u column=%u read_back=%d grid=[%d,%d]\n",
						codec,column,read_back,minimum,maximum);
					failures++; /* sign extension must stay inside the grid */
				}
			}
		}
		/* Scale addressing: last column's group must be the last index. */
		{
			uint64_t blocks = SparkGlm52ValScaleBlocksPerExpert(codec,3u,columns);
			uint64_t last_index = SparkGlm52ValScaleIndex(codec,3u,columns,5u,2u,columns - 1u);
			uint64_t groups = columns / SparkGlm52ValCodecScaleGroup(codec);
			if ( blocks != 3u * groups )
			{
				fprintf(stderr,"codec=%u scale_blocks=%llu expected=%llu\n",
					codec,(unsigned long long)blocks,(unsigned long long)(3u * groups));
				failures++;
			}
			if ( last_index != 5u * blocks + 2u * groups + (groups - 1u) )
			{
				fprintf(stderr,"codec=%u last_scale_index=%llu\n",codec,(unsigned long long)last_index);
				failures++;
			}
		}
		free(row);
		if ( failures != 0 )
			return(1);
	}
	/* Dequant of a written slab must reproduce the encoded grid exactly. */
	{
		uint32_t codec;
		for (codec = 2u; codec <= 7u; codec++)
		{
			uint32_t rows = 4u,columns = SparkGlm52ValCodecScaleGroup(codec) * 2u;
			uint64_t payload_bytes = SparkGlm52ValPayloadBytesPerExpert(codec,rows,columns);
			uint64_t scale_bytes = SparkGlm52ValScaleBufferBytes(codec,256u,rows,columns);
			uint8_t *payload = (uint8_t *)calloc(1,(size_t)payload_bytes);
			uint8_t *scales = (uint8_t *)calloc(1,(size_t)scale_bytes);
			uint32_t row,column;
			int failures = 0;
			if ( payload == 0 || scales == 0 )
				return(1);
			SparkGlm52ValFillExpertSlab(codec,payload,
				scales + SparkGlm52ValScaleExpertOffset(codec,256u,1u,rows,columns),rows,columns);
			if ( codec == SPARK_GLM52_VAL_CODEC_NVFP4 )
			{
				float global_scale = 4.0f;
				uint32_t expert;
				for (expert = 0u; expert < 256u; expert++)
					memcpy(scales + expert * sizeof(float),&global_scale,sizeof(float));
			}
			for (row = 0u; row < rows; row++)
				for (column = 0u; column < columns; column++)
				{
					float weight = SparkGlm52ValDequantWeight(payload,scales,codec,256u,1u,row,rows,columns,column);
					if ( !isfinite(weight) )
					{
						fprintf(stderr,"codec=%u dequant_not_finite row=%u column=%u\n",codec,row,column);
						failures++;
					}
				}
			free(payload);
			free(scales);
			if ( failures != 0 )
				return(1);
		}
	}
	return(0);
}

static int SparkGlm52ValSelftestSelection(void)
{
	float scores[64],bias[64];
	uint32_t indices[8];
	float weights[8],reference[64];
	uint32_t index,slot;
	/* Distinct scores across a wide keyed range. */
	for (index = 0u; index < 64u; index++)
	{
		scores[index] = -3.0f + 0.11f * (float)index;
		bias[index] = (index % 5u) * 0.03f;
		reference[index] = SparkGlm52ValSigmoid(scores[index]) + bias[index];
	}
	SparkGlm52ValReferenceTopk(scores,bias,64u,8u,indices,weights);
	{
		float total = 0.0f;
		for (slot = 0u; slot < 8u; slot++)
		{
			/* Every chosen slot must rank inside the top eight keyed scores. */
			uint32_t rank = 0u,other;
			for (other = 0u; other < 64u; other++)
				if ( SparkGlm52ValTopkKey(reference[other]) > SparkGlm52ValTopkKey(reference[indices[slot]]) )
					rank++;
			if ( rank >= 8u )
				return(1);
			total += weights[slot];
		}
		/* Renormalised mixture sums to ROUTED_SCALING_FACTOR. */
		if ( fabs((double)total - (double)SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR) > 1.0e-3 )
			return(1);
	}
	/* The bias selects but does not weigh: with a strong pro-tail bias the
	 * chosen SET moves, while each emitted weight stays the UNBIASED sigmoid
	 * of its own expert. */
	{
		float strong[64],chosen_weights[8];
		uint32_t chosen[8];
		for (index = 0u; index < 64u; index++)
			strong[index] = index < 32u ? -8.0f : 8.0f;
		SparkGlm52ValReferenceTopk(scores,strong,64u,8u,chosen,chosen_weights);
		for (slot = 0u; slot < 8u; slot++)
			if ( chosen[slot] < 32u )
				return(1); /* selection followed the bias into the tail half */
		for (slot = 0u; slot < 8u; slot++)
		{
			double unbiased = (double)SparkGlm52ValSigmoid(scores[chosen[slot]]);
			double normalized = unbiased / ((double)SparkGlm52ValSigmoid(scores[chosen[0]]) +
				(double)SparkGlm52ValSigmoid(scores[chosen[1]]) +
				(double)SparkGlm52ValSigmoid(scores[chosen[2]]) +
				(double)SparkGlm52ValSigmoid(scores[chosen[3]]) +
				(double)SparkGlm52ValSigmoid(scores[chosen[4]]) +
				(double)SparkGlm52ValSigmoid(scores[chosen[5]]) +
				(double)SparkGlm52ValSigmoid(scores[chosen[6]]) +
				(double)SparkGlm52ValSigmoid(scores[chosen[7]]));
			if ( fabs((double)chosen_weights[slot] - normalized * (double)SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR) > 1.0e-3 )
				return(1);
		}
	}
	return(0);
}

static int SparkGlm52ValSelftestShaping(void)
{
	float centroid[SPARK_GLM52_VDSA_DIM];
	uint16_t *keys;
	SparkGlm52ValDsaShape shape;
	uint32_t dimension,trial;
	keys = (uint16_t *)malloc((size_t)SPARK_GLM52_VALIDATION_DSA_CONTEXT * SPARK_GLM52_VDSA_DIM * sizeof(uint16_t));
	if ( keys == 0 )
		return(1);
	for (trial = 0u; trial < 3u; trial++)
	{
		for (dimension = 0u; dimension < SPARK_GLM52_VDSA_DIM; dimension++)
			centroid[dimension] = ((float)((int32_t)(SparkGlm52ValNext() & 0xffffu) - 32768) / 32768.0f) *
				(trial == 2u ? 0.01f : 1.0f);
		centroid[0] += 0.5f; /* keep the centroid non-degenerate */
		if ( SparkGlm52ValShapeIndexCache(keys,SPARK_GLM52_VALIDATION_DSA_CONTEXT,SPARK_GLM52_VDSA_SELECTED,
			centroid,SPARK_GLM52_VDSA_DIM,100.0f,0.05f,&shape) != 0 )
		{
			free(keys);
			return(1);
		}
		if ( shape.separable == 0u || shape.top_minimum <= shape.tail_maximum )
		{
			free(keys);
			return(1);
		}
		for (dimension = 0u; dimension < SPARK_GLM52_VDSA_SELECTED; dimension++)
			if ( shape.selected_positions[dimension] != dimension )
			{
				free(keys);
				return(1);
			}
	}
	free(keys);
	return(0);
}

int main(int argc,char **argv)
{
	int failures = 0;
	(void)argc;
	(void)argv;
	if ( SparkGlm52ValSelftestCodecRoundTrip() != 0 )
	{
		fprintf(stderr,"glm52_validator_selftest failure=codec_round_trip\n");
		failures++;
	}
	if ( SparkGlm52ValSelftestSelection() != 0 )
	{
		fprintf(stderr,"glm52_validator_selftest failure=selection_reference\n");
		failures++;
	}
	if ( SparkGlm52ValSelftestShaping() != 0 )
	{
		fprintf(stderr,"glm52_validator_selftest failure=dsa_shaping\n");
		failures++;
	}
	if ( failures == 0 )
		printf("glm52_validator_selftest PASS codec=%u stored_bits=%u scale_group=%u\n",
			SPARK_GLM52_VAL_CODEC,SparkGlm52ValCodecStoredBits(SPARK_GLM52_VAL_CODEC),
			SparkGlm52ValCodecScaleGroup(SPARK_GLM52_VAL_CODEC));
	return(failures == 0 ? 0 : 1);
}

#endif /* SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST */
