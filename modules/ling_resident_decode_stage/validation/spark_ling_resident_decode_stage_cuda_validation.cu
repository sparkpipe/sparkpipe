#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_ling_model.h"
#include "sparkpipe/spark_ling_resident_decode_stage_firmware.h"
#include "spark_ling_resident_decode_stage_internal.h"


#define SPARK_LING_VALIDATION_TOKENS 4u
#define SPARK_LING_VALIDATION_LANES 1u
#define SPARK_LING_VALIDATION_EMBED_ROWS 8u

#define SPARK_LING_VALIDATION_PAGES 34u
#define SPARK_LING_VALIDATION_KV_ACCESS_WORDS 6u

#define SPARK_LING_VALIDATION_DSA_CONTEXT 2064u
#define SPARK_LING_VALIDATION_POOLS \
	(SPARK_LING_VALIDATION_DSA_CONTEXT / 4u)

extern "C" int32_t SparkLingConfigureCudaModule(uint32_t *multiprocessor_count);
extern "C" int32_t SparkLingLaunchCudaWaveBegin(const SparkLingCudaWave *wave);
extern "C" int32_t SparkLingLaunchCudaLayerAttention(const SparkLingCudaWave *wave,uint32_t local_layer);
extern "C" int32_t SparkLingLaunchCudaLayerMlp(const SparkLingCudaWave *wave,uint32_t local_layer);
extern "C" int32_t SparkLingLaunchCudaLayerAttentionPost(const SparkLingCudaWave *wave,uint32_t local_layer);
extern "C" int32_t SparkLingLaunchCudaLayerMlpPost(const SparkLingCudaWave *wave,uint32_t local_layer);

#define SPARK_LING_VHIDDEN SPARK_LING_MODEL_HIDDEN_DIMENSION
#define SPARK_LING_VKDA_HEADS SPARK_LING_MODEL_KDA_HEAD_COUNT
#define SPARK_LING_VKDA_DIM SPARK_LING_MODEL_KDA_QKV_DIMENSION
#define SPARK_LING_VKDA_LOW_RANK SPARK_LING_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK
#define SPARK_LING_VKDA_CONV SPARK_LING_MODEL_KDA_SHORT_CONV_KERNEL
#define SPARK_LING_VHEADS SPARK_LING_MODEL_MLA_HEAD_COUNT
#define SPARK_LING_VQUERY_A SPARK_LING_MODEL_MLA_QUERY_A_DIMENSION
#define SPARK_LING_VLATENT SPARK_LING_MODEL_MLA_LATENT_DIMENSION
#define SPARK_LING_VQK_NOPE SPARK_LING_MODEL_MLA_QK_NOPE_HEAD_DIMENSION
#define SPARK_LING_VVALUE_DIM SPARK_LING_MODEL_MLA_VALUE_HEAD_DIMENSION
#define SPARK_LING_VATTN_COLS (SPARK_LING_VHEADS * SPARK_LING_VVALUE_DIM)
#define SPARK_LING_VQ_B_ROWS (SPARK_LING_VHEADS * SPARK_LING_VQK_NOPE)
#define SPARK_LING_VKV_SLOT_ELEMENTS SPARK_LING_MODEL_MLA_KV_A_DIMENSION
#define SPARK_LING_VDSA_HEADS SPARK_LING_MODEL_INDEX_HEAD_COUNT
#define SPARK_LING_VDSA_DIM SPARK_LING_MODEL_INDEX_HEAD_DIMENSION
#define SPARK_LING_VDSA_QUERY_DIM (SPARK_LING_VDSA_HEADS * SPARK_LING_VDSA_DIM)
#define SPARK_LING_VHC SPARK_LING_MODEL_HC_MULT
#define SPARK_LING_VHC_MIX SPARK_LING_MODEL_HC_MIX_DIMENSION
#define SPARK_LING_VHC_FLAT (SPARK_LING_VHC * SPARK_LING_VHIDDEN)
#define SPARK_LING_VDENSE_INTER SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define SPARK_LING_VGATE_UP_ROWS (2u * SPARK_LING_VDENSE_INTER)
#define SPARK_LING_VEXPERTS SPARK_LING_MODEL_MOE_EXPERT_COUNT
#define SPARK_LING_VTOP_K SPARK_LING_MODEL_MOE_TOP_K
#define SPARK_LING_VEXPERT_INTER SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION
#define SPARK_LING_VW1_ROWS (2u * SPARK_LING_VEXPERT_INTER)
#define SPARK_LING_VW2_ROWS SPARK_LING_VHIDDEN
#define SPARK_LING_VW2_COLUMNS SPARK_LING_VEXPERT_INTER
#define SPARK_LING_VEXPERT_COLUMNS SPARK_LING_VHIDDEN
#define SPARK_LING_VROUTED_SCALE SPARK_LING_MODEL_MOE_ROUTED_SCALING_FACTOR
#define SPARK_LING_VINDEX_PACKED SPARK_LING_MODEL_INDEX_PACKED_TOKEN_DIMENSION
#define SPARK_LING_VINDEX_TOPK SPARK_LING_MODEL_INDEX_TOP_K
#define SPARK_LING_VINDEX_WIDTH SPARK_LING_MODEL_INDEX_OUTPUT_WIDTH

static uint32_t SparkLingValRandomState;

static uint32_t SparkLingValNext(void)
{
	uint32_t value = SparkLingValRandomState;
	value ^= value << 13;
	value ^= value >> 17;
	value ^= value << 5;
	SparkLingValRandomState = value;
	return(value);
}

static uint16_t SparkLingValBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	uint32_t lsb = (bits >> 16) & 1u;
	uint32_t rounded = (bits + 0x7fffu + lsb) >> 16;
	return((uint16_t)(rounded & 0xffffu));
}

static float SparkLingValFromBf16(uint16_t value)
{
	uint32_t bits = ((uint32_t)value) << 16;
	float out;
	memcpy(&out,&bits,sizeof(out));
	return(out);
}

static void SparkLingValFill(uint16_t *packed,float *exact,uint64_t count,float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		float value = (((float)(int32_t)(SparkLingValNext() & 0xffffu) -
			32768.0f) / 32768.0f) * scale;
		exact[index] = value;
		packed[index] = SparkLingValBf16(value);
	}
}

static void SparkLingValFillNorm(uint16_t *packed,float *exact,uint64_t count)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		exact[index] = 1.0f;
		packed[index] = SparkLingValBf16(1.0f);
	}
}

static int SparkLingValFail(const char *check,const char *detail)
{
	printf("FAIL %s: %s\n",check,detail);
	return(1);
}

typedef struct SparkLingValMetrics
{
	double max_relative_l2;
	double cosine;
	double max_abs;
} SparkLingValMetrics;

static void SparkLingValMeasure(SparkLingValMetrics *metrics,const float *actual,const float *reference,uint64_t count)
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

static int SparkLingValReport(const char *check,const SparkLingValMetrics *metrics,double max_relative_l2,double minimum_cosine)
{
	int ok = metrics->max_relative_l2 <= max_relative_l2 && metrics->cosine >= minimum_cosine;
	printf("%s %-42s rel_l2 %.5f (max %.5f) cos %.7f (min %.7f) maxabs %.3e\n",
		ok ? "PASS" : "FAIL",check,metrics->max_relative_l2,max_relative_l2,
		metrics->cosine,minimum_cosine,metrics->max_abs);
	return(ok ? 0 : 1);
}

#define SPARK_LING_VAL_CODEC_INT6 2u
#define SPARK_LING_VAL_CODEC_INT7 3u
#define SPARK_LING_VAL_CODEC_INT8 4u
#define SPARK_LING_VAL_CODEC_FP8 5u
#define SPARK_LING_VAL_CODEC_NVFP4 6u
#define SPARK_LING_VAL_CODEC_MXFP4 7u

#if defined(LING_EXPERT_WEIGHT_CODEC)
#if LING_EXPERT_WEIGHT_CODEC == 2
#define SPARK_LING_VAL_CODEC SPARK_LING_VAL_CODEC_INT6
#elif LING_EXPERT_WEIGHT_CODEC == 3
#define SPARK_LING_VAL_CODEC SPARK_LING_VAL_CODEC_INT7
#elif LING_EXPERT_WEIGHT_CODEC == 4
#define SPARK_LING_VAL_CODEC SPARK_LING_VAL_CODEC_INT8
#elif LING_EXPERT_WEIGHT_CODEC == 5
#define SPARK_LING_VAL_CODEC SPARK_LING_VAL_CODEC_FP8
#elif LING_EXPERT_WEIGHT_CODEC == 6
#define SPARK_LING_VAL_CODEC SPARK_LING_VAL_CODEC_NVFP4
#elif LING_EXPERT_WEIGHT_CODEC == 7
#define SPARK_LING_VAL_CODEC SPARK_LING_VAL_CODEC_MXFP4
#else
#error "unsupported LING_EXPERT_WEIGHT_CODEC for the validator oracle"
#endif
#endif

static uint32_t SparkLingValCodecStoredBits(uint32_t codec)
{
	return(codec == SPARK_LING_VAL_CODEC_INT6 || codec == SPARK_LING_VAL_CODEC_NVFP4 ||
		codec == SPARK_LING_VAL_CODEC_MXFP4 ? 4u : 8u);
}

static uint32_t SparkLingValCodecScaleGroup(uint32_t codec)
{
	return(codec == SPARK_LING_VAL_CODEC_INT6 || codec == SPARK_LING_VAL_CODEC_MXFP4 ? 32u :
		(codec == SPARK_LING_VAL_CODEC_NVFP4 ? 16u : 128u));
}

static uint32_t SparkLingValCodecUsesSignedIntGrid(uint32_t codec)
{
	return(codec == SPARK_LING_VAL_CODEC_INT6 || codec == SPARK_LING_VAL_CODEC_INT7 ||
		codec == SPARK_LING_VAL_CODEC_INT8 ? 1u : 0u);
}

static float SparkLingValE4m3Decode(uint8_t code)
{
	int32_t sign = (code & 0x80u) != 0u ? -1 : 1;
	uint32_t exponent = (code >> 3) & 0xfu;
	uint32_t mantissa = code & 7u;
	float value;
	if ( exponent == 0u )
		return((float)sign * ((float)mantissa * (1.0f / 512.0f)));
	if ( exponent == 15u && mantissa == 7u )
		return((float)sign * NAN);
	value = (1.0f + ((float)mantissa) * 0.125f) *
		(float)(1u << (int32_t)(exponent - 7u < 31u ? exponent - 7u : 0u));
	if ( exponent < 7u )
		value = (1.0f + ((float)mantissa) * 0.125f) /
			(float)(1u << (7u - exponent));
	return((float)sign * value);
}

static float SparkLingValE2m1Decode(uint8_t nibble)
{
	static const float magnitudes[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	float magnitude = magnitudes[nibble & 7u];
	return((nibble & 8u) != 0u ? -magnitude : magnitude);
}

static int32_t SparkLingValCodecCodeMinimum(uint32_t codec)
{
	return(codec == SPARK_LING_VAL_CODEC_INT6 ? -31 :
		codec == SPARK_LING_VAL_CODEC_INT7 ? -63 :
		codec == SPARK_LING_VAL_CODEC_INT8 ? -127 : 0);
}

static int32_t SparkLingValCodecCodeMaximum(uint32_t codec)
{
	return(codec == SPARK_LING_VAL_CODEC_INT6 ? 31 :
		codec == SPARK_LING_VAL_CODEC_INT7 ? 63 :
		codec == SPARK_LING_VAL_CODEC_INT8 ? 127 : 0);
}

static uint64_t SparkLingValScaleBlocksPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	uint32_t group = SparkLingValCodecScaleGroup(codec);
	return(((uint64_t)((rows + 127u) / 128u)) * ((columns + group - 1u) / group));
}

static uint64_t SparkLingValScaleBytesPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	return(SparkLingValScaleBlocksPerExpert(codec,rows,columns) * 4u);
}

static uint64_t SparkLingValScaleBufferBytes(uint32_t codec,uint32_t expert_count,uint32_t rows,uint32_t columns)
{
	uint64_t per = SparkLingValScaleBytesPerExpert(codec,rows,columns);
	return(codec == SPARK_LING_VAL_CODEC_NVFP4
		? (uint64_t)expert_count * (per + sizeof(float))
		: (uint64_t)expert_count * per);
}

static uint64_t SparkLingValScaleExpertOffset(uint32_t codec,uint32_t expert_count,uint32_t expert,uint32_t rows,uint32_t columns)
{
	uint64_t per = SparkLingValScaleBytesPerExpert(codec,rows,columns);
	return(codec == SPARK_LING_VAL_CODEC_NVFP4
		? (uint64_t)expert * (per + sizeof(float)) + ((uint64_t)expert_count - 1u - expert) * 0u
		: (uint64_t)expert * per);
}

static uint64_t SparkLingValPayloadRowBytes(uint32_t codec,uint32_t columns)
{
	return(((uint64_t)columns * SparkLingValCodecStoredBits(codec) + 7u) / 8u);
}

static uint64_t SparkLingValPayloadBytesPerExpert(uint32_t codec,uint32_t rows,uint32_t columns)
{
	return((uint64_t)rows * SparkLingValPayloadRowBytes(codec,columns));
}

static uint64_t SparkLingValPayloadExpertOffset(uint32_t codec,uint32_t expert,uint32_t rows,uint32_t columns)
{
	return((uint64_t)expert * SparkLingValPayloadBytesPerExpert(codec,rows,columns));
}

static uint8_t SparkLingValPayloadCode(const uint8_t *payload,uint32_t codec,uint32_t row,uint32_t column,uint32_t columns)
{
	uint64_t row_bytes = SparkLingValPayloadRowBytes(codec,columns);
	const uint8_t *row_base = payload + (uint64_t)row * row_bytes;
	if ( SparkLingValCodecStoredBits(codec) == 8u )
		return(row_base[column]);
	return((row_base[column / 2u] >> ((column & 1u) * 4u)) & 0xfu);
}



static float SparkLingValSigmoid(float value)
{
	return(1.0f / (1.0f + expf(-value)));
}

static float SparkLingValBoundedDecay(float logit,float bias,float head_log_scale,float lower_bound)
{
	return(expf(lower_bound * SparkLingValSigmoid(expf(head_log_scale) * (logit + bias))));
}

static void SparkLingValRmsNorm(float *row,const float *weight,uint32_t dimension,float epsilon)
{
	float sum = 0.0f;
	uint32_t index;
	for (index = 0u; index < dimension; index++)
		sum += row[index] * row[index];
	float inverse = 1.0f / sqrtf(sum / (float)dimension + epsilon);
	for (index = 0u; index < dimension; index++)
		row[index] = row[index] * inverse * weight[index];
}

static void SparkLingValHcSite(const float *streams,const float *fn,const float *base,const float *scale,
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
		pre_out[h] = SparkLingValSigmoid(mixes_out[h] * scale[0] + base[h]) + epsilon;
		post_out[h] = 2.0f * SparkLingValSigmoid(mixes_out[hc + h] * scale[1] + base[hc + h]);
	}
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

static void SparkLingValHcPost(const float *out,const float *snapshot,const float *post,const float *comb,
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

static void SparkLingValKdaToken(
	const float *collapsed_in,
	const float *attn_norm_weight,
	const float *qkv_beta,
	const float *conv_q, const float *conv_k, const float *conv_v,
	const float *decay_down_gate,
	const float *decay_up,
	const float *gate_up,
	const float *dt_bias, const float *a_log,
	const float *out_norm, const float *out_weight,
	uint16_t *q_window, uint16_t *k_window, uint16_t *v_window,
	float *state,
	float *output)
{
	const uint32_t heads = SPARK_LING_VKDA_HEADS;
	const uint32_t dim_per_head = SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION;
	const uint32_t qk = SPARK_LING_VKDA_DIM;
	const uint32_t kernel = SPARK_LING_VKDA_CONV;
	float q[SPARK_LING_VKDA_DIM], k[SPARK_LING_VKDA_DIM], v[SPARK_LING_VKDA_DIM];
	float beta[SPARK_LING_VKDA_HEADS];
	float decay_latent[SPARK_LING_VKDA_LOW_RANK], gate_latent[SPARK_LING_VKDA_LOW_RANK];
	float retention[SPARK_LING_VKDA_DIM], gate[SPARK_LING_VKDA_DIM];
	float core[SPARK_LING_VKDA_DIM];
	uint32_t index, head, channel;

	float collapsed[SPARK_LING_VHIDDEN];
	{
		float total = 0.0f, inv;
		for (index = 0u; index < SPARK_LING_VHIDDEN; index++)
			total += collapsed_in[index] * collapsed_in[index];
		inv = 1.0f / sqrtf(total / (float)SPARK_LING_VHIDDEN + 1e-5f);
		for (index = 0u; index < SPARK_LING_VHIDDEN; index++)
			collapsed[index] = collapsed_in[index] * inv * attn_norm_weight[index];
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += qkv_beta[index * SPARK_LING_VHIDDEN + j] * collapsed[j];
		q[index] = sum;
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += qkv_beta[(qk + index) * SPARK_LING_VHIDDEN + j] * collapsed[j];
		k[index] = sum;
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += qkv_beta[(2u * qk + index) * SPARK_LING_VHIDDEN + j] * collapsed[j];
		v[index] = sum;
	}
	for (head = 0u; head < heads; head++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += qkv_beta[(2u * qk + qk + head) * SPARK_LING_VHIDDEN + j] * collapsed[j];
		beta[head] = SparkLingValSigmoid(sum);
	}
	for (index = 0u; index < qk; index++)
	{
		float taps[8];
		float total;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			taps[tap] = SparkLingValFromBf16(q_window[index * kernel + tap]);
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			q_window[index * kernel + tap] = q_window[index * kernel + tap + 1u];
		q_window[index * kernel + kernel - 1u] = SparkLingValBf16(q[index]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += SparkLingValFromBf16(q_window[index * kernel + tap]) * conv_q[index * kernel + tap];
		total = total * SparkLingValSigmoid(total);
		q[index] = total;
	}
	for (index = 0u; index < qk; index++)
	{
		float total;
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			k_window[index * kernel + tap] = k_window[index * kernel + tap + 1u];
		k_window[index * kernel + kernel - 1u] = SparkLingValBf16(k[index]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += SparkLingValFromBf16(k_window[index * kernel + tap]) * conv_k[index * kernel + tap];
		total = total * SparkLingValSigmoid(total);
		k[index] = total;
	}
	for (index = 0u; index < qk; index++)
	{
		float total;
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			v_window[index * kernel + tap] = v_window[index * kernel + tap + 1u];
		v_window[index * kernel + kernel - 1u] = SparkLingValBf16(v[index]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += SparkLingValFromBf16(v_window[index * kernel + tap]) * conv_v[index * kernel + tap];
		total = total * SparkLingValSigmoid(total);
		v[index] = total;
	}
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
		nq = 1.0f / sqrtf(nq + 1e-6f);
		nk = 1.0f / sqrtf(nk + 1e-6f);
		for (channel = 0u; channel < dim_per_head; channel++)
		{
			qh[channel] *= nq;
			kh[channel] *= nk;
		}
	}
	for (index = 0u; index < SPARK_LING_VKDA_LOW_RANK; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += decay_down_gate[index * SPARK_LING_VHIDDEN + j] * collapsed[j];
		decay_latent[index] = sum;
	}
	for (index = 0u; index < SPARK_LING_VKDA_LOW_RANK; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += decay_down_gate[(SPARK_LING_VKDA_LOW_RANK + index) * SPARK_LING_VHIDDEN + j] * collapsed[j];
		gate_latent[index] = sum;
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VKDA_LOW_RANK; j++)
			sum += decay_up[(uint64_t)index * SPARK_LING_VKDA_LOW_RANK + j] * decay_latent[j];
		retention[index] = SparkLingValBoundedDecay(sum,dt_bias[index],
			a_log[index / dim_per_head],-5.0f);
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VKDA_LOW_RANK; j++)
			sum += gate_up[(uint64_t)index * SPARK_LING_VKDA_LOW_RANK + j] * gate_latent[j];
		gate[index] = sum;
	}
	for (head = 0u; head < heads; head++)
	{
		const float *qh = q + head * dim_per_head;
		const float *kh = k + head * dim_per_head;
		const float *vh = v + head * dim_per_head;
		const float *ah = retention + head * dim_per_head;
		float *sh = state + (uint64_t)head * dim_per_head * dim_per_head;
		float predicted[SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION];
		for (uint32_t key = 0u; key < dim_per_head; key++)
			for (uint32_t value = 0u; value < dim_per_head; value++)
				sh[(uint64_t)key * dim_per_head + value] *= ah[key];
		for (uint32_t value = 0u; value < dim_per_head; value++)
		{
			float dot = 0.0f;
			for (uint32_t key = 0u; key < dim_per_head; key++)
				dot += sh[(uint64_t)key * dim_per_head + value] *
					kh[key] * ah[key];
			predicted[value] = dot;
		}
		for (uint32_t value = 0u; value < dim_per_head; value++)
		{
			float scale = beta[head] * (vh[value] - predicted[value]);
			for (uint32_t key = 0u; key < dim_per_head; key++)
				sh[(uint64_t)key * dim_per_head + value] += scale * kh[key];
		}
		for (uint32_t value = 0u; value < dim_per_head; value++)
		{
			float dot = 0.0f;
			for (uint32_t key = 0u; key < dim_per_head; key++)
				dot += sh[(uint64_t)key * dim_per_head + value] * qh[key];
			core[head * dim_per_head + value] = dot;
		}
	}
	for (head = 0u; head < heads; head++)
	{
		float *row = core + head * dim_per_head;
		float sum = 0.0f;
		for (channel = 0u; channel < dim_per_head; channel++)
			sum += row[channel] * row[channel];
		float inverse = 1.0f / sqrtf(sum / (float)dim_per_head + 1e-5f);
		for (channel = 0u; channel < dim_per_head; channel++)
			row[channel] = row[channel] * inverse * out_norm[channel] *
				SparkLingValSigmoid(gate[head * dim_per_head + channel]);
	}
	for (index = 0u; index < SPARK_LING_VHIDDEN; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < qk; j++)
			sum += out_weight[(uint64_t)index * qk + j] * core[j];
		output[index] = sum;
	}
}

static void SparkLingValMlaToken(
	const float *collapsed,
	const float *q_a, const float *q_a_norm, const float *q_b,
	const float *kv_a, const float *kv_a_norm,
	const float *kv_b_key, const float *kv_b_value, const float *o_proj,
	const float *latents,
	uint32_t context,
	float *latent_out,
	float *output)
{
	const uint32_t heads = SPARK_LING_VHEADS;
	const uint32_t latent = SPARK_LING_VLATENT;
	const uint32_t nope = SPARK_LING_VQK_NOPE;
	const uint32_t vdim = SPARK_LING_VVALUE_DIM;
	float q_lora[SPARK_LING_VQUERY_A];
	float q_rows[SPARK_LING_VQ_B_ROWS];
	float scale = 1.0f / 256.0f;
	(void)scale;
	uint32_t index;
	for (index = 0u; index < SPARK_LING_VQUERY_A; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += q_a[(uint64_t)index * SPARK_LING_VHIDDEN + j] * collapsed[j];
		q_lora[index] = sum;
	}
	SparkLingValRmsNorm(q_lora,q_a_norm,SPARK_LING_VQUERY_A,1e-5f);
	for (index = 0u; index < SPARK_LING_VQ_B_ROWS; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VQUERY_A; j++)
			sum += q_b[(uint64_t)index * SPARK_LING_VQUERY_A + j] * q_lora[j];
		q_rows[index] = sum;
	}
	for (index = 0u; index < latent; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += kv_a[(uint64_t)index * SPARK_LING_VHIDDEN + j] * collapsed[j];
		latent_out[index] = sum;
	}
	SparkLingValRmsNorm(latent_out,kv_a_norm,latent,1e-5f);
	{
		float values[SPARK_LING_VATTN_COLS];
		float scores[512];
		for (uint32_t head = 0u; head < heads; head++)
		{
			const float *qh = q_rows + head * nope;
			const float *key_t = kv_b_key + (uint64_t)head * latent * nope;
			const float *value_w = kv_b_value + (uint64_t)head * vdim * latent;
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
					scores[position] = dot * 0.0625f;
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
		for (index = 0u; index < SPARK_LING_VHIDDEN; index++)
		{
			float sum = 0.0f;
			for (uint32_t j = 0u; j < SPARK_LING_VATTN_COLS; j++)
				sum += o_proj[(uint64_t)index * SPARK_LING_VATTN_COLS + j] * values[j];
			output[index] = sum;
		}
	}
}

static void SparkLingValRouter(const float *router,const float *correction,const float *hidden,
	uint32_t *selected,float *weights)
{
	float scores[SPARK_LING_VEXPERTS];
	float choice[SPARK_LING_VEXPERTS];
	uint32_t expert;
	for (expert = 0u; expert < SPARK_LING_VEXPERTS; expert++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VHIDDEN; j++)
			sum += router[(uint64_t)expert * SPARK_LING_VHIDDEN + j] * hidden[j];
		scores[expert] = SparkLingValSigmoid(sum);
		choice[expert] = scores[expert] + correction[expert];
	}
	for (uint32_t slot = 0u; slot < SPARK_LING_VTOP_K; slot++)
	{
		uint32_t best = 0u;
		for (expert = 1u; expert < SPARK_LING_VEXPERTS; expert++)
			if (choice[expert] > choice[best])
				best = expert;
		selected[slot] = best;
		choice[best] = -3.0e38f;
	}
	{
		float total = 0.0f;
		for (uint32_t slot = 0u; slot < SPARK_LING_VTOP_K; slot++)
		{
			weights[slot] = scores[selected[slot]];
			total += weights[slot];
		}
		for (uint32_t slot = 0u; slot < SPARK_LING_VTOP_K; slot++)
			weights[slot] = weights[slot] / (total + 1e-20f) * SPARK_LING_VROUTED_SCALE;
	}
}


typedef struct SparkLingValMatrix
{
	uint32_t rows;
	uint32_t columns;
	float *host;
	void *device;
} SparkLingValMatrix;

static int SparkLingValAllocMatrix(SparkLingValMatrix *matrix,uint32_t rows,uint32_t columns,int mode,float scale)
{
	uint16_t *packed;
	uint64_t count = (uint64_t)rows * columns;
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->host = (float *)malloc(count * sizeof(float));
	packed = (uint16_t *)malloc(count * sizeof(uint16_t));
	if (matrix->host == 0 || packed == 0)
		return(SparkLingValFail("fixture","host_alloc"));
	if (cudaMalloc((void **)&matrix->device,count * sizeof(uint16_t)) != cudaSuccess)
		return(SparkLingValFail("fixture","device_alloc"));
	SparkLingValRandomState += 101u;
	if (mode == 1)
		SparkLingValFillNorm(packed,matrix->host,count);
	else
		SparkLingValFill(packed,matrix->host,count,scale);
	if (cudaMemcpy(matrix->device,packed,count * sizeof(uint16_t),cudaMemcpyHostToDevice) != cudaSuccess)
		return(SparkLingValFail("fixture","weight_upload"));
	free(packed);
	return(0);
}

static void SparkLingValFreeMatrix(SparkLingValMatrix *matrix)
{
	free(matrix->host);
	cudaFree(matrix->device);
	memset(matrix,0,sizeof(*matrix));
}

static void *SparkLingValAllocZeroed(uint64_t bytes)
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

static int SparkLingValSelftestAssert(int condition,const char *what)
{
	if (!condition)
	{
		printf("FAIL selftest: %s\n",what);
		return(1);
	}
	return(0);
}

static int SparkLingValOracleSelftest(void)
{
	int failures = 0;
	failures += SparkLingValSelftestAssert(
		fabsf(SparkLingValBoundedDecay(0.0f,0.0f,0.0f,-5.0f) -
			expf(-5.0f * 0.5f)) < 1e-6f,"bounded decay at zero");
	{
		float value = SparkLingValBoundedDecay(2.0f,-1.0f,0.5f,-5.0f);
		float expect = expf(-5.0f * SparkLingValSigmoid(expf(0.5f) * 1.0f));
		failures += SparkLingValSelftestAssert(fabsf(value - expect) < 1e-6f,
			"bounded decay formula");
		failures += SparkLingValSelftestAssert(value > 0.0f && value <= 1.0f,
			"bounded decay range (0,1]");
	}
	{
		uint64_t slab = SparkLingValPayloadBytesPerExpert(SPARK_LING_VAL_CODEC_FP8,64u,128u);
		failures += SparkLingValSelftestAssert(slab == 64u * 128u,
			"fp8 payload slab = rows*columns");
		failures += SparkLingValSelftestAssert(
			SparkLingValPayloadExpertOffset(SPARK_LING_VAL_CODEC_FP8,3u,64u,128u) ==
			3u * slab,"expert-major payload offset");
		failures += SparkLingValSelftestAssert(
			SparkLingValScaleExpertOffset(SPARK_LING_VAL_CODEC_FP8,288u,7u,64u,128u) ==
			7u * SparkLingValScaleBytesPerExpert(SPARK_LING_VAL_CODEC_FP8,64u,128u),
			"expert-major scale offset");
	}
	{
		uint8_t code;
		int ok = 1;
		for (code = 0u; code < 255u; code++)
		{
			if ((code & 0x7fu) == 0x7fu)
				continue;
			float decoded = SparkLingValE4m3Decode(code);
			if (!(decoded == decoded))
				ok = 0;
		}
		failures += SparkLingValSelftestAssert(ok,"e4m3 finite decode");
		failures += SparkLingValSelftestAssert(
			SparkLingValE4m3Decode(0x38u) == 1.0f &&
			SparkLingValE4m3Decode(0xb8u) == -1.0f,"e4m3 decode values");
	}
	{
		float streams[SPARK_LING_VHC * 4];
		float fn[SPARK_LING_VHC_MIX * 16];
		float base[SPARK_LING_VHC_MIX];
		float scale[3] = {0.5f,0.5f,0.5f};
		float mixes[SPARK_LING_VHC_MIX];
		float pre[SPARK_LING_VHC],post[SPARK_LING_VHC];
		float comb[SPARK_LING_VHC * SPARK_LING_VHC];
		float collapsed[4],snapshot[SPARK_LING_VHC * 4];
		uint32_t i,j;
		for (i = 0u; i < SPARK_LING_VHC * 4u; i++)
			streams[i] = ((float)(i % 7u) - 3.0f) * 0.25f;
		for (i = 0u; i < SPARK_LING_VHC_MIX * 16u; i++)
			fn[i] = ((float)(i % 5u) - 2.0f) * 0.125f;
		for (i = 0u; i < SPARK_LING_VHC_MIX; i++)
			base[i] = ((float)(i % 3u) - 1.0f) * 0.5f;
		SparkLingValHcSite(streams,fn,base,scale,1e-6f,
			SPARK_LING_VHC,4u,mixes,pre,post,comb,collapsed,snapshot);
		for (i = 0u; i < SPARK_LING_VHC; i++)
		{
			failures += SparkLingValSelftestAssert(pre[i] > 1e-6f && pre[i] <= 1.0f + 1e-6f + 1e-6f,
				"pre in (eps, 1+eps]");
			failures += SparkLingValSelftestAssert(post[i] >= 0.0f && post[i] <= 2.0f,
				"post in [0,2]");
		}
		for (j = 0u; j < SPARK_LING_VHC; j++)
		{
			float total = 0.0f;
			for (i = 0u; i < SPARK_LING_VHC; i++)
				total += comb[i * SPARK_LING_VHC + j];
			failures += SparkLingValSelftestAssert(fabsf(total - 1.0f) < 1e-3f,
				"comb column sums to 1 (doubly stochastic)");
		}
	}
	{
		uint32_t selected_pools[2] = {0u,1u};
		uint32_t context = 4u * 5u + 2u;
		uint32_t sequence = 0u, width = 11u;
		uint32_t list[11];
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
		failures += SparkLingValSelftestAssert(list[7] == 7u,"pool 1 last token");
		failures += SparkLingValSelftestAssert(list[8] == 20u && list[9] == 21u,"tail placement");
		failures += SparkLingValSelftestAssert(list[10] == 0xFFFFFFFFu,"empty tail slot sentinel");
		(void)sequence;
	}
	{
		uint32_t heads = 64u, dim = 128u;
		uint32_t qk = heads * dim;
		failures += SparkLingValSelftestAssert(
			2u * qk + qk + heads == 3u * qk + heads,"fused q|k|v|beta row count");
	}
	{
		static float collapsed[SPARK_LING_VHIDDEN];
		static float qkv_beta[(3u * SPARK_LING_VKDA_DIM + SPARK_LING_VKDA_HEADS) * SPARK_LING_VHIDDEN];
		static float conv[3u][SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_CONV];
		static float down_gate[2u * SPARK_LING_VKDA_LOW_RANK * SPARK_LING_VHIDDEN];
		static float up[2u][SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_LOW_RANK];
		static float dt[SPARK_LING_VKDA_DIM], alog[SPARK_LING_VKDA_HEADS];
		static float onorm[SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION];
		static float hidden_norm[SPARK_LING_VHIDDEN];
		static float ow[SPARK_LING_VHIDDEN * SPARK_LING_VKDA_DIM];
		static float state[(uint64_t)SPARK_LING_VKDA_HEADS * 128u * 128u];
		static float output[SPARK_LING_VHIDDEN];
		static uint16_t wq[SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_CONV];
		static uint16_t wk[SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_CONV];
		static uint16_t wv[SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_CONV];
		uint64_t i;
		uint32_t j;
		int finite = 1;
		float magnitude = 0.0f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VHIDDEN; i++)
			collapsed[i] = ((float)(i % 11u) - 5.0f) * 0.02f;
		for (i = 0u; i < sizeof(qkv_beta) / sizeof(float); i++)
			qkv_beta[i] = ((float)(i % 7u) - 3.0f) * 0.01f;
		for (i = 0u; i < sizeof(conv) / sizeof(float); i++)
			((float *)conv)[i] = ((float)(i % 5u) - 2.0f) * 0.05f;
		for (i = 0u; i < sizeof(down_gate) / sizeof(float); i++)
			down_gate[i] = ((float)(i % 9u) - 4.0f) * 0.01f;
		for (i = 0u; i < sizeof(up) / sizeof(float); i++)
			((float *)up)[i] = ((float)(i % 5u) - 2.0f) * 0.02f;
		for (i = 0u; i < SPARK_LING_VKDA_DIM; i++)
			dt[i] = -0.5f;
		for (i = 0u; i < SPARK_LING_VKDA_HEADS; i++)
			alog[i] = 0.1f;
		for (i = 0u; i < SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION; i++)
			onorm[i] = 1.0f;
		for (i = 0u; i < SPARK_LING_VHIDDEN; i++)
			hidden_norm[i] = 1.0f;
		for (i = 0u; i < sizeof(ow) / sizeof(float); i++)
			ow[i] = ((float)(i % 7u) - 3.0f) * 0.005f;
		memset(wq,0,sizeof(wq));
		memset(wk,0,sizeof(wk));
		memset(wv,0,sizeof(wv));
		memset(state,0,sizeof(state));
		SparkLingValKdaToken(collapsed,hidden_norm,qkv_beta,conv[0],conv[1],conv[2],
			down_gate,up[0],up[1],dt,alog,onorm,ow,wq,wk,wv,state,output);
		for (j = 0u; j < SPARK_LING_VHIDDEN; j++)
		{
			if (!(output[j] == output[j]) || fabsf(output[j]) > 1e30f)
				finite = 0;
			magnitude += fabsf(output[j]);
		}
		failures += SparkLingValSelftestAssert(finite,"kda oracle output finite");
		failures += SparkLingValSelftestAssert(magnitude > 0.0f,"kda oracle output non-zero");
	}
	{
		static float collapsed[SPARK_LING_VHIDDEN];
		static float q_a[SPARK_LING_VQUERY_A * SPARK_LING_VHIDDEN];
		static float q_a_norm[SPARK_LING_VQUERY_A];
		static float q_b[(uint64_t)SPARK_LING_VQ_B_ROWS * SPARK_LING_VQUERY_A];
		static float kv_a[(uint64_t)SPARK_LING_VKV_SLOT_ELEMENTS * SPARK_LING_VHIDDEN];
		static float kv_a_norm[SPARK_LING_VLATENT];
		static float kv_b_key[(uint64_t)SPARK_LING_VHEADS * SPARK_LING_VLATENT * SPARK_LING_VQK_NOPE];
		static float kv_b_value[(uint64_t)SPARK_LING_VHEADS * SPARK_LING_VVALUE_DIM * SPARK_LING_VLATENT];
		static float o_proj[(uint64_t)SPARK_LING_VHIDDEN * SPARK_LING_VATTN_COLS];
		static float latents[3u * SPARK_LING_VLATENT];
		static float latent_out[SPARK_LING_VLATENT];
		static float output[SPARK_LING_VHIDDEN];
		uint64_t i;
		uint32_t j;
		int finite = 1;
		float magnitude = 0.0f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VHIDDEN; i++)
			collapsed[i] = ((float)(i % 13u) - 6.0f) * 0.015f;
		for (i = 0u; i < sizeof(q_a) / sizeof(float); i++)
			q_a[i] = ((float)(i % 7u) - 3.0f) * 0.01f;
		for (i = 0u; i < SPARK_LING_VQUERY_A; i++)
			q_a_norm[i] = 1.0f;
		for (i = 0u; i < sizeof(q_b) / sizeof(float); i++)
			q_b[i] = ((float)(i % 5u) - 2.0f) * 0.01f;
		for (i = 0u; i < sizeof(kv_a) / sizeof(float); i++)
			kv_a[i] = ((float)(i % 9u) - 4.0f) * 0.01f;
		for (i = 0u; i < SPARK_LING_VLATENT; i++)
			kv_a_norm[i] = 1.0f;
		for (i = 0u; i < sizeof(kv_b_key) / sizeof(float); i++)
			kv_b_key[i] = ((float)(i % 5u) - 2.0f) * 0.01f;
		for (i = 0u; i < sizeof(kv_b_value) / sizeof(float); i++)
			kv_b_value[i] = ((float)(i % 7u) - 3.0f) * 0.01f;
		for (i = 0u; i < sizeof(o_proj) / sizeof(float); i++)
			o_proj[i] = ((float)(i % 11u) - 5.0f) * 0.005f;
		for (i = 0u; i < sizeof(latents) / sizeof(float); i++)
			latents[i] = ((float)(i % 5u) - 2.0f) * 0.05f;
		SparkLingValMlaToken(collapsed,q_a,q_a_norm,q_b,kv_a,kv_a_norm,
			kv_b_key,kv_b_value,o_proj,latents,3u,latent_out,output);
		for (j = 0u; j < SPARK_LING_VHIDDEN; j++)
		{
			if (!(output[j] == output[j]) || fabsf(output[j]) > 1e30f)
				finite = 0;
			magnitude += fabsf(output[j]);
		}
		failures += SparkLingValSelftestAssert(finite,"mla oracle output finite");
		failures += SparkLingValSelftestAssert(magnitude > 0.0f,"mla oracle output non-zero");
	}
	{
		static float router[(uint64_t)SPARK_LING_VEXPERTS * SPARK_LING_VHIDDEN];
		static float correction[SPARK_LING_VEXPERTS];
		static float hidden[SPARK_LING_VHIDDEN];
		uint32_t selected[SPARK_LING_VTOP_K];
		float weights[SPARK_LING_VTOP_K];
		uint64_t i;
		float total = 0.0f;
		int distinct = 1;
		for (i = 0u; i < sizeof(router) / sizeof(float); i++)
			router[i] = ((float)(i % 17u) - 8.0f) * 0.001f;
		for (i = 0u; i < SPARK_LING_VEXPERTS; i++)
			correction[i] = ((float)(i % 3u)) * 0.25f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VHIDDEN; i++)
			hidden[i] = ((float)(i % 7u) - 3.0f) * 0.05f;
		SparkLingValRouter(router,correction,hidden,selected,weights);
		for (i = 0u; i < SPARK_LING_VTOP_K; i++)
		{
			total += weights[i];
			for (uint64_t j = i + 1u; j < SPARK_LING_VTOP_K; j++)
				if (selected[i] == selected[j])
					distinct = 0;
		}
		failures += SparkLingValSelftestAssert(distinct,"router selects 8 distinct experts");
		failures += SparkLingValSelftestAssert(fabsf(total - SPARK_LING_VROUTED_SCALE) < 1e-3f,
			"router weights renormalise to the routed scale");
	}
	if (failures == 0)
		printf("ling_validator_selftest PASS (bounded decay, expert-major "
		       "addressing, e4m3, mHC sinkhorn, kpool expand, fused sections)\n");
	return(failures);
}

#ifndef SPARK_LING_VALIDATOR_ORACLE_SELFTEST


typedef struct SparkLingValFixture
{
	SparkLingLayerWeights weights;
	SparkLingExecutionSlot slot;
	SparkLingCudaWave wave;
	cudaStream_t stream;
	uint32_t multiprocessors;
	SparkLingValMatrix attn_norm,mlp_norm;
	SparkLingValMatrix kda_qkv_beta,kda_decay_gate_down,kda_decay_up,kda_gate_up;
	SparkLingValMatrix kda_q_conv,kda_k_conv,kda_v_conv,kda_out;
	float *kda_out_norm_dev;
	float kda_out_norm_host[SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION];
	float *kda_dt_bias;      float kda_dt_bias_host[SPARK_LING_VKDA_DIM];
	float *kda_a_log;        float kda_a_log_host[SPARK_LING_VKDA_HEADS];
	SparkLingValMatrix q_a,q_a_norm,q_b,kv_a,kv_a_norm,kv_b_key,kv_b_value,attn_output;
	SparkLingValMatrix index_q,index_k,index_head,index_norm_weight,index_norm_bias,index_compress_gate;
	float *index_compress_ape_dev;
	float index_compress_ape_host[SPARK_LING_MODEL_INDEX_KPOOL * SPARK_LING_VDSA_DIM];
	float *hc_attn_fn_dev,*hc_attn_base_dev,*hc_attn_scale_dev;
	float *hc_ffn_fn_dev,*hc_ffn_base_dev,*hc_ffn_scale_dev;
	float hc_attn_fn_host[SPARK_LING_VHC_MIX * SPARK_LING_VHC_FLAT];
	float hc_attn_base_host[SPARK_LING_VHC_MIX];
	float hc_attn_scale_host[3u];
	float hc_ffn_fn_host[SPARK_LING_VHC_MIX * SPARK_LING_VHC_FLAT];
	float hc_ffn_base_host[SPARK_LING_VHC_MIX];
	float hc_ffn_scale_host[3u];
	SparkLingValMatrix dense_gate_up,dense_down,router,shared_gate_up,shared_down;
	float *router_correction; float router_correction_host[SPARK_LING_VEXPERTS];
	uint16_t *streams,*residual,*normed,*q_compressed,*q_bf16,*kv_slot;
	uint16_t *query_latent,*attention_latent,*attention_value,*attention_out;
	uint16_t *gate_up,*intermediate,*expert_out,*shared_out;
	uint16_t *fused_qkvb,*fused_decay_gate,*decay_latent,*gate_latent;
	uint16_t *kda_beta_logit,*kda_gate_bf16,*kda_decay_logit,*kda_output;
	uint16_t *hc_collapsed,*hc_snapshot,*hc_mean,*index_query,*index_key;
	uint16_t *index_gate,*index_packed,*index_head_buf;
	uint32_t *selected_pools,*selected_positions;
	float *hc_mixes,*hc_pre,*hc_post,*hc_comb;
	float *kda_retention,*kda_write_gate,*router_logits,*selection_scores,*route_weight;
	uint32_t *token_ids,*resident_slots,*positions,*context_lengths;
	uint32_t *dense_row_offset,*dense_tile_prefix,*page_table;
	uint32_t *route_expert,*route_packed_row,*route_source_token,*group_row_offset;
	uint32_t *group_tile_prefix_w1,*group_tile_prefix_w2;
	uint64_t *head_maxloc;
	uint8_t *kv_cache,*index_cache,*kda_state_pools,*kda_window_pools;
	uint16_t *boundary_input;
	uint16_t boundary_host_rows[8u * SPARK_LING_VHIDDEN];
	uint32_t host_resident_slots_stage[1u];
	uint32_t host_positions_stage[1u];
	uint32_t host_token_ids_stage[1u];
	uint32_t *kda_state_index;
	uint32_t host_page_table[SPARK_LING_VALIDATION_PAGES];
	uint32_t host_kv_ordinals[SPARK_LING_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t host_index_ordinals[SPARK_LING_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t host_kda_ordinals[SPARK_LING_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t *kv_access_error;
} SparkLingValFixture;

static int SparkLingValFixtureBuild(SparkLingValFixture *fixture)
{
	uint32_t lane;
	memset(fixture,0,sizeof(*fixture));
	SparkLingValRandomState = 0x5eed1234u;
	if (SparkLingValAllocMatrix(&fixture->attn_norm,1u,SPARK_LING_VHIDDEN,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mlp_norm,1u,SPARK_LING_VHIDDEN,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_qkv_beta,
			3u * SPARK_LING_VKDA_DIM + SPARK_LING_VKDA_HEADS,SPARK_LING_VHIDDEN,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_decay_gate_down,
			2u * SPARK_LING_VKDA_LOW_RANK,SPARK_LING_VHIDDEN,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_decay_up,SPARK_LING_VKDA_DIM,SPARK_LING_VKDA_LOW_RANK,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_gate_up,SPARK_LING_VKDA_DIM,SPARK_LING_VKDA_LOW_RANK,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_q_conv,SPARK_LING_VKDA_DIM,SPARK_LING_VKDA_CONV,0,0.05f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_k_conv,SPARK_LING_VKDA_DIM,SPARK_LING_VKDA_CONV,0,0.05f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_v_conv,SPARK_LING_VKDA_DIM,SPARK_LING_VKDA_CONV,0,0.05f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_out,SPARK_LING_VHIDDEN,SPARK_LING_VKDA_DIM,0,0.005f) != 0 ||
		SparkLingValAllocMatrix(&fixture->q_a,SPARK_LING_VQUERY_A,SPARK_LING_VHIDDEN,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->q_a_norm,1u,SPARK_LING_VQUERY_A,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->q_b,SPARK_LING_VQ_B_ROWS,SPARK_LING_VQUERY_A,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kv_a,SPARK_LING_VKV_SLOT_ELEMENTS,SPARK_LING_VHIDDEN,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kv_a_norm,1u,SPARK_LING_VLATENT,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kv_b_key,
			(uint64_t)SPARK_LING_VHEADS * SPARK_LING_VLATENT,SPARK_LING_VQK_NOPE,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kv_b_value,
			(uint64_t)SPARK_LING_VHEADS * SPARK_LING_VVALUE_DIM,SPARK_LING_VLATENT,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->attn_output,SPARK_LING_VHIDDEN,SPARK_LING_VATTN_COLS,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->index_q,SPARK_LING_VDSA_QUERY_DIM,SPARK_LING_VQUERY_A,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->index_k,SPARK_LING_VDSA_DIM,SPARK_LING_VHIDDEN,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->index_head,SPARK_LING_VDSA_HEADS,SPARK_LING_VHIDDEN,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->index_norm_weight,1u,SPARK_LING_VDSA_DIM,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->index_norm_bias,1u,SPARK_LING_VDSA_DIM,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->index_compress_gate,SPARK_LING_VDSA_DIM,SPARK_LING_VHIDDEN,0,0.02f) != 0 ||
		SparkLingValAllocMatrix(&fixture->dense_gate_up,SPARK_LING_VGATE_UP_ROWS,SPARK_LING_VHIDDEN,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->dense_down,SPARK_LING_VHIDDEN,SPARK_LING_VDENSE_INTER,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->router,SPARK_LING_VEXPERTS,SPARK_LING_VHIDDEN,0,0.002f) != 0 ||
		SparkLingValAllocMatrix(&fixture->shared_gate_up,SPARK_LING_VW1_ROWS,SPARK_LING_VHIDDEN,0,0.005f) != 0 ||
		SparkLingValAllocMatrix(&fixture->shared_down,SPARK_LING_VW2_ROWS,SPARK_LING_VW2_COLUMNS,0,0.005f) != 0)
		return(1);
	return(0);
}

static int SparkLingValFixtureComplete(SparkLingValFixture *fixture)
{
	uint32_t state_bytes = SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER;
	fixture->streams = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHC_FLAT * 8u * sizeof(uint16_t));
	fixture->residual = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->normed = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->q_compressed = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VQUERY_A * 8u * sizeof(uint16_t));
	fixture->q_bf16 = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VQ_B_ROWS * 8u * sizeof(uint16_t));
	fixture->kv_slot = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VKDA_DIM * 8u * sizeof(uint16_t));
	fixture->query_latent = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHEADS * SPARK_LING_VLATENT * 8u * sizeof(uint16_t));
	fixture->attention_latent = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHEADS * SPARK_LING_VLATENT * 8u * sizeof(uint16_t));
	fixture->attention_value = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VATTN_COLS * 8u * sizeof(uint16_t));
	fixture->attention_out = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->gate_up = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VGATE_UP_ROWS * 8u * sizeof(uint16_t));
	fixture->intermediate = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VDENSE_INTER * 8u * sizeof(uint16_t));
	fixture->expert_out = (uint16_t *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VTOP_K * SPARK_LING_VHIDDEN * sizeof(uint16_t));
	fixture->shared_out = (uint16_t *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VHIDDEN * sizeof(uint16_t));
	fixture->fused_qkvb = (uint16_t *)SparkLingValAllocZeroed((uint64_t)(3u * SPARK_LING_VKDA_DIM + SPARK_LING_VKDA_HEADS) * 8u * sizeof(uint16_t));
	fixture->fused_decay_gate = (uint16_t *)SparkLingValAllocZeroed((uint64_t)2u * SPARK_LING_VKDA_LOW_RANK * 8u * sizeof(uint16_t));
	fixture->decay_latent = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VKDA_LOW_RANK * 8u * sizeof(uint16_t));
	fixture->gate_latent = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VKDA_LOW_RANK * 8u * sizeof(uint16_t));
	fixture->kda_beta_logit = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VKDA_HEADS * 8u * sizeof(uint16_t));
	fixture->kda_gate_bf16 = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VKDA_DIM * 8u * sizeof(uint16_t));
	fixture->kda_decay_logit = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VKDA_DIM * 8u * sizeof(uint16_t));
	fixture->kda_output = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->hc_collapsed = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->hc_snapshot = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHC_FLAT * 8u * sizeof(uint16_t));
	fixture->hc_mean = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VHIDDEN * 8u * sizeof(uint16_t));
	fixture->index_query = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VDSA_QUERY_DIM * 8u * sizeof(uint16_t));
	fixture->index_key = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VDSA_DIM * 8u * sizeof(uint16_t));
	fixture->index_gate = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VDSA_DIM * 8u * sizeof(uint16_t));
	fixture->index_packed = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VINDEX_PACKED * 8u * sizeof(uint16_t));
	fixture->index_head_buf = (uint16_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VDSA_HEADS * 8u * sizeof(uint16_t));
	fixture->selected_pools = (uint32_t *)SparkLingValAllocZeroed((uint64_t)8u * 512u * sizeof(uint32_t));
	fixture->selected_positions = (uint32_t *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VINDEX_WIDTH * sizeof(uint32_t));
	fixture->hc_mixes = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VHC_MIX * sizeof(float));
	fixture->hc_pre = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VHC * sizeof(float));
	fixture->hc_post = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VHC * sizeof(float));
	fixture->hc_comb = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VHC * SPARK_LING_VHC * sizeof(float));
	fixture->kda_retention = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VKDA_DIM * sizeof(float));
	fixture->kda_write_gate = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VKDA_HEADS * sizeof(float));
	fixture->router_logits = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VEXPERTS * sizeof(float));
	fixture->selection_scores = (float *)SparkLingValAllocZeroed((uint64_t)8u * 1024u * sizeof(float));
	fixture->route_weight = (float *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VTOP_K * sizeof(float));
	fixture->token_ids = (uint32_t *)SparkLingValAllocZeroed(8u * sizeof(uint32_t));
	fixture->resident_slots = (uint32_t *)SparkLingValAllocZeroed(8u * sizeof(uint32_t));
	fixture->positions = (uint32_t *)SparkLingValAllocZeroed(8u * sizeof(uint32_t));
	fixture->context_lengths = (uint32_t *)SparkLingValAllocZeroed(8u * sizeof(uint32_t));
	fixture->dense_row_offset = (uint32_t *)SparkLingValAllocZeroed(4u * sizeof(uint32_t));
	fixture->dense_tile_prefix = (uint32_t *)SparkLingValAllocZeroed(4u * sizeof(uint32_t));
	fixture->page_table = (uint32_t *)SparkLingValAllocZeroed((uint64_t)SPARK_LING_VALIDATION_PAGES * sizeof(uint32_t));
	fixture->route_expert = (uint32_t *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VTOP_K * sizeof(uint32_t));
	fixture->route_packed_row = (uint32_t *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VTOP_K * sizeof(uint32_t));
	fixture->route_source_token = (uint32_t *)SparkLingValAllocZeroed((uint64_t)8u * SPARK_LING_VTOP_K * sizeof(uint32_t));
	fixture->group_row_offset = (uint32_t *)SparkLingValAllocZeroed((SPARK_LING_VEXPERTS + 1u) * sizeof(uint32_t));
	fixture->group_tile_prefix_w1 = (uint32_t *)SparkLingValAllocZeroed((SPARK_LING_VEXPERTS + 1u) * sizeof(uint32_t));
	fixture->group_tile_prefix_w2 = (uint32_t *)SparkLingValAllocZeroed((SPARK_LING_VEXPERTS + 1u) * sizeof(uint32_t));
	fixture->head_maxloc = (uint64_t *)SparkLingValAllocZeroed(8u * sizeof(uint64_t));
	fixture->kv_access_error = (uint32_t *)SparkLingValAllocZeroed(SPARK_LING_VALIDATION_KV_ACCESS_WORDS * sizeof(uint32_t));
	if (fixture->streams == 0 || fixture->attention_out == 0 || fixture->kda_output == 0 ||
		fixture->hc_mixes == 0 || fixture->kda_retention == 0 || fixture->kv_access_error == 0)
		return(SparkLingValFail("fixture","scratch_alloc"));
	{
		uint64_t i;
		for (i = 0u; i < (uint64_t)SPARK_LING_VHC_MIX * SPARK_LING_VHC_FLAT; i++)
		{
			fixture->hc_attn_fn_host[i] = ((float)(i % 7u) - 3.0f) * 0.01f;
			fixture->hc_ffn_fn_host[i] = ((float)(i % 9u) - 4.0f) * 0.01f;
		}
		for (i = 0u; i < SPARK_LING_VHC_MIX; i++)
		{
			fixture->hc_attn_base_host[i] = ((float)(i % 5u) - 2.0f) * 0.25f;
			fixture->hc_ffn_base_host[i] = ((float)(i % 3u) - 1.0f) * 0.25f;
		}
		fixture->hc_attn_scale_host[0] = 0.5f;
		fixture->hc_attn_scale_host[1] = 0.5f;
		fixture->hc_attn_scale_host[2] = 0.5f;
		fixture->hc_ffn_scale_host[0] = 0.5f;
		fixture->hc_ffn_scale_host[1] = 0.5f;
		fixture->hc_ffn_scale_host[2] = 0.5f;
		fixture->hc_attn_fn_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->hc_attn_fn_host));
		fixture->hc_attn_base_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->hc_attn_base_host));
		fixture->hc_attn_scale_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->hc_attn_scale_host));
		fixture->hc_ffn_fn_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->hc_ffn_fn_host));
		fixture->hc_ffn_base_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->hc_ffn_base_host));
		fixture->hc_ffn_scale_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->hc_ffn_scale_host));
		if (fixture->hc_attn_fn_dev == 0 || fixture->hc_attn_base_dev == 0 ||
			fixture->hc_attn_scale_dev == 0 || fixture->hc_ffn_fn_dev == 0 ||
			fixture->hc_ffn_base_dev == 0 || fixture->hc_ffn_scale_dev == 0 ||
			cudaMemcpy(fixture->hc_attn_fn_dev,fixture->hc_attn_fn_host,sizeof(fixture->hc_attn_fn_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->hc_attn_base_dev,fixture->hc_attn_base_host,sizeof(fixture->hc_attn_base_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->hc_attn_scale_dev,fixture->hc_attn_scale_host,sizeof(fixture->hc_attn_scale_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->hc_ffn_fn_dev,fixture->hc_ffn_fn_host,sizeof(fixture->hc_ffn_fn_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->hc_ffn_base_dev,fixture->hc_ffn_base_host,sizeof(fixture->hc_ffn_base_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->hc_ffn_scale_dev,fixture->hc_ffn_scale_host,sizeof(fixture->hc_ffn_scale_host),cudaMemcpyHostToDevice) != cudaSuccess)
			return(SparkLingValFail("fixture","hc_upload"));
	}
	{
		uint32_t row,element;
		for (row = 0u; row < 8u; row++)
			for (element = 0u; element < SPARK_LING_VHIDDEN; element++)
				fixture->boundary_host_rows[(uint64_t)row * SPARK_LING_VHIDDEN + element] =
					SparkLingValBf16((((row * 7u + element) % 23u) - 11.0f) * 0.02f);
		fixture->boundary_input = (uint16_t *)SparkLingValAllocZeroed(sizeof(fixture->boundary_host_rows));
		if (fixture->boundary_input == 0 ||
			cudaMemcpy(fixture->boundary_input,fixture->boundary_host_rows,sizeof(fixture->boundary_host_rows),cudaMemcpyHostToDevice) != cudaSuccess)
			return(SparkLingValFail("fixture","boundary_input"));
	}
	fixture->kda_state_pools = (uint8_t *)SparkLingValAllocZeroed(state_bytes);
	fixture->kda_window_pools = (uint8_t *)SparkLingValAllocZeroed(SPARK_LING_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER);
	fixture->kda_state_index = (uint32_t *)SparkLingValAllocZeroed(sizeof(uint32_t));
	fixture->kv_cache = (uint8_t *)SparkLingValAllocZeroed(
		(uint64_t)SPARK_LING_VALIDATION_PAGES * 64u * SPARK_LING_MODEL_KV_SLOT_BYTES);
	fixture->index_cache = (uint8_t *)SparkLingValAllocZeroed(
		(uint64_t)SPARK_LING_VALIDATION_PAGES * 64u * SPARK_LING_VINDEX_PACKED * 2u);
	{
		uint32_t index;
		uint32_t zero = 0u;
		for (index = 0u; index < SPARK_LING_VALIDATION_PAGES; index++)
			fixture->host_page_table[index] = index;
		if (cudaMemcpy(fixture->page_table,fixture->host_page_table,sizeof(fixture->host_page_table),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->kda_state_index,&zero,sizeof(zero),cudaMemcpyHostToDevice) != cudaSuccess)
			return(SparkLingValFail("fixture","page_table"));
	}
	{
		uint32_t index;
		for (index = 0u; index < SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION; index++)
			fixture->kda_out_norm_host[index] = 1.0f;
		fixture->kda_out_norm_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->kda_out_norm_host));
		if (fixture->kda_out_norm_dev == 0 ||
			cudaMemcpy(fixture->kda_out_norm_dev,fixture->kda_out_norm_host,sizeof(fixture->kda_out_norm_host),cudaMemcpyHostToDevice) != cudaSuccess)
			return(SparkLingValFail("fixture","out_norm"));
	}
	{
		uint32_t index;
		for (index = 0u; index < SPARK_LING_VKDA_DIM; index++)
			fixture->kda_dt_bias_host[index] = -0.25f;
		for (index = 0u; index < SPARK_LING_VKDA_HEADS; index++)
			fixture->kda_a_log_host[index] = 0.1f;
		for (index = 0u; index < SPARK_LING_VEXPERTS; index++)
			fixture->router_correction_host[index] = index < 8u ? 4.0f : -4.0f;
		for (index = 0u; index < SPARK_LING_MODEL_INDEX_KPOOL * SPARK_LING_VDSA_DIM; index++)
			fixture->index_compress_ape_host[index] = ((float)(index % 5u) - 2.0f) * 0.25f;
		fixture->kda_dt_bias = (float *)SparkLingValAllocZeroed(sizeof(fixture->kda_dt_bias_host));
		fixture->kda_a_log = (float *)SparkLingValAllocZeroed(sizeof(fixture->kda_a_log_host));
		fixture->router_correction = (float *)SparkLingValAllocZeroed(sizeof(fixture->router_correction_host));
		fixture->index_compress_ape_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->index_compress_ape_host));
		if (fixture->kda_state_pools == 0 || fixture->kda_window_pools == 0 ||
			fixture->kv_cache == 0 || fixture->index_cache == 0 ||
			fixture->kda_dt_bias == 0 || fixture->kda_a_log == 0 ||
			fixture->router_correction == 0)
			return(SparkLingValFail("fixture","pool_alloc"));
		if (cudaMemcpy(fixture->kda_dt_bias,fixture->kda_dt_bias_host,sizeof(fixture->kda_dt_bias_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->kda_a_log,fixture->kda_a_log_host,sizeof(fixture->kda_a_log_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->router_correction,fixture->router_correction_host,sizeof(fixture->router_correction_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->index_compress_ape_dev,fixture->index_compress_ape_host,sizeof(fixture->index_compress_ape_host),cudaMemcpyHostToDevice) != cudaSuccess)
			return(SparkLingValFail("fixture","f32_upload"));
	}
	if (cudaStreamCreate(&fixture->stream) != cudaSuccess)
		return(SparkLingValFail("fixture","stream"));
	fixture->slot.stream = fixture->stream;
	fixture->multiprocessors = 16u;
	return(0);
}

static void SparkLingValBuildWave(SparkLingValFixture *fixture,uint32_t layer,uint32_t token,uint32_t position)
{
	SparkLingCudaWave *wave = &fixture->wave;
	SparkLingExecutionSlot *slot = &fixture->slot;
	SparkLingLayerWeights *weights = &fixture->weights;
	uint32_t token_words[8],slot_words[8],position_words[8],context_words[8];
	uint32_t row;
	memset(weights,0,sizeof(*weights));
	weights->attn_norm_bf16 = fixture->attn_norm.device;
	weights->post_attn_norm_bf16 = fixture->mlp_norm.device;
	weights->kda_qkv_beta_bf16 = fixture->kda_qkv_beta.device;
	weights->kda_decay_gate_down_bf16 = fixture->kda_decay_gate_down.device;
	weights->kda_decay_up_bf16 = fixture->kda_decay_up.device;
	weights->kda_gate_up_bf16 = fixture->kda_gate_up.device;
	weights->kda_q_conv_bf16 = fixture->kda_q_conv.device;
	weights->kda_k_conv_bf16 = fixture->kda_k_conv.device;
	weights->kda_v_conv_bf16 = fixture->kda_v_conv.device;
	weights->kda_out_norm_bf16 = fixture->kda_out_norm_dev;
	weights->kda_out_bf16 = fixture->kda_out.device;
	weights->kda_decay_bias_f32 = fixture->kda_dt_bias;
	weights->kda_head_log_scale_f32 = fixture->kda_a_log;
	weights->q_a_bf16 = fixture->q_a.device;
	weights->q_a_norm_bf16 = fixture->q_a_norm.device;
	weights->q_b_bf16 = fixture->q_b.device;
	weights->kv_a_bf16 = fixture->kv_a.device;
	weights->kv_a_norm_bf16 = fixture->kv_a_norm.device;
	weights->kv_b_key_transposed_bf16 = fixture->kv_b_key.device;
	weights->kv_b_value_bf16 = fixture->kv_b_value.device;
	weights->attn_output_bf16 = fixture->attn_output.device;
	weights->hc_attn_fn_f32 = fixture->hc_attn_fn_dev;
	weights->hc_attn_base_f32 = fixture->hc_attn_base_dev;
	weights->hc_attn_scale_f32 = fixture->hc_attn_scale_dev;
	weights->hc_ffn_fn_f32 = fixture->hc_ffn_fn_dev;
	weights->hc_ffn_base_f32 = fixture->hc_ffn_base_dev;
	weights->hc_ffn_scale_f32 = fixture->hc_ffn_scale_dev;
	weights->index_q_bf16 = fixture->index_q.device;
	weights->index_k_bf16 = fixture->index_k.device;
	weights->index_head_bf16 = fixture->index_head.device;
	weights->index_norm_weight_bf16 = fixture->index_norm_weight.device;
	weights->index_norm_bias_bf16 = fixture->index_norm_bias.device;
	weights->index_compress_gate_bf16 = fixture->index_compress_gate.device;
	weights->index_compress_ape_f32 = fixture->index_compress_ape_dev;
	weights->dense_gate_up_bf16 = fixture->dense_gate_up.device;
	weights->dense_down_bf16 = fixture->dense_down.device;
	weights->router_bf16 = fixture->router.device;
	weights->router_correction_f32 = fixture->router_correction;
	weights->shared_gate_up_bf16 = fixture->shared_gate_up.device;
	weights->shared_down_bf16 = fixture->shared_down.device;
	slot->hidden_bf16 = fixture->streams;
	slot->residual_bf16 = fixture->residual;
	slot->normed_bf16 = fixture->normed;
	slot->q_compressed_bf16 = fixture->q_compressed;
	slot->q_bf16 = fixture->q_bf16;
	slot->kv_slot_bf16 = fixture->kv_slot;
	slot->query_latent_bf16 = fixture->query_latent;
	slot->attention_latent_bf16 = fixture->attention_latent;
	slot->attention_value_bf16 = fixture->attention_value;
	slot->attention_out_bf16 = fixture->attention_out;
	slot->gate_up_bf16 = fixture->gate_up;
	slot->intermediate_bf16 = fixture->intermediate;
	slot->expert_out_bf16 = fixture->expert_out;
	slot->shared_out_bf16 = fixture->shared_out;
	slot->fused_qkvb_bf16 = fixture->fused_qkvb;
	slot->fused_decay_gate_bf16 = fixture->fused_decay_gate;
	slot->kda_decay_latent_bf16 = fixture->decay_latent;
	slot->kda_gate_latent_bf16 = fixture->gate_latent;
	slot->kda_beta_logit = fixture->kda_beta_logit;
	slot->kda_gate_bf16 = fixture->kda_gate_bf16;
	slot->kda_decay_logit_bf16 = fixture->kda_decay_logit;
	slot->kda_output_bf16 = fixture->kda_output;
	slot->hc_mixes_f32 = fixture->hc_mixes;
	slot->hc_pre_f32 = fixture->hc_pre;
	slot->hc_post_f32 = fixture->hc_post;
	slot->hc_comb_f32 = fixture->hc_comb;
	slot->hc_collapsed_bf16 = fixture->hc_collapsed;
	slot->hc_snapshot_bf16 = fixture->hc_snapshot;
	slot->hc_mean_bf16 = fixture->hc_mean;
	slot->kda_retention = fixture->kda_retention;
	slot->kda_write_gate = fixture->kda_write_gate;
	slot->index_query_bf16 = fixture->index_query;
	slot->index_key_bf16 = fixture->index_key;
	slot->index_gate_bf16 = fixture->index_gate;
	slot->index_packed_bf16 = fixture->index_packed;
	slot->selected_pools = fixture->selected_pools;
	slot->index_head_weight_bf16 = fixture->index_head_buf;
	slot->router_logits_f32 = fixture->router_logits;
	slot->selection_scores_f32 = fixture->selection_scores;
	slot->route_weight = fixture->route_weight;
	slot->route_expert = fixture->route_expert;
	slot->route_source_token = fixture->route_source_token;
	slot->route_packed_row = fixture->route_packed_row;
	slot->group_row_offset = fixture->group_row_offset;
	slot->group_tile_prefix_w1 = fixture->group_tile_prefix_w1;
	slot->group_tile_prefix_w2 = fixture->group_tile_prefix_w2;
	slot->selected_positions = fixture->selected_positions;
	slot->output_token = fixture->selected_pools;
	slot->output_score = fixture->selection_scores;
	slot->head_candidate_score = fixture->selection_scores;
	slot->head_candidate_token = fixture->selected_pools;
	slot->token_ids = fixture->token_ids;
	slot->resident_slots = fixture->resident_slots;
	slot->positions = fixture->positions;
	slot->context_lengths = fixture->context_lengths;
	slot->dense_row_offset = fixture->dense_row_offset;
	slot->dense_tile_prefix = fixture->dense_tile_prefix;
	slot->kv_access_error = fixture->kv_access_error;
	for (row = 0u; row < 8u; row++)
	{
		token_words[row] = token;
		slot_words[row] = 0u;
		position_words[row] = position;
		context_words[row] = position + 1u;
	}
	(void)cudaMemcpy(fixture->token_ids,token_words,sizeof(token_words),cudaMemcpyHostToDevice);
	(void)cudaMemcpy(fixture->resident_slots,slot_words,sizeof(slot_words),cudaMemcpyHostToDevice);
	(void)cudaMemcpy(fixture->positions,position_words,sizeof(position_words),cudaMemcpyHostToDevice);
	(void)cudaMemcpy(fixture->context_lengths,context_words,sizeof(context_words),cudaMemcpyHostToDevice);
	memset(wave,0,sizeof(*wave));
	wave->stage_index = 0u;
	wave->first_layer_index = layer;
	wave->layer_count = 1u;
	wave->tp_degree = 1u;
	wave->tp_rank = 0u;
	wave->row_count = 1u;
	wave->maximum_context = position + 1u;
	wave->resident_sequence_capacity = 1u;
	wave->max_sequence_positions = SPARK_LING_VALIDATION_PAGES * 64u;
	wave->pages_per_sequence = SPARK_LING_VALIDATION_PAGES;
	wave->owns_embedding = 0u;
	wave->owns_final_head = 0u;
	wave->hidden_input_bf16 = fixture->boundary_input;
	wave->boundary_row_offset = position;
	wave->layers = weights;
	wave->slot = slot;
	wave->kv_cache = fixture->kv_cache;
	wave->kv_layer_stride_bytes = (uint64_t)SPARK_LING_VALIDATION_PAGES * 64u * SPARK_LING_MODEL_KV_SLOT_BYTES;
	wave->index_cache = fixture->index_cache;
	wave->index_layer_stride_bytes = (uint64_t)SPARK_LING_VALIDATION_PAGES * 64u * SPARK_LING_VINDEX_PACKED * 2u;
	fixture->host_index_ordinals[0] = UINT32_MAX;
	fixture->host_kda_ordinals[0] = UINT32_MAX;
	if (SPARK_LING_MODEL_LAYER_IS_KDA(layer))
		fixture->host_kda_ordinals[0] = 0u;
	else
		fixture->host_index_ordinals[0] = 0u;
	wave->index_ordinal_by_local_layer = fixture->host_index_ordinals;
	wave->kda_ordinal_by_local_layer = fixture->host_kda_ordinals;
	wave->kda_state_pools = fixture->kda_state_pools;
	wave->kda_state_layer_stride_bytes = SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER;
	{
		uint64_t window_bytes = SPARK_LING_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER / 3u;
		wave->kda_q_window_pool = fixture->kda_window_pools;
		wave->kda_k_window_pool = fixture->kda_window_pools + window_bytes;
		wave->kda_v_window_pool = fixture->kda_window_pools + 2u * window_bytes;
		wave->kda_window_layer_stride_bytes = window_bytes;
	}
	wave->kda_state_index = fixture->kda_state_index;
	wave->page_table = fixture->page_table;
	wave->multiprocessor_count = fixture->multiprocessors;
	fixture->host_resident_slots_stage[0] = 0u;
	fixture->host_positions_stage[0] = position;
	fixture->host_token_ids_stage[0] = token;
	wave->host_resident_slots = fixture->host_resident_slots_stage;
	wave->host_positions = fixture->host_positions_stage;
	wave->host_token_ids = fixture->host_token_ids_stage;
}

static int SparkLingValRunTier1(SparkLingValFixture *fixture,uint32_t pass,uint16_t *streams_out)
{
	static const uint32_t tokens[SPARK_LING_VALIDATION_TOKENS] = {1u,3u,2u,6u};
	uint32_t step;
	int32_t status;
	for (step = 0u; step < SPARK_LING_VALIDATION_TOKENS; step++)
	{
		SparkLingValBuildWave(fixture,0u,tokens[step],step);
		status = SparkLingLaunchCudaWaveBegin(&fixture->wave);
		if (status != 0)
		{
			fprintf(stderr,"ling_validation tier1 begin status=%d\n",status);
			return(SparkLingValFail("tier1_begin","status"));
		}
		status = SparkLingLaunchCudaLayerAttention(&fixture->wave,0u);
		if (status != 0)
		{
			fprintf(stderr,"ling_validation tier1 attention status=%d step=%u pass=%u cuda=%s\n",
				status,step,pass,cudaGetErrorString(cudaGetLastError()));
			return(SparkLingValFail("tier1_attention","status"));
		}
		status = SparkLingLaunchCudaLayerAttentionPost(&fixture->wave,0u);
		if (status != 0)
			return(SparkLingValFail("tier1_attention_post","status"));
		status = SparkLingLaunchCudaLayerMlp(&fixture->wave,0u);
		if (status != 0)
		{
			fprintf(stderr,"ling_validation tier1 mlp status=%d step=%u pass=%u\n",status,step,pass);
			return(SparkLingValFail("tier1_mlp","status"));
		}
		status = SparkLingLaunchCudaLayerMlpPost(&fixture->wave,0u);
		if (status != 0)
			return(SparkLingValFail("tier1_mlp_post","status"));
		if (cudaStreamSynchronize(fixture->stream) != cudaSuccess)
			return(SparkLingValFail("tier1","sync"));
	}
	if (cudaMemcpy(streams_out,fixture->streams,
		(uint64_t)SPARK_LING_VHC_FLAT * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess)
		return(SparkLingValFail("tier1","streams_readback"));
	return(0);
}

static int SparkLingValRunTier1AttentionStep0(SparkLingValFixture *fixture)
{
	int32_t status;
	SparkLingValBuildWave(fixture,0u,1u,0u);
	status = SparkLingLaunchCudaWaveBegin(&fixture->wave);
	if (status != 0)
		return(SparkLingValFail("probe0_begin","status"));
	status = SparkLingLaunchCudaLayerAttention(&fixture->wave,0u);
	if (status != 0)
		return(SparkLingValFail("probe0_attention","status"));
	if (cudaStreamSynchronize(fixture->stream) != cudaSuccess)
		return(SparkLingValFail("probe0","sync"));
	return(0);
}

static int SparkLingValRunTier1AttentionOnly(SparkLingValFixture *fixture,uint16_t *sublayer_out)
{
	static const uint32_t tokens[SPARK_LING_VALIDATION_TOKENS] = {1u,3u,2u,6u};
	uint32_t step;
	int32_t status;
	for (step = 0u; step < SPARK_LING_VALIDATION_TOKENS; step++)
	{
		SparkLingValBuildWave(fixture,0u,tokens[step],step);
		status = SparkLingLaunchCudaWaveBegin(&fixture->wave);
		if (status != 0)
			return(SparkLingValFail("isolate_begin","status"));
		status = SparkLingLaunchCudaLayerAttention(&fixture->wave,0u);
		if (status != 0)
			return(SparkLingValFail("isolate_attention","status"));
		if (cudaStreamSynchronize(fixture->stream) != cudaSuccess)
			return(SparkLingValFail("isolate","sync"));
	}
	(void)sublayer_out;
	return(0);
}

static int SparkLingValRunTier2aAttention(SparkLingValFixture *fixture,uint32_t pass,uint16_t *streams_out)
{
	static const uint32_t tokens[SPARK_LING_VALIDATION_TOKENS] = {2u,5u,1u,4u};
	uint32_t step;
	int32_t status;
	SparkLingValBuildWave(fixture,3u,tokens[0],0u);
	{
		const SparkLingLayerWeights *w = &fixture->weights;
		const SparkLingExecutionSlot *slot = &fixture->slot;
		struct { const char *name; const void *pointer; } audit[] = {
			{"q_a",w->q_a_bf16},{"q_a_norm",w->q_a_norm_bf16},{"q_b",w->q_b_bf16},
			{"kv_a",w->kv_a_bf16},{"kv_a_norm",w->kv_a_norm_bf16},
			{"kv_b_key",w->kv_b_key_transposed_bf16},{"kv_b_value",w->kv_b_value_bf16},
			{"attn_output",w->attn_output_bf16},{"attn_norm",w->attn_norm_bf16},
			{"index_q",w->index_q_bf16},{"index_k",w->index_k_bf16},
			{"index_head",w->index_head_bf16},{"index_norm_w",w->index_norm_weight_bf16},
			{"index_norm_b",w->index_norm_bias_bf16},
			{"compress_ape",w->index_compress_ape_f32},{"compress_gate",w->index_compress_gate_bf16},
			{"hc_attn_fn",w->hc_attn_fn_f32},{"hc_attn_base",w->hc_attn_base_f32},
			{"hc_attn_scale",w->hc_attn_scale_f32},
			{"kv_slot",slot->kv_slot_bf16},{"q_compressed",slot->q_compressed_bf16},
			{"q_bf16",slot->q_bf16},{"query_latent",slot->query_latent_bf16},
			{"attention_latent",slot->attention_latent_bf16},
			{"attention_value",slot->attention_value_bf16},
			{"attention_out",slot->attention_out_bf16},
			{"index_query",slot->index_query_bf16},{"index_key",slot->index_key_bf16},
			{"index_gate",slot->index_gate_bf16},{"index_packed",slot->index_packed_bf16},
			{"index_head_w",slot->index_head_weight_bf16},
			{"hidden",slot->hidden_bf16},{"normed",slot->normed_bf16},
			{"selected",slot->selected_positions},{"selection_scores",slot->selection_scores_f32},
			{"hc_collapsed",slot->hc_collapsed_bf16},{"hc_snapshot",slot->hc_snapshot_bf16},
			{"hc_mixes",slot->hc_mixes_f32},
		};
		uint32_t i;
		for (i = 0u; i < sizeof(audit)/sizeof(audit[0]); i++)
			if (audit[i].pointer == 0)
				fprintf(stderr,"tier2a null: %s\n",audit[i].name);
	}
	for (step = 0u; step < SPARK_LING_VALIDATION_TOKENS; step++)
	{
		SparkLingValBuildWave(fixture,3u,tokens[step],step);
		status = SparkLingLaunchCudaWaveBegin(&fixture->wave);
		if (status != 0)
			return(SparkLingValFail("tier2a_begin","status"));
		status = SparkLingLaunchCudaLayerAttention(&fixture->wave,0u);
		if (status != 0)
		{
			fprintf(stderr,"ling_validation tier2a attention status=%d step=%u pass=%u\n",status,step,pass);
			return(SparkLingValFail("tier2a_attention","status"));
		}
		status = SparkLingLaunchCudaLayerAttentionPost(&fixture->wave,0u);
		if (status != 0)
			return(SparkLingValFail("tier2a_attention_post","status"));
		if (cudaStreamSynchronize(fixture->stream) != cudaSuccess)
			return(SparkLingValFail("tier2a","sync"));
	}
	if (cudaMemcpy(streams_out,fixture->streams,
		(uint64_t)SPARK_LING_VHC_FLAT * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess)
		return(SparkLingValFail("tier2a","streams_readback"));
	return(0);
}

static int SparkLingValCheckDeterminism(SparkLingValFixture *fixture,int (*driver)(SparkLingValFixture*,uint32_t,uint16_t*),const char *label)
{
	uint16_t *first = (uint16_t *)malloc((uint64_t)SPARK_LING_VHC_FLAT * sizeof(uint16_t));
	uint16_t *second = (uint16_t *)malloc((uint64_t)SPARK_LING_VHC_FLAT * sizeof(uint16_t));
	int status;
	if (first == 0 || second == 0)
	{
		free(first); free(second);
		return(SparkLingValFail(label,"determinism_alloc"));
	}
#define SPARK_LING_VAL_RESET_POOLS() do { \
	status = cudaMemset(fixture->kda_state_pools,0,SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER) == cudaSuccess && \
		cudaMemset(fixture->kda_window_pools,0,SPARK_LING_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER) == cudaSuccess && \
		cudaMemset(fixture->kv_cache,0,(uint64_t)SPARK_LING_VALIDATION_PAGES * 64u * SPARK_LING_MODEL_KV_SLOT_BYTES) == cudaSuccess && \
		cudaMemset(fixture->index_cache,0,(uint64_t)SPARK_LING_VALIDATION_PAGES * 64u * SPARK_LING_VINDEX_PACKED * 2u) == cudaSuccess ? 0 : 1; \
	if (status != 0) { free(first); free(second); return(SparkLingValFail(label,"determinism_reset")); } } while (0)
	SPARK_LING_VAL_RESET_POOLS();
	if (driver(fixture,0u,first) != 0)
	{
		free(first); free(second);
		return(SparkLingValFail(label,"determinism_first_walk"));
	}
	SPARK_LING_VAL_RESET_POOLS();
	if (driver(fixture,1u,second) != 0)
	{
		free(first); free(second);
		return(SparkLingValFail(label,"determinism_second_walk"));
	}
#undef SPARK_LING_VAL_RESET_POOLS
	{
		int identical = memcmp(first,second,(uint64_t)SPARK_LING_VHC_FLAT * sizeof(uint16_t)) == 0;
		printf("%s %-42s %s\n",identical ? "PASS" : "FAIL",label,
			identical ? "bit-exact re-walk" : "DIVERGED");
		free(first); free(second);
		return(identical ? 0 : 1);
	}
}


typedef struct SparkLingValOracleWalk
{
	float streams[SPARK_LING_VHC * SPARK_LING_VHIDDEN];
	uint16_t q_window[SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_CONV];
	uint16_t k_window[SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_CONV];
	uint16_t v_window[SPARK_LING_VKDA_DIM * SPARK_LING_VKDA_CONV];
	float state[(uint64_t)SPARK_LING_VKDA_HEADS * 128u * 128u];
} SparkLingValOracleWalk;

static void SparkLingValOracleTier1Token(SparkLingValOracleWalk *walk,const SparkLingValFixture *fixture,uint32_t position)
{
	float mixes[SPARK_LING_VHC_MIX],pre[SPARK_LING_VHC],post[SPARK_LING_VHC];
	float comb[SPARK_LING_VHC * SPARK_LING_VHC];
	float collapsed[SPARK_LING_VHIDDEN],snapshot[SPARK_LING_VHC * SPARK_LING_VHIDDEN];
	float sublayer[SPARK_LING_VHIDDEN];
	float normed[SPARK_LING_VHIDDEN];
	uint32_t index,i;
	for (i = 0u; i < SPARK_LING_VHC * SPARK_LING_VHIDDEN; i++)
		walk->streams[i] = SparkLingValFromBf16(
			fixture->boundary_host_rows[(uint64_t)position * SPARK_LING_VHIDDEN +
				(i % SPARK_LING_VHIDDEN)]);
	SparkLingValHcSite(walk->streams,fixture->hc_attn_fn_host,fixture->hc_attn_base_host,
		fixture->hc_attn_scale_host,SPARK_LING_MODEL_HC_EPSILON,
		SPARK_LING_VHC,SPARK_LING_VHIDDEN,mixes,pre,post,comb,collapsed,snapshot);
	SparkLingValKdaToken(collapsed,fixture->attn_norm.host,fixture->kda_qkv_beta.host,
		fixture->kda_q_conv.host,fixture->kda_k_conv.host,fixture->kda_v_conv.host,
		fixture->kda_decay_gate_down.host,fixture->kda_decay_up.host,
		fixture->kda_gate_up.host,fixture->kda_dt_bias_host,fixture->kda_a_log_host,
		fixture->kda_out_norm_host,fixture->kda_out.host,
		walk->q_window,walk->k_window,walk->v_window,walk->state,sublayer);
	SparkLingValHcPost(sublayer,snapshot,post,comb,SPARK_LING_VHC,
		SPARK_LING_VHIDDEN,walk->streams);
	SparkLingValHcSite(walk->streams,fixture->hc_ffn_fn_host,fixture->hc_ffn_base_host,
		fixture->hc_ffn_scale_host,SPARK_LING_MODEL_HC_EPSILON,
		SPARK_LING_VHC,SPARK_LING_VHIDDEN,mixes,pre,post,comb,collapsed,snapshot);
	for (index = 0u; index < SPARK_LING_VHIDDEN; index++)
		normed[index] = collapsed[index];
	SparkLingValRmsNorm(normed,fixture->mlp_norm.host,SPARK_LING_VHIDDEN,1e-5f);
	{
		static float gate_up[SPARK_LING_VGATE_UP_ROWS];
		float intermediate[SPARK_LING_VDENSE_INTER];
		for (index = 0u; index < SPARK_LING_VGATE_UP_ROWS; index++)
		{
			float sum = 0.0f;
			for (i = 0u; i < SPARK_LING_VHIDDEN; i++)
				sum += fixture->dense_gate_up.host[(uint64_t)index * SPARK_LING_VHIDDEN + i] * normed[i];
			gate_up[index] = sum;
		}
		for (index = 0u; index < SPARK_LING_VDENSE_INTER; index++)
		{
			float up = gate_up[index];
			float gate = gate_up[SPARK_LING_VDENSE_INTER + index];
			intermediate[index] = (gate * SparkLingValSigmoid(gate)) * up;
		}
		for (index = 0u; index < SPARK_LING_VHIDDEN; index++)
		{
			float sum = 0.0f;
			for (i = 0u; i < SPARK_LING_VDENSE_INTER; i++)
				sum += fixture->dense_down.host[(uint64_t)index * SPARK_LING_VDENSE_INTER + i] * intermediate[i];
			sublayer[index] = sum;
		}
	}
	SparkLingValHcPost(sublayer,snapshot,post,comb,SPARK_LING_VHC,
		SPARK_LING_VHIDDEN,walk->streams);
}

int main(int argc,char **argv)
{
	static SparkLingValFixture fixture;
	static SparkLingValOracleWalk walk;
	static uint16_t device_streams[SPARK_LING_VHC * SPARK_LING_VHIDDEN];
	float reference[SPARK_LING_VHC * SPARK_LING_VHIDDEN];
	float actual[SPARK_LING_VHC * SPARK_LING_VHIDDEN];
	SparkLingValMetrics metrics;
	int failures = 0;
	uint32_t step;
	if (argc != 2)
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	printf("ling validator: configuration %s\n",argv[1]);
	if (SparkLingValOracleSelftest() != 0)
		return(1);
	if (SparkLingValFixtureBuild(&fixture) != 0)
		return(1);
	if (SparkLingValFixtureComplete(&fixture) != 0)
		return(1);

	{
		float mixes[SPARK_LING_VHC_MIX],pre[SPARK_LING_VHC],post[SPARK_LING_VHC];
		float comb[SPARK_LING_VHC * SPARK_LING_VHC];
		float collapsed[SPARK_LING_VHIDDEN],snapshot[SPARK_LING_VHC * SPARK_LING_VHIDDEN];
		float sublayer[SPARK_LING_VHIDDEN];
		float device_collapsed[SPARK_LING_VHIDDEN];
		float device_sublayer[SPARK_LING_VHIDDEN];
		uint16_t read_collapsed[SPARK_LING_VHIDDEN];
		uint16_t read_sublayer[SPARK_LING_VHIDDEN];
		SparkLingValMetrics probe_metrics;
		uint32_t i;
		if (cudaMemset(fixture.kda_state_pools,0,SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER) != cudaSuccess ||
			cudaMemset(fixture.kda_window_pools,0,SPARK_LING_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER) != cudaSuccess)
			return(SparkLingValFail("probe","state_reset"));
		if (SparkLingValRunTier1AttentionStep0(&fixture) != 0)
			return(1);
		memset(&walk,0,sizeof(walk));
		for (step = 0u; step < SPARK_LING_VALIDATION_TOKENS; step++)
		{
			for (i = 0u; i < SPARK_LING_VHC * SPARK_LING_VHIDDEN; i++)
				walk.streams[i] = SparkLingValFromBf16(
					fixture.boundary_host_rows[(uint64_t)step * SPARK_LING_VHIDDEN +
						(i % SPARK_LING_VHIDDEN)]);
			SparkLingValHcSite(walk.streams,fixture.hc_attn_fn_host,fixture.hc_attn_base_host,
				fixture.hc_attn_scale_host,SPARK_LING_MODEL_HC_EPSILON,
				SPARK_LING_VHC,SPARK_LING_VHIDDEN,mixes,pre,post,comb,collapsed,snapshot);
			SparkLingValKdaToken(collapsed,fixture.attn_norm.host,fixture.kda_qkv_beta.host,
				fixture.kda_q_conv.host,fixture.kda_k_conv.host,fixture.kda_v_conv.host,
				fixture.kda_decay_gate_down.host,fixture.kda_decay_up.host,
				fixture.kda_gate_up.host,fixture.kda_dt_bias_host,fixture.kda_a_log_host,
				fixture.kda_out_norm_host,fixture.kda_out.host,
				walk.q_window,walk.k_window,walk.v_window,walk.state,sublayer);
			if (step == 0u)
			{
				printf("probe0 oracle sublayer[0..3] %f %f %f %f\n",
					sublayer[0],sublayer[1],sublayer[2],sublayer[3]);
				printf("probe0 oracle state head0 [0..3] %f %f %f %f\n",
					walk.state[0],walk.state[1],walk.state[2],walk.state[3]);
				break;
			}
		}
		if (cudaMemcpy(read_collapsed,fixture.hc_collapsed,
			(uint64_t)SPARK_LING_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(read_sublayer,fixture.attention_out,
			(uint64_t)SPARK_LING_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess)
			return(SparkLingValFail("probe","readback"));
		for (i = 0u; i < SPARK_LING_VHIDDEN; i++)
		{
			device_collapsed[i] = SparkLingValFromBf16(read_collapsed[i]);
			device_sublayer[i] = SparkLingValFromBf16(read_sublayer[i]);
		}
		SparkLingValMeasure(&probe_metrics,device_collapsed,collapsed,SPARK_LING_VHIDDEN);
		(void)SparkLingValReport("probe0 hc collapsed",&probe_metrics,0.02,0.999);
		SparkLingValMeasure(&probe_metrics,device_sublayer,sublayer,SPARK_LING_VHIDDEN);
		(void)SparkLingValReport("probe0 kda sublayer",&probe_metrics,0.05,0.995);
		printf("probe0 device sublayer[0..3] %f %f %f %f\n",
			device_sublayer[0],device_sublayer[1],device_sublayer[2],device_sublayer[3]);
		{
			float retention_probe[128];
			float write_gate_probe[SPARK_LING_VKDA_HEADS];
			float q_probe[128];
			uint16_t q_read[128];
			if (cudaMemcpy(retention_probe,fixture.kda_retention,128u * sizeof(float),cudaMemcpyDeviceToHost) == cudaSuccess &&
				cudaMemcpy(write_gate_probe,fixture.kda_write_gate,SPARK_LING_VKDA_HEADS * sizeof(float),cudaMemcpyDeviceToHost) == cudaSuccess &&
				cudaMemcpy(q_read,fixture.q_bf16,128u * sizeof(uint16_t),cudaMemcpyDeviceToHost) == cudaSuccess)
			{
				uint32_t j;
				float gate_probe[128];
				float v_probe[128];
				uint16_t gate_read[128];
				uint16_t v_read[128];
				printf("probe retention[0..3] %f %f %f %f  write_gate[0..3] %f %f %f %f\n",
					retention_probe[0],retention_probe[1],retention_probe[2],retention_probe[3],
					write_gate_probe[0],write_gate_probe[1],write_gate_probe[2],write_gate_probe[3]);
				for (j = 0u; j < 128u; j++)
					q_probe[j] = SparkLingValFromBf16(q_read[j]);
				printf("probe q_norm_sq %f\n",
					q_probe[0]*q_probe[0] + q_probe[1]*q_probe[1] + q_probe[2]*q_probe[2] + q_probe[3]*q_probe[3]);
				if (cudaMemcpy(gate_read,fixture.kda_gate_bf16,128u * sizeof(uint16_t),cudaMemcpyDeviceToHost) == cudaSuccess &&
					cudaMemcpy(v_read,fixture.gate_up,128u * sizeof(uint16_t),cudaMemcpyDeviceToHost) == cudaSuccess)
				{
					for (j = 0u; j < 8u; j++)
					{
						gate_probe[j] = SparkLingValFromBf16(gate_read[j]);
						v_probe[j] = SparkLingValFromBf16(v_read[j]);
					}
					printf("probe gate[0..3] %f %f %f %f  v[0..3] %f %f %f %f\n",
						gate_probe[0],gate_probe[1],gate_probe[2],gate_probe[3],
						v_probe[0],v_probe[1],v_probe[2],v_probe[3]);
				}
				{
					uint16_t y_read[8];
					if (cudaMemcpy(y_read,fixture.kv_slot,8u * sizeof(uint16_t),cudaMemcpyDeviceToHost) == cudaSuccess)
						printf("probe y post-norm-gate head0 [0..3] %f %f %f %f\n",
							SparkLingValFromBf16(y_read[0]),SparkLingValFromBf16(y_read[1]),
							SparkLingValFromBf16(y_read[2]),SparkLingValFromBf16(y_read[3]));
				}
				{
					static uint16_t y_all[SPARK_LING_VKDA_DIM];
					static float recomputed[SPARK_LING_VHIDDEN];
					static float actual_out[SPARK_LING_VHIDDEN];
					uint16_t out_read[SPARK_LING_VHIDDEN];
					uint32_t row, column;
					SparkLingValMetrics gemm_metrics;
					if (cudaMemcpy(y_all,fixture.kv_slot,
						(uint64_t)SPARK_LING_VKDA_DIM * sizeof(uint16_t),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(out_read,fixture.attention_out,
						(uint64_t)SPARK_LING_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) == cudaSuccess)
					{
						for (row = 0u; row < SPARK_LING_VHIDDEN; row++)
						{
							float acc = 0.0f;
							for (column = 0u; column < SPARK_LING_VKDA_DIM; column++)
								acc += SparkLingValFromBf16(y_all[column]) *
									fixture.kda_out.host[(uint64_t)row * SPARK_LING_VKDA_DIM + column];
							recomputed[row] = acc;
							actual_out[row] = SparkLingValFromBf16(out_read[row]);
						}
						SparkLingValMeasure(&gemm_metrics,actual_out,recomputed,
							SPARK_LING_VHIDDEN);
						(void)SparkLingValReport("probe o_proj gemm (device y vs host recompute)",
							&gemm_metrics,0.02,0.999);
						printf("probe o_proj recomputed[0..3] %f %f %f %f actual %f %f %f %f\n",
							recomputed[0],recomputed[1],recomputed[2],recomputed[3],
							actual_out[0],actual_out[1],actual_out[2],actual_out[3]);
					}
				}
			}
		}
	}

	if (cudaMemset(fixture.kda_state_pools,0,SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER) != cudaSuccess ||
		cudaMemset(fixture.kda_window_pools,0,SPARK_LING_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER) != cudaSuccess)
		return(SparkLingValFail("tier1","state_reset"));
	if (SparkLingValRunTier1(&fixture,0u,device_streams) != 0)
		return(1);
	memset(&walk,0,sizeof(walk));
	for (step = 0u; step < SPARK_LING_VALIDATION_TOKENS; step++)
		SparkLingValOracleTier1Token(&walk,&fixture,step);
	{
		uint64_t i;
		for (i = 0u; i < (uint64_t)SPARK_LING_VHC * SPARK_LING_VHIDDEN; i++)
		{
			reference[i] = walk.streams[i];
			actual[i] = SparkLingValFromBf16(device_streams[i]);
		}
	}
	SparkLingValMeasure(&metrics,actual,reference,
		(uint64_t)SPARK_LING_VHC * SPARK_LING_VHIDDEN);
	failures += SparkLingValReport("tier1 kda+dense+hc streams",&metrics,0.02,0.999);
	failures += SparkLingValCheckDeterminism(&fixture,SparkLingValRunTier1,"tier1 determinism");

	if (SparkLingValRunTier2aAttention(&fixture,0u,device_streams) != 0)
		return(1);
	failures += SparkLingValCheckDeterminism(&fixture,SparkLingValRunTier2aAttention,"tier2a determinism");

	printf("ling validator: %s (%d failures)\n",
		failures == 0 ? "PASS" : "FAIL",failures);
	return(failures == 0 ? 0 : 1);
}
#else
int main(void)
{
	return(SparkLingValOracleSelftest() == 0 ? 0 : 1);
}
#endif
