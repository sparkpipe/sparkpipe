#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_ling_model.h"
#include "sparkpipe/spark_ling_resident_decode_stage_firmware.h"
#include "spark_ling_resident_decode_stage_internal.h"


extern "C" int32_t SparkLingConfigureCudaModule(uint32_t *multiprocessor_count);
extern "C" int32_t SparkLingLaunchCudaWaveBegin(const SparkLingCudaWave *wave);
extern "C" int32_t SparkLingLaunchCudaLayerAttention(const SparkLingCudaWave *wave,uint32_t local_layer);
extern "C" int32_t SparkLingLaunchCudaLayerMlp(const SparkLingCudaWave *wave,uint32_t local_layer);
extern "C" int32_t SparkLingLaunchCudaWaveHead(const SparkLingCudaWave *wave);

#define SPARK_LING_VAL_TOKENS 4u
#define SPARK_LING_VAL_PREFILL_ROWS 4u
#define SPARK_LING_VAL_ROWS 8u
#define SPARK_LING_VAL_PAGES 34u
#define SPARK_LING_VAL_PAGE_SLOTS SPARK_LING_MODEL_KV_PAGE_SLOTS
#define SPARK_LING_VAL_SEQUENCES 8u

#define SPARK_LING_VAL_HIDDEN SPARK_LING_MODEL_HIDDEN_DIMENSION
#define SPARK_LING_VAL_RMS_EPS SPARK_LING_MODEL_RMS_NORM_EPSILON
#define SPARK_LING_VAL_KDA_HEADS SPARK_LING_MODEL_KDA_HEAD_COUNT
#define SPARK_LING_VAL_KDA_KEY SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION
#define SPARK_LING_VAL_KDA_VALUE SPARK_LING_MODEL_KDA_HEAD_VALUE_DIMENSION
#define SPARK_LING_VAL_KDA_QK (SPARK_LING_VAL_KDA_HEADS * SPARK_LING_VAL_KDA_KEY)
#define SPARK_LING_VAL_KDA_V (SPARK_LING_VAL_KDA_HEADS * SPARK_LING_VAL_KDA_VALUE)
#define SPARK_LING_VAL_KDA_FUSED \
	(2u * SPARK_LING_VAL_KDA_QK + SPARK_LING_VAL_KDA_V + SPARK_LING_VAL_KDA_HEADS)
#define SPARK_LING_VAL_KDA_CONV SPARK_LING_MODEL_KDA_SHORT_CONV_KERNEL
#define SPARK_LING_VAL_HEADS SPARK_LING_MODEL_MLA_HEAD_COUNT
#define SPARK_LING_VAL_NOPE SPARK_LING_MODEL_MLA_QK_NOPE_HEAD_DIMENSION
#define SPARK_LING_VAL_ROPE SPARK_LING_MODEL_MLA_QK_ROPE_HEAD_DIMENSION
#define SPARK_LING_VAL_HEAD_DIM (SPARK_LING_VAL_NOPE + SPARK_LING_VAL_ROPE)
#define SPARK_LING_VAL_Q_ROWS (SPARK_LING_VAL_HEADS * SPARK_LING_VAL_HEAD_DIM)
#define SPARK_LING_VAL_LATENT SPARK_LING_MODEL_MLA_LATENT_DIMENSION
#define SPARK_LING_VAL_KV_ROW SPARK_LING_MODEL_MLA_KV_A_DIMENSION
#define SPARK_LING_VAL_VALUE SPARK_LING_MODEL_MLA_VALUE_HEAD_DIMENSION
#define SPARK_LING_VAL_ATTN_COLS (SPARK_LING_VAL_HEADS * SPARK_LING_VAL_VALUE)
#define SPARK_LING_VAL_QK_SCALE SPARK_LING_MODEL_MLA_QK_SCALE
#define SPARK_LING_VAL_ROPE_THETA SPARK_LING_MODEL_MLA_ROPE_THETA
#define SPARK_LING_VAL_DENSE_INTER SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define SPARK_LING_VAL_DENSE_GATE_UP_ROWS (2u * SPARK_LING_VAL_DENSE_INTER)
#define SPARK_LING_VAL_EXPERTS SPARK_LING_MODEL_MOE_EXPERT_COUNT
#define SPARK_LING_VAL_TOP_K SPARK_LING_MODEL_MOE_TOP_K
#define SPARK_LING_VAL_GROUPS SPARK_LING_MODEL_MOE_ROUTER_GROUP_COUNT
#define SPARK_LING_VAL_TOP_GROUPS SPARK_LING_MODEL_MOE_ROUTER_TOP_GROUPS
#define SPARK_LING_VAL_EXPERT_INTER SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION
#define SPARK_LING_VAL_W1_ROWS (2u * SPARK_LING_VAL_EXPERT_INTER)
#define SPARK_LING_VAL_ROUTED_SCALE SPARK_LING_MODEL_MOE_ROUTED_SCALING_FACTOR
#define SPARK_LING_VAL_LOWER SPARK_LING_MODEL_KDA_GATE_LOWER_BOUND

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
	printf("%s %-44s rel_l2 %.5f (max %.5f) cos %.7f (min %.7f) maxabs %.3e\n",
		ok ? "PASS" : "FAIL",check,metrics->max_relative_l2,max_relative_l2,
		metrics->cosine,minimum_cosine,metrics->max_abs);
	return(ok ? 0 : 1);
}

#define SPARK_LING_VAL_CODEC_BF16 1u
#define SPARK_LING_VAL_CODEC_INT6 2u
#define SPARK_LING_VAL_CODEC_INT7 3u
#define SPARK_LING_VAL_CODEC_INT8 4u
#define SPARK_LING_VAL_CODEC_FP8 5u
#define SPARK_LING_VAL_CODEC_NVFP4 6u
#define SPARK_LING_VAL_CODEC_MXFP4 7u

#if defined(LING_EXPERT_WEIGHT_CODEC)
#if LING_EXPERT_WEIGHT_CODEC == 1
#define SPARK_LING_VAL_CODEC SPARK_LING_VAL_CODEC_BF16
#elif LING_EXPERT_WEIGHT_CODEC == 2
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
	return(codec == SPARK_LING_VAL_CODEC_BF16 ? 16u :
		(codec == SPARK_LING_VAL_CODEC_INT6 || codec == SPARK_LING_VAL_CODEC_NVFP4 ||
		 codec == SPARK_LING_VAL_CODEC_MXFP4 ? 4u : 8u));
}

static uint32_t SparkLingValCodecScaleGroup(uint32_t codec)
{
	return(codec == SPARK_LING_VAL_CODEC_BF16 ? 0u :
		(codec == SPARK_LING_VAL_CODEC_INT6 || codec == SPARK_LING_VAL_CODEC_MXFP4 ? 32u :
		(codec == SPARK_LING_VAL_CODEC_NVFP4 ? 16u : 128u)));
}

static uint32_t SparkLingValCodecSigned(uint32_t codec)
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

static uint64_t SparkLingValScaleGroupsPerRow(uint32_t codec,uint32_t columns)
{
	uint32_t group = SparkLingValCodecScaleGroup(codec);
	return(group == 0u ? 0u : ((uint64_t)columns + group - 1u) / group);
}

static uint64_t SparkLingValPayloadRowBytes(uint32_t codec,uint32_t columns)
{
	return(((uint64_t)columns * SparkLingValCodecStoredBits(codec) + 7u) / 8u);
}

static uint64_t SparkLingValPayloadExpertBytes(uint32_t codec,uint32_t rows,uint32_t columns)
{
	return((uint64_t)rows * SparkLingValPayloadRowBytes(codec,columns));
}

static uint64_t SparkLingValScaleExpertBytes(uint32_t codec,uint32_t rows,uint32_t columns)
{
	uint64_t groups = SparkLingValScaleGroupsPerRow(codec,columns);
	return(groups == 0u ? 0u : (uint64_t)rows * groups * sizeof(float));
}

static uint8_t SparkLingValPayloadCode(const uint8_t *payload,uint32_t codec,uint64_t flat_row,uint32_t column,uint32_t columns)
{
	uint32_t bits = SparkLingValCodecStoredBits(codec);
	uint64_t bit_index = flat_row * (uint64_t)columns * bits + (uint64_t)column * bits;
	uint64_t byte_index = bit_index >> 3u;
	uint32_t shift = (uint32_t)(bit_index & 7u);
	if ( bits >= 8u )
		return(payload[byte_index]);
	uint32_t value = payload[byte_index] >> shift;
	if ( shift + bits > 8u )
		value |= (uint32_t)payload[byte_index + 1u] << (8u - shift);
	return((uint8_t)(value & (bits == 4u ? 0xfu : 0x3fu)));
}

static float SparkLingValCodeValue(uint8_t code,uint32_t codec)
{
	if ( codec == SPARK_LING_VAL_CODEC_BF16 )
		return(0.0f);
	if ( codec == SPARK_LING_VAL_CODEC_FP8 )
		return(SparkLingValE4m3Decode(code));
	if ( codec == SPARK_LING_VAL_CODEC_NVFP4 || codec == SPARK_LING_VAL_CODEC_MXFP4 )
		return(SparkLingValE2m1Decode(code));
	if ( codec == SPARK_LING_VAL_CODEC_INT8 )
		return((float)(int32_t)(int8_t)code);
	if ( codec == SPARK_LING_VAL_CODEC_INT7 )
		return((float)(int32_t)(code & 0x7fu) - ((code & 0x7fu) >= 64u ? 128u : 0u));
	return((float)(int32_t)(code & 0x3fu) - ((code & 0x3fu) >= 32u ? 64u : 0u));
}

static float SparkLingValExpertWeight(
	const uint8_t *payload,const float *scales,uint32_t codec,
	uint32_t rows,uint32_t columns,uint32_t expert,
	uint32_t row,uint32_t column)
{
	if ( codec == SPARK_LING_VAL_CODEC_BF16 )
	{
		const uint16_t *packed = (const uint16_t *)payload;
		uint64_t index = ((((uint64_t)expert * rows) + row) * columns) + column;
		return(SparkLingValFromBf16(packed[index]));
	}
	uint64_t groups = SparkLingValScaleGroupsPerRow(codec,columns);
	uint64_t scale_index = ((((uint64_t)expert * rows) + row) * groups) +
		(column / SparkLingValCodecScaleGroup(codec));
	float scale = scales != 0 ? scales[scale_index] : 1.0f;
	uint64_t flat_row = (((uint64_t)expert * rows) + row);
	uint8_t code = SparkLingValPayloadCode(payload,codec,flat_row,column,columns);
	return(SparkLingValCodeValue(code,codec) * scale);
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

static void SparkLingValL2PerHead(float *row,uint32_t heads,uint32_t head_dim,float epsilon)
{
	uint32_t head,index;
	for (head = 0u; head < heads; head++)
	{
		float *section = row + (uint64_t)head * head_dim;
		float sum = 0.0f;
		for (index = 0u; index < head_dim; index++)
			sum += section[index] * section[index];
		float inverse = 1.0f / sqrtf(sum + epsilon);
		for (index = 0u; index < head_dim; index++)
			section[index] *= inverse;
	}
}

static void SparkLingValRopeInterleaved(float *section,uint32_t rope_dim,float position,float theta)
{
	uint32_t half = rope_dim / 2u,index;
	for (index = 0u; index < half; index++)
	{
		float angle = position * powf(theta,-2.0f * (float)index / (float)rope_dim);
		float c = cosf(angle),s = sinf(angle);
		float a = section[index * 2u],b = section[index * 2u + 1u];
		section[index * 2u] = a * c - b * s;
		section[index * 2u + 1u] = a * s + b * c;
	}
}

static void SparkLingValSiluMul(const float *gate_up,float *out,uint32_t dimension)
{
	uint32_t index;
	for (index = 0u; index < dimension; index++)
	{
		float up = gate_up[index];
		float gate = gate_up[dimension + index];
		out[index] = (gate / (1.0f + expf(-gate))) * up;
	}
}

static void SparkLingValRouter(const float *router,const float *correction,const float *hidden,
	uint32_t *selected,float *weights)
{
	static float scores[SPARK_LING_VAL_EXPERTS];
	static float biased[SPARK_LING_VAL_EXPERTS];
	uint32_t expert,group;
	for (expert = 0u; expert < SPARK_LING_VAL_EXPERTS; expert++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += router[(uint64_t)expert * SPARK_LING_VAL_HIDDEN + j] * hidden[j];
		scores[expert] = SparkLingValSigmoid(sum);
		biased[expert] = scores[expert] + correction[expert];
	}
	{
		uint32_t per_group = SPARK_LING_VAL_EXPERTS / SPARK_LING_VAL_GROUPS;
		float group_key[SPARK_LING_VAL_GROUPS];
		uint8_t chosen[SPARK_LING_VAL_GROUPS];
		for (group = 0u; group < SPARK_LING_VAL_GROUPS; group++)
		{
			float best = -3.0e38f,second = -3.0e38f;
			for (uint32_t member = 0u; member < per_group; member++)
			{
				float value = biased[group * per_group + member];
				if ( value > best ) { second = best; best = value; }
				else if ( value > second ) second = value;
			}
			group_key[group] = best + second;
			chosen[group] = 0u;
		}
		for (uint32_t round = 0u; round < SPARK_LING_VAL_TOP_GROUPS; round++)
		{
			uint32_t best = SPARK_LING_VAL_GROUPS;
			for (group = 0u; group < SPARK_LING_VAL_GROUPS; group++)
			{
				if ( chosen[group] != 0u )
					continue;
				if ( best == SPARK_LING_VAL_GROUPS ||
					group_key[group] > group_key[best] )
					best = group;
			}
			chosen[best] = 1u;
		}
		for (uint32_t slot = 0u; slot < SPARK_LING_VAL_TOP_K; slot++)
		{
			uint32_t best = SPARK_LING_VAL_EXPERTS;
			float best_key = -3.0e38f;
			for (expert = 0u; expert < SPARK_LING_VAL_EXPERTS; expert++)
			{
				uint32_t taken,prior;
				if ( chosen[expert / per_group] == 0u )
					continue;
				taken = 0u;
				for (prior = 0u; prior < slot; prior++)
					if ( selected[prior] == expert )
						taken = 1u;
				if ( taken != 0u )
					continue;
				if ( best == SPARK_LING_VAL_EXPERTS || biased[expert] > best_key )
				{
					best = expert;
					best_key = biased[expert];
				}
			}
			selected[slot] = best;
		}
	}
	{
		float total = 0.0f;
		for (uint32_t slot = 0u; slot < SPARK_LING_VAL_TOP_K; slot++)
		{
			weights[slot] = scores[selected[slot]];
			total += weights[slot];
		}
		for (uint32_t slot = 0u; slot < SPARK_LING_VAL_TOP_K; slot++)
			weights[slot] = (weights[slot] / (total + 1e-20f)) * SPARK_LING_VAL_ROUTED_SCALE;
	}
}

typedef struct SparkLingValKdaWeights
{
	const float *attn_norm;
	const float *post_norm;
	const float *qkv_beta;
	const float *conv_q;
	const float *conv_k;
	const float *conv_v;
	const float *decay_proj;
	const float *gate_proj;
	const float *dt_bias;
	const float *a_log;
	const float *out_norm;
	const float *out_weight;
} SparkLingValKdaWeights;

typedef struct SparkLingValMlaWeights
{
	const float *attn_norm;
	const float *post_norm;
	const float *q_proj;
	const float *kv_a;
	const float *kv_a_norm;
	const float *kv_b_key;
	const float *kv_b_value;
	const float *attn_gate;
	const float *o_proj;
} SparkLingValMlaWeights;

typedef struct SparkLingValKdaDump
{
	float normed[SPARK_LING_VAL_HIDDEN];
	float q_raw[SPARK_LING_VAL_KDA_QK];
	float k_raw[SPARK_LING_VAL_KDA_QK];
	float v_raw[SPARK_LING_VAL_KDA_V];
	float q_conv[SPARK_LING_VAL_KDA_QK];
	float k_conv[SPARK_LING_VAL_KDA_QK];
	float v_conv[SPARK_LING_VAL_KDA_V];
	float retention[SPARK_LING_VAL_KDA_QK];
	float beta[SPARK_LING_VAL_KDA_HEADS];
	float o[SPARK_LING_VAL_KDA_V];
} SparkLingValKdaDump;

static void SparkLingValKdaAttention(
	const SparkLingValKdaWeights *w,
	const float *hidden,const float *residual,
	uint16_t *q_window,uint16_t *k_window,uint16_t *v_window,
	float *state,float *output,SparkLingValKdaDump *dump)
{
	SparkLingValKdaDump local_dump;
	if ( dump == 0 )
		dump = &local_dump;
	const uint32_t heads = SPARK_LING_VAL_KDA_HEADS;
	const uint32_t key = SPARK_LING_VAL_KDA_KEY;
	const uint32_t qk = SPARK_LING_VAL_KDA_QK;
	const uint32_t v_dim = SPARK_LING_VAL_KDA_V;
	const uint32_t kernel = SPARK_LING_VAL_KDA_CONV;
	static float normed[SPARK_LING_VAL_HIDDEN];
	static float q[SPARK_LING_VAL_KDA_QK];
	static float k[SPARK_LING_VAL_KDA_QK];
	static float v[SPARK_LING_VAL_KDA_V];
	static float core[SPARK_LING_VAL_KDA_V];
	static float retention[SPARK_LING_VAL_KDA_QK];
	static float gate[SPARK_LING_VAL_KDA_V];
	float beta[SPARK_LING_VAL_KDA_HEADS];
	uint32_t index,head;
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		normed[index] = hidden[index] + (residual != 0 ? residual[index] : 0.0f);
	SparkLingValRmsNorm(normed,w->attn_norm,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_RMS_EPS);
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->qkv_beta[(uint64_t)index * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		q[index] = sum;
	}
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->qkv_beta[(qk + (uint64_t)index) * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		k[index] = sum;
	}
	for (index = 0u; index < v_dim; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->qkv_beta[(2u * qk + (uint64_t)index) * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		v[index] = sum;
	}
	for (head = 0u; head < heads; head++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->qkv_beta[(2u * qk + v_dim + (uint64_t)head) * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		beta[head] = SparkLingValSigmoid(sum);
	}
	memcpy(dump->q_raw,q,sizeof(dump->q_raw));
	memcpy(dump->k_raw,k,sizeof(dump->k_raw));
	memcpy(dump->v_raw,v,sizeof(dump->v_raw));
	for (index = 0u; index < qk; index++)
	{
		float window[SPARK_LING_VAL_KDA_CONV];
		float total;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			window[tap] = SparkLingValFromBf16(q_window[index * kernel + tap]);
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			q_window[index * kernel + tap] = q_window[index * kernel + tap + 1u];
		q_window[index * kernel + kernel - 1u] = SparkLingValBf16(q[index]);
		window[kernel - 1u] = SparkLingValFromBf16(q_window[index * kernel + kernel - 1u]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += window[tap] * w->conv_q[(uint64_t)index * kernel + tap];
		q[index] = total * SparkLingValSigmoid(total);
	}
	for (index = 0u; index < qk; index++)
	{
		float window[SPARK_LING_VAL_KDA_CONV];
		float total;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			window[tap] = SparkLingValFromBf16(k_window[index * kernel + tap]);
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			k_window[index * kernel + tap] = k_window[index * kernel + tap + 1u];
		k_window[index * kernel + kernel - 1u] = SparkLingValBf16(k[index]);
		window[kernel - 1u] = SparkLingValFromBf16(k_window[index * kernel + kernel - 1u]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += window[tap] * w->conv_k[(uint64_t)index * kernel + tap];
		k[index] = total * SparkLingValSigmoid(total);
	}
	for (index = 0u; index < v_dim; index++)
	{
		float window[SPARK_LING_VAL_KDA_CONV];
		float total;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			window[tap] = SparkLingValFromBf16(v_window[index * kernel + tap]);
		for (uint32_t tap = 0u; tap + 1u < kernel; tap++)
			v_window[index * kernel + tap] = v_window[index * kernel + tap + 1u];
		v_window[index * kernel + kernel - 1u] = SparkLingValBf16(v[index]);
		window[kernel - 1u] = SparkLingValFromBf16(v_window[index * kernel + kernel - 1u]);
		total = 0.0f;
		for (uint32_t tap = 0u; tap < kernel; tap++)
			total += window[tap] * w->conv_v[(uint64_t)index * kernel + tap];
		v[index] = total * SparkLingValSigmoid(total);
	}
	memcpy(dump->q_conv,q,sizeof(dump->q_conv));
	for (index = 0u; index < v_dim; index++)
		dump->v_conv[index] = v[index];
	SparkLingValL2PerHead(q,heads,key,SPARK_LING_VAL_RMS_EPS);
	SparkLingValL2PerHead(k,heads,key,SPARK_LING_VAL_RMS_EPS);
	for (index = 0u; index < qk; index++)
		dump->k_conv[index] = k[index];
	for (index = 0u; index < qk; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->decay_proj[(uint64_t)index * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		retention[index] = SparkLingValBoundedDecay(sum,
			w->dt_bias[index],w->a_log[index / key],SPARK_LING_VAL_LOWER);
	}
	for (index = 0u; index < v_dim; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->gate_proj[(uint64_t)index * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		gate[index] = sum;
	}
	for (head = 0u; head < heads; head++)
	{
		const float *qh = q + (uint64_t)head * key;
		const float *kh = k + (uint64_t)head * key;
		const float *vh = v + (uint64_t)head * key;
		const float *fh = retention + (uint64_t)head * key;
		float *sh = state + (uint64_t)head * key * key;
		float *oh = core + (uint64_t)head * key;
		float key_norm = 0.0f,query_norm = 0.0f;
		float nk,nq;
		float predicted[SPARK_LING_VAL_KDA_KEY];
		float lf[SPARK_LING_VAL_KDA_KEY];
		float lk[SPARK_LING_VAL_KDA_KEY];
		float lq[SPARK_LING_VAL_KDA_KEY];
		for (uint32_t c = 0u; c < key; c++)
		{
			key_norm += kh[c] * kh[c];
			query_norm += qh[c] * qh[c];
		}
		nk = 1.0f / sqrtf(key_norm + 1e-6f);
		nq = 1.0f / sqrtf(query_norm + 1e-6f) / sqrtf((float)key);
		for (uint32_t c = 0u; c < key; c++)
		{
			lk[c] = kh[c] * nk;
			lq[c] = qh[c] * nq;
			lf[c] = fh[c];
		}
		for (uint32_t e = 0u; e < key; e++)
		{
			float dot = 0.0f;
			for (uint32_t c = 0u; c < key; c++)
				dot += sh[(uint64_t)c * key + e] * lk[c] * lf[c];
			predicted[e] = dot;
		}
		for (uint32_t e = 0u; e < key; e++)
		{
			float delta = beta[head] * (vh[e] - predicted[e]);
			for (uint32_t c = 0u; c < key; c++)
				sh[(uint64_t)c * key + e] = lf[c] * sh[(uint64_t)c * key + e] + delta * lk[c];
		}
		for (uint32_t e = 0u; e < key; e++)
		{
			float dot = 0.0f;
			for (uint32_t c = 0u; c < key; c++)
				dot += sh[(uint64_t)c * key + e] * lq[c];
			oh[e] = dot;
		}
	}
	for (head = 0u; head < heads; head++)
	{
		float *row = core + (uint64_t)head * key;
		SparkLingValRmsNorm(row,w->out_norm,key,SPARK_LING_VAL_RMS_EPS);
		for (uint32_t e = 0u; e < key; e++)
			row[e] *= SparkLingValSigmoid(gate[(uint64_t)head * key + e]);
	}
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < v_dim; j++)
			sum += w->out_weight[(uint64_t)index * v_dim + j] * core[j];
		output[index] = sum;
	}
}

typedef struct SparkLingValMlaCache
{
	uint16_t *slots;
	uint32_t context;
} SparkLingValMlaCache;

static void SparkLingValMlaAttention(
	const SparkLingValMlaWeights *w,
	const float *hidden,const float *residual,
	uint32_t position,SparkLingValMlaCache *cache,
	float *output)
{
	const uint32_t heads = SPARK_LING_VAL_HEADS;
	const uint32_t nope = SPARK_LING_VAL_NOPE;
	const uint32_t rope = SPARK_LING_VAL_ROPE;
	const uint32_t latent = SPARK_LING_VAL_LATENT;
	static float normed[SPARK_LING_VAL_HIDDEN];
	static float q[SPARK_LING_VAL_Q_ROWS];
	static float kv[SPARK_LING_VAL_KV_ROW];
	static float query_latent[SPARK_LING_VAL_HEADS * SPARK_LING_VAL_LATENT];
	static float attention_latent[SPARK_LING_VAL_HEADS * SPARK_LING_VAL_LATENT];
	static float values[SPARK_LING_VAL_ATTN_COLS];
	uint32_t index,head;
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		normed[index] = hidden[index] + (residual != 0 ? residual[index] : 0.0f);
	SparkLingValRmsNorm(normed,w->attn_norm,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_RMS_EPS);
	for (index = 0u; index < SPARK_LING_VAL_Q_ROWS; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->q_proj[(uint64_t)index * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		q[index] = sum;
	}
	for (head = 0u; head < heads; head++)
		SparkLingValRopeInterleaved(q + (uint64_t)head * SPARK_LING_VAL_HEAD_DIM + nope,
			rope,(float)position,SPARK_LING_VAL_ROPE_THETA);
	for (index = 0u; index < SPARK_LING_VAL_KV_ROW; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->kv_a[(uint64_t)index * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		kv[index] = sum;
	}
	SparkLingValRmsNorm(kv,w->kv_a_norm,latent,SPARK_LING_VAL_RMS_EPS);
	SparkLingValRopeInterleaved(kv + latent,rope,(float)position,SPARK_LING_VAL_ROPE_THETA);
	for (index = 0u; index < SPARK_LING_VAL_KV_ROW; index++)
		cache->slots[(uint64_t)position * SPARK_LING_VAL_KV_ROW + index] =
			SparkLingValBf16(kv[index]);
	for (head = 0u; head < heads; head++)
	{
		const float *qh = q + (uint64_t)head * SPARK_LING_VAL_HEAD_DIM;
		for (uint32_t l = 0u; l < latent; l++)
		{
			float sum = 0.0f;
			for (uint32_t j = 0u; j < nope; j++)
				sum += w->kv_b_key[((((uint64_t)head * latent) + l) * nope) + j] * qh[j];
			query_latent[(uint64_t)head * latent + l] = sum;
		}
	}
	for (head = 0u; head < heads; head++)
	{
		const float *q_latent = query_latent + (uint64_t)head * latent;
		const float *q_rope = q + (uint64_t)head * SPARK_LING_VAL_HEAD_DIM + nope;
		float scores[SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS];
		float maximum = -3.0e38f,total = 0.0f;
		uint32_t position_index;
		for (position_index = 0u; position_index < cache->context; position_index++)
		{
			const uint16_t *slot = cache->slots + (uint64_t)position_index * SPARK_LING_VAL_KV_ROW;
			float dot = 0.0f;
			for (uint32_t l = 0u; l < latent; l++)
				dot += q_latent[l] * SparkLingValFromBf16(slot[l]);
			for (uint32_t r = 0u; r < rope; r++)
				dot += q_rope[r] * SparkLingValFromBf16(slot[latent + r]);
			scores[position_index] = dot * SPARK_LING_VAL_QK_SCALE;
			if ( scores[position_index] > maximum )
				maximum = scores[position_index];
		}
		for (position_index = 0u; position_index < cache->context; position_index++)
		{
			scores[position_index] = expf(scores[position_index] - maximum);
			total += scores[position_index];
		}
		for (uint32_t l = 0u; l < latent; l++)
		{
			float sum = 0.0f;
			for (position_index = 0u; position_index < cache->context; position_index++)
			{
				const uint16_t *slot = cache->slots + (uint64_t)position_index * SPARK_LING_VAL_KV_ROW;
				sum += (scores[position_index] / total) * SparkLingValFromBf16(slot[l]);
			}
			attention_latent[(uint64_t)head * latent + l] = sum;
		}
	}
	for (head = 0u; head < heads; head++)
	{
		float gate_logit = 0.0f;
		float gate;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			gate_logit += w->attn_gate[(uint64_t)head * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		gate = SparkLingValSigmoid(gate_logit);
		for (uint32_t e = 0u; e < SPARK_LING_VAL_VALUE; e++)
		{
			float sum = 0.0f;
			for (uint32_t l = 0u; l < latent; l++)
				sum += w->kv_b_value[((((uint64_t)head * SPARK_LING_VAL_VALUE) + e) * latent) + l] *
					attention_latent[(uint64_t)head * latent + l];
			values[(uint64_t)head * SPARK_LING_VAL_VALUE + e] = sum * gate;
		}
	}
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_ATTN_COLS; j++)
			sum += w->o_proj[(uint64_t)index * SPARK_LING_VAL_ATTN_COLS + j] * values[j];
		output[index] = sum;
	}
}

static void SparkLingValDenseMlp(
	const float *post_norm,const float *gate_up_weight,const float *down_weight,
	const float *hidden,const float *residual,float *output)
{
	static float normed[SPARK_LING_VAL_HIDDEN];
	static float gate_up[SPARK_LING_VAL_DENSE_GATE_UP_ROWS];
	static float intermediate[SPARK_LING_VAL_DENSE_INTER];
	uint32_t index;
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		normed[index] = hidden[index] + (residual != 0 ? residual[index] : 0.0f);
	SparkLingValRmsNorm(normed,post_norm,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_RMS_EPS);
	for (index = 0u; index < SPARK_LING_VAL_DENSE_GATE_UP_ROWS; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += gate_up_weight[(uint64_t)index * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		gate_up[index] = sum;
	}
	SparkLingValSiluMul(gate_up,intermediate,SPARK_LING_VAL_DENSE_INTER);
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_DENSE_INTER; j++)
			sum += down_weight[(uint64_t)index * SPARK_LING_VAL_DENSE_INTER + j] * intermediate[j];
		output[index] = sum;
	}
}

typedef struct SparkLingValMoeWeights
{
	const float *post_norm;
	const float *router;
	const float *correction;
	const uint8_t *w1_payload;
	const float *w1_scales;
	const uint8_t *w2_payload;
	const float *w2_scales;
	const float *shared_gate_up;
	const float *shared_down;
	uint32_t codec;
} SparkLingValMoeWeights;

static void SparkLingValMoe(
	const SparkLingValMoeWeights *w,
	const float *hidden,const float *residual,float *output,
	uint32_t *selected_out,float *weights_out)
{
	static float normed[SPARK_LING_VAL_HIDDEN];
	static float intermediate[SPARK_LING_VAL_EXPERT_INTER];
	static float routed[SPARK_LING_VAL_HIDDEN];
	static float scratch[SPARK_LING_VAL_W1_ROWS];
	uint32_t index,slot;
	uint32_t selected[SPARK_LING_VAL_TOP_K];
	float weights[SPARK_LING_VAL_TOP_K];
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		normed[index] = hidden[index] + (residual != 0 ? residual[index] : 0.0f);
	SparkLingValRmsNorm(normed,w->post_norm,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_RMS_EPS);
	SparkLingValRouter(w->router,w->correction,normed,selected,weights);
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		routed[index] = 0.0f;
	for (slot = 0u; slot < SPARK_LING_VAL_TOP_K; slot++)
	{
		uint32_t expert = selected[slot];
		for (uint32_t row = 0u; row < SPARK_LING_VAL_W1_ROWS; row++)
		{
			float sum = 0.0f;
			for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
				sum += SparkLingValExpertWeight(w->w1_payload,w->w1_scales,w->codec,
					SPARK_LING_VAL_W1_ROWS,SPARK_LING_VAL_HIDDEN,expert,row,j) * normed[j];
			scratch[row] = sum;
		}
		SparkLingValSiluMul(scratch,intermediate,SPARK_LING_VAL_EXPERT_INTER);
		for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		{
			float sum = 0.0f;
			for (uint32_t j = 0u; j < SPARK_LING_VAL_EXPERT_INTER; j++)
				sum += SparkLingValExpertWeight(w->w2_payload,w->w2_scales,w->codec,
					SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_EXPERT_INTER,expert,index,j) *
					intermediate[j];
			routed[index] += weights[slot] * sum;
		}
	}
	for (index = 0u; index < SPARK_LING_VAL_W1_ROWS; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
			sum += w->shared_gate_up[(uint64_t)index * SPARK_LING_VAL_HIDDEN + j] * normed[j];
		scratch[index] = sum;
	}
	SparkLingValSiluMul(scratch,intermediate,SPARK_LING_VAL_EXPERT_INTER);
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
	{
		float sum = 0.0f;
		for (uint32_t j = 0u; j < SPARK_LING_VAL_EXPERT_INTER; j++)
			sum += w->shared_down[(uint64_t)index * SPARK_LING_VAL_EXPERT_INTER + j] * intermediate[j];
		output[index] = routed[index] + sum;
	}
	for (slot = 0u; slot < SPARK_LING_VAL_TOP_K; slot++)
	{
		selected_out[slot] = selected[slot];
		weights_out[slot] = weights[slot];
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
		fabsf(SparkLingValBoundedDecay(0.0f,0.0f,0.0f,SPARK_LING_VAL_LOWER) -
			expf(SPARK_LING_VAL_LOWER * 0.5f)) < 1e-6f,"bounded decay at zero");
	{
		float value = SparkLingValBoundedDecay(2.0f,-1.0f,0.5f,SPARK_LING_VAL_LOWER);
		float expect = expf(SPARK_LING_VAL_LOWER * SparkLingValSigmoid(expf(0.5f) * 1.0f));
		failures += SparkLingValSelftestAssert(fabsf(value - expect) < 1e-6f,
			"bounded decay formula");
		failures += SparkLingValSelftestAssert(value > 0.0f && value <= 1.0f,
			"bounded decay range (0,1]");
	}
	{
		uint32_t mismatch = 0u;
		for (uint32_t i = 0u; i < 256u; i++)
		{
			float value = ((float)(int32_t)(i % 31u) - 15.0f) * 0.1f;
			float back = SparkLingValFromBf16(SparkLingValBf16(value));
			if ( (back == 0.0f) != (value == 0.0f) )
				mismatch += 1u;
		}
		failures += SparkLingValSelftestAssert(mismatch == 0u,
			"bf16 round trip preserves zero and nonzero");
	}
	{
		float section[8] = {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
		SparkLingValRopeInterleaved(section,8u,0.0f,SPARK_LING_VAL_ROPE_THETA);
		failures += SparkLingValSelftestAssert(fabsf(section[0] - 1.0f) < 1e-6f &&
			fabsf(section[1]) < 1e-6f,"rope at position zero is identity");
	}
	{
		uint32_t selected[SPARK_LING_VAL_TOP_K];
		float weights[SPARK_LING_VAL_TOP_K];
		static float router[SPARK_LING_VAL_EXPERTS * SPARK_LING_VAL_HIDDEN];
		static float hidden[SPARK_LING_VAL_HIDDEN];
		float correction[SPARK_LING_VAL_EXPERTS];
		float total = 0.0f;
		int distinct = 1,grouped = 1;
		for (uint64_t i = 0u; i < (uint64_t)SPARK_LING_VAL_EXPERTS * SPARK_LING_VAL_HIDDEN; i++)
			router[i] = ((float)(int32_t)(i % 17u) - 8.0f) * 0.05f;
		for (uint32_t i = 0u; i < SPARK_LING_VAL_HIDDEN; i++)
			hidden[i] = ((float)(int32_t)(i % 7u) - 3.0f) * 0.05f;
		for (uint32_t e = 0u; e < SPARK_LING_VAL_EXPERTS; e++)
			correction[e] = e < 8u ? 4.0f : -4.0f;
		SparkLingValRouter(router,correction,hidden,selected,weights);
		for (uint32_t i = 0u; i < SPARK_LING_VAL_TOP_K; i++)
		{
			total += weights[i];
			for (uint32_t j = i + 1u; j < SPARK_LING_VAL_TOP_K; j++)
				if ( selected[i] == selected[j] )
					distinct = 0;
			if ( selected[i] / (SPARK_LING_VAL_EXPERTS / SPARK_LING_VAL_GROUPS) >= 1u )
				grouped = 0;
		}
		failures += SparkLingValSelftestAssert(distinct,"router selects distinct experts");
		failures += SparkLingValSelftestAssert(grouped,
			"router honours the group limit under a skewed bias");
		failures += SparkLingValSelftestAssert(fabsf(total - SPARK_LING_VAL_ROUTED_SCALE) < 1e-3f,
			"router weights renormalise to the routed scale");
	}
	{
		uint32_t codec = SPARK_LING_VAL_CODEC;
		uint64_t row_bytes = SparkLingValPayloadRowBytes(codec,256u);
		failures += SparkLingValSelftestAssert(
			SparkLingValPayloadExpertBytes(codec,64u,256u) == 64u * row_bytes,
			"payload slab = rows*row_bytes");
		failures += SparkLingValSelftestAssert(
			SparkLingValScaleExpertBytes(codec,64u,256u) ==
				64u * SparkLingValScaleGroupsPerRow(codec,256u) * sizeof(float),
			"scale slab = rows*groups*4");
	}
	{
		static float hidden[SPARK_LING_VAL_HIDDEN];
		static float output[SPARK_LING_VAL_HIDDEN];
		static float norm[SPARK_LING_VAL_HIDDEN];
		static float qkv[(uint64_t)SPARK_LING_VAL_KDA_FUSED * SPARK_LING_VAL_HIDDEN];
		static float conv[3u][SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV];
		static float proj[2u][SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_HIDDEN];
		static float dt[SPARK_LING_VAL_KDA_QK];
		static float alog[SPARK_LING_VAL_KDA_HEADS];
		static float onorm[SPARK_LING_VAL_KDA_KEY];
		static float ow[SPARK_LING_VAL_HIDDEN * SPARK_LING_VAL_KDA_V];
		static float state[(uint64_t)SPARK_LING_VAL_KDA_HEADS * SPARK_LING_VAL_KDA_KEY * SPARK_LING_VAL_KDA_KEY];
		static uint16_t windows[3u][SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV];
		SparkLingValKdaWeights kda;
		uint64_t i;
		float magnitude = 0.0f;
		int finite = 1;
		for (i = 0u; i < SPARK_LING_VAL_HIDDEN; i++)
		{
			hidden[i] = ((float)(int32_t)(i % 11u) - 5.0f) * 0.02f;
			norm[i] = 1.0f;
		}
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_KDA_FUSED * SPARK_LING_VAL_HIDDEN; i++)
			qkv[i] = ((float)(int32_t)(i % 7u) - 3.0f) * 0.004f;
		for (uint32_t c = 0u; c < 3u; c++)
			for (i = 0u; i < (uint64_t)SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV; i++)
				conv[c][i] = ((float)(int32_t)(i % 5u) - 2.0f) * 0.05f;
		for (uint32_t p = 0u; p < 2u; p++)
			for (i = 0u; i < (uint64_t)SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_HIDDEN; i++)
				proj[p][i] = ((float)(int32_t)(i % 9u) - 4.0f) * 0.004f;
		for (i = 0u; i < SPARK_LING_VAL_KDA_QK; i++)
			dt[i] = -0.25f;
		for (i = 0u; i < SPARK_LING_VAL_KDA_HEADS; i++)
			alog[i] = 0.1f;
		for (i = 0u; i < SPARK_LING_VAL_KDA_KEY; i++)
			onorm[i] = 1.0f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_HIDDEN * SPARK_LING_VAL_KDA_V; i++)
			ow[i] = ((float)(int32_t)(i % 7u) - 3.0f) * 0.005f;
		memset(windows,0,sizeof(windows));
		memset(state,0,sizeof(state));
		kda.attn_norm = norm;
		kda.post_norm = norm;
		kda.qkv_beta = qkv;
		kda.conv_q = conv[0];
		kda.conv_k = conv[1];
		kda.conv_v = conv[2];
		kda.decay_proj = proj[0];
		kda.gate_proj = proj[1];
		kda.dt_bias = dt;
		kda.a_log = alog;
		kda.out_norm = onorm;
		kda.out_weight = ow;
		for (uint32_t step = 0u; step < 3u; step++)
			SparkLingValKdaAttention(&kda,hidden,0,
				windows[0],windows[1],windows[2],state,output,0);
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
		{
			if (!(output[j] == output[j]) || fabsf(output[j]) > 1e30f)
				finite = 0;
			magnitude += fabsf(output[j]);
		}
		failures += SparkLingValSelftestAssert(finite,"kda oracle output finite");
		failures += SparkLingValSelftestAssert(magnitude > 0.0f,"kda oracle output non-zero");
	}
	{
		static float hidden[SPARK_LING_VAL_HIDDEN];
		static float output[SPARK_LING_VAL_HIDDEN];
		static float norm[SPARK_LING_VAL_LATENT];
		static float q_proj[SPARK_LING_VAL_Q_ROWS * SPARK_LING_VAL_HIDDEN];
		static float kv_a[SPARK_LING_VAL_KV_ROW * SPARK_LING_VAL_HIDDEN];
		static float kv_b_key[(uint64_t)SPARK_LING_VAL_HEADS * SPARK_LING_VAL_LATENT * SPARK_LING_VAL_NOPE];
		static float kv_b_value[(uint64_t)SPARK_LING_VAL_HEADS * SPARK_LING_VAL_VALUE * SPARK_LING_VAL_LATENT];
		static float gate[SPARK_LING_VAL_HEADS * SPARK_LING_VAL_HIDDEN];
		static float o_proj[(uint64_t)SPARK_LING_VAL_HIDDEN * SPARK_LING_VAL_ATTN_COLS];
		static uint16_t cache_slots[4u * SPARK_LING_VAL_KV_ROW];
		SparkLingValMlaWeights mla;
		SparkLingValMlaCache cache;
		uint64_t i;
		float magnitude = 0.0f;
		int finite = 1;
		for (i = 0u; i < SPARK_LING_VAL_HIDDEN; i++)
			hidden[i] = ((float)(int32_t)(i % 13u) - 6.0f) * 0.015f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_Q_ROWS * SPARK_LING_VAL_HIDDEN; i++)
			q_proj[i] = ((float)(int32_t)(i % 7u) - 3.0f) * 0.008f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_KV_ROW * SPARK_LING_VAL_HIDDEN; i++)
			kv_a[i] = ((float)(int32_t)(i % 9u) - 4.0f) * 0.008f;
		for (i = 0u; i < SPARK_LING_VAL_LATENT; i++)
			norm[i] = 1.0f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_HEADS * SPARK_LING_VAL_LATENT * SPARK_LING_VAL_NOPE; i++)
			kv_b_key[i] = ((float)(int32_t)(i % 5u) - 2.0f) * 0.01f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_HEADS * SPARK_LING_VAL_VALUE * SPARK_LING_VAL_LATENT; i++)
			kv_b_value[i] = ((float)(int32_t)(i % 7u) - 3.0f) * 0.01f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_HEADS * SPARK_LING_VAL_HIDDEN; i++)
			gate[i] = ((float)(int32_t)(i % 11u) - 5.0f) * 0.008f;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_HIDDEN * SPARK_LING_VAL_ATTN_COLS; i++)
			o_proj[i] = ((float)(int32_t)(i % 11u) - 5.0f) * 0.004f;
		mla.attn_norm = norm;
		mla.post_norm = norm;
		mla.q_proj = q_proj;
		mla.kv_a = kv_a;
		mla.kv_a_norm = norm;
		mla.kv_b_key = kv_b_key;
		mla.kv_b_value = kv_b_value;
		mla.attn_gate = gate;
		mla.o_proj = o_proj;
		cache.slots = cache_slots;
		for (uint32_t step = 0u; step < 3u; step++)
		{
			cache.context = step + 1u;
			SparkLingValMlaAttention(&mla,hidden,0,step,&cache,output);
		}
		for (uint32_t j = 0u; j < SPARK_LING_VAL_HIDDEN; j++)
		{
			if (!(output[j] == output[j]) || fabsf(output[j]) > 1e30f)
				finite = 0;
			magnitude += fabsf(output[j]);
		}
		failures += SparkLingValSelftestAssert(finite,"mla oracle output finite");
		failures += SparkLingValSelftestAssert(magnitude > 0.0f,"mla oracle output non-zero");
	}
	if (failures == 0)
		printf("ling_validator_selftest PASS (bounded decay, bf16 round trip, "
		       "rope identity, group-limited router, codec addressing, "
		       "KDA/MLA oracle smoke)\n");
	return(failures);
}

#ifndef SPARK_LING_VALIDATOR_ORACLE_SELFTEST

typedef struct SparkLingValFixture
{
	SparkLingLayerWeights layer_weights[2];
	SparkLingExecutionSlot slot;
	SparkLingCudaWave wave;
	cudaStream_t stream;
	uint32_t multiprocessors;
	SparkLingValMatrix dense_attn_norm,dense_post_norm,dense_gate_up,dense_down;
	SparkLingValMatrix kda_qkv_beta,kda_conv_q,kda_conv_k,kda_conv_v;
	SparkLingValMatrix kda_decay_proj,kda_gate_proj,kda_out;
	SparkLingValMatrix mla_attn_norm,mla_post_norm,mla_q_proj,mla_kv_a,mla_kv_a_norm;
	SparkLingValMatrix mla_kv_b_key,mla_kv_b_value,mla_attn_gate,mla_o_proj;
	SparkLingValMatrix router,shared_gate_up,shared_down;
	float *router_correction_dev;
	float router_correction_host[SPARK_LING_VAL_EXPERTS];
	uint8_t *w1_payload_dev,*w2_payload_dev;
	float *w1_scales_dev,*w2_scales_dev;
	uint8_t *w1_payload_host,*w2_payload_host;
	float *w1_scales_host,*w2_scales_host;
	uint64_t w1_payload_bytes,w2_payload_bytes,w1_scale_bytes,w2_scale_bytes;
	float *kda_out_norm_dev,*kda_dt_bias_dev,*kda_a_log_dev;
	float kda_out_norm_host[SPARK_LING_VAL_KDA_KEY];
	float kda_dt_bias_host[SPARK_LING_VAL_KDA_QK];
	float kda_a_log_host[SPARK_LING_VAL_KDA_HEADS];
	uint16_t *normed_dev,*hidden_dev,*boundary_in_dev,*boundary_out_dev;
	uint16_t *boundary_host;
	uint16_t *q_dev,*query_latent_dev,*query_rope_dev,*attn_gate_dev,*kv_slot_dev;
	uint16_t *attention_latent_dev,*attention_value_dev,*attention_out_dev;
	uint16_t *gate_up_dev,*intermediate_dev,*expert_out_dev,*shared_out_dev;
	uint16_t *fused_qkvb_dev,*kda_beta_logit_dev,*kda_gate_dev,*kda_decay_logit_dev;
	float *kda_retention_dev,*kda_write_gate_dev,*router_logits_dev;
	uint32_t *route_expert_dev,*route_source_token_dev,*route_packed_row_dev;
	float *route_weight_dev;
	uint32_t *group_row_offset_dev,*group_tile_prefix_w1_dev,*group_tile_prefix_w2_dev;
	uint32_t *token_ids_dev,*resident_slots_dev,*positions_dev,*context_lengths_dev;
	uint32_t *dense_row_offset_dev,*dense_tile_prefix_dev;
	uint32_t *page_table_dev,*kda_state_index_dev,*run_begin_dev,*run_state_dev;
	uint32_t host_run_begin[2];
	uint32_t host_run_state[1];
	uint32_t host_positions[SPARK_LING_VAL_ROWS];
	uint32_t host_slots[SPARK_LING_VAL_ROWS];
	uint32_t host_kv_ordinals[2];
	uint32_t host_kda_ordinals[2];
	uint32_t kda_ordinals_used;
	uint32_t kv_ordinals_used;
	uint64_t *head_maxloc_dev;
	uint32_t *kv_access_error_dev;
	uint8_t *kv_cache_dev,*kda_state_pool_dev,*kda_window_pool_dev;
	uint32_t host_page_table[SPARK_LING_VAL_PAGES];
} SparkLingValFixture;

static int SparkLingValSynthExperts(SparkLingValFixture *fixture)
{
	uint32_t codec = SPARK_LING_VAL_CODEC;
	uint64_t w1_bytes = (uint64_t)SPARK_LING_VAL_EXPERTS *
		SparkLingValPayloadExpertBytes(codec,SPARK_LING_VAL_W1_ROWS,SPARK_LING_VAL_HIDDEN);
	uint64_t w2_bytes = (uint64_t)SPARK_LING_VAL_EXPERTS *
		SparkLingValPayloadExpertBytes(codec,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_EXPERT_INTER);
	uint64_t w1_scales = (uint64_t)SPARK_LING_VAL_EXPERTS *
		SparkLingValScaleExpertBytes(codec,SPARK_LING_VAL_W1_ROWS,SPARK_LING_VAL_HIDDEN);
	uint64_t w2_scales = (uint64_t)SPARK_LING_VAL_EXPERTS *
		SparkLingValScaleExpertBytes(codec,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_EXPERT_INTER);
	fixture->w1_payload_bytes = w1_bytes;
	fixture->w2_payload_bytes = w2_bytes;
	fixture->w1_scale_bytes = w1_scales;
	fixture->w2_scale_bytes = w2_scales;
	fixture->w1_payload_host = (uint8_t *)malloc(w1_bytes);
	fixture->w2_payload_host = (uint8_t *)malloc(w2_bytes);
	fixture->w1_scales_host = w1_scales != 0u ? (float *)malloc(w1_scales) : 0;
	fixture->w2_scales_host = w2_scales != 0u ? (float *)malloc(w2_scales) : 0;
	if (fixture->w1_payload_host == 0 || fixture->w2_payload_host == 0 ||
		(w1_scales != 0u && fixture->w1_scales_host == 0) ||
		(w2_scales != 0u && fixture->w2_scales_host == 0))
		return(SparkLingValFail("fixture","expert_host_alloc"));
	{
		uint64_t i;
		uint8_t mask = SparkLingValCodecSigned(codec) != 0u ? 0xffu : 0x7eu;
		for (i = 0u; i < w1_bytes; i++)
			fixture->w1_payload_host[i] = (uint8_t)((SparkLingValNext() >> 24) & mask);
		for (i = 0u; i < w2_bytes; i++)
			fixture->w2_payload_host[i] = (uint8_t)((SparkLingValNext() >> 24) & mask);
		if ( codec == SPARK_LING_VAL_CODEC_BF16 )
		{
			uint16_t *packed = (uint16_t *)fixture->w1_payload_host;
			uint32_t row;
			for (row = 0u; row < SPARK_LING_VAL_W1_ROWS * SPARK_LING_VAL_EXPERTS; row++)
				for (uint32_t column = 0u; column < SPARK_LING_VAL_HIDDEN; column++)
				{
					uint64_t index = ((uint64_t)row * SPARK_LING_VAL_HIDDEN) + column;
					packed[index] = SparkLingValBf16(
						((float)(int32_t)((index * 37u) % 201u) - 100.0f) * 0.002f);
				}
			packed = (uint16_t *)fixture->w2_payload_host;
			for (uint64_t index = 0u; index < w2_bytes / 2u; index++)
				packed[index] = SparkLingValBf16(
					((float)(int32_t)((index * 41u) % 201u) - 100.0f) * 0.002f);
		}
		if ( w1_scales != 0u )
			for (i = 0u; i < w1_scales / sizeof(float); i++)
				fixture->w1_scales_host[i] = 1.0f + ((float)(int32_t)(i % 7u) - 3.0f) * 0.01f;
		if ( w2_scales != 0u )
			for (i = 0u; i < w2_scales / sizeof(float); i++)
				fixture->w2_scales_host[i] = 1.0f + ((float)(int32_t)(i % 5u) - 2.0f) * 0.01f;
	}
	if ( cudaMalloc((void **)&fixture->w1_payload_dev,w1_bytes) != cudaSuccess ||
		cudaMalloc((void **)&fixture->w2_payload_dev,w2_bytes) != cudaSuccess )
		return(SparkLingValFail("fixture","expert_device_alloc"));
	if ( cudaMemcpy(fixture->w1_payload_dev,fixture->w1_payload_host,w1_bytes,cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(fixture->w2_payload_dev,fixture->w2_payload_host,w2_bytes,cudaMemcpyHostToDevice) != cudaSuccess )
		return(SparkLingValFail("fixture","expert_upload"));
	if ( w1_scales != 0u )
	{
		if ( cudaMalloc((void **)&fixture->w1_scales_dev,w1_scales) != cudaSuccess ||
			cudaMemcpy(fixture->w1_scales_dev,fixture->w1_scales_host,w1_scales,cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkLingValFail("fixture","w1_scale_upload"));
	}
	if ( w2_scales != 0u )
	{
		if ( cudaMalloc((void **)&fixture->w2_scales_dev,w2_scales) != cudaSuccess ||
			cudaMemcpy(fixture->w2_scales_dev,fixture->w2_scales_host,w2_scales,cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkLingValFail("fixture","w2_scale_upload"));
	}
	return(0);
}

static void SparkLingValBindWeights(SparkLingValFixture *fixture,uint32_t local)
{
	SparkLingLayerWeights *weights = &fixture->layer_weights[local];
	memset(weights,0,sizeof(*weights));
	weights->attn_norm_bf16 = fixture->dense_attn_norm.device;
	weights->post_attn_norm_bf16 = fixture->dense_post_norm.device;
	weights->dense_gate_up_bf16 = fixture->dense_gate_up.device;
	weights->dense_down_bf16 = fixture->dense_down.device;
	weights->kda_qkv_beta_bf16 = fixture->kda_qkv_beta.device;
	weights->kda_decay_proj_bf16 = fixture->kda_decay_proj.device;
	weights->kda_gate_proj_bf16 = fixture->kda_gate_proj.device;
	weights->kda_q_conv_bf16 = fixture->kda_conv_q.device;
	weights->kda_k_conv_bf16 = fixture->kda_conv_k.device;
	weights->kda_v_conv_bf16 = fixture->kda_conv_v.device;
	weights->kda_decay_bias_f32 = fixture->kda_dt_bias_dev;
	weights->kda_head_log_scale_f32 = fixture->kda_a_log_dev;
	weights->kda_out_norm_bf16 = fixture->kda_out_norm_dev;
	weights->kda_out_bf16 = fixture->kda_out.device;
	weights->q_bf16 = fixture->mla_q_proj.device;
	weights->kv_a_bf16 = fixture->mla_kv_a.device;
	weights->kv_a_norm_bf16 = fixture->mla_kv_a_norm.device;
	weights->kv_b_key_transposed_bf16 = fixture->mla_kv_b_key.device;
	weights->kv_b_value_bf16 = fixture->mla_kv_b_value.device;
	weights->attn_gate_bf16 = fixture->mla_attn_gate.device;
	weights->attn_output_bf16 = fixture->mla_o_proj.device;
	weights->router_bf16 = fixture->router.device;
	weights->router_correction_f32 = fixture->router_correction_dev;
	weights->expert_up_gate_payload = fixture->w1_payload_dev;
	weights->expert_up_gate_scale = fixture->w1_scales_dev;
	weights->expert_down_payload = fixture->w2_payload_dev;
	weights->expert_down_scale = fixture->w2_scales_dev;
	weights->shared_gate_up_bf16 = fixture->shared_gate_up.device;
	weights->shared_down_bf16 = fixture->shared_down.device;
}

static void SparkLingValBindSlot(SparkLingValFixture *fixture)
{
	SparkLingExecutionSlot *slot = &fixture->slot;
	memset(slot,0,sizeof(*slot));
	slot->hidden_bf16 = fixture->hidden_dev;
	slot->normed_bf16 = fixture->normed_dev;
	slot->q_bf16 = fixture->q_dev;
	slot->query_latent_bf16 = fixture->query_latent_dev;
	slot->query_rope_bf16 = fixture->query_rope_dev;
	slot->attn_gate_bf16 = fixture->attn_gate_dev;
	slot->kv_slot_bf16 = fixture->kv_slot_dev;
	slot->attention_latent_bf16 = fixture->attention_latent_dev;
	slot->attention_value_bf16 = fixture->attention_value_dev;
	slot->attention_out_bf16 = fixture->attention_out_dev;
	slot->gate_up_bf16 = fixture->gate_up_dev;
	slot->intermediate_bf16 = fixture->intermediate_dev;
	slot->expert_out_bf16 = fixture->expert_out_dev;
	slot->shared_out_bf16 = fixture->shared_out_dev;
	slot->fused_qkvb_bf16 = fixture->fused_qkvb_dev;
	slot->kda_beta_logit = fixture->kda_beta_logit_dev;
	slot->kda_gate_bf16 = fixture->kda_gate_dev;
	slot->kda_decay_logit_bf16 = fixture->kda_decay_logit_dev;
	slot->kda_retention = fixture->kda_retention_dev;
	slot->kda_write_gate = fixture->kda_write_gate_dev;
	slot->router_logits_f32 = fixture->router_logits_dev;
	slot->route_expert = fixture->route_expert_dev;
	slot->route_weight = fixture->route_weight_dev;
	slot->route_source_token = fixture->route_source_token_dev;
	slot->route_packed_row = fixture->route_packed_row_dev;
	slot->group_row_offset = fixture->group_row_offset_dev;
	slot->group_tile_prefix_w1 = fixture->group_tile_prefix_w1_dev;
	slot->group_tile_prefix_w2 = fixture->group_tile_prefix_w2_dev;
	slot->token_ids = fixture->token_ids_dev;
	slot->resident_slots = fixture->resident_slots_dev;
	slot->positions = fixture->positions_dev;
	slot->context_lengths = fixture->context_lengths_dev;
	slot->dense_row_offset = fixture->dense_row_offset_dev;
	slot->dense_tile_prefix = fixture->dense_tile_prefix_dev;
	slot->output_token = fixture->token_ids_dev;
	slot->output_score = (float *)fixture->head_maxloc_dev;
	slot->head_candidate_score = (float *)fixture->head_maxloc_dev;
	slot->head_candidate_token = fixture->token_ids_dev;
	slot->head_maxloc_u64 = fixture->head_maxloc_dev;
	slot->kv_access_error = fixture->kv_access_error_dev;
}

static int SparkLingValFixtureBuild(SparkLingValFixture *fixture)
{
	memset(fixture,0,sizeof(*fixture));
	SparkLingValRandomState = 0x5eed1234u;
	if (SparkLingValAllocMatrix(&fixture->dense_attn_norm,1u,SPARK_LING_VAL_HIDDEN,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->dense_post_norm,1u,SPARK_LING_VAL_HIDDEN,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->dense_gate_up,SPARK_LING_VAL_DENSE_GATE_UP_ROWS,SPARK_LING_VAL_HIDDEN,0,0.004f) != 0 ||
		SparkLingValAllocMatrix(&fixture->dense_down,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_DENSE_INTER,0,0.004f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_qkv_beta,SPARK_LING_VAL_KDA_FUSED,SPARK_LING_VAL_HIDDEN,0,0.004f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_conv_q,SPARK_LING_VAL_KDA_QK,SPARK_LING_VAL_KDA_CONV,0,0.05f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_conv_k,SPARK_LING_VAL_KDA_QK,SPARK_LING_VAL_KDA_CONV,0,0.05f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_conv_v,SPARK_LING_VAL_KDA_V,SPARK_LING_VAL_KDA_CONV,0,0.05f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_decay_proj,SPARK_LING_VAL_KDA_QK,SPARK_LING_VAL_HIDDEN,0,0.004f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_gate_proj,SPARK_LING_VAL_KDA_V,SPARK_LING_VAL_HIDDEN,0,0.004f) != 0 ||
		SparkLingValAllocMatrix(&fixture->kda_out,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_KDA_V,0,0.005f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_attn_norm,1u,SPARK_LING_VAL_HIDDEN,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_post_norm,1u,SPARK_LING_VAL_HIDDEN,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_q_proj,SPARK_LING_VAL_Q_ROWS,SPARK_LING_VAL_HIDDEN,0,0.008f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_kv_a,SPARK_LING_VAL_KV_ROW,SPARK_LING_VAL_HIDDEN,0,0.008f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_kv_a_norm,1u,SPARK_LING_VAL_LATENT,1,0.0f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_kv_b_key,
			SPARK_LING_VAL_HEADS * SPARK_LING_VAL_LATENT,SPARK_LING_VAL_NOPE,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_kv_b_value,
			SPARK_LING_VAL_HEADS * SPARK_LING_VAL_VALUE,SPARK_LING_VAL_LATENT,0,0.01f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_attn_gate,SPARK_LING_VAL_HEADS,SPARK_LING_VAL_HIDDEN,0,0.008f) != 0 ||
		SparkLingValAllocMatrix(&fixture->mla_o_proj,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_ATTN_COLS,0,0.004f) != 0 ||
		SparkLingValAllocMatrix(&fixture->router,SPARK_LING_VAL_EXPERTS,SPARK_LING_VAL_HIDDEN,0,0.002f) != 0 ||
		SparkLingValAllocMatrix(&fixture->shared_gate_up,SPARK_LING_VAL_W1_ROWS,SPARK_LING_VAL_HIDDEN,0,0.005f) != 0 ||
		SparkLingValAllocMatrix(&fixture->shared_down,SPARK_LING_VAL_HIDDEN,SPARK_LING_VAL_EXPERT_INTER,0,0.005f) != 0)
		return(1);
	{
		uint32_t e;
		for (e = 0u; e < SPARK_LING_VAL_EXPERTS; e++)
			fixture->router_correction_host[e] = ((float)(int32_t)(e % 11u) - 5.0f) * 0.1f;
		fixture->router_correction_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->router_correction_host));
		if ( fixture->router_correction_dev == 0 ||
			cudaMemcpy(fixture->router_correction_dev,fixture->router_correction_host,
				sizeof(fixture->router_correction_host),cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkLingValFail("fixture","router_correction"));
	}
	if ( SparkLingValSynthExperts(fixture) != 0 )
		return(1);
	{
		uint32_t i;
		for (i = 0u; i < SPARK_LING_VAL_KDA_KEY; i++)
			fixture->kda_out_norm_host[i] = 1.0f;
		for (i = 0u; i < SPARK_LING_VAL_KDA_QK; i++)
			fixture->kda_dt_bias_host[i] = -0.25f;
		for (i = 0u; i < SPARK_LING_VAL_KDA_HEADS; i++)
			fixture->kda_a_log_host[i] = 0.1f;
		fixture->kda_out_norm_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->kda_out_norm_host));
		fixture->kda_dt_bias_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->kda_dt_bias_host));
		fixture->kda_a_log_dev = (float *)SparkLingValAllocZeroed(sizeof(fixture->kda_a_log_host));
		if ( fixture->kda_out_norm_dev == 0 || fixture->kda_dt_bias_dev == 0 || fixture->kda_a_log_dev == 0 ||
			cudaMemcpy(fixture->kda_out_norm_dev,fixture->kda_out_norm_host,sizeof(fixture->kda_out_norm_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->kda_dt_bias_dev,fixture->kda_dt_bias_host,sizeof(fixture->kda_dt_bias_host),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->kda_a_log_dev,fixture->kda_a_log_host,sizeof(fixture->kda_a_log_host),cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkLingValFail("fixture","kda_f32_upload"));
	}
	{
		uint64_t slot_bytes = SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER;
		uint64_t window_slot = (uint64_t)SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV * 2u;
		fixture->kda_state_pool_dev = (uint8_t *)SparkLingValAllocZeroed(2u * slot_bytes * SPARK_LING_VAL_SEQUENCES);
		fixture->kda_window_pool_dev = (uint8_t *)SparkLingValAllocZeroed(2u * window_slot * 3u * SPARK_LING_VAL_SEQUENCES);
		fixture->kda_state_index_dev = (uint32_t *)SparkLingValAllocZeroed(4u * SPARK_LING_VAL_SEQUENCES);
		fixture->kv_cache_dev = (uint8_t *)SparkLingValAllocZeroed(
			(uint64_t)SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS * SPARK_LING_VAL_KV_ROW * 2u);
		if ( fixture->kda_state_pool_dev == 0 || fixture->kda_window_pool_dev == 0 ||
			fixture->kda_state_index_dev == 0 || fixture->kv_cache_dev == 0 )
			return(SparkLingValFail("fixture","pools"));
	}
	{
		uint64_t rows_bytes = (uint64_t)SPARK_LING_VAL_ROWS;
		fixture->normed_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HIDDEN * 2u);
		fixture->hidden_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HIDDEN * 2u);
		fixture->boundary_in_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HIDDEN * 2u);
		fixture->boundary_out_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HIDDEN * 2u);
		fixture->q_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_Q_ROWS * 2u);
		fixture->query_latent_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HEADS * SPARK_LING_VAL_LATENT * 2u);
		fixture->query_rope_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HEADS * SPARK_LING_VAL_ROPE * 2u);
		fixture->attn_gate_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HEADS * 2u);
		fixture->kv_slot_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_KV_ROW * 2u);
		fixture->attention_latent_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HEADS * SPARK_LING_VAL_LATENT * 2u);
		fixture->attention_value_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_ATTN_COLS * 2u);
		fixture->attention_out_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HIDDEN * 2u);
		fixture->gate_up_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_DENSE_GATE_UP_ROWS * 2u);
		fixture->intermediate_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_DENSE_INTER * 2u);
		fixture->expert_out_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_TOP_K * SPARK_LING_VAL_HIDDEN * 2u);
		fixture->shared_out_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_HIDDEN * 2u);
		fixture->fused_qkvb_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_KDA_FUSED * 2u);
		fixture->kda_beta_logit_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_KDA_HEADS * 2u);
		fixture->kda_gate_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_KDA_V * 2u);
		fixture->kda_decay_logit_dev = (uint16_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_KDA_QK * 2u);
		fixture->kda_retention_dev = (float *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_KDA_QK * 4u);
		fixture->kda_write_gate_dev = (float *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_KDA_HEADS * 4u);
		fixture->router_logits_dev = (float *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_EXPERTS * 4u);
		fixture->route_expert_dev = (uint32_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_TOP_K * 4u);
		fixture->route_weight_dev = (float *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_TOP_K * 4u);
		fixture->route_source_token_dev = (uint32_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_TOP_K * 4u);
		fixture->route_packed_row_dev = (uint32_t *)SparkLingValAllocZeroed(rows_bytes * SPARK_LING_VAL_TOP_K * 4u);
		fixture->group_row_offset_dev = (uint32_t *)SparkLingValAllocZeroed((SPARK_LING_VAL_EXPERTS + 1u) * 4u);
		fixture->group_tile_prefix_w1_dev = (uint32_t *)SparkLingValAllocZeroed((SPARK_LING_VAL_EXPERTS + 1u) * 4u);
		fixture->group_tile_prefix_w2_dev = (uint32_t *)SparkLingValAllocZeroed((SPARK_LING_VAL_EXPERTS + 1u) * 4u);
		fixture->token_ids_dev = (uint32_t *)SparkLingValAllocZeroed(rows_bytes * 4u);
		fixture->resident_slots_dev = (uint32_t *)SparkLingValAllocZeroed(rows_bytes * 4u);
		fixture->positions_dev = (uint32_t *)SparkLingValAllocZeroed(rows_bytes * 4u);
		fixture->context_lengths_dev = (uint32_t *)SparkLingValAllocZeroed(rows_bytes * 4u);
		fixture->dense_row_offset_dev = (uint32_t *)SparkLingValAllocZeroed(4u * 4u);
		fixture->dense_tile_prefix_dev = (uint32_t *)SparkLingValAllocZeroed(4u * 4u);
		fixture->page_table_dev = (uint32_t *)SparkLingValAllocZeroed(SPARK_LING_VAL_PAGES * 4u);
		fixture->run_begin_dev = (uint32_t *)SparkLingValAllocZeroed(4u * 4u);
		fixture->run_state_dev = (uint32_t *)SparkLingValAllocZeroed(4u * 4u);
		fixture->head_maxloc_dev = (uint64_t *)SparkLingValAllocZeroed(rows_bytes * 8u);
		fixture->kv_access_error_dev = (uint32_t *)SparkLingValAllocZeroed(6u * 4u);
		fixture->boundary_host = (uint16_t *)malloc(rows_bytes * SPARK_LING_VAL_HIDDEN * 2u);
		if ( fixture->normed_dev == 0 || fixture->hidden_dev == 0 || fixture->boundary_in_dev == 0 || fixture->boundary_out_dev == 0 ||
			fixture->q_dev == 0 || fixture->query_latent_dev == 0 || fixture->query_rope_dev == 0 ||
			fixture->attn_gate_dev == 0 || fixture->kv_slot_dev == 0 || fixture->attention_latent_dev == 0 ||
			fixture->attention_value_dev == 0 || fixture->attention_out_dev == 0 || fixture->gate_up_dev == 0 ||
			fixture->intermediate_dev == 0 || fixture->expert_out_dev == 0 || fixture->shared_out_dev == 0 ||
			fixture->fused_qkvb_dev == 0 || fixture->kda_beta_logit_dev == 0 || fixture->kda_gate_dev == 0 ||
			fixture->kda_decay_logit_dev == 0 || fixture->kda_retention_dev == 0 || fixture->kda_write_gate_dev == 0 ||
			fixture->router_logits_dev == 0 || fixture->route_expert_dev == 0 || fixture->route_weight_dev == 0 ||
			fixture->route_source_token_dev == 0 || fixture->route_packed_row_dev == 0 ||
			fixture->group_row_offset_dev == 0 || fixture->group_tile_prefix_w1_dev == 0 ||
			fixture->group_tile_prefix_w2_dev == 0 || fixture->token_ids_dev == 0 ||
			fixture->resident_slots_dev == 0 || fixture->positions_dev == 0 || fixture->context_lengths_dev == 0 ||
			fixture->dense_row_offset_dev == 0 || fixture->dense_tile_prefix_dev == 0 ||
			fixture->page_table_dev == 0 || fixture->run_begin_dev == 0 || fixture->run_state_dev == 0 ||
			fixture->head_maxloc_dev == 0 || fixture->kv_access_error_dev == 0 || fixture->boundary_host == 0 )
			return(SparkLingValFail("fixture","scratch_alloc"));
	}
	{
		uint64_t i;
		uint32_t seq;
		for (i = 0u; i < (uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN; i++)
			fixture->boundary_host[i] = SparkLingValBf16(
				(((i % 23u)) - 11.0f) * 0.02f);
		if ( cudaMemcpy(fixture->boundary_in_dev,fixture->boundary_host,
			(uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u,cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkLingValFail("fixture","boundary_upload"));
		for (i = 0u; i < SPARK_LING_VAL_PAGES; i++)
			fixture->host_page_table[i] = (uint32_t)i;
		if ( cudaMemcpy(fixture->page_table_dev,fixture->host_page_table,
			sizeof(fixture->host_page_table),cudaMemcpyHostToDevice) != cudaSuccess )
			return(SparkLingValFail("fixture","page_table"));
		for (seq = 0u; seq < SPARK_LING_VAL_SEQUENCES; seq++)
		{
			uint32_t value = seq;
			if ( cudaMemcpy(fixture->kda_state_index_dev + seq,&value,sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess )
				return(SparkLingValFail("fixture","state_index"));
		}
		fixture->host_kv_ordinals[0] = UINT32_MAX;
		fixture->host_kv_ordinals[1] = UINT32_MAX;
		fixture->host_kda_ordinals[0] = UINT32_MAX;
		fixture->host_kda_ordinals[1] = UINT32_MAX;
	}
	SparkLingValBindSlot(fixture);
	if ( cudaStreamCreate(&fixture->stream) != cudaSuccess )
		return(SparkLingValFail("fixture","stream"));
	fixture->slot.stream = fixture->stream;
	fixture->multiprocessors = 16u;
	return(0);
}

static void SparkLingValBuildWave(SparkLingValFixture *fixture,
	const uint32_t *layers,const uint32_t *positions,uint32_t rows,uint32_t run_count)
{
	SparkLingCudaWave *wave = &fixture->wave;
	uint64_t state_stride = (uint64_t)SPARK_LING_VAL_SEQUENCES * SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER;
	uint64_t window_slot = (uint64_t)SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV * 2u;
	uint64_t window_stride = window_slot * 3u * SPARK_LING_VAL_SEQUENCES;
	uint32_t index;
	fixture->host_run_begin[0] = 0u;
	for (index = 0u; index < rows; index++)
	{
		fixture->host_slots[index] = 0u;
		fixture->host_positions[index] = positions[index];
	}
	if ( cudaMemcpy(fixture->resident_slots_dev,fixture->host_slots,rows * sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(fixture->positions_dev,fixture->host_positions,rows * sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess )
		SparkLingValFail("wave","metadata_upload");
	if ( run_count != 0u )
	{
		fixture->host_run_begin[run_count] = rows;
		if ( cudaMemcpy(fixture->run_begin_dev,fixture->host_run_begin,
			(run_count + 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(fixture->run_state_dev,fixture->host_run_state,
				run_count * sizeof(uint32_t),cudaMemcpyHostToDevice) != cudaSuccess )
			SparkLingValFail("wave","run_contract_upload");
	}
	fixture->kda_ordinals_used = 0u;
	fixture->kv_ordinals_used = 0u;
	for (index = 0u; index < 2u; index++)
	{
		uint32_t layer = layers[index];
		if ( SPARK_LING_MODEL_LAYER_IS_KDA(layer) )
		{
			fixture->host_kda_ordinals[index] = fixture->kda_ordinals_used++;
			fixture->host_kv_ordinals[index] = UINT32_MAX;
		}
		else
		{
			fixture->host_kda_ordinals[index] = UINT32_MAX;
			fixture->host_kv_ordinals[index] = fixture->kv_ordinals_used++;
		}
		SparkLingValBindWeights(fixture,index);
	}
	memset(wave,0,sizeof(*wave));
	wave->stage_index = 0u;
	wave->first_layer_index = layers[0];
	wave->layer_count = 2u;
	wave->tp_degree = 1u;
	wave->tp_rank = 0u;
	wave->row_count = rows;
	wave->maximum_context = SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS;
	wave->resident_sequence_capacity = SPARK_LING_VAL_SEQUENCES;
	wave->max_sequence_positions = SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS;
	wave->pages_per_sequence = SPARK_LING_VAL_PAGES;
	wave->owns_embedding = 0u;
	wave->owns_final_head = 0u;
	wave->boundary_row_offset = 0u;
	wave->hidden_input_bf16 = fixture->boundary_in_dev;
	wave->hidden_output_bf16 = fixture->boundary_out_dev;
	wave->layers = fixture->layer_weights;
	wave->slot = &fixture->slot;
	wave->kv_cache = fixture->kv_cache_dev;
	wave->kv_layer_stride_bytes = (uint64_t)SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS * SPARK_LING_VAL_KV_ROW * 2u;
	wave->kda_state_pools = fixture->kda_state_pool_dev;
	wave->kda_state_layer_stride_bytes = state_stride;
	wave->kda_q_window_pool = fixture->kda_window_pool_dev;
	wave->kda_k_window_pool = fixture->kda_window_pool_dev + window_slot * SPARK_LING_VAL_SEQUENCES;
	wave->kda_v_window_pool = fixture->kda_window_pool_dev + 2u * window_slot * SPARK_LING_VAL_SEQUENCES;
	wave->kda_window_layer_stride_bytes = window_stride;
	wave->kda_state_index = fixture->kda_state_index_dev;
	wave->page_table = fixture->page_table_dev;
	wave->multiprocessor_count = fixture->multiprocessors;
	wave->decode_split_context_threshold = SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS;
	wave->execution_row_capacity = SPARK_LING_VAL_ROWS;
	wave->kda_layer_count = 2u;
	wave->run_count = run_count;
	wave->sequence_row_begin = run_count != 0u ? fixture->run_begin_dev : 0;
	wave->run_state_index = run_count != 0u ? fixture->run_state_dev : 0;
	wave->host_sequence_row_begin = fixture->host_run_begin;
	wave->host_run_state_index = fixture->host_run_state;
	wave->host_resident_slots = fixture->host_slots;
	wave->host_positions = fixture->host_positions;
	wave->host_token_ids = fixture->token_ids_dev;
	wave->kv_ordinal_by_local_layer = fixture->host_kv_ordinals;
	wave->kda_ordinal_by_local_layer = fixture->host_kda_ordinals;
}

typedef struct SparkLingValWalk
{
	uint16_t kda_windows[2][3][SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV];
	float kda_state[2][(uint64_t)SPARK_LING_VAL_KDA_HEADS * SPARK_LING_VAL_KDA_KEY * SPARK_LING_VAL_KDA_KEY];
	uint16_t mla_cache[SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS][SPARK_LING_VAL_KV_ROW];
	uint32_t mla_context;
	uint32_t selected[SPARK_LING_VAL_TOP_K];
	float route_weights[SPARK_LING_VAL_TOP_K];
	float row_hidden[SPARK_LING_VAL_ROWS][SPARK_LING_VAL_HIDDEN];
	float row_sublayer[SPARK_LING_VAL_ROWS][SPARK_LING_VAL_HIDDEN];
	float boundary_rows[SPARK_LING_VAL_ROWS][SPARK_LING_VAL_HIDDEN];
} SparkLingValWalk;

static void SparkLingValRunAttentionOracle(SparkLingValFixture *fixture,
	SparkLingValWalk *walk,uint32_t layer,uint32_t local,const float *hidden,
	const float *residual,uint32_t row,float *sublayer_out,
	SparkLingValKdaDump *dump)
{
	if ( SPARK_LING_MODEL_LAYER_IS_KDA(layer) )
	{
		SparkLingValKdaWeights kda;
		kda.attn_norm = fixture->dense_attn_norm.host;
		kda.qkv_beta = fixture->kda_qkv_beta.host;
		kda.conv_q = fixture->kda_conv_q.host;
		kda.conv_k = fixture->kda_conv_k.host;
		kda.conv_v = fixture->kda_conv_v.host;
		kda.decay_proj = fixture->kda_decay_proj.host;
		kda.gate_proj = fixture->kda_gate_proj.host;
		kda.dt_bias = fixture->kda_dt_bias_host;
		kda.a_log = fixture->kda_a_log_host;
		kda.out_norm = fixture->kda_out_norm_host;
		kda.out_weight = fixture->kda_out.host;
		SparkLingValKdaAttention(&kda,hidden,residual,
			walk->kda_windows[local][0],walk->kda_windows[local][1],
			walk->kda_windows[local][2],walk->kda_state[local],sublayer_out,
			dump);
	}
	else
	{
		SparkLingValMlaWeights mla;
		SparkLingValMlaCache cache;
		mla.attn_norm = fixture->dense_attn_norm.host;
		mla.q_proj = fixture->mla_q_proj.host;
		mla.kv_a = fixture->mla_kv_a.host;
		mla.kv_a_norm = fixture->mla_kv_a_norm.host;
		mla.kv_b_key = fixture->mla_kv_b_key.host;
		mla.kv_b_value = fixture->mla_kv_b_value.host;
		mla.attn_gate = fixture->mla_attn_gate.host;
		mla.o_proj = fixture->mla_o_proj.host;
		cache.slots = walk->mla_cache[0];
		cache.context = walk->mla_context + row + 1u;
		SparkLingValMlaAttention(&mla,hidden,residual,
			walk->mla_context + row,&cache,sublayer_out);
	}
}

static void SparkLingValRunMlpOracle(SparkLingValFixture *fixture,
	SparkLingValWalk *walk,uint32_t layer,const float *hidden,float *sublayer_out)
{
	if ( layer < SPARK_LING_MODEL_FIRST_ROUTED_LAYER )
	{
		SparkLingValDenseMlp(fixture->dense_post_norm.host,
			fixture->dense_gate_up.host,fixture->dense_down.host,
			hidden,0,sublayer_out);
	}
	else
	{
		SparkLingValMoeWeights moe;
		moe.post_norm = fixture->dense_post_norm.host;
		moe.router = fixture->router.host;
		moe.correction = fixture->router_correction_host;
		moe.w1_payload = fixture->w1_payload_host;
		moe.w1_scales = fixture->w1_scales_host;
		moe.w2_payload = fixture->w2_payload_host;
		moe.w2_scales = fixture->w2_scales_host;
		moe.shared_gate_up = fixture->shared_gate_up.host;
		moe.shared_down = fixture->shared_down.host;
		moe.codec = SPARK_LING_VAL_CODEC;
		SparkLingValMoe(&moe,hidden,0,sublayer_out,
			walk->selected,walk->route_weights);
	}
}

static int SparkLingValDeviceClean(const SparkLingValFixture *fixture)
{
	uint32_t error[6];
	if ( cudaMemcpy(error,fixture->kv_access_error_dev,sizeof(error),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(0);
	return(error[0] == 0u);
}

static uint32_t SparkLingValHashBytes(const void *data,uint64_t bytes)
{
	const uint8_t *cursor = (const uint8_t *)data;
	uint32_t hash = 2166136261u;
	uint64_t index;
	for (index = 0u; index < bytes; index++)
	{
		hash ^= cursor[index];
		hash *= 16777619u;
	}
	return(hash);
}
static void SparkLingValCarryProbe(SparkLingValFixture *fixture,
	SparkLingValWalk *walk,const uint32_t *layers,uint32_t wave_index,uint32_t rows)
{
	static float device_state[SPARK_LING_VAL_KDA_HEADS * SPARK_LING_VAL_KDA_KEY * SPARK_LING_VAL_KDA_KEY];
	printf("probe pools q=%p k=%p v=%p stride=%llu slot=%llu\n",
		(const void *)fixture->wave.kda_q_window_pool,
		(const void *)fixture->wave.kda_k_window_pool,
		(const void *)fixture->wave.kda_v_window_pool,
		(unsigned long long)fixture->wave.kda_window_layer_stride_bytes,
		(unsigned long long)SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV * 2u);
	static uint16_t packed_window[SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV];
	static uint16_t packed_kv[4u * SPARK_LING_VAL_KV_ROW];
	static float device_kv[4u * SPARK_LING_VAL_KV_ROW];
	static uint16_t packed_boundary[SPARK_LING_VAL_HIDDEN];
	static float device_boundary[SPARK_LING_VAL_HIDDEN];
	uint64_t state_stride = (uint64_t)SPARK_LING_VAL_SEQUENCES * SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER;
	uint64_t window_slot = (uint64_t)SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV * 2u;
	uint64_t window_stride = window_slot * 3u * SPARK_LING_VAL_SEQUENCES;
	uint32_t local,index;
	SparkLingValMetrics metrics;
	printf("carry wave=%u rows=%u",wave_index,rows);
	for (local = 0u; local < 2u; local++)
	{
		if ( SPARK_LING_MODEL_LAYER_IS_KDA(layers[local]) )
		{
			const uint8_t *state_base = fixture->kda_state_pool_dev +
				(uint64_t)local * state_stride;
			const uint16_t *window_base = (const uint16_t *)
				((const uint8_t *)fixture->wave.kda_q_window_pool +
					(uint64_t)local * window_stride);
			if ( cudaMemcpy(device_state,state_base,sizeof(device_state),cudaMemcpyDeviceToHost) != cudaSuccess )
				return;
			SparkLingValMeasure(&metrics,device_state,walk->kda_state[local],
				SPARK_LING_VAL_KDA_HEADS * SPARK_LING_VAL_KDA_KEY * SPARK_LING_VAL_KDA_KEY);
			printf(" | l%u st rel %.3e cos %.7f",local,
				metrics.max_relative_l2,metrics.cosine);
			if ( cudaMemcpy(packed_window,window_base,sizeof(packed_window),cudaMemcpyDeviceToHost) != cudaSuccess )
				return;
			{
				double total = 0.0;
				for (index = 0u; index < SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV; index++)
				{
					double delta = (double)SparkLingValFromBf16(packed_window[index]) -
						(double)SparkLingValFromBf16(walk->kda_windows[local][0][index]);
					total += delta * delta;
				}
				printf(" win l2 %.3e",sqrt(total));
			}
		}
		else
		{
			uint32_t positions = walk->mla_context < 4u ? walk->mla_context : 4u;
			if ( cudaMemcpy(packed_kv,fixture->kv_cache_dev,
				(uint64_t)positions * SPARK_LING_VAL_KV_ROW * 2u,cudaMemcpyDeviceToHost) != cudaSuccess )
				return;
			for (index = 0u; index < (uint64_t)positions * SPARK_LING_VAL_KV_ROW; index++)
				device_kv[index] = (double)SparkLingValFromBf16(packed_kv[index]) -
					(double)SparkLingValFromBf16(walk->mla_cache[0][index]);
			{
				double total = 0.0;
				for (index = 0u; index < (uint64_t)positions * SPARK_LING_VAL_KV_ROW; index++)
					total += device_kv[index] * device_kv[index];
				printf(" | l%u kv[%u] l2 %.3e",local,positions,sqrt(total));
			}
		}
	}
	if ( cudaMemcpy(packed_boundary,fixture->boundary_out_dev,
		sizeof(packed_boundary),cudaMemcpyDeviceToHost) != cudaSuccess )
		return;
	for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		device_boundary[index] = (double)SparkLingValFromBf16(packed_boundary[index]) -
			(double)walk->boundary_rows[0][index];
	{
		double total = 0.0;
		for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
			total += device_boundary[index] * device_boundary[index];
		printf(" | bd l2 %.3e\n",sqrt(total));
	}
}

static int SparkLingValDriveWave(SparkLingValFixture *fixture,
	const uint32_t *layers,const uint32_t *positions,uint32_t rows,
	uint32_t run_count,SparkLingValWalk *walk,int probe,uint32_t wave_index)
{
	int32_t status;
	uint32_t local,row,index;
	SparkLingValBuildWave(fixture,layers,positions,rows,run_count);
	if ( probe != 0 )
		fprintf(stderr,"drive w%u: begin\n",wave_index);
	status = SparkLingLaunchCudaWaveBegin(&fixture->wave);
	if ( status != 0 )
	{
		fprintf(stderr,"ling_validation begin status=%d cuda=%s\n",
			status,cudaGetErrorString(cudaGetLastError()));
		return(SparkLingValFail("drive","begin"));
	}
	for (local = 0u; local < 2u; local++)
	{
		uint32_t layer = layers[local];
		if ( probe != 0 )
			fprintf(stderr,"drive w%u: l%u attention launching\n",wave_index,local);
		status = SparkLingLaunchCudaLayerAttention(&fixture->wave,local);
		if ( status != 0 )
		{
			fprintf(stderr,"ling_validation attention status=%d local=%u cuda=%s\n",
				status,local,cudaGetErrorString(cudaGetLastError()));
			return(SparkLingValFail("drive","attention"));
		}
		for (row = 0u; row < rows; row++)
		{
			const float *residual = local == 0u ? 0 : walk->row_sublayer[row];
			if ( probe != 0 && local == 0u && row == 0u )
			{
				static float actual[SPARK_LING_VAL_HIDDEN];
				static uint16_t device_sublayer[SPARK_LING_VAL_HIDDEN];
				SparkLingValMetrics metrics;
				static SparkLingValKdaDump oracle_dump;
				static uint16_t device_stage[SPARK_LING_VAL_KDA_FUSED];
				static uint16_t device_q[SPARK_LING_VAL_KDA_QK];
				static float device_ret[SPARK_LING_VAL_KDA_QK];
				static float device_beta[SPARK_LING_VAL_KDA_HEADS];
				static uint16_t device_window[3u][SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV];
				static uint16_t device_vconv[SPARK_LING_VAL_KDA_V];
				SparkLingValRunAttentionOracle(fixture,walk,layer,local,
					walk->row_hidden[row],residual,row,walk->row_sublayer[row],
					&oracle_dump);
				if ( cudaDeviceSynchronize() != cudaSuccess ||
					cudaMemcpy(device_sublayer,fixture->attention_out_dev,
						SPARK_LING_VAL_HIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
					return(SparkLingValFail("drive","attention_readback"));
				for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
					actual[index] = SparkLingValFromBf16(device_sublayer[index]);
				SparkLingValMeasure(&metrics,actual,walk->row_sublayer[row],SPARK_LING_VAL_HIDDEN);
				if ( SPARK_LING_MODEL_LAYER_IS_KDA(layer) )
				{
					int stage_fail = metrics.max_relative_l2 > 0.05 ||
						metrics.cosine < 0.995;
					if ( cudaDeviceSynchronize() == cudaSuccess &&
						cudaMemcpy(device_stage,fixture->fused_qkvb_dev,
							sizeof(device_stage),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(device_q,fixture->q_dev,
							sizeof(device_q),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(device_ret,fixture->kda_retention_dev,
							sizeof(device_ret),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(device_beta,fixture->kda_write_gate_dev,
							sizeof(device_beta),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(device_window[0],fixture->wave.kda_q_window_pool,
							sizeof(device_window[0]),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(device_window[1],fixture->wave.kda_k_window_pool,
							sizeof(device_window[1]),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(device_window[2],fixture->wave.kda_v_window_pool,
							sizeof(device_window[2]),cudaMemcpyDeviceToHost) == cudaSuccess &&
						cudaMemcpy(device_vconv,fixture->gate_up_dev,
							sizeof(device_vconv),cudaMemcpyDeviceToHost) == cudaSuccess )
					{
						uint32_t first_q = SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV;
						uint32_t first_w = SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV;
						for (index = 0u; index < SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV; index++)
							if ( device_window[0][index] != walk->kda_windows[0][0][index] )
							{
								if ( first_w == SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV )
									first_w = index;
							}
						for (index = 0u; index < SPARK_LING_VAL_KDA_QK; index++)
						{
							float d = SparkLingValFromBf16(device_q[index]) - oracle_dump.q_conv[index];
							if ( fabsf(d) > 0.002f * (fabsf(oracle_dump.q_conv[index]) + 0.001f) &&
								first_q == SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV )
								first_q = index;
						}
						printf(" win_first_diff %u (dev %04x or %04x) q_first_diff %u (dev %.5f or %.5f)",
							first_w,device_window[0][first_w],walk->kda_windows[0][0][first_w],
							first_q,SparkLingValFromBf16(device_q[first_q]),oracle_dump.q_conv[first_q]);
						static float device_stage_f[SPARK_LING_VAL_KDA_FUSED];
						static float device_q_f[SPARK_LING_VAL_KDA_QK];
						SparkLingValMetrics m;
						for (index = 0u; index < SPARK_LING_VAL_KDA_FUSED; index++)
							device_stage_f[index] = SparkLingValFromBf16(device_stage[index]);
						for (index = 0u; index < SPARK_LING_VAL_KDA_QK; index++)
							device_q_f[index] = SparkLingValFromBf16(device_q[index]);
						SparkLingValMeasure(&m,device_stage_f,oracle_dump.q_raw,SPARK_LING_VAL_KDA_QK);
						printf("stage w%u q_raw rel %.4f cos %.6f",wave_index,m.max_relative_l2,m.cosine);
						SparkLingValMeasure(&m,device_stage_f + 2u * SPARK_LING_VAL_KDA_QK,oracle_dump.v_raw,SPARK_LING_VAL_KDA_V);
						printf(" v_raw rel %.4f",m.max_relative_l2);
						SparkLingValMeasure(&m,device_q_f,oracle_dump.q_conv,SPARK_LING_VAL_KDA_QK);
						printf(" q_conv+l2 rel %.4f cos %.6f",m.max_relative_l2,m.cosine);
						SparkLingValMeasure(&m,device_ret,oracle_dump.retention,SPARK_LING_VAL_KDA_QK);
						printf(" ret rel %.4f",m.max_relative_l2);
						SparkLingValMeasure(&m,device_beta,oracle_dump.beta,SPARK_LING_VAL_KDA_HEADS);
						printf(" beta rel %.4f | sublayer",m.max_relative_l2);
					}
					if ( SparkLingValReport("kda attention sublayer",&metrics,0.05,0.995) != 0 && !stage_fail )
						return(1);
					if ( stage_fail )
						return(1);
				}
				else if ( SparkLingValReport("mla attention sublayer",&metrics,0.02,0.999) != 0 )
					return(1);
			}
			else
				SparkLingValRunAttentionOracle(fixture,walk,layer,local,
					walk->row_hidden[row],residual,row,walk->row_sublayer[row],
					0);
			if ( residual != 0 )
				for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
					walk->row_hidden[row][index] = SparkLingValFromBf16(
						SparkLingValBf16(walk->row_hidden[row][index] + residual[index]));
			for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
				walk->row_hidden[row][index] = SparkLingValFromBf16(
					SparkLingValBf16(walk->row_hidden[row][index] + walk->row_sublayer[row][index]));
		}
		if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
			return(SparkLingValFail("drive","attn_sync"));
		if ( probe != 0 )
			fprintf(stderr,"drive w%u: l%u attention synced\n",wave_index,local);
		status = SparkLingLaunchCudaLayerMlp(&fixture->wave,local);
		if ( status != 0 )
		{
			fprintf(stderr,"ling_validation mlp status=%d local=%u cuda=%s\n",
				status,local,cudaGetErrorString(cudaGetLastError()));
			return(SparkLingValFail("drive","mlp"));
		}
		if ( probe != 0 )
			fprintf(stderr,"drive w%u: l%u mlp launched\n",wave_index,local);
		for (row = 0u; row < rows; row++)
			SparkLingValRunMlpOracle(fixture,walk,layer,
				walk->row_hidden[row],walk->row_sublayer[row]);
		if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
			return(SparkLingValFail("drive","mlp_sync"));
		if ( probe != 0 )
			fprintf(stderr,"drive w%u: l%u mlp synced\n",wave_index,local);
	}
	if ( probe != 0 )
		fprintf(stderr,"drive w%u: head launching\n",wave_index);
	status = SparkLingLaunchCudaWaveHead(&fixture->wave);
	if ( status != 0 )
		return(SparkLingValFail("drive","head"));
	if ( cudaStreamSynchronize(fixture->stream) != cudaSuccess )
		return(SparkLingValFail("drive","head_sync"));
	if ( probe != 0 )
		fprintf(stderr,"drive w%u: head synced\n",wave_index);
	walk->mla_context += rows;
	for (row = 0u; row < rows; row++)
	{
		for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
		{
			float merged = walk->row_hidden[row][index] + walk->row_sublayer[row][index];
			walk->boundary_rows[row][index] =
				SparkLingValFromBf16(SparkLingValBf16(merged));
		}
		memcpy(walk->row_hidden[row],walk->boundary_rows[row],
			sizeof(walk->row_hidden[row]));
	}
	if ( probe != 0 )
		SparkLingValCarryProbe(fixture,walk,layers,wave_index,rows);
	return(0);
}

static void SparkLingValResetPools(SparkLingValFixture *fixture)
{
	uint64_t slot_bytes = SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER;
	uint64_t window_slot = (uint64_t)SPARK_LING_VAL_KDA_QK * SPARK_LING_VAL_KDA_CONV * 2u;
	cudaMemset(fixture->kda_state_pool_dev,0,2u * slot_bytes * SPARK_LING_VAL_SEQUENCES);
	cudaMemset(fixture->kda_window_pool_dev,0,2u * window_slot * 3u * SPARK_LING_VAL_SEQUENCES);
	cudaMemset(fixture->kv_cache_dev,0,(uint64_t)SPARK_LING_VAL_PAGES * SPARK_LING_VAL_PAGE_SLOTS * SPARK_LING_VAL_KV_ROW * 2u);
	cudaMemset(fixture->boundary_out_dev,0,(uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u);
	cudaMemset(fixture->attention_out_dev,0,(uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u);
}

typedef struct SparkLingValWavePlan
{
	uint32_t rows;
	uint32_t run_count;
} SparkLingValWavePlan;

static int SparkLingValRunTier(SparkLingValFixture *fixture,
	const uint32_t *layers,const SparkLingValWavePlan *plans,uint32_t plan_count,
	const char *label,int compare_routes)
{
	SparkLingValWalk walk;
	uint16_t *first = (uint16_t *)malloc((uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u);
	uint16_t *second = (uint16_t *)malloc((uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u);
	uint32_t pass,plan,index,row;
	int failures = 0;
	if ( first == 0 || second == 0 )
	{
		free(first);
		free(second);
		return(SparkLingValFail(label,"alloc"));
	}
	for (pass = 0u; pass < 2u; pass++)
	{
		uint32_t position = 0u;
		SparkLingValResetPools(fixture);
		memset(&walk,0,sizeof(walk));
		if ( cudaMemcpy(fixture->boundary_in_dev,fixture->boundary_host,
			(uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u,cudaMemcpyHostToDevice) != cudaSuccess )
		{
			free(first);
			free(second);
			return(SparkLingValFail(label,"boundary_reset"));
		}
		for (row = 0u; row < SPARK_LING_VAL_ROWS; row++)
			for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
				walk.row_hidden[row][index] = SparkLingValFromBf16(
					fixture->boundary_host[(uint64_t)row * SPARK_LING_VAL_HIDDEN + index]);
		for (plan = 0u; plan < plan_count; plan++)
		{
			uint32_t rows = plans[plan].rows;
			uint32_t positions[SPARK_LING_VAL_ROWS];
			for (index = 0u; index < rows; index++)
				positions[index] = position + index;
			if ( SparkLingValDriveWave(fixture,layers,positions,rows,
				plans[plan].run_count,&walk,pass == 0u,plan) != 0 )
			{
				free(first);
				free(second);
				return(1);
			}
			position += rows;
			if ( cudaMemcpy(fixture->boundary_in_dev,fixture->boundary_out_dev,
				(uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u,cudaMemcpyDeviceToDevice) != cudaSuccess )
			{
				free(first);
				free(second);
				return(SparkLingValFail(label,"boundary_swap"));
			}
		}
		if ( cudaMemcpy(pass == 0u ? first : second,fixture->boundary_out_dev,
			(uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u,cudaMemcpyDeviceToHost) != cudaSuccess )
		{
			free(first);
			free(second);
			return(SparkLingValFail(label,"readback"));
		}
		if ( pass == 0u && compare_routes != 0 )
		{
			uint32_t device_selected[SPARK_LING_VAL_TOP_K];
			float device_weights[SPARK_LING_VAL_TOP_K];
			if ( cudaMemcpy(device_selected,fixture->route_expert_dev,
				SPARK_LING_VAL_TOP_K * sizeof(uint32_t),cudaMemcpyDeviceToHost) == cudaSuccess &&
				cudaMemcpy(device_weights,fixture->route_weight_dev,
					SPARK_LING_VAL_TOP_K * sizeof(float),cudaMemcpyDeviceToHost) == cudaSuccess )
			{
				int set_match = 1;
				float weight_error = 0.0f;
				for (uint32_t slot = 0u; slot < SPARK_LING_VAL_TOP_K; slot++)
				{
					int found = 0;
					for (uint32_t other = 0u; other < SPARK_LING_VAL_TOP_K; other++)
						if ( device_selected[other] == walk.selected[slot] )
						{
							weight_error += fabsf(device_weights[other] - walk.route_weights[slot]);
							found = 1;
						}
					if ( found == 0 )
						set_match = 0;
				}
				printf("%s %-44s set_match %d weight_abs %.3e\n",
					set_match != 0 && weight_error < 1e-4f ? "PASS" : "FAIL",
					"router selection and weights",set_match,weight_error);
				failures += set_match != 0 && weight_error < 1e-4f ? 0 : 1;
			}
			{
				static float actual[SPARK_LING_VAL_HIDDEN];
				static float reference[SPARK_LING_VAL_HIDDEN];
				SparkLingValMetrics metrics,worst;
				memset(&worst,0,sizeof(worst));
				worst.cosine = 2.0;
				for (uint32_t row = 0u; row < SPARK_LING_VAL_ROWS; row++)
				{
					for (index = 0u; index < SPARK_LING_VAL_HIDDEN; index++)
					{
						actual[index] = SparkLingValFromBf16(
							first[(uint64_t)row * SPARK_LING_VAL_HIDDEN + index]);
						reference[index] = walk.boundary_rows[row][index];
					}
					SparkLingValMeasure(&metrics,actual,reference,SPARK_LING_VAL_HIDDEN);
					if ( metrics.max_relative_l2 > worst.max_relative_l2 )
						worst = metrics;
				}
				{
					char worst_label[160];
					snprintf(worst_label,sizeof(worst_label),"%s boundary stream",label);
					failures += SparkLingValReport(worst_label,&worst,0.02,0.999);
				}
			}
		}
	}
	{
		int identical = memcmp(first,second,(uint64_t)SPARK_LING_VAL_ROWS * SPARK_LING_VAL_HIDDEN * 2u) == 0;
		char determinism_label[160];
		snprintf(determinism_label,sizeof(determinism_label),"%s determinism",label);
		printf("%s %-44s %s\n",identical ? "PASS" : "FAIL",determinism_label,
			identical ? "bit-exact re-walk" : "DIVERGED");
		failures += identical ? 0 : 1;
	}
	if ( SparkLingValDeviceClean(fixture) == 0 )
		failures += SparkLingValFail(label,"kv_access_error");
	else
		printf("PASS %-44s\n",label);
	free(first);
	free(second);
	return(failures);
}

int main(int argc,char **argv)
{
	static SparkLingValFixture fixture;
	static const uint32_t kda_layers[2] = {0u,1u};
	static const uint32_t mixed_layers[2] = {5u,6u};
	SparkLingValWavePlan decode_plans[SPARK_LING_VAL_TOKENS];
	SparkLingValWavePlan prefill_plans[2];
	uint32_t step;
	int failures = 0;
	setvbuf(stdout,0,_IOLBF,0);
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	printf("ling validator: configuration %s codec %s\n",argv[1],LING_EXPERT_CODEC_NAME);
	if ( SparkLingValOracleSelftest() != 0 )
		return(1);
	if ( SparkLingValFixtureBuild(&fixture) != 0 )
		return(1);
	{
		uint32_t multiprocessors = 0u;
		if ( SparkLingConfigureCudaModule(&multiprocessors) == 0 )
			fixture.multiprocessors = multiprocessors;
	}
	for (step = 0u; step < SPARK_LING_VAL_TOKENS; step++)
	{
		decode_plans[step].rows = 1u;
		decode_plans[step].run_count = 0u;
	}
	prefill_plans[0].rows = SPARK_LING_VAL_PREFILL_ROWS;
	prefill_plans[0].run_count = 1u;
	prefill_plans[1].rows = 1u;
	prefill_plans[1].run_count = 0u;
	failures += SparkLingValRunTier(&fixture,kda_layers,decode_plans,
		SPARK_LING_VAL_TOKENS,"tier1 kda+dense decode",0);
	failures += SparkLingValRunTier(&fixture,mixed_layers,decode_plans,
		SPARK_LING_VAL_TOKENS,"tier2a mla+moe decode",1);
	failures += SparkLingValRunTier(&fixture,kda_layers,prefill_plans,2u,
		"tier3 prefill+cached decode",0);
	printf("ling validator: %s (%d failures)\n",
		failures == 0 ? "PASS" : "FAIL",failures);
	cudaDeviceSynchronize();
	return(failures == 0 ? 0 : 1);
}
#else
int main(void)
{
	return(SparkLingValOracleSelftest() == 0 ? 0 : 1);
}
#endif
