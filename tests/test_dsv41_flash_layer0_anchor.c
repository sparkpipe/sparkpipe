#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HIDDEN 5120u
#define HEADS_LOCAL 8u
#define HEAD_DIM 512u
#define ROPE_PAIRS 32u
#define Q_LORA 1280u
#define KV_LATENT 512u
#define OA_ROWS 1024u
#define OA_FULL_COLS 4096u
#define OB_COLS 1024u
#define SINKS_LOCAL 8u
#define N_EXPERTS 384u
#define TOPK 6u
#define MOE_INTER 2304u
#define HC 4u
#define HC_ROWS 24u
#define HC_FLAT (HC * HIDDEN)
#define SINKHORN_ITERS 20u
#define NORM_EPS 1e-20f
#define HC_EPS 1e-6f
#define WINDOW 128u
#define ROUTE_SCALE 1.5f
#define SWIGLU_LIMIT 10.0f
#define TOTAL_POSITIONS 131u
#define LOCAL_EXPERTS 48u
#define COMP_RATIO 2u
#define IDX_HEADS_LOCAL 4u
#define IDX_DIM 128u
#define IDX_TOPK 512u
#define MAX_COMP_POS 67u
#define CSA2_STREAM_HORIZON 64u
#define LADDER_LAYERS 3u
#define ROPE_TABLE_PURE 0u
#define ROPE_TABLE_YARN 1u
#define RANK 0u
#define TOKEN_SLOTS 16u

#define PACK_MAGIC UINT32_C(0x31413444)
#define ENTRY_BYTES 64u
#define DIRECTORY_OFFSET 512u
#define GLOBAL_LAYER UINT32_MAX

#define K_EMBED 0u
#define K_FNORM 1u
#define K_ATTN_NORM 3u
#define K_FFN_NORM 4u
#define K_QA 5u
#define K_QB 6u
#define K_KVA 7u
#define K_QNORM 8u
#define K_KVNORM 9u
#define K_SINK 10u
#define K_OA 11u
#define K_OB 12u
#define K_IQB 13u
#define K_IWK 14u
#define K_IWP 15u
#define K_IKN 16u
#define K_CWKV 17u
#define K_CWGATE 18u
#define K_CNORM 19u
#define K_HCAF 20u
#define K_HCAB 21u
#define K_HCAS 22u
#define K_HCFF 23u
#define K_HCFB 24u
#define K_HCFS 25u
#define K_ROUTER 26u
#define K_RBIAS 27u
#define K_W1 29u
#define K_W2 30u
#define K_W3 31u
#define K_SW1 32u
#define K_SW2 33u
#define K_SW3 34u

#define FAIL_EXIT 2

typedef struct PackEntry
{
	uint32_t kind, layer, payload_type, codec, scale_encoding, groups, rows, cols;
	uint64_t payload_offset, payload_bytes, scale_offset, scale_bytes;
} PackEntry;

typedef struct Piece
{
	char name[64];
	uint32_t offset, count;
	float tol_rel, tol_abs, tol_scale;
	float max_abs;
	uint32_t seen, bad;
} Piece;

typedef struct Expect
{
	uint8_t *data;
	Piece *pieces;
	uint32_t count;
	uint32_t piece_failures;
	uint32_t missing;
} Expect;

typedef struct Pack
{
	uint8_t *map;
	uint64_t map_bytes;
	PackEntry *entries;
	uint32_t entry_count;
} Pack;

typedef struct BlockWeights
{
	uint32_t layer, is_csa2, rope_table;
	const uint16_t *embed, *attn_norm, *ffn_norm, *q_norm, *kv_norm, *router;
	const uint16_t *i_wk, *i_wp, *i_kn, *c_wkv, *c_wgate, *c_norm;
	const float *sink, *hc_a_fn, *hc_a_base, *hc_a_scale;
	const float *hc_f_fn, *hc_f_base, *hc_f_scale, *rbias;
	const uint8_t *qa, *qa_s, *qb, *qb_s, *kva, *kva_s, *oa, *oa_s, *ob, *ob_s;
	const uint8_t *i_qb, *i_qb_s;
	const uint8_t *sw1, *sw1_s, *sw2, *sw2_s, *sw3, *sw3_s;
	const uint8_t *w1, *w1_s, *w2, *w2_s, *w3, *w3_s;
} BlockWeights;

typedef struct BlockState
{
	uint16_t window[WINDOW][KV_LATENT];
	uint16_t comp_cache[MAX_COMP_POS][KV_LATENT];
	uint16_t k_cache[MAX_COMP_POS][IDX_DIM];
	float kv_state[COMP_RATIO][KV_LATENT];
	float score_state[COMP_RATIO][KV_LATENT];
	float pre_mix[HC];
	uint32_t window_len;
} BlockState;

static float fp8_lut[256];
static float fp4_lut[16];
static float fp4_ties[7] = { 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f };
static float fp4_tie_mag[7] = { 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 4.0f, 4.0f };
static float g_cos[2u][TOTAL_POSITIONS + 1u][ROPE_PAIRS];
static float g_sin[2u][TOTAL_POSITIONS + 1u][ROPE_PAIRS];
static Pack g_pack;
static int32_t g_tokens[TOKEN_SLOTS];
static uint32_t g_token_count;

static float Bf16ToF32(uint16_t v)
{
	union { uint32_t u; float f; } c;
	c.u = (uint32_t)v << 16;
	return(c.f);
}

static uint16_t F32ToBf16(float v)
{
	union { uint32_t u; float f; } c;
	c.f = v;
	return (uint16_t)((c.u + 0x7FFFu + ((c.u >> 16) & 1u)) >> 16);
}

static void InitLuts(void)
{
	static const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
	uint32_t i;
	float s, e, m;
	for (i = 0u; i < 256u; i++)
	{
		s = (i & 0x80u) != 0u ? -1.0f : 1.0f;
		e = (float)((i >> 3) & 0xF);
		m = (float)(i & 0x7);
		if ( e == 0.0f )
			fp8_lut[i] = s * (m * 0.125f) * 0.015625f;
		else if ( e == 15.0f && m == 7.0f )
			fp8_lut[i] = 0.0f;
		else
			fp8_lut[i] = s * (1.0f + m * 0.125f) * ldexpf(1.0f, (int)e - 7);
	}
	for (i = 0u; i < 16u; i++)
		fp4_lut[i] = (i & 0x8u) != 0u ? -mag[i & 0x7u] : mag[i & 0x7u];
}

static float E8M0ToF32(uint8_t b)
{
	return ldexpf(1.0f, (int)b - 127);
}

static uint32_t RneShift32(uint32_t v, uint32_t s)
{
	uint32_t q, rem, half;
	if ( s == 0u )
		return(v);
	q = v >> s;
	rem = v & ((1u << s) - 1u);
	half = 1u << (s - 1u);
	if ( rem > half || (rem == half && (q & 1u) != 0u) )
		q += 1u;
	return(q);
}

static uint8_t Fp8Code(float v)
{
	union { uint32_t u; float f; } c;
	uint32_t bits, sign;
	int32_t exp, m3;
	c.f = v < -448.0f ? -448.0f : (v > 448.0f ? 448.0f : v);
	bits = c.u & 0x7FFFFFFFu;
	sign = (c.u >> 24) & 0x80u;
	if ( bits == 0u )
		return (uint8_t)sign;
	exp = (int32_t)(bits >> 23) - 127;
	if ( bits >= 0x3D000000u )
	{
		m3 = (int32_t)RneShift32(bits & 0x7FFFFFu, 20u);
		if ( m3 > 7 )
		{
			m3 = 0;
			exp += 1;
		}
		return (uint8_t)(sign | (uint32_t)((exp + 7) << 3) | (uint32_t)m3);
	}
	m3 = (int32_t)RneShift32(0x800000u | (bits & 0x7FFFFFu),
		(uint32_t)(14 - exp) > 31u ? 31u : (uint32_t)(14 - exp));
	return (uint8_t)(sign | (uint32_t)m3);
}

static float Fp4Nearest(float v)
{
	float a = fabsf(v), mag;
	uint32_t i = 0u, j = 0u;
	while ( i < 7u && fp4_ties[i] <= a )
		i++;
	mag = fp4_lut[i];
	while ( j < 7u && fp4_ties[j] < a )
		j++;
	if ( j < 7u && fp4_ties[j] == a )
		mag = fp4_tie_mag[j];
	return v < 0.0f ? -mag : mag;
}

static void ActQuant(const float *x, uint32_t n, float *vals, float *scales)
{
	union { uint32_t u; float f; } c;
	uint32_t b, j, blocks;
	float amax, t, q;
	blocks = n / 32u;
	for (b = 0u; b < blocks; b++)
	{
		amax = 1e-4f;
		for (j = 0u; j < 32u; j++)
		{
			float a = fabsf(x[b * 32u + j]);
			if ( a > amax )
				amax = a;
		}
		c.f = amax * (1.0f / 448.0f);
		t = ldexpf(1.0f, (int32_t)(c.u >> 23) - 127 + ((c.u & 0x7FFFFFu) != 0u ? 1 : 0));
		scales[b] = t;
		for (j = 0u; j < 32u; j++)
		{
			q = x[b * 32u + j] / t;
			q = q < -448.0f ? -448.0f : (q > 448.0f ? 448.0f : q);
			vals[b * 32u + j] = fp8_lut[Fp8Code(q)];
		}
	}
}

static void ActDequantRow(const uint16_t *x, uint32_t n, uint16_t *out)
{
	float vals[HIDDEN], scales[HIDDEN / 32u];
	uint32_t i, b;
	float xf[HIDDEN];
	for (i = 0u; i < n; i++)
		xf[i] = Bf16ToF32(x[i]);
	ActQuant(xf, n, vals, scales);
	for (i = 0u; i < n / 32u; i++)
		for (b = 0u; b < 32u; b++)
			out[i * 32u + b] = F32ToBf16(vals[i * 32u + b] * scales[i]);
}

static void Fp4RoundTripE8M0(uint16_t *row, uint32_t n)
{
	union { uint32_t u; float f; } c;
	uint32_t b, j;
	float amax, scale, v;
	int32_t lc;
	for (b = 0u; b < n / 32u; b++)
	{
		amax = 6.0f * ldexpf(1.0f, -126);
		for (j = 0u; j < 32u; j++)
		{
			float a = fabsf(Bf16ToF32(row[b * 32u + j]));
			if ( a > amax )
				amax = a;
		}
		c.f = amax * (1.0f / 6.0f);
		lc = (int32_t)(c.u >> 23) - 127 + ((c.u & 0x7FFFFFu) != 0u ? 1 : 0);
		scale = ldexpf(1.0f, lc);
		for (j = 0u; j < 32u; j++)
		{
			v = Bf16ToF32(row[b * 32u + j]) / scale;
			v = v < -6.0f ? -6.0f : (v > 6.0f ? 6.0f : v);
			row[b * 32u + j] = F32ToBf16(Fp4Nearest(v) * scale);
		}
	}
}

static void Fp4RoundTripE4M3(uint16_t *row, uint32_t n)
{
	uint32_t b, j;
	float amax, scale, v;
	for (b = 0u; b < n / 16u; b++)
	{
		amax = 6.0f * ldexpf(1.0f, -9);
		for (j = 0u; j < 16u; j++)
		{
			float a = fabsf(Bf16ToF32(row[b * 16u + j]));
			if ( a > amax )
				amax = a;
		}
		scale = fp8_lut[Fp8Code(amax * (1.0f / 6.0f))];
		for (j = 0u; j < 16u; j++)
		{
			v = Bf16ToF32(row[b * 16u + j]) / scale;
			v = v < -6.0f ? -6.0f : (v > 6.0f ? 6.0f : v);
			row[b * 16u + j] = F32ToBf16(Fp4Nearest(v) * scale);
		}
	}
}

static void GemvFp8(const uint16_t *x, const uint8_t *w, const uint8_t *ws,
	uint32_t rows, uint32_t k, uint16_t *out)
{
	float xf[HIDDEN], aq[HIDDEN], sa[HIDDEN / 32u], acc;
	uint32_t r, b, j;
	for (r = 0u; r < k; r++)
		xf[r] = Bf16ToF32(x[r]);
	ActQuant(xf, k, aq, sa);
	for (r = 0u; r < rows; r++)
	{
		acc = 0.0f;
		for (b = 0u; b < k / 32u; b++)
		{
			float part = 0.0f;
			float sb = E8M0ToF32(ws[((uint64_t)r / 32u) * (k / 32u) + b]);
			for (j = 0u; j < 32u; j++)
				part += aq[b * 32u + j] * fp8_lut[w[(uint64_t)r * k + b * 32u + j]];
			acc += part * sa[b] * sb;
		}
		out[r] = F32ToBf16(acc);
	}
}

static void GemvFp4(const uint16_t *x, const uint8_t *w, const uint8_t *ws,
	uint32_t rows, uint32_t k, uint16_t *out)
{
	float xf[HIDDEN], aq[HIDDEN], sa[HIDDEN / 32u], acc;
	uint32_t r, b, j;
	for (r = 0u; r < k; r++)
		xf[r] = Bf16ToF32(x[r]);
	ActQuant(xf, k, aq, sa);
	for (r = 0u; r < rows; r++)
	{
		acc = 0.0f;
		for (b = 0u; b < k / 32u; b++)
		{
			float part = 0.0f;
			for (j = 0u; j < 32u; j++)
			{
				uint8_t byte = w[(uint64_t)r * (k / 2u) + (b * 32u + j) / 2u];
				uint8_t nib = ((b * 32u + j) & 1u) != 0u ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xFu);
				part += aq[b * 32u + j] * fp4_lut[nib];
			}
			acc += part * sa[b] * E8M0ToF32(ws[(uint64_t)r * (k / 32u) + b]);
		}
		out[r] = F32ToBf16(acc);
	}
}

static void Bf16Gemv(const uint16_t *x, const uint16_t *w, uint32_t rows,
	uint32_t k, uint16_t *out)
{
	uint32_t r, d;
	for (r = 0u; r < rows; r++)
	{
		float acc = 0.0f;
		for (d = 0u; d < k; d++)
			acc += Bf16ToF32(w[(uint64_t)r * k + d]) * Bf16ToF32(x[d]);
		out[r] = F32ToBf16(acc);
	}
}

static void RmsNorm(const uint16_t *x, const uint16_t *w, uint32_t n, uint16_t *out)
{
	float total = 0.0f, rstd;
	uint32_t i;
	for (i = 0u; i < n; i++)
	{
		float v = Bf16ToF32(x[i]);
		total += v * v;
	}
	rstd = 1.0f / sqrtf(total / (float)n + NORM_EPS);
	for (i = 0u; i < n; i++)
		out[i] = F32ToBf16(Bf16ToF32(w[i]) * Bf16ToF32(x[i]) * rstd);
}

static double YarnCorrected(int rotations)
{
	return 32.0 * log(65536.0 / ((double)rotations * 2.0 * 3.14159265358979323846)) /
		log(160000.0);
}

static void RopeInit(void)
{
	int low = (int)floor(YarnCorrected(32));
	int high = (int)ceil(YarnCorrected(1));
	uint32_t p, d;
	float span;
	if ( low < 0 )
		low = 0;
	if ( high > 63 )
		high = 63;
	span = (float)(high - low > 0 ? high - low : 1);
	for (p = 0u; p <= TOTAL_POSITIONS; p++)
	{
		for (d = 0u; d < ROPE_PAIRS; d++)
		{
			float inv_pure = powf(10000.0f, -2.0f * (float)d / 64.0f);
			float inv_yarn = powf(160000.0f, -2.0f * (float)d / 64.0f);
			float ramp = ((float)d - (float)low) / span;
			float smooth;
			ramp = ramp < 0.0f ? 0.0f : (ramp > 1.0f ? 1.0f : ramp);
			smooth = 1.0f - ramp;
			inv_yarn = inv_yarn / 16.0f * (1.0f - smooth) + inv_yarn * smooth;
			g_cos[ROPE_TABLE_PURE][p][d] = cosf((float)p * inv_pure);
			g_sin[ROPE_TABLE_PURE][p][d] = sinf((float)p * inv_pure);
			g_cos[ROPE_TABLE_YARN][p][d] = cosf((float)p * inv_yarn);
			g_sin[ROPE_TABLE_YARN][p][d] = sinf((float)p * inv_yarn);
		}
	}
}

static void ApplyRopeTailN(uint16_t *vec, uint32_t n, uint32_t pos, uint32_t table,
	uint32_t inverse)
{
	float re, im, c, s;
	uint32_t d, base = n - 64u;
	for (d = 0u; d < ROPE_PAIRS; d++)
	{
		c = g_cos[table][pos][d];
		s = inverse != 0u ? -g_sin[table][pos][d] : g_sin[table][pos][d];
		re = Bf16ToF32(vec[base + 2u * d]);
		im = Bf16ToF32(vec[base + 2u * d + 1u]);
		vec[base + 2u * d] = F32ToBf16(re * c - im * s);
		vec[base + 2u * d + 1u] = F32ToBf16(re * s + im * c);
	}
}

static void HcMixes(const uint16_t *stream, const float *fn, const float *scale3,
	const float *base, float *mixes, float *pre, float *post, float *comb)
{
	float x[HC_FLAT], total = 0.0f, rstd, v, mx, sum;
	uint32_t m, i, r, c, it;
	for (i = 0u; i < HC_FLAT; i++)
	{
		x[i] = Bf16ToF32(stream[i]);
		total += x[i] * x[i];
	}
	rstd = 1.0f / sqrtf(total / (float)HC_FLAT + NORM_EPS);
	for (m = 0u; m < HC_ROWS; m++)
	{
		v = 0.0f;
		for (i = 0u; i < HC_FLAT; i++)
			v += fn[(uint64_t)m * HC_FLAT + i] * x[i];
		mixes[m] = v * rstd;
	}
	for (i = 0u; i < HC; i++)
		pre[i] = 1.0f / (1.0f + expf(-(mixes[i] * scale3[0] + base[i]))) + HC_EPS;
	for (i = 0u; i < HC; i++)
		post[i] = 2.0f / (1.0f + expf(-(mixes[HC + i] * scale3[1] + base[HC + i])));
	for (r = 0u; r < HC; r++)
		for (c = 0u; c < HC; c++)
			comb[r * HC + c] = mixes[2u * HC + r * HC + c] * scale3[2] +
				base[2u * HC + r * HC + c];
	for (r = 0u; r < HC; r++)
	{
		mx = comb[r * HC];
		for (c = 1u; c < HC; c++)
			if ( comb[r * HC + c] > mx )
				mx = comb[r * HC + c];
		sum = 0.0f;
		for (c = 0u; c < HC; c++)
		{
			comb[r * HC + c] = expf(comb[r * HC + c] - mx);
			sum += comb[r * HC + c];
		}
		for (c = 0u; c < HC; c++)
			comb[r * HC + c] = comb[r * HC + c] / sum + HC_EPS;
	}
	for (c = 0u; c < HC; c++)
	{
		sum = 0.0f;
		for (r = 0u; r < HC; r++)
			sum += comb[r * HC + c];
		for (r = 0u; r < HC; r++)
			comb[r * HC + c] = comb[r * HC + c] / (sum + HC_EPS);
	}
	for (it = 1u; it < SINKHORN_ITERS; it++)
	{
		for (r = 0u; r < HC; r++)
		{
			sum = 0.0f;
			for (c = 0u; c < HC; c++)
				sum += comb[r * HC + c];
			for (c = 0u; c < HC; c++)
				comb[r * HC + c] = comb[r * HC + c] / (sum + HC_EPS);
		}
		for (c = 0u; c < HC; c++)
		{
			sum = 0.0f;
			for (r = 0u; r < HC; r++)
				sum += comb[r * HC + c];
			for (r = 0u; r < HC; r++)
				comb[r * HC + c] = comb[r * HC + c] / (sum + HC_EPS);
		}
	}
}

static void HcPre(const uint16_t *stream, const float *pre, uint16_t *out)
{
	float acc;
	uint32_t s, d;
	for (d = 0u; d < HIDDEN; d++)
	{
		acc = 0.0f;
		for (s = 0u; s < HC; s++)
			acc += pre[s] * Bf16ToF32(stream[s * HIDDEN + d]);
		out[d] = F32ToBf16(acc);
	}
}

static void HcPost(const uint16_t *x, const uint16_t *residual, const float *post,
	const float *comb, uint16_t *out)
{
	float acc;
	uint32_t r, s, d;
	for (r = 0u; r < HC; r++)
	{
		for (d = 0u; d < HIDDEN; d++)
		{
			acc = post[r] * Bf16ToF32(x[d]);
			for (s = 0u; s < HC; s++)
				acc += comb[s * HC + r] * Bf16ToF32(residual[s * HIDDEN + d]);
			out[r * HIDDEN + d] = F32ToBf16(acc);
		}
	}
}

static float StableSigmoid(float x)
{
	x = x < -80.0f ? -80.0f : (x > 80.0f ? 80.0f : x);
	return 1.0f / (1.0f + expf(-x));
}

static float SoftplusSqrt(float v)
{
	return sqrtf(log1pf(expf(v < -30.0f ? -30.0f : (v > 30.0f ? 30.0f : v))));
}

static void GateScores(const uint16_t *x, const uint16_t *router, float *scores)
{
	float v;
	uint32_t e, d;
	for (e = 0u; e < N_EXPERTS; e++)
	{
		v = 0.0f;
		for (d = 0u; d < HIDDEN; d++)
			v += Bf16ToF32(router[(uint64_t)e * HIDDEN + d]) * Bf16ToF32(x[d]);
		scores[e] = v;
	}
}

static void GateSelect(const float *scores, const float *bias, uint32_t *idx, float *weights)
{
	float top, sum = 0.0f;
	uint32_t e, slot, best;
	float work[N_EXPERTS];
	memcpy(work, scores, sizeof(work));
	for (slot = 0u; slot < TOPK; slot++)
	{
		best = N_EXPERTS;
		top = -3.4e38f;
		for (e = 0u; e < N_EXPERTS; e++)
		{
			float sp = SoftplusSqrt(work[e]) + bias[e];
			if ( work[e] != 3.4e38f && sp > top )
			{
				top = sp;
				best = e;
			}
		}
		idx[slot] = best;
		weights[slot] = SoftplusSqrt(work[best]);
		work[best] = 3.4e38f;
	}
	for (slot = 0u; slot < TOPK; slot++)
		sum += weights[slot];
	for (slot = 0u; slot < TOPK; slot++)
		weights[slot] = weights[slot] / (sum + 1e-20f) * ROUTE_SCALE;
}

static void SwiClamp(const float *gate, const float *up, uint32_t n, uint16_t *out, float weight)
{
	uint32_t i;
	float g, u;
	for (i = 0u; i < n; i++)
	{
		g = gate[i] < SWIGLU_LIMIT ? gate[i] : SWIGLU_LIMIT;
		u = up[i] < -SWIGLU_LIMIT ? -SWIGLU_LIMIT : (up[i] > SWIGLU_LIMIT ? SWIGLU_LIMIT : up[i]);
		out[i] = F32ToBf16(g * StableSigmoid(g) * u * weight);
	}
}

static void ExpertMlp(const uint16_t *x, const uint8_t *w1, const uint8_t *w1s,
	const uint8_t *w3, const uint8_t *w3s, const uint8_t *w2, const uint8_t *w2s,
	float weight, uint16_t *out)
{
	float gate[MOE_INTER], up[MOE_INTER];
	uint16_t gate_b[MOE_INTER], up_b[MOE_INTER], act_b[MOE_INTER];
	uint32_t i;
	GemvFp4(x, w1, w1s, MOE_INTER, HIDDEN, gate_b);
	GemvFp4(x, w3, w3s, MOE_INTER, HIDDEN, up_b);
	for (i = 0u; i < MOE_INTER; i++)
	{
		gate[i] = Bf16ToF32(gate_b[i]);
		up[i] = Bf16ToF32(up_b[i]);
	}
	SwiClamp(gate, up, MOE_INTER, act_b, weight);
	GemvFp4(act_b, w2, w2s, HIDDEN, MOE_INTER, out);
}

static void SharedMlp(const uint16_t *x, const uint8_t *w1, const uint8_t *w1s,
	const uint8_t *w3, const uint8_t *w3s, const uint8_t *w2, const uint8_t *w2s,
	uint16_t *out)
{
	float gate[MOE_INTER], up[MOE_INTER];
	uint16_t gate_b[MOE_INTER], up_b[MOE_INTER], act_b[MOE_INTER];
	uint32_t i;
	GemvFp8(x, w1, w1s, MOE_INTER, HIDDEN, gate_b);
	GemvFp8(x, w3, w3s, MOE_INTER, HIDDEN, up_b);
	for (i = 0u; i < MOE_INTER; i++)
	{
		gate[i] = Bf16ToF32(gate_b[i]);
		up[i] = Bf16ToF32(up_b[i]);
	}
	SwiClamp(gate, up, MOE_INTER, act_b, 1.0f);
	GemvFp8(act_b, w2, w2s, HIDDEN, MOE_INTER, out);
}

static const uint16_t *AttnRow(const BlockState *st, uint32_t idx)
{
	return idx < WINDOW ? st->window[idx] : st->comp_cache[idx - WINDOW];
}

static void SparseAttn(const uint16_t *q, const BlockState *st, const int32_t *valid,
	uint32_t count, const float *sinks, uint16_t *out)
{
	float scores[WINDOW + MAX_COMP_POS], acc[HEAD_DIM], top, denom, v;
	uint32_t h, j, d;
	for (h = 0u; h < HEADS_LOCAL; h++)
	{
		const uint16_t *row;
		top = -3.4e38f;
		for (j = 0u; j < count; j++)
		{
			row = AttnRow(st, (uint32_t)valid[j]);
			v = 0.0f;
			for (d = 0u; d < HEAD_DIM; d++)
				v += Bf16ToF32(q[h * HEAD_DIM + d]) * Bf16ToF32(row[d]);
			scores[j] = v * (1.0f / sqrtf(512.0f));
			if ( scores[j] > top )
				top = scores[j];
		}
		denom = expf(sinks[h] - top);
		for (d = 0u; d < HEAD_DIM; d++)
			acc[d] = 0.0f;
		for (j = 0u; j < count; j++)
		{
			row = AttnRow(st, (uint32_t)valid[j]);
			float ex = expf(scores[j] - top);
			denom += ex;
			for (d = 0u; d < HEAD_DIM; d++)
				acc[d] += ex * Bf16ToF32(row[d]);
		}
		for (d = 0u; d < HEAD_DIM; d++)
			out[h * HEAD_DIM + d] = F32ToBf16(acc[d] / denom);
	}
}

static void WindowIdxs(uint32_t pos, int32_t *idxs, uint32_t *count)
{
	uint32_t oldest, i, n = 0u;
	if ( pos == 0u )
	{
		idxs[0] = 0;
		*count = 1u;
		return;
	}
	oldest = pos % WINDOW + 1u;
	for (i = oldest; i < WINDOW; i++)
		idxs[n++] = (int32_t)i;
	for (i = 0u; i < oldest; i++)
		idxs[n++] = (int32_t)i;
	for (i = 0u; i < n; i++)
	{
		if ( (uint32_t)idxs[i] > pos )
			idxs[i] = -1;
	}
	*count = n;
}

static void EmitF32(Expect *expect, const char *name, const float *values, uint32_t count)
{
	Piece *piece = 0;
	const float *expected;
	float delta, limit;
	uint32_t i;
	for (i = 0u; i < expect->count; i++)
	{
		if ( strcmp(expect->pieces[i].name, name) == 0 )
		{
			piece = &expect->pieces[i];
			break;
		}
	}
	if ( piece == 0 )
	{
		expect->missing += 1u;
		return;
	}
	if ( piece->count != count )
	{
		(void)fprintf(stderr, "SHAPE %s: got %u want %u\n", name, count, piece->count);
		expect->piece_failures += 1u;
		return;
	}
	piece->seen = 1u;
	expected = (const float *)(expect->data + piece->offset);
	for (i = 0u; i < count; i++)
	{
		delta = fabsf(values[i] - expected[i]);
		limit = piece->tol_abs + piece->tol_rel *
			(fabsf(expected[i]) + piece->tol_scale);
		if ( delta > piece->max_abs )
			piece->max_abs = delta;
		if ( delta > limit )
		{
			if ( piece->bad == 0u )
				(void)fprintf(stderr, "FAIL %s [%u]: got %.7g want %.7g (limit %.7g)\n",
					name, i, (double)values[i], (double)expected[i], (double)limit);
			piece->bad += 1u;
		}
	}
	if ( piece->bad != 0u )
		expect->piece_failures += 1u;
}

static const uint8_t *PackPlane(const Pack *pack, uint32_t layer, uint32_t kind,
	uint32_t want_scale, uint64_t *bytes)
{
	uint32_t i, want_layer;
	want_layer = kind <= K_FNORM ? GLOBAL_LAYER : layer;
	for (i = 0u; i < pack->entry_count; i++)
	{
		const PackEntry *entry = &pack->entries[i];
		if ( entry->kind != kind || entry->layer != want_layer )
			continue;
		if ( want_scale == 0u )
		{
			*bytes = entry->payload_bytes;
			return pack->map + entry->payload_offset;
		}
		*bytes = entry->scale_bytes;
		return pack->map + entry->scale_offset;
	}
	(void)fprintf(stderr, "pack missing kind %u plane %u layer %u\n", kind,
		want_scale, layer);
	exit(FAIL_EXIT);
}

static const uint8_t *PackExpert(const Pack *pack, uint32_t layer, uint32_t kind,
	uint32_t group, uint64_t *bytes)
{
	uint32_t i;
	for (i = 0u; i < pack->entry_count; i++)
	{
		const PackEntry *entry = &pack->entries[i];
		if ( entry->kind != kind || entry->layer != layer )
			continue;
		*bytes = entry->payload_bytes / entry->groups;
		return pack->map + entry->payload_offset + (uint64_t)group * (*bytes);
	}
	(void)fprintf(stderr, "pack missing expert kind %u\n", kind);
	exit(FAIL_EXIT);
}

static const uint8_t *PackExpertScale(const Pack *pack, uint32_t layer, uint32_t kind,
	uint32_t group, uint64_t *bytes)
{
	uint32_t i;
	for (i = 0u; i < pack->entry_count; i++)
	{
		const PackEntry *entry = &pack->entries[i];
		if ( entry->kind != kind || entry->layer != layer )
			continue;
		*bytes = entry->scale_bytes / entry->groups;
		return pack->map + entry->scale_offset + (uint64_t)group * (*bytes);
	}
	(void)fprintf(stderr, "pack missing expert scale %u\n", kind);
	exit(FAIL_EXIT);
}

static void EmitPiece(const char *tag, const char *suffix, Expect *expect,
	const float *values, uint32_t count)
{
	char name[64];
	if ( tag == 0 )
		return;
	(void)snprintf(name, sizeof(name), "%s.%s", tag, suffix);
	EmitF32(expect, name, values, count);
}

static void EmitBf16(const char *tag, const char *suffix, Expect *expect,
	const uint16_t *values, uint32_t count)
{
	float piece[HC * HIDDEN];
	uint32_t i;
	if ( tag == 0 )
		return;
	for (i = 0u; i < count; i++)
		piece[i] = Bf16ToF32(values[i]);
	EmitPiece(tag, suffix, expect, piece, count);
}

static uint32_t CompressorStep(const BlockWeights *w, BlockState *st,
	const uint16_t *x, uint32_t pos, Expect *expect, const char *tag,
	uint16_t *latent)
{
	float kv[KV_LATENT], score[KV_LATENT], pooled[KV_LATENT], lf[KV_LATENT];
	uint16_t latent_b[KV_LATENT];
	uint32_t r, d, slot;
	for (r = 0u; r < KV_LATENT; r++)
	{
		float kv_acc = 0.0f, sc_acc = 0.0f;
		for (d = 0u; d < HIDDEN; d++)
		{
			float xd = Bf16ToF32(x[d]);
			kv_acc += Bf16ToF32(w->c_wkv[(uint64_t)r * HIDDEN + d]) * xd;
			sc_acc += Bf16ToF32(w->c_wgate[(uint64_t)r * HIDDEN + d]) * xd;
		}
		kv[r] = kv_acc;
		score[r] = sc_acc;
	}
	slot = pos % COMP_RATIO;
	for (d = 0u; d < KV_LATENT; d++)
	{
		st->kv_state[slot][d] = kv[d];
		st->score_state[slot][d] = score[d];
	}
	EmitPiece(tag, "c_kv_proj", expect, kv, KV_LATENT);
	EmitPiece(tag, "c_gate_score", expect, score, KV_LATENT);
	EmitPiece(tag, "c_state_kv", expect, &st->kv_state[0][0], COMP_RATIO * KV_LATENT);
	EmitPiece(tag, "c_state_score", expect, &st->score_state[0][0],
		COMP_RATIO * KV_LATENT);
	if ( (pos + 1u) % COMP_RATIO != 0u )
		return 0u;
	for (d = 0u; d < KV_LATENT; d++)
	{
		float s0 = st->score_state[0][d], s1 = st->score_state[1][d];
		float mx = s0 > s1 ? s0 : s1;
		float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
		float sum = e0 + e1;
		pooled[d] = st->kv_state[0][d] * (e0 / sum) + st->kv_state[1][d] * (e1 / sum);
		latent_b[d] = F32ToBf16(pooled[d]);
	}
	RmsNorm(latent_b, w->c_norm, KV_LATENT, latent);
	for (d = 0u; d < KV_LATENT; d++)
		lf[d] = Bf16ToF32(latent[d]);
	EmitPiece(tag, "c_latent", expect, lf, KV_LATENT);
	return 1u;
}

static void IndexerPublishK(const BlockWeights *w, BlockState *st,
	const uint16_t *latent, uint32_t pos, Expect *expect, const char *tag)
{
	uint16_t k[IDX_DIM];
	float kf[IDX_DIM];
	uint32_t i;
	Bf16Gemv(latent, w->i_wk, IDX_DIM, KV_LATENT, k);
	RmsNorm(k, w->i_kn, IDX_DIM, k);
	ApplyRopeTailN(k, IDX_DIM, pos + 1u - COMP_RATIO, w->rope_table, 0u);
	Fp4RoundTripE8M0(k, IDX_DIM);
	memcpy(st->k_cache[pos / COMP_RATIO], k, IDX_DIM * sizeof(uint16_t));
	for (i = 0u; i < IDX_DIM; i++)
		kf[i] = Bf16ToF32(k[i]);
	EmitPiece(tag, "idx_k", expect, kf, IDX_DIM);
}

static void CompressKvPublish(const BlockWeights *w, BlockState *st,
	const uint16_t *latent, uint32_t pos, Expect *expect, const char *tag)
{
	uint16_t row[KV_LATENT];
	float rf[KV_LATENT];
	uint32_t i;
	memcpy(row, latent, KV_LATENT * sizeof(uint16_t));
	ApplyRopeTailN(row, KV_LATENT, pos + 1u - COMP_RATIO, w->rope_table, 0u);
	Fp4RoundTripE4M3(row, KV_LATENT);
	memcpy(st->comp_cache[pos / COMP_RATIO], row, KV_LATENT * sizeof(uint16_t));
	for (i = 0u; i < KV_LATENT; i++)
		rf[i] = Bf16ToF32(row[i]);
	EmitPiece(tag, "c_kv_row", expect, rf, KV_LATENT);
}

static void IndexerScore(const BlockWeights *w, BlockState *st, const uint16_t *x,
	const uint16_t *qr, uint32_t pos, Expect *expect, const char *tag,
	int32_t *mapped, uint32_t *n_out)
{
	uint16_t iq[IDX_HEADS_LOCAL * IDX_DIM], wgt[IDX_HEADS_LOCAL];
	float score[MAX_COMP_POS], sf[MAX_COMP_POS], tf[MAX_COMP_POS];
	uint32_t h, t, d, i, n;
	n = (pos + 1u) / COMP_RATIO;
	GemvFp8(qr, w->i_qb, w->i_qb_s, IDX_HEADS_LOCAL * IDX_DIM, Q_LORA, iq);
	for (h = 0u; h < IDX_HEADS_LOCAL; h++)
		ApplyRopeTailN(iq + h * IDX_DIM, IDX_DIM, pos, w->rope_table, 0u);
	Fp4RoundTripE8M0(iq, IDX_HEADS_LOCAL * IDX_DIM);
	Bf16Gemv(x, w->i_wp, IDX_HEADS_LOCAL, HIDDEN, wgt);
	for (i = 0u; i < IDX_HEADS_LOCAL; i++)
		wgt[i] = F32ToBf16(Bf16ToF32(wgt[i]) * 0.015625f);
	for (t = 0u; t < n; t++)
	{
		float acc = 0.0f;
		for (h = 0u; h < IDX_HEADS_LOCAL; h++)
		{
			float dot = 0.0f, rf;
			uint16_t db, pb;
			for (d = 0u; d < IDX_DIM; d++)
				dot += Bf16ToF32(iq[h * IDX_DIM + d]) * Bf16ToF32(st->k_cache[t][d]);
			db = F32ToBf16(dot);
			rf = Bf16ToF32(db);
			rf = rf < 0.0f ? 0.0f : rf;
			pb = F32ToBf16(rf * Bf16ToF32(wgt[h]));
			acc += Bf16ToF32(pb);
		}
		score[t] = Bf16ToF32(F32ToBf16(acc));
		mapped[t] = (int32_t)(t + WINDOW);
	}
	EmitBf16(tag, "idx_q", expect, iq, IDX_HEADS_LOCAL * IDX_DIM);
	EmitBf16(tag, "idx_w", expect, wgt, IDX_HEADS_LOCAL);
	if ( n != 0u )
	{
		for (t = 0u; t < n; t++)
		{
			sf[t] = score[t];
			tf[t] = (float)mapped[t];
		}
		EmitPiece(tag, "idx_score", expect, sf, n);
		EmitPiece(tag, "idx_topk", expect, tf, n);
	}
	*n_out = n;
}

static void OaProject(const BlockWeights *w, const uint16_t *o, uint16_t *partial_b)
{
	float acc, oa_val, weight_val;
	uint32_t r, d;
	for (r = 0u; r < OA_ROWS; r++)
	{
		acc = 0.0f;
		for (d = 0u; d < OA_FULL_COLS; d++)
		{
			oa_val = fp8_lut[w->oa[(uint64_t)r * OA_FULL_COLS + d]] *
				E8M0ToF32(w->oa_s[(uint64_t)(r / 32u) * (OA_FULL_COLS / 32u) + d / 32u]);
			weight_val = Bf16ToF32(F32ToBf16(oa_val));
			acc += weight_val * Bf16ToF32(o[d]);
		}
		partial_b[r] = F32ToBf16(acc);
	}
}

static void QkvProject(const BlockWeights *w, const uint16_t *x, uint32_t pos,
	Expect *expect, const char *tag, uint16_t *qr, uint16_t *q, uint16_t *kv_row)
{
	uint16_t qh[HEAD_DIM], kv_normed[KV_LATENT];
	uint32_t h, d;
	GemvFp8(x, w->qa, w->qa_s, Q_LORA, HIDDEN, qr);
	RmsNorm(qr, w->q_norm, Q_LORA, qr);
	EmitBf16(tag, "q_lora", expect, qr, Q_LORA);
	for (h = 0u; h < HEADS_LOCAL; h++)
	{
		GemvFp8(qr, w->qb + (uint64_t)h * HEAD_DIM * Q_LORA,
			w->qb_s + (uint64_t)h * (HEAD_DIM / 32u) * (Q_LORA / 32u),
			HEAD_DIM, Q_LORA, qh);
		ApplyRopeTailN(qh, HEAD_DIM, pos, w->rope_table, 0u);
		for (d = 0u; d < HEAD_DIM; d++)
			q[h * HEAD_DIM + d] = qh[d];
	}
	EmitBf16(tag, "q", expect, q, HEADS_LOCAL * HEAD_DIM);
	GemvFp8(x, w->kva, w->kva_s, KV_LATENT, HIDDEN, kv_normed);
	RmsNorm(kv_normed, w->kv_norm, KV_LATENT, kv_normed);
	ApplyRopeTailN(kv_normed, KV_LATENT, pos, w->rope_table, 0u);
	ActDequantRow(kv_normed, KV_LATENT, kv_row);
	EmitBf16(tag, "kv_row", expect, kv_row, KV_LATENT);
}

static void AttentionStep(const BlockWeights *w, BlockState *st, const uint16_t *x,
	uint32_t pos, Expect *expect, const char *tag, uint16_t *attn_out)
{
	float sinks[SINKS_LOCAL];
	uint16_t qr[Q_LORA], q[HEADS_LOCAL * HEAD_DIM];
	uint16_t kv_row[KV_LATENT], o[HEADS_LOCAL * HEAD_DIM];
	uint16_t partial_b[OA_ROWS], latent[KV_LATENT];
	int32_t idxs[WINDOW], valid[WINDOW + MAX_COMP_POS], mapped[MAX_COMP_POS];
	uint32_t h, i, count = 0u, valid_count = 0u, slot, n_comp = 0u, published;
	QkvProject(w, x, pos, expect, tag, qr, q, kv_row);
	slot = pos % WINDOW;
	for (i = 0u; i < KV_LATENT; i++)
		st->window[slot][i] = kv_row[i];
	if ( pos == 0u || slot >= st->window_len )
		st->window_len = slot + 1u;
	published = 0u;
	if ( w->is_csa2 != 0u )
		published = CompressorStep(w, st, x, pos, expect, tag, latent);
	if ( published != 0u )
	{
		IndexerPublishK(w, st, latent, pos, expect, tag);
		CompressKvPublish(w, st, latent, pos, expect, tag);
	}
	WindowIdxs(pos, idxs, &count);
	for (i = 0u; i < count; i++)
	{
		if ( idxs[i] >= 0 && (uint32_t)idxs[i] < st->window_len )
			valid[valid_count++] = idxs[i];
	}
	if ( w->is_csa2 != 0u )
	{
		IndexerScore(w, st, x, qr, pos, expect, tag, mapped, &n_comp);
		for (i = 0u; i < n_comp; i++)
			valid[valid_count++] = mapped[i];
	}
	for (i = 0u; i < SINKS_LOCAL; i++)
		sinks[i] = w->sink[i];
	SparseAttn(q, st, valid, valid_count, sinks, o);
	for (h = 0u; h < HEADS_LOCAL; h++)
		ApplyRopeTailN(o + h * HEAD_DIM, HEAD_DIM, pos, w->rope_table, 1u);
	EmitBf16(tag, "attn_out_rope_inv", expect, o, HEADS_LOCAL * HEAD_DIM);
	OaProject(w, o, partial_b);
	GemvFp8(partial_b, w->ob, w->ob_s, HIDDEN, OB_COLS, attn_out);
	EmitBf16(tag, "wo_b_out", expect, attn_out, HIDDEN);
}

static void MoeStep(const BlockWeights *w, const Pack *pack, const uint16_t *x,
	Expect *expect, const char *tag, uint16_t *moe_out)
{
	float scores[N_EXPERTS], routed[HIDDEN];
	uint32_t idx[TOPK], slot, d, local, expert;
	float weights[TOPK];
	uint16_t contrib[HIDDEN], shared[HIDDEN];
	const uint8_t *w1, *w1s, *w3, *w3s, *w2, *w2s;
	uint64_t nbytes;
	GateScores(x, w->router, scores);
	EmitPiece(tag, "router_scores", expect, scores, N_EXPERTS);
	GateSelect(scores, w->rbias, idx, weights);
	if ( tag != 0 )
	{
		float piece[TOPK];
		for (slot = 0u; slot < TOPK; slot++)
			piece[slot] = (float)idx[slot];
		EmitPiece(tag, "router_indices", expect, piece, TOPK);
		EmitPiece(tag, "router_weights", expect, weights, TOPK);
	}
	for (d = 0u; d < HIDDEN; d++)
		routed[d] = 0.0f;
	for (slot = 0u; slot < TOPK; slot++)
	{
		expert = idx[slot];
		if ( expert < LOCAL_EXPERTS )
		{
			local = expert;
			w1 = PackExpert(pack, w->layer, K_W1, local, &nbytes);
			w1s = PackExpertScale(pack, w->layer, K_W1, local, &nbytes);
			w3 = PackExpert(pack, w->layer, K_W3, local, &nbytes);
			w3s = PackExpertScale(pack, w->layer, K_W3, local, &nbytes);
			w2 = PackExpert(pack, w->layer, K_W2, local, &nbytes);
			w2s = PackExpertScale(pack, w->layer, K_W2, local, &nbytes);
			ExpertMlp(x, w1, w1s, w3, w3s, w2, w2s, weights[slot], contrib);
			for (d = 0u; d < HIDDEN; d++)
				routed[d] += Bf16ToF32(contrib[d]);
		}
	}
	SharedMlp(x, w->sw1, w->sw1_s, w->sw3, w->sw3_s, w->sw2, w->sw2_s, shared);
	for (d = 0u; d < HIDDEN; d++)
		moe_out[d] = F32ToBf16(routed[d] + Bf16ToF32(shared[d]));
	EmitPiece(tag, "routed_sum", expect, routed, HIDDEN);
	EmitBf16(tag, "shared_out", expect, shared, HIDDEN);
	EmitBf16(tag, "moe_out", expect, moe_out, HIDDEN);
}

static void BlockForward(const BlockWeights *w, BlockState *st,
	const uint16_t *stream_in, uint32_t pos, Expect *expect, const char *tag,
	uint16_t *stream_out)
{
	float mixes_a[HC_ROWS], pre_a[HC], post_a[HC], comb_a[HC * HC];
	float mixes_f[HC_ROWS], pre_f[HC], post_f[HC], comb_f[HC * HC];
	uint16_t collapsed[HIDDEN], normed[HIDDEN], attn_out[HIDDEN];
	uint16_t stream_a[HC * HIDDEN], moe_out[HIDDEN];
	HcMixes(stream_in, w->hc_a_fn, w->hc_a_scale, w->hc_a_base, mixes_a, pre_a,
		post_a, comb_a);
	HcPre(stream_in, st->pre_mix, collapsed);
	RmsNorm(collapsed, w->attn_norm, HIDDEN, normed);
	if ( tag != 0 )
	{
		EmitBf16(tag, "collapsed_attn", expect, collapsed, HIDDEN);
		EmitBf16(tag, "normed_attn", expect, normed, HIDDEN);
		EmitPiece(tag, "mixes_attn", expect, mixes_a, HC_ROWS);
		EmitPiece(tag, "pre_attn", expect, pre_a, HC);
		EmitPiece(tag, "post_attn", expect, post_a, HC);
		EmitPiece(tag, "comb_attn", expect, comb_a, HC * HC);
	}
	AttentionStep(w, st, normed, pos, expect, tag, attn_out);
	HcPost(attn_out, stream_in, post_a, comb_a, stream_a);
	EmitBf16(tag, "stream_after_attn", expect, stream_a, HC * HIDDEN);
	HcMixes(stream_a, w->hc_f_fn, w->hc_f_scale, w->hc_f_base, mixes_f, pre_f,
		post_f, comb_f);
	HcPre(stream_a, pre_a, collapsed);
	RmsNorm(collapsed, w->ffn_norm, HIDDEN, normed);
	if ( tag != 0 )
	{
		EmitBf16(tag, "collapsed_ffn", expect, collapsed, HIDDEN);
		EmitBf16(tag, "normed_ffn", expect, normed, HIDDEN);
		EmitPiece(tag, "mixes_ffn", expect, mixes_f, HC_ROWS);
		EmitPiece(tag, "pre_ffn", expect, pre_f, HC);
		EmitPiece(tag, "post_ffn", expect, post_f, HC);
		EmitPiece(tag, "comb_ffn", expect, comb_f, HC * HC);
	}
	MoeStep(w, &g_pack, normed, expect, tag, moe_out);
	HcPost(moe_out, stream_a, post_f, comb_f, stream_out);
	memcpy(st->pre_mix, pre_f, sizeof(pre_f));
}

static void LoadWeights(BlockWeights *w, const Pack *pack, uint32_t layer)
{
	uint64_t ignore = 0u;
	memset(w, 0, sizeof(*w));
	w->layer = layer;
	w->is_csa2 = layer != 0u;
	w->rope_table = layer == 0u ? ROPE_TABLE_PURE : ROPE_TABLE_YARN;
	w->embed = (const uint16_t *)PackPlane(pack, layer, K_EMBED, 0u, &ignore);
	w->attn_norm = (const uint16_t *)PackPlane(pack, layer, K_ATTN_NORM, 0u, &ignore);
	w->ffn_norm = (const uint16_t *)PackPlane(pack, layer, K_FFN_NORM, 0u, &ignore);
	w->q_norm = (const uint16_t *)PackPlane(pack, layer, K_QNORM, 0u, &ignore);
	w->kv_norm = (const uint16_t *)PackPlane(pack, layer, K_KVNORM, 0u, &ignore);
	w->router = (const uint16_t *)PackPlane(pack, layer, K_ROUTER, 0u, &ignore);
	w->sink = (const float *)PackPlane(pack, layer, K_SINK, 0u, &ignore);
	w->hc_a_fn = (const float *)PackPlane(pack, layer, K_HCAF, 0u, &ignore);
	w->hc_a_base = (const float *)PackPlane(pack, layer, K_HCAB, 0u, &ignore);
	w->hc_a_scale = (const float *)PackPlane(pack, layer, K_HCAS, 0u, &ignore);
	w->hc_f_fn = (const float *)PackPlane(pack, layer, K_HCFF, 0u, &ignore);
	w->hc_f_base = (const float *)PackPlane(pack, layer, K_HCFB, 0u, &ignore);
	w->hc_f_scale = (const float *)PackPlane(pack, layer, K_HCFS, 0u, &ignore);
	w->rbias = (const float *)PackPlane(pack, layer, K_RBIAS, 0u, &ignore);
	w->qa = PackPlane(pack, layer, K_QA, 0u, &ignore);
	w->qa_s = PackPlane(pack, layer, K_QA, 1u, &ignore);
	w->qb = PackPlane(pack, layer, K_QB, 0u, &ignore);
	w->qb_s = PackPlane(pack, layer, K_QB, 1u, &ignore);
	w->kva = PackPlane(pack, layer, K_KVA, 0u, &ignore);
	w->kva_s = PackPlane(pack, layer, K_KVA, 1u, &ignore);
	w->oa = PackPlane(pack, layer, K_OA, 0u, &ignore);
	w->oa_s = PackPlane(pack, layer, K_OA, 1u, &ignore);
	w->ob = PackPlane(pack, layer, K_OB, 0u, &ignore);
	w->ob_s = PackPlane(pack, layer, K_OB, 1u, &ignore);
	w->sw1 = PackPlane(pack, layer, K_SW1, 0u, &ignore);
	w->sw1_s = PackPlane(pack, layer, K_SW1, 1u, &ignore);
	w->sw2 = PackPlane(pack, layer, K_SW2, 0u, &ignore);
	w->sw2_s = PackPlane(pack, layer, K_SW2, 1u, &ignore);
	w->sw3 = PackPlane(pack, layer, K_SW3, 0u, &ignore);
	w->sw3_s = PackPlane(pack, layer, K_SW3, 1u, &ignore);
	if ( w->is_csa2 == 0u )
		return;
	w->i_qb = PackPlane(pack, layer, K_IQB, 0u, &ignore);
	w->i_qb_s = PackPlane(pack, layer, K_IQB, 1u, &ignore);
	w->i_wk = (const uint16_t *)PackPlane(pack, layer, K_IWK, 0u, &ignore);
	w->i_wp = (const uint16_t *)PackPlane(pack, layer, K_IWP, 0u, &ignore);
	w->i_kn = (const uint16_t *)PackPlane(pack, layer, K_IKN, 0u, &ignore);
	w->c_wkv = (const uint16_t *)PackPlane(pack, layer, K_CWKV, 0u, &ignore);
	w->c_wgate = (const uint16_t *)PackPlane(pack, layer, K_CWGATE, 0u, &ignore);
	w->c_norm = (const uint16_t *)PackPlane(pack, layer, K_CNORM, 0u, &ignore);
}

static void PackOpen(Pack *pack, const char *path)
{
	int fd;
	struct stat st;
	uint32_t i;
	fd = open(path, O_RDONLY);
	if ( fd < 0 || fstat(fd, &st) != 0 )
	{
		(void)fprintf(stderr, "cannot open pack %s\n", path);
		exit(FAIL_EXIT);
	}
	pack->map_bytes = (uint64_t)st.st_size;
	pack->map = (uint8_t *)mmap(0, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if ( pack->map == MAP_FAILED )
	{
		(void)fprintf(stderr, "cannot map pack\n");
		exit(FAIL_EXIT);
	}
	if ( *(uint32_t *)(void *)pack->map != PACK_MAGIC )
	{
		(void)fprintf(stderr, "bad pack magic\n");
		exit(FAIL_EXIT);
	}
	memcpy(&pack->entry_count, pack->map + 24u, 4u);
	pack->entries = (PackEntry *)calloc(pack->entry_count, sizeof(PackEntry));
	if ( pack->entries == 0 )
		exit(FAIL_EXIT);
	for (i = 0u; i < pack->entry_count; i++)
	{
		const uint8_t *record = pack->map + DIRECTORY_OFFSET + (uint64_t)i * ENTRY_BYTES;
		PackEntry *entry = &pack->entries[i];
		memcpy(&entry->kind, record, 4u);
		memcpy(&entry->layer, record + 4u, 4u);
		memcpy(&entry->payload_type, record + 8u, 4u);
		memcpy(&entry->codec, record + 12u, 4u);
		memcpy(&entry->scale_encoding, record + 16u, 4u);
		memcpy(&entry->groups, record + 20u, 4u);
		memcpy(&entry->rows, record + 24u, 4u);
		memcpy(&entry->cols, record + 28u, 4u);
		memcpy(&entry->payload_offset, record + 32u, 8u);
		memcpy(&entry->payload_bytes, record + 40u, 8u);
		memcpy(&entry->scale_offset, record + 48u, 8u);
		memcpy(&entry->scale_bytes, record + 56u, 8u);
	}
}

static void ExpectLoad(Expect *expect, const char *bin_path, const char *table_path)
{
	FILE *table;
	struct stat st;
	uint8_t header[108];
	uint32_t i;
	int fd;
	fd = open(bin_path, O_RDONLY);
	if ( fd < 0 || fstat(fd, &st) != 0 )
	{
		(void)fprintf(stderr, "cannot open %s\n", bin_path);
		exit(FAIL_EXIT);
	}
	expect->data = (uint8_t *)mmap(0, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	(void)close(fd);
	if ( expect->data == MAP_FAILED )
	{
		(void)fprintf(stderr, "cannot map expectations\n");
		exit(FAIL_EXIT);
	}
	table = fopen(table_path, "rb");
	if ( table == 0 || fread(header, 1u, sizeof(header), table) != sizeof(header) )
	{
		(void)fprintf(stderr, "cannot read piece table\n");
		exit(FAIL_EXIT);
	}
	memcpy(&expect->count, header + 104u, 4u);
	expect->pieces = (Piece *)calloc(expect->count, sizeof(Piece));
	if ( expect->pieces == 0 )
		exit(FAIL_EXIT);
	for (i = 0u; i < expect->count; i++)
	{
		Piece *piece = &expect->pieces[i];
		uint8_t record[84];
		if ( fread(record, 1u, sizeof(record), table) != sizeof(record) )
		{
			(void)fprintf(stderr, "short piece table\n");
			exit(FAIL_EXIT);
		}
		memcpy(piece->name, record, 63u);
		memcpy(&piece->offset, record + 64u, 4u);
		memcpy(&piece->count, record + 68u, 4u);
		memcpy(&piece->tol_rel, record + 72u, 4u);
		memcpy(&piece->tol_abs, record + 76u, 4u);
		memcpy(&piece->tol_scale, record + 80u, 4u);
	}
	(void)fclose(table);
}

static void RunLadder(const Pack *pack, const uint16_t *embed, Expect *expect,
	uint32_t layer)
{
	static const float identity[HC] = { 1.0f, 0.0f, 0.0f, 0.0f };
	BlockWeights w;
	BlockState *st;
	char tag[32];
	const char *tag_used;
	uint32_t pos, i;
	static float stream_values[HC * HIDDEN];
	static uint16_t stream_in[HC * HIDDEN], stream_out[HC * HIDDEN];
	LoadWeights(&w, pack, layer);
	st = (BlockState *)calloc(1u, sizeof(BlockState));
	if ( st == 0 )
		exit(FAIL_EXIT);
	memcpy(st->pre_mix, identity, sizeof(identity));
	for (pos = 0u; pos <= TOTAL_POSITIONS; pos++)
	{
		const uint16_t *embed_row;
		int32_t token;
		uint32_t nan_found = 0u;
		token = g_tokens[pos % g_token_count];
		embed_row = embed + (uint64_t)token * HIDDEN;
		for (i = 0u; i < HC * HIDDEN; i++)
			stream_in[i] = embed_row[i % HIDDEN];
		tag_used = 0;
		if ( layer == 0u && pos <= 2u )
		{
			(void)snprintf(tag, sizeof(tag), "pos%u", pos);
			tag_used = tag;
		}
		if ( layer != 0u && pos <= 5u )
		{
			(void)snprintf(tag, sizeof(tag), "l%u.p%u", layer, pos);
			tag_used = tag;
		}
		BlockForward(&w, st, stream_in, pos, expect, tag_used, stream_out);
		if ( layer != 0u && pos > CSA2_STREAM_HORIZON )
			continue;
		if ( layer == 0u )
			(void)snprintf(tag, sizeof(tag), "stream_pos%u", pos);
		else
			(void)snprintf(tag, sizeof(tag), "l%u.stream_pos%u", layer, pos);
		for (i = 0u; i < HC * HIDDEN; i++)
		{
			stream_values[i] = Bf16ToF32(stream_out[i]);
			if ( stream_values[i] != stream_values[i] )
				nan_found = 1u;
		}
		EmitF32(expect, tag, stream_values, HC * HIDDEN);
		if ( nan_found != 0u )
		{
			(void)fprintf(stderr, "NaN in stream at layer %u pos %u\n", layer, pos);
			expect->piece_failures += 1u;
			break;
		}
	}
	free(st);
}

int main(int argc, char **argv)
{
	Expect expect;
	uint8_t header[108];
	FILE *table;
	const uint16_t *embed;
	uint64_t ignore = 0u;
	uint32_t ladder[LADDER_LAYERS] = { 0u, 2u, 8u }, l, i, unseen = 0u;
	if ( argc != 4 )
	{
		(void)fprintf(stderr, "usage: %s <pack> <piece_table> <expectations_bin>\n", argv[0]);
		return(FAIL_EXIT);
	}
	InitLuts();
	RopeInit();
	PackOpen(&g_pack, argv[1]);
	table = fopen(argv[2], "rb");
	if ( table == 0 || fread(header, 1u, sizeof(header), table) != sizeof(header) )
	{
		(void)fprintf(stderr, "cannot read table header\n");
		return(FAIL_EXIT);
	}
	(void)fclose(table);
	memcpy(&g_token_count, header + 16u, 4u);
	memcpy(g_tokens, header + 32u, sizeof(g_tokens));
	ExpectLoad(&expect, argv[3], argv[2]);
	embed = (const uint16_t *)PackPlane(&g_pack, 0u, K_EMBED, 0u, &ignore);
	for (l = 0u; l < LADDER_LAYERS; l++)
		RunLadder(&g_pack, embed, &expect, ladder[l]);
	for (i = 0u; i < expect.count; i++)
	{
		Piece *piece = &expect.pieces[i];
		if ( piece->seen == 0u )
		{
			(void)fprintf(stderr, "UNSEEN piece %s\n", piece->name);
			unseen++;
		}
	}
	(void)fprintf(stderr, "pieces=%u piece_failures=%u missing=%u unseen=%u\n",
		expect.count, expect.piece_failures, expect.missing, unseen);
	for (i = 0u; i < expect.count; i++)
	{
		Piece *piece = &expect.pieces[i];
		if ( piece->bad != 0u )
			(void)printf("FAIL %s max_abs=%.7g bad=%u\n", piece->name,
				(double)piece->max_abs, piece->bad);
	}
	if ( expect.piece_failures == 0u && expect.missing == 0u )
	{
		(void)printf("PASS dsv41 flash anchor: %u pieces agree with host oracle\n",
			expect.count);
		return(0);
	}
	return(1);
}
