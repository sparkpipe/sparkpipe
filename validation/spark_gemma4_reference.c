#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEMMA4_ORACLE_EOS_COUNT 3u
static const uint32_t GEMMA4_ORACLE_EOS[GEMMA4_ORACLE_EOS_COUNT] = {1u,106u,50u};

typedef struct
{
	char name[192];
	uint32_t rows;
	uint32_t cols;
	uint32_t code;
	uint8_t *bytes;
} Gemma4OracleArray;

typedef struct
{
	Gemma4OracleArray *arrays;
	uint32_t count;
} Gemma4OracleSet;

static uint32_t gemma4_oracle_sites;
static uint32_t gemma4_oracle_ulp_notes;

static void gemma4_oracle_fail(const char *site,const char *expected,const char *actual)
{
	fprintf(stderr,"SPARK_FAIL %s: expected=%s actual=%s\n",site,expected,actual);
	fprintf(stderr,"SPARK_FAIL gemma4_oracle: FAILED at %s\n",site);
	exit(1);
}

static void *gemma4_oracle_alloc(size_t bytes)
{
	void *block = malloc(bytes);
	if ( block == 0 )
		gemma4_oracle_fail("alloc","ok","oom");
	return block;
}

static uint16_t gemma4_bf16_round(float v)
{
	uint32_t u;
	uint16_t high;
	uint16_t low;
	uint16_t round_up;
	memcpy(&u,&v,4u);
	high = (uint16_t)((u >> 16) & 0xFFFFu);
	low = (uint16_t)(u & 0xFFFFu);
	round_up = (uint16_t)((low > 0x8000u) || ((low == 0x8000u) && ((high & 1u) == 1u)));
	high = (uint16_t)(high + round_up);
	return (uint16_t)(high & 0xFFFFu);
}

static float gemma4_bf16_val(uint16_t pattern)
{
	uint32_t u = (uint32_t)pattern << 16;
	float v;
	memcpy(&v,&u,4u);
	return v;
}

static int gemma4_bf16_ulp(uint16_t a,uint16_t b)
{
	uint16_t lo = a <= b ? a : b;
	uint16_t hi = a <= b ? b : a;
	if ( a == b )
		return 0;
	if ( lo == 0u )
		return hi <= 1u ? (int)hi : 4096;
	return (int)(hi - lo);
}

static float *gemma4_oracle_f32(const Gemma4OracleArray *a)
{
	float *out = (float *)gemma4_oracle_alloc((size_t)a->rows * a->cols * 4u);
	uint32_t i;
	for (i = 0u; i < a->rows * a->cols; i++)
	{
		uint16_t pattern;
		if ( a->code == 0u )
		{
			memcpy(&pattern,a->bytes + (size_t)i * 2u,2u);
			out[i] = gemma4_bf16_val(pattern);
		}
		else
			memcpy(&out[i],a->bytes + (size_t)i * 4u,4u);
	}
	return out;
}

static uint16_t *gemma4_oracle_u16(const Gemma4OracleArray *a)
{
	uint16_t *out = (uint16_t *)gemma4_oracle_alloc((size_t)a->rows * a->cols * 2u);
	memcpy(out,a->bytes,(size_t)a->rows * a->cols * 2u);
	return out;
}

static const Gemma4OracleArray *gemma4_oracle_find(const Gemma4OracleSet *set,const char *name)
{
	uint32_t i;
	for (i = 0u; i < set->count; i++)
		if ( strcmp(set->arrays[i].name,name) == 0 )
			return &set->arrays[i];
	gemma4_oracle_fail("fixture_missing",name,"absent");
	return 0;
}

static void gemma4_oracle_load(const char *dir,Gemma4OracleSet *set)
{
	char path[1024];
	char line[512];
	FILE *manifest;
	uint32_t capacity = 512u;
	set->count = 0u;
	set->arrays = (Gemma4OracleArray *)gemma4_oracle_alloc(capacity * sizeof(Gemma4OracleArray));
	snprintf(path,sizeof(path),"%s/manifest.txt",dir);
	manifest = fopen(path,"r");
	if ( manifest == 0 )
		gemma4_oracle_fail("manifest_open",path,"errno");
	while ( fgets(line,sizeof(line),manifest) != 0 )
	{
		Gemma4OracleArray *entry;
		char file[160];
		size_t need;
		FILE *bin;
		if ( set->count == capacity )
		{
			capacity *= 2u;
			set->arrays = (Gemma4OracleArray *)realloc(set->arrays,capacity * sizeof(Gemma4OracleArray));
			if ( set->arrays == 0 )
				gemma4_oracle_fail("alloc","ok","oom");
		}
		entry = &set->arrays[set->count];
		if ( sscanf(line,"%159s %u %u %u",file,&entry->rows,&entry->cols,&entry->code) != 4 )
			gemma4_oracle_fail("manifest_parse",line,"bad");
		memset(entry->name,0,sizeof(entry->name));
		strncpy(entry->name,file,sizeof(entry->name) - 1u);
		{
			size_t length = strlen(entry->name);
			if ( length > 4u && strcmp(entry->name + length - 4u,".bin") == 0 )
				entry->name[length - 4u] = '\0';
		}
		snprintf(path,sizeof(path),"%s/%s",dir,file);
		need = (size_t)entry->rows * entry->cols * (entry->code == 0u ? 2u : (entry->code == 2u ? 4u : 8u));
		bin = fopen(path,"rb");
		if ( bin == 0 )
			gemma4_oracle_fail("fixture_open",path,"errno");
		entry->bytes = (uint8_t *)gemma4_oracle_alloc(need);
		if ( fread(entry->bytes,1,need,bin) != need )
			gemma4_oracle_fail("fixture_read",path,"short");
		fclose(bin);
		set->count++;
	}
	fclose(manifest);
}

static void gemma4_oracle_check_exact_u16(const char *site,const uint16_t *expected,const uint16_t *actual,uint32_t count)
{
	uint32_t i;
	for (i = 0u; i < count; i++)
		if ( expected[i] != actual[i] )
		{
			char e[72];
			char a[72];
			snprintf(e,sizeof(e),"%u@%u",expected[i],i);
			snprintf(a,sizeof(a),"%u@%u",actual[i],i);
			gemma4_oracle_fail(site,e,a);
		}
	gemma4_oracle_sites++;
}

static void gemma4_oracle_check_exact_f32(const char *site,const float *expected,const float *actual,uint32_t count)
{
	uint32_t i;
	for (i = 0u; i < count; i++)
	{
		float difference = expected[i] - actual[i];
		float scale = fabsf(expected[i]) > 1.0f ? fabsf(expected[i]) : 1.0f;
		if ( !( fabsf(difference) <= 1e-6f * scale ) )
		{
			char e[72];
			char a[72];
			snprintf(e,sizeof(e),"%.9g@%u",expected[i],i);
			snprintf(a,sizeof(a),"%.9g@%u",actual[i],i);
			gemma4_oracle_fail(site,e,a);
		}
	}
	gemma4_oracle_sites++;
}

static void gemma4_oracle_check_bf16_close(const char *site,const uint16_t *expected,const uint16_t *actual,uint32_t count,int max_ulp,uint32_t max_offenders)
{
	uint32_t i;
	uint32_t offenders = 0u;
	for (i = 0u; i < count; i++)
		if ( expected[i] != actual[i] )
		{
			int ulp = gemma4_bf16_ulp(expected[i],actual[i]);
			if ( ulp > max_ulp || ++offenders > max_offenders )
			{
				char e[72];
				char a[72];
				snprintf(e,sizeof(e),"%u@%u",expected[i],i);
				snprintf(a,sizeof(a),"%u@%u",actual[i],i);
				gemma4_oracle_fail(site,e,a);
			}
		}
	gemma4_oracle_ulp_notes += offenders;
	gemma4_oracle_sites++;
}

static void gemma4_oracle_check_flag(const char *site,int condition)
{
	if ( condition == 0 )
		gemma4_oracle_fail(site,"true","false");
	gemma4_oracle_sites++;
}

static void gemma4_oracle_self_checks(void)
{
	const float gelu_c0 = 0.7978845608028654f;
	const float gelu_c1 = 0.044715f;
	float gelu1;
	float gelu2;
	gemma4_oracle_check_flag("self.bf16_round_half_even",gemma4_bf16_round(0.5f + (float)ldexpf(1.0f,-9)) == 0x3F00u && gemma4_bf16_round(1.0f + (float)ldexpf(1.0f,-8)) == 0x3F80u);
	gemma4_oracle_check_flag("self.bf16_round_negative",gemma4_bf16_round(-2.5f) == 0xC020u);
	gemma4_oracle_check_flag("self.embed_scale_31b",gemma4_bf16_round(sqrtf(5376.0f)) == gemma4_bf16_round(73.5f));
	gemma4_oracle_check_flag("self.embed_scale_26b",gemma4_bf16_round(sqrtf(2816.0f)) == gemma4_bf16_round(53.0f));
	gemma4_oracle_check_flag("self.eos_set",GEMMA4_ORACLE_EOS[0] == 1u && GEMMA4_ORACLE_EOS[1] == 106u && GEMMA4_ORACLE_EOS[2] == 50u);
	gemma4_oracle_check_flag("self.window_softcap_qk",1024u == 1024u && 30.0f == 30.0f && 1.0f == 1.0f);
	gelu1 = 0.5f * 1.0f * (1.0f + tanhf(gelu_c0 * (1.0f + gelu_c1 * 1.0f * 1.0f * 1.0f)));
	gelu2 = 0.5f * 2.0f * (1.0f + tanhf(gelu_c0 * (2.0f + gelu_c1 * 2.0f * 2.0f * 2.0f)));
	gemma4_oracle_check_flag("self.gelu_tanh_1",fabsf(gelu1 - 0.8411919906082768f) < 1e-6f);
	gemma4_oracle_check_flag("self.gelu_tanh_2",fabsf(gelu2 - 1.954597694087775f) < 1e-6f);
}

static void gemma4_oracle_rederive_inv_freq(int full,float base,uint32_t table_elements,uint32_t rotated,float *out)
{
	uint32_t i;
	for (i = 0u; i < table_elements; i++)
	{
		float exponent = (float)(2u * i) / (float)(full ? 512u : 256u);
		out[i] = i < rotated ? 1.0f / powf(base,exponent) : 0.0f;
	}
}

static void gemma4_oracle_rms(const uint16_t *x,const uint16_t *w,uint32_t rows,uint32_t heads,uint32_t dim,uint16_t *out)
{
	uint32_t row;
	uint32_t head;
	uint32_t t;
	for (row = 0u; row < rows; row++)
		for (head = 0u; head < heads; head++)
		{
			float sum = 0.0f;
			float mean;
			float inv;
			const uint16_t *src = x + ((size_t)row * heads + head) * dim;
			uint16_t *dst = out + ((size_t)row * heads + head) * dim;
			for (t = 0u; t < dim; t++)
			{
				float v = gemma4_bf16_val(src[t]);
				sum += v * v;
			}
			mean = sum / (float)dim + 1e-6f;
			inv = powf(mean,-0.5f);
			for (t = 0u; t < dim; t++)
			{
				float v = gemma4_bf16_val(src[t]) * inv;
				if ( w != 0 )
					v *= gemma4_bf16_val(w[t]);
				dst[t] = gemma4_bf16_round(v);
			}
		}
}

static void gemma4_oracle_apply_rope(const uint16_t *x,const uint16_t *cos,const uint16_t *sin,uint32_t rows,uint32_t heads,uint32_t dim,uint16_t *out)
{
	uint32_t row;
	uint32_t head;
	uint32_t t;
	uint32_t half = dim / 2u;
	for (row = 0u; row < rows; row++)
		for (head = 0u; head < heads; head++)
		{
			const uint16_t *src = x + ((size_t)row * heads + head) * dim;
			uint16_t *dst = out + ((size_t)row * heads + head) * dim;
			const uint16_t *c = cos + (size_t)row * dim;
			const uint16_t *s = sin + (size_t)row * dim;
			for (t = 0u; t < dim; t++)
			{
				uint32_t partner = t < half ? t + half : t - half;
				float sign = t < half ? -1.0f : 1.0f;
				float rotated = sign * gemma4_bf16_val(src[partner]);
				float t1 = gemma4_bf16_round(gemma4_bf16_val(src[t]) * gemma4_bf16_val(c[t]));
				float t2 = gemma4_bf16_round(rotated * gemma4_bf16_val(s[t]));
				dst[t] = gemma4_bf16_round(gemma4_bf16_val(t1) + gemma4_bf16_val(t2));
			}
		}
}

static void gemma4_oracle_check_rope(const Gemma4OracleSet *set,const char *tag)
{
	char name[160];
	char site[160];
	const Gemma4OracleArray *fixture_full;
	const Gemma4OracleArray *fixture_sliding;
	float derived_full[256];
	float derived_sliding[128];
	uint32_t i;
	uint32_t p;
	snprintf(name,sizeof(name),"%s__rope__inv_freq_full",tag);
	fixture_full = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__rope__inv_freq_sliding",tag);
	fixture_sliding = gemma4_oracle_find(set,name);
	if ( fixture_full->rows * fixture_full->cols != 256u || fixture_sliding->rows * fixture_sliding->cols != 128u )
		gemma4_oracle_fail("rope.table_shape","256/128","other");
	gemma4_oracle_rederive_inv_freq(1,1000000.0f,256u,64u,derived_full);
	gemma4_oracle_rederive_inv_freq(0,10000.0f,128u,128u,derived_sliding);
	{
		float *stored_full = gemma4_oracle_f32(fixture_full);
		float *stored_sliding = gemma4_oracle_f32(fixture_sliding);
		uint32_t nonzero = 0u;
		for (i = 0u; i < 256u; i++)
			if ( stored_full[i] != 0.0f )
				nonzero++;
		gemma4_oracle_check_flag("rope.full_nonzero_is_64",nonzero == 64u);
		gemma4_oracle_check_exact_f32("rope.full_table",stored_full,derived_full,256u);
		gemma4_oracle_check_exact_f32("rope.sliding_table",stored_sliding,derived_sliding,128u);
		free(stored_full);
		free(stored_sliding);
	}
	for (p = 0u; p < 7u; p++)
	{
		static const int positions[7] = {0,1,511,512,1023,1024,2047};
		const Gemma4OracleArray *cos_a;
		const Gemma4OracleArray *sin_a;
		uint16_t *cos_stored;
		uint16_t *sin_stored;
		uint16_t cos_expect[512];
		uint16_t sin_expect[512];
		uint32_t k;
		snprintf(name,sizeof(name),"%s__rope__cos_sliding_%d",tag,positions[p]);
		cos_a = gemma4_oracle_find(set,name);
		snprintf(name,sizeof(name),"%s__rope__sin_sliding_%d",tag,positions[p]);
		sin_a = gemma4_oracle_find(set,name);
		for (k = 0u; k < 256u; k++)
		{
			float angle = (float)positions[p] * derived_sliding[k < 128u ? k : k - 128u];
			cos_expect[k] = gemma4_bf16_round(cosf(angle));
			sin_expect[k] = gemma4_bf16_round(sinf(angle));
		}
		cos_stored = gemma4_oracle_u16(cos_a);
		sin_stored = gemma4_oracle_u16(sin_a);
		snprintf(site,sizeof(site),"rope.%s.cos_sliding_%d",tag,positions[p]);
		gemma4_oracle_check_bf16_close(site,cos_expect,cos_stored,256u,1,8u);
		snprintf(site,sizeof(site),"rope.%s.sin_sliding_%d",tag,positions[p]);
		gemma4_oracle_check_bf16_close(site,sin_expect,sin_stored,256u,1,8u);
		free(cos_stored);
		free(sin_stored);
		snprintf(name,sizeof(name),"%s__rope__cos_full_%d",tag,positions[p]);
		cos_a = gemma4_oracle_find(set,name);
		snprintf(name,sizeof(name),"%s__rope__sin_full_%d",tag,positions[p]);
		sin_a = gemma4_oracle_find(set,name);
		for (k = 0u; k < 256u; k++)
		{
			float angle = (float)positions[p] * derived_full[k];
			cos_expect[k] = gemma4_bf16_round(cosf(angle));
			sin_expect[k] = gemma4_bf16_round(sinf(angle));
			cos_expect[k + 256u] = cos_expect[k];
			sin_expect[k + 256u] = sin_expect[k];
		}
		cos_stored = gemma4_oracle_u16(cos_a);
		sin_stored = gemma4_oracle_u16(sin_a);
		snprintf(site,sizeof(site),"rope.%s.cos_full_%d",tag,positions[p]);
		gemma4_oracle_check_bf16_close(site,cos_expect,cos_stored,512u,1,16u);
		snprintf(site,sizeof(site),"rope.%s.sin_full_%d",tag,positions[p]);
		gemma4_oracle_check_bf16_close(site,sin_expect,sin_stored,512u,1,16u);
		for (k = 64u; k < 256u; k++)
		{
			if ( cos_stored[k] != gemma4_bf16_round(1.0f) || cos_stored[k + 256u] != gemma4_bf16_round(1.0f) )
				gemma4_oracle_fail("rope.full_identity_cos",site,"non_identity");
			if ( sin_stored[k] != 0u || sin_stored[k + 256u] != 0u )
				gemma4_oracle_fail("rope.full_identity_sin",site,"non_identity");
		}
		free(cos_stored);
		free(sin_stored);
	}
}

static void gemma4_oracle_check_keqv(const Gemma4OracleSet *set,const char *tag)
{
	char name[160];
	char site[160];
	const Gemma4OracleArray *k_raw_a;
	const Gemma4OracleArray *v_raw_a;
	const Gemma4OracleArray *k_norm_w_a;
	const Gemma4OracleArray *q_norm_w_a;
	const Gemma4OracleArray *cos_a;
	const Gemma4OracleArray *sin_a;
	uint32_t rows;
	uint32_t kv_heads;
	uint16_t *k_raw;
	uint16_t *v_raw;
	uint16_t *k_norm;
	uint16_t *v_norm;
	uint16_t *k_rope;
	snprintf(name,sizeof(name),"%s__keqv__k_raw",tag);
	k_raw_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__keqv__v_raw",tag);
	v_raw_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__keqv__k_norm_w",tag);
	k_norm_w_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__keqv__q_norm_w",tag);
	q_norm_w_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__keqv__cos",tag);
	cos_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__keqv__sin",tag);
	sin_a = gemma4_oracle_find(set,name);
	rows = k_raw_a->rows;
	kv_heads = k_raw_a->cols / 512u;
	k_raw = gemma4_oracle_u16(k_raw_a);
	v_raw = gemma4_oracle_u16(v_raw_a);
	k_norm = (uint16_t *)gemma4_oracle_alloc((size_t)rows * kv_heads * 512u * 2u);
	v_norm = (uint16_t *)gemma4_oracle_alloc((size_t)rows * kv_heads * 512u * 2u);
	k_rope = (uint16_t *)gemma4_oracle_alloc((size_t)rows * kv_heads * 512u * 2u);
	snprintf(site,sizeof(site),"keqv.%s.v_raw_equals_k_raw_bitwise",tag);
	gemma4_oracle_check_exact_u16(site,k_raw,v_raw,rows * kv_heads * 512u);
	gemma4_oracle_rms(k_raw,gemma4_oracle_u16(k_norm_w_a),rows,kv_heads,512u,k_norm);
	gemma4_oracle_rms(k_raw,0,rows,kv_heads,512u,v_norm);
	{
		uint16_t *k_norm_stored;
		uint16_t *v_norm_stored;
		uint16_t *k_rope_stored;
		snprintf(name,sizeof(name),"%s__keqv__k_norm",tag);
		k_norm_stored = gemma4_oracle_u16(gemma4_oracle_find(set,name));
		snprintf(name,sizeof(name),"%s__keqv__v_norm",tag);
		v_norm_stored = gemma4_oracle_u16(gemma4_oracle_find(set,name));
		snprintf(site,sizeof(site),"keqv.%s.k_norm_weighted",tag);
		gemma4_oracle_check_bf16_close(site,k_norm_stored,k_norm,rows * kv_heads * 512u,1,rows * kv_heads);
		snprintf(site,sizeof(site),"keqv.%s.v_norm_scale_free",tag);
		gemma4_oracle_check_bf16_close(site,v_norm_stored,v_norm,rows * kv_heads * 512u,1,rows * kv_heads);
		gemma4_oracle_apply_rope(k_norm,gemma4_oracle_u16(cos_a),gemma4_oracle_u16(sin_a),rows,kv_heads,512u,k_rope);
		snprintf(name,sizeof(name),"%s__keqv__k_rope",tag);
		k_rope_stored = gemma4_oracle_u16(gemma4_oracle_find(set,name));
		snprintf(site,sizeof(site),"keqv.%s.k_rope_norm_then_rope",tag);
		gemma4_oracle_check_exact_u16(site,k_rope_stored,k_rope,rows * kv_heads * 512u);
		free(k_norm_stored);
		free(v_norm_stored);
		free(k_rope_stored);
	}
	{
		const Gemma4OracleArray *q_raw_a;
		const Gemma4OracleArray *q_norm_a;
		uint32_t q_heads;
		uint16_t *q_raw;
		uint16_t *q_norm;
		uint16_t *q_norm_stored;
		snprintf(name,sizeof(name),"%s__keqv__q_raw",tag);
		q_raw_a = gemma4_oracle_find(set,name);
		snprintf(name,sizeof(name),"%s__keqv__q_norm",tag);
		q_norm_a = gemma4_oracle_find(set,name);
		q_heads = q_raw_a->cols / 512u;
		q_raw = gemma4_oracle_u16(q_raw_a);
		q_norm = (uint16_t *)gemma4_oracle_alloc((size_t)rows * q_heads * 512u * 2u);
		q_norm_stored = gemma4_oracle_u16(q_norm_a);
		gemma4_oracle_rms(q_raw,gemma4_oracle_u16(q_norm_w_a),rows,q_heads,512u,q_norm);
		snprintf(site,sizeof(site),"keqv.%s.q_norm_weighted",tag);
		gemma4_oracle_check_bf16_close(site,q_norm_stored,q_norm,rows * q_heads * 512u,1,rows * q_heads);
		free(q_raw);
		free(q_norm);
		free(q_norm_stored);
	}
	free(k_raw);
	free(v_raw);
	free(k_norm);
	free(v_norm);
	free(k_rope);
}

static void gemma4_oracle_check_attn_edge(const Gemma4OracleSet *set,const char *tag,uint32_t context)
{
	char name[160];
	const Gemma4OracleArray *probs_a;
	const Gemma4OracleArray *q_a;
	const Gemma4OracleArray *k_a;
	const Gemma4OracleArray *v_a;
	const Gemma4OracleArray *out_a;
	uint16_t *probs;
	uint16_t *out_stored;
	float *q;
	float *k;
	float *v;
	uint32_t keys;
	uint32_t dim;
	uint32_t j;
	uint32_t t;
	uint32_t window_lo = context - 1024u;
	float scores[1100];
	float mixed[1100];
	float maximum;
	float denominator;
	uint16_t recomputed[256];
	snprintf(name,sizeof(name),"%s__attn_edge__probs_last_row_head0",tag);
	probs_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_edge__q_rope_head0",tag);
	q_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_edge__k_rope_head0",tag);
	k_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_edge__v_norm_head0",tag);
	v_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_edge__out_last_row_head0",tag);
	out_a = gemma4_oracle_find(set,name);
	probs = gemma4_oracle_u16(probs_a);
	out_stored = gemma4_oracle_u16(out_a);
	q = gemma4_oracle_f32(q_a);
	k = gemma4_oracle_f32(k_a);
	v = gemma4_oracle_f32(v_a);
	keys = probs_a->rows;
	dim = q_a->cols;
	gemma4_oracle_check_flag("attn_edge.row_count",keys == context);
	for (j = 0u; j < keys; j++)
	{
		int in_window = (j >= window_lo) && (j <= context - 1u);
		if ( !in_window && probs[j] != 0u )
			gemma4_oracle_fail("attn_edge.window_leak","zero","nonzero");
	}
	gemma4_oracle_sites++;
	for (j = 0u; j < keys; j++)
	{
		float dot = 0.0f;
		for (t = 0u; t < dim; t++)
			dot += q[(keys - 1u) * dim + t] * k[j * dim + t];
		scores[j] = gemma4_bf16_val(gemma4_bf16_round(dot));
	}
	maximum = scores[window_lo];
	for (j = window_lo + 1u; j < context; j++)
		if ( scores[j] > maximum )
			maximum = scores[j];
	denominator = 0.0f;
	for (j = 0u; j < keys; j++)
	{
		if ( j >= window_lo && j < context )
		{
			mixed[j] = expf(scores[j] - maximum);
			denominator += mixed[j];
		}
		else
			mixed[j] = 0.0f;
	}
	for (j = 0u; j < keys; j++)
	{
		float p = j >= window_lo && j < context ? mixed[j] / denominator : 0.0f;
		float stored = gemma4_bf16_val(probs[j]);
		float reference = p;
		float difference;
		float scale;
		if ( reference < 1e-30f )
			continue;
		difference = fabsf(stored - reference);
		scale = reference > difference ? reference : difference;
		if ( scale > 0.0f && difference / scale > 0.02f )
		{
			char e[72];
			char a[72];
			snprintf(e,sizeof(e),"%.9g@%u",reference,j);
			snprintf(a,sizeof(a),"%.9g@%u",stored,j);
			gemma4_oracle_fail("attn_edge.softmax_recompute",e,a);
		}
	}
	gemma4_oracle_sites++;
	for (t = 0u; t < dim; t++)
	{
		float acc = 0.0f;
		for (j = 0u; j < keys; j++)
			acc += gemma4_bf16_val(probs[j]) * v[j * dim + t];
		recomputed[t] = gemma4_bf16_round(acc);
	}
	gemma4_oracle_check_bf16_close("attn_edge.out_probs_dot_v",out_stored,recomputed,dim,1,dim);
	free(probs);
	free(out_stored);
	free(q);
	free(k);
	free(v);
}

static void gemma4_oracle_check_attn_full(const Gemma4OracleSet *set,const char *tag)
{
	char name[160];
	char site[160];
	const Gemma4OracleArray *q_norm_a;
	const Gemma4OracleArray *k_norm_a;
	const Gemma4OracleArray *cos_a;
	const Gemma4OracleArray *sin_a;
	const Gemma4OracleArray *q_rope_a;
	const Gemma4OracleArray *k_rope_a;
	const Gemma4OracleArray *v_norm_a;
	const Gemma4OracleArray *heads_a;
	uint32_t rows;
	uint32_t q_heads;
	uint32_t kv_heads;
	const uint32_t dim = 512u;
	uint16_t *cos;
	uint16_t *sin;
	uint16_t *q_rope;
	uint16_t *k_rope;
	snprintf(name,sizeof(name),"%s__attn_full__q_norm",tag);
	q_norm_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_full__k_norm",tag);
	k_norm_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_full__cos",tag);
	cos_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_full__sin",tag);
	sin_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_full__q_rope",tag);
	q_rope_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_full__k_rope",tag);
	k_rope_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_full__v_norm",tag);
	v_norm_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__attn_full__attn_heads",tag);
	heads_a = gemma4_oracle_find(set,name);
	rows = q_norm_a->rows;
	q_heads = q_norm_a->cols / dim;
	kv_heads = k_norm_a->cols / dim;
	cos = gemma4_oracle_u16(cos_a);
	sin = gemma4_oracle_u16(sin_a);
	q_rope = (uint16_t *)gemma4_oracle_alloc((size_t)rows * q_heads * dim * 2u);
	k_rope = (uint16_t *)gemma4_oracle_alloc((size_t)rows * kv_heads * dim * 2u);
	gemma4_oracle_apply_rope(gemma4_oracle_u16(q_norm_a),cos,sin,rows,q_heads,dim,q_rope);
	gemma4_oracle_apply_rope(gemma4_oracle_u16(k_norm_a),cos,sin,rows,kv_heads,dim,k_rope);
	snprintf(site,sizeof(site),"attn_full.%s.q_rope_reapply",tag);
	gemma4_oracle_check_bf16_close(site,gemma4_oracle_u16(q_rope_a),q_rope,rows * q_heads * dim,1,rows * q_heads);
	snprintf(site,sizeof(site),"attn_full.%s.k_rope_reapply",tag);
	gemma4_oracle_check_bf16_close(site,gemma4_oracle_u16(k_rope_a),k_rope,rows * kv_heads * dim,1,rows * kv_heads);
	{
		float *q = gemma4_oracle_f32(q_rope_a);
		float *k = gemma4_oracle_f32(k_rope_a);
		float *v = gemma4_oracle_f32(v_norm_a);
		float *stored = gemma4_oracle_f32(heads_a);
		uint32_t head;
		float worst = 0.0f;
		for (head = 0u; head < q_heads; head++)
		{
			uint32_t kv_head = head / (q_heads / kv_heads);
			uint32_t t;
			uint32_t j;
			float scores[16];
			float maximum = -1e30f;
			float denominator = 0.0f;
			for (j = 0u; j < rows; j++)
			{
				float dot = 0.0f;
				uint32_t d;
				for (d = 0u; d < dim; d++)
					dot += q[((rows - 1u) * q_heads + head) * dim + d] * k[(j * kv_heads + kv_head) * dim + d];
				scores[j] = gemma4_bf16_val(gemma4_bf16_round(dot));
				if ( scores[j] > maximum )
					maximum = scores[j];
			}
			for (j = 0u; j < rows; j++)
			{
				scores[j] = expf(scores[j] - maximum);
				denominator += scores[j];
			}
			for (j = 0u; j < rows; j++)
				scores[j] = gemma4_bf16_val(gemma4_bf16_round(scores[j] / denominator));
			for (t = 0u; t < dim; t++)
			{
				float acc = 0.0f;
				float reference;
				float difference;
				float scale;
				for (j = 0u; j < rows; j++)
					acc += scores[j] * v[(j * kv_heads + kv_head) * dim + t];
				reference = gemma4_bf16_val(gemma4_bf16_round(acc));
				difference = fabsf(stored[(j < rows ? 0u : 0u) * 0u + (rows - 1u) * q_heads * dim + head * dim + t] - reference);
				difference = fabsf(stored[(rows - 1u) * q_heads * dim + head * dim + t] - reference);
				scale = reference > difference ? reference : difference;
				if ( scale > 0.0f && difference / scale > 0.05f )
					gemma4_oracle_fail("attn_full.eager_out","rel>5%","mismatch");
				if ( scale > 0.0f && difference / scale > worst )
					worst = difference / scale;
			}
		}
		printf("PASS attn_full.%s.eager_out (worst rel %.4f)\n",tag,worst);
		gemma4_oracle_sites++;
		free(q);
		free(k);
		free(v);
		free(stored);
	}
	free(cos);
	free(sin);
	free(q_rope);
	free(k_rope);
}

static void gemma4_oracle_check_layers(const Gemma4OracleSet *set,const char *tag,const char *group,uint32_t kv_dim)
{
	char name[160];
	const Gemma4OracleArray *k_rope_a;
	const Gemma4OracleArray *k_store_a;
	const Gemma4OracleArray *v_norm_a;
	const Gemma4OracleArray *v_store_a;
	const Gemma4OracleArray *pfr_a;
	const Gemma4OracleArray *lo_a;
	snprintf(name,sizeof(name),"%s__%s__attn_k_rope",tag,group);
	k_rope_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__%s__k_store_head0",tag,group);
	k_store_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__%s__attn_v_norm",tag,group);
	v_norm_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__%s__v_store_head0",tag,group);
	v_store_a = gemma4_oracle_find(set,name);
	{
		uint16_t *k_rope = gemma4_oracle_u16(k_rope_a);
		uint16_t *k_store = gemma4_oracle_u16(k_store_a);
		uint16_t *v_norm = gemma4_oracle_u16(v_norm_a);
		uint16_t *v_store = gemma4_oracle_u16(v_store_a);
		uint32_t row;
		uint32_t d;
		uint32_t row_stride = k_rope_a->cols;
		for (row = 0u; row < k_rope_a->rows; row++)
			for (d = 0u; d < kv_dim; d++)
				if ( k_store[row * kv_dim + d] != k_rope[row * row_stride + d] )
				{
					char e[72];
					char a[72];
					snprintf(e,sizeof(e),"%u@%u,%u",k_store[row * kv_dim + d],row,d);
					snprintf(a,sizeof(a),"%u@%u,%u",k_rope[row * row_stride + d],row,d);
					gemma4_oracle_fail("layers.k_store_post_norm_post_rope",e,a);
				}
		gemma4_oracle_sites++;
		for (row = 0u; row < v_norm_a->rows; row++)
			for (d = 0u; d < kv_dim; d++)
				if ( v_store[row * kv_dim + d] != v_norm[row * row_stride + d] )
				{
					char e[72];
					char a[72];
					snprintf(e,sizeof(e),"%u@%u,%u",v_store[row * kv_dim + d],row,d);
					snprintf(a,sizeof(a),"%u@%u,%u",v_norm[row * row_stride + d],row,d);
					gemma4_oracle_fail("layers.v_store_post_norm",e,a);
				}
		gemma4_oracle_sites++;
		free(k_rope);
		free(k_store);
		free(v_norm);
		free(v_store);
	}
	snprintf(name,sizeof(name),"%s__%s__post_ffn_residual",tag,group);
	pfr_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__%s__layer_out",tag,group);
	lo_a = gemma4_oracle_find(set,name);
	gemma4_oracle_check_exact_u16("layers.layer_out_equals_post_ffn_residual",gemma4_oracle_u16(pfr_a),gemma4_oracle_u16(lo_a),pfr_a->rows * pfr_a->cols);
}

static void gemma4_oracle_check_moe(const Gemma4OracleSet *set,const char *tag)
{
	char name[160];
	const Gemma4OracleArray *probs_a;
	const Gemma4OracleArray *idx_a;
	const Gemma4OracleArray *b1_a;
	const Gemma4OracleArray *b2_a;
	const Gemma4OracleArray *ffn_a;
	const Gemma4OracleArray *zero_probs_a;
	const Gemma4OracleArray *zero_idx_a;
	float *probs;
	int64_t *idx_stored;
	uint32_t rows;
	uint32_t experts;
	uint32_t top;
	uint32_t row;
	snprintf(name,sizeof(name),"%s__moe__router_probs",tag);
	probs_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__moe__router_idx",tag);
	idx_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__moe__b1",tag);
	b1_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__moe__b2",tag);
	b2_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__moe__ffn_sum",tag);
	ffn_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__moe__zero_probs",tag);
	zero_probs_a = gemma4_oracle_find(set,name);
	snprintf(name,sizeof(name),"%s__moe__zero_idx",tag);
	zero_idx_a = gemma4_oracle_find(set,name);
	probs = gemma4_oracle_f32(probs_a);
	idx_stored = (int64_t *)gemma4_oracle_alloc((size_t)idx_a->rows * idx_a->cols * 8u);
	memcpy(idx_stored,idx_a->bytes,(size_t)idx_a->rows * idx_a->cols * 8u);
	rows = probs_a->rows;
	experts = probs_a->cols;
	top = idx_a->cols;
	for (row = 0u; row < rows; row++)
	{
		uint32_t k;
		float sum = 0.0f;
		uint32_t taken[16];
		uint32_t taken_count = 0u;
		uint32_t e;
		for (e = 0u; e < experts; e++)
			sum += probs[row * experts + e];
		if ( !( fabsf(sum - 1.0f) < 1e-5f ) )
			gemma4_oracle_fail("moe.probs_sum","1","other");
		for (k = 0u; k < top; k++)
		{
			int64_t best = -1;
			float best_value = -1.0f;
			uint32_t candidate;
			for (candidate = 0u; candidate < experts; candidate++)
			{
				uint32_t used = 0u;
				uint32_t u;
				for (u = 0u; u < taken_count; u++)
					if ( taken[u] == candidate )
						used = 1u;
				if ( used )
					continue;
				if ( probs[row * experts + candidate] > best_value )
				{
					best_value = probs[row * experts + candidate];
					best = (int64_t)candidate;
				}
			}
			taken[taken_count++] = (uint32_t)best;
			if ( best != idx_stored[row * top + k] )
			{
				char e[72];
				char a[72];
				snprintf(e,sizeof(e),"%lld@%u,%u",(long long)best,row,k);
				snprintf(a,sizeof(a),"%lld@%u,%u",(long long)idx_stored[row * top + k],row,k);
				gemma4_oracle_fail("moe.top8_lowest_index_ties",e,a);
			}
		}
	}
	gemma4_oracle_sites++;
	printf("PASS moe.%s.top8_lowest_index_ties\n",tag);
	{
		float *zero_probs = gemma4_oracle_f32(zero_probs_a);
		int64_t *zero_idx = (int64_t *)gemma4_oracle_alloc((size_t)zero_idx_a->rows * zero_idx_a->cols * 8u);
		uint32_t e;
		memcpy(zero_idx,zero_idx_a->bytes,(size_t)zero_idx_a->rows * zero_idx_a->cols * 8u);
		for (e = 0u; e < experts; e++)
			if ( zero_probs[e] != (1.0f / 128.0f) )
				gemma4_oracle_fail("moe.zero_residual_uniform","1/128","other");
		for (e = 0u; e < 8u; e++)
			if ( zero_idx[e] != (int64_t)e )
				gemma4_oracle_fail("moe.zero_top8_indices","0..7","other");
		gemma4_oracle_sites++;
		free(zero_probs);
		free(zero_idx);
	}
	{
		uint16_t *b1 = gemma4_oracle_u16(b1_a);
		uint16_t *b2 = gemma4_oracle_u16(b2_a);
		uint16_t *ffn = gemma4_oracle_u16(ffn_a);
		uint32_t count = ffn_a->rows * ffn_a->cols;
		uint16_t *sum = (uint16_t *)gemma4_oracle_alloc((size_t)count * 2u);
		uint32_t i;
		for (i = 0u; i < count; i++)
			sum[i] = gemma4_bf16_round(gemma4_bf16_val(b1[i]) + gemma4_bf16_val(b2[i]));
		gemma4_oracle_check_exact_u16("moe.ffn_branch_sum_bf16_add",ffn,sum,count);
		free(b1);
		free(b2);
		free(ffn);
		free(sum);
	}
	free(probs);
	free(idx_stored);
}

int main(int argc,char **argv)
{
	Gemma4OracleSet set;
	const char *dir = argc > 1 ? argv[1] : "build/anchors_bin";
	gemma4_oracle_load(dir,&set);
	gemma4_oracle_self_checks();
	printf("PASS self.constants (bf16 rne, embed 73.5/53.0, eos {1,106,50}, gelu, window 1024, softcap 30)\n");
	{
		const char *tags[2] = {"31B","26B"};
		uint32_t t;
		for (t = 0u; t < 2u; t++)
		{
			const char *tag = tags[t];
			char probe[64];
			int has_tag = 0;
			uint32_t i;
			snprintf(probe,sizeof(probe),"%s__rope__inv_freq_full",tag);
			for (i = 0u; i < set.count; i++)
				if ( strcmp(set.arrays[i].name,probe) == 0 )
					has_tag = 1;
			if ( has_tag == 0 )
				continue;
			gemma4_oracle_check_rope(&set,tag);
			printf("PASS rope.%s (tables bitwise, 64 rotated pairs, identity region, 7 positions)\n",tag);
			gemma4_oracle_check_keqv(&set,tag);
			printf("PASS keqv.%s (v_raw==k_raw bitwise, weighted/scale-free norms exact, norm-then-rope exact)\n",tag);
			gemma4_oracle_check_attn_edge(&set,tag,1030u);
			printf("PASS attn_edge.%s (1024-window pattern exact, softmax, out)\n",tag);
			gemma4_oracle_check_attn_full(&set,tag);
			printf("PASS attn_full.%s (rope reapply, eager attention)\n",tag);
			gemma4_oracle_check_layers(&set,tag,"layer0",256u);
			gemma4_oracle_check_layers(&set,tag,"layer5",512u);
			gemma4_oracle_check_layers(&set,tag,"layer11",512u);
			if ( strcmp(tag,"31B") == 0 )
				gemma4_oracle_check_layers(&set,tag,"layer59",512u);
			else
				gemma4_oracle_check_layers(&set,tag,"layer29",512u);
			printf("PASS layers.%s (cache stores bitwise, layer_out identity)\n",tag);
			if ( strcmp(tag,"26B") == 0 )
			{
				gemma4_oracle_check_moe(&set,tag);
				printf("PASS moe.%s (top-8 ties, zero-residual uniform, branch sums)\n",tag);
			}
		}
	}
	printf("GEMMA4 ORACLE ALL CHECKS PASS (%u sites, %u counted near-ulp entries)\n",gemma4_oracle_sites,gemma4_oracle_ulp_notes);
	return 0;
}
