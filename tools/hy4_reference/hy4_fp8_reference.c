#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define HIDDEN 6144
#define LAYERS 78
#define HEADS 64
#define QK_HEAD 256
#define QK_NOPE 192
#define QK_ROPE 64
#define V_HEAD 256
#define KV_LORA 512
#define Q_LORA 2048
#define KV_B_ROWS_PER_HEAD (QK_NOPE + V_HEAD)
#define HC 4
#define HC_FLAT (HC * HIDDEN)
#define EXPERTS 256
#define TOP_K 8
#define MOE_INTER 2048
#define DENSE_INTER 18432
#define VOCAB 120832
#define INDEX_TOP_K 2048
#define GROUP 32
#define RMS_EPS 1e-5f
#define HC_EPS 1e-6f
#define HC_MAGNITUDE 2.0f
#define ROUTE_SCALE 2.827f
#define SWIGLU_LIMIT 10.0f
#define ROPE_THETA 10000000.0
#define MAX_SEQ 8
#define MAX_POS 128
#define MAX_ROWS 128
#define MAX_TENSORS 4096
#define MAX_FILES 256
#define MAX_CAPTURE 8
#define HEAD_TOP 5
#define ROW_BLOCK 64

enum { DT_F8 = 1, DT_U8 = 2, DT_BF16 = 3, DT_F32 = 4 };

typedef struct
{
	char name[160];
	int file;
	uint64_t offset;
	uint64_t bytes;
	int dtype;
	int ndim;
	int64_t dims[3];
} Tensor;

typedef struct
{
	char name[64];
	int token_count;
	int new_tokens;
	int tokens[MAX_POS];
	int capture_count;
	int capture_layers[MAX_CAPTURE];
	int generated;
	int head_token[MAX_POS];
	float head_score[MAX_POS];
	int head_top_token[MAX_POS][HEAD_TOP];
	float head_top_score[MAX_POS][HEAD_TOP];
	int32_t *route_ids;
	float *route_weights;
	float *streams;
} Sequence;

typedef struct
{
	int seq;
	int pos;
	int token;
} Row;

static Tensor tensors[MAX_TENSORS];
static int tensor_count;
static int file_fds[MAX_FILES];
static int file_count;
static Sequence sequences[MAX_SEQ];
static int sequence_count;
static float *kv_latent;
static float *kv_rope;
static float e4m3_table[256];
static float e8m0_table[256];
static double rope_inv_freq[QK_ROPE / 2];
static double t_start;

static double now_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *message, const char *detail)
{
	fprintf(stderr, "hy4_fp8_reference: %s %s\n", message, detail ? detail : "");
	exit(1);
}

static void *xmalloc(size_t bytes)
{
	void *p = malloc(bytes);
	if ( p == 0 )
		die("out of memory", 0);
	return p;
}

static void *xcalloc(size_t count, size_t bytes)
{
	void *p = calloc(count, bytes);
	if ( p == 0 )
		die("out of memory", 0);
	return p;
}

static void build_tables(void)
{
	int i;
	for ( i = 0; i < 256; i++ )
	{
		int sign = (i >> 7) & 1;
		int exponent = (i >> 3) & 15;
		int mantissa = i & 7;
		double value;
		if ( exponent == 0 )
			value = ldexp((double)mantissa / 8.0, -6);
		else if ( exponent == 15 && mantissa == 7 )
			value = NAN;
		else
			value = ldexp(1.0 + (double)mantissa / 8.0, exponent - 7);
		e4m3_table[i] = (float)(sign ? -value : value);
		e8m0_table[i] = i == 255 ? NAN : (float)ldexp(1.0, i - 127);
	}
	for ( i = 0; i < QK_ROPE / 2; i++ )
		rope_inv_freq[i] = 1.0 / pow(ROPE_THETA, (double)(2 * i) / (double)QK_ROPE);
}

static int tensor_compare(const void *a, const void *b)
{
	return strcmp(((const Tensor *)a)->name, ((const Tensor *)b)->name);
}

static int parse_dtype(const char *text)
{
	if ( strcmp(text, "F8_E4M3") == 0 )
		return DT_F8;
	if ( strcmp(text, "U8") == 0 )
		return DT_U8;
	if ( strcmp(text, "BF16") == 0 )
		return DT_BF16;
	if ( strcmp(text, "F32") == 0 )
		return DT_F32;
	die("unsupported dtype", text);
	return 0;
}

static void load_manifest(const char *path)
{
	FILE *fh = fopen(path, "r");
	char kind[8];
	if ( fh == 0 )
		die("cannot open manifest", path);
	while ( fscanf(fh, "%7s", kind) == 1 )
	{
		if ( strcmp(kind, "F") == 0 )
		{
			char file_path[1024];
			int index;
			if ( fscanf(fh, "%d %1023s", &index, file_path) != 2 || index != file_count || file_count >= MAX_FILES )
				die("bad file line in", path);
			file_fds[file_count] = open(file_path, O_RDONLY);
			if ( file_fds[file_count] < 0 )
				die("cannot open", file_path);
			file_count++;
		}
		else if ( strcmp(kind, "T") == 0 )
		{
			Tensor *t;
			char dtype[16];
			int d;
			if ( tensor_count >= MAX_TENSORS )
				die("too many tensors in", path);
			t = &tensors[tensor_count];
			memset(t, 0, sizeof(*t));
			if ( fscanf(fh, "%159s %d %llu %llu %15s %d", t->name, &t->file, (unsigned long long *)&t->offset, (unsigned long long *)&t->bytes, dtype, &t->ndim) != 6 || t->ndim < 1 || t->ndim > 3 || t->file < 0 || t->file >= file_count )
				die("bad tensor line in", path);
			for ( d = 0; d < t->ndim; d++ )
				if ( fscanf(fh, "%lld", (long long *)&t->dims[d]) != 1 )
					die("bad tensor dims in", path);
			t->dtype = parse_dtype(dtype);
			tensor_count++;
		}
		else
			die("bad manifest record", kind);
	}
	fclose(fh);
	qsort(tensors, (size_t)tensor_count, sizeof(Tensor), tensor_compare);
}

static const Tensor *find_tensor(const char *name)
{
	Tensor key;
	const Tensor *t;
	snprintf(key.name, sizeof(key.name), "%s", name);
	t = bsearch(&key, tensors, (size_t)tensor_count, sizeof(Tensor), tensor_compare);
	if ( t == 0 )
		die("tensor missing", name);
	return t;
}

static const Tensor *find_named(const char *format, int layer)
{
	char name[160];
	snprintf(name, sizeof(name), format, layer);
	return find_tensor(name);
}

static void read_exact(const Tensor *t, uint64_t offset, void *buffer, uint64_t bytes)
{
	uint64_t done = 0;
	if ( offset + bytes > t->bytes )
		die("read past tensor end", t->name);
	while ( done < bytes )
	{
		ssize_t got = pread(file_fds[t->file], (char *)buffer + done, bytes - done, (off_t)(t->offset + offset + done));
		if ( got < 0 && errno == EINTR )
			continue;
		if ( got <= 0 )
			die("short read", t->name);
		done += (uint64_t)got;
	}
}

static float bf16_value(uint16_t bits)
{
	uint32_t word = (uint32_t)bits << 16;
	float value;
	memcpy(&value, &word, sizeof(value));
	return value;
}

static float *load_vector(const Tensor *t, int64_t count)
{
	float *out = xmalloc((size_t)count * sizeof(float));
	int64_t i;
	if ( t->dtype == DT_F32 )
	{
		if ( t->bytes != (uint64_t)count * 4u )
			die("f32 size mismatch", t->name);
		read_exact(t, 0, out, t->bytes);
	}
	else if ( t->dtype == DT_BF16 )
	{
		uint16_t *raw = xmalloc((size_t)count * 2u);
		if ( t->bytes != (uint64_t)count * 2u )
			die("bf16 size mismatch", t->name);
		read_exact(t, 0, raw, t->bytes);
		for ( i = 0; i < count; i++ )
			out[i] = bf16_value(raw[i]);
		free(raw);
	}
	else
		die("vector dtype unsupported", t->name);
	return out;
}

static float dot(const float *a, const float *b, int n)
{
	float acc[16];
	float total = 0.0f;
	int i;
	int j;
	for ( j = 0; j < 16; j++ )
		acc[j] = 0.0f;
	for ( i = 0; i < n; i += 16 )
		for ( j = 0; j < 16; j++ )
			acc[j] += a[i + j] * b[i + j];
	for ( j = 0; j < 16; j++ )
		total += acc[j];
	return total;
}

static void linear_fp8_slab(const Tensor *weight, const Tensor *scale, int64_t slab, int rows, int columns, const float *x, int count, float *y)
{
	int blocks = (rows + ROW_BLOCK - 1) / ROW_BLOCK;
	int groups = columns / GROUP;
	int block;
	if ( weight->dtype != DT_F8 || scale->dtype != DT_U8 || columns % GROUP != 0 || columns % 16 != 0 )
		die("fp8 plane geometry", weight->name);
	if ( (uint64_t)(slab + 1) * (uint64_t)rows * (uint64_t)columns > weight->bytes || (uint64_t)(slab + 1) * (uint64_t)rows * (uint64_t)groups > scale->bytes )
		die("fp8 plane extent", weight->name);
#pragma omp parallel for schedule(dynamic, 1)
	for ( block = 0; block < blocks; block++ )
	{
		int first = block * ROW_BLOCK;
		int take = rows - first < ROW_BLOCK ? rows - first : ROW_BLOCK;
		uint8_t *payload = xmalloc((size_t)take * (size_t)columns);
		uint8_t *scales = xmalloc((size_t)take * (size_t)groups);
		float *row = xmalloc((size_t)columns * sizeof(float));
		int r;
		read_exact(weight, ((uint64_t)slab * (uint64_t)rows + (uint64_t)first) * (uint64_t)columns, payload, (uint64_t)take * (uint64_t)columns);
		read_exact(scale, ((uint64_t)slab * (uint64_t)rows + (uint64_t)first) * (uint64_t)groups, scales, (uint64_t)take * (uint64_t)groups);
		for ( r = 0; r < take; r++ )
		{
			const uint8_t *p = payload + (size_t)r * (size_t)columns;
			const uint8_t *s = scales + (size_t)r * (size_t)groups;
			int c;
			int i;
			for ( c = 0; c < columns; c++ )
				row[c] = e4m3_table[p[c]] * e8m0_table[s[c / GROUP]];
			for ( i = 0; i < count; i++ )
				y[(size_t)i * (size_t)rows + (size_t)(first + r)] = dot(row, x + (size_t)i * (size_t)columns, columns);
		}
		free(payload);
		free(scales);
		free(row);
	}
}

static void linear_fp8(const char *base, int layer, int rows, int columns, const float *x, int count, float *y)
{
	char weight_name[160];
	char scale_name[160];
	char stem[160];
	snprintf(stem, sizeof(stem), base, layer);
	snprintf(weight_name, sizeof(weight_name), "%s.weight", stem);
	snprintf(scale_name, sizeof(scale_name), "%s.weight_scale", stem);
	{
		const Tensor *w = find_tensor(weight_name);
		const Tensor *s = find_tensor(scale_name);
		if ( w->ndim != 2 || w->dims[0] != rows || w->dims[1] != columns )
			die("fp8 shape mismatch", weight_name);
		linear_fp8_slab(w, s, 0, rows, columns, x, count, y);
	}
}

static void linear_bf16(const Tensor *weight, int rows, int columns, const float *x, int count, float *y)
{
	int blocks = (rows + ROW_BLOCK - 1) / ROW_BLOCK;
	int block;
	if ( weight->dtype != DT_BF16 || weight->ndim != 2 || weight->dims[0] != rows || weight->dims[1] != columns )
		die("bf16 plane geometry", weight->name);
#pragma omp parallel for schedule(dynamic, 1)
	for ( block = 0; block < blocks; block++ )
	{
		int first = block * ROW_BLOCK;
		int take = rows - first < ROW_BLOCK ? rows - first : ROW_BLOCK;
		uint16_t *payload = xmalloc((size_t)take * (size_t)columns * 2u);
		float *row = xmalloc((size_t)columns * sizeof(float));
		int r;
		read_exact(weight, (uint64_t)first * (uint64_t)columns * 2u, payload, (uint64_t)take * (uint64_t)columns * 2u);
		for ( r = 0; r < take; r++ )
		{
			int c;
			int i;
			for ( c = 0; c < columns; c++ )
				row[c] = bf16_value(payload[(size_t)r * (size_t)columns + (size_t)c]);
			for ( i = 0; i < count; i++ )
				y[(size_t)i * (size_t)rows + (size_t)(first + r)] = dot(row, x + (size_t)i * (size_t)columns, columns);
		}
		free(payload);
		free(row);
	}
}

static float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static float silu(float x)
{
	return x / (1.0f + expf(-x));
}

static void rms_norm(float *x, const float *weight, int n)
{
	double sum = 0.0;
	float scale;
	int i;
	for ( i = 0; i < n; i++ )
		sum += (double)x[i] * (double)x[i];
	scale = 1.0f / sqrtf((float)(sum / (double)n) + RMS_EPS);
	for ( i = 0; i < n; i++ )
		x[i] = x[i] * scale * weight[i];
}

static void apply_rope(float *x, int position)
{
	int i;
	int half = QK_ROPE / 2;
	for ( i = 0; i < half; i++ )
	{
		float angle = (float)((double)position * rope_inv_freq[i]);
		float c = cosf(angle);
		float s = sinf(angle);
		float a = x[i];
		float b = x[i + half];
		x[i] = a * c - b * s;
		x[i + half] = b * c + a * s;
	}
}

static void hc_pre(const float *streams, const float *fn, const float *scale, const float *base, int mix_count, float *pre, float *post, float *reduced)
{
	double sum = 0.0;
	float inv;
	float mixes[2 * HC];
	int i;
	int m;
	for ( i = 0; i < HC_FLAT; i++ )
		sum += (double)streams[i] * (double)streams[i];
	inv = 1.0f / sqrtf((float)(sum / (double)HC_FLAT) + RMS_EPS);
	for ( m = 0; m < mix_count; m++ )
		mixes[m] = dot(fn + (size_t)m * HC_FLAT, streams, HC_FLAT) * inv;
	for ( m = 0; m < HC; m++ )
		pre[m] = sigmoid(mixes[m] * scale[0] + base[m]) + HC_EPS;
	if ( post != 0 )
		for ( m = 0; m < HC; m++ )
			post[m] = HC_MAGNITUDE * sigmoid(mixes[HC + m] * scale[1] + base[HC + m]) + HC_EPS;
	for ( i = 0; i < HIDDEN; i++ )
	{
		float acc = 0.0f;
		for ( m = 0; m < HC; m++ )
			acc += pre[m] * streams[(size_t)m * HIDDEN + (size_t)i];
		reduced[i] = acc;
	}
}

static float *cache_latent(int layer, int seq, int pos)
{
	return kv_latent + (((size_t)layer * MAX_SEQ + (size_t)seq) * MAX_POS + (size_t)pos) * KV_LORA;
}

static float *cache_rope(int layer, int seq, int pos)
{
	return kv_rope + (((size_t)layer * MAX_SEQ + (size_t)seq) * MAX_POS + (size_t)pos) * QK_ROPE;
}

static void attention(int layer, const Row *rows, int count, const float *x, float *out)
{
	float *gate = xmalloc((size_t)count * HEADS * V_HEAD * sizeof(float));
	float *qa = xmalloc((size_t)count * Q_LORA * sizeof(float));
	float *q = xmalloc((size_t)count * HEADS * QK_HEAD * sizeof(float));
	float *ckv = xmalloc((size_t)count * (KV_LORA + QK_ROPE) * sizeof(float));
	float *heads = xmalloc((size_t)count * HEADS * V_HEAD * sizeof(float));
	float *kv_b = xmalloc((size_t)HEADS * KV_B_ROWS_PER_HEAD * KV_LORA * sizeof(float));
	float *qa_norm = load_vector(find_named("model.layers.%d.self_attn.q_a_layernorm.weight", layer), Q_LORA);
	float *kv_norm = load_vector(find_named("model.layers.%d.self_attn.kv_a_layernorm.weight", layer), KV_LORA);
	float *sinks = load_vector(find_named("model.layers.%d.self_attn.learnable_sink_param", layer), HEADS);
	float scaling = 1.0f / sqrtf((float)QK_HEAD);
	int i;
	int work;
	linear_bf16(find_named("model.layers.%d.self_attn.linear_gate.weight", layer), HEADS * V_HEAD, HIDDEN, x, count, gate);
	for ( i = 0; i < count * HEADS * V_HEAD; i++ )
		gate[i] = sigmoid(gate[i]);
	linear_fp8("model.layers.%d.self_attn.q_a_proj", layer, Q_LORA, HIDDEN, x, count, qa);
	for ( i = 0; i < count; i++ )
		rms_norm(qa + (size_t)i * Q_LORA, qa_norm, Q_LORA);
	linear_fp8("model.layers.%d.self_attn.q_b_proj", layer, HEADS * QK_HEAD, Q_LORA, qa, count, q);
	linear_fp8("model.layers.%d.self_attn.kv_a_proj_with_mqa", layer, KV_LORA + QK_ROPE, HIDDEN, x, count, ckv);
	for ( i = 0; i < count; i++ )
	{
		float *c = ckv + (size_t)i * (KV_LORA + QK_ROPE);
		int h;
		rms_norm(c, kv_norm, KV_LORA);
		apply_rope(c + KV_LORA, rows[i].pos);
		memcpy(cache_latent(layer, rows[i].seq, rows[i].pos), c, KV_LORA * sizeof(float));
		memcpy(cache_rope(layer, rows[i].seq, rows[i].pos), c + KV_LORA, QK_ROPE * sizeof(float));
		for ( h = 0; h < HEADS; h++ )
			apply_rope(q + ((size_t)i * HEADS + (size_t)h) * QK_HEAD + QK_NOPE, rows[i].pos);
	}
	{
		const Tensor *w = find_named("model.layers.%d.self_attn.kv_b_proj.weight", layer);
		const Tensor *s = find_named("model.layers.%d.self_attn.kv_b_proj.weight_scale", layer);
		uint8_t *payload = xmalloc(w->bytes);
		uint8_t *scales = xmalloc(s->bytes);
		size_t e;
		if ( w->bytes != (uint64_t)HEADS * KV_B_ROWS_PER_HEAD * KV_LORA || s->bytes != w->bytes / GROUP )
			die("kv_b geometry", w->name);
		read_exact(w, 0, payload, w->bytes);
		read_exact(s, 0, scales, s->bytes);
		for ( e = 0; e < w->bytes; e++ )
			kv_b[e] = e4m3_table[payload[e]] * e8m0_table[scales[e / GROUP]];
		free(payload);
		free(scales);
	}
#pragma omp parallel for schedule(dynamic, 1)
	for ( work = 0; work < count * HEADS; work++ )
	{
		int r = work / HEADS;
		int h = work % HEADS;
		const float *qh = q + ((size_t)r * HEADS + (size_t)h) * QK_HEAD;
		const float *wk = kv_b + (size_t)h * KV_B_ROWS_PER_HEAD * KV_LORA;
		const float *wv = wk + (size_t)QK_NOPE * KV_LORA;
		float absorbed[KV_LORA];
		float context[KV_LORA];
		float scores[MAX_POS];
		float ceiling;
		float denominator;
		int n;
		int t;
		int span = rows[r].pos + 1;
		for ( n = 0; n < KV_LORA; n++ )
			absorbed[n] = 0.0f;
		for ( t = 0; t < QK_NOPE; t++ )
			for ( n = 0; n < KV_LORA; n++ )
				absorbed[n] += qh[t] * wk[(size_t)t * KV_LORA + (size_t)n];
		ceiling = sinks[h];
		for ( t = 0; t < span; t++ )
		{
			scores[t] = (dot(absorbed, cache_latent(layer, rows[r].seq, t), KV_LORA) + dot(qh + QK_NOPE, cache_rope(layer, rows[r].seq, t), QK_ROPE)) * scaling;
			if ( scores[t] > ceiling )
				ceiling = scores[t];
		}
		denominator = expf(sinks[h] - ceiling);
		for ( t = 0; t < span; t++ )
		{
			scores[t] = expf(scores[t] - ceiling);
			denominator += scores[t];
		}
		for ( n = 0; n < KV_LORA; n++ )
			context[n] = 0.0f;
		for ( t = 0; t < span; t++ )
		{
			const float *latent = cache_latent(layer, rows[r].seq, t);
			float p = scores[t] / denominator;
			for ( n = 0; n < KV_LORA; n++ )
				context[n] += p * latent[n];
		}
		for ( n = 0; n < V_HEAD; n++ )
		{
			size_t index = ((size_t)r * HEADS + (size_t)h) * V_HEAD + (size_t)n;
			heads[index] = dot(wv + (size_t)n * KV_LORA, context, KV_LORA) * gate[index];
		}
	}
	linear_fp8("model.layers.%d.self_attn.o_proj", layer, HIDDEN, HEADS * V_HEAD, heads, count, out);
	free(gate);
	free(qa);
	free(q);
	free(ckv);
	free(heads);
	free(kv_b);
	free(qa_norm);
	free(kv_norm);
	free(sinks);
}

static void swiglu_mlp(const char *stem, int layer, int inter, const float *x, int count, float *out)
{
	char name[160];
	float *gate = xmalloc((size_t)count * (size_t)inter * sizeof(float));
	float *up = xmalloc((size_t)count * (size_t)inter * sizeof(float));
	size_t i;
	snprintf(name, sizeof(name), "%s.gate_proj", stem);
	linear_fp8(name, layer, inter, HIDDEN, x, count, gate);
	snprintf(name, sizeof(name), "%s.up_proj", stem);
	linear_fp8(name, layer, inter, HIDDEN, x, count, up);
	for ( i = 0; i < (size_t)count * (size_t)inter; i++ )
		gate[i] = silu(gate[i]) * up[i];
	snprintf(name, sizeof(name), "%s.down_proj", stem);
	linear_fp8(name, layer, HIDDEN, inter, gate, count, out);
	free(gate);
	free(up);
}

static void moe(int layer, const Row *rows, int count, const float *x, float *out, int32_t *route_ids, float *route_weights)
{
	float *logits = xmalloc((size_t)count * EXPERTS * sizeof(float));
	float *bias = load_vector(find_named("model.layers.%d.mlp.gate.e_score_correction_bias", layer), EXPERTS);
	const Tensor *gate_up = find_named("model.layers.%d.mlp.experts.gate_up_proj", layer);
	const Tensor *gate_up_scale = find_named("model.layers.%d.mlp.experts.gate_up_proj_scale", layer);
	const Tensor *down = find_named("model.layers.%d.mlp.experts.down_proj", layer);
	const Tensor *down_scale = find_named("model.layers.%d.mlp.experts.down_proj_scale", layer);
	int *selected = xmalloc((size_t)count * TOP_K * sizeof(int));
	float *weights = xmalloc((size_t)count * TOP_K * sizeof(float));
	float *shared = xmalloc((size_t)count * HIDDEN * sizeof(float));
	int r;
	int e;
	(void)rows;
	linear_bf16(find_named("model.layers.%d.mlp.gate.weight", layer), EXPERTS, HIDDEN, x, count, logits);
	for ( r = 0; r < count; r++ )
	{
		float scores[EXPERTS];
		float choice[EXPERTS];
		float total = 0.0f;
		int k;
		for ( e = 0; e < EXPERTS; e++ )
		{
			scores[e] = sigmoid(logits[(size_t)r * EXPERTS + (size_t)e]);
			choice[e] = scores[e] + bias[e];
		}
		for ( k = 0; k < TOP_K; k++ )
		{
			int best = -1;
			for ( e = 0; e < EXPERTS; e++ )
				if ( choice[e] != -INFINITY && (best < 0 || choice[e] > choice[best]) )
					best = e;
			selected[r * TOP_K + k] = best;
			weights[r * TOP_K + k] = scores[best];
			total += scores[best];
			choice[best] = -INFINITY;
		}
		for ( k = 0; k < TOP_K; k++ )
		{
			weights[r * TOP_K + k] = weights[r * TOP_K + k] / (total + 1e-20f) * ROUTE_SCALE;
			route_ids[r * TOP_K + k] = selected[r * TOP_K + k];
			route_weights[r * TOP_K + k] = weights[r * TOP_K + k];
		}
	}
	memset(out, 0, (size_t)count * HIDDEN * sizeof(float));
	for ( e = 0; e < EXPERTS; e++ )
	{
		int members[MAX_ROWS];
		float member_weight[MAX_ROWS];
		int member_count = 0;
		float *xe;
		float *gu;
		float *act;
		float *ye;
		int m;
		int k;
		for ( r = 0; r < count; r++ )
			for ( k = 0; k < TOP_K; k++ )
				if ( selected[r * TOP_K + k] == e )
				{
					members[member_count] = r;
					member_weight[member_count] = weights[r * TOP_K + k];
					member_count++;
				}
		if ( member_count == 0 )
			continue;
		xe = xmalloc((size_t)member_count * HIDDEN * sizeof(float));
		gu = xmalloc((size_t)member_count * 2 * MOE_INTER * sizeof(float));
		act = xmalloc((size_t)member_count * MOE_INTER * sizeof(float));
		ye = xmalloc((size_t)member_count * HIDDEN * sizeof(float));
		for ( m = 0; m < member_count; m++ )
			memcpy(xe + (size_t)m * HIDDEN, x + (size_t)members[m] * HIDDEN, HIDDEN * sizeof(float));
		linear_fp8_slab(gate_up, gate_up_scale, e, 2 * MOE_INTER, HIDDEN, xe, member_count, gu);
		for ( m = 0; m < member_count; m++ )
		{
			int i;
			for ( i = 0; i < MOE_INTER; i++ )
			{
				float g = gu[(size_t)m * 2 * MOE_INTER + (size_t)i];
				float u = gu[(size_t)m * 2 * MOE_INTER + MOE_INTER + (size_t)i];
				if ( g > SWIGLU_LIMIT )
					g = SWIGLU_LIMIT;
				if ( u > SWIGLU_LIMIT )
					u = SWIGLU_LIMIT;
				if ( u < -SWIGLU_LIMIT )
					u = -SWIGLU_LIMIT;
				act[(size_t)m * MOE_INTER + (size_t)i] = silu(g) * u;
			}
		}
		linear_fp8_slab(down, down_scale, e, HIDDEN, MOE_INTER, act, member_count, ye);
		for ( m = 0; m < member_count; m++ )
		{
			int i;
			for ( i = 0; i < HIDDEN; i++ )
				out[(size_t)members[m] * HIDDEN + (size_t)i] += member_weight[m] * ye[(size_t)m * HIDDEN + (size_t)i];
		}
		free(xe);
		free(gu);
		free(act);
		free(ye);
	}
	{
		char stem[160];
		snprintf(stem, sizeof(stem), "model.layers.%%d.mlp.shared_experts");
		swiglu_mlp(stem, layer, MOE_INTER, x, count, shared);
	}
	for ( r = 0; r < count * HIDDEN; r++ )
		out[r] += shared[r];
	free(logits);
	free(bias);
	free(selected);
	free(weights);
	free(shared);
}

static void hc_stage(int layer, const char *kind, const float *streams, int count, float *post, float *reduced, const char *norm_name)
{
	char name[160];
	float *fn;
	float *scale;
	float *base;
	float *norm;
	int r;
	snprintf(name, sizeof(name), "model.layers.%d.%s.hc_pre.hc_fn", layer, kind);
	fn = load_vector(find_tensor(name), 2 * HC * HC_FLAT);
	snprintf(name, sizeof(name), "model.layers.%d.%s.hc_pre.hc_scale", layer, kind);
	scale = load_vector(find_tensor(name), 2);
	snprintf(name, sizeof(name), "model.layers.%d.%s.hc_pre.hc_base", layer, kind);
	base = load_vector(find_tensor(name), 2 * HC);
	norm = load_vector(find_named(norm_name, layer), HIDDEN);
#pragma omp parallel for schedule(static)
	for ( r = 0; r < count; r++ )
	{
		float pre[HC];
		hc_pre(streams + (size_t)r * HC_FLAT, fn, scale, base, 2 * HC, pre, post + (size_t)r * HC, reduced + (size_t)r * HIDDEN);
		rms_norm(reduced + (size_t)r * HIDDEN, norm, HIDDEN);
	}
	free(fn);
	free(scale);
	free(base);
	free(norm);
}

static void hc_post(float *streams, const float *post, const float *branch, int count)
{
	int r;
	for ( r = 0; r < count; r++ )
	{
		int m;
		int i;
		for ( m = 0; m < HC; m++ )
			for ( i = 0; i < HIDDEN; i++ )
				streams[((size_t)r * HC + (size_t)m) * HIDDEN + (size_t)i] += post[(size_t)r * HC + (size_t)m] * branch[(size_t)r * HIDDEN + (size_t)i];
	}
}

static void capture_streams(int layer, const Row *rows, int count, const float *streams)
{
	int r;
	for ( r = 0; r < count; r++ )
	{
		Sequence *s = &sequences[rows[r].seq];
		int c;
		for ( c = 0; c < s->capture_count; c++ )
			if ( s->capture_layers[c] == layer )
				memcpy(s->streams + ((size_t)c * MAX_POS + (size_t)rows[r].pos) * HC_FLAT, streams + (size_t)r * HC_FLAT, HC_FLAT * sizeof(float));
	}
}

static void head(const Row *rows, int count, const float *streams, const int *wanted)
{
	float *fn = load_vector(find_tensor("model.hc_head.hc_head_fn"), HC * HC_FLAT);
	float *scale = load_vector(find_tensor("model.hc_head.hc_head_scale"), 1);
	float *base = load_vector(find_tensor("model.hc_head.hc_head_base"), HC);
	float *norm = load_vector(find_tensor("model.norm.weight"), HIDDEN);
	float *x = xmalloc((size_t)count * HIDDEN * sizeof(float));
	float *logits = xmalloc((size_t)count * VOCAB * sizeof(float));
	int r;
	for ( r = 0; r < count; r++ )
	{
		float pre[HC];
		float scales[2];
		scales[0] = scale[0];
		scales[1] = 0.0f;
		hc_pre(streams + (size_t)r * HC_FLAT, fn, scales, base, HC, pre, 0, x + (size_t)r * HIDDEN);
		rms_norm(x + (size_t)r * HIDDEN, norm, HIDDEN);
	}
	linear_bf16(find_tensor("lm_head.weight"), VOCAB, HIDDEN, x, count, logits);
	for ( r = 0; r < count; r++ )
	{
		Sequence *s = &sequences[rows[r].seq];
		const float *l = logits + (size_t)r * VOCAB;
		int top[HEAD_TOP];
		int k;
		if ( !wanted[r] )
			continue;
		for ( k = 0; k < HEAD_TOP; k++ )
		{
			int best = -1;
			int v;
			int j;
			for ( v = 0; v < VOCAB; v++ )
			{
				int used = 0;
				for ( j = 0; j < k; j++ )
					if ( top[j] == v )
						used = 1;
				if ( !used && (best < 0 || l[v] > l[best]) )
					best = v;
			}
			top[k] = best;
			s->head_top_token[rows[r].pos][k] = best;
			s->head_top_score[rows[r].pos][k] = l[best];
		}
		s->head_token[rows[r].pos] = top[0];
		s->head_score[rows[r].pos] = l[top[0]];
	}
	free(fn);
	free(scale);
	free(base);
	free(norm);
	free(x);
	free(logits);
}

static void forward(const Row *rows, int count, const int *wanted)
{
	float *streams = xmalloc((size_t)count * HC_FLAT * sizeof(float));
	float *post = xmalloc((size_t)count * HC * sizeof(float));
	float *x = xmalloc((size_t)count * HIDDEN * sizeof(float));
	float *branch = xmalloc((size_t)count * HIDDEN * sizeof(float));
	int32_t *route_ids = xmalloc((size_t)count * TOP_K * sizeof(int32_t));
	float *route_weights = xmalloc((size_t)count * TOP_K * sizeof(float));
	const Tensor *embed = find_tensor("model.embed_tokens.weight");
	int r;
	int layer;
	for ( r = 0; r < count; r++ )
	{
		uint16_t raw[HIDDEN];
		int m;
		int i;
		if ( rows[r].pos >= INDEX_TOP_K || rows[r].pos >= MAX_POS )
			die("position beyond the reference bound", 0);
		read_exact(embed, (uint64_t)rows[r].token * HIDDEN * 2u, raw, HIDDEN * 2u);
		for ( m = 0; m < HC; m++ )
			for ( i = 0; i < HIDDEN; i++ )
				streams[((size_t)r * HC + (size_t)m) * HIDDEN + (size_t)i] = bf16_value(raw[i]);
	}
	for ( layer = 0; layer < LAYERS; layer++ )
	{
		double t0 = now_seconds();
		hc_stage(layer, "hc_attn_layer", streams, count, post, x, "model.layers.%d.input_layernorm.weight");
		attention(layer, rows, count, x, branch);
		hc_post(streams, post, branch, count);
		hc_stage(layer, "hc_mlp_layer", streams, count, post, x, "model.layers.%d.post_attention_layernorm.weight");
		if ( layer == 0 )
			swiglu_mlp("model.layers.%d.mlp", layer, DENSE_INTER, x, count, branch);
		else
		{
			moe(layer, rows, count, x, branch, route_ids, route_weights);
			for ( r = 0; r < count; r++ )
			{
				Sequence *s = &sequences[rows[r].seq];
				size_t slot = ((size_t)rows[r].pos * (LAYERS - 1) + (size_t)(layer - 1)) * TOP_K;
				memcpy(s->route_ids + slot, route_ids + (size_t)r * TOP_K, TOP_K * sizeof(int32_t));
				memcpy(s->route_weights + slot, route_weights + (size_t)r * TOP_K, TOP_K * sizeof(float));
			}
		}
		hc_post(streams, post, branch, count);
		capture_streams(layer, rows, count, streams);
		fprintf(stderr, "layer %d rows %d %.1fs elapsed %.0fs\n", layer, count, now_seconds() - t0, now_seconds() - t_start);
	}
	head(rows, count, streams, wanted);
	free(streams);
	free(post);
	free(x);
	free(branch);
	free(route_ids);
	free(route_weights);
}

static void load_prompts(const char *path)
{
	FILE *fh = fopen(path, "r");
	if ( fh == 0 )
		die("cannot open prompts", path);
	while ( sequence_count < MAX_SEQ )
	{
		Sequence *s = &sequences[sequence_count];
		int i;
		memset(s, 0, sizeof(*s));
		if ( fscanf(fh, "%63s %d %d %d", s->name, &s->new_tokens, &s->capture_count, &s->token_count) != 4 )
			break;
		if ( s->capture_count > MAX_CAPTURE || s->token_count < 1 || s->token_count + s->new_tokens > MAX_POS )
			die("prompt bounds", s->name);
		for ( i = 0; i < s->capture_count; i++ )
			if ( fscanf(fh, "%d", &s->capture_layers[i]) != 1 || s->capture_layers[i] < 0 || s->capture_layers[i] >= LAYERS )
				die("bad capture layer", s->name);
		for ( i = 0; i < s->token_count; i++ )
			if ( fscanf(fh, "%d", &s->tokens[i]) != 1 || s->tokens[i] < 0 || s->tokens[i] >= VOCAB )
				die("bad token", s->name);
		s->route_ids = xcalloc((size_t)MAX_POS * (LAYERS - 1) * TOP_K, sizeof(int32_t));
		s->route_weights = xcalloc((size_t)MAX_POS * (LAYERS - 1) * TOP_K, sizeof(float));
		s->streams = xcalloc((size_t)(s->capture_count ? s->capture_count : 1) * MAX_POS * HC_FLAT, sizeof(float));
		sequence_count++;
	}
	fclose(fh);
	if ( sequence_count == 0 )
		die("no prompts in", path);
}

static void write_outputs(const char *directory)
{
	int q;
	for ( q = 0; q < sequence_count; q++ )
	{
		Sequence *s = &sequences[q];
		char path[1024];
		FILE *fh;
		int total = s->token_count + s->generated;
		int p;
		int k;
		snprintf(path, sizeof(path), "%s/%s.txt", directory, s->name);
		fh = fopen(path, "w");
		if ( fh == 0 )
			die("cannot write", path);
		fprintf(fh, "prompt");
		for ( p = 0; p < s->token_count; p++ )
			fprintf(fh, " %d", s->tokens[p]);
		fprintf(fh, "\ngenerated");
		for ( p = s->token_count; p < total; p++ )
			fprintf(fh, " %d", s->tokens[p]);
		fprintf(fh, "\n");
		for ( p = s->token_count - 1; p < total - 1; p++ )
		{
			fprintf(fh, "head %d", p);
			for ( k = 0; k < HEAD_TOP; k++ )
				fprintf(fh, " %d:%.6e", s->head_top_token[p][k], (double)s->head_top_score[p][k]);
			fprintf(fh, "\n");
		}
		fclose(fh);
		snprintf(path, sizeof(path), "%s/%s.routes.bin", directory, s->name);
		fh = fopen(path, "wb");
		if ( fh == 0 )
			die("cannot write", path);
		fwrite(s->route_ids, sizeof(int32_t), (size_t)total * (LAYERS - 1) * TOP_K, fh);
		fwrite(s->route_weights, sizeof(float), (size_t)total * (LAYERS - 1) * TOP_K, fh);
		fclose(fh);
		snprintf(path, sizeof(path), "%s/%s.streams.bin", directory, s->name);
		fh = fopen(path, "wb");
		if ( fh == 0 )
			die("cannot write", path);
		for ( k = 0; k < s->capture_count; k++ )
			fwrite(s->streams + (size_t)k * MAX_POS * HC_FLAT, sizeof(float), (size_t)total * HC_FLAT, fh);
		fclose(fh);
	}
}

int main(int argc, char **argv)
{
	Row rows[MAX_ROWS];
	int wanted[MAX_ROWS];
	int count = 0;
	int q;
	int step;
	int max_new = 0;
	if ( argc != 4 )
	{
		fprintf(stderr, "usage: %s <manifest> <prompts> <out_dir>\n", argv[0]);
		return 2;
	}
	t_start = now_seconds();
	build_tables();
	load_manifest(argv[1]);
	load_prompts(argv[2]);
	kv_latent = xcalloc((size_t)LAYERS * MAX_SEQ * MAX_POS * KV_LORA, sizeof(float));
	kv_rope = xcalloc((size_t)LAYERS * MAX_SEQ * MAX_POS * QK_ROPE, sizeof(float));
	fprintf(stderr, "tensors %d files %d prompts %d threads %d\n", tensor_count, file_count, sequence_count, omp_get_max_threads());
	for ( q = 0; q < sequence_count; q++ )
	{
		int p;
		if ( sequences[q].new_tokens > max_new )
			max_new = sequences[q].new_tokens;
		for ( p = 0; p < sequences[q].token_count; p++ )
		{
			if ( count >= MAX_ROWS )
				die("too many prefill rows", 0);
			rows[count].seq = q;
			rows[count].pos = p;
			rows[count].token = sequences[q].tokens[p];
			wanted[count] = p == sequences[q].token_count - 1;
			count++;
		}
	}
	for ( step = 0; step < max_new; step++ )
	{
		forward(rows, count, wanted);
		count = 0;
		for ( q = 0; q < sequence_count; q++ )
		{
			Sequence *s = &sequences[q];
			int last = s->token_count + s->generated - 1;
			if ( s->generated >= s->new_tokens )
				continue;
			s->tokens[last + 1] = s->head_token[last];
			s->generated++;
			fprintf(stderr, "step %d %s pos %d token %d score %.4f\n", step, s->name, last + 1, s->tokens[last + 1], (double)s->head_score[last]);
			if ( s->generated < s->new_tokens )
			{
				rows[count].seq = q;
				rows[count].pos = last + 1;
				rows[count].token = s->tokens[last + 1];
				wanted[count] = 1;
				count++;
			}
		}
		write_outputs(argv[3]);
		if ( count == 0 )
			break;
	}
	fprintf(stderr, "done %.0fs\n", now_seconds() - t_start);
	return 0;
}
