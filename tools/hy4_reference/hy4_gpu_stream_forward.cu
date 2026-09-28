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

#include <vector>

#include "sparkpipe/spark_lm_kernels.cuh"

#define HIDDEN 6144
#define LAYERS 78
#define HEADS 64
#define QK_HEAD 256
#define QK_NOPE 192
#define QK_ROPE 64
#define V_HEAD 256
#define KV_LORA 512
#define KV_A_ROWS (KV_LORA + QK_ROPE)
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
#define EXPERT_BATCH 16
#define THREADS 256

#define CUDA_CHECK(call) do { cudaError_t cuda_status = (call); if ( cuda_status != cudaSuccess ) { fprintf(stderr, "hy4_gpu_stream_forward: %s failed: %s\n", #call, cudaGetErrorString(cuda_status)); exit(1); } } while (0)

enum { DT_F8 = 1, DT_U8 = 2, DT_BF16 = 3, DT_F32 = 4 };

struct Tensor
{
	char name[160];
	int file;
	uint64_t offset;
	uint64_t bytes;
	int dtype;
	int ndim;
	int64_t dims[3];
};

struct Sequence
{
	char name[64];
	int token_count;
	int new_tokens;
	int tokens[MAX_POS];
	int capture_count;
	int capture_layers[MAX_CAPTURE];
	int generated;
	int head_token[MAX_POS];
	int forced[MAX_POS];
	int forced_count;
	int head_top_token[MAX_POS][HEAD_TOP];
	float head_top_score[MAX_POS][HEAD_TOP];
	std::vector<int32_t> route_ids;
	std::vector<float> route_weights;
	std::vector<float> streams;
};

struct Row
{
	int seq;
	int pos;
	int token;
};

static Tensor tensors[MAX_TENSORS];
static int tensor_count;
static int file_fds[MAX_FILES];
static int file_count;
static Sequence sequences[MAX_SEQ];
static int sequence_count;
static double t_start;
static int layer_limit = LAYERS;

static double now_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *message, const char *detail)
{
	fprintf(stderr, "hy4_gpu_stream_forward: %s %s\n", message, detail ? detail : "");
	exit(1);
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
			unsigned long long offset;
			unsigned long long bytes;
			int d;
			if ( tensor_count >= MAX_TENSORS )
				die("too many tensors in", path);
			t = &tensors[tensor_count];
			memset(t, 0, sizeof(*t));
			if ( fscanf(fh, "%159s %d %llu %llu %15s %d", t->name, &t->file, &offset, &bytes, dtype, &t->ndim) != 6 || t->ndim < 1 || t->ndim > 3 || t->file < 0 || t->file >= file_count )
				die("bad tensor line in", path);
			t->offset = offset;
			t->bytes = bytes;
			for ( d = 0; d < t->ndim; d++ )
			{
				long long dim;
				if ( fscanf(fh, "%lld", &dim) != 1 )
					die("bad tensor dims in", path);
				t->dims[d] = dim;
			}
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
	t = (const Tensor *)bsearch(&key, tensors, (size_t)tensor_count, sizeof(Tensor), tensor_compare);
	if ( t == 0 )
		die("tensor missing", name);
	return t;
}

static const Tensor *find_layer(int layer, const char *suffix)
{
	char name[160];
	snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
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

static void read_parallel(const Tensor *t, uint64_t offset, void *buffer, uint64_t bytes)
{
	const uint64_t chunk = 8ull << 20;
	int64_t chunks = (int64_t)((bytes + chunk - 1) / chunk);
	int64_t c;
#pragma omp parallel for schedule(dynamic, 1)
	for ( c = 0; c < chunks; c++ )
	{
		uint64_t first = (uint64_t)c * chunk;
		uint64_t take = bytes - first < chunk ? bytes - first : chunk;
		read_exact(t, offset + first, (char *)buffer + first, take);
	}
}

struct DeviceArena
{
	std::vector<void *> blocks;
	void *host;
	uint64_t host_bytes;
};

static DeviceArena layer_arena;

static void *host_staging(DeviceArena *arena, uint64_t bytes)
{
	if ( arena->host_bytes < bytes )
	{
		free(arena->host);
		arena->host = malloc(bytes);
		if ( arena->host == 0 )
			die("out of host memory", 0);
		arena->host_bytes = bytes;
	}
	return arena->host;
}

static void *upload_range(DeviceArena *arena, const Tensor *t, uint64_t offset, uint64_t bytes)
{
	void *host = host_staging(arena, bytes);
	void *device;
	read_parallel(t, offset, host, bytes);
	CUDA_CHECK(cudaMalloc(&device, bytes));
	CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
	arena->blocks.push_back(device);
	return device;
}

static void *upload(DeviceArena *arena, const Tensor *t)
{
	return upload_range(arena, t, 0, t->bytes);
}

static void *upload_layer(int layer, const char *suffix, int dtype, int64_t rows, int64_t columns)
{
	const Tensor *t = find_layer(layer, suffix);
	uint64_t element = dtype == DT_F32 ? 4u : dtype == DT_BF16 ? 2u : 1u;
	if ( t->dtype != dtype || (uint64_t)rows * (uint64_t)columns * element != t->bytes )
		die("tensor geometry", t->name);
	return upload(&layer_arena, t);
}

static void release_arena(DeviceArena *arena)
{
	size_t i;
	for ( i = 0; i < arena->blocks.size(); i++ )
		CUDA_CHECK(cudaFree(arena->blocks[i]));
	arena->blocks.clear();
}

static void *device_alloc(uint64_t bytes)
{
	void *p;
	CUDA_CHECK(cudaMalloc(&p, bytes));
	CUDA_CHECK(cudaMemset(p, 0, bytes));
	return p;
}

static __device__ float block_sum(float value, float *scratch)
{
	int lane = threadIdx.x & 31;
	int warp = threadIdx.x >> 5;
	int offset;
	for ( offset = 16; offset > 0; offset >>= 1 )
		value += __shfl_xor_sync(0xffffffffu, value, offset);
	__syncthreads();
	if ( lane == 0 )
		scratch[warp] = value;
	__syncthreads();
	if ( threadIdx.x == 0 )
	{
		float total = 0.0f;
		for ( offset = 0; offset < (int)(blockDim.x >> 5); offset++ )
			total += scratch[offset];
		scratch[0] = total;
	}
	__syncthreads();
	value = scratch[0];
	__syncthreads();
	return value;
}

static __device__ float bf16_at(const uint16_t *p, uint64_t i)
{
	return __uint_as_float((uint32_t)p[i] << 16);
}

static __device__ uint16_t to_bf16(float v)
{
	return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

static __device__ float sigmoid_d(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static __device__ float fp8_weight(const uint8_t *payload, const uint8_t *scale, uint64_t row, uint32_t column, uint32_t columns)
{
	return SparkLmDecodeE4m3(payload[row * columns + column]) * SparkLmDecodeE8m0(scale[row * (columns / GROUP) + column / GROUP]);
}

__global__ void embed_kernel(const uint16_t *embedded, float *streams)
{
	int r = blockIdx.x;
	int i;
	for ( i = threadIdx.x; i < HIDDEN; i += blockDim.x )
	{
		float v = bf16_at(embedded, (uint64_t)r * HIDDEN + (uint64_t)i);
		int m;
		for ( m = 0; m < HC; m++ )
			streams[((uint64_t)r * HC + m) * HIDDEN + i] = v;
	}
}

__global__ void hc_pre_kernel(const float *streams, const float *fn, const float *scale, const float *base, int mix_count, float *post, const uint16_t *gain, uint16_t *out)
{
	__shared__ float scratch[32];
	__shared__ float reduced[HIDDEN];
	__shared__ float pre[HC];
	__shared__ float mixes[2 * HC];
	int r = blockIdx.x;
	const float *s = streams + (uint64_t)r * HC_FLAT;
	float sum = 0.0f;
	float inv;
	int i;
	int m;
	for ( i = threadIdx.x; i < HC_FLAT; i += blockDim.x )
		sum += s[i] * s[i];
	sum = block_sum(sum, scratch);
	inv = rsqrtf(sum / (float)HC_FLAT + RMS_EPS);
	for ( m = 0; m < mix_count; m++ )
	{
		float acc = 0.0f;
		for ( i = threadIdx.x; i < HC_FLAT; i += blockDim.x )
			acc += fn[(uint64_t)m * HC_FLAT + i] * s[i];
		acc = block_sum(acc, scratch);
		if ( threadIdx.x == 0 )
			mixes[m] = acc * inv;
	}
	__syncthreads();
	if ( threadIdx.x < HC )
	{
		pre[threadIdx.x] = sigmoid_d(mixes[threadIdx.x] * scale[0] + base[threadIdx.x]) + HC_EPS;
		if ( post != 0 )
			post[r * HC + threadIdx.x] = HC_MAGNITUDE * sigmoid_d(mixes[HC + threadIdx.x] * scale[1] + base[HC + threadIdx.x]) + HC_EPS;
	}
	__syncthreads();
	sum = 0.0f;
	for ( i = threadIdx.x; i < HIDDEN; i += blockDim.x )
	{
		float acc = 0.0f;
		for ( m = 0; m < HC; m++ )
			acc += pre[m] * s[(uint64_t)m * HIDDEN + i];
		reduced[i] = acc;
		sum += acc * acc;
	}
	sum = block_sum(sum, scratch);
	inv = rsqrtf(sum / (float)HIDDEN + RMS_EPS);
	for ( i = threadIdx.x; i < HIDDEN; i += blockDim.x )
		out[(uint64_t)r * HIDDEN + i] = to_bf16(reduced[i] * inv * bf16_at(gain, i));
}

__global__ void hc_post_kernel(float *streams, const float *post, const float *branch)
{
	int r = blockIdx.x;
	int i;
	for ( i = threadIdx.x; i < HIDDEN; i += blockDim.x )
	{
		float b = branch[(uint64_t)r * HIDDEN + i];
		int m;
		for ( m = 0; m < HC; m++ )
			streams[((uint64_t)r * HC + m) * HIDDEN + i] += post[r * HC + m] * b;
	}
}

__global__ void bf16_to_f32_kernel(const uint16_t *in, float *out, uint64_t count)
{
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < count )
		out[i] = bf16_at(in, i);
}

__global__ void kv_write_kernel(const uint16_t *ckv, const uint16_t *gain, const int *seqs, const int *positions, float *cache)
{
	__shared__ float scratch[32];
	int r = blockIdx.x;
	const uint16_t *c = ckv + (uint64_t)r * KV_A_ROWS;
	float *dst = cache + ((uint64_t)seqs[r] * MAX_POS + positions[r]) * KV_A_ROWS;
	float sum = 0.0f;
	float inv;
	int i;
	for ( i = threadIdx.x; i < KV_LORA; i += blockDim.x )
		sum += bf16_at(c, i) * bf16_at(c, i);
	sum = block_sum(sum, scratch);
	inv = rsqrtf(sum / (float)KV_LORA + RMS_EPS);
	for ( i = threadIdx.x; i < KV_LORA; i += blockDim.x )
		dst[i] = bf16_at(c, i) * inv * bf16_at(gain, i);
	if ( threadIdx.x < QK_ROPE / 2 )
	{
		int half = QK_ROPE / 2;
		double inv_freq = 1.0 / pow(ROPE_THETA, (double)(2 * threadIdx.x) / (double)QK_ROPE);
		float angle = (float)((double)positions[r] * inv_freq);
		float cs = cosf(angle);
		float sn = sinf(angle);
		float a = bf16_at(c, KV_LORA + threadIdx.x);
		float b = bf16_at(c, KV_LORA + threadIdx.x + half);
		dst[KV_LORA + threadIdx.x] = a * cs - b * sn;
		dst[KV_LORA + threadIdx.x + half] = b * cs + a * sn;
	}
}

__global__ void attention_kernel(const uint16_t *q, const uint16_t *gate, const uint8_t *kv_b, const uint8_t *kv_b_scale, const float *sinks, const float *cache, const int *seqs, const int *positions, uint16_t *out)
{
	__shared__ float query[QK_HEAD];
	__shared__ float absorbed[KV_LORA];
	__shared__ float context[KV_LORA];
	__shared__ float scores[MAX_POS];
	__shared__ float denominator;
	int r = blockIdx.x / HEADS;
	int h = blockIdx.x % HEADS;
	int span = positions[r] + 1;
	int lane = threadIdx.x & 31;
	int warp = threadIdx.x >> 5;
	int warps = blockDim.x >> 5;
	const float *seq_cache = cache + (uint64_t)seqs[r] * MAX_POS * KV_A_ROWS;
	uint64_t head_row = (uint64_t)h * KV_B_ROWS_PER_HEAD;
	int i;
	int t;
	for ( i = threadIdx.x; i < QK_HEAD; i += blockDim.x )
		query[i] = bf16_at(q, ((uint64_t)r * HEADS + h) * QK_HEAD + i);
	__syncthreads();
	if ( threadIdx.x < QK_ROPE / 2 )
	{
		int half = QK_ROPE / 2;
		double inv_freq = 1.0 / pow(ROPE_THETA, (double)(2 * threadIdx.x) / (double)QK_ROPE);
		float angle = (float)((double)positions[r] * inv_freq);
		float cs = cosf(angle);
		float sn = sinf(angle);
		float a = query[QK_NOPE + threadIdx.x];
		float b = query[QK_NOPE + threadIdx.x + half];
		__syncwarp();
		query[QK_NOPE + threadIdx.x] = a * cs - b * sn;
		query[QK_NOPE + threadIdx.x + half] = b * cs + a * sn;
	}
	for ( i = threadIdx.x; i < KV_LORA; i += blockDim.x )
	{
		float acc = 0.0f;
		int n;
		for ( n = 0; n < QK_NOPE; n++ )
			acc += query[n] * fp8_weight(kv_b, kv_b_scale, head_row + n, i, KV_LORA);
		absorbed[i] = acc;
	}
	__syncthreads();
	for ( t = warp; t < span; t += warps )
	{
		const float *entry = seq_cache + (uint64_t)t * KV_A_ROWS;
		float acc = 0.0f;
		for ( i = lane; i < KV_LORA; i += 32 )
			acc += absorbed[i] * entry[i];
		for ( i = lane; i < QK_ROPE; i += 32 )
			acc += query[QK_NOPE + i] * entry[KV_LORA + i];
		for ( i = 16; i > 0; i >>= 1 )
			acc += __shfl_xor_sync(0xffffffffu, acc, i);
		if ( lane == 0 )
			scores[t] = acc * rsqrtf((float)QK_HEAD);
	}
	__syncthreads();
	if ( threadIdx.x == 0 )
	{
		float ceiling = sinks[h];
		float total;
		for ( t = 0; t < span; t++ )
			ceiling = fmaxf(ceiling, scores[t]);
		total = expf(sinks[h] - ceiling);
		for ( t = 0; t < span; t++ )
		{
			scores[t] = expf(scores[t] - ceiling);
			total += scores[t];
		}
		denominator = total;
	}
	__syncthreads();
	for ( i = threadIdx.x; i < KV_LORA; i += blockDim.x )
	{
		float acc = 0.0f;
		for ( t = 0; t < span; t++ )
			acc += scores[t] * seq_cache[(uint64_t)t * KV_A_ROWS + i];
		context[i] = acc / denominator;
	}
	__syncthreads();
	for ( i = warp; i < V_HEAD; i += warps )
	{
		float acc = 0.0f;
		int n;
		for ( n = lane; n < KV_LORA; n += 32 )
			acc += fp8_weight(kv_b, kv_b_scale, head_row + QK_NOPE + i, n, KV_LORA) * context[n];
		for ( n = 16; n > 0; n >>= 1 )
			acc += __shfl_xor_sync(0xffffffffu, acc, n);
		if ( lane == 0 )
		{
			uint64_t index = ((uint64_t)r * HEADS + h) * V_HEAD + i;
			out[index] = to_bf16(acc * sigmoid_d(bf16_at(gate, index)));
		}
	}
}

__global__ void swiglu_kernel(const uint16_t *gate, const uint16_t *up, uint64_t gate_stride, uint16_t *out, int inter, int clamp)
{
	int r = blockIdx.x;
	int i;
	for ( i = threadIdx.x; i < inter; i += blockDim.x )
	{
		float g = bf16_at(gate, (uint64_t)r * gate_stride + i);
		float u = bf16_at(up, (uint64_t)r * gate_stride + i);
		if ( clamp )
		{
			g = fminf(g, SWIGLU_LIMIT);
			u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
		}
		out[(uint64_t)r * inter + i] = to_bf16(g / (1.0f + expf(-g)) * u);
	}
}

__global__ void dot_rows_bf16_kernel(const uint16_t *x, const uint16_t *weight, float *out, int rows, int outputs)
{
	int lane = threadIdx.x & 31;
	uint64_t item = (uint64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
	int r;
	int o;
	float acc = 0.0f;
	int i;
	if ( item >= (uint64_t)rows * outputs )
		return;
	r = (int)(item / outputs);
	o = (int)(item % outputs);
	for ( i = lane; i < HIDDEN; i += 32 )
		acc += bf16_at(x, (uint64_t)r * HIDDEN + i) * bf16_at(weight, (uint64_t)o * HIDDEN + i);
	for ( i = 16; i > 0; i >>= 1 )
		acc += __shfl_xor_sync(0xffffffffu, acc, i);
	if ( lane == 0 )
		out[(uint64_t)r * outputs + o] = acc;
}

__global__ void gather_kernel(const uint16_t *x, const int *map, uint16_t *out)
{
	int m = blockIdx.x;
	int i;
	for ( i = threadIdx.x; i < HIDDEN; i += blockDim.x )
		out[(uint64_t)m * HIDDEN + i] = x[(uint64_t)map[m] * HIDDEN + i];
}

__global__ void scatter_add_kernel(const uint16_t *y, const int *map, const float *weights, float *accum)
{
	int m = blockIdx.x;
	int i;
	for ( i = threadIdx.x; i < HIDDEN; i += blockDim.x )
		accum[(uint64_t)map[m] * HIDDEN + i] += weights[m] * bf16_at(y, (uint64_t)m * HIDDEN + i);
}

static void linear(uint32_t format, const void *weight, const void *scale, const void *input, void *output, int rows, int columns, int outputs)
{
	dim3 grid((unsigned)rows, (unsigned)((outputs + SPARK_LM_CTA_WARPS - 1) / SPARK_LM_CTA_WARPS));
	SparkLmLinearKernel<32u, SPARK_ACTIVATION_CODEC_NONE, SPARK_LM_CTA_WARPS><<<grid, SPARK_LM_CTA_THREADS, (size_t)columns * sizeof(float)>>>(format, weight, scale, input, output, (uint32_t)rows, (uint32_t)columns, (uint32_t)outputs);
	CUDA_CHECK(cudaGetLastError());
}

static void linear_fp8_layer(int layer, const char *stem, const void *input, void *output, int rows, int columns, int outputs)
{
	char name[160];
	void *weight;
	void *scale;
	snprintf(name, sizeof(name), "%s.weight", stem);
	weight = upload_layer(layer, name, DT_F8, outputs, columns);
	snprintf(name, sizeof(name), "%s.weight_scale", stem);
	scale = upload_layer(layer, name, DT_U8, outputs, columns / GROUP);
	linear(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3, weight, scale, input, output, rows, columns, outputs);
}

static void rms_norm_rows(void *data, const void *gain, int rows, int dimension)
{
	SparkLmRmsNormKernel<<<rows, SPARK_LM_CTA_THREADS, (size_t)dimension * sizeof(float)>>>(data, gain, data, (uint32_t)rows, (uint32_t)dimension, RMS_EPS);
	CUDA_CHECK(cudaGetLastError());
}

struct Work
{
	float *streams;
	float *post;
	uint16_t *x;
	float *branch;
	uint16_t *gate;
	uint16_t *qa;
	uint16_t *q;
	uint16_t *ckv;
	uint16_t *heads;
	uint16_t *wide_a;
	uint16_t *wide_b;
	uint16_t *act;
	uint16_t *expert_x;
	uint16_t *expert_gu;
	uint16_t *expert_act;
	uint16_t *expert_y;
	float *router;
	float *route_weights;
	int *route_map;
	int *tokens;
	int *seqs;
	int *positions;
	float *cache;
	float *head_logits;
};

static Work work;

static void allocate_work(void)
{
	work.streams = (float *)device_alloc((uint64_t)MAX_ROWS * HC_FLAT * 4u);
	work.post = (float *)device_alloc((uint64_t)MAX_ROWS * HC * 4u);
	work.x = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * HIDDEN * 2u);
	work.branch = (float *)device_alloc((uint64_t)MAX_ROWS * HIDDEN * 4u);
	work.gate = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * HEADS * V_HEAD * 2u);
	work.qa = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * Q_LORA * 2u);
	work.q = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * HEADS * QK_HEAD * 2u);
	work.ckv = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * KV_A_ROWS * 2u);
	work.heads = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * HEADS * V_HEAD * 2u);
	work.wide_a = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * DENSE_INTER * 2u);
	work.wide_b = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * DENSE_INTER * 2u);
	work.act = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * DENSE_INTER * 2u);
	work.expert_x = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * HIDDEN * 2u);
	work.expert_gu = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * 2u * MOE_INTER * 2u);
	work.expert_act = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * MOE_INTER * 2u);
	work.expert_y = (uint16_t *)device_alloc((uint64_t)MAX_ROWS * HIDDEN * 2u);
	work.router = (float *)device_alloc((uint64_t)MAX_ROWS * EXPERTS * 4u);
	work.route_weights = (float *)device_alloc((uint64_t)MAX_ROWS * 4u);
	work.route_map = (int *)device_alloc((uint64_t)MAX_ROWS * 4u);
	work.tokens = (int *)device_alloc((uint64_t)MAX_ROWS * 4u);
	work.seqs = (int *)device_alloc((uint64_t)MAX_ROWS * 4u);
	work.positions = (int *)device_alloc((uint64_t)MAX_ROWS * 4u);
	work.cache = (float *)device_alloc((uint64_t)LAYERS * MAX_SEQ * MAX_POS * KV_A_ROWS * 4u);
	work.head_logits = (float *)device_alloc((uint64_t)MAX_SEQ * VOCAB * 4u);
}

static void hc_stage(int layer, const char *kind, const char *norm, int count)
{
	char name[160];
	void *fn;
	void *scale;
	void *base;
	void *gain;
	snprintf(name, sizeof(name), "%s.hc_pre.hc_fn", kind);
	fn = upload_layer(layer, name, DT_F32, 2 * HC, HC_FLAT);
	snprintf(name, sizeof(name), "%s.hc_pre.hc_scale", kind);
	scale = upload_layer(layer, name, DT_F32, 1, 2);
	snprintf(name, sizeof(name), "%s.hc_pre.hc_base", kind);
	base = upload_layer(layer, name, DT_F32, 1, 2 * HC);
	gain = upload_layer(layer, norm, DT_BF16, 1, HIDDEN);
	hc_pre_kernel<<<count, THREADS>>>(work.streams, (const float *)fn, (const float *)scale, (const float *)base, 2 * HC, work.post, (const uint16_t *)gain, work.x);
	CUDA_CHECK(cudaGetLastError());
}

static void attention(int layer, int count)
{
	void *gate_weight = upload_layer(layer, "self_attn.linear_gate.weight", DT_BF16, HEADS * V_HEAD, HIDDEN);
	void *qa_gain = upload_layer(layer, "self_attn.q_a_layernorm.weight", DT_BF16, 1, Q_LORA);
	void *kv_gain = upload_layer(layer, "self_attn.kv_a_layernorm.weight", DT_BF16, 1, KV_LORA);
	void *kv_b = upload_layer(layer, "self_attn.kv_b_proj.weight", DT_F8, HEADS * KV_B_ROWS_PER_HEAD, KV_LORA);
	void *kv_b_scale = upload_layer(layer, "self_attn.kv_b_proj.weight_scale", DT_U8, HEADS * KV_B_ROWS_PER_HEAD, KV_LORA / GROUP);
	void *sinks = upload_layer(layer, "self_attn.learnable_sink_param", DT_F32, 1, HEADS);
	uint64_t count64 = (uint64_t)count * HIDDEN;
	linear(SPARK_LM_WEIGHT_FORMAT_BF16, gate_weight, 0, work.x, work.gate, count, HIDDEN, HEADS * V_HEAD);
	linear_fp8_layer(layer, "self_attn.q_a_proj", work.x, work.qa, count, HIDDEN, Q_LORA);
	rms_norm_rows(work.qa, qa_gain, count, Q_LORA);
	linear_fp8_layer(layer, "self_attn.q_b_proj", work.qa, work.q, count, Q_LORA, HEADS * QK_HEAD);
	linear_fp8_layer(layer, "self_attn.kv_a_proj_with_mqa", work.x, work.ckv, count, HIDDEN, KV_A_ROWS);
	kv_write_kernel<<<count, THREADS>>>(work.ckv, (const uint16_t *)kv_gain, work.seqs, work.positions, work.cache + (uint64_t)layer * MAX_SEQ * MAX_POS * KV_A_ROWS);
	CUDA_CHECK(cudaGetLastError());
	attention_kernel<<<count * HEADS, THREADS>>>(work.q, work.gate, (const uint8_t *)kv_b, (const uint8_t *)kv_b_scale, (const float *)sinks, work.cache + (uint64_t)layer * MAX_SEQ * MAX_POS * KV_A_ROWS, work.seqs, work.positions, work.heads);
	CUDA_CHECK(cudaGetLastError());
	linear_fp8_layer(layer, "self_attn.o_proj", work.heads, work.expert_y, count, HEADS * V_HEAD, HIDDEN);
	bf16_to_f32_kernel<<<(unsigned)((count64 + THREADS - 1) / THREADS), THREADS>>>(work.expert_y, work.branch, count64);
	CUDA_CHECK(cudaGetLastError());
}

static void dense_mlp(int layer, const char *stem, int count, int inter)
{
	char name[160];
	uint64_t count64 = (uint64_t)count * HIDDEN;
	snprintf(name, sizeof(name), "%s.gate_proj", stem);
	linear_fp8_layer(layer, name, work.x, work.wide_a, count, HIDDEN, inter);
	snprintf(name, sizeof(name), "%s.up_proj", stem);
	linear_fp8_layer(layer, name, work.x, work.wide_b, count, HIDDEN, inter);
	swiglu_kernel<<<count, THREADS>>>(work.wide_a, work.wide_b, (uint64_t)inter, work.act, inter, 0);
	CUDA_CHECK(cudaGetLastError());
	snprintf(name, sizeof(name), "%s.down_proj", stem);
	linear_fp8_layer(layer, name, work.act, work.expert_y, count, inter, HIDDEN);
	bf16_to_f32_kernel<<<(unsigned)((count64 + THREADS - 1) / THREADS), THREADS>>>(work.expert_y, work.branch, count64);
	CUDA_CHECK(cudaGetLastError());
}

static void moe(int layer, const Row *rows, int count)
{
	void *router_weight = upload_layer(layer, "mlp.gate.weight", DT_BF16, EXPERTS, HIDDEN);
	const Tensor *bias_tensor = find_layer(layer, "mlp.gate.e_score_correction_bias");
	const Tensor *gate_up = find_layer(layer, "mlp.experts.gate_up_proj");
	const Tensor *gate_up_scale = find_layer(layer, "mlp.experts.gate_up_proj_scale");
	const Tensor *down = find_layer(layer, "mlp.experts.down_proj");
	const Tensor *down_scale = find_layer(layer, "mlp.experts.down_proj_scale");
	const uint64_t gu_bytes = 2ull * MOE_INTER * HIDDEN;
	const uint64_t gu_scale_bytes = gu_bytes / GROUP;
	const uint64_t down_bytes = (uint64_t)HIDDEN * MOE_INTER;
	const uint64_t down_scale_bytes = down_bytes / GROUP;
	std::vector<float> logits((size_t)count * EXPERTS);
	std::vector<float> bias(EXPERTS);
	std::vector<int> selected((size_t)count * TOP_K);
	std::vector<float> weights((size_t)count * TOP_K);
	std::vector<int> used;
	unsigned warps_per_block = THREADS / 32;
	uint64_t items = (uint64_t)count * EXPERTS;
	int r;
	int e;
	size_t batch;
	if ( bias_tensor->dtype != DT_F32 || bias_tensor->bytes != EXPERTS * 4u || gate_up->bytes != EXPERTS * gu_bytes || down->bytes != EXPERTS * down_bytes )
		die("moe geometry", gate_up->name);
	read_exact(bias_tensor, 0, bias.data(), EXPERTS * 4u);
	dot_rows_bf16_kernel<<<(unsigned)((items + warps_per_block - 1) / warps_per_block), THREADS>>>(work.x, (const uint16_t *)router_weight, work.router, count, EXPERTS);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(logits.data(), work.router, logits.size() * 4u, cudaMemcpyDeviceToHost));
	for ( r = 0; r < count; r++ )
	{
		float scores[EXPERTS];
		float choice[EXPERTS];
		float total = 0.0f;
		int k;
		for ( e = 0; e < EXPERTS; e++ )
		{
			scores[e] = 1.0f / (1.0f + expf(-logits[(size_t)r * EXPERTS + e]));
			choice[e] = scores[e] + bias[e];
		}
		for ( k = 0; k < TOP_K; k++ )
		{
			int best = -1;
			for ( e = 0; e < EXPERTS; e++ )
				if ( choice[e] != -INFINITY && (best < 0 || choice[e] > choice[best]) )
					best = e;
			selected[(size_t)r * TOP_K + k] = best;
			weights[(size_t)r * TOP_K + k] = scores[best];
			total += scores[best];
			choice[best] = -INFINITY;
		}
		for ( k = 0; k < TOP_K; k++ )
		{
			Sequence *s = &sequences[rows[r].seq];
			size_t slot = ((size_t)rows[r].pos * (LAYERS - 1) + (size_t)(layer - 1)) * TOP_K + k;
			weights[(size_t)r * TOP_K + k] = weights[(size_t)r * TOP_K + k] / (total + 1e-20f) * ROUTE_SCALE;
			s->route_ids[slot] = selected[(size_t)r * TOP_K + k];
			s->route_weights[slot] = weights[(size_t)r * TOP_K + k];
		}
	}
	for ( e = 0; e < EXPERTS; e++ )
		for ( r = 0; r < count * TOP_K; r++ )
			if ( selected[r] == e )
			{
				used.push_back(e);
				break;
			}
	dense_mlp(layer, "mlp.shared_experts", count, MOE_INTER);
	for ( batch = 0; batch < used.size(); batch += EXPERT_BATCH )
	{
		size_t take = used.size() - batch < EXPERT_BATCH ? used.size() - batch : EXPERT_BATCH;
		DeviceArena expert_arena = { std::vector<void *>(), 0, 0 };
		std::vector<void *> device_gu(take);
		std::vector<void *> device_gu_scale(take);
		std::vector<void *> device_down(take);
		std::vector<void *> device_down_scale(take);
		size_t j;
		for ( j = 0; j < take; j++ )
		{
			int expert = used[batch + j];
			device_gu[j] = upload_range(&expert_arena, gate_up, (uint64_t)expert * gu_bytes, gu_bytes);
			device_gu_scale[j] = upload_range(&expert_arena, gate_up_scale, (uint64_t)expert * gu_scale_bytes, gu_scale_bytes);
			device_down[j] = upload_range(&expert_arena, down, (uint64_t)expert * down_bytes, down_bytes);
			device_down_scale[j] = upload_range(&expert_arena, down_scale, (uint64_t)expert * down_scale_bytes, down_scale_bytes);
		}
		for ( j = 0; j < take; j++ )
		{
			int expert = used[batch + j];
			std::vector<int> map;
			std::vector<float> member_weights;
			for ( r = 0; r < count; r++ )
			{
				int k;
				for ( k = 0; k < TOP_K; k++ )
					if ( selected[(size_t)r * TOP_K + k] == expert )
					{
						map.push_back(r);
						member_weights.push_back(weights[(size_t)r * TOP_K + k]);
					}
			}
			CUDA_CHECK(cudaMemcpy(work.route_map, map.data(), map.size() * 4u, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(work.route_weights, member_weights.data(), member_weights.size() * 4u, cudaMemcpyHostToDevice));
			gather_kernel<<<(unsigned)map.size(), THREADS>>>(work.x, work.route_map, work.expert_x);
			CUDA_CHECK(cudaGetLastError());
			linear(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3, device_gu[j], device_gu_scale[j], work.expert_x, work.expert_gu, (int)map.size(), HIDDEN, 2 * MOE_INTER);
			swiglu_kernel<<<(unsigned)map.size(), THREADS>>>(work.expert_gu, work.expert_gu + MOE_INTER, 2ull * MOE_INTER, work.expert_act, MOE_INTER, 1);
			CUDA_CHECK(cudaGetLastError());
			linear(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3, device_down[j], device_down_scale[j], work.expert_act, work.expert_y, (int)map.size(), MOE_INTER, HIDDEN);
			scatter_add_kernel<<<(unsigned)map.size(), THREADS>>>(work.expert_y, work.route_map, work.route_weights, work.branch);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		release_arena(&expert_arena);
		free(expert_arena.host);
	}
}

static void capture(int layer, const Row *rows, int count)
{
	std::vector<float> host;
	int r;
	int needed = 0;
	for ( r = 0; r < count; r++ )
	{
		Sequence *s = &sequences[rows[r].seq];
		int c;
		for ( c = 0; c < s->capture_count; c++ )
			if ( s->capture_layers[c] == layer )
				needed = 1;
	}
	if ( !needed )
		return;
	host.resize((size_t)count * HC_FLAT);
	CUDA_CHECK(cudaMemcpy(host.data(), work.streams, host.size() * 4u, cudaMemcpyDeviceToHost));
	for ( r = 0; r < count; r++ )
	{
		Sequence *s = &sequences[rows[r].seq];
		int c;
		for ( c = 0; c < s->capture_count; c++ )
			if ( s->capture_layers[c] == layer )
				memcpy(s->streams.data() + ((size_t)c * MAX_POS + (size_t)rows[r].pos) * HC_FLAT, host.data() + (size_t)r * HC_FLAT, HC_FLAT * 4u);
	}
}

static void head(const Row *rows, int count, const int *wanted)
{
	DeviceArena head_arena = { std::vector<void *>(), 0, 0 };
	void *fn = upload(&head_arena, find_tensor("model.hc_head.hc_head_fn"));
	void *scale = upload(&head_arena, find_tensor("model.hc_head.hc_head_scale"));
	void *base = upload(&head_arena, find_tensor("model.hc_head.hc_head_base"));
	void *gain = upload(&head_arena, find_tensor("model.norm.weight"));
	void *lm_head = upload(&head_arena, find_tensor("lm_head.weight"));
	std::vector<float> logits((size_t)VOCAB);
	unsigned warps_per_block = THREADS / 32;
	int r;
	hc_pre_kernel<<<count, THREADS>>>(work.streams, (const float *)fn, (const float *)scale, (const float *)base, HC, 0, (const uint16_t *)gain, work.x);
	CUDA_CHECK(cudaGetLastError());
	for ( r = 0; r < count; r++ )
	{
		Sequence *s = &sequences[rows[r].seq];
		int top[HEAD_TOP];
		int k;
		if ( !wanted[r] )
			continue;
		dot_rows_bf16_kernel<<<(unsigned)((VOCAB + warps_per_block - 1) / warps_per_block), THREADS>>>(work.x + (uint64_t)r * HIDDEN, (const uint16_t *)lm_head, work.head_logits, 1, VOCAB);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaMemcpy(logits.data(), work.head_logits, (size_t)VOCAB * 4u, cudaMemcpyDeviceToHost));
		for ( k = 0; k < HEAD_TOP; k++ )
		{
			int best = -1;
			int v;
			for ( v = 0; v < VOCAB; v++ )
			{
				int j;
				int taken = 0;
				for ( j = 0; j < k; j++ )
					if ( top[j] == v )
						taken = 1;
				if ( !taken && (best < 0 || logits[v] > logits[best]) )
					best = v;
			}
			top[k] = best;
			s->head_top_token[rows[r].pos][k] = best;
			s->head_top_score[rows[r].pos][k] = logits[best];
		}
		s->head_token[rows[r].pos] = top[0];
	}
	release_arena(&head_arena);
	free(head_arena.host);
}

static void forward(const Row *rows, int count, const int *wanted)
{
	std::vector<int> tokens(count);
	std::vector<int> seqs(count);
	std::vector<int> positions(count);
	int r;
	int layer;
	for ( r = 0; r < count; r++ )
	{
		if ( rows[r].pos >= INDEX_TOP_K || rows[r].pos >= MAX_POS )
			die("position beyond the harness bound", 0);
		tokens[r] = rows[r].token;
		seqs[r] = rows[r].seq;
		positions[r] = rows[r].pos;
	}
	CUDA_CHECK(cudaMemcpy(work.tokens, tokens.data(), (size_t)count * 4u, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(work.seqs, seqs.data(), (size_t)count * 4u, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(work.positions, positions.data(), (size_t)count * 4u, cudaMemcpyHostToDevice));
	{
		const Tensor *embed = find_tensor("model.embed_tokens.weight");
		std::vector<uint16_t> embedded((size_t)count * HIDDEN);
		if ( embed->dtype != DT_BF16 || embed->bytes != (uint64_t)VOCAB * HIDDEN * 2u )
			die("embedding geometry", embed->name);
		for ( r = 0; r < count; r++ )
			read_exact(embed, (uint64_t)rows[r].token * HIDDEN * 2u, embedded.data() + (size_t)r * HIDDEN, HIDDEN * 2u);
		CUDA_CHECK(cudaMemcpy(work.x, embedded.data(), embedded.size() * 2u, cudaMemcpyHostToDevice));
		embed_kernel<<<count, THREADS>>>(work.x, work.streams);
		CUDA_CHECK(cudaGetLastError());
	}
	for ( layer = 0; layer < layer_limit; layer++ )
	{
		double t0 = now_seconds();
		hc_stage(layer, "hc_attn_layer", "input_layernorm.weight", count);
		attention(layer, count);
		hc_post_kernel<<<count, THREADS>>>(work.streams, work.post, work.branch);
		CUDA_CHECK(cudaGetLastError());
		hc_stage(layer, "hc_mlp_layer", "post_attention_layernorm.weight", count);
		if ( layer == 0 )
			dense_mlp(layer, "mlp", count, DENSE_INTER);
		else
			moe(layer, rows, count);
		hc_post_kernel<<<count, THREADS>>>(work.streams, work.post, work.branch);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		release_arena(&layer_arena);
		capture(layer, rows, count);
		fprintf(stderr, "layer %d rows %d %.1fs elapsed %.0fs\n", layer, count, now_seconds() - t0, now_seconds() - t_start);
	}
	if ( layer_limit == LAYERS )
		head(rows, count, wanted);
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
		s->route_ids.assign((size_t)MAX_POS * (LAYERS - 1) * TOP_K, 0);
		s->route_weights.assign((size_t)MAX_POS * (LAYERS - 1) * TOP_K, 0.0f);
		s->streams.assign((size_t)(s->capture_count ? s->capture_count : 1) * MAX_POS * HC_FLAT, 0.0f);
		sequence_count++;
	}
	fclose(fh);
	if ( sequence_count == 0 )
		die("no prompts in", path);
}

static void load_forced(const char *directory)
{
	int q;
	for ( q = 0; q < sequence_count; q++ )
	{
		Sequence *s = &sequences[q];
		char path[1024];
		char word[32];
		FILE *fh;
		snprintf(path, sizeof(path), "%s/%s.txt", directory, s->name);
		fh = fopen(path, "r");
		if ( fh == 0 )
			die("cannot open forced tokens", path);
		while ( fscanf(fh, "%31s", word) == 1 && strcmp(word, "generated") != 0 )
			;
		while ( s->forced_count < s->new_tokens && fscanf(fh, "%d", &s->forced[s->forced_count]) == 1 )
			s->forced_count++;
		fclose(fh);
		if ( s->forced_count != s->new_tokens )
			die("forced token count disagrees with new_tokens for", s->name);
	}
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
		fwrite(s->route_ids.data(), sizeof(int32_t), (size_t)total * (LAYERS - 1) * TOP_K, fh);
		fwrite(s->route_weights.data(), sizeof(float), (size_t)total * (LAYERS - 1) * TOP_K, fh);
		fclose(fh);
		snprintf(path, sizeof(path), "%s/%s.streams.bin", directory, s->name);
		fh = fopen(path, "wb");
		if ( fh == 0 )
			die("cannot write", path);
		for ( k = 0; k < s->capture_count; k++ )
			fwrite(s->streams.data() + (size_t)k * MAX_POS * HC_FLAT, sizeof(float), (size_t)total * HC_FLAT, fh);
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
	int step_limit;
	if ( argc != 6 && argc != 7 )
	{
		fprintf(stderr, "usage: %s <manifest> <prompts> <out_dir> <layer_count> <decode_steps> [forced_token_dir]\n", argv[0]);
		return 2;
	}
	t_start = now_seconds();
	layer_limit = atoi(argv[4]);
	step_limit = atoi(argv[5]);
	if ( layer_limit < 1 || layer_limit > LAYERS || step_limit < 1 )
		die("bad layer count or decode steps", 0);
	load_manifest(argv[1]);
	load_prompts(argv[2]);
	if ( argc == 7 )
		load_forced(argv[6]);
	CUDA_CHECK(cudaFuncSetAttribute(SparkLmLinearKernel<32u, SPARK_ACTIVATION_CODEC_NONE, SPARK_LM_CTA_WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, DENSE_INTER * (int)sizeof(float)));
	CUDA_CHECK(cudaFuncSetAttribute(SparkLmRmsNormKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, HIDDEN * (int)sizeof(float)));
	allocate_work();
	fprintf(stderr, "tensors %d files %d prompts %d layers %d steps %d\n", tensor_count, file_count, sequence_count, layer_limit, step_limit);
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
	if ( step_limit > max_new )
		step_limit = max_new;
	for ( step = 0; step < step_limit; step++ )
	{
		forward(rows, count, wanted);
		if ( layer_limit != LAYERS )
		{
			write_outputs(argv[3]);
			break;
		}
		count = 0;
		for ( q = 0; q < sequence_count; q++ )
		{
			Sequence *s = &sequences[q];
			int last = s->token_count + s->generated - 1;
			if ( s->generated >= s->new_tokens )
				continue;
			s->tokens[last + 1] = s->forced_count > s->generated ? s->forced[s->generated] : s->head_token[last];
			s->generated++;
			fprintf(stderr, "step %d %s pos %d token %d score %.4f\n", step, s->name, last + 1, s->tokens[last + 1], (double)s->head_top_score[last][0]);
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
