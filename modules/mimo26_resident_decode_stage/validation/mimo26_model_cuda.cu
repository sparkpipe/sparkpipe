#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/kernels/skinny.cuh"
#include "inference/kernels/gqa.cuh"
#include "sparkpipe/spark_mimo26_model.h"

#define M26_HIDDEN SPARK_MIMO26_MODEL_HIDDEN_DIMENSION
#define M26_LAYERS SPARK_MIMO26_MODEL_LAYER_COUNT
#define M26_VOCAB SPARK_MIMO26_MODEL_VOCAB_COUNT
#define M26_HEADS SPARK_MIMO26_MODEL_ATTN_HEAD_COUNT
#define M26_HEAD_DIM SPARK_MIMO26_MODEL_ATTN_HEAD_DIMENSION
#define M26_VALUE_DIM SPARK_MIMO26_MODEL_ATTN_VALUE_DIMENSION
#define M26_ROPE_DIM SPARK_MIMO26_MODEL_ATTN_ROPE_DIMENSION
#define M26_FULL_KV SPARK_MIMO26_MODEL_FULL_KV_HEAD_COUNT
#define M26_SWA_KV SPARK_MIMO26_MODEL_SWA_KV_HEAD_COUNT
#define M26_FULL_QKV SPARK_MIMO26_MODEL_FULL_QKV_DIMENSION
#define M26_SWA_QKV SPARK_MIMO26_MODEL_SWA_QKV_DIMENSION
#define M26_O_INPUT SPARK_MIMO26_MODEL_O_INPUT_DIMENSION
#define M26_EXPERTS SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT
#define M26_TOP_K SPARK_MIMO26_MODEL_EXPERTS_PER_TOKEN
#define M26_INTER SPARK_MIMO26_MODEL_EXPERT_INTERMEDIATE_DIMENSION
#define M26_DENSE SPARK_MIMO26_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define M26_EPS SPARK_MIMO26_MODEL_RMS_NORM_EPSILON
#define M26_VALUE_SCALE SPARK_MIMO26_MODEL_ATTN_VALUE_SCALE
#define M26_WINDOW SPARK_MIMO26_MODEL_SLIDING_WINDOW_TOKENS
#define M26_RANKS 4u
#define M26_THREADS 256u
#define M26_PAGE_SLOTS 64u
#define M26_MAX_POSITIONS 128u
#define M26_PAGES (M26_MAX_POSITIONS / M26_PAGE_SLOTS)
#define M26_SPARE_SLOTS 16u

using M26FullKv = LmKvGeometry<M26_FULL_KV * (M26_HEAD_DIM + M26_VALUE_DIM) * 2u, M26_PAGE_SLOTS, true>;
using M26SwaKv = LmKvGeometry<M26_SWA_KV * (M26_HEAD_DIM + M26_VALUE_DIM) * 2u, M26_PAGE_SLOTS, true>;

typedef struct M26Layer
{
	uint32_t swa,moe,resident,capacity,demand;
	uint16_t *attn_norm,*mlp_norm,*sink,*o_proj,*router;
	uint8_t *qkv,*projection[3],*projection_scale[3];
	float *qkv_scale,*dense_scale[3];
	std::vector<float> bias;
	std::vector<int32_t> slot;
	uint8_t *pool;
	LmKvView cache;
}
M26Layer;

typedef struct M26Buffers
{
	uint16_t *stream,*normed,*fused,*query,*key,*value,*attended,*projected,*gate,*up,*act,*down,*final_norm,*lm_head;
	float *logits,*route_weight;
	uint32_t *sequence,*context,*position,*page_table,*route,*packed;
	LmKvAccessError *error;
}
M26Buffers;

static const char *m26_directory;
static const char *m26_projections[3] = {"gate_proj","up_proj","down_proj"};

static void M26Check(cudaError_t status, const char *what)
{
	if ( status != cudaSuccess )
	{
		fprintf(stderr,"FAIL %s: %s\n",what,cudaGetErrorString(status));
		exit(1);
	}
}

static void M26Require(int32_t status, const char *what, uint32_t layer)
{
	if ( status != LM_LAUNCH_OK )
	{
		fprintf(stderr,"FAIL layer %u %s launch status %d\n",layer,what,status);
		exit(1);
	}
}

static std::vector<uint8_t> M26Read(const char *name, size_t bytes)
{
	char path[1024];
	std::vector<uint8_t> data(bytes);
	FILE *file;
	snprintf(path,sizeof(path),"%s/%s",m26_directory,name);
	file = fopen(path,"rb");
	if ( file == 0 || fread(data.data(),1,bytes,file) != bytes || fgetc(file) != EOF )
	{
		fprintf(stderr,"FAIL %s is missing or is not exactly %zu bytes\n",path,bytes);
		exit(1);
	}
	fclose(file);
	return(data);
}

static size_t M26FileBytes(const char *name)
{
	char path[1024];
	FILE *file;
	long bytes;
	snprintf(path,sizeof(path),"%s/%s",m26_directory,name);
	file = fopen(path,"rb");
	if ( file == 0 || fseek(file,0,SEEK_END) != 0 || (bytes = ftell(file)) < 0 )
	{
		fprintf(stderr,"FAIL %s is missing\n",path);
		exit(1);
	}
	fclose(file);
	return((size_t)bytes);
}

template<class T>
static T *M26Upload(const char *name, size_t bytes)
{
	std::vector<uint8_t> data = M26Read(name,bytes);
	void *device;
	M26Check(cudaMalloc(&device,bytes),name);
	M26Check(cudaMemcpy(device,data.data(),bytes,cudaMemcpyHostToDevice),name);
	return((T *)device);
}

template<class T>
static T *M26Alloc(size_t count)
{
	void *device;
	M26Check(cudaMalloc(&device,count * sizeof(T)),"alloc");
	M26Check(cudaMemset(device,0,count * sizeof(T)),"clear");
	return((T *)device);
}

static float M26Float(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out,&bits,sizeof(out));
	return(out);
}

template<uint32_t KV, uint32_t QKV>
__global__ void M26SplitQkvKernel(const uint16_t *fused, uint16_t *query, uint16_t *key, uint16_t *value, uint32_t position, float theta)
{
	const uint32_t q_local = (M26_HEADS / M26_RANKS) * M26_HEAD_DIM, k_local = (KV / M26_RANKS) * M26_HEAD_DIM, v_local = (KV / M26_RANKS) * M26_VALUE_DIM, per = q_local + k_local + v_local;
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x, rank, local, dim, pair;
	float angle, first, second;
	uint16_t *target;
	if ( row >= QKV )
		return;
	rank = row / per;
	local = row % per;
	if ( local >= q_local + k_local )
	{
		value[rank * v_local + (local - q_local - k_local)] = LmFloatToBf16(LmBf16ToFloat(fused[row]) * M26_VALUE_SCALE);
		return;
	}
	target = local < q_local ? query + rank * q_local : key + rank * k_local;
	local = local < q_local ? local : local - q_local;
	dim = local % M26_HEAD_DIM;
	if ( dim >= M26_ROPE_DIM )
	{
		target[local] = fused[row];
		return;
	}
	pair = dim < M26_ROPE_DIM / 2u ? dim + M26_ROPE_DIM / 2u : dim - M26_ROPE_DIM / 2u;
	angle = (float)position * powf(theta,-(2.0f * (float)(dim % (M26_ROPE_DIM / 2u))) / (float)M26_ROPE_DIM);
	first = LmBf16ToFloat(fused[row]);
	second = LmBf16ToFloat(fused[row - dim + pair]);
	target[local] = LmFloatToBf16(dim < M26_ROPE_DIM / 2u ? first * cosf(angle) - second * sinf(angle) : first * cosf(angle) + second * sinf(angle));
}

__global__ void M26AddKernel(const uint16_t *right, uint16_t *stream, uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		stream[index] = LmFloatToBf16(LmBf16ToFloat(stream[index]) + LmBf16ToFloat(right[index]));
}

__global__ void M26SwigluKernel(const uint16_t *gate, const uint16_t *up, uint16_t *out, uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	float g;
	if ( index >= count )
		return;
	g = LmBf16ToFloat(gate[index]);
	out[index] = LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(g / (1.0f + __expf(-g)))) * LmBf16ToFloat(up[index]));
}

__global__ void M26CombineKernel(const uint16_t *down, const float *weights, uint16_t *stream)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x, k;
	float total = 0.0f;
	if ( index >= M26_HIDDEN )
		return;
	for (k = 0u; k < M26_TOP_K; k++)
		total += weights[k] * LmBf16ToFloat(down[k * M26_HIDDEN + index]);
	stream[index] = LmFloatToBf16(LmBf16ToFloat(stream[index]) + LmBf16ToFloat(LmFloatToBf16(total)));
}

static void M26LoadAttention(uint32_t index, M26Layer *layer, uint32_t *page_table, LmKvAccessError *error)
{
	char name[128];
	uint32_t rows = layer->swa ? M26_SWA_QKV : M26_FULL_QKV;
	size_t slot_bytes = layer->swa ? M26SwaKv::kSlotBytes : M26FullKv::kSlotBytes;
	snprintf(name,sizeof(name),"l%02u_attn_norm.bf16",index);
	layer->attn_norm = M26Upload<uint16_t>(name,M26_HIDDEN * 2u);
	snprintf(name,sizeof(name),"l%02u_mlp_norm.bf16",index);
	layer->mlp_norm = M26Upload<uint16_t>(name,M26_HIDDEN * 2u);
	snprintf(name,sizeof(name),"l%02u_qkv.fp8",index);
	layer->qkv = M26Upload<uint8_t>(name,(size_t)rows * M26_HIDDEN);
	snprintf(name,sizeof(name),"l%02u_qkv_scale_rows.f32",index);
	layer->qkv_scale = M26Upload<float>(name,(size_t)rows * (M26_HIDDEN / 128u) * 4u);
	snprintf(name,sizeof(name),"l%02u_o_proj.bf16",index);
	layer->o_proj = M26Upload<uint16_t>(name,(size_t)M26_HIDDEN * M26_O_INPUT * 2u);
	if ( layer->swa )
	{
		snprintf(name,sizeof(name),"l%02u_sink.bf16",index);
		layer->sink = M26Upload<uint16_t>(name,M26_HEADS * 2u);
	}
	layer->pool = M26Alloc<uint8_t>((size_t)M26_MAX_POSITIONS * slot_bytes);
	if ( LmKvViewInitialize(&layer->cache,layer->pool,page_table,M26_PAGES,1u,M26_PAGES,error) != 0 )
	{
		fprintf(stderr,"FAIL layer %u kv view\n",index);
		exit(1);
	}
}

static void M26LoadMlp(uint32_t index, M26Layer *layer)
{
	char name[128];
	uint32_t p, rows, columns;
	std::vector<uint8_t> raw;
	for (p = 0u; p < 3u; p++)
	{
		rows = p == 2u ? M26_HIDDEN : (layer->moe ? M26_INTER : M26_DENSE);
		columns = p == 2u ? (layer->moe ? M26_INTER : M26_DENSE) : M26_HIDDEN;
		if ( !layer->moe )
		{
			snprintf(name,sizeof(name),"l%02u_dense_%s.fp8",index,m26_projections[p]);
			layer->projection[p] = M26Upload<uint8_t>(name,(size_t)rows * columns);
			snprintf(name,sizeof(name),"l%02u_dense_%s_scale_rows.f32",index,m26_projections[p]);
			layer->dense_scale[p] = M26Upload<float>(name,(size_t)rows * (columns / 128u) * 4u);
			continue;
		}
		layer->projection[p] = M26Alloc<uint8_t>((size_t)layer->capacity * rows * columns / 2u);
		layer->projection_scale[p] = M26Alloc<uint8_t>((size_t)layer->capacity * rows * columns / 32u);
	}
	if ( !layer->moe )
		return;
	snprintf(name,sizeof(name),"l%02u_router.bf16",index);
	layer->router = M26Upload<uint16_t>(name,(size_t)M26_EXPERTS * M26_HIDDEN * 2u);
	snprintf(name,sizeof(name),"l%02u_router_bias.f32",index);
	raw = M26Read(name,M26_EXPERTS * 4u);
	layer->bias.assign((const float *)raw.data(),(const float *)raw.data() + M26_EXPERTS);
}

static void M26ReadAt(const char *name, size_t offset, void *out, size_t bytes)
{
	char path[1024];
	FILE *file;
	snprintf(path,sizeof(path),"%s/%s",m26_directory,name);
	file = fopen(path,"rb");
	if ( file == 0 || fseek(file,(long)offset,SEEK_SET) != 0 || fread(out,1,bytes,file) != bytes )
	{
		fprintf(stderr,"FAIL %s has no %zu bytes at %zu\n",path,bytes,offset);
		exit(1);
	}
	fclose(file);
}

static void M26LoadExpert(uint32_t index, M26Layer *layer, uint32_t expert, uint32_t slot)
{
	char name[128];
	uint32_t p;
	size_t payload = (size_t)M26_INTER * M26_HIDDEN / 2u, scale = (size_t)M26_INTER * M26_HIDDEN / 32u;
	std::vector<uint8_t> host(payload);
	for (p = 0u; p < 3u; p++)
	{
		snprintf(name,sizeof(name),"l%02u_%s.mxfp4",index,m26_projections[p]);
		M26ReadAt(name,(size_t)expert * payload,host.data(),payload);
		M26Check(cudaMemcpy(layer->projection[p] + (size_t)slot * payload,host.data(),payload,cudaMemcpyHostToDevice),"expert payload");
		snprintf(name,sizeof(name),"l%02u_%s.ue8m0",index,m26_projections[p]);
		M26ReadAt(name,(size_t)expert * scale,host.data(),scale);
		M26Check(cudaMemcpy(layer->projection_scale[p] + (size_t)slot * scale,host.data(),scale,cudaMemcpyHostToDevice),"expert scale");
	}
	layer->slot[expert] = (int32_t)slot;
}

static void M26LoadLayer(uint32_t index, M26Layer *layer, uint32_t *page_table, LmKvAccessError *error)
{
	char name[128];
	std::vector<uint8_t> raw;
	uint32_t e, count = 0u;
	layer->swa = SparkMimo26ModelLayerKind(index) == SPARK_MIMO26_MODEL_LAYER_KIND_SWA ? 1u : 0u;
	layer->moe = SparkMimo26ModelLayerIsMoe(index);
	layer->resident = 0u;
	layer->capacity = 0u;
	layer->demand = 0u;
	if ( layer->moe )
	{
		snprintf(name,sizeof(name),"l%02u_expert_used.i32",index);
		count = (uint32_t)(M26FileBytes(name) / 4u);
		raw = M26Read(name,count * 4u);
		layer->capacity = count + M26_SPARE_SLOTS;
		layer->slot.assign(M26_EXPERTS,-1);
	}
	M26LoadAttention(index,layer,page_table,error);
	M26LoadMlp(index,layer);
	for (e = 0u; layer->moe && e < count; e++)
		M26LoadExpert(index,layer,(uint32_t)((const int32_t *)raw.data())[e],layer->resident++);
}

static void M26Attention(uint32_t index, M26Layer *layer, M26Buffers *b, uint32_t position)
{
	uint32_t rows = layer->swa ? M26_SWA_QKV : M26_FULL_QKV;
	float scale = 1.0f / sqrtf((float)M26_HEAD_DIM);
	LmBf16RmsNormKernel<M26_THREADS><<<1,M26_THREADS,(M26_HIDDEN + 64u) * sizeof(float)>>>(b->stream,layer->attn_norm,b->normed,M26_HIDDEN,M26_HIDDEN,M26_EPS);
	M26Require(LmSkinnyExperts<LmFp8>(layer->qkv,LmScaleTensorBuild(layer->qkv_scale,LM_SCALE_ENCODING_F32,1u,rows,M26_HIDDEN,1u,128u),b->normed,b->fused,b->sequence,b->sequence,1u,1u,0u,M26_HIDDEN,rows,0),"qkv fp8 skinny",index);
	if ( layer->swa )
	{
		M26SplitQkvKernel<M26_SWA_KV,M26_SWA_QKV><<<(rows + 255u) / 256u,256u>>>(b->fused,b->query,b->key,b->value,position,SPARK_MIMO26_MODEL_SWA_ROPE_THETA);
		LmGqaKvStoreKernel<M26SwaKv,M26_THREADS,M26_SWA_KV,M26_HEAD_DIM,M26_VALUE_DIM><<<1,M26_THREADS>>>(layer->cache,b->key,b->value,b->sequence,b->position,1u);
		LmGqaSinkAttentionDecodeKernel<M26SwaKv,M26_THREADS,M26_SWA_KV,M26_HEAD_DIM,M26_VALUE_DIM><<<dim3(1u,M26_HEADS),M26_THREADS>>>(b->query,layer->cache,b->sequence,b->context,0,0u,M26_HEADS,scale,b->attended,b->position,layer->sink);
	}
	else
	{
		M26SplitQkvKernel<M26_FULL_KV,M26_FULL_QKV><<<(rows + 255u) / 256u,256u>>>(b->fused,b->query,b->key,b->value,position,SPARK_MIMO26_MODEL_FULL_ROPE_THETA);
		LmGqaKvStoreKernel<M26FullKv,M26_THREADS,M26_FULL_KV,M26_HEAD_DIM,M26_VALUE_DIM><<<1,M26_THREADS>>>(layer->cache,b->key,b->value,b->sequence,b->position,1u);
		LmGqaAttentionDecodeKernel<M26FullKv,M26_THREADS,M26_FULL_KV,M26_HEAD_DIM,M26_VALUE_DIM><<<dim3(1u,M26_HEADS),M26_THREADS>>>(b->query,layer->cache,b->sequence,b->context,0,0u,M26_HEADS,scale,b->attended,b->position);
	}
	M26Require(LmSkinnyDense<LmBf16Format>(layer->o_proj,b->attended,b->projected,0,1u,M26_O_INPUT,M26_HIDDEN,M26_HIDDEN,0u,0),"o_proj bf16 skinny",index);
	M26AddKernel<<<(M26_HIDDEN + 255u) / 256u,256u>>>(b->projected,b->stream,M26_HIDDEN);
	LmBf16RmsNormKernel<M26_THREADS><<<1,M26_THREADS,(M26_HIDDEN + 64u) * sizeof(float)>>>(b->stream,layer->mlp_norm,b->normed,M26_HIDDEN,M26_HIDDEN,M26_EPS);
}

static void M26Dense(uint32_t index, M26Layer *layer, M26Buffers *b)
{
	M26Require(LmSkinnyExperts<LmFp8>(layer->projection[0],LmScaleTensorBuild(layer->dense_scale[0],LM_SCALE_ENCODING_F32,1u,M26_DENSE,M26_HIDDEN,1u,128u),b->normed,b->gate,b->sequence,b->sequence,1u,1u,0u,M26_HIDDEN,M26_DENSE,0),"dense gate",index);
	M26Require(LmSkinnyExperts<LmFp8>(layer->projection[1],LmScaleTensorBuild(layer->dense_scale[1],LM_SCALE_ENCODING_F32,1u,M26_DENSE,M26_HIDDEN,1u,128u),b->normed,b->up,b->sequence,b->sequence,1u,1u,0u,M26_HIDDEN,M26_DENSE,0),"dense up",index);
	M26SwigluKernel<<<(M26_DENSE + 255u) / 256u,256u>>>(b->gate,b->up,b->act,M26_DENSE);
	M26Require(LmSkinnyExperts<LmFp8>(layer->projection[2],LmScaleTensorBuild(layer->dense_scale[2],LM_SCALE_ENCODING_F32,1u,M26_HIDDEN,M26_DENSE,1u,128u),b->act,b->down,b->sequence,b->sequence,1u,1u,0u,M26_DENSE,M26_HIDDEN,0),"dense down",index);
	M26AddKernel<<<(M26_HIDDEN + 255u) / 256u,256u>>>(b->down,b->stream,M26_HIDDEN);
}

static void M26Route(uint32_t index, M26Layer *layer, const float *logits, uint32_t *ids, uint32_t *slots, float *weights)
{
	float scores[M26_EXPERTS], total = 0.0f;
	bool used[M26_EXPERTS] = {false};
	uint32_t k, e, best;
	for (e = 0u; e < M26_EXPERTS; e++)
		scores[e] = 1.0f / (1.0f + expf(-logits[e]));
	for (k = 0u; k < M26_TOP_K; k++)
	{
		best = M26_EXPERTS;
		for (e = 0u; e < M26_EXPERTS; e++)
			if ( !used[e] && (best == M26_EXPERTS || scores[e] + layer->bias[e] > scores[best] + layer->bias[best]) )
				best = e;
		used[best] = true;
		ids[k] = best;
		weights[k] = scores[best];
		total += scores[best];
		if ( layer->slot[best] < 0 )
		{
			if ( layer->resident == layer->capacity )
			{
				fprintf(stderr,"FAIL layer %u expert %u: all %u spare slots already demand-loaded\n",index,best,M26_SPARE_SLOTS);
				exit(1);
			}
			printf("demand-load layer %u expert %u (outside the CPU reference route set)\n",index,best);
			M26LoadExpert(index,layer,best,layer->resident);
			layer->resident++;
			layer->demand++;
		}
		slots[k] = (uint32_t)layer->slot[best];
	}
	for (k = 0u; k < M26_TOP_K; k++)
		weights[k] /= total + SPARK_MIMO26_MODEL_ROUTER_NORM_EPSILON;
}

static void M26Moe(uint32_t index, M26Layer *layer, M26Buffers *b, uint32_t *ids)
{
	std::vector<float> logits(M26_EXPERTS);
	uint32_t slots[M26_TOP_K];
	float weights[M26_TOP_K];
	M26Require(LmSkinnyDense<LmBf16Format>(layer->router,b->normed,0,b->logits,1u,M26_HIDDEN,M26_EXPERTS,M26_EXPERTS,0u,0),"router",index);
	M26Check(cudaMemcpy(logits.data(),b->logits,M26_EXPERTS * 4u,cudaMemcpyDeviceToHost),"logits");
	M26Route(index,layer,logits.data(),ids,slots,weights);
	M26Check(cudaMemcpy(b->route,slots,M26_TOP_K * 4u,cudaMemcpyHostToDevice),"route");
	M26Check(cudaMemcpy(b->route_weight,weights,M26_TOP_K * 4u,cudaMemcpyHostToDevice),"route weights");
	M26Require(LmSkinnyExperts<LmMxfp4>(layer->projection[0],LmScaleTensorBlockUe8m0(layer->projection_scale[0],layer->capacity,M26_INTER,M26_HIDDEN,1u,32u),b->normed,b->gate,b->route,b->packed,M26_TOP_K,M26_TOP_K,0u,M26_HIDDEN,M26_INTER,0),"expert gate",index);
	M26Require(LmSkinnyExperts<LmMxfp4>(layer->projection[1],LmScaleTensorBlockUe8m0(layer->projection_scale[1],layer->capacity,M26_INTER,M26_HIDDEN,1u,32u),b->normed,b->up,b->route,b->packed,M26_TOP_K,M26_TOP_K,0u,M26_HIDDEN,M26_INTER,0),"expert up",index);
	M26SwigluKernel<<<(M26_TOP_K * M26_INTER + 255u) / 256u,256u>>>(b->gate,b->up,b->act,M26_TOP_K * M26_INTER);
	M26Require(LmSkinnyExperts<LmMxfp4>(layer->projection[2],LmScaleTensorBlockUe8m0(layer->projection_scale[2],layer->capacity,M26_HIDDEN,M26_INTER,1u,32u),b->act,b->down,b->route,b->packed,M26_TOP_K,M26_TOP_K,1u,M26_INTER,M26_HIDDEN,0),"expert down",index);
	M26CombineKernel<<<(M26_HIDDEN + 255u) / 256u,256u>>>(b->down,b->route_weight,b->stream);
}

static void M26Allocate(M26Buffers *b)
{
	uint32_t zero = 0u, i, packed[M26_TOP_K], pages[M26_PAGES];
	b->stream = M26Alloc<uint16_t>(M26_HIDDEN);
	b->normed = M26Alloc<uint16_t>(M26_HIDDEN);
	b->fused = M26Alloc<uint16_t>(M26_SWA_QKV);
	b->query = M26Alloc<uint16_t>(M26_HEADS * M26_HEAD_DIM);
	b->key = M26Alloc<uint16_t>(M26_SWA_KV * M26_HEAD_DIM);
	b->value = M26Alloc<uint16_t>(M26_SWA_KV * M26_VALUE_DIM);
	b->attended = M26Alloc<uint16_t>(M26_O_INPUT);
	b->projected = M26Alloc<uint16_t>(M26_HIDDEN);
	b->gate = M26Alloc<uint16_t>(M26_DENSE);
	b->up = M26Alloc<uint16_t>(M26_DENSE);
	b->act = M26Alloc<uint16_t>(M26_DENSE);
	b->down = M26Alloc<uint16_t>(M26_TOP_K * M26_HIDDEN);
	b->logits = M26Alloc<float>(M26_VOCAB);
	b->route_weight = M26Alloc<float>(M26_TOP_K);
	b->sequence = M26Alloc<uint32_t>(1u);
	b->context = M26Alloc<uint32_t>(1u);
	b->position = M26Alloc<uint32_t>(1u);
	b->page_table = M26Alloc<uint32_t>(M26_PAGES);
	b->route = M26Alloc<uint32_t>(M26_TOP_K);
	b->packed = M26Alloc<uint32_t>(M26_TOP_K);
	b->error = M26Alloc<LmKvAccessError>(1u);
	for (i = 0u; i < M26_TOP_K; i++)
		packed[i] = i;
	for (i = 0u; i < M26_PAGES; i++)
		pages[i] = i;
	M26Check(cudaMemcpy(b->packed,packed,sizeof(packed),cudaMemcpyHostToDevice),"packed");
	M26Check(cudaMemcpy(b->page_table,pages,sizeof(pages),cudaMemcpyHostToDevice),"pages");
	M26Check(cudaMemcpy(b->sequence,&zero,4u,cudaMemcpyHostToDevice),"sequence");
	b->final_norm = M26Upload<uint16_t>("final_norm.bf16",M26_HIDDEN * 2u);
	b->lm_head = M26Upload<uint16_t>("lm_head.bf16",(size_t)M26_VOCAB * M26_HIDDEN * 2u);
}

static uint32_t M26Head(M26Buffers *b, float *score)
{
	std::vector<float> logits(M26_VOCAB);
	uint32_t best = 0u, v;
	LmBf16RmsNormKernel<M26_THREADS><<<1,M26_THREADS,(M26_HIDDEN + 64u) * sizeof(float)>>>(b->stream,b->final_norm,b->normed,M26_HIDDEN,M26_HIDDEN,M26_EPS);
	M26Require(LmSkinnyDense<LmBf16Format>(b->lm_head,b->normed,0,b->logits,1u,M26_HIDDEN,M26_VOCAB,M26_VOCAB,0u,0),"lm_head",M26_LAYERS);
	M26Check(cudaMemcpy(logits.data(),b->logits,M26_VOCAB * 4u,cudaMemcpyDeviceToHost),"head logits");
	for (v = 1u; v < M26_VOCAB; v++)
		best = logits[v] > logits[best] ? v : best;
	*score = logits[best];
	return(best);
}

static double M26StreamError(M26Buffers *b, uint32_t position, uint32_t layer)
{
	char name[128];
	std::vector<uint16_t> got(M26_HIDDEN);
	double error = 0.0, norm = 0.0, want;
	uint32_t k;
	snprintf(name,sizeof(name),"pos%04u_layer%04u_streams.bf16",position,layer);
	std::vector<uint8_t> raw = M26Read(name,M26_HIDDEN * 2u);
	M26Check(cudaMemcpy(got.data(),b->stream,M26_HIDDEN * 2u,cudaMemcpyDeviceToHost),"stream");
	for (k = 0u; k < M26_HIDDEN; k++)
	{
		want = M26Float(((const uint16_t *)raw.data())[k]);
		error += (M26Float(got[k]) - want) * (M26Float(got[k]) - want);
		norm += want * want;
	}
	return(sqrt(error / norm));
}

static void M26Embed(M26Buffers *b, uint32_t token)
{
	char path[1024];
	uint16_t row[M26_HIDDEN];
	FILE *file;
	snprintf(path,sizeof(path),"%s/embed.bf16",m26_directory);
	file = fopen(path,"rb");
	if ( token >= M26_VOCAB || file == 0 || fseek(file,(long)token * M26_HIDDEN * 2L,SEEK_SET) != 0 || fread(row,2u,M26_HIDDEN,file) != M26_HIDDEN )
	{
		fprintf(stderr,"FAIL embedding row %u\n",token);
		exit(1);
	}
	fclose(file);
	M26Check(cudaMemcpy(b->stream,row,sizeof(row),cudaMemcpyHostToDevice),"embed");
}

int main(int argc, char **argv)
{
	static M26Layer layers[M26_LAYERS];
	static const uint32_t anchors[4] = {0u,1u,5u,47u};
	M26Buffers b;
	uint32_t prompt_count, budget, total, position, layer, a, token, ids[M26_TOP_K], mismatches = 0u, demand = 0u;
	float score;
	double error, worst = 0.0;
	LmKvAccessError access;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s <model-input-dir>\n",argv[0]);
		return(2);
	}
	m26_directory = argv[1];
	prompt_count = (uint32_t)(M26FileBytes("prompt.i32") / 4u);
	budget = (uint32_t)(M26FileBytes("generated.i32") / 4u);
	total = prompt_count + budget;
	std::vector<uint8_t> prompt = M26Read("prompt.i32",prompt_count * 4u), expected = M26Read("generated.i32",budget * 4u);
	std::vector<uint32_t> generated;
	if ( total > M26_MAX_POSITIONS || total > M26_WINDOW )
	{
		fprintf(stderr,"FAIL %u positions exceed the %u-slot harness cache\n",total,M26_WINDOW);
		return(2);
	}
	M26Allocate(&b);
	for (layer = 0u; layer < M26_LAYERS; layer++)
		M26LoadLayer(layer,&layers[layer],b.page_table,b.error);
	for (position = 0u; position + 1u < total; position++)
	{
		token = position < prompt_count ? ((const uint32_t *)prompt.data())[position] : generated[position - prompt_count];
		M26Embed(&b,token);
		M26Check(cudaMemcpy(b.position,&position,4u,cudaMemcpyHostToDevice),"position");
		uint32_t context = position + 1u;
		M26Check(cudaMemcpy(b.context,&context,4u,cudaMemcpyHostToDevice),"context");
		for (layer = 0u; layer < M26_LAYERS; layer++)
		{
			M26Attention(layer,&layers[layer],&b,position);
			if ( layers[layer].moe )
				M26Moe(layer,&layers[layer],&b,ids);
			else
				M26Dense(layer,&layers[layer],&b);
			M26Check(cudaDeviceSynchronize(),"layer");
			for (a = 0u; a < 4u; a++)
				if ( anchors[a] == layer )
				{
					error = M26StreamError(&b,position,layer);
					worst = error > worst ? error : worst;
					printf("position %u layer %u stream_rel_l2 %.3e\n",position,layer,error);
				}
		}
		token = M26Head(&b,&score);
		if ( position + 1u >= prompt_count )
		{
			generated.push_back(token);
			mismatches += token != ((const uint32_t *)expected.data())[position + 1u - prompt_count] ? 1u : 0u;
			printf("position %u greedy %u reference %u score %.4f\n",position,token,((const uint32_t *)expected.data())[position + 1u - prompt_count],score);
		}
	}
	M26Check(cudaMemcpy(&access,b.error,sizeof(access),cudaMemcpyDeviceToHost),"kv error");
	for (layer = 0u; layer < M26_LAYERS; layer++)
		demand += layers[layer].demand;
	printf("model summary tokens=%zu mismatches=%u worst_anchor_stream_rel_l2=%.3e kv_error=%u demand_loaded_experts=%u\n",generated.size(),mismatches,worst,access.error_code,demand);
	if ( mismatches != 0u || access.error_code != 0u )
	{
		fprintf(stderr,"FAIL greedy tokens differ from the CPU reference\n");
		return(1);
	}
	printf("PASS mimo26 flash on one GPU: %zu greedy tokens equal the CPU reference\n",generated.size());
	return(0);
}
