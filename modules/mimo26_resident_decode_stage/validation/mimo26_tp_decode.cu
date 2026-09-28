#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "inference/kernels/skinny.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/topk.cuh"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_mimo26_model.h"

#include "spark_mimo26_rank_pack.h"

extern "C" SparkStatus SparkStageModuleCudaStatus(const char *module_tag, cudaError_t error, const char *site);

#define SPARK_FAMILY_CAMEL Mimo26Tp
#define SPARK_FAMILY_UPPER MIMO26_TP
#define SPARK_FAMILY_LOWER mimo26_tp
#include "sparkpipe/family/spark_family.h"
#define SPARK_MIMO26_TP_MODULE_TAG "mimo26_tp_decode"
#include "sparkpipe/family/module/spark_module_combine.h"

#define TP_HIDDEN SPARK_MIMO26_MODEL_HIDDEN_DIMENSION
#define TP_LAYERS SPARK_MIMO26_MODEL_LAYER_COUNT
#define TP_VOCAB SPARK_MIMO26_MODEL_VOCAB_COUNT
#define TP_HEADS SPARK_MIMO26_MODEL_ATTN_HEAD_COUNT
#define TP_HEAD_DIM SPARK_MIMO26_MODEL_ATTN_HEAD_DIMENSION
#define TP_VALUE_DIM SPARK_MIMO26_MODEL_ATTN_VALUE_DIMENSION
#define TP_ROPE_DIM SPARK_MIMO26_MODEL_ATTN_ROPE_DIMENSION
#define TP_FULL_KV SPARK_MIMO26_MODEL_FULL_KV_HEAD_COUNT
#define TP_SWA_KV SPARK_MIMO26_MODEL_SWA_KV_HEAD_COUNT
#define TP_EXPERTS SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT
#define TP_TOP_K SPARK_MIMO26_MODEL_EXPERTS_PER_TOKEN
#define TP_INTER SPARK_MIMO26_MODEL_EXPERT_INTERMEDIATE_DIMENSION
#define TP_DENSE SPARK_MIMO26_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define TP_EPS SPARK_MIMO26_MODEL_RMS_NORM_EPSILON
#define TP_VALUE_SCALE SPARK_MIMO26_MODEL_ATTN_VALUE_SCALE
#define TP_WINDOW SPARK_MIMO26_MODEL_SLIDING_WINDOW_TOKENS
#define TP_FP8_BLOCK SPARK_MIMO26_MODEL_FP8_SCALE_BLOCK
#define TP_DEGREE 4u
#define TP_THREADS 256u
#define TP_PAGE_SLOTS 64u
#define TP_MAX_POSITIONS 128u
#define TP_PAGES (TP_MAX_POSITIONS / TP_PAGE_SLOTS)
#define TP_RANK_HEADS (TP_HEADS / TP_DEGREE)
#define TP_RANK_O_INPUT (SPARK_MIMO26_MODEL_O_INPUT_DIMENSION / TP_DEGREE)
#define TP_RANK_VOCAB (TP_VOCAB / TP_DEGREE)
#define TP_RANK_EXPERTS (TP_EXPERTS / TP_DEGREE)
#define TP_RANK_DENSE (TP_DENSE / TP_DEGREE)
#define TP_WAIT_NS UINT64_C(300000000000)

using TpFullKv = LmKvGeometry<(TP_FULL_KV / TP_DEGREE) * (TP_HEAD_DIM + TP_VALUE_DIM) * sizeof(uint16_t), TP_PAGE_SLOTS, true>;
using TpSwaKv = LmKvGeometry<(TP_SWA_KV / TP_DEGREE) * (TP_HEAD_DIM + TP_VALUE_DIM) * sizeof(uint16_t), TP_PAGE_SLOTS, true>;

typedef struct TpLayer
{
	uint32_t swa, moe;
	const uint16_t *attn_norm, *mlp_norm, *sink, *o_proj, *router;
	const uint8_t *qkv, *dense[3];
	const float *router_bias;
	float *qkv_scale, *dense_scale[3];
	uint32_t qkv_rows;
	uint64_t expert_payload[3], expert_scale[3];
	uint8_t *pool;
	LmKvView cache;
} TpLayer;

typedef struct TpBuffers
{
	uint16_t *stream, *normed, *fused, *query, *key, *value, *attended, *partial, *reduced, *gate, *up, *act, *down;
	float *logits, *route_weight, *local_weight;
	uint32_t *route_expert, *local_route, *packed, *sequence, *context, *position, *page_table;
	uint64_t *head_key, *head_best;
	LmKvAccessError *error;
	const uint16_t *embedding, *final_norm, *lm_head;
} TpBuffers;

typedef struct TpCompletion
{
	pthread_mutex_t lock;
	pthread_cond_t wake;
	uint32_t done;
	uint32_t status;
} TpCompletion;

typedef struct TpContext
{
	uint32_t rank;
	SparkWeightdLazyPack *pack;
	SparkMimo26StagePackEntry *entries;
	SparkMimo26RankPackLayout layout;
	SparkTpDeviceCollective collective;
	TpCompletion completion;
	cudaStream_t collective_stream;
	uint64_t ordinal, reduce_ns, expert_ns, reduce_count;
} TpContext;

static TpContext tp;

static void TpFail(const char *what, const char *detail)
{
	fprintf(stderr,"M26TP-FAIL rank=%u phase=%s %s\n",tp.rank,what,detail != 0 ? detail : "");
	fflush(stderr);
	exit(1);
}

static void TpCheck(cudaError_t status, const char *what)
{
	if ( status != cudaSuccess )
		TpFail(what,cudaGetErrorString(status));
}

static void TpStatus(SparkStatus status, const char *what)
{
	if ( status != SPARK_STATUS_OK )
		TpFail(what,SparkStatusToString(status));
}

static void TpLaunch(int32_t status, const char *what, uint32_t layer)
{
	char detail[64];
	if ( status != LM_LAUNCH_OK )
	{
		snprintf(detail,sizeof(detail),"layer=%u launch=%d",layer,status);
		TpFail(what,detail);
	}
}

static uint64_t TpNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

template<class T>
static T *TpAlloc(size_t count)
{
	void *device;
	TpCheck(cudaMalloc(&device,count * sizeof(T)),"alloc");
	TpCheck(cudaMemset(device,0,count * sizeof(T)),"clear");
	return((T *)device);
}

static const SparkMimo26StagePackEntry *TpEntry(uint32_t layer, uint32_t kind)
{
	uint32_t index = layer == SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER ? tp.layout.global_entry[kind] : tp.layout.layer_entry[layer][kind];
	if ( index == SPARK_MIMO26_RANK_PACK_ABSENT )
		TpFail("entry","absent");
	return(&tp.entries[index]);
}

template<class T>
static const T *TpSpine(uint64_t offset, uint64_t bytes)
{
	const void *pointer = 0;
	TpStatus(SparkWeightdLazyPackSlice(tp.pack,offset,bytes,&pointer),"spine-slice");
	return((const T *)pointer);
}

template<class T>
static const T *TpPayload(uint32_t layer, uint32_t kind)
{
	const SparkMimo26StagePackEntry *entry = TpEntry(layer,kind);
	return(TpSpine<T>(entry->payload_offset,entry->payload_bytes));
}

__global__ void TpExpandBlockScaleKernel(const float *block, float *rows_out, uint32_t rows, uint32_t column_blocks)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < rows * column_blocks )
		rows_out[index] = block[(index / column_blocks / TP_FP8_BLOCK) * column_blocks + index % column_blocks];
}

static float *TpRowScales(uint32_t layer, uint32_t kind)
{
	const SparkMimo26StagePackEntry *entry = TpEntry(layer,kind);
	uint32_t column_blocks = entry->columns / TP_FP8_BLOCK;
	const float *block = TpSpine<float>(entry->scale_offset,entry->scale_bytes);
	float *rows = TpAlloc<float>((size_t)entry->rows * column_blocks);
	TpExpandBlockScaleKernel<<<(entry->rows * column_blocks + 255u) / 256u,256u>>>(block,rows,entry->rows,column_blocks);
	TpCheck(cudaGetLastError(),"expand-scale");
	return(rows);
}

template<uint32_t HEADS, uint32_t KV, uint32_t QKV>
__global__ void TpSplitQkvKernel(const uint16_t *fused, uint16_t *query, uint16_t *key, uint16_t *value, uint32_t position, float theta)
{
	const uint32_t q_local = HEADS * TP_HEAD_DIM, k_local = KV * TP_HEAD_DIM;
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x, local, dim, pair;
	float angle, first, second;
	uint16_t *target;
	if ( row >= QKV )
		return;
	local = row;
	if ( local >= q_local + k_local )
	{
		value[local - q_local - k_local] = LmFloatToBf16(LmBf16ToFloat(fused[row]) * TP_VALUE_SCALE);
		return;
	}
	target = local < q_local ? query : key;
	local = local < q_local ? local : local - q_local;
	dim = local % TP_HEAD_DIM;
	if ( dim >= TP_ROPE_DIM )
	{
		target[local] = fused[row];
		return;
	}
	pair = dim < TP_ROPE_DIM / 2u ? dim + TP_ROPE_DIM / 2u : dim - TP_ROPE_DIM / 2u;
	angle = (float)position * powf(theta,-(2.0f * (float)(dim % (TP_ROPE_DIM / 2u))) / (float)TP_ROPE_DIM);
	first = LmBf16ToFloat(fused[row]);
	second = LmBf16ToFloat(fused[row - dim + pair]);
	target[local] = LmFloatToBf16(dim < TP_ROPE_DIM / 2u ? first * cosf(angle) - second * sinf(angle) : first * cosf(angle) + second * sinf(angle));
}

__global__ void TpAddKernel(const uint16_t *right, uint16_t *stream, uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		stream[index] = LmFloatToBf16(LmBf16ToFloat(stream[index]) + LmBf16ToFloat(right[index]));
}

__global__ void TpSwigluKernel(const uint16_t *gate, const uint16_t *up, uint16_t *out, uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	float g;
	if ( index >= count )
		return;
	g = LmBf16ToFloat(gate[index]);
	out[index] = LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(g / (1.0f + __expf(-g)))) * LmBf16ToFloat(up[index]));
}

__global__ void TpCombineLocalKernel(const uint16_t *down, const float *weights, uint32_t count, uint16_t *partial)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x, k;
	float total = 0.0f;
	if ( index >= TP_HIDDEN )
		return;
	for (k = 0u; k < count; k++)
		total += weights[k] * LmBf16ToFloat(down[k * TP_HIDDEN + index]);
	partial[index] = LmFloatToBf16(total);
}

__global__ void TpEmbedKernel(const uint16_t *embedding, uint32_t token, uint32_t first_row, uint16_t *out)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= TP_HIDDEN )
		return;
	out[index] = token >= first_row && token < first_row + TP_RANK_VOCAB ? embedding[(size_t)(token - first_row) * TP_HIDDEN + index] : (uint16_t)0u;
}

__global__ void TpHeadKeyKernel(const float *logits, uint32_t first_row, uint64_t *key_out)
{
	__shared__ uint64_t best[1024];
	uint32_t index, bits, ordered;
	uint64_t mine = 0u, candidate;
	for (index = threadIdx.x; index < TP_RANK_VOCAB; index += blockDim.x)
	{
		bits = __float_as_uint(logits[index]);
		ordered = (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
		candidate = ((uint64_t)ordered << 32) | (uint64_t)(0xFFFFFFFFu - (first_row + index));
		mine = candidate > mine ? candidate : mine;
	}
	best[threadIdx.x] = mine;
	__syncthreads();
	for (index = blockDim.x / 2u; index > 0u; index >>= 1u)
	{
		if ( threadIdx.x < index && best[threadIdx.x + index] > best[threadIdx.x] )
			best[threadIdx.x] = best[threadIdx.x + index];
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
		*key_out = best[0];
}

static void TpCompletionMark(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	TpCompletion *state = (TpCompletion *)context;
	pthread_mutex_lock(&state->lock);
	state->done++;
	state->status = (uint32_t)completion->status;
	pthread_cond_signal(&state->wake);
	pthread_mutex_unlock(&state->lock);
}

static void TpWaitStream(cudaStream_t stream)
{
	uint64_t deadline = TpNow() + TP_WAIT_NS;
	cudaError_t status;
	for (;;)
	{
		status = cudaStreamQuery(stream);
		if ( status == cudaSuccess )
			return;
		if ( status != cudaErrorNotReady || TpNow() >= deadline )
			TpFail("collective-stream",cudaGetErrorString(status));
		usleep(20u);
	}
}

static void TpCollective(uint32_t operation, const void *local, void *full)
{
	SparkTpDeviceCollectiveSubmission submission;
	struct timespec deadline;
	uint64_t started = TpNow();
	uint32_t before;
	TpCheck(cudaDeviceSynchronize(),"pre-collective");
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.active_sequence_count = 1u;
	submission.logical_sequence_count = 1u;
	submission.ordinal = tp.ordinal++;
	submission.local_device = local;
	submission.full_device = full;
	submission.cuda_stream = tp.collective_stream;
	submission.completion_function = TpCompletionMark;
	submission.completion_context = &tp.completion;
	pthread_mutex_lock(&tp.completion.lock);
	before = tp.completion.done;
	pthread_mutex_unlock(&tp.completion.lock);
	TpStatus(SparkTpDeviceCollectiveEnqueue(&tp.collective,&submission,operation),"collective-enqueue");
	clock_gettime(CLOCK_MONOTONIC,&deadline);
	deadline.tv_sec += (time_t)(TP_WAIT_NS / 1000000000ull);
	pthread_mutex_lock(&tp.completion.lock);
	while ( tp.completion.done == before )
		if ( pthread_cond_timedwait(&tp.completion.wake,&tp.completion.lock,&deadline) != 0 )
			break;
	if ( tp.completion.done != before + 1u || tp.completion.status != (uint32_t)SPARK_STATUS_OK )
	{
		pthread_mutex_unlock(&tp.completion.lock);
		TpFail("collective-completion",operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 ? "max_u64" : "sum_bf16");
	}
	pthread_mutex_unlock(&tp.completion.lock);
	TpWaitStream(tp.collective_stream);
	if ( SparkTpDeviceCollectiveGraphError(&tp.collective) != 0u )
		TpFail("collective-error","device word set");
	tp.reduce_ns += TpNow() - started;
	tp.reduce_count++;
}

static void TpReduceAdd(TpBuffers *b)
{
	TpCollective(SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,b->partial,b->reduced);
	TpAddKernel<<<(TP_HIDDEN + 255u) / 256u,256u>>>(b->reduced,b->stream,TP_HIDDEN);
}

static void TpBindLayer(uint32_t index, TpLayer *layer, uint32_t *page_table, LmKvAccessError *error)
{
	const SparkMimo26StagePackEntry *entry;
	size_t slot_bytes;
	uint32_t p;
	const uint32_t dense_kinds[3] = {SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_GATE,SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_UP,SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_DOWN};
	const uint32_t expert_kinds[3] = {SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_GATE,SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_UP,SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_DOWN};
	memset(layer,0,sizeof(*layer));
	layer->swa = SparkMimo26ModelLayerKind(index) == SPARK_MIMO26_MODEL_LAYER_KIND_SWA ? 1u : 0u;
	layer->moe = SparkMimo26ModelLayerIsMoe(index);
	layer->attn_norm = TpPayload<uint16_t>(index,SPARK_MIMO26_STAGEPACK_TENSOR_ATTENTION_NORM);
	layer->mlp_norm = TpPayload<uint16_t>(index,SPARK_MIMO26_STAGEPACK_TENSOR_MLP_NORM);
	layer->o_proj = TpPayload<uint16_t>(index,SPARK_MIMO26_STAGEPACK_TENSOR_O_PROJ);
	layer->qkv = TpPayload<uint8_t>(index,SPARK_MIMO26_STAGEPACK_TENSOR_QKV);
	layer->qkv_scale = TpRowScales(index,SPARK_MIMO26_STAGEPACK_TENSOR_QKV);
	layer->qkv_rows = TpEntry(index,SPARK_MIMO26_STAGEPACK_TENSOR_QKV)->rows;
	if ( layer->swa != 0u )
		layer->sink = TpPayload<uint16_t>(index,SPARK_MIMO26_STAGEPACK_TENSOR_SINK_BIAS);
	if ( layer->moe != 0u )
	{
		layer->router = TpPayload<uint16_t>(index,SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE);
		layer->router_bias = TpPayload<float>(index,SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE_BIAS);
		for (p = 0u; p < 3u; p++)
		{
			entry = TpEntry(index,expert_kinds[p]);
			layer->expert_payload[p] = entry->payload_offset;
			layer->expert_scale[p] = entry->scale_offset;
		}
	}
	else
		for (p = 0u; p < 3u; p++)
		{
			layer->dense[p] = TpPayload<uint8_t>(index,dense_kinds[p]);
			layer->dense_scale[p] = TpRowScales(index,dense_kinds[p]);
		}
	slot_bytes = layer->swa != 0u ? TpSwaKv::kSlotBytes : TpFullKv::kSlotBytes;
	layer->pool = TpAlloc<uint8_t>((size_t)TP_MAX_POSITIONS * slot_bytes);
	if ( LmKvViewInitialize(&layer->cache,layer->pool,page_table,TP_PAGES,1u,TP_PAGES,error) != 0 )
		TpFail("kv-view","initialize");
}

template<uint32_t KV, class Geometry>
static void TpAttentionRank(uint32_t index, TpLayer *layer, TpBuffers *b, uint32_t position, float theta)
{
	const uint32_t rank_kv = KV / TP_DEGREE, rows = TP_RANK_HEADS * TP_HEAD_DIM + rank_kv * (TP_HEAD_DIM + TP_VALUE_DIM);
	float scale = 1.0f / sqrtf((float)TP_HEAD_DIM);
	if ( rows != layer->qkv_rows )
		TpFail("qkv-rows","pack rows differ from the rank geometry");
	LmBf16RmsNormKernel<TP_THREADS><<<1,TP_THREADS,(TP_HIDDEN + 64u) * sizeof(float)>>>(b->stream,layer->attn_norm,b->normed,TP_HIDDEN,TP_HIDDEN,TP_EPS);
	TpLaunch(LmSkinnyExperts<LmFp8>(layer->qkv,LmScaleTensorBuild(layer->qkv_scale,LM_SCALE_ENCODING_F32,1u,rows,TP_HIDDEN,1u,TP_FP8_BLOCK),b->normed,b->fused,b->sequence,b->sequence,1u,1u,0u,TP_HIDDEN,rows,0),"qkv",index);
	TpSplitQkvKernel<TP_RANK_HEADS,KV / TP_DEGREE,TP_RANK_HEADS * TP_HEAD_DIM + (KV / TP_DEGREE) * (TP_HEAD_DIM + TP_VALUE_DIM)><<<(rows + 255u) / 256u,256u>>>(b->fused,b->query,b->key,b->value,position,theta);
	LmGqaKvStoreKernel<Geometry,TP_THREADS,KV / TP_DEGREE,TP_HEAD_DIM,TP_VALUE_DIM><<<1,TP_THREADS>>>(layer->cache,b->key,b->value,b->sequence,b->position,1u);
	if ( layer->swa != 0u )
		LmGqaSinkAttentionDecodeKernel<Geometry,TP_THREADS,KV / TP_DEGREE,TP_HEAD_DIM,TP_VALUE_DIM><<<dim3(1u,TP_RANK_HEADS),TP_THREADS>>>(b->query,layer->cache,b->sequence,b->context,0,0u,TP_RANK_HEADS,scale,b->attended,b->position,layer->sink);
	else
		LmGqaAttentionDecodeKernel<Geometry,TP_THREADS,KV / TP_DEGREE,TP_HEAD_DIM,TP_VALUE_DIM><<<dim3(1u,TP_RANK_HEADS),TP_THREADS>>>(b->query,layer->cache,b->sequence,b->context,0,0u,TP_RANK_HEADS,scale,b->attended,b->position);
	TpLaunch(LmSkinnyDense<LmBf16Format>(layer->o_proj,b->attended,b->partial,0,1u,TP_RANK_O_INPUT,TP_HIDDEN,TP_HIDDEN,0u,0),"o_proj",index);
	TpReduceAdd(b);
	LmBf16RmsNormKernel<TP_THREADS><<<1,TP_THREADS,(TP_HIDDEN + 64u) * sizeof(float)>>>(b->stream,layer->mlp_norm,b->normed,TP_HIDDEN,TP_HIDDEN,TP_EPS);
}

static void TpDense(uint32_t index, TpLayer *layer, TpBuffers *b)
{
	TpLaunch(LmSkinnyExperts<LmFp8>(layer->dense[0],LmScaleTensorBuild(layer->dense_scale[0],LM_SCALE_ENCODING_F32,1u,TP_RANK_DENSE,TP_HIDDEN,1u,TP_FP8_BLOCK),b->normed,b->gate,b->sequence,b->sequence,1u,1u,0u,TP_HIDDEN,TP_RANK_DENSE,0),"dense-gate",index);
	TpLaunch(LmSkinnyExperts<LmFp8>(layer->dense[1],LmScaleTensorBuild(layer->dense_scale[1],LM_SCALE_ENCODING_F32,1u,TP_RANK_DENSE,TP_HIDDEN,1u,TP_FP8_BLOCK),b->normed,b->up,b->sequence,b->sequence,1u,1u,0u,TP_HIDDEN,TP_RANK_DENSE,0),"dense-up",index);
	TpSwigluKernel<<<(TP_RANK_DENSE + 255u) / 256u,256u>>>(b->gate,b->up,b->act,TP_RANK_DENSE);
	TpLaunch(LmSkinnyExperts<LmFp8>(layer->dense[2],LmScaleTensorBuild(layer->dense_scale[2],LM_SCALE_ENCODING_F32,1u,TP_HIDDEN,TP_RANK_DENSE,1u,TP_FP8_BLOCK),b->act,b->partial,b->sequence,b->sequence,1u,1u,0u,TP_RANK_DENSE,TP_HIDDEN,0),"dense-down",index);
	TpReduceAdd(b);
}

static void TpMoe(uint32_t index, TpLayer *layer, TpBuffers *b)
{
	SparkWeightdExpertKey keys[TP_TOP_K];
	uint32_t ids[TP_TOP_K], local[TP_TOP_K], count = 0u, k, p;
	float weights[TP_TOP_K], local_weights[TP_TOP_K];
	uint64_t lease = 0u, started;
	void *base = 0;
	const uint8_t *weight[3], *scale[3];
	TpLaunch(LmSkinnyDense<LmBf16Format>(layer->router,b->normed,0,b->logits,1u,TP_HIDDEN,TP_EXPERTS,TP_EXPERTS,0u,0),"router",index);
	LmTopkSmallKernel<TP_THREADS,TP_TOP_K,true,1u,1u,LM_TOPK_SCORE_SIGMOID><<<1,TP_THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t)>>>(b->logits,TP_EXPERTS,b->route_expert,b->route_weight,layer->router_bias,0,SPARK_MIMO26_MODEL_ROUTED_SCALING_FACTOR);
	TpCheck(cudaMemcpy(ids,b->route_expert,sizeof(ids),cudaMemcpyDeviceToHost),"route-ids");
	TpCheck(cudaMemcpy(weights,b->route_weight,sizeof(weights),cudaMemcpyDeviceToHost),"route-weights");
	for (k = 0u; k < TP_TOP_K; k++)
	{
		if ( ids[k] >= TP_EXPERTS )
			TpFail("route","expert id out of range");
		if ( ids[k] / TP_RANK_EXPERTS != tp.rank )
			continue;
		local[count] = ids[k] - tp.rank * TP_RANK_EXPERTS;
		local_weights[count] = weights[k];
		keys[count].layer = index;
		keys[count].expert = local[count];
		count++;
	}
	if ( count == 0u )
	{
		TpCheck(cudaMemset(b->partial,0,TP_HIDDEN * sizeof(uint16_t)),"empty-partial");
		TpReduceAdd(b);
		return;
	}
	started = TpNow();
	TpCheck(cudaMemcpy(b->local_route,local,count * sizeof(uint32_t),cudaMemcpyHostToDevice),"local-route");
	TpCheck(cudaMemcpy(b->local_weight,local_weights,count * sizeof(float),cudaMemcpyHostToDevice),"local-weight");
	TpStatus(SparkWeightdMapAcquire(tp.pack->map,keys,count,&lease,TP_WAIT_NS),"expert-acquire");
	TpStatus(SparkWeightdMapBeginUse(tp.pack->map,lease,&base),"expert-begin");
	for (p = 0u; p < 3u; p++)
	{
		weight[p] = (const uint8_t *)base + layer->expert_payload[p];
		scale[p] = (const uint8_t *)base + layer->expert_scale[p];
	}
	TpLaunch(LmSkinnyExperts<LmMxfp4>(weight[0],LmScaleTensorBlockUe8m0(scale[0],TP_RANK_EXPERTS,TP_INTER,TP_HIDDEN,1u,32u),b->normed,b->gate,b->local_route,b->packed,count,count,0u,TP_HIDDEN,TP_INTER,0),"expert-gate",index);
	TpLaunch(LmSkinnyExperts<LmMxfp4>(weight[1],LmScaleTensorBlockUe8m0(scale[1],TP_RANK_EXPERTS,TP_INTER,TP_HIDDEN,1u,32u),b->normed,b->up,b->local_route,b->packed,count,count,0u,TP_HIDDEN,TP_INTER,0),"expert-up",index);
	TpSwigluKernel<<<(count * TP_INTER + 255u) / 256u,256u>>>(b->gate,b->up,b->act,count * TP_INTER);
	TpLaunch(LmSkinnyExperts<LmMxfp4>(weight[2],LmScaleTensorBlockUe8m0(scale[2],TP_RANK_EXPERTS,TP_HIDDEN,TP_INTER,1u,32u),b->act,b->down,b->local_route,b->packed,count,count,1u,TP_INTER,TP_HIDDEN,0),"expert-down",index);
	TpCombineLocalKernel<<<(TP_HIDDEN + 255u) / 256u,256u>>>(b->down,b->local_weight,count,b->partial);
	TpStatus(SparkWeightdMapRecordCompletion(tp.pack->map,lease,0),"expert-record");
	TpCheck(cudaDeviceSynchronize(),"experts");
	TpStatus(SparkWeightdMapRelease(tp.pack->map,lease,TP_WAIT_NS),"expert-release");
	tp.expert_ns += TpNow() - started;
	TpReduceAdd(b);
}

static void TpAllocate(TpBuffers *b)
{
	uint32_t zero = 0u, i, packed[TP_TOP_K], pages[TP_PAGES];
	b->stream = TpAlloc<uint16_t>(TP_HIDDEN);
	b->normed = TpAlloc<uint16_t>(TP_HIDDEN);
	b->fused = TpAlloc<uint16_t>(TP_RANK_HEADS * TP_HEAD_DIM + (TP_SWA_KV / TP_DEGREE) * (TP_HEAD_DIM + TP_VALUE_DIM));
	b->query = TpAlloc<uint16_t>(TP_RANK_HEADS * TP_HEAD_DIM);
	b->key = TpAlloc<uint16_t>((TP_SWA_KV / TP_DEGREE) * TP_HEAD_DIM);
	b->value = TpAlloc<uint16_t>((TP_SWA_KV / TP_DEGREE) * TP_VALUE_DIM);
	b->attended = TpAlloc<uint16_t>(TP_RANK_O_INPUT);
	b->partial = TpAlloc<uint16_t>(TP_HIDDEN);
	b->reduced = TpAlloc<uint16_t>(TP_HIDDEN + 128u);
	b->gate = TpAlloc<uint16_t>(TP_RANK_DENSE > TP_TOP_K * TP_INTER ? TP_RANK_DENSE : TP_TOP_K * TP_INTER);
	b->up = TpAlloc<uint16_t>(TP_RANK_DENSE > TP_TOP_K * TP_INTER ? TP_RANK_DENSE : TP_TOP_K * TP_INTER);
	b->act = TpAlloc<uint16_t>(TP_RANK_DENSE > TP_TOP_K * TP_INTER ? TP_RANK_DENSE : TP_TOP_K * TP_INTER);
	b->down = TpAlloc<uint16_t>(TP_TOP_K * TP_HIDDEN);
	b->logits = TpAlloc<float>(TP_RANK_VOCAB);
	b->route_weight = TpAlloc<float>(TP_TOP_K);
	b->local_weight = TpAlloc<float>(TP_TOP_K);
	b->route_expert = TpAlloc<uint32_t>(TP_TOP_K);
	b->local_route = TpAlloc<uint32_t>(TP_TOP_K);
	b->packed = TpAlloc<uint32_t>(TP_TOP_K);
	b->sequence = TpAlloc<uint32_t>(1u);
	b->context = TpAlloc<uint32_t>(1u);
	b->position = TpAlloc<uint32_t>(1u);
	b->page_table = TpAlloc<uint32_t>(TP_PAGES);
	b->head_key = TpAlloc<uint64_t>(1u);
	b->head_best = TpAlloc<uint64_t>(16u);
	b->error = TpAlloc<LmKvAccessError>(1u);
	for (i = 0u; i < TP_TOP_K; i++)
		packed[i] = i;
	for (i = 0u; i < TP_PAGES; i++)
		pages[i] = i;
	TpCheck(cudaMemcpy(b->packed,packed,sizeof(packed),cudaMemcpyHostToDevice),"packed");
	TpCheck(cudaMemcpy(b->page_table,pages,sizeof(pages),cudaMemcpyHostToDevice),"pages");
	TpCheck(cudaMemcpy(b->sequence,&zero,sizeof(zero),cudaMemcpyHostToDevice),"sequence");
	b->embedding = TpPayload<uint16_t>(SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER,SPARK_MIMO26_STAGEPACK_TENSOR_EMBEDDING);
	b->final_norm = TpPayload<uint16_t>(SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER,SPARK_MIMO26_STAGEPACK_TENSOR_FINAL_NORM);
	b->lm_head = TpPayload<uint16_t>(SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER,SPARK_MIMO26_STAGEPACK_TENSOR_LM_HEAD);
}

static uint32_t TpHead(TpBuffers *b, float *score)
{
	uint64_t best;
	uint32_t ordered, bits;
	LmBf16RmsNormKernel<TP_THREADS><<<1,TP_THREADS,(TP_HIDDEN + 64u) * sizeof(float)>>>(b->stream,b->final_norm,b->normed,TP_HIDDEN,TP_HIDDEN,TP_EPS);
	TpLaunch(LmSkinnyDense<LmBf16Format>(b->lm_head,b->normed,0,b->logits,1u,TP_HIDDEN,TP_RANK_VOCAB,TP_RANK_VOCAB,0u,0),"lm_head",TP_LAYERS);
	TpHeadKeyKernel<<<1,1024>>>(b->logits,tp.rank * TP_RANK_VOCAB,b->head_key);
	TpCollective(SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64,b->head_key,b->head_best);
	TpCheck(cudaMemcpy(&best,b->head_best,sizeof(best),cudaMemcpyDeviceToHost),"head-best");
	ordered = (uint32_t)(best >> 32);
	bits = (ordered & 0x80000000u) != 0u ? ordered & 0x7FFFFFFFu : ~ordered;
	memcpy(score,&bits,sizeof(bits));
	return(0xFFFFFFFFu - (uint32_t)(best & 0xFFFFFFFFu));
}

static void TpDump(const char *directory, const char *name, uint32_t position, uint32_t layer, const void *device, size_t bytes)
{
	char path[1024];
	std::vector<uint8_t> host(bytes);
	FILE *file;
	TpCheck(cudaDeviceSynchronize(),"dump-sync");
	TpCheck(cudaMemcpy(host.data(),device,bytes,cudaMemcpyDeviceToHost),"dump-copy");
	snprintf(path,sizeof(path),"%s/rank%u_pos%04u_layer%04u_%s",directory,tp.rank,position,layer,name);
	file = fopen(path,"wb");
	if ( file == 0 || fwrite(host.data(),1u,bytes,file) != bytes || fclose(file) != 0 )
		TpFail("dump",path);
}

static std::vector<uint32_t> TpReadTokens(const char *path)
{
	std::vector<uint32_t> tokens;
	FILE *file = fopen(path,"rb");
	int32_t value;
	if ( file == 0 )
		TpFail("tokens",path);
	while ( fread(&value,sizeof(value),1u,file) == 1u )
	{
		if ( value < 0 || (uint32_t)value >= TP_VOCAB )
			TpFail("tokens","id out of range");
		tokens.push_back((uint32_t)value);
	}
	fclose(file);
	if ( tokens.empty() )
		TpFail("tokens","empty");
	return(tokens);
}

static void TpOpenPack(const char *path, const char *sha256, uint64_t expert_pool_bytes, uint64_t spine_budget)
{
	SparkMimo26StagePackHeader header;
	SparkWeightdLazyAttachRequest request;
	struct stat info;
	SparkStatus status;
	int fd = open(path,O_RDONLY);
	if ( fd < 0 || fstat(fd,&info) != 0 || pread(fd,&header,sizeof(header),0) != (ssize_t)sizeof(header) || header.tensor_count == 0u || header.tensor_count > 65536u )
		TpFail("pack-header",path);
	tp.entries = (SparkMimo26StagePackEntry *)calloc(header.tensor_count,sizeof(*tp.entries));
	if ( tp.entries == 0 || pread(fd,tp.entries,(size_t)header.tensor_count * sizeof(*tp.entries),(off_t)header.directory_offset) != (ssize_t)((size_t)header.tensor_count * sizeof(*tp.entries)) )
		TpFail("pack-directory",path);
	close(fd);
	TpStatus(SparkMimo26RankPackBind(&header,tp.entries,header.tensor_count,(uint64_t)info.st_size,TP_DEGREE,&tp.layout),"pack-bind");
	if ( strlen(sha256) != SPARK_WEIGHTD_SHA256_HEX_BYTES - 1u || strlen(path) >= sizeof(request.pack_path) )
		TpFail("pack-identity","sha256 must be 64 hex characters");
	memset(&request,0,sizeof(request));
	memcpy(request.identity.pack_sha256,sha256,SPARK_WEIGHTD_SHA256_HEX_BYTES - 1u);
	snprintf(request.identity.model,sizeof(request.identity.model),"%s",SPARK_MIMO26_TP_MODULE_TAG);
	snprintf(request.identity.revision,sizeof(request.identity.revision),"%s","mimo26flash.mxfp4.tp4.v2");
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (uint64_t)info.st_size;
	request.identity.topology = TP_DEGREE;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	request.expert_pool_bytes = expert_pool_bytes;
	status = SparkWeightdLazyPackCreateChecked(getenv("SPARK_WEIGHTD_SOCKET"),&request,spine_budget,TP_WAIT_NS,0,0,&tp.pack);
	TpStatus(status,"weightd-lazy-attach");
	printf("M26TP-ATTACH rank=%u spine_bytes=%llu expert_pool_bytes=%llu experts=%u\n",tp.rank,(unsigned long long)tp.pack->spine_allocation_bytes,(unsigned long long)tp.pack->attached.expert_pool_bytes,tp.pack->attached.expert_count);
	fflush(stdout);
}

static void TpOpenCollective(void)
{
	SparkTpDeviceCollectiveConfig configuration;
	pthread_condattr_t attributes;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = TP_DEGREE;
	configuration.tp_rank = tp.rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = 1u;
	configuration.local_hidden_dimension = TP_HIDDEN;
	configuration.max_active_sequence_count = 1u;
	configuration.connect_timeout_milli = 300000u;
	configuration.operation_timeout_milli = 300000u;
	SparkMimo26TpModuleRegisterCombines(&configuration);
	if ( pthread_mutex_init(&tp.completion.lock,0) != 0 || pthread_condattr_init(&attributes) != 0 || pthread_condattr_setclock(&attributes,CLOCK_MONOTONIC) != 0 || pthread_cond_init(&tp.completion.wake,&attributes) != 0 )
		TpFail("completion","init");
	TpCheck(cudaStreamCreateWithFlags(&tp.collective_stream,cudaStreamNonBlocking),"collective-stream");
	TpStatus(SparkTpDeviceCollectiveCreate(&configuration,&tp.collective),"collective-create");
	TpStatus(SparkTpDeviceCollectiveAttach(&tp.collective,0),"collective-attach");
}

static void TpBarrier(char expected)
{
	struct pollfd input = {STDIN_FILENO,POLLIN,0};
	char actual = 0;
	if ( poll(&input,1u,600000) != 1 || (input.revents & POLLIN) == 0 || read(STDIN_FILENO,&actual,1u) != 1 || actual != expected )
		TpFail("coordinator-barrier","no release from the coordinator");
}

int main(int argc, char **argv)
{
	static TpLayer layers[TP_LAYERS];
	TpBuffers b;
	uint32_t position, layer, token, context, mismatches = 0u, total;
	uint64_t decode_started = 0u, decode_ns = 0u;
	float score;
	LmKvAccessError access;
	const char *dump = argc == 9 ? argv[8] : 0;
	if ( argc != 8 && argc != 9 )
	{
		fprintf(stderr,"usage: %s RANK PACK PACK_SHA256 PROMPT_I32 EXPECTED_I32 EXPERT_POOL_BYTES SPINE_BUDGET_BYTES [DUMP_DIRECTORY]\n",argv[0]);
		return(2);
	}
	tp.rank = (uint32_t)strtoul(argv[1],0,10);
	if ( tp.rank >= TP_DEGREE || getenv("SPARK_WEIGHTD_SOCKET") == 0 || getenv("SPARK_WEIGHTD_LANE") == 0 || getenv("SPARK_TP_MESH_RANKS") == 0 )
	{
		fprintf(stderr,"M26TP-FAIL rank out of range or SPARK_WEIGHTD_SOCKET / SPARK_WEIGHTD_LANE / SPARK_TP_MESH_RANKS unset\n");
		return(2);
	}
	std::vector<uint32_t> prompt = TpReadTokens(argv[4]), expected = TpReadTokens(argv[5]), generated;
	total = (uint32_t)(prompt.size() + expected.size());
	if ( total > TP_MAX_POSITIONS || total > TP_WINDOW )
		TpFail("positions","prompt plus expected tokens exceed the 128-position validation cache");
	TpCheck(cudaFree(0),"cuda-init");
	TpOpenPack(argv[2],argv[3],strtoull(argv[6],0,10),strtoull(argv[7],0,10));
	TpAllocate(&b);
	for (layer = 0u; layer < TP_LAYERS; layer++)
		TpBindLayer(layer,&layers[layer],b.page_table,b.error);
	TpCheck(cudaDeviceSynchronize(),"bind");
	TpOpenCollective();
	printf("M26TP-READY rank=%u lane=%s mesh=%s prompt=%zu expected=%zu\n",tp.rank,getenv("SPARK_WEIGHTD_LANE"),getenv("SPARK_TP_MESH_RANKS"),prompt.size(),expected.size());
	fflush(stdout);
	TpBarrier('G');
	TpStatus(SparkTpDeviceCollectiveChainKey(&tp.collective,1u),"chain-key");
	for (position = 0u; position + 1u < total; position++)
	{
		if ( position + 1u == prompt.size() )
			decode_started = TpNow();
		token = position < prompt.size() ? prompt[position] : generated[position - prompt.size()];
		TpEmbedKernel<<<(TP_HIDDEN + 255u) / 256u,256u>>>(b.embedding,token,tp.rank * TP_RANK_VOCAB,b.partial);
		TpCheck(cudaMemset(b.stream,0,TP_HIDDEN * sizeof(uint16_t)),"stream-clear");
		TpReduceAdd(&b);
		if ( dump != 0 && position < 2u )
			TpDump(dump,"embed.bf16",position,0u,b.stream,TP_HIDDEN * sizeof(uint16_t));
		context = position + 1u;
		TpCheck(cudaMemcpy(b.position,&position,sizeof(position),cudaMemcpyHostToDevice),"position");
		TpCheck(cudaMemcpy(b.context,&context,sizeof(context),cudaMemcpyHostToDevice),"context");
		for (layer = 0u; layer < TP_LAYERS; layer++)
		{
			if ( layers[layer].swa != 0u )
				TpAttentionRank<TP_SWA_KV,TpSwaKv>(layer,&layers[layer],&b,position,SPARK_MIMO26_MODEL_SWA_ROPE_THETA);
			else
				TpAttentionRank<TP_FULL_KV,TpFullKv>(layer,&layers[layer],&b,position,SPARK_MIMO26_MODEL_FULL_ROPE_THETA);
			if ( layers[layer].moe != 0u )
				TpMoe(layer,&layers[layer],&b);
			else
				TpDense(layer,&layers[layer],&b);
			if ( dump != 0 && position < 2u )
			{
				TpDump(dump,"streams.bf16",position,layer,b.stream,TP_HIDDEN * sizeof(uint16_t));
				if ( layers[layer].moe != 0u )
					TpDump(dump,"route_ids.i32",position,layer,b.route_expert,TP_TOP_K * sizeof(uint32_t));
			}
		}
		token = TpHead(&b,&score);
		if ( position + 1u >= prompt.size() )
		{
			generated.push_back(token);
			mismatches += token != expected[position + 1u - prompt.size()] ? 1u : 0u;
			printf("M26TP-TOKEN rank=%u position=%u greedy=%u reference=%u score=%.4f\n",tp.rank,position,token,expected[position + 1u - prompt.size()],score);
			fflush(stdout);
		}
	}
	decode_ns = TpNow() - decode_started;
	TpCheck(cudaMemcpy(&access,b.error,sizeof(access),cudaMemcpyDeviceToHost),"kv-error");
	TpStatus(SparkTpDeviceCollectiveEndChain(&tp.collective,tp.collective_stream),"chain-end");
	printf("M26TP-SUMMARY rank=%u tokens=%zu mismatches=%u kv_error=%u collectives=%llu collective_ms=%.1f expert_ms=%.1f decode_ms_per_token=%.2f\n",tp.rank,generated.size(),mismatches,access.error_code,(unsigned long long)tp.reduce_count,(double)tp.reduce_ns / 1e6,(double)tp.expert_ns / 1e6,generated.empty() ? 0.0 : (double)decode_ns / 1e6 / (double)generated.size());
	if ( mismatches != 0u || access.error_code != 0u )
	{
		fprintf(stderr,"M26TP-FAIL rank=%u greedy tokens differ from the CPU reference\n",tp.rank);
		return(1);
	}
	printf("M26TP-PASS rank=%u %zu greedy tokens equal the CPU reference\n",tp.rank,generated.size());
	SparkTpDeviceCollectiveDestroy(&tp.collective);
	TpStatus(SparkWeightdLazyPackDestroy(tp.pack),"pack-destroy");
	return(0);
}
