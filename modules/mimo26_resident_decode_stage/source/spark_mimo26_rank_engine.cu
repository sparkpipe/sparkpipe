#include <fcntl.h>
#include <math.h>
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
#include "spark_mimo26_rank_engine.h"

extern "C" SparkStatus SparkStageModuleCudaStatus(const char *module_tag, cudaError_t error, const char *site);

#define SPARK_FAMILY_CAMEL Mimo26Rank
#define SPARK_FAMILY_UPPER MIMO26_RANK
#define SPARK_FAMILY_LOWER mimo26_rank
#include "sparkpipe/family/spark_family.h"
#define SPARK_MIMO26_RANK_MODULE_TAG "mimo26_rank_engine"
#include "sparkpipe/family/module/spark_module_combine.h"

#define RE_HIDDEN SPARK_MIMO26_MODEL_HIDDEN_DIMENSION
#define RE_LAYERS SPARK_MIMO26_MODEL_LAYER_COUNT
#define RE_VOCAB SPARK_MIMO26_MODEL_VOCAB_COUNT
#define RE_HEADS SPARK_MIMO26_MODEL_ATTN_HEAD_COUNT
#define RE_HEAD_DIM SPARK_MIMO26_MODEL_ATTN_HEAD_DIMENSION
#define RE_VALUE_DIM SPARK_MIMO26_MODEL_ATTN_VALUE_DIMENSION
#define RE_ROPE_DIM SPARK_MIMO26_MODEL_ATTN_ROPE_DIMENSION
#define RE_FULL_KV SPARK_MIMO26_MODEL_FULL_KV_HEAD_COUNT
#define RE_SWA_KV SPARK_MIMO26_MODEL_SWA_KV_HEAD_COUNT
#define RE_EXPERTS SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT
#define RE_TOP_K SPARK_MIMO26_MODEL_EXPERTS_PER_TOKEN
#define RE_INTER SPARK_MIMO26_MODEL_EXPERT_INTERMEDIATE_DIMENSION
#define RE_DENSE SPARK_MIMO26_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define RE_EPS SPARK_MIMO26_MODEL_RMS_NORM_EPSILON
#define RE_VALUE_SCALE SPARK_MIMO26_MODEL_ATTN_VALUE_SCALE
#define RE_WINDOW SPARK_MIMO26_MODEL_SLIDING_WINDOW_TOKENS
#define RE_FP8_BLOCK SPARK_MIMO26_MODEL_FP8_SCALE_BLOCK
#define RE_DEGREE 4u
#define RE_THREADS 256u
#define RE_PAGE_SLOTS 64u
#define RE_RANK_HEADS (RE_HEADS / RE_DEGREE)
#define RE_RANK_O_INPUT (SPARK_MIMO26_MODEL_O_INPUT_DIMENSION / RE_DEGREE)
#define RE_RANK_VOCAB (RE_VOCAB / RE_DEGREE)
#define RE_RANK_EXPERTS (RE_EXPERTS / RE_DEGREE)
#define RE_RANK_DENSE (RE_DENSE / RE_DEGREE)
#define RE_EXPERT_ROWS (RE_RANK_DENSE > RE_TOP_K * RE_INTER ? RE_RANK_DENSE : RE_TOP_K * RE_INTER)
#define RE_MAX_LEASES 16u

using ReFullKv = LmKvGeometry<(RE_FULL_KV / RE_DEGREE) * (RE_HEAD_DIM + RE_VALUE_DIM) * sizeof(uint16_t), RE_PAGE_SLOTS, true>;
using ReSwaKv = LmKvGeometry<(RE_SWA_KV / RE_DEGREE) * (RE_HEAD_DIM + RE_VALUE_DIM) * sizeof(uint16_t), RE_PAGE_SLOTS, true>;

typedef struct ReLayer
{
	uint32_t swa, moe, qkv_rows;
	const uint16_t *attn_norm, *mlp_norm, *sink, *o_proj, *router;
	const uint8_t *qkv, *dense[3], *expert[3], *expert_scale[3];
	const float *router_bias;
	float *qkv_scale, *dense_scale[3];
	uint64_t expert_payload[3], expert_scale_offset[3];
	uint8_t *pool;
	LmKvView cache;
} ReLayer;

struct SparkMimo26RankEngine
{
	SparkMimo26RankEngineConfig config;
	SparkWeightdLazyPack *pack;
	SparkMimo26StagePackEntry *entries;
	SparkMimo26RankPackLayout layout;
	SparkTpDeviceCollective collective;
	uint32_t collective_ready;
	cudaStream_t stream;
	cudaGraphExec_t graph;
	uint64_t leases[RE_MAX_LEASES];
	uint32_t lease_count, pinned_experts;
	const uint8_t *expert_base;
	uint64_t ordinal, chain, collectives, steps, graph_steps, step_ns;
	uint32_t pages_per_lane, dump_position, walking_dump;
	ReLayer layers[RE_LAYERS];
	uint16_t *stream_bf16, *normed, *fused, *query, *key, *value, *attended, *partial, *reduced, *gate, *up, *act, *down;
	float *logits, *route_weight;
	uint32_t *route_expert, *row_of_k, *group_offset, *source_token, *sequence, *context, *position, *page_table, *window, *step_in, *step_out;
	uint32_t *host_in, *host_out;
	uint64_t *head_key, *head_best;
	LmKvAccessError *error;
	const uint16_t *embedding, *final_norm, *lm_head;
};

#define RE_TRY(expression) do { SparkStatus re_status_ = (expression); if ( re_status_ != SPARK_STATUS_OK ) return(re_status_); } while ( 0 )
#define RE_CUDA(expression, site) do { cudaError_t re_error_ = (expression); if ( re_error_ != cudaSuccess ) { fprintf(stderr,"M26RE-CUDA rank=%u site=%s %s\n",engine->config.rank,site,cudaGetErrorString(re_error_)); return(SPARK_STATUS_IO_ERROR); } } while ( 0 )
#define RE_LAUNCH(expression, site, layer) do { int32_t re_launch_ = (expression); if ( re_launch_ != LM_LAUNCH_OK ) { fprintf(stderr,"M26RE-LAUNCH rank=%u site=%s layer=%u code=%d\n",engine->config.rank,site,(unsigned)(layer),re_launch_); return(SPARK_STATUS_INTERNAL_ERROR); } } while ( 0 )
#define RE_KERNEL(site) RE_CUDA(cudaPeekAtLastError(),site)

static uint64_t ReNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

template<class T>
static SparkStatus ReAlloc(SparkMimo26RankEngine *engine, size_t count, T **out)
{
	void *device = 0;
	RE_CUDA(cudaMalloc(&device,count * sizeof(T)),"alloc");
	RE_CUDA(cudaMemsetAsync(device,0,count * sizeof(T),engine->stream),"clear");
	*out = (T *)device;
	return(SPARK_STATUS_OK);
}

static const SparkMimo26StagePackEntry *ReEntry(SparkMimo26RankEngine *engine, uint32_t layer, uint32_t kind)
{
	uint32_t index = layer == SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER ? engine->layout.global_entry[kind] : engine->layout.layer_entry[layer][kind];
	return(index == SPARK_MIMO26_RANK_PACK_ABSENT ? 0 : &engine->entries[index]);
}

template<class T>
static SparkStatus RePayload(SparkMimo26RankEngine *engine, uint32_t layer, uint32_t kind, const T **out)
{
	const SparkMimo26StagePackEntry *entry = ReEntry(engine,layer,kind);
	const void *pointer = 0;
	if ( entry == 0 )
	{
		fprintf(stderr,"M26RE-ENTRY rank=%u layer=%u kind=%u absent\n",engine->config.rank,layer,kind);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	RE_TRY(SparkWeightdLazyPackSlice(engine->pack,entry->payload_offset,entry->payload_bytes,&pointer));
	*out = (const T *)pointer;
	return(SPARK_STATUS_OK);
}

__global__ void ReExpandBlockScaleKernel(const float *block, float *rows_out, uint32_t rows, uint32_t column_blocks)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < rows * column_blocks )
		rows_out[index] = block[(index / column_blocks / RE_FP8_BLOCK) * column_blocks + index % column_blocks];
}

static SparkStatus ReRowScales(SparkMimo26RankEngine *engine, uint32_t layer, uint32_t kind, float **out)
{
	const SparkMimo26StagePackEntry *entry = ReEntry(engine,layer,kind);
	const void *block = 0;
	uint32_t column_blocks;
	if ( entry == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	column_blocks = entry->columns / RE_FP8_BLOCK;
	RE_TRY(SparkWeightdLazyPackSlice(engine->pack,entry->scale_offset,entry->scale_bytes,&block));
	RE_TRY(ReAlloc<float>(engine,(size_t)entry->rows * column_blocks,out));
	ReExpandBlockScaleKernel<<<(entry->rows * column_blocks + 255u) / 256u,256u,0,engine->stream>>>((const float *)block,*out,entry->rows,column_blocks);
	RE_KERNEL("expand-scale");
	return(SPARK_STATUS_OK);
}

__global__ void RePrepareKernel(const uint32_t *step_in, uint32_t *sequence, uint32_t *position, uint32_t *context)
{
	if ( threadIdx.x == 0u )
	{
		sequence[0] = step_in[2];
		position[0] = step_in[1];
		context[step_in[2]] = step_in[1] + 1u;
	}
}

template<uint32_t HEADS, uint32_t KV, uint32_t QKV>
__global__ void ReSplitQkvKernel(const uint16_t *fused, uint16_t *query, uint16_t *key, uint16_t *value, const uint32_t *position_of_row, float theta)
{
	const uint32_t q_local = HEADS * RE_HEAD_DIM, k_local = KV * RE_HEAD_DIM;
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x, local, dim, pair;
	float angle, first, second;
	uint16_t *target;
	if ( row >= QKV )
		return;
	local = row;
	if ( local >= q_local + k_local )
	{
		value[local - q_local - k_local] = LmFloatToBf16(LmBf16ToFloat(fused[row]) * RE_VALUE_SCALE);
		return;
	}
	target = local < q_local ? query : key;
	local = local < q_local ? local : local - q_local;
	dim = local % RE_HEAD_DIM;
	if ( dim >= RE_ROPE_DIM )
	{
		target[local] = fused[row];
		return;
	}
	pair = dim < RE_ROPE_DIM / 2u ? dim + RE_ROPE_DIM / 2u : dim - RE_ROPE_DIM / 2u;
	angle = (float)position_of_row[0] * powf(theta,-(2.0f * (float)(dim % (RE_ROPE_DIM / 2u))) / (float)RE_ROPE_DIM);
	first = LmBf16ToFloat(fused[row]);
	second = LmBf16ToFloat(fused[row - dim + pair]);
	target[local] = LmFloatToBf16(dim < RE_ROPE_DIM / 2u ? first * cosf(angle) - second * sinf(angle) : first * cosf(angle) + second * sinf(angle));
}

__global__ void ReAddKernel(const uint16_t *right, uint16_t *stream, uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		stream[index] = LmFloatToBf16(LmBf16ToFloat(stream[index]) + LmBf16ToFloat(right[index]));
}

__global__ void ReSwigluKernel(const uint16_t *gate, const uint16_t *up, uint16_t *out, uint32_t count)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	float g;
	if ( index >= count )
		return;
	g = LmBf16ToFloat(gate[index]);
	out[index] = LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(g / (1.0f + __expf(-g)))) * LmBf16ToFloat(up[index]));
}

__global__ void ReLocalRouteKernel(const uint32_t *expert, uint32_t rank, uint32_t *group_offset, uint32_t *row_of_k)
{
	uint32_t group = threadIdx.x, k, e, below = 0u;
	if ( group > RE_RANK_EXPERTS )
		return;
	for (k = 0u; k < RE_TOP_K; k++)
	{
		e = expert[k];
		if ( e / RE_RANK_EXPERTS == rank && e % RE_RANK_EXPERTS < group )
			below++;
	}
	group_offset[group] = below;
	if ( group < RE_TOP_K )
	{
		e = expert[group];
		if ( e / RE_RANK_EXPERTS != rank )
			row_of_k[group] = UINT32_MAX;
		else
		{
			below = 0u;
			for (k = 0u; k < RE_TOP_K; k++)
				if ( expert[k] / RE_RANK_EXPERTS == rank && expert[k] % RE_RANK_EXPERTS < e % RE_RANK_EXPERTS )
					below++;
			row_of_k[group] = below;
		}
	}
}

__global__ void ReCombineKernel(const uint16_t *down, const float *weights, const uint32_t *row_of_k, uint16_t *partial)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x, k;
	float total = 0.0f;
	if ( index >= RE_HIDDEN )
		return;
	for (k = 0u; k < RE_TOP_K; k++)
		if ( row_of_k[k] != UINT32_MAX )
			total += weights[k] * LmBf16ToFloat(down[row_of_k[k] * RE_HIDDEN + index]);
	partial[index] = LmFloatToBf16(total);
}

__global__ void ReEmbedKernel(const uint16_t *embedding, const uint32_t *step_in, uint32_t first_row, uint16_t *out)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x, token = step_in[0];
	if ( index >= RE_HIDDEN )
		return;
	out[index] = token >= first_row && token < first_row + RE_RANK_VOCAB ? embedding[(size_t)(token - first_row) * RE_HIDDEN + index] : (uint16_t)0u;
}

__global__ void ReHeadKeyKernel(const float *logits, uint32_t first_row, uint64_t *key_out)
{
	__shared__ uint64_t best[1024];
	uint32_t index, bits, ordered;
	uint64_t mine = 0u, candidate;
	for (index = threadIdx.x; index < RE_RANK_VOCAB; index += blockDim.x)
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

__global__ void ReResolveKernel(const uint64_t *best, uint32_t *step_out)
{
	uint32_t ordered, bits;
	if ( threadIdx.x != 0u )
		return;
	ordered = (uint32_t)(best[0] >> 32);
	bits = (ordered & 0x80000000u) != 0u ? ordered & 0x7FFFFFFFu : ~ordered;
	step_out[0] = 0xFFFFFFFFu - (uint32_t)(best[0] & 0xFFFFFFFFu);
	step_out[1] = bits;
}

static SparkStatus ReCollective(SparkMimo26RankEngine *engine, uint32_t operation, const void *local, void *full)
{
	SparkTpDeviceCollectiveSubmission submission;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.active_sequence_count = 1u;
	submission.logical_sequence_count = 1u;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = engine->ordinal++;
	submission.local_device = local;
	submission.full_device = full;
	submission.cuda_stream = engine->stream;
	engine->collectives++;
	return(SparkTpDeviceCollectiveEnqueue(&engine->collective,&submission,operation));
}

static SparkStatus ReReduceAdd(SparkMimo26RankEngine *engine)
{
	RE_TRY(ReCollective(engine,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,engine->partial,engine->reduced));
	ReAddKernel<<<(RE_HIDDEN + 255u) / 256u,256u,0,engine->stream>>>(engine->reduced,engine->stream_bf16,RE_HIDDEN);
	RE_KERNEL("reduce-add");
	return(SPARK_STATUS_OK);
}

static SparkStatus ReDump(SparkMimo26RankEngine *engine, const char *name, uint32_t layer, const void *device, size_t bytes)
{
	char path[1024];
	std::vector<uint8_t> host(bytes);
	FILE *file;
	if ( engine->walking_dump == 0u )
		return(SPARK_STATUS_OK);
	RE_CUDA(cudaStreamSynchronize(engine->stream),"dump-sync");
	RE_CUDA(cudaMemcpy(host.data(),device,bytes,cudaMemcpyDeviceToHost),"dump-copy");
	snprintf(path,sizeof(path),"%s/rank%u_pos%04u_layer%04u_%s",engine->config.dump_directory,engine->config.rank,engine->dump_position,layer,name);
	file = fopen(path,"wb");
	if ( file == 0 || fwrite(host.data(),1u,bytes,file) != bytes || fclose(file) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

template<uint32_t KV, class Geometry>
static SparkStatus ReAttention(SparkMimo26RankEngine *engine, uint32_t index, ReLayer *layer, float theta)
{
	const uint32_t rank_kv = KV / RE_DEGREE, rows = RE_RANK_HEADS * RE_HEAD_DIM + rank_kv * (RE_HEAD_DIM + RE_VALUE_DIM);
	cudaStream_t s = engine->stream;
	float scale = 1.0f / sqrtf((float)RE_HEAD_DIM);
	if ( rows != layer->qkv_rows )
		return(SPARK_STATUS_VALIDATION_FAILED);
	LmBf16RmsNormKernel<RE_THREADS><<<1,RE_THREADS,(RE_HIDDEN + 64u) * sizeof(float),s>>>(engine->stream_bf16,layer->attn_norm,engine->normed,RE_HIDDEN,RE_HIDDEN,RE_EPS);
	RE_KERNEL("attn-norm");
	RE_LAUNCH(LmSkinnyExperts<LmFp8>(layer->qkv,LmScaleTensorBuild(layer->qkv_scale,LM_SCALE_ENCODING_F32,1u,rows,RE_HIDDEN,1u,RE_FP8_BLOCK),engine->normed,engine->fused,engine->source_token,engine->source_token,1u,1u,0u,RE_HIDDEN,rows,s),"qkv",index);
	ReSplitQkvKernel<RE_RANK_HEADS,KV / RE_DEGREE,RE_RANK_HEADS * RE_HEAD_DIM + (KV / RE_DEGREE) * (RE_HEAD_DIM + RE_VALUE_DIM)><<<(rows + 255u) / 256u,256u,0,s>>>(engine->fused,engine->query,engine->key,engine->value,engine->position,theta);
	RE_KERNEL("split-qkv");
	LmGqaKvStoreKernel<Geometry,RE_THREADS,KV / RE_DEGREE,RE_HEAD_DIM,RE_VALUE_DIM><<<1,RE_THREADS,0,s>>>(layer->cache,engine->key,engine->value,engine->sequence,engine->position,1u);
	RE_KERNEL("kv-store");
	if ( layer->swa != 0u )
	{
		LmBuildSlidingWindowPositionsKernel<128u><<<1,128u,0,s>>>(engine->sequence,engine->context,engine->position,1u,RE_WINDOW,engine->window);
		RE_KERNEL("window");
		LmGqaSinkAttentionDecodeKernel<Geometry,RE_THREADS,KV / RE_DEGREE,RE_HEAD_DIM,RE_VALUE_DIM><<<dim3(1u,RE_RANK_HEADS),RE_THREADS,0,s>>>(engine->query,layer->cache,engine->sequence,engine->context,engine->window,RE_WINDOW,RE_RANK_HEADS,scale,engine->attended,engine->position,layer->sink);
	}
	else
		LmGqaAttentionDecodeKernel<Geometry,RE_THREADS,KV / RE_DEGREE,RE_HEAD_DIM,RE_VALUE_DIM><<<dim3(1u,RE_RANK_HEADS),RE_THREADS,0,s>>>(engine->query,layer->cache,engine->sequence,engine->context,0,0u,RE_RANK_HEADS,scale,engine->attended,engine->position);
	RE_KERNEL("attention");
	RE_LAUNCH(LmSkinnyDense<LmBf16Format>(layer->o_proj,engine->attended,engine->partial,0,1u,RE_RANK_O_INPUT,RE_HIDDEN,RE_HIDDEN,0u,s),"o_proj",index);
	RE_TRY(ReReduceAdd(engine));
	LmBf16RmsNormKernel<RE_THREADS><<<1,RE_THREADS,(RE_HIDDEN + 64u) * sizeof(float),s>>>(engine->stream_bf16,layer->mlp_norm,engine->normed,RE_HIDDEN,RE_HIDDEN,RE_EPS);
	RE_KERNEL("mlp-norm");
	return(SPARK_STATUS_OK);
}

static SparkStatus ReDense(SparkMimo26RankEngine *engine, uint32_t index, ReLayer *layer)
{
	cudaStream_t s = engine->stream;
	RE_LAUNCH(LmSkinnyExperts<LmFp8>(layer->dense[0],LmScaleTensorBuild(layer->dense_scale[0],LM_SCALE_ENCODING_F32,1u,RE_RANK_DENSE,RE_HIDDEN,1u,RE_FP8_BLOCK),engine->normed,engine->gate,engine->source_token,engine->source_token,1u,1u,0u,RE_HIDDEN,RE_RANK_DENSE,s),"dense-gate",index);
	RE_LAUNCH(LmSkinnyExperts<LmFp8>(layer->dense[1],LmScaleTensorBuild(layer->dense_scale[1],LM_SCALE_ENCODING_F32,1u,RE_RANK_DENSE,RE_HIDDEN,1u,RE_FP8_BLOCK),engine->normed,engine->up,engine->source_token,engine->source_token,1u,1u,0u,RE_HIDDEN,RE_RANK_DENSE,s),"dense-up",index);
	ReSwigluKernel<<<(RE_RANK_DENSE + 255u) / 256u,256u,0,s>>>(engine->gate,engine->up,engine->act,RE_RANK_DENSE);
	RE_KERNEL("dense-swiglu");
	RE_LAUNCH(LmSkinnyExperts<LmFp8>(layer->dense[2],LmScaleTensorBuild(layer->dense_scale[2],LM_SCALE_ENCODING_F32,1u,RE_HIDDEN,RE_RANK_DENSE,1u,RE_FP8_BLOCK),engine->act,engine->partial,engine->source_token,engine->source_token,1u,1u,0u,RE_RANK_DENSE,RE_HIDDEN,s),"dense-down",index);
	return(ReReduceAdd(engine));
}

static SparkStatus ReMoe(SparkMimo26RankEngine *engine, uint32_t index, ReLayer *layer)
{
	cudaStream_t s = engine->stream;
	RE_LAUNCH(LmSkinnyDense<LmBf16Format>(layer->router,engine->normed,0,engine->logits,1u,RE_HIDDEN,RE_EXPERTS,RE_EXPERTS,0u,s),"router",index);
	LmTopkSmallKernel<RE_THREADS,RE_TOP_K,true,1u,1u,LM_TOPK_SCORE_SIGMOID><<<1,RE_THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),s>>>(engine->logits,RE_EXPERTS,engine->route_expert,engine->route_weight,layer->router_bias,0,SPARK_MIMO26_MODEL_ROUTED_SCALING_FACTOR);
	RE_KERNEL("topk");
	ReLocalRouteKernel<<<1,RE_RANK_EXPERTS + 1u,0,s>>>(engine->route_expert,engine->config.rank,engine->group_offset,engine->row_of_k);
	RE_KERNEL("local-route");
	RE_LAUNCH(LmSkinnyGroupedExperts<LmMxfp4>(layer->expert[0],LmScaleTensorBlockUe8m0(layer->expert_scale[0],RE_RANK_EXPERTS,RE_INTER,RE_HIDDEN,1u,32u),engine->normed,engine->gate,engine->group_offset,engine->source_token,RE_RANK_EXPERTS,RE_TOP_K,0u,RE_HIDDEN,RE_INTER,s),"expert-gate",index);
	RE_LAUNCH(LmSkinnyGroupedExperts<LmMxfp4>(layer->expert[1],LmScaleTensorBlockUe8m0(layer->expert_scale[1],RE_RANK_EXPERTS,RE_INTER,RE_HIDDEN,1u,32u),engine->normed,engine->up,engine->group_offset,engine->source_token,RE_RANK_EXPERTS,RE_TOP_K,0u,RE_HIDDEN,RE_INTER,s),"expert-up",index);
	ReSwigluKernel<<<(RE_TOP_K * RE_INTER + 255u) / 256u,256u,0,s>>>(engine->gate,engine->up,engine->act,RE_TOP_K * RE_INTER);
	RE_KERNEL("expert-swiglu");
	RE_LAUNCH(LmSkinnyGroupedExperts<LmMxfp4>(layer->expert[2],LmScaleTensorBlockUe8m0(layer->expert_scale[2],RE_RANK_EXPERTS,RE_HIDDEN,RE_INTER,1u,32u),engine->act,engine->down,engine->group_offset,engine->source_token,RE_RANK_EXPERTS,RE_TOP_K,1u,RE_INTER,RE_HIDDEN,s),"expert-down",index);
	ReCombineKernel<<<(RE_HIDDEN + 255u) / 256u,256u,0,s>>>(engine->down,engine->route_weight,engine->row_of_k,engine->partial);
	RE_KERNEL("combine");
	return(ReReduceAdd(engine));
}

static SparkStatus ReWalk(SparkMimo26RankEngine *engine)
{
	cudaStream_t s = engine->stream;
	uint32_t layer;
	RE_CUDA(cudaMemcpyAsync(engine->step_in,engine->host_in,3u * sizeof(uint32_t),cudaMemcpyHostToDevice,s),"step-in");
	RePrepareKernel<<<1,32,0,s>>>(engine->step_in,engine->sequence,engine->position,engine->context);
	RE_KERNEL("prepare");
	ReEmbedKernel<<<(RE_HIDDEN + 255u) / 256u,256u,0,s>>>(engine->embedding,engine->step_in,engine->config.rank * RE_RANK_VOCAB,engine->partial);
	RE_KERNEL("embed");
	RE_CUDA(cudaMemsetAsync(engine->stream_bf16,0,RE_HIDDEN * sizeof(uint16_t),s),"stream-clear");
	RE_TRY(ReReduceAdd(engine));
	RE_TRY(ReDump(engine,"embed.bf16",0u,engine->stream_bf16,RE_HIDDEN * sizeof(uint16_t)));
	for (layer = 0u; layer < RE_LAYERS; layer++)
	{
		ReLayer *current = &engine->layers[layer];
		if ( current->swa != 0u )
			RE_TRY((ReAttention<RE_SWA_KV,ReSwaKv>(engine,layer,current,SPARK_MIMO26_MODEL_SWA_ROPE_THETA)));
		else
			RE_TRY((ReAttention<RE_FULL_KV,ReFullKv>(engine,layer,current,SPARK_MIMO26_MODEL_FULL_ROPE_THETA)));
		if ( current->moe != 0u )
			RE_TRY(ReMoe(engine,layer,current));
		else
			RE_TRY(ReDense(engine,layer,current));
		RE_TRY(ReDump(engine,"streams.bf16",layer,engine->stream_bf16,RE_HIDDEN * sizeof(uint16_t)));
		if ( current->moe != 0u )
			RE_TRY(ReDump(engine,"route_ids.i32",layer,engine->route_expert,RE_TOP_K * sizeof(uint32_t)));
	}
	LmBf16RmsNormKernel<RE_THREADS><<<1,RE_THREADS,(RE_HIDDEN + 64u) * sizeof(float),s>>>(engine->stream_bf16,engine->final_norm,engine->normed,RE_HIDDEN,RE_HIDDEN,RE_EPS);
	RE_KERNEL("final-norm");
	RE_LAUNCH(LmSkinnyDense<LmBf16Format>(engine->lm_head,engine->normed,0,engine->logits,1u,RE_HIDDEN,RE_RANK_VOCAB,RE_RANK_VOCAB,0u,s),"lm_head",RE_LAYERS);
	ReHeadKeyKernel<<<1,1024,0,s>>>(engine->logits,engine->config.rank * RE_RANK_VOCAB,engine->head_key);
	RE_KERNEL("head-key");
	RE_TRY(ReCollective(engine,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64,engine->head_key,engine->head_best));
	ReResolveKernel<<<1,32,0,s>>>(engine->head_best,engine->step_out);
	RE_KERNEL("resolve");
	RE_CUDA(cudaMemcpyAsync(engine->host_out,engine->step_out,2u * sizeof(uint32_t),cudaMemcpyDeviceToHost,s),"step-out");
	return(SPARK_STATUS_OK);
}

static SparkStatus ReWaitStream(SparkMimo26RankEngine *engine)
{
	uint64_t deadline = ReNow() + engine->config.wait_ns;
	cudaError_t status;
	for (;;)
	{
		status = cudaStreamQuery(engine->stream);
		if ( status == cudaSuccess )
			return(SPARK_STATUS_OK);
		if ( status != cudaErrorNotReady )
		{
			fprintf(stderr,"M26RE-STREAM rank=%u %s\n",engine->config.rank,cudaGetErrorString(status));
			return(SPARK_STATUS_IO_ERROR);
		}
		if ( ReNow() >= deadline )
		{
			fprintf(stderr,"M26RE-STREAM-TIMEOUT rank=%u progress=%llu\n",engine->config.rank,(unsigned long long)SparkTpDeviceCollectiveGraphProgress(&engine->collective,0));
			SparkTpDeviceCollectiveBroadcastCancel(&engine->collective);
			return(SPARK_STATUS_IO_ERROR);
		}
	}
}

static SparkStatus ReBindLayer(SparkMimo26RankEngine *engine, uint32_t index)
{
	ReLayer *layer = &engine->layers[index];
	const SparkMimo26StagePackEntry *entry;
	size_t slot_bytes;
	uint32_t p;
	const uint32_t dense_kinds[3] = {SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_GATE,SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_UP,SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_DOWN};
	const uint32_t expert_kinds[3] = {SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_GATE,SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_UP,SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_DOWN};
	memset(layer,0,sizeof(*layer));
	layer->swa = SparkMimo26ModelLayerKind(index) == SPARK_MIMO26_MODEL_LAYER_KIND_SWA ? 1u : 0u;
	layer->moe = SparkMimo26ModelLayerIsMoe(index);
	RE_TRY(RePayload<uint16_t>(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_ATTENTION_NORM,&layer->attn_norm));
	RE_TRY(RePayload<uint16_t>(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_MLP_NORM,&layer->mlp_norm));
	RE_TRY(RePayload<uint16_t>(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_O_PROJ,&layer->o_proj));
	RE_TRY(RePayload<uint8_t>(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_QKV,&layer->qkv));
	RE_TRY(ReRowScales(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_QKV,&layer->qkv_scale));
	layer->qkv_rows = ReEntry(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_QKV)->rows;
	if ( layer->swa != 0u )
		RE_TRY(RePayload<uint16_t>(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_SINK_BIAS,&layer->sink));
	if ( layer->moe != 0u )
	{
		RE_TRY(RePayload<uint16_t>(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE,&layer->router));
		RE_TRY(RePayload<float>(engine,index,SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE_BIAS,&layer->router_bias));
		for (p = 0u; p < 3u; p++)
		{
			entry = ReEntry(engine,index,expert_kinds[p]);
			if ( entry == 0 )
				return(SPARK_STATUS_VALIDATION_FAILED);
			layer->expert_payload[p] = entry->payload_offset;
			layer->expert_scale_offset[p] = entry->scale_offset;
		}
	}
	else
		for (p = 0u; p < 3u; p++)
		{
			RE_TRY(RePayload<uint8_t>(engine,index,dense_kinds[p],&layer->dense[p]));
			RE_TRY(ReRowScales(engine,index,dense_kinds[p],&layer->dense_scale[p]));
		}
	slot_bytes = layer->swa != 0u ? ReSwaKv::kSlotBytes : ReFullKv::kSlotBytes;
	RE_TRY(ReAlloc<uint8_t>(engine,(size_t)engine->config.lane_count * engine->config.max_positions * slot_bytes,&layer->pool));
	if ( LmKvViewInitialize(&layer->cache,layer->pool,engine->page_table,engine->pages_per_lane,engine->config.lane_count,engine->pages_per_lane * engine->config.lane_count,engine->error) != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus RePinExperts(SparkMimo26RankEngine *engine)
{
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	uint32_t layer, expert, count = 0u, p;
	void *base = 0;
	for (layer = 0u; layer < RE_LAYERS; layer++)
	{
		if ( SparkMimo26ModelLayerIsMoe(layer) == 0u )
			continue;
		for (expert = 0u; expert < RE_RANK_EXPERTS; expert++)
		{
			keys[count].layer = layer;
			keys[count].expert = expert;
			count++;
			if ( count == SPARK_WEIGHTD_LEASE_GROUPS_MAX || (layer + 1u == RE_LAYERS && expert + 1u == RE_RANK_EXPERTS) )
			{
				uint64_t lease = 0u;
				SparkStatus status;
				if ( engine->lease_count == RE_MAX_LEASES )
					return(SPARK_STATUS_CAPACITY_EXCEEDED);
				status = SparkWeightdMapAcquire(engine->pack->map,keys,count,&lease,engine->config.wait_ns);
				if ( lease != 0u )
					engine->leases[engine->lease_count++] = lease;
				if ( status != SPARK_STATUS_OK )
				{
					fprintf(stderr,"M26RE-PIN rank=%u layer=%u acquired=%u status=%s\n",engine->config.rank,layer,engine->pinned_experts,SparkStatusToString(status));
					return(status);
				}
				RE_TRY(SparkWeightdMapBeginUse(engine->pack->map,lease,&base));
				if ( base == 0 || (engine->expert_base != 0 && engine->expert_base != (const uint8_t *)base) )
					return(SPARK_STATUS_VALIDATION_FAILED);
				engine->expert_base = (const uint8_t *)base;
				engine->pinned_experts += count;
				count = 0u;
			}
		}
	}
	for (layer = 0u; layer < RE_LAYERS; layer++)
		if ( engine->layers[layer].moe != 0u )
			for (p = 0u; p < 3u; p++)
			{
				engine->layers[layer].expert[p] = engine->expert_base + engine->layers[layer].expert_payload[p];
				engine->layers[layer].expert_scale[p] = engine->expert_base + engine->layers[layer].expert_scale_offset[p];
			}
	return(SPARK_STATUS_OK);
}

static SparkStatus ReOpenPack(SparkMimo26RankEngine *engine)
{
	SparkMimo26StagePackHeader header;
	SparkWeightdLazyAttachRequest request;
	struct stat info;
	const char *path = engine->config.pack_path, *sha256 = engine->config.pack_sha256;
	int fd = open(path,O_RDONLY);
	ssize_t directory_bytes;
	if ( fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( fstat(fd,&info) != 0 || pread(fd,&header,sizeof(header),0) != (ssize_t)sizeof(header) || header.tensor_count == 0u || header.tensor_count > 65536u )
	{
		close(fd);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	engine->entries = (SparkMimo26StagePackEntry *)calloc(header.tensor_count,sizeof(*engine->entries));
	directory_bytes = (ssize_t)((size_t)header.tensor_count * sizeof(*engine->entries));
	if ( engine->entries == 0 || pread(fd,engine->entries,(size_t)directory_bytes,(off_t)header.directory_offset) != directory_bytes )
	{
		close(fd);
		return(SPARK_STATUS_IO_ERROR);
	}
	close(fd);
	RE_TRY(SparkMimo26RankPackBind(&header,engine->entries,header.tensor_count,(uint64_t)info.st_size,RE_DEGREE,&engine->layout));
	if ( sha256 == 0 || strlen(sha256) != 64u || strlen(path) >= sizeof(request.pack_path) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&request,0,sizeof(request));
	memcpy(request.identity.pack_sha256,sha256,64u);
	snprintf(request.identity.model,sizeof(request.identity.model),"%s","mimo26_tp_decode");
	snprintf(request.identity.revision,sizeof(request.identity.revision),"%s","mimo26flash.mxfp4.tp4.v2");
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (uint64_t)info.st_size;
	request.identity.topology = RE_DEGREE;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	request.expert_pool_bytes = engine->config.expert_pool_bytes;
	return(SparkWeightdLazyPackCreateChecked(getenv("SPARK_WEIGHTD_SOCKET"),&request,engine->config.spine_budget_bytes,engine->config.wait_ns,0,0,&engine->pack));
}

static SparkStatus ReAllocate(SparkMimo26RankEngine *engine)
{
	std::vector<uint32_t> pages;
	uint32_t i;
	engine->pages_per_lane = engine->config.max_positions / RE_PAGE_SLOTS;
	RE_TRY(ReAlloc<uint16_t>(engine,RE_HIDDEN,&engine->stream_bf16));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_HIDDEN,&engine->normed));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_RANK_HEADS * RE_HEAD_DIM + (RE_SWA_KV / RE_DEGREE) * (RE_HEAD_DIM + RE_VALUE_DIM),&engine->fused));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_RANK_HEADS * RE_HEAD_DIM,&engine->query));
	RE_TRY(ReAlloc<uint16_t>(engine,(RE_SWA_KV / RE_DEGREE) * RE_HEAD_DIM,&engine->key));
	RE_TRY(ReAlloc<uint16_t>(engine,(RE_SWA_KV / RE_DEGREE) * RE_VALUE_DIM,&engine->value));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_RANK_O_INPUT,&engine->attended));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_HIDDEN,&engine->partial));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_HIDDEN + 128u,&engine->reduced));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_EXPERT_ROWS,&engine->gate));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_EXPERT_ROWS,&engine->up));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_EXPERT_ROWS,&engine->act));
	RE_TRY(ReAlloc<uint16_t>(engine,RE_TOP_K * RE_HIDDEN,&engine->down));
	RE_TRY(ReAlloc<float>(engine,RE_RANK_VOCAB,&engine->logits));
	RE_TRY(ReAlloc<float>(engine,RE_TOP_K,&engine->route_weight));
	RE_TRY(ReAlloc<uint32_t>(engine,RE_TOP_K,&engine->route_expert));
	RE_TRY(ReAlloc<uint32_t>(engine,RE_TOP_K,&engine->row_of_k));
	RE_TRY(ReAlloc<uint32_t>(engine,RE_RANK_EXPERTS + 1u,&engine->group_offset));
	RE_TRY(ReAlloc<uint32_t>(engine,RE_TOP_K,&engine->source_token));
	RE_TRY(ReAlloc<uint32_t>(engine,1u,&engine->sequence));
	RE_TRY(ReAlloc<uint32_t>(engine,engine->config.lane_count,&engine->context));
	RE_TRY(ReAlloc<uint32_t>(engine,1u,&engine->position));
	RE_TRY(ReAlloc<uint32_t>(engine,(size_t)engine->pages_per_lane * engine->config.lane_count,&engine->page_table));
	RE_TRY(ReAlloc<uint32_t>(engine,RE_WINDOW,&engine->window));
	RE_TRY(ReAlloc<uint32_t>(engine,4u,&engine->step_in));
	RE_TRY(ReAlloc<uint32_t>(engine,4u,&engine->step_out));
	RE_TRY(ReAlloc<uint64_t>(engine,1u,&engine->head_key));
	RE_TRY(ReAlloc<uint64_t>(engine,16u,&engine->head_best));
	RE_TRY(ReAlloc<LmKvAccessError>(engine,1u,&engine->error));
	RE_CUDA(cudaHostAlloc((void **)&engine->host_in,4u * sizeof(uint32_t),cudaHostAllocDefault),"host-in");
	RE_CUDA(cudaHostAlloc((void **)&engine->host_out,4u * sizeof(uint32_t),cudaHostAllocDefault),"host-out");
	pages.resize((size_t)engine->pages_per_lane * engine->config.lane_count);
	for (i = 0u; i < pages.size(); i++)
		pages[i] = i;
	RE_CUDA(cudaMemcpy(engine->page_table,pages.data(),pages.size() * sizeof(uint32_t),cudaMemcpyHostToDevice),"pages");
	RE_TRY(RePayload<uint16_t>(engine,SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER,SPARK_MIMO26_STAGEPACK_TENSOR_EMBEDDING,&engine->embedding));
	RE_TRY(RePayload<uint16_t>(engine,SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER,SPARK_MIMO26_STAGEPACK_TENSOR_FINAL_NORM,&engine->final_norm));
	RE_TRY(RePayload<uint16_t>(engine,SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER,SPARK_MIMO26_STAGEPACK_TENSOR_LM_HEAD,&engine->lm_head));
	return(SPARK_STATUS_OK);
}

static SparkStatus ReOpenCollective(SparkMimo26RankEngine *engine)
{
	SparkTpDeviceCollectiveConfig configuration;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = RE_DEGREE;
	configuration.tp_rank = engine->config.rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = 1u;
	configuration.local_hidden_dimension = RE_HIDDEN;
	configuration.max_active_sequence_count = 1u;
	configuration.connect_timeout_milli = (uint32_t)(engine->config.wait_ns / 1000000ull);
	configuration.operation_timeout_milli = (uint32_t)(engine->config.wait_ns / 1000000ull);
	configuration.registration_cuda_stream = engine->stream;
	SparkMimo26RankModuleRegisterCombines(&configuration);
	RE_TRY(SparkTpDeviceCollectiveCreate(&configuration,&engine->collective));
	engine->collective_ready = 1u;
	RE_TRY(SparkTpDeviceCollectiveAttach(&engine->collective,0));
	if ( SparkTpDeviceCollectiveStreamOrdered(&engine->collective) == 0u )
	{
		fprintf(stderr,"M26RE-COLLECTIVE rank=%u stream-ordered collectives need SPARK_TP_WAIT_MODE=hardware and the weightd mesh\n",engine->config.rank);
		return(SPARK_STATUS_UNSUPPORTED);
	}
	return(SPARK_STATUS_OK);
}

extern "C" SparkStatus SparkMimo26RankEngineCreate(const SparkMimo26RankEngineConfig *config, SparkMimo26RankEngine **out)
{
	SparkMimo26RankEngine *engine;
	uint32_t layer;
	if ( config == 0 || out == 0 || config->rank >= RE_DEGREE || config->lane_count == 0u || config->max_positions == 0u || (config->max_positions % RE_PAGE_SLOTS) != 0u || config->wait_ns == 0u || config->mode > SPARK_MIMO26_RANK_ENGINE_MODE_GRAPH || (config->mode == SPARK_MIMO26_RANK_ENGINE_MODE_GRAPH && config->dump_directory != 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	engine = (SparkMimo26RankEngine *)calloc(1u,sizeof(*engine));
	if ( engine == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*out = engine;
	engine->config = *config;
	RE_CUDA(cudaFree(0),"cuda-init");
	RE_CUDA(cudaStreamCreateWithFlags(&engine->stream,cudaStreamNonBlocking),"stream");
	RE_TRY(ReOpenCollective(engine));
	RE_TRY(ReOpenPack(engine));
	RE_CUDA(cudaDeviceSynchronize(),"spine-ready");
	RE_TRY(ReAllocate(engine));
	for (layer = 0u; layer < RE_LAYERS; layer++)
		RE_TRY(ReBindLayer(engine,layer));
	RE_TRY(RePinExperts(engine));
	RE_CUDA(cudaStreamSynchronize(engine->stream),"bind");
	return(SPARK_STATUS_OK);
}

static SparkStatus ReCapture(SparkMimo26RankEngine *engine)
{
	cudaGraph_t graph = 0;
	SparkStatus status;
	RE_TRY(SparkTpDeviceCollectiveArmCapture(&engine->collective));
	if ( cudaStreamBeginCapture(engine->stream,cudaStreamCaptureModeThreadLocal) != cudaSuccess )
	{
		(void)SparkTpDeviceCollectiveDisarmCapture(&engine->collective);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = ReWalk(engine);
	if ( cudaStreamEndCapture(engine->stream,&graph) != cudaSuccess || graph == 0 )
		status = status == SPARK_STATUS_OK ? SPARK_STATUS_IO_ERROR : status;
	(void)SparkTpDeviceCollectiveDisarmCapture(&engine->collective);
	if ( status == SPARK_STATUS_OK && (cudaGraphInstantiate(&engine->graph,graph,0) != cudaSuccess || cudaGraphUpload(engine->graph,engine->stream) != cudaSuccess) )
		status = SPARK_STATUS_IO_ERROR;
	if ( graph != 0 )
		(void)cudaGraphDestroy(graph);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"M26RE-CAPTURE rank=%u status=%s cuda=%s\n",engine->config.rank,SparkStatusToString(status),cudaGetErrorString(cudaGetLastError()));
	return(status);
}

extern "C" SparkStatus SparkMimo26RankEngineStep(SparkMimo26RankEngine *engine, uint32_t lane, uint32_t token, uint32_t position, uint32_t *next_token, float *score)
{
	uint64_t started = ReNow();
	uint32_t graph_step;
	if ( engine == 0 || next_token == 0 || lane >= engine->config.lane_count || position >= engine->config.max_positions || token >= RE_VOCAB )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	engine->host_in[0] = token;
	engine->host_in[1] = position;
	engine->host_in[2] = lane;
	engine->chain = engine->chain % SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK + 1u;
	RE_TRY(SparkTpDeviceCollectiveChainKey(&engine->collective,engine->chain));
	graph_step = engine->graph != 0 ? 1u : 0u;
	if ( graph_step != 0u )
	{
		RE_TRY(SparkTpDeviceCollectiveGraphPreLaunch(&engine->collective,engine->stream));
		RE_TRY(SparkTpDeviceCollectiveGraphCancelSeed(&engine->collective,engine->stream));
		RE_CUDA(cudaGraphLaunch(engine->graph,engine->stream),"graph-launch");
	}
	else
	{
		engine->dump_position = position;
		engine->walking_dump = engine->config.dump_directory != 0 && position < 2u ? 1u : 0u;
		RE_TRY(ReWalk(engine));
		engine->walking_dump = 0u;
	}
	RE_TRY(ReWaitStream(engine));
	if ( SparkTpDeviceCollectiveGraphError(&engine->collective) != 0u )
	{
		fprintf(stderr,"M26RE-COLLECTIVE-ERROR rank=%u graph=%u word=%llu\n",engine->config.rank,graph_step,(unsigned long long)SparkTpDeviceCollectiveGraphError(&engine->collective));
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( graph_step == 0u )
		RE_TRY(SparkTpDeviceCollectiveVerifyDeferred(&engine->collective,engine->stream));
	*next_token = engine->host_out[0];
	if ( score != 0 )
		memcpy(score,&engine->host_out[1],sizeof(*score));
	if ( graph_step == 0u && engine->config.mode == SPARK_MIMO26_RANK_ENGINE_MODE_GRAPH )
		RE_TRY(ReCapture(engine));
	RE_TRY(SparkTpDeviceCollectiveEndChain(&engine->collective,engine->stream));
	engine->steps++;
	engine->graph_steps += graph_step;
	engine->step_ns += ReNow() - started;
	return(SPARK_STATUS_OK);
}

extern "C" SparkStatus SparkMimo26RankEngineReadStats(SparkMimo26RankEngine *engine, SparkMimo26RankEngineStats *stats)
{
	LmKvAccessError access;
	if ( engine == 0 || stats == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(stats,0,sizeof(*stats));
	RE_CUDA(cudaMemcpy(&access,engine->error,sizeof(access),cudaMemcpyDeviceToHost),"kv-error");
	stats->steps = engine->steps;
	stats->graph_steps = engine->graph_steps;
	stats->step_ns = engine->step_ns;
	stats->collectives = engine->collectives;
	stats->spine_bytes = engine->pack != 0 ? engine->pack->spine_allocation_bytes : 0u;
	stats->expert_pool_bytes = engine->pack != 0 ? engine->pack->attached.expert_pool_bytes : 0u;
	stats->pinned_experts = engine->pinned_experts;
	stats->kv_error = access.error_code;
	return(SPARK_STATUS_OK);
}

extern "C" SparkStatus SparkMimo26RankEngineDestroy(SparkMimo26RankEngine *engine)
{
	SparkStatus status = SPARK_STATUS_OK, step;
	uint32_t layer, p;
	if ( engine == 0 )
		return(SPARK_STATUS_OK);
	if ( engine->stream != 0 )
		(void)cudaStreamSynchronize(engine->stream);
	if ( engine->graph != 0 )
		(void)cudaGraphExecDestroy(engine->graph);
	if ( engine->collective_ready != 0u )
		SparkTpDeviceCollectiveDestroy(&engine->collective);
	while ( engine->lease_count != 0u )
	{
		uint64_t lease = engine->leases[engine->lease_count - 1u];
		step = SparkWeightdMapRecordCompletion(engine->pack->map,lease,engine->stream);
		if ( step == SPARK_STATUS_OK )
			step = SparkWeightdMapRelease(engine->pack->map,lease,engine->config.wait_ns);
		if ( step != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
			status = step;
		engine->lease_count--;
	}
	for (layer = 0u; layer < RE_LAYERS; layer++)
	{
		(void)cudaFree(engine->layers[layer].pool);
		(void)cudaFree(engine->layers[layer].qkv_scale);
		for (p = 0u; p < 3u; p++)
			(void)cudaFree(engine->layers[layer].dense_scale[p]);
	}
	if ( engine->pack != 0 )
	{
		step = SparkWeightdLazyPackDestroy(engine->pack);
		if ( step != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
			status = step;
	}
	if ( engine->host_in != 0 )
		(void)cudaFreeHost(engine->host_in);
	if ( engine->host_out != 0 )
		(void)cudaFreeHost(engine->host_out);
	if ( engine->stream != 0 )
		(void)cudaStreamDestroy(engine->stream);
	free(engine->entries);
	free(engine);
	return(status);
}
