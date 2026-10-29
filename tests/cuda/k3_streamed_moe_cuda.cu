#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/llms/kimi_k3/streamed.cuh"

#define MOE_SHARDS 16u
#define MOE_ROWS 256u
#define MOE_SLICE (K3_ROUTED_EXPERT_HIDDEN / MOE_SHARDS)
#define MOE_EXPERT_SLICE (K3_EXPERTS / MOE_SHARDS)
#define MOE_W1_OUT (2u * K3_EXPERT_INTERMEDIATE / MOE_SHARDS)
#define MOE_CHANNELS (K3_EXPERT_INTERMEDIATE / MOE_SHARDS)
#define MOE_SHARED_ROWS (2u * K3_SHARED_INTERMEDIATE / MOE_SHARDS)
#define MOE_SHARED_IN (K3_SHARED_INTERMEDIATE / MOE_SHARDS)
#define MOE_PACKED (MOE_ROWS * K3_TOP_K)

#define MOE_CUDA(call) do { cudaError_t moe_error = (call); if ( moe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(moe_error)); exit(1); } } while (0)

static int moe_failures;

static void MoeCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	moe_failures++;
}

__global__ void MoeFillBf16(uint16_t *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = LmFloatToBf16(offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
}

__global__ void MoeFillF32(float *data,uint64_t count,uint32_t salt,float scale)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f);
	}
}

__global__ void MoeFillCells(uint8_t *data,uint64_t count,uint32_t salt)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = (uint8_t)(((index / LM_SKINNY_CHUNK_BYTES) % LM_SKINNY_CELL_ROWS) == LM_SKINNY_CELL_NEURONS ? 0x76u + (x & 3u) : x);
	}
}

static double MoeHostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

template<class T>
static T *MoeAlloc(uint64_t count)
{
	T *device;
	MOE_CUDA(cudaMalloc((void **)&device,count * sizeof(T) + 256u));
	MOE_CUDA(cudaMemset(device,0,count * sizeof(T) + 256u));
	return device;
}

static void MoeBuffers(K3LayerBuffers *b,uint32_t rank)
{
	memset(b,0,sizeof(*b));
	b->hidden_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
	b->normed_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
	b->latent_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * (K3_ROUTED_EXPERT_HIDDEN + 2u * K3_EXPERTS));
	b->latent_full_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_ROUTED_EXPERT_HIDDEN);
	b->router_logits = MoeAlloc<float>((uint64_t)MOE_ROWS * K3_EXPERTS);
	b->route_expert = MoeAlloc<uint32_t>(MOE_PACKED);
	b->route_weight = MoeAlloc<float>(MOE_PACKED);
	b->route_packed_row = MoeAlloc<uint32_t>(MOE_PACKED);
	b->route_source_token = MoeAlloc<uint32_t>(MOE_PACKED);
	b->group_row_offset = MoeAlloc<uint32_t>(K3_EXPERTS + 1u);
	b->group_tile_prefix_w1 = MoeAlloc<uint32_t>(K3_EXPERTS + 1u);
	b->group_tile_prefix_w2 = MoeAlloc<uint32_t>(K3_EXPERTS + 1u);
	b->gate_up_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_PACKED * K3_ROUTED_EXPERT_HIDDEN);
	b->intermediate_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_PACKED * MOE_CHANNELS);
	b->shared_mid_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * MOE_SHARED_IN);
	b->shared_out_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
	b->attnres_partial_bf16 = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
	uint32_t *dense_offsets = MoeAlloc<uint32_t>(2u),offsets[2] = {0u,MOE_ROWS};
	MOE_CUDA(cudaMemcpy(dense_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice));
	b->dense_row_offset = dense_offsets;
	b->dense_tile_prefix = MoeAlloc<uint32_t>(2u);
	b->tp_sharded = 1u;
	b->tp_rank = rank;
	b->expert_w1_output = MOE_W1_OUT;
	b->expert_w2_input = MOE_CHANNELS;
	b->routed_down_rows = MOE_SLICE;
	b->routed_up_input = MOE_SLICE;
	b->shared_w1_rows = MOE_SHARED_ROWS;
	b->shared_w2_input = MOE_SHARED_IN;
}

struct MoeRank
{
	uint32_t rank;
	K3LayerBuffers buffers;
	K3SliceState state;
	cudaStream_t stream;
	cudaEvent_t contributed;
	cudaEvent_t consumed;
	int32_t status;
};

static MoeRank moe_ranks[MOE_SHARDS];
static pthread_barrier_t moe_barrier;

static void MoeWaitAll(MoeRank *self,cudaEvent_t MoeRank::*which)
{
	for (uint32_t peer = 0u; peer < MOE_SHARDS; peer++)
		MOE_CUDA(cudaStreamWaitEvent(self->stream,moe_ranks[peer].*which,0u));
}

static void MoeCollective(void *context,void *stream_word,uint32_t layer,uint32_t phase)
{
	MoeRank *self = (MoeRank *)context;
	const uint32_t base = phase & ~K3_COLLECTIVE_FINISH;
	K3LayerBuffers *b = &self->buffers;
	(void)stream_word;
	(void)layer;
	if ( (phase & K3_COLLECTIVE_FINISH) == 0u )
	{
		MOE_CUDA(cudaEventRecord(self->contributed,self->stream));
		return;
	}
	pthread_barrier_wait(&moe_barrier);
	MoeWaitAll(self,&MoeRank::contributed);
	for (uint32_t peer = 0u; peer < MOE_SHARDS; peer++)
	{
		const K3LayerBuffers *p = &moe_ranks[peer].buffers;
		if ( base == 2u )
		{
			MOE_CUDA(cudaMemcpy2DAsync(b->latent_full_bf16 + peer * MOE_SLICE,K3_ROUTED_EXPERT_HIDDEN * 2u,p->latent_bf16,MOE_SLICE * 2u,MOE_SLICE * 2u,MOE_ROWS,
				cudaMemcpyDeviceToDevice,self->stream));
			MOE_CUDA(cudaMemcpy2DAsync(b->router_logits + peer * MOE_EXPERT_SLICE,K3_EXPERTS * 4u,K3_ROUTER_SLICE(p,MOE_ROWS),MOE_EXPERT_SLICE * 4u,
				MOE_EXPERT_SLICE * 4u,MOE_ROWS,cudaMemcpyDeviceToDevice,self->stream));
		}
	}
	if ( base == 3u )
	{
		LmTpBf16Contributions<MOE_SHARDS> inputs;
		for (uint32_t peer = 0u; peer < MOE_SHARDS; peer++)
			inputs.rank[peer] = moe_ranks[peer].buffers.latent_bf16;
		LmTpBf16SumKernel<MOE_SHARDS><<<MOE_ROWS,256u,0,self->stream>>>(b->latent_full_bf16,inputs,MOE_ROWS,K3_ROUTED_EXPERT_HIDDEN);
		MOE_CUDA(cudaGetLastError());
	}
	MOE_CUDA(cudaEventRecord(self->consumed,self->stream));
	pthread_barrier_wait(&moe_barrier);
	MoeWaitAll(self,&MoeRank::consumed);
}

static void *MoeRankThread(void *argument)
{
	MoeRank *self = (MoeRank *)argument;
	self->status = K3LaunchCellsMoe<LmMxfp4>(&self->state,&self->buffers,K3_FIRST_ROUTED_LAYER,MOE_ROWS,MOE_PACKED,48u,self->stream);
	MOE_CUDA(cudaStreamSynchronize(self->stream));
	return 0;
}

int main(void)
{
	const uint64_t w1_bytes = (uint64_t)K3_EXPERTS * (K3_ROUTED_EXPERT_HIDDEN / 32u) * (MOE_W1_OUT / 16u) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	const uint64_t w2_bytes = (uint64_t)K3_EXPERTS * (MOE_CHANNELS / 32u) * (K3_ROUTED_EXPERT_HIDDEN / 16u) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	std::vector<K3LayerWeights> shards(MOE_SHARDS);
	uint16_t *router = MoeAlloc<uint16_t>((uint64_t)K3_EXPERTS * K3_HIDDEN);
	uint16_t *mlp_norm = MoeAlloc<uint16_t>(K3_HIDDEN);
	uint16_t *routed_norm = MoeAlloc<uint16_t>(K3_ROUTED_EXPERT_HIDDEN);
	uint16_t *hidden = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
	uint16_t *partial = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
	float *bias = MoeAlloc<float>(K3_EXPERTS);
	float *latent_accumulator = MoeAlloc<float>((uint64_t)MOE_ROWS * K3_ROUTED_EXPERT_HIDDEN);
	float *hidden_accumulator = MoeAlloc<float>((uint64_t)MOE_ROWS * K3_HIDDEN);
	K3LayerBuffers streamed;
	cudaStream_t stream;
	MoeFillBf16<<<512,256>>>(router,(uint64_t)K3_EXPERTS * K3_HIDDEN,1u,0.02f,0.0f);
	MoeFillBf16<<<64,256>>>(mlp_norm,K3_HIDDEN,2u,0.1f,1.0f);
	MoeFillBf16<<<64,256>>>(routed_norm,K3_ROUTED_EXPERT_HIDDEN,3u,0.1f,1.0f);
	MoeFillBf16<<<512,256>>>(hidden,(uint64_t)MOE_ROWS * K3_HIDDEN,4u,1.0f,0.0f);
	MoeFillBf16<<<512,256>>>(partial,(uint64_t)MOE_ROWS * K3_HIDDEN,5u,1.0f,0.0f);
	MoeFillF32<<<8,128>>>(bias,K3_EXPERTS,6u,0.01f);
	for (uint32_t shard = 0u; shard < MOE_SHARDS; shard++)
	{
		K3LayerWeights *w = &shards[shard];
		memset(w,0,sizeof(*w));
		w->mlp_norm_weight = mlp_norm;
		w->router_weight = router;
		w->router_bias = bias;
		w->routed_norm_weight = routed_norm;
		uint16_t *down = MoeAlloc<uint16_t>((uint64_t)MOE_SLICE * K3_HIDDEN), *up = MoeAlloc<uint16_t>((uint64_t)K3_HIDDEN * MOE_SLICE);
		uint16_t *shared_w1 = MoeAlloc<uint16_t>((uint64_t)MOE_SHARED_ROWS * K3_HIDDEN), *shared_w2 = MoeAlloc<uint16_t>((uint64_t)K3_HIDDEN * MOE_SHARED_IN);
		uint8_t *w1 = MoeAlloc<uint8_t>(w1_bytes), *w2 = MoeAlloc<uint8_t>(w2_bytes);
		MoeFillBf16<<<512,256>>>(down,(uint64_t)MOE_SLICE * K3_HIDDEN,10u + shard,0.02f,0.0f);
		MoeFillBf16<<<512,256>>>(up,(uint64_t)K3_HIDDEN * MOE_SLICE,30u + shard,0.05f,0.0f);
		MoeFillBf16<<<512,256>>>(shared_w1,(uint64_t)MOE_SHARED_ROWS * K3_HIDDEN,50u + shard,0.02f,0.0f);
		MoeFillBf16<<<512,256>>>(shared_w2,(uint64_t)K3_HIDDEN * MOE_SHARED_IN,70u + shard,0.05f,0.0f);
		MoeFillCells<<<1024,256>>>(w1,w1_bytes,90u + shard);
		MoeFillCells<<<1024,256>>>(w2,w2_bytes,110u + shard);
		w->routed_down_weight = down;
		w->routed_up_weight = up;
		w->shared_w1_weight = shared_w1;
		w->shared_w2_weight = shared_w2;
		w->expert_w1_weight = w1;
		w->expert_w2_weight = w2;
		w->expert_interleave = 1u;
		w->expert_tile_k = 32u;
	}
	MOE_CUDA(cudaDeviceSynchronize());
	pthread_barrier_init(&moe_barrier,0,MOE_SHARDS);
	for (uint32_t rank = 0u; rank < MOE_SHARDS; rank++)
	{
		MoeRank *r = &moe_ranks[rank];
		r->rank = rank;
		MoeBuffers(&r->buffers,rank);
		K3BindLayer(&shards[rank],&r->buffers);
		MOE_CUDA(cudaMemcpy(r->buffers.hidden_bf16,hidden,(uint64_t)MOE_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
		MOE_CUDA(cudaMemcpy(r->buffers.attnres_partial_bf16,partial,(uint64_t)MOE_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
		memset(&r->state,0,sizeof(r->state));
		r->state.layer_collective = MoeCollective;
		r->state.collective_context = r;
		MOE_CUDA(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking));
		MOE_CUDA(cudaEventCreateWithFlags(&r->contributed,cudaEventDisableTiming));
		MOE_CUDA(cudaEventCreateWithFlags(&r->consumed,cudaEventDisableTiming));
	}
	{
		pthread_t threads[MOE_SHARDS];
		for (uint32_t rank = 0u; rank < MOE_SHARDS; rank++)
			pthread_create(&threads[rank],0,MoeRankThread,&moe_ranks[rank]);
		for (uint32_t rank = 0u; rank < MOE_SHARDS; rank++)
			pthread_join(threads[rank],0);
	}
	for (uint32_t rank = 0u; rank < MOE_SHARDS; rank++)
		MoeCheck(moe_ranks[rank].status == LM_LAUNCH_OK,"every TP16 rank runs the cells MoE");
	{
		LmTpBf16Contributions<MOE_SHARDS> inputs;
		uint16_t *fused[MOE_SHARDS];
		uint16_t *sum = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
		for (uint32_t rank = 0u; rank < MOE_SHARDS; rank++)
		{
			fused[rank] = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_HIDDEN);
			LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,MOE_ROWS),K3_LAYER_THREADS,0,0,
				moe_ranks[rank].buffers.hidden_bf16,moe_ranks[rank].buffers.shared_out_bf16,fused[rank],MOE_ROWS,K3_HIDDEN);
			inputs.rank[rank] = fused[rank];
		}
		LmTpBf16SumKernel<MOE_SHARDS><<<MOE_ROWS,256u>>>(sum,inputs,MOE_ROWS,K3_HIDDEN);
		K3PartialAdd(&moe_ranks[0].buffers,sum,MOE_ROWS,0);
		MOE_CUDA(cudaDeviceSynchronize());
	}
	MoeBuffers(&streamed,0u);
	MOE_CUDA(cudaMemcpy(streamed.hidden_bf16,hidden,(uint64_t)MOE_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	MOE_CUDA(cudaMemcpy(streamed.attnres_partial_bf16,partial,(uint64_t)MOE_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	MOE_CUDA(cudaStreamCreate(&stream));
	K3StreamedJoin join;
	join.weight = MoeAlloc<uint16_t>((uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM);
	join.weight_capacity = (uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM;
	join.activation = MoeAlloc<uint16_t>((uint64_t)MOE_ROWS * K3_MLA_HEADS * K3_V_HEAD_DIM);
	join.activation_capacity = (uint64_t)MOE_ROWS * K3_MLA_HEADS * K3_V_HEAD_DIM;
	MoeCheck(K3StreamedMoe<LmMxfp4>(&streamed,shards.data(),MOE_SHARDS,&join,latent_accumulator,MOE_ROWS,MOE_PACKED,48u,stream) == LM_LAUNCH_OK,
		"the streamed MoE runs all sixteen shards on one device");
	MOE_CUDA(cudaStreamSynchronize(stream));
	{
		std::vector<uint16_t> reference((uint64_t)MOE_ROWS * K3_HIDDEN),candidate(reference.size()),before(reference.size());
		std::vector<uint32_t> routes_reference(MOE_PACKED),routes_candidate(MOE_PACKED);
		double worst = 0.0,scale = 0.0,delta_scale = 0.0;
		uint64_t nonfinite = 0u,index;
		MOE_CUDA(cudaMemcpy(reference.data(),moe_ranks[0].buffers.attnres_partial_bf16,reference.size() * 2u,cudaMemcpyDeviceToHost));
		MOE_CUDA(cudaMemcpy(candidate.data(),streamed.attnres_partial_bf16,candidate.size() * 2u,cudaMemcpyDeviceToHost));
		MOE_CUDA(cudaMemcpy(before.data(),partial,before.size() * 2u,cudaMemcpyDeviceToHost));
		MOE_CUDA(cudaMemcpy(routes_reference.data(),moe_ranks[0].buffers.route_expert,MOE_PACKED * 4u,cudaMemcpyDeviceToHost));
		MOE_CUDA(cudaMemcpy(routes_candidate.data(),streamed.route_expert,MOE_PACKED * 4u,cudaMemcpyDeviceToHost));
		for (index = 0u; index < reference.size(); index++)
		{
			const double r = MoeHostFloat(reference[index]),c = MoeHostFloat(candidate[index]),b = MoeHostFloat(before[index]);
			if ( !isfinite(r) || !isfinite(c) )
			{
				nonfinite++;
				continue;
			}
			scale = fmax(scale,fabs(r));
			delta_scale = fmax(delta_scale,fabs(r - b));
			worst = fmax(worst,fabs(r - c));
		}
		MoeCheck(nonfinite == 0u,"every output element is finite");
		MoeCheck(memcmp(routes_reference.data(),routes_candidate.data(),MOE_PACKED * 4u) == 0,"both paths route every token to the same sixteen experts");
		MoeCheck(delta_scale > 0.0,"the MoE changes the residual");
		MoeCheck(worst <= 0.02 * delta_scale,"the streamed MoE matches TP16 within 2% of the largest MoE contribution");
		printf("%s streamed MoE vs TP16: worst |diff| %.4g, largest MoE contribution %.4g, largest output %.4g, nonfinite %llu\n",
			moe_failures == 0 ? "PASS" : "FAIL",worst,delta_scale,scale,(unsigned long long)nonfinite);
	}
	return moe_failures != 0;
}
