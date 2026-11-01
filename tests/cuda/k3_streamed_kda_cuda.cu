#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/llms/kimi_k3/streamed.cuh"

#define KDA_SHARDS 16u
#define KDA_ROWS 256u
#define KDA_RANK_HEADS (K3_KDA_HEADS / KDA_SHARDS)
#define KDA_RANK_QK (KDA_RANK_HEADS * K3_KDA_KEY_DIM)
#define KDA_RANK_V (KDA_RANK_HEADS * K3_KDA_VALUE_DIM)
#define KDA_RANK_QKVB (2u * KDA_RANK_QK + KDA_RANK_V + KDA_RANK_HEADS)

#define KDA_CUDA(call) do { cudaError_t kda_error = (call); if ( kda_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(kda_error)); exit(1); } } while (0)

static int kda_failures;

static void KdaCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	kda_failures++;
}

__global__ void KdaFillBf16(uint16_t *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = LmFloatToBf16(offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
}

__global__ void KdaFillF32(float *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f);
	}
}

static double KdaHostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

template<class T>
static T *KdaAlloc(uint64_t count)
{
	T *device;
	KDA_CUDA(cudaMalloc((void **)&device,count * sizeof(T) + 256u));
	KDA_CUDA(cudaMemset(device,0,count * sizeof(T) + 256u));
	return device;
}

static K3StreamedKdaState KdaState(void)
{
	K3StreamedKdaState state;
	state.state_pool = KdaAlloc<uint8_t>(K3_KDA_RANK_STATE_SLOT_BYTES(KDA_RANK_HEADS,0u));
	state.q_window = KdaAlloc<uint16_t>((uint64_t)KDA_RANK_QK * K3_KDA_CONV_KERNEL);
	state.k_window = KdaAlloc<uint16_t>((uint64_t)KDA_RANK_QK * K3_KDA_CONV_KERNEL);
	state.v_window = KdaAlloc<uint16_t>((uint64_t)KDA_RANK_V * K3_KDA_CONV_KERNEL);
	return state;
}

static void KdaBuffers(K3LayerBuffers *b,uint32_t rank,const uint32_t *state_index,const uint32_t *row_begin,const K3StreamedKdaState *state)
{
	memset(b,0,sizeof(*b));
	b->hidden_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * K3_HIDDEN);
	b->normed_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * K3_HIDDEN);
	b->fused_qkvb_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_QKVB);
	b->query_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_QK);
	b->key_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_QK);
	b->value_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_V);
	b->kda_beta_logit = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_HEADS);
	b->latent_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * K3_KDA_KEY_DIM);
	b->gate_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_V);
	b->decay_logit_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_QK);
	b->kda_retention = KdaAlloc<float>((uint64_t)KDA_ROWS * KDA_RANK_QK);
	b->kda_write_gate_out = KdaAlloc<float>((uint64_t)KDA_ROWS * KDA_RANK_HEADS);
	b->attention_out_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * KDA_RANK_V);
	b->attnres_partial_bf16 = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * K3_HIDDEN);
	uint32_t *dense_offsets = KdaAlloc<uint32_t>(2u),offsets[2] = {0u,KDA_ROWS};
	KDA_CUDA(cudaMemcpy(dense_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice));
	b->dense_row_offset = dense_offsets;
	b->dense_tile_prefix = KdaAlloc<uint32_t>(2u);
	b->kda_state_index = state_index;
	b->sequence_row_begin = row_begin;
	if ( state != 0 )
	{
		b->kda_state_pool = state->state_pool;
		b->kda_q_window = state->q_window;
		b->kda_k_window = state->k_window;
		b->kda_v_window = state->v_window;
	}
	b->tp_sharded = 1u;
	b->tp_rank = rank;
	b->kda_heads_rank = KDA_RANK_HEADS;
	b->kda_qkvb_rows = KDA_RANK_QKVB;
	b->kda_gate_rows = KDA_RANK_V;
	b->kda_decay_up_rows = KDA_RANK_QK;
	b->kda_out_input = KDA_RANK_V;
}

int main(void)
{
	std::vector<K3LayerWeights> shards(KDA_SHARDS);
	std::vector<K3StreamedKdaState> reference_states(KDA_SHARDS),streamed_states(KDA_SHARDS);
	uint16_t *attn_norm = KdaAlloc<uint16_t>(K3_HIDDEN),*decay_down = KdaAlloc<uint16_t>((uint64_t)K3_KDA_KEY_DIM * K3_HIDDEN);
	uint16_t *hidden = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * K3_HIDDEN),*partial = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * K3_HIDDEN);
	uint16_t *scratch = KdaAlloc<uint16_t>((uint64_t)KDA_ROWS * K3_HIDDEN);
	float *decay_bias = KdaAlloc<float>((uint64_t)K3_KDA_HEADS * K3_KDA_KEY_DIM),*log_scale = KdaAlloc<float>(K3_KDA_HEADS);
	float *out_norm = KdaAlloc<float>(K3_KDA_VALUE_DIM),*accumulator = KdaAlloc<float>((uint64_t)KDA_ROWS * K3_HIDDEN);
	uint32_t *state_index = KdaAlloc<uint32_t>(1u),*row_begin = KdaAlloc<uint32_t>(2u),begin[2] = {0u,KDA_ROWS};
	K3LayerBuffers reference[KDA_SHARDS],streamed;
	cudaStream_t stream;
	KDA_CUDA(cudaMemcpy(row_begin,begin,sizeof(begin),cudaMemcpyHostToDevice));
	KdaFillBf16<<<64,256>>>(attn_norm,K3_HIDDEN,1u,0.1f,1.0f);
	KdaFillBf16<<<512,256>>>(decay_down,(uint64_t)K3_KDA_KEY_DIM * K3_HIDDEN,2u,0.02f,0.0f);
	KdaFillBf16<<<512,256>>>(hidden,(uint64_t)KDA_ROWS * K3_HIDDEN,3u,1.0f,0.0f);
	KdaFillBf16<<<512,256>>>(partial,(uint64_t)KDA_ROWS * K3_HIDDEN,4u,1.0f,0.0f);
	KdaFillF32<<<64,256>>>(decay_bias,(uint64_t)K3_KDA_HEADS * K3_KDA_KEY_DIM,5u,0.5f,0.0f);
	KdaFillF32<<<8,128>>>(log_scale,K3_KDA_HEADS,6u,0.5f,0.0f);
	KdaFillF32<<<8,128>>>(out_norm,K3_KDA_VALUE_DIM,7u,0.1f,1.0f);
	for (uint32_t shard = 0u; shard < KDA_SHARDS; shard++)
	{
		K3LayerWeights *w = &shards[shard];
		uint16_t *qkvb = KdaAlloc<uint16_t>((uint64_t)KDA_RANK_QKVB * K3_HIDDEN),*decay_up = KdaAlloc<uint16_t>((uint64_t)KDA_RANK_QK * K3_KDA_KEY_DIM);
		uint16_t *gate = KdaAlloc<uint16_t>((uint64_t)KDA_RANK_V * K3_HIDDEN),*out = KdaAlloc<uint16_t>((uint64_t)K3_HIDDEN * KDA_RANK_V);
		float *q_conv = KdaAlloc<float>((uint64_t)KDA_RANK_QK * K3_KDA_CONV_KERNEL),*k_conv = KdaAlloc<float>((uint64_t)KDA_RANK_QK * K3_KDA_CONV_KERNEL);
		float *v_conv = KdaAlloc<float>((uint64_t)KDA_RANK_V * K3_KDA_CONV_KERNEL);
		KdaFillBf16<<<512,256>>>(qkvb,(uint64_t)KDA_RANK_QKVB * K3_HIDDEN,10u + shard,0.02f,0.0f);
		KdaFillBf16<<<64,256>>>(decay_up,(uint64_t)KDA_RANK_QK * K3_KDA_KEY_DIM,30u + shard,0.1f,0.0f);
		KdaFillBf16<<<512,256>>>(gate,(uint64_t)KDA_RANK_V * K3_HIDDEN,50u + shard,0.02f,0.0f);
		KdaFillBf16<<<512,256>>>(out,(uint64_t)K3_HIDDEN * KDA_RANK_V,70u + shard,0.05f,0.0f);
		KdaFillF32<<<64,256>>>(q_conv,(uint64_t)KDA_RANK_QK * K3_KDA_CONV_KERNEL,90u + shard,0.5f,0.0f);
		KdaFillF32<<<64,256>>>(k_conv,(uint64_t)KDA_RANK_QK * K3_KDA_CONV_KERNEL,110u + shard,0.5f,0.0f);
		KdaFillF32<<<64,256>>>(v_conv,(uint64_t)KDA_RANK_V * K3_KDA_CONV_KERNEL,130u + shard,0.5f,0.0f);
		memset(w,0,sizeof(*w));
		w->attn_norm_weight = attn_norm;
		w->kda_qkv_beta_weight = qkvb;
		w->kda_decay_down_weight = decay_down;
		w->kda_decay_up_weight = decay_up;
		w->kda_decay_bias = decay_bias;
		w->kda_head_log_scale = log_scale;
		w->kda_gate_weight = gate;
		w->kda_out_norm_weight = out_norm;
		w->kda_out_weight = out;
		w->kda_q_conv_weight = q_conv;
		w->kda_k_conv_weight = k_conv;
		w->kda_v_conv_weight = v_conv;
		reference_states[shard] = KdaState();
		streamed_states[shard] = KdaState();
	}
	KDA_CUDA(cudaDeviceSynchronize());
	{
		LmTpBf16Contributions<KDA_SHARDS> inputs;
		for (uint32_t rank = 0u; rank < KDA_SHARDS; rank++)
		{
			KdaBuffers(&reference[rank],rank,state_index,row_begin,&reference_states[rank]);
			K3BindLayer(&shards[rank],&reference[rank]);
			KDA_CUDA(cudaMemcpy(reference[rank].hidden_bf16,hidden,(uint64_t)KDA_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
			KdaCheck(K3LayerKda<LmMxfp4>(&reference[rank],KDA_ROWS,1u,1u,(uint16_t *)0,48u,0) == LM_LAUNCH_OK,"every TP16 rank runs its KDA heads");
			inputs.rank[rank] = reference[rank].hidden_bf16;
		}
		LmTpBf16SumKernel<KDA_SHARDS><<<KDA_ROWS,256u>>>(scratch,inputs,KDA_ROWS,K3_HIDDEN);
		KDA_CUDA(cudaMemcpy(reference[0].attnres_partial_bf16,partial,(uint64_t)KDA_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
		K3PartialAdd(&reference[0],scratch,KDA_ROWS,0);
		KDA_CUDA(cudaDeviceSynchronize());
	}
	KdaBuffers(&streamed,0u,state_index,row_begin,0);
	KDA_CUDA(cudaMemcpy(streamed.hidden_bf16,hidden,(uint64_t)KDA_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	KDA_CUDA(cudaMemcpy(streamed.attnres_partial_bf16,partial,(uint64_t)KDA_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	KDA_CUDA(cudaStreamCreate(&stream));
	KdaCheck(K3StreamedKda<LmMxfp4>(&streamed,shards.data(),streamed_states.data(),KDA_SHARDS,scratch,accumulator,0u,KDA_ROWS,1u,48u,stream) == LM_LAUNCH_OK,
		"the streamed KDA runs all sixteen shards' heads on one device");
	KDA_CUDA(cudaStreamSynchronize(stream));
	{
		std::vector<uint16_t> want((uint64_t)KDA_ROWS * K3_HIDDEN),got(want.size()),before(want.size());
		std::vector<uint8_t> state_want(K3_KDA_RANK_STATE_SLOT_BYTES(KDA_RANK_HEADS,0u)),state_got(state_want.size());
		double worst = 0.0,delta = 0.0;
		uint64_t nonfinite = 0u,states_equal = 0u;
		KDA_CUDA(cudaMemcpy(want.data(),reference[0].attnres_partial_bf16,want.size() * 2u,cudaMemcpyDeviceToHost));
		KDA_CUDA(cudaMemcpy(got.data(),streamed.attnres_partial_bf16,got.size() * 2u,cudaMemcpyDeviceToHost));
		KDA_CUDA(cudaMemcpy(before.data(),partial,before.size() * 2u,cudaMemcpyDeviceToHost));
		for (uint64_t index = 0u; index < want.size(); index++)
		{
			const double r = KdaHostFloat(want[index]),c = KdaHostFloat(got[index]),b = KdaHostFloat(before[index]);
			if ( !isfinite(r) || !isfinite(c) )
			{
				nonfinite++;
				continue;
			}
			delta = fmax(delta,fabs(r - b));
			worst = fmax(worst,fabs(r - c));
		}
		for (uint32_t shard = 0u; shard < KDA_SHARDS; shard++)
		{
			KDA_CUDA(cudaMemcpy(state_want.data(),reference_states[shard].state_pool,state_want.size(),cudaMemcpyDeviceToHost));
			KDA_CUDA(cudaMemcpy(state_got.data(),streamed_states[shard].state_pool,state_got.size(),cudaMemcpyDeviceToHost));
			states_equal += memcmp(state_want.data(),state_got.data(),state_want.size()) == 0 ? 1u : 0u;
		}
		KdaCheck(nonfinite == 0u,"every output element is finite");
		KdaCheck(delta > 0.0,"the KDA layer changes the residual");
		KdaCheck(states_equal == KDA_SHARDS,"every shard's recurrent state ends bit-identical to its TP16 rank's");
		KdaCheck(worst <= 0.02 * delta,"the streamed KDA matches TP16 within 2% of the largest attention contribution");
		printf("%s streamed KDA vs TP16: worst |diff| %.4g, largest attention contribution %.4g, recurrent states identical %llu of %u, nonfinite %llu\n",
			kda_failures == 0 ? "PASS" : "FAIL",worst,delta,(unsigned long long)states_equal,KDA_SHARDS,(unsigned long long)nonfinite);
	}
	return kda_failures != 0;
}
