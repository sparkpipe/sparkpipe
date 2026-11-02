#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/llms/kimi_k3/streamed.cuh"

#define ML_SHARDS 16u
#define ML_ROWS 128u
#define ML_HEADS (K3_MLA_HEADS / ML_SHARDS)
#define ML_PAGES ((ML_ROWS + K3_KV_PAGE_SLOTS - 1u) / K3_KV_PAGE_SLOTS)
#define ML_Q_UP_ROWS (ML_HEADS * K3_MLA_KV_A_DIM)

#define ML_CUDA(call) do { cudaError_t ml_error = (call); if ( ml_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(ml_error)); exit(1); } } while (0)

static int ml_failures;

static void MlCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	ml_failures++;
}

__global__ void MlFillBf16(uint16_t *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = LmFloatToBf16(offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
}

__global__ void MlAbsorb(const uint16_t *q_b,const uint16_t *kv_b_key,uint16_t *q_up,uint32_t first_head)
{
	const uint64_t total = (uint64_t)ML_Q_UP_ROWS * K3_Q_LORA_RANK;
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < total; index += (uint64_t)gridDim.x * blockDim.x)
	{
		const uint32_t row = (uint32_t)(index / K3_Q_LORA_RANK),column = (uint32_t)(index % K3_Q_LORA_RANK);
		const uint32_t head = first_head + row / K3_MLA_KV_A_DIM,latent = row % K3_MLA_KV_A_DIM;
		float value = 0.0f;
		if ( latent < K3_KV_LORA_RANK )
			for (uint32_t nope = 0u; nope < K3_QK_NOPE_DIM; nope++)
				value += LmBf16ToFloat(kv_b_key[((uint64_t)head * K3_QK_NOPE_DIM + nope) * K3_KV_LORA_RANK + latent]) *
					LmBf16ToFloat(q_b[((uint64_t)head * K3_MHA_QK + nope) * K3_Q_LORA_RANK + column]);
		else
			value = LmBf16ToFloat(q_b[((uint64_t)head * K3_MHA_QK + K3_QK_NOPE_DIM + latent - K3_KV_LORA_RANK) * K3_Q_LORA_RANK + column]);
		q_up[index] = LmFloatToBf16(value);
	}
}

static double MlHostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

template<class T>
static T *MlAlloc(uint64_t count)
{
	T *device;
	ML_CUDA(cudaMalloc((void **)&device,count * sizeof(T) + 256u));
	ML_CUDA(cudaMemset(device,0,count * sizeof(T) + 256u));
	return device;
}

static uint16_t *MlBf16(uint64_t count,uint32_t salt,float scale,float offset)
{
	uint16_t *data = MlAlloc<uint16_t>(count);
	MlFillBf16<<<512,256>>>(data,count,salt,scale,offset);
	return data;
}

static void MlBuffers(K3LayerBuffers *b,uint32_t rank,uint32_t query_width,const uint32_t *sequence_of_row,const uint32_t *context,const uint32_t *positions)
{
	uint32_t *dense_offsets = MlAlloc<uint32_t>(2u),offsets[2] = {0u,ML_ROWS};
	memset(b,0,sizeof(*b));
	b->hidden_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	b->normed_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	b->latent_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_Q_LORA_RANK);
	b->kv_slot_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_MLA_KV_A_DIM);
	b->query_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * query_width);
	b->attention_out_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	b->value_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * ML_HEADS * K3_V_HEAD_DIM);
	b->gate_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * ML_HEADS * K3_V_HEAD_DIM);
	b->attnres_partial_bf16 = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	ML_CUDA(cudaMemcpy(dense_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice));
	b->dense_row_offset = dense_offsets;
	b->dense_tile_prefix = MlAlloc<uint32_t>(2u);
	b->sequence_of_row = sequence_of_row;
	b->context_length = context;
	b->positions = positions;
	b->tp_sharded = 1u;
	b->tp_rank = rank;
	b->mla_heads_rank = ML_HEADS;
	b->mla_q_up_rows = ML_Q_UP_ROWS;
	b->mla_gate_rows = ML_HEADS * K3_V_HEAD_DIM;
	b->mla_out_input = ML_HEADS * K3_V_HEAD_DIM;
}

static LmKvShardView MlCache(uint32_t *table,LmKvAccessError *error)
{
	SparkKvShard shard = {1u,0u,1u};
	LmKvShardView view;
	const uint64_t bytes = SparkKvShardPoolBytes(shard,K3GlobalKv::kPageSlots,K3GlobalKv::kSlotBytes,ML_PAGES);
	uint8_t *pool = MlAlloc<uint8_t>(bytes);
	if ( LmKvShardViewInitialize<K3GlobalKv>(&view,pool,table,ML_PAGES,1u,ML_PAGES,error,shard) != 0 )
	{
		printf("FAIL cache view\n");
		exit(1);
	}
	return view;
}

int main(void)
{
	std::vector<K3LayerWeights> shards(ML_SHARDS);
	std::vector<uint32_t> table(ML_PAGES),positions(ML_ROWS),zeros(ML_ROWS,0u);
	uint16_t *hidden = MlBf16((uint64_t)ML_ROWS * K3_HIDDEN,1u,1.0f,0.0f),*partial_in = MlBf16((uint64_t)ML_ROWS * K3_HIDDEN,2u,1.0f,0.0f);
	uint16_t *attn_norm = MlBf16(K3_HIDDEN,3u,0.1f,1.0f),*q_norm = MlBf16(K3_Q_LORA_RANK,4u,0.1f,1.0f),*kv_norm = MlBf16(K3_KV_LORA_RANK,5u,0.1f,1.0f);
	uint16_t *q_down = MlBf16((uint64_t)K3_Q_LORA_RANK * K3_HIDDEN,6u,0.02f,0.0f),*kv_a = MlBf16((uint64_t)K3_MLA_KV_A_DIM * K3_HIDDEN,7u,0.02f,0.0f);
	uint16_t *q_b = MlBf16((uint64_t)K3_MLA_HEADS * K3_MHA_QK * K3_Q_LORA_RANK,8u,0.03f,0.0f);
	uint16_t *kv_b_key = MlBf16((uint64_t)K3_MLA_HEADS * K3_QK_NOPE_DIM * K3_KV_LORA_RANK,9u,0.05f,0.0f);
	uint16_t *kv_b_value = MlBf16((uint64_t)K3_MLA_HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK,10u,0.05f,0.0f);
	uint32_t *table_device,*positions_device,*sequence_device,*context_device,*plan_device,plan[3] = {0u,0u,ML_ROWS},context_host = ML_ROWS;
	uint16_t *scratch = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	float *accumulator = MlAlloc<float>((uint64_t)ML_ROWS * K3_HIDDEN);
	LmKvAccessError *error_device = MlAlloc<LmKvAccessError>(1u),error_host;
	K3LayerBuffers reference[ML_SHARDS],streamed;
	K3StreamedMlaScratch mla;
	cudaStream_t stream;
	for (uint32_t page = 0u; page < ML_PAGES; page++)
		table[page] = page;
	for (uint32_t row = 0u; row < ML_ROWS; row++)
		positions[row] = row;
	table_device = MlAlloc<uint32_t>(ML_PAGES); positions_device = MlAlloc<uint32_t>(ML_ROWS); sequence_device = MlAlloc<uint32_t>(ML_ROWS);
	context_device = MlAlloc<uint32_t>(1u); plan_device = MlAlloc<uint32_t>(3u);
	ML_CUDA(cudaMemcpy(table_device,table.data(),ML_PAGES * 4u,cudaMemcpyHostToDevice));
	ML_CUDA(cudaMemcpy(positions_device,positions.data(),ML_ROWS * 4u,cudaMemcpyHostToDevice));
	ML_CUDA(cudaMemcpy(sequence_device,zeros.data(),ML_ROWS * 4u,cudaMemcpyHostToDevice));
	ML_CUDA(cudaMemcpy(context_device,&context_host,4u,cudaMemcpyHostToDevice));
	ML_CUDA(cudaMemcpy(plan_device,plan,sizeof(plan),cudaMemcpyHostToDevice));
	LmKvAccessErrorReset(&error_host);
	ML_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	for (uint32_t shard = 0u; shard < ML_SHARDS; shard++)
	{
		K3LayerWeights *w = &shards[shard];
		uint16_t *q_up = MlAlloc<uint16_t>((uint64_t)ML_Q_UP_ROWS * K3_Q_LORA_RANK);
		MlAbsorb<<<2048,256>>>(q_b,kv_b_key,q_up,shard * ML_HEADS);
		memset(w,0,sizeof(*w));
		w->attn_norm_weight = attn_norm;
		w->mla_q_down_weight = q_down;
		w->mla_q_norm_weight = q_norm;
		w->mla_kv_a_weight = kv_a;
		w->mla_kv_a_norm_weight = kv_norm;
		w->mla_q_up_weight = q_up;
		w->mla_kv_b_value_weight = kv_b_value + (uint64_t)shard * ML_HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK;
		w->mla_q_b_weight = q_b + (uint64_t)shard * ML_HEADS * K3_MHA_QK * K3_Q_LORA_RANK;
		w->mla_kv_b_key_full_weight = kv_b_key;
		w->mla_kv_b_value_full_weight = kv_b_value;
		w->mla_gate_weight = MlBf16((uint64_t)ML_HEADS * K3_V_HEAD_DIM * K3_HIDDEN,20u + shard,0.02f,0.0f);
		w->mla_out_weight = MlBf16((uint64_t)K3_HIDDEN * ML_HEADS * K3_V_HEAD_DIM,40u + shard,0.05f,0.0f);
	}
	ML_CUDA(cudaDeviceSynchronize());
	{
		LmKvShardView cache = MlCache(table_device,error_device);
		LmTpBf16Contributions<ML_SHARDS> inputs;
		for (uint32_t rank = 0u; rank < ML_SHARDS; rank++)
		{
			MlBuffers(&reference[rank],rank,ML_Q_UP_ROWS,sequence_device,context_device,positions_device);
			reference[rank].cache = cache.pages;
			K3BindLayer(&shards[rank],&reference[rank]);
			ML_CUDA(cudaMemcpy(reference[rank].hidden_bf16,hidden,(uint64_t)ML_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
			MlCheck(K3LayerMla<LmMxfp4,K3GlobalKv>(&reference[rank],ML_ROWS,ML_ROWS,(uint16_t *)0,48u,0) == LM_LAUNCH_OK,"every TP16 rank runs latent attention for its heads");
			ML_CUDA(cudaDeviceSynchronize());
			inputs.rank[rank] = reference[rank].attention_out_bf16;
		}
		LmTpBf16SumKernel<ML_SHARDS><<<ML_ROWS,256u>>>(scratch,inputs,ML_ROWS,K3_HIDDEN);
		ML_CUDA(cudaMemcpy(reference[0].attnres_partial_bf16,partial_in,(uint64_t)ML_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
		K3PartialAdd(&reference[0],scratch,ML_ROWS,0);
		ML_CUDA(cudaDeviceSynchronize());
	}
	MlBuffers(&streamed,0u,ML_HEADS * K3_MHA_QK,sequence_device,context_device,positions_device);
	streamed.cache_shard = MlCache(table_device,error_device);
	streamed.kv_shard = streamed.cache_shard.shard;
	streamed.shard_gather_list = plan_device;
	streamed.shard_gather_offset = plan_device + 1u;
	streamed.shard_gather_context = plan_device + 2u;
	mla.latent_keys = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3GlobalKv::kSlotBytes);
	mla.expanded = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3StreamedHeadKv<ML_HEADS,K3GlobalKv>::kSlotBytes);
	mla.partials = MlAlloc<float>((uint64_t)ML_ROWS * LmGqaShardRecordFloats(ML_HEADS,K3_V_HEAD_DIM,1u));
	mla.key_capacity = ML_ROWS;
	ML_CUDA(cudaMemcpy(streamed.hidden_bf16,hidden,(uint64_t)ML_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	ML_CUDA(cudaMemcpy(streamed.attnres_partial_bf16,partial_in,(uint64_t)ML_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	ML_CUDA(cudaStreamCreate(&stream));
	MlCheck(K3StreamedMla<LmMxfp4,K3GlobalKv,ML_HEADS>(&streamed,shards.data(),ML_SHARDS,&mla,scratch,accumulator,0u,ML_ROWS,ML_ROWS,48u,stream) == LM_LAUNCH_OK,
		"the streamed MLA runs all sixteen head groups on one device");
	ML_CUDA(cudaStreamSynchronize(stream));
	{
		std::vector<uint16_t> want((uint64_t)ML_ROWS * K3_HIDDEN),got(want.size()),before(want.size());
		double worst = 0.0,delta = 0.0,squared = 0.0,error = 0.0;
		uint64_t nonfinite = 0u;
		ML_CUDA(cudaMemcpy(want.data(),reference[0].attnres_partial_bf16,want.size() * 2u,cudaMemcpyDeviceToHost));
		ML_CUDA(cudaMemcpy(got.data(),streamed.attnres_partial_bf16,got.size() * 2u,cudaMemcpyDeviceToHost));
		ML_CUDA(cudaMemcpy(before.data(),partial_in,before.size() * 2u,cudaMemcpyDeviceToHost));
		ML_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
		for (uint64_t index = 0u; index < want.size(); index++)
		{
			const double r = MlHostFloat(want[index]),c = MlHostFloat(got[index]),b = MlHostFloat(before[index]);
			if ( !isfinite(r) || !isfinite(c) )
			{
				nonfinite++;
				continue;
			}
			delta = fmax(delta,fabs(r - b));
			worst = fmax(worst,fabs(r - c));
			squared += (r - b) * (r - b);
			error += (r - c) * (r - c);
		}
		MlCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access errors");
		MlCheck(nonfinite == 0u,"every output element is finite");
		MlCheck(delta > 0.0,"the MLA layer changes the residual");
		MlCheck(sqrt(error / (squared > 0.0 ? squared : 1.0)) <= 0.02,"expanded head-group MLA matches TP16 latent attention within 2% relative RMS of the attention contribution");
		printf("%s streamed MLA vs TP16 latent attention: worst |diff| %.4g, largest attention contribution %.4g, relative RMS %.3g, nonfinite %llu\n",
			ml_failures == 0 ? "PASS" : "FAIL",worst,delta,sqrt(error / (squared > 0.0 ? squared : 1.0)),(unsigned long long)nonfinite);
	}
	return ml_failures != 0;
}
