#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"

#define DECODE_PAGE 64u
#define DECODE_LATENT 512u
#define DECODE_ROPE 64u
#define DECODE_WIDTH (DECODE_LATENT + DECODE_ROPE)
#define DECODE_RECORD LM_LATENT_SHARD_RECORD_FLOATS(DECODE_LATENT)
#define DECODE_HEAD_TOLERANCE (1.0 / 64.0)

struct DecodeKv
{
	static constexpr uint32_t kSlotBytes = DECODE_WIDTH * 2u;
	static constexpr uint32_t kPageSlots = DECODE_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * DECODE_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / DECODE_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % DECODE_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + DECODE_PAGE - 1u) / DECODE_PAGE; }
};

static int decode_failures;

#define DECODE_CUDA(call) do { cudaError_t decode_error = (call); if ( decode_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(decode_error)); exit(1); } } while (0)

static void DecodeCheck(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	decode_failures++;
}

__global__ void DecodeFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint64_t z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
		z ^= z >> 31u;
		values[index] = LmFloatToBf16((float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 2048.0f);
	}
}

template<class T>
static T *DecodeDevice(const std::vector<T> &host)
{
	T *device;
	DECODE_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	DECODE_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static float DecodeBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

static void DecodeCase(uint32_t rows,uint32_t context,uint32_t degree,uint32_t heads_per_rank)
{
	const uint32_t pages = (context + DECODE_PAGE - 1u) / DECODE_PAGE;
	const uint64_t query_stride = (uint64_t)rows * heads_per_rank * DECODE_WIDTH,rank_stride = (uint64_t)rows * heads_per_rank * DECODE_RECORD;
	const uint64_t split_floats = LmLatentShardDecodeSplitFloats(rows,heads_per_rank * degree,DECODE_LATENT);
	const uint64_t out_stride = (uint64_t)rows * heads_per_rank * DECODE_LATENT;
	const float scale = 1.0f / sqrtf((float)DECODE_WIDTH);
	std::vector<uint32_t> table(pages),positions(rows),sequences(rows,0u),contexts(1,context),token_sequence(context,0u),token_position(context);
	std::vector<uint8_t *> pools(degree);
	std::vector<uint16_t> tiled((uint64_t)degree * out_stride),oracle(tiled.size());
	uint32_t *table_device,*positions_device,*sequences_device,*contexts_device,*token_sequence_device,*token_position_device;
	uint16_t *values_device,*query_device,*tiled_device,*oracle_device;
	float *split_device,*decode_device,*oracle_send_device,*decode_receive_device,*oracle_receive_device,*pair_send_device;
	LmKvAccessError *error_device,error_host;
	uint32_t page,row,rank,source,target,index;
	double worst = 0.0;
	char label[128];

	snprintf(label,sizeof(label),"rows%u ctx%u degree%u heads%u splits%u",rows,context,degree,heads_per_rank,LmLatentShardDecodeSplits(rows));
	for (page=0u; page<pages; page++)
		table[page] = (page * 7u + 3u) % pages;
	for (row=0u; row<rows; row++)
		positions[row] = context - rows + row;
	for (index=0u; index<context; index++)
		token_position[index] = index;
	table_device = DecodeDevice(table);
	positions_device = DecodeDevice(positions);
	sequences_device = DecodeDevice(sequences);
	contexts_device = DecodeDevice(contexts);
	token_sequence_device = DecodeDevice(token_sequence);
	token_position_device = DecodeDevice(token_position);
	DECODE_CUDA(cudaMalloc((void **)&values_device,(uint64_t)context * DECODE_WIDTH * 2u));
	DECODE_CUDA(cudaMalloc((void **)&query_device,(uint64_t)degree * query_stride * 2u));
	DECODE_CUDA(cudaMalloc((void **)&split_device,split_floats * sizeof(float)));
	DECODE_CUDA(cudaMalloc((void **)&decode_device,(uint64_t)degree * degree * rank_stride * sizeof(float)));
	DECODE_CUDA(cudaMalloc((void **)&oracle_send_device,(uint64_t)degree * degree * rank_stride * sizeof(float)));
	DECODE_CUDA(cudaMalloc((void **)&pair_send_device,(uint64_t)degree * degree * 2u * rank_stride * sizeof(float)));
	DECODE_CUDA(cudaMalloc((void **)&decode_receive_device,(uint64_t)degree * degree * rank_stride * sizeof(float)));
	DECODE_CUDA(cudaMalloc((void **)&oracle_receive_device,(uint64_t)degree * degree * rank_stride * sizeof(float)));
	DECODE_CUDA(cudaMalloc((void **)&tiled_device,tiled.size() * 2u));
	DECODE_CUDA(cudaMalloc((void **)&oracle_device,oracle.size() * 2u));
	DECODE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	DECODE_CUDA(cudaMemset(decode_device,0x7f,(uint64_t)degree * degree * rank_stride * sizeof(float)));
	LmKvAccessErrorReset(&error_host);
	DECODE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	DecodeFillKernel<<<512,256>>>(values_device,(uint64_t)context * DECODE_WIDTH,0u);
	DecodeFillKernel<<<256,256>>>(query_device,(uint64_t)degree * query_stride,1ull << 60u);
	DECODE_CUDA(cudaGetLastError());
	for (rank=0u; rank<degree; rank++)
	{
		SparkKvShard shard = {degree,rank,1u};
		LmKvShardView local;
		uint64_t bytes = SparkKvShardPoolBytes(shard,DECODE_PAGE,DecodeKv::kSlotBytes,pages);
		DECODE_CUDA(cudaMalloc((void **)&pools[rank],bytes));
		DECODE_CUDA(cudaMemset(pools[rank],0,bytes));
		DecodeCheck(LmKvShardViewInitialize<DecodeKv>(&local,pools[rank],table_device,pages,1u,pages,error_device,shard) == 0,"shard view",label);
		LmKvShardStoreKernel<DecodeKv,256u><<<context,256>>>(local,values_device,token_sequence_device,token_position_device,context,DECODE_WIDTH);
		DECODE_CUDA(cudaGetLastError());
		DecodeCheck(LmLatentShardDecodeLaunch<DecodeKv,LmKvShardView,DECODE_LATENT,DECODE_ROPE>(local,query_device,query_stride,heads_per_rank,sequences_device,
			contexts_device,positions_device,scale,split_device,split_floats,decode_device + (uint64_t)rank * degree * rank_stride,rank_stride,rows,0) == cudaSuccess,
			"split decode launch",label);
		DecodeCheck(LmLatentShardPartialLaunch<DecodeKv,LmKvShardView,DECODE_LATENT,DECODE_ROPE>(local,query_device,query_stride,heads_per_rank,sequences_device,
			contexts_device,positions_device,0,0u,0u,scale,oracle_send_device + (uint64_t)rank * degree * rank_stride,rank_stride,rows,0) == cudaSuccess,
			"scalar partial launch",label);
		DecodeCheck(LmLatentShardPartialHeadsLaunch<DecodeKv,LmKvShardView,DECODE_LATENT,DECODE_ROPE>(local,query_device,query_stride,heads_per_rank,
			heads_per_rank * degree,sequences_device,contexts_device,positions_device,0,0u,0u,scale,pair_send_device + (uint64_t)rank * degree * 2u * rank_stride,
			2u * rank_stride,2u * heads_per_rank,2u,rows,0) == cudaSuccess,"pair-grouped partial launch",label);
	}
	DecodeCheck(LmLatentShardDecodeLaunch<DecodeKv,LmKvShardView,DECODE_LATENT,DECODE_ROPE>(LmKvShardView(),query_device,query_stride,heads_per_rank,sequences_device,
		contexts_device,positions_device,scale,split_device,LmLatentShardDecodeSplits(rows) > 1u ? split_floats - 1u : 0u,decode_device,rank_stride,rows,0) != cudaSuccess,
		"a split buffer one float short or an invalid view is refused before launch",label);
	for (target=0u; target<degree; target++)
		for (source=0u; source<degree; source++)
		{
			DECODE_CUDA(cudaMemcpy(decode_receive_device + ((uint64_t)target * degree + source) * rank_stride,decode_device + ((uint64_t)source * degree + target) * rank_stride,
				rank_stride * sizeof(float),cudaMemcpyDeviceToDevice));
			DECODE_CUDA(cudaMemcpy(oracle_receive_device + ((uint64_t)target * degree + source) * rank_stride,oracle_send_device + ((uint64_t)source * degree + target) * rank_stride,
				rank_stride * sizeof(float),cudaMemcpyDeviceToDevice));
		}
	for (target=0u; target<degree; target++)
	{
		DecodeCheck(LmLatentShardMergeLaunch<DECODE_LATENT>(decode_receive_device + (uint64_t)target * degree * rank_stride,rank_stride,degree,heads_per_rank,
			tiled_device + (uint64_t)target * out_stride,rows,0) == cudaSuccess,"split decode merge",label);
		DecodeCheck(LmLatentShardMergeLaunch<DECODE_LATENT>(oracle_receive_device + (uint64_t)target * degree * rank_stride,rank_stride,degree,heads_per_rank,
			oracle_device + (uint64_t)target * out_stride,rows,0) == cudaSuccess,"scalar merge",label);
	}
	DECODE_CUDA(cudaDeviceSynchronize());
	DECODE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	DecodeCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	{
		std::vector<float> single((uint64_t)degree * degree * rank_stride),pair((uint64_t)degree * degree * 2u * rank_stride);
		uint32_t mismatched = 0u,head,copy;
		DECODE_CUDA(cudaMemcpy(single.data(),oracle_send_device,single.size() * sizeof(float),cudaMemcpyDeviceToHost));
		DECODE_CUDA(cudaMemcpy(pair.data(),pair_send_device,pair.size() * sizeof(float),cudaMemcpyDeviceToHost));
		for (rank=0u; rank<degree; rank++)
			for (row=0u; row<rows; row++)
				for (head=0u; head<heads_per_rank * degree; head++)
					for (copy=0u; copy<2u; copy++)
					{
						const float *a = single.data() + ((uint64_t)rank * degree + head / heads_per_rank) * rank_stride + ((uint64_t)row * heads_per_rank + head % heads_per_rank) * DECODE_RECORD;
						const float *b = pair.data() + (uint64_t)rank * degree * 2u * rank_stride + ((uint64_t)(head / (2u * heads_per_rank)) * 2u + copy) * 2u * rank_stride +
							((uint64_t)row * 2u * heads_per_rank + head % (2u * heads_per_rank)) * DECODE_RECORD;
						mismatched += memcmp(a,b,DECODE_RECORD * sizeof(float)) != 0 ? 1u : 0u;
					}
		DecodeCheck(mismatched == 0u,"pair-grouped records hold every head's record for both ranks of its pair, bit for bit the per-rank records",label);
	}
	DECODE_CUDA(cudaMemcpy(tiled.data(),tiled_device,tiled.size() * 2u,cudaMemcpyDeviceToHost));
	DECODE_CUDA(cudaMemcpy(oracle.data(),oracle_device,oracle.size() * 2u,cudaMemcpyDeviceToHost));
	for (index=0u; index<tiled.size(); index+=DECODE_LATENT)
	{
		double peak = 0.0,difference = 0.0;
		uint32_t element;
		for (element=0u; element<DECODE_LATENT; element++)
		{
			const double a = DecodeBf16(tiled[index + element]),b = DecodeBf16(oracle[index + element]),gap = fabs(a - b);
			peak = fabs(b) > peak ? fabs(b) : peak;
			difference = gap > difference || gap != gap ? gap : difference;
		}
		{
			const double error = difference / (peak > 1e-6 ? peak : 1e-6);
			worst = error > worst || error != error ? error : worst;
		}
	}
	DecodeCheck(worst <= DECODE_HEAD_TOLERANCE,"the split tensor-core decode attention matches the scalar partial kernel within two bf16 steps of each head's peak output",label);
	printf("%s %s: worst difference to the scalar kernel relative to the head's peak output %.5f\n",worst <= DECODE_HEAD_TOLERANCE ? "ok  " : "FAIL",label,worst);
	for (rank=0u; rank<degree; rank++)
		cudaFree(pools[rank]);
	cudaFree(table_device); cudaFree(positions_device); cudaFree(sequences_device); cudaFree(contexts_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(values_device); cudaFree(query_device);
	cudaFree(split_device); cudaFree(decode_device); cudaFree(oracle_send_device); cudaFree(pair_send_device); cudaFree(decode_receive_device); cudaFree(oracle_receive_device);
	cudaFree(tiled_device); cudaFree(oracle_device); cudaFree(error_device);
}

int main(void)
{
	DecodeCase(1u,40u,16u,6u);
	DecodeCase(1u,3000u,16u,6u);
	DecodeCase(1u,65536u,16u,6u);
	DecodeCase(2u,9001u,16u,6u);
	DecodeCase(3u,5003u,16u,6u);
	DecodeCase(4u,20000u,16u,6u);
	DecodeCase(8u,4099u,16u,6u);
	DecodeCase(16u,12000u,16u,6u);
	if ( decode_failures != 0 )
	{
		printf("latent shard decode: %d failure(s)\n",decode_failures);
		return 1;
	}
	printf("latent shard decode: PASS\n");
	return 0;
}
