#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"

#define GATHER_PAGE 64u
#define GATHER_LATENT 512u
#define GATHER_ROPE 64u
#define GATHER_WIDTH (GATHER_LATENT + GATHER_ROPE)
#define GATHER_RECORD LM_LATENT_SHARD_RECORD_FLOATS(GATHER_LATENT)

struct GatherKv
{
	static constexpr uint32_t kSlotBytes = GATHER_WIDTH * 2u;
	static constexpr uint32_t kPageSlots = GATHER_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * GATHER_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / GATHER_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % GATHER_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + GATHER_PAGE - 1u) / GATHER_PAGE; }
};

static int gather_failures;

#define GATHER_CUDA(call) do { cudaError_t gather_error = (call); if ( gather_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(gather_error)); exit(1); } } while (0)

static void GatherCheck(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	gather_failures++;
}

__global__ void GatherFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
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
static T *GatherDevice(const std::vector<T> &host)
{
	T *device;
	GATHER_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	GATHER_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static void GatherCase(uint32_t rows,uint32_t context,uint32_t degree,uint32_t heads_per_rank)
{
	const uint32_t heads = heads_per_rank * degree,pages = (context + GATHER_PAGE - 1u) / GATHER_PAGE;
	const uint64_t query_stride = (uint64_t)rows * heads_per_rank * GATHER_WIDTH,rank_stride = (uint64_t)rows * heads_per_rank * GATHER_RECORD;
	const float scale = 1.0f / sqrtf((float)GATHER_WIDTH);
	SparkKvShard probe = {degree,0u,1u};
	const uint32_t most = SparkKvShardGatherKeys(probe,context);
	const uint64_t key_stride = (uint64_t)most * GatherKv::kSlotBytes;
	std::vector<uint32_t> table(pages),positions(rows),sequences(rows,0u),contexts(1,context),token_sequence(context,0u),token_position(context);
	std::vector<uint32_t> plan(3u,0u);
	std::vector<float> scatter((uint64_t)degree * degree * rank_stride),gather(scatter.size());
	std::vector<uint8_t *> pools(degree);
	uint32_t *table_device,*positions_device,*sequences_device,*contexts_device,*token_sequence_device,*token_position_device,*plan_device;
	uint16_t *values_device,*query_device;
	uint8_t *keys_device;
	float *scatter_device,*gather_device;
	LmKvAccessError *error_device,error_host;
	uint32_t page,row,rank,source,target,index;
	uint64_t mismatches = 0u;
	char label[128];

	snprintf(label,sizeof(label),"rows%u ctx%u degree%u heads%u",rows,context,degree,heads_per_rank);
	for (page=0u; page<pages; page++)
		table[page] = (page * 7u + 3u) % pages;
	for (row=0u; row<rows; row++)
		positions[row] = context - rows + row;
	for (index=0u; index<context; index++)
		token_position[index] = index;
	plan[1] = 0u;
	plan[2] = context;
	table_device = GatherDevice(table);
	positions_device = GatherDevice(positions);
	sequences_device = GatherDevice(sequences);
	contexts_device = GatherDevice(contexts);
	token_sequence_device = GatherDevice(token_sequence);
	token_position_device = GatherDevice(token_position);
	plan_device = GatherDevice(plan);
	GATHER_CUDA(cudaMalloc((void **)&values_device,(uint64_t)context * GATHER_WIDTH * 2u));
	GATHER_CUDA(cudaMalloc((void **)&query_device,(uint64_t)degree * query_stride * 2u));
	GATHER_CUDA(cudaMalloc((void **)&keys_device,(uint64_t)degree * key_stride));
	GATHER_CUDA(cudaMalloc((void **)&scatter_device,scatter.size() * sizeof(float)));
	GATHER_CUDA(cudaMalloc((void **)&gather_device,gather.size() * sizeof(float)));
	GATHER_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	GATHER_CUDA(cudaMemset(scatter_device,0xff,scatter.size() * sizeof(float)));
	GATHER_CUDA(cudaMemset(gather_device,0x7f,gather.size() * sizeof(float)));
	LmKvAccessErrorReset(&error_host);
	GATHER_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	GatherFillKernel<<<512,256>>>(values_device,(uint64_t)context * GATHER_WIDTH,0u);
	GatherFillKernel<<<256,256>>>(query_device,(uint64_t)degree * query_stride,1ull << 60u);
	GATHER_CUDA(cudaGetLastError());
	for (rank=0u; rank<degree; rank++)
	{
		SparkKvShard shard = {degree,rank,1u};
		LmKvShardView local;
		uint64_t bytes = SparkKvShardPoolBytes(shard,GATHER_PAGE,GatherKv::kSlotBytes,pages);
		GATHER_CUDA(cudaMalloc((void **)&pools[rank],bytes));
		GATHER_CUDA(cudaMemset(pools[rank],0,bytes));
		GatherCheck(LmKvShardViewInitialize<GatherKv>(&local,pools[rank],table_device,pages,1u,pages,error_device,shard) == 0,"shard view",label);
		LmKvShardStoreKernel<GatherKv,256u><<<context,256>>>(local,values_device,token_sequence_device,token_position_device,context,GATHER_WIDTH);
		GATHER_CUDA(cudaGetLastError());
		GatherCheck(LmLatentShardPrefillLaunch<GatherKv,LmKvShardView,GATHER_LATENT,GATHER_ROPE>(local,query_device,query_stride,heads_per_rank,
			sequences_device,contexts_device,positions_device,scale,scatter_device + (uint64_t)rank * degree * rank_stride,rank_stride,rows,0) == cudaSuccess,
			"scatter prefill launch",label);
		GatherCheck(LmKvShardGatherPackLaunch<GatherKv,256u>(local,plan_device,plan_device + 1u,plan_device + 2u,1u,most,
			keys_device + (uint64_t)rank * key_stride,0) == cudaSuccess,"gather pack launch",label);
	}
	for (target=0u; target<degree; target++)
		for (source=0u; source<degree; source++)
		{
			SparkKvShard shard = {degree,source,1u};
			LmKvShardView pages_view;
			LmKvShardGatherView view;
			GatherCheck(LmKvShardViewInitialize<GatherKv>(&pages_view,pools[source],table_device,pages,1u,pages,error_device,shard) == 0,"source pages",label);
			GatherCheck(LmKvShardGatherViewInitialize<GatherKv>(&view,pages_view.pages,shard,keys_device + (uint64_t)source * key_stride,plan_device + 1u,plan_device + 2u) == 0,
				"gather view",label);
			GatherCheck(LmLatentGatherPrefillLaunch<GatherKv,LmKvShardGatherView,GATHER_LATENT,GATHER_ROPE>(view,query_device + (uint64_t)target * query_stride,heads_per_rank,
				sequences_device,contexts_device,positions_device,scale,gather_device + ((uint64_t)target * degree + source) * rank_stride,rows,0) == cudaSuccess,
				"gather prefill launch",label);
		}
	GATHER_CUDA(cudaDeviceSynchronize());
	GATHER_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	GatherCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error on a single-sequence wave",label);
	GATHER_CUDA(cudaMemcpy(scatter.data(),scatter_device,scatter.size() * sizeof(float),cudaMemcpyDeviceToHost));
	GATHER_CUDA(cudaMemcpy(gather.data(),gather_device,gather.size() * sizeof(float),cudaMemcpyDeviceToHost));
	for (target=0u; target<degree; target++)
		for (source=0u; source<degree; source++)
			if ( memcmp(&gather[((uint64_t)target * degree + source) * rank_stride],&scatter[((uint64_t)source * degree + target) * rank_stride],rank_stride * sizeof(float)) != 0 )
				mismatches++;
	GatherCheck(mismatches == 0u,"every source's own-head partials equal that rank's scatter partials bit for bit",label);
	{
		const uint64_t out_stride = (uint64_t)rows * heads_per_rank * GATHER_LATENT;
		std::vector<uint16_t> tiled((uint64_t)degree * out_stride),oracle(tiled.size());
		uint16_t *tiled_device,*oracle_device;
		float *row_device,*receive_device;
		double worst = 0.0;
		GATHER_CUDA(cudaMalloc((void **)&tiled_device,tiled.size() * 2u));
		GATHER_CUDA(cudaMalloc((void **)&oracle_device,oracle.size() * 2u));
		GATHER_CUDA(cudaMalloc((void **)&row_device,scatter.size() * sizeof(float)));
		GATHER_CUDA(cudaMalloc((void **)&receive_device,scatter.size() * sizeof(float)));
		for (rank=0u; rank<degree; rank++)
		{
			SparkKvShard shard = {degree,rank,1u};
			LmKvShardView local;
			GatherCheck(LmKvShardViewInitialize<GatherKv>(&local,pools[rank],table_device,pages,1u,pages,error_device,shard) == 0,"oracle view",label);
			GatherCheck(LmLatentShardPartialLaunch<GatherKv,LmKvShardView,GATHER_LATENT,GATHER_ROPE>(local,query_device,query_stride,heads_per_rank,sequences_device,
				contexts_device,positions_device,0,0u,0u,scale,row_device + (uint64_t)rank * degree * rank_stride,rank_stride,rows,0) == cudaSuccess,"per-row partial launch",label);
		}
		for (target=0u; target<degree; target++)
			for (source=0u; source<degree; source++)
			{
				GATHER_CUDA(cudaMemcpy(receive_device + ((uint64_t)target * degree + source) * rank_stride,row_device + ((uint64_t)source * degree + target) * rank_stride,
					rank_stride * sizeof(float),cudaMemcpyDeviceToDevice));
			}
		for (target=0u; target<degree; target++)
		{
			GatherCheck(LmLatentShardMergeLaunch<GATHER_LATENT>(gather_device + (uint64_t)target * degree * rank_stride,rank_stride,degree,heads_per_rank,tiled_device + (uint64_t)target * out_stride,rows,0) == cudaSuccess,"tiled merge",label);
			GatherCheck(LmLatentShardMergeLaunch<GATHER_LATENT>(receive_device + (uint64_t)target * degree * rank_stride,rank_stride,degree,heads_per_rank,oracle_device + (uint64_t)target * out_stride,rows,0) == cudaSuccess,"per-row merge",label);
		}
		GATHER_CUDA(cudaDeviceSynchronize());
		GATHER_CUDA(cudaMemcpy(tiled.data(),tiled_device,tiled.size() * 2u,cudaMemcpyDeviceToHost));
		GATHER_CUDA(cudaMemcpy(oracle.data(),oracle_device,oracle.size() * 2u,cudaMemcpyDeviceToHost));
		for (index=0u; index<tiled.size(); index++)
		{
			uint32_t a = (uint32_t)tiled[index] << 16u,b = (uint32_t)oracle[index] << 16u;
			float fa,fb;
			memcpy(&fa,&a,4u);
			memcpy(&fb,&b,4u);
			double error = fabs((double)fa - (double)fb) / (fabs((double)fb) + 1e-2);
			worst = error > worst ? error : worst;
		}
		GatherCheck(worst <= 1.0 / 64.0,"the tiled gather attention matches the per-row partial kernel within 1/64 relative",label);
		printf("%s %s: worst relative difference to the per-row kernel %.5f\n",worst <= 1.0 / 64.0 ? "ok  " : "FAIL",label,worst);
		cudaFree(tiled_device); cudaFree(oracle_device); cudaFree(row_device); cudaFree(receive_device);
	}
	printf("%s %s: %llu of %u source/target partial blocks differ (heads %u)\n",mismatches == 0u ? "ok  " : "FAIL",label,(unsigned long long)mismatches,degree * degree,heads);

	if ( rows > 8u )
	{
		std::vector<uint32_t> mixed(sequences);
		uint32_t *mixed_device;
		SparkKvShard shard = {degree,0u,1u};
		LmKvShardView pages_view;
		LmKvShardGatherView view;
		mixed[1] = 1u;
		mixed_device = GatherDevice(mixed);
		LmKvAccessErrorReset(&error_host);
		GATHER_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
		GatherCheck(LmKvShardViewInitialize<GatherKv>(&pages_view,pools[0],table_device,pages,2u,pages,error_device,shard) == 0,"two-sequence pages",label);
		GatherCheck(LmKvShardGatherViewInitialize<GatherKv>(&view,pages_view.pages,shard,keys_device,plan_device + 1u,plan_device + 2u) == 0,"two-sequence view",label);
		GatherCheck(LmLatentGatherPrefillLaunch<GatherKv,LmKvShardGatherView,GATHER_LATENT,GATHER_ROPE>(view,query_device,heads_per_rank,mixed_device,contexts_device,
			positions_device,scale,gather_device,rows,0) == cudaSuccess,"two-sequence launch",label);
		GATHER_CUDA(cudaDeviceSynchronize());
		GATHER_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
		GatherCheck(error_host.error_code != LM_FRAME_ERROR_NONE,"a block whose rows span two sequences reports a KV access error instead of attending",label);
		cudaFree(mixed_device);
	}
	for (rank=0u; rank<degree; rank++)
		cudaFree(pools[rank]);
	cudaFree(table_device); cudaFree(positions_device); cudaFree(sequences_device); cudaFree(contexts_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(plan_device);
	cudaFree(values_device); cudaFree(query_device); cudaFree(keys_device);
	cudaFree(scatter_device); cudaFree(gather_device); cudaFree(error_device);
}

int main(void)
{
	GatherCase(17u,640u,16u,6u);
	GatherCase(100u,3000u,16u,6u);
	GatherCase(203u,4099u,16u,6u);
	GatherCase(64u,2048u,8u,12u);
	if ( gather_failures != 0 )
	{
		printf("latent gather prefill: %d failure(s)\n",gather_failures);
		return 1;
	}
	printf("latent gather prefill: PASS\n");
	return 0;
}
