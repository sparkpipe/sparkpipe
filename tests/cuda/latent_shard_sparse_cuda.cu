#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"

#define SPARSE_PAGE 64u
#define SPARSE_LATENT 512u
#define SPARSE_RECORD LM_LATENT_SHARD_RECORD_FLOATS(SPARSE_LATENT)
#define SPARSE_TOLERANCE (1.0 / 8192.0)

template<uint32_t WIDTH>
struct SparseKv
{
	static constexpr uint32_t kSlotBytes = WIDTH * 2u;
	static constexpr uint32_t kPageSlots = SPARSE_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * SPARSE_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / SPARSE_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % SPARSE_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + SPARSE_PAGE - 1u) / SPARSE_PAGE; }
};

static int sparse_failures;

#define SPARSE_CUDA(call) do { cudaError_t sparse_error = (call); if ( sparse_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(sparse_error)); exit(1); } } while (0)

static void SparseCheck(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	sparse_failures++;
}

__global__ void SparseFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint64_t z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
		z ^= z >> 31u;
		values[index] = LmFloatToBf16((float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 1024.0f);
	}
}

template<class T>
static T *SparseDevice(const std::vector<T> &host)
{
	T *device;
	SPARSE_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T) + 16u));
	SPARSE_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static uint64_t SparseNext(uint64_t *state)
{
	*state = *state * 6364136223846793005ull + 1442695040888963407ull;
	return *state >> 33u;
}

static double SparseCompare(const std::vector<float> &a,const std::vector<float> &b,uint64_t records,uint32_t *empty_mismatch)
{
	double worst = 0.0;
	*empty_mismatch = 0u;
	for (uint64_t record = 0u; record < records; record++)
	{
		const float *x = a.data() + record * SPARSE_RECORD,*y = b.data() + record * SPARSE_RECORD;
		if ( x[1] == 0.0f || y[1] == 0.0f )
		{
			*empty_mismatch += (x[1] == 0.0f) != (y[1] == 0.0f) ? 1u : 0u;
			continue;
		}
		double peak = 1e-9,gap = 0.0;
		for (uint32_t element = 0u; element < SPARSE_LATENT; element++)
		{
			const double u = x[2u + element] / x[1],v = y[2u + element] / y[1];
			peak = fabs(u) > peak ? fabs(u) : peak;
			gap = fabs(u - v) > gap || u != u || v != v ? fabs(u - v) : gap;
		}
		const double lse = fabs(((double)x[0] + log((double)x[1])) - ((double)y[0] + log((double)y[1])));
		worst = gap / peak > worst ? gap / peak : worst;
		worst = lse > worst ? lse : worst;
	}
	return worst;
}

template<uint32_t ROPE>
static void SparseCase(uint32_t rows,uint32_t context,uint32_t degree,uint32_t heads_per_rank,uint32_t selected_count,uint32_t dense_limit,uint32_t timed)
{
	typedef SparseKv<SPARSE_LATENT + ROPE> Kv;
	constexpr uint32_t WIDTH = SPARSE_LATENT + ROPE;
	const uint32_t pages = (context + SPARSE_PAGE - 1u) / SPARSE_PAGE,total_heads = heads_per_rank * degree,paired = degree % 2u == 0u ? 1u : 0u;
	const uint64_t query_stride = (uint64_t)rows * heads_per_rank * WIDTH,rank_stride = (uint64_t)rows * heads_per_rank * SPARSE_RECORD;
	const uint64_t pair_stride = 2u * rank_stride,rank_floats = (uint64_t)degree * rank_stride;
	const float scale = 1.0f / sqrtf((float)WIDTH);
	std::vector<uint32_t> table(pages),positions(rows),sequences(rows,0u),contexts(1,context),token_sequence(context,0u),token_position(context);
	std::vector<uint32_t> selected((uint64_t)rows * selected_count),wrong;
	uint64_t state = 0x5eedull + rows * 31u + context;
	char label[160];
	snprintf(label,sizeof(label),"rows%u ctx%u degree%u heads%u selected%u limit%u width%u",rows,context,degree,heads_per_rank,selected_count,dense_limit,WIDTH);
	for (uint32_t page = 0u; page < pages; page++)
		table[page] = (page * 7u + 3u) % pages;
	for (uint32_t row = 0u; row < rows; row++)
		positions[row] = (uint32_t)(((uint64_t)row + 1u) * (context - 1u) / rows);
	if ( rows > 3u )
		positions[rows / 3u] = 0xffffffffu;
	for (uint32_t index = 0u; index < context; index++)
		token_position[index] = index;
	for (uint32_t row = 0u; row < rows; row++)
	{
		uint32_t *list = selected.data() + (uint64_t)row * selected_count;
		const uint32_t bound = positions[row] == 0xffffffffu ? 0u : positions[row];
		for (uint32_t index = 0u; index < selected_count; index++)
		{
			const uint64_t pick = SparseNext(&state);
			list[index] = index % 37u == 5u ? 0xffffffffu : index % 41u == 7u && bound + 1u < context ? bound + 1u + (uint32_t)(pick % (context - bound - 1u)) :
				(uint32_t)(pick % ((uint64_t)bound + 1u));
		}
	}
	wrong = selected;
	{
		uint32_t *list = wrong.data() + (uint64_t)(rows - 1u) * selected_count;
		for (uint32_t index = 0u; index < selected_count; index += 3u)
			if ( list[index] != 0xffffffffu && list[index] <= positions[rows - 1u] )
				list[index] = (list[index] + 1u) % (positions[rows - 1u] + 1u);
	}
	uint32_t *table_device = SparseDevice(table),*positions_device = SparseDevice(positions),*sequences_device = SparseDevice(sequences);
	uint32_t *contexts_device = SparseDevice(contexts),*token_sequence_device = SparseDevice(token_sequence),*token_position_device = SparseDevice(token_position);
	uint32_t *selected_device = SparseDevice(selected),*wrong_device = SparseDevice(wrong),*split_positions,*split_offsets;
	uint16_t *values_device,*query_device;
	float *scalar_device,*tensor_device,*split_device,*alone_device,*pair_scalar_device,*pair_tensor_device,*wrong_out_device;
	LmKvAccessError *error_device,error_host;
	SPARSE_CUDA(cudaMalloc((void **)&values_device,(uint64_t)context * WIDTH * 2u));
	SPARSE_CUDA(cudaMalloc((void **)&query_device,(uint64_t)degree * query_stride * 2u));
	SPARSE_CUDA(cudaMalloc((void **)&scalar_device,(uint64_t)degree * rank_floats * sizeof(float)));
	SPARSE_CUDA(cudaMalloc((void **)&tensor_device,(uint64_t)degree * rank_floats * sizeof(float)));
	SPARSE_CUDA(cudaMalloc((void **)&split_device,(uint64_t)degree * rank_floats * sizeof(float)));
	SPARSE_CUDA(cudaMalloc((void **)&alone_device,(uint64_t)degree * rank_floats * sizeof(float)));
	SPARSE_CUDA(cudaMalloc((void **)&wrong_out_device,(uint64_t)degree * rank_floats * sizeof(float)));
	SPARSE_CUDA(cudaMalloc((void **)&pair_scalar_device,(uint64_t)degree * degree * pair_stride * sizeof(float)));
	SPARSE_CUDA(cudaMalloc((void **)&pair_tensor_device,(uint64_t)degree * degree * pair_stride * sizeof(float)));
	SPARSE_CUDA(cudaMalloc((void **)&split_positions,(uint64_t)rows * selected_count * sizeof(uint32_t)));
	SPARSE_CUDA(cudaMalloc((void **)&split_offsets,(uint64_t)rows * (degree + 1u) * sizeof(uint32_t)));
	SPARSE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	SPARSE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	SparseFillKernel<<<512,256>>>(values_device,(uint64_t)context * WIDTH,0u);
	SparseFillKernel<<<256,256>>>(query_device,(uint64_t)degree * query_stride,1ull << 60u);
	SPARSE_CUDA(cudaGetLastError());
	float scalar_ms = 0.0f,tensor_ms = 0.0f;
	const uint32_t checked_ranks = timed != 0u ? 1u : degree;
	for (uint32_t rank = 0u; rank < checked_ranks; rank++)
	{
		SparkKvShard shard = {degree,rank,1u};
		LmKvShardView local;
		uint8_t *pool;
		const uint64_t bytes = SparkKvShardPoolBytes(shard,SPARSE_PAGE,Kv::kSlotBytes,pages);
		const uint64_t at = (uint64_t)rank * rank_floats;
		SPARSE_CUDA(cudaMalloc((void **)&pool,bytes));
		SPARSE_CUDA(cudaMemset(pool,0,bytes));
		SparseCheck(LmKvShardViewInitialize<Kv>(&local,pool,table_device,pages,1u,pages,error_device,shard) == 0,"shard view",label);
		LmKvShardStoreKernel<Kv,256u><<<context,256>>>(local,values_device,token_sequence_device,token_position_device,context,WIDTH);
		SPARSE_CUDA(cudaGetLastError());
		cudaEvent_t begin,end;
		SPARSE_CUDA(cudaEventCreate(&begin));
		SPARSE_CUDA(cudaEventCreate(&end));
		for (uint32_t repeat = 0u; repeat < (timed != 0u ? 3u : 1u); repeat++)
		{
			SPARSE_CUDA(cudaEventRecord(begin));
			SparseCheck(LmLatentShardPartialHeadsLaunch<Kv,LmKvShardView,SPARSE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,total_heads,sequences_device,
				contexts_device,positions_device,selected_device,selected_count,dense_limit,scale,scalar_device + at,rank_stride,heads_per_rank,1u,rows,0) == cudaSuccess,
				"scalar launch",label);
			SPARSE_CUDA(cudaEventRecord(end));
			SPARSE_CUDA(cudaEventSynchronize(end));
			SPARSE_CUDA(cudaEventElapsedTime(&scalar_ms,begin,end));
			SPARSE_CUDA(cudaEventRecord(begin));
			SparseCheck(LmLatentShardAttendHeadsLaunch<Kv,LmKvShardView,SPARSE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,total_heads,sequences_device,
				contexts_device,positions_device,selected_device,selected_count,dense_limit,scale,tensor_device + at,rank_stride,heads_per_rank,1u,rows,0) == cudaSuccess,
				"tensor launch",label);
			SPARSE_CUDA(cudaEventRecord(end));
			SPARSE_CUDA(cudaEventSynchronize(end));
			SPARSE_CUDA(cudaEventElapsedTime(&tensor_ms,begin,end));
		}
		if ( timed == 0u )
		{
			SparseCheck(LmLatentShardSplitLaunch(shard,selected_device,selected_count,positions_device,split_positions,split_offsets,rows,0) == cudaSuccess,"split lists",label);
			SparseCheck(LmLatentShardAttendHeadsLaunch<Kv,LmKvShardView,SPARSE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,total_heads,sequences_device,
				contexts_device,positions_device,selected_device,selected_count,dense_limit,scale,split_device + at,rank_stride,heads_per_rank,1u,rows,0,split_positions,
				split_offsets) == cudaSuccess,"tensor launch on split lists",label);
			for (uint32_t row = 0u; row < rows; row++)
				SparseCheck(LmLatentShardAttendHeadsLaunch<Kv,LmKvShardView,SPARSE_LATENT,ROPE>(local,query_device + (uint64_t)row * heads_per_rank * WIDTH,query_stride,
					heads_per_rank,total_heads,sequences_device + row,contexts_device,positions_device + row,selected_device + (uint64_t)row * selected_count,selected_count,
					dense_limit,scale,alone_device + at + (uint64_t)row * heads_per_rank * SPARSE_RECORD,rank_stride,heads_per_rank,1u,1u,0) == cudaSuccess,
					"one-row tensor launch",label);
			SparseCheck(LmLatentShardAttendHeadsLaunch<Kv,LmKvShardView,SPARSE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,total_heads,sequences_device,
				contexts_device,positions_device,wrong_device,selected_count,dense_limit,scale,wrong_out_device + at,rank_stride,heads_per_rank,1u,rows,0) == cudaSuccess,
				"tensor launch on a corrupted list",label);
			if ( paired != 0u )
			{
				SparseCheck(LmLatentShardPartialHeadsLaunch<Kv,LmKvShardView,SPARSE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,total_heads,sequences_device,
					contexts_device,positions_device,selected_device,selected_count,dense_limit,scale,pair_scalar_device + (uint64_t)rank * degree * pair_stride,pair_stride,
					2u * heads_per_rank,2u,rows,0) == cudaSuccess,"scalar pair-grouped launch",label);
				SparseCheck(LmLatentShardAttendHeadsLaunch<Kv,LmKvShardView,SPARSE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,total_heads,sequences_device,
					contexts_device,positions_device,selected_device,selected_count,dense_limit,scale,pair_tensor_device + (uint64_t)rank * degree * pair_stride,pair_stride,
					2u * heads_per_rank,2u,rows,0) == cudaSuccess,"tensor pair-grouped launch",label);
			}
		}
		SPARSE_CUDA(cudaDeviceSynchronize());
		SPARSE_CUDA(cudaEventDestroy(begin));
		SPARSE_CUDA(cudaEventDestroy(end));
		cudaFree(pool);
	}
	SPARSE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	SparseCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	{
		const uint64_t used = (uint64_t)checked_ranks * rank_floats,used_records = used / SPARSE_RECORD;
		std::vector<float> scalar(used),tensor(used),split(used),alone(used),wrong_out(used),pair_scalar((uint64_t)checked_ranks * degree * pair_stride),pair_tensor(pair_scalar.size());
		uint32_t empty_mismatch = 0u,pair_empty = 0u,wrong_empty = 0u;
		SPARSE_CUDA(cudaMemcpy(scalar.data(),scalar_device,used * sizeof(float),cudaMemcpyDeviceToHost));
		SPARSE_CUDA(cudaMemcpy(tensor.data(),tensor_device,used * sizeof(float),cudaMemcpyDeviceToHost));
		const double worst = SparseCompare(scalar,tensor,used_records,&empty_mismatch);
		SparseCheck(worst <= SPARSE_TOLERANCE && empty_mismatch == 0u,
			"the tensor-core partials match the scalar kernel on dense rows, selected lists with padding and future positions, and empty rows",label);
		if ( timed == 0u )
		{
			SPARSE_CUDA(cudaMemcpy(split.data(),split_device,used * sizeof(float),cudaMemcpyDeviceToHost));
			SPARSE_CUDA(cudaMemcpy(alone.data(),alone_device,used * sizeof(float),cudaMemcpyDeviceToHost));
			SPARSE_CUDA(cudaMemcpy(wrong_out.data(),wrong_out_device,used * sizeof(float),cudaMemcpyDeviceToHost));
			SPARSE_CUDA(cudaMemcpy(pair_scalar.data(),pair_scalar_device,pair_scalar.size() * sizeof(float),cudaMemcpyDeviceToHost));
			SPARSE_CUDA(cudaMemcpy(pair_tensor.data(),pair_tensor_device,pair_tensor.size() * sizeof(float),cudaMemcpyDeviceToHost));
			SparseCheck(memcmp(split.data(),tensor.data(),used * sizeof(float)) == 0,"precomputed split lists give the in-kernel compaction's records bit for bit",label);
			SparseCheck(memcmp(alone.data(),tensor.data(),used * sizeof(float)) == 0,"every row launched alone gives its record in the batch bit for bit",label);
			const double pair_worst = SparseCompare(pair_scalar,pair_tensor,pair_scalar.size() / SPARSE_RECORD,&pair_empty);
			uint32_t copies_equal = 1u;
			for (uint32_t rank = 0u; rank < checked_ranks; rank++)
				for (uint32_t group = 0u; group < degree / 2u; group++)
				{
					const float *first = pair_tensor.data() + (uint64_t)rank * degree * pair_stride + (uint64_t)group * 2u * pair_stride;
					copies_equal &= memcmp(first,first + pair_stride,pair_stride * sizeof(float)) == 0 ? 1u : 0u;
				}
			SparseCheck(paired == 0u || (pair_worst <= SPARSE_TOLERANCE && pair_empty == 0u && copies_equal != 0u),
				"pair-grouped records hold each head for both ranks of its pair and match the scalar kernel's",label);
			const double wrong_worst = SparseCompare(scalar,wrong_out,used_records,&wrong_empty);
			SparseCheck(wrong_worst > SPARSE_TOLERANCE,"negative control: a corrupted selected list on one row is caught by the comparison",label);
			printf("%s %s: worst %.5f, corrupted-list control %.5f\n",worst <= SPARSE_TOLERANCE ? "ok  " : "FAIL",label,worst,wrong_worst);
		}
		else
			printf("%s %s: scalar %.3f ms tensor %.3f ms (%.1fx), worst %.5f\n",worst <= SPARSE_TOLERANCE ? "ok  " : "FAIL",label,scalar_ms,tensor_ms,scalar_ms / tensor_ms,worst);
	}
	cudaFree(table_device); cudaFree(positions_device); cudaFree(sequences_device); cudaFree(contexts_device); cudaFree(token_sequence_device); cudaFree(token_position_device);
	cudaFree(selected_device); cudaFree(wrong_device); cudaFree(split_positions); cudaFree(split_offsets); cudaFree(values_device); cudaFree(query_device);
	cudaFree(scalar_device); cudaFree(tensor_device); cudaFree(split_device); cudaFree(alone_device); cudaFree(wrong_out_device); cudaFree(pair_scalar_device);
	cudaFree(pair_tensor_device); cudaFree(error_device);
}

int main(void)
{
	SparseCase<0u>(48u,4096u,4u,16u,320u,256u,0u);
	SparseCase<64u>(48u,4096u,4u,16u,320u,256u,0u);
	SparseCase<0u>(37u,9000u,16u,4u,2112u,2048u,0u);
	SparseCase<64u>(19u,6000u,16u,4u,2048u,2048u,0u);
	SparseCase<0u>(37u,9000u,1u,4u,2112u,2048u,0u);
	SparseCase<64u>(19u,6000u,1u,8u,2048u,2048u,0u);
	SparseCase<0u>(4096u,16384u,16u,4u,2112u,2048u,1u);
	SparseCase<64u>(4096u,16384u,16u,4u,2048u,2048u,1u);
	SparseCase<0u>(4096u,16384u,1u,4u,2112u,2048u,1u);
	SparseCase<64u>(4096u,16384u,16u,4u,2112u,16384u,1u);
	SparseCase<64u>(1u,131072u,16u,4u,2112u,2048u,1u);
	SparseCase<64u>(4u,131072u,16u,4u,2112u,2048u,1u);
	SparseCase<64u>(16u,131072u,16u,4u,2112u,2048u,1u);
	if ( sparse_failures != 0 )
	{
		printf("FAIL latent_shard_sparse_cuda: %d checks failed\n",sparse_failures);
		return 1;
	}
	printf("PASS latent_shard_sparse_cuda: the tensor-core shard attention matches the scalar kernel on selected lists, split lists, pair-grouped records, padded head groups and empty rows, and is row-invariant\n");
	return 0;
}
