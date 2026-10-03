#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv_shard.cuh"

#define PROBE_PAGE 64u
#define PROBE_HEADS 32u
#define PROBE_DIM 128u
#define PROBE_SELECTED 2048u
#define PROBE_THREADS 256u

struct ProbeIndexKv
{
	static constexpr uint32_t kSlotBytes = PROBE_DIM * 2u;
	static constexpr uint32_t kPageSlots = PROBE_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * PROBE_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / PROBE_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % PROBE_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + PROBE_PAGE - 1u) / PROBE_PAGE; }
};

static int probe_failures;
static uint32_t probe_state = 77u;

static uint32_t ProbeNext(void)
{
	probe_state = probe_state * 1664525u + 1013904223u;
	return probe_state;
}

static uint16_t ProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static void ProbeCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	probe_failures++;
}

#define PROBE_CUDA(call) ProbeCheck((call) == cudaSuccess,#call)

template<class T>
static T *ProbeUpload(const std::vector<T> &host)
{
	T *device = 0;
	PROBE_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	PROBE_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

template<class T>
static std::vector<T> ProbeDownload(const T *device,uint64_t count)
{
	std::vector<T> host(count);
	PROBE_CUDA(cudaMemcpy(host.data(),device,count * sizeof(T),cudaMemcpyDeviceToHost));
	return host;
}

static void ProbeTopk(const float *scores,uint32_t rows,uint32_t n,uint32_t k,uint32_t *selected)
{
	PROBE_CUDA((LmTopkExactLaunch<PROBE_THREADS>(scores,rows,n,k,n,0u,0,0,0u,selected,0)));
}

static void ProbeCase(uint32_t context,uint32_t degree)
{
	const uint32_t rows = 3u,pages = (context + PROBE_PAGE - 1u) / PROBE_PAGE;
	const float scale = 0.0883883f;
	std::vector<uint32_t> table(pages),sequence(rows,0u),bound(rows),position(rows),token_sequence(context,0u),token_position(context),context_length(1,context);
	std::vector<uint16_t> keys((uint64_t)context * PROBE_DIM),query((uint64_t)rows * PROBE_HEADS * PROBE_DIM),weight((uint64_t)rows * PROBE_HEADS);
	uint32_t index,row,rank;
	LmKvAccessError *error = 0;
	LmKvView full;
	char label[96];
	snprintf(label,sizeof(label),"ctx%u degree%u",context,degree);
	for (index=0u; index<pages; index++)
		table[index] = (index * 7u + 3u) % pages;
	for (index=0u; index<context; index++)
		token_position[index] = index;
	for (index=0u; index<keys.size(); index++)
		keys[index] = ProbeBf16((float)((int32_t)(ProbeNext() >> 20u) - 2048) / 2048.0f);
	for (index=0u; index<query.size(); index++)
		query[index] = ProbeBf16((float)((int32_t)(ProbeNext() >> 20u) - 2048) / 2048.0f);
	for (index=0u; index<weight.size(); index++)
		weight[index] = ProbeBf16((float)(ProbeNext() >> 24u) / 256.0f);
	for (row=0u; row<rows; row++)
	{
		bound[row] = row == 0u ? context : context - 1u - ProbeNext() % (context / 3u);
		position[row] = bound[row] - 1u;
	}
	PROBE_CUDA(cudaMallocManaged((void **)&error,sizeof(*error)));
	LmKvAccessErrorReset(error);
	uint32_t *d_table = ProbeUpload(table),*d_sequence = ProbeUpload(sequence),*d_position = ProbeUpload(position),*d_token_sequence = ProbeUpload(token_sequence),*d_token_position = ProbeUpload(token_position),*d_context = ProbeUpload(context_length);
	uint16_t *d_keys = ProbeUpload(keys),*d_query = ProbeUpload(query),*d_weight = ProbeUpload(weight);
	uint8_t *d_full = 0;
	float *d_dense = 0,*d_merged = 0;
	uint32_t *d_reference = 0,*d_result = 0;
	PROBE_CUDA(cudaMalloc((void **)&d_full,(uint64_t)pages * ProbeIndexKv::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&d_dense,(uint64_t)rows * context * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&d_merged,(uint64_t)rows * context * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&d_reference,(uint64_t)rows * PROBE_SELECTED * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&d_result,(uint64_t)rows * PROBE_SELECTED * sizeof(uint32_t)));
	ProbeCheck(LmKvViewInitialize(&full,d_full,d_table,pages,1u,pages,error) == 0,"full view");
	LmKvStoreKernel<ProbeIndexKv,128u><<<context,128u>>>(full,d_keys,d_token_sequence,d_token_position,context,PROBE_DIM);
	PROBE_CUDA((LmWeightedSparseScoreLaunch<ProbeIndexKv,PROBE_HEADS,PROBE_DIM>(d_query,d_weight,full,d_sequence,d_context,d_position,rows,context,scale,d_dense,0)));
	ProbeTopk(d_dense,rows,context,PROBE_SELECTED,d_reference);
	PROBE_CUDA(cudaDeviceSynchronize());
	std::vector<float> dense = ProbeDownload(d_dense,(uint64_t)rows * context);
	SparkKvShard shard = {degree,0u,1u};
	const uint32_t stride = SparkKvShardGatherKeys(shard,context);
	const uint32_t keep = stride < PROBE_SELECTED ? stride : PROBE_SELECTED;
	uint2 *d_gathered = 0;
	PROBE_CUDA(cudaMalloc((void **)&d_gathered,(uint64_t)degree * rows * keep * sizeof(uint2)));
	uint64_t mismatches = 0u;
	for (rank=0u; rank<degree; rank++)
	{
		uint8_t *d_pool = 0;
		float *d_local = 0;
		uint32_t *d_local_selected = 0;
		LmKvShardView local;
		shard.rank = rank;
		PROBE_CUDA(cudaMalloc((void **)&d_pool,SparkKvShardPoolBytes(shard,PROBE_PAGE,ProbeIndexKv::kSlotBytes,pages)));
		PROBE_CUDA(cudaMalloc((void **)&d_local,(uint64_t)rows * stride * sizeof(float)));
		PROBE_CUDA(cudaMalloc((void **)&d_local_selected,(uint64_t)rows * keep * sizeof(uint32_t)));
		ProbeCheck(LmKvShardViewInitialize<ProbeIndexKv>(&local,d_pool,d_table,pages,1u,pages,error,shard) == 0,"shard view");
		LmKvShardStoreKernel<ProbeIndexKv,128u><<<context,128u>>>(local,d_keys,d_token_sequence,d_token_position,context,PROBE_DIM);
		PROBE_CUDA((LmWeightedSparseScoreLaunch<ProbeIndexKv,PROBE_HEADS,PROBE_DIM>(d_query,d_weight,local,d_sequence,d_context,d_position,rows,stride,scale,d_local,0)));
		ProbeTopk(d_local,rows,stride,keep,d_local_selected);
		PROBE_CUDA(LmIndexShardCandidatePackLaunch<PROBE_THREADS>(shard,d_local,stride,d_local_selected,keep,rows,d_gathered + (uint64_t)rank * rows * keep,0));
		PROBE_CUDA(cudaDeviceSynchronize());
		std::vector<float> local_scores = ProbeDownload(d_local,(uint64_t)rows * stride);
		for (row=0u; row<rows; row++)
			for (index=0u; index<stride; index++)
			{
				uint32_t at = SparkKvShardLocalPosition(shard,index);
				float expected = at < bound[row] ? dense[(uint64_t)row * context + at] : -INFINITY;
				if ( memcmp(&expected,&local_scores[(uint64_t)row * stride + index],sizeof(float)) != 0 )
					mismatches++;
			}
		cudaFree(d_pool);
		cudaFree(d_local);
		cudaFree(d_local_selected);
	}
	ProbeCheck(mismatches == 0u,"every sharded score equals the replicated score at its position, bit for bit");
	PROBE_CUDA(LmIndexShardCandidateScatterLaunch<PROBE_THREADS>(d_gathered,(uint64_t)rows * keep,degree,keep,rows,context,d_merged,0));
	ProbeTopk(d_merged,rows,context,PROBE_SELECTED,d_result);
	PROBE_CUDA(cudaDeviceSynchronize());
	ProbeCheck(ProbeDownload(d_reference,(uint64_t)rows * PROBE_SELECTED) == ProbeDownload(d_result,(uint64_t)rows * PROBE_SELECTED),"the merged candidates select exactly the replicated top-k");
	ProbeCheck(error->error_code == LM_FRAME_ERROR_NONE,"no KV access error");
	printf("%s %s: stride %u keep %u, score mismatches %llu\n",probe_failures == 0 ? "PASS" : "FAIL",label,stride,keep,(unsigned long long)mismatches);
	cudaFree(d_table); cudaFree(d_sequence); cudaFree(d_position); cudaFree(d_token_sequence); cudaFree(d_token_position); cudaFree(d_context);
	cudaFree(d_keys); cudaFree(d_query); cudaFree(d_weight); cudaFree(d_full); cudaFree(d_dense); cudaFree(d_merged); cudaFree(d_reference); cudaFree(d_result); cudaFree(d_gathered); cudaFree(error);
}

static void ProbeRemapCase(uint32_t context,uint32_t rows,uint32_t timing)
{
	typedef LmKvGeometry<ProbeIndexKv::kSlotBytes,1u,true> Remap;
	const uint32_t degree = 16u,old_bound = context - rows,pages = (context + PROBE_PAGE - 1u) / PROBE_PAGE,capacity = context + 64u;
	const float scale = 0.0883883f;
	std::vector<uint32_t> table(pages),sequence(rows,0u),position(rows),token_sequence(context,0u),token_position(context),context_length(1,context),old(1,old_bound),list(1,0u),offset(1,0u);
	std::vector<uint16_t> keys((uint64_t)context * PROBE_DIM),query((uint64_t)rows * PROBE_HEADS * PROBE_DIM),weight((uint64_t)rows * PROBE_HEADS);
	uint32_t index,row,rank;
	LmKvAccessError *error = 0;
	LmKvView full,remapped;
	char label[96];
	snprintf(label,sizeof(label),"remap ctx%u rows%u",context,rows);
	for (index=0u; index<pages; index++)
		table[index] = (uint32_t)(((uint64_t)index * 2654435761u) % pages);
	{
		std::vector<uint8_t> used(pages,0u);
		for (index=0u; index<pages; index++)
		{
			while ( used[table[index]] != 0u )
				table[index] = (table[index] + 1u) % pages;
			used[table[index]] = 1u;
		}
	}
	for (index=0u; index<context; index++)
		token_position[index] = index;
	for (index=0u; index<keys.size(); index++)
		keys[index] = ProbeBf16((float)((int32_t)(ProbeNext() >> 20u) - 2048) / 2048.0f);
	for (index=0u; index<query.size(); index++)
		query[index] = ProbeBf16((float)((int32_t)(ProbeNext() >> 20u) - 2048) / 2048.0f);
	for (index=0u; index<weight.size(); index++)
		weight[index] = ProbeBf16((float)(ProbeNext() >> 24u) / 256.0f);
	for (row=0u; row<rows; row++)
		position[row] = old_bound + row;
	PROBE_CUDA(cudaMallocManaged((void **)&error,sizeof(*error)));
	LmKvAccessErrorReset(error);
	uint32_t *d_table = ProbeUpload(table),*d_sequence = ProbeUpload(sequence),*d_position = ProbeUpload(position),*d_token_sequence = ProbeUpload(token_sequence),*d_token_position = ProbeUpload(token_position),*d_context = ProbeUpload(context_length),*d_old = ProbeUpload(old),*d_list = ProbeUpload(list),*d_offset = ProbeUpload(offset);
	uint16_t *d_keys = ProbeUpload(keys),*d_query = ProbeUpload(query),*d_weight = ProbeUpload(weight);
	uint8_t *d_full = 0,*d_pack = 0,*d_received = 0;
	uint32_t *d_remap = 0;
	float *d_dense = 0,*d_remapped = 0,elapsed[2] = {0.0f,0.0f};
	SparkKvShard shard = {degree,0u,1u};
	const uint32_t old_keys = SparkKvShardGatherKeys(shard,old_bound);
	const SparkKvShardSectionLayout layout = SparkKvShardSectionLayoutBuild(old_keys,ProbeIndexKv::kSlotBytes,12288u,1u,1023u);
	const uint64_t pool_bytes = (uint64_t)degree * layout.chunks * layout.chunk_bytes;
	PROBE_CUDA(cudaMalloc((void **)&d_full,(uint64_t)pages * ProbeIndexKv::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&d_pack,(uint64_t)layout.chunks * layout.chunk_bytes + ProbeIndexKv::kSlotBytes));
	PROBE_CUDA(cudaMalloc((void **)&d_received,pool_bytes + (uint64_t)rows * ProbeIndexKv::kSlotBytes));
	PROBE_CUDA(cudaMalloc((void **)&d_remap,(uint64_t)capacity * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&d_dense,(uint64_t)rows * context * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&d_remapped,(uint64_t)rows * context * sizeof(float)));
	ProbeCheck(LmKvViewInitialize(&full,d_full,d_table,pages,1u,pages,error) == 0,"remap full view");
	LmKvStoreKernel<ProbeIndexKv,128u><<<context,128u>>>(full,d_keys,d_token_sequence,d_token_position,context,PROBE_DIM);
	for (rank=0u; rank<degree; rank++)
	{
		uint8_t *d_pool = 0;
		LmKvShardView local;
		shard.rank = rank;
		PROBE_CUDA(cudaMalloc((void **)&d_pool,SparkKvShardPoolBytes(shard,PROBE_PAGE,ProbeIndexKv::kSlotBytes,pages)));
		ProbeCheck(LmKvShardViewInitialize<ProbeIndexKv>(&local,d_pool,d_table,pages,1u,pages,error,shard) == 0,"remap shard view");
		LmKvShardStoreKernel<ProbeIndexKv,128u><<<context,128u>>>(local,d_keys,d_token_sequence,d_token_position,context,PROBE_DIM);
		PROBE_CUDA((LmKvShardGatherPackLaunch<ProbeIndexKv,128u>(local,d_list,d_offset,d_old,1u,old_keys,d_pack,0)));
		for (uint64_t chunk=0u; chunk<layout.chunks; chunk++)
			PROBE_CUDA(cudaMemcpy(d_received + (chunk * degree + rank) * layout.chunk_bytes,d_pack + chunk * layout.chunk_bytes,layout.chunk_bytes,cudaMemcpyDeviceToDevice));
		PROBE_CUDA(cudaDeviceSynchronize());
		cudaFree(d_pool);
	}
	PROBE_CUDA(cudaMemcpy(d_received + pool_bytes,d_keys + (uint64_t)old_bound * PROBE_DIM,(uint64_t)rows * ProbeIndexKv::kSlotBytes,cudaMemcpyDeviceToDevice));
	shard.rank = 0u;
	PROBE_CUDA((LmKvShardRemapLaunch<256u>(shard,layout,d_old,capacity,d_remap,0)));
	PROBE_CUDA((LmKvRowsRemapLaunch<256u>(d_position,rows,(uint32_t)(pool_bytes / ProbeIndexKv::kSlotBytes),capacity,d_remap,error,0)));
	ProbeCheck(LmKvViewInitialize(&remapped,d_received,d_remap,capacity,1u,(uint32_t)(pool_bytes / ProbeIndexKv::kSlotBytes) + rows,error) == 0,"remap view");
	PROBE_CUDA((LmWeightedSparseScoreLaunch<ProbeIndexKv,PROBE_HEADS,PROBE_DIM>(d_query,d_weight,full,d_sequence,d_context,d_position,rows,context,scale,d_dense,0)));
	PROBE_CUDA((LmWeightedSparseScoreLaunch<Remap,PROBE_HEADS,PROBE_DIM>(d_query,d_weight,remapped,d_sequence,d_context,d_position,rows,context,scale,d_remapped,0)));
	PROBE_CUDA(cudaDeviceSynchronize());
	ProbeCheck(ProbeDownload(d_dense,(uint64_t)rows * context) == ProbeDownload(d_remapped,(uint64_t)rows * context),"scores read through the remap equal the paged replicated scores bit for bit");
	ProbeCheck(error->error_code == LM_FRAME_ERROR_NONE,"no KV access error through the remap");
	if ( timing != 0u )
	{
		cudaEvent_t begin,end;
		uint32_t pass,repeat;
		PROBE_CUDA(cudaEventCreate(&begin));
		PROBE_CUDA(cudaEventCreate(&end));
		for (pass=0u; pass<2u; pass++)
		{
			PROBE_CUDA(cudaEventRecord(begin));
			for (repeat=0u; repeat<5u; repeat++)
			{
				if ( pass == 0u )
					PROBE_CUDA((LmWeightedSparseScoreLaunch<ProbeIndexKv,PROBE_HEADS,PROBE_DIM>(d_query,d_weight,full,d_sequence,d_context,d_position,rows,context,scale,d_dense,0)));
				else
					PROBE_CUDA((LmWeightedSparseScoreLaunch<Remap,PROBE_HEADS,PROBE_DIM>(d_query,d_weight,remapped,d_sequence,d_context,d_position,rows,context,scale,d_remapped,0)));
			}
			PROBE_CUDA(cudaEventRecord(end));
			PROBE_CUDA(cudaEventSynchronize(end));
			PROBE_CUDA(cudaEventElapsedTime(&elapsed[pass],begin,end));
			elapsed[pass] /= 5.0f;
		}
		cudaEventDestroy(begin);
		cudaEventDestroy(end);
	}
	printf("%s %s: %u old keys per rank in %u chunk(s)",probe_failures == 0 ? "PASS" : "FAIL",label,old_keys,layout.chunks);
	if ( timing != 0u )
		printf(", score ms paged %.3f remap %.3f (%+.1f%%)",elapsed[0],elapsed[1],100.0f * (elapsed[1] - elapsed[0]) / elapsed[0]);
	printf("\n");
	cudaFree(d_table); cudaFree(d_sequence); cudaFree(d_position); cudaFree(d_token_sequence); cudaFree(d_token_position); cudaFree(d_context); cudaFree(d_old); cudaFree(d_list); cudaFree(d_offset);
	cudaFree(d_keys); cudaFree(d_query); cudaFree(d_weight); cudaFree(d_full); cudaFree(d_pack); cudaFree(d_received); cudaFree(d_remap); cudaFree(d_dense); cudaFree(d_remapped); cudaFree(error);
}

int main(int argc,char **argv)
{
	const uint32_t timing = argc > 1 && strcmp(argv[1],"--time") == 0 ? 1u : 0u;
	ProbeCase(2100u,16u);
	ProbeCase(40000u,16u);
	ProbeCase(70000u,16u);
	ProbeCase(9000u,4u);
	ProbeRemapCase(2048u + 64u,64u,0u);
	ProbeRemapCase(2049u + 64u,64u,0u);
	ProbeRemapCase(16384u,64u,timing);
	ProbeRemapCase(65536u,64u,timing);
	ProbeRemapCase(175000u,64u,timing);
	ProbeRemapCase(260000u,64u,timing);
	if ( probe_failures != 0 )
	{
		printf("FAIL index shard cuda: %d failure(s)\n",probe_failures);
		return 1;
	}
	printf("PASS index shard cuda: sharded index scores equal the replicated scores bit for bit and the merged per-rank candidates select exactly the replicated top-k\n");
	return 0;
}
