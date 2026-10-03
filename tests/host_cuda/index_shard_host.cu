#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "inference/kernels/dtype.cuh"
#define __CUDACC__ 1
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/attn_shard.cuh"

#define HOST_THREADS 64u
#define HOST_PAGE 64u
#define HOST_SELECTED 2048u

template<uint32_t WIDTH>
struct HostPageKv
{
	static constexpr uint32_t kSlotBytes = WIDTH * 2u;
	static constexpr uint32_t kPageSlots = HOST_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * HOST_PAGE;
	static constexpr bool kGrows = true;
	static constexpr uint32_t PageOf(uint32_t position) { return position / HOST_PAGE; }
	static constexpr uint32_t SlotInPage(uint32_t position) { return position % HOST_PAGE; }
	static constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + HOST_PAGE - 1u) / HOST_PAGE; }
};

static uint32_t host_state = 20261003u;
static int host_failures;

static uint32_t HostNext(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return host_state;
}

static void HostCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	host_failures++;
}

static void HostTopk(const std::vector<float> &scores,uint32_t rows,uint32_t n,uint32_t k,std::vector<uint32_t> &selected)
{
	selected.assign((uint64_t)rows * k,0xffffffffu);
	HostCheck(LmTopkExactLaunch<HOST_THREADS>(scores.data(),rows,n,k,n,0u,0,0,0u,selected.data(),0) == cudaSuccess,"exact top-k launch");
}

static void HostMergeCase(uint32_t context,uint32_t degree,uint32_t levels)
{
	const uint32_t rows = 3u;
	std::vector<float> dense((uint64_t)rows * context),merged((uint64_t)rows * context,0.0f);
	std::vector<uint32_t> bound(rows),reference,result;
	SparkKvShard shard = {degree,0u,1u};
	const uint32_t stride = SparkKvShardGatherKeys(shard,context);
	const uint32_t keep = stride < HOST_SELECTED ? stride : HOST_SELECTED;
	std::vector<uint2> gathered((uint64_t)degree * rows * keep);
	uint32_t row,position,rank,local,index;
	char label[96];
	snprintf(label,sizeof(label),"ctx%u degree%u levels%u",context,degree,levels);
	for (row=0u; row<rows; row++)
		bound[row] = row == 0u ? context : context - 1u - (HostNext() % (context / 3u));
	for (row=0u; row<rows; row++)
		for (position=0u; position<context; position++)
			dense[(uint64_t)row * context + position] = position < bound[row] ? (levels != 0u ? (float)(HostNext() % levels) : (float)(int32_t)(HostNext() >> 8u) / 8388608.0f) : -INFINITY;
	HostTopk(dense,rows,context,HOST_SELECTED,reference);
	for (rank=0u; rank<degree; rank++)
	{
		std::vector<float> scores((uint64_t)rows * stride,-INFINITY);
		std::vector<uint32_t> local_selected;
		shard.rank = rank;
		for (row=0u; row<rows; row++)
			for (local=0u; local<SparkKvShardLocalKeys(shard,bound[row]); local++)
				scores[(uint64_t)row * stride + local] = dense[(uint64_t)row * context + SparkKvShardLocalPosition(shard,local)];
		HostTopk(scores,rows,stride,keep,local_selected);
		HostCheck(LmIndexShardCandidatePackLaunch<HOST_THREADS>(shard,scores.data(),stride,local_selected.data(),keep,rows,gathered.data() + (uint64_t)rank * rows * keep,0) == cudaSuccess,"candidate pack launch");
	}
	for (index=0u; index<gathered.size(); index++)
		if ( gathered[index].y != LM_INDEX_SHARD_NO_POSITION && (gathered[index].y >= context || SparkKvShardOwner(shard,gathered[index].y) != index / (rows * keep)) )
		{
			HostCheck(0,"a candidate names a position its rank owns");
			break;
		}
	HostCheck(LmIndexShardCandidateScatterLaunch<HOST_THREADS>(gathered.data(),(uint64_t)rows * keep,degree,keep,rows,context,merged.data(),0) == cudaSuccess,"candidate scatter launch");
	HostTopk(merged,rows,context,HOST_SELECTED,result);
	HostCheck(reference == result,"the merged per-rank candidates select exactly the replicated top-k");
	printf("%s merge %s: keep %u of stride %u\n",host_failures == 0 ? "PASS" : "FAIL",label,keep,stride);
}

template<uint32_t WIDTH>
static void HostGatherCase(uint32_t degree,const std::vector<uint32_t> &contexts,uint32_t chunk_slots,uint64_t section)
{
	typedef HostPageKv<WIDTH> Geometry;
	const uint32_t sequences = (uint32_t)contexts.size();
	SparkKvShard shard = {degree,0u,1u};
	uint32_t most = 0u,pages_per_sequence = 0u,sequence,position,rank,total = 0u,pages = 0u,most_keys = 0u;
	for (sequence=0u; sequence<sequences; sequence++)
		most = contexts[sequence] > most ? contexts[sequence] : most;
	pages_per_sequence = (most + HOST_PAGE - 1u) / HOST_PAGE;
	pages = pages_per_sequence * sequences;
	std::vector<uint32_t> table(pages),order(pages),list(sequences),offset(sequences),context(sequences),base(sequences),gathered_table((uint64_t)sequences * pages_per_sequence,0xffffffffu);
	std::vector<uint8_t> replicated((uint64_t)pages * Geometry::kPageBytes),gathered_pages((uint64_t)pages * Geometry::kPageBytes,0u);
	std::vector<uint16_t> values((uint64_t)sequences * most * WIDTH);
	std::vector<uint32_t> token_sequence,token_position;
	std::vector<uint8_t> keys;
	LmKvAccessError error;
	LmKvView view,gathered_view;
	uint32_t page,index,page_cursor = 0u;
	char label[96];
	snprintf(label,sizeof(label),"width%u degree%u sequences%u most%u chunk%u section%llu",WIDTH,degree,sequences,most,chunk_slots,(unsigned long long)section);
	for (page=0u; page<pages; page++)
		order[page] = page;
	for (page=pages; page>1u; page--)
	{
		uint32_t pick = HostNext() % page,swap = order[page - 1u];
		order[page - 1u] = order[pick];
		order[pick] = swap;
	}
	for (index=0u; index<pages; index++)
		table[index] = order[index];
	for (index=0u; index<values.size(); index++)
		values[index] = (uint16_t)HostNext();
	for (sequence=0u; sequence<sequences; sequence++)
	{
		for (position=0u; position<contexts[sequence]; position++)
		{
			token_sequence.push_back(sequence);
			token_position.push_back(position);
		}
		list[sequence] = sequence;
		offset[sequence] = total;
		context[sequence] = contexts[sequence];
		base[sequence] = page_cursor;
		page_cursor += (contexts[sequence] + HOST_PAGE - 1u) / HOST_PAGE;
		total += SparkKvShardGatherKeys(shard,contexts[sequence]);
		most_keys = SparkKvShardGatherKeys(shard,contexts[sequence]) > most_keys ? SparkKvShardGatherKeys(shard,contexts[sequence]) : most_keys;
	}
	std::vector<uint16_t> rows_bf16(token_position.size() * WIDTH);
	for (index=0u; index<token_position.size(); index++)
		memcpy(rows_bf16.data() + (uint64_t)index * WIDTH,values.data() + ((uint64_t)token_sequence[index] * most + token_position[index]) * WIDTH,WIDTH * 2u);
	LmKvAccessErrorReset(&error);
	HostCheck(LmKvViewInitialize(&view,replicated.data(),table.data(),pages_per_sequence,sequences,pages,&error) == 0,"replicated view");
	LM_LAUNCH((LmKvStoreKernel<Geometry,1u>),(uint32_t)token_position.size(),1u,0,0,view,rows_bf16.data(),token_sequence.data(),token_position.data(),(uint32_t)token_position.size(),WIDTH);
	const uint64_t rank_bytes = (uint64_t)total * Geometry::kSlotBytes + section;
	const uint64_t chunk_bytes = chunk_slots != 0u ? (uint64_t)chunk_slots * Geometry::kSlotBytes : rank_bytes;
	const uint64_t chunks = (rank_bytes + chunk_bytes - 1u) / chunk_bytes;
	std::vector<uint8_t> packed;
	packed.assign(chunks * chunk_bytes,0x5au);
	keys.assign((uint64_t)degree * chunks * chunk_bytes,0x5au);
	for (rank=0u; rank<degree; rank++)
	{
		std::vector<uint8_t> pool(SparkKvShardPoolBytes(shard,HOST_PAGE,Geometry::kSlotBytes,pages),0u);
		LmKvShardView local;
		shard.rank = rank;
		HostCheck(LmKvShardViewInitialize<Geometry>(&local,pool.data(),table.data(),pages_per_sequence,sequences,pages,&error,shard) == 0,"shard view");
		LM_LAUNCH((LmKvShardStoreKernel<Geometry,1u>),(uint32_t)token_position.size(),1u,0,0,local,rows_bf16.data(),token_sequence.data(),token_position.data(),(uint32_t)token_position.size(),WIDTH);
		HostCheck((LmKvShardGatherPackLaunch<Geometry,1u>(local,list.data(),offset.data(),context.data(),sequences,most_keys,packed.data() + section,0)) == cudaSuccess,"gather pack launch");
		for (uint64_t chunk=0u; chunk<chunks; chunk++)
			memcpy(keys.data() + (chunk * degree + rank) * chunk_bytes,packed.data() + chunk * chunk_bytes,chunk_bytes);
	}
	HostCheck((LmKvShardGatherUnpackLaunch<Geometry::kSlotBytes,HOST_PAGE,1u>(shard,keys.data(),chunk_bytes,section,list.data(),offset.data(),context.data(),base.data(),sequences,most,pages_per_sequence,gathered_table.data(),gathered_pages.data(),0)) == cudaSuccess,"gather unpack launch");
	HostCheck(LmKvViewInitialize(&gathered_view,gathered_pages.data(),gathered_table.data(),pages_per_sequence,sequences,pages,&error) == 0,"gathered view");
	for (index=0u; index<token_position.size(); index++)
	{
		const uint8_t *full = replicated.data() + (uint64_t)table[token_sequence[index] * pages_per_sequence + token_position[index] / HOST_PAGE] * Geometry::kPageBytes + (uint64_t)(token_position[index] % HOST_PAGE) * Geometry::kSlotBytes;
		const uint8_t *seen = gathered_pages.data() + (uint64_t)gathered_table[token_sequence[index] * pages_per_sequence + token_position[index] / HOST_PAGE] * Geometry::kPageBytes + (uint64_t)(token_position[index] % HOST_PAGE) * Geometry::kSlotBytes;
		if ( memcmp(full,seen,Geometry::kSlotBytes) != 0 )
		{
			HostCheck(0,"every gathered slot holds the replicated bytes");
			break;
		}
	}
	HostCheck(error.error_code == LM_FRAME_ERROR_NONE,"no KV access error in the gather path");
	printf("%s gather %s: %u keys per rank-sequence set\n",host_failures == 0 ? "PASS" : "FAIL",label,total);
}

static void HostQueryPack(void)
{
	const uint32_t rows = 3u,heads = 4u;
	std::vector<uint16_t> latent((uint64_t)rows * heads * 512u),rope((uint64_t)rows * heads * 64u),packed((uint64_t)rows * heads * 576u);
	uint32_t row,head,element,ok = 1u;
	for (element=0u; element<latent.size(); element++)
		latent[element] = (uint16_t)HostNext();
	for (element=0u; element<rope.size(); element++)
		rope[element] = (uint16_t)HostNext();
	HostCheck((LmLatentShardQueryPackLaunch<512u,64u>(latent.data(),rope.data(),heads,rows,packed.data(),0)) == cudaSuccess,"query pack launch");
	for (row=0u; row<rows; row++)
		for (head=0u; head<heads; head++)
			for (element=0u; element<576u; element++)
				if ( packed[((uint64_t)row * heads + head) * 576u + element] != (element < 512u ? latent[((uint64_t)row * heads + head) * 512u + element] : rope[((uint64_t)row * heads + head) * 64u + element - 512u]) )
					ok = 0u;
	HostCheck(ok != 0u,"the packed query is each head's latent followed by its rope part");
}

int main(void)
{
	HostMergeCase(2100u,16u,0u);
	HostMergeCase(9000u,16u,0u);
	HostMergeCase(9000u,16u,5u);
	HostMergeCase(36000u,16u,0u);
	HostMergeCase(36000u,16u,3u);
	HostMergeCase(36000u,4u,7u);
	HostGatherCase<576u>(16u,std::vector<uint32_t>{777u,3000u},0u,0u);
	HostGatherCase<128u>(16u,std::vector<uint32_t>{5000u,64u,1u},0u,0u);
	HostGatherCase<576u>(4u,std::vector<uint32_t>{130u},0u,0u);
	HostGatherCase<576u>(16u,std::vector<uint32_t>{9000u,1300u},96u,0u);
	HostGatherCase<128u>(16u,std::vector<uint32_t>{9000u,1300u},48u,1024u);
	HostQueryPack();
	if ( host_failures != 0 )
	{
		printf("FAIL index shard host: %d failure(s)\n",host_failures);
		return 1;
	}
	printf("PASS index shard host: per-rank top-k candidates merge to exactly the replicated selection (ties included), gathered keys unpack to the replicated bytes, queries pack latent then rope\n");
	return 0;
}
