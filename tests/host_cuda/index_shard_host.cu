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
#include "inference/kernels/row_digest.cuh"

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

static void HostSectionLayouts(void)
{
	const uint32_t caps[] = {1023u,1008u,3u};
	const uint64_t keys[] = {1u,31u,32u,33u,4096u,10912u,10913u,16384u,65536u};
	const uint64_t slots[] = {1152u,256u};
	const uint32_t aligns[] = {3u,1u};
	uint32_t c,k,s,ok = 1u;
	for (s=0u; s<2u; s++)
		for (c=0u; c<3u; c++)
			for (k=0u; k<sizeof(keys)/sizeof(keys[0]); k++)
			{
				SparkKvShardSectionLayout layout = SparkKvShardSectionLayoutBuild(keys[k],slots[s],12288u,aligns[s],caps[c]);
				if ( caps[c] < aligns[s] )
				{
					ok &= layout.chunks == 0u ? 1u : 0u;
					continue;
				}
				ok &= layout.chunks != 0u && layout.chunk_bytes % slots[s] == 0u && layout.chunk_units % aligns[s] == 0u &&
					layout.chunk_units <= caps[c] / aligns[s] * aligns[s] &&
					(uint64_t)layout.chunks * layout.chunk_bytes >= keys[k] * slots[s] &&
					(uint64_t)(layout.chunks - 1u) * layout.chunk_bytes < keys[k] * slots[s] + (uint64_t)aligns[s] * 12288u * layout.chunks ? 1u : 0u;
			}
	HostCheck(ok,"section layouts: whole slots per chunk, aligned units under the cap, balanced chunks that cover every key");
	HostCheck(SparkKvShardSectionLayoutBuild(10912u,1152u,12288u,3u,1023u).chunks == 1u &&
		SparkKvShardSectionLayoutBuild(10913u,1152u,12288u,3u,1023u).chunks == 2u,"one 1023-unit chunk holds 10912 latent keys; the next key opens a second chunk");
	for (k=0u; k<3u; k++)
	{
		SparkKvShard shard = {16u,0u,1u};
		SparkKvShardSectionLayout layout = SparkKvShardSectionLayoutBuild(keys[k + 5u],1152u,12288u,3u,1023u);
		uint32_t position,bad = 0u;
		for (position=0u; position<keys[k + 5u] * 16u; position += 37u)
		{
			uint64_t local = position / 16u,byte = local * 1152u,owner = position % 16u;
			uint64_t expect = ((byte / layout.chunk_bytes) * 16u * layout.chunk_bytes + owner * layout.chunk_bytes + byte % layout.chunk_bytes) / 1152u;
			bad += SparkKvShardSectionSlot(shard,layout,position) != expect ? 1u : 0u;
		}
		HostCheck(bad == 0u,"section slot equals the brute-force [chunk][rank][bytes] address");
	}
	printf("%s section layouts\n",host_failures == 0 ? "PASS" : "FAIL");
}

template<uint32_t WIDTH>
static void HostRemapCase(uint32_t degree,uint32_t first,uint32_t rows,uint32_t cap_units,uint32_t gap,uint32_t duplicates)
{
	typedef HostPageKv<WIDTH> Geometry;
	typedef LmKvGeometry<Geometry::kSlotBytes,1u,true> Remap;
	std::vector<uint32_t> wave_position;
	uint32_t index,rank,position;
	for (index=0u; index<rows; index++)
		wave_position.push_back(first + index + (gap != 0u && index >= rows / 2u ? gap : 0u));
	for (index=0u; index<duplicates; index++)
		wave_position.push_back(wave_position.back());
	const uint32_t total_rows = (uint32_t)wave_position.size(),context = wave_position.back() + 1u;
	const uint32_t old_bound = gap == 0u ? first : context,pages = (context + HOST_PAGE - 1u) / HOST_PAGE,capacity = context + 64u;
	const uint32_t align = Geometry::kSlotBytes % 1152u == 0u ? 3u : 1u;
	SparkKvShard shard = {degree,0u,1u};
	std::vector<uint32_t> table(pages),token_sequence(context,0u),token_position(context),list(1,0u),offset(1,0u),old(1,old_bound),remap(capacity,0u);
	std::vector<uint16_t> values((uint64_t)context * WIDTH);
	std::vector<uint8_t> replicated((uint64_t)pages * Geometry::kPageBytes);
	LmKvAccessError error;
	LmKvView view,remapped;
	char label[128];
	snprintf(label,sizeof(label),"width%u degree%u first%u rows%u gap%u duplicates%u cap%u",WIDTH,degree,first,rows,gap,duplicates,cap_units);
	for (index=0u; index<pages; index++)
		table[index] = index;
	for (index=pages; index>1u; index--)
	{
		uint32_t pick = HostNext() % index,swap = table[index - 1u];
		table[index - 1u] = table[pick];
		table[pick] = swap;
	}
	for (index=0u; index<context; index++)
		token_position[index] = index;
	for (index=0u; index<values.size(); index++)
		values[index] = (uint16_t)HostNext();
	LmKvAccessErrorReset(&error);
	HostCheck(LmKvViewInitialize(&view,replicated.data(),table.data(),pages,1u,pages,&error) == 0,"remap replicated view");
	LM_LAUNCH((LmKvStoreKernel<Geometry,1u>),context,1u,0,0,view,values.data(),token_sequence.data(),token_position.data(),context,WIDTH);
	const uint32_t keys = SparkKvShardGatherKeys(shard,old_bound);
	const SparkKvShardSectionLayout layout = SparkKvShardSectionLayoutBuild(keys,Geometry::kSlotBytes,12288u,align,cap_units);
	const uint64_t pool_bytes = (uint64_t)degree * layout.chunks * layout.chunk_bytes;
	std::vector<uint8_t> packed((uint64_t)layout.chunks * layout.chunk_bytes + Geometry::kSlotBytes,0x5au),received(pool_bytes + (uint64_t)total_rows * Geometry::kSlotBytes,0xa5u);
	for (rank=0u; rank<degree; rank++)
	{
		std::vector<uint8_t> pool(SparkKvShardPoolBytes(shard,HOST_PAGE,Geometry::kSlotBytes,pages),0u);
		LmKvShardView local;
		shard.rank = rank;
		HostCheck(LmKvShardViewInitialize<Geometry>(&local,pool.data(),table.data(),pages,1u,pages,&error,shard) == 0,"remap shard view");
		LM_LAUNCH((LmKvShardStoreKernel<Geometry,1u>),context,1u,0,0,local,values.data(),token_sequence.data(),token_position.data(),context,WIDTH);
		if ( keys != 0u )
			HostCheck((LmKvShardGatherPackLaunch<Geometry,1u>(local,list.data(),offset.data(),old.data(),1u,keys,packed.data(),0)) == cudaSuccess,"old-context pack launch");
		for (uint64_t chunk=0u; chunk<layout.chunks; chunk++)
			memcpy(received.data() + (chunk * degree + rank) * layout.chunk_bytes,packed.data() + chunk * layout.chunk_bytes,layout.chunk_bytes);
	}
	std::vector<uint16_t> wave_rows((uint64_t)total_rows * WIDTH);
	for (index=0u; index<total_rows; index++)
		memcpy(wave_rows.data() + (uint64_t)index * WIDTH,values.data() + (uint64_t)wave_position[index] * WIDTH,WIDTH * 2u);
	memcpy(received.data() + pool_bytes,wave_rows.data(),(uint64_t)total_rows * Geometry::kSlotBytes);
	shard.rank = 0u;
	HostCheck((LmKvShardRemapLaunch<1u>(shard,layout,old.data(),capacity,remap.data(),0)) == cudaSuccess,"old-context remap launch");
	HostCheck((LmKvRowsRemapLaunch<1u>(wave_position.data(),total_rows,(uint32_t)(pool_bytes / Geometry::kSlotBytes),capacity,remap.data(),&error,0)) == cudaSuccess,"row overlay launch");
	HostCheck(LmKvViewInitialize(&remapped,received.data(),remap.data(),capacity,1u,(uint32_t)(received.size() / Geometry::kSlotBytes),&error) == 0,"remap view");
	for (position=0u; position<context; position++)
	{
		const uint8_t *full = replicated.data() + (uint64_t)table[position / HOST_PAGE] * Geometry::kPageBytes + (uint64_t)(position % HOST_PAGE) * Geometry::kSlotBytes;
		const uint8_t *seen = LmKvSlotRequired<Remap>(remapped,0u,position,position,LM_KV_ACCESS_READ);
		if ( seen == 0 || memcmp(full,seen,Geometry::kSlotBytes) != 0 )
		{
			HostCheck(0,"every position read through the remap holds the replicated bytes");
			break;
		}
	}
	HostCheck(error.error_code == LM_FRAME_ERROR_NONE,"no KV access error inside the context");
	HostCheck(LmKvSlotRequired<Remap>(remapped,0u,context,context,LM_KV_ACCESS_READ) == 0 && error.error_code != LM_FRAME_ERROR_NONE,
		"a position past the context fails loudly");
	printf("%s remap %s: %u old keys per rank in %u chunk(s) of %llu bytes\n",host_failures == 0 ? "PASS" : "FAIL",label,keys,layout.chunks,(unsigned long long)layout.chunk_bytes);
}

static void HostRowDigest(void)
{
	const uint32_t rows = 96u,elements = 576u;
	std::vector<uint16_t> data((uint64_t)(rows + 1u) * elements),reversed((uint64_t)rows * elements);
	std::vector<uint32_t> position(rows + 1u),reversed_position(rows);
	unsigned long long forward = 0u,backward = 0u,changed = 0u,duplicated = 0u,salted = 0u,words[4],digests[2];
	LmFrameError error;
	uint32_t index;
	for (index=0u; index<data.size(); index++)
		data[index] = (uint16_t)HostNext();
	for (index=0u; index<rows; index++)
		position[index] = 5000u + index;
	position[rows] = position[rows - 1u];
	memcpy(data.data() + (uint64_t)rows * elements,data.data() + (uint64_t)(rows - 1u) * elements,elements * 2u);
	for (index=0u; index<rows; index++)
	{
		reversed_position[index] = position[rows - 1u - index];
		memcpy(reversed.data() + (uint64_t)index * elements,data.data() + (uint64_t)(rows - 1u - index) * elements,elements * 2u);
	}
	HostCheck((LmRowDigestLaunch<HOST_THREADS>(data.data(),elements,position.data(),rows,7u,&forward,0)) == cudaSuccess &&
		(LmRowDigestLaunch<HOST_THREADS>(reversed.data(),elements,reversed_position.data(),rows,7u,&backward,0)) == cudaSuccess &&
		forward == backward,"the row digest does not depend on row order");
	HostCheck((LmRowDigestLaunch<HOST_THREADS>(data.data(),elements,position.data(),rows + 1u,7u,&duplicated,0)) == cudaSuccess &&
		duplicated != forward,"a duplicated padded row adds to the digest instead of cancelling");
	HostCheck((LmRowDigestLaunch<HOST_THREADS>(data.data(),elements,position.data(),rows,8u,&salted,0)) == cudaSuccess &&
		salted != forward,"the salt separates layers");
	reversed[(uint64_t)(rows / 2u) * elements + 3u] ^= 0x10u;
	HostCheck((LmRowDigestLaunch<HOST_THREADS>(reversed.data(),elements,reversed_position.data(),rows,7u,&changed,0)) == cudaSuccess &&
		changed != forward,"one changed bit changes the digest");
	digests[0] = forward;
	digests[1] = salted;
	HostCheck(LmRowDigestFinishLaunch(digests,2u,words,0) == cudaSuccess && words[0] == forward && words[1] == ~forward,"finish writes each digest and its complement");
	LmFrameErrorReset(&error);
	HostCheck(LmRowDigestCheckLaunch(words,2u,&error,0) == cudaSuccess && error.error_code == LM_FRAME_ERROR_NONE,"equal digests on every rank pass the check");
	words[0] = forward > changed ? forward : changed;
	words[1] = ~(forward < changed ? forward : changed);
	HostCheck(LmRowDigestCheckLaunch(words,2u,&error,0) == cudaSuccess && error.error_code == LM_FRAME_ERROR_ROW_DIGEST_MISMATCH,
		"a rank with different bytes fails the check loudly");
	printf("%s row digest\n",host_failures == 0 ? "PASS" : "FAIL");
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
	HostSectionLayouts();
	HostRemapCase<576u>(16u,8192u,1024u,1023u,0u,0u);
	HostRemapCase<128u>(16u,8192u,1024u,1023u,0u,3u);
	HostRemapCase<576u>(16u,0u,700u,1023u,0u,0u);
	HostRemapCase<576u>(4u,3001u,777u,1023u,0u,1u);
	HostRemapCase<576u>(16u,175000u,64u,1023u,0u,0u);
	HostRemapCase<128u>(16u,70001u,1024u,1008u,0u,0u);
	HostRemapCase<576u>(16u,5000u,300u,1023u,17u,2u);
	HostRowDigest();
	HostQueryPack();
	if ( host_failures != 0 )
	{
		printf("FAIL index shard host: %d failure(s)\n",host_failures);
		return 1;
	}
	printf("PASS index shard host: per-rank top-k candidates merge to exactly the replicated selection (ties included), gathered keys unpack to the replicated bytes, queries pack latent then rope\n");
	return 0;
}
