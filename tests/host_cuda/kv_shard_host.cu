#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "inference/kernels/dtype.cuh"
#define __CUDACC__ 1

#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/attn_shard.cuh"

#define HOST_PAGE 64u
#define HOST_LATENT 512u
#define HOST_DENSE_LIMIT 2048u
#define HOST_SELECTED 2051u

template<uint32_t WIDTH>
struct HostShardKv
{
	static constexpr uint32_t kSlotBytes = WIDTH * 2u;
	static constexpr uint32_t kPageSlots = HOST_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * HOST_PAGE;
	static constexpr bool kGrows = true;
	static constexpr uint32_t PageOf(uint32_t position) { return position / HOST_PAGE; }
	static constexpr uint32_t SlotInPage(uint32_t position) { return position % HOST_PAGE; }
	static constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + HOST_PAGE - 1u) / HOST_PAGE; }
};

static uint32_t host_state = 20260928u;
static int host_failures;

static uint32_t HostNext(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return host_state;
}

static float HostSigned(void)
{
	return (float)((int32_t)(HostNext() >> 20u) - 2048) / 2048.0f;
}

static uint16_t HostBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static double HostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

static void HostCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	host_failures++;
}

struct HostCase
{
	uint32_t rows,context,degree,heads_per_rank,grain;
};

template<uint32_t ROPE>
static void HostRunCase(HostCase test)
{
	typedef HostShardKv<HOST_LATENT + ROPE> Geometry;
	const uint32_t width = HOST_LATENT + ROPE, record = HOST_LATENT + 2u;
	const uint32_t heads = test.heads_per_rank * test.degree;
	const uint32_t pages_per_sequence = (test.context + HOST_PAGE - 1u) / HOST_PAGE;
	const uint32_t pages = pages_per_sequence * test.rows;
	const uint64_t tokens = (uint64_t)test.rows * test.context;
	const float scale = 1.0f / sqrtf((float)width);
	std::vector<uint32_t> table(pages),order(pages),contexts(test.rows),row_position(test.rows),sequences(test.rows);
	std::vector<uint32_t> token_sequence(tokens),token_position(tokens),selected((uint64_t)test.rows * HOST_SELECTED,0xffffffffu);
	std::vector<uint16_t> values(tokens * width),query((uint64_t)heads * test.rows * width);
	std::vector<uint8_t> replicated((uint64_t)pages * Geometry::kPageBytes,0u);
	std::vector<std::vector<uint8_t> > shards(test.degree);
	std::vector<float> oracle_send((uint64_t)test.degree * test.degree * test.rows * test.heads_per_rank * record);
	std::vector<float> shard_send(oracle_send.size()),receive(oracle_send.size());
	std::vector<uint16_t> oracle_out((uint64_t)test.degree * test.rows * test.heads_per_rank * HOST_LATENT);
	std::vector<uint16_t> shard_out(oracle_out.size()),gather_out(oracle_out.size());
	LmKvAccessError error;
	LmKvView view;
	SparkKvShard shard;
	uint64_t rank_stride = (uint64_t)test.rows * test.heads_per_rank * record,query_stride = (uint64_t)test.rows * test.heads_per_rank * width;
	uint64_t shard_bytes = 0u,owned_tokens = 0u;
	uint32_t row,page,rank,index,listed = 0u;
	double worst = 0.0;
	char label[160];

	snprintf(label,sizeof(label),"B%u ctx%u degree%u heads%u grain%u rope%u",test.rows,test.context,test.degree,heads,test.grain,ROPE);
	for (page=0u; page<pages; page++)
		order[page] = page;
	for (page=pages; page>1u; page--)
	{
		uint32_t pick = HostNext() % page,swap = order[page - 1u];
		order[page - 1u] = order[pick];
		order[pick] = swap;
	}
	for (row=0u; row<test.rows; row++)
	{
		sequences[row] = row;
		contexts[row] = test.context - (row * 37u) % (test.context / 4u);
		row_position[row] = contexts[row] - 1u;
		for (page=0u; page<pages_per_sequence; page++)
			table[row * pages_per_sequence + page] = order[row * pages_per_sequence + page];
		for (index=0u; index<test.context; index++)
		{
			token_sequence[(uint64_t)row * test.context + index] = row;
			token_position[(uint64_t)row * test.context + index] = index;
		}
		if ( contexts[row] > HOST_DENSE_LIMIT )
		{
			uint32_t pools = contexts[row] / 4u,chosen = 0u,start = HostNext() % pools;
			listed++;
			for (index=0u; index<pools && chosen<HOST_SELECTED / 4u; index++)
			{
				uint32_t pool = (start + index * 7919u) % pools,token;
				for (token=0u; token<4u; token++)
					selected[(uint64_t)row * HOST_SELECTED + chosen * 4u + token] = pool * 4u + token;
				chosen++;
			}
			for (index=0u; index<contexts[row] % 4u; index++)
				selected[(uint64_t)row * HOST_SELECTED + chosen * 4u + index] = pools * 4u + index;
		}
	}
	for (index=0u; index<values.size(); index++)
		values[index] = HostBf16(HostSigned());
	for (index=0u; index<query.size(); index++)
		query[index] = HostBf16(HostSigned());
	LmKvAccessErrorReset(&error);
	HostCheck(LmKvViewInitialize(&view,replicated.data(),table.data(),pages_per_sequence,test.rows,pages,&error) == 0,"replicated view");
	LM_LAUNCH((LmKvStoreKernel<Geometry,1u>),(uint32_t)tokens,1u,0,0,view,values.data(),token_sequence.data(),token_position.data(),(uint32_t)tokens,width);
	for (rank=0u; rank<test.degree; rank++)
	{
		LmKvShardView local;
		LmKvShardReplicaView replica;
		shard.degree = test.degree;
		shard.rank = rank;
		shard.grain = test.grain;
		shards[rank].assign(SparkKvShardPoolBytes(shard,HOST_PAGE,Geometry::kSlotBytes,pages),0u);
		shard_bytes += shards[rank].size();
		HostCheck((uint64_t)shards[rank].size() * test.degree == replicated.size(),"per-rank KV bytes are total / degree");
		HostCheck(LmKvShardViewInitialize<Geometry>(&local,shards[rank].data(),table.data(),pages_per_sequence,test.rows,pages,&error,shard) == 0,"shard view");
		HostCheck(LmKvShardReplicaViewInitialize<Geometry>(&replica,view,shard) == 0,"replica view");
		LM_LAUNCH((LmKvShardStoreKernel<Geometry,1u>),(uint32_t)tokens,1u,0,0,local,values.data(),token_sequence.data(),token_position.data(),(uint32_t)tokens,width);
		for (index=0u; index<tokens; index++)
		{
			uint32_t position = token_position[index],sequence_row = token_sequence[index];
			const uint8_t *full = replicated.data() + (uint64_t)table[sequence_row * pages_per_sequence + position / HOST_PAGE] * Geometry::kPageBytes + (uint64_t)(position % HOST_PAGE) * Geometry::kSlotBytes;
			const uint8_t *part = shards[rank].data() + (uint64_t)table[sequence_row * pages_per_sequence + position / HOST_PAGE] * SparkKvShardPageBytes(shard,HOST_PAGE,Geometry::kSlotBytes) + (uint64_t)SparkKvShardSlotInPage(shard,HOST_PAGE,position) * Geometry::kSlotBytes;
			if ( SparkKvShardOwns(shard,position) == 0u )
				continue;
			owned_tokens++;
			if ( memcmp(full,part,Geometry::kSlotBytes) != 0 )
			{
				HostCheck(0,"an owned slot holds the replicated bytes");
				break;
			}
		}
		HostCheck(LmLatentShardPartialLaunch<Geometry,LmKvShardReplicaView,HOST_LATENT,ROPE>(replica,query.data(),query_stride,test.heads_per_rank,sequences.data(),contexts.data(),row_position.data(),selected.data(),HOST_SELECTED,HOST_DENSE_LIMIT,scale,oracle_send.data() + (uint64_t)rank * test.degree * rank_stride,rank_stride,test.rows,0) == cudaSuccess,"oracle partial launch");
		HostCheck(LmLatentShardPartialLaunch<Geometry,LmKvShardView,HOST_LATENT,ROPE>(local,query.data(),query_stride,test.heads_per_rank,sequences.data(),contexts.data(),row_position.data(),selected.data(),HOST_SELECTED,HOST_DENSE_LIMIT,scale,shard_send.data() + (uint64_t)rank * test.degree * rank_stride,rank_stride,test.rows,0) == cudaSuccess,"shard partial launch");
	}
	HostCheck(error.error_code == LM_FRAME_ERROR_NONE,"no KV access error");
	HostCheck(owned_tokens == tokens && shard_bytes == replicated.size(),"every token is stored on exactly one rank");
	HostCheck(memcmp(oracle_send.data(),shard_send.data(),oracle_send.size() * sizeof(float)) == 0,"sharded partials equal the replicated-storage oracle bit for bit");
	for (rank=0u; rank<test.degree; rank++)
	{
		uint32_t source;
		for (source=0u; source<test.degree; source++)
			memcpy(receive.data() + ((uint64_t)rank * test.degree + source) * rank_stride,shard_send.data() + ((uint64_t)source * test.degree + rank) * rank_stride,rank_stride * sizeof(float));
		HostCheck(LmLatentShardMergeLaunch<HOST_LATENT>(receive.data() + (uint64_t)rank * test.degree * rank_stride,rank_stride,test.degree,test.heads_per_rank,shard_out.data() + (uint64_t)rank * test.rows * test.heads_per_rank * HOST_LATENT,test.rows,0) == cudaSuccess,"merge after all-to-all");
		HostCheck(LmLatentShardMergeLaunch<HOST_LATENT>(shard_send.data() + (uint64_t)rank * rank_stride,(uint64_t)test.degree * rank_stride,test.degree,test.heads_per_rank,gather_out.data() + (uint64_t)rank * test.rows * test.heads_per_rank * HOST_LATENT,test.rows,0) == cudaSuccess,"merge after all-gather");
		for (source=0u; source<test.degree; source++)
			memcpy(receive.data() + ((uint64_t)rank * test.degree + source) * rank_stride,oracle_send.data() + ((uint64_t)source * test.degree + rank) * rank_stride,rank_stride * sizeof(float));
		HostCheck(LmLatentShardMergeLaunch<HOST_LATENT>(receive.data() + (uint64_t)rank * test.degree * rank_stride,rank_stride,test.degree,test.heads_per_rank,oracle_out.data() + (uint64_t)rank * test.rows * test.heads_per_rank * HOST_LATENT,test.rows,0) == cudaSuccess,"oracle merge");
	}
	HostCheck(memcmp(oracle_out.data(),shard_out.data(),oracle_out.size() * 2u) == 0,"merged output equals the oracle bit for bit");
	HostCheck(memcmp(gather_out.data(),shard_out.data(),oracle_out.size() * 2u) == 0,"all-gather and all-to-all layouts merge to the same bits");
	for (row=0u; row<test.rows; row++)
	{
		uint32_t head;
		std::vector<uint32_t> keys;
		if ( contexts[row] > HOST_DENSE_LIMIT )
		{
			for (index=0u; index<HOST_SELECTED; index++)
				if ( selected[(uint64_t)row * HOST_SELECTED + index] <= row_position[row] )
					keys.push_back(selected[(uint64_t)row * HOST_SELECTED + index]);
		}
		else
			for (index=0u; index<contexts[row]; index++)
				keys.push_back(index);
		for (head=0u; head<heads; head++)
		{
			const uint16_t *q = query.data() + (uint64_t)(head / test.heads_per_rank) * query_stride + ((uint64_t)row * test.heads_per_rank + head % test.heads_per_rank) * width;
			const uint16_t *out = shard_out.data() + (uint64_t)(head / test.heads_per_rank) * test.rows * test.heads_per_rank * HOST_LATENT + ((uint64_t)row * test.heads_per_rank + head % test.heads_per_rank) * HOST_LATENT;
			std::vector<double> scores(keys.size()),result(HOST_LATENT,0.0);
			double top = -1.0e300,total = 0.0;
			uint32_t key,element;
			for (key=0u; key<keys.size(); key++)
			{
				const uint16_t *k = values.data() + ((uint64_t)row * test.context + keys[key]) * width;
				double dot = 0.0;
				for (element=0u; element<width; element++)
					dot += HostFloat(q[element]) * HostFloat(k[element]);
				scores[key] = dot * scale;
				top = scores[key] > top ? scores[key] : top;
			}
			for (key=0u; key<keys.size(); key++)
			{
				const uint16_t *k = values.data() + ((uint64_t)row * test.context + keys[key]) * width;
				double weight = exp(scores[key] - top);
				total += weight;
				for (element=0u; element<HOST_LATENT; element++)
					result[element] += weight * HostFloat(k[element]);
			}
			for (element=0u; element<HOST_LATENT; element++)
			{
				double difference = fabs(result[element] / total - HostFloat(out[element]));
				worst = difference > worst ? difference : worst;
			}
		}
	}
	HostCheck(worst < 1.0e-2,"merged output matches the f64 attention reference");
	printf("%s %s: listed rows %u, per-rank KV %llu of %llu bytes, worst |out-f64| %.2e\n",host_failures == 0 ? "PASS" : "FAIL",label,listed,(unsigned long long)shards[0].size(),(unsigned long long)replicated.size(),worst);
}

static void HostRejects(void)
{
	SparkKvShard shard;
	uint32_t degree,grain,valid = 1u;
	for (degree=1u; degree<=16u; degree++)
		for (grain=1u; grain<=4u; grain++)
		{
			shard.degree = degree;
			shard.rank = degree - 1u;
			shard.grain = grain;
			if ( SparkKvShardValid(shard,HOST_PAGE) != (HOST_PAGE % (degree * grain) == 0u ? 1u : 0u) )
				valid = 0u;
		}
	shard.degree = 32u;
	shard.rank = 0u;
	shard.grain = 1u;
	HostCheck(valid != 0u && SparkKvShardValid(shard,HOST_PAGE) == 0u,"a shard that cannot split a page is refused");
	shard.degree = 16u;
	shard.rank = 16u;
	HostCheck(SparkKvShardValid(shard,HOST_PAGE) == 0u,"a rank outside the degree is refused");
	for (degree=1u; degree<=16u; degree*=2u)
		for (grain=1u; grain<=4u; grain*=2u)
		{
			uint32_t keys,rank,local,total;
			shard.degree = degree;
			shard.grain = grain;
			for (keys=0u; keys<300u; keys++)
			{
				total = 0u;
				for (rank=0u; rank<degree; rank++)
				{
					shard.rank = rank;
					for (local=0u; local<SparkKvShardLocalKeys(shard,keys); local++)
						if ( SparkKvShardOwner(shard,SparkKvShardLocalPosition(shard,local)) != rank || SparkKvShardLocalPosition(shard,local) >= keys )
							valid = 0u;
					total += SparkKvShardLocalKeys(shard,keys);
				}
				if ( total != keys )
					valid = 0u;
			}
		}
	HostCheck(valid != 0u,"local key enumeration covers every position once, on its owner");
}

template<class Geometry>
__global__ void HostShardProbeKernel(LmKvShardView view,uint32_t sequence,uint32_t position)
{
	(void)LmKvShardSlotRequired<Geometry>(view,sequence,position,0u,LM_KV_ACCESS_READ);
}

template<uint32_t ROPE>
static void HostForeign(void)
{
	typedef HostShardKv<HOST_LATENT + ROPE> Geometry;
	std::vector<uint8_t> pool(Geometry::kPageBytes);
	uint32_t table = 0u,sequence = 0u,context = 64u,position = 1u;
	std::vector<uint16_t> query(HOST_LATENT + ROPE,0u);
	std::vector<float> partials(LM_LATENT_SHARD_RECORD_FLOATS(HOST_LATENT) * 2u);
	uint32_t list[1] = {1u};
	LmKvAccessError error;
	LmKvShardView view;
	SparkKvShard shard = {2u,0u,1u};
	LmKvAccessErrorReset(&error);
	HostCheck(LmKvShardViewInitialize<Geometry>(&view,pool.data(),&table,1u,1u,1u,&error,shard) == 0,"foreign view");
	LM_LAUNCH((LmKvShardStoreKernel<Geometry,1u>),1u,1u,0,0,view,query.data(),&sequence,&position,1u,HOST_LATENT + ROPE);
	HostCheck(error.error_code == LM_FRAME_ERROR_NONE,"a store of a foreign position is skipped, not written");
	LM_LAUNCH((LmLatentShardPartialKernel<Geometry,LmKvShardView,HOST_LATENT,ROPE,1u>),dim3(1u,2u),LM_LATENT_SHARD_THREADS,0,0,view,query.data(),(uint64_t)(HOST_LATENT + ROPE),1u,&sequence,&context,&position,list,1u,0u,1.0f,partials.data(),(uint64_t)LM_LATENT_SHARD_RECORD_FLOATS(HOST_LATENT));
	HostCheck(error.error_code == LM_FRAME_ERROR_NONE,"a selected foreign position is left to its owner");
	LM_LAUNCH((HostShardProbeKernel<Geometry>),1u,1u,0,0,view,sequence,position);
	HostCheck(error.error_code == LM_FRAME_ERROR_SHARD_NOT_OWNED && error.position == position,"reading a foreign position reports SHARD_NOT_OWNED");
}

int main(int argc,char **argv)
{
	HostRejects();
	HostForeign<0u>();
	if ( argc > 1 && strcmp(argv[1],"--full") == 0 )
	{
		HostRunCase<0u>({1u,1024u,16u,4u,1u});
		HostRunCase<0u>({8u,1024u,16u,4u,1u});
		HostRunCase<0u>({1u,8192u,16u,4u,1u});
		HostRunCase<0u>({8u,8192u,16u,4u,1u});
		HostRunCase<64u>({4u,1024u,16u,2u,1u});
		HostRunCase<0u>({64u,1024u,16u,4u,1u});
	}
	else
	{
		HostRunCase<0u>({1u,300u,16u,1u,1u});
		HostRunCase<0u>({1u,2200u,16u,1u,1u});
		HostRunCase<64u>({1u,200u,16u,1u,1u});
		HostRunCase<0u>({2u,2100u,4u,1u,1u});
		HostRunCase<0u>({1u,500u,2u,2u,4u});
	}
	if ( host_failures != 0 )
	{
		printf("FAIL %d checks\n",host_failures);
		return 1;
	}
	printf("PASS latent KV shard: per-rank bytes are total / degree, sharded store and attention equal the replicated oracle bit for bit, merged output matches f64\n");
	return 0;
}
