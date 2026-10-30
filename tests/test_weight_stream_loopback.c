#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_weight_stream.h"

#define LOOP_RANKS 4u
#define LOOP_CHUNK (1024u * 1024u)
#define LOOP_DENSE ((uint64_t)9u * LOOP_CHUNK + 4096u)
#define LOOP_BLOCK ((uint64_t)6u * LOOP_CHUNK + 8192u)
#define LOOP_BROADCASTS 4u
#define LOOP_ROUNDS 3u
#define LOOP_SLOW_RANK (LOOP_RANKS - 1u)
#define LOOP_SLOW_NS 150000000L

typedef struct LoopShared
{
	pthread_barrier_t barrier;
	uint8_t rows[LOOP_RANKS][2u * SPARK_WEIGHT_STREAM_MAX_RANKS * sizeof(SparkWeightStreamEndpoint)];
	const char *device;
	uint32_t gid_index;
} LoopShared;

typedef struct LoopRank
{
	LoopShared *shared;
	uint32_t rank;
	uint64_t bad;
	uint64_t held_bad;
	int status;
} LoopRank;

static uint64_t LoopPattern(uint32_t seed, uint32_t source, uint64_t word)
{
	uint64_t x = ((uint64_t)seed << 48) ^ ((uint64_t)source << 40) ^ (word * 0x9e3779b97f4a7c15ull);
	x ^= x >> 31u;
	return x * 0xbf58476d1ce4e5b9ull;
}

static void LoopFill(uint8_t *data, uint64_t bytes, uint32_t seed, uint32_t source)
{
	uint64_t *words = (uint64_t *)data,index;
	for (index=0u; index<bytes / 8u; index++)
		words[index] = LoopPattern(seed,source,index);
}

static uint64_t LoopCheck(const uint8_t *data, uint64_t bytes, uint64_t first_word, uint32_t seed, uint32_t source)
{
	const uint64_t *words = (const uint64_t *)data;
	uint64_t index,bad = 0u;
	for (index=0u; index<bytes / 8u; index++)
		bad += words[index] != LoopPattern(seed,source,first_word + index);
	return bad;
}

static void LoopHold(void)
{
	struct timespec pause = {0, LOOP_SLOW_NS};
	nanosleep(&pause,0);
}

static SparkStatus LoopGather(void *context, const void *mine, void *all, uint64_t bytes_per_rank)
{
	LoopRank *self = (LoopRank *)context;
	uint32_t rank;
	if ( bytes_per_rank > sizeof(self->shared->rows[0]) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memcpy(self->shared->rows[self->rank],mine,bytes_per_rank);
	pthread_barrier_wait(&self->shared->barrier);
	for (rank=0u; rank<LOOP_RANKS; rank++)
		memcpy((uint8_t *)all + (uint64_t)rank * bytes_per_rank,self->shared->rows[rank],bytes_per_rank);
	pthread_barrier_wait(&self->shared->barrier);
	return SPARK_STATUS_OK;
}

static void *LoopRun(void *argument)
{
	LoopRank *self = (LoopRank *)argument;
	const uint64_t dense_area = (uint64_t)LOOP_RANKS * LOOP_DENSE;
	const uint64_t own = 2u * dense_area,partner = own + LOOP_BLOCK,landing = partner + LOOP_BLOCK,region_bytes = landing + 2u * LOOP_BLOCK;
	const uint64_t source_offset = region_bytes;
	uint8_t *region = (uint8_t *)aligned_alloc(4096u,(size_t)(region_bytes + LOOP_DENSE + 4095u) & ~(size_t)4095u);
	SparkWeightStreamConfig config;
	SparkWeightStream *stream = 0;
	uint32_t call,step,source;
	self->status = 1;
	if ( region == 0 )
		return 0;
	memset(region,0,(size_t)(region_bytes + LOOP_DENSE));
	memset(&config,0,sizeof(config));
	config.rank = self->rank;
	config.ranks = LOOP_RANKS;
	config.chunk_bytes = LOOP_CHUNK;
	config.depth = 16u;
	config.fabric_gid_index = self->shared->gid_index;
	config.pair_gid_index = self->shared->gid_index;
	config.traffic_class = 106u;
	config.fabric_device = self->shared->device;
	config.pair_device = self->shared->device;
	config.region = region;
	config.region_bytes = region_bytes + LOOP_DENSE;
	config.gather = LoopGather;
	config.gather_context = self;
	if ( SparkWeightStreamOpen(&config,&stream) != SPARK_STATUS_OK )
		return 0;
	for (call=0u; call<LOOP_BROADCASTS; call++)
	{
		const uint64_t area = (uint64_t)(call & 1u) * dense_area;
		LoopFill(region + source_offset,LOOP_DENSE,100u + call,self->rank);
		if ( SparkWeightStreamBroadcast(stream,source_offset,LOOP_DENSE,area,1u + call) != SPARK_STATUS_OK )
		{
			SparkWeightStreamClose(stream);
			return 0;
		}
		for (source=0u; source<LOOP_RANKS; source++)
			self->bad += LoopCheck(region + area + (uint64_t)source * LOOP_DENSE,LOOP_DENSE,0u,100u + call,source);
		if ( (call & 1u) != 0u )
			pthread_barrier_wait(&self->shared->barrier);
	}
	LoopFill(region + own,LOOP_BLOCK,7u,self->rank);
	LoopFill(region + partner,LOOP_BLOCK,7u,self->rank ^ 1u);
	for (step=0u; step<LOOP_ROUNDS * LOOP_RANKS; step++)
	{
		SparkWeightStreamBlock block;
		if ( SparkWeightStreamRotate(stream,step,own,partner,LOOP_BLOCK,landing + (uint64_t)(step & 1u) * LOOP_BLOCK,40u + step,&block) != SPARK_STATUS_OK )
		{
			SparkWeightStreamClose(stream);
			return 0;
		}
		if ( block.source != SparkWeightStreamRotationSource(self->rank,LOOP_RANKS,step) )
			self->bad++;
		self->bad += LoopCheck(region + block.half_offset[0],LOOP_BLOCK / 2u,0u,7u,block.source);
		self->bad += LoopCheck(region + block.half_offset[1],LOOP_BLOCK / 2u,LOOP_BLOCK / 16u,7u,block.source);
		pthread_barrier_wait(&self->shared->barrier);
	}
	for (call=0u; call<LOOP_BROADCASTS; call++)
	{
		LoopFill(region + source_offset,LOOP_DENSE,200u + call,self->rank);
		if ( SparkWeightStreamBroadcast(stream,source_offset,LOOP_DENSE,0u,60u + call) != SPARK_STATUS_OK )
		{
			SparkWeightStreamClose(stream);
			return 0;
		}
		for (source=0u; source<LOOP_RANKS; source++)
			self->bad += LoopCheck(region + (uint64_t)source * LOOP_DENSE,LOOP_DENSE,0u,200u + call,source);
		if ( self->rank == LOOP_SLOW_RANK )
		{
			LoopHold();
			for (source=0u; source<LOOP_RANKS; source++)
				self->held_bad += LoopCheck(region + (uint64_t)source * LOOP_DENSE,LOOP_DENSE,0u,200u + call,source);
		}
	}
	for (step=0u; step<LOOP_ROUNDS * LOOP_RANKS; step++)
	{
		SparkWeightStreamBlock block;
		if ( SparkWeightStreamRotate(stream,step,own,partner,LOOP_BLOCK,landing + (uint64_t)(step & 1u) * LOOP_BLOCK,70u + step,&block) != SPARK_STATUS_OK )
		{
			SparkWeightStreamClose(stream);
			return 0;
		}
		self->bad += LoopCheck(region + block.half_offset[0],LOOP_BLOCK / 2u,0u,7u,block.source);
		self->bad += LoopCheck(region + block.half_offset[1],LOOP_BLOCK / 2u,LOOP_BLOCK / 16u,7u,block.source);
		if ( self->rank == LOOP_SLOW_RANK )
		{
			LoopHold();
			self->held_bad += LoopCheck(region + block.half_offset[0],LOOP_BLOCK / 2u,0u,7u,block.source);
			self->held_bad += LoopCheck(region + block.half_offset[1],LOOP_BLOCK / 2u,LOOP_BLOCK / 16u,7u,block.source);
		}
	}
	SparkWeightStreamClose(stream);
	free(region);
	self->status = 0;
	return 0;
}

int main(int argc, char **argv)
{
	static LoopShared shared;
	LoopRank ranks[LOOP_RANKS];
	pthread_t threads[LOOP_RANKS];
	uint64_t bad = 0u,held_bad = 0u;
	uint32_t rank;
	int failed = 0;
	if ( argc != 3 )
	{
		fprintf(stderr,"usage: %s RDMA_DEVICE GID_INDEX\n",argv[0]);
		return 2;
	}
	shared.device = argv[1];
	shared.gid_index = (uint32_t)atoi(argv[2]);
	pthread_barrier_init(&shared.barrier,0,LOOP_RANKS);
	for (rank=0u; rank<LOOP_RANKS; rank++)
	{
		ranks[rank].shared = &shared;
		ranks[rank].rank = rank;
		ranks[rank].bad = 0u;
		ranks[rank].held_bad = 0u;
		ranks[rank].status = 1;
		pthread_create(&threads[rank],0,LoopRun,&ranks[rank]);
	}
	for (rank=0u; rank<LOOP_RANKS; rank++)
	{
		pthread_join(threads[rank],0);
		failed |= ranks[rank].status;
		bad += ranks[rank].bad;
		held_bad += ranks[rank].held_bad;
	}
	if ( failed != 0 || bad != 0u || held_bad != 0u )
	{
		printf("FAIL weight stream loopback: %s, %llu bad words, %llu words of a slow rank's landing overwritten before it entered the next call\n",
			failed != 0 ? "a rank failed" : "every rank finished",(unsigned long long)bad,(unsigned long long)held_bad);
		return 1;
	}
	printf("PASS weight stream loopback: %u ranks, %u back-to-back broadcasts of %llu bytes (pair relay, partial last chunk), %u rotation steps of %llu-byte blocks in halves, every byte checked; "
		"then one reused landing with no barriers while rank %u holds each landing %ld ms: nothing written before it entered the next call\n",
		LOOP_RANKS,LOOP_BROADCASTS,(unsigned long long)LOOP_DENSE,LOOP_ROUNDS * LOOP_RANKS,(unsigned long long)LOOP_BLOCK,LOOP_SLOW_RANK,LOOP_SLOW_NS / 1000000L);
	return 0;
}
