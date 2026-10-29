#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_pack_window.h"

#define PW_RANKS 4u
#define PW_LAYERS 3u
#define PW_CHUNK (1024u * 1024u)
#define PW_DENSE_CAPACITY ((uint64_t)6u * PW_CHUNK)
#define PW_EXPERT_CAPACITY ((uint64_t)8u * PW_CHUNK)

typedef struct PwShared
{
	pthread_barrier_t barrier;
	uint8_t rows[PW_RANKS][2u * SPARK_WEIGHT_STREAM_MAX_RANKS * sizeof(SparkWeightStreamEndpoint)];
	const char *device;
	uint32_t gid_index;
} PwShared;

typedef struct PwRank
{
	PwShared *shared;
	uint32_t rank;
	uint32_t partner;
	uint64_t bad;
	uint64_t checked;
	int status;
} PwRank;

static uint64_t PwWord(uint32_t rank, uint64_t word)
{
	uint64_t x = ((uint64_t)(rank + 1u) << 52) ^ (word * 0x9e3779b97f4a7c15ull);
	x ^= x >> 29u;
	return x * 0xbf58476d1ce4e5b9ull;
}

static SparkStatus PwFill(uint32_t rank, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	uint64_t index;
	if ( (pack_offset % 8u) != 0u || (bytes % 8u) != 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (index=0u; index<bytes / 8u; index++)
		((uint64_t *)destination)[index] = PwWord(rank,pack_offset / 8u + index);
	return SPARK_STATUS_OK;
}

static SparkStatus PwReadOwn(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	return PwFill(((PwRank *)context)->rank,pack_offset,bytes,destination);
}

static SparkStatus PwReadPartner(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	return PwFill(((PwRank *)context)->partner,pack_offset,bytes,destination);
}

static uint64_t PwCheck(const uint8_t *data, uint32_t rank, uint64_t pack_offset, uint64_t bytes)
{
	uint64_t index,bad = 0u;
	for (index=0u; index<bytes / 8u; index++)
		bad += ((const uint64_t *)data)[index] != PwWord(rank,pack_offset / 8u + index);
	return bad;
}

static SparkStatus PwGather(void *context, const void *mine, void *all, uint64_t bytes_per_rank)
{
	PwRank *self = (PwRank *)context;
	uint32_t rank;
	if ( bytes_per_rank > sizeof(self->shared->rows[0]) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memcpy(self->shared->rows[self->rank],mine,bytes_per_rank);
	pthread_barrier_wait(&self->shared->barrier);
	for (rank=0u; rank<PW_RANKS; rank++)
		memcpy((uint8_t *)all + (uint64_t)rank * bytes_per_rank,self->shared->rows[rank],bytes_per_rank);
	pthread_barrier_wait(&self->shared->barrier);
	return SPARK_STATUS_OK;
}

static void PwLayer(uint32_t layer, SparkPackWindowLayer *out)
{
	const uint64_t base = (uint64_t)layer * 64u * PW_CHUNK;
	memset(out,0,sizeof(*out));
	out->dense_count = 2u;
	out->dense[0].offset = base;
	out->dense[0].bytes = (uint64_t)3u * PW_CHUNK + 4096u * (layer + 1u);
	out->experts.offset = base + 4u * PW_CHUNK;
	out->experts.bytes = (uint64_t)6u * PW_CHUNK + 16384u;
	out->dense[1].offset = out->experts.offset + out->experts.bytes;
	out->dense[1].bytes = (uint64_t)PW_CHUNK + 8192u;
}

static void *PwRun(void *argument)
{
	PwRank *self = (PwRank *)argument;
	const uint64_t region_bytes = SparkPackWindowRegionBytes(PW_RANKS,PW_DENSE_CAPACITY,PW_EXPERT_CAPACITY);
	uint8_t *region = (uint8_t *)aligned_alloc(4096u,(size_t)(region_bytes + 4095u) & ~(size_t)4095u);
	SparkWeightStreamConfig stream_config;
	SparkPackWindowConfig window_config;
	SparkWeightStream *stream = 0;
	SparkPackWindow window;
	uint32_t layer,step,rank,tag = 1u;
	self->status = 1;
	if ( region == 0 )
		return 0;
	memset(region,0,(size_t)region_bytes);
	memset(&stream_config,0,sizeof(stream_config));
	stream_config.rank = self->rank;
	stream_config.ranks = PW_RANKS;
	stream_config.chunk_bytes = PW_CHUNK;
	stream_config.depth = 16u;
	stream_config.fabric_gid_index = self->shared->gid_index;
	stream_config.pair_gid_index = self->shared->gid_index;
	stream_config.traffic_class = 106u;
	stream_config.fabric_device = self->shared->device;
	stream_config.pair_device = self->shared->device;
	stream_config.region = region;
	stream_config.region_bytes = region_bytes;
	stream_config.gather = PwGather;
	stream_config.gather_context = self;
	if ( SparkWeightStreamOpen(&stream_config,&stream) != SPARK_STATUS_OK )
		return 0;
	memset(&window_config,0,sizeof(window_config));
	window_config.stream = stream;
	window_config.region = region;
	window_config.region_bytes = region_bytes;
	window_config.rank = self->rank;
	window_config.ranks = PW_RANKS;
	window_config.dense_capacity = PW_DENSE_CAPACITY;
	window_config.expert_capacity = PW_EXPERT_CAPACITY;
	window_config.read_own = PwReadOwn;
	window_config.own_context = self;
	window_config.read_partner = PwReadPartner;
	window_config.partner_context = self;
	if ( SparkPackWindowInitialize(&window,&window_config) != SPARK_STATUS_OK )
	{
		SparkWeightStreamClose(stream);
		return 0;
	}
	for (layer=0u; layer<PW_LAYERS; layer++)
	{
		SparkPackWindowLayer ranges;
		uint8_t *bases[PW_RANKS];
		PwLayer(layer,&ranges);
		if ( SparkPackWindowDense(&window,&ranges,layer & 1u,tag++,bases) != SPARK_STATUS_OK )
			break;
		for (rank=0u; rank<PW_RANKS; rank++)
		{
			self->bad += PwCheck(bases[rank],rank,ranges.dense[0].offset,ranges.dense[0].bytes);
			self->bad += PwCheck(bases[rank] + ranges.dense[0].bytes,rank,ranges.dense[1].offset,ranges.dense[1].bytes);
			self->checked++;
		}
		for (step=0u; step<PW_RANKS; step++)
		{
			const uint8_t *block = 0;
			uint32_t source = UINT32_MAX;
			if ( SparkPackWindowExpertStep(&window,&ranges,layer * PW_RANKS + step,tag++,&source,&block) != SPARK_STATUS_OK )
			{
				SparkWeightStreamClose(stream);
				return 0;
			}
			if ( source != SparkWeightStreamRotationSource(self->rank,PW_RANKS,layer * PW_RANKS + step) )
				self->bad++;
			if ( source != self->rank )
			{
				self->bad += block == 0 ? 1u : PwCheck(block,source,ranges.experts.offset,ranges.experts.bytes);
				self->checked++;
			}
			pthread_barrier_wait(&self->shared->barrier);
		}
	}
	SparkWeightStreamClose(stream);
	free(region);
	self->status = layer == PW_LAYERS ? 0 : 1;
	return 0;
}

int main(int argc, char **argv)
{
	static PwShared shared;
	PwRank ranks[PW_RANKS];
	pthread_t threads[PW_RANKS];
	uint64_t bad = 0u,checked = 0u;
	uint32_t rank;
	int failed = 0;
	if ( argc != 3 )
	{
		fprintf(stderr,"usage: %s RDMA_DEVICE GID_INDEX\n",argv[0]);
		return 2;
	}
	shared.device = argv[1];
	shared.gid_index = (uint32_t)atoi(argv[2]);
	pthread_barrier_init(&shared.barrier,0,PW_RANKS);
	for (rank=0u; rank<PW_RANKS; rank++)
	{
		ranks[rank].shared = &shared;
		ranks[rank].rank = rank;
		ranks[rank].partner = rank ^ 1u;
		ranks[rank].bad = 0u;
		ranks[rank].checked = 0u;
		ranks[rank].status = 1;
		pthread_create(&threads[rank],0,PwRun,&ranks[rank]);
	}
	for (rank=0u; rank<PW_RANKS; rank++)
	{
		pthread_join(threads[rank],0);
		failed |= ranks[rank].status;
		bad += ranks[rank].bad;
		checked += ranks[rank].checked;
	}
	if ( failed != 0 || bad != 0u || checked != (uint64_t)PW_RANKS * PW_LAYERS * (PW_RANKS + PW_RANKS - 1u) )
	{
		printf("FAIL pack window loopback: %s, %llu bad words, %llu of %llu blocks checked\n",failed != 0 ? "a rank failed" : "every rank finished",
			(unsigned long long)bad,(unsigned long long)checked,(unsigned long long)PW_RANKS * PW_LAYERS * (PW_RANKS + PW_RANKS - 1u));
		return 1;
	}
	printf("PASS pack window loopback: %u ranks x %u layers, every rank's two dense ranges and every peer's expert block (own half from the pack, partner half from the copy) byte-equal to the source packs, %llu blocks\n",
		PW_RANKS,PW_LAYERS,(unsigned long long)checked);
	return 0;
}
