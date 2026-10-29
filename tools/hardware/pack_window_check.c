#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_pack_window.h"

#define CHECK_MAX_LAYERS 128u
#define CHECK_ROW_BYTES (2u * SPARK_WEIGHT_STREAM_MAX_RANKS * sizeof(SparkWeightStreamEndpoint))

typedef struct CheckOptions
{
	uint32_t rank;
	uint32_t ranks;
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t verify;
	uint32_t chunk_bytes;
	uint32_t depth;
	uint32_t gid_index;
	uint32_t pair_gid_index;
	uint32_t port;
	uint64_t dense_capacity;
	uint64_t expert_capacity;
	const char *pack;
	const char *partner;
	const char *ranges;
	const char *device;
	const char *pair_device;
	const char *coordinator;
} CheckOptions;

typedef struct CheckCoordinator
{
	const CheckOptions *options;
	int clients[SPARK_WEIGHT_STREAM_MAX_RANKS];
	int server;
} CheckCoordinator;

typedef struct CheckPack
{
	int fd;
} CheckPack;

static uint64_t CheckNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void CheckFail(const char *what)
{
	fprintf(stderr,"PACKWINDOW-FAIL %s errno=%d\n",what,errno);
	exit(3);
}

static void CheckSend(int fd, const void *data, size_t bytes)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while ( bytes != 0u )
	{
		const ssize_t sent = send(fd,cursor,bytes,0);
		if ( sent <= 0 )
			CheckFail("send");
		cursor += sent;
		bytes -= (size_t)sent;
	}
}

static void CheckReceive(int fd, void *data, size_t bytes)
{
	uint8_t *cursor = (uint8_t *)data;
	while ( bytes != 0u )
	{
		const ssize_t got = recv(fd,cursor,bytes,0);
		if ( got <= 0 )
			CheckFail("receive");
		cursor += got;
		bytes -= (size_t)got;
	}
}

static void CheckConnect(CheckCoordinator *coordinator)
{
	const CheckOptions *options = coordinator->options;
	struct sockaddr_in address;
	int one = 1;
	uint32_t index;
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)options->port);
	coordinator->server = -1;
	for (index=0u; index<SPARK_WEIGHT_STREAM_MAX_RANKS; index++)
		coordinator->clients[index] = -1;
	if ( options->rank == 0u )
	{
		coordinator->server = socket(AF_INET,SOCK_STREAM,0);
		setsockopt(coordinator->server,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
		address.sin_addr.s_addr = htonl(INADDR_ANY);
		if ( bind(coordinator->server,(struct sockaddr *)&address,sizeof(address)) != 0 || listen(coordinator->server,32) != 0 )
			CheckFail("listen");
		for (index=1u; index<options->ranks; index++)
		{
			uint32_t peer;
			const int fd = accept(coordinator->server,0,0);
			if ( fd < 0 )
				CheckFail("accept");
			setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
			CheckReceive(fd,&peer,sizeof(peer));
			if ( peer == 0u || peer >= options->ranks || coordinator->clients[peer] >= 0 )
				CheckFail("peer rank");
			coordinator->clients[peer] = fd;
		}
		return;
	}
	inet_pton(AF_INET,options->coordinator,&address.sin_addr);
	for (index=0u; index<600u; index++)
	{
		coordinator->server = socket(AF_INET,SOCK_STREAM,0);
		if ( connect(coordinator->server,(struct sockaddr *)&address,sizeof(address)) == 0 )
			break;
		close(coordinator->server);
		coordinator->server = -1;
		usleep(100000);
	}
	if ( coordinator->server < 0 )
		CheckFail("connect");
	setsockopt(coordinator->server,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
	CheckSend(coordinator->server,&options->rank,sizeof(options->rank));
}

static void CheckAllGather(CheckCoordinator *coordinator, const void *mine, void *all, uint64_t row)
{
	const CheckOptions *options = coordinator->options;
	uint32_t index;
	if ( options->rank == 0u )
	{
		memcpy(all,mine,row);
		for (index=1u; index<options->ranks; index++)
			CheckReceive(coordinator->clients[index],(uint8_t *)all + (uint64_t)index * row,row);
		for (index=1u; index<options->ranks; index++)
			CheckSend(coordinator->clients[index],all,(size_t)(row * options->ranks));
		return;
	}
	CheckSend(coordinator->server,mine,row);
	CheckReceive(coordinator->server,all,(size_t)(row * options->ranks));
}

static void CheckBarrier(CheckCoordinator *coordinator)
{
	uint64_t mine = 0u,all[SPARK_WEIGHT_STREAM_MAX_RANKS];
	CheckAllGather(coordinator,&mine,all,sizeof(mine));
}

static SparkStatus CheckGather(void *context, const void *mine, void *all, uint64_t bytes_per_rank)
{
	CheckAllGather((CheckCoordinator *)context,mine,all,bytes_per_rank);
	return SPARK_STATUS_OK;
}

static SparkStatus CheckRead(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination)
{
	const CheckPack *pack = (const CheckPack *)context;
	uint64_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t got = pread(pack->fd,destination + done,(size_t)(bytes - done < (1ull << 30) ? bytes - done : (1ull << 30)),(off_t)(pack_offset + done));
		if ( got <= 0 )
			return SPARK_STATUS_IO_ERROR;
		done += (uint64_t)got;
	}
	posix_fadvise(pack->fd,(off_t)pack_offset,(off_t)bytes,POSIX_FADV_DONTNEED);
	return SPARK_STATUS_OK;
}

static uint64_t CheckHash(const uint8_t *data, uint64_t bytes)
{
	const uint64_t *words = (const uint64_t *)data;
	uint64_t hash = 0x9e3779b97f4a7c15ull ^ bytes,index;
	for (index=0u; index<bytes / 8u; index++)
		hash = (hash ^ words[index]) * 0x100000001b3ull + (hash >> 29u);
	for (index=bytes & ~(uint64_t)7u; index<bytes; index++)
		hash = (hash ^ data[index]) * 0x100000001b3ull;
	return hash;
}

static uint32_t CheckLoadRanges(const char *path, SparkPackWindowLayer *layers, uint32_t capacity)
{
	FILE *file = fopen(path,"r");
	char line[1024];
	uint32_t count = 0u;
	if ( file == 0 )
		CheckFail("ranges file");
	while ( fgets(line,sizeof(line),file) != 0 && count < capacity )
	{
		SparkPackWindowLayer *layer = &layers[count];
		unsigned long long values[16];
		char *cursor = line,*bar = strchr(line,'|');
		uint32_t found = 0u,index;
		uint64_t base;
		if ( bar == 0 )
			continue;
		*bar = '\0';
		while ( found < 16u )
		{
			char *end;
			values[found] = strtoull(cursor,&end,10);
			if ( end == cursor )
				break;
			cursor = end;
			found++;
		}
		if ( found < 4u || ((found - 2u) % 2u) != 0u || (found - 2u) / 2u > SPARK_PACK_WINDOW_MAX_DENSE_RANGES || values[0] != count )
			CheckFail("ranges line");
		memset(layer,0,sizeof(*layer));
		base = values[1];
		layer->dense_count = (found - 2u) / 2u;
		for (index=0u; index<layer->dense_count; index++)
		{
			layer->dense[index].offset = base + values[2u + 2u * index];
			layer->dense[index].bytes = values[3u + 2u * index];
		}
		if ( sscanf(bar + 1,"%llu %llu",&values[0],&values[1]) != 2 )
			CheckFail("ranges experts");
		layer->experts.offset = values[1] != 0u ? base + values[0] : 0u;
		layer->experts.bytes = values[1];
		count++;
	}
	fclose(file);
	return count;
}

int main(int argc, char **argv)
{
	CheckOptions options = { 0u,16u,1u,4u,1u,4194304u,32u,3u,3u,47340u,104857600ull,982646784ull,0,0,0,"rocep1s0f1","roceP2p1s0f0","10.10.100.10" };
	static SparkPackWindowLayer layers[CHECK_MAX_LAYERS];
	CheckCoordinator coordinator;
	CheckPack own,partner;
	SparkWeightStreamConfig stream_config;
	SparkPackWindowConfig window_config;
	SparkWeightStream *stream = 0;
	SparkPackWindow window;
	uint64_t region_bytes,bad = 0u,checked = 0u,layer_ns[CHECK_MAX_LAYERS],step_max_ns = 0u,dense_max_ns = 0u,total_ns,any_bad;
	uint64_t gathered[SPARK_WEIGHT_STREAM_MAX_RANKS];
	uint32_t layer_total,index,tag = 1u;
	uint8_t *region;
	int argi;
	for (argi=1; argi + 1 < argc; argi+=2)
	{
		if ( strcmp(argv[argi],"--rank") == 0 ) options.rank = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--ranks") == 0 ) options.ranks = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--first-layer") == 0 ) options.first_layer = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--layers") == 0 ) options.layer_count = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--verify") == 0 ) options.verify = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--chunk-bytes") == 0 ) options.chunk_bytes = (uint32_t)strtoul(argv[argi + 1],0,10);
		else if ( strcmp(argv[argi],"--depth") == 0 ) options.depth = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--gid-index") == 0 ) options.gid_index = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--pair-gid-index") == 0 ) options.pair_gid_index = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--port") == 0 ) options.port = (uint32_t)atoi(argv[argi + 1]);
		else if ( strcmp(argv[argi],"--dense-capacity") == 0 ) options.dense_capacity = strtoull(argv[argi + 1],0,10);
		else if ( strcmp(argv[argi],"--expert-capacity") == 0 ) options.expert_capacity = strtoull(argv[argi + 1],0,10);
		else if ( strcmp(argv[argi],"--pack") == 0 ) options.pack = argv[argi + 1];
		else if ( strcmp(argv[argi],"--partner-pack") == 0 ) options.partner = argv[argi + 1];
		else if ( strcmp(argv[argi],"--ranges") == 0 ) options.ranges = argv[argi + 1];
		else if ( strcmp(argv[argi],"--device") == 0 ) options.device = argv[argi + 1];
		else if ( strcmp(argv[argi],"--pair-device") == 0 ) options.pair_device = argv[argi + 1];
		else if ( strcmp(argv[argi],"--coordinator") == 0 ) options.coordinator = argv[argi + 1];
		else
		{
			fprintf(stderr,"usage: %s --rank R --pack OWN --partner-pack COPY --ranges FILE [--ranks N] [--first-layer L] [--layers C] [--verify 0 timing|1 hash every block] [--chunk-bytes B] [--depth D] [--device D] [--pair-device D] [--gid-index G] [--pair-gid-index G] [--dense-capacity B] [--expert-capacity B] [--coordinator IP] [--port P]\n",argv[0]);
			return 2;
		}
	}
	if ( options.pack == 0 || options.partner == 0 || options.ranges == 0 || options.ranks < 2u || options.ranks > SPARK_WEIGHT_STREAM_MAX_RANKS ||
		options.rank >= options.ranks || options.layer_count == 0u || options.verify > 1u )
		return 2;
	layer_total = CheckLoadRanges(options.ranges,layers,CHECK_MAX_LAYERS);
	if ( options.first_layer + options.layer_count > layer_total )
		CheckFail("layer range");
	for (index=options.first_layer; index<options.first_layer + options.layer_count; index++)
		if ( layers[index].experts.bytes == 0u || layers[index].experts.bytes > options.expert_capacity ||
			SparkPackWindowDenseBytes(&layers[index]) > options.dense_capacity )
			CheckFail("layer shape for the window");
	own.fd = open(options.pack,O_RDONLY);
	partner.fd = open(options.partner,O_RDONLY);
	if ( own.fd < 0 || partner.fd < 0 )
		CheckFail("open packs");
	region_bytes = SparkPackWindowRegionBytes(options.ranks,options.dense_capacity,options.expert_capacity);
	region = (uint8_t *)aligned_alloc(2097152u,(size_t)(region_bytes + 2097151u) & ~(size_t)2097151u);
	if ( region == 0 )
		CheckFail("allocate");
	memset(region,0,(size_t)region_bytes);
	coordinator.options = &options;
	CheckConnect(&coordinator);
	memset(&stream_config,0,sizeof(stream_config));
	stream_config.rank = options.rank;
	stream_config.ranks = options.ranks;
	stream_config.chunk_bytes = options.chunk_bytes;
	stream_config.depth = options.depth;
	stream_config.fabric_gid_index = options.gid_index;
	stream_config.pair_gid_index = options.pair_gid_index;
	stream_config.traffic_class = 106u;
	stream_config.fabric_device = options.device;
	stream_config.pair_device = options.pair_device;
	stream_config.region = region;
	stream_config.region_bytes = region_bytes;
	stream_config.gather = CheckGather;
	stream_config.gather_context = &coordinator;
	if ( SparkWeightStreamOpen(&stream_config,&stream) != SPARK_STATUS_OK )
		CheckFail("stream open");
	memset(&window_config,0,sizeof(window_config));
	window_config.stream = stream;
	window_config.region = region;
	window_config.region_bytes = region_bytes;
	window_config.rank = options.rank;
	window_config.ranks = options.ranks;
	window_config.dense_capacity = options.dense_capacity;
	window_config.expert_capacity = options.expert_capacity;
	window_config.read_own = CheckRead;
	window_config.own_context = &own;
	window_config.read_partner = CheckRead;
	window_config.partner_context = &partner;
	if ( SparkPackWindowInitialize(&window,&window_config) != SPARK_STATUS_OK )
		CheckFail("window");
	CheckBarrier(&coordinator);
	total_ns = CheckNowNs();
	for (index=0u; index<options.layer_count; index++)
	{
		const SparkPackWindowLayer *layer = &layers[options.first_layer + index];
		uint64_t dense_hash[SPARK_WEIGHT_STREAM_MAX_RANKS],expert_hash[SPARK_WEIGHT_STREAM_MAX_RANKS],start,dense_ns;
		uint8_t *bases[SPARK_WEIGHT_STREAM_MAX_RANKS];
		uint32_t step,rank;
		if ( options.verify != 0u )
		{
			uint8_t *scratch = (uint8_t *)malloc((size_t)layer->experts.bytes);
			uint64_t mine[2],all[2u * SPARK_WEIGHT_STREAM_MAX_RANKS],placed = 0u;
			uint32_t range;
			if ( scratch == 0 )
				CheckFail("scratch");
			for (range=0u; range<layer->dense_count; range++)
			{
				if ( CheckRead(&own,layer->dense[range].offset,layer->dense[range].bytes,scratch + placed) != SPARK_STATUS_OK )
					CheckFail("hash read");
				placed += layer->dense[range].bytes;
			}
			mine[0] = CheckHash(scratch,placed);
			if ( CheckRead(&own,layer->experts.offset,layer->experts.bytes,scratch) != SPARK_STATUS_OK )
				CheckFail("hash read experts");
			mine[1] = CheckHash(scratch,layer->experts.bytes);
			free(scratch);
			CheckAllGather(&coordinator,mine,all,sizeof(mine));
			for (rank=0u; rank<options.ranks; rank++)
			{
				dense_hash[rank] = all[2u * rank];
				expert_hash[rank] = all[2u * rank + 1u];
			}
		}
		CheckBarrier(&coordinator);
		start = CheckNowNs();
		if ( SparkPackWindowDense(&window,layer,index & 1u,tag++,bases) != SPARK_STATUS_OK )
			CheckFail("dense broadcast");
		dense_ns = CheckNowNs() - start;
		dense_max_ns = dense_ns > dense_max_ns ? dense_ns : dense_max_ns;
		if ( options.verify != 0u )
			for (rank=0u; rank<options.ranks; rank++)
			{
				bad += CheckHash(bases[rank],SparkPackWindowDenseBytes(layer)) != dense_hash[rank];
				checked++;
			}
		for (step=0u; step<options.ranks; step++)
		{
			const uint8_t *block = 0;
			uint32_t source = UINT32_MAX;
			const uint64_t step_start = CheckNowNs();
			uint64_t step_ns;
			if ( SparkPackWindowExpertStep(&window,layer,index * options.ranks + step,tag++,&source,&block) != SPARK_STATUS_OK )
				CheckFail("expert step");
			step_ns = CheckNowNs() - step_start;
			if ( step > 0u )
				step_max_ns = step_ns > step_max_ns ? step_ns : step_max_ns;
			if ( options.verify != 0u && source != options.rank )
			{
				bad += block == 0 || CheckHash(block,layer->experts.bytes) != expert_hash[source];
				checked++;
			}
			CheckBarrier(&coordinator);
		}
		layer_ns[index] = CheckNowNs() - start;
	}
	total_ns = CheckNowNs() - total_ns;
	CheckAllGather(&coordinator,&bad,gathered,sizeof(bad));
	any_bad = 0u;
	for (index=0u; index<options.ranks; index++)
		any_bad += gathered[index];
	{
		uint64_t sorted[CHECK_MAX_LAYERS],swap;
		uint32_t a,b;
		memcpy(sorted,layer_ns,(size_t)options.layer_count * sizeof(uint64_t));
		for (a=0u; a<options.layer_count; a++)
			for (b=a + 1u; b<options.layer_count; b++)
				if ( sorted[b] < sorted[a] ) { swap = sorted[a]; sorted[a] = sorted[b]; sorted[b] = swap; }
		printf("PACKWINDOW rank=%u ranks=%u layers=%u first=%u verify=%u layer_p50_ms=%.1f layer_max_ms=%.1f dense_max_ms=%.1f step_max_ms=%.1f total_s=%.2f checked=%llu bad=%llu any_rank_bad=%llu\n",
			options.rank,options.ranks,options.layer_count,options.first_layer,options.verify,sorted[options.layer_count / 2u] / 1e6,sorted[options.layer_count - 1u] / 1e6,
			dense_max_ns / 1e6,step_max_ns / 1e6,total_ns / 1e9,(unsigned long long)checked,(unsigned long long)bad,(unsigned long long)any_bad);
	}
	SparkWeightStreamClose(stream);
	return any_bad != 0u ? 4 : 0;
}
