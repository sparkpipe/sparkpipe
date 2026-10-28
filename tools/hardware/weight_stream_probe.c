#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define STREAM_MAX_RANKS 16u
#define STREAM_STAMP_BYTES 1048576ull
#define STREAM_RECV_DEPTH 512u
#define STREAM_COUNTERS 6u

typedef struct StreamEndpoint
{
	uint32_t qpn;
	uint32_t rkey;
	uint64_t address;
	uint8_t gid[16];
} StreamEndpoint;

typedef struct StreamOptions
{
	uint32_t rank;
	uint32_t ranks;
	uint64_t shard_bytes;
	uint32_t layers;
	uint32_t chunk_bytes;
	uint32_t depth;
	uint32_t pair;
	uint32_t gid_index;
	uint32_t pair_gid_index;
	uint32_t traffic_class;
	uint32_t port;
	uint32_t verify;
	const char *device;
	const char *pair_device;
	const char *coordinator;
} StreamOptions;

typedef struct StreamLink
{
	struct ibv_context *context;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_mr *mr;
	struct ibv_qp *qps[STREAM_MAX_RANKS];
	uint32_t outstanding[STREAM_MAX_RANKS];
	uint64_t counters[STREAM_COUNTERS];
	union ibv_gid gid;
} StreamLink;

typedef struct StreamRelay
{
	uint32_t source;
	uint32_t chunk;
} StreamRelay;

static const char *stream_counter_names[STREAM_COUNTERS] = { "out_of_sequence","packet_seq_err","roce_adp_retrans","local_ack_timeout_err","np_cnp_sent","rp_cnp_handled" };

static uint64_t StreamNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void StreamFail(const char *what)
{
	fprintf(stderr,"STREAM-FAIL %s errno=%d\n",what,errno);
	exit(2);
}

static void StreamSendAll(int fd,const void *data,size_t bytes)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while ( bytes != 0u )
	{
		ssize_t sent = send(fd,cursor,bytes,0);
		if ( sent <= 0 )
			StreamFail("tcp send");
		cursor += sent;
		bytes -= (size_t)sent;
	}
}

static void StreamReceiveAll(int fd,void *data,size_t bytes)
{
	uint8_t *cursor = (uint8_t *)data;
	while ( bytes != 0u )
	{
		ssize_t received = recv(fd,cursor,bytes,0);
		if ( received <= 0 )
			StreamFail("tcp receive");
		cursor += received;
		bytes -= (size_t)received;
	}
}

static void StreamExchange(const StreamOptions *options,int *clients,int *server_fd,StreamEndpoint *table,size_t row_bytes)
{
	struct sockaddr_in address;
	uint32_t rank,one = 1u;
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)options->port);
	if ( options->rank == 0u )
	{
		int listener = socket(AF_INET,SOCK_STREAM,0);
		setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
		address.sin_addr.s_addr = htonl(INADDR_ANY);
		if ( listener < 0 || bind(listener,(struct sockaddr *)&address,sizeof(address)) != 0 || listen(listener,STREAM_MAX_RANKS) != 0 )
			StreamFail("listen");
		for (rank=1u; rank<options->ranks; rank++)
		{
			uint32_t peer_rank;
			int fd = accept(listener,0,0);
			if ( fd < 0 )
				StreamFail("accept");
			setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
			StreamReceiveAll(fd,&peer_rank,sizeof(peer_rank));
			if ( peer_rank == 0u || peer_rank >= options->ranks || clients[peer_rank] >= 0 )
				StreamFail("duplicate rank");
			clients[peer_rank] = fd;
			StreamReceiveAll(fd,(uint8_t *)table + peer_rank * row_bytes,row_bytes);
		}
		close(listener);
		for (rank=1u; rank<options->ranks; rank++)
			StreamSendAll(clients[rank],table,options->ranks * row_bytes);
		return;
	}
	*server_fd = socket(AF_INET,SOCK_STREAM,0);
	if ( inet_pton(AF_INET,options->coordinator,&address.sin_addr) != 1 )
		StreamFail("coordinator address");
	for (rank=0u; connect(*server_fd,(struct sockaddr *)&address,sizeof(address)) != 0; rank++)
	{
		if ( rank > 600u )
			StreamFail("connect");
		close(*server_fd);
		*server_fd = socket(AF_INET,SOCK_STREAM,0);
		usleep(100000);
	}
	setsockopt(*server_fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
	StreamSendAll(*server_fd,&options->rank,sizeof(options->rank));
	StreamSendAll(*server_fd,(uint8_t *)table + options->rank * row_bytes,row_bytes);
	StreamReceiveAll(*server_fd,table,options->ranks * row_bytes);
}

static void StreamBarrier(const StreamOptions *options,const int *clients,int server_fd,uint64_t *value)
{
	uint64_t mine = *value,peer;
	uint32_t rank;
	if ( options->rank == 0u )
	{
		for (rank=1u; rank<options->ranks; rank++)
		{
			StreamReceiveAll(clients[rank],&peer,sizeof(peer));
			mine |= peer;
		}
		for (rank=1u; rank<options->ranks; rank++)
			StreamSendAll(clients[rank],&mine,sizeof(mine));
		*value = mine;
		return;
	}
	StreamSendAll(server_fd,&mine,sizeof(mine));
	StreamReceiveAll(server_fd,value,sizeof(*value));
}

static uint64_t StreamCounter(const char *device,const char *name)
{
	char path[256];
	unsigned long long value = 0ull;
	FILE *file;
	snprintf(path,sizeof(path),"/sys/class/infiniband/%s/ports/1/hw_counters/%s",device,name);
	file = fopen(path,"r");
	if ( file != 0 )
	{
		if ( fscanf(file,"%llu",&value) != 1 )
			value = 0ull;
		fclose(file);
	}
	return value;
}

static void StreamOpen(StreamLink *link,const char *device,uint8_t *region,uint64_t region_bytes,uint32_t gid_index)
{
	struct ibv_device **devices = ibv_get_device_list(0);
	int index;
	memset(link,0,sizeof(*link));
	for (index=0; devices != 0 && devices[index] != 0; index++)
		if ( strcmp(ibv_get_device_name(devices[index]),device) == 0 )
			link->context = ibv_open_device(devices[index]);
	if ( devices != 0 )
		ibv_free_device_list(devices);
	if ( link->context == 0 )
		StreamFail("open device");
	link->pd = ibv_alloc_pd(link->context);
	link->cq = ibv_create_cq(link->context,16384,0,0,0);
	if ( link->pd == 0 || link->cq == 0 )
		StreamFail("device resources");
	link->mr = ibv_reg_mr(link->pd,region,region_bytes,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	if ( link->mr == 0 || ibv_query_gid(link->context,1,(int)gid_index,&link->gid) != 0 )
		StreamFail("register");
	for (index=0; index<(int)STREAM_COUNTERS; index++)
		link->counters[index] = StreamCounter(device,stream_counter_names[index]);
}

static void StreamCreateQp(StreamLink *link,uint32_t peer,uint32_t depth,StreamEndpoint *mine,uint8_t *region)
{
	struct ibv_qp_init_attr init;
	memset(&init,0,sizeof(init));
	init.send_cq = link->cq;
	init.recv_cq = link->cq;
	init.qp_type = IBV_QPT_RC;
	init.cap.max_send_wr = depth;
	init.cap.max_recv_wr = STREAM_RECV_DEPTH;
	init.cap.max_send_sge = 1;
	init.cap.max_recv_sge = 1;
	link->qps[peer] = ibv_create_qp(link->pd,&init);
	if ( link->qps[peer] == 0 )
		StreamFail("create qp");
	mine->qpn = link->qps[peer]->qp_num;
	mine->rkey = link->mr->rkey;
	mine->address = (uint64_t)(uintptr_t)region;
	memcpy(mine->gid,link->gid.raw,sizeof(link->gid.raw));
}

static void StreamConnect(struct ibv_qp *qp,const StreamEndpoint *remote,uint32_t gid_index,uint32_t traffic_class)
{
	struct ibv_qp_attr attributes;
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_INIT;
	attributes.port_num = 1;
	attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0 )
		StreamFail("qp init");
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_RTR;
	attributes.path_mtu = IBV_MTU_4096;
	attributes.dest_qp_num = remote->qpn;
	attributes.max_dest_rd_atomic = 1;
	attributes.min_rnr_timer = 12;
	attributes.ah_attr.is_global = 1;
	attributes.ah_attr.port_num = 1;
	memcpy(attributes.ah_attr.grh.dgid.raw,remote->gid,sizeof(remote->gid));
	attributes.ah_attr.grh.sgid_index = (uint8_t)gid_index;
	attributes.ah_attr.grh.hop_limit = 1;
	attributes.ah_attr.grh.traffic_class = (uint8_t)traffic_class;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0 )
		StreamFail("qp rtr");
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_RTS;
	attributes.timeout = 14;
	attributes.retry_cnt = 7;
	attributes.rnr_retry = 7;
	attributes.max_rd_atomic = 1;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0 )
		StreamFail("qp rts");
}

static void StreamPostReceive(struct ibv_qp *qp,uint32_t count)
{
	struct ibv_recv_wr request,*bad;
	uint32_t index;
	memset(&request,0,sizeof(request));
	for (index=0u; index<count; index++)
		if ( ibv_post_recv(qp,&request,&bad) != 0 )
			StreamFail("post receive");
}

static uint32_t StreamImmediate(uint32_t layer,uint32_t source,uint32_t chunk)
{
	return ((layer & 0xffu) << 24) | ((source & 0xffu) << 16) | (chunk & 0xffffu);
}

static uint64_t StreamSlot(const StreamOptions *options,uint32_t layer,uint32_t source)
{
	return ((uint64_t)(layer & 1u) * options->ranks + source) * options->shard_bytes;
}

static void StreamWrite(StreamLink *link,uint32_t peer,const StreamEndpoint *remote,uint8_t *region,const StreamOptions *options,uint32_t layer,uint32_t source,uint32_t chunk)
{
	struct ibv_sge scatter;
	struct ibv_send_wr request,*bad;
	uint64_t offset = StreamSlot(options,layer,source) + (uint64_t)chunk * options->chunk_bytes;
	uint64_t length = options->shard_bytes - (uint64_t)chunk * options->chunk_bytes;
	memset(&scatter,0,sizeof(scatter));
	scatter.addr = (uint64_t)(uintptr_t)(region + offset);
	scatter.length = (uint32_t)(length < options->chunk_bytes ? length : options->chunk_bytes);
	scatter.lkey = link->mr->lkey;
	memset(&request,0,sizeof(request));
	request.wr_id = peer;
	request.sg_list = &scatter;
	request.num_sge = 1;
	request.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	request.send_flags = IBV_SEND_SIGNALED;
	request.imm_data = htonl(StreamImmediate(layer,source,chunk));
	request.wr.rdma.remote_addr = remote->address + offset;
	request.wr.rdma.rkey = remote->rkey;
	if ( ibv_post_send(link->qps[peer],&request,&bad) != 0 )
		StreamFail("post send");
	link->outstanding[peer]++;
}

static uint64_t StreamPattern(uint32_t source,uint64_t word)
{
	return (word + 1u) * 0x9E3779B97F4A7C15ull ^ (uint64_t)(source + 1u) * 0xC2B2AE3D27D4EB4Full;
}

static uint64_t StreamStamp(uint32_t layer,uint32_t source,uint64_t stamp)
{
	return ((uint64_t)layer << 40) | ((uint64_t)source << 32) | stamp | 0x8000000000000000ull;
}

static void StreamFillShard(const StreamOptions *options,uint8_t *region)
{
	uint64_t *words,count = options->shard_bytes / sizeof(uint64_t),word;
	uint32_t buffer;
	for (buffer=0u; buffer<2u; buffer++)
	{
		words = (uint64_t *)(region + StreamSlot(options,buffer,options->rank));
		for (word=0u; word<count; word++)
			words[word] = StreamPattern(options->rank,word);
	}
}

static void StreamStampShard(const StreamOptions *options,uint8_t *region,uint32_t layer)
{
	uint8_t *slot = region + StreamSlot(options,layer,options->rank);
	uint64_t stamp;
	for (stamp=0u; stamp * STREAM_STAMP_BYTES < options->shard_bytes; stamp++)
		*(uint64_t *)(slot + stamp * STREAM_STAMP_BYTES) = StreamStamp(layer,options->rank,stamp);
}

static uint64_t StreamCheck(const StreamOptions *options,const uint8_t *region,uint32_t layer,uint32_t full)
{
	uint64_t bad = 0u,stamp,word,count = options->shard_bytes / sizeof(uint64_t);
	uint32_t source;
	for (source=0u; source<options->ranks; source++)
	{
		const uint8_t *slot = region + StreamSlot(options,layer,source);
		if ( source == options->rank )
			continue;
		for (stamp=0u; stamp * STREAM_STAMP_BYTES < options->shard_bytes; stamp++)
			if ( *(const uint64_t *)(slot + stamp * STREAM_STAMP_BYTES) != StreamStamp(layer,source,stamp) )
				bad++;
		if ( full != 0u )
			for (word=0u; word<count; word++)
				if ( (word * sizeof(uint64_t)) % STREAM_STAMP_BYTES != 0u && ((const uint64_t *)slot)[word] != StreamPattern(source,word) )
					bad++;
	}
	return bad;
}

static int StreamCompare(const void *left,const void *right)
{
	uint64_t a = *(const uint64_t *)left,b = *(const uint64_t *)right;
	return a < b ? -1 : a > b;
}

static void StreamCompletion(const struct ibv_wc *done,uint32_t layer)
{
	fprintf(stderr,"STREAM-FAIL completion status=%d opcode=%d peer=%llu layer=%u\n",(int)done->status,(int)done->opcode,(unsigned long long)done->wr_id,layer);
	exit(3);
}

int main(int argc,char **argv)
{
	StreamOptions options = { 0u,16u,681574400ull,48u,4194304u,32u,0u,3u,3u,0u,47200u,1u,"rocep1s0f1","roceP2p1s0f0","10.10.100.10" };
	StreamEndpoint *table;
	StreamLink fabric,pair;
	StreamRelay *relays;
	uint64_t *samples,region_bytes,total_ns,bad_total = 0u,chunks;
	uint8_t *region;
	size_t row_bytes;
	int clients[STREAM_MAX_RANKS],server_fd = -1,index;
	uint32_t rank,layer,partner,fabric_peers = 0u,fabric_list[STREAM_MAX_RANKS],expected_fabric,expected_pair;
	for (index=1; index + 1 < argc; index+=2)
	{
		if ( strcmp(argv[index],"--rank") == 0 ) options.rank = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--ranks") == 0 ) options.ranks = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--shard-bytes") == 0 ) options.shard_bytes = strtoull(argv[index + 1],0,10);
		else if ( strcmp(argv[index],"--layers") == 0 ) options.layers = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--chunk-bytes") == 0 ) options.chunk_bytes = (uint32_t)strtoul(argv[index + 1],0,10);
		else if ( strcmp(argv[index],"--depth") == 0 ) options.depth = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--pair") == 0 ) options.pair = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--gid-index") == 0 ) options.gid_index = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--pair-gid-index") == 0 ) options.pair_gid_index = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--traffic-class") == 0 ) options.traffic_class = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--port") == 0 ) options.port = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--verify") == 0 ) options.verify = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--device") == 0 ) options.device = argv[index + 1];
		else if ( strcmp(argv[index],"--pair-device") == 0 ) options.pair_device = argv[index + 1];
		else if ( strcmp(argv[index],"--coordinator") == 0 ) options.coordinator = argv[index + 1];
		else
		{
			fprintf(stderr,"usage: %s --rank R [--ranks N] [--shard-bytes B] [--layers L] [--chunk-bytes C] [--depth D] [--pair 0|1 (relay the other parity through rank ^ 1 on --pair-device)] [--verify 0 stamps|1 stamps+last layer|2 every layer] [--device D] [--pair-device D] [--gid-index G] [--pair-gid-index G] [--traffic-class T] [--coordinator IP] [--port P]\n",argv[0]);
			return 2;
		}
	}
	chunks = options.chunk_bytes == 0u ? 0u : (options.shard_bytes + options.chunk_bytes - 1u) / options.chunk_bytes;
	if ( options.ranks < 2u || options.ranks > STREAM_MAX_RANKS || options.rank >= options.ranks || options.layers == 0u || options.layers > 255u || options.depth == 0u || options.traffic_class > 255u ||
	    options.shard_bytes == 0u || options.shard_bytes % sizeof(uint64_t) != 0u || options.chunk_bytes == 0u || options.chunk_bytes % 4096u != 0u || chunks > 65536u || options.verify > 2u ||
	    (options.pair != 0u && ((options.ranks & 1u) != 0u || options.ranks < 4u)) )
		return 2;
	partner = options.rank ^ 1u;
	region_bytes = 2u * options.ranks * options.shard_bytes;
	region = (uint8_t *)aligned_alloc(2097152u,(region_bytes + 2097151u) & ~(uint64_t)2097151u);
	if ( region == 0 )
		StreamFail("allocate");
	madvise(region,region_bytes,MADV_HUGEPAGE);
	memset(region,0,region_bytes);
	StreamFillShard(&options,region);
	row_bytes = 2u * STREAM_MAX_RANKS * sizeof(StreamEndpoint);
	table = (StreamEndpoint *)calloc(options.ranks,row_bytes);
	relays = (StreamRelay *)calloc((size_t)options.ranks * chunks,sizeof(StreamRelay));
	samples = (uint64_t *)calloc(options.layers,sizeof(uint64_t));
	if ( table == 0 || relays == 0 || samples == 0 )
		StreamFail("tables");
	StreamOpen(&fabric,options.device,region,region_bytes,options.gid_index);
	if ( options.pair != 0u )
		StreamOpen(&pair,options.pair_device,region,region_bytes,options.pair_gid_index);
	for (rank=0u; rank<options.ranks; rank++)
	{
		StreamEndpoint *row = (StreamEndpoint *)((uint8_t *)table + options.rank * row_bytes);
		clients[rank] = -1;
		if ( rank == options.rank )
			continue;
		if ( options.pair == 0u || (rank != partner && (rank & 1u) == (options.rank & 1u)) )
			StreamCreateQp(&fabric,rank,options.depth,&row[rank],region);
		if ( options.pair != 0u && rank == partner )
			StreamCreateQp(&pair,rank,options.depth,&row[STREAM_MAX_RANKS + rank],region);
	}
	StreamExchange(&options,clients,&server_fd,table,row_bytes);
	for (rank=0u; rank<options.ranks; rank++)
	{
		const StreamEndpoint *remote = (const StreamEndpoint *)((const uint8_t *)table + rank * row_bytes);
		if ( fabric.qps[rank] != 0 )
		{
			StreamConnect(fabric.qps[rank],&remote[options.rank],options.gid_index,options.traffic_class);
			StreamPostReceive(fabric.qps[rank],STREAM_RECV_DEPTH);
		}
		if ( options.pair != 0u && pair.qps[rank] != 0 )
		{
			StreamConnect(pair.qps[rank],&remote[STREAM_MAX_RANKS + options.rank],options.pair_gid_index,options.traffic_class);
			StreamPostReceive(pair.qps[rank],STREAM_RECV_DEPTH);
		}
	}
	for (rank=1u; rank<options.ranks; rank++)
		if ( fabric.qps[(options.rank + rank) % options.ranks] != 0 )
			fabric_list[fabric_peers++] = (options.rank + rank) % options.ranks;
	expected_fabric = fabric_peers * (uint32_t)chunks;
	expected_pair = options.pair != 0u ? (1u + fabric_peers) * (uint32_t)chunks : 0u;
	total_ns = StreamNowNs();
	for (layer=0u; layer<options.layers; layer++)
	{
		uint64_t start,flag = 0u;
		uint32_t next_chunk = 0u,next_peer = 0u,sent_fabric = 0u,own_pair = 0u,relay_head = 0u,relay_tail = 0u,relay_turn = 1u;
		uint32_t got_fabric = 0u,got_pair = 0u,done_fabric = 0u,done_pair = 0u;
		StreamStampShard(&options,region,layer);
		StreamBarrier(&options,clients,server_fd,&flag);
		start = StreamNowNs();
		while ( done_fabric < expected_fabric || done_pair < expected_pair || got_fabric < expected_fabric || got_pair < expected_pair )
		{
			struct ibv_wc done[64];
			int count,item;
			while ( sent_fabric < expected_fabric )
			{
				uint32_t peer = fabric_list[next_peer];
				const StreamEndpoint *remote = (const StreamEndpoint *)((const uint8_t *)table + peer * row_bytes);
				if ( fabric.outstanding[peer] >= options.depth )
					break;
				StreamWrite(&fabric,peer,&remote[options.rank],region,&options,layer,options.rank,next_chunk);
				sent_fabric++;
				next_peer = (next_peer + 1u) % fabric_peers;
				if ( next_peer == 0u )
					next_chunk++;
			}
			if ( options.pair != 0u )
			{
				const StreamEndpoint *remote = (const StreamEndpoint *)((const uint8_t *)table + partner * row_bytes);
				while ( pair.outstanding[partner] < options.depth && (own_pair < chunks || relay_head != relay_tail) )
				{
					if ( relay_head != relay_tail && (own_pair >= chunks || relay_turn != 0u) )
					{
						StreamWrite(&pair,partner,&remote[STREAM_MAX_RANKS + options.rank],region,&options,layer,relays[relay_head].source,relays[relay_head].chunk);
						relay_head++;
					}
					else
						StreamWrite(&pair,partner,&remote[STREAM_MAX_RANKS + options.rank],region,&options,layer,options.rank,own_pair++);
					relay_turn ^= 1u;
				}
			}
			if ( StreamNowNs() - start > 300000000000ull )
			{
				fprintf(stderr,"STREAM-FAIL stall layer=%u sent=%u done=%u/%u got=%u/%u pair_done=%u/%u pair_got=%u/%u\n",layer,sent_fabric,done_fabric,expected_fabric,got_fabric,expected_fabric,done_pair,expected_pair,got_pair,expected_pair);
				return 5;
			}
			count = ibv_poll_cq(fabric.cq,64,done);
			for (item=0; item<count; item++)
			{
				if ( done[item].status != IBV_WC_SUCCESS )
					StreamCompletion(&done[item],layer);
				if ( done[item].opcode == IBV_WC_RECV_RDMA_WITH_IMM )
				{
					uint32_t immediate = ntohl(done[item].imm_data);
					if ( (immediate >> 24) != (layer & 0xffu) )
						StreamCompletion(&done[item],layer);
					got_fabric++;
					StreamPostReceive(fabric.qps[(immediate >> 16) & 0xffu],1u);
					if ( options.pair != 0u )
					{
						relays[relay_tail].source = (immediate >> 16) & 0xffu;
						relays[relay_tail].chunk = immediate & 0xffffu;
						relay_tail++;
					}
				}
				else
				{
					fabric.outstanding[done[item].wr_id]--;
					done_fabric++;
				}
			}
			if ( options.pair != 0u )
			{
				count = ibv_poll_cq(pair.cq,64,done);
				for (item=0; item<count; item++)
				{
					if ( done[item].status != IBV_WC_SUCCESS )
						StreamCompletion(&done[item],layer);
					if ( done[item].opcode == IBV_WC_RECV_RDMA_WITH_IMM )
					{
						if ( (ntohl(done[item].imm_data) >> 24) != (layer & 0xffu) )
							StreamCompletion(&done[item],layer);
						got_pair++;
						StreamPostReceive(pair.qps[partner],1u);
					}
					else
					{
						pair.outstanding[partner]--;
						done_pair++;
					}
				}
			}
		}
		samples[layer] = StreamNowNs() - start;
		if ( options.verify == 2u || (options.verify == 1u && layer + 1u == options.layers) )
			bad_total += StreamCheck(&options,region,layer,1u);
		else
			bad_total += StreamCheck(&options,region,layer,0u);
	}
	{
		uint64_t flag = bad_total != 0u ? 1u : 0u;
		StreamBarrier(&options,clients,server_fd,&flag);
		total_ns = StreamNowNs() - total_ns;
		qsort(samples,options.layers,sizeof(uint64_t),StreamCompare);
		printf("STREAM rank=%u ranks=%u pair=%u shard_bytes=%llu chunk_bytes=%u depth=%u layers=%u total_s=%.2f p50_ms=%.1f p90_ms=%.1f max_ms=%.1f ingress_gbps_p50=%.1f bad_words=%llu any_rank_bad=%llu",
		    options.rank,options.ranks,options.pair,(unsigned long long)options.shard_bytes,options.chunk_bytes,options.depth,options.layers,total_ns / 1e9,
		    samples[options.layers / 2u] / 1e6,samples[(options.layers * 9u) / 10u] / 1e6,samples[options.layers - 1u] / 1e6,
		    (double)(options.ranks - 1u) * (double)options.shard_bytes * 8.0 / (double)samples[options.layers / 2u],(unsigned long long)bad_total,(unsigned long long)flag);
		for (index=0; index<(int)STREAM_COUNTERS; index++)
			printf(" %s=%llu",stream_counter_names[index],(unsigned long long)(StreamCounter(options.device,stream_counter_names[index]) - fabric.counters[index]));
		if ( options.pair != 0u )
			for (index=0; index<(int)STREAM_COUNTERS; index++)
				printf(" pair_%s=%llu",stream_counter_names[index],(unsigned long long)(StreamCounter(options.pair_device,stream_counter_names[index]) - pair.counters[index]));
		printf("\n");
		return flag != 0u ? 4 : 0;
	}
}
