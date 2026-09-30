#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define PROBE_MAX_RANKS 16u

typedef struct ProbeEndpoint
{
	uint32_t qpn;
	uint32_t rkey;
	uint64_t address;
	uint8_t gid[16];
} ProbeEndpoint;

typedef struct ProbeOptions
{
	uint32_t rank;
	uint32_t ranks;
	uint32_t bytes;
	uint32_t phases;
	uint32_t window;
	uint32_t gid_index;
	uint32_t port;
	uint32_t free_running;
	const char *device;
	const char *coordinator;
} ProbeOptions;

static uint64_t ProbeNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void ProbeFail(const char *what)
{
	fprintf(stderr,"PROBE-FAIL %s errno=%d\n",what,errno);
	exit(2);
}

static void ProbeSendAll(int fd,const void *data,size_t bytes)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while ( bytes != 0u )
	{
		ssize_t sent = send(fd,cursor,bytes,0);
		if ( sent <= 0 )
			ProbeFail("tcp send");
		cursor += sent;
		bytes -= (size_t)sent;
	}
}

static void ProbeReceiveAll(int fd,void *data,size_t bytes)
{
	uint8_t *cursor = (uint8_t *)data;
	while ( bytes != 0u )
	{
		ssize_t received = recv(fd,cursor,bytes,0);
		if ( received <= 0 )
			ProbeFail("tcp receive");
		cursor += received;
		bytes -= (size_t)received;
	}
}

static void ProbeExchange(const ProbeOptions *options,int *clients,int *server_fd,ProbeEndpoint *table)
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
		if ( listener < 0 || bind(listener,(struct sockaddr *)&address,sizeof(address)) != 0 || listen(listener,PROBE_MAX_RANKS) != 0 )
			ProbeFail("listen");
		for (rank=1u; rank<options->ranks; rank++)
		{
			uint32_t peer_rank;
			int fd = accept(listener,0,0);
			if ( fd < 0 )
				ProbeFail("accept");
			setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
			ProbeReceiveAll(fd,&peer_rank,sizeof(peer_rank));
			if ( peer_rank == 0u || peer_rank >= options->ranks || clients[peer_rank] >= 0 )
				ProbeFail("duplicate rank");
			clients[peer_rank] = fd;
			ProbeReceiveAll(fd,&table[peer_rank * options->ranks],options->ranks * sizeof(ProbeEndpoint));
		}
		close(listener);
		for (rank=1u; rank<options->ranks; rank++)
			ProbeSendAll(clients[rank],table,(size_t)options->ranks * options->ranks * sizeof(ProbeEndpoint));
		return;
	}
	*server_fd = socket(AF_INET,SOCK_STREAM,0);
	if ( inet_pton(AF_INET,options->coordinator,&address.sin_addr) != 1 )
		ProbeFail("coordinator address");
	for (rank=0u; connect(*server_fd,(struct sockaddr *)&address,sizeof(address)) != 0; rank++)
	{
		if ( rank > 600u )
			ProbeFail("connect");
		close(*server_fd);
		*server_fd = socket(AF_INET,SOCK_STREAM,0);
		usleep(100000);
	}
	setsockopt(*server_fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
	ProbeSendAll(*server_fd,&options->rank,sizeof(options->rank));
	ProbeSendAll(*server_fd,&table[options->rank * options->ranks],options->ranks * sizeof(ProbeEndpoint));
	ProbeReceiveAll(*server_fd,table,(size_t)options->ranks * options->ranks * sizeof(ProbeEndpoint));
}

static void ProbeBarrier(const ProbeOptions *options,const int *clients,int server_fd)
{
	uint8_t token = 1u;
	uint32_t rank;
	if ( options->rank == 0u )
	{
		for (rank=1u; rank<options->ranks; rank++)
			ProbeReceiveAll(clients[rank],&token,1u);
		for (rank=1u; rank<options->ranks; rank++)
			ProbeSendAll(clients[rank],&token,1u);
		return;
	}
	ProbeSendAll(server_fd,&token,1u);
	ProbeReceiveAll(server_fd,&token,1u);
}

static void ProbeConnect(struct ibv_qp *qp,const ProbeEndpoint *remote,uint32_t gid_index)
{
	struct ibv_qp_attr attributes;
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_INIT;
	attributes.port_num = 1;
	attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0 )
		ProbeFail("qp init");
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_RTR;
	attributes.path_mtu = IBV_MTU_4096;
	attributes.dest_qp_num = remote->qpn;
	attributes.max_dest_rd_atomic = 1;
	attributes.min_rnr_timer = 12;
	attributes.ah_attr.is_global = 1;
	attributes.ah_attr.port_num = 1;
	memcpy(attributes.ah_attr.grh.dgid.raw,remote->gid,16);
	attributes.ah_attr.grh.sgid_index = (uint8_t)gid_index;
	attributes.ah_attr.grh.hop_limit = 1;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0 )
		ProbeFail("qp rtr");
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_RTS;
	attributes.timeout = 14;
	attributes.retry_cnt = 7;
	attributes.rnr_retry = 7;
	attributes.max_rd_atomic = 1;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0 )
		ProbeFail("qp rts");
}

static void ProbePost(struct ibv_qp *qp,struct ibv_mr *mr,uint64_t local,uint32_t length,const ProbeEndpoint *remote,uint64_t remote_offset,uint64_t work_id,int signaled)
{
	struct ibv_sge scatter;
	struct ibv_send_wr request,*bad;
	memset(&scatter,0,sizeof(scatter));
	scatter.addr = local;
	scatter.length = length;
	scatter.lkey = mr->lkey;
	memset(&request,0,sizeof(request));
	request.wr_id = work_id;
	request.sg_list = &scatter;
	request.num_sge = 1;
	request.opcode = IBV_WR_RDMA_WRITE;
	request.send_flags = signaled != 0 ? IBV_SEND_SIGNALED : 0;
	request.wr.rdma.remote_addr = remote->address + remote_offset;
	request.wr.rdma.rkey = remote->rkey;
	if ( ibv_post_send(qp,&request,&bad) != 0 )
		ProbeFail("post send");
}

static int ProbeCompare(const void *left,const void *right)
{
	uint64_t a = *(const uint64_t *)left,b = *(const uint64_t *)right;
	return a < b ? -1 : a > b;
}

static uint64_t ProbeCounter(const char *device,const char *name)
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

int main(int argc,char **argv)
{
	static const char *counters[] = { "out_of_sequence","packet_seq_err","roce_adp_retrans","local_ack_timeout_err" };
	ProbeOptions options = { 0u,16u,262144u,200u,0u,3u,47100u,0u,"rocep1s0f1","10.10.100.10" };
	ProbeEndpoint table[PROBE_MAX_RANKS * PROBE_MAX_RANKS];
	struct ibv_qp *qps[PROBE_MAX_RANKS];
	struct ibv_device **devices;
	struct ibv_context *context = 0;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_mr *mr;
	union ibv_gid gid;
	uint64_t *samples,before[4],region_bytes,flag_base,total_ns;
	volatile uint64_t *flags;
	uint8_t *region;
	int clients[PROBE_MAX_RANKS],server_fd = -1,index;
	uint32_t rank,phase;
	for (index=1; index + 1 < argc; index+=2)
	{
		if ( strcmp(argv[index],"--rank") == 0 ) options.rank = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--ranks") == 0 ) options.ranks = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--bytes") == 0 ) options.bytes = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--phases") == 0 ) options.phases = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--window") == 0 ) options.window = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--gid-index") == 0 ) options.gid_index = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--port") == 0 ) options.port = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--free-running") == 0 ) options.free_running = (uint32_t)atoi(argv[index + 1]);
		else if ( strcmp(argv[index],"--device") == 0 ) options.device = argv[index + 1];
		else if ( strcmp(argv[index],"--coordinator") == 0 ) options.coordinator = argv[index + 1];
		else
		{
			fprintf(stderr,"usage: %s --rank R [--ranks N] [--bytes B] [--phases P] [--window W (0 = post every peer at once)] [--free-running 0|1] [--device D] [--gid-index G] [--coordinator IP] [--port P]\n",argv[0]);
			return 2;
		}
	}
	if ( options.ranks < 2u || options.ranks > PROBE_MAX_RANKS || options.rank >= options.ranks || options.bytes == 0u || options.phases == 0u )
		return 2;
	devices = ibv_get_device_list(0);
	for (index=0; devices != 0 && devices[index] != 0; index++)
		if ( strcmp(ibv_get_device_name(devices[index]),options.device) == 0 )
			context = ibv_open_device(devices[index]);
	if ( context == 0 )
		ProbeFail("open device");
	pd = ibv_alloc_pd(context);
	cq = ibv_create_cq(context,4096,0,0,0);
	flag_base = (uint64_t)options.ranks * options.bytes;
	region_bytes = flag_base + (uint64_t)options.ranks * sizeof(uint64_t) + options.bytes;
	region = (uint8_t *)aligned_alloc(4096,(region_bytes + 4095u) & ~(uint64_t)4095u);
	if ( pd == 0 || cq == 0 || region == 0 )
		ProbeFail("resources");
	memset(region,0,region_bytes);
	flags = (volatile uint64_t *)(region + flag_base);
	mr = ibv_reg_mr(pd,region,region_bytes,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	if ( mr == 0 || ibv_query_gid(context,1,(int)options.gid_index,&gid) != 0 )
		ProbeFail("register");
	memset(table,0,sizeof(table));
	for (rank=0u; rank<options.ranks; rank++)
	{
		struct ibv_qp_init_attr init;
		clients[rank] = -1;
		qps[rank] = 0;
		if ( rank == options.rank )
			continue;
		memset(&init,0,sizeof(init));
		init.send_cq = cq;
		init.recv_cq = cq;
		init.qp_type = IBV_QPT_RC;
		init.cap.max_send_wr = 64;
		init.cap.max_recv_wr = 1;
		init.cap.max_send_sge = 1;
		init.cap.max_recv_sge = 1;
		qps[rank] = ibv_create_qp(pd,&init);
		if ( qps[rank] == 0 )
			ProbeFail("create qp");
		table[options.rank * options.ranks + rank].qpn = qps[rank]->qp_num;
		table[options.rank * options.ranks + rank].rkey = mr->rkey;
		table[options.rank * options.ranks + rank].address = (uint64_t)(uintptr_t)region;
		memcpy(table[options.rank * options.ranks + rank].gid,gid.raw,16);
	}
	ProbeExchange(&options,clients,&server_fd,table);
	for (rank=0u; rank<options.ranks; rank++)
		if ( rank != options.rank )
			ProbeConnect(qps[rank],&table[rank * options.ranks + options.rank],options.gid_index);
	samples = (uint64_t *)calloc(options.phases,sizeof(uint64_t));
	for (index=0; index<4; index++)
		before[index] = ProbeCounter(options.device,counters[index]);
	ProbeBarrier(&options,clients,server_fd);
	total_ns = ProbeNowNs();
	for (phase=1u; phase<=options.phases; phase++)
	{
		uint64_t start,*sequence = (uint64_t *)(region + flag_base + (uint64_t)options.ranks * sizeof(uint64_t));
		uint32_t next = 1u,outstanding = 0u,completed = 0u,peers = options.ranks - 1u,arrived;
		uint32_t window = options.window == 0u || options.window > peers ? peers : options.window;
		*sequence = phase;
		if ( options.free_running == 0u )
			ProbeBarrier(&options,clients,server_fd);
		start = ProbeNowNs();
		while ( completed < peers )
		{
			struct ibv_wc done[16];
			int count;
			while ( outstanding < window && next <= peers )
			{
				uint32_t peer = (options.rank + next) % options.ranks;
				const ProbeEndpoint *remote = &table[peer * options.ranks + options.rank];
				ProbePost(qps[peer],mr,(uint64_t)(uintptr_t)region + flag_base + (uint64_t)options.ranks * sizeof(uint64_t) + sizeof(uint64_t),options.bytes - sizeof(uint64_t),remote,(uint64_t)options.rank * options.bytes,0u,0);
				ProbePost(qps[peer],mr,(uint64_t)(uintptr_t)sequence,sizeof(uint64_t),remote,flag_base + (uint64_t)options.rank * sizeof(uint64_t),peer,1);
				outstanding++;
				next++;
			}
			count = ibv_poll_cq(cq,16,done);
			for (index=0; index<count; index++)
			{
				if ( done[index].status != IBV_WC_SUCCESS )
				{
					fprintf(stderr,"PROBE-FAIL completion status=%d peer=%llu phase=%u\n",(int)done[index].status,(unsigned long long)done[index].wr_id,phase);
					return 3;
				}
				outstanding--;
				completed++;
			}
		}
		do
		{
			arrived = 0u;
			for (rank=0u; rank<options.ranks; rank++)
				if ( rank != options.rank && flags[rank] >= phase )
					arrived++;
		}
		while ( arrived < peers );
		samples[phase - 1u] = ProbeNowNs() - start;
	}
	ProbeBarrier(&options,clients,server_fd);
	total_ns = ProbeNowNs() - total_ns;
	qsort(samples,options.phases,sizeof(uint64_t),ProbeCompare);
	printf("PROBE rank=%u ranks=%u bytes=%u window=%u phases=%u free_running=%u total_ms=%.1f p50_us=%.1f p90_us=%.1f p99_us=%.1f max_us=%.1f gbps_p50=%.2f",
	    options.rank,options.ranks,options.bytes,options.window,options.phases,options.free_running,total_ns / 1e6,
	    samples[options.phases / 2u] / 1e3,samples[(options.phases * 9u) / 10u] / 1e3,samples[(options.phases * 99u) / 100u] / 1e3,samples[options.phases - 1u] / 1e3,
	    (double)(options.ranks - 1u) * options.bytes * 8.0 / (double)samples[options.phases / 2u]);
	for (index=0; index<4; index++)
		printf(" %s=%llu",counters[index],(unsigned long long)(ProbeCounter(options.device,counters[index]) - before[index]));
	printf("\n");
	return 0;
}
