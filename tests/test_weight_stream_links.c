#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#include "../ring/transport/weight_stream.c"

#define LINKS_RANKS 4u
#define LINKS_FABRIC "rocep1s0f1"
#define LINKS_PAIR "rocep1s0f0"

static uint32_t links_checks;
static uint32_t links_failures;
static uint32_t links_remote[LINKS_RANKS];
static uint8_t links_region[1u << 20];

#define CHECK(cond, name) do { \
		links_checks++; \
		if ( !(cond) ) { \
			links_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

static void LinksEndpoint(SparkWeightStreamEndpoint *endpoint, uint32_t qpn, uint8_t third, uint8_t fourth)
{
	memset(endpoint,0,sizeof(*endpoint));
	endpoint->qpn = qpn;
	endpoint->rkey = 0x51u;
	endpoint->address = 0x10000u;
	endpoint->gid[10] = 0xffu;
	endpoint->gid[11] = 0xffu;
	endpoint->gid[12] = 10u;
	endpoint->gid[13] = 10u;
	endpoint->gid[14] = third;
	endpoint->gid[15] = fourth;
}

static SparkStatus LinksGather(void *context, const void *mine, void *all, uint64_t bytes_per_rank)
{
	const uint32_t rank = *(const uint32_t *)context;
	SparkWeightStreamEndpoint (*table)[WEIGHT_STREAM_ROW] = (SparkWeightStreamEndpoint (*)[WEIGHT_STREAM_ROW])all;
	uint32_t peer;
	if ( bytes_per_rank != sizeof(table[0]) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(all,0,(size_t)bytes_per_rank * LINKS_RANKS);
	memcpy(table[rank],mine,(size_t)bytes_per_rank);
	for (peer=0u; peer<LINKS_RANKS; peer++)
	{
		if ( peer == rank )
			continue;
		if ( (links_remote[peer] & 1u) != 0u )
			LinksEndpoint(&table[peer][rank],0x7000u + peer,100u,(uint8_t)(10u + peer));
		if ( (links_remote[peer] & 2u) != 0u )
			LinksEndpoint(&table[peer][SPARK_WEIGHT_STREAM_MAX_RANKS + rank],0x8000u + peer,201u,(uint8_t)peer);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus LinksOpen(const char *pair, uint32_t remote, SparkWeightStream **stream, const char *log_path)
{
	static uint32_t rank = 0u;
	SparkWeightStreamConfig config;
	SparkStatus status;
	int saved,capture;
	uint32_t peer;
	for (peer=0u; peer<LINKS_RANKS; peer++)
		links_remote[peer] = remote;
	memset(&config,0,sizeof(config));
	config.rank = rank;
	config.ranks = LINKS_RANKS;
	config.chunk_bytes = 4096u;
	config.depth = 8u;
	config.fabric_gid_index = 3u;
	config.pair_gid_index = 3u;
	config.traffic_class = 106u;
	config.fabric_device = LINKS_FABRIC;
	config.pair_device = pair;
	config.region = links_region;
	config.region_bytes = sizeof(links_region);
	config.gather = LinksGather;
	config.gather_context = &rank;
	fflush(stderr);
	saved = dup(2);
	capture = open(log_path,O_WRONLY | O_CREAT | O_TRUNC,0644);
	if ( saved >= 0 && capture >= 0 )
		(void)dup2(capture,2);
	status = SparkWeightStreamOpen(&config,stream);
	fflush(stderr);
	if ( saved >= 0 )
	{
		(void)dup2(saved,2);
		(void)close(saved);
	}
	if ( capture >= 0 )
		(void)close(capture);
	return status;
}

static int LinksLogged(const char *path, const char *token)
{
	char text[16384];
	ssize_t got;
	int fd = open(path,O_RDONLY);
	if ( fd < 0 )
		return 0;
	got = read(fd,text,sizeof(text) - 1u);
	(void)close(fd);
	if ( got < 0 )
		return 0;
	text[got] = '\0';
	return strstr(text,token) != 0;
}

static uint32_t LinksAimed(const SparkWeightStream *stream, uint32_t peer, uint32_t link)
{
	const struct ibv_qp *qp = stream->links[link].qps[peer];
	return qp != 0 && spark_stub_ibv_qp_state(qp->qp_num) == IBV_QPS_RTS &&
		spark_stub_ibv_qp_remote_qpn(qp->qp_num) == (int)((link == WEIGHT_STREAM_PAIR ? 0x8000u : 0x7000u) + peer) ? 1u : 0u;
}

int main(void)
{
	char log_path[] = "/tmp/weight-stream-links-XXXXXX";
	SparkWeightStream *stream = 0;
	int fd = mkstemp(log_path);
	if ( fd < 0 )
		return 2;
	(void)close(fd);

	CHECK(LinksOpen(LINKS_PAIR,3u,&stream,log_path) == SPARK_STATUS_OK && stream != 0,"two links: the stream opens");
	CHECK(stream != 0 && stream->links_up == 3u && stream->link_of[1] == WEIGHT_STREAM_PAIR && stream->link_of[2] == WEIGHT_STREAM_FABRIC &&
		stream->link_of[3] == WEIGHT_STREAM_PAIR,"two links: a peer of the other parity takes the pair link, one of the same parity the fabric");
	CHECK(stream != 0 && LinksAimed(stream,1u,WEIGHT_STREAM_PAIR) != 0u && LinksAimed(stream,2u,WEIGHT_STREAM_FABRIC) != 0u &&
		spark_stub_ibv_qp_state(stream->links[WEIGHT_STREAM_FABRIC].qps[1]->qp_num) == IBV_QPS_RESET,
		"two links: only the chosen link's queue pair is connected, at the peer's endpoint on that link");
	CHECK(LinksLogged(log_path,"weight_stream: rank 0 of 4 on fabric+pair, degraded peers 0x0000"),"two links: the layout is logged");
	SparkWeightStreamClose(stream);
	stream = 0;

	spark_stub_ibv_set_port_state(LINKS_PAIR,IBV_PORT_DOWN);
	CHECK(LinksOpen(LINKS_PAIR,3u,&stream,log_path) == SPARK_STATUS_OK && stream != 0 && stream->links_up == 1u &&
		stream->link_of[1] == WEIGHT_STREAM_FABRIC && stream->link_of[3] == WEIGHT_STREAM_FABRIC && LinksAimed(stream,1u,WEIGHT_STREAM_FABRIC) != 0u,
		"pair port down: every peer takes the fabric link");
	CHECK(LinksLogged(log_path,"weight_stream: " LINKS_PAIR " gid 3: port 1 is not active; the link is not used") &&
		LinksLogged(log_path,"weight_stream: rank 0 (fabric) reaches rank 1 (fabric+pair) on its fabric link only") &&
		LinksLogged(log_path,"degraded peers 0x000e"),"pair port down: the unused link and each degraded peer are logged");
	SparkWeightStreamClose(stream);
	stream = 0;

	CHECK(LinksOpen(LINKS_PAIR,1u,&stream,log_path) == SPARK_STATUS_OK && stream != 0 && stream->links_up == 1u &&
		LinksLogged(log_path,"degraded peers 0x0000"),"one switch: every rank on the fabric link alone is not degraded");
	SparkWeightStreamClose(stream);
	stream = 0;

	CHECK(LinksOpen(LINKS_PAIR,2u,&stream,log_path) == SPARK_STATUS_NOT_FOUND && stream == 0 &&
		LinksLogged(log_path,"weight_stream: rank 0 (fabric) and rank 1 (pair) share no link"),"no link in common: the open fails loudly");

	spark_stub_ibv_set_port_state(LINKS_PAIR,IBV_PORT_ACTIVE);
	CHECK(LinksOpen("rocep9s0f9",3u,&stream,log_path) == SPARK_STATUS_OK && stream != 0 && stream->links_up == 1u &&
		LinksLogged(log_path,"weight_stream: rocep9s0f9 gid 3: no such RDMA device; the link is not used"),"a configured device that does not exist is left out");
	SparkWeightStreamClose(stream);
	stream = 0;

	CHECK(LinksOpen(0,1u,&stream,log_path) == SPARK_STATUS_OK && stream != 0 && stream->links_up == 1u &&
		stream->link_of[1] == WEIGHT_STREAM_FABRIC,"no pair device configured: one link");
	SparkWeightStreamClose(stream);
	stream = 0;

	spark_stub_ibv_set_port_state(LINKS_FABRIC,IBV_PORT_DOWN);
	CHECK(LinksOpen(LINKS_PAIR,3u,&stream,log_path) == SPARK_STATUS_OK && stream != 0 && stream->links_up == 2u &&
		stream->link_of[1] == WEIGHT_STREAM_PAIR && stream->link_of[2] == WEIGHT_STREAM_PAIR && LinksAimed(stream,2u,WEIGHT_STREAM_PAIR) != 0u,
		"fabric port down: every peer takes the pair link");
	SparkWeightStreamClose(stream);
	stream = 0;

	spark_stub_ibv_set_port_state(LINKS_PAIR,IBV_PORT_DOWN);
	CHECK(LinksOpen(LINKS_PAIR,3u,&stream,log_path) == SPARK_STATUS_NOT_FOUND && stream == 0 &&
		LinksLogged(log_path,"weight_stream: rank 0 has no active link (fabric " LINKS_FABRIC ", pair " LINKS_PAIR ")"),"no active link: the open fails loudly");
	spark_stub_ibv_set_port_state(LINKS_FABRIC,IBV_PORT_ACTIVE);
	spark_stub_ibv_set_port_state(LINKS_PAIR,IBV_PORT_ACTIVE);

	{
		const uint32_t sizes[] = {2u,3u,4u,6u,8u,16u};
		uint32_t s;
		for (s=0u; s<sizeof(sizes) / sizeof(sizes[0]); s++)
		{
			const uint32_t n = sizes[s];
			uint32_t step,r,ok = 1u;
			for (step=1u; step<n; step++)
			{
				uint32_t seen_dst = 0u;
				for (r=0u; r<n; r++)
				{
					uint32_t dst = WeightStreamSchedulePeer(r,n,step);
					if ( dst >= n || dst == r || (seen_dst & (1u << dst)) != 0u )
						ok = 0u;
					seen_dst |= 1u << dst;
				}
				if ( seen_dst != ((1u << n) - 1u) )
					ok = 0u;
			}
			for (r=0u; r<n; r++)
			{
				uint32_t seen_peer = 0u;
				for (step=1u; step<n; step++)
					seen_peer |= 1u << WeightStreamSchedulePeer(r,n,step);
				if ( seen_peer != (((1u << n) - 1u) & ~(1u << r)) )
					ok = 0u;
			}
			CHECK(ok != 0u,"peer schedule is a conflict-free latin-square permutation");
		}
	}

	(void)unlink(log_path);
	fprintf(stderr,"test_weight_stream_links: %u checks, %u failures\n",links_checks,links_failures);
	return links_failures != 0u ? 1 : 0;
}
