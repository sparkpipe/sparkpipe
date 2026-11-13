#include <arpa/inet.h>
#include "sparkpipe/spark_clock.h"
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weight_stream.h"

#define WEIGHT_STREAM_CQ_DEPTH 16384
#define WEIGHT_STREAM_POLL 64
#define WEIGHT_STREAM_ROW (2u * SPARK_WEIGHT_STREAM_MAX_RANKS)
#define WEIGHT_STREAM_FABRIC 0u
#define WEIGHT_STREAM_PAIR 1u
#define WEIGHT_STREAM_CHUNK_BITS 18u
#define WEIGHT_STREAM_MAX_CHUNKS (1u << WEIGHT_STREAM_CHUNK_BITS)
#define WEIGHT_STREAM_READY_CHUNK (WEIGHT_STREAM_MAX_CHUNKS - 1u)
#define WEIGHT_STREAM_READY_WR (UINT64_C(1) << 32)

typedef struct WeightStreamLink
{
	struct ibv_context *context;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_mr *mr;
	struct ibv_qp *qps[SPARK_WEIGHT_STREAM_MAX_RANKS];
	uint32_t outstanding[SPARK_WEIGHT_STREAM_MAX_RANKS];
	union ibv_gid gid;
} WeightStreamLink;

typedef struct WeightStreamArrival
{
	uint32_t immediate;
	uint32_t link;
} WeightStreamArrival;

typedef struct WeightStreamCall
{
	uint32_t tag;
	uint32_t got[2];
	uint32_t done[2];
	uint32_t ready[2];
} WeightStreamCall;

struct SparkWeightStream
{
	SparkWeightStreamConfig config;
	uint32_t paired;
	WeightStreamLink links[2];
	SparkWeightStreamEndpoint table[SPARK_WEIGHT_STREAM_MAX_RANKS][WEIGHT_STREAM_ROW];
	WeightStreamArrival *early;
	uint32_t early_count;
	uint32_t early_capacity;
	uint32_t broken;
};

static uint32_t WeightStreamImmediate(uint32_t tag, uint32_t source, uint32_t half, uint32_t chunk)
{
	return ((tag & 0xffu) << 24) | ((source & 0x1fu) << 19) | ((half & 1u) << WEIGHT_STREAM_CHUNK_BITS) | (chunk & (WEIGHT_STREAM_MAX_CHUNKS - 1u));
}

static uint32_t WeightStreamTag(uint32_t immediate) { return immediate >> 24; }
static uint32_t WeightStreamSource(uint32_t immediate) { return (immediate >> 19) & 0x1fu; }
static uint32_t WeightStreamChunk(uint32_t immediate) { return immediate & (WEIGHT_STREAM_MAX_CHUNKS - 1u); }

static SparkStatus WeightStreamOpenLink(WeightStreamLink *link, const char *device, uint8_t *region, uint64_t region_bytes, uint32_t gid_index)
{
	struct ibv_device **devices = ibv_get_device_list(0);
	int index;
	memset(link,0,sizeof(*link));
	for (index=0; devices != 0 && devices[index] != 0 && link->context == 0; index++)
		if ( strcmp(ibv_get_device_name(devices[index]),device) == 0 )
			link->context = ibv_open_device(devices[index]);
	if ( devices != 0 )
		ibv_free_device_list(devices);
	if ( link->context == 0 )
	{
		fprintf(stderr,"weight_stream: RDMA device %s not found\n",device);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	link->pd = ibv_alloc_pd(link->context);
	link->cq = link->pd != 0 ? ibv_create_cq(link->context,WEIGHT_STREAM_CQ_DEPTH,0,0,0) : 0;
	link->mr = link->cq != 0 ? ibv_reg_mr(link->pd,region,region_bytes,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) : 0;
	if ( link->mr == 0 || ibv_query_gid(link->context,1,(int)gid_index,&link->gid) != 0 )
	{
		fprintf(stderr,"weight_stream: %s could not register %llu bytes or read gid %u\n",device,(unsigned long long)region_bytes,gid_index);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return SPARK_STATUS_OK;
}

static void WeightStreamCloseLink(WeightStreamLink *link)
{
	uint32_t peer;
	for (peer=0u; peer<SPARK_WEIGHT_STREAM_MAX_RANKS; peer++)
		if ( link->qps[peer] != 0 )
			ibv_destroy_qp(link->qps[peer]);
	if ( link->mr != 0 )
		ibv_dereg_mr(link->mr);
	if ( link->cq != 0 )
		ibv_destroy_cq(link->cq);
	if ( link->pd != 0 )
		ibv_dealloc_pd(link->pd);
	if ( link->context != 0 )
		ibv_close_device(link->context);
	memset(link,0,sizeof(*link));
}

static SparkStatus WeightStreamCreateQp(WeightStreamLink *link, uint32_t peer, uint32_t depth, const uint8_t *region, SparkWeightStreamEndpoint *mine)
{
	struct ibv_qp_init_attr init;
	memset(&init,0,sizeof(init));
	init.send_cq = link->cq;
	init.recv_cq = link->cq;
	init.qp_type = IBV_QPT_RC;
	init.cap.max_send_wr = depth;
	init.cap.max_recv_wr = SPARK_WEIGHT_STREAM_RECEIVE_DEPTH;
	init.cap.max_send_sge = 1;
	init.cap.max_recv_sge = 1;
	link->qps[peer] = ibv_create_qp(link->pd,&init);
	if ( link->qps[peer] == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	mine->qpn = link->qps[peer]->qp_num;
	mine->rkey = link->mr->rkey;
	mine->address = (uint64_t)(uintptr_t)region;
	memcpy(mine->gid,link->gid.raw,sizeof(mine->gid));
	return SPARK_STATUS_OK;
}

static SparkStatus WeightStreamPostReceives(struct ibv_qp *qp, uint32_t count)
{
	struct ibv_recv_wr request,*bad;
	uint32_t index;
	memset(&request,0,sizeof(request));
	for (index=0u; index<count; index++)
		if ( ibv_post_recv(qp,&request,&bad) != 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus WeightStreamConnect(struct ibv_qp *qp, const SparkWeightStreamEndpoint *remote, uint32_t gid_index, uint32_t traffic_class)
{
	struct ibv_qp_attr attributes;
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_INIT;
	attributes.port_num = 1;
	attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
			IBV_QP_MIN_RNR_TIMER) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	memset(&attributes,0,sizeof(attributes));
	attributes.qp_state = IBV_QPS_RTS;
	attributes.timeout = 14;
	attributes.retry_cnt = 7;
	attributes.rnr_retry = 7;
	attributes.max_rd_atomic = 1;
	if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return WeightStreamPostReceives(qp,SPARK_WEIGHT_STREAM_RECEIVE_DEPTH);
}

static uint32_t WeightStreamPartner(const SparkWeightStream *stream)
{
	return stream->config.rank ^ 1u;
}

static uint32_t WeightStreamLinkOf(const SparkWeightStream *stream, uint32_t peer)
{
	return stream->paired != 0u && ((peer ^ stream->config.rank) & 1u) != 0u ? WEIGHT_STREAM_PAIR : WEIGHT_STREAM_FABRIC;
}

static const SparkWeightStreamEndpoint *WeightStreamEndpointOf(const SparkWeightStream *stream, uint32_t peer)
{
	return &stream->table[peer][(WeightStreamLinkOf(stream,peer) == WEIGHT_STREAM_PAIR ? SPARK_WEIGHT_STREAM_MAX_RANKS : 0u) + stream->config.rank];
}

static void WeightStreamCloseAll(SparkWeightStream *stream)
{
	WeightStreamCloseLink(&stream->links[WEIGHT_STREAM_FABRIC]);
	if ( stream->paired != 0u )
		WeightStreamCloseLink(&stream->links[WEIGHT_STREAM_PAIR]);
	free(stream->early);
	free(stream);
}

uint32_t SparkWeightStreamRotationSource(uint32_t rank, uint32_t ranks, uint32_t step)
{
	return ranks == 0u ? 0u : (2u * (rank / 2u) + step % ranks) % ranks;
}

SparkStatus SparkWeightStreamOpen(const SparkWeightStreamConfig *config, SparkWeightStream **stream_out)
{
	SparkWeightStream *stream;
	SparkWeightStreamEndpoint mine[WEIGHT_STREAM_ROW];
	SparkStatus status;
	uint32_t peer;
	if ( stream_out != 0 )
		*stream_out = 0;
	if ( config == 0 || stream_out == 0 || config->ranks < 2u || config->ranks > SPARK_WEIGHT_STREAM_MAX_RANKS || config->rank >= config->ranks ||
		config->chunk_bytes == 0u || config->depth == 0u || config->fabric_device == 0 || config->region == 0 || config->region_bytes == 0u ||
		config->gather == 0 || config->traffic_class > 255u || (config->pair_device != 0 && (config->ranks % 2u) != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	stream = (SparkWeightStream *)calloc(1u,sizeof(*stream));
	if ( stream == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	stream->config = *config;
	stream->paired = config->pair_device != 0 ? 1u : 0u;
	status = WeightStreamOpenLink(&stream->links[WEIGHT_STREAM_FABRIC],config->fabric_device,config->region,config->region_bytes,config->fabric_gid_index);
	if ( status == SPARK_STATUS_OK && stream->paired != 0u )
		status = WeightStreamOpenLink(&stream->links[WEIGHT_STREAM_PAIR],config->pair_device,config->region,config->region_bytes,config->pair_gid_index);
	memset(mine,0,sizeof(mine));
	for (peer=0u; peer<config->ranks && status == SPARK_STATUS_OK; peer++)
		if ( peer != config->rank )
		{
			const uint32_t link = WeightStreamLinkOf(stream,peer);
			status = WeightStreamCreateQp(&stream->links[link],peer,config->depth,config->region,&mine[(link == WEIGHT_STREAM_PAIR ? SPARK_WEIGHT_STREAM_MAX_RANKS : 0u) + peer]);
		}
	if ( status == SPARK_STATUS_OK )
		status = config->gather(config->gather_context,mine,stream->table,sizeof(mine));
	for (peer=0u; peer<config->ranks && status == SPARK_STATUS_OK; peer++)
		if ( peer != config->rank )
		{
			const uint32_t link = WeightStreamLinkOf(stream,peer);
			status = WeightStreamConnect(stream->links[link].qps[peer],WeightStreamEndpointOf(stream,peer),link == WEIGHT_STREAM_PAIR ? config->pair_gid_index : config->fabric_gid_index,
				config->traffic_class);
		}
	if ( status != SPARK_STATUS_OK )
	{
		WeightStreamCloseAll(stream);
		return status;
	}
	*stream_out = stream;
	return SPARK_STATUS_OK;
}

void SparkWeightStreamClose(SparkWeightStream *stream)
{
	if ( stream != 0 )
		WeightStreamCloseAll(stream);
}

static SparkStatus WeightStreamWrite(SparkWeightStream *stream, uint32_t peer, uint64_t local_offset, uint64_t remote_offset, uint64_t length, uint32_t immediate)
{
	WeightStreamLink *link = &stream->links[WeightStreamLinkOf(stream,peer)];
	const SparkWeightStreamEndpoint *remote = WeightStreamEndpointOf(stream,peer);
	struct ibv_sge scatter;
	struct ibv_send_wr request,*bad;
	if ( local_offset + length > stream->config.region_bytes || remote_offset + length > stream->config.region_bytes || length > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&scatter,0,sizeof(scatter));
	scatter.addr = (uint64_t)(uintptr_t)(stream->config.region + local_offset);
	scatter.length = (uint32_t)length;
	scatter.lkey = link->mr->lkey;
	memset(&request,0,sizeof(request));
	request.wr_id = peer;
	request.sg_list = &scatter;
	request.num_sge = 1;
	request.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	request.send_flags = IBV_SEND_SIGNALED;
	request.imm_data = htonl(immediate);
	request.wr.rdma.remote_addr = remote->address + remote_offset;
	request.wr.rdma.rkey = remote->rkey;
	if ( ibv_post_send(link->qps[peer],&request,&bad) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	link->outstanding[peer]++;
	return SPARK_STATUS_OK;
}

static SparkStatus WeightStreamReady(SparkWeightStream *stream, uint32_t peer, uint32_t tag)
{
	WeightStreamLink *link = &stream->links[WeightStreamLinkOf(stream,peer)];
	const SparkWeightStreamEndpoint *remote = WeightStreamEndpointOf(stream,peer);
	struct ibv_send_wr request,*bad;
	memset(&request,0,sizeof(request));
	request.wr_id = WEIGHT_STREAM_READY_WR | peer;
	request.num_sge = 0;
	request.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	request.send_flags = IBV_SEND_SIGNALED;
	request.imm_data = htonl(WeightStreamImmediate(tag,stream->config.rank,1u,WEIGHT_STREAM_READY_CHUNK));
	request.wr.rdma.remote_addr = remote->address;
	request.wr.rdma.rkey = remote->rkey;
	if ( ibv_post_send(link->qps[peer],&request,&bad) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	link->outstanding[peer]++;
	return SPARK_STATUS_OK;
}

static SparkStatus WeightStreamStash(SparkWeightStream *stream, uint32_t immediate, uint32_t link)
{
	if ( stream->early_count == stream->early_capacity )
	{
		const uint32_t capacity = stream->early_capacity != 0u ? 2u * stream->early_capacity : 1024u;
		WeightStreamArrival *grown = (WeightStreamArrival *)realloc(stream->early,(size_t)capacity * sizeof(*grown));
		if ( grown == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		stream->early = grown;
		stream->early_capacity = capacity;
	}
	stream->early[stream->early_count].immediate = immediate;
	stream->early[stream->early_count].link = link;
	stream->early_count++;
	return SPARK_STATUS_OK;
}

static SparkStatus WeightStreamArrive(WeightStreamCall *call, uint32_t immediate, uint32_t link)
{
	if ( WeightStreamChunk(immediate) == WEIGHT_STREAM_READY_CHUNK )
	{
		call->ready[link] |= 1u << WeightStreamSource(immediate);
		return SPARK_STATUS_OK;
	}
	call->got[link]++;
	return SPARK_STATUS_OK;
}

static SparkStatus WeightStreamTakeEarly(SparkWeightStream *stream, WeightStreamCall *call)
{
	uint32_t index,kept = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	for (index=0u; index<stream->early_count; index++)
	{
		if ( status == SPARK_STATUS_OK && WeightStreamTag(stream->early[index].immediate) == (call->tag & 0xffu) )
			status = WeightStreamArrive(call,stream->early[index].immediate,stream->early[index].link);
		else
			stream->early[kept++] = stream->early[index];
	}
	stream->early_count = kept;
	return status;
}

static SparkStatus WeightStreamPoll(SparkWeightStream *stream, WeightStreamCall *call, uint32_t link_index)
{
	WeightStreamLink *link = &stream->links[link_index];
	struct ibv_wc done[WEIGHT_STREAM_POLL];
	SparkStatus status;
	uint32_t peer;
	int count,item;
	count = ibv_poll_cq(link->cq,WEIGHT_STREAM_POLL,done);
	if ( count < 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	for (item=0; item<count; item++)
	{
		if ( done[item].status != IBV_WC_SUCCESS )
		{
			fprintf(stderr,"weight_stream: rank %u %s completion failed: %s (vendor 0x%x)\n",stream->config.rank,link_index == WEIGHT_STREAM_PAIR ? "pair" : "fabric",
				ibv_wc_status_str(done[item].status),done[item].vendor_err);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		if ( done[item].opcode != IBV_WC_RECV_RDMA_WITH_IMM )
		{
			link->outstanding[(uint32_t)done[item].wr_id]--;
			if ( (done[item].wr_id & WEIGHT_STREAM_READY_WR) == 0u )
				call->done[link_index]++;
			continue;
		}
		for (peer=0u; peer<stream->config.ranks; peer++)
			if ( link->qps[peer] != 0 && link->qps[peer]->qp_num == done[item].qp_num )
				break;
		if ( peer == stream->config.ranks || WeightStreamPostReceives(link->qps[peer],1u) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		{
			const uint32_t immediate = ntohl(done[item].imm_data);
			status = WeightStreamTag(immediate) == (call->tag & 0xffu) ? WeightStreamArrive(call,immediate,link_index) : WeightStreamStash(stream,immediate,link_index);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus WeightStreamCallBegin(SparkWeightStream *stream, WeightStreamCall *call, uint32_t tag)
{
	memset(call,0,sizeof(*call));
	if ( stream->broken != 0u )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	call->tag = tag;
	return WeightStreamTakeEarly(stream,call);
}

static SparkStatus WeightStreamFail(SparkWeightStream *stream, SparkStatus status)
{
	stream->broken = 1u;
	return status;
}

static uint64_t WeightStreamChunkLength(uint64_t total, uint32_t chunk_bytes, uint32_t chunk)
{
	const uint64_t start = (uint64_t)chunk * chunk_bytes;
	return total - start < chunk_bytes ? total - start : chunk_bytes;
}

static SparkStatus WeightStreamPollAll(SparkWeightStream *stream, WeightStreamCall *call)
{
	SparkStatus status = WeightStreamPoll(stream,call,WEIGHT_STREAM_FABRIC);
	if ( status == SPARK_STATUS_OK && stream->paired != 0u )
		status = WeightStreamPoll(stream,call,WEIGHT_STREAM_PAIR);
	return status;
}

static SparkStatus WeightStreamEveryPeer(SparkWeightStream *stream, uint64_t send_offset, uint64_t send_pitch, uint64_t bytes, uint64_t landing_offset, uint32_t tag)
{
	const SparkWeightStreamConfig *config = &stream->config;
	const uint32_t chunks = (uint32_t)((bytes + config->chunk_bytes - 1u) / config->chunk_bytes);
	uint32_t next_chunk[SPARK_WEIGHT_STREAM_MAX_RANKS],expected[2] = {0u,0u},peer,moved;
	uint64_t start,mine;
	WeightStreamCall call;
	SparkStatus status;
	memset(next_chunk,0,sizeof(next_chunk));
	for (peer=0u; peer<config->ranks; peer++)
		if ( peer != config->rank )
			expected[WeightStreamLinkOf(stream,peer)] += chunks;
	status = WeightStreamCallBegin(stream,&call,tag);
	if ( status != SPARK_STATUS_OK )
		return WeightStreamFail(stream,status);
	mine = send_offset + (uint64_t)config->rank * send_pitch;
	if ( mine != landing_offset + (uint64_t)config->rank * bytes )
		memmove(config->region + landing_offset + (uint64_t)config->rank * bytes,config->region + mine,bytes);
	for (peer=0u; peer<config->ranks && status == SPARK_STATUS_OK; peer++)
		if ( peer != config->rank )
			status = WeightStreamReady(stream,peer,tag);
	if ( status != SPARK_STATUS_OK )
		return WeightStreamFail(stream,status);
	start = SparkMonotonicNs();
	while ( call.done[WEIGHT_STREAM_FABRIC] < expected[WEIGHT_STREAM_FABRIC] || call.done[WEIGHT_STREAM_PAIR] < expected[WEIGHT_STREAM_PAIR] ||
		call.got[WEIGHT_STREAM_FABRIC] < expected[WEIGHT_STREAM_FABRIC] || call.got[WEIGHT_STREAM_PAIR] < expected[WEIGHT_STREAM_PAIR] )
	{
		do
		{
			moved = 0u;
			for (peer=0u; peer<config->ranks && status == SPARK_STATUS_OK; peer++)
			{
				const uint32_t link = WeightStreamLinkOf(stream,peer),chunk = next_chunk[peer];
				if ( peer == config->rank || chunk >= chunks || (call.ready[link] & (1u << peer)) == 0u || stream->links[link].outstanding[peer] >= config->depth )
					continue;
				status = WeightStreamWrite(stream,peer,send_offset + (uint64_t)peer * send_pitch + (uint64_t)chunk * config->chunk_bytes,
					landing_offset + (uint64_t)config->rank * bytes + (uint64_t)chunk * config->chunk_bytes,WeightStreamChunkLength(bytes,config->chunk_bytes,chunk),
					WeightStreamImmediate(tag,config->rank,0u,chunk));
				next_chunk[peer]++;
				moved++;
			}
		}
		while ( moved != 0u && status == SPARK_STATUS_OK );
		if ( status == SPARK_STATUS_OK )
			status = WeightStreamPollAll(stream,&call);
		if ( status != SPARK_STATUS_OK )
			return WeightStreamFail(stream,status);
		if ( SparkMonotonicNs() - start > SPARK_WEIGHT_STREAM_TIMEOUT_NS )
		{
			fprintf(stderr,"weight_stream: rank %u transfer tag %u stalled: done %u/%u + %u/%u got %u/%u + %u/%u ready 0x%x + 0x%x\n",config->rank,tag,
				call.done[WEIGHT_STREAM_FABRIC],expected[WEIGHT_STREAM_FABRIC],call.done[WEIGHT_STREAM_PAIR],expected[WEIGHT_STREAM_PAIR],call.got[WEIGHT_STREAM_FABRIC],
				expected[WEIGHT_STREAM_FABRIC],call.got[WEIGHT_STREAM_PAIR],expected[WEIGHT_STREAM_PAIR],call.ready[WEIGHT_STREAM_FABRIC],call.ready[WEIGHT_STREAM_PAIR]);
			return WeightStreamFail(stream,SPARK_STATUS_IO_ERROR);
		}
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkWeightStreamBroadcast(SparkWeightStream *stream, uint64_t source_offset, uint64_t bytes, uint64_t landing_offset, uint32_t tag)
{
	const SparkWeightStreamConfig *config = stream != 0 ? &stream->config : 0;
	if ( config == 0 || bytes == 0u || source_offset + bytes > config->region_bytes || landing_offset + (uint64_t)config->ranks * bytes > config->region_bytes ||
		(bytes + config->chunk_bytes - 1u) / config->chunk_bytes >= WEIGHT_STREAM_MAX_CHUNKS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return WeightStreamEveryPeer(stream,source_offset,0u,bytes,landing_offset,tag);
}

SparkStatus SparkWeightStreamExchange(SparkWeightStream *stream, uint64_t send_offset, uint64_t send_pitch, uint64_t bytes, uint64_t landing_offset, uint32_t tag)
{
	const SparkWeightStreamConfig *config = stream != 0 ? &stream->config : 0;
	if ( config == 0 || bytes == 0u || send_pitch < bytes || send_offset + (uint64_t)(config->ranks - 1u) * send_pitch + bytes > config->region_bytes ||
		landing_offset + (uint64_t)config->ranks * bytes > config->region_bytes || (bytes + config->chunk_bytes - 1u) / config->chunk_bytes >= WEIGHT_STREAM_MAX_CHUNKS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return WeightStreamEveryPeer(stream,send_offset,send_pitch,bytes,landing_offset,tag);
}

static uint32_t WeightStreamRotatePair(uint32_t rank, uint32_t ranks, uint32_t k, uint32_t *holds_own)
{
	*holds_own = (rank & 1u) == (k & 1u) ? 1u : 0u;
	return (((*holds_own != 0u ? rank : rank ^ 1u) + ranks - k) % ranks) / 2u;
}

static uint32_t WeightStreamAllReady(const WeightStreamCall *call, const SparkWeightStream *stream, const uint32_t *targets, uint32_t count)
{
	uint32_t index;
	for (index=0u; index<count; index++)
		if ( (call->ready[WeightStreamLinkOf(stream,targets[index])] & (1u << targets[index])) == 0u )
			return 0u;
	return 1u;
}

static uint32_t WeightStreamAllSent(const uint32_t *sent, uint32_t count, uint32_t per_half)
{
	uint32_t index;
	for (index=0u; index<count; index++)
		if ( sent[index] < per_half )
			return 0u;
	return 1u;
}

SparkStatus SparkWeightStreamRotate(SparkWeightStream *stream, uint32_t step, uint64_t own_offset, uint64_t partner_offset, uint64_t bytes,
	uint64_t landing_offset, uint64_t pair_landing_offset, uint32_t tag, SparkWeightStreamBlock *block)
{
	const SparkWeightStreamConfig *config = stream != 0 ? &stream->config : 0;
	uint32_t ranks,rank,partner,k,source,per_half,holds_own,target_pair,send_half,targets[2],target_count = 0u,sent[2] = {0u,0u},writers[2],writer_count = 0u;
	uint32_t expect_done[2] = {0u,0u},expect_got[2] = {0u,0u},index,moved;
	uint64_t half,send_base,start;
	WeightStreamCall call;
	SparkStatus status;
	if ( config == 0 || block == 0 || stream->paired == 0u || bytes == 0u || (bytes % 2u) != 0u || own_offset + bytes > config->region_bytes ||
		partner_offset + bytes > config->region_bytes || landing_offset + bytes > config->region_bytes || pair_landing_offset + bytes > config->region_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	ranks = config->ranks;
	rank = config->rank;
	partner = WeightStreamPartner(stream);
	k = step % ranks;
	source = SparkWeightStreamRotationSource(rank,ranks,step);
	half = bytes / 2u;
	if ( (half + config->chunk_bytes - 1u) / config->chunk_bytes >= WEIGHT_STREAM_MAX_CHUNKS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	per_half = (uint32_t)((half + config->chunk_bytes - 1u) / config->chunk_bytes);
	target_pair = WeightStreamRotatePair(rank,ranks,k,&holds_own);
	send_half = holds_own != 0u ? 0u : 1u;
	send_base = send_half == 0u ? own_offset : partner_offset + half;
	if ( target_pair != rank / 2u )
	{
		targets[target_count++] = 2u * target_pair;
		targets[target_count++] = 2u * target_pair + 1u;
	}
	else if ( holds_own != 0u )
		targets[target_count++] = partner;
	if ( source == partner )
		writers[writer_count++] = partner;
	else if ( source != rank )
	{
		writers[writer_count++] = source;
		writers[writer_count++] = source ^ 1u;
	}
	for (index=0u; index<target_count; index++)
		expect_done[WeightStreamLinkOf(stream,targets[index])] += per_half;
	for (index=0u; index<writer_count; index++)
		expect_got[WeightStreamLinkOf(stream,writers[index])] += per_half;
	status = WeightStreamCallBegin(stream,&call,tag);
	for (index=0u; index<writer_count && status == SPARK_STATUS_OK; index++)
		status = WeightStreamReady(stream,writers[index],tag);
	if ( status != SPARK_STATUS_OK )
		return WeightStreamFail(stream,status);
	start = SparkMonotonicNs();
	block->ready_ns = block->posted_ns = 0u;
	while ( call.done[WEIGHT_STREAM_FABRIC] < expect_done[WEIGHT_STREAM_FABRIC] || call.done[WEIGHT_STREAM_PAIR] < expect_done[WEIGHT_STREAM_PAIR] ||
		call.got[WEIGHT_STREAM_FABRIC] < expect_got[WEIGHT_STREAM_FABRIC] || call.got[WEIGHT_STREAM_PAIR] < expect_got[WEIGHT_STREAM_PAIR] )
	{
		do
		{
			moved = 0u;
			for (index=0u; index<target_count; index++)
			{
				const uint32_t target = targets[index],link = WeightStreamLinkOf(stream,target);
				if ( sent[index] >= per_half || (call.ready[link] & (1u << target)) == 0u || stream->links[link].outstanding[target] >= config->depth )
					continue;
				status = WeightStreamWrite(stream,target,send_base + (uint64_t)sent[index] * config->chunk_bytes,
					(target == partner ? pair_landing_offset : landing_offset) + (uint64_t)send_half * half + (uint64_t)sent[index] * config->chunk_bytes,WeightStreamChunkLength(half,config->chunk_bytes,sent[index]),
					WeightStreamImmediate(tag,rank,send_half,sent[index]));
				if ( status != SPARK_STATUS_OK )
					return WeightStreamFail(stream,status);
				sent[index]++;
				moved++;
			}
		}
		while ( moved != 0u );
		if ( block->ready_ns == 0u && WeightStreamAllReady(&call,stream,targets,target_count) != 0u )
			block->ready_ns = SparkMonotonicNs() - start;
		if ( block->posted_ns == 0u && WeightStreamAllSent(sent,target_count,per_half) != 0u )
			block->posted_ns = SparkMonotonicNs() - start;
		status = WeightStreamPollAll(stream,&call);
		if ( status != SPARK_STATUS_OK )
			return WeightStreamFail(stream,status);
		if ( SparkMonotonicNs() - start > SPARK_WEIGHT_STREAM_TIMEOUT_NS )
		{
			fprintf(stderr,"weight_stream: rank %u rotation step %u (block of rank %u) stalled: sent %u + %u done %u/%u + %u/%u got %u/%u + %u/%u ready 0x%x + 0x%x\n",rank,step,
				source,sent[0],sent[1],call.done[WEIGHT_STREAM_FABRIC],expect_done[WEIGHT_STREAM_FABRIC],call.done[WEIGHT_STREAM_PAIR],expect_done[WEIGHT_STREAM_PAIR],
				call.got[WEIGHT_STREAM_FABRIC],expect_got[WEIGHT_STREAM_FABRIC],call.got[WEIGHT_STREAM_PAIR],expect_got[WEIGHT_STREAM_PAIR],call.ready[WEIGHT_STREAM_FABRIC],
				call.ready[WEIGHT_STREAM_PAIR]);
			return WeightStreamFail(stream,SPARK_STATUS_IO_ERROR);
		}
	}
	block->source = source;
	block->reserved0 = 0u;
	block->total_ns = SparkMonotonicNs() - start;
	if ( source == rank )
	{
		block->half_offset[0] = own_offset;
		block->half_offset[1] = own_offset + half;
	}
	else if ( source == partner )
	{
		block->half_offset[0] = pair_landing_offset;
		block->half_offset[1] = partner_offset + half;
	}
	else
	{
		block->half_offset[0] = landing_offset;
		block->half_offset[1] = landing_offset + half;
	}
	return SPARK_STATUS_OK;
}
