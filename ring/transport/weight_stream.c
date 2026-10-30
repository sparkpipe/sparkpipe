#include <arpa/inet.h>
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
	uint32_t relay_head;
	uint32_t relay_tail;
	uint32_t *relay;
	uint32_t relay_capacity;
	uint32_t relay_fabric;
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
	uint32_t *relay;
	uint32_t relay_capacity;
	uint32_t broken;
};

static uint64_t WeightStreamNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static uint32_t WeightStreamImmediate(uint32_t tag, uint32_t source, uint32_t half, uint32_t chunk)
{
	return ((tag & 0xffu) << 24) | ((source & 0x1fu) << 19) | ((half & 1u) << WEIGHT_STREAM_CHUNK_BITS) | (chunk & (WEIGHT_STREAM_MAX_CHUNKS - 1u));
}

static uint32_t WeightStreamTag(uint32_t immediate) { return immediate >> 24; }
static uint32_t WeightStreamSource(uint32_t immediate) { return (immediate >> 19) & 0x1fu; }
static uint32_t WeightStreamHalf(uint32_t immediate) { return (immediate >> WEIGHT_STREAM_CHUNK_BITS) & 1u; }
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

static uint32_t WeightStreamUsesFabric(const SparkWeightStream *stream, uint32_t peer)
{
	return peer != stream->config.rank && (stream->paired == 0u || peer != WeightStreamPartner(stream)) ? 1u : 0u;
}

static void WeightStreamCloseAll(SparkWeightStream *stream)
{
	WeightStreamCloseLink(&stream->links[WEIGHT_STREAM_FABRIC]);
	if ( stream->paired != 0u )
		WeightStreamCloseLink(&stream->links[WEIGHT_STREAM_PAIR]);
	free(stream->early);
	free(stream->relay);
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
	{
		if ( WeightStreamUsesFabric(stream,peer) != 0u )
			status = WeightStreamCreateQp(&stream->links[WEIGHT_STREAM_FABRIC],peer,config->depth,config->region,&mine[peer]);
		else if ( stream->paired != 0u && peer == WeightStreamPartner(stream) )
			status = WeightStreamCreateQp(&stream->links[WEIGHT_STREAM_PAIR],peer,config->depth,config->region,&mine[SPARK_WEIGHT_STREAM_MAX_RANKS + peer]);
	}
	if ( status == SPARK_STATUS_OK )
		status = config->gather(config->gather_context,mine,stream->table,sizeof(mine));
	for (peer=0u; peer<config->ranks && status == SPARK_STATUS_OK; peer++)
	{
		if ( WeightStreamUsesFabric(stream,peer) != 0u )
			status = WeightStreamConnect(stream->links[WEIGHT_STREAM_FABRIC].qps[peer],&stream->table[peer][config->rank],config->fabric_gid_index,config->traffic_class);
		else if ( stream->paired != 0u && peer == WeightStreamPartner(stream) )
			status = WeightStreamConnect(stream->links[WEIGHT_STREAM_PAIR].qps[peer],&stream->table[peer][SPARK_WEIGHT_STREAM_MAX_RANKS + config->rank],config->pair_gid_index,
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

static SparkStatus WeightStreamWrite(SparkWeightStream *stream, uint32_t link_index, uint32_t peer, uint64_t local_offset, uint64_t remote_offset, uint64_t length,
	uint32_t immediate)
{
	WeightStreamLink *link = &stream->links[link_index];
	const SparkWeightStreamEndpoint *remote = &stream->table[peer][(link_index == WEIGHT_STREAM_PAIR ? SPARK_WEIGHT_STREAM_MAX_RANKS : 0u) + stream->config.rank];
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

static SparkStatus WeightStreamReady(SparkWeightStream *stream, uint32_t link_index, uint32_t peer, uint32_t tag)
{
	WeightStreamLink *link = &stream->links[link_index];
	const SparkWeightStreamEndpoint *remote = &stream->table[peer][(link_index == WEIGHT_STREAM_PAIR ? SPARK_WEIGHT_STREAM_MAX_RANKS : 0u) + stream->config.rank];
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

static SparkStatus WeightStreamRelayPush(WeightStreamCall *call, uint32_t immediate)
{
	if ( call->relay_tail >= call->relay_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	call->relay[call->relay_tail++] = immediate;
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
	return link == WEIGHT_STREAM_FABRIC && call->relay_fabric != 0u ? WeightStreamRelayPush(call,immediate) : SPARK_STATUS_OK;
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

static SparkStatus WeightStreamCallBegin(SparkWeightStream *stream, WeightStreamCall *call, uint32_t tag, uint32_t relay_fabric, uint32_t relay_capacity)
{
	memset(call,0,sizeof(*call));
	if ( stream->broken != 0u )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( relay_capacity > stream->relay_capacity )
	{
		uint32_t *grown = (uint32_t *)realloc(stream->relay,(size_t)relay_capacity * sizeof(*grown));
		if ( grown == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		stream->relay = grown;
		stream->relay_capacity = relay_capacity;
	}
	call->tag = tag;
	call->relay = stream->relay;
	call->relay_capacity = relay_capacity;
	call->relay_fabric = relay_fabric;
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

SparkStatus SparkWeightStreamBroadcast(SparkWeightStream *stream, uint64_t source_offset, uint64_t bytes, uint64_t landing_offset, uint32_t tag)
{
	const SparkWeightStreamConfig *config = stream != 0 ? &stream->config : 0;
	uint32_t fabric_list[SPARK_WEIGHT_STREAM_MAX_RANKS],fabric_peers = 0u,peer,chunks,expected_fabric,expected_pair,sent_fabric = 0u,next_peer = 0u,next_chunk = 0u;
	uint32_t own_pair = 0u,turn = 1u,partner,fabric_mask = 0u;
	uint64_t start;
	WeightStreamCall call;
	SparkStatus status;
	if ( config == 0 || bytes == 0u || source_offset + bytes > config->region_bytes || landing_offset + (uint64_t)config->ranks * bytes > config->region_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	chunks = (uint32_t)((bytes + config->chunk_bytes - 1u) / config->chunk_bytes);
	if ( (bytes + config->chunk_bytes - 1u) / config->chunk_bytes >= WEIGHT_STREAM_MAX_CHUNKS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	partner = WeightStreamPartner(stream);
	for (peer=1u; peer<config->ranks; peer++)
	{
		const uint32_t other = (config->rank + peer) % config->ranks;
		if ( WeightStreamUsesFabric(stream,other) != 0u && (stream->paired == 0u || (other & 1u) == (config->rank & 1u)) )
		{
			fabric_list[fabric_peers++] = other;
			fabric_mask |= 1u << other;
		}
	}
	expected_fabric = fabric_peers * chunks;
	expected_pair = stream->paired != 0u ? (1u + fabric_peers) * chunks : 0u;
	status = WeightStreamCallBegin(stream,&call,tag,stream->paired,expected_fabric);
	if ( status != SPARK_STATUS_OK )
		return WeightStreamFail(stream,status);
	if ( source_offset != landing_offset + (uint64_t)config->rank * bytes )
		memmove(config->region + landing_offset + (uint64_t)config->rank * bytes,config->region + source_offset,bytes);
	for (peer=0u; peer<fabric_peers && status == SPARK_STATUS_OK; peer++)
		status = WeightStreamReady(stream,WEIGHT_STREAM_FABRIC,fabric_list[peer],tag);
	if ( status == SPARK_STATUS_OK && stream->paired != 0u )
		status = WeightStreamReady(stream,WEIGHT_STREAM_PAIR,partner,tag);
	if ( status != SPARK_STATUS_OK )
		return WeightStreamFail(stream,status);
	start = WeightStreamNowNs();
	while ( call.done[WEIGHT_STREAM_FABRIC] < expected_fabric || call.done[WEIGHT_STREAM_PAIR] < expected_pair ||
		call.got[WEIGHT_STREAM_FABRIC] < expected_fabric || call.got[WEIGHT_STREAM_PAIR] < expected_pair )
	{
		while ( sent_fabric < expected_fabric && (call.ready[WEIGHT_STREAM_FABRIC] & fabric_mask) == fabric_mask &&
			stream->links[WEIGHT_STREAM_FABRIC].outstanding[fabric_list[next_peer]] < config->depth )
		{
			const uint64_t offset = landing_offset + (uint64_t)config->rank * bytes + (uint64_t)next_chunk * config->chunk_bytes;
			status = WeightStreamWrite(stream,WEIGHT_STREAM_FABRIC,fabric_list[next_peer],offset,offset,WeightStreamChunkLength(bytes,config->chunk_bytes,next_chunk),
				WeightStreamImmediate(tag,config->rank,0u,next_chunk));
			if ( status != SPARK_STATUS_OK )
				return WeightStreamFail(stream,status);
			sent_fabric++;
			next_peer = (next_peer + 1u) % fabric_peers;
			if ( next_peer == 0u )
				next_chunk++;
		}
		while ( stream->paired != 0u && (call.ready[WEIGHT_STREAM_PAIR] & (1u << partner)) != 0u && stream->links[WEIGHT_STREAM_PAIR].outstanding[partner] < config->depth &&
			(own_pair < chunks || call.relay_head != call.relay_tail) )
		{
			uint32_t source,chunk;
			if ( call.relay_head != call.relay_tail && (own_pair >= chunks || turn != 0u) )
			{
				source = WeightStreamSource(call.relay[call.relay_head]);
				chunk = WeightStreamChunk(call.relay[call.relay_head]);
				call.relay_head++;
			}
			else
			{
				source = config->rank;
				chunk = own_pair++;
			}
			{
				const uint64_t offset = landing_offset + (uint64_t)source * bytes + (uint64_t)chunk * config->chunk_bytes;
				status = WeightStreamWrite(stream,WEIGHT_STREAM_PAIR,partner,offset,offset,WeightStreamChunkLength(bytes,config->chunk_bytes,chunk),
					WeightStreamImmediate(tag,source,0u,chunk));
			}
			if ( status != SPARK_STATUS_OK )
				return WeightStreamFail(stream,status);
			turn ^= 1u;
		}
		status = WeightStreamPoll(stream,&call,WEIGHT_STREAM_FABRIC);
		if ( status == SPARK_STATUS_OK && stream->paired != 0u )
			status = WeightStreamPoll(stream,&call,WEIGHT_STREAM_PAIR);
		if ( status != SPARK_STATUS_OK )
			return WeightStreamFail(stream,status);
		if ( WeightStreamNowNs() - start > SPARK_WEIGHT_STREAM_TIMEOUT_NS )
		{
			fprintf(stderr,"weight_stream: rank %u broadcast tag %u stalled: sent %u done %u/%u + %u/%u got %u/%u + %u/%u ready 0x%x/0x%x + 0x%x\n",config->rank,tag,sent_fabric,
				call.done[WEIGHT_STREAM_FABRIC],expected_fabric,call.done[WEIGHT_STREAM_PAIR],expected_pair,call.got[WEIGHT_STREAM_FABRIC],expected_fabric,
				call.got[WEIGHT_STREAM_PAIR],expected_pair,call.ready[WEIGHT_STREAM_FABRIC],fabric_mask,call.ready[WEIGHT_STREAM_PAIR]);
			return WeightStreamFail(stream,SPARK_STATUS_IO_ERROR);
		}
	}
	return SPARK_STATUS_OK;
}

static uint32_t WeightStreamRotateTarget(uint32_t rank, uint32_t ranks, uint32_t k, uint32_t *pair_own)
{
	const uint32_t holds_own = (rank & 1u) == (k & 1u) ? 1u : 0u, target = (((holds_own != 0u ? rank : rank ^ 1u) + ranks - k) % ranks) / 2u;
	*pair_own = 0u;
	if ( target == rank / 2u )
	{
		*pair_own = holds_own;
		return(UINT32_MAX);
	}
	return(2u * target + (holds_own != 0u ? 0u : 1u));
}

SparkStatus SparkWeightStreamRotate(SparkWeightStream *stream, uint32_t step, uint64_t own_offset, uint64_t partner_offset, uint64_t bytes,
	uint64_t landing_offset, uint32_t tag, SparkWeightStreamBlock *block)
{
	const SparkWeightStreamConfig *config = stream != 0 ? &stream->config : 0;
	uint32_t ranks,rank,partner,k,source,per_half,send_peer,send_half = 0u,pair_own = 0u,receive_fabric = 0u,receive_pair = 0u,writer = UINT32_MAX,peer,unused;
	uint32_t sent = 0u,own_sent = 0u,expect_fabric_done,expect_pair_done;
	uint64_t half,send_base = 0u,start;
	WeightStreamCall call;
	SparkStatus status;
	if ( config == 0 || block == 0 || stream->paired == 0u || bytes == 0u || (bytes % 2u) != 0u || own_offset + bytes > config->region_bytes ||
		partner_offset + bytes > config->region_bytes || landing_offset + bytes > config->region_bytes )
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
	send_peer = WeightStreamRotateTarget(rank,ranks,k,&pair_own);
	if ( send_peer != UINT32_MAX )
	{
		send_half = send_peer & 1u;
		send_base = send_half == 0u ? own_offset : partner_offset + half;
	}
	for (peer=0u; peer<ranks; peer++)
		if ( peer != rank && WeightStreamRotateTarget(peer,ranks,k,&unused) == rank )
			writer = peer;
	if ( source == partner )
		receive_pair = per_half;
	else if ( source != rank )
	{
		receive_fabric = per_half;
		receive_pair = per_half;
	}
	expect_fabric_done = send_peer != UINT32_MAX ? per_half : 0u;
	expect_pair_done = (pair_own != 0u ? per_half : 0u) + receive_fabric;
	if ( (receive_fabric != 0u) != (writer != UINT32_MAX) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = WeightStreamCallBegin(stream,&call,tag,receive_fabric != 0u ? 1u : 0u,receive_fabric);
	if ( status == SPARK_STATUS_OK && receive_fabric != 0u )
		status = WeightStreamReady(stream,WEIGHT_STREAM_FABRIC,writer,tag);
	if ( status == SPARK_STATUS_OK && receive_pair != 0u )
		status = WeightStreamReady(stream,WEIGHT_STREAM_PAIR,partner,tag);
	if ( status != SPARK_STATUS_OK )
		return WeightStreamFail(stream,status);
	start = WeightStreamNowNs();
	while ( call.done[WEIGHT_STREAM_FABRIC] < expect_fabric_done || call.done[WEIGHT_STREAM_PAIR] < expect_pair_done ||
		call.got[WEIGHT_STREAM_FABRIC] < receive_fabric || call.got[WEIGHT_STREAM_PAIR] < receive_pair )
	{
		while ( send_peer != UINT32_MAX && sent < per_half && (call.ready[WEIGHT_STREAM_FABRIC] & (1u << send_peer)) != 0u &&
			stream->links[WEIGHT_STREAM_FABRIC].outstanding[send_peer] < config->depth )
		{
			status = WeightStreamWrite(stream,WEIGHT_STREAM_FABRIC,send_peer,send_base + (uint64_t)sent * config->chunk_bytes,
				landing_offset + (uint64_t)send_half * half + (uint64_t)sent * config->chunk_bytes,WeightStreamChunkLength(half,config->chunk_bytes,sent),
				WeightStreamImmediate(tag,rank,send_half,sent));
			if ( status != SPARK_STATUS_OK )
				return WeightStreamFail(stream,status);
			sent++;
		}
		while ( (call.ready[WEIGHT_STREAM_PAIR] & (1u << partner)) != 0u && stream->links[WEIGHT_STREAM_PAIR].outstanding[partner] < config->depth &&
			((pair_own != 0u && own_sent < per_half) || call.relay_head != call.relay_tail) )
		{
			if ( call.relay_head != call.relay_tail )
			{
				const uint32_t immediate = call.relay[call.relay_head++];
				const uint32_t chunk = WeightStreamChunk(immediate),which = WeightStreamHalf(immediate);
				const uint64_t offset = landing_offset + (uint64_t)which * half + (uint64_t)chunk * config->chunk_bytes;
				status = WeightStreamWrite(stream,WEIGHT_STREAM_PAIR,partner,offset,offset,WeightStreamChunkLength(half,config->chunk_bytes,chunk),
					WeightStreamImmediate(tag,rank,which,chunk));
			}
			else
			{
				status = WeightStreamWrite(stream,WEIGHT_STREAM_PAIR,partner,own_offset + (uint64_t)own_sent * config->chunk_bytes,
					landing_offset + (uint64_t)own_sent * config->chunk_bytes,WeightStreamChunkLength(half,config->chunk_bytes,own_sent),
					WeightStreamImmediate(tag,rank,0u,own_sent));
				own_sent++;
			}
			if ( status != SPARK_STATUS_OK )
				return WeightStreamFail(stream,status);
		}
		status = WeightStreamPoll(stream,&call,WEIGHT_STREAM_FABRIC);
		if ( status == SPARK_STATUS_OK )
			status = WeightStreamPoll(stream,&call,WEIGHT_STREAM_PAIR);
		if ( status != SPARK_STATUS_OK )
			return WeightStreamFail(stream,status);
		if ( WeightStreamNowNs() - start > SPARK_WEIGHT_STREAM_TIMEOUT_NS )
		{
			fprintf(stderr,"weight_stream: rank %u rotation step %u (block of rank %u) stalled: sent %u done %u/%u + %u/%u got %u/%u + %u/%u ready 0x%x + 0x%x\n",rank,step,
				source,sent,call.done[WEIGHT_STREAM_FABRIC],expect_fabric_done,call.done[WEIGHT_STREAM_PAIR],expect_pair_done,call.got[WEIGHT_STREAM_FABRIC],receive_fabric,
				call.got[WEIGHT_STREAM_PAIR],receive_pair,call.ready[WEIGHT_STREAM_FABRIC],call.ready[WEIGHT_STREAM_PAIR]);
			return WeightStreamFail(stream,SPARK_STATUS_IO_ERROR);
		}
	}
	block->source = source;
	block->reserved0 = 0u;
	if ( source == rank )
	{
		block->half_offset[0] = own_offset;
		block->half_offset[1] = own_offset + half;
	}
	else if ( source == partner )
	{
		block->half_offset[0] = landing_offset;
		block->half_offset[1] = partner_offset + half;
	}
	else
	{
		block->half_offset[0] = landing_offset;
		block->half_offset[1] = landing_offset + half;
	}
	return SPARK_STATUS_OK;
}
