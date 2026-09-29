#include "sparkpipe/spark_speculation_relay_link.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_error_site.h"

uint64_t SparkSpeculationRelayNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static SparkStatus SparkSpeculationRelayParseEndpoint(const char *endpoint,struct sockaddr_in *address)
{
	char host[64];
	const char *colon;
	char *end;
	unsigned long port;
	size_t length;
	if ( endpoint == 0 || address == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	colon = strrchr(endpoint,':');
	if ( colon == 0 || colon == endpoint || colon[1] == '\0' )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	length = (size_t)(colon - endpoint);
	if ( length >= sizeof(host) )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	memcpy(host,endpoint,length);
	host[length] = '\0';
	errno = 0;
	port = strtoul(colon + 1,&end,10);
	if ( errno != 0 || *end != '\0' || port > 65535u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	memset(address,0,sizeof(*address));
	address->sin_family = AF_INET;
	address->sin_port = htons((uint16_t)port);
	if ( inet_pton(AF_INET,host,&address->sin_addr) != 1 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayLinkOpen(SparkSpeculationRelayLink *link,const char *local_endpoint,const char *peer_endpoint)
{
	struct sockaddr_in local,peer;
	socklen_t option_length;
	int flags,buffer_bytes,effective;
	SparkStatus status;
	if ( link == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(link,0,sizeof(*link));
	link->socket_fd = -1;
	status = SparkSpeculationRelayParseEndpoint(local_endpoint,&local);
	if ( status == SPARK_STATUS_OK )
		status = SparkSpeculationRelayParseEndpoint(peer_endpoint,&peer);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( peer.sin_port == 0u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	link->socket_fd = socket(AF_INET,SOCK_DGRAM,0);
	if ( link->socket_fd < 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	flags = fcntl(link->socket_fd,F_GETFL,0);
	buffer_bytes = (int)SPARK_SPECULATION_RELAY_SOCKET_BUFFER_BYTES;
	(void)setsockopt(link->socket_fd,SOL_SOCKET,SO_SNDBUF,&buffer_bytes,sizeof(buffer_bytes));
	(void)setsockopt(link->socket_fd,SOL_SOCKET,SO_RCVBUF,&buffer_bytes,sizeof(buffer_bytes));
	option_length = sizeof(effective);
	if ( getsockopt(link->socket_fd,SOL_SOCKET,SO_SNDBUF,&effective,&option_length) == 0 && effective > 0 )
		link->send_buffer_bytes = (uint32_t)effective;
	option_length = sizeof(effective);
	if ( getsockopt(link->socket_fd,SOL_SOCKET,SO_RCVBUF,&effective,&option_length) == 0 && effective > 0 )
		link->receive_buffer_bytes = (uint32_t)effective;
	if ( flags < 0 || fcntl(link->socket_fd,F_SETFL,flags | O_NONBLOCK) != 0 ||
		bind(link->socket_fd,(const struct sockaddr *)&local,sizeof(local)) != 0 ||
		connect(link->socket_fd,(const struct sockaddr *)&peer,sizeof(peer)) != 0 )
	{
		close(link->socket_fd);
		link->socket_fd = -1;
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	memcpy(link->peer,&peer,sizeof(peer));
	link->peer_length = (uint32_t)sizeof(peer);
	return(SPARK_STATUS_OK);
}

void SparkSpeculationRelayLinkClose(SparkSpeculationRelayLink *link)
{
	if ( link == 0 || link->socket_fd < 0 )
		return;
	close(link->socket_fd);
	link->socket_fd = -1;
}

SparkStatus SparkSpeculationRelayLinkSend(SparkSpeculationRelayLink *link,const uint8_t *bytes,uint32_t length)
{
	ssize_t sent;
	if ( link == 0 || link->socket_fd < 0 || bytes == 0 || length == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sent = send(link->socket_fd,bytes,length,MSG_DONTWAIT);
	if ( sent != (ssize_t)length )
	{
		link->send_errors++;
		return(SPARK_STATUS_BUSY);
	}
	link->datagrams_sent++;
	link->bytes_sent += length;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayLinkReceive(SparkSpeculationRelayLink *link,uint8_t *buffer,uint32_t capacity,uint32_t *length_out)
{
	ssize_t received;
	if ( link == 0 || link->socket_fd < 0 || buffer == 0 || capacity == 0u || length_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*length_out = 0u;
	received = recv(link->socket_fd,buffer,capacity,MSG_DONTWAIT);
	if ( received < 0 )
	{
		if ( errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNREFUSED || errno == EINTR )
			return(SPARK_STATUS_NOT_FOUND);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	link->datagrams_received++;
	*length_out = (uint32_t)received;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayLinkSendTap(SparkSpeculationRelayLink *link,uint64_t fingerprint,const SparkSpeculationTapRecord *record,const uint8_t *payload,uint32_t record_bytes,uint32_t payload_max)
{
	uint8_t datagram[SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX];
	uint32_t offset,bytes,length;
	SparkStatus status;
	if ( link == 0 || record == 0 || payload == 0 || record_bytes == 0u || SparkSpeculationTapFragmentCount(record_bytes,payload_max) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (offset=0u; offset<record_bytes; offset+=bytes)
	{
		bytes = record_bytes - offset < payload_max ? record_bytes - offset : payload_max;
		status = SparkSpeculationTapEncodeFragment(record,fingerprint,payload,record_bytes,offset,bytes,datagram,sizeof(datagram),&length);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( SparkSpeculationRelayLinkSend(link,datagram,length) != SPARK_STATUS_OK )
		{
			link->tap_records_unsent++;
			return(SPARK_STATUS_BUSY);
		}
	}
	link->tap_records_sent++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayRemoteInitialize(SparkSpeculationRelayRemote *remote,SparkSpeculationRelayLink *link,uint64_t engine_generation,uint32_t vocab_size,uint64_t await_ns,uint32_t shadow)
{
	SparkStatus status;
	if ( remote == 0 || link == 0 || link->socket_fd < 0 || shadow > 1u || (shadow == 0u && await_ns == 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(remote,0,sizeof(*remote));
	status = SparkSpeculationRelayDraftInitialize(&remote->mailbox,engine_generation,vocab_size);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	remote->link = link;
	remote->await_ns = await_ns;
	remote->shadow = shadow;
	remote->history_sequence = UINT64_MAX;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkSpeculationRelayHistoryHas(const SparkSpeculationRelayRemote *remote,uint64_t position)
{
	uint64_t low = remote->history_next_position > SPARK_SPECULATION_RELAY_HISTORY_TOKENS ? remote->history_next_position - SPARK_SPECULATION_RELAY_HISTORY_TOKENS : 0u;
	if ( low < remote->history_first_position )
		low = remote->history_first_position;
	return(position >= low && position < remote->history_next_position ? 1u : 0u);
}

static void SparkSpeculationRelayShadowClose(SparkSpeculationRelayRemote *remote,SparkSpeculationRelayShadowSlot *slot,uint32_t accepted)
{
	slot->open = 0u;
	remote->shadow_rounds++;
	remote->shadow_proposed += slot->token_count;
	remote->shadow_accepted += accepted;
	remote->shadow_accepted_at[accepted]++;
}

static void SparkSpeculationRelayShadowScore(SparkSpeculationRelayRemote *remote,SparkSpeculationRelayShadowSlot *slot)
{
	uint64_t position;
	uint32_t index;
	if ( slot->open == 0u )
		return;
	if ( slot->sequence_id != remote->history_sequence )
	{
		slot->open = 0u;
		remote->shadow_evicted++;
		return;
	}
	for (index=0u; index<slot->token_count; index++)
	{
		position = slot->anchor_position + 1u + index;
		if ( position >= remote->history_next_position )
			return;
		if ( SparkSpeculationRelayHistoryHas(remote,position) == 0u )
		{
			slot->open = 0u;
			remote->shadow_evicted++;
			return;
		}
		if ( remote->history_tokens[position % SPARK_SPECULATION_RELAY_HISTORY_TOKENS] != slot->token_ids[index] )
		{
			SparkSpeculationRelayShadowClose(remote,slot,index);
			return;
		}
	}
	SparkSpeculationRelayShadowClose(remote,slot,slot->token_count);
}

static void SparkSpeculationRelayShadowScoreAll(SparkSpeculationRelayRemote *remote)
{
	uint32_t index;
	for (index=0u; index<SPARK_SPECULATION_RELAY_SHADOW_SLOTS; index++)
		SparkSpeculationRelayShadowScore(remote,&remote->shadow_slots[index]);
}

static void SparkSpeculationRelayShadowStore(SparkSpeculationRelayRemote *remote,const SparkSpeculationRelayFrame *ready)
{
	SparkSpeculationRelayShadowSlot *slot = &remote->shadow_slots[remote->shadow_next_slot++ % SPARK_SPECULATION_RELAY_SHADOW_SLOTS];
	if ( slot->open != 0u )
		remote->shadow_evicted++;
	memset(slot,0,sizeof(*slot));
	slot->sequence_id = ready->sequence_id;
	slot->anchor_position = ready->anchor_position;
	slot->token_count = ready->token_count;
	memcpy(slot->token_ids,ready->token_ids,(size_t)ready->token_count * sizeof(uint32_t));
	slot->open = 1u;
	remote->mailbox.ready_valid = 0u;
	if ( ready->token_count == 0u )
	{
		SparkSpeculationRelayShadowClose(remote,slot,0u);
		return;
	}
	SparkSpeculationRelayShadowScore(remote,slot);
}

SparkStatus SparkSpeculationRelayRemoteObserve(SparkSpeculationRelayRemote *remote,uint64_t sequence_id,uint64_t first_position,const uint32_t *token_ids,uint32_t token_count)
{
	uint64_t position;
	uint32_t index,slot;
	if ( remote == 0 || token_ids == 0 || token_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( sequence_id != remote->history_sequence || first_position > remote->history_next_position || first_position < remote->history_first_position )
	{
		remote->history_sequence = sequence_id;
		remote->history_first_position = first_position;
		remote->history_next_position = first_position;
		remote->history_sent_position = first_position;
	}
	for (index=0u; index<token_count; index++)
	{
		position = first_position + index;
		slot = (uint32_t)(position % SPARK_SPECULATION_RELAY_HISTORY_TOKENS);
		if ( position < remote->history_next_position )
		{
			if ( SparkSpeculationRelayHistoryHas(remote,position) != 0u && remote->history_tokens[slot] == token_ids[index] )
				continue;
			remote->history_next_position = position;
			if ( remote->history_sent_position > position )
				remote->history_sent_position = position;
			if ( remote->history_first_position > position )
				remote->history_first_position = position;
		}
		remote->history_tokens[slot] = token_ids[index];
		remote->history_next_position = position + 1u;
	}
	SparkSpeculationRelayShadowScoreAll(remote);
	return(SPARK_STATUS_OK);
}

static void SparkSpeculationRelayRecordRtt(SparkSpeculationRelayRemote *remote,uint64_t rtt_ns)
{
	uint64_t bucket = rtt_ns / SPARK_SPECULATION_RELAY_RTT_BUCKET_NS;
	if ( bucket >= SPARK_SPECULATION_RELAY_RTT_BUCKETS )
		bucket = SPARK_SPECULATION_RELAY_RTT_BUCKETS - 1u;
	remote->rtt_buckets[bucket]++;
	remote->rtt_count++;
	if ( rtt_ns > remote->rtt_max_ns )
		remote->rtt_max_ns = rtt_ns;
}

static uint32_t SparkSpeculationRelayDrain(SparkSpeculationRelayRemote *remote,uint32_t in_time)
{
	uint8_t buffer[SPARK_SPECULATION_TAP_FRAGMENT_BYTES_MAX];
	SparkSpeculationRelayFrame frame;
	uint32_t length;
	uint32_t answered = 0u;
	while ( SparkSpeculationRelayLinkReceive(remote->link,buffer,sizeof(buffer),&length) == SPARK_STATUS_OK )
	{
		if ( length != SPARK_SPECULATION_RELAY_FRAME_BYTES || SparkSpeculationRelayDecode(buffer,length,remote->mailbox.vocab_size,&frame) != SPARK_STATUS_OK ||
			frame.kind != SPARK_SPECULATION_RELAY_KIND_DRAFT || SparkSpeculationRelayDraftDeliver(&remote->mailbox,buffer,length) != SPARK_STATUS_OK )
		{
			remote->link->datagrams_foreign++;
			continue;
		}
		if ( remote->mailbox.ready_valid == 0u )
		{
			if ( remote->shadow != 0u && frame.engine_generation == remote->mailbox.engine_generation && frame.round_id < remote->mailbox.round_id )
			{
				remote->answered_late++;
				SparkSpeculationRelayShadowStore(remote,&frame);
			}
			continue;
		}
		if ( remote->round_open == 0u )
		{
			remote->answered_late++;
			remote->mailbox.ready_valid = 0u;
			continue;
		}
		remote->round_open = 0u;
		answered = 1u;
		if ( in_time != 0u )
		{
			remote->answered_in_time++;
			SparkSpeculationRelayRecordRtt(remote,SparkSpeculationRelayNowNs() - remote->request_sent_ns);
		}
		else
			remote->answered_late++;
		if ( remote->shadow != 0u )
			SparkSpeculationRelayShadowStore(remote,&remote->mailbox.ready);
		else if ( in_time == 0u )
			remote->mailbox.ready_valid = 0u;
		break;
	}
	return(answered);
}

SparkStatus SparkSpeculationRelayRemotePoll(SparkSpeculationRelayRemote *remote)
{
	if ( remote == 0 || remote->link == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	while ( SparkSpeculationRelayDrain(remote,0u) != 0u )
		;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayRemoteRound(SparkSpeculationRelayRemote *remote,uint64_t sequence_id,uint64_t anchor_position,uint32_t requested_token_count,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationPolicyDraftRequest request;
	uint8_t bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES];
	uint32_t tokens[SPARK_SPECULATION_RELAY_MAX_TOKENS];
	uint64_t first,low,position,deadline;
	uint32_t count,answered;
	SparkStatus status;
	if ( remote == 0 || remote->link == 0 || result == 0 || requested_token_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( requested_token_count > SPARK_SPECULATION_RELAY_MAX_TOKENS )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	result->token_count = 0u;
	(void)SparkSpeculationRelayRemotePoll(remote);
	if ( remote->round_open != 0u )
	{
		remote->round_open = 0u;
		remote->deadline_misses += remote->shadow != 0u && remote->await_ns == 0u ? 1u : 0u;
	}
	if ( sequence_id != remote->history_sequence || remote->history_next_position != anchor_position + 1u )
	{
		remote->unsynced++;
		return(SPARK_STATUS_OK);
	}
	low = remote->history_next_position > SPARK_SPECULATION_RELAY_MAX_TOKENS ? remote->history_next_position - SPARK_SPECULATION_RELAY_MAX_TOKENS : 0u;
	first = remote->history_sent_position;
	if ( first < low )
		first = low;
	if ( first < remote->history_first_position )
		first = remote->history_first_position;
	if ( first > anchor_position )
		first = anchor_position;
	count = (uint32_t)(anchor_position + 1u - first);
	for (position=first; position<=anchor_position; position++)
		tokens[position - first] = remote->history_tokens[position % SPARK_SPECULATION_RELAY_HISTORY_TOKENS];
	status = SparkSpeculationRelayDraftIssue(&remote->mailbox,sequence_id,anchor_position,tokens[count - 1u],tokens,count,requested_token_count,bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	remote->request_sent_ns = SparkSpeculationRelayNowNs();
	if ( SparkSpeculationRelayLinkSend(remote->link,bytes,sizeof(bytes)) != SPARK_STATUS_OK )
	{
		remote->unsent++;
		return(SPARK_STATUS_OK);
	}
	remote->requests++;
	remote->history_sent_position = anchor_position + 1u;
	remote->round_open = 1u;
	answered = 0u;
	if ( remote->await_ns != 0u )
	{
		deadline = remote->request_sent_ns + remote->await_ns;
		do
			answered = SparkSpeculationRelayDrain(remote,1u);
		while ( answered == 0u && SparkSpeculationRelayNowNs() < deadline );
		if ( answered == 0u )
			remote->deadline_misses++;
	}
	if ( remote->shadow != 0u )
		return(SPARK_STATUS_OK);
	remote->round_open = 0u;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_SPECULATION_ABI_VERSION;
	request.descriptor_bytes = SPARK_SPECULATION_DRAFT_REQUEST_DESCRIPTOR_BYTES;
	request.requested_token_count = requested_token_count;
	request.sequence_id = sequence_id;
	request.sequence_position = anchor_position;
	SPARK_RETURN(SparkSpeculationRelayDraftTokens(&remote->mailbox,&request,result));
}

SparkStatus SparkSpeculationRelayRemoteDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	if ( context == 0 || request == 0 || result == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SPARK_RETURN(SparkSpeculationRelayRemoteRound((SparkSpeculationRelayRemote *)context,request->sequence_id,request->sequence_position,request->requested_token_count,result));
}

uint64_t SparkSpeculationRelayRemoteRttPercentileNs(const SparkSpeculationRelayRemote *remote,uint32_t per_mille)
{
	uint64_t target,seen = 0u;
	uint32_t bucket;
	if ( remote == 0 || remote->rtt_count == 0u || per_mille > 1000u )
		return(0u);
	target = (remote->rtt_count * per_mille + 999u) / 1000u;
	if ( target == 0u )
		target = 1u;
	for (bucket=0u; bucket<SPARK_SPECULATION_RELAY_RTT_BUCKETS; bucket++)
	{
		seen += remote->rtt_buckets[bucket];
		if ( seen >= target )
			return(bucket + 1u == SPARK_SPECULATION_RELAY_RTT_BUCKETS ? remote->rtt_max_ns : (uint64_t)(bucket + 1u) * SPARK_SPECULATION_RELAY_RTT_BUCKET_NS);
	}
	return(remote->rtt_max_ns);
}
