#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define HOST_STAGED_MAGIC UINT64_C(0x3150435448535053)
#define HOST_STAGED_POSTED_MAX 256u
#define HOST_STAGED_COMPLETIONS_MAX 256u
#define HOST_STAGED_CONNECT_TIMEOUT_NS UINT64_C(120000000000)
#define HOST_STAGED_PORT_SPAN 16u
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

typedef struct HostStagedHeader
{
	uint64_t magic;
	uint64_t route_identifier;
	uint64_t sequence_id;
	uint64_t token_index;
	uint32_t active_sequence_count;
	uint32_t reserved;
	uint64_t hidden_bytes;
	uint64_t sideband_bytes;
} HostStagedHeader;

typedef struct HostStagedFrame
{
	HostStagedHeader header;
	uint8_t *payload;
	struct HostStagedFrame *next;
} HostStagedFrame;

typedef struct HostStagedState
{
	uint32_t input;
	uint32_t stop;
	uint32_t thread_started;
	int32_t listen_fd;
	int32_t connection_fd;
	int32_t send_fd;
	int32_t wake[2];
	uint16_t peer_port;
	char peer_host[256];
	uint64_t route_identifier;
	uint64_t max_packet_bytes;
	pthread_t thread;
	pthread_mutex_t lock;
	HostStagedFrame *frames;
	SparkHiddenTransportPacket posted[HOST_STAGED_POSTED_MAX];
	uint32_t posted_count;
	SparkHiddenTransportCompletion completions[HOST_STAGED_COMPLETIONS_MAX];
	uint32_t completion_head;
	uint32_t completion_count;
	uint8_t *staging;
	uint64_t staging_bytes;
} HostStagedState;

static int32_t host_staged_descriptor_flags(int32_t fd,uint32_t nonblocking)
{
	int32_t flags;
	if ( fd < 0 )
		return(fd);
	(void)fcntl(fd,F_SETFD,FD_CLOEXEC);
	if ( nonblocking != 0u )
	{
		flags = fcntl(fd,F_GETFL,0);
		(void)fcntl(fd,F_SETFL,flags | O_NONBLOCK);
	}
	return(fd);
}

static uint64_t host_staged_now_ns(void)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static void host_staged_signal(HostStagedState *state)
{
	uint8_t byte = 1u;
	ssize_t written = write(state->wake[1],&byte,1u);
	(void)written;
}

static void host_staged_drain(HostStagedState *state)
{
	uint8_t wake_tokens[64];
	while ( read(state->wake[0],wake_tokens,sizeof(wake_tokens)) > 0 )
		;
}

static uint32_t host_staged_wait_readable(HostStagedState *state,int32_t fd)
{
	struct pollfd descriptor;
	while ( state->stop == 0u )
	{
		descriptor.fd = fd;
		descriptor.events = POLLIN;
		descriptor.revents = 0;
		if ( poll(&descriptor,1u,200) > 0 )
			return(1u);
	}
	return(0u);
}

static SparkStatus host_staged_read_all(HostStagedState *state,int32_t fd,void *buffer,uint64_t bytes)
{
	uint8_t *cursor = buffer;
	ssize_t moved;
	while ( bytes > 0u )
	{
		if ( host_staged_wait_readable(state,fd) == 0u )
			return(SPARK_STATUS_IO_ERROR);
		moved = recv(fd,cursor,(size_t)bytes,0);
		if ( moved < 0 && errno == EINTR )
			continue;
		if ( moved <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		cursor += moved;
		bytes -= (uint64_t)moved;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus host_staged_write_all(int32_t fd,const void *buffer,uint64_t bytes)
{
	const uint8_t *cursor = buffer;
	ssize_t moved;
	while ( bytes > 0u )
	{
		moved = send(fd,cursor,(size_t)bytes,MSG_NOSIGNAL);
		if ( moved < 0 && errno == EINTR )
			continue;
		if ( moved <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		cursor += moved;
		bytes -= (uint64_t)moved;
	}
	return(SPARK_STATUS_OK);
}

static void *host_staged_receiver(void *argument)
{
	HostStagedState *state = argument;
	HostStagedFrame *frame,**tail;
	while ( state->stop == 0u )
	{
		int32_t fd,one = 1;
		if ( host_staged_wait_readable(state,state->listen_fd) == 0u )
			break;
		fd = accept(state->listen_fd,0,0);
		if ( fd < 0 )
		{
			if ( errno == EINTR )
				continue;
			break;
		}
		(void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
		pthread_mutex_lock(&state->lock);
		state->connection_fd = fd;
		pthread_mutex_unlock(&state->lock);
		while ( state->stop == 0u )
		{
			frame = calloc(1u,sizeof(*frame));
			if ( frame == 0 )
				break;
			if ( host_staged_read_all(state,fd,&frame->header,sizeof(frame->header)) != SPARK_STATUS_OK ||
				frame->header.magic != HOST_STAGED_MAGIC ||
				frame->header.route_identifier != state->route_identifier ||
				frame->header.hidden_bytes > state->max_packet_bytes ||
				frame->header.sideband_bytes > state->max_packet_bytes - frame->header.hidden_bytes )
			{
				if ( frame->header.magic != 0u )
					fprintf(stderr,"host_staged_tcp rejected frame magic=%llx route=%llx expected_route=%llx hidden=%llu sideband=%llu\n",(unsigned long long)frame->header.magic,(unsigned long long)frame->header.route_identifier,(unsigned long long)state->route_identifier,(unsigned long long)frame->header.hidden_bytes,(unsigned long long)frame->header.sideband_bytes);
				free(frame);
				break;
			}
			frame->payload = malloc((size_t)(frame->header.hidden_bytes + frame->header.sideband_bytes + 1u));
			if ( frame->payload == 0 || host_staged_read_all(state,fd,frame->payload,frame->header.hidden_bytes + frame->header.sideband_bytes) != SPARK_STATUS_OK )
			{
				free(frame->payload);
				free(frame);
				break;
			}
			pthread_mutex_lock(&state->lock);
			tail = &state->frames;
			while ( *tail != 0 )
				tail = &(*tail)->next;
			*tail = frame;
			pthread_mutex_unlock(&state->lock);
			host_staged_signal(state);
		}
		pthread_mutex_lock(&state->lock);
		state->connection_fd = -1;
		pthread_mutex_unlock(&state->lock);
		(void)close(fd);
	}
	return(0);
}

static SparkStatus host_staged_listen(HostStagedState *state,uint16_t port)
{
	struct sockaddr_in address;
	int32_t one = 1;
	state->listen_fd = host_staged_descriptor_flags(socket(AF_INET,SOCK_STREAM,0),0u);
	if ( state->listen_fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	(void)setsockopt(state->listen_fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_ANY);
	address.sin_port = htons(port);
	if ( bind(state->listen_fd,(struct sockaddr *)&address,sizeof(address)) != 0 || listen(state->listen_fd,4) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( pthread_create(&state->thread,0,host_staged_receiver,state) != 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->thread_started = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus host_staged_connect(HostStagedState *state)
{
	struct addrinfo hints,*addresses = 0,*address;
	char port[16];
	uint64_t deadline = host_staged_now_ns() + HOST_STAGED_CONNECT_TIMEOUT_NS;
	int32_t one = 1;
	if ( state->send_fd >= 0 )
		return(SPARK_STATUS_OK);
	(void)snprintf(port,sizeof(port),"%u",(unsigned)state->peer_port);
	memset(&hints,0,sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	if ( getaddrinfo(state->peer_host,port,&hints,&addresses) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	while ( state->send_fd < 0 && host_staged_now_ns() < deadline )
	{
		for (address = addresses; address != 0 && state->send_fd < 0; address = address->ai_next)
		{
			int32_t fd = host_staged_descriptor_flags(socket(address->ai_family,address->ai_socktype,address->ai_protocol),0u);
			if ( fd < 0 )
				continue;
			if ( connect(fd,address->ai_addr,address->ai_addrlen) == 0 )
			{
				(void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
				state->send_fd = fd;
			}
			else
				(void)close(fd);
		}
		if ( state->send_fd < 0 )
		{
			struct timespec pause = {0,100000000};
			(void)nanosleep(&pause,0);
		}
	}
	freeaddrinfo(addresses);
	return(state->send_fd >= 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static void host_staged_complete(HostStagedState *state,const SparkHiddenTransportPacket *packet,SparkStatus status)
{
	SparkHiddenTransportCompletion *completion;
	uint32_t slot;
	if ( state->completion_count >= HOST_STAGED_COMPLETIONS_MAX )
		return;
	slot = (state->completion_head + state->completion_count) % HOST_STAGED_COMPLETIONS_MAX;
	completion = &state->completions[slot];
	memset(completion,0,sizeof(*completion));
	completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
	completion->status = status;
	completion->active_sequence_count = packet->active_sequence_count;
	completion->sequence_id = packet->sequence_id;
	completion->token_index = packet->token_index;
	completion->transfer_bytes = (uint64_t)packet->active_sequence_count * ((uint64_t)packet->bytes_per_sequence + packet->sideband_bytes_per_sequence);
	state->completion_count++;
}

static void host_staged_match(HostStagedState *state)
{
	HostStagedFrame **link,*frame;
	uint32_t index;
	for (index = 0u; index < state->posted_count; )
	{
		SparkHiddenTransportPacket *packet = &state->posted[index];
		uint64_t hidden = (uint64_t)packet->active_sequence_count * packet->bytes_per_sequence;
		uint64_t sideband = (uint64_t)packet->active_sequence_count * packet->sideband_bytes_per_sequence;
		SparkStatus status = SPARK_STATUS_OK;
		for (link = &state->frames; *link != 0; link = &(*link)->next)
			if ( (*link)->header.sequence_id == packet->sequence_id && (*link)->header.token_index == packet->token_index )
				break;
		if ( *link == 0 || state->completion_count >= HOST_STAGED_COMPLETIONS_MAX )
		{
			index++;
			continue;
		}
		frame = *link;
		*link = frame->next;
		if ( frame->header.active_sequence_count != packet->active_sequence_count || frame->header.hidden_bytes != hidden || frame->header.sideband_bytes != sideband )
			status = SPARK_STATUS_VALIDATION_FAILED;
		if ( status == SPARK_STATUS_OK && cudaMemcpy((void *)packet->hidden_bf16,frame->payload,(size_t)hidden,cudaMemcpyDefault) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK && sideband != 0u && cudaMemcpy((void *)packet->sideband_payload,frame->payload + hidden,(size_t)sideband,cudaMemcpyDefault) != cudaSuccess )
			status = SPARK_STATUS_IO_ERROR;
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"host_staged_tcp receive failed status=%d cuda=%d hidden=%llu sideband=%llu frame_hidden=%llu frame_sideband=%llu rows=%u frame_rows=%u\n",(int)status,(int)cudaGetLastError(),(unsigned long long)hidden,(unsigned long long)sideband,(unsigned long long)frame->header.hidden_bytes,(unsigned long long)frame->header.sideband_bytes,packet->active_sequence_count,frame->header.active_sequence_count);
		host_staged_complete(state,packet,status);
		free(frame->payload);
		free(frame);
		state->posted[index] = state->posted[state->posted_count - 1u];
		state->posted_count--;
	}
}

static SparkStatus HostStagedInitialize(const SparkHiddenTransportEndpoint *endpoint,void **transport_state)
{
	HostStagedState *state;
	SparkStatus status;
	const char *host;
	uint32_t rank;
	if ( endpoint == 0 || transport_state == 0 || endpoint->control_port_base == 0u || endpoint->control_port_base > 65535u - HOST_STAGED_PORT_SPAN )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*transport_state = 0;
	state = calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->listen_fd = -1;
	state->connection_fd = -1;
	state->send_fd = -1;
	state->wake[0] = -1;
	state->wake[1] = -1;
	state->route_identifier = endpoint->route_identifier;
	state->max_packet_bytes = endpoint->max_packet_bytes;
	state->input = endpoint->sink_rank_index == endpoint->local_rank_index ? 1u : 0u;
	pthread_mutex_init(&state->lock,0);
	if ( pipe(state->wake) != 0 )
	{
		free(state);
		return(SPARK_STATUS_IO_ERROR);
	}
	(void)host_staged_descriptor_flags(state->wake[0],1u);
	(void)host_staged_descriptor_flags(state->wake[1],1u);
	rank = state->input != 0u ? endpoint->local_rank_index : endpoint->sink_rank_index;
	host = endpoint->sink_host;
	if ( host == 0 || strlen(host) >= sizeof(state->peer_host) )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	else
	{
		(void)snprintf(state->peer_host,sizeof(state->peer_host),"%s",host);
		state->peer_port = (uint16_t)(endpoint->control_port_base + (rank % HOST_STAGED_PORT_SPAN));
		status = state->input != 0u ? host_staged_listen(state,state->peer_port) : SPARK_STATUS_OK;
	}
	if ( status != SPARK_STATUS_OK )
	{
		if ( state->listen_fd >= 0 )
			(void)close(state->listen_fd);
		(void)close(state->wake[0]);
		(void)close(state->wake[1]);
		free(state);
		return(status);
	}
	*transport_state = state;
	return(SPARK_STATUS_OK);
}

static void HostStagedDestroy(void *transport_state)
{
	HostStagedState *state = transport_state;
	HostStagedFrame *frame;
	if ( state == 0 )
		return;
	state->stop = 1u;
	if ( state->listen_fd >= 0 )
		(void)shutdown(state->listen_fd,SHUT_RDWR);
	pthread_mutex_lock(&state->lock);
	if ( state->connection_fd >= 0 )
		(void)shutdown(state->connection_fd,SHUT_RDWR);
	pthread_mutex_unlock(&state->lock);
	if ( state->thread_started != 0u )
		pthread_join(state->thread,0);
	if ( state->listen_fd >= 0 )
		(void)close(state->listen_fd);
	if ( state->send_fd >= 0 )
		(void)close(state->send_fd);
	while ( (frame = state->frames) != 0 )
	{
		state->frames = frame->next;
		free(frame->payload);
		free(frame);
	}
	(void)close(state->wake[0]);
	(void)close(state->wake[1]);
	free(state->staging);
	pthread_mutex_destroy(&state->lock);
	free(state);
}

static SparkStatus HostStagedPostReceive(void *transport_state,SparkHiddenTransportPacket *packet)
{
	HostStagedState *state = transport_state;
	if ( state == 0 || packet == 0 || state->input == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pthread_mutex_lock(&state->lock);
	if ( state->posted_count >= HOST_STAGED_POSTED_MAX )
	{
		pthread_mutex_unlock(&state->lock);
		return(SPARK_STATUS_BUSY);
	}
	state->posted[state->posted_count++] = *packet;
	pthread_mutex_unlock(&state->lock);
	host_staged_signal(state);
	return(SPARK_STATUS_OK);
}

static SparkStatus HostStagedSend(void *transport_state,const SparkHiddenTransportPacket *packet)
{
	HostStagedState *state = transport_state;
	HostStagedHeader header;
	uint64_t hidden,sideband;
	SparkStatus status;
	if ( state == 0 || packet == 0 || state->input != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	hidden = (uint64_t)packet->active_sequence_count * packet->bytes_per_sequence;
	sideband = (uint64_t)packet->active_sequence_count * packet->sideband_bytes_per_sequence;
	if ( hidden + sideband > state->max_packet_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->staging_bytes < hidden + sideband )
	{
		uint8_t *grown = realloc(state->staging,(size_t)state->max_packet_bytes);
		if ( grown == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->staging = grown;
		state->staging_bytes = state->max_packet_bytes;
	}
	if ( packet->cuda_stream != 0 && cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	if ( cudaMemcpy(state->staging,packet->hidden_bf16,(size_t)hidden,cudaMemcpyDefault) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	if ( sideband != 0u && cudaMemcpy(state->staging + hidden,packet->sideband_payload,(size_t)sideband,cudaMemcpyDefault) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	status = host_staged_connect(state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&header,0,sizeof(header));
	header.magic = HOST_STAGED_MAGIC;
	header.route_identifier = state->route_identifier;
	header.sequence_id = packet->sequence_id;
	header.token_index = packet->token_index;
	header.active_sequence_count = packet->active_sequence_count;
	header.hidden_bytes = hidden;
	header.sideband_bytes = sideband;
	status = host_staged_write_all(state->send_fd,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = host_staged_write_all(state->send_fd,state->staging,hidden + sideband);
	if ( status != SPARK_STATUS_OK )
	{
		(void)close(state->send_fd);
		state->send_fd = -1;
		return(status);
	}
	pthread_mutex_lock(&state->lock);
	host_staged_complete(state,packet,SPARK_STATUS_OK);
	pthread_mutex_unlock(&state->lock);
	host_staged_signal(state);
	return(SPARK_STATUS_OK);
}

static SparkStatus HostStagedPoll(void *transport_state,SparkHiddenTransportCompletion *completion)
{
	HostStagedState *state = transport_state;
	if ( state == 0 || completion == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	host_staged_drain(state);
	pthread_mutex_lock(&state->lock);
	if ( state->input != 0u )
		host_staged_match(state);
	if ( state->completion_count == 0u )
	{
		pthread_mutex_unlock(&state->lock);
		memset(completion,0,sizeof(*completion));
		completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
		completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
		completion->status = SPARK_STATUS_BUSY;
		return(SPARK_STATUS_OK);
	}
	*completion = state->completions[state->completion_head];
	state->completion_head = (state->completion_head + 1u) % HOST_STAGED_COMPLETIONS_MAX;
	state->completion_count--;
	if ( state->completion_count != 0u )
		host_staged_signal(state);
	pthread_mutex_unlock(&state->lock);
	return(SPARK_STATUS_OK);
}

static SparkStatus HostStagedGetPollDescriptors(void *transport_state,SparkHiddenTransportPollDescriptor *descriptors,uint32_t descriptor_capacity,uint32_t *descriptor_count)
{
	HostStagedState *state = transport_state;
	if ( state == 0 || descriptor_count == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*descriptor_count = 0u;
	if ( descriptor_capacity == 0u || descriptors == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&descriptors[0],0,sizeof(descriptors[0]));
	descriptors[0].abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	descriptors[0].descriptor_bytes = SPARK_HIDDEN_TRANSPORT_POLL_DESCRIPTOR_BYTES;
	descriptors[0].fd = state->wake[0];
	descriptors[0].events = SPARK_HIDDEN_TRANSPORT_POLL_READ;
	*descriptor_count = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus HostStagedPostReceiveBatch(void *transport_state,SparkHiddenTransportPacket *packets,uint32_t packet_count)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	for (index = 0u; index < packet_count && status == SPARK_STATUS_OK; index++)
		status = HostStagedPostReceive(transport_state,&packets[index]);
	return(status);
}

static SparkStatus HostStagedSendBatch(void *transport_state,const SparkHiddenTransportPacket *packets,uint32_t packet_count)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	for (index = 0u; index < packet_count && status == SPARK_STATUS_OK; index++)
		status = HostStagedSend(transport_state,&packets[index]);
	return(status);
}

static SparkHiddenTransportInterface host_staged_interface;

const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
	memset(&host_staged_interface,0,sizeof(host_staged_interface));
	host_staged_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	host_staged_interface.descriptor_bytes = sizeof(SparkHiddenTransportInterface);
	host_staged_interface.capability_flags = SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PIPELINE_HOST_STAGED_CAPS | SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS;
	host_staged_interface.initialize = HostStagedInitialize;
	host_staged_interface.destroy = HostStagedDestroy;
	host_staged_interface.post_receive = HostStagedPostReceive;
	host_staged_interface.send = HostStagedSend;
	host_staged_interface.poll = HostStagedPoll;
	host_staged_interface.post_receive_batch = HostStagedPostReceiveBatch;
	host_staged_interface.send_batch = HostStagedSendBatch;
	host_staged_interface.get_poll_descriptors = HostStagedGetPollDescriptors;
	return(&host_staged_interface);
}
