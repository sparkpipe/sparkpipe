#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "inference/runner/device_collective.h"

#define TCP_COLLECTIVE_MAGIC 0x4c4f4354u
#define TCP_COLLECTIVE_CONNECT_SECONDS 600u

typedef struct TcpMessage
{
	uint32_t magic;
	uint32_t rank;
	uint64_t round;
	uint64_t bytes;
} TcpMessage;

typedef struct TcpPeer
{
	int fd;
	TcpMessage header_out;
	TcpMessage header_in;
	uint64_t sent;
	uint64_t received;
} TcpPeer;

typedef struct TcpCollective
{
	uint32_t degree;
	uint32_t rank;
	uint32_t failed;
	uint64_t round;
	TcpPeer peers[SPARK_DEVICE_COLLECTIVE_MAX_RANKS];
	SparkDeviceBuffer out;
	SparkDeviceBuffer in;
} TcpCollective;

static SparkStatus TcpRefuse(TcpCollective *tcp, const char *what)
{
	fprintf(stderr,"device_collective_tcp: rank %u: %s (%s)\n",tcp->rank,what,strerror(errno));
	tcp->failed = 1u;
	return(SPARK_STATUS_IO_ERROR);
}

static int TcpTune(int fd)
{
	const int one = 1, buffer = 8 << 20;
	(void)setsockopt(fd,SOL_SOCKET,SO_SNDBUF,&buffer,sizeof(buffer));
	(void)setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&buffer,sizeof(buffer));
	return(setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one)));
}

static SparkStatus TcpWriteAll(int fd, const void *data, size_t bytes)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while ( bytes != 0u )
	{
		const ssize_t wrote = send(fd,cursor,bytes,0);
		if ( wrote <= 0 && errno != EINTR )
			return(SPARK_STATUS_IO_ERROR);
		if ( wrote > 0 )
		{
			cursor += wrote;
			bytes -= (size_t)wrote;
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus TcpReadAll(int fd, void *data, size_t bytes)
{
	uint8_t *cursor = (uint8_t *)data;
	while ( bytes != 0u )
	{
		const ssize_t got = recv(fd,cursor,bytes,0);
		if ( got <= 0 && errno != EINTR )
			return(SPARK_STATUS_IO_ERROR);
		if ( got > 0 )
		{
			cursor += got;
			bytes -= (size_t)got;
		}
	}
	return(SPARK_STATUS_OK);
}

static int TcpConnect(const char *host, uint16_t port)
{
	struct addrinfo hints, *found = 0, *entry;
	char service[16];
	int fd = -1;
	const time_t deadline = time(0) + TCP_COLLECTIVE_CONNECT_SECONDS;
	memset(&hints,0,sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	snprintf(service,sizeof(service),"%u",(unsigned)port);
	while ( fd < 0 && time(0) < deadline )
	{
		if ( getaddrinfo(host,service,&hints,&found) == 0 )
		{
			for ( entry = found; entry != 0 && fd < 0; entry = entry->ai_next )
			{
				fd = socket(entry->ai_family,entry->ai_socktype,entry->ai_protocol);
				if ( fd >= 0 && connect(fd,entry->ai_addr,entry->ai_addrlen) != 0 )
				{
					close(fd);
					fd = -1;
				}
			}
			freeaddrinfo(found);
			found = 0;
		}
		if ( fd < 0 )
			usleep(200000);
	}
	return(fd);
}

static SparkStatus TcpStaging(TcpCollective *tcp, SparkDeviceBuffer *buffer, uint64_t bytes, const char *owner)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( buffer->handle != 0u && buffer->bytes >= bytes )
		return(SPARK_STATUS_OK);
	if ( buffer->handle != 0u )
		status = SparkDeviceFree(buffer);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAllocate(buffer,SPARK_DEVICE_SPACE_SHARED_COHERENT,(bytes + 0xfffffu) & ~(uint64_t)0xfffffu,owner);
	if ( status != SPARK_STATUS_OK )
		tcp->failed = 1u;
	return(status);
}

static SparkStatus TcpMove(TcpCollective *tcp, uint64_t stride, uint64_t bytes)
{
	struct pollfd polls[SPARK_DEVICE_COLLECTIVE_MAX_RANKS];
	uint32_t peer, pending = 0u, count;
	for ( peer = 0u; peer < tcp->degree; peer++ )
	{
		TcpPeer *link = &tcp->peers[peer];
		if ( peer == tcp->rank )
			continue;
		link->header_out.magic = TCP_COLLECTIVE_MAGIC;
		link->header_out.rank = tcp->rank;
		link->header_out.round = tcp->round;
		link->header_out.bytes = bytes;
		link->sent = 0u;
		link->received = 0u;
		pending++;
	}
	while ( pending != 0u )
	{
		uint32_t map[SPARK_DEVICE_COLLECTIVE_MAX_RANKS];
		count = 0u;
		for ( peer = 0u; peer < tcp->degree; peer++ )
		{
			const TcpPeer *link = &tcp->peers[peer];
			const uint64_t total = sizeof(TcpMessage) + bytes;
			if ( peer == tcp->rank || (link->sent == total && link->received == total) )
				continue;
			polls[count].fd = link->fd;
			polls[count].events = (short)((link->sent < total ? POLLOUT : 0) | (link->received < total ? POLLIN : 0));
			polls[count].revents = 0;
			map[count++] = peer;
		}
		if ( poll(polls,count,-1) < 0 )
		{
			if ( errno == EINTR )
				continue;
			return(TcpRefuse(tcp,"poll failed"));
		}
		pending = 0u;
		for ( uint32_t index = 0u; index < count; index++ )
		{
			TcpPeer *link = &tcp->peers[map[index]];
			const uint64_t total = sizeof(TcpMessage) + bytes;
			if ( (polls[index].revents & (POLLERR | POLLNVAL)) != 0 )
				return(TcpRefuse(tcp,"a peer connection failed"));
			if ( (polls[index].revents & POLLOUT) != 0 && link->sent < total )
			{
				const uint8_t *data = link->sent < sizeof(TcpMessage) ? (const uint8_t *)&link->header_out + link->sent :
					(const uint8_t *)tcp->out.host + (stride != 0u ? (uint64_t)map[index] * stride : 0u) + (link->sent - sizeof(TcpMessage));
				const uint64_t left = link->sent < sizeof(TcpMessage) ? sizeof(TcpMessage) - link->sent : total - link->sent;
				const ssize_t wrote = send(link->fd,data,(size_t)left,0);
				if ( wrote < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR )
					return(TcpRefuse(tcp,"send to a peer failed"));
				if ( wrote > 0 )
					link->sent += (uint64_t)wrote;
			}
			if ( (polls[index].revents & (POLLIN | POLLHUP)) != 0 && link->received < total )
			{
				uint8_t *data = link->received < sizeof(TcpMessage) ? (uint8_t *)&link->header_in + link->received :
					(uint8_t *)tcp->in.host + (uint64_t)map[index] * bytes + (link->received - sizeof(TcpMessage));
				const uint64_t left = link->received < sizeof(TcpMessage) ? sizeof(TcpMessage) - link->received : total - link->received;
				const ssize_t got = recv(link->fd,data,(size_t)left,0);
				if ( got == 0 || (got < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) )
					return(TcpRefuse(tcp,"a peer closed or failed while receiving"));
				if ( got > 0 )
					link->received += (uint64_t)got;
				if ( link->received >= sizeof(TcpMessage) && (link->header_in.magic != TCP_COLLECTIVE_MAGIC || link->header_in.rank != map[index] ||
					link->header_in.round != tcp->round || link->header_in.bytes != bytes) )
				{
					errno = EPROTO;
					return(TcpRefuse(tcp,"a peer is at another round or moves another size; the ranks diverged"));
				}
			}
		}
		for ( peer = 0u; peer < tcp->degree; peer++ )
			if ( peer != tcp->rank && (tcp->peers[peer].sent < sizeof(TcpMessage) + bytes || tcp->peers[peer].received < sizeof(TcpMessage) + bytes) )
				pending++;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus TcpExchange(void *context, uint32_t rank, SparkDeviceQueue *queue, SparkDeviceRange local, uint64_t source_stride, uint64_t bytes,
	SparkDeviceRange lanes)
{
	TcpCollective *tcp = (TcpCollective *)context;
	const uint64_t span = source_stride != 0u ? (uint64_t)(tcp->degree - 1u) * source_stride + bytes : bytes;
	SparkDeviceRange staged;
	SparkStatus status;
	if ( tcp->failed != 0u || rank != tcp->rank || bytes == 0u )
		return(SPARK_STATUS_IO_ERROR);
	status = TcpStaging(tcp,&tcp->out,span,"tcp collective send staging");
	if ( status == SPARK_STATUS_OK )
		status = TcpStaging(tcp,&tcp->in,(uint64_t)tcp->degree * bytes,"tcp collective receive staging");
	staged.buffer = &tcp->out;
	staged.offset = 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceCopy(queue,staged,local,span);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status != SPARK_STATUS_OK )
	{
		tcp->failed = 1u;
		return(status);
	}
	memcpy((uint8_t *)tcp->in.host + (uint64_t)rank * bytes,(const uint8_t *)tcp->out.host + (source_stride != 0u ? (uint64_t)rank * source_stride : 0u),
		(size_t)bytes);
	status = TcpMove(tcp,source_stride,bytes);
	tcp->round++;
	staged.buffer = &tcp->in;
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceCopy(queue,lanes,staged,(uint64_t)tcp->degree * bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status != SPARK_STATUS_OK )
		tcp->failed = 1u;
	return(status);
}

static void TcpAbort(void *context)
{
	TcpCollective *tcp = (TcpCollective *)context;
	uint32_t peer;
	tcp->failed = 1u;
	for ( peer = 0u; peer < tcp->degree; peer++ )
		if ( tcp->peers[peer].fd >= 0 )
			shutdown(tcp->peers[peer].fd,SHUT_RDWR);
}

SparkStatus SparkDeviceCollectiveTcpCreate(SparkDeviceCollective *collective, uint32_t degree, uint32_t rank, const char *const *hosts,
	uint16_t base_port)
{
	struct sockaddr_in6 address;
	const int one = 1;
	TcpCollective *tcp;
	int listener = -1;
	uint32_t peer, accepted;
	SparkStatus status = SPARK_STATUS_OK;
	memset(collective,0,sizeof(*collective));
	if ( degree < 2u || degree > SPARK_DEVICE_COLLECTIVE_MAX_RANKS || rank >= degree || hosts == 0 || base_port == 0u ||
		(uint32_t)base_port + degree > 65535u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	tcp = (TcpCollective *)calloc(1u,sizeof(*tcp));
	if ( tcp == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	tcp->degree = degree;
	tcp->rank = rank;
	for ( peer = 0u; peer < SPARK_DEVICE_COLLECTIVE_MAX_RANKS; peer++ )
		tcp->peers[peer].fd = -1;
	if ( rank + 1u < degree )
	{
		memset(&address,0,sizeof(address));
		address.sin6_family = AF_INET6;
		address.sin6_addr = in6addr_any;
		address.sin6_port = htons((uint16_t)(base_port + rank));
		listener = socket(AF_INET6,SOCK_STREAM,0);
		if ( listener < 0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one)) != 0 ||
			bind(listener,(const struct sockaddr *)&address,sizeof(address)) != 0 || listen(listener,(int)degree) != 0 )
			status = TcpRefuse(tcp,"cannot listen for the higher ranks");
	}
	for ( peer = 0u; status == SPARK_STATUS_OK && peer < rank; peer++ )
	{
		TcpMessage hello;
		tcp->peers[peer].fd = TcpConnect(hosts[peer],(uint16_t)(base_port + peer));
		memset(&hello,0,sizeof(hello));
		hello.magic = TCP_COLLECTIVE_MAGIC;
		hello.rank = rank;
		if ( tcp->peers[peer].fd < 0 || TcpTune(tcp->peers[peer].fd) != 0 || TcpWriteAll(tcp->peers[peer].fd,&hello,sizeof(hello)) != SPARK_STATUS_OK )
			status = TcpRefuse(tcp,"cannot reach a lower rank");
	}
	for ( accepted = rank + 1u; status == SPARK_STATUS_OK && accepted < degree; accepted++ )
	{
		TcpMessage hello;
		const int fd = accept(listener,0,0);
		if ( fd < 0 || TcpTune(fd) != 0 || TcpReadAll(fd,&hello,sizeof(hello)) != SPARK_STATUS_OK || hello.magic != TCP_COLLECTIVE_MAGIC ||
			hello.rank <= rank || hello.rank >= degree || tcp->peers[hello.rank].fd >= 0 )
		{
			if ( fd >= 0 )
				close(fd);
			status = TcpRefuse(tcp,"a higher rank's hello is wrong");
		}
		else
			tcp->peers[hello.rank].fd = fd;
	}
	if ( listener >= 0 )
		close(listener);
	for ( peer = 0u; status == SPARK_STATUS_OK && peer < degree; peer++ )
		if ( peer != rank && fcntl(tcp->peers[peer].fd,F_SETFL,fcntl(tcp->peers[peer].fd,F_GETFL) | O_NONBLOCK) != 0 )
			status = TcpRefuse(tcp,"cannot make a peer connection non-blocking");
	collective->context = tcp;
	collective->degree = degree;
	collective->exchange = TcpExchange;
	collective->abort = TcpAbort;
	if ( status != SPARK_STATUS_OK )
		SparkDeviceCollectiveTcpDestroy(collective);
	return(status);
}

void SparkDeviceCollectiveTcpDestroy(SparkDeviceCollective *collective)
{
	TcpCollective *tcp = (TcpCollective *)collective->context;
	uint32_t peer;
	if ( tcp != 0 )
	{
		for ( peer = 0u; peer < SPARK_DEVICE_COLLECTIVE_MAX_RANKS; peer++ )
			if ( tcp->peers[peer].fd >= 0 )
				close(tcp->peers[peer].fd);
		if ( tcp->out.handle != 0u )
			(void)SparkDeviceFree(&tcp->out);
		if ( tcp->in.handle != 0u )
			(void)SparkDeviceFree(&tcp->in);
		free(tcp);
	}
	memset(collective,0,sizeof(*collective));
}
