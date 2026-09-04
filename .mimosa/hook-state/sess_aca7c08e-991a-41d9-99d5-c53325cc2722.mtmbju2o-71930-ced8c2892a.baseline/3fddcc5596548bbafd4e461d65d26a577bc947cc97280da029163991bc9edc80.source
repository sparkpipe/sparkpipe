#pragma once


#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SPARK_NET_LISTEN_BACKLOG 64

static inline int32_t SparkNetParseU32(const char *text, uint32_t *value_out)
{
	uint64_t value;
	uint32_t index;
	if (text == 0 || text[0] == '\0' || value_out == 0)
		return -201;
	value = 0u;
	for (index = 0u; text[index] != '\0'; ++index)
	{
		if (text[index] < '0' || text[index] > '9')
			return -202;
		value = (value * 10u) + (uint32_t)(text[index] - '0');
		if (value > 0xffffffffull)
			return -203;
	}
	*value_out = (uint32_t)value;
	return 0;
}

static inline int32_t SparkNetSetNonblocking(int32_t fd)
{
	int32_t flags;
	flags = fcntl(fd,F_GETFL,0);
	if (flags < 0)
		return -211;
	if (fcntl(fd,F_SETFL,(flags | O_NONBLOCK)) < 0)
		return -212;
	return 0;
}


static inline int32_t SparkNetConfigureLowLatencyTcp(int32_t fd)
{
	int32_t enabled;

	if (fd < 0)
		return -231;
	enabled = 1;
	if (setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&enabled,sizeof(enabled)) < 0)
		return -232;
	if (setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&enabled,sizeof(enabled)) < 0)
		return -233;
	return 0;
}

static inline uint64_t SparkNetMonotonicNs(void)
{
	struct timespec timestamp;
	if (clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0)
		return 0u;
	return (((uint64_t)timestamp.tv_sec * 1000000000ull) +
		(uint64_t)timestamp.tv_nsec);
}

static inline int32_t SparkNetCreateListenSocket(const char *bind_address, uint32_t port)
{
	struct sockaddr_in address;
	int32_t fd,option;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if (fd < 0)
		return -221;
	option = 1;
	if (setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&option,sizeof(option)) < 0)
	{
		close(fd);
		return -222;
	}
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)port);
	if (inet_pton(AF_INET,bind_address,&address.sin_addr) != 1)
	{
		close(fd);
		return -223;
	}
	if (bind(fd,(struct sockaddr *)&address,sizeof(address)) < 0)
	{
		close(fd);
		return -224;
	}
	if (listen(fd,SPARK_NET_LISTEN_BACKLOG) < 0)
	{
		close(fd);
		return -225;
	}
	return fd;
}
