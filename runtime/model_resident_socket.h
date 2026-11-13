#pragma once

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include "sparkpipe/spark_status.h"

static SparkStatus SparkModelResidentConfigureSocket(int fd)
{
#if defined(__APPLE__)
	int enabled = 1;
	if ( setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&enabled,sizeof(enabled)) != 0 )
		return(SPARK_STATUS_IO_ERROR);
#else
	(void)fd;
#endif
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentConfigureTcp(int fd)
{
	int enabled = 1, idle = 10, interval = 5, count = 3;
#if defined(__APPLE__)
	const int idle_option = TCP_KEEPALIVE;
#else
	const int idle_option = TCP_KEEPIDLE;
#endif
	if ( setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&enabled,sizeof(enabled)) != 0 ||
		setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&enabled,sizeof(enabled)) != 0 ||
		setsockopt(fd,IPPROTO_TCP,idle_option,&idle,sizeof(idle)) != 0 ||
		setsockopt(fd,IPPROTO_TCP,TCP_KEEPINTVL,&interval,sizeof(interval)) != 0 ||
		setsockopt(fd,IPPROTO_TCP,TCP_KEEPCNT,&count,sizeof(count)) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}
