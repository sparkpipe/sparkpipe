#include "sparkpipe/spark_io.h"

#include <errno.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_error_site.h"

ssize_t SparkIoSend(int fd, const void *data, size_t bytes)
{
#if defined(MSG_NOSIGNAL)
	return(send(fd,data,bytes,MSG_NOSIGNAL));
#else
	return(send(fd,data,bytes,0));
#endif
}

SparkStatus SparkIoReadAll(int fd, void *data, size_t bytes)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t got = read(fd,(uint8_t *)data + done,bytes - done);
		if ( got < 0 && errno == EINTR )
			continue;
		if ( got <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)got;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkIoWriteAll(int fd, const void *data, size_t bytes)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t put = write(fd,(const uint8_t *)data + done,bytes - done);
		if ( put < 0 && errno == EINTR )
			continue;
		if ( put <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)put;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkIoRecvAll(int fd, void *data, size_t bytes)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t got = recv(fd,(uint8_t *)data + done,bytes - done,0);
		if ( got < 0 && errno == EINTR )
			continue;
		if ( got <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)got;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkIoSendAll(int fd, const void *data, size_t bytes)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t put = SparkIoSend(fd,(const uint8_t *)data + done,bytes - done);
		if ( put < 0 && errno == EINTR )
			continue;
		if ( put <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)put;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkIoWait(int fd, short events, uint64_t deadline_ns)
{
	struct pollfd descriptor;
	const uint64_t now = SparkMonotonicNs();
	int ready;
	if ( now == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( now >= deadline_ns )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	descriptor.fd = fd;
	descriptor.events = events;
	descriptor.revents = 0;
	ready = poll(&descriptor,1u,(int)((deadline_ns - now) / UINT64_C(1000000)) + 1);
	if ( ready < 0 && errno == EINTR )
		return(SPARK_STATUS_OK);
	if ( ready == 0 )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	if ( ready < 0 || (descriptor.revents & (events | POLLHUP | POLLERR)) == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkIoRecvAllBefore(int fd, void *data, size_t bytes, uint64_t deadline_ns)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		const SparkStatus status = SparkIoWait(fd,POLLIN,deadline_ns);
		ssize_t got;
		if ( status != SPARK_STATUS_OK )
			return(status);
		got = recv(fd,(uint8_t *)data + done,bytes - done,0);
		if ( got < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) )
			continue;
		if ( got <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)got;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkIoSendAllBefore(int fd, const void *data, size_t bytes, uint64_t deadline_ns)
{
	size_t done = 0u;
	while ( done < bytes )
	{
		const ssize_t put = SparkIoSend(fd,(const uint8_t *)data + done,bytes - done);
		if ( put < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) )
		{
			const SparkStatus status = errno == EINTR ? SPARK_STATUS_OK : SparkIoWait(fd,POLLOUT,deadline_ns);
			if ( status != SPARK_STATUS_OK )
				return(status);
			continue;
		}
		if ( put <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		done += (size_t)put;
	}
	return(SPARK_STATUS_OK);
}
