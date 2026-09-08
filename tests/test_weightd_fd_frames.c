#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include "../runtime/spark_weightd.c"

static uint32_t fd_count(void)
{
	uint32_t i,count = 0u;
	for (i=0u; i<1024u; i++)
		count += fcntl((int32_t)i,F_GETFD) >= 0;
	return(count);
}

static void send_fds(int32_t socket_fd,int32_t source,uint32_t count)
{
	union { struct cmsghdr align; uint8_t bytes[CMSG_SPACE(65u * sizeof(int))]; } control;
	struct msghdr message = {0};
	struct iovec vector;
	struct cmsghdr *header;
	uint8_t byte = 1u;
	uint32_t i;
	vector.iov_base = &byte;
	vector.iov_len = 1u;
	message.msg_iov = &vector;
	message.msg_iovlen = 1u;
	message.msg_control = control.bytes;
	message.msg_controllen = CMSG_SPACE(count * sizeof(int));
	memset(&control,0,sizeof(control));
	header = CMSG_FIRSTHDR(&message);
	header->cmsg_level = SOL_SOCKET;
	header->cmsg_type = SCM_RIGHTS;
	header->cmsg_len = CMSG_LEN(count * sizeof(int));
	for (i=0u; i<count; i++)
		memcpy((uint8_t *)CMSG_DATA(header) + (i * sizeof(int)),&source,sizeof(source));
	assert(sendmsg(socket_fd,&message,0) == 1);
}

static void check_frame(uint32_t sent,uint32_t capacity,SparkStatus expected)
{
	int32_t sockets[2],source,fds[64];
	SparkWeightdClient client = {0};
	uint32_t before,received = 0u,i;
	uint8_t byte;
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	source = open("/dev/null",O_RDONLY);
	assert(source >= 0);
	client.fd = sockets[0];
	before = fd_count();
	send_fds(sockets[1],source,sent);
	assert(SparkWeightdClientReadFrameWithFds(&client,&byte,1u,SparkWeightdMonotonicTimeNs() + UINT64_C(1000000000),fds,capacity,&received) == expected);
	assert(received == (expected == SPARK_STATUS_OK ? sent : 0u));
	for (i=0u; i<received; i++)
	{
		assert((fcntl(fds[i],F_GETFD) & FD_CLOEXEC) != 0);
		assert(close(fds[i]) == 0);
	}
	if ( fd_count() != before )
		fprintf(stderr,"FD leak: sent=%u capacity=%u received=%u before=%u after=%u\n",sent,capacity,received,before,fd_count());
	assert(fd_count() == before);
	assert(close(source) == 0 && close(sockets[0]) == 0 && close(sockets[1]) == 0);
}

static void check_reported_truncation(void)
{
	union { struct cmsghdr align; uint8_t bytes[CMSG_SPACE(sizeof(int))]; } control;
	struct msghdr message = {0};
	struct cmsghdr *header;
	int32_t fd = dup(STDERR_FILENO),fds[1];
	uint32_t count = 0u;
	assert(fd >= 0);
	message.msg_control = control.bytes;
	message.msg_controllen = sizeof(control);
	message.msg_flags = MSG_CTRUNC;
	header = CMSG_FIRSTHDR(&message);
	header->cmsg_level = SOL_SOCKET;
	header->cmsg_type = SCM_RIGHTS;
	header->cmsg_len = CMSG_LEN(sizeof(int));
	memcpy(CMSG_DATA(header),&fd,sizeof(fd));
	assert(SparkWeightdReceiveFds(&message,fds,1u,&count) == SPARK_STATUS_IO_ERROR);
	assert(count == 1u && fds[0] == fd);
	assert(close(fds[0]) == 0);
}

static void check_lease_frame_shape(void)
{
	SparkWeightdIpcExportLease request = {0};
	SparkWeightdIpcExportLeaseResult response = {0},bad;
	SparkWeightdBuildHeader((uint8_t *)&request,SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE,1u);
	SparkWeightdBuildHeader((uint8_t *)&response,SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE_RESULT,1u);
	request.arena_generation = response.base.arena_generation = 2u;
	request.lease_identifier = response.lease_identifier = 3u;
	response.base.chunk_bytes = 2097152u;
	response.base.chunk_count = 3u;
	response.base.batch_count = response.lease_chunk_count = 2u;
	response.chunk_indices[1] = 2u;
	assert(SparkWeightdValidateLeaseExport(&request,&response,2u) == SPARK_STATUS_OK);
	bad = response;
	bad.chunk_indices[1] = 0u;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = response;
	bad.lease_identifier++;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = response;
	bad.base.arena_generation++;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkWeightdValidateLeaseExport(&request,&response,1u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = response;
	bad.base.batch_offset = 2u;
	assert(SparkWeightdValidateLeaseExport(&request,&bad,2u) == SPARK_STATUS_SCHEMA_ERROR);
}

int main(void)
{
	check_frame(2u,2u,SPARK_STATUS_OK);
	check_frame(64u,64u,SPARK_STATUS_OK);
	check_frame(2u,1u,SPARK_STATUS_IO_ERROR);
	check_frame(65u,64u,SPARK_STATUS_IO_ERROR);
	check_reported_truncation();
	check_lease_frame_shape();
	puts("PASS FD frames: CLOEXEC, oversized ancillary cleanup and truncation rejection");
	return(0);
}
