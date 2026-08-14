#define _POSIX_C_SOURCE 200809L

#include "tp_device_collective_nccl.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SPARK_TP_NCCL_UNIQUE_ID_BYTES 128u
#define SPARK_TP_NCCL_BOOTSTRAP_MAGIC 0x53504e43u
#define SPARK_TP_NCCL_BOOTSTRAP_ABI_VERSION 1u
#define SPARK_TP_NCCL_CONNECT_RETRY_NANO 2000000L
#define SPARK_TP_NCCL_MINIMUM_VERSION 23000
#define SPARK_TP_NCCL_SUCCESS 0
#define SPARK_TP_NCCL_IN_PROGRESS 7
#define SPARK_TP_NCCL_DATA_TYPE_U64 5
#define SPARK_TP_NCCL_DATA_TYPE_BF16 9
#define SPARK_TP_NCCL_REDUCTION_SUM 0
#define SPARK_TP_NCCL_REDUCTION_MAX 2

typedef int32_t SparkTpNcclResult;
typedef struct SparkTpNcclComm *SparkTpNcclCommHandle;

typedef struct SparkTpNcclUniqueId
{
	uint8_t bytes[SPARK_TP_NCCL_UNIQUE_ID_BYTES];
} SparkTpNcclUniqueId;

typedef SparkTpNcclResult (*SparkTpNcclGetVersionFunction)(int32_t *version);
typedef SparkTpNcclResult (*SparkTpNcclGetUniqueIdFunction)(SparkTpNcclUniqueId *unique_id);
typedef SparkTpNcclResult (*SparkTpNcclCommInitRankFunction)(SparkTpNcclCommHandle *communicator,int32_t rank_count,SparkTpNcclUniqueId unique_id,int32_t rank);
typedef SparkTpNcclResult (*SparkTpNcclAllReduceFunction)(const void *send_device,void *receive_device,size_t element_count,int32_t data_type,int32_t reduction_operation,SparkTpNcclCommHandle communicator,void *cuda_stream);
typedef SparkTpNcclResult (*SparkTpNcclCommGetAsyncErrorFunction)(SparkTpNcclCommHandle communicator,SparkTpNcclResult *async_error);
typedef SparkTpNcclResult (*SparkTpNcclCommDestroyFunction)(SparkTpNcclCommHandle communicator);
typedef SparkTpNcclResult (*SparkTpNcclCommAbortFunction)(SparkTpNcclCommHandle communicator);
typedef const char *(*SparkTpNcclGetErrorStringFunction)(SparkTpNcclResult result);

typedef struct SparkTpNcclLibrary
{
	void *dynamic_library;
	SparkTpNcclGetVersionFunction get_version;
	SparkTpNcclGetUniqueIdFunction get_unique_id;
	SparkTpNcclCommInitRankFunction comm_init_rank;
	SparkTpNcclAllReduceFunction all_reduce;
	SparkTpNcclCommGetAsyncErrorFunction comm_get_async_error;
	SparkTpNcclCommDestroyFunction comm_destroy;
	SparkTpNcclCommAbortFunction comm_abort;
	SparkTpNcclGetErrorStringFunction get_error_string;
} SparkTpNcclLibrary;

typedef struct SparkTpNcclBootstrapHello
{
	uint32_t magic;
	uint32_t abi_version;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t identifier_high;
	uint32_t identifier_low;
} SparkTpNcclBootstrapHello;

typedef struct SparkTpNcclBootstrapResponse
{
	uint32_t magic;
	uint32_t abi_version;
	uint32_t status;
	uint32_t tp_degree;
	uint32_t identifier_high;
	uint32_t identifier_low;
	uint8_t unique_id[SPARK_TP_NCCL_UNIQUE_ID_BYTES];
} SparkTpNcclBootstrapResponse;

typedef struct SparkTpDeviceNcclImplementation
{
	SparkTpNcclLibrary library;
	SparkTpNcclCommHandle communicator;
	pthread_mutex_t mutex;
	atomic_uint admission_open;
	atomic_int failure_status;
	atomic_uint_fast64_t next_ordinal;
	uint32_t mutex_initialized;
	uint32_t communicator_aborted;
} SparkTpDeviceNcclImplementation;

static uint32_t SparkTpNcclDegreeIsSupported(uint32_t tp_degree)
{
	return(tp_degree == 2u || tp_degree == 4u || tp_degree == 8u ||
		tp_degree == 16u ? 1u : 0u);
}

static uint32_t SparkTpNcclTextIsValid(const char *text)
{
	return(text != 0 && text[0] != '\0' ? 1u : 0u);
}

static uint64_t SparkTpNcclNowMilli(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(UINT64_MAX);
	return((uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u);
}

static uint64_t SparkTpNcclDeadline(uint32_t timeout_milli)
{
	uint64_t now;
	now = SparkTpNcclNowMilli();
	if ( now == UINT64_MAX || UINT64_MAX - now < timeout_milli )
		return(UINT64_MAX);
	return(now + timeout_milli);
}

static int32_t SparkTpNcclPollTimeout(uint64_t deadline_milli)
{
	uint64_t now,remaining;
	now = SparkTpNcclNowMilli();
	if ( now == UINT64_MAX || now >= deadline_milli )
		return(0);
	remaining = deadline_milli - now;
	return(remaining > INT_MAX ? INT_MAX : (int32_t)remaining);
}

static SparkStatus SparkTpNcclPollSocket(int32_t socket_descriptor,short events,uint64_t deadline_milli,short *returned_events)
{
	struct pollfd descriptor;
	int32_t result,timeout;
	if ( socket_descriptor < 0 || returned_events == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (;;)
	{
		timeout = SparkTpNcclPollTimeout(deadline_milli);
		if ( timeout <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		memset(&descriptor,0,sizeof(descriptor));
		descriptor.fd = socket_descriptor;
		descriptor.events = events;
		result = poll(&descriptor,1u,timeout);
		if ( result > 0 )
		{
			*returned_events = descriptor.revents;
			return(SPARK_STATUS_OK);
		}
		if ( result == 0 || errno != EINTR )
			return(SPARK_STATUS_IO_ERROR);
	}
}

static SparkStatus SparkTpNcclSetNonblocking(int32_t socket_descriptor)
{
	int32_t flags;
	flags = fcntl(socket_descriptor,F_GETFL,0);
	if ( flags < 0 || fcntl(socket_descriptor,F_SETFL,flags | O_NONBLOCK) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclConfigureSocket(int32_t socket_descriptor)
{
	int32_t enabled;
	enabled = 1;
	if ( setsockopt(socket_descriptor,IPPROTO_TCP,TCP_NODELAY,&enabled,
		sizeof(enabled)) != 0 )
		return(SPARK_STATUS_IO_ERROR);
#ifdef SO_NOSIGPIPE
	if ( setsockopt(socket_descriptor,SOL_SOCKET,SO_NOSIGPIPE,&enabled,
		sizeof(enabled)) != 0 )
		return(SPARK_STATUS_IO_ERROR);
#endif
	return(SparkTpNcclSetNonblocking(socket_descriptor));
}

static SparkStatus SparkTpNcclSendAll(int32_t socket_descriptor,const void *data,uint32_t data_bytes,uint64_t deadline_milli)
{
	const uint8_t *source;
	uint32_t sent;
	source = (const uint8_t *)data;
	for (sent=0u; sent<data_bytes; )
	{
		ssize_t result;
		short events;
		int32_t flags;
#ifdef MSG_NOSIGNAL
		flags = MSG_NOSIGNAL;
#else
		flags = 0;
#endif
		result = send(socket_descriptor,source + sent,data_bytes - sent,flags);
		if ( result > 0 )
		{
			sent += (uint32_t)result;
			continue;
		}
		if ( result == 0 )
			return(SPARK_STATUS_IO_ERROR);
		if ( errno == EINTR )
			continue;
		if ( (errno != EAGAIN && errno != EWOULDBLOCK) ||
			SparkTpNcclPollSocket(socket_descriptor,POLLOUT,deadline_milli,
			&events) != SPARK_STATUS_OK || (events & POLLOUT) == 0 ||
			(events & (POLLERR | POLLHUP | POLLNVAL)) != 0 )
			return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclReceiveAll(int32_t socket_descriptor,void *data,uint32_t data_bytes,uint64_t deadline_milli)
{
	uint8_t *destination;
	uint32_t received;
	destination = (uint8_t *)data;
	for (received=0u; received<data_bytes; )
	{
		ssize_t result;
		short events;
		result = recv(socket_descriptor,destination + received,
			data_bytes - received,0);
		if ( result > 0 )
		{
			received += (uint32_t)result;
			continue;
		}
		if ( result == 0 )
			return(SPARK_STATUS_IO_ERROR);
		if ( errno == EINTR )
			continue;
		if ( (errno != EAGAIN && errno != EWOULDBLOCK) ||
			SparkTpNcclPollSocket(socket_descriptor,POLLIN,deadline_milli,
			&events) != SPARK_STATUS_OK || (events & POLLIN) == 0 ||
			(events & (POLLERR | POLLNVAL)) != 0 )
			return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static void SparkTpNcclRetryPause(void)
{
	struct timespec delay;
	delay.tv_sec = 0;
	delay.tv_nsec = SPARK_TP_NCCL_CONNECT_RETRY_NANO;
	while ( nanosleep(&delay,&delay) != 0 && errno == EINTR )
		;
}

static SparkStatus SparkTpNcclConnectAddress(const struct sockaddr *address,socklen_t address_bytes,uint64_t deadline_milli,int32_t *socket_out)
{
	int32_t descriptor,result,socket_error;
	socklen_t socket_error_bytes;
	short events;
	SparkStatus status;
	descriptor = socket(address->sa_family,SOCK_STREAM,0);
	if ( descriptor < 0 )
		return(SPARK_STATUS_IO_ERROR);
	status = SparkTpNcclConfigureSocket(descriptor);
	result = status == SPARK_STATUS_OK ?
		connect(descriptor,address,address_bytes) : -1;
	if ( result == 0 )
	{
		*socket_out = descriptor;
		return(SPARK_STATUS_OK);
	}
	if ( status == SPARK_STATUS_OK &&
		(errno == EINPROGRESS || errno == EALREADY || errno == EWOULDBLOCK) &&
		SparkTpNcclPollSocket(descriptor,POLLOUT,deadline_milli,&events) ==
		SPARK_STATUS_OK && (events & (POLLOUT | POLLERR | POLLHUP)) != 0 )
	{
		socket_error = 0;
		socket_error_bytes = sizeof(socket_error);
		if ( getsockopt(descriptor,SOL_SOCKET,SO_ERROR,&socket_error,
			&socket_error_bytes) == 0 && socket_error == 0 )
		{
			*socket_out = descriptor;
			return(SPARK_STATUS_OK);
		}
	}
	(void)close(descriptor);
	return(SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkTpNcclConnectUntil(const char *host,uint16_t port,uint64_t deadline_milli,int32_t *socket_out)
{
	struct addrinfo hints,*addresses,*address;
	char service[16];
	SparkStatus status;
	if ( snprintf(service,sizeof(service),"%u",(uint32_t)port) <= 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	memset(&hints,0,sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	if ( getaddrinfo(host,service,&hints,&addresses) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	status = SPARK_STATUS_IO_ERROR;
	while ( SparkTpNcclPollTimeout(deadline_milli) > 0 )
	{
		for (address=addresses; address!=0; address=address->ai_next)
		{
			status = SparkTpNcclConnectAddress(address->ai_addr,
				(socklen_t)address->ai_addrlen,deadline_milli,socket_out);
			if ( status == SPARK_STATUS_OK )
				break;
		}
		if ( status == SPARK_STATUS_OK )
			break;
		SparkTpNcclRetryPause();
	}
	freeaddrinfo(addresses);
	return(status);
}

static SparkStatus SparkTpNcclListen(uint16_t port,int32_t *socket_out)
{
	struct sockaddr_in address;
	int32_t descriptor,enabled;
	descriptor = socket(AF_INET,SOCK_STREAM,0);
	if ( descriptor < 0 )
		return(SPARK_STATUS_IO_ERROR);
	enabled = 1;
	if ( setsockopt(descriptor,SOL_SOCKET,SO_REUSEADDR,&enabled,
		sizeof(enabled)) != 0 || SparkTpNcclSetNonblocking(descriptor) !=
		SPARK_STATUS_OK )
	{
		(void)close(descriptor);
		return(SPARK_STATUS_IO_ERROR);
	}
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_ANY);
	address.sin_port = htons(port);
	if ( bind(descriptor,(const struct sockaddr *)&address,sizeof(address)) != 0 ||
		listen(descriptor,(int32_t)SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE) != 0 )
	{
		(void)close(descriptor);
		return(SPARK_STATUS_IO_ERROR);
	}
	*socket_out = descriptor;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclAcceptUntil(int32_t listen_socket,uint64_t deadline_milli,int32_t *socket_out)
{
	short events;
	SparkStatus status;
	for (;;)
	{
		*socket_out = accept(listen_socket,0,0);
		if ( *socket_out >= 0 )
		{
			status = SparkTpNcclConfigureSocket(*socket_out);
			if ( status != SPARK_STATUS_OK )
			{
				(void)close(*socket_out);
				*socket_out = -1;
			}
			return(status);
		}
		if ( errno == EINTR )
			continue;
		if ( errno != EAGAIN && errno != EWOULDBLOCK )
			return(SPARK_STATUS_IO_ERROR);
		status = SparkTpNcclPollSocket(listen_socket,POLLIN,deadline_milli,
			&events);
		if ( status != SPARK_STATUS_OK || (events & POLLIN) == 0 ||
			(events & (POLLERR | POLLHUP | POLLNVAL)) != 0 )
			return(SPARK_STATUS_IO_ERROR);
	}
}

static void SparkTpNcclEncodeIdentifier(uint64_t identifier,uint32_t *high,uint32_t *low)
{
	*high = htonl((uint32_t)(identifier >> 32u));
	*low = htonl((uint32_t)identifier);
}

static uint64_t SparkTpNcclDecodeIdentifier(uint32_t high,uint32_t low)
{
	return((uint64_t)ntohl(high) << 32u | (uint64_t)ntohl(low));
}

static void SparkTpNcclBuildHello(const SparkTpDeviceCollectiveConfig *config,SparkTpNcclBootstrapHello *hello)
{
	memset(hello,0,sizeof(*hello));
	hello->magic = htonl(SPARK_TP_NCCL_BOOTSTRAP_MAGIC);
	hello->abi_version = htonl(SPARK_TP_NCCL_BOOTSTRAP_ABI_VERSION);
	hello->tp_degree = htonl(config->tp_degree);
	hello->tp_rank = htonl(config->tp_rank);
	SparkTpNcclEncodeIdentifier(config->collective_identifier,
		&hello->identifier_high,&hello->identifier_low);
}

static SparkStatus SparkTpNcclValidateHello(const SparkTpDeviceCollectiveConfig *config,const SparkTpNcclBootstrapHello *hello,uint32_t seen_mask,uint32_t *rank_out)
{
	uint32_t rank;
	rank = ntohl(hello->tp_rank);
	if ( ntohl(hello->magic) != SPARK_TP_NCCL_BOOTSTRAP_MAGIC ||
		SparkTpNcclDecodeIdentifier(hello->identifier_high,
		hello->identifier_low) != config->collective_identifier )
		return(SPARK_STATUS_NOT_FOUND);
	if ( ntohl(hello->abi_version) != SPARK_TP_NCCL_BOOTSTRAP_ABI_VERSION ||
		ntohl(hello->tp_degree) != config->tp_degree || rank == 0u ||
		rank >= config->tp_degree )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( (seen_mask & (1u << rank)) != 0u )
		return(SPARK_STATUS_DUPLICATE);
	*rank_out = rank;
	return(SPARK_STATUS_OK);
}

static void SparkTpNcclBuildResponse(const SparkTpDeviceCollectiveConfig *config,const SparkTpNcclUniqueId *unique_id,SparkStatus status,SparkTpNcclBootstrapResponse *response)
{
	memset(response,0,sizeof(*response));
	response->magic = htonl(SPARK_TP_NCCL_BOOTSTRAP_MAGIC);
	response->abi_version = htonl(SPARK_TP_NCCL_BOOTSTRAP_ABI_VERSION);
	response->status = htonl((uint32_t)status);
	response->tp_degree = htonl(config->tp_degree);
	SparkTpNcclEncodeIdentifier(config->collective_identifier,
		&response->identifier_high,&response->identifier_low);
	if ( unique_id != 0 )
		memcpy(response->unique_id,unique_id->bytes,sizeof(response->unique_id));
}

static SparkStatus SparkTpNcclValidateResponse(const SparkTpDeviceCollectiveConfig *config,const SparkTpNcclBootstrapResponse *response,SparkTpNcclUniqueId *unique_id)
{
	uint32_t status;
	if ( ntohl(response->magic) != SPARK_TP_NCCL_BOOTSTRAP_MAGIC ||
		ntohl(response->abi_version) != SPARK_TP_NCCL_BOOTSTRAP_ABI_VERSION ||
		ntohl(response->tp_degree) != config->tp_degree ||
		SparkTpNcclDecodeIdentifier(response->identifier_high,
		response->identifier_low) != config->collective_identifier )
		return(SPARK_STATUS_VALIDATION_FAILED);
	status = ntohl(response->status);
	if ( status > (uint32_t)SPARK_STATUS_UNSUPPORTED ||
		(SparkStatus)status != SPARK_STATUS_OK )
		return(status <= (uint32_t)SPARK_STATUS_UNSUPPORTED ?
			(SparkStatus)status : SPARK_STATUS_VALIDATION_FAILED);
	memcpy(unique_id->bytes,response->unique_id,sizeof(unique_id->bytes));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclServeUniqueId(const SparkTpDeviceCollectiveConfig *config,const SparkTpNcclUniqueId *unique_id,uint64_t deadline_milli)
{
	SparkTpNcclBootstrapHello hello;
	SparkTpNcclBootstrapResponse response;
	uint32_t accepted,rank,seen_mask;
	int32_t listen_socket,peer_socket;
	SparkStatus status;
	listen_socket = -1;
	rank = 0u;
	status = SparkTpNcclListen((uint16_t)config->control_port_base,
		&listen_socket);
	seen_mask = 1u;
	for (accepted=1u; status==SPARK_STATUS_OK && accepted<config->tp_degree;
		accepted++)
	{
		peer_socket = -1;
		status = SparkTpNcclAcceptUntil(listen_socket,deadline_milli,&peer_socket);
		if ( status == SPARK_STATUS_OK )
			status = SparkTpNcclReceiveAll(peer_socket,&hello,sizeof(hello),
				deadline_milli);
		if ( status == SPARK_STATUS_OK )
			status = SparkTpNcclValidateHello(config,&hello,seen_mask,&rank);
		SparkTpNcclBuildResponse(config,unique_id,status,&response);
		if ( peer_socket >= 0 )
		{
			if ( SparkTpNcclSendAll(peer_socket,&response,sizeof(response),
				deadline_milli) != SPARK_STATUS_OK && status == SPARK_STATUS_OK )
				status = SPARK_STATUS_IO_ERROR;
			(void)close(peer_socket);
		}
		if ( status == SPARK_STATUS_OK )
			seen_mask |= 1u << rank;
	}
	if ( listen_socket >= 0 )
		(void)close(listen_socket);
	return(status);
}

static SparkStatus SparkTpNcclFetchUniqueId(const SparkTpDeviceCollectiveConfig *config,SparkTpNcclUniqueId *unique_id,uint64_t deadline_milli)
{
	SparkTpNcclBootstrapHello hello;
	SparkTpNcclBootstrapResponse response;
	int32_t socket_descriptor;
	SparkStatus status;
	socket_descriptor = -1;
	status = SparkTpNcclConnectUntil(config->rank_hosts[0],
		(uint16_t)config->control_port_base,deadline_milli,&socket_descriptor);
	SparkTpNcclBuildHello(config,&hello);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpNcclSendAll(socket_descriptor,&hello,sizeof(hello),
			deadline_milli);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpNcclReceiveAll(socket_descriptor,&response,
			sizeof(response),deadline_milli);
	if ( socket_descriptor >= 0 )
		(void)close(socket_descriptor);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpNcclValidateResponse(config,&response,unique_id);
	return(status);
}

static SparkStatus SparkTpNcclAssignSymbol(void *dynamic_library,const char *name,void *function_out,uint32_t function_bytes)
{
	void *symbol;
	if ( dynamic_library == 0 || name == 0 || function_out == 0 ||
		function_bytes != sizeof(symbol) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	(void)dlerror();
	symbol = dlsym(dynamic_library,name);
	if ( symbol == 0 || dlerror() != 0 )
		return(SPARK_STATUS_DRIVER_LOAD_ERROR);
	memcpy(function_out,&symbol,sizeof(symbol));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclLoadLibrary(const char *path,SparkTpNcclLibrary *library,int32_t *version_out)
{
	SparkStatus status;
	memset(library,0,sizeof(*library));
	library->dynamic_library = dlopen(path,RTLD_NOW | RTLD_LOCAL);
	if ( library->dynamic_library == 0 )
		return(SPARK_STATUS_DRIVER_LOAD_ERROR);
#define SPARK_TP_NCCL_LOAD_SYMBOL(member,name) \
	status = SparkTpNcclAssignSymbol(library->dynamic_library,name, \
		&library->member,sizeof(library->member)); \
	if ( status != SPARK_STATUS_OK ) \
		goto fail_load
	SPARK_TP_NCCL_LOAD_SYMBOL(get_version,"ncclGetVersion");
	SPARK_TP_NCCL_LOAD_SYMBOL(get_unique_id,"ncclGetUniqueId");
	SPARK_TP_NCCL_LOAD_SYMBOL(comm_init_rank,"ncclCommInitRank");
	SPARK_TP_NCCL_LOAD_SYMBOL(all_reduce,"ncclAllReduce");
	SPARK_TP_NCCL_LOAD_SYMBOL(comm_get_async_error,"ncclCommGetAsyncError");
	SPARK_TP_NCCL_LOAD_SYMBOL(comm_destroy,"ncclCommDestroy");
	SPARK_TP_NCCL_LOAD_SYMBOL(comm_abort,"ncclCommAbort");
	SPARK_TP_NCCL_LOAD_SYMBOL(get_error_string,"ncclGetErrorString");
#undef SPARK_TP_NCCL_LOAD_SYMBOL
	if ( library->get_version(version_out) != SPARK_TP_NCCL_SUCCESS ||
		*version_out < SPARK_TP_NCCL_MINIMUM_VERSION )
	{
		status = SPARK_STATUS_MODULE_NOT_VALIDATED;
		goto fail_load;
	}
	return(SPARK_STATUS_OK);
fail_load:
	(void)dlclose(library->dynamic_library);
	memset(library,0,sizeof(*library));
	return(status);
}

static void SparkTpNcclReportError(const SparkTpDeviceCollective *collective,const SparkTpDeviceNcclImplementation *implementation,const char *operation,SparkTpNcclResult result)
{
	const char *message;
	message = implementation->library.get_error_string != 0 ?
		implementation->library.get_error_string(result) : "unknown";
	fprintf(stderr,"sparkpipe_tp_nccl_error tp_rank=%u operation=%s result=%d message=%s\n",
		collective != 0 ? collective->tp_rank : UINT32_MAX,operation,(int32_t)result,
		message != 0 ? message : "unknown");
}

static SparkStatus SparkTpNcclValidateConfig(const SparkTpDeviceCollectiveConfig *config)
{
	uint32_t rank;
	if ( config == 0 || config->abi_version !=
		SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION || config->backend_kind !=
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL ||
		SparkTpNcclDegreeIsSupported(config->tp_degree) == 0u ||
		config->tp_rank >= config->tp_degree || config->operation_kind !=
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 ||
		config->credit_count == 0u || config->credit_count >
		SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
		config->local_hidden_dimension == 0u ||
		config->max_active_sequence_count == 0u ||
		config->algorithm_mask != 0u || config->rail_count != 0u ||
		config->direct_all_to_all_max_payload_bytes != 0u ||
		config->split_ring_min_payload_bytes != 0u ||
		config->connect_timeout_milli == 0u ||
		config->operation_timeout_milli == 0u ||
		config->control_port_base == 0u ||
		config->control_port_base > UINT16_MAX ||
		config->collective_identifier == 0u ||
		SparkTpNcclTextIsValid(config->backend_module_path) == 0u ||
		SparkTpNcclTextIsValid(config->local_host) == 0u ||
		config->credit_bindings != 0 || config->credit_binding_count != 0u ||
		config->combine_bf16_function != 0 ||
		config->combine_relay_bf16_function != 0 ||
		config->combine_tp4_bf16_function != 0 ||
		config->combine_u64_max_function != 0 || config->combine_context != 0 ||
		config->debug_hooks != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (rank=0u; rank<config->tp_degree; rank++)
		if ( SparkTpNcclTextIsValid(config->rank_hosts[rank]) == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	for (; rank<SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE; rank++)
		if ( config->rank_hosts[rank] != 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclDistributeUniqueId(const SparkTpDeviceCollectiveConfig *config,SparkTpDeviceNcclImplementation *implementation,SparkTpNcclUniqueId *unique_id)
{
	uint64_t deadline_milli;
	SparkTpNcclResult result;
	deadline_milli = SparkTpNcclDeadline(config->connect_timeout_milli);
	if ( deadline_milli == UINT64_MAX )
		return(SPARK_STATUS_IO_ERROR);
	if ( config->tp_rank != 0u )
		return(SparkTpNcclFetchUniqueId(config,unique_id,deadline_milli));
	result = implementation->library.get_unique_id(unique_id);
	if ( result != SPARK_TP_NCCL_SUCCESS )
	{
		SparkTpNcclReportError(0,implementation,"get_unique_id",result);
		return(SPARK_STATUS_DRIVER_LOAD_ERROR);
	}
	return(SparkTpNcclServeUniqueId(config,unique_id,deadline_milli));
}

static void SparkTpNcclResetCollective(SparkTpDeviceCollective *collective)
{
	memset(collective,0,sizeof(*collective));
}

SparkStatus SparkTpDeviceCollectiveNcclCreate(const SparkTpDeviceCollectiveConfig *config,SparkTpDeviceCollective *collective_out)
{
	SparkTpDeviceNcclImplementation *implementation;
	SparkTpNcclUniqueId unique_id;
	SparkTpNcclResult result;
	SparkStatus status;
	int32_t version;
	if ( collective_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkTpNcclResetCollective(collective_out);
	status = SparkTpNcclValidateConfig(config);
	if ( status != SPARK_STATUS_OK )
		return(status);
	implementation = (SparkTpDeviceNcclImplementation *)calloc(1u,
		sizeof(*implementation));
	if ( implementation == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = pthread_mutex_init(&implementation->mutex,0) == 0 ?
		SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	implementation->mutex_initialized = status == SPARK_STATUS_OK ? 1u : 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkTpNcclLoadLibrary(config->backend_module_path,
			&implementation->library,&version);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpNcclDistributeUniqueId(config,implementation,&unique_id);
	if ( status == SPARK_STATUS_OK )
	{
		result = implementation->library.comm_init_rank(
			&implementation->communicator,(int32_t)config->tp_degree,unique_id,
			(int32_t)config->tp_rank);
		if ( result != SPARK_TP_NCCL_SUCCESS )
		{
			SparkTpNcclReportError(collective_out,implementation,
				"comm_init_rank",result);
			status = SPARK_STATUS_DRIVER_LOAD_ERROR;
		}
	}
	if ( status != SPARK_STATUS_OK )
	{
		if ( implementation->communicator != 0 &&
			implementation->library.comm_abort != 0 )
			(void)implementation->library.comm_abort(
				implementation->communicator);
		if ( implementation->library.dynamic_library != 0 )
			(void)dlclose(implementation->library.dynamic_library);
		if ( implementation->mutex_initialized != 0u )
			(void)pthread_mutex_destroy(&implementation->mutex);
		free(implementation);
		return(status);
	}
	atomic_init(&implementation->admission_open,1u);
	atomic_init(&implementation->failure_status,SPARK_STATUS_OK);
	atomic_init(&implementation->next_ordinal,0u);
	collective_out->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	collective_out->backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	collective_out->tp_degree = config->tp_degree;
	collective_out->tp_rank = config->tp_rank;
	collective_out->operation_kind = config->operation_kind;
	collective_out->credit_count = config->credit_count;
	collective_out->local_hidden_dimension = config->local_hidden_dimension;
	collective_out->max_active_sequence_count = config->max_active_sequence_count;
	collective_out->operation_timeout_milli = config->operation_timeout_milli;
	collective_out->memory_mode = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE;
	collective_out->collective_identifier = config->collective_identifier;
	collective_out->implementation = implementation;
	fprintf(stderr,"sparkpipe_tp_collective backend=nccl version=%d tp_rank=%u tp_degree=%u\n",
		version,config->tp_rank,config->tp_degree);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclValidateSubmission(const SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission)
{
	if ( collective == 0 || collective->abi_version !=
		SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION || collective->backend_kind !=
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL ||
		collective->implementation == 0 || submission == 0 ||
		submission->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
		submission->descriptor_bytes != sizeof(*submission) ||
		submission->active_sequence_count == 0u ||
		submission->active_sequence_count > collective->max_active_sequence_count ||
		(submission->flags & ~SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_KNOWN_FLAGS) != 0u ||
		(submission->flags &
		 SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_EXTERNAL_GRAPH_ORDER) != 0u ||
		submission->local_device == 0 || submission->full_device == 0 ||
		submission->cuda_stream == 0 || submission->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpNcclCheckAsyncError(SparkTpDeviceCollective *collective,SparkTpDeviceNcclImplementation *implementation)
{
	SparkTpNcclResult async_error,result;
	async_error = SPARK_TP_NCCL_SUCCESS;
	result = implementation->library.comm_get_async_error(
		implementation->communicator,&async_error);
	if ( result == SPARK_TP_NCCL_SUCCESS &&
		(async_error == SPARK_TP_NCCL_SUCCESS ||
		 async_error == SPARK_TP_NCCL_IN_PROGRESS) )
		return(SPARK_STATUS_OK);
	SparkTpNcclReportError(collective,implementation,"comm_async_error",
		result != SPARK_TP_NCCL_SUCCESS ? result : async_error);
	atomic_store_explicit(&implementation->admission_open,0u,
		memory_order_release);
	atomic_store_explicit(&implementation->failure_status,
		SPARK_STATUS_IO_ERROR,memory_order_release);
	return(SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkTpNcclSubmitAllReduce(
	SparkTpDeviceCollective *collective,
	const SparkTpDeviceCollectiveSubmission *submission,
	uint64_t element_count,
	int32_t data_type,
	int32_t reduction_operation)
{
	SparkTpDeviceNcclImplementation *implementation;
	SparkTpDeviceCollectiveCompletion completion;
	SparkTpNcclResult result;
	SparkStatus status;
	status = SparkTpNcclValidateSubmission(collective,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	implementation = (SparkTpDeviceNcclImplementation *)collective->implementation;
	if ( pthread_mutex_lock(&implementation->mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = atomic_load_explicit(&implementation->admission_open,
		memory_order_acquire) != 0u ? SPARK_STATUS_OK :
		(SparkStatus)atomic_load_explicit(&implementation->failure_status,
		memory_order_acquire);
	if ( status == SPARK_STATUS_OK && submission->ordinal !=
		atomic_load_explicit(&implementation->next_ordinal,
			memory_order_relaxed) )
		status = SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
		status = SparkTpNcclCheckAsyncError(collective,implementation);
	if ( status == SPARK_STATUS_OK && element_count > SIZE_MAX )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
	{
		result = implementation->library.all_reduce(submission->local_device,
			submission->full_device,(size_t)element_count,
			data_type,reduction_operation,
			implementation->communicator,submission->cuda_stream);
		if ( result != SPARK_TP_NCCL_SUCCESS )
		{
			SparkTpNcclReportError(collective,implementation,"all_reduce",result);
			status = SPARK_STATUS_IO_ERROR;
			atomic_store_explicit(&implementation->admission_open,0u,
				memory_order_release);
			atomic_store_explicit(&implementation->failure_status,status,
				memory_order_release);
		}
		else
			(void)atomic_fetch_add_explicit(&implementation->next_ordinal,1u,
				memory_order_relaxed);
	}
	(void)pthread_mutex_unlock(&implementation->mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	completion.descriptor_bytes = sizeof(completion);
	completion.status = SPARK_STATUS_OK;
	completion.slot_index = submission->slot_index;
	completion.credit_index = (uint32_t)(submission->ordinal %
		collective->credit_count);
	completion.ordinal = submission->ordinal;
	completion.generation = submission->ordinal / collective->credit_count + 1u;
	/* The callback is a stream-order continuation: the reduction is enqueued,
	 * and the callback may enqueue dependent work on the same stream. */
	submission->completion_function(submission->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkTpDeviceCollectiveNcclSubmitBf16(SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission)
{
	uint64_t element_count;
	if ( collective == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	element_count = (uint64_t)submission->active_sequence_count *
		collective->local_hidden_dimension;
	return(SparkTpNcclSubmitAllReduce(collective,submission,element_count,
		SPARK_TP_NCCL_DATA_TYPE_BF16,SPARK_TP_NCCL_REDUCTION_SUM));
}

SparkStatus SparkTpDeviceCollectiveNcclSubmitU64Max(SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission)
{
	if ( submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkTpNcclSubmitAllReduce(collective,submission,
		submission->active_sequence_count,SPARK_TP_NCCL_DATA_TYPE_U64,
		SPARK_TP_NCCL_REDUCTION_MAX));
}

static SparkStatus SparkTpNcclFailureIsValid(SparkStatus failure_status)
{
	if ( failure_status == SPARK_STATUS_OK || failure_status ==
		SPARK_STATUS_BUSY || failure_status == SPARK_STATUS_PENDING )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkTpDeviceCollectiveNcclRequestFailure(SparkTpDeviceCollective *collective,SparkStatus failure_status)
{
	SparkTpDeviceNcclImplementation *implementation;
	SparkTpNcclResult result;
	if ( collective == 0 || collective->implementation == 0 ||
		collective->backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL ||
		SparkTpNcclFailureIsValid(failure_status) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	implementation = (SparkTpDeviceNcclImplementation *)collective->implementation;
	if ( pthread_mutex_lock(&implementation->mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	atomic_store_explicit(&implementation->admission_open,0u,
		memory_order_release);
	atomic_store_explicit(&implementation->failure_status,failure_status,
		memory_order_release);
	result = SPARK_TP_NCCL_SUCCESS;
	if ( implementation->communicator != 0 &&
		implementation->communicator_aborted == 0u )
	{
		result = implementation->library.comm_abort(implementation->communicator);
		implementation->communicator_aborted = 1u;
	}
	(void)pthread_mutex_unlock(&implementation->mutex);
	if ( result != SPARK_TP_NCCL_SUCCESS )
		SparkTpNcclReportError(collective,implementation,"comm_abort",result);
	return(result == SPARK_TP_NCCL_SUCCESS ? SPARK_STATUS_OK :
		SPARK_STATUS_IO_ERROR);
}

SparkStatus SparkTpDeviceCollectiveNcclRequestOperationFailure(SparkTpDeviceCollective *collective,uint64_t ordinal,SparkStatus failure_status)
{
	SparkTpDeviceNcclImplementation *implementation;
	if ( collective == 0 || collective->implementation == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	implementation = (SparkTpDeviceNcclImplementation *)collective->implementation;
	if ( ordinal < atomic_load_explicit(&implementation->next_ordinal,
		memory_order_acquire) )
		return(SPARK_STATUS_NOT_FOUND);
	return(SparkTpDeviceCollectiveNcclRequestFailure(collective,failure_status));
}

SparkStatus SparkTpDeviceCollectiveNcclOperationPhase(const SparkTpDeviceCollective *collective,uint64_t ordinal,uint32_t *phase_out,uint32_t *failure_requested_out)
{
	const SparkTpDeviceNcclImplementation *implementation;
	if ( collective == 0 || collective->implementation == 0 ||
		phase_out == 0 || failure_requested_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	implementation = (const SparkTpDeviceNcclImplementation *)
		collective->implementation;
	if ( ordinal < atomic_load_explicit(&implementation->next_ordinal,
		memory_order_acquire) )
		return(SPARK_STATUS_NOT_FOUND);
	*phase_out = SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE;
	*failure_requested_out = atomic_load_explicit(
		&implementation->admission_open,memory_order_acquire) == 0u ? 1u : 0u;
	return(SPARK_STATUS_OK);
}

void SparkTpDeviceCollectiveNcclDestroy(SparkTpDeviceCollective *collective)
{
	SparkTpDeviceNcclImplementation *implementation;
	SparkTpNcclResult result;
	if ( collective == 0 )
		return;
	implementation = (SparkTpDeviceNcclImplementation *)collective->implementation;
	if ( implementation == 0 )
	{
		SparkTpNcclResetCollective(collective);
		return;
	}
	result = SPARK_TP_NCCL_SUCCESS;
	if ( implementation->mutex_initialized != 0u )
		(void)pthread_mutex_lock(&implementation->mutex);
	atomic_store_explicit(&implementation->admission_open,0u,
		memory_order_release);
	if ( implementation->communicator != 0 )
	{
		result = implementation->communicator_aborted != 0u ?
			SPARK_TP_NCCL_SUCCESS :
			implementation->library.comm_destroy(implementation->communicator);
		implementation->communicator = 0;
	}
	if ( implementation->mutex_initialized != 0u )
	{
		(void)pthread_mutex_unlock(&implementation->mutex);
		(void)pthread_mutex_destroy(&implementation->mutex);
	}
	if ( result != SPARK_TP_NCCL_SUCCESS )
		SparkTpNcclReportError(collective,implementation,"comm_destroy",result);
	if ( implementation->library.dynamic_library != 0 )
		(void)dlclose(implementation->library.dynamic_library);
	free(implementation);
	SparkTpNcclResetCollective(collective);
}
