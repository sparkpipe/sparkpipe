#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "runtime/model_continuation_lease.h"
#include "spark_filesystem.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_model_resident_ipc.h"
#include "sparkpipe/spark_pipeline_runtime.h"

#define SPARK_MODEL_RESIDENTD_TRANSPORT_POLL_CAPACITY 32u
#define SPARK_MODEL_RESIDENTD_PROGRESS_STEPS 64u
#define SPARK_MODEL_RESIDENTD_QUIESCE_TIMEOUT_NS UINT64_C(5000000000)
#define SPARK_MODEL_RESIDENTD_QUIESCE_POLL_NS 1000000L
#define SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_ROUTE 1u
#define SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_RESIDENCY 2u
#define SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_IDENTITY 3u
#define SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_ACCEPTED_TOKENS 4u
#define SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_STATE 5u
#define SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_STATUS 6u
#define SPARK_MODEL_RESIDENTD_FAILURE_DEACTIVATE_ROUTE 7u
#define SPARK_MODEL_RESIDENTD_FAILURE_CONTINUE_LEASE 8u
#define SPARK_MODEL_RESIDENTD_FAILURE_CLIENT_LEASE_DISCONNECT 9u

typedef struct SparkModelResidentdConfiguration
{
	const SparkModelResidentDeployment *deployment;
	const char *runtime_root;
	const char *socket_path;
	const char *listen_address;
	char adapter_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	char adapter_configuration_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	char driver_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	const char *driver_program_name;
	char transport_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	const char *transport_mode;
	const char *node_target;
	const char *transport_host;
	const char *previous_transport_host;
	const char *next_transport_host;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t previous_rank_index;
	uint32_t next_rank_index;
	uint32_t max_inflight_submission_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
	uint32_t port_base;
	uint32_t listen_port;
} SparkModelResidentdConfiguration;

typedef struct SparkModelResidentdLaunch
{
	const char *deployment_path;
	uint32_t rank_index;
} SparkModelResidentdLaunch;

#define SPARK_MODEL_RESIDENTD_UNSET_UINT32 UINT32_MAX

typedef enum SparkModelResidentdMemoryMode
{
	SPARK_MODEL_RESIDENTD_MEMORY_DEVICE = 1,
	SPARK_MODEL_RESIDENTD_MEMORY_MAPPED_HOST = 2
} SparkModelResidentdMemoryMode;

typedef enum SparkModelResidentdRouteState
{
	SPARK_MODEL_RESIDENTD_ROUTE_IDLE = 0,
	SPARK_MODEL_RESIDENTD_ROUTE_RESERVED = 1,
	SPARK_MODEL_RESIDENTD_ROUTE_READY_INPUT = 2,
	SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT = 3,
	SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER = 4,
	SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER = 5,
	SPARK_MODEL_RESIDENTD_ROUTE_READY_OUTPUT = 6,
	SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT = 7,
	SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION = 8,
	SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING = 9,
	SPARK_MODEL_RESIDENTD_ROUTE_FENCED = 10,
	SPARK_MODEL_RESIDENTD_ROUTE_CONTINUATION_PREPARING = 11
} SparkModelResidentdRouteState;

typedef struct SparkModelResidentdBoundary
{
	void *allocation;
	void *cuda_address;
	uint64_t bytes;
} SparkModelResidentdBoundary;

typedef struct SparkModelResidentdSlot
{
	uint8_t *message;
	SparkModelResidentdBoundary input;
	SparkModelResidentdBoundary output;
} SparkModelResidentdSlot;

typedef struct SparkModelResidentdOutput
{
	uint32_t message_bytes;
	uint32_t sent_bytes;
	uint8_t *message;
} SparkModelResidentdOutput;

typedef struct SparkModelResidentdClient
{
	int32_t fd;
	uint32_t hello_complete;
	uint32_t close_after_output;
	uint8_t *input;
	uint32_t input_bytes;
	uint32_t target_bytes;
	uint32_t output_head;
	uint32_t output_count;
	uint32_t output_capacity;
	uint32_t output_message_capacity;
	uint32_t input_capacity;
	uint64_t generation;
	uint64_t last_message_id;
	uint64_t last_submission_id;
	SparkModelResidentdOutput *output;
	uint8_t *output_storage;
} SparkModelResidentdClient;

typedef struct SparkModelResidentdRoute
{
	uint32_t active;
	uint32_t result_queued;
	uint32_t abandoned;
	uint32_t state;
	uint32_t slot_index;
	uint32_t message_bytes;
	uint32_t ready_state;
	uint32_t decision_required;
	uint32_t resident_slots_claimed;
	uint32_t prepared_cache;
	uint32_t committed_fifo_queued;
	uint32_t committed_fifo_next;
	uint32_t deadline_expired;
	uint32_t deadline_completion_queued;
	uint32_t deadline_wait_state;
	uint64_t message_id;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t client_generation;
	uint64_t adapter_submit_time_ns;
	SparkModelServingSubmission submission;
	SparkHiddenTransportPacket input_packet;
	SparkHiddenTransportPacket output_packet;
	SparkModelServingCompletion completion;
} SparkModelResidentdRoute;

typedef struct SparkModelResidentdSequenceSlot
{
	uint32_t active_owner;
	uint32_t bound;
	uint64_t request_id;
	uint64_t request_generation;
	uint64_t sequence_id;
	SparkModelContinuationLease lease;
} SparkModelResidentdSequenceSlot;

typedef struct SparkModelResidentdRuntime
{
	SparkModelServingAdapterDynamicLibrary adapter_library;
	void *adapter_state;
	SparkHiddenTransportDynamicLibrary transport_library;
	SparkHiddenTransportSession *input_transport;
	SparkHiddenTransportSession *output_transport;
	SparkPipelineRuntimeRankPlan rank_plan;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkModelResidentdClient client;
	SparkModelResidentdRoute *routes;
	SparkModelResidentdSlot *slots;
	SparkModelResidentdSequenceSlot *sequence_slots;
	uint8_t *route_messages;
	uint32_t route_capacity;
	uint32_t route_message_capacity;
	uint32_t next_adapter_route;
	uint32_t committed_fifo_head;
	uint32_t committed_fifo_tail;
	SparkModelResidentdMemoryMode memory_mode;
	cudaStream_t execution_stream;
	cudaStream_t transport_stream;
	int32_t listen_fd;
	int32_t wake_read_fd;
	int32_t wake_write_fd;
	uint32_t control_endpoint_kind;
	uint32_t failed_reason;
	uint32_t failed_route_state;
	uint32_t failed_work_kind;
	uint64_t failed_submission_id;
	const char *initialize_phase;
	atomic_uint failed_status;
	pthread_mutex_t mutex;
} SparkModelResidentdRuntime;

static volatile sig_atomic_t SparkModelResidentdStop;

static uint64_t SparkModelResidentdMonotonicTimeNs(void);

static void SparkModelResidentdFailLocked(
	SparkModelResidentdRuntime *runtime,
	SparkStatus status,
	uint32_t reason,
	const SparkModelResidentdRoute *route)
{
	if ( atomic_load(&runtime->failed_status) != SPARK_STATUS_OK )
		return;
	runtime->failed_reason = reason;
	runtime->failed_route_state = route != 0 ? route->state : 0u;
	runtime->failed_work_kind = route != 0 ? route->submission.work_kind : 0u;
	runtime->failed_submission_id = route != 0 ? route->submission_id : 0u;
	atomic_store(&runtime->failed_status,(uint32_t)status);
	fprintf(stderr, "model_residentd route_failed status=%d reason=%u work_kind=%u submission=%llu\n", (int)status, reason, route != 0 ? route->submission.work_kind : 0u, route != 0 ? (unsigned long long)route->submission_id : 0ull);
}

static void SparkModelResidentdSignal(int32_t signal_number)
{
	(void)signal_number;
	SparkModelResidentdStop = 1;
}

static SparkStatus SparkModelResidentdParseUnsigned(
	const char *text,
	uint32_t *value)
{
	char *end;
	unsigned long parsed;
	if ( text == 0 || text[0] == '\0' || value == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	errno = 0;
	end = 0;
	parsed = strtoul(text,&end,10);
	if ( errno != 0 || end == text || end[0] != '\0' || parsed > UINT32_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*value = (uint32_t)parsed;
	return(SPARK_STATUS_OK);
}

static void SparkModelResidentdUsage(const char *program)
{
	fprintf(stderr,"usage: %s --deployment PATH --rank-index N\n",program);
}

static SparkStatus SparkModelResidentdSetText(
	const char **destination,
	const char *value)
{
	if ( destination == 0 || *destination != 0 || value == 0 || value[0] == '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*destination = value;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdParseLaunch(
	int32_t argument_count,
	char **arguments,
	SparkModelResidentdLaunch *launch)
{
	SparkStatus status;
	int32_t index;
	if ( launch == 0 || argument_count != 5 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(launch,0,sizeof(*launch));
	launch->rank_index = SPARK_MODEL_RESIDENTD_UNSET_UINT32;
	status = SPARK_STATUS_OK;
	for (index=1; status == SPARK_STATUS_OK && index<argument_count; index+=2)
	{
		if ( strcmp(arguments[index],"--deployment") == 0 )
			status = SparkModelResidentdSetText(&launch->deployment_path,arguments[index + 1]);
		else if ( strcmp(arguments[index],"--rank-index") == 0 && launch->rank_index == SPARK_MODEL_RESIDENTD_UNSET_UINT32 )
			status = SparkModelResidentdParseUnsigned(arguments[index + 1],&launch->rank_index);
		else
			status = SPARK_STATUS_INVALID_ARGUMENT;
	}
	return(status == SPARK_STATUS_OK && launch->deployment_path != 0 && launch->rank_index != SPARK_MODEL_RESIDENTD_UNSET_UINT32 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkModelResidentdBuildConfiguration(
	const SparkModelResidentDeployment *deployment,
	uint32_t rank_index,
	SparkModelResidentdConfiguration *configuration)
{
	const SparkModelResidentDeploymentNode *node,*previous,*next;
	SparkStatus status;
	node = SparkModelResidentDeploymentFindRank(deployment,rank_index);
	if ( node == 0 || configuration == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	previous = node->stage_index != 0u ? SparkModelResidentDeploymentFindStage(deployment,node->stage_index - 1u) : 0;
	next = node->stage_index + 1u < deployment->node_count ? SparkModelResidentDeploymentFindStage(deployment,node->stage_index + 1u) : 0;
	if ( (node->stage_index != 0u && previous == 0) || (node->stage_index + 1u < deployment->node_count && next == 0) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	memset(configuration,0,sizeof(*configuration));
	configuration->deployment = deployment;
	configuration->runtime_root = node->runtime_root;
	configuration->socket_path = node->control_endpoint.unix_socket_path;
	configuration->listen_address = node->control_endpoint.tcp_host;
	configuration->listen_port = node->control_endpoint.tcp_port;
	status = SparkModelResidentDeploymentResolvePath(node,deployment->adapter_shared_object_path,configuration->adapter_path,sizeof(configuration->adapter_path));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentResolvePath(node,node->adapter_configuration_path,configuration->adapter_configuration_path,sizeof(configuration->adapter_configuration_path));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentResolvePath(node,deployment->driver_shared_object_path,configuration->driver_path,sizeof(configuration->driver_path));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentResolvePath(node,deployment->transport_shared_object_path,configuration->transport_path,sizeof(configuration->transport_path));
	if ( status != SPARK_STATUS_OK )
		return(status);
	configuration->driver_program_name = deployment->driver_program_name;
	configuration->transport_mode = deployment->transport_mode;
	configuration->node_target = node->node_target;
	configuration->transport_host = node->transport_host;
	configuration->previous_transport_host = previous != 0 ? previous->transport_host : 0;
	configuration->next_transport_host = next != 0 ? next->transport_host : 0;
	configuration->kv_backing_directory = node->kv_backing_directory;
	configuration->kv_backing_maximum_bytes = node->kv_backing_maximum_bytes;
	configuration->rank_index = node->rank_index;
	configuration->stage_index = node->stage_index;
	configuration->previous_rank_index = previous != 0 ? previous->rank_index : SPARK_PIPELINE_RUNTIME_NO_RANK;
	configuration->next_rank_index = next != 0 ? next->rank_index : SPARK_PIPELINE_RUNTIME_NO_RANK;
	configuration->max_inflight_submission_count = deployment->runtime_limits.max_inflight_submission_count;
	configuration->max_active_sequence_count = deployment->runtime_limits.max_active_sequence_count;
	configuration->max_input_row_count = deployment->runtime_limits.max_input_row_count;
	configuration->resident_sequence_capacity = deployment->runtime_limits.resident_sequence_capacity;
	configuration->kv_logical_page_capacity =
		deployment->runtime_limits.kv_logical_page_capacity;
	configuration->kv_physical_page_capacity =
		deployment->runtime_limits.kv_physical_page_capacity;
	configuration->port_base = deployment->transport_control_port_base;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdValidateDirectories(
	const SparkModelResidentdConfiguration *configuration)
{
	if ( configuration == 0 ||
		!SparkPathIsRealDirectoryTree(configuration->runtime_root) )
		return(SPARK_STATUS_IO_ERROR);
	if ( configuration->kv_backing_directory != 0 &&
		!SparkPathIsRealDirectoryTree(configuration->kv_backing_directory) )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdTransportContract(
	const char *mode,
	uint32_t *capabilities,
	const char **module_id,
	SparkModelResidentdMemoryMode *memory_mode)
{
	if ( mode == 0 || capabilities == 0 || module_id == 0 || memory_mode == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( strcmp(mode,"host-rdma") == 0 )
	{
		*capabilities = SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS;
		*module_id = SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID;
		*memory_mode = SPARK_MODEL_RESIDENTD_MEMORY_MAPPED_HOST;
	}
	else if ( strcmp(mode,"gpudirect-rdma") == 0 )
	{
		*capabilities = SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_GPUDIRECT_RDMA_CAPS;
		*module_id = SPARK_HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA_VERBS_MODULE_ID;
		*memory_mode = SPARK_MODEL_RESIDENTD_MEMORY_DEVICE;
	}
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static int32_t SparkModelResidentdSetNonblocking(int32_t fd)
{
	int32_t flags;
	flags = fcntl(fd,F_GETFL,0);
	return(flags < 0 || fcntl(fd,F_SETFL,flags | O_NONBLOCK) != 0 ? -1 : 0);
}

static SparkStatus SparkModelResidentdOpenWakePipe(
	SparkModelResidentdRuntime *runtime)
{
	int32_t fds[2];
	if ( pipe(fds) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( SparkModelResidentdSetNonblocking(fds[0]) != 0 || SparkModelResidentdSetNonblocking(fds[1]) != 0 )
	{
		close(fds[0]);
		close(fds[1]);
		return(SPARK_STATUS_IO_ERROR);
	}
	runtime->wake_read_fd = fds[0];
	runtime->wake_write_fd = fds[1];
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdOpenUnixListenSocket(
	SparkModelResidentdRuntime *runtime,
	const char *socket_path)
{
	struct sockaddr_un address;
	struct stat existing;
	int32_t fd;
	if ( strlen(socket_path) >= sizeof(address.sun_path) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( lstat(socket_path,&existing) == 0 )
	{
		if ( !S_ISSOCK(existing.st_mode) || existing.st_uid != geteuid() )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( unlink(socket_path) != 0 )
			return(SPARK_STATUS_IO_ERROR);
	}
	else if ( errno != ENOENT )
		return(SPARK_STATUS_IO_ERROR);
	fd = socket(AF_UNIX,SOCK_STREAM,0);
	if ( fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	memset(&address,0,sizeof(address));
	address.sun_family = AF_UNIX;
	memcpy(address.sun_path,socket_path,strlen(socket_path) + 1u);
	if ( bind(fd,(const struct sockaddr *)&address,sizeof(address)) != 0 || listen(fd,1) != 0 || SparkModelResidentdSetNonblocking(fd) != 0 )
	{
		close(fd);
		unlink(socket_path);
		return(SPARK_STATUS_IO_ERROR);
	}
	(void)chmod(socket_path,0600);
	runtime->listen_fd = fd;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdOpenTcpListenSocket(
	SparkModelResidentdRuntime *runtime,
	const char *listen_address,
	uint32_t listen_port)
{
	struct addrinfo hints,*addresses,*address;
	char service[16];
	int32_t enabled,fd;
	SparkStatus status;
	memset(&hints,0,sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_NUMERICSERV;
	if ( snprintf(service,sizeof(service),"%u",listen_port) < 0 || getaddrinfo(listen_address,service,&hints,&addresses) != 0 )
		return(SPARK_STATUS_ROUTE_NOT_FOUND);
	status = SPARK_STATUS_IO_ERROR;
	for (address=addresses; address!=0 && status!=SPARK_STATUS_OK; address=address->ai_next)
	{
		fd = socket(address->ai_family,address->ai_socktype,address->ai_protocol);
		if ( fd < 0 )
			continue;
		enabled = 1;
		if ( setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&enabled,sizeof(enabled)) == 0 && bind(fd,address->ai_addr,(socklen_t)address->ai_addrlen) == 0 && listen(fd,1) == 0 && SparkModelResidentdSetNonblocking(fd) == 0 )
		{
			runtime->listen_fd = fd;
			status = SPARK_STATUS_OK;
		}
		else
			close(fd);
	}
	freeaddrinfo(addresses);
	return(status);
}

static SparkStatus SparkModelResidentdOpenControlListener(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration->socket_path != 0 )
	{
		status = SparkModelResidentdOpenUnixListenSocket(runtime,configuration->socket_path);
		if ( status == SPARK_STATUS_OK )
			runtime->control_endpoint_kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
	}
	else
	{
		status = SparkModelResidentdOpenTcpListenSocket(runtime,configuration->listen_address,configuration->listen_port);
		if ( status == SPARK_STATUS_OK )
			runtime->control_endpoint_kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
	}
	return(status);
}

static SparkStatus SparkModelResidentdOpenTransports(
	SparkModelResidentdRuntime *runtime,
	uint32_t required_capabilities)
{
	const SparkModelServingAdapterDescriptor *descriptor;
	SparkHiddenTransportEndpoint endpoint;
	SparkStatus status;
	descriptor = runtime->adapter_library.adapter_interface.descriptor;
	status = SPARK_STATUS_OK;
	if ( (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u )
	{
		status = SparkPipelineRuntimeBuildInputEndpoint(descriptor,&runtime->rank_plan,&endpoint);
		if ( status == SPARK_STATUS_OK )
			status = SparkHiddenTransportOpen(&endpoint,&runtime->transport_library.transport_interface,required_capabilities,&runtime->input_transport);
	}
	if ( status == SPARK_STATUS_OK && (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u )
	{
		status = SparkPipelineRuntimeBuildOutputEndpoint(descriptor,&runtime->rank_plan,&endpoint);
		if ( status == SPARK_STATUS_OK )
			status = SparkHiddenTransportOpen(&endpoint,&runtime->transport_library.transport_interface,required_capabilities,&runtime->output_transport);
	}
	return(status);
}

static SparkStatus SparkModelResidentdAllocateBoundary(
	SparkModelResidentdBoundary *boundary,
	SparkModelResidentdMemoryMode memory_mode,
	uint64_t bytes)
{
	cudaError_t error;
	if ( boundary == 0 || bytes == 0u || bytes > SIZE_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(boundary,0,sizeof(*boundary));
	boundary->bytes = bytes;
	if ( memory_mode == SPARK_MODEL_RESIDENTD_MEMORY_MAPPED_HOST )
	{
		error = cudaHostAlloc(&boundary->allocation,(size_t)bytes,cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&boundary->cuda_address,boundary->allocation,0u);
	}
	else
	{
		error = cudaMalloc(&boundary->allocation,(size_t)bytes);
		boundary->cuda_address = boundary->allocation;
	}
	return(error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void SparkModelResidentdFreeBoundary(
	SparkModelResidentdBoundary *boundary,
	SparkModelResidentdMemoryMode memory_mode)
{
	if ( boundary == 0 )
		return;
	if ( boundary->allocation != 0 && memory_mode == SPARK_MODEL_RESIDENTD_MEMORY_MAPPED_HOST )
		(void)cudaFreeHost(boundary->allocation);
	else if ( boundary->allocation != 0 )
		(void)cudaFree(boundary->allocation);
	memset(boundary,0,sizeof(*boundary));
}

static SparkStatus SparkModelResidentdAllocateCuda(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdMemoryMode memory_mode)
{
	SparkStatus status;
	uint32_t index;
	runtime->memory_mode = memory_mode;
	if ( cudaStreamCreateWithFlags(&runtime->execution_stream,cudaStreamNonBlocking) != cudaSuccess )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( cudaStreamCreateWithFlags(&runtime->transport_stream,cudaStreamNonBlocking) != cudaSuccess )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_OK;
	for (index=0u; status == SPARK_STATUS_OK && index<runtime->route_capacity; index++)
	{
		if ( (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u )
			status = SparkModelResidentdAllocateBoundary(&runtime->slots[index].input,memory_mode,runtime->rank_plan.input_max_packet_bytes);
		if ( status == SPARK_STATUS_OK && (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u )
			status = SparkModelResidentdAllocateBoundary(&runtime->slots[index].output,memory_mode,runtime->rank_plan.output_max_packet_bytes);
	}
	return(status);
}

static uint32_t SparkModelResidentdMaximumU32(uint32_t left,uint32_t right)
{
	return(left > right ? left : right);
}

static SparkStatus SparkModelResidentdAllocateHostStorage(
	SparkModelResidentdRuntime *runtime)
{
	const SparkModelServingAdapterDescriptor *descriptor;
	uint32_t completion_bytes,index,output_bytes,submit_bytes;
	size_t bytes;
	SparkStatus status;
	descriptor = runtime->adapter_library.adapter_interface.descriptor;
	status = SparkModelResidentIpcCalculateSubmitBytes(runtime->runtime_limits.max_active_sequence_count,runtime->runtime_limits.max_input_row_count,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,&submit_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcCalculateCompletionBytes(descriptor->max_output_token_count,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,&completion_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	runtime->route_capacity = runtime->runtime_limits.max_inflight_submission_count;
	runtime->route_message_capacity = submit_bytes;
	runtime->client.input_capacity = SparkModelResidentdMaximumU32(submit_bytes,SPARK_MODEL_RESIDENT_IPC_HELLO_BYTES);
	runtime->client.output_capacity = (2u * runtime->route_capacity) + 2u;
	output_bytes = SparkModelResidentdMaximumU32(completion_bytes,SPARK_MODEL_RESIDENT_IPC_HELLO_ACK_BYTES);
	runtime->client.output_message_capacity = SparkModelResidentdMaximumU32(output_bytes,SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES);
	if ( runtime->route_message_capacity > SPARK_MODEL_RESIDENT_IPC_MAX_MESSAGE_BYTES || runtime->client.input_capacity > SPARK_MODEL_RESIDENT_IPC_MAX_MESSAGE_BYTES || runtime->client.output_message_capacity > SPARK_MODEL_RESIDENT_IPC_MAX_MESSAGE_BYTES )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	runtime->routes = (SparkModelResidentdRoute *)calloc(runtime->route_capacity,sizeof(runtime->routes[0]));
	runtime->slots = (SparkModelResidentdSlot *)calloc(runtime->route_capacity,sizeof(runtime->slots[0]));
	runtime->sequence_slots = (SparkModelResidentdSequenceSlot *)calloc(runtime->runtime_limits.resident_sequence_capacity,sizeof(runtime->sequence_slots[0]));
	runtime->client.output = (SparkModelResidentdOutput *)calloc(runtime->client.output_capacity,sizeof(runtime->client.output[0]));
	bytes = (size_t)runtime->route_capacity * runtime->route_message_capacity;
	runtime->route_messages = (uint8_t *)malloc(bytes);
	bytes = (size_t)runtime->client.output_capacity * runtime->client.output_message_capacity;
	runtime->client.output_storage = (uint8_t *)malloc(bytes);
	runtime->client.input = (uint8_t *)malloc(runtime->client.input_capacity);
	if ( runtime->routes == 0 || runtime->slots == 0 || runtime->sequence_slots == 0 || runtime->route_messages == 0 || runtime->client.output == 0 || runtime->client.output_storage == 0 || runtime->client.input == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<runtime->route_capacity; index++)
		runtime->slots[index].message = runtime->route_messages + ((size_t)index * runtime->route_message_capacity);
	for (index=0u; index<runtime->client.output_capacity; index++)
		runtime->client.output[index].message = runtime->client.output_storage + ((size_t)index * runtime->client.output_message_capacity);
	return(SPARK_STATUS_OK);
}

static void SparkModelResidentdWake(void *wake_context)
{
	SparkModelResidentdRuntime *runtime;
	ssize_t written;
	uint8_t value;
	runtime = (SparkModelResidentdRuntime *)wake_context;
	value = 1u;
	if ( runtime != 0 && runtime->wake_write_fd >= 0 )
	{
		written = write(runtime->wake_write_fd,&value,sizeof(value));
		(void)written;
	}
}

static SparkModelResidentdRoute *SparkModelResidentdFindRoute(
	SparkModelResidentdRuntime *runtime,
	uint64_t submission_id)
{
	uint32_t index;
	for (index=0u; index<runtime->route_capacity; index++)
		if ( runtime->routes[index].active != 0u && runtime->routes[index].submission_id == submission_id )
			return(&runtime->routes[index]);
	return(0);
}

static uint32_t SparkModelResidentdSequenceSlotMatches(
	const SparkModelResidentdSequenceSlot *slot,
	const SparkModelServingLane *lane)
{
	return(slot->bound != 0u && slot->request_id == lane->request_id && slot->request_generation == lane->request_generation && slot->sequence_id == lane->sequence_id ? 1u : 0u);
}

static uint32_t SparkModelResidentdLaneStartsAtPositionZero(
	const SparkModelServingSubmission *submission,
	uint32_t lane_index)
{
	uint32_t row;
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE || submission->lanes[lane_index].sequence_position != 0u )
		return(0u);
	for (row=0u; row<submission->row_count; row++)
		if ( submission->row_lane_indices[row] == lane_index && submission->row_positions[row] == 0u )
			return(1u);
	return(0u);
}

static SparkStatus SparkModelResidentdValidatePersistentSlot(
	const SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdRoute *route,
	uint32_t lane_index)
{
	const SparkModelServingAdapterDescriptor *descriptor;
	const SparkModelResidentdSequenceSlot *slot;
	const SparkModelServingLane *lane;
	descriptor = runtime->adapter_library.adapter_interface.descriptor;
	lane = &route->submission.lanes[lane_index];
	slot = &runtime->sequence_slots[lane->resident_sequence_slot];
	if ( descriptor->resident_sequence_slot_reuse == SPARK_MODEL_SERVING_SLOT_REUSE_NONE )
		return(SPARK_STATUS_OK);
	if ( route->submission.work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		if ( slot->bound == 0u )
			return(SPARK_STATUS_NOT_FOUND);
		return(SparkModelResidentdSequenceSlotMatches(slot,lane) != 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( slot->bound == 0u || SparkModelResidentdSequenceSlotMatches(slot,lane) != 0u )
		return(SPARK_STATUS_OK);
	if ( descriptor->resident_sequence_slot_reuse == SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO && SparkModelResidentdLaneStartsAtPositionZero(&route->submission,lane_index) != 0u )
		return(SPARK_STATUS_OK);
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkModelResidentdClaimResidentSlotsLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkStatus status;
	uint32_t lane,owner,slot;
	if ( route->resident_slots_claimed != 0u )
		return(SPARK_STATUS_DUPLICATE);
	owner = route->slot_index + 1u;
	for (lane=0u; lane<route->submission.active_sequence_count; lane++)
	{
		slot = route->submission.lanes[lane].resident_sequence_slot;
		if ( slot >= runtime->runtime_limits.resident_sequence_capacity || runtime->sequence_slots[slot].active_owner != 0u )
			return(slot < runtime->runtime_limits.resident_sequence_capacity ? SPARK_STATUS_BUSY : SPARK_STATUS_CAPACITY_EXCEEDED);
		status = SparkModelResidentdValidatePersistentSlot(runtime,route,lane);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	for (lane=0u; lane<route->submission.active_sequence_count; lane++)
		runtime->sequence_slots[route->submission.lanes[lane].resident_sequence_slot].active_owner = owner;
	route->resident_slots_claimed = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdReleaseResidentSlotsLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	uint32_t lane,owner,slot;
	if ( route->resident_slots_claimed == 0u )
		return(SPARK_STATUS_OK);
	owner = route->slot_index + 1u;
	for (lane=0u; lane<route->submission.active_sequence_count; lane++)
	{
		slot = route->submission.lanes[lane].resident_sequence_slot;
		if ( slot >= runtime->runtime_limits.resident_sequence_capacity || runtime->sequence_slots[slot].active_owner != owner )
			return(SPARK_STATUS_INTERNAL_ERROR);
	}
	for (lane=0u; lane<route->submission.active_sequence_count; lane++)
		runtime->sequence_slots[route->submission.lanes[lane].resident_sequence_slot].active_owner = 0u;
	route->resident_slots_claimed = 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdCompleteContinuationLease(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route,
	const SparkModelServingLane *lane,
	SparkModelResidentdSequenceSlot *slot)
{
	SparkStatus status;
	uint64_t next_sequence_position;
	if ( route->submission.work_kind ==
		SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		SparkModelContinuationLeaseInvalidate(&slot->lease);
		return(SPARK_STATUS_OK);
	}
	next_sequence_position = lane->context_token_count;
	if ( route->submission.work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
	{
		/* The lease must land on the position the engine actually advanced
		 * to: the completion's EMITTED count (1 + accepted), not the
		 * coordinator-rank chain width (spec bucket = 8). A partial-accept
		 * verify burst emits fewer tokens than the admitted chain; the
		 * batch side (c8f76e5) already mirrors this, so both leases must
		 * advance by the same count. */
		/* accepted_token_count already includes the anchor (the module sets it
		 * to 1 + accepted), so it IS the emitted count — no +1. */
		uint32_t completed_tokens = route->completion.accepted_token_count;
		if ( completed_tokens == 0u ||
			(completed_tokens > route->completion.tokens_per_sequence &&
			 route->completion.tokens_per_sequence != 0u) )
			completed_tokens = route->completion.tokens_per_sequence;
		status = SparkModelContinuationLeaseDecodePosition(
			lane->context_token_count,
			completed_tokens,&next_sequence_position);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	/* The lease must be fenced by the CURRENT client generation, not the
		 * route's (captured at reservation): the ASYNC verify completion can
		 * land after a reconnect, which would leave the route's generation
		 * stale and reject the next continuation. */
	return(SparkModelContinuationLeaseEstablish(&slot->lease,
		runtime->client.generation,route->submission.control_generation,
		next_sequence_position,lane->step_generation));
}

static SparkStatus SparkModelResidentdCompleteResidentSlotsLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	const SparkModelServingAdapterDescriptor *descriptor;
	SparkModelResidentdSequenceSlot *slot;
	const SparkModelServingLane *lane;
	SparkStatus status;
	uint32_t lane_index,owner;
	if ( route->resident_slots_claimed == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	descriptor = runtime->adapter_library.adapter_interface.descriptor;
	owner = route->slot_index + 1u;
	for (lane_index=0u; lane_index<route->submission.active_sequence_count; lane_index++)
	{
		lane = &route->submission.lanes[lane_index];
		slot = &runtime->sequence_slots[lane->resident_sequence_slot];
		if ( slot->active_owner != owner )
			return(SPARK_STATUS_INTERNAL_ERROR);
		status = SparkModelResidentdValidatePersistentSlot(runtime,route,lane_index);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	for (lane_index=0u; lane_index<route->submission.active_sequence_count; lane_index++)
	{
		lane = &route->submission.lanes[lane_index];
		slot = &runtime->sequence_slots[lane->resident_sequence_slot];
		if ( descriptor->resident_sequence_slot_reuse != SPARK_MODEL_SERVING_SLOT_REUSE_NONE )
		{
			slot->bound = route->submission.work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE ? 1u : 0u;
			slot->request_id = slot->bound != 0u ? lane->request_id : 0u;
			slot->request_generation = slot->bound != 0u ? lane->request_generation : 0u;
			slot->sequence_id = slot->bound != 0u ? lane->sequence_id : 0u;
		}
		if ( (descriptor->capability_flags &
			SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) != 0u )
		{
			status = SparkModelResidentdCompleteContinuationLease(runtime,route,lane,slot);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
		slot->active_owner = 0u;
	}
	route->resident_slots_claimed = 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdEnqueueCommittedLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	uint32_t encoded_index;
	if ( route->committed_fifo_queued != 0u )
		return(SPARK_STATUS_DUPLICATE);
	encoded_index = route->slot_index + 1u;
	if ( runtime->committed_fifo_tail != 0u )
		runtime->routes[runtime->committed_fifo_tail - 1u].
			committed_fifo_next = encoded_index;
	else
		runtime->committed_fifo_head = encoded_index;
	runtime->committed_fifo_tail = encoded_index;
	route->committed_fifo_queued = 1u;
	route->committed_fifo_next = 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdRemoveCommittedLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkModelResidentdRoute *previous;
	uint32_t encoded_index,index,next,previous_index;
	if ( route->committed_fifo_queued == 0u )
		return(SPARK_STATUS_OK);
	encoded_index = route->slot_index + 1u;
	previous_index = 0u;
	index = runtime->committed_fifo_head;
	while ( index != 0u && index != encoded_index )
	{
		previous_index = index;
		index = runtime->routes[index - 1u].committed_fifo_next;
	}
	if ( index == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	next = route->committed_fifo_next;
	if ( previous_index != 0u )
	{
		previous = &runtime->routes[previous_index - 1u];
		previous->committed_fifo_next = next;
	}
	else
		runtime->committed_fifo_head = next;
	if ( runtime->committed_fifo_tail == encoded_index )
		runtime->committed_fifo_tail = previous_index;
	route->committed_fifo_queued = 0u;
	route->committed_fifo_next = 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdDeactivateRouteLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkStatus status;
	status = SparkModelResidentdRemoveCommittedLocked(runtime,route);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdReleaseResidentSlotsLocked(runtime,route);
	if ( status == SPARK_STATUS_OK )
	{
		route->active = 0u;
		route->state = SPARK_MODEL_RESIDENTD_ROUTE_IDLE;
	}
	return(status);
}

static SparkStatus SparkModelResidentdQueueRawLocked(
	SparkModelResidentdRuntime *runtime,
	const void *message,
	uint32_t message_bytes)
{
	SparkModelResidentdOutput *output;
	uint32_t index;
	if ( message == 0 || message_bytes == 0u || message_bytes > runtime->client.output_message_capacity || runtime->client.output_count >= runtime->client.output_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	index = (runtime->client.output_head + runtime->client.output_count) % runtime->client.output_capacity;
	output = &runtime->client.output[index];
	memcpy(output->message,message,message_bytes);
	output->message_bytes = message_bytes;
	output->sent_bytes = 0u;
	runtime->client.output_count++;
	return(SPARK_STATUS_OK);
}

/*
 * A transport timeout is client-visible immediately, but the route and its
 * boundary allocation stay quarantined until transport reports a terminal
 * completion. The one-shot PP API has no safe cancellation/reclaim contract.
 */
static SparkStatus SparkModelResidentdQueueDeadlineCompletionLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkModelResidentdOutput *output;
	SparkModelServingCompletion completion;
	uint32_t index,message_bytes;
	SparkStatus status;
	if ( route->deadline_completion_queued != 0u || route->abandoned != 0u ||
		runtime->client.fd < 0 ||
		route->client_generation != runtime->client.generation )
		return(SPARK_STATUS_OK);
	if ( route->result_queued == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( runtime->client.output_count >= runtime->client.output_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = SPARK_STATUS_IO_ERROR;
	completion.submission_id = route->submission.submission_id;
	completion.request_id = route->submission.request_id;
	completion.sequence_id = route->submission.sequence_id;
	completion.sequence_position = route->submission.sequence_position;
	completion.control_generation = route->submission.control_generation;
	completion.transaction_id = route->submission.transaction_id;
	completion.dispatch_generation = route->submission.dispatch_generation;
	completion.request_generation = route->submission.request_generation;
	completion.step_generation = route->submission.step_generation;
	completion.residency = route->submission.residency;
	index = (runtime->client.output_head + runtime->client.output_count) %
		runtime->client.output_capacity;
	output = &runtime->client.output[index];
	status = SparkModelResidentIpcEncodeCompletion(&completion,route->message_id,
		output->message,runtime->client.output_message_capacity,&message_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	output->message_bytes = message_bytes;
	output->sent_bytes = 0u;
	runtime->client.output_count++;
	route->deadline_completion_queued = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdExpireTransportRouteLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route,
	uint32_t wait_state)
{
	uint64_t now;
	if ( route->deadline_expired == 0u )
	{
		if ( route->submission.deadline_time_ns == 0u )
			return(SPARK_STATUS_OK);
		now = SparkModelResidentdMonotonicTimeNs();
		if ( now == 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		if ( now < route->submission.deadline_time_ns )
			return(SPARK_STATUS_OK);
		route->deadline_expired = 1u;
		route->deadline_wait_state = wait_state;
	}
	return(SparkModelResidentdQueueDeadlineCompletionLocked(runtime,route));
}

static SparkStatus SparkModelResidentdQueueCompletionLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkModelResidentdOutput *output;
	uint32_t index,message_bytes;
	SparkStatus status;
	if ( runtime->client.output_count >= runtime->client.output_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	index = (runtime->client.output_head + runtime->client.output_count) % runtime->client.output_capacity;
	output = &runtime->client.output[index];
	status = SparkModelResidentIpcEncodeCompletion(&route->completion,route->message_id,output->message,runtime->client.output_message_capacity,&message_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdCompleteResidentSlotsLocked(runtime,route);
	if ( status != SPARK_STATUS_OK )
		return(status);
	output->message_bytes = message_bytes;
	output->sent_bytes = 0u;
	runtime->client.output_count++;
	route->active = 0u;
	route->state = SPARK_MODEL_RESIDENTD_ROUTE_IDLE;
	return(SPARK_STATUS_OK);
}

static void SparkModelResidentdCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	SparkModelResidentdRuntime *runtime;
	SparkModelResidentdRoute *route;
	SparkStatus status;
	uint64_t completed_time_ns;
	uint32_t failure_reason;
	runtime = (SparkModelResidentdRuntime *)completion_context;
	if ( runtime == 0 || completion == 0 )
		return;
	pthread_mutex_lock(&runtime->mutex);
	route = SparkModelResidentdFindRoute(runtime,completion->submission_id);
	status = route == 0 ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_OK;
	failure_reason = route == 0 ? SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_ROUTE : 0u;
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkModelServingAdapterValidateCompletionResidency(runtime->adapter_library.adapter_interface.descriptor,&route->submission.residency,completion);
		if ( status != SPARK_STATUS_OK )
		{
			const unsigned char *expected = (const unsigned char *)&route->submission.residency;
			const unsigned char *actual = (const unsigned char *)&completion->residency;
			failure_reason = SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_RESIDENCY;
			/* Print the FULL 32-byte tokens so a word1/generation/owner mismatch is
			 * attributed in one line instead of looking like a clean echo (word0 is
			 * the submission id and almost always matches). */
			fprintf(stderr,"model_residentd residency_mismatch submission=%llu "
				"expected=%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x "
				"actual=%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",
				(unsigned long long)route->submission.submission_id,
				expected[0],expected[1],expected[2],expected[3],expected[4],expected[5],expected[6],expected[7],
				expected[8],expected[9],expected[10],expected[11],expected[12],expected[13],expected[14],expected[15],
				expected[16],expected[17],expected[18],expected[19],expected[20],expected[21],expected[22],expected[23],
				expected[24],expected[25],expected[26],expected[27],expected[28],expected[29],expected[30],expected[31],
				actual[0],actual[1],actual[2],actual[3],actual[4],actual[5],actual[6],actual[7],
				actual[8],actual[9],actual[10],actual[11],actual[12],actual[13],actual[14],actual[15],
				actual[16],actual[17],actual[18],actual[19],actual[20],actual[21],actual[22],actual[23],
				actual[24],actual[25],actual[26],actual[27],actual[28],actual[29],actual[30],actual[31]);
		}
	}
	if ( status == SPARK_STATUS_OK && (route->request_id != completion->request_id || route->sequence_id != completion->sequence_id || route->sequence_position != completion->sequence_position || route->submission.control_generation != completion->control_generation || route->submission.transaction_id != completion->transaction_id || route->submission.dispatch_generation != completion->dispatch_generation || route->submission.request_generation != completion->request_generation || route->submission.step_generation != completion->step_generation) )
	{
		status = SPARK_STATUS_SCHEMA_ERROR;
		failure_reason = SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_IDENTITY;
	}
	if ( status == SPARK_STATUS_OK && completion->accepted_token_count > route->submission.new_token_count + runtime->adapter_library.adapter_interface.descriptor->max_speculative_token_count )
	{
		status = SPARK_STATUS_SCHEMA_ERROR;
		failure_reason = SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_ACCEPTED_TOKENS;
	}
	if ( status == SPARK_STATUS_OK && route->state != SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER )
	{
		status = SPARK_STATUS_SCHEMA_ERROR;
		failure_reason = SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_STATE;
	}
	if ( status == SPARK_STATUS_OK && completion->status != SPARK_STATUS_OK )
	{
		status = (SparkStatus)completion->status;
		failure_reason = SPARK_MODEL_RESIDENTD_FAILURE_COMPLETION_STATUS;
	}
	if ( route != 0 && status == SPARK_STATUS_OK )
	{
		route->completion = *completion;
		if ( route->completion.service_time_ns == 0u && route->adapter_submit_time_ns != 0u )
		{
			completed_time_ns = SparkModelResidentdMonotonicTimeNs();
			if ( completed_time_ns >= route->adapter_submit_time_ns )
				route->completion.service_time_ns = completed_time_ns - route->adapter_submit_time_ns;
		}
		route->state = route->submission.work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE && (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u ?
			SPARK_MODEL_RESIDENTD_ROUTE_READY_OUTPUT :
			SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION;
	}
	if ( status != SPARK_STATUS_OK )
		SparkModelResidentdFailLocked(runtime,status,failure_reason,route);
	pthread_mutex_unlock(&runtime->mutex);
	SparkModelResidentdWake(runtime);
}

static SparkStatus SparkModelResidentdInitializeAdapter(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration)
{
	SparkModelServingAdapterConfiguration adapter_configuration;
	memset(&adapter_configuration,0,sizeof(adapter_configuration));
	adapter_configuration.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	adapter_configuration.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	adapter_configuration.rank_index = configuration->rank_index;
	adapter_configuration.stage_index = configuration->stage_index;
	adapter_configuration.runtime_limits = runtime->runtime_limits;
	adapter_configuration.runtime_root = configuration->runtime_root;
	adapter_configuration.node_id = runtime->rank_plan.host_name;
	adapter_configuration.node_target = configuration->node_target;
	adapter_configuration.adapter_configuration_path = configuration->adapter_configuration_path;
	adapter_configuration.driver_shared_object_path = configuration->driver_path;
	adapter_configuration.driver_program_name = configuration->driver_program_name;
	adapter_configuration.kv_backing_directory =
		configuration->kv_backing_directory;
	adapter_configuration.kv_backing_maximum_bytes =
		configuration->kv_backing_maximum_bytes;
	adapter_configuration.execution_stream = runtime->execution_stream;
	adapter_configuration.completion_function = SparkModelResidentdCompletion;
	adapter_configuration.completion_context = runtime;
	adapter_configuration.wake_function = SparkModelResidentdWake;
	adapter_configuration.wake_context = runtime;
	return(runtime->adapter_library.adapter_interface.initialize(&adapter_configuration,&runtime->adapter_state));
}

static void SparkModelResidentdResetRuntime(SparkModelResidentdRuntime *runtime)
{
	memset(runtime,0,sizeof(*runtime));
	runtime->listen_fd = -1;
	runtime->wake_read_fd = -1;
	runtime->wake_write_fd = -1;
	runtime->client.fd = -1;
	atomic_init(&runtime->failed_status,SPARK_STATUS_OK);
	pthread_mutex_init(&runtime->mutex,0);
}

static SparkStatus SparkModelResidentdInitializeLimits(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration)
{
	runtime->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	runtime->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	runtime->runtime_limits.max_inflight_submission_count = configuration->max_inflight_submission_count;
	runtime->runtime_limits.max_active_sequence_count = configuration->max_active_sequence_count;
	runtime->runtime_limits.max_input_row_count = configuration->max_input_row_count;
	runtime->runtime_limits.resident_sequence_capacity = configuration->resident_sequence_capacity;
	runtime->runtime_limits.kv_logical_page_capacity =
		configuration->kv_logical_page_capacity;
	runtime->runtime_limits.kv_physical_page_capacity =
		configuration->kv_physical_page_capacity;
	return(SparkModelServingAdapterValidateRuntimeLimits(runtime->adapter_library.adapter_interface.descriptor,&runtime->runtime_limits));
}

static SparkStatus SparkModelResidentdBuildRankPlan(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration,
	uint32_t transport_capabilities,
	const char *transport_module_id)
{
	const SparkModelServingAdapterDescriptor *descriptor;
	const SparkModelResidentDeploymentNode *next,*previous;
	SparkPipelineRuntimeLinearNode linear_node;
	SparkPipelineRuntimeFanoutNode fanout_node;
	uint32_t hybrid,group_size;
	descriptor = runtime->adapter_library.adapter_interface.descriptor;
	hybrid = (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP) != 0u ? 1u : 0u;
	if ( hybrid == 0u && (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT) != 0u )
	{
		memset(&fanout_node,0,sizeof(fanout_node));
		fanout_node.abi_version = SPARK_PIPELINE_RUNTIME_ABI_VERSION;
		fanout_node.descriptor_bytes = SPARK_PIPELINE_RUNTIME_FANOUT_NODE_BYTES;
		fanout_node.rank_index = configuration->rank_index;
		fanout_node.stage_index = configuration->stage_index;
		fanout_node.stage_count = configuration->deployment->node_count;
		fanout_node.host_name = configuration->transport_host;
		return(SparkPipelineRuntimeBuildFanoutRankPlan(descriptor,&fanout_node,runtime->runtime_limits.max_active_sequence_count,runtime->runtime_limits.max_input_row_count,0u,&runtime->rank_plan));
	}
	memset(&linear_node,0,sizeof(linear_node));
	linear_node.abi_version = SPARK_PIPELINE_RUNTIME_ABI_VERSION;
	linear_node.descriptor_bytes = SPARK_PIPELINE_RUNTIME_LINEAR_NODE_BYTES;
	linear_node.rank_index = configuration->rank_index;
	linear_node.stage_index = configuration->stage_index;
	linear_node.stage_count = configuration->deployment->node_count;
	linear_node.host_name = configuration->transport_host;
	if ( hybrid != 0u )
	{
		group_size = descriptor->parallel_group_size;
		previous = configuration->rank_index >= group_size ?
			SparkModelResidentDeploymentFindRank(configuration->deployment,
				configuration->rank_index - group_size) : 0;
		next = configuration->rank_index + group_size < descriptor->stage_count ?
			SparkModelResidentDeploymentFindRank(configuration->deployment,
				configuration->rank_index + group_size) : 0;
		if ( (configuration->rank_index >= group_size && previous == 0) ||
			(configuration->rank_index + group_size < descriptor->stage_count &&
			 next == 0) )
			return(SPARK_STATUS_SCHEMA_ERROR);
		linear_node.previous_rank_index = previous != 0 ? previous->rank_index :
			SPARK_PIPELINE_RUNTIME_NO_RANK;
		linear_node.next_rank_index = next != 0 ? next->rank_index :
			SPARK_PIPELINE_RUNTIME_NO_RANK;
		linear_node.previous_host_name = previous != 0 ? previous->transport_host : 0;
		linear_node.next_host_name = next != 0 ? next->transport_host : 0;
		return(SparkPipelineRuntimeBuildHybridRankPlan(descriptor,&linear_node,
			runtime->runtime_limits.max_active_sequence_count,
			runtime->runtime_limits.max_input_row_count,transport_capabilities,
			configuration->port_base,transport_module_id,&runtime->rank_plan));
	}
	linear_node.previous_rank_index = configuration->previous_rank_index;
	linear_node.next_rank_index = configuration->next_rank_index;
	linear_node.previous_host_name = configuration->previous_transport_host;
	linear_node.next_host_name = configuration->next_transport_host;
	return(SparkPipelineRuntimeBuildLinearRankPlan(descriptor,&linear_node,runtime->runtime_limits.max_active_sequence_count,runtime->runtime_limits.max_input_row_count,transport_capabilities,configuration->port_base,transport_module_id,&runtime->rank_plan));
}

static SparkStatus SparkModelResidentdInitializePlan(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration,
	uint32_t *transport_capabilities,
	const char **transport_module_id,
	SparkModelResidentdMemoryMode *memory_mode)
{
	SparkStatus status;
	runtime->initialize_phase = "adapter_load";
	status = SparkModelServingAdapterLoadInterfaceFromSharedObject(configuration->adapter_path,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE,&runtime->adapter_library);
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "deployment_validation";
		status = SparkModelResidentDeploymentValidateForAdapter(configuration->deployment,runtime->adapter_library.adapter_interface.descriptor);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "runtime_limits";
		status = SparkModelResidentdInitializeLimits(runtime,configuration);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "transport_contract";
		if ( (runtime->adapter_library.adapter_interface.descriptor->capability_flags &
			(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
			 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP)) ==
			SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT )
		{
			*transport_capabilities = 0u;
			*transport_module_id = 0;
			*memory_mode = SPARK_MODEL_RESIDENTD_MEMORY_DEVICE;
			status = SPARK_STATUS_OK;
		}
		else
			status = SparkModelResidentdTransportContract(configuration->transport_mode,transport_capabilities,transport_module_id,memory_mode);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "rank_plan";
		status = SparkModelResidentdBuildRankPlan(runtime,configuration,*transport_capabilities,*transport_module_id);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "transport_load";
		if ( (runtime->adapter_library.adapter_interface.descriptor->capability_flags &
			(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
			 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP)) ==
			SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT )
			status = SPARK_STATUS_OK;
		else
			status = SparkHiddenTransportLoadInterfaceFromSharedObject(configuration->transport_path,*transport_capabilities,&runtime->transport_library);
	}
	return(status);
}

static SparkStatus SparkModelResidentdInitializeResources(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration,
	uint32_t transport_capabilities,
	SparkModelResidentdMemoryMode memory_mode)
{
	SparkStatus status;
	status = SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "transport_open";
		status = SparkModelResidentdOpenTransports(runtime,transport_capabilities);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "host_storage";
		status = SparkModelResidentdAllocateHostStorage(runtime);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "cuda_storage";
		status = SparkModelResidentdAllocateCuda(runtime,memory_mode);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "wake_pipe";
		status = SparkModelResidentdOpenWakePipe(runtime);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "adapter_initialize";
		status = SparkModelResidentdInitializeAdapter(runtime,configuration);
	}
	if ( status == SPARK_STATUS_OK )
	{
		runtime->initialize_phase = "control_listener";
		status = SparkModelResidentdOpenControlListener(runtime,configuration);
	}
	return(status);
}

static SparkStatus SparkModelResidentdInitialize(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration)
{
	uint32_t transport_capabilities;
	const char *transport_module_id;
	SparkModelResidentdMemoryMode memory_mode;
	SparkStatus status;
	SparkModelResidentdResetRuntime(runtime);
	status = SparkModelResidentdInitializePlan(runtime,configuration,&transport_capabilities,&transport_module_id,&memory_mode);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdInitializeResources(runtime,configuration,transport_capabilities,memory_mode);
	return(status);
}

static void SparkModelResidentdCloseClientLocked(
	SparkModelResidentdRuntime *runtime)
{
	uint32_t index,live_lease;
	live_lease = 0u;
	if ( runtime->sequence_slots != 0 )
		for (index=0u; index<runtime->runtime_limits.resident_sequence_capacity;
			index++)
			if ( runtime->sequence_slots[index].lease.
				lease_client_generation == runtime->client.generation )
			{
				live_lease = 1u;
				SparkModelContinuationLeaseInvalidate(
					&runtime->sequence_slots[index].lease);
			}
	if ( live_lease != 0u && SparkModelResidentdStop == 0 )
		SparkModelResidentdFailLocked(runtime,SPARK_STATUS_IO_ERROR,
			SPARK_MODEL_RESIDENTD_FAILURE_CLIENT_LEASE_DISCONNECT,0);
	if ( runtime->client.fd >= 0 )
		close(runtime->client.fd);
	runtime->client.fd = -1;
	runtime->client.hello_complete = 0u;
	runtime->client.close_after_output = 0u;
	runtime->client.input_bytes = 0u;
	runtime->client.target_bytes = SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES;
	runtime->client.last_submission_id = 0u;
	runtime->client.last_message_id = 0u;
	runtime->client.output_head = 0u;
	runtime->client.output_count = 0u;
	if ( runtime->routes != 0 )
		for (index=0u; index<runtime->route_capacity; index++)
			if ( runtime->routes[index].active != 0u && runtime->routes[index].client_generation == runtime->client.generation )
				if ( runtime->routes[index].state !=
					SPARK_MODEL_RESIDENTD_ROUTE_RESERVED )
					runtime->routes[index].abandoned = 1u;
}

static void SparkModelResidentdCloseClient(SparkModelResidentdRuntime *runtime)
{
	SparkModelResidentdRoute *route;
	SparkStatus cleanup_status,status;
	uint64_t client_generation;
	uint32_t index;
	pthread_mutex_lock(&runtime->mutex);
	client_generation = runtime->client.generation;
	SparkModelResidentdCloseClientLocked(runtime);
	pthread_mutex_unlock(&runtime->mutex);
	if ( runtime->routes == 0 )
		return;
	for (index=0u; index<runtime->route_capacity; index++)
	{
		pthread_mutex_lock(&runtime->mutex);
		route = &runtime->routes[index];
		if ( route->active == 0u ||
			route->client_generation != client_generation ||
			route->state != SPARK_MODEL_RESIDENTD_ROUTE_RESERVED )
		{
			pthread_mutex_unlock(&runtime->mutex);
			continue;
		}
		route->state = SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING;
		pthread_mutex_unlock(&runtime->mutex);
		status = SparkModelServingAdapterResolvePrefetch(
			&runtime->adapter_library.adapter_interface,runtime->adapter_state,
			&route->submission,SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT);
		pthread_mutex_lock(&runtime->mutex);
		if ( route->active != 0u &&
			route->client_generation == client_generation &&
			route->state == SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING )
		{
			if ( status == SPARK_STATUS_OK )
			{
				route->prepared_cache = 0u;
				cleanup_status = SparkModelResidentdDeactivateRouteLocked(
					runtime,route);
				if ( cleanup_status != SPARK_STATUS_OK )
					SparkModelResidentdFailLocked(runtime,cleanup_status,
						SPARK_MODEL_RESIDENTD_FAILURE_DEACTIVATE_ROUTE,route);
			}
			else
			{
				route->state = SPARK_MODEL_RESIDENTD_ROUTE_FENCED;
				route->abandoned = 1u;
			}
		}
		pthread_mutex_unlock(&runtime->mutex);
	}
}

static uint64_t SparkModelResidentdMonotonicTimeNs(void)
{
	struct timespec timestamp;
	if ( clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0 )
		return(0u);
	return(((uint64_t)timestamp.tv_sec * UINT64_C(1000000000)) + (uint64_t)timestamp.tv_nsec);
}

static SparkStatus SparkModelResidentdQuiesceAdapter(
	SparkModelResidentdRuntime *runtime)
{
	const struct timespec sleep_time = {0,SPARK_MODEL_RESIDENTD_QUIESCE_POLL_NS};
	uint64_t deadline,now;
	SparkStatus progress_status,quiesce_status;
	if ( runtime->adapter_state == 0 )
		quiesce_status = SPARK_STATUS_OK;
	else
	{
		now = SparkModelResidentdMonotonicTimeNs();
		if ( now == 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		deadline = now + SPARK_MODEL_RESIDENTD_QUIESCE_TIMEOUT_NS;
		for (;;)
		{
			quiesce_status = runtime->adapter_library.adapter_interface.quiesce(runtime->adapter_state,deadline);
			if ( quiesce_status == SPARK_STATUS_OK )
				break;
			if ( quiesce_status != SPARK_STATUS_BUSY && quiesce_status != SPARK_STATUS_PENDING )
				return(quiesce_status);
			progress_status = runtime->adapter_library.adapter_interface.progress(runtime->adapter_state,SPARK_MODEL_RESIDENTD_PROGRESS_STEPS);
			if ( progress_status != SPARK_STATUS_OK && progress_status != SPARK_STATUS_BUSY && progress_status != SPARK_STATUS_PENDING )
				return(progress_status);
			(void)nanosleep(&sleep_time,0);
			now = SparkModelResidentdMonotonicTimeNs();
			if ( now == 0u )
				return(SPARK_STATUS_INTERNAL_ERROR);
			if ( now >= deadline )
				return(SPARK_STATUS_BUSY);
		}
	}
	if ( runtime->execution_stream != 0 && cudaStreamSynchronize(runtime->execution_stream) != cudaSuccess )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( runtime->transport_stream != 0 && cudaStreamSynchronize(runtime->transport_stream) != cudaSuccess )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkModelResidentdDestroy(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentdConfiguration *configuration)
{
	SparkStatus status;
	uint32_t index;
	if ( runtime == 0 )
		return;
	SparkModelResidentdCloseClient(runtime);
	if ( runtime->listen_fd >= 0 )
		close(runtime->listen_fd);
	if ( configuration != 0 && runtime->control_endpoint_kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX && configuration->socket_path != 0 )
		unlink(configuration->socket_path);
	status = SparkModelResidentdQuiesceAdapter(runtime);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"model_residentd quiesce=%s; preserving live resources until process exit\n",SparkStatusToString(status));
		return;
	}
	SparkHiddenTransportClose(runtime->input_transport);
	SparkHiddenTransportClose(runtime->output_transport);
	runtime->input_transport = 0;
	runtime->output_transport = 0;
	SparkHiddenTransportUnloadInterface(&runtime->transport_library);
	if ( runtime->adapter_library.adapter_interface.destroy != 0 && runtime->adapter_state != 0 )
		runtime->adapter_library.adapter_interface.destroy(runtime->adapter_state);
	runtime->adapter_state = 0;
	SparkModelServingAdapterUnloadInterface(&runtime->adapter_library);
	if ( runtime->slots != 0 )
		for (index=0u; index<runtime->route_capacity; index++)
		{
			SparkModelResidentdFreeBoundary(&runtime->slots[index].input,runtime->memory_mode);
			SparkModelResidentdFreeBoundary(&runtime->slots[index].output,runtime->memory_mode);
		}
	if ( runtime->execution_stream != 0 )
		(void)cudaStreamDestroy(runtime->execution_stream);
	if ( runtime->transport_stream != 0 )
		(void)cudaStreamDestroy(runtime->transport_stream);
	if ( runtime->wake_read_fd >= 0 )
		close(runtime->wake_read_fd);
	if ( runtime->wake_write_fd >= 0 )
		close(runtime->wake_write_fd);
	free(runtime->client.input);
	free(runtime->client.output_storage);
	free(runtime->client.output);
	free(runtime->route_messages);
	free(runtime->sequence_slots);
	free(runtime->slots);
	free(runtime->routes);
	pthread_mutex_destroy(&runtime->mutex);
}

static void SparkModelResidentdAcceptClient(SparkModelResidentdRuntime *runtime)
{
	int32_t fd;
	if ( runtime->client.fd >= 0 )
		return;
	fd = accept(runtime->listen_fd,0,0);
	if ( fd < 0 )
		return;
	if ( SparkModelResidentdSetNonblocking(fd) != 0 )
	{
		close(fd);
		return;
	}
	if ( runtime->control_endpoint_kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP )
	{
		int32_t enabled;
		enabled = 1;
		if ( setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&enabled,sizeof(enabled)) != 0 )
		{
			close(fd);
			return;
		}
	}
	pthread_mutex_lock(&runtime->mutex);
	runtime->client.fd = fd;
	runtime->client.target_bytes = SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES;
	runtime->client.generation++;
	if ( runtime->client.generation == 0u )
		runtime->client.generation = 1u;
	pthread_mutex_unlock(&runtime->mutex);
}

static SparkModelResidentdRoute *SparkModelResidentdReserveRoute(
	SparkModelResidentdRuntime *runtime,
	const SparkModelServingSubmission *submission,
	uint64_t message_id)
{
	SparkModelResidentdRoute *route;
	uint32_t index;
	if ( SparkModelResidentdFindRoute(runtime,submission->submission_id) != 0 )
		return(0);
	for (index=0u; index<runtime->route_capacity; index++)
	{
		route = &runtime->routes[index];
		if ( route->active == 0u )
		{
			memset(route,0,sizeof(*route));
			route->active = 1u;
			route->slot_index = index;
			route->message_id = message_id;
			route->submission_id = submission->submission_id;
			route->request_id = submission->request_id;
			route->sequence_id = submission->sequence_id;
			route->sequence_position = submission->sequence_position;
			route->client_generation = runtime->client.generation;
			return(route);
		}
	}
	return(0);
}

static void SparkModelResidentdInitializePacket(
	SparkModelResidentdRuntime *runtime,
	SparkHiddenTransportPacket *packet,
	const SparkModelServingSubmission *submission,
	const void *hidden_bf16,
	const void *sideband_payload,
	uint32_t sideband_kind,
	uint32_t sideband_bytes_per_sequence)
{
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
		SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = submission->row_count;
	packet->hidden_dimension = runtime->rank_plan.boundary_element_count;
	packet->bytes_per_sequence = (uint32_t)runtime->rank_plan.boundary_bytes_per_sequence;
	packet->sequence_id = submission->submission_id;
	packet->token_index = submission->dispatch_generation;
	packet->hidden_bf16 = hidden_bf16;
	packet->cuda_stream = runtime->transport_stream;
	if ( sideband_bytes_per_sequence != 0u )
	{
		packet->flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
		packet->sideband_payload = sideband_payload;
		packet->sideband_kind = sideband_kind;
		packet->sideband_bytes_per_sequence = sideband_bytes_per_sequence;
	}
}

static void *SparkModelResidentdSidebandAddress(
	const SparkModelResidentdBoundary *boundary,
	const SparkPipelineRuntimeRankPlan *rank_plan)
{
	uint64_t primary_capacity;
	if ( boundary == 0 || rank_plan == 0 || boundary->cuda_address == 0 )
		return(0);
	primary_capacity = rank_plan->boundary_bytes_per_sequence * rank_plan->max_input_row_count;
	return((uint8_t *)boundary->cuda_address + primary_capacity);
}

static SparkStatus SparkModelResidentdBindRoute(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route,
	const void *message,
	uint32_t message_bytes,
	uint32_t decision_required)
{
	SparkModelResidentdSlot *slot;
	uint64_t hidden_bytes,input_sideband_bytes,output_sideband_bytes;
	void *input_sideband_address,*output_sideband_address;
	SparkStatus status;
	if ( runtime == 0 || route == 0 || message == 0 || message_bytes > runtime->route_message_capacity || route->slot_index >= runtime->route_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot = &runtime->slots[route->slot_index];
	memcpy(slot->message,message,message_bytes);
	route->message_bytes = message_bytes;
	status = SparkModelResidentIpcDecodeSubmission(slot->message,message_bytes,&route->submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterValidateRuntimeSubmission(runtime->adapter_library.adapter_interface.descriptor,&runtime->runtime_limits,&route->submission);
	hidden_bytes = status == SPARK_STATUS_OK &&
		((runtime->rank_plan.flags &
		 SPARK_PIPELINE_RUNTIME_RANK_FLAG_PARALLEL_FANOUT) == 0u ||
		 (runtime->adapter_library.adapter_interface.descriptor->capability_flags &
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP) != 0u) ?
		(uint64_t)route->submission.row_count *
		runtime->rank_plan.boundary_bytes_per_sequence : 0u;
	input_sideband_bytes = status == SPARK_STATUS_OK && (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u ? (uint64_t)route->submission.row_count * runtime->rank_plan.input_sideband_bytes_per_sequence : 0u;
	output_sideband_bytes = status == SPARK_STATUS_OK && (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u ? (uint64_t)route->submission.row_count * runtime->rank_plan.output_sideband_bytes_per_sequence : 0u;
	if ( status == SPARK_STATUS_OK && (hidden_bytes + input_sideband_bytes > runtime->rank_plan.input_max_packet_bytes || hidden_bytes + output_sideband_bytes > runtime->rank_plan.output_max_packet_bytes) )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&runtime->mutex);
		status = SparkModelResidentdClaimResidentSlotsLocked(runtime,route);
		pthread_mutex_unlock(&runtime->mutex);
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( route->submission.work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE && (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u )
	{
		input_sideband_address = runtime->rank_plan.input_sideband_bytes_per_sequence != 0u ? SparkModelResidentdSidebandAddress(&slot->input,&runtime->rank_plan) : 0;
		route->submission.hidden_input_address = slot->input.cuda_address;
		route->submission.hidden_input_bytes = hidden_bytes;
		route->submission.boundary_sideband_input_address = input_sideband_address;
		route->submission.boundary_sideband_input_bytes = input_sideband_bytes;
		SparkModelResidentdInitializePacket(runtime,&route->input_packet,&route->submission,slot->input.cuda_address,input_sideband_address,runtime->rank_plan.input_sideband_kind,runtime->rank_plan.input_sideband_bytes_per_sequence);
		route->ready_state = SPARK_MODEL_RESIDENTD_ROUTE_READY_INPUT;
	}
	else
		route->ready_state = SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER;
	if ( route->submission.work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE && (runtime->rank_plan.flags & SPARK_PIPELINE_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u )
	{
		output_sideband_address = runtime->rank_plan.output_sideband_bytes_per_sequence != 0u ? SparkModelResidentdSidebandAddress(&slot->output,&runtime->rank_plan) : 0;
		route->submission.hidden_output_address = slot->output.cuda_address;
		route->submission.hidden_output_bytes = hidden_bytes;
		route->submission.boundary_sideband_output_address = output_sideband_address;
		route->submission.boundary_sideband_output_bytes = output_sideband_bytes;
		SparkModelResidentdInitializePacket(runtime,&route->output_packet,&route->submission,slot->output.cuda_address,output_sideband_address,runtime->rank_plan.output_sideband_kind,runtime->rank_plan.output_sideband_bytes_per_sequence);
	}
	route->decision_required = decision_required;
	route->prepared_cache = decision_required;
	route->state = decision_required != 0u ? SPARK_MODEL_RESIDENTD_ROUTE_RESERVED : route->ready_state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdProcessHello(
	SparkModelResidentdRuntime *runtime,
	const void *message,
	uint32_t message_bytes)
{
	const SparkModelResidentIpcHello *hello;
	SparkModelResidentIpcHelloAck ack;
	SparkStatus status,queue_status;
	hello = (const SparkModelResidentIpcHello *)message;
	status = SparkModelResidentIpcValidateHello(hello,message_bytes,runtime->rank_plan.rank_index,runtime->rank_plan.stage_index,runtime->adapter_library.adapter_interface.descriptor);
	queue_status = SparkModelResidentIpcInitializeHelloAck(&ack,
		hello->header.message_id,status,runtime->rank_plan.rank_index,
		runtime->rank_plan.stage_index,runtime->client.generation,
		runtime->adapter_library.adapter_interface.descriptor,
		&runtime->runtime_limits);
	if ( queue_status == SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&runtime->mutex);
		queue_status = SparkModelResidentdQueueRawLocked(runtime,&ack,sizeof(ack));
		pthread_mutex_unlock(&runtime->mutex);
	}
	if ( status == SPARK_STATUS_OK && queue_status == SPARK_STATUS_OK )
		runtime->client.hello_complete = 1u;
	else
		runtime->client.close_after_output = 1u;
	return(queue_status);
}

static SparkStatus SparkModelResidentdQueueSubmitResult(
	SparkModelResidentdRuntime *runtime,
	uint64_t message_id,
	uint64_t submission_id,
	SparkStatus submit_status,
	SparkModelResidentdRoute *route)
{
	SparkModelResidentIpcSubmitResult result;
	SparkStatus status;
	status = SparkModelResidentIpcInitializeSubmitResult(&result,message_id,submission_id,submit_status);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdQueueRawLocked(runtime,&result,sizeof(result));
	if ( route != 0 && submit_status == SPARK_STATUS_OK && status == SPARK_STATUS_OK )
		route->result_queued = 1u;
	return(status);
}

static SparkStatus SparkModelResidentdProcessSubmission(
	SparkModelResidentdRuntime *runtime,
	const void *message,
	uint32_t message_bytes,
	uint32_t decision_required)
{
	const SparkModelResidentIpcSubmit *wire;
	SparkModelServingSubmission submission;
	SparkModelResidentdRoute *route;
	SparkStatus cleanup_status,queue_status,resolution_status,status;
	uint32_t cache_committed,cache_prepared,cache_transactional;
	wire = (const SparkModelResidentIpcSubmit *)message;
	status = SparkModelResidentIpcDecodeSubmission(message,message_bytes,&submission);
	if ( status == SPARK_STATUS_OK && decision_required == 0u )
		status = SparkModelResidentIpcValidateDirectSubmitDescriptor(
			runtime->adapter_library.adapter_interface.descriptor);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterValidateRuntimeSubmission(runtime->adapter_library.adapter_interface.descriptor,&runtime->runtime_limits,&submission);
	if ( status == SPARK_STATUS_OK && submission.submission_id <= runtime->client.last_submission_id )
		status = submission.submission_id == runtime->client.last_submission_id ? SPARK_STATUS_DUPLICATE : SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterPrepareSubmission(&runtime->adapter_library.adapter_interface,runtime->adapter_state,&submission);
	cache_transactional =
		(runtime->adapter_library.adapter_interface.descriptor->capability_flags &
		 (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH)) ==
		 (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH) ? 1u : 0u;
	cache_prepared = status == SPARK_STATUS_OK && cache_transactional != 0u ?
		1u : 0u;
	cache_committed = 0u;
	route = 0;
	if ( status == SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&runtime->mutex);
		route = SparkModelResidentdReserveRoute(runtime,&submission,wire->header.message_id);
		pthread_mutex_unlock(&runtime->mutex);
		if ( route == 0 )
			status = SPARK_STATUS_BUSY;
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkModelResidentdBindRoute(runtime,route,message,message_bytes,
			decision_required);
		if ( status == SPARK_STATUS_OK )
			route->prepared_cache = cache_prepared;
	}
	resolution_status = SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK && cache_prepared != 0u &&
		decision_required == 0u )
	{
		cache_prepared = 0u;
		resolution_status = SparkModelServingAdapterResolvePrefetch(
			&runtime->adapter_library.adapter_interface,runtime->adapter_state,
			&route->submission,
			SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT);
		if ( resolution_status == SPARK_STATUS_OK )
		{
			cache_committed = 1u;
			route->prepared_cache = 0u;
		}
		else
			status = resolution_status;
	}
	if ( status != SPARK_STATUS_OK && cache_prepared != 0u )
	{
		resolution_status = SparkModelServingAdapterResolvePrefetch(
			&runtime->adapter_library.adapter_interface,runtime->adapter_state,
			&submission,SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT);
		if ( resolution_status == SPARK_STATUS_OK )
			cache_prepared = 0u;
		else
			status = resolution_status;
	}
	pthread_mutex_lock(&runtime->mutex);
	if ( route != 0 && status == SPARK_STATUS_OK && cache_committed != 0u &&
		decision_required == 0u )
		status = SparkModelResidentdEnqueueCommittedLocked(runtime,route);
	if ( route != 0 && status != SPARK_STATUS_OK )
	{
		if ( resolution_status == SPARK_STATUS_OK && cache_committed == 0u )
			cleanup_status = SparkModelResidentdDeactivateRouteLocked(runtime,
				route);
		else
		{
			route->state = SPARK_MODEL_RESIDENTD_ROUTE_FENCED;
			cleanup_status = SPARK_STATUS_OK;
		}
		if ( cleanup_status != SPARK_STATUS_OK )
			status = cleanup_status;
	}
	queue_status = SparkModelResidentdQueueSubmitResult(runtime,wire->header.message_id,wire->submission_id,status,route);
	if ( queue_status == SPARK_STATUS_OK && status == SPARK_STATUS_OK )
		runtime->client.last_submission_id = submission.submission_id;
	pthread_mutex_unlock(&runtime->mutex);
	if ( queue_status != SPARK_STATUS_OK && route != 0 && status == SPARK_STATUS_OK )
	{
		if ( cache_prepared != 0u )
			resolution_status = SparkModelServingAdapterResolvePrefetch(
				&runtime->adapter_library.adapter_interface,
				runtime->adapter_state,&route->submission,
				SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT);
		pthread_mutex_lock(&runtime->mutex);
		if ( resolution_status == SPARK_STATUS_OK && cache_committed == 0u )
			cleanup_status = SparkModelResidentdDeactivateRouteLocked(runtime,
				route);
		else
		{
			route->state = SPARK_MODEL_RESIDENTD_ROUTE_FENCED;
			cleanup_status = resolution_status;
		}
		pthread_mutex_unlock(&runtime->mutex);
		if ( cleanup_status != SPARK_STATUS_OK )
			return(cleanup_status);
	}
	return(queue_status);
}

static SparkStatus SparkModelResidentdValidateContinuationLease(
	const SparkModelResidentdRuntime *runtime,
	const SparkModelResidentIpcSubmit *wire,
	const SparkModelServingSubmission *submission)
{
	const SparkModelResidentdSequenceSlot *slot;
	const SparkModelServingLane *lane;
	SparkStatus status;
	uint32_t lane_index;
	if ( (runtime->adapter_library.adapter_interface.descriptor->
		capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( wire->client_generation != runtime->client.generation )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (lane_index=0u; lane_index<submission->active_sequence_count;
		lane_index++)
	{
		lane = &submission->lanes[lane_index];
		slot = &runtime->sequence_slots[lane->resident_sequence_slot];
		if ( SparkModelResidentdSequenceSlotMatches(slot,lane) == 0u )
			return(SPARK_STATUS_SCHEMA_ERROR);
		status = SparkModelContinuationLeaseValidate(&slot->lease,
			wire->client_generation,submission->control_generation,
			lane->sequence_position,lane->step_generation);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdProcessContinuation(
	SparkModelResidentdRuntime *runtime,
	const void *message,
	uint32_t message_bytes)
{
	const SparkModelResidentIpcSubmit *wire;
	SparkModelServingSubmission submission;
	SparkModelResidentdRoute *route;
	SparkStatus cleanup_status,status;
	wire = (const SparkModelResidentIpcSubmit *)message;
	status = SparkModelResidentIpcDecodeSubmission(message,message_bytes,
		&submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterValidateRuntimeSubmission(
			runtime->adapter_library.adapter_interface.descriptor,
			&runtime->runtime_limits,&submission);
	if ( status == SPARK_STATUS_OK && submission.submission_id <=
		runtime->client.last_submission_id )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	pthread_mutex_lock(&runtime->mutex);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdValidateContinuationLease(runtime,wire,
			&submission);
	route = status == SPARK_STATUS_OK ? SparkModelResidentdReserveRoute(runtime,
		&submission,wire->header.message_id) : 0;
	pthread_mutex_unlock(&runtime->mutex);
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&runtime->mutex);
		SparkModelResidentdFailLocked(runtime,status,
			SPARK_MODEL_RESIDENTD_FAILURE_CONTINUE_LEASE,0);
		pthread_mutex_unlock(&runtime->mutex);
		return(status);
	}
	if ( route == 0 )
		return(SPARK_STATUS_BUSY);
	status = SparkModelResidentdBindRoute(runtime,route,message,message_bytes,0u);
	pthread_mutex_lock(&runtime->mutex);
	if ( status == SPARK_STATUS_OK )
	{
		route->state = SPARK_MODEL_RESIDENTD_ROUTE_CONTINUATION_PREPARING;
		runtime->client.last_submission_id = submission.submission_id;
	}
	else
	{
		cleanup_status = SparkModelResidentdDeactivateRouteLocked(runtime,route);
		if ( cleanup_status != SPARK_STATUS_OK )
			status = cleanup_status;
		SparkModelResidentdFailLocked(runtime,status,
			SPARK_MODEL_RESIDENTD_FAILURE_CONTINUE_LEASE,route);
	}
	pthread_mutex_unlock(&runtime->mutex);
	return(status);
}

static SparkStatus SparkModelResidentdProcessDecision(
	SparkModelResidentdRuntime *runtime,
	const SparkModelResidentIpcDecision *decision,
	uint32_t message_bytes)
{
	SparkModelResidentdRoute *route;
	SparkModelResidentIpcDecisionResult result;
	SparkStatus queue_status,status;
	uint32_t resolution,slot_index;
	status = SparkModelResidentIpcValidateDecision(decision,message_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pthread_mutex_lock(&runtime->mutex);
	route = SparkModelResidentdFindRoute(runtime,decision->submission_id);
	if ( route == 0 )
		status = SPARK_STATUS_NOT_FOUND;
	if ( status == SPARK_STATUS_OK && (route->decision_required == 0u || route->state != SPARK_MODEL_RESIDENTD_ROUTE_RESERVED || route->submission.control_generation != decision->control_generation || route->submission.transaction_id != decision->transaction_id || route->submission.dispatch_generation != decision->dispatch_generation) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	slot_index = status == SPARK_STATUS_OK ? route->slot_index : UINT32_MAX;
	if ( status == SPARK_STATUS_OK )
		route->state = SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING;
	pthread_mutex_unlock(&runtime->mutex);
	if ( status == SPARK_STATUS_OK )
	{
		resolution = decision->decision ==
			SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT ?
			SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT :
			SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT;
		status = SparkModelServingAdapterResolvePrefetch(
			&runtime->adapter_library.adapter_interface,runtime->adapter_state,
			&route->submission,resolution);
	}
	pthread_mutex_lock(&runtime->mutex);
	if ( slot_index != UINT32_MAX && (route != &runtime->routes[slot_index] ||
		route->active == 0u ||
		route->state != SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING ||
		route->submission_id != decision->submission_id) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( slot_index != UINT32_MAX && status == SPARK_STATUS_OK )
	{
		route->prepared_cache = 0u;
		if ( decision->decision == SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT )
		{
			status = SparkModelResidentdEnqueueCommittedLocked(runtime,route);
			if ( status == SPARK_STATUS_OK )
				route->state = route->ready_state;
			else
				route->state = SPARK_MODEL_RESIDENTD_ROUTE_FENCED;
		}
		else
			status = SparkModelResidentdDeactivateRouteLocked(runtime,route);
	}
	else if ( slot_index != UINT32_MAX && route->active != 0u &&
		route->state == SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING )
		route->state = SPARK_MODEL_RESIDENTD_ROUTE_FENCED;
	queue_status = SparkModelResidentIpcInitializeDecisionResult(&result,decision,status);
	if ( queue_status == SPARK_STATUS_OK )
		queue_status = SparkModelResidentdQueueRawLocked(runtime,&result,sizeof(result));
	pthread_mutex_unlock(&runtime->mutex);
	return(queue_status);
}

static SparkStatus SparkModelResidentdProcessMessage(
	SparkModelResidentdRuntime *runtime,
	const void *message,
	uint32_t message_bytes)
{
	const SparkModelResidentIpcHeader *header;
	SparkStatus status;
	if ( message == 0 || message_bytes < SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	header = (const SparkModelResidentIpcHeader *)message;
	if ( header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_HELLO )
	{
		status = SparkModelResidentIpcValidateHeader(header,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_HELLO,SPARK_MODEL_RESIDENT_IPC_HELLO_BYTES);
		if ( status != SPARK_STATUS_OK )
			return(status);
		status = runtime->client.hello_complete == 0u ? SparkModelResidentdProcessHello(runtime,message,message_bytes) : SPARK_STATUS_DUPLICATE;
		if ( status == SPARK_STATUS_OK )
			runtime->client.last_message_id = header->message_id;
		return(status);
	}
	if ( runtime->client.hello_complete == 0u || header->message_id <= runtime->client.last_message_id )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT ||
		header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE ||
		header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE )
	{
		status = SparkModelResidentIpcValidateHeader(header,message_bytes,header->kind,SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE )
			status = SparkModelResidentdProcessContinuation(runtime,message,
				message_bytes);
		else
			status = SparkModelResidentdProcessSubmission(runtime,message,
				message_bytes,header->kind ==
				SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE ? 1u : 0u);
	}
	else if ( header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_DECISION )
		status = SparkModelResidentdProcessDecision(runtime,(const SparkModelResidentIpcDecision *)message,message_bytes);
	else
		status = SPARK_STATUS_UNSUPPORTED;
	if ( status == SPARK_STATUS_OK )
		runtime->client.last_message_id = header->message_id;
	return(status);
}

static SparkStatus SparkModelResidentdReadClient(
	SparkModelResidentdRuntime *runtime,
	uint32_t *submission_processed_out)
{
	SparkModelResidentIpcHeader *header;
	ssize_t bytes_read;
	SparkStatus status;
	if ( submission_processed_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*submission_processed_out = 0u;
	while ( runtime->client.fd >= 0 )
	{
		bytes_read = read(runtime->client.fd,runtime->client.input + runtime->client.input_bytes,runtime->client.target_bytes - runtime->client.input_bytes);
		if ( bytes_read == 0 )
		{
			SparkModelResidentdCloseClient(runtime);
			return(SPARK_STATUS_OK);
		}
		if ( bytes_read < 0 )
			return(errno == EAGAIN || errno == EWOULDBLOCK ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
		runtime->client.input_bytes += (uint32_t)bytes_read;
		if ( runtime->client.input_bytes == SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES && runtime->client.target_bytes == SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES )
		{
			header = (SparkModelResidentIpcHeader *)runtime->client.input;
			if ( header->message_bytes < SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES || header->message_bytes > runtime->client.input_capacity )
				return(SPARK_STATUS_SCHEMA_ERROR);
			runtime->client.target_bytes = header->message_bytes;
		}
		if ( runtime->client.input_bytes == runtime->client.target_bytes )
		{
			header = (SparkModelResidentIpcHeader *)runtime->client.input;
			status = SparkModelResidentdProcessMessage(runtime,runtime->client.input,runtime->client.target_bytes);
			if ( status == SPARK_STATUS_OK &&
				(header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT ||
				 header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE ||
				 header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE) )
				*submission_processed_out = 1u;
			runtime->client.input_bytes = 0u;
			runtime->client.target_bytes = SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES;
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdWriteClient(SparkModelResidentdRuntime *runtime)
{
	SparkModelResidentdOutput *output;
	ssize_t bytes_written;
	uint32_t close_client;
	close_client = 0u;
	pthread_mutex_lock(&runtime->mutex);
	while ( runtime->client.fd >= 0 && runtime->client.output_count != 0u )
	{
		output = &runtime->client.output[runtime->client.output_head];
		bytes_written = write(runtime->client.fd,output->message + output->sent_bytes,output->message_bytes - output->sent_bytes);
		if ( bytes_written <= 0 )
		{
			pthread_mutex_unlock(&runtime->mutex);
			return(bytes_written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
		}
		output->sent_bytes += (uint32_t)bytes_written;
		if ( output->sent_bytes != output->message_bytes )
			break;
		runtime->client.output_head = (runtime->client.output_head + 1u) % runtime->client.output_capacity;
		runtime->client.output_count--;
	}
	if ( runtime->client.close_after_output != 0u && runtime->client.output_count == 0u )
		close_client = 1u;
	pthread_mutex_unlock(&runtime->mutex);
	if ( close_client != 0u )
		SparkModelResidentdCloseClient(runtime);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelResidentdTransportCompletionMatches(
	const SparkHiddenTransportCompletion *completion,
	const SparkHiddenTransportPacket *packet)
{
	uint64_t transfer_bytes;
	if ( completion == 0 || packet == 0 )
		return(0u);
	transfer_bytes = (uint64_t)packet->active_sequence_count * ((uint64_t)packet->bytes_per_sequence + packet->sideband_bytes_per_sequence);
	return(completion->sequence_id == packet->sequence_id && completion->token_index == packet->token_index && completion->active_sequence_count == packet->active_sequence_count && completion->transfer_bytes == transfer_bytes ? 1u : 0u);
}

static SparkStatus SparkModelResidentdFinishRoute(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkStatus status;
	pthread_mutex_lock(&runtime->mutex);
	if ( route->active == 0u || route->state != SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION )
		status = SPARK_STATUS_OK;
	else if ( route->deadline_expired != 0u )
	{
		status = SparkModelResidentdQueueDeadlineCompletionLocked(runtime,route);
		if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
			status = SPARK_STATUS_OK;
		else if ( status == SPARK_STATUS_OK &&
			route->deadline_wait_state ==
			SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT )
			status = SparkModelResidentdCompleteResidentSlotsLocked(runtime,route);
		else if ( status == SPARK_STATUS_OK )
			status = SparkModelResidentdReleaseResidentSlotsLocked(runtime,route);
		if ( status == SPARK_STATUS_OK &&
			(route->deadline_completion_queued != 0u ||
			 route->abandoned != 0u || runtime->client.fd < 0 ||
			 route->client_generation != runtime->client.generation) )
			status = SparkModelResidentdDeactivateRouteLocked(runtime,route);
	}
	else if ( route->abandoned != 0u || runtime->client.fd < 0 || route->client_generation != runtime->client.generation )
	{
		status = SparkModelResidentdCompleteResidentSlotsLocked(runtime,route);
		if ( status == SPARK_STATUS_OK )
			status = SparkModelResidentdDeactivateRouteLocked(runtime,route);
	}
	else if ( route->result_queued == 0u )
		status = SPARK_STATUS_INTERNAL_ERROR;
	else
	{
		status = SparkModelResidentdQueueCompletionLocked(runtime,route);
		if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
			status = SPARK_STATUS_OK;
	}
	pthread_mutex_unlock(&runtime->mutex);
	return(status);
}

static SparkStatus SparkModelResidentdApplyTransportCompletionLocked(
	SparkModelResidentdRuntime *runtime,
	const SparkHiddenTransportCompletion *completion,
	uint32_t input)
{
	SparkModelResidentdRoute *route,*matched;
	const SparkHiddenTransportPacket *packet;
	uint32_t index,state;
	matched = 0;
	state = input != 0u ? SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT : SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT;
	for (index=0u; index<runtime->route_capacity; index++)
	{
		route = &runtime->routes[index];
		packet = input != 0u ? &route->input_packet : &route->output_packet;
		if ( route->active != 0u && route->state == state && SparkModelResidentdTransportCompletionMatches(completion,packet) != 0u )
		{
			if ( matched != 0 )
				return(SPARK_STATUS_DUPLICATE);
			matched = route;
		}
	}
	if ( matched == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( matched->deadline_expired == 0u )
	{
		SparkStatus deadline_status;
		deadline_status = SparkModelResidentdExpireTransportRouteLocked(runtime,
			matched,state);
		if ( deadline_status != SPARK_STATUS_OK &&
			deadline_status != SPARK_STATUS_CAPACITY_EXCEEDED )
			return(deadline_status);
	}
	if ( matched->deadline_expired != 0u )
	{
		matched->state = SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION;
		return(SPARK_STATUS_OK);
	}
	if ( completion->status != SPARK_STATUS_OK )
		return(completion->status);
	matched->state = input != 0u ? SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER : SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdProgressTransport(
	SparkModelResidentdRuntime *runtime,
	SparkHiddenTransportSession *session,
	uint32_t input)
{
	SparkHiddenTransportCompletion completion;
	SparkStatus status;
	uint32_t step;
	if ( session == 0 )
		return(SPARK_STATUS_OK);
	for (step=0u; step<runtime->route_capacity; step++)
	{
		memset(&completion,0,sizeof(completion));
		status = SparkHiddenTransportPoll(session,&completion);
		if ( status != SPARK_STATUS_OK || completion.status == SPARK_STATUS_BUSY )
			return(status);
		pthread_mutex_lock(&runtime->mutex);
		status = SparkModelResidentdApplyTransportCompletionLocked(runtime,&completion,input);
		pthread_mutex_unlock(&runtime->mutex);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdSubmitAdapter(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkStatus status,result;
	uint32_t state;
	pthread_mutex_lock(&runtime->mutex);
	if ( route->active == 0u || route->state != SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER )
	{
		pthread_mutex_unlock(&runtime->mutex);
		return(SPARK_STATUS_OK);
	}
	if ( runtime->committed_fifo_head != 0u &&
		runtime->committed_fifo_head != route->slot_index + 1u )
	{
		pthread_mutex_unlock(&runtime->mutex);
		return(SPARK_STATUS_OK);
	}
	if ( runtime->committed_fifo_head == 0u &&
		route->committed_fifo_queued != 0u )
	{
		pthread_mutex_unlock(&runtime->mutex);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	route->state = SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER;
	route->adapter_submit_time_ns = SparkModelResidentdMonotonicTimeNs();
	pthread_mutex_unlock(&runtime->mutex);
	status = runtime->adapter_library.adapter_interface.submit(runtime->adapter_state,&route->submission);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY )
		fprintf(stderr,"model_residentd adapter_submit status=%s rank=%u stage=%u submission=%llu kind=%u rows=%u lanes=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index,runtime->rank_plan.stage_index,(unsigned long long)route->submission.submission_id,route->submission.work_kind,route->submission.row_count,route->submission.active_sequence_count);
	pthread_mutex_lock(&runtime->mutex);
	state = route->state;
	result = SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_BUSY && state == SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER )
	{
		route->state = SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER;
		route->adapter_submit_time_ns = 0u;
		result = SPARK_STATUS_BUSY;
	}
	else if ( status == SPARK_STATUS_BUSY )
		result = SPARK_STATUS_SCHEMA_ERROR;
	else if ( status != SPARK_STATUS_OK )
		result = status;
	else
	{
		result = SparkModelResidentdRemoveCommittedLocked(runtime,route);
		if ( result == SPARK_STATUS_OK &&
			state != SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER &&
			state != SPARK_MODEL_RESIDENTD_ROUTE_READY_OUTPUT &&
			state != SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION )
			result = SPARK_STATUS_SCHEMA_ERROR;
	}
	pthread_mutex_unlock(&runtime->mutex);
	return(result);
}

static SparkStatus SparkModelResidentdFailContinuationLocked(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route,
	SparkStatus status)
{
	route->state = SPARK_MODEL_RESIDENTD_ROUTE_FENCED;
	SparkModelResidentdFailLocked(runtime,status,
		SPARK_MODEL_RESIDENTD_FAILURE_CONTINUE_LEASE,route);
	return(status);
}

static SparkStatus SparkModelResidentdCommitContinuation(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkStatus status;
	status = SparkModelServingAdapterResolvePrefetch(
		&runtime->adapter_library.adapter_interface,runtime->adapter_state,
		&route->submission,SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT);
	pthread_mutex_lock(&runtime->mutex);
	if ( route->active == 0u || route->state !=
		SPARK_MODEL_RESIDENTD_ROUTE_CONTINUATION_PREPARING ||
		route->client_generation != runtime->client.generation )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		route->prepared_cache = 0u;
		status = SparkModelResidentdEnqueueCommittedLocked(runtime,route);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdQueueSubmitResult(runtime,route->message_id,
			route->submission_id,SPARK_STATUS_OK,route);
	if ( status == SPARK_STATUS_OK )
		route->state = route->ready_state;
	else
		status = SparkModelResidentdFailContinuationLocked(runtime,route,status);
	pthread_mutex_unlock(&runtime->mutex);
	return(status);
}

static SparkStatus SparkModelResidentdPrepareContinuation(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route)
{
	SparkStatus status;
	uint32_t transactional;
	pthread_mutex_lock(&runtime->mutex);
	if ( route->active == 0u || route->state !=
		SPARK_MODEL_RESIDENTD_ROUTE_CONTINUATION_PREPARING ||
		route->client_generation != runtime->client.generation )
	{
		pthread_mutex_unlock(&runtime->mutex);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	pthread_mutex_unlock(&runtime->mutex);
	status = SparkModelServingAdapterPrepareSubmission(
		&runtime->adapter_library.adapter_interface,runtime->adapter_state,
		&route->submission);
	if ( status == SPARK_STATUS_BUSY )
		return(status);
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&runtime->mutex);
		status = SparkModelResidentdFailContinuationLocked(runtime,route,status);
		pthread_mutex_unlock(&runtime->mutex);
		return(status);
	}
	transactional = (runtime->adapter_library.adapter_interface.descriptor->
		capability_flags & (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH)) ==
		(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH) ? 1u : 0u;
	if ( transactional != 0u )
		route->prepared_cache = 1u;
	return(SparkModelResidentdCommitContinuation(runtime,route));
}

static SparkStatus SparkModelResidentdPostTransport(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route,
	uint32_t input)
{
	SparkHiddenTransportSession *session;
	SparkHiddenTransportPacket *packet;
	SparkStatus status;
	uint32_t ready_state,wait_state;
	ready_state = input != 0u ? SPARK_MODEL_RESIDENTD_ROUTE_READY_INPUT : SPARK_MODEL_RESIDENTD_ROUTE_READY_OUTPUT;
	wait_state = input != 0u ? SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT : SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT;
	session = input != 0u ? runtime->input_transport : runtime->output_transport;
	packet = input != 0u ? &route->input_packet : &route->output_packet;
	pthread_mutex_lock(&runtime->mutex);
	if ( route->active == 0u || route->state != ready_state )
	{
		pthread_mutex_unlock(&runtime->mutex);
		return(SPARK_STATUS_OK);
	}
	route->state = wait_state;
	pthread_mutex_unlock(&runtime->mutex);
	status = input != 0u ? SparkHiddenTransportPostReceive(session,packet) : SparkHiddenTransportSend(session,packet);
	if ( status == SPARK_STATUS_BUSY )
	{
		pthread_mutex_lock(&runtime->mutex);
		if ( route->active != 0u && route->state == wait_state )
			route->state = ready_state;
		pthread_mutex_unlock(&runtime->mutex);
	}
	return(status);
}

static SparkStatus SparkModelResidentdProgressRoute(
	SparkModelResidentdRuntime *runtime,
	SparkModelResidentdRoute *route,
	uint32_t *adapter_submitted)
{
	SparkStatus status;
	uint32_t state,step;
	for (step=0u; step<4u; step++)
	{
		pthread_mutex_lock(&runtime->mutex);
		state = route->active != 0u ? route->state : SPARK_MODEL_RESIDENTD_ROUTE_IDLE;
		pthread_mutex_unlock(&runtime->mutex);
		if ( state == SPARK_MODEL_RESIDENTD_ROUTE_IDLE ||
			state == SPARK_MODEL_RESIDENTD_ROUTE_RESERVED ||
			state == SPARK_MODEL_RESIDENTD_ROUTE_RESOLVING ||
			state == SPARK_MODEL_RESIDENTD_ROUTE_FENCED ||
			state == SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER )
			return(SPARK_STATUS_OK);
		if ( state == SPARK_MODEL_RESIDENTD_ROUTE_CONTINUATION_PREPARING )
		{
			if ( adapter_submitted == 0 || *adapter_submitted != 0u )
				return(SPARK_STATUS_OK);
			status = SparkModelResidentdPrepareContinuation(runtime,route);
			if ( status == SPARK_STATUS_BUSY )
				return(SPARK_STATUS_OK);
			if ( status != SPARK_STATUS_OK )
				return(status);
			*adapter_submitted = 1u;
			SparkModelResidentdWake(runtime);
			continue;
		}
		if ( state == SPARK_MODEL_RESIDENTD_ROUTE_WAIT_INPUT ||
			state == SPARK_MODEL_RESIDENTD_ROUTE_WAIT_OUTPUT )
		{
			pthread_mutex_lock(&runtime->mutex);
			if ( route->active != 0u && route->state == state )
				status = SparkModelResidentdExpireTransportRouteLocked(runtime,
					route,state);
			else
				status = SPARK_STATUS_OK;
			pthread_mutex_unlock(&runtime->mutex);
			return(status == SPARK_STATUS_CAPACITY_EXCEEDED ? SPARK_STATUS_OK :
				status);
		}
		if ( state == SPARK_MODEL_RESIDENTD_ROUTE_READY_INPUT )
		{
			status = SparkModelResidentdPostTransport(runtime,route,1u);
			return(status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY ? SPARK_STATUS_OK : status);
		}
		if ( state == SPARK_MODEL_RESIDENTD_ROUTE_READY_ADAPTER )
		{
			if ( adapter_submitted == 0 || *adapter_submitted != 0u )
				return(SPARK_STATUS_OK);
			status = SparkModelResidentdSubmitAdapter(runtime,route);
			if ( status == SPARK_STATUS_BUSY )
				return(SPARK_STATUS_OK);
			if ( status != SPARK_STATUS_OK )
			{
				/* A refused route is a per-route outcome, not a daemon
				 * failure: queue a failed completion for the client and let
				 * the route drain through the normal completion path. */
				pthread_mutex_lock(&runtime->mutex);
				if ( route->active != 0u && route->state ==
					SPARK_MODEL_RESIDENTD_ROUTE_WAIT_ADAPTER )
				{
					(void)SparkModelResidentdRemoveCommittedLocked(runtime,
						route);
					memset(&route->completion,0,sizeof(route->completion));
					route->completion.abi_version =
						SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
					route->completion.descriptor_bytes =
						SPARK_MODEL_SERVING_COMPLETION_BYTES;
					route->completion.status = (uint32_t)status;
					route->completion.submission_id =
						route->submission.submission_id;
					route->completion.request_id =
						route->submission.request_id;
					route->completion.sequence_id =
						route->submission.sequence_id;
					route->completion.sequence_position =
						route->submission.sequence_position;
					route->completion.control_generation =
						route->submission.control_generation;
					route->completion.transaction_id =
						route->submission.transaction_id;
					route->completion.dispatch_generation =
						route->submission.dispatch_generation;
					route->completion.request_generation =
						route->submission.request_generation;
					route->completion.step_generation =
						route->submission.step_generation;
					route->completion.residency =
						route->submission.residency;
					route->state =
						SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION;
				}
				pthread_mutex_unlock(&runtime->mutex);
				return(SPARK_STATUS_OK);
			}
			*adapter_submitted = 1u;
			SparkModelResidentdWake(runtime);
			continue;
		}
		if ( state == SPARK_MODEL_RESIDENTD_ROUTE_READY_OUTPUT )
		{
			status = SparkModelResidentdPostTransport(runtime,route,0u);
			return(status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY ? SPARK_STATUS_OK : status);
		}
		if ( state == SPARK_MODEL_RESIDENTD_ROUTE_READY_COMPLETION )
			return(SparkModelResidentdFinishRoute(runtime,route));
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentdProgressRoutes(
	SparkModelResidentdRuntime *runtime,
	uint32_t allow_adapter)
{
	SparkStatus status;
	uint32_t adapter_submitted,index,offset,start;
	status = SPARK_STATUS_OK;
	adapter_submitted = 0u;
	start = allow_adapter != 0u ? runtime->next_adapter_route : 0u;
	for (offset=0u; status == SPARK_STATUS_OK && offset<runtime->route_capacity && adapter_submitted == 0u; offset++)
	{
		index = (start + offset) % runtime->route_capacity;
		status = SparkModelResidentdProgressRoute(runtime,&runtime->routes[index],allow_adapter != 0u ? &adapter_submitted : 0);
		if ( adapter_submitted != 0u )
			runtime->next_adapter_route = (index + 1u) % runtime->route_capacity;
	}
	return(status);
}

static SparkStatus SparkModelResidentdProgress(SparkModelResidentdRuntime *runtime)
{
	SparkStatus status;
	/* ADMIT FIRST. ProgressRoutes(allow_adapter=1) submits the next
	 * decode-draft (adapter.submit = spec phase one). The adapter's progress
	 * scan (spec phase two of the prior submission) runs AFTER admission so
	 * the driver can run verify(N) + decode-draft(N+1) concurrently under
	 * max_inflight_submission_count, instead of serializing phase two ahead
	 * of the next admission. The committed-fifo head ordering and the
	 * continuation-lease chain are untouched: they live in the submit and
	 * completion paths, not in this scan order. */
	status = SparkModelResidentdProgressRoutes(runtime,0u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=routes-pre status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdProgressTransport(runtime,runtime->input_transport,1u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=input-transport-pre status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdProgressTransport(runtime,runtime->output_transport,0u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=output-transport-pre status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdProgressRoutes(runtime,0u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=routes-mid status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdProgressRoutes(runtime,1u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=routes-adapter status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	if ( status == SPARK_STATUS_OK )
	{
		status = runtime->adapter_library.adapter_interface.progress(runtime->adapter_state,SPARK_MODEL_RESIDENTD_PROGRESS_STEPS);
		if ( status == SPARK_STATUS_BUSY )
			status = SPARK_STATUS_OK;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdProgressTransport(runtime,runtime->input_transport,1u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=input-transport-post status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdProgressTransport(runtime,runtime->output_transport,0u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=output-transport-post status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdProgressRoutes(runtime,0u);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"model_residentd progress stage=routes-final status=%s rank=%u\n",SparkStatusToString(status),runtime->rank_plan.rank_index);
	return(status);
}

static SparkStatus SparkModelResidentdAppendTransportFds(
	SparkHiddenTransportSession *session,
	struct pollfd *fds,
	uint32_t capacity,
	uint32_t *count)
{
	SparkHiddenTransportPollDescriptor descriptors[SPARK_MODEL_RESIDENTD_TRANSPORT_POLL_CAPACITY];
	SparkStatus status;
	uint32_t descriptor_count,index;
	if ( session == 0 )
		return(SPARK_STATUS_OK);
	descriptor_count = 0u;
	status = SparkHiddenTransportGetPollDescriptors(session,descriptors,SPARK_MODEL_RESIDENTD_TRANSPORT_POLL_CAPACITY,&descriptor_count);
	for (index=0u; status == SPARK_STATUS_OK && index<descriptor_count; index++)
	{
		if ( *count >= capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		fds[*count].fd = descriptors[index].fd;
		fds[*count].events = 0;
		if ( (descriptors[index].events & SPARK_HIDDEN_TRANSPORT_POLL_READ) != 0u )
			fds[*count].events |= POLLIN;
		if ( (descriptors[index].events & SPARK_HIDDEN_TRANSPORT_POLL_WRITE) != 0u )
			fds[*count].events |= POLLOUT;
		(*count)++;
	}
	return(status);
}

static SparkStatus SparkModelResidentdBuildPollFds(
	SparkModelResidentdRuntime *runtime,
	struct pollfd *fds,
	uint32_t capacity,
	uint32_t *count_out)
{
	SparkStatus status;
	uint32_t count;
	if ( capacity < 3u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(fds,0,capacity * sizeof(fds[0]));
	fds[0].fd = runtime->listen_fd;
	pthread_mutex_lock(&runtime->mutex);
	fds[0].events = runtime->client.fd < 0 ? POLLIN : 0;
	fds[1].fd = runtime->client.fd;
	fds[1].events = runtime->client.fd >= 0 && runtime->client.close_after_output == 0u ? POLLIN : 0;
	/* Request POLLOUT only while output is genuinely pending. Requesting it
	 * unconditionally makes poll() return immediately on a writable socket,
	 * turning this loop into a 100%-CPU busy-spin for the whole GPU decode and
	 * starving the driver's completion callback (both IPC round-trips). The
	 * queued ACK is still flushed here: ReadClient queues it and the run loop
	 * calls WriteClient in the same iteration (submission_processed path). */
	if ( runtime->client.fd >= 0 && runtime->client.close_after_output == 0u && runtime->client.output_count != 0u )
		fds[1].events |= POLLOUT;
	pthread_mutex_unlock(&runtime->mutex);
	fds[2].fd = runtime->wake_read_fd;
	fds[2].events = POLLIN;
	count = 3u;
	status = SparkModelResidentdAppendTransportFds(runtime->input_transport,fds,capacity,&count);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdAppendTransportFds(runtime->output_transport,fds,capacity,&count);
	*count_out = count;
	return(status);
}

static void SparkModelResidentdDrainWake(SparkModelResidentdRuntime *runtime)
{
	uint8_t values[64];
	while ( read(runtime->wake_read_fd,values,sizeof(values)) > 0 )
		;
}

static int32_t SparkModelResidentdPollTimeoutMs(
	SparkModelResidentdRuntime *runtime)
{
	uint32_t active,index;
	active = 0u;
	pthread_mutex_lock(&runtime->mutex);
	for (index=0u; active == 0u && index<runtime->route_capacity; index++)
		active = runtime->routes[index].active;
	pthread_mutex_unlock(&runtime->mutex);
	return(active != 0u ? 10 : 1000);
}

static SparkStatus SparkModelResidentdRun(SparkModelResidentdRuntime *runtime)
{
	struct pollfd fds[3u + (2u * SPARK_MODEL_RESIDENTD_TRANSPORT_POLL_CAPACITY)];
	uint32_t count;
	SparkStatus status;
	SparkStatus client_status;
	uint32_t failed_status,submission_processed;
	int32_t poll_status;
	status = SPARK_STATUS_OK;
	while ( SparkModelResidentdStop == 0 && atomic_load(&runtime->failed_status) == SPARK_STATUS_OK && status == SPARK_STATUS_OK )
	{
		status = SparkModelResidentdBuildPollFds(runtime,fds,sizeof(fds) / sizeof(fds[0]),&count);
		if ( status != SPARK_STATUS_OK )
			break;
		poll_status = poll(fds,count,SparkModelResidentdPollTimeoutMs(runtime));
		if ( poll_status < 0 && errno != EINTR )
			status = SPARK_STATUS_IO_ERROR;
		if ( poll_status > 0 && (fds[0].revents & POLLIN) != 0 )
			SparkModelResidentdAcceptClient(runtime);
		if ( poll_status > 0 && fds[1].fd >= 0 && (fds[1].revents & POLLIN) != 0 )
		{
			client_status = SparkModelResidentdReadClient(runtime,
				&submission_processed);
			if ( client_status != SPARK_STATUS_OK )
				SparkModelResidentdCloseClient(runtime);
			else if ( submission_processed != 0u )
			{
				client_status = SparkModelResidentdWriteClient(runtime);
				if ( client_status != SPARK_STATUS_OK )
					SparkModelResidentdCloseClient(runtime);
			}
		}
		if ( status == SPARK_STATUS_OK && poll_status > 0 && fds[1].fd >= 0 && (fds[1].revents & POLLOUT) != 0 )
		{
			client_status = SparkModelResidentdWriteClient(runtime);
			if ( client_status != SPARK_STATUS_OK )
				SparkModelResidentdCloseClient(runtime);
		}
		if ( poll_status > 0 && fds[1].fd >= 0 && (fds[1].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0 )
			SparkModelResidentdCloseClient(runtime);
		if ( poll_status > 0 && (fds[2].revents & POLLIN) != 0 )
			SparkModelResidentdDrainWake(runtime);
		if ( status == SPARK_STATUS_OK )
			status = SparkModelResidentdProgress(runtime);
	}
	failed_status = atomic_load(&runtime->failed_status);
	return(failed_status != SPARK_STATUS_OK && status == SPARK_STATUS_OK ? (SparkStatus)failed_status : status);
}

int main(int argument_count,char **arguments)
{
	SparkModelResidentDeployment deployment;
	SparkModelResidentdConfiguration configuration;
	SparkModelResidentdLaunch launch;
	SparkModelResidentdRuntime runtime;
	SparkStatus status;
	SparkModelResidentDeploymentReset(&deployment);
	status = SparkModelResidentdParseLaunch(argument_count,arguments,&launch);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentLoad(launch.deployment_path,&deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdBuildConfiguration(&deployment,launch.rank_index,&configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentdValidateDirectories(&configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelResidentdUsage(arguments[0]);
		fprintf(stderr,"model_residentd deployment=%s\n",SparkStatusToString(status));
		SparkModelResidentDeploymentDestroy(&deployment);
		return(2);
	}
	signal(SIGINT,SparkModelResidentdSignal);
	signal(SIGTERM,SparkModelResidentdSignal);
	status = SparkModelResidentdInitialize(&runtime,&configuration);
	if ( status == SPARK_STATUS_OK )
	{
		if ( configuration.socket_path != 0 )
			printf("model_residentd ready rank=%u stage=%u inflight=%u active=%u rows=%u resident=%u adapter=%s model=%s revision=%s unix=%s\n",runtime.rank_plan.rank_index,runtime.rank_plan.stage_index,runtime.runtime_limits.max_inflight_submission_count,runtime.runtime_limits.max_active_sequence_count,runtime.runtime_limits.max_input_row_count,runtime.runtime_limits.resident_sequence_capacity,runtime.adapter_library.adapter_interface.descriptor->adapter_id,runtime.adapter_library.adapter_interface.descriptor->model_id,runtime.adapter_library.adapter_interface.descriptor->model_revision,configuration.socket_path);
		else
			printf("model_residentd ready rank=%u stage=%u inflight=%u active=%u rows=%u resident=%u adapter=%s model=%s revision=%s tcp=%s:%u\n",runtime.rank_plan.rank_index,runtime.rank_plan.stage_index,runtime.runtime_limits.max_inflight_submission_count,runtime.runtime_limits.max_active_sequence_count,runtime.runtime_limits.max_input_row_count,runtime.runtime_limits.resident_sequence_capacity,runtime.adapter_library.adapter_interface.descriptor->adapter_id,runtime.adapter_library.adapter_interface.descriptor->model_id,runtime.adapter_library.adapter_interface.descriptor->model_revision,configuration.listen_address,configuration.listen_port);
		fflush(stdout);
		status = SparkModelResidentdRun(&runtime);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"model_residentd run=%s status=%u rank=%u stage=%u reason=%u submission=%llu kind=%u route_state=%u\n",SparkStatusToString(status),(uint32_t)status,runtime.rank_plan.rank_index,runtime.rank_plan.stage_index,runtime.failed_reason,(unsigned long long)runtime.failed_submission_id,runtime.failed_work_kind,runtime.failed_route_state);
	}
	else
		fprintf(stderr,"model_residentd initialize=%s status=%u phase=%s rank=%u stage=%u\n",SparkStatusToString(status),(uint32_t)status,runtime.initialize_phase != 0 ? runtime.initialize_phase : "reset",configuration.rank_index,configuration.stage_index);
	SparkModelResidentdDestroy(&runtime,&configuration);
	SparkModelResidentDeploymentDestroy(&deployment);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}
