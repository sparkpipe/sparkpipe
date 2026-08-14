#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_model_resident_client.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

typedef struct SparkModelResidentClientOutput
{
	uint32_t message_bytes;
	uint32_t sent_bytes;
} SparkModelResidentClientOutput;

typedef struct SparkModelResidentClientPending
{
	uint32_t active;
	uint32_t result_received;
	uint32_t requires_decision;
	uint32_t prepared;
	uint32_t committed;
	uint32_t continued;
	uint32_t decision_kind;
	uint32_t decision_result_received;
	uint32_t work_kind;
	uint32_t active_sequence_count;
	uint32_t tokens_per_sequence;
	uint64_t message_id;
	uint64_t decision_message_id;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	SparkModelDriverResidencyToken residency;
} SparkModelResidentClientPending;

struct SparkModelResidentClient
{
	int32_t fd;
	uint32_t connected;
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t queue_capacity;
	uint32_t output_head;
	uint32_t output_count;
	uint32_t output_message_capacity;
	uint32_t input_capacity;
	uint32_t input_bytes;
	uint32_t input_target_bytes;
	uint32_t pending_count;
	uint32_t prepared_count;
	uint64_t client_generation;
	SparkModelServingRuntimeLimits runtime_limits;
	uint64_t next_message_id;
	uint64_t last_submission_id;
	uint64_t submitted_count;
	uint64_t prepared_total;
	uint64_t continued_total;
	uint64_t admitted_count;
	uint64_t rejected_count;
	uint64_t aborted_count;
	uint64_t completed_count;
	const SparkModelServingAdapterDescriptor *adapter_descriptor;
	SparkModelResidentSubmitResultFunction submit_result_function;
	void *submit_result_context;
	SparkModelResidentDecisionResultFunction decision_result_function;
	void *decision_result_context;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkModelResidentClientOutput *outputs;
	SparkModelResidentClientPending *pending;
	uint8_t *output_storage;
	uint8_t *input;
};

static SparkStatus SparkModelResidentClientSetNonblocking(int32_t fd)
{
	int32_t flags;
	flags = fcntl(fd,F_GETFL,0);
	return(flags < 0 || fcntl(fd,F_SETFL,flags | O_NONBLOCK) != 0 ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

static ssize_t SparkModelResidentClientSend(
	int32_t fd,
	const void *bytes,
	uint32_t byte_count)
{
	return(send(fd,bytes,byte_count,MSG_NOSIGNAL));
}

static SparkStatus SparkModelResidentClientWait(
	int32_t fd,
	int16_t events,
	uint32_t timeout_ms)
{
	struct pollfd descriptor;
	int32_t status;
	memset(&descriptor,0,sizeof(descriptor));
	descriptor.fd = fd;
	descriptor.events = events;
	status = poll(&descriptor,1,(int32_t)timeout_ms);
	if ( status <= 0 )
		return(SPARK_STATUS_IO_ERROR);
	return((descriptor.revents & events) != 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkModelResidentClientWriteFull(
	int32_t fd,
	const void *message,
	uint32_t message_bytes,
	uint32_t timeout_ms)
{
	const uint8_t *bytes;
	uint32_t offset;
	ssize_t written;
	SparkStatus status;
	bytes = (const uint8_t *)message;
	offset = 0u;
	while ( offset < message_bytes )
	{
		status = SparkModelResidentClientWait(fd,POLLOUT,timeout_ms);
		if ( status != SPARK_STATUS_OK )
			return(status);
		written = SparkModelResidentClientSend(fd,bytes + offset,message_bytes - offset);
		if ( written <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		offset += (uint32_t)written;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientReadFull(
	int32_t fd,
	void *message,
	uint32_t message_bytes,
	uint32_t timeout_ms)
{
	uint8_t *bytes;
	uint32_t offset;
	ssize_t received;
	SparkStatus status;
	bytes = (uint8_t *)message;
	offset = 0u;
	while ( offset < message_bytes )
	{
		status = SparkModelResidentClientWait(fd,POLLIN,timeout_ms);
		if ( status != SPARK_STATUS_OK )
			return(status);
		received = recv(fd,bytes + offset,message_bytes - offset,0);
		if ( received <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		offset += (uint32_t)received;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientValidateConfiguration(
	const SparkModelResidentClientConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_RESIDENT_CLIENT_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(configuration->adapter_descriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkModelResidentEndpointValidate(&configuration->endpoint);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->rank_index >= configuration->adapter_descriptor->stage_count || configuration->stage_index >= configuration->adapter_descriptor->stage_count || configuration->runtime_limits.max_inflight_submission_count > SPARK_MODEL_RESIDENT_CLIENT_MAX_QUEUE_CAPACITY || configuration->connect_timeout_ms == 0u || configuration->reserved0 != 0u || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientAllocate(
	const SparkModelResidentClientConfiguration *configuration,
	SparkModelResidentClient **client_out)
{
	SparkModelResidentClient *client;
	uint32_t input_bytes,output_bytes;
	SparkStatus status;
	status = SparkModelResidentIpcCalculateSubmitBytes(configuration->runtime_limits.max_active_sequence_count,configuration->runtime_limits.max_input_row_count,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,&output_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcCalculateCompletionBytes(configuration->adapter_descriptor->max_output_token_count,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,&input_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	client = (SparkModelResidentClient *)calloc(1u,sizeof(*client));
	if ( client == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	client->fd = -1;
	client->queue_capacity = configuration->runtime_limits.max_inflight_submission_count;
	client->output_message_capacity = output_bytes;
	client->input_capacity = input_bytes > SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES ? input_bytes : SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
	client->outputs = (SparkModelResidentClientOutput *)calloc(client->queue_capacity,sizeof(client->outputs[0]));
	client->pending = (SparkModelResidentClientPending *)calloc(client->queue_capacity,sizeof(client->pending[0]));
	client->output_storage = (uint8_t *)calloc(client->queue_capacity,output_bytes);
	client->input = (uint8_t *)malloc(client->input_capacity);
	if ( client->outputs == 0 || client->pending == 0 || client->output_storage == 0 || client->input == 0 )
	{
		SparkModelResidentClientDestroy(client);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	*client_out = client;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientPrepareSocket(
	int32_t fd)
{
#if defined(SO_NOSIGPIPE)
	int32_t enabled;
	enabled = 1;
	if ( setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&enabled,sizeof(enabled)) != 0 )
		return(SPARK_STATUS_IO_ERROR);
#endif
	return(SparkModelResidentClientSetNonblocking(fd));
}

static SparkStatus SparkModelResidentClientFinishConnect(
	int32_t fd,
	const struct sockaddr *address,
	socklen_t address_bytes,
	uint32_t timeout_ms)
{
	int32_t error;
	socklen_t error_bytes;
	SparkStatus status;
	if ( connect(fd,address,address_bytes) == 0 )
		return(SPARK_STATUS_OK);
	if ( errno != EINPROGRESS )
		return(SPARK_STATUS_IO_ERROR);
	status = SparkModelResidentClientWait(fd,POLLOUT,timeout_ms);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = 0;
	error_bytes = sizeof(error);
	if ( getsockopt(fd,SOL_SOCKET,SO_ERROR,&error,&error_bytes) != 0 || error != 0 )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientOpenUnix(
	SparkModelResidentClient *client,
	const SparkModelResidentEndpoint *endpoint,
	uint32_t timeout_ms)
{
	struct sockaddr_un address;
	SparkStatus status;
	if ( strlen(endpoint->unix_socket_path) >= sizeof(address.sun_path) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	client->fd = socket(AF_UNIX,SOCK_STREAM,0);
	if ( client->fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	status = SparkModelResidentClientPrepareSocket(client->fd);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&address,0,sizeof(address));
	address.sun_family = AF_UNIX;
	memcpy(address.sun_path,endpoint->unix_socket_path,strlen(endpoint->unix_socket_path) + 1u);
	return(SparkModelResidentClientFinishConnect(client->fd,(const struct sockaddr *)&address,sizeof(address),timeout_ms));
}

static SparkStatus SparkModelResidentClientOpenTcp(
	SparkModelResidentClient *client,
	const SparkModelResidentEndpoint *endpoint,
	uint32_t timeout_ms)
{
	struct addrinfo hints,*addresses,*address;
	char service[16];
	int32_t enabled;
	SparkStatus status;
	memset(&hints,0,sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	if ( snprintf(service,sizeof(service),"%u",endpoint->tcp_port) < 0 || getaddrinfo(endpoint->tcp_host,service,&hints,&addresses) != 0 )
		return(SPARK_STATUS_ROUTE_NOT_FOUND);
	status = SPARK_STATUS_IO_ERROR;
	for (address=addresses; address!=0 && status!=SPARK_STATUS_OK; address=address->ai_next)
	{
		client->fd = socket(address->ai_family,address->ai_socktype,address->ai_protocol);
		if ( client->fd < 0 )
			continue;
		enabled = 1;
		if ( setsockopt(client->fd,IPPROTO_TCP,TCP_NODELAY,&enabled,sizeof(enabled)) == 0 )
			status = SparkModelResidentClientPrepareSocket(client->fd);
		if ( status == SPARK_STATUS_OK )
			status = SparkModelResidentClientFinishConnect(client->fd,address->ai_addr,(socklen_t)address->ai_addrlen,timeout_ms);
		if ( status != SPARK_STATUS_OK )
		{
			close(client->fd);
			client->fd = -1;
		}
	}
	freeaddrinfo(addresses);
	return(status);
}

static SparkStatus SparkModelResidentClientOpenEndpoint(
	SparkModelResidentClient *client,
	const SparkModelResidentEndpoint *endpoint,
	uint32_t timeout_ms)
{
	if ( endpoint->kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX )
		return(SparkModelResidentClientOpenUnix(client,endpoint,timeout_ms));
	if ( endpoint->kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP )
		return(SparkModelResidentClientOpenTcp(client,endpoint,timeout_ms));
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkModelResidentClientHandshake(
	SparkModelResidentClient *client,
	const SparkModelResidentClientConfiguration *configuration)
{
	SparkModelResidentIpcHello hello;
	SparkModelResidentIpcHelloAck ack;
	SparkStatus status;
	status = SparkModelResidentIpcInitializeHello(&hello,1u,configuration->rank_index,configuration->stage_index,configuration->adapter_descriptor);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentClientWriteFull(client->fd,&hello,sizeof(hello),configuration->connect_timeout_ms);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentClientReadFull(client->fd,&ack.header,sizeof(ack.header),configuration->connect_timeout_ms);
	if ( status == SPARK_STATUS_OK && ack.header.message_bytes != sizeof(ack) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentClientReadFull(client->fd,(uint8_t *)&ack + sizeof(ack.header),sizeof(ack) - sizeof(ack.header),configuration->connect_timeout_ms);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcValidateHelloAck(&ack,sizeof(ack),1u,configuration->rank_index,configuration->stage_index,configuration->adapter_descriptor,&configuration->runtime_limits);
	if ( status == SPARK_STATUS_OK && ack.status != SPARK_STATUS_OK )
		status = (SparkStatus)ack.status;
	if ( status == SPARK_STATUS_OK )
		client->client_generation = ack.client_generation;
	return(status);
}

SparkStatus SparkModelResidentClientConnect(
	const SparkModelResidentClientConfiguration *configuration,
	SparkModelResidentClient **client_out)
{
	SparkModelResidentClient *client;
	SparkStatus status;
	if ( client_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*client_out = 0;
	client = 0;
	status = SparkModelResidentClientValidateConfiguration(configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentClientAllocate(configuration,&client);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentClientOpenEndpoint(client,&configuration->endpoint,configuration->connect_timeout_ms);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentClientHandshake(client,configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelResidentClientDestroy(client);
		return(status);
	}
	client->connected = 1u;
	client->rank_index = configuration->rank_index;
	client->stage_index = configuration->stage_index;
	client->next_message_id = 2u;
	client->input_target_bytes = SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES;
	client->adapter_descriptor = configuration->adapter_descriptor;
	client->runtime_limits = configuration->runtime_limits;
	client->submit_result_function = configuration->submit_result_function;
	client->submit_result_context = configuration->submit_result_context;
	client->decision_result_function = configuration->decision_result_function;
	client->decision_result_context = configuration->decision_result_context;
	client->completion_function = configuration->completion_function;
	client->completion_context = configuration->completion_context;
	*client_out = client;
	return(SPARK_STATUS_OK);
}

void SparkModelResidentClientDestroy(SparkModelResidentClient *client)
{
	if ( client == 0 )
		return;
	if ( client->fd >= 0 )
		close(client->fd);
	free(client->input);
	free(client->output_storage);
	free(client->pending);
	free(client->outputs);
	free(client);
}

void SparkModelResidentClientFailStop(SparkModelResidentClient *client)
{
	if ( client == 0 || client->connected == 0u )
		return;
	client->connected = 0u;
	if ( client->fd >= 0 )
	{
		(void)shutdown(client->fd,SHUT_RDWR);
		close(client->fd);
		client->fd = -1;
	}
}

static SparkModelResidentClientPending *SparkModelResidentClientFindPending(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	uint32_t index;
	for (index=0u; index<client->queue_capacity; index++)
		if ( client->pending[index].active != 0u && client->pending[index].submission_id == submission_id )
			return(&client->pending[index]);
	return(0);
}

static const SparkModelResidentClientPending *
	SparkModelResidentClientFindPendingConst(
		const SparkModelResidentClient *client,
		uint64_t submission_id)
{
	uint32_t index;
	for (index=0u; index<client->queue_capacity; index++)
		if ( client->pending[index].active != 0u &&
			client->pending[index].submission_id == submission_id )
			return(&client->pending[index]);
	return(0);
}

static SparkModelResidentClientPending *SparkModelResidentClientReservePending(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission,
	uint32_t requires_decision)
{
	SparkModelResidentClientPending *pending;
	uint32_t index;
	if ( SparkModelResidentClientFindPending(client,submission->submission_id) != 0 )
		return(0);
	for (index=0u; index<client->queue_capacity; index++)
	{
		pending = &client->pending[index];
		if ( pending->active == 0u )
		{
			memset(pending,0,sizeof(*pending));
			pending->active = 1u;
				pending->requires_decision = requires_decision;
				pending->work_kind = submission->work_kind;
				pending->active_sequence_count = submission->active_sequence_count;
				pending->tokens_per_sequence = submission->tokens_per_sequence;
			pending->submission_id = submission->submission_id;
			pending->request_id = submission->request_id;
			pending->sequence_id = submission->sequence_id;
			pending->sequence_position = submission->sequence_position;
			pending->control_generation = submission->control_generation;
			pending->transaction_id = submission->transaction_id;
			pending->dispatch_generation = submission->dispatch_generation;
				pending->request_generation = submission->request_generation;
				pending->step_generation = submission->step_generation;
				pending->residency = submission->residency;
			pending->message_id = client->next_message_id++;
			client->pending_count++;
			return(pending);
		}
	}
	return(0);
}

static void SparkModelResidentClientReleasePending(
	SparkModelResidentClient *client,
	SparkModelResidentClientPending *pending)
{
	memset(pending,0,sizeof(*pending));
	client->pending_count--;
}

static SparkStatus SparkModelResidentClientSubmitKind(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission,
	uint32_t message_kind)
{
	SparkModelResidentClientPending *pending;
	SparkModelResidentClientOutput *output;
	uint8_t *message;
	uint32_t index,message_bytes;
	SparkStatus status;
	if ( client == 0 || client->connected == 0u || submission == 0 || submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u || submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelServingAdapterValidateRuntimeSubmission(client->adapter_descriptor,&client->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->submission_id <= client->last_submission_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->output_count >= client->queue_capacity || client->pending_count >= client->queue_capacity )
		return(SPARK_STATUS_BUSY);
	pending = SparkModelResidentClientReservePending(client,submission,
		message_kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE ? 1u : 0u);
	if ( pending == 0 )
		return(SPARK_STATUS_DUPLICATE);
	index = (client->output_head + client->output_count) % client->queue_capacity;
	output = &client->outputs[index];
	message = client->output_storage + ((uint64_t)index * client->output_message_capacity);
	if ( message_kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE )
		status = SparkModelResidentIpcEncodePreparation(submission,
			pending->message_id,message,client->output_message_capacity,
			&message_bytes);
	else if ( message_kind == SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE )
		status = SparkModelResidentIpcEncodeContinuation(submission,
			pending->message_id,client->client_generation,message,
			client->output_message_capacity,&message_bytes);
	else
		status = SparkModelResidentIpcEncodeSubmission(submission,
			pending->message_id,message,client->output_message_capacity,
			&message_bytes);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelResidentClientReleasePending(client,pending);
		return(status);
	}
	output->message_bytes = message_bytes;
	output->sent_bytes = 0u;
	pending->continued = message_kind == SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE ?
		1u : 0u;
	client->output_count++;
	client->submitted_count++;
	client->last_submission_id = submission->submission_id;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientSubmit(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	return(SparkModelResidentClientSubmitKind(client,submission,
		SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT));
}

SparkStatus SparkModelResidentClientPrepare(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	return(SparkModelResidentClientSubmitKind(client,submission,
		SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE));
}

SparkStatus SparkModelResidentClientCanQueueContinuation(
	const SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	uint32_t message_bytes;
	SparkStatus status;
	if ( client == 0 || client->connected == 0u || submission == 0 ||
		client->client_generation == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (client->adapter_descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	status = SparkModelServingAdapterValidateRuntimeSubmission(
		client->adapter_descriptor,&client->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->submission_id <= client->last_submission_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->output_count >= client->queue_capacity ||
		client->pending_count >= client->queue_capacity )
		return(SPARK_STATUS_BUSY);
	status = SparkModelResidentIpcCalculateSubmitBytes(submission->lane_count,
		submission->row_count,submission->model_extension_bytes,&message_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(message_bytes <= client->output_message_capacity ? SPARK_STATUS_OK :
		SPARK_STATUS_CAPACITY_EXCEEDED);
}

SparkStatus SparkModelResidentClientContinue(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	status = SparkModelResidentClientCanQueueContinuation(client,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkModelResidentClientSubmitKind(client,submission,
		SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE));
}

static SparkStatus SparkModelResidentClientQueueDecision(
	SparkModelResidentClient *client,
	SparkModelResidentClientPending *pending,
	uint32_t decision_kind)
{
	SparkModelResidentIpcDecision *decision;
	SparkModelResidentClientOutput *output;
	SparkModelServingSubmission submission;
	uint8_t *message;
	uint32_t index;
	SparkStatus status;
	status = SparkModelResidentClientCanQueueDecision(client,
		pending->submission_id,decision_kind);
	if ( status != SPARK_STATUS_OK )
		return(status);
	index = (client->output_head + client->output_count) % client->queue_capacity;
	output = &client->outputs[index];
	message = client->output_storage + ((uint64_t)index * client->output_message_capacity);
	memset(&submission,0,sizeof(submission));
	submission.submission_id = pending->submission_id;
	submission.control_generation = pending->control_generation;
	submission.transaction_id = pending->transaction_id;
	submission.dispatch_generation = pending->dispatch_generation;
	decision = (SparkModelResidentIpcDecision *)message;
	status = SparkModelResidentIpcInitializeDecision(decision,client->next_message_id,decision_kind,&submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending->decision_message_id = client->next_message_id++;
	pending->decision_kind = decision_kind;
	output->message_bytes = SPARK_MODEL_RESIDENT_IPC_DECISION_BYTES;
	output->sent_bytes = 0u;
	client->output_count++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientCanQueueDecision(
	const SparkModelResidentClient *client,
	uint64_t submission_id,
	uint32_t decision_kind)
{
	const SparkModelResidentClientPending *pending;
	SparkModelResidentIpcDecision decision;
	SparkModelServingSubmission submission;
	if ( client == 0 || client->connected == 0u || submission_id == 0u ||
		(decision_kind != SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT &&
		 decision_kind != SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pending = SparkModelResidentClientFindPendingConst(client,submission_id);
	if ( pending == 0 || pending->requires_decision == 0u ||
		pending->prepared == 0u || pending->committed != 0u ||
		pending->decision_kind != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->output_count >= client->queue_capacity )
		return(SPARK_STATUS_BUSY);
	if ( client->output_message_capacity < SPARK_MODEL_RESIDENT_IPC_DECISION_BYTES )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&submission,0,sizeof(submission));
	submission.submission_id = pending->submission_id;
	submission.control_generation = pending->control_generation;
	submission.transaction_id = pending->transaction_id;
	submission.dispatch_generation = pending->dispatch_generation;
	return(SparkModelResidentIpcInitializeDecision(&decision,
		client->next_message_id,decision_kind,&submission));
}

SparkStatus SparkModelResidentClientCommit(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	SparkModelResidentClientPending *pending;
	SparkStatus status;
	if ( client == 0 || client->connected == 0u || submission_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pending = SparkModelResidentClientFindPending(client,submission_id);
	if ( pending == 0 || pending->requires_decision == 0u || pending->prepared == 0u || pending->committed != 0u || pending->decision_kind != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelResidentClientQueueDecision(client,pending,SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientAbort(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	SparkModelResidentClientPending *pending;
	SparkStatus status;
	if ( client == 0 || client->connected == 0u || submission_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pending = SparkModelResidentClientFindPending(client,submission_id);
	if ( pending == 0 || pending->requires_decision == 0u || pending->prepared == 0u || pending->committed != 0u || pending->decision_kind != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelResidentClientQueueDecision(client,pending,SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientFlush(SparkModelResidentClient *client)
{
	SparkModelResidentClientOutput *output;
	const uint8_t *message;
	ssize_t written;
	while ( client->output_count != 0u )
	{
		output = &client->outputs[client->output_head];
		message = client->output_storage + ((uint64_t)client->output_head * client->output_message_capacity);
		written = SparkModelResidentClientSend(client->fd,message + output->sent_bytes,output->message_bytes - output->sent_bytes);
		if ( written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) )
			return(SPARK_STATUS_OK);
		if ( written <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		output->sent_bytes += (uint32_t)written;
		if ( output->sent_bytes != output->message_bytes )
			return(SPARK_STATUS_OK);
		client->output_head = (client->output_head + 1u) % client->queue_capacity;
		client->output_count--;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientProcessResult(
	SparkModelResidentClient *client,
	const SparkModelResidentIpcSubmitResult *result,
	uint32_t message_bytes)
{
	SparkModelResidentClientPending *pending;
	SparkStatus status;
	status = SparkModelResidentIpcValidateHeader(result != 0 ? &result->header : 0,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT_RESULT,SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SparkModelResidentClientFindPending(client,result->submission_id);
	if ( pending == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	status = SparkModelResidentIpcValidateSubmitResult(result,message_bytes,pending->message_id,pending->submission_id);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( pending->result_received != 0u )
		return(SPARK_STATUS_DUPLICATE);
	pending->result_received = 1u;
	status = (SparkStatus)result->status;
	if ( status == SPARK_STATUS_OK )
	{
		if ( pending->requires_decision != 0u )
		{
			pending->prepared = 1u;
			client->prepared_count++;
			client->prepared_total++;
		}
		else
		{
			pending->committed = 1u;
			client->admitted_count++;
			if ( pending->continued != 0u )
				client->continued_total++;
		}
	}
	else
	{
		client->rejected_count++;
		SparkModelResidentClientReleasePending(client,pending);
	}
	if ( client->submit_result_function != 0 )
		client->submit_result_function(client->submit_result_context,result->submission_id,status);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientProcessCompletion(
	SparkModelResidentClient *client,
	const void *message,
	uint32_t message_bytes)
{
	SparkModelServingCompletion completion;
	SparkModelResidentClientPending *pending;
	SparkStatus status;
	status = SparkModelResidentIpcDecodeCompletion(message,message_bytes,&completion);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SparkModelResidentClientFindPending(client,completion.submission_id);
	if ( pending == 0 || pending->committed == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	status = SparkModelServingAdapterValidateStageCompletion(client->adapter_descriptor,client->stage_index,pending->work_kind,pending->active_sequence_count,pending->tokens_per_sequence,&pending->residency,&completion);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( ((const SparkModelResidentIpcHeader *)message)->message_id != pending->message_id || pending->request_id != completion.request_id || pending->sequence_id != completion.sequence_id || pending->sequence_position != completion.sequence_position || pending->control_generation != completion.control_generation || pending->transaction_id != completion.transaction_id || pending->dispatch_generation != completion.dispatch_generation || pending->request_generation != completion.request_generation || pending->step_generation != completion.step_generation )
		return(SPARK_STATUS_SCHEMA_ERROR);
	client->completed_count++;
	SparkModelResidentClientReleasePending(client,pending);
	client->completion_function(client->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientProcessDecisionResult(
	SparkModelResidentClient *client,
	const SparkModelResidentIpcDecisionResult *result,
	uint32_t message_bytes)
{
	SparkModelResidentClientPending *pending;
	SparkModelServingSubmission submission;
	SparkModelResidentDecisionResultFunction callback;
	void *callback_context;
	uint64_t submission_id;
	uint32_t decision_kind,release;
	SparkStatus status;
	pending = result != 0 ? SparkModelResidentClientFindPending(client,result->submission_id) : 0;
	if ( pending == 0 || pending->prepared == 0u || pending->decision_kind == 0u || pending->decision_result_received != 0u )
		return(SPARK_STATUS_NOT_FOUND);
	memset(&submission,0,sizeof(submission));
	submission.submission_id = pending->submission_id;
	submission.control_generation = pending->control_generation;
	submission.transaction_id = pending->transaction_id;
	submission.dispatch_generation = pending->dispatch_generation;
	status = SparkModelResidentIpcValidateDecisionResult(result,message_bytes,pending->decision_message_id,pending->decision_kind,&submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending->decision_result_received = 1u;
	status = (SparkStatus)result->status;
	decision_kind = pending->decision_kind;
	submission_id = pending->submission_id;
	client->prepared_count--;
	release = status != SPARK_STATUS_OK || decision_kind == SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT ? 1u : 0u;
	if ( status == SPARK_STATUS_OK && decision_kind == SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT )
	{
		pending->prepared = 0u;
		pending->committed = 1u;
		client->admitted_count++;
	}
	else if ( status == SPARK_STATUS_OK )
		client->aborted_count++;
	else
		client->rejected_count++;
	callback = client->decision_result_function;
	callback_context = client->decision_result_context;
	if ( release != 0u )
		SparkModelResidentClientReleasePending(client,pending);
	if ( callback != 0 )
		callback(callback_context,submission_id,decision_kind,status);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentClientProcessMessage(
	SparkModelResidentClient *client,
	const void *message,
	uint32_t message_bytes)
{
	const SparkModelResidentIpcHeader *header;
	header = (const SparkModelResidentIpcHeader *)message;
	if ( header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT_RESULT )
		return(SparkModelResidentClientProcessResult(client,(const SparkModelResidentIpcSubmitResult *)message,message_bytes));
	if ( header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_COMPLETION )
		return(SparkModelResidentClientProcessCompletion(client,message,message_bytes));
	if ( header->kind == SPARK_MODEL_RESIDENT_IPC_KIND_DECISION_RESULT )
		return(SparkModelResidentClientProcessDecisionResult(client,(const SparkModelResidentIpcDecisionResult *)message,message_bytes));
	return(SPARK_STATUS_UNSUPPORTED);
}

static SparkStatus SparkModelResidentClientRead(
	SparkModelResidentClient *client,
	uint32_t maximum_message_count)
{
	SparkModelResidentIpcHeader *header;
	uint32_t processed;
	ssize_t received;
	SparkStatus status;
	processed = 0u;
	while ( processed < maximum_message_count )
	{
		received = recv(client->fd,client->input + client->input_bytes,client->input_target_bytes - client->input_bytes,0);
		if ( received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) )
			return(SPARK_STATUS_OK);
		if ( received <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		client->input_bytes += (uint32_t)received;
		if ( client->input_bytes == SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES && client->input_target_bytes == SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES )
		{
			header = (SparkModelResidentIpcHeader *)client->input;
			if ( header->message_bytes < SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES || header->message_bytes > client->input_capacity )
				return(SPARK_STATUS_SCHEMA_ERROR);
			client->input_target_bytes = header->message_bytes;
		}
		if ( client->input_bytes == client->input_target_bytes )
		{
			status = SparkModelResidentClientProcessMessage(client,client->input,client->input_target_bytes);
			if ( status != SPARK_STATUS_OK )
				return(status);
			client->input_bytes = 0u;
			client->input_target_bytes = SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES;
			processed++;
		}
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientProgress(
	SparkModelResidentClient *client,
	uint32_t maximum_message_count)
{
	SparkStatus status;
	if ( client == 0 || client->connected == 0u || maximum_message_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelResidentClientFlush(client);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentClientRead(client,maximum_message_count);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelResidentClientFailStop(client);
	}
	return(status);
}

SparkStatus SparkModelResidentClientGetPollDescriptor(
	const SparkModelResidentClient *client,
	SparkModelResidentClientPollDescriptor *descriptor)
{
	if ( client == 0 || descriptor == 0 || client->connected == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_RESIDENT_CLIENT_POLL_DESCRIPTOR_BYTES;
	descriptor->fd = client->fd;
	descriptor->events = SPARK_MODEL_RESIDENT_CLIENT_POLL_READ;
	if ( client->output_count != 0u )
		descriptor->events |= SPARK_MODEL_RESIDENT_CLIENT_POLL_WRITE;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientGetView(
	const SparkModelResidentClient *client,
	SparkModelResidentClientView *view)
{
	if ( client == 0 || view == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION;
	view->descriptor_bytes = SPARK_MODEL_RESIDENT_CLIENT_VIEW_BYTES;
	view->connected = client->connected;
	view->queued_message_count = client->output_count;
	view->pending_submission_count = client->pending_count;
	view->queue_capacity = client->queue_capacity;
	view->max_active_sequence_count = client->runtime_limits.max_active_sequence_count;
	view->max_input_row_count = client->runtime_limits.max_input_row_count;
	view->resident_sequence_capacity = client->runtime_limits.resident_sequence_capacity;
	view->kv_logical_page_capacity =
		client->runtime_limits.kv_logical_page_capacity;
	view->kv_physical_page_capacity =
		client->runtime_limits.kv_physical_page_capacity;
	view->prepared_submission_count = client->prepared_count;
	view->client_generation = client->client_generation;
	view->submitted_count = client->submitted_count;
	view->prepared_count = client->prepared_total;
	view->continued_count = client->continued_total;
	view->admitted_count = client->admitted_count;
	view->rejected_count = client->rejected_count;
	view->aborted_count = client->aborted_count;
	view->completed_count = client->completed_count;
	return(SPARK_STATUS_OK);
}
