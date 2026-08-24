/* model_api - the standard API entry point (2026-08-23).
 *
 * The problem this solves: the only way to drive the model was
 * model_residentd's private IPC with hand-built batch JSON files, and
 * because EVERY caller dialed the daemon directly, concurrent clients
 * collided on the daemon's single-client slot/submission state (measured:
 * a second client's first claim -> INVALID_ARGUMENT) and dead sessions
 * poisoned resident slots ("all cells fail until restart").
 *
 * The architecture: ONE long-lived engine session (the same client
 * library model_batch uses) owned by this process. External callers
 * speak OpenAI-style HTTP; every request multiplexes through the single
 * session, so there are no client collisions and no session churn.
 *
 * Endpoints (v0 - token-id in/out; text needs a tokenizer sidecar):
 *   GET  /health              -> {"status":"ok","completed":N}
 *   POST /v1/completions      -> {"prompt_token_ids":[...],"max_tokens":N}
 *      response: {"object":"text_completion","tokens":[...],"count":N}
 *   POST /v1/chat/completions -> same shape (chat template is the
 *      caller's job in v0: send the templated token ids)
 *
 * Build: make build/sparkpipe_model_api
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <poll.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#define MODEL_API_MAX_BODY (8u * 1024u * 1024u)
#define MODEL_API_MAX_PROMPT_TOKENS (260000u)
#define MODEL_API_MAX_OUTPUT_TOKENS (8192u)
#define MODEL_API_INFLIGHT 32u

typedef struct ModelApiRequest
{
	uint64_t request_id;
	uint32_t *token_ids;
	uint32_t token_count;
	uint32_t max_tokens;
	char response[48u * 1024u];
	uint32_t response_length;
	volatile int done;
	volatile uint32_t status;
} ModelApiRequest;

typedef struct ModelApiState
{
	SparkModelBatchEngine *engine;
	pthread_mutex_t mutex;
	ModelApiRequest *inflight[MODEL_API_INFLIGHT];
	volatile int stopping;
	uint64_t next_request_id;
	uint64_t completed;
} ModelApiState;

static ModelApiState api_state;

/* ---------------- engine events (called from EngineProgress) --------- */

static void model_api_event(void *context,const SparkModelBatchEvent *event)
{
	uint32_t index;
	(void)context;
	if ( event == 0 )
		return;
	pthread_mutex_lock(&api_state.mutex);
	for ( index = 0u; index < MODEL_API_INFLIGHT; index++ )
	{
		ModelApiRequest *request = api_state.inflight[index];
		if ( request == 0 || request->request_id != event->request_id )
			continue;
		if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN && request->response_length + 12u < sizeof(request->response) )
			request->response_length += (uint32_t)snprintf(request->response + request->response_length,
				sizeof(request->response) - request->response_length,"%s%u",
				request->response_length != 0u ? "," : "",(unsigned)event->token_id);
		if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED ||
			event->kind == SPARK_MODEL_BATCH_EVENT_ERROR )
		{
			request->status = event->status;
			request->done = 1;
			api_state.completed++;
		}
		break;
	}
	pthread_mutex_unlock(&api_state.mutex);
}

/* ---------------- request execution on the single session ------------ */

static uint32_t model_api_execute(ModelApiRequest *request)
{
	SparkModelBatchSubmitRequest submit;
	SparkModelBatchRequestHandle handle;
	SparkStatus status;
	uint32_t slot = 0u,index;
	pthread_mutex_lock(&api_state.mutex);
	for ( index = 0u; index < MODEL_API_INFLIGHT; index++ )
		if ( api_state.inflight[index] == 0 )
		{
			api_state.inflight[index] = request;
			slot = index + 1u;
			break;
		}
	pthread_mutex_unlock(&api_state.mutex);
	if ( slot == 0u )
		return(503u);
	memset(&submit,0,sizeof(submit));
	submit.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	submit.descriptor_bytes = (uint32_t)sizeof(submit);
	submit.request_id = request->request_id;
	submit.sequence_id = request->request_id;
	submit.output_token_budget = request->max_tokens;
	submit.prompt_token_ids = request->token_ids;
	submit.prompt_token_count = request->token_count;
	status = SparkModelBatchEngineSubmit(api_state.engine,&submit,&handle);
	if ( status == SPARK_STATUS_OK )
		(void)SparkModelBatchEngineCloseAdmission(api_state.engine);
	if ( status != SPARK_STATUS_OK )
	{
		pthread_mutex_lock(&api_state.mutex);
		api_state.inflight[slot - 1u] = 0;
		pthread_mutex_unlock(&api_state.mutex);
		return(500u);
	}
	/* EXACT model_batch Run shape: Progress + poll descriptors. The
	 * pipeline client needs its socket polled between Progress calls
	 * to process completions (the tight Progress-only spin verified to
	 * stall after prefill - completions sit unread). */
	while ( request->done == 0 && api_state.stopping == 0 )
	{
		SparkModelResidentClientPollDescriptor descriptors[4];
		uint32_t descriptor_count = 0u;
		status = SparkModelBatchEngineProgress(api_state.engine,4u);
		if ( status != SPARK_STATUS_OK )
		{
			request->done = 1;
			request->status = (uint32_t)status;
			break;
		}
		status = SparkModelBatchEngineGetPollDescriptors(api_state.engine,
			descriptors,4u,&descriptor_count);
		if ( status == SPARK_STATUS_OK && descriptor_count > 0u )
		{
			struct pollfd poll_descriptors[4];
			uint32_t index2;
			for ( index2 = 0u; index2 < descriptor_count; index2++ )
			{
				poll_descriptors[index2].fd = descriptors[index2].fd;
				poll_descriptors[index2].events = 0;
				poll_descriptors[index2].revents = 0;
				if ( (descriptors[index2].events & 1u) != 0u )
					poll_descriptors[index2].events |= POLLIN;
				if ( (descriptors[index2].events & 2u) != 0u )
					poll_descriptors[index2].events |= POLLOUT;
			}
			(void)poll(poll_descriptors,(nfds_t)descriptor_count,10);
		}
		else
			usleep(1000);
	}
	pthread_mutex_lock(&api_state.mutex);
	api_state.inflight[slot - 1u] = 0;
	pthread_mutex_unlock(&api_state.mutex);
	return(request->status == SPARK_STATUS_OK ? 200u : 500u);
}

/* ---------------- HTTP plumbing (minimal blocking server) ------------ */

static void model_api_send_all(int fd,const char *data,size_t bytes)
{
	size_t sent = 0u;
	while ( sent < bytes )
	{
		ssize_t n = send(fd,data + sent,bytes - sent,MSG_NOSIGNAL);
		if ( n <= 0 )
			return;
		sent += (size_t)n;
	}
}

static void model_api_respond(int fd,int code,const char *body)
{
	char header[160];
	int written = snprintf(header,sizeof(header),
		"HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
		code,code == 200 ? "OK" : "Error",strlen(body));
	if ( written > 0 )
		model_api_send_all(fd,header,(size_t)written);
	model_api_send_all(fd,body,strlen(body));
}

static int model_api_read_body(int fd,char *body,uint32_t *body_length)
{
	char header[8192];
	uint32_t header_bytes = 0u,content_length = 0u;
	ssize_t n;
	while ( header_bytes + 1u < sizeof(header) )
	{
		n = recv(fd,header + header_bytes,1u,0);
		if ( n <= 0 )
			return(0);
		header_bytes += (uint32_t)n;
		if ( header_bytes >= 4u && memcmp(header + header_bytes - 4u,"\r\n\r\n",4u) == 0 )
			break;
	}
	header[header_bytes] = '\0';
	{
		const char *cl = strstr(header,"Content-Length:");
		if ( cl != 0 )
			content_length = (uint32_t)strtoul(cl + 15u,0,10);
	}
	if ( content_length == 0u || content_length > MODEL_API_MAX_BODY )
		return(0);
	while ( *body_length < content_length )
	{
		n = recv(fd,body + *body_length,content_length - *body_length,0);
		if ( n <= 0 )
			return(0);
		*body_length += (uint32_t)n;
	}
	body[*body_length] = '\0';
	return(1);
}

static uint32_t model_api_tokens_from_json(SparkJsonDocument *document,int32_t root,
	const char *name,uint32_t **tokens_out)
{
	int32_t member,index;
	uint32_t value,count;
	uint32_t *tokens;
	member = SparkJsonFindObjectMember(document,root,name);
	if ( member < 0 || !SparkJsonTokenIsType(document,member,SPARK_JSON_TOKEN_ARRAY) )
		return(0u);
	count = (uint32_t)SparkJsonGetArrayElementCount(document,member);
	if ( count == 0u || count > MODEL_API_MAX_PROMPT_TOKENS )
		return(0u);
	tokens = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
	if ( tokens == 0 )
		return(0u);
	for ( index = 0u; index < (int32_t)count; index++ )
		if ( SparkJsonGetUInt32(document,SparkJsonGetArrayElement(document,member,index),&value) != SPARK_STATUS_OK ||
			value > 260000u )
		{
			free(tokens);
			return(0u);
		}
		else
			tokens[index] = value;
	*tokens_out = tokens;
	return(count);
}

static void model_api_handle_completions(int fd,char *body,uint32_t body_bytes)
{
	SparkJsonDocument document;
	int32_t root;
	uint32_t *prompt = 0;
	uint32_t prompt_count = 0u,max_tokens = 32u;
	ModelApiRequest *request;
	uint32_t code;
	char *response;
	if ( SparkJsonParseText(body,body_bytes,&document) != SPARK_STATUS_OK )
	{
		model_api_respond(fd,400,"{\"error\":\"invalid json\"}");
		return;
	}
	root = SparkJsonGetRootToken(&document);
	prompt_count = model_api_tokens_from_json(&document,root,"prompt_token_ids",&prompt);
	if ( root >= 0 && prompt_count != 0u )
	{
		int32_t member = SparkJsonFindObjectMember(&document,root,"max_tokens");
		uint32_t value;
		if ( member >= 0 && SparkJsonGetUInt32(&document,member,&value) == SPARK_STATUS_OK && value != 0u )
			max_tokens = value > MODEL_API_MAX_OUTPUT_TOKENS ? MODEL_API_MAX_OUTPUT_TOKENS : value;
	}
	if ( prompt_count == 0u )
	{
		SparkJsonDocumentDestroy(&document);
		model_api_respond(fd,400,"{\"error\":\"prompt_token_ids required (v0: token-id in/out)\"}");
		return;
	}
	request = (ModelApiRequest *)calloc(1u,sizeof(*request));
	if ( request == 0 )
	{
		free(prompt);
		SparkJsonDocumentDestroy(&document);
		model_api_respond(fd,500,"{\"error\":\"out of memory\"}");
		return;
	}
	request->request_id = ++api_state.next_request_id + 100000u;
	request->token_ids = prompt;
	request->token_count = prompt_count;
	request->max_tokens = max_tokens;
	code = model_api_execute(request);
	response = (char *)malloc(request->response_length + 128u);
	if ( response != 0 )
	{
		(void)snprintf(response,request->response_length + 128u,
			"{\"object\":\"text_completion\",\"tokens\":[%s],\"status\":%u}",
			request->response,request->status);
		model_api_respond(fd,code == 200u ? 200 : 500,response[0] != 0 ? response : "{}");
		free(response);
	}
	else
		model_api_respond(fd,500,"{\"error\":\"out of memory\"}");
	free(request->token_ids);
	free(request);
	SparkJsonDocumentDestroy(&document);
}

static void *model_api_connection(void *context)
{
	int fd = (int)(intptr_t)context;
	char header[8192];
	uint32_t header_bytes = 0u;
	ssize_t n;
	char *body = 0;
	uint32_t body_bytes = 0u;
	int is_get = 0,is_post = 0;
	while ( header_bytes + 1u < sizeof(header) )
	{
		n = recv(fd,header + header_bytes,1u,0);
		if ( n <= 0 )
			break;
		header_bytes += (uint32_t)n;
		if ( header_bytes >= 4u && memcmp(header + header_bytes - 4u,"\r\n\r\n",4u) == 0 )
			break;
	}
	if ( header_bytes == 0u )
	{
		close(fd);
		return(0);
	}
	header[header_bytes] = '\0';
	is_get = header_bytes > 4u && memcmp(header,"GET ",4u) == 0;
	is_post = header_bytes > 5u && memcmp(header,"POST ",5u) == 0;
	if ( is_get && strstr(header,"/health") != 0 )
	{
		char payload[96u];
		(void)snprintf(payload,sizeof(payload),"{\"status\":\"ok\",\"completed\":%llu}",
			(unsigned long long)api_state.completed);
		model_api_respond(fd,200,payload);
		close(fd);
		return(0);
	}
	if ( !is_post || (strstr(header,"/v1/completions") == 0 && strstr(header,"/v1/chat/completions") == 0) )
	{
		model_api_respond(fd,404,"{\"error\":\"see /health, POST /v1/completions\"}");
		close(fd);
		return(0);
	}
	body = (char *)malloc(MODEL_API_MAX_BODY + 1u);
	if ( body != 0 && model_api_read_body(fd,body,&body_bytes) )
		model_api_handle_completions(fd,body,body_bytes);
	else
		model_api_respond(fd,400,"{\"error\":\"body read failed\"}");
	free(body);
	close(fd);
	return(0);
}

int main(int argc,char **argv)
{
	const char *deployment_path = 0;
	const char *runtime_root = 0;
	const char *listen_port = "8080";
	SparkModelResidentDeployment deployment;
	SparkModelBatchEngineConfiguration configuration;
	int server_fd;
	int index;
	for ( index = 1; index < argc; index++ )
	{
		if ( strcmp(argv[index],"--deployment") == 0 && index + 1 < argc )
			deployment_path = argv[++index];
		else if ( strcmp(argv[index],"--runtime-root") == 0 && index + 1 < argc )
			runtime_root = argv[++index];
		else if ( strcmp(argv[index],"--port") == 0 && index + 1 < argc )
			listen_port = argv[++index];
	}
	if ( deployment_path == 0 || runtime_root == 0 )
	{
		fprintf(stderr,"usage: %s --deployment PATH --runtime-root PATH [--port N]\n",argv[0]);
		return(1);
	}
	SparkModelResidentDeploymentReset(&deployment);
	if ( SparkModelResidentDeploymentLoad(deployment_path,&deployment) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"model_api: deployment load failed: %s\n",deployment_path);
		return(1);
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	configuration.descriptor_bytes = (uint32_t)sizeof(configuration);
	configuration.deployment = &deployment;
	configuration.runtime_root = runtime_root;
	configuration.request_capacity = 16u;
	configuration.max_context_tokens = 8192u;
	configuration.max_prefill_rows_per_submission = 8u;
	configuration.connect_timeout_ms = 30000u;
	configuration.maximum_messages_per_rank_per_progress = 16u;
	configuration.event_function = model_api_event;
	configuration.event_context = 0;
	if ( SparkModelBatchEngineConnect(&configuration,&api_state.engine) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"model_api: engine connect failed (is the daemon up?)\n");
		return(1);
	}
	signal(SIGPIPE,SIG_IGN);
	{
		struct sockaddr_in address;
		int enabled = 1;
		server_fd = socket(AF_INET,SOCK_STREAM,0);
		(void)setsockopt(server_fd,SOL_SOCKET,SO_REUSEADDR,&enabled,sizeof(enabled));
		memset(&address,0,sizeof(address));
		address.sin_family = AF_INET;
		address.sin_addr.s_addr = htonl(INADDR_ANY);
		address.sin_port = htons((uint16_t)strtoul(listen_port,0,10));
		if ( bind(server_fd,(struct sockaddr *)&address,sizeof(address)) != 0 ||
			listen(server_fd,64) != 0 )
		{
			fprintf(stderr,"model_api: listen failed on port %s: %s\n",listen_port,strerror(errno));
			return(1);
		}
	}
	fprintf(stderr,"model_api ready port=%s (single engine session - the ONLY daemon client)\n",listen_port);
	for ( ;; )
	{
		int client_fd = accept(server_fd,0,0);
		pthread_t thread;
		if ( client_fd < 0 )
			continue;
		{
			int enabled = 1;
			(void)setsockopt(client_fd,IPPROTO_TCP,TCP_NODELAY,&enabled,sizeof(enabled));
		}
		if ( pthread_create(&thread,0,model_api_connection,(void *)(intptr_t)client_fd) == 0 )
			pthread_detach(thread);
		else
			close(client_fd);
	}
	(void)SparkModelBatchEngineDestroy(api_state.engine);
	return(0);
}
