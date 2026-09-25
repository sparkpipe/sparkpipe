#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdarg.h>
#include <time.h>
#include <fcntl.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_tp_chain_ordinal.h"
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_tokenizer_sidecar.h"

#define API_MAX_BODY		(8u * 1024u * 1024u)
#define API_MAX_PROMPT_TOKENS	(260000u)
#define API_MAX_STOP_TOKENS 16
#define API_MAX_OUTPUT_TOKENS	(8192u)
#define API_TOKEN_BUF_BYTES	(64u * 1024u)

#define API_MAX_INFLIGHT 16u

#define API_DEFAULT_MODEL_ID "sparkpipe-model"
#define API_WAIT_SLICE_MS 250u
#define API_STREAM_BATCH_TOKENS 64u
#define API_UTF8_MAX_SEQUENCE 4u
#define API_STREAM_EVENT_END "]}\n\n"
#define API_WAIT_DONE 0u
#define API_WAIT_TOKENS 1u
#define API_WAIT_ADMISSION 2u
#define API_BUSY_RETRY_MS 5
#define API_SEQUENCE_SAVE_MS 60000u

typedef struct ApiRequest
{
	uint64_t id;
	uint64_t started_ms;
	uint32_t *prompt_tokens;
	uint32_t prompt_count;
	uint32_t max_tokens;
	char tokens_json[API_TOKEN_BUF_BYTES];
	volatile uint32_t tokens_json_len;
	uint32_t *output_token_ids;
	uint64_t token_ready_ns[API_MAX_OUTPUT_TOKENS];
	uint64_t accepted_ns;
	uint32_t cached_prompt_token_count;
	uint32_t engine_completed;
	volatile uint32_t output_token_count;
	volatile int done;
	volatile int submitted;
	volatile int inflight;
	volatile int orphaned;
	uint32_t cancel_pending;
	volatile uint32_t status;
	SparkModelBatchRequestHandle handle;
	uint32_t *stop_tokens;
	uint32_t stop_token_count;
	uint32_t priority;
	uint32_t stream;
	uint32_t deadline_expired;
	uint64_t deadline_ms;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	struct ApiRequest *next;
} ApiRequest;

typedef struct ApiState
{
	SparkModelBatchEngine *engine;
	pthread_mutex_t queue_mutex;
	int wake_fds[2];
	ApiRequest *queue_head;
	ApiRequest *queue_tail;
	volatile int running;
	uint64_t next_id;
	uint64_t served;
	const char *runtime_root;
	uint64_t seq_saved_ms;
} ApiState;

static ApiState S;

static SparkTokenizerSidecar Sidecar;
static int HaveSidecar;
static uint32_t EngineStopTokens[SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT];
static uint32_t EngineStopTokenCount;
static char ApiBootTag[32];
static volatile uint32_t ApiSessionsAccepted;

static void api_logf(const char *format, ...)
{
	va_list args;
	flockfile(stderr);
	fprintf(stderr, "model_api[%s] ", ApiBootTag);
	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);
	fputc('\n', stderr);
	fflush(stderr);
	funlockfile(stderr);
}

static uint64_t api_now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

static void api_term_signal(int signal_number)
{
	const char line[] = "model_api api_exit reason=signal\n";
	ssize_t written;
	(void)signal_number;
	written = write(2, line, sizeof(line) - 1u);
	(void)written;
	_exit(0);
}

static const char *api_finish_reason(const ApiRequest *req)
{
	return req->output_token_count >= req->max_tokens ? "length" : "stop";
}

static const char *api_identity(const char *value)
{
	return value != 0 ? value : "";
}

static void api_log_request_measurements(const ApiRequest *request)
{
	const SparkModelServingAdapterDescriptor *adapter = S.engine != 0 ? SparkModelBatchEngineGetAdapterDescriptor(S.engine) : 0;
	char prompt_sha256[SPARK_SHA256_HEX_BYTES];
	uint32_t index;
	if (SparkSha256Bytes(request->prompt_tokens,(size_t)request->prompt_count * sizeof(uint32_t),prompt_sha256) != SPARK_STATUS_OK)
		prompt_sha256[0] = '\0';
	flockfile(stderr);
	fprintf(stderr,"{\"event\":\"request_measurements\",\"boot_pid\":%d,\"request_id\":%llu,\"status\":%u,\"engine_completed\":%u,\"accepted_ns\":%llu,\"prompt_tokens\":%u,\"cached_prompt_tokens\":%u,\"prompt_sha256\":\"%s\",\"adapter_id\":\"%s\",\"model_id\":\"%s\",\"model_revision\":\"%s\",\"driver_program\":\"%s\",\"driver_artifact_sha256\":\"%s\",\"session_fingerprint\":%llu,\"priority\":%u,\"deadline_expired\":%u,\"stream\":%u,\"finish_reason\":\"%s\",\"tokens\":[",(int)getpid(),(unsigned long long)request->id,request->status,request->engine_completed,(unsigned long long)request->accepted_ns,request->prompt_count,request->cached_prompt_token_count,prompt_sha256,api_identity(adapter != 0 ? adapter->adapter_id : 0),api_identity(adapter != 0 ? adapter->model_id : 0),api_identity(adapter != 0 ? adapter->model_revision : 0),api_identity(adapter != 0 ? adapter->driver_program_name : 0),api_identity(adapter != 0 ? adapter->artifact_sha256 : 0),(unsigned long long)(S.engine != 0 ? SparkModelBatchEngineSessionFingerprint(S.engine) : 0u),request->priority,request->deadline_expired,request->stream,request->status == 0u && request->deadline_expired == 0u ? api_finish_reason(request) : "error");
	for (index=0u; index<request->output_token_count; index++)
		fprintf(stderr,"%s[%u,%llu]",index == 0u ? "" : ",",request->output_token_ids[index],(unsigned long long)request->token_ready_ns[index]);
	fputs("]}\n",stderr);
	funlockfile(stderr);
}

static void api_request_destroy(ApiRequest *req)
{
	pthread_mutex_destroy(&req->mutex);
	pthread_cond_destroy(&req->cond);
	free(req->stop_tokens);
	free(req->prompt_tokens);
	free(req->output_token_ids);
	free(req);
}

static void api_queue_unlink(ApiRequest *req)
{
	ApiRequest **pp = &S.queue_head;
	while (*pp != 0)
	{
		if (*pp == req)
		{
			*pp = req->next;
			if (S.queue_tail == req)
			{
				ApiRequest *pred = S.queue_head;
				S.queue_tail = 0;
				while (pred != 0)
				{
					S.queue_tail = pred;
					pred = pred->next;
				}
			}
			req->next = 0;
			return;
		}
		pp = &(*pp)->next;
	}
}


static void api_event(void *ctx, const SparkModelBatchEvent *ev)
{
	ApiRequest *r;
	(void)ctx;
	if (ev == 0)
		return;
	pthread_mutex_lock(&S.queue_mutex);
	for (r = S.queue_head; r != 0; r = r->next)
	{
		if (r->id != ev->request_id)
			continue;
		r->cached_prompt_token_count = ev->cached_prompt_token_count;
		if ( ev->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED )
			r->accepted_ns = ev->monotonic_ns;
		if (ev->kind == SPARK_MODEL_BATCH_EVENT_TOKEN)
		{
			uint32_t stop_index;
			int is_request_stop = 0;
			for ( stop_index = 0u; stop_index < r->stop_token_count; ++stop_index )
				if ( r->stop_tokens[stop_index] == ev->token_id )
				{
					is_request_stop = 1;
					break;
				}
			if ( is_request_stop )
			{
				pthread_mutex_lock(&r->mutex);
				r->done = 1;
				pthread_cond_signal(&r->cond);
				{
					SparkModelBatchRequestHandle cancel_handle =
						(r->submitted && r->handle != 0) ? r->handle : 0;
					pthread_mutex_unlock(&r->mutex);
					if ( cancel_handle != 0 )
					{
						r->inflight = 1;
						pthread_mutex_unlock(&S.queue_mutex);
						(void)SparkModelBatchEngineCancel(S.engine, cancel_handle);
						pthread_mutex_lock(&S.queue_mutex);
						r->inflight = 0;
					}
				}
			}
			else if (r->tokens_json_len + 12u < sizeof(r->tokens_json) &&
				r->output_token_count < r->max_tokens)
			{
				r->tokens_json_len += (uint32_t)snprintf(
					r->tokens_json + r->tokens_json_len,
					sizeof(r->tokens_json) - r->tokens_json_len,
					"%s%u", r->tokens_json_len ? "," : "",
					(unsigned)ev->token_id);
				r->token_ready_ns[r->output_token_count] = ev->monotonic_ns;
				r->output_token_ids[r->output_token_count++] = ev->token_id;
				pthread_mutex_lock(&r->mutex);
				pthread_cond_signal(&r->cond);
				pthread_mutex_unlock(&r->mutex);
			}
		}
		if (ev->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED ||
			ev->kind == SPARK_MODEL_BATCH_EVENT_ERROR)
		{
			pthread_mutex_lock(&r->mutex);
			r->status = ev->status;
			r->engine_completed = ev->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED ? 1u : 0u;
			r->done = 1;
			S.served++;
			pthread_cond_signal(&r->cond);
			pthread_mutex_unlock(&r->mutex);
		}
		pthread_mutex_unlock(&S.queue_mutex);
		return;
	}
	pthread_mutex_unlock(&S.queue_mutex);
}

static void api_orphan_cancel_after_submit(ApiRequest *r)
{
	SparkModelBatchRequestHandle orphan_handle = 0;
	pthread_mutex_lock(&r->mutex);
	if (r->orphaned)
		orphan_handle = r->handle;
	pthread_mutex_unlock(&r->mutex);
	if (orphan_handle != 0)
		(void)SparkModelBatchEngineCancel(S.engine, orphan_handle);
}

static void api_wake_worker(void)
{
	uint8_t value = 1u;
	ssize_t written;
	do
		written = write(S.wake_fds[1],&value,sizeof(value));
	while ( written < 0 && errno == EINTR );
	if ( written != sizeof(value) && errno != EAGAIN && errno != EWOULDBLOCK )
	{
		api_logf("worker wake failed errno=%d",errno);
		_exit(1);
	}
}

static void api_drain_worker_wake(void)
{
	uint8_t values[64];
	ssize_t received;
	do
		received = read(S.wake_fds[0],values,sizeof(values));
	while ( received > 0 || (received < 0 && errno == EINTR) );
}

static void api_cancel_queued(void)
{
	ApiRequest *request;
	SparkModelBatchRequestHandle handle;
	for (;;)
	{
		pthread_mutex_lock(&S.queue_mutex);
		for (request=S.queue_head; request!=0; request=request->next)
			if ( request->cancel_pending != 0u && request->submitted != 0 )
				break;
		if ( request == 0 )
		{
			pthread_mutex_unlock(&S.queue_mutex);
			break;
		}
		handle = request->handle;
		request->cancel_pending = 0u;
		request->inflight = 1;
		pthread_mutex_unlock(&S.queue_mutex);
		(void)SparkModelBatchEngineCancel(S.engine,handle);
		pthread_mutex_lock(&S.queue_mutex);
		request->inflight = 0;
		pthread_mutex_unlock(&S.queue_mutex);
	}
}

static uint32_t api_before(const ApiRequest *a, const ApiRequest *b)
{
	uint64_t deadline_a = a->deadline_ms != 0u ? a->deadline_ms : UINT64_MAX, deadline_b = b->deadline_ms != 0u ? b->deadline_ms : UINT64_MAX;
	if ( a->priority != b->priority )
		return(a->priority > b->priority ? 1u : 0u);
	if ( deadline_a != deadline_b )
		return(deadline_a < deadline_b ? 1u : 0u);
	return(a->id < b->id ? 1u : 0u);
}

static uint32_t api_pick_pending(ApiRequest **pending, uint32_t capacity)
{
	ApiRequest *request,*best;
	uint32_t count = 0u;
	while ( count < capacity )
	{
		best = 0;
		for (request = S.queue_head; request != 0; request = request->next)
			if ( request->done == 0 && request->submitted == 0 && request->inflight == 0 && (best == 0 || api_before(request,best) != 0u) )
				best = request;
		if ( best == 0 )
			break;
		best->inflight = 1;
		pending[count++] = best;
	}
	return(count);
}

static void api_reap_orphans(void)
{
	ApiRequest *victim,*next,*reaped = 0;
	pthread_mutex_lock(&S.queue_mutex);
	for (victim = S.queue_head; victim != 0; victim = next)
	{
		next = victim->next;
		if ( victim->orphaned == 0 || victim->inflight != 0 )
			continue;
		api_queue_unlink(victim);
		victim->next = reaped;
		reaped = victim;
	}
	pthread_mutex_unlock(&S.queue_mutex);
	for (victim = reaped; victim != 0; victim = next)
	{
		next = victim->next;
		api_request_destroy(victim);
	}
}

static uint32_t api_submit(ApiRequest *r)
{
	SparkModelBatchSubmitRequest sub;
	SparkModelBatchRequestHandle h;
	SparkStatus st;
	memset(&sub,0,sizeof(sub));
	sub.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	sub.descriptor_bytes = (uint32_t)sizeof(sub);
	sub.request_id = r->id;
	sub.sequence_id = r->id;
	sub.priority = r->priority;
	sub.output_token_budget = r->max_tokens;
	sub.prompt_token_ids = r->prompt_tokens;
	sub.prompt_token_count = r->prompt_count;
	(void)SparkModelBatchEngineReopenAdmission(S.engine);
	st = SparkModelBatchEngineSubmit(S.engine,&sub,&h);
	pthread_mutex_lock(&S.queue_mutex);
	if ( st == SPARK_STATUS_OK )
	{
		r->submitted = 1;
		r->handle = h;
	}
	else if ( st != SPARK_STATUS_BUSY )
	{
		r->status = (uint32_t)st;
		r->done = 1;
	}
	pthread_mutex_lock(&r->mutex);
	pthread_cond_signal(&r->cond);
	pthread_mutex_unlock(&r->mutex);
	pthread_mutex_unlock(&S.queue_mutex);
	api_orphan_cancel_after_submit(r);
	return st == SPARK_STATUS_BUSY ? 1u : 0u;
}

static uint32_t api_submit_pending(void)
{
	ApiRequest *pending[API_MAX_INFLIGHT];
	uint32_t count,index,busy = 0u;
	pthread_mutex_lock(&S.queue_mutex);
	count = api_pick_pending(pending,API_MAX_INFLIGHT);
	pthread_mutex_unlock(&S.queue_mutex);
	for (index = 0u; index < count; index++)
	{
		if ( pending[index]->done == 0 && busy == 0u )
			busy = api_submit(pending[index]);
		pthread_mutex_lock(&S.queue_mutex);
		pending[index]->inflight = 0;
		pthread_mutex_unlock(&S.queue_mutex);
	}
	return busy;
}

static void api_save_sequence(void)
{
	char seq_path[1024];
	FILE *seq_out;
	uint64_t now_ms = api_now_ms();
	if ( now_ms - S.seq_saved_ms < API_SEQUENCE_SAVE_MS )
		return;
	(void)snprintf(seq_path,sizeof(seq_path),"%s/api_submission.seq",S.runtime_root);
	seq_out = fopen(seq_path,"w");
	if ( seq_out != 0 )
	{
		(void)fprintf(seq_out,"%llu %llu\n",(unsigned long long)SparkModelBatchEngineSessionFingerprint(S.engine),(unsigned long long)SparkModelBatchEnginePeekSubmissionId(S.engine));
		(void)fclose(seq_out);
	}
	S.seq_saved_ms = now_ms;
}

static int api_poll_timeout(uint32_t busy)
{
	uint64_t deadline_ns,deadline_ms,save_ms,now_ms;
	int timeout_ms;
	deadline_ns = SparkModelBatchEngineNextProgressNs(S.engine);
	deadline_ms = deadline_ns / UINT64_C(1000000) + (deadline_ns % UINT64_C(1000000) != 0u);
	save_ms = S.seq_saved_ms + API_SEQUENCE_SAVE_MS;
	if ( deadline_ms == 0u || save_ms < deadline_ms )
		deadline_ms = save_ms;
	now_ms = api_now_ms();
	timeout_ms = deadline_ms <= now_ms ? 0 : (int)(deadline_ms - now_ms);
	return busy != 0u && timeout_ms > API_BUSY_RETRY_MS ? API_BUSY_RETRY_MS : timeout_ms;
}

static void api_poll_engine(uint32_t busy)
{
	SparkModelResidentClientPollDescriptor fds[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT];
	struct pollfd pfds[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT + 1u];
	uint32_t count = 0u,index;
	SparkStatus status;
	status = SparkModelBatchEngineGetPollDescriptors(S.engine,fds,SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT,&count);
	if ( status != SPARK_STATUS_OK )
	{
		api_logf("worker poll descriptors failed status=%u",(unsigned)status);
		_exit(1);
	}
	for (index = 0u; index < count; index++)
	{
		pfds[index].fd = fds[index].fd;
		pfds[index].events = (short)(((fds[index].events & SPARK_MODEL_RESIDENT_CLIENT_POLL_READ) != 0u ? POLLIN : 0) | ((fds[index].events & SPARK_MODEL_RESIDENT_CLIENT_POLL_WRITE) != 0u ? POLLOUT : 0));
		pfds[index].revents = 0;
	}
	pfds[count].fd = S.wake_fds[0];
	pfds[count].events = POLLIN;
	pfds[count].revents = 0;
	if ( poll(pfds,(nfds_t)count + 1u,api_poll_timeout(busy)) < 0 && errno != EINTR )
	{
		api_logf("worker poll failed errno=%d",errno);
		_exit(1);
	}
}

static void *api_worker(void *arg)
{
	uint32_t busy;
	(void)arg;
	while (S.running)
	{
		api_drain_worker_wake();
		api_cancel_queued();
		api_reap_orphans();
		busy = api_submit_pending();
		(void)SparkModelBatchEngineProgress(S.engine,4u);
		api_save_sequence();
		api_poll_engine(busy);
	}
	return 0;
}

static int send_all(int fd, const char *data, size_t len)
{
	size_t off = 0;
	while (off < len)
	{
		ssize_t n = send(fd, data + off, len - off, MSG_NOSIGNAL);
		if (n <= 0)
			return 0;
		off += (size_t)n;
	}
	return 1;
}

static void send_response(int fd, int code, const char *body)
{
	char hdr[192];
	int n = snprintf(hdr, sizeof(hdr),
		"HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
		"Content-Length: %zu\r\nConnection: close\r\n\r\n",
		code, code == 200 ? "OK" : "Error", strlen(body));
	if (n > 0)
		send_all(fd, hdr, (size_t)n);
	send_all(fd, body, strlen(body));
}

static int read_http_request(int fd, char *method, size_t method_sz,
	char *path, size_t path_sz, char **body, char **body_base,
	uint32_t *body_len)
{
	char *buf = (char *)malloc(API_MAX_BODY + 8192u);
	size_t buf_cap = API_MAX_BODY + 8192u;
	size_t total = 0, header_end = 0;
	uint32_t content_length = 0;
	ssize_t n;
	if (buf == 0)
		return 0;
	while (total < buf_cap - 1)
	{
		n = recv(fd, buf + total, buf_cap - 1 - total, 0);
		if (n <= 0)
		{
			free(buf);
			return 0;
		}
		total += (size_t)n;
		buf[total] = '\0';
		{
			char *end = strstr(buf, "\r\n\r\n");
			if (end != 0)
			{
				header_end = (size_t)(end - buf) + 4u;
				break;
			}
		}
	}
	if (header_end == 0)
	{
		free(buf);
		return 0;
	}
	{
		char *sp1 = strchr(buf, ' ');
		char *sp2;
		if (sp1 == 0)
		{
			free(buf);
			return 0;
		}
		sp2 = strchr(sp1 + 1, ' ');
		if (sp2 == 0)
		{
			free(buf);
			return 0;
		}
		{
			size_t ml = (size_t)(sp1 - buf);
			if (ml >= method_sz) ml = method_sz - 1;
			memcpy(method, buf, ml); method[ml] = '\0';
		}
		{
			size_t pl = (size_t)(sp2 - sp1 - 1);
			if (pl >= path_sz) pl = path_sz - 1;
			memcpy(path, sp1 + 1, pl); path[pl] = '\0';
		}
	}
	{
		char *cl = strcasestr(buf, "content-length:");
		if (cl != 0)
			content_length = (uint32_t)strtoul(cl + 15, 0, 10);
	}
	if (content_length > API_MAX_BODY)
	{
		free(buf);
		return 0;
	}
	if (content_length == 0)
	{
		buf[header_end] = '\0';
		*body = buf + header_end;
		*body_base = buf;
		*body_len = 0;
		return 1;
	}
	while (total < header_end + content_length)
	{
		n = recv(fd, buf + total, header_end + content_length - total, 0);
		if (n <= 0)
		{
			free(buf);
			return 0;
		}
		total += (size_t)n;
	}
	*body = buf + header_end;
	*body_base = buf;
	*body_len = content_length;
	return 1;
}

static uint32_t parse_token_array(SparkJsonDocument *doc, int32_t root,
	const char *name, uint32_t **out)
{
	int32_t m = SparkJsonFindObjectMember(doc, root, name);
	int32_t e;
	uint32_t count, i;
	uint32_t *tokens;
	if (m < 0 || !SparkJsonTokenIsType(doc, m, SPARK_JSON_TOKEN_ARRAY))
		return 0;
	count = (uint32_t)SparkJsonGetArrayElementCount(doc, m);
	if (count == 0 || count > API_MAX_PROMPT_TOKENS)
		return 0;
	tokens = malloc((size_t)count * sizeof(uint32_t));
	if (tokens == 0)
		return 0;
	e = SparkJsonGetArrayElementFirst(doc, m);
	for (i = 0; i < count; i++)
	{
		uint32_t v;
		if (e < 0 || SparkJsonGetUInt32(doc, e, &v) != SPARK_STATUS_OK ||
			v > 260000)
		{
			free(tokens);
			return 0;
		}
		tokens[i] = v;
		e = SparkJsonGetArrayElementNext(doc, m, e);
	}
	*out = tokens;
	return count;
}

static int append_json_escaped(char *buf, size_t cap, size_t *len,
	const char *text, uint32_t text_bytes)
{
	uint32_t i;
	for (i = 0; i < text_bytes; i++)
	{
		unsigned char c = (unsigned char)text[i];
		char escape_buf[8];
		const char *escape = 0;
		switch (c)
		{
			case '"': escape = "\\\""; break;
			case '\\': escape = "\\\\"; break;
			case '\n': escape = "\\n"; break;
			case '\r': escape = "\\r"; break;
			case '\t': escape = "\\t"; break;
			case '\b': escape = "\\b"; break;
			case '\f': escape = "\\f"; break;
			default:
				if (c < 0x20)
				{
					(void)snprintf(escape_buf, sizeof(escape_buf),
						"\\u%04x", (unsigned)c);
					escape = escape_buf;
				}
				break;
		}
		if (escape != 0)
		{
			size_t escape_len = strlen(escape);
			if (*len + escape_len + 1u > cap)
				return 0;
			memcpy(buf + *len, escape, escape_len);
			*len += escape_len;
		}
		else
		{
			if (*len + 2u > cap)
				return 0;
			buf[(*len)++] = (char)c;
		}
	}
	buf[*len] = '\0';
	return 1;
}

static void send_tokenizer_unavailable(int fd)
{
	send_response(fd, 400,
		"{\"error\":{\"message\":\"deployment has no tokenizer sidecar: "
		"text prompts require a \\\"tokenizer\\\":{\\\"path\\\":...} entry in "
		"the deployment config (asset shipped beside the pack); use "
		"prompt_token_ids or deploy a tokenizer\","
		"\"type\":\"invalid_request_error\","
		"\"code\":\"tokenizer_unavailable\"}}");
}

typedef struct ApiStream
{
	char *pending;
	char *piece;
	char *escaped;
	char *event;
	uint32_t pending_bytes;
	uint32_t pending_capacity;
	uint32_t piece_capacity;
	uint32_t sent_tokens;
	uint32_t first;
	size_t escaped_capacity;
	size_t event_capacity;
} ApiStream;

static int api_parse_serving_options(const SparkJsonDocument *doc, int32_t root, uint32_t *stream, uint32_t *priority, uint32_t *deadline_ms)
{
	int32_t member;
	bool flag;
	*stream = *priority = *deadline_ms = 0u;
	if ( root < 0 )
		return 1;
	member = SparkJsonFindObjectMember(doc,root,"stream");
	if ( member >= 0 && SparkJsonGetBoolean(doc,member,&flag) != SPARK_STATUS_OK )
		return 0;
	*stream = member >= 0 && flag ? 1u : 0u;
	member = SparkJsonFindObjectMember(doc,root,"priority");
	if ( member >= 0 && SparkJsonGetUInt32(doc,member,priority) != SPARK_STATUS_OK )
		return 0;
	member = SparkJsonFindObjectMember(doc,root,"deadline_ms");
	if ( member >= 0 && (SparkJsonGetUInt32(doc,member,deadline_ms) != SPARK_STATUS_OK || *deadline_ms == 0u) )
		return 0;
	return 1;
}

static uint32_t api_stop_list(const ApiRequest *req, uint32_t *stops)
{
	uint32_t count = EngineStopTokenCount;
	memcpy(stops,EngineStopTokens,(size_t)EngineStopTokenCount * sizeof(uint32_t));
	if ( req->stop_token_count <= API_MAX_STOP_TOKENS )
	{
		memcpy(stops + count,req->stop_tokens,(size_t)req->stop_token_count * sizeof(uint32_t));
		count += req->stop_token_count;
	}
	return count;
}

static void api_usage_json(char *buffer, size_t capacity, const ApiRequest *req)
{
	(void)snprintf(buffer,capacity,",\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}",req->prompt_count,req->output_token_count,req->prompt_count + req->output_token_count);
}

static uint32_t api_client_gone(int fd)
{
	char probe;
	ssize_t received = recv(fd,&probe,1,MSG_PEEK | MSG_DONTWAIT);
	return received == 0 || (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK) ? 1u : 0u;
}

static void api_abandon(ApiRequest *req, uint32_t status, uint32_t deadline_expired)
{
	pthread_mutex_lock(&S.queue_mutex);
	pthread_mutex_lock(&req->mutex);
	req->cancel_pending = 1u;
	req->deadline_expired = deadline_expired;
	req->status = req->status == 0u ? status : req->status;
	req->done = 1;
	pthread_mutex_unlock(&req->mutex);
	pthread_mutex_unlock(&S.queue_mutex);
	api_wake_worker();
}

static uint32_t api_ready(const ApiRequest *req, uint32_t sent, uint32_t mode)
{
	return req->done != 0 || (mode == API_WAIT_TOKENS && req->output_token_count != sent) || (mode == API_WAIT_ADMISSION && req->submitted != 0) ? 1u : 0u;
}

static void api_wait_slice(ApiRequest *req, uint32_t sent, uint32_t mode, uint64_t now_ms)
{
	struct timespec until;
	uint64_t slice = API_WAIT_SLICE_MS;
	if ( req->deadline_ms != 0u && req->deadline_ms - now_ms < slice )
		slice = req->deadline_ms - now_ms;
	clock_gettime(CLOCK_REALTIME,&until);
	until.tv_sec += (time_t)(slice / 1000u);
	until.tv_nsec += (long)(slice % 1000u) * 1000000L;
	if ( until.tv_nsec >= 1000000000L )
	{
		until.tv_sec += 1;
		until.tv_nsec -= 1000000000L;
	}
	pthread_mutex_lock(&req->mutex);
	if ( api_ready(req,sent,mode) == 0u )
		pthread_cond_timedwait(&req->cond,&req->mutex,&until);
	pthread_mutex_unlock(&req->mutex);
}

static void api_await(int fd, ApiRequest *req, uint32_t sent, uint32_t mode)
{
	uint64_t now_ms;
	for (;;)
	{
		if ( api_ready(req,sent,mode) != 0u || S.running == 0 )
			return;
		if ( api_client_gone(fd) != 0u )
		{
			api_abandon(req,SPARK_STATUS_IO_ERROR,0u);
			return;
		}
		now_ms = api_now_ms();
		if ( req->deadline_ms != 0u && now_ms >= req->deadline_ms )
		{
			api_abandon(req,SPARK_STATUS_BUSY,1u);
			return;
		}
		api_wait_slice(req,sent,mode,now_ms);
	}
}

static uint32_t api_utf8_complete(const char *text, uint32_t length)
{
	uint32_t start = length,need;
	uint8_t lead;
	while ( start > 0u && length - start < API_UTF8_MAX_SEQUENCE && ((uint8_t)text[start - 1u] & 0xc0u) == 0x80u )
		start--;
	if ( start == 0u )
		return length;
	lead = (uint8_t)text[start - 1u];
	need = lead < 0x80u ? 1u : (lead & 0xe0u) == 0xc0u ? 2u : (lead & 0xf0u) == 0xe0u ? 3u : (lead & 0xf8u) == 0xf0u ? 4u : 1u;
	return start - 1u + need > length ? start - 1u : length;
}

static int api_stream_open(int fd, ApiStream *stream)
{
	static const char headers[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
	memset(stream,0,sizeof(*stream));
	stream->first = 1u;
	stream->piece_capacity = (HaveSidecar ? Sidecar.maximum_token_text_bytes : 0u) + 1u;
	stream->pending_capacity = API_STREAM_BATCH_TOKENS * stream->piece_capacity + API_UTF8_MAX_SEQUENCE;
	stream->escaped_capacity = (size_t)stream->pending_capacity * 6u + 8u;
	stream->event_capacity = stream->escaped_capacity + (size_t)API_STREAM_BATCH_TOKENS * 12u + 512u;
	stream->pending = malloc(stream->pending_capacity);
	stream->piece = malloc(stream->piece_capacity);
	stream->escaped = malloc(stream->escaped_capacity);
	stream->event = malloc(stream->event_capacity);
	if ( stream->pending == 0 || stream->piece == 0 || stream->escaped == 0 || stream->event == 0 )
		return 0;
	stream->escaped[0] = '\0';
	return send_all(fd,headers,sizeof(headers) - 1u);
}

static uint32_t api_stream_take(ApiRequest *req, ApiStream *stream, uint32_t *tokens)
{
	uint32_t count;
	pthread_mutex_lock(&S.queue_mutex);
	count = req->output_token_count - stream->sent_tokens;
	count = count > API_STREAM_BATCH_TOKENS ? API_STREAM_BATCH_TOKENS : count;
	memcpy(tokens,req->output_token_ids + stream->sent_tokens,(size_t)count * sizeof(uint32_t));
	stream->sent_tokens += count;
	pthread_mutex_unlock(&S.queue_mutex);
	return count;
}

static int api_stream_text(ApiStream *stream, const uint32_t *tokens, uint32_t count, const uint32_t *stops, uint32_t stop_count, uint32_t flush)
{
	uint32_t index,bytes,ready;
	size_t escaped_length = 0u;
	stream->escaped[0] = '\0';
	if ( !HaveSidecar )
		return 1;
	for (index = 0u; index < count; index++)
	{
		if ( SparkTokenizerSidecarDecodeText(&Sidecar,tokens + index,1u,stops,stop_count,0u,stream->piece,stream->piece_capacity,&bytes) != SPARK_STATUS_OK || stream->pending_bytes + bytes > stream->pending_capacity )
			return 0;
		memcpy(stream->pending + stream->pending_bytes,stream->piece,bytes);
		stream->pending_bytes += bytes;
	}
	ready = flush != 0u ? stream->pending_bytes : api_utf8_complete(stream->pending,stream->pending_bytes);
	if ( !append_json_escaped(stream->escaped,stream->escaped_capacity,&escaped_length,stream->pending,ready) )
		return 0;
	memmove(stream->pending,stream->pending + ready,stream->pending_bytes - ready);
	stream->pending_bytes -= ready;
	return 1;
}

static int api_stream_event(int fd, ApiStream *stream, int chat_format, const uint32_t *tokens, uint32_t count, const ApiRequest *finished)
{
	char usage[160],reason[24];
	size_t length;
	uint32_t index;
	int written;
	usage[0] = '\0';
	(void)snprintf(reason,sizeof(reason),finished != 0 ? "\"%s\"" : "null",finished != 0 ? api_finish_reason(finished) : "");
	if ( finished != 0 )
		api_usage_json(usage,sizeof(usage),finished);
	written = chat_format
		? snprintf(stream->event,stream->event_capacity,"data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{%s\"content\":\"%s\"},\"finish_reason\":%s}]%s,\"tokens\":[",stream->first != 0u ? "\"role\":\"assistant\"," : "",stream->escaped,reason,usage)
		: snprintf(stream->event,stream->event_capacity,"data: {\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"%s\",\"finish_reason\":%s}]%s,\"tokens\":[",stream->escaped,reason,usage);
	if ( written < 0 || (size_t)written >= stream->event_capacity )
		return 0;
	length = (size_t)written;
	for (index = 0u; index < count && length + 16u < stream->event_capacity; index++)
		length += (size_t)snprintf(stream->event + length,stream->event_capacity - length,"%s%u",index != 0u ? "," : "",tokens[index]);
	if ( index != count || length + 8u > stream->event_capacity )
		return 0;
	memcpy(stream->event + length,API_STREAM_EVENT_END,sizeof(API_STREAM_EVENT_END) - 1u);
	stream->first = 0u;
	return send_all(fd,stream->event,length + sizeof(API_STREAM_EVENT_END) - 1u);
}

static int api_failure(const ApiRequest *req, char *body, size_t capacity)
{
	int code = 500;
	if ( req->deadline_expired != 0u )
		code = 504, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"deadline exceeded\",\"type\":\"timeout\",\"code\":\"deadline_exceeded\"}}");
	else if ( req->submitted == 0 && req->status == (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED )
		code = 400, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"prompt plus max_tokens exceeds the deployment's context or KV capacity\",\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\"}}");
	else
		(void)snprintf(body,capacity,"{\"error\":{\"message\":\"model status %u\",\"type\":\"model_error\",\"code\":%u}}",req->status,req->status);
	return code;
}

static void api_stream_close(int fd, const ApiRequest *req, ApiStream *stream, int chat_format, const uint32_t *stops, uint32_t stop_count, int alive)
{
	static const char done[] = "data: [DONE]\n\n";
	char error[224],event[240];
	if ( alive && (req->deadline_expired != 0u || req->status != 0u) )
	{
		(void)api_failure(req,error,sizeof(error));
		(void)snprintf(event,sizeof(event),"data: %s\n\n",error);
		alive = send_all(fd,event,strlen(event));
	}
	else if ( alive )
		alive = api_stream_text(stream,0,0u,stops,stop_count,1u) && api_stream_event(fd,stream,chat_format,0,0u,req);
	if ( alive )
		(void)send_all(fd,done,sizeof(done) - 1u);
	free(stream->pending);
	free(stream->piece);
	free(stream->escaped);
	free(stream->event);
}

static void api_send_completion(int fd, const ApiRequest *req, int chat_format)
{
	uint32_t stops[API_MAX_STOP_TOKENS + SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT],stop_count,text_capacity = 1u,text_bytes = 0u;
	char *text = 0,*escaped = 0,*response = 0,usage[160],err[160];
	size_t escaped_capacity,escaped_length = 0u,response_capacity;
	SparkStatus status = SPARK_STATUS_OK;
	stop_count = api_stop_list(req,stops);
	if (HaveSidecar)
		text_capacity = req->output_token_count * Sidecar.maximum_token_text_bytes + 1u;
	escaped_capacity = (size_t)text_capacity * 6u + 8u;
	response_capacity = (size_t)req->tokens_json_len + escaped_capacity + 384u;
	text = malloc(text_capacity);
	escaped = malloc(escaped_capacity);
	response = malloc(response_capacity);
	if (text == 0 || escaped == 0 || response == 0)
		status = SPARK_STATUS_INTERNAL_ERROR;
	else if (HaveSidecar)
		status = SparkTokenizerSidecarDecodeText(&Sidecar,req->output_token_ids,req->output_token_count,stops,stop_count,0u,text,text_capacity,&text_bytes);
	if (status == SPARK_STATUS_OK && !append_json_escaped(escaped,escaped_capacity,&escaped_length,text,text_bytes))
		status = SPARK_STATUS_INTERNAL_ERROR;
	api_usage_json(usage,sizeof(usage),req);
	if (status == SPARK_STATUS_OK && !HaveSidecar)
		(void)snprintf(response,response_capacity,"{\"object\":\"text_completion\",\"tokens\":[%s],\"finish_reason\":\"%s\"%s,\"status\":0}",req->tokens_json,api_finish_reason(req),usage);
	else if (status == SPARK_STATUS_OK && chat_format)
		(void)snprintf(response,response_capacity,"{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"%s\"}]%s,\"tokens\":[%s],\"status\":0}",escaped,api_finish_reason(req),usage,req->tokens_json);
	else if (status == SPARK_STATUS_OK)
		(void)snprintf(response,response_capacity,"{\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"%s\",\"finish_reason\":\"%s\"}]%s,\"tokens\":[%s],\"status\":0}",escaped,api_finish_reason(req),usage,req->tokens_json);
	if (status == SPARK_STATUS_OK)
	{
		send_response(fd,200,response);
		api_logf("request_done fd=%d id=%llu status=%u output_tokens=%u ms=%llu",fd,(unsigned long long)req->id,(unsigned)req->status,req->output_token_count,(unsigned long long)(api_now_ms() - req->started_ms));
	}
	else
	{
		(void)snprintf(err,sizeof(err),"{\"error\":{\"message\":\"tokenizer sidecar failed to decode the completion (status %u)\",\"type\":\"model_error\",\"code\":%u}}",(unsigned)status,(unsigned)status);
		send_response(fd,500,err);
	}
	free(text);
	free(escaped);
	free(response);
}

static void api_release_request(ApiRequest *req)
{
	api_log_request_measurements(req);
	pthread_mutex_lock(&S.queue_mutex);
	if (req->inflight || req->cancel_pending)
	{
		req->orphaned = 1;
		pthread_mutex_unlock(&S.queue_mutex);
		api_wake_worker();
		return;
	}
	api_queue_unlink(req);
	pthread_mutex_unlock(&S.queue_mutex);
	api_request_destroy(req);
}

static void api_stream(int fd, ApiRequest *req, int chat_format)
{
	ApiStream stream;
	uint32_t tokens[API_STREAM_BATCH_TOKENS],stops[API_MAX_STOP_TOKENS + SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT],stop_count,count;
	char error[224];
	int alive;
	api_await(fd,req,0u,API_WAIT_ADMISSION);
	if ( req->done == 0 && req->submitted == 0 )
		api_abandon(req,SPARK_STATUS_INTERNAL_ERROR,0u);
	if ( req->deadline_expired != 0u || req->status != 0u )
	{
		send_response(fd,api_failure(req,error,sizeof(error)),error);
		return;
	}
	stop_count = api_stop_list(req,stops);
	alive = api_stream_open(fd,&stream);
	while ( alive )
	{
		api_await(fd,req,stream.sent_tokens,API_WAIT_TOKENS);
		count = api_stream_take(req,&stream,tokens);
		if ( count == 0u && (req->done != 0 || S.running == 0) )
			break;
		if ( count != 0u )
			alive = api_stream_text(&stream,tokens,count,stops,stop_count,0u) && api_stream_event(fd,&stream,chat_format,tokens,count,0);
	}
	if ( req->done == 0 )
		api_abandon(req,alive != 0 ? SPARK_STATUS_INTERNAL_ERROR : SPARK_STATUS_IO_ERROR,0u);
	api_stream_close(fd,req,&stream,chat_format,stops,stop_count,alive);
}

static void handle_completion(int fd, char *body, uint32_t body_len,
	int chat_format)
{
	SparkJsonDocument doc;
	int32_t root, mt;
	uint32_t *request_stops = 0;
	uint32_t request_stop_count = 0;
	memset(&doc,0,sizeof(doc));
	uint32_t *prompt = 0, prompt_len = 0, max_tokens = 32;
	uint32_t stream_mode = 0u, priority = 0u, deadline_ms = 0u;
	char *prompt_text = 0;
	uint32_t prompt_text_bytes = 0;
	ApiRequest *req;
	if (SparkJsonParseText(body, body_len, &doc) != SPARK_STATUS_OK)
	{
		send_response(fd, 400, "{\"error\":\"invalid json\"}");
		return;
	}
	root = SparkJsonGetRootToken(&doc);
	if (root >= 0)
		prompt_len = parse_token_array(&doc, root, "prompt_token_ids", &prompt);
	if (root >= 0)
	{
		int32_t pm = SparkJsonFindObjectMember(&doc, root, "prompt");
		if (pm >= 0)
		{
			if (!SparkJsonTokenIsType(&doc, pm, SPARK_JSON_TOKEN_STRING) ||
				SparkJsonCopyString(&doc, pm, &prompt_text) != SPARK_STATUS_OK)
			{
				SparkJsonDocumentDestroy(&doc);
				free(prompt);
				send_response(fd, 400,
					"{\"error\":{\"message\":\"prompt must be a string\","
					"\"type\":\"invalid_request_error\",\"code\":\"invalid_prompt\"}}");
				return;
			}
			prompt_text_bytes = (uint32_t)strlen(prompt_text);
		}
	}
	if (prompt_text == 0 && prompt == 0 && root >= 0)
	{
		int32_t messages = SparkJsonFindObjectMember(&doc, root, "messages");
		size_t chat_cap = 4096u;
		size_t chat_len = 0u;
		uint32_t message_index;
		uint32_t message_count = 0u;
		char *chat_text;
		if (messages >= 0 &&
			SparkJsonTokenIsType(&doc, messages, SPARK_JSON_TOKEN_ARRAY))
			message_count = SparkJsonGetArrayElementCount(&doc, messages);
		chat_text = message_count > 0u ? malloc(chat_cap) : 0;
		for (message_index = 0u;
			chat_text != 0 && message_index < message_count;
			++message_index)
		{
			int32_t entry = SparkJsonGetArrayElement(&doc, messages, message_index);
			int32_t content;
			char *piece = 0;
			size_t piece_bytes;
			size_t need;
			if (entry < 0 ||
				!SparkJsonTokenIsType(&doc, entry, SPARK_JSON_TOKEN_OBJECT))
				continue;
			content = SparkJsonFindObjectMember(&doc, entry, "content");
			if (content < 0 ||
				!SparkJsonTokenIsType(&doc, content, SPARK_JSON_TOKEN_STRING) ||
				SparkJsonCopyString(&doc, content, &piece) != SPARK_STATUS_OK)
				continue;
			piece_bytes = strlen(piece);
			need = chat_len + piece_bytes + 2u;
			if (need > chat_cap)
			{
				char *grown;
				while (need > chat_cap)
					chat_cap *= 2u;
				grown = realloc(chat_text, chat_cap);
				if (grown == 0)
				{
					free(piece);
					free(chat_text);
					chat_text = 0;
					break;
				}
				chat_text = grown;
			}
			memcpy(chat_text + chat_len, piece, piece_bytes);
			chat_len += piece_bytes;
			chat_text[chat_len++] = '\n';
			free(piece);
		}
		if (chat_text != 0)
		{
			chat_text[chat_len] = '\0';
			prompt_text = chat_text;
			prompt_text_bytes = (uint32_t)chat_len;
		}
	}
	if (root >= 0)
	{
		uint32_t *stops = 0;
		uint32_t stop_len = parse_token_array(&doc, root, "stop_token_ids", &stops);
		if (stop_len > 0 && stop_len <= API_MAX_STOP_TOKENS)
			request_stops = stops, request_stop_count = stop_len;
		else
			free(stops);
	}
	mt = SparkJsonFindObjectMember(&doc, root, "max_tokens");
	if (mt >= 0)
	{
		uint32_t v;
		if (SparkJsonGetUInt32(&doc, mt, &v) == SPARK_STATUS_OK && v > 0)
			max_tokens = v > API_MAX_OUTPUT_TOKENS ? API_MAX_OUTPUT_TOKENS : v;
	}
	if (!api_parse_serving_options(&doc, root, &stream_mode, &priority, &deadline_ms))
	{
		SparkJsonDocumentDestroy(&doc);
		free(prompt);
		free(prompt_text);
		free(request_stops);
		send_response(fd, 400,
			"{\"error\":{\"message\":\"stream must be a boolean; priority and deadline_ms "
			"must be unsigned integers, deadline_ms nonzero\","
			"\"type\":\"invalid_request_error\",\"code\":\"invalid_option\"}}");
		return;
	}
	SparkJsonDocumentDestroy(&doc);
	if (prompt_text != 0 && prompt != 0)
	{
		free(prompt_text);
		free(prompt);
		send_response(fd, 400,
			"{\"error\":{\"message\":\"prompt and prompt_token_ids are "
			"mutually exclusive\",\"type\":\"invalid_request_error\","
			"\"code\":\"ambiguous_prompt\"}}");
		return;
	}
	if (prompt_text != 0 && !HaveSidecar)
	{
		free(prompt_text);
		send_tokenizer_unavailable(fd);
		return;
	}
	if (prompt_text != 0)
	{
		SparkTokenizerWorkspace workspace;
		SparkTokenizerEncoding encoding;
		SparkStatus encode_status;
		if (prompt_text_bytes == 0u)
		{
			free(prompt_text);
			send_response(fd, 400,
				"{\"error\":{\"message\":\"prompt is empty\","
				"\"type\":\"invalid_request_error\",\"code\":\"invalid_prompt\"}}");
			return;
		}
		SparkTokenizerWorkspaceReset(&workspace);
		prompt = malloc((size_t)prompt_text_bytes * sizeof(uint32_t) + sizeof(uint32_t));
		if (prompt == 0 ||
			SparkTokenizerWorkspaceInitialize(&workspace, prompt_text_bytes + 1u) != SPARK_STATUS_OK)
		{
			free(prompt_text);
			free(prompt);
			send_response(fd, 500, "{\"error\":\"oom\"}");
			return;
		}
		SparkTokenizerEncodingReset(&encoding);
		encoding.token_capacity = prompt_text_bytes + 1u;
		encoding.token_ids = prompt;
		encode_status = SparkTokenizerSidecarEncodeText(&Sidecar, prompt_text,
			prompt_text_bytes, 0u, &workspace, &encoding);
		SparkTokenizerWorkspaceDestroy(&workspace);
		free(prompt_text);
		if (encode_status != SPARK_STATUS_OK)
		{
			free(prompt);
			{
				char err[160];
				(void)snprintf(err, sizeof(err),
					"{\"error\":{\"message\":\"tokenizer sidecar failed to "
					"encode the prompt (status %u)\","
					"\"type\":\"invalid_request_error\",\"code\":\"encode_failed\"}}",
					(unsigned)encode_status);
				send_response(fd, 400, err);
			}
			return;
		}
		prompt_len = encoding.token_count;
	}
	if (prompt_len == 0)
	{
		free(prompt);
		send_response(fd, 400, "{\"error\":\"prompt_token_ids required\"}");
		return;
	}
	if ((uint64_t)prompt_len + max_tokens > API_MAX_PROMPT_TOKENS + API_MAX_OUTPUT_TOKENS)
	{
		free(prompt);
		send_response(fd, 400, "{\"error\":\"prompt + max_tokens exceeds context limit\"}");
		return;
	}
	req = calloc(1, sizeof(*req));
	if (req == 0)
	{
		free(prompt);
		send_response(fd, 500, "{\"error\":\"oom\"}");
		return;
	}
	req->output_token_ids = malloc((size_t)max_tokens * sizeof(uint32_t));
	if (req->output_token_ids == 0)
	{
		free(prompt);
		free(req);
		send_response(fd, 500, "{\"error\":\"oom\"}");
		return;
	}
	pthread_mutex_init(&req->mutex, 0);
	pthread_cond_init(&req->cond, 0);
	req->started_ms = api_now_ms();
	req->stop_tokens = request_stops;
	req->stop_token_count = request_stop_count;
	req->stream = stream_mode;
	req->priority = priority;
	req->deadline_ms = deadline_ms != 0u ? req->started_ms + deadline_ms : 0u;
	pthread_mutex_lock(&S.queue_mutex);
	req->id = ++S.next_id + 100000;
	req->prompt_tokens = prompt;
	req->prompt_count = prompt_len;
	req->max_tokens = max_tokens;
	if (S.queue_tail != 0)
		S.queue_tail->next = req;
	else
		S.queue_head = req;
	S.queue_tail = req;
	pthread_mutex_unlock(&S.queue_mutex);
	api_wake_worker();
	if (stream_mode)
	{
		api_stream(fd, req, chat_format);
		api_release_request(req);
		return;
	}
	api_await(fd, req, 0u, API_WAIT_DONE);
	if (req->done == 0)
		api_abandon(req, SPARK_STATUS_INTERNAL_ERROR, 0u);
	if (req->deadline_expired == 0u && req->status == 0)
		api_send_completion(fd, req, chat_format);
	else
	{
		char err[224];
		int code = api_failure(req, err, sizeof(err));
		send_response(fd, code, err);
	}
	api_release_request(req);
}

static void *api_connection(void *arg)
{
	int fd = (int)(intptr_t)arg;
	char method[8], path[128];
	char *body = 0, *body_base = 0;
	uint32_t body_len = 0;
	int on = 1;
	(void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
	{
		int rr = read_http_request(fd, method, sizeof(method), path, sizeof(path),
			&body, &body_base, &body_len);
		if (!rr)
		{
			send_response(fd, 400, "{\"error\":\"bad request\"}");
			close(fd);
			return 0;
		}
	}
	if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0)
	{
		char b[128];
		(void)snprintf(b, sizeof(b),
			"{\"status\":\"ok\",\"served\":%llu,\"tokenizer\":%s}",
			(unsigned long long)S.served, HaveSidecar ? "true" : "false");
		send_response(fd, 200, b);
	}
	else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0)
	{
		const char *model_id = getenv("SPARK_MODEL_ID");
		char b[256];
		if (model_id == 0 || model_id[0] == '\0')
			model_id = API_DEFAULT_MODEL_ID;
		(void)snprintf(b, sizeof(b),
			"{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
			"\"object\":\"model\",\"owned_by\":\"sparkpipe\","
			"\"served\":%llu}]}",
			model_id, (unsigned long long)S.served);
		send_response(fd, 200, b);
	}
	else if (strcmp(method, "POST") == 0 &&
		(strcmp(path, "/v1/completions") == 0 ||
		 strcmp(path, "/v1/chat/completions") == 0))
		handle_completion(fd, body, body_len,
			strcmp(path, "/v1/chat/completions") == 0);
	else
		send_response(fd, 404, "{\"error\":\"not found\"}");
	free(body_base);
	close(fd);
	return 0;
}

int main(int argc, char **argv)
{
	const char *dep_path = 0, *root = 0, *port_s = "8080";
	SparkModelResidentDeployment dep;
	SparkModelBatchEngineConfiguration cfg;
	pthread_t worker;
	int srv, i;
	for (i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--deployment") && i + 1 < argc)
			dep_path = argv[++i];
		else if (!strcmp(argv[i], "--runtime-root") && i + 1 < argc)
			root = argv[++i];
		else if (!strcmp(argv[i], "--port") && i + 1 < argc)
			port_s = argv[++i];
	}
	if (dep_path == 0 || root == 0)
	{
		fprintf(stderr, "usage: %s --deployment PATH --runtime-root PATH [--port N]\n", argv[0]);
		return 1;
	}
	(void)snprintf(ApiBootTag, sizeof(ApiBootTag), "%d", (int)getpid());
	{
		const char *log_path = getenv("SPARK_MODEL_API_LOG");
		if (log_path != 0 && log_path[0] != '\0')
		{
			int log_fd = open(log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
			if (log_fd >= 0)
			{
				(void)dup2(log_fd, 2);
				(void)close(log_fd);
			}
		}
	}
	api_logf("api_start pid=%d deployment=%s runtime_root=%s", (int)getpid(), dep_path, root);
	S.runtime_root = root;
	S.seq_saved_ms = 0u;
	SparkModelResidentDeploymentReset(&dep);
	if (SparkModelResidentDeploymentLoad(dep_path, &dep) != SPARK_STATUS_OK)
	{
		fprintf(stderr, "model_api: deployment load failed\n");
		return 1;
	}
	memset(&cfg, 0, sizeof(cfg));
	cfg.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	cfg.descriptor_bytes = (uint32_t)sizeof(cfg);
	cfg.deployment = &dep;
	cfg.runtime_root = root;
	cfg.request_capacity = 64;
	cfg.max_context_tokens = API_MAX_PROMPT_TOKENS + API_MAX_OUTPUT_TOKENS;
	cfg.max_prefill_rows_per_submission = dep.runtime_limits.max_input_row_count;
	{
		const char *rows_env = getenv("SPARK_MODEL_API_MAX_PREFILL_ROWS");
		if ( rows_env != 0 && rows_env[0] != '\0' )
		{
			uint32_t clamp = (uint32_t)strtoul(rows_env,0,10);
			if ( clamp != 0u && clamp < cfg.max_prefill_rows_per_submission )
				cfg.max_prefill_rows_per_submission = clamp;
		}
	}
	cfg.connect_timeout_ms = 30000;
	cfg.maximum_messages_per_rank_per_progress = 8;
	cfg.inflight_budget_ns = SPARK_MODEL_BATCH_ENGINE_DEFAULT_INFLIGHT_BUDGET_NS;
	{
		const char *budget_env = getenv("SPARK_BATCH_INFLIGHT_BUDGET_NS");
		if ( budget_env != 0 && budget_env[0] != '\0' )
		{
			cfg.inflight_budget_ns = strtoull(budget_env,0,10);
			if ( cfg.inflight_budget_ns < SPARK_MODEL_BATCH_ENGINE_MIN_INFLIGHT_BUDGET_NS )
			{
				fprintf(stderr,"model_api: SPARK_BATCH_INFLIGHT_BUDGET_NS=%s is below the %llu ns minimum\n",budget_env,(unsigned long long)SPARK_MODEL_BATCH_ENGINE_MIN_INFLIGHT_BUDGET_NS);
				return 1;
			}
		}
	}
	cfg.event_function = api_event;
	cfg.event_context = 0;
	EngineStopTokenCount = dep.eos_token_count;
	if ( EngineStopTokenCount == 0u || EngineStopTokenCount > SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT )
	{
		fprintf(stderr,"model_api: required model EOS metadata missing or invalid\n");
		return 1;
	}
	memcpy(EngineStopTokens,dep.eos_token_ids,EngineStopTokenCount * sizeof(uint32_t));
	if (dep.tokenizer_asset_path != 0)
	{
		char asset_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
		SparkTokenizerSidecarConfiguration sidecar_configuration;
		if (SparkResolveRuntimePath(root, dep.tokenizer_asset_path,
				asset_path, (uint32_t)sizeof(asset_path)) != SPARK_STATUS_OK)
		{
			fprintf(stderr, "model_api: tokenizer asset path %s is not a "
				"valid runtime-root-relative path\n", dep.tokenizer_asset_path);
			return 1;
		}
		memset(&sidecar_configuration, 0, sizeof(sidecar_configuration));
		sidecar_configuration.abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
		sidecar_configuration.descriptor_bytes =
			SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES;
		sidecar_configuration.asset_path = asset_path;
		sidecar_configuration.format = SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO;
		if (SparkTokenizerSidecarLoad(&Sidecar, &sidecar_configuration) != SPARK_STATUS_OK)
		{
			fprintf(stderr, "model_api: tokenizer sidecar load FAILED for %s "
				"(deployment promised text serving); refusing to start\n",
				asset_path);
			return 1;
		}
		{
			char actual_sha256[SPARK_SHA256_HEX_BYTES];
			uint32_t mismatch = 0u;
			if ((uint64_t)Sidecar.tokenizer.maximum_token_id + 1u != dep.tokenizer_vocabulary_size)
				mismatch |= 1u;
			if (SparkSha256File(asset_path, actual_sha256) != SPARK_STATUS_OK ||
				strcmp(actual_sha256, dep.tokenizer_asset_sha256) != 0)
				mismatch |= 2u;
			if (mismatch != 0u)
			{
				fprintf(stderr, "model_api: tokenizer asset %s does not match the "
					"deployment (%s%s): declared vocab=%u sha256=%s, actual vocab=%llu; "
					"refusing to start\n",
					asset_path,
					(mismatch & 1u) != 0u ? "vocabulary_size " : "",
					(mismatch & 2u) != 0u ? "sha256" : "",
					dep.tokenizer_vocabulary_size, dep.tokenizer_asset_sha256,
					(unsigned long long)Sidecar.tokenizer.maximum_token_id + 1ull);
				return 1;
			}
		}
		HaveSidecar = 1;
		fprintf(stderr, "model_api: tokenizer sidecar ready format=%u "
			"vocab=%llu asset=%s\n", Sidecar.format,
			(unsigned long long)Sidecar.tokenizer.maximum_token_id + 1ull, asset_path);
	}
	else
		fprintf(stderr, "model_api: no tokenizer in deployment; text prompts "
			"will be rejected (prompt_token_ids accepted)\n");
	{
		uint64_t connect_started_ms = api_now_ms();
		uint64_t connect_deadline_ms = 120000u;
		const char *deadline_env = getenv("SPARK_MODEL_API_CONNECT_DEADLINE_MS");
		unsigned connect_attempt = 0;
		SparkStatus connect_status;
		if (deadline_env != 0 && deadline_env[0] != '\0')
			connect_deadline_ms = (uint64_t)strtoull(deadline_env, 0, 10);
		for (;;)
		{
			connect_attempt++;
			api_logf("engine_connect attempt=%u elapsed_ms=%llu", connect_attempt,
				(unsigned long long)(api_now_ms() - connect_started_ms));
			connect_status = SparkModelBatchEngineConnect(&cfg, &S.engine);
			if (connect_status == SPARK_STATUS_OK)
				break;
			api_logf("engine_connect_failed attempt=%u status=%u", connect_attempt,
				(unsigned)connect_status);
			if (api_now_ms() - connect_started_ms >= connect_deadline_ms)
			{
				api_logf("api_exit reason=engine_connect_deadline attempts=%u", connect_attempt);
				return 1;
			}
			sleep(1);
		}
		{
			unsigned ready_attempt = 0;
			for (;;)
			{
				if ( SparkModelBatchEngineAllRanksReady(S.engine) != 0u )
					break;
				if ( (ready_attempt % 10u) == 0u )
					api_logf("engine_ranks_waiting — not every rank is connected+helloed on one generation yet");
				if ( api_now_ms() - connect_started_ms >= connect_deadline_ms )
				{
					api_logf("api_exit reason=engine_ranks_not_ready_deadline");
					return 1;
				}
				ready_attempt++;
				sleep(1);
			}
		}
		api_logf("engine_connected attempts=%u elapsed_ms=%llu all_ranks_ready=1", connect_attempt,
			(unsigned long long)(api_now_ms() - connect_started_ms));
	}
	{
		char seq_path[1024];
		uint64_t session = SparkModelBatchEngineSessionFingerprint(S.engine);
		uint64_t saved_session = 0u;
		uint64_t saved_id = 0u;
		uint64_t seeded = 1000000u;
		(void)snprintf(seq_path,sizeof(seq_path),"%s/api_submission.seq",root);
		{
			FILE *seq_in = fopen(seq_path,"r");
			if ( seq_in != 0 )
			{
				unsigned long long fs = 0ull,fi = 0ull;
				if ( fscanf(seq_in,"%llu %llu",&fs,&fi) == 2 ||
				     fscanf(seq_in,"%llu",&fi) == 1 )
				{
					saved_session = (uint64_t)fs;
					saved_id = (uint64_t)fi;
				}
				(void)fclose(seq_in);
			}
		}
		if ( saved_session == session && saved_id >= 1000000u &&
		     saved_id < SparkTpChainIdCapacity(4u,6946816u) - 10001u )
			seeded = saved_id + 10001u;
		SparkModelBatchEngineSeedSubmissionId(S.engine,seeded);
		{
			FILE *seq_out = fopen(seq_path,"w");
			if ( seq_out != 0 )
			{
				(void)fprintf(seq_out,"%llu %llu\n",
				    (unsigned long long)session,
				    (unsigned long long)SparkModelBatchEnginePeekSubmissionId(S.engine));
				(void)fclose(seq_out);
			}
		}
		api_logf("submission_id_seeded next=%llu session=%llu (%s)",
		    (unsigned long long)SparkModelBatchEnginePeekSubmissionId(S.engine),
		    (unsigned long long)session,
		    seeded != 1000000u ? "continued within engine session, crash window skipped" :
		        "rebased — new engine session");
	}
	signal(SIGPIPE, SIG_IGN);
	signal(SIGTERM, api_term_signal);
	signal(SIGINT, api_term_signal);
	pthread_mutex_init(&S.queue_mutex, 0);
	if ( pipe(S.wake_fds) != 0 ||
	     fcntl(S.wake_fds[0],F_SETFL,O_NONBLOCK) < 0 ||
	     fcntl(S.wake_fds[1],F_SETFL,O_NONBLOCK) < 0 ||
	     fcntl(S.wake_fds[0],F_SETFD,FD_CLOEXEC) < 0 ||
	     fcntl(S.wake_fds[1],F_SETFD,FD_CLOEXEC) < 0 )
	{
		api_logf("worker wake pipe failed errno=%d",errno);
		return 1;
	}
	S.running = 1;
	if ( pthread_create(&worker,0,api_worker,0) != 0 )
	{
		api_logf("worker create failed");
		return 1;
	}
	{
		struct sockaddr_in addr;
		int on = 1;
		srv = socket(AF_INET, SOCK_STREAM, 0);
		(void)setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
		memset(&addr, 0, sizeof(addr));
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = htonl(INADDR_ANY);
		addr.sin_port = htons((uint16_t)strtoul(port_s, 0, 10));
		if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
			listen(srv, 128) != 0)
		{
			fprintf(stderr, "model_api: listen %s: %s\n", port_s, strerror(errno));
			return 1;
		}
	}
	api_logf("model_api ready port=%s boot_pid=%d sessions=%s (single session, worker-driven)",
		port_s, (int)getpid(), "queued-on-engine");
	for (;;)
	{
		int cfd = accept(srv, 0, 0);
		pthread_t t;
		if (cfd < 0)
			continue;
		ApiSessionsAccepted++;
		api_logf("session_accepted n=%u fd=%d queued_behind=%llu served=%llu",
			ApiSessionsAccepted, cfd,
			(unsigned long long)(S.next_id - S.served),
			(unsigned long long)S.served);
		if (pthread_create(&t, 0, api_connection, (void *)(intptr_t)cfd) == 0)
			pthread_detach(t);
		else
			close(cfd);
	}
	S.running = 0;
	(void)SparkModelBatchEngineDestroy(S.engine);
	return 0;
}
