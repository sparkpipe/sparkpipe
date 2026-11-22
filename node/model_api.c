#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdarg.h>
#include <time.h>
#include <fcntl.h>
#include <sys/random.h>
#include <sys/resource.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_chat_template.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_tp_chain_ordinal.h"
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_quant_arm.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_tokenizer_sidecar.h"
#include "sparkpipe/spark_sampling.h"

#define API_MAX_BODY		(16u * 1024u * 1024u)
#define API_MAX_HEADER		(64u * 1024u)
#define API_MAX_PROMPT_TOKENS	(1048576u)
#define API_MAX_STOP_TOKENS 16
#define API_MAX_OUTPUT_TOKENS	(131072u)
#define API_TOKEN_TEXT_BYTES	12u

#define API_MAX_INFLIGHT 16u
#define API_MAX_EMBEDDING_INPUTS 256u

#define API_WAIT_SLICE_MS 10u
#define API_STREAM_BATCH_TOKENS 64u
#define API_UTF8_MAX_SEQUENCE 4u
#define API_STREAM_EVENT_END "}\n\n"
#define API_WAIT_DONE 0u
#define API_WAIT_TOKENS 1u
#define API_WAIT_ADMISSION 2u
#define API_BUSY_RETRY_MS 5
#define API_SEQUENCE_SAVE_MS 60000u
#define API_HEALTH_REFRESH_MS 1000u
#define API_STOP_DRAIN_MS 10000u
#define API_CACHE_STATE_WAIT_MS 5000u
#define API_CACHE_STATE_SLICE_MS 50u
#define API_CACHE_STATE_GRACE_MS 1000u
#define API_PROBE_IDLE 0u
#define API_PROBE_ASKED 1u
#define API_PROBE_WAITING 2u
#define API_PROBE_DONE 3u
#define API_PROBE_ABANDONED 4u
#define API_CACHE_JSON_BYTES 320u
#define API_USAGE_JSON_BYTES (160u + API_CACHE_JSON_BYTES)

typedef struct ApiOptions
{
	uint32_t stream;
	uint32_t priority;
	uint32_t deadline_ms;
	uint32_t seeded;
	float temperature;
	uint64_t seed;
	float top_p;
	uint32_t top_k;
	uint32_t logprobs;
	uint32_t drafts_disabled;
	uint32_t cache_point_count;
	uint32_t cache_points[SPARK_MODEL_BATCH_ENGINE_MAX_CACHE_POINTS];
} ApiOptions;

typedef struct ApiRequest
{
	uint64_t id;
	uint64_t started_ms;
	uint32_t *prompt_tokens;
	uint32_t prompt_count;
	uint32_t max_tokens;
	char *tokens_json;
	size_t tokens_json_capacity;
	volatile uint32_t tokens_json_len;
	uint32_t *output_token_ids;
	uint64_t *token_ready_ns;
	uint64_t accepted_ns;
	uint64_t first_dispatch_ns;
	uint32_t stale_prefix_recompute_count;
	uint32_t cached_prompt_token_count;
	uint32_t matched_prompt_token_count;
	uint64_t restore_wait_ns;
	uint64_t background_wait_ns;
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
	uint32_t served_counted;
	uint32_t stop_token_count;
	uint32_t priority;
	uint32_t stream;
	uint32_t deadline_expired;
	uint32_t engine_latched;
	uint64_t deadline_ms;
	float temperature;
	uint64_t seed;
	float top_p;
	uint32_t top_k;
	uint32_t logprob_count;
	uint32_t drafts_disabled;
	uint32_t cache_point_count;
	uint32_t cache_points[SPARK_MODEL_BATCH_ENGINE_MAX_CACHE_POINTS];
	SparkSamplingLogprob *logprobs;
	uint32_t output_row;
	uint32_t row_value_count;
	uint16_t *row_values;
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
	const char *prefix_index_path;
	uint64_t seq_saved_ms;
	uint32_t context_limit;
	volatile int ready;
	volatile uint32_t bringup_status;
	uint32_t health_rank_count;
	uint32_t health_connected_rank_count;
	uint32_t health_missing_rank;
	uint32_t health_degraded_rank;
	uint32_t health_write_alert_rank;
	uint64_t health_kv_capacity_tokens;
	uint64_t health_kv_capacity_max_tokens;
	uint32_t health_failed_status;
	uint32_t health_live_requests;
	uint64_t last_activity_ms;
	volatile uint32_t bringup_attempts;
	const char *volatile bringup_phase;
	uint64_t bringup_started_ms;
} ApiState;

static ApiState S;

static SparkTokenizerSidecar Sidecar;
static int HaveSidecar;
static uint32_t ApiOutputVocabulary;
static SparkQuantArm QuantArm;
static int HaveQuantArm;
static const SparkModelResidentChatTemplate *ChatTemplate;
static uint32_t TextPromptPrefix[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_TEXT_PROMPT_PREFIX_COUNT];
static uint32_t TextPromptPrefixCount;
static uint32_t TextPromptSuffix[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_TEXT_PROMPT_SUFFIX_COUNT];
static uint32_t TextPromptSuffixCount;
static uint32_t ChatStopTokens[SPARK_MODEL_RESIDENT_CHAT_TEMPLATE_MAX_STOP_MARKERS];
static uint32_t ChatStopTokenCount;
static char ApiBootTag[32];
static char ApiBootId[48];

typedef struct ApiCacheProbe
{
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	uint32_t stage;
	uint32_t status;
	uint32_t *tokens;
	uint32_t token_count;
	uint32_t by_identity;
	uint64_t deadline_ms;
	SparkModelServingCacheIdentity identity;
	SparkModelBatchCacheState state;
} ApiCacheProbe;

static ApiCacheProbe Probe = {.mutex = PTHREAD_MUTEX_INITIALIZER,.cond = PTHREAD_COND_INITIALIZER};
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

#define API_ACCEPT_BACKOFF_MS 50u

static uint64_t api_raise_open_files(void)
{
	struct rlimit limit;
	if ( getrlimit(RLIMIT_NOFILE,&limit) != 0 )
		return(0u);
	if ( limit.rlim_cur < limit.rlim_max )
	{
		struct rlimit raised = limit;
		raised.rlim_cur = limit.rlim_max;
		if ( setrlimit(RLIMIT_NOFILE,&raised) != 0 && limit.rlim_max == RLIM_INFINITY )
		{
			raised.rlim_cur = (rlim_t)1048576u;
			while ( raised.rlim_cur > limit.rlim_cur && setrlimit(RLIMIT_NOFILE,&raised) != 0 )
				raised.rlim_cur /= 2u;
		}
	}
	if ( getrlimit(RLIMIT_NOFILE,&limit) != 0 )
		return(0u);
	return((uint64_t)limit.rlim_cur);
}

static int api_accept(int srv)
{
	int cfd;
	static uint32_t starved;
	for (;;)
	{
		cfd = accept(srv,0,0);
		if ( cfd >= 0 )
		{
			if ( starved != 0u )
				api_logf("accept recovered after %u retries",starved);
			starved = 0u;
			return(cfd);
		}
		if ( errno == EINTR || errno == ECONNABORTED )
			continue;
		if ( starved++ == 0u )
			api_logf("accept failed: %s; the API retries every %u ms and keeps serving the connections it holds",strerror(errno),API_ACCEPT_BACKOFF_MS);
		{
			struct timespec pause = {0,(long)API_ACCEPT_BACKOFF_MS * 1000000L};
			nanosleep(&pause,0);
		}
	}
}

static volatile sig_atomic_t api_stop_requested;
static volatile sig_atomic_t api_worker_started;

static void api_term_signal(int signal_number)
{
	const char line[] = "model_api api_exit reason=signal\n";
	uint8_t value = 1u;
	ssize_t written;
	(void)signal_number;
	if ( api_worker_started == 0 || api_stop_requested != 0 )
	{
		written = write(2, line, sizeof(line) - 1u);
		(void)written;
		_exit(0);
	}
	api_stop_requested = 1;
	written = write(S.wake_fds[1], &value, sizeof(value));
	(void)written;
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
	fprintf(stderr,"{\"event\":\"request_measurements\",\"boot_pid\":%d,\"request_id\":%llu,\"status\":%u,\"engine_completed\":%u,\"accepted_ns\":%llu,\"first_dispatch_ns\":%llu,\"stale_prefix_recomputes\":%u,\"prompt_tokens\":%u,\"cached_prompt_tokens\":%u,\"matched_prompt_tokens\":%u,\"restore_wait_ns\":%llu,\"background_wait_ns\":%llu,\"prompt_sha256\":\"%s\",\"adapter_id\":\"%s\",\"model_id\":\"%s\",\"model_revision\":\"%s\",\"driver_program\":\"%s\",\"driver_artifact_sha256\":\"%s\",\"session_fingerprint\":%llu,\"priority\":%u,\"deadline_expired\":%u,\"stream\":%u,\"temperature\":%.9g,\"seed\":%llu,\"top_p\":%.9g,\"top_k\":%u,\"logprobs\":%u,\"finish_reason\":\"%s\",\"tokens\":[",(int)getpid(),(unsigned long long)request->id,request->status,request->engine_completed,(unsigned long long)request->accepted_ns,(unsigned long long)request->first_dispatch_ns,request->stale_prefix_recompute_count,request->prompt_count,request->cached_prompt_token_count,request->matched_prompt_token_count,(unsigned long long)request->restore_wait_ns,(unsigned long long)request->background_wait_ns,prompt_sha256,api_identity(adapter != 0 ? adapter->adapter_id : 0),api_identity(adapter != 0 ? adapter->model_id : 0),api_identity(adapter != 0 ? adapter->model_revision : 0),api_identity(adapter != 0 ? adapter->driver_program_name : 0),api_identity(adapter != 0 ? adapter->artifact_sha256 : 0),(unsigned long long)(S.engine != 0 ? SparkModelBatchEngineSessionFingerprint(S.engine) : 0u),request->priority,request->deadline_expired,request->stream,(double)request->temperature,(unsigned long long)request->seed,(double)request->top_p,request->top_k,request->logprob_count,request->status == 0u && request->deadline_expired == 0u ? api_finish_reason(request) : "error");
	for (index=0u; index<request->output_token_count; index++)
		fprintf(stderr,"%s[%u,%llu]",index == 0u ? "" : ",",request->output_token_ids[index],(unsigned long long)request->token_ready_ns[index]);
	fputs("]",stderr);
	if ( HaveQuantArm )
		fprintf(stderr,",\"arm_id\":\"%s\",\"arm_digest\":\"%s\",\"arm_kv\":\"%s\",\"pack_set_sha256\":\"%s\"",QuantArm.arm_id,QuantArm.arm_digest,QuantArm.kv_text,QuantArm.pack_set_sha256);
	fputs("}\n",stderr);
	funlockfile(stderr);
}

static void api_request_destroy(ApiRequest *req)
{
	pthread_mutex_destroy(&req->mutex);
	pthread_cond_destroy(&req->cond);
	free(req->stop_tokens);
	free(req->prompt_tokens);
	free(req->output_token_ids);
	free(req->token_ready_ns);
	free(req->tokens_json);
	free(req->logprobs);
	free(req->row_values);
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
	__atomic_store_n(&S.last_activity_ms,api_now_ms(),__ATOMIC_RELEASE);
	pthread_mutex_lock(&S.queue_mutex);
	for (r = S.queue_head; r != 0; r = r->next)
	{
		if (r->id != ev->request_id)
			continue;
		r->cached_prompt_token_count = ev->cached_prompt_token_count;
		r->matched_prompt_token_count = ev->matched_prompt_token_count;
		r->restore_wait_ns = ev->restore_wait_ns;
		r->background_wait_ns = ev->background_wait_ns;
		r->first_dispatch_ns = ev->first_dispatch_ns;
		r->stale_prefix_recompute_count = ev->stale_prefix_recompute_count;
		if ( ev->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED )
			r->accepted_ns = ev->monotonic_ns;
		if ( ev->kind == SPARK_MODEL_BATCH_EVENT_TOKEN && r->output_row != 0u && r->row_values == 0 && ev->row_values != 0 && ev->row_value_count != 0u &&
			(r->row_values = malloc((size_t)ev->row_value_count * sizeof(uint16_t))) != 0 )
		{
			memcpy(r->row_values,ev->row_values,(size_t)ev->row_value_count * sizeof(uint16_t));
			r->row_value_count = ev->row_value_count;
		}
		if (ev->kind == SPARK_MODEL_BATCH_EVENT_TOKEN)
		{
			const int is_stop = (ev->flags & SPARK_MODEL_BATCH_EVENT_FLAG_STOP_TOKEN) != 0u;
			if (is_stop == 0 && r->tokens_json_len + API_TOKEN_TEXT_BYTES < r->tokens_json_capacity &&
				r->output_token_count < r->max_tokens)
			{
				r->tokens_json_len += (uint32_t)snprintf(
					r->tokens_json + r->tokens_json_len,
					r->tokens_json_capacity - r->tokens_json_len,
					"%s%u", r->tokens_json_len ? "," : "",
					(unsigned)ev->token_id);
				r->token_ready_ns[r->output_token_count] = ev->monotonic_ns;
				if ( r->logprob_count != 0u )
					memcpy(r->logprobs + (size_t)r->output_token_count * r->logprob_count,ev->logprobs,(size_t)r->logprob_count * sizeof(r->logprobs[0]));
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
			if ( r->served_counted == 0u )
			{
				r->served_counted = 1u;
				S.served++;
			}
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
	sub.deadline_ns = r->deadline_ms != 0u ? r->deadline_ms * UINT64_C(1000000) : 0u;
	sub.temperature = r->temperature;
	sub.seed = r->seed;
	sub.top_p = r->top_p;
	sub.top_k = r->top_k;
	sub.logprobs = r->logprob_count;
	sub.drafts_disabled = r->drafts_disabled;
	sub.output_row = r->output_row;
	sub.cache_point_count = r->cache_point_count;
	memcpy(sub.cache_points,r->cache_points,sizeof(sub.cache_points));
	sub.stop_token_count = r->stop_token_count;
	sub.stop_token_ids = r->stop_tokens;
	sub.output_token_budget = r->max_tokens;
	sub.prompt_token_ids = r->prompt_tokens;
	sub.prompt_token_count = r->prompt_count;
	st = SparkModelBatchEngineReopenAdmission(S.engine);
	r->engine_latched = st != SPARK_STATUS_OK ? 1u : 0u;
	if ( st == SPARK_STATUS_OK )
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
	if ( deadline_ms > now_ms + API_HEALTH_REFRESH_MS )
		deadline_ms = now_ms + API_HEALTH_REFRESH_MS;
	timeout_ms = deadline_ms <= now_ms ? 0 : (int)(deadline_ms - now_ms);
	return busy != 0u && timeout_ms > API_BUSY_RETRY_MS ? API_BUSY_RETRY_MS : timeout_ms;
}

static void api_poll_engine(uint32_t busy)
{
	SparkModelResidentClientPollDescriptor fds[SPARK_MODEL_BATCH_ENGINE_MAX_POLL_DESCRIPTORS];
	struct pollfd pfds[SPARK_MODEL_BATCH_ENGINE_MAX_POLL_DESCRIPTORS + 1u];
	uint32_t count = 0u,index;
	SparkStatus status;
	status = SparkModelBatchEngineGetPollDescriptors(S.engine,fds,SPARK_MODEL_BATCH_ENGINE_MAX_POLL_DESCRIPTORS,&count);
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

static void api_log_engine_measurements(void)
{
	static uint64_t logged_terminals,logged_rejections;
	SparkModelBatchEngineView view;
	uint64_t terminals;
	if ( SparkModelBatchEngineGetView(S.engine,&view) != SPARK_STATUS_OK )
		return;
	terminals = view.completed_request_count + view.cancelled_request_count;
	if ( terminals == logged_terminals && view.rejected_lane_count == logged_rejections )
		return;
	logged_terminals = terminals;
	logged_rejections = view.rejected_lane_count;
	fprintf(stderr,"{\"event\":\"engine_measurements\",\"boot_pid\":%d,\"completed\":%llu,\"cancelled\":%llu,\"first_tokens\":%llu,\"queue_ns_total\":%llu,\"prefill_ns_total\":%llu,\"ttft_ns_total\":%llu,\"ttft_ns_maximum\":%llu,\"prefix_hits\":%llu,\"prefix_misses\":%llu,\"prefix_hit_tokens\":%llu,\"stale_prefix_recomputes\":%llu,\"stale_prefix_isolations\":%llu,\"rejected_lanes\":%llu,\"rejected_waves_busy\":%llu,\"rejected_waves_not_found\":%llu,\"rejected_waves_validation_failed\":%llu,\"rejected_waves_io_error\":%llu,\"prefix_index_saves\":%llu,\"prefix_index_save_failures\":%llu,\"prefix_index_write_ns\":%llu,\"prefix_index_write_ns_max\":%llu,\"prefix_index_export_ns_max\":%llu,\"prefix_index_loaded\":%llu,\"prefix_index_refused\":%llu,\"prefix_index_reimported\":%llu,\"verify_lanes\":%llu,\"draft_tokens\":%llu,\"accepted_draft_tokens\":%llu,\"tree_frames\":%llu,\"tree_branch_frames\":%llu,\"tree_branch_accepted\":%llu}\n",(int)getpid(),(unsigned long long)view.completed_request_count,(unsigned long long)view.cancelled_request_count,(unsigned long long)view.first_token_count,(unsigned long long)view.queue_ns_total,(unsigned long long)view.prefill_ns_total,(unsigned long long)view.ttft_ns_total,(unsigned long long)view.ttft_ns_maximum,(unsigned long long)view.prefix_hit_count,(unsigned long long)view.prefix_miss_count,(unsigned long long)view.prefix_hit_token_count,(unsigned long long)view.stale_prefix_recompute_count,(unsigned long long)view.stale_prefix_isolation_count,(unsigned long long)view.rejected_lane_count,(unsigned long long)view.rejected_submission_count_by_status[SPARK_STATUS_BUSY],(unsigned long long)view.rejected_submission_count_by_status[SPARK_STATUS_NOT_FOUND],(unsigned long long)view.rejected_submission_count_by_status[SPARK_STATUS_VALIDATION_FAILED],(unsigned long long)view.rejected_submission_count_by_status[SPARK_STATUS_IO_ERROR],(unsigned long long)view.prefix_index_save_count,(unsigned long long)view.prefix_index_save_failure_count,(unsigned long long)view.prefix_index_write_ns_total,(unsigned long long)view.prefix_index_write_ns_maximum,(unsigned long long)view.prefix_index_export_ns_maximum,(unsigned long long)view.prefix_index_loaded_record_count,(unsigned long long)view.prefix_index_refused_count,(unsigned long long)view.prefix_index_reimported_record_count,(unsigned long long)view.speculative_verify_lane_count,(unsigned long long)view.speculative_draft_token_count,(unsigned long long)view.speculative_accepted_token_count,(unsigned long long)view.speculative_tree_frame_count,(unsigned long long)view.speculative_tree_branch_frame_count,(unsigned long long)view.speculative_tree_branch_accepted_count);
}

static uint64_t api_per_page(uint64_t total,uint64_t pages)
{
	return(pages != 0u ? total / pages : 0u);
}

static void api_log_status_reports(void)
{
	static SparkModelResidentStatusReport logged[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT];
	SparkModelResidentStatusReport report;
	const SparkModelDriverKvStoreCounters *kv;
	uint32_t rank;
	for (rank=0u; rank<SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT; rank++)
	{
		if ( SparkModelBatchEngineGetRankStatus(S.engine,rank,&report) != SPARK_STATUS_OK )
			break;
		if ( report.generation == 0u || (memcmp(&report.adapter_snapshot.kv_store,&logged[rank].adapter_snapshot.kv_store,sizeof(report.adapter_snapshot.kv_store)) == 0 &&
			report.residentd_pid == logged[rank].residentd_pid && report.client_generation == logged[rank].client_generation) )
			continue;
		logged[rank] = report;
		kv = &report.adapter_snapshot.kv_store;
		fprintf(stderr,"{\"event\":\"kv_store_report\",\"boot_pid\":%d,\"rank\":%u,\"stage\":%u,\"residentd_pid\":%u,\"client_generation\":%llu,\"status\":%u,\"generation\":%llu,\"attached\":%u,\"save_count\":%llu,\"save_pages\":%llu,\"save_ns\":%llu,\"save_ns_per_page\":%llu,\"save_failures\":%llu,\"save_deferred\":%llu,\"restore_count\":%llu,\"restore_pages\":%llu,\"restore_ns\":%llu,\"restore_ns_per_page\":%llu,\"restore_misses\":%llu,\"restore_corrupt\":%llu,\"restore_read_errors\":%llu,\"restore_failures\":%llu,\"store_used_bytes\":%llu,\"store_maximum_bytes\":%llu,\"store_files\":%llu,\"store_foreign_layout_files\":%llu,\"store_checksum_failures\":%llu,\"store_removed_temporaries\":%llu,\"store_evictions\":%llu,\"store_write_failures\":%llu,\"store_queue_full\":%llu,\"store_queued\":%llu,\"store_failed_status\":%u,\"pool_resident_pages\":%llu,\"pool_physical_pages\":%llu,\"pool_logical_pages\":%llu,\"pool_retained_pages\":%llu,\"pool_evicted_entries\":%llu,\"pool_resident_evictions\":%llu,\"spill_write_bytes\":%llu,\"spill_read_bytes\":%llu,\"spill_digest_mismatches\":%llu,\"spill_read_errors\":%llu,\"pool_device_bytes\":%llu,\"pool_generation\":%llu,\"pool_reattached\":%llu,\"pool_adopted_pages\":%llu,\"write_bytes\":%llu,\"write_bytes_this_hour\":%llu,\"write_bytes_previous_hour\":%llu,\"write_alerts\":%llu,\"write_alerting\":%llu}\n",
			(int)getpid(),rank,report.stage_index,report.residentd_pid,(unsigned long long)report.client_generation,report.status,(unsigned long long)report.generation,kv->attached,
			(unsigned long long)kv->save_count,(unsigned long long)kv->save_page_count,(unsigned long long)kv->save_ns,(unsigned long long)api_per_page(kv->save_ns,kv->save_page_count),
			(unsigned long long)kv->save_failure_count,(unsigned long long)kv->save_deferred_count,(unsigned long long)kv->restore_count,(unsigned long long)kv->restore_page_count,
			(unsigned long long)kv->restore_ns,(unsigned long long)api_per_page(kv->restore_ns,kv->restore_page_count),(unsigned long long)kv->restore_miss_count,(unsigned long long)kv->restore_corrupt_count,
			(unsigned long long)kv->restore_read_error_count,(unsigned long long)kv->restore_failure_count,(unsigned long long)kv->store_used_bytes,(unsigned long long)kv->store_maximum_bytes,
			(unsigned long long)kv->store_file_count,(unsigned long long)kv->store_foreign_layout_file_count,(unsigned long long)kv->store_checksum_failure_count,(unsigned long long)kv->store_removed_temporary_count,
			(unsigned long long)kv->store_eviction_count,(unsigned long long)kv->store_write_failure_count,(unsigned long long)kv->store_queue_full_count,(unsigned long long)kv->store_queued_count,kv->store_failed_status,
			(unsigned long long)kv->pool_resident_pages,(unsigned long long)kv->pool_physical_pages,(unsigned long long)kv->pool_logical_pages,(unsigned long long)kv->pool_retained_pages,
			(unsigned long long)kv->pool_evicted_entries,(unsigned long long)kv->pool_resident_evictions,(unsigned long long)kv->spill_write_bytes,(unsigned long long)kv->spill_read_bytes,
			(unsigned long long)kv->spill_digest_mismatches,(unsigned long long)kv->spill_read_errors,(unsigned long long)kv->pool_device_bytes,
			(unsigned long long)kv->pool_generation,(unsigned long long)kv->pool_reattached,(unsigned long long)kv->pool_adopted_pages,
			(unsigned long long)kv->write_bytes,(unsigned long long)kv->write_bytes_this_hour,(unsigned long long)kv->write_bytes_previous_hour,
			(unsigned long long)kv->write_alerts,(unsigned long long)kv->write_alerting);
	}
}

static uint32_t api_degraded_rank(void)
{
	SparkModelResidentStatusReport report;
	uint32_t rank;
	for (rank=0u; rank<SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT; rank++)
	{
		if ( SparkModelBatchEngineGetRankStatus(S.engine,rank,&report) != SPARK_STATUS_OK )
			break;
		if ( report.generation != 0u && (report.adapter_snapshot.degraded_flags & SPARK_MODEL_DRIVER_DEGRADED_EAGER_PATH) != 0u )
			return(rank);
	}
	return(UINT32_MAX);
}

static uint32_t api_write_alert_rank(void)
{
	SparkModelResidentStatusReport report;
	uint32_t rank;
	for (rank=0u; rank<SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT; rank++)
	{
		if ( SparkModelBatchEngineGetRankStatus(S.engine,rank,&report) != SPARK_STATUS_OK )
			break;
		if ( report.generation != 0u && report.adapter_snapshot.kv_store.write_alerting != 0u )
			return(rank);
	}
	return(UINT32_MAX);
}

static uint64_t api_kv_capacity_pages(void)
{
	SparkModelResidentStatusReport report;
	uint64_t pages = UINT64_MAX;
	uint32_t rank;
	for (rank=0u; rank<SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT; rank++)
	{
		if ( SparkModelBatchEngineGetRankStatus(S.engine,rank,&report) != SPARK_STATUS_OK )
			break;
		if ( report.generation != 0u && report.adapter_snapshot.kv_store.pool_physical_pages < pages )
			pages = report.adapter_snapshot.kv_store.pool_physical_pages;
	}
	return(pages == UINT64_MAX ? 0u : pages);
}

static void api_publish_health(void)
{
	SparkModelBatchEngineView view;
	if ( S.engine == 0 || SparkModelBatchEngineGetView(S.engine,&view) != SPARK_STATUS_OK )
		return;
	__atomic_store_n(&S.health_kv_capacity_tokens,api_kv_capacity_pages() * view.cache_block_token_count,__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_kv_capacity_max_tokens,(uint64_t)view.kv_physical_page_capacity * view.cache_block_token_count,__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_rank_count,view.pipeline.rank_count,__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_connected_rank_count,view.pipeline.connected_rank_count,__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_missing_rank,view.pipeline.first_disconnected_rank,__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_failed_status,view.failed_status,__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_live_requests,view.live_request_count,__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_degraded_rank,api_degraded_rank(),__ATOMIC_RELEASE);
	__atomic_store_n(&S.health_write_alert_rank,api_write_alert_rank(),__ATOMIC_RELEASE);
	if ( view.context_limit != 0u && view.context_limit != __atomic_load_n(&S.context_limit,__ATOMIC_ACQUIRE) )
	{
		__atomic_store_n(&S.context_limit,view.context_limit,__ATOMIC_RELEASE);
		api_logf("api_context_limit max_context_tokens=%u source=engine", view.context_limit);
	}
}

static void api_stop(void)
{
	const uint64_t deadline_ms = api_now_ms() + API_STOP_DRAIN_MS;
	SparkStatus status = SparkModelBatchEngineBeginShutdown(S.engine);
	api_logf("api_stop reason=signal: admission closed, %s", status == SPARK_STATUS_OK ? "no live requests" : "cancelling the live requests");
	for (;;)
	{
		status = SparkModelBatchEngineBeginShutdown(S.engine);
		if ( status == SPARK_STATUS_OK )
			status = SparkModelBatchEngineDestroy(S.engine);
		if ( status == SPARK_STATUS_OK )
		{
			api_logf("api_exit reason=signal drained=1");
			_exit(0);
		}
		if ( api_now_ms() >= deadline_ms )
		{
			api_logf("api_exit reason=signal drained=0 drain_timeout_ms=%u status=%s: live work did not finish, so the prefix index keeps its last save",
				API_STOP_DRAIN_MS, SparkStatusToString(status));
			_exit(0);
		}
		api_drain_worker_wake();
		(void)SparkModelBatchEngineProgress(S.engine,4u);
		api_poll_engine(1u);
	}
}

static void api_cache_probe_step(void)
{
	SparkStatus status;
	pthread_mutex_lock(&Probe.mutex);
	if ( Probe.stage == API_PROBE_ABANDONED )
	{
		SparkModelBatchEngineCacheStateEnd(S.engine);
		Probe.stage = API_PROBE_IDLE;
	}
	if ( Probe.stage == API_PROBE_ASKED )
	{
		status = SparkModelBatchEngineCacheStateBegin(S.engine,Probe.by_identity != 0u ? 0 : Probe.tokens,Probe.token_count,Probe.by_identity != 0u ? &Probe.identity : 0,&Probe.state);
		Probe.status = (uint32_t)status;
		Probe.stage = status == SPARK_STATUS_PENDING ? API_PROBE_WAITING : API_PROBE_DONE;
	}
	if ( Probe.stage == API_PROBE_WAITING )
	{
		status = SparkModelBatchEngineCacheStatePoll(S.engine,&Probe.state);
		if ( status != SPARK_STATUS_PENDING || api_now_ms() >= Probe.deadline_ms )
		{
			if ( status == SPARK_STATUS_PENDING )
				SparkModelBatchEngineCacheStateEnd(S.engine);
			Probe.status = (uint32_t)status;
			Probe.stage = API_PROBE_DONE;
		}
	}
	if ( Probe.stage == API_PROBE_DONE )
		pthread_cond_broadcast(&Probe.cond);
	pthread_mutex_unlock(&Probe.mutex);
}

static void *api_worker(void *arg)
{
	uint32_t busy;
	(void)arg;
	api_worker_started = 1;
	while (S.running)
	{
		if ( api_stop_requested != 0 )
			api_stop();
		api_drain_worker_wake();
		api_cancel_queued();
		api_reap_orphans();
		busy = api_submit_pending();
		(void)SparkModelBatchEngineProgress(S.engine,4u);
		api_cache_probe_step();
		api_publish_health();
		api_log_engine_measurements();
		api_log_status_reports();
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

static uint32_t api_context_limit(const SparkModelResidentDeployment *deployment)
{
	uint32_t limit;
	limit = API_MAX_PROMPT_TOKENS + API_MAX_OUTPUT_TOKENS;
	if ( deployment->max_sequence_positions != 0u && deployment->max_sequence_positions < limit )
		limit = deployment->max_sequence_positions;
	return(limit);
}

static int api_fit_context(uint32_t prompt_len, uint32_t *max_tokens)
{
	uint32_t limit = __atomic_load_n(&S.context_limit,__ATOMIC_ACQUIRE);
	if ( prompt_len >= limit )
		return(0);
	if ( *max_tokens > limit - prompt_len )
		*max_tokens = limit - prompt_len;
	return(1);
}

static void send_response(int fd, int code, const char *body)
{
	char hdr[224];
	int n = snprintf(hdr, sizeof(hdr),
		"HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
		"Content-Length: %zu\r\n%sConnection: close\r\n\r\n",
		code, code == 200 ? "OK" : code == 503 ? "Service Unavailable" : "Error", strlen(body),
		code == 503 ? "Retry-After: 5\r\n" : "");
	if (n > 0)
		send_all(fd, hdr, (size_t)n);
	send_all(fd, body, strlen(body));
}

static int api_engines_ready(void)
{
	return __atomic_load_n(&S.ready, __ATOMIC_ACQUIRE);
}

static void api_bringup_json(char *buffer, size_t bytes)
{
	(void)snprintf(buffer, bytes,
		"{\"status\":\"starting\",\"phase\":\"%s\",\"attempts\":%u,\"last_status\":\"%s\",\"elapsed_ms\":%llu,\"tokenizer\":%s}",
		S.bringup_phase != 0 ? S.bringup_phase : "starting", S.bringup_attempts,
		S.bringup_status != 0u ? SparkStatusToString((SparkStatus)S.bringup_status) : "none",
		(unsigned long long)(api_now_ms() - S.bringup_started_ms), HaveSidecar ? "true" : "false");
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
		if (total >= API_MAX_HEADER)
		{
			free(buf);
			return 0;
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
		char saved = buf[header_end];
		char *cl;
		buf[header_end] = '\0';
		cl = strcasestr(buf, "\r\ncontent-length:");
		if (cl != 0)
		{
			const char *digits = cl + 17;
			char *end = 0;
			unsigned long long value;
			while (*digits == ' ' || *digits == '\t')
				digits++;
			if (*digits < '0' || *digits > '9')
			{
				free(buf);
				return 0;
			}
			errno = 0;
			value = strtoull(digits, &end, 10);
			while (end != 0 && (*end == ' ' || *end == '\t'))
				end++;
			if (errno != 0 || end == 0 || end[0] != '\r' || end[1] != '\n' || value > API_MAX_BODY)
			{
				free(buf);
				return 0;
			}
			content_length = (uint32_t)value;
		}
		buf[header_end] = saved;
	}
	if (content_length > API_MAX_BODY || header_end + (size_t)content_length > buf_cap - 1u)
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
	buf[header_end + content_length] = '\0';
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
			v >= ApiOutputVocabulary)
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
	uint32_t cached_tokens;
	uint32_t matched_tokens;
	uint64_t restore_wait_ns;
	uint64_t background_wait_ns;
	uint32_t first;
	size_t escaped_capacity;
	size_t event_capacity;
} ApiStream;

static int api_parse_sampling_options(const SparkJsonDocument *doc, int32_t root, ApiOptions *options)
{
	int32_t member;
	uint32_t top_alternatives = 0u;
	bool speculation = true;
	member = SparkJsonFindObjectMember(doc,root,"temperature");
	if ( member >= 0 && (SparkJsonGetFloat(doc,member,&options->temperature) != SPARK_STATUS_OK || SparkSamplingTemperatureValid(options->temperature) == 0u) )
		return 0;
	member = SparkJsonFindObjectMember(doc,root,"seed");
	options->seeded = member >= 0 ? 1u : 0u;
	if ( member >= 0 && SparkJsonGetUInt64(doc,member,&options->seed) != SPARK_STATUS_OK )
		return 0;
	member = SparkJsonFindObjectMember(doc,root,"top_p");
	if ( member >= 0 && (SparkJsonGetFloat(doc,member,&options->top_p) != SPARK_STATUS_OK || SparkSamplingTopPValid(options->top_p) == 0u) )
		return 0;
	member = SparkJsonFindObjectMember(doc,root,"top_k");
	if ( member >= 0 && (SparkJsonGetUInt32(doc,member,&options->top_k) != SPARK_STATUS_OK || options->top_k > SPARK_SAMPLING_MAX_TOP_K) )
		return 0;
	member = SparkJsonFindObjectMember(doc,root,"logprobs");
	if ( member >= 0 && (SparkJsonGetUInt32(doc,member,&top_alternatives) != SPARK_STATUS_OK || top_alternatives > SPARK_SAMPLING_MAX_TOP_LOGPROBS) )
		return 0;
	options->logprobs = member >= 0 ? top_alternatives + 1u : 0u;
	member = SparkJsonFindObjectMember(doc,root,"speculation");
	if ( member >= 0 && SparkJsonGetBoolean(doc,member,&speculation) != SPARK_STATUS_OK )
		return 0;
	options->drafts_disabled = member >= 0 && speculation == false ? 1u : 0u;
	member = SparkJsonFindObjectMember(doc,root,"cache_points");
	if ( member >= 0 )
	{
		int32_t element;
		if ( !SparkJsonTokenIsType(doc,member,SPARK_JSON_TOKEN_ARRAY) ||
			(uint32_t)SparkJsonGetArrayElementCount(doc,member) > SPARK_MODEL_BATCH_ENGINE_MAX_CACHE_POINTS )
			return 0;
		element = SparkJsonGetArrayElementFirst(doc,member);
		for (options->cache_point_count=0u; element >= 0; element=SparkJsonGetArrayElementNext(doc,member,element))
		{
			uint32_t point;
			if ( SparkJsonGetUInt32(doc,element,&point) != SPARK_STATUS_OK || point == 0u ||
				(options->cache_point_count != 0u && point <= options->cache_points[options->cache_point_count - 1u]) )
				return 0;
			options->cache_points[options->cache_point_count++] = point;
		}
	}
	if ( options->temperature == 0.0f )
	{
		options->top_p = 1.0f;
		options->top_k = 0u;
	}
	return 1;
}

static int api_parse_serving_options(const SparkJsonDocument *doc, int32_t root, ApiOptions *options)
{
	int32_t member;
	bool flag;
	memset(options,0,sizeof(*options));
	options->top_p = 1.0f;
	options->priority = SPARK_MODEL_BATCH_PRIORITY_INTERACTIVE;
	if ( root < 0 )
		return 1;
	member = SparkJsonFindObjectMember(doc,root,"stream");
	if ( member >= 0 && SparkJsonGetBoolean(doc,member,&flag) != SPARK_STATUS_OK )
		return 0;
	options->stream = member >= 0 && flag ? 1u : 0u;
	member = SparkJsonFindObjectMember(doc,root,"priority");
	if ( member >= 0 && SparkJsonGetUInt32(doc,member,&options->priority) != SPARK_STATUS_OK )
		return 0;
	member = SparkJsonFindObjectMember(doc,root,"deadline_ms");
	if ( member >= 0 && (SparkJsonGetUInt32(doc,member,&options->deadline_ms) != SPARK_STATUS_OK || options->deadline_ms == 0u) )
		return 0;
	return api_parse_sampling_options(doc,root,options);
}

static int api_resolve_seed(ApiOptions *options)
{
	if ( options->temperature == 0.0f )
		options->seed = 0u;
	else if ( options->seeded == 0u )
	{
#if defined(__APPLE__)
		return getentropy(&options->seed,sizeof(options->seed)) == 0;
#else
		return getrandom(&options->seed,sizeof(options->seed),0) == (ssize_t)sizeof(options->seed);
#endif
	}
	return 1;
}

_Static_assert(SPARK_MODEL_RESIDENT_CHAT_TEMPLATE_MAX_STOP_MARKERS <= API_MAX_STOP_TOKENS, "a chat template's stop markers fit a request's stop list");
_Static_assert(API_MAX_STOP_TOKENS <= SPARK_MODEL_BATCH_ENGINE_MAX_REQUEST_STOP_TOKEN_COUNT, "a request's stop list fits the engine's per-request stops");

static const char *api_artifact(void)
{
	const SparkModelServingAdapterDescriptor *adapter = S.engine != 0 ? SparkModelBatchEngineGetAdapterDescriptor(S.engine) : 0;
	return(api_identity(adapter != 0 ? adapter->artifact_sha256 : 0));
}

static void api_usage_counts_json(char *buffer, size_t capacity, uint32_t prompt, uint32_t output, uint32_t cached, uint32_t matched, uint64_t restore_wait_ns, uint64_t background_wait_ns)
{
	(void)snprintf(buffer,capacity,",\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u,\"prompt_tokens_details\":{\"cached_tokens\":%u}},"
		"\"cache\":{\"matched_tokens\":%u,\"resumed_tokens\":%u,\"restore_ms\":%.3f,\"background_wait_ms\":%.3f,\"boot\":\"%s\",\"artifact\":\"%.16s\"}",
		prompt,output,prompt + output,cached,matched > cached ? matched : cached,cached,(double)restore_wait_ns / 1e6,(double)background_wait_ns / 1e6,ApiBootId,api_artifact());
}

static void api_usage_json(char *buffer, size_t capacity, const ApiRequest *req)
{
	api_usage_counts_json(buffer,capacity,req->prompt_count,req->output_token_count,req->cached_prompt_token_count,req->matched_prompt_token_count,
		req->restore_wait_ns,req->background_wait_ns);
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

static size_t api_logprobs_json_capacity(const ApiRequest *req, uint32_t token_count)
{
	return req->logprob_count != 0u ? (size_t)token_count * ((size_t)req->logprob_count * 36u + 4u) + 32u : 0u;
}

static int api_logprobs_json(char *out, size_t capacity, size_t *length, const ApiRequest *req, uint32_t first, uint32_t token_count)
{
	const SparkSamplingLogprob *entry;
	uint32_t token,index;
	int written;
	if ( req->logprob_count == 0u )
		return 1;
	written = snprintf(out + *length,capacity - *length,",\"token_logprobs\":[");
	if ( written < 0 || (size_t)written >= capacity - *length )
		return 0;
	*length += (size_t)written;
	for (token = 0u; token < token_count; token++)
		for (index = 0u; index < req->logprob_count; index++)
		{
			entry = &req->logprobs[(size_t)(first + token) * req->logprob_count + index];
			written = isfinite(entry->logprob)
				? snprintf(out + *length,capacity - *length,"%s%s[%u,%.9g]%s",index == 0u && token != 0u ? "," : "",index == 0u ? "[" : ",",entry->token,(double)entry->logprob,index + 1u == req->logprob_count ? "]" : "")
				: snprintf(out + *length,capacity - *length,"%s%s[%u,null]%s",index == 0u && token != 0u ? "," : "",index == 0u ? "[" : ",",entry->token,index + 1u == req->logprob_count ? "]" : "");
			if ( written < 0 || (size_t)written >= capacity - *length )
				return 0;
			*length += (size_t)written;
		}
	if ( *length + 2u > capacity )
		return 0;
	out[(*length)++] = ']';
	out[*length] = '\0';
	return 1;
}

static int api_stream_open(int fd, ApiStream *stream, const ApiRequest *req)
{
	static const char headers[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
	memset(stream,0,sizeof(*stream));
	stream->first = 1u;
	stream->piece_capacity = (HaveSidecar ? Sidecar.maximum_token_text_bytes : 0u) + 1u;
	stream->pending_capacity = API_STREAM_BATCH_TOKENS * stream->piece_capacity + API_UTF8_MAX_SEQUENCE;
	stream->escaped_capacity = (size_t)stream->pending_capacity * 6u + 8u;
	stream->event_capacity = stream->escaped_capacity + (size_t)API_STREAM_BATCH_TOKENS * 12u + api_logprobs_json_capacity(req,API_STREAM_BATCH_TOKENS) + 512u + API_CACHE_JSON_BYTES;
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
	stream->cached_tokens = req->cached_prompt_token_count;
	stream->matched_tokens = req->matched_prompt_token_count;
	stream->restore_wait_ns = req->restore_wait_ns;
	stream->background_wait_ns = req->background_wait_ns;
	pthread_mutex_unlock(&S.queue_mutex);
	return count;
}

static int api_stream_text(ApiStream *stream, const uint32_t *tokens, uint32_t count, uint32_t flush)
{
	uint32_t index,bytes,ready;
	size_t escaped_length = 0u;
	stream->escaped[0] = '\0';
	if ( !HaveSidecar )
		return 1;
	for (index = 0u; index < count; index++)
	{
		if ( SparkTokenizerSidecarDecodeText(&Sidecar,tokens + index,1u,0,0u,0u,stream->piece,stream->piece_capacity,&bytes) != SPARK_STATUS_OK || stream->pending_bytes + bytes > stream->pending_capacity )
			return 0;
		memcpy(stream->pending + stream->pending_bytes,stream->piece,bytes);
		stream->pending_bytes += bytes;
	}
	ready = flush != 0u ? stream->pending_bytes : api_utf8_complete(stream->pending,stream->pending_bytes);
	if ( !SparkJsonAppendEscaped(stream->escaped,stream->escaped_capacity,&escaped_length,stream->pending,ready) )
		return 0;
	memmove(stream->pending,stream->pending + ready,stream->pending_bytes - ready);
	stream->pending_bytes -= ready;
	return 1;
}

static int api_stream_event(int fd, ApiStream *stream, int chat_format, const ApiRequest *req, uint32_t first_token, const uint32_t *tokens, uint32_t count, const ApiRequest *finished)
{
	char usage[API_USAGE_JSON_BYTES],reason[24];
	size_t length;
	uint32_t index;
	int written;
	usage[0] = '\0';
	(void)snprintf(reason,sizeof(reason),finished != 0 ? "\"%s\"" : "null",finished != 0 ? api_finish_reason(finished) : "");
	if ( finished != 0 )
		api_usage_json(usage,sizeof(usage),finished);
	else if ( stream->first != 0u )
		api_usage_counts_json(usage,sizeof(usage),req->prompt_count,stream->sent_tokens,stream->cached_tokens,stream->matched_tokens,stream->restore_wait_ns,
			stream->background_wait_ns);
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
	stream->event[length++] = ']';
	if ( !api_logprobs_json(stream->event,stream->event_capacity,&length,req,first_token,count) || length + 8u > stream->event_capacity )
		return 0;
	memcpy(stream->event + length,API_STREAM_EVENT_END,sizeof(API_STREAM_EVENT_END) - 1u);
	stream->first = 0u;
	return send_all(fd,stream->event,length + sizeof(API_STREAM_EVENT_END) - 1u);
}

static int api_failure(const ApiRequest *req, char *body, size_t capacity)
{
	const SparkModelServingAdapterDescriptor *adapter = S.engine != 0 ? SparkModelBatchEngineGetAdapterDescriptor(S.engine) : 0;
	uint32_t capabilities = adapter != 0 ? adapter->capability_flags : 0u;
	uint32_t missing = __atomic_load_n(&S.health_missing_rank,__ATOMIC_ACQUIRE);
	uint32_t expired = req->deadline_expired != 0u || (req->status != 0u && req->deadline_ms != 0u && api_now_ms() >= req->deadline_ms) ? 1u : 0u;
	int code = 500;
	if ( expired != 0u && missing != UINT32_MAX )
		code = 504, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"deadline exceeded while rank %u is disconnected\",\"type\":\"timeout\",\"code\":\"deadline_exceeded\",\"missing_rank\":%u}}",missing,missing);
	else if ( expired != 0u )
		code = 504, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"deadline exceeded\",\"type\":\"timeout\",\"code\":\"deadline_exceeded\"}}");
	else if ( req->engine_latched != 0u )
		code = 503, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"the engine latched %s; restart or reset the deployment's ranks\",\"type\":\"server_error\",\"code\":\"engine_failed\"}}",SparkStatusToString((SparkStatus)req->status));
	else if ( req->submitted == 0 && req->status == (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED )
		code = 400, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"prompt plus max_tokens exceeds the deployment's context or KV capacity\",\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\"}}");
	else if ( req->submitted == 0 && req->status == (uint32_t)SPARK_STATUS_UNSUPPORTED && req->logprob_count != 0u && (capabilities & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS) == 0u )
		code = 400, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"this deployment does not return logprobs\",\"type\":\"invalid_request_error\",\"code\":\"logprobs_unsupported\"}}");
	else if ( req->submitted == 0 && req->status == (uint32_t)SPARK_STATUS_UNSUPPORTED && req->temperature != 0.0f && (capabilities & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING) != 0u )
		code = 400, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"this deployment samples with temperature only; send top_p 1 and top_k 0\",\"type\":\"invalid_request_error\",\"code\":\"truncation_unsupported\"}}");
	else if ( req->submitted == 0 && req->status == (uint32_t)SPARK_STATUS_UNSUPPORTED )
		code = 400, (void)snprintf(body,capacity,"{\"error\":{\"message\":\"this deployment decodes greedily only; send temperature 0\",\"type\":\"invalid_request_error\",\"code\":\"sampling_unsupported\"}}");
	else
		(void)snprintf(body,capacity,"{\"error\":{\"message\":\"model status %u\",\"type\":\"model_error\",\"code\":%u}}",req->status,req->status);
	return code;
}

static void api_stream_close(int fd, const ApiRequest *req, ApiStream *stream, int chat_format, int alive)
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
		alive = api_stream_text(stream,0,0u,1u) && api_stream_event(fd,stream,chat_format,req,stream->sent_tokens,0,0u,req);
	if ( alive )
		(void)send_all(fd,done,sizeof(done) - 1u);
	free(stream->pending);
	free(stream->piece);
	free(stream->escaped);
	free(stream->event);
}

static void api_send_completion(int fd, const ApiRequest *req, int chat_format)
{
	uint32_t text_capacity = 1u,text_bytes = 0u;
	char *text = 0,*escaped = 0,*response = 0,*logprobs = 0,usage[API_USAGE_JSON_BYTES],err[160];
	size_t escaped_capacity,escaped_length = 0u,response_capacity,logprobs_capacity,logprobs_length = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	if (HaveSidecar)
		text_capacity = req->output_token_count * Sidecar.maximum_token_text_bytes + 1u;
	escaped_capacity = (size_t)text_capacity * 6u + 8u;
	logprobs_capacity = api_logprobs_json_capacity(req,req->output_token_count) + 1u;
	response_capacity = (size_t)req->tokens_json_len + escaped_capacity + logprobs_capacity + 384u + API_CACHE_JSON_BYTES;
	text = malloc(text_capacity);
	escaped = malloc(escaped_capacity);
	response = malloc(response_capacity);
	logprobs = malloc(logprobs_capacity);
	if (text == 0 || escaped == 0 || response == 0 || logprobs == 0)
		status = SPARK_STATUS_INTERNAL_ERROR;
	else
	{
		logprobs[0] = '\0';
		if (!api_logprobs_json(logprobs,logprobs_capacity,&logprobs_length,req,0u,req->output_token_count))
			status = SPARK_STATUS_INTERNAL_ERROR;
	}
	if (status == SPARK_STATUS_OK && HaveSidecar)
		status = SparkTokenizerSidecarDecodeText(&Sidecar,req->output_token_ids,req->output_token_count,0,0u,0u,text,text_capacity,&text_bytes);
	if (status == SPARK_STATUS_OK && !SparkJsonAppendEscaped(escaped,escaped_capacity,&escaped_length,text,text_bytes))
		status = SPARK_STATUS_INTERNAL_ERROR;
	api_usage_json(usage,sizeof(usage),req);
	if (status == SPARK_STATUS_OK && !HaveSidecar)
		(void)snprintf(response,response_capacity,"{\"object\":\"text_completion\",\"tokens\":[%s]%s,\"finish_reason\":\"%s\"%s,\"status\":0}",req->tokens_json,logprobs,api_finish_reason(req),usage);
	else if (status == SPARK_STATUS_OK && chat_format)
		(void)snprintf(response,response_capacity,"{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"%s\"}]%s,\"tokens\":[%s]%s,\"status\":0}",escaped,api_finish_reason(req),usage,req->tokens_json,logprobs);
	else if (status == SPARK_STATUS_OK)
		(void)snprintf(response,response_capacity,"{\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"%s\",\"finish_reason\":\"%s\"}]%s,\"tokens\":[%s]%s,\"status\":0}",escaped,api_finish_reason(req),usage,req->tokens_json,logprobs);
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
	free(logprobs);
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
	uint32_t tokens[API_STREAM_BATCH_TOKENS],count,first_token;
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
	alive = api_stream_open(fd,&stream,req);
	while ( alive )
	{
		api_await(fd,req,stream.sent_tokens,API_WAIT_TOKENS);
		first_token = stream.sent_tokens;
		count = api_stream_take(req,&stream,tokens);
		if ( count == 0u && (req->done != 0 || S.running == 0) )
			break;
		if ( count != 0u )
			alive = api_stream_text(&stream,tokens,count,0u) && api_stream_event(fd,&stream,chat_format,req,first_token,tokens,count,0);
	}
	if ( req->done == 0 )
		api_abandon(req,alive != 0 ? SPARK_STATUS_INTERNAL_ERROR : SPARK_STATUS_IO_ERROR,0u);
	api_stream_close(fd,req,&stream,chat_format,alive);
}

static int api_parse_chat_thinking(const SparkJsonDocument *doc, int32_t root, bool *thinking)
{
	static const char *const kwargs_members[] = {"enable_thinking"};
	int32_t kwargs;
	*thinking = false;
	kwargs = root >= 0 ? SparkJsonFindObjectMember(doc,root,"chat_template_kwargs") : -1;
	if ( kwargs < 0 )
		return 1;
	return SparkJsonValidateObjectMembersExact(doc,kwargs,kwargs_members,1u) == SPARK_STATUS_OK &&
		SparkJsonGetBoolean(doc,SparkJsonFindObjectMember(doc,kwargs,"enable_thinking"),thinking) == SPARK_STATUS_OK;
}

static uint32_t api_adapter_capabilities(void)
{
	const SparkModelServingAdapterDescriptor *adapter = S.engine != 0 ? SparkModelBatchEngineGetAdapterDescriptor(S.engine) : 0;
	return adapter != 0 ? adapter->capability_flags : 0u;
}

static void handle_completion(int fd, char *body, uint32_t body_len,
	int chat_format)
{
	SparkJsonDocument doc;
	int32_t root, mt;
	uint32_t request_stops[API_MAX_STOP_TOKENS];
	uint32_t request_stop_count = 0;
	uint32_t chat_request = 0;
	bool chat_thinking = false;
	memset(&doc,0,sizeof(doc));
	uint32_t *prompt = 0, prompt_len = 0, max_tokens = 32;
	ApiOptions options;
	char *prompt_text = 0;
	uint32_t prompt_text_bytes = 0;
	ApiRequest *req;
	if ( (api_adapter_capabilities() & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ROWS_ONLY) != 0u )
	{
		send_response(fd, 400, "{\"error\":{\"message\":\"this deployment's model has no vocabulary head and writes no text; "
			"send its inputs to /v1/embeddings\",\"type\":\"invalid_request_error\",\"code\":\"completions_unsupported\"}}");
		return;
	}
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
	if (!api_parse_chat_thinking(&doc, root, &chat_thinking) ||
		((prompt_text != 0 || prompt != 0) &&
			SparkJsonFindObjectMember(&doc, root, "chat_template_kwargs") >= 0))
	{
		SparkJsonDocumentDestroy(&doc);
		free(prompt);
		free(prompt_text);
		send_response(fd, 400,
			"{\"error\":{\"message\":\"chat_template_kwargs applies only to messages and "
			"must be exactly {\\\"enable_thinking\\\": boolean}\","
			"\"type\":\"invalid_request_error\",\"code\":\"invalid_option\"}}");
		return;
	}
	if (prompt_text == 0 && prompt == 0 && root >= 0)
	{
		char *chat_text = 0;
		uint32_t chat_bytes = 0u;
		SparkChatTemplateOutcome chat_outcome = SparkChatTemplateRender(ChatTemplate,
			&doc, root, chat_thinking, &chat_text, &chat_bytes);
		if (chat_outcome == SPARK_CHAT_TEMPLATE_RENDERED)
		{
			prompt_text = chat_text;
			prompt_text_bytes = chat_bytes;
			chat_request = 1u;
		}
		else if (chat_outcome != SPARK_CHAT_TEMPLATE_NOT_CHAT)
		{
			char error_body[512];
			SparkJsonDocumentDestroy(&doc);
			(void)snprintf(error_body, sizeof(error_body),
				"{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\",\"code\":\"%s\"}}",
				SparkChatTemplateOutcomeMessage(chat_outcome), SparkChatTemplateOutcomeCode(chat_outcome));
			send_response(fd, chat_outcome == SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY ? 500 : 400, error_body);
			return;
		}
	}
	if (root >= 0)
	{
		int32_t stop_member = SparkJsonFindObjectMember(&doc, root, "stop_token_ids");
		uint32_t *stops = 0;
		uint32_t stop_len = 0u;
		if (stop_member >= 0 &&
			!(SparkJsonTokenIsType(&doc, stop_member, SPARK_JSON_TOKEN_ARRAY) &&
				SparkJsonGetArrayElementCount(&doc, stop_member) == 0))
		{
			stop_len = parse_token_array(&doc, root, "stop_token_ids", &stops);
			if (stop_len == 0u || stop_len > API_MAX_STOP_TOKENS)
			{
				free(stops);
				SparkJsonDocumentDestroy(&doc);
				free(prompt);
				free(prompt_text);
				send_response(fd, 400,
					"{\"error\":{\"message\":\"stop_token_ids must be an array of at most 16 token ids\","
					"\"type\":\"invalid_request_error\",\"code\":\"invalid_stop_token_ids\"}}");
				return;
			}
			memcpy(request_stops, stops, (size_t)stop_len * sizeof(uint32_t));
			request_stop_count = stop_len;
			free(stops);
		}
	}
	for (uint32_t marker = 0u; chat_request != 0u && marker < ChatStopTokenCount; ++marker)
	{
		uint32_t present = 0u;
		for (uint32_t index = 0u; index < request_stop_count; ++index)
			present |= request_stops[index] == ChatStopTokens[marker] ? 1u : 0u;
		if (present == 0u && request_stop_count < API_MAX_STOP_TOKENS)
			request_stops[request_stop_count++] = ChatStopTokens[marker];
	}
	mt = SparkJsonFindObjectMember(&doc, root, "max_tokens");
	if (mt >= 0)
	{
		uint32_t v;
		if (SparkJsonGetUInt32(&doc, mt, &v) != SPARK_STATUS_OK || v == 0u || v > API_MAX_OUTPUT_TOKENS)
		{
			SparkJsonDocumentDestroy(&doc);
			free(prompt);
			free(prompt_text);
			send_response(fd, 400, "{\"error\":{\"message\":\"max_tokens must be an integer from 1 to 131072\",\"type\":\"invalid_request_error\"}}");
			return;
		}
		max_tokens = v;
	}
	if (!api_parse_serving_options(&doc, root, &options))
	{
		SparkJsonDocumentDestroy(&doc);
		free(prompt);
		free(prompt_text);
		send_response(fd, 400,
			"{\"error\":{\"message\":\"stream must be a boolean; priority, deadline_ms and seed "
			"must be unsigned integers, deadline_ms nonzero; temperature must be 0 or between "
			"0.0001 and 2; top_p must be above 0 and at most 1; top_k must be an unsigned "
			"integer of at most 1048576; logprobs must be an unsigned integer of at most 20\","
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
		const uint32_t prefix_count = chat_request == 0u ? TextPromptPrefixCount : 0u;
		SparkTokenizerWorkspaceReset(&workspace);
		prompt = malloc(((size_t)prompt_text_bytes + 1u + prefix_count) * sizeof(uint32_t));
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
		encoding.token_ids = prompt + prefix_count;
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
		memcpy(prompt, TextPromptPrefix, (size_t)prefix_count * sizeof(uint32_t));
		prompt_len = encoding.token_count + prefix_count;
	}
	if (prompt_len == 0)
	{
		free(prompt);
		send_response(fd, 400, "{\"error\":\"prompt_token_ids required\"}");
		return;
	}
	if (!api_fit_context(prompt_len, &max_tokens))
	{
		char err[224];
		free(prompt);
		(void)snprintf(err, sizeof(err), "{\"error\":{\"message\":\"the prompt has %u tokens and this deployment serves %u positions, so no output fits\",\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\"}}", prompt_len, __atomic_load_n(&S.context_limit,__ATOMIC_ACQUIRE));
		send_response(fd, 400, err);
		return;
	}
	if (!api_resolve_seed(&options))
	{
		free(prompt);
		send_response(fd, 500, "{\"error\":{\"message\":\"no entropy for a sampling seed\",\"type\":\"server_error\",\"code\":\"seed_unavailable\"}}");
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
	req->token_ready_ns = malloc((size_t)max_tokens * sizeof(uint64_t));
	req->tokens_json_capacity = (size_t)max_tokens * API_TOKEN_TEXT_BYTES + API_TOKEN_TEXT_BYTES;
	req->tokens_json = malloc(req->tokens_json_capacity);
	req->logprobs = options.logprobs != 0u ? malloc((size_t)max_tokens * options.logprobs * sizeof(req->logprobs[0])) : 0;
	req->stop_tokens = request_stop_count != 0u ? malloc((size_t)request_stop_count * sizeof(uint32_t)) : 0;
	if (req->output_token_ids == 0 || req->token_ready_ns == 0 || req->tokens_json == 0 || (options.logprobs != 0u && req->logprobs == 0) ||
		(request_stop_count != 0u && req->stop_tokens == 0))
	{
		free(prompt);
		free(req->output_token_ids);
		free(req->token_ready_ns);
		free(req->tokens_json);
		free(req->logprobs);
		free(req->stop_tokens);
		free(req);
		send_response(fd, 500, "{\"error\":\"oom\"}");
		return;
	}
	req->tokens_json[0] = '\0';
	pthread_mutex_init(&req->mutex, 0);
	pthread_cond_init(&req->cond, 0);
	req->started_ms = api_now_ms();
	if (request_stop_count != 0u)
		memcpy(req->stop_tokens, request_stops, (size_t)request_stop_count * sizeof(uint32_t));
	req->stop_token_count = request_stop_count;
	req->stream = options.stream;
	req->priority = options.priority;
	req->deadline_ms = options.deadline_ms != 0u ? req->started_ms + options.deadline_ms : 0u;
	req->temperature = options.temperature;
	req->seed = options.seed;
	req->top_p = options.top_p;
	req->top_k = options.top_k;
	req->logprob_count = options.logprobs;
	req->drafts_disabled = options.drafts_disabled;
	req->cache_point_count = options.cache_point_count;
	memcpy(req->cache_points,options.cache_points,sizeof(req->cache_points));
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
	if (options.stream)
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

static float api_bf16_value(uint16_t bits)
{
	const uint32_t wide = (uint32_t)bits << 16;
	float value;
	memcpy(&value,&wide,sizeof(value));
	return value;
}

static size_t api_base64(const uint8_t *data, size_t bytes, char *out)
{
	static const char digits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	size_t in = 0u,length = 0u;
	for (; in + 2u < bytes; in += 3u)
	{
		const uint32_t word = ((uint32_t)data[in] << 16) | ((uint32_t)data[in + 1u] << 8) | data[in + 2u];
		out[length++] = digits[word >> 18];
		out[length++] = digits[(word >> 12) & 63u];
		out[length++] = digits[(word >> 6) & 63u];
		out[length++] = digits[word & 63u];
	}
	if ( in < bytes )
	{
		const uint32_t word = ((uint32_t)data[in] << 16) | (in + 1u < bytes ? (uint32_t)data[in + 1u] << 8 : 0u);
		out[length++] = digits[word >> 18];
		out[length++] = digits[(word >> 12) & 63u];
		out[length++] = in + 1u < bytes ? digits[(word >> 6) & 63u] : '=';
		out[length++] = '=';
	}
	return length;
}

static void api_embedding_error(int fd, int code, const char *message, const char *error_code)
{
	char body[384];
	(void)snprintf(body,sizeof(body),"{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\",\"code\":\"%s\"}}",message,error_code);
	send_response(fd,code,body);
}

static uint32_t api_embedding_tokens(SparkJsonDocument *doc, int32_t element, uint32_t **tokens_out)
{
	uint32_t *tokens = 0,count = 0u;
	*tokens_out = 0;
	if ( SparkJsonTokenIsType(doc,element,SPARK_JSON_TOKEN_STRING) )
	{
		SparkTokenizerWorkspace workspace;
		SparkTokenizerEncoding encoding;
		char *text = 0;
		uint32_t bytes;
		SparkStatus status;
		if ( !HaveSidecar || SparkJsonCopyString(doc,element,&text) != SPARK_STATUS_OK )
			return 0u;
		bytes = (uint32_t)strlen(text);
		SparkTokenizerWorkspaceReset(&workspace);
		tokens = malloc(((size_t)bytes + 1u + TextPromptPrefixCount + TextPromptSuffixCount) * sizeof(uint32_t));
		if ( bytes == 0u || tokens == 0 || SparkTokenizerWorkspaceInitialize(&workspace,bytes + 1u) != SPARK_STATUS_OK )
		{
			free(text);
			free(tokens);
			return 0u;
		}
		SparkTokenizerEncodingReset(&encoding);
		encoding.token_capacity = bytes + 1u;
		encoding.token_ids = tokens + TextPromptPrefixCount;
		status = SparkTokenizerSidecarEncodeText(&Sidecar,text,bytes,0u,&workspace,&encoding);
		SparkTokenizerWorkspaceDestroy(&workspace);
		free(text);
		if ( status != SPARK_STATUS_OK )
		{
			free(tokens);
			return 0u;
		}
		memcpy(tokens,TextPromptPrefix,(size_t)TextPromptPrefixCount * sizeof(uint32_t));
		memcpy(tokens + TextPromptPrefixCount + encoding.token_count,TextPromptSuffix,(size_t)TextPromptSuffixCount * sizeof(uint32_t));
		count = TextPromptPrefixCount + encoding.token_count + TextPromptSuffixCount;
	}
	else if ( SparkJsonTokenIsType(doc,element,SPARK_JSON_TOKEN_ARRAY) )
	{
		int32_t item = SparkJsonGetArrayElementFirst(doc,element);
		uint32_t index;
		count = (uint32_t)SparkJsonGetArrayElementCount(doc,element);
		if ( count == 0u || count > API_MAX_PROMPT_TOKENS || (tokens = malloc((size_t)count * sizeof(uint32_t))) == 0 )
			return 0u;
		for (index=0u; index<count; index++)
		{
			if ( item < 0 || SparkJsonGetUInt32(doc,item,&tokens[index]) != SPARK_STATUS_OK || tokens[index] >= ApiOutputVocabulary )
			{
				free(tokens);
				return 0u;
			}
			item = SparkJsonGetArrayElementNext(doc,element,item);
		}
	}
	*tokens_out = tokens;
	return count;
}

static ApiRequest *api_embedding_request(uint32_t *tokens, uint32_t count, const ApiOptions *options, uint64_t started_ms)
{
	ApiRequest *req = calloc(1,sizeof(*req));
	if ( req == 0 )
		return 0;
	req->output_token_ids = malloc(sizeof(uint32_t));
	req->token_ready_ns = malloc(sizeof(uint64_t));
	req->tokens_json_capacity = 2u * API_TOKEN_TEXT_BYTES;
	req->tokens_json = malloc(req->tokens_json_capacity);
	if ( req->output_token_ids == 0 || req->token_ready_ns == 0 || req->tokens_json == 0 )
	{
		free(req->output_token_ids);
		free(req->token_ready_ns);
		free(req->tokens_json);
		free(req);
		return 0;
	}
	req->tokens_json[0] = '\0';
	pthread_mutex_init(&req->mutex,0);
	pthread_cond_init(&req->cond,0);
	req->started_ms = started_ms;
	req->priority = options->priority;
	req->deadline_ms = options->deadline_ms != 0u ? started_ms + options->deadline_ms : 0u;
	req->top_p = 1.0f;
	req->drafts_disabled = 1u;
	req->output_row = 1u;
	req->max_tokens = 1u;
	req->prompt_tokens = tokens;
	req->prompt_count = count;
	return req;
}

static size_t api_embedding_json(const ApiRequest *req, uint32_t index, uint32_t dimensions, int base64, char *out)
{
	const uint32_t width = dimensions != 0u && dimensions < req->row_value_count ? dimensions : req->row_value_count;
	size_t length = (size_t)sprintf(out,"%s{\"object\":\"embedding\",\"index\":%u,\"embedding\":%s",index != 0u ? "," : "",index,base64 ? "\"" : "[");
	double sum = 0.0,scale;
	float *values = malloc((size_t)width * sizeof(float));
	uint32_t element;
	if ( values == 0 )
		return 0u;
	for (element=0u; element<width; element++)
	{
		const double value = (double)api_bf16_value(req->row_values[element]);
		sum += value * value;
	}
	scale = sum > 0.0 ? 1.0 / sqrt(sum) : 0.0;
	for (element=0u; element<width; element++)
		values[element] = (float)((double)api_bf16_value(req->row_values[element]) * scale);
	if ( base64 )
		length += api_base64((const uint8_t *)values,(size_t)width * sizeof(float),out + length);
	else
		for (element=0u; element<width; element++)
			length += (size_t)sprintf(out + length,"%s%.9g",element != 0u ? "," : "",(double)values[element]);
	free(values);
	length += (size_t)sprintf(out + length,"%s}",base64 ? "\"" : "]");
	return length;
}

static void handle_embeddings(int fd, char *body, uint32_t body_len)
{
	const SparkModelServingAdapterDescriptor *adapter = SparkModelBatchEngineGetAdapterDescriptor(S.engine);
	SparkJsonDocument doc;
	ApiOptions options;
	ApiRequest **requests = 0;
	uint32_t **inputs = 0,*counts = 0;
	uint32_t count = 0u,index,dimensions = 0u,prompt_tokens = 0u,failed = UINT32_MAX,text_inputs = 0u;
	int32_t root,input,member,element;
	int base64 = 0;
	char *format = 0,*response = 0;
	size_t length = 0u,capacity;
	uint64_t started_ms = api_now_ms();
	memset(&doc,0,sizeof(doc));
	if ( adapter == 0 || (api_adapter_capabilities() & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ROW_OUTPUT) == 0u )
	{
		api_embedding_error(fd,400,"this deployment's model returns no embeddings","embeddings_unsupported");
		return;
	}
	if ( SparkJsonParseText(body,body_len,&doc) != SPARK_STATUS_OK )
	{
		send_response(fd,400,"{\"error\":\"invalid json\"}");
		return;
	}
	root = SparkJsonGetRootToken(&doc);
	input = root >= 0 ? SparkJsonFindObjectMember(&doc,root,"input") : -1;
	if ( input >= 0 && SparkJsonTokenIsType(&doc,input,SPARK_JSON_TOKEN_STRING) )
		count = 1u;
	else if ( input >= 0 && SparkJsonTokenIsType(&doc,input,SPARK_JSON_TOKEN_ARRAY) )
	{
		element = SparkJsonGetArrayElementFirst(&doc,input);
		count = element >= 0 && SparkJsonTokenIsType(&doc,element,SPARK_JSON_TOKEN_PRIMITIVE) ? 1u : (uint32_t)SparkJsonGetArrayElementCount(&doc,input);
	}
	member = root >= 0 ? SparkJsonFindObjectMember(&doc,root,"encoding_format") : -1;
	if ( member >= 0 && (SparkJsonCopyString(&doc,member,&format) != SPARK_STATUS_OK || (strcmp(format,"float") != 0 && strcmp(format,"base64") != 0)) )
		count = UINT32_MAX;
	base64 = format != 0 && strcmp(format,"base64") == 0;
	free(format);
	member = root >= 0 ? SparkJsonFindObjectMember(&doc,root,"dimensions") : -1;
	if ( member >= 0 && (SparkJsonGetUInt32(&doc,member,&dimensions) != SPARK_STATUS_OK || dimensions == 0u || dimensions > adapter->boundary_element_count) )
		count = UINT32_MAX;
	if ( count == UINT32_MAX || !api_parse_serving_options(&doc,root,&options) || options.stream != 0u || options.temperature != 0.0f || options.logprobs != 0u )
	{
		SparkJsonDocumentDestroy(&doc);
		api_embedding_error(fd,400,"encoding_format is float or base64; dimensions is an integer from 1 to the model's width; priority and deadline_ms are unsigned integers; an embedding takes no stream, sampling or logprobs","invalid_option");
		return;
	}
	if ( count == 0u || count > API_MAX_EMBEDDING_INPUTS )
	{
		SparkJsonDocumentDestroy(&doc);
		api_embedding_error(fd,400,"input is a text, a list of texts, a list of token ids or a list of such lists, with 1 to 256 entries","invalid_input");
		return;
	}
	inputs = calloc(count,sizeof(*inputs));
	counts = calloc(count,sizeof(*counts));
	requests = calloc(count,sizeof(*requests));
	if ( inputs == 0 || counts == 0 || requests == 0 )
		failed = 0u;
	element = SparkJsonTokenIsType(&doc,input,SPARK_JSON_TOKEN_ARRAY) ? SparkJsonGetArrayElementFirst(&doc,input) : input;
	if ( element >= 0 && SparkJsonTokenIsType(&doc,element,SPARK_JSON_TOKEN_PRIMITIVE) )
		element = input;
	for (index=0u; failed == UINT32_MAX && index<count; index++)
	{
		uint32_t budget = 1u;
		text_inputs += SparkJsonTokenIsType(&doc,element,SPARK_JSON_TOKEN_STRING) ? 1u : 0u;
		counts[index] = api_embedding_tokens(&doc,element,&inputs[index]);
		if ( counts[index] == 0u || !api_fit_context(counts[index],&budget) )
			failed = index;
		prompt_tokens += counts[index];
		if ( element != input )
			element = SparkJsonGetArrayElementNext(&doc,input,element);
	}
	SparkJsonDocumentDestroy(&doc);
	for (index=0u; failed == UINT32_MAX && index<count; index++)
	{
		requests[index] = api_embedding_request(inputs[index],counts[index],&options,started_ms);
		if ( requests[index] == 0 )
			failed = index;
		else
			inputs[index] = 0;
	}
	if ( failed != UINT32_MAX )
	{
		char message[256];
		for (index=0u; requests != 0 && index<count; index++)
			if ( requests[index] != 0 )
				api_request_destroy(requests[index]);
		for (index=0u; inputs != 0 && index<count; index++)
			free(inputs[index]);
		if ( inputs == 0 || counts == 0 || requests == 0 )
			send_response(fd,500,"{\"error\":\"oom\"}");
		else if ( text_inputs != 0u && !HaveSidecar )
			send_tokenizer_unavailable(fd);
		else
		{
			(void)snprintf(message,sizeof(message),"input %u is empty, holds a token outside the vocabulary, could not be encoded, or does not fit the %u positions this deployment serves",failed,__atomic_load_n(&S.context_limit,__ATOMIC_ACQUIRE));
			api_embedding_error(fd,400,message,"invalid_input");
		}
		free(inputs);
		free(counts);
		free(requests);
		return;
	}
	free(inputs);
	free(counts);
	pthread_mutex_lock(&S.queue_mutex);
	for (index=0u; index<count; index++)
	{
		requests[index]->id = ++S.next_id + 100000;
		if ( S.queue_tail != 0 )
			S.queue_tail->next = requests[index];
		else
			S.queue_head = requests[index];
		S.queue_tail = requests[index];
	}
	pthread_mutex_unlock(&S.queue_mutex);
	api_wake_worker();
	for (index=0u; index<count; index++)
	{
		api_await(fd,requests[index],0u,API_WAIT_DONE);
		if ( requests[index]->done == 0 )
			api_abandon(requests[index],SPARK_STATUS_INTERNAL_ERROR,0u);
		if ( failed == UINT32_MAX && (requests[index]->deadline_expired != 0u || requests[index]->status != 0u || requests[index]->row_values == 0) )
			failed = index;
	}
	capacity = 256u + (size_t)count * (96u + (size_t)adapter->boundary_element_count * (base64 ? 6u : 18u));
	response = failed == UINT32_MAX ? malloc(capacity) : 0;
	if ( response != 0 )
	{
		length = (size_t)sprintf(response,"{\"object\":\"list\",\"data\":[");
		for (index=0u; index<count; index++)
		{
			const size_t written = api_embedding_json(requests[index],index,dimensions,base64,response + length);
			if ( written == 0u )
				break;
			length += written;
		}
		if ( index == count )
		{
			char id[512];
			size_t id_length = 0u;
			if ( adapter->model_id == 0 || !SparkJsonAppendEscaped(id,sizeof(id),&id_length,adapter->model_id,strlen(adapter->model_id)) )
				id[0] = '\0';
			(void)sprintf(response + length,"],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%u,\"total_tokens\":%u}}",id,prompt_tokens,prompt_tokens);
			send_response(fd,200,response);
			api_logf("embeddings_done fd=%d inputs=%u prompt_tokens=%u ms=%llu",fd,count,prompt_tokens,(unsigned long long)(api_now_ms() - started_ms));
		}
		else
			send_response(fd,500,"{\"error\":\"oom\"}");
	}
	else if ( failed == UINT32_MAX )
		send_response(fd,500,"{\"error\":\"oom\"}");
	else if ( requests[failed]->deadline_expired == 0u && requests[failed]->status == 0u )
		send_response(fd,500,"{\"error\":{\"message\":\"the engine completed an embedding without its row\",\"type\":\"model_error\",\"code\":\"row_missing\"}}");
	else
	{
		char err[224];
		const int code = api_failure(requests[failed],err,sizeof(err));
		send_response(fd,code,err);
	}
	free(response);
	for (index=0u; index<count; index++)
		api_release_request(requests[index]);
	free(requests);
}

static int api_parse_identity(const char *text, SparkModelServingCacheIdentity *identity)
{
	uint32_t index;
	unsigned int byte;
	if ( text == 0 || strlen(text) != 2u * sizeof(identity->sha256) )
		return 0;
	for (index=0u; index<sizeof(identity->sha256); index++)
	{
		if ( strspn(text + 2u * index,"0123456789abcdefABCDEF") < 2u || sscanf(text + 2u * index,"%2x",&byte) != 1 )
			return 0;
		identity->sha256[index] = (uint8_t)byte;
	}
	return 1;
}

static void handle_cache_state(int fd, char *body, uint32_t body_len)
{
	SparkJsonDocument doc;
	SparkModelServingCacheIdentity identity;
	SparkModelBatchCacheState state;
	char response[768],hex[2u * sizeof(identity.sha256) + 1u];
	char *identity_text = 0;
	uint32_t *tokens = 0,token_count = 0u,by_identity = 0u,status,index;
	int32_t root,member;
	memset(&doc,0,sizeof(doc));
	memset(&identity,0,sizeof(identity));
	if ( SparkJsonParseText(body,body_len,&doc) != SPARK_STATUS_OK )
	{
		send_response(fd,400,"{\"error\":\"invalid json\"}");
		return;
	}
	root = SparkJsonGetRootToken(&doc);
	if ( root >= 0 )
		token_count = parse_token_array(&doc,root,"prompt_token_ids",&tokens);
	member = root >= 0 ? SparkJsonFindObjectMember(&doc,root,"identity") : -1;
	if ( member >= 0 && tokens == 0 && SparkJsonTokenIsType(&doc,member,SPARK_JSON_TOKEN_STRING) && SparkJsonCopyString(&doc,member,&identity_text) == SPARK_STATUS_OK &&
		api_parse_identity(identity_text,&identity) != 0 &&
		SparkJsonGetUInt32(&doc,SparkJsonFindObjectMember(&doc,root,"token_count"),&token_count) == SPARK_STATUS_OK && token_count != 0u )
		by_identity = 1u;
	free(identity_text);
	SparkJsonDocumentDestroy(&doc);
	if ( (tokens == 0) == (by_identity == 0u) )
	{
		free(tokens);
		send_response(fd,400,"{\"error\":{\"message\":\"send prompt_token_ids, or identity (64 hex digits) with token_count\",\"type\":\"invalid_request_error\",\"code\":\"invalid_request\"}}");
		return;
	}
	pthread_mutex_lock(&Probe.mutex);
	if ( Probe.stage != API_PROBE_IDLE )
	{
		pthread_mutex_unlock(&Probe.mutex);
		free(tokens);
		send_response(fd,429,"{\"error\":{\"message\":\"another cache state request is running; try again\",\"type\":\"rate_limit_error\",\"code\":\"busy\"}}");
		return;
	}
	Probe.tokens = tokens;
	Probe.token_count = token_count;
	Probe.by_identity = by_identity;
	Probe.identity = identity;
	Probe.deadline_ms = api_now_ms() + API_CACHE_STATE_WAIT_MS;
	Probe.stage = API_PROBE_ASKED;
	pthread_mutex_unlock(&Probe.mutex);
	api_wake_worker();
	pthread_mutex_lock(&Probe.mutex);
	while ( Probe.stage != API_PROBE_DONE && S.running != 0 && api_now_ms() < Probe.deadline_ms + API_CACHE_STATE_GRACE_MS )
	{
		struct timespec until;
		clock_gettime(CLOCK_REALTIME,&until);
		until.tv_nsec += (long)API_CACHE_STATE_SLICE_MS * 1000000L;
		if ( until.tv_nsec >= 1000000000L )
		{
			until.tv_sec += 1;
			until.tv_nsec -= 1000000000L;
		}
		(void)pthread_cond_timedwait(&Probe.cond,&Probe.mutex,&until);
		if ( Probe.stage != API_PROBE_DONE )
			api_wake_worker();
	}
	state = Probe.state;
	status = Probe.stage == API_PROBE_DONE ? Probe.status : Probe.stage == API_PROBE_WAITING ? (uint32_t)SPARK_STATUS_PENDING : (uint32_t)SPARK_STATUS_IO_ERROR;
	Probe.tokens = 0;
	Probe.stage = Probe.stage == API_PROBE_WAITING ? API_PROBE_ABANDONED : API_PROBE_IDLE;
	pthread_mutex_unlock(&Probe.mutex);
	api_wake_worker();
	free(tokens);
	if ( status != (uint32_t)SPARK_STATUS_OK && status != (uint32_t)SPARK_STATUS_PENDING )
	{
		(void)snprintf(response,sizeof(response),"{\"error\":{\"message\":\"the cache state could not be read: %s\",\"type\":\"server_error\",\"code\":\"cache_state_unavailable\"}}",
			SparkStatusToString((SparkStatus)status));
		send_response(fd,status == (uint32_t)SPARK_STATUS_BUSY ? 429 : 503,response);
		return;
	}
	for (index=0u; index<sizeof(state.identity.sha256); index++)
		(void)snprintf(hex + 2u * index,3u,"%02x",state.identity.sha256[index]);
	(void)snprintf(response,sizeof(response),
		"{\"object\":\"cache.state\",\"complete\":%s,\"prompt_tokens\":%u,\"matched_tokens\":%u,\"resume_tokens\":%u,\"probed_tokens\":%u,\"identity\":\"%s\","
		"\"ranks\":%u,\"ranks_reported\":%u,\"ranks_failed\":%u,\"held_tokens\":%u,\"resident_tokens\":%u,\"saved_tokens\":%u,\"state_tokens\":%u,\"read_tokens\":%u,"
		"\"block_tokens\":%u,\"checkpoint_tokens\":%u,\"boot\":\"%s\",\"artifact\":\"%.16s\"}",
		status == (uint32_t)SPARK_STATUS_OK ? "true" : "false",state.prompt_token_count,state.matched_token_count,state.resume_token_count,state.probed_token_count,
		state.probed_token_count != 0u ? hex : "",state.rank_count,state.reported_rank_count,state.failed_rank_count,state.held_token_count,state.resident_token_count,
		state.saved_token_count,
		state.state_token_count,state.read_token_count,state.cache_block_token_count,state.cache_checkpoint_token_count,ApiBootId,api_artifact());
	send_response(fd,200,response);
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
	if (strcmp(method, "GET") == 0 && strcmp(path, "/health/live") == 0)
		send_response(fd, 200, api_engines_ready() ? "{\"status\":\"live\",\"ready\":true}" : "{\"status\":\"live\",\"ready\":false}");
	else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0 && api_engines_ready() == 0)
	{
		char b[256];
		api_bringup_json(b, sizeof(b));
		send_response(fd, 503, b);
	}
	else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0)
	{
		char b[640];
		uint32_t ranks = __atomic_load_n(&S.health_rank_count, __ATOMIC_ACQUIRE);
		uint32_t write_alert = __atomic_load_n(&S.health_write_alert_rank, __ATOMIC_ACQUIRE);
		uint32_t connected = __atomic_load_n(&S.health_connected_rank_count, __ATOMIC_ACQUIRE);
		uint32_t missing = __atomic_load_n(&S.health_missing_rank, __ATOMIC_ACQUIRE);
		uint32_t failed = __atomic_load_n(&S.health_failed_status, __ATOMIC_ACQUIRE);
		uint32_t eager = __atomic_load_n(&S.health_degraded_rank, __ATOMIC_ACQUIRE);
		uint32_t degraded = connected < ranks || eager != UINT32_MAX || (failed != 0u && failed != (uint32_t)SPARK_STATUS_IO_ERROR) ? 1u : 0u;
		(void)snprintf(b, sizeof(b),
			"{\"status\":\"%s\",\"served\":%llu,\"tokenizer\":%s,\"ranks\":%u,\"connected_ranks\":%u,\"missing_rank\":%d,\"degraded_rank\":%d,\"degraded_path\":\"%s\",\"engine_status\":\"%s\",\"live_requests\":%u,\"idle_ms\":%llu,\"kv_write_alert_rank\":%d,\"kv_capacity_tokens\":%llu,\"kv_capacity_max_tokens\":%llu}",
			degraded != 0u ? "degraded" : "ok", (unsigned long long)S.served, HaveSidecar ? "true" : "false", ranks, connected,
			missing != UINT32_MAX ? (int)missing : -1, eager != UINT32_MAX ? (int)eager : -1, eager != UINT32_MAX ? "eager" : "none",
			SparkStatusToString((SparkStatus)failed), __atomic_load_n(&S.health_live_requests, __ATOMIC_ACQUIRE),
			(unsigned long long)(api_now_ms() - __atomic_load_n(&S.last_activity_ms, __ATOMIC_ACQUIRE)),
			write_alert != UINT32_MAX ? (int)write_alert : -1, (unsigned long long)__atomic_load_n(&S.health_kv_capacity_tokens, __ATOMIC_ACQUIRE),
			(unsigned long long)__atomic_load_n(&S.health_kv_capacity_max_tokens, __ATOMIC_ACQUIRE));
		send_response(fd, degraded != 0u ? 503 : 200, b);
	}
	else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0)
	{
		const SparkModelServingAdapterDescriptor *adapter = api_engines_ready() ? SparkModelBatchEngineGetAdapterDescriptor(S.engine) : 0;
		char b[1024],id[512];
		size_t id_length = 0u;
		if (adapter == 0 || adapter->model_id == 0 || !SparkJsonAppendEscaped(id, sizeof(id), &id_length, adapter->model_id, strlen(adapter->model_id)))
		{
			api_bringup_json(b, sizeof(b));
			send_response(fd, 503, b);
		}
		else
		{
			(void)snprintf(b, sizeof(b),
				"{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
				"\"object\":\"model\",\"owned_by\":\"sparkpipe\","
				"\"max_model_len\":%u,\"served\":%llu}]}",
				id, __atomic_load_n(&S.context_limit, __ATOMIC_ACQUIRE), (unsigned long long)S.served);
			send_response(fd, 200, b);
		}
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/cache/state") == 0 && api_engines_ready() != 0)
		handle_cache_state(fd, body, body_len);
	else if (strcmp(method, "POST") == 0 &&
		(strcmp(path, "/v1/completions") == 0 || strcmp(path, "/v1/embeddings") == 0 ||
		 strcmp(path, "/v1/chat/completions") == 0 || strcmp(path, "/v1/cache/state") == 0) && api_engines_ready() == 0)
	{
		char b[256];
		api_bringup_json(b, sizeof(b));
		send_response(fd, 503, b);
	}
	else if (strcmp(method, "POST") == 0 &&
		(strcmp(path, "/v1/completions") == 0 ||
		 strcmp(path, "/v1/chat/completions") == 0))
		handle_completion(fd, body, body_len,
			strcmp(path, "/v1/chat/completions") == 0);
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/embeddings") == 0)
		handle_embeddings(fd, body, body_len);
	else
		send_response(fd, 404, "{\"error\":\"not found\"}");
	free(body_base);
	close(fd);
	return 0;
}

static SparkModelBatchEngineConfiguration ApiEngineConfiguration;
static pthread_t ApiWorker;

static void api_seed_submission_ids(void)
{
	char seq_path[1024];
	uint64_t session = SparkModelBatchEngineSessionFingerprint(S.engine);
	uint64_t saved_session = 0u;
	uint64_t saved_id = 0u;
	uint64_t seeded = 1000000u;
	(void)snprintf(seq_path,sizeof(seq_path),"%s/api_submission.seq",S.runtime_root);
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
	    SparkModelBatchEnginePeekSubmissionId(S.engine) != seeded ? "above the highest submission the ranks have accepted" :
	    seeded != 1000000u ? "continued within engine session, crash window skipped" :
	        "rebased — new engine session");
}

static void *api_bringup(void *argument)
{
	unsigned attempt = 0u, waited = 0u, pause_s = 1u;
	SparkStatus status;
	(void)argument;
	for (;;)
	{
		attempt++;
		S.bringup_attempts = attempt;
		S.bringup_phase = "connecting";
		status = SparkModelBatchEngineConnect(&ApiEngineConfiguration, &S.engine);
		if (status == SPARK_STATUS_OK)
			break;
		S.bringup_status = (uint32_t)status;
		if (attempt <= 10u || attempt % 30u == 0u)
			api_logf("engine_connect_failed attempt=%u status=%s elapsed_ms=%llu; retrying in %us", attempt,
				SparkStatusToString(status), (unsigned long long)(api_now_ms() - S.bringup_started_ms), pause_s);
		sleep(pause_s);
		if (pause_s < 8u)
			pause_s *= 2u;
	}
	S.bringup_phase = "waiting_ranks";
	while (SparkModelBatchEngineAllRanksReady(S.engine) == 0u)
	{
		if ((waited % 10u) == 0u)
			api_logf("engine_ranks_waiting elapsed_ms=%llu: not every rank is connected and helloed on one generation yet",
				(unsigned long long)(api_now_ms() - S.bringup_started_ms));
		waited++;
		sleep(1);
	}
	api_logf("engine_connected attempts=%u elapsed_ms=%llu all_ranks_ready=1", attempt,
		(unsigned long long)(api_now_ms() - S.bringup_started_ms));
	api_seed_submission_ids();
	S.running = 1;
	S.health_missing_rank = UINT32_MAX;
	S.health_degraded_rank = UINT32_MAX;
	S.health_write_alert_rank = UINT32_MAX;
	if (pthread_create(&ApiWorker, 0, api_worker, 0) != 0)
	{
		api_logf("api_exit reason=worker_create_failed");
		_exit(1);
	}
	S.bringup_phase = "ready";
	__atomic_store_n(&S.ready, 1, __ATOMIC_RELEASE);
	api_logf("model_api ready boot_pid=%d sessions=queued-on-engine (single session, worker-driven)", (int)getpid());
	return 0;
}

int main(int argc, char **argv)
{
	const char *dep_path = 0, *root = 0, *port_s = "8080", *quant_arm_path = 0;
	const char *peer_roots[SPARK_MODEL_BATCH_ENGINE_PEER_INDEX_MAX];
	static char peer_index_storage[SPARK_MODEL_BATCH_ENGINE_PEER_INDEX_MAX][4200];
	static const char *peer_indexes[SPARK_MODEL_BATCH_ENGINE_PEER_INDEX_MAX];
	uint32_t peer_count = 0u, peer;
	SparkModelResidentDeployment dep;
	SparkModelBatchEngineConfiguration cfg;
	pthread_t bringup;
	int srv, i;
	for (i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--deployment") && i + 1 < argc)
			dep_path = argv[++i];
		else if (!strcmp(argv[i], "--runtime-root") && i + 1 < argc)
			root = argv[++i];
		else if (!strcmp(argv[i], "--peer-runtime-root") && i + 1 < argc)
		{
			if (peer_count == SPARK_MODEL_BATCH_ENGINE_PEER_INDEX_MAX)
			{
				fprintf(stderr, "model_api: more than %u --peer-runtime-root values\n", SPARK_MODEL_BATCH_ENGINE_PEER_INDEX_MAX);
				return 1;
			}
			peer_roots[peer_count++] = argv[++i];
		}
		else if (!strcmp(argv[i], "--port") && i + 1 < argc)
			port_s = argv[++i];
		else if (!strcmp(argv[i], "--quant-arm") && i + 1 < argc)
			quant_arm_path = argv[++i];
	}
	if (dep_path == 0 || root == 0)
	{
		fprintf(stderr, "usage: %s --deployment PATH --runtime-root PATH [--peer-runtime-root PATH]... [--port N] [--quant-arm ARM_JSON]\n", argv[0]);
		return 1;
	}
	if (quant_arm_path != 0)
	{
		char quant_arm_error[512];
		if (SparkQuantArmLoadFile(quant_arm_path, &QuantArm, quant_arm_error, (uint32_t)sizeof(quant_arm_error)) != SPARK_STATUS_OK)
		{
			fprintf(stderr, "model_api: --quant-arm %s REFUSED: %s\n", quant_arm_path, quant_arm_error);
			return 1;
		}
		HaveQuantArm = 1;
	}
	(void)snprintf(ApiBootTag, sizeof(ApiBootTag), "%d", (int)getpid());
	(void)snprintf(ApiBootId, sizeof(ApiBootId), "%d-%lld", (int)getpid(), (long long)time(0));
	{
		static char resolved_root[4096];
		static char index_path[4200];
		if (realpath(root, resolved_root) == 0)
		{
			fprintf(stderr, "model_api: --runtime-root %s cannot be resolved errno=%d\n", root, errno);
			return 1;
		}
		root = resolved_root;
		(void)snprintf(index_path, sizeof(index_path), "%s/prefix_index.spi", resolved_root);
		S.prefix_index_path = index_path;
		for (peer = 0u; peer < peer_count; peer++)
		{
			char resolved_peer[4096];
			if (realpath(peer_roots[peer], resolved_peer) == 0)
			{
				fprintf(stderr, "model_api: --peer-runtime-root %s cannot be resolved errno=%d\n", peer_roots[peer], errno);
				return 1;
			}
			(void)snprintf(peer_index_storage[peer], sizeof(peer_index_storage[peer]), "%s/prefix_index.spi", resolved_peer);
			peer_indexes[peer] = peer_index_storage[peer];
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
	cfg.prefix_index_path = S.prefix_index_path;
	cfg.peer_prefix_index_paths = peer_count != 0u ? peer_indexes : 0;
	cfg.peer_prefix_index_count = peer_count;
	cfg.request_capacity = 128;
	cfg.max_context_tokens = api_context_limit(&dep);
	S.context_limit = cfg.max_context_tokens;
	api_logf("api_context_limit max_context_tokens=%u deployment_max_sequence_positions=%u", cfg.max_context_tokens, dep.max_sequence_positions);
	cfg.max_prefill_rows_per_submission = dep.runtime_limits.max_input_row_count;
	ApiOutputVocabulary = dep.runtime_limits.output_vocabulary;
	api_logf("api_prefill_rows max_prefill_rows_per_submission=%u deployment_max_input_row_count=%u max_shared_input_row_count=%u", cfg.max_prefill_rows_per_submission, dep.runtime_limits.max_input_row_count, dep.max_shared_input_row_count != 0u ? dep.max_shared_input_row_count : dep.runtime_limits.max_input_row_count);
	cfg.connect_timeout_ms = 30000;
	cfg.maximum_messages_per_rank_per_progress = 8;
	cfg.inflight_budget_ns = dep.request_inflight_budget_ms != 0u ? (uint64_t)dep.request_inflight_budget_ms * UINT64_C(1000000) :
		SPARK_MODEL_BATCH_ENGINE_DEFAULT_INFLIGHT_BUDGET_NS;
	cfg.event_function = api_event;
	cfg.event_context = 0;
	if ( dep.eos_token_count == 0u || dep.eos_token_count > SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT )
	{
		fprintf(stderr,"model_api: required model EOS metadata missing or invalid\n");
		return 1;
	}
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
		TextPromptPrefixCount = dep.tokenizer_text_prompt_prefix_count;
		memcpy(TextPromptPrefix, dep.tokenizer_text_prompt_prefix_ids, (size_t)TextPromptPrefixCount * sizeof(uint32_t));
		TextPromptSuffixCount = dep.tokenizer_text_prompt_suffix_count;
		memcpy(TextPromptSuffix, dep.tokenizer_text_prompt_suffix_ids, (size_t)TextPromptSuffixCount * sizeof(uint32_t));
		fprintf(stderr, "model_api: tokenizer sidecar ready format=%u "
			"vocab=%llu asset=%s text_prompt_prefix_tokens=%u text_prompt_suffix_tokens=%u\n", Sidecar.format,
			(unsigned long long)Sidecar.tokenizer.maximum_token_id + 1ull, asset_path, TextPromptPrefixCount, TextPromptSuffixCount);
	}
	else
		fprintf(stderr, "model_api: no tokenizer in deployment; text prompts "
			"will be rejected (prompt_token_ids accepted)\n");
	if (dep.chat_template.declared != 0u)
	{
		const char *unresolved = 0;
		if (!HaveSidecar ||
			SparkChatTemplateResolveStops(&dep.chat_template, Sidecar.tokenizer.special_tokens,
				Sidecar.tokenizer.special_token_count, ChatStopTokens, &ChatStopTokenCount,
				&unresolved) != SPARK_STATUS_OK)
		{
			fprintf(stderr, "model_api: chat_template declared but %s%s%s; refusing to start\n",
				HaveSidecar ? "stop marker " : "the deployment has no tokenizer",
				unresolved != 0 ? unresolved : "",
				unresolved != 0 ? " is not exactly one special token of the tokenizer, or a declared control marker" : "");
			return 1;
		}
		ChatTemplate = &dep.chat_template;
		fprintf(stderr, "model_api: chat_template declared stop_markers=%u\n", ChatStopTokenCount);
	}
	else
		fprintf(stderr, "model_api: no chat_template in deployment; messages requests "
			"will be rejected (prompt and prompt_token_ids accepted)\n");
	signal(SIGPIPE, SIG_IGN);
	fprintf(stderr, "model_api: open files limit %llu\n", (unsigned long long)api_raise_open_files());
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
	{
		struct sockaddr_in addr;
		int on = 1;
		srv = socket(AF_INET, SOCK_STREAM, 0);
		(void)setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
		memset(&addr, 0, sizeof(addr));
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
		addr.sin_port = htons((uint16_t)strtoul(port_s, 0, 10));
		if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
			listen(srv, SOMAXCONN) != 0)
		{
			fprintf(stderr, "model_api: listen %s: %s\n", port_s, strerror(errno));
			return 1;
		}
	}
	api_logf("model_api listening port=%s boot_pid=%d; engines come up in the background, /health answers 503 until every rank is ready",
		port_s, (int)getpid());
	ApiEngineConfiguration = cfg;
	S.bringup_started_ms = api_now_ms();
	if ( pthread_create(&bringup,0,api_bringup,0) != 0 )
	{
		api_logf("engine bringup thread create failed");
		return 1;
	}
	for (;;)
	{
		int cfd = api_accept(srv);
		pthread_t t;
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
