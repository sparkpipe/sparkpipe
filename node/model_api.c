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
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_tokenizer_sidecar.h"

#define API_MAX_BODY		(8u * 1024u * 1024u)
#define API_MAX_PROMPT_TOKENS	(260000u)
#define API_MAX_STOP_TOKENS 16
#define API_MAX_OUTPUT_TOKENS	(8192u)
#define API_TOKEN_BUF_BYTES	(64u * 1024u)

#define API_MAX_INFLIGHT 16u

#define API_DEFAULT_MODEL_ID "sparkpipe-model"

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
	volatile uint32_t output_token_count;
	volatile int done;
	volatile int submitted;
	volatile int inflight;
	volatile int orphaned;
	volatile uint32_t status;
	SparkModelBatchRequestHandle handle;
	uint32_t *stop_tokens;
	uint32_t stop_token_count;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	struct ApiRequest *next;
} ApiRequest;

typedef struct ApiState
{
	SparkModelBatchEngine *engine;
	pthread_mutex_t queue_mutex;
	ApiRequest *queue_head;
	ApiRequest *queue_tail;
	volatile int running;
	uint64_t next_id;
	uint64_t served;
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
	fprintf(stderr, "model_api[%s] ", ApiBootTag);
	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);
	fputc('\n', stderr);
	fflush(stderr);
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
				r->output_token_ids[r->output_token_count++] = ev->token_id;
			}
		}
		if (ev->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED ||
			ev->kind == SPARK_MODEL_BATCH_EVENT_ERROR)
		{
			pthread_mutex_lock(&r->mutex);
			r->status = ev->status;
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

static void *api_worker(void *arg)
{
	SparkModelResidentClientPollDescriptor fds[4];
	struct pollfd pfds[4];
	(void)arg;
	while (S.running)
	{
		{
			ApiRequest *pending[API_MAX_INFLIGHT];
			uint32_t pending_count = 0;
			ApiRequest *r;
			uint32_t i;
			pthread_mutex_lock(&S.queue_mutex);
			{
				ApiRequest **pp = &S.queue_head;
				while (*pp != 0)
				{
					ApiRequest *victim = *pp;
					if (victim->orphaned && !victim->inflight)
					{
						*pp = victim->next;
						if (S.queue_tail == victim)
							S.queue_tail = 0;
						pthread_mutex_unlock(&S.queue_mutex);
						pthread_mutex_destroy(&victim->mutex);
						pthread_cond_destroy(&victim->cond);
						free(victim->stop_tokens);
						free(victim->prompt_tokens);
						free(victim);
						pthread_mutex_lock(&S.queue_mutex);
						continue;
					}
					pp = &victim->next;
				}
			}
			for (r = S.queue_head; r != 0 && pending_count < API_MAX_INFLIGHT; r = r->next)
			{
				if (!r->done && !r->submitted && !r->inflight)
				{
					r->inflight = 1;
					pending[pending_count++] = r;
				}
			}
			pthread_mutex_unlock(&S.queue_mutex);
			for (i = 0; i < pending_count; i++)
			{
				SparkModelBatchSubmitRequest sub;
				SparkModelBatchRequestHandle h;
				SparkStatus st;
				r = pending[i];
				if (r->done)
				{
					pthread_mutex_lock(&S.queue_mutex);
					r->inflight = 0;
					pthread_mutex_unlock(&S.queue_mutex);
					continue;
				}
				memset(&sub, 0, sizeof(sub));
				sub.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
				sub.descriptor_bytes = (uint32_t)sizeof(sub);
				sub.request_id = r->id;
				sub.sequence_id = r->id;
				sub.output_token_budget = r->max_tokens;
				sub.prompt_token_ids = r->prompt_tokens;
				sub.prompt_token_count = r->prompt_count;
				(void)SparkModelBatchEngineReopenAdmission(S.engine);
				st = SparkModelBatchEngineSubmit(S.engine, &sub, &h);
				pthread_mutex_lock(&S.queue_mutex);
				if (st == SPARK_STATUS_OK)
				{
					r->submitted = 1;
					r->handle = h;
				}
				else
				{
					r->status = (uint32_t)st;
					r->done = 1;
					pthread_mutex_lock(&r->mutex);
					pthread_cond_signal(&r->cond);
					pthread_mutex_unlock(&r->mutex);
				}
				pthread_mutex_unlock(&S.queue_mutex);
				api_orphan_cancel_after_submit(r);
				pthread_mutex_lock(&S.queue_mutex);
				r->inflight = 0;
				pthread_mutex_unlock(&S.queue_mutex);
			}
		}
		if (S.queue_head != 0)
		{
			(void)SparkModelBatchEngineProgress(S.engine, 4u);
			{
				uint32_t n = 0;
				if (SparkModelBatchEngineGetPollDescriptors(
					S.engine, fds, 4u, &n) == SPARK_STATUS_OK && n > 0)
				{
					uint32_t i;
					for (i = 0; i < n; i++)
					{
						pfds[i].fd = fds[i].fd;
						pfds[i].events = 0;
						pfds[i].revents = 0;
						if (fds[i].events & 1u)
							pfds[i].events |= POLLIN;
						if (fds[i].events & 2u)
							pfds[i].events |= POLLOUT;
					}
					(void)poll(pfds, (nfds_t)n, 10);
				}
				else
					usleep(1000);
			}
		}
		else
			usleep(5000);
	}
	return 0;
}


static void send_all(int fd, const char *data, size_t len)
{
	size_t off = 0;
	while (off < len)
	{
		ssize_t n = send(fd, data + off, len - off, MSG_NOSIGNAL);
		if (n <= 0)
			return;
		off += (size_t)n;
	}
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
	while (total < buf_cap - 1)
	{
		n = recv(fd, buf + total, buf_cap - 1 - total, 0);
		if (n <= 0)
			return 0;
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

static void handle_completion(int fd, char *body, uint32_t body_len,
	int chat_format)
{
	SparkJsonDocument doc;
	int32_t root, mt;
	uint32_t *request_stops = 0;
	uint32_t request_stop_count = 0;
	memset(&doc,0,sizeof(doc));
	uint32_t *prompt = 0, prompt_len = 0, max_tokens = 32;
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
	while (!req->done && S.running)
	{
		struct pollfd disconnect_probe;
		int poll_status;
		disconnect_probe.fd = fd;
		disconnect_probe.events = POLLIN;
		poll_status = poll(&disconnect_probe, 1, 250);
		if (poll_status > 0 && (disconnect_probe.revents & (POLLHUP | POLLERR)) != 0)
			break;
		if (poll_status > 0 && (disconnect_probe.revents & POLLIN) != 0)
		{
			char probe;
			ssize_t received = recv(fd, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
			if (received == 0)
				break;
		}
		if (!req->done)
		{
			pthread_mutex_lock(&req->mutex);
			if (!req->done)
				{
					struct timespec api_wait_until;
					clock_gettime(CLOCK_REALTIME, &api_wait_until);
					api_wait_until.tv_nsec += 250000000L;
					if (api_wait_until.tv_nsec >= 1000000000L)
					{
						api_wait_until.tv_sec += 1u;
						api_wait_until.tv_nsec -= 1000000000L;
					}
					pthread_cond_timedwait(&req->cond, &req->mutex, &api_wait_until);
				}
			pthread_mutex_unlock(&req->mutex);
		}
	}
	if (!req->done && S.running)
	{
		uint32_t cancel_submitted;
		SparkModelBatchRequestHandle cancel_handle;
		pthread_mutex_lock(&req->mutex);
		cancel_submitted = req->submitted;
		cancel_handle = cancel_submitted ? req->handle : 0;
		req->orphaned = 1;
		req->done = 1;
		pthread_mutex_unlock(&req->mutex);
		if (cancel_submitted && cancel_handle != 0)
			(void)SparkModelBatchEngineCancel(S.engine, cancel_handle);
	}
	if (req->status == 0 && HaveSidecar)
	{
		uint32_t decode_stops[API_MAX_STOP_TOKENS +
			SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT];
		uint32_t decode_stop_count = 0;
		uint32_t text_capacity;
		char *text_buffer = 0;
		char *escaped = 0;
		size_t escaped_cap;
		size_t escaped_len = 0;
		size_t response_len = 0;
		size_t response_cap;
		char *resp_text = 0;
		uint32_t text_bytes = 0;
		SparkStatus decode_status;
		memcpy(decode_stops, EngineStopTokens,
			(size_t)EngineStopTokenCount * sizeof(uint32_t));
		decode_stop_count = EngineStopTokenCount;
		if (req->stop_token_count <= API_MAX_STOP_TOKENS)
		{
			memcpy(decode_stops + decode_stop_count, req->stop_tokens,
				(size_t)req->stop_token_count * sizeof(uint32_t));
			decode_stop_count += req->stop_token_count;
		}
		text_capacity = req->output_token_count * Sidecar.maximum_token_text_bytes + 1u;
		text_buffer = malloc((size_t)text_capacity);
		escaped_cap = (size_t)text_capacity * 6u + 8u;
		escaped = malloc(escaped_cap);
		response_cap = (size_t)req->tokens_json_len + escaped_cap + 160u;
		resp_text = malloc(response_cap);
		if (text_buffer != 0 && escaped != 0 && resp_text != 0)
			decode_status = SparkTokenizerSidecarDecodeText(&Sidecar,
				req->output_token_ids, req->output_token_count,
				decode_stops, decode_stop_count, 0u,
				text_buffer, text_capacity, &text_bytes);
		else
			decode_status = SPARK_STATUS_INTERNAL_ERROR;
		if (decode_status == SPARK_STATUS_OK)
		{
			if (!append_json_escaped(escaped, escaped_cap, &escaped_len,
					text_buffer, text_bytes))
				decode_status = SPARK_STATUS_INTERNAL_ERROR;
		}
		if (decode_status == SPARK_STATUS_OK)
		{
			if (chat_format)
			{
				memcpy(resp_text, "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"", 91u);
				response_len = 91u;
			}
			else
			{
				memcpy(resp_text, "{\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"", 58u);
				response_len = 58u;
			}
			memcpy(resp_text + response_len, escaped, escaped_len);
			response_len += escaped_len;
			if (chat_format)
			{
				memcpy(resp_text + response_len, "\"}}],\"tokens\":[", 15u);
				response_len += 15u;
			}
			else
			{
				memcpy(resp_text + response_len, "\"}],\"tokens\":[", 14u);
				response_len += 14u;
			}
			memcpy(resp_text + response_len, req->tokens_json, req->tokens_json_len);
			response_len += req->tokens_json_len;
			memcpy(resp_text + response_len, "],\"status\":0}", 13u);
			response_len += 13u;
			resp_text[response_len] = '\0';
			send_response(fd, 200, resp_text);
			api_logf("request_done fd=%d id=%llu status=%u output_tokens=%u ms=%llu",
				fd, (unsigned long long)req->id, (unsigned)req->status,
				req->output_token_count,
				(unsigned long long)(api_now_ms() - req->started_ms));
		}
		else
		{
			char err[160];
			(void)snprintf(err, sizeof(err),
				"{\"error\":{\"message\":\"tokenizer sidecar failed to decode "
				"the completion (status %u)\",\"type\":\"model_error\","
				"\"code\":%u}}", (unsigned)decode_status, (unsigned)decode_status);
			send_response(fd, 500, err);
		}
		free(text_buffer);
		free(escaped);
		free(resp_text);
	}
	else if (req->status == 0)
	{
		char *resp = malloc(req->tokens_json_len + 128);
		if (resp != 0)
		{
			(void)snprintf(resp, req->tokens_json_len + 128,
				"{\"object\":\"text_completion\",\"tokens\":[%s],\"status\":0}",
				req->tokens_json);
				send_response(fd, 200, resp);
				api_logf("request_done fd=%d id=%llu status=%u output_tokens=%u ms=%llu",
					fd, (unsigned long long)req->id, (unsigned)req->status,
					req->output_token_count,
					(unsigned long long)(api_now_ms() - req->started_ms));
				free(resp);
		}
		else
			send_response(fd, 500, "{\"error\":\"oom\"}");
	}
	else
	{
		char err[128];
		(void)snprintf(err, sizeof(err),
			"{\"error\":{\"message\":\"model status %u\","
			"\"type\":\"model_error\",\"code\":%u}}",
			req->status, req->status);
		send_response(fd, 500, err);
	}
	pthread_mutex_lock(&S.queue_mutex);
	if (req->inflight)
	{
		req->orphaned = 1;
		pthread_mutex_unlock(&S.queue_mutex);
		return;
	}
	{
		ApiRequest **pp = &S.queue_head;
		while (*pp != 0)
		{
			if (*pp == req)
			{
				*pp = req->next;
				if (S.queue_tail == req)
					S.queue_tail = 0;
				break;
			}
			pp = &(*pp)->next;
		}
	}
	pthread_mutex_unlock(&S.queue_mutex);
	pthread_mutex_destroy(&req->mutex);
	pthread_cond_destroy(&req->cond);
	free(req->stop_tokens);
	free(req->prompt_tokens);
	free(req);
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
	cfg.connect_timeout_ms = 30000;
	cfg.maximum_messages_per_rank_per_progress = 8;
	cfg.event_function = api_event;
	cfg.event_context = 0;
	{
		const char *eos_env = getenv("SPARK_EOS_TOKEN_IDS");
		if ( eos_env != 0 && eos_env[0] != '\0' )
		{
			unsigned long value;
			char *cursor = (char *)eos_env, *next;
			while ( *cursor != '\0' &&
				cfg.stop_token_count < SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT )
			{
				value = strtoul(cursor,&next,10);
				if ( next == cursor )
					break;
				cfg.stop_token_ids[cfg.stop_token_count++] = (uint32_t)value;
				cursor = ( *next == ',' ) ? next + 1 : next;
				if ( *cursor == '\0' ) break;
			}
			fprintf(stderr,"model_api: %u EOS token(s) from env\n",cfg.stop_token_count);
		}
	}
	EngineStopTokenCount = cfg.stop_token_count;
	memcpy(EngineStopTokens, cfg.stop_token_ids,
		sizeof(uint32_t) * (size_t)cfg.stop_token_count);
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
		HaveSidecar = 1;
		fprintf(stderr, "model_api: tokenizer sidecar ready format=%u "
			"vocab=%u asset=%s\n", Sidecar.format,
			Sidecar.tokenizer.vocabulary_count, asset_path);
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
		api_logf("engine_connected attempts=%u elapsed_ms=%llu", connect_attempt,
			(unsigned long long)(api_now_ms() - connect_started_ms));
	}
	signal(SIGPIPE, SIG_IGN);
	signal(SIGTERM, api_term_signal);
	signal(SIGINT, api_term_signal);
	pthread_mutex_init(&S.queue_mutex, 0);
	S.running = 1;
	pthread_create(&worker, 0, api_worker, 0);
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
