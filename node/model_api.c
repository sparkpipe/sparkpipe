/* model_api — the production API entry point.
 *
 * Architecture: one persistent engine session driven by a single worker
 * thread; HTTP connection threads enqueue requests and wait on
 * per-request condition variables. The daemon is single-client; the
 * engine worker IS that client.
 *
 *   HTTP callers (many)
 *          |
 *   [accept → parse → enqueue → wait on condvar]
 *          |
 *   [engine worker thread]
 *     loop: dequeue → Submit → CloseAdmission
 *           Progress + poll → events fire → signal waiting threads
 *
 * Endpoints:
 *   GET  /health           → {"status":"ok","served":N}
 *   POST /v1/completions   → {"prompt_token_ids":[...],"max_tokens":N}
 *        → {"object":"text_completion","tokens":[...],"status":0}
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
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#define API_MAX_BODY		(8u * 1024u * 1024u)
#define API_MAX_PROMPT_TOKENS	(260000u)
#define API_MAX_OUTPUT_TOKENS	(8192u)
#define API_TOKEN_BUF_BYTES	(64u * 1024u)

typedef struct ApiRequest
{
	uint64_t id;
	uint32_t *prompt_tokens;
	uint32_t prompt_count;
	uint32_t max_tokens;
	char tokens_json[API_TOKEN_BUF_BYTES];
	volatile uint32_t tokens_json_len;
	volatile int done;
	volatile int submitted;
	volatile uint32_t status;
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

/* ===================== worker: engine driving ===================== */

static void api_event(void *ctx, const SparkModelBatchEvent *ev)
{
	ApiRequest *r;
	(void)ctx;
	if (ev == 0)
		return;
	/* The walk must hold the queue lock: a completed request is unlinked
	 * and freed by its connection thread under the same lock, and an
	 * unlocked walk could chase a node through freed memory. */
	pthread_mutex_lock(&S.queue_mutex);
	for (r = S.queue_head; r != 0; r = r->next)
	{
		if (r->id != ev->request_id)
			continue;
		if (ev->kind == SPARK_MODEL_BATCH_EVENT_TOKEN &&
			r->tokens_json_len + 12u < sizeof(r->tokens_json))
			r->tokens_json_len += (uint32_t)snprintf(
				r->tokens_json + r->tokens_json_len,
				sizeof(r->tokens_json) - r->tokens_json_len,
				"%s%u", r->tokens_json_len ? "," : "",
				(unsigned)ev->token_id);
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

static void *api_worker(void *arg)
{
	SparkModelResidentClientPollDescriptor fds[4];
	struct pollfd pfds[4];
	(void)arg;
	while (S.running)
	{
		/* submit every waiting request, oldest first: the engine batches
		 * concurrent submissions into shared prefill/decode rounds, so
		 * holding back everything behind the head serialized the API and
		 * starved batching (audit: "strictly serial queue head"). */
		pthread_mutex_lock(&S.queue_mutex);
		{
			ApiRequest *r;
			for (r = S.queue_head; r != 0; r = r->next)
			{
			if (!r->done && !r->submitted)
			{
			SparkModelBatchSubmitRequest sub;
			SparkModelBatchRequestHandle h;
			SparkStatus st;
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
			if (st == SPARK_STATUS_OK)
				r->submitted = 1;
			else
			{
				r->status = (uint32_t)st;
				r->done = 1;
				pthread_cond_signal(&r->cond);
			}
			}
			}
		}
		pthread_mutex_unlock(&S.queue_mutex);
		/* drive the engine only when we have submitted work — calling
		 * Progress/poll on a freshly-connected idle engine segfaults */
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

/* ===================== HTTP layer ===================== */

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
	/* heap buffer: a static __thread of 8MB overflows TLS when the first
	 * connection thread is created (8MB TLS + 8MB stack = crash) */
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
		char *cl = strstr(buf, "Content-Length:");
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
		/* bodyless request (GET): succeed with an empty body */
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
	for (i = 0; i < count; i++)
	{
		uint32_t v;
		if (SparkJsonGetUInt32(doc,
			SparkJsonGetArrayElement(doc, m, (int32_t)i), &v)
			!= SPARK_STATUS_OK || v > 260000)
		{
			free(tokens);
			return 0;
		}
		tokens[i] = v;
	}
	*out = tokens;
	return count;
}

static void handle_completion(int fd, char *body, uint32_t body_len)
{
	SparkJsonDocument doc;
	int32_t root, mt;
	/* zero-init: the JSON parser's internal Destroy can free uninitialized
	 * pointers if the document is stack garbage (valgrind: invalid free
	 * from SparkJsonDocumentDestroy json.c:446) */
	memset(&doc,0,sizeof(doc));
	uint32_t *prompt = 0, prompt_len = 0, max_tokens = 32;
	ApiRequest *req;
	char *resp;
	if (SparkJsonParseText(body, body_len, &doc) != SPARK_STATUS_OK)
	{
		send_response(fd, 400, "{\"error\":\"invalid json\"}");
		return;
	}
	root = SparkJsonGetRootToken(&doc);
	if (root >= 0)
		prompt_len = parse_token_array(&doc, root, "prompt_token_ids", &prompt);
	mt = SparkJsonFindObjectMember(&doc, root, "max_tokens");
	if (mt >= 0)
	{
		uint32_t v;
		if (SparkJsonGetUInt32(&doc, mt, &v) == SPARK_STATUS_OK && v > 0)
			max_tokens = v > API_MAX_OUTPUT_TOKENS ? API_MAX_OUTPUT_TOKENS : v;
	}
	SparkJsonDocumentDestroy(&doc);
	if (prompt_len == 0)
	{
		send_response(fd, 400, "{\"error\":\"prompt_token_ids required\"}");
		return;
	}
	if ((uint64_t)prompt_len + max_tokens > API_MAX_PROMPT_TOKENS + API_MAX_OUTPUT_TOKENS)
	{
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
	pthread_mutex_init(&req->mutex, 0);
	pthread_cond_init(&req->cond, 0);
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
	pthread_mutex_lock(&req->mutex);
	while (!req->done && S.running)
		pthread_cond_wait(&req->cond, &req->mutex);
	pthread_mutex_unlock(&req->mutex);
	resp = malloc(req->tokens_json_len + 128);
	if (resp != 0)
	{
		(void)snprintf(resp, req->tokens_json_len + 128,
			"{\"object\":\"text_completion\",\"tokens\":[%s],\"status\":%u}",
			req->tokens_json, req->status);
		send_response(fd, req->status == 0 ? 200 : 500, resp);
		free(resp);
	}
	else
		send_response(fd, 500, "{\"error\":\"oom\"}");
	/* dequeue */
	pthread_mutex_lock(&S.queue_mutex);
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
	/* The request is unlinked under the queue lock and the worker's event
	 * walk holds the same lock, so nothing can reach it anymore: the
	 * waiter already woke (done was signaled under req->mutex) and the
	 * worker's last touch of this request preceded that signal. */
	pthread_mutex_destroy(&req->mutex);
	pthread_cond_destroy(&req->cond);
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
		char b[96];
		(void)snprintf(b, sizeof(b), "{\"status\":\"ok\",\"served\":%llu}",
			(unsigned long long)S.served);
		send_response(fd, 200, b);
	}
	else if (strcmp(method, "POST") == 0 &&
		(strcmp(path, "/v1/completions") == 0 ||
		 strcmp(path, "/v1/chat/completions") == 0))
		handle_completion(fd, body, body_len);
	else
		send_response(fd, 404, "{\"error\":\"not found\"}");
	/* the receive buffer (up to API_MAX_BODY + 8K) is the connection's to
	 * release; body itself is an interior pointer into it */
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
	cfg.max_prefill_rows_per_submission = 16;
	cfg.connect_timeout_ms = 30000;
	cfg.maximum_messages_per_rank_per_progress = 8;
	cfg.event_function = api_event;
	cfg.event_context = 0;
	if (SparkModelBatchEngineConnect(&cfg, &S.engine) != SPARK_STATUS_OK)
	{
		fprintf(stderr, "model_api: engine connect failed (daemon up?)\n");
		return 1;
	}
	signal(SIGPIPE, SIG_IGN);
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
	fprintf(stderr, "model_api ready port=%s (single session, worker-driven)\n", port_s);
	for (;;)
	{
		int cfd = accept(srv, 0, 0);
		pthread_t t;
		if (cfd < 0)
			continue;
		if (pthread_create(&t, 0, api_connection, (void *)(intptr_t)cfd) == 0)
			pthread_detach(t);
		else
			close(cfd);
	}
	S.running = 0;
	(void)SparkModelBatchEngineDestroy(S.engine);
	return 0;
}
