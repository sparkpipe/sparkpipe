#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#define main SparkModelApiMain
#include "node/model_api.c"
#undef main

typedef struct
{
	int32_t client;
	pthread_t thread;
	ApiRequest *request;
} TestClient;

static ApiRequest *TestFindQueued(uint64_t id)
{
	ApiRequest *request,*found;
	uint32_t attempt;
	struct timespec delay = {0,1000000};
	for (attempt=0u; attempt<2000u; attempt++)
	{
		found = 0;
		pthread_mutex_lock(&S.queue_mutex);
		for (request=S.queue_head; request != 0; request=request->next)
			if ( request->id == id )
				found = request;
		pthread_mutex_unlock(&S.queue_mutex);
		if ( found != 0 )
			return(found);
		nanosleep(&delay,0);
	}
	assert(0);
	return(0);
}

static TestClient TestSend(const char *body)
{
	TestClient client;
	int32_t sockets[2];
	char request[1024];
	int32_t length;
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	client.client = sockets[0];
	client.request = 0;
	assert(pthread_create(&client.thread,0,api_connection,(void *)(intptr_t)sockets[1]) == 0);
	length = snprintf(request,sizeof(request),"POST /v1/completions HTTP/1.1\r\nContent-Length: %u\r\n\r\n%s",(uint32_t)strlen(body),body);
	assert(write(client.client,request,length) == length);
	return(client);
}

static void TestRead(TestClient *client,char *response,size_t capacity)
{
	int32_t bytes,total = 0;
	memset(response,0,capacity);
	while ( (bytes = read(client->client,response + total,capacity - 1u - total)) > 0 )
		total += bytes;
	assert(bytes == 0);
	assert(pthread_join(client->thread,0) == 0);
	close(client->client);
}

static void TestAdmit(ApiRequest *request)
{
	pthread_mutex_lock(&S.queue_mutex);
	request->submitted = 1;
	pthread_mutex_unlock(&S.queue_mutex);
	pthread_mutex_lock(&request->mutex);
	pthread_cond_signal(&request->cond);
	pthread_mutex_unlock(&request->mutex);
}

static void TestEvents(uint64_t id,uint32_t token,uint32_t cached,uint32_t drafts,uint32_t accepted)
{
	SparkModelBatchEvent event = {0};
	event.request_id = id;
	event.cached_prompt_token_count = cached;
	event.kind = SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED;
	event.monotonic_ns = UINT64_C(1000000);
	api_event(0,&event);
	event.first_dispatch_ns = UINT64_C(2000000);
	event.kind = SPARK_MODEL_BATCH_EVENT_TOKEN;
	event.monotonic_ns = UINT64_C(7250000);
	event.token_id = token;
	api_event(0,&event);
	event.kind = SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED;
	event.draft_token_count = drafts;
	event.accepted_draft_token_count = accepted;
	event.published_token_count = 4u;
	api_event(0,&event);
}

int32_t main(void)
{
	TestClient client;
	ApiRequest *request;
	char response[4096];
	uint64_t id = 100000u;
	alarm(30);
	assert(pthread_mutex_init(&S.queue_mutex,0) == 0);
	assert(pipe(S.wake_fds) == 0);
	assert(fcntl(S.wake_fds[0],F_SETFL,O_NONBLOCK) == 0);
	assert(fcntl(S.wake_fds[1],F_SETFL,O_NONBLOCK) == 0);
	S.running = 1;
	S.ready = 1;
	S.context_limit = 4096u;

	client = TestSend("{\"prompt_token_ids\":[1,2,3,4],\"commit_only\":true}");
	request = TestFindQueued(++id);
	assert(request->commit_only == 1u && request->max_tokens == 1u);
	TestEvents(id,77u,2u,0u,0u);
	TestRead(&client,response,sizeof(response));
	assert(strstr(response,"200 OK") != 0);
	assert(strstr(response,"\"finish_reason\":\"commit\"") != 0);
	assert(strstr(response,"\"tokens\":[]") != 0 && strstr(response,"77") == 0);
	assert(strstr(response,"\"prompt_tokens\":4,\"completion_tokens\":0,\"total_tokens\":4,\"prompt_tokens_details\":{\"cached_tokens\":2}") != 0);
	assert(strstr(response,"\"published_tokens\":4,\"prefill_ms\":5.250") != 0);
	fprintf(stdout,"commit_only reply: %s\n",strstr(response,"{"));

	client = TestSend("{\"prompt_token_ids\":[1,2,3,4,5],\"commit_only\":true,\"stream\":true}");
	request = TestFindQueued(++id);
	assert(request->commit_only == 1u && request->max_tokens == 1u && request->stream == 1u);
	TestAdmit(request);
	TestEvents(id,88u,4u,0u,0u);
	TestRead(&client,response,sizeof(response));
	assert(strstr(response,"text/event-stream") != 0);
	assert(strstr(response,"\"finish_reason\":\"commit\"") != 0 && strstr(response,"data: [DONE]") != 0);
	assert(strstr(response,"\"tokens\":[]") != 0 && strstr(response,"88") == 0);
	assert(strstr(response,"\"completion_tokens\":0") != 0 && strstr(response,"\"cached_tokens\":4") != 0);
	assert(strstr(strstr(response,"data: ") + 6,"data: {") == 0);

	client = TestSend("{\"prompt_token_ids\":[1,2],\"commit_only\":true,\"max_tokens\":8}");
	TestRead(&client,response,sizeof(response));
	assert(strstr(response,"400") != 0 && strstr(response,"\"code\":\"commit_only_conflict\"") != 0);
	client = TestSend("{\"prompt_token_ids\":[1,2],\"commit_only\":true,\"logprobs\":2}");
	TestRead(&client,response,sizeof(response));
	assert(strstr(response,"400") != 0 && strstr(response,"\"code\":\"commit_only_conflict\"") != 0);
	client = TestSend("{\"prompt_token_ids\":[1,2],\"commit_only\":1}");
	TestRead(&client,response,sizeof(response));
	assert(strstr(response,"400") != 0 && strstr(response,"commit_only must be booleans") != 0);

	{
		const char body[] = "{\"pins\":[]}";
		char put[256];
		int32_t sockets[2],length;
		assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
		client.client = sockets[0];
		assert(pthread_create(&client.thread,0,api_connection,(void *)(intptr_t)sockets[1]) == 0);
		length = snprintf(put,sizeof(put),"PUT /v1/cache/pins HTTP/1.1\r\nContent-Length: %u\r\n\r\n%s",(uint32_t)strlen(body),body);
		assert(write(client.client,put,length) == length);
		TestRead(&client,response,sizeof(response));
		assert(strstr(response,"HTTP/1.1 501") != 0 && strstr(response,"\"code\":\"pins_unsupported\"") != 0);
	}
	client = TestSend("{\"prompt_token_ids\":[1,2,3],\"max_tokens\":4}");
	request = TestFindQueued(++id);
	assert(request->commit_only == 0u && request->max_tokens == 4u);
	TestEvents(id,99u,0u,6u,4u);
	TestRead(&client,response,sizeof(response));
	assert(strstr(response,"200 OK") != 0 && strstr(response,"\"tokens\":[99]") != 0 && strstr(response,"\"finish_reason\":\"stop\"") != 0);
	assert(strstr(response,"\"completion_tokens\":1") != 0 && strstr(response,"\"draft_tokens\":6,\"accepted_draft_tokens\":4") != 0);
	fprintf(stdout,"generating reply: %s\n",strstr(response,"{"));
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="api-cache-controls-") as directory:
        source = Path(directory) / "probe.c"
        binary = Path(directory) / "probe"
        source.write_text(HARNESS)
        subprocess.run([
            "cc", "-std=c11", "-O1", "-D_GNU_SOURCE", "-pthread",
            "-I.", "-Iinclude", "-Isrc", "-Imodel-families/common/include",
            str(source), "build/libsparkpipe_runtime.a",
            "build/libsparkpipe_model_common.a", "build/libsparkpipe_core.a",
            "-ldl", "-lpthread", "-o", str(binary),
        ], cwd=ROOT, check=True)
        result = subprocess.run([str(binary)], cwd=ROOT, timeout=40, capture_output=True, text=True)
        if result.returncode:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            result.check_returncode()
        records = [line for line in result.stderr.splitlines() if '"event":"request_measurements"' in line]
        assert len(records) == 3, records
        assert all('"finish_reason":"commit"' in line and '"tokens":[]' in line for line in records[:2])
    print("PASS commit_only commits the prompt with no token in the reply, stream or record, finish_reason commit, completion_tokens 0 and "
          "the engine's prefill_ms; commit_only with max_tokens or logprobs, or not a boolean, is a named 400; every completion's usage "
          "carries the request's own draft_tokens and accepted_draft_tokens; a pin set is refused with 501 pins_unsupported until the "
          "ranks can hold pins")


if __name__ == "__main__":
    main()
