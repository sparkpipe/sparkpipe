#!/usr/bin/env python3
from pathlib import Path
import subprocess
import tempfile
import json

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
} TestClient;

static void TestWaitQueued(uint64_t id)
{
	ApiRequest *request;
	uint32_t found,attempt;
	struct timespec delay = {0,1000000};
	for (attempt=0u; attempt<1000u; attempt++)
	{
		found = 0u;
		pthread_mutex_lock(&S.queue_mutex);
		for (request=S.queue_head; request != 0; request=request->next)
			if ( request->id == id )
				found = 1u;
		pthread_mutex_unlock(&S.queue_mutex);
		if ( found != 0u )
			return;
		nanosleep(&delay,0);
	}
	assert(0);
}

static TestClient TestEnqueue(uint64_t id)
{
	TestClient client;
	int32_t sockets[2];
	char request[512];
	const char body[] = "{\"prompt_token_ids\":[1,2],\"max_tokens\":1}";
	int32_t length;
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	client.client = sockets[0];
	assert(pthread_create(&client.thread,0,api_connection,(void *)(intptr_t)sockets[1]) == 0);
	length = snprintf(request,sizeof(request),"POST /v1/completions HTTP/1.1\r\nContent-Length: %u\r\n\r\n%s",(uint32_t)strlen(body),body);
	assert(write(client.client,request,length) == length);
	TestWaitQueued(id);
	return(client);
}

static void TestComplete(TestClient *client,uint64_t id,uint32_t token)
{
	SparkModelBatchEvent event = {0};
	char response[1024] = {0},expected[64];
	int32_t bytes,total = 0;
	event.request_id = id;
	event.kind = SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED;
	event.monotonic_ns = id * 100u;
	api_event(0,&event);
	event.kind = SPARK_MODEL_BATCH_EVENT_TOKEN;
	event.monotonic_ns += 50u;
	event.token_id = token;
	api_event(0,&event);
	event.kind = SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED;
	api_event(0,&event);
	while ( (bytes = read(client->client,response + total,sizeof(response) - 1u - total)) > 0 )
		total += bytes;
	assert(bytes == 0);
	assert(pthread_join(client->thread,0) == 0);
	assert(strstr(response,"200 OK") != 0);
	snprintf(expected,sizeof(expected),"\"tokens\":[%u]",token);
	assert(strstr(response,expected) != 0);
	close(client->client);
}

static void *TestConcurrentMeasurements(void *context)
{
	ApiRequest request = {0};
	uint32_t index;
	request.id = (uint64_t)(uintptr_t)context;
	for (index=0u; index<128u; index++)
	{
		api_logf("concurrent status %u",index);
		api_log_request_measurements(&request);
	}
	return(0);
}

int32_t main(void)
{
	TestClient a,b,c;
	pthread_t loggers[8];
	uint32_t index;
	alarm(10);
	assert(pthread_mutex_init(&S.queue_mutex,0) == 0);
	S.running = 1;
	a = TestEnqueue(100001u);
	b = TestEnqueue(100002u);
	TestComplete(&b,100002u,22u);
	c = TestEnqueue(100003u);
	TestComplete(&a,100001u,11u);
	TestComplete(&c,100003u,33u);
	assert(S.queue_head == 0 && S.queue_tail == 0 && S.served == 3u);
	assert(pthread_mutex_destroy(&S.queue_mutex) == 0);
	for (index=0u; index<8u; index++)
		assert(pthread_create(&loggers[index],0,TestConcurrentMeasurements,(void *)(uintptr_t)(700u + index)) == 0);
	for (index=0u; index<8u; index++)
		assert(pthread_join(loggers[index],0) == 0);
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="api-queue-") as directory:
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
        result = subprocess.run([str(binary)], cwd=ROOT, check=True, timeout=15, capture_output=True, text=True)
        records = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{')]
        assert len(records) == 3 + 8 * 128
        for record, token in zip(records[:3], (22, 11, 33)):
            assert record['event'] == 'request_measurements'
            assert record['engine_completed'] == 1 and record['status'] == 0
            assert record['accepted_ns'] == record['request_id'] * 100
            assert record['tokens'] == [[token, record['accepted_ns'] + 50]]
            assert record['cached_prompt_tokens'] == 0
        for identity in range(700,708):
            assert sum(record['request_id'] == identity for record in records[3:]) == 128
    print("PASS real API enqueue, tail completion, enqueue again, remaining replies and empty queue")


if __name__ == "__main__":
    main()
