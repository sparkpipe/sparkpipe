#!/usr/bin/env python3
from pathlib import Path
import subprocess
import tempfile
import json
import sys

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

static void TestQueueOrder(uint64_t first,const uint32_t *done,uint32_t count)
{
	ApiRequest *request,*tail = 0;
	uint32_t index;
	pthread_mutex_lock(&S.queue_mutex);
	request = S.queue_head;
	for (index=0u; index<count; index++)
	{
		if ( done[index] != 0u )
			continue;
		assert(request != 0 && request->id == first + index);
		tail = request;
		request = request->next;
	}
	assert(request == 0 && S.queue_tail == tail);
	pthread_mutex_unlock(&S.queue_mutex);
}

int32_t main(int argc,char **argv)
{
	TestClient a,b,c;
	pthread_t loggers[8];
	uint32_t index,round,seed = 1u;
	uint64_t next_id = 100004u;
	if ( argc == 2 )
	{
		char *end;
		unsigned long value = strtoul(argv[1],&end,10);
		assert(end != argv[1] && *end == '\0' && value <= UINT32_MAX);
		seed = (uint32_t)value;
	}
	assert(argc == 1 || argc == 2);
	fprintf(stderr,"api queue seed=%u rounds=32 width=7\n",seed);
	alarm(30);
	assert(pthread_mutex_init(&S.queue_mutex,0) == 0);
	assert(pipe(S.wake_fds) == 0);
	assert(fcntl(S.wake_fds[0],F_SETFL,O_NONBLOCK) == 0);
	assert(fcntl(S.wake_fds[1],F_SETFL,O_NONBLOCK) == 0);
	S.running = 1;
	a = TestEnqueue(100001u);
	b = TestEnqueue(100002u);
	TestComplete(&b,100002u,22u);
	c = TestEnqueue(100003u);
	TestComplete(&a,100001u,11u);
	TestComplete(&c,100003u,33u);
	assert(S.queue_head == 0 && S.queue_tail == 0 && S.served == 3u);
	for (round=0u; round<32u; round++)
	{
		TestClient clients[7];
		uint32_t done[7] = {0};
		uint64_t first = next_id;
		for (index=0u; index<7u; index++)
			clients[index] = TestEnqueue(next_id++);
		TestQueueOrder(first,done,7u);
		for (index=0u; index<7u; index++)
		{
			uint32_t victim;
			seed = seed * 1664525u + 1013904223u;
			victim = (seed >> 16u) % 7u;
			while ( done[victim] != 0u )
				victim = (victim + 1u) % 7u;
			TestComplete(&clients[victim],first + victim,(uint32_t)(first + victim));
			done[victim] = 1u;
			TestQueueOrder(first,done,7u);
		}
	}
	assert(S.served == 227u);
	{
		uint8_t bytes[1024] = {0};
		while ( write(S.wake_fds[1],bytes,sizeof(bytes)) > 0 )
			;
		assert(errno == EAGAIN || errno == EWOULDBLOCK);
		api_wake_worker();
		api_drain_worker_wake();
		assert(read(S.wake_fds[0],bytes,sizeof(bytes)) < 0 &&
			(errno == EAGAIN || errno == EWOULDBLOCK));
	}
	assert(close(S.wake_fds[0]) == 0);
	assert(close(S.wake_fds[1]) == 0);
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
        result = subprocess.run([str(binary), *sys.argv[1:]], cwd=ROOT, timeout=40, capture_output=True, text=True)
        if result.returncode:
            sys.stderr.write(result.stderr)
            result.check_returncode()
        records = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{')]
        assert len(records) == 227 + 8 * 128
        for record, token in zip(records[:3], (22, 11, 33)):
            assert record['event'] == 'request_measurements'
            assert record['engine_completed'] == 1 and record['status'] == 0
            assert record['accepted_ns'] == record['request_id'] * 100
            assert record['tokens'] == [[token, record['accepted_ns'] + 50]]
            assert record['cached_prompt_tokens'] == 0
        queued = records[3:227]
        assert {record['request_id'] for record in queued} == set(range(100004,100228))
        for record in queued:
            identity = record['request_id']
            assert record['engine_completed'] == 1 and record['status'] == 0
            assert record['tokens'] == [[identity, identity * 100 + 50]]
        for identity in range(700,708):
            assert sum(record['request_id'] == identity for record in records[227:]) == 128
    print("PASS 227 real API replies; seeded completion permutations preserve every queue owner, token and terminal")


if __name__ == "__main__":
    main()
