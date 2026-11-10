#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = 6
BOUND_MS = 60
HARNESS = r'''
#include <assert.h>
#define main SparkModelApiMain
#include "node/model_api.c"
#undef main

static ApiRequest *TestFind(uint64_t id)
{
	ApiRequest *request,*found = 0;
	pthread_mutex_lock(&S.queue_mutex);
	for (request=S.queue_head; request != 0; request=request->next)
		if ( request->id == id )
			found = request;
	pthread_mutex_unlock(&S.queue_mutex);
	return(found);
}

static ApiRequest *TestWaitQueued(uint64_t id)
{
	struct timespec delay = {0,1000000};
	uint32_t attempt;
	for (attempt=0u; attempt<2000u; attempt++)
	{
		ApiRequest *request = TestFind(id);
		if ( request != 0 )
			return(request);
		nanosleep(&delay,0);
	}
	assert(0);
	return(0);
}

static uint64_t TestNowUs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000u + (uint64_t)now.tv_nsec / 1000u);
}

static pthread_t TestEnqueue(int32_t *client,const char *body)
{
	pthread_t thread;
	int32_t sockets[2],length;
	char request[512];
	assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
	*client = sockets[0];
	assert(pthread_create(&thread,0,api_connection,(void *)(intptr_t)sockets[1]) == 0);
	length = snprintf(request,sizeof(request),"POST /v1/completions HTTP/1.1\r\nContent-Length: %u\r\n\r\n%s",(uint32_t)strlen(body),body);
	assert(write(*client,request,length) == length);
	return(thread);
}

int32_t main(int argc,char **argv)
{
	uint32_t round,rounds;
	uint64_t id = 100000u;
	assert(argc == 2);
	rounds = (uint32_t)strtoul(argv[1],0,10);
	alarm(60);
	signal(SIGPIPE,SIG_IGN);
	assert(pthread_mutex_init(&S.queue_mutex,0) == 0);
	assert(pipe(S.wake_fds) == 0);
	assert(fcntl(S.wake_fds[0],F_SETFL,O_NONBLOCK) == 0);
	assert(fcntl(S.wake_fds[1],F_SETFL,O_NONBLOCK) == 0);
	S.running = 1;
	S.ready = 1;
	S.context_limit = 262144u;
	ApiOutputVocabulary = 262144u;
	{
		int32_t client;
		pthread_t thread = TestEnqueue(&client,"{\"prompt_token_ids\":[1,2],\"max_tokens\":1}");
		ApiRequest *request = TestWaitQueued(++id);
		printf("default_priority %u\n",request->priority);
		close(client);
		assert(pthread_join(thread,0) == 0);
		thread = TestEnqueue(&client,"{\"prompt_token_ids\":[1,2],\"max_tokens\":1,\"priority\":1}");
		request = TestWaitQueued(++id);
		printf("explicit_priority %u\n",request->priority);
		close(client);
		assert(pthread_join(thread,0) == 0);
	}
	for (round=0u; round<rounds; round++)
	{
		int32_t client;
		struct timespec settle = {0,(long)(round % 3u) * 3000000L};
		pthread_t thread = TestEnqueue(&client,"{\"prompt_token_ids\":[1,2,3],\"max_tokens\":8,\"stream\":true}");
		ApiRequest *request = TestWaitQueued(++id);
		uint64_t closed;
		nanosleep(&settle,0);
		closed = TestNowUs();
		close(client);
		for (;;)
		{
			uint32_t gone;
			pthread_mutex_lock(&S.queue_mutex);
			gone = request->cancel_pending != 0u && request->done != 0 ? 1u : 0u;
			pthread_mutex_unlock(&S.queue_mutex);
			if ( gone != 0u )
				break;
			usleep(200);
		}
		printf("disconnect_us %llu io_error %u\n",(unsigned long long)(TestNowUs() - closed),request->status == (uint32_t)SPARK_STATUS_IO_ERROR ? 1u : 0u);
		assert(pthread_join(thread,0) == 0);
	}
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="api-cancel-") as directory:
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
        result = subprocess.run([str(binary), str(ROUNDS)], cwd=ROOT, timeout=90, capture_output=True, text=True)
        if result.returncode:
            sys.stderr.write(result.stderr)
            result.check_returncode()
    lines = [line.split() for line in result.stdout.splitlines()]
    values = {line[0]: line[1:] for line in lines if line[0].endswith("_priority")}
    failures = []
    if values.get("default_priority") != ["2"]:
        failures.append(f"a request that names no priority must run as interactive (2), the class every client gets by default: {values.get('default_priority')}")
    if values.get("explicit_priority") != ["1"]:
        failures.append(f"a request's own priority must be kept: {values.get('explicit_priority')}")
    waits = [int(line[1]) for line in lines if line[0] == "disconnect_us"]
    statuses = {line[3] for line in lines if line[0] == "disconnect_us"}
    if len(waits) != ROUNDS:
        failures.append(f"every round must report its disconnect: {result.stdout}")
    if waits and max(waits) > BOUND_MS * 1000:
        failures.append(f"a client that disconnects while its request waits must be noticed within {BOUND_MS} ms, about one decode step, so the "
                        f"engine can free the sequence at once; the slowest took {max(waits) / 1000:.1f} ms ({waits})")
    if statuses != {"1"}:
        failures.append(f"an abandoned request is closed with the I/O status, never a success: {statuses}")
    for failure in failures:
        print("FAIL test_model_api_cancel:", failure)
    if failures:
        raise SystemExit(1)
    print(f"PASS test_model_api_cancel: requests default to interactive; {ROUNDS} disconnects noticed within {max(waits) / 1000:.1f} ms (bound {BOUND_MS} ms)")


if __name__ == "__main__":
    main()
