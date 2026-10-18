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
#include <arpa/inet.h>
#include <sys/time.h>

static volatile int Accepted = -2;

static void *TestAcceptThread(void *argument)
{
	Accepted = api_accept((int)(intptr_t)argument);
	return(0);
}

static double TestCpuSeconds(void)
{
	struct rusage usage;
	assert(getrusage(RUSAGE_SELF,&usage) == 0);
	return((double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1e6 + (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / 1e6);
}

int main(void)
{
	struct rlimit limit,low;
	struct sockaddr_in address;
	socklen_t length = sizeof(address);
	int server,client,fillers[256],filled = 0,index;
	uint64_t raised;
	pthread_t thread;
	double cpu_before,cpu_after;
	struct timespec wait = {0,300000000L},tick = {0,10000000L};
	assert(getrlimit(RLIMIT_NOFILE,&limit) == 0);
	low = limit;
	low.rlim_cur = 64u;
	assert(setrlimit(RLIMIT_NOFILE,&low) == 0);
	raised = api_raise_open_files();
	assert(raised > 64u);
	assert(limit.rlim_max == RLIM_INFINITY || raised == (uint64_t)limit.rlim_max);
	fprintf(stdout,"raised %llu\n",(unsigned long long)raised);

	server = socket(AF_INET,SOCK_STREAM,0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	assert(bind(server,(struct sockaddr *)&address,sizeof(address)) == 0 && listen(server,SOMAXCONN) == 0);
	assert(getsockname(server,(struct sockaddr *)&address,&length) == 0);
	client = socket(AF_INET,SOCK_STREAM,0);
	assert(connect(client,(struct sockaddr *)&address,sizeof(address)) == 0);
	low.rlim_cur = 48u;
	assert(setrlimit(RLIMIT_NOFILE,&low) == 0);
	for (;;)
	{
		int fd = open("/dev/null",O_RDONLY);
		if ( fd < 0 )
		{
			assert(errno == EMFILE);
			break;
		}
		assert(filled < 256);
		fillers[filled++] = fd;
	}
	cpu_before = TestCpuSeconds();
	assert(pthread_create(&thread,0,TestAcceptThread,(void *)(intptr_t)server) == 0);
	nanosleep(&wait,0);
	cpu_after = TestCpuSeconds();
	assert(Accepted == -2);
	fprintf(stdout,"starved cpu %.3f s over 0.300 s\n",cpu_after - cpu_before);
	assert(cpu_after - cpu_before < 0.1);
	for (index=0; index<4 && filled > 0; index++)
		close(fillers[--filled]);
	for (index=0; index<100 && Accepted == -2; index++)
		nanosleep(&tick,0);
	assert(Accepted >= 0);
	assert(pthread_join(thread,0) == 0);
	assert(write(client,"x",1) == 1);
	{
		char byte = 0;
		assert(read(Accepted,&byte,1) == 1 && byte == 'x');
	}
	fprintf(stdout,"accepted after fds freed\n");
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="api-accept-") as directory:
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
            sys.stderr.write(result.stdout + result.stderr)
            result.check_returncode()
        assert "accept failed: " in result.stderr and result.stderr.count("accept failed: ") == 1, result.stderr
        assert "accept recovered after " in result.stderr, result.stderr
    print("PASS the API raises its open-files limit to the hard limit, and an accept starved of descriptors waits without spinning, logs once and serves the connection once descriptors free up")


if __name__ == "__main__":
    main()
