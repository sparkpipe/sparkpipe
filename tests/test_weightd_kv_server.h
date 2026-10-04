#pragma once

#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_weightd.h"

static SparkWeightdServer *TestKvServer;
static pthread_t TestKvServerThread;
static volatile sig_atomic_t TestKvServerStopFlag;
static char TestKvServerSocket[96];

static void *TestKvServerMain(void *argument)
{
	(void)SparkWeightdServerRun((SparkWeightdServer *)argument,&TestKvServerStopFlag);
	return(0);
}

static void TestKvServerStart(uint64_t kv_reserve_bytes)
{
	SparkWeightdServerConfig config;
	signal(SIGPIPE,SIG_IGN);
	(void)snprintf(TestKvServerSocket,sizeof(TestKvServerSocket),"/tmp/spark-kv-weightd-%ld.sock",(long)getpid());
	(void)unlink(TestKvServerSocket);
	memset(&config,0,sizeof(config));
	config.socket_path = TestKvServerSocket;
	config.device_bytes_max = kv_reserve_bytes + (64ull << 20);
	config.kv_reserve_bytes = kv_reserve_bytes;
	TestKvServerStopFlag = 0;
	assert(SparkWeightdServerCreate(&config,&TestKvServer) == SPARK_STATUS_OK);
	assert(pthread_create(&TestKvServerThread,0,TestKvServerMain,TestKvServer) == 0);
	assert(setenv("SPARK_WEIGHTD_SOCKET",TestKvServerSocket,1) == 0);
	assert(unsetenv("SPARK_WEIGHTD_ATTACH") == 0);
}

static void TestKvServerFinish(void)
{
	TestKvServerStopFlag = 1;
	assert(pthread_join(TestKvServerThread,0) == 0);
	SparkWeightdServerDestroy(TestKvServer);
	TestKvServer = 0;
	(void)unsetenv("SPARK_WEIGHTD_SOCKET");
}
