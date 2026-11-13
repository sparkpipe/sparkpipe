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
static char TestKvServerMeminfo[96];
static uint64_t TestKvServerSharedWindowBytes;

static void *TestKvServerMain(void *argument)
{
	(void)SparkWeightdServerRun((SparkWeightdServer *)argument,&TestKvServerStopFlag);
	return(0);
}

static const char *TestKvServerMeminfoPath(void)
{
	FILE *file;
	if ( access(SPARK_WEIGHTD_MEMINFO_PATH,R_OK) == 0 )
		return(SPARK_WEIGHTD_MEMINFO_PATH);
	(void)snprintf(TestKvServerMeminfo,sizeof(TestKvServerMeminfo),"/tmp/spark-kv-meminfo-%ld",(long)getpid());
	file = fopen(TestKvServerMeminfo,"w");
	assert(file != 0 && fprintf(file,"MemTotal: 134217728 kB\nMemAvailable: 134217728 kB\n") > 0 && fclose(file) == 0);
	return(TestKvServerMeminfo);
}

static void TestKvServerStart(uint64_t kv_reserve_bytes)
{
	SparkWeightdServerConfig config;
	signal(SIGPIPE,SIG_IGN);
	(void)snprintf(TestKvServerSocket,sizeof(TestKvServerSocket),"/tmp/spark-kv-weightd-%ld.sock",(long)getpid());
	(void)unlink(TestKvServerSocket);
	memset(&config,0,sizeof(config));
	config.socket_path = TestKvServerSocket;
	config.meminfo_path = TestKvServerMeminfoPath();
	config.device_bytes_max = kv_reserve_bytes + (64ull << 20);
	config.kv_reserve_bytes = kv_reserve_bytes;
	config.kv_shared_window_bytes = TestKvServerSharedWindowBytes;
	TestKvServerStopFlag = 0;
	assert(SparkWeightdServerCreate(&config,&TestKvServer) == SPARK_STATUS_OK);
	assert(pthread_create(&TestKvServerThread,0,TestKvServerMain,TestKvServer) == 0);
}

static void TestKvServerFinish(void)
{
	TestKvServerStopFlag = 1;
	assert(pthread_join(TestKvServerThread,0) == 0);
	SparkWeightdServerDestroy(TestKvServer);
	TestKvServer = 0;
	if ( TestKvServerMeminfo[0] != '\0' )
		(void)unlink(TestKvServerMeminfo);
}
