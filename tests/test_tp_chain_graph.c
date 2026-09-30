#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tp_chain_graph.h"

#define LOG_CAPACITY 64u

static char LOG[LOG_CAPACITY][32];
static uint32_t LOG_COUNT;
static SparkStatus ARM_STATUS[2];
static SparkStatus PRELAUNCH_STATUS[2];
static SparkStatus VERIFY_STATUS[2];
static uint64_t GRAPH_ERROR[2];
static uint32_t STREAM_ORDERED[2];
static uint32_t CAPTURING;
static uint32_t WALK_SAW_CAPTURE;
static uint32_t WALK_SITE;
static cudaError_t INSTANTIATE_ERROR;
static uint32_t GRAPHS_LIVE;
static uint32_t EXECS_LIVE;
static int FAILURES;
static SparkTpDeviceCollective COLLECTIVE[2];

#define CHECK(condition) do { if ( !(condition) ) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#condition); FAILURES++; } } while (0)

static uint32_t Which(const SparkTpDeviceCollective *collective)
{
	return(collective == &COLLECTIVE[1] ? 1u : 0u);
}

static void Log(const char *verb,uint32_t index)
{
	if ( LOG_COUNT < LOG_CAPACITY )
		(void)snprintf(LOG[LOG_COUNT++],sizeof(LOG[0]),"%s%u",verb,index);
}

static uint32_t LogHas(uint32_t position,const char *entry)
{
	return(position < LOG_COUNT && strcmp(LOG[position],entry) == 0 ? 1u : 0u);
}

static void Reset(void)
{
	memset(LOG,0,sizeof(LOG));
	LOG_COUNT = 0u;
	memset(ARM_STATUS,0,sizeof(ARM_STATUS));
	memset(PRELAUNCH_STATUS,0,sizeof(PRELAUNCH_STATUS));
	memset(VERIFY_STATUS,0,sizeof(VERIFY_STATUS));
	memset(GRAPH_ERROR,0,sizeof(GRAPH_ERROR));
	STREAM_ORDERED[0] = STREAM_ORDERED[1] = 1u;
	CAPTURING = 0u;
	WALK_SAW_CAPTURE = 0u;
	WALK_SITE = 0u;
	INSTANTIATE_ERROR = cudaSuccess;
}

SparkStatus SparkTpDeviceCollectiveArmCapture(SparkTpDeviceCollective *collective)
{
	Log("arm",Which(collective));
	return(ARM_STATUS[Which(collective)]);
}

SparkStatus SparkTpDeviceCollectiveDisarmCapture(SparkTpDeviceCollective *collective)
{
	Log("disarm",Which(collective));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkTpDeviceCollectiveGraphPreLaunch(SparkTpDeviceCollective *collective,void *stream)
{
	(void)stream;
	Log("pre",Which(collective));
	return(PRELAUNCH_STATUS[Which(collective)]);
}

SparkStatus SparkTpDeviceCollectiveGraphCancelSeed(SparkTpDeviceCollective *collective,void *stream)
{
	(void)stream;
	Log("seed",Which(collective));
	return(SPARK_STATUS_OK);
}

void SparkTpDeviceCollectiveBroadcastCancel(SparkTpDeviceCollective *collective)
{
	Log("cancel",Which(collective));
}

SparkStatus SparkTpDeviceCollectiveVerifyDeferred(SparkTpDeviceCollective *collective,void *stream)
{
	(void)stream;
	Log("verify",Which(collective));
	return(VERIFY_STATUS[Which(collective)]);
}

uint64_t SparkTpDeviceCollectiveGraphError(SparkTpDeviceCollective *collective)
{
	Log("error",Which(collective));
	return(GRAPH_ERROR[Which(collective)]);
}

uint32_t SparkTpDeviceCollectiveStreamOrdered(const SparkTpDeviceCollective *collective)
{
	return(STREAM_ORDERED[Which(collective)]);
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream,cudaStreamCaptureMode mode)
{
	(void)stream;
	(void)mode;
	CAPTURING = 1u;
	Log("begin",0u);
	return(cudaSuccess);
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream,cudaGraph_t *graph)
{
	(void)stream;
	CAPTURING = 0u;
	Log("end",0u);
	*graph = (cudaGraph_t)(uintptr_t)0x1000u;
	GRAPHS_LIVE++;
	return(cudaSuccess);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec,cudaGraph_t graph,...)
{
	(void)graph;
	if ( INSTANTIATE_ERROR != cudaSuccess )
		return(INSTANTIATE_ERROR);
	*exec = (cudaGraphExec_t)(uintptr_t)0x2000u;
	EXECS_LIVE++;
	return(cudaSuccess);
}

cudaError_t cudaGraphUpload(cudaGraphExec_t exec,cudaStream_t stream)
{
	(void)exec;
	(void)stream;
	return(cudaSuccess);
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
	(void)graph;
	GRAPHS_LIVE--;
	return(cudaSuccess);
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec)
{
	(void)exec;
	EXECS_LIVE--;
	return(cudaSuccess);
}

cudaError_t cudaGetLastError(void)
{
	return(cudaSuccess);
}

static uint32_t Walk(void *context)
{
	(void)context;
	WALK_SAW_CAPTURE = CAPTURING;
	Log("walk",0u);
	return(WALK_SITE);
}

static SparkTpChainCollectives Two(void)
{
	SparkTpChainCollectives collectives;
	memset(&collectives,0,sizeof(collectives));
	collectives.items[0] = &COLLECTIVE[0];
	collectives.items[1] = &COLLECTIVE[1];
	collectives.count = 2u;
	return(collectives);
}

static void TestModeParse(void)
{
	uint32_t mode = 99u;
	CHECK(SparkTpChainModeParse(0,&mode) == SPARK_STATUS_OK && mode == SPARK_TP_CHAIN_MODE_EAGER);
	CHECK(SparkTpChainModeParse("linear",&mode) == SPARK_STATUS_OK && mode == SPARK_TP_CHAIN_MODE_LINEAR);
	CHECK(SparkTpChainModeParse("graph",&mode) == SPARK_STATUS_OK && mode == SPARK_TP_CHAIN_MODE_GRAPH);
	CHECK(SparkTpChainModeParse("eager",&mode) == SPARK_STATUS_OK && mode == SPARK_TP_CHAIN_MODE_EAGER);
	mode = SPARK_TP_CHAIN_MODE_GRAPH;
	CHECK(SparkTpChainModeParse("Graph",&mode) == SPARK_STATUS_INVALID_ARGUMENT && mode == SPARK_TP_CHAIN_MODE_GRAPH);
	CHECK(SparkTpChainModeParse("",&mode) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(strcmp(SparkTpChainModeName(SPARK_TP_CHAIN_MODE_LINEAR),"linear") == 0);
}

static void TestStreamOrdered(void)
{
	SparkTpChainCollectives collectives = Two();
	Reset();
	CHECK(SparkTpChainStreamOrdered(&collectives) == 1u);
	STREAM_ORDERED[1] = 0u;
	CHECK(SparkTpChainStreamOrdered(&collectives) == 0u);
	collectives.count = 0u;
	CHECK(SparkTpChainStreamOrdered(&collectives) == 1u);
}

static void TestArmFailureDisarmsEveryCollective(void)
{
	SparkTpChainCollectives collectives = Two();
	Reset();
	ARM_STATUS[1] = SPARK_STATUS_IO_ERROR;
	CHECK(SparkTpChainArm(&collectives) == SPARK_STATUS_IO_ERROR);
	CHECK(LOG_COUNT == 4u && LogHas(0u,"arm0") && LogHas(1u,"arm1") && LogHas(2u,"disarm0") && LogHas(3u,"disarm1"));
}

static void TestRecordCapturesTheWalkBetweenArmAndDisarm(void)
{
	SparkTpChainCollectives collectives = Two();
	void *exec = 0;
	uint32_t site = 77u;
	Reset();
	CHECK(SparkTpChainGraphRecord(&collectives,(void *)0x10,Walk,0,&exec,&site) == SPARK_STATUS_OK);
	CHECK(exec != 0 && site == 0u && WALK_SAW_CAPTURE == 1u && CAPTURING == 0u);
	CHECK(LogHas(0u,"arm0") && LogHas(1u,"arm1") && LogHas(2u,"begin0") && LogHas(3u,"walk0") && LogHas(4u,"end0") && LogHas(5u,"disarm0") && LogHas(6u,"disarm1") && LOG_COUNT == 7u);
	CHECK(GRAPHS_LIVE == 0u && EXECS_LIVE == 1u);
	(void)cudaGraphExecDestroy((cudaGraphExec_t)exec);
}

static void TestRecordWalkFailureClosesTheCapture(void)
{
	SparkTpChainCollectives collectives = Two();
	void *exec = (void *)0x1;
	uint32_t site = 0u;
	Reset();
	WALK_SITE = 7u;
	CHECK(SparkTpChainGraphRecord(&collectives,(void *)0x10,Walk,0,&exec,&site) == SPARK_STATUS_INTERNAL_ERROR);
	CHECK(exec == 0 && site == 7u && CAPTURING == 0u && GRAPHS_LIVE == 0u && EXECS_LIVE == 0u);
	CHECK(LogHas(LOG_COUNT - 2u,"disarm0") && LogHas(LOG_COUNT - 1u,"disarm1"));
}

static void TestRecordInstantiateFailureLeaksNothing(void)
{
	SparkTpChainCollectives collectives = Two();
	void *exec = 0;
	uint32_t site = 0u;
	Reset();
	INSTANTIATE_ERROR = cudaErrorInvalidValue;
	CHECK(SparkTpChainGraphRecord(&collectives,(void *)0x10,Walk,0,&exec,&site) == SPARK_STATUS_IO_ERROR);
	CHECK(exec == 0 && GRAPHS_LIVE == 0u && EXECS_LIVE == 0u);
}

static void TestRecordArmFailureNeverCaptures(void)
{
	SparkTpChainCollectives collectives = Two();
	void *exec = 0;
	uint32_t site = 0u;
	Reset();
	ARM_STATUS[0] = SPARK_STATUS_UNSUPPORTED;
	CHECK(SparkTpChainGraphRecord(&collectives,(void *)0x10,Walk,0,&exec,&site) == SPARK_STATUS_UNSUPPORTED);
	CHECK(exec == 0 && WALK_SAW_CAPTURE == 0u);
	CHECK(LogHas(0u,"arm0") && LogHas(1u,"disarm0") && LogHas(2u,"disarm1") && LOG_COUNT == 3u);
}

static void TestPreLaunchSeedsEachCollectiveInOrder(void)
{
	SparkTpChainCollectives collectives = Two();
	Reset();
	CHECK(SparkTpChainGraphPreLaunch(&collectives,(void *)0x10) == SPARK_STATUS_OK);
	CHECK(LOG_COUNT == 4u && LogHas(0u,"pre0") && LogHas(1u,"seed0") && LogHas(2u,"pre1") && LogHas(3u,"seed1"));
	Reset();
	PRELAUNCH_STATUS[0] = SPARK_STATUS_INTERNAL_ERROR;
	CHECK(SparkTpChainGraphPreLaunch(&collectives,(void *)0x10) == SPARK_STATUS_INTERNAL_ERROR);
	CHECK(LOG_COUNT == 1u && LogHas(0u,"pre0"));
}

static void TestLinearSettleVerifiesAndCancelsOnFailure(void)
{
	SparkTpChainCollectives collectives = Two();
	Reset();
	CHECK(SparkTpChainSettle(&collectives,(void *)0x10,0u) == SPARK_STATUS_OK);
	CHECK(LOG_COUNT == 2u && LogHas(0u,"verify0") && LogHas(1u,"verify1"));
	Reset();
	VERIFY_STATUS[0] = SPARK_STATUS_IO_ERROR;
	CHECK(SparkTpChainSettle(&collectives,(void *)0x10,0u) == SPARK_STATUS_IO_ERROR);
	CHECK(LOG_COUNT == 3u && LogHas(0u,"verify0") && LogHas(1u,"cancel0") && LogHas(2u,"cancel1"));
}

static void TestGraphSettleChecksErrorsAndDisarms(void)
{
	SparkTpChainCollectives collectives = Two();
	Reset();
	CHECK(SparkTpChainSettle(&collectives,(void *)0x10,1u) == SPARK_STATUS_OK);
	CHECK(LOG_COUNT == 4u && LogHas(0u,"error0") && LogHas(1u,"error1") && LogHas(2u,"disarm0") && LogHas(3u,"disarm1"));
	Reset();
	GRAPH_ERROR[1] = 5u;
	CHECK(SparkTpChainSettle(&collectives,(void *)0x10,1u) == SPARK_STATUS_IO_ERROR);
	CHECK(LOG_COUNT == 6u && LogHas(4u,"cancel0") && LogHas(5u,"cancel1"));
}

static void TestGraphTableKeysAndDestroy(void)
{
	SparkTpChainGraphTable table;
	void **entry;
	memset(&table,0,sizeof(table));
	CHECK(SparkTpChainGraphEntry(&table,0u,0u) == 0);
	CHECK(SparkTpChainGraphEntry(&table,SPARK_TP_CHAIN_GRAPH_MAX_REGIMES,1u) == 0);
	CHECK(SparkTpChainGraphEntry(&table,0u,SPARK_TP_CHAIN_GRAPH_MAX_ROWS + 1u) == 0);
	CHECK(SparkTpChainGraphEntry(&table,1u,3u) != SparkTpChainGraphEntry(&table,0u,3u));
	CHECK(SparkTpChainGraphEntry(&table,1u,3u) == SparkTpChainGraphEntry(&table,1u,4u));
	CHECK(SparkTpChainGraphEntry(&table,1u,4u) != SparkTpChainGraphEntry(&table,1u,5u));
	CHECK(SparkTpChainGraphEntry(&table,1u,129u) == SparkTpChainGraphEntry(&table,1u,SPARK_TP_CHAIN_GRAPH_MAX_ROWS));
	CHECK(SparkTpChainGraphEntry(&table,1u,SPARK_TP_CHAIN_GRAPH_MAX_ROWS) == &table.exec[1][SPARK_TP_CHAIN_GRAPH_BUCKETS - 1u]);
	CHECK(sizeof(table.exec) == SPARK_TP_CHAIN_GRAPH_MAX_REGIMES * SPARK_TP_CHAIN_GRAPH_BUCKETS * sizeof(void *));
	entry = SparkTpChainGraphEntry(&table,1u,SPARK_TP_CHAIN_GRAPH_MAX_ROWS);
	CHECK(entry != 0);
	EXECS_LIVE = 1u;
	*entry = (void *)0x2000;
	SparkTpChainGraphTableDestroy(&table);
	CHECK(*entry == 0 && EXECS_LIVE == 0u);
}

static void TestGraphBucketRows(void)
{
	CHECK(SparkTpChainGraphBucketRows(0u,256u) == 0u);
	CHECK(SparkTpChainGraphBucketRows(1u,256u) == 1u);
	CHECK(SparkTpChainGraphBucketRows(2u,256u) == 2u);
	CHECK(SparkTpChainGraphBucketRows(3u,256u) == 4u);
	CHECK(SparkTpChainGraphBucketRows(9u,256u) == 16u);
	CHECK(SparkTpChainGraphBucketRows(17u,256u) == 32u);
	CHECK(SparkTpChainGraphBucketRows(129u,256u) == 256u);
	CHECK(SparkTpChainGraphBucketRows(256u,256u) == 256u);
	CHECK(SparkTpChainGraphBucketRows(257u,512u) == 0u);
	CHECK(SparkTpChainGraphBucketRows(17u,17u) == 17u);
	CHECK(SparkTpChainGraphBucketRows(9u,12u) == 12u);
	CHECK(SparkTpChainGraphBucketRows(13u,12u) == 0u);
	CHECK((1u << (SPARK_TP_CHAIN_GRAPH_BUCKETS - 1u)) == SPARK_TP_CHAIN_GRAPH_MAX_ROWS);
}

int main(void)
{
	TestModeParse();
	TestStreamOrdered();
	TestArmFailureDisarmsEveryCollective();
	TestRecordCapturesTheWalkBetweenArmAndDisarm();
	TestRecordWalkFailureClosesTheCapture();
	TestRecordInstantiateFailureLeaksNothing();
	TestRecordArmFailureNeverCaptures();
	TestPreLaunchSeedsEachCollectiveInOrder();
	TestLinearSettleVerifiesAndCancelsOnFailure();
	TestGraphSettleChecksErrorsAndDisarms();
	TestGraphTableKeysAndDestroy();
	TestGraphBucketRows();
	if ( FAILURES != 0 )
	{
		fprintf(stderr,"test_tp_chain_graph: %d failures\n",FAILURES);
		return(1);
	}
	printf("test_tp_chain_graph: ok\n");
	return(0);
}
