#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_kv_pool.h"
#include "sparkpipe/spark_error_site.h"

SparkStatus SparkWeightdKvPoolGranularity(uint64_t *bytes)
{
	(void)bytes;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkWeightdKvReserveAddress(uint64_t bytes,void **base_out)
{
	(void)bytes;
	(void)base_out;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

void SparkWeightdKvFreeAddress(void *base,uint64_t bytes)
{
	(void)base;
	(void)bytes;
}

SparkStatus SparkWeightdKvPoolMap(const char *socket,const SparkWeightdKvPoolRequest *request,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds,SparkWeightdKvPoolMapping *mapping)
{
	(void)socket;
	(void)request;
	(void)chunk_offsets;
	(void)timeout_nanoseconds;
	(void)mapping;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

void SparkWeightdKvPoolUnmap(SparkWeightdKvPoolMapping *mapping)
{
	(void)mapping;
}

SparkStatus SparkWeightdKvPoolGrow(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds)
{
	(void)mapping;
	(void)target_chunks;
	(void)state;
	(void)timeout_nanoseconds;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkWeightdKvPoolShrink(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds)
{
	(void)mapping;
	(void)target_chunks;
	(void)state;
	(void)timeout_nanoseconds;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkWeightdKvPoolStatus(SparkWeightdKvPoolMapping *mapping,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds)
{
	(void)mapping;
	(void)state;
	(void)timeout_nanoseconds;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkWeightdKvSharedAttach(const char *socket,const SparkWeightdKvSharedRequest *request,uint64_t timeout_nanoseconds,SparkWeightdKvSharedMapping *mapping)
{
	(void)socket;
	(void)request;
	(void)timeout_nanoseconds;
	(void)mapping;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkWeightdKvSharedMapChunks(SparkWeightdKvSharedMapping *mapping,void *device_base,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds)
{
	(void)mapping;
	(void)device_base;
	(void)chunk_offsets;
	(void)timeout_nanoseconds;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

void SparkWeightdKvSharedUnmap(SparkWeightdKvSharedMapping *mapping)
{
	(void)mapping;
}

SparkStatus SparkWeightdClientConnect(const char *socket_path,SparkWeightdClient **client,SparkWeightdHelloResult *hello_out)
{
	(void)socket_path;
	(void)hello_out;
	if ( client != 0 )
		*client = 0;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkWeightdClientLaneAcquire(SparkWeightdClient *client,uint32_t requested_lane,const SparkWeightdMeshTopology *topology,uint32_t *lane_out,uint64_t timeout_nanoseconds)
{
	(void)client;
	(void)requested_lane;
	(void)topology;
	(void)lane_out;
	(void)timeout_nanoseconds;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

void SparkWeightdClientClose(SparkWeightdClient *client)
{
	(void)client;
}
