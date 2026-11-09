#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_error_site.h"

SparkStatus SparkTpDeviceCollectiveMeshTopology(uint32_t rank,uint32_t degree,struct SparkWeightdMeshTopology *topology)
{
	(void)rank;
	(void)degree;
	(void)topology;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkTpDeviceCollectiveApplyTopology(const SparkTpDeviceCollectiveTopology *topology,SparkTpDeviceCollectiveConfig *config)
{
	(void)topology;
	(void)config;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}
