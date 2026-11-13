#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_error_site.h"

SparkStatus SparkTpDeviceCollectiveMeshTopology(uint32_t rank,uint32_t degree,const uint32_t *mesh_ranks,struct SparkWeightdMeshTopology *topology)
{
	(void)rank;
	(void)degree;
	(void)mesh_ranks;
	(void)topology;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkTpDeviceCollectiveApplyTopology(const SparkTpDeviceCollectiveTopology *topology,SparkTpDeviceCollectiveConfig *config)
{
	(void)topology;
	(void)config;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}
