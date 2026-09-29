#pragma once

#include <stdint.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_device_collective.h"

#define SPARK_TP_CHAIN_MODE_EAGER 0u
#define SPARK_TP_CHAIN_MODE_LINEAR 1u
#define SPARK_TP_CHAIN_MODE_GRAPH 2u
#define SPARK_TP_CHAIN_MODE_COUNT 3u
#define SPARK_TP_CHAIN_MAX_COLLECTIVES 2u
#define SPARK_TP_CHAIN_GRAPH_MAX_REGIMES 4u
#define SPARK_TP_CHAIN_GRAPH_MAX_ROWS 64u

typedef uint32_t (*SparkTpChainWalkFunction)(void *context);

typedef struct SparkTpChainCollectives
{
	SparkTpDeviceCollective *items[SPARK_TP_CHAIN_MAX_COLLECTIVES];
	uint32_t count;
} SparkTpChainCollectives;

typedef struct SparkTpChainGraphTable
{
	void *exec[SPARK_TP_CHAIN_GRAPH_MAX_REGIMES][SPARK_TP_CHAIN_GRAPH_MAX_ROWS];
	uint32_t captures;
	uint32_t failed;
} SparkTpChainGraphTable;

static inline SparkStatus SparkTpChainModeParse(const char *text,uint32_t *mode)
{
	if ( mode == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( text == 0 || strcmp(text,"eager") == 0 )
		*mode = SPARK_TP_CHAIN_MODE_EAGER;
	else if ( strcmp(text,"linear") == 0 )
		*mode = SPARK_TP_CHAIN_MODE_LINEAR;
	else if ( strcmp(text,"graph") == 0 )
		*mode = SPARK_TP_CHAIN_MODE_GRAPH;
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static inline const char *SparkTpChainModeName(uint32_t mode)
{
	return(mode == SPARK_TP_CHAIN_MODE_LINEAR ? "linear" : mode == SPARK_TP_CHAIN_MODE_GRAPH ? "graph" : "eager");
}

static inline uint32_t SparkTpChainStreamOrdered(const SparkTpChainCollectives *collectives)
{
	uint32_t index;
	if ( collectives == 0 )
		return(0u);
	for (index=0u; index<collectives->count; index++)
		if ( SparkTpDeviceCollectiveStreamOrdered(collectives->items[index]) == 0u )
			return(0u);
	return(1u);
}

static inline void SparkTpChainDisarm(const SparkTpChainCollectives *collectives)
{
	uint32_t index;
	for (index=0u; collectives != 0 && index<collectives->count; index++)
		(void)SparkTpDeviceCollectiveDisarmCapture(collectives->items[index]);
}

static inline SparkStatus SparkTpChainArm(const SparkTpChainCollectives *collectives)
{
	uint32_t index;
	SparkStatus status;
	if ( collectives == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<collectives->count; index++)
	{
		status = SparkTpDeviceCollectiveArmCapture(collectives->items[index]);
		if ( status != SPARK_STATUS_OK )
		{
			SparkTpChainDisarm(collectives);
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkTpChainGraphPreLaunch(const SparkTpChainCollectives *collectives,void *stream)
{
	uint32_t index;
	SparkStatus status;
	if ( collectives == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<collectives->count; index++)
	{
		status = SparkTpDeviceCollectiveGraphPreLaunch(collectives->items[index],stream);
		if ( status == SPARK_STATUS_OK )
			status = SparkTpDeviceCollectiveGraphCancelSeed(collectives->items[index],stream);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static inline void SparkTpChainCancel(const SparkTpChainCollectives *collectives)
{
	uint32_t index;
	for (index=0u; collectives != 0 && index<collectives->count; index++)
		SparkTpDeviceCollectiveBroadcastCancel(collectives->items[index]);
}

static inline SparkStatus SparkTpChainSettle(const SparkTpChainCollectives *collectives,void *stream,uint32_t graph)
{
	uint32_t index;
	SparkStatus status = SPARK_STATUS_OK;
	if ( collectives == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; status == SPARK_STATUS_OK && index<collectives->count; index++)
	{
		if ( graph != 0u )
			status = SparkTpDeviceCollectiveGraphError(collectives->items[index]) == 0ull ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
		else
			status = SparkTpDeviceCollectiveVerifyDeferred(collectives->items[index],stream);
	}
	if ( graph != 0u )
		SparkTpChainDisarm(collectives);
	if ( status != SPARK_STATUS_OK )
		SparkTpChainCancel(collectives);
	return(status);
}

static inline void **SparkTpChainGraphEntry(SparkTpChainGraphTable *table,uint32_t regime,uint32_t rows)
{
	if ( table == 0 || regime >= SPARK_TP_CHAIN_GRAPH_MAX_REGIMES || rows == 0u || rows > SPARK_TP_CHAIN_GRAPH_MAX_ROWS )
		return(0);
	return(&table->exec[regime][rows - 1u]);
}

static inline void SparkTpChainGraphTableDestroy(SparkTpChainGraphTable *table)
{
	uint32_t regime,row;
	if ( table == 0 )
		return;
	for (regime=0u; regime<SPARK_TP_CHAIN_GRAPH_MAX_REGIMES; regime++)
		for (row=0u; row<SPARK_TP_CHAIN_GRAPH_MAX_ROWS; row++)
			if ( table->exec[regime][row] != 0 )
			{
				(void)cudaGraphExecDestroy((cudaGraphExec_t)table->exec[regime][row]);
				table->exec[regime][row] = 0;
			}
}

static inline SparkStatus SparkTpChainGraphRecord(
	const SparkTpChainCollectives *collectives,
	void *stream,
	SparkTpChainWalkFunction walk,
	void *context,
	void **exec_out,
	uint32_t *site_out)
{
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec = 0;
	cudaError_t end_error;
	uint32_t site;
	SparkStatus status;
	if ( collectives == 0 || stream == 0 || walk == 0 || exec_out == 0 || site_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*exec_out = 0;
	*site_out = 0u;
	status = SparkTpChainArm(collectives);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( cudaStreamBeginCapture((cudaStream_t)stream,cudaStreamCaptureModeThreadLocal) != cudaSuccess )
	{
		SparkTpChainDisarm(collectives);
		return(SPARK_STATUS_IO_ERROR);
	}
	site = walk(context);
	end_error = cudaStreamEndCapture((cudaStream_t)stream,&graph);
	*site_out = site;
	if ( site != 0u || end_error != cudaSuccess || graph == 0 )
	{
		if ( graph != 0 )
			(void)cudaGraphDestroy(graph);
		(void)cudaGetLastError();
		SparkTpChainDisarm(collectives);
		return(site != 0u ? SPARK_STATUS_INTERNAL_ERROR : SPARK_STATUS_IO_ERROR);
	}
	if ( cudaGraphInstantiate(&exec,graph,0) != cudaSuccess || cudaGraphUpload(exec,(cudaStream_t)stream) != cudaSuccess )
	{
		if ( exec != 0 )
			(void)cudaGraphExecDestroy(exec);
		(void)cudaGraphDestroy(graph);
		(void)cudaGetLastError();
		SparkTpChainDisarm(collectives);
		return(SPARK_STATUS_IO_ERROR);
	}
	(void)cudaGraphDestroy(graph);
	SparkTpChainDisarm(collectives);
	*exec_out = exec;
	return(SPARK_STATUS_OK);
}
