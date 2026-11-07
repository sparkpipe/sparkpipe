#pragma once

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "sparkpipe/spark_tp_device_collective.h"

#define NODE_EMULATION_MAX_NODES 16u

struct NodeEmulationWorld
{
	uint32_t nodes;
	pthread_barrier_t barrier;
	const void *send[NODE_EMULATION_MAX_NODES];
};

struct NodeEmulationNode
{
	NodeEmulationWorld *world;
	uint32_t node;
};

static void NodeEmulationFail(const char *what)
{
	printf("FAIL node emulation: %s\n",what);
	exit(1);
}

static void NodeEmulationStart(NodeEmulationWorld *world,uint32_t nodes)
{
	if ( nodes == 0u || nodes > NODE_EMULATION_MAX_NODES || pthread_barrier_init(&world->barrier,0,nodes) != 0 )
		NodeEmulationFail("world size");
	world->nodes = nodes;
}

static void NodeEmulationCollective(void *context,void *stream_word,uint32_t operation,uint32_t rows,uint32_t row_elements,const void *send,void *receive)
{
	NodeEmulationNode *self = (NodeEmulationNode *)context;
	NodeEmulationWorld *world = self->world;
	cudaStream_t stream = (cudaStream_t)(uintptr_t)stream_word;
	const size_t block = (size_t)rows * row_elements * sizeof(uint16_t);
	if ( operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
		NodeEmulationFail("operation");
	if ( cudaStreamSynchronize(stream) != cudaSuccess )
		NodeEmulationFail("stream before the exchange");
	world->send[self->node] = send;
	pthread_barrier_wait(&world->barrier);
	for (uint32_t source = 0u; source < world->nodes; source++)
	{
		const uint8_t *from = (const uint8_t *)world->send[source] + (operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? (size_t)self->node * block : 0u);
		if ( cudaMemcpyAsync((uint8_t *)receive + (size_t)source * block,from,block,cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
			NodeEmulationFail("copy");
	}
	if ( cudaStreamSynchronize(stream) != cudaSuccess )
		NodeEmulationFail("stream after the exchange");
	pthread_barrier_wait(&world->barrier);
}
