#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_error_site.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

typedef struct SparkHiddenSparkHostRdmaState
{
    SparkHiddenTransportInterface transport_interface;
    SparkWeightdClient client;
    void *mesh_buffer;
    uint32_t mesh_buffer_bytes;
    uint32_t mesh_ready;
    char socket_path[128];
} SparkHiddenSparkHostRdmaState;

static SparkStatus SparkHiddenSparkHostRdmaInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportInterface *interface,
    SparkHiddenTransportSession **session)
{
    SparkHiddenSparkHostRdmaState *state;
    const char *socket;

    if ( endpoint == 0 || interface == 0 || session == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    *session = 0;
    state = calloc(1u,sizeof(*state));
    if ( state == 0 )
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    state->transport_interface = *interface;
    socket = getenv("SPARK_WEIGHTD_SOCKET");
    snprintf(state->socket_path,sizeof(state->socket_path),"%s",
        socket != 0 ? socket : "/tmp/spark_weightd.sock");
    if ( SparkWeightdClientConnect(state->socket_path,&state->client,0) !=
            SPARK_STATUS_OK )
    {
        free(state);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    *session = (SparkHiddenTransportSession *)state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaDestroy(void *transport_state)
{
    SparkHiddenSparkHostRdmaState *state =
        (SparkHiddenSparkHostRdmaState *)transport_state;
    if ( state == 0 )
        return;
    (void)SparkWeightdClientDisconnect(&state->client);
    free(state);
}

static SparkStatus SparkHiddenSparkHostRdmaSendFixed(
    void *transport_state,
    const void *local_buffer,
    uint64_t bytes,
    uint64_t remote_offset,
    uint32_t sequence)
{
    SparkHiddenSparkHostRdmaState *state =
        (SparkHiddenSparkHostRdmaState *)transport_state;
    if ( state == 0 || local_buffer == 0 || bytes == 0u )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memcpy((uint8_t *)state->mesh_buffer + (remote_offset & 0xFFFFu),
        local_buffer,(size_t)bytes);
    return SparkWeightdClientMeshBroadcast(&state->client,
        0x7FFFu,remote_offset & 0xFFFFu,remote_offset,
        (uint32_t)bytes,5000000000ull);
}

static SparkStatus SparkHiddenSparkHostRdmaPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    if ( transport_state == 0 || completion == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(completion,0,sizeof(*completion));
    completion->status = SPARK_STATUS_BUSY;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSetFixedLocal(
    void *transport_state,
    const void *buffer,
    uint64_t bytes)
{
    SparkHiddenSparkHostRdmaState *state =
        (SparkHiddenSparkHostRdmaState *)transport_state;
    if ( state == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    state->mesh_buffer = (void *)buffer;
    state->mesh_buffer_bytes = (uint32_t)bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSetFixedRemote(
    void *transport_state,
    uint64_t remote_addr,
    uint32_t rkey)
{
    (void)transport_state;
    (void)remote_addr;
    (void)rkey;
    return SPARK_STATUS_OK;
}

static const SparkHiddenTransportInterface spark_hidden_spark_host_rdma_interface =
{
    .abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION,
    .descriptor_bytes = sizeof(SparkHiddenTransportInterface),
    .capability_flags = SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS,
    .initialize = SparkHiddenSparkHostRdmaInitialize,
    .destroy = SparkHiddenSparkHostRdmaDestroy,
    .send = SparkHiddenSparkHostRdmaSendFixed,
    .poll = SparkHiddenSparkHostRdmaPoll,
    .set_fixed_local = SparkHiddenSparkHostRdmaSetFixedLocal,
    .set_fixed_remote = SparkHiddenSparkHostRdmaSetFixedRemote,
};

const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    return &spark_hidden_spark_host_rdma_interface;
}
