#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct SparkWeightdClient SparkWeightdClient;

extern SparkStatus SparkWeightdClientConnect(const char *path,
    SparkWeightdClient *client, uint64_t reserved);
extern SparkStatus SparkWeightdClientDisconnect(SparkWeightdClient *client);
extern SparkStatus SparkWeightdClientMeshBroadcast(SparkWeightdClient *client,
    uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset,
    uint32_t length, uint64_t timeout_nanoseconds);

typedef struct SparkHiddenSparkHostRdmaState
{
    SparkWeightdClient client;
    void *mesh_buffer;
    uint32_t mesh_buffer_bytes;
} SparkHiddenSparkHostRdmaState;

static SparkStatus SparkHiddenSparkHostRdmaInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportInterface *interface,
    SparkHiddenTransportSession **session)
{
    SparkHiddenSparkHostRdmaState *state;
    const char *socket;

    if ( endpoint == 0 || interface == 0 || session == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    *session = 0;
    state = calloc(1u,sizeof(*state));
    if ( state == 0 )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    socket = getenv("SPARK_WEIGHTD_SOCKET");
    {
        char path[128];
        snprintf(path,sizeof(path),"%s",
            socket != 0 ? socket : "/tmp/spark_weightd.sock");
        if ( SparkWeightdClientConnect(path,&state->client,0) !=
                SPARK_STATUS_OK )
        {
            free(state);
            return SPARK_STATUS_IO_ERROR;
        }
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
    if ( state == 0 || local_buffer == 0 || bytes == 0u ||
         state->mesh_buffer == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    memcpy((uint8_t *)state->mesh_buffer,local_buffer,(size_t)bytes);
    return SparkWeightdClientMeshBroadcast(&state->client,
        0x7FFFu,0,remote_offset,(uint32_t)bytes,5000000000ull);
}

static SparkStatus SparkHiddenSparkHostRdmaPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    if ( transport_state == 0 || completion == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
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
        return SPARK_STATUS_INVALID_ARGUMENT;
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

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void);

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface interface;
    interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    interface.descriptor_bytes = sizeof(SparkHiddenTransportInterface);
    interface.capability_flags = SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    interface.initialize = SparkHiddenSparkHostRdmaInitialize;
    interface.destroy = SparkHiddenSparkHostRdmaDestroy;
    interface.send = SparkHiddenSparkHostRdmaSendFixed;
    interface.poll = SparkHiddenSparkHostRdmaPoll;
    interface.set_fixed_local = SparkHiddenSparkHostRdmaSetFixedLocal;
    interface.set_fixed_remote = SparkHiddenSparkHostRdmaSetFixedRemote;
    return &interface;
}
