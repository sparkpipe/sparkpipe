#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern SparkStatus SparkWeightdClientConnect(const char *path,
    void *client, uint64_t reserved);
extern SparkStatus SparkWeightdClientDisconnect(void *client);
extern SparkStatus SparkWeightdClientMeshBroadcast(void *client,
    uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset,
    uint32_t length, uint64_t timeout_nanoseconds);

typedef struct SparkHiddenSparkHostRdmaState
{
    char client[4096];
    void *mesh_buffer;
    uint32_t mesh_buffer_bytes;
} SparkHiddenSparkHostRdmaState;

static SparkStatus SparkHiddenSparkHostRdmaInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    SparkHiddenSparkHostRdmaState *state;
    const char *socket;

    if ( endpoint == 0 || transport_state == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    *transport_state = 0;
    state = (SparkHiddenSparkHostRdmaState *)calloc(1u,sizeof(*state));
    if ( state == 0 )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    socket = getenv("SPARK_WEIGHTD_SOCKET");
    {
        char path[128];
        snprintf(path,sizeof(path),"%s",
            socket != 0 ? socket : "/tmp/spark_weightd.sock");
        if ( SparkWeightdClientConnect(path,state->client,0) !=
                SPARK_STATUS_OK )
        {
            free(state);
            return SPARK_STATUS_IO_ERROR;
        }
    }
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaDestroy(void *transport_state)
{
    SparkHiddenSparkHostRdmaState *state =
        (SparkHiddenSparkHostRdmaState *)transport_state;
    if ( state == 0 )
        return;
    (void)SparkWeightdClientDisconnect(state->client);
    free(state);
}

static SparkStatus SparkHiddenSparkHostRdmaSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaState *state =
        (SparkHiddenSparkHostRdmaState *)transport_state;
    uint64_t bytes;
    if ( state == 0 || packet == 0 || state->mesh_buffer == 0 ||
         packet->hidden_bf16 == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    bytes = (uint64_t)packet->active_sequence_count *
        packet->hidden_dimension * packet->bytes_per_sequence;
    memcpy(state->mesh_buffer,packet->hidden_bf16,(size_t)bytes);
    return SparkWeightdClientMeshBroadcast(state->client,
        0x7FFFu,0,0,(uint32_t)bytes,5000000000ull);
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
    memcpy(state->mesh_buffer,local_buffer,(size_t)bytes);
    return SparkWeightdClientMeshBroadcast(state->client,
        0x7FFFu,0,remote_offset,(uint32_t)bytes,5000000000ull);
}

static SparkStatus SparkHiddenSparkHostRdmaPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    if ( transport_state == 0 || completion == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(completion,0,sizeof(*completion));
    completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    completion->descriptor_bytes = sizeof(*completion);
    completion->status = SPARK_STATUS_BUSY;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaGetPollDescriptors(
    void *transport_state,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count)
{
    (void)transport_state;(void)descriptors;(void)descriptor_capacity;
    if ( descriptor_count != 0 )
        *descriptor_count = 0;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    (void)transport_state;(void)packet;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSetFixedLocal(
    void *transport_state,
    void *buffer,
    uint64_t bytes)
{
    SparkHiddenSparkHostRdmaState *state =
        (SparkHiddenSparkHostRdmaState *)transport_state;
    if ( state == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    state->mesh_buffer = buffer;
    state->mesh_buffer_bytes = (uint32_t)bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSetFixedRemote(
    void *transport_state,
    uint64_t remote_addr,
    uint64_t remote_bytes,
    uint32_t rkey)
{
    (void)transport_state;(void)remote_addr;(void)remote_bytes;(void)rkey;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaRegisterPersistent(
    void *transport_state,
    uint32_t credit_index,
    SparkHiddenTransportPacket *packet_template)
{
    (void)transport_state;(void)credit_index;(void)packet_template;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCreditReady(
    void *transport_state,
    uint32_t credit_index)
{
    (void)transport_state;(void)credit_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaActivatePersistent(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    SparkHiddenTransportPacket *packet)
{
    (void)transport_state;(void)credit_index;(void)generation;(void)packet;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCancelPersistent(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation)
{
    (void)transport_state;(void)credit_index;(void)generation;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaReleasePersistent(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    void *consumer_cuda_stream)
{
    (void)transport_state;(void)credit_index;(void)generation;
    (void)consumer_cuda_stream;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSendPersistent(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    (void)credit_index;(void)generation;
    return SparkHiddenSparkHostRdmaSend(transport_state,packet);
}

static SparkStatus SparkHiddenSparkHostRdmaReservePersistent(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    (void)transport_state;(void)credit_index;(void)generation;(void)packet;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCancelPersistentSend(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation)
{
    (void)transport_state;(void)credit_index;(void)generation;
    return SPARK_STATUS_OK;
}

static SparkHiddenTransportInterface spark_hidden_spark_host_rdma_interface;

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    memset(&spark_hidden_spark_host_rdma_interface,0,
        sizeof(spark_hidden_spark_host_rdma_interface));
    spark_hidden_spark_host_rdma_interface.abi_version =
        SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    spark_hidden_spark_host_rdma_interface.descriptor_bytes =
        sizeof(SparkHiddenTransportInterface);
    spark_hidden_spark_host_rdma_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    spark_hidden_spark_host_rdma_interface.initialize =
        SparkHiddenSparkHostRdmaInitialize;
    spark_hidden_spark_host_rdma_interface.destroy =
        SparkHiddenSparkHostRdmaDestroy;
    spark_hidden_spark_host_rdma_interface.send =
        SparkHiddenSparkHostRdmaSend;
    spark_hidden_spark_host_rdma_interface.send_fixed =
        SparkHiddenSparkHostRdmaSendFixed;
    spark_hidden_spark_host_rdma_interface.poll =
        SparkHiddenSparkHostRdmaPoll;
    spark_hidden_spark_host_rdma_interface.get_poll_descriptors =
        SparkHiddenSparkHostRdmaGetPollDescriptors;
    spark_hidden_spark_host_rdma_interface.post_receive =
        SparkHiddenSparkHostRdmaPostReceive;
    spark_hidden_spark_host_rdma_interface.set_fixed_local =
        SparkHiddenSparkHostRdmaSetFixedLocal;
    spark_hidden_spark_host_rdma_interface.set_fixed_remote =
        SparkHiddenSparkHostRdmaSetFixedRemote;
    spark_hidden_spark_host_rdma_interface.register_persistent_receive =
        SparkHiddenSparkHostRdmaRegisterPersistent;
    spark_hidden_spark_host_rdma_interface.persistent_remote_credit_ready =
        SparkHiddenSparkHostRdmaCreditReady;
    spark_hidden_spark_host_rdma_interface.activate_persistent_receive =
        SparkHiddenSparkHostRdmaActivatePersistent;
    spark_hidden_spark_host_rdma_interface.cancel_persistent_receive =
        SparkHiddenSparkHostRdmaCancelPersistent;
    spark_hidden_spark_host_rdma_interface.release_persistent_receive =
        SparkHiddenSparkHostRdmaReleasePersistent;
    spark_hidden_spark_host_rdma_interface.send_persistent =
        SparkHiddenSparkHostRdmaSendPersistent;
    spark_hidden_spark_host_rdma_interface.reserve_persistent_send =
        SparkHiddenSparkHostRdmaReservePersistent;
    spark_hidden_spark_host_rdma_interface.cancel_persistent_send =
        SparkHiddenSparkHostRdmaCancelPersistentSend;
    return &spark_hidden_spark_host_rdma_interface;
}
