#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_hidden_transport_rdma_control.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct SparkTestRdmaHelloThread
{
    int fd;
    uint32_t timeout_milli;
    SparkHiddenTransportRdmaV4Identity identity;
    SparkStatus status;
} SparkTestRdmaHelloThread;

static void SparkTestInitializeRdmaIdentityPair(
    SparkHiddenTransportRdmaV4Identity *source,
    SparkHiddenTransportRdmaV4Identity *sink)
{
    memset(source,0,sizeof(*source));
    source->magic = SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MAGIC;
    source->protocol_version = SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_VERSION;
    source->transport_abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    source->descriptor_bytes = sizeof(*source);
    source->sender_role = 1u;
    source->peer_sender_role = 0u;
    source->local_rank = 2u;
    source->peer_rank = 7u;
    source->source_rank = 2u;
    source->sink_rank = 7u;
    source->control_port = 59007u;
    source->hidden_dimension = 2048u;
    source->bytes_per_sequence = 4096u;
    source->max_active_sequence_count = 1024u;
    source->persistent_credit_count = 64u;
    source->lane_count = 8u;
    source->doorbell_max_bytes = 262144u;
    source->memory_mode = 1u;
    source->capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS |
        SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    source->max_packet_bytes = 4194304u;
    source->route_identifier = 0xfedcba9876543210ull;
    strcpy(source->transport_module_id,
        SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID);
    strcpy(source->route_name,"tp-device.fedcba9876543210.1.2.7");
    strcpy(source->source_host,"node-source.example");
    strcpy(source->sink_host,"node-sink.example");
    *sink = *source;
    sink->sender_role = 0u;
    sink->peer_sender_role = 1u;
    sink->local_rank = 7u;
    sink->peer_rank = 2u;
}

static void *SparkTestExchangeRdmaHello(void *context)
{
    SparkTestRdmaHelloThread *thread = (SparkTestRdmaHelloThread *)context;
    thread->status = SparkHiddenTransportRdmaV4ExchangeCompatibilityHello(
        thread->fd,
        SparkHiddenTransportRdmaControlDeadlineNs(thread->timeout_milli),
        &thread->identity);
    return 0;
}

static void SparkTestRdmaHelloPair(
    SparkHiddenTransportRdmaV4Identity *source,
    SparkHiddenTransportRdmaV4Identity *sink,
    SparkStatus expected)
{
    SparkTestRdmaHelloThread thread;
    pthread_t thread_id;
    SparkStatus status;
    int sockets[2];

    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    memset(&thread,0,sizeof(thread));
    thread.fd = sockets[1];
    thread.timeout_milli = 500u;
    thread.identity = *sink;
    assert(pthread_create(&thread_id,0,SparkTestExchangeRdmaHello,&thread) == 0);
    status = SparkHiddenTransportRdmaV4ExchangeCompatibilityHello(sockets[0],
        SparkHiddenTransportRdmaControlDeadlineNs(500u),source);
    assert(pthread_join(thread_id,0) == 0);
    assert(status == expected);
    assert(thread.status == expected);
    close(sockets[0]);
    close(sockets[1]);
}

static void SparkTestRdmaV4HelloIdentity(void)
{
    SparkHiddenTransportRdmaV4Identity source,sink,mutated;
    uint32_t mismatch;

    SparkTestInitializeRdmaIdentityPair(&source,&sink);
    SparkTestRdmaHelloPair(&source,&sink,SPARK_STATUS_OK);
    for (mismatch=0u; mismatch<15u; mismatch++)
    {
        mutated = sink;
        switch (mismatch)
        {
            case 0u: mutated.protocol_version = 3u; break;
            case 1u: mutated.route_identifier++; break;
            case 2u: strcpy(mutated.transport_module_id,"wrong.module"); break;
            case 3u: strcpy(mutated.route_name,"wrong.route"); break;
            case 4u: strcpy(mutated.source_host,"wrong-source"); break;
            case 5u: strcpy(mutated.sink_host,"wrong-sink"); break;
            case 6u: mutated.local_rank++; break;
            case 7u: mutated.sender_role = 1u; break;
            case 8u: mutated.control_port++; break;
            case 9u: mutated.hidden_dimension++; break;
            case 10u: mutated.max_active_sequence_count++; break;
            case 11u: mutated.persistent_credit_count++; break;
            case 12u: mutated.doorbell_max_bytes++; break;
            case 13u: mutated.lane_count++; break;
            default: mutated.memory_mode++; break;
        }
        SparkTestRdmaHelloPair(&source,&mutated,
            SPARK_STATUS_VALIDATION_FAILED);
    }
}

static void SparkTestRdmaV4HelloTimeoutAndNoSigpipe(void)
{
    SparkHiddenTransportRdmaV4Identity source,sink;
    SparkStatus status;
    uint64_t before,after;
    int sockets[2];

    SparkTestInitializeRdmaIdentityPair(&source,&sink);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    before = SparkHiddenTransportRdmaControlMonotonicNs();
    status = SparkHiddenTransportRdmaV4ExchangeCompatibilityHello(sockets[0],
        SparkHiddenTransportRdmaControlDeadlineNs(25u),&source);
    after = SparkHiddenTransportRdmaControlMonotonicNs();
    assert(status == SPARK_STATUS_BUSY);
    assert(after >= before + 15000000ull);
    assert(after < before + 500000000ull);
    close(sockets[0]);
    close(sockets[1]);

    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    close(sockets[1]);
    status = SparkHiddenTransportRdmaV4ExchangeCompatibilityHello(sockets[0],
        SparkHiddenTransportRdmaControlDeadlineNs(25u),&source);
    assert(status == SPARK_STATUS_IO_ERROR);
    close(sockets[0]);
}

static void SparkTestRdmaSessionFenceIsPeerVisible(void)
{
    char byte;
    int sockets[2];

    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    assert(send(sockets[1],"x",1u,MSG_NOSIGNAL) == 1);
    assert(recv(sockets[0],&byte,1u,0) == 1);
    assert(byte == 'x');
    assert(SparkHiddenTransportRdmaControlFenceSession(sockets[0]) ==
        SPARK_STATUS_OK);
    assert(SparkHiddenTransportRdmaControlFenceSession(sockets[0]) ==
        SPARK_STATUS_OK);
    assert(recv(sockets[0],&byte,1u,0) == 0);
    assert(recv(sockets[1],&byte,1u,0) == 0);
    close(sockets[0]);
    close(sockets[1]);
}

int main(void)
{
    SparkTestRdmaV4HelloIdentity();
    SparkTestRdmaV4HelloTimeoutAndNoSigpipe();
    SparkTestRdmaSessionFenceIsPeerVisible();
    puts("hidden_transport_rdma_control: ok");
    return 0;
}
