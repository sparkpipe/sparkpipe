#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_hidden_transport_rdma_control.h"
#include "sparkpipe/spark_memlink.h"

#include <cuda_runtime_api.h>
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifndef SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT
#error "SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT must be explicitly set to 0 or 1"
#endif

#if SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT != 0 && \
    SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT != 1
#error "SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT must be 0 or 1"
#endif

#if SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT == 1 && \
    (!defined(CUDART_VERSION) || CUDART_VERSION < 11030)
#error "GPUDirect RDMA transport requires CUDA runtime 11.3 or newer"
#endif

#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_MAGIC \
    SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MAGIC
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_VERSION \
    SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_VERSION
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_LANE_COUNT 1u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT 32u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_BATCH_COUNT 16u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_QUEUE_DEPTH 256u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_WR_PER_PACKET 2u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE \
    (SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT * \
     SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_WR_PER_PACKET)
#define SPARK_HIDDEN_SPARK_HOST_RDMA_COMPLETION_QUEUE_DEPTH 256u
/* NET-004: work completions drained per ibv_poll_cq call; polling one WC
 * at a time used to pay the full verbs call overhead per completion. */
#define SPARK_HIDDEN_SPARK_HOST_RDMA_COMPLETION_POLL_BATCH_COUNT 32u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT \
    (SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT * \
     SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_WR_PER_PACKET)
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS 50u
/* Bounded control-plane bring-up: the receiver accept() and the sender
 * connect retry used to wait for the peer forever, which is how one failed
 * rank wedged every later rank at 0% CPU with no diagnostic. The timeout is
 * per transport session open, env-overridable for slow fabrics. */
#define SPARK_HIDDEN_SPARK_HOST_RDMA_OPEN_TIMEOUT_DEFAULT_MS 120000u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_OPEN_TIMEOUT_ENV "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_OPEN_TIMEOUT_MS"
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_NAME_BYTES 32u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_RATE_PATH_BYTES 256u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_RATE_TEXT_BYTES 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_ROCE_V2_TEXT_BYTES 7u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_REQUIRED_LINK_RATE_GBPS 100u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_GID_COUNT 256u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_POLL_TIMEOUT_MS 1000u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_DOORBELL_MAX_BYTES 262144u

#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HELLO 1u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY 2u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_TRANSFER_COMPLETE 3u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_PERSISTENT_ADVERTISE 4u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_PERSISTENT_RETURN 5u

#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_BATCH 0x2000000000000000ull
#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_SEND 0x4000000000000000ull
#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_DOORBELL_RECEIVE 0x8000000000000000ull
#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_PERSISTENT_RETURN 0x1000000000000000ull
#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_STRIPED_COMPLETION 0x0800000000000000ull
#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_INDEX_SHIFT 8u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_LANE_MASK 0xffull
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_BITS 6u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_MASK \
    ((1u << SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_BITS) - 1u)
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_GENERATION_SHIFT \
    SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_BITS
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_RETURN_FLAG 0x80000000u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_GENERATION_MASK \
    ((1u << (31u - SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_BITS)) - 1u)

#if SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT > \
    (SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_MASK + 1u)
#error "doorbell immediate credit field is too small"
#endif

#define SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX 0xffffffffu
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_MAPPED_HOST 1u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_DEVICE_DIRECT 2u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_LIVE 0u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCING 1u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCED 2u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCE_FAILED 3u

typedef struct SparkHiddenSparkHostRdmaMemoryRegionDescriptor
{
    uint64_t address;
    uint64_t bytes;
    uint32_t rkey;
    uint32_t reserved;
} SparkHiddenSparkHostRdmaMemoryRegionDescriptor;

typedef struct SparkHiddenSparkHostRdmaControlMessage
{
    uint32_t magic;
    uint32_t version;
    uint32_t type;
    uint32_t status;
    uint64_t sequence_id;
    uint64_t token_index;
    uint32_t active_sequence_count;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
    uint32_t reserved;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor hidden;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor sideband;
} SparkHiddenSparkHostRdmaControlMessage;

typedef struct SparkHiddenSparkHostRdmaQueuePairWireInfo
{
    uint32_t qp_number;
    uint32_t packet_sequence_number;
    uint16_t lid;
    uint16_t memory_mode;
    uint8_t active_mtu;
    uint8_t reserved[3];
    uint8_t gid[sizeof(union ibv_gid)];
} SparkHiddenSparkHostRdmaQueuePairWireInfo;

typedef struct SparkHiddenSparkHostRdmaLane
{
    struct ibv_cq *completion_queue;
    struct ibv_qp *queue_pair;
    SparkHiddenSparkHostRdmaQueuePairWireInfo local_info;
    SparkHiddenSparkHostRdmaQueuePairWireInfo remote_info;
} SparkHiddenSparkHostRdmaLane;

typedef struct SparkHiddenSparkHostRdmaCachedMemoryRegion
{
    const void *cuda_visible_pointer;
    const void *pointer;
    uint64_t bytes;
    uint64_t last_use_epoch;
    /* NET-001: pins the slot against LRU eviction while in-flight work
     * still references the registration: posted send WRs (sender side)
     * or an advertised rkey a remote write may still target (receiver
     * side). Deregistering either would fault the wire, so only slots
     * with in_flight_count == 0 are eviction candidates. */
    uint32_t in_flight_count;
    struct ibv_mr *memory_region;
} SparkHiddenSparkHostRdmaCachedMemoryRegion;

typedef struct SparkHiddenSparkHostRdmaPendingReceive
{
    uint32_t active;
    uint32_t complete;
    uint32_t advertised;
    uint32_t completion_generation_tag;
    uint32_t visibility_flushed;
    uint32_t persistent_registered;
    uint32_t completion_published;
    uint32_t release_event_recorded;
    uint32_t receive_index;
    uint64_t generation;
    uint64_t returned_generation;
    /* NET-001: cached-region slots pinned while this receive's rkeys are
     * advertised to the peer; NO_INDEX when the region is absent. */
    uint32_t hidden_region_index;
    uint32_t sideband_region_index;
    SparkHiddenTransportPacket packet_template;
    SparkHiddenTransportPacket packet_snapshot;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor hidden_descriptor;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor sideband_descriptor;
    SparkStatus completion_status;
} SparkHiddenSparkHostRdmaPendingReceive;

typedef struct SparkHiddenSparkHostRdmaRemoteReceive
{
    uint32_t active;
    uint32_t used;
    uint32_t persistent;
    uint32_t receive_index;
    uint64_t generation;
    uint64_t returned_generation;
    uint64_t sequence_id;
    uint64_t token_index;
    uint32_t active_sequence_count;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor hidden_descriptor;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor sideband_descriptor;
} SparkHiddenSparkHostRdmaRemoteReceive;

typedef struct SparkHiddenSparkHostRdmaInflightSend
{
    uint32_t active;
    uint32_t complete;
    uint32_t remote_receive_index;
    uint32_t posted_lane_mask;
    uint32_t completed_lane_mask;
    uint32_t doorbell;
    /* NET-001: cached-region slots pinned until every posted WR for this
     * send completes; NO_INDEX when the region is absent. */
    uint32_t hidden_region_index;
    uint32_t sideband_region_index;
    SparkStatus status;
    uint64_t start_time_ns;
    uint8_t posted_wr_counts[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
    SparkHiddenTransportPacket packet_snapshot;
} SparkHiddenSparkHostRdmaInflightSend;

typedef struct SparkHiddenSparkHostRdmaStripedCompletion
{
    uint32_t active;
    uint32_t complete;
    uint32_t lane_index;
    uint32_t remote_receive_index;
    SparkStatus status;
    uint64_t service_time_ns;
    SparkHiddenTransportPacket packet_snapshot;
} SparkHiddenSparkHostRdmaStripedCompletion;

typedef struct SparkHiddenSparkHostRdmaInflightBatch
{
    uint32_t active;
    uint32_t packet_count;
    uint32_t posted_lane_mask;
    uint32_t completed_lane_mask;
    uint32_t send_indices[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT];
} SparkHiddenSparkHostRdmaInflightBatch;

typedef struct SparkHiddenSparkHostRdmaPreparedSend
{
    SparkHiddenSparkHostRdmaInflightSend *send;
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receive;
    struct ibv_mr *hidden_memory_region;
    struct ibv_mr *sideband_memory_region;
    void *hidden_local_pointer;
    void *sideband_local_pointer;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
} SparkHiddenSparkHostRdmaPreparedSend;

typedef struct SparkHiddenSparkHostRdmaState
{
    SparkHiddenTransportEndpoint endpoint;
    int32_t local_rank;
    int32_t source_rank;
    int32_t sink_rank;
    uint32_t is_sender;
    uint32_t lane_count;
    uint32_t control_port_base;
    uint32_t open_timeout_milli;
    uint64_t open_deadline_ns;
    uint8_t verbs_port;
    uint8_t active_mtu;
    int32_t gid_index;
    int listen_fd;
    int control_fd;
    int event_fd;
    uint32_t terminal_state;
    uint32_t debug_enabled;
    uint32_t memory_mode;
    uint32_t gpudirect_flush_required;
    char source_host[SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES];
    char sink_host[SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES];
    char verbs_device_name[SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_NAME_BYTES];
    struct ibv_context *verbs_context;
    struct ibv_pd *protection_domain;
    struct ibv_comp_channel *completion_channel;
    union ibv_gid local_gid;
    uint16_t local_lid;
    SparkHiddenSparkHostRdmaLane lanes[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
    SparkHiddenSparkHostRdmaCachedMemoryRegion cached_regions[SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT];
    uint64_t memory_region_epoch;
    SparkHiddenSparkHostRdmaPendingReceive pending_receives[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT];
    SparkHiddenSparkHostRdmaRemoteReceive remote_receives[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT];
    SparkHiddenSparkHostRdmaInflightSend inflight_sends[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT];
    SparkHiddenSparkHostRdmaStripedCompletion striped_completions[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT];
    SparkHiddenSparkHostRdmaInflightBatch inflight_batches[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_BATCH_COUNT];
    uint32_t outstanding_send_wr_counts[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
    cudaEvent_t send_ready_events[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT];
    uint32_t send_ready_recorded[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT];
    cudaEvent_t receive_release_events[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT];
    SparkHiddenSparkHostRdmaControlMessage control_queue[SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_QUEUE_DEPTH];
    uint32_t control_queue_head;
    uint32_t control_queue_count;
    uint32_t control_queue_write_offset;
    SparkHiddenTransportCompletionQueue completion_queue;
    uint32_t doorbell_max_bytes;
    uint64_t doorbell_send_count;
    uint64_t striped_send_count;
    uint64_t memory_region_cache_hit_count;
    uint64_t pointer_attribute_query_count;
    uint64_t memory_region_register_count;
    uint64_t memory_region_eviction_count;
    uint64_t asynchronous_send_count;
    uint64_t completed_send_count;
    uint64_t control_queue_busy_count;
    uint64_t mapped_host_zero_copy_transfer_count;
    uint64_t mapped_host_zero_copy_transfer_bytes;
    uint64_t gpudirect_transfer_count;
    uint64_t gpudirect_transfer_bytes;
} SparkHiddenSparkHostRdmaState;

typedef struct SparkHiddenSparkHostRdmaResolveRequest
{
    pthread_mutex_t mutex;
    pthread_t thread;
    uint32_t abandoned;
    uint32_t complete;
    int result_code;
    struct addrinfo hints;
    struct addrinfo *result;
    char host[SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES];
    char port[16u];
} SparkHiddenSparkHostRdmaResolveRequest;

static SparkStatus SparkHiddenSparkHostRdmaParseUintEnv(
    const char *name,
    uint32_t default_value,
    uint32_t *value_out)
{
    const char *text;
    char *end;
    unsigned long value;

    if (name == 0 || value_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    text = getenv(name);
    if (text == 0)
    {
        *value_out = default_value;
        return SPARK_STATUS_OK;
    }
    if (text[0] < '0' || text[0] > '9')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    end = 0;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value > 0xfffffffful)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *value_out = (uint32_t)value;
    return SPARK_STATUS_OK;
}

static int SparkHiddenSparkHostRdmaCloseFd(int fd)
{
    if (fd >= 0)
    {
        (void)close(fd);
    }
    return -1;
}

static void SparkHiddenSparkHostRdmaSignalEvent(SparkHiddenSparkHostRdmaState *state)
{
    ssize_t written;
    uint64_t value;

    if (state == 0 || state->event_fd < 0)
    {
        return;
    }
    value = 1u;
    written = write(state->event_fd, &value, sizeof(value));
    if (written != (ssize_t)sizeof(value))
    {
        return;
    }
}

static SparkStatus SparkHiddenSparkHostRdmaTerminalStatus(
    const SparkHiddenSparkHostRdmaState *state)
{
    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return __atomic_load_n(&state->terminal_state,__ATOMIC_ACQUIRE) ==
        SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_LIVE ?
        SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkHiddenSparkHostRdmaFenceSession(
    SparkHiddenSparkHostRdmaState *state)
{
    struct ibv_qp_attr attributes;
    SparkStatus status;
    uint32_t expected;
    uint32_t lane_index;
    uint32_t terminal_state;

    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    expected = SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_LIVE;
    if (!__atomic_compare_exchange_n(&state->terminal_state,&expected,
            SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCING,0,
            __ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))
    {
        if (expected == SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCED)
            return SPARK_STATUS_OK;
        if (expected ==
            SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCE_FAILED)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        return SPARK_STATUS_BUSY;
    }
    status = SPARK_STATUS_OK;
    memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_ERR;
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        if (state->lanes[lane_index].queue_pair != 0 &&
            ibv_modify_qp(state->lanes[lane_index].queue_pair,&attributes,
                IBV_QP_STATE) != 0)
        {
            status = SPARK_STATUS_IO_ERROR;
        }
    }
    if (state->control_fd >= 0 &&
        SparkHiddenTransportRdmaControlFenceSession(state->control_fd) !=
            SPARK_STATUS_OK)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    terminal_state = status == SPARK_STATUS_OK ?
        SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCED :
        SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCE_FAILED;
    __atomic_store_n(&state->terminal_state,terminal_state,__ATOMIC_RELEASE);
    SparkHiddenSparkHostRdmaSignalEvent(state);
    return status;
}

static void CUDART_CB SparkHiddenSparkHostRdmaSignalCudaReady(void *context)
{
    SparkHiddenSparkHostRdmaSignalEvent(
        (SparkHiddenSparkHostRdmaState *)context);
}

static uint64_t SparkHiddenSparkHostRdmaMonotonicNs(void)
{
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        return 0u;
    return ((uint64_t)value.tv_sec * 1000000000ull) +
        (uint64_t)value.tv_nsec;
}

static int SparkHiddenSparkHostRdmaDeadlinePollMilliseconds(
    uint64_t deadline_ns,
    uint32_t maximum_milli)
{
    uint64_t now_ns,remaining_milli;

    now_ns = SparkHiddenSparkHostRdmaMonotonicNs();
    if (now_ns == 0u || now_ns >= deadline_ns)
        return 0;
    remaining_milli = (deadline_ns - now_ns + 999999ull) / 1000000ull;
    if (maximum_milli != 0u && remaining_milli > maximum_milli)
        remaining_milli = maximum_milli;
    if (remaining_milli > (uint64_t)INT32_MAX)
        remaining_milli = INT32_MAX;
    return (int)remaining_milli;
}

static uint32_t SparkHiddenSparkHostRdmaPacketsMatch(
    const SparkHiddenTransportPacket *left,
    const SparkHiddenTransportPacket *right)
{
    return left != 0 && right != 0 &&
        left->sequence_id == right->sequence_id &&
        left->token_index == right->token_index &&
        left->active_sequence_count == right->active_sequence_count &&
        left->sideband_kind == right->sideband_kind &&
        left->sideband_bytes_per_sequence ==
            right->sideband_bytes_per_sequence;
}

static uint64_t SparkHiddenSparkHostRdmaBuildSendWorkRequestId(
    uint32_t send_index,
    uint32_t lane_index)
{
    return SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_SEND |
        ((uint64_t)send_index <<
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_INDEX_SHIFT) |
        (uint64_t)lane_index;
}

static uint64_t SparkHiddenSparkHostRdmaBuildBatchWorkRequestId(
    uint32_t batch_index,
    uint32_t lane_index)
{
    return SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_BATCH |
        ((uint64_t)batch_index <<
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_INDEX_SHIFT) |
        (uint64_t)lane_index;
}

static uint32_t SparkHiddenSparkHostRdmaSendIndexFromWorkRequestId(
    uint64_t work_request_id)
{
    return (uint32_t)((work_request_id &
        ~SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_SEND) >>
        SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_INDEX_SHIFT);
}

static uint32_t SparkHiddenSparkHostRdmaLaneFromWorkRequestId(
    uint64_t work_request_id)
{
    return (uint32_t)(work_request_id &
        SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_LANE_MASK);
}

static uint32_t SparkHiddenSparkHostRdmaBatchIndexFromWorkRequestId(
    uint64_t work_request_id)
{
    return (uint32_t)((work_request_id &
        ~SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_BATCH) >>
        SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_INDEX_SHIFT);
}

static void SparkHiddenSparkHostRdmaDrainEvent(SparkHiddenSparkHostRdmaState *state)
{
    uint64_t value;

    if (state == 0 || state->event_fd < 0)
    {
        return;
    }
    while (read(state->event_fd, &value, sizeof(value)) == sizeof(value))
    {
    }
}

static SparkStatus SparkHiddenSparkHostRdmaSetNonblocking(int fd)
{
    return SparkHiddenTransportRdmaControlSetNonblocking(fd);
}

static void SparkHiddenSparkHostRdmaReportProgressError(
    const SparkHiddenSparkHostRdmaState *state,
    const char *origin,
    SparkStatus status)
{
    if (state == 0 || origin == 0 || state->debug_enabled == 0u)
        return;
    fprintf(stderr,
        "hidden_spark_rdma_progress_error route=%s origin=%s status=%u errno=%d\n",
        state->endpoint.route_name,origin,(uint32_t)status,errno);
}

static void SparkHiddenSparkHostRdmaReportCompletionError(
    const SparkHiddenSparkHostRdmaState *state,
    uint32_t lane_index,
    const struct ibv_wc *completion,
    SparkStatus status)
{
    if (state == 0 || completion == 0 || state->debug_enabled == 0u)
        return;
    fprintf(stderr,
        "hidden_spark_rdma_completion_error route=%s lane=%u status=%u wc_status=%u opcode=%u flags=%u wr_id=%llu immediate=%u\n",
        state->endpoint.route_name,lane_index,(uint32_t)status,
        (uint32_t)completion->status,(uint32_t)completion->opcode,
        completion->wc_flags,(unsigned long long)completion->wr_id,
        ntohl(completion->imm_data));
}

static void SparkHiddenSparkHostRdmaReportControlError(
    const SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message,
    SparkStatus status)
{
    if (state == 0 || message == 0 || state->debug_enabled == 0u)
        return;
    fprintf(stderr,
        "hidden_spark_rdma_control_error route=%s status=%u type=%u reserved=%u sequence=%llu token=%llu active=%u\n",
        state->endpoint.route_name,(uint32_t)status,message->type,
        message->reserved,(unsigned long long)message->sequence_id,
        (unsigned long long)message->token_index,
        message->active_sequence_count);
}

static void SparkHiddenSparkHostRdmaConfigureSocket(int fd, uint32_t timeout_ms)
{
    struct timeval timeout;
    int value;

    if (fd < 0)
    {
        return;
    }
    value = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value));
    (void)setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &value, sizeof(value));
    memset(&timeout,0,sizeof(timeout));
    timeout.tv_sec = (time_t)(timeout_ms / 1000u);
    timeout.tv_usec = (suseconds_t)((timeout_ms % 1000u) * 1000u);
    (void)setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    (void)setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
}

static int SparkHiddenSparkHostRdmaListen(uint32_t port)
{
    struct sockaddr_in address;
    int fd;
    int value;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        return -1;
    }
    value = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &value, sizeof(value));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(fd, 1) != 0)
    {
        (void)close(fd);
        return -1;
    }
    return fd;
}

static SparkStatus SparkHiddenSparkHostRdmaReadFull(
    int fd,
    void *buffer,
    uint64_t bytes,
    uint64_t deadline_ns)
{
    return SparkHiddenTransportRdmaControlReadFullDeadline(
        fd,buffer,bytes,deadline_ns);
}

static SparkStatus SparkHiddenSparkHostRdmaWriteFull(
    int fd,
    const void *buffer,
    uint64_t bytes,
    uint64_t deadline_ns)
{
    return SparkHiddenTransportRdmaControlWriteFullDeadline(
        fd,buffer,bytes,deadline_ns);
}

static void *SparkHiddenSparkHostRdmaResolveHostMain(void *context)
{
    SparkHiddenSparkHostRdmaResolveRequest *request;
    struct addrinfo *result;
    int result_code;

    request = (SparkHiddenSparkHostRdmaResolveRequest *)context;
    result = 0;
    result_code = getaddrinfo(request->host,request->port,&request->hints,
        &result);
    (void)pthread_mutex_lock(&request->mutex);
    if (request->abandoned != 0u)
    {
        (void)pthread_mutex_unlock(&request->mutex);
        if (result != 0)
            freeaddrinfo(result);
        (void)pthread_mutex_destroy(&request->mutex);
        free(request);
        return 0;
    }
    request->result_code = result_code;
    request->result = result;
    request->complete = 1u;
    (void)pthread_mutex_unlock(&request->mutex);
    return 0;
}

static SparkStatus SparkHiddenSparkHostRdmaResolveHostDeadline(
    const char *host,
    const char *port,
    const struct addrinfo *hints,
    uint64_t deadline_ns,
    struct addrinfo **result_out)
{
    SparkHiddenSparkHostRdmaResolveRequest *request;
    struct addrinfo *result;
    pthread_t resolver_thread;
    uint64_t now_ns;
    int result_code,written;

    if (host == 0 || port == 0 || hints == 0 || deadline_ns == 0u ||
        result_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *result_out = 0;
    request = (SparkHiddenSparkHostRdmaResolveRequest *)calloc(
        1u,sizeof(*request));
    if (request == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    written = snprintf(request->host,sizeof(request->host),"%s",host);
    if (written < 0 || (uint32_t)written >= sizeof(request->host))
    {
        free(request);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    written = snprintf(request->port,sizeof(request->port),"%s",port);
    if (written < 0 || (uint32_t)written >= sizeof(request->port))
    {
        free(request);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    request->hints = *hints;
    if (pthread_mutex_init(&request->mutex,0) != 0)
    {
        free(request);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (pthread_create(&request->thread,0,
            SparkHiddenSparkHostRdmaResolveHostMain,request) != 0)
    {
        (void)pthread_mutex_destroy(&request->mutex);
        free(request);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    for (;;)
    {
        now_ns = SparkHiddenSparkHostRdmaMonotonicNs();
        (void)pthread_mutex_lock(&request->mutex);
        if (request->complete != 0u)
        {
            result_code = request->result_code;
            result = request->result;
            (void)pthread_mutex_unlock(&request->mutex);
            (void)pthread_join(request->thread,0);
            (void)pthread_mutex_destroy(&request->mutex);
            free(request);
            if (result_code != 0 || result == 0)
            {
                fprintf(stderr,
                    "hidden_spark_rdma_control_resolve_failed host=%s port=%s code=%d\n",
                    request->host,request->port,result_code);
                if (result != 0)
                    freeaddrinfo(result);
                return SPARK_STATUS_ROUTE_NOT_FOUND;
            }
            *result_out = result;
            return SPARK_STATUS_OK;
        }
        if (now_ns == 0u || now_ns >= deadline_ns)
        {
            request->abandoned = 1u;
            resolver_thread = request->thread;
            (void)pthread_mutex_unlock(&request->mutex);
            (void)pthread_detach(resolver_thread);
            return now_ns == 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_BUSY;
        }
        (void)pthread_mutex_unlock(&request->mutex);
        (void)poll(0,0,SparkHiddenSparkHostRdmaDeadlinePollMilliseconds(
            deadline_ns,1u));
    }
}

static SparkStatus SparkHiddenSparkHostRdmaConnectControl(
    SparkHiddenSparkHostRdmaState *state)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    struct pollfd listen_poll;
    char port_text[16u];
    uint64_t now_ns;
    int fd;
    int poll_result;
    const char *host;
    SparkStatus status;

    if (state == 0 || state->open_deadline_ns == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->is_sender == 0u)
    {
        state->listen_fd = SparkHiddenSparkHostRdmaListen(
            state->control_port_base + (uint32_t)state->sink_rank);
        if (state->listen_fd < 0)
        {
            fprintf(stderr,
                "hidden_spark_rdma_control_listen_failed port=%u route=%s errno=%d\n",
                state->control_port_base + (uint32_t)state->sink_rank,
                state->endpoint.route_name,errno);
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        status = SparkHiddenSparkHostRdmaSetNonblocking(state->listen_fd);
        if (status != SPARK_STATUS_OK)
            return status;
        for (;;)
        {
            now_ns = SparkHiddenSparkHostRdmaMonotonicNs();
            if (now_ns == 0u || now_ns >= state->open_deadline_ns)
            {
                fprintf(stderr,
                    "hidden_spark_rdma_open_timeout route=%s role=receiver port=%u waited_ms=%u\n",
                    state->endpoint.route_name,
                    state->control_port_base + (uint32_t)state->sink_rank,
                    state->open_timeout_milli);
                return SPARK_STATUS_BUSY;
            }
            memset(&listen_poll, 0, sizeof(listen_poll));
            listen_poll.fd = state->listen_fd;
            listen_poll.events = POLLIN;
            poll_result = poll(&listen_poll, 1,
                SparkHiddenSparkHostRdmaDeadlinePollMilliseconds(
                    state->open_deadline_ns,0u));
            if (poll_result > 0)
            {
                fd = accept(state->listen_fd, 0, 0);
                if (fd >= 0)
                    break;
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                return SPARK_STATUS_IO_ERROR;
            }
            if (poll_result < 0 && errno != EINTR)
            {
                return SPARK_STATUS_IO_ERROR;
            }
        }
        state->control_fd = fd;
        SparkHiddenSparkHostRdmaConfigureSocket(state->control_fd,
            state->open_timeout_milli);
        return SparkHiddenSparkHostRdmaSetNonblocking(state->control_fd);
    }

    host = state->sink_host;
    while (state->control_fd < 0)
    {
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        snprintf(port_text, sizeof(port_text), "%u",
            state->control_port_base + (uint32_t)state->sink_rank);
        result = 0;
        status = SparkHiddenSparkHostRdmaResolveHostDeadline(host,port_text,
            &hints,state->open_deadline_ns,&result);
        if (status == SPARK_STATUS_BUSY || status == SPARK_STATUS_IO_ERROR ||
            status == SPARK_STATUS_INTERNAL_ERROR ||
            status == SPARK_STATUS_CAPACITY_EXCEEDED)
            return status;
        if (status != SPARK_STATUS_OK)
        {
            (void)poll(0,0,
                SparkHiddenSparkHostRdmaDeadlinePollMilliseconds(
                    state->open_deadline_ns,
                    SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS));
        }
        else
        {
            for (entry = result; entry != 0 && state->control_fd < 0;
                 entry = entry->ai_next)
            {
                struct pollfd connect_poll;
                int socket_error;
                socklen_t socket_error_bytes;

                fd = socket(entry->ai_family, entry->ai_socktype,
                    entry->ai_protocol);
                if (fd < 0)
                {
                    continue;
                }
                status = SparkHiddenSparkHostRdmaSetNonblocking(fd);
                if (status != SPARK_STATUS_OK)
                {
                    (void)close(fd);
                    continue;
                }
                if (connect(fd, entry->ai_addr, entry->ai_addrlen) == 0)
                    state->control_fd = fd;
                else if (errno == EINPROGRESS || errno == EALREADY ||
                    errno == EWOULDBLOCK)
                {
                    now_ns = SparkHiddenSparkHostRdmaMonotonicNs();
                    if (now_ns != 0u && now_ns < state->open_deadline_ns)
                    {
                        memset(&connect_poll,0,sizeof(connect_poll));
                        connect_poll.fd = fd;
                        connect_poll.events = POLLOUT;
                        poll_result = poll(&connect_poll,1,
                            SparkHiddenSparkHostRdmaDeadlinePollMilliseconds(
                                state->open_deadline_ns,
                                SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS));
                        socket_error = 0;
                        socket_error_bytes = sizeof(socket_error);
                        if (poll_result > 0 &&
                            getsockopt(fd,SOL_SOCKET,SO_ERROR,&socket_error,
                                &socket_error_bytes) == 0 &&
                            socket_error == 0)
                            state->control_fd = fd;
                    }
                }
                if (state->control_fd < 0)
                    (void)close(fd);
                else
                    SparkHiddenSparkHostRdmaConfigureSocket(
                        state->control_fd,state->open_timeout_milli);
            }
            freeaddrinfo(result);
            if (state->control_fd < 0)
            {
                (void)poll(0,0,
                    SparkHiddenSparkHostRdmaDeadlinePollMilliseconds(
                        state->open_deadline_ns,
                        SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS));
            }
        }
        if (state->control_fd < 0 &&
            SparkHiddenSparkHostRdmaMonotonicNs() >= state->open_deadline_ns)
        {
            fprintf(stderr,
                "hidden_spark_rdma_open_timeout route=%s role=sender host=%s port=%u waited_ms=%u\n",
                state->endpoint.route_name,
                host,
                state->control_port_base + (uint32_t)state->sink_rank,
                state->open_timeout_milli);
            return SPARK_STATUS_BUSY;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaExchangeCompatibilityHello(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkHiddenTransportRdmaV4Identity identity;
    int written;

    if (state == 0 || state->control_fd < 0 || state->source_rank < 0 ||
        state->sink_rank < 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&identity,0,sizeof(identity));
    identity.magic = SPARK_HIDDEN_TRANSPORT_RDMA_CONTROL_MAGIC;
    identity.protocol_version = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_VERSION;
    identity.transport_abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    identity.descriptor_bytes = sizeof(identity);
    identity.sender_role = state->is_sender;
    identity.peer_sender_role = state->is_sender == 0u ? 1u : 0u;
    identity.local_rank = (uint32_t)state->local_rank;
    identity.peer_rank = state->is_sender != 0u ?
        (uint32_t)state->sink_rank : (uint32_t)state->source_rank;
    identity.source_rank = (uint32_t)state->source_rank;
    identity.sink_rank = (uint32_t)state->sink_rank;
    identity.control_port = state->control_port_base +
        (uint32_t)state->sink_rank;
    identity.hidden_dimension = state->endpoint.hidden_dimension;
    identity.bytes_per_sequence = state->endpoint.bytes_per_sequence;
    identity.max_active_sequence_count =
        state->endpoint.max_active_sequence_count;
    identity.persistent_credit_count =
        (state->endpoint.capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS) != 0u ?
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT : 0u;
    identity.lane_count = state->lane_count;
    identity.doorbell_max_bytes = state->doorbell_max_bytes;
    identity.memory_mode = state->memory_mode;
    identity.capability_flags = state->endpoint.capability_flags;
    identity.max_packet_bytes = state->endpoint.max_packet_bytes;
    identity.route_identifier = state->endpoint.route_identifier;
    written = snprintf(identity.transport_module_id,
        sizeof(identity.transport_module_id),"%s",
        state->endpoint.transport_module_id);
    if (written < 0 || (uint32_t)written >=
        sizeof(identity.transport_module_id))
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    written = snprintf(identity.route_name,sizeof(identity.route_name),"%s",
        state->endpoint.route_name);
    if (written < 0 || (uint32_t)written >= sizeof(identity.route_name))
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    written = snprintf(identity.source_host,sizeof(identity.source_host),"%s",
        state->source_host);
    if (written < 0 || (uint32_t)written >= sizeof(identity.source_host))
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    written = snprintf(identity.sink_host,sizeof(identity.sink_host),"%s",
        state->sink_host);
    if (written < 0 || (uint32_t)written >= sizeof(identity.sink_host))
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    return SparkHiddenTransportRdmaV4ExchangeCompatibilityHello(
        state->control_fd,state->open_deadline_ns,&identity);
}

static SparkStatus SparkHiddenSparkHostRdmaDeviceRateGbps(
    const char *device_name,
    uint8_t verbs_port,
    uint32_t *rate_gbps_out)
{
    char path[SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_RATE_PATH_BYTES];
    char rate_text[SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_RATE_TEXT_BYTES];
    char unit[16u];
    uint32_t rate_gbps;
    FILE *stream;
    int written;

    if (device_name == 0 || device_name[0] == '\0' || verbs_port == 0u ||
        rate_gbps_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    written = snprintf(path,sizeof(path),"%s/%s/ports/%u/rate",
        SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_INFINIBAND_SYSFS_PATH,
        device_name,(uint32_t)verbs_port);
    if (written < 0 || (uint32_t)written >= sizeof(path))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    stream = fopen(path,"r");
    if (stream == 0)
        return SPARK_STATUS_IO_ERROR;
    rate_text[0] = '\0';
    if (fgets(rate_text,sizeof(rate_text),stream) == 0)
    {
        (void)fclose(stream);
        return SPARK_STATUS_IO_ERROR;
    }
    (void)fclose(stream);
    unit[0] = '\0';
    if (sscanf(rate_text,"%u %15s",&rate_gbps,unit) != 2 ||
        strcmp(unit,"Gb/sec") != 0)
        return SPARK_STATUS_SCHEMA_ERROR;
    *rate_gbps_out = (uint32_t)rate_gbps;
    return SPARK_STATUS_OK;
}

static int SparkHiddenSparkHostRdmaPortIsActive(
    struct ibv_device *device,
    uint8_t verbs_port)
{
    struct ibv_context *probe;
    struct ibv_port_attr port_attributes;
    int active;

    probe = ibv_open_device(device);
    if (probe == 0)
    {
        return 0;
    }
    memset(&port_attributes, 0, sizeof(port_attributes));
    active = ibv_query_port(probe, verbs_port, &port_attributes) == 0 &&
        port_attributes.state == IBV_PORT_ACTIVE;
    (void)ibv_close_device(probe);
    return active;
}

static SparkStatus SparkHiddenSparkHostRdmaResolveIpv4(
    const char *host,
    struct in_addr *address_out)
{
    struct addrinfo hints;
    struct addrinfo *addresses;
    struct sockaddr_in *address;

    if (host == 0 || host[0] == '\0' || address_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (inet_pton(AF_INET,host,address_out) == 1)
        return SPARK_STATUS_OK;
    memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addresses = 0;
    if (getaddrinfo(host,0,&hints,&addresses) != 0 || addresses == 0)
    {
        fprintf(stderr,"hidden_spark_rdma_ipv4_resolve_failed host=%s\n",host);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    address = (struct sockaddr_in *)addresses->ai_addr;
    *address_out = address->sin_addr;
    freeaddrinfo(addresses);
    return SPARK_STATUS_OK;
}

static uint32_t SparkHiddenSparkHostRdmaGidMatchesIpv4(
    const union ibv_gid *gid,
    const struct in_addr *address)
{
    static const uint8_t mapped_prefix[12u] =
        {0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0xffu,0xffu};

    if (gid == 0 || address == 0 ||
        memcmp(gid->raw,mapped_prefix,sizeof(mapped_prefix)) != 0)
        return 0u;
    return memcmp(&gid->raw[12u],&address->s_addr,
        sizeof(address->s_addr)) == 0 ? 1u : 0u;
}

static uint32_t SparkHiddenSparkHostRdmaGidIsRocEv2(
    const char *device_name,
    uint8_t verbs_port,
    int32_t gid_index)
{
    char path[SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_RATE_PATH_BYTES];
    char type[SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_RATE_TEXT_BYTES];
    FILE *stream;
    int written;

    if (device_name == 0 || verbs_port == 0u || gid_index < 0)
        return 0u;
    written = snprintf(path,sizeof(path),"%s/%s/ports/%u/gid_attrs/types/%d",
        SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_INFINIBAND_SYSFS_PATH,
        device_name,(uint32_t)verbs_port,gid_index);
    if (written < 0 || (uint32_t)written >= sizeof(path))
        return 0u;
    stream = fopen(path,"r");
    if (stream == 0)
        return 0u;
    type[0] = '\0';
    if (fgets(type,sizeof(type),stream) == 0)
    {
        (void)fclose(stream);
        return 0u;
    }
    (void)fclose(stream);
    return strncmp(type,"RoCE v2",
        SPARK_HIDDEN_SPARK_HOST_RDMA_ROCE_V2_TEXT_BYTES) == 0 ? 1u : 0u;
}

static int32_t SparkHiddenSparkHostRdmaFindGidIndex(
    struct ibv_context *context,
    const char *device_name,
    uint8_t verbs_port,
    const struct in_addr *address)
{
    union ibv_gid gid;
    int32_t gid_index;

    if (context == 0 || device_name == 0 || verbs_port == 0u ||
        address == 0)
        return -1;
    for (gid_index=0;
        gid_index<(int32_t)SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_GID_COUNT;
        gid_index++)
    {
        if (ibv_query_gid(context,verbs_port,gid_index,&gid) == 0 &&
            SparkHiddenSparkHostRdmaGidMatchesIpv4(&gid,address) != 0u &&
            SparkHiddenSparkHostRdmaGidIsRocEv2(
                device_name,verbs_port,gid_index) != 0u)
            return gid_index;
    }
    return -1;
}

static SparkStatus SparkHiddenSparkHostRdmaDiscoverDevice(
    const char *local_host,
    uint8_t verbs_port,
    char *device_name,
    uint32_t device_name_bytes,
    int32_t *gid_index_out)
{
    struct ibv_device **devices;
    struct ibv_device *selected_device;
    struct ibv_context *probe;
    struct in_addr local_address;
    int32_t candidate_gid_index;
    int32_t selected_gid_index;
    uint32_t matching_count;
    uint32_t rate_gbps;
    SparkStatus rate_status;
    SparkStatus status;
    int count;
    int index;
    int written;

    if (local_host == 0 || verbs_port == 0u || device_name == 0 ||
        device_name_bytes == 0u || gid_index_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenSparkHostRdmaResolveIpv4(local_host,&local_address);
    if (status != SPARK_STATUS_OK)
        return status;
    devices = ibv_get_device_list(&count);
    if (devices == 0)
    {
        fprintf(stderr,"hidden_spark_rdma_list_null host=%s\n",local_host);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    if (count <= 0)
    {
        ibv_free_device_list(devices);
        fprintf(stderr,"hidden_spark_rdma_list_empty host=%s\n",local_host);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    selected_device = 0;
    selected_gid_index = -1;
    matching_count = 0u;
    for (index = 0; index < count; ++index)
    {
        if (SparkHiddenSparkHostRdmaPortIsActive(
                devices[index],verbs_port) == 0)
            continue;
        rate_status = SparkHiddenSparkHostRdmaDeviceRateGbps(
            ibv_get_device_name(devices[index]),verbs_port,&rate_gbps);
        if (rate_status != SPARK_STATUS_OK || rate_gbps <
            SPARK_HIDDEN_SPARK_HOST_RDMA_REQUIRED_LINK_RATE_GBPS)
            continue;
        probe = ibv_open_device(devices[index]);
        if (probe == 0)
            continue;
        candidate_gid_index = SparkHiddenSparkHostRdmaFindGidIndex(
            probe,ibv_get_device_name(devices[index]),verbs_port,
            &local_address);
        (void)ibv_close_device(probe);
        if (candidate_gid_index < 0)
            continue;
        selected_device = devices[index];
        selected_gid_index = candidate_gid_index;
        matching_count++;
    }
    if (matching_count != 1u || selected_device == 0)
    {
        fprintf(stderr,
            "hidden_spark_rdma_fabric_invalid local_host=%s matching_active_%ug_count=%u\n",
            local_host,
            SPARK_HIDDEN_SPARK_HOST_RDMA_REQUIRED_LINK_RATE_GBPS,
            matching_count);
        ibv_free_device_list(devices);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    written = snprintf(device_name,device_name_bytes,"%s",
        ibv_get_device_name(selected_device));
    if (written < 0 || (uint32_t)written >= device_name_bytes)
    {
        ibv_free_device_list(devices);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *gid_index_out = selected_gid_index;
    ibv_free_device_list(devices);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaOpenVerbsDevice(
    SparkHiddenSparkHostRdmaState *state)
{
    struct ibv_device **devices;
    struct ibv_device *selected_device;
    char device_name[SPARK_HIDDEN_SPARK_HOST_RDMA_DEVICE_NAME_BYTES];
    struct ibv_port_attr port_attributes;
    SparkStatus status;
    const char *local_host;
    int count;
    int index;

    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    local_host = state->is_sender != 0u ?
        state->source_host : state->sink_host;
    status = SparkHiddenSparkHostRdmaDiscoverDevice(local_host,
        state->verbs_port,device_name,sizeof(device_name),&state->gid_index);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "hidden_spark_rdma_discover_failed local_host=%s port=%u status=%d\n",
            local_host,(uint32_t)state->verbs_port,(int)status);
        return status;
    }
    devices = ibv_get_device_list(&count);
    if (devices == 0)
    {
        fprintf(stderr,"hidden_spark_rdma_device_list_null local_host=%s\n",local_host);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    if (count <= 0)
    {
        ibv_free_device_list(devices);
        fprintf(stderr,"hidden_spark_rdma_device_list_empty local_host=%s\n",local_host);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    selected_device = 0;
    for (index=0; index<count; index++)
        if (strcmp(ibv_get_device_name(devices[index]),device_name) == 0)
        {
            selected_device = devices[index];
            break;
        }
    if (selected_device == 0)
    {
        ibv_free_device_list(devices);
        fprintf(stderr,"hidden_spark_rdma_device_missing local_host=%s want=%s\n",local_host,device_name);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    (void)snprintf(state->verbs_device_name,
        sizeof(state->verbs_device_name),"%s",device_name);
    state->verbs_context = ibv_open_device(selected_device);
    ibv_free_device_list(devices);
    if (state->verbs_context == 0)
    {
        fprintf(stderr,"hidden_spark_rdma_open_failed local_host=%s device=%s\n",local_host,device_name);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->protection_domain = ibv_alloc_pd(state->verbs_context);
    if (state->protection_domain == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    if (ibv_query_port(state->verbs_context, state->verbs_port,
            &port_attributes) != 0 ||
        port_attributes.state != IBV_PORT_ACTIVE ||
        port_attributes.active_mtu < IBV_MTU_256 ||
        port_attributes.active_mtu > IBV_MTU_4096)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    state->local_lid = port_attributes.lid;
    state->active_mtu = (uint8_t)port_attributes.active_mtu;
    memset(&state->local_gid, 0, sizeof(state->local_gid));
    if (ibv_query_gid(state->verbs_context, state->verbs_port,
            state->gid_index, &state->local_gid) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    fprintf(stderr,
        "hidden_spark_rdma_fabric_ready local_host=%s device=%s port=%u gid_index=%d\n",
        local_host,state->verbs_device_name,(uint32_t)state->verbs_port,
        state->gid_index);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaModifyQueuePairToInit(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaLane *lane)
{
    struct ibv_qp_attr attributes;

    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_INIT;
    attributes.pkey_index = 0;
    attributes.port_num = state->verbs_port;
    attributes.qp_access_flags = IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(lane->queue_pair, &attributes,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                IBV_QP_ACCESS_FLAGS) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCreateCompletionChannel(
    SparkHiddenSparkHostRdmaState *state)
{
    if (state == 0 || state->verbs_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->completion_channel = ibv_create_comp_channel(state->verbs_context);
    if (state->completion_channel == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SparkHiddenSparkHostRdmaSetNonblocking(
        state->completion_channel->fd);
}

static uint32_t SparkHiddenSparkHostRdmaDoorbellLane(
    const SparkHiddenSparkHostRdmaState *state,
    uint32_t receive_index)
{
    if (state == 0 || state->lane_count <= 1u)
        return 0u;
    return receive_index % state->lane_count;
}

static SparkStatus SparkHiddenSparkHostRdmaPostDoorbellCredit(
    SparkHiddenSparkHostRdmaState *state,
    uint32_t credit_index)
{
    struct ibv_recv_wr work_request;
    struct ibv_recv_wr *bad_work_request;

    if (state == 0 || credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(&work_request,0,sizeof(work_request));
    work_request.wr_id = SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_DOORBELL_RECEIVE |
        (uint64_t)credit_index;
    bad_work_request = 0;
    if (ibv_post_recv(state->lanes[SparkHiddenSparkHostRdmaDoorbellLane(
            state,credit_index)].queue_pair,&work_request,
            &bad_work_request) != 0)
        return SPARK_STATUS_IO_ERROR;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPrepostDoorbellCredits(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkStatus status;
    uint32_t credit_index;

    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (credit_index = 0u;
         credit_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++credit_index)
    {
        status = SparkHiddenSparkHostRdmaPostDoorbellCredit(
            state,credit_index);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCreateQueuePairs(SparkHiddenSparkHostRdmaState *state)
{
    struct ibv_qp_init_attr init_attributes;
    uint32_t lane_index;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        SparkHiddenSparkHostRdmaLane *lane;

        lane = &state->lanes[lane_index];
        lane->completion_queue = ibv_create_cq(state->verbs_context,
            SPARK_HIDDEN_SPARK_HOST_RDMA_COMPLETION_QUEUE_DEPTH,0,
            state->completion_channel, 0);
        if (lane->completion_queue == 0)
        {
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        memset(&init_attributes, 0, sizeof(init_attributes));
        init_attributes.send_cq = lane->completion_queue;
        init_attributes.recv_cq = lane->completion_queue;
        init_attributes.qp_type = IBV_QPT_RC;
        init_attributes.cap.max_send_wr =
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE;
        init_attributes.cap.max_recv_wr =
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
        init_attributes.cap.max_send_sge = 1;
        init_attributes.cap.max_recv_sge = 1;
        lane->queue_pair = ibv_create_qp(state->protection_domain,
            &init_attributes);
        if (lane->queue_pair == 0)
        {
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        if (SparkHiddenSparkHostRdmaModifyQueuePairToInit(state, lane) !=
            SPARK_STATUS_OK)
        {
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        if (ibv_req_notify_cq(lane->completion_queue, 0) != 0)
        {
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        memset(&lane->local_info, 0, sizeof(lane->local_info));
        lane->local_info.qp_number = lane->queue_pair->qp_num;
        lane->local_info.packet_sequence_number =
            0x778800u + ((uint32_t)getpid() & 0xfffu) + lane_index;
        lane->local_info.lid = state->local_lid;
        lane->local_info.memory_mode = (uint16_t)state->memory_mode;
        lane->local_info.active_mtu = state->active_mtu;
        memcpy(lane->local_info.gid, state->local_gid.raw,
            sizeof(lane->local_info.gid));
    }
    return SparkHiddenSparkHostRdmaPrepostDoorbellCredits(state);
}

static SparkStatus SparkHiddenSparkHostRdmaExchangeQueuePairInfo(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkHiddenSparkHostRdmaQueuePairWireInfo local_infos[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
    SparkHiddenSparkHostRdmaQueuePairWireInfo remote_infos[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
    uint32_t bytes;
    uint32_t lane_index;
    SparkStatus status;

    memset(local_infos, 0, sizeof(local_infos));
    memset(remote_infos, 0, sizeof(remote_infos));
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        local_infos[lane_index] = state->lanes[lane_index].local_info;
    }
    bytes = state->lane_count *
        (uint32_t)sizeof(SparkHiddenSparkHostRdmaQueuePairWireInfo);
    if (state->is_sender != 0u)
    {
        status = SparkHiddenSparkHostRdmaWriteFull(state->control_fd,
            &state->lane_count,sizeof(state->lane_count),
            state->open_deadline_ns);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenSparkHostRdmaWriteFull(state->control_fd,
            local_infos,bytes,state->open_deadline_ns);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenSparkHostRdmaReadFull(state->control_fd,
            remote_infos,bytes,state->open_deadline_ns);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    else
    {
        uint32_t remote_lane_count;

        remote_lane_count = 0u;
        status = SparkHiddenSparkHostRdmaReadFull(state->control_fd,
            &remote_lane_count,sizeof(remote_lane_count),
            state->open_deadline_ns);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (remote_lane_count != state->lane_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkHiddenSparkHostRdmaReadFull(state->control_fd,
            remote_infos,bytes,state->open_deadline_ns);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenSparkHostRdmaWriteFull(state->control_fd,
            local_infos,bytes,state->open_deadline_ns);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        if (remote_infos[lane_index].memory_mode != state->memory_mode ||
            remote_infos[lane_index].active_mtu < IBV_MTU_256 ||
            remote_infos[lane_index].active_mtu > IBV_MTU_4096 ||
            remote_infos[lane_index].reserved[0] != 0u ||
            remote_infos[lane_index].reserved[1] != 0u ||
            remote_infos[lane_index].reserved[2] != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        state->lanes[lane_index].remote_info = remote_infos[lane_index];
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaModifyQueuePairToReady(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaLane *lane)
{
    struct ibv_qp_attr attributes;
    union ibv_gid remote_gid;
    uint8_t path_mtu;

    memcpy(remote_gid.raw, lane->remote_info.gid, sizeof(remote_gid.raw));
    path_mtu = lane->local_info.active_mtu < lane->remote_info.active_mtu ?
        lane->local_info.active_mtu : lane->remote_info.active_mtu;
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = (enum ibv_mtu)path_mtu;
    attributes.dest_qp_num = lane->remote_info.qp_number;
    attributes.rq_psn = lane->remote_info.packet_sequence_number;
    attributes.max_dest_rd_atomic = 1;
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.grh.dgid = remote_gid;
    attributes.ah_attr.grh.sgid_index = state->gid_index;
    attributes.ah_attr.grh.hop_limit = 1;
    attributes.ah_attr.dlid = lane->remote_info.lid;
    attributes.ah_attr.sl = 0;
    attributes.ah_attr.src_path_bits = 0;
    attributes.ah_attr.port_num = state->verbs_port;
    if (ibv_modify_qp(lane->queue_pair, &attributes,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }

    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTS;
    attributes.timeout = 14;
    attributes.retry_cnt = 7;
    attributes.rnr_retry = 7;
    attributes.sq_psn = lane->local_info.packet_sequence_number;
    attributes.max_rd_atomic = 1;
    if (ibv_modify_qp(lane->queue_pair, &attributes,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                IBV_QP_MAX_QP_RD_ATOMIC) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkHiddenSparkHostRdmaPathMtuBytes(
    const SparkHiddenSparkHostRdmaLane *lane)
{
    uint8_t path_mtu;

    if (lane == 0)
        return 0u;
    path_mtu = lane->local_info.active_mtu < lane->remote_info.active_mtu ?
        lane->local_info.active_mtu : lane->remote_info.active_mtu;
    return 128u << path_mtu;
}

static SparkStatus SparkHiddenSparkHostRdmaReadyQueuePairs(SparkHiddenSparkHostRdmaState *state)
{
    uint32_t lane_index;
    SparkStatus status;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        status = SparkHiddenSparkHostRdmaModifyQueuePairToReady(state,
            &state->lanes[lane_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaDeregisterCachedMemoryRegions(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT; ++index)
    {
        if (state->cached_regions[index].memory_region != 0)
        {
            ibv_dereg_mr(state->cached_regions[index].memory_region);
        }
        memset(&state->cached_regions[index], 0,
            sizeof(state->cached_regions[index]));
    }
}
static uint32_t SparkHiddenSparkHostRdmaCachedRegionIndex(
    const SparkHiddenSparkHostRdmaState *state,
    const struct ibv_mr *memory_region)
{
    uint32_t index;

    if (state == 0 || memory_region == 0)
    {
        return SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
    }
    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT;
         ++index)
    {
        if (state->cached_regions[index].memory_region == memory_region)
        {
            return index;
        }
    }
    return SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
}

static void SparkHiddenSparkHostRdmaAcquireSendRegions(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send,
    struct ibv_mr *hidden_memory_region,
    struct ibv_mr *sideband_memory_region)
{
    send->hidden_region_index = SparkHiddenSparkHostRdmaCachedRegionIndex(
        state, hidden_memory_region);
    send->sideband_region_index = SparkHiddenSparkHostRdmaCachedRegionIndex(
        state, sideband_memory_region);
    if (send->hidden_region_index != SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX)
    {
        state->cached_regions[send->hidden_region_index].in_flight_count +=
            1u;
    }
    if (send->sideband_region_index != SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX)
    {
        state->cached_regions[send->sideband_region_index].in_flight_count +=
            1u;
    }
}

static void SparkHiddenSparkHostRdmaReleaseSendRegions(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send)
{
    if (state == 0 || send == 0)
    {
        return;
    }
    if (send->hidden_region_index <
            SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT &&
        state->cached_regions[send->hidden_region_index].in_flight_count !=
            0u)
    {
        state->cached_regions[send->hidden_region_index].in_flight_count -=
            1u;
    }
    if (send->sideband_region_index <
            SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT &&
        state->cached_regions[send->sideband_region_index].in_flight_count !=
            0u)
    {
        state->cached_regions[send->sideband_region_index].in_flight_count -=
            1u;
    }
    send->hidden_region_index = SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
    send->sideband_region_index = SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
}


static SparkStatus SparkHiddenSparkHostRdmaResolveRegistrationPointer(
    const SparkHiddenSparkHostRdmaState *state,
    const void *cuda_visible_pointer,
    uint64_t bytes,
    void **registration_pointer_out)
{
    cudaPointerAttributes attributes;
    cudaError_t cuda_status;

    if (state == 0 || cuda_visible_pointer == 0 || bytes == 0u ||
        registration_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *registration_pointer_out = 0;
    memset(&attributes, 0, sizeof(attributes));
    cuda_status = cudaPointerGetAttributes(&attributes, cuda_visible_pointer);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
    if (state->memory_mode ==
        SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_DEVICE_DIRECT)
    {
        if (attributes.type != cudaMemoryTypeDevice ||
            attributes.devicePointer == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        *registration_pointer_out = attributes.devicePointer;
        return SPARK_STATUS_OK;
    }
    if (state->memory_mode !=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_MAPPED_HOST ||
        attributes.type != cudaMemoryTypeHost ||
        attributes.hostPointer == 0 || attributes.devicePointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *registration_pointer_out = attributes.hostPointer;
#else
    if (state->memory_mode !=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_MAPPED_HOST ||
        attributes.memoryType != cudaMemoryTypeHost)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *registration_pointer_out = (void *)cuda_visible_pointer;
#endif
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaConfigureGpudirectVisibility(
    SparkHiddenSparkHostRdmaState *state)
{
    if (state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->gpudirect_flush_required = 0u;
    if (state->memory_mode !=
        SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_DEVICE_DIRECT)
    {
        return SPARK_STATUS_OK;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11030
    int cuda_device;
    int native_write_ordering;
    int flush_options;

    if (cudaGetDevice(&cuda_device) != cudaSuccess ||
        cudaDeviceGetAttribute(
            &native_write_ordering,
            cudaDevAttrGPUDirectRDMAWritesOrdering,
            cuda_device) != cudaSuccess)
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (native_write_ordering >=
        (int)cudaGPUDirectRDMAWritesOrderingOwner)
    {
        return SPARK_STATUS_OK;
    }
    if (cudaDeviceGetAttribute(
            &flush_options,
            cudaDevAttrGPUDirectRDMAFlushWritesOptions,
            cuda_device) != cudaSuccess ||
        (flush_options &
            (int)cudaFlushGPUDirectRDMAWritesOptionHost) == 0)
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    state->gpudirect_flush_required = 1u;
    return SPARK_STATUS_OK;
#else
    return SPARK_STATUS_MODULE_NOT_VALIDATED;
#endif
}

static SparkStatus SparkHiddenSparkHostRdmaFlushGpudirectWrites(
    const SparkHiddenSparkHostRdmaState *state)
{
    if (state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->memory_mode !=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_DEVICE_DIRECT ||
        state->gpudirect_flush_required == 0u)
    {
        return SPARK_STATUS_OK;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11030
    if (cudaDeviceFlushGPUDirectRDMAWrites(
            cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
            cudaFlushGPUDirectRDMAWritesToOwner) != cudaSuccess)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
#else
    return SPARK_STATUS_MODULE_NOT_VALIDATED;
#endif
}

static SparkStatus SparkHiddenSparkHostRdmaGetCachedMemoryRegion(
    SparkHiddenSparkHostRdmaState *state,
    const void *pointer,
    uint64_t bytes,
    struct ibv_mr **memory_region_out,
    void **registered_pointer_out)
{
    uint32_t index;
    uint32_t free_index;
    int access_flags;
    struct ibv_mr *memory_region;
    void *registration_pointer;
    SparkStatus status;

    if (state == 0 || pointer == 0 || bytes == 0u ||
        memory_region_out == 0 || registered_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->memory_region_epoch += 1u;
    free_index = SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT;
    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT; ++index)
    {
        if (state->cached_regions[index].memory_region == 0)
        {
            free_index = index;
            break;
        }
        if (state->cached_regions[index].cuda_visible_pointer == pointer &&
            state->cached_regions[index].bytes == bytes)
        {
            state->cached_regions[index].last_use_epoch =
                state->memory_region_epoch;
            *memory_region_out = state->cached_regions[index].memory_region;
            *registered_pointer_out =
                (void *)state->cached_regions[index].pointer;
            state->memory_region_cache_hit_count += 1u;
            return SPARK_STATUS_OK;
        }
    }
    state->pointer_attribute_query_count += 1u;
    status = SparkHiddenSparkHostRdmaResolveRegistrationPointer(
        state, pointer, bytes, &registration_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *registered_pointer_out = registration_pointer;
    for (index = 0u; index < free_index; ++index)
    {
        if (state->cached_regions[index].pointer == registration_pointer &&
            state->cached_regions[index].bytes == bytes)
        {
            state->cached_regions[index].cuda_visible_pointer = pointer;
            state->cached_regions[index].last_use_epoch =
                state->memory_region_epoch;
            *memory_region_out = state->cached_regions[index].memory_region;
            state->memory_region_cache_hit_count += 1u;
            return SPARK_STATUS_OK;
        }
    }
    if (free_index == SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT)
    {
        /* NET-001: the cache used to return a permanent
         * CAPACITY_EXCEEDED once all slots filled, even though
         * last_use_epoch was already tracked. Evict the
         * least-recently-used registration that no in-flight work
         * references; slots pinned by posted send WRs or advertised
         * receive rkeys are never evicted because deregistering them
         * would fault in-flight RDMA. */
        uint64_t oldest_epoch;
        uint32_t victim_index;

        oldest_epoch = UINT64_MAX;
        victim_index = SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT;
        for (index = 0u;
             index < SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT; ++index)
        {
            if (state->cached_regions[index].in_flight_count != 0u)
            {
                continue;
            }
            if (state->cached_regions[index].last_use_epoch < oldest_epoch)
            {
                oldest_epoch =
                    state->cached_regions[index].last_use_epoch;
                victim_index = index;
            }
        }
        if (victim_index == SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT)
        {
            /* Every registration is pinned by in-flight work: transient
             * back-pressure the caller may retry, not exhaustion. */
            return SPARK_STATUS_BUSY;
        }
        ibv_dereg_mr(state->cached_regions[victim_index].memory_region);
        memset(&state->cached_regions[victim_index], 0,
            sizeof(state->cached_regions[victim_index]));
        state->memory_region_eviction_count += 1u;
        free_index = victim_index;
    }
    access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    errno = 0;
    memory_region = ibv_reg_mr(state->protection_domain, registration_pointer,
        (size_t)bytes, access_flags);
    if (memory_region == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    state->cached_regions[free_index].cuda_visible_pointer = pointer;
    state->cached_regions[free_index].pointer = registration_pointer;
    state->cached_regions[free_index].bytes = bytes;
    state->cached_regions[free_index].last_use_epoch =
        state->memory_region_epoch;
    state->cached_regions[free_index].memory_region = memory_region;
    state->memory_region_register_count += 1u;
    *memory_region_out = memory_region;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaRegisterReceiveRegion(
    SparkHiddenSparkHostRdmaState *state,
    const void *pointer,
    uint64_t bytes,
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor *descriptor,
    uint32_t *region_index_out)
{
    struct ibv_mr *memory_region;
    void *registration_pointer;
    uint32_t region_index;
    SparkStatus status;

    if (descriptor == 0 || region_index_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(descriptor, 0, sizeof(*descriptor));
    *region_index_out = SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
    if (bytes == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (state == 0 || pointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaGetCachedMemoryRegion(state, pointer,
        bytes, &memory_region, &registration_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    /* NET-001: the descriptor's rkey stays reachable by the peer until
     * the receive completes, so pin the cached region against LRU
     * eviction for exactly that long. The cache owns the MR lifetime;
     * callers must release through
     * SparkHiddenSparkHostRdmaReleasePendingReceive, never ibv_dereg_mr. */
    region_index = SparkHiddenSparkHostRdmaCachedRegionIndex(state,
        memory_region);
    if (region_index == SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->cached_regions[region_index].in_flight_count += 1u;
    *region_index_out = region_index;
    descriptor->address = (uint64_t)(uintptr_t)registration_pointer;
    descriptor->bytes = bytes;
    descriptor->rkey = memory_region->rkey;
    return SPARK_STATUS_OK;
}

static uint64_t SparkHiddenSparkHostRdmaPacketHiddenBytes(
    const SparkHiddenTransportPacket *packet)
{
    return (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
}

static uint64_t SparkHiddenSparkHostRdmaPacketSidebandBytes(
    const SparkHiddenTransportPacket *packet)
{
    if ((packet->flags &
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD) == 0u)
    {
        return 0u;
    }
    return (uint64_t)packet->sideband_bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
}

static SparkStatus SparkHiddenSparkHostRdmaPreparePacketMemory(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    struct ibv_mr **hidden_memory_region_out,
    void **hidden_local_pointer_out,
    struct ibv_mr **sideband_memory_region_out,
    void **sideband_local_pointer_out)
{
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint64_t transfer_bytes;
    SparkStatus status;

    if (state == 0 || packet == 0 ||
        hidden_memory_region_out == 0 || hidden_local_pointer_out == 0 ||
        sideband_memory_region_out == 0 || sideband_local_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *hidden_memory_region_out = 0;
    *hidden_local_pointer_out = 0;
    *sideband_memory_region_out = 0;
    *sideband_local_pointer_out = 0;
    if ((packet->flags &
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    hidden_bytes = SparkHiddenSparkHostRdmaPacketHiddenBytes(packet);
    sideband_bytes = SparkHiddenSparkHostRdmaPacketSidebandBytes(packet);
    transfer_bytes = hidden_bytes + sideband_bytes;
    if (transfer_bytes < hidden_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    status = SparkHiddenSparkHostRdmaGetCachedMemoryRegion(
        state,
        packet->hidden_bf16,
        hidden_bytes,
        hidden_memory_region_out,
        hidden_local_pointer_out);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sideband_bytes != 0u)
    {
        status = SparkHiddenSparkHostRdmaGetCachedMemoryRegion(
            state,
            packet->sideband_payload,
            sideband_bytes,
            sideband_memory_region_out,
            sideband_local_pointer_out);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    if (state->memory_mode ==
        SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_DEVICE_DIRECT)
    {
        state->gpudirect_transfer_count += 1u;
        state->gpudirect_transfer_bytes += transfer_bytes;
    }
    else
    {
        state->mapped_host_zero_copy_transfer_count += 1u;
        state->mapped_host_zero_copy_transfer_bytes += transfer_bytes;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkHiddenSparkHostRdmaPacketUsesDoorbell(
    const SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint64_t bytes;

    if (state == 0 || packet == 0 || state->doorbell_max_bytes == 0u)
    {
        return 0u;
    }
    bytes = SparkHiddenSparkHostRdmaPacketHiddenBytes(packet) +
        SparkHiddenSparkHostRdmaPacketSidebandBytes(packet);
    return bytes <= (uint64_t)state->doorbell_max_bytes ? 1u : 0u;
}

static uint32_t SparkHiddenSparkHostRdmaRemoteReceiveMatchesPacket(
    const SparkHiddenSparkHostRdmaRemoteReceive *receive,
    const SparkHiddenTransportPacket *packet)
{
    return receive != 0 && packet != 0 && receive->active != 0u &&
        ((receive->persistent != 0u && receive->used != 0u) ||
         (receive->persistent == 0u && receive->used == 0u)) &&
        receive->sequence_id == packet->sequence_id &&
        receive->token_index == packet->token_index &&
        receive->active_sequence_count == packet->active_sequence_count &&
        receive->sideband_kind == packet->sideband_kind &&
        receive->sideband_bytes_per_sequence == packet->sideband_bytes_per_sequence;
}

static uint32_t SparkHiddenSparkHostRdmaPendingReceiveMatchesPacket(
    const SparkHiddenSparkHostRdmaPendingReceive *receive,
    const SparkHiddenTransportPacket *packet)
{
    return receive != 0 && packet != 0 && receive->active != 0u &&
        receive->packet_snapshot.sequence_id == packet->sequence_id &&
        receive->packet_snapshot.token_index == packet->token_index &&
        receive->packet_snapshot.active_sequence_count ==
            packet->active_sequence_count &&
        receive->packet_snapshot.sideband_kind == packet->sideband_kind &&
        receive->packet_snapshot.sideband_bytes_per_sequence ==
            packet->sideband_bytes_per_sequence;
}

static SparkHiddenSparkHostRdmaPendingReceive *SparkHiddenSparkHostRdmaFindPendingReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        if (SparkHiddenSparkHostRdmaPendingReceiveMatchesPacket(
                &state->pending_receives[index], packet) != 0u)
        {
            return &state->pending_receives[index];
        }
    }
    return 0;
}

static SparkHiddenSparkHostRdmaPendingReceive *SparkHiddenSparkHostRdmaReservePendingReceive(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        if (state->pending_receives[index].active == 0u)
        {
            memset(&state->pending_receives[index], 0,
                sizeof(state->pending_receives[index]));
            state->pending_receives[index].active = 1u;
            state->pending_receives[index].receive_index = index;
            state->pending_receives[index].hidden_region_index =
                SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
            state->pending_receives[index].sideband_region_index =
                SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
            return &state->pending_receives[index];
        }
    }
    return 0;
}

static SparkHiddenSparkHostRdmaRemoteReceive *SparkHiddenSparkHostRdmaFindRemoteReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT;
         ++index)
    {
        if (SparkHiddenSparkHostRdmaRemoteReceiveMatchesPacket(
                &state->remote_receives[index], packet) != 0u)
        {
            return &state->remote_receives[index];
        }
    }
    return 0;
}

static void SparkHiddenSparkHostRdmaReleaseRemoteReceive(
    SparkHiddenSparkHostRdmaState *state,
    uint32_t index)
{
    if (state == 0 ||
        index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
        return;
    if (state->remote_receives[index].persistent == 0u)
    {
        memset(&state->remote_receives[index], 0,
            sizeof(state->remote_receives[index]));
    }
    state->send_ready_recorded[index] = 0u;
}

static SparkStatus SparkHiddenSparkHostRdmaInsertPersistentRemoteReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    SparkHiddenSparkHostRdmaRemoteReceive *receive;
    uint32_t index;

    if (state == 0 || message == 0 || state->is_sender == 0u ||
        message->reserved == 0u || message->reserved >
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    index = message->reserved - 1u;
    receive = &state->remote_receives[index];
    if (receive->active != 0u)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    memset(receive,0,sizeof(*receive));
    receive->active = 1u;
    receive->persistent = 1u;
    receive->receive_index = index;
    receive->sideband_kind = message->sideband_kind;
    receive->sideband_bytes_per_sequence =
        message->sideband_bytes_per_sequence;
    receive->hidden_descriptor = message->hidden;
    receive->sideband_descriptor = message->sideband;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplyPersistentReturnValue(
    SparkHiddenSparkHostRdmaState *state,
    uint32_t index,
    uint64_t generation)
{
    SparkHiddenSparkHostRdmaRemoteReceive *receive;

    if (state == 0 || state->is_sender == 0u || index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT ||
        generation == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    receive = &state->remote_receives[index];
    if (receive->active == 0u || receive->persistent == 0u ||
        receive->used == 0u || receive->generation != generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    receive->used = 0u;
    receive->returned_generation = generation;
    receive->generation = 0u;
    receive->sequence_id = 0u;
    receive->token_index = 0u;
    receive->active_sequence_count = 0u;
    state->send_ready_recorded[index] = 0u;
    SparkHiddenSparkHostRdmaSignalEvent(state);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplyPersistentReturn(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    if (message == 0 || message->reserved == 0u || message->reserved >
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SparkHiddenSparkHostRdmaApplyPersistentReturnValue(
        state,message->reserved - 1u,message->sequence_id);
}

static SparkStatus SparkHiddenSparkHostRdmaInsertRemoteReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    uint32_t index;
    SparkHiddenSparkHostRdmaRemoteReceive *receive;

    if (message->reserved >
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT;
         ++index)
    {
        if (state->remote_receives[index].active == 0u)
        {
            receive = &state->remote_receives[index];
            memset(receive, 0, sizeof(*receive));
            state->send_ready_recorded[index] = 0u;
            receive->active = 1u;
            receive->receive_index = message->reserved == 0u ?
                SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX : message->reserved - 1u;
            receive->sequence_id = message->sequence_id;
            receive->token_index = message->token_index;
            receive->active_sequence_count = message->active_sequence_count;
            receive->sideband_kind = message->sideband_kind;
            receive->sideband_bytes_per_sequence =
                message->sideband_bytes_per_sequence;
            receive->hidden_descriptor = message->hidden;
            receive->sideband_descriptor = message->sideband;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_BUSY;
}

static SparkStatus SparkHiddenSparkHostRdmaFlushControlQueue(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkHiddenSparkHostRdmaControlMessage *message;
    const uint8_t *cursor;
    uint32_t remaining;
    ssize_t result;

    if (state == 0 || state->control_fd < 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    while (state->control_queue_count != 0u)
    {
        message = &state->control_queue[state->control_queue_head];
        cursor = (const uint8_t *)message +
            state->control_queue_write_offset;
        remaining = (uint32_t)sizeof(*message) -
            state->control_queue_write_offset;
        result = send(state->control_fd,cursor,remaining,MSG_NOSIGNAL);
        if (result < 0 && errno == EINTR)
            continue;
        if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            state->control_queue_busy_count += 1u;
            return SPARK_STATUS_BUSY;
        }
        if (result <= 0)
            return SPARK_STATUS_IO_ERROR;
        state->control_queue_write_offset += (uint32_t)result;
        if (state->control_queue_write_offset != sizeof(*message))
            continue;
        state->control_queue_write_offset = 0u;
        state->control_queue_head = (state->control_queue_head + 1u) %
            SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_QUEUE_DEPTH;
        state->control_queue_count -= 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaEnqueueControlMessage(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    SparkHiddenSparkHostRdmaControlMessage *queued_message;
    uint32_t tail;

    if (state == 0 || message == 0 || state->control_fd < 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->control_queue_count >=
        SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_QUEUE_DEPTH)
        return SPARK_STATUS_BUSY;
    tail = (state->control_queue_head + state->control_queue_count) %
        SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_QUEUE_DEPTH;
    queued_message = &state->control_queue[tail];
    *queued_message = *message;
    queued_message->magic = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_MAGIC;
    queued_message->version = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_VERSION;
    state->control_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaQueueControlMessage(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    SparkStatus status;

    status = SparkHiddenSparkHostRdmaEnqueueControlMessage(state,message);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaFlushControlQueue(state);
    return status == SPARK_STATUS_BUSY ? SPARK_STATUS_OK : status;
}

static uint32_t SparkHiddenSparkHostRdmaControlQueueHasRoom(
    const SparkHiddenSparkHostRdmaState *state,
    uint32_t message_count)
{
    return state != 0 &&
        message_count <= SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_QUEUE_DEPTH -
            state->control_queue_count;
}

static SparkStatus SparkHiddenSparkHostRdmaWriteControlMessage(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaControlMessage *message)
{
    return SparkHiddenSparkHostRdmaQueueControlMessage(state, message);
}

static SparkStatus SparkHiddenSparkHostRdmaReadControlMessageNonblocking(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaControlMessage *message)
{
    ssize_t result;
    uint8_t *cursor;
    uint64_t done;

    if (state == 0 || message == 0 || state->control_fd < 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    {
        int available_bytes;

        available_bytes = 0;
        if (ioctl(state->control_fd, FIONREAD, &available_bytes) != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        if (available_bytes < (int)sizeof(*message))
        {
            return SPARK_STATUS_BUSY;
        }
    }
    cursor = (uint8_t *)message;
    done = 0u;
    while (done < sizeof(*message))
    {
        result = read(state->control_fd, cursor + done,
            (size_t)(sizeof(*message) - done));
        if (result < 0 && errno == EINTR)
        {
            continue;
        }
        if (result <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        done += (uint64_t)result;
    }
    if (message->magic != SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_MAGIC ||
        message->version != SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_VERSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkHiddenSparkHostRdmaDoorbellGenerationTag(
    uint64_t generation)
{
    return (uint32_t)generation &
        SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_GENERATION_MASK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplyDoorbellCompletion(
    SparkHiddenSparkHostRdmaState *state,
    const struct ibv_wc *work_completion)
{
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receive;
    SparkStatus status;
    uint64_t receive_credit_index;
    uint32_t completion_receive_index;
    uint32_t generation_tag;
    uint32_t immediate;
    uint32_t receive_index;

    if (state == 0 || work_completion == 0 ||
        (work_completion->opcode != IBV_WC_RECV_RDMA_WITH_IMM &&
         work_completion->opcode != IBV_WC_RECV) ||
        (work_completion->wc_flags & IBV_WC_WITH_IMM) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    /* FIFO receive WQEs are replenished lane credits. Immediate data owns
     * the logical receive identity and its persistent generation tag. */
    immediate = ntohl(work_completion->imm_data);
    receive_index = immediate &
        SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_MASK;
    completion_receive_index = (uint32_t)(work_completion->wr_id &
        ~SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_DOORBELL_RECEIVE);
    generation_tag = (immediate >>
        SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_GENERATION_SHIFT) &
        SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_GENERATION_MASK;
    receive_credit_index = completion_receive_index;
    if (receive_index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT ||
        (work_completion->wr_id &
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_DOORBELL_RECEIVE) == 0u ||
        receive_credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    status = SparkHiddenSparkHostRdmaPostDoorbellCredit(
        state,(uint32_t)receive_credit_index);
    if (status != SPARK_STATUS_OK)
        return status;
    if ((immediate &
            SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_RETURN_FLAG) != 0u)
    {
        if (state->is_sender == 0u || work_completion->opcode != IBV_WC_RECV)
            return SPARK_STATUS_IO_ERROR;
        remote_receive = &state->remote_receives[receive_index];
        if (remote_receive->active == 0u ||
            remote_receive->persistent == 0u ||
            remote_receive->used == 0u ||
            generation_tag != SparkHiddenSparkHostRdmaDoorbellGenerationTag(
                remote_receive->generation))
            return SPARK_STATUS_VALIDATION_FAILED;
        if (state->debug_enabled != 0u)
        {
            fprintf(stderr,
                "hidden_spark_rdma_persistent_return_receive route=%s credit=%u generation=%llu\n",
                state->endpoint.route_name,receive_index,
                (unsigned long long)remote_receive->generation);
        }
        return SparkHiddenSparkHostRdmaApplyPersistentReturnValue(
            state,receive_index,remote_receive->generation);
    }
    if (state->is_sender != 0u ||
        (work_completion->opcode != IBV_WC_RECV_RDMA_WITH_IMM &&
         work_completion->opcode != IBV_WC_RECV))
        return SPARK_STATUS_IO_ERROR;
    receive = &state->pending_receives[receive_index];
    if (receive->complete != 0u ||
        (receive->persistent_registered == 0u &&
         (receive->active == 0u || generation_tag != 0u)) ||
        (receive->persistent_registered != 0u && receive->active != 0u &&
         generation_tag != SparkHiddenSparkHostRdmaDoorbellGenerationTag(
            receive->generation)))
    {
        return SPARK_STATUS_IO_ERROR;
    }
    receive->complete = 1u;
    receive->completion_generation_tag = generation_tag;
    receive->completion_status = SPARK_STATUS_OK;
    SparkHiddenSparkHostRdmaSignalEvent(state);
    return SPARK_STATUS_OK;
}

static SparkHiddenSparkHostRdmaInflightSend *
SparkHiddenSparkHostRdmaFindInflightSend(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t index;

    for (index = 0u;
         index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT;
         ++index)
    {
        if (state->inflight_sends[index].active != 0u &&
            SparkHiddenSparkHostRdmaPacketsMatch(
                &state->inflight_sends[index].packet_snapshot,
                packet) != 0u)
            return &state->inflight_sends[index];
    }
    return 0;
}

static SparkHiddenSparkHostRdmaInflightSend *
SparkHiddenSparkHostRdmaReserveInflightSend(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t index;

    for (index = 0u;
         index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT;
         ++index)
    {
        if (state->inflight_sends[index].active == 0u)
        {
            memset(&state->inflight_sends[index], 0,
                sizeof(state->inflight_sends[index]));
            state->inflight_sends[index].active = 1u;
            state->inflight_sends[index].status = SPARK_STATUS_OK;
            state->inflight_sends[index].hidden_region_index =
                SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
            state->inflight_sends[index].sideband_region_index =
                SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
            return &state->inflight_sends[index];
        }
    }
    return 0;
}

static SparkHiddenSparkHostRdmaStripedCompletion *
SparkHiddenSparkHostRdmaReserveStripedCompletion(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t index;

    for (index=0u;
         index<SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT; index++)
    {
        if (state->striped_completions[index].active == 0u)
        {
            memset(&state->striped_completions[index],0,
                sizeof(state->striped_completions[index]));
            state->striped_completions[index].active = 1u;
            state->striped_completions[index].status = SPARK_STATUS_OK;
            return &state->striped_completions[index];
        }
    }
    return 0;
}

static SparkHiddenSparkHostRdmaInflightBatch *
SparkHiddenSparkHostRdmaReserveInflightBatch(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t index;

    for (index = 0u;
         index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_BATCH_COUNT;
         ++index)
    {
        if (state->inflight_batches[index].active == 0u)
        {
            memset(&state->inflight_batches[index],0,
                sizeof(state->inflight_batches[index]));
            state->inflight_batches[index].active = 1u;
            return &state->inflight_batches[index];
        }
    }
    return 0;
}

static SparkStatus SparkHiddenSparkHostRdmaRetireSendLane(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send,
    uint32_t lane_index)
{
    uint32_t posted_count;

    if (state == 0 || send == 0 || lane_index >= state->lane_count)
        return SPARK_STATUS_INVALID_ARGUMENT;
    posted_count = send->posted_wr_counts[lane_index];
    if (posted_count == 0u ||
        state->outstanding_send_wr_counts[lane_index] < posted_count)
        return SPARK_STATUS_IO_ERROR;
    state->outstanding_send_wr_counts[lane_index] -= posted_count;
    send->posted_wr_counts[lane_index] = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplySendCompletion(
    SparkHiddenSparkHostRdmaState *state,
    const struct ibv_wc *work_completion)
{
    SparkHiddenSparkHostRdmaInflightSend *send;
    SparkStatus status;
    uint32_t lane_index;
    uint32_t send_index;
    uint32_t lane_mask;

    send_index = SparkHiddenSparkHostRdmaSendIndexFromWorkRequestId(
        work_completion->wr_id);
    lane_index = SparkHiddenSparkHostRdmaLaneFromWorkRequestId(
        work_completion->wr_id);
    if (send_index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT ||
        lane_index >= state->lane_count)
        return SPARK_STATUS_IO_ERROR;
    send = &state->inflight_sends[send_index];
    lane_mask = 1u << lane_index;
    if (send->active == 0u ||
        (send->posted_lane_mask & lane_mask) == 0u ||
        (send->completed_lane_mask & lane_mask) != 0u)
        return SPARK_STATUS_IO_ERROR;
    status = SparkHiddenSparkHostRdmaRetireSendLane(
        state,send,lane_index);
    if (status != SPARK_STATUS_OK)
        return status;
    if (work_completion->status != IBV_WC_SUCCESS)
        send->status = SPARK_STATUS_IO_ERROR;
    send->completed_lane_mask |= lane_mask;
    if (send->completed_lane_mask == send->posted_lane_mask)
    {
        send->complete = 1u;
        SparkHiddenSparkHostRdmaSignalEvent(state);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplyBatchCompletion(
    SparkHiddenSparkHostRdmaState *state,
    const struct ibv_wc *work_completion)
{
    SparkHiddenSparkHostRdmaInflightBatch *batch;
    SparkHiddenSparkHostRdmaInflightSend *send;
    uint32_t batch_index;
    uint32_t lane_index;
    uint32_t lane_mask;
    uint32_t packet_index;
    uint32_t send_index;
    uint32_t posted_count;
    uint32_t total_posted_count;

    batch_index = SparkHiddenSparkHostRdmaBatchIndexFromWorkRequestId(
        work_completion->wr_id);
    lane_index = SparkHiddenSparkHostRdmaLaneFromWorkRequestId(
        work_completion->wr_id);
    if (batch_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_BATCH_COUNT ||
        lane_index >= state->lane_count)
        return SPARK_STATUS_IO_ERROR;
    batch = &state->inflight_batches[batch_index];
    lane_mask = 1u << lane_index;
    if (batch->active == 0u ||
        (batch->posted_lane_mask & lane_mask) == 0u ||
        (batch->completed_lane_mask & lane_mask) != 0u)
        return SPARK_STATUS_IO_ERROR;
    total_posted_count = 0u;
    for (packet_index = 0u;
         packet_index < batch->packet_count;
         ++packet_index)
    {
        send_index = batch->send_indices[packet_index];
        if (send_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT)
            return SPARK_STATUS_IO_ERROR;
        send = &state->inflight_sends[send_index];
        if (send->active == 0u)
            return SPARK_STATUS_IO_ERROR;
        posted_count = send->posted_wr_counts[lane_index];
        if ((send->posted_lane_mask & lane_mask) != 0u &&
            posted_count == 0u)
            return SPARK_STATUS_IO_ERROR;
        total_posted_count += posted_count;
    }
    if (state->outstanding_send_wr_counts[lane_index] <
        total_posted_count)
        return SPARK_STATUS_IO_ERROR;
    state->outstanding_send_wr_counts[lane_index] -= total_posted_count;
    for (packet_index = 0u;
         packet_index < batch->packet_count;
         ++packet_index)
    {
        send_index = batch->send_indices[packet_index];
        if (send_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT)
            return SPARK_STATUS_IO_ERROR;
        send = &state->inflight_sends[send_index];
        if ((send->posted_lane_mask & lane_mask) == 0u)
            continue;
        send->posted_wr_counts[lane_index] = 0u;
        if (work_completion->status != IBV_WC_SUCCESS)
            send->status = SPARK_STATUS_IO_ERROR;
        send->completed_lane_mask |= lane_mask;
        if (send->completed_lane_mask == send->posted_lane_mask)
            send->complete = 1u;
    }
    batch->completed_lane_mask |= lane_mask;
    if (batch->completed_lane_mask == batch->posted_lane_mask)
        memset(batch,0,sizeof(*batch));
    SparkHiddenSparkHostRdmaSignalEvent(state);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplyPersistentReturnSendCompletion(
    SparkHiddenSparkHostRdmaState *state,
    const struct ibv_wc *work_completion)
{
    uint32_t credit_index;
    uint32_t lane_index;

    if (state == 0 || work_completion == 0 || state->is_sender != 0u ||
        work_completion->status != IBV_WC_SUCCESS ||
        work_completion->opcode != IBV_WC_SEND)
        return SPARK_STATUS_IO_ERROR;
    credit_index = (uint32_t)(work_completion->wr_id &
        SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_CREDIT_MASK);
    lane_index = SparkHiddenSparkHostRdmaDoorbellLane(state,credit_index);
    if (credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT ||
        state->outstanding_send_wr_counts[lane_index] == 0u)
        return SPARK_STATUS_IO_ERROR;
    state->outstanding_send_wr_counts[lane_index] -= 1u;
    SparkHiddenSparkHostRdmaSignalEvent(state);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplyStripedCompletionSendCompletion(
    SparkHiddenSparkHostRdmaState *state,
    const struct ibv_wc *work_completion)
{
    SparkHiddenSparkHostRdmaStripedCompletion *completion;
    uint32_t lane_index;
    uint32_t send_index;

    if (state == 0 || work_completion == 0 || state->is_sender == 0u ||
        work_completion->status != IBV_WC_SUCCESS ||
        work_completion->opcode != IBV_WC_SEND)
        return SPARK_STATUS_IO_ERROR;
    send_index = (uint32_t)((work_completion->wr_id &
        ~SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_STRIPED_COMPLETION) >>
        SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_INDEX_SHIFT);
    lane_index = SparkHiddenSparkHostRdmaLaneFromWorkRequestId(
        work_completion->wr_id);
    if (send_index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT ||
        lane_index >= state->lane_count ||
        state->outstanding_send_wr_counts[lane_index] == 0u)
        return SPARK_STATUS_IO_ERROR;
    completion = &state->striped_completions[send_index];
    if (completion->active == 0u || completion->complete != 0u ||
        completion->lane_index != lane_index)
        return SPARK_STATUS_IO_ERROR;
    state->outstanding_send_wr_counts[lane_index] -= 1u;
    completion->complete = 1u;
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_spark_rdma_striped_fence_complete route=%s slot=%u lane=%u sequence=%llu token=%llu\n",
            state->endpoint.route_name,send_index,lane_index,
            (unsigned long long)completion->packet_snapshot.sequence_id,
            (unsigned long long)completion->packet_snapshot.token_index);
    }
    SparkHiddenSparkHostRdmaSignalEvent(state);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaApplyWorkCompletion(
    SparkHiddenSparkHostRdmaState *state,
    const struct ibv_wc *work_completion)
{
    if ((work_completion->wr_id &
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_DOORBELL_RECEIVE) != 0u)
    {
        if (work_completion->status != IBV_WC_SUCCESS)
            return SPARK_STATUS_IO_ERROR;
        return SparkHiddenSparkHostRdmaApplyDoorbellCompletion(state,
            work_completion);
    }
    if ((work_completion->wr_id &
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_PERSISTENT_RETURN) != 0u)
        return SparkHiddenSparkHostRdmaApplyPersistentReturnSendCompletion(
            state,work_completion);
    if ((work_completion->wr_id &
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_STRIPED_COMPLETION) != 0u)
        return SparkHiddenSparkHostRdmaApplyStripedCompletionSendCompletion(
            state,work_completion);
    if ((work_completion->wr_id &
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_BATCH) != 0u)
    {
        return SparkHiddenSparkHostRdmaApplyBatchCompletion(
            state,work_completion);
    }
    if ((work_completion->wr_id &
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_SEND) != 0u)
    {
        return SparkHiddenSparkHostRdmaApplySendCompletion(
            state,work_completion);
    }
    return SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkHiddenSparkHostRdmaPollCompletionQueue(
    SparkHiddenSparkHostRdmaState *state,
    uint32_t lane_index)
{
    /* NET-004: drain the CQ in batches instead of one ibv_poll_cq call
     * per completion; per-WC call overhead dominated the pump loop for
     * small doorbell packets (vLLM/FlashInfer poll their CQs the same
     * way). A short final batch means the queue is empty. */
    struct ibv_wc work_completions[
        SPARK_HIDDEN_SPARK_HOST_RDMA_COMPLETION_POLL_BATCH_COUNT];
    SparkStatus status;
    uint32_t completion_index;
    int result;

    while (1)
    {
        memset(work_completions, 0, sizeof(work_completions));
        result = ibv_poll_cq(state->lanes[lane_index].completion_queue,
            (int)SPARK_HIDDEN_SPARK_HOST_RDMA_COMPLETION_POLL_BATCH_COUNT,
            work_completions);
        if (result < 0)
        {
            SparkHiddenSparkHostRdmaReportProgressError(
                state,"poll_cq",SPARK_STATUS_IO_ERROR);
            return SPARK_STATUS_IO_ERROR;
        }
        if (result == 0)
        {
            return SPARK_STATUS_OK;
        }
        for (completion_index = 0u; completion_index < (uint32_t)result;
             ++completion_index)
        {
            status = SparkHiddenSparkHostRdmaApplyWorkCompletion(
                state,&work_completions[completion_index]);
            if (status != SPARK_STATUS_OK ||
                work_completions[completion_index].status != IBV_WC_SUCCESS)
            {
                SparkHiddenSparkHostRdmaReportCompletionError(
                    state,lane_index,&work_completions[completion_index],status);
            }
            if (status != SPARK_STATUS_OK)
                return status;
        }
        if (result <
            (int)SPARK_HIDDEN_SPARK_HOST_RDMA_COMPLETION_POLL_BATCH_COUNT)
        {
            return SPARK_STATUS_OK;
        }
    }
}

static SparkStatus SparkHiddenSparkHostRdmaPollCompletionQueues(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t lane_index;
    SparkStatus status;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        status = SparkHiddenSparkHostRdmaPollCompletionQueue(state, lane_index);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPumpDoorbells(
    SparkHiddenSparkHostRdmaState *state)
{
    return SparkHiddenSparkHostRdmaPollCompletionQueues(state);
}

static void SparkHiddenSparkHostRdmaMarkPendingReceiveComplete(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    SparkHiddenTransportPacket *packet;
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        packet = &state->pending_receives[index].packet_snapshot;
        if (state->pending_receives[index].active != 0u &&
            packet->sequence_id == message->sequence_id &&
            packet->token_index == message->token_index &&
            packet->active_sequence_count == message->active_sequence_count &&
            packet->sideband_kind == message->sideband_kind &&
            packet->sideband_bytes_per_sequence ==
                message->sideband_bytes_per_sequence)
        {
            state->pending_receives[index].complete = 1u;
            state->pending_receives[index].completion_status =
                (SparkStatus)message->status;
            SparkHiddenSparkHostRdmaSignalEvent(state);
            return;
        }
    }
}

static SparkStatus SparkHiddenSparkHostRdmaPumpControl(SparkHiddenSparkHostRdmaState *state)
{
    SparkStatus status;
    SparkHiddenSparkHostRdmaControlMessage message;

    if (state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaFlushControlQueue(state);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        return status;
    while (1)
    {
        memset(&message, 0, sizeof(message));
        status = SparkHiddenSparkHostRdmaReadControlMessageNonblocking(state, &message);
        if (status == SPARK_STATUS_BUSY)
        {
            return SPARK_STATUS_OK;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (message.type == SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY)
        {
            status = SparkHiddenSparkHostRdmaInsertRemoteReceive(state, &message);
            if (status != SPARK_STATUS_OK)
            {
                SparkHiddenSparkHostRdmaReportControlError(
                    state,&message,status);
                return status;
            }
            SparkHiddenSparkHostRdmaSignalEvent(state);
        }
        else if (message.type ==
            SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_PERSISTENT_ADVERTISE)
        {
            status = SparkHiddenSparkHostRdmaInsertPersistentRemoteReceive(
                state,&message);
            if (status != SPARK_STATUS_OK)
            {
                SparkHiddenSparkHostRdmaReportControlError(
                    state,&message,status);
                return status;
            }
            SparkHiddenSparkHostRdmaSignalEvent(state);
        }
        else if (message.type ==
            SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_PERSISTENT_RETURN)
        {
            status = SparkHiddenSparkHostRdmaApplyPersistentReturn(
                state,&message);
            if (status != SPARK_STATUS_OK)
            {
                SparkHiddenSparkHostRdmaReportControlError(
                    state,&message,status);
                return status;
            }
        }
        else if (message.type == SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_TRANSFER_COMPLETE)
        {
            SparkHiddenSparkHostRdmaMarkPendingReceiveComplete(state, &message);
        }
        else
        {
            SparkHiddenSparkHostRdmaReportControlError(
                state,&message,SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
}

static SparkStatus SparkHiddenSparkHostRdmaPumpProgress(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkStatus status;

    status = SparkHiddenSparkHostRdmaTerminalStatus(state);
    if (status != SPARK_STATUS_OK)
        return status;
    SparkHiddenSparkHostRdmaDrainEvent(state);
    status = SparkHiddenSparkHostRdmaPumpDoorbells(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReportProgressError(
            state,"doorbells",status);
        (void)SparkHiddenSparkHostRdmaFenceSession(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaPumpControl(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReportProgressError(state,"control",status);
        (void)SparkHiddenSparkHostRdmaFenceSession(state);
    }
    return status;
}

static uint32_t SparkHiddenSparkHostRdmaCompletionQueueHasRoom(
    const SparkHiddenSparkHostRdmaState *state,
    uint32_t completion_count)
{
    return state != 0 &&
        completion_count <= SPARK_HIDDEN_TRANSPORT_COMPLETION_QUEUE_DEPTH -
            state->completion_queue.count;
}

static SparkStatus SparkHiddenSparkHostRdmaBuildCompletion(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    SparkStatus status,
    uint64_t service_time_ns)
{
    SparkStatus queue_status;
    queue_status = SparkHiddenTransportCompletionQueuePushPacket(
        &state->completion_queue,packet,status,service_time_ns);
    if (queue_status == SPARK_STATUS_OK)
        SparkHiddenSparkHostRdmaSignalEvent(state);
    return queue_status;
}

static SparkStatus SparkHiddenSparkHostRdmaBuildReceiveReadyMessage(
    const SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    const SparkHiddenSparkHostRdmaPendingReceive *receive,
    SparkHiddenSparkHostRdmaControlMessage *message)
{
    if (state == 0 || packet == 0 || receive == 0 || message == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(message,0,sizeof(*message));
    message->type = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY;
    message->sequence_id = packet->sequence_id;
    message->token_index = packet->token_index;
    message->active_sequence_count = packet->active_sequence_count;
    message->sideband_kind = packet->sideband_kind;
    message->sideband_bytes_per_sequence =
        packet->sideband_bytes_per_sequence;
    message->reserved = SparkHiddenSparkHostRdmaPacketUsesDoorbell(
        state,packet) != 0u ? receive->receive_index + 1u : 0u;
    message->hidden = receive->hidden_descriptor;
    message->sideband = receive->sideband_descriptor;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaAdvertiseReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    SparkHiddenSparkHostRdmaPendingReceive *receive)
{
    SparkHiddenSparkHostRdmaControlMessage message;
    SparkStatus status;

    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenSparkHostRdmaBuildReceiveReadyMessage(
        state,packet,receive,&message);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaWriteControlMessage(state,&message);
    if (status == SPARK_STATUS_OK)
        receive->advertised = 1u;
    return status;
}

static void SparkHiddenSparkHostRdmaReleasePendingReceive(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaPendingReceive *receive)
{
    if (receive == 0)
    {
        return;
    }
    /* NET-001: drop the eviction pins taken by
     * SparkHiddenSparkHostRdmaRegisterReceiveRegion; the MR cache owns
     * the registrations themselves. */
    if (state != 0)
    {
        if (receive->hidden_region_index <
                SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT &&
            state->cached_regions[receive->hidden_region_index]
                .in_flight_count != 0u)
        {
            state->cached_regions[receive->hidden_region_index]
                .in_flight_count -= 1u;
        }
        if (receive->sideband_region_index <
                SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT &&
            state->cached_regions[receive->sideband_region_index]
                .in_flight_count != 0u)
        {
            state->cached_regions[receive->sideband_region_index]
                .in_flight_count -= 1u;
        }
    }
    memset(receive, 0, sizeof(*receive));
}

static SparkStatus SparkHiddenSparkHostRdmaCreatePendingReceive(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenTransportPacket *packet,
    SparkHiddenSparkHostRdmaPendingReceive **receive_out)
{
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkStatus status;
    uint64_t hidden_capacity_bytes;
    uint64_t sideband_capacity_bytes;

    if (state == 0 || packet == 0 || receive_out == 0 ||
        state->is_sender != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *receive_out = 0;
    receive = SparkHiddenSparkHostRdmaReservePendingReceive(state);
    if (receive == 0)
        return SPARK_STATUS_BUSY;
    receive->packet_snapshot = *packet;
    hidden_capacity_bytes = (uint64_t)state->endpoint.bytes_per_sequence *
        (uint64_t)state->endpoint.max_active_sequence_count;
    sideband_capacity_bytes =
        (uint64_t)packet->sideband_bytes_per_sequence *
        (uint64_t)state->endpoint.max_active_sequence_count;
    status = SparkHiddenSparkHostRdmaRegisterReceiveRegion(state,
        packet->hidden_bf16, hidden_capacity_bytes,
        &receive->hidden_descriptor, &receive->hidden_region_index);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReleasePendingReceive(state, receive);
        return status;
    }
    status = SparkHiddenSparkHostRdmaRegisterReceiveRegion(state,
        packet->sideband_payload, sideband_capacity_bytes,
        &receive->sideband_descriptor, &receive->sideband_region_index);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReleasePendingReceive(state, receive);
        return status;
    }
    *receive_out = receive;
    /* Advertising the registered destination accepts the receive. Poll
     * owns finalization after the peer signals transfer completion. */
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPreparePendingReceive(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenTransportPacket *packet,
    SparkHiddenSparkHostRdmaPendingReceive **receive_out)
{
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkStatus status;

    if (state == 0 || packet == 0 || receive_out == 0 ||
        state->is_sender != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    receive = SparkHiddenSparkHostRdmaFindPendingReceive(state,packet);
    if (receive == 0)
    {
        status = SparkHiddenSparkHostRdmaCreatePendingReceive(
            state,packet,&receive);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    if (receive->advertised == 0u)
    {
        status = SparkHiddenSparkHostRdmaAdvertiseReceive(
            state,packet,receive);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    *receive_out = receive;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaFinalizePendingReceive(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaPendingReceive *receive)
{
    SparkStatus status;

    if (state == 0 || receive == 0 || receive->active == 0u ||
        receive->complete == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkHiddenTransportCompletionQueueIsFull(
            &state->completion_queue) != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if (receive->completion_status == SPARK_STATUS_OK &&
        receive->visibility_flushed == 0u)
    {
        status = SparkHiddenSparkHostRdmaFlushGpudirectWrites(state);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        receive->visibility_flushed = 1u;
    }
    status = SparkHiddenSparkHostRdmaBuildCompletion(
        state,
        &receive->packet_snapshot,
        receive->completion_status,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = receive->completion_status;
    if (receive->persistent_registered != 0u)
    {
        receive->completion_published = 1u;
        return status;
    }
    SparkHiddenSparkHostRdmaReleasePendingReceive(state, receive);
    return status;
}

static SparkStatus SparkHiddenSparkHostRdmaPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacket(&state->endpoint,packet);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaPumpProgress(state);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaPreparePendingReceive(
        state,packet,&receive);
    return status;
}

static SparkStatus SparkHiddenSparkHostRdmaAdvertiseReceiveBatch(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaPendingReceive **receives,
    uint32_t packet_count)
{
    SparkHiddenSparkHostRdmaControlMessage messages[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT];
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkStatus status;
    uint32_t message_count;
    uint32_t packet_index;

    message_count = 0u;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        receive = receives[packet_index];
        if (receive->advertised != 0u)
            continue;
        status = SparkHiddenSparkHostRdmaBuildReceiveReadyMessage(
            state,&receive->packet_snapshot,receive,&messages[message_count]);
        if (status != SPARK_STATUS_OK)
            return status;
        message_count += 1u;
    }
    if (message_count == 0u)
        return SPARK_STATUS_OK;
    if (SparkHiddenSparkHostRdmaControlQueueHasRoom(
            state,message_count) == 0u)
        return SPARK_STATUS_BUSY;
    for (packet_index = 0u; packet_index < message_count; ++packet_index)
    {
        status = SparkHiddenSparkHostRdmaEnqueueControlMessage(
            state,&messages[packet_index]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
        receives[packet_index]->advertised = 1u;
    status = SparkHiddenSparkHostRdmaFlushControlQueue(state);
    return status == SPARK_STATUS_BUSY ? SPARK_STATUS_OK : status;
}

static SparkStatus SparkHiddenSparkHostRdmaPostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkHiddenSparkHostRdmaPendingReceive *receives[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT];
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;
    uint64_t receive_mask;
    uint32_t packet_index;
    uint32_t receive_index;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packets == 0 || packet_count == 0u ||
        packet_count >
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT ||
        state->is_sender != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacketBatch(
        &state->endpoint,packets,packet_count);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaPumpProgress(state);
    if (status != SPARK_STATUS_OK)
        return status;
    receive_mask = 0u;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        receives[packet_index] =
            SparkHiddenSparkHostRdmaFindPendingReceive(
                state,&packets[packet_index]);
        if (receives[packet_index] == 0)
        {
            status = SparkHiddenSparkHostRdmaCreatePendingReceive(
                state,&packets[packet_index],&receives[packet_index]);
            if (status != SPARK_STATUS_OK)
                return status;
        }
        receive_index = (uint32_t)(receives[packet_index] -
            state->pending_receives);
        if (receive_index >=
                SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT ||
            (receive_mask & (1ull << receive_index)) != 0u)
            return SPARK_STATUS_DUPLICATE;
        receive_mask |= 1ull << receive_index;
    }
    status = SparkHiddenSparkHostRdmaAdvertiseReceiveBatch(
        state,receives,packet_count);
    return status;
}

static SparkStatus SparkHiddenSparkHostRdmaBuildLaneWrite(
    SparkHiddenSparkHostRdmaState *state,
    const void *local_pointer,
    uint64_t local_bytes,
    const SparkHiddenSparkHostRdmaMemoryRegionDescriptor *remote_descriptor,
    struct ibv_mr *local_memory_region,
    uint32_t lane_index,
    uint32_t striped,
    struct ibv_sge *scatter_gather,
    struct ibv_send_wr *work_request,
    uint32_t *write_present_out)
{
    SparkMemlinkTransferPartition partition;
    SparkStatus status;

    *write_present_out = 0u;
    if (local_bytes == 0u)
        return SPARK_STATUS_OK;
    if (state == 0 || local_pointer == 0 || remote_descriptor == 0 ||
        local_memory_region == 0 || local_bytes > remote_descriptor->bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (striped != 0u)
        status = SparkMemlinkBuildTransferPartition(
            local_bytes,state->lane_count,lane_index,&partition);
    else
    {
        /* Non-striped (doorbell) writes go whole to the caller-selected
         * lane; callers only invoke this on the packet's assigned
         * doorbell lane (NET-003). */
        partition.offset = 0u;
        partition.byte_count = local_bytes;
        status = SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_OK)
        return status;
    if (partition.byte_count == 0u)
        return SPARK_STATUS_OK;
    if (partition.byte_count > UINT32_MAX)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    memset(scatter_gather,0,sizeof(*scatter_gather));
    scatter_gather->addr =
        (uint64_t)(uintptr_t)local_pointer + partition.offset;
    scatter_gather->length = (uint32_t)partition.byte_count;
    scatter_gather->lkey = local_memory_region->lkey;
    memset(work_request,0,sizeof(*work_request));
    work_request->sg_list = scatter_gather;
    work_request->num_sge = 1;
    work_request->opcode = IBV_WR_RDMA_WRITE;
    work_request->wr.rdma.remote_addr =
        remote_descriptor->address + partition.offset;
    work_request->wr.rdma.rkey = remote_descriptor->rkey;
    *write_present_out = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCountPayloadLaneWrite(
    SparkHiddenSparkHostRdmaState *state,
    uint64_t bytes,
    uint32_t lane_index,
    uint32_t striped,
    uint32_t *write_count)
{
    SparkMemlinkTransferPartition partition;
    SparkStatus status;

    if (state == 0 || write_count == 0 || lane_index >= state->lane_count)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (bytes == 0u)
        return SPARK_STATUS_OK;
    if (striped != 0u)
        status = SparkMemlinkBuildTransferPartition(
            bytes,state->lane_count,lane_index,&partition);
    else
    {
        /* Non-striped (doorbell) writes count whole on the
         * caller-selected lane; callers only invoke this on the
         * packet's assigned doorbell lane (NET-003). */
        partition.offset = 0u;
        partition.byte_count = bytes;
        status = SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_OK)
        return status;
    if (partition.byte_count != 0u)
        *write_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCountPacketLaneWrites(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    uint32_t doorbell,
    uint32_t doorbell_lane,
    uint32_t lane_index,
    uint32_t *write_count)
{
    SparkStatus status;

    if (write_count == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *write_count = 0u;
    if (doorbell != 0u && lane_index != doorbell_lane)
        return SPARK_STATUS_OK;
    status = SparkHiddenSparkHostRdmaCountPayloadLaneWrite(
        state,SparkHiddenSparkHostRdmaPacketHiddenBytes(packet),lane_index,
        doorbell == 0u,write_count);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkHiddenSparkHostRdmaCountPayloadLaneWrite(
        state,SparkHiddenSparkHostRdmaPacketSidebandBytes(packet),lane_index,
        doorbell == 0u,write_count);
}

static SparkStatus SparkHiddenSparkHostRdmaCheckPacketQueueCapacity(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    uint32_t doorbell,
    uint32_t doorbell_lane)
{
    SparkStatus status;
    uint32_t lane_count;
    uint32_t lane_index;
    uint32_t write_count;

    /* NET-003: a doorbell packet only consumes WR budget on its
     * assigned lane; striped packets consume on every lane. */
    lane_index = doorbell != 0u ? doorbell_lane : 0u;
    lane_count = doorbell != 0u ? doorbell_lane + 1u : state->lane_count;
    for (; lane_index < lane_count; ++lane_index)
    {
        status = SparkHiddenSparkHostRdmaCountPacketLaneWrites(
            state,packet,doorbell,doorbell_lane,lane_index,&write_count);
        if (status != SPARK_STATUS_OK)
            return status;
        if (state->outstanding_send_wr_counts[lane_index] >
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE - write_count)
            return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkHiddenSparkHostRdmaBuildDoorbellImmediate(
    const SparkHiddenSparkHostRdmaRemoteReceive *receive)
{
    uint32_t generation_tag;

    if (receive == 0)
        return UINT32_MAX;
    generation_tag = receive->persistent != 0u ?
        SparkHiddenSparkHostRdmaDoorbellGenerationTag(receive->generation) :
        0u;
    return receive->receive_index |
        (generation_tag <<
            SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_GENERATION_SHIFT);
}

static SparkStatus SparkHiddenSparkHostRdmaPostPersistentReturn(
    SparkHiddenSparkHostRdmaState *state,
    uint32_t credit_index,
    uint64_t generation)
{
    struct ibv_send_wr work_request;
    struct ibv_send_wr *bad_work_request;
    uint32_t immediate;
    uint32_t lane_index;

    if (state == 0 || state->is_sender != 0u || generation == 0u ||
        credit_index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    lane_index = SparkHiddenSparkHostRdmaDoorbellLane(state,credit_index);
    if (state->outstanding_send_wr_counts[lane_index] >=
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE)
        return SPARK_STATUS_BUSY;
    immediate = SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_RETURN_FLAG |
        credit_index |
        (SparkHiddenSparkHostRdmaDoorbellGenerationTag(generation) <<
            SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_GENERATION_SHIFT);
    memset(&work_request,0,sizeof(work_request));
    work_request.wr_id =
        SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_PERSISTENT_RETURN | credit_index;
    work_request.opcode = IBV_WR_SEND_WITH_IMM;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.imm_data = htonl(immediate);
    bad_work_request = 0;
    if (ibv_post_send(state->lanes[lane_index].queue_pair,
            &work_request,&bad_work_request) != 0)
        return SPARK_STATUS_IO_ERROR;
    state->outstanding_send_wr_counts[lane_index] += 1u;
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_spark_rdma_persistent_return_post route=%s credit=%u generation=%llu lane=%u\n",
            state->endpoint.route_name,credit_index,
            (unsigned long long)generation,lane_index);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPostLaneWrites(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send,
    uint32_t send_index,
    uint32_t lane_index,
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receive,
    struct ibv_mr *hidden_memory_region,
    void *hidden_local_pointer,
    struct ibv_mr *sideband_memory_region,
    void *sideband_local_pointer,
    uint64_t hidden_bytes,
    uint64_t sideband_bytes)
{
    struct ibv_sge scatter_gathers[2];
    struct ibv_send_wr work_requests[2];
    struct ibv_send_wr *bad_work_request;
    SparkStatus status;
    uint32_t present;
    uint32_t write_count;

    write_count = 0u;
    status = SparkHiddenSparkHostRdmaBuildLaneWrite(
        state,hidden_local_pointer,hidden_bytes,
        &remote_receive->hidden_descriptor,hidden_memory_region,lane_index,
        send->doorbell == 0u,&scatter_gathers[write_count],
        &work_requests[write_count],&present);
    if (status != SPARK_STATUS_OK)
        return status;
    write_count += present;
    status = SparkHiddenSparkHostRdmaBuildLaneWrite(
        state,sideband_local_pointer,sideband_bytes,
        &remote_receive->sideband_descriptor,sideband_memory_region,lane_index,
        send->doorbell == 0u,&scatter_gathers[write_count],
        &work_requests[write_count],&present);
    if (status != SPARK_STATUS_OK)
        return status;
    write_count += present;
    if (write_count == 0u)
        return SPARK_STATUS_OK;
    if (state->outstanding_send_wr_counts[lane_index] >
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE - write_count)
        return SPARK_STATUS_BUSY;
    if (write_count == 2u)
        work_requests[0].next = &work_requests[1];
    work_requests[write_count - 1u].wr_id =
        SparkHiddenSparkHostRdmaBuildSendWorkRequestId(
            send_index,lane_index);
    work_requests[write_count - 1u].send_flags = IBV_SEND_SIGNALED;
    if (send->doorbell != 0u)
    {
        work_requests[write_count - 1u].opcode =
            IBV_WR_RDMA_WRITE_WITH_IMM;
        work_requests[write_count - 1u].imm_data =
            htonl(SparkHiddenSparkHostRdmaBuildDoorbellImmediate(
                remote_receive));
    }
    bad_work_request = 0;
    if (ibv_post_send(state->lanes[lane_index].queue_pair,
            &work_requests[0],&bad_work_request) != 0)
        return SPARK_STATUS_IO_ERROR;
    state->outstanding_send_wr_counts[lane_index] += write_count;
    send->posted_wr_counts[lane_index] = (uint8_t)write_count;
    send->posted_lane_mask |= 1u << lane_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPostPacketWrites(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send,
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receive,
    struct ibv_mr *hidden_memory_region,
    void *hidden_local_pointer,
    struct ibv_mr *sideband_memory_region,
    void *sideband_local_pointer,
    uint64_t hidden_bytes,
    uint64_t sideband_bytes)
{
    SparkStatus status;
    uint32_t lane_count;
    uint32_t lane_index;
    uint32_t send_index;

    send_index = (uint32_t)(send - state->inflight_sends);
    /* NET-003: doorbell packets post only on their receive slot's
     * assigned lane; striped packets still post across all lanes. */
    lane_index = send->doorbell != 0u ?
        SparkHiddenSparkHostRdmaDoorbellLane(
            state,remote_receive->receive_index) : 0u;
    lane_count = send->doorbell != 0u ? lane_index + 1u :
        state->lane_count;
    for (; lane_index < lane_count; ++lane_index)
    {
        status = SparkHiddenSparkHostRdmaPostLaneWrites(
            state,send,send_index,lane_index,remote_receive,
            hidden_memory_region,hidden_local_pointer,
            sideband_memory_region,sideband_local_pointer,
            hidden_bytes,sideband_bytes);
        if (status != SPARK_STATUS_OK)
        {
            send->status = status;
            if (send->posted_lane_mask == 0u)
                send->complete = 1u;
            else
                state->asynchronous_send_count += 1u;
            return status;
        }
    }
    if (send->posted_lane_mask == 0u)
        return SPARK_STATUS_INTERNAL_ERROR;
    state->asynchronous_send_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaBuildSendCompletionMessage(
    const SparkHiddenTransportPacket *packet,
    SparkStatus transfer_status,
    SparkHiddenSparkHostRdmaControlMessage *message)
{
    memset(message,0,sizeof(*message));
    message->type = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_TRANSFER_COMPLETE;
    message->status = (uint32_t)transfer_status;
    message->sequence_id = packet->sequence_id;
    message->token_index = packet->token_index;
    message->active_sequence_count = packet->active_sequence_count;
    message->sideband_kind = packet->sideband_kind;
    message->sideband_bytes_per_sequence =
        packet->sideband_bytes_per_sequence;
}

static SparkStatus SparkHiddenSparkHostRdmaPostStripedCompletion(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send)
{
    SparkHiddenSparkHostRdmaStripedCompletion *completion;
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receive;
    struct ibv_send_wr work_request;
    struct ibv_send_wr *bad_work_request;
    uint32_t immediate;
    uint32_t lane_index;
    uint32_t send_index;

    if (state == 0 || send == 0 || state->is_sender == 0u ||
        send->active == 0u || send->complete == 0u || send->doorbell != 0u ||
        send->status != SPARK_STATUS_OK || send->remote_receive_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    remote_receive = &state->remote_receives[send->remote_receive_index];
    if (remote_receive->active == 0u || remote_receive->used == 0u ||
        remote_receive->receive_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
        return SPARK_STATUS_VALIDATION_FAILED;
    lane_index = SparkHiddenSparkHostRdmaDoorbellLane(
        state,remote_receive->receive_index);
    if (state->outstanding_send_wr_counts[lane_index] >=
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE)
        return SPARK_STATUS_BUSY;
    completion = SparkHiddenSparkHostRdmaReserveStripedCompletion(state);
    if (completion == 0)
        return SPARK_STATUS_BUSY;
    send_index = (uint32_t)(completion - state->striped_completions);
    immediate = SparkHiddenSparkHostRdmaBuildDoorbellImmediate(remote_receive);
    memset(&work_request,0,sizeof(work_request));
    work_request.wr_id =
        SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_STRIPED_COMPLETION |
        ((uint64_t)send_index <<
            SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_INDEX_SHIFT) |
        lane_index;
    work_request.opcode = IBV_WR_SEND_WITH_IMM;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.imm_data = htonl(immediate);
    bad_work_request = 0;
    if (ibv_post_send(state->lanes[lane_index].queue_pair,
            &work_request,&bad_work_request) != 0)
    {
        memset(completion,0,sizeof(*completion));
        return SPARK_STATUS_IO_ERROR;
    }
    state->outstanding_send_wr_counts[lane_index] += 1u;
    completion->lane_index = lane_index;
    completion->remote_receive_index = send->remote_receive_index;
    completion->status = send->status;
    completion->service_time_ns = SparkHiddenSparkHostRdmaMonotonicNs() -
        send->start_time_ns;
    completion->packet_snapshot = send->packet_snapshot;
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_spark_rdma_striped_fence_post route=%s slot=%u lane=%u sequence=%llu token=%llu\n",
            state->endpoint.route_name,send_index,lane_index,
            (unsigned long long)completion->packet_snapshot.sequence_id,
            (unsigned long long)completion->packet_snapshot.token_index);
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkHiddenSparkHostRdmaSendNeedsControl(
    const SparkHiddenSparkHostRdmaInflightSend *send)
{
    return send->status != SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCommitInflightSend(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send)
{
    SparkStatus status;
    uint64_t now_ns;
    uint64_t service_time_ns;

    if (state == 0 || send == 0 || send->active == 0u ||
        send->complete == 0u || send->remote_receive_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    now_ns = SparkHiddenSparkHostRdmaMonotonicNs();
    service_time_ns = now_ns >= send->start_time_ns ?
        now_ns - send->start_time_ns : 0u;
    if (send->doorbell != 0u || send->status != SPARK_STATUS_OK)
    {
        status = SparkHiddenSparkHostRdmaBuildCompletion(
            state,&send->packet_snapshot,send->status,service_time_ns);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    status = send->status;
    if (send->doorbell != 0u)
        state->doorbell_send_count += 1u;
    else
        state->striped_send_count += 1u;
    state->completed_send_count += 1u;
    if (send->doorbell != 0u || send->status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReleaseRemoteReceive(
            state,send->remote_receive_index);
    }
    /* NET-001: all WRs for this send completed, so unpin its cached
     * regions before the slot is wiped. */
    SparkHiddenSparkHostRdmaReleaseSendRegions(state,send);
    memset(send,0,sizeof(*send));
    return status;
}

static SparkStatus SparkHiddenSparkHostRdmaFinalizeInflightSend(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend *send)
{
    SparkHiddenSparkHostRdmaControlMessage message;
    SparkStatus result_status;
    SparkStatus status;
    uint32_t control_message_count;

    if (state == 0 || send == 0 || send->active == 0u ||
        send->complete == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    control_message_count =
        SparkHiddenSparkHostRdmaSendNeedsControl(send);
    if (SparkHiddenSparkHostRdmaCompletionQueueHasRoom(state,1u) == 0u ||
        SparkHiddenSparkHostRdmaControlQueueHasRoom(
            state,control_message_count) == 0u)
        return SPARK_STATUS_BUSY;
    if (control_message_count != 0u)
    {
        SparkHiddenSparkHostRdmaBuildSendCompletionMessage(
            &send->packet_snapshot,send->status,&message);
        status = SparkHiddenSparkHostRdmaEnqueueControlMessage(state,&message);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    result_status = SparkHiddenSparkHostRdmaCommitInflightSend(state,send);
    status = SparkHiddenSparkHostRdmaFlushControlQueue(state);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        return status;
    return result_status;
}

static SparkStatus SparkHiddenSparkHostRdmaPrepareInflightSend(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaInflightSend *send;
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receive;
    struct ibv_mr *hidden_memory_region;
    struct ibv_mr *sideband_memory_region;
    void *hidden_local_pointer;
    void *sideband_local_pointer;
    SparkStatus status;
    uint32_t remote_receive_index;
    uint32_t doorbell;
    uint32_t doorbell_lane;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;

    if (state == 0 || packet == 0 || state->is_sender == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    send = SparkHiddenSparkHostRdmaFindInflightSend(state,packet);
    if (send != 0)
        return send->complete != 0u ?
            SparkHiddenSparkHostRdmaFinalizeInflightSend(state,send) :
            SPARK_STATUS_BUSY;
    remote_receive = SparkHiddenSparkHostRdmaFindRemoteReceive(state, packet);
    if (remote_receive == 0 || SparkHiddenTransportCompletionQueueIsFull(
            &state->completion_queue) != 0u)
        return SPARK_STATUS_BUSY;
    remote_receive_index = (uint32_t)(remote_receive - state->remote_receives);
    if (remote_receive_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (state->send_ready_recorded[remote_receive_index] == 0u)
    {
        if (cudaEventRecord(
                state->send_ready_events[remote_receive_index],
                (cudaStream_t)packet->cuda_stream) != cudaSuccess)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        state->send_ready_recorded[remote_receive_index] = 1u;
        if (remote_receive->persistent == 0u &&
            cudaLaunchHostFunc(
                (cudaStream_t)packet->cuda_stream,
                SparkHiddenSparkHostRdmaSignalCudaReady,
                state) != cudaSuccess)
            return SPARK_STATUS_IO_ERROR;
    }
    switch (cudaEventQuery(state->send_ready_events[remote_receive_index]))
    {
        case cudaErrorNotReady:
            return SPARK_STATUS_BUSY;
        case cudaSuccess:
            state->send_ready_recorded[remote_receive_index] = 0u;
            break;
        default:
            state->send_ready_recorded[remote_receive_index] = 0u;
            return SPARK_STATUS_IO_ERROR;
    }
    doorbell = SparkHiddenSparkHostRdmaPacketUsesDoorbell(state,packet);
    doorbell_lane = doorbell != 0u && remote_receive->receive_index !=
            SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX ?
        SparkHiddenSparkHostRdmaDoorbellLane(
            state,remote_receive->receive_index) : 0u;
    status = SparkHiddenSparkHostRdmaCheckPacketQueueCapacity(
        state,packet,doorbell,doorbell_lane);
    if (status != SPARK_STATUS_OK)
        return status;
    send = SparkHiddenSparkHostRdmaReserveInflightSend(state);
    if (send == 0)
        return SPARK_STATUS_BUSY;
    send->packet_snapshot = *packet;
    send->remote_receive_index = remote_receive_index;
    send->doorbell =
        SparkHiddenSparkHostRdmaPacketUsesDoorbell(state,packet);
    send->start_time_ns = SparkHiddenSparkHostRdmaMonotonicNs();
    hidden_bytes = SparkHiddenSparkHostRdmaPacketHiddenBytes(packet);
    sideband_bytes = SparkHiddenSparkHostRdmaPacketSidebandBytes(packet);
    status = SparkHiddenSparkHostRdmaPreparePacketMemory(
        state,packet,
        &hidden_memory_region,&hidden_local_pointer,
        &sideband_memory_region,&sideband_local_pointer);
    if (status != SPARK_STATUS_OK)
        goto fail_send;
    SparkHiddenSparkHostRdmaAcquireSendRegions(
        state,send,hidden_memory_region,sideband_memory_region);
    if (send->doorbell != 0u && remote_receive->receive_index ==
            SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX)
    {
        status = SPARK_STATUS_INVALID_ARGUMENT;
        goto fail_send;
    }
    remote_receive->used = 1u;
    status = SparkHiddenSparkHostRdmaPostPacketWrites(
        state,send,remote_receive,hidden_memory_region,hidden_local_pointer,
        sideband_memory_region,sideband_local_pointer,hidden_bytes,
        sideband_bytes);
    if (status != SPARK_STATUS_OK && send->posted_lane_mask == 0u)
    {
        remote_receive->used = 0u;
        goto fail_send;
    }
    /* Posted WRs accept the packet. The source registration and remote
     * receive slot stay pinned until poll retires every signaled lane. */
    return SPARK_STATUS_OK;

fail_send:
    SparkHiddenSparkHostRdmaReleaseSendRegions(state,send);
    memset(send,0,sizeof(*send));
    return status;
}

static SparkStatus SparkHiddenSparkHostRdmaSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacket(&state->endpoint,packet);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaPumpProgress(state);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkHiddenSparkHostRdmaPrepareInflightSend(state,packet);
}

static SparkStatus SparkHiddenSparkHostRdmaFindBatchRemoteReceives(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count,
    SparkHiddenSparkHostRdmaRemoteReceive **remote_receives)
{
    uint64_t receive_mask;
    uint32_t packet_index;
    uint32_t receive_index;

    receive_mask = 0u;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        remote_receives[packet_index] =
            SparkHiddenSparkHostRdmaFindRemoteReceive(
                state,&packets[packet_index]);
        if (remote_receives[packet_index] == 0)
            return SPARK_STATUS_BUSY;
        receive_index = (uint32_t)(remote_receives[packet_index] -
            state->remote_receives);
        if (receive_index >=
                SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT ||
            (receive_mask & (1ull << receive_index)) != 0u)
            return SPARK_STATUS_DUPLICATE;
        receive_mask |= 1ull << receive_index;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaArmBatchCudaReady(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packets,
    SparkHiddenSparkHostRdmaRemoteReceive **remote_receives,
    uint32_t packet_count)
{
    cudaError_t cuda_status;
    uint32_t packet_index;
    uint32_t receive_index;
    uint32_t recorded;

    recorded = 0u;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        receive_index = (uint32_t)(remote_receives[packet_index] -
            state->remote_receives);
        if (state->send_ready_recorded[receive_index] != 0u)
            continue;
        if (cudaEventRecord(state->send_ready_events[receive_index],
                (cudaStream_t)packets[packet_index].cuda_stream) !=
                cudaSuccess ||
            cudaLaunchHostFunc(
                (cudaStream_t)packets[packet_index].cuda_stream,
                SparkHiddenSparkHostRdmaSignalCudaReady,state) != cudaSuccess)
            return SPARK_STATUS_IO_ERROR;
        state->send_ready_recorded[receive_index] = 1u;
        recorded = 1u;
    }
    if (recorded != 0u)
        return SPARK_STATUS_BUSY;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        receive_index = (uint32_t)(remote_receives[packet_index] -
            state->remote_receives);
        cuda_status = cudaEventQuery(
            state->send_ready_events[receive_index]);
        if (cuda_status == cudaErrorNotReady)
            return SPARK_STATUS_BUSY;
        if (cuda_status != cudaSuccess)
            return SPARK_STATUS_IO_ERROR;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        receive_index = (uint32_t)(remote_receives[packet_index] -
            state->remote_receives);
        state->send_ready_recorded[receive_index] = 0u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCheckBatchQueueCapacity(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packets,
    SparkHiddenSparkHostRdmaRemoteReceive **remote_receives,
    uint32_t packet_count)
{
    SparkStatus status;
    uint32_t doorbell;
    uint32_t doorbell_lane;
    uint32_t lane_index;
    uint32_t packet_index;
    uint32_t packet_write_count;
    uint32_t write_count;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        write_count = 0u;
        for (packet_index = 0u; packet_index < packet_count; ++packet_index)
        {
            doorbell = SparkHiddenSparkHostRdmaPacketUsesDoorbell(
                state,&packets[packet_index]);
            doorbell_lane = doorbell != 0u &&
                remote_receives[packet_index]->receive_index !=
                    SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX ?
                SparkHiddenSparkHostRdmaDoorbellLane(
                    state,remote_receives[packet_index]->receive_index) :
                0u;
            status = SparkHiddenSparkHostRdmaCountPacketLaneWrites(
                state,&packets[packet_index],doorbell,doorbell_lane,
                lane_index,&packet_write_count);
            if (status != SPARK_STATUS_OK)
                return status;
            write_count += packet_write_count;
        }
        if (state->outstanding_send_wr_counts[lane_index] >
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE - write_count)
            return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaRollbackPreparedBatch(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightBatch *batch,
    SparkHiddenSparkHostRdmaPreparedSend *prepared,
    uint32_t prepared_count)
{
    uint32_t packet_index;

    for (packet_index = 0u; packet_index < prepared_count; ++packet_index)
    {
        if (prepared[packet_index].remote_receive != 0)
            prepared[packet_index].remote_receive->used = 0u;
        if (prepared[packet_index].send != 0)
        {
            SparkHiddenSparkHostRdmaReleaseSendRegions(
                state,prepared[packet_index].send);
            memset(prepared[packet_index].send,0,
                sizeof(*prepared[packet_index].send));
        }
    }
    if (batch != 0)
        memset(batch,0,sizeof(*batch));
}

static SparkStatus SparkHiddenSparkHostRdmaPrepareBatchResources(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packets,
    SparkHiddenSparkHostRdmaRemoteReceive **remote_receives,
    uint32_t packet_count,
    SparkHiddenSparkHostRdmaPreparedSend *prepared,
    SparkHiddenSparkHostRdmaInflightBatch **batch_out)
{
    SparkHiddenSparkHostRdmaInflightBatch *batch;
    SparkHiddenSparkHostRdmaInflightSend *send;
    SparkStatus status;
    uint32_t packet_index;

    batch = SparkHiddenSparkHostRdmaReserveInflightBatch(state);
    if (batch == 0)
        return SPARK_STATUS_BUSY;
    memset(prepared,0,
        packet_count * sizeof(SparkHiddenSparkHostRdmaPreparedSend));
    batch->packet_count = packet_count;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        send = SparkHiddenSparkHostRdmaReserveInflightSend(state);
        if (send == 0)
        {
            SparkHiddenSparkHostRdmaRollbackPreparedBatch(
                state,batch,prepared,packet_index);
            return SPARK_STATUS_BUSY;
        }
        prepared[packet_index].send = send;
        prepared[packet_index].remote_receive = remote_receives[packet_index];
        send->packet_snapshot = packets[packet_index];
        send->remote_receive_index = (uint32_t)(remote_receives[packet_index] -
            state->remote_receives);
        send->doorbell = SparkHiddenSparkHostRdmaPacketUsesDoorbell(
            state,&packets[packet_index]);
        send->start_time_ns = SparkHiddenSparkHostRdmaMonotonicNs();
        prepared[packet_index].hidden_bytes =
            SparkHiddenSparkHostRdmaPacketHiddenBytes(&packets[packet_index]);
        prepared[packet_index].sideband_bytes =
            SparkHiddenSparkHostRdmaPacketSidebandBytes(&packets[packet_index]);
        status = SparkHiddenSparkHostRdmaPreparePacketMemory(
            state,&packets[packet_index],
            &prepared[packet_index].hidden_memory_region,
            &prepared[packet_index].hidden_local_pointer,
            &prepared[packet_index].sideband_memory_region,
            &prepared[packet_index].sideband_local_pointer);
        if (status == SPARK_STATUS_OK)
        {
            SparkHiddenSparkHostRdmaAcquireSendRegions(
                state,send,
                prepared[packet_index].hidden_memory_region,
                prepared[packet_index].sideband_memory_region);
        }
        if (status != SPARK_STATUS_OK ||
            (send->doorbell != 0u &&
             remote_receives[packet_index]->receive_index ==
                SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX))
        {
            if (status == SPARK_STATUS_OK)
                status = SPARK_STATUS_INVALID_ARGUMENT;
            break;
        }
        batch->send_indices[packet_index] =
            (uint32_t)(send - state->inflight_sends);
    }
    if (packet_index != packet_count)
    {
        SparkHiddenSparkHostRdmaRollbackPreparedBatch(
            state,batch,prepared,packet_index + 1u);
        return status;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
        prepared[packet_index].remote_receive->used = 1u;
    *batch_out = batch;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPostBatchLane(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightBatch *batch,
    SparkHiddenSparkHostRdmaPreparedSend *prepared,
    uint32_t packet_count,
    uint32_t lane_index)
{
    struct ibv_sge scatter_gathers[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE];
    struct ibv_send_wr work_requests[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE];
    struct ibv_send_wr *bad_work_request;
    SparkHiddenSparkHostRdmaPreparedSend *item;
    SparkStatus status;
    uint8_t packet_wr_counts[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT];
    uint32_t packet_index;
    uint32_t present;
    uint32_t start_count;
    uint32_t write_count;
    uint32_t batch_index;

    memset(packet_wr_counts,0,sizeof(packet_wr_counts));
    write_count = 0u;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        item = &prepared[packet_index];
        if (item->send->doorbell != 0u &&
            lane_index != SparkHiddenSparkHostRdmaDoorbellLane(
                state,item->remote_receive->receive_index))
            continue;
        start_count = write_count;
        status = SparkHiddenSparkHostRdmaBuildLaneWrite(
            state,item->hidden_local_pointer,item->hidden_bytes,
            &item->remote_receive->hidden_descriptor,
            item->hidden_memory_region,lane_index,
            item->send->doorbell == 0u,&scatter_gathers[write_count],
            &work_requests[write_count],&present);
        if (status != SPARK_STATUS_OK)
            return status;
        write_count += present;
        status = SparkHiddenSparkHostRdmaBuildLaneWrite(
            state,item->sideband_local_pointer,item->sideband_bytes,
            &item->remote_receive->sideband_descriptor,
            item->sideband_memory_region,lane_index,
            item->send->doorbell == 0u,&scatter_gathers[write_count],
            &work_requests[write_count],&present);
        if (status != SPARK_STATUS_OK)
            return status;
        write_count += present;
        if (write_count == start_count)
            continue;
        packet_wr_counts[packet_index] =
            (uint8_t)(write_count - start_count);
        if (item->send->doorbell != 0u)
        {
            work_requests[write_count - 1u].opcode =
                IBV_WR_RDMA_WRITE_WITH_IMM;
            work_requests[write_count - 1u].imm_data =
                htonl(SparkHiddenSparkHostRdmaBuildDoorbellImmediate(
                    item->remote_receive));
        }
    }
    if (write_count == 0u)
        return SPARK_STATUS_OK;
    if (state->outstanding_send_wr_counts[lane_index] >
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE - write_count)
        return SPARK_STATUS_BUSY;
    for (packet_index = 1u; packet_index < write_count; ++packet_index)
        work_requests[packet_index - 1u].next = &work_requests[packet_index];
    batch_index = (uint32_t)(batch - state->inflight_batches);
    work_requests[write_count - 1u].wr_id =
        SparkHiddenSparkHostRdmaBuildBatchWorkRequestId(
            batch_index,lane_index);
    work_requests[write_count - 1u].send_flags = IBV_SEND_SIGNALED;
    bad_work_request = 0;
    if (ibv_post_send(state->lanes[lane_index].queue_pair,
            &work_requests[0],&bad_work_request) != 0)
        return SPARK_STATUS_IO_ERROR;
    state->outstanding_send_wr_counts[lane_index] += write_count;
    batch->posted_lane_mask |= 1u << lane_index;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        if (packet_wr_counts[packet_index] != 0u)
        {
            prepared[packet_index].send->posted_wr_counts[lane_index] =
                packet_wr_counts[packet_index];
            prepared[packet_index].send->posted_lane_mask |=
                1u << lane_index;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaMarkBatchPostFailed(
    SparkHiddenSparkHostRdmaPreparedSend *prepared,
    uint32_t packet_count,
    SparkStatus status)
{
    SparkHiddenSparkHostRdmaInflightSend *send;
    uint32_t packet_index;

    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        send = prepared[packet_index].send;
        send->status = status;
        if (send->posted_lane_mask == 0u)
            send->complete = 1u;
    }
}

static SparkStatus SparkHiddenSparkHostRdmaStartSendBatch(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receives[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT];
    SparkHiddenSparkHostRdmaPreparedSend prepared[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT];
    SparkHiddenSparkHostRdmaInflightBatch *batch;
    SparkStatus status;
    uint32_t lane_index;

    status = SparkHiddenSparkHostRdmaFindBatchRemoteReceives(
        state,packets,packet_count,remote_receives);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaArmBatchCudaReady(
        state,packets,remote_receives,packet_count);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaCheckBatchQueueCapacity(
        state,packets,remote_receives,packet_count);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaPrepareBatchResources(
        state,packets,remote_receives,packet_count,prepared,&batch);
    if (status != SPARK_STATUS_OK)
        return status;
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        status = SparkHiddenSparkHostRdmaPostBatchLane(
            state,batch,prepared,packet_count,lane_index);
        if (status != SPARK_STATUS_OK)
        {
            if (batch->posted_lane_mask == 0u)
            {
                SparkHiddenSparkHostRdmaRollbackPreparedBatch(
                    state,batch,prepared,packet_count);
                return status;
            }
            SparkHiddenSparkHostRdmaMarkBatchPostFailed(
                prepared,packet_count,status);
            state->asynchronous_send_count += packet_count;
            return SPARK_STATUS_OK;
        }
    }
    if (batch->posted_lane_mask == 0u)
    {
        SparkHiddenSparkHostRdmaRollbackPreparedBatch(
            state,batch,prepared,packet_count);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->asynchronous_send_count += packet_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaFinalizeSendBatch(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaInflightSend **sends,
    uint32_t packet_count)
{
    SparkHiddenSparkHostRdmaControlMessage message;
    SparkStatus result_status;
    SparkStatus status;
    uint32_t control_message_count;
    uint32_t packet_index;

    control_message_count = 0u;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        if (sends[packet_index]->complete == 0u)
            return SPARK_STATUS_BUSY;
        if (sends[packet_index]->doorbell == 0u ||
            sends[packet_index]->status != SPARK_STATUS_OK)
            control_message_count += 1u;
    }
    if (SparkHiddenSparkHostRdmaCompletionQueueHasRoom(
            state,packet_count) == 0u ||
        SparkHiddenSparkHostRdmaControlQueueHasRoom(
            state,control_message_count) == 0u)
        return SPARK_STATUS_BUSY;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        if (SparkHiddenSparkHostRdmaSendNeedsControl(
                sends[packet_index]) == 0u)
            continue;
        SparkHiddenSparkHostRdmaBuildSendCompletionMessage(
            &sends[packet_index]->packet_snapshot,
            sends[packet_index]->status,&message);
        status = SparkHiddenSparkHostRdmaEnqueueControlMessage(
            state,&message);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    result_status = SPARK_STATUS_OK;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenSparkHostRdmaCommitInflightSend(
            state,sends[packet_index]);
        if (status != SPARK_STATUS_OK && result_status == SPARK_STATUS_OK)
            result_status = status;
    }
    status = SparkHiddenSparkHostRdmaFlushControlQueue(state);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        return status;
    return result_status;
}

static SparkStatus SparkHiddenSparkHostRdmaSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkHiddenSparkHostRdmaInflightSend *sends[
        SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT];
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;
    uint64_t send_mask;
    uint32_t active_count;
    uint32_t packet_index;
    uint32_t send_index;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packets == 0 || packet_count == 0u ||
        packet_count >
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT ||
        state->is_sender == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacketBatch(
        &state->endpoint,packets,packet_count);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaPumpProgress(state);
    if (status != SPARK_STATUS_OK)
        return status;
    active_count = 0u;
    send_mask = 0u;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        sends[packet_index] = SparkHiddenSparkHostRdmaFindInflightSend(
            state,&packets[packet_index]);
        if (sends[packet_index] == 0)
            continue;
        send_index = (uint32_t)(sends[packet_index] -
            state->inflight_sends);
        if (send_index >=
                SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT ||
            (send_mask & (1ull << send_index)) != 0u)
            return SPARK_STATUS_DUPLICATE;
        send_mask |= 1ull << send_index;
        active_count += 1u;
    }
    if (active_count == 0u)
        return SparkHiddenSparkHostRdmaStartSendBatch(
            state,packets,packet_count);
    if (active_count != packet_count)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SparkHiddenSparkHostRdmaFinalizeSendBatch(
        state,sends,packet_count);
}

static void SparkHiddenSparkHostRdmaResetPersistentActivation(
    SparkHiddenSparkHostRdmaPendingReceive *receive)
{
    if (receive == 0)
    {
        return;
    }
    receive->active = 0u;
    receive->complete = 0u;
    receive->completion_generation_tag = 0u;
    receive->visibility_flushed = 0u;
    receive->completion_published = 0u;
    receive->release_event_recorded = 0u;
    receive->generation = 0u;
    receive->completion_status = SPARK_STATUS_OK;
    receive->packet_snapshot = receive->packet_template;
}

static SparkStatus SparkHiddenSparkHostRdmaRegisterPersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    SparkHiddenTransportPacket *packet_template)
{
    SparkHiddenSparkHostRdmaControlMessage message;
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;
    uint64_t hidden_capacity_bytes;
    uint64_t sideband_capacity_bytes;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet_template == 0 || state->is_sender != 0u ||
        credit_index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaTerminalStatus(state);
    if (status != SPARK_STATUS_OK)
        return status;
    receive = &state->pending_receives[credit_index];
    if (receive->persistent_registered != 0u || receive->active != 0u)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    memset(receive,0,sizeof(*receive));
    receive->receive_index = credit_index;
    receive->hidden_region_index = SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
    receive->sideband_region_index = SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX;
    receive->packet_template = *packet_template;
    receive->packet_snapshot = *packet_template;
    hidden_capacity_bytes = (uint64_t)state->endpoint.bytes_per_sequence *
        state->endpoint.max_active_sequence_count;
    sideband_capacity_bytes =
        (uint64_t)packet_template->sideband_bytes_per_sequence *
        state->endpoint.max_active_sequence_count;
    status = SparkHiddenSparkHostRdmaRegisterReceiveRegion(state,
        packet_template->hidden_bf16,hidden_capacity_bytes,
        &receive->hidden_descriptor,&receive->hidden_region_index);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReleasePendingReceive(state,receive);
        return status;
    }
    status = SparkHiddenSparkHostRdmaRegisterReceiveRegion(state,
        packet_template->sideband_payload,sideband_capacity_bytes,
        &receive->sideband_descriptor,&receive->sideband_region_index);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReleasePendingReceive(state,receive);
        return status;
    }
    memset(&message,0,sizeof(message));
    message.type =
        SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_PERSISTENT_ADVERTISE;
    /* The template describes maximum buffer capacity. Doorbell eligibility
     * is evaluated against each actual packet when it is sent. */
    message.reserved = credit_index + 1u;
    message.active_sequence_count = state->endpoint.max_active_sequence_count;
    message.sideband_kind = packet_template->sideband_kind;
    message.sideband_bytes_per_sequence =
        packet_template->sideband_bytes_per_sequence;
    message.hidden = receive->hidden_descriptor;
    message.sideband = receive->sideband_descriptor;
    status = SparkHiddenSparkHostRdmaWriteControlMessage(state,&message);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaReleasePendingReceive(state,receive);
        return status;
    }
    receive->persistent_registered = 1u;
    receive->advertised = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPersistentRemoteCreditReady(
    void *transport_state,
    uint32_t credit_index)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkHiddenSparkHostRdmaRemoteReceive *receive;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || state->is_sender == 0u || credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaPumpProgress(state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    receive = &state->remote_receives[credit_index];
    return receive->active != 0u && receive->persistent != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static SparkStatus SparkHiddenSparkHostRdmaReservePersistentSend(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkHiddenSparkHostRdmaRemoteReceive *receive;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u ||
        generation == 0u || credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaPumpProgress(state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    receive = &state->remote_receives[credit_index];
    if (receive->active == 0u || receive->persistent == 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if (receive->used != 0u)
    {
        return receive->generation == generation &&
            SparkHiddenSparkHostRdmaRemoteReceiveMatchesPacket(
                receive,packet) != 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY;
    }
    if (generation <= receive->returned_generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    receive->used = 1u;
    receive->generation = generation;
    receive->sequence_id = packet->sequence_id;
    receive->token_index = packet->token_index;
    receive->active_sequence_count = packet->active_sequence_count;
    receive->sideband_kind = packet->sideband_kind;
    receive->sideband_bytes_per_sequence =
        packet->sideband_bytes_per_sequence;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCancelPersistentSend(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkHiddenSparkHostRdmaRemoteReceive *receive;
    uint32_t send_index;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || state->is_sender == 0u || generation == 0u ||
        credit_index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    receive = &state->remote_receives[credit_index];
    if (receive->used == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (receive->persistent == 0u || receive->generation != generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    for (send_index = 0u;
         send_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT;
         ++send_index)
    {
        if (state->inflight_sends[send_index].active != 0u &&
            state->inflight_sends[send_index].remote_receive_index ==
                credit_index)
        {
            return SPARK_STATUS_BUSY;
        }
    }
    receive->used = 0u;
    receive->returned_generation = generation;
    receive->generation = 0u;
    receive->sequence_id = 0u;
    receive->token_index = 0u;
    receive->active_sequence_count = 0u;
    state->send_ready_recorded[credit_index] = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaActivatePersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;
    uint32_t early_complete;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender != 0u ||
        generation == 0u || credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaTerminalStatus(state);
    if (status != SPARK_STATUS_OK)
        return status;
    receive = &state->pending_receives[credit_index];
    if (receive->persistent_registered == 0u ||
        packet->hidden_bf16 != receive->packet_template.hidden_bf16 ||
        packet->sideband_payload !=
            receive->packet_template.sideband_payload ||
        packet->sideband_kind != receive->packet_template.sideband_kind ||
        packet->sideband_bytes_per_sequence !=
            receive->packet_template.sideband_bytes_per_sequence)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (receive->active != 0u)
    {
        return receive->generation == generation ?
            SPARK_STATUS_OK : SPARK_STATUS_BUSY;
    }
    if (generation <= receive->returned_generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (receive->complete != 0u &&
        receive->completion_generation_tag !=
            SparkHiddenSparkHostRdmaDoorbellGenerationTag(generation))
        return SPARK_STATUS_VALIDATION_FAILED;
    early_complete = receive->complete;
    receive->active = 1u;
    receive->complete = early_complete;
    receive->completion_generation_tag =
        SparkHiddenSparkHostRdmaDoorbellGenerationTag(generation);
    receive->completion_published = 0u;
    receive->release_event_recorded = 0u;
    receive->visibility_flushed = 0u;
    receive->generation = generation;
    receive->packet_snapshot = *packet;
    receive->completion_status = SPARK_STATUS_OK;
    if (early_complete != 0u)
        SparkHiddenSparkHostRdmaSignalEvent(state);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaCancelPersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation)
{
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || state->is_sender != 0u || generation == 0u ||
        credit_index >= SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    receive = &state->pending_receives[credit_index];
    if (receive->active == 0u)
    {
        if (receive->complete != 0u)
        {
            if (receive->completion_generation_tag !=
                    SparkHiddenSparkHostRdmaDoorbellGenerationTag(generation))
                return SPARK_STATUS_VALIDATION_FAILED;
            status = SparkHiddenSparkHostRdmaFenceSession(state);
            if (status == SPARK_STATUS_OK)
            {
                receive->returned_generation = generation;
                SparkHiddenSparkHostRdmaResetPersistentActivation(receive);
            }
            return status;
        }
        return SPARK_STATUS_OK;
    }
    if (receive->persistent_registered == 0u ||
        receive->generation != generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (receive->complete != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    status = SparkHiddenSparkHostRdmaFenceSession(state);
    if (status != SPARK_STATUS_OK)
        return status;
    receive->returned_generation = generation;
    SparkHiddenSparkHostRdmaResetPersistentActivation(receive);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSendPersistent(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaRemoteReceive *receive;
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u ||
        generation == 0u || credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaTerminalStatus(state);
    if (status != SPARK_STATUS_OK)
        return status;
    receive = &state->remote_receives[credit_index];
    if (receive->persistent == 0u || receive->used == 0u ||
        receive->generation != generation ||
        SparkHiddenSparkHostRdmaRemoteReceiveMatchesPacket(
            receive,packet) == 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    return SparkHiddenSparkHostRdmaPrepareInflightSend(state,packet);
}

static SparkStatus SparkHiddenSparkHostRdmaReleasePersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    void *consumer_cuda_stream)
{
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkHiddenSparkHostRdmaState *state;
    cudaError_t cuda_status;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || state->is_sender != 0u || generation == 0u ||
        consumer_cuda_stream == 0 || credit_index >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaTerminalStatus(state);
    if (status != SPARK_STATUS_OK)
        return status;
    receive = &state->pending_receives[credit_index];
    if (receive->persistent_registered == 0u || receive->active == 0u ||
        receive->completion_published == 0u ||
        receive->generation != generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (receive->release_event_recorded == 0u)
    {
        if (cudaEventRecord(state->receive_release_events[credit_index],
                (cudaStream_t)consumer_cuda_stream) != cudaSuccess)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        receive->release_event_recorded = 1u;
    }
    cuda_status = cudaEventQuery(
        state->receive_release_events[credit_index]);
    if (cuda_status == cudaErrorNotReady)
    {
        return SPARK_STATUS_BUSY;
    }
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    status = SparkHiddenSparkHostRdmaPostPersistentReturn(
        state,credit_index,generation);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    receive->returned_generation = generation;
    SparkHiddenSparkHostRdmaResetPersistentActivation(receive);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaRetireCompletedSends(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkHiddenSparkHostRdmaStripedCompletion *completion;
    SparkStatus status;
    uint32_t completion_index;
    uint32_t send_index;

    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (send_index = 0u;
         send_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT;
         ++send_index)
    {
        if (state->inflight_sends[send_index].active == 0u ||
            state->inflight_sends[send_index].complete == 0u)
            continue;
        if (state->inflight_sends[send_index].doorbell == 0u &&
            state->inflight_sends[send_index].status == SPARK_STATUS_OK)
        {
            status = SparkHiddenSparkHostRdmaPostStripedCompletion(
                state,&state->inflight_sends[send_index]);
            if (status == SPARK_STATUS_BUSY)
                return SPARK_STATUS_OK;
            if (status != SPARK_STATUS_OK)
                return status;
        }
        status = SparkHiddenSparkHostRdmaFinalizeInflightSend(
            state,&state->inflight_sends[send_index]);
        if (status == SPARK_STATUS_BUSY)
            return SPARK_STATUS_OK;
        if (status != SPARK_STATUS_OK)
            return status;
    }
    for (completion_index=0u;
         completion_index<SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_INFLIGHT_SEND_COUNT;
         completion_index++)
    {
        completion = &state->striped_completions[completion_index];
        if (completion->active == 0u || completion->complete == 0u)
            continue;
        if (SparkHiddenSparkHostRdmaCompletionQueueHasRoom(state,1u) == 0u)
            return SPARK_STATUS_OK;
        if (state->debug_enabled != 0u)
        {
            fprintf(stderr,
                "hidden_spark_rdma_striped_fence_retire route=%s slot=%u sequence=%llu token=%llu\n",
                state->endpoint.route_name,completion_index,
                (unsigned long long)completion->packet_snapshot.sequence_id,
                (unsigned long long)completion->packet_snapshot.token_index);
        }
        status = SparkHiddenSparkHostRdmaBuildCompletion(state,
            &completion->packet_snapshot,completion->status,
            completion->service_time_ns);
        if (status != SPARK_STATUS_OK)
            return status;
        SparkHiddenSparkHostRdmaReleaseRemoteReceive(
            state,completion->remote_receive_index);
        memset(completion,0,sizeof(*completion));
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaRetireCompletedReceives(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkStatus status;
    uint32_t receive_index;

    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (receive_index = 0u;
         receive_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++receive_index)
    {
        if (state->pending_receives[receive_index].active == 0u ||
            state->pending_receives[receive_index].complete == 0u ||
            state->pending_receives[receive_index].completion_published != 0u)
            continue;
        status = SparkHiddenSparkHostRdmaFinalizePendingReceive(
            state,&state->pending_receives[receive_index]);
        if (status == SPARK_STATUS_BUSY)
            return SPARK_STATUS_OK;
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || completion == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenSparkHostRdmaPumpDoorbells(state);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaRetireCompletedSends(state);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenSparkHostRdmaRetireCompletedReceives(state);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkHiddenTransportCompletionQueuePop(
        &state->completion_queue,completion);
}

static SparkStatus SparkHiddenSparkHostRdmaAppendPollDescriptor(
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count,
    int fd,
    uint32_t events)
{
    SparkHiddenTransportPollDescriptor *descriptor;

    if (descriptor_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (fd < 0 || events == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (*descriptor_count >= descriptor_capacity || descriptors == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    descriptor = &descriptors[*descriptor_count];
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    descriptor->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_POLL_DESCRIPTOR_BYTES;
    descriptor->fd = fd;
    descriptor->events = events;
    *descriptor_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaGetPollDescriptors(
    void *transport_state,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out)
{
    SparkHiddenSparkHostRdmaState *state;
    uint32_t descriptor_count;
    uint32_t control_events;
    SparkStatus status;

    if (transport_state == 0 || descriptor_count_out == 0 ||
        (descriptors == 0 && descriptor_capacity != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    status = SparkHiddenSparkHostRdmaTerminalStatus(state);
    if (status != SPARK_STATUS_OK)
        return status;
    descriptor_count = 0u;
    status = SparkHiddenSparkHostRdmaAppendPollDescriptor(descriptors,
        descriptor_capacity, &descriptor_count, state->event_fd,
        SPARK_HIDDEN_TRANSPORT_POLL_READ);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    control_events = SPARK_HIDDEN_TRANSPORT_POLL_READ;
    if (state->control_queue_count != 0u)
        control_events |= SPARK_HIDDEN_TRANSPORT_POLL_WRITE;
    status = SparkHiddenSparkHostRdmaAppendPollDescriptor(descriptors,
        descriptor_capacity,&descriptor_count,state->control_fd,
        control_events);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkHiddenSparkHostRdmaAppendPollDescriptor(descriptors,
        descriptor_capacity, &descriptor_count,
        state->completion_channel != 0 ? state->completion_channel->fd : -1,
        SPARK_HIDDEN_TRANSPORT_POLL_READ);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *descriptor_count_out = descriptor_count;
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaDestroyState(SparkHiddenSparkHostRdmaState *state)
{
    uint32_t lane_index;
    uint32_t receive_index;

    if (state == 0)
    {
        return;
    }
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_spark_rdma_stats mode=%u doorbell_sends=%llu striped_sends=%llu async_sends=%llu completed_sends=%llu control_busy=%llu mr_cache_hits=%llu pointer_attribute_queries=%llu mr_registrations=%llu mr_evictions=%llu mapped_host_zero_copy_transfers=%llu mapped_host_zero_copy_bytes=%llu gpudirect_transfers=%llu gpudirect_bytes=%llu\n",
            state->memory_mode,
            (unsigned long long)state->doorbell_send_count,
            (unsigned long long)state->striped_send_count,
            (unsigned long long)state->asynchronous_send_count,
            (unsigned long long)state->completed_send_count,
            (unsigned long long)state->control_queue_busy_count,
            (unsigned long long)state->memory_region_cache_hit_count,
            (unsigned long long)state->pointer_attribute_query_count,
            (unsigned long long)state->memory_region_register_count,
            (unsigned long long)state->memory_region_eviction_count,
            (unsigned long long)state->mapped_host_zero_copy_transfer_count,
            (unsigned long long)state->mapped_host_zero_copy_transfer_bytes,
            (unsigned long long)state->gpudirect_transfer_count,
            (unsigned long long)state->gpudirect_transfer_bytes);
    }
    for (receive_index = 0u;
         receive_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++receive_index)
    {
        SparkHiddenSparkHostRdmaReleasePendingReceive(
            state, &state->pending_receives[receive_index]);
    }
    for (receive_index = 0u;
         receive_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT;
         ++receive_index)
    {
        if (state->send_ready_events[receive_index] != 0)
        {
            cudaEventDestroy(state->send_ready_events[receive_index]);
        }
        if (state->receive_release_events[receive_index] != 0)
        {
            cudaEventDestroy(state->receive_release_events[receive_index]);
        }
    }
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        if (state->lanes[lane_index].queue_pair != 0)
        {
            ibv_destroy_qp(state->lanes[lane_index].queue_pair);
        }
        if (state->lanes[lane_index].completion_queue != 0)
        {
            ibv_destroy_cq(state->lanes[lane_index].completion_queue);
        }
    }
    SparkHiddenSparkHostRdmaDeregisterCachedMemoryRegions(state);
    if (state->completion_channel != 0)
    {
        ibv_destroy_comp_channel(state->completion_channel);
    }
    if (state->protection_domain != 0)
    {
        ibv_dealloc_pd(state->protection_domain);
    }
    if (state->verbs_context != 0)
    {
        ibv_close_device(state->verbs_context);
    }
    state->control_fd = SparkHiddenSparkHostRdmaCloseFd(state->control_fd);
    state->listen_fd = SparkHiddenSparkHostRdmaCloseFd(state->listen_fd);
    state->event_fd = SparkHiddenSparkHostRdmaCloseFd(state->event_fd);
    free(state);
}

static SparkStatus SparkHiddenSparkHostRdmaConfigureRoute(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportEndpoint *endpoint)
{
    int written;

    if (state == 0 || endpoint == 0 ||
        (endpoint->configuration_flags &
            SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION) == 0u ||
        endpoint->local_rank_index > (uint32_t)INT32_MAX ||
        endpoint->source_rank_index > (uint32_t)INT32_MAX ||
        endpoint->sink_rank_index > (uint32_t)INT32_MAX ||
        endpoint->control_port_base == 0u ||
        endpoint->control_port_base > 65535u - endpoint->sink_rank_index)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state->local_rank = (int32_t)endpoint->local_rank_index;
    state->source_rank = (int32_t)endpoint->source_rank_index;
    state->sink_rank = (int32_t)endpoint->sink_rank_index;
    state->control_port_base = endpoint->control_port_base;
    written = snprintf(state->source_host,sizeof(state->source_host),"%s",
        endpoint->source_host);
    if (written < 0 || (uint32_t)written >= sizeof(state->source_host))
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    written = snprintf(state->sink_host,sizeof(state->sink_host),"%s",
        endpoint->sink_host);
    if (written < 0 || (uint32_t)written >= sizeof(state->sink_host))
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    state->is_sender = state->local_rank == state->source_rank ? 1u : 0u;
    if (state->is_sender == 0u && state->local_rank != state->sink_rank)
    {
        fprintf(stderr,
            "hidden_spark_rdma_configure_rank_mismatch local=%d source=%d sink=%d route=%s source_host=%s sink_host=%s\n",
            state->local_rank,state->source_rank,state->sink_rank,
            endpoint->route_name,endpoint->source_host,endpoint->sink_host);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;
    uint32_t config_value;
    uint32_t lane_count;
    uint32_t receive_index;

    if (endpoint == 0 || transport_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
#if SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT
    status = SparkHiddenTransportValidateSparkGpudirectRdmaEndpoint(endpoint);
#else
    status = SparkHiddenTransportValidateSparkHostRdmaEndpoint(endpoint);
#endif
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    state = (SparkHiddenSparkHostRdmaState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->endpoint = *endpoint;
    if ((endpoint->configuration_flags &
            SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_OPEN_TIMEOUT) != 0u)
    {
        state->open_timeout_milli = endpoint->reserved0;
        status = state->open_timeout_milli != 0u ? SPARK_STATUS_OK :
            SPARK_STATUS_INVALID_ARGUMENT;
    }
    else
    {
        status = SparkHiddenSparkHostRdmaParseUintEnv(
            SPARK_HIDDEN_SPARK_HOST_RDMA_OPEN_TIMEOUT_ENV,
            SPARK_HIDDEN_SPARK_HOST_RDMA_OPEN_TIMEOUT_DEFAULT_MS,
            &state->open_timeout_milli);
    }
    state->open_deadline_ns = status == SPARK_STATUS_OK ?
        SparkHiddenTransportRdmaControlDeadlineNs(
            state->open_timeout_milli) : 0u;
    if (status != SPARK_STATUS_OK || state->open_deadline_ns == 0u)
    {
        free(state);
        return status != SPARK_STATUS_OK ? status : SPARK_STATUS_IO_ERROR;
    }
    state->listen_fd = -1;
    state->control_fd = -1;
    state->event_fd = -1;
#if SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT
    state->memory_mode =
        SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_DEVICE_DIRECT;
#else
    state->memory_mode =
        SPARK_HIDDEN_SPARK_HOST_RDMA_MEMORY_MODE_MAPPED_HOST;
#endif
    status = SparkHiddenSparkHostRdmaConfigureGpudirectVisibility(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaParseUintEnv(
        "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DEBUG",
        0u,
        &state->debug_enabled);
    if (status != SPARK_STATUS_OK || state->debug_enabled > 1u)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaParseUintEnv(
        "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_LANES",
        SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_LANE_COUNT,
        &lane_count);
    if (status != SPARK_STATUS_OK || lane_count == 0u ||
        lane_count > SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->lane_count = lane_count;
    status = SparkHiddenSparkHostRdmaConfigureRoute(state, endpoint);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaParseUintEnv(
        "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_IB_PORT",
        1u,
        &config_value);
    if (status != SPARK_STATUS_OK || config_value == 0u ||
        config_value > 255u)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->verbs_port = (uint8_t)config_value;
    status = SparkHiddenSparkHostRdmaParseUintEnv(
        "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DOORBELL_MAX_BYTES",
        SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_DOORBELL_MAX_BYTES,
        &state->doorbell_max_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    if (state->is_sender == 0u && state->local_rank != state->sink_rank)
    {
        fprintf(stderr,
            "hidden_spark_rdma_receiver_rank_mismatch local_rank=%u sink_rank=%u source_rank=%u route=%s\n",
            state->local_rank,state->sink_rank,state->source_rank,
            state->endpoint.route_name);
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->event_fd = eventfd(0u, EFD_NONBLOCK | EFD_CLOEXEC);
    if (state->event_fd < 0)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (state->is_sender != 0u)
    {
        for (receive_index = 0u;
             receive_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT;
             ++receive_index)
        {
            if (cudaEventCreateWithFlags(
                    &state->send_ready_events[receive_index],
                    cudaEventDisableTiming) != cudaSuccess)
            {
                SparkHiddenSparkHostRdmaDestroyState(state);
                return SPARK_STATUS_DRIVER_LOAD_ERROR;
            }
        }
    }
    else
    {
        for (receive_index = 0u;
             receive_index <
                SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
             ++receive_index)
        {
            if (cudaEventCreateWithFlags(
                    &state->receive_release_events[receive_index],
                    cudaEventDisableTiming) != cudaSuccess)
            {
                SparkHiddenSparkHostRdmaDestroyState(state);
                return SPARK_STATUS_DRIVER_LOAD_ERROR;
            }
        }
    }
    status = SparkHiddenSparkHostRdmaOpenVerbsDevice(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaCreateCompletionChannel(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaCreateQueuePairs(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaConnectControl(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaExchangeCompatibilityHello(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaExchangeQueuePairInfo(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaReadyQueuePairs(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_spark_rdma_ready route=%s rank=%d sender=%u lanes=%u device=%s path_mtu_bytes=%u memory_mode=%u doorbell_max_bytes=%u\n",
            endpoint->route_name,
            state->local_rank,
            state->is_sender,
            state->lane_count,
            state->verbs_device_name,
            SparkHiddenSparkHostRdmaPathMtuBytes(&state->lanes[0]),
            state->memory_mode,
            state->doorbell_max_bytes);
    }
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaDestroy(void *transport_state)
{
    SparkHiddenSparkHostRdmaDestroyState((SparkHiddenSparkHostRdmaState *)transport_state);
}

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface transport_interface;

    memset(&transport_interface, 0, sizeof(transport_interface));
    transport_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
#if SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_GPUDIRECT_RDMA_CAPS;
#else
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS;
#endif
    transport_interface.capability_flags |=
        SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    transport_interface.initialize = SparkHiddenSparkHostRdmaInitialize;
    transport_interface.destroy = SparkHiddenSparkHostRdmaDestroy;
    transport_interface.post_receive = SparkHiddenSparkHostRdmaPostReceive;
    transport_interface.send = SparkHiddenSparkHostRdmaSend;
    transport_interface.poll = SparkHiddenSparkHostRdmaPoll;
    transport_interface.post_receive_batch =
        SparkHiddenSparkHostRdmaPostReceiveBatch;
    transport_interface.send_batch = SparkHiddenSparkHostRdmaSendBatch;
    transport_interface.get_poll_descriptors = SparkHiddenSparkHostRdmaGetPollDescriptors;
    transport_interface.register_persistent_receive =
        SparkHiddenSparkHostRdmaRegisterPersistentReceive;
    transport_interface.persistent_remote_credit_ready =
        SparkHiddenSparkHostRdmaPersistentRemoteCreditReady;
    transport_interface.reserve_persistent_send =
        SparkHiddenSparkHostRdmaReservePersistentSend;
    transport_interface.cancel_persistent_send =
        SparkHiddenSparkHostRdmaCancelPersistentSend;
    transport_interface.activate_persistent_receive =
        SparkHiddenSparkHostRdmaActivatePersistentReceive;
    transport_interface.cancel_persistent_receive =
        SparkHiddenSparkHostRdmaCancelPersistentReceive;
    transport_interface.send_persistent =
        SparkHiddenSparkHostRdmaSendPersistent;
    transport_interface.release_persistent_receive =
        SparkHiddenSparkHostRdmaReleasePersistentReceive;
    return &transport_interface;
}
