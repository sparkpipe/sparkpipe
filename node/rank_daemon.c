#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200112L

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_distributed_work.h"
#include "sparkpipe/spark_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_ring_node_context_builder.h"
#include "sparkpipe/spark_ring_runtime.h"
#include "sparkpipe/spark_ring_work_control.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "runtime/net.h"

#define SPARK_RING_DAEMON_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_RING_DAEMON_DEFAULT_PROGRAM "glm52.ring.rank.production"
#define SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY 64u
#define SPARK_RING_DAEMON_WORK_QUEUE_HASH_SLOTS 4096u
#define SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY 4096u
#define SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY \
    SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY
#define SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY \
    (SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY * 4u)
#define SPARK_RING_DAEMON_DRIVER_COMPLETION_QUEUE_CAPACITY \
    SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY
#define SPARK_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY 2048u
#define SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT UINT32_MAX
#define SPARK_RING_DAEMON_POLL_FD_CAPACITY 32u
#define SPARK_RING_DAEMON_WORK_STATE_READY 0u
#define SPARK_RING_DAEMON_WORK_STATE_WAITING_FORWARD 1u
#define SPARK_RING_DAEMON_WORK_STATE_WAITING_SUBMIT 2u
#define SPARK_RING_DAEMON_WORK_PHASE_PREFILL 0u
#define SPARK_RING_DAEMON_WORK_PHASE_DECODE 1u
#define SPARK_RING_DAEMON_WORK_PHASE_VERIFY 2u
#define SPARK_RING_DAEMON_WORK_PHASE_RELEASE 3u
#define SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_FREE 0u
#define SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_ACTIVE 1u
#define SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_TOMBSTONE 2u
#define SPARK_RING_DAEMON_DEPENDENCY_HASH_CAPACITY 2048u
#define SPARK_RING_DAEMON_WORK_HASH_OFFSET UINT32_C(2166136261)
#define SPARK_RING_DAEMON_WORK_HASH_PRIME UINT32_C(16777619)
#define SPARK_RING_DAEMON_POLL_KIND_WORK 0x00000001u
#define SPARK_RING_DAEMON_POLL_KIND_FINAL_EVENT 0x00000002u
#define SPARK_RING_DAEMON_POLL_KIND_INPUT_TRANSPORT 0x00000004u
#define SPARK_RING_DAEMON_POLL_KIND_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_RING_DAEMON_POLL_KIND_COMPLETION_WAKE 0x00000010u
#define SPARK_RING_DAEMON_POLL_KIND_WORK_OUTPUT 0x00000020u
#define SPARK_RING_DAEMON_POLL_KIND_TIMER 0x00000040u
#define SPARK_RING_DAEMON_POLL_KIND_CUDA_RESIDENT 0x00000080u
#define SPARK_RING_DAEMON_CONNECT_RETRY_NS 250000ull
#define SPARK_RING_DAEMON_RUNNER_PROGRESS_NS 250000ull

typedef SparkGlm52DsparkDraftResult SparkRingDaemonDraftResult;

#define SPARK_RING_DAEMON_DRAFT_ABI_VERSION \
    SPARK_GLM52_DSPARK_ABI_VERSION
#define SPARK_RING_DAEMON_DRAFT_DESCRIPTOR_BYTES ((uint32_t)sizeof(SparkRingDaemonDraftResult))
_Static_assert(SPARK_RING_DAEMON_DRAFT_DESCRIPTOR_BYTES == SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES, "rank daemon draft-result descriptor must match the DSpark wire ABI");

typedef struct SparkRingDaemonConfig
{
    const char *moe_pack_root;
    const char *stagepack_root;
    const char *transport_shared_object_path;
    const char *driver_path;
    const char *node_context_builder_shared_object_path;
    const char *embedding_pack_path;
    const char *cuda_resident_socket_path;
    const char *program_name;
    const char *node_target;
    const char *final_event_bind_address;
    const char *final_event_return_host;
    uint32_t rank_index;
    uint32_t rank_is_set;
    uint32_t own_final_event;
    uint32_t transport_busy_poll;
    uint32_t max_active_sequence_count;
    uint32_t port_base;
    uint32_t model_quantization_mode;
} SparkRingDaemonConfig;

typedef struct SparkRingDaemonInflightTransaction
{
    uint32_t active;
    uint32_t remaining_completion_count;
    uint32_t terminal_state;
    uint32_t terminal_status;
    SparkDistributedWorkIdentity identity;
    uint64_t packet_hash;
} SparkRingDaemonInflightTransaction;

typedef struct SparkRingDaemonInflightCompletion
{
    uint32_t state;
    uint32_t transaction_index;
    uint64_t request_id;
    uint64_t request_generation;
    uint64_t sequence_id;
    uint64_t sequence_position;
} SparkRingDaemonInflightCompletion;

typedef struct SparkRingDaemonDriverCompletionRecord
{
    SparkModelDriverCompletion completion;
    SparkRingDaemonDraftResult dspark_draft;
    uint32_t dspark_draft_valid;
    uint32_t reserved0;
} SparkRingDaemonDriverCompletionRecord;

typedef struct SparkRingDaemonRuntime
{
    SparkRingRuntimeRankPlan rank_plan;
    SparkRingRuntimeFinalEventRoute final_event_route;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportSession *input_transport_session;
    SparkHiddenTransportSession *output_transport_session;
    SparkRingNodeContextBuilderDynamicLibrary builder_library;
    void *builder_state;
    SparkRingNodeContextBuilderResult builder_result;
    SparkLoadedModelDriver loaded_driver;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    SparkResidentDecodeStageProductionRunner runner;
    int32_t cuda_resident_fd;
    uint32_t trace_enabled;
    const char *cuda_resident_socket_path;
    uint64_t cuda_resident_retry_after_ns;
    uint64_t cuda_resident_next_sequence_number;
    uint64_t cuda_resident_submit_count;
    uint64_t cuda_resident_completion_count;
    uint64_t cuda_resident_error_count;
    SparkCudaResidentIpcHeader cuda_resident_read_header;
    uint32_t cuda_resident_read_header_offset;
    uint32_t cuda_resident_read_payload_offset;
    uint8_t cuda_resident_payload[SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES];
    int32_t work_listen_fd;
    int32_t work_input_socket_fd;
    int32_t work_output_socket_fd;
    uint32_t work_output_connecting;
    uint32_t work_output_write_offset;
    uint64_t work_output_retry_mono_ns;
    SparkDistributedWorkAcknowledgement work_output_acknowledgement;
    uint32_t work_output_acknowledgement_read_offset;
    uint32_t work_output_waiting_for_acknowledgement;
    uint64_t work_output_packet_hash;
    SparkDistributedWorkAcknowledgement work_input_acknowledgement;
    uint32_t work_input_acknowledgement_write_offset;
    uint32_t work_input_acknowledgement_pending;
    int32_t final_event_listen_fd;
    int32_t final_event_socket_fd;
    uint32_t final_event_socket_connecting;
    uint32_t final_event_write_offset;
    uint64_t final_event_retry_mono_ns;
    uint8_t work_read_buffer[SPARK_RING_WORK_CONTROL_PACKET_BYTES];
    uint32_t work_read_offset;
    SparkRingWorkControlPacket work_queue[SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_hash_heads[SPARK_RING_DAEMON_WORK_QUEUE_HASH_SLOTS];
    uint32_t work_queue_hash_next[SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_submitted[SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_forwarded[SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint32_t work_queue_state[SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY];
    uint64_t dependency_sequence_ids[
        SPARK_RING_DAEMON_DEPENDENCY_HASH_CAPACITY];
    uint64_t dependency_sequence_positions[
        SPARK_RING_DAEMON_DEPENDENCY_HASH_CAPACITY];
    uint32_t work_queue_head;
    uint32_t work_queue_count;
    SparkDistributedWorkTransactionEntry transaction_entries[
        SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY];
    uint32_t transaction_hash_heads[
        SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY];
    uint32_t transaction_hash_next[
        SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY];
    SparkDistributedWorkTransactionLedger transaction_ledger;
    SparkRingDaemonInflightTransaction inflight_transactions[
        SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY];
    SparkRingDaemonInflightCompletion inflight_completions[
        SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY];
    pthread_mutex_t driver_completion_mutex;
    uint32_t driver_completion_mutex_initialized;
    pthread_mutex_t builder_completion_mutex;
    uint32_t builder_completion_mutex_initialized;
    SparkRingDaemonDriverCompletionRecord driver_completion_queue[
        SPARK_RING_DAEMON_DRIVER_COMPLETION_QUEUE_CAPACITY];
    uint32_t driver_completion_queue_head;
    uint32_t driver_completion_queue_count;
    uint32_t driver_completion_queue_overflow;
    uint32_t reserved_completion_queue;
    uint64_t work_receive_count;
    uint64_t work_forward_count;
    uint64_t work_submit_count;
    uint64_t work_error_count;
    uint64_t work_duplicate_count;
    uint64_t work_deferred_count;
    uint64_t work_wake_count;
    uint64_t driver_completion_count;
    uint64_t driver_inflight_open_ns;
    uint32_t driver_inflight_warned;
    uint32_t driver_inflight_count;
    uint8_t final_event_read_buffer[
        SPARK_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES];
    uint32_t final_event_read_offset;
    SparkRingRuntimeFinalEvent final_event_queue[
        SPARK_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY];
    uint32_t final_event_queue_head;
    uint32_t final_event_queue_count;
    int32_t wake_pipe_read_fd;
    int32_t wake_pipe_write_fd;
    uint64_t wake_signal_count;
    uint64_t wake_drop_count;
    uint64_t timer_wake_count;
    uint64_t final_event_send_count;
    uint64_t final_event_receive_count;
    uint64_t final_event_send_error_count;
    uint64_t final_event_receive_error_count;
} SparkRingDaemonRuntime;

static volatile sig_atomic_t SparkRingDaemonRunning = 1;

static void SparkRingDaemonCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion);
static SparkStatus SparkRingDaemonQueueDriverCompletion(
    SparkRingDaemonRuntime *runtime,
    const SparkModelDriverCompletion *completion,
    const SparkRingDaemonDraftResult *dspark_draft);
static uint32_t SparkRingDaemonPumpDriverCompletions(
    SparkRingDaemonRuntime *runtime);

static void SparkRingDaemonSignal(int signal_number)
{
    (void)signal_number;
    SparkRingDaemonRunning = 0;
}

static void SparkRingDaemonInitializeConfig(
    SparkRingDaemonConfig *configuration)
{
    memset(configuration,0,sizeof(*configuration));
    configuration->program_name = SPARK_RING_DAEMON_DEFAULT_PROGRAM;
    configuration->final_event_bind_address = "0.0.0.0";
    configuration->final_event_return_host = "spark0";
    configuration->max_active_sequence_count =
        SPARK_RING_DAEMON_DEFAULT_MAX_ACTIVE;
    configuration->port_base = SPARK_RING_RUNTIME_DEFAULT_PORT_BASE;
    configuration->model_quantization_mode =
        SPARK_RING_RUNTIME_DEFAULT_QUANTIZATION_MODE;
}

static void SparkRingDaemonSetDefaultError(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    const char *message)
{
    if (error_buffer == 0 || error_buffer_bytes == 0u ||
        error_buffer[0] != '\0')
        return;
    snprintf(error_buffer,error_buffer_bytes,"%s",message);
}

static void SparkRingDaemonSetStatusError(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    const char *message,
    SparkStatus status)
{
    if (error_buffer == 0 || error_buffer_bytes == 0u ||
        error_buffer[0] != '\0')
        return;
    snprintf(error_buffer,error_buffer_bytes,"%s status=%d",
        message,(int32_t)status);
}

#include <stddef.h>
#include "sparkpipe/spark_options.h"

static const SparkOption spark_ring_daemon_options[] =
{
	{ "--rank", SPARK_OPTION_U32, -1, (uint16_t)offsetof(SparkRingDaemonConfig, rank_index), (uint16_t)offsetof(SparkRingDaemonConfig, rank_is_set) },
	{ "--moe-pack-root", SPARK_OPTION_STRING, -2, (uint16_t)offsetof(SparkRingDaemonConfig, moe_pack_root), SPARK_OPTION_NO_SET_FLAG },
	{ "--model-quantization", SPARK_OPTION_QUANT, -14, (uint16_t)offsetof(SparkRingDaemonConfig, model_quantization_mode), SPARK_OPTION_NO_SET_FLAG },
	{ "--stagepack-root", SPARK_OPTION_STRING, -3, (uint16_t)offsetof(SparkRingDaemonConfig, stagepack_root), SPARK_OPTION_NO_SET_FLAG },
	{ "--transport-so", SPARK_OPTION_STRING, -4, (uint16_t)offsetof(SparkRingDaemonConfig, transport_shared_object_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--driver-so", SPARK_OPTION_STRING, -5, (uint16_t)offsetof(SparkRingDaemonConfig, driver_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--node-context-builder-so", SPARK_OPTION_STRING, -6, (uint16_t)offsetof(SparkRingDaemonConfig, node_context_builder_shared_object_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--embedding-pack", SPARK_OPTION_STRING, -7, (uint16_t)offsetof(SparkRingDaemonConfig, embedding_pack_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--cuda-resident-socket", SPARK_OPTION_STRING, -7, (uint16_t)offsetof(SparkRingDaemonConfig, cuda_resident_socket_path), SPARK_OPTION_NO_SET_FLAG },
	{ "--program", SPARK_OPTION_STRING, -8, (uint16_t)offsetof(SparkRingDaemonConfig, program_name), SPARK_OPTION_NO_SET_FLAG },
	{ "--node-target", SPARK_OPTION_STRING, -8, (uint16_t)offsetof(SparkRingDaemonConfig, node_target), SPARK_OPTION_NO_SET_FLAG },
	{ "--max-active", SPARK_OPTION_U32, -9, (uint16_t)offsetof(SparkRingDaemonConfig, max_active_sequence_count), SPARK_OPTION_NO_SET_FLAG },
	{ "--port-base", SPARK_OPTION_U32, -10, (uint16_t)offsetof(SparkRingDaemonConfig, port_base), SPARK_OPTION_NO_SET_FLAG },
	{ "--final-event-bind", SPARK_OPTION_STRING, -11, (uint16_t)offsetof(SparkRingDaemonConfig, final_event_bind_address), SPARK_OPTION_NO_SET_FLAG },
	{ "--final-event-return-host", SPARK_OPTION_STRING, -12, (uint16_t)offsetof(SparkRingDaemonConfig, final_event_return_host), SPARK_OPTION_NO_SET_FLAG },
	{ "--own-final-event", SPARK_OPTION_FLAG, -1, (uint16_t)offsetof(SparkRingDaemonConfig, own_final_event), SPARK_OPTION_NO_SET_FLAG },
	{ "--transport-busy-poll", SPARK_OPTION_FLAG, -1, (uint16_t)offsetof(SparkRingDaemonConfig, transport_busy_poll), SPARK_OPTION_NO_SET_FLAG },
};

static int32_t SparkRingDaemonApplyArgument(
    SparkRingDaemonConfig *configuration,
    int argc,
    char **argv,
    int32_t *index)
{
	int32_t status;
	status = SparkOptionsApply(spark_ring_daemon_options,
		(uint32_t)(sizeof(spark_ring_daemon_options) / sizeof(spark_ring_daemon_options[0])),
		configuration, argc, argv, index);
	if (status <= 0)
		return status;
	return -1;
}

static int32_t SparkRingDaemonParseArguments(
    SparkRingDaemonConfig *configuration,
    int argc,
    char **argv)
{
    int32_t index;

    for (index = 1; index < argc; ++index)
    {
        if (SparkRingDaemonApplyArgument(
                configuration,argc,argv,&index) < 0)
            return -1;
    }
    if (configuration->rank_is_set == 0u)
        return -2;
    if (SparkRingRuntimeQuantizationModeName(
            configuration->model_quantization_mode) == 0)
        return -3;
    if (configuration->cuda_resident_socket_path != 0)
        return 0;
    if (configuration->moe_pack_root == 0 ||
        configuration->stagepack_root == 0 ||
        configuration->transport_shared_object_path == 0 ||
        configuration->driver_path == 0 ||
        configuration->node_context_builder_shared_object_path == 0 ||
        configuration->embedding_pack_path == 0)
        return -2;
    return 0;
}

static void SparkRingDaemonConfigureTcpSocket(int32_t fd)
{
    (void)SparkNetConfigureLowLatencyTcp(fd);
}

static uint64_t SparkRingDaemonMinNonzeroNs(
    uint64_t left,
    uint64_t right)
{
    if (left == 0u)
        return right;
    if (right == 0u)
        return left;
    return left < right ? left : right;
}

static uint32_t SparkRingDaemonHasWaitingWork(
    const SparkRingDaemonRuntime *runtime)
{
    uint32_t slot_index;
    uint32_t scan_index;

    if (runtime == 0 || runtime->work_queue_count == 0u)
        return 0u;
    slot_index = runtime->work_queue_head;
    for (scan_index = 0u;
         scan_index < runtime->work_queue_count;
         ++scan_index)
    {
        if (runtime->work_queue_state[slot_index] !=
            SPARK_RING_DAEMON_WORK_STATE_READY)
            return 1u;
        slot_index = (slot_index + 1u) %
            SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
    }
    return 0u;
}

static uint64_t SparkRingDaemonNextTimerNs(
    const SparkRingDaemonRuntime *runtime)
{
    uint64_t next_ns;
    uint64_t now_ns;

    if (runtime == 0)
        return 0u;
    now_ns = SparkNetMonotonicNs();
    next_ns = 0u;
    if (runtime->driver_inflight_count != 0u)
        next_ns = now_ns + SPARK_RING_DAEMON_RUNNER_PROGRESS_NS;
    if (SparkRingDaemonHasWaitingWork(runtime) != 0u)
        next_ns = SparkRingDaemonMinNonzeroNs(
            next_ns,
            now_ns + SPARK_RING_DAEMON_RUNNER_PROGRESS_NS);
    if (runtime->work_queue_count != 0u)
        next_ns = SparkRingDaemonMinNonzeroNs(
            next_ns,
            runtime->work_output_retry_mono_ns);
    if (runtime->final_event_queue_count != 0u)
        next_ns = SparkRingDaemonMinNonzeroNs(
            next_ns,
            runtime->final_event_retry_mono_ns);
    return next_ns;
}

#if !defined(__linux__)
static int32_t SparkRingDaemonPollTimeoutMs(uint64_t timeout_ns)
{
    uint64_t timeout_ms;

    if (timeout_ns == 0u)
        return -1;
    timeout_ms = ((timeout_ns + 999999ull) / 1000000ull);
    if (timeout_ms == 0u)
        timeout_ms = 1u;
    if (timeout_ms > 0x7fffffffull)
        timeout_ms = 0x7fffffffull;
    return (int32_t)timeout_ms;
}
#endif

static int32_t SparkRingDaemonAppendPollFd(
    struct pollfd *fds,
    uint32_t *fd_kinds,
    uint32_t fd_capacity,
    uint32_t *fd_count,
    int32_t fd,
    int16_t events,
    uint32_t kind)
{
    if (fds == 0 || fd_count == 0)
        return -1;
    if (fd < 0 || events == 0)
        return 0;
    if (*fd_count >= fd_capacity)
        return -2;
    memset(&fds[*fd_count],0,sizeof(fds[*fd_count]));
    fds[*fd_count].fd = fd;
    fds[*fd_count].events = events;
    if (fd_kinds != 0)
        fd_kinds[*fd_count] = kind;
    *fd_count += 1u;
    return 0;
}

static int16_t SparkRingDaemonPollEventsFromTransport(
    uint32_t transport_events)
{
    int16_t events;
    events = 0;
    if ((transport_events & SPARK_HIDDEN_TRANSPORT_POLL_READ) != 0u)
        events |= POLLIN;
    if ((transport_events & SPARK_HIDDEN_TRANSPORT_POLL_WRITE) != 0u)
        events |= POLLOUT;
    return events;
}

static void SparkRingDaemonAppendTransportPollFds(
    SparkHiddenTransportSession *session,
    struct pollfd *fds,
    uint32_t *fd_kinds,
    uint32_t fd_capacity,
    uint32_t *fd_count,
    uint32_t kind)
{
    SparkHiddenTransportPollDescriptor descriptors[4];
    SparkStatus status;
    uint32_t descriptor_count;
    uint32_t descriptor_index;
    int16_t events;
    if (session == 0 || fds == 0 || fd_count == 0)
        return;
    descriptor_count = 0u;
    status = SparkHiddenTransportGetPollDescriptors(
        session,
        descriptors,
        4u,
        &descriptor_count);
    if (status != SPARK_STATUS_OK)
        return;
    for (descriptor_index = 0u;
         descriptor_index < descriptor_count;
         ++descriptor_index)
    {
        events = SparkRingDaemonPollEventsFromTransport(
            descriptors[descriptor_index].events);
        if (SparkRingDaemonAppendPollFd(
                fds,
                fd_kinds,
                fd_capacity,
                fd_count,
                descriptors[descriptor_index].fd,
                events,
                kind) < 0)
            return;
    }
}

static uint32_t SparkRingDaemonWorkInputCanRead(
    const SparkRingDaemonRuntime *runtime)
{
    return runtime != 0 && runtime->work_input_socket_fd >= 0 &&
        runtime->work_input_acknowledgement_pending == 0u;
}

static uint32_t SparkRingDaemonWaitForEvents(
    SparkRingDaemonRuntime *runtime,
    uint64_t timeout_ns)
{
    struct pollfd fds[SPARK_RING_DAEMON_POLL_FD_CAPACITY];
    uint32_t fd_kinds[SPARK_RING_DAEMON_POLL_FD_CAPACITY];
    uint32_t fd_count;
    uint32_t fd_index;
    uint32_t event_mask;
    int32_t result;
#if defined(__linux__)
    struct timespec timeout;
    struct timespec *timeout_pointer;
#endif

    if (runtime == 0)
        return 0u;
    fd_count = 0u;
    if (runtime->work_listen_fd >= 0 && runtime->work_input_socket_fd < 0)
        (void)SparkRingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_listen_fd,
            POLLIN,
            SPARK_RING_DAEMON_POLL_KIND_WORK);
    if (runtime->work_input_socket_fd >= 0)
    {
        int16_t work_input_events;

        work_input_events = runtime->work_input_acknowledgement_pending != 0u ?
            POLLOUT : POLLIN;
        (void)SparkRingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_input_socket_fd,
            work_input_events,
            SPARK_RING_DAEMON_POLL_KIND_WORK);
    }
    if (runtime->work_output_socket_fd >= 0 &&
        (runtime->work_output_connecting != 0u ||
         runtime->work_queue_count != 0u))
    {
        int16_t work_output_events;

        work_output_events = runtime->work_output_waiting_for_acknowledgement != 0u ?
            POLLIN : POLLOUT;
        if (runtime->work_output_connecting != 0u)
            work_output_events = POLLOUT;
        (void)SparkRingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->work_output_socket_fd,
            work_output_events,
            SPARK_RING_DAEMON_POLL_KIND_WORK_OUTPUT);
    }
    if (runtime->final_event_listen_fd >= 0 &&
        runtime->final_event_socket_fd < 0)
        (void)SparkRingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->final_event_listen_fd,
            POLLIN,
            SPARK_RING_DAEMON_POLL_KIND_FINAL_EVENT);
    if (runtime->final_event_socket_fd >= 0)
    {
        int16_t final_events;
        final_events = POLLIN;
        if ((runtime->rank_plan.flags &
                SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
            (runtime->final_event_socket_connecting != 0u ||
             runtime->final_event_queue_count != 0u))
            final_events |= POLLOUT;
        (void)SparkRingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->final_event_socket_fd,
            final_events,
            SPARK_RING_DAEMON_POLL_KIND_FINAL_EVENT);
    }
    if (runtime->wake_pipe_read_fd >= 0)
        (void)SparkRingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->wake_pipe_read_fd,
            POLLIN,
            SPARK_RING_DAEMON_POLL_KIND_COMPLETION_WAKE);
    if (runtime->cuda_resident_fd >= 0)
        (void)SparkRingDaemonAppendPollFd(
            fds,
            fd_kinds,
            SPARK_RING_DAEMON_POLL_FD_CAPACITY,
            &fd_count,
            runtime->cuda_resident_fd,
            POLLIN,
            SPARK_RING_DAEMON_POLL_KIND_CUDA_RESIDENT);
    SparkRingDaemonAppendTransportPollFds(
        runtime->input_transport_session,
        fds,
        fd_kinds,
        SPARK_RING_DAEMON_POLL_FD_CAPACITY,
        &fd_count,
        SPARK_RING_DAEMON_POLL_KIND_INPUT_TRANSPORT);
    SparkRingDaemonAppendTransportPollFds(
        runtime->output_transport_session,
        fds,
        fd_kinds,
        SPARK_RING_DAEMON_POLL_FD_CAPACITY,
        &fd_count,
        SPARK_RING_DAEMON_POLL_KIND_OUTPUT_TRANSPORT);
    if (fd_count == 0u)
        return 0u;
#if defined(__linux__)
    timeout_pointer = 0;
    if (timeout_ns != 0u)
    {
        timeout.tv_sec = (time_t)(timeout_ns / 1000000000ull);
        timeout.tv_nsec = (long)(timeout_ns % 1000000000ull);
        timeout_pointer = &timeout;
    }
#endif
    for (;;)
    {
#if defined(__linux__)
        result = ppoll(fds,fd_count,timeout_pointer,0);
#else
        result = poll(fds,fd_count,
            SparkRingDaemonPollTimeoutMs(timeout_ns));
#endif
        if (result < 0 && errno == EINTR)
            return 0u;
        if (result == 0)
            return SPARK_RING_DAEMON_POLL_KIND_TIMER;
        if (result < 0)
            return 0u;
        event_mask = 0u;
        for (fd_index = 0u; fd_index < fd_count; ++fd_index)
        {
            if (fds[fd_index].revents != 0)
                event_mask |= fd_kinds[fd_index];
        }
        return event_mask;
    }
}

static int32_t SparkRingDaemonStartConnect(
    const char *host,
    uint32_t port,
    uint32_t *connecting_out)
{
    struct addrinfo hints;
    struct addrinfo *results;
    struct addrinfo *entry;
    char service[16];
    int32_t fd;
    int32_t connect_status;

    if (host == 0 || connecting_out == 0)
        return -1;
    *connecting_out = 0u;
    if (snprintf(service,sizeof(service),"%u",port) <= 0)
        return -2;
    memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host,service,&hints,&results) != 0)
        return -3;
    fd = -1;
    for (entry = results; entry != 0; entry = entry->ai_next)
    {
        fd = socket(entry->ai_family,entry->ai_socktype,entry->ai_protocol);
        if (fd < 0)
            continue;
        if (SparkNetSetNonblocking(fd) < 0)
        {
            close(fd);
            fd = -1;
            continue;
        }
        SparkRingDaemonConfigureTcpSocket(fd);
        connect_status = connect(fd,entry->ai_addr,entry->ai_addrlen);
        if (connect_status == 0)
            break;
        if (errno == EINPROGRESS)
        {
            *connecting_out = 1u;
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    return fd;
}

static SparkStatus SparkRingDaemonFinishConnect(
    int32_t *fd,
    uint32_t *connecting)
{
    socklen_t error_bytes;
    int32_t error_value;

    if (fd == 0 || connecting == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (*fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    if (*connecting == 0u)
        return SPARK_STATUS_OK;
    error_value = 0;
    error_bytes = (socklen_t)sizeof(error_value);
    if (getsockopt(*fd,SOL_SOCKET,SO_ERROR,&error_value,&error_bytes) < 0)
    {
        close(*fd);
        *fd = -1;
        *connecting = 0u;
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    if (error_value == 0)
    {
        *connecting = 0u;
        SparkRingDaemonConfigureTcpSocket(*fd);
        return SPARK_STATUS_OK;
    }
    close(*fd);
    *fd = -1;
    *connecting = 0u;
    return SPARK_STATUS_ROUTE_NOT_FOUND;
}

static SparkStatus SparkRingDaemonWriteBuffered(
    int32_t fd,
    const void *buffer,
    uint32_t buffer_bytes,
    uint32_t *offset)
{
    const uint8_t *cursor;
    ssize_t written;

    if (fd < 0 || buffer == 0 || offset == 0 || *offset > buffer_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (const uint8_t *)buffer;
    while (*offset < buffer_bytes)
    {
        written = write(fd,cursor + *offset,buffer_bytes - *offset);
        if (written < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                return SPARK_STATUS_BUSY;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        if (written == 0)
            return SPARK_STATUS_BUSY;
        *offset += (uint32_t)written;
    }
    *offset = 0u;
    return SPARK_STATUS_OK;
}


static SparkStatus SparkRingDaemonWriteFull(
    int32_t fd,
    const void *buffer,
    uint32_t buffer_bytes)
{
    const uint8_t *cursor;
    uint32_t offset;
    ssize_t written;

    if (fd < 0 || buffer == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (const uint8_t *)buffer;
    offset = 0u;
    while (offset < buffer_bytes)
    {
        written = write(fd,cursor + offset,buffer_bytes - offset);
        if (written < 0)
        {
            if (errno == EINTR)
                continue;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        if (written == 0)
            return SPARK_STATUS_BUSY;
        offset += (uint32_t)written;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonReadFull(
    int32_t fd,
    void *buffer,
    uint32_t buffer_bytes)
{
    uint8_t *cursor;
    uint32_t offset;
    ssize_t got;

    if (fd < 0 || buffer == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (uint8_t *)buffer;
    offset = 0u;
    while (offset < buffer_bytes)
    {
        got = read(fd,cursor + offset,buffer_bytes - offset);
        if (got < 0)
        {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return SPARK_STATUS_BUSY;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        if (got == 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        offset += (uint32_t)got;
    }
    return SPARK_STATUS_OK;
}

static void SparkRingDaemonResetResidentRead(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0)
        return;
    runtime->cuda_resident_read_header_offset = 0u;
    runtime->cuda_resident_read_payload_offset = 0u;
}

static SparkStatus SparkRingDaemonReadResidentBytes(
    SparkRingDaemonRuntime *runtime,
    void *buffer,
    uint32_t buffer_bytes,
    uint32_t *offset,
    uint32_t timeout_ms)
{
    struct pollfd descriptor;
    uint8_t *cursor;
    ssize_t got;

    if (runtime == 0 || runtime->cuda_resident_fd < 0 || buffer == 0 ||
        offset == 0 || *offset > buffer_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    cursor = (uint8_t *)buffer;
    descriptor.fd = runtime->cuda_resident_fd;
    descriptor.events = POLLIN;
    while (*offset < buffer_bytes)
    {
        got = read(
            runtime->cuda_resident_fd,
            cursor + *offset,
            buffer_bytes - *offset);
        if (got > 0)
        {
            *offset += (uint32_t)got;
            continue;
        }
        if (got == 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        if (errno == EINTR)
            continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        if (timeout_ms == 0u)
            return SPARK_STATUS_BUSY;
        descriptor.revents = 0;
        if (poll(&descriptor,1u,(int32_t)timeout_ms) <= 0)
            return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonReadResidentMessage(
    SparkRingDaemonRuntime *runtime,
    uint32_t timeout_ms,
    SparkCudaResidentIpcHeader *header_out)
{
    SparkStatus status;
    if (runtime == 0 || header_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkRingDaemonReadResidentBytes(
        runtime,&runtime->cuda_resident_read_header,
        sizeof(runtime->cuda_resident_read_header),
        &runtime->cuda_resident_read_header_offset,timeout_ms);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkCudaResidentIpcValidateHeader(
        &runtime->cuda_resident_read_header,0u,
        SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkRingDaemonReadResidentBytes(
        runtime,runtime->cuda_resident_payload,
        runtime->cuda_resident_read_header.payload_bytes,
        &runtime->cuda_resident_read_payload_offset,timeout_ms);
    if (status != SPARK_STATUS_OK)
        return status;
    *header_out = runtime->cuda_resident_read_header;
    SparkRingDaemonResetResidentRead(runtime);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonWriteResidentMessage(
    SparkRingDaemonRuntime *runtime,
    uint32_t kind,
    const void *payload,
    uint32_t payload_bytes)
{
    SparkCudaResidentIpcHeader header;
    SparkStatus status;

    if (runtime == 0 || runtime->cuda_resident_fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    SparkCudaResidentIpcInitializeHeader(
        &header,
        kind,
        runtime->rank_plan.rank_index,
        runtime->cuda_resident_next_sequence_number++,
        payload_bytes);
    status = SparkRingDaemonWriteFull(
        runtime->cuda_resident_fd,
        &header,
        sizeof(header));
    if (status != SPARK_STATUS_OK)
        return status;
    if (payload_bytes != 0u)
        status = SparkRingDaemonWriteFull(
            runtime->cuda_resident_fd,
            payload,
            payload_bytes);
    return status;
}


static SparkStatus SparkRingDaemonConnectCudaResident(
    SparkRingDaemonRuntime *runtime,
    const char *socket_path)
{
    struct sockaddr_un address;
    SparkCudaResidentIpcHello hello;
    SparkCudaResidentIpcHeader header;
    SparkCudaResidentIpcStats stats;
    SparkStatus status;
    uint32_t expected_moe_backend_kind;
    int32_t fd;

    if (runtime == 0 || socket_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkRingRuntimeExpectedMoeBackendKind(
        runtime->rank_plan.quantization_mode,
        &expected_moe_backend_kind);
    if (status != SPARK_STATUS_OK)
        return status;
    fd = socket(AF_UNIX,SOCK_STREAM,0);
    if (fd < 0)
        return SPARK_STATUS_IO_ERROR;
    memset(&address,0,sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof(address.sun_path))
    {
        close(fd);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    snprintf(address.sun_path,sizeof(address.sun_path),"%s",socket_path);
    if (connect(fd,(const struct sockaddr *)&address,sizeof(address)) != 0)
    {
        close(fd);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->cuda_resident_fd = fd;
    memset(&hello,0,sizeof(hello));
    hello.descriptor_bytes = SPARK_CUDA_RESIDENT_IPC_HELLO_BYTES;
    hello.rank_index = runtime->rank_plan.rank_index;
    hello.rank_count = SPARK_RING_RUNTIME_STAGE_COUNT;
    hello.process_id = (uint64_t)getpid();
    status = SparkRingDaemonWriteResidentMessage(
        runtime,
        SPARK_CUDA_RESIDENT_IPC_KIND_HELLO,
        &hello,
        sizeof(hello));
    if (status == SPARK_STATUS_OK)
        status = SparkRingDaemonReadFull(fd,&header,sizeof(header));
    if (status == SPARK_STATUS_OK)
        status = SparkCudaResidentIpcValidateHeader(
            &header,
            SPARK_CUDA_RESIDENT_IPC_KIND_HELLO_ACK,
            sizeof(stats));
    if (status == SPARK_STATUS_OK && header.payload_bytes != sizeof(stats))
        status = SPARK_STATUS_ABI_MISMATCH;
    if (status == SPARK_STATUS_OK)
        status = SparkRingDaemonReadFull(fd,&stats,sizeof(stats));
    if (status == SPARK_STATUS_OK &&
        stats.state != SPARK_CUDA_RESIDENT_IPC_STATE_READY)
        status = SPARK_STATUS_BUSY;
    if (status == SPARK_STATUS_OK &&
        (stats.logical_lane_capacity != runtime->rank_plan.logical_lane_capacity ||
         stats.execution_row_capacity != runtime->rank_plan.execution_row_capacity ||
         stats.kv_physical_block_capacity == 0u ||
         stats.kv_logical_block_capacity < stats.kv_physical_block_capacity ||
         stats.model_quantization_mode != runtime->rank_plan.quantization_mode ||
         stats.moe_backend_kind != expected_moe_backend_kind ||
         stats.moe_bound_layer_count == 0u ||
         stats.moe_bound_layer_count != stats.moe_expected_layer_count ||
         (stats.kv_nvme_enabled != 0u &&
          stats.kv_nvme_mode !=
            SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_SYNCHRONOUS_FULL_HISTORY &&
          stats.kv_nvme_mode !=
            SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT &&
          stats.kv_nvme_mode !=
            SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT) ||
         (runtime->rank_plan.logical_lane_capacity >=
            SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT &&
          (stats.kv_nvme_enabled == 0u ||
           (stats.kv_nvme_mode !=
                SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT &&
            stats.kv_nvme_mode !=
                SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT) ||
           stats.kv_resident_bytes_per_token == 0u ||
           stats.kv_resident_pool_bytes == 0u ||
           stats.kv_nvme_capacity_bytes == 0u ||
           stats.kv_nvme_batch_block_capacity == 0u))))
    {
        fprintf(
            stderr,
            "rank_cuda_resident_contract_mismatch rank=%u logical=%u/%u "
			"execution=%u/%u kv_blocks=%u logical_blocks=%u "
			"quantization=%u/%u moe_backend=%u moe_layers=%u/%u nvme=%u "
            "nvme_mode=%u "
            "blocker=%.*s\n",
            runtime->rank_plan.rank_index,
            stats.logical_lane_capacity,
            runtime->rank_plan.logical_lane_capacity,
            stats.execution_row_capacity,
            runtime->rank_plan.execution_row_capacity,
			stats.kv_physical_block_capacity,
			stats.kv_logical_block_capacity,
            stats.model_quantization_mode,
            runtime->rank_plan.quantization_mode,
            stats.moe_backend_kind,
            stats.moe_bound_layer_count,
            stats.moe_expected_layer_count,
            stats.kv_nvme_enabled,
            stats.kv_nvme_mode,
            (int32_t)sizeof(stats.blocker),
            stats.blocker);
        status = SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (status == SPARK_STATUS_OK &&
        SparkNetSetNonblocking(fd) < 0)
        status = SPARK_STATUS_INTERNAL_ERROR;
    if (status != SPARK_STATUS_OK)
    {
        close(fd);
        runtime->cuda_resident_fd = -1;
        return status;
    }
    return SPARK_STATUS_OK;
}

static void SparkRingDaemonTeardownCudaResident(
    SparkRingDaemonRuntime *runtime,
    const char *reason)
{
    if (runtime == 0 || runtime->cuda_resident_fd < 0)
        return;
    fprintf(stderr,"rank_cuda_resident_disconnected rank=%u reason=%s\n",
        runtime->rank_plan.rank_index,reason);
    close(runtime->cuda_resident_fd);
    runtime->cuda_resident_fd = -1;
    SparkRingDaemonResetResidentRead(runtime);
    runtime->cuda_resident_error_count += 1u;
}

static SparkStatus SparkRingDaemonEnsureCudaResident(
    SparkRingDaemonRuntime *runtime)
{
    uint64_t now_ns;
    SparkStatus status;

    if (runtime == 0 || runtime->cuda_resident_socket_path == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (runtime->cuda_resident_fd >= 0)
        return SPARK_STATUS_OK;
    now_ns = SparkNetMonotonicNs();
    if (now_ns < runtime->cuda_resident_retry_after_ns)
        return SPARK_STATUS_BUSY;
    runtime->cuda_resident_retry_after_ns = now_ns + 250000000ull;
    status = SparkRingDaemonConnectCudaResident(
        runtime,
        runtime->cuda_resident_socket_path);
    if (status == SPARK_STATUS_OK)
    {
        fprintf(stderr,"rank_cuda_resident_connected rank=%u\n",
            runtime->rank_plan.rank_index);
        return SPARK_STATUS_OK;
    }
    fprintf(stderr,"rank_cuda_resident_connect_retry rank=%u status=%u\n",
        runtime->rank_plan.rank_index,(uint32_t)status);
    return SPARK_STATUS_BUSY;
}

static uint32_t SparkRingDaemonHandleCudaResidentMessage(
    SparkRingDaemonRuntime *runtime,
    const SparkCudaResidentIpcHeader *header,
    const uint8_t *payload)
{
    const SparkCudaResidentIpcCompletion *completion_message;
    const SparkCudaResidentIpcStats *stats;

    if (runtime == 0 || header == 0)
        return 0u;
    if (header->kind == SPARK_CUDA_RESIDENT_IPC_KIND_COMPLETION)
    {
        if (payload == 0 || header->payload_bytes !=
            SPARK_CUDA_RESIDENT_IPC_COMPLETION_BYTES)
        {
            runtime->cuda_resident_error_count += 1u;
            return 0u;
        }
        completion_message =
            (const SparkCudaResidentIpcCompletion *)payload;
        if (completion_message->descriptor_bytes !=
                SPARK_CUDA_RESIDENT_IPC_COMPLETION_BYTES ||
            (completion_message->flags &
                ~SPARK_CUDA_RESIDENT_IPC_COMPLETION_KNOWN_FLAGS) != 0u)
        {
            runtime->cuda_resident_error_count += 1u;
            return 0u;
        }
        if ((completion_message->flags &
                SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT) != 0u)
        {
            if (completion_message->dspark_draft.abi_version !=
                    SPARK_RING_DAEMON_DRAFT_ABI_VERSION ||
                completion_message->dspark_draft.descriptor_bytes !=
                    SPARK_RING_DAEMON_DRAFT_DESCRIPTOR_BYTES)
            {
                runtime->cuda_resident_error_count += 1u;
                return 0u;
            }
        }
        runtime->cuda_resident_completion_count += 1u;
        if (SparkRingDaemonQueueDriverCompletion(
                runtime,
                &completion_message->completion,
                (completion_message->flags &
                    SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT) != 0u ?
                    &completion_message->dspark_draft : 0) !=
            SPARK_STATUS_OK)
        {
            runtime->cuda_resident_error_count += 1u;
        }
        return 1u;
    }
    if (header->kind == SPARK_CUDA_RESIDENT_IPC_KIND_STATS)
    {
        if (payload == 0 || header->payload_bytes !=
            SPARK_CUDA_RESIDENT_IPC_STATS_BYTES)
            runtime->cuda_resident_error_count += 1u;
        else
        {
            stats = (const SparkCudaResidentIpcStats *)payload;
            if (stats->state == SPARK_CUDA_RESIDENT_IPC_STATE_FAILED)
                runtime->cuda_resident_error_count += 1u;
        }
        return 1u;
    }
    if (header->kind == SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT)
    {
        const SparkCudaResidentIpcSubmitResult *result;
        if (payload == 0 || header->payload_bytes !=
            SPARK_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES)
        {
            runtime->cuda_resident_error_count += 1u;
            return 0u;
        }
        result = (const SparkCudaResidentIpcSubmitResult *)payload;
        if (result->status != (uint32_t)SPARK_STATUS_OK)
            runtime->cuda_resident_error_count += 1u;
        return 1u;
    }
    runtime->cuda_resident_error_count += 1u;
    return 0u;
}

static void SparkRingDaemonCheckInflightOverdue(
    SparkRingDaemonRuntime *runtime)
{
    uint64_t now_ns;
    if (runtime == 0 || runtime->driver_inflight_count == 0u ||
        runtime->driver_inflight_warned != 0u)
        return;
    now_ns = SparkNetMonotonicNs();
    if (now_ns - runtime->driver_inflight_open_ns <= 30000000000ull)
        return;
    runtime->driver_inflight_warned = 1u;
    fprintf(stderr,"rank_completion_overdue rank=%u inflight=%u age_s=%llu submitted=%llu completed=%llu\n",runtime->rank_plan.rank_index,runtime->driver_inflight_count,(unsigned long long)((now_ns - runtime->driver_inflight_open_ns) / 1000000000ull),(unsigned long long)runtime->cuda_resident_submit_count,(unsigned long long)runtime->driver_completion_count);
}

static uint32_t SparkRingDaemonPumpCudaResident(
    SparkRingDaemonRuntime *runtime)
{
    SparkCudaResidentIpcHeader header;
    SparkStatus status;

    if (runtime == 0 || runtime->cuda_resident_fd < 0)
        return 0u;
    status = SparkRingDaemonReadResidentMessage(runtime,0u,&header);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonTeardownCudaResident(runtime,"pump_message_read");
        return 1u;
    }
    (void)SparkRingDaemonHandleCudaResidentMessage(
        runtime,
        &header,
        header.payload_bytes == 0u ? 0 : runtime->cuda_resident_payload);
    return 1u;
}

static SparkStatus SparkRingDaemonOpenWakePipe(
    SparkRingDaemonRuntime *runtime)
{
    int32_t pipe_fds[2];

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (pipe(pipe_fds) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (SparkNetSetNonblocking(pipe_fds[0]) < 0 ||
        SparkNetSetNonblocking(pipe_fds[1]) < 0)
    {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->wake_pipe_read_fd = pipe_fds[0];
    runtime->wake_pipe_write_fd = pipe_fds[1];
    return SPARK_STATUS_OK;
}

static void SparkRingDaemonSignalWake(
    SparkRingDaemonRuntime *runtime)
{
    uint8_t byte;
    ssize_t wrote;

    if (runtime == 0 || runtime->wake_pipe_write_fd < 0)
        return;
    byte = 1u;
    wrote = write(runtime->wake_pipe_write_fd,&byte,1u);
    if (wrote == 1)
    {
        (void)__atomic_fetch_add(
            &runtime->wake_signal_count,
            1u,
            __ATOMIC_RELAXED);
        return;
    }
    if (wrote < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
        (void)__atomic_fetch_add(
            &runtime->wake_drop_count,
            1u,
            __ATOMIC_RELAXED);
}

static SparkStatus SparkRingDaemonInitializeCompletionState(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(
        runtime->inflight_transactions,
        0,
        sizeof(runtime->inflight_transactions));
    memset(
        runtime->inflight_completions,
        0,
        sizeof(runtime->inflight_completions));
    memset(
        runtime->driver_completion_queue,
        0,
        sizeof(runtime->driver_completion_queue));
    runtime->driver_completion_queue_head = 0u;
    runtime->driver_completion_queue_count = 0u;
    runtime->driver_completion_queue_overflow = 0u;
    if (pthread_mutex_init(&runtime->driver_completion_mutex,0) != 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->driver_completion_mutex_initialized = 1u;
    if (pthread_mutex_init(&runtime->builder_completion_mutex,0) != 0)
    {
        (void)pthread_mutex_destroy(&runtime->driver_completion_mutex);
        runtime->driver_completion_mutex_initialized = 0u;
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->builder_completion_mutex_initialized = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonQueueDriverCompletion(
    SparkRingDaemonRuntime *runtime,
    const SparkModelDriverCompletion *completion,
    const SparkRingDaemonDraftResult *dspark_draft)
{
    SparkRingDaemonDriverCompletionRecord *record;
    uint32_t tail;

    if (runtime == 0 || completion == 0 ||
        runtime->driver_completion_mutex_initialized == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (dspark_draft != 0 &&
        (dspark_draft->abi_version != SPARK_RING_DAEMON_DRAFT_ABI_VERSION ||
         dspark_draft->descriptor_bytes !=
            SPARK_RING_DAEMON_DRAFT_DESCRIPTOR_BYTES))
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (pthread_mutex_lock(&runtime->driver_completion_mutex) != 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (runtime->driver_completion_queue_count >=
        SPARK_RING_DAEMON_DRIVER_COMPLETION_QUEUE_CAPACITY)
    {
        runtime->driver_completion_queue_overflow = 1u;
        (void)pthread_mutex_unlock(&runtime->driver_completion_mutex);
        SparkRingDaemonSignalWake(runtime);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    tail = (runtime->driver_completion_queue_head +
        runtime->driver_completion_queue_count) %
        SPARK_RING_DAEMON_DRIVER_COMPLETION_QUEUE_CAPACITY;
    record = &runtime->driver_completion_queue[tail];
    memset(record,0,sizeof(*record));
    record->completion = *completion;
    if (dspark_draft != 0)
    {
        record->dspark_draft = *dspark_draft;
        record->dspark_draft_valid = 1u;
    }
    runtime->driver_completion_queue_count += 1u;
    (void)pthread_mutex_unlock(&runtime->driver_completion_mutex);
    SparkRingDaemonSignalWake(runtime);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonPeekDriverCompletion(
    SparkRingDaemonRuntime *runtime,
    SparkRingDaemonDriverCompletionRecord *record_out)
{
    if (runtime == 0 || record_out == 0 ||
        runtime->driver_completion_mutex_initialized == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (pthread_mutex_lock(&runtime->driver_completion_mutex) != 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (runtime->driver_completion_queue_count == 0u)
    {
        (void)pthread_mutex_unlock(&runtime->driver_completion_mutex);
        return SPARK_STATUS_NOT_FOUND;
    }
    *record_out = runtime->driver_completion_queue[
        runtime->driver_completion_queue_head];
    (void)pthread_mutex_unlock(&runtime->driver_completion_mutex);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonPopDriverCompletion(
    SparkRingDaemonRuntime *runtime)
{
    SparkRingDaemonDriverCompletionRecord *record;

    if (runtime == 0 ||
        runtime->driver_completion_mutex_initialized == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (pthread_mutex_lock(&runtime->driver_completion_mutex) != 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (runtime->driver_completion_queue_count == 0u)
    {
        (void)pthread_mutex_unlock(&runtime->driver_completion_mutex);
        return SPARK_STATUS_NOT_FOUND;
    }
    record = &runtime->driver_completion_queue[
        runtime->driver_completion_queue_head];
    memset(record,0,sizeof(*record));
    runtime->driver_completion_queue_head =
        (runtime->driver_completion_queue_head + 1u) %
        SPARK_RING_DAEMON_DRIVER_COMPLETION_QUEUE_CAPACITY;
    runtime->driver_completion_queue_count -= 1u;
    (void)pthread_mutex_unlock(&runtime->driver_completion_mutex);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonQueueFinalEvent(
    SparkRingDaemonRuntime *runtime,
    const SparkRingRuntimeFinalEvent *event)
{
    uint32_t tail;

    if (runtime == 0 || event == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (runtime->final_event_queue_count >=
        SPARK_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    tail =
        (runtime->final_event_queue_head + runtime->final_event_queue_count) %
        SPARK_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY;
    runtime->final_event_queue[tail] = *event;
    runtime->final_event_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkRingDaemonPopFinalEvent(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->final_event_queue_count == 0u)
        return;
    runtime->final_event_queue_head =
        (runtime->final_event_queue_head + 1u) %
        SPARK_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY;
    runtime->final_event_queue_count -= 1u;
}

static uint64_t SparkRingDaemonInflightCompletionHash(
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t sequence_position)
{
    uint64_t words[3];

    words[0u] = request_id;
    words[1u] = sequence_id;
    words[2u] = sequence_position;
    return SparkWorkTransactionFingerprintBytes(
        words,
        (uint32_t)sizeof(words));
}

static uint32_t SparkRingDaemonInflightCompletionMatches(
    const SparkRingDaemonInflightCompletion *entry,
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t sequence_position)
{
    return entry != 0 &&
        entry->state == SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_ACTIVE &&
        entry->request_id == request_id &&
        entry->sequence_id == sequence_id &&
        entry->sequence_position == sequence_position;
}

static SparkStatus SparkRingDaemonFindInflightCompletion(
    SparkRingDaemonRuntime *runtime,
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint32_t *completion_index_out)
{
    uint64_t hash;
    uint32_t completion_index;
    uint32_t probe_index;
    uint32_t start_index;

    if (runtime == 0 || completion_index_out == 0 ||
        request_id == 0u || sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    hash = SparkRingDaemonInflightCompletionHash(
        request_id,
        sequence_id,
        sequence_position);
    start_index = (uint32_t)(hash %
        SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY);
    for (probe_index = 0u;
         probe_index < SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY;
         ++probe_index)
    {
        SparkRingDaemonInflightCompletion *entry;

        completion_index = (start_index + probe_index) %
            SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY;
        entry = &runtime->inflight_completions[completion_index];
        if (entry->state ==
            SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_FREE)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        if (SparkRingDaemonInflightCompletionMatches(
                entry,
                request_id,
                sequence_id,
                sequence_position) != 0u)
        {
            *completion_index_out = completion_index;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}

static SparkStatus SparkRingDaemonInsertInflightCompletion(
    SparkRingDaemonRuntime *runtime,
    uint32_t transaction_index,
    const SparkRingWorkControlLane *lane,
    uint32_t *completion_index_out)
{
    uint64_t hash;
    uint32_t completion_index;
    uint32_t first_tombstone_index;
    uint32_t probe_index;
    uint32_t start_index;

    if (runtime == 0 || lane == 0 || completion_index_out == 0 ||
        transaction_index >= SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY ||
        lane->request_id == 0u || lane->request_generation == 0u ||
        lane->sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    hash = SparkRingDaemonInflightCompletionHash(
        lane->request_id,
        lane->sequence_id,
        lane->sequence_position);
    start_index = (uint32_t)(hash %
        SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY);
    first_tombstone_index = SPARK_DISTRIBUTED_WORK_INVALID_INDEX;
    for (probe_index = 0u;
         probe_index < SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY;
         ++probe_index)
    {
        SparkRingDaemonInflightCompletion *entry;

        completion_index = (start_index + probe_index) %
            SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY;
        entry = &runtime->inflight_completions[completion_index];
        if (entry->state ==
                SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_TOMBSTONE &&
            first_tombstone_index == SPARK_DISTRIBUTED_WORK_INVALID_INDEX)
        {
            first_tombstone_index = completion_index;
            continue;
        }
        if (entry->state ==
            SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_ACTIVE)
        {
            if (SparkRingDaemonInflightCompletionMatches(
                    entry,
                    lane->request_id,
                    lane->sequence_id,
                    lane->sequence_position) != 0u)
            {
                return SPARK_STATUS_BUSY;
            }
            continue;
        }
        if (entry->state ==
            SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_FREE)
        {
            if (first_tombstone_index !=
                SPARK_DISTRIBUTED_WORK_INVALID_INDEX)
            {
                completion_index = first_tombstone_index;
                entry = &runtime->inflight_completions[completion_index];
            }
            memset(entry,0,sizeof(*entry));
            entry->state =
                SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_ACTIVE;
            entry->transaction_index = transaction_index;
            entry->request_id = lane->request_id;
            entry->request_generation = lane->request_generation;
            entry->sequence_id = lane->sequence_id;
            entry->sequence_position = lane->sequence_position;
            *completion_index_out = completion_index;
            return SPARK_STATUS_OK;
        }
    }
    if (first_tombstone_index != SPARK_DISTRIBUTED_WORK_INVALID_INDEX)
    {
        SparkRingDaemonInflightCompletion *entry;

        completion_index = first_tombstone_index;
        entry = &runtime->inflight_completions[completion_index];
        memset(entry,0,sizeof(*entry));
        entry->state = SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_ACTIVE;
        entry->transaction_index = transaction_index;
        entry->request_id = lane->request_id;
        entry->request_generation = lane->request_generation;
        entry->sequence_id = lane->sequence_id;
        entry->sequence_position = lane->sequence_position;
        *completion_index_out = completion_index;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}

static void SparkRingDaemonRemoveInflightCompletion(
    SparkRingDaemonRuntime *runtime,
    uint32_t completion_index)
{
    SparkRingDaemonInflightCompletion *entry;

    if (runtime == 0 ||
        completion_index >= SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY)
    {
        return;
    }
    entry = &runtime->inflight_completions[completion_index];
    memset(entry,0,sizeof(*entry));
    entry->state = SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_TOMBSTONE;
}

static SparkStatus SparkRingDaemonAllocateInflightTransaction(
    SparkRingDaemonRuntime *runtime,
    uint32_t *transaction_index_out)
{
    uint32_t transaction_index;

    if (runtime == 0 || transaction_index_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (transaction_index = 0u;
         transaction_index < SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY;
         ++transaction_index)
    {
        if (runtime->inflight_transactions[transaction_index].active == 0u)
        {
            memset(
                &runtime->inflight_transactions[transaction_index],
                0,
                sizeof(runtime->inflight_transactions[transaction_index]));
            runtime->inflight_transactions[transaction_index].active = 1u;
            *transaction_index_out = transaction_index;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}

static void SparkRingDaemonReleaseInflightTransaction(
    SparkRingDaemonRuntime *runtime,
    uint32_t transaction_index)
{
    if (runtime == 0 ||
        transaction_index >= SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY)
    {
        return;
    }
    memset(
        &runtime->inflight_transactions[transaction_index],
        0,
        sizeof(runtime->inflight_transactions[transaction_index]));
}

static uint32_t SparkRingDaemonRemoveTransactionCompletionMappings(
    SparkRingDaemonRuntime *runtime,
    uint32_t transaction_index)
{
    uint32_t completion_index;
    uint32_t removed_count;

    if (runtime == 0 ||
        transaction_index >= SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY)
    {
        return 0u;
    }
    removed_count = 0u;
    for (completion_index = 0u;
         completion_index < SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY;
         ++completion_index)
    {
        SparkRingDaemonInflightCompletion *entry;

        entry = &runtime->inflight_completions[completion_index];
        if (entry->state ==
                SPARK_RING_DAEMON_INFLIGHT_COMPLETION_STATE_ACTIVE &&
            entry->transaction_index == transaction_index)
        {
            SparkRingDaemonRemoveInflightCompletion(runtime,completion_index);
            removed_count += 1u;
        }
    }
    return removed_count;
}

static SparkStatus SparkRingDaemonRegisterInflightTransaction(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet,
    uint32_t *transaction_index_out)
{
    SparkRingDaemonInflightTransaction *transaction;
    SparkDistributedWorkIdentity identity;
    SparkStatus status;
    uint64_t packet_hash;
    uint32_t completion_index;
    uint32_t lane_index;
    uint32_t transaction_index;

    if (runtime == 0 || packet == 0 || transaction_index_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *transaction_index_out = SPARK_DISTRIBUTED_WORK_INVALID_INDEX;
    if ((packet->flags &
            SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (packet->lane_count == 0u ||
        packet->lane_count > SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkRingWorkControlGetTransactionIdentity(packet,&identity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    packet_hash = SparkRingWorkControlPacketFingerprint(packet);
    if (packet_hash == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkRingDaemonAllocateInflightTransaction(
        runtime,
        &transaction_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    transaction = &runtime->inflight_transactions[transaction_index];
    transaction->identity = identity;
    transaction->packet_hash = packet_hash;
    transaction->remaining_completion_count = packet->lane_count;
    transaction->terminal_state = 0u;
    transaction->terminal_status = (uint32_t)SPARK_STATUS_PENDING;
    for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
    {
        status = SparkRingDaemonInsertInflightCompletion(
            runtime,
            transaction_index,
            &packet->lanes[lane_index],
            &completion_index);
        if (status != SPARK_STATUS_OK)
        {
            (void)SparkRingDaemonRemoveTransactionCompletionMappings(
                runtime,
                transaction_index);
            SparkRingDaemonReleaseInflightTransaction(
                runtime,
                transaction_index);
            return status;
        }
    }
    if (runtime->driver_inflight_count == 0u)
    {
        runtime->driver_inflight_open_ns = SparkNetMonotonicNs();
        runtime->driver_inflight_warned = 0u;
    }
    if (packet->lane_count > UINT32_MAX - runtime->driver_inflight_count)
    {
        (void)SparkRingDaemonRemoveTransactionCompletionMappings(
            runtime,
            transaction_index);
        SparkRingDaemonReleaseInflightTransaction(runtime,transaction_index);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    runtime->driver_inflight_count += packet->lane_count;
    *transaction_index_out = transaction_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonCancelInflightTransaction(
    SparkRingDaemonRuntime *runtime,
    uint32_t transaction_index)
{
    SparkRingDaemonInflightTransaction *transaction;
    uint32_t removed_count;

    if (runtime == 0 ||
        transaction_index >= SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    transaction = &runtime->inflight_transactions[transaction_index];
    if (transaction->active == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    removed_count = SparkRingDaemonRemoveTransactionCompletionMappings(
        runtime,
        transaction_index);
    if (removed_count > runtime->driver_inflight_count)
    {
        runtime->driver_inflight_count = 0u;
    }
    else
    {
        runtime->driver_inflight_count -= removed_count;
    }
    SparkRingDaemonReleaseInflightTransaction(runtime,transaction_index);
    if (runtime->driver_inflight_count == 0u)
    {
        runtime->driver_inflight_warned = 0u;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkRingDaemonDrainWakePipe(
    SparkRingDaemonRuntime *runtime)
{
    uint8_t buffer[64];
    ssize_t got;
    uint32_t progress;

    progress = 0u;
    if (runtime == 0 || runtime->wake_pipe_read_fd < 0)
        return 0u;
    for (;;)
    {
        got = read(runtime->wake_pipe_read_fd,buffer,sizeof(buffer));
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
            return progress;
        if (got <= 0)
            return progress;
        progress = 1u;
    }
}

static SparkStatus SparkRingDaemonValidateDriverCompletionRecord(
    const SparkRingDaemonDriverCompletionRecord *record)
{
    const SparkModelDriverCompletion *completion;
    uint32_t has_draft_tokens;
    uint32_t has_token_ids;

    if (record == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    completion = &record->completion;
    if (completion->request_id == 0u || completion->sequence_id == 0u ||
        completion->token_count >
            SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY ||
        completion->draft_token_count >
            SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    has_token_ids = (completion->completion_flags &
        SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) != 0u;
    has_draft_tokens = (completion->completion_flags &
        SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS) != 0u;
    if ((has_token_ids == 0u) != (completion->token_count == 0u) ||
        (has_draft_tokens == 0u) !=
            (completion->draft_token_count == 0u))
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (record->dspark_draft_valid != 0u &&
        (record->dspark_draft.abi_version !=
            SPARK_RING_DAEMON_DRAFT_ABI_VERSION ||
         record->dspark_draft.descriptor_bytes !=
            SPARK_RING_DAEMON_DRAFT_DESCRIPTOR_BYTES))
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkRingDaemonCompletionNeedsFinalEvent(
    const SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonDriverCompletionRecord *record,
    SparkStatus effective_status,
    uint32_t transaction_terminal_state)
{
    if (runtime == 0 || record == 0 ||
        (runtime->rank_plan.flags &
            SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u ||
        transaction_terminal_state != 0u)
    {
        return 0u;
    }
    if (effective_status != SPARK_STATUS_OK)
    {
        return 1u;
    }
    return (record->completion.completion_flags &
            SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) != 0u &&
        record->completion.token_count != 0u;
}

static void SparkRingDaemonBuildFinalEvent(
    const SparkRingDaemonInflightTransaction *transaction,
    const SparkRingDaemonInflightCompletion *inflight_completion,
    const SparkRingDaemonDriverCompletionRecord *record,
    SparkStatus effective_status,
    SparkRingRuntimeFinalEvent *event)
{
    const SparkModelDriverCompletion *completion;

    memset(event,0,sizeof(*event));
    completion = &record->completion;
    event->magic = SPARK_RING_RUNTIME_FINAL_EVENT_MAGIC;
    event->descriptor_bytes = SPARK_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES;
    event->status = (uint32_t)effective_status;
    event->program_id = completion->program_id;
    event->driver_dispatch_slot = completion->driver_dispatch_slot;
    event->accepted_token_count = completion->accepted_token_count;
    event->request_id = completion->request_id;
    event->sequence_id = completion->sequence_id;
    event->sequence_position = completion->sequence_position;
    event->service_time_ns = completion->service_time_ns;
    event->control_generation = transaction->identity.control_generation;
    event->transaction_id = transaction->identity.transaction_id;
    event->dispatch_generation = transaction->identity.dispatch_generation;
    event->request_generation = inflight_completion->request_generation;
    event->step_generation = transaction->identity.step_generation;
    event->step_chunk_index = transaction->identity.step_chunk_index;
    event->step_chunk_count = transaction->identity.step_chunk_count;
    event->transaction_phase = transaction->identity.phase;
    if (effective_status != SPARK_STATUS_OK)
    {
        return;
    }
    event->completion_flags = completion->completion_flags;
    event->token_count = completion->token_count;
    memcpy(
        event->token_ids,
        completion->token_ids,
        event->token_count * sizeof(event->token_ids[0u]));
    event->draft_token_count = completion->draft_token_count;
    memcpy(
        event->draft_token_ids,
        completion->draft_token_ids,
        event->draft_token_count * sizeof(event->draft_token_ids[0u]));
    if (record->dspark_draft_valid != 0u)
    {
        event->extension_flags |=
            SPARK_RING_RUNTIME_FINAL_EVENT_FLAG_DSPARK_DRAFT;
        event->dspark_draft = record->dspark_draft;
    }
}

static SparkStatus SparkRingDaemonProcessDriverCompletion(
    SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonDriverCompletionRecord *record)
{
    SparkRingDaemonInflightCompletion inflight_completion;
    SparkRingDaemonInflightTransaction *transaction;
    SparkRingRuntimeFinalEvent event;
    SparkStatus effective_status;
    SparkStatus status;
    uint64_t transaction_id;
    uint32_t completion_index;
    uint32_t emit_final_event;
    uint32_t remaining_completion_count;
    uint32_t terminal_state;

    if (runtime == 0 || record == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkRingDaemonFindInflightCompletion(
        runtime,
        record->completion.request_id,
        record->completion.sequence_id,
        record->completion.sequence_position,
        &completion_index);
    if (status != SPARK_STATUS_OK)
    {
        return status == SPARK_STATUS_NOT_FOUND ?
            SPARK_STATUS_VALIDATION_FAILED : status;
    }
    inflight_completion = runtime->inflight_completions[completion_index];
    if (inflight_completion.transaction_index >=
        SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    transaction = &runtime->inflight_transactions[
        inflight_completion.transaction_index];
    if (transaction->active == 0u ||
        transaction->remaining_completion_count == 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    status = SparkRingDaemonValidateDriverCompletionRecord(record);
    effective_status = status == SPARK_STATUS_OK ?
        record->completion.status : SPARK_STATUS_VALIDATION_FAILED;
    emit_final_event = SparkRingDaemonCompletionNeedsFinalEvent(
        runtime,
        record,
        effective_status,
        transaction->terminal_state);
    if (emit_final_event != 0u &&
        runtime->final_event_queue_count >=
            SPARK_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY)
    {
        return SPARK_STATUS_BUSY;
    }
    if (transaction->terminal_state == 0u &&
        effective_status != SPARK_STATUS_OK)
    {
        transaction->terminal_state =
            SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FAILED;
        transaction->terminal_status = (uint32_t)effective_status;
    }
    if (emit_final_event != 0u)
    {
        SparkRingDaemonBuildFinalEvent(
            transaction,
            &inflight_completion,
            record,
            effective_status,
            &event);
    }
    transaction_id = transaction->identity.transaction_id;
    remaining_completion_count =
        transaction->remaining_completion_count - 1u;
    if (remaining_completion_count == 0u)
    {
        terminal_state = transaction->terminal_state != 0u ?
            transaction->terminal_state :
            SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED;
        status = SparkDistributedWorkTransitionTransaction(
            &runtime->transaction_ledger,
            &transaction->identity,
            transaction->packet_hash,
            terminal_state,
            terminal_state ==
                SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED ?
                SPARK_STATUS_OK :
                (SparkStatus)transaction->terminal_status);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_DUPLICATE)
        {
            return status;
        }
    }
    SparkRingDaemonRemoveInflightCompletion(runtime,completion_index);
    transaction->remaining_completion_count = remaining_completion_count;
    if (runtime->driver_inflight_count != 0u)
    {
        runtime->driver_inflight_count -= 1u;
    }
    runtime->driver_completion_count += 1u;
    if (runtime->driver_inflight_count == 0u)
    {
        runtime->driver_inflight_warned = 0u;
    }
    else
    {
        runtime->driver_inflight_open_ns = SparkNetMonotonicNs();
    }
    if (remaining_completion_count == 0u)
    {
        SparkRingDaemonReleaseInflightTransaction(
            runtime,
            inflight_completion.transaction_index);
    }
    if (emit_final_event != 0u)
    {
        status = SparkRingDaemonQueueFinalEvent(runtime,&event);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    if (runtime->trace_enabled != 0u)
    {
        fprintf(
            stderr,
            "ring_trace rank=%u completion request=%llu position=%llu flags=0x%x tokens=%u id0=%u status=%u transaction=%llu remaining=%u\n",
            runtime->rank_plan.rank_index,
            (unsigned long long)record->completion.request_id,
            (unsigned long long)record->completion.sequence_position,
            record->completion.completion_flags,
            record->completion.token_count,
            record->completion.token_count != 0u ?
                record->completion.token_ids[0u] : 0u,
            (uint32_t)effective_status,
            (unsigned long long)transaction_id,
            remaining_completion_count);
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkRingDaemonPumpDriverCompletions(
    SparkRingDaemonRuntime *runtime)
{
    SparkRingDaemonDriverCompletionRecord record;
    SparkStatus status;
    uint32_t overflow;
    uint32_t progress;

    if (runtime == 0)
    {
        return 0u;
    }
    progress = 0u;
    overflow = 0u;
    if (runtime->driver_completion_mutex_initialized != 0u &&
        pthread_mutex_lock(&runtime->driver_completion_mutex) == 0)
    {
        overflow = runtime->driver_completion_queue_overflow;
        runtime->driver_completion_queue_overflow = 0u;
        (void)pthread_mutex_unlock(&runtime->driver_completion_mutex);
    }
    if (overflow != 0u)
    {
        fprintf(stderr,"rank_driver_completion_queue_overflow rank=%u\n",
            runtime->rank_plan.rank_index);
        runtime->work_error_count += 1u;
        SparkRingDaemonRunning = 0;
        return 1u;
    }
    for (;;)
    {
        status = SparkRingDaemonPeekDriverCompletion(runtime,&record);
        if (status == SPARK_STATUS_NOT_FOUND)
        {
            return progress;
        }
        if (status != SPARK_STATUS_OK)
        {
            runtime->work_error_count += 1u;
            return progress;
        }
        status = SparkRingDaemonProcessDriverCompletion(runtime,&record);
        if (status == SPARK_STATUS_BUSY)
        {
            return progress;
        }
        if (status != SPARK_STATUS_OK)
        {
            runtime->work_error_count += 1u;
            fprintf(
                stderr,
                "rank_driver_completion_rejected rank=%u status=%u request=%llu sequence=%llu position=%llu\n",
                runtime->rank_plan.rank_index,
                (uint32_t)status,
                (unsigned long long)record.completion.request_id,
                (unsigned long long)record.completion.sequence_id,
                (unsigned long long)record.completion.sequence_position);
        }
        (void)SparkRingDaemonPopDriverCompletion(runtime);
        progress = 1u;
    }
}

static void SparkRingDaemonCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkRingDaemonDraftResult dspark_draft;
    SparkRingDaemonRuntime *runtime;
    SparkStatus status;

    runtime = (SparkRingDaemonRuntime *)completion_context;
    if (runtime == 0 || completion == 0)
    {
        return;
    }
    memset(&dspark_draft,0,sizeof(dspark_draft));
    status = SPARK_STATUS_NOT_FOUND;
    if (runtime->builder_completion_mutex_initialized != 0u &&
        pthread_mutex_lock(&runtime->builder_completion_mutex) == 0)
    {
        if (runtime->builder_library.builder_interface.take_dspark_draft != 0 &&
            runtime->builder_state != 0)
        {
            status =
                runtime->builder_library.builder_interface.take_dspark_draft(
                    runtime->builder_state,
                    &dspark_draft);
        }
        if (SparkRingDaemonQueueDriverCompletion(
                runtime,
                completion,
                status == SPARK_STATUS_OK ? &dspark_draft : 0) !=
            SPARK_STATUS_OK)
        {
            SparkRingDaemonSignalWake(runtime);
        }
        (void)pthread_mutex_unlock(&runtime->builder_completion_mutex);
        return;
    }
    if (SparkRingDaemonQueueDriverCompletion(
            runtime,
            completion,
            status == SPARK_STATUS_OK ? &dspark_draft : 0) !=
        SPARK_STATUS_OK)
    {
        SparkRingDaemonSignalWake(runtime);
    }
}

static SparkStatus SparkRingDaemonLoadTransport(
    SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonConfig *configuration)
{
    char rank_buffer[16];
    char port_buffer[16];
    if (snprintf(rank_buffer, sizeof(rank_buffer), "%u",
            runtime->rank_plan.rank_index) < 0 ||
        snprintf(port_buffer, sizeof(port_buffer), "%u",
            configuration->port_base) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    if (setenv("SPARKPIPE_RING_TRANSPORT_RANK",rank_buffer,1) != 0 ||
        setenv("SPARKPIPE_RING_TRANSPORT_PORT_BASE",port_buffer,1) != 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SparkHiddenTransportLoadInterfaceFromSharedObject(
        configuration->transport_shared_object_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        &runtime->transport_library);
}

static SparkStatus SparkRingDaemonOpenHiddenTransport(
    SparkRingDaemonRuntime *runtime)
{
    SparkStatus status;

    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
    {
        status = SparkHiddenTransportOpen(
            &runtime->rank_plan.input_endpoint,
            &runtime->transport_library.transport_interface,
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
            &runtime->input_transport_session);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
    {
        status = SparkHiddenTransportOpen(
            &runtime->rank_plan.output_endpoint,
            &runtime->transport_library.transport_interface,
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
            &runtime->output_transport_session);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonLoadDriver(
    SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonConfig *configuration,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkModelDriverCreateRequest create_request;
    SparkStatus status;

    status = SparkLoadModelDriver(
        configuration->driver_path,
        configuration->node_target,
        &runtime->loaded_driver,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetStatusError(
            error_buffer,
            error_buffer_bytes,
            "failed to dlopen/validate GLM52 model driver",
            status);
        return status;
    }
    runtime->program = SparkFindLoadedModelDriverProgram(
        &runtime->loaded_driver,
        configuration->program_name);
    if (runtime->program == 0)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "decode program not found in GLM52 model driver");
        return SPARK_STATUS_NOT_FOUND;
    }
    SparkModelDriverInitializeCreateRequest(&create_request);
    create_request.node_id = runtime->rank_plan.host_name;
    create_request.node_target = configuration->node_target;
    create_request.node_context = runtime->builder_result.node_context;
    create_request.completion_function = SparkRingDaemonCompletion;
    create_request.completion_context = runtime;
    status = runtime->loaded_driver.interface->create(
        &create_request,
        &runtime->driver_instance);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetStatusError(
            error_buffer,
            error_buffer_bytes,
            "GLM52 model driver create failed",
            status);
        return status;
    }
    if (runtime->driver_instance == 0)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "GLM52 model driver returned NULL instance");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonBuildNodeContext(
    SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonConfig *configuration)
{
    SparkRingNodeContextBuilderConfiguration builder_configuration;
    SparkStatus status;

    memset(&builder_configuration,0,sizeof(builder_configuration));
    builder_configuration.abi_version =
        SPARK_RING_NODE_CONTEXT_BUILDER_ABI_VERSION;
    builder_configuration.descriptor_bytes =
        SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES;
    builder_configuration.rank_index = runtime->rank_plan.rank_index;
    builder_configuration.max_active_sequence_count =
        configuration->max_active_sequence_count;
	builder_configuration.kv_pool_token_capacity = SPARK_GLM52_KV_POOL_TOKENS;
	builder_configuration.maximum_resident_sequence_count =
		SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_RESIDENT_SEQUENCE_COUNT;
    builder_configuration.port_base = configuration->port_base;
    builder_configuration.moe_pack_root = configuration->moe_pack_root;
    builder_configuration.stagepack_root = configuration->stagepack_root;
    builder_configuration.embedding_pack_path =
        configuration->embedding_pack_path;
    builder_configuration.node_target = configuration->node_target;
    builder_configuration.rank_plan = &runtime->rank_plan;
    status = SparkRingNodeContextBuilderLoadInterfaceFromSharedObject(
        configuration->node_context_builder_shared_object_path,
        SPARK_RING_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
        &runtime->builder_library);
    if (status != SPARK_STATUS_OK)
        return status;
    status = runtime->builder_library.builder_interface.initialize(
        &builder_configuration,
        &runtime->builder_state);
    if (status != SPARK_STATUS_OK || runtime->builder_state == 0)
        return status == SPARK_STATUS_OK ?
            SPARK_STATUS_INVALID_ARGUMENT : status;
    memset(&runtime->builder_result,0,sizeof(runtime->builder_result));
    status = runtime->builder_library.builder_interface.build(
        runtime->builder_state,
        &runtime->builder_result);
    if (status != SPARK_STATUS_OK)
        return status;
    return SparkRingNodeContextBuilderValidateResult(
        &runtime->builder_result,
        &runtime->rank_plan);
}

static SparkStatus SparkRingDaemonAttachBuilderDriver(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->builder_state == 0 ||
        runtime->builder_library.builder_interface.attach_driver == 0 ||
        runtime->loaded_driver.interface == 0 ||
        runtime->driver_instance == 0 ||
        runtime->program == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return runtime->builder_library.builder_interface.attach_driver(
        runtime->builder_state,
        runtime->loaded_driver.interface,
        runtime->driver_instance,
        runtime->program,
        runtime->output_transport_session);
}

static SparkStatus SparkRingDaemonInitializeRunner(
    SparkRingDaemonRuntime *runtime)
{
    SparkResidentDecodeStageProductionRunnerConfiguration configuration;

    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version =
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
    configuration.flags =
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION;
    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
        configuration.flags |=
            SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT;
    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
        configuration.flags |=
            SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
    configuration.driver_interface = runtime->loaded_driver.interface;
    configuration.driver_instance = runtime->driver_instance;
    configuration.program = runtime->program;
    configuration.execution_stream = 0;
    return SparkResidentDecodeStageProductionRunnerInitialize(
        &runtime->runner,
        &configuration);
}

static SparkStatus SparkRingDaemonOpenWorkControlPath(
    SparkRingDaemonRuntime *runtime)
{
    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u)
        return SPARK_STATUS_OK;
    runtime->work_listen_fd =
        SparkNetCreateListenSocket(
            "0.0.0.0",
            runtime->rank_plan.listen_port);
    if (runtime->work_listen_fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    if (SparkNetSetNonblocking(runtime->work_listen_fd) < 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    return SPARK_STATUS_OK;
}

static void SparkRingDaemonResetWorkInputSocket(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0)
    {
        return;
    }
    if (runtime->work_input_socket_fd >= 0)
    {
        close(runtime->work_input_socket_fd);
    }
    runtime->work_input_socket_fd = -1;
    runtime->work_read_offset = 0u;
    runtime->work_input_acknowledgement_pending = 0u;
    runtime->work_input_acknowledgement_write_offset = 0u;
    memset(
        &runtime->work_input_acknowledgement,
        0,
        sizeof(runtime->work_input_acknowledgement));
}

static void SparkRingDaemonResetWorkOutputSocket(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0)
    {
        return;
    }
    if (runtime->work_output_socket_fd >= 0)
    {
        close(runtime->work_output_socket_fd);
    }
    runtime->work_output_socket_fd = -1;
    runtime->work_output_connecting = 0u;
    runtime->work_output_write_offset = 0u;
    runtime->work_output_waiting_for_acknowledgement = 0u;
    runtime->work_output_acknowledgement_read_offset = 0u;
    runtime->work_output_packet_hash = 0u;
    memset(
        &runtime->work_output_acknowledgement,
        0,
        sizeof(runtime->work_output_acknowledgement));
}

static SparkStatus SparkRingDaemonFlushWorkInputAcknowledgement(
    SparkRingDaemonRuntime *runtime)
{
    SparkStatus status;

    if (runtime == 0 || runtime->work_input_socket_fd < 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (runtime->work_input_acknowledgement_pending == 0u)
    {
        return SPARK_STATUS_OK;
    }
    status = SparkRingDaemonWriteBuffered(
        runtime->work_input_socket_fd,
        &runtime->work_input_acknowledgement,
        SPARK_DISTRIBUTED_WORK_ACKNOWLEDGEMENT_BYTES,
        &runtime->work_input_acknowledgement_write_offset);
    if (status == SPARK_STATUS_BUSY)
    {
        return status;
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonResetWorkInputSocket(runtime);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->work_input_acknowledgement_pending = 0u;
    runtime->work_input_acknowledgement_write_offset = 0u;
    memset(
        &runtime->work_input_acknowledgement,
        0,
        sizeof(runtime->work_input_acknowledgement));
    return SPARK_STATUS_OK;
}

static uint32_t SparkRingDaemonAcceptWorkSocket(
    SparkRingDaemonRuntime *runtime)
{
    int32_t fd;

    if (runtime == 0 || runtime->work_listen_fd < 0 ||
        runtime->work_input_socket_fd >= 0)
    {
        return 0u;
    }
    fd = accept(runtime->work_listen_fd,0,0);
    if (fd < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            runtime->work_error_count += 1u;
        }
        return 0u;
    }
    if (SparkNetSetNonblocking(fd) < 0 ||
        SparkNetConfigureLowLatencyTcp(fd) < 0)
    {
        close(fd);
        runtime->work_error_count += 1u;
        return 0u;
    }
    runtime->work_input_socket_fd = fd;
    runtime->work_read_offset = 0u;
    runtime->work_input_acknowledgement_pending = 0u;
    runtime->work_input_acknowledgement_write_offset = 0u;
    memset(
        &runtime->work_input_acknowledgement,
        0,
        sizeof(runtime->work_input_acknowledgement));
    return 1u;
}

static SparkStatus SparkRingDaemonEnsureWorkOutputSocket(
    SparkRingDaemonRuntime *runtime)
{
    SparkStatus status;
    uint64_t now_ns;

    if (runtime == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((runtime->rank_plan.flags &
            SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (runtime->work_output_socket_fd >= 0)
    {
        status = SparkRingDaemonFinishConnect(
            &runtime->work_output_socket_fd,
            &runtime->work_output_connecting);
        if (status == SPARK_STATUS_OK)
        {
            runtime->work_output_retry_mono_ns = 0u;
            return SPARK_STATUS_OK;
        }
        if (runtime->work_output_socket_fd >= 0)
        {
            return status;
        }
        runtime->work_output_write_offset = 0u;
        runtime->work_output_waiting_for_acknowledgement = 0u;
        runtime->work_output_acknowledgement_read_offset = 0u;
        runtime->work_output_packet_hash = 0u;
        runtime->work_output_retry_mono_ns =
            SparkNetMonotonicNs() + SPARK_RING_DAEMON_CONNECT_RETRY_NS;
        return status;
    }
    now_ns = SparkNetMonotonicNs();
    if (runtime->work_output_retry_mono_ns != 0u &&
        now_ns < runtime->work_output_retry_mono_ns)
    {
        return SPARK_STATUS_BUSY;
    }
    runtime->work_output_socket_fd = SparkRingDaemonStartConnect(
        runtime->rank_plan.next_host_name,
        runtime->rank_plan.next_port,
        &runtime->work_output_connecting);
    if (runtime->work_output_socket_fd < 0)
    {
        runtime->work_output_retry_mono_ns =
            now_ns + SPARK_RING_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->work_output_retry_mono_ns = 0u;
    return runtime->work_output_connecting == 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static SparkStatus SparkRingDaemonReadWorkOutputAcknowledgement(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    SparkDistributedWorkIdentity identity;
    SparkStatus status;
    uint8_t *acknowledgement_bytes;
    ssize_t got;
    uint32_t remaining;

    if (runtime == 0 || packet == 0 || runtime->work_output_socket_fd < 0 ||
        runtime->work_output_waiting_for_acknowledgement == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    acknowledgement_bytes =
        (uint8_t *)&runtime->work_output_acknowledgement;
    while (runtime->work_output_acknowledgement_read_offset <
        SPARK_DISTRIBUTED_WORK_ACKNOWLEDGEMENT_BYTES)
    {
        remaining = SPARK_DISTRIBUTED_WORK_ACKNOWLEDGEMENT_BYTES -
            runtime->work_output_acknowledgement_read_offset;
        got = read(
            runtime->work_output_socket_fd,
            acknowledgement_bytes +
                runtime->work_output_acknowledgement_read_offset,
            remaining);
        if (got < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                return SPARK_STATUS_BUSY;
            }
            SparkRingDaemonResetWorkOutputSocket(runtime);
            runtime->work_output_retry_mono_ns =
                SparkNetMonotonicNs() + SPARK_RING_DAEMON_CONNECT_RETRY_NS;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        if (got == 0)
        {
            SparkRingDaemonResetWorkOutputSocket(runtime);
            runtime->work_output_retry_mono_ns =
                SparkNetMonotonicNs() + SPARK_RING_DAEMON_CONNECT_RETRY_NS;
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        runtime->work_output_acknowledgement_read_offset += (uint32_t)got;
    }
    status = SparkRingWorkControlGetTransactionIdentity(packet,&identity);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkDistributedWorkValidateAcknowledgement(
            &runtime->work_output_acknowledgement,
            &identity,
            runtime->work_output_packet_hash);
    }
    runtime->work_output_acknowledgement_read_offset = 0u;
    runtime->work_output_waiting_for_acknowledgement = 0u;
    memset(
        &runtime->work_output_acknowledgement,
        0,
        sizeof(runtime->work_output_acknowledgement));
    if (status == SPARK_STATUS_BUSY ||
        status == SPARK_STATUS_CAPACITY_EXCEEDED)
    {
        runtime->work_output_write_offset = 0u;
        runtime->work_output_packet_hash = 0u;
        runtime->work_output_retry_mono_ns =
            SparkNetMonotonicNs() + SPARK_RING_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_BUSY;
    }
    if (status == SPARK_STATUS_OK || status == SPARK_STATUS_DUPLICATE)
    {
        runtime->work_output_write_offset = 0u;
        runtime->work_output_packet_hash = 0u;
        runtime->work_forward_count += 1u;
        return SPARK_STATUS_OK;
    }
    runtime->work_output_write_offset = 0u;
    runtime->work_output_packet_hash = 0u;
    return status;
}

static SparkStatus SparkRingDaemonForwardWork(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    SparkStatus status;

    if (runtime == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((runtime->rank_plan.flags &
            SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (runtime->work_output_waiting_for_acknowledgement != 0u)
    {
        return SparkRingDaemonReadWorkOutputAcknowledgement(runtime,packet);
    }
    status = SparkRingDaemonEnsureWorkOutputSocket(runtime);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRingDaemonWriteBuffered(
        runtime->work_output_socket_fd,
        packet,
        packet->descriptor_bytes,
        &runtime->work_output_write_offset);
    if (status == SPARK_STATUS_BUSY)
    {
        return status;
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonResetWorkOutputSocket(runtime);
        runtime->work_output_retry_mono_ns =
            SparkNetMonotonicNs() + SPARK_RING_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    runtime->work_output_packet_hash = SparkDistributedWorkHashBytes(
        packet,
        packet->descriptor_bytes);
    if (runtime->work_output_packet_hash == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->work_output_waiting_for_acknowledgement = 1u;
    runtime->work_output_acknowledgement_read_offset = 0u;
    memset(
        &runtime->work_output_acknowledgement,
        0,
        sizeof(runtime->work_output_acknowledgement));
    return SPARK_STATUS_BUSY;
}

static SparkStatus SparkRingDaemonAwaitResidentSubmitResult(
    SparkRingDaemonRuntime *runtime)
{
    SparkCudaResidentIpcHeader header;
    const SparkCudaResidentIpcSubmitResult *submit_result;
    const SparkCudaResidentIpcCompletion *completion_message;
    SparkStatus status;

    for (;;)
    {
        status = SparkRingDaemonReadResidentMessage(
            runtime,30000u,&header);
        if (status != SPARK_STATUS_OK)
        {
            SparkRingDaemonTeardownCudaResident(runtime,"submit_message_read");
            return SPARK_STATUS_BUSY;
        }
        if (header.kind == SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT)
        {
            if (header.payload_bytes != sizeof(*submit_result))
            {
                SparkRingDaemonTeardownCudaResident(runtime,"submit_result_read");
                return SPARK_STATUS_BUSY;
            }
            submit_result = (const SparkCudaResidentIpcSubmitResult *)
                runtime->cuda_resident_payload;
            return (SparkStatus)submit_result->status;
        }
        if (header.kind == SPARK_CUDA_RESIDENT_IPC_KIND_COMPLETION)
        {
            if (header.payload_bytes != sizeof(*completion_message))
            {
                SparkRingDaemonTeardownCudaResident(runtime,"submit_completion_read");
                return SPARK_STATUS_BUSY;
            }
            completion_message = (const SparkCudaResidentIpcCompletion *)
                runtime->cuda_resident_payload;
            if (SparkRingDaemonQueueDriverCompletion(
                    runtime,
                    &completion_message->completion,
                    (completion_message->flags &
                        SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT) != 0u ?
                        &completion_message->dspark_draft : 0) !=
                SPARK_STATUS_OK)
            {
                SparkRingDaemonTeardownCudaResident(
                    runtime,
                    "submit_completion_queue");
                return SPARK_STATUS_BUSY;
            }
            continue;
        }
        SparkRingDaemonTeardownCudaResident(runtime,"submit_unknown_kind");
        return SPARK_STATUS_BUSY;
    }
}

static SparkStatus SparkRingDaemonSubmitWork(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    SparkCudaResidentIpcSubmitWork submit_message;
    SparkStatus status;
    uint64_t trace_begin_ns;

    if (runtime == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    trace_begin_ns = runtime->trace_enabled != 0u ? SparkNetMonotonicNs() : 0u;
    if (runtime->cuda_resident_socket_path != 0)
    {
        status = SparkRingDaemonEnsureCudaResident(runtime);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkCudaResidentIpcInitializeSubmitWork(
            &submit_message,packet,
            SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT);
        if (status != SPARK_STATUS_OK)
            return status;
        status = SparkRingDaemonWriteResidentMessage(
            runtime,
            SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK,
            &submit_message,
			submit_message.descriptor_bytes);
        if (status != SPARK_STATUS_OK)
        {
            SparkRingDaemonTeardownCudaResident(runtime,"submit_write");
            return SPARK_STATUS_BUSY;
        }
        runtime->cuda_resident_submit_count += 1u;
        status = SparkRingDaemonAwaitResidentSubmitResult(runtime);
        if (runtime->trace_enabled != 0u)
            fprintf(stderr,"ring_trace rank=%u work_submit request=%llu position=%llu status=%u dur_us=%llu\n",runtime->rank_plan.rank_index,(unsigned long long)packet->request_id,(unsigned long long)packet->sequence_position,(uint32_t)status,(unsigned long long)((SparkNetMonotonicNs() - trace_begin_ns) / 1000ull));
        return status;
    }
    if (runtime->builder_state == 0 ||
        runtime->builder_library.builder_interface.submit_work == 0)
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    return runtime->builder_library.builder_interface.submit_work(
        runtime->builder_state,
        packet,
        runtime->input_transport_session,
        runtime->output_transport_session,
        SparkRingDaemonCompletion,
        runtime);
}

static SparkStatus SparkRingDaemonProgressBuilder(
	SparkRingDaemonRuntime *runtime)
{
	if (runtime == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (runtime->cuda_resident_socket_path != 0)
		return SparkRingDaemonEnsureCudaResident(runtime);
	if (runtime->builder_state == 0 ||
		runtime->builder_library.builder_interface.progress == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	return runtime->builder_library.builder_interface.progress(
		runtime->builder_state);
}

static uint32_t SparkRingDaemonWorkPacketHash(
	const SparkRingWorkControlPacket *packet)
{
	const uint8_t *packet_bytes;
	uint32_t byte_index;
	uint32_t hash;

    if (packet == 0 || packet->descriptor_bytes <
            SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES ||
        packet->descriptor_bytes > SPARK_RING_WORK_CONTROL_PACKET_BYTES)
        return 0u;
    packet_bytes = (const uint8_t *)packet;
    hash = SPARK_RING_DAEMON_WORK_HASH_OFFSET;
    for (byte_index = 0u; byte_index < packet->descriptor_bytes; ++byte_index)
        hash = (hash ^ packet_bytes[byte_index]) *
            SPARK_RING_DAEMON_WORK_HASH_PRIME;
    return hash;
}

static uint32_t SparkRingDaemonWorkPacketMatches(
    const SparkRingWorkControlPacket *left,
    const SparkRingWorkControlPacket *right)
{
    if (left == 0 || right == 0 ||
        left->descriptor_bytes != right->descriptor_bytes ||
        left->descriptor_bytes <
            SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES ||
        left->descriptor_bytes > SPARK_RING_WORK_CONTROL_PACKET_BYTES)
        return 0u;
    return memcmp(left,right,left->descriptor_bytes) == 0;
}

static uint32_t SparkRingDaemonWorkPacketPhase(
    const SparkRingWorkControlPacket *packet)
{
    if ((packet->flags & SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
        return SPARK_RING_DAEMON_WORK_PHASE_RELEASE;
    if ((packet->flags & SPARK_RING_WORK_CONTROL_FLAG_PREFILL) != 0u)
        return SPARK_RING_DAEMON_WORK_PHASE_PREFILL;
    if ((packet->flags &
            (SPARK_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY |
             SPARK_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY)) != 0u)
        return SPARK_RING_DAEMON_WORK_PHASE_VERIFY;
    return SPARK_RING_DAEMON_WORK_PHASE_DECODE;
}

static uint32_t SparkRingDaemonDependencyHash(uint64_t sequence_id)
{
    return (uint32_t)(sequence_id ^ (sequence_id >> 32u)) &
        (SPARK_RING_DAEMON_DEPENDENCY_HASH_CAPACITY - 1u);
}

static void SparkRingDaemonIndexDependencyLanes(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    uint32_t hash_slot;
    uint32_t lane_index;
    memset(runtime->dependency_sequence_ids,0,
        sizeof(runtime->dependency_sequence_ids));
    for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
    {
        hash_slot = SparkRingDaemonDependencyHash(
            packet->lanes[lane_index].sequence_id);
        while (runtime->dependency_sequence_ids[hash_slot] != 0u)
            hash_slot = (hash_slot + 1u) &
                (SPARK_RING_DAEMON_DEPENDENCY_HASH_CAPACITY - 1u);
        runtime->dependency_sequence_ids[hash_slot] =
            packet->lanes[lane_index].sequence_id;
        runtime->dependency_sequence_positions[hash_slot] =
            packet->lanes[lane_index].sequence_position;
    }
}

static uint32_t SparkRingDaemonCandidateIsDependency(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *candidate,
    const SparkRingWorkControlPacket *packet)
{
    uint32_t candidate_phase;
    uint32_t hash_slot;
    uint32_t lane_index;
    uint32_t packet_phase;
    if (candidate == packet || candidate->control_generation !=
        packet->control_generation)
        return 0u;
    candidate_phase = SparkRingDaemonWorkPacketPhase(candidate);
    packet_phase = SparkRingDaemonWorkPacketPhase(packet);
    for (lane_index = 0u; lane_index < candidate->lane_count; ++lane_index)
    {
        hash_slot = SparkRingDaemonDependencyHash(
            candidate->lanes[lane_index].sequence_id);
        while (runtime->dependency_sequence_ids[hash_slot] != 0u &&
            runtime->dependency_sequence_ids[hash_slot] !=
                candidate->lanes[lane_index].sequence_id)
            hash_slot = (hash_slot + 1u) &
                (SPARK_RING_DAEMON_DEPENDENCY_HASH_CAPACITY - 1u);
        if (runtime->dependency_sequence_ids[hash_slot] == 0u)
            continue;
        if (packet_phase == SPARK_RING_DAEMON_WORK_PHASE_RELEASE &&
            candidate_phase != SPARK_RING_DAEMON_WORK_PHASE_RELEASE)
            return 1u;
        if (candidate->lanes[lane_index].sequence_position <
                runtime->dependency_sequence_positions[hash_slot] ||
            (candidate->lanes[lane_index].sequence_position ==
                 runtime->dependency_sequence_positions[hash_slot] &&
             candidate_phase < packet_phase))
            return 1u;
    }
    return 0u;
}

static void SparkRingDaemonInitializeWorkQueue(
    SparkRingDaemonRuntime *runtime)
{
    uint32_t slot_index;

    if (runtime == 0)
    {
        return;
    }
    runtime->work_queue_head = 0u;
    runtime->work_queue_count = 0u;
    for (slot_index = 0u;
         slot_index < SPARK_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
         ++slot_index)
    {
        runtime->work_queue_hash_heads[slot_index] =
            SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT;
    }
    for (slot_index = 0u;
         slot_index < SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
         ++slot_index)
    {
        runtime->work_queue_hash_next[slot_index] =
            SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT;
        runtime->work_queue_submitted[slot_index] = 0u;
        runtime->work_queue_forwarded[slot_index] = 0u;
        runtime->work_queue_state[slot_index] =
            SPARK_RING_DAEMON_WORK_STATE_READY;
    }
}

static uint32_t SparkRingDaemonFindQueuedWorkSlot(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    uint32_t hash_slot;
    uint32_t slot_index;

    if (runtime == 0 || packet == 0)
        return SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT;
    hash_slot = SparkRingDaemonWorkPacketHash(packet) %
        SPARK_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
    slot_index = runtime->work_queue_hash_heads[hash_slot];
    while (slot_index != SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (slot_index >= SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY)
            return SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT;
        if (SparkRingDaemonWorkPacketMatches(
                &runtime->work_queue[slot_index],
                packet) != 0u)
            return slot_index;
        slot_index = runtime->work_queue_hash_next[slot_index];
    }
    return SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT;
}

static uint32_t SparkRingDaemonHasQueuedDependency(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    uint32_t scan_index;
    uint32_t slot_index;

    if (runtime == 0 || packet == 0 || runtime->work_queue_count <= 1u)
        return 0u;
    SparkRingDaemonIndexDependencyLanes(runtime,packet);
    slot_index = runtime->work_queue_head;
    for (scan_index = 0u; scan_index < runtime->work_queue_count; ++scan_index)
    {
        if (SparkRingDaemonCandidateIsDependency(
                runtime,&runtime->work_queue[slot_index],packet) != 0u)
            return 1u;
        slot_index = (slot_index + 1u) %
            SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
    }
    return 0u;
}

static void SparkRingDaemonInsertQueuedWorkHash(
    SparkRingDaemonRuntime *runtime,
    uint32_t queue_slot)
{
    uint32_t hash_slot;

    if (runtime == 0 ||
        queue_slot >= SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY)
        return;
    hash_slot =
        SparkRingDaemonWorkPacketHash(&runtime->work_queue[queue_slot]) %
        SPARK_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
    runtime->work_queue_hash_next[queue_slot] =
        runtime->work_queue_hash_heads[hash_slot];
    runtime->work_queue_hash_heads[hash_slot] = queue_slot;
}

static void SparkRingDaemonRemoveQueuedWorkHash(
    SparkRingDaemonRuntime *runtime,
    uint32_t queue_slot)
{
    uint32_t hash_slot;
    uint32_t slot_index;
    uint32_t previous_slot;

    if (runtime == 0 ||
        queue_slot >= SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY)
        return;
    hash_slot =
        SparkRingDaemonWorkPacketHash(&runtime->work_queue[queue_slot]) %
        SPARK_RING_DAEMON_WORK_QUEUE_HASH_SLOTS;
    slot_index = runtime->work_queue_hash_heads[hash_slot];
    previous_slot = SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT;
    while (slot_index != SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (slot_index == queue_slot)
        {
            if (previous_slot == SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT)
                runtime->work_queue_hash_heads[hash_slot] =
                    runtime->work_queue_hash_next[slot_index];
            else
                runtime->work_queue_hash_next[previous_slot] =
                    runtime->work_queue_hash_next[slot_index];
            runtime->work_queue_hash_next[slot_index] =
                SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT;
            return;
        }
        if (slot_index >= SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY)
            return;
        previous_slot = slot_index;
        slot_index = runtime->work_queue_hash_next[slot_index];
    }
}

static SparkStatus SparkRingDaemonQueueWork(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    uint32_t tail;
    uint32_t existing_slot;

    if (runtime == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    runtime->work_receive_count += 1u;
    existing_slot = SparkRingDaemonFindQueuedWorkSlot(runtime,packet);
    if (existing_slot != SPARK_RING_DAEMON_NO_WORK_QUEUE_SLOT)
    {
        if (runtime->work_queue_submitted[existing_slot] == 0u)
        {
            runtime->work_queue[existing_slot] = *packet;
            runtime->work_queue_forwarded[existing_slot] = 0u;
            runtime->work_queue_state[existing_slot] =
                SPARK_RING_DAEMON_WORK_STATE_READY;
        }
        runtime->work_duplicate_count += 1u;
        return SPARK_STATUS_OK;
    }
    if (runtime->work_queue_count >= SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    tail = (runtime->work_queue_head + runtime->work_queue_count) %
        SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
    runtime->work_queue[tail] = *packet;
    runtime->work_queue_submitted[tail] = 0u;
    runtime->work_queue_forwarded[tail] = 0u;
    runtime->work_queue_state[tail] =
        SPARK_RING_DAEMON_WORK_STATE_READY;
    SparkRingDaemonInsertQueuedWorkHash(runtime,tail);
    runtime->work_queue_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkRingDaemonPopWork(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0 || runtime->work_queue_count == 0u)
        return;
    SparkRingDaemonRemoveQueuedWorkHash(runtime,runtime->work_queue_head);
    runtime->work_queue_submitted[runtime->work_queue_head] = 0u;
    runtime->work_queue_forwarded[runtime->work_queue_head] = 0u;
    runtime->work_queue_state[runtime->work_queue_head] =
        SPARK_RING_DAEMON_WORK_STATE_READY;
    runtime->work_queue_head = (runtime->work_queue_head + 1u) %
        SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
    runtime->work_queue_count -= 1u;
}

static void SparkRingDaemonDeferWork(
    SparkRingDaemonRuntime *runtime)
{
    SparkRingWorkControlPacket packet;
    uint32_t old_head;
    uint32_t tail;

    if (runtime == 0 || runtime->work_queue_count <= 1u)
        return;
    old_head = runtime->work_queue_head;
    if (runtime->work_queue_count < SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY)
    {
        tail = (runtime->work_queue_head + runtime->work_queue_count) %
            SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
        packet = runtime->work_queue[old_head];
        SparkRingDaemonRemoveQueuedWorkHash(runtime,old_head);
        runtime->work_queue[tail] = packet;
        runtime->work_queue_submitted[tail] =
            runtime->work_queue_submitted[old_head];
        runtime->work_queue_forwarded[tail] =
            runtime->work_queue_forwarded[old_head];
        runtime->work_queue_state[tail] =
            runtime->work_queue_state[old_head];
        runtime->work_queue_submitted[old_head] = 0u;
        runtime->work_queue_forwarded[old_head] = 0u;
        runtime->work_queue_state[old_head] =
            SPARK_RING_DAEMON_WORK_STATE_READY;
        SparkRingDaemonInsertQueuedWorkHash(runtime,tail);
    }
    runtime->work_queue_head = (runtime->work_queue_head + 1u) %
        SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
}

static void SparkRingDaemonWakeDeferredWork(
    SparkRingDaemonRuntime *runtime)
{
    uint32_t slot_index;
    uint32_t scan_index;

    if (runtime == 0 || runtime->work_queue_count == 0u)
        return;
    slot_index = runtime->work_queue_head;
    for (scan_index = 0u;
         scan_index < runtime->work_queue_count;
         ++scan_index)
    {
        if (runtime->work_queue_state[slot_index] !=
            SPARK_RING_DAEMON_WORK_STATE_READY)
        {
            runtime->work_queue_state[slot_index] =
                SPARK_RING_DAEMON_WORK_STATE_READY;
            runtime->work_wake_count += 1u;
        }
        slot_index = (slot_index + 1u) %
            SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY;
    }
}

static SparkStatus SparkRingDaemonTransitionPacket(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet,
    uint32_t next_state,
    SparkStatus terminal_status)
{
    SparkDistributedWorkIdentity identity;
    SparkStatus status;
    uint64_t packet_hash;

    if (runtime == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkRingWorkControlGetTransactionIdentity(packet,&identity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    packet_hash = SparkRingWorkControlPacketFingerprint(packet);
    if (packet_hash == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SparkDistributedWorkTransitionTransaction(
        &runtime->transaction_ledger,
        &identity,
        packet_hash,
        next_state,
        terminal_status);
}

static void SparkRingDaemonFailHeadWork(
    SparkRingDaemonRuntime *runtime,
    SparkStatus status)
{
    SparkRingWorkControlPacket *packet;

    if (runtime == 0 || runtime->work_queue_count == 0u)
    {
        return;
    }
    packet = &runtime->work_queue[runtime->work_queue_head];
    (void)SparkRingDaemonTransitionPacket(
        runtime,
        packet,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FAILED,
        status);
    runtime->work_error_count += 1u;
    fprintf(
        stderr,
        "rank_work_failed status=%u transaction=%llu request=%llu sequence=%llu position=%llu queued=%u\n",
        (uint32_t)status,
        (unsigned long long)packet->step_generation,
        (unsigned long long)packet->request_id,
        (unsigned long long)packet->sequence_id,
        (unsigned long long)packet->sequence_position,
        runtime->work_queue_count);
    SparkRingDaemonPopWork(runtime);
}

static uint32_t SparkRingDaemonPumpQueuedWork(
    SparkRingDaemonRuntime *runtime)
{
    SparkRingWorkControlPacket *packet;
    SparkStatus status;
    uint32_t attempts;
    uint32_t forward_done;
    uint32_t queue_slot;
    uint32_t transaction_index;

    if (runtime == 0 || runtime->work_queue_count == 0u)
    {
        return 0u;
    }
    attempts = runtime->work_queue_count;
    while (attempts != 0u)
    {
        attempts -= 1u;
        queue_slot = runtime->work_queue_head;
        packet = &runtime->work_queue[queue_slot];
        if (runtime->work_queue_state[queue_slot] ==
            SPARK_RING_DAEMON_WORK_STATE_WAITING_SUBMIT)
        {
            SparkRingDaemonDeferWork(runtime);
            continue;
        }
        if (runtime->work_queue_state[queue_slot] !=
                SPARK_RING_DAEMON_WORK_STATE_WAITING_FORWARD &&
            SparkRingDaemonHasQueuedDependency(runtime,packet) != 0u)
        {
            SparkRingDaemonDeferWork(runtime);
            continue;
        }
        forward_done =
            ((runtime->rank_plan.flags &
                SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u ||
             runtime->work_queue_forwarded[queue_slot] != 0u) ? 1u : 0u;
        if (forward_done == 0u)
        {
            status = SparkRingDaemonTransitionPacket(
                runtime,
                packet,
                SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
                SPARK_STATUS_PENDING);
            if (status != SPARK_STATUS_OK && status != SPARK_STATUS_DUPLICATE)
            {
                SparkRingDaemonFailHeadWork(runtime,status);
                return 1u;
            }
            status = SparkRingDaemonForwardWork(runtime,packet);
            if (status == SPARK_STATUS_OK)
            {
                runtime->work_queue_forwarded[queue_slot] = 1u;
                runtime->work_queue_state[queue_slot] =
                    SPARK_RING_DAEMON_WORK_STATE_READY;
                forward_done = 1u;
            }
            else if (status == SPARK_STATUS_BUSY ||
                     status == SPARK_STATUS_ROUTE_NOT_FOUND)
            {
                runtime->work_queue_state[queue_slot] =
                    SPARK_RING_DAEMON_WORK_STATE_WAITING_FORWARD;
                runtime->work_deferred_count += 1u;
                return 0u;
            }
            else
            {
                SparkRingDaemonFailHeadWork(runtime,status);
                return 1u;
            }
        }
        status = SparkRingDaemonTransitionPacket(
            runtime,
            packet,
            SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
            SPARK_STATUS_PENDING);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_DUPLICATE)
        {
            SparkRingDaemonFailHeadWork(runtime,status);
            return 1u;
        }
        if (runtime->work_queue_submitted[queue_slot] == 0u)
        {
            transaction_index = SPARK_DISTRIBUTED_WORK_INVALID_INDEX;
            status = SparkRingDaemonRegisterInflightTransaction(
                runtime,
                packet,
                &transaction_index);
            if (status == SPARK_STATUS_BUSY ||
                status == SPARK_STATUS_CAPACITY_EXCEEDED)
            {
                runtime->work_queue_state[queue_slot] =
                    SPARK_RING_DAEMON_WORK_STATE_WAITING_SUBMIT;
                runtime->work_deferred_count += 1u;
                SparkRingDaemonDeferWork(runtime);
                continue;
            }
            if (status != SPARK_STATUS_OK)
            {
                SparkRingDaemonFailHeadWork(runtime,status);
                return 1u;
            }
            status = SparkRingDaemonSubmitWork(runtime,packet);
            if (status == SPARK_STATUS_OK)
            {
                runtime->work_submit_count += 1u;
                runtime->work_queue_submitted[queue_slot] = 1u;
                status = SparkRingDaemonTransitionPacket(
                    runtime,
                    packet,
                    SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_EXECUTING,
                    SPARK_STATUS_PENDING);
                if (status != SPARK_STATUS_OK &&
                    status != SPARK_STATUS_DUPLICATE)
                {
                    fprintf(
                        stderr,
                        "rank_work_transition_failed status=%u request=%llu sequence=%llu position=%llu\n",
                        (uint32_t)status,
                        (unsigned long long)packet->request_id,
                        (unsigned long long)packet->sequence_id,
                        (unsigned long long)packet->sequence_position);
                    runtime->work_error_count += 1u;
                    SparkRingDaemonRunning = 0;
                    return 1u;
                }
                if ((packet->flags &
                        SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
                {
                    status = SparkRingDaemonTransitionPacket(
                        runtime,
                        packet,
                        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_COMMITTED,
                        SPARK_STATUS_OK);
                    if (status != SPARK_STATUS_OK &&
                        status != SPARK_STATUS_DUPLICATE)
                    {
                        runtime->work_error_count += 1u;
                        SparkRingDaemonRunning = 0;
                        return 1u;
                    }
                }
            }
            else if (status == SPARK_STATUS_BUSY)
            {
                if (transaction_index != SPARK_DISTRIBUTED_WORK_INVALID_INDEX)
                {
                    (void)SparkRingDaemonCancelInflightTransaction(
                        runtime,
                        transaction_index);
                }
                runtime->work_queue_state[queue_slot] =
                    SPARK_RING_DAEMON_WORK_STATE_WAITING_SUBMIT;
                runtime->work_deferred_count += 1u;
                SparkRingDaemonDeferWork(runtime);
                continue;
            }
            else
            {
                if (transaction_index != SPARK_DISTRIBUTED_WORK_INVALID_INDEX)
                {
                    (void)SparkRingDaemonCancelInflightTransaction(
                        runtime,
                        transaction_index);
                }
                SparkRingDaemonFailHeadWork(runtime,status);
                return 1u;
            }
        }
        if (forward_done != 0u &&
            runtime->work_queue_submitted[queue_slot] != 0u)
        {
            SparkRingDaemonPopWork(runtime);
            return 1u;
        }
    }
    return 0u;
}

static SparkStatus SparkRingDaemonHandleWork(
    SparkRingDaemonRuntime *runtime,
    const SparkRingWorkControlPacket *packet)
{
    SparkDistributedWorkIdentity identity;
    SparkStatus status;
    SparkStatus terminal_status;
    uint64_t packet_hash;
    uint32_t existing_state;

    if (runtime == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    existing_state = SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FREE;
    terminal_status = SPARK_STATUS_OK;
    status = SparkRingWorkControlValidatePacket(
        packet,
        runtime->rank_plan.execution_row_capacity,
        SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRingWorkControlGetTransactionIdentity(packet,&identity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    packet_hash = SparkRingWorkControlPacketFingerprint(packet);
    if (packet_hash == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (runtime->work_queue_count >= SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkDistributedWorkObserveTransaction(
        &runtime->transaction_ledger,
        &identity,
        packet_hash,
        &existing_state,
        &terminal_status);
    if (status == SPARK_STATUS_DUPLICATE)
    {
        runtime->work_duplicate_count += 1u;
        if (existing_state ==
                SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FAILED ||
            existing_state ==
                SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_CANCELLED)
        {
            return terminal_status;
        }
        return SPARK_STATUS_DUPLICATE;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkDistributedWorkTransitionTransaction(
        &runtime->transaction_ledger,
        &identity,
        packet_hash,
        SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_ACCEPTED,
        SPARK_STATUS_PENDING);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkRingDaemonQueueWork(runtime,packet);
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkDistributedWorkTransitionTransaction(
            &runtime->transaction_ledger,
            &identity,
            packet_hash,
            SPARK_DISTRIBUTED_WORK_TRANSACTION_STATE_FAILED,
            status);
    }
    return status;
}

static uint32_t SparkRingDaemonPumpWorkControl(
    SparkRingDaemonRuntime *runtime)
{
    SparkDistributedWorkIdentity identity;
    SparkRingWorkControlPacket *packet;
    SparkStatus acknowledgement_status;
    SparkStatus status;
    ssize_t got;
    uint64_t packet_hash;
    uint32_t expected_bytes;
    uint32_t remaining;
    uint32_t progress;

    if (runtime == 0)
    {
        return 0u;
    }
    progress = SparkRingDaemonAcceptWorkSocket(runtime);
    if (runtime->work_listen_fd < 0 || runtime->work_input_socket_fd < 0)
    {
        return progress;
    }
    if (runtime->work_input_acknowledgement_pending != 0u)
    {
        status = SparkRingDaemonFlushWorkInputAcknowledgement(runtime);
        if (status == SPARK_STATUS_OK)
        {
            return 1u;
        }
        return progress;
    }
    for (;;)
    {
        if (SparkRingDaemonWorkInputCanRead(runtime) == 0u)
        {
            return progress;
        }
        packet = (SparkRingWorkControlPacket *)runtime->work_read_buffer;
        expected_bytes = SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES;
        if (runtime->work_read_offset >=
            (uint32_t)offsetof(SparkRingWorkControlPacket,flags))
        {
            expected_bytes = packet->descriptor_bytes;
            if (expected_bytes < SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES ||
                expected_bytes > SPARK_RING_WORK_CONTROL_PACKET_BYTES)
            {
                runtime->work_error_count += 1u;
                SparkRingDaemonResetWorkInputSocket(runtime);
                return 1u;
            }
        }
        remaining = expected_bytes - runtime->work_read_offset;
        got = read(
            runtime->work_input_socket_fd,
            runtime->work_read_buffer + runtime->work_read_offset,
            remaining);
        if (got < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                return progress;
            }
            runtime->work_error_count += 1u;
            SparkRingDaemonResetWorkInputSocket(runtime);
            return 1u;
        }
        if (got == 0)
        {
            SparkRingDaemonResetWorkInputSocket(runtime);
            return 1u;
        }
        progress = 1u;
        runtime->work_read_offset += (uint32_t)got;
        if (runtime->work_read_offset ==
                SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES &&
            packet->descriptor_bytes >
                SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES)
        {
            continue;
        }
        if (runtime->work_read_offset < expected_bytes)
        {
            return progress;
        }
        packet_hash = SparkDistributedWorkHashBytes(
            packet,
            packet->descriptor_bytes);
        status = SparkRingWorkControlGetTransactionIdentity(packet,&identity);
        if (status != SPARK_STATUS_OK || packet_hash == 0u)
        {
            runtime->work_error_count += 1u;
            SparkRingDaemonResetWorkInputSocket(runtime);
            return 1u;
        }
        acknowledgement_status = SparkRingDaemonHandleWork(runtime,packet);
        if (acknowledgement_status != SPARK_STATUS_OK &&
            acknowledgement_status != SPARK_STATUS_DUPLICATE &&
            acknowledgement_status != SPARK_STATUS_BUSY &&
            acknowledgement_status != SPARK_STATUS_CAPACITY_EXCEEDED)
        {
            runtime->work_error_count += 1u;
            fprintf(
                stderr,
                "rank_work_accept_failed status=%u transaction=%llu request=%llu queued=%u\n",
                (uint32_t)acknowledgement_status,
                (unsigned long long)identity.step_generation,
                (unsigned long long)packet->request_id,
                runtime->work_queue_count);
        }
        SparkDistributedWorkInitializeAcknowledgement(
            &runtime->work_input_acknowledgement,
            &identity,
            packet_hash,
            acknowledgement_status);
        runtime->work_input_acknowledgement_pending = 1u;
        runtime->work_input_acknowledgement_write_offset = 0u;
        runtime->work_read_offset = 0u;
        status = SparkRingDaemonFlushWorkInputAcknowledgement(runtime);
        if (status != SPARK_STATUS_OK)
        {
            return 1u;
        }
    }
}

static SparkStatus SparkRingDaemonOpenFinalEventPath(
    SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonConfig *configuration)
{
    if (runtime->rank_plan.rank_index == 0u &&
        configuration->own_final_event != 0u)
    {
        runtime->final_event_listen_fd =
            SparkNetCreateListenSocket(
                configuration->final_event_bind_address,
                runtime->final_event_route.listen_port);
        if (runtime->final_event_listen_fd < 0)
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        if (SparkNetSetNonblocking(
                runtime->final_event_listen_fd) < 0)
            return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u)
        (void)configuration;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingDaemonEnsureFinalEventSocket(
    SparkRingDaemonRuntime *runtime)
{
    SparkStatus status;
    uint64_t now_ns;

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((runtime->rank_plan.flags &
        SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u)
        return SPARK_STATUS_OK;
    if (runtime->final_event_socket_fd >= 0)
    {
        status = SparkRingDaemonFinishConnect(
            &runtime->final_event_socket_fd,
            &runtime->final_event_socket_connecting);
        if (status == SPARK_STATUS_OK)
            runtime->final_event_retry_mono_ns = 0u;
        else if (runtime->final_event_socket_fd < 0)
            runtime->final_event_retry_mono_ns =
                SparkNetMonotonicNs() +
                SPARK_RING_DAEMON_CONNECT_RETRY_NS;
        return status;
    }
    now_ns = SparkNetMonotonicNs();
    if (runtime->final_event_retry_mono_ns != 0u &&
        now_ns < runtime->final_event_retry_mono_ns)
        return SPARK_STATUS_BUSY;
    runtime->final_event_socket_fd =
        SparkRingDaemonStartConnect(
            runtime->final_event_route.sink_host_name,
            runtime->final_event_route.connect_port,
            &runtime->final_event_socket_connecting);
    if (runtime->final_event_socket_fd < 0)
    {
        runtime->final_event_retry_mono_ns =
            now_ns + SPARK_RING_DAEMON_CONNECT_RETRY_NS;
        return SPARK_STATUS_BUSY;
    }
    runtime->final_event_retry_mono_ns = 0u;
    return runtime->final_event_socket_connecting == 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static void SparkRingDaemonResetFinalEventSocket(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime->final_event_socket_fd >= 0)
        close(runtime->final_event_socket_fd);
    runtime->final_event_socket_fd = -1;
    runtime->final_event_socket_connecting = 0u;
    runtime->final_event_write_offset = 0u;
    runtime->final_event_read_offset = 0u;
}

static uint32_t SparkRingDaemonPumpFinalEventSend(
    SparkRingDaemonRuntime *runtime)
{
    const SparkRingRuntimeFinalEvent *event;
    SparkStatus status;

    if (runtime == 0 ||
        (runtime->rank_plan.flags &
            SPARK_RING_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u ||
        runtime->final_event_queue_count == 0u)
        return 0u;
    status = SparkRingDaemonEnsureFinalEventSocket(runtime);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
        return 0u;
    event = &runtime->final_event_queue[runtime->final_event_queue_head];
    status = SparkRingDaemonWriteBuffered(
            runtime->final_event_socket_fd,
            event,
            sizeof(*event),
            &runtime->final_event_write_offset);
    if (status == SPARK_STATUS_BUSY)
        return 0u;
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonResetFinalEventSocket(runtime);
        runtime->final_event_send_error_count += 1u;
        return 0u;
    }
    SparkRingDaemonPopFinalEvent(runtime);
    runtime->final_event_send_count += 1u;
    return 1u;
}

static uint32_t SparkRingDaemonAcceptFinalEventSocket(
    SparkRingDaemonRuntime *runtime)
{
    int32_t fd;

    if (runtime->final_event_listen_fd < 0 ||
        runtime->final_event_socket_fd >= 0)
        return 0u;
    fd = accept(runtime->final_event_listen_fd,0,0);
    if (fd < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            runtime->final_event_receive_error_count += 1u;
        return 0u;
    }
    if (SparkNetSetNonblocking(fd) < 0)
    {
        close(fd);
        runtime->final_event_receive_error_count += 1u;
        return 0u;
    }
    SparkRingDaemonConfigureTcpSocket(fd);
    runtime->final_event_socket_fd = fd;
    return 1u;
}

static void SparkRingDaemonPublishFinalEvent(
    SparkRingDaemonRuntime *runtime,
    const SparkRingRuntimeFinalEvent *event)
{
    uint32_t token_index;

    if (event->magic != SPARK_RING_RUNTIME_FINAL_EVENT_MAGIC ||
        event->descriptor_bytes != (uint32_t)sizeof(*event))
    {
        runtime->final_event_receive_error_count += 1u;
        return;
    }
    runtime->final_event_receive_count += 1u;
    printf("glm52_ring_final_event=1 request=%llu sequence=%llu position=%llu status=%u accepted=%u token_count=%u service_ns=%llu",
        (unsigned long long)event->request_id,
        (unsigned long long)event->sequence_id,
        (unsigned long long)event->sequence_position,
        event->status,
        event->accepted_token_count,
        event->token_count,
        (unsigned long long)event->service_time_ns);
    if ((event->completion_flags & SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) != 0u)
    {
        for (token_index = 0u;
             token_index < event->token_count &&
                token_index < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
             ++token_index)
        {
            printf(" token%u=%u",token_index,event->token_ids[token_index]);
        }
    }
    printf("\n");
    fflush(stdout);
}

static uint32_t SparkRingDaemonPumpFinalEventReceive(
    SparkRingDaemonRuntime *runtime)
{
    SparkRingRuntimeFinalEvent *event;
    ssize_t got;
    uint32_t remaining;
    uint32_t progress;

    progress = 0u;
    if (runtime == 0 || runtime->final_event_socket_fd < 0)
        return 0u;
    for (;;)
    {
        remaining = ((uint32_t)sizeof(runtime->final_event_read_buffer) -
            runtime->final_event_read_offset);
        got = read(
            runtime->final_event_socket_fd,
            runtime->final_event_read_buffer + runtime->final_event_read_offset,
            remaining);
        if (got < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                return progress;
            SparkRingDaemonResetFinalEventSocket(runtime);
            runtime->final_event_receive_error_count += 1u;
            return progress;
        }
        if (got == 0)
        {
            SparkRingDaemonResetFinalEventSocket(runtime);
            return 1u;
        }
        progress = 1u;
        runtime->final_event_read_offset += (uint32_t)got;
        if (runtime->final_event_read_offset < sizeof(*event))
            return progress;
        event =
            (SparkRingRuntimeFinalEvent *)runtime->final_event_read_buffer;
        SparkRingDaemonPublishFinalEvent(runtime,event);
        runtime->final_event_read_offset = 0u;
    }
}

static uint32_t SparkRingDaemonPumpFinalEvents(
    SparkRingDaemonRuntime *runtime)
{
    uint32_t progress;

    progress = SparkRingDaemonAcceptFinalEventSocket(runtime);
    progress |= SparkRingDaemonPumpFinalEventReceive(runtime);
    progress |= SparkRingDaemonPumpFinalEventSend(runtime);
    return progress;
}

static SparkStatus SparkRingDaemonInitialize(
    SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonConfig *configuration,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;

    memset(runtime,0,sizeof(*runtime));
    runtime->cuda_resident_fd = -1;
    runtime->trace_enabled = getenv("SPARKPIPE_RING_TRACE") != 0 ? 1u : 0u;
    runtime->work_listen_fd = -1;
    runtime->work_input_socket_fd = -1;
    runtime->work_output_socket_fd = -1;
    runtime->final_event_listen_fd = -1;
    runtime->final_event_socket_fd = -1;
    runtime->wake_pipe_read_fd = -1;
    runtime->wake_pipe_write_fd = -1;
    SparkRingDaemonInitializeWorkQueue(runtime);
    status = SparkRingDaemonInitializeCompletionState(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to initialize rank completion state");
        return status;
    }
    status = SparkDistributedWorkInitializeTransactionLedger(
        &runtime->transaction_ledger,
        runtime->transaction_entries,
        runtime->transaction_hash_heads,
        runtime->transaction_hash_next,
        SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to initialize distributed work ledger");
        return status;
    }
    SparkLoadedModelDriverReset(&runtime->loaded_driver);
    status = SparkRingDaemonOpenWakePipe(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open rank daemon wake pipe");
        return status;
    }
    status = SparkRingRuntimeBuildRankPlan(
        configuration->rank_index,
        configuration->max_active_sequence_count,
        configuration->port_base,
        configuration->model_quantization_mode,
        &runtime->rank_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build RING rank plan");
        return status;
    }
    status = SparkRingRuntimeBuildFinalEventRoute(
        configuration->port_base,
        &runtime->final_event_route,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build RING final event route");
        return status;
    }
    if (configuration->cuda_resident_socket_path != 0)
    {
        runtime->cuda_resident_socket_path = configuration->cuda_resident_socket_path;
        status = SparkRingDaemonEnsureCudaResident(runtime);
        if (status != SPARK_STATUS_OK)
            fprintf(stderr,"rank_cuda_resident_not_ready_at_start rank=%u status=%u\n",runtime->rank_plan.rank_index,(uint32_t)status);
        status = SparkRingDaemonOpenWorkControlPath(runtime);
        if (status != SPARK_STATUS_OK)
        {
            SparkRingDaemonSetDefaultError(
                error_buffer,
                error_buffer_bytes,
                "failed to open production work-control path");
            return status;
        }
        status = SparkRingDaemonOpenFinalEventPath(runtime,configuration);
        if (status != SPARK_STATUS_OK)
        {
            SparkRingDaemonSetDefaultError(
                error_buffer,
                error_buffer_bytes,
                "failed to open final event route");
            return status;
        }
        return SPARK_STATUS_OK;
    }
    status = SparkRingRuntimeValidateStageMoePackFiles(
        &runtime->rank_plan,
        configuration->moe_pack_root,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to validate rank FP8 pack files");
        return status;
    }
    status = SparkRingDaemonBuildNodeContext(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to build resident node context");
        return status;
    }
    status = SparkRingDaemonLoadTransport(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to load production hidden transport");
        return status;
    }
    status = SparkRingDaemonOpenHiddenTransport(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open neighbor hidden transport session");
        return status;
    }
    status = SparkRingDaemonLoadDriver(
        runtime,
        configuration,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to load GLM52 model driver");
        return status;
    }
    status = SparkRingDaemonAttachBuilderDriver(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to attach node-context builder to driver");
        return status;
    }
    status = SparkRingDaemonInitializeRunner(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to initialize production stage runner");
        return status;
    }
    status = SparkRingDaemonOpenWorkControlPath(runtime);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open production work-control path");
        return status;
    }
    status = SparkRingDaemonOpenFinalEventPath(runtime,configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkRingDaemonSetDefaultError(
            error_buffer,
            error_buffer_bytes,
            "failed to open final event route");
        return status;
    }
    return SPARK_STATUS_OK;
}

static void SparkRingDaemonDestroy(
    SparkRingDaemonRuntime *runtime)
{
    if (runtime == 0)
        return;
    if (runtime->cuda_resident_fd >= 0)
        close(runtime->cuda_resident_fd);
    if (runtime->work_output_socket_fd >= 0)
        close(runtime->work_output_socket_fd);
    if (runtime->work_input_socket_fd >= 0)
        close(runtime->work_input_socket_fd);
    if (runtime->work_listen_fd >= 0)
        close(runtime->work_listen_fd);
    if (runtime->final_event_socket_fd >= 0)
        close(runtime->final_event_socket_fd);
    if (runtime->final_event_listen_fd >= 0)
        close(runtime->final_event_listen_fd);
    if (runtime->wake_pipe_read_fd >= 0)
        close(runtime->wake_pipe_read_fd);
    if (runtime->wake_pipe_write_fd >= 0)
        close(runtime->wake_pipe_write_fd);
    if (runtime->loaded_driver.interface != 0 &&
        runtime->loaded_driver.interface->destroy != 0 &&
        runtime->driver_instance != 0)
        runtime->loaded_driver.interface->destroy(runtime->driver_instance);
    SparkUnloadModelDriver(&runtime->loaded_driver);
    if (runtime->builder_library.builder_interface.destroy_result != 0 &&
        runtime->builder_state != 0)
        runtime->builder_library.builder_interface.destroy_result(
            runtime->builder_state,
            &runtime->builder_result);
    if (runtime->builder_library.builder_interface.destroy != 0 &&
        runtime->builder_state != 0)
        runtime->builder_library.builder_interface.destroy(
            runtime->builder_state);
    SparkRingNodeContextBuilderUnloadInterface(&runtime->builder_library);
    SparkHiddenTransportClose(runtime->input_transport_session);
    SparkHiddenTransportClose(runtime->output_transport_session);
    SparkHiddenTransportUnloadInterface(&runtime->transport_library);
    if (runtime->driver_completion_mutex_initialized != 0u)
    {
        (void)pthread_mutex_destroy(&runtime->driver_completion_mutex);
        runtime->driver_completion_mutex_initialized = 0u;
    }
    if (runtime->builder_completion_mutex_initialized != 0u)
    {
        (void)pthread_mutex_destroy(&runtime->builder_completion_mutex);
        runtime->builder_completion_mutex_initialized = 0u;
    }
}

static void SparkRingDaemonPrintReady(
    const SparkRingDaemonRuntime *runtime,
    const SparkRingDaemonConfig *configuration)
{
    printf("glm52_ring_rank_daemon=1\n");
    printf("rank=%u\n",runtime->rank_plan.rank_index);
    printf("host=%s\n",runtime->rank_plan.host_name);
    printf("stage=%u:%u\n",
        runtime->rank_plan.first_layer_index,
        runtime->rank_plan.layer_count);
    printf("cuda_resident_socket=%s\n",
        configuration->cuda_resident_socket_path != 0 ?
            configuration->cuda_resident_socket_path : "");
    printf("model_quantization=%s\n",
        SparkRingRuntimeQuantizationModeName(
            configuration->model_quantization_mode));
    printf("moe_pack_root=%s\n",
        configuration->moe_pack_root != 0 ? configuration->moe_pack_root : "");
    printf("stagepack_root=%s\n",
        configuration->stagepack_root != 0 ? configuration->stagepack_root : "");
    printf("transport_so=%s\n",
        configuration->transport_shared_object_path != 0 ?
            configuration->transport_shared_object_path : "");
    printf("driver_so=%s\n",
        configuration->driver_path != 0 ? configuration->driver_path : "");
    printf("node_context_builder_so=%s\n",
        configuration->node_context_builder_shared_object_path != 0 ?
            configuration->node_context_builder_shared_object_path : "");
    printf("embedding_pack=%s\n",
        configuration->embedding_pack_path != 0 ?
            configuration->embedding_pack_path : "");
    printf("input_transport=%u\n",
        runtime->input_transport_session != 0 ? 1u : 0u);
    printf("output_transport=%u\n",
        runtime->output_transport_session != 0 ? 1u : 0u);
    printf("transport_busy_poll=%u\n",configuration->transport_busy_poll);
    printf("work_listen=%d\n",runtime->work_listen_fd);
    printf("work_input_socket=%d\n",runtime->work_input_socket_fd);
    printf("work_output_socket=%d\n",runtime->work_output_socket_fd);
    printf("work_received=%llu\n",
        (unsigned long long)runtime->work_receive_count);
    printf("work_forwarded=%llu\n",
        (unsigned long long)runtime->work_forward_count);
    printf("work_submitted=%llu\n",
        (unsigned long long)runtime->work_submit_count);
    printf("work_duplicates=%llu\n",
        (unsigned long long)runtime->work_duplicate_count);
    printf("work_deferred=%llu\n",
        (unsigned long long)runtime->work_deferred_count);
    printf("work_woken=%llu\n",
        (unsigned long long)runtime->work_wake_count);
    printf("work_errors=%llu\n",
        (unsigned long long)runtime->work_error_count);
    printf("wake_signals=%llu\n",
        (unsigned long long)__atomic_load_n(
            &runtime->wake_signal_count,
            __ATOMIC_RELAXED));
    printf("driver_inflight=%u\n",runtime->driver_inflight_count);
    printf("driver_completions=%llu\n",
        (unsigned long long)runtime->driver_completion_count);
    printf("cuda_resident_submits=%llu\n",
        (unsigned long long)runtime->cuda_resident_submit_count);
    printf("cuda_resident_completions=%llu\n",
        (unsigned long long)runtime->cuda_resident_completion_count);
    printf("cuda_resident_errors=%llu\n",
        (unsigned long long)runtime->cuda_resident_error_count);
    printf("wake_dropped=%llu\n",
        (unsigned long long)__atomic_load_n(
            &runtime->wake_drop_count,
            __ATOMIC_RELAXED));
    printf("timer_wakes=%llu\n",
        (unsigned long long)runtime->timer_wake_count);
    printf("final_event_listen=%d\n",runtime->final_event_listen_fd);
    printf("final_event_socket=%d\n",runtime->final_event_socket_fd);
    printf("final_event_sent=%llu\n",
        (unsigned long long)runtime->final_event_send_count);
    printf("final_event_received=%llu\n",
        (unsigned long long)runtime->final_event_receive_count);
    printf("final_event_queued=%u\n",runtime->final_event_queue_count);
    fflush(stdout);
}

int main(int argc,char **argv)
{
    SparkRingDaemonConfig configuration;
    static SparkRingDaemonRuntime runtime;
    char error_buffer[512];
    SparkStatus status;
    uint32_t event_mask;
    uint32_t progress;
    uint64_t timeout_ns;
    uint64_t next_timer_ns;
    uint64_t now_ns;
	SparkStatus builder_status;

    SparkRingDaemonInitializeConfig(&configuration);
    if (SparkRingDaemonParseArguments(&configuration,argc,argv) < 0)
    {
        fprintf(stderr,
            "usage: %s --rank n [--cuda-resident-socket path | --model-quantization fp8|nvfp4 --moe-pack-root dir --stagepack-root dir --transport-so path --driver-so path --node-context-builder-so path --embedding-pack path] [--program name] [--node-target target] [--max-active n] [--port-base n] [--final-event-bind ip] [--final-event-return-host host] [--transport-busy-poll]\n",
            argv[0]);
        return 2;
    }
    signal(SIGINT,SparkRingDaemonSignal);
    signal(SIGTERM,SparkRingDaemonSignal);
    signal(SIGPIPE,SIG_IGN);
    error_buffer[0] = '\0';
    status = SparkRingDaemonInitialize(
        &runtime,
        &configuration,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"rank daemon init failed status=%u error=%s\n",
            (uint32_t)status,
            error_buffer);
        SparkRingDaemonDestroy(&runtime);
        return 3;
    }
    if (configuration.transport_busy_poll != 0u &&
        (runtime.transport_library.transport_interface.capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA) == 0u)
    {
        fprintf(stderr,
            "rank daemon transport busy poll requires Spark host RDMA\n");
        SparkRingDaemonDestroy(&runtime);
        return 3;
    }
    SparkRingDaemonPrintReady(&runtime,&configuration);
    while (SparkRingDaemonRunning != 0)
    {
        progress = 0u;
        if (SparkRingDaemonDrainWakePipe(&runtime) != 0u)
        {
            progress = 1u;
            SparkRingDaemonWakeDeferredWork(&runtime);
        }
        progress |= SparkRingDaemonPumpWorkControl(&runtime);
        progress |= SparkRingDaemonPumpCudaResident(&runtime);
        progress |= SparkRingDaemonPumpDriverCompletions(&runtime);
        SparkRingDaemonCheckInflightOverdue(&runtime);
        if (runtime.cuda_resident_fd < 0)
            (void)SparkResidentDecodeStageProductionRunnerProgress(
                &runtime.runner);
		builder_status = SparkRingDaemonProgressBuilder(&runtime);
		if (builder_status != SPARK_STATUS_OK &&
			builder_status != SPARK_STATUS_BUSY)
		{
			fprintf(stderr,
				"rank daemon builder progress failed status=%u\n",
				(uint32_t)builder_status);
			break;
		}
        progress |= SparkRingDaemonPumpQueuedWork(&runtime);
        progress |= SparkRingDaemonPumpDriverCompletions(&runtime);
        progress |= SparkRingDaemonPumpFinalEvents(&runtime);
        if (progress == 0u)
        {
            if (configuration.transport_busy_poll != 0u)
                continue;
            timeout_ns = builder_status == SPARK_STATUS_BUSY ?
				UINT64_C(1000000) : 0u;
            next_timer_ns = SparkRingDaemonNextTimerNs(&runtime);
            if (next_timer_ns != 0u)
            {
                now_ns = SparkNetMonotonicNs();
				next_timer_ns = next_timer_ns > now_ns ?
					next_timer_ns - now_ns : 1u;
				if (timeout_ns == 0u || next_timer_ns < timeout_ns)
					timeout_ns = next_timer_ns;
            }
            event_mask = SparkRingDaemonWaitForEvents(
                &runtime,
                timeout_ns);
            if ((event_mask &
                (SPARK_RING_DAEMON_POLL_KIND_INPUT_TRANSPORT |
                 SPARK_RING_DAEMON_POLL_KIND_OUTPUT_TRANSPORT |
                 SPARK_RING_DAEMON_POLL_KIND_WORK |
                 SPARK_RING_DAEMON_POLL_KIND_WORK_OUTPUT |
                 SPARK_RING_DAEMON_POLL_KIND_CUDA_RESIDENT |
                 SPARK_RING_DAEMON_POLL_KIND_FINAL_EVENT |
                 SPARK_RING_DAEMON_POLL_KIND_COMPLETION_WAKE |
                 SPARK_RING_DAEMON_POLL_KIND_TIMER)) != 0u)
            {
                if ((event_mask & SPARK_RING_DAEMON_POLL_KIND_TIMER) != 0u)
                    runtime.timer_wake_count += 1u;
                SparkRingDaemonWakeDeferredWork(&runtime);
            }
        }
    }
    SparkRingDaemonDestroy(&runtime);
    return 0;
}
