#define main SparkTestGlm52RingRankDaemonMain
#include "../node/rank_daemon.c"
#undef main

#include <assert.h>

static void SparkTestRankDaemonBuildPacket(
    SparkGlm52RingWorkControlPacket *packet,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint32_t flags)
{
    memset(packet,0,sizeof(*packet));
    packet->descriptor_bytes =
        SparkGlm52RingWorkControlCalculatePacketBytes(1u);
    packet->flags = flags;
    packet->control_generation = 7u;
    packet->request_id = sequence_id;
    packet->sequence_id = sequence_id;
    packet->sequence_position = sequence_position;
    packet->active_sequence_count = 1u;
    packet->lane_count = 1u;
    packet->lanes[0u].request_id = sequence_id;
    packet->lanes[0u].sequence_id = sequence_id;
    packet->lanes[0u].sequence_position = sequence_position;
}

static void SparkTestRankDaemonReadsSplitResidentMessage(void)
{
    static SparkGlm52RingDaemonRuntime runtime;
    SparkGlm52CudaResidentIpcHeader header;
    SparkGlm52CudaResidentIpcHeader received_header;
    SparkGlm52CudaResidentIpcSubmitResult payload;
    int32_t sockets[2];
    uint32_t header_split;
    uint32_t payload_split;

    memset(&runtime,0,sizeof(runtime));
    memset(&payload,0,sizeof(payload));
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    runtime.cuda_resident_fd = sockets[0];
    assert(SparkGlm52RingDaemonSetNonblocking(sockets[0]) == 0);
    payload.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
    payload.status = SPARK_STATUS_OK;
    SparkGlm52CudaResidentIpcInitializeHeader(
        &header,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT,
        4u,9u,sizeof(payload));
    header_split = (uint32_t)sizeof(header) / 2u;
    payload_split = (uint32_t)sizeof(payload) / 2u;
    assert(write(sockets[1],&header,header_split) == (ssize_t)header_split);
    assert(SparkGlm52RingDaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_BUSY);
    assert(runtime.cuda_resident_read_header_offset == header_split);
    assert(write(sockets[1],((const uint8_t *)&header) + header_split,
        sizeof(header) - header_split) ==
        (ssize_t)(sizeof(header) - header_split));
    assert(write(sockets[1],&payload,payload_split) ==
        (ssize_t)payload_split);
    assert(SparkGlm52RingDaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_BUSY);
    assert(runtime.cuda_resident_read_header_offset == sizeof(header));
    assert(runtime.cuda_resident_read_payload_offset == payload_split);
    assert(write(sockets[1],((const uint8_t *)&payload) + payload_split,
        sizeof(payload) - payload_split) ==
        (ssize_t)(sizeof(payload) - payload_split));
    assert(SparkGlm52RingDaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_OK);
    assert(received_header.kind ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT);
    assert(memcmp(runtime.cuda_resident_payload,&payload,sizeof(payload)) == 0);
    assert(runtime.cuda_resident_read_header_offset == 0u);
    assert(runtime.cuda_resident_read_payload_offset == 0u);
    close(sockets[0]);
    close(sockets[1]);
}

static void SparkTestRankDaemonRequestsSubmitResult(void)
{
    static SparkGlm52RingDaemonRuntime runtime;
    static SparkGlm52CudaResidentIpcSubmitWork submitted;
    SparkGlm52CudaResidentIpcSubmitResult result;
    SparkGlm52CudaResidentIpcHeader header;
    SparkGlm52CudaResidentIpcHeader submitted_header;
    SparkGlm52RingWorkControlPacket packet;
    int32_t sockets[2];

    memset(&runtime,0,sizeof(runtime));
    memset(&result,0,sizeof(result));
    memset(&submitted,0,sizeof(submitted));
    SparkTestRankDaemonBuildPacket(&packet,31u,2u,
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    runtime.cuda_resident_fd = sockets[0];
    runtime.cuda_resident_socket_path = "attached";
    runtime.rank_plan.rank_index = 1u;
    result.descriptor_bytes =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
    result.status = SPARK_STATUS_OK;
    SparkGlm52CudaResidentIpcInitializeHeader(
        &header,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT,
        runtime.rank_plan.rank_index,1u,sizeof(result));
    assert(write(sockets[1],&header,sizeof(header)) == (ssize_t)sizeof(header));
    assert(write(sockets[1],&result,sizeof(result)) == (ssize_t)sizeof(result));
    assert(SparkGlm52RingDaemonSubmitWork(&runtime,&packet) == SPARK_STATUS_OK);
    assert(SparkGlm52RingDaemonReadFull(
        sockets[1],&submitted_header,sizeof(submitted_header)) ==
        SPARK_STATUS_OK);
    assert(submitted_header.kind ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK);
    assert(submitted_header.payload_bytes ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES +
        packet.descriptor_bytes);
    assert(SparkGlm52RingDaemonReadFull(
        sockets[1],&submitted,submitted_header.payload_bytes) ==
        SPARK_STATUS_OK);
    assert(submitted.flags ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT);
    assert(SparkGlm52CudaResidentIpcValidateSubmitWork(
        &submitted,submitted_header.payload_bytes) == SPARK_STATUS_OK);
    close(sockets[0]);
    close(sockets[1]);
}

static void SparkTestRankDaemonPacketIdentityIncludesPhase(void)
{
    static SparkGlm52RingDaemonRuntime runtime;
    SparkGlm52RingWorkControlPacket decode;
    SparkGlm52RingWorkControlPacket prefill;
    memset(&runtime,0,sizeof(runtime));
    SparkGlm52RingDaemonInitializeWorkQueue(&runtime);
    SparkTestRankDaemonBuildPacket(&prefill,41u,8u,
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL);
    decode = prefill;
    decode.flags = SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT;
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&prefill) == SPARK_STATUS_OK);
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&decode) == SPARK_STATUS_OK);
    assert(runtime.work_queue_count == 2u);
    assert(runtime.work_duplicate_count == 0u);
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&decode) == SPARK_STATUS_OK);
    assert(runtime.work_queue_count == 2u);
    assert(runtime.work_duplicate_count == 1u);
}

static void SparkTestRankDaemonFindsLaneDependency(void)
{
    static SparkGlm52RingDaemonRuntime runtime;
    SparkGlm52RingWorkControlPacket current;
    SparkGlm52RingWorkControlPacket earlier;
    memset(&runtime,0,sizeof(runtime));
    SparkGlm52RingDaemonInitializeWorkQueue(&runtime);
    SparkTestRankDaemonBuildPacket(&earlier,51u,4u,
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL);
    earlier.lane_count = 2u;
    earlier.active_sequence_count = 2u;
    earlier.descriptor_bytes =
        SparkGlm52RingWorkControlCalculatePacketBytes(2u);
    earlier.lanes[1u].request_id = 52u;
    earlier.lanes[1u].sequence_id = 52u;
    earlier.lanes[1u].sequence_position = 4u;
    SparkTestRankDaemonBuildPacket(&current,52u,5u,
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL);
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&earlier) == SPARK_STATUS_OK);
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&current) == SPARK_STATUS_OK);
    assert(SparkGlm52RingDaemonHasQueuedDependency(&runtime,
        &runtime.work_queue[1u]) == 1u);
    SparkGlm52RingDaemonPopWork(&runtime);
    assert(SparkGlm52RingDaemonHasQueuedDependency(&runtime,
        &runtime.work_queue[runtime.work_queue_head]) == 0u);
}

static void SparkTestRankDaemonDecodeWaitsForSamePositionPrefill(void)
{
    static SparkGlm52RingDaemonRuntime runtime;
    SparkGlm52RingWorkControlPacket decode;
    SparkGlm52RingWorkControlPacket prefill;
    memset(&runtime,0,sizeof(runtime));
    SparkGlm52RingDaemonInitializeWorkQueue(&runtime);
    SparkTestRankDaemonBuildPacket(&prefill,61u,8u,
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL);
    SparkTestRankDaemonBuildPacket(&decode,61u,8u,
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT);
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&prefill) == SPARK_STATUS_OK);
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&decode) == SPARK_STATUS_OK);
    assert(SparkGlm52RingDaemonHasQueuedDependency(&runtime,
        &runtime.work_queue[1u]) == 1u);
}

static void SparkTestRankDaemonForwardWaitPreservesFifo(void)
{
    static SparkGlm52RingDaemonRuntime runtime;
    SparkGlm52RingWorkControlPacket packet;
    memset(&runtime,0,sizeof(runtime));
    SparkGlm52RingDaemonInitializeWorkQueue(&runtime);
    SparkTestRankDaemonBuildPacket(&packet,71u,0u,
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL);
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&packet) == SPARK_STATUS_OK);
    packet.sequence_position = 1u;
    packet.lanes[0u].sequence_position = 1u;
    assert(SparkGlm52RingDaemonQueueWork(&runtime,&packet) == SPARK_STATUS_OK);
    runtime.work_queue_state[runtime.work_queue_head] =
        SPARK_GLM52_RING_DAEMON_WORK_STATE_WAITING_FORWARD;
    assert(SparkGlm52RingDaemonPumpQueuedWork(&runtime) == 0u);
    assert(runtime.work_queue_head == 0u);
    assert(runtime.work_queue_count == 2u);
}

static void SparkTestRankDaemonBackpressuresFullWorkQueue(void)
{
    static SparkGlm52RingDaemonRuntime runtime;
    int32_t sockets[2];
    uint8_t byte;

    memset(&runtime,0,sizeof(runtime));
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    assert(SparkGlm52RingDaemonSetNonblocking(sockets[0]) == 0);
    runtime.work_listen_fd = 0;
    runtime.work_input_socket_fd = sockets[0];
    runtime.work_queue_count = SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
    byte = 0x5au;
    assert(write(sockets[1],&byte,sizeof(byte)) == (ssize_t)sizeof(byte));
    assert(SparkGlm52RingDaemonWorkInputCanRead(&runtime) == 0u);
    assert(SparkGlm52RingDaemonPumpWorkControl(&runtime) == 0u);
    assert(runtime.work_read_offset == 0u);
    assert(recv(sockets[0],&byte,sizeof(byte),MSG_PEEK) ==
        (ssize_t)sizeof(byte));
    runtime.work_queue_count -= 1u;
    assert(SparkGlm52RingDaemonWorkInputCanRead(&runtime) == 1u);
    assert(SparkGlm52RingDaemonPumpWorkControl(&runtime) == 1u);
    assert(runtime.work_read_offset == 1u);
    runtime.work_queue_count = SPARK_GLM52_RING_DAEMON_WORK_QUEUE_CAPACITY;
    assert(SparkGlm52RingDaemonWorkInputCanRead(&runtime) == 1u);
    close(sockets[0]);
    close(sockets[1]);
}

int main(void)
{
    SparkTestRankDaemonReadsSplitResidentMessage();
    SparkTestRankDaemonRequestsSubmitResult();
    SparkTestRankDaemonPacketIdentityIncludesPhase();
    SparkTestRankDaemonFindsLaneDependency();
    SparkTestRankDaemonDecodeWaitsForSamePositionPrefill();
    SparkTestRankDaemonForwardWaitPreservesFifo();
    SparkTestRankDaemonBackpressuresFullWorkQueue();
    return 0;
}
