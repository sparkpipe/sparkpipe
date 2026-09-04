from pathlib import Path

p = Path("/Users/mac/lane-glm53/ring/transport/rdma.cu")
s = p.open().read()

api = """
#define SPARK_HIDDEN_SPARK_FIXED_DEPTH 8u
#define SPARK_HIDDEN_SPARK_FIXED_MASK (SPARK_HIDDEN_SPARK_FIXED_DEPTH - 1u)

static SparkStatus SparkHiddenSparkHostRdmaSendFixed(
    void *transport_state,
    const void *local_buffer,
    uint64_t bytes,
    uint32_t sequence)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkHiddenSparkHostRdmaInflightSend *send;
    struct ibv_send_wr work_requests[2];
    struct ibv_send_wr *bad_work_request;
    struct ibv_sge scatter_entries[1];
    uint32_t immediate;
    uint32_t lane_index;
    uint32_t slot;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || local_buffer == 0 || state->is_sender == 0u ||
        bytes == 0u || state->fixed_remote.address == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    slot = sequence & SPARK_HIDDEN_SPARK_FIXED_MASK;
    lane_index = slot & 1u;
    if (state->outstanding_send_wr_counts[lane_index] >=
            SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_SEND_WR_PER_LANE)
        return SPARK_STATUS_BUSY;
    send = SparkHiddenSparkHostRdmaReserveInflightSend(state);
    if (send == 0)
        return SPARK_STATUS_BUSY;
    memset(send,0,sizeof(*send));
    send->active = 1u;
    send->posted_lane_mask = 1u << lane_index;
    send->start_time_ns = SparkHiddenSparkHostRdmaMonotonicNs();
    send->packet_snapshot.sequence_id = sequence + 1u;
    send->packet_snapshot.token_index = sequence;
    send->packet_snapshot.active_sequence_count = 1u;
    send->packet_snapshot.hidden_bf16 = local_buffer;
    send->packet_snapshot.bytes_per_sequence = (uint32_t)bytes;
    send->packet_snapshot.hidden_dimension =
        (uint32_t)(bytes / 2u);
    scatter_entries[0].addr = (uintptr_t)local_buffer;
    scatter_entries[0].length = (uint32_t)bytes;
    scatter_entries[0].lkey = state->fixed_local_lkey;
    immediate = sequence;
    memset(work_requests,0,sizeof(work_requests));
    work_requests[0].wr_id = (uint64_t)(send - state->inflight_sends);
    work_requests[0].opcode = IBV_WR_RDMA_WRITE;
    work_requests[0].sg_list = scatter_entries;
    work_requests[0].num_sge = 1;
    work_requests[0].send_flags = IBV_SEND_SIGNALED;
    work_requests[0].wr.rdma.remote_addr =
        state->fixed_remote.address + (uint64_t)slot * bytes;
    work_requests[0].wr.rdma.rkey = state->fixed_remote.rkey;
    work_requests[1].wr_id = (uint64_t)(send - state->inflight_sends);
    work_requests[1].opcode = IBV_WR_SEND_WITH_IMM;
    work_requests[1].sg_list = scatter_entries;
    work_requests[1].num_sge = 0;
    work_requests[1].send_flags = IBV_SEND_SIGNALED;
    work_requests[1].imm_data = htonl(immediate);
    work_requests[0].next = &work_requests[1];
    work_requests[1].next = 0;
    if (ibv_post_send(state->lanes[lane_index].queue_pair,
            work_requests,&bad_work_request) != 0)
    {
        memset(send,0,sizeof(*send));
        return SPARK_STATUS_IO_ERROR;
    }
    state->outstanding_send_wr_counts[lane_index] += 2u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSetFixedRemote(
    void *transport_state,
    uint64_t remote_address,
    uint64_t remote_bytes,
    uint32_t remote_rkey)
{
    SparkHiddenSparkHostRdmaState *state;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || remote_address == 0u || remote_bytes == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state->fixed_remote.address = remote_address;
    state->fixed_remote.bytes = remote_bytes;
    state->fixed_remote.rkey = remote_rkey;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSetFixedLocal(
    void *transport_state,
    void *local_buffer,
    uint64_t local_bytes)
{
    SparkHiddenSparkHostRdmaState *state;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || local_buffer == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state->fixed_local = local_buffer;
    state->fixed_local_lkey = state->hidden_memory_region != 0 ?
        state->hidden_memory_region->lkey : 0u;
    return SPARK_STATUS_OK;
}
"""

# Insert before the extern C interface block
anchor = 'extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)'
assert s.count(anchor) == 1, "interface anchor"
s = s.replace(anchor, api + "\n" + anchor)

# Add fields to the state struct
old_fields = """    uint32_t lane_count;"""
assert s.count(old_fields) >= 1
s = s.replace(old_fields, """    uint32_t lane_count;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor fixed_remote;
    void *fixed_local;
    uint32_t fixed_local_lkey;""", 1)

p.open("w").write(s)
print("fixed-slot API installed")
