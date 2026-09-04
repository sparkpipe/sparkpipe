from pathlib import Path

rp = Path("/Users/mac/lane-glm53/ring/transport/rdma.cu")
rs = rp.open().read()

# 1. Add FIXED_FLAG constant after RETURN_FLAG
old_const = "#define SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_RETURN_FLAG 0x80000000u"
assert rs.count(old_const) == 1
rs = rs.replace(old_const, old_const + "\n#define SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_FIXED_FLAG 0x40000000u")

# 2. Modify SendFixed to include the flag
rs = rs.replace(
    "    immediate = sequence;",
    "    immediate = sequence | SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_FIXED_FLAG;",
    1)

# 3. Add the fixed branch in ApplyDoorbellCompletion, right after the immediate is parsed
old_branch = """    status = SparkHiddenSparkHostRdmaPostDoorbellCredit(
        state,(uint32_t)receive_credit_index);
    if (status != SPARK_STATUS_OK)
        return status;
    if ((immediate &
            SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_RETURN_FLAG) != 0u)"""
assert rs.count(old_branch) == 1
new_branch = """    status = SparkHiddenSparkHostRdmaPostDoorbellCredit(
        state,(uint32_t)receive_credit_index);
    if (status != SPARK_STATUS_OK)
        return status;
    if ((immediate &
            SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_FIXED_FLAG) != 0u)
    {
        SparkHiddenTransportCompletion completion;
        uint32_t fixed_sequence = immediate &
            ~SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_FIXED_FLAG &
            ~SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_RETURN_FLAG;
        if (state->is_sender != 0u)
            return SPARK_STATUS_IO_ERROR;
        if (SparkHiddenTransportCompletionQueueIsFull(
                &state->completion_queue) != 0u)
            return SPARK_STATUS_BUSY;
        memset(&completion,0,sizeof(completion));
        completion.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
        completion.descriptor_bytes =
            SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
        completion.status = SPARK_STATUS_OK;
        completion.sequence_id = (uint64_t)fixed_sequence + 1u;
        completion.token_index = fixed_sequence;
        completion.transfer_bytes = 0u;
        completion.service_time_ns = 0u;
        SparkHiddenSparkHostRdmaSignalEvent(state);
        return SparkHiddenTransportCompletionQueuePush(
            &state->completion_queue,&completion);
    }
    if ((immediate &
            SPARK_HIDDEN_SPARK_HOST_RDMA_DOORBELL_RETURN_FLAG) != 0u)"""
rs = rs.replace(old_branch, new_branch)

rp.open("w").write(rs)
print("fixed CQ handler branch + FIXED_FLAG added")
