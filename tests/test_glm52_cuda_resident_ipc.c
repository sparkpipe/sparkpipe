#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"

enum
{
    SPARK_TEST_RESIDENT_IPC_HEADER_FRAGMENT_BYTES = 5u,
    SPARK_TEST_RESIDENT_IPC_PAYLOAD_FRAGMENT_BYTES = 7u,
    SPARK_TEST_RESIDENT_IPC_PAYLOAD_BYTES = 17u
};

static SparkGlm52CudaResidentIpcSubmitDecode *SparkTestBuildWideDecode(
    uint32_t lane_count,
    uint32_t blocks_per_lane,
    uint32_t *payload_bytes_out)
{
    SparkGlm52CudaResidentIpcSubmitDecode *message;
    uint32_t block_index_count;
    uint32_t payload_bytes;
    uint32_t lane_index;
    uint32_t block_index;
    block_index_count = lane_count * blocks_per_lane;
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        block_index_count,&payload_bytes) == SPARK_STATUS_OK);
    message = (SparkGlm52CudaResidentIpcSubmitDecode *)calloc(1u,payload_bytes);
    assert(message != 0);
    message->descriptor_bytes =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES;
    message->control_generation =
        SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
    message->dispatch_kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
    message->lane_count = lane_count;
    message->active_sequence_count = lane_count;
    message->execution_batch_bucket =
        SparkGlm52StagePlanSelectBatchBucketValue(lane_count);
    message->kv_block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
    message->kv_block_index_count = block_index_count;
    for (lane_index = 0u; lane_index < lane_count; ++lane_index)
    {
        SparkGlm52CudaResidentIpcDecodeLane *lane;
        lane = &message->lanes[lane_index];
        lane->request_id = 1000u + lane_index;
        lane->sequence_id = 2000u + lane_index;
        lane->sequence_position = 8192u + lane_index;
        lane->request_slot_index = lane_index;
        lane->context_token_count = 8192u;
        lane->input_token_id = 17u + lane_index;
        lane->kv_block_offset = lane_index * blocks_per_lane;
        lane->kv_block_count = blocks_per_lane;
        for (block_index = 0u; block_index < blocks_per_lane; ++block_index)
            message->kv_physical_block_indices[
                lane->kv_block_offset + block_index] =
                lane_index * 4096u + block_index;
    }
    *payload_bytes_out = payload_bytes;
    return message;
}

static void SparkTestWideDecodePayload(void)
{
    SparkGlm52CudaResidentIpcSubmitDecode *message;
    uint32_t payload_bytes;
    message = SparkTestBuildWideDecode(
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT,17u,&payload_bytes);
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) == SPARK_STATUS_OK);
    message->execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B512;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    message->execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
    message->control_generation = 0u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    message->control_generation =
        SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
    assert(message->lanes[1023u].kv_block_offset == 1023u * 17u);
    assert(message->kv_physical_block_indices[
        message->lanes[1023u].kv_block_offset + 16u] ==
        1023u * 4096u + 16u);

    message->lanes[511u].kv_block_offset += 1u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    message->lanes[511u].kv_block_offset -= 1u;

    message->dispatch_kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
    message->speculative_token_count = 6u;
    for (uint32_t lane_index = 0u; lane_index < message->lane_count;
         ++lane_index)
        message->lanes[lane_index].speculative_token_count = 6u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) == SPARK_STATUS_OK);
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT - 1u) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    free(message);
}

static void SparkTestDecodePayloadCapacity(void)
{
    uint32_t maximum_block_index_count;
    uint32_t payload_bytes;
    maximum_block_index_count =
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT *
        SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS;
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        maximum_block_index_count,&payload_bytes) == SPARK_STATUS_OK);
    assert(payload_bytes ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES);
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        maximum_block_index_count + 1u,&payload_bytes) ==
        SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void SparkTestInternalDirectoryDecodeHasNoBlockPayload(void)
{
    SparkGlm52CudaResidentIpcSubmitDecode *message;
    uint32_t payload_bytes;
    uint32_t lane_index;
    assert(SparkGlm52CudaResidentIpcDecodePayloadBytes(
        0u,&payload_bytes) == SPARK_STATUS_OK);
    assert(payload_bytes ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES);
    message = (SparkGlm52CudaResidentIpcSubmitDecode *)calloc(
        1u,payload_bytes);
    assert(message != 0);
    message->descriptor_bytes =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES;
    message->control_generation =
        SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
    message->dispatch_kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
    message->lane_count = SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT;
    message->active_sequence_count = message->lane_count;
    message->execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
    message->kv_block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
    message->resident_flags =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY;
    for (lane_index = 0u; lane_index < message->lane_count; ++lane_index)
    {
        message->lanes[lane_index].request_id = 1000u + lane_index;
        message->lanes[lane_index].sequence_id = 2000u + lane_index;
        message->lanes[lane_index].request_slot_index = lane_index;
        message->lanes[lane_index].context_token_count = 4096u;
        message->lanes[lane_index].input_token_id = 123u;
    }
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) == SPARK_STATUS_OK);
    message->lanes[0u].mtp_resolution_proposed_token_count = 3u;
    message->lanes[0u].mtp_resolution_accepted_token_count = 1u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) == SPARK_STATUS_OK);
    message->lanes[0u].mtp_resolution_accepted_token_count = 4u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    message->lanes[0u].mtp_resolution_accepted_token_count = 1u;
    message->lanes[0u].mtp_resolution_path_id =
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH1;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    message->lanes[0u].mtp_resolution_path_id =
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
    message->lanes[9u].kv_block_count = 1u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitDecode(
        message,payload_bytes,
        SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    free(message);
}

static void SparkTestWideSubmitWorkUsesVariablePayload(void)
{
    SparkGlm52CudaResidentIpcSubmitWork *message;
    SparkGlm52RingWorkControlPacket *packet;
    uint32_t lane_index;
    uint32_t packet_bytes;
    uint32_t submit_bytes;

    packet = (SparkGlm52RingWorkControlPacket *)calloc(1u,sizeof(*packet));
    assert(packet != 0);
    message = (SparkGlm52CudaResidentIpcSubmitWork *)calloc(
        1u,sizeof(*message));
    assert(message != 0);
    packet->magic = SPARK_GLM52_RING_WORK_CONTROL_PACKET_MAGIC;
    packet->abi_version = SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION;
    packet->control_generation =
        SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
    packet->flags =
        SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
    packet->lane_count = SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT;
    packet->active_sequence_count = packet->lane_count;
    packet->rows_per_lane = SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT + 1u;
    packet->execution_row_count = packet->lane_count * packet->rows_per_lane;
    packet->execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
    packet->new_token_count = packet->rows_per_lane;
    packet->speculative_token_count =
        SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
    packet_bytes = SparkGlm52RingWorkControlCalculatePacketBytes(
        packet->lane_count);
    assert(packet_bytes == SPARK_GLM52_RING_WORK_CONTROL_PACKET_BYTES);
    packet->descriptor_bytes = packet_bytes;
    for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
    {
        packet->lanes[lane_index].request_id = 1000u + lane_index;
        packet->lanes[lane_index].sequence_id = 2000u + lane_index;
        packet->lanes[lane_index].request_slot_index = lane_index;
        packet->lanes[lane_index].speculative_token_count =
            packet->speculative_token_count;
    }
    submit_bytes = SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(packet);
    assert(submit_bytes ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES + packet_bytes);
    assert(submit_bytes == SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES);
    assert(SparkGlm52CudaResidentIpcInitializeSubmitWork(
        message,packet,
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT) ==
        SPARK_STATUS_OK);
    assert(message->descriptor_bytes == submit_bytes);
    assert(message->flags ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT);
    assert(SparkGlm52CudaResidentIpcValidateSubmitWork(
        message,submit_bytes) == SPARK_STATUS_OK);
    message->flags = 0x80000000u;
    assert(SparkGlm52CudaResidentIpcValidateSubmitWork(
        message,submit_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
    message->flags =
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT;
    assert(SparkGlm52CudaResidentIpcValidateSubmitWork(
        message,submit_bytes - 1u) == SPARK_STATUS_ABI_MISMATCH);

    packet->descriptor_bytes =
        SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES - 1u;
    assert(SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(packet) == 0u);
    packet->descriptor_bytes =
        SPARK_GLM52_RING_WORK_CONTROL_PACKET_BYTES + 1u;
    assert(SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(packet) == 0u);
    packet->descriptor_bytes = packet_bytes;
    packet->flags = SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES;
    assert(SparkGlm52CudaResidentIpcInitializeSubmitWork(
        message,packet,0u) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkGlm52CudaResidentIpcInitializeSubmitWork(
        message,packet,
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT) ==
        SPARK_STATUS_OK);
    free(message);
    free(packet);
}

static void SparkTestMtpTreeExecutionContract(void)
{
    SparkGlm52RequestApiDispatch dispatch;
    uint32_t batch_bucket;
    uint32_t mtp_budget;
    memset(&dispatch,0,sizeof(dispatch));
    dispatch.kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
    dispatch.decode_batch_decision.batch_bucket =
        SPARK_GLM52_STAGE_PLAN_BUCKET_B128;
    assert(SparkGlm52RingWorkControlSelectExecutionBatchBucket(
        &dispatch,
        SPARK_GLM52_STAGE_PLAN_BUCKET_B128,
        &batch_bucket) == SPARK_STATUS_OK);
    assert(batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B128);
    /* Rows beyond the dispatch bucket widen to the next compiled
     * bucket instead of rejecting the dispatch. */
    assert(SparkGlm52RingWorkControlSelectExecutionBatchBucket(
        &dispatch,
        SPARK_GLM52_STAGE_PLAN_BUCKET_B128 + 1u,
        &batch_bucket) == SPARK_STATUS_OK);
    assert(batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B256);
    assert(SparkGlm52RingWorkControlSelectExecutionBatchBucket(
        &dispatch,
        SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET + 1u,
        &batch_bucket) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(SparkGlm52RingWorkControlSelectMtpDraftBudget(
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH,
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        &mtp_budget) == SPARK_STATUS_OK);
    assert(mtp_budget == SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(SparkGlm52RingWorkControlSelectMtpDraftBudget(
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH,
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        &mtp_budget) == SPARK_STATUS_OK);
    assert(mtp_budget == SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(SparkGlm52RingWorkControlSelectMtpDraftBudget(
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH,
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        &mtp_budget) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkGlm52RingWorkControlSelectMtpDraftBudget(
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH,
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY,
        0u,
        &mtp_budget) == SPARK_STATUS_MODULE_NOT_VALIDATED);
}

static void SparkTestResidentIpcReaderPreservesFragments(void)
{
    SparkGlm52CudaResidentIpcHeader header;
    SparkGlm52CudaResidentIpcReader reader;
    uint8_t payload[SPARK_TEST_RESIDENT_IPC_PAYLOAD_BYTES];
    uint8_t received[SPARK_TEST_RESIDENT_IPC_PAYLOAD_BYTES];
    int32_t sockets[2u];
    int32_t flags;
    uint32_t index;

    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    flags = fcntl(sockets[1u],F_GETFL,0);
    assert(flags >= 0);
    assert(fcntl(sockets[1u],F_SETFL,flags | O_NONBLOCK) == 0);
    for (index = 0u; index < sizeof(payload); ++index)
        payload[index] = (uint8_t)(index + 1u);
    SparkGlm52CudaResidentIpcInitializeHeader(
        &header,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_QUERY,3u,91u,
        (uint32_t)sizeof(payload));
    SparkGlm52CudaResidentIpcReaderReset(&reader);
    assert(write(sockets[0u],&header,
        SPARK_TEST_RESIDENT_IPC_HEADER_FRAGMENT_BYTES) ==
        SPARK_TEST_RESIDENT_IPC_HEADER_FRAGMENT_BYTES);
    assert(SparkGlm52CudaResidentIpcReadHeader(
        &reader,sockets[1u],sizeof(payload)) == SPARK_STATUS_BUSY);
    assert(reader.header_offset ==
        SPARK_TEST_RESIDENT_IPC_HEADER_FRAGMENT_BYTES);
    assert(write(sockets[0u],((uint8_t *)&header) +
        SPARK_TEST_RESIDENT_IPC_HEADER_FRAGMENT_BYTES,
        sizeof(header) - SPARK_TEST_RESIDENT_IPC_HEADER_FRAGMENT_BYTES) ==
        (ssize_t)(sizeof(header) -
            SPARK_TEST_RESIDENT_IPC_HEADER_FRAGMENT_BYTES));
    assert(SparkGlm52CudaResidentIpcReadHeader(
        &reader,sockets[1u],sizeof(payload)) == SPARK_STATUS_OK);
    assert(reader.header.kind == SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_QUERY);
    assert(write(sockets[0u],payload,
        SPARK_TEST_RESIDENT_IPC_PAYLOAD_FRAGMENT_BYTES) ==
        SPARK_TEST_RESIDENT_IPC_PAYLOAD_FRAGMENT_BYTES);
    assert(SparkGlm52CudaResidentIpcReadPayload(
        &reader,sockets[1u],received,sizeof(received)) == SPARK_STATUS_BUSY);
    assert(reader.payload_offset ==
        SPARK_TEST_RESIDENT_IPC_PAYLOAD_FRAGMENT_BYTES);
    assert(write(sockets[0u],
        payload + SPARK_TEST_RESIDENT_IPC_PAYLOAD_FRAGMENT_BYTES,
        sizeof(payload) - SPARK_TEST_RESIDENT_IPC_PAYLOAD_FRAGMENT_BYTES) ==
        (ssize_t)(sizeof(payload) -
            SPARK_TEST_RESIDENT_IPC_PAYLOAD_FRAGMENT_BYTES));
    assert(SparkGlm52CudaResidentIpcReadPayload(
        &reader,sockets[1u],received,sizeof(received)) == SPARK_STATUS_OK);
    assert(memcmp(payload,received,sizeof(payload)) == 0);
    close(sockets[0u]);
    close(sockets[1u]);
}

/* Regression: a prompt longer than one bucket of tokens must not be
 * rejected. The scheduler buckets a prefill dispatch by sequence count
 * (1 sequence -> B16) while the packet carries lanes x tokens rows
 * (17 rows); the bucket must widen to cover the rows. */
static void SparkTestPrefillRowsWidenExecutionBucket(void)
{
    SparkGlm52RequestApiDispatch dispatch;
    uint32_t batch_bucket;

    memset(&dispatch,0,sizeof(dispatch));
    dispatch.kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL;
    dispatch.prefill_decision.batch_bucket =
        SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
    assert(SparkGlm52RingWorkControlSelectExecutionBatchBucket(
        &dispatch,
        17u,
        &batch_bucket) == SPARK_STATUS_OK);
    assert(batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B32);
    assert(SparkGlm52RingWorkControlSelectExecutionBatchBucket(
        &dispatch,
        16u,
        &batch_bucket) == SPARK_STATUS_OK);
    assert(batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    /* 16 lanes x 6 verifier rows widen B16 to B128. */
    dispatch.kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
    dispatch.decode_batch_decision.batch_bucket =
        SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
    assert(SparkGlm52RingWorkControlSelectExecutionBatchBucket(
        &dispatch,
        96u,
        &batch_bucket) == SPARK_STATUS_OK);
    assert(batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B128);
}

int main(void)
{
    SparkTestWideDecodePayload();
    SparkTestDecodePayloadCapacity();
    SparkTestInternalDirectoryDecodeHasNoBlockPayload();
    SparkTestWideSubmitWorkUsesVariablePayload();
    SparkTestMtpTreeExecutionContract();
    SparkTestPrefillRowsWidenExecutionBucket();
    SparkTestResidentIpcReaderPreservesFragments();
    return 0;
}
