#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"

#include <errno.h>
#include <string.h>
#include <sys/socket.h>

void SparkGlm52CudaResidentIpcInitializeHeader(
    SparkGlm52CudaResidentIpcHeader *header,
    uint32_t kind,
    uint32_t rank_index,
    uint64_t sequence_number,
    uint32_t payload_bytes)
{
    if (header == 0)
        return;
    memset(header, 0, sizeof(*header));
    header->magic = SPARK_GLM52_CUDA_RESIDENT_IPC_MAGIC;
    header->abi_version = SPARK_GLM52_CUDA_RESIDENT_IPC_ABI_VERSION;
    header->descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_HEADER_BYTES;
    header->kind = kind;
    header->payload_bytes = payload_bytes;
    header->rank_index = rank_index;
    header->sequence_number = sequence_number;
}

SparkStatus SparkGlm52CudaResidentIpcValidateHeader(
    const SparkGlm52CudaResidentIpcHeader *header,
    uint32_t expected_kind,
    uint32_t maximum_payload_bytes)
{
    if (header == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (header->magic != SPARK_GLM52_CUDA_RESIDENT_IPC_MAGIC)
        return SPARK_STATUS_PARSE_ERROR;
    if (header->abi_version != SPARK_GLM52_CUDA_RESIDENT_IPC_ABI_VERSION ||
        header->descriptor_bytes != SPARK_GLM52_CUDA_RESIDENT_IPC_HEADER_BYTES)
        return SPARK_STATUS_ABI_MISMATCH;
    if (expected_kind != 0u && header->kind != expected_kind)
        return SPARK_STATUS_SCHEMA_ERROR;
    if (header->payload_bytes > maximum_payload_bytes)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    return SPARK_STATUS_OK;
}

void SparkGlm52CudaResidentIpcReaderReset(
    SparkGlm52CudaResidentIpcReader *reader)
{
    if (reader == 0)
        return;
    memset(reader,0,sizeof(*reader));
}

SparkStatus SparkGlm52CudaResidentIpcReadHeader(
    SparkGlm52CudaResidentIpcReader *reader,
    int32_t fd,
    uint32_t maximum_payload_bytes)
{
    uint8_t *cursor;
    uint32_t remaining;
    ssize_t got;
    if (reader == 0 || fd < 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    while (reader->header_offset < sizeof(reader->header))
    {
        cursor = (uint8_t *)&reader->header;
        remaining = (uint32_t)sizeof(reader->header) - reader->header_offset;
        got = recv(fd,cursor + reader->header_offset,remaining,MSG_DONTWAIT);
        if (got < 0)
        {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return SPARK_STATUS_BUSY;
            return SPARK_STATUS_IO_ERROR;
        }
        if (got == 0)
            return SPARK_STATUS_IO_ERROR;
        reader->header_offset += (uint32_t)got;
    }
    if (reader->header_ready == 0u)
    {
        SparkStatus status;
        status = SparkGlm52CudaResidentIpcValidateHeader(
            &reader->header,0u,maximum_payload_bytes);
        if (status != SPARK_STATUS_OK)
            return status;
        reader->header_ready = 1u;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52CudaResidentIpcReadPayload(
    SparkGlm52CudaResidentIpcReader *reader,
    int32_t fd,
    uint8_t *payload,
    uint32_t payload_capacity)
{
    uint32_t remaining;
    ssize_t got;
    if (reader == 0 || fd < 0 || reader->header_ready == 0u ||
        reader->header.payload_bytes > payload_capacity ||
        (payload == 0 && reader->header.payload_bytes != 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    while (reader->payload_offset < reader->header.payload_bytes)
    {
        remaining = reader->header.payload_bytes - reader->payload_offset;
        got = recv(
            fd,payload + reader->payload_offset,remaining,MSG_DONTWAIT);
        if (got < 0)
        {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return SPARK_STATUS_BUSY;
            return SPARK_STATUS_IO_ERROR;
        }
        if (got == 0)
            return SPARK_STATUS_IO_ERROR;
        reader->payload_offset += (uint32_t)got;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52CudaResidentIpcCalculateWorkMessageBytes(
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t prefix_bytes,
	uint32_t maximum_bytes)
{
    uint64_t message_bytes;

    if (work_packet == 0 ||
        work_packet->descriptor_bytes <
            SPARK_GLM52_PP13_WORK_CONTROL_PACKET_PREFIX_BYTES ||
        work_packet->descriptor_bytes >
            SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES)
        return 0u;
    message_bytes = (uint64_t)prefix_bytes + work_packet->descriptor_bytes;
    return message_bytes <= maximum_bytes ? (uint32_t)message_bytes : 0u;
}

uint32_t SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52CudaResidentIpcCalculateWorkMessageBytes(
		work_packet,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES);
}

SparkStatus SparkGlm52CudaResidentIpcInitializeSubmitWork(
    SparkGlm52CudaResidentIpcSubmitWork *message,
    const SparkGlm52Pp13WorkControlPacket *work_packet,
    uint32_t flags)
{
    uint32_t message_bytes;
    if (message == 0 || work_packet == 0 ||
        (flags & ~SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_KNOWN_FLAGS) != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(message,0,sizeof(*message));
    message->flags = flags;
    message->work_packet = *work_packet;
    message_bytes = SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(
        &message->work_packet);
    if (message_bytes == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    message->descriptor_bytes = message_bytes;
    return SparkGlm52CudaResidentIpcValidateSubmitWork(message,message_bytes);
}

SparkStatus SparkGlm52CudaResidentIpcValidateSubmitWork(
    const SparkGlm52CudaResidentIpcSubmitWork *message,
    uint32_t payload_bytes)
{
    uint32_t expected_bytes;
    if (message == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    expected_bytes = SparkGlm52CudaResidentIpcCalculateSubmitWorkBytes(
        &message->work_packet);
    if (expected_bytes == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (payload_bytes != expected_bytes ||
        message->descriptor_bytes != expected_bytes)
        return SPARK_STATUS_ABI_MISMATCH;
    if ((message->flags &
            ~SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_KNOWN_FLAGS) != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((message->work_packet.flags &
            SPARK_GLM52_PP13_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u &&
        (message->flags &
            SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT) == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

uint32_t SparkGlm52CudaResidentIpcCalculateSubmitPrefillBytes(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52CudaResidentIpcCalculateWorkMessageBytes(
		work_packet,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_PREFILL_PREFIX_BYTES,
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_PREFILL_BYTES);
}

SparkStatus SparkGlm52CudaResidentIpcDecodePayloadBytes(
    uint32_t kv_block_index_count,
    uint32_t *payload_bytes_out)
{
    uint64_t payload_bytes;
    if (payload_bytes_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    payload_bytes =
        (uint64_t)SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES +
        (uint64_t)kv_block_index_count * sizeof(uint32_t);
    if (payload_bytes > UINT32_MAX ||
        payload_bytes > SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    *payload_bytes_out = (uint32_t)payload_bytes;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52CudaResidentIpcValidateSubmitDecode(
    const SparkGlm52CudaResidentIpcSubmitDecode *message,
    uint32_t payload_bytes,
    uint32_t maximum_lane_count)
{
    uint32_t expected_payload_bytes;
    uint32_t expected_block_offset;
    uint32_t expected_mtp_budget;
    uint32_t internal_kv_directory;
    uint32_t lane_index;
    SparkStatus status;
    if (message == 0 || maximum_lane_count == 0u ||
        maximum_lane_count > SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52CudaResidentIpcDecodePayloadBytes(
        message->kv_block_index_count,&expected_payload_bytes);
    if (status != SPARK_STATUS_OK)
        return status;
    if (message->descriptor_bytes !=
            SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES)
        return SPARK_STATUS_ABI_MISMATCH;
    if (payload_bytes != expected_payload_bytes ||
        message->control_generation == 0u ||
        message->lane_count == 0u ||
        message->lane_count > maximum_lane_count ||
        message->active_sequence_count != message->lane_count ||
        SparkGlm52StagePlanBatchBucketIsSupported(
            message->execution_batch_bucket) == 0u ||
        message->lane_count > message->execution_batch_bucket ||
        message->kv_block_token_count == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((message->resident_flags &
            ~SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_KNOWN_FLAGS) != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    internal_kv_directory = (message->resident_flags &
        SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY) != 0u;
    if ((internal_kv_directory != 0u) !=
        (message->kv_block_index_count == 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ((message->dispatch_kind !=
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
         message->dispatch_kind !=
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH) ||
        message->speculative_token_count >
            SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
        (message->dispatch_kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
         message->speculative_token_count != 0u) ||
        (message->dispatch_kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
         message->speculative_token_count == 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52Pp13WorkControlSelectMtpDraftBudget(
        message->dispatch_kind,
        message->request_flags,
        message->lanes[0u].mtp_draft_token_budget,
        &expected_mtp_budget);
    if (status != SPARK_STATUS_OK)
        return status;
    expected_block_offset = 0u;
    for (lane_index = 0u; lane_index < message->lane_count; ++lane_index)
    {
        const SparkGlm52CudaResidentIpcDecodeLane *lane;
        lane = &message->lanes[lane_index];
        if (lane->request_id == 0u || lane->sequence_id == 0u ||
            lane->request_slot_index == UINT32_MAX ||
            lane->context_token_count == 0u ||
            lane->mtp_draft_token_budget != expected_mtp_budget ||
            lane->speculative_token_count !=
                message->speculative_token_count ||
            (lane->mtp_resolution_proposed_token_count == 0u &&
             (lane->mtp_resolution_accepted_token_count != 0u ||
              lane->mtp_resolution_path_id !=
                SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE)) ||
            lane->mtp_resolution_proposed_token_count >
                SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
            SparkGlm52MtpTreeResolutionIsValid(
                lane->mtp_resolution_proposed_token_count,
                lane->mtp_resolution_accepted_token_count,
                lane->mtp_resolution_path_id) == 0u ||
            lane->kv_block_offset != expected_block_offset ||
            (internal_kv_directory != 0u && lane->kv_block_count != 0u) ||
            (internal_kv_directory == 0u && lane->kv_block_count == 0u) ||
            lane->kv_block_count >
                SPARK_GLM52_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS ||
            lane->kv_block_count >
                message->kv_block_index_count - expected_block_offset)
            return SPARK_STATUS_INVALID_ARGUMENT;
        expected_block_offset += lane->kv_block_count;
    }
    if (expected_block_offset != message->kv_block_index_count)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}
