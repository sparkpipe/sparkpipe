#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_ring_kernels.h"

static uint64_t SparkCudaStageBoundaryRowBytes(const SparkCudaStageBoundaryKernelRequest *request)
{
    if (request == 0)
    {
        return 0;
    }

    return (uint64_t)request->element_size_bytes * (uint64_t)request->hidden_size;
}

static uint64_t SparkCudaStageBoundarySourceBytes(const SparkCudaStageBoundaryKernelRequest *request)
{
    if (request == 0)
    {
        return 0;
    }

    return request->row_stride_bytes * (uint64_t)request->physical_slot_count;
}

uint64_t SparkComputeCudaStageBoundaryHostChecksum(const uint8_t *bytes, uint64_t byte_count)
{
    uint64_t byte_index;
    uint64_t checksum;

    if (bytes == 0 || byte_count == 0)
    {
        return 0;
    }

    checksum = 0x535047505543484Bull;
    checksum = SparkMixU64(checksum, byte_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkMixU64(checksum, bytes[byte_index]);
    }

    return checksum;
}

SparkStatus SparkValidateCudaStageBoundaryKernelRequest(const SparkCudaStageBoundaryKernelRequest *request)
{
    uint64_t row_bytes;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    row_bytes = SparkCudaStageBoundaryRowBytes(request);
    if (row_bytes == 0 || request->row_stride_bytes < row_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (request->physical_slot_count == 0 || request->active_slot_count == 0 || request->active_slot_count > request->physical_slot_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkCudaStageBoundarySourceBytes(request) == 0 || request->sentinel == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaStageBoundaryDeviceKernels(const SparkCudaStageBoundaryKernelRequest *request, const uint8_t *device_input, uint64_t device_input_bytes, uint8_t *device_packet, uint64_t device_packet_bytes, uint8_t *device_output, uint64_t device_output_bytes, SparkCudaStageBoundaryKernelReport *device_report, uint32_t compute_checksum)
{
    SparkStatus status;
    uint64_t source_bytes;
    uint64_t packed_bytes;

    if (device_report != 0)
    {
        memset(device_report, 0, sizeof(*device_report));
    }
    status = SparkValidateCudaStageBoundaryKernelRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    source_bytes = SparkCudaStageBoundarySourceBytes(request);
    packed_bytes = SparkCudaStageBoundaryRowBytes(request) * (uint64_t)request->active_slot_count;
    if (device_input == 0 || device_packet == 0 || device_output == 0 || device_report == 0 || device_input_bytes < source_bytes || device_packet_bytes < packed_bytes || device_output_bytes < source_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    (void)compute_checksum;
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaStageBoundaryKernels(const SparkCudaStageBoundaryKernelRequest *request, const uint8_t *input_host, uint8_t *unpacked_host, uint64_t unpacked_host_bytes, SparkCudaStageBoundaryKernelReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }

    status = SparkValidateCudaStageBoundaryKernelRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (input_host == 0 || unpacked_host == 0 || unpacked_host_bytes < SparkCudaStageBoundarySourceBytes(request) || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
