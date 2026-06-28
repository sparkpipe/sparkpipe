#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_ring_kernels.h"

static __device__ uint64_t SparkCudaRingMixU64Device(uint64_t checksum, uint64_t value)
{
    checksum ^= value + 0x9e3779b97f4a7c15ull + (checksum << 6u) + (checksum >> 2u);
    return checksum;
}

static __device__ uint64_t SparkCudaRingPayloadChecksumDevice(const uint8_t *bytes, uint64_t byte_count)
{
    uint64_t byte_index;
    uint64_t checksum;

    checksum = 0x535047505543484Bull;
    checksum = SparkCudaRingMixU64Device(checksum, byte_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkCudaRingMixU64Device(checksum, bytes[byte_index]);
    }

    return checksum;
}

static __global__ void SparkCudaStageBoundaryPackKernel(const uint8_t *input, uint8_t *packet, uint64_t row_bytes, uint64_t row_stride_bytes, uint64_t packed_bytes)
{
    uint64_t byte_index;
    uint64_t row_index;
    uint64_t row_offset;

    byte_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (byte_index < packed_bytes)
    {
        row_index = byte_index / row_bytes;
        row_offset = byte_index - (row_index * row_bytes);
        packet[byte_index] = input[(row_index * row_stride_bytes) + row_offset];
        byte_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaStageBoundaryUnpackKernel(const uint8_t *packet, uint8_t *output, uint64_t row_bytes, uint64_t row_stride_bytes, uint64_t packed_bytes)
{
    uint64_t byte_index;
    uint64_t row_index;
    uint64_t row_offset;

    byte_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (byte_index < packed_bytes)
    {
        row_index = byte_index / row_bytes;
        row_offset = byte_index - (row_index * row_bytes);
        output[(row_index * row_stride_bytes) + row_offset] = packet[byte_index];
        byte_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaStageBoundaryPackVector16Kernel(const uint8_t *input, uint8_t *packet, uint64_t row_chunks, uint64_t row_stride_chunks, uint64_t packed_chunks)
{
    const uint4 *input_chunks;
    uint4 *packet_chunks;
    uint64_t chunk_index;
    uint64_t row_index;
    uint64_t row_offset;

    input_chunks = (const uint4 *)input;
    packet_chunks = (uint4 *)packet;
    chunk_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (chunk_index < packed_chunks)
    {
        row_index = chunk_index / row_chunks;
        row_offset = chunk_index - (row_index * row_chunks);
        packet_chunks[chunk_index] = input_chunks[(row_index * row_stride_chunks) + row_offset];
        chunk_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaStageBoundaryUnpackVector16Kernel(const uint8_t *packet, uint8_t *output, uint64_t row_chunks, uint64_t row_stride_chunks, uint64_t packed_chunks)
{
    const uint4 *packet_chunks;
    uint4 *output_chunks;
    uint64_t chunk_index;
    uint64_t row_index;
    uint64_t row_offset;

    packet_chunks = (const uint4 *)packet;
    output_chunks = (uint4 *)output;
    chunk_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (chunk_index < packed_chunks)
    {
        row_index = chunk_index / row_chunks;
        row_offset = chunk_index - (row_index * row_chunks);
        output_chunks[(row_index * row_stride_chunks) + row_offset] = packet_chunks[chunk_index];
        chunk_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaStageBoundaryChecksumKernel(const uint8_t *packet, uint64_t packed_bytes, uint64_t expected_checksum, uint64_t sentinel, uint64_t packet_sentinel, SparkCudaStageBoundaryKernelReport *report)
{
    uint64_t checksum;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    checksum = SparkCudaRingPayloadChecksumDevice(packet, packed_bytes);
    report->payload_checksum = checksum;
    report->checksum_kernel_count += 1u;
    if (expected_checksum != 0u && expected_checksum != checksum)
    {
        report->checksum_mismatch_count += 1u;
    }

    if (packet_sentinel != sentinel)
    {
        report->sentinel_violation_count += 1u;
    }
}

static __global__ void SparkCudaStageBoundaryUnpackedChecksumKernel(const uint8_t *output, uint64_t row_bytes, uint64_t row_stride_bytes, uint64_t active_slot_count, SparkCudaStageBoundaryKernelReport *report)
{
    uint64_t row_index;
    uint64_t byte_index;
    uint64_t checksum;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    checksum = 0x535047505543484Bull;
    checksum = SparkCudaRingMixU64Device(checksum, row_bytes * active_slot_count);
    for (row_index = 0; row_index < active_slot_count; ++row_index)
    {
        for (byte_index = 0; byte_index < row_bytes; ++byte_index)
        {
            checksum = SparkCudaRingMixU64Device(checksum, output[(row_index * row_stride_bytes) + byte_index]);
        }
    }

    report->unpacked_checksum = checksum;
}

static __global__ void SparkCudaStageBoundaryTraceKernel(SparkCudaStageBoundaryKernelRequest request, uint64_t source_bytes, uint64_t packed_bytes, SparkCudaStageBoundaryKernelReport *report)
{
    uint64_t trace_checksum;

    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    report->source_bytes = source_bytes;
    report->packed_bytes = packed_bytes;
    report->unpacked_bytes = source_bytes;
    report->pack_kernel_count += 1u;
    report->unpack_kernel_count += 1u;
    report->trace_counter_kernel_count += 1u;
    trace_checksum = 0x5350545241434531ull;
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, request.element_size_bytes);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, request.hidden_size);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, request.physical_slot_count);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, request.active_slot_count);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, request.row_stride_bytes);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, source_bytes);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, packed_bytes);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, report->payload_checksum);
    trace_checksum = SparkCudaRingMixU64Device(trace_checksum, report->unpacked_checksum);
    report->trace_checksum = trace_checksum;
}

static uint64_t SparkCudaStageBoundaryRowBytesHost(const SparkCudaStageBoundaryKernelRequest *request)
{
    return (uint64_t)request->element_size_bytes * (uint64_t)request->hidden_size;
}

static uint64_t SparkCudaStageBoundarySourceBytesHost(const SparkCudaStageBoundaryKernelRequest *request)
{
    return request->row_stride_bytes * (uint64_t)request->physical_slot_count;
}

static uint32_t SparkCudaStageBoundaryBlockCountHost(uint64_t item_count)
{
    uint32_t block_count;

    block_count = (uint32_t)((item_count + 255u) / 256u);
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 4096u)
    {
        block_count = 4096u;
    }
    return block_count;
}

static bool SparkCudaStageBoundaryCanUseVector16Host(const void *input, const void *packet, const void *output, uint64_t row_bytes, uint64_t row_stride_bytes)
{
    if ((row_bytes & 15u) != 0u || (row_stride_bytes & 15u) != 0u)
    {
        return false;
    }
    if ((((uintptr_t)input | (uintptr_t)packet | (uintptr_t)output) & 15u) != 0u)
    {
        return false;
    }
    return true;
}

extern "C" SparkStatus SparkRunCudaStageBoundaryDeviceKernels(const SparkCudaStageBoundaryKernelRequest *request, const uint8_t *device_input, uint64_t device_input_bytes, uint8_t *device_packet, uint64_t device_packet_bytes, uint8_t *device_output, uint64_t device_output_bytes, SparkCudaStageBoundaryKernelReport *device_report, uint32_t compute_checksum)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t row_bytes;
    uint64_t source_bytes;
    uint64_t packed_bytes;
    uint64_t row_chunks;
    uint64_t row_stride_chunks;
    uint64_t packed_chunks;
    uint32_t block_count;

    status = SparkValidateCudaStageBoundaryKernelRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    source_bytes = SparkCudaStageBoundarySourceBytesHost(request);
    row_bytes = SparkCudaStageBoundaryRowBytesHost(request);
    packed_bytes = row_bytes * (uint64_t)request->active_slot_count;
    if (device_input == 0 || device_packet == 0 || device_output == 0 || device_report == 0 || device_input_bytes < source_bytes || device_packet_bytes < packed_bytes || device_output_bytes < source_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cuda_status = cudaMemset(device_report, 0, sizeof(*device_report));
    if (SparkCudaStageBoundaryCanUseVector16Host(device_input, device_packet, device_output, row_bytes, request->row_stride_bytes))
    {
        row_chunks = row_bytes >> 4u;
        row_stride_chunks = request->row_stride_bytes >> 4u;
        packed_chunks = packed_bytes >> 4u;
        block_count = SparkCudaStageBoundaryBlockCountHost(packed_chunks);
        if (cuda_status == cudaSuccess)
        {
            SparkCudaStageBoundaryPackVector16Kernel<<<block_count, 256>>>(device_input, device_packet, row_chunks, row_stride_chunks, packed_chunks);
            cuda_status = cudaGetLastError();
        }
        if (cuda_status == cudaSuccess)
        {
            SparkCudaStageBoundaryUnpackVector16Kernel<<<block_count, 256>>>(device_packet, device_output, row_chunks, row_stride_chunks, packed_chunks);
            cuda_status = cudaGetLastError();
        }
    }
    else
    {
        block_count = SparkCudaStageBoundaryBlockCountHost(packed_bytes);
        if (cuda_status == cudaSuccess)
        {
            SparkCudaStageBoundaryPackKernel<<<block_count, 256>>>(device_input, device_packet, row_bytes, request->row_stride_bytes, packed_bytes);
            cuda_status = cudaGetLastError();
        }
        if (cuda_status == cudaSuccess)
        {
            SparkCudaStageBoundaryUnpackKernel<<<block_count, 256>>>(device_packet, device_output, row_bytes, request->row_stride_bytes, packed_bytes);
            cuda_status = cudaGetLastError();
        }
    }
    if (cuda_status == cudaSuccess && compute_checksum != 0u)
    {
        SparkCudaStageBoundaryChecksumKernel<<<1, 32>>>(device_packet, packed_bytes, request->expected_payload_checksum, request->sentinel, request->sentinel, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess && compute_checksum != 0u)
    {
        SparkCudaStageBoundaryUnpackedChecksumKernel<<<1, 32>>>(device_output, row_bytes, request->row_stride_bytes, request->active_slot_count, device_report);
        cuda_status = cudaGetLastError();
    }
    if (cuda_status == cudaSuccess)
    {
        SparkCudaStageBoundaryTraceKernel<<<1, 32>>>(*request, source_bytes, packed_bytes, device_report);
        cuda_status = cudaGetLastError();
    }
    return cuda_status == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkRunCudaStageBoundaryKernels(const SparkCudaStageBoundaryKernelRequest *request, const uint8_t *input_host, uint8_t *unpacked_host, uint64_t unpacked_host_bytes, SparkCudaStageBoundaryKernelReport *report)
{
    uint8_t *device_input;
    uint8_t *device_packet;
    uint8_t *device_output;
    SparkCudaStageBoundaryKernelReport *device_report;
    SparkCudaStageBoundaryKernelReport host_report;
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t row_bytes;
    uint64_t source_bytes;
    uint64_t packed_bytes;
    SparkStatus device_status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }

    status = SparkValidateCudaStageBoundaryKernelRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    source_bytes = SparkCudaStageBoundarySourceBytesHost(request);
    row_bytes = SparkCudaStageBoundaryRowBytesHost(request);
    packed_bytes = row_bytes * (uint64_t)request->active_slot_count;
    if (input_host == 0 || unpacked_host == 0 || unpacked_host_bytes < source_bytes || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    device_input = 0;
    device_packet = 0;
    device_output = 0;
    device_report = 0;
    cuda_status = cudaMalloc((void **)&device_input, source_bytes);
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_packet, packed_bytes);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_output, source_bytes);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMalloc((void **)&device_report, sizeof(*device_report));
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(device_input, input_host, source_bytes, cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemset(device_output, 0, source_bytes);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemset(device_report, 0, sizeof(*device_report));
    }

    if (cuda_status == cudaSuccess)
    {
        device_status = SparkRunCudaStageBoundaryDeviceKernels(request, device_input, source_bytes, device_packet, packed_bytes, device_output, source_bytes, device_report, 1u);
        if (device_status != SPARK_STATUS_OK)
        {
            cuda_status = cudaErrorUnknown;
        }
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaDeviceSynchronize();
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(unpacked_host, device_output, source_bytes, cudaMemcpyDeviceToHost);
    }
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(&host_report, device_report, sizeof(host_report), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_report);
    cudaFree(device_output);
    cudaFree(device_packet);
    cudaFree(device_input);

    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    *report = host_report;
    if (report->shape_violation_count != 0u || report->sentinel_violation_count != 0u || report->checksum_mismatch_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}
