#ifndef SPARKPIPE_SPARK_CUDA_RING_KERNELS_H
#define SPARKPIPE_SPARK_CUDA_RING_KERNELS_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_STAGE_BOUNDARY_SENTINEL 0x535041524B52494Eull

typedef struct SparkCudaStageBoundaryKernelRequest
{
    uint32_t element_size_bytes;
    uint32_t hidden_size;
    uint32_t physical_slot_count;
    uint32_t active_slot_count;
    uint64_t row_stride_bytes;
    uint64_t expected_payload_checksum;
    uint64_t sentinel;
} SparkCudaStageBoundaryKernelRequest;

typedef struct SparkCudaStageBoundaryKernelReport
{
    uint64_t source_bytes;
    uint64_t packed_bytes;
    uint64_t unpacked_bytes;
    uint64_t payload_checksum;
    uint64_t unpacked_checksum;
    uint64_t trace_checksum;
    uint32_t pack_kernel_count;
    uint32_t unpack_kernel_count;
    uint32_t checksum_kernel_count;
    uint32_t trace_counter_kernel_count;
    uint32_t shape_violation_count;
    uint32_t sentinel_violation_count;
    uint32_t checksum_mismatch_count;
} SparkCudaStageBoundaryKernelReport;

uint64_t SparkComputeCudaStageBoundaryHostChecksum(const uint8_t *bytes, uint64_t byte_count);
SparkStatus SparkValidateCudaStageBoundaryKernelRequest(const SparkCudaStageBoundaryKernelRequest *request);
SparkStatus SparkRunCudaStageBoundaryDeviceKernels(const SparkCudaStageBoundaryKernelRequest *request, const uint8_t *device_input, uint64_t device_input_bytes, uint8_t *device_packet, uint64_t device_packet_bytes, uint8_t *device_output, uint64_t device_output_bytes, SparkCudaStageBoundaryKernelReport *device_report, uint32_t compute_checksum);
SparkStatus SparkRunCudaStageBoundaryKernels(const SparkCudaStageBoundaryKernelRequest *request, const uint8_t *input_host, uint8_t *unpacked_host, uint64_t unpacked_host_bytes, SparkCudaStageBoundaryKernelReport *report);

#ifdef __cplusplus
}
#endif

#endif
