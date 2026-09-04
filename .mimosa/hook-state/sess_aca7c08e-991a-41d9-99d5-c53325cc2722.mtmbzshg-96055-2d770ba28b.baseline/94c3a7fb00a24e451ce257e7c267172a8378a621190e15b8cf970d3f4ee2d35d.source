#pragma once


#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_pager.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV4_JIT_KV_ABI_VERSION 1u
#define SPARK_DSV4_KV_FRAMES_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkDsv4KvFramesConfiguration))
#define SPARK_DSV4_KV_FRAMES_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkDsv4KvFrames))
#define SPARK_DSV4_KV_FRAME_OP_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkDsv4KvFrameOp))
#define SPARK_DSV4_JIT_KV_PARKABILITY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkDsv4JitKvParkability))

#define SPARK_DSV4_KV_FRAMES_OP_SAVE_OUT 0x00001000u
#define SPARK_DSV4_KV_FRAMES_OP_RESTORE_IN 0x00002000u

typedef enum SparkDsv4KvFramesBackend
{
    SPARK_DSV4_KV_FRAMES_BACKEND_TERM_COPY = 0u,
    SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS = 1u
}
SparkDsv4KvFramesBackend;

typedef struct SparkDsv4KvFrameOp
{
    uint32_t op_code;
    uint32_t block_count;
    uint64_t key_bytes;
    uint64_t value_bytes;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
    void *host_staging;
}
SparkDsv4KvFrameOp;

typedef SparkStatus (*SparkDsv4KvFrameSubmitFunction)(
    void *submit_context,
    const SparkDsv4KvFrameOp *op);

typedef struct SparkDsv4KvFramesConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t backend;
    uint32_t reserved0;
    uint64_t key_block_stride_bytes;
    uint64_t value_block_stride_bytes;
    void *submit_context;
    SparkDsv4KvFrameSubmitFunction submit;
}
SparkDsv4KvFramesConfiguration;

typedef struct SparkDsv4KvFrames
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t backend;
    uint32_t spark_receipt_valid;
    uint64_t key_block_stride_bytes;
    uint64_t value_block_stride_bytes;
    SparkDsv4KvFrameOp staged;
    void *submit_context;
    SparkDsv4KvFrameSubmitFunction submit;
    uint64_t staged_count;
    uint64_t host_copy_count;
    uint64_t device_run_count;
    uint64_t refused_count;
}
SparkDsv4KvFrames;

typedef struct SparkDsv4JitKvParkability
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t parkable_block_count;
    uint32_t pinned_block_count;
    uint32_t non_resident_block_count;
    uint32_t unprotected_active_count;
}
SparkDsv4JitKvParkability;

SparkStatus SparkDsv4KvFramesInitialize(
    SparkDsv4KvFrames *frames,
    const SparkDsv4KvFramesConfiguration *configuration);

void SparkDsv4KvFramesAcceptSparkReceipt(SparkDsv4KvFrames *frames);

SparkStatus SparkDsv4KvFramesSave(
    void *module_context,
    const SparkKvPagerBlockView *view);

SparkStatus SparkDsv4KvFramesRestore(
    void *module_context,
    const SparkKvPagerBlockView *view);

SparkStatus SparkDsv4KvFramesSubmitHostCopy(
    void *submit_context,
    const SparkDsv4KvFrameOp *op);

SparkStatus SparkDsv4JitKvDecideParkability(
    const SparkKvCacheArena *arena,
    const uint32_t *active_block_indices,
    uint32_t active_block_count,
    SparkDsv4JitKvParkability *parkability_out);

#ifdef __cplusplus
}
#endif
