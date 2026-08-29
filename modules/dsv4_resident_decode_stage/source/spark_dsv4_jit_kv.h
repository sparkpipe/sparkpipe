#pragma once

// The family wiring of the JIT-KV pager (docs/JIT_KV_DESIGN.md step 2,
// docs/JIT_KV_RESPONSE.md W1) for this module's resident decode stage: the
// two frame-context ops the pager's module seam calls, and the deployment
// parkability condition admission agrees with.
//
// THE FRAME OPS. KV_BLOCKS_SAVE_OUT (0x00001000u) streams a block's device
// planes into the pager's staging; KV_BLOCKS_RESTORE_IN (0x00002000u)
// streams the staged bytes back. The layout is the pager's staging
// contract: the key plane then the value plane, contiguous. On the host
// (TERM) the device planes are host mappings and the op is a plain copy.
// On the device the op is the frame-context submission installed at the
// spark-side module open, gated on that open's validated receipt: a
// device-plane copy is STAGED here (op code, plane addresses, byte counts
// - exactly what the frame context will execute) and NEVER RUN without the
// receipt; the host answer is UNSUPPORTED and the staging stays untouched.
//
// PARKABILITY. The deployment condition is decided against the shared
// arena predicate (SparkKvCacheArenaBlockIsParkable), which IS the
// resident-eviction selector's exclusions: the active decode set is
// protected by residency pins, everything else resident is parkable, and
// admission counts the very same test - the deployment's answer and the
// pager's pool can never disagree.

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

/* the design doc's frame-context op codes */
#define SPARK_DSV4_KV_FRAMES_OP_SAVE_OUT 0x00001000u
#define SPARK_DSV4_KV_FRAMES_OP_RESTORE_IN 0x00002000u

typedef enum SparkDsv4KvFramesBackend
{
    /* host TERM: the device planes are host mappings; the op is a copy */
    SPARK_DSV4_KV_FRAMES_BACKEND_TERM_COPY = 0u,
    /* the device-plane submission; requires the spark-side receipt */
    SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS = 1u
}
SparkDsv4KvFramesBackend;

/* One staged device-plane copy: exactly what the frame context executes on
 * a spark, and the receipt of what the TERM copy ran on the host. */
typedef struct SparkDsv4KvFrameOp
{
    uint32_t op_code;          /* SAVE_OUT streams planes OUT of the device */
    uint32_t block_count;      /* the pager stages/restores per block */
    uint64_t key_bytes;
    uint64_t value_bytes;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
    void *host_staging;        /* save: destination; restore: source */
}
SparkDsv4KvFrameOp;

/* The device-plane submission the module installs at its spark-side open.
 * Zero on a host build: the spark-gated backend cannot be initialized
 * without it. */
typedef SparkStatus (*SparkDsv4KvFrameSubmitFunction)(
    void *submit_context,
    const SparkDsv4KvFrameOp *op);

typedef struct SparkDsv4KvFramesConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t backend;          /* SparkDsv4KvFramesBackend */
    uint32_t reserved0;
    uint64_t key_block_stride_bytes;   /* the arena's strides == the pager's
                                          block payload layout */
    uint64_t value_block_stride_bytes; /* zero for key-only arenas */
    void *submit_context;
    SparkDsv4KvFrameSubmitFunction submit; /* TERM: zero installs the host
                                              copy primitive */
}
SparkDsv4KvFramesConfiguration;

typedef struct SparkDsv4KvFrames
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t backend;
    uint32_t spark_receipt_valid; /* the spark-side module open sets this */
    uint64_t key_block_stride_bytes;
    uint64_t value_block_stride_bytes;
    SparkDsv4KvFrameOp staged;    /* the last op, staged */
    void *submit_context;
    SparkDsv4KvFrameSubmitFunction submit;
    uint64_t staged_count;        /* ops staged for the device plane */
    uint64_t host_copy_count;     /* TERM copies run */
    uint64_t device_run_count;    /* device-plane submissions handed off */
    uint64_t refused_count;       /* device ops refused: receipt missing */
}
SparkDsv4KvFrames;

typedef struct SparkDsv4JitKvParkability
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t parkable_block_count;     /* residents admission may park */
    uint32_t pinned_block_count;       /* residents the selector protects */
    uint32_t non_resident_block_count; /* parked or blank: not device state */
    uint32_t unprotected_active_count; /* active-set blocks found parkable */
}
SparkDsv4JitKvParkability;

/* Fences the configuration and installs the submit primitive. The
 * device-plane backend refuses without its submission: that installs at
 * the spark-side module open, never in a host build. */
SparkStatus SparkDsv4KvFramesInitialize(
    SparkDsv4KvFrames *frames,
    const SparkDsv4KvFramesConfiguration *configuration);

/* Spark-side module open ONLY: the validated module receipt enables the
 * device-plane backend. A host build never holds a receipt. */
void SparkDsv4KvFramesAcceptSparkReceipt(SparkDsv4KvFrames *frames);

/* The pager's module seam (SparkKvPagerModuleSaveFunction): stage the
 * device planes into pager staging under KV_BLOCKS_SAVE_OUT. */
SparkStatus SparkDsv4KvFramesSave(
    void *module_context,
    const SparkKvPagerBlockView *view);

/* The pager's module seam, reversed (SparkKvPagerModuleRestoreFunction):
 * KV_BLOCKS_RESTORE_IN streams the staged bytes back into the planes. */
SparkStatus SparkDsv4KvFramesRestore(
    void *module_context,
    const SparkKvPagerBlockView *view);

/* The host TERM copy primitive (the default submit for the TERM backend):
 * the frame op executed against host mappings. */
SparkStatus SparkDsv4KvFramesSubmitHostCopy(
    void *submit_context,
    const SparkDsv4KvFrameOp *op);

/* The deployment parkability condition, over the shared arena predicate.
 * Counts every block of the deployment's arena as parkable, pinned, or
 * non-resident, and verifies the ACTIVE decode set is protected: a block
 * listed active that the predicate calls parkable fails the condition
 * (BUSY, with the count named) - pin the active set before offering work;
 * dispatch may not run against a block the pager is free to park. */
SparkStatus SparkDsv4JitKvDecideParkability(
    const SparkKvCacheArena *arena,
    const uint32_t *active_block_indices,
    uint32_t active_block_count,
    SparkDsv4JitKvParkability *parkability_out);

#ifdef __cplusplus
}
#endif
