#pragma once

// spark_hw_iface.h - SparkPipe hardware-agnostic model-device interface, v1.
//
// Frozen contract: hwiface_v1.md sections 4.0 (status model), 4.1 (target
// descriptor), and 4.2 (primitive families - exact signatures). The header is
// owned by the interface, not by either target; every symbol below is C
// linkage and is provided only by the per-target archive:
//
//   cuda.sm121.gb10    -> spark_hw_cuda_*  (the GB10 island archive)
//   rocm.gfx950.mi350p -> spark_hw_rocm_*  (the MI350P island archive)
//
// Handles are opaque pointers (rule R3). The portable core moves them; it
// never inspects them. A missing target symbol is a link error, never a
// runtime search (rule R2).

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status model (hwiface_v1.md section 4.0). Targets must not invent further
// mappings. Allocation failure returns SPARK_HW_EXHAUSTED, never INVALID;
// device/context loss returns SPARK_HW_LOST from any primitive on affected
// handles; there is no in-place recovery in v1.
// ---------------------------------------------------------------------------

typedef enum SparkHwStatus
{
    SPARK_HW_OK = 0,
    SPARK_HW_NOT_READY,       /* async op not yet complete (query only) */
    SPARK_HW_INVALID,         /* bad handle/argument */
    SPARK_HW_UNSUPPORTED,     /* capability/descriptor check failed */
    SPARK_HW_EXHAUSTED,       /* capacity refusal: pool/pinned OOM, graph
                                 resources; retryable, feeds core admission/
                                 backpressure policy */
    SPARK_HW_LOST             /* device/context lost; instance is fail-closed */
} SparkHwStatus;

const char *SparkHwStatusToString(SparkHwStatus status);

// ---------------------------------------------------------------------------
// Target descriptor (hwiface_v1.md section 4.1). Read once in
// ModuleInitialize; never consulted in the hot path. wavefront_lanes is
// informational and must feed no gate.
// ---------------------------------------------------------------------------

typedef struct SparkHwTarget
{
    const char *target_id;                /* frozen strings, section 6 */
    uint32_t    abi_version;              /* 1 */
    uint32_t    multiprocessor_count;     /* gfx950: CU count */
    uint64_t    max_dynamic_shared_bytes; /* gfx950: its LDS limit */
    uint32_t    capability_major, capability_minor;
    uint32_t    wavefront_lanes;          /* GB10: 32; gfx950: 64 (advisory,
                                             informational) */
} SparkHwTarget;

// ---------------------------------------------------------------------------
// Primitive families (hwiface_v1.md section 4.2) - names, argument shapes,
// and semantics are frozen.
// ---------------------------------------------------------------------------

/* Handles are opaque pointers (rule R3). The frozen contract sketches them as
 * "typedef void <Name>"; because ISO C forbids void parameters, the binding
 * rendering used by both targets is pointer-to-incomplete-tag: still fully
 * opaque to the portable core (it moves them, never inspects them), and it
 * keeps distinct handle kinds distinct at compile time. */
typedef struct SparkHwMemoryTag *SparkHwMemory;   /* opaque pool/workspace backing */
typedef struct SparkHwQueueTag *SparkHwQueue;
typedef struct SparkHwEventTag *SparkHwEvent;
typedef struct SparkHwGraphExecTag *SparkHwGraphExec;

/* Copy direction for spark_hw_copy_async. */
typedef enum SparkHwCopyKind
{
    SPARK_HW_COPY_H2D = 0,
    SPARK_HW_COPY_D2D = 1,
    SPARK_HW_COPY_D2H = 2
} SparkHwCopyKind;

/* Flag values mirror the CUDA/HIP numeric encodings so the per-target wrapper
 * translation is an identity mapping. Each target archive must enforce the
 * mirror mechanically (the ROCm target does so with _Static_assert in its
 * internal header); a target that translates by other means opts out of the
 * mirror but keeps these frozen wire values. Unknown flag bits are refused with
 * SPARK_HW_UNSUPPORTED rather than passed through silently. */
#define SPARK_HW_QUEUE_DEFAULT        0x0u
#define SPARK_HW_QUEUE_NONBLOCKING    0x1u

#define SPARK_HW_PINNED_DEFAULT       0x0u
#define SPARK_HW_PINNED_PORTABLE      0x1u
#define SPARK_HW_PINNED_MAPPED        0x2u

#define SPARK_HW_EVENT_DISABLE_TIMING 0x2u

/* Graph capture mode: relaxed capture preserved by the mode argument.
 * v1 accepts only the relaxed mode. */
#define SPARK_HW_CAPTURE_RELAXED      0x1u

/* Memory */
SparkHwStatus spark_hw_memory_pool_alloc(SparkHwMemory **mem, uint64_t bytes);
SparkHwStatus spark_hw_memory_pool_free(SparkHwMemory *mem);
SparkHwStatus spark_hw_host_pinned_alloc(void **host, uint64_t bytes, uint32_t flags);
SparkHwStatus spark_hw_host_pinned_free(void *host);
SparkHwStatus spark_hw_host_device_pointer(void *host, void **device);
SparkHwStatus spark_hw_copy_async(void *dst, const void *src, uint64_t bytes,
                                  SparkHwCopyKind kind /*H2D|D2D|D2H*/, SparkHwQueue q);
SparkHwStatus spark_hw_memset_async(void *dst, uint32_t value, uint64_t bytes, SparkHwQueue q);
SparkHwStatus spark_hw_read_ahead(SparkHwQueue q, void *sink_u32,
                                  uint32_t word_capacity, void *context);

/* Queue */
SparkHwStatus spark_hw_queue_create(SparkHwQueue *q, uint32_t flags, int32_t priority);
SparkHwStatus spark_hw_queue_destroy(SparkHwQueue q);
SparkHwStatus spark_hw_queue_query(SparkHwQueue q);          /* SPARK_HW_NOT_READY allowed */
SparkHwStatus spark_hw_queue_synchronize(SparkHwQueue q);    /* failure paths ONLY */
SparkHwStatus spark_hw_queue_enqueue_host_callback(SparkHwQueue q, void (*fn)(void*), void *arg);

/* Event - timing NOT required (today: cudaEventDisableTiming) */
SparkHwStatus spark_hw_event_create(SparkHwEvent *e, uint32_t flags);
SparkHwStatus spark_hw_event_destroy(SparkHwEvent e);
SparkHwStatus spark_hw_event_record(SparkHwEvent e, SparkHwQueue q);
SparkHwStatus spark_hw_queue_wait_event(SparkHwQueue q, SparkHwEvent e);

/* Graph - relaxed capture mode preserved by the mode argument */
SparkHwStatus spark_hw_graph_capture_begin(SparkHwQueue q, uint32_t mode /*relaxed*/);
SparkHwStatus spark_hw_graph_capture_end(SparkHwQueue q, void *graph_out);
SparkHwStatus spark_hw_graph_instantiate(void *graph, SparkHwGraphExec *exec);
SparkHwStatus spark_hw_graph_launch(SparkHwGraphExec exec, SparkHwQueue q);
SparkHwStatus spark_hw_graph_destroy(void *graph);
SparkHwStatus spark_hw_graph_exec_destroy(SparkHwGraphExec exec);

#ifdef __cplusplus
}
#endif
