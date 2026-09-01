#ifndef SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_RUNNER_H
#define SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_RUNNER_H

#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_collective.h"
#include "sparkpipe/spark_tp_device_collective.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * K3 resident decode stage runner: one PP stage's embed -> slice -> head
 * execution over the dispatch (pack, bind, pools, K3StageSlice), with the
 * TP4 all-reduce wiring. The slice's layer_collective hook folds the summed
 * input-sharded projection outputs into the AttnRes partial per layer; the
 * embedding and the head candidates use the same ring collective, the head
 * through a slot-encoded sum (the sum over the rank slots IS the all-gather
 * of the four local argmaxes, and the winner is reduced locally - the sum
 * collective needs no max primitive).
 */

#define SPARK_K3_STAGE_RUNNER_ABI_VERSION 1u
#define SPARK_K3_STAGE_RUNNER_CONFIGURATION_BYTES \
    ((uint32_t)sizeof(SparkK3StageRunnerConfiguration))
#define SPARK_K3_STAGE_RUNNER_DISPATCH_BYTES \
    ((uint32_t)sizeof(SparkK3StageRunnerDispatch))
#define SPARK_K3_STAGE_RUNNER_BYTES \
    ((uint32_t)sizeof(SparkK3StageRunner))

#define SPARK_K3_STAGE_RUNNER_FLAG_TENSOR_PARALLEL 0x00000001u
/* Capture the per-shape slice launch (dense offsets + all layers) into a
 * CUDA graph and replay it on later submits. Requires a non-default
 * execution stream and a capture-safe tier (NCCL device collective, or
 * tp_degree 1 with no collective); otherwise the runner stays on direct
 * launches. The first submit of a shape warms (shared-memory opt-ins +
 * tensor-map encodes must precede capture); the second captures; replays
 * follow. */
#define SPARK_K3_STAGE_RUNNER_FLAG_CAPTURE_GRAPHS 0x00000002u
#define SPARK_K3_STAGE_RUNNER_KNOWN_FLAGS \
    (SPARK_K3_STAGE_RUNNER_FLAG_TENSOR_PARALLEL | \
     SPARK_K3_STAGE_RUNNER_FLAG_CAPTURE_GRAPHS)

typedef struct SparkK3StageRunnerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t stage_index;          /* PP stage 0..3 */
    uint32_t stage_count;          /* 4 */
    uint32_t tp_degree;            /* 4; 1 disables the collectives */
    uint32_t tp_rank;
    uint32_t max_active_sequence_count;
    uint32_t max_input_row_count;
    uint32_t resident_sequence_capacity;
    uint32_t kv_pages_per_sequence;
    uint64_t kv_page_bytes;
    const char *rank_pack_path;
    uint32_t multiprocessors;
    void *execution_stream;        /* cudaStream_t */
    /* Non-null when tp_degree > 1: the ring collective's step topology.
     * The runner owns no connection state outside this config; the serving
     * adapter fills it from its configuration file. */
    const SparkTpCollectiveConfig *tp_collective;
    /* Optional DEVICE-DIRECT tier: when non-null the per-layer hook packs
     * each phase's sharded projections and submits a stream-ordered combine
     * through SparkTpDeviceCollectiveSubmitBf16 (no syncs, no host
     * staging); the host TCP tier above remains the fallback. */
    const SparkTpDeviceCollectiveConfig *device_collective;
    /* Diagnostic override: when set, the slice's layer_collective hook is
     * THIS callback instead of the TP all-reduce (even at tp_degree 1), so
     * a test can observe the stream per layer on the exact serving path.
     * phase 0 = after the attention half, phase 1 = after the MLP half. */
    void (*layer_collective_override)(void *context, void *stream,
        uint32_t layer, uint32_t phase);
    void *layer_collective_context;
} SparkK3StageRunnerConfiguration;

typedef struct SparkK3StageRunnerDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t deadline_time_ns;
    uint32_t row_count;
    uint32_t active_sequence_count;
    /* Stage 0 consumes token_ids and ignores the hidden input; stages 1..3
     * consume hidden_input_bf16 and ignore token_ids. */
    const uint32_t *token_ids;
    const uint32_t *positions;         /* rows, device */
    const uint32_t *context_length;    /* rows, device */
    const uint32_t *sequence_of_row;   /* rows, device */
    const uint32_t *kda_state_index;   /* sequences, device */
    /* The recurrence kernels' run prefix: active_sequence_count+1 entries,
     * device, runs[0]=0, runs[s] < runs[s+1] monotone, runs[active] ==
     * row_count. Consecutive rows runs[s]..runs[s+1)-1 of one sequence
     * chain through the KDA state in row order; a NULL prefix means row i
     * IS sequence i (pure decode, run of one per row). The bit-exactness
     * contract is "a run of T is identical to T one-row waves". */
    const uint32_t *sequence_row_begin;
    const void *hidden_input_bf16;
    uint64_t hidden_input_bytes;
    void *hidden_output_bf16;
    uint64_t hidden_output_bytes;
    /* The head stage writes the committed tokens/scores here. */
    uint32_t *output_token_ids;
    float *output_scores;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
} SparkK3StageRunnerDispatch;

typedef struct SparkK3StageRunnerStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t last_status;
    uint64_t submitted_count;
    uint64_t completed_count;
    uint64_t failed_count;
    uint64_t collective_count;
} SparkK3StageRunnerStats;

typedef struct SparkK3StageRunner
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t stage_index;
    uint32_t stage_count;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t owns_embedding;
    uint32_t owns_final_head;
    void *private_state;
    SparkK3StageRunnerStats stats;
} SparkK3StageRunner;

SparkStatus SparkK3StageRunnerInitialize(
    SparkK3StageRunner *runner,
    const SparkK3StageRunnerConfiguration *configuration);

SparkStatus SparkK3StageRunnerSubmit(
    SparkK3StageRunner *runner,
    const SparkK3StageRunnerDispatch *dispatch);

SparkStatus SparkK3StageRunnerGetStats(
    const SparkK3StageRunner *runner,
    SparkK3StageRunnerStats *stats_out);

void SparkK3StageRunnerDestroy(SparkK3StageRunner *runner);

/* Diagnostic: the host-side buffers struct (device pointers inside), for
 * probe hooks and tests that observe the stream per layer. */
const void *SparkK3StageRunnerProbeBuffers(const SparkK3StageRunner *runner);

/* Serial-TP half step (docs/serial_tp_replay.md): run ONE layer's attention
 * half (phase 0) or MLP half (phase 1) with the FULL hidden and the FULL
 * AttnRes partial replicated in, and the rank's partial copied out. The
 * harness loops layers and halves, host-summing the rank partials between
 * halves. rows is the single decode token (the replay is B1). */
SparkStatus SparkK3StageRunnerStepHalf(
    SparkK3StageRunner *runner,
    uint32_t layer,                 /* absolute layer index in the pack slice */
    uint32_t phase,                 /* 0 = attention half, 1 = MLP half */
    const void *hidden_input_bf16,  /* full hidden, device (phase 0) */
    const void *partial_input_bf16, /* full AttnRes partial, device (may be NULL) */
    void *partial_output_bf16);     /* rank's partial, device, K3_HIDDEN */

#ifdef __cplusplus
}
#endif

#endif
