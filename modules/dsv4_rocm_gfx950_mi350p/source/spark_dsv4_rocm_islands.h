#ifndef SPARKPIPE_SPARK_DSV4_ROCM_ISLANDS_H
#define SPARKPIPE_SPARK_DSV4_ROCM_ISLANDS_H

/* spark_dsv4_rocm_islands.h - island entry surface of the
 * dsv4_rocm_gfx950_mi350p archive (hwiface_v1.md section 6 link unit).
 *
 * Scope: the S7 bring-up order steps 1 and 3 from
 * docs/coord/plan_amd_gfx950_mi350p.md section 5.2 - step 1 "E0 + L5
 * first" (integer-exact E0, simplest C3 accumulation island L5) and step 3
 * "L1 then L4": L4 `layer.moe_routed`, driven by synthetic sealed-route
 * batches over slot-workspace views at TP1 scope. Later islands land beside
 * these as separate entries; nothing here changes when they do.
 *
 * Naming note on "L3": the frozen contract's L3 is `layer.cache_transition`
 * (hwiface_v1.md section 3.1). Earlier notes in this file explained that the
 * FFN down projection asked about under the "L3" label lives in L4/L5; that
 * landed with the L4 island. THIS file now also declares the real L3:
 * three symbols share its frozen id+name prefix because their lifecycles are
 * disjoint (per-frame transition vs allocation-time page setup vs page-table
 * commit) while remaining one semantic island (R1: kernel count is free).
 *
 * Contract basis (freeze §F1, hwiface_v1_freeze.md):
 *
 *   | #  | island                 | declared inputs              | declared outputs               | handles                        | class |
 *   | E0 | prologue.embed         | token ids, embedding view    | boundary packet (hc_mult x     | queue, memory                  | C2    |
 *   |    |                        |                              | hidden bf16)                   |                                |       |
 *   | L5 | layer.moe_shared       | normalized hidden            | FFN accumulator (partial)      | aux queue, fork/join event     | C3    |
 *   |    |                        |                              |                                | pair, memory                   |       |
 *
 * Naming note for reviewers: the frozen contract's embedding island is E0
 * `prologue.embed` and its head island is F1 `head.final`; L1 is
 * `layer.boundary_norm_project` and L5 is `layer.moe_shared`. These entries
 * carry both the frozen id and the frozen name in the symbol so no mapping
 * table is ever needed at link time.
 *
 * Linkage: plain externs with C linkage (plan section 3.1). Missing symbol =
 * link error (R2); no dispatch tables anywhere. Entries are geometry-free:
 * every dimension arrives as an argument, mirroring the stage firmware rule
 * that views carry no model constants and kernels receive budgets as
 * arguments (plan section 8 risk table).
 *
 * Determinism:
 *   - E0 is C2: a pure byte gather + stream expand; bit-exactness against
 *     CUDA holds by construction because bytes are moved, never re-rounded.
 *   - L5 is C3: one fixed reduction tree for EVERY shape bucket (B3) -
 *     per-lane sequential fp32 k-stride accumulation, then a single
 *     log2(64)-step shuffle-down butterfly over the 64-lane wavefront,
 *     then one round-to-nearest bf16 store. Documented once, never tuned
 *     per shape; tolerance authority stays with the shared recipe (Q4).
 */

#include <stdint.h>

#include <sparkpipe/spark_hw_iface.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Weight format codes. Values mirror the shared logical inventory enum
 * (SPARK_LM_WEIGHT_FORMAT_*, lm_kernels.cuh:58) so packs bind without a
 * translation table, but this header is target-owned: it deliberately does
 * NOT include spark_lm_kernels.cuh (freeze F4 landmine rule - that header
 * errors out under any non-cuda.sm121.gb10 target id). Only formats the v1
 * islands consume are named here.
 */
#define SPARK_DSV4_ROCM_WEIGHT_FORMAT_BF16      0u
#define SPARK_DSV4_ROCM_WEIGHT_FORMAT_MXFP4_E2M1 3u
#define SPARK_DSV4_ROCM_WEIGHT_FORMAT_FP8_E4M3  4u

/* Target-private linear weight view. Field-for-field mirror of the portable
 * SparkDsv4LinearView shape (firmware.h:139) minus abi_version: the logical
 * weight inventory is shared (boundary row 8) while packed layout and dequant
 * path are target-private below the seam. payload/scale_data are DEVICE
 * pointers. Row-major packed layout, stacked-expert rows addressed as
 * expert*per_expert_rows + row. scale_data semantics per format:
 *   BF16       - NULL, scale-free.
 *   FP8_E4M3   - one E8M0 byte per scale_group elements per row
 *                (stagepack rule: fp8 block 128 columns).
 *   MXFP4_E2M1 - one E8M0 byte per scale_group elements per row (stagepack
 *                rule: fp4 experts block 32); payload nibbles are E2M1,
 *                low nibble first within each little-endian byte, matching
 *                the shared machinery's word decode exactly. scale_group
 *                must be a multiple of 8 so an 8-element run never crosses
 *                a group boundary (shared decode invariant).
 */
typedef struct SparkDsv4RocmLinearView
{
    uint32_t weight_format;
    uint32_t rows;
    uint32_t columns;
    uint32_t scale_group;   /* elements per E8M0 scale byte; 0 = none */
    const void *payload;
    const void *scale_data;
} SparkDsv4RocmLinearView;

/* E0 `prologue.embed` - C2 (integer exact).
 *
 * Gathers one embedding row per token id and expands it into hc_mult
 * identical hyper-connection streams, producing row_count boundary packet
 * rows of hc_mult x hidden bf16 elements (firmware.h:26 - mHC never
 * collapses between layers; stage 0 is where the four identical streams are
 * born). Bytes are copied verbatim: no conversion round trip touches the
 * payload, which is what makes the C2 half trivially exact.
 *
 * queue          required; every kernel enqueues here, completion stays
 *                stream-ordered via the primitive family callback (never a
 *                sync on the success path).
 * memory         optional in v1: the gather needs no workspace; validated
 *                when provided, kept in the signature because freeze F1
 *                declares the handle for this island.
 * geometry       hidden_dimension must be a multiple of 8 (pack rule: model
 *                dimensions are multiples of 64) so every gather row base
 *                stays 16-byte aligned; the boundary_packet device buffer
 *                must be 8-byte aligned. Refused with SPARK_HW_INVALID.
 * read_ahead_*   optional E0 weight read-ahead start (guide section 4 E0:
 *                "kick your spark_hw_read_ahead equivalent here"). When
 *                read_ahead_sink_u32 is non-NULL the entry calls
 *                spark_hw_read_ahead(queue, sink, capacity, context) - the
 *                three-clause no-op-safe kick contract of section 4.2.
 *                Contents stay target-defined; correctness never depends
 *                on the kick.
 *
 * Returns SPARK_HW_OK once every launch/enqueue was ACCEPTED (stream-ordered
 * work may still be in flight); INVALID on any bad argument; the status
 * model of spark_hw_iface.h governs everything else.
 */
SparkHwStatus spark_dsv4_rocm_e0_prologue_embed(
    SparkHwQueue queue,
    SparkHwMemory memory,
    const uint32_t *token_ids_device,
    const void *embedding_bf16_device,
    void *boundary_packet_bf16_device,
    uint32_t row_count,
    uint32_t hidden_dimension,
    uint32_t hc_mult,
    uint32_t vocab_size,
    void *read_ahead_sink_u32,
    uint32_t read_ahead_word_capacity,
    void *read_ahead_context);

/* L5 `layer.moe_shared` - C3 (bounded identity), simplest accumulation
 * island; the plan's tolerance-plumbing shakeout before attention arrives.
 *
 * Shared-expert FFN leg over normalized hidden, DSV4 swiglu semantics
 * mirrored from the CUDA shared path (spark_dsv4_resident_decode_stage_cuda.cu
 * :1437): up clamped two-sided to +/-limit, gate clamped one-sided to
 * +limit, activated = swish(gate) * up computed in fp32 and stored bf16
 * round-to-nearest-even; limit <= 0 disables clamping exactly like the CUDA
 * kernel. Projections accumulate in fp32 through the fixed tree documented
 * in the header banner.
 *
 * aux_queue      required; the fork leg runs here concurrently with L4.
 * fork_event     optional: when non-NULL, hipStreamWaitEvent orders this leg
 *                after the producer (the L4 fork point) before any launch.
 * join_event     optional: when non-NULL, recorded on aux_queue after the
 *                final enqueue so the pre-hcPost join can wait it. Both may
 *                be NULL during TP1 bring-up before L4 exists (plan 5.2
 *                step 3 drives L4 later); the events themselves are created
 *                timing-disabled per the primitive family.
 * w1/w3          gate/up projections: rows = moe_intermediate_dimension,
 *                columns = hidden_dimension, matching formats.
 * w2             down projection: rows = hidden_dimension,
 *                columns = moe_intermediate_dimension, matching format.
 * ffn_accum_bf16 device INOUT: the island ADDS its contribution into this
 *                accumulator in fp32 per element with one final RNE bf16
 *                store - the "FFN accumulator (partial)" output joined
 *                before block hcPost. Callers zero it at slot creation;
 *                add order across legs is fixed by queue/event order per
 *                bucket (C3 stable-tree requirement).
 *
 * Scratch (gate/up activations) is caller-provided slot-workspace device
 * memory sized row_count x 2 x moe_intermediate_dimension bf16, passed as
 * gate_up_scratch_bf16_device - fixed-offset slot workspace, never dynamic
 * allocation (freeze F2 discipline).
 */
SparkHwStatus spark_dsv4_rocm_l5_moe_shared(
    SparkHwQueue aux_queue,
    SparkHwEvent fork_event,
    SparkHwEvent join_event,
    SparkHwMemory memory,
    const void *normalized_hidden_bf16_device,
    const SparkDsv4RocmLinearView *w1_gate,
    const SparkDsv4RocmLinearView *w3_up,
    const SparkDsv4RocmLinearView *w2_down,
    float swiglu_limit,
    void *gate_up_scratch_bf16_device,
    void *ffn_accum_bf16_device,
    uint32_t row_count,
    uint32_t hidden_dimension,
    uint32_t moe_intermediate_dimension);

/* L4 `layer.moe_routed` - split verdict (freeze section F1): route
 * realization C2 integer-exact, grouped-GEMM output C3.
 *
 * Aggregation split discipline (freeze section 3.4 / rule 5): the logical
 * route is CORE-computed and arrives here as the sealed-route batch content
 * - per-token top-k expert indices and normalized routing weights over the
 * slot-workspace views. This entry realizes it: deterministic grouping
 * (counting, exclusive scan, scatter), stacked-expert grouped W13 gate/up
 * projections with clamped swiglu, the grouped W2 down projection (the FFN
 * down projection for the routed leg), and the weighted pair reduce that
 * accumulates into the shared FFN accumulator before the block hcPost join.
 * No routing policy crosses the seam in either direction; nothing is
 * dynamically allocated - every scratch view below is caller-provided
 * fixed-offset slot workspace sized by the shape bucket (freeze F2).
 *
 * queue           required; every kernel enqueues stream-ordered here.
 * fork_event      optional: hipStreamWaitEvent orders this leg after the
 *                 producer fork point before any launch.
 * join_event      optional: recorded after the final enqueue so the pre-
 *                 hcPost consumer can wait it (L5 joins this same point).
 * route_indices   device uint32[row_count x experts_per_token]; every value
 *                 MUST be < expert_count - core-sealed routes are
 *                 well-formed by construction (core owns admission policy;
 *                 the CUDA target trusts them identically). An out-of-range
 *                 index is a core contract violation and is not guarded
 *                 here beyond refusing the empty expert_count case.
 * route_weights   device float[row_count x experts_per_token], applied at
 *                 the pair reduce (CUDA parity: unweighted down outputs,
 *                 weights folded once at combine).
 * w13_gate/up     stacked experts: rows = expert_count x
 *                 moe_intermediate_dimension, columns = hidden_dimension,
 *                 matching formats.
 * w2_down         stacked experts: rows = expert_count x hidden_dimension,
 *                 columns = moe_intermediate_dimension, matching format.
 * swiglu_limit    identical clamp semantics to L5; <=0 disables.
 * Workspace views (device, caller-owned slot workspace):
 *   grouped_source_token_u32  pairs -> source row (scatter result)
 *   grouped_expert_u32        pairs -> expert id
 *   expert_offsets_u32        expert_count+1 entries: exclusive prefix of
 *                             per-expert pair counts, [expert_count] =
 *                             total pairs. THE declared C2 output buffer
 *                             (freeze section 3.5): canonical ascending
 *                             expert order, order-independent of scatter.
 *   inverse_pair_u32          flat route element (row*K+k) -> packed pair
 *                             slot; within an expert, pairs are packed in
 *                             ascending flat-route-element order - a FIXED
 *                             deterministic tree, documented once (B3).
 *   up_scratch_bf16           pairs x moe_intermediate_dimension activated
 *                             gate/up values (weight NOT yet folded)
 *   pair_out_scratch_bf16     pairs x hidden_dimension unweighted W2 outputs
 * ffn_accum       device INOUT, AccumAdd semantics like L5.
 *
 * Determinism: grouping phase is a single-CTA counting rank (no atomics in
 * the rank computation), so C2 quantities are run-invariant, not merely
 * usually-stable. GEMM phases reuse the fixed L5 tree per lane subset with
 * one RNE bf16 store each; the reduce sums k in ascending order in fp32.
 */
SparkHwStatus spark_dsv4_rocm_l4_moe_routed(
    SparkHwQueue queue,
    SparkHwEvent fork_event,
    SparkHwEvent join_event,
    SparkHwMemory memory,
    const void *normalized_hidden_bf16_device,
    const uint32_t *route_indices_u32_device,
    const float *route_weights_f32_device,
    const SparkDsv4RocmLinearView *w13_gate,
    const SparkDsv4RocmLinearView *w13_up,
    const SparkDsv4RocmLinearView *w2_down,
    float swiglu_limit,
    void *grouped_source_token_u32_device,
    void *grouped_expert_u32_device,
    void *expert_offsets_u32_device,
    void *inverse_pair_u32_device,
    void *up_scratch_bf16_device,
    void *pair_out_scratch_bf16_device,
    void *ffn_accum_bf16_device,
    uint32_t row_count,
    uint32_t expert_count,
    uint32_t experts_per_token,
    uint32_t hidden_dimension,
    uint32_t moe_intermediate_dimension);

/* L1 `layer.boundary_norm_project` - hcEnter half (C3 with fixed trees).
 *
 * The contract's L1 contents row (hwiface_v1.md section 3.1) names
 * "hcPost fold, hcEnter, RMSNorm, attention input projections (+ TP shard
 * pack/unpack)". The hcEnter half lands here as one entry mirroring the
 * CUDA module's SparkDsv4ModuleHcEnter sequence exactly (module.c :3226):
 *
 *   1. split-K mix dots + sum-of-squares partials over the flattened
 *      boundary packet (stream_count x hidden), then one finalize kernel
 *      that folds the partials on a FIXED tree, applies the flat
 *      rsqrt(mean-square + rms_epsilon) norm, and runs the inference
 *      Sinkhorn (sigmoid pre, doubled sigmoid post, comb row-softmax then
 *      sinkhorn_iterations alternating row/column normalizations with
 *      +hc_epsilon inside every division) - producing mixes/pre/post/comb.
 *   2. pre-reduce: verbatim residual copy (bf16 byte moves, unrounded)
 *      plus the pre[]-weighted stream fold into the normalized hidden row,
 *      one RNE bf16 store per element.
 *
 * Determinism: mix/sum-of-squares partials fold across splits and lanes on
 * ONE fixed tree documented at the kernels below (never re-tuned per
 * bucket); the Sinkhorn walk is serial per row and wave-size independent;
 * the residual copy moves bytes unrounded. The deliberate target deltas,
 * both inside the shared C3 tolerance set, are recorded at the kernels:
 * wave64 lane subsets for the dots, and expf where the CUDA reference uses
 * fast-math __expf in its sigmoid/exp helpers.
 *
 * partials_f32_device is caller-owned slot workspace sized
 * row_count x split_count x (1 + mix_rows) f32 with
 *   mix_rows    = (2 + stream_count) * stream_count
 *   split_count = (stream_count * hidden_dimension) / 256   (must divide)
 * fn_f32_device is mix_rows x (stream_count x hidden) f32 (one mix row
 * per flat packet position); scale3_f32_device is the 3-entry
 * [pre|post|comb] scale triple and base_f32_device the matching
 * 3 x stream_count bias vector. mixes_f32_device is row_count x mix_rows;
 * pre/post are row_count x stream_count; comb_f32_device is row_count x
 * stream_count^2; reduced_hidden takes row_count rows of hidden_dimension
 * bf16 elements, residual takes row_count rows of stream_count*hidden
 * bf16 elements.
 * stream_count must be within [1,4]; every other rule (queue ordering,
 * fail-closed validation, empty-frame behavior) matches the entries above.
 */
SparkHwStatus spark_dsv4_rocm_l1_hc_enter(
    SparkHwQueue queue,
    SparkHwMemory memory,
    const void *streams_bf16_device,
    const float *fn_f32_device,
    const float *scale3_f32_device,
    const float *base_f32_device,
    void *partials_f32_device,
    void *mixes_f32_device,
    void *pre_f32_device,
    void *post_f32_device,
    void *comb_f32_device,
    void *reduced_hidden_bf16_device,
    void *residual_bf16_device,
    uint32_t row_count,
    uint32_t stream_count,
    uint32_t hidden_dimension,
    uint32_t sinkhorn_iterations,
    float rms_epsilon,
    float hc_epsilon);

/* L3 `layer.cache_transition` - split verdict (freeze F1, section 3.5):
 * page addresses, ring indices and emit counters are C2 integer-pure;
 * emitted payloads are a verbatim bf16 copy on the direct path
 * (C2-eligible: copied unrounded) or quantized sim output on the emission
 * path (C3, recipe-time emitted-field table due before S7 acceptance).
 *
 * The three symbols below are ONE island with disjoint lifecycles:
 *
 *   ..._l3_cache_transition    per-frame: compressor state advance,
 *                              emission post-processing, ring scatter.
 *   ..._l3_initialize_pages    at lane page allocation (parent copy +
 *                              -inf score spans).
 *   ..._l3_update_page_table   scattered physical table commits.
 *
 * All three fail closed on bad arguments; every launch is stream-ordered on
 * the queue; the success path never synchronizes.
 */

/* Frame-path transition. Compressor phases run when compress_ratio != 0
 * (HCA/CSA layers); direct_kv_bf16_device adds a plain verbatim kv scatter
 * when non-NULL (sliding-window layers, ratio-0 ring addressing). Both may
 * run in one call. Emitted flags are written for every row when the
 * compressor runs; emit counters increment once per boundary via integer
 * atomics (exact totals - C2); ring_slots_out receives each row's computed
 * slot when non-NULL (explicit C2 observable).
 *
 * emission_scratch_bf16_device is caller-owned fixed-offset slot workspace
 * sized row_count x compress_width bf16 (freeze F2 discipline - the pooled
 * emit rows staged between the compressor and the emission post pass). */
SparkHwStatus spark_dsv4_rocm_l3_cache_transition(
    SparkHwQueue queue,
    SparkHwMemory memory,
    const void *compressor_kv_bf16_device,
    const void *compressor_score_bf16_device,
    const float *ape_f32_device,
    void *compress_kv_state_f32_device,
    void *compress_score_state_f32_device,
    uint64_t state_lane_stride_elements,
    uint32_t compress_ratio,
    uint32_t overlap_factor,
    uint32_t compress_width,
    const void *emission_norm_weight_bf16_device,
    const float *emission_freqs_f32_device,
    const uint64_t *row_emit_positions_u64_device,
    float rms_epsilon,
    uint32_t rope_dimension,
    uint32_t quant_block,
    float quant_format_max,
    uint32_t rotate_enable,
    void *emission_scratch_bf16_device,
    const uint32_t *row_lane_indices_u32_device,
    const uint64_t *row_positions_u64_device,
    uint32_t row_count,
    void *cache_pages_bf16_device,
    uint64_t cache_lane_stride_elements,
    uint64_t compressed_base_slot,
    uint32_t ring_slots,
    const void *direct_kv_bf16_device,
    uint32_t direct_width,
    void *emitted_flags_u32_device,
    void *emit_counters_u32_device,
    void *ring_slots_out_u32_device);

/* Allocation-time new-page init: root pages zero-fill, child pages byte-copy
 * their parent, then declared score spans poison to -inf. C2 byte/integer
 * semantics; page id 0xffffffff is the shared no-page sentinel. */
SparkHwStatus spark_dsv4_rocm_l3_initialize_pages(
    SparkHwQueue queue,
    SparkHwMemory memory,
    void *page_pool_u32_device,
    uint64_t page_stride_words,
    const uint32_t *page_indices_u32_device,
    const uint32_t *parent_page_indices_u32_device,
    uint32_t page_count,
    const uint64_t *score_span_offset_words_u64_device,
    const uint64_t *score_span_element_count_u64_device,
    uint32_t score_span_count);

/* Scattered physical-page-table commit: table[indices[i]] = values[i]. */
SparkHwStatus spark_dsv4_rocm_l3_update_page_table(
    SparkHwQueue queue,
    SparkHwMemory memory,
    void *page_table_u32_device,
    const uint32_t *update_indices_u32_device,
    const uint32_t *update_values_u32_device,
    uint32_t update_count);

#ifdef __cplusplus
}
#endif

#endif /* SPARKPIPE_SPARK_DSV4_ROCM_ISLANDS_H */
