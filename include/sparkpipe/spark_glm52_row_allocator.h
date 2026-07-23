#ifndef SPARKPIPE_SPARK_GLM52_ROW_ALLOCATOR_H
#define SPARKPIPE_SPARK_GLM52_ROW_ALLOCATOR_H

#include <stdint.h>

// Global firing-row allocator: divides a wave's row budget between real rows
// (one per active slot, marginal expected commit exactly one token) and
// speculative draft rows (marginal expected commit alpha^depth, the chain
// survival probability). Real rows always dominate speculative rows of equal
// cost, so every active slot receives its base row first; remaining capacity
// goes to draft rows in descending marginal value, which makes the plane
// self-balancing: a saturated wave converges to all-real and an undersubscribed
// wave fills with deep speculation on the lanes whose measured acceptance
// justifies it. Per-slot alpha is derived from the commit EMA the request API
// already maintains: for chain drafting the expected committed tokens per cycle
// is the geometric sum (1-alpha^(k+1))/(1-alpha), and alpha = (ema-1000)/ema in
// milli fixed point inverts it to within a few parts per thousand across the
// operating range (exact as k grows, and the error vanishes as alpha falls).

typedef struct SparkGlm52RowAllocatorSlotInput
{
    // EMA of committed tokens per verify cycle, times 1000 (the request API's
    // mtp_commit_ema_milli). Values at or below 1000 imply no speculative value.
    uint32_t commit_ema_milli;
    // Maximum draft rows this slot may receive. Zero for suppressed slots whose
    // reprobe countdown has not elapsed; they receive only their base row.
    uint32_t maximum_draft_depth;
    // Nonzero when the slot's reprobe countdown has elapsed: the slot is
    // guaranteed a two-row probe grant (capacity permitting) so the EMA can
    // observe beyond-first acceptance, and competes no further that wave.
    uint32_t probe;
} SparkGlm52RowAllocatorSlotInput;

// Fills draft_budgets_out[slot] with the draft rows granted to each slot and
// returns the total rows assigned including base rows. Base rows are granted in
// slot-index order up to the cap; probe grants next; remaining capacity is
// granted greedily by marginal value with deterministic lower-index tie
// breaking. All buffers are caller owned; the function allocates nothing.
uint32_t SparkGlm52RowAllocatorAssign(
    const SparkGlm52RowAllocatorSlotInput *slots,
    uint32_t slot_count,
    uint32_t firing_row_cap,
    uint32_t *draft_budgets_out);

#endif
