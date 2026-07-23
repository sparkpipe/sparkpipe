#include "sparkpipe/spark_glm52_row_allocator.h"

#define SPARK_GLM52_ROW_ALLOCATOR_MILLI 1000u
#define SPARK_GLM52_ROW_ALLOCATOR_ALPHA_CEILING 999u
#define SPARK_GLM52_ROW_ALLOCATOR_PROBE_ROWS 2u

// alpha = (ema - 1000) / ema in milli fixed point, clamped to [0, 999]. An EMA
// at or below one committed token per cycle means drafting has shown no value,
// so alpha is zero and the slot's speculative rows never win a grant.
static uint32_t SparkGlm52RowAllocatorAlphaMilli(uint32_t commit_ema_milli)
{
    uint32_t alpha_milli;
    if (commit_ema_milli <= SPARK_GLM52_ROW_ALLOCATOR_MILLI)
        return 0u;
    alpha_milli = (uint32_t)(((uint64_t)(commit_ema_milli - SPARK_GLM52_ROW_ALLOCATOR_MILLI) *
        SPARK_GLM52_ROW_ALLOCATOR_MILLI) / commit_ema_milli);
    if (alpha_milli > SPARK_GLM52_ROW_ALLOCATOR_ALPHA_CEILING)
        alpha_milli = SPARK_GLM52_ROW_ALLOCATOR_ALPHA_CEILING;
    return alpha_milli;
}

// Grant probing slots their fixed two-row probe, in slot-index order, so the
// EMA can observe beyond-first acceptance at minimal cost. Probing slots do not
// compete in the value-ranked phase.
static uint32_t SparkGlm52RowAllocatorGrantProbes(const SparkGlm52RowAllocatorSlotInput *slots,uint32_t slot_count,uint32_t remaining,uint32_t *draft_budgets_out)
{
    uint32_t slot_index,grant;
    for (slot_index = 0u; slot_index < slot_count && remaining != 0u; ++slot_index)
    {
        if (slots[slot_index].probe == 0u || slots[slot_index].maximum_draft_depth == 0u)
            continue;
        grant = SPARK_GLM52_ROW_ALLOCATOR_PROBE_ROWS;
        if (grant > slots[slot_index].maximum_draft_depth)
            grant = slots[slot_index].maximum_draft_depth;
        if (grant > remaining)
            grant = remaining;
        draft_budgets_out[slot_index] = grant;
        remaining -= grant;
    }
    return remaining;
}

// Exact greedy: repeatedly grant the single highest-marginal-value draft row
// among non-probing slots, lower slot index winning ties, until capacity or
// value is exhausted. The marginal value of a slot's next draft row at depth d
// is alpha^d in milli fixed point, maintained incrementally per slot. The scan
// is O(grants x slots); at wave cadence with bounded depth this is negligible,
// and a threshold binary search is the drop-in replacement if it ever profiles.
static void SparkGlm52RowAllocatorGreedyGrant(const SparkGlm52RowAllocatorSlotInput *slots,uint32_t slot_count,uint32_t remaining,uint32_t *draft_budgets_out)
{
    uint32_t slot_index,best_index,best_value;
    uint64_t next_value;
    // marginal_value[i] tracks alpha_i^(granted_i + 1); recomputed incrementally.
    // Stored in the output array's mirror via local recompute to avoid scratch
    // allocations: value_i = alpha_i^(draft_budgets_out[i] + 1).
    while (remaining != 0u)
    {
        best_index = slot_count;
        best_value = 0u;
        for (slot_index = 0u; slot_index < slot_count; ++slot_index)
        {
            uint32_t alpha_milli,granted,depth,value;
            if (slots[slot_index].probe != 0u)
                continue;
            granted = draft_budgets_out[slot_index];
            if (granted >= slots[slot_index].maximum_draft_depth)
                continue;
            alpha_milli = SparkGlm52RowAllocatorAlphaMilli(slots[slot_index].commit_ema_milli);
            if (alpha_milli == 0u)
                continue;
            value = alpha_milli;
            for (depth = 0u; depth < granted; ++depth)
            {
                next_value = ((uint64_t)value * alpha_milli) / SPARK_GLM52_ROW_ALLOCATOR_MILLI;
                value = (uint32_t)next_value;
            }
            if (value > best_value)
            {
                best_value = value;
                best_index = slot_index;
            }
        }
        if (best_index == slot_count || best_value == 0u)
            break;
        draft_budgets_out[best_index] += 1u;
        remaining -= 1u;
    }
}

uint32_t SparkGlm52RowAllocatorAssign(
    const SparkGlm52RowAllocatorSlotInput *slots,
    uint32_t slot_count,
    uint32_t firing_row_cap,
    uint32_t *draft_budgets_out)
{
    uint32_t slot_index,base_rows,remaining,total;
    if (slots == 0 || draft_budgets_out == 0 || firing_row_cap == 0u)
        return 0u;
    for (slot_index = 0u; slot_index < slot_count; ++slot_index)
        draft_budgets_out[slot_index] = 0u;
    // Base rows first: a real row's marginal expected commit is one token, which
    // no speculative row can match, so speculation never displaces a queued
    // real row. If more slots are active than the cap admits, the surplus slots
    // receive nothing this wave; admission control upstream owns that case.
    base_rows = slot_count;
    if (base_rows > firing_row_cap)
        base_rows = firing_row_cap;
    remaining = firing_row_cap - base_rows;
    remaining = SparkGlm52RowAllocatorGrantProbes(slots, slot_count, remaining, draft_budgets_out);
    SparkGlm52RowAllocatorGreedyGrant(slots, slot_count, remaining, draft_budgets_out);
    total = base_rows;
    for (slot_index = 0u; slot_index < slot_count; ++slot_index)
        total += draft_budgets_out[slot_index];
    return total;
}
