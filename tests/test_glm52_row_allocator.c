#include "sparkpipe/spark_glm52_row_allocator.h"

#include <assert.h>
#include <string.h>

// A saturated wave converges to all-real: with as many slots as rows, every
// slot gets its base row and zero draft rows, regardless of how strong the
// acceptance signal is. Speculation never displaces a queued real row.
static void SparkTestRowAllocatorSaturatedWaveIsAllReal(void)
{
    SparkGlm52RowAllocatorSlotInput slots[8];
    uint32_t budgets[8];
    uint32_t slot_index,total;
    memset(slots, 0, sizeof(slots));
    for (slot_index = 0u; slot_index < 8u; ++slot_index)
    {
        slots[slot_index].commit_ema_milli = 2900u;
        slots[slot_index].maximum_draft_depth = 5u;
    }
    total = SparkGlm52RowAllocatorAssign(slots, 8u, 8u, budgets);
    assert(total == 8u);
    for (slot_index = 0u; slot_index < 8u; ++slot_index)
        assert(budgets[slot_index] == 0u);
}

// An undersubscribed wave fills with speculation, deepest on the strongest
// lane, and the total never exceeds the cap.
static void SparkTestRowAllocatorUndersubscribedFillsByAlpha(void)
{
    SparkGlm52RowAllocatorSlotInput slots[2];
    uint32_t budgets[2];
    uint32_t total;
    memset(slots, 0, sizeof(slots));
    slots[0].commit_ema_milli = 2900u;
    slots[0].maximum_draft_depth = 5u;
    slots[1].commit_ema_milli = 1300u;
    slots[1].maximum_draft_depth = 5u;
    total = SparkGlm52RowAllocatorAssign(slots, 2u, 8u, budgets);
    assert(total <= 8u);
    assert(budgets[0] > budgets[1]);
    assert(budgets[0] == 5u);
    // slot 1 alpha ~ 231 milli: depth-1 value 231 beats nothing after slot 0's
    // depth-2 (~438) but wins over slot 0 depth-4/5 tail? Verify exact greedy:
    // slot0 alpha ~655: values 655, 429, 281, 184, 120. slot1: 231, 53, ...
    // grant order: 655, 429, 281, 231, 184, 120 -> six grants but only
    // 8 - 2 = 6 remain: slot0 gets 5, slot1 gets 1.
    assert(budgets[1] == 1u);
    assert(total == 8u);
}

// Suppressed slots (maximum depth zero) receive only their base row even when
// capacity is abundant.
static void SparkTestRowAllocatorSuppressedSlotStaysPlain(void)
{
    SparkGlm52RowAllocatorSlotInput slots[2];
    uint32_t budgets[2];
    uint32_t total;
    memset(slots, 0, sizeof(slots));
    slots[0].commit_ema_milli = 900u;
    slots[0].maximum_draft_depth = 0u;
    slots[1].commit_ema_milli = 2900u;
    slots[1].maximum_draft_depth = 5u;
    total = SparkGlm52RowAllocatorAssign(slots, 2u, 16u, budgets);
    assert(budgets[0] == 0u);
    assert(budgets[1] == 5u);
    assert(total == 7u);
}

// A probing slot is guaranteed its two-row probe before value-ranked grants and
// competes no further; the probe is capacity-bounded.
static void SparkTestRowAllocatorProbeGrant(void)
{
    SparkGlm52RowAllocatorSlotInput slots[2];
    uint32_t budgets[2];
    uint32_t total;
    memset(slots, 0, sizeof(slots));
    slots[0].commit_ema_milli = 1000u;
    slots[0].maximum_draft_depth = 5u;
    slots[0].probe = 1u;
    slots[1].commit_ema_milli = 2900u;
    slots[1].maximum_draft_depth = 5u;
    total = SparkGlm52RowAllocatorAssign(slots, 2u, 16u, budgets);
    assert(budgets[0] == 2u);
    assert(budgets[1] == 5u);
    assert(total == 9u);
    // Tight capacity: base rows consume 2 of 3; the single remaining row goes
    // to the probe, which precedes value grants.
    total = SparkGlm52RowAllocatorAssign(slots, 2u, 3u, budgets);
    assert(budgets[0] == 1u);
    assert(budgets[1] == 0u);
    assert(total == 3u);
}

// EMA at or below one committed token per cycle yields no speculative grants:
// drafting has demonstrated no value on that lane.
static void SparkTestRowAllocatorNoValueNoSpec(void)
{
    SparkGlm52RowAllocatorSlotInput slots[1];
    uint32_t budgets[1];
    uint32_t total;
    memset(slots, 0, sizeof(slots));
    slots[0].commit_ema_milli = 1000u;
    slots[0].maximum_draft_depth = 5u;
    total = SparkGlm52RowAllocatorAssign(slots, 1u, 16u, budgets);
    assert(budgets[0] == 0u);
    assert(total == 1u);
}

// Determinism and tie breaking: equal-alpha slots receive grants in lower
// slot-index order, so repeated runs produce identical assignments.
static void SparkTestRowAllocatorDeterministicTies(void)
{
    SparkGlm52RowAllocatorSlotInput slots[3];
    uint32_t budgets[3];
    uint32_t reference[3];
    uint32_t slot_index,total,run;
    memset(slots, 0, sizeof(slots));
    for (slot_index = 0u; slot_index < 3u; ++slot_index)
    {
        slots[slot_index].commit_ema_milli = 2000u;
        slots[slot_index].maximum_draft_depth = 5u;
    }
    total = SparkGlm52RowAllocatorAssign(slots, 3u, 5u, budgets);
    assert(total == 5u);
    assert(budgets[0] == 1u && budgets[1] == 1u && budgets[2] == 0u);
    memcpy(reference, budgets, sizeof(reference));
    for (run = 0u; run < 4u; ++run)
    {
        total = SparkGlm52RowAllocatorAssign(slots, 3u, 5u, budgets);
        assert(total == 5u);
        assert(memcmp(reference, budgets, sizeof(reference)) == 0);
    }
}

// More active slots than the cap: the surplus receives nothing and the total
// equals the cap exactly; admission upstream owns the overflow.
static void SparkTestRowAllocatorOverflowClampsToCap(void)
{
    SparkGlm52RowAllocatorSlotInput slots[4];
    uint32_t budgets[4];
    uint32_t slot_index,total;
    memset(slots, 0, sizeof(slots));
    for (slot_index = 0u; slot_index < 4u; ++slot_index)
    {
        slots[slot_index].commit_ema_milli = 2900u;
        slots[slot_index].maximum_draft_depth = 5u;
    }
    total = SparkGlm52RowAllocatorAssign(slots, 4u, 2u, budgets);
    assert(total == 2u);
    for (slot_index = 0u; slot_index < 4u; ++slot_index)
        assert(budgets[slot_index] == 0u);
}

int main(void)
{
    SparkTestRowAllocatorSaturatedWaveIsAllReal();
    SparkTestRowAllocatorUndersubscribedFillsByAlpha();
    SparkTestRowAllocatorSuppressedSlotStaysPlain();
    SparkTestRowAllocatorProbeGrant();
    SparkTestRowAllocatorNoValueNoSpec();
    SparkTestRowAllocatorDeterministicTies();
    SparkTestRowAllocatorOverflowClampsToCap();
    return 0;
}
