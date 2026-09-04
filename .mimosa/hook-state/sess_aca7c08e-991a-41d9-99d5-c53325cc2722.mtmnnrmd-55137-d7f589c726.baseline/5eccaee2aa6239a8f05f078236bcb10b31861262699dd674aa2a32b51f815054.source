#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "runtime/arena.h"

static int32_t SparkTestArenaCycles(void)
{
    SparkArena arena;
    SparkArenaClassDescriptor classes[2];
    SparkArenaAllocation first[3];
    SparkArenaAllocation second;
    SparkArenaAllocation third;
    SparkArenaAllocation replacement;
    SparkArenaAllocation stale;
    uint32_t index;

    classes[0u].slot_bytes = 100u;
    classes[0u].slot_count = 3u;
    classes[1u].slot_bytes = 4096u;
    classes[1u].slot_count = 2u;
    assert(SparkArenaInitialize(&arena, classes, 2u) == 0);
    assert(SparkArenaReservedBytes(&arena) ==
        3ull * 112ull + 2ull * 4096ull +
            (3ull + 2ull) * (uint64_t)sizeof(uint32_t) + 4ull +
            (3ull + 2ull) * (uint64_t)sizeof(uint64_t));

    for (index = 0u; index < 3u; ++index)
    {
        assert(SparkArenaAcquire(
            &arena,
            (uint32_t)(index + 1u),
            &first[index]) == 0);
        assert(first[index].pointer != 0);
        assert(((uintptr_t)first[index].pointer % SPARK_ARENA_ALIGNMENT) == 0u);
        memset(
            first[index].pointer,
            (int)(index + 1u),
            (size_t)(index + 1u));
    }

    assert(SparkArenaAcquire(&arena, 1u, &replacement) == -30913);
    assert(SparkArenaClassGeneration(&arena, 0u) == 3u);

    assert(SparkArenaAcquire(&arena, 4096u, &second) == 0);
    assert(SparkArenaAcquire(&arena, 2000u, &third) == 0);
    assert(second.pointer != 0 && third.pointer != 0);
    assert(((uintptr_t)second.pointer % SPARK_ARENA_ALIGNMENT) == 0u);
    assert((uint8_t *)third.pointer >= (uint8_t *)second.pointer + 4096u ||
        (uint8_t *)second.pointer >= (uint8_t *)third.pointer + 4096u);
    assert(SparkArenaAcquire(&arena, 4096u, &replacement) == -30913);
    assert(SparkArenaAcquire(&arena, 8192u, &replacement) == -30915);

    stale = first[1u];
    assert(SparkArenaRelease(&arena, &first[1u]) == 0);
    assert(SparkArenaAcquire(&arena, 64u, &replacement) == 0);
    assert(replacement.pointer == stale.pointer);
    assert(replacement.generation != stale.generation);
    assert(SparkArenaRelease(&arena, &stale) == -30917);
    assert(SparkArenaRelease(&arena, &replacement) == 0);
    assert(SparkArenaClassGeneration(&arena, 0u) == 4u);
    assert(*(uint8_t *)first[0u].pointer == 1u);

    assert(SparkArenaRelease(&arena, &second) == 0);
    assert(SparkArenaRelease(&arena, &second) == -30908);

    stale = third;
    stale.pointer = (uint8_t *)third.pointer + 16u;
    assert(SparkArenaRelease(&arena, &stale) == -30907);

    stale = third;
    stale.class_index = 7u;
    assert(SparkArenaRelease(&arena, &stale) == -30909);

    SparkArenaDestroy(&arena);
    assert(arena.backing == 0);
    SparkArenaDestroy(&arena);
    return 0;
}

static int32_t SparkTestArenaRejectsBadGeometry(void)
{
    SparkArena arena;
    SparkArenaClassDescriptor classes[2];

    memset(&arena, 0, sizeof(arena));
    classes[0u].slot_bytes = 4096u;
    classes[0u].slot_count = 1u;
    classes[1u].slot_bytes = 64u;
    classes[1u].slot_count = 1u;
    assert(SparkArenaInitialize(&arena, classes, 2u) == -30903);
    assert(arena.backing == 0);

    classes[0u].slot_bytes = 17u;
    classes[0u].slot_count = 1u;
    classes[1u].slot_bytes = 31u;
    classes[1u].slot_count = 1u;
    assert(SparkArenaInitialize(&arena, classes, 2u) == -30903);
    assert(arena.backing == 0);

    classes[0u].slot_bytes = 64u;
    classes[0u].slot_count = 0u;
    assert(SparkArenaInitialize(&arena, classes, 1u) == -30902);

    classes[0u].slot_bytes = 0xffffffffu;
    classes[0u].slot_count = 0xffffffffu;
    assert(SparkArenaInitialize(&arena, classes, 1u) == -30904);
    assert(arena.backing == 0);

    SparkArenaDestroy(&arena);
    return 0;
}


static int32_t SparkTestArenaGenerationExhaustion(void)
{
    SparkArena arena;
    SparkArenaClassDescriptor arena_class;
    SparkArenaAllocation allocation;
    uint32_t free_count;

    arena_class.slot_bytes = 64u;
    arena_class.slot_count = 1u;
    assert(SparkArenaInitialize(&arena, &arena_class, 1u) == 0);
    arena.classes[0u].slot_generations[0u] = UINT64_MAX;
    free_count = arena.classes[0u].free_count;
    assert(SparkArenaAcquire(&arena, 64u, &allocation) == -30918);
    assert(arena.classes[0u].free_count == free_count);
    assert(arena.classes[0u].free_head == 0u);
    SparkArenaDestroy(&arena);
    return 0;
}

int main(void)
{
    SparkTestArenaCycles();
    SparkTestArenaRejectsBadGeometry();
    SparkTestArenaGenerationExhaustion();
    printf("arena: generation handles, alignment, loud exhaustion, geometry\n");
    return 0;
}
