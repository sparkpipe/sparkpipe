#ifndef SPARKPIPE_RUNTIME_ARENA_H
#define SPARKPIPE_RUNTIME_ARENA_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_ARENA_MAX_CLASS_COUNT 8u
#define SPARK_ARENA_ALIGNMENT 16u
#define SPARK_ARENA_NO_SLOT 0xffffffffu
#define SPARK_ARENA_IN_USE 0xfffffffeu

typedef struct SparkArenaClassDescriptor
{
    uint32_t slot_bytes;
    uint32_t slot_count;
} SparkArenaClassDescriptor;

typedef struct SparkArenaAllocation
{
    void *pointer;
    uint64_t generation;
    uint32_t class_index;
    uint32_t slot_index;
    uint32_t usable_bytes;
    uint32_t reserved0;
} SparkArenaAllocation;

typedef struct SparkArenaClass
{
    uint8_t *slots;
    uint32_t *links;
    uint64_t *slot_generations;
    uint32_t slot_bytes;
    uint32_t slot_count;
    uint32_t free_head;
    uint32_t free_count;
    uint64_t generation;
} SparkArenaClass;

typedef struct SparkArena
{
    uint8_t *backing;
    uint64_t backing_bytes;
    SparkArenaClass classes[SPARK_ARENA_MAX_CLASS_COUNT];
    uint32_t class_count;
    uint32_t reserved0;
} SparkArena;

static inline uint64_t SparkArenaRoundSlotBytes(
    uint32_t slot_bytes)
{
    return ((uint64_t)slot_bytes + (SPARK_ARENA_ALIGNMENT - 1u)) &
        ~((uint64_t)SPARK_ARENA_ALIGNMENT - 1u);
}

static inline uint64_t SparkArenaAlignBytes(
    uint64_t value,
    uint64_t alignment)
{
    if (alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u ||
        value > UINT64_MAX - (alignment - 1u))
    {
        return UINT64_MAX;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static inline int32_t SparkArenaAddRegionBytes(
    uint64_t *total,
    uint64_t count,
    uint64_t element_bytes)
{
    if (total == 0 || element_bytes == 0u ||
        count > (UINT64_MAX - *total) / element_bytes)
    {
        return -1;
    }
    *total += count * element_bytes;
    return 0;
}

static inline int32_t SparkArenaInitialize(
    SparkArena *arena,
    const SparkArenaClassDescriptor *descriptors,
    uint32_t class_count)
{
    uint64_t generation_region_offset;
    uint64_t generation_region_bytes;
    uint64_t link_region_bytes;
    uint64_t previous_slot_bytes;
    uint64_t region_offset;
    uint64_t slot_region_bytes;
    uint64_t total_bytes;
    uint32_t class_index;
    uint32_t slot_index;

    if (arena == 0 || descriptors == 0 || class_count == 0u ||
        class_count > SPARK_ARENA_MAX_CLASS_COUNT)
    {
        return -30901;
    }

    memset(arena, 0, sizeof(*arena));
    slot_region_bytes = 0u;
    link_region_bytes = 0u;
    generation_region_bytes = 0u;
    previous_slot_bytes = 0u;

    for (class_index = 0u; class_index < class_count; ++class_index)
    {
        uint64_t slot_bytes;
        uint64_t slot_count;

        slot_bytes = SparkArenaRoundSlotBytes(
            descriptors[class_index].slot_bytes);
        slot_count = descriptors[class_index].slot_count;
        if (descriptors[class_index].slot_bytes == 0u || slot_count == 0u)
        {
            return -30902;
        }
        if (class_index != 0u && slot_bytes <= previous_slot_bytes)
        {
            return -30903;
        }
        if (SparkArenaAddRegionBytes(
                &slot_region_bytes,
                slot_count,
                slot_bytes) != 0 ||
            SparkArenaAddRegionBytes(
                &link_region_bytes,
                slot_count,
                sizeof(uint32_t)) != 0 ||
            SparkArenaAddRegionBytes(
                &generation_region_bytes,
                slot_count,
                sizeof(uint64_t)) != 0)
        {
            return -30904;
        }
        previous_slot_bytes = slot_bytes;
    }

    if (slot_region_bytes > UINT64_MAX - link_region_bytes)
    {
        return -30904;
    }
    generation_region_offset = SparkArenaAlignBytes(
        slot_region_bytes + link_region_bytes,
        sizeof(uint64_t));
    if (generation_region_offset == UINT64_MAX ||
        generation_region_bytes > UINT64_MAX - generation_region_offset)
    {
        return -30904;
    }
    total_bytes = generation_region_offset + generation_region_bytes;
    if (total_bytes > (uint64_t)SIZE_MAX)
    {
        return -30910;
    }

    arena->backing = (uint8_t *)malloc((size_t)total_bytes);
    if (arena->backing == 0)
    {
        return -30905;
    }

    arena->backing_bytes = total_bytes;
    arena->class_count = class_count;
    region_offset = 0u;

    for (class_index = 0u; class_index < class_count; ++class_index)
    {
        uint64_t slot_bytes;

        slot_bytes = SparkArenaRoundSlotBytes(
            descriptors[class_index].slot_bytes);
        arena->classes[class_index].slots =
            arena->backing + region_offset;
        region_offset +=
            (uint64_t)descriptors[class_index].slot_count * slot_bytes;
    }

    for (class_index = 0u; class_index < class_count; ++class_index)
    {
        SparkArenaClass *arena_class;
        uint64_t slot_count;

        arena_class = &arena->classes[class_index];
        slot_count = descriptors[class_index].slot_count;
        arena_class->links =
            (uint32_t *)(void *)(arena->backing + region_offset);
        region_offset += slot_count * (uint64_t)sizeof(uint32_t);
    }

    region_offset = SparkArenaAlignBytes(region_offset, sizeof(uint64_t));
    if (region_offset != generation_region_offset)
    {
        free(arena->backing);
        memset(arena, 0, sizeof(*arena));
        return -30911;
    }

    for (class_index = 0u; class_index < class_count; ++class_index)
    {
        SparkArenaClass *arena_class;
        uint64_t slot_bytes;
        uint64_t slot_count;

        arena_class = &arena->classes[class_index];
        slot_bytes = SparkArenaRoundSlotBytes(
            descriptors[class_index].slot_bytes);
        slot_count = descriptors[class_index].slot_count;
        arena_class->slot_generations =
            (uint64_t *)(void *)(arena->backing + region_offset);
        region_offset += slot_count * (uint64_t)sizeof(uint64_t);
        arena_class->slot_bytes = (uint32_t)slot_bytes;
        arena_class->slot_count = (uint32_t)slot_count;
        arena_class->free_head = 0u;
        arena_class->free_count = (uint32_t)slot_count;
        arena_class->generation = 0u;

        for (slot_index = 0u; slot_index < slot_count; ++slot_index)
        {
            arena_class->links[slot_index] = slot_index + 1u;
            arena_class->slot_generations[slot_index] = 0u;
        }
        arena_class->links[slot_count - 1u] = SPARK_ARENA_NO_SLOT;
    }

    if (region_offset != total_bytes)
    {
        free(arena->backing);
        memset(arena, 0, sizeof(*arena));
        return -30911;
    }

    return 0;
}

static inline void SparkArenaDestroy(
    SparkArena *arena)
{
    if (arena == 0)
    {
        return;
    }

    free(arena->backing);
    memset(arena, 0, sizeof(*arena));
}

static inline int32_t SparkArenaAcquire(
    SparkArena *arena,
    uint32_t bytes,
    SparkArenaAllocation *allocation_out)
{
    uint32_t class_index;

    if (arena == 0 || bytes == 0u || allocation_out == 0)
    {
        return -30912;
    }

    memset(allocation_out, 0, sizeof(*allocation_out));

    for (class_index = 0u; class_index < arena->class_count; ++class_index)
    {
        SparkArenaClass *arena_class;
        uint64_t generation;
        uint32_t slot;

        arena_class = &arena->classes[class_index];
        if (bytes > arena_class->slot_bytes)
        {
            continue;
        }
        if (arena_class->free_count == 0u)
        {
            return -30913;
        }

        slot = arena_class->free_head;
        if (slot >= arena_class->slot_count)
        {
            return -30914;
        }

        if (arena_class->slot_generations[slot] == UINT64_MAX)
        {
            return -30918;
        }

        arena_class->free_head = arena_class->links[slot];
        arena_class->links[slot] = SPARK_ARENA_IN_USE;
        arena_class->free_count -= 1u;
        if (arena_class->generation != UINT64_MAX)
        {
            arena_class->generation += 1u;
        }

        generation = arena_class->slot_generations[slot] + 1u;
        arena_class->slot_generations[slot] = generation;

        allocation_out->pointer = arena_class->slots +
            ((uint64_t)slot * arena_class->slot_bytes);
        allocation_out->class_index = class_index;
        allocation_out->slot_index = slot;
        allocation_out->generation = generation;
        allocation_out->usable_bytes = arena_class->slot_bytes;
        return 0;
    }

    return -30915;
}

static inline int32_t SparkArenaRelease(
    SparkArena *arena,
    const SparkArenaAllocation *allocation)
{
    SparkArenaClass *arena_class;
    uint8_t *expected_pointer;

    if (arena == 0 || allocation == 0 || allocation->pointer == 0)
    {
        return -30906;
    }
    if (allocation->class_index >= arena->class_count)
    {
        return -30909;
    }

    arena_class = &arena->classes[allocation->class_index];
    if (allocation->slot_index >= arena_class->slot_count)
    {
        return -30909;
    }

    expected_pointer = arena_class->slots +
        ((uint64_t)allocation->slot_index * arena_class->slot_bytes);
    if ((uint8_t *)allocation->pointer != expected_pointer)
    {
        return -30907;
    }
    if (allocation->usable_bytes != arena_class->slot_bytes)
    {
        return -30916;
    }
    if (arena_class->links[allocation->slot_index] != SPARK_ARENA_IN_USE)
    {
        return -30908;
    }
    if (allocation->generation == 0u ||
        arena_class->slot_generations[allocation->slot_index] !=
            allocation->generation)
    {
        return -30917;
    }

    arena_class->links[allocation->slot_index] = arena_class->free_head;
    arena_class->free_head = allocation->slot_index;
    arena_class->free_count += 1u;
    return 0;
}

static inline uint64_t SparkArenaReservedBytes(
    const SparkArena *arena)
{
    if (arena == 0)
    {
        return 0u;
    }
    return arena->backing_bytes;
}

static inline uint64_t SparkArenaClassGeneration(
    const SparkArena *arena,
    uint32_t class_index)
{
    if (arena == 0 || class_index >= arena->class_count)
    {
        return 0u;
    }
    return arena->classes[class_index].generation;
}

#endif
