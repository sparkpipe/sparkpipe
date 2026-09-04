#pragma once

#include <stdint.h>

#include "inference/llms/kimi_k3/config.h"
#include "inference/llms/kimi_k3/generated_config.h"

#define K3_PIPELINE_SIDEBAND_ABI_VERSION 1u
#define K3_PIPELINE_SIDEBAND_FLAG_ATTNRES_BANK (1u << 0u)
#define K3_PIPELINE_SIDEBAND_FLAG_ATTNRES_PARTIAL (1u << 1u)
#define K3_PIPELINE_SIDEBAND_KNOWN_FLAGS \
    (K3_PIPELINE_SIDEBAND_FLAG_ATTNRES_BANK | \
     K3_PIPELINE_SIDEBAND_FLAG_ATTNRES_PARTIAL)
#define K3_PIPELINE_SIDEBAND_PAYLOAD_ALIGNMENT 16u

typedef struct K3PipelineSideband
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_generation;
    uint64_t sequence_position;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t next_layer_index;
    uint32_t attnres_block_index;
    uint32_t attnres_site_index;
    uint32_t attnres_source_count;
    uint32_t attnres_bank_offset;
    uint32_t attnres_bank_bytes;
    uint32_t attnres_partial_offset;
    uint32_t attnres_partial_bytes;
    uint32_t payload_bytes;
    uint32_t reserved1[3];
} K3PipelineSideband;

#define K3_PIPELINE_SIDEBAND_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(K3PipelineSideband))

static inline int K3PipelineSidebandRangeFits(
    uint32_t offset,
    uint32_t bytes,
    uint32_t capacity)
{
    return offset <= capacity && bytes <= capacity - offset;
}

static inline int K3PipelineSidebandRangesOverlap(
    uint32_t first_offset,
    uint32_t first_bytes,
    uint32_t second_offset,
    uint32_t second_bytes)
{
    uint32_t first_end;
    uint32_t second_end;

    first_end = first_offset + first_bytes;
    second_end = second_offset + second_bytes;
    return first_offset < second_end && second_offset < first_end;
}

static inline int K3PipelineSidebandValidate(
    const K3PipelineSideband *sideband)
{
    uint32_t expected_bank_bytes;

    if (sideband == 0)
        return 0;
    if (sideband->abi_version != K3_PIPELINE_SIDEBAND_ABI_VERSION ||
        sideband->descriptor_bytes != K3_PIPELINE_SIDEBAND_DESCRIPTOR_BYTES ||
        (sideband->flags & ~K3_PIPELINE_SIDEBAND_KNOWN_FLAGS) != 0u ||
        sideband->reserved0 != 0u ||
        sideband->reserved1[0] != 0u ||
        sideband->reserved1[1] != 0u ||
        sideband->reserved1[2] != 0u)
        return 0;
    if (sideband->request_id == 0u ||
        sideband->sequence_id == 0u ||
        sideband->sequence_generation == 0u)
        return 0;
    if (sideband->layer_count == 0u ||
        sideband->first_layer_index >= K3_LAYERS ||
        sideband->layer_count > K3_LAYERS - sideband->first_layer_index ||
        sideband->next_layer_index !=
            sideband->first_layer_index + sideband->layer_count)
        return 0;
    if (sideband->attnres_block_index > K3_ATTNRES_BLOCK_COUNT ||
        sideband->attnres_site_index >= K3_ATTNRES_SITES_PER_LAYER ||
        sideband->attnres_source_count > K3_ATTNRES_BANK_SLOTS)
        return 0;

    expected_bank_bytes =
        sideband->attnres_source_count * K3_ATTNRES_PARTIAL_BYTES;
    if ((sideband->flags & K3_PIPELINE_SIDEBAND_FLAG_ATTNRES_BANK) != 0u)
    {
        if (sideband->attnres_source_count == 0u ||
            sideband->attnres_bank_bytes != expected_bank_bytes ||
            sideband->attnres_bank_bytes > K3_ATTNRES_BANK_BYTES ||
            (sideband->attnres_bank_offset %
                K3_PIPELINE_SIDEBAND_PAYLOAD_ALIGNMENT) != 0u ||
            !K3PipelineSidebandRangeFits(
                sideband->attnres_bank_offset,
                sideband->attnres_bank_bytes,
                sideband->payload_bytes))
            return 0;
    }
    else if (sideband->attnres_source_count != 0u ||
        sideband->attnres_bank_offset != 0u ||
        sideband->attnres_bank_bytes != 0u)
    {
        return 0;
    }

    if ((sideband->flags & K3_PIPELINE_SIDEBAND_FLAG_ATTNRES_PARTIAL) != 0u)
    {
        if (sideband->attnres_partial_bytes != K3_ATTNRES_PARTIAL_BYTES ||
            (sideband->attnres_partial_offset %
                K3_PIPELINE_SIDEBAND_PAYLOAD_ALIGNMENT) != 0u ||
            !K3PipelineSidebandRangeFits(
                sideband->attnres_partial_offset,
                sideband->attnres_partial_bytes,
                sideband->payload_bytes))
            return 0;
    }
    else if (sideband->attnres_partial_offset != 0u ||
        sideband->attnres_partial_bytes != 0u)
    {
        return 0;
    }

    if ((sideband->flags & K3_PIPELINE_SIDEBAND_KNOWN_FLAGS) ==
            K3_PIPELINE_SIDEBAND_KNOWN_FLAGS &&
        K3PipelineSidebandRangesOverlap(
            sideband->attnres_bank_offset,
            sideband->attnres_bank_bytes,
            sideband->attnres_partial_offset,
            sideband->attnres_partial_bytes))
        return 0;

    return 1;
}

#if defined(__cplusplus)
static_assert(sizeof(K3PipelineSideband) == 104u,
    "K3 pipeline sideband wire descriptor changed");
#else
_Static_assert(sizeof(K3PipelineSideband) == 104u,
    "K3 pipeline sideband wire descriptor changed");
#endif
