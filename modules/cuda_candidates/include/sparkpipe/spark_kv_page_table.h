#ifndef SPARKPIPE_SPARK_KV_PAGE_TABLE_H
#define SPARKPIPE_SPARK_KV_PAGE_TABLE_H

#include "sparkpipe/spark_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkKvPageResidencyKind
{
    SPARK_KV_PAGE_RESIDENCY_FREE = 0,
    SPARK_KV_PAGE_RESIDENCY_COLD = 1,
    SPARK_KV_PAGE_RESIDENCY_WARM = 2,
    SPARK_KV_PAGE_RESIDENCY_HOT = 3
} SparkKvPageResidencyKind;

typedef struct SparkKvPageTableEntry
{
    bool allocated;
    uint32_t stage_id;
    uint32_t slot_id;
    uint32_t logical_page_index;
    SparkModelLaneKind model_lane;
    uint64_t request_id;
    uint64_t kv_handle;
    uint64_t required_by_tick;
    uint64_t safe_release_after_tick;
    uint64_t last_touched_tick;
    SparkKvPageResidencyKind residency;
} SparkKvPageTableEntry;

typedef struct SparkKvPageTable
{
    uint32_t stage_id;
    uint32_t capacity;
    uint32_t allocated_pages;
    uint64_t represented_hot_pages;
    uint64_t represented_warm_pages;
    uint64_t represented_cold_pages;
    SparkKvPageTableEntry entries[SPARKPIPE_MAX_KV_PAGE_TABLE_ENTRIES_PER_STAGE];
} SparkKvPageTable;

SparkStatus SparkInitializeKvPageTable(SparkKvPageTable *page_table, uint32_t stage_id, uint32_t capacity);
SparkStatus SparkReserveKvPageRange(SparkKvPageTable *page_table, SparkModelLaneKind model_lane, uint32_t slot_id, uint64_t request_id, uint64_t kv_handle, uint32_t first_logical_page_index, uint32_t page_count, SparkKvPageResidencyKind residency, uint64_t required_by_tick, uint64_t safe_release_after_tick);
SparkStatus SparkReleaseKvPagesForHandle(SparkKvPageTable *page_table, uint64_t kv_handle, uint32_t *released_pages);
uint32_t SparkCountKvPagesByResidency(const SparkKvPageTable *page_table, SparkKvPageResidencyKind residency);
uint64_t SparkCountRepresentedKvPagesByResidency(const SparkKvPageTable *page_table, SparkKvPageResidencyKind residency);
SparkStatus SparkSetRepresentedKvPageCounts(SparkKvPageTable *page_table, uint64_t hot_pages, uint64_t warm_pages, uint64_t cold_pages);
const SparkKvPageTableEntry *SparkFindKvPageEntryConst(const SparkKvPageTable *page_table, uint64_t kv_handle, uint32_t logical_page_index);

#ifdef __cplusplus
}
#endif

#endif
