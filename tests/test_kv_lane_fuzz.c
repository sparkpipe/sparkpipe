#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_kv_page_cache.h"

#define SEQ_CAP 4u
#define PAGE_CAP 4u
#define ENTRY_CAP 8u
#define CASE_COUNT 13u

static uint64_t initial_seed = 7u;
static uint64_t fuzz_state;
static uint32_t fuzz_round,test_checks;
static uint32_t coverage[CASE_COUNT];

#define CHECK(cond) do { test_checks++; if ( !(cond) ) { \
    fprintf(stderr,"FAIL seed=%llu round=%u line=%u %s\n", \
    (unsigned long long)initial_seed,fuzz_round,__LINE__,#cond); exit(1); } } while (0)

typedef struct FuzzHarness
{
    SparkKvCacheArena arena;
    SparkKvCacheBlock blocks[ENTRY_CAP];
    SparkKvPageCacheEntry entries[ENTRY_CAP];
    SparkKvPageCacheSequence sequences[SEQ_CAP];
    uint32_t bucket_heads[ENTRY_CAP],resident_slots[ENTRY_CAP];
    uint32_t entry_map[ENTRY_CAP],logical[SEQ_CAP * PAGE_CAP],physical[SEQ_CAP * PAGE_CAP];
    uint8_t device[ENTRY_CAP * 4u];
    SparkKvPageCache cache;
    SparkKvLaneTransaction owners[SEQ_CAP];
    SparkKvLaneTransactions transactions;
    SparkModelDriverCacheLane lanes[SEQ_CAP];
    SparkModelDriverAdmissionRequest request;
} FuzzHarness;

static uint64_t FuzzRand(void)
{
    fuzz_state = fuzz_state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return(fuzz_state >> 17u);
}

static void HarnessInit(FuzzHarness *h,uint32_t count,uint32_t capacity)
{
    SparkKvCacheConfiguration arena = {0};
    SparkKvPageCacheConfiguration config = {0};
    uint32_t i;
    memset(h,0,sizeof(*h));
    arena.abi_version = SPARK_KV_CACHE_ABI_VERSION;
    arena.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    arena.logical_block_count = ENTRY_CAP;
    arena.block_token_count = 4u;
    arena.resident_block_capacity = capacity;
    arena.layer_count = arena.kv_head_count = arena.head_dim = arena.bytes_per_scalar = 1u;
    arena.key_block_stride_bytes = 4u;
    arena.key_device_base = h->device;
    arena.blocks = h->blocks;
    arena.resident_slot_logical_block_indices = h->resident_slots;
    CHECK(SparkKvCacheArenaInitialize(&h->arena,&arena) == SPARK_STATUS_OK);
    config.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
    config.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
    config.sequence_capacity = SEQ_CAP;
    config.entry_capacity = config.hash_bucket_count = ENTRY_CAP;
    config.kv_cache_arena = &h->arena;
    config.entries = h->entries;
    config.sequences = h->sequences;
    config.hash_bucket_heads = h->bucket_heads;
    config.entry_indices_by_logical_page = h->entry_map;
    CHECK(SparkKvPageCacheInitialize(&h->cache,&config) == SPARK_STATUS_OK);
    h->transactions.cache = &h->cache;
    h->transactions.lanes = h->owners;
    h->transactions.logical_pages = h->logical;
    h->transactions.physical_pages = h->physical;
    h->transactions.page_capacity = PAGE_CAP;
    for (i=0u; i<count; i++)
    {
        h->lanes[i].request_generation = h->lanes[i].step_generation = 1u;
        h->lanes[i].sequence_id = 100u + i;
        h->lanes[i].resident_sequence_slot = i;
        h->lanes[i].context_token_count = 1u;
    }
    h->request.descriptor_bytes = sizeof(h->request);
    h->request.program_id = 1u;
    h->request.request_id = 11u;
    h->request.submission_id = 21u;
    h->request.control_generation = 31u;
    h->request.transaction_id = 41u;
    h->request.request_generation = 51u;
    h->request.step_generation = 61u;
    h->request.deadline_time_ns = UINT64_MAX - 1u;
    h->request.active_slot_count = h->request.new_token_count = count;
    h->request.cache_lane_count = count;
    h->request.cache_lanes = h->lanes;
}

static SparkStatus Admit(FuzzHarness *h,uint32_t flag)
{
    h->request.admission_flags = flag;
    return(SparkKvLaneTransactionsAdmit(&h->transactions,&h->request));
}

static SparkModelDriverFrame Frame(FuzzHarness *h)
{
    SparkModelDriverFrame frame = {0};
    frame.program_id = h->request.program_id;
    frame.request_id = h->request.request_id;
    frame.sequence_id = h->request.sequence_id;
    frame.sequence_position = h->request.sequence_position;
    frame.deadline_time_ns = h->request.deadline_time_ns;
    frame.active_slot_count = h->request.active_slot_count;
    frame.new_token_count = h->request.new_token_count;
    frame.cache_lane_count = h->request.cache_lane_count;
    frame.cache_lanes = h->request.cache_lanes;
    frame.driver_dispatch_generation = h->request.control_generation;
    frame.driver_dispatch_cookie0 = h->request.transaction_id;
    frame.driver_dispatch_cookie1 = h->request.submission_id;
    frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
    return(frame);
}

static void Ledger(FuzzHarness *h,uint32_t count,uint32_t phase,uint32_t live,uint32_t position)
{
    uint32_t i,allocated = 0u,pins = 0u,bound = 0u;
    for (i=0u; i<SEQ_CAP; i++)
    {
        CHECK(h->owners[i].phase == (i < count ? phase : SPARK_KV_LANE_TRANSACTION_EMPTY));
        CHECK(h->sequences[i].sequence_id == (i < live ? 100u + i : 0u));
        if ( i < live )
        {
            CHECK(h->sequences[i].next_token_position == position);
            bound++;
        }
        if ( i < count && phase != SPARK_KV_LANE_TRANSACTION_EMPTY )
        {
            uint32_t page = h->logical[i * PAGE_CAP];
            CHECK(h->owners[i].page_count == 1u && page < ENTRY_CAP);
            CHECK(h->owners[i].request.request_id == 11u);
            CHECK(h->blocks[page].residency_reference_count == 1u);
            CHECK(h->physical[i * PAGE_CAP] == h->blocks[page].resident_slot_index);
        }
    }
    for (i=0u; i<ENTRY_CAP; i++)
    {
        uint32_t is_allocated = (h->blocks[i].flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u;
        allocated += is_allocated;
        pins += h->blocks[i].residency_reference_count;
        CHECK(h->blocks[i].reference_count == is_allocated);
        CHECK(h->entry_map[i] == SPARK_KV_PAGE_CACHE_NO_INDEX);
        if ( is_allocated )
        {
            CHECK(h->blocks[i].resident_slot_index < h->arena.resident_block_capacity);
            CHECK(h->resident_slots[h->blocks[i].resident_slot_index] == i);
        }
    }
    CHECK(allocated == live && bound == h->cache.live_sequence_count);
    CHECK(pins == (phase == SPARK_KV_LANE_TRANSACTION_EMPTY ? 0u : count));
    CHECK(h->arena.resident_block_count == live);
}

static void MutateIdentity(FuzzHarness *h,uint32_t field)
{
    switch (field)
    {
    case 0: h->request.request_id++; break;
    case 1: h->request.submission_id++; break;
    case 2: h->request.control_generation++; break;
    case 3: h->request.transaction_id++; break;
    case 4: h->request.request_generation++; break;
    case 5: h->request.step_generation++; break;
    case 6: h->lanes[0].request_generation++; break;
    default: h->lanes[0].step_generation++; break;
    }
}

static void RunCowCase(void)
{
    FuzzHarness h;
    SparkKvPageStore store;
    SparkKvPageStoreConfiguration configuration = {0};
    SparkModelDriverCacheLane original_lane;
    SparkKvCacheBlockView original_view,mutable_view;
    SparkModelDriverFrame frame;
    uint32_t source,index,step,operation,prefix = 1u + (uint32_t)(FuzzRand() % 3u),allocated;
    uint8_t staging[4],original[4];
    char path[] = "/tmp/sparkpipe-kv-cow-fuzz-XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0 && close(descriptor) == 0 && unlink(path) == 0);
    configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
    configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
    configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
    configuration.logical_page_capacity = ENTRY_CAP;
    configuration.transfer_capacity = 1u;
    configuration.page_bytes = configuration.staging_bytes = sizeof(staging);
    configuration.maximum_backing_bytes = ENTRY_CAP * sizeof(staging);
    configuration.backing_path = path;
    configuration.staging_address = staging;
    CHECK(SparkKvPageStoreInitialize(&store,&configuration) == SPARK_STATUS_OK);
    HarnessInit(&h,1u,SEQ_CAP);
    h.cache.page_store = &store;
    original_lane = h.lanes[0];
    original_lane.context_token_count = original_lane.publish_token_count = prefix;
    original_lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
    memset(&original_lane.publish_identity,31,sizeof(original_lane.publish_identity));
    CHECK(SparkKvPageCacheBeginLane(&h.cache,&original_lane,&source) == SPARK_STATUS_OK);
    CHECK(SparkKvCacheArenaResolveBlock(&h.arena,source,&original_view) == SPARK_STATUS_OK);
    for (index=0u; index<sizeof(original); index++) original[index] = (uint8_t)FuzzRand();
    memcpy((void *)original_view.key_device_address,original,sizeof(original));
    CHECK(SparkKvPageCacheCompleteLane(&h.cache,&original_lane) == SPARK_STATUS_OK);
    for (step=0u; step<8u; step++)
    {
        operation = step < 4u ? step : (uint32_t)(FuzzRand() % 4u);
        h.lanes[0] = original_lane;
        h.lanes[0].sequence_id = 200u + step;
        h.lanes[0].resident_sequence_slot = 1u;
        h.lanes[0].sequence_position = prefix;
        h.lanes[0].context_token_count = h.lanes[0].publish_token_count = prefix + 1u;
        h.lanes[0].prefix_token_count = prefix;
        h.lanes[0].prefix_identity = original_lane.publish_identity;
        h.lanes[0].flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
        memset(&h.lanes[0].publish_identity,(int)(41u + step),sizeof(h.lanes[0].publish_identity));
        h.request.submission_id++;
        h.request.transaction_id++;
        CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
        CHECK(h.owners[1].page_count == 1u);
        CHECK(h.logical[PAGE_CAP] != source && h.blocks[source].residency_reference_count == 0u);
        CHECK(SparkKvCacheArenaResolveBlock(&h.arena,h.logical[PAGE_CAP],&mutable_view) == SPARK_STATUS_OK);
        CHECK(memcmp((void *)mutable_view.key_device_address,original,sizeof(original)) == 0);
        if ( operation == 0u )
            CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
        else
        {
            CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
            frame = Frame(&h);
            CHECK(SparkKvLaneTransactionsClaim(&h.transactions,&frame) == SPARK_STATUS_OK);
            ((uint8_t *)mutable_view.key_device_address)[prefix] ^= UINT8_C(0xff);
            CHECK(SparkKvLaneTransactionsFinish(&h.transactions,(uint32_t[]){1u},1u,operation == 1u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK,operation == 3u ? 1u : 0u) == (operation == 1u ? SPARK_STATUS_IO_ERROR : operation == 3u ? SPARK_STATUS_UNSUPPORTED : SPARK_STATUS_OK));
            if ( operation == 2u )
            {
                CHECK(SparkKvPageCacheReleaseLane(&h.cache,1u,200u + step) == SPARK_STATUS_OK);
                CHECK(SparkKvPageCacheEvictUnused(&h.cache) == SPARK_STATUS_OK);
            }
        }
        CHECK(memcmp((void *)original_view.key_device_address,original,sizeof(original)) == 0);
        CHECK(h.cache.live_sequence_count == 1u && h.sequences[0].sequence_id == 100u);
        CHECK(h.entries[h.entry_map[source]].reference_count == 1u);
        allocated = 0u;
        for (index=0u; index<ENTRY_CAP; index++)
        {
            allocated += (h.blocks[index].flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u;
            CHECK(h.blocks[index].residency_reference_count == 0u);
        }
        CHECK(allocated == 1u && h.arena.resident_block_count == 1u);
        CHECK(h.owners[1].phase == SPARK_KV_LANE_TRANSACTION_EMPTY && h.sequences[1].sequence_id == 0u);
    }
    CHECK(SparkKvLaneTransactionsReset(&h.transactions) == SPARK_STATUS_OK);
    Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
    SparkKvPageStoreDestroy(&store);
    CHECK(unlink(path) == 0);
}

static void RunCase(uint32_t scenario,uint32_t count)
{
    FuzzHarness h;
    SparkModelDriverFrame frame;
    SparkModelDriverAdmissionRequest saved;
    SparkModelDriverCacheLane saved_lane;
    uint32_t slots[SEQ_CAP] = {0u,1u,2u,3u},i;
    SparkStatus status;
    coverage[scenario]++;
    if ( scenario == 12u ) { RunCowCase(); return; }
    if ( scenario < 2u && count < 2u ) count = 2u;
    HarnessInit(&h,count,scenario == 0u ? count - 1u : SEQ_CAP);
    if ( scenario == 0u )
    {
        CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_CAPACITY_EXCEEDED);
        Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
        return;
    }
    if ( scenario == 1u )
    {
        h.lanes[1].resident_sequence_slot = 0u;
        CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_INVALID_ARGUMENT);
        Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
        return;
    }
    CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
    CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_PREPARED,count,0u);
    saved = h.request;
    saved_lane = h.lanes[0];
    MutateIdentity(&h,(uint32_t)(FuzzRand() % 8u));
    CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_VALIDATION_FAILED);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_PREPARED,count,0u);
    h.request = saved;
    h.lanes[0] = saved_lane;
    if ( scenario == 2u || scenario == 3u )
    {
        if ( scenario == 2u )
            CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
        else
            CHECK(SparkKvLaneTransactionsReset(&h.transactions) == SPARK_STATUS_OK);
        Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
        return;
    }
    if ( scenario == 4u )
    {
        for (i=0u; i<count; i++) h.owners[i].prepared_since_ns = 1u;
        h.request.request_id++;
        h.request.deadline_time_ns++;
        status = Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
        CHECK(status == SPARK_STATUS_VALIDATION_FAILED);
        Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_PREPARED,count,0u);
        h.request = saved;
    }
    CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_COMMITTED,count,0u);
    if ( scenario == 5u )
    {
        h.request.request_id++;
        h.request.deadline_time_ns++;
        CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_BUSY);
        Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_COMMITTED,count,0u);
        h.request = saved;
        CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
        Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
        return;
    }
    frame = Frame(&h);
    frame.driver_dispatch_generation++;
    CHECK(SparkKvLaneTransactionsClaim(&h.transactions,&frame) == SPARK_STATUS_VALIDATION_FAILED);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_COMMITTED,count,0u);
    frame.driver_dispatch_generation--;
    CHECK(SparkKvLaneTransactionsClaim(&h.transactions,&frame) == SPARK_STATUS_OK);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_EXECUTING,count,0u);
    if ( scenario == 6u || scenario == 7u )
    {
        for (i=0u; i<count; i++) h.owners[i].executing_since_ns = 1u;
        h.request.request_id++;
        h.request.deadline_time_ns++;
        CHECK(Admit(&h,scenario == 6u ? SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE : SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_BUSY);
        Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_EXECUTING,count,0u);
        h.request = saved;
    }
    CHECK(SparkKvLaneTransactionsReset(&h.transactions) == SPARK_STATUS_BUSY);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_EXECUTING,count,0u);
    CHECK(SparkKvLaneTransactionsFinish(&h.transactions,slots,count - 1u,SPARK_STATUS_OK,0u) == SPARK_STATUS_INVALID_ARGUMENT);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_EXECUTING,count,0u);
    slots[count > 1u ? 1u : 0u] = count > 1u ? 0u : SEQ_CAP;
    CHECK(SparkKvLaneTransactionsFinish(&h.transactions,slots,count,SPARK_STATUS_OK,0u) == SPARK_STATUS_INVALID_ARGUMENT);
    Ledger(&h,count,SPARK_KV_LANE_TRANSACTION_EXECUTING,count,0u);
    slots[0] = 0u;
    slots[1] = 1u;
    status = scenario == 8u ? SPARK_STATUS_IO_ERROR : scenario == 10u ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_OK;
    CHECK(SparkKvLaneTransactionsFinish(&h.transactions,slots,count,scenario == 10u ? SPARK_STATUS_OK : status,scenario == 10u ? UINT32_MAX : 0u) == status);
    Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,status == SPARK_STATUS_OK ? count : 0u,1u);
    CHECK(SparkKvLaneTransactionsFinish(&h.transactions,slots,count,status,0u) == SPARK_STATUS_INVALID_ARGUMENT);
    if ( status == SPARK_STATUS_OK && scenario == 9u )
    {
        h.request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
        for (i=0u; i<count; i++) h.lanes[i].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
        h.lanes[0].sequence_id++;
        CHECK(Admit(&h,0u) == SPARK_STATUS_NOT_FOUND);
        Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,count,1u);
        h.lanes[0].sequence_id--;
        CHECK(Admit(&h,0u) == SPARK_STATUS_OK);
    }
    if ( scenario == 11u )
    {
        for (i=0u; i<count; i++)
        {
            h.lanes[i].sequence_position = 1u;
            h.lanes[i].context_token_count = 2u;
        }
        CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
        CHECK(h.owners[0].mutation_flags == 0u);
        CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
        frame = Frame(&h);
        CHECK(SparkKvLaneTransactionsClaim(&h.transactions,&frame) == SPARK_STATUS_OK);
        CHECK(SparkKvLaneTransactionsFinish(&h.transactions,slots,count,SPARK_STATUS_IO_ERROR,0u) == SPARK_STATUS_IO_ERROR);
        Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
        for (i=0u; i<count; i++)
        {
            h.lanes[i].sequence_position = 0u;
            h.lanes[i].context_token_count = 1u;
        }
    }
    CHECK(SparkKvLaneTransactionsReset(&h.transactions) == SPARK_STATUS_OK);
    Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
    for (i=0u; i<count; i++) h.lanes[i].flags = 0u;
    CHECK(SparkKvLaneTransactionsClaim(&h.transactions,&frame) == SPARK_STATUS_BUSY);
    h.request = saved;
    h.request.request_id = 12u;
    h.request.control_generation++;
    CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
    h.request = saved;
    CHECK(Admit(&h,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_VALIDATION_FAILED);
    CHECK(h.owners[0].request.request_id == 12u);
    CHECK(SparkKvLaneTransactionsReset(&h.transactions) == SPARK_STATUS_OK);
    Ledger(&h,0u,SPARK_KV_LANE_TRANSACTION_EMPTY,0u,0u);
}

int main(int argc,char **argv)
{
    uint32_t rounds = 500u,i,only = CASE_COUNT;
    if ( argc > 1 ) initial_seed = strtoull(argv[1],0,0);
    if ( argc > 2 ) rounds = (uint32_t)strtoul(argv[2],0,0);
    if ( argc > 3 ) only = (uint32_t)strtoul(argv[3],0,0);
    fuzz_state = initial_seed;
    if ( only < CASE_COUNT )
    {
        RunCase(only,3u);
        printf("test_kv_lane_fuzz: PASS seed=%llu scenario=%u checks=%u\n",(unsigned long long)initial_seed,only,test_checks);
        return(0);
    }
    for (fuzz_round=0u; fuzz_round<CASE_COUNT + rounds; fuzz_round++)
        RunCase(fuzz_round < CASE_COUNT ? fuzz_round : (uint32_t)(FuzzRand() % CASE_COUNT),1u + (uint32_t)(FuzzRand() % SEQ_CAP));
    for (i=0u; i<CASE_COUNT; i++) CHECK(coverage[i] != 0u);
    printf("test_kv_lane_fuzz: PASS seed=%llu rounds=%u scenarios=%u checks=%u\n",(unsigned long long)initial_seed,rounds,CASE_COUNT,test_checks);
    return(0);
}
