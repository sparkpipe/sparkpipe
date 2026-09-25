#include <assert.h>
#include <stdatomic.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include <infiniband/verbs.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

#include "../node/weightd_mesh.c"

#ifndef SPARK_WEIGHTD_MESH_DIR
#define SPARK_WEIGHTD_MESH_DIR "/tmp/weightd-mesh"
#endif

#define TEST_MESH_PEERS (SPARK_WEIGHTD_MESH_RANKS_PER_BAND - 1u)
#define TEST_MESH_MAGIC UINT64_C(0x4d45534830303034)
#define TEST_MESH_LIVE_DIR "/tmp/weightd-mesh"

typedef struct TestMeshRecord
{
    uint64_t magic;
    uint32_t rank;
    uint32_t send_qpn[TEST_MESH_PEERS];
    uint32_t recv_qpn[TEST_MESH_PEERS];
    uint32_t rkey;
    uint64_t recv_addr;
    uint16_t lid;
    uint8_t gid[16];
    uint32_t rank_mask;
    uint64_t boot_ns;
} TestMeshRecord;

static uint32_t test_rank_mask = 0xffffu;

#define TEST_MESH_INTERFACE "rocep1s0f1"

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir, uint32_t rank_mask);
uint32_t SparkWeightdMeshReady(void);
void SparkWeightdMeshPoll(void);
uint32_t SparkWeightdMeshBroadcast(uint32_t peer_rank_mask,
    uint64_t source_offset, uint32_t length, uint64_t remote_offset,
    uint64_t seq_value, uint64_t seq_remote_offset);

#if defined(__APPLE__)
int memfd_create(const char *name, unsigned int flags)
{
    char path[128];
    int fd;
    (void)flags;
    (void)snprintf(path,sizeof(path),"/tmp/spark-mesh-mock-%s-%d-XXXXXX",
        name,(int)getpid());
    fd = mkstemp(path);
    if (fd >= 0)
        (void)unlink(path);
    return fd;
}
#endif

static uint32_t test_checks;
static uint32_t test_failures;

#define CHECK(cond, name) do { \
        test_checks++; \
        if ( !(cond) ) { \
            test_failures++; \
            fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
        } \
    } while (0)

static uint32_t test_peer_rank(uint32_t peer, uint32_t local_rank)
{
    return peer < local_rank ? peer : peer + 1u;
}

static uint32_t test_my_index(uint32_t peer_rank, uint32_t local_rank)
{
    return local_rank < peer_rank ? local_rank : local_rank - 1u;
}

static void test_record_path(uint32_t rank, char *path, size_t path_bytes)
{
    (void)snprintf(path,path_bytes,"%s/mesh-%x.rec",
        SPARK_WEIGHTD_MESH_DIR,rank);
}

static void test_fill_record(TestMeshRecord *record, uint32_t rank,
    uint32_t generation)
{
    uint32_t index;
    memset(record,0,sizeof(*record));
    record->magic = TEST_MESH_MAGIC;
    record->rank_mask = test_rank_mask;
    record->rank = rank;
    for (index = 0u; index < TEST_MESH_PEERS; index++)
    {
        record->send_qpn[index] =
            0x40000u + generation * 0x20000u + rank * 0x400u + index;
        record->recv_qpn[index] = record->send_qpn[index] + 0x200u;
    }
    record->rkey = 0x9000u + generation * 0x100u + rank;
    record->recv_addr = UINT64_C(0x7f0000000000) +
        generation * UINT64_C(0x1000000) + rank * UINT64_C(0x10000);
    record->lid = (uint16_t)(0x2000u + rank);
    for (index = 0u; index < 16u; index++)
        record->gid[index] = (uint8_t)(rank * 16u + index);
    record->boot_ns = UINT64_C(0xb000000000000000) +
        generation * UINT64_C(0x100000) + rank;
}

static int test_write_fully(int fd, const void *buffer, size_t bytes)
{
    const char *cursor = (const char *)buffer;
    size_t remaining = bytes;
    while (remaining > 0u)
    {
        ssize_t written = write(fd,cursor,remaining);
        if (written <= 0)
            return -1;
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
    return 0;
}

static int test_write_record(uint32_t rank, uint32_t generation)
{
    TestMeshRecord record;
    char path[256];
    int fd;
    int result;
    test_fill_record(&record,rank,generation);
    test_record_path(rank,path,sizeof(path));
    fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0644);
    if (fd < 0)
        return -1;
    result = test_write_fully(fd,&record,sizeof(record));
    if (result == 0 && fsync(fd) != 0)
        result = -1;
    if (close(fd) != 0)
        result = -1;
    return result;
}

static int test_read_record(uint32_t rank, TestMeshRecord *record)
{
    char path[256];
    int fd;
    char *cursor;
    size_t remaining;
    test_record_path(rank,path,sizeof(path));
    fd = open(path,O_RDONLY);
    if (fd < 0)
        return -1;
    cursor = (char *)record;
    remaining = sizeof(*record);
    while (remaining > 0u)
    {
        ssize_t got = read(fd,cursor,remaining);
        if (got <= 0)
        {
            (void)close(fd);
            return -1;
        }
        cursor += (size_t)got;
        remaining -= (size_t)got;
    }
    (void)close(fd);
    return record->magic == TEST_MESH_MAGIC ? 0 : -1;
}

static void test_expect_peer_wired(const TestMeshRecord *own_record,
    uint32_t local_rank, uint32_t peer, uint32_t generation,
    const char *tag)
{
    TestMeshRecord expected;
    uint32_t rank;
    uint32_t my_index;
    char name[128];
    rank = test_peer_rank(peer,local_rank);
    my_index = test_my_index(rank,local_rank);
    test_fill_record(&expected,rank,generation);
    (void)snprintf(name,sizeof(name),"%s peer=%u send qp aimed at record",
        tag,peer);
    CHECK(spark_stub_ibv_qp_remote_qpn(own_record->send_qpn[peer]) ==
        (int)expected.recv_qpn[my_index],name);
    (void)snprintf(name,sizeof(name),"%s peer=%u recv qp aimed at record",
        tag,peer);
    CHECK(spark_stub_ibv_qp_remote_qpn(own_record->recv_qpn[peer]) ==
        (int)expected.send_qpn[my_index],name);
    (void)snprintf(name,sizeof(name),"%s peer=%u send qp rts",tag,peer);
    CHECK(spark_stub_ibv_qp_state(own_record->send_qpn[peer]) ==
        IBV_QPS_RTS,name);
    (void)snprintf(name,sizeof(name),"%s peer=%u recv qp rts",tag,peer);
    CHECK(spark_stub_ibv_qp_state(own_record->recv_qpn[peer]) ==
        IBV_QPS_RTS,name);
}

static int test_stderr_saved = -1;
static int test_stderr_capture_fd = -1;

static void test_capture_begin(const char *path)
{
    fflush(stderr);
    test_stderr_saved = dup(2);
    test_stderr_capture_fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0644);
    if (test_stderr_saved < 0 || test_stderr_capture_fd < 0)
        return;
    (void)dup2(test_stderr_capture_fd,2);
}

static void test_capture_end(void)
{
    fflush(stderr);
    if (test_stderr_saved >= 0)
    {
        (void)dup2(test_stderr_saved,2);
        (void)close(test_stderr_saved);
        test_stderr_saved = -1;
    }
    if (test_stderr_capture_fd >= 0)
    {
        (void)close(test_stderr_capture_fd);
        test_stderr_capture_fd = -1;
    }
}

static int test_file_contains(const char *path, const char *token)
{
    char buffer[65536];
    int fd;
    ssize_t got;
    int found;
    fd = open(path,O_RDONLY);
    if (fd < 0)
        return 0;
    got = read(fd,buffer,sizeof(buffer) - 1u);
    (void)close(fd);
    if (got < 0)
        return 0;
    buffer[got] = '\0';
    found = strstr(buffer,token) != 0;
    return found;
}

static void test_ready_path(char *path, size_t path_bytes)
{
    (void)snprintf(path,path_bytes,"%s/.ready",SPARK_WEIGHTD_MESH_DIR);
}

static void test_clean_dir(void)
{
    char path[256];
    uint32_t rank;
    test_ready_path(path,sizeof(path));
    (void)unlink(path);
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        test_record_path(rank,path,sizeof(path));
        (void)unlink(path);
    }
}

static void test_sleep_ns(uint64_t ns)
{
    struct timespec pause;
    pause.tv_sec = (time_t)(ns / 1000000000ull);
    pause.tv_nsec = (long)(ns % 1000000000ull);
    (void)nanosleep(&pause,0);
}

static uint64_t test_shipped(uint32_t band, uint32_t rank)
{
    return *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank));
}

static SparkStatus test_post_slot(uint32_t band, uint32_t rank,
    uint64_t seq, uint32_t mask)
{
    SparkStatus status;
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    status = SparkWeightdMeshPostSlot(band,rank,seq,
        (uint64_t)rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u)),64u,mask);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return status;
}

static void test_complete_range(uint32_t first, uint32_t last)
{
    uint32_t i;
    CHECK(first <= last && last <= spark_stub_ibv_posted_count(),"completion range names accepted work only");
    if (first > last || last > spark_stub_ibv_posted_count()) return;
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,"posted WR has a reproducible completion identity");
        CHECK(spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0,"completion queue has declared capacity");
        if ((i - first + 1u) % 64u == 0u) SparkWeightdMeshDrainCq();
    }
    SparkWeightdMeshDrainCq();
}

static SparkWeightdMeshTopology test_identity_topology(uint32_t count,uint32_t local)
{
    SparkWeightdMeshTopology topology = {0};
    uint32_t rank;
    topology.rank_count = count;
    topology.local_rank = local;
    for (rank=0u; rank<count; rank++) topology.physical_ranks[rank] = rank;
    return topology;
}

static void test_slot_lifetimes(uint32_t local_rank)
{
    uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) &
        ~(1u << local_rank);
    uint32_t first, last, i;
    uint64_t seq = (UINT64_C(7) << 32u) | 1u;
    uint64_t slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
    volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u,local_rank));
    uint8_t *payload = (uint8_t *)weightd_mesh.recv_buffer + slot * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    SparkStubIbvPostedWork stale;
    test_complete_range(0u,spark_stub_ibv_posted_count());
    first = spark_stub_ibv_posted_count();
    memset(payload,0x5a,64u);
    entry[2] = slot;
    entry[1] = 64u;
    entry[3] = all_peers;
    __sync_synchronize();
    *(volatile uint64_t *)(payload + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u) = seq;
    __sync_synchronize();
    entry[0] = seq;
    SparkWeightdMeshDoorbellPoll();
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u,"B1 posts unchanged payload and tail to every peer");
    CHECK(test_shipped(0u,local_rank) == 0u,"posting transfers does not release the source slot");
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,"B1 WR captured");
        CHECK((work.flags & IBV_SEND_SIGNALED) != 0u,"every owned WR has terminal evidence");
        if ( (work.wr_id & 3u) == 3u )
            CHECK(spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0,"tail completion delivered before payload");
        else
        {
            CHECK(work.source == (uint64_t)(uintptr_t)payload && work.length == 64u,
                "B1 source and payload extent remain unchanged");
            CHECK(*(const uint8_t *)(uintptr_t)work.source == 0x5au,"NIC source retains original contribution until completion");
        }
    }
    SparkWeightdMeshDrainCq();
    CHECK(test_shipped(0u,local_rank) == 0u,"tails alone cannot release outstanding payload reads");
    CHECK(spark_stub_ibv_posted(first + 1u,&stale) == 0,"capture duplicate tail identity");
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_SUCCESS) == 0,"duplicate tail injected");
    SparkWeightdMeshDoorbellPoll();
    CHECK(test_shipped(0u,local_rank) == 0u,"duplicate completion does not consume another WR");
    CHECK(spark_stub_ibv_posted_count() == last,"pending doorbell is not posted a second time");
    test_complete_range(first,last - 2u);
    CHECK(test_shipped(0u,local_rank) == 0u,"last outstanding peer retains ownership");
    test_complete_range(last - 2u,last);
    CHECK(test_shipped(0u,local_rank) == seq,"all terminal WRs release the source generation");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(0u,local_rank,seq + 1u,all_peers) == SPARK_STATUS_OK,
        "completed source slot admits its next generation");
    last = spark_stub_ibv_posted_count();
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_SUCCESS) == 0,"prior generation completion injected");
    SparkWeightdMeshDrainCq();
    CHECK(test_shipped(0u,local_rank) == seq,"stale completion never releases the new generation");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq + 1u,"new generation releases after its own completions");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(0u,local_rank,seq + (UINT64_C(1) << 32u),all_peers) == SPARK_STATUS_OK,
        "new epoch permits reused low sequence bits");
    last = spark_stub_ibv_posted_count();
    CHECK(test_shipped(0u,local_rank) == seq + 1u,
        "prior epoch acknowledgement cannot release a new epoch");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq + (UINT64_C(1) << 32u),
        "completion acknowledges the full epoch and sequence tag");
    entry[0] = 0u;

    first = spark_stub_ibv_posted_count();
    spark_stub_ibv_fail_post_call(spark_stub_ibv_post_send_calls() + 2u);
    CHECK(test_post_slot(1u,local_rank,seq,all_peers) == SPARK_STATUS_IO_ERROR,
        "partial post failure reports explicit failure");
    last = spark_stub_ibv_posted_count();
    spark_stub_ibv_fail_post_call(0u);
    CHECK(last > first && test_shipped(1u,local_rank) == 0u,
        "partial failure retains successfully posted work without false ACK");
    CHECK(test_post_slot(1u,local_rank,seq + 1u,all_peers) == SPARK_STATUS_IO_ERROR,
        "partial failure fences new source ownership while draining");
    test_complete_range(first,last);
    CHECK(test_shipped(1u,local_rank) == 0u &&
        weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank].pending == 0u,
        "failed generation drains all accepted work but does not advertise success");

    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(2u,local_rank,seq,all_peers) == SPARK_STATUS_OK,"CQ failure scenario posts");
    last = spark_stub_ibv_posted_count();
    CHECK(spark_stub_ibv_posted(first,&stale) == 0,"capture failing WR identity");
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_RETRY_EXC_ERR) == 0,"terminal transport error injected");
    SparkWeightdMeshDrainCq();
    test_complete_range(first + 1u,last);
    CHECK(test_shipped(2u,local_rank) == 0u &&
        weightd_mesh.transfers[2u * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank].failed != 0u,
        "CQ error cannot be converted into a successful shipment");

    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(3u,local_rank,seq,1u << 2u) == SPARK_STATUS_OK,
        "tree destination uses the common posting path");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == 2u,"masked transfer posts only payload and tail for selected peer");
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0 &&
            work.qp_number == weightd_mesh.send_qps[2u]->qp_num,
            "peer mask resolves the existing physical-rank mapping");
    }
    test_complete_range(first,last);
    CHECK(test_shipped(3u,local_rank) == seq,"masked transfer requires only its selected peer completions");

    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(4u,local_rank,seq + 4u,all_peers) == SPARK_STATUS_OK,
        "sparse phase sequence publishes current payload");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u,
        "sparse phases never replay stale source slots");
    test_complete_range(first,last - 1u);
    CHECK(test_shipped(4u,local_rank) == 0u,"sparse shipment retains source until final completion");
    test_complete_range(last - 1u,last);
    CHECK(test_shipped(4u,local_rank) == seq + 4u,"sparse shipment acknowledges its actual tag");

    first = spark_stub_ibv_posted_count();
    for ( i = 0u; i < SPARK_WEIGHTD_MESH_SEND_CAPACITY; i++ )
        CHECK(SparkWeightdMeshBroadcast(all_peers,0u,64u,0u,0u,0u) == TEST_MESH_PEERS,
            "RPCs fill the shared physical SQ with independently owned sends");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == SPARK_WEIGHTD_MESH_PEERS * SPARK_WEIGHTD_MESH_SEND_CAPACITY,
        "maximum pending completions match actual configured SQ capacity");
    CHECK(test_post_slot(7u,local_rank,seq + 4u,all_peers) == SPARK_STATUS_BUSY,
        "capacity pressure rejects before any partial posting");
    CHECK(spark_stub_ibv_posted_count() == last,"BUSY leaves publication and completion ledgers unchanged");
    test_complete_range(first,first + TEST_MESH_PEERS * 2u);
    CHECK(test_post_slot(7u,local_rank,seq + 4u,all_peers) == SPARK_STATUS_OK,
        "completed work makes capacity retry useful");
    test_complete_range(first + TEST_MESH_PEERS * 2u,spark_stub_ibv_posted_count());
    for ( i = 0u; i < TEST_MESH_PEERS; i++ )
        CHECK(weightd_mesh.send_pending[i] == 0u,"all accepted SQ ownership returns after terminal completions");
}

typedef struct TestMeshActivityThread
{
    SparkWeightdServer *server;
    volatile sig_atomic_t stop;
    atomic_uint waiting;
    atomic_uint returned;
    uint64_t cpu_ns;
} TestMeshActivityThread;

static void *test_mesh_server_run(void *raw)
{
    TestMeshActivityThread *state = raw;
    assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
    return 0;
}

static void *test_mesh_wait(void *raw)
{
    TestMeshActivityThread *state = raw;
    struct timespec before,after;
    assert(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&before) == 0);
    atomic_store(&state->waiting,1u);
    SparkWeightdMeshWaitForActivity();
    SparkWeightdMeshDoorbellPoll();
    assert(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&after) == 0);
    state->cpu_ns = (uint64_t)(after.tv_sec - before.tv_sec) * UINT64_C(1000000000) +
        (uint64_t)after.tv_nsec - (uint64_t)before.tv_nsec;
    atomic_store(&state->returned,1u);
    return 0;
}

static void test_mesh_join_waiter(pthread_t thread,TestMeshActivityThread *waiter)
{
    uint32_t attempt;
    for (attempt=0u; attempt<250u && atomic_load(&waiter->returned) == 0u; attempt++)
        test_sleep_ns(UINT64_C(1000000));
    CHECK(atomic_load(&waiter->returned) != 0u,"mesh wake completes without periodic polling delay");
    assert(atomic_load(&waiter->returned) != 0u);
    assert(pthread_join(thread,0) == 0);
}

static uint32_t test_mesh_owner_count(void)
{
    uint32_t count;
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    count = weightd_mesh.activity_owners;
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return count;
}

static SparkWeightdMeshWaitRequest *test_wait_request(uint32_t band,uint32_t rank)
{
    return (SparkWeightdMeshWaitRequest *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_WAIT_ENTRY(band,rank));
}

static uint64_t *test_peer_tail(uint32_t band,uint32_t rank,uint64_t tag)
{
    uint64_t slot = (uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND +
        rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
        ((tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    return (uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        (slot + 1u) * SPARK_WEIGHTD_MESH_SLOT_BYTES - sizeof(uint64_t));
}

static uint64_t test_wait_publish(SparkWeightdMeshWaitRequest *request,
    uint64_t kind,uint64_t tag,uint64_t mask,uint64_t cancel)
{
    uint64_t id = request->request_id + 1u;
    request->ready = 0u;
    request->error = UINT64_C(0xdeadbeef);
    request->diag = UINT64_C(0xabcdef);
    request->kind = kind;
    request->tag = tag;
    request->peer_mask = mask;
    request->cancel_expected = cancel;
    request->timeout_ns = 1000u;
    request->version = SPARK_WEIGHTD_MESH_WAIT_VERSION;
    request->upstream_error = 0u;
    __atomic_store_n(&request->request_id,id,__ATOMIC_RELEASE);
    return id;
}

static void test_all_transfer_identities(void)
{
    uint32_t index;
    for (index=0u; index<SPARK_WEIGHTD_MESH_BANDS * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
    {
        SparkWeightdMeshTransfer saved = weightd_mesh.transfers[index];
        uint32_t band = index / SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        uint32_t rank = index % SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        uint64_t previous = weightd_mesh.doorbell_posted[index];
        uint64_t shipped = test_shipped(band,rank);
        uint32_t first = spark_stub_ibv_posted_count();
        uint32_t pending = weightd_mesh.send_pending[0];
        assert(saved.pending == 0u);
        memset(&weightd_mesh.transfers[index],0,sizeof(saved));
        weightd_mesh.transfers[index].generation = ++SparkWeightdMeshNextTransferGeneration;
        weightd_mesh.transfers[index].seq = previous + 1u;
        CHECK(SparkWeightdMeshPostTransfer(index,0u,2u,0u,64u) == SPARK_STATUS_OK,
            "every lane, band and rank posts a distinct completion identity");
        CHECK(SparkWeightdMeshPostTransfer(index,0u,3u,64u,8u) == SPARK_STATUS_OK,
            "every completion identity owns its tail write");
        test_complete_range(first,spark_stub_ibv_posted_count());
        CHECK(weightd_mesh.transfers[index].pending == 0u &&
            weightd_mesh.doorbell_posted[index] == previous + 1u &&
            test_shipped(band,rank) == previous + 1u,
            "terminal completions acknowledge the exact lane, band and rank");
        CHECK(weightd_mesh.send_pending[0] == pending,
            "every completion identity returns exactly its own send credits");
        weightd_mesh.transfers[index] = saved;
        weightd_mesh.doorbell_posted[index] = previous;
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank)) = shipped;
        weightd_mesh.send_pending[0] = pending;
    }
}

static void test_mesh_hardware_wait(void)
{
    const uint32_t band = 2u,rank = 0u;
    const uint32_t index = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank;
    const uint64_t tag = (UINT64_C(17) << 32u) | 1u;
    SparkWeightdMeshWaitRequest *request = test_wait_request(band,rank);
    uint64_t *cancel = (uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
        (SPARK_WEIGHTD_MESH_DOORBELL_CELL_CANCEL + 2u * band) *
            SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
    uint64_t *peer1 = test_peer_tail(band,1u,tag);
    uint64_t *peer2 = test_peer_tail(band,2u,tag);
    uint64_t id,old_error,old_diag;
    uint32_t first,last,invalid;
    CHECK(SPARK_WEIGHTD_IPC_ABI_VERSION == 8u &&
        SPARK_WEIGHTD_MESH_WAIT_OFFSET - SPARK_WEIGHTD_MESH_DOORBELL_OFFSET == 22528u &&
        (uint8_t *)test_wait_request(SPARK_WEIGHTD_MESH_BANDS - 1u,SPARK_WEIGHTD_MESH_RANKS_PER_BAND - 1u) + sizeof(*request) <=
            (uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_REGION_BYTES &&
        sizeof(*request) == 128u && offsetof(SparkWeightdMeshWaitRequest,ready) == 64u,
        "ABI8 gate geometry has separate producer and terminal cache lines within registered region");
    CHECK(SparkWeightdMeshSetActivity(band / 2u,1u) == SPARK_STATUS_OK,
        "hardware wait producer begins before publishing any GPU request");
    id = test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,0u);
    CHECK(SparkWeightdMeshHasWaitWork() != 0u,"unhandled gate prevents idle");
    SparkWeightdMeshWaitRequestsPoll(100u);
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u &&
        request->error == 0u && weightd_mesh.wait_terminal[index] == id &&
        SparkWeightdMeshHasWaitWork() == 0u,"initial source credit terminates exactly once");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(band,rank,tag,0x6u) == SPARK_STATUS_OK,
        "source credit test owns actual payload and tail WRs for two peers");
    last = spark_stub_ibv_posted_count();
    assert(last - first == 4u);
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,tag,0u,0u);
    SparkWeightdMeshWaitRequestsPoll(100u);
    CHECK(request->ready == 0u && test_shipped(band,rank) == 0u,
        "posting WRs cannot open source reuse gate");
    test_complete_range(first,last - 1u);
    SparkWeightdMeshWaitRequestsPoll(101u);
    CHECK(request->ready == 0u && test_shipped(band,rank) == 0u,
        "partial NIC completions retain source credit");
    test_complete_range(last - 1u,last);
    SparkWeightdMeshWaitRequestsPoll(102u);
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u &&
        request->error == 0u && test_shipped(band,rank) == tag,
        "last terminal WR opens exact source credit gate");
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,tag - 2u,0u,0u);
    SparkWeightdMeshWaitRequestsPoll(200u);
    CHECK(request->ready == 0u,"newer shipped tag cannot satisfy older full tag");
    SparkWeightdMeshWaitRequestsPoll(1200u);
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u &&
        request->error == tag - 2u && request->diag == tag &&
        test_shipped(band,rank) == tag,"timeout reports observed credit without forging shipment");
    *peer1 = 1u;
    *peer2 = tag;
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0x6u,0u);
    SparkWeightdMeshWaitRequestsPoll(2000u);
    CHECK(request->ready == 0u,"stale epoch with matching round cannot release peer wait");
    *peer1 = tag;
    *peer2 = tag + 2u;
    SparkWeightdMeshWaitRequestsPoll(2001u);
    CHECK(request->ready == 0u,"future peer round cannot satisfy exact tag");
    *peer2 = tag;
    SparkWeightdMeshWaitRequestsPoll(2002u);
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u && request->error == 0u,
        "all requested peer tails release gate without waiting for unrequested rank");
    *peer2 = 0u;
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0x2u,0u);
    SparkWeightdMeshWaitRequestsPoll(3000u);
    CHECK(request->ready == 1u && request->error == 0u,"sparse tree waits only on its declared peer");
    old_error = request->error;
    old_diag = request->diag;
    *cancel = 9u;
    SparkWeightdMeshWaitRequestsPoll(4000u);
    CHECK(request->ready == 1u && request->error == old_error && request->diag == old_diag,
        "late cancellation never rewrites a terminal record");
    id = test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0x4u,0u);
    SparkWeightdMeshWaitRequestsPoll(4000u);
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u &&
        request->error == (SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag) &&
        request->diag == 9u && weightd_mesh.wait_terminal[index] == id &&
        *peer2 == 0u && test_shipped(band,rank) == tag,
        "cancel publishes error and terminal identity before ready without forged peer progress");
    request->request_id = id - 1u;
    request->ready = 0u;
    request->error = 123u;
    SparkWeightdMeshWaitRequestsPoll(4001u);
    CHECK(request->ready == 0u && request->error == 123u &&
        SparkWeightdMeshHasWaitWork() == 0u,"stale handled ID cannot regain terminal-write ownership");
    request->request_id = id;
    request->ready = 1u;
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0x4u,9u);
    request->upstream_error = UINT64_C(0x123456789);
    SparkWeightdMeshWaitRequestsPoll(5000u);
    CHECK(request->ready == 1u && request->error == UINT64_C(0x123456789) && *peer2 == 0u,
        "upstream error drains later graph gate without nonexistent peer arrivals");
    for (invalid=0u; invalid<8u; invalid++)
    {
        test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0x4u,9u);
        if ( invalid == 0u ) request->version++;
        if ( invalid == 1u ) request->kind = 99u;
        if ( invalid == 2u ) request->peer_mask = 0u;
        if ( invalid == 3u ) request->peer_mask = 1u;
        if ( invalid == 4u ) request->peer_mask = 0x10u;
        if ( invalid == 5u ) request->tag = UINT64_C(17) << 32u;
        if ( invalid == 6u ) request->kind = SPARK_WEIGHTD_MESH_WAIT_SHIPPED;
        if ( invalid == 7u ) request->timeout_ns = 0u;
        SparkWeightdMeshWaitRequestsPoll(6000u);
        CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u &&
            request->error == UINT64_MAX,"invalid gate metadata terminates with explicit failure");
    }
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0x4u,9u);
    SparkWeightdMeshWaitRequestsPoll(7000u);
    assert(request->ready == 0u);
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0x4u,9u);
    SparkWeightdMeshWaitRequestsPoll(7001u);
    CHECK(request->ready == 1u && request->error == UINT64_MAX,
        "new ID cannot replace an outstanding consumer before terminal release");
    {
        SparkWeightdMeshWaitRequest *failed = test_wait_request(3u,rank);
        first = spark_stub_ibv_posted_count();
        spark_stub_ibv_fail_post_call(spark_stub_ibv_post_send_calls() + 2u);
        CHECK(test_post_slot(3u,rank,tag,0x6u) == SPARK_STATUS_IO_ERROR,
            "hardware gate sees actual partial transport post failure");
        spark_stub_ibv_fail_post_call(0u);
        last = spark_stub_ibv_posted_count();
        test_wait_publish(failed,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,tag,0u,0u);
        SparkWeightdMeshWaitRequestsPoll(7100u);
        CHECK(failed->ready == 1u && failed->error == UINT64_MAX &&
            test_shipped(3u,rank) == 0u && SparkWeightdMeshHasPending() != 0u,
            "failed source gate drains GPU while accepted NIC reads remain owned");
        test_complete_range(first,last);
        CHECK(test_shipped(3u,rank) == 0u && SparkWeightdMeshHasPending() == 0u,
            "terminal NIC failure never creates successful source credit");
        failed = test_wait_request(band,4u);
        test_wait_publish(failed,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,9u);
        SparkWeightdMeshWaitRequestsPoll(7200u);
        CHECK(failed->ready == 1u && failed->error == UINT64_MAX,
            "producer rank outside configured participants cannot claim a gate");
    }
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,9u);
    weightd_mesh.mesh_ready = 0u;
    SparkWeightdMeshWaitForActivity();
    SparkWeightdMeshWaitRequestsPoll(7300u);
    CHECK(request->ready == 1u && request->error == UINT64_MAX,
        "pending gate terminates explicitly if mesh loses readiness");
    weightd_mesh.mesh_ready = 1u;
    CHECK(SparkWeightdMeshSetActivity(band / 2u,0u) == SPARK_STATUS_OK,"hardware wait producer ends after gates drain");
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,9u);
    CHECK(SparkWeightdMeshHasWaitWork() != 0u,"unhandled request remains work after last producer ends");
    SparkWeightdMeshWaitForActivity();
    SparkWeightdMeshWaitRequestsPoll(8000u);
    CHECK(request->ready == 1u && request->error == UINT64_MAX &&
        SparkWeightdMeshHasWaitWork() == 0u,"request without live producer fails before returning idle");
    {
        uint64_t started;
        CHECK(SparkWeightdMeshSetActivity(band / 2u,1u) == SPARK_STATUS_OK,"idle producer holds activity");
        weightd_mesh.work_ns = 0u;
        started = SparkWeightdMeshMonotonicNs();
        SparkWeightdMeshWaitForActivity();
        CHECK(SparkWeightdMeshMonotonicNs() - started >= UINT64_C(150000),
            "an active lane without traffic polls at the dormant cadence instead of spinning");
        SparkWeightdMeshDoorbellPoll();
        CHECK(SparkWeightdMeshSetActivity(band / 2u,0u) == SPARK_STATUS_OK,"idle producer releases activity");
    }
    *cancel = 0u;
}

static void test_mesh_topology(void)
{
    SparkWeightdMeshTopology permuted = test_identity_topology(16u,4u);
    SparkWeightdMeshTopology subset = test_identity_topology(4u,3u),bad;
    uint32_t rank,first,last,i;
    uint64_t tag = (UINT64_C(91) << 32u) | 1u;
    const uint32_t band = 14u,local = 3u;
    SparkWeightdMeshWaitRequest *request = test_wait_request(band,local);
    uint64_t *entry = (uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band,local));
    for (rank=0u; rank<16u; rank++) permuted.physical_ranks[rank] = (rank + 3u) % 16u;
    for (rank=0u; rank<4u; rank++) subset.physical_ranks[rank] = rank + 4u;
    CHECK(test_post_slot(band,local,tag,1u) == SPARK_STATUS_INVALID_ARGUMENT &&
        SparkWeightdMeshSetActivity(7u,1u) == SPARK_STATUS_INVALID_ARGUMENT,
        "unconfigured lane cannot route or begin GPU work");
    bad = subset;bad.physical_ranks[2] = bad.physical_ranks[1];
    CHECK(SparkWeightdMeshLaneConfigure(7u,&bad) == SPARK_STATUS_INVALID_ARGUMENT,
        "duplicate physical ranks do not install a topology");
    bad = subset;bad.physical_ranks[0] = 16u;
    CHECK(SparkWeightdMeshLaneConfigure(7u,&bad) == SPARK_STATUS_INVALID_ARGUMENT,
        "out of range physical rank never aliases a valid rank");
    bad = subset;bad.local_rank = 0u;
    CHECK(SparkWeightdMeshLaneConfigure(7u,&bad) == SPARK_STATUS_INVALID_ARGUMENT,
        "logical local rank must map to this physical daemon");
    CHECK(SparkWeightdMeshLaneConfigure(6u,&permuted) == SPARK_STATUS_OK &&
        SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_OK,
        "permuted TP16 and physical ranks four through seven coexist on independent lanes");
    CHECK(test_post_slot(band,7u,tag,1u) == SPARK_STATUS_INVALID_ARGUMENT &&
        test_post_slot(band,local,tag,1u << 4u) == SPARK_STATUS_INVALID_ARGUMENT &&
        test_post_slot(band,local,tag,1u << local) == SPARK_STATUS_INVALID_ARGUMENT,
        "GPU source rank and mask remain logical and exclude self");
    CHECK(SparkWeightdMeshSetActivity(7u,1u) == SPARK_STATUS_OK &&
        SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_BUSY &&
        SparkWeightdMeshLaneConfigure(6u,&permuted) == SPARK_STATUS_OK,
        "live producer blocks its lane even between doorbells but not an independent lane");
    CHECK(SparkWeightdMeshSetActivity(7u,0u) == SPARK_STATUS_OK,
        "terminal producer releases only its lane activity");
    entry[1] = 64u;entry[2] = local * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
    entry[3] = 1u << 1u;entry[0] = tag;
    CHECK(SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_BUSY,
        "unposted GPU publication retains its lane after active interval");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(band,local,tag,1u << 1u) == SPARK_STATUS_OK,
        "TP4 logical peer one posts to physical five");
    last = spark_stub_ibv_posted_count();
    CHECK(last == first + 2u && SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_BUSY,
        "pending source reads retain lane configuration");
    for (i=first; i<last; i++)
    {
        SparkStubIbvPostedWork work;
        uint64_t offset = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND +
            local * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
        assert(spark_stub_ibv_posted(i,&work) == 0);
        if (i != first) offset += SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u;
        CHECK(work.qp_number == weightd_mesh.send_qps[5u]->qp_num &&
            work.source == (uint64_t)(uintptr_t)weightd_mesh.recv_buffer + offset &&
            work.remote == weightd_mesh.qp_info[5u].remote_addr + offset,
            "physical QP translation preserves the logical payload and remote offsets");
    }
    test_complete_range(first,last - 1u);
    CHECK(SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_BUSY,
        "partial completion cannot release lane configuration");
    test_complete_range(last - 1u,last);
    CHECK(test_shipped(band,local) == tag &&
        SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_OK && entry[0] == tag,
        "exact same topology reuses drained lane without resetting tags");
    bad = subset;bad.physical_ranks[0] = 0u;
    CHECK(SparkWeightdMeshLaneConfigure(7u,&bad) == SPARK_STATUS_UNSUPPORTED,
        "even drained lane rejects a changed topology until daemon restart");
    CHECK(SparkWeightdMeshSetActivity(7u,1u) == SPARK_STATUS_OK,"mapped wait producer begins");
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,1u << 1u,0u);
    CHECK(SparkWeightdMeshSetActivity(7u,0u) == SPARK_STATUS_OK,"wait outlives terminal producer");
    CHECK(SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_BUSY,
        "pending wait request retains lane ownership");
    CHECK(SparkWeightdMeshSetActivity(7u,1u) == SPARK_STATUS_OK,"explicit producer continues wait");
    SparkWeightdMeshWaitRequestsPoll(100u);
    CHECK(request->ready == 0u,"logical peer tail is required before mapped wait completes");
    *test_peer_tail(band,1u,tag) = tag;
    SparkWeightdMeshWaitRequestsPoll(101u);
    CHECK(request->ready == 1u && request->error == 0u,
        "mapped wait checks logical peer one rather than physical five");
    CHECK(SparkWeightdMeshSetActivity(7u,0u) == SPARK_STATUS_OK &&
        SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_OK,
        "wait terminal plus last producer end permits same profile reuse");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(12u,4u,tag,1u << 0u) == SPARK_STATUS_OK,
        "permuted TP16 logical zero maps to physical three");
    last = spark_stub_ibv_posted_count();
    {
        SparkStubIbvPostedWork work;
        assert(spark_stub_ibv_posted(first,&work) == 0);
        CHECK(work.qp_number == weightd_mesh.send_qps[3u]->qp_num,
            "TP16 permutation selects the declared destination");
        assert(spark_stub_ibv_complete(work.wr_id,IBV_WC_RETRY_EXC_ERR) == 0);
    }
    SparkWeightdMeshDrainCq();
    test_complete_range(first + 1u,last);
    CHECK(SparkWeightdMeshLaneConfigure(6u,&permuted) == SPARK_STATUS_IO_ERROR &&
        SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_OK,
        "failed source generation remains fenced without poisoning a different lane");
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshBroadcast(1u << 0u,0u,64u,0u,0u,0u) == 1u &&
        SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_BUSY,
        "unclassified CPU transfer conservatively retains every lane until terminal");
    test_complete_range(first,spark_stub_ibv_posted_count());
    CHECK(SparkWeightdMeshLaneConfigure(7u,&subset) == SPARK_STATUS_OK,
        "terminal CPU transfer allows configuration retry");
}

static void test_mesh_lane_protocol(void)
{
    SparkWeightdServerConfig config = {0};
    TestMeshActivityThread servers[2] = {{0},{0}};
    pthread_t threads[2];
    SparkWeightdClient *clients[2][SPARK_WEIGHTD_MESH_MAX_LANES] = {{0}};
    SparkWeightdClient *contender;
    char paths[2][128];
    uint32_t seed = 0x719e4a31u,lane,out,count,round,rank,job,attempt;
    SparkWeightdMeshTopology topology = test_identity_topology(4u,0u);
    const uint64_t timeout = UINT64_C(1000000000);
    SparkStatus status;
    for (rank=0u; rank<2u; rank++)
    {
        (void)snprintf(paths[rank],sizeof(paths[rank]),
            "/tmp/spark-lanes-%ld-%u.sock",(long)getpid(),rank);
        config.socket_path = paths[rank];
        config.device_bytes_max = UINT64_C(1048576);
        assert(SparkWeightdServerCreate(&config,&servers[rank].server) == SPARK_STATUS_OK);
        assert(pthread_create(&threads[rank],0,test_mesh_server_run,&servers[rank]) == 0);
    }
    {
        SparkWeightdClient *owner,*peer,*foreign;
        assert(SparkWeightdClientConnect(paths[0],&owner,0) == SPARK_STATUS_OK);
        assert(SparkWeightdClientConnect(paths[0],&peer,0) == SPARK_STATUS_OK);
        assert(SparkWeightdClientConnect(paths[1],&foreign,0) == SPARK_STATUS_OK);
        CHECK(SparkWeightdClientLaneBind(owner,peer,0u,&topology,&out) == SPARK_STATUS_INVALID_ARGUMENT,
            "unreserved client cannot lend a mesh band");
        CHECK(SparkWeightdClientLaneAcquire(owner,7u,&topology,&out,timeout) == SPARK_STATUS_OK,
            "owner reserves explicit topology before lending bands");
        {
            SparkWeightdMeshTopology wrong = topology;
            wrong.local_rank = 1u;
            CHECK(SparkWeightdClientLaneBind(owner,peer,0u,&wrong,&out) == SPARK_STATUS_INVALID_ARGUMENT,
                "borrower cannot change the local logical rank");
            wrong = topology;wrong.rank_count = 3u;
            CHECK(SparkWeightdClientLaneBind(owner,peer,0u,&wrong,&out) == SPARK_STATUS_INVALID_ARGUMENT,
                "borrower cannot change the degree");
            wrong = topology;wrong.physical_ranks[1] = 2u;wrong.physical_ranks[2] = 1u;
            CHECK(SparkWeightdClientLaneBind(owner,peer,0u,&wrong,&out) == SPARK_STATUS_INVALID_ARGUMENT,
                "borrower cannot change peer ordering");
        }
        CHECK(SparkWeightdClientLaneBind(owner,peer,0u,&topology,&out) == SPARK_STATUS_OK && out == 7u &&
            SparkWeightdClientLaneBind(owner,peer,1u,&topology,&out) == SPARK_STATUS_OK && out == 7u,
            "same daemon lends two distinct bands of its reserved lane");
        CHECK(SparkWeightdClientLaneBind(owner,peer,0u,&topology,&out) == SPARK_STATUS_DUPLICATE &&
            SparkWeightdClientLaneBind(owner,foreign,0u,&topology,&out) == SPARK_STATUS_INVALID_ARGUMENT &&
            SparkWeightdClientLaneBind(owner,peer,2u,&topology,&out) == SPARK_STATUS_INVALID_ARGUMENT,
            "duplicate band, foreign daemon and out of range band fail closed");
        CHECK(SparkWeightdClientLaneUnbind(owner,0u) == SPARK_STATUS_OK &&
            SparkWeightdClientLaneUnbind(owner,0u) == SPARK_STATUS_INVALID_ARGUMENT &&
            SparkWeightdClientLaneBind(owner,peer,1u,&topology,&out) == SPARK_STATUS_DUPLICATE &&
            SparkWeightdClientLaneUnbind(owner,1u) == SPARK_STATUS_OK,
            "unbinding main never releases or duplicates HC ownership");
        SparkWeightdClientClose(peer);
        SparkWeightdClientClose(foreign);
        SparkWeightdClientClose(owner);
    }
    for (count=2u; count<=4u; count++)
        for (round=0u; round<8u; round++)
        {
            uint32_t lanes[SPARK_WEIGHTD_MESH_MAX_LANES];
            for (lane=0u; lane<SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
                lanes[lane] = lane;
            for (lane=SPARK_WEIGHTD_MESH_MAX_LANES - 1u; lane>0u; lane--)
            {
                uint32_t other,saved;
                seed = seed * 1664525u + 1013904223u;
                other = seed % (lane + 1u);
                saved = lanes[lane];
                lanes[lane] = lanes[other];
                lanes[other] = saved;
            }
            fprintf(stderr,"lane protocol seed=0x719e4a31 jobs=%u schedule=%u\n",count,round);
            for (rank=0u; rank<2u; rank++)
            {
                for (job=0u; job<count; job++)
                {
                    uint32_t ordered = rank == 0u ? job : count - job - 1u;
                    assert(SparkWeightdClientConnect(paths[rank],&clients[rank][ordered],0) == SPARK_STATUS_OK);
                    out = SPARK_WEIGHTD_LANE_NONE;
                    for (attempt=0u; attempt<1000u; attempt++)
                    {
                        status = SparkWeightdClientLaneAcquire(clients[rank][ordered],
                            lanes[ordered],0,&out,timeout);
                        if (status != SPARK_STATUS_NO_LANE)
                            break;
                        test_sleep_ns(UINT64_C(1000000));
                    }
                    CHECK(status == SPARK_STATUS_OK && out == lanes[ordered],
                        "reversed job startup preserves coordinator lane across ranks");
                    out = SPARK_WEIGHTD_LANE_NONE;
                    CHECK(SparkWeightdClientLaneAcquire(clients[rank][ordered],
                        lanes[count],0,&out,timeout) == SPARK_STATUS_DUPLICATE &&
                        out == SPARK_WEIGHTD_LANE_NONE,
                        "duplicate acquisition cannot reserve a second lane");
                }
                assert(SparkWeightdClientConnect(paths[rank],&contender,0) == SPARK_STATUS_OK);
                out = SPARK_WEIGHTD_LANE_NONE;
                CHECK(SparkWeightdClientLaneAcquire(contender,SPARK_WEIGHTD_MESH_MAX_LANES,0,
                    &out,timeout) == SPARK_STATUS_INVALID_ARGUMENT && out == SPARK_WEIGHTD_LANE_NONE,
                    "out of range lane does not reserve or alter result");
                CHECK(SparkWeightdClientLaneAcquire(contender,lanes[0],0,&out,timeout) == SPARK_STATUS_NO_LANE &&
                    out == SPARK_WEIGHTD_LANE_NONE,"occupied explicit lane never redirects to a free lane");
                SparkWeightdClientClose(clients[rank][0]);
                clients[rank][0] = 0;
                for (attempt=0u; attempt<1000u; attempt++)
                {
                    status = SparkWeightdClientLaneAcquire(contender,lanes[0],0,&out,timeout);
                    if (status != SPARK_STATUS_NO_LANE)
                        break;
                    test_sleep_ns(UINT64_C(1000000));
                }
                CHECK(status == SPARK_STATUS_OK && out == lanes[0],
                    "closed idle owner releases only its reserved lane");
                SparkWeightdClientClose(contender);
                assert(SparkWeightdClientConnect(paths[rank],&contender,0) == SPARK_STATUS_OK);
                for (job=1u; job<count; job++)
                {
                    out = SPARK_WEIGHTD_LANE_NONE;
                    CHECK(SparkWeightdClientLaneAcquire(contender,lanes[job],0,&out,timeout) == SPARK_STATUS_NO_LANE &&
                        out == SPARK_WEIGHTD_LANE_NONE,"another job remains reserved after neighbor disconnect and reuse");
                    SparkWeightdClientClose(clients[rank][job]);
                    clients[rank][job] = 0;
                }
                SparkWeightdClientClose(contender);
            }
        }
    for (rank=0u; rank<2u; rank++)
    {
        uint32_t observed = 0u;
        for (job=0u; job<SPARK_WEIGHTD_MESH_MAX_LANES; job++)
        {
            assert(SparkWeightdClientConnect(paths[rank],&clients[rank][job],0) == SPARK_STATUS_OK);
            for (attempt=0u; attempt<1000u; attempt++)
            {
                status = SparkWeightdClientLaneAcquire(clients[rank][job],SPARK_WEIGHTD_LANE_NONE,0,&out,timeout);
                if (status != SPARK_STATUS_NO_LANE)
                    break;
                test_sleep_ns(UINT64_C(1000000));
            }
            CHECK(status == SPARK_STATUS_OK && out < SPARK_WEIGHTD_MESH_MAX_LANES &&
                (observed & (1u << out)) == 0u,"automatic mode reserves distinct free lanes up to capacity");
            if (out < SPARK_WEIGHTD_MESH_MAX_LANES)
                observed |= 1u << out;
        }
        assert(SparkWeightdClientConnect(paths[rank],&contender,0) == SPARK_STATUS_OK);
        CHECK(SparkWeightdClientLaneAcquire(contender,SPARK_WEIGHTD_LANE_NONE,0,&out,timeout) == SPARK_STATUS_NO_LANE &&
            observed == (1u << SPARK_WEIGHTD_MESH_MAX_LANES) - 1u,
            "ninth owner cannot alias any occupied pair of bands");
        SparkWeightdClientClose(contender);
        for (job=0u; job<SPARK_WEIGHTD_MESH_MAX_LANES; job++)
            SparkWeightdClientClose(clients[rank][job]);
        __atomic_store_n(&servers[rank].stop,1,__ATOMIC_SEQ_CST);
        assert(pthread_join(threads[rank],0) == 0);
        SparkWeightdServerDestroy(servers[rank].server);
    }
}

static void test_mesh_activity_protocol(uint32_t pending_first)
{
    SparkWeightdServerConfig config;
    SparkWeightdClient *first = 0,*second = 0;
    SparkWeightdHelloResult hello;
    TestMeshActivityThread server = {0},waiter = {0};
    pthread_t server_thread,wait_thread;
    char path[128];
    uint32_t post_first,post_last,round,lane;
    SparkWeightdMeshTopology topology = test_identity_topology(4u,0u);
    const uint64_t timeout = UINT64_C(1000000000);
    test_complete_range(pending_first,spark_stub_ibv_posted_count());
    assert(test_mesh_owner_count() == 0u);
    memset(&config,0,sizeof(config));
    (void)snprintf(path,sizeof(path),"/tmp/spark-mesh-activity-%ld.sock",(long)getpid());
    config.socket_path = path;
    config.device_bytes_max = UINT64_C(1048576);
    assert(SparkWeightdServerCreate(&config,&server.server) == SPARK_STATUS_OK);
    assert(pthread_create(&server_thread,0,test_mesh_server_run,&server) == 0);
    assert(SparkWeightdClientConnect(path,&first,&hello) == SPARK_STATUS_OK);
    assert(SparkWeightdClientConnect(path,&second,&hello) == SPARK_STATUS_OK);
    CHECK(SparkWeightdClientLaneAcquire(first,0u,&topology,&lane,timeout) == SPARK_STATUS_OK && lane == 0u &&
        SparkWeightdClientLaneAcquire(second,2u,&topology,&lane,timeout) == SPARK_STATUS_OK && lane == 2u,
        "active owners retain distinct reserved lanes");
    assert(pthread_create(&wait_thread,0,test_mesh_wait,&waiter) == 0);
    while ( atomic_load(&waiter.waiting) == 0u )
        test_sleep_ns(UINT64_C(100000));
    test_sleep_ns(UINT64_C(30000000));
    CHECK(atomic_load(&waiter.returned) == 0u,"idle mesh blocks instead of polling");
    CHECK(SparkWeightdClientMeshActivity(first,1u,1u,timeout) == SPARK_STATUS_OK,
        "begin acknowledges a live producer and wakes idle mesh");
    test_mesh_join_waiter(wait_thread,&waiter);
    CHECK(waiter.cpu_ns < UINT64_C(10000000),"idle mesh consumes bounded thread CPU");
    CHECK(SparkWeightdClientMeshActivity(first,1u,1u,timeout) == SPARK_STATUS_DUPLICATE &&
        test_mesh_owner_count() == 1u,"duplicate begin cannot acquire another owner");
    CHECK(SparkWeightdClientMeshActivity(first,2u,1u,timeout) == SPARK_STATUS_BUSY &&
        test_mesh_owner_count() == 1u,"new generation cannot replace active owner");
    CHECK(SparkWeightdClientMeshActivity(second,1u,1u,timeout) == SPARK_STATUS_OK &&
        test_mesh_owner_count() == 2u,"concurrent model owns a separate interval");
    CHECK(SparkWeightdClientMeshActivity(first,2u,0u,timeout) == SPARK_STATUS_INVALID_ARGUMENT &&
        test_mesh_owner_count() == 2u,"stale end cannot release either owner");
    CHECK(SparkWeightdClientMeshActivity(first,1u,0u,timeout) == SPARK_STATUS_OK &&
        test_mesh_owner_count() == 1u,"first terminal owner cannot retire second producer");
    CHECK(SparkWeightdClientMeshActivity(first,1u,0u,timeout) == SPARK_STATUS_INVALID_ARGUMENT &&
        test_mesh_owner_count() == 1u,"duplicate end cannot underflow ownership");
    CHECK(SparkWeightdClientMeshActivity(first,1u,1u,timeout) == SPARK_STATUS_INVALID_ARGUMENT,
        "retired generation cannot be reused");
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    post_first = spark_stub_ibv_posted_count();
    {
        volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u,0u));
        entry[1] = 64u;
        entry[2] = 1u;
        entry[3] = 0xeu;
        __sync_synchronize();
        entry[0] = 2u;
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    CHECK(SparkWeightdClientMeshActivity(second,1u,0u,timeout) == SPARK_STATUS_OK &&
        test_mesh_owner_count() == 0u,"last producer ends after publishing and before host shipment");
    memset(&waiter,0,sizeof(waiter));
    assert(pthread_create(&wait_thread,0,test_mesh_wait,&waiter) == 0);
    test_mesh_join_waiter(wait_thread,&waiter);
    CHECK(atomic_load(&waiter.returned) == 1u,"queued GPU publication keeps progress awake without active producers");
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    post_last = spark_stub_ibv_posted_count();
    CHECK(post_last - post_first == 6u && test_shipped(0u,0u) == 1u,
        "final queued publication posts payload and tail while retaining its source");
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    memset(&waiter,0,sizeof(waiter));
    assert(pthread_create(&wait_thread,0,test_mesh_wait,&waiter) == 0);
    test_mesh_join_waiter(wait_thread,&waiter);
    CHECK(atomic_load(&waiter.returned) == 1u,"outstanding NIC ownership keeps progress awake");
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    for (round=post_first; round<post_last; round++)
    {
        SparkStubIbvPostedWork work;
        assert(spark_stub_ibv_posted(round,&work) == 0);
        assert(spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0);
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    SparkWeightdMeshDrainCq();
    CHECK(test_shipped(0u,0u) == 2u,"terminal NIC completions release the source before idle");
    memset(&waiter,0,sizeof(waiter));
    assert(pthread_create(&wait_thread,0,test_mesh_wait,&waiter) == 0);
    CHECK(SparkWeightdClientMeshWrite(second,1u,0u,0u,64u,timeout) == SPARK_STATUS_OK,
        "explicit mesh RPC wakes idle progress without a GPU interval");
    test_mesh_join_waiter(wait_thread,&waiter);
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted_count() == post_last + 1u &&
            SparkWeightdMeshHasPending() != 0u,"RPC transfer owns exactly one pending WR");
        assert(spark_stub_ibv_posted(post_last,&work) == 0);
        assert(spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0);
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    SparkWeightdMeshDrainCq();
    for (round=2u; round<34u; round++)
    {
        memset(&waiter,0,sizeof(waiter));
        assert(pthread_create(&wait_thread,0,test_mesh_wait,&waiter) == 0);
        if ( round == 2u )
        {
            test_sleep_ns(UINT64_C(10000000));
            CHECK(atomic_load(&waiter.returned) == 0u,"fully drained mesh returns to blocking idle");
        }
        CHECK(SparkWeightdClientMeshActivity(first,round,1u,timeout) == SPARK_STATUS_OK,
            "producer transition wakes whether it precedes or follows wait enrollment");
        test_mesh_join_waiter(wait_thread,&waiter);
        CHECK(SparkWeightdClientMeshActivity(first,round,0u,timeout) == SPARK_STATUS_OK,
            "matching terminal generation releases repeated activity");
    }
    CHECK(SparkWeightdClientMeshActivity(first,34u,1u,timeout) == SPARK_STATUS_OK,
        "socket loss scenario begins known active generation");
    SparkWeightdClientClose(first);
    test_sleep_ns(UINT64_C(50000000));
    CHECK(test_mesh_owner_count() == 0u,"unexpected active disconnect releases its producer count");
    CHECK(SparkWeightdClientMeshActivity(second,2u,1u,timeout) == SPARK_STATUS_OK &&
        test_mesh_owner_count() == 1u,"a disconnected producer fences only its own lane");
    CHECK(SparkWeightdClientMeshActivity(second,2u,0u,timeout) == SPARK_STATUS_OK,
        "other lanes keep releasing activity normally");
    {
        SparkWeightdClient *replacement = 0;
        assert(SparkWeightdClientConnect(path,&replacement,0) == SPARK_STATUS_OK);
        lane = SPARK_WEIGHTD_LANE_NONE;
        CHECK(SparkWeightdClientLaneAcquire(replacement,0u,0,&lane,timeout) == SPARK_STATUS_BUSY &&
            lane == SPARK_WEIGHTD_LANE_NONE,"an orphaned lane is not reused without a verifiable topology");
        CHECK(SparkWeightdClientLaneAcquire(replacement,0u,&topology,&lane,timeout) == SPARK_STATUS_OK &&
            lane == 0u,"a verified reconfigure clears the orphaned lane");
        CHECK(SparkWeightdClientMeshActivity(replacement,1u,1u,timeout) == SPARK_STATUS_OK &&
            SparkWeightdClientMeshActivity(replacement,1u,0u,timeout) == SPARK_STATUS_OK,
            "the reconfigured lane serves new mesh activity");
        SparkWeightdClientClose(replacement);
    }
    SparkWeightdClientClose(second);
    __atomic_store_n(&server.stop,1,__ATOMIC_SEQ_CST);
    assert(pthread_join(server_thread,0) == 0);
    SparkWeightdServerDestroy(server.server);
}

int main(void)
{
    alarm(30);
    TestMeshRecord own_record;
    TestMeshRecord expected;
    struct stat st;
    char path[512];
    char log_path[320];
    uint32_t local_rank;
    uint32_t rank;
    uint32_t peer;
    uint32_t dead_qpn;
    uint32_t my_index;
    uint64_t modify_before;
    uint64_t failures_before;
    uint64_t post_before;
    uint32_t protocol_post_first;
    SparkStatus status;

    if (strcmp(SPARK_WEIGHTD_MESH_DIR,TEST_MESH_LIVE_DIR) == 0)
    {
        fprintf(stderr,"SETUP FAIL test_weightd_mesh_mock: built without a private "
            "SPARK_WEIGHTD_MESH_DIR\n");
        return 2;
    }
    test_ready_path(path,sizeof(path));
    if (stat(path,&st) == 0)
    {
        fprintf(stderr,"SETUP FAIL test_weightd_mesh_mock: %s looks like a live mesh\n",
            SPARK_WEIGHTD_MESH_DIR);
        return 2;
    }
    local_rank = 7u; /* explicit: the rank now comes from --mesh-rank */
    test_clean_dir();
    if (mkdir(SPARK_WEIGHTD_MESH_DIR,0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr,"SETUP FAIL test_weightd_mesh_mock: cannot create %s errno=%d\n",
            SPARK_WEIGHTD_MESH_DIR,errno);
        return 2;
    }

    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        if (rank == local_rank)
            continue;
        CHECK(test_write_record(rank,1u) == 0,"case1 write peer record");
    }
    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask);
    CHECK(status == SPARK_STATUS_BUSY,"case1 init publishes and defers");
    CHECK(test_read_record(local_rank,&own_record) == 0,
        "case1 own record published");
    CHECK(own_record.boot_ns != 0ull,"case1 own boot_ns nonzero");
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case1 mesh ready after cold wire");
    test_ready_path(path,sizeof(path));
    CHECK(stat(path,&st) == 0,"case1 .ready marker exists");
    for (peer = 0u; peer < TEST_MESH_PEERS; peer++)
        test_expect_peer_wired(&own_record,local_rank,peer,1u,"case1");
    CHECK(spark_stub_ibv_modify_qp_calls() ==
        (uint64_t)TEST_MESH_PEERS * 2u * 4u,
        "case1 modify_qp count = peers*2qps*4transitions");

    {
        SparkWeightdMeshRecord incompatible;
        int fd;
        test_record_path(0u,path,sizeof(path));
        fd = open(path,O_WRONLY);
        CHECK(fd >= 0,"open peer record for ABI qualification");
        if ( fd >= 0 )
        {
            uint64_t old_magic = UINT64_C(0x4d45534830303031);
            CHECK(test_write_fully(fd,&old_magic,sizeof(old_magic)) == 0,
                "write previous mesh ABI magic");
            close(fd);
        }
        CHECK(SparkWeightdMeshReadPeerRecord(0u,&incompatible) == SPARK_STATUS_ABI_MISMATCH,
            "prior mesh ABI is rejected explicitly");
        SparkWeightdMeshTryWire();
        test_ready_path(path,sizeof(path));
        CHECK(SparkWeightdMeshReady() == 0u && stat(path,&st) != 0,
            "incompatible peer clears readiness and its marker");
        CHECK(test_write_record(0u,1u) == 0,"restore compatible peer record");
        SparkWeightdMeshTryWire();
        CHECK(SparkWeightdMeshReady() == 1u,"compatible records restore readiness");
        test_record_path(0u,path,sizeof(path));
        CHECK(unlink(path) == 0,"remove peer record for missing-peer qualification");
        SparkWeightdMeshTryWire();
        test_ready_path(path,sizeof(path));
        CHECK(SparkWeightdMeshReady() == 0u && stat(path,&st) != 0,
            "missing peer cannot inherit an old ready marker");
        CHECK(test_write_record(0u,1u) == 0,"restore missing peer record");
        SparkWeightdMeshTryWire();
        CHECK(SparkWeightdMeshReady() == 1u,"restored peer permits readiness");
    }

    peer = 2u;
    rank = test_peer_rank(peer,local_rank);
    CHECK(test_write_record(rank,2u) == 0,"case2 peer record rewritten");
    modify_before = spark_stub_ibv_modify_qp_calls();
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case2 stays ready");
    CHECK(spark_stub_ibv_modify_qp_calls() - modify_before == 8ull,
        "case2 only the restarted peer is retransitioned");
    test_expect_peer_wired(&own_record,local_rank,peer,2u,"case2");
    test_expect_peer_wired(&own_record,local_rank,0u,1u,
        "case2 untouched peer keeps generation");

    modify_before = spark_stub_ibv_modify_qp_calls();
    test_sleep_ns(1100000000ull);
    SparkWeightdMeshPoll();
    CHECK(spark_stub_ibv_modify_qp_calls() == modify_before,
        "case3 unchanged records are a wiring no-op");
    CHECK(SparkWeightdMeshReady() == 1u,"case3 stays ready");

    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask);
    CHECK(status == SPARK_STATUS_BUSY,"case4 init republishes");
    CHECK(test_read_record(local_rank,&own_record) == 0,
        "case4 own record republished");
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        if (rank == local_rank)
            continue;
        CHECK(test_write_record(rank,3u) == 0,"case4 write peer record");
    }
    peer = 5u;
    rank = test_peer_rank(peer,local_rank);
    my_index = test_my_index(rank,local_rank);
    test_fill_record(&expected,rank,3u);
    dead_qpn = expected.recv_qpn[my_index];
    spark_stub_ibv_fail_modify_qp_for_qpn(dead_qpn);
    failures_before = spark_stub_ibv_modify_qp_failures();
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-wire-fail.log",
        SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    SparkWeightdMeshPoll();
    test_capture_end();
    CHECK(SparkWeightdMeshReady() == 0u,
        "case4 dead qpn keeps mesh unready");
    test_ready_path(path,sizeof(path));
    CHECK(stat(path,&st) != 0,"case4 no false .ready marker");
    CHECK(spark_stub_ibv_modify_qp_failures() - failures_before == 1ull,
        "case4 exactly one dead-qpn modify rejected");
    CHECK(test_file_contains(log_path,"WD-WIRE-FAIL"),
        "case4 WD-WIRE-FAIL reported");
    spark_stub_ibv_fail_modify_qp_for_qpn(0u);
    modify_before = spark_stub_ibv_modify_qp_calls();
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case4 good record completes wiring");
    CHECK(stat(path,&st) == 0,"case4 .ready marker after recovery");
    CHECK(spark_stub_ibv_modify_qp_calls() - modify_before == 8ull,
        "case4 only the failed peer is retransitioned on retry");
    test_expect_peer_wired(&own_record,local_rank,peer,3u,"case4");
    test_expect_peer_wired(&own_record,local_rank,0u,3u,"case4");

    modify_before = spark_stub_ibv_modify_qp_calls();
    SparkWeightdMeshPoll();
    CHECK(spark_stub_ibv_modify_qp_calls() == modify_before,
        "case5 wired record check is a no-op");
    peer = 7u;
    rank = test_peer_rank(peer,local_rank);
    CHECK(test_write_record(rank,4u) == 0,"case5 restarted peer record");
    spark_stub_ibv_poll_cq_inject(IBV_WC_RETRY_EXC_ERR,1u);
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-cqerr.log",
        SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    SparkWeightdMeshPoll();
    test_capture_end();
    CHECK(test_file_contains(log_path,"WD-MESH-CQERR"),
        "case5 WD-MESH-CQERR reported");
    CHECK(spark_stub_ibv_modify_qp_calls() - modify_before == 8ull,
        "case5 cqerr repair rewires the restarted peer");
    test_expect_peer_wired(&own_record,local_rank,peer,4u,"case5");
    CHECK(SparkWeightdMeshReady() == 1u,"case5 stays ready through repair");
    post_before = spark_stub_ibv_post_send_calls();
    CHECK(SparkWeightdMeshBroadcast(1u << rank,0ull,64u,0ull,0ull,0ull) == 1u,
        "case5 broadcast posts to repaired peer");
    CHECK(spark_stub_ibv_post_send_calls() - post_before == 1ull,
        "case5 post_send flows after repair");
    spark_stub_ibv_poll_cq_inject(IBV_WC_SUCCESS,1u);
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case5 clean poll after recovery");

    {
        SparkWeightdMeshTopology topology = test_identity_topology(16u,local_rank);
        test_complete_range(0u,spark_stub_ibv_posted_count());
        uint32_t lane;
        for (lane=0u; lane<6u; lane++)
            CHECK(SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_OK,
                "identity topology is explicitly configured for lifetime fixtures");
    }
    test_all_transfer_identities();
    test_slot_lifetimes(local_rank);
    test_mesh_topology();

    /* case 6: a second daemon instance with its own record dir must not
     * touch ours — the two-daemons-one-host separation (the fleet's
     * weightd vs the driver developers' standalone weightsd). */
    {
        char dir2[256];
        uint64_t dir1_boot = own_record.boot_ns;
        (void)snprintf(dir2,sizeof(dir2),"%s-second",SPARK_WEIGHTD_MESH_DIR);
        (void)mkdir(dir2,0755);
        status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,dir2,test_rank_mask);
        CHECK(status == SPARK_STATUS_BUSY,"case6 second init publishes");
        {
            (void)snprintf(path,sizeof(path),"%s/mesh-%x.rec",dir2,local_rank);
            CHECK(stat(path,&st) == 0,"case6 record lands in the second dir");
        }
        /* our original record is untouched */
        {
            TestMeshRecord first_again;
            CHECK(test_read_record(local_rank,&first_again) == 0,
                "case6 original record still readable");
            CHECK(first_again.boot_ns == dir1_boot,
                "case6 original record not clobbered by the second instance");
        }
        CHECK(unlink(path) == 0,"case6 remove second instance record");
        CHECK(rmdir(dir2) == 0,"case6 remove second instance directory");
    }

    test_clean_dir();
    test_rank_mask = 0xfu;
    CHECK(SparkWeightdMeshInit(0u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,0u) ==
        SPARK_STATUS_INVALID_ARGUMENT,"empty participant mask rejects");
    CHECK(SparkWeightdMeshInit(4u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask) ==
        SPARK_STATUS_INVALID_ARGUMENT,"rank outside participant mask rejects");
    CHECK(SparkWeightdMeshInit(0u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask) ==
        SPARK_STATUS_BUSY,"TP4 explicit group initializes");
    for ( rank = 1u; rank < 4u; rank++ )
        CHECK(test_write_record(rank,9u) == 0,"TP4 required record published");
    SparkWeightdMeshTryWire();
    CHECK(SparkWeightdMeshReady() == 1u,"TP4 needs only its three configured peers");
    post_before = spark_stub_ibv_post_send_calls();
    CHECK(SparkWeightdMeshBroadcast(1u << 4u,0u,64u,0u,0u,0u) == 0u &&
        spark_stub_ibv_post_send_calls() == post_before,
        "broadcast outside configured group rejects before posting");
    CHECK(test_post_slot(0u,0u,1u,1u << 4u) == SPARK_STATUS_INVALID_ARGUMENT,
        "doorbell cannot send to an absent participant");
    {
        SparkWeightdMeshTopology topology = test_identity_topology(4u,0u);
        uint32_t lane;
        SparkWeightdMeshTopology outside = topology;
        outside.physical_ranks[1] = 7u;
        CHECK(SparkWeightdMeshLaneConfigure(0u,&outside) == SPARK_STATUS_INVALID_ARGUMENT,
            "physical participant outside daemon rank mask is rejected without identity fallback");
        for (lane=0u; lane<SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
            CHECK(SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_OK,
                "TP4 hardware and activity fixtures explicitly configure identity ranks");
    }
    test_mesh_hardware_wait();
    post_before = spark_stub_ibv_post_send_calls();
    protocol_post_first = spark_stub_ibv_posted_count();
    {
        volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u,0u));
        entry[2] = 0u;
        entry[1] = 64u;
        entry[3] = 0xeu;
        entry[0] = 1u;
        SparkWeightdMeshDoorbellPoll();
    }
    CHECK(spark_stub_ibv_post_send_calls() - post_before == 6u,
        "TP4 B1 posts payload and tail to three peers only");
    test_mesh_lane_protocol();
    test_mesh_activity_protocol(protocol_post_first);
    test_rank_mask = 0xffu;
    CHECK(test_write_record(1u,9u) == 0,"peer with inconsistent group published");
    SparkWeightdMeshTryWire();
    CHECK(SparkWeightdMeshReady() == 0u,"inconsistent participant groups cannot become ready");

    test_clean_dir();
    (void)snprintf(path,sizeof(path),"%s/capture-wire-fail.log",
        SPARK_WEIGHTD_MESH_DIR);
    (void)unlink(path);
    (void)snprintf(path,sizeof(path),"%s/capture-cqerr.log",
        SPARK_WEIGHTD_MESH_DIR);
    (void)unlink(path);
    (void)rmdir(SPARK_WEIGHTD_MESH_DIR);
    fprintf(stderr,"%s: %u checks, %u failures\n",
        "test_weightd_mesh_mock",test_checks,test_failures);
    return test_failures != 0u ? 1 : 0;
}
