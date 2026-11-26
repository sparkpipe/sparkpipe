#include <assert.h>
#include <stdatomic.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <infiniband/verbs.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_sha256.h"

#include "../node/weightd_mesh.c"

uint64_t spark_stub_mlx5_wr_id(uint32_t qpn,uint32_t index)
{
    uint32_t peer;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
        if ( weightd_mesh.send_qps[peer] != 0 && weightd_mesh.send_queues[peer].qpn == qpn )
            return weightd_mesh.send_queues[peer].wr_id[index & weightd_mesh.send_queues[peer].mask];
    return UINT64_MAX;
}

int spark_stub_mlx5_cq_live(const struct ibv_cq *cq)
{
    return cq != 0 && (cq == weightd_mesh.cq || cq == weightd_mesh.pair_cq) ? 1 : 0;
}

#ifndef SPARK_WEIGHTD_MESH_DIR
#error "SPARK_WEIGHTD_MESH_DIR must name the mock mesh record directory"
#endif

#define TEST_MESH_PEERS (SPARK_WEIGHTD_MESH_RANKS_PER_BAND - 1u)
#define TEST_MESH_MAGIC UINT64_C(0x4d45534830303131)
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
    uint32_t pair_rkey;
    uint16_t pair_lid;
    uint8_t pair_gid[16];
    uint64_t wire_generation;
    uint64_t seen_generation[TEST_MESH_PEERS];
    uint32_t pair_send_qpn[TEST_MESH_PEERS];
    uint32_t pair_recv_qpn[TEST_MESH_PEERS];
    uint32_t ports;
    uint32_t switched_pair;
} TestMeshRecord;

_Static_assert(sizeof(TestMeshRecord) == sizeof(SparkWeightdMeshRecord),"the test record mirrors the daemon record");

static uint32_t test_rank_mask = 0xffffu;
static uint32_t test_record_ports = 1u;
static uint32_t test_record_switched = 0u;

#define TEST_MESH_INTERFACE "rocep1s0f1"

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir, uint32_t rank_mask,
    const char *pair_interface_name, uint32_t pair_sgid_index, uint32_t traffic_class);
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
    record->gid[10] = 0xffu;
    record->gid[11] = 0xffu;
    record->gid[12] = 10u;
    record->gid[13] = 10u;
    record->gid[14] = 100u;
    record->gid[15] = (uint8_t)(10u + rank);
    record->pair_rkey = 0xa000u + generation * 0x100u + rank;
    record->pair_lid = (uint16_t)(0x3000u + rank);
    memcpy(record->pair_gid,record->gid,sizeof(record->pair_gid));
    record->pair_gid[14] = 200u;
    record->pair_gid[15] = (uint8_t)rank;
    record->boot_ns = UINT64_C(0xb000000000000000) +
        generation * UINT64_C(0x100000) + rank;
    for (index = 0u; index < TEST_MESH_PEERS; index++)
    {
        record->pair_send_qpn[index] = record->send_qpn[index] + 0x100u;
        record->pair_recv_qpn[index] = record->recv_qpn[index] + 0x100u;
    }
    record->ports = test_record_ports;
    record->switched_pair = test_record_switched;
    if (test_record_ports == 2u)
    {
        memcpy(record->send_qpn,record->pair_send_qpn,sizeof(record->send_qpn));
        memcpy(record->recv_qpn,record->pair_recv_qpn,sizeof(record->recv_qpn));
        record->rkey = record->pair_rkey;
        record->lid = record->pair_lid;
        memcpy(record->gid,record->pair_gid,sizeof(record->gid));
    }
    if (test_record_ports != 3u)
    {
        memset(record->pair_send_qpn,0,sizeof(record->pair_send_qpn));
        memset(record->pair_recv_qpn,0,sizeof(record->pair_recv_qpn));
        memset(record->pair_gid,0,sizeof(record->pair_gid));
        record->pair_rkey = 0u;
        record->pair_lid = 0u;
    }
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

static int test_write_record_raw(const TestMeshRecord *record)
{
    char path[256];
    int fd;
    int result;
    test_record_path(record->rank,path,sizeof(path));
    fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0644);
    if (fd < 0)
        return -1;
    result = test_write_fully(fd,record,sizeof(*record));
    if (close(fd) != 0)
        result = -1;
    return result;
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

static uint32_t test_file_count(const char *path, const char *token)
{
    char buffer[65536];
    const char *cursor;
    uint32_t count = 0u;
    int fd;
    ssize_t got;
    fd = open(path,O_RDONLY);
    if (fd < 0)
        return 0u;
    got = read(fd,buffer,sizeof(buffer) - 1u);
    (void)close(fd);
    if (got < 0)
        return 0u;
    buffer[got] = '\0';
    for (cursor = strstr(buffer,token); cursor != 0; cursor = strstr(cursor + 1,token))
        count++;
    return count;
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

static SparkStatus test_post_route(uint32_t band, uint32_t rank,
    uint64_t seq, uint64_t bytes, SparkWeightdMeshRoute route)
{
    SparkStatus status;
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    status = SparkWeightdMeshPostSlot(band,rank,seq,
        (uint64_t)rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u)),bytes,route,
        SparkMonotonicNs());
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return status;
}

static SparkStatus test_post_slot(uint32_t band, uint32_t rank,
    uint64_t seq, uint32_t mask)
{
    SparkWeightdMeshRoute route = {0};
    route.fields.peer_mask = mask;
    return test_post_route(band,rank,seq,64u,route);
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
    memset(payload,0x5a,1024u);
    entry[2] = slot;
    entry[1] = 1024u;
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
        if ( (work.wr_id & 3u) == 3u )
            CHECK((work.flags & IBV_SEND_SIGNALED) != 0u && (work.flags & IBV_SEND_INLINE) != 0u &&
                work.length == 8u && work.inline_word == seq,
                "every tail is signaled and carries the slot tag inline");
        else
        {
            SparkStubIbvPostedWork tail;
            CHECK((work.flags & IBV_SEND_SIGNALED) == 0u && spark_stub_ibv_posted(i + 1u,&tail) == 0 &&
                tail.qp_number == work.qp_number && tail.wqe_index == work.wqe_index + 1u && (tail.wr_id & 3u) == 3u,
                "every payload is chained directly before its signaled tail on the same queue");
            CHECK(work.source == (uint64_t)(uintptr_t)payload && work.length == 1024u,
                "B1 source and payload extent remain unchanged");
            CHECK(*(const uint8_t *)(uintptr_t)work.source == 0x5au,"NIC source retains original contribution until completion");
        }
    }
    test_complete_range(first,last - 2u);
    CHECK(test_shipped(0u,local_rank) == 0u,"the last outstanding peer retains ownership");
    CHECK(spark_stub_ibv_posted(first + 1u,&stale) == 0 && (stale.wr_id & 3u) == 3u,"capture duplicate tail identity");
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_SUCCESS) == 0,"duplicate tail injected");
    SparkWeightdMeshDoorbellPoll();
    CHECK(test_shipped(0u,local_rank) == 0u,"duplicate completion does not consume another WR");
    CHECK(spark_stub_ibv_posted_count() == last,"pending doorbell is not posted a second time");
    test_complete_range(last - 2u,last);
    CHECK(test_shipped(0u,local_rank) == seq,"the last tail completion releases the source generation");
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
    CHECK(test_post_slot(1u,local_rank,seq,all_peers) == SPARK_STATUS_OK,"tail failure scenario posts");
    last = spark_stub_ibv_posted_count();
    CHECK(spark_stub_ibv_posted(first + 1u,&stale) == 0 && (stale.wr_id & 3u) == 3u &&
        spark_stub_ibv_complete(stale.wr_id,IBV_WC_RETRY_EXC_ERR) == 0,"one tail fails on the wire");
    SparkWeightdMeshDrainCq();
    CHECK(test_shipped(1u,local_rank) == 0u,
        "a failed tail retains the remaining posted work without false ACK");
    CHECK(test_post_slot(1u,local_rank,seq + 1u,all_peers) == SPARK_STATUS_IO_ERROR,
        "a failed transfer fences new source ownership while draining");
    test_complete_range(first + 2u,last);
    CHECK(test_shipped(1u,local_rank) == 0u &&
        weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RECORD_OF(SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank,seq)].pending == 0u,
        "failed generation drains all accepted work but does not advertise success");

    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(2u,local_rank,seq,all_peers) == SPARK_STATUS_OK,"CQ failure scenario posts");
    last = spark_stub_ibv_posted_count();
    CHECK(spark_stub_ibv_posted(first,&stale) == 0,"capture failing WR identity");
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_RETRY_EXC_ERR) == 0,"terminal transport error injected");
    SparkWeightdMeshDrainCq();
    test_complete_range(first + 1u,last);
    CHECK(test_shipped(2u,local_rank) == 0u &&
        weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RECORD_OF(2u * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank,seq)].failed != 0u,
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
    for ( i = 0u; i < TEST_MESH_PEERS; i++ )
        test_complete_range(first + i * SPARK_WEIGHTD_MESH_SEND_CAPACITY,first + i * SPARK_WEIGHTD_MESH_SEND_CAPACITY + 2u);
    CHECK(test_post_slot(7u,local_rank,seq + 4u,all_peers) == SPARK_STATUS_OK,
        "two completed sends on every peer make capacity retry useful");
    for ( i = 0u; i < TEST_MESH_PEERS; i++ )
        test_complete_range(first + i * SPARK_WEIGHTD_MESH_SEND_CAPACITY + 2u,first + (i + 1u) * SPARK_WEIGHTD_MESH_SEND_CAPACITY);
    test_complete_range(last,spark_stub_ibv_posted_count());
    for ( i = 0u; i < TEST_MESH_PEERS; i++ )
        CHECK(weightd_mesh.send_pending[i] == 0u,"all accepted SQ ownership returns after terminal completions");
}

static void test_expect_route(uint32_t first, uint32_t last, uint64_t slot_base,
    SparkWeightdMeshRoute route, uint32_t local_rank, uint64_t bytes, const char *label)
{
    uint32_t i, peer, peer_rank, data = 0u, tails = 0u;
    uint64_t offset, length;
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,label);
        for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
            if ( weightd_mesh.send_qps[peer]->qp_num == work.qp_number ) break;
        peer_rank = test_peer_rank(peer,local_rank);
        offset = SparkWeightdMeshRouteSpan(route,peer_rank,local_rank,bytes,&length);
        if ( (work.wr_id & 3u) == 3u )
        {
            tails++;
            CHECK((work.flags & IBV_SEND_INLINE) != 0u && work.length == 8u &&
                work.inline_word == *(const uint64_t *)(uintptr_t)(slot_base + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u),label);
            continue;
        }
        data++;
        if ( length <= weightd_mesh.send_queues[peer].max_inline )
            CHECK(length != 0u && (work.flags & IBV_SEND_INLINE) != 0u && work.length == length &&
                work.remote - weightd_mesh.qp_info[peer].remote_addr ==
                    slot_base + offset - (uint64_t)(uintptr_t)weightd_mesh.recv_buffer,label);
        else
        {
            CHECK(length != 0u && work.source == slot_base + offset && work.length == length,label);
            CHECK(work.remote - weightd_mesh.qp_info[peer].remote_addr ==
                work.source - (uint64_t)(uintptr_t)weightd_mesh.recv_buffer,label);
        }
    }
    CHECK(tails == TEST_MESH_PEERS,label);
    CHECK(data <= TEST_MESH_PEERS,label);
}

static SparkWeightdMeshWaitRequest *test_wait_request(uint32_t band,uint32_t rank);
static uint64_t test_wait_publish(SparkWeightdMeshWaitRequest *request,
    uint64_t kind,uint64_t tag,uint64_t mask,uint64_t cancel);

static uint64_t test_half_slot_base(uint32_t band, uint32_t rank, uint64_t seq)
{
    return ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + (uint64_t)rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
        ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u))) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
}

static void test_half_slot_publish(uint32_t band, uint32_t rank, uint64_t seq, uint64_t bytes, uint32_t peers)
{
    uint32_t half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq);
    uint8_t *slot = (uint8_t *)weightd_mesh.recv_buffer + test_half_slot_base(band,rank,seq);
    volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band,rank)) + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(seq);
    SparkWeightdMeshRoute route = {0};
    route.fields.peer_mask = peers;
    route.fields.half = half;
    memset(slot + SPARK_WEIGHTD_MESH_HALF_OFFSET(half),(int)(seq & 0xffu),bytes);
    *(volatile uint64_t *)(slot + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half)) = seq;
    entry[1] = bytes;
    entry[2] = (uint64_t)rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    entry[3] = route.word;
    __sync_synchronize();
    entry[0] = seq;
    __sync_synchronize();
}

static void test_expect_half_chunks(uint32_t first, uint32_t last, uint32_t band, uint32_t local_rank,
    uint64_t lowest, uint32_t chunks, const char *label)
{
    uint32_t i, peer, tails[2] = {0u,0u}, data[2] = {0u,0u}, newest_seen[TEST_MESH_PEERS];
    memset(newest_seen,0,sizeof(newest_seen));
    CHECK(last - first == chunks * 2u * TEST_MESH_PEERS,label);
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        uint32_t chunk, half;
        uint64_t seq, slot_base;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,label);
        for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
            if ( weightd_mesh.send_qps[peer]->qp_num == work.qp_number ) break;
        CHECK(peer < TEST_MESH_PEERS,label);
        if ( peer >= TEST_MESH_PEERS ) continue;
        chunk = (uint32_t)(((work.wr_id >> 6u) & 1u) ^ (lowest & 1u));
        CHECK(chunk < chunks && (uint32_t)((work.wr_id >> 7u) & 511u) == band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank,label);
        CHECK(chunk >= newest_seen[peer],label);
        newest_seen[peer] = chunk;
        seq = lowest + chunk;
        half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq);
        slot_base = test_half_slot_base(band,local_rank,seq);
        if ( (work.wr_id & 3u) == 3u )
        {
            tails[chunk]++;
            CHECK((work.flags & IBV_SEND_INLINE) != 0u && work.length == 8u && work.inline_word == seq &&
                work.remote - weightd_mesh.qp_info[peer].remote_addr == slot_base + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half),label);
            continue;
        }
        data[chunk]++;
        CHECK(work.length == 4096u &&
            work.source == (uint64_t)(uintptr_t)weightd_mesh.recv_buffer + slot_base + SPARK_WEIGHTD_MESH_HALF_OFFSET(half) &&
            work.remote - weightd_mesh.qp_info[peer].remote_addr == slot_base + SPARK_WEIGHTD_MESH_HALF_OFFSET(half) &&
            *(const uint8_t *)(uintptr_t)work.source == (uint8_t)(seq & 0xffu),label);
    }
    for ( i = 0u; i < chunks; i++ )
        CHECK(tails[i] == TEST_MESH_PEERS && data[i] == TEST_MESH_PEERS,label);
}

static uint32_t test_chunk_range(uint32_t first, uint32_t last, uint64_t seq, uint32_t *selected)
{
    uint32_t i, count = 0u;
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        assert(spark_stub_ibv_posted(i,&work) == 0);
        if ( ((work.wr_id >> 6u) & 1u) == (seq & 1u) )
            selected[count++] = i;
    }
    return count;
}

static void test_complete_selected(const uint32_t *selected, uint32_t count, uint32_t failed_first)
{
    uint32_t i;
    for ( i = 0u; i < count; i++ )
    {
        SparkStubIbvPostedWork work;
        assert(spark_stub_ibv_posted(selected[i],&work) == 0);
        assert(spark_stub_ibv_complete(work.wr_id,failed_first != 0u && i == 0u ? IBV_WC_RETRY_EXC_ERR : IBV_WC_SUCCESS) == 0);
        if ( (i + 1u) % 64u == 0u ) SparkWeightdMeshDrainCq();
    }
    SparkWeightdMeshDrainCq();
}

static void test_half_slots(uint32_t local_rank)
{
    const uint32_t band = 2u, index = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank;
    uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) & ~(1u << local_rank);
    uint64_t seq = (UINT64_C(11) << 32u) | 1u, bytes = 4096u;
    uint32_t first, middle, last, count, selected[4u * TEST_MESH_PEERS];
    uint32_t positions[2u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + 1u], chunk, other, distinct;
    SparkWeightdMeshRoute route = {0};
    SparkWeightdMeshTopology fresh = test_identity_topology(16u,local_rank);
    test_complete_range(0u,spark_stub_ibv_posted_count());
    CHECK(SparkWeightdMeshLaneConfigure(1u,&fresh) == SPARK_STATUS_OK,"the half-slot lane starts from zero cells");
    CHECK((SPARK_WEIGHTD_MESH_CAPABILITIES & SPARK_WEIGHTD_MESH_CAPABILITY_HALF_SLOTS) != 0u,
        "weightd advertises half-slot routes");
    CHECK(SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES == 524288u &&
        SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_ROUTE_HALF_UPPER) == 524288u &&
        SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_ROUTE_HALF_LOWER) == 0u &&
        SPARK_WEIGHTD_MESH_TAIL_OFFSET(SPARK_WEIGHTD_MESH_ROUTE_HALF_LOWER) == SPARK_WEIGHTD_MESH_SLOT_BYTES - 16u &&
        SPARK_WEIGHTD_MESH_TAIL_OFFSET(SPARK_WEIGHTD_MESH_ROUTE_HALF_UPPER) == SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u &&
        SPARK_WEIGHTD_MESH_TAIL_OFFSET(SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE) == SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u,
        "both halves and their tails share one receive slot without overlapping a whole-slot payload");
    for (chunk = 0u; chunk <= 2u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK; chunk++)
        positions[chunk] = (uint32_t)(((seq + chunk - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u)) * 2u +
            (SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq + chunk) == SPARK_WEIGHTD_MESH_ROUTE_HALF_UPPER ? 1u : 0u));
    distinct = 1u;
    for (chunk = 0u; chunk < 2u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK; chunk++)
        for (other = chunk + 1u; other < 2u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK; other++)
            distinct &= positions[chunk] != positions[other] ? 1u : 0u;
    CHECK(distinct != 0u && positions[2u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK] == positions[0] &&
        SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq) == SPARK_WEIGHTD_MESH_ROUTE_HALF_LOWER &&
        SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq + SPARK_WEIGHTD_MESH_SLOTS_PER_RANK) == SPARK_WEIGHTD_MESH_ROUTE_HALF_UPPER,
        "consecutive half chunks take every slot and half of a rank's ring once before the first position comes back");
    route.fields.peer_mask = all_peers;
    route.fields.half = SPARK_WEIGHTD_MESH_ROUTE_HALF_LOWER;
    CHECK(route.word == ((uint64_t)all_peers | (UINT64_C(1) << 58u)),"the half code sits above the route mode bits");
    CHECK(test_post_route(band,local_rank,seq,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES + 8u,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a half route carries at most one half payload");
    route.fields.half = 3u;
    CHECK(test_post_route(band,local_rank,seq,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "the fourth half code is rejected");
    route.fields.half = SPARK_WEIGHTD_MESH_ROUTE_HALF_UPPER;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_GATHER;
    route.fields.slice_bytes = 64u;
    CHECK(test_post_route(band,local_rank,seq,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a half route is a whole-payload copy, never a slice route");

    test_complete_range(0u,spark_stub_ibv_posted_count());
    first = spark_stub_ibv_posted_count();
    test_half_slot_publish(band,local_rank,seq,bytes,all_peers);
    test_half_slot_publish(band,local_rank,seq + 1u,bytes,all_peers);
    SparkWeightdMeshDoorbellPoll();
    last = spark_stub_ibv_posted_count();
    test_expect_half_chunks(first,last,band,local_rank,seq,2u,
        "two half chunks in the two doorbell cells are in flight together, older first on every queue");
    CHECK(weightd_mesh.issued[index] == seq + 1u && test_shipped(band,local_rank) == 0u,
        "posting both half chunks releases neither source");
    CHECK(SparkWeightdMeshHasDoorbellWork() == 0u,"posted cells are not doorbell work");
    count = test_chunk_range(first,last,seq + 1u,selected);
    test_complete_selected(selected,count,0u);
    CHECK(test_shipped(band,local_rank) == 0u && weightd_mesh.doorbell_posted[index] == 0u,
        "a newer chunk that drains first never advances the shipped word past an older pending chunk");
    test_half_slot_publish(band,local_rank,seq + 2u,bytes,all_peers);
    SparkWeightdMeshDoorbellPoll();
    CHECK(spark_stub_ibv_posted_count() == last,"a third chunk waits while its record still carries a pending chunk");
    count = test_chunk_range(first,last,seq,selected);
    test_complete_selected(selected,count,0u);
    CHECK(test_shipped(band,local_rank) == seq + 1u && weightd_mesh.doorbell_posted[index] == seq + 1u,
        "the older chunk's completion advances the shipped word over both chunks at once");
    SparkWeightdMeshDoorbellPoll();
    middle = spark_stub_ibv_posted_count();
    test_expect_half_chunks(last,middle,band,local_rank,seq + 2u,1u,"the third chunk posts once its record is free, into the upper half");
    test_complete_range(last,middle);
    CHECK(test_shipped(band,local_rank) == seq + 2u,"in-order completion advances the shipped word one chunk at a time");
    CHECK(SparkWeightdMeshHasDoorbellWork() == 0u && SparkWeightdMeshDoorbellShippable(index) == 0u,
        "an older chunk left in the alternate cell is neither work nor shippable");

    first = spark_stub_ibv_posted_count();
    test_half_slot_publish(band,local_rank,seq + 4u,bytes,all_peers);
    test_half_slot_publish(band,local_rank,seq + 3u,bytes,all_peers);
    CHECK(SparkWeightdMeshHasDoorbellWork() == 1u && SparkWeightdMeshDoorbellShippable(index) == 1u,
        "unposted chunks in either cell are doorbell work");
    SparkWeightdMeshDoorbellPoll();
    last = spark_stub_ibv_posted_count();
    test_expect_half_chunks(first,last,band,local_rank,seq + 3u,2u,"the lower sequence posts first whichever cell holds it");
    count = test_chunk_range(first,last,seq + 4u,selected);
    test_complete_selected(selected,count,1u);
    count = test_chunk_range(first,last,seq + 3u,selected);
    test_complete_selected(selected,count,0u);
    CHECK(test_shipped(band,local_rank) == seq + 3u,"the older chunk ships although the newer one failed");
    {
        SparkWeightdMeshWaitRequest *request = test_wait_request(band,local_rank);
        uint64_t cancel = *(const uint64_t *)((const uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
            (uint64_t)(SPARK_WEIGHTD_MESH_DOORBELL_CELL_CANCEL + 2u * band) * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
        CHECK(SparkWeightdMeshSetActivity(1u,1u) == SPARK_STATUS_OK,"half-slot lane producer begins");
        (void)test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,seq + 4u,0u,cancel);
        SparkWeightdMeshWaitRequestsPoll(SparkMonotonicNs());
        CHECK(request->ready == 1u && request->error == UINT64_MAX,"a shipped wait on the failed chunk reports the failure");
        CHECK(SparkWeightdMeshSetActivity(1u,0u) == SPARK_STATUS_OK,"half-slot lane producer ends");
    }
    test_half_slot_publish(band,local_rank,seq + 5u,bytes,all_peers);
    SparkWeightdMeshDoorbellPoll();
    CHECK(spark_stub_ibv_posted_count() == last,"a failed record fences every later chunk of its cell");
    {
        SparkWeightdMeshTopology topology = test_identity_topology(16u,local_rank);
        CHECK(SparkWeightdMeshLaneConfigure(1u,&topology) == SPARK_STATUS_OK &&
            SparkWeightdMeshCellFailed(index) == 0u && weightd_mesh.issued[index] == 0u &&
            SparkWeightdMeshDoorbellAt(index,0u)[0] == 0u && SparkWeightdMeshDoorbellAt(index,1u)[0] == 0u,
            "re-configuring the idle lane clears both cells, both records and the issue mark");
    }
}

typedef struct TestDoorbellRaceContext
{
    uint32_t band;
    uint32_t local_rank;
    uint32_t all_peers;
    uint32_t pairs;
    uint64_t base_seq;
    _Atomic uint32_t stop;
} TestDoorbellRaceContext;

static void *test_doorbell_stable_writer(void *argument)
{
    const TestDoorbellRaceContext *ctx = (const TestDoorbellRaceContext *)argument;
    uint64_t seq = ctx->base_seq;
    while ( atomic_load_explicit(&ctx->stop,memory_order_acquire) == 0u )
    {
        test_half_slot_publish(ctx->band,ctx->local_rank,seq,4096u,ctx->all_peers);
        test_half_slot_publish(ctx->band,ctx->local_rank,seq + 1u,4096u,ctx->all_peers);
        seq += 2u;
    }
    return 0;
}

static void *test_doorbell_race_writer(void *argument)
{
    const TestDoorbellRaceContext *ctx = (const TestDoorbellRaceContext *)argument;
    uint32_t pair;
    for ( pair = 0u; pair < ctx->pairs; pair++ )
    {
        uint64_t first = ctx->base_seq + (uint64_t)pair * 2u;
        test_half_slot_publish(ctx->band,ctx->local_rank,first,4096u,ctx->all_peers);
        test_half_slot_publish(ctx->band,ctx->local_rank,first + 1u,4096u,ctx->all_peers);
        while ( test_shipped(ctx->band,ctx->local_rank) < first + 1u )
            sched_yield();
    }
    return 0;
}

static void test_doorbell_pair_race(uint32_t local_rank)
{
    SparkWeightdMeshTopology topology = test_identity_topology(16u,local_rank);
    TestDoorbellRaceContext ctx;
    pthread_t writer;
    uint32_t index = 2u * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank;
    uint32_t skipped = 0u, completed = 0u, first_work, samples = 0u, torn = 0u;
    uint64_t expected_issued, snapshot[2][4];
    ctx.band = 2u;
    ctx.local_rank = local_rank;
    ctx.all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) & ~(1u << local_rank);
    ctx.pairs = 64u;
    ctx.base_seq = (UINT64_C(0xfcb31) << 32u) | 584u;
    atomic_store(&ctx.stop,0u);
    CHECK(SparkWeightdMeshLaneConfigure(1u,&topology) == SPARK_STATUS_OK,"doorbell pair race lane configured");
    CHECK(pthread_create(&writer,0,test_doorbell_stable_writer,&ctx) == 0,"doorbell stable writer started");
    while ( samples < 20000u )
    {
        if ( SparkWeightdMeshDoorbellStable(index,snapshot) == 0u )
            continue;
        if ( snapshot[1][0] >= ctx.base_seq + 1u &&
             (snapshot[0][0] + 1u < snapshot[1][0] || snapshot[0][0] > snapshot[1][0] + 1u) )
            torn++;
        samples++;
    }
    atomic_store_explicit(&ctx.stop,1u,memory_order_release);
    CHECK(pthread_join(writer,0) == 0,"doorbell stable writer joined");
    CHECK(torn == 0u,"two-cell doorbell snapshot never observes an odd round without its preceding even round");
    memset((void *)SparkWeightdMeshDoorbellAt(index,0u),0,4u * sizeof(uint64_t));
    memset((void *)SparkWeightdMeshDoorbellAt(index,1u),0,4u * sizeof(uint64_t));
    CHECK(SparkWeightdMeshLaneConfigure(1u,&topology) == SPARK_STATUS_OK,"doorbell pair race lane reset after stable stress");
    test_complete_range(0u,spark_stub_ibv_posted_count());
    first_work = spark_stub_ibv_posted_count();
    expected_issued = ctx.base_seq - 1u;
    CHECK(pthread_create(&writer,0,test_doorbell_race_writer,&ctx) == 0,"doorbell pair race writer started");
    while ( completed < ctx.pairs * 2u )
    {
        uint32_t before = spark_stub_ibv_posted_count(), after, posted_chunks;
        SparkWeightdMeshDoorbellPoll();
        after = spark_stub_ibv_posted_count();
        if ( after == before )
            continue;
        posted_chunks = (after - before) / (2u * TEST_MESH_PEERS);
        if ( (after - before) != posted_chunks * 2u * TEST_MESH_PEERS ||
             weightd_mesh.issued[index] != expected_issued + posted_chunks )
            skipped++;
        expected_issued = weightd_mesh.issued[index];
        completed += posted_chunks;
        test_complete_range(before,after);
    }
    CHECK(pthread_join(writer,0) == 0,"doorbell pair race writer joined");
    CHECK(skipped == 0u &&
        spark_stub_ibv_posted_count() - first_work == ctx.pairs * 2u * 2u * TEST_MESH_PEERS &&
        test_shipped(ctx.band,local_rank) == ctx.base_seq + (uint64_t)ctx.pairs * 2u - 1u,
        "concurrent two-cell even-then-odd doorbell pairs never skip the lower sequence");
    CHECK(SparkWeightdMeshLaneConfigure(1u,&topology) == SPARK_STATUS_OK,"doorbell pair race lane reset");
}

static void test_slice_routes(uint32_t local_rank)
{
    uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) &
        ~(1u << local_rank);
    uint64_t seq = (UINT64_C(9) << 32u) | 1u, bytes = 1000u, slot_base;
    uint32_t first, last;
    SparkWeightdMeshRoute route = {0};
    const SparkWeightdMeshWaitRequest *request = (const SparkWeightdMeshWaitRequest *)(
        (const uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_ENTRY(0u,local_rank));
    CHECK(request->capabilities == SPARK_WEIGHTD_MESH_CAPABILITIES,
        "configured lane advertises slice routes to its drivers");
    slot_base = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer +
        ((uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK) *
        SPARK_WEIGHTD_MESH_SLOT_BYTES;
    route.fields.peer_mask = all_peers;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
    route.fields.slice_bytes = 68u;
    CHECK(test_post_route(0u,local_rank,seq,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "slice widths must keep 8-byte alignment");
    route.fields.slice_bytes = 64u;
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_route(0u,local_rank,seq,bytes,route) == SPARK_STATUS_OK,"scatter route posts");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u,"scatter posts one slice and one tail per peer");
    test_expect_route(first,last,slot_base,route,local_rank,bytes,
        "scatter sends each peer only its own slice at the same slot offset");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq,"scatter releases after every slice and tail completes");
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_GATHER;
    route.fields.slice_bytes = 80u;
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_route(0u,local_rank,seq + 1u,bytes,route) == SPARK_STATUS_OK,"gather route posts");
    last = spark_stub_ibv_posted_count();
    test_expect_route(first,last,slot_base + SPARK_WEIGHTD_MESH_SLOT_BYTES,route,local_rank,bytes,
        "gather sends the local slice to every peer");
    CHECK(last - first == (local_rank * 80u < bytes ? 2u : 1u) * TEST_MESH_PEERS,
        "an empty gather slice still delivers the tail");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq + 1u,"gather releases after its tails complete");
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
    route.fields.pair = 1u;
    CHECK(test_post_route(0u,local_rank,seq + 2u,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a pair route carries the whole payload, never slices");
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    route.fields.slice_bytes = 0u;
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_route(0u,local_rank,seq + 2u,bytes,route) == SPARK_STATUS_OK,"pair route posts");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS + 1u,"a pair route sends one payload and a tail to every peer");
    test_expect_route(first,last,slot_base + (uint64_t)(2u & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u)) * SPARK_WEIGHTD_MESH_SLOT_BYTES,route,local_rank,bytes,
        "a pair route sends the payload only to the rank it is paired with");
    {
        uint64_t length = 0u;
        CHECK(SparkWeightdMeshRouteSpan(route,local_rank ^ 1u,local_rank,bytes,&length) == 0u && length == bytes,
            "the paired rank receives the whole payload");
        CHECK(SparkWeightdMeshRouteSpan(route,(local_rank + 2u) % SPARK_WEIGHTD_MESH_RANKS_PER_BAND,local_rank,bytes,&length) == 0u && length == 0u,
            "every other rank receives the tail alone");
    }
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq + 2u,"a pair route releases after its payload and tails complete");
    route.fields.pair = 0u;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = 80u;
    CHECK(test_post_route(0u,local_rank,seq + 3u,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a peer route must carry exactly its per-peer length");
    route.fields.reserved = 1u;
    route.fields.slice_bytes = 1000u;
    CHECK(test_post_route(0u,local_rank,seq + 3u,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "reserved route bits are rejected");
}

static uint64_t test_peer_route_bytes(uint32_t rows)
{
    uint64_t bytes = (uint64_t)rows * 4112u * sizeof(uint16_t);
    return bytes < SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES ? bytes : SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES;
}

static void test_peer_routes(uint32_t local_rank)
{
    static const uint32_t rows[] = {1u,8u,128u,129u,512u,1024u,2048u,4096u};
    uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) & ~(1u << local_rank);
    uint64_t seq = (UINT64_C(10) << 32u) | 1u, slot_base, staging = (uint64_t)(uintptr_t)weightd_mesh.staging_buffer;
    uint32_t case_index, first, last, i, peer, peer_rank, data, tails;
    SparkWeightdMeshRoute route = {0};
    const SparkWeightdMeshWaitRequest *request = (const SparkWeightdMeshWaitRequest *)(
        (const uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_ENTRY(0u,local_rank));
    int fd = SparkWeightdMeshStagingFd();
    struct stat info;
    CHECK((request->capabilities & SPARK_WEIGHTD_MESH_CAPABILITY_PEER_ROUTES) != 0u &&
        (request->capabilities & SPARK_WEIGHTD_MESH_CAPABILITY_SLICE_ROUTES) != 0u,
        "configured lane advertises peer routes next to slice routes");
    CHECK(fd >= 0 && fstat(fd,&info) == 0 && (uint64_t)info.st_size == SPARK_WEIGHTD_MESH_STAGING_BYTES,
        "the staging export is one fd of exactly the staging bytes");
    if ( fd >= 0 ) (void)close(fd);
    CHECK(SPARK_WEIGHTD_MESH_STAGING_BYTES == UINT64_C(134217728) &&
        SPARK_WEIGHTD_MESH_REGION_BYTES == UINT64_C(536903680) + SPARK_WEIGHTD_MESH_DOORBELL_BYTES,
        "per-peer staging is 128 MiB and the region holds four 1 MiB slots a rank");
    route.fields.peer_mask = all_peers;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES + 8u;
    CHECK(test_post_route(0u,local_rank,seq,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES + 8u,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a peer route never exceeds one staging slot");
    route.fields.slice_bytes = 12u;
    CHECK(test_post_route(0u,local_rank,seq,12u,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "peer routes keep 8-byte alignment");
    for ( case_index = 0u; case_index < sizeof(rows) / sizeof(rows[0]); case_index++ )
    {
        uint64_t bytes = test_peer_route_bytes(rows[case_index]);
        CHECK(rows[case_index] <= SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS,"the mesh row cap admits every tested exchange height up to 4096 rows");
        uint64_t slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        slot_base = (uint64_t)slot * SPARK_WEIGHTD_MESH_SLOT_BYTES;
        route.fields.slice_bytes = bytes;
        first = spark_stub_ibv_posted_count();
        CHECK(test_post_route(0u,local_rank,seq,bytes,route) == SPARK_STATUS_OK,"peer route posts");
        last = spark_stub_ibv_posted_count();
        CHECK(last - first == TEST_MESH_PEERS * 2u,"peer route posts one staged payload and one tail per peer");
        data = tails = 0u;
        for ( i = first; i < last; i++ )
        {
            SparkStubIbvPostedWork work;
            CHECK(spark_stub_ibv_posted(i,&work) == 0,"peer route work is recorded");
            for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
                if ( weightd_mesh.send_qps[peer]->qp_num == work.qp_number ) break;
            peer_rank = test_peer_rank(peer,local_rank);
            CHECK(work.remote - weightd_mesh.qp_info[peer].remote_addr == slot_base + ((work.wr_id & 3u) == 3u ? SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u : 0u),
                "peer route lands at the start of this rank's receive slot and its tail at the slot end");
            if ( (work.wr_id & 3u) == 3u )
            {
                tails++;
                CHECK((work.flags & IBV_SEND_INLINE) != 0u && work.length == 8u &&
                    work.inline_word == *(const uint64_t *)((const uint8_t *)weightd_mesh.recv_buffer + slot_base + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u),
                    "the tail carries the tag of the sender's own slot inline");
                continue;
            }
            data++;
            CHECK(work.lkey == weightd_mesh.staging_mr->lkey && work.length == bytes &&
                work.source == staging + SPARK_WEIGHTD_MESH_STAGING_OFFSET(0u,peer_rank),
                "each peer receives its own full staging slot");
        }
        CHECK(data == TEST_MESH_PEERS && tails == TEST_MESH_PEERS,"every peer gets one payload and one tail");
        CHECK(test_shipped(0u,local_rank) != seq,"the peer route is not shipped before its completions");
        test_complete_range(first,last);
        CHECK(test_shipped(0u,local_rank) == seq,"peer route releases after every payload and tail completes");
        seq++;
    }
    route.fields.peer_mask = all_peers & (all_peers - 1u);
    route.fields.slice_bytes = 4096u;
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_route(0u,local_rank,seq,4096u,route) == SPARK_STATUS_OK,"partial peer masks post");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == (TEST_MESH_PEERS - 1u) * 2u,"a peer route skips peers outside its mask");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq,"partial peer route releases");
}

static uint32_t test_parity_mask(uint32_t local_rank)
{
    uint32_t rank, mask = 0u;
    for ( rank = local_rank & 1u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank += 2u )
        mask |= 1u << rank;
    return mask;
}

static void test_peer_halves(uint32_t local_rank)
{
    uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) & ~(1u << local_rank);
    uint64_t seq = (UINT64_C(12) << 32u) | 1u, staging = (uint64_t)(uintptr_t)weightd_mesh.staging_buffer, bytes = 4096u;
    uint32_t chunk, first, last, i, peer, peer_rank, data[2] = {0u,0u}, tails[2] = {0u,0u};
    SparkWeightdMeshRoute route = {0};
    const SparkWeightdMeshWaitRequest *request = (const SparkWeightdMeshWaitRequest *)(
        (const uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_ENTRY(0u,local_rank));
    CHECK((request->capabilities & SPARK_WEIGHTD_MESH_CAPABILITY_PEER_HALVES) != 0u &&
        (SPARK_WEIGHTD_MESH_CAPABILITIES & SPARK_WEIGHTD_MESH_CAPABILITY_PEER_HALVES) != 0u,
        "configured lane advertises half-slot peer routes");
    route.fields.peer_mask = all_peers;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.half = SPARK_WEIGHTD_MESH_ROUTE_HALF_LOWER;
    route.fields.slice_bytes = SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES + 8u;
    CHECK(test_post_route(0u,local_rank,seq,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES + 8u,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a half peer route never exceeds one half payload");
    route.fields.slice_bytes = bytes;
    CHECK(test_post_route(0u,local_rank,seq,bytes + 8u,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a half peer route carries exactly its per-peer length");
    route.fields.pair = 1u;
    CHECK(test_post_route(0u,local_rank,seq,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a half peer route is never a pair route");
    route.fields.pair = 0u;
    first = spark_stub_ibv_posted_count();
    for ( chunk = 0u; chunk < 2u; chunk++ )
    {
        uint64_t slot_base = test_half_slot_base(0u,local_rank,seq + chunk);
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base +
            SPARK_WEIGHTD_MESH_TAIL_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq + chunk))) = seq + chunk;
        route.fields.half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq + chunk);
        CHECK(test_post_route(0u,local_rank,seq + chunk,bytes,route) == SPARK_STATUS_OK,"half peer routes post");
    }
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == 2u * TEST_MESH_PEERS * 2u,"two half peer chunks post one payload and one tail per peer each");
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        uint64_t chunk_seq, slot_base;
        uint32_t half;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,"half peer route work is recorded");
        for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
            if ( weightd_mesh.send_qps[peer]->qp_num == work.qp_number ) break;
        CHECK(peer < TEST_MESH_PEERS,"half peer route work names a mesh peer");
        if ( peer >= TEST_MESH_PEERS ) continue;
        peer_rank = test_peer_rank(peer,local_rank);
        chunk = (uint32_t)(((work.wr_id >> 6u) & 1u) ^ (seq & 1u));
        chunk_seq = seq + chunk;
        half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(chunk_seq);
        slot_base = test_half_slot_base(0u,local_rank,chunk_seq);
        if ( (work.wr_id & 3u) == 3u )
        {
            tails[chunk]++;
            CHECK(work.length == 8u && work.inline_word == chunk_seq &&
                work.remote - weightd_mesh.qp_info[peer].remote_addr == slot_base + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half),
                "a half peer chunk's tail lands at its half's tail");
            continue;
        }
        data[chunk]++;
        CHECK(work.lkey == weightd_mesh.staging_mr->lkey && work.length == bytes &&
            work.source == staging + SPARK_WEIGHTD_MESH_STAGING_OFFSET(0u,peer_rank) +
                SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot_base / SPARK_WEIGHTD_MESH_SLOT_BYTES) &&
            work.remote - weightd_mesh.qp_info[peer].remote_addr == slot_base + SPARK_WEIGHTD_MESH_HALF_OFFSET(half),
            "a half peer chunk reads its staging half and lands in its receive half");
    }
    CHECK(SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(test_half_slot_base(0u,local_rank,seq) / SPARK_WEIGHTD_MESH_SLOT_BYTES) !=
        SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(test_half_slot_base(0u,local_rank,seq + 1u) / SPARK_WEIGHTD_MESH_SLOT_BYTES),
        "two consecutive half peer chunks read different staging halves, so both can be in flight");
    CHECK(data[0] == TEST_MESH_PEERS && data[1] == TEST_MESH_PEERS && tails[0] == TEST_MESH_PEERS && tails[1] == TEST_MESH_PEERS,
        "every peer gets one payload and one tail per half chunk");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq + 1u,"both half peer chunks release after their completions");
}

static void test_pair_first_routes(uint32_t local_rank)
{
    uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) & ~(1u << local_rank);
    const uint64_t seq = (UINT64_C(13) << 32u) | 1u, staging = (uint64_t)(uintptr_t)weightd_mesh.staging_buffer, group = 4096u, pair = 2048u;
    const uint64_t bytes = SPARK_WEIGHTD_MESH_PAIR_FIRST_BYTES(group,pair), slot_base = test_half_slot_base(0u,local_rank,seq);
    const uint64_t slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    const uint32_t half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq), partner = local_rank ^ 1u;
    uint32_t first, last, i, peer, peer_rank, data[SPARK_WEIGHTD_MESH_RANKS_PER_BAND], tails = 0u, pending_before = 0u, pending_after = 0u, landed = 0u;
    SparkWeightdMeshRoute route = {0};
    memset(data,0,sizeof(data));
    CHECK((SPARK_WEIGHTD_MESH_CAPABILITIES & SPARK_WEIGHTD_MESH_CAPABILITY_PAIR_FIRST) != 0u,"weightd advertises pair-first routes");
    route.fields.peer_mask = all_peers;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.pair_first = 1u;
    route.fields.half = SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE;
    CHECK(test_post_route(0u,local_rank,seq,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,"a pair-first route always names a half");
    route.fields.half = half;
    route.fields.slice_bytes = 64u;
    CHECK(test_post_route(0u,local_rank,seq,bytes,route) == SPARK_STATUS_INVALID_ARGUMENT,"a pair-first route carries its sizes in the doorbell bytes, never a slice");
    route.fields.slice_bytes = 0u;
    CHECK(test_post_route(0u,local_rank,seq,SPARK_WEIGHTD_MESH_PAIR_FIRST_BYTES(group,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES + 8u),route) == SPARK_STATUS_INVALID_ARGUMENT,
        "a pair piece never exceeds one half payload");
    CHECK(test_post_route(0u,local_rank,seq,SPARK_WEIGHTD_MESH_PAIR_FIRST_BYTES(group + 4u,pair),route) == SPARK_STATUS_INVALID_ARGUMENT,
        "pair-first pieces keep 8-byte alignment");
    for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
        pending_before += weightd_mesh.send_pending[peer];
    *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half)) = seq;
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_route(0u,local_rank,seq,bytes,route) == SPARK_STATUS_OK,"a pair-first route posts");
    last = spark_stub_ibv_posted_count();
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,"pair-first work is recorded");
        for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
            if ( weightd_mesh.send_qps[peer]->qp_num == work.qp_number ) break;
        CHECK(peer < TEST_MESH_PEERS,"pair-first work names a mesh peer");
        if ( peer >= TEST_MESH_PEERS ) continue;
        peer_rank = test_peer_rank(peer,local_rank);
        if ( (work.wr_id & 3u) == 3u )
        {
            tails++;
            CHECK(work.length == 8u && work.inline_word == seq &&
                work.remote - weightd_mesh.qp_info[peer].remote_addr == slot_base + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half),
                "every peer gets this rank's tail in this rank's own area");
            continue;
        }
        data[peer_rank]++;
        if ( peer_rank == partner )
        {
            uint32_t segment = (uint32_t)((work.source - staging) / SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
            CHECK(work.lkey == weightd_mesh.staging_mr->lkey && work.length == pair && (segment & 1u) == (partner & 1u) &&
                work.source == staging + SPARK_WEIGHTD_MESH_STAGING_OFFSET(0u,segment) + SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot) &&
                work.remote - weightd_mesh.qp_info[peer].remote_addr == SparkWeightdMeshPairFirstArea(0u,segment ^ 1u,slot,half),
                "each pair piece reads the staging half of a partner-parity segment and lands in the area of that segment's partner");
            landed |= 1u << (segment ^ 1u);
        }
        else
            CHECK(((peer_rank ^ local_rank) & 1u) == 0u && work.length == group &&
                work.source == staging + SPARK_WEIGHTD_MESH_STAGING_OFFSET(0u,peer_rank) + SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot) &&
                work.remote - weightd_mesh.qp_info[peer].remote_addr == slot_base + SPARK_WEIGHTD_MESH_HALF_OFFSET(half),
                "a group piece goes only to a same-parity peer, from that peer's staging half into this rank's area");
    }
    for ( peer_rank = 0u; peer_rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer_rank++ )
        if ( peer_rank != local_rank )
            CHECK(data[peer_rank] == (peer_rank == partner ? SPARK_WEIGHTD_MESH_RANKS_PER_BAND / 2u : ((peer_rank ^ local_rank) & 1u) == 0u ? 1u : 0u),
                "the partner gets one piece per partner-parity segment, a group peer one piece, every other peer only the tail");
    CHECK(tails == TEST_MESH_PEERS,"every peer gets one tail");
    CHECK(landed == test_parity_mask(local_rank),"the pair pieces land in the areas of every same-parity sender, this rank's own area included");
    CHECK(test_shipped(0u,local_rank) != seq,"a pair-first round is not shipped before its completions");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq,"a pair-first round releases after its tails complete");
    for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
        pending_after += weightd_mesh.send_pending[peer];
    CHECK(pending_after == pending_before,"completing the tails retires every unsignaled piece from the send queues");
}

static void test_permutation_schedule(uint32_t local_rank)
{
    const uint32_t degrees[] = {2u, 3u, 4u, 5u, 6u, 7u, 8u, 12u, 16u};
    const uint32_t saved_rank = weightd_mesh.local_rank;
    const SparkWeightdMeshTopology saved_lane = weightd_mesh.lane_topology[2u];
    const uint32_t band = 4u;
    uint32_t d, r, s, mode;
    uint32_t targets[SPARK_WEIGHTD_MESH_RANKS_PER_BAND][SPARK_WEIGHTD_MESH_PEERS];

    CHECK(SparkWeightdMeshSchedulePeer(0u, 1u, 0u) == 0u &&
        SparkWeightdMeshSchedulePeer(2u, 4u, 0u) == 2u &&
        SparkWeightdMeshSchedulePeer(2u, 4u, 4u) == 2u &&
        SparkWeightdMeshSchedulePeer(5u, 4u, 1u) == 5u,
        "schedule helper guards degenerate degree, step and rank bounds");

    for ( d = 0u; d < sizeof(degrees) / sizeof(degrees[0]); d++ )
    {
        const uint32_t degree = degrees[d];
        const uint32_t full_mask = (1u << degree) - 1u;
        const uint32_t pow2 = (degree & (degree - 1u)) == 0u ? 1u : 0u;
        for ( s = 1u; s < degree; s++ )
        {
            uint32_t seen = 0u;
            for ( r = 0u; r < degree; r++ )
            {
                const uint32_t target = SparkWeightdMeshSchedulePeer(r, degree, s);
                CHECK(target < degree && target != r, "scheduled peer is a valid distinct rank");
                CHECK((seen & (1u << target)) == 0u, "zero incast: no two ranks share a target at the same step");
                seen |= 1u << target;
                if ( pow2 != 0u )
                {
                    CHECK(SparkWeightdMeshSchedulePeer(target, degree, s) == r,
                        "power-of-two schedule pairs bidirectional flows symmetrically");
                    CHECK(((r ^ target) & 1u) == (s & 1u),
                        "power-of-two schedule alternates rail parity cleanly at every step");
                    if ( s == 1u )
                        CHECK(target == (r ^ 1u), "step 1 targets pair partner first");
                }
            }
            CHECK(seen == full_mask, "target mapping is a bijection across all ranks at every step");
        }
        for ( r = 0u; r < degree; r++ )
        {
            uint32_t visited = 1u << r;
            for ( s = 1u; s < degree; s++ )
                visited |= 1u << SparkWeightdMeshSchedulePeer(r, degree, s);
            CHECK(visited == full_mask, "every rank reaches every peer exactly once across all steps");
        }
    }

    for ( mode = 0u; mode < 6u; mode++ )
    {
        memset(targets, 0, sizeof(targets));
        for ( r = 0u; r < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; r++ )
        {
            SparkWeightdMeshTopology topo = test_identity_topology(SPARK_WEIGHTD_MESH_RANKS_PER_BAND, r);
            uint32_t all_peers = 0xffffu & ~(1u << r);
            uint32_t first;
            weightd_mesh.local_rank = r;
            CHECK(SparkWeightdMeshLaneConfigure(band / 2u, &topo) == SPARK_STATUS_OK,
                "permutation test configures lane for each rank");
            first = spark_stub_ibv_posted_count();
            if ( mode == 5u )
            {
                CHECK(SparkWeightdMeshBroadcast(all_peers, 0u, 64u, 0u, 0u, 0u) == TEST_MESH_PEERS,
                    "broadcast posts to all 15 peers");
            }
            else
            {
                SparkWeightdMeshRoute route = {0};
                uint64_t seq = ((uint64_t)(20u + mode) << 32u) | 1u;
                uint64_t bytes = 4096u;
                route.fields.peer_mask = all_peers;
                if ( mode == 1u )
                {
                    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
                    route.fields.slice_bytes = 256u;
                }
                else if ( mode == 2u )
                {
                    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_GATHER;
                    route.fields.slice_bytes = 256u;
                }
                else if ( mode == 3u )
                {
                    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
                    route.fields.slice_bytes = 4096u;
                }
                else if ( mode == 4u )
                {
                    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
                    route.fields.pair_first = 1u;
                    route.fields.half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(seq);
                    bytes = SPARK_WEIGHTD_MESH_PAIR_FIRST_BYTES(2048u, 1024u);
                }
                CHECK(test_post_route(band, r, seq, bytes, route) == SPARK_STATUS_OK,
                    "post route succeeds across all 16 ranks and collective modes");
            }
            CHECK(weightd_mesh.ring_count == TEST_MESH_PEERS,
                "doorbell ring covers all 15 peers");
            for ( s = 0u; s < TEST_MESH_PEERS; s++ )
                targets[r][s] = test_peer_rank(weightd_mesh.ring_order[s], r);
            test_complete_range(first, spark_stub_ibv_posted_count());
        }
        for ( s = 0u; s < TEST_MESH_PEERS; s++ )
        {
            uint32_t seen = 0u;
            for ( r = 0u; r < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; r++ )
            {
                uint32_t dst = targets[r][s];
                CHECK(dst < SPARK_WEIGHTD_MESH_RANKS_PER_BAND && dst != r,
                    "doorbell ring target is a distinct peer rank");
                CHECK((seen & (1u << dst)) == 0u,
                    "doorbell ring has zero incast across all 16 ranks at every step");
                CHECK(targets[dst][s] == r,
                    "doorbell ring pairs ranks symmetrically at every step for degree 16");
                seen |= 1u << dst;
            }
            CHECK(seen == 0xffffu,
                "doorbell ring target mapping is a 16-rank bijection at every step");
        }
    }

    {
        const uint32_t phys4[4] = {2u, 7u, 11u, 14u};
        const uint32_t phys4_mask = (1u << 2u) | (1u << 7u) | (1u << 11u) | (1u << 14u);
        uint32_t sub_targets[4][3];
        for ( r = 0u; r < 4u; r++ )
        {
            SparkWeightdMeshTopology topo;
            SparkWeightdMeshRoute route = {0};
            uint32_t first;
            memset(&topo, 0, sizeof(topo));
            topo.rank_count = 4u;
            topo.local_rank = r;
            for ( s = 0u; s < 4u; s++ ) topo.physical_ranks[s] = phys4[s];
            weightd_mesh.local_rank = phys4[r];
            CHECK(SparkWeightdMeshLaneConfigure(band / 2u, &topo) == SPARK_STATUS_OK,
                "non-identity TP4 topology configures");
            route.fields.peer_mask = 0xfu & ~(1u << r);
            first = spark_stub_ibv_posted_count();
            CHECK(test_post_route(band, r, (UINT64_C(30) << 32u) | 1u, 1024u, route) == SPARK_STATUS_OK,
                "non-identity TP4 route posts");
            CHECK(weightd_mesh.ring_count == 3u, "non-identity TP4 rings 3 peers");
            for ( s = 0u; s < 3u; s++ )
                sub_targets[r][s] = test_peer_rank(weightd_mesh.ring_order[s], phys4[r]);
            test_complete_range(first, spark_stub_ibv_posted_count());
        }
        for ( s = 0u; s < 3u; s++ )
        {
            uint32_t seen = 0u;
            for ( r = 0u; r < 4u; r++ )
            {
                CHECK((seen & (1u << sub_targets[r][s])) == 0u,
                    "non-identity TP4 has zero incast at every step");
                seen |= 1u << sub_targets[r][s];
            }
            CHECK(seen == phys4_mask, "non-identity TP4 hits all 4 physical ranks bijectively");
        }
    }

    {
        uint32_t sub_targets[6][5];
        for ( r = 0u; r < 6u; r++ )
        {
            SparkWeightdMeshTopology topo = test_identity_topology(6u, r);
            SparkWeightdMeshRoute route = {0};
            uint32_t first;
            weightd_mesh.local_rank = r;
            CHECK(SparkWeightdMeshLaneConfigure(band / 2u, &topo) == SPARK_STATUS_OK,
                "non-power-of-two TP6 topology configures");
            route.fields.peer_mask = 0x3fu & ~(1u << r);
            first = spark_stub_ibv_posted_count();
            CHECK(test_post_route(band, r, (UINT64_C(31) << 32u) | 1u, 1024u, route) == SPARK_STATUS_OK,
                "non-power-of-two TP6 route posts");
            CHECK(weightd_mesh.ring_count == 5u, "non-power-of-two TP6 rings 5 peers");
            for ( s = 0u; s < 5u; s++ )
                sub_targets[r][s] = test_peer_rank(weightd_mesh.ring_order[s], r);
            test_complete_range(first, spark_stub_ibv_posted_count());
        }
        for ( s = 0u; s < 5u; s++ )
        {
            uint32_t seen = 0u;
            for ( r = 0u; r < 6u; r++ )
            {
                CHECK((seen & (1u << sub_targets[r][s])) == 0u,
                    "non-power-of-two TP6 cyclic schedule has zero incast at every step");
                seen |= 1u << sub_targets[r][s];
            }
            CHECK(seen == 0x3fu, "non-power-of-two TP6 hits all 6 ranks bijectively");
        }
    }

    {
        uint64_t fake_bf[2][64];
        volatile uint64_t *saved_db[TEST_MESH_PEERS];
        uint32_t saved_comb[TEST_MESH_PEERS];
        uint64_t barriers_before = weightd_mesh.timing.ring_barriers;
        SparkWeightdMeshTopology topo = test_identity_topology(SPARK_WEIGHTD_MESH_RANKS_PER_BAND, 3u);
        SparkWeightdMeshRoute route = {0};
        uint32_t first;
        memset(fake_bf, 0, sizeof(fake_bf));
        weightd_mesh.local_rank = 3u;
        CHECK(SparkWeightdMeshLaneConfigure(band / 2u, &topo) == SPARK_STATUS_OK,
            "dual-rail BlueFlame batching test configures lane");
        for ( s = 0u; s < TEST_MESH_PEERS; s++ )
        {
            uint32_t pr = test_peer_rank(s, 3u);
            uint32_t rail = ((pr + 3u) & 1u) != 0u ? 1u : 0u;
            saved_db[s] = weightd_mesh.send_queues[s].doorbell;
            saved_comb[s] = weightd_mesh.send_queues[s].combining_bytes;
            weightd_mesh.send_queues[s].doorbell = &fake_bf[rail][0];
            weightd_mesh.send_queues[s].combining_bytes = 256u;
        }
        SparkWeightdMeshRebuildFastMaps();
        route.fields.peer_mask = 0xffffu & ~(1u << 3u);
        first = spark_stub_ibv_posted_count();
        CHECK(test_post_route(band, 3u, (UINT64_C(32) << 32u) | 1u, 1024u, route) == SPARK_STATUS_OK,
            "dual-rail BlueFlame route posts");
        CHECK(weightd_mesh.timing.ring_barriers - barriers_before == 3u,
            "alternating parity schedule flushes 4 doorbells per barrier across dual rails without parity gaps");
        test_complete_range(first, spark_stub_ibv_posted_count());
        for ( s = 0u; s < TEST_MESH_PEERS; s++ )
        {
            weightd_mesh.send_queues[s].doorbell = saved_db[s];
            weightd_mesh.send_queues[s].combining_bytes = saved_comb[s];
        }
        SparkWeightdMeshRebuildFastMaps();
    }

    weightd_mesh.local_rank = saved_rank;
    CHECK(SparkWeightdMeshLaneConfigure(band / 2u, &saved_lane) == SPARK_STATUS_OK,
        "permutation test restores original lane topology");
    (void)local_rank;
}

static void test_e14_rdma_verbs_fastpath(uint32_t local_rank)
{
    const uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) & ~(1u << local_rank);
    const uint32_t band = 4u;
    const uint64_t seq = (UINT64_C(40) << 32u) | 1u;
    const uint64_t slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
    const uint64_t slot_base = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    const uint64_t payload_word = UINT64_C(0x1122334455667788);
    SparkWeightdMeshRoute route = {0};
    uint32_t first, last, i, peer, inline_payloads = 0u, signaled_tails = 0u;
    uint32_t unsignaled_bcast = 0u, signaled_seq = 0u;

    route.fields.peer_mask = all_peers;
    *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base) = payload_word;
    *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u) = seq;
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_route(band, local_rank, seq, 8u, route) == SPARK_STATUS_OK,
        "e14: 8-byte small payload route posts");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u,
        "e14: 8-byte route posts chained payload and tail per peer");
    for ( i = first; i < last; i += 2u )
    {
        SparkStubIbvPostedWork pwork, twork;
        CHECK(spark_stub_ibv_posted(i, &pwork) == 0 && spark_stub_ibv_posted(i + 1u, &twork) == 0,
            "e14: paired payload and tail WRs recorded");
        if ( (pwork.wr_id & 3u) == 2u && (pwork.flags & IBV_SEND_INLINE) != 0u &&
             (pwork.flags & IBV_SEND_SIGNALED) == 0u && pwork.length == 8u &&
             pwork.inline_word == payload_word )
            inline_payloads++;
        if ( (twork.wr_id & 3u) == 3u && (twork.flags & IBV_SEND_INLINE) != 0u &&
             (twork.flags & IBV_SEND_SIGNALED) != 0u && twork.length == 8u &&
             twork.inline_word == seq && twork.qp_number == pwork.qp_number &&
             twork.wqe_index == pwork.wqe_index + 1u )
            signaled_tails++;
    }
    CHECK(inline_payloads == TEST_MESH_PEERS && signaled_tails == TEST_MESH_PEERS,
        "e14: 8-byte payloads use inline WRs chained unsignaled before signaled inline tails");
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork w;
        if ( spark_stub_ibv_posted(i, &w) == 0 && (w.flags & IBV_SEND_SIGNALED) != 0u )
            CHECK(spark_stub_ibv_complete(w.wr_id, IBV_WC_SUCCESS) == 0, "e14: tail CQE injected");
    }
    SparkWeightdMeshDrainCq();
    CHECK(test_shipped(band, local_rank) == seq,
        "e14: O(1) QPN CQ drain advances shipped on signaled tails alone");

    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshBroadcast(all_peers, 0u, 64u, 0u, seq + 7u, 64u) == TEST_MESH_PEERS,
        "e14: broadcast with seq posts to all 15 peers in Latin-square order");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u,
        "e14: broadcast with seq emits payload and seq WRs per peer");
    for ( i = 0u; i < TEST_MESH_PEERS; i++ )
    {
        uint32_t expected_rank = SparkWeightdMeshSchedulePeer(local_rank, SPARK_WEIGHTD_MESH_RANKS_PER_BAND, i + 1u);
        uint32_t expected_peer = expected_rank < local_rank ? expected_rank : expected_rank - 1u;
        CHECK(weightd_mesh.ring_order[i] == expected_peer,
            "e14: broadcast rings doorbells in conflict-free Latin-square peer order");
    }
    for ( i = first; i < last; i += 2u )
    {
        SparkStubIbvPostedWork pwork, swork;
        CHECK(spark_stub_ibv_posted(i, &pwork) == 0 && spark_stub_ibv_posted(i + 1u, &swork) == 0,
            "e14: broadcast payload and seq WRs captured");
        if ( (pwork.flags & IBV_SEND_SIGNALED) == 0u && (pwork.flags & IBV_SEND_INLINE) != 0u && pwork.length == 64u )
            unsignaled_bcast++;
        if ( (swork.flags & IBV_SEND_SIGNALED) != 0u && (swork.flags & IBV_SEND_INLINE) != 0u &&
             swork.inline_word == seq + 7u && swork.qp_number == pwork.qp_number &&
             swork.wqe_index == ((pwork.wqe_index + 2u) & 0xffffu) )
            signaled_seq++;
    }
    CHECK(unsignaled_bcast == TEST_MESH_PEERS && signaled_seq == TEST_MESH_PEERS,
        "e14: broadcast chains unsignaled payload before signaled inline seq WR");
    for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
        CHECK(weightd_mesh.send_pending[peer] == 2u && weightd_mesh.rpc_pending[peer] == 2u,
            "e14: chained broadcast tracks both payload and seq SQ occupancy");
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork w;
        if ( spark_stub_ibv_posted(i, &w) == 0 && (w.flags & IBV_SEND_SIGNALED) != 0u )
            CHECK(spark_stub_ibv_complete(w.wr_id, IBV_WC_SUCCESS) == 0, "e14: broadcast seq CQE injected");
    }
    SparkWeightdMeshDrainCq();
    for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
        CHECK(weightd_mesh.send_pending[peer] == 0u && weightd_mesh.rpc_pending[peer] == 0u,
            "e14: single signaled seq CQE retires both chained broadcast WRs");

    {
        uint64_t unique_bf[TEST_MESH_PEERS][64];
        volatile uint64_t *saved_db[TEST_MESH_PEERS];
        uint32_t saved_comb[TEST_MESH_PEERS];
        uint32_t saved_unique = weightd_mesh.bf_all_unique;
        uint64_t barriers_before = weightd_mesh.timing.ring_barriers;
        memset(unique_bf, 0, sizeof(unique_bf));
        for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
        {
            saved_db[peer] = weightd_mesh.send_queues[peer].doorbell;
            saved_comb[peer] = weightd_mesh.send_queues[peer].combining_bytes;
            weightd_mesh.send_queues[peer].doorbell = &unique_bf[peer][0];
            weightd_mesh.send_queues[peer].combining_bytes = 256u;
            weightd_mesh.send_queues[peer].bf_offset = 0u;
        }
        SparkWeightdMeshRebuildFastMaps();
        CHECK(weightd_mesh.bf_all_unique == 1u,
            "e14: distinct per-QP BlueFlame registers enable zero-barrier fast path");
        first = spark_stub_ibv_posted_count();
        CHECK(test_post_route(band, local_rank, seq + 1u, 64u, route) == SPARK_STATUS_OK,
            "e14: round 1 on unique BF registers posts");
        CHECK(weightd_mesh.timing.ring_barriers == barriers_before &&
              weightd_mesh.send_queues[0].bf_offset == 256u && unique_bf[0][0] != 0u,
            "e14: round 1 rings offset 0 with zero barriers and toggles bf_offset to 256");
        CHECK(test_post_route(band, local_rank, seq + 2u, 64u, route) == SPARK_STATUS_OK,
            "e14: round 2 on unique BF registers posts");
        CHECK(weightd_mesh.timing.ring_barriers == barriers_before &&
              weightd_mesh.send_queues[0].bf_offset == 0u && unique_bf[0][32] != 0u,
            "e14: round 2 ping-pongs to offset 256 with zero barriers and toggles back to 0");
        test_complete_range(first, spark_stub_ibv_posted_count());
        for ( peer = 0u; peer < TEST_MESH_PEERS; peer++ )
        {
            weightd_mesh.send_queues[peer].doorbell = saved_db[peer];
            weightd_mesh.send_queues[peer].combining_bytes = saved_comb[peer];
            weightd_mesh.send_queues[peer].bf_offset = 0u;
        }
        weightd_mesh.bf_all_unique = saved_unique;
    }
}

static void test_e14_r2_poll_fastpath(uint32_t local_rank)
{
    const uint32_t band = 4u;
    const uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) & ~(1u << local_rank);
    const uint64_t seq = (UINT64_C(41) << 32u) | 1u;
    const uint64_t slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
    volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band, local_rank));
    char log_path[320];
    SparkWeightdMeshRoute route = {0};
    uint32_t first, last, sweep;

    route.fields.peer_mask = all_peers;
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshHasPending() == 0u && weightd_mesh.send_pending_total == 0u,
        "e14-r2: send_pending_total starts at zero when idle");
    entry[1] = 1024u;
    entry[2] = slot;
    entry[3] = route.word;
    entry[0] = seq;
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    CHECK(SparkWeightdMeshDoorbellCell(band, local_rank, UINT64_C(1000)) == 1u,
        "e14-r2: initial doorbell sweep posts the round");
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u &&
          SparkWeightdMeshHasPending() != 0u &&
          weightd_mesh.send_pending_total == TEST_MESH_PEERS * 2u,
        "e14-r2: O(1) send_pending_total matches exact posted WR count");

    (void)snprintf(log_path, sizeof(log_path), "%s/e14-r2-no-spam.log", SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    for ( sweep = 0u; sweep < 6000u; sweep++ )
        CHECK(SparkWeightdMeshDoorbellCell(band, local_rank, UINT64_C(2000) + (uint64_t)sweep) == 0u,
            "e14-r2: in-flight sweep returns 0 without re-posting");
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    test_capture_end();
    CHECK(test_file_contains(log_path, "WD-SEEN") == 0 &&
          test_file_contains(log_path, "WD-TRANSFER-STUCK") == 0 &&
          test_file_contains(log_path, "WD-STUCK") == 0,
        "e14-r2: 6000 sub-1s in-flight sweeps emit zero stderr log spam");

    entry[1] = 1024u;
    entry[2] = slot + 1u;
    entry[3] = route.word;
    entry[0] = seq + 1u;
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    CHECK(SparkWeightdMeshDoorbellCell(band, local_rank, UINT64_C(9000)) == 1u,
        "e14-r2: second concurrent round fills the second transfer record");
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 4u &&
          weightd_mesh.send_pending_total == TEST_MESH_PEERS * 4u,
        "e14-r2: send_pending_total tracks both in-flight rounds accurately");

    entry[1] = 1024u;
    entry[2] = slot + 2u;
    entry[3] = route.word;
    entry[0] = seq + 2u;
    test_capture_begin(log_path);
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    CHECK(SparkWeightdMeshDoorbellCell(band, local_rank, UINT64_C(10000)) == 0u,
        "e14-r2: third round waits while both transfer records are in flight");
    for ( sweep = 0u; sweep < 5500u; sweep++ )
        (void)SparkWeightdMeshDoorbellCell(band, local_rank, UINT64_C(10001) + (uint64_t)sweep);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    test_capture_end();
    CHECK(test_file_contains(log_path, "WD-SEEN") == 0 &&
          test_file_contains(log_path, "WD-TRANSFER-STUCK") == 0 &&
          test_file_contains(log_path, "WD-STUCK") == 0,
        "e14-r2: 5500 sub-1s queued-round sweeps emit zero stderr log spam");

    test_capture_begin(log_path);
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    (void)SparkWeightdMeshDoorbellCell(band, local_rank, UINT64_C(10000) + SPARK_WEIGHTD_MESH_STUCK_LOG_NS + UINT64_C(1000));
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    test_capture_end();
    CHECK(test_file_contains(log_path, "WD-STUCK") != 0 &&
          test_file_contains(log_path, "WD-TRANSFER-STUCK") != 0,
        "e14-r2: genuine >1s stall still emits WD-STUCK and WD-TRANSFER-STUCK diagnostics");
    (void)unlink(log_path);

    test_complete_range(first, last);
    CHECK(SparkWeightdMeshHasPending() == 0u && weightd_mesh.send_pending_total == 0u,
        "e14-r2: send_pending_total returns to zero after both in-flight rounds drain");
    first = spark_stub_ibv_posted_count();
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    CHECK(SparkWeightdMeshDoorbellCell(band, local_rank, UINT64_C(20000) + SPARK_WEIGHTD_MESH_STUCK_LOG_NS) == 1u,
        "e14-r2: queued third round posts cleanly once transfer records drain");
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    test_complete_range(first, spark_stub_ibv_posted_count());
    CHECK(SparkWeightdMeshHasPending() == 0u && weightd_mesh.send_pending_total == 0u &&
          test_shipped(band, local_rank) == seq + 2u,
        "e14-r2: third round completes with zero leaked send_pending_total");
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
    uint32_t record;
    for (record=0u; record<SPARK_WEIGHTD_MESH_RECORDS; record++)
    {
        SparkWeightdMeshTransfer saved = weightd_mesh.transfers[record];
        uint32_t index = record / 2u;
        uint32_t band = index / SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        uint32_t rank = index % SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        uint64_t previous = weightd_mesh.doorbell_posted[index];
        uint64_t shipped = test_shipped(band,rank);
        uint32_t first = spark_stub_ibv_posted_count();
        uint32_t pending = weightd_mesh.send_pending[0];
        assert(saved.pending == 0u);
        memset(&weightd_mesh.transfers[record],0,sizeof(saved));
        weightd_mesh.transfers[record].generation = ++SparkWeightdMeshNextTransferGeneration;
        weightd_mesh.transfers[record].seq = previous + 1u;
        {
            SparkWeightdMeshRoute route = {0};
            uint8_t logical[SPARK_WEIGHTD_MESH_RANKS_PER_BAND] = {0};
            uint64_t tag = previous + 1u;
            SparkWeightdMeshPostPeer(record,band,0u,0u,0u,64u,route,logical,&tag);
            SparkWeightdMeshSqRing(1u);
        }
        CHECK(spark_stub_ibv_posted_count() - first == 2u,
            "every lane, band and rank posts its payload chained to an inline tail with a distinct completion identity");
        test_complete_range(first,spark_stub_ibv_posted_count());
        CHECK(weightd_mesh.transfers[record].pending == 0u &&
            weightd_mesh.doorbell_posted[index] == previous + 1u &&
            test_shipped(band,rank) == previous + 1u,
            "terminal completions acknowledge the exact lane, band, rank and record");
        CHECK(weightd_mesh.send_pending[0] == pending,
            "every completion identity returns exactly its own send credits");
        weightd_mesh.transfers[record] = saved;
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
    _Static_assert(SPARK_WEIGHTD_IPC_ABI_VERSION != 17u || (SPARK_WEIGHTD_MESH_REGION_BYTES == UINT64_C(537034752) &&
        SPARK_WEIGHTD_MESH_DOORBELL_OFFSET == UINT64_C(536903680) && SPARK_WEIGHTD_MESH_SHIPPED_OFFSET == UINT64_C(536908288) &&
        SPARK_WEIGHTD_MESH_WAIT_OFFSET == UINT64_C(536909312) && SPARK_WEIGHTD_MESH_SLOT_BYTES == 1048640u &&
        SPARK_WEIGHTD_MESH_SLOTS_PER_RANK == 4u &&
        SPARK_WEIGHTD_MESH_STAGING_BYTES == UINT64_C(134217728) && SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES == 128u &&
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES == 32u && sizeof(SparkWeightdMeshWaitRequest) == 128u),
        "the mesh layout changed under IPC ABI 17: bump SPARK_WEIGHTD_IPC_ABI_VERSION and pin the new layout here");
    CHECK(SPARK_WEIGHTD_IPC_ABI_VERSION == 17u &&
        SPARK_WEIGHTD_MESH_WAIT_OFFSET - SPARK_WEIGHTD_MESH_DOORBELL_OFFSET == 5632u &&
        (uint8_t *)test_wait_request(SPARK_WEIGHTD_MESH_BANDS - 1u,SPARK_WEIGHTD_MESH_RANKS_PER_BAND - 1u) + sizeof(*request) <=
            (uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_REGION_BYTES &&
        sizeof(*request) == 128u && offsetof(SparkWeightdMeshWaitRequest,ready) == 64u,
        "the ABI 8 gate geometry served under ABI 17 has separate producer and terminal cache lines within the registered region");
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
        CHECK(test_post_slot(3u,rank,tag,0x6u) == SPARK_STATUS_OK,"hardware gate transfer posts to two peers");
        last = spark_stub_ibv_posted_count();
        {
            SparkStubIbvPostedWork work;
            CHECK(spark_stub_ibv_posted(first + 1u,&work) == 0 && (work.wr_id & 3u) == 3u &&
                spark_stub_ibv_complete(work.wr_id,IBV_WC_RETRY_EXC_ERR) == 0,"hardware gate sees one tail fail on the wire");
            SparkWeightdMeshDrainCq();
            first += 2u;
        }
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
        started = SparkMonotonicNs();
        SparkWeightdMeshWaitForActivity();
        CHECK(SparkMonotonicNs() - started >= UINT64_C(150000),
            "an active lane without traffic polls at the dormant cadence instead of spinning");
        SparkWeightdMeshDoorbellPoll();
        CHECK(SparkWeightdMeshSetActivity(band / 2u,0u) == SPARK_STATUS_OK,"idle producer releases activity");
    }
    *cancel = 0u;
}

static void test_mesh_timing_steady(SparkWeightdMeshWaitRequest *request,uint32_t band,const char *log_path)
{
    const uint64_t tag = (UINT64_C(23) << 32u) | 2u;
    uint64_t *peer1 = test_peer_tail(band,1u,tag),*peer2 = test_peer_tail(band,2u,tag),*peer3 = test_peer_tail(band,3u,tag);
    *peer1 = *peer2 = 0u;
    *peer3 = tag;
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0xeu,0u);
    request->timeout_ns = UINT64_C(1000000000);
    SparkWeightdMeshWaitRequestsPoll(UINT64_C(1200000));
    *peer1 = *peer2 = tag;
    SparkWeightdMeshWaitRequestsPoll(UINT64_C(3700000));
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u && request->error == 0u &&
        SparkLatencyCount(&weightd_mesh.timing.start) == 0u && weightd_mesh.timing.last[2] == 1u &&
        weightd_mesh.timing.start_last[2] == 0u && weightd_mesh.timing.excess_ns[2] == 0u &&
        weightd_mesh.timing.gate.total_ns == UINT64_C(2500000),
        "a later round of the chain is not a start gate, and tails landing in one poll add no excess wait");
    test_capture_begin(log_path);
    SparkWeightdMeshTimingReport();
    test_capture_end();
    CHECK(test_file_contains(log_path,"gates=1 gate_us=4096/4096 gate_ms=2 self=0 starts=0 start_us=0/0 start_ms=0 start_self=0 worst_us=2500 worst_tag=23:2 worst_closer=2 peers=1:4096/4096/0/0/0,2:4096/4096/1/0/0,3:1/1/0/0/0") != 0,
        "the report totals gate wait in milliseconds and names the worst gate by epoch, round and closer");
    *peer1 = *peer2 = *peer3 = 0u;
}

static void test_mesh_timing(void)
{
    const uint32_t band = 2u,rank = 0u;
    const uint64_t tag = (UINT64_C(23) << 32u) | 1u;
    volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band,rank));
    SparkWeightdMeshWaitRequest *request = test_wait_request(band,rank);
    uint64_t *peer1 = test_peer_tail(band,1u,tag),*peer2 = test_peer_tail(band,2u,tag),*peer3 = test_peer_tail(band,3u,tag);
    char log_path[320];
    uint32_t first,last;
    memset(&weightd_mesh.timing,0,sizeof(weightd_mesh.timing));
    CHECK(SparkWeightdMeshSetActivity(band / 2u,1u) == SPARK_STATUS_OK,"timing lane is active");
    first = spark_stub_ibv_posted_count();
    entry[2] = (tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u);
    entry[1] = 64u;
    entry[3] = 0xeu;
    entry[0] = tag;
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    (void)SparkWeightdMeshDoorbellCell(band,rank,UINT64_C(1000000));
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == 6u && SparkLatencyCount(&weightd_mesh.timing.post) == 1u && weightd_mesh.timing.post.count[0] == 1u,
        "a round posted in the sweep that saw its doorbell records a sub-microsecond relay post");
    test_complete_range(first,last);
    CHECK(SparkLatencyCount(&weightd_mesh.timing.ship) == 1u,"only the last transfer completion records the ship time");
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,tag,0u,0u);
    request->timeout_ns = UINT64_C(1000000000);
    SparkWeightdMeshWaitRequestsPoll(UINT64_C(1010000));
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u && request->error == 0u &&
        SparkLatencyCount(&weightd_mesh.timing.credit) == 1u && weightd_mesh.timing.credit.count[0] == 1u,
        "a source credit that is already shipped records a zero-wait credit gate");
    *peer1 = tag;
    *peer2 = 0u;
    *peer3 = 0u;
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0xeu,0u);
    request->timeout_ns = UINT64_C(1000000000);
    SparkWeightdMeshWaitRequestsPoll(UINT64_C(1020000));
    *peer3 = tag;
    SparkWeightdMeshWaitRequestsPoll(UINT64_C(1023000));
    CHECK(request->ready == 0u,"the gate stays closed while a peer is missing");
    *peer2 = tag;
    SparkWeightdMeshWaitRequestsPoll(UINT64_C(1060000));
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u && request->error == 0u,"all peers release the timed gate");
    CHECK(weightd_mesh.timing.lag[1].count[5] == 1u && weightd_mesh.timing.lag[3].count[5] == 1u && weightd_mesh.timing.lag[2].count[6] == 1u &&
        weightd_mesh.timing.last[2] == 1u && weightd_mesh.timing.last[1] == 0u && weightd_mesh.timing.last[3] == 0u &&
        SparkLatencyCount(&weightd_mesh.timing.gate) == 1u && weightd_mesh.timing.gate.count[6] == 1u,
        "lags are measured from the local publish, not from the gate request, and only the latest peer counts as last");
    CHECK(weightd_mesh.timing.start_last[2] == 1u && weightd_mesh.timing.excess_ns[2] == UINT64_C(37000) &&
        SparkLatencyCount(&weightd_mesh.timing.start) == 1u,
        "the chain's first round is a start gate, and the last peer is charged the wait since the previous arrival");
    CHECK(SparkLatencyPercentileUs(&weightd_mesh.timing.lag[2],50u) == 64u &&
        SparkLatencyPercentileUs(&weightd_mesh.timing.lag[3],99u) == 32u,
        "percentiles report the bucket upper bound in microseconds");
    (void)snprintf(log_path,sizeof(log_path),"%s/timing.log",SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    SparkWeightdMeshTimingReport();
    test_capture_end();
    CHECK(test_file_contains(log_path,"WD-MESH-TIMING posts=1 post_us=1/1") != 0 &&
        test_file_contains(log_path,"credits=1 credit_us=1/1 gates=1") != 0 &&
        test_file_contains(log_path,"gates=1 gate_us=64/64 gate_ms=0 self=0 starts=1 start_us=64/64 start_ms=0 start_self=0 worst_us=40 worst_tag=23:1 worst_closer=2 peers=1:32/32/0/0/0,2:64/64/1/1/37,3:32/32/0/0/0") != 0,
        "the report names every peer's lag percentiles, last and start-last counts and excess wait");
    CHECK(SparkLatencyCount(&weightd_mesh.timing.gate) == 0u && weightd_mesh.timing.last[2] == 0u,
        "each report starts a new window");
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,0xeu,0u);
    request->timeout_ns = UINT64_C(1000000000);
    SparkWeightdMeshWaitRequestsPoll(UINT64_C(1100000));
    CHECK(__atomic_load_n(&request->ready,__ATOMIC_ACQUIRE) == 1u && request->error == 0u &&
        weightd_mesh.timing.self == 1u && weightd_mesh.timing.start_self == 1u && weightd_mesh.timing.last[1] == 0u &&
        weightd_mesh.timing.last[2] == 0u && weightd_mesh.timing.last[3] == 0u && weightd_mesh.timing.gate.count[0] == 1u,
        "a gate whose peers all arrived before weightd saw it is closed by the local rank, not by the last scanned peer");
    test_capture_begin(log_path);
    SparkWeightdMeshTimingReport();
    test_capture_end();
    CHECK(test_file_contains(log_path,"gates=1 gate_us=1/1 gate_ms=0 self=1 starts=1 start_us=1/1 start_ms=0 start_self=1 worst_us=0 worst_tag=0:0 worst_closer=0 peers=1:128/128/0/0/0,2:128/128/0/0/0,3:128/128/0/0/0") != 0,
        "the report counts gates the local rank closed");
    test_mesh_timing_steady(request,band,log_path);
    (void)unlink(log_path);
    *peer1 = *peer2 = *peer3 = 0u;
    CHECK(SparkWeightdMeshSetActivity(band / 2u,0u) == SPARK_STATUS_OK,"timing lane releases activity");
}

static void test_mesh_topology(void)
{
    SparkWeightdMeshTopology permuted = test_identity_topology(16u,4u);
    SparkWeightdMeshTopology subset = test_identity_topology(4u,3u),bad;
    uint32_t rank,first,last,i;
    uint64_t tag = (UINT64_C(91) << 32u) | 1u;
    const uint32_t band = 6u,local = 3u;
    SparkWeightdMeshWaitRequest *request = test_wait_request(band,local);
    uint64_t *entry = (uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band,local));
    for (rank=0u; rank<16u; rank++) permuted.physical_ranks[rank] = (rank + 3u) % 16u;
    for (rank=0u; rank<4u; rank++) subset.physical_ranks[rank] = rank + 4u;
    CHECK(test_post_slot(band,local,tag,1u) == SPARK_STATUS_INVALID_ARGUMENT &&
        SparkWeightdMeshSetActivity(3u,1u) == SPARK_STATUS_INVALID_ARGUMENT,
        "unconfigured lane cannot route or begin GPU work");
    bad = subset;bad.physical_ranks[2] = bad.physical_ranks[1];
    CHECK(SparkWeightdMeshLaneConfigure(3u,&bad) == SPARK_STATUS_INVALID_ARGUMENT,
        "duplicate physical ranks do not install a topology");
    bad = subset;bad.physical_ranks[0] = 16u;
    CHECK(SparkWeightdMeshLaneConfigure(3u,&bad) == SPARK_STATUS_INVALID_ARGUMENT,
        "out of range physical rank never aliases a valid rank");
    bad = subset;bad.local_rank = 0u;
    CHECK(SparkWeightdMeshLaneConfigure(3u,&bad) == SPARK_STATUS_INVALID_ARGUMENT,
        "logical local rank must map to this physical daemon");
    CHECK(SparkWeightdMeshLaneConfigure(2u,&permuted) == SPARK_STATUS_OK &&
        SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_OK,
        "permuted TP16 and physical ranks four through seven coexist on independent lanes");
    CHECK(test_post_slot(band,7u,tag,1u) == SPARK_STATUS_INVALID_ARGUMENT &&
        test_post_slot(band,local,tag,1u << 4u) == SPARK_STATUS_INVALID_ARGUMENT &&
        test_post_slot(band,local,tag,1u << local) == SPARK_STATUS_INVALID_ARGUMENT,
        "GPU source rank and mask remain logical and exclude self");
    CHECK(SparkWeightdMeshSetActivity(3u,1u) == SPARK_STATUS_OK &&
        SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_BUSY &&
        SparkWeightdMeshLaneConfigure(2u,&permuted) == SPARK_STATUS_OK,
        "live producer blocks its lane even between doorbells but not an independent lane");
    CHECK(SparkWeightdMeshSetActivity(3u,0u) == SPARK_STATUS_OK,
        "terminal producer releases only its lane activity");
    entry[1] = 64u;entry[2] = local * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
    entry[3] = 1u << 1u;entry[0] = tag;
    CHECK(SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_BUSY,
        "unposted GPU publication retains its lane after active interval");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(band,local,tag,1u << 1u) == SPARK_STATUS_OK,
        "TP4 logical peer one posts to physical five");
    last = spark_stub_ibv_posted_count();
    CHECK(last == first + 2u && SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_BUSY,
        "pending source reads retain lane configuration");
    for (i=first; i<last; i++)
    {
        SparkStubIbvPostedWork work;
        uint64_t offset = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND +
            local * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
        assert(spark_stub_ibv_posted(i,&work) == 0);
        if (i != first) offset += SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u;
        CHECK(work.qp_number == weightd_mesh.send_qps[5u]->qp_num &&
            (i != first ? work.inline_word == *(const uint64_t *)((const uint8_t *)weightd_mesh.recv_buffer + offset) :
                ((work.flags & IBV_SEND_INLINE) != 0u ?
                    work.inline_word == *(const uint64_t *)((const uint8_t *)weightd_mesh.recv_buffer + offset) :
                    work.source == (uint64_t)(uintptr_t)weightd_mesh.recv_buffer + offset)) &&
            work.remote == weightd_mesh.qp_info[5u].remote_addr + offset,
            "physical QP translation preserves the logical payload, tail tag and remote offsets");
    }
    test_complete_range(first,last - 1u);
    CHECK(SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_BUSY,
        "partial completion cannot release lane configuration");
    test_complete_range(last - 1u,last);
    CHECK(test_shipped(band,local) == tag,"the drained round shipped before re-configuration");
    CHECK(SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_OK && entry[0] == 0u &&
        test_shipped(band,local) == 0u && weightd_mesh.doorbell_posted[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local] == 0u,
        "re-acquired idle lane starts from zero cells (WD-LANE-RESET)");
    bad = subset;bad.physical_ranks[0] = 0u;
    CHECK(SparkWeightdMeshLaneConfigure(3u,&bad) == SPARK_STATUS_OK && entry[0] == 0u && test_shipped(band,local) == 0u,
        "a drained lane takes a changed topology and starts from zero cells");
    CHECK(SparkWeightdMeshSetActivity(3u,1u) == SPARK_STATUS_OK &&
        SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_UNSUPPORTED &&
        SparkWeightdMeshSetActivity(3u,0u) == SPARK_STATUS_OK,
        "a lane in use refuses a changed topology");
    CHECK(SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_OK,
        "the drained lane returns to its first topology");
    CHECK(SparkWeightdMeshSetActivity(3u,1u) == SPARK_STATUS_OK,"mapped wait producer begins");
    test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_PEERS,tag,1u << 1u,0u);
    CHECK(SparkWeightdMeshSetActivity(3u,0u) == SPARK_STATUS_OK,"wait outlives terminal producer");
    CHECK(SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_BUSY,
        "pending wait request retains lane ownership");
    CHECK(SparkWeightdMeshSetActivity(3u,1u) == SPARK_STATUS_OK,"explicit producer continues wait");
    SparkWeightdMeshWaitRequestsPoll(100u);
    CHECK(request->ready == 0u,"logical peer tail is required before mapped wait completes");
    *test_peer_tail(band,1u,tag) = tag;
    SparkWeightdMeshWaitRequestsPoll(101u);
    CHECK(request->ready == 1u && request->error == 0u,
        "mapped wait checks logical peer one rather than physical five");
    CHECK(SparkWeightdMeshSetActivity(3u,0u) == SPARK_STATUS_OK &&
        SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_OK,
        "wait terminal plus last producer end permits same profile reuse");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(4u,4u,tag,1u << 0u) == SPARK_STATUS_OK,
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
    CHECK(SparkWeightdMeshCellFailed(4u * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + 4u) != 0u &&
        test_post_slot(4u,4u,tag + 1u,1u << 0u) == SPARK_STATUS_IO_ERROR,
        "a failed cell fences the next publication while its lane stays configured");
    CHECK(SparkWeightdMeshLaneConfigure(2u,&permuted) == SPARK_STATUS_OK &&
        SparkWeightdMeshCellFailed(4u * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + 4u) == 0u &&
        SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_OK,
        "re-configuring the idle lane resets its failed cell without poisoning a different lane");
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshBroadcast(1u << 0u,0u,64u,0u,0u,0u) == 1u &&
        SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_BUSY,
        "unclassified CPU transfer conservatively retains every lane until terminal");
    test_complete_range(first,spark_stub_ibv_posted_count());
    CHECK(SparkWeightdMeshLaneConfigure(3u,&subset) == SPARK_STATUS_OK,
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
        CHECK(SparkWeightdClientLaneAcquire(owner,3u,&topology,&out,timeout) == SPARK_STATUS_OK,
            "owner reserves explicit topology before lending bands");
        {
            uint64_t band_bytes = (uint64_t)SPARK_WEIGHTD_MESH_SLOT_BYTES * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND;
            CHECK(SparkWeightdClientMeshBroadcast(owner,2u,0u,0u,64u,0u,0u,timeout) == SPARK_STATUS_INVALID_ARGUMENT &&
                SparkWeightdClientMeshWrite(owner,1u,band_bytes * 3u,band_bytes * 3u,64u,timeout) == SPARK_STATUS_INVALID_ARGUMENT,
                "a lane owner cannot write another lane's band");
            CHECK(SparkWeightdClientMeshBroadcast(owner,2u,SPARK_WEIGHTD_MESH_REGION_BYTES - 32u,SPARK_WEIGHTD_MESH_REGION_BYTES - 32u,64u,0u,0u,timeout) == SPARK_STATUS_INVALID_ARGUMENT &&
                SparkWeightdClientMeshBroadcast(owner,2u,band_bytes * 6u,band_bytes * 6u,0u,0u,0u,timeout) == SPARK_STATUS_INVALID_ARGUMENT &&
                SparkWeightdClientMeshBroadcast(owner,2u,band_bytes * 6u,band_bytes * 6u,64u,1u,SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(3u,0u),timeout) == SPARK_STATUS_INVALID_ARGUMENT,
                "a write past the registered region, an empty write and a sequence word outside the lane fail closed");
            CHECK(SparkWeightdClientMeshBroadcast(owner,0u,band_bytes * 6u,band_bytes * 7u + 64u,64u,1u,SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(7u,1u),timeout) == SPARK_STATUS_BUSY,
                "a write inside the owner's two bands passes the range check (an empty peer mask then posts nothing)");
        }
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
        CHECK(SparkWeightdClientLaneBind(owner,peer,0u,&topology,&out) == SPARK_STATUS_OK && out == 3u &&
            SparkWeightdClientLaneBind(owner,peer,1u,&topology,&out) == SPARK_STATUS_OK && out == 3u,
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
            "owner beyond SPARK_WEIGHTD_MESH_MAX_LANES cannot alias an occupied lane");
        SparkWeightdClientClose(contender);
        for (job=0u; job<SPARK_WEIGHTD_MESH_MAX_LANES; job++)
            SparkWeightdClientClose(clients[rank][job]);
        __atomic_store_n(&servers[rank].stop,1,__ATOMIC_SEQ_CST);
        assert(pthread_join(threads[rank],0) == 0);
        SparkWeightdServerDestroy(servers[rank].server);
    }
}

static void test_mesh_peer_reset(uint32_t local_rank)
{
    SparkWeightdMeshTopology topology = test_identity_topology(16u,local_rank);
    const uint32_t lane = 1u,band = 2u,peer = 7u;
    const uint32_t physical = test_peer_rank(peer,local_rank);
    const uint32_t index = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank;
    SparkWeightdMeshWaitRequest *request = test_wait_request(band,local_rank);
    uint64_t *base = (uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
        (uint64_t)(SPARK_WEIGHTD_MESH_DOORBELL_CELL_BASE + 2u * band) * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
    uint64_t *cancel = (uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
        (uint64_t)(SPARK_WEIGHTD_MESH_DOORBELL_CELL_CANCEL + 2u * band) * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
    uint64_t *tail = test_peer_tail(band,physical,9u);
    const uint64_t tag = 41u,base_value = UINT64_C(0x1234),cancel_value = UINT64_C(0x77),tail_value = UINT64_C(0x5151);
    uint32_t first,last,i;
    char log_path[256];
    SparkStatus status;
    test_complete_range(0u,spark_stub_ibv_posted_count());
    CHECK(SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_OK,"peer reset lane starts configured");
    *base = base_value;
    *cancel = cancel_value;
    *tail = tail_value;
    CHECK(SparkWeightdMeshSetActivity(lane,1u) == SPARK_STATUS_OK,"peer reset producer begins");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(band,local_rank,tag,1u << physical) == SPARK_STATUS_OK &&
        weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RECORD_OF(index,tag)].pending != 0u,"payload and tail to one peer are in flight");
    last = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshBroadcast(1u << physical,0u,64u,0u,0u,0u) == 1u &&
        weightd_mesh.rpc_pending[peer] == 1u && weightd_mesh.send_pending[peer] == 3u,
        "an rpc write to the same peer is in flight beside the round");
    CHECK(test_write_record(physical,31u) == 0,"the peer restarts and publishes a new record");
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-peer-reset.log",SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    SparkWeightdMeshTryWire();
    test_capture_end();
    CHECK(test_file_contains(log_path,"WD-PEER-RESET") && test_file_contains(log_path,"reason=record") &&
        test_file_contains(log_path,"cells=1 bits=1 send_pending=3 rpc_pending=1"),
        "the restarted peer's in-flight work is reset before re-wiring");
    CHECK(weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RECORD_OF(index,tag)].pending == 0u &&
        weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RECORD_OF(index,tag)].failed == SPARK_WEIGHTD_MESH_TRANSFER_PEER_RESET &&
        weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RECORD_OF(index,tag)].reset_peer == physical,"the lost round fails and names the peer");
    CHECK(weightd_mesh.send_pending[peer] == 0u && weightd_mesh.rpc_pending[peer] == 0u &&
        SparkWeightdMeshReady() == 1u,"the peer's counters restart from zero on a ready mesh");
    (void)test_wait_publish(request,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,tag,0u,cancel_value);
    SparkWeightdMeshWaitRequestsPoll(200u);
    CHECK(request->ready == 1u && request->error == (SPARK_WEIGHTD_MESH_WAIT_ERROR_PEER_RESET | physical),
        "the shipped wait of the lost round reports the reset peer");
    for (i=first; i<=last; i++)
    {
        SparkStubIbvPostedWork work;
        assert(spark_stub_ibv_posted(i,&work) == 0);
        assert(spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0);
    }
    SparkWeightdMeshDrainCq();
    CHECK(weightd_mesh.send_pending[peer] == 0u && weightd_mesh.rpc_pending[peer] == 0u &&
        test_shipped(band,local_rank) != tag,"late completions from before the reset change nothing");
    CHECK(SparkWeightdMeshSetActivity(lane,0u) == SPARK_STATUS_OK,"the producer of the lost round ends");
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-lane-reset.log",SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    status = SparkWeightdMeshLaneConfigure(lane,&topology);
    test_capture_end();
    CHECK(status == SPARK_STATUS_OK && test_file_contains(log_path,"WD-LANE-RESET") &&
        test_file_contains(log_path,"failed_cells=1"),"re-acquiring the idle lane resets its cells instead of refusing");
    CHECK(SparkWeightdMeshCellFailed(index) == 0u && weightd_mesh.doorbell_posted[index] == 0u && weightd_mesh.issued[index] == 0u &&
        test_shipped(band,local_rank) == 0u && request->request_id == 0u &&
        request->capabilities == SPARK_WEIGHTD_MESH_CAPABILITIES,"the lane's cells equal a fresh daemon's");
    CHECK(*base == base_value && *cancel == cancel_value && *tail == tail_value,
        "BASE, CANCEL and peer slot tails survive the lane reset");
    CHECK(test_post_slot(band,local_rank,tag + 1u,1u << physical) == SPARK_STATUS_OK,
        "the reset lane posts its next round to the restarted peer");
    test_complete_range(spark_stub_ibv_posted_count() - 2u,spark_stub_ibv_posted_count());
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
    CHECK(post_last - post_first == 6u && test_shipped(0u,0u) == 0u,
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
    CHECK(SparkWeightdClientMeshWrite(second,1u,4u * SPARK_WEIGHTD_MESH_SLOT_BYTES * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND,
        4u * SPARK_WEIGHTD_MESH_SLOT_BYTES * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND,64u,timeout) == SPARK_STATUS_OK,
        "explicit mesh RPC inside the client's lane wakes idle progress without a GPU interval");
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

#define TEST_PACE_CHUNK (2u * 1024u * 1024u)

static uint64_t test_pace_now_ns(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC,&now);
    return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static void test_pace_write_pack(const char *path,const char *manifest_path,uint32_t salt)
{
    FILE *pack = fopen(path,"wb"),*manifest = fopen(manifest_path,"wb");
    uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,2u,0u},i;
    uint8_t *data = malloc(TEST_PACE_CHUNK);
    assert(pack != 0 && manifest != 0 && data != 0);
    assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
    for (i=0u; i<2u; i++)
    {
        uint8_t record[48] = {0},digest[16];
        uint64_t offset = (uint64_t)i * TEST_PACE_CHUNK,bytes = TEST_PACE_CHUNK;
        SparkCk128Context ck;
        memset(data,(int)(0x40u + salt * 4u + i),TEST_PACE_CHUNK);
        SparkCk128Initialize(&ck);
        SparkCk128Update(&ck,data,TEST_PACE_CHUNK);
        SparkCk128Finalize(&ck,digest);
        memcpy(record + 4u,&i,4u);
        memcpy(record + 16u,&offset,8u);
        memcpy(record + 24u,&bytes,8u);
        memcpy(record + 32u,digest,16u);
        assert(fwrite(record,1u,sizeof(record),manifest) == sizeof(record));
        assert(fwrite(data,1u,TEST_PACE_CHUNK,pack) == TEST_PACE_CHUNK);
    }
    free(data);
    assert(fclose(pack) == 0 && fclose(manifest) == 0);
}

static SparkStatus test_pace_attach(SparkWeightdClient *client,const char *path,uint64_t *generation)
{
    SparkWeightdLazyAttachRequest request = {0};
    SparkWeightdLazyAttachResult result;
    SparkStatus status;
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = 2u * TEST_PACE_CHUNK;
    memcpy(request.identity.model,"pace-test",10u);
    assert(SparkSha256File(path,request.identity.pack_sha256) == SPARK_STATUS_OK);
    assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
    snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
    request.expert_pool_bytes = 2u * TEST_PACE_CHUNK;
    status = SparkWeightdClientAttachLazy(client,&request,&result,UINT64_C(10000000000));
    *generation = result.arena_generation;
    return(status != SPARK_STATUS_OK ? status : (SparkStatus)result.status);
}

static pid_t test_pace_serving_process(const char *path,const int commands[2],const int replies[2])
{
    pid_t pid = fork();
    int command_fd = commands[0],reply_fd = replies[1];
    if ( pid != 0 )
    {
        (void)close(commands[0]);
        (void)close(replies[1]);
        return(pid);
    }
    (void)close(commands[1]);
    (void)close(replies[0]);
    {
        SparkWeightdClient *serving = 0;
        SparkWeightdMeshTopology topology = test_identity_topology(4u,0u);
        const uint64_t timeout = UINT64_C(1000000000);
        uint32_t lane = 0u;
        uint8_t status,command;
        status = SparkWeightdClientConnect(path,&serving,0) == SPARK_STATUS_OK &&
            SparkWeightdClientLaneAcquire(serving,0u,&topology,&lane,timeout) == SPARK_STATUS_OK && lane == 0u &&
            SparkWeightdClientMeshActivity(serving,1u,1u,timeout) == SPARK_STATUS_OK ? 1u : 0u;
        if ( write(reply_fd,&status,1u) != 1 )
            _exit(2);
        while ( read(command_fd,&command,1u) == 1 && command == 'i' )
        {
            status = SparkWeightdClientMeshActivity(serving,1u,0u,timeout) == SPARK_STATUS_OK ? 1u : 0u;
            if ( write(reply_fd,&status,1u) != 1 )
                _exit(2);
        }
        SparkWeightdClientClose(serving);
        _exit(0);
    }
}

static uint8_t test_pace_command(int command_fd,int reply_fd,uint8_t command)
{
    uint8_t reply = 0u;
    if ( command != 0u && write(command_fd,&command,1u) != 1 )
        return(0u);
    return(read(reply_fd,&reply,1u) == 1 ? reply : 0u);
}

static void test_load_pace_while_serving(uint64_t pace)
{
    SparkWeightdServerConfig config;
    SparkWeightdClient *own = 0,*loading = 0;
    SparkWeightdDetachResult detached;
    TestMeshActivityThread server = {0};
    pthread_t server_thread;
    char path[128],pack[128],manifest[144],receipt[160],other[128],other_manifest[144],other_receipt[160];
    int commands[2],replies[2],wait_status = 0;
    pid_t child;
    uint32_t lane;
    uint64_t generation = 0u,other_generation = 0u;
    SparkWeightdMeshTopology topology = test_identity_topology(4u,0u);
    const uint64_t timeout = UINT64_C(1000000000);
    memset(&config,0,sizeof(config));
    (void)snprintf(path,sizeof(path),"/tmp/spark-mesh-pace-%ld.sock",(long)getpid());
    (void)snprintf(pack,sizeof(pack),"/tmp/spark-mesh-pace-%ld.pack",(long)getpid());
    (void)snprintf(manifest,sizeof(manifest),"%s.experts",pack);
    (void)snprintf(receipt,sizeof(receipt),"%s.verified",pack);
    (void)snprintf(other,sizeof(other),"/tmp/spark-mesh-pace-%ld.other",(long)getpid());
    (void)snprintf(other_manifest,sizeof(other_manifest),"%s.experts",other);
    (void)snprintf(other_receipt,sizeof(other_receipt),"%s.verified",other);
    test_pace_write_pack(pack,manifest,0u);
    test_pace_write_pack(other,other_manifest,1u);
    config.socket_path = path;
    config.device_bytes_max = UINT64_C(1073741824);
    config.load_pace_bytes_per_second = pace;
    assert(SparkWeightdServerCreate(&config,&server.server) == SPARK_STATUS_OK);
    assert(pthread_create(&server_thread,0,test_mesh_server_run,&server) == 0);
    assert(SparkWeightdClientConnect(path,&own,0) == SPARK_STATUS_OK);
    assert(SparkWeightdClientConnect(path,&loading,0) == SPARK_STATUS_OK);
    if ( pace == 0u )
    {
        CHECK(SparkWeightdClientLaneAcquire(own,1u,&topology,&lane,timeout) == SPARK_STATUS_OK && lane == 1u &&
            SparkWeightdClientMeshActivity(own,1u,1u,timeout) == SPARK_STATUS_OK,"pace: this process serves a lane on one connection");
        CHECK(test_pace_attach(loading,other,&other_generation) == SPARK_STATUS_OK,
            "pace: a process's own serving lane never paces its own loads on another connection");
        (void)SparkWeightdClientDetach(loading,other_generation,&detached,timeout);
        CHECK(SparkWeightdClientMeshActivity(own,1u,0u,timeout) == SPARK_STATUS_OK,"pace: this process's lane goes idle");
    }
    assert(pipe(commands) == 0 && pipe(replies) == 0);
    child = test_pace_serving_process(path,commands,replies);
    assert(child > 0);
    CHECK(test_pace_command(commands[1],replies[0],0u) == 1u,"pace: another process owns an active serving lane");
    if ( pace == 0u )
    {
        CHECK(test_pace_attach(loading,pack,&generation) == SPARK_STATUS_CAPACITY_EXCEEDED,
            "pace: a load while another process's lane serves is refused when weightd has no load cap");
        CHECK(test_pace_command(commands[1],replies[0],'i') == 1u,"pace: the serving process goes idle");
        test_sleep_ns(UINT64_C(1200000000));
        CHECK(test_pace_attach(loading,pack,&generation) == SPARK_STATUS_OK,
            "pace: once no lane has served for the linger, the same load runs unpaced");
    }
    else
    {
        uint64_t start = test_pace_now_ns();
        CHECK(test_pace_attach(loading,pack,&generation) == SPARK_STATUS_OK &&
            test_pace_attach(loading,other,&other_generation) == SPARK_STATUS_OK,"pace: capped loads run while another process's lane serves");
        CHECK((unsigned __int128)(test_pace_now_ns() - start) * pace >= (unsigned __int128)2u * TEST_PACE_CHUNK * UINT64_C(1000000000),
            "pace: the second load waits until the first one's bytes have drained at the cap");
        (void)SparkWeightdClientDetach(loading,other_generation,&detached,timeout);
        CHECK(test_pace_command(commands[1],replies[0],'i') == 1u,"pace: the serving process goes idle");
    }
    (void)close(commands[1]);
    CHECK(waitpid(child,&wait_status,0) == child && WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0,
        "pace: the serving process exits cleanly");
    (void)close(replies[0]);
    (void)SparkWeightdClientDetach(loading,generation,&detached,timeout);
    SparkWeightdClientClose(loading);
    SparkWeightdClientClose(own);
    __atomic_store_n(&server.stop,1,__ATOMIC_SEQ_CST);
    assert(pthread_join(server_thread,0) == 0);
    SparkWeightdServerDestroy(server.server);
    (void)unlink(receipt);
    (void)unlink(manifest);
    (void)unlink(pack);
    (void)unlink(other_receipt);
    (void)unlink(other_manifest);
    (void)unlink(other);
}

static void test_status(SparkWeightdIpcMeshStatusResult *status);
static int test_ready_text(char *text,size_t bytes);

static uint32_t test_address(uint32_t a,uint32_t b,uint32_t c,uint32_t d)
{
    return a << 24 | b << 16 | c << 8 | d;
}

static void test_mesh_second_rail(uint32_t local_rank)
{
    TestMeshRecord record;
    SparkStubIbvPostedWork work;
    SparkWeightdIpcMeshStatusResult status;
    char log_path[320],text[512],expected[160];
    uint32_t rank,peer,first,second,a,b,partner = local_rank ^ 1u,moved = UINT32_MAX,my_index;
    struct ibv_qp *pair_send;
    for (a = 0u; a < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; a++)
        for (b = 0u; b < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; b++)
            if (a != b)
                CHECK(((SparkWeightdMeshSecondRailPeers(a,0xffffu) >> (b < a ? b : b - 1u)) & 1u) ==
                    ((SparkWeightdMeshSecondRailPeers(b,0xffffu) >> (a < b ? a : a - 1u)) & 1u),
                    "both ends of a link pick the same rail");
    CHECK(__builtin_popcount(SparkWeightdMeshSecondRailPeers(local_rank,0xffffu)) == 8 &&
        SparkWeightdMeshSecondRailPeers(0u,0x000fu) == 0x5u,"eight of fifteen peers ride the second rail; rank 0 of four sends to ranks 1 and 3 on it");
    test_rank_mask = 0xffffu & ~(1u << partner);
    test_record_ports = 3u;
    test_record_switched = 1u;
    test_clean_dir();
    (void)mkdir(SPARK_WEIGHTD_MESH_DIR,0755);
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
        if (rank != local_rank && rank != partner)
            CHECK(test_write_record(rank,9u) == 0,"second rail case writes peer records");
    SparkWeightdMeshUseSecondRail();
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-two-ports.log",SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,
        test_rank_mask,"rocep1s0f0",3u,104u) == SPARK_STATUS_BUSY,"a second rail needs no partner rank in the mesh");
    SparkWeightdMeshPoll();
    test_capture_end();
    CHECK(SparkWeightdMeshReady() == 1u,"second rail mesh wires");
    CHECK(weightd_mesh.ports == 3u && test_file_contains(log_path,"WD-MESH-PORTS rank=7 ports=rocep1s0f1+rocep1s0f0: a peer with both ports up"),
        "two active ports are both used and the startup log names them");
    CHECK(weightd_mesh.pair_slot_peers == SparkWeightdMeshSecondRailPeers(local_rank,test_rank_mask) &&
        __builtin_popcount(weightd_mesh.pair_slot_peers) == 7,"without the partner seven of fourteen peers ride the second rail");
    (void)snprintf(expected,sizeof(expected),"WD-MESH-LAYOUT-READY rank=%u ports=rocep1s0f1+rocep1s0f0 two_ports=0x%04x one_port=0x0000 degraded=0x0000",
        local_rank,test_rank_mask & ~(1u << local_rank));
    CHECK(test_file_count(log_path,"layout=two-ports") == 14u && test_file_count(log_path,"layout=degraded") == 0u &&
        test_file_contains(log_path,expected),"every link reports the two-port layout and the ready summary has no degraded peer");
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        if (rank == local_rank || rank == partner)
            continue;
        peer = rank < local_rank ? rank : rank - 1u;
        my_index = test_my_index(rank,local_rank);
        second = (rank + local_rank) & 1u;
        CHECK(test_read_record(rank,&record) == 0 &&
            weightd_mesh.qp_info[peer].rkey == (second != 0u ? record.pair_rkey : record.rkey) &&
            spark_stub_ibv_qp_remote_qpn(weightd_mesh.send_qps[peer]->qp_num) == (int)(second != 0u ? record.pair_recv_qpn[my_index] : record.recv_qpn[my_index]) &&
            spark_stub_ibv_qp_remote_address(weightd_mesh.send_qps[peer]->qp_num) == (second != 0u ? test_address(10u,10u,200u,rank) : test_address(10u,10u,100u,10u + rank)),
            "a peer of the other parity is wired through the second port, one of the same parity through the first");
        first = spark_stub_ibv_posted_count();
        CHECK(SparkWeightdMeshPostWrite(rank,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer,weightd_mesh.recv_mr->lkey,1024u,0u) == SPARK_STATUS_OK &&
            spark_stub_ibv_posted(first,&work) == 0 &&
            work.lkey == (second != 0u ? weightd_mesh.pair_recv_mr->lkey : weightd_mesh.recv_mr->lkey),
            "a write posts with the registration of its peer's port");
        if (second != 0u && moved == UINT32_MAX)
            moved = rank;
    }
    test_status(&status);
    CHECK(status.pair_rank == UINT32_MAX,"a second rail offers no pair-first routes");
    CHECK(status.peers[moved].qp_flags == (SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS | SPARK_WEIGHTD_MESH_PEER_QP_PAIR_LINK),
        "status marks a peer on the second port without a degraded flag");
    peer = moved < local_rank ? moved : moved - 1u;
    my_index = test_my_index(moved,local_rank);
    pair_send = weightd_mesh.send_qps[peer];
    test_record_ports = 1u;
    CHECK(test_write_record(moved,10u) == 0,"a peer restarts with only its first port up");
    test_record_ports = 3u;
    test_capture_begin(log_path);
    SparkWeightdMeshTryWire();
    test_capture_end();
    test_fill_record(&record,moved,10u);
    CHECK(SparkWeightdMeshReady() == 1u && weightd_mesh.send_qps[peer] == weightd_mesh.slot_send_qps[0][peer] &&
        ((weightd_mesh.pair_slot_peers >> peer) & 1u) == 0u &&
        spark_stub_ibv_qp_state(pair_send->qp_num) == IBV_QPS_RESET &&
        spark_stub_ibv_qp_remote_qpn(weightd_mesh.send_qps[peer]->qp_num) == (int)record.recv_qpn[my_index] &&
        spark_stub_ibv_qp_remote_address(weightd_mesh.send_qps[peer]->qp_num) == test_address(10u,10u,100u,10u + moved),
        "the peer that lost its second port moves to the first port and its second-port queue pairs are reset");
    (void)snprintf(expected,sizeof(expected),"peer_rank=%u port=rocep1s0f1 layout=degraded ports=rocep1s0f1+rocep1s0f0/rocep1s0f1: rank %u has no rocep1s0f0 port up",moved,moved);
    CHECK(test_file_contains(log_path,expected),"the degraded link is logged with the port the peer lacks");
    test_status(&status);
    CHECK(status.peers[moved].qp_flags == (SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS | SPARK_WEIGHTD_MESH_PEER_QP_DEGRADED),
        "status flags the link as degraded on the first port");
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshPostWrite(moved,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer,weightd_mesh.recv_mr->lkey,1024u,0u) == SPARK_STATUS_OK &&
        spark_stub_ibv_posted(first,&work) == 0 && work.lkey == weightd_mesh.recv_mr->lkey &&
        work.qp_number == weightd_mesh.slot_send_qps[0][peer]->qp_num,
        "a write to the degraded peer posts on the first port's queue pair and registration");
    (void)snprintf(expected,sizeof(expected),"ports=rocep1s0f1+rocep1s0f0 degraded=0x%04x\n",1u << moved);
    CHECK(test_ready_text(text,sizeof(text)) == 0 && strstr(text,expected) != 0,"the ready marker names the degraded peer");
    weightd_mesh_second_rail = 0u;
    test_record_ports = 1u;
    test_record_switched = 0u;
    test_rank_mask = 0xffffu;
    test_clean_dir();
    (void)unlink(log_path);
    (void)rmdir(SPARK_WEIGHTD_MESH_DIR);
}

static void test_mesh_ports(uint32_t local_rank)
{
    TestMeshRecord record,own;
    SparkWeightdIpcMeshStatusResult status;
    char log_path[320],text[512],expected[200];
    uint32_t rank,peer,my_index;
    uint64_t started;
    SparkStatus result;
    test_rank_mask = 0xffffu;
    SparkWeightdMeshUseSecondRail();
    (void)mkdir(SPARK_WEIGHTD_MESH_DIR,0755);
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-ports.log",SPARK_WEIGHTD_MESH_DIR);
    weightd_mesh_port_grace_ns = UINT64_C(300000000);

    test_clean_dir();
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
        if (rank != local_rank)
            CHECK(test_write_record(rank,30u) == 0,"one switch: every peer has its first port only");
    spark_stub_ibv_set_port_state("rocep1s0f0",IBV_PORT_DOWN);
    started = SparkMonotonicNs();
    test_capture_begin(log_path);
    result = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,"rocep1s0f0",3u,104u);
    SparkWeightdMeshPoll();
    test_capture_end();
    CHECK(result == SPARK_STATUS_BUSY && SparkMonotonicNs() - started >= weightd_mesh_port_grace_ns &&
        test_file_contains(log_path,"WD-MESH-PORT interface=rocep1s0f0 sgid=3: port 1 not active") &&
        test_file_contains(log_path,"WD-MESH-PORTS rank=7 ports=rocep1s0f1: every peer uses rocep1s0f1; not used: rocep1s0f0"),
        "one switch: a second port that stays down past the grace is left out and the layout is logged");
    CHECK(weightd_mesh.ports == 1u && weightd_mesh.pair_protection_domain == 0 && weightd_mesh.slot_send_qps[1][0] == 0 &&
        test_read_record(local_rank,&own) == 0 && own.ports == 1u && own.switched_pair == 1u && own.pair_rkey == 0u && own.pair_send_qpn[0] == 0u,
        "one switch: no second-port resources are made and the record publishes the first port alone");
    CHECK(SparkWeightdMeshReady() == 1u && test_file_count(log_path,"layout=one-port") == 15u && test_file_count(log_path,"layout=degraded") == 0u,
        "one switch: every peer wires on the shared port and nothing is degraded");
    CHECK(test_ready_text(text,sizeof(text)) == 0 && strstr(text," ports=rocep1s0f1 degraded=0x0000\n") != 0,
        "one switch: the ready marker names one port and no degraded peer");
    rank = test_peer_rank(2u,local_rank);
    peer = 2u;
    test_record_ports = 3u;
    test_record_switched = 1u;
    CHECK(test_write_record(rank,31u) == 0,"one switch: a peer with a second port joins");
    test_capture_begin(log_path);
    SparkWeightdMeshTryWire();
    test_capture_end();
    (void)snprintf(expected,sizeof(expected),"peer_rank=%u port=rocep1s0f1 layout=degraded ports=rocep1s0f1/rocep1s0f1+rocep1s0f0: this node, rank %u has no rocep1s0f0 port up",
        rank,local_rank);
    test_status(&status);
    CHECK(SparkWeightdMeshReady() == 1u && test_file_contains(log_path,expected) &&
        (status.peers[rank].qp_flags & SPARK_WEIGHTD_MESH_PEER_QP_DEGRADED) != 0u &&
        weightd_mesh.send_qps[peer] == weightd_mesh.slot_send_qps[0][peer],
        "one switch: a peer that has the port this node lacks is wired on the shared port and flagged degraded");
    test_record_ports = 1u;
    test_record_switched = 0u;

    test_clean_dir();
    test_record_ports = 3u;
    test_record_switched = 1u;
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
        if (rank != local_rank && rank != test_peer_rank(4u,local_rank))
            CHECK(test_write_record(rank,32u) == 0,"first port down: the peers have both ports");
    test_record_ports = 1u;
    CHECK(test_write_record(test_peer_rank(4u,local_rank),32u) == 0,"first port down: one peer has the first port only");
    test_record_ports = 3u;
    spark_stub_ibv_set_port_state("rocep1s0f0",IBV_PORT_ACTIVE);
    spark_stub_ibv_set_port_state(TEST_MESH_INTERFACE,IBV_PORT_DOWN);
    test_capture_begin(log_path);
    result = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,"rocep1s0f0",3u,104u);
    SparkWeightdMeshPoll();
    SparkWeightdMeshPoll();
    test_capture_end();
    CHECK(result == SPARK_STATUS_BUSY && weightd_mesh.ports == 2u && weightd_mesh.home_port == 1u &&
        test_file_contains(log_path,"WD-MESH-PORTS rank=7 ports=rocep1s0f0: every peer uses rocep1s0f0; not used: rocep1s0f1") &&
        test_read_record(local_rank,&own) == 0 && own.ports == 2u && own.gid[15] == 2u && own.pair_rkey == 0u,
        "first port down: the second port carries the record's own address alone");
    rank = test_peer_rank(4u,local_rank);
    test_status(&status);
    CHECK(SparkWeightdMeshReady() == 0u && status.peers[rank].state == SPARK_WEIGHTD_MESH_PEER_RECORD_INVALID &&
        test_file_contains(log_path,"ports=rocep1s0f0/rocep1s0f1: no port in common"),
        "first port down: a peer with no port in common stays unwired and keeps the mesh unready");
    rank = test_peer_rank(5u,local_rank);
    peer = 5u;
    my_index = test_my_index(rank,local_rank);
    test_fill_record(&record,rank,32u);
    CHECK(weightd_mesh.peer_state[peer] == SPARK_WEIGHTD_MESH_PEER_WIRED && weightd_mesh.qp_info[peer].rkey == record.pair_rkey &&
        spark_stub_ibv_qp_remote_qpn(weightd_mesh.send_qps[peer]->qp_num) == (int)record.pair_recv_qpn[my_index] &&
        spark_stub_ibv_qp_remote_address(weightd_mesh.send_qps[peer]->qp_num) == test_address(10u,10u,200u,rank) &&
        (status.peers[rank].qp_flags & (SPARK_WEIGHTD_MESH_PEER_QP_PAIR_LINK | SPARK_WEIGHTD_MESH_PEER_QP_DEGRADED)) ==
            (SPARK_WEIGHTD_MESH_PEER_QP_PAIR_LINK | SPARK_WEIGHTD_MESH_PEER_QP_DEGRADED),
        "first port down: a peer with both ports is reached on its second port and flagged degraded");
    test_record_ports = 2u;
    CHECK(test_write_record(test_peer_rank(4u,local_rank),33u) == 0,"first port down: the stray peer comes back on its second port");
    test_record_ports = 3u;
    test_capture_begin(log_path);
    SparkWeightdMeshTryWire();
    test_capture_end();
    rank = test_peer_rank(4u,local_rank);
    test_status(&status);
    CHECK(SparkWeightdMeshReady() == 1u && (status.peers[rank].qp_flags & SPARK_WEIGHTD_MESH_PEER_QP_DEGRADED) == 0u &&
        test_file_contains(log_path,"layout=one-port ports=rocep1s0f0/rocep1s0f0"),
        "first port down: a peer with the same single port is a one-port link, not a degraded one");

    test_clean_dir();
    spark_stub_ibv_set_port_state("rocep1s0f0",IBV_PORT_DOWN);
    weightd_mesh_address_wait_ns = UINT64_C(600000000);
    test_capture_begin(log_path);
    result = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,"rocep1s0f0",3u,104u);
    test_capture_end();
    weightd_mesh_address_wait_ns = UINT64_C(120000000000);
    CHECK(result == SPARK_STATUS_DRIVER_LOAD_ERROR && test_file_contains(log_path,"WD-MESH-NO-PORT rank=7") &&
        test_file_contains(log_path,"WD-MESH-ADDRESS-DEADLINE interface=rocep1s0f0") &&
        test_file_contains(log_path,"WD-MESH-ADDRESS-DEADLINE interface=rocep1s0f1"),
        "no port: startup fails loudly after the address deadline and names both ports");
    spark_stub_ibv_set_port_state(TEST_MESH_INTERFACE,IBV_PORT_ACTIVE);
    spark_stub_ibv_set_port_state("rocep1s0f0",IBV_PORT_ACTIVE);

    test_clean_dir();
    started = SparkMonotonicNs();
    test_capture_begin(log_path);
    result = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,"rocep9s0f9",3u,104u);
    test_capture_end();
    CHECK(result == SPARK_STATUS_BUSY && weightd_mesh.ports == 1u && SparkMonotonicNs() - started < weightd_mesh_port_grace_ns &&
        test_file_contains(log_path,"WD-MESH-PORT interface=rocep9s0f9: no such RDMA device; the port is not used"),
        "a configured port with no device is left out at once");

    weightd_mesh_port_grace_ns = UINT64_C(10000000000);
    weightd_mesh_second_rail = 0u;
    test_record_ports = 1u;
    test_record_switched = 0u;
    test_clean_dir();
    (void)unlink(log_path);
    (void)rmdir(SPARK_WEIGHTD_MESH_DIR);
}

static void test_mesh_pair_link(uint32_t local_rank)
{
    TestMeshRecord own_record,partner_record,other_record;
    SparkStubIbvPostedWork work;
    uint32_t rank,partner = local_rank ^ 1u,other = local_rank == 0u ? 2u : 0u,first;
    uint32_t partner_peer = partner < local_rank ? partner : partner - 1u;
    uint32_t other_peer = other < local_rank ? other : other - 1u;
    SparkWeightdIpcMeshStatusResult status;
    test_rank_mask = 0xffffu;
    test_record_ports = 3u;
    test_clean_dir();
    (void)mkdir(SPARK_WEIGHTD_MESH_DIR,0755);
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
        if (rank != local_rank)
            CHECK(test_write_record(rank,9u) == 0,"pair case writes peer records");
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,
        test_rank_mask & ~(1u << partner),"rocep1s0f0",3u,104u) == SPARK_STATUS_INVALID_ARGUMENT,
        "a pair link needs its partner rank in the mesh");
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,
        test_rank_mask,"rocep1s0f0",3u,256u) == SPARK_STATUS_INVALID_ARGUMENT,
        "a traffic class above 255 rejects");
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,
        test_rank_mask,"rocep1s0f0",3u,104u) == SPARK_STATUS_BUSY,"pair init publishes");
    CHECK(weightd_mesh.pair_recv_mr != 0 && weightd_mesh.pair_recv_mr->lkey != weightd_mesh.recv_mr->lkey,
        "the pair link registers the mesh region on its own protection domain");
    CHECK(test_read_record(local_rank,&own_record) == 0 && own_record.pair_rkey == weightd_mesh.pair_recv_mr->rkey,
        "own record publishes the pair link's rkey");
    spark_stub_ibv_rtr_traffic_class_clear();
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"pair mesh wires");
    CHECK(spark_stub_ibv_rtr_traffic_class() == 104u,
        "every switch and pair QP moves to RTR with the configured traffic class");
    CHECK(test_read_record(partner,&partner_record) == 0 && test_read_record(other,&other_record) == 0 &&
        weightd_mesh.qp_info[partner_peer].rkey == partner_record.pair_rkey &&
        weightd_mesh.qp_info[other_peer].rkey == other_record.rkey,
        "the partner is wired through the pair link, every other peer through the switch");
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshPostWrite(partner,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer,weightd_mesh.recv_mr->lkey,1024u,0u) == SPARK_STATUS_OK &&
        spark_stub_ibv_posted(first,&work) == 0 && work.lkey == weightd_mesh.pair_recv_mr->lkey &&
        work.qp_number == weightd_mesh.send_qps[partner_peer]->qp_num,
        "a write to the partner posts with the pair link's registration");
    CHECK(SparkWeightdMeshPostWrite(other,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer,weightd_mesh.recv_mr->lkey,1024u,0u) == SPARK_STATUS_OK &&
        spark_stub_ibv_posted(first + 1u,&work) == 0 && work.lkey == weightd_mesh.recv_mr->lkey,
        "a write to any other peer posts with the switch registration");
    test_status(&status);
    CHECK(status.pair_rank == partner &&
        status.peers[partner].qp_flags == (SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS | SPARK_WEIGHTD_MESH_PEER_QP_PAIR_LINK) &&
        status.peers[other].qp_flags == (SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS) &&
        weightd_mesh.slot_send_qps[1][other_peer] == 0 && weightd_mesh.slot_send_qps[1][partner_peer] != 0,
        "a pair link carries the partner alone and no link is degraded");
    test_record_ports = 1u;
    test_clean_dir();
    (void)rmdir(SPARK_WEIGHTD_MESH_DIR);
}

static void test_status(SparkWeightdIpcMeshStatusResult *status)
{
    memset(status,0,sizeof(*status));
    CHECK(SparkWeightdMeshStatusFill(status) == SPARK_STATUS_OK,"mesh status fill answers");
}

static int test_ready_text(char *text,size_t bytes)
{
    char path[512];
    ssize_t got;
    int fd;
    test_ready_path(path,sizeof(path));
    fd = open(path,O_RDONLY);
    if ( fd < 0 )
        return -1;
    got = read(fd,text,bytes - 1u);
    (void)close(fd);
    if ( got < 0 )
        return -1;
    text[got] = '\0';
    return 0;
}

static void test_mesh_status_surface(uint32_t local_rank)
{
    SparkWeightdIpcMeshStatusResult status;
    SparkWeightdMeshTopology topology = test_identity_topology(16u,local_rank);
    TestMeshRecord record,own;
    struct stat st;
    char path[512],text[256],expected[256];
    const uint32_t gap = test_peer_rank(2u,local_rank),noisy_peer = 4u,noisy = test_peer_rank(noisy_peer,local_rank);
    const uint32_t lane = 1u,cell = 2u * lane * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank;
    uint64_t generation,ready_since;
    uint32_t rank,first;
    SparkStubIbvPostedWork work;
    test_clean_dir();
    for (rank=0u; rank<SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
        if ( rank != local_rank && rank != gap )
            CHECK(test_write_record(rank,40u) == 0,"status: peer records published except one");
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u) ==
        SPARK_STATUS_BUSY,"status: init publishes");
    CHECK(test_read_record(local_rank,&own) == 0,"status: own record readable");
    test_status(&status);
    CHECK(status.mesh_state == SPARK_WEIGHTD_MESH_STATE_WIRING && status.mesh_generation == 1u &&
        status.boot_ns == own.boot_ns && status.local_rank == local_rank && status.rank_mask == test_rank_mask &&
        status.pair_rank == UINT32_MAX && status.wired_mask == (1u << local_rank),"status: a fresh mesh reports wiring at generation 1");
    CHECK(status.peers[local_rank].state == SPARK_WEIGHTD_MESH_PEER_SELF &&
        status.peers[gap].state == SPARK_WEIGHTD_MESH_PEER_NO_RECORD && status.peers[gap].record_status == SPARK_STATUS_BUSY,
        "status: peers start without a record");
    SparkWeightdMeshTryWire();
    test_status(&status);
    test_ready_path(path,sizeof(path));
    CHECK(status.mesh_state == SPARK_WEIGHTD_MESH_STATE_WIRING && status.wired_mask == (0xffffu & ~(1u << gap)) &&
        status.peers[gap].state == SPARK_WEIGHTD_MESH_PEER_NO_RECORD && stat(path,&st) != 0,
        "status: a missing record keeps the mesh wiring, names the peer, and leaves no marker");
    CHECK(status.peers[noisy].state == SPARK_WEIGHTD_MESH_PEER_WIRED &&
        status.peers[noisy].qp_flags == (SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS) &&
        status.peers[noisy].wired_boot_ns == status.peers[noisy].record_boot_ns && status.peers[noisy].since_mono_ns != 0u,
        "status: a wired peer reports both queue pairs in RTS and its record identity");
    generation = status.mesh_generation;
    CHECK(generation > 1u,"status: wiring peers advances the generation");
    CHECK(test_write_record(gap,40u) == 0,"status: the missing record appears");
    SparkWeightdMeshTryWire();
    test_status(&status);
    CHECK(status.mesh_state == SPARK_WEIGHTD_MESH_STATE_READY && status.wired_mask == 0xffffu &&
        status.ready_since_mono_ns != 0u && status.mesh_generation > generation,"status: the last record makes the mesh ready");
    (void)snprintf(expected,sizeof(expected),"weightd-ready v1 pid=%ld boot_ns=%llu rank=%u rank_mask=0x%04x ports=rocep1s0f1 degraded=0x0000\n",
        (long)getpid(),(unsigned long long)own.boot_ns,local_rank,test_rank_mask);
    CHECK(test_ready_text(text,sizeof(text)) == 0 && strcmp(text,expected) == 0,"status: .ready holds the v1 line with pid and boot_ns");
    (void)snprintf(path,sizeof(path),"%s/.ready.tmp",SPARK_WEIGHTD_MESH_DIR);
    CHECK(stat(path,&st) != 0,"status: no temporary marker remains");
    generation = status.mesh_generation;
    ready_since = status.ready_since_mono_ns;
    SparkWeightdMeshTryWire();
    test_status(&status);
    CHECK(status.mesh_generation == generation && status.ready_since_mono_ns == ready_since,"status: a pass with no change keeps the generation");
    CHECK(status.counters[0] == weightd_mesh.stat_trywire && status.counters[1] == weightd_mesh.stat_record_failures &&
        status.counters[2] == weightd_mesh.stat_record_invalid && status.counters[3] == weightd_mesh.stat_wire_failures &&
        status.counters[4] == weightd_mesh.stat_rewires && status.counters[5] == weightd_mesh.stat_unready &&
        status.counters[6] == weightd_mesh.stat_repairs && status.counters[7] == weightd_mesh.send_ok &&
        status.counters[8] == weightd_mesh.send_err && status.counters[9] == weightd_mesh.stat_peer_resets &&
        status.counters[10] == weightd_mesh.stat_peer_reset_bits && status.counters[11] == weightd_mesh.stat_lane_resets,
        "status: counters are the daemon's own statistics");
    test_fill_record(&record,gap,41u);
    record.magic = UINT64_C(0x4d45534830303031);
    CHECK(test_write_record_raw(&record) == 0,"status: an old-format record is published");
    SparkWeightdMeshTryWire();
    test_status(&status);
    test_ready_path(path,sizeof(path));
    CHECK(status.peers[gap].state == SPARK_WEIGHTD_MESH_PEER_RECORD_REJECTED && status.peers[gap].record_status == SPARK_STATUS_ABI_MISMATCH &&
        status.mesh_state == SPARK_WEIGHTD_MESH_STATE_WIRING && status.ready_since_mono_ns == 0u && stat(path,&st) != 0 &&
        status.mesh_generation > generation,"status: an incompatible record is rejected by name and drops readiness and the marker");
    test_fill_record(&record,gap,41u);
    record.rank_mask = 0xffu;
    CHECK(test_write_record_raw(&record) == 0,"status: a record from another group is published");
    SparkWeightdMeshTryWire();
    test_status(&status);
    CHECK(status.peers[gap].state == SPARK_WEIGHTD_MESH_PEER_RECORD_REJECTED && status.peers[gap].record_status == SPARK_STATUS_INVALID_ARGUMENT,
        "status: a group mismatch is rejected with its own status");
    test_fill_record(&record,gap,41u);
    memset(record.gid,0,sizeof(record.gid));
    CHECK(test_write_record_raw(&record) == 0,"status: a record without an IPv4 address is published");
    SparkWeightdMeshTryWire();
    test_status(&status);
    CHECK(status.peers[gap].state == SPARK_WEIGHTD_MESH_PEER_RECORD_INVALID && status.peers[gap].record_status == SPARK_STATUS_OK &&
        status.peers[gap].record_boot_ns == record.boot_ns,"status: an unusable record is reported as invalid with its boot");
    test_fill_record(&record,gap,42u);
    spark_stub_ibv_fail_modify_qp_for_qpn(record.recv_qpn[test_my_index(gap,local_rank)]);
    CHECK(test_write_record(gap,42u) == 0,"status: a record with a dead queue pair is published");
    SparkWeightdMeshTryWire();
    spark_stub_ibv_fail_modify_qp_for_qpn(0u);
    test_status(&status);
    CHECK(status.peers[gap].state == SPARK_WEIGHTD_MESH_PEER_WIRE_FAILED && status.peers[gap].record_status == SPARK_STATUS_DRIVER_LOAD_ERROR &&
        status.peers[gap].record_boot_ns == record.boot_ns && (status.peers[gap].qp_flags & 3u) == 0u,
        "status: a failed wire names the record it failed on");
    SparkWeightdMeshTryWire();
    test_status(&status);
    CHECK(status.peers[gap].state == SPARK_WEIGHTD_MESH_PEER_WIRED && status.peers[gap].wired_boot_ns == record.boot_ns &&
        status.mesh_state == SPARK_WEIGHTD_MESH_STATE_READY,"status: the retry wires the peer and the mesh is ready again");
    test_complete_range(0u,spark_stub_ibv_posted_count());
    generation = status.mesh_generation;
    CHECK(SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_OK,"status: lane configured");
    test_status(&status);
    CHECK((status.lanes[lane].flags & SPARK_WEIGHTD_MESH_LANE_CONFIGURED) != 0u && status.lanes[lane].rank_count == 16u &&
        status.lanes[lane].local_rank == local_rank && status.lanes[lane].physical_mask == 0xffffu &&
        status.lanes[lane].packed_ranks == UINT64_C(0xfedcba9876543210) && status.lanes[lane].configure_status == SPARK_STATUS_OK &&
        status.mesh_generation > generation,"status: a configured lane reports its topology and an open gate");
    generation = status.mesh_generation;
    CHECK(SparkWeightdMeshSetActivity(lane,1u) == SPARK_STATUS_OK,"status: lane becomes active");
    test_status(&status);
    CHECK(status.lanes[lane].activity == 1u && status.lanes[lane].configure_status == SPARK_STATUS_BUSY &&
        status.lanes[lane].busy_reason == SPARK_WEIGHTD_MESH_LANE_BUSY_ACTIVITY &&
        SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_BUSY,"status: an active lane reports the same refusal configure gives");
    CHECK(SparkWeightdMeshSetActivity(lane,0u) == SPARK_STATUS_OK,"status: lane goes idle");
    test_status(&status);
    CHECK(status.mesh_generation == generation && status.lanes[lane].activity == 0u,"status: activity toggles do not move the generation");
    weightd_mesh.transfers[2u * cell + 1u].pending = 1u;
    test_status(&status);
    CHECK(status.lanes[lane].pending_cells == 1u && status.lanes[lane].busy_reason == SPARK_WEIGHTD_MESH_LANE_BUSY_PENDING &&
        status.lanes[lane].busy_index == cell && SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_BUSY,
        "status: a pending cell is reported where configure refuses");
    weightd_mesh.transfers[2u * cell + 1u].pending = 0u;
    weightd_mesh.transfers[2u * cell].failed = SPARK_WEIGHTD_MESH_TRANSFER_FAILED;
    test_status(&status);
    CHECK(status.lanes[lane].failed_cells == 1u && status.lanes[lane].configure_status == SPARK_STATUS_OK,
        "status: a failed cell is visible and does not block a reconfigure");
    CHECK(SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_OK,"status: reconfigure resets the failed cell");
    test_status(&status);
    CHECK(status.lanes[lane].failed_cells == 0u,"status: the reset cell is no longer failed");
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshBroadcast(1u << noisy,0u,64u,0u,0u,0u) == 1u &&
        spark_stub_ibv_posted(first,&work) == 0,"status: one write to a peer is posted");
    CHECK(spark_stub_ibv_complete(work.wr_id,IBV_WC_RETRY_EXC_ERR) == 0,"status: the write fails on the wire");
    SparkWeightdMeshDrainCq();
    test_status(&status);
    CHECK(status.peers[noisy].cq_err_since_ok == 1u && status.peers[noisy].last_err_mono_ns != 0u,
        "status: a completion error is charged to its peer");
    {
        int32_t reset_peer = SparkWeightdMeshPeerIndexFromRank(noisy);
        uint64_t saved_boot = weightd_mesh.wired_boot_ns[reset_peer];
        uint64_t calls_before = spark_stub_ibv_post_send_calls();
        weightd_mesh.wired_boot_ns[reset_peer] = 0u;
        CHECK(SparkWeightdMeshBroadcast(1u << noisy,0u,64u,0u,0u,0u) == 0u && spark_stub_ibv_post_send_calls() == calls_before,
            "status: a broadcast to a peer whose wiring was reset posts nothing and reports busy, as a single write does");
        weightd_mesh.wired_boot_ns[reset_peer] = saved_boot;
    }
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshBroadcast(1u << noisy,0u,64u,0u,0u,0u) == 1u &&
        spark_stub_ibv_posted(first,&work) == 0 && spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0,
        "status: a second write completes");
    SparkWeightdMeshDrainCq();
    test_status(&status);
    CHECK(status.peers[noisy].cq_err_since_ok == 0u && status.peers[noisy].last_ok_mono_ns >= status.peers[noisy].last_err_mono_ns,
        "status: a success clears the peer's error run");
    (void)noisy_peer;
    test_ready_path(path,sizeof(path));
    SparkWeightdMeshTryWire();
    CHECK(stat(path,&st) == 0,"status: the marker exists before stop");
    SparkWeightdMeshStop();
    CHECK(stat(path,&st) != 0 && SparkWeightdMeshReady() == 1u,"status: stop removes the daemon's own marker and leaves readiness alone");
    {
        int fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0644);
        const char *foreign = "weightd-ready v1 pid=1 boot_ns=1 rank=0 rank_mask=0xffff\n";
        CHECK(fd >= 0 && test_write_fully(fd,foreign,strlen(foreign)) == 0,"status: a foreign marker is planted");
        if ( fd >= 0 )
            (void)close(fd);
    }
    SparkWeightdMeshStop();
    CHECK(stat(path,&st) == 0,"status: stop leaves a marker it did not write");
    weightd_mesh.artifact_check_ns = 0u;
    SparkWeightdMeshPoll();
    CHECK(test_ready_text(text,sizeof(text)) == 0 && strcmp(text,expected) == 0,"status: the self-heal replaces a foreign marker with its own");
}

static void test_mesh_record_addresses(uint32_t local_rank)
{
    TestMeshRecord record,own;
    char log_path[320];
    uint64_t modify_before,started;
    uint32_t rank = test_peer_rank(0u,local_rank),my_index = test_my_index(rank,local_rank);
    SparkStatus status;

    uint32_t peer_rank;
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u) ==
        SPARK_STATUS_BUSY,"address fixtures start a fresh mesh");
    for (peer_rank = 0u; peer_rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer_rank++)
        if (peer_rank != local_rank)
            CHECK(test_write_record(peer_rank,19u) == 0,"address fixtures publish every peer");
    SparkWeightdMeshTryWire();
    CHECK(SparkWeightdMeshReady() == 1u,"address fixtures start ready");
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-record-invalid.log",SPARK_WEIGHTD_MESH_DIR);
    test_fill_record(&record,rank,20u);
    memset(record.gid,0,sizeof(record.gid));
    CHECK(test_write_record_raw(&record) == 0,"a peer publishes an empty switch gid");
    modify_before = spark_stub_ibv_modify_qp_calls();
    test_capture_begin(log_path);
    SparkWeightdMeshTryWire();
    SparkWeightdMeshTryWire();
    test_capture_end();
    CHECK(SparkWeightdMeshReady() == 0u,"an empty peer gid keeps the mesh unready");
    CHECK(spark_stub_ibv_modify_qp_calls() == modify_before,"an empty peer gid is never dialled");
    CHECK(test_file_count(log_path,"WD-MESH-RECORD-INVALID") == 1u,"an invalid record is reported once per generation");
    CHECK(test_file_contains(log_path,"link=switch gid=00000000000000000000000000000000"),"the report names the published gid");

    test_fill_record(&record,rank,21u);
    record.gid[10] = 0u;
    CHECK(test_write_record_raw(&record) == 0,"a peer publishes a non IPv4 gid");
    test_capture_begin(log_path);
    SparkWeightdMeshTryWire();
    test_capture_end();
    CHECK(SparkWeightdMeshReady() == 0u && test_file_count(log_path,"WD-MESH-RECORD-INVALID") == 1u,
        "a new generation with another bad gid is reported again");

    CHECK(test_write_record(rank,22u) == 0,"the peer republishes a valid record");
    SparkWeightdMeshTryWire();
    CHECK(SparkWeightdMeshReady() == 1u,"a valid record restores readiness without a local restart");
    test_fill_record(&record,rank,22u);
    CHECK(spark_stub_ibv_qp_remote_qpn(weightd_mesh.send_qps[0]->qp_num) == (int)record.recv_qpn[my_index],
        "the peer is wired to its new generation");

    test_fill_record(&record,rank,23u);
    spark_stub_ibv_fail_modify_qp_for_qpn(record.recv_qpn[my_index]);
    CHECK(test_write_record(rank,23u) == 0,"the peer republishes a record the driver rejects");
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-wire-repeat.log",SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    SparkWeightdMeshTryWire();
    SparkWeightdMeshTryWire();
    SparkWeightdMeshTryWire();
    test_capture_end();
    CHECK(SparkWeightdMeshReady() == 0u,"a rejected transition keeps the mesh unready");
    CHECK(test_file_count(log_path,"WD-WIRE-FAIL") == 1u,"repeated wire failures on one generation log once");
    CHECK(test_file_contains(log_path,"qp=send transition=RTR errno=22"),"the wire failure names the qp, transition and errno");
    CHECK(test_file_contains(log_path,"gid=10.10.100."),"the wire failure names the dialled address");
    spark_stub_ibv_fail_modify_qp_for_qpn(0u);
    SparkWeightdMeshTryWire();
    CHECK(SparkWeightdMeshReady() == 1u,"the same generation wires once the driver accepts it");
    {
        TestMeshRecord own,peer_record;
        uint64_t generation;
        CHECK(test_read_record(local_rank,&own) == 0,"the local record is readable before a repair");
        generation = own.wire_generation;
        spark_stub_ibv_set_qp_state(weightd_mesh.send_qps[0]->qp_num,IBV_QPS_ERR);
        (void)snprintf(log_path,sizeof(log_path),"%s/capture-resync.log",SPARK_WEIGHTD_MESH_DIR);
        test_capture_begin(log_path);
        SparkWeightdMeshTryWire();
        SparkWeightdMeshTryWire();
        test_capture_end();
        CHECK(spark_stub_ibv_qp_state(weightd_mesh.send_qps[0]->qp_num) == IBV_QPS_RTS,"the repaired queue pair is back in RTS");
        CHECK(test_read_record(local_rank,&own) == 0 && own.wire_generation == generation + 1u,
            "the repair bumps the wire generation and publishes it");
        CHECK(SparkWeightdMeshReady() == 0u,"the mesh stays unready until the peer re-transitions to PSN 0 too");
        CHECK(test_file_count(log_path,"WD-QP-REPAIR") == 1u && test_file_count(log_path,"WD-QP-RESYNC-DONE") == 0u,
            "the repair is reported once and the resync is still open");
        test_fill_record(&peer_record,rank,23u);
        peer_record.seen_generation[my_index] = generation + 1u;
        CHECK(test_write_record_raw(&peer_record) == 0,"the peer acknowledges the generation in its record");
        test_capture_begin(log_path);
        SparkWeightdMeshTryWire();
        test_capture_end();
        CHECK(SparkWeightdMeshReady() == 1u && test_file_count(log_path,"WD-QP-RESYNC-DONE") == 1u,
            "the peer's acknowledgement completes the resync and the mesh is ready again");
        peer_record.wire_generation = 5u;
        CHECK(test_write_record_raw(&peer_record) == 0,"the peer repairs its own queue pairs");
        modify_before = spark_stub_ibv_modify_qp_calls();
        test_capture_begin(log_path);
        SparkWeightdMeshTryWire();
        test_capture_end();
        CHECK(spark_stub_ibv_modify_qp_calls() > modify_before && SparkWeightdMeshReady() == 1u &&
            test_file_count(log_path,"WD-QP-RESYNC rank=") == 1u,"a peer's repair re-transitions this side without a local repair");
        CHECK(test_read_record(local_rank,&own) == 0 && own.seen_generation[0] == 5u && own.wire_generation == generation + 1u,
            "this side echoes the peer's generation and keeps its own");
    }

    (void)snprintf(log_path,sizeof(log_path),"%s/capture-address-wait.log",SPARK_WEIGHTD_MESH_DIR);
    spark_stub_ibv_empty_gids(3u);
    started = SparkMonotonicNs();
    test_capture_begin(log_path);
    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u);
    test_capture_end();
    CHECK(status == SPARK_STATUS_BUSY,"init waits for its own address and then publishes");
    CHECK(SparkMonotonicNs() - started >= 3ull * (uint64_t)SPARK_WEIGHTD_MESH_ADDRESS_POLL_NS,
        "init polls the gid table instead of publishing an empty gid");
    CHECK(test_file_count(log_path,"WD-MESH-WAIT-ADDRESS") == 1u,"the address wait is reported once");
    CHECK(test_file_contains(log_path,"WD-MESH-ADDRESS interface=rocep1s0f1 sgid=3 address=10.103.0.1"),
        "the published address is reported");
    CHECK(test_read_record(local_rank,&own) == 0 && own.gid[10] == 0xffu && own.gid[13] == 103u,
        "the published record carries the IPv4 RoCE gid");

    (void)snprintf(log_path,sizeof(log_path),"%s/capture-address-deadline.log",SPARK_WEIGHTD_MESH_DIR);
    spark_stub_ibv_empty_gids(1000u);
    weightd_mesh_address_wait_ns = UINT64_C(1000000000);
    test_capture_begin(log_path);
    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u);
    test_capture_end();
    spark_stub_ibv_empty_gids(0u);
    weightd_mesh_address_wait_ns = UINT64_C(120000000000);
    CHECK(status == SPARK_STATUS_DRIVER_LOAD_ERROR,"init fails loudly when the address never appears");
    CHECK(test_file_contains(log_path,"WD-MESH-ADDRESS-DEADLINE interface=rocep1s0f1 sgid=3"),"the deadline names the interface");
}

static void test_sq_occupancy_guard(uint32_t local_rank)
{
    const uint32_t peer = 0u;
    const uint32_t peer_rank = test_peer_rank(peer,local_rank);
    SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[peer];
    const uint64_t saved_wait_ns = weightd_mesh_sq_wait_ns;
    const uint64_t saved_stat_logged_ns = weightd_mesh.stat_logged_ns;
    uint32_t i,depth,produced_before,retired_before,slot0,slot2;
    uint64_t waits_before,tag = UINT64_C(0x123456789abcdef0),saved_wr0,saved_wr2;
    uint8_t saved_wqe0[64],saved_wqe2[64];
    char log_path[256];
    test_complete_range(0u,spark_stub_ibv_posted_count());
    depth = queue->depth;
    CHECK(depth == 1024u && depth <= SPARK_WEIGHTD_MESH_SQ_SLOTS,"SQ guard fixture uses full 1024-slot mlx5 SQ");
    CHECK(queue->produced == queue->retired,"SQ starts drained before occupancy guard test");
    produced_before = queue->produced;
    retired_before = queue->retired;
    waits_before = weightd_mesh.stat_sq_full_waits;
    weightd_mesh_sq_wait_ns = UINT64_C(100000);
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    for ( i = 0u; i < depth; i++ )
    {
        uint64_t wr_id = ((uint64_t)(weightd_mesh.wire_epoch[peer] + 1000u + i) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)peer;
        uint32_t signaled = (i & 1u) != 0u ? 1u : 0u;
        CHECK(SparkWeightdMeshSqWrite(peer,wr_id,signaled,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer,
            weightd_mesh.recv_mr->lkey,signaled != 0u ? 0u : 1024u,(uint64_t)i * 64u,signaled != 0u ? &tag : 0) == SPARK_STATUS_OK,
            "filling SQ up to depth succeeds without waiting");
        if ( signaled != 0u )
            SparkWeightdMeshSqRing(1u << peer);
    }
    CHECK(queue->produced - queue->retired == depth,"SQ is completely full at depth 512");
    CHECK(weightd_mesh.stat_sq_full_waits == waits_before,"filling up to depth does not trigger SQ full waits");
    slot0 = produced_before & queue->mask;
    slot2 = (produced_before + 2u) & queue->mask;
    memcpy(saved_wqe0,queue->buffer + (uint64_t)slot0 * 64u,sizeof(saved_wqe0));
    memcpy(saved_wqe2,queue->buffer + (uint64_t)slot2 * 64u,sizeof(saved_wqe2));
    saved_wr0 = queue->wr_id[slot0];
    saved_wr2 = queue->wr_id[slot2];
    {
        uint64_t overflow_wr = ((uint64_t)(weightd_mesh.wire_epoch[peer] + 5000u) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)peer;
        CHECK(SparkWeightdMeshSqWrite(peer,overflow_wr,1u,0u,0u,0u,0u,&tag) == SPARK_STATUS_BUSY,
            "stalled completions refuse write 513 instead of overwriting in-flight WQE");
    }
    CHECK(weightd_mesh.stat_sq_full_waits == waits_before + 1u,"SQ full wait counter increments on stalled overflow");
    CHECK(queue->produced == produced_before + depth,"produced does not advance on rejected overflow write");
    CHECK(memcmp(queue->buffer + (uint64_t)slot0 * 64u,saved_wqe0,sizeof(saved_wqe0)) == 0 && queue->wr_id[slot0] == saved_wr0,
        "in-flight WQE slot 0 is untouched after stalled overflow");
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    CHECK(SparkWeightdMeshPostWrite(peer_rank,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer,weightd_mesh.recv_mr->lkey,64u,0u) == SPARK_STATUS_BUSY,
        "SparkWeightdMeshPostWrite also refuses when SQ is full");
    CHECK(weightd_mesh.stat_sq_full_waits == waits_before + 2u,"PostWrite increments SQ full wait counter");
    CHECK(memcmp(queue->buffer + (uint64_t)slot0 * 64u,saved_wqe0,sizeof(saved_wqe0)) == 0 && queue->wr_id[slot0] == saved_wr0,
        "PostWrite refusal leaves in-flight WQE slot 0 intact");
    {
        uint64_t first_tail_wr = ((uint64_t)(weightd_mesh.wire_epoch[peer] + 1001u) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)peer;
        uint64_t next_wr0 = ((uint64_t)(weightd_mesh.wire_epoch[peer] + 5001u) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)peer;
        uint64_t next_wr1 = ((uint64_t)(weightd_mesh.wire_epoch[peer] + 5002u) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)peer;
        uint64_t next_wr2 = ((uint64_t)(weightd_mesh.wire_epoch[peer] + 5003u) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)peer;
        CHECK(spark_stub_ibv_complete(first_tail_wr,IBV_WC_SUCCESS) == 0,"inject single signaled completion for WQE pair 0..1");
        pthread_mutex_lock(&SparkWeightdMeshWireLock);
        CHECK(SparkWeightdMeshSqWrite(peer,next_wr0,0u,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer,
            weightd_mesh.recv_mr->lkey,1024u,0u,0) == SPARK_STATUS_OK,
            "SQ guard drains CQ internally and retires unsignaled + signaled WQEs");
        CHECK(weightd_mesh.stat_sq_full_waits == waits_before + 3u,"CQ-drain recovery counts as one SQ full wait");
        CHECK(queue->retired == retired_before + 2u,"signaled completion advances retired past both unsignaled WQE 0 and signaled WQE 1");
        CHECK(SparkWeightdMeshSqWrite(peer,next_wr1,1u,0u,0u,0u,64u,&tag) == SPARK_STATUS_OK,
            "second freed slot admits paired signaled write without waiting");
        SparkWeightdMeshSqRing(1u << peer);
        CHECK(weightd_mesh.stat_sq_full_waits == waits_before + 3u,"second slot uses retired headroom without extra wait");
        CHECK(SparkWeightdMeshSqWrite(peer,next_wr2,1u,0u,0u,0u,128u,&tag) == SPARK_STATUS_BUSY,
            "re-filled SQ rejects again before overwriting in-flight slot 2");
        CHECK(memcmp(queue->buffer + (uint64_t)slot2 * 64u,saved_wqe2,sizeof(saved_wqe2)) == 0 && queue->wr_id[slot2] == saved_wr2,
            "in-flight WQE slot 2 remains intact after wraparound boundary");
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    }
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-sq-stats.log",SPARK_WEIGHTD_MESH_DIR);
    weightd_mesh.stat_logged_ns = 0u;
    test_capture_begin(log_path);
    SparkWeightdMeshRepairAndReport();
    test_capture_end();
    CHECK(test_file_contains(log_path,"WD-MESH-STATS") && test_file_contains(log_path,"sq_full_waits=4"),
        "WD-MESH-STATS reports exact sq_full_waits count");
    (void)unlink(log_path);
    weightd_mesh.stat_logged_ns = saved_stat_logged_ns;
    {
        uint64_t final_tail_wr = ((uint64_t)(weightd_mesh.wire_epoch[peer] + 5002u) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)peer;
        CHECK(spark_stub_ibv_complete(final_tail_wr,IBV_WC_SUCCESS) == 0,"complete final signaled WQE to drain entire SQ");
        SparkWeightdMeshDrainCq();
        CHECK(queue->produced == queue->retired,"final signaled completion retires all 514 WQEs cleanly");
    }
    weightd_mesh_sq_wait_ns = saved_wait_ns;
}

static void test_tp16_20us_weightd_upgrades(uint32_t local_rank)
{
    uint32_t band = 6u;
    uint32_t peer,p2,first,count,ok,i;
    uint32_t peer_mask = 0xffffu & ~(1u << local_rank);
    uint64_t seq = (UINT64_C(51) << 32u) | 1u;
    uint64_t slot,slot_base,tag,word0,barriers_before;
    SparkWeightdMeshRoute route;
    SparkStubIbvPostedWork work;
    test_complete_range(0u,spark_stub_ibv_posted_count());

    ok = 1u;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        if (weightd_mesh.slot_send_tds[0][peer] == 0 || weightd_mesh.slot_send_pds[0][peer] == 0 ||
            weightd_mesh.slot_send_inline[0][peer] != 828u || weightd_mesh.send_queues[peer].max_inline != 828u)
            ok = 0u;
        for (p2 = peer + 1u; p2 < SPARK_WEIGHTD_MESH_PEERS; p2++)
            if (weightd_mesh.send_queues[peer].doorbell == weightd_mesh.send_queues[p2].doorbell)
                ok = 0u;
    }
    CHECK(ok == 1u && weightd_mesh.bf_all_unique == 1u,
        "tp16-20us: every send QP gets a dedicated TD+parent PD, unique BlueFlame UAR, and 828 B max_inline");

    slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + (seq & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    slot_base = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    tag = (UINT64_C(0x11) << 32u) | seq;
    word0 = UINT64_C(0x55aa112233445566);
    *(uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base) = word0;
    *(uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base + 640u) = tag;
    memset(&route,0,sizeof(route));
    route.fields.peer_mask = peer_mask;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    route.fields.compact_tail = 1u;
    first = spark_stub_ibv_posted_count();
    barriers_before = weightd_mesh.timing.ring_barriers;
    CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq,slot,640u,route,UINT64_C(1000)) == SPARK_STATUS_OK,
        "tp16-20us: compact_tail FULL 640 B route posts cleanly");
    count = spark_stub_ibv_posted_count() - first;
    ok = count == SPARK_WEIGHTD_MESH_PEERS ? 1u : 0u;
    for (i = 0u; i < count; i++)
    {
        if (spark_stub_ibv_posted(first + i,&work) != 0 || work.source != 0u ||
            work.length != 648u || (work.flags & IBV_SEND_SIGNALED) == 0u ||
            work.inline_word != word0 || work.inline_tail != tag)
            ok = 0u;
    }
    CHECK(ok == 1u && weightd_mesh.timing.ring_barriers == barriers_before,
        "tp16-20us: compact_tail FULL 640 B emits 15 single inline 648 B WQEs with zero ring barriers");
    test_complete_range(first,spark_stub_ibv_posted_count());

    seq++;
    slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + (seq & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    slot_base = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    tag = (UINT64_C(0x12) << 32u) | seq;
    *(uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base) = UINT64_C(0x9988776655443322);
    *(uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base + 4096u) = tag;
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq,slot,4096u,route,UINT64_C(2000)) == SPARK_STATUS_OK,
        "tp16-20us: compact_tail FULL 4096 B DMA route posts cleanly");
    count = spark_stub_ibv_posted_count() - first;
    ok = count == SPARK_WEIGHTD_MESH_PEERS ? 1u : 0u;
    for (i = 0u; i < count; i++)
    {
        if (spark_stub_ibv_posted(first + i,&work) != 0 || work.source == 0u ||
            work.length != 4104u || (work.flags & IBV_SEND_SIGNALED) == 0u ||
            *(uint64_t *)(uintptr_t)(work.source + 4096u) != tag)
            ok = 0u;
    }
    CHECK(ok == 1u,"tp16-20us: compact_tail FULL 4096 B emits 15 single fused 4104 B DMA WQEs");
    test_complete_range(first,spark_stub_ibv_posted_count());

    for (i = 0u; i < 2u; i++)
    {
        uint64_t half_off;
        seq++;
        slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((slot + 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        slot_base = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
        half_off = SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot);
        tag = (((uint64_t)(0x20u + i)) << 32u) | seq;
        for (peer = 0u; peer < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer++)
        {
            uint8_t *dst = (uint8_t *)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,peer) + half_off;
            *(uint64_t *)dst = UINT64_C(0xabc0000000000000) | ((uint64_t)i << 8u) | (uint64_t)peer;
            *(uint64_t *)(dst + 640u) = tag;
        }
        memset(&route,0,sizeof(route));
        route.fields.peer_mask = peer_mask;
        route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
        route.fields.slice_bytes = 640u;
        route.fields.compact_tail = 1u;
        first = spark_stub_ibv_posted_count();
        CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq,slot,640u,route,UINT64_C(3000) + i * 1000u) == SPARK_STATUS_OK,
            "tp16-20us: compact_tail PEER 640 B double-buffered staging half posts cleanly");
        count = spark_stub_ibv_posted_count() - first;
        ok = count == SPARK_WEIGHTD_MESH_PEERS ? 1u : 0u;
        for (p2 = 0u; p2 < count; p2++)
        {
            if (spark_stub_ibv_posted(first + p2,&work) != 0 || work.source != 0u ||
                work.length != 648u || (work.flags & IBV_SEND_SIGNALED) == 0u ||
                work.inline_tail != tag || ((work.inline_word >> 8u) & 0xffu) != i)
                ok = 0u;
        }
        CHECK(ok == 1u,"tp16-20us: compact_tail PEER 640 B emits 15 single inline 648 B WQEs from rotating staging halves");
        test_complete_range(first,spark_stub_ibv_posted_count());
    }

    seq++;
    slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((slot + 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    slot_base = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    tag = (UINT64_C(0x30) << 32u) | seq;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer++)
    {
        uint8_t *dst = (uint8_t *)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,peer);
        *(uint64_t *)(dst + 600000u) = tag;
    }
    memset(&route,0,sizeof(route));
    route.fields.peer_mask = peer_mask;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = 600000u;
    route.fields.compact_tail = 1u;
    first = spark_stub_ibv_posted_count();
    CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq,slot,600000u,route,UINT64_C(5000)) == SPARK_STATUS_OK,
        "tp16-20us: compact_tail PEER 600 KB above half-staging uses full slot offset 0");
    count = spark_stub_ibv_posted_count() - first;
    ok = count == SPARK_WEIGHTD_MESH_PEERS ? 1u : 0u;
    for (p2 = 0u; p2 < count; p2++)
    {
        if (spark_stub_ibv_posted(first + p2,&work) != 0 || work.source == 0u ||
            work.length != 600008u || (work.flags & IBV_SEND_SIGNALED) == 0u || *(uint64_t *)(uintptr_t)(work.source + 600000u) != tag)
            ok = 0u;
    }
    CHECK(ok == 1u,"tp16-20us: compact_tail PEER 600 KB emits 15 fused 600008 B DMA WQEs");
    test_complete_range(first,spark_stub_ibv_posted_count());

    {
        SparkWeightdMeshRoute bad;
        memset(&bad,0,sizeof(bad));
        bad.fields.peer_mask = peer_mask;
        bad.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
        bad.fields.compact_tail = 1u;
        CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq + 1u,slot,644u,bad,UINT64_C(6000)) == SPARK_STATUS_INVALID_ARGUMENT,
            "tp16-20us: unaligned compact_tail rejects");
        bad.fields.pair = 1u;
        CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq + 1u,slot,640u,bad,UINT64_C(6000)) == SPARK_STATUS_INVALID_ARGUMENT,
            "tp16-20us: compact_tail with pair rejects");
        bad.fields.pair = 0u;
        bad.fields.half = SPARK_WEIGHTD_MESH_ROUTE_HALF_LOWER;
        CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq + 1u,slot,640u,bad,UINT64_C(6000)) == SPARK_STATUS_INVALID_ARGUMENT,
            "tp16-20us: compact_tail with half-slot rejects");
        bad.fields.half = SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE;
        bad.fields.reserved = 1u;
        CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq + 1u,slot,640u,bad,UINT64_C(6000)) == SPARK_STATUS_INVALID_ARGUMENT,
            "tp16-20us: compact_tail with reserved bit rejects");
        bad.fields.reserved = 0u;
        bad.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
        bad.fields.slice_bytes = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES;
        CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq + 1u,slot,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES,bad,UINT64_C(6000)) == SPARK_STATUS_INVALID_ARGUMENT,
            "tp16-20us: compact_tail PEER overflowing staging slot by 8 B rejects");
    }

    {
        SparkWeightdMeshSendQueue *q = &weightd_mesh.send_queues[0];
        uint32_t wrap_ok = 1u;
        uint8_t buf[648];
        for (i = 0u; i < 120u; i++)
        {
            uint64_t wr_id = ((uint64_t)(weightd_mesh.wire_epoch[0] + 7000u + i) << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT);
            uint64_t w_head = UINT64_C(0xdeadbeef00000000) | (uint64_t)i;
            uint64_t w_tail = UINT64_C(0xcafebabe00000000) | (uint64_t)i;
            first = spark_stub_ibv_posted_count();
            memset(buf,(int)(i & 0xffu),sizeof(buf));
            memcpy(buf,&w_head,8u);
            memcpy(buf + 640u,&w_tail,8u);
            pthread_mutex_lock(&SparkWeightdMeshWireLock);
            if (SparkWeightdMeshSqWrite(0u,wr_id,1u,(uint64_t)(uintptr_t)buf,weightd_mesh.recv_mr->lkey,648u,(uint64_t)i * 64u,0) != SPARK_STATUS_OK)
                wrap_ok = 0u;
            SparkWeightdMeshSqRing(1u);
            pthread_mutex_unlock(&SparkWeightdMeshWireLock);
            if (spark_stub_ibv_posted_count() != first + 1u || spark_stub_ibv_posted(first,&work) != 0 ||
                work.source != 0u || work.length != 648u || work.inline_word != w_head || work.inline_tail != w_tail)
                wrap_ok = 0u;
            if (spark_stub_ibv_complete(wr_id,IBV_WC_SUCCESS) != 0)
                wrap_ok = 0u;
            SparkWeightdMeshDrainCq();
        }
        CHECK(wrap_ok == 1u && q->produced == q->retired,
            "tp16-20us: 120x 648 B (1320 WQEBBs) multi-BB inline writes wrap cleanly across 1024-BB SQ ring");
    }

    {
        uint32_t row_mask = (0xfu << ((local_rank >> 2u) << 2u)) & ~(1u << local_rank);
        uint32_t col_mask = (0x1111u << (local_rank & 3u)) & ~(1u << local_rank);
        uint32_t masks[2] = { row_mask, col_mask };
        uint32_t m;
        for (m = 0u; m < 2u; m++)
        {
            uint64_t half_off;
            seq++;
            slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((slot + 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
            half_off = SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot);
            tag = (UINT64_C(0x40) << 32u) | seq;
            for (peer = 0u; peer < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer++)
            {
                uint8_t *dst = (uint8_t *)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,peer) + half_off;
                *(uint64_t *)dst = UINT64_C(0x2d44000000000000) | (uint64_t)peer;
                *(uint64_t *)(dst + 640u) = tag;
            }
            memset(&route,0,sizeof(route));
            route.fields.peer_mask = masks[m];
            route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
            route.fields.slice_bytes = 640u;
            route.fields.compact_tail = 1u;
            first = spark_stub_ibv_posted_count();
            CHECK(SparkWeightdMeshPostSlot(band,local_rank,seq,slot,640u,route,UINT64_C(7000) + m * 1000u) == SPARK_STATUS_OK,
                "tp16-20us phase2: 2D 4x4 3-peer subset compact_tail PEER route posts cleanly");
            count = spark_stub_ibv_posted_count() - first;
            ok = count == 3u ? 1u : 0u;
            for (p2 = 0u; p2 < count; p2++)
            {
                if (spark_stub_ibv_posted(first + p2,&work) != 0 || work.source != 0u ||
                    work.length != 648u || (work.flags & IBV_SEND_SIGNALED) == 0u || work.inline_tail != tag)
                    ok = 0u;
            }
            CHECK(ok == 1u,"tp16-20us phase2: 2D 4x4 3-peer subset emits exactly 3 inline 648 B WQEs");
            test_complete_range(first,spark_stub_ibv_posted_count());
        }
    }
}

static uint32_t test_hybrid_write_wqe(SparkWeightdMeshHybridCtrl *ctrl,uint32_t band,uint64_t slot,
    uint64_t seq,uint64_t tag,uint32_t peer_mask,uint64_t bytes,uint32_t is_inline,uint32_t is_peer,uint64_t half_off,uint64_t pattern_base)
{
    uint32_t xfer_len = (uint32_t)(bytes + 8u);
    uint32_t ds = is_inline != 0u ? ((36u + xfer_len + 15u) >> 4u) : 3u;
    uint32_t num_bb = (ds + 3u) >> 2u;
    uint64_t slot_base = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    uint32_t i,gen;
    SparkWeightdMeshHybridTrigger *trig;
    __atomic_store_n(&ctrl->gpu_busy,1u,__ATOMIC_SEQ_CST);
    if ( __atomic_load_n(&ctrl->cpu_mode,__ATOMIC_ACQUIRE) != 1u || ctrl->owner_band != band )
    {
        __atomic_store_n(&ctrl->gpu_busy,0u,__ATOMIC_RELEASE);
        return 0u;
    }
    gen = (uint32_t)__atomic_load_n(&ctrl->grant_gen,__ATOMIC_ACQUIRE);
    for (i = 0u; i < ctrl->peer_count; i++)
    {
        const SparkWeightdMeshHybridPeer *hp = &ctrl->peers[i];
        uint32_t sq_lim,wr_lim;
        if ( (peer_mask & (1u << hp->target)) == 0u )
            continue;
        sq_lim = __atomic_load_n(&ctrl->cpu_sq_limit[i],__ATOMIC_ACQUIRE);
        wr_lim = __atomic_load_n(&ctrl->cpu_wr_limit[i],__ATOMIC_ACQUIRE);
        if ( (uint32_t)(sq_lim - ctrl->gpu_pi[i]) < num_bb || (uint32_t)(wr_lim - ctrl->gpu_wrs[i]) < 1u )
        {
            __atomic_store_n(&ctrl->gpu_busy,0u,__ATOMIC_RELEASE);
            return 0u;
        }
    }
    for (i = 0u; i < ctrl->peer_count; i++)
    {
        const SparkWeightdMeshHybridPeer *hp = &ctrl->peers[i];
        SparkWeightdMeshSendQueue *q = &weightd_mesh.send_queues[hp->peer];
        uint32_t pi = ctrl->gpu_pi[i],sq_idx = pi & hp->sq_mask,sq_bytes = hp->sq_depth * 64u;
        uint8_t *sq = weightd_mesh.hybrid_shm.base + hp->sq_shm_off;
        struct mlx5_wqe_ctrl_seg *c = (struct mlx5_wqe_ctrl_seg *)(sq + (size_t)sq_idx * 64u);
        struct mlx5_wqe_raddr_seg *r = (struct mlx5_wqe_raddr_seg *)(c + 1);
        uint32_t w0,w1,next_pi;
        if ( (peer_mask & (1u << hp->target)) == 0u )
            continue;
        w0 = htobe32(((pi & 0xffffu) << 8u) | MLX5_OPCODE_RDMA_WRITE);
        w1 = htobe32(hp->qpn_shifted | ds);
        c->opmod_idx_opcode = w0;
        c->qpn_ds = w1;
        c->signature = 0u;
        c->dci_stream_channel_id = 0u;
        c->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
        c->imm = 0u;
        r->raddr = htobe64(hp->remote_base + slot_base);
        r->rkey = hp->rkey_be;
        r->reserved = 0u;
        if ( is_inline != 0u )
        {
            struct mlx5_wqe_inl_data_seg *seg = (struct mlx5_wqe_inl_data_seg *)(r + 1);
            uint32_t dst_off = sq_idx * 64u + 36u,b;
            uint64_t payload_word = pattern_base | (uint64_t)hp->target;
            seg->byte_count = htobe32(xfer_len | MLX5_INLINE_SEG);
            for (b = 0u; b < bytes; b += 4u)
            {
                uint32_t val = b == 0u ? (uint32_t)payload_word : b == 4u ? (uint32_t)(payload_word >> 32u) : (uint32_t)(0x33000000u + b);
                memcpy(sq + ((dst_off + b) & (sq_bytes - 1u)),&val,4u);
            }
            {
                uint32_t tlo = (uint32_t)tag,thi = (uint32_t)(tag >> 32u);
                memcpy(sq + ((dst_off + (uint32_t)bytes) & (sq_bytes - 1u)),&tlo,4u);
                memcpy(sq + ((dst_off + (uint32_t)bytes + 4u) & (sq_bytes - 1u)),&thi,4u);
            }
        }
        else
        {
            struct mlx5_wqe_data_seg *d = (struct mlx5_wqe_data_seg *)(r + 1);
            uint64_t local_iova = is_peer != 0u ?
                (ctrl->staging_iova_base + (uint64_t)band * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES + hp->staging_peer_off + half_off) :
                (ctrl->recv_iova_base + slot_base);
            d->byte_count = htobe32(xfer_len);
            d->lkey = is_peer != 0u ? hp->staging_lkey_be : hp->recv_lkey_be;
            d->addr = htobe64(local_iova);
        }
        {
            uint32_t copy_bytes = num_bb * 64u,start_off = sq_idx * 64u;
            if ( start_off + copy_bytes <= sq_bytes )
                memcpy(q->buffer + start_off,sq + start_off,copy_bytes);
            else
            {
                uint32_t first = sq_bytes - start_off;
                memcpy(q->buffer + start_off,sq + start_off,first);
                memcpy(q->buffer,sq,copy_bytes - first);
            }
        }
        next_pi = pi + num_bb;
        ctrl->gpu_pi[i] = next_pi;
        ctrl->gpu_wrs[i]++;
        if ( ctrl->dbrec_in_shm != 0u )
        {
            uint32_t dbr_be = htobe32(next_pi & 0xffffu);
            memcpy(weightd_mesh.hybrid_shm.base + hp->dbrec_shm_off + 4u,&dbr_be,4u);
            q->record[MLX5_SND_DBR] = dbr_be;
        }
    }
    trig = &ctrl->trigger[seq & 1u];
    trig->meta = SparkWeightdMeshHybridMetaPack(peer_mask,ds,num_bb,gen);
    trig->slot = slot;
    trig->bytes = bytes;
    __atomic_store_n(&trig->seq,tag,__ATOMIC_RELEASE);
    __atomic_store_n(&ctrl->gpu_busy,0u,__ATOMIC_RELEASE);
    return 1u;
}

static void test_tp16_20us_phase3_hybrid(uint32_t local_rank)
{
    SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
    SparkWeightdMeshWaitRequest *req;
    void *dbr_ptrs[SPARK_WEIGHTD_MESH_PEERS],*sq_ptrs[SPARK_WEIGHTD_MESH_PEERS],*reused,*bad;
    uint32_t peer,i,first,count,ok,band = 0u,index = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank;
    uint32_t peer_mask = 0xffffu & ~(1u << local_rank);
    uint64_t seq,slot,tag,rings_before,revokes_before;
    SparkStubIbvPostedWork work;

    CHECK(weightd_mesh.hybrid_shm.base != 0 && weightd_mesh.hybrid_shm.bytes == SPARK_WEIGHTD_MESH_HYBRID_SHM_BYTES &&
          ctrl != 0 && ctrl->magic == SPARK_WEIGHTD_MESH_HYBRID_MAGIC && ctrl->version == SPARK_WEIGHTD_MESH_HYBRID_VERSION &&
          ctrl->shm_path[0] != '\0' && ctrl->recv_iova_base == (uint64_t)(uintptr_t)weightd_mesh.recv_buffer &&
          ctrl->staging_iova_base == (uint64_t)(uintptr_t)weightd_mesh.staging_buffer,
        "tp16-20us phase3: SHM slab and hybrid control header initialized cleanly inside doorbell tail");

    ok = 1u;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        dbr_ptrs[peer] = SparkWeightdMeshShmSlabAlloc(&weightd_mesh.hybrid_shm,8u,8u);
        sq_ptrs[peer] = SparkWeightdMeshShmSlabAlloc(&weightd_mesh.hybrid_shm,65536u,4096u);
        if ( dbr_ptrs[peer] == 0 || sq_ptrs[peer] == 0 )
            ok = 0u;
        else
        {
            uint32_t doff = SparkWeightdMeshShmOffset(dbr_ptrs[peer],8u);
            uint32_t soff = SparkWeightdMeshShmOffset(sq_ptrs[peer],65536u);
            if ( doff >= SPARK_WEIGHTD_MESH_HYBRID_DBR_ARENA_BYTES || (doff & 63u) != 0u ||
                 soff < SPARK_WEIGHTD_MESH_HYBRID_DBR_ARENA_BYTES || (soff & 4095u) != 0u )
                ok = 0u;
            if ( peer > 0u && doff == SparkWeightdMeshShmOffset(dbr_ptrs[peer - 1u],8u) )
                ok = 0u;
            weightd_mesh.send_queues[peer].dbrec_shm_off = doff;
            weightd_mesh.send_queues[peer].sq_shm_off = soff;
        }
    }
    CHECK(ok == 1u,"tp16-20us phase3: SHM slab allocates 15 cache-line-isolated DBRs and 15 page-aligned SQ buffers");
    SparkWeightdMeshShmSlabFree(&weightd_mesh.hybrid_shm,dbr_ptrs[0]);
    reused = SparkWeightdMeshShmSlabAlloc(&weightd_mesh.hybrid_shm,8u,8u);
    bad = SparkWeightdMeshShmSlabAlloc(&weightd_mesh.hybrid_shm,2u * 1024u * 1024u,4096u);
    CHECK(reused == dbr_ptrs[0] && bad == 0 && SparkWeightdMeshShmSlabAlloc(&weightd_mesh.hybrid_shm,64u,48u) == 0,
        "tp16-20us phase3: SHM slab reuses freed slots and rejects oversize or non-pow2 requests");

    CHECK(SparkWeightdMeshSetActivity(0u,1u) == SPARK_STATUS_OK,
        "tp16-20us phase3: activating sole lane 0 enables hybrid mode");
    req = (SparkWeightdMeshWaitRequest *)((uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET +
        index * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
    CHECK(weightd_mesh.hybrid_active == 1u && ctrl->cpu_mode == 1u && ctrl->owner_lane == 0u &&
          ctrl->owner_band == 0u && ctrl->owner_rank == local_rank && ctrl->peer_count == SPARK_WEIGHTD_MESH_PEERS &&
          ctrl->dbrec_in_shm == 0u && ctrl->grant_gen >= 1u &&
          (req->capabilities & SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ) != 0u,
        "tp16-20us phase3: hybrid grant activates cpu_mode=1, advertises CAPABILITY_HYBRID_SQ, and keeps CPU dbrec ownership");

    seq = weightd_mesh.issued[index] + 1u;
    slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    tag = (UINT64_C(0x41) << 32u) | seq;
    rings_before = weightd_mesh.stat_hybrid_rings;
    first = spark_stub_ibv_posted_count();
    CHECK(test_hybrid_write_wqe(ctrl,band,slot,seq,tag,peer_mask,640u,1u,1u,0u,UINT64_C(0xdeadbeef00000000)) == 1u,
        "tp16-20us phase3: mock GPU writes 15x 648B inline WQEs and dbrec directly into SHM");
    SparkWeightdMeshDoorbellPoll();
    count = spark_stub_ibv_posted_count() - first;
    ok = count == SPARK_WEIGHTD_MESH_PEERS && weightd_mesh.stat_hybrid_rings == rings_before + 1u ? 1u : 0u;
    for (i = 0u; i < count; i++)
    {
        if ( spark_stub_ibv_posted(first + i,&work) != 0 || work.source != 0u ||
             work.length != 648u || (work.flags & (IBV_SEND_INLINE | IBV_SEND_SIGNALED)) != (IBV_SEND_INLINE | IBV_SEND_SIGNALED) ||
             work.inline_tail != tag || (work.inline_word >> 32u) != UINT64_C(0xdeadbeef) )
            ok = 0u;
    }
    CHECK(ok == 1u,"tp16-20us phase3: 150ns CPU BlueFlame-only poll ships all 15 GPU-written 648B inline WQEs");
    test_complete_range(first,spark_stub_ibv_posted_count());
    CHECK(test_shipped(band,local_rank) == tag &&
          ctrl->cpu_sq_limit[0] == weightd_mesh.send_queues[ctrl->peers[0].peer].retired + weightd_mesh.send_queues[ctrl->peers[0].peer].depth &&
          ctrl->cpu_wr_limit[0] == weightd_mesh.send_queues[ctrl->peers[0].peer].wr_retired + SPARK_WEIGHTD_MESH_SEND_CAPACITY,
        "tp16-20us phase3: CQE drain advances shipped cell and replenishes GPU SQ and WR credits");

    seq++;
    slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    tag = (UINT64_C(0x42) << 32u) | seq;
    {
        uint64_t half_off = SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot);
        for (i = 0u; i < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; i++)
        {
            uint8_t *dst = (uint8_t *)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,i) + half_off;
            *(uint64_t *)(dst + 4096u) = tag;
        }
        first = spark_stub_ibv_posted_count();
        CHECK(test_hybrid_write_wqe(ctrl,band,slot,seq,tag,peer_mask,4096u,0u,1u,half_off,0u) == 1u,
            "tp16-20us phase3: mock GPU writes 15x 48B DMA WQEs for 4104B B4 payload");
        SparkWeightdMeshDoorbellPoll();
        count = spark_stub_ibv_posted_count() - first;
        ok = count == SPARK_WEIGHTD_MESH_PEERS ? 1u : 0u;
        for (i = 0u; i < count; i++)
        {
            if ( spark_stub_ibv_posted(first + i,&work) != 0 || work.source == 0u ||
                 work.length != 4104u || (work.flags & IBV_SEND_SIGNALED) == 0u ||
                 work.lkey != weightd_mesh.staging_mr->lkey ||
                 *(uint64_t *)(uintptr_t)(work.source + 4096u) != tag )
                ok = 0u;
        }
        CHECK(ok == 1u,"tp16-20us phase3: CPU BlueFlame-only poll ships all 15 GPU-written 48B DMA WQEs with exact IOVA");
        test_complete_range(first,spark_stub_ibv_posted_count());
    }

    {
        uint32_t row_mask = (0xfu << (local_rank & ~3u)) & ~(1u << local_rank);
        seq++;
        slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        tag = (UINT64_C(0x43) << 32u) | seq;
        first = spark_stub_ibv_posted_count();
        CHECK(test_hybrid_write_wqe(ctrl,band,slot,seq,tag,row_mask,640u,1u,1u,0u,UINT64_C(0x2d44000000000000)) == 1u,
            "tp16-20us phase3: mock GPU writes 2D 4x4 3-peer subset hybrid WQEs");
        SparkWeightdMeshDoorbellPoll();
        count = spark_stub_ibv_posted_count() - first;
        CHECK(count == 3u,"tp16-20us phase3: 2D 4x4 3-peer subset rings exactly 3 peers");
        test_complete_range(first,spark_stub_ibv_posted_count());
    }

    ok = 1u;
    for (i = 0u; i < 120u; i++)
    {
        uint32_t w;
        seq++;
        slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        tag = (UINT64_C(0x50) << 32u) | seq;
        first = spark_stub_ibv_posted_count();
        if ( test_hybrid_write_wqe(ctrl,band,slot,seq,tag,peer_mask,640u,1u,1u,0u,UINT64_C(0x7700000000000000) | ((uint64_t)i << 16u)) == 0u )
        { ok = 0u; break; }
        SparkWeightdMeshDoorbellPoll();
        count = spark_stub_ibv_posted_count() - first;
        if ( count != SPARK_WEIGHTD_MESH_PEERS ) { ok = 0u; break; }
        for (w = 0u; w < count; w++)
            if ( spark_stub_ibv_posted(first + w,&work) != 0 || work.length != 648u || work.inline_tail != tag )
                ok = 0u;
        test_complete_range(first,spark_stub_ibv_posted_count());
    }
    CHECK(ok == 1u && test_shipped(band,local_rank) == tag,
        "tp16-20us phase3: 120 consecutive 648B inline hybrid rounds (1320 WQEBBs > 1024) wrap SQ ring cleanly");

    {
        uint64_t gen_before = ctrl->grant_gen;
        uint32_t other_rank = local_rank == 0u ? 1u : 0u;
        revokes_before = weightd_mesh.stat_hybrid_revokes;
        seq++;
        slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        tag = (UINT64_C(0x60) << 32u) | seq;
        first = spark_stub_ibv_posted_count();
        CHECK(test_hybrid_write_wqe(ctrl,band,slot,seq,tag,peer_mask,640u,1u,1u,0u,UINT64_C(0x6060000000000000)) == 1u,
            "tp16-20us phase3: queue pending hybrid trigger right before CPU Broadcast revoke");
        CHECK(SparkWeightdMeshBroadcast(1u << other_rank,0u,64u,0u,0u,0u) == 1u,
            "tp16-20us phase3: CPU Broadcast revokes hybrid, drains pending GPU trigger first, then posts CPU WQE");
        count = spark_stub_ibv_posted_count() - first;
        CHECK(count == SPARK_WEIGHTD_MESH_PEERS + 1u && weightd_mesh.stat_hybrid_revokes > revokes_before &&
              weightd_mesh.hybrid_active == 1u && ctrl->cpu_mode == 1u && ctrl->grant_gen > gen_before &&
              ctrl->cpu_issued_seq == tag &&
              ctrl->gpu_pi[0] == weightd_mesh.send_queues[ctrl->peers[0].peer].produced,
            "tp16-20us phase3: revoke preserved pending 15 GPU WQEs + 1 Broadcast WQE and re-granted synced gpu_pi");
        test_complete_range(first,spark_stub_ibv_posted_count());
    }

    CHECK(SparkWeightdMeshSetActivity(1u,1u) == SPARK_STATUS_OK && weightd_mesh.hybrid_active == 0u && ctrl->cpu_mode == 0u,
        "tp16-20us phase3: activating second concurrent lane revokes hybrid mode cleanly");
    CHECK(SparkWeightdMeshSetActivity(1u,0u) == SPARK_STATUS_OK && weightd_mesh.hybrid_active == 1u && ctrl->cpu_mode == 1u,
        "tp16-20us phase3: returning to single active lane automatically re-grants hybrid mode");

    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
        weightd_mesh.send_queues[peer].dbrec_shm_off = SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE;
    SparkWeightdMeshHybridRevokeLocked();
    SparkWeightdMeshHybridRefreshLocked(band);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    CHECK(weightd_mesh.hybrid_active == 1u && ctrl->dbrec_in_shm == 0u,
        "tp16-20us phase3: dbrec_in_shm=0 mode stays active when SQ is in SHM but dbrec is outside SHM");
    seq++;
    slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    tag = (UINT64_C(0x70) << 32u) | seq;
    first = spark_stub_ibv_posted_count();
    CHECK(test_hybrid_write_wqe(ctrl,band,slot,seq,tag,peer_mask,640u,1u,1u,0u,UINT64_C(0x7070000000000000)) == 1u,
        "tp16-20us phase3: mock GPU writes 648B inline WQEs without touching dbrec when dbrec_in_shm=0");
    SparkWeightdMeshDoorbellPoll();
    CHECK(spark_stub_ibv_posted_count() - first == SPARK_WEIGHTD_MESH_PEERS,
        "tp16-20us phase3: CPU writes dbrec[1] and rings BlueFlame cleanly in dbrec_in_shm=0 mode");
    test_complete_range(first,spark_stub_ibv_posted_count());

    {
        uint64_t cpu_seq = ++seq;
        uint64_t cpu_tag = (UINT64_C(0x71) << 32u) | cpu_seq;
        uint64_t cpu_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((cpu_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        uint64_t hyb_seq = ++seq;
        uint64_t hyb_tag = (UINT64_C(0x71) << 32u) | hyb_seq;
        uint64_t hyb_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((hyb_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        SparkWeightdMeshRoute cpu_route = {0};
        volatile uint64_t *cpu_cell = SparkWeightdMeshDoorbellAt(
            band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank,(uint32_t)(cpu_seq & 1u));
        cpu_route.fields.peer_mask = peer_mask;
        cpu_route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + cpu_slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES +
            SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u) = cpu_tag;
        cpu_cell[1] = 256u;
        cpu_cell[2] = cpu_slot;
        cpu_cell[3] = cpu_route.word;
        __atomic_store_n(&cpu_cell[0],cpu_tag,__ATOMIC_RELEASE);
        CHECK(test_hybrid_write_wqe(ctrl,band,hyb_slot,hyb_seq,hyb_tag,peer_mask,640u,1u,1u,0u,UINT64_C(0x7171000000000000)) == 1u,
            "tp16-20us phase3: queue higher-seq hybrid trigger while lower-seq CPU doorbell is pending");
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        CHECK(weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == cpu_tag &&
              ctrl->cpu_issued_seq == cpu_tag &&
              weightd_mesh.hybrid_deferred_count == 1u && weightd_mesh.hybrid_active == 0u &&
              spark_stub_ibv_posted_count() - first == 2u * SPARK_WEIGHTD_MESH_PEERS,
            "tp16-20us phase3: hybrid poll yields to lower-seq CPU doorbell and defers higher-seq hybrid trigger");
        test_complete_range(first,spark_stub_ibv_posted_count());
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        count = spark_stub_ibv_posted_count() - first;
        ok = count == SPARK_WEIGHTD_MESH_PEERS ? 1u : 0u;
        for (i = 0u; i < count; i++)
        {
            if ( spark_stub_ibv_posted(first + i,&work) != 0 || work.length != 648u ||
                 work.inline_tail != hyb_tag || (work.inline_word >> 32u) != UINT64_C(0x71710000) )
                ok = 0u;
        }
        CHECK(ok == 1u && weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == hyb_tag &&
              ctrl->cpu_issued_seq == hyb_tag && weightd_mesh.hybrid_deferred_count == 0u &&
              weightd_mesh.hybrid_active == 1u && ctrl->cpu_mode == 1u,
            "tp16-20us phase3: deferred inline hybrid trigger drains in order after CPU doorbell and re-grants hybrid");
        test_complete_range(first,spark_stub_ibv_posted_count());
    }

    {
        uint32_t row_mask = (0xfu << (local_rank & ~3u)) & ~(1u << local_rank);
        uint32_t col_mask = (0x1111u << (local_rank & 3u)) & ~(1u << local_rank);
        uint64_t cpu_seq = ++seq;
        uint64_t cpu_tag = (UINT64_C(0x71) << 32u) | cpu_seq;
        uint64_t cpu_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((cpu_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        uint64_t hyb_seq = ++seq;
        uint64_t hyb_tag = (UINT64_C(0x71) << 32u) | hyb_seq;
        uint64_t hyb_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((hyb_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        uint64_t cpu_half_off = SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(cpu_slot);
        SparkWeightdMeshRoute cpu_route = {0};
        volatile uint64_t *cpu_cell = SparkWeightdMeshDoorbellAt(
            band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank,(uint32_t)(cpu_seq & 1u));
        cpu_route.fields.peer_mask = col_mask;
        cpu_route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
        cpu_route.fields.compact_tail = 1u;
        cpu_route.fields.slice_bytes = 32256u;
        for (i = 0u; i < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; i++)
            *(uint64_t *)((uint8_t *)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,i) + cpu_half_off + 32256u) = cpu_tag;
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + hyb_slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES + 129024u) = hyb_tag;
        cpu_cell[1] = 32256u;
        cpu_cell[2] = cpu_slot;
        cpu_cell[3] = cpu_route.word;
        __atomic_store_n(&cpu_cell[0],cpu_tag,__ATOMIC_RELEASE);
        CHECK(test_hybrid_write_wqe(ctrl,band,hyb_slot,hyb_seq,hyb_tag,row_mask,129024u,0u,0u,0u,0u) == 1u,
            "tp16-20us phase3: queue 2D 4x4 Hop4 129KB FULL DMA hybrid trigger behind lower-seq col CPU doorbell");
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        CHECK(weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == cpu_tag &&
              spark_stub_ibv_posted_count() - first == 3u && weightd_mesh.hybrid_deferred_count == 1u,
            "tp16-20us phase3: 2D col CPU doorbell posts 3 WQEs and defers 2D Hop4 row DMA trigger");
        test_complete_range(first,spark_stub_ibv_posted_count());
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        count = spark_stub_ibv_posted_count() - first;
        ok = count == 3u ? 1u : 0u;
        for (i = 0u; i < count; i++)
        {
            if ( spark_stub_ibv_posted(first + i,&work) != 0 || work.length != 129032u ||
                 work.lkey != weightd_mesh.recv_mr->lkey ||
                 *(uint64_t *)(uintptr_t)(work.source + 129024u) != hyb_tag )
                ok = 0u;
        }
        CHECK(ok == 1u && weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == hyb_tag &&
              ctrl->cpu_issued_seq == hyb_tag && weightd_mesh.hybrid_deferred_count == 0u &&
              weightd_mesh.hybrid_active == 1u && ctrl->cpu_mode == 1u,
            "tp16-20us phase3: deferred 2D Hop4 129KB FULL DMA trigger drains all 3 row peers and re-grants hybrid");
        test_complete_range(first,spark_stub_ibv_posted_count());

        cpu_seq = ++seq;
        cpu_tag = (UINT64_C(0x71) << 32u) | cpu_seq;
        cpu_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((cpu_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        hyb_seq = ++seq;
        hyb_tag = (UINT64_C(0x71) << 32u) | hyb_seq;
        hyb_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((hyb_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        uint64_t hyb_half_off = SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(hyb_slot);
        memset(&cpu_route,0,sizeof(cpu_route));
        cpu_route.fields.peer_mask = row_mask;
        cpu_route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
        cpu_route.fields.compact_tail = 1u;
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + cpu_slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES + 129024u) = cpu_tag;
        for (i = 0u; i < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; i++)
            *(uint64_t *)((uint8_t *)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,i) + hyb_half_off + 32256u) = hyb_tag;
        cpu_cell = SparkWeightdMeshDoorbellAt(
            band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank,(uint32_t)(cpu_seq & 1u));
        cpu_cell[1] = 129024u;
        cpu_cell[2] = cpu_slot;
        cpu_cell[3] = cpu_route.word;
        __atomic_store_n(&cpu_cell[0],cpu_tag,__ATOMIC_RELEASE);
        CHECK(test_hybrid_write_wqe(ctrl,band,hyb_slot,hyb_seq,hyb_tag,col_mask,32256u,0u,1u,hyb_half_off,0u) == 1u,
            "tp16-20us phase3: queue 2D 4x4 PEER DMA hybrid trigger behind lower-seq row CPU doorbell");
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        CHECK(weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == cpu_tag &&
              spark_stub_ibv_posted_count() - first == 3u && weightd_mesh.hybrid_deferred_count == 1u,
            "tp16-20us phase3: 2D row CPU doorbell posts 3 WQEs and defers 2D PEER DMA trigger");
        test_complete_range(first,spark_stub_ibv_posted_count());
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        count = spark_stub_ibv_posted_count() - first;
        ok = count == 3u ? 1u : 0u;
        for (i = 0u; i < count; i++)
        {
            if ( spark_stub_ibv_posted(first + i,&work) != 0 || work.length != 32264u ||
                 work.lkey != weightd_mesh.staging_mr->lkey ||
                 *(uint64_t *)(uintptr_t)(work.source + 32256u) != hyb_tag )
                ok = 0u;
        }
        CHECK(ok == 1u && weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == hyb_tag &&
              ctrl->cpu_issued_seq == hyb_tag && weightd_mesh.hybrid_deferred_count == 0u &&
              weightd_mesh.hybrid_active == 1u && ctrl->cpu_mode == 1u,
            "tp16-20us phase3: deferred 2D PEER DMA trigger preserves staging lkey/half_off and re-grants hybrid");
        test_complete_range(first,spark_stub_ibv_posted_count());
    }

    {
        uint64_t old_seq = ++seq;
        uint64_t old_tag = (UINT64_C(0x72) << 32u) | old_seq;
        uint64_t old_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((old_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        uint32_t old_first = spark_stub_ibv_posted_count(),old_last,k;
        CHECK(test_hybrid_write_wqe(ctrl,band,old_slot,old_seq,old_tag,peer_mask,640u,1u,1u,0u,UINT64_C(0x7201000000000000)) == 1u,
            "tp16-20us phase3: issue hybrid round whose record slot stays pending until CQ drains");
        SparkWeightdMeshDoorbellPoll();
        old_last = spark_stub_ibv_posted_count();
        for (k = old_first; k < old_last; k++)
        {
            SparkStubIbvPostedWork w;
            CHECK(spark_stub_ibv_posted(k,&w) == 0 && spark_stub_ibv_complete(w.wr_id,IBV_WC_SUCCESS) == 0,
                "tp16-20us phase3: stage CQE for pending slot before revoke");
        }
        uint64_t hyb_seq = old_seq + SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
        uint64_t hyb_tag = (UINT64_C(0x72) << 32u) | hyb_seq;
        uint64_t hyb_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((hyb_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        uint64_t next_seq = hyb_seq + 1u;
        uint64_t next_tag = (UINT64_C(0x72) << 32u) | next_seq;
        uint64_t next_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((next_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        SparkWeightdMeshRoute next_route = {0};
        next_route.fields.peer_mask = peer_mask;
        next_route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + next_slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES +
            SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u) = next_tag;
        CHECK(test_hybrid_write_wqe(ctrl,band,hyb_slot,hyb_seq,hyb_tag,peer_mask,640u,1u,1u,0u,UINT64_C(0x7202000000000000)) == 1u,
            "tp16-20us phase3: queue hybrid trigger colliding with pending record slot before CPU revoke");
        first = spark_stub_ibv_posted_count();
        pthread_mutex_lock(&SparkWeightdMeshWireLock);
        CHECK(SparkWeightdMeshPostSlot(band,local_rank,next_tag,next_slot,256u,next_route,SparkMonotonicNs()) == SPARK_STATUS_OK,
            "tp16-20us phase3: CPU PostSlot revoke drains CQ and rings pending hybrid trigger before posting CPU round");
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        CHECK(weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == next_tag &&
              spark_stub_ibv_posted_count() - first == 3u * SPARK_WEIGHTD_MESH_PEERS,
            "tp16-20us phase3: revoke preserved collided hybrid trigger (15 WQEs) + CPU round (30 WQEs)");
        test_complete_range(first,spark_stub_ibv_posted_count());
        seq = next_seq;
    }

    {
        uint64_t base_seq = (uint32_t)seq;
        uint64_t e_seq = (base_seq & 1u) == 0u ? (base_seq + 2u) : (base_seq + 1u);
        uint64_t o_seq = e_seq + 1u;
        uint64_t next_epoch = (weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] >> 32u) + 1u;
        uint64_t e_tag = (next_epoch << 32u) | e_seq;
        uint64_t o_tag = (next_epoch << 32u) | o_seq;
        uint64_t e_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((e_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        uint64_t o_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((o_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        CHECK(test_hybrid_write_wqe(ctrl,band,e_slot,e_seq,e_tag,peer_mask,768u,1u,1u,0u,UINT64_C(0x7301000000000000)) == 1u &&
              test_hybrid_write_wqe(ctrl,band,o_slot,o_seq,o_tag,peer_mask,336u,1u,1u,0u,UINT64_C(0x7302000000000000)) == 1u,
            "tp16-20us phase3: stage back-to-back even (ds=51) and odd (ds=24) hybrid triggers");
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        CHECK(weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] == o_tag &&
              spark_stub_ibv_posted_count() - first == 2u * SPARK_WEIGHTD_MESH_PEERS &&
              SparkWeightdMeshCellFailed(band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank) == 0u,
            "tp16-20us phase3: hybrid poll rings both back-to-back variable-ds triggers in order with 0 mismatches");
        test_complete_range(first,spark_stub_ibv_posted_count());
        seq = o_seq;
    }

    {
        uint64_t bad_seq = (uint32_t)seq + 1u;
        uint64_t bad_epoch = (weightd_mesh.issued[band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank] >> 32u) + 1u;
        uint64_t bad_tag = (bad_epoch << 32u) | bad_seq;
        uint64_t bad_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((bad_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        const SparkWeightdMeshHybridPeer *hp0 = &ctrl->peers[0];
        SparkWeightdMeshSendQueue *q0 = &weightd_mesh.send_queues[hp0->peer];
        uint32_t bad_idx = q0->produced & q0->mask;
        CHECK(test_hybrid_write_wqe(ctrl,band,bad_slot,bad_seq,bad_tag,peer_mask,640u,1u,1u,0u,UINT64_C(0x7300000000000000)) == 1u,
            "tp16-20us phase3: stage hybrid WQE before corrupting SQ header");
        memset(q0->buffer + (size_t)bad_idx * 64u,0,8u);
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        CHECK(spark_stub_ibv_posted_count() == first &&
              SparkWeightdMeshCellFailed(band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank) != 0u,
            "tp16-20us phase3: hybrid ring rejects corrupted WQE header without ringing QP doorbells");
        pthread_mutex_lock(&SparkWeightdMeshWireLock);
        SparkWeightdMeshHybridRevokeLocked();
        SparkWeightdMeshLaneResetLocked(0u);
        SparkWeightdMeshHybridRefreshLocked(band);
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        CHECK(SparkWeightdMeshCellFailed(band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank) == 0u &&
              ctrl->cpu_issued_seq == 0u && weightd_mesh.hybrid_active == 1u,
            "tp16-20us phase3: lane reset clears failed cell and cpu_issued_seq after hybrid header mismatch");

        bad_seq = 1u;
        bad_tag = (UINT64_C(0x74) << 32u) | bad_seq;
        bad_slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
        bad_idx = q0->produced & q0->mask;
        CHECK(test_hybrid_write_wqe(ctrl,band,bad_slot,bad_seq,bad_tag,peer_mask,4096u,0u,0u,0u,0u) == 1u,
            "tp16-20us phase3: stage hybrid DMA WQE before corrupting remote address");
        memset(q0->buffer + (size_t)bad_idx * 64u + 16u,0xaa,8u);
        first = spark_stub_ibv_posted_count();
        SparkWeightdMeshDoorbellPoll();
        CHECK(spark_stub_ibv_posted_count() == first &&
              SparkWeightdMeshCellFailed(band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank) != 0u,
            "tp16-20us phase3: hybrid ring rejects corrupted DMA WQE raddr without ringing QP doorbells");
        pthread_mutex_lock(&SparkWeightdMeshWireLock);
        SparkWeightdMeshHybridRevokeLocked();
        SparkWeightdMeshLaneResetLocked(0u);
        SparkWeightdMeshHybridRefreshLocked(band);
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    }

    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    SparkWeightdMeshHybridRevokeLocked();
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        weightd_mesh.send_queues[peer].sq_shm_off = SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE;
        weightd_mesh.send_queues[peer].dbrec_shm_off = SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE;
        SparkWeightdMeshShmSlabFree(&weightd_mesh.hybrid_shm,dbr_ptrs[peer]);
        SparkWeightdMeshShmSlabFree(&weightd_mesh.hybrid_shm,sq_ptrs[peer]);
    }
    SparkWeightdMeshHybridRefreshLocked(0u);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    CHECK(SparkWeightdMeshSetActivity(0u,0u) == SPARK_STATUS_OK && weightd_mesh.hybrid_active == 0u,
        "tp16-20us phase3: teardown cleanly revokes hybrid mode and restores lane state");
}

static void test_tp16_20us_fallback(uint32_t local_rank)
{
    uint32_t peer,ok;
    test_rank_mask = 0xffffu;
    test_clean_dir();
    (void)mkdir(SPARK_WEIGHTD_MESH_DIR,0755);
    spark_stub_ibv_set_td_fail(1u);
    spark_stub_ibv_set_max_inline(256u);
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u) == SPARK_STATUS_BUSY,
        "tp16-20us fallback: init succeeds when ibv_alloc_td fails and max_inline is capped at 256 B");
    ok = 1u;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
        if (weightd_mesh.slot_send_tds[0][peer] != 0 || weightd_mesh.slot_send_pds[0][peer] != 0 ||
            weightd_mesh.slot_send_qps[0][peer] == 0 || weightd_mesh.slot_send_inline[0][peer] != 256u)
            ok = 0u;
    CHECK(ok == 1u,"tp16-20us fallback: falls back cleanly to shared PD and 256 B negotiated inline limit");
    spark_stub_ibv_set_td_fail(0u);
    spark_stub_ibv_set_max_inline(828u);
    (void)setenv("SPARK_WEIGHTD_MESH_HYBRID","0",1);
    CHECK(SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u) == SPARK_STATUS_BUSY &&
          weightd_mesh.hybrid_shm.base == 0 && weightd_mesh.hybrid_active == 0u,
        "tp16-20us phase3 fallback: SPARK_WEIGHTD_MESH_HYBRID=0 cleanly disables SHM slab and hybrid mode");
    (void)unsetenv("SPARK_WEIGHTD_MESH_HYBRID");
    test_clean_dir();
    (void)rmdir(SPARK_WEIGHTD_MESH_DIR);
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
    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u);
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

    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u);
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

    test_complete_range(0u,spark_stub_ibv_posted_count());
    test_mesh_topology();
    {
        SparkWeightdMeshTopology topology = test_identity_topology(16u,local_rank);
        test_complete_range(0u,spark_stub_ibv_posted_count());
        uint32_t lane;
        for (lane=0u; lane<SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
            CHECK(SparkWeightdMeshLaneConfigure(lane,&topology) == SPARK_STATUS_OK,
                "identity topology is explicitly configured for lifetime fixtures");
    }
    test_all_transfer_identities();
    test_slot_lifetimes(local_rank);
    test_slice_routes(local_rank);
    test_half_slots(local_rank);
    test_doorbell_pair_race(local_rank);
    test_peer_routes(local_rank);
    test_peer_halves(local_rank);
    test_pair_first_routes(local_rank);
    test_permutation_schedule(local_rank);
    test_e14_rdma_verbs_fastpath(local_rank);
    test_e14_r2_poll_fastpath(local_rank);
    test_sq_occupancy_guard(local_rank);
    test_tp16_20us_weightd_upgrades(local_rank);
    test_tp16_20us_phase3_hybrid(local_rank);
    test_mesh_peer_reset(local_rank);

    /* case 6: a second daemon instance with its own record dir must not
     * touch ours — the two-daemons-one-host separation (the fleet's
     * weightd vs the driver developers' standalone weightsd). */
    {
        char dir2[256];
        uint64_t dir1_boot = own_record.boot_ns;
        (void)snprintf(dir2,sizeof(dir2),"%s-second",SPARK_WEIGHTD_MESH_DIR);
        (void)mkdir(dir2,0755);
        status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,dir2,test_rank_mask,0,0u,0u);
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

    test_mesh_record_addresses(local_rank);
    test_mesh_status_surface(local_rank);

    test_clean_dir();
    test_rank_mask = 0xfu;
    CHECK(SparkWeightdMeshInit(0u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,0u,0,0u,0u) ==
        SPARK_STATUS_INVALID_ARGUMENT,"empty participant mask rejects");
    CHECK(SparkWeightdMeshInit(4u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u) ==
        SPARK_STATUS_INVALID_ARGUMENT,"rank outside participant mask rejects");
    CHECK(SparkWeightdMeshInit(0u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask,0,0u,0u) ==
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
    test_mesh_timing();
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
    CHECK(spark_stub_ibv_post_send_calls() - post_before == 3u,
        "TP4 B1 rings the queues of three peers only");
    test_mesh_lane_protocol();
    test_mesh_activity_protocol(protocol_post_first);
    test_load_pace_while_serving(0u);
    test_load_pace_while_serving(UINT64_C(8388608));
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
    test_mesh_pair_link(local_rank);
    test_mesh_second_rail(local_rank);
    test_mesh_ports(local_rank);
    test_tp16_20us_fallback(local_rank);
    fprintf(stderr,"%s: %u checks, %u failures\n",
        "test_weightd_mesh_mock",test_checks,test_failures);
    return test_failures != 0u ? 1 : 0;
}
