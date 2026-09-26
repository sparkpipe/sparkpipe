#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include <infiniband/verbs.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define SPARK_WEIGHTD_MESH_PEERS \
    (SPARK_WEIGHTD_MESH_RANKS_PER_BAND - 1u)
#define SPARK_WEIGHTD_MESH_CQ_ENTRIES 16384u
#define SPARK_WEIGHTD_MESH_SEND_CAPACITY 64u
#define SPARK_WEIGHTD_MESH_MAGIC UINT64_C(0x4d45534830303035)
#define SPARK_WEIGHTD_MESH_DORMANT_AFTER_NS UINT64_C(20000000)
#define SPARK_WEIGHTD_MESH_DORMANT_POLL_NS 200000L
#define SPARK_WEIGHTD_MESH_FULL_SCAN_NS UINT64_C(1000000)
#define SPARK_WEIGHTD_MESH_STUCK_LIVE_SWEEPS 100000u
#define SPARK_WEIGHTD_MESH_REPAIR_INTERVAL_NS UINT64_C(10000000)
#define SPARK_WEIGHTD_MESH_STATS_NS UINT64_C(10000000000)
#define SPARK_WEIGHTD_MESH_TIMING_BUCKETS 24u
#define SPARK_WEIGHTD_MESH_TIMING_PEER_TEXT 48u
#ifndef SPARK_WEIGHTD_MESH_DIR
#define SPARK_WEIGHTD_MESH_DIR "/tmp/weightd-mesh"
#endif

/* Two weightd-line daemons can share one host (the fleet's weightd and the
 * driver developers' standalone weightsd): the record directory must be
 * per-deployment or they clobber each other's mesh-<rank>.rec and .ready.
 * Set at init; the define is only the default. */
static const char *weightd_mesh_dir = SPARK_WEIGHTD_MESH_DIR;

static void SparkWeightdMeshReadyPath(char *path, uint64_t bytes)
{
    (void)snprintf(path,(size_t)bytes,"%s/.ready",weightd_mesh_dir);
}

typedef struct SparkWeightdMeshQpInfo
{
    uint32_t qp_number;
    uint32_t remote_qpn;
    uint16_t lid;
    uint8_t gid[16];
    uint32_t rkey;
    uint64_t remote_addr;
} SparkWeightdMeshQpInfo;

typedef struct SparkWeightdMeshRecord
{
    uint64_t magic;
    uint32_t rank;
    uint32_t send_qpn[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t recv_qpn[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t rkey;
    uint64_t recv_addr;
    uint16_t lid;
    uint8_t gid[16];
    uint32_t rank_mask;
    uint64_t boot_ns;
} SparkWeightdMeshRecord;

#define SPARK_WEIGHTD_MESH_TRANSFER_ID (UINT64_C(1) << 63u)

_Static_assert(SPARK_WEIGHTD_MESH_PEERS * 4u <= 64u,"mesh completion bitmap capacity");
_Static_assert(SPARK_WEIGHTD_MESH_PEERS * SPARK_WEIGHTD_MESH_SEND_CAPACITY <
    SPARK_WEIGHTD_MESH_CQ_ENTRIES,"mesh send completions fit CQ");
_Static_assert(SPARK_WEIGHTD_MESH_BANDS * SPARK_WEIGHTD_MESH_RANKS_PER_BAND <=
    512u,"mesh completion identity capacity");

typedef struct SparkWeightdMeshTransfer
{
    uint64_t seq;
    uint64_t generation;
    uint64_t pending;
    uint32_t failed;
} SparkWeightdMeshTransfer;

typedef struct SparkWeightdMeshTiming
{
    uint64_t post[SPARK_WEIGHTD_MESH_TIMING_BUCKETS];
    uint64_t ship[SPARK_WEIGHTD_MESH_TIMING_BUCKETS];
    uint64_t credit[SPARK_WEIGHTD_MESH_TIMING_BUCKETS];
    uint64_t gate[SPARK_WEIGHTD_MESH_TIMING_BUCKETS];
    uint64_t lag[SPARK_WEIGHTD_MESH_RANKS][SPARK_WEIGHTD_MESH_TIMING_BUCKETS];
    uint64_t last[SPARK_WEIGHTD_MESH_RANKS];
    uint64_t self;
} SparkWeightdMeshTiming;

typedef struct SparkWeightdMesh
{
    struct ibv_context *context;
    struct ibv_pd *protection_domain;
    struct ibv_mr *recv_mr;
    struct ibv_cq *cq;
    struct ibv_qp *send_qps[SPARK_WEIGHTD_MESH_PEERS];
    struct ibv_qp *recv_qps[SPARK_WEIGHTD_MESH_PEERS];
    SparkWeightdMeshQpInfo qp_info[SPARK_WEIGHTD_MESH_PEERS];
    void *recv_buffer;
    int memfd;
    uint64_t boot_ns;
    uint64_t record_check_ns;
    uint64_t artifact_check_ns;
    uint16_t lid;
    uint8_t gid[16];
    uint64_t wired_boot_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t send_pending[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t rpc_pending[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t send_ok;
    uint64_t send_err;
    uint64_t send_err_window;
    uint64_t quiesce_until_ns;
    uint32_t degraded_logged;
    uint64_t send_logged;
    uint64_t seq_storage;
    struct ibv_mr *seq_mr;
    uint64_t doorbell_posted[SPARK_WEIGHTD_MESH_BANDS *
        SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    uint32_t doorbell_stuck[SPARK_WEIGHTD_MESH_BANDS *
        SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    uint64_t ship_log_count;
    uint64_t ship_log_key;
    uint32_t ship_log_key_budget;
    SparkWeightdMeshTransfer transfers[SPARK_WEIGHTD_MESH_BANDS *
        SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    uint64_t wait_seen[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t wait_terminal[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t wait_started[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t publish_seq[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t publish_ns[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t post_ns[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t arrived[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint32_t last_peer[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    SparkWeightdMeshTiming timing;
    uint32_t activity_owners;
    uint32_t repair_pending;
    uint64_t repair_ns;
    uint64_t stat_trywire;
    uint64_t stat_record_failures;
    uint64_t stat_rewires;
    uint64_t stat_unready;
    uint64_t stat_repairs;
    uint64_t stat_logged_ns;
    uint64_t work_ns;
    uint64_t full_scan_ns;
    uint32_t lane_activity[SPARK_WEIGHTD_MESH_MAX_LANES];
    SparkWeightdMeshTopology lane_topology[SPARK_WEIGHTD_MESH_MAX_LANES];
    uint32_t mesh_active;
    uint32_t mesh_ready;
    uint32_t local_rank;
    uint32_t rank_mask;
    uint32_t sgid_index;
} SparkWeightdMesh;

static SparkWeightdMesh weightd_mesh;
static uint64_t SparkWeightdMeshNextTransferGeneration;


static uint64_t SparkWeightdMeshRealtimeNs(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME,&now) != 0)
        return 0ull;
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static uint64_t SparkWeightdMeshMonotonicNs(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC,&now) != 0)
        return 0ull;
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static uint64_t weightd_mesh_boot_phase_ns;
static void SparkWeightdMeshPhase(const char *name)
{
    uint64_t now = SparkWeightdMeshRealtimeNs();
    fprintf(stderr,"weightd-mesh phase %s at +%llu ms\n",name,
        (unsigned long long)((now - weightd_mesh_boot_phase_ns) / 1000000ull));
}


static SparkStatus SparkWeightdMeshWriteRecord(
    const SparkWeightdMeshRecord *record)
{
    char path[256];
    char temp[280];
    int fd;
    const char *cursor;
    size_t remaining;
    ssize_t written;

    (void)mkdir(weightd_mesh_dir,0755);
    snprintf(path,sizeof(path),"%s/mesh-%x.rec",
        weightd_mesh_dir,weightd_mesh.local_rank);
    snprintf(temp,sizeof(temp),"%s.tmp",path);
    fd = open(temp,O_WRONLY | O_CREAT | O_TRUNC,0644);
    if (fd < 0)
    {
        fprintf(stderr,"weightd-mesh: open %s failed errno=%d\n",temp,errno);
        return SPARK_STATUS_IO_ERROR;
    }
    cursor = (const char *)record;
    remaining = sizeof(*record);
    while (remaining > 0u)
    {
        written = write(fd,cursor,remaining);
        if (written <= 0)
        {
            (void)close(fd);
            (void)unlink(temp);
            return SPARK_STATUS_IO_ERROR;
        }
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
    if (fsync(fd) != 0)
    {
        (void)close(fd);
        (void)unlink(temp);
        return SPARK_STATUS_IO_ERROR;
    }
    (void)close(fd);
    if (rename(temp,path) != 0)
    {
        (void)unlink(temp);
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdMeshReadPeerRecord(
    uint32_t peer_rank,
    SparkWeightdMeshRecord *record)
{
    char path[256];
    int fd;
    char *cursor;
    size_t remaining;
    ssize_t bytes_read;

    snprintf(path,sizeof(path),"%s/mesh-%x.rec",
        weightd_mesh_dir,peer_rank);
    fd = open(path,O_RDONLY);
    if (fd < 0)
        return SPARK_STATUS_BUSY;
    cursor = (char *)record;
    remaining = sizeof(*record);
    while (remaining > 0u)
    {
        bytes_read = read(fd,cursor,remaining);
        if (bytes_read <= 0)
        {
            (void)close(fd);
            return SPARK_STATUS_BUSY;
        }
        cursor += (size_t)bytes_read;
        remaining -= (size_t)bytes_read;
    }
    (void)close(fd);
    if (record->magic != SPARK_WEIGHTD_MESH_MAGIC)
    {
        fprintf(stderr,"WD-MESH-ABI-MISMATCH peer=%u magic=%llx expected=%llx\n",
            peer_rank,(unsigned long long)record->magic,
            (unsigned long long)SPARK_WEIGHTD_MESH_MAGIC);
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if ( record->rank != peer_rank || record->rank_mask != weightd_mesh.rank_mask )
    {
        fprintf(stderr,"WD-MESH-GROUP-MISMATCH peer=%u rank=%u mask=%x expected=%x\n",
            peer_rank,record->rank,record->rank_mask,weightd_mesh.rank_mask);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdMeshTransitionQp(
    struct ibv_qp *qp,
    uint32_t remote_qpn,
    uint16_t dlid,
    const uint8_t *dgid)
{
    struct ibv_qp_attr attributes;
    int flags;

    memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RESET;
    if (ibv_modify_qp(qp,&attributes,IBV_QP_STATE) != 0)
    {
        fprintf(stderr,"weightd-mesh RESET failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_INIT;
    attributes.pkey_index = 0;
    attributes.port_num = 1;
    attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp,&attributes,flags) != 0)
    {
        fprintf(stderr,"weightd-mesh INIT failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = IBV_MTU_4096;
    attributes.dest_qp_num = remote_qpn;
    attributes.rq_psn = 0;
    attributes.max_dest_rd_atomic = 1;
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.dlid = dlid;
    attributes.ah_attr.port_num = 1;
    memcpy(attributes.ah_attr.grh.dgid.raw,dgid,16);
    attributes.ah_attr.grh.sgid_index = (uint8_t)weightd_mesh.sgid_index;
    attributes.ah_attr.grh.hop_limit = 1;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp,&attributes,flags) != 0)
    {
        fprintf(stderr,"weightd-mesh RTR failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTS;
    attributes.sq_psn = 0;
    attributes.timeout = 14;
    attributes.retry_cnt = 7;
    attributes.rnr_retry = 7;
    attributes.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp,&attributes,flags) != 0)
    {
        fprintf(stderr,"weightd-mesh RTS failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static pthread_mutex_t SparkWeightdMeshWireLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t SparkWeightdMeshActivityCondition = PTHREAD_COND_INITIALIZER;
static uint32_t SparkWeightdMeshWireWaiters;
static __thread uint32_t SparkWeightdMeshPollingThread;

static inline void SparkWeightdMeshCpuRelax(void)
{
#if defined(__aarch64__)
    __asm__ volatile ("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#else
#error "SparkWeightdMeshCpuRelax requires aarch64 or x86_64"
#endif
}

static void SparkWeightdMeshWireAcquire(void)
{
    if ( SparkWeightdMeshPollingThread != 0u )
    {
        while ( __atomic_load_n(&SparkWeightdMeshWireWaiters,__ATOMIC_ACQUIRE) != 0u )
            SparkWeightdMeshCpuRelax();
        pthread_mutex_lock(&SparkWeightdMeshWireLock);
        return;
    }
    __atomic_add_fetch(&SparkWeightdMeshWireWaiters,1u,__ATOMIC_ACQ_REL);
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    __atomic_sub_fetch(&SparkWeightdMeshWireWaiters,1u,__ATOMIC_ACQ_REL);
}

static void SparkWeightdMeshWake(void)
{
    weightd_mesh.work_ns = SparkWeightdMeshMonotonicNs();
    pthread_cond_broadcast(&SparkWeightdMeshActivityCondition);
}

static void SparkWeightdMeshAdvertiseLane(uint32_t lane)
{
    SparkWeightdMeshWaitRequest *requests;
    uint32_t index;
    requests = (SparkWeightdMeshWaitRequest *)((uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET);
    for (index=2u * lane * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index<2u * (lane + 1u) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
        __atomic_store_n(&requests[index].capabilities,(uint64_t)SPARK_WEIGHTD_MESH_CAPABILITIES,__ATOMIC_RELEASE);
}

SparkStatus SparkWeightdMeshLaneConfigure(uint32_t lane,
    const SparkWeightdMeshTopology *topology)
{
    SparkStatus status = SPARK_STATUS_OK;
    uint32_t rank,mask = 0u,index;
    if ( lane >= SPARK_WEIGHTD_MESH_MAX_LANES || topology == 0 ||
         topology->rank_count > SPARK_WEIGHTD_MESH_RANKS_PER_BAND )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( topology->rank_count == 0u ) return SPARK_STATUS_OK;
    if ( topology->local_rank >= topology->rank_count )
        return SPARK_STATUS_INVALID_ARGUMENT;
    SparkWeightdMeshWireAcquire();
    for (rank=0u; rank<topology->rank_count; rank++)
    {
        uint32_t physical = topology->physical_ranks[rank];
        if ( physical >= SPARK_WEIGHTD_MESH_RANKS_PER_BAND ||
             (mask & (1u << physical)) != 0u ||
             (weightd_mesh.rank_mask & (1u << physical)) == 0u )
        { status = SPARK_STATUS_INVALID_ARGUMENT; goto done; }
        mask |= 1u << physical;
    }
    if ( topology->physical_ranks[topology->local_rank] != weightd_mesh.local_rank )
    { status = SPARK_STATUS_INVALID_ARGUMENT; goto done; }
    if ( weightd_mesh.lane_topology[lane].rank_count != 0u &&
         (weightd_mesh.lane_topology[lane].rank_count != topology->rank_count ||
          weightd_mesh.lane_topology[lane].local_rank != topology->local_rank ||
          memcmp(weightd_mesh.lane_topology[lane].physical_ranks,topology->physical_ranks,
              topology->rank_count * sizeof(uint32_t)) != 0) )
    { status = SPARK_STATUS_UNSUPPORTED; goto done; }
    if ( weightd_mesh.lane_activity[lane] != 0u )
    { status = SPARK_STATUS_BUSY; goto done; }
    for (rank=0u; rank<SPARK_WEIGHTD_MESH_PEERS; rank++)
        if ( weightd_mesh.rpc_pending[rank] != 0u )
        { status = SPARK_STATUS_BUSY; goto done; }
    if ( weightd_mesh.recv_buffer == 0 )
    { status = SPARK_STATUS_BUSY; goto done; }
    for (index=2u * lane * SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
         index<2u * (lane + 1u) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
    {
        const uint64_t *entry = (const uint64_t *)((const uint8_t *)weightd_mesh.recv_buffer +
            SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + index * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
        const SparkWeightdMeshWaitRequest *request = (const SparkWeightdMeshWaitRequest *)(
            (const uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET +
            index * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
        if ( weightd_mesh.transfers[index].failed != 0u )
        { status = SPARK_STATUS_IO_ERROR; goto done; }
        if ( weightd_mesh.transfers[index].pending != 0u ||
             (__atomic_load_n(&entry[0],__ATOMIC_ACQUIRE) != 0u && entry[1] != 0u &&
              entry[0] != weightd_mesh.doorbell_posted[index]) ||
             __atomic_load_n(&request->request_id,__ATOMIC_ACQUIRE) > weightd_mesh.wait_terminal[index] )
        { status = SPARK_STATUS_BUSY; goto done; }
    }
    weightd_mesh.lane_topology[lane] = *topology;
    SparkWeightdMeshAdvertiseLane(lane);
done:
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return status;
}

SparkStatus SparkWeightdMeshSetActivity(uint32_t lane,uint32_t active)
{
    SparkStatus status = SPARK_STATUS_OK;
    SparkWeightdMeshWireAcquire();
    if ( active > 1u || lane >= SPARK_WEIGHTD_MESH_MAX_LANES )
        status = SPARK_STATUS_INVALID_ARGUMENT;
    else if ( weightd_mesh.lane_topology[lane].rank_count == 0u )
        status = SPARK_STATUS_INVALID_ARGUMENT;
    else if ( active != 0u && weightd_mesh.mesh_ready == 0u )
        status = SPARK_STATUS_BUSY;
    else if ( active != 0u && weightd_mesh.activity_owners == UINT32_MAX )
        status = SPARK_STATUS_CAPACITY_EXCEEDED;
    else if ( active == 0u && weightd_mesh.lane_activity[lane] == 0u )
        status = SPARK_STATUS_INVALID_ARGUMENT;
    else
    {
        if ( active != 0u )
        { weightd_mesh.activity_owners++; weightd_mesh.lane_activity[lane]++; }
        else
        { weightd_mesh.activity_owners--; weightd_mesh.lane_activity[lane]--; }
        SparkWeightdMeshWake();
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return status;
}

static uint32_t SparkWeightdMeshHasPending(void)
{
    uint32_t peer;
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
        if ( weightd_mesh.send_pending[peer] != 0u )
            return 1u;
    return 0u;
}

static uint32_t SparkWeightdMeshHasDoorbellWork(void)
{
    const volatile uint64_t *entries = (const volatile uint64_t *)(
        (const uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET);
    uint32_t index;
    for (index=0u; index<SPARK_WEIGHTD_MESH_BANDS * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
    {
        const volatile uint64_t *entry = entries + index *
            (SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES / 8u);
        if ( entry[0] != 0u && entry[1] != 0u &&
             entry[0] != weightd_mesh.doorbell_posted[index] )
            return 1u;
    }
    return 0u;
}

static uint32_t SparkWeightdMeshHasWaitWork(void)
{
    uint32_t index;
    if ( weightd_mesh.recv_buffer == 0 )
        return 0u;
    for (index=0u; index<SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS; index++)
    {
        SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)(
            (uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET +
            (uint64_t)index * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
        if ( __atomic_load_n(&request->request_id,__ATOMIC_ACQUIRE) >
             weightd_mesh.wait_terminal[index] )
            return 1u;
    }
    return 0u;
}

static uint32_t SparkWeightdMeshTimingBucket(uint64_t ns)
{
    uint64_t us = ns / 1000u;
    uint32_t bucket = us == 0u ? 0u : 64u - (uint32_t)__builtin_clzll(us);
    return(bucket < SPARK_WEIGHTD_MESH_TIMING_BUCKETS ? bucket : SPARK_WEIGHTD_MESH_TIMING_BUCKETS - 1u);
}

static void SparkWeightdMeshTimingAdd(uint64_t *histogram,uint64_t start_ns,uint64_t end_ns)
{
    if ( start_ns != 0u && end_ns >= start_ns )
        histogram[SparkWeightdMeshTimingBucket(end_ns - start_ns)]++;
}

static uint32_t SparkWeightdMeshPeersArrived(uint32_t index,uint32_t band,const SparkWeightdMeshWaitRequest *request,uint64_t now_ns,uint32_t first,uint64_t *diag)
{
    const SparkWeightdMeshTopology *topology = &weightd_mesh.lane_topology[band / 2u];
    uint64_t ring = (request->tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u),slot,tail,start;
    uint32_t peer,physical,complete = 1u;
    start = weightd_mesh.publish_seq[index] == request->tag ? weightd_mesh.publish_ns[index] : weightd_mesh.wait_started[index];
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer++)
    {
        if ( (request->peer_mask & (UINT64_C(1) << peer)) == 0u )
            continue;
        slot = (uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + (uint64_t)peer * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring;
        tail = __atomic_load_n((uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + (slot + 1u) * SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u),__ATOMIC_ACQUIRE);
        if ( tail != request->tag )
        {
            *diag = complete != 0u ? tail : *diag;
            complete = 0u;
            continue;
        }
        physical = topology->physical_ranks[peer];
        if ( (weightd_mesh.arrived[index] & (UINT64_C(1) << peer)) == 0u )
        {
            weightd_mesh.arrived[index] |= UINT64_C(1) << peer;
            weightd_mesh.last_peer[index] = physical;
            SparkWeightdMeshTimingAdd(weightd_mesh.timing.lag[physical],start,now_ns);
        }
    }
    if ( complete != 0u )
    {
        if ( first != 0u )
            weightd_mesh.timing.self++;
        else
            weightd_mesh.timing.last[weightd_mesh.last_peer[index]]++;
        SparkWeightdMeshTimingAdd(weightd_mesh.timing.gate,weightd_mesh.wait_started[index],now_ns);
    }
    return(complete);
}

static uint32_t SparkWeightdMeshWaitRequestsPollRange(uint64_t now_ns,uint32_t first,uint32_t end)
{
    uint32_t index,outstanding = 0u;
    if ( weightd_mesh.recv_buffer == 0 )
        return(0u);
    for (index=first; index<end; index++)
    {
        SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)(
            (uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET +
            (uint64_t)index * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
        uint64_t id = __atomic_load_n(&request->request_id,__ATOMIC_ACQUIRE);
        uint64_t error = 0u,diag = 0u;
        uint32_t complete = 0u,first = weightd_mesh.wait_seen[index] != id ? 1u : 0u;
        uint32_t band = index / SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        uint32_t rank = index % SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        const SparkWeightdMeshTopology *topology = &weightd_mesh.lane_topology[band / 2u];
        uint32_t logical_mask = (1u << topology->rank_count) - 1u;
        uint64_t cancel;
        if ( id == 0u || id <= weightd_mesh.wait_terminal[index] )
            continue;
        outstanding++;
        if ( first != 0u )
        {
            if ( weightd_mesh.wait_seen[index] > weightd_mesh.wait_terminal[index] )
                error = UINT64_MAX;
            weightd_mesh.wait_seen[index] = id;
            weightd_mesh.wait_started[index] = now_ns;
            weightd_mesh.arrived[index] = 0u;
        }
        cancel = __atomic_load_n((uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
            (uint64_t)(SPARK_WEIGHTD_MESH_DOORBELL_CELL_CANCEL + 2u * band) *
                SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES),__ATOMIC_ACQUIRE);
        if ( request->upstream_error != 0u )
            error = request->upstream_error;
        else if ( request->version != SPARK_WEIGHTD_MESH_WAIT_VERSION ||
                  (request->kind != SPARK_WEIGHTD_MESH_WAIT_SHIPPED &&
                   request->kind != SPARK_WEIGHTD_MESH_WAIT_PEERS) ||
                  request->timeout_ns == 0u || now_ns == 0u ||
                  topology->rank_count == 0u || rank != topology->local_rank ||
                  (request->tag != 0u && (uint32_t)request->tag == 0u) ||
                  (request->kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED && request->peer_mask != 0u) ||
                  (request->kind == SPARK_WEIGHTD_MESH_WAIT_PEERS &&
                   (request->tag == 0u || request->peer_mask == 0u ||
                    (request->peer_mask & ~(uint64_t)logical_mask) != 0u ||
                    (request->peer_mask & (UINT64_C(1) << rank)) != 0u)) ||
                  weightd_mesh.lane_activity[band / 2u] == 0u || weightd_mesh.mesh_ready == 0u )
            error = UINT64_MAX;
        else if ( cancel != request->cancel_expected )
        {
            error = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | request->tag;
            diag = cancel;
        }
        else if ( error == 0u && request->kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED )
        {
            diag = __atomic_load_n((uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
                SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank)),__ATOMIC_ACQUIRE);
            complete = request->tag == 0u || diag == request->tag;
            if ( complete == 0u && weightd_mesh.transfers[index].failed != 0u )
                error = UINT64_MAX;
            if ( complete != 0u )
                SparkWeightdMeshTimingAdd(weightd_mesh.timing.credit,weightd_mesh.wait_started[index],now_ns);
        }
        else if ( error == 0u )
            complete = SparkWeightdMeshPeersArrived(index,band,request,now_ns,first,&diag);
        if ( complete == 0u && error == 0u &&
             now_ns - weightd_mesh.wait_started[index] >= request->timeout_ns )
            error = request->tag != 0u ? request->tag : UINT64_MAX;
        if ( complete != 0u || error != 0u )
        {
            weightd_mesh.wait_terminal[index] = id;
            __atomic_store_n(&request->error,error,__ATOMIC_RELAXED);
            __atomic_store_n(&request->diag,diag,__ATOMIC_RELAXED);
            __atomic_store_n(&request->ready,1u,__ATOMIC_RELEASE);
        }
    }
    return(outstanding);
}

static uint32_t SparkWeightdMeshWaitRequestsPoll(uint64_t now_ns)
{
    return(SparkWeightdMeshWaitRequestsPollRange(now_ns,0u,SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS));
}

static void SparkWeightdMeshDormantWait(void)
{
    struct timespec wake;
    if ( SparkWeightdMeshMonotonicNs() - weightd_mesh.work_ns < SPARK_WEIGHTD_MESH_DORMANT_AFTER_NS ||
         clock_gettime(CLOCK_REALTIME,&wake) != 0 )
        return;
    wake.tv_nsec += SPARK_WEIGHTD_MESH_DORMANT_POLL_NS;
    if ( wake.tv_nsec >= 1000000000L )
    {
        wake.tv_sec++;
        wake.tv_nsec -= 1000000000L;
    }
    (void)pthread_cond_timedwait(&SparkWeightdMeshActivityCondition,&SparkWeightdMeshWireLock,&wake);
}

static void SparkWeightdMeshWaitForActivity(void)
{
    SparkWeightdMeshWireAcquire();
    while ( (weightd_mesh.mesh_ready == 0u && SparkWeightdMeshHasWaitWork() == 0u) ||
            (weightd_mesh.activity_owners == 0u &&
             SparkWeightdMeshHasPending() == 0u &&
             SparkWeightdMeshHasDoorbellWork() == 0u &&
             SparkWeightdMeshHasWaitWork() == 0u) )
        pthread_cond_wait(&SparkWeightdMeshActivityCondition,
            &SparkWeightdMeshWireLock);
    SparkWeightdMeshDormantWait();
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
}


typedef struct SparkWeightdMeshWirePlan
{
    SparkWeightdMeshRecord records[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t force_wire[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t changed, wired;
}
SparkWeightdMeshWirePlan;

static uint32_t SparkWeightdMeshQpInRts(struct ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init;
    if ( qp == 0 )
        return 1u;
    memset(&attr,0,sizeof(attr));
    memset(&init,0,sizeof(init));
    return ibv_query_qp(qp,&attr,IBV_QP_STATE,&init) == 0 && attr.qp_state == IBV_QPS_RTS ? 1u : 0u;
}

static void SparkWeightdMeshWireScanPeer(SparkWeightdMeshWirePlan *plan, uint32_t peer)
{
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u, send_in_rts, recv_in_rts;
    plan->force_wire[peer] = 0u;
    if ( (weightd_mesh.rank_mask & (1u << peer_rank)) == 0u )
        return;
    if ( SparkWeightdMeshReadPeerRecord(peer_rank,&plan->records[peer]) != SPARK_STATUS_OK )
    {
        plan->wired = 0u;
        plan->changed = 1u;
        weightd_mesh.stat_record_failures++;
        return;
    }
    if ( plan->records[peer].boot_ns != weightd_mesh.wired_boot_ns[peer] )
    {
        plan->changed = 1u;
        plan->force_wire[peer] = 1u;
        return;
    }
    send_in_rts = SparkWeightdMeshQpInRts(weightd_mesh.send_qps[peer]);
    recv_in_rts = SparkWeightdMeshQpInRts(weightd_mesh.recv_qps[peer]);
    plan->force_wire[peer] = SparkWeightdMeshRewireNeeded(plan->records[peer].boot_ns,weightd_mesh.wired_boot_ns[peer],send_in_rts,recv_in_rts);
    if ( plan->force_wire[peer] == 0u )
        return;
    plan->changed = 1u;
    if ( send_in_rts == 0u || recv_in_rts == 0u )
        fprintf(stderr,"WD-QP-REPAIR rank=%u peer=%u — qp left RTS after errors; re-transitioning on the same record\n",weightd_mesh.local_rank,peer);
}

static uint32_t SparkWeightdMeshWirePeerLocked(const SparkWeightdMeshWirePlan *plan, uint32_t peer)
{
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
    uint32_t mine = weightd_mesh.local_rank < peer_rank ? weightd_mesh.local_rank : weightd_mesh.local_rank - 1u;
    const SparkWeightdMeshRecord *record = &plan->records[peer];
    if ( SparkWeightdMeshTransitionQp(weightd_mesh.send_qps[peer],record->recv_qpn[mine],record->lid,record->gid) != SPARK_STATUS_OK ||
         SparkWeightdMeshTransitionQp(weightd_mesh.recv_qps[peer],record->send_qpn[mine],record->lid,record->gid) != SPARK_STATUS_OK )
    {
        fprintf(stderr,"WD-WIRE-FAIL rank=%u peer=%u errno=%d target_qpn=%u\n",weightd_mesh.local_rank,peer,errno,record->recv_qpn[mine]);
        return 0u;
    }
    weightd_mesh.qp_info[peer].remote_qpn = record->recv_qpn[mine];
    weightd_mesh.qp_info[peer].rkey = record->rkey;
    weightd_mesh.qp_info[peer].remote_addr = record->recv_addr;
    weightd_mesh.wired_boot_ns[peer] = record->boot_ns;
    weightd_mesh.stat_rewires++;
    fprintf(stderr,"WD-WIRED rank=%u peer=%u addr=%llx rkey=%u boot=%llu\n",weightd_mesh.local_rank,peer,(unsigned long long)record->recv_addr,record->rkey,(unsigned long long)record->boot_ns);
    return 1u;
}

static void SparkWeightdMeshWireReadyLocked(uint32_t wired)
{
    char ready_path[256];
    FILE *marker;
    SparkWeightdMeshReadyPath(ready_path,sizeof(ready_path));
    if ( wired == 0u )
    {
        if ( weightd_mesh.mesh_ready != 0u )
            weightd_mesh.stat_unready++;
        weightd_mesh.mesh_ready = 0u;
        (void)unlink(ready_path);
        return;
    }
    if ( weightd_mesh.mesh_ready == 0u && (marker = fopen(ready_path,"w")) != 0 )
        (void)fclose(marker);
    SparkWeightdMeshPhase("wired");
    weightd_mesh.mesh_ready = 1u;
    SparkWeightdMeshWake();
    weightd_mesh.send_err_window = 0ull;
    weightd_mesh.degraded_logged = 0u;
    weightd_mesh.quiesce_until_ns = 0ull;
    printf("weightd-mesh: ready rank=%u peers=%u rkey=%u\n",weightd_mesh.local_rank,(uint32_t)__builtin_popcount(weightd_mesh.rank_mask) - 1u,weightd_mesh.recv_mr->rkey);
    fflush(stdout);
}

static void SparkWeightdMeshTryWire(void)
{
    SparkWeightdMeshWirePlan plan;
    uint32_t peer;
    plan.changed = weightd_mesh.mesh_ready == 0u;
    plan.wired = 1u;
    weightd_mesh.stat_trywire++;
    for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++ )
        SparkWeightdMeshWireScanPeer(&plan,peer);
    if ( plan.changed == 0u )
        return;
    SparkWeightdMeshWireAcquire();
    for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++ )
        if ( plan.force_wire[peer] != 0u && SparkWeightdMeshWirePeerLocked(&plan,peer) == 0u )
            plan.wired = 0u;
    SparkWeightdMeshWireReadyLocked(plan.wired);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
}

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir, uint32_t rank_mask)
{
    struct ibv_device **devices;
    struct ibv_port_attr port_attr;
    struct ibv_qp_init_attr qp_attributes;
    SparkWeightdMeshRecord own_record;
    int device_count;
    uint32_t peer;

    if ( mesh_dir != 0 && mesh_dir[0] != '\0' )
        weightd_mesh_dir = mesh_dir;
    if (rank >= SPARK_WEIGHTD_MESH_RANKS || interface_name == 0 ||
        interface_name[0] == '\0' || sgid_index > 255u ||
        (rank_mask & (1u << rank)) == 0u ||
        (rank_mask >> SPARK_WEIGHTD_MESH_RANKS) != 0u ||
        __builtin_popcount(rank_mask) < 2)
    {
        fprintf(stderr,"weightd-mesh: bad init rank=%u interface=%s sgid=%u\n",
            rank,interface_name != 0 ? interface_name : "(null)",sgid_index);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&weightd_mesh,0,sizeof(weightd_mesh));
    weightd_mesh.local_rank = rank;
    weightd_mesh.rank_mask = rank_mask;
    weightd_mesh.sgid_index = sgid_index;
    weightd_mesh.boot_ns = SparkWeightdMeshRealtimeNs();
    weightd_mesh_boot_phase_ns = weightd_mesh.boot_ns;
    SparkWeightdMeshPhase("init-begin");
    {
        char ready_path[256];
        SparkWeightdMeshReadyPath(ready_path,sizeof(ready_path));
        (void)unlink(ready_path);
    }

    devices = ibv_get_device_list(&device_count);
    if (devices == 0 || device_count == 0)
    {
        fprintf(stderr,"weightd-mesh: no RDMA devices\n");
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    {
        int device_index;
        int found = 0;
        for (device_index = 0; device_index < device_count; device_index++)
        {
            const char *name = ibv_get_device_name(devices[device_index]);
            if (name != 0 && strcmp(name,interface_name) == 0)
            {
                weightd_mesh.context = ibv_open_device(devices[device_index]);
                found = 1;
                break;
            }
        }
        if (found == 0)
        {
            fprintf(stderr,"weightd-mesh: %s not found\n",interface_name);
            ibv_free_device_list(devices);
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
    }
    ibv_free_device_list(devices);
    if (ibv_query_port(weightd_mesh.context,1,&port_attr) != 0)
    {
        fprintf(stderr,"weightd-mesh: port query failed\n");
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    if (port_attr.state != IBV_PORT_ACTIVE)
    {
        fprintf(stderr,"weightd-mesh: switch port not active (state=%d)\n",
            port_attr.state);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    if (ibv_query_port(weightd_mesh.context,1,&port_attr) != 0)
    {
        fprintf(stderr,"weightd-mesh: port query failed\n");
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    weightd_mesh.protection_domain = ibv_alloc_pd(weightd_mesh.context);
    if (weightd_mesh.protection_domain == 0)
    {
        fprintf(stderr,"weightd-mesh: pd failed\n");
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    weightd_mesh.cq = ibv_create_cq(weightd_mesh.context,
        SPARK_WEIGHTD_MESH_CQ_ENTRIES,0,0,0);
    if (weightd_mesh.cq == 0)
    {
        fprintf(stderr,"weightd-mesh: cq failed\n");
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    weightd_mesh.memfd = memfd_create("spark-mesh",0u);
    if (weightd_mesh.memfd < 0)
    {
        fprintf(stderr,"weightd-mesh: memfd failed errno=%d\n",errno);
        return SPARK_STATUS_IO_ERROR;
    }
    if (ftruncate(weightd_mesh.memfd,SPARK_WEIGHTD_MESH_REGION_BYTES) != 0)
    {
        fprintf(stderr,"weightd-mesh: ftruncate failed errno=%d\n",errno);
        return SPARK_STATUS_IO_ERROR;
    }
    weightd_mesh.recv_buffer = mmap(0,SPARK_WEIGHTD_MESH_REGION_BYTES,
        PROT_READ | PROT_WRITE,MAP_SHARED,weightd_mesh.memfd,0);
    if (weightd_mesh.recv_buffer == MAP_FAILED)
    {
        fprintf(stderr,"weightd-mesh: mmap failed errno=%d\n",errno);
        weightd_mesh.recv_buffer = 0;
        return SPARK_STATUS_IO_ERROR;
    }
    {
        struct ibv_mr *burn[8];
        uint32_t burn_index;
        for (burn_index = 0u; burn_index < 8u; burn_index++)
        {
            burn[burn_index] = ibv_reg_mr(weightd_mesh.protection_domain,
                &weightd_mesh.seq_storage,sizeof(weightd_mesh.seq_storage),
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            if ( burn[burn_index] == 0 )
                break;
        }
        (void)burn;
    }
    weightd_mesh.recv_mr = ibv_reg_mr(weightd_mesh.protection_domain,
        weightd_mesh.recv_buffer,SPARK_WEIGHTD_MESH_REGION_BYTES,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (weightd_mesh.recv_mr == 0)
    {
        fprintf(stderr,"weightd-mesh: mr failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    printf("weightd-mesh region bytes=%llu (bands=%u ranks=%u slots/rank=%u rows/slot=%u slot_bytes=%u)\n",
        (unsigned long long)SPARK_WEIGHTD_MESH_REGION_BYTES,
        (unsigned)SPARK_WEIGHTD_MESH_BANDS,
        (unsigned)SPARK_WEIGHTD_MESH_RANKS_PER_BAND,
        (unsigned)SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        (unsigned)SPARK_WEIGHTD_MESH_SLOT_ROWS,
        (unsigned)SPARK_WEIGHTD_MESH_SLOT_BYTES);
    fflush(stdout);
    SparkWeightdMeshPhase("recv-mr-registered");
    weightd_mesh.seq_mr = ibv_reg_mr(weightd_mesh.protection_domain,
        &weightd_mesh.seq_storage,sizeof(weightd_mesh.seq_storage),
        IBV_ACCESS_LOCAL_WRITE);
    if (weightd_mesh.seq_mr == 0)
    {
        fprintf(stderr,"weightd-mesh: seq mr failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    memset(&qp_attributes,0,sizeof(qp_attributes));
    qp_attributes.send_cq = weightd_mesh.cq;
    qp_attributes.recv_cq = weightd_mesh.cq;
    qp_attributes.cap.max_send_wr = SPARK_WEIGHTD_MESH_SEND_CAPACITY;
    qp_attributes.cap.max_recv_wr = 64;
    qp_attributes.cap.max_send_sge = 1;
    qp_attributes.cap.max_recv_sge = 1;
    qp_attributes.cap.max_inline_data = 64;
    qp_attributes.qp_type = IBV_QPT_RC;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        uint32_t peer_rank = peer < rank ? peer : peer + 1u;
        if ( (rank_mask & (1u << peer_rank)) == 0u ) continue;
        weightd_mesh.send_qps[peer] =
            ibv_create_qp(weightd_mesh.protection_domain,&qp_attributes);
        weightd_mesh.recv_qps[peer] =
            ibv_create_qp(weightd_mesh.protection_domain,&qp_attributes);
        if (weightd_mesh.send_qps[peer] == 0 ||
            weightd_mesh.recv_qps[peer] == 0)
        {
            fprintf(stderr,"weightd-mesh: qp create peer=%u failed\n",peer);
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
    }
    memset(&own_record,0,sizeof(own_record));
    own_record.magic = SPARK_WEIGHTD_MESH_MAGIC;
    own_record.rank = weightd_mesh.local_rank;
    own_record.rank_mask = weightd_mesh.rank_mask;
    own_record.boot_ns = weightd_mesh.boot_ns;
    own_record.rkey = weightd_mesh.recv_mr->rkey;
    own_record.recv_addr = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
    own_record.lid = (uint16_t)port_attr.lid;
    weightd_mesh.lid = (uint16_t)port_attr.lid;
    {
        union ibv_gid gid;
        if (ibv_query_gid(weightd_mesh.context,1,(uint8_t)sgid_index,&gid) != 0)
        {
            fprintf(stderr,"weightd-mesh: gid query failed\n");
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        memcpy(own_record.gid,gid.raw,16);
        memcpy(weightd_mesh.gid,gid.raw,16);
    }
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        own_record.send_qpn[peer] = weightd_mesh.send_qps[peer] != 0 ? weightd_mesh.send_qps[peer]->qp_num : 0u;
        own_record.recv_qpn[peer] = weightd_mesh.recv_qps[peer] != 0 ? weightd_mesh.recv_qps[peer]->qp_num : 0u;
    }
    printf("weightd-mesh: rank=%u publishing %u QPs\n",
        weightd_mesh.local_rank,2u * ((uint32_t)__builtin_popcount(rank_mask) - 1u));
    fflush(stdout);
    if (SparkWeightdMeshWriteRecord(&own_record) != SPARK_STATUS_OK)
        return SPARK_STATUS_IO_ERROR;
    printf("weightd-mesh: published; wiring continues as peers appear\n");
    fflush(stdout);
    SparkWeightdMeshPhase("published");
    weightd_mesh.mesh_active = 1u;
    return SPARK_STATUS_BUSY;
}

void SparkWeightdMeshDeviceProbe(const char *tag,void *device_pointer,
    uint64_t bytes)
{
    struct ibv_mr *mr;
    if ( getenv("SPARK_WEIGHTD_MESH_DEVICE_PROBE") == 0 ||
         weightd_mesh.protection_domain == 0 || device_pointer == 0 ||
         bytes == 0ull )
        return;
    mr = ibv_reg_mr(weightd_mesh.protection_domain,device_pointer,(size_t)bytes,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if ( mr != 0 )
    {
        fprintf(stderr,
            "WD-DEVPROBE VA-OK tag=%s ptr=%llx bytes=%llu rkey=%u lkey=%u\n",
            tag,(unsigned long long)(uintptr_t)device_pointer,
            (unsigned long long)bytes,mr->rkey,mr->lkey);
        (void)ibv_dereg_mr(mr);
    }
    else
    {
        int dmabuf_fd = -1;
        struct ibv_mr *dmabuf_mr;
        fprintf(stderr,
            "WD-DEVPROBE VA-FAIL tag=%s errno=%d — trying the dmabuf route\n",
            tag,errno);
        if ( cuMemGetHandleForAddressRange((void *)&dmabuf_fd,
                 (CUdeviceptr)(uintptr_t)device_pointer,(size_t)bytes,
                 CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,0ull) == CUDA_SUCCESS &&
             dmabuf_fd >= 0 )
        {
            dmabuf_mr = ibv_reg_dmabuf_mr(weightd_mesh.protection_domain,0u,
                (size_t)bytes,(uint64_t)(uintptr_t)device_pointer,dmabuf_fd,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            if ( dmabuf_mr != 0 )
                fprintf(stderr,
                    "WD-DEVPROBE DMABUF-OK tag=%s fd=%d rkey=%u lkey=%u — GPUDirect via dmabuf WORKS; S2.5 unblocked\n",
                    tag,dmabuf_fd,dmabuf_mr->rkey,dmabuf_mr->lkey);
            else
                fprintf(stderr,
                    "WD-DEVPROBE DMABUF-FAIL tag=%s fd=%d errno=%d — S2.5 dead on this hardware; the 100us route is the graph path only\n",
                    tag,dmabuf_fd,errno);
            if ( dmabuf_mr != 0 )
                (void)ibv_dereg_mr(dmabuf_mr);
            (void)close(dmabuf_fd);
        }
        else
            fprintf(stderr,
                "WD-DEVPROBE DMABUF-HANDLE-FAIL tag=%s fd=%d — no dmabuf handle for the range; S2.5 dead on this hardware\n",
                tag,dmabuf_fd);
    }
    fflush(stderr);
}

uint32_t SparkWeightdMeshReady(void)
{
    return weightd_mesh.mesh_ready;
}

static void SparkWeightdMeshCompleteTransfer(uint64_t work_id, int success, uint64_t now_ns)
{
    uint32_t index;
    uint32_t bit;
    SparkWeightdMeshTransfer *transfer;
    if ( (work_id & SPARK_WEIGHTD_MESH_TRANSFER_ID) == 0u )
    {
        if ( work_id < SPARK_WEIGHTD_MESH_PEERS &&
             weightd_mesh.rpc_pending[work_id] != 0u )
        {
            weightd_mesh.rpc_pending[work_id]--;
            weightd_mesh.send_pending[work_id]--;
        }
        return;
    }
    index = (uint32_t)((work_id >> 6u) & 511u);
    bit = (uint32_t)(work_id & 63u);
    transfer = &weightd_mesh.transfers[index];
    if ( transfer->generation !=
            ((work_id & ~SPARK_WEIGHTD_MESH_TRANSFER_ID) >> 16u) ||
         (transfer->pending & (UINT64_C(1) << bit)) == 0u )
        return;
    transfer->pending &= ~(UINT64_C(1) << bit);
    weightd_mesh.send_pending[bit / 4u]--;
    if ( success == 0 )
        transfer->failed = 1u;
    if ( transfer->pending == 0u && transfer->failed == 0u )
    {
        uint32_t band = index / SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        uint32_t rank = index % SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
        weightd_mesh.doorbell_posted[index] = transfer->seq;
        weightd_mesh.doorbell_stuck[index] = 0u;
        SparkWeightdMeshTimingAdd(weightd_mesh.timing.ship,weightd_mesh.post_ns[index],now_ns);
        __sync_synchronize();
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank)) = transfer->seq;
        __sync_synchronize();
        weightd_mesh.ship_log_count++;
    }
}

static SparkStatus SparkWeightdMeshPostTransfer(uint32_t index, uint32_t peer,
    uint32_t kind, uint64_t source_offset, uint32_t length)
{
    SparkWeightdMeshTransfer *transfer = &weightd_mesh.transfers[index];
    struct ibv_sge scatter;
    struct ibv_send_wr request;
    struct ibv_send_wr *bad;
    uint32_t bit = peer * 4u + kind;
    uint64_t pending = UINT64_C(1) << bit;
    memset(&scatter,0,sizeof(scatter));
    scatter.addr = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer + source_offset;
    scatter.length = length;
    scatter.lkey = weightd_mesh.recv_mr->lkey;
    memset(&request,0,sizeof(request));
    request.wr_id = SPARK_WEIGHTD_MESH_TRANSFER_ID |
        (transfer->generation << 16u) | ((uint64_t)index << 6u) | bit;
    request.sg_list = &scatter;
    request.num_sge = 1;
    request.opcode = IBV_WR_RDMA_WRITE;
    request.send_flags = IBV_SEND_SIGNALED;
    request.wr.rdma.remote_addr = weightd_mesh.qp_info[peer].remote_addr + source_offset;
    request.wr.rdma.rkey = weightd_mesh.qp_info[peer].rkey;
    transfer->pending |= pending;
    if ( ibv_post_send(weightd_mesh.send_qps[peer],&request,&bad) != 0 )
    {
        transfer->pending &= ~pending;
        transfer->failed = 1u;
        fprintf(stderr,"WD-SHIP-POST-FAIL index=%u generation=%llu peer=%u kind=%u errno=%d\n",
            index,(unsigned long long)transfer->generation,peer,kind,errno);
        return SPARK_STATUS_IO_ERROR;
    }
    weightd_mesh.send_pending[peer]++;
    SparkWeightdMeshWake();
    return SPARK_STATUS_OK;
}

static uint32_t SparkWeightdMeshPhysicalMask(const SparkWeightdMeshTopology *topology,uint32_t peer_rank_mask,uint8_t *logical)
{
    uint32_t peer,physical_mask = 0u;
    for (peer=0u; peer<topology->rank_count; peer++)
    {
        logical[topology->physical_ranks[peer]] = (uint8_t)peer;
        if ( (peer_rank_mask & (1u << peer)) != 0u )
            physical_mask |= 1u << topology->physical_ranks[peer];
    }
    return(physical_mask);
}

static uint32_t SparkWeightdMeshSendRoom(uint32_t physical_mask)
{
    uint32_t peer,peer_rank;
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
        if ( (physical_mask & (1u << peer_rank)) != 0u && weightd_mesh.send_pending[peer] > SPARK_WEIGHTD_MESH_SEND_CAPACITY - 2u )
            return(0u);
    }
    return(1u);
}

static void SparkWeightdMeshPostRoute(uint32_t index,uint64_t slot_base,uint64_t bytes,SparkWeightdMeshRoute route,uint32_t physical_mask,const uint8_t *logical,uint32_t local)
{
    uint64_t offset,length;
    uint32_t peer,peer_rank;
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
        if ( (physical_mask & (1u << peer_rank)) == 0u )
            continue;
        offset = SparkWeightdMeshRouteSpan(route,logical[peer_rank],local,bytes,&length);
        if ( length == 0u || SparkWeightdMeshPostTransfer(index,peer,2u,slot_base + offset,(uint32_t)length) == SPARK_STATUS_OK )
            (void)SparkWeightdMeshPostTransfer(index,peer,3u,slot_base + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u,8u);
    }
}

static SparkStatus SparkWeightdMeshPostSlot(uint32_t band,uint32_t rank,uint64_t seq,uint64_t slot,uint64_t bytes,SparkWeightdMeshRoute route,uint64_t now_ns)
{
    uint32_t index = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank,physical_mask,peer_rank_mask = (uint32_t)route.fields.peer_mask;
    const SparkWeightdMeshTopology *topology;
    SparkWeightdMeshTransfer *transfer;
    uint8_t logical[SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    if ( band >= SPARK_WEIGHTD_MESH_BANDS || rank >= SPARK_WEIGHTD_MESH_RANKS_PER_BAND || seq == 0u || slot >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND || slot / SPARK_WEIGHTD_MESH_SLOTS_PER_RANK != rank || bytes == 0u || bytes > SPARK_WEIGHTD_MESH_SLOT_BYTES - 16u || peer_rank_mask == 0u || SparkWeightdMeshRouteValid(route) == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    topology = &weightd_mesh.lane_topology[band / 2u];
    if ( topology->rank_count == 0u || rank != topology->local_rank || (peer_rank_mask & ~((1u << topology->rank_count) - 1u)) != 0u || (peer_rank_mask & (1u << rank)) != 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    physical_mask = SparkWeightdMeshPhysicalMask(topology,peer_rank_mask,logical);
    if ( weightd_mesh.mesh_ready == 0u ) return SPARK_STATUS_BUSY;
    transfer = &weightd_mesh.transfers[index];
    if ( transfer->pending != 0u || transfer->failed != 0u )
        return transfer->failed != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_BUSY;
    if ( seq <= weightd_mesh.doorbell_posted[index] )
        return seq == weightd_mesh.doorbell_posted[index] ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
    if ( SparkWeightdMeshNextTransferGeneration == (SPARK_WEIGHTD_MESH_TRANSFER_ID >> 16u) - 1u )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    if ( SparkWeightdMeshSendRoom(physical_mask) == 0u )
        return SPARK_STATUS_BUSY;
    memset(transfer,0,sizeof(*transfer));
    transfer->seq = seq;
    transfer->generation = ++SparkWeightdMeshNextTransferGeneration;
    SparkWeightdMeshPostRoute(index,((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES,bytes,route,physical_mask,logical,rank);
    weightd_mesh.post_ns[index] = now_ns;
    if ( weightd_mesh.publish_seq[index] == seq )
        SparkWeightdMeshTimingAdd(weightd_mesh.timing.post,weightd_mesh.publish_ns[index],now_ns);
    return transfer->failed != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
}

static void SparkWeightdMeshDrainCq(void)
{
    struct ibv_wc completions[64];
    int completed;
    int index;
    uint32_t repair_needed = 0u;
    uint64_t now_ns;

    SparkWeightdMeshWireAcquire();
    now_ns = SparkWeightdMeshMonotonicNs();
    for (;;)
    {
        completed = ibv_poll_cq(weightd_mesh.cq,64,completions);
        if (completed <= 0)
            break;
        for (index = 0; index < completed; index++)
        {
            SparkWeightdMeshCompleteTransfer(completions[index].wr_id,
                completions[index].status == IBV_WC_SUCCESS,now_ns);
            if (completions[index].status == IBV_WC_SUCCESS)
            {
                weightd_mesh.send_ok++;
                weightd_mesh.send_err_window = 0ull;
            }
            else
            {
                weightd_mesh.send_err++;
                weightd_mesh.send_err_window++;
                if (completions[index].status != IBV_WC_WR_FLUSH_ERR)
                    fprintf(stderr,
                        "WD-MESH-CQERR wr=%llu opcode=%u status=%u vendor=%u\n",
                        (unsigned long long)completions[index].wr_id,
                        (unsigned)completions[index].opcode,
                        (unsigned)completions[index].status,
                        (unsigned)completions[index].vendor_err);
                repair_needed = 1u;
            }
        }
    }
    if (weightd_mesh.send_err_window > 256ull &&
        weightd_mesh.degraded_logged == 0u)
    {
        fprintf(stderr,
            "WD-MESH-DEGRADED window=%llu total=%llu — one 100ms quiesce pass\n",
            (unsigned long long)weightd_mesh.send_err_window,
            (unsigned long long)weightd_mesh.send_err);
        weightd_mesh.degraded_logged = 1u;
        weightd_mesh.quiesce_until_ns =
            SparkWeightdMeshRealtimeNs() + 100000000ull;
        weightd_mesh.send_err_window = 0ull;
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    if (repair_needed != 0u)
        __atomic_store_n(&weightd_mesh.repair_pending,1u,__ATOMIC_RELEASE);
}

static uint64_t SparkWeightdMeshTimingCount(const uint64_t *histogram)
{
    uint64_t total = 0u;
    uint32_t bucket;
    for (bucket=0u; bucket<SPARK_WEIGHTD_MESH_TIMING_BUCKETS; bucket++)
        total += histogram[bucket];
    return(total);
}

static uint64_t SparkWeightdMeshTimingPercentileUs(const uint64_t *histogram,uint32_t percent)
{
    uint64_t total = SparkWeightdMeshTimingCount(histogram),seen = 0u;
    uint32_t bucket;
    for (bucket=0u; bucket<SPARK_WEIGHTD_MESH_TIMING_BUCKETS && total != 0u; bucket++)
    {
        seen += histogram[bucket];
        if ( seen * 100u >= total * percent )
            return(UINT64_C(1) << bucket);
    }
    return(0u);
}

static void SparkWeightdMeshTimingPeers(const SparkWeightdMeshTiming *timing,char *text,size_t capacity)
{
    size_t used = 0u;
    uint32_t peer;
    text[0] = '\0';
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_RANKS && used < capacity; peer++)
        if ( SparkWeightdMeshTimingCount(timing->lag[peer]) != 0u )
            used += (size_t)snprintf(text + used,capacity - used,"%s%u:%llu/%llu/%llu",used != 0u ? "," : "",peer,(unsigned long long)SparkWeightdMeshTimingPercentileUs(timing->lag[peer],50u),(unsigned long long)SparkWeightdMeshTimingPercentileUs(timing->lag[peer],99u),(unsigned long long)timing->last[peer]);
}

static void SparkWeightdMeshTimingReport(void)
{
    SparkWeightdMeshTiming timing;
    char peers[SPARK_WEIGHTD_MESH_RANKS * SPARK_WEIGHTD_MESH_TIMING_PEER_TEXT];
    SparkWeightdMeshWireAcquire();
    timing = weightd_mesh.timing;
    memset(&weightd_mesh.timing,0,sizeof(weightd_mesh.timing));
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    if ( SparkWeightdMeshTimingCount(timing.post) == 0u && SparkWeightdMeshTimingCount(timing.credit) == 0u && SparkWeightdMeshTimingCount(timing.gate) == 0u )
        return;
    SparkWeightdMeshTimingPeers(&timing,peers,sizeof(peers));
    fprintf(stderr,"WD-MESH-TIMING posts=%llu post_us=%llu/%llu ship_us=%llu/%llu credits=%llu credit_us=%llu/%llu gates=%llu gate_us=%llu/%llu self=%llu peers=%s\n",
        (unsigned long long)SparkWeightdMeshTimingCount(timing.post),
        (unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.post,50u),(unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.post,99u),
        (unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.ship,50u),(unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.ship,99u),
        (unsigned long long)SparkWeightdMeshTimingCount(timing.credit),
        (unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.credit,50u),(unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.credit,99u),
        (unsigned long long)SparkWeightdMeshTimingCount(timing.gate),
        (unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.gate,50u),(unsigned long long)SparkWeightdMeshTimingPercentileUs(timing.gate,99u),
        (unsigned long long)timing.self,peers);
}

static void SparkWeightdMeshRepairAndReport(void)
{
    uint64_t now = SparkWeightdMeshMonotonicNs();
    if ( __atomic_load_n(&weightd_mesh.repair_pending,__ATOMIC_ACQUIRE) != 0u &&
         now - weightd_mesh.repair_ns >= SPARK_WEIGHTD_MESH_REPAIR_INTERVAL_NS )
    {
        weightd_mesh.repair_ns = now;
        __atomic_store_n(&weightd_mesh.repair_pending,0u,__ATOMIC_RELEASE);
        weightd_mesh.stat_repairs++;
        SparkWeightdMeshTryWire();
    }
    if ( now - weightd_mesh.stat_logged_ns < SPARK_WEIGHTD_MESH_STATS_NS )
        return;
    weightd_mesh.stat_logged_ns = now;
    fprintf(stderr,"WD-MESH-STATS trywire=%llu record_failures=%llu rewires=%llu unready=%llu repairs=%llu cq_ok=%llu cq_err=%llu ready=%u\n",
        (unsigned long long)weightd_mesh.stat_trywire,(unsigned long long)weightd_mesh.stat_record_failures,
        (unsigned long long)weightd_mesh.stat_rewires,(unsigned long long)weightd_mesh.stat_unready,
        (unsigned long long)weightd_mesh.stat_repairs,(unsigned long long)weightd_mesh.send_ok,
        (unsigned long long)weightd_mesh.send_err,weightd_mesh.mesh_ready);
    SparkWeightdMeshTimingReport();
}

void SparkWeightdMeshPoll(void)
{
    if (weightd_mesh.mesh_active == 0u)
        return;
    if (weightd_mesh.mesh_ready == 0u)
    {
        SparkWeightdMeshTryWire();
    }
    else if (SparkWeightdMeshRealtimeNs() - weightd_mesh.record_check_ns >=
        1000000000ull)
    {
        weightd_mesh.record_check_ns = SparkWeightdMeshRealtimeNs();
        SparkWeightdMeshTryWire();
    }
    if (weightd_mesh.mesh_ready == 0u)
        return;
    if (SparkWeightdMeshRealtimeNs() - weightd_mesh.artifact_check_ns >=
        1000000000ull)
    {
        struct stat artifact_st;
        char artifact_path[256];
        SparkWeightdMeshRecord own_record;
        uint32_t peer;
        weightd_mesh.artifact_check_ns = SparkWeightdMeshRealtimeNs();
        {
            char ready_path[256];
            SparkWeightdMeshReadyPath(ready_path,sizeof(ready_path));
            if (stat(ready_path,&artifact_st) != 0)
            {
                FILE *marker = fopen(ready_path,"w");
                if (marker != 0)
                    (void)fclose(marker);
            }
        }
        (void)snprintf(artifact_path,sizeof(artifact_path),"%s/mesh-%x.rec",
            weightd_mesh_dir,weightd_mesh.local_rank);
        {
            SparkWeightdMeshRecord existing;
            int artifact_fd = open(artifact_path,O_RDONLY);
            uint32_t stale = 1u;
            if (artifact_fd >= 0)
            {
                memset(&existing,0,sizeof(existing));
                if (read(artifact_fd,&existing,sizeof(existing)) ==
                        (ssize_t)sizeof(existing) &&
                    existing.magic == SPARK_WEIGHTD_MESH_MAGIC &&
                    existing.boot_ns == weightd_mesh.boot_ns &&
                    existing.recv_addr ==
                        (uint64_t)(uintptr_t)weightd_mesh.recv_buffer)
                    stale = 0u;
                close(artifact_fd);
            }
            if (stale == 0u)
                goto artifact_current;
        }
        if (1u)
        {
            memset(&own_record,0,sizeof(own_record));
            own_record.magic = SPARK_WEIGHTD_MESH_MAGIC;
            own_record.rank = weightd_mesh.local_rank;
            own_record.rank_mask = weightd_mesh.rank_mask;
            own_record.boot_ns = weightd_mesh.boot_ns;
            own_record.rkey = weightd_mesh.recv_mr != 0 ? weightd_mesh.recv_mr->rkey : 0u;
            own_record.recv_addr = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
            own_record.lid = weightd_mesh.lid;
            memcpy(own_record.gid,weightd_mesh.gid,16);
            for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
            {
                own_record.send_qpn[peer] = weightd_mesh.send_qps[peer] != 0 ? weightd_mesh.send_qps[peer]->qp_num : 0u;
                own_record.recv_qpn[peer] = weightd_mesh.recv_qps[peer] != 0 ? weightd_mesh.recv_qps[peer]->qp_num : 0u;
            }
            (void)SparkWeightdMeshWriteRecord(&own_record);
        }
artifact_current:;
    }
    SparkWeightdMeshDrainCq();
    SparkWeightdMeshRepairAndReport();
    if (weightd_mesh.send_ok - weightd_mesh.send_logged >= 2048ull)
    {
        weightd_mesh.send_logged = weightd_mesh.send_ok;
        fprintf(stderr,"WD-MESH-CQ ok=%llu err=%llu\n",
            (unsigned long long)weightd_mesh.send_ok,
            (unsigned long long)weightd_mesh.send_err);
    }
}

uint64_t SparkWeightdMeshBufferAddress(void)
{
    return (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
}

int SparkWeightdMeshBufferFd(void)
{
    return weightd_mesh.mesh_ready != 0u ? dup(weightd_mesh.memfd) : -1;
}



static uint32_t SparkWeightdMeshDoorbellStable(volatile uint64_t *entry,uint64_t *seq,uint64_t *bytes,uint64_t *slot,uint64_t *destinations)
{
    uint32_t tries;
    for ( tries = 0u; tries < 64u; tries++ )
    {
        *seq = entry[0];
        *bytes = entry[1];
        *slot = entry[2];
        *destinations = entry[3];
        __sync_synchronize();
        if ( *seq == entry[0] && *bytes == entry[1] && *slot == entry[2] && *destinations == entry[3] )
            return(1u);
    }
    return(0u);
}

static void SparkWeightdMeshDoorbellStuck(uint64_t index,const char *phase,uint64_t seq,uint64_t bytes,uint64_t slot)
{
    weightd_mesh.doorbell_stuck[index]++;
    if ( (weightd_mesh.doorbell_stuck[index] % 500u) == 0u )
        fprintf(stderr,"WD-STUCK idx=%llu phase=%s seq=%llu posted=%llu bytes=%llu slot=%llu\n",(unsigned long long)index,phase,(unsigned long long)seq,(unsigned long long)weightd_mesh.doorbell_posted[index],(unsigned long long)bytes,(unsigned long long)slot);
}

static uint32_t SparkWeightdMeshDoorbellCell(uint32_t band,uint32_t rank,uint64_t now_ns)
{
    uint64_t index = (uint64_t)band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank;
    volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band,rank));
    uint64_t seq = entry[0], bytes, slot, destinations;
    SparkWeightdMeshRoute route;
    if ( seq == 0ull || seq == weightd_mesh.doorbell_posted[index] )
    {
        weightd_mesh.doorbell_stuck[index] = 0u;
        return(0u);
    }
    if ( SparkWeightdMeshDoorbellStable(entry,&seq,&bytes,&slot,&destinations) == 0u )
    {
        SparkWeightdMeshDoorbellStuck(index,"unstable",0u,0u,0u);
        return(weightd_mesh.doorbell_stuck[index] < SPARK_WEIGHTD_MESH_STUCK_LIVE_SWEEPS ? 1u : 0u);
    }
    if ( seq == 0ull || bytes == 0ull || seq == weightd_mesh.doorbell_posted[index] )
        return(0u);
    if ( weightd_mesh.doorbell_stuck[index] != 0u && (weightd_mesh.doorbell_stuck[index] % 200u) == 0u )
        fprintf(stderr,"WD-SEEN idx=%llu seq=%llu posted=%llu bytes=%llu slot=%llu\n",(unsigned long long)index,(unsigned long long)seq,(unsigned long long)weightd_mesh.doorbell_posted[index],(unsigned long long)bytes,(unsigned long long)slot);
    route.word = destinations;
    if ( slot >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND || bytes > SPARK_WEIGHTD_MESH_SLOT_BYTES - 16u || SparkWeightdMeshRouteValid(route) == 0u )
    {
        SparkWeightdMeshDoorbellStuck(index,"invalid",seq,bytes,slot);
        return(0u);
    }
    SparkWeightdMeshDoorbellStuck(index,"pending",seq,bytes,slot);
    if ( weightd_mesh.publish_seq[index] != seq )
    {
        weightd_mesh.publish_seq[index] = seq;
        weightd_mesh.publish_ns[index] = now_ns;
    }
    (void)SparkWeightdMeshPostSlot(band,rank,seq,slot,bytes,route,now_ns);
    return(weightd_mesh.doorbell_stuck[index] < SPARK_WEIGHTD_MESH_STUCK_LIVE_SWEEPS ? 1u : 0u);
}

static uint32_t SparkWeightdMeshDoorbellFullScan(uint64_t now_ns)
{
    uint32_t band, rank, work = 0u;
    weightd_mesh.full_scan_ns = now_ns;
    for ( band = 0u; band < SPARK_WEIGHTD_MESH_BANDS; band++ )
        for ( rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++ )
            work += SparkWeightdMeshDoorbellCell(band,rank,now_ns);
    return(work + SparkWeightdMeshWaitRequestsPoll(now_ns));
}

static uint32_t SparkWeightdMeshDoorbellSweep(uint64_t now_ns)
{
    uint32_t lane, band, rank, cell, work = 0u;
    if ( now_ns - weightd_mesh.full_scan_ns >= SPARK_WEIGHTD_MESH_FULL_SCAN_NS )
        return(SparkWeightdMeshDoorbellFullScan(now_ns));
    for ( lane = 0u; lane < SPARK_WEIGHTD_MESH_MAX_LANES; lane++ )
    {
        if ( weightd_mesh.lane_topology[lane].rank_count == 0u )
            continue;
        rank = weightd_mesh.lane_topology[lane].local_rank;
        for ( band = 2u * lane; band < 2u * lane + 2u; band++ )
        {
            cell = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank;
            work += SparkWeightdMeshWaitRequestsPollRange(now_ns,cell,cell + 1u);
            work += SparkWeightdMeshDoorbellCell(band,rank,now_ns);
            work += SparkWeightdMeshWaitRequestsPollRange(now_ns,cell,cell + 1u);
        }
    }
    return(work);
}

static void SparkWeightdMeshDoorbellPoll(void)
{
    uint64_t now_ns;
    uint32_t work;
    SparkWeightdMeshDrainCq();
    SparkWeightdMeshWireAcquire();
    now_ns = SparkWeightdMeshMonotonicNs();
    if ( weightd_mesh.mesh_ready == 0u || SparkWeightdMeshRealtimeNs() < weightd_mesh.quiesce_until_ns )
        work = SparkWeightdMeshWaitRequestsPoll(now_ns);
    else
        work = SparkWeightdMeshDoorbellSweep(now_ns);
    if ( work != 0u || SparkWeightdMeshHasPending() != 0u )
        weightd_mesh.work_ns = now_ns;
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
}

static uint32_t SparkWeightdMeshCpuCapacity(uint32_t cpu)
{
    char path[96];
    unsigned int capacity = 0u;
    FILE *file;
    (void)snprintf(path,sizeof(path),"/sys/devices/system/cpu/cpu%u/cpu_capacity",cpu);
    file = fopen(path,"r");
    if ( file == 0 )
        return(0u);
    if ( fscanf(file,"%u",&capacity) != 1 )
        capacity = 0u;
    (void)fclose(file);
    return(capacity);
}

static void SparkWeightdMeshPinFastCores(void)
{
    uint32_t cpu, cpus, highest = 0u, lowest = UINT32_MAX, capacity, count = 0u;
    long configured = sysconf(_SC_NPROCESSORS_CONF);
    cpu_set_t set;
    cpus = configured > 0 && configured < CPU_SETSIZE ? (uint32_t)configured : 0u;
    for ( cpu = 0u; cpu < cpus; cpu++ )
    {
        capacity = SparkWeightdMeshCpuCapacity(cpu);
        highest = capacity > highest ? capacity : highest;
        lowest = capacity < lowest ? capacity : lowest;
    }
    if ( highest == 0u || highest == lowest )
        return;
    CPU_ZERO(&set);
    for ( cpu = 0u; cpu < cpus; cpu++ )
        if ( SparkWeightdMeshCpuCapacity(cpu) == highest )
        {
            CPU_SET(cpu,&set);
            count++;
        }
    fprintf(stderr,"WD-MESH-PIN doorbell thread cpus=%u capacity=%u min_capacity=%u rc=%d\n",count,highest,lowest,pthread_setaffinity_np(pthread_self(),sizeof(set),&set));
}

void SparkWeightdMeshDoorbellLoop(void)
{
    SparkWeightdMeshPollingThread = 1u;
    SparkWeightdMeshPinFastCores();
    for (;;)
    {
        SparkWeightdMeshWaitForActivity();
        SparkWeightdMeshDoorbellPoll();
        SparkWeightdMeshCpuRelax();
    }
}

uint32_t SparkWeightdMeshBufferLkey(void)
{
    return weightd_mesh.recv_mr != 0 ? weightd_mesh.recv_mr->lkey : 0u;
}

static uint32_t SparkWeightdMeshRankCount(void)
{
    return SPARK_WEIGHTD_MESH_PEERS + 1u;
}

static int32_t SparkWeightdMeshPeerIndexFromRank(uint32_t peer_rank)
{
    if (peer_rank >= SparkWeightdMeshRankCount() ||
        peer_rank == weightd_mesh.local_rank ||
        (weightd_mesh.rank_mask & (1u << peer_rank)) == 0u)
        return -1;
    return (int32_t)(peer_rank > weightd_mesh.local_rank
        ? peer_rank - 1u : peer_rank);
}

uint32_t SparkWeightdMeshBroadcast(
    uint32_t peer_rank_mask,
    uint64_t source_offset,
    uint32_t length,
    uint64_t remote_offset,
    uint64_t seq_value,
    uint64_t seq_remote_offset)
{
    struct ibv_sge scatter[2];
    struct ibv_send_wr work_request;
    struct ibv_send_wr *bad;
    uint32_t rank;
    uint32_t posted = 0u;

    SparkWeightdMeshWireAcquire();
    if (weightd_mesh.mesh_ready == 0u || peer_rank_mask == 0u ||
        (peer_rank_mask & ~weightd_mesh.rank_mask) != 0u ||
        (peer_rank_mask & (1u << weightd_mesh.local_rank)) != 0u)
    {
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        return 0u;
    }
    for ( rank = 0u; rank < SparkWeightdMeshRankCount(); rank++ )
    {
        int32_t peer = SparkWeightdMeshPeerIndexFromRank(rank);
        uint32_t needed = seq_remote_offset != 0ull ? 2u : 1u;
        if ( peer >= 0 && (peer_rank_mask & (1u << rank)) != 0u &&
             weightd_mesh.send_pending[peer] > SPARK_WEIGHTD_MESH_SEND_CAPACITY - needed )
        {
            pthread_mutex_unlock(&SparkWeightdMeshWireLock);
            return 0u;
        }
    }
    if ( seq_remote_offset != 0ull && weightd_mesh.seq_mr != 0 )
    {
        weightd_mesh.seq_storage = seq_value;
        memset(&scatter[1],0,sizeof(scatter[1]));
        scatter[1].addr = (uint64_t)(uintptr_t)&weightd_mesh.seq_storage;
        scatter[1].length = sizeof(weightd_mesh.seq_storage);
        scatter[1].lkey = weightd_mesh.seq_mr->lkey;
    }
    memset(&scatter[0],0,sizeof(scatter[0]));
    scatter[0].addr = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer +
        source_offset;
    scatter[0].length = length;
    scatter[0].lkey = weightd_mesh.recv_mr->lkey;
    for (rank = 0u; rank < SparkWeightdMeshRankCount(); rank++)
    {
        int32_t peer;
        if ((peer_rank_mask & (1u << rank)) == 0u)
            continue;
        peer = SparkWeightdMeshPeerIndexFromRank(rank);
        if (peer < 0)
            continue;
        memset(&work_request,0,sizeof(work_request));
        work_request.wr_id = (uint64_t)(uint32_t)peer;
        work_request.sg_list = &scatter[0];
        work_request.num_sge = 1;
        work_request.opcode = IBV_WR_RDMA_WRITE;
        work_request.send_flags = IBV_SEND_SIGNALED;
        work_request.wr.rdma.remote_addr =
            weightd_mesh.qp_info[peer].remote_addr + remote_offset;
        work_request.wr.rdma.rkey = weightd_mesh.qp_info[peer].rkey;
        if (ibv_post_send(weightd_mesh.send_qps[peer],
                &work_request,&bad) == 0)
        {
            posted++;
            weightd_mesh.send_pending[peer]++;
            SparkWeightdMeshWake();
            weightd_mesh.rpc_pending[peer]++;
        if ( seq_remote_offset != 0ull && weightd_mesh.seq_mr != 0 )
        {
            struct ibv_send_wr seq_request;
            memset(&seq_request,0,sizeof(seq_request));
            seq_request.wr_id = (uint64_t)(uint32_t)peer;
            seq_request.sg_list = &scatter[1];
            seq_request.num_sge = 1;
            seq_request.opcode = IBV_WR_RDMA_WRITE;
            seq_request.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            seq_request.wr.rdma.remote_addr =
                weightd_mesh.qp_info[peer].remote_addr + seq_remote_offset;
            seq_request.wr.rdma.rkey = weightd_mesh.qp_info[peer].rkey;
            if ( ibv_post_send(weightd_mesh.send_qps[peer],&seq_request,&bad) == 0 )
            {
                weightd_mesh.send_pending[peer]++;
                SparkWeightdMeshWake();
                weightd_mesh.rpc_pending[peer]++;
            }
            else
                posted--;
        }
        }
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return posted;
}

SparkStatus SparkWeightdMeshPostWrite(
    uint32_t peer_rank,
    uint64_t local_addr,
    uint32_t lkey,
    uint32_t length,
    uint64_t remote_offset)
{
    struct ibv_sge scatter;
    struct ibv_send_wr work_request;
    struct ibv_send_wr *bad;
    int32_t peer;

    if (weightd_mesh.mesh_ready == 0u)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    peer = SparkWeightdMeshPeerIndexFromRank(peer_rank);
    if (peer < 0)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(&scatter,0,sizeof(scatter));
    scatter.addr = local_addr;
    scatter.length = length;
    scatter.lkey = lkey;
    memset(&work_request,0,sizeof(work_request));
    work_request.wr_id = peer;
    work_request.sg_list = &scatter;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_RDMA_WRITE;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.wr.rdma.remote_addr =
        weightd_mesh.qp_info[peer].remote_addr + remote_offset;
    work_request.wr.rdma.rkey = weightd_mesh.qp_info[peer].rkey;
    SparkWeightdMeshWireAcquire();
    if ( weightd_mesh.mesh_ready == 0u || weightd_mesh.wired_boot_ns[peer] == 0u ||
         weightd_mesh.send_pending[peer] >= SPARK_WEIGHTD_MESH_SEND_CAPACITY )
    {
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        return SPARK_STATUS_BUSY;
    }
    if (ibv_post_send(weightd_mesh.send_qps[peer],&work_request,&bad) != 0)
    {
        fprintf(stderr,"weightd-mesh: post peer=%u errno=%d\n",peer,errno);
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    weightd_mesh.send_pending[peer]++;
    SparkWeightdMeshWake();
    weightd_mesh.rpc_pending[peer]++;
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return SPARK_STATUS_OK;
}
