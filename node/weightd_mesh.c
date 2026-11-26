#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_mesh_hybrid.h"
#include "sparkpipe/spark_latency_histogram.h"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <endian.h>
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
#define SPARK_WEIGHTD_MESH_SQ_SLOTS 2048u
#define SPARK_WEIGHTD_MESH_MAGIC UINT64_C(0x4d45534830303131)
#define SPARK_WEIGHTD_MESH_PORTS 2u
#define SPARK_WEIGHTD_MESH_PORT_PENDING 0u
#define SPARK_WEIGHTD_MESH_PORT_UP 1u
#define SPARK_WEIGHTD_MESH_PORT_UNUSED 2u
#define SPARK_WEIGHTD_MESH_LAYOUT_TWO_PORTS 1u
#define SPARK_WEIGHTD_MESH_LAYOUT_ONE_PORT 2u
#define SPARK_WEIGHTD_MESH_LAYOUT_DEGRADED 3u
#define SPARK_WEIGHTD_MESH_DORMANT_AFTER_NS UINT64_C(20000000)
#define SPARK_WEIGHTD_MESH_DORMANT_POLL_NS 200000L
#define SPARK_WEIGHTD_MESH_FULL_SCAN_NS UINT64_C(1000000)
#define SPARK_WEIGHTD_MESH_ACTIVE_FULL_SCAN_NS UINT64_C(25000000)
#define SPARK_WEIGHTD_MESH_STUCK_LIVE_SWEEPS 100000u
#define SPARK_WEIGHTD_MESH_STUCK_LOG_NS UINT64_C(1000000000)
#define SPARK_WEIGHTD_MESH_REPAIR_INTERVAL_NS UINT64_C(10000000)
#define SPARK_WEIGHTD_MESH_STATS_NS UINT64_C(10000000000)
#define SPARK_WEIGHTD_MESH_TIMING_PEER_TEXT 80u
#define SPARK_WEIGHTD_MESH_DEFAULT_DIR "/tmp/weightd-mesh"
#define SPARK_WEIGHTD_MESH_ADDRESS_POLL_NS 250000000L
#define SPARK_WEIGHTD_MESH_SQ_WAIT_POLLS 64u
#define SPARK_WEIGHTD_MESH_HYBRID_REVOKE_NS UINT64_C(50000000)
#define SPARK_WEIGHTD_MESH_HYBRID_ALLOC_MAX 128u

static const char *weightd_mesh_dir = SPARK_WEIGHTD_MESH_DEFAULT_DIR;
static uint64_t weightd_mesh_address_wait_ns = UINT64_C(120000000000);
static uint64_t weightd_mesh_port_grace_ns = UINT64_C(10000000000);
static uint64_t weightd_mesh_sq_wait_ns = UINT64_C(2000000);

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
    uint32_t pair_rkey;
    uint16_t pair_lid;
    uint8_t pair_gid[16];
    uint64_t wire_generation;
    uint64_t seen_generation[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t pair_send_qpn[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t pair_recv_qpn[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t ports;
    uint32_t switched_pair;
} SparkWeightdMeshRecord;

typedef struct SparkWeightdMeshPort
{
    const char *name;
    const char *logged;
    struct ibv_context *context;
    uint32_t sgid_index;
    uint32_t state;
    uint16_t lid;
    uint8_t gid[16];
} SparkWeightdMeshPort;

#define SPARK_WEIGHTD_MESH_TRANSFER_ID (UINT64_C(1) << 63u)
#define SPARK_WEIGHTD_MESH_RPC_CHAINED (UINT64_C(1) << 62u)
#define SPARK_WEIGHTD_MESH_INLINE_BYTES 28u
#define SPARK_WEIGHTD_MESH_INLINE_MAX_BYTES 912u
#define SPARK_WEIGHTD_MESH_QPN_MAP_BITS 6u
#define SPARK_WEIGHTD_MESH_QPN_MAP_SIZE (1u << SPARK_WEIGHTD_MESH_QPN_MAP_BITS)
#define SPARK_WEIGHTD_MESH_QPN_MAP_EMPTY 0xffu
#define SPARK_WEIGHTD_MESH_QPN_MAP_COLLISION 0xfeu

_Static_assert(SPARK_WEIGHTD_MESH_PEERS * 4u <= 64u,"mesh completion bitmap capacity");
_Static_assert(SPARK_WEIGHTD_MESH_PEERS * SPARK_WEIGHTD_MESH_SEND_CAPACITY <
    SPARK_WEIGHTD_MESH_CQ_ENTRIES,"mesh send completions fit CQ");
_Static_assert(2u * SPARK_WEIGHTD_MESH_BANDS * SPARK_WEIGHTD_MESH_RANKS_PER_BAND <=
    1024u,"mesh completion identity capacity");

#define SPARK_WEIGHTD_MESH_TRANSFER_FAILED 1u
#define SPARK_WEIGHTD_MESH_TRANSFER_PEER_RESET 2u
#define SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT 8u
#define SPARK_WEIGHTD_MESH_CELLS (SPARK_WEIGHTD_MESH_BANDS * SPARK_WEIGHTD_MESH_RANKS_PER_BAND)
#define SPARK_WEIGHTD_MESH_RECORDS (2u * SPARK_WEIGHTD_MESH_CELLS)
#define SPARK_WEIGHTD_MESH_RECORD_OF(index,seq) (2u * (index) + (uint32_t)((seq) & 1u))

typedef struct SparkWeightdMeshTransfer
{
    uint64_t seq;
    uint64_t generation;
    uint64_t pending;
    uint32_t failed;
    uint32_t reset_peer;
    uint32_t chained;
    uint8_t chained_writes[SPARK_WEIGHTD_MESH_PEERS];
} SparkWeightdMeshTransfer;

typedef struct SparkWeightdMeshTiming
{
    SparkLatencyHistogram post;
    SparkLatencyHistogram ship;
    SparkLatencyHistogram credit;
    SparkLatencyHistogram gate;
    SparkLatencyHistogram start;
    SparkLatencyHistogram lag[SPARK_WEIGHTD_MESH_RANKS];
    uint64_t last[SPARK_WEIGHTD_MESH_RANKS];
    uint64_t start_last[SPARK_WEIGHTD_MESH_RANKS];
    uint64_t excess_ns[SPARK_WEIGHTD_MESH_RANKS];
    uint64_t self;
    uint64_t start_self;
    uint64_t worst_ns;
    uint64_t worst_tag;
    uint32_t worst_closer;
    uint64_t route_count;
    uint64_t route_ns;
    uint64_t route_max_ns;
    uint64_t ring_barriers;
} SparkWeightdMeshTiming;

typedef struct SparkWeightdMeshSendQueue
{
    uint8_t *buffer;
    volatile uint32_t *record;
    volatile uint64_t *doorbell;
    uint64_t ring_word;
    uint32_t depth;
    uint32_t mask;
    uint32_t qpn;
    uint32_t produced;
    uint32_t retired;
    uint32_t completed;
    uint32_t completion_queue;
    uint32_t combining_bytes;
    uint32_t bf_offset;
    uint32_t max_inline;
    uint32_t wr_issued;
    uint32_t wr_retired;
    uint32_t sq_shm_off;
    uint32_t dbrec_shm_off;
    uint64_t wr_id[SPARK_WEIGHTD_MESH_SQ_SLOTS];
    uint8_t wqe_bbs[SPARK_WEIGHTD_MESH_SQ_SLOTS];
} SparkWeightdMeshSendQueue;

typedef struct SparkWeightdMeshCompletionQueue
{
    uint8_t *buffer;
    volatile uint32_t *record;
    uint32_t count;
    uint32_t size;
    uint32_t consumed;
} SparkWeightdMeshCompletionQueue;

typedef struct SparkWeightdMeshPlanPeer
{
    uint8_t peer;
    uint8_t target;
    uint16_t pad;
    uint32_t rkey_be;
    uint32_t recv_lkey_be;
    uint32_t staging_lkey_be;
    uint32_t qpn_shifted;
    uint64_t remote_base;
    uint64_t staging_peer_off;
} SparkWeightdMeshPlanPeer;

typedef struct SparkWeightdMeshLanePlan
{
    uint32_t valid;
    uint32_t degree;
    uint32_t local_rank;
    uint32_t all_peers_mask;
    uint32_t peer_count;
    uint32_t min_max_inline;
    SparkWeightdMeshPlanPeer peers[SPARK_WEIGHTD_MESH_PEERS];
} SparkWeightdMeshLanePlan;

typedef struct SparkWeightdMeshShmAllocEntry
{
    uint32_t offset;
    uint32_t bytes;
    uint8_t in_use;
    uint8_t is_dbr;
    uint16_t pad;
} SparkWeightdMeshShmAllocEntry;

typedef struct SparkWeightdMeshShmSlab
{
    uint8_t *base;
    uint64_t bytes;
    uint32_t page_size;
    uint32_t dbr_cursor;
    uint32_t sq_cursor;
    uint32_t entry_count;
    int fd;
    char path[SPARK_WEIGHTD_MESH_HYBRID_SHM_PATH_BYTES];
    SparkWeightdMeshShmAllocEntry entries[SPARK_WEIGHTD_MESH_HYBRID_ALLOC_MAX];
} SparkWeightdMeshShmSlab;

typedef struct SparkWeightdMesh
{
    struct ibv_context *context;
    struct ibv_pd *protection_domain;
    struct ibv_mr *recv_mr;
    struct ibv_mr *staging_mr;
    struct ibv_cq *cq;
    struct ibv_qp *send_qps[SPARK_WEIGHTD_MESH_PEERS];
    struct ibv_qp *recv_qps[SPARK_WEIGHTD_MESH_PEERS];
    SparkWeightdMeshSendQueue send_queues[SPARK_WEIGHTD_MESH_PEERS];
    SparkWeightdMeshCompletionQueue completion_queues[2];
    SparkWeightdMeshQpInfo qp_info[SPARK_WEIGHTD_MESH_PEERS];
    void *recv_buffer;
    void *staging_buffer;
    int memfd;
    int staging_memfd;
    uint64_t boot_ns;
    uint64_t record_check_ns;
    uint64_t artifact_check_ns;
    uint16_t lid;
    uint8_t gid[16];
    uint64_t wired_boot_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t wire_generation;
    uint64_t peer_generation_seen[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t resync_generation[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t stat_resyncs;
    uint64_t invalid_boot_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t failed_boot_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t send_pending[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t send_pending_total;
    uint32_t rpc_pending[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t wire_epoch[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t peer_resets[SPARK_WEIGHTD_MESH_PEERS];
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
    uint64_t doorbell_seen_seq[SPARK_WEIGHTD_MESH_BANDS *
        SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    uint64_t doorbell_seen_ns[SPARK_WEIGHTD_MESH_BANDS *
        SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    uint64_t doorbell_logged_ns[SPARK_WEIGHTD_MESH_BANDS *
        SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    uint64_t ship_log_count;
    uint64_t ship_log_key;
    uint32_t ship_log_key_budget;
    SparkWeightdMeshTransfer transfers[SPARK_WEIGHTD_MESH_RECORDS];
    uint64_t issued[SPARK_WEIGHTD_MESH_CELLS];
    uint64_t wait_seen[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t wait_terminal[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t wait_started[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t publish_seq[SPARK_WEIGHTD_MESH_RECORDS];
    uint64_t publish_ns[SPARK_WEIGHTD_MESH_RECORDS];
    uint64_t post_ns[SPARK_WEIGHTD_MESH_RECORDS];
    uint64_t arrived[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t arrival_ns[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint64_t previous_arrival_ns[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    uint32_t last_peer[SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS];
    SparkWeightdMeshTiming timing;
    uint32_t activity_owners;
    uint32_t repair_pending;
    uint64_t repair_ns;
    uint64_t stat_trywire;
    uint64_t stat_record_failures;
    uint64_t stat_record_invalid;
    uint64_t stat_wire_failures;
    uint64_t stat_rewires;
    uint64_t stat_unready;
    uint64_t stat_repairs;
    uint64_t stat_peer_resets;
    uint64_t stat_peer_reset_bits;
    uint64_t stat_lane_resets;
    uint64_t stat_sq_full_waits;
    uint32_t lane_busy_logged[SPARK_WEIGHTD_MESH_MAX_LANES];
    uint64_t stat_logged_ns;
    uint64_t work_ns;
    uint64_t full_scan_ns;
    uint32_t lane_activity[SPARK_WEIGHTD_MESH_MAX_LANES];
    uint32_t configured_lane_mask;
    SparkWeightdMeshTopology lane_topology[SPARK_WEIGHTD_MESH_MAX_LANES];
    uint32_t mesh_active;
    uint32_t mesh_ready;
    uint32_t local_rank;
    uint32_t rank_mask;
    uint32_t sgid_index;
    uint32_t traffic_class;
    struct ibv_context *pair_context;
    struct ibv_pd *pair_protection_domain;
    struct ibv_mr *pair_recv_mr;
    struct ibv_mr *pair_staging_mr;
    struct ibv_mr *pair_seq_mr;
    struct ibv_cq *pair_cq;
    uint32_t second_rail;
    uint32_t switched_pair;
    uint32_t pair_slot_peers;
    uint32_t pair_sgid_index;
    uint16_t pair_lid;
    uint8_t pair_gid[16];
    uint32_t ports;
    uint32_t home_port;
    const char *port_name[SPARK_WEIGHTD_MESH_PORTS];
    struct ibv_qp *slot_send_qps[2][SPARK_WEIGHTD_MESH_PEERS];
    struct ibv_qp *slot_recv_qps[2][SPARK_WEIGHTD_MESH_PEERS];
    struct ibv_td *slot_send_tds[2][SPARK_WEIGHTD_MESH_PEERS];
    struct ibv_pd *slot_send_pds[2][SPARK_WEIGHTD_MESH_PEERS];
    uint32_t slot_send_inline[2][SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer_link_flags[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer_layout[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer_state[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer_record_status[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer_qp_flags[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer_cq_err_since_ok[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t peer_record_boot_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t peer_since_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t peer_last_ok_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t peer_last_err_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t mesh_generation;
    uint64_t ready_since_ns;
    uint32_t ready_write_logged;
    uint8_t ring_order[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t ring_count;
    uint32_t bf_all_unique;
    uint8_t qpn_map[2][SPARK_WEIGHTD_MESH_QPN_MAP_SIZE];
    SparkWeightdMeshLanePlan lane_plan[SPARK_WEIGHTD_MESH_MAX_LANES];
    SparkWeightdMeshShmSlab hybrid_shm;
    uint32_t hybrid_active;
    uint32_t hybrid_owner_lane;
    uint32_t hybrid_owner_band;
    uint32_t hybrid_grant_gen;
    uint64_t hybrid_shm_gen;
    uint8_t hybrid_lane_capable[SPARK_WEIGHTD_MESH_MAX_LANES];
    uint64_t stat_hybrid_rings;
    uint64_t stat_hybrid_revokes;
} SparkWeightdMesh;

static SparkWeightdMesh weightd_mesh;
static uint64_t SparkWeightdMeshNextTransferGeneration;

static volatile uint64_t *SparkWeightdMeshDoorbellAt(uint32_t index,uint32_t alternate)
{
    return (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        (alternate != 0u ? SPARK_WEIGHTD_MESH_DOORBELL_ALT_OFFSET : SPARK_WEIGHTD_MESH_DOORBELL_OFFSET) +
        (uint64_t)index * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
}

static uint32_t SparkWeightdMeshCellFailed(uint32_t index)
{
    return weightd_mesh.transfers[2u * index].failed != 0u || weightd_mesh.transfers[2u * index + 1u].failed != 0u ? 1u : 0u;
}

static uint32_t SparkWeightdMeshCellPending(uint32_t index)
{
    return weightd_mesh.transfers[2u * index].pending != 0u || weightd_mesh.transfers[2u * index + 1u].pending != 0u ? 1u : 0u;
}

static const SparkWeightdMeshTransfer *SparkWeightdMeshCellFailure(uint32_t index)
{
    const SparkWeightdMeshTransfer *first = &weightd_mesh.transfers[2u * index],*second = first + 1u;
    if ( first->failed != 0u && (second->failed == 0u || first->seq <= second->seq) )
        return first;
    return second->failed != 0u ? second : 0;
}

static uint32_t weightd_mesh_second_rail = 0u;

void SparkWeightdMeshUseSecondRail(void)
{
    weightd_mesh_second_rail = 1u;
}

static uint32_t SparkWeightdMeshIsPairPeer(uint32_t peer)
{
    return weightd_mesh.pair_protection_domain != 0 && peer < SPARK_WEIGHTD_MESH_PEERS && ((weightd_mesh.pair_slot_peers >> peer) & 1u) != 0u ? 1u : 0u;
}

static struct ibv_mr *SparkWeightdMeshPeerMr(uint32_t peer,struct ibv_mr *switch_mr,struct ibv_mr *pair_mr)
{
    return SparkWeightdMeshIsPairPeer(peer) != 0u ? pair_mr : switch_mr;
}

static uint32_t SparkWeightdMeshHomePort(uint32_t ports)
{
    return (ports & 1u) != 0u ? 0u : 1u;
}

static const char *SparkWeightdMeshPortName(uint32_t port)
{
    if ( port < SPARK_WEIGHTD_MESH_PORTS && weightd_mesh.port_name[port] != 0 )
        return weightd_mesh.port_name[port];
    return port == 0u ? "port0" : "port1";
}

static void SparkWeightdMeshPortsText(uint32_t ports,char *text,size_t bytes)
{
    if ( (ports & 3u) == 0u )
        (void)snprintf(text,bytes,"none");
    else
        (void)snprintf(text,bytes,"%s%s%s",(ports & 1u) != 0u ? SparkWeightdMeshPortName(0u) : "",(ports & 3u) == 3u ? "+" : "",
            (ports & 2u) != 0u ? SparkWeightdMeshPortName(1u) : "");
}

static uint32_t SparkWeightdMeshLinkPorts(uint32_t peer_rank,uint32_t remote_switched_pair)
{
    return (weightd_mesh.switched_pair != 0u && remote_switched_pair != 0u) || peer_rank == (weightd_mesh.local_rank ^ 1u) ? 3u : 1u;
}

static uint32_t SparkWeightdMeshPeerPort(uint32_t peer,uint32_t common)
{
    if ( common == 3u )
        return (SparkWeightdMeshSecondRailPeers(weightd_mesh.local_rank,weightd_mesh.rank_mask) >> peer) & 1u;
    return common == 2u ? 1u : 0u;
}

static uint32_t SparkWeightdMeshDegradedMask(void)
{
    uint32_t peer,mask = 0u;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
        if ( weightd_mesh.peer_state[peer] == SPARK_WEIGHTD_MESH_PEER_WIRED &&
             (weightd_mesh.peer_link_flags[peer] & SPARK_WEIGHTD_MESH_PEER_QP_DEGRADED) != 0u )
            mask |= 1u << (peer < weightd_mesh.local_rank ? peer : peer + 1u);
    return mask;
}

static void SparkWeightdMeshFillRecord(SparkWeightdMeshRecord *record)
{
    uint32_t peer;
    memset(record,0,sizeof(*record));
    record->magic = SPARK_WEIGHTD_MESH_MAGIC;
    record->rank = weightd_mesh.local_rank;
    record->rank_mask = weightd_mesh.rank_mask;
    record->boot_ns = weightd_mesh.boot_ns;
    record->rkey = weightd_mesh.recv_mr != 0 ? weightd_mesh.recv_mr->rkey : 0u;
    record->recv_addr = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
    record->lid = weightd_mesh.lid;
    memcpy(record->gid,weightd_mesh.gid,sizeof(record->gid));
    record->pair_rkey = weightd_mesh.pair_recv_mr != 0 ? weightd_mesh.pair_recv_mr->rkey : 0u;
    record->pair_lid = weightd_mesh.pair_lid;
    memcpy(record->pair_gid,weightd_mesh.pair_gid,sizeof(record->pair_gid));
    record->wire_generation = weightd_mesh.wire_generation;
    record->ports = weightd_mesh.ports;
    record->switched_pair = weightd_mesh.switched_pair;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        record->send_qpn[peer] = weightd_mesh.slot_send_qps[0][peer] != 0 ? weightd_mesh.slot_send_qps[0][peer]->qp_num : 0u;
        record->recv_qpn[peer] = weightd_mesh.slot_recv_qps[0][peer] != 0 ? weightd_mesh.slot_recv_qps[0][peer]->qp_num : 0u;
        record->pair_send_qpn[peer] = weightd_mesh.slot_send_qps[1][peer] != 0 ? weightd_mesh.slot_send_qps[1][peer]->qp_num : 0u;
        record->pair_recv_qpn[peer] = weightd_mesh.slot_recv_qps[1][peer] != 0 ? weightd_mesh.slot_recv_qps[1][peer]->qp_num : 0u;
        record->seen_generation[peer] = weightd_mesh.peer_generation_seen[peer];
    }
}


static uint64_t SparkWeightdMeshRealtimeNs(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME,&now) != 0)
        return 0ull;
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void SparkWeightdMeshChanged(void)
{
    (void)__atomic_add_fetch(&weightd_mesh.mesh_generation,1u,__ATOMIC_RELAXED);
}

static void SparkWeightdMeshPeerSet(uint32_t peer,uint32_t state,uint32_t record_status)
{
    weightd_mesh.peer_record_status[peer] = record_status;
    if ( weightd_mesh.peer_state[peer] == state )
        return;
    weightd_mesh.peer_state[peer] = state;
    weightd_mesh.peer_since_ns[peer] = SparkMonotonicNs();
    SparkWeightdMeshChanged();
}

static uint32_t SparkWeightdMeshReadyMarkerRead(char *text,size_t bytes)
{
    char path[256];
    ssize_t got;
    int fd;
    SparkWeightdMeshReadyPath(path,sizeof(path));
    fd = open(path,O_RDONLY | O_CLOEXEC);
    if ( fd < 0 )
        return 0u;
    got = read(fd,text,bytes - 1u);
    (void)close(fd);
    if ( got <= 0 )
        return 0u;
    text[got] = '\0';
    return 1u;
}

static int SparkWeightdMeshReadyText(char *text,size_t bytes)
{
    char ports[160];
    SparkWeightdMeshPortsText(weightd_mesh.ports,ports,sizeof(ports));
    return snprintf(text,bytes,"weightd-ready v1 pid=%ld boot_ns=%llu rank=%u rank_mask=0x%04x ports=%s degraded=0x%04x\n",
        (long)getpid(),(unsigned long long)weightd_mesh.boot_ns,weightd_mesh.local_rank,weightd_mesh.rank_mask,ports,
        SparkWeightdMeshDegradedMask());
}

static uint32_t SparkWeightdMeshReadyMarkerOwned(void)
{
    char text[384],expected[96];
    if ( SparkWeightdMeshReadyMarkerRead(text,sizeof(text)) == 0u )
        return 0u;
    (void)snprintf(expected,sizeof(expected),"weightd-ready v1 pid=%ld boot_ns=%llu ",(long)getpid(),(unsigned long long)weightd_mesh.boot_ns);
    return strncmp(text,expected,strlen(expected)) == 0 ? 1u : 0u;
}

static uint32_t SparkWeightdMeshReadyMarkerCurrent(void)
{
    char text[384],expected[384];
    if ( SparkWeightdMeshReadyMarkerRead(text,sizeof(text)) == 0u )
        return 0u;
    (void)SparkWeightdMeshReadyText(expected,sizeof(expected));
    return strcmp(text,expected) == 0 ? 1u : 0u;
}

static void SparkWeightdMeshReadyMarkerWrite(void)
{
    char path[256],temp[264],text[384];
    int fd,length,error = 0;
    SparkWeightdMeshReadyPath(path,sizeof(path));
    (void)snprintf(temp,sizeof(temp),"%s.tmp",path);
    length = SparkWeightdMeshReadyText(text,sizeof(text));
    length = length < 0 ? 0 : length < (int)sizeof(text) ? length : (int)sizeof(text) - 1;
    fd = open(temp,O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,0644);
    if ( fd < 0 )
        error = errno;
    else
    {
        if ( write(fd,text,(size_t)length) != (ssize_t)length )
            error = errno != 0 ? errno : EIO;
        (void)close(fd);
        if ( error == 0 && rename(temp,path) != 0 )
            error = errno;
        if ( error != 0 )
            (void)unlink(temp);
    }
    if ( error == 0 )
    {
        weightd_mesh.ready_write_logged = 0u;
        return;
    }
    if ( weightd_mesh.ready_write_logged == 0u )
        fprintf(stderr,"WD-MESH-READY-WRITE-FAIL path=%s errno=%d\n",path,error);
    weightd_mesh.ready_write_logged = 1u;
}

static uint32_t SparkWeightdMeshReadyMarkerRemove(uint32_t only_owned)
{
    char path[256];
    if ( only_owned != 0u && SparkWeightdMeshReadyMarkerOwned() == 0u )
        return 0u;
    SparkWeightdMeshReadyPath(path,sizeof(path));
    return unlink(path) == 0 ? 1u : 0u;
}

static void SparkWeightdMeshReadyMarkerSync(void)
{
    if ( weightd_mesh.mesh_ready == 0u )
        (void)SparkWeightdMeshReadyMarkerRemove(0u);
    else if ( SparkWeightdMeshReadyMarkerCurrent() == 0u )
        SparkWeightdMeshReadyMarkerWrite();
}

static uint32_t SparkWeightdMeshGidAddress(const uint8_t *gid)
{
    static const uint8_t mapped[12] = {0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0xffu,0xffu};
    if ( memcmp(gid,mapped,sizeof(mapped)) != 0 )
        return 0u;
    return (uint32_t)gid[12] << 24 | (uint32_t)gid[13] << 16 | (uint32_t)gid[14] << 8 | (uint32_t)gid[15];
}

static void SparkWeightdMeshGidText(const uint8_t *gid,char *text,size_t bytes)
{
    uint32_t address = SparkWeightdMeshGidAddress(gid);
    uint32_t index;
    size_t used;
    if ( address != 0u )
    {
        (void)snprintf(text,bytes,"%u.%u.%u.%u",address >> 24,(address >> 16) & 0xffu,(address >> 8) & 0xffu,address & 0xffu);
        return;
    }
    text[0] = '\0';
    for (index = 0u, used = 0u; index < 16u && used + 3u < bytes; index++, used += 2u)
        (void)snprintf(text + used,bytes - used,"%02x",gid[index]);
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

static inline void SparkWeightdMeshDmaStoreBarrier(void)
{
#if defined(__aarch64__)
    __asm__ volatile ("dmb oshst" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile ("sfence" ::: "memory");
#else
#error "SparkWeightdMeshDmaStoreBarrier requires aarch64 or x86_64"
#endif
}

static inline void SparkWeightdMeshStoreBarrier(void)
{
#if defined(__aarch64__)
    __asm__ volatile ("dsb st" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile ("sfence" ::: "memory");
#else
#error "SparkWeightdMeshStoreBarrier requires aarch64 or x86_64"
#endif
}

static inline void SparkWeightdMeshLoadBarrier(void)
{
#if defined(__aarch64__)
    __asm__ volatile ("dmb ishld" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile ("lfence" ::: "memory");
#else
#error "SparkWeightdMeshLoadBarrier requires aarch64 or x86_64"
#endif
}

static void SparkWeightdMeshLanePlanBuild(uint32_t lane)
{
    const SparkWeightdMeshTopology *topology;
    SparkWeightdMeshLanePlan *plan;
    uint32_t degree,local,step,count = 0u,min_inline = UINT32_MAX;
    if ( lane >= SPARK_WEIGHTD_MESH_MAX_LANES )
        return;
    topology = &weightd_mesh.lane_topology[lane];
    plan = &weightd_mesh.lane_plan[lane];
    plan->valid = 0u;
    if ( topology->rank_count < 2u || weightd_mesh.recv_mr == 0 || weightd_mesh.staging_mr == 0 )
        return;
    degree = topology->rank_count;
    local = topology->local_rank;
    for (step = 1u; step < degree; step++)
    {
        uint32_t target = SparkWeightdMeshSchedulePeer(local,degree,step);
        uint32_t peer_rank = topology->physical_ranks[target];
        uint32_t peer,q_inline;
        SparkWeightdMeshPlanPeer *dst;
        struct ibv_mr *rmr,*smr;
        if ( peer_rank == weightd_mesh.local_rank )
            continue;
        peer = peer_rank < weightd_mesh.local_rank ? peer_rank : peer_rank - 1u;
        if ( peer >= SPARK_WEIGHTD_MESH_PEERS || weightd_mesh.send_queues[peer].buffer == 0 ||
             weightd_mesh.qp_info[peer].remote_addr == 0u )
            return;
        rmr = SparkWeightdMeshPeerMr(peer,weightd_mesh.recv_mr,weightd_mesh.pair_recv_mr);
        smr = SparkWeightdMeshPeerMr(peer,weightd_mesh.staging_mr,weightd_mesh.pair_staging_mr);
        if ( rmr == 0 || smr == 0 )
            return;
        dst = &plan->peers[count++];
        dst->peer = (uint8_t)peer;
        dst->target = (uint8_t)target;
        dst->pad = 0u;
        dst->rkey_be = htobe32(weightd_mesh.qp_info[peer].rkey);
        dst->recv_lkey_be = htobe32(rmr->lkey);
        dst->staging_lkey_be = htobe32(smr->lkey);
        dst->qpn_shifted = weightd_mesh.send_queues[peer].qpn << 8u;
        dst->remote_base = weightd_mesh.qp_info[peer].remote_addr;
        dst->staging_peer_off = (uint64_t)target * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES;
        q_inline = weightd_mesh.send_queues[peer].max_inline != 0u ?
            weightd_mesh.send_queues[peer].max_inline : SPARK_WEIGHTD_MESH_INLINE_BYTES;
        if ( q_inline < min_inline )
            min_inline = q_inline;
    }
    plan->degree = degree;
    plan->local_rank = local;
    plan->all_peers_mask = ((1u << degree) - 1u) & ~(1u << local);
    plan->peer_count = count;
    plan->min_max_inline = min_inline == UINT32_MAX ? 0u : min_inline;
    plan->valid = count != 0u ? 1u : 0u;
}

static void SparkWeightdMeshLanePlanRefreshAll(void)
{
    uint32_t lane;
    for (lane = 0u; lane < SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
        SparkWeightdMeshLanePlanBuild(lane);
}

static void SparkWeightdMeshRebuildFastMaps(void)
{
    uint32_t peer,other,unique = 1u,has_bf = 0u;
    memset(weightd_mesh.qpn_map,SPARK_WEIGHTD_MESH_QPN_MAP_EMPTY,sizeof(weightd_mesh.qpn_map));
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        const SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[peer];
        uint32_t which,slot;
        if ( weightd_mesh.send_qps[peer] == 0 || queue->buffer == 0 )
            continue;
        which = queue->completion_queue < 2u ? queue->completion_queue : 0u;
        slot = queue->qpn & (SPARK_WEIGHTD_MESH_QPN_MAP_SIZE - 1u);
        weightd_mesh.qpn_map[which][slot] = weightd_mesh.qpn_map[which][slot] == SPARK_WEIGHTD_MESH_QPN_MAP_EMPTY ?
            (uint8_t)peer : SPARK_WEIGHTD_MESH_QPN_MAP_COLLISION;
        if ( queue->combining_bytes == 0u )
            continue;
        has_bf = 1u;
        for (other=0u; other<peer && unique != 0u; other++)
            if ( weightd_mesh.send_qps[other] != 0 && weightd_mesh.send_queues[other].combining_bytes != 0u &&
                 weightd_mesh.send_queues[other].doorbell == queue->doorbell )
                unique = 0u;
    }
    weightd_mesh.bf_all_unique = has_bf != 0u && unique != 0u ? 1u : 0u;
    SparkWeightdMeshLanePlanRefreshAll();
}

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

static uint32_t SparkWeightdMeshDrainCqLocked(uint64_t now_ns);
static void SparkWeightdMeshHybridRevokeLocked(void);
static void SparkWeightdMeshHybridRefreshLocked(uint32_t preferred_band);
static void SparkWeightdMeshHybridUpdateCreditsLocked(void);
static uint32_t SparkWeightdMeshHybridPollLocked(uint64_t now_ns);

static SparkWeightdMeshHybridCtrl *SparkWeightdMeshHybridCtrlPtr(void)
{
    if ( weightd_mesh.recv_buffer == 0 )
        return 0;
    return (SparkWeightdMeshHybridCtrl *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + SPARK_WEIGHTD_MESH_HYBRID_OFFSET);
}

static uint32_t SparkWeightdMeshShmOffset(const void *ptr,size_t bytes)
{
    uintptr_t base,addr,end;
    if ( weightd_mesh.hybrid_shm.base == 0 || ptr == 0 || bytes == 0u )
        return SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE;
    base = (uintptr_t)weightd_mesh.hybrid_shm.base;
    addr = (uintptr_t)ptr;
    end = base + (uintptr_t)weightd_mesh.hybrid_shm.bytes;
    if ( addr < base || addr > end || bytes > (size_t)(end - addr) )
        return SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE;
    return (uint32_t)(addr - base);
}

static void SparkWeightdMeshShmSlabRelease(uint32_t unmap_region)
{
    if ( weightd_mesh.hybrid_shm.path[0] != '\0' )
    {
        (void)unlink(weightd_mesh.hybrid_shm.path);
        weightd_mesh.hybrid_shm.path[0] = '\0';
    }
    if ( unmap_region != 0u && weightd_mesh.hybrid_shm.base != 0 )
    {
        (void)munmap(weightd_mesh.hybrid_shm.base,(size_t)weightd_mesh.hybrid_shm.bytes);
        weightd_mesh.hybrid_shm.base = 0;
    }
    if ( weightd_mesh.hybrid_shm.fd >= 0 )
    {
        (void)close(weightd_mesh.hybrid_shm.fd);
        weightd_mesh.hybrid_shm.fd = -1;
    }
}

static void SparkWeightdMeshShmSlabInit(uint32_t rank)
{
    const char *env = getenv("SPARK_WEIGHTD_MESH_HYBRID");
    long ps;
    int fd;
    void *mapped;
    weightd_mesh.hybrid_shm.fd = -1;
    if ( env != 0 && strcmp(env,"0") == 0 )
        return;
    ps = sysconf(_SC_PAGESIZE);
    weightd_mesh.hybrid_shm.page_size = ps > 4096L ? (uint32_t)ps : 4096u;
    (void)snprintf(weightd_mesh.hybrid_shm.path,sizeof(weightd_mesh.hybrid_shm.path),
        "/dev/shm/spark-mesh-sq-%x-%ld-%llu",rank,(long)getpid(),(unsigned long long)weightd_mesh.boot_ns);
    fd = open(weightd_mesh.hybrid_shm.path,O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,0600);
    if ( fd < 0 && (errno == ENOENT || errno == EROFS || errno == EACCES) )
    {
        (void)snprintf(weightd_mesh.hybrid_shm.path,sizeof(weightd_mesh.hybrid_shm.path),
            "/tmp/spark-mesh-sq-%x-%ld-%llu",rank,(long)getpid(),(unsigned long long)weightd_mesh.boot_ns);
        fd = open(weightd_mesh.hybrid_shm.path,O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,0600);
    }
    if ( fd < 0 )
    {
        weightd_mesh.hybrid_shm.path[0] = '\0';
        return;
    }
    if ( ftruncate(fd,(off_t)SPARK_WEIGHTD_MESH_HYBRID_SHM_BYTES) != 0 )
    {
        (void)close(fd);
        (void)unlink(weightd_mesh.hybrid_shm.path);
        weightd_mesh.hybrid_shm.path[0] = '\0';
        return;
    }
    mapped = mmap(0,SPARK_WEIGHTD_MESH_HYBRID_SHM_BYTES,PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
    if ( mapped == MAP_FAILED )
    {
        (void)close(fd);
        (void)unlink(weightd_mesh.hybrid_shm.path);
        weightd_mesh.hybrid_shm.path[0] = '\0';
        return;
    }
    memset(mapped,0,SPARK_WEIGHTD_MESH_HYBRID_SHM_BYTES);
    weightd_mesh.hybrid_shm.base = (uint8_t *)mapped;
    weightd_mesh.hybrid_shm.bytes = SPARK_WEIGHTD_MESH_HYBRID_SHM_BYTES;
    weightd_mesh.hybrid_shm.dbr_cursor = 0u;
    weightd_mesh.hybrid_shm.sq_cursor = SPARK_WEIGHTD_MESH_HYBRID_DBR_ARENA_BYTES;
    weightd_mesh.hybrid_shm.entry_count = 0u;
    weightd_mesh.hybrid_shm.fd = fd;
    weightd_mesh.hybrid_shm_gen = weightd_mesh.boot_ns != 0u ? weightd_mesh.boot_ns : 1u;
}

#if defined(IBV_ALLOCATOR_USE_DEFAULT) || defined(SPARK_WEIGHTD_MESH_DIR)
static void *SparkWeightdMeshShmSlabAlloc(SparkWeightdMeshShmSlab *slab,size_t size,size_t alignment)
{
    uint32_t i,is_dbr,align,span,off,limit;
    if ( slab == 0 || slab->base == 0 || size == 0u || size > 1048576u ||
         (alignment != 0u && (alignment & (alignment - 1u)) != 0u) )
        return 0;
    is_dbr = size <= 64u ? 1u : 0u;
    align = is_dbr != 0u ? (alignment > 64u ? (uint32_t)alignment : 64u) :
        (alignment > slab->page_size ? (uint32_t)alignment : slab->page_size);
    span = is_dbr != 0u ? (uint32_t)((size + 63u) & ~(size_t)63u) :
        (uint32_t)((size + slab->page_size - 1u) & ~(size_t)(slab->page_size - 1u));
    for (i = 0u; i < slab->entry_count; i++)
    {
        SparkWeightdMeshShmAllocEntry *e = &slab->entries[i];
        if ( e->in_use == 0u && e->is_dbr == is_dbr && e->bytes >= span && (e->offset & (align - 1u)) == 0u )
        {
            e->in_use = 1u;
            memset(slab->base + e->offset,0,size);
            return slab->base + e->offset;
        }
    }
    if ( slab->entry_count >= SPARK_WEIGHTD_MESH_HYBRID_ALLOC_MAX )
        return 0;
    off = is_dbr != 0u ? ((slab->dbr_cursor + align - 1u) & ~(align - 1u)) :
        ((slab->sq_cursor + align - 1u) & ~(align - 1u));
    limit = is_dbr != 0u ? SPARK_WEIGHTD_MESH_HYBRID_DBR_ARENA_BYTES : (uint32_t)slab->bytes;
    if ( off > limit || span > limit - off )
        return 0;
    slab->entries[slab->entry_count].offset = off;
    slab->entries[slab->entry_count].bytes = span;
    slab->entries[slab->entry_count].in_use = 1u;
    slab->entries[slab->entry_count].is_dbr = (uint8_t)is_dbr;
    slab->entries[slab->entry_count].pad = 0u;
    slab->entry_count++;
    if ( is_dbr != 0u )
        slab->dbr_cursor = off + span;
    else
        slab->sq_cursor = off + span;
    memset(slab->base + off,0,size);
    return slab->base + off;
}

static void SparkWeightdMeshShmSlabFree(SparkWeightdMeshShmSlab *slab,void *ptr)
{
    uint32_t i,off;
    if ( slab == 0 || slab->base == 0 || ptr == 0 )
        return;
    if ( (uintptr_t)ptr < (uintptr_t)slab->base || (uintptr_t)ptr >= (uintptr_t)(slab->base + slab->bytes) )
        return;
    off = (uint32_t)((uintptr_t)ptr - (uintptr_t)slab->base);
    for (i = 0u; i < slab->entry_count; i++)
    {
        if ( slab->entries[i].offset == off && slab->entries[i].in_use != 0u )
        {
            slab->entries[i].in_use = 0u;
            if ( i + 1u == slab->entry_count )
            {
                if ( slab->entries[i].is_dbr != 0u && off + slab->entries[i].bytes == slab->dbr_cursor )
                {
                    slab->dbr_cursor = off;
                    slab->entry_count--;
                }
                else if ( slab->entries[i].is_dbr == 0u && off + slab->entries[i].bytes == slab->sq_cursor )
                {
                    slab->sq_cursor = off;
                    slab->entry_count--;
                }
            }
            return;
        }
    }
}
#endif

#ifdef IBV_ALLOCATOR_USE_DEFAULT
static void *SparkWeightdMeshShmAllocCb(struct ibv_pd *pd,void *pd_context,size_t size,size_t alignment,uint64_t resource_type)
{
    void *ptr;
    (void)pd;
    (void)resource_type;
    ptr = SparkWeightdMeshShmSlabAlloc((SparkWeightdMeshShmSlab *)pd_context,size,alignment);
    return ptr != 0 ? ptr : IBV_ALLOCATOR_USE_DEFAULT;
}

static void SparkWeightdMeshShmFreeCb(struct ibv_pd *pd,void *pd_context,void *ptr,uint64_t resource_type)
{
    (void)pd;
    (void)resource_type;
    SparkWeightdMeshShmSlabFree((SparkWeightdMeshShmSlab *)pd_context,ptr);
}
#endif

static void SparkWeightdMeshHybridHeaderInit(void)
{
    SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
    if ( ctrl == 0 )
        return;
    memset(ctrl,0,sizeof(*ctrl));
    if ( weightd_mesh.hybrid_shm.base == 0 )
        return;
    ctrl->magic = SPARK_WEIGHTD_MESH_HYBRID_MAGIC;
    ctrl->version = SPARK_WEIGHTD_MESH_HYBRID_VERSION;
    ctrl->shm_generation = weightd_mesh.hybrid_shm_gen;
    ctrl->shm_bytes = weightd_mesh.hybrid_shm.bytes;
    ctrl->recv_iova_base = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
    ctrl->staging_iova_base = (uint64_t)(uintptr_t)weightd_mesh.staging_buffer;
    ctrl->owner_lane = UINT32_MAX;
    ctrl->owner_band = UINT32_MAX;
    memcpy(ctrl->shm_path,weightd_mesh.hybrid_shm.path,sizeof(ctrl->shm_path));
}

static void SparkWeightdMeshSendQueueReset(uint32_t peer)
{
    SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[peer];
    queue->produced = 0u;
    queue->retired = 0u;
    queue->completed = 0u;
    queue->wr_issued = 0u;
    queue->wr_retired = 0u;
    queue->ring_word = 0u;
    queue->bf_offset = 0u;
    memset(queue->wr_id,0xff,sizeof(queue->wr_id));
    memset(queue->wqe_bbs,1,sizeof(queue->wqe_bbs));
}

static uint32_t SparkWeightdMeshInlineLimit(uint32_t requested,uint32_t depth)
{
    uint32_t limit = requested > SPARK_WEIGHTD_MESH_INLINE_MAX_BYTES ? SPARK_WEIGHTD_MESH_INLINE_MAX_BYTES : requested;
    uint32_t max_bbs = depth / 4u,max_by_depth;
    if ( max_bbs > 15u ) max_bbs = 15u;
    max_by_depth = max_bbs != 0u ? max_bbs * 64u - 36u : SPARK_WEIGHTD_MESH_INLINE_BYTES;
    if ( limit > max_by_depth ) limit = max_by_depth;
    return limit >= SPARK_WEIGHTD_MESH_INLINE_BYTES ? limit : SPARK_WEIGHTD_MESH_INLINE_BYTES;
}

static SparkStatus SparkWeightdMeshSendQueueInit(uint32_t peer,struct ibv_qp *qp,uint32_t completion_queue,uint32_t max_inline)
{
    SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[peer];
    struct mlx5dv_qp view;
    struct mlx5dv_obj object;
    memset(&view,0,sizeof(view));
    memset(&object,0,sizeof(object));
    object.qp.in = qp;
    object.qp.out = &view;
    if ( mlx5dv_init_obj(&object,MLX5DV_OBJ_QP) != 0 || view.sq.stride != 64u || view.sq.wqe_cnt < SPARK_WEIGHTD_MESH_SEND_CAPACITY ||
         view.sq.wqe_cnt > SPARK_WEIGHTD_MESH_SQ_SLOTS || (view.sq.wqe_cnt & (view.sq.wqe_cnt - 1u)) != 0u || view.bf.reg == 0 || view.dbrec == 0 )
    {
        fprintf(stderr,"WD-MESH-DIRECT-SQ-FAIL peer=%u errno=%d stride=%u wqe_cnt=%u bf=%p\n",peer,errno,view.sq.stride,view.sq.wqe_cnt,view.bf.reg);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    queue->buffer = (uint8_t *)view.sq.buf;
    queue->record = (volatile uint32_t *)view.dbrec;
    queue->doorbell = (volatile uint64_t *)view.bf.reg;
    queue->depth = view.sq.wqe_cnt;
    queue->mask = view.sq.wqe_cnt - 1u;
    queue->qpn = qp->qp_num;
    queue->completion_queue = completion_queue;
    queue->combining_bytes = view.bf.size;
    queue->max_inline = SparkWeightdMeshInlineLimit(max_inline,view.sq.wqe_cnt);
    queue->sq_shm_off = SparkWeightdMeshShmOffset(view.sq.buf,(size_t)view.sq.wqe_cnt * 64u);
    queue->dbrec_shm_off = SparkWeightdMeshShmOffset((const void *)view.dbrec,8u);
    SparkWeightdMeshSendQueueReset(peer);
    SparkWeightdMeshRebuildFastMaps();
    fprintf(stderr,"WD-MESH-DIRECT-SQ peer=%u qpn=%u wqe_cnt=%u doorbell=%p bf_size=%u max_inline=%u bf_unique=%u sq_shm=%u dbrec_shm=%u\n",
        peer,queue->qpn,view.sq.wqe_cnt,view.bf.reg,view.bf.size,queue->max_inline,weightd_mesh.bf_all_unique,
        queue->sq_shm_off != SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE ? 1u : 0u,
        queue->dbrec_shm_off != SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE ? 1u : 0u);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdMeshCompletionQueueInit(uint32_t which,struct ibv_cq *cq)
{
    SparkWeightdMeshCompletionQueue *queue = &weightd_mesh.completion_queues[which];
    struct mlx5dv_cq view;
    struct mlx5dv_obj object;
    memset(&view,0,sizeof(view));
    memset(&object,0,sizeof(object));
    object.cq.in = cq;
    object.cq.out = &view;
    if ( mlx5dv_init_obj(&object,MLX5DV_OBJ_CQ) != 0 || view.cqe_cnt == 0u || (view.cqe_cnt & (view.cqe_cnt - 1u)) != 0u ||
         (view.cqe_size != 64u && view.cqe_size != 128u) || view.dbrec == 0 )
    {
        fprintf(stderr,"WD-MESH-DIRECT-CQ-FAIL which=%u errno=%d cqe_cnt=%u cqe_size=%u\n",which,errno,view.cqe_cnt,view.cqe_size);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    queue->buffer = (uint8_t *)view.buf;
    queue->record = (volatile uint32_t *)view.dbrec;
    queue->count = view.cqe_cnt;
    queue->size = view.cqe_size;
    queue->consumed = 0u;
    return SPARK_STATUS_OK;
}

static uint32_t SparkWeightdMeshSqFree(const SparkWeightdMeshSendQueue *queue)
{
    uint32_t inflight = queue->produced - queue->retired;
    return inflight < queue->depth ? queue->depth - inflight : 0u;
}

static SparkStatus SparkWeightdMeshSqEnsureRoom(uint32_t peer,uint32_t needed)
{
    SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[peer];
    uint64_t deadline;
    uint32_t repair_needed = 0u,polls = 0u;
    if ( queue->buffer == 0 || needed == 0u || needed > queue->depth )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    if ( SparkWeightdMeshSqFree(queue) >= needed )
        return SPARK_STATUS_OK;
    weightd_mesh.stat_sq_full_waits++;
    deadline = SparkMonotonicNs() + weightd_mesh_sq_wait_ns;
    for (;;)
    {
        uint64_t now = SparkMonotonicNs();
        repair_needed |= SparkWeightdMeshDrainCqLocked(now);
        if ( SparkWeightdMeshSqFree(queue) >= needed )
        {
            if ( repair_needed != 0u )
                __atomic_store_n(&weightd_mesh.repair_pending,1u,__ATOMIC_RELEASE);
            return SPARK_STATUS_OK;
        }
        if ( ++polls >= SPARK_WEIGHTD_MESH_SQ_WAIT_POLLS && now >= deadline )
            break;
        SparkWeightdMeshCpuRelax();
    }
    if ( repair_needed != 0u )
        __atomic_store_n(&weightd_mesh.repair_pending,1u,__ATOMIC_RELEASE);
    SPARK_FAIL(SPARK_STATUS_BUSY);
}

static SparkStatus SparkWeightdMeshSqWrite(uint32_t peer,uint64_t wr_id,uint32_t signaled,uint64_t local_address,uint32_t lkey,
    uint32_t length,uint64_t remote_offset,const uint64_t *inline_word)
{
    SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[peer];
    uint32_t index,inline_len,num_bb,max_inline = queue->max_inline != 0u ? queue->max_inline : SPARK_WEIGHTD_MESH_INLINE_BYTES;
    struct mlx5_wqe_ctrl_seg *control;
    struct mlx5_wqe_raddr_seg *remote;
    const void *inline_src;
    uint8_t ds;
    inline_src = inline_word != 0 ? (const void *)inline_word :
        (length != 0u && length <= max_inline && local_address != 0u ? (const void *)(uintptr_t)local_address : 0);
    inline_len = inline_word != 0 ? (uint32_t)sizeof(*inline_word) : length;
    ds = inline_src != 0 ? (uint8_t)((36u + inline_len + 15u) >> 4u) : 3u;
    num_bb = (uint32_t)((ds + 3u) >> 2u);
    if ( queue->buffer == 0 || SparkWeightdMeshSqFree(queue) < num_bb )
    {
        SparkStatus room = SparkWeightdMeshSqEnsureRoom(peer,num_bb);
        if ( room != SPARK_STATUS_OK )
            return room;
    }
    index = queue->produced & queue->mask;
    control = (struct mlx5_wqe_ctrl_seg *)(queue->buffer + (uint64_t)index * 64u);
    remote = (struct mlx5_wqe_raddr_seg *)(control + 1);
    control->dci_stream_channel_id = 0u;
    mlx5dv_set_ctrl_seg(control,(uint16_t)queue->produced,MLX5_OPCODE_RDMA_WRITE,0u,queue->qpn,
        signaled != 0u ? MLX5_WQE_CTRL_CQ_UPDATE : 0u,ds,0u,0u);
    remote->raddr = htobe64(weightd_mesh.qp_info[peer].remote_addr + remote_offset);
    remote->rkey = htobe32(weightd_mesh.qp_info[peer].rkey);
    remote->reserved = 0u;
    if ( inline_src != 0 )
    {
        struct mlx5_wqe_inl_data_seg *segment = (struct mlx5_wqe_inl_data_seg *)(remote + 1);
        uint32_t sq_bytes = queue->depth * 64u,dst_off = index * 64u + 36u,tail_room = sq_bytes - dst_off;
        segment->byte_count = htobe32(inline_len | MLX5_INLINE_SEG);
        if ( inline_len <= tail_room )
            memcpy(queue->buffer + dst_off,inline_src,inline_len);
        else
        {
            memcpy(queue->buffer + dst_off,inline_src,tail_room);
            memcpy(queue->buffer,(const uint8_t *)inline_src + tail_room,inline_len - tail_room);
        }
    }
    else
        mlx5dv_set_data_seg((struct mlx5_wqe_data_seg *)(remote + 1),length,lkey,(uintptr_t)local_address);
    queue->wr_id[index] = wr_id;
    queue->wqe_bbs[index] = (uint8_t)num_bb;
    memcpy(&queue->ring_word,control,sizeof(queue->ring_word));
    queue->produced += num_bb;
    return SPARK_STATUS_OK;
}

static void SparkWeightdMeshSqRingOrder(const uint8_t *order,uint32_t count)
{
    volatile uint64_t *written[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t i,peer,used = 0u,k;
    weightd_mesh.ring_count = count;
    if ( count == 0u )
        return;
    SparkWeightdMeshDmaStoreBarrier();
    for (i=0u; i<count; i++)
    {
        peer = order[i];
        weightd_mesh.ring_order[i] = (uint8_t)peer;
        weightd_mesh.send_queues[peer].record[MLX5_SND_DBR] = htobe32(weightd_mesh.send_queues[peer].produced & 0xffffu);
    }
    SparkWeightdMeshStoreBarrier();
    if ( weightd_mesh.bf_all_unique != 0u )
    {
        for (i=0u; i<count; i++)
        {
            SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[order[i]];
            volatile uint64_t *doorbell = (volatile uint64_t *)((volatile uint8_t *)queue->doorbell + queue->bf_offset);
            *doorbell = queue->ring_word;
            queue->bf_offset ^= queue->combining_bytes;
        }
        SparkWeightdMeshStoreBarrier();
        return;
    }
    for (i=0u; i<count; i++)
    {
        SparkWeightdMeshSendQueue *queue = &weightd_mesh.send_queues[order[i]];
        volatile uint64_t *doorbell = queue->doorbell;
        if ( queue->combining_bytes != 0u )
        {
            for (k=0u; k<used && written[k] != doorbell; k++) ;
            if ( k != used )
            {
                doorbell = (volatile uint64_t *)((volatile uint8_t *)queue->doorbell + queue->combining_bytes);
                for (k=0u; k<used && written[k] != doorbell; k++) ;
            }
            if ( k != used )
            {
                SparkWeightdMeshStoreBarrier();
                weightd_mesh.timing.ring_barriers++;
                used = 0u;
                doorbell = queue->doorbell;
            }
            written[used++] = doorbell;
        }
        *doorbell = queue->ring_word;
    }
    SparkWeightdMeshStoreBarrier();
}

static void SparkWeightdMeshSqRing(uint32_t peers)
{
    uint8_t order[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer,count = 0u;
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
        if ( (peers & (1u << peer)) != 0u ) order[count++] = (uint8_t)peer;
    SparkWeightdMeshSqRingOrder(order,count);
}

static int SparkWeightdMeshCqeStatus(uint8_t syndrome)
{
    switch ( syndrome )
    {
        case MLX5_CQE_SYNDROME_WR_FLUSH_ERR: return IBV_WC_WR_FLUSH_ERR;
        case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR: return IBV_WC_LOC_LEN_ERR;
        case MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR: return IBV_WC_LOC_QP_OP_ERR;
        case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR: return IBV_WC_LOC_PROT_ERR;
        case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR: return IBV_WC_REM_ACCESS_ERR;
        case MLX5_CQE_SYNDROME_REMOTE_OP_ERR: return IBV_WC_REM_OP_ERR;
        case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR: return IBV_WC_RETRY_EXC_ERR;
        case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR: return IBV_WC_RNR_RETRY_EXC_ERR;
        default: return IBV_WC_GENERAL_ERR;
    }
}

static uint32_t SparkWeightdMeshPeerByQpn(uint32_t which,uint32_t qpn)
{
    uint32_t peer = weightd_mesh.qpn_map[which][qpn & (SPARK_WEIGHTD_MESH_QPN_MAP_SIZE - 1u)];
    if ( peer < SPARK_WEIGHTD_MESH_PEERS && weightd_mesh.send_qps[peer] != 0 &&
         weightd_mesh.send_queues[peer].qpn == qpn && weightd_mesh.send_queues[peer].completion_queue == which )
        return peer;
    if ( peer == SPARK_WEIGHTD_MESH_QPN_MAP_EMPTY )
        return SPARK_WEIGHTD_MESH_PEERS;
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
        if ( weightd_mesh.send_qps[peer] != 0 && weightd_mesh.send_queues[peer].qpn == qpn && weightd_mesh.send_queues[peer].completion_queue == which )
            return peer;
    return SPARK_WEIGHTD_MESH_PEERS;
}

static SparkStatus SparkWeightdMeshTransitionQp(
    struct ibv_qp *qp,
    uint32_t remote_qpn,
    uint16_t dlid,
    const uint8_t *dgid,
    uint32_t sgid_index,
    const char **stage)
{
    struct ibv_qp_attr attributes;
    int flags;

    memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RESET;
    if (ibv_modify_qp(qp,&attributes,IBV_QP_STATE) != 0)
    {
        *stage = "RESET";
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
        *stage = "INIT";
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
    memcpy(attributes.ah_attr.grh.dgid.raw,dgid,sizeof(attributes.ah_attr.grh.dgid.raw));
    attributes.ah_attr.grh.sgid_index = (uint8_t)sgid_index;
    attributes.ah_attr.grh.hop_limit = 1;
    attributes.ah_attr.grh.traffic_class = (uint8_t)weightd_mesh.traffic_class;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp,&attributes,flags) != 0)
    {
        *stage = "RTR";
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
        *stage = "RTS";
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static pthread_mutex_t SparkWeightdMeshWireLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t SparkWeightdMeshActivityCondition = PTHREAD_COND_INITIALIZER;
static uint32_t SparkWeightdMeshWireWaiters;
static uint32_t SparkWeightdMeshDoorbellRunning;
static __thread uint32_t SparkWeightdMeshPollingThread;

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
    weightd_mesh.work_ns = SparkMonotonicNs();
    if ( SparkWeightdMeshPollingThread == 0u )
        pthread_cond_broadcast(&SparkWeightdMeshActivityCondition);
}

static void SparkWeightdMeshAdvertiseLane(uint32_t lane)
{
    SparkWeightdMeshWaitRequest *requests;
    uint32_t index;
    uint64_t caps;
    if ( weightd_mesh.recv_buffer == 0 || lane >= SPARK_WEIGHTD_MESH_MAX_LANES )
        return;
    caps = (uint64_t)SPARK_WEIGHTD_MESH_CAPABILITIES |
        (weightd_mesh.pair_protection_domain != 0 && weightd_mesh.second_rail == 0u ? (uint64_t)SPARK_WEIGHTD_MESH_CAPABILITY_PAIR_LINK : UINT64_C(0)) |
        (weightd_mesh.hybrid_lane_capable[lane] != 0u ? (uint64_t)SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ : UINT64_C(0));
    requests = (SparkWeightdMeshWaitRequest *)((uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET);
    for (index=2u * lane * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index<2u * (lane + 1u) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
        __atomic_store_n(&requests[index].capabilities,caps,__ATOMIC_RELEASE);
}

static uint32_t SparkWeightdMeshDoorbellShippable(uint32_t index);

static void SparkWeightdMeshLaneBusyLocked(uint32_t lane,uint32_t reason,uint32_t index)
{
    static const char *const names[] = {"","activity","rpc","buffer","pending","wait","doorbell"};
    if ( weightd_mesh.lane_busy_logged[lane] == reason )
        return;
    weightd_mesh.lane_busy_logged[lane] = reason;
    fprintf(stderr,"WD-LANE-BUSY rank=%u lane=%u reason=%s index=%u\n",weightd_mesh.local_rank,lane,names[reason],index);
}

static void SparkWeightdMeshLaneResetLocked(uint32_t lane)
{
    uint32_t index,failed = 0u,stale = 0u;
    for (index=2u * lane * SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
         index<2u * (lane + 1u) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
    {
        uint32_t band = index / SPARK_WEIGHTD_MESH_RANKS_PER_BAND,rank = index % SPARK_WEIGHTD_MESH_RANKS_PER_BAND,alternate;
        SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)(
            (uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET +
            index * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
        if ( SparkWeightdMeshCellFailed(index) != 0u )
            failed++;
        for (alternate=0u; alternate<2u; alternate++)
        {
            volatile uint64_t *entry = SparkWeightdMeshDoorbellAt(index,alternate);
            if ( __atomic_load_n(&entry[0],__ATOMIC_ACQUIRE) > weightd_mesh.issued[index] )
                stale++;
            __atomic_store_n(&entry[0],0u,__ATOMIC_RELEASE);
            entry[1] = 0u;
            entry[2] = 0u;
            entry[3] = 0u;
        }
        __atomic_store_n((uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank)),0u,__ATOMIC_RELEASE);
        memset(request,0,SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
        memset(&weightd_mesh.transfers[2u * index],0,2u * sizeof(weightd_mesh.transfers[0]));
        weightd_mesh.issued[index] = 0u;
        weightd_mesh.doorbell_posted[index] = 0u;
        weightd_mesh.doorbell_stuck[index] = 0u;
        weightd_mesh.doorbell_seen_seq[index] = 0u;
        weightd_mesh.doorbell_seen_ns[index] = 0u;
        weightd_mesh.doorbell_logged_ns[index] = 0u;
        weightd_mesh.wait_seen[index] = 0u;
        weightd_mesh.wait_terminal[index] = 0u;
        weightd_mesh.wait_started[index] = 0u;
        memset(&weightd_mesh.publish_seq[2u * index],0,2u * sizeof(weightd_mesh.publish_seq[0]));
        memset(&weightd_mesh.publish_ns[2u * index],0,2u * sizeof(weightd_mesh.publish_ns[0]));
        memset(&weightd_mesh.post_ns[2u * index],0,2u * sizeof(weightd_mesh.post_ns[0]));
        weightd_mesh.arrived[index] = 0u;
        weightd_mesh.arrival_ns[index] = 0u;
        weightd_mesh.previous_arrival_ns[index] = 0u;
        weightd_mesh.last_peer[index] = 0u;
    }
    if ( weightd_mesh.hybrid_owner_lane == lane )
    {
        SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
        if ( ctrl != 0 )
        {
            memset(ctrl->trigger,0,sizeof(ctrl->trigger));
            ctrl->owner_band = UINT32_MAX;
        }
    }
    weightd_mesh.stat_lane_resets++;
    fprintf(stderr,"WD-LANE-RESET rank=%u lane=%u bands=%u,%u failed_cells=%u stale_doorbells=%u\n",
        weightd_mesh.local_rank,lane,2u * lane,2u * lane + 1u,failed,stale);
}

static uint32_t SparkWeightdMeshLaneCheckLocked(uint32_t lane,uint32_t *where)
{
    uint32_t rank,index,reason = 0u;
    *where = 0u;
    if ( weightd_mesh.lane_activity[lane] != 0u )
        reason = SPARK_WEIGHTD_MESH_LANE_BUSY_ACTIVITY;
    for (rank=0u; reason == 0u && rank<SPARK_WEIGHTD_MESH_PEERS; rank++)
        if ( weightd_mesh.rpc_pending[rank] != 0u )
        { reason = SPARK_WEIGHTD_MESH_LANE_BUSY_RPC; *where = rank; }
    if ( reason == 0u && weightd_mesh.recv_buffer == 0 )
        reason = SPARK_WEIGHTD_MESH_LANE_BUSY_BUFFER;
    for (index=2u * lane * SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
         reason == 0u && index<2u * (lane + 1u) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
    {
        const SparkWeightdMeshWaitRequest *request = (const SparkWeightdMeshWaitRequest *)(
            (const uint8_t *)weightd_mesh.recv_buffer + SPARK_WEIGHTD_MESH_WAIT_OFFSET +
            index * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
        *where = index;
        if ( SparkWeightdMeshCellPending(index) != 0u )
            reason = SPARK_WEIGHTD_MESH_LANE_BUSY_PENDING;
        else if ( __atomic_load_n(&request->request_id,__ATOMIC_ACQUIRE) > weightd_mesh.wait_terminal[index] )
            reason = SPARK_WEIGHTD_MESH_LANE_BUSY_WAIT;
        else if ( SparkWeightdMeshDoorbellShippable(index) != 0u )
            reason = SPARK_WEIGHTD_MESH_LANE_BUSY_DOORBELL;
    }
    return reason;
}

SparkStatus SparkWeightdMeshLaneConfigure(uint32_t lane,
    const SparkWeightdMeshTopology *topology)
{
    SparkStatus status = SPARK_STATUS_OK;
    uint32_t rank,mask = 0u,reason = 0u,where = 0u,changed;
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
    changed = weightd_mesh.lane_topology[lane].rank_count != 0u &&
        (weightd_mesh.lane_topology[lane].rank_count != topology->rank_count ||
         weightd_mesh.lane_topology[lane].local_rank != topology->local_rank ||
         memcmp(weightd_mesh.lane_topology[lane].physical_ranks,topology->physical_ranks,
             topology->rank_count * sizeof(uint32_t)) != 0) ? 1u : 0u;
    reason = SparkWeightdMeshLaneCheckLocked(lane,&where);
    if ( reason != 0u )
    {
        SparkWeightdMeshLaneBusyLocked(lane,reason,where);
        status = changed != 0u ? SPARK_STATUS_UNSUPPORTED : SPARK_STATUS_BUSY;
        goto done;
    }
    if ( changed != 0u )
        fprintf(stderr,"WD-LANE-TOPOLOGY rank=%u lane=%u ranks=%u->%u local=%u->%u\n",weightd_mesh.local_rank,lane,
            weightd_mesh.lane_topology[lane].rank_count,topology->rank_count,
            weightd_mesh.lane_topology[lane].local_rank,topology->local_rank);
    weightd_mesh.lane_busy_logged[lane] = 0u;
    SparkWeightdMeshHybridRevokeLocked();
    SparkWeightdMeshLaneResetLocked(lane);
    weightd_mesh.lane_topology[lane] = *topology;
    weightd_mesh.configured_lane_mask |= 1u << lane;
    SparkWeightdMeshLanePlanBuild(lane);
    SparkWeightdMeshHybridRefreshLocked(2u * lane);
    SparkWeightdMeshAdvertiseLane(lane);
    SparkWeightdMeshChanged();
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
        SparkWeightdMeshHybridRefreshLocked(active != 0u ? 2u * lane : weightd_mesh.hybrid_owner_band);
        SparkWeightdMeshWake();
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return status;
}

static uint32_t SparkWeightdMeshHasPending(void)
{
    return weightd_mesh.send_pending_total != 0u ? 1u : 0u;
}

static uint32_t SparkWeightdMeshHasDoorbellWork(void)
{
    uint32_t index,alternate;
    if ( weightd_mesh.hybrid_active != 0u )
    {
        SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
        if ( ctrl != 0 && weightd_mesh.hybrid_owner_band < SPARK_WEIGHTD_MESH_BANDS &&
             ctrl->owner_rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND )
        {
            uint32_t hidx = weightd_mesh.hybrid_owner_band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + ctrl->owner_rank;
            uint64_t issued = weightd_mesh.issued[hidx];
            if ( __atomic_load_n(&ctrl->trigger[0].seq,__ATOMIC_ACQUIRE) > issued ||
                 __atomic_load_n(&ctrl->trigger[1].seq,__ATOMIC_ACQUIRE) > issued )
                return 1u;
        }
    }
    for (index=0u; index<SPARK_WEIGHTD_MESH_CELLS; index++)
        for (alternate=0u; alternate<2u; alternate++)
        {
            const volatile uint64_t *entry = SparkWeightdMeshDoorbellAt(index,alternate);
            if ( entry[0] > weightd_mesh.issued[index] && entry[1] != 0u )
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

static void SparkWeightdMeshGateClosed(uint32_t index,uint32_t band,const SparkWeightdMeshWaitRequest *request,uint64_t now_ns,uint32_t first)
{
    const SparkWeightdMeshTopology *topology = &weightd_mesh.lane_topology[band / 2u];
    SparkWeightdMeshTiming *timing = &weightd_mesh.timing;
    uint64_t started = weightd_mesh.wait_started[index],waited = now_ns >= started ? now_ns - started : 0u;
    uint32_t start = (uint32_t)request->tag == 1u ? 1u : 0u,closer = first != 0u ? topology->physical_ranks[topology->local_rank] : weightd_mesh.last_peer[index];
    SparkLatencyAdd(&timing->gate,started,now_ns);
    if ( start != 0u )
        SparkLatencyAdd(&timing->start,started,now_ns);
    if ( first != 0u )
    {
        timing->self++;
        timing->start_self += start;
    }
    else
    {
        timing->last[closer]++;
        timing->start_last[closer] += start;
        if ( weightd_mesh.previous_arrival_ns[index] != 0u )
            timing->excess_ns[closer] += weightd_mesh.arrival_ns[index] - weightd_mesh.previous_arrival_ns[index];
    }
    if ( waited > timing->worst_ns )
    {
        timing->worst_ns = waited;
        timing->worst_tag = request->tag;
        timing->worst_closer = closer;
    }
}

static uint32_t SparkWeightdMeshPeersArrived(uint32_t index,uint32_t band,const SparkWeightdMeshWaitRequest *request,uint64_t now_ns,uint32_t first,uint64_t *diag)
{
    const SparkWeightdMeshTopology *topology = &weightd_mesh.lane_topology[band / 2u];
    uint64_t ring = (request->tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u),slot,tail,start;
    uint32_t peer,physical,complete = 1u;
    start = weightd_mesh.publish_seq[SPARK_WEIGHTD_MESH_RECORD_OF(index,request->tag)] == request->tag ?
        weightd_mesh.publish_ns[SPARK_WEIGHTD_MESH_RECORD_OF(index,request->tag)] : weightd_mesh.wait_started[index];
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
            weightd_mesh.previous_arrival_ns[index] = weightd_mesh.arrival_ns[index];
            weightd_mesh.arrival_ns[index] = now_ns;
            SparkLatencyAdd(&weightd_mesh.timing.lag[physical],start,now_ns);
        }
    }
    if ( complete != 0u )
        SparkWeightdMeshGateClosed(index,band,request,now_ns,first);
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
            weightd_mesh.arrival_ns[index] = 0u;
            weightd_mesh.previous_arrival_ns[index] = 0u;
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
            if ( complete == 0u && SparkWeightdMeshCellFailure(index) != 0 )
            {
                const SparkWeightdMeshTransfer *failure = SparkWeightdMeshCellFailure(index);
                error = failure->failed == SPARK_WEIGHTD_MESH_TRANSFER_PEER_RESET ?
                    (SPARK_WEIGHTD_MESH_WAIT_ERROR_PEER_RESET | failure->reset_peer) : UINT64_MAX;
            }
            if ( complete != 0u )
                SparkLatencyAdd(&weightd_mesh.timing.credit,weightd_mesh.wait_started[index],now_ns);
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
    if ( SparkMonotonicNs() - weightd_mesh.work_ns < SPARK_WEIGHTD_MESH_DORMANT_AFTER_NS ||
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
    uint32_t repair[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t port[SPARK_WEIGHTD_MESH_PEERS];
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

static uint32_t SparkWeightdMeshRecordSlot(const SparkWeightdMeshRecord *record,uint32_t port)
{
    return port != SparkWeightdMeshHomePort(record->ports) ? 1u : 0u;
}

static uint32_t SparkWeightdMeshRecordUsable(uint32_t peer,const SparkWeightdMeshRecord *record,uint32_t *port)
{
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
    uint32_t mine = weightd_mesh.local_rank < peer_rank ? weightd_mesh.local_rank : weightd_mesh.local_rank - 1u;
    uint32_t link = SparkWeightdMeshLinkPorts(peer_rank,record->switched_pair);
    uint32_t local = weightd_mesh.ports & link,remote = record->ports & link,remote_slot;
    const uint8_t *gid = record->gid;
    const char *reason = 0;
    char text[40],local_text[160],remote_text[160];
    *port = 0u;
    if ( record->ports == 0u || (record->ports & ~3u) != 0u )
        reason = "peer record names no usable port";
    else if ( (local & remote) == 0u )
        reason = "no port in common";
    else
    {
        *port = SparkWeightdMeshPeerPort(peer,local & remote);
        remote_slot = SparkWeightdMeshRecordSlot(record,*port);
        gid = remote_slot != 0u ? record->pair_gid : record->gid;
        if ( remote_slot != 0u && record->pair_rkey == 0u )
            reason = "peer record has no key on the second link";
        else if ( (remote_slot != 0u ? record->pair_send_qpn[mine] : record->send_qpn[mine]) == 0u ||
                  (remote_slot != 0u ? record->pair_recv_qpn[mine] : record->recv_qpn[mine]) == 0u )
            reason = "peer record has no queue pair for this link";
        else if ( SparkWeightdMeshGidAddress(gid) == 0u )
            reason = "gid is not an IPv4 RoCE address";
    }
    if ( reason == 0 )
    {
        weightd_mesh.invalid_boot_ns[peer] = 0ull;
        return 1u;
    }
    weightd_mesh.stat_record_invalid++;
    if ( weightd_mesh.invalid_boot_ns[peer] != record->boot_ns )
    {
        weightd_mesh.invalid_boot_ns[peer] = record->boot_ns;
        SparkWeightdMeshGidText(gid,text,sizeof(text));
        SparkWeightdMeshPortsText(local,local_text,sizeof(local_text));
        SparkWeightdMeshPortsText(remote,remote_text,sizeof(remote_text));
        fprintf(stderr,"WD-MESH-RECORD-INVALID rank=%u peer=%u link=%s gid=%s boot=%llu ports=%s/%s: %s; waiting for a new record\n",
            weightd_mesh.local_rank,peer,*port != 0u ? "pair" : "switch",text,(unsigned long long)record->boot_ns,
            local_text,remote_text,reason);
    }
    return 0u;
}

static SparkStatus SparkWeightdMeshPeerSlotLocked(uint32_t peer,uint32_t slot,const char **stage)
{
    struct ibv_qp *send = weightd_mesh.slot_send_qps[slot][peer],*recv = weightd_mesh.slot_recv_qps[slot][peer];
    struct ibv_qp_attr attributes;
    if ( send == 0 || recv == 0 )
    {
        *stage = "SLOT";
        errno = ENODEV;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ( weightd_mesh.send_qps[peer] == send && weightd_mesh.recv_qps[peer] == recv )
        return SPARK_STATUS_OK;
    memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RESET;
    if ( weightd_mesh.send_qps[peer] != 0 )
        (void)ibv_modify_qp(weightd_mesh.send_qps[peer],&attributes,IBV_QP_STATE);
    if ( weightd_mesh.recv_qps[peer] != 0 )
        (void)ibv_modify_qp(weightd_mesh.recv_qps[peer],&attributes,IBV_QP_STATE);
    weightd_mesh.send_qps[peer] = send;
    weightd_mesh.recv_qps[peer] = recv;
    if ( slot != 0u )
        weightd_mesh.pair_slot_peers |= 1u << peer;
    else
        weightd_mesh.pair_slot_peers &= ~(1u << peer);
    *stage = "SQ";
    return SparkWeightdMeshSendQueueInit(peer,send,slot,weightd_mesh.slot_send_inline[slot][peer]);
}

static void SparkWeightdMeshLayoutLocked(uint32_t peer,uint32_t port,uint32_t local,uint32_t remote)
{
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
    uint32_t layout = local != remote ? SPARK_WEIGHTD_MESH_LAYOUT_DEGRADED : local == 3u ? SPARK_WEIGHTD_MESH_LAYOUT_TWO_PORTS : SPARK_WEIGHTD_MESH_LAYOUT_ONE_PORT;
    uint32_t packed = layout | port << 2u | local << 4u | remote << 6u,missing;
    char local_text[160],remote_text[160];
    weightd_mesh.peer_link_flags[peer] = (port != 0u ? SPARK_WEIGHTD_MESH_PEER_QP_PAIR_LINK : 0u) |
        (layout == SPARK_WEIGHTD_MESH_LAYOUT_DEGRADED ? SPARK_WEIGHTD_MESH_PEER_QP_DEGRADED : 0u);
    if ( weightd_mesh.peer_layout[peer] == packed )
        return;
    weightd_mesh.peer_layout[peer] = packed;
    SparkWeightdMeshPortsText(local,local_text,sizeof(local_text));
    SparkWeightdMeshPortsText(remote,remote_text,sizeof(remote_text));
    if ( layout != SPARK_WEIGHTD_MESH_LAYOUT_DEGRADED )
    {
        fprintf(stderr,"WD-MESH-LAYOUT rank=%u peer=%u peer_rank=%u port=%s layout=%s ports=%s/%s\n",weightd_mesh.local_rank,peer,peer_rank,
            SparkWeightdMeshPortName(port),layout == SPARK_WEIGHTD_MESH_LAYOUT_TWO_PORTS ? "two-ports" : "one-port",local_text,remote_text);
        return;
    }
    missing = (local & ~remote) != 0u ? local & ~remote : remote & ~local;
    fprintf(stderr,"WD-MESH-LAYOUT rank=%u peer=%u peer_rank=%u port=%s layout=degraded ports=%s/%s: %s %u has no %s port up, so the link shares %s\n",
        weightd_mesh.local_rank,peer,peer_rank,SparkWeightdMeshPortName(port),local_text,remote_text,
        (local & ~remote) != 0u ? "rank" : "this node, rank",(local & ~remote) != 0u ? peer_rank : weightd_mesh.local_rank,
        SparkWeightdMeshPortName((missing & 1u) != 0u ? 0u : 1u),SparkWeightdMeshPortName(port));
}

static void SparkWeightdMeshWireScanPeer(SparkWeightdMeshWirePlan *plan, uint32_t peer)
{
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u, send_in_rts, recv_in_rts;
    uint32_t mine = weightd_mesh.local_rank < peer_rank ? weightd_mesh.local_rank : weightd_mesh.local_rank - 1u;
    SparkStatus status;
    plan->force_wire[peer] = 0u;
    plan->repair[peer] = 0u;
    if ( (weightd_mesh.rank_mask & (1u << peer_rank)) == 0u )
        return;
    status = SparkWeightdMeshReadPeerRecord(peer_rank,&plan->records[peer]);
    if ( status != SPARK_STATUS_OK )
    {
        plan->wired = 0u;
        plan->changed = 1u;
        weightd_mesh.stat_record_failures++;
        SparkWeightdMeshPeerSet(peer,status == SPARK_STATUS_BUSY ? SPARK_WEIGHTD_MESH_PEER_NO_RECORD : SPARK_WEIGHTD_MESH_PEER_RECORD_REJECTED,(uint32_t)status);
        return;
    }
    weightd_mesh.peer_record_boot_ns[peer] = plan->records[peer].boot_ns;
    if ( SparkWeightdMeshRecordUsable(peer,&plan->records[peer],&plan->port[peer]) == 0u )
    {
        plan->wired = 0u;
        plan->changed = 1u;
        SparkWeightdMeshPeerSet(peer,SPARK_WEIGHTD_MESH_PEER_RECORD_INVALID,SPARK_STATUS_OK);
        return;
    }
    if ( plan->records[peer].boot_ns != weightd_mesh.wired_boot_ns[peer] )
    {
        plan->changed = 1u;
        plan->force_wire[peer] = 1u;
        return;
    }
    if ( weightd_mesh.resync_generation[peer] != 0u && plan->records[peer].seen_generation[mine] >= weightd_mesh.resync_generation[peer] )
    {
        fprintf(stderr,"WD-QP-RESYNC-DONE rank=%u peer=%u generation=%llu — the peer re-transitioned its queue pairs; both sides restart at PSN 0\n",
            weightd_mesh.local_rank,peer,(unsigned long long)weightd_mesh.resync_generation[peer]);
        weightd_mesh.resync_generation[peer] = 0u;
        plan->changed = 1u;
    }
    send_in_rts = SparkWeightdMeshQpInRts(weightd_mesh.send_qps[peer]);
    recv_in_rts = SparkWeightdMeshQpInRts(weightd_mesh.recv_qps[peer]);
    weightd_mesh.peer_qp_flags[peer] = send_in_rts * SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | recv_in_rts * SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS |
        weightd_mesh.peer_link_flags[peer];
    plan->force_wire[peer] = SparkWeightdMeshRewireNeeded(plan->records[peer].boot_ns,weightd_mesh.wired_boot_ns[peer],send_in_rts,recv_in_rts);
    plan->repair[peer] = plan->force_wire[peer];
    if ( plan->records[peer].wire_generation != weightd_mesh.peer_generation_seen[peer] )
    {
        plan->force_wire[peer] = 1u;
        fprintf(stderr,"WD-QP-RESYNC rank=%u peer=%u generation=%llu — the peer repaired its queue pairs; re-transitioning ours so both restart at PSN 0\n",
            weightd_mesh.local_rank,peer,(unsigned long long)plan->records[peer].wire_generation);
    }
    if ( plan->force_wire[peer] == 0u )
    {
        if ( weightd_mesh.resync_generation[peer] != 0u )
        {
            plan->wired = 0u;
            plan->changed = 1u;
            return;
        }
        SparkWeightdMeshPeerSet(peer,SPARK_WEIGHTD_MESH_PEER_WIRED,SPARK_STATUS_OK);
        return;
    }
    plan->changed = 1u;
    if ( plan->repair[peer] != 0u )
        fprintf(stderr,"WD-QP-REPAIR rank=%u peer=%u — qp left RTS after errors; re-transitioning on the same record and asking the peer to resynchronize\n",weightd_mesh.local_rank,peer);
}

static uint32_t SparkWeightdMeshWirePeerLocked(const SparkWeightdMeshWirePlan *plan, uint32_t peer)
{
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
    uint32_t mine = weightd_mesh.local_rank < peer_rank ? weightd_mesh.local_rank : weightd_mesh.local_rank - 1u;
    const SparkWeightdMeshRecord *record = &plan->records[peer];
    uint32_t port = plan->port[peer];
    uint32_t link = SparkWeightdMeshLinkPorts(peer_rank,record->switched_pair);
    uint32_t pair = port != weightd_mesh.home_port ? 1u : 0u;
    uint32_t remote_pair = SparkWeightdMeshRecordSlot(record,port);
    uint32_t sgid = pair != 0u ? weightd_mesh.pair_sgid_index : weightd_mesh.sgid_index;
    uint16_t lid = remote_pair != 0u ? record->pair_lid : record->lid;
    const uint8_t *gid = remote_pair != 0u ? record->pair_gid : record->gid;
    uint32_t remote_send = remote_pair != 0u ? record->pair_send_qpn[mine] : record->send_qpn[mine];
    uint32_t remote_recv = remote_pair != 0u ? record->pair_recv_qpn[mine] : record->recv_qpn[mine];
    uint32_t rkey = remote_pair != 0u ? record->pair_rkey : record->rkey;
    const char *qp_name = "send";
    const char *stage = "";
    char text[40];
    SparkStatus status;
    int error;
    (void)SparkWeightdMeshDrainCqLocked(SparkMonotonicNs());
    status = SparkWeightdMeshPeerSlotLocked(peer,pair,&stage);
    if ( status == SPARK_STATUS_OK )
        status = SparkWeightdMeshTransitionQp(weightd_mesh.send_qps[peer],remote_recv,lid,gid,sgid,&stage);
    (void)SparkWeightdMeshDrainCqLocked(SparkMonotonicNs());
    SparkWeightdMeshSendQueueReset(peer);
    if ( status == SPARK_STATUS_OK )
    {
        qp_name = "recv";
        status = SparkWeightdMeshTransitionQp(weightd_mesh.recv_qps[peer],remote_send,lid,gid,sgid,&stage);
    }
    if ( status != SPARK_STATUS_OK )
    {
        error = errno;
        weightd_mesh.stat_wire_failures++;
        if ( weightd_mesh.failed_boot_ns[peer] != record->boot_ns )
        {
            weightd_mesh.failed_boot_ns[peer] = record->boot_ns;
            SparkWeightdMeshGidText(gid,text,sizeof(text));
            fprintf(stderr,"WD-WIRE-FAIL rank=%u peer=%u link=%s qp=%s transition=%s errno=%d target_qpn=%u gid=%s boot=%llu; repeats are counted in WD-MESH-STATS\n",
                weightd_mesh.local_rank,peer,port != 0u ? "pair" : "switch",qp_name,stage,error,
                qp_name[0] == 's' ? remote_recv : remote_send,text,(unsigned long long)record->boot_ns);
        }
        weightd_mesh.peer_qp_flags[peer] &= ~(SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS);
        SparkWeightdMeshPeerSet(peer,SPARK_WEIGHTD_MESH_PEER_WIRE_FAILED,SPARK_STATUS_DRIVER_LOAD_ERROR);
        return 0u;
    }
    if ( weightd_mesh.peer_state[peer] == SPARK_WEIGHTD_MESH_PEER_WIRED && weightd_mesh.wired_boot_ns[peer] != record->boot_ns )
        SparkWeightdMeshChanged();
    SparkWeightdMeshLayoutLocked(peer,port,weightd_mesh.ports & link,record->ports & link);
    weightd_mesh.peer_qp_flags[peer] = SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS | SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS | weightd_mesh.peer_link_flags[peer];
    SparkWeightdMeshPeerSet(peer,SPARK_WEIGHTD_MESH_PEER_WIRED,SPARK_STATUS_OK);
    weightd_mesh.failed_boot_ns[peer] = 0ull;
    weightd_mesh.qp_info[peer].remote_qpn = remote_recv;
    weightd_mesh.qp_info[peer].rkey = rkey;
    weightd_mesh.qp_info[peer].remote_addr = record->recv_addr;
    weightd_mesh.wired_boot_ns[peer] = record->boot_ns;
    SparkWeightdMeshLanePlanRefreshAll();
    weightd_mesh.stat_rewires++;
    fprintf(stderr,"WD-WIRED rank=%u peer=%u addr=%llx rkey=%u boot=%llu\n",weightd_mesh.local_rank,peer,(unsigned long long)record->recv_addr,rkey,(unsigned long long)record->boot_ns);
    return 1u;
}

static void SparkWeightdMeshWireReadyLocked(uint32_t wired)
{
    if ( wired == 0u )
    {
        if ( weightd_mesh.mesh_ready != 0u )
        {
            weightd_mesh.stat_unready++;
            weightd_mesh.ready_since_ns = 0u;
            SparkWeightdMeshChanged();
        }
        weightd_mesh.mesh_ready = 0u;
        return;
    }
    if ( weightd_mesh.mesh_ready == 0u )
    {
        weightd_mesh.ready_since_ns = SparkMonotonicNs();
        SparkWeightdMeshChanged();
    }
    SparkWeightdMeshPhase("wired");
    weightd_mesh.mesh_ready = 1u;
    SparkWeightdMeshWake();
    weightd_mesh.send_err_window = 0ull;
    weightd_mesh.degraded_logged = 0u;
    weightd_mesh.quiesce_until_ns = 0ull;
    printf("weightd-mesh: ready rank=%u peers=%u rkey=%u\n",weightd_mesh.local_rank,(uint32_t)__builtin_popcount(weightd_mesh.rank_mask) - 1u,weightd_mesh.recv_mr->rkey);
    fflush(stdout);
    {
        char ports[160];
        uint32_t peer,two_ports = 0u,one_port = 0u;
        for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
        {
            uint32_t bit = 1u << (peer < weightd_mesh.local_rank ? peer : peer + 1u);
            if ( weightd_mesh.peer_state[peer] != SPARK_WEIGHTD_MESH_PEER_WIRED )
                continue;
            if ( (weightd_mesh.peer_layout[peer] & 3u) == SPARK_WEIGHTD_MESH_LAYOUT_TWO_PORTS )
                two_ports |= bit;
            else if ( (weightd_mesh.peer_layout[peer] & 3u) == SPARK_WEIGHTD_MESH_LAYOUT_ONE_PORT )
                one_port |= bit;
        }
        SparkWeightdMeshPortsText(weightd_mesh.ports,ports,sizeof(ports));
        fprintf(stderr,"WD-MESH-LAYOUT-READY rank=%u ports=%s two_ports=0x%04x one_port=0x%04x degraded=0x%04x\n",
            weightd_mesh.local_rank,ports,two_ports,one_port,SparkWeightdMeshDegradedMask());
    }
}


static void SparkWeightdMeshPeerResetLocked(uint32_t peer,const char *reason,uint64_t record_boot_ns)
{
    uint64_t mask = UINT64_C(0xF) << (4u * peer),lost;
    uint32_t index,cells = 0u,bits = 0u,physical = peer < weightd_mesh.local_rank ? peer : peer + 1u;
    uint32_t send = weightd_mesh.send_pending[peer],rpc = weightd_mesh.rpc_pending[peer];
    for (index=0u; index<SPARK_WEIGHTD_MESH_RECORDS; index++)
    {
        lost = weightd_mesh.transfers[index].pending & mask;
        if ( lost == 0u )
            continue;
        weightd_mesh.transfers[index].pending &= ~mask;
        weightd_mesh.transfers[index].chained &= ~(1u << peer);
        if ( weightd_mesh.transfers[index].failed == 0u )
        {
            weightd_mesh.transfers[index].failed = SPARK_WEIGHTD_MESH_TRANSFER_PEER_RESET;
            weightd_mesh.transfers[index].reset_peer = physical;
            SparkWeightdMeshChanged();
        }
        cells++;
        bits += (uint32_t)__builtin_popcountll(lost);
    }
    weightd_mesh.send_pending_total -= send <= weightd_mesh.send_pending_total ? send : weightd_mesh.send_pending_total;
    weightd_mesh.send_pending[peer] = 0u;
    weightd_mesh.send_queues[peer].wr_retired = weightd_mesh.send_queues[peer].wr_issued;
    weightd_mesh.rpc_pending[peer] = 0u;
    weightd_mesh.wire_epoch[peer]++;
    if ( cells == 0u && send == 0u && rpc == 0u )
        return;
    weightd_mesh.peer_resets[peer]++;
    weightd_mesh.stat_peer_resets++;
    weightd_mesh.stat_peer_reset_bits += bits;
    fprintf(stderr,"WD-PEER-RESET rank=%u peer=%u physical=%u reason=%s wired_boot=%llu record_boot=%llu cells=%u bits=%u send_pending=%u rpc_pending=%u\n",
        weightd_mesh.local_rank,peer,physical,reason,(unsigned long long)weightd_mesh.wired_boot_ns[peer],
        (unsigned long long)record_boot_ns,cells,bits,send,rpc);
}

static void SparkWeightdMeshTryWire(void)
{
    SparkWeightdMeshWirePlan plan;
    uint32_t peer,repair,republish,bumped = 0u;
    plan.changed = weightd_mesh.mesh_ready == 0u;
    plan.wired = 1u;
    weightd_mesh.stat_trywire++;
    for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++ )
        SparkWeightdMeshWireScanPeer(&plan,peer);
    if ( plan.changed == 0u )
        return;
    SparkWeightdMeshWireAcquire();
    SparkWeightdMeshHybridRevokeLocked();
    repair = SparkWeightdMeshDrainCqLocked(SparkMonotonicNs());
    republish = 0u;
    for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++ )
        if ( plan.repair[peer] != 0u && plan.records[peer].boot_ns == weightd_mesh.wired_boot_ns[peer] )
            bumped = 1u;
    if ( bumped != 0u )
        weightd_mesh.wire_generation++;
    for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++ )
    {
        uint32_t booted;
        if ( plan.force_wire[peer] == 0u )
            continue;
        booted = plan.records[peer].boot_ns != weightd_mesh.wired_boot_ns[peer] ? 1u : 0u;
        SparkWeightdMeshPeerResetLocked(peer,booted != 0u ? "record" : plan.repair[peer] != 0u ? "qp-repair" : "peer-resync",plan.records[peer].boot_ns);
        if ( SparkWeightdMeshWirePeerLocked(&plan,peer) == 0u )
        {
            plan.wired = 0u;
            continue;
        }
        if ( weightd_mesh.peer_generation_seen[peer] != plan.records[peer].wire_generation )
        {
            weightd_mesh.peer_generation_seen[peer] = plan.records[peer].wire_generation;
            republish = 1u;
        }
        if ( booted != 0u )
            weightd_mesh.resync_generation[peer] = 0u;
        else if ( plan.repair[peer] != 0u )
        {
            weightd_mesh.resync_generation[peer] = weightd_mesh.wire_generation;
            weightd_mesh.stat_resyncs++;
            republish = 1u;
        }
    }
    for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++ )
        if ( weightd_mesh.resync_generation[peer] != 0u )
            plan.wired = 0u;
    if ( republish != 0u || bumped != 0u )
    {
        SparkWeightdMeshRecord own_record;
        SparkWeightdMeshFillRecord(&own_record);
        if ( SparkWeightdMeshWriteRecord(&own_record) != SPARK_STATUS_OK )
            fprintf(stderr,"WD-QP-RESYNC-PUBLISH-FAILED rank=%u generation=%llu; peers keep their old view until the next scan\n",weightd_mesh.local_rank,(unsigned long long)weightd_mesh.wire_generation);
    }
    SparkWeightdMeshWireReadyLocked(plan.wired);
    if ( plan.wired != 0u )
        SparkWeightdMeshHybridRefreshLocked(weightd_mesh.hybrid_owner_band);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    SparkWeightdMeshReadyMarkerSync();
    if ( repair != 0u )
        __atomic_store_n(&weightd_mesh.repair_pending,1u,__ATOMIC_RELEASE);
}

static struct ibv_context *SparkWeightdMeshOpenContext(const char *interface_name)
{
    struct ibv_device **devices;
    struct ibv_context *context = 0;
    int device_count = 0,device_index;
    devices = ibv_get_device_list(&device_count);
    for (device_index = 0; devices != 0 && device_index < device_count && context == 0; device_index++)
    {
        const char *name = ibv_get_device_name(devices[device_index]);
        if (name != 0 && strcmp(name,interface_name) == 0)
            context = ibv_open_device(devices[device_index]);
    }
    if (devices != 0)
        ibv_free_device_list(devices);
    if (context == 0)
        fprintf(stderr,"WD-MESH-PORT interface=%s: no such RDMA device; the port is not used\n",interface_name);
    return context;
}

static const char *SparkWeightdMeshPortProbe(SparkWeightdMeshPort *port,uint32_t *active)
{
    struct ibv_port_attr attributes;
    union ibv_gid gid;
    *active = 0u;
    if (ibv_query_port(port->context,1,&attributes) != 0 || attributes.state != IBV_PORT_ACTIVE)
        return "port 1 not active";
    *active = 1u;
    if (ibv_query_gid(port->context,1,(int)port->sgid_index,&gid) != 0)
        return "gid query failed";
    if (SparkWeightdMeshGidAddress(gid.raw) == 0u)
        return "gid holds no IPv4 RoCE address";
    port->lid = (uint16_t)attributes.lid;
    memcpy(port->gid,gid.raw,sizeof(port->gid));
    return 0;
}

static SparkStatus SparkWeightdMeshWaitPorts(SparkWeightdMeshPort *ports,uint32_t count,uint32_t *usable)
{
    struct timespec pause = {0,SPARK_WEIGHTD_MESH_ADDRESS_POLL_NS};
    uint64_t started = SparkMonotonicNs(),first_up = 0u,now;
    uint32_t index,pending,active;
    const char *reason;
    char text[40];
    *usable = 0u;
    for (;;)
    {
        pending = 0u;
        now = SparkMonotonicNs();
        for (index = 0u; index < count; index++)
        {
            SparkWeightdMeshPort *port = &ports[index];
            if (port->state != SPARK_WEIGHTD_MESH_PORT_PENDING)
                continue;
            reason = SparkWeightdMeshPortProbe(port,&active);
            if (reason == 0)
            {
                port->state = SPARK_WEIGHTD_MESH_PORT_UP;
                first_up = *usable == 0u ? now : first_up;
                *usable |= 1u << index;
                SparkWeightdMeshGidText(port->gid,text,sizeof(text));
                fprintf(stderr,"WD-MESH-ADDRESS interface=%s sgid=%u address=%s waited=%llu ms\n",port->name,port->sgid_index,text,
                    (unsigned long long)((now - started) / 1000000ull));
                continue;
            }
            if (reason != port->logged)
            {
                fprintf(stderr,"WD-MESH-WAIT-ADDRESS interface=%s sgid=%u: %s\n",port->name,port->sgid_index,reason);
                port->logged = reason;
            }
            if (now - started >= weightd_mesh_address_wait_ns)
            {
                fprintf(stderr,"WD-MESH-ADDRESS-DEADLINE interface=%s sgid=%u after %llu ms: %s; the port is not used\n",
                    port->name,port->sgid_index,(unsigned long long)((now - started) / 1000000ull),reason);
                port->state = SPARK_WEIGHTD_MESH_PORT_UNUSED;
                continue;
            }
            if (active == 0u && *usable != 0u && now - first_up >= weightd_mesh_port_grace_ns)
            {
                fprintf(stderr,"WD-MESH-PORT interface=%s sgid=%u: %s %llu ms after another port came up; the port is not used\n",
                    port->name,port->sgid_index,reason,(unsigned long long)((now - first_up) / 1000000ull));
                port->state = SPARK_WEIGHTD_MESH_PORT_UNUSED;
                continue;
            }
            pending |= 1u << index;
        }
        if (pending == 0u)
            break;
        if (nanosleep(&pause,0) != 0 && errno == EINTR)
        {
            fprintf(stderr,"WD-MESH-WAIT-ADDRESS: interrupted; startup stopped\n");
            return SPARK_STATUS_IO_ERROR;
        }
    }
    if (*usable == 0u)
    {
        fprintf(stderr,"WD-MESH-NO-PORT rank=%u: no configured port is active with an IPv4 RoCE address; startup failed\n",weightd_mesh.local_rank);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdMeshOpenPair(const SparkWeightdMeshPort *port)
{
    weightd_mesh.pair_context = port->context;
    weightd_mesh.pair_sgid_index = port->sgid_index;
    weightd_mesh.pair_lid = port->lid;
    memcpy(weightd_mesh.pair_gid,port->gid,sizeof(weightd_mesh.pair_gid));
    weightd_mesh.pair_protection_domain = ibv_alloc_pd(weightd_mesh.pair_context);
    weightd_mesh.pair_cq = weightd_mesh.pair_protection_domain != 0 ?
        ibv_create_cq(weightd_mesh.pair_context,SPARK_WEIGHTD_MESH_CQ_ENTRIES,0,0,0) : 0;
    if (weightd_mesh.pair_cq != 0)
    {
        weightd_mesh.pair_recv_mr = ibv_reg_mr(weightd_mesh.pair_protection_domain,weightd_mesh.recv_buffer,
            SPARK_WEIGHTD_MESH_REGION_BYTES,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        weightd_mesh.pair_staging_mr = ibv_reg_mr(weightd_mesh.pair_protection_domain,weightd_mesh.staging_buffer,
            SPARK_WEIGHTD_MESH_STAGING_BYTES,IBV_ACCESS_LOCAL_WRITE);
        weightd_mesh.pair_seq_mr = ibv_reg_mr(weightd_mesh.pair_protection_domain,&weightd_mesh.seq_storage,
            sizeof(weightd_mesh.seq_storage),IBV_ACCESS_LOCAL_WRITE);
    }
    if (weightd_mesh.pair_cq != 0 && SparkWeightdMeshCompletionQueueInit(1u,weightd_mesh.pair_cq) != SPARK_STATUS_OK)
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    if (weightd_mesh.pair_recv_mr == 0 || weightd_mesh.pair_staging_mr == 0 || weightd_mesh.pair_seq_mr == 0)
    {
        fprintf(stderr,"weightd-mesh: pair %s pd/cq/mr failed errno=%d\n",port->name,errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static struct ibv_pd *SparkWeightdMeshSendDomain(uint32_t slot,uint32_t peer,struct ibv_context *context,struct ibv_pd *base_pd)
{
    struct ibv_td_init_attr td_attr;
    struct ibv_parent_domain_init_attr pd_attr;
    struct ibv_td *td;
    struct ibv_pd *parent = 0;
    memset(&td_attr,0,sizeof(td_attr));
    td = ibv_alloc_td(context,&td_attr);
    if ( td == 0 )
        return base_pd;
#ifdef IBV_ALLOCATOR_USE_DEFAULT
    if ( weightd_mesh.hybrid_shm.base != 0 )
    {
        memset(&pd_attr,0,sizeof(pd_attr));
        pd_attr.pd = base_pd;
        pd_attr.td = td;
        pd_attr.comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS | IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT;
        pd_attr.alloc = SparkWeightdMeshShmAllocCb;
        pd_attr.free = SparkWeightdMeshShmFreeCb;
        pd_attr.pd_context = &weightd_mesh.hybrid_shm;
        parent = ibv_alloc_parent_domain(context,&pd_attr);
    }
#endif
    if ( parent == 0 )
    {
        memset(&pd_attr,0,sizeof(pd_attr));
        pd_attr.pd = base_pd;
        pd_attr.td = td;
        parent = ibv_alloc_parent_domain(context,&pd_attr);
    }
    if ( parent == 0 )
    {
        (void)ibv_dealloc_td(td);
        return base_pd;
    }
    weightd_mesh.slot_send_tds[slot][peer] = td;
    weightd_mesh.slot_send_pds[slot][peer] = parent;
    return parent;
}

static struct ibv_qp *SparkWeightdMeshCreateSendQp(uint32_t slot,uint32_t peer,struct ibv_context *context,
    struct ibv_pd *base_pd,struct ibv_cq *cq,uint32_t *granted_inline)
{
    static const uint32_t ladder[] = { 896u, 768u, 512u, 256u, 64u };
    struct ibv_pd *domain = SparkWeightdMeshSendDomain(slot,peer,context,base_pd);
    struct ibv_qp_init_attr attributes;
    struct ibv_qp *qp = 0;
    uint32_t attempt,i;
    *granted_inline = SPARK_WEIGHTD_MESH_INLINE_BYTES;
    for (attempt = 0u; attempt < 2u && qp == 0; attempt++)
    {
        struct ibv_pd *target = attempt == 0u ? domain : base_pd;
        if ( attempt != 0u && target == domain )
            break;
        for (i = 0u; i < sizeof(ladder) / sizeof(ladder[0]); i++)
        {
            memset(&attributes,0,sizeof(attributes));
            attributes.send_cq = cq;
            attributes.recv_cq = cq;
            attributes.cap.max_send_wr = SPARK_WEIGHTD_MESH_SEND_CAPACITY;
            attributes.cap.max_recv_wr = 64u;
            attributes.cap.max_send_sge = 1u;
            attributes.cap.max_recv_sge = 1u;
            attributes.cap.max_inline_data = ladder[i];
            attributes.qp_type = IBV_QPT_RC;
            qp = ibv_create_qp(target,&attributes);
            if ( qp != 0 )
            {
                *granted_inline = attributes.cap.max_inline_data;
                return qp;
            }
        }
    }
    return 0;
}

static SparkStatus SparkWeightdMeshInitQps(uint32_t rank,uint32_t rank_mask,uint32_t partner)
{
    struct ibv_qp_init_attr qp_attributes;
    SparkStatus status;
    uint32_t peer,slot;
    memset(&qp_attributes,0,sizeof(qp_attributes));
    qp_attributes.cap.max_send_wr = SPARK_WEIGHTD_MESH_SEND_CAPACITY;
    qp_attributes.cap.max_recv_wr = 64;
    qp_attributes.cap.max_send_sge = 1;
    qp_attributes.cap.max_recv_sge = 1;
    qp_attributes.cap.max_inline_data = 0;
    qp_attributes.qp_type = IBV_QPT_RC;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        uint32_t peer_rank = peer < rank ? peer : peer + 1u;
        if ( (rank_mask & (1u << peer_rank)) == 0u ) continue;
        for (slot = 0u; slot < 2u; slot++)
        {
            struct ibv_context *ctx = slot != 0u ? weightd_mesh.pair_context : weightd_mesh.context;
            struct ibv_pd *domain = slot != 0u ? weightd_mesh.pair_protection_domain : weightd_mesh.protection_domain;
            struct ibv_cq *slot_cq = slot != 0u ? weightd_mesh.pair_cq : weightd_mesh.cq;
            if ( slot != 0u && (domain == 0 || (weightd_mesh.second_rail == 0u && peer_rank != partner)) )
                continue;
            weightd_mesh.slot_send_qps[slot][peer] = SparkWeightdMeshCreateSendQp(slot,peer,ctx,domain,slot_cq,&weightd_mesh.slot_send_inline[slot][peer]);
            if (weightd_mesh.slot_send_qps[slot][peer] == 0)
            {
                fprintf(stderr,"weightd-mesh: send qp create peer=%u slot=%u failed\n",peer,slot);
                return SPARK_STATUS_DRIVER_LOAD_ERROR;
            }
        }
    }
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        uint32_t peer_rank = peer < rank ? peer : peer + 1u;
        if ( (rank_mask & (1u << peer_rank)) == 0u ) continue;
        for (slot = 0u; slot < 2u; slot++)
        {
            struct ibv_pd *domain = slot != 0u ? weightd_mesh.pair_protection_domain : weightd_mesh.protection_domain;
            struct ibv_cq *slot_cq = slot != 0u ? weightd_mesh.pair_cq : weightd_mesh.cq;
            if ( slot != 0u && (domain == 0 || (weightd_mesh.second_rail == 0u && peer_rank != partner)) )
                continue;
            qp_attributes.send_cq = slot_cq;
            qp_attributes.recv_cq = slot_cq;
            weightd_mesh.slot_recv_qps[slot][peer] = ibv_create_qp(domain,&qp_attributes);
            if (weightd_mesh.slot_recv_qps[slot][peer] == 0)
            {
                fprintf(stderr,"weightd-mesh: recv qp create peer=%u slot=%u failed\n",peer,slot);
                return SPARK_STATUS_DRIVER_LOAD_ERROR;
            }
        }
        weightd_mesh.send_qps[peer] = weightd_mesh.slot_send_qps[0][peer];
        weightd_mesh.recv_qps[peer] = weightd_mesh.slot_recv_qps[0][peer];
        status = SparkWeightdMeshSendQueueInit(peer,weightd_mesh.send_qps[peer],0u,weightd_mesh.slot_send_inline[0][peer]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir, uint32_t rank_mask,
    const char *pair_interface_name, uint32_t pair_sgid_index, uint32_t traffic_class)
{
    SparkWeightdMeshRecord own_record;
    SparkWeightdMeshPort ports[SPARK_WEIGHTD_MESH_PORTS];
    SparkStatus status;
    uint32_t peer,port,count = 1u,usable,partner = rank ^ 1u;
    char ports_text[160];

    if ( mesh_dir != 0 && mesh_dir[0] != '\0' )
        weightd_mesh_dir = mesh_dir;
    if (rank >= SPARK_WEIGHTD_MESH_RANKS || interface_name == 0 ||
        interface_name[0] == '\0' || sgid_index > 255u || traffic_class > 255u ||
        (rank_mask & (1u << rank)) == 0u ||
        (rank_mask >> SPARK_WEIGHTD_MESH_RANKS) != 0u ||
        __builtin_popcount(rank_mask) < 2)
    {
        fprintf(stderr,"weightd-mesh: bad init rank=%u interface=%s sgid=%u traffic_class=%u\n",
            rank,interface_name != 0 ? interface_name : "(null)",sgid_index,traffic_class);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (pair_interface_name != 0 && pair_interface_name[0] != '\0')
    {
        if (pair_sgid_index > 255u || (weightd_mesh_second_rail == 0u && (rank_mask & (1u << partner)) == 0u))
        {
            fprintf(stderr,"weightd-mesh: pair %s sgid=%u needs sgid 0..255 and, without --mesh-second-rail, partner rank %u in the mesh\n",
                pair_interface_name,pair_sgid_index,partner);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        count = 2u;
    }
    if ( weightd_mesh.hybrid_shm.base != 0 || weightd_mesh.hybrid_shm.path[0] != '\0' )
        SparkWeightdMeshShmSlabRelease(1u);
    memset(&weightd_mesh,0,sizeof(weightd_mesh));
    weightd_mesh.hybrid_shm.fd = -1;
    weightd_mesh.hybrid_owner_lane = UINT32_MAX;
    weightd_mesh.hybrid_owner_band = UINT32_MAX;
    weightd_mesh.local_rank = rank;
    weightd_mesh.rank_mask = rank_mask;
    weightd_mesh.traffic_class = traffic_class;
    weightd_mesh.second_rail = count == 2u ? weightd_mesh_second_rail : 0u;
    weightd_mesh.switched_pair = count == 2u ? weightd_mesh_second_rail : 1u;
    weightd_mesh.port_name[0] = interface_name;
    weightd_mesh.port_name[1] = count == 2u ? pair_interface_name : 0;
    weightd_mesh.boot_ns = SparkWeightdMeshRealtimeNs();
    weightd_mesh_boot_phase_ns = weightd_mesh.boot_ns;
    SparkWeightdMeshPhase("init-begin");
    {
        char ready_path[256],temp_path[264];
        SparkWeightdMeshReadyPath(ready_path,sizeof(ready_path));
        (void)snprintf(temp_path,sizeof(temp_path),"%s.tmp",ready_path);
        (void)unlink(ready_path);
        (void)unlink(temp_path);
    }
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
        if ( (rank_mask & (1u << (peer < rank ? peer : peer + 1u))) != 0u )
        {
            weightd_mesh.peer_state[peer] = SPARK_WEIGHTD_MESH_PEER_NO_RECORD;
            weightd_mesh.peer_record_status[peer] = SPARK_STATUS_BUSY;
            weightd_mesh.peer_since_ns[peer] = SparkMonotonicNs();
        }
    memset(ports,0,sizeof(ports));
    ports[0].name = interface_name;
    ports[0].sgid_index = sgid_index;
    ports[1].name = pair_interface_name;
    ports[1].sgid_index = pair_sgid_index;
    (void)setenv("MLX5_TOTAL_UUARS","64",0);
    (void)setenv("MLX5_NUM_LOW_LAT_UUARS","32",0);
    for (port = 0u; port < count; port++)
    {
        ports[port].context = SparkWeightdMeshOpenContext(ports[port].name);
        ports[port].state = ports[port].context != 0 ? SPARK_WEIGHTD_MESH_PORT_PENDING : SPARK_WEIGHTD_MESH_PORT_UNUSED;
    }
    status = SparkWeightdMeshWaitPorts(ports,count,&usable);
    for (port = 0u; port < count; port++)
        if (ports[port].context != 0 && (usable & (1u << port)) == 0u)
        {
            (void)ibv_close_device(ports[port].context);
            ports[port].context = 0;
        }
    if (status != SPARK_STATUS_OK)
        return status;
    weightd_mesh.ports = usable;
    weightd_mesh.home_port = SparkWeightdMeshHomePort(usable);
    weightd_mesh.context = ports[weightd_mesh.home_port].context;
    weightd_mesh.sgid_index = ports[weightd_mesh.home_port].sgid_index;
    weightd_mesh.lid = ports[weightd_mesh.home_port].lid;
    memcpy(weightd_mesh.gid,ports[weightd_mesh.home_port].gid,sizeof(weightd_mesh.gid));
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
    status = SparkWeightdMeshCompletionQueueInit(0u,weightd_mesh.cq);
    if (status != SPARK_STATUS_OK)
        return status;
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
    weightd_mesh.staging_memfd = memfd_create("spark-mesh-staging",0u);
    if (weightd_mesh.staging_memfd < 0 ||
        ftruncate(weightd_mesh.staging_memfd,(off_t)SPARK_WEIGHTD_MESH_STAGING_BYTES) != 0)
    {
        fprintf(stderr,"weightd-mesh: staging memfd failed errno=%d\n",errno);
        return SPARK_STATUS_IO_ERROR;
    }
    weightd_mesh.staging_buffer = mmap(0,SPARK_WEIGHTD_MESH_STAGING_BYTES,
        PROT_READ | PROT_WRITE,MAP_SHARED,weightd_mesh.staging_memfd,0);
    if (weightd_mesh.staging_buffer == MAP_FAILED)
    {
        fprintf(stderr,"weightd-mesh: staging mmap failed errno=%d\n",errno);
        weightd_mesh.staging_buffer = 0;
        return SPARK_STATUS_IO_ERROR;
    }
    weightd_mesh.staging_mr = ibv_reg_mr(weightd_mesh.protection_domain,
        weightd_mesh.staging_buffer,SPARK_WEIGHTD_MESH_STAGING_BYTES,IBV_ACCESS_LOCAL_WRITE);
    if (weightd_mesh.staging_mr == 0)
    {
        fprintf(stderr,"weightd-mesh: staging mr failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    printf("weightd-mesh staging bytes=%llu (bands=%u peers=%u slot_bytes=%llu capabilities=%u abi=%u)\n",
        (unsigned long long)SPARK_WEIGHTD_MESH_STAGING_BYTES,
        (unsigned)SPARK_WEIGHTD_MESH_BANDS,
        (unsigned)SPARK_WEIGHTD_MESH_RANKS_PER_BAND,
        (unsigned long long)SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES,
        (unsigned)SPARK_WEIGHTD_MESH_CAPABILITIES,
        (unsigned)SPARK_WEIGHTD_IPC_ABI_VERSION);
    fflush(stdout);
    SparkWeightdMeshPhase("staging-mr-registered");
    weightd_mesh.seq_mr = ibv_reg_mr(weightd_mesh.protection_domain,
        &weightd_mesh.seq_storage,sizeof(weightd_mesh.seq_storage),
        IBV_ACCESS_LOCAL_WRITE);
    if (weightd_mesh.seq_mr == 0)
    {
        fprintf(stderr,"weightd-mesh: seq mr failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    if (usable == 3u)
    {
        status = SparkWeightdMeshOpenPair(&ports[1]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    SparkWeightdMeshShmSlabInit(rank);
    status = SparkWeightdMeshInitQps(rank,rank_mask,partner);
    if (status != SPARK_STATUS_OK)
        return status;
    SparkWeightdMeshHybridHeaderInit();
    SparkWeightdMeshPortsText(usable,ports_text,sizeof(ports_text));
    if (usable == 3u)
        fprintf(stderr,"WD-MESH-PORTS rank=%u ports=%s: %s\n",rank,ports_text,weightd_mesh.second_rail != 0u ?
            "a peer with both ports up uses the second port when its rank parity differs from ours, the first otherwise" :
            "the partner uses the second port, every other peer the first");
    else
        fprintf(stderr,"WD-MESH-PORTS rank=%u ports=%s: every peer uses %s%s%s\n",rank,ports_text,
            SparkWeightdMeshPortName(weightd_mesh.home_port),count == 2u ? "; not used: " : "",
            count == 2u ? SparkWeightdMeshPortName(weightd_mesh.home_port ^ 1u) : "");
    SparkWeightdMeshFillRecord(&own_record);
    for (peer = 0u, port = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
        port += (own_record.send_qpn[peer] != 0u) + (own_record.recv_qpn[peer] != 0u) +
            (own_record.pair_send_qpn[peer] != 0u) + (own_record.pair_recv_qpn[peer] != 0u);
    printf("weightd-mesh: rank=%u publishing %u QPs traffic_class=%u\n",weightd_mesh.local_rank,port,weightd_mesh.traffic_class);
    fflush(stdout);
    if (SparkWeightdMeshWriteRecord(&own_record) != SPARK_STATUS_OK)
        return SPARK_STATUS_IO_ERROR;
    printf("weightd-mesh: published; wiring continues as peers appear\n");
    fflush(stdout);
    SparkWeightdMeshPhase("published");
    weightd_mesh.mesh_generation = 1u;
    weightd_mesh.mesh_active = 1u;
    return SPARK_STATUS_BUSY;
}

uint32_t SparkWeightdMeshReady(void)
{
    return weightd_mesh.mesh_ready;
}

SparkStatus SparkWeightdMeshStatusFill(SparkWeightdIpcMeshStatusResult *result)
{
    uint32_t peer,rank,lane,index,where;
    if ( result == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    result->pair_rank = UINT32_MAX;
    if ( weightd_mesh.mesh_active == 0u )
    {
        result->mesh_state = SPARK_WEIGHTD_MESH_STATE_DISABLED;
        return SPARK_STATUS_OK;
    }
    result->local_rank = weightd_mesh.local_rank;
    result->rank_mask = weightd_mesh.rank_mask;
    result->boot_ns = weightd_mesh.boot_ns;
    if ( weightd_mesh.pair_protection_domain != 0 && weightd_mesh.second_rail == 0u )
        result->pair_rank = weightd_mesh.local_rank ^ 1u;
    result->now_mono_ns = SparkMonotonicNs();
    result->peers[weightd_mesh.local_rank].state = SPARK_WEIGHTD_MESH_PEER_SELF;
    result->wired_mask = 1u << weightd_mesh.local_rank;
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
        if ( (weightd_mesh.rank_mask & (1u << rank)) == 0u )
            continue;
        result->peers[rank].state = weightd_mesh.peer_state[peer];
        result->peers[rank].record_status = weightd_mesh.peer_record_status[peer];
        result->peers[rank].qp_flags = weightd_mesh.peer_qp_flags[peer];
        result->peers[rank].record_boot_ns = weightd_mesh.peer_record_boot_ns[peer];
        result->peers[rank].since_mono_ns = weightd_mesh.peer_since_ns[peer];
        if ( weightd_mesh.peer_state[peer] == SPARK_WEIGHTD_MESH_PEER_WIRED )
            result->wired_mask |= 1u << rank;
    }
    SparkWeightdMeshWireAcquire();
    result->mesh_state = weightd_mesh.mesh_ready != 0u ? SPARK_WEIGHTD_MESH_STATE_READY : SPARK_WEIGHTD_MESH_STATE_WIRING;
    result->ready_since_mono_ns = weightd_mesh.ready_since_ns;
    result->mesh_generation = __atomic_load_n(&weightd_mesh.mesh_generation,__ATOMIC_RELAXED);
    result->counters[0] = weightd_mesh.stat_trywire;
    result->counters[1] = weightd_mesh.stat_record_failures;
    result->counters[2] = weightd_mesh.stat_record_invalid;
    result->counters[3] = weightd_mesh.stat_wire_failures;
    result->counters[4] = weightd_mesh.stat_rewires;
    result->counters[5] = weightd_mesh.stat_unready;
    result->counters[6] = weightd_mesh.stat_repairs;
    result->counters[7] = weightd_mesh.send_ok;
    result->counters[8] = weightd_mesh.send_err;
    result->counters[9] = weightd_mesh.stat_peer_resets;
    result->counters[10] = weightd_mesh.stat_peer_reset_bits;
    result->counters[11] = weightd_mesh.stat_lane_resets;
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
        if ( (weightd_mesh.rank_mask & (1u << rank)) == 0u )
            continue;
        result->peers[rank].wired_boot_ns = weightd_mesh.wired_boot_ns[peer];
        result->peers[rank].send_pending = weightd_mesh.send_pending[peer];
        result->peers[rank].rpc_pending = weightd_mesh.rpc_pending[peer];
        result->peers[rank].last_ok_mono_ns = weightd_mesh.peer_last_ok_ns[peer];
        result->peers[rank].last_err_mono_ns = weightd_mesh.peer_last_err_ns[peer];
        result->peers[rank].cq_err_since_ok = weightd_mesh.peer_cq_err_since_ok[peer];
    }
    for (lane=0u; lane<SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
    {
        const SparkWeightdMeshTopology *topology = &weightd_mesh.lane_topology[lane];
        SparkWeightdMeshLaneStatus *status = &result->lanes[lane];
        status->rank_count = topology->rank_count;
        status->local_rank = topology->local_rank;
        for (rank=0u; rank<topology->rank_count; rank++)
        {
            status->packed_ranks |= (uint64_t)(topology->physical_ranks[rank] & 0xfu) << (4u * rank);
            status->physical_mask |= 1u << topology->physical_ranks[rank];
        }
        status->flags = topology->rank_count != 0u ? SPARK_WEIGHTD_MESH_LANE_CONFIGURED : 0u;
        status->activity = weightd_mesh.lane_activity[lane];
        for (index=2u * lane * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index<2u * (lane + 1u) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND; index++)
        {
            status->failed_cells += SparkWeightdMeshCellFailed(index);
            status->pending_cells += SparkWeightdMeshCellPending(index);
        }
        status->busy_reason = SparkWeightdMeshLaneCheckLocked(lane,&where);
        status->busy_index = where;
        status->configure_status = status->busy_reason != 0u ? SPARK_STATUS_BUSY : SPARK_STATUS_OK;
    }
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return SPARK_STATUS_OK;
}

void SparkWeightdMeshStop(void)
{
    uint32_t removed;
    if ( weightd_mesh.mesh_active == 0u )
        return;
    SparkWeightdMeshWireAcquire();
    SparkWeightdMeshHybridRevokeLocked();
    SparkWeightdMeshShmSlabRelease(0u);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    removed = SparkWeightdMeshReadyMarkerRemove(1u);
    fprintf(stderr,"WD-MESH-STOP rank=%u ready_marker_removed=%u\n",weightd_mesh.local_rank,removed);
}

static void SparkWeightdMeshShipAdvance(uint32_t record,uint64_t now_ns)
{
    uint32_t index = record / 2u,band = index / SPARK_WEIGHTD_MESH_RANKS_PER_BAND,rank = index % SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
    const SparkWeightdMeshTransfer *transfer = &weightd_mesh.transfers[record],*other = &weightd_mesh.transfers[record ^ 1u];
    uint64_t shipped = transfer->seq;
    SparkLatencyAdd(&weightd_mesh.timing.ship,weightd_mesh.post_ns[record],now_ns);
    if ( other->seq < transfer->seq && other->seq > weightd_mesh.doorbell_posted[index] &&
         (other->pending != 0u || other->failed != 0u) )
        return;
    if ( other->seq > shipped && other->seq <= weightd_mesh.issued[index] && other->pending == 0u && other->failed == 0u )
        shipped = other->seq;
    if ( shipped <= weightd_mesh.doorbell_posted[index] )
        return;
    weightd_mesh.doorbell_posted[index] = shipped;
    weightd_mesh.doorbell_stuck[index] = 0u;
    __atomic_store_n((uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank)),shipped,__ATOMIC_RELEASE);
    weightd_mesh.ship_log_count++;
}

static void SparkWeightdMeshCompleteTransfer(uint64_t work_id, int success, uint64_t now_ns)
{
    uint32_t record;
    uint32_t bit;
    SparkWeightdMeshTransfer *transfer;
    if ( (work_id & SPARK_WEIGHTD_MESH_TRANSFER_ID) == 0u )
    {
        uint64_t peer = work_id & 0xffu;
        uint32_t retire = (work_id & SPARK_WEIGHTD_MESH_RPC_CHAINED) != 0u ? 2u : 1u;
        if ( peer < SPARK_WEIGHTD_MESH_PEERS &&
             (uint32_t)((work_id & ~SPARK_WEIGHTD_MESH_RPC_CHAINED) >> SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) == weightd_mesh.wire_epoch[peer] &&
             weightd_mesh.rpc_pending[peer] != 0u )
        {
            uint32_t drop = retire <= weightd_mesh.rpc_pending[peer] ? retire : weightd_mesh.rpc_pending[peer];
            uint32_t drop_send = drop <= weightd_mesh.send_pending[peer] ? drop : weightd_mesh.send_pending[peer];
            weightd_mesh.rpc_pending[peer] -= drop;
            weightd_mesh.send_pending[peer] -= drop_send;
            weightd_mesh.send_queues[peer].wr_retired += drop_send;
            weightd_mesh.send_pending_total -= drop_send <= weightd_mesh.send_pending_total ? drop_send : weightd_mesh.send_pending_total;
        }
        return;
    }
    record = (uint32_t)((work_id >> 6u) & 1023u);
    bit = (uint32_t)(work_id & 63u);
    if ( record >= SPARK_WEIGHTD_MESH_RECORDS )
        return;
    transfer = &weightd_mesh.transfers[record];
    if ( transfer->generation != ((work_id & ~SPARK_WEIGHTD_MESH_TRANSFER_ID) >> 16u) )
        return;
    if ( (transfer->pending & (UINT64_C(1) << bit)) == 0u )
    {
        if ( success == 0 && (bit & 3u) == 2u && (transfer->chained & (1u << (bit / 4u))) != 0u && transfer->failed == 0u )
        {
            transfer->failed = SPARK_WEIGHTD_MESH_TRANSFER_FAILED;
            SparkWeightdMeshChanged();
        }
        return;
    }
    transfer->pending &= ~(UINT64_C(1) << bit);
    if ( (bit & 3u) == 3u && (transfer->chained & (1u << (bit / 4u))) != 0u )
    {
        uint32_t chained_writes = transfer->chained_writes[bit / 4u];
        transfer->chained &= ~(1u << (bit / 4u));
        weightd_mesh.send_pending[bit / 4u] -= chained_writes;
        weightd_mesh.send_queues[bit / 4u].wr_retired += chained_writes;
        weightd_mesh.send_pending_total -= chained_writes;
    }
    weightd_mesh.send_pending[bit / 4u]--;
    weightd_mesh.send_queues[bit / 4u].wr_retired++;
    weightd_mesh.send_pending_total--;
    if ( success == 0 && transfer->failed == 0u )
    {
        transfer->failed = SPARK_WEIGHTD_MESH_TRANSFER_FAILED;
        SparkWeightdMeshChanged();
    }
    if ( transfer->pending == 0u && transfer->failed == 0u )
        SparkWeightdMeshShipAdvance(record,now_ns);
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

static uint32_t SparkWeightdMeshSendRoom(uint32_t physical_mask,uint32_t needed,uint32_t needed_bbs)
{
    uint32_t mask = physical_mask & ~(1u << weightd_mesh.local_rank);
    while ( mask != 0u )
    {
        uint32_t peer_rank = (uint32_t)__builtin_ctz(mask);
        uint32_t peer = peer_rank < weightd_mesh.local_rank ? peer_rank : peer_rank - 1u;
        mask &= mask - 1u;
        if ( peer >= SPARK_WEIGHTD_MESH_PEERS )
            continue;
        if ( weightd_mesh.send_pending[peer] > SPARK_WEIGHTD_MESH_SEND_CAPACITY - needed ||
             (SparkWeightdMeshSqFree(&weightd_mesh.send_queues[peer]) < needed_bbs &&
              SparkWeightdMeshSqEnsureRoom(peer,needed_bbs) != SPARK_STATUS_OK) )
            return(0u);
    }
    return(1u);
}

static void SparkWeightdMeshPostPeer(uint32_t record,uint32_t band,uint32_t peer,uint64_t slot_base,uint64_t offset,uint64_t length,
    SparkWeightdMeshRoute route,const uint8_t *logical,const uint64_t *tag)
{
    SparkWeightdMeshTransfer *transfer = &weightd_mesh.transfers[record];
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
    uint64_t base = SPARK_WEIGHTD_MESH_TRANSFER_ID | (transfer->generation << 16u) | ((uint64_t)record << 6u) | (uint64_t)(peer * 4u);
    SparkStatus status = SPARK_STATUS_OK;
    if ( route.fields.compact_tail != 0u && length != 0u )
    {
        uint64_t half_off = length + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ?
            SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot_base / SPARK_WEIGHTD_MESH_SLOT_BYTES) : 0u;
        uint64_t local_addr = route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER ?
            ((uint64_t)(uintptr_t)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,logical[peer_rank]) + half_off) :
            ((uint64_t)(uintptr_t)weightd_mesh.recv_buffer + slot_base + offset);
        uint32_t lkey = route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER ?
            SparkWeightdMeshPeerMr(peer,weightd_mesh.staging_mr,weightd_mesh.pair_staging_mr)->lkey :
            SparkWeightdMeshPeerMr(peer,weightd_mesh.recv_mr,weightd_mesh.pair_recv_mr)->lkey;
        if ( SparkWeightdMeshSqWrite(peer,base | 3u,1u,local_addr,lkey,(uint32_t)(length + 8u),slot_base + offset,0) != SPARK_STATUS_OK )
        {
            transfer->failed = SPARK_WEIGHTD_MESH_TRANSFER_FAILED;
            return;
        }
        transfer->pending |= UINT64_C(1) << (peer * 4u + 3u);
        weightd_mesh.send_pending[peer]++;
        weightd_mesh.send_queues[peer].wr_issued++;
        weightd_mesh.send_pending_total++;
        return;
    }
    if ( length != 0u && route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER )
        status = SparkWeightdMeshSqWrite(peer,base | 2u,0u,(uint64_t)(uintptr_t)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,logical[peer_rank]) +
                (route.fields.half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ? SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot_base / SPARK_WEIGHTD_MESH_SLOT_BYTES) : 0u),
            SparkWeightdMeshPeerMr(peer,weightd_mesh.staging_mr,weightd_mesh.pair_staging_mr)->lkey,(uint32_t)length,slot_base + offset,0);
    else if ( length != 0u )
        status = SparkWeightdMeshSqWrite(peer,base | 2u,0u,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer + slot_base + offset,
            SparkWeightdMeshPeerMr(peer,weightd_mesh.recv_mr,weightd_mesh.pair_recv_mr)->lkey,(uint32_t)length,slot_base + offset,0);
    if ( status != SPARK_STATUS_OK )
    {
        transfer->failed = SPARK_WEIGHTD_MESH_TRANSFER_FAILED;
        return;
    }
    if ( length != 0u )
    {
        transfer->chained |= 1u << peer;
        transfer->chained_writes[peer] = 1u;
        weightd_mesh.send_pending[peer]++;
        weightd_mesh.send_queues[peer].wr_issued++;
        weightd_mesh.send_pending_total++;
    }
    if ( SparkWeightdMeshSqWrite(peer,base | 3u,1u,0u,0u,0u,slot_base + SPARK_WEIGHTD_MESH_TAIL_OFFSET(route.fields.half),tag) != SPARK_STATUS_OK )
    {
        transfer->failed = SPARK_WEIGHTD_MESH_TRANSFER_FAILED;
        return;
    }
    transfer->pending |= UINT64_C(1) << (peer * 4u + 3u);
    weightd_mesh.send_pending[peer]++;
    weightd_mesh.send_queues[peer].wr_issued++;
    weightd_mesh.send_pending_total++;
}

static void SparkWeightdMeshPostPairFirstPeer(uint32_t record,uint32_t band,uint32_t peer,uint64_t slot,uint64_t slot_base,uint64_t bytes,
    SparkWeightdMeshRoute route,const uint8_t *logical,uint32_t local,uint32_t ranks,const uint64_t *tag)
{
    SparkWeightdMeshTransfer *transfer = &weightd_mesh.transfers[record];
    uint32_t peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u, target = logical[peer_rank], segment, writes = 0u;
    uint64_t base = SPARK_WEIGHTD_MESH_TRANSFER_ID | (transfer->generation << 16u) | ((uint64_t)record << 6u) | (uint64_t)(peer * 4u);
    const uint64_t group = SparkWeightdMeshPairFirstGroupBytes(bytes), pair = SparkWeightdMeshPairFirstPairBytes(bytes);
    const uint64_t half = SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot);
    const uint32_t lkey = SparkWeightdMeshPeerMr(peer,weightd_mesh.staging_mr,weightd_mesh.pair_staging_mr)->lkey;
    SparkStatus status = SPARK_STATUS_OK;
    if ( ((target ^ local) & 1u) == 0u && group != 0u )
    {
        status = SparkWeightdMeshSqWrite(peer,base | 2u,0u,(uint64_t)(uintptr_t)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,target) + half,
            lkey,(uint32_t)group,slot_base + SPARK_WEIGHTD_MESH_HALF_OFFSET(route.fields.half),0);
        if ( status == SPARK_STATUS_OK )
            writes = 1u;
    }
    else if ( target == (local ^ 1u) && pair != 0u )
        for (segment = target & 1u; segment < ranks && status == SPARK_STATUS_OK; segment += 2u)
        {
            status = SparkWeightdMeshSqWrite(peer,base | 2u,0u,(uint64_t)(uintptr_t)weightd_mesh.staging_buffer + SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,segment) + half,
                lkey,(uint32_t)pair,SparkWeightdMeshPairFirstArea(band,segment ^ 1u,slot,route.fields.half),0);
            if ( status == SPARK_STATUS_OK )
                writes++;
        }
    if ( writes != 0u )
    {
        transfer->chained |= 1u << peer;
        transfer->chained_writes[peer] = (uint8_t)writes;
        weightd_mesh.send_pending[peer] += writes;
        weightd_mesh.send_queues[peer].wr_issued += writes;
        weightd_mesh.send_pending_total += writes;
    }
    if ( status != SPARK_STATUS_OK ||
         SparkWeightdMeshSqWrite(peer,base | 3u,1u,0u,0u,0u,slot_base + SPARK_WEIGHTD_MESH_TAIL_OFFSET(route.fields.half),tag) != SPARK_STATUS_OK )
    {
        transfer->failed = SPARK_WEIGHTD_MESH_TRANSFER_FAILED;
        return;
    }
    transfer->pending |= UINT64_C(1) << (peer * 4u + 3u);
    weightd_mesh.send_pending[peer]++;
    weightd_mesh.send_queues[peer].wr_issued++;
    weightd_mesh.send_pending_total++;
}

static void SparkWeightdMeshPostRoute(uint32_t record,uint32_t band,uint64_t slot,uint64_t slot_base,uint64_t bytes,SparkWeightdMeshRoute route,uint32_t physical_mask,const uint8_t *logical,uint32_t local)
{
    const SparkWeightdMeshTopology *topology = &weightd_mesh.lane_topology[band / 2u];
    uint64_t offset,length,tag;
    uint32_t step,target,peer,peer_rank,count = 0u,degree = topology->rank_count;
    uint8_t order[SPARK_WEIGHTD_MESH_PEERS];
    tag = route.fields.compact_tail != 0u ? 0u :
        *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer + slot_base + SPARK_WEIGHTD_MESH_TAIL_OFFSET(route.fields.half));
    for (step=1u; step<degree; step++)
    {
        target = SparkWeightdMeshSchedulePeer(local,degree,step);
        peer_rank = topology->physical_ranks[target];
        if ( (physical_mask & (1u << peer_rank)) == 0u || peer_rank == weightd_mesh.local_rank ) continue;
        peer = peer_rank < weightd_mesh.local_rank ? peer_rank : peer_rank - 1u;
        if ( route.fields.pair_first != 0u )
            SparkWeightdMeshPostPairFirstPeer(record,band,peer,slot,slot_base,bytes,route,logical,local,degree,&tag);
        else
        {
            offset = SparkWeightdMeshRouteSpan(route,target,local,bytes,&length);
            SparkWeightdMeshPostPeer(record,band,peer,slot_base,offset,length,route,logical,&tag);
        }
        order[count++] = (uint8_t)peer;
    }
    SparkWeightdMeshSqRingOrder(order,count);
    if ( SparkWeightdMeshPollingThread == 0u )
        SparkWeightdMeshWake();
}

static SparkStatus SparkWeightdMeshSlotPostable(uint32_t band,uint32_t rank,uint64_t seq,uint64_t slot,uint64_t bytes,SparkWeightdMeshRoute route)
{
    uint32_t peer_rank_mask = (uint32_t)route.fields.peer_mask;
    const SparkWeightdMeshTopology *topology;
    if ( band >= SPARK_WEIGHTD_MESH_BANDS || rank >= SPARK_WEIGHTD_MESH_RANKS_PER_BAND || seq == 0u || slot >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND || slot / SPARK_WEIGHTD_MESH_SLOTS_PER_RANK != rank || peer_rank_mask == 0u || SparkWeightdMeshRouteFits(route,bytes) == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER && weightd_mesh.staging_mr == 0 )
        return SPARK_STATUS_UNSUPPORTED;
    topology = &weightd_mesh.lane_topology[band / 2u];
    if ( topology->rank_count == 0u || rank != topology->local_rank || (peer_rank_mask & ~((1u << topology->rank_count) - 1u)) != 0u || (peer_rank_mask & (1u << rank)) != 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static uint32_t SparkWeightdMeshDoorbellShippable(uint32_t index)
{
    uint32_t alternate;
    if ( SparkWeightdMeshCellFailed(index) != 0u )
        return 0u;
    if ( weightd_mesh.hybrid_active != 0u )
    {
        SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
        if ( ctrl != 0 && weightd_mesh.hybrid_owner_band < SPARK_WEIGHTD_MESH_BANDS &&
             ctrl->owner_rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND &&
             index == weightd_mesh.hybrid_owner_band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + ctrl->owner_rank )
        {
            uint64_t issued = weightd_mesh.issued[index];
            if ( __atomic_load_n(&ctrl->trigger[0].seq,__ATOMIC_ACQUIRE) > issued ||
                 __atomic_load_n(&ctrl->trigger[1].seq,__ATOMIC_ACQUIRE) > issued )
                return 1u;
        }
    }
    for (alternate=0u; alternate<2u; alternate++)
    {
        volatile uint64_t *entry = SparkWeightdMeshDoorbellAt(index,alternate);
        uint64_t seq = __atomic_load_n(&entry[0],__ATOMIC_ACQUIRE);
        SparkWeightdMeshRoute route;
        route.word = entry[3];
        if ( seq > weightd_mesh.issued[index] && entry[1] != 0u &&
             SparkWeightdMeshSlotPostable(index / SPARK_WEIGHTD_MESH_RANKS_PER_BAND,index % SPARK_WEIGHTD_MESH_RANKS_PER_BAND,
                 seq,entry[2],entry[1],route) == SPARK_STATUS_OK )
            return 1u;
    }
    return 0u;
}

static uint32_t SparkWeightdMeshHybridLaneCapable(uint32_t lane,uint32_t *dbrec_in_shm_out)
{
    const SparkWeightdMeshLanePlan *plan;
    uint32_t i,all_dbr = 1u;
    if ( dbrec_in_shm_out != 0 )
        *dbrec_in_shm_out = 0u;
    if ( lane >= SPARK_WEIGHTD_MESH_MAX_LANES || weightd_mesh.hybrid_shm.base == 0 ||
         SparkWeightdMeshHybridCtrlPtr() == 0 )
        return 0u;
    plan = &weightd_mesh.lane_plan[lane];
    if ( plan->valid == 0u || plan->peer_count == 0u || plan->peer_count > SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS )
        return 0u;
    for (i = 0u; i < plan->peer_count; i++)
    {
        const SparkWeightdMeshSendQueue *q = &weightd_mesh.send_queues[plan->peers[i].peer];
        if ( q->buffer == 0 || q->sq_shm_off == SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE ||
             q->depth < SPARK_WEIGHTD_MESH_SEND_CAPACITY || (q->depth & (q->depth - 1u)) != 0u )
            return 0u;
        if ( q->dbrec_shm_off == SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE )
            all_dbr = 0u;
    }
    if ( dbrec_in_shm_out != 0 )
        *dbrec_in_shm_out = 0u;
    (void)all_dbr;
    return 1u;
}

static void SparkWeightdMeshHybridUpdateCreditsLocked(void)
{
    SparkWeightdMeshHybridCtrl *ctrl;
    const SparkWeightdMeshLanePlan *plan;
    uint32_t i;
    if ( weightd_mesh.hybrid_active == 0u || weightd_mesh.hybrid_owner_lane >= SPARK_WEIGHTD_MESH_MAX_LANES )
        return;
    ctrl = SparkWeightdMeshHybridCtrlPtr();
    if ( ctrl == 0 )
        return;
    plan = &weightd_mesh.lane_plan[weightd_mesh.hybrid_owner_lane];
    for (i = 0u; i < plan->peer_count; i++)
    {
        const SparkWeightdMeshSendQueue *q = &weightd_mesh.send_queues[plan->peers[i].peer];
        __atomic_store_n(&ctrl->cpu_sq_limit[i],q->retired + q->depth,__ATOMIC_RELEASE);
        __atomic_store_n(&ctrl->cpu_wr_limit[i],q->wr_retired + SPARK_WEIGHTD_MESH_SEND_CAPACITY,__ATOMIC_RELEASE);
    }
}

static uint32_t SparkWeightdMeshHybridRingOneLocked(SparkWeightdMeshHybridCtrl *ctrl,uint32_t trig_slot,uint64_t seq,uint64_t now_ns)
{
    const SparkWeightdMeshLanePlan *plan;
    SparkWeightdMeshTransfer *transfer;
    uint64_t meta = ctrl->trigger[trig_slot].meta;
    uint32_t peer_mask = SparkWeightdMeshHybridMetaPeerMask(meta);
    uint32_t ds = SparkWeightdMeshHybridMetaDs(meta);
    uint32_t num_bb = SparkWeightdMeshHybridMetaNumBb(meta);
    uint32_t gen = SparkWeightdMeshHybridMetaGrantGen(meta);
    uint32_t owner_lane = weightd_mesh.hybrid_owner_lane;
    uint32_t owner_band = weightd_mesh.hybrid_owner_band;
    uint32_t index,record,all_peers,i,count = 0u;
    uint64_t wr_base,pending_mask = 0u;
    uint8_t order[SPARK_WEIGHTD_MESH_PEERS];
    if ( owner_lane >= SPARK_WEIGHTD_MESH_MAX_LANES || owner_band >= SPARK_WEIGHTD_MESH_BANDS ||
         gen != weightd_mesh.hybrid_grant_gen )
        return 0u;
    plan = &weightd_mesh.lane_plan[owner_lane];
    if ( plan->valid == 0u || peer_mask == 0u || (peer_mask & ~plan->all_peers_mask) != 0u ||
         ds < 3u || num_bb == 0u || num_bb > 15u )
        return 0u;
    index = owner_band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + plan->local_rank;
    if ( seq <= weightd_mesh.issued[index] || SparkWeightdMeshCellFailed(index) != 0u )
        return 0u;
    record = SPARK_WEIGHTD_MESH_RECORD_OF(index,seq);
    transfer = &weightd_mesh.transfers[record];
    if ( transfer->pending != 0u )
    {
        (void)SparkWeightdMeshDrainCqLocked(now_ns);
        if ( transfer->pending != 0u )
            return 0u;
    }
    if ( SparkWeightdMeshNextTransferGeneration == (SPARK_WEIGHTD_MESH_TRANSFER_ID >> 16u) - 1u )
        return 0u;
    all_peers = peer_mask == plan->all_peers_mask ? 1u : 0u;
    if ( ctrl->dbrec_in_shm == 0u )
    {
        for (i = 0u; i < plan->peer_count; i++)
        {
            const SparkWeightdMeshPlanPeer *p = &plan->peers[i];
            if ( all_peers == 0u && (peer_mask & (1u << p->target)) == 0u )
                continue;
            weightd_mesh.send_queues[p->peer].record[MLX5_SND_DBR] =
                htobe32((weightd_mesh.send_queues[p->peer].produced + num_bb) & 0xffffu);
        }
        SparkWeightdMeshStoreBarrier();
    }
    if ( weightd_mesh.bf_all_unique != 0u )
    {
        for (i = 0u; i < plan->peer_count; i++)
        {
            const SparkWeightdMeshPlanPeer *p = &plan->peers[i];
            SparkWeightdMeshSendQueue *queue;
            uint32_t w0,w1;
            uint64_t ring_word;
            volatile uint64_t *doorbell;
            if ( all_peers == 0u && (peer_mask & (1u << p->target)) == 0u )
                continue;
            queue = &weightd_mesh.send_queues[p->peer];
            w0 = htobe32(((uint32_t)(uint16_t)queue->produced << 8u) | MLX5_OPCODE_RDMA_WRITE);
            w1 = htobe32(p->qpn_shifted | ds);
            ring_word = (uint64_t)w0 | ((uint64_t)w1 << 32u);
            doorbell = (volatile uint64_t *)((volatile uint8_t *)queue->doorbell + queue->bf_offset);
            *doorbell = ring_word;
            queue->bf_offset ^= queue->combining_bytes;
            queue->ring_word = ring_word;
        }
        SparkWeightdMeshStoreBarrier();
    }
    transfer->seq = seq;
    transfer->generation = ++SparkWeightdMeshNextTransferGeneration;
    transfer->failed = 0u;
    transfer->reset_peer = 0u;
    transfer->chained = 0u;
    weightd_mesh.issued[index] = seq;
    wr_base = SPARK_WEIGHTD_MESH_TRANSFER_ID | (transfer->generation << 16u) | ((uint64_t)record << 6u) | 3u;
    for (i = 0u; i < plan->peer_count; i++)
    {
        const SparkWeightdMeshPlanPeer *p = &plan->peers[i];
        SparkWeightdMeshSendQueue *queue;
        uint32_t peer,sq_idx;
        if ( all_peers == 0u && (peer_mask & (1u << p->target)) == 0u )
            continue;
        peer = p->peer;
        queue = &weightd_mesh.send_queues[peer];
        sq_idx = queue->produced & queue->mask;
        if ( weightd_mesh.bf_all_unique == 0u )
        {
            uint32_t w0 = htobe32(((uint32_t)(uint16_t)queue->produced << 8u) | MLX5_OPCODE_RDMA_WRITE);
            uint32_t w1 = htobe32(p->qpn_shifted | ds);
            queue->ring_word = (uint64_t)w0 | ((uint64_t)w1 << 32u);
        }
        queue->wr_id[sq_idx] = wr_base | ((uint64_t)peer << 2u);
        queue->wqe_bbs[sq_idx] = (uint8_t)num_bb;
        queue->produced += num_bb;
        queue->wr_issued++;
        pending_mask |= UINT64_C(1) << (peer * 4u + 3u);
        weightd_mesh.send_pending[peer]++;
        order[count++] = (uint8_t)peer;
    }
    transfer->pending = pending_mask;
    weightd_mesh.send_pending_total += count;
    if ( weightd_mesh.bf_all_unique == 0u )
        SparkWeightdMeshSqRingOrder(order,count);
    else
    {
        memcpy(weightd_mesh.ring_order,order,count);
        weightd_mesh.ring_count = count;
    }
    weightd_mesh.stat_hybrid_rings++;
    weightd_mesh.post_ns[record] = now_ns;
    if ( weightd_mesh.publish_seq[record] != seq )
    {
        weightd_mesh.publish_seq[record] = seq;
        weightd_mesh.publish_ns[record] = now_ns;
    }
    {
        uint64_t spent = SparkMonotonicNs() - now_ns;
        weightd_mesh.timing.route_count++;
        weightd_mesh.timing.route_ns += spent;
        weightd_mesh.timing.route_max_ns = spent > weightd_mesh.timing.route_max_ns ? spent : weightd_mesh.timing.route_max_ns;
    }
    SparkLatencyAdd(&weightd_mesh.timing.post,weightd_mesh.publish_ns[record],now_ns);
    return 1u;
}

static uint32_t SparkWeightdMeshHybridPollLocked(uint64_t now_ns)
{
    SparkWeightdMeshHybridCtrl *ctrl;
    uint32_t index,owner_band,owner_rank,order[2] = {0u, 0u},count = 0u,c,rung = 0u;
    uint64_t issued,seqs[2];
    if ( weightd_mesh.hybrid_active == 0u || weightd_mesh.mesh_ready == 0u )
        return 0u;
    ctrl = SparkWeightdMeshHybridCtrlPtr();
    owner_band = weightd_mesh.hybrid_owner_band;
    if ( ctrl == 0 || owner_band >= SPARK_WEIGHTD_MESH_BANDS || ctrl->owner_rank >= SPARK_WEIGHTD_MESH_RANKS_PER_BAND )
        return 0u;
    owner_rank = ctrl->owner_rank;
    index = owner_band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + owner_rank;
    issued = weightd_mesh.issued[index];
    seqs[0] = __atomic_load_n(&ctrl->trigger[0].seq,__ATOMIC_ACQUIRE);
    seqs[1] = __atomic_load_n(&ctrl->trigger[1].seq,__ATOMIC_ACQUIRE);
    if ( seqs[0] <= issued && seqs[1] <= issued )
        return 0u;
    if ( seqs[0] > issued ) order[count++] = 0u;
    if ( seqs[1] > issued ) order[count++] = 1u;
    if ( count == 2u && seqs[order[0]] > seqs[order[1]] )
    {
        uint32_t tmp = order[0];
        order[0] = order[1];
        order[1] = tmp;
    }
    const volatile uint64_t *db0 = SparkWeightdMeshDoorbellAt(index,0u);
    const volatile uint64_t *db1 = SparkWeightdMeshDoorbellAt(index,1u);
    uint64_t cpu0 = __atomic_load_n(&db0[0],__ATOMIC_ACQUIRE);
    uint64_t cpu1 = __atomic_load_n(&db1[0],__ATOMIC_ACQUIRE);
    for (c = 0u; c < count; c++)
    {
        uint64_t cand = seqs[order[c]];
        if ( (cpu0 > issued && cpu0 < cand) || (cpu1 > issued && cpu1 < cand) )
            break;
        if ( SparkWeightdMeshHybridRingOneLocked(ctrl,order[c],cand,now_ns) == 0u )
            break;
        issued = weightd_mesh.issued[index];
        rung++;
    }
    return rung;
}

static uint32_t SparkWeightdMeshHybridPendingTriggerLocked(const SparkWeightdMeshHybridCtrl *ctrl)
{
    uint32_t owner_band = weightd_mesh.hybrid_owner_band,index;
    uint64_t issued,s0,s1,cpu0,cpu1;
    if ( ctrl == 0 || owner_band >= SPARK_WEIGHTD_MESH_BANDS || ctrl->owner_rank >= SPARK_WEIGHTD_MESH_RANKS_PER_BAND )
        return 0u;
    index = owner_band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + ctrl->owner_rank;
    if ( SparkWeightdMeshCellFailed(index) != 0u )
        return 0u;
    issued = weightd_mesh.issued[index];
    s0 = __atomic_load_n(&ctrl->trigger[0].seq,__ATOMIC_ACQUIRE);
    s1 = __atomic_load_n(&ctrl->trigger[1].seq,__ATOMIC_ACQUIRE);
    cpu0 = __atomic_load_n(&SparkWeightdMeshDoorbellAt(index,0u)[0],__ATOMIC_ACQUIRE);
    cpu1 = __atomic_load_n(&SparkWeightdMeshDoorbellAt(index,1u)[0],__ATOMIC_ACQUIRE);
    if ( s0 > issued && !(cpu0 > issued && cpu0 < s0) && !(cpu1 > issued && cpu1 < s0) )
        return 1u;
    if ( s1 > issued && !(cpu0 > issued && cpu0 < s1) && !(cpu1 > issued && cpu1 < s1) )
        return 1u;
    return 0u;
}

static void SparkWeightdMeshHybridRevokeLocked(void)
{
    SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
    uint64_t deadline;
    if ( weightd_mesh.hybrid_active == 0u )
    {
        if ( ctrl != 0 )
            __atomic_store_n(&ctrl->cpu_mode,0u,__ATOMIC_SEQ_CST);
        return;
    }
    if ( ctrl != 0 )
    {
        __atomic_store_n(&ctrl->cpu_mode,0u,__ATOMIC_SEQ_CST);
        SparkWeightdMeshStoreBarrier();
        deadline = SparkMonotonicNs() + SPARK_WEIGHTD_MESH_HYBRID_REVOKE_NS;
        while ( __atomic_load_n(&ctrl->gpu_busy,__ATOMIC_ACQUIRE) != 0u )
        {
            if ( SparkMonotonicNs() >= deadline )
                break;
            SparkWeightdMeshCpuRelax();
        }
        deadline = SparkMonotonicNs() + SPARK_WEIGHTD_MESH_HYBRID_REVOKE_NS;
        while ( SparkWeightdMeshHybridPendingTriggerLocked(ctrl) != 0u )
        {
            uint64_t now = SparkMonotonicNs();
            (void)SparkWeightdMeshDrainCqLocked(now);
            if ( SparkWeightdMeshHybridPollLocked(now) != 0u )
                (void)SparkWeightdMeshDrainCqLocked(SparkMonotonicNs());
            else if ( SparkMonotonicNs() >= deadline )
                break;
            else
                SparkWeightdMeshCpuRelax();
        }
    }
    weightd_mesh.hybrid_active = 0u;
    weightd_mesh.stat_hybrid_revokes++;
}

static void SparkWeightdMeshHybridGrantLocked(uint32_t lane,uint32_t band,uint32_t dbrec_in_shm)
{
    SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
    const SparkWeightdMeshLanePlan *plan = &weightd_mesh.lane_plan[lane];
    uint32_t i,gen;
    if ( ctrl == 0 || plan->valid == 0u )
        return;
    if ( weightd_mesh.hybrid_active != 0u )
        SparkWeightdMeshHybridRevokeLocked();
    memset(ctrl->trigger,0,sizeof(ctrl->trigger));
    ctrl->recv_iova_base = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
    ctrl->staging_iova_base = (uint64_t)(uintptr_t)weightd_mesh.staging_buffer;
    ctrl->owner_lane = lane;
    ctrl->owner_band = band;
    ctrl->owner_rank = plan->local_rank;
    ctrl->degree = plan->degree;
    ctrl->peer_count = plan->peer_count;
    ctrl->all_peers_mask = plan->all_peers_mask;
    ctrl->min_max_inline = plan->min_max_inline;
    ctrl->dbrec_in_shm = dbrec_in_shm;
    ctrl->flags = (dbrec_in_shm != 0u ? SPARK_WEIGHTD_MESH_HYBRID_FLAG_DBREC_IN_SHM : 0u) |
        (weightd_mesh.bf_all_unique != 0u ? SPARK_WEIGHTD_MESH_HYBRID_FLAG_BF_UNIQUE : 0u);
    for (i = 0u; i < plan->peer_count; i++)
    {
        const SparkWeightdMeshPlanPeer *p = &plan->peers[i];
        const SparkWeightdMeshSendQueue *q = &weightd_mesh.send_queues[p->peer];
        SparkWeightdMeshHybridPeer *hp = &ctrl->peers[i];
        hp->sq_shm_off = q->sq_shm_off;
        hp->dbrec_shm_off = q->dbrec_shm_off;
        hp->sq_depth = q->depth;
        hp->sq_mask = q->mask;
        hp->qpn_shifted = p->qpn_shifted;
        hp->rkey_be = p->rkey_be;
        hp->recv_lkey_be = p->recv_lkey_be;
        hp->staging_lkey_be = p->staging_lkey_be;
        hp->remote_base = p->remote_base;
        hp->staging_peer_off = p->staging_peer_off;
        hp->max_inline = q->max_inline;
        hp->peer = p->peer;
        hp->target = p->target;
        hp->flags = 0u;
        hp->pad0 = 0u;
        hp->pad1 = 0u;
        ctrl->gpu_pi[i] = q->produced;
        ctrl->gpu_wrs[i] = q->wr_issued;
        ctrl->cpu_sq_limit[i] = q->retired + q->depth;
        ctrl->cpu_wr_limit[i] = q->wr_retired + SPARK_WEIGHTD_MESH_SEND_CAPACITY;
    }
    gen = weightd_mesh.hybrid_grant_gen + 1u;
    if ( gen == 0u ) gen = 1u;
    weightd_mesh.hybrid_grant_gen = gen;
    weightd_mesh.hybrid_owner_lane = lane;
    weightd_mesh.hybrid_owner_band = band;
    weightd_mesh.hybrid_active = 1u;
    __atomic_store_n(&ctrl->grant_gen,(uint64_t)gen,__ATOMIC_RELEASE);
    __atomic_store_n(&ctrl->cpu_mode,1u,__ATOMIC_SEQ_CST);
}

static void SparkWeightdMeshHybridRefreshLocked(uint32_t preferred_band)
{
    uint32_t lane,active_count = 0u,active_lane = UINT32_MAX,cfg_count = 0u,cfg_lane = UINT32_MAX;
    uint32_t candidate = UINT32_MAX,band,dbrec_in_shm = 0u;
    for (lane = 0u; lane < SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
    {
        uint32_t dbr = 0u,cap = SparkWeightdMeshHybridLaneCapable(lane,&dbr);
        if ( weightd_mesh.hybrid_lane_capable[lane] != (uint8_t)cap )
        {
            weightd_mesh.hybrid_lane_capable[lane] = (uint8_t)cap;
            SparkWeightdMeshAdvertiseLane(lane);
        }
        if ( weightd_mesh.lane_activity[lane] != 0u )
        {
            active_count++;
            active_lane = lane;
        }
        if ( (weightd_mesh.configured_lane_mask & (1u << lane)) != 0u &&
             weightd_mesh.lane_topology[lane].rank_count >= 2u )
        {
            cfg_count++;
            cfg_lane = lane;
        }
    }
    if ( weightd_mesh.mesh_ready == 0u || weightd_mesh.hybrid_shm.base == 0 )
    {
        SparkWeightdMeshHybridRevokeLocked();
        return;
    }
    if ( active_count == 1u )
        candidate = active_lane;
    else if ( active_count == 0u && cfg_count == 1u )
        candidate = cfg_lane;
    if ( candidate == UINT32_MAX || SparkWeightdMeshHybridLaneCapable(candidate,&dbrec_in_shm) == 0u )
    {
        SparkWeightdMeshHybridRevokeLocked();
        return;
    }
    band = preferred_band < SPARK_WEIGHTD_MESH_BANDS && preferred_band / 2u == candidate ?
        preferred_band : (2u * candidate);
    if ( weightd_mesh.hybrid_active != 0u && weightd_mesh.hybrid_owner_lane == candidate &&
         weightd_mesh.hybrid_owner_band == band )
    {
        SparkWeightdMeshHybridUpdateCreditsLocked();
        return;
    }
    SparkWeightdMeshHybridGrantLocked(candidate,band,dbrec_in_shm);
}

static SparkStatus SparkWeightdMeshPostCompactFast(uint32_t band,uint32_t index,uint32_t record,uint64_t seq,uint64_t slot,
    uint64_t bytes,SparkWeightdMeshRoute route,uint64_t now_ns)
{
    const SparkWeightdMeshLanePlan *plan = &weightd_mesh.lane_plan[band / 2u];
    SparkWeightdMeshTransfer *transfer = &weightd_mesh.transfers[record];
    uint32_t peer_rank_mask = (uint32_t)route.fields.peer_mask;
    uint32_t all_peers = peer_rank_mask == plan->all_peers_mask ? 1u : 0u;
    uint32_t xfer_len = (uint32_t)(bytes + 8u);
    uint32_t is_peer = route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER ? 1u : 0u;
    uint32_t is_inline = xfer_len <= plan->min_max_inline ? 1u : 0u;
    uint8_t ds = is_inline != 0u ? (uint8_t)((36u + xfer_len + 15u) >> 4u) : 3u;
    uint32_t num_bb = (uint32_t)((ds + 3u) >> 2u);
    uint32_t inl_cnt_be = htobe32(xfer_len | MLX5_INLINE_SEG);
    uint32_t dma_len_be = htobe32(xfer_len);
    uint64_t slot_base = ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    uint64_t half_off = xfer_len <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ? SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot) : 0u;
    uint64_t staging_band_base = (uint64_t)(uintptr_t)weightd_mesh.staging_buffer +
        (uint64_t)band * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES + half_off;
    uint64_t recv_src = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer + slot_base;
    uint64_t wr_base,pending_mask = 0u;
    uint32_t i,count = 0u;
    uint8_t order[SPARK_WEIGHTD_MESH_PEERS];
    for (i = 0u; i < plan->peer_count; i++)
    {
        uint32_t peer;
        if ( all_peers == 0u && (peer_rank_mask & (1u << plan->peers[i].target)) == 0u )
            continue;
        peer = plan->peers[i].peer;
        if ( weightd_mesh.send_pending[peer] >= SPARK_WEIGHTD_MESH_SEND_CAPACITY ||
             (SparkWeightdMeshSqFree(&weightd_mesh.send_queues[peer]) < num_bb &&
              SparkWeightdMeshSqEnsureRoom(peer,num_bb) != SPARK_STATUS_OK) )
            return SPARK_STATUS_BUSY;
    }
    transfer->seq = seq;
    transfer->generation = ++SparkWeightdMeshNextTransferGeneration;
    transfer->failed = 0u;
    transfer->reset_peer = 0u;
    transfer->chained = 0u;
    weightd_mesh.issued[index] = seq;
    wr_base = SPARK_WEIGHTD_MESH_TRANSFER_ID | (transfer->generation << 16u) | ((uint64_t)record << 6u) | 3u;
    for (i = 0u; i < plan->peer_count; i++)
    {
        const SparkWeightdMeshPlanPeer *p = &plan->peers[i];
        SparkWeightdMeshSendQueue *queue;
        uint32_t peer,sq_idx,w0,w1;
        uint64_t local_addr;
        struct mlx5_wqe_ctrl_seg *control;
        struct mlx5_wqe_raddr_seg *remote;
        if ( all_peers == 0u && (peer_rank_mask & (1u << p->target)) == 0u )
            continue;
        if ( is_peer != 0u && i + 1u < plan->peer_count )
        {
            const uint8_t *next_src = (const uint8_t *)(uintptr_t)(staging_band_base + plan->peers[i + 1u].staging_peer_off);
            __builtin_prefetch(next_src,0,0);
            if ( is_inline != 0u && xfer_len > 64u )
                __builtin_prefetch(next_src + 64u,0,0);
            if ( is_inline != 0u && xfer_len > 128u )
                __builtin_prefetch(next_src + 128u,0,0);
        }
        peer = p->peer;
        queue = &weightd_mesh.send_queues[peer];
        sq_idx = queue->produced & queue->mask;
        control = (struct mlx5_wqe_ctrl_seg *)(queue->buffer + (uint64_t)sq_idx * 64u);
        remote = (struct mlx5_wqe_raddr_seg *)(control + 1);
        w0 = htobe32(((uint32_t)(uint16_t)queue->produced << 8u) | MLX5_OPCODE_RDMA_WRITE);
        w1 = htobe32(p->qpn_shifted | ds);
        control->opmod_idx_opcode = w0;
        control->qpn_ds = w1;
        control->signature = 0u;
        control->dci_stream_channel_id = 0u;
        control->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
        control->imm = 0u;
        remote->raddr = htobe64(p->remote_base + slot_base);
        remote->rkey = p->rkey_be;
        remote->reserved = 0u;
        local_addr = is_peer != 0u ? (staging_band_base + p->staging_peer_off) : recv_src;
        if ( is_inline != 0u )
        {
            struct mlx5_wqe_inl_data_seg *segment = (struct mlx5_wqe_inl_data_seg *)(remote + 1);
            uint32_t sq_bytes = queue->depth * 64u,dst_off = sq_idx * 64u + 36u,tail_room = sq_bytes - dst_off;
            segment->byte_count = inl_cnt_be;
            if ( __builtin_expect(xfer_len <= tail_room,1) )
                memcpy(queue->buffer + dst_off,(const void *)(uintptr_t)local_addr,xfer_len);
            else
            {
                memcpy(queue->buffer + dst_off,(const void *)(uintptr_t)local_addr,tail_room);
                memcpy(queue->buffer,(const uint8_t *)(uintptr_t)local_addr + tail_room,xfer_len - tail_room);
            }
        }
        else
        {
            struct mlx5_wqe_data_seg *data = (struct mlx5_wqe_data_seg *)(remote + 1);
            data->byte_count = dma_len_be;
            data->lkey = is_peer != 0u ? p->staging_lkey_be : p->recv_lkey_be;
            data->addr = htobe64(local_addr);
        }
        queue->wr_id[sq_idx] = wr_base | ((uint64_t)peer << 2u);
        queue->wqe_bbs[sq_idx] = (uint8_t)num_bb;
        queue->ring_word = (uint64_t)w0 | ((uint64_t)w1 << 32u);
        queue->produced += num_bb;
        queue->wr_issued++;
        pending_mask |= UINT64_C(1) << (peer * 4u + 3u);
        weightd_mesh.send_pending[peer]++;
        order[count++] = (uint8_t)peer;
    }
    transfer->pending = pending_mask;
    weightd_mesh.send_pending_total += count;
    SparkWeightdMeshSqRingOrder(order,count);
    if ( SparkWeightdMeshPollingThread == 0u )
        SparkWeightdMeshWake();
    {
        uint64_t spent = SparkMonotonicNs() - now_ns;
        weightd_mesh.timing.route_count++;
        weightd_mesh.timing.route_ns += spent;
        weightd_mesh.timing.route_max_ns = spent > weightd_mesh.timing.route_max_ns ? spent : weightd_mesh.timing.route_max_ns;
    }
    weightd_mesh.post_ns[record] = now_ns;
    if ( weightd_mesh.publish_seq[record] == seq )
        SparkLatencyAdd(&weightd_mesh.timing.post,weightd_mesh.publish_ns[record],now_ns);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdMeshPostSlot(uint32_t band,uint32_t rank,uint64_t seq,uint64_t slot,uint64_t bytes,SparkWeightdMeshRoute route,uint64_t now_ns)
{
    uint32_t index = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank,physical_mask,peer_rank_mask = (uint32_t)route.fields.peer_mask;
    uint32_t record = SPARK_WEIGHTD_MESH_RECORD_OF(index,seq),had_hybrid;
    SparkWeightdMeshTransfer *transfer;
    uint8_t logical[SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    SparkStatus status = SparkWeightdMeshSlotPostable(band,rank,seq,slot,bytes,route);
    if ( status != SPARK_STATUS_OK )
        return status;
    if ( weightd_mesh.mesh_ready == 0u ) return SPARK_STATUS_BUSY;
    if ( SparkWeightdMeshCellFailed(index) != 0u ) SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    if ( seq <= weightd_mesh.issued[index] ) return SPARK_STATUS_OK;
    transfer = &weightd_mesh.transfers[record];
    if ( transfer->pending != 0u ) return SPARK_STATUS_BUSY;
    if ( SparkWeightdMeshNextTransferGeneration == (SPARK_WEIGHTD_MESH_TRANSFER_ID >> 16u) - 1u ) SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    had_hybrid = weightd_mesh.hybrid_active;
    if ( had_hybrid != 0u )
    {
        SparkWeightdMeshHybridRevokeLocked();
        if ( seq <= weightd_mesh.issued[index] )
        {
            SparkWeightdMeshHybridRefreshLocked(band);
            return SPARK_STATUS_OK;
        }
        if ( transfer->pending != 0u )
        {
            SparkWeightdMeshHybridRefreshLocked(band);
            return SPARK_STATUS_BUSY;
        }
    }
    if ( route.fields.compact_tail != 0u && weightd_mesh.lane_plan[band / 2u].valid != 0u )
    {
        status = SparkWeightdMeshPostCompactFast(band,index,record,seq,slot,bytes,route,now_ns);
        if ( had_hybrid != 0u )
            SparkWeightdMeshHybridRefreshLocked(band);
        return status;
    }
    physical_mask = SparkWeightdMeshPhysicalMask(&weightd_mesh.lane_topology[band / 2u],peer_rank_mask,logical);
    if ( SparkWeightdMeshSendRoom(physical_mask,route.fields.pair_first != 0u ? 2u + SPARK_WEIGHTD_MESH_RANKS_PER_BAND / 2u :
             route.fields.compact_tail != 0u ? 1u : 2u,
             route.fields.pair_first != 0u ? 16u + SPARK_WEIGHTD_MESH_RANKS_PER_BAND * 8u : 16u) == 0u )
    {
        if ( had_hybrid != 0u )
            SparkWeightdMeshHybridRefreshLocked(band);
        return SPARK_STATUS_BUSY;
    }
    memset(transfer,0,sizeof(*transfer));
    transfer->seq = seq;
    transfer->generation = ++SparkWeightdMeshNextTransferGeneration;
    weightd_mesh.issued[index] = seq;
    SparkWeightdMeshPostRoute(record,band,slot,((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * SPARK_WEIGHTD_MESH_SLOT_BYTES,bytes,route,physical_mask,logical,rank);
    {
        uint64_t spent = SparkMonotonicNs() - now_ns;
        weightd_mesh.timing.route_count++;
        weightd_mesh.timing.route_ns += spent;
        weightd_mesh.timing.route_max_ns = spent > weightd_mesh.timing.route_max_ns ? spent : weightd_mesh.timing.route_max_ns;
    }
    weightd_mesh.post_ns[record] = now_ns;
    if ( weightd_mesh.publish_seq[record] == seq )
        SparkLatencyAdd(&weightd_mesh.timing.post,weightd_mesh.publish_ns[record],now_ns);
    if ( had_hybrid != 0u )
        SparkWeightdMeshHybridRefreshLocked(band);
    if ( transfer->failed != 0u ) SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    return SPARK_STATUS_OK;
}

static void SparkWeightdMeshPeerCompletion(uint64_t work_id,int success,uint64_t now_ns)
{
    uint32_t peer = (work_id & SPARK_WEIGHTD_MESH_TRANSFER_ID) != 0u ? (uint32_t)(work_id & 63u) / 4u : (uint32_t)(work_id & 0xffu);
    if ( peer >= SPARK_WEIGHTD_MESH_PEERS )
        return;
    if ( success != 0 )
    {
        weightd_mesh.peer_last_ok_ns[peer] = now_ns;
        weightd_mesh.peer_cq_err_since_ok[peer] = 0u;
        return;
    }
    weightd_mesh.peer_last_err_ns[peer] = now_ns;
    if ( weightd_mesh.peer_cq_err_since_ok[peer] != UINT32_MAX )
        weightd_mesh.peer_cq_err_since_ok[peer]++;
}

static uint32_t SparkWeightdMeshDrainCqQueueLocked(uint32_t which,uint64_t now_ns)
{
    SparkWeightdMeshCompletionQueue *queue = &weightd_mesh.completion_queues[which];
    uint32_t repair_needed = 0u,drained = 0u;
    if ( queue->buffer == 0 )
        return 0u;
    for (;;)
    {
        struct mlx5_cqe64 *entry = (struct mlx5_cqe64 *)(queue->buffer + (uint64_t)(queue->consumed & (queue->count - 1u)) * queue->size + queue->size - 64u);
        uint8_t op_own = __atomic_load_n(&entry->op_own,__ATOMIC_ACQUIRE);
        uint8_t opcode = op_own >> 4u;
        uint32_t qpn,peer;
        uint16_t counter;
        uint64_t wr_id = UINT64_MAX;
        int status = IBV_WC_SUCCESS;
        uint32_t vendor_err = 0u;
        if ( opcode == MLX5_CQE_INVALID || (op_own & MLX5_CQE_OWNER_MASK) != ((queue->consumed & queue->count) != 0u ? 1u : 0u) )
            break;
        qpn = be32toh(entry->sop_drop_qpn) & 0xffffffu;
        counter = be16toh(entry->wqe_counter);
        if ( opcode != MLX5_CQE_REQ )
        {
            const struct mlx5_err_cqe *error = (const struct mlx5_err_cqe *)entry;
            status = opcode == MLX5_CQE_REQ_ERR || opcode == MLX5_CQE_RESP_ERR ? SparkWeightdMeshCqeStatus(error->syndrome) : IBV_WC_GENERAL_ERR;
            vendor_err = error->vendor_err_synd;
        }
        peer = SparkWeightdMeshPeerByQpn(which,qpn);
        if ( peer < SPARK_WEIGHTD_MESH_PEERS )
        {
            SparkWeightdMeshSendQueue *send_queue = &weightd_mesh.send_queues[peer];
            uint32_t inflight = send_queue->produced - send_queue->retired;
            uint32_t bbs = send_queue->wqe_bbs[counter & send_queue->mask];
            uint32_t advance = (uint16_t)((uint16_t)(counter + (bbs != 0u ? bbs : 1u)) - (uint16_t)send_queue->retired);
            wr_id = send_queue->wr_id[counter & send_queue->mask];
            send_queue->completed = counter;
            if ( advance != 0u && advance <= inflight )
                send_queue->retired += advance;
        }
        queue->consumed++;
        drained++;
        SparkWeightdMeshCompleteTransfer(wr_id,status == IBV_WC_SUCCESS,now_ns);
        SparkWeightdMeshPeerCompletion(wr_id,status == IBV_WC_SUCCESS,now_ns);
        if ( status == IBV_WC_SUCCESS )
        {
            weightd_mesh.send_ok++;
            weightd_mesh.send_err_window = 0ull;
        }
        else
        {
            weightd_mesh.send_err++;
            weightd_mesh.send_err_window++;
            if ( status != IBV_WC_WR_FLUSH_ERR )
                fprintf(stderr,
                    "WD-MESH-CQERR wr=%llu opcode=%u status=%u vendor=%u\n",
                    (unsigned long long)wr_id,
                    (unsigned)IBV_WC_RDMA_WRITE,
                    (unsigned)status,
                    (unsigned)vendor_err);
            repair_needed = 1u;
        }
    }
    if ( drained != 0u )
    {
        __atomic_store_n(&queue->record[0],htobe32(queue->consumed & 0xffffffu),__ATOMIC_RELEASE);
        SparkWeightdMeshHybridUpdateCreditsLocked();
    }
    return repair_needed;
}

static uint32_t SparkWeightdMeshDrainCqLocked(uint64_t now_ns)
{
    uint32_t repair_needed = SparkWeightdMeshDrainCqQueueLocked(0u,now_ns);
    if ( weightd_mesh.pair_cq != 0 )
        repair_needed |= SparkWeightdMeshDrainCqQueueLocked(1u,now_ns);
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
    return repair_needed;
}

static void SparkWeightdMeshDrainCq(void)
{
    uint32_t repair_needed;
    SparkWeightdMeshWireAcquire();
    repair_needed = SparkWeightdMeshDrainCqLocked(SparkMonotonicNs());
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    if (repair_needed != 0u)
        __atomic_store_n(&weightd_mesh.repair_pending,1u,__ATOMIC_RELEASE);
}

static void SparkWeightdMeshTimingPeers(const SparkWeightdMeshTiming *timing,char *text,size_t capacity)
{
    size_t used = 0u;
    uint32_t peer;
    text[0] = '\0';
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_RANKS && used < capacity; peer++)
        if ( SparkLatencyCount(&timing->lag[peer]) != 0u )
            used += (size_t)snprintf(text + used,capacity - used,"%s%u:%llu/%llu/%llu/%llu/%llu",used != 0u ? "," : "",peer,(unsigned long long)SparkLatencyPercentileUs(&timing->lag[peer],50u),(unsigned long long)SparkLatencyPercentileUs(&timing->lag[peer],99u),(unsigned long long)timing->last[peer],(unsigned long long)timing->start_last[peer],(unsigned long long)(timing->excess_ns[peer] / 1000u));
}

static void SparkWeightdMeshTimingReport(void)
{
    SparkWeightdMeshTiming timing;
    char peers[SPARK_WEIGHTD_MESH_RANKS * SPARK_WEIGHTD_MESH_TIMING_PEER_TEXT];
    SparkWeightdMeshWireAcquire();
    timing = weightd_mesh.timing;
    memset(&weightd_mesh.timing,0,sizeof(weightd_mesh.timing));
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    if ( SparkLatencyCount(&timing.post) == 0u && SparkLatencyCount(&timing.credit) == 0u && SparkLatencyCount(&timing.gate) == 0u )
        return;
    SparkWeightdMeshTimingPeers(&timing,peers,sizeof(peers));
    fprintf(stderr,"WD-MESH-ROUTE routes=%llu route_avg_ns=%llu route_max_ns=%llu ring_barriers=%llu\n",(unsigned long long)timing.route_count,
        (unsigned long long)(timing.route_count != 0u ? timing.route_ns / timing.route_count : 0u),(unsigned long long)timing.route_max_ns,(unsigned long long)timing.ring_barriers);
    fprintf(stderr,"WD-MESH-TIMING posts=%llu post_us=%llu/%llu ship_us=%llu/%llu credits=%llu credit_us=%llu/%llu gates=%llu gate_us=%llu/%llu gate_ms=%llu self=%llu starts=%llu start_us=%llu/%llu start_ms=%llu start_self=%llu worst_us=%llu worst_tag=%llu:%llu worst_closer=%u peers=%s\n",
        (unsigned long long)SparkLatencyCount(&timing.post),
        (unsigned long long)SparkLatencyPercentileUs(&timing.post,50u),(unsigned long long)SparkLatencyPercentileUs(&timing.post,99u),
        (unsigned long long)SparkLatencyPercentileUs(&timing.ship,50u),(unsigned long long)SparkLatencyPercentileUs(&timing.ship,99u),
        (unsigned long long)SparkLatencyCount(&timing.credit),
        (unsigned long long)SparkLatencyPercentileUs(&timing.credit,50u),(unsigned long long)SparkLatencyPercentileUs(&timing.credit,99u),
        (unsigned long long)SparkLatencyCount(&timing.gate),
        (unsigned long long)SparkLatencyPercentileUs(&timing.gate,50u),(unsigned long long)SparkLatencyPercentileUs(&timing.gate,99u),
        (unsigned long long)(timing.gate.total_ns / 1000000u),(unsigned long long)timing.self,
        (unsigned long long)SparkLatencyCount(&timing.start),
        (unsigned long long)SparkLatencyPercentileUs(&timing.start,50u),(unsigned long long)SparkLatencyPercentileUs(&timing.start,99u),
        (unsigned long long)(timing.start.total_ns / 1000000u),(unsigned long long)timing.start_self,
        (unsigned long long)(timing.worst_ns / 1000u),(unsigned long long)(timing.worst_tag >> 32u),(unsigned long long)(uint32_t)timing.worst_tag,
        timing.worst_closer,peers);
}

static void SparkWeightdMeshRepairAndReport(void)
{
    uint64_t now = SparkMonotonicNs();
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
    fprintf(stderr,"WD-MESH-STATS trywire=%llu record_failures=%llu record_invalid=%llu wire_failures=%llu rewires=%llu unready=%llu repairs=%llu cq_ok=%llu cq_err=%llu ready=%u peer_resets=%llu peer_reset_bits=%llu lane_resets=%llu resyncs=%llu wire_generation=%llu degraded=0x%04x sq_full_waits=%llu hybrid_rings=%llu hybrid_revokes=%llu hybrid_active=%u\n",
        (unsigned long long)weightd_mesh.stat_trywire,(unsigned long long)weightd_mesh.stat_record_failures,
        (unsigned long long)weightd_mesh.stat_record_invalid,(unsigned long long)weightd_mesh.stat_wire_failures,
        (unsigned long long)weightd_mesh.stat_rewires,(unsigned long long)weightd_mesh.stat_unready,
        (unsigned long long)weightd_mesh.stat_repairs,(unsigned long long)weightd_mesh.send_ok,
        (unsigned long long)weightd_mesh.send_err,weightd_mesh.mesh_ready,
        (unsigned long long)weightd_mesh.stat_peer_resets,(unsigned long long)weightd_mesh.stat_peer_reset_bits,
        (unsigned long long)weightd_mesh.stat_lane_resets,(unsigned long long)weightd_mesh.stat_resyncs,
        (unsigned long long)weightd_mesh.wire_generation,SparkWeightdMeshDegradedMask(),
        (unsigned long long)weightd_mesh.stat_sq_full_waits,
        (unsigned long long)weightd_mesh.stat_hybrid_rings,
        (unsigned long long)weightd_mesh.stat_hybrid_revokes,
        weightd_mesh.hybrid_active);
    SparkWeightdMeshTimingReport();
}

void SparkWeightdMeshPoll(void)
{
    if (weightd_mesh.mesh_active == 0u) return;
    if (weightd_mesh.mesh_ready == 0u) SparkWeightdMeshTryWire();
    else if (SparkWeightdMeshRealtimeNs() - weightd_mesh.record_check_ns >= 1000000000ull)
    {
        weightd_mesh.record_check_ns = SparkWeightdMeshRealtimeNs();
        SparkWeightdMeshTryWire();
    }
    if (weightd_mesh.mesh_ready == 0u) return;
    if (SparkWeightdMeshRealtimeNs() - weightd_mesh.artifact_check_ns >= 1000000000ull)
    {
        char artifact_path[256];
        SparkWeightdMeshRecord own_record,existing;
        int artifact_fd;
        uint32_t stale = 1u;
        weightd_mesh.artifact_check_ns = SparkWeightdMeshRealtimeNs();
        if (SparkWeightdMeshReadyMarkerCurrent() == 0u) SparkWeightdMeshReadyMarkerWrite();
        (void)snprintf(artifact_path,sizeof(artifact_path),"%s/mesh-%x.rec",weightd_mesh_dir,weightd_mesh.local_rank);
        artifact_fd = open(artifact_path,O_RDONLY);
        if (artifact_fd >= 0)
        {
            memset(&existing,0,sizeof(existing));
            if (read(artifact_fd,&existing,sizeof(existing)) == (ssize_t)sizeof(existing) &&
                existing.magic == SPARK_WEIGHTD_MESH_MAGIC && existing.boot_ns == weightd_mesh.boot_ns &&
                existing.wire_generation == weightd_mesh.wire_generation &&
                existing.recv_addr == (uint64_t)(uintptr_t)weightd_mesh.recv_buffer)
                stale = 0u;
            close(artifact_fd);
        }
        if (stale != 0u)
        {
            SparkWeightdMeshFillRecord(&own_record);
            (void)SparkWeightdMeshWriteRecord(&own_record);
        }
    }
    if ( __atomic_load_n(&SparkWeightdMeshDoorbellRunning,__ATOMIC_ACQUIRE) == 0u ||
         __atomic_load_n(&weightd_mesh.activity_owners,__ATOMIC_RELAXED) == 0u )
        SparkWeightdMeshDrainCq();
    SparkWeightdMeshRepairAndReport();
    if (weightd_mesh.send_ok - weightd_mesh.send_logged >= 2048ull)
    {
        weightd_mesh.send_logged = weightd_mesh.send_ok;
        fprintf(stderr,"WD-MESH-CQ ok=%llu err=%llu\n",(unsigned long long)weightd_mesh.send_ok,(unsigned long long)weightd_mesh.send_err);
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

int SparkWeightdMeshStagingFd(void)
{
    return weightd_mesh.mesh_ready != 0u && weightd_mesh.staging_mr != 0 ? dup(weightd_mesh.staging_memfd) : -1;
}
static uint32_t SparkWeightdMeshDoorbellStable(uint32_t index,uint64_t snapshot[2][4])
{
    uint32_t tries,alternate,word,agree;
    for ( tries = 0u; tries < 64u; tries++ )
    {
        agree = 1u;
        for (alternate=0u; alternate<2u; alternate++)
        {
            volatile uint64_t *cell = SparkWeightdMeshDoorbellAt(index,alternate);
            snapshot[alternate][0] = __atomic_load_n(&cell[0],__ATOMIC_ACQUIRE);
            for (word=1u; word<4u; word++)
                snapshot[alternate][word] = cell[word];
        }
        SparkWeightdMeshLoadBarrier();
        for (alternate=0u; alternate<2u; alternate++)
        {
            volatile uint64_t *cell = SparkWeightdMeshDoorbellAt(index,alternate);
            agree &= snapshot[alternate][0] == __atomic_load_n(&cell[0],__ATOMIC_ACQUIRE) ? 1u : 0u;
            for (word=1u; word<4u; word++)
                agree &= snapshot[alternate][word] == cell[word] ? 1u : 0u;
        }
        if ( agree != 0u )
            return(1u);
    }
    return(0u);
}

static uint32_t SparkWeightdMeshDoorbellStuck(uint64_t index,const char *phase,uint64_t seq,uint64_t bytes,uint64_t slot,uint64_t now_ns)
{
    weightd_mesh.doorbell_stuck[index]++;
    if ( seq != weightd_mesh.doorbell_seen_seq[index] )
    {
        weightd_mesh.doorbell_seen_seq[index] = seq;
        weightd_mesh.doorbell_seen_ns[index] = now_ns;
        return(0u);
    }
    if ( now_ns - weightd_mesh.doorbell_seen_ns[index] < SPARK_WEIGHTD_MESH_STUCK_LOG_NS ||
         now_ns - weightd_mesh.doorbell_logged_ns[index] < SPARK_WEIGHTD_MESH_STUCK_LOG_NS )
        return(0u);
    weightd_mesh.doorbell_logged_ns[index] = now_ns;
    fprintf(stderr,"WD-STUCK idx=%llu phase=%s seq=%llu posted=%llu issued=%llu bytes=%llu slot=%llu for_ms=%llu sweeps=%u\n",(unsigned long long)index,phase,
        (unsigned long long)seq,(unsigned long long)weightd_mesh.doorbell_posted[index],(unsigned long long)weightd_mesh.issued[index],(unsigned long long)bytes,
        (unsigned long long)slot,(unsigned long long)((now_ns - weightd_mesh.doorbell_seen_ns[index]) / 1000000u),weightd_mesh.doorbell_stuck[index]);
    return(1u);
}

static void SparkWeightdMeshTransferStuck(uint32_t index,uint64_t seq)
{
    const SparkWeightdMeshTransfer *transfer = &weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RECORD_OF(index,seq)];
    uint32_t peer;
    fprintf(stderr,"WD-TRANSFER-STUCK idx=%u seq=%llu record_seq=%llu pending=%llx chained=%x failed=%u\n",index,(unsigned long long)seq,
        (unsigned long long)transfer->seq,(unsigned long long)transfer->pending,transfer->chained,transfer->failed);
    for (peer=0u; peer<SPARK_WEIGHTD_MESH_PEERS; peer++)
        if ( (transfer->pending & (UINT64_C(0xf) << (peer * 4u))) != 0u )
            fprintf(stderr,"WD-TRANSFER-PEER idx=%u peer=%u rank=%u qpn=%u produced=%u retired=%u completed=%u send_pending=%u chained_writes=%u\n",index,peer,
                peer < weightd_mesh.local_rank ? peer : peer + 1u,weightd_mesh.send_queues[peer].qpn,weightd_mesh.send_queues[peer].produced & 0xffffu,
                weightd_mesh.send_queues[peer].retired & 0xffffu,weightd_mesh.send_queues[peer].completed,weightd_mesh.send_pending[peer],transfer->chained_writes[peer]);
}

static uint32_t SparkWeightdMeshDoorbellCell(uint32_t band,uint32_t rank,uint64_t now_ns)
{
    uint32_t index = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank,alternate,candidate,count = 0u,order[2],posted_any = 0u;
    uint64_t snapshot[2][4],newest = 0u,posted = weightd_mesh.doorbell_posted[index],issued = weightd_mesh.issued[index];
    uint64_t s0 = __atomic_load_n(&SparkWeightdMeshDoorbellAt(index,0u)[0],__ATOMIC_ACQUIRE);
    uint64_t s1 = __atomic_load_n(&SparkWeightdMeshDoorbellAt(index,1u)[0],__ATOMIC_ACQUIRE);
    if ( s0 <= posted && s1 <= posted )
    {
        weightd_mesh.doorbell_stuck[index] = 0u;
        return(0u);
    }
    if ( s0 <= issued && s1 <= issued )
    {
        newest = s0 > s1 ? s0 : s1;
        if ( SparkWeightdMeshDoorbellStuck(index,"pending",newest,0u,0u,now_ns) != 0u )
            SparkWeightdMeshTransferStuck(index,newest);
        return(0u);
    }
    if ( SparkWeightdMeshDoorbellStable(index,snapshot) == 0u )
    {
        (void)SparkWeightdMeshDoorbellStuck(index,"unstable",0u,0u,0u,now_ns);
        return(weightd_mesh.doorbell_stuck[index] < SPARK_WEIGHTD_MESH_STUCK_LIVE_SWEEPS ? 1u : 0u);
    }
    for (alternate=0u; alternate<2u; alternate++)
        if ( snapshot[alternate][0] > posted && snapshot[alternate][1] != 0u )
        {
            newest = snapshot[alternate][0] > newest ? snapshot[alternate][0] : newest;
            if ( snapshot[alternate][0] > issued )
                order[count++] = alternate;
        }
    if ( newest == 0u )
        return(0u);
    if ( count == 2u && snapshot[order[0]][0] > snapshot[order[1]][0] )
    {
        candidate = order[0];
        order[0] = order[1];
        order[1] = candidate;
    }
    if ( SparkWeightdMeshDoorbellStuck(index,"pending",newest,0u,0u,now_ns) != 0u )
        SparkWeightdMeshTransferStuck(index,newest);
    for (candidate=0u; candidate<count; candidate++)
    {
        const uint64_t *cell = snapshot[order[candidate]];
        uint32_t record = SPARK_WEIGHTD_MESH_RECORD_OF(index,cell[0]);
        SparkWeightdMeshRoute route;
        route.word = cell[3];
        if ( cell[2] >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND || SparkWeightdMeshRouteFits(route,cell[1]) == 0u )
        {
            (void)SparkWeightdMeshDoorbellStuck(index,"invalid",cell[0],cell[1],cell[2],now_ns);
            break;
        }
        if ( route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER && weightd_mesh.lane_plan[band / 2u].valid != 0u )
        {
            uint64_t half_off = cell[1] + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ?
                SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(cell[2]) : 0u;
            __builtin_prefetch((const uint8_t *)weightd_mesh.staging_buffer +
                (uint64_t)band * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES +
                weightd_mesh.lane_plan[band / 2u].peers[0].staging_peer_off + half_off,0,0);
        }
        else
            __builtin_prefetch((const uint8_t *)weightd_mesh.recv_buffer +
                ((uint64_t)band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + cell[2]) * SPARK_WEIGHTD_MESH_SLOT_BYTES +
                (route.fields.compact_tail != 0u ? cell[1] : SPARK_WEIGHTD_MESH_TAIL_OFFSET(route.fields.half)),0,0);
        if ( weightd_mesh.publish_seq[record] != cell[0] )
        {
            weightd_mesh.publish_seq[record] = cell[0];
            weightd_mesh.publish_ns[record] = now_ns;
        }
        if ( SparkWeightdMeshPostSlot(band,rank,cell[0],cell[2],cell[1],route,now_ns) != SPARK_STATUS_OK )
            break;
        posted_any = 1u;
    }
    return(posted_any);
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
    uint32_t lane, band, rank, cell, work = 0u, mask = weightd_mesh.configured_lane_mask;
    const uint64_t scan_interval = weightd_mesh.activity_owners != 0u ?
        SPARK_WEIGHTD_MESH_ACTIVE_FULL_SCAN_NS : SPARK_WEIGHTD_MESH_FULL_SCAN_NS;
    if ( weightd_mesh.activity_owners == 0u && now_ns - weightd_mesh.full_scan_ns >= scan_interval )
        return(SparkWeightdMeshDoorbellFullScan(now_ns));
    while ( mask != 0u )
    {
        lane = (uint32_t)__builtin_ctz(mask);
        mask &= mask - 1u;
        if ( weightd_mesh.lane_topology[lane].rank_count == 0u )
            continue;
        rank = weightd_mesh.lane_topology[lane].local_rank;
        for ( band = 2u * lane; band < 2u * lane + 2u; band++ )
        {
            uint32_t posted;
            cell = band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank;
            work += SparkWeightdMeshWaitRequestsPollRange(now_ns,cell,cell + 1u);
            posted = SparkWeightdMeshDoorbellCell(band,rank,now_ns);
            work += posted;
            if ( posted != 0u )
                work += SparkWeightdMeshWaitRequestsPollRange(now_ns,cell,cell + 1u);
        }
    }
    if ( work == 0u && now_ns - weightd_mesh.full_scan_ns >= scan_interval )
        return(SparkWeightdMeshDoorbellFullScan(now_ns));
    return(work);
}

static uint32_t SparkWeightdMeshDoorbellPollLocked(uint64_t now_ns,uint32_t *repair_out)
{
    uint32_t work = 0u,quiescing;
    quiescing = weightd_mesh.quiesce_until_ns != 0u && SparkWeightdMeshRealtimeNs() < weightd_mesh.quiesce_until_ns ? 1u : 0u;
    if ( weightd_mesh.hybrid_active != 0u && weightd_mesh.mesh_ready != 0u && quiescing == 0u )
        work = SparkWeightdMeshHybridPollLocked(now_ns);
    *repair_out = SparkWeightdMeshDrainCqLocked(now_ns);
    if ( weightd_mesh.mesh_ready == 0u || quiescing != 0u )
        work += SparkWeightdMeshWaitRequestsPoll(now_ns);
    else if ( work != 0u )
    {
        SparkWeightdMeshHybridCtrl *ctrl = SparkWeightdMeshHybridCtrlPtr();
        if ( ctrl != 0 && weightd_mesh.hybrid_owner_band < SPARK_WEIGHTD_MESH_BANDS &&
             ctrl->owner_rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND )
        {
            uint32_t cell = weightd_mesh.hybrid_owner_band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + ctrl->owner_rank;
            work += SparkWeightdMeshWaitRequestsPollRange(now_ns,cell,cell + 1u);
        }
    }
    else
        work = SparkWeightdMeshDoorbellSweep(now_ns);
    if ( work != 0u || SparkWeightdMeshHasPending() != 0u )
        weightd_mesh.work_ns = now_ns;
    return work;
}

static void SparkWeightdMeshDoorbellPoll(void)
{
    uint32_t repair_needed;
    SparkWeightdMeshWireAcquire();
    (void)SparkWeightdMeshDoorbellPollLocked(SparkMonotonicNs(),&repair_needed);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    if ( repair_needed != 0u )
        __atomic_store_n(&weightd_mesh.repair_pending,1u,__ATOMIC_RELEASE);
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

static void SparkWeightdMeshPinFastCores(int32_t doorbell_cpu)
{
    uint32_t cpu, cpus, highest = 0u, lowest = UINT32_MAX, capacity, count = 0u;
    long configured = sysconf(_SC_NPROCESSORS_CONF);
    cpu_set_t set;
    cpus = configured > 0 && configured < CPU_SETSIZE ? (uint32_t)configured : 0u;
    if ( doorbell_cpu >= 0 )
    {
        CPU_ZERO(&set);
        CPU_SET((uint32_t)doorbell_cpu,&set);
        fprintf(stderr,"WD-MESH-PIN doorbell thread cpu=%d capacity=%u rc=%d\n",doorbell_cpu,SparkWeightdMeshCpuCapacity((uint32_t)doorbell_cpu),pthread_setaffinity_np(pthread_self(),sizeof(set),&set));
        return;
    }
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

void SparkWeightdMeshDoorbellLoop(int32_t doorbell_cpu)
{
    SparkWeightdMeshPollingThread = 1u;
    __atomic_store_n(&SparkWeightdMeshDoorbellRunning,1u,__ATOMIC_RELEASE);
    SparkWeightdMeshPinFastCores(doorbell_cpu);
    for (;;)
    {
        uint64_t now_ns;
        uint32_t empty = 0u,repair_needed = 0u;
        SparkWeightdMeshWaitForActivity();
        if ( __atomic_load_n(&SparkWeightdMeshWireWaiters,__ATOMIC_ACQUIRE) != 0u )
        {
            SparkWeightdMeshDoorbellPoll();
            SparkWeightdMeshCpuRelax();
            continue;
        }
        SparkWeightdMeshWireAcquire();
        now_ns = SparkMonotonicNs();
        while ( __atomic_load_n(&SparkWeightdMeshWireWaiters,__ATOMIC_ACQUIRE) == 0u )
        {
            uint32_t repair = 0u,work = SparkWeightdMeshDoorbellPollLocked(now_ns,&repair);
            repair_needed |= repair;
            if ( work != 0u )
            {
                now_ns = SparkMonotonicNs();
                weightd_mesh.work_ns = now_ns;
                empty = 0u;
                continue;
            }
            if ( weightd_mesh.mesh_ready == 0u || weightd_mesh.activity_owners == 0u )
                break;
            if ( (++empty & 63u) == 0u )
            {
                now_ns = SparkMonotonicNs();
                if ( now_ns - weightd_mesh.work_ns >= SPARK_WEIGHTD_MESH_DORMANT_AFTER_NS )
                    break;
            }
            SparkWeightdMeshCpuRelax();
        }
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        if ( repair_needed != 0u )
            __atomic_store_n(&weightd_mesh.repair_pending,1u,__ATOMIC_RELEASE);
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
    uint32_t rank,step,degree = SparkWeightdMeshRankCount(),posted = 0u,had_hybrid;
    uint8_t order[SPARK_WEIGHTD_MESH_PEERS];

    SparkWeightdMeshWireAcquire();
    if (weightd_mesh.mesh_ready == 0u || peer_rank_mask == 0u ||
        (peer_rank_mask & ~weightd_mesh.rank_mask) != 0u ||
        (peer_rank_mask & (1u << weightd_mesh.local_rank)) != 0u)
    {
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        return 0u;
    }
    had_hybrid = weightd_mesh.hybrid_active;
    if ( had_hybrid != 0u )
        SparkWeightdMeshHybridRevokeLocked();
    for ( rank = 0u; rank < degree; rank++ )
    {
        int32_t peer = SparkWeightdMeshPeerIndexFromRank(rank);
        uint32_t needed = seq_remote_offset != 0ull ? 2u : 1u;
        if ( peer >= 0 && (peer_rank_mask & (1u << rank)) != 0u &&
             (weightd_mesh.wired_boot_ns[peer] == 0u ||
              weightd_mesh.send_pending[peer] > SPARK_WEIGHTD_MESH_SEND_CAPACITY - needed ||
              SparkWeightdMeshSqEnsureRoom((uint32_t)peer,16u) != SPARK_STATUS_OK) )
        {
            if ( had_hybrid != 0u )
                SparkWeightdMeshHybridRefreshLocked(weightd_mesh.hybrid_owner_band);
            pthread_mutex_unlock(&SparkWeightdMeshWireLock);
            return 0u;
        }
    }
    for (step = 1u; step < degree; step++)
    {
        int32_t peer;
        uint64_t wr_id;
        rank = SparkWeightdMeshSchedulePeer(weightd_mesh.local_rank,degree,step);
        if ((peer_rank_mask & (1u << rank)) == 0u) continue;
        peer = SparkWeightdMeshPeerIndexFromRank(rank);
        if (peer < 0) continue;
        wr_id = ((uint64_t)weightd_mesh.wire_epoch[peer] << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)(uint32_t)peer;
        {
            const uint32_t chain_seq = seq_remote_offset != 0ull && weightd_mesh.seq_mr != 0 ? 1u : 0u;
            if ( SparkWeightdMeshSqWrite((uint32_t)peer,wr_id,chain_seq != 0u ? 0u : 1u,(uint64_t)(uintptr_t)weightd_mesh.recv_buffer + source_offset,
                    SparkWeightdMeshPeerMr((uint32_t)peer,weightd_mesh.recv_mr,weightd_mesh.pair_recv_mr)->lkey,length,remote_offset,0) != SPARK_STATUS_OK )
                continue;
            weightd_mesh.send_pending[peer]++;
            weightd_mesh.send_queues[peer].wr_issued++;
            weightd_mesh.send_pending_total++;
            weightd_mesh.rpc_pending[peer]++;
            if ( chain_seq != 0u &&
                 SparkWeightdMeshSqWrite((uint32_t)peer,wr_id | SPARK_WEIGHTD_MESH_RPC_CHAINED,1u,0u,0u,0u,seq_remote_offset,&seq_value) == SPARK_STATUS_OK )
            {
                weightd_mesh.send_pending[peer]++;
                weightd_mesh.send_queues[peer].wr_issued++;
                weightd_mesh.send_pending_total++;
                weightd_mesh.rpc_pending[peer]++;
            }
        }
        order[posted++] = (uint8_t)peer;
    }
    SparkWeightdMeshSqRingOrder(order,posted);
    if ( had_hybrid != 0u )
        SparkWeightdMeshHybridRefreshLocked(weightd_mesh.hybrid_owner_band);
    SparkWeightdMeshWake();
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
    int32_t peer;
    uint32_t had_hybrid;
    SparkStatus status;

    if (weightd_mesh.mesh_ready == 0u)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    peer = SparkWeightdMeshPeerIndexFromRank(peer_rank);
    if (peer < 0)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    if (SparkWeightdMeshIsPairPeer((uint32_t)peer) != 0u)
    {
        if (lkey == weightd_mesh.recv_mr->lkey)
            lkey = weightd_mesh.pair_recv_mr->lkey;
        else if (lkey == weightd_mesh.staging_mr->lkey)
            lkey = weightd_mesh.pair_staging_mr->lkey;
        else
            SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    SparkWeightdMeshWireAcquire();
    had_hybrid = weightd_mesh.hybrid_active;
    if ( had_hybrid != 0u )
        SparkWeightdMeshHybridRevokeLocked();
    if ( weightd_mesh.mesh_ready == 0u || weightd_mesh.wired_boot_ns[peer] == 0u ||
         weightd_mesh.send_pending[peer] >= SPARK_WEIGHTD_MESH_SEND_CAPACITY )
    {
        if ( had_hybrid != 0u )
            SparkWeightdMeshHybridRefreshLocked(weightd_mesh.hybrid_owner_band);
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        return SPARK_STATUS_BUSY;
    }
    status = SparkWeightdMeshSqWrite((uint32_t)peer,((uint64_t)weightd_mesh.wire_epoch[peer] << SPARK_WEIGHTD_MESH_RPC_EPOCH_SHIFT) | (uint64_t)(uint32_t)peer,
        1u,local_addr,lkey,length,remote_offset,0);
    if ( status != SPARK_STATUS_OK )
    {
        if ( had_hybrid != 0u )
            SparkWeightdMeshHybridRefreshLocked(weightd_mesh.hybrid_owner_band);
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        return status;
    }
    SparkWeightdMeshSqRing(1u << (uint32_t)peer);
    weightd_mesh.send_pending[peer]++;
    weightd_mesh.send_queues[peer].wr_issued++;
    weightd_mesh.send_pending_total++;
    SparkWeightdMeshWake();
    weightd_mesh.rpc_pending[peer]++;
    if ( had_hybrid != 0u )
        SparkWeightdMeshHybridRefreshLocked(weightd_mesh.hybrid_owner_band);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return SPARK_STATUS_OK;
}
