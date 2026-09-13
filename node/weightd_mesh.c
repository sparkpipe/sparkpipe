#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include <infiniband/verbs.h>
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

#define SPARK_WEIGHTD_MESH_PEERS 15
#define SPARK_WEIGHTD_MESH_CQ_ENTRIES 16384u
#define SPARK_WEIGHTD_MESH_MAGIC UINT64_C(0x4d45534830303031)
#define SPARK_WEIGHTD_MESH_DIR "/tmp/weightd-mesh"

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
    uint8_t reserved[4];
    uint64_t boot_ns;
} SparkWeightdMeshRecord;

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
    uint64_t wired_boot_ns[SPARK_WEIGHTD_MESH_PEERS];
    uint64_t send_ok;
    uint64_t send_err;
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
    uint32_t mesh_active;
    uint32_t mesh_ready;
    uint32_t local_rank;
} SparkWeightdMesh;

static SparkWeightdMesh weightd_mesh;

static uint64_t SparkWeightdMeshRealtimeNs(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME,&now) != 0)
        return 0ull;
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static uint32_t SparkWeightdMeshRankFromHost(void)
{
    char hostname[64];
    char tail;
    if (gethostname(hostname,sizeof(hostname)) != 0)
        return 0;
    tail = hostname[strlen(hostname) - 1];
    if (tail >= '0' && tail <= '9')
        return (uint32_t)(tail - '0');
    if (tail >= 'a' && tail <= 'f')
        return (uint32_t)(tail - 'a' + 10);
    return 0;
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

    (void)mkdir(SPARK_WEIGHTD_MESH_DIR,0755);
    snprintf(path,sizeof(path),"%s/mesh-%x.rec",
        SPARK_WEIGHTD_MESH_DIR,weightd_mesh.local_rank);
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
        SPARK_WEIGHTD_MESH_DIR,peer_rank);
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
        return SPARK_STATUS_BUSY;
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
    attributes.ah_attr.grh.sgid_index = 3;
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

static void SparkWeightdMeshTryWireLocked(void)
{
    SparkWeightdMeshRecord peer_records[SPARK_WEIGHTD_MESH_PEERS];
    uint32_t peer;
    uint32_t peer_rank;
    uint32_t my_index_in_peer;
    uint32_t changed;
    uint32_t wired;

    changed = weightd_mesh.mesh_ready == 0u;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
        if (SparkWeightdMeshReadPeerRecord(peer_rank,
                &peer_records[peer]) != SPARK_STATUS_OK)
        {
            peer_records[peer].boot_ns =
                weightd_mesh.wired_boot_ns[peer];
            continue;
        }
        if (peer_records[peer].boot_ns != weightd_mesh.wired_boot_ns[peer])
            changed = 1u;
    }
    if (changed == 0u)
        return;
    wired = 1u;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        if (peer_records[peer].boot_ns == weightd_mesh.wired_boot_ns[peer])
            continue;
        peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
        my_index_in_peer = weightd_mesh.local_rank < peer_rank ?
            weightd_mesh.local_rank : weightd_mesh.local_rank - 1u;
        if (SparkWeightdMeshTransitionQp(
                weightd_mesh.send_qps[peer],
                peer_records[peer].recv_qpn[my_index_in_peer],
                peer_records[peer].lid,
                peer_records[peer].gid) != SPARK_STATUS_OK ||
            SparkWeightdMeshTransitionQp(
                weightd_mesh.recv_qps[peer],
                peer_records[peer].send_qpn[my_index_in_peer],
                peer_records[peer].lid,
                peer_records[peer].gid) != SPARK_STATUS_OK)
        {
            wired = 0u;
            fprintf(stderr,"WD-WIRE-FAIL rank=%u peer=%u errno=%d target_qpn=%u\n",
                weightd_mesh.local_rank,peer,errno,
                peer_records[peer].recv_qpn[my_index_in_peer]);
            continue;
        }
        weightd_mesh.qp_info[peer].remote_qpn =
            peer_records[peer].recv_qpn[my_index_in_peer];
        weightd_mesh.qp_info[peer].rkey = peer_records[peer].rkey;
        weightd_mesh.qp_info[peer].remote_addr =
            peer_records[peer].recv_addr;
        weightd_mesh.wired_boot_ns[peer] = peer_records[peer].boot_ns;
        fprintf(stderr,"WD-WIRED rank=%u peer=%u addr=%llx rkey=%u boot=%llu\n",
            weightd_mesh.local_rank,peer,
            (unsigned long long)peer_records[peer].recv_addr,
            peer_records[peer].rkey,
            (unsigned long long)peer_records[peer].boot_ns);
    }
    if (wired == 0u)
        return;
    if (weightd_mesh.mesh_ready == 0u)
    {
        FILE *marker = fopen(SPARK_WEIGHTD_MESH_DIR "/.ready","w");
        if (marker != 0)
            (void)fclose(marker);
    }
    weightd_mesh.mesh_ready = 1u;
    printf("weightd-mesh: ready rank=%u peers=%u rkey=%u\n",
        weightd_mesh.local_rank,SPARK_WEIGHTD_MESH_PEERS,
        weightd_mesh.recv_mr->rkey);
    fflush(stdout);
}

static void SparkWeightdMeshTryWire(void)
{
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    SparkWeightdMeshTryWireLocked();
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
}

SparkStatus SparkWeightdMeshInit(void)
{
    struct ibv_device **devices;
    struct ibv_port_attr port_attr;
    struct ibv_qp_init_attr qp_attributes;
    SparkWeightdMeshRecord own_record;
    int device_count;
    uint32_t peer;

    memset(&weightd_mesh,0,sizeof(weightd_mesh));
    weightd_mesh.local_rank = SparkWeightdMeshRankFromHost();
    weightd_mesh.boot_ns = SparkWeightdMeshRealtimeNs();
    (void)unlink(SPARK_WEIGHTD_MESH_DIR "/.ready");

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
            if (name != 0 && strcmp(name,"rocep1s0f1") == 0)
            {
                weightd_mesh.context = ibv_open_device(devices[device_index]);
                found = 1;
                break;
            }
        }
        if (found == 0)
        {
            fprintf(stderr,"weightd-mesh: rocep1s0f1 (switch) not found\n");
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
    qp_attributes.cap.max_send_wr = 64;
    qp_attributes.cap.max_recv_wr = 64;
    qp_attributes.cap.max_send_sge = 1;
    qp_attributes.cap.max_recv_sge = 1;
    qp_attributes.cap.max_inline_data = 64;
    qp_attributes.qp_type = IBV_QPT_RC;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
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
    own_record.boot_ns = weightd_mesh.boot_ns;
    own_record.rkey = weightd_mesh.recv_mr->rkey;
    own_record.recv_addr = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
    own_record.lid = (uint16_t)port_attr.lid;
    {
        union ibv_gid gid;
        if (ibv_query_gid(weightd_mesh.context,1,3,&gid) != 0)
        {
            fprintf(stderr,"weightd-mesh: gid query failed\n");
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        memcpy(own_record.gid,gid.raw,16);
    }
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        own_record.send_qpn[peer] = weightd_mesh.send_qps[peer]->qp_num;
        own_record.recv_qpn[peer] = weightd_mesh.recv_qps[peer]->qp_num;
    }
    printf("weightd-mesh: rank=%u publishing %u QPs\n",
        weightd_mesh.local_rank,SPARK_WEIGHTD_MESH_PEERS * 2u);
    fflush(stdout);
    if (SparkWeightdMeshWriteRecord(&own_record) != SPARK_STATUS_OK)
        return SPARK_STATUS_IO_ERROR;
    printf("weightd-mesh: published; wiring continues as peers appear\n");
    fflush(stdout);
    weightd_mesh.mesh_active = 1u;
    return SPARK_STATUS_BUSY;
}

uint32_t SparkWeightdMeshReady(void)
{
    return weightd_mesh.mesh_ready;
}

static void SparkWeightdMeshDrainCq(void)
{
    struct ibv_wc completions[64];
    int completed;
    int index;
    uint32_t repair_needed = 0u;

    for (;;)
    {
        completed = ibv_poll_cq(weightd_mesh.cq,64,completions);
        if (completed <= 0)
            break;
        for (index = 0; index < completed; index++)
        {
            if (completions[index].status == IBV_WC_SUCCESS)
                weightd_mesh.send_ok++;
            else if (completions[index].status == IBV_WC_WR_FLUSH_ERR)
            {
                weightd_mesh.send_err++;
                repair_needed = 1u;
            }
            else
            {
                weightd_mesh.send_err++;
                fprintf(stderr,
                    "WD-MESH-CQERR wr=%llu opcode=%u status=%u vendor=%u\n",
                    (unsigned long long)completions[index].wr_id,
                    (unsigned)completions[index].opcode,
                    (unsigned)completions[index].status,
                    (unsigned)completions[index].vendor_err);
            }
        }
    }
    if (repair_needed != 0u)
        SparkWeightdMeshTryWire();
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
    SparkWeightdMeshDrainCq();
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



void SparkWeightdMeshDoorbellLoop(void)
{
    volatile uint64_t *entries;
    uint32_t band;
    uint32_t rank;
    uint32_t peer;

    while (weightd_mesh.mesh_ready == 0u)
    {
        struct timespec pause = {0,1000000};
        nanosleep(&pause,0);
    }
    entries = (volatile uint64_t *)
        ((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_OFFSET);
    {
        SparkWeightdMeshQpInfo qp_snapshot[SPARK_WEIGHTD_MESH_PEERS];
    for (;;)
    {
        struct timespec pause = {0,2000};
        SparkWeightdMeshDrainCq();
        pthread_mutex_lock(&SparkWeightdMeshWireLock);
        memcpy(qp_snapshot,weightd_mesh.qp_info,sizeof(qp_snapshot));
        pthread_mutex_unlock(&SparkWeightdMeshWireLock);
        for (band = 0u; band < SPARK_WEIGHTD_MESH_BANDS; band++)
        {
            for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
            {
                uint64_t index =
                    (uint64_t)band * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank;
                volatile uint64_t *entry = entries + index * 3u;
                uint64_t seq;
                uint64_t bytes;
                uint64_t slot;
                {
                    uint32_t stable = 0u;
                    uint32_t tries;
                    for ( tries = 0u; tries < 64u && stable == 0u; tries++ )
                    {
                        seq = entry[0];
                        bytes = entry[1];
                        slot = entry[2];
                        __sync_synchronize();
                        if ( seq == entry[0] && bytes == entry[1] &&
                            slot == entry[2] )
                            stable = 1u;
                    }
                    if ( stable == 0u )
                    {
                        weightd_mesh.doorbell_stuck[index]++;
                        if ( (weightd_mesh.doorbell_stuck[index] % 500u) == 0u )
                            fprintf(stderr,
                                "WD-STUCK idx=%llu phase=unstable\n",
                                (unsigned long long)index);
                        continue;
                    }
                }
                if ( seq == 0ull || bytes == 0ull )
                    continue;
                if ( seq == weightd_mesh.doorbell_posted[index])
                {
                    weightd_mesh.doorbell_stuck[index] = 0u;
                    continue;
                }
                if ( slot >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND ||
                    bytes + 8u > SPARK_WEIGHTD_MESH_SLOT_BYTES )
                {
                    weightd_mesh.doorbell_stuck[index]++;
                    if ( (weightd_mesh.doorbell_stuck[index] % 500u) == 0u )
                        fprintf(stderr,
                            "WD-STUCK idx=%llu phase=invalid seq=%llu bytes=%llu slot=%llu\n",
                            (unsigned long long)index,
                            (unsigned long long)seq,
                            (unsigned long long)bytes,
                            (unsigned long long)slot);
                    continue;
                }
                weightd_mesh.doorbell_stuck[index]++;
                if ( (weightd_mesh.doorbell_stuck[index] % 500u) == 0u )
                    fprintf(stderr,
                        "WD-STUCK idx=%llu phase=pending seq=%llu posted=%llu bytes=%llu slot=%llu\n",
                        (unsigned long long)index,
                        (unsigned long long)seq,
                        (unsigned long long)weightd_mesh.doorbell_posted[index],
                        (unsigned long long)bytes,
                        (unsigned long long)slot);
                {
                    uint64_t slot_base = (uint64_t)band *
                        SPARK_WEIGHTD_MESH_SLOTS_PER_BAND *
                        SPARK_WEIGHTD_MESH_SLOT_BYTES +
                        slot * SPARK_WEIGHTD_MESH_SLOT_BYTES;
                    struct ibv_sge scatter;
                    struct ibv_send_wr work_request;
                    struct ibv_send_wr *bad;
                    uint32_t posted_failed = 0u;
                    uint64_t last_addr = 0ull;
                    uint32_t last_rkey = 0u;
                    uint32_t resync_mask = 0u;
                    {
                        uint64_t key_lo = (seq & ~0xffffull) + 1ull;
                        uint64_t first_missed =
                            weightd_mesh.doorbell_posted[index] + 1ull;
                        uint32_t missed_count;
                        uint64_t missed;
                        if ( first_missed < key_lo )
                            first_missed = key_lo;
                        if ( seq > first_missed )
                        {
                            missed_count = (uint32_t)(seq - first_missed);
                            if ( missed_count >
                                    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK )
                                first_missed = seq -
                                    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
                            for ( missed = first_missed; missed < seq;
                                missed++ )
                                resync_mask |=
                                    (uint32_t)1u << (uint32_t)(missed &
                                        (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
                        }
                    }
                    if ( resync_mask != 0u )
                    {
                        uint32_t ring;
                        for (ring = 0u; ring < SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
                            ring++)
                        {
                            if ( (resync_mask & (1u << ring)) == 0u )
                                continue;
                            uint64_t rank_slots = slot -
                                (slot % (uint64_t)
                                    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK);
                            uint64_t ring_base = (uint64_t)band *
                                SPARK_WEIGHTD_MESH_SLOTS_PER_BAND *
                                SPARK_WEIGHTD_MESH_SLOT_BYTES +
                                (rank_slots + ring) *
                                SPARK_WEIGHTD_MESH_SLOT_BYTES;
                            for (peer = 0u;
                                 peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
                            {
                                struct ibv_sge rsge;
                                struct ibv_send_wr rwr;
                                struct ibv_send_wr *rbad;
                                memset(&rsge,0,sizeof(rsge));
                                rsge.addr = (uint64_t)(uintptr_t)
                                    weightd_mesh.recv_buffer + ring_base;
                                rsge.length = SPARK_WEIGHTD_MESH_SLOT_BYTES;
                                rsge.lkey = weightd_mesh.recv_mr->lkey;
                                memset(&rwr,0,sizeof(rwr));
                                rwr.wr_id = (uint64_t)peer;
                                rwr.sg_list = &rsge;
                                rwr.num_sge = 1;
                                rwr.opcode = IBV_WR_RDMA_WRITE;
                                rwr.send_flags = 0u;
                                rwr.wr.rdma.remote_addr =
                                    qp_snapshot[peer].remote_addr + ring_base;
                                rwr.wr.rdma.rkey =
                                    qp_snapshot[peer].rkey;
                                if ( ibv_post_send(weightd_mesh.send_qps[peer],
                                        &rwr,&rbad) != 0 )
                                    posted_failed = 1u;
                            }
                        }
                    }
                    memset(&scatter,0,sizeof(scatter));
                    scatter.addr = (uint64_t)(uintptr_t)
                        weightd_mesh.recv_buffer + slot_base;
                    scatter.length = (uint32_t)(bytes + 8u);
                    scatter.lkey = weightd_mesh.recv_mr->lkey;
                    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
                    {
                        memset(&work_request,0,sizeof(work_request));
                        work_request.wr_id = (uint64_t)peer;
                        work_request.sg_list = &scatter;
                        work_request.num_sge = 1;
                        work_request.opcode = IBV_WR_RDMA_WRITE;
                        work_request.send_flags = IBV_SEND_SIGNALED;
                        work_request.wr.rdma.remote_addr =
                            qp_snapshot[peer].remote_addr + slot_base;
                        work_request.wr.rdma.rkey =
                            qp_snapshot[peer].rkey;
                        last_addr = qp_snapshot[peer].remote_addr;
                        last_rkey = qp_snapshot[peer].rkey;
                        if ( ibv_post_send(weightd_mesh.send_qps[peer],
                                &work_request,&bad) != 0 )
                        {
                            SparkWeightdMeshTryWire();
                            if ( ibv_post_send(weightd_mesh.send_qps[peer],
                                    &work_request,&bad) != 0 )
                                posted_failed = 1u;
                        }
                    }
                    if ( posted_failed == 0u )
                    {
                        weightd_mesh.doorbell_posted[index] = seq;
                        weightd_mesh.doorbell_stuck[index] = 0u;
                        weightd_mesh.ship_log_count++;
                        if ( (seq >> 16) != weightd_mesh.ship_log_key )
                        {
                            weightd_mesh.ship_log_key = seq >> 16;
                            weightd_mesh.ship_log_key_budget = 8u;
                        }
                        if ( (weightd_mesh.ship_log_count % 256u) == 0u ||
                             weightd_mesh.ship_log_key_budget != 0u )
                        {
                            if ( weightd_mesh.ship_log_key_budget != 0u )
                                weightd_mesh.ship_log_key_budget--;
                            fprintf(stderr,"WD-SHIP idx=%llu seq=%llu addr=%llx rkey=%u posted=%llu total=%llu\n",
                                (unsigned long long)index,
                                (unsigned long long)seq,
                                (unsigned long long)last_addr,
                                last_rkey,
                                (unsigned long long)weightd_mesh.doorbell_posted[index],
                                (unsigned long long)weightd_mesh.ship_log_count);
                        }
                    }
                    else if ( weightd_mesh.doorbell_stuck[index] % 500u == 0u )
                        fprintf(stderr,"WD-SHIP FAILED idx=%llu seq=%llu errno=%d\n",
                            (unsigned long long)index,
                            (unsigned long long)seq,errno);
                }
            }
        }
        nanosleep(&pause,0);
    }
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
        peer_rank == weightd_mesh.local_rank)
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

    if (weightd_mesh.mesh_ready == 0u)
        return 0u;
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
        if ( seq_remote_offset != 0ull && weightd_mesh.seq_mr != 0 )
        {
            struct ibv_send_wr seq_request;
            memset(&seq_request,0,sizeof(seq_request));
            seq_request.wr_id = (uint64_t)(uint32_t)peer;
            seq_request.sg_list = &scatter[1];
            seq_request.num_sge = 1;
            seq_request.opcode = IBV_WR_RDMA_WRITE;
            seq_request.send_flags = 0;
            seq_request.wr.rdma.remote_addr =
                weightd_mesh.qp_info[peer].remote_addr + seq_remote_offset;
            seq_request.wr.rdma.rkey = weightd_mesh.qp_info[peer].rkey;
            (void)ibv_post_send(weightd_mesh.send_qps[peer],
                &seq_request,&bad);
        }
        }
    }
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
    if (ibv_post_send(weightd_mesh.send_qps[peer],&work_request,&bad) != 0)
    {
        fprintf(stderr,"weightd-mesh: post peer=%u errno=%d\n",peer,errno);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    return SPARK_STATUS_OK;
}
