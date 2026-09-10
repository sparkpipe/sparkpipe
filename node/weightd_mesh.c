#define _GNU_SOURCE
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
#include <sys/mman.h>
#include <sys/stat.h>

#define SPARK_WEIGHTD_MESH_PEERS 15
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
    uint32_t mesh_ready;
    uint32_t local_rank;
} SparkWeightdMesh;

static SparkWeightdMesh weightd_mesh;

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

SparkStatus SparkWeightdMeshInit(void)
{
    struct ibv_device **devices;
    struct ibv_port_attr port_attr;
    struct ibv_qp_init_attr qp_attributes;
    SparkWeightdMeshRecord own_record;
    SparkWeightdMeshRecord peer_records[SPARK_WEIGHTD_MESH_PEERS];
    int device_count;
    uint32_t peer;
    uint32_t peer_rank;
    uint32_t my_index_in_peer;
    time_t deadline;

    memset(&weightd_mesh,0,sizeof(weightd_mesh));
    weightd_mesh.local_rank = SparkWeightdMeshRankFromHost();

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
    weightd_mesh.cq = ibv_create_cq(weightd_mesh.context,256,0,0,0);
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
    if (ftruncate(weightd_mesh.memfd,SPARK_WEIGHTD_MESH_BUFFER_BYTES) != 0)
    {
        fprintf(stderr,"weightd-mesh: ftruncate failed errno=%d\n",errno);
        return SPARK_STATUS_IO_ERROR;
    }
    weightd_mesh.recv_buffer = mmap(0,SPARK_WEIGHTD_MESH_BUFFER_BYTES,
        PROT_READ | PROT_WRITE,MAP_SHARED,weightd_mesh.memfd,0);
    if (weightd_mesh.recv_buffer == MAP_FAILED)
    {
        fprintf(stderr,"weightd-mesh: mmap failed errno=%d\n",errno);
        weightd_mesh.recv_buffer = 0;
        return SPARK_STATUS_IO_ERROR;
    }
    weightd_mesh.recv_mr = ibv_reg_mr(weightd_mesh.protection_domain,
        weightd_mesh.recv_buffer,SPARK_WEIGHTD_MESH_BUFFER_BYTES,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (weightd_mesh.recv_mr == 0)
    {
        fprintf(stderr,"weightd-mesh: mr failed errno=%d\n",errno);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    memset(&qp_attributes,0,sizeof(qp_attributes));
    qp_attributes.send_cq = weightd_mesh.cq;
    qp_attributes.recv_cq = weightd_mesh.cq;
    qp_attributes.cap.max_send_wr = 16;
    qp_attributes.cap.max_recv_wr = 16;
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
    printf("weightd-mesh: published, awaiting %u peers\n",
        SPARK_WEIGHTD_MESH_PEERS);
    fflush(stdout);
    deadline = time(0) + 30;
    {
        uint32_t found[SPARK_WEIGHTD_MESH_PEERS];
        uint32_t found_count;
        memset(found,0,sizeof(found));
        for (;;)
        {
            found_count = 0;
            for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
            {
                if (found[peer] != 0u)
                {
                    found_count++;
                    continue;
                }
                peer_rank = peer < weightd_mesh.local_rank ?
                    peer : peer + 1u;
                if (SparkWeightdMeshReadPeerRecord(peer_rank,
                        &peer_records[peer]) == SPARK_STATUS_OK)
                {
                    found[peer] = 1u;
                    found_count++;
                    printf("weightd-mesh: peer %u found\n",peer_rank);
                    fflush(stdout);
                }
            }
            if (found_count == SPARK_WEIGHTD_MESH_PEERS)
                break;
            if (time(0) >= deadline)
            {
                fprintf(stderr,"weightd-mesh: %u/%u peers found, timeout\n",
                    found_count,SPARK_WEIGHTD_MESH_PEERS);
                return SPARK_STATUS_BUSY;
            }
            usleep(500000);
        }
    }
    printf("weightd-mesh: all peers found, transitioning QPs\n");
    fflush(stdout);
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        peer_rank = peer < weightd_mesh.local_rank ? peer : peer + 1u;
        my_index_in_peer = weightd_mesh.local_rank < peer_rank ?
            weightd_mesh.local_rank : weightd_mesh.local_rank - 1u;
        if (SparkWeightdMeshTransitionQp(
                weightd_mesh.send_qps[peer],
                peer_records[peer].recv_qpn[my_index_in_peer],
                peer_records[peer].lid,
                peer_records[peer].gid) != SPARK_STATUS_OK)
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        if (SparkWeightdMeshTransitionQp(
                weightd_mesh.recv_qps[peer],
                peer_records[peer].send_qpn[my_index_in_peer],
                peer_records[peer].lid,
                peer_records[peer].gid) != SPARK_STATUS_OK)
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        weightd_mesh.qp_info[peer].remote_qpn =
            peer_records[peer].recv_qpn[my_index_in_peer];
        weightd_mesh.qp_info[peer].rkey = peer_records[peer].rkey;
        weightd_mesh.qp_info[peer].remote_addr =
            peer_records[peer].recv_addr;
    }
    weightd_mesh.mesh_ready = 1u;
    printf("weightd-mesh: ready rank=%u peers=%u rkey=%u\n",
        weightd_mesh.local_rank,SPARK_WEIGHTD_MESH_PEERS,
        weightd_mesh.recv_mr->rkey);
    fflush(stdout);
    return SPARK_STATUS_OK;
}

uint32_t SparkWeightdMeshReady(void)
{
    return weightd_mesh.mesh_ready;
}

uint64_t SparkWeightdMeshBufferAddress(void)
{
    return (uint64_t)(uintptr_t)weightd_mesh.recv_buffer;
}

int SparkWeightdMeshBufferFd(void)
{
    return weightd_mesh.mesh_ready != 0u ? dup(weightd_mesh.memfd) : -1;
}

uint32_t SparkWeightdMeshBufferLkey(void)
{
    return weightd_mesh.recv_mr != 0 ? weightd_mesh.recv_mr->lkey : 0u;
}

uint32_t SparkWeightdMeshBroadcast(
    uint32_t peer_mask,
    uint64_t source_offset,
    uint32_t length,
    uint64_t remote_offset)
{
    struct ibv_sge scatter;
    struct ibv_send_wr work_request;
    struct ibv_send_wr *bad;
    uint32_t peer;
    uint32_t posted = 0u;

    if (weightd_mesh.mesh_ready == 0u)
        return 0u;
    memset(&scatter,0,sizeof(scatter));
    scatter.addr = (uint64_t)(uintptr_t)weightd_mesh.recv_buffer +
        source_offset;
    scatter.length = length;
    scatter.lkey = weightd_mesh.recv_mr->lkey;
    for (peer = 0u; peer < SPARK_WEIGHTD_MESH_PEERS; peer++)
    {
        if ((peer_mask & (1u << peer)) == 0u)
            continue;
        memset(&work_request,0,sizeof(work_request));
        work_request.wr_id = peer;
        work_request.sg_list = &scatter;
        work_request.num_sge = 1;
        work_request.opcode = IBV_WR_RDMA_WRITE;
        work_request.send_flags = IBV_SEND_SIGNALED;
        work_request.wr.rdma.remote_addr =
            weightd_mesh.qp_info[peer].remote_addr + remote_offset;
        work_request.wr.rdma.rkey = weightd_mesh.qp_info[peer].rkey;
        if (ibv_post_send(weightd_mesh.send_qps[peer],
                &work_request,&bad) == 0)
            posted++;
    }
    return posted;
}

SparkStatus SparkWeightdMeshPostWrite(
    uint32_t peer,
    uint64_t local_addr,
    uint32_t lkey,
    uint32_t length,
    uint64_t remote_offset)
{
    struct ibv_sge scatter;
    struct ibv_send_wr work_request;
    struct ibv_send_wr *bad;

    if (weightd_mesh.mesh_ready == 0u || peer >= SPARK_WEIGHTD_MESH_PEERS)
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
