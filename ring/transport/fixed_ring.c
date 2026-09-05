#include "sparkpipe/spark_fixed_ring.h"

#include <infiniband/verbs.h>
#include <nng/nng.h>
#include <nng/protocol/survey0/respond.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define GID_INDEX 3
#define ELEMS 4096
#define ACC_BYTES (ELEMS * 2)
#define LANDING_OFF ACC_BYTES
#define TOTAL_BYTES (ACC_BYTES + 16 * 2048)
#define ENTRY_BYTES 128
#define BROKER_HOST_LAST_OCTET 19

typedef struct
{
    uint32_t qp_number;
    uint32_t psn;
    uint32_t lid;
    uint16_t active_mtu;
    uint16_t memory_mode;
    uint32_t reserved[3];
    uint8_t gid[16];
    uint32_t rkey;
    uint64_t buf_addr;
} wire_info;

typedef struct
{
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    uint8_t *buf;
    wire_info local;
    wire_info remote;
} link_qp;

struct SparkFixedRing
{
    uint32_t rank;
    uint32_t degree;
    uint32_t chunk_elems;
    uint32_t chunk_bytes;
    uint32_t slot_stride;
    link_qp qp_next;
    link_qp qp_prev;
    uint32_t recv_depth;
};

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + ts.tv_nsec / 1000ull;
}

static int broker_exchange(uint32_t rank, uint32_t degree, uint32_t port,
    const uint8_t *my_entry, uint8_t *table_out)
{
    nng_socket sock;
    uint8_t join[4 + ENTRY_BYTES];
    int32_t rank32;
    uint8_t *msg = 0;
    size_t sz = 0;
    char url[48];
    int rv;
    if ((rv = nng_respondent0_open(&sock)) != 0)
        return -1;
    nng_socket_set_ms(sock, NNG_OPT_RECVTIMEO, 240000);
    snprintf(url, sizeof(url), "tcp://10.10.100.%d:%d",
        BROKER_HOST_LAST_OCTET, (int)port);
    if ((rv = nng_dial(sock, url, NULL, NNG_FLAG_NONBLOCK)) != 0)
    {
        nng_close(sock);
        return -1;
    }
    for (;;)
    {
        if ((rv = nng_recv(sock, &msg, &sz, NNG_FLAG_ALLOC)) != 0)
        {
            nng_close(sock);
            return -1;
        }
        if (sz == 4u)
        {
            nng_free(msg, sz);
            break;
        }
        if (sz == (size_t)degree * ENTRY_BYTES)
        {
            memcpy(table_out, msg, (size_t)degree * ENTRY_BYTES);
            nng_free(msg, sz);
            nng_close(sock);
            return 0;
        }
        nng_free(msg, sz);
    }
    rank32 = (int32_t)rank;
    memcpy(join, &rank32, sizeof(rank32));
    memcpy(join + 4, my_entry, ENTRY_BYTES);
    if ((rv = nng_send(sock, join, sizeof(join), 0)) != 0)
    {
        nng_close(sock);
        return -1;
    }
    for (;;)
    {
        if ((rv = nng_recv(sock, &msg, &sz, NNG_FLAG_ALLOC)) != 0)
        {
            nng_close(sock);
            return -1;
        }
        if (sz == (size_t)degree * ENTRY_BYTES)
        {
            memcpy(table_out, msg, (size_t)degree * ENTRY_BYTES);
            nng_free(msg, sz);
            nng_close(sock);
            return 0;
        }
        nng_free(msg, sz);
    }
}

static int open_qp(link_qp *q)
{
    struct ibv_device **list = ibv_get_device_list(0);
    int i;
    struct ibv_qp_init_attr init;
    struct ibv_qp_attr attr;
    union ibv_gid gid;
    struct ibv_port_attr port;
    memset(q, 0, sizeof(*q));
    for (i = 0; list != 0 && list[i] != 0; ++i)
    {
        if (strcmp(ibv_get_device_name(list[i]), "rocep1s0f1") == 0)
        {
            q->ctx = ibv_open_device(list[i]);
            break;
        }
    }
    if (list != 0)
        ibv_free_device_list(list);
    if (q->ctx == 0)
        return -1;
    q->pd = ibv_alloc_pd(q->ctx);
    q->buf = malloc(TOTAL_BYTES);
    q->mr = ibv_reg_mr(q->pd, q->buf, TOTAL_BYTES,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);
    q->cq = ibv_create_cq(q->ctx, 64, 0, 0, 0);
    memset(&init, 0, sizeof(init));
    init.send_cq = q->cq;
    init.recv_cq = q->cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 64;
    init.cap.max_recv_wr = 64;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    q->qp = ibv_create_qp(q->pd, &init);
    if (q->pd == 0 || q->buf == 0 || q->mr == 0 || q->cq == 0 || q->qp == 0)
        return -1;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(q->qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
            IBV_QP_ACCESS_FLAGS) != 0)
        return -1;
    memset(&q->local, 0, sizeof(q->local));
    q->local.qp_number = q->qp->qp_num;
    q->local.psn = (uint32_t)(getpid() * 7919u + q->qp->qp_num * 104729u) &
        0x00ffffffu;
    ibv_query_port(q->ctx, 1, &port);
    q->local.lid = port.lid;
    q->local.active_mtu = (uint16_t)port.active_mtu;
    q->local.memory_mode = 1;
    ibv_query_gid(q->ctx, 1, GID_INDEX, &gid);
    memcpy(q->local.gid, gid.raw, 16);
    q->local.rkey = q->mr->rkey;
    q->local.buf_addr = (uint64_t)(uintptr_t)q->buf;
    return 0;
}

static int bring_rts(link_qp *q)
{
    struct ibv_qp_attr attr;
    union ibv_gid dgid;
    uint8_t mtu = q->local.active_mtu < q->remote.active_mtu ?
        q->local.active_mtu : q->remote.active_mtu;
    if (mtu < IBV_MTU_512)
        mtu = IBV_MTU_512;
    memcpy(dgid.raw, q->remote.gid, 16);
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = (enum ibv_mtu)mtu;
    attr.dest_qp_num = q->remote.qp_number;
    attr.rq_psn = q->remote.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = dgid;
    attr.ah_attr.grh.sgid_index = GID_INDEX;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.dlid = q->remote.lid;
    attr.ah_attr.port_num = 1;
    if (ibv_modify_qp(q->qp, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
            IBV_QP_MIN_RNR_TIMER) != 0)
        return -1;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = q->local.psn;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(q->qp, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
        return -1;
    return 0;
}

static void post_recv(link_qp *q)
{
    struct ibv_recv_wr wr;
    struct ibv_recv_wr *bad = 0;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0x9000000000000000ull;
    wr.num_sge = 0;
    if (ibv_post_recv(q->qp, &wr, &bad) != 0)
    {
        fprintf(stderr, "fixed_ring: post_recv failed\n");
        exit(1);
    }
}

static void send_wr(link_qp *q, const void *buffer, uint32_t bytes,
    uint32_t immediate, uint32_t slot_stride)
{
    struct ibv_send_wr wr[2];
    struct ibv_sge sge;
    struct ibv_send_wr *bad = 0;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)buffer;
    sge.length = bytes;
    sge.lkey = q->mr->lkey;
    memset(wr, 0, sizeof(wr));
    wr[0].wr_id = 0x11;
    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].sg_list = &sge;
    wr[0].num_sge = 1;
    wr[0].wr.rdma.remote_addr = q->remote.buf_addr +
        (uint64_t)LANDING_OFF +
        (uint64_t)(immediate & 15u) * (uint64_t)slot_stride;
    wr[0].wr.rdma.rkey = q->remote.rkey;
    wr[1].wr_id = 0x11;
    wr[1].opcode = IBV_WR_SEND_WITH_IMM;
    wr[1].send_flags = IBV_SEND_SIGNALED;
    wr[1].imm_data = htonl(immediate);
    wr[0].next = &wr[1];
    if (ibv_post_send(q->qp, &wr[0], &bad) != 0)
    {
        fprintf(stderr, "fixed_ring: post_send failed\n");
        exit(1);
    }
}

static void drain_send_cq(link_qp *q)
{
    struct ibv_wc wc;
    uint64_t deadline = now_us() + 2500000ull;
    for (;;)
    {
        int n = ibv_poll_cq(q->cq, 1, &wc);
        if (n > 0)
        {
            if ((wc.wr_id & 0x9000000000000000ull) != 0ull)
            {
                post_recv(q);
                return;
            }
            if (wc.status != IBV_WC_SUCCESS)
                fprintf(stderr, "fixed_ring: send wc %u (%s)\n",
                    wc.status, ibv_wc_status_str(wc.status));
            return;
        }
        if (now_us() > deadline)
        {
            fprintf(stderr, "fixed_ring: send drain timeout\n");
            return;
        }
    }
}

static int wait_imm(link_qp *q, uint32_t expect, uint64_t timeout_ns)
{
    struct ibv_wc wc;
    uint64_t deadline = now_us() + timeout_ns / 1000ull;
    for (;;)
    {
        int n = ibv_poll_cq(q->cq, 1, &wc);
        if (n > 0)
        {
            uint32_t imm;
            struct ibv_send_wr echo;
            struct ibv_send_wr *bad = 0;
            if ((wc.wr_id & 0x9000000000000000ull) == 0ull)
                continue;
            post_recv(q);
            if (wc.status != IBV_WC_SUCCESS)
                return -1;
            imm = ntohl(wc.imm_data);
            if (imm == expect)
                return 0;
            if ((imm & 0x20000000u) != 0u)
            {
                memset(&echo, 0, sizeof(echo));
                echo.wr_id = 0x21ull;
                echo.opcode = IBV_WR_SEND_WITH_IMM;
                echo.send_flags = IBV_SEND_SIGNALED;
                echo.imm_data = htonl(imm | 0x10000000u);
                if (ibv_post_send(q->qp, &echo, &bad) != 0)
                    return -1;
                continue;
            }
            fprintf(stderr, "fixed_ring wait mismatch got=%08x expect=%08x\n",
                imm, expect);
            return -2;
        }
        if (now_us() > deadline)
        {
            fprintf(stderr, "fixed_ring wait timeout expect=%08x\n", expect);
            return -3;
        }
    }
}

static int probe_pair(SparkFixedRing *ring)
{
    uint32_t attempt;
    uint8_t *scratch = (uint8_t *)ring->qp_next.buf;
    for (attempt = 0u; attempt < 16u; ++attempt)
    {
        uint32_t want = 0x70000000u | (uint32_t)attempt;
        uint64_t deadline = now_us() + 1000000ull;
        send_wr(&ring->qp_next, scratch, 8u,
            0x60000000u | (uint32_t)attempt, 8u);
        for (;;)
        {
            struct ibv_wc wc;
            int n = ibv_poll_cq(ring->qp_next.cq, 1, &wc);
            if (n > 0 && (wc.wr_id & 0x9000000000000000ull) != 0ull)
            {
                post_recv(&ring->qp_next);
                if (wc.status == IBV_WC_SUCCESS &&
                    ntohl(wc.imm_data) == want)
                    return 0;
            }
            n = ibv_poll_cq(ring->qp_prev.cq, 1, &wc);
            if (n > 0 && (wc.wr_id & 0x9000000000000000ull) != 0ull)
            {
                uint32_t imm = ntohl(wc.imm_data);
                struct ibv_send_wr echo;
                struct ibv_send_wr *bad = 0;
                post_recv(&ring->qp_prev);
                if (wc.status == IBV_WC_SUCCESS &&
                    (imm & 0x20000000u) != 0u)
                {
                    memset(&echo, 0, sizeof(echo));
                    echo.wr_id = 0x21ull;
                    echo.opcode = IBV_WR_SEND_WITH_IMM;
                    echo.send_flags = IBV_SEND_SIGNALED;
                    echo.imm_data = htonl(imm | 0x10000000u);
                    (void)ibv_post_send(ring->qp_prev.qp, &echo, &bad);
                }
            }
            if (now_us() > deadline)
                break;
        }
    }
    return -1;
}

static int probe_pair(SparkFixedRing *ring);

SparkStatus SparkFixedRingCreate(
    uint32_t rank,
    uint32_t degree,
    uint32_t start_rank,
    uint32_t broker_port,
    SparkFixedRing **ring_out)
{
    SparkFixedRing *ring;
    uint8_t my_entry[ENTRY_BYTES];
    uint8_t table[16 * ENTRY_BYTES];
    uint32_t next;
    uint32_t prev;
    int i;
    (void)start_rank;
    if (ring_out == 0 || degree == 0u || degree > 16u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *ring_out = 0;
    ring = (SparkFixedRing *)calloc(1u, sizeof(*ring));
    if (ring == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    ring->rank = rank;
    ring->degree = degree;
    ring->chunk_elems = ELEMS / degree;
    ring->chunk_bytes = ring->chunk_elems * 2u;
    ring->slot_stride = 2048u;
    if (open_qp(&ring->qp_next) != 0 || open_qp(&ring->qp_prev) != 0)
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    memset(my_entry, 0, sizeof(my_entry));
    memcpy(my_entry, &ring->qp_next.local, sizeof(wire_info));
    memcpy(my_entry + 64, &ring->qp_prev.local, sizeof(wire_info));
    if (broker_exchange(rank, degree, broker_port, my_entry, table) != 0)
        return SPARK_STATUS_IO_ERROR;
    next = (rank + 1u) % degree;
    prev = (rank + degree - 1u) % degree;
    memcpy(&ring->qp_next.remote, table + (size_t)next * ENTRY_BYTES + 64,
        sizeof(wire_info));
    memcpy(&ring->qp_prev.remote, table + (size_t)prev * ENTRY_BYTES,
        sizeof(wire_info));
    if (bring_rts(&ring->qp_next) != 0 || bring_rts(&ring->qp_prev) != 0)
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    for (i = 0; i < 32; ++i)
        post_recv(&ring->qp_prev);
    for (i = 0; i < 4; ++i)
        post_recv(&ring->qp_next);
    if (probe_pair(ring) != 0)
    {
        fprintf(stderr, "fixed_ring: rank %u probe failed\n", rank);
        return SPARK_STATUS_IO_ERROR;
    }
    memset(my_entry, 0, sizeof(my_entry));
    if (broker_exchange(rank, degree, broker_port, my_entry, table) != 0)
        return SPARK_STATUS_IO_ERROR;
    *ring_out = ring;
    return SPARK_STATUS_OK;
}

SparkStatus SparkFixedRingSetChunkBytes(
    SparkFixedRing *ring,
    uint32_t chunk_bytes)
{
    if (ring == 0 || chunk_bytes == 0u || chunk_bytes > 2048u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    ring->slot_stride = chunk_bytes;
    return SPARK_STATUS_OK;
}

SparkStatus SparkFixedRingSendNext(
    SparkFixedRing *ring,
    const void *buffer,
    uint32_t bytes,
    uint32_t immediate)
{
    if (ring == 0 || buffer == 0 || bytes == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    send_wr(&ring->qp_next, buffer, bytes, immediate, ring->slot_stride);
    return SPARK_STATUS_OK;
}

SparkStatus SparkFixedRingWaitPrev(
    SparkFixedRing *ring,
    uint32_t expect_immediate,
    uint64_t timeout_ns)
{
    if (ring == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (wait_imm(&ring->qp_prev, expect_immediate, timeout_ns) != 0)
        return SPARK_STATUS_IO_ERROR;
    drain_send_cq(&ring->qp_next);
    return SPARK_STATUS_OK;
}

const void *SparkFixedRingLanding(SparkFixedRing *ring)
{
    return ring != 0 ? ring->qp_prev.buf + LANDING_OFF : 0;
}

uint16_t *SparkFixedRingAccumulator(SparkFixedRing *ring)
{
    return ring != 0 ? (uint16_t *)ring->qp_next.buf : 0;
}

uint32_t SparkFixedRingChunkElems(SparkFixedRing *ring)
{
    return ring != 0 ? ring->chunk_elems : 0u;
}

uint32_t SparkFixedRingChunkBytes(SparkFixedRing *ring)
{
    return ring != 0 ? ring->chunk_bytes : 0u;
}

void SparkFixedRingDestroy(SparkFixedRing *ring)
{
    if (ring == 0)
        return;
    free(ring);
}
