#include <infiniband/verbs.h>
#include <nng/nng.h>
#include <nng/protocol/survey0/respond.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEGREE 16
#define ELEMS 4096
#define PAYLOAD (ELEMS * 2)
#define SLOT_BYTES (PAYLOAD + 8)
#define FIFO_DEPTH 4
#define WIRE_BYTES 56
#define ENTRY_BYTES 1024
#define ITERATIONS 50
#define WEIGHT_MS 25.0
#define COLLECTIVES_PER_TOKEN 90.0
#define GID_INDEX 3

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
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    uint8_t *landing;
    wire_info local;
    wire_info remote;
} peer_link;

static int rank_g;
static int degree_g = 16;
static struct ibv_context *ctx_g;
static struct ibv_pd *pd_g;
static peer_link peers[DEGREE];

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + ts.tv_nsec / 1000ull;
}

static uint16_t f32_to_bf16(float f);
static float bf16_to_f32(uint16_t b);

static uint16_t pattern_val(int r, int i)
{
    return (uint16_t)((uint32_t)(r * 251 + (i % 241) + 1) & 0x3fffu);
}

static uint16_t expected_sum(int i)
{
    int r;
    float total = 0.0f;
    for (r = 0; r < degree_g; ++r)
        total += bf16_to_f32(f32_to_bf16((float)pattern_val(r, i)));
    return f32_to_bf16(total);
}

static int broker_round(uint32_t rank, uint32_t degree, uint32_t port,
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
    snprintf(url, sizeof(url), "tcp://10.10.100.19:%d", (int)port);
    rv = nng_dial(sock, url, NULL, NNG_FLAG_NONBLOCK);
    fprintf(stderr, "rank %d: BR-DIAL rv=%d url=%s\n", rank_g, rv, url);
    if (rv != 0)
    {
        nng_close(sock);
        return -1;
    }
    for (;;)
    {
        rv = nng_recv(sock, &msg, &sz, NNG_FLAG_ALLOC);
        fprintf(stderr, "rank %d: BR-RECV rv=%d sz=%zu\n", rank_g, rv, sz);
        if (rv != 0)
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

static struct ibv_pd *pd_g;

static int open_peer(peer_link *p)
{
    struct ibv_qp_init_attr init;
    struct ibv_qp_attr attr;
    union ibv_gid gid;
    struct ibv_port_attr port;
    memset(p, 0, sizeof(*p));
    p->landing = calloc(1u, SLOT_BYTES);
    p->mr = ibv_reg_mr(pd_g, p->landing, SLOT_BYTES,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);
    p->cq = ibv_create_cq(ctx_g, 128, 0, 0, 0);
    memset(&init, 0, sizeof(init));
    init.send_cq = p->cq;
    init.recv_cq = p->cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 32;
    init.cap.max_recv_wr = 32;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 0;
    p->qp = ibv_create_qp(pd_g, &init);
    if (p->landing == 0 || p->mr == 0 || p->cq == 0 || p->qp == 0)
        return -1;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(p->qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
            IBV_QP_ACCESS_FLAGS) != 0)
        return -1;
    memset(&p->local, 0, sizeof(p->local));
    p->local.qp_number = p->qp->qp_num;
    p->local.psn = (uint32_t)(getpid() * 7919u + p->qp->qp_num * 104729u) &
        0x00ffffffu;
    ibv_query_port(ctx_g, 1, &port);
    p->local.lid = port.lid;
    p->local.active_mtu = (uint16_t)port.active_mtu;
    p->local.memory_mode = 1;
    ibv_query_gid(ctx_g, 1, GID_INDEX, &gid);
    memcpy(p->local.gid, gid.raw, 16);
    p->local.rkey = p->mr->rkey;
    p->local.buf_addr = (uint64_t)(uintptr_t)p->landing;
    return 0;
}

static int wire_peer(peer_link *p)
{
    struct ibv_qp_attr attr;
    union ibv_gid dgid;
    uint8_t mtu = p->local.active_mtu < p->remote.active_mtu ?
        p->local.active_mtu : p->remote.active_mtu;
    if (mtu < IBV_MTU_512)
        mtu = IBV_MTU_512;
    memcpy(dgid.raw, p->remote.gid, 16);
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = (enum ibv_mtu)mtu;
    attr.dest_qp_num = p->remote.qp_number;
    attr.rq_psn = p->remote.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = dgid;
    attr.ah_attr.grh.sgid_index = GID_INDEX;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.dlid = p->remote.lid;
    attr.ah_attr.port_num = 1;
    if (ibv_modify_qp(p->qp, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
            IBV_QP_MIN_RNR_TIMER) != 0)
        return -1;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = p->local.psn;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(p->qp, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
        return -1;
    return 0;
}

static uint8_t *send_buf_g;
static struct ibv_mr *send_mr_g;
static float acc_g[FIFO_DEPTH][ELEMS];
static uint32_t bits_g[FIFO_DEPTH];

static float bf16_to_f32(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static void write_slot(peer_link *p)
{
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    struct ibv_send_wr *bad = 0;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)send_buf_g;
    sge.length = SLOT_BYTES;
    sge.lkey = send_mr_g->lkey;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0x11;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = p->remote.buf_addr;
    wr.wr.rdma.rkey = p->remote.rkey;
    if (ibv_post_send(p->qp, &wr, &bad) != 0)
    {
        fprintf(stderr, "rank %d: post WRITE peer=%u errno=%d\n",
            rank_g, p->local.qp_number, errno);
        exit(1);
    }
}

static int poll_fold(uint32_t gen, uint32_t slot, uint64_t timeout_ns)
{
    struct ibv_wc wc;
    uint64_t deadline = now_us() + timeout_ns / 1000ull;
    uint32_t want = (1u << (uint32_t)degree_g) - 1u;
    bits_g[slot] = 1u << (uint32_t)rank_g;
    for (;;)
    {
        int peer;
        for (peer = 0; peer < degree_g; ++peer)
        {
            if (peer == rank_g)
                continue;
            while (ibv_poll_cq(peers[peer].cq, 1, &wc) > 0)
            {
                if (wc.status != IBV_WC_SUCCESS)
                {
                    fprintf(stderr, "rank %d: SEND-WC-ERR peer=%d status=%u\n",
                        rank_g, peer, (uint32_t)wc.status);
                    return -1;
                }
            }
            if ((bits_g[slot] & (1u << (uint32_t)peer)) == 0u &&
                *(volatile uint64_t *)(peers[peer].landing + PAYLOAD) >=
                    (uint64_t)gen)
            {
                const uint16_t *src =
                    (const uint16_t *)peers[peer].landing;
                float *dst = acc_g[slot];
                int i;
                for (i = 0; i < ELEMS; ++i)
                    dst[i] += bf16_to_f32(src[i]);
                bits_g[slot] |= 1u << (uint32_t)peer;
            }
        }
        if (bits_g[slot] == want)
            return 0;
        if (now_us() > deadline)
        {
            fprintf(stderr,
                "rank %d: FOLD-TIMEOUT gen=%u missing=%08x\n",
                rank_g, gen, want & ~bits_g[slot]);
            return -3;
        }
    }
}

int main(int argc, char **argv)
{
    struct ibv_device **list;
    uint8_t my_entry[ENTRY_BYTES];
    uint8_t table[DEGREE * ENTRY_BYTES];
    int i;
    int peer;
    int iter;
    uint64_t total_us = 0;
    uint64_t min_us = (uint64_t)-1;
    int verify_fail = 0;
    if (argc < 5)
    {
        fprintf(stderr, "usage: mock_allgather rank degree start port\n");
        return 2;
    }
    rank_g = atoi(argv[1]);
    degree_g = atoi(argv[2]);
    if (degree_g != 4 && degree_g != 8 && degree_g != 16)
        return 2;
    setvbuf(stdout, 0, _IOLBF, 0);
    list = ibv_get_device_list(0);
    for (i = 0; list != 0 && list[i] != 0; ++i)
    {
        if (strcmp(ibv_get_device_name(list[i]), "rocep1s0f1") == 0)
        {
            ctx_g = ibv_open_device(list[i]);
            break;
        }
    }
    if (list != 0)
        ibv_free_device_list(list);
    if (ctx_g == 0)
        return 1;
    pd_g = ibv_alloc_pd(ctx_g);
    send_buf_g = calloc(1u, SLOT_BYTES);
    send_mr_g = ibv_reg_mr(pd_g, send_buf_g, SLOT_BYTES,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (send_buf_g == 0 || send_mr_g == 0)
        return 1;
    {
        uint16_t *dst = (uint16_t *)send_buf_g;
        for (i = 0; i < ELEMS; ++i)
            dst[i] = f32_to_bf16((float)pattern_val(rank_g, i));
    }
    memset(my_entry, 0, sizeof(my_entry));
    for (peer = 0; peer < degree_g; ++peer)
    {
        if (peer == rank_g)
            continue;
        if (open_peer(&peers[peer]) != 0)
            return 1;
        memcpy(my_entry + (size_t)peer * WIRE_BYTES, &peers[peer].local,
            WIRE_BYTES);
    }
    if (broker_round((uint32_t)rank_g, (uint32_t)degree_g,
            (uint32_t)atoi(argv[4]), my_entry, table) != 0)
        return 1;
    for (peer = 0; peer < degree_g; ++peer)
    {
        if (peer == rank_g)
            continue;
        memcpy(&peers[peer].remote,
            table + (size_t)peer * ENTRY_BYTES +
                (size_t)rank_g * WIRE_BYTES,
            WIRE_BYTES);
        if (peers[peer].remote.qp_number == 0u)
        {
            fprintf(stderr, "rank %d: peer %d missing wire info\n",
                rank_g, peer);
            return 1;
        }
        if (wire_peer(&peers[peer]) != 0)
            return 1;
    }
    memset(my_entry, 0, sizeof(my_entry));
    if (broker_round((uint32_t)rank_g, (uint32_t)degree_g,
            (uint32_t)atoi(argv[4]), my_entry, table) != 0)
        return 1;
    for (iter = 0; iter < ITERATIONS; ++iter)
    {
        uint64_t t0;
        uint64_t us;
        uint16_t out[ELEMS];
        uint32_t gen = (uint32_t)iter + 1u;
        uint32_t slot = gen % FIFO_DEPTH;
        float *acc = acc_g[slot];
        t0 = now_us();
        *(volatile uint64_t *)(send_buf_g + PAYLOAD) = (uint64_t)gen;
        memset(acc, 0, sizeof(acc_g[0]));
        {
            const uint16_t *mine = (const uint16_t *)send_buf_g;
            for (i = 0; i < ELEMS; ++i)
                acc[i] = bf16_to_f32(mine[i]);
        }
        for (peer = 0; peer < degree_g; ++peer)
        {
            if (peer == rank_g)
                continue;
            write_slot(&peers[peer]);
        }
        if (poll_fold(gen, slot, 2500000000ull) != 0)
            return 1;
        for (i = 0; i < ELEMS; ++i)
            out[i] = f32_to_bf16(acc[i]);
        us = now_us() - t0;
        total_us += us;
        if (us < min_us)
            min_us = us;
        if ((iter % 10) == 9 || iter == ITERATIONS - 1)
        {
            for (i = 0; i < ELEMS; ++i)
            {
                if (out[i] != expected_sum(i))
                {
                    fprintf(stderr,
                        "rank %d VERIFY FAIL iter=%d elem=%d acc=%f accbits=%08x outbits=%04x expect=%u own=%u L1=%u L2=%u L3=%u\n",
                        rank_g, iter, i, acc[i],
                        *(uint32_t *)(void *)&acc[i], (unsigned)out[i],
                        (unsigned)expected_sum(i),
                        (unsigned)pattern_val(rank_g, i),
                        degree_g > 1 && 1 != rank_g ? (unsigned)((const uint16_t *)peers[1].landing)[i] : 0u,
                        degree_g > 2 && 2 != rank_g ? (unsigned)((const uint16_t *)peers[2].landing)[i] : 0u,
                        degree_g > 3 && 3 != rank_g ? (unsigned)((const uint16_t *)peers[3].landing)[i] : 0u);
                    ++verify_fail;
                    break;
                }
            }
        }
    }
    {
        double mean_us = (double)total_us / (double)ITERATIONS;
        double b1 = 1000000.0 /
            (WEIGHT_MS * 1000.0 + COLLECTIVES_PER_TOKEN * mean_us);
        printf("ALLGATHER rank=%2d iters=%d mean_us=%.1f min_us=%.1f payload=%dB verify=%s\n",
            rank_g, ITERATIONS, mean_us, (double)min_us, PAYLOAD,
            verify_fail != 0 ? "FAIL" : "OK");
        printf("TOKS rank=%2d per_op_us=%.1f b1_tok_s=%.1f overlap_ceiling_tok_s=%.1f\n",
            rank_g, mean_us, b1, 1000.0 / WEIGHT_MS);
    }
    return verify_fail != 0 ? 1 : 0;
}
