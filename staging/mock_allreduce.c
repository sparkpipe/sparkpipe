#include <infiniband/verbs.h>
#include <nng/nng.h>
#include <nng/protocol/survey0/respond.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int degree_g = 16;
static int chunk_elems_g = 256;
static uint32_t chunk_bytes_g = 512;
static int broker_port_g = 58399;
#define GID_INDEX 3
#define ELEMS 4096
#define ACC_BYTES (ELEMS * 2)
#define LANDING_OFF ACC_BYTES
#define TOTAL_BYTES (ACC_BYTES + 16 * 2048)
#define ITERATIONS 50
#define WEIGHT_MS 25.0
#define COLLECTIVES_PER_TOKEN 90.0
#define ENTRY_BYTES 128
#define BROKER_PORT 58399

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

static int rank_g;
static link_qp qp_next;
static link_qp qp_prev;
static uint32_t sink_errors;

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + ts.tv_nsec / 1000ull;
}

static int parse_rank(const char *text)
{
    long value = strtol(text, 0, 10);
    if (value < 0 || value >= degree_g)
        return -1;
    return (int)value;
}

static int broker_exchange(const uint8_t *my_entry, uint8_t *table_out)
{
    nng_socket sock;
    uint8_t join[4 + ENTRY_BYTES];
    int32_t rank32;
    uint8_t *msg = 0;
    size_t sz = 0;
    int rv;
    if ((rv = nng_respondent0_open(&sock)) != 0)
    {
        fprintf(stderr, "rank %d: nng open: %s\n", rank_g, nng_strerror(rv));
        return -1;
    }
    nng_socket_set_ms(sock, NNG_OPT_RECVTIMEO, 240000);
    {
        char url[48];
        snprintf(url, sizeof(url), "tcp://10.10.100.19:%d", broker_port_g);
        if ((rv = nng_dial(sock, url, NULL, NNG_FLAG_NONBLOCK)) != 0)
        {
            fprintf(stderr, "rank %d: nng dial: %s\n", rank_g,
                nng_strerror(rv));
            return -1;
        }
    }
    for (;;)
    {
        if ((rv = nng_recv(sock, &msg, &sz, NNG_FLAG_ALLOC)) != 0)
        {
            fprintf(stderr, "rank %d: nng survey recv: %s\n", rank_g,
                nng_strerror(rv));
            return -1;
        }
        if (sz == 4u)
        {
            nng_free(msg, sz);
            break;
        }
        if (sz == (size_t)(degree_g * ENTRY_BYTES))
        {
            memcpy(table_out, msg, (size_t)degree_g * ENTRY_BYTES);
            nng_free(msg, sz);
            nng_close(sock);
            return 0;
        }
        nng_free(msg, sz);
    }
    rank32 = (int32_t)rank_g;
    memcpy(join, &rank32, sizeof(rank32));
    memcpy(join + 4, my_entry, ENTRY_BYTES);
    if ((rv = nng_send(sock, join, sizeof(join), 0)) != 0)
    {
        fprintf(stderr, "rank %d: nng send: %s\n", rank_g, nng_strerror(rv));
        return -1;
    }
    for (;;)
    {
        if ((rv = nng_recv(sock, &msg, &sz, NNG_FLAG_ALLOC)) != 0)
        {
            fprintf(stderr, "rank %d: nng table recv: %s\n", rank_g,
                nng_strerror(rv));
            return -1;
        }
        if (sz == (size_t)(degree_g * ENTRY_BYTES))
        {
            memcpy(table_out, msg, (size_t)degree_g * ENTRY_BYTES);
            nng_free(msg, sz);
            nng_close(sock);
            return 0;
        }
        nng_free(msg, sz);
    }
}

static void open_qp(link_qp *q)
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
    {
        fprintf(stderr, "rank %d: switch device open failed\n", rank_g);
        exit(1);
    }
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
    init.cap.max_send_wr = 32;
    init.cap.max_recv_wr = 32;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    q->qp = ibv_create_qp(q->pd, &init);
    if (q->pd == 0 || q->buf == 0 || q->mr == 0 || q->cq == 0 || q->qp == 0)
    {
        fprintf(stderr, "rank %d: resource alloc failed\n", rank_g);
        exit(1);
    }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(q->qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
            IBV_QP_ACCESS_FLAGS) != 0)
    {
        fprintf(stderr, "rank %d: QP INIT failed\n", rank_g);
        exit(1);
    }
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
}

static void bring_rts(link_qp *q)
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
    {
        fprintf(stderr, "rank %d: RTR failed errno=%d\n", rank_g, errno);
        exit(1);
    }
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
    {
        fprintf(stderr, "rank %d: RTS failed errno=%d\n", rank_g, errno);
        exit(1);
    }
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
        fprintf(stderr, "rank %d: post_recv failed\n", rank_g);
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
            {
                fprintf(stderr, "rank %d: send wc %u (%s)\n", rank_g,
                    wc.status, ibv_wc_status_str(wc.status));
                ++sink_errors;
            }
            return;
        }
        if (now_us() > deadline)
        {
            fprintf(stderr, "rank %d: send drain timeout\n", rank_g);
            ++sink_errors;
            return;
        }
    }
}

static void send_chunk(link_qp *q, uint32_t chunk_index, uint32_t imm,
    int to_landing)
{
    struct ibv_send_wr wr[2];
    struct ibv_sge sge;
    struct ibv_send_wr *bad = 0;
    uint64_t local_off = (uint64_t)chunk_index * chunk_bytes_g;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)(q->buf + local_off);
    sge.length = chunk_bytes_g;
    sge.lkey = q->mr->lkey;
    memset(wr, 0, sizeof(wr));
    wr[0].wr_id = 0x11;
    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].sg_list = &sge;
    wr[0].num_sge = 1;
    wr[0].wr.rdma.remote_addr = q->remote.buf_addr +
        (to_landing != 0 ? (uint64_t)LANDING_OFF +
            (uint64_t)(imm & 15u) * chunk_bytes_g : local_off);
    wr[0].wr.rdma.rkey = q->remote.rkey;
    wr[1].wr_id = 0x11;
    wr[1].opcode = IBV_WR_SEND_WITH_IMM;
    wr[1].send_flags = IBV_SEND_SIGNALED;
    wr[1].imm_data = htonl(imm);
    wr[0].next = &wr[1];
    if (ibv_post_send(q->qp, &wr[0], &bad) != 0)
    {
        fprintf(stderr, "rank %d: post_send failed\n", rank_g);
        ++sink_errors;
    }
}

static int wait_doorbell(link_qp *q, uint32_t expect_imm)
{
    struct ibv_wc wc;
    uint64_t deadline = now_us() + 2500000ull;
    for (;;)
    {
        int n = ibv_poll_cq(q->cq, 1, &wc);
        if (n > 0)
        {
            uint32_t imm;
            if ((wc.wr_id & 0x9000000000000000ull) == 0ull)
                continue;
            post_recv(q);
            if (wc.status != IBV_WC_SUCCESS)
            {
                fprintf(stderr, "rank %d: recv wc %u (%s)\n", rank_g,
                    wc.status, ibv_wc_status_str(wc.status));
                ++sink_errors;
                return -1;
            }
            imm = ntohl(wc.imm_data);
            if (imm != expect_imm)
            {
                fprintf(stderr, "rank %d: imm mismatch got=%u expect=%u\n",
                    rank_g, imm, expect_imm);
                ++sink_errors;
                return -1;
            }
            return 0;
        }
        if (now_us() > deadline)
        {
            fprintf(stderr, "rank %d: doorbell timeout imm=%u\n", rank_g,
                expect_imm);
            ++sink_errors;
            return -1;
        }
    }
}

static uint16_t pattern_val(int r, int i)
{
    return (uint16_t)((uint32_t)(r * 251 + (i % 241) + 1) & 0x3fffu);
}

static uint16_t expected_sum(int i)
{
    int r;
    uint32_t total = 0;
    for (r = 0; r < degree_g; ++r)
        total += pattern_val(r, i);
    return (uint16_t)(total & 0xffffu);
}

int main(int argc, char **argv)
{
    int next;
    int prev;
    uint8_t my_entry[ENTRY_BYTES];
    uint8_t table[16 * ENTRY_BYTES];
    uint16_t *acc;
    const uint16_t *landing_unused;
    int i;
    int p;
    int iter;
    uint64_t total_us = 0;
    uint64_t min_us = UINT64_MAX;
    int verify_fail = 0;
    double mean_us;
    double token_serial_us;
    double b1_tok_s;
    double overlap_ceiling;
    if (argc < 5)
    {
        fprintf(stderr, "usage: mock_allreduce rank degree start broker_port\n");
        return 2;
    }
    degree_g = atoi(argv[2]);
    broker_port_g = atoi(argv[4]);
    if (degree_g != 4 && degree_g != 8 && degree_g != 16)
    {
        fprintf(stderr, "degree must be 4, 8 or 16\n");
        return 2;
    }
    chunk_elems_g = ELEMS / degree_g;
    chunk_bytes_g = (uint32_t)chunk_elems_g * 2;
    rank_g = parse_rank(argv[1]);
    if (rank_g < 0)
    {
        fprintf(stderr, "bad rank\n");
        return 2;
    }
    setvbuf(stdout, 0, _IOLBF, 0);
    next = (rank_g + 1) % degree_g;
    prev = (rank_g + degree_g - 1) % degree_g;
    open_qp(&qp_next);
    open_qp(&qp_prev);
    memset(my_entry, 0, sizeof(my_entry));
    memcpy(my_entry, &qp_next.local, sizeof(wire_info));
    memcpy(my_entry + 64, &qp_prev.local, sizeof(wire_info));
    if (broker_exchange(my_entry, table) != 0)
        return 1;
    memcpy(&qp_next.remote, table + (size_t)next * ENTRY_BYTES + 64,
        sizeof(wire_info));
    memcpy(&qp_prev.remote, table + (size_t)prev * ENTRY_BYTES,
        sizeof(wire_info));
    bring_rts(&qp_next);
    bring_rts(&qp_prev);
    for (i = 0; i < 4; ++i)
        post_recv(&qp_prev);
    acc = (uint16_t *)qp_next.buf;
    for (iter = 0; iter < ITERATIONS; ++iter)
    {
        uint64_t t0;
        uint64_t us;
        for (i = 0; i < ELEMS; ++i)
            acc[i] = pattern_val(rank_g, i);
        t0 = now_us();
        for (p = 0; p < degree_g - 1; ++p)
        {
            uint32_t imm = (uint32_t)iter * 64u + (uint32_t)p;
            const uint16_t *src = (const uint16_t *)(qp_prev.buf +
                LANDING_OFF + (size_t)(imm & 15u) * chunk_bytes_g);
            uint32_t send_index = (uint32_t)((rank_g - p + degree_g) % degree_g);
            uint32_t recv_index =
                (uint32_t)((rank_g - p - 1 + degree_g) % degree_g);
            uint16_t *dst = acc + (size_t)recv_index * chunk_elems_g;
            send_chunk(&qp_next, send_index, imm, 1);
            if (wait_doorbell(&qp_prev, imm))
                return 1;
            drain_send_cq(&qp_next);
            for (i = 0; i < chunk_elems_g; ++i)
                dst[i] = (uint16_t)(dst[i] + src[i]);
        }
        for (p = 0; p < degree_g - 1; ++p)
        {
            uint32_t imm = (uint32_t)iter * 64u +
                (uint32_t)(degree_g - 1) + (uint32_t)p;
            const uint16_t *src = (const uint16_t *)(qp_prev.buf +
                LANDING_OFF + (size_t)(imm & 15u) * chunk_bytes_g);
            uint32_t send_index = (uint32_t)((rank_g + 1 - p + degree_g) % degree_g);
            uint32_t recv_index = (uint32_t)((rank_g - p + degree_g) % degree_g);
            uint16_t *dst = acc + (size_t)recv_index * chunk_elems_g;
            send_chunk(&qp_next, send_index, imm, 1);
            if (wait_doorbell(&qp_prev, imm))
                return 1;
            drain_send_cq(&qp_next);
            memcpy(dst, src, chunk_bytes_g);
        }
        us = now_us() - t0;
        total_us += us;
        if (us < min_us)
            min_us = us;
        if ((iter % 10) == 9 || iter == ITERATIONS - 1)
        {
            for (i = 0; i < ELEMS; ++i)
            {
                if (acc[i] != expected_sum(i))
                {
                    fprintf(stderr,
                        "rank %d VERIFY FAIL iter=%d elem=%d got=%u expect=%u\n",
                        rank_g, iter, i, (unsigned)acc[i],
                        (unsigned)expected_sum(i));
                    ++verify_fail;
                    break;
                }
            }
        }
    }
    mean_us = (double)total_us / (double)ITERATIONS;
    token_serial_us = WEIGHT_MS * 1000.0 + COLLECTIVES_PER_TOKEN * mean_us;
    b1_tok_s = 1000000.0 / token_serial_us;
    overlap_ceiling = 1000.0 / WEIGHT_MS;
    printf("ALLREDUCE rank=%2d iters=%d mean_us=%.1f min_us=%.1f payload=%dB verify=%s\n",
        rank_g, ITERATIONS, mean_us, (double)min_us, ELEMS * 2,
        verify_fail != 0 ? "FAIL" : "OK");
    printf("TOKS rank=%2d per_op_us=%.1f b1_tok_s=%.1f overlap_ceiling_tok_s=%.1f weights_ms=%.0f collectives_per_token=%.0f\n",
        rank_g, mean_us, b1_tok_s, overlap_ceiling, WEIGHT_MS,
        COLLECTIVES_PER_TOKEN);
    return (verify_fail != 0 || sink_errors != 0) ? 1 : 0;
}
