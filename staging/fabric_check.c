#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define DEGREE 16
#define PORT_BASE 58300
#define GID_INDEX 3
#define MSG_BYTES 8192
#define TEST_TIMEOUT_MS 2500

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
    int live;
} link_qp;

static int rank_g;
static int peer_fd[DEGREE];
static link_qp switch_qp[DEGREE];
static link_qp direct_qp;
static uint32_t seq_g;

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + ts.tv_nsec / 1000ull;
}

static int tcp_listen(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0 ||
        listen(fd, DEGREE + 2) != 0)
    {
        perror("listen");
        exit(1);
    }
    return fd;
}

static int tcp_connect(int peer, int port)
{
    struct sockaddr_in a;
    int fd;
    uint64_t deadline = now_us() + 20000000ull;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = inet_addr("10.10.100.10");
    a.sin_addr.s_addr = htonl(ntohl(a.sin_addr.s_addr) + peer);
    a.sin_port = htons((uint16_t)port);
    for (;;)
    {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(fd, (struct sockaddr *)&a, sizeof(a)) == 0)
            return fd;
        close(fd);
        if (now_us() > deadline)
        {
            fprintf(stderr, "rank %d: connect to %d failed\n", rank_g, peer);
            exit(1);
        }
        usleep(20000);
    }
}

static int read_full(int fd, void *buf, uint32_t bytes)
{
    uint8_t *p = buf;
    while (bytes != 0u)
    {
        ssize_t n = read(fd, p, bytes);
        if (n <= 0)
            return -1;
        p += n;
        bytes -= (uint32_t)n;
    }
    return 0;
}

static int write_full(int fd, const void *buf, uint32_t bytes)
{
    const uint8_t *p = buf;
    while (bytes != 0u)
    {
        ssize_t n = write(fd, p, bytes);
        if (n <= 0)
            return -1;
        p += n;
        bytes -= (uint32_t)n;
    }
    return 0;
}

static int open_context(link_qp *q, const char *dev)
{
    struct ibv_device **list = ibv_get_device_list(0);
    int i;
    q->ctx = 0;
    for (i = 0; list != 0 && list[i] != 0; ++i)
    {
        if (strcmp(ibv_get_device_name(list[i]), dev) == 0)
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
    q->buf = malloc(MSG_BYTES);
    q->mr = ibv_reg_mr(q->pd, q->buf, MSG_BYTES,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);
    q->cq = ibv_create_cq(q->ctx, 32, 0, 0, 0);
    return (q->pd != 0 && q->buf != 0 && q->mr != 0 && q->cq != 0) ? 0 : -1;
}

static int create_qp(link_qp *q)
{
    struct ibv_qp_init_attr init;
    struct ibv_qp_attr attr;
    memset(&init, 0, sizeof(init));
    init.send_cq = q->cq;
    init.recv_cq = q->cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 16;
    init.cap.max_recv_wr = 16;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 0;
    q->qp = ibv_create_qp(q->pd, &init);
    if (q->qp == 0)
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
    return 0;
}

static void fill_local(link_qp *q)
{
    union ibv_gid gid;
    struct ibv_port_attr port;
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
        fprintf(stderr, "rank %d: post_recv failed\n", rank_g);
}

static int poll_send(link_qp *q)
{
    struct ibv_wc wc;
    uint64_t deadline = now_us() + TEST_TIMEOUT_MS * 1000ull;
    for (;;)
    {
        int n = ibv_poll_cq(q->cq, 1, &wc);
        if (n > 0)
        {
            if (wc.status != IBV_WC_SUCCESS)
            {
                fprintf(stderr, "rank %d: send wc status %u (%s)\n",
                    rank_g, wc.status, ibv_wc_status_str(wc.status));
                return -1;
            }
            if ((wc.wr_id & 0x9000000000000000ull) != 0ull)
            {
                post_recv(q);
                return 0;
            }
            return 0;
        }
        if (now_us() > deadline)
            return -2;
    }
}

static int run_sender(link_qp *q, int dst, uint32_t seq)
{
    struct ibv_send_wr wr[2];
    struct ibv_sge sge;
    struct ibv_send_wr *bad = 0;
    uint32_t *w = (uint32_t *)q->buf;
    uint64_t t0;
    int i;
    for (i = 0; i < MSG_BYTES / 4; ++i)
        w[i] = (rank_g << 24) | (dst << 16) | (uint32_t)seq;
    w[MSG_BYTES / 4 - 1] += 1;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)q->buf;
    sge.length = MSG_BYTES;
    sge.lkey = q->mr->lkey;
    memset(wr, 0, sizeof(wr));
    wr[0].wr_id = 0x11;
    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].sg_list = &sge;
    wr[0].num_sge = 1;
    wr[0].wr.rdma.remote_addr = q->remote.buf_addr;
    wr[0].wr.rdma.rkey = q->remote.rkey;
    wr[1].wr_id = 0x11;
    wr[1].opcode = IBV_WR_SEND_WITH_IMM;
    wr[1].send_flags = IBV_SEND_SIGNALED;
    wr[1].imm_data = htonl(seq);
    wr[0].next = &wr[1];
    t0 = now_us();
    if (ibv_post_send(q->qp, &wr[0], &bad) != 0)
    {
        printf("LINK %2d->%2d rail=%s FAIL post\n", rank_g, dst,
            q == &direct_qp ? "direct" : "switch");
        return -1;
    }
    if (poll_send(q) != 0)
    {
        printf("LINK %2d->%2d rail=%s FAIL send-timeout\n", rank_g, dst,
            q == &direct_qp ? "direct" : "switch");
        return -1;
    }
    if (poll_send(q) != 0)
    {
        printf("LINK %2d->%2d rail=%s FAIL resp-timeout\n", rank_g, dst,
            q == &direct_qp ? "direct" : "switch");
        return -1;
    }
    {
        double us = (double)(now_us() - t0);
        printf("LINK %2d->%2d rail=%s OK %7.1f us\n", rank_g, dst,
            q == &direct_qp ? "direct" : "switch", us);
    }
    return 0;
}

static int run_receiver(link_qp *q, int src)
{
    struct ibv_wc wc;
    uint64_t deadline = now_us() + TEST_TIMEOUT_MS * 1000ull;
    for (;;)
    {
        int n = ibv_poll_cq(q->cq, 1, &wc);
        if (n > 0)
        {
            uint32_t *w;
            uint32_t seq;
            struct ibv_send_wr wr;
            struct ibv_send_wr *bad = 0;
            if (wc.status != IBV_WC_SUCCESS)
            {
                fprintf(stderr, "rank %d: recv wc status %u (%s)\n",
                    rank_g, wc.status, ibv_wc_status_str(wc.status));
                return -1;
            }
            post_recv(q);
            seq = ntohl(wc.imm_data);
            w = (uint32_t *)q->buf;
            if (w[0] != ((src << 24) | (rank_g << 16) | seq) ||
                w[MSG_BYTES / 4 - 1] !=
                    w[0] + 1u)
            {
                fprintf(stderr,
                    "rank %d: DATA MISMATCH from %d w0=%08x expect=%08x\n",
                    rank_g, src, w[0],
                    ((src << 24) | (rank_g << 16) | seq));
                return -1;
            }
            memset(&wr, 0, sizeof(wr));
            wr.wr_id = 0x11;
            wr.opcode = IBV_WR_SEND_WITH_IMM;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.imm_data = htonl(seq + 1u);
            if (ibv_post_send(q->qp, &wr, &bad) != 0)
                return -1;
            return 0;
        }
        if (now_us() > deadline)
            return -2;
    }
}

int main(int argc, char **argv)
{
    int listen_fd;
    int a;
    int b;
    int i;
    int ok = 0;
    int fail = 0;
    if (argc < 2)
    {
        fprintf(stderr, "usage: fabric_check rank\n");
        return 2;
    }
    rank_g = atoi(argv[1]);
    setvbuf(stdout, 0, _IOLBF, 0);
    for (i = 0; i < DEGREE; ++i)
        peer_fd[i] = -1;
    if (open_context(&direct_qp, "rocep1s0f0") != 0)
        memset(&direct_qp, 0, sizeof(direct_qp));
    listen_fd = tcp_listen(PORT_BASE + rank_g);
    for (b = rank_g + 1; b < DEGREE; ++b)
        peer_fd[b] = tcp_connect(b, PORT_BASE + b);
    for (a = 0; a < rank_g; ++a)
    {
        int fd = accept(listen_fd, 0, 0);
        int peer = -1;
        if (read_full(fd, &peer, sizeof(peer)) != 0)
        {
            fprintf(stderr, "rank %d: hello read failed\n", rank_g);
            return 1;
        }
        peer_fd[peer] = fd;
    }
    {
        int hello = rank_g;
        for (b = rank_g + 1; b < DEGREE; ++b)
            if (write_full(peer_fd[b], &hello, sizeof(hello)) != 0)
                return 1;
    }
    for (i = 0; i < DEGREE; ++i)
    {
        if (i == rank_g)
            continue;
        if (open_context(&switch_qp[i], "rocep1s0f1") != 0)
        {
            printf("rank %d: switch device open failed\n", rank_g);
            return 1;
        }
        if (create_qp(&switch_qp[i]) != 0)
            return 1;
        fill_local(&switch_qp[i]);
    }
    if (direct_qp.ctx != 0)
    {
        int mate = rank_g ^ 1;
        if (create_qp(&direct_qp) != 0)
            return 1;
        fill_local(&direct_qp);
        if (mate < rank_g)
        {
            if (write_full(peer_fd[mate], &direct_qp.local,
                    sizeof(wire_info)) != 0 ||
                read_full(peer_fd[mate], &direct_qp.remote,
                    sizeof(wire_info)) != 0)
                return 1;
        }
        else
        {
            if (read_full(peer_fd[mate], &direct_qp.remote,
                    sizeof(wire_info)) != 0 ||
                write_full(peer_fd[mate], &direct_qp.local,
                    sizeof(wire_info)) != 0)
                return 1;
        }
        direct_qp.live = bring_rts(&direct_qp) == 0;
        post_recv(&direct_qp);
    }
    for (i = 0; i < DEGREE; ++i)
    {
        if (i == rank_g)
            continue;
        if (i < rank_g)
        {
            if (write_full(peer_fd[i], &switch_qp[i].local,
                    sizeof(wire_info)) != 0 ||
                read_full(peer_fd[i], &switch_qp[i].remote,
                    sizeof(wire_info)) != 0)
                return 1;
        }
        else
        {
            if (read_full(peer_fd[i], &switch_qp[i].remote,
                    sizeof(wire_info)) != 0 ||
                write_full(peer_fd[i], &switch_qp[i].local,
                    sizeof(wire_info)) != 0)
                return 1;
        }
        switch_qp[i].live = bring_rts(&switch_qp[i]) == 0;
        post_recv(&switch_qp[i]);
    }
    for (a = 0; a < DEGREE; ++a)
    {
        for (b = a + 1; b < DEGREE; ++b)
        {
            ++seq_g;
            if (a == rank_g)
            {
                if (run_sender(&switch_qp[b], b, seq_g) == 0)
                    ++ok;
                else
                    ++fail;
            }
            else if (b == rank_g)
            {
                if (run_receiver(&switch_qp[a], a) != 0)
                    ++fail;
            }
            else
                usleep(50);
        }
    }
    for (a = 0; a < DEGREE; a += 2)
    {
        ++seq_g;
        if (a == rank_g)
        {
            if (direct_qp.live != 0)
            {
                if (run_sender(&direct_qp, a + 1, seq_g) == 0)
                    ++ok;
                else
                    ++fail;
            }
        }
        else if (a + 1 == rank_g)
        {
            if (direct_qp.live != 0)
            {
                if (run_receiver(&direct_qp, a) != 0)
                    ++fail;
            }
        }
        else
            usleep(50);
    }
    printf("RANK %2d done ok=%d fail=%d\n", rank_g, ok, fail);
    return fail != 0 ? 1 : 0;
}
