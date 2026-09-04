#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ibv_context { int unused; };
struct ibv_device { int unused; };
struct ibv_pd { int unused; };
struct ibv_cq { int unused; };
struct ibv_comp_channel { int fd; };
struct ibv_qp { uint32_t qp_num; };

union ibv_gid { uint8_t raw[16]; };

struct ibv_mr
{
    void *addr;
    size_t length;
    uint32_t lkey;
    uint32_t rkey;
};

struct ibv_port_attr
{
    int state;
    uint16_t lid;
    int active_mtu;
};

struct ibv_qp_cap
{
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_send_sge;
    uint32_t max_recv_sge;
};

struct ibv_qp_init_attr
{
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    int qp_type;
    struct ibv_qp_cap cap;
};

struct ibv_global_route
{
    union ibv_gid dgid;
    uint8_t sgid_index;
    uint8_t hop_limit;
};

struct ibv_ah_attr
{
    struct ibv_global_route grh;
    uint16_t dlid;
    uint8_t sl;
    uint8_t src_path_bits;
    uint8_t port_num;
    uint8_t is_global;
};

struct ibv_qp_attr
{
    int qp_state;
    int path_mtu;
    uint32_t dest_qp_num;
    uint32_t rq_psn;
    uint32_t sq_psn;
    uint32_t max_dest_rd_atomic;
    uint32_t max_rd_atomic;
    uint32_t min_rnr_timer;
    uint32_t pkey_index;
    uint32_t port_num;
    uint32_t qp_access_flags;
    uint32_t timeout;
    uint32_t retry_cnt;
    uint32_t rnr_retry;
    struct ibv_ah_attr ah_attr;
};

struct ibv_sge
{
    uint64_t addr;
    uint32_t length;
    uint32_t lkey;
};

struct ibv_recv_wr
{
    uint64_t wr_id;
    struct ibv_recv_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
};

struct ibv_send_wr
{
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
    int opcode;
    unsigned int send_flags;
    uint32_t imm_data;
    union
    {
        struct
        {
            uint64_t remote_addr;
            uint32_t rkey;
        } rdma;
    } wr;
};

struct ibv_wc
{
    uint64_t wr_id;
    int status;
    int opcode;
    uint32_t wc_flags;
    uint32_t imm_data;
};

#define IBV_PORT_ACTIVE 4
#define IBV_QPT_RC 2
#define IBV_QPS_INIT 1
#define IBV_QPS_RTR 2
#define IBV_QPS_RTS 3
#define IBV_QPS_ERR 6
enum ibv_mtu
{
    IBV_MTU_256 = 1,
    IBV_MTU_512 = 2,
    IBV_MTU_1024 = 3,
    IBV_MTU_2048 = 4,
    IBV_MTU_4096 = 5
};
#define IBV_ACCESS_LOCAL_WRITE 1
#define IBV_ACCESS_REMOTE_WRITE 2
#define IBV_ACCESS_REMOTE_READ 4
#define IBV_QP_STATE (1 << 0)
#define IBV_QP_PKEY_INDEX (1 << 1)
#define IBV_QP_PORT (1 << 2)
#define IBV_QP_ACCESS_FLAGS (1 << 3)
#define IBV_QP_AV (1 << 4)
#define IBV_QP_PATH_MTU (1 << 5)
#define IBV_QP_DEST_QPN (1 << 6)
#define IBV_QP_RQ_PSN (1 << 7)
#define IBV_QP_MAX_DEST_RD_ATOMIC (1 << 8)
#define IBV_QP_MIN_RNR_TIMER (1 << 9)
#define IBV_QP_TIMEOUT (1 << 10)
#define IBV_QP_RETRY_CNT (1 << 11)
#define IBV_QP_RNR_RETRY (1 << 12)
#define IBV_QP_SQ_PSN (1 << 13)
#define IBV_QP_MAX_QP_RD_ATOMIC (1 << 14)
#define IBV_WR_RDMA_WRITE 0
#define IBV_WR_RDMA_WRITE_WITH_IMM 1
#define IBV_WR_SEND_WITH_IMM 2
#define IBV_SEND_SIGNALED 2
#define IBV_WC_SUCCESS 0
#define IBV_WC_RECV_RDMA_WITH_IMM 1
#define IBV_WC_SEND 2
#define IBV_WC_RECV 3
#define IBV_WC_WITH_IMM 1

struct ibv_device **ibv_get_device_list(int *count);
void ibv_free_device_list(struct ibv_device **list);
const char *ibv_get_device_name(struct ibv_device *device);
struct ibv_context *ibv_open_device(struct ibv_device *device);
int ibv_close_device(struct ibv_context *context);
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);
int ibv_dealloc_pd(struct ibv_pd *pd);
int ibv_query_port(struct ibv_context *context, uint8_t port,
    struct ibv_port_attr *attributes);
int ibv_query_gid(struct ibv_context *context, uint8_t port, int index,
    union ibv_gid *gid);
struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context);
int ibv_destroy_comp_channel(struct ibv_comp_channel *channel);
struct ibv_cq *ibv_create_cq(struct ibv_context *context, int entries,
    void *context_pointer, struct ibv_comp_channel *channel, int vector);
int ibv_destroy_cq(struct ibv_cq *cq);
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
    struct ibv_qp_init_attr *attributes);
int ibv_destroy_qp(struct ibv_qp *qp);
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attributes,
    int mask);
int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only);
int ibv_get_cq_event(struct ibv_comp_channel *channel, struct ibv_cq **cq,
    void **context_pointer);
void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int count);
int ibv_poll_cq(struct ibv_cq *cq, int entries, struct ibv_wc *completions);
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *address, size_t length,
    int access);
int ibv_dereg_mr(struct ibv_mr *mr);
int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *request,
    struct ibv_recv_wr **bad_request);
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *request,
    struct ibv_send_wr **bad_request);

#ifdef __cplusplus
}
#endif
