#pragma once

#include <endian.h>
#include <stdint.h>
#include <infiniband/verbs.h>

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    MLX5DV_OBJ_QP = 1 << 0,
    MLX5DV_OBJ_CQ = 1 << 1
};

enum
{
    MLX5_RCV_DBR = 0,
    MLX5_SND_DBR = 1
};

enum
{
    MLX5_OPCODE_RDMA_WRITE = 0x08,
    MLX5_WQE_CTRL_CQ_UPDATE = 2 << 2,
    MLX5_CQE_OWNER_MASK = 1,
    MLX5_CQE_REQ = 0,
    MLX5_CQE_REQ_ERR = 13,
    MLX5_CQE_RESP_ERR = 14,
    MLX5_CQE_INVALID = 15
};

#define MLX5_INLINE_SEG 0x80000000u

enum
{
    MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR = 0x01,
    MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR = 0x02,
    MLX5_CQE_SYNDROME_LOCAL_PROT_ERR = 0x04,
    MLX5_CQE_SYNDROME_WR_FLUSH_ERR = 0x05,
    MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR = 0x13,
    MLX5_CQE_SYNDROME_REMOTE_OP_ERR = 0x14,
    MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR = 0x15,
    MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR = 0x16
};

struct mlx5_wqe_ctrl_seg
{
    uint32_t opmod_idx_opcode;
    uint32_t qpn_ds;
    uint8_t signature;
    uint16_t dci_stream_channel_id;
    uint8_t fm_ce_se;
    uint32_t imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

struct mlx5_wqe_raddr_seg
{
    uint64_t raddr;
    uint32_t rkey;
    uint32_t reserved;
};

struct mlx5_wqe_data_seg
{
    uint32_t byte_count;
    uint32_t lkey;
    uint64_t addr;
};

struct mlx5_wqe_inl_data_seg
{
    uint32_t byte_count;
};

struct mlx5_cqe64
{
    uint8_t rsvd0[56];
    uint32_t sop_drop_qpn;
    uint16_t wqe_counter;
    uint8_t signature;
    uint8_t op_own;
};

struct mlx5_err_cqe
{
    uint8_t rsvd0[32];
    uint32_t srqn;
    uint8_t rsvd1[18];
    uint8_t vendor_err_synd;
    uint8_t syndrome;
    uint32_t s_wqe_opcode_qpn;
    uint16_t wqe_counter;
    uint8_t signature;
    uint8_t op_own;
};

struct mlx5dv_qp
{
    uint32_t *dbrec;
    struct
    {
        void *buf;
        uint32_t wqe_cnt;
        uint32_t stride;
    } sq;
    struct
    {
        void *buf;
        uint32_t wqe_cnt;
        uint32_t stride;
    } rq;
    struct
    {
        void *reg;
        uint32_t size;
    } bf;
    uint64_t comp_mask;
};

struct mlx5dv_cq
{
    void *buf;
    uint32_t *dbrec;
    uint32_t cqe_cnt;
    uint32_t cqe_size;
    void *cq_uar;
    uint32_t cqn;
    uint64_t comp_mask;
};

struct mlx5dv_obj
{
    struct
    {
        struct ibv_qp *in;
        struct mlx5dv_qp *out;
    } qp;
    struct
    {
        struct ibv_cq *in;
        struct mlx5dv_cq *out;
    } cq;
};

int mlx5dv_init_obj(struct mlx5dv_obj *obj, uint64_t obj_type);

static inline void mlx5dv_set_ctrl_seg(struct mlx5_wqe_ctrl_seg *seg, uint16_t pi,
    uint8_t opcode, uint8_t opmod, uint32_t qp_num, uint8_t fm_ce_se, uint8_t ds,
    uint8_t signature, uint32_t imm)
{
    seg->opmod_idx_opcode = htobe32(((uint32_t)opmod << 24) | ((uint32_t)pi << 8) | opcode);
    seg->qpn_ds = htobe32((qp_num << 8) | ds);
    seg->fm_ce_se = fm_ce_se;
    seg->signature = signature;
    seg->imm = imm;
}

static inline void mlx5dv_set_data_seg(struct mlx5_wqe_data_seg *seg, uint32_t length,
    uint32_t lkey, uintptr_t address)
{
    seg->byte_count = htobe32(length);
    seg->lkey = htobe32(lkey);
    seg->addr = htobe64(address);
}

static inline uint8_t mlx5dv_get_cqe_owner(struct mlx5_cqe64 *cqe)
{
    return cqe->op_own & 0x1;
}

static inline uint8_t mlx5dv_get_cqe_opcode(struct mlx5_cqe64 *cqe)
{
    return cqe->op_own >> 4;
}

void spark_stub_mlx5_scan(void);

#ifdef __cplusplus
}
#endif
