#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#define SPARK_STUB_IBV_DEVICES 2u
#define SPARK_STUB_IBV_QPS_MAX 4096u
#define SPARK_STUB_IBV_COMPLETIONS_MAX 1024u
#define SPARK_STUB_IBV_LID 0x1234u
#define SPARK_STUB_MLX5_WQES 512u
#define SPARK_STUB_MLX5_CQES 1024u
#define SPARK_STUB_MLX5_CQS 512u

typedef struct SparkStubIbvDevice
{
    struct ibv_device pub;
    char name[32];
} SparkStubIbvDevice;

typedef struct SparkStubIbvQp
{
    struct ibv_qp pub;
    int state;
    uint32_t remote_qpn;
    int live;
    struct ibv_cq *send_cq;
    uint8_t *send_queue;
    uint32_t record[2];
    uint64_t doorbell;
    uint32_t parsed;
} SparkStubIbvQp;

typedef struct SparkStubIbvCq
{
    struct ibv_cq pub;
    uint8_t *entries;
    uint32_t record[2];
    uint32_t produced;
} SparkStubIbvCq;

typedef struct SparkStubIbvCompletion
{
    uint64_t wr_id;
    int status;
} SparkStubIbvCompletion;

static SparkStubIbvDevice spark_stub_ibv_device_storage[SPARK_STUB_IBV_DEVICES];
static struct ibv_device *spark_stub_ibv_device_list[SPARK_STUB_IBV_DEVICES + 1u];
static uint32_t spark_stub_ibv_devices_ready;
static SparkStubIbvQp spark_stub_ibv_qps[SPARK_STUB_IBV_QPS_MAX];
static uint32_t spark_stub_ibv_qp_count;
static uint32_t spark_stub_ibv_next_key = 0x3100u;
static uint32_t spark_stub_ibv_rtr_class = UINT32_MAX - 1u;
static SparkStubIbvCompletion
    spark_stub_ibv_completions[SPARK_STUB_IBV_COMPLETIONS_MAX];
static uint32_t spark_stub_ibv_completion_head;
static uint32_t spark_stub_ibv_completion_tail;
static uint64_t spark_stub_ibv_modify_calls;
static uint64_t spark_stub_ibv_modify_failures;
static uint64_t spark_stub_ibv_post_sends;
static uint32_t spark_stub_ibv_fail_qpn;
static uint64_t spark_stub_ibv_next_wr_id = 1ull;
static SparkStubIbvPostedWork spark_stub_ibv_posted_work[16384];
static uint32_t spark_stub_ibv_posted_work_count;
static uint64_t spark_stub_ibv_fail_post_at;
static uint32_t spark_stub_ibv_empty_gid_queries;
static uint64_t spark_stub_ibv_gid_queries;
static SparkStubIbvCq spark_stub_ibv_cqs[SPARK_STUB_MLX5_CQS];
static uint32_t spark_stub_ibv_cq_count;
uint64_t spark_stub_mlx5_wr_id(uint32_t qpn,uint32_t index) __attribute__((weak));
int spark_stub_mlx5_cq_live(const struct ibv_cq *cq) __attribute__((weak));

static void spark_stub_ibv_init_devices(void)
{
    if (spark_stub_ibv_devices_ready != 0u)
        return;
    memset(&spark_stub_ibv_device_storage[0],0,
        sizeof(spark_stub_ibv_device_storage[0]));
    memset(&spark_stub_ibv_device_storage[1],0,
        sizeof(spark_stub_ibv_device_storage[1]));
    memcpy(spark_stub_ibv_device_storage[0].name,"rocep1s0f0",10u);
    memcpy(spark_stub_ibv_device_storage[1].name,"rocep1s0f1",10u);
    spark_stub_ibv_device_list[0] = &spark_stub_ibv_device_storage[0].pub;
    spark_stub_ibv_device_list[1] = &spark_stub_ibv_device_storage[1].pub;
    spark_stub_ibv_device_list[2] = 0;
    spark_stub_ibv_devices_ready = 1u;
}

static SparkStubIbvQp *spark_stub_ibv_qp_from_pub(struct ibv_qp *qp)
{
    return (SparkStubIbvQp *)(void *)((char *)qp -
        offsetof(SparkStubIbvQp,pub));
}

static SparkStubIbvQp *spark_stub_ibv_qp_find(uint32_t qpn)
{
    uint32_t index;
    for (index = 0u; index < spark_stub_ibv_qp_count; index++)
    {
        if (spark_stub_ibv_qps[index].live != 0 &&
            spark_stub_ibv_qps[index].pub.qp_num == qpn)
            return &spark_stub_ibv_qps[index];
    }
    return 0;
}

struct ibv_device **ibv_get_device_list(int *count)
{
    spark_stub_ibv_init_devices();
    if (count != 0)
        *count = (int)SPARK_STUB_IBV_DEVICES;
    return spark_stub_ibv_device_list;
}

void ibv_free_device_list(struct ibv_device **list)
{
    (void)list;
}

const char *ibv_get_device_name(struct ibv_device *device)
{
    SparkStubIbvDevice *stub_device;
    if (device == 0)
        return 0;
    stub_device = (SparkStubIbvDevice *)(void *)((char *)device -
        offsetof(SparkStubIbvDevice,pub));
    return stub_device->name;
}

struct ibv_context *ibv_open_device(struct ibv_device *device)
{
    (void)device;
    return (struct ibv_context *)calloc(1u,sizeof(struct ibv_context));
}

int ibv_close_device(struct ibv_context *context)
{
    free(context);
    return 0;
}

struct ibv_pd *ibv_alloc_pd(struct ibv_context *context)
{
    (void)context;
    return (struct ibv_pd *)calloc(1u,sizeof(struct ibv_pd));
}

int ibv_dealloc_pd(struct ibv_pd *pd)
{
    free(pd);
    return 0;
}

int ibv_query_port(struct ibv_context *context, uint8_t port,
    struct ibv_port_attr *attributes)
{
    (void)context;
    if (port != 1u || attributes == 0)
    {
        errno = EINVAL;
        return -1;
    }
    memset(attributes,0,sizeof(*attributes));
    attributes->state = IBV_PORT_ACTIVE;
    attributes->lid = SPARK_STUB_IBV_LID;
    attributes->active_mtu = IBV_MTU_4096;
    return 0;
}

int ibv_query_gid(struct ibv_context *context, uint8_t port, int index,
    union ibv_gid *gid)
{
    (void)context;
    if (port != 1u || gid == 0)
    {
        errno = EINVAL;
        return -1;
    }
    spark_stub_ibv_gid_queries++;
    memset(gid->raw,0,sizeof(gid->raw));
    if (spark_stub_ibv_empty_gid_queries != 0u)
    {
        spark_stub_ibv_empty_gid_queries--;
        return 0;
    }
    gid->raw[10] = 0xffu;
    gid->raw[11] = 0xffu;
    gid->raw[12] = 10u;
    gid->raw[13] = (uint8_t)(100 + index);
    gid->raw[14] = 0u;
    gid->raw[15] = 1u;
    return 0;
}

struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context)
{
    (void)context;
    return (struct ibv_comp_channel *)
        calloc(1u,sizeof(struct ibv_comp_channel));
}

int ibv_destroy_comp_channel(struct ibv_comp_channel *channel)
{
    free(channel);
    return 0;
}

struct ibv_cq *ibv_create_cq(struct ibv_context *context, int entries,
    void *context_pointer, struct ibv_comp_channel *channel, int vector)
{
    (void)context;
    (void)entries;
    (void)context_pointer;
    (void)channel;
    (void)vector;
    if ( spark_stub_ibv_cq_count >= SPARK_STUB_MLX5_CQS )
    {
        errno = ENOMEM;
        return 0;
    }
    memset(&spark_stub_ibv_cqs[spark_stub_ibv_cq_count],0,sizeof(spark_stub_ibv_cqs[0]));
    return &spark_stub_ibv_cqs[spark_stub_ibv_cq_count++].pub;
}

int ibv_destroy_cq(struct ibv_cq *cq)
{
    (void)cq;
    return 0;
}

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
    struct ibv_qp_init_attr *attributes)
{
    SparkStubIbvQp *qp;
    (void)pd;
    if (spark_stub_ibv_qp_count >= SPARK_STUB_IBV_QPS_MAX)
    {
        errno = ENOMEM;
        return 0;
    }
    qp = &spark_stub_ibv_qps[spark_stub_ibv_qp_count];
    qp->pub.qp_num = spark_stub_ibv_qp_count + 1u;
    qp->state = IBV_QPS_RESET;
    qp->remote_qpn = 0u;
    qp->live = 1;
    qp->send_cq = attributes != 0 ? attributes->send_cq : 0;
    qp->send_queue = 0;
    qp->record[0] = qp->record[1] = 0u;
    qp->parsed = 0u;
    spark_stub_ibv_qp_count++;
    return &qp->pub;
}

int ibv_destroy_qp(struct ibv_qp *qp)
{
    SparkStubIbvQp *stub_qp;
    if (qp == 0)
        return 0;
    stub_qp = spark_stub_ibv_qp_from_pub(qp);
    stub_qp->live = 0;
    return 0;
}

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attributes,
    int mask)
{
    SparkStubIbvQp *stub_qp;
    int next_state;
    if (qp == 0 || attributes == 0)
    {
        errno = EINVAL;
        return -1;
    }
    spark_stub_ibv_modify_calls++;
    stub_qp = spark_stub_ibv_qp_from_pub(qp);
    if ((mask & IBV_QP_STATE) == 0)
        return 0;
    next_state = attributes->qp_state;
    if (next_state == IBV_QPS_RESET)
    {
        spark_stub_mlx5_scan();
        stub_qp->state = IBV_QPS_RESET;
        stub_qp->remote_qpn = 0u;
        stub_qp->record[0] = stub_qp->record[1] = 0u;
        stub_qp->parsed = 0u;
        return 0;
    }
    if (next_state == IBV_QPS_INIT)
    {
        if (stub_qp->state != IBV_QPS_RESET && stub_qp->state != IBV_QPS_INIT)
        {
            spark_stub_ibv_modify_failures++;
            errno = EINVAL;
            return -1;
        }
        stub_qp->state = IBV_QPS_INIT;
        return 0;
    }
    if (next_state == IBV_QPS_RTR)
    {
        if (stub_qp->state != IBV_QPS_INIT && stub_qp->state != IBV_QPS_RTR)
        {
            spark_stub_ibv_modify_failures++;
            errno = EINVAL;
            return -1;
        }
        if ((mask & IBV_QP_DEST_QPN) != 0 &&
            spark_stub_ibv_fail_qpn != 0u &&
            attributes->dest_qp_num == spark_stub_ibv_fail_qpn)
        {
            spark_stub_ibv_modify_failures++;
            errno = EINVAL;
            return -1;
        }
        if ((mask & IBV_QP_DEST_QPN) != 0)
            stub_qp->remote_qpn = attributes->dest_qp_num;
        if ((mask & IBV_QP_AV) != 0)
            spark_stub_ibv_rtr_class = spark_stub_ibv_rtr_class == UINT32_MAX - 1u ||
                spark_stub_ibv_rtr_class == attributes->ah_attr.grh.traffic_class ?
                attributes->ah_attr.grh.traffic_class : UINT32_MAX;
        stub_qp->state = IBV_QPS_RTR;
        return 0;
    }
    if (next_state == IBV_QPS_RTS)
    {
        if (stub_qp->state != IBV_QPS_RTR && stub_qp->state != IBV_QPS_RTS)
        {
            spark_stub_ibv_modify_failures++;
            errno = EINVAL;
            return -1;
        }
        stub_qp->state = IBV_QPS_RTS;
        return 0;
    }
    stub_qp->state = next_state;
    return 0;
}

int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attributes,
    int mask, struct ibv_qp_init_attr *initial)
{
    SparkStubIbvQp *stub;
    (void)mask;
    if ( qp == 0 || attributes == 0 || initial == 0 )
    {
        errno = EINVAL;
        return -1;
    }
    stub = spark_stub_ibv_qp_from_pub(qp);
    if ( stub->live == 0 )
    {
        errno = EINVAL;
        return -1;
    }
    memset(attributes,0,sizeof(*attributes));
    memset(initial,0,sizeof(*initial));
    attributes->qp_state = stub->state;
    attributes->dest_qp_num = stub->remote_qpn;
    return 0;
}

int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only)
{
    (void)cq;
    (void)solicited_only;
    return 0;
}

int ibv_get_cq_event(struct ibv_comp_channel *channel, struct ibv_cq **cq,
    void **context_pointer)
{
    (void)channel;
    (void)cq;
    (void)context_pointer;
    errno = ENOSYS;
    return -1;
}

void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int count)
{
    (void)cq;
    (void)count;
}

int ibv_poll_cq(struct ibv_cq *cq, int entries, struct ibv_wc *completions)
{
    int delivered;
    (void)cq;
    if (entries <= 0 || completions == 0)
    {
        errno = EINVAL;
        return -1;
    }
    delivered = 0;
    while (delivered < entries &&
        spark_stub_ibv_completion_head != spark_stub_ibv_completion_tail)
    {
        SparkStubIbvCompletion *slot =
            &spark_stub_ibv_completions[
                spark_stub_ibv_completion_head %
                SPARK_STUB_IBV_COMPLETIONS_MAX];
        memset(&completions[delivered],0,sizeof(completions[delivered]));
        completions[delivered].wr_id = slot->wr_id;
        completions[delivered].status = slot->status;
        completions[delivered].opcode = IBV_WC_SEND;
        spark_stub_ibv_completion_head = (spark_stub_ibv_completion_head + 1u) %
            SPARK_STUB_IBV_COMPLETIONS_MAX;
        delivered++;
    }
    return delivered;
}

struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *address, size_t length,
    int access)
{
    struct ibv_mr *mr;
    (void)pd;
    (void)access;
    mr = (struct ibv_mr *)calloc(1u,sizeof(struct ibv_mr));
    if (mr == 0)
        return 0;
    mr->addr = address;
    mr->length = length;
    mr->lkey = spark_stub_ibv_next_key;
    mr->rkey = spark_stub_ibv_next_key;
    spark_stub_ibv_next_key += 0x10u;
    return mr;
}

int ibv_dereg_mr(struct ibv_mr *mr)
{
    free(mr);
    return 0;
}

int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *request,
    struct ibv_recv_wr **bad_request)
{
    (void)qp;
    (void)request;
    (void)bad_request;
    return 0;
}

int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *request,
    struct ibv_send_wr **bad_request)
{
    SparkStubIbvPostedWork *work;
    if ( qp == 0 || request == 0 || request->num_sge != 1 || request->sg_list == 0 )
    {
        errno = EINVAL;
        return -1;
    }
    spark_stub_ibv_post_sends++;
    if ( spark_stub_ibv_post_sends == spark_stub_ibv_fail_post_at ||
         spark_stub_ibv_posted_work_count >= 16384u )
    {
        if ( bad_request != 0 ) *bad_request = request;
        errno = EIO;
        return -1;
    }
    work = &spark_stub_ibv_posted_work[spark_stub_ibv_posted_work_count++];
    work->wr_id = request->wr_id;
    work->source = request->sg_list[0].addr;
    work->remote = request->wr.rdma.remote_addr;
    work->length = request->sg_list[0].length;
    work->qp_number = qp->qp_num;
    work->flags = request->send_flags;
    work->lkey = request->sg_list[0].lkey;
    return 0;
}

int mlx5dv_init_obj(struct mlx5dv_obj *obj, uint64_t obj_type)
{
    if ( obj == 0 )
    {
        errno = EINVAL;
        return -1;
    }
    if ( (obj_type & MLX5DV_OBJ_QP) != 0u )
    {
        SparkStubIbvQp *qp = spark_stub_ibv_qp_from_pub(obj->qp.in);
        if ( qp->send_queue == 0 && (qp->send_queue = (uint8_t *)calloc(SPARK_STUB_MLX5_WQES,64u)) == 0 )
        {
            errno = ENOMEM;
            return -1;
        }
        memset(obj->qp.out,0,sizeof(*obj->qp.out));
        obj->qp.out->dbrec = qp->record;
        obj->qp.out->sq.buf = qp->send_queue;
        obj->qp.out->sq.wqe_cnt = SPARK_STUB_MLX5_WQES;
        obj->qp.out->sq.stride = 64u;
        obj->qp.out->bf.reg = &qp->doorbell;
        obj->qp.out->bf.size = 0u;
    }
    if ( (obj_type & MLX5DV_OBJ_CQ) != 0u )
    {
        SparkStubIbvCq *cq = (SparkStubIbvCq *)obj->cq.in;
        uint32_t index;
        if ( cq->entries == 0 )
        {
            if ( (cq->entries = (uint8_t *)calloc(SPARK_STUB_MLX5_CQES,64u)) == 0 )
            {
                errno = ENOMEM;
                return -1;
            }
            for ( index = 0u; index < SPARK_STUB_MLX5_CQES; index++ )
                ((struct mlx5_cqe64 *)(cq->entries + (uint64_t)index * 64u))->op_own = (uint8_t)(MLX5_CQE_INVALID << 4);
        }
        memset(obj->cq.out,0,sizeof(*obj->cq.out));
        obj->cq.out->buf = cq->entries;
        obj->cq.out->dbrec = cq->record;
        obj->cq.out->cqe_cnt = SPARK_STUB_MLX5_CQES;
        obj->cq.out->cqe_size = 64u;
    }
    return 0;
}

void spark_stub_mlx5_scan(void)
{
    uint32_t q;
    for ( q = 0u; q < spark_stub_ibv_qp_count; q++ )
    {
        SparkStubIbvQp *qp = &spark_stub_ibv_qps[q];
        uint32_t producer = be32toh(qp->record[MLX5_SND_DBR]) & 0xffffu,advanced = 0u;
        if ( qp->send_queue == 0 )
            continue;
        while ( (qp->parsed & 0xffffu) != producer && spark_stub_ibv_posted_work_count < 16384u )
        {
            uint32_t index = qp->parsed & (SPARK_STUB_MLX5_WQES - 1u);
            const struct mlx5_wqe_ctrl_seg *control = (const struct mlx5_wqe_ctrl_seg *)(qp->send_queue + (uint64_t)index * 64u);
            const struct mlx5_wqe_raddr_seg *remote = (const struct mlx5_wqe_raddr_seg *)(control + 1);
            const struct mlx5_wqe_data_seg *data = (const struct mlx5_wqe_data_seg *)(remote + 1);
            uint32_t count = be32toh(data->byte_count);
            SparkStubIbvPostedWork *work = &spark_stub_ibv_posted_work[spark_stub_ibv_posted_work_count++];
            memset(work,0,sizeof(*work));
            work->qp_number = be32toh(control->qpn_ds) >> 8;
            work->wqe_index = qp->parsed & 0xffffu;
            work->remote = be64toh(remote->raddr);
            work->flags = (control->fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE) != 0u ? IBV_SEND_SIGNALED : 0u;
            if ( (count & MLX5_INLINE_SEG) != 0u )
            {
                work->flags |= IBV_SEND_INLINE;
                work->length = count & ~MLX5_INLINE_SEG;
                memcpy(&work->inline_word,(const uint8_t *)data + 4u,sizeof(work->inline_word));
            }
            else
            {
                work->length = count;
                work->lkey = be32toh(data->lkey);
                work->source = be64toh(data->addr);
            }
            work->wr_id = spark_stub_mlx5_wr_id != 0 ? spark_stub_mlx5_wr_id(work->qp_number,work->wqe_index) : 0u;
            qp->parsed++;
            advanced = 1u;
        }
        spark_stub_ibv_post_sends += advanced;
    }
}

uint32_t spark_stub_ibv_posted_count(void)
{
    spark_stub_mlx5_scan();
    return spark_stub_ibv_posted_work_count;
}

int spark_stub_ibv_posted(uint32_t index, SparkStubIbvPostedWork *work)
{
    spark_stub_mlx5_scan();
    if ( index >= spark_stub_ibv_posted_work_count || work == 0 ) return -1;
    *work = spark_stub_ibv_posted_work[index];
    return 0;
}

static uint8_t spark_stub_mlx5_syndrome(int status)
{
    switch ( status )
    {
        case IBV_WC_WR_FLUSH_ERR: return MLX5_CQE_SYNDROME_WR_FLUSH_ERR;
        case IBV_WC_RETRY_EXC_ERR: return MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR;
        case IBV_WC_RNR_RETRY_EXC_ERR: return MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR;
        case IBV_WC_REM_ACCESS_ERR: return MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR;
        default: return 0xffu;
    }
}

static int spark_stub_mlx5_produce(SparkStubIbvCq *cq,uint32_t qpn,uint32_t wqe_index,int status)
{
    struct mlx5_cqe64 *entry;
    uint32_t consumed;
    if ( cq == 0 || cq->entries == 0 ) return -1;
    consumed = be32toh(cq->record[0]) & 0xffffffu;
    if ( ((cq->produced - consumed) & 0xffffffu) >= SPARK_STUB_MLX5_CQES ) return -1;
    entry = (struct mlx5_cqe64 *)(cq->entries + (uint64_t)(cq->produced & (SPARK_STUB_MLX5_CQES - 1u)) * 64u);
    memset(entry,0,sizeof(*entry) - 1u);
    entry->sop_drop_qpn = htobe32(qpn & 0xffffffu);
    entry->wqe_counter = htobe16((uint16_t)wqe_index);
    if ( status != IBV_WC_SUCCESS )
    {
        ((struct mlx5_err_cqe *)entry)->syndrome = spark_stub_mlx5_syndrome(status);
        ((struct mlx5_err_cqe *)entry)->vendor_err_synd = 0x81u;
    }
    __sync_synchronize();
    entry->op_own = (uint8_t)(((status == IBV_WC_SUCCESS ? MLX5_CQE_REQ : MLX5_CQE_REQ_ERR) << 4) |
        ((cq->produced & SPARK_STUB_MLX5_CQES) != 0u ? 1u : 0u));
    cq->produced++;
    return 0;
}

int spark_stub_ibv_complete(uint64_t work_id, int status)
{
    uint32_t index;
    spark_stub_mlx5_scan();
    for ( index = spark_stub_ibv_posted_work_count; index-- > 0u; )
        if ( spark_stub_ibv_posted_work[index].wr_id == work_id )
        {
            SparkStubIbvQp *qp = spark_stub_ibv_qp_find(spark_stub_ibv_posted_work[index].qp_number);
            if ( qp != 0 && spark_stub_mlx5_cq_live != 0 && spark_stub_mlx5_cq_live(qp->send_cq) == 0 )
                return 0;
            return qp != 0 ? spark_stub_mlx5_produce((SparkStubIbvCq *)qp->send_cq,qp->pub.qp_num,spark_stub_ibv_posted_work[index].wqe_index,status) : -1;
        }
    return -1;
}

void spark_stub_ibv_fail_post_call(uint64_t call)
{
    spark_stub_ibv_fail_post_at = call;
}

struct ibv_mr *ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
    size_t length, uint64_t iova, int fd, int access)
{
    (void)pd; (void)offset; (void)length; (void)iova; (void)fd; (void)access;
    errno = ENOTSUP;
    return 0;
}

void spark_stub_ibv_reset(void)
{
    spark_stub_ibv_completion_head = 0u;
    spark_stub_ibv_completion_tail = 0u;
    spark_stub_ibv_modify_calls = 0ull;
    spark_stub_ibv_modify_failures = 0ull;
    spark_stub_ibv_post_sends = 0ull;
    spark_stub_ibv_fail_qpn = 0u;
    spark_stub_ibv_fail_post_at = 0u;
    spark_stub_ibv_posted_work_count = 0u;
    spark_stub_ibv_empty_gid_queries = 0u;
}

void spark_stub_ibv_empty_gids(uint32_t queries)
{
    spark_stub_ibv_empty_gid_queries = queries;
}

uint64_t spark_stub_ibv_gid_query_count(void)
{
    return spark_stub_ibv_gid_queries;
}

uint64_t spark_stub_ibv_modify_qp_calls(void)
{
    return spark_stub_ibv_modify_calls;
}

void spark_stub_ibv_rtr_traffic_class_clear(void)
{
    spark_stub_ibv_rtr_class = UINT32_MAX - 1u;
}

uint32_t spark_stub_ibv_rtr_traffic_class(void)
{
    return spark_stub_ibv_rtr_class;
}

uint64_t spark_stub_ibv_modify_qp_failures(void)
{
    return spark_stub_ibv_modify_failures;
}

uint64_t spark_stub_ibv_post_send_calls(void)
{
    spark_stub_mlx5_scan();
    return spark_stub_ibv_post_sends;
}

void spark_stub_ibv_fail_modify_qp_for_qpn(uint32_t remote_qpn)
{
    spark_stub_ibv_fail_qpn = remote_qpn;
}

void spark_stub_ibv_poll_cq_inject(int status, uint32_t count)
{
    SparkStubIbvCq *cq = &spark_stub_ibv_cqs[0];
    uint32_t index;
    for (index = 0u; index < spark_stub_ibv_cq_count; index++)
        if ( spark_stub_mlx5_cq_live != 0 && spark_stub_mlx5_cq_live(&spark_stub_ibv_cqs[index].pub) != 0 )
        {
            cq = &spark_stub_ibv_cqs[index];
            break;
        }
    for (index = 0u; index < count; index++)
        if ( spark_stub_mlx5_produce(cq,0xffffffu,(uint32_t)spark_stub_ibv_next_wr_id++,status) != 0 )
            break;
}

void spark_stub_ibv_set_qp_state(uint32_t qpn, int state)
{
    SparkStubIbvQp *qp = spark_stub_ibv_qp_find(qpn);
    if (qp != 0)
        qp->state = state;
}

int spark_stub_ibv_qp_state(uint32_t qpn)
{
    SparkStubIbvQp *qp = spark_stub_ibv_qp_find(qpn);
    return qp != 0 ? qp->state : -1;
}

int spark_stub_ibv_qp_remote_qpn(uint32_t qpn)
{
    SparkStubIbvQp *qp = spark_stub_ibv_qp_find(qpn);
    return qp != 0 ? (int)qp->remote_qpn : -1;
}
