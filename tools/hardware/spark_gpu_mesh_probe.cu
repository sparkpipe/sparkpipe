#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define MESH_MAX_RANKS 16u
#define MESH_MODE_RELAY 0u
#define MESH_MODE_PROXY 1u
#define MESH_MODE_STREAM 2u
#define MESH_GRAPH_EXCHANGES 50u
#define MESH_RING_EACH 0u
#define MESH_RING_ONCE 1u
#define MESH_RING_BLUEFLAME 2u
#define MESH_FAIL(...) do { std::fprintf(stderr,"PROBE-FAIL " __VA_ARGS__); std::fprintf(stderr," errno=%d\n",errno); std::exit(2); } while (0)
#define MESH_CUDA(call) do { cudaError_t status_=(call); if (status_ != cudaSuccess) MESH_FAIL("%s cuda=%d %s",#call,(int)status_,cudaGetErrorString(status_)); } while (0)

struct MeshEndpoint
{
    uint32_t qpn;
    uint32_t rkey;
    uint64_t address;
    uint8_t gid[16];
};

struct MeshQueue
{
    uint8_t *sq;
    uint32_t *dbrec;
    uint64_t remote;
    uint32_t wqe_mask;
    uint32_t qpn;
    uint32_t rkey;
    uint32_t peer;
};

struct MeshShared
{
    volatile uint64_t producer;
    uint64_t pad0[7];
    volatile uint64_t gate;
    uint64_t pad1[7];
    volatile uint64_t completed;
    uint64_t pad2[7];
    volatile uint64_t ready;
    uint64_t pad3[7];
};

struct MeshKernel
{
    MeshQueue queues[MESH_MAX_RANKS];
    uint8_t *send;
    uint8_t *receive;
    MeshShared *shared;
    uint64_t *samples;
    uint32_t *result;
    uint64_t timeout_ns;
    uint32_t lkey;
    uint32_t rank;
    uint32_t ranks;
    uint32_t peers;
    uint32_t bytes;
    uint32_t slots;
    uint32_t exchanges;
    uint32_t mode;
};

struct MeshOptions
{
    uint32_t rank;
    uint32_t ranks;
    uint32_t bytes;
    uint32_t exchanges;
    uint32_t warmup;
    uint32_t slots;
    uint32_t mode;
    uint32_t ring;
    uint32_t guard;
    uint32_t gid_index;
    uint32_t traffic_class;
    uint32_t port;
    int cpu;
    const char *device;
    const char *coordinator;
};

struct MeshHost
{
    const MeshOptions *options;
    MeshShared *shared;
    ibv_cq *cq;
    ibv_qp *qps[MESH_MAX_RANKS];
    ibv_mr *send_mr;
    uint8_t *send;
    uint8_t *receive;
    volatile uint64_t *doorbells[MESH_MAX_RANKS];
    const uint8_t *sq[MESH_MAX_RANKS];
    uint32_t wqe_mask[MESH_MAX_RANKS];
    uint32_t bf_size[MESH_MAX_RANKS];
    uint32_t bf_share[MESH_MAX_RANKS];
    uint32_t bf_offset[MESH_MAX_RANKS];
    uint64_t busy_ticks;
    uint64_t busy_count;
    MeshEndpoint remote[MESH_MAX_RANKS];
    uint32_t peer_of[MESH_MAX_RANKS];
    uint32_t peers;
    std::atomic<bool> stop;
    std::atomic<uint64_t> completions;
    std::atomic<uint64_t> errors;
};

static __host__ __device__ __forceinline__ uint64_t MeshWord(uint32_t exchange,uint32_t sender,uint32_t word)
{
    return ((uint64_t)(exchange + 1u) << 32) | ((uint64_t)sender << 24) | (word & 0xffffffu);
}

static __device__ __forceinline__ uint64_t MeshNow(void)
{
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

static __device__ __forceinline__ uint32_t MeshBe32(uint32_t value)
{
    return __byte_perm(value,0u,0x0123u);
}

static __device__ __forceinline__ uint64_t MeshBe64(uint64_t value)
{
    return ((uint64_t)MeshBe32((uint32_t)value) << 32) | MeshBe32((uint32_t)(value >> 32));
}

static __device__ __forceinline__ uint64_t MeshAcquire(const volatile uint64_t *address)
{
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

static __device__ void MeshPostQueue(const MeshKernel &k,const MeshQueue &q,uint32_t exchange,uint32_t slot)
{
    uint4 *wqe = (uint4 *)(q.sq + (uint64_t)(exchange & q.wqe_mask) * 64u);
    uint32_t control0 = MeshBe32(((exchange & 0xffffu) << 8) | 0x08u);
    uint32_t control1 = MeshBe32((q.qpn << 8) | 3u);
    uint64_t remote = MeshBe64(q.remote + ((uint64_t)slot * k.ranks + k.rank) * k.bytes);
    uint64_t local = MeshBe64((uint64_t)(uintptr_t)(k.send + (uint64_t)slot * k.bytes));
    wqe[0] = make_uint4(control0,control1,0x08000000u,0u);
    wqe[1] = make_uint4((uint32_t)remote,(uint32_t)(remote >> 32),MeshBe32(q.rkey),0u);
    wqe[2] = make_uint4(MeshBe32(k.bytes),MeshBe32(k.lkey),(uint32_t)local,(uint32_t)(local >> 32));
    __threadfence_system();
    *(volatile uint32_t *)(q.dbrec + 1) = MeshBe32((exchange + 1u) & 0xffffu);
}

static __global__ void MeshExchangeKernel(MeshKernel k)
{
    __shared__ uint32_t failed;
    __shared__ uint32_t mismatch;
    __shared__ uint64_t posted;
    __shared__ uint64_t arrival[MESH_MAX_RANKS];
    uint32_t t = threadIdx.x,words = k.bytes / 8u,per_peer = (words - 1u + 511u) / 512u + 1u;
    if ( t == 0u ) { failed = 0u; mismatch = 0u; }
    __syncthreads();
    for ( uint32_t e = 0u; e < k.exchanges; e++ )
    {
        uint32_t slot = e % k.slots;
        uint64_t started = MeshNow();
        if ( t == 0u && e >= k.slots && MeshAcquire(&k.shared->completed) < (uint64_t)(e - k.slots + 1u) )
        {
            k.result[4]++;
            while ( MeshAcquire(&k.shared->completed) < (uint64_t)(e - k.slots + 1u) )
                if ( MeshNow() - started > k.timeout_ns ) { failed = 3u; break; }
        }
        __syncthreads();
        if ( failed != 0u ) { if ( t == 0u ) k.result[2] = e; break; }
        volatile uint64_t *out = (volatile uint64_t *)(k.send + (uint64_t)slot * k.bytes);
        for ( uint32_t w = t; w < words; w += blockDim.x ) out[w] = MeshWord(e,k.rank,w);
        __threadfence_system();
        __syncthreads();
        uint64_t filled = MeshNow();
        if ( k.mode == MESH_MODE_PROXY )
        {
            if ( t < k.peers ) MeshPostQueue(k,k.queues[t],e,slot);
            __threadfence_system();
            __syncthreads();
            if ( t == 0u ) { k.shared->producer = e + 1u; __threadfence_system(); posted = MeshNow(); }
            if ( t < k.peers )
            {
                uint32_t peer = k.queues[t].peer;
                const volatile uint64_t *tag = (const volatile uint64_t *)(k.receive + ((uint64_t)slot * k.ranks + peer) * k.bytes) + words - 1u;
                uint64_t expected = MeshWord(e,peer,words - 1u);
                while ( MeshAcquire(tag) != expected )
                    if ( MeshNow() - started > k.timeout_ns ) { failed = 2u; k.result[3] = peer; break; }
                arrival[t] = MeshNow();
            }
        }
        else if ( t == 0u )
        {
            k.shared->producer = e + 1u;
            __threadfence_system();
            posted = MeshNow();
            while ( MeshAcquire(&k.shared->gate) < (uint64_t)(e + 1u) )
                if ( MeshNow() - started > k.timeout_ns ) { failed = 2u; break; }
            for ( uint32_t j = 0u; j < k.peers; j++ ) arrival[j] = MeshNow();
        }
        __syncthreads();
        if ( failed != 0u ) { if ( t == 0u ) k.result[2] = e; break; }
        uint64_t done = MeshNow();
        for ( uint32_t i = t; i < k.peers * per_peer; i += blockDim.x )
        {
            uint32_t peer = k.queues[i / per_peer].peer,s = i % per_peer;
            uint32_t w = s + 1u < per_peer ? s * 512u : (words >= 2u ? words - 2u : 0u);
            const volatile uint64_t *data = (const volatile uint64_t *)(k.receive + ((uint64_t)slot * k.ranks + peer) * k.bytes);
            if ( data[w] != MeshWord(e,peer,w) ) atomicAdd(&mismatch,1u);
        }
        if ( t == 0u )
        {
            uint64_t first = arrival[0],last = arrival[0];
            for ( uint32_t j = 1u; j < k.peers; j++ ) { first = min(first,arrival[j]); last = max(last,arrival[j]); }
            k.samples[5u * e] = done - started;
            k.samples[5u * e + 1u] = done - filled;
            k.samples[5u * e + 2u] = posted - filled;
            k.samples[5u * e + 3u] = first > posted ? first - posted : 0u;
            k.samples[5u * e + 4u] = last - first;
        }
        __syncthreads();
    }
    if ( t == 0u ) { k.result[0] = failed == 0u ? 1u : failed; k.result[1] = mismatch; }
}

static __global__ void MeshPublishKernel(MeshKernel k,uint32_t *cursor)
{
    __shared__ uint32_t failed;
    uint32_t t = threadIdx.x,e = *cursor,words = k.bytes / 8u,slot = e % k.slots;
    if ( t == 0u ) failed = 0u;
    __syncthreads();
    uint64_t started = MeshNow();
    if ( t == 0u && e >= k.slots && MeshAcquire(&k.shared->completed) < (uint64_t)(e - k.slots + 1u) )
    {
        k.result[4]++;
        while ( MeshAcquire(&k.shared->completed) < (uint64_t)(e - k.slots + 1u) )
            if ( MeshNow() - started > k.timeout_ns ) { failed = 3u; k.result[0] = 3u; k.result[2] = e; break; }
    }
    __syncthreads();
    if ( failed != 0u ) return;
    volatile uint64_t *out = (volatile uint64_t *)(k.send + (uint64_t)slot * k.bytes);
    for ( uint32_t w = t; w < words; w += blockDim.x ) out[w] = MeshWord(e,k.rank,w);
    __threadfence_system();
    __syncthreads();
    if ( t == 0u )
    {
        uint64_t filled = MeshNow();
        k.shared->ready = 0u;
        __threadfence_system();
        k.shared->producer = e + 1u;
        __threadfence_system();
        k.samples[5u * e] = started;
        k.samples[5u * e + 2u] = filled;
        k.samples[5u * e + 3u] = MeshNow();
    }
}

static __global__ void MeshPollKernel(MeshShared *shared,uint32_t *result,uint64_t timeout_ns)
{
    uint64_t started = MeshNow();
    while ( MeshAcquire(&shared->ready) != 1u )
        if ( MeshNow() - started > timeout_ns ) { result[0] = 2u; break; }
}

static __global__ void MeshGuardKernel(const MeshShared *shared,uint32_t *result)
{
    if ( shared->ready != 1u ) result[5]++;
}

static __global__ void MeshConsumeKernel(MeshKernel k,uint32_t *cursor)
{
    __shared__ uint32_t mismatch;
    uint64_t done = MeshNow();
    uint32_t t = threadIdx.x,e = *cursor,words = k.bytes / 8u,slot = e % k.slots,per_peer = (words - 1u + 511u) / 512u + 1u;
    if ( t == 0u ) mismatch = 0u;
    __syncthreads();
    for ( uint32_t i = t; i < k.peers * per_peer; i += blockDim.x )
    {
        uint32_t peer = k.queues[i / per_peer].peer,s = i % per_peer;
        uint32_t w = s + 1u < per_peer ? s * 512u : (words >= 2u ? words - 2u : 0u);
        const volatile uint64_t *data = (const volatile uint64_t *)(k.receive + ((uint64_t)slot * k.ranks + peer) * k.bytes);
        if ( data[w] != MeshWord(e,peer,w) ) atomicAdd(&mismatch,1u);
    }
    __syncthreads();
    if ( t == 0u )
    {
        uint64_t started = k.samples[5u * e],filled = k.samples[5u * e + 2u],posted = k.samples[5u * e + 3u];
        k.samples[5u * e] = done - started;
        k.samples[5u * e + 1u] = done - filled;
        k.samples[5u * e + 2u] = posted - filled;
        k.samples[5u * e + 3u] = done - posted;
        k.samples[5u * e + 4u] = 0u;
        k.result[1] += mismatch;
        *cursor = e + 1u;
        if ( e + 1u == k.exchanges && k.result[0] == 0u ) k.result[0] = 1u;
    }
}

static void MeshDriver(CUresult status,const char *what)
{
    if ( status != CUDA_SUCCESS ) MESH_FAIL("%s cu=%d",what,(int)status);
}

static void MeshCaptureWait(cudaStream_t stream,MeshShared *shared)
{
    CUstreamCaptureStatus capture;
    cuuint64_t id;
    CUgraph graph;
    const CUgraphNode *dependencies;
    size_t dependency_count;
    MeshDriver(cuStreamGetCaptureInfo((CUstream)stream,&capture,&id,&graph,&dependencies,0,&dependency_count),"capture-info");
    CUstreamBatchMemOpParams operation = {};
    operation.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    operation.waitValue.address = (CUdeviceptr)(uintptr_t)&shared->ready;
    operation.waitValue.value64 = 1u;
    operation.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    CUDA_BATCH_MEM_OP_NODE_PARAMS params = {};
    MeshDriver(cuCtxGetCurrent(&params.ctx),"capture-context");
    params.count = 1u;
    params.paramArray = &operation;
    CUgraphNode wait;
    MeshDriver(cuGraphAddBatchMemOpNode(&wait,graph,dependencies,dependency_count,&params),"capture-wait");
    MeshDriver(cuStreamUpdateCaptureDependencies((CUstream)stream,&wait,0,1u,CU_STREAM_SET_CAPTURE_DEPENDENCIES),"capture-dependency");
}

static void MeshSendAll(int fd,const void *data,size_t bytes)
{
    const uint8_t *cursor = (const uint8_t *)data;
    while ( bytes != 0u )
    {
        ssize_t sent = send(fd,cursor,bytes,0);
        if ( sent <= 0 ) MESH_FAIL("tcp send");
        cursor += sent;
        bytes -= (size_t)sent;
    }
}

static void MeshReceiveAll(int fd,void *data,size_t bytes)
{
    uint8_t *cursor = (uint8_t *)data;
    while ( bytes != 0u )
    {
        ssize_t received = recv(fd,cursor,bytes,0);
        if ( received <= 0 ) MESH_FAIL("tcp receive");
        cursor += received;
        bytes -= (size_t)received;
    }
}

static void MeshExchangeTable(const MeshOptions &options,int *clients,int *server_fd,MeshEndpoint *table)
{
    sockaddr_in address;
    uint32_t one = 1u;
    std::memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)options.port);
    size_t row = options.ranks * sizeof(MeshEndpoint);
    if ( options.rank == 0u )
    {
        int listener = socket(AF_INET,SOCK_STREAM,0);
        setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        if ( listener < 0 || bind(listener,(sockaddr *)&address,sizeof(address)) != 0 || listen(listener,MESH_MAX_RANKS) != 0 ) MESH_FAIL("listen");
        for ( uint32_t rank = 1u; rank < options.ranks; rank++ )
        {
            uint32_t peer_rank;
            int fd = accept(listener,0,0);
            if ( fd < 0 ) MESH_FAIL("accept");
            setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
            MeshReceiveAll(fd,&peer_rank,sizeof(peer_rank));
            if ( peer_rank == 0u || peer_rank >= options.ranks || clients[peer_rank] >= 0 ) MESH_FAIL("duplicate rank");
            clients[peer_rank] = fd;
            MeshReceiveAll(fd,&table[peer_rank * options.ranks],row);
        }
        close(listener);
        for ( uint32_t rank = 1u; rank < options.ranks; rank++ ) MeshSendAll(clients[rank],table,row * options.ranks);
        return;
    }
    if ( inet_pton(AF_INET,options.coordinator,&address.sin_addr) != 1 ) MESH_FAIL("coordinator address");
    *server_fd = socket(AF_INET,SOCK_STREAM,0);
    for ( uint32_t attempt = 0u; connect(*server_fd,(sockaddr *)&address,sizeof(address)) != 0; attempt++ )
    {
        if ( attempt > 600u ) MESH_FAIL("connect");
        close(*server_fd);
        *server_fd = socket(AF_INET,SOCK_STREAM,0);
        usleep(100000);
    }
    setsockopt(*server_fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
    MeshSendAll(*server_fd,&options.rank,sizeof(options.rank));
    MeshSendAll(*server_fd,&table[options.rank * options.ranks],row);
    MeshReceiveAll(*server_fd,table,row * options.ranks);
}

static void MeshBarrier(const MeshOptions &options,const int *clients,int server_fd)
{
    uint8_t token = 1u;
    if ( options.rank == 0u )
    {
        for ( uint32_t rank = 1u; rank < options.ranks; rank++ ) MeshReceiveAll(clients[rank],&token,1u);
        for ( uint32_t rank = 1u; rank < options.ranks; rank++ ) MeshSendAll(clients[rank],&token,1u);
        return;
    }
    MeshSendAll(server_fd,&token,1u);
    MeshReceiveAll(server_fd,&token,1u);
}

static void MeshConnect(ibv_qp *qp,const MeshEndpoint &remote,uint32_t gid_index,uint32_t traffic_class)
{
    ibv_qp_attr attributes;
    std::memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_INIT;
    attributes.port_num = 1;
    attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0 ) MESH_FAIL("qp init");
    std::memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = IBV_MTU_4096;
    attributes.dest_qp_num = remote.qpn;
    attributes.max_dest_rd_atomic = 1;
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.port_num = 1;
    std::memcpy(attributes.ah_attr.grh.dgid.raw,remote.gid,sizeof(remote.gid));
    attributes.ah_attr.grh.sgid_index = (uint8_t)gid_index;
    attributes.ah_attr.grh.hop_limit = 1;
    attributes.ah_attr.grh.traffic_class = (uint8_t)traffic_class;
    if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0 ) MESH_FAIL("qp rtr");
    std::memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTS;
    attributes.timeout = 14;
    attributes.retry_cnt = 7;
    attributes.rnr_retry = 7;
    attributes.max_rd_atomic = 1;
    if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0 ) MESH_FAIL("qp rts");
}

static inline uint64_t MeshTicks(void)
{
    uint64_t value;
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(value) :: "memory");
    return value;
}

static inline uint64_t MeshTickHz(void)
{
    uint64_t value;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
}

static void MeshPin(int cpu)
{
    if ( cpu < 0 ) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu,&set);
    if ( pthread_setaffinity_np(pthread_self(),sizeof(set),&set) != 0 ) MESH_FAIL("pin cpu %d",cpu);
}

static void MeshDrain(MeshHost *h,std::vector<uint64_t> &done,uint64_t &published)
{
    ibv_wc wc[32];
    int count = ibv_poll_cq(h->cq,32,wc);
    if ( count <= 0 ) return;
    for ( int i = 0; i < count; i++ )
    {
        if ( wc[i].status != IBV_WC_SUCCESS )
        {
            if ( h->errors++ < 4u ) std::fprintf(stderr,"PROBE-CQE-ERROR status=%d %s vendor=0x%x qpn=0x%x\n",(int)wc[i].status,ibv_wc_status_str(wc[i].status),wc[i].vendor_err,wc[i].qp_num);
            continue;
        }
        for ( uint32_t j = 0u; j < h->peers; j++ ) if ( h->qps[j]->qp_num == wc[i].qp_num ) { done[j]++; break; }
    }
    h->completions += (uint64_t)count;
    uint64_t least = *std::min_element(done.begin(),done.end());
    if ( least > published ) { published = least; __atomic_thread_fence(__ATOMIC_RELEASE); h->shared->completed = least; }
}

static void MeshProxy(MeshHost *h)
{
    MeshPin(h->options->cpu);
    std::vector<uint64_t> done(h->peers,0u);
    uint64_t rung = 0u,published = 0u;
    while ( !h->stop.load(std::memory_order_relaxed) )
    {
        uint64_t produced = h->shared->producer;
        if ( produced > rung )
        {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            uint64_t begin = MeshTicks();
            uint32_t e = (uint32_t)(produced - 1u);
            if ( h->options->ring != MESH_RING_EACH ) asm volatile("dmb oshst" ::: "memory");
            for ( uint32_t j = 0u; j < h->peers; j++ )
            {
                if ( h->options->ring == MESH_RING_BLUEFLAME )
                {
                    uint32_t share = h->bf_share[j];
                    const uint64_t *wqe = (const uint64_t *)(h->sq[j] + (uint64_t)(e & h->wqe_mask[j]) * 64u);
                    asm volatile("ld1 {v0.2d,v1.2d,v2.2d,v3.2d}, [%1]\n\tst1 {v0.2d,v1.2d,v2.2d,v3.2d}, [%0]" :: "r"((uint8_t *)h->doorbells[j] + h->bf_offset[share]), "r"(wqe) : "v0","v1","v2","v3","memory");
                    h->bf_offset[share] ^= h->bf_size[j];
                    continue;
                }
                uint32_t control0 = __builtin_bswap32(((e & 0xffffu) << 8) | 0x08u);
                uint32_t control1 = __builtin_bswap32((h->qps[j]->qp_num << 8) | 3u);
                if ( h->options->ring == MESH_RING_EACH ) asm volatile("dmb oshst" ::: "memory");
                *h->doorbells[j] = ((uint64_t)control1 << 32) | control0;
            }
            asm volatile("dsb st" ::: "memory");
            h->busy_ticks += MeshTicks() - begin;
            h->busy_count++;
            rung = produced;
        }
        MeshDrain(h,done,published);
    }
}

static void MeshRelay(MeshHost *h)
{
    MeshPin(h->options->cpu);
    const MeshOptions &o = *h->options;
    std::vector<uint64_t> done(h->peers,0u);
    uint64_t posted = 0u,gated = 0u,published = 0u;
    uint32_t words = o.bytes / 8u;
    while ( !h->stop.load(std::memory_order_relaxed) )
    {
        uint64_t requested = h->shared->producer;
        while ( posted < requested )
        {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            uint64_t begin = MeshTicks();
            uint32_t slot = (uint32_t)(posted % o.slots);
            for ( uint32_t j = 0u; j < h->peers; j++ )
            {
                ibv_sge scatter;
                ibv_send_wr request,*bad = 0;
                scatter.addr = (uint64_t)(uintptr_t)(h->send + (uint64_t)slot * o.bytes);
                scatter.length = o.bytes;
                scatter.lkey = h->send_mr->lkey;
                std::memset(&request,0,sizeof(request));
                request.wr_id = j;
                request.sg_list = &scatter;
                request.num_sge = 1;
                request.opcode = IBV_WR_RDMA_WRITE;
                request.send_flags = IBV_SEND_SIGNALED;
                request.wr.rdma.remote_addr = h->remote[j].address + ((uint64_t)slot * o.ranks + o.rank) * o.bytes;
                request.wr.rdma.rkey = h->remote[j].rkey;
                if ( ibv_post_send(h->qps[j],&request,&bad) != 0 ) MESH_FAIL("post send");
            }
            h->busy_ticks += MeshTicks() - begin;
            h->busy_count++;
            posted++;
        }
        if ( gated < o.exchanges )
        {
            uint32_t slot = (uint32_t)(gated % o.slots),arrived = 0u;
            for ( uint32_t j = 0u; j < h->peers; j++ )
            {
                uint32_t peer = h->peer_of[j];
                const volatile uint64_t *tag = (const volatile uint64_t *)(h->receive + ((uint64_t)slot * o.ranks + peer) * o.bytes) + words - 1u;
                if ( *tag == MeshWord((uint32_t)gated,peer,words - 1u) ) arrived++;
            }
            if ( arrived == h->peers && (o.mode != MESH_MODE_STREAM || gated < requested) )
            {
                gated++;
                __atomic_thread_fence(__ATOMIC_RELEASE);
                h->shared->gate = gated;
                if ( o.mode == MESH_MODE_STREAM ) h->shared->ready = 1u;
            }
        }
        MeshDrain(h,done,published);
    }
}

static uint64_t MeshCounter(const char *device,const char *name)
{
    char path[256];
    unsigned long long value = 0ull;
    std::snprintf(path,sizeof(path),"/sys/class/infiniband/%s/ports/1/hw_counters/%s",device,name);
    FILE *file = std::fopen(path,"r");
    if ( file != 0 )
    {
        if ( std::fscanf(file,"%llu",&value) != 1 ) value = 0ull;
        std::fclose(file);
    }
    return value;
}

static void *MeshAllocate(size_t bytes)
{
    void *pointer = 0;
    size_t rounded = (bytes + 4095u) & ~(size_t)4095u;
    if ( posix_memalign(&pointer,4096u,rounded) != 0 ) MESH_FAIL("alloc %zu",bytes);
    std::memset(pointer,0,rounded);
    MESH_CUDA(cudaHostRegister(pointer,rounded,cudaHostRegisterPortable | cudaHostRegisterMapped));
    return pointer;
}

static void MeshUsage(const char *program)
{
    std::fprintf(stderr,"usage: %s --rank R [--ranks N] [--bytes B (multiple of 8, >= 16)] [--exchanges E] [--warmup W] [--slots S] [--mode proxy|relay|stream] [--guard 0|1|2 (2 = polling kernel instead of the stream wait)] [--ring each|once|blueflame] [--cpu C] [--device D] [--gid-index G] [--traffic-class T] [--coordinator IP] [--port P]\n",program);
    std::exit(2);
}

int main(int argc,char **argv)
{
    static const char *counters[] = { "out_of_sequence","packet_seq_err","roce_adp_retrans","local_ack_timeout_err" };
    MeshOptions o = { 0u,16u,4096u,20000u,1000u,4u,MESH_MODE_PROXY,MESH_RING_EACH,0u,3u,106u,47600u,17,"rocep1s0f1","10.10.100.10" };
    for ( int i = 1; i < argc; i += 2 )
    {
        if ( i + 1 >= argc ) MeshUsage(argv[0]);
        const char *flag = argv[i],*value = argv[i + 1];
        if ( std::strcmp(flag,"--rank") == 0 ) o.rank = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--ranks") == 0 ) o.ranks = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--bytes") == 0 ) o.bytes = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--exchanges") == 0 ) o.exchanges = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--warmup") == 0 ) o.warmup = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--slots") == 0 ) o.slots = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--mode") == 0 ) o.mode = std::strcmp(value,"proxy") == 0 ? MESH_MODE_PROXY : std::strcmp(value,"relay") == 0 ? MESH_MODE_RELAY : std::strcmp(value,"stream") == 0 ? MESH_MODE_STREAM : 99u;
        else if ( std::strcmp(flag,"--guard") == 0 ) o.guard = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--ring") == 0 ) o.ring = std::strcmp(value,"each") == 0 ? MESH_RING_EACH : std::strcmp(value,"once") == 0 ? MESH_RING_ONCE : std::strcmp(value,"blueflame") == 0 ? MESH_RING_BLUEFLAME : 99u;
        else if ( std::strcmp(flag,"--cpu") == 0 ) o.cpu = std::atoi(value);
        else if ( std::strcmp(flag,"--device") == 0 ) o.device = value;
        else if ( std::strcmp(flag,"--gid-index") == 0 ) o.gid_index = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--traffic-class") == 0 ) o.traffic_class = (uint32_t)std::atoi(value);
        else if ( std::strcmp(flag,"--coordinator") == 0 ) o.coordinator = value;
        else if ( std::strcmp(flag,"--port") == 0 ) o.port = (uint32_t)std::atoi(value);
        else MeshUsage(argv[0]);
    }
    if ( o.ranks < 2u || o.ranks > MESH_MAX_RANKS || o.rank >= o.ranks || o.bytes < 16u || (o.bytes & 7u) != 0u || o.exchanges <= o.warmup || o.slots < 2u || o.slots > 64u || o.mode > MESH_MODE_STREAM || o.guard > 2u || (o.mode == MESH_MODE_STREAM && o.exchanges % MESH_GRAPH_EXCHANGES != 0u) || o.ring > MESH_RING_BLUEFLAME || o.traffic_class > 255u ) MeshUsage(argv[0]);
    if ( o.ring == MESH_RING_BLUEFLAME ) { setenv("MLX5_TOTAL_UUARS","32",1); setenv("MLX5_NUM_LOW_LAT_UUARS","16",1); }
    MESH_CUDA(cudaSetDevice(0));
    int pageable = 0;
    MESH_CUDA(cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccessUsesHostPageTables,0));
    if ( pageable != 1 ) MESH_FAIL("the GPU cannot use host page tables; the SQ and doorbell records would need registration");
    int count = 0;
    ibv_device **list = ibv_get_device_list(&count);
    ibv_context *context = 0;
    for ( int i = 0; list != 0 && i < count; i++ ) if ( std::strcmp(ibv_get_device_name(list[i]),o.device) == 0 ) context = ibv_open_device(list[i]);
    if ( context == 0 ) MESH_FAIL("open %s",o.device);
    ibv_pd *pd = ibv_alloc_pd(context);
    ibv_cq *cq = pd != 0 ? ibv_create_cq(context,8192,0,0,0) : 0;
    if ( cq == 0 ) MESH_FAIL("pd/cq");
    MeshHost host;
    host.options = &o;
    host.cq = cq;
    host.stop.store(false);
    host.completions.store(0u);
    host.errors.store(0u);
    host.peers = o.ranks - 1u;
    host.busy_ticks = 0u;
    host.busy_count = 0u;
    host.send = (uint8_t *)MeshAllocate((size_t)o.slots * o.bytes);
    host.receive = (uint8_t *)MeshAllocate((size_t)o.slots * o.ranks * o.bytes);
    host.shared = (MeshShared *)MeshAllocate(sizeof(MeshShared));
    host.send_mr = ibv_reg_mr(pd,host.send,(size_t)o.slots * o.bytes,IBV_ACCESS_LOCAL_WRITE);
    ibv_mr *receive_mr = ibv_reg_mr(pd,host.receive,(size_t)o.slots * o.ranks * o.bytes,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ibv_gid gid;
    if ( host.send_mr == 0 || receive_mr == 0 || ibv_query_gid(context,1,(int)o.gid_index,&gid) != 0 ) MESH_FAIL("register");
    std::vector<MeshEndpoint> table((size_t)o.ranks * o.ranks);
    for ( uint32_t j = 0u; j < host.peers; j++ )
    {
        uint32_t peer = (o.rank + 1u + j) % o.ranks;
        ibv_qp_init_attr init;
        std::memset(&init,0,sizeof(init));
        init.send_cq = cq; init.recv_cq = cq; init.qp_type = IBV_QPT_RC; init.sq_sig_all = 1;
        init.cap.max_send_wr = 128; init.cap.max_recv_wr = 1; init.cap.max_send_sge = 1; init.cap.max_recv_sge = 1;
        host.qps[j] = ibv_create_qp(pd,&init);
        for ( uint32_t attempt = 0u; o.ring == MESH_RING_BLUEFLAME && host.qps[j] != 0 && attempt < 8u; attempt++ )
        {
            mlx5dv_qp probe;
            mlx5dv_obj objects;
            std::memset(&probe,0,sizeof(probe));
            std::memset(&objects,0,sizeof(objects));
            objects.qp.in = host.qps[j];
            objects.qp.out = &probe;
            if ( mlx5dv_init_obj(&objects,MLX5DV_OBJ_QP) != 0 || probe.bf.size >= 64u ) break;
            host.qps[j] = ibv_create_qp(pd,&init);
        }
        if ( host.qps[j] == 0 ) MESH_FAIL("create qp");
        host.peer_of[j] = peer;
        MeshEndpoint &mine = table[(size_t)o.rank * o.ranks + peer];
        mine.qpn = host.qps[j]->qp_num;
        mine.rkey = receive_mr->rkey;
        mine.address = (uint64_t)(uintptr_t)host.receive;
        std::memcpy(mine.gid,gid.raw,sizeof(gid.raw));
    }
    int clients[MESH_MAX_RANKS],server_fd = -1;
    for ( uint32_t r = 0u; r < MESH_MAX_RANKS; r++ ) clients[r] = -1;
    MeshExchangeTable(o,clients,&server_fd,table.data());
    MeshKernel k;
    std::memset(&k,0,sizeof(k));
    for ( uint32_t j = 0u; j < host.peers; j++ )
    {
        uint32_t peer = host.peer_of[j];
        host.remote[j] = table[(size_t)peer * o.ranks + o.rank];
        MeshConnect(host.qps[j],host.remote[j],o.gid_index,o.traffic_class);
        mlx5dv_qp dv;
        mlx5dv_obj objects;
        std::memset(&dv,0,sizeof(dv));
        std::memset(&objects,0,sizeof(objects));
        objects.qp.in = host.qps[j];
        objects.qp.out = &dv;
        if ( mlx5dv_init_obj(&objects,MLX5DV_OBJ_QP) != 0 ) MESH_FAIL("mlx5dv_init_obj");
        if ( dv.sq.stride != 64u || dv.sq.wqe_cnt < o.slots || (dv.sq.wqe_cnt & (dv.sq.wqe_cnt - 1u)) != 0u ) MESH_FAIL("sq geometry wqe_cnt=%u stride=%u",dv.sq.wqe_cnt,dv.sq.stride);
        host.doorbells[j] = (volatile uint64_t *)dv.bf.reg;
        host.sq[j] = (const uint8_t *)dv.sq.buf;
        host.wqe_mask[j] = dv.sq.wqe_cnt - 1u;
        host.bf_size[j] = dv.bf.size;
        host.bf_share[j] = j;
        host.bf_offset[j] = 0u;
        for ( uint32_t i = 0u; i < j; i++ ) if ( host.doorbells[i] == host.doorbells[j] ) { host.bf_share[j] = host.bf_share[i]; break; }
        if ( o.ring == MESH_RING_BLUEFLAME && dv.bf.size < 64u ) MESH_FAIL("blueflame unavailable queue=%u bf.size=%u bf.reg=%p",j,dv.bf.size,dv.bf.reg);
        k.queues[j].sq = (uint8_t *)dv.sq.buf;
        k.queues[j].dbrec = (uint32_t *)dv.dbrec;
        k.queues[j].remote = host.remote[j].address;
        k.queues[j].wqe_mask = dv.sq.wqe_cnt - 1u;
        k.queues[j].qpn = host.qps[j]->qp_num;
        k.queues[j].rkey = host.remote[j].rkey;
        k.queues[j].peer = peer;
    }
    uint64_t *samples = 0;
    uint32_t *result = 0;
    MESH_CUDA(cudaMalloc(&samples,(size_t)o.exchanges * 5u * sizeof(uint64_t)));
    MESH_CUDA(cudaMalloc(&result,8u * sizeof(uint32_t)));
    MESH_CUDA(cudaMemset(result,0,8u * sizeof(uint32_t)));
    k.send = host.send;
    k.receive = host.receive;
    k.shared = host.shared;
    k.samples = samples;
    k.result = result;
    k.timeout_ns = UINT64_C(2000000000);
    k.lkey = host.send_mr->lkey;
    k.rank = o.rank;
    k.ranks = o.ranks;
    k.peers = host.peers;
    k.bytes = o.bytes;
    k.slots = o.slots;
    k.exchanges = o.exchanges;
    k.mode = o.mode;
    cudaStream_t stream;
    MESH_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    uint64_t before[4];
    for ( int i = 0; i < 4; i++ ) before[i] = MeshCounter(o.device,counters[i]);
    uint32_t *cursor = 0;
    cudaGraphExec_t exec = 0;
    if ( o.mode == MESH_MODE_STREAM )
    {
        MESH_CUDA(cudaMalloc(&cursor,sizeof(uint32_t)));
        MESH_CUDA(cudaMemset(cursor,0,sizeof(uint32_t)));
        cudaGraph_t graph;
        MESH_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for ( uint32_t i = 0u; i < MESH_GRAPH_EXCHANGES; i++ )
        {
            MeshPublishKernel<<<1,256,0,stream>>>(k,cursor);
            if ( o.guard == 2u ) MeshPollKernel<<<1,1,0,stream>>>(host.shared,result,k.timeout_ns);
            else MeshCaptureWait(stream,host.shared);
            if ( o.guard == 1u ) MeshGuardKernel<<<1,1,0,stream>>>(host.shared,result);
            MeshConsumeKernel<<<1,256,0,stream>>>(k,cursor);
        }
        MESH_CUDA(cudaStreamEndCapture(stream,&graph));
        MESH_CUDA(cudaGraphInstantiate(&exec,graph,0));
    }
    std::thread worker(o.mode == MESH_MODE_PROXY ? MeshProxy : MeshRelay,&host);
    MeshBarrier(o,clients,server_fd);
    if ( o.mode == MESH_MODE_STREAM )
        for ( uint32_t g = 0u; g < o.exchanges / MESH_GRAPH_EXCHANGES; g++ ) MESH_CUDA(cudaGraphLaunch(exec,stream));
    else
        MeshExchangeKernel<<<1,256,0,stream>>>(k);
    MESH_CUDA(cudaGetLastError());
    MESH_CUDA(cudaStreamSynchronize(stream));
    uint32_t outcome[8];
    MESH_CUDA(cudaMemcpy(outcome,result,sizeof(outcome),cudaMemcpyDeviceToHost));
    for ( uint32_t wait = 0u; outcome[0] == 1u && host.shared->completed < o.exchanges && host.errors.load() == 0u && wait < 2000u; wait++ ) usleep(1000);
    MeshBarrier(o,clients,server_fd);
    host.stop.store(true);
    worker.join();
    std::vector<uint64_t> raw((size_t)o.exchanges * 5u);
    MESH_CUDA(cudaMemcpy(raw.data(),samples,raw.size() * sizeof(uint64_t),cudaMemcpyDeviceToHost));
    uint32_t measured = outcome[0] == 1u ? o.exchanges - o.warmup : 0u;
    std::vector<double> exchange,period,post,first,spread;
    double period_sum = 0.0;
    for ( uint32_t e = o.warmup; e < o.warmup + measured; e++ )
    {
        period.push_back(raw[5u * e] / 1000.0);
        exchange.push_back(raw[5u * e + 1u] / 1000.0);
        post.push_back(raw[5u * e + 2u] / 1000.0);
        first.push_back(raw[5u * e + 3u] / 1000.0);
        spread.push_back(raw[5u * e + 4u] / 1000.0);
        period_sum += raw[5u * e] / 1000.0;
    }
    for ( std::vector<double> *v : { &exchange,&period,&post,&first,&spread } ) std::sort(v->begin(),v->end());
    auto pct = [](const std::vector<double> &v,double p) { return v.empty() ? 0.0 : v[std::min(v.size() - 1u,(size_t)(p * v.size()))]; };
    std::printf("PROBE-MESH mode=%s ring=%s busy_ns=%.0f rank=%u ranks=%u bytes=%u exchanges=%u warmup=%u slots=%u status=%u stalled_exchange=%u missing_peer=%u exchange_p50_us=%.2f exchange_p90_us=%.2f exchange_p99_us=%.2f exchange_max_us=%.2f period_mean_us=%.2f period_p50_us=%.2f post_p50_us=%.2f first_p50_us=%.2f spread_p50_us=%.2f violations=%u reuse_waits=%u completions=%llu cqe_errors=%llu",
        o.mode == MESH_MODE_PROXY ? "proxy" : o.mode == MESH_MODE_STREAM ? (o.guard == 1u ? "stream+guard" : o.guard == 2u ? "pollkernel" : "stream") : "relay",o.ring == MESH_RING_EACH ? "each" : o.ring == MESH_RING_ONCE ? "once" : "blueflame",
        host.busy_count != 0u ? (double)host.busy_ticks * 1e9 / (double)MeshTickHz() / (double)host.busy_count : 0.0,o.rank,o.ranks,o.bytes,o.exchanges,o.warmup,o.slots,outcome[0],outcome[2],outcome[3],
        pct(exchange,0.5),pct(exchange,0.9),pct(exchange,0.99),exchange.empty() ? 0.0 : exchange.back(),measured != 0u ? period_sum / measured : 0.0,pct(period,0.5),pct(post,0.5),pct(first,0.5),pct(spread,0.5),
        outcome[1],outcome[4],(unsigned long long)host.completions.load(),(unsigned long long)host.errors.load());
    for ( int i = 0; i < 4; i++ ) std::printf(" %s=%llu",counters[i],(unsigned long long)(MeshCounter(o.device,counters[i]) - before[i]));
    std::printf("\n");
    return outcome[0] == 1u && outcome[1] == 0u && host.errors.load() == 0u ? 0 : 3;
}
