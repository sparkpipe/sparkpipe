#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <unistd.h>

#define PROBE_FAIL(...) do { std::fprintf(stderr,"PROBE-FAIL " __VA_ARGS__); std::fprintf(stderr,"\n"); std::exit(1); } while (0)
#define PROBE_CUDA(call) do { cudaError_t status_=(call); if (status_ != cudaSuccess) PROBE_FAIL("%s cuda=%d %s",#call,(int)status_,cudaGetErrorString(status_)); } while (0)

struct ProbeQueue
{
    uint8_t *sq;
    uint32_t wqe_count;
    uint32_t stride;
    uint32_t *dbrec;
    uint64_t *doorbell;
    uint64_t *doorbell_host;
    uint32_t qpn;
    uint32_t lkey;
    uint32_t rkey;
    uint64_t source;
    uint64_t destination;
    uint32_t bytes;
};

static __device__ __forceinline__ uint64_t ProbeNow(void)
{
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

static __device__ __forceinline__ uint32_t ProbeBe32(uint32_t value)
{
    return __byte_perm(value,0u,0x0123u);
}

static __device__ __forceinline__ uint64_t ProbeBe64(uint64_t value)
{
    return ((uint64_t)ProbeBe32((uint32_t)value) << 32) | ProbeBe32((uint32_t)(value >> 32));
}

static __device__ __forceinline__ void ProbeMmio64(uint64_t *address,uint64_t value)
{
    asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;" :: "l"(address), "l"(value) : "memory");
}

static __global__ void ProbeRingKernel(ProbeQueue queue,uint64_t *producer,uint8_t *source_host,const volatile uint64_t *landing,uint64_t *samples,uint32_t iterations,uint64_t timeout_ns,uint32_t *status,uint32_t ring)
{
    if ( threadIdx.x != 0u || blockIdx.x != 0u ) return;
    for ( uint32_t i = 0u; i < iterations; i++ )
    {
        uint64_t index = *producer;
        uint64_t tag = (index + 1u) | (UINT64_C(0x5a) << 56);
        *(volatile uint64_t *)(source_host + queue.bytes - 8u) = tag;
        uint4 *wqe = (uint4 *)(queue.sq + (index & (queue.wqe_count - 1u)) * queue.stride);
        uint32_t control0 = ProbeBe32((uint32_t)((index & 0xffffu) << 8) | 0x08u);
        uint32_t control1 = ProbeBe32((queue.qpn << 8) | 3u);
        uint64_t remote = ProbeBe64(queue.destination);
        uint64_t local = ProbeBe64(queue.source);
        wqe[0] = make_uint4(control0,control1,0x08000000u,0u);
        wqe[1] = make_uint4((uint32_t)remote,(uint32_t)(remote >> 32),ProbeBe32(queue.rkey),0u);
        wqe[2] = make_uint4(ProbeBe32(queue.bytes),ProbeBe32(queue.lkey),(uint32_t)local,(uint32_t)(local >> 32));
        __threadfence_system();
        uint64_t started = ProbeNow();
        *(volatile uint32_t *)(queue.dbrec + 1) = ProbeBe32((uint32_t)((index + 1u) & 0xffffu));
        __threadfence_system();
        uint64_t doorbell_value = ((uint64_t)control1 << 32) | control0;
        if ( ring == 1u ) ProbeMmio64(queue.doorbell,doorbell_value);
        else if ( ring == 2u ) { *(volatile uint64_t *)queue.doorbell = doorbell_value; __threadfence_system(); }
        else if ( ring == 3u ) { ProbeMmio64((uint64_t *)queue.doorbell_host,doorbell_value); }
        else if ( ring == 4u ) { *(volatile uint64_t *)queue.doorbell_host = doorbell_value; __threadfence_system(); }
        uint64_t landed_at = 0u;
        while ( true )
        {
            if ( *landing == tag ) { landed_at = ProbeNow(); break; }
            if ( ProbeNow() - started > timeout_ns ) { *status = 2u; return; }
        }
        samples[i] = landed_at - started;
        *producer = index + 1u;
    }
    *status = 1u;
}

static uint32_t ProbeHostBe32(uint32_t value) { return __builtin_bswap32(value); }

static int ProbeCpuExchange(const ProbeQueue &host,uint8_t *source,volatile uint64_t *landing,uint64_t index,uint32_t gpu_wrote,const uint8_t *sq_cpu,volatile uint32_t *dbrec_cpu,volatile uint64_t *doorbell_cpu,double *micros)
{
    uint64_t tag = (index + 1u) | (UINT64_C(0x5a) << 56);
    uint32_t control0 = ProbeHostBe32((uint32_t)((index & 0xffffu) << 8) | 0x08u);
    uint32_t control1 = ProbeHostBe32((host.qpn << 8) | 3u);
    if ( gpu_wrote == 0u )
    {
        *(volatile uint64_t *)(source + host.bytes - 8u) = tag;
        uint32_t *wqe = (uint32_t *)(sq_cpu + (index & (host.wqe_count - 1u)) * host.stride);
        uint64_t remote = __builtin_bswap64(host.destination),local = __builtin_bswap64(host.source);
        wqe[0] = control0; wqe[1] = control1; wqe[2] = 0x08000000u; wqe[3] = 0u;
        std::memcpy(&wqe[4],&remote,sizeof(remote)); wqe[6] = ProbeHostBe32(host.rkey); wqe[7] = 0u;
        wqe[8] = ProbeHostBe32(host.bytes); wqe[9] = ProbeHostBe32(host.lkey); std::memcpy(&wqe[10],&local,sizeof(local));
        __sync_synchronize();
        dbrec_cpu[1] = ProbeHostBe32((uint32_t)((index + 1u) & 0xffffu));
    }
    __sync_synchronize();
    timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    *doorbell_cpu = ((uint64_t)control1 << 32) | control0;
    __sync_synchronize();
    while ( *landing != tag )
    {
        clock_gettime(CLOCK_MONOTONIC,&t1);
        if ( (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec) > 1000000000LL ) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    *micros = ((t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec)) / 1000.0;
    return 1;
}

static void ProbeConnect(ibv_qp *qp,uint32_t remote_qpn,const ibv_gid &gid,uint32_t gid_index,uint32_t traffic_class)
{
    ibv_qp_attr attributes;
    std::memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_INIT;
    attributes.port_num = 1;
    attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0 ) PROBE_FAIL("init");
    std::memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = IBV_MTU_4096;
    attributes.dest_qp_num = remote_qpn;
    attributes.max_dest_rd_atomic = 1;
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.port_num = 1;
    attributes.ah_attr.grh.dgid = gid;
    attributes.ah_attr.grh.sgid_index = (uint8_t)gid_index;
    attributes.ah_attr.grh.hop_limit = 1;
    attributes.ah_attr.grh.traffic_class = (uint8_t)traffic_class;
    if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0 ) PROBE_FAIL("rtr");
    std::memset(&attributes,0,sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTS;
    attributes.timeout = 14;
    attributes.retry_cnt = 7;
    attributes.rnr_retry = 7;
    attributes.max_rd_atomic = 1;
    if ( ibv_modify_qp(qp,&attributes,IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0 ) PROBE_FAIL("rts");
}

static void *ProbeHostMap(void *pointer,size_t bytes,unsigned flags,const char *what)
{
    long page = sysconf(_SC_PAGESIZE);
    uintptr_t base = (uintptr_t)pointer & ~(uintptr_t)(page - 1);
    size_t span = (((uintptr_t)pointer + bytes + page - 1) & ~(uintptr_t)(page - 1)) - base;
    cudaError_t status = cudaHostRegister((void *)base,span,flags);
    std::printf("PROBE-REGISTER %s base=%p span=%zu flags=0x%x cuda=%d %s\n",what,(void *)base,span,flags,(int)status,cudaGetErrorString(status));
    if ( status != cudaSuccess ) std::exit(3);
    void *device = 0;
    PROBE_CUDA(cudaHostGetDevicePointer(&device,(void *)base,0));
    return (uint8_t *)device + ((uintptr_t)pointer - base);
}

int main(int argc,char **argv)
{
    const char *device_name = argc > 1 ? argv[1] : "rocep1s0f1";
    uint32_t bytes = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 4096u;
    uint32_t iterations = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 10000u;
    uint32_t gid_index = 3u,traffic_class = 106u;
    if ( bytes < 8u || (bytes & 7u) != 0u || iterations == 0u ) PROBE_FAIL("usage: %s [device] [bytes multiple of 8] [iterations]",argv[0]);
    PROBE_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    int count = 0;
    ibv_device **list = ibv_get_device_list(&count);
    ibv_context *context = 0;
    for ( int i = 0; list != 0 && i < count; i++ ) if ( std::strcmp(ibv_get_device_name(list[i]),device_name) == 0 ) context = ibv_open_device(list[i]);
    if ( context == 0 ) PROBE_FAIL("open %s",device_name);
    ibv_pd *pd = ibv_alloc_pd(context);
    ibv_cq *cq = pd != 0 ? ibv_create_cq(context,4096,0,0,0) : 0;
    if ( cq == 0 ) PROBE_FAIL("pd/cq");
    ibv_qp_init_attr init;
    std::memset(&init,0,sizeof(init));
    init.send_cq = cq; init.recv_cq = cq; init.qp_type = IBV_QPT_RC; init.sq_sig_all = 1;
    init.cap.max_send_wr = 256; init.cap.max_recv_wr = 1; init.cap.max_send_sge = 1; init.cap.max_recv_sge = 1;
    ibv_qp *sender = ibv_create_qp(pd,&init),*receiver = ibv_create_qp(pd,&init);
    if ( sender == 0 || receiver == 0 ) PROBE_FAIL("qp");
    ibv_gid gid;
    if ( ibv_query_gid(context,1,(int)gid_index,&gid) != 0 ) PROBE_FAIL("gid");
    ProbeConnect(sender,receiver->qp_num,gid,gid_index,traffic_class);
    ProbeConnect(receiver,sender->qp_num,gid,gid_index,traffic_class);
    long page = sysconf(_SC_PAGESIZE);
    uint8_t *source = 0,*destination = 0;
    if ( posix_memalign((void **)&source,(size_t)page,(size_t)page * 16u) != 0 || posix_memalign((void **)&destination,(size_t)page,(size_t)page * 16u) != 0 ) PROBE_FAIL("alloc");
    std::memset(source,0,(size_t)page * 16u); std::memset(destination,0,(size_t)page * 16u);
    ibv_mr *source_mr = ibv_reg_mr(pd,source,(size_t)page * 16u,IBV_ACCESS_LOCAL_WRITE);
    ibv_mr *destination_mr = ibv_reg_mr(pd,destination,(size_t)page * 16u,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if ( source_mr == 0 || destination_mr == 0 ) PROBE_FAIL("mr");
    mlx5dv_qp dv_qp;
    mlx5dv_obj objects;
    std::memset(&dv_qp,0,sizeof(dv_qp)); std::memset(&objects,0,sizeof(objects));
    dv_qp.comp_mask = MLX5DV_QP_MASK_UAR_MMAP_OFFSET;
    objects.qp.in = sender; objects.qp.out = &dv_qp;
    if ( mlx5dv_init_obj(&objects,MLX5DV_OBJ_QP) != 0 ) PROBE_FAIL("mlx5dv_init_obj");
    std::printf("PROBE-QP qpn=0x%x sq=%p wqe_cnt=%u stride=%u dbrec=%p bf.reg=%p bf.size=%u uar_off=0x%llx\n",sender->qp_num,dv_qp.sq.buf,dv_qp.sq.wqe_cnt,dv_qp.sq.stride,(void *)dv_qp.dbrec,dv_qp.bf.reg,dv_qp.bf.size,(unsigned long long)dv_qp.uar_mmap_offset);
    if ( dv_qp.sq.stride != 64u || (dv_qp.sq.wqe_cnt & (dv_qp.sq.wqe_cnt - 1u)) != 0u ) PROBE_FAIL("unexpected sq geometry");
    ProbeQueue queue;
    queue.sq = (uint8_t *)ProbeHostMap(dv_qp.sq.buf,(size_t)dv_qp.sq.wqe_cnt * dv_qp.sq.stride,cudaHostRegisterPortable | cudaHostRegisterMapped,"sq");
    queue.dbrec = (uint32_t *)ProbeHostMap((void *)dv_qp.dbrec,8u,cudaHostRegisterPortable | cudaHostRegisterMapped,"dbrec");
    queue.doorbell = (uint64_t *)ProbeHostMap(dv_qp.bf.reg,8u,cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory,"uar");
    queue.doorbell_host = (uint64_t *)dv_qp.bf.reg;
    queue.wqe_count = dv_qp.sq.wqe_cnt; queue.stride = dv_qp.sq.stride; queue.qpn = sender->qp_num;
    queue.lkey = source_mr->lkey; queue.rkey = destination_mr->rkey;
    queue.source = (uint64_t)(uintptr_t)source; queue.destination = (uint64_t)(uintptr_t)destination; queue.bytes = bytes;
    uint8_t *source_device = (uint8_t *)ProbeHostMap(source,(size_t)page * 16u,cudaHostRegisterPortable | cudaHostRegisterMapped,"source");
    uint8_t *destination_device = (uint8_t *)ProbeHostMap(destination,(size_t)page * 16u,cudaHostRegisterPortable | cudaHostRegisterMapped,"destination");
    std::atomic<bool> stop(false);
    std::atomic<uint64_t> completions(0),errors(0);
    std::thread drainer([&] {
        ibv_wc wc[64];
        while ( !stop.load() )
        {
            int n = ibv_poll_cq(cq,64,wc);
            for ( int i = 0; i < n; i++ )
            {
                completions++;
                if ( wc[i].status != IBV_WC_SUCCESS && errors++ < 4u ) std::fprintf(stderr,"PROBE-CQE-ERROR status=%d %s vendor=0x%x\n",(int)wc[i].status,ibv_wc_status_str(wc[i].status),wc[i].vendor_err);
            }
        }
    });
    uint64_t *producer = 0,*samples = 0;
    uint32_t *status = 0;
    PROBE_CUDA(cudaMalloc(&producer,sizeof(uint64_t))); PROBE_CUDA(cudaMemset(producer,0,sizeof(uint64_t)));
    PROBE_CUDA(cudaMalloc(&samples,(size_t)iterations * sizeof(uint64_t)));
    PROBE_CUDA(cudaMalloc(&status,sizeof(uint32_t)));
    cudaStream_t stream;
    PROBE_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaGraph_t graph; cudaGraphExec_t exec;
    PROBE_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    PROBE_CUDA(cudaMemsetAsync(status,0,sizeof(uint32_t),stream));
    ProbeRingKernel<<<1,32,0,stream>>>(queue,producer,source_device,(const volatile uint64_t *)(destination_device + bytes - 8u),samples,iterations,UINT64_C(1000000000),status,(uint32_t)(argc > 6 ? std::atoi(argv[6]) : 1));
    PROBE_CUDA(cudaStreamEndCapture(stream,&graph));
    PROBE_CUDA(cudaGraphInstantiate(&exec,graph,0));
    std::vector<uint64_t> all;
    uint32_t replays = argc > 4 ? (uint32_t)std::atoi(argv[4]) : 3u;
    const char *mode = argc > 5 ? argv[5] : "gpu";
    if ( std::strcmp(mode,"cpu") == 0 || std::strcmp(mode,"gpu-wqe-cpu-ring") == 0 )
    {
        uint32_t gpu_wrote = std::strcmp(mode,"cpu") != 0;
        std::vector<double> lat;
        uint64_t index = 0u;
        for ( uint32_t i = 0u; i < iterations; i++, index++ )
        {
            if ( gpu_wrote != 0u )
            {
                PROBE_CUDA(cudaMemcpy(producer,&index,sizeof(index),cudaMemcpyHostToDevice));
                ProbeRingKernel<<<1,32,0,stream>>>(queue,producer,source_device,(const volatile uint64_t *)(destination_device + bytes - 8u),samples,1u,UINT64_C(1000),status,0u);
                PROBE_CUDA(cudaStreamSynchronize(stream));
            }
            double us = 0.0;
            if ( ProbeCpuExchange(queue,source,(volatile uint64_t *)(destination + bytes - 8u),index,gpu_wrote,(const uint8_t *)dv_qp.sq.buf,dv_qp.dbrec,(volatile uint64_t *)dv_qp.bf.reg,&us) == 0 )
            {
                stop.store(true); drainer.join();
                std::printf("PROBE-RESULT mode=%s doorbell=NO-LANDING exchange=%u completions=%llu errors=%llu\n",mode,i,(unsigned long long)completions.load(),(unsigned long long)errors.load());
                return 4;
            }
            lat.push_back(us);
        }
        stop.store(true); drainer.join();
        std::sort(lat.begin(),lat.end());
        std::printf("PROBE-RESULT mode=%s doorbell=OK exchanges=%zu ring_to_land_us p50=%.2f p99=%.2f completions=%llu errors=%llu\n",mode,lat.size(),lat[lat.size()/2],lat[lat.size()*99/100],(unsigned long long)completions.load(),(unsigned long long)errors.load());
        return 0;
    }
    for ( uint32_t replay = 0u; replay < replays; replay++ )
    {
        PROBE_CUDA(cudaGraphLaunch(exec,stream));
        PROBE_CUDA(cudaStreamSynchronize(stream));
        uint32_t result = 0u;
        PROBE_CUDA(cudaMemcpy(&result,status,sizeof(result),cudaMemcpyDeviceToHost));
        if ( result != 1u )
        {
            stop.store(true); drainer.join();
            std::printf("PROBE-RESULT doorbell=NO-LANDING replay=%u status=%u completions=%llu errors=%llu\n",replay,result,(unsigned long long)completions.load(),(unsigned long long)errors.load());
            return 4;
        }
        std::vector<uint64_t> part(iterations);
        PROBE_CUDA(cudaMemcpy(part.data(),samples,(size_t)iterations * sizeof(uint64_t),cudaMemcpyDeviceToHost));
        if ( replay > 0u ) all.insert(all.end(),part.begin(),part.end());
    }
    while ( completions.load() < (uint64_t)iterations * replays && errors.load() == 0u ) std::this_thread::yield();
    stop.store(true); drainer.join();
    std::sort(all.begin(),all.end());
    auto pct = [&](double p) { return all.empty() ? 0.0 : all[std::min(all.size() - 1u,(size_t)(p * all.size()))] / 1000.0; };
    std::printf("PROBE-RESULT doorbell=OK uid=%u bytes=%u exchanges=%zu ring_to_land_us p50=%.2f p90=%.2f p99=%.2f p999=%.2f max=%.2f completions=%llu errors=%llu\n",
        (unsigned)getuid(),bytes,all.size(),pct(0.5),pct(0.9),pct(0.99),pct(0.999),all.empty() ? 0.0 : all.back() / 1000.0,(unsigned long long)completions.load(),(unsigned long long)errors.load());
    return errors.load() == 0u ? 0 : 5;
}
