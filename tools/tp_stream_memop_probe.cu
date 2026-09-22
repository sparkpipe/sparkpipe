#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <arpa/inet.h>
#include <cerrno>
#include <endian.h>
#include <infiniband/verbs.h>
#include <sys/socket.h>
#include <sys/mman.h>

#define RT(call) do { cudaError_t rc = (call); if (rc != cudaSuccess) { std::fprintf(stderr,"FAIL line=%d runtime=%s\n",__LINE__,cudaGetErrorString(rc)); return 1; } } while (0)
#define DR(call) do { CUresult rc = (call); if (rc != CUDA_SUCCESS) { const char *name = nullptr; cuGetErrorName(rc,&name); std::fprintf(stderr,"FAIL line=%d driver=%s\n",__LINE__,name); return 1; } } while (0)
#define REQUIRE(cond) do { if (!(cond)) { std::fprintf(stderr,"FAIL assertion line=%d errno=%d\n",__LINE__,errno); return 1; } } while (0)

struct Gate {
    alignas(64) unsigned long long ready;
    alignas(64) unsigned long long error;
    alignas(64) unsigned long long payload[512];
    alignas(64) unsigned long long output;
    unsigned long long consumed;
    unsigned long long status;
    unsigned long long expected_entry;
    unsigned long long entered;
};

__global__ void Enter(volatile Gate *gate)
{
    gate->entered = gate->expected_entry;
    __threadfence_system();
}

__global__ void Consume(volatile Gate *gate)
{
    unsigned long long error = gate->error;
    if (error == 0u) {
        unsigned long long checksum = 0u;
        for (unsigned int i = 0; i < 512u; ++i) checksum += gate->payload[i] * (i + 1u);
        gate->output = checksum;
        gate->consumed = gate->consumed + 1u;
    }
    gate->status = error;
    __threadfence_system();
}

static bool WaitHostWord(const unsigned long long *word,unsigned long long expected,bool different)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        bool equal = __atomic_load_n(word,__ATOMIC_ACQUIRE) == expected;
        if (different ? !equal : equal) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

struct RdmaOptions {
    bool receive = false;
    bool send = false;
    const char *device = nullptr;
    const char *address = nullptr;
    unsigned int port = 0u;
    unsigned int ib_port = 0u;
    unsigned int gid = 256u;
    unsigned int iterations = 8u;
};

struct RdmaRecord {
    uint64_t magic;
    uint64_t address;
    uint32_t rkey;
    uint32_t qpn;
    uint32_t iterations;
    uint16_t lid;
    uint8_t gid[16];
};

struct RdmaLink {
    ibv_context *context = nullptr;
    ibv_pd *pd = nullptr;
    ibv_cq *cq = nullptr;
    ibv_qp *qp = nullptr;
    ibv_mr *mr = nullptr;
    int socket = -1;
    int listener = -1;
    RdmaRecord peer = {};
    uint64_t sequence = 0u;
    const char *phase = "device";

    bool Close() {
        if (qp != nullptr) {
            if (ibv_destroy_qp(qp) != 0) return false;
            qp = nullptr;
        }
        if (mr != nullptr) {
            if (ibv_dereg_mr(mr) != 0) return false;
            mr = nullptr;
        }
        if (cq != nullptr) {
            if (ibv_destroy_cq(cq) != 0) return false;
            cq = nullptr;
        }
        if (pd != nullptr) {
            if (ibv_dealloc_pd(pd) != 0) return false;
            pd = nullptr;
        }
        if (context != nullptr) {
            if (ibv_close_device(context) != 0) return false;
            context = nullptr;
        }
        if (socket >= 0) { close(socket); socket = -1; }
        if (listener >= 0) { close(listener); listener = -1; }
        return true;
    }
    ~RdmaLink() {
        if (!Close()) std::fprintf(stderr,"FAIL RDMA teardown errno=%d\n",errno);
    }
    bool Bytes(void *buffer,size_t bytes,bool write) {
        auto *cursor = static_cast<uint8_t *>(buffer);
        while (bytes != 0u) {
            ssize_t count = write ? ::send(socket,cursor,bytes,MSG_NOSIGNAL) : recv(socket,cursor,bytes,0);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) return false;
            cursor += count;
            bytes -= static_cast<size_t>(count);
        }
        return true;
    }
    bool Token(uint32_t expected,bool write) {
        uint32_t token = htonl(expected);
        return Bytes(&token,sizeof(token),write) && ntohl(token) == expected;
    }
    bool Open(const RdmaOptions &options,Gate *gate) {
        int count = 0;
        ibv_device **devices = ibv_get_device_list(&count);
        if (devices == nullptr) return false;
        for (int i = 0; i < count; ++i)
            if (std::strcmp(ibv_get_device_name(devices[i]),options.device) == 0)
                context = ibv_open_device(devices[i]);
        ibv_free_device_list(devices);
        if (context == nullptr) return false;
        phase = "port-gid";
        ibv_port_attr port = {};
        ibv_gid gid = {};
        if (ibv_query_port(context,options.ib_port,&port) != 0 || port.state != IBV_PORT_ACTIVE ||
            port.active_mtu < IBV_MTU_4096 || ibv_query_gid(context,options.ib_port,options.gid,&gid) != 0)
            return false;
        uint8_t zero_gid[16] = {};
        if (std::memcmp(gid.raw,zero_gid,sizeof(zero_gid)) == 0) return false;
        phase = "protection-domain";
        pd = ibv_alloc_pd(context);
        if (pd == nullptr) return false;
        phase = "completion-queue-and-memory-region";
        cq = ibv_create_cq(context,8,nullptr,nullptr,0);
        mr = ibv_reg_mr(pd,gate,sizeof(*gate),IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (cq == nullptr || mr == nullptr) return false;
        ibv_qp_init_attr init = {};
        init.send_cq = cq;
        init.recv_cq = cq;
        init.qp_type = IBV_QPT_RC;
        init.cap.max_send_wr = 4;
        init.cap.max_recv_wr = 1;
        init.cap.max_send_sge = 1;
        init.cap.max_recv_sge = 1;
        phase = "queue-pair";
        qp = ibv_create_qp(pd,&init);
        if (qp == nullptr) return false;
        phase = "tcp-address";
        sockaddr_in endpoint = {};
        endpoint.sin_family = AF_INET;
        endpoint.sin_port = htons(options.port);
        if (inet_pton(AF_INET,options.address,&endpoint.sin_addr) != 1) return false;
        phase = "tcp-socket";
        int descriptor = ::socket(AF_INET,SOCK_STREAM,0);
        if (descriptor < 0) return false;
        phase = "tcp-connect-or-accept";
        if (options.receive) {
            listener = descriptor;
            if (bind(listener,reinterpret_cast<sockaddr *>(&endpoint),sizeof(endpoint)) != 0 ||
                listen(listener,1) != 0) return false;
            std::printf("RDMA-LISTEN address=%s port=%u device=%s ib_port=%u gid=%u\n",
                options.address,options.port,options.device,options.ib_port,options.gid);
            std::fflush(stdout);
            socket = accept(listener,nullptr,nullptr);
            if (socket < 0) return false;
            close(listener);
            listener = -1;
        } else {
            socket = descriptor;
            if (connect(socket,reinterpret_cast<sockaddr *>(&endpoint),sizeof(endpoint)) != 0) return false;
        }
        phase = "tcp-timeout";
        timeval timeout = {5,0};
        if (setsockopt(socket,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)) != 0 ||
            setsockopt(socket,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout)) != 0) return false;
        phase = "record-exchange";
        RdmaRecord local = {};
        local.magic = htobe64(UINT64_C(0x53504d454d4f5031));
        local.address = htobe64(reinterpret_cast<uint64_t>(gate));
        local.rkey = htonl(mr->rkey);
        local.qpn = htonl(qp->qp_num);
        local.iterations = htonl(options.iterations);
        local.lid = htons(port.lid);
        std::memcpy(local.gid,gid.raw,16);
        if (!Bytes(&local,sizeof(local),true) || !Bytes(&peer,sizeof(peer),false) ||
            peer.magic != local.magic || peer.iterations != local.iterations) return false;
        peer.address = be64toh(peer.address);
        peer.rkey = ntohl(peer.rkey);
        peer.qpn = ntohl(peer.qpn);
        peer.lid = ntohs(peer.lid);
        phase = "qp-init";
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = options.ib_port;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(qp,&attr,IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0)
            return false;
        attr = {};
        phase = "qp-rtr";
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096;
        attr.dest_qp_num = peer.qpn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = peer.lid;
        attr.ah_attr.port_num = options.ib_port;
        attr.ah_attr.grh.sgid_index = options.gid;
        attr.ah_attr.grh.hop_limit = 1;
        std::memcpy(attr.ah_attr.grh.dgid.raw,peer.gid,16);
        if (ibv_modify_qp(qp,&attr,IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) return false;
        attr = {};
        phase = "qp-rts";
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp,&attr,IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) return false;
        phase = "peer-ready";
        return Token(1u,true) && Token(1u,false);
    }
    bool Write(Gate *source,const size_t *offsets,const size_t *lengths,unsigned int count) {
        ibv_sge scatter[3] = {};
        ibv_send_wr requests[3] = {};
        if (count == 0u || count > 3u) return false;
        uint64_t first = sequence + 1u;
        for (unsigned int i = 0; i < count; ++i) {
            scatter[i].addr = reinterpret_cast<uint64_t>(source) + offsets[i];
            scatter[i].length = lengths[i];
            scatter[i].lkey = mr->lkey;
            requests[i].wr_id = ++sequence;
            requests[i].next = i + 1u < count ? &requests[i+1u] : nullptr;
            requests[i].sg_list = &scatter[i];
            requests[i].num_sge = 1;
            requests[i].opcode = IBV_WR_RDMA_WRITE;
            requests[i].send_flags = IBV_SEND_SIGNALED;
            requests[i].wr.rdma.remote_addr = peer.address + offsets[i];
            requests[i].wr.rdma.rkey = peer.rkey;
        }
        ibv_send_wr *bad = nullptr;
        if (ibv_post_send(qp,requests,&bad) != 0) return false;
        unsigned int completed = 0u;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (completed != count) {
            ibv_wc entries[3] = {};
            int found = ibv_poll_cq(cq,3,entries);
            if (found < 0) return false;
            for (int i = 0; i < found; ++i) {
                if (entries[i].status != IBV_WC_SUCCESS || entries[i].wr_id != first + completed ||
                    entries[i].opcode != IBV_WC_RDMA_WRITE) {
                    std::fprintf(stderr,"FAIL CQ status=%u vendor=%u wr=%llu expected=%llu\n",
                        entries[i].status,entries[i].vendor_err,
                        static_cast<unsigned long long>(entries[i].wr_id),
                        static_cast<unsigned long long>(first + completed));
                    return false;
                }
                ++completed;
            }
            if (std::chrono::steady_clock::now() >= deadline) return false;
            if (found == 0) std::this_thread::yield();
        }
        return true;
    }
};

static unsigned long long PayloadValue(unsigned int trial,unsigned int index)
{
    return 0x1234567800000000ull + static_cast<unsigned long long>(trial) * 0x10001ull + index;
}

static unsigned long long PayloadChecksum(unsigned int trial)
{
    unsigned long long sum = 0u;
    for (unsigned int i = 0; i < 512u; ++i) sum += PayloadValue(trial,i) * (i + 1u);
    return sum;
}

static int RunRdmaSender(const RdmaOptions &options)
{
    Gate gate = {};
    RdmaLink link;
    if (!link.Open(options,&gate)) {
        std::fprintf(stderr,"FAIL RDMA-open phase=%s errno=%d\n",link.phase,errno);
        return 1;
    }
    for (unsigned int trial = 0u; trial < options.iterations * 3u; ++trial) {
        unsigned long long generation = 2000u + trial;
        REQUIRE(link.Token(10u + trial * 3u,false));
        gate.ready = generation - 1u;
        size_t stale_offset = offsetof(Gate,ready),stale_bytes = sizeof(gate.ready);
        REQUIRE(link.Write(&gate,&stale_offset,&stale_bytes,1u));
        REQUIRE(link.Token(10u + trial * 3u,true));
        REQUIRE(link.Token(11u + trial * 3u,false));
        for (unsigned int i = 0; i < 512u; ++i) gate.payload[i] = PayloadValue(trial,i);
        gate.error = trial % 3u == 1u ? 1u : 0u;
        gate.ready = generation;
        size_t offsets[3] = {offsetof(Gate,payload),offsetof(Gate,error),offsetof(Gate,ready)};
        size_t lengths[3] = {sizeof(gate.payload),sizeof(gate.error),sizeof(gate.ready)};
        REQUIRE(link.Write(&gate,offsets,lengths,3u));
        REQUIRE(link.Token(11u + trial * 3u,true));
        REQUIRE(link.Token(12u + trial * 3u,false));
    }
    REQUIRE(link.Close());
    std::printf("PASS RDMA sender trials=%u source_reuse=all_signaled_completions cuda_contexts=0\n",options.iterations * 3u);
    return 0;
}

int main(int argc,char **argv)
{
    if (argc == 1 || (argc == 2 && std::strcmp(argv[1],"--help") == 0)) {
        std::printf("usage: %s --run [--gpu-waits | --rdma-receive | --rdma-send] [--memfd] [--ib-device NAME --ib-port N --gid-index N --address IPv4 --tcp-port N --iterations 1..128]\nWithout --run no CUDA or RDMA calls are made.\n",argv[0]);
        return 0;
    }
    if (std::strcmp(argv[1],"--run") != 0) return 2;
    bool run = false;
    bool memfd = false;
    bool rdma_arguments = false;
    RdmaOptions rdma;
    for (int i = 2; i < argc; ++i) {
        const char *name = argv[i];
        if (std::strcmp(name,"--memfd") == 0) { memfd = true; continue; }
        if (std::strcmp(name,"--gpu-waits") == 0) { run = true; continue; }
        if (std::strcmp(name,"--rdma-receive") == 0) { rdma.receive = true; continue; }
        if (std::strcmp(name,"--rdma-send") == 0) { rdma.send = true; continue; }
        rdma_arguments = true;
        if (++i == argc) return 2;
        if (std::strcmp(name,"--ib-device") == 0) { rdma.device = argv[i]; continue; }
        if (std::strcmp(name,"--address") == 0) { rdma.address = argv[i]; continue; }
        char *end = nullptr;
        unsigned long value = std::strtoul(argv[i],&end,10);
        if (end == argv[i] || *end != '\0' || value > 65535u) return 2;
        if (std::strcmp(name,"--ib-port") == 0) rdma.ib_port = value;
        else if (std::strcmp(name,"--gid-index") == 0) rdma.gid = value;
        else if (std::strcmp(name,"--tcp-port") == 0) rdma.port = value;
        else if (std::strcmp(name,"--iterations") == 0) rdma.iterations = value;
        else return 2;
    }
    if ((memfd && rdma.send) || (rdma.receive && rdma.send) || ((rdma.receive || rdma.send) &&
        (run || rdma.device == nullptr || rdma.address == nullptr || rdma.port == 0u ||
        rdma.ib_port == 0u || rdma.ib_port > 255u || rdma.gid > 255u ||
        rdma.iterations == 0u || rdma.iterations > 128u))) return 2;
    if (!rdma.receive && !rdma.send && rdma_arguments) return 2;
    alarm(30);
    if (rdma.send) return RunRdmaSender(rdma);
    DR(cuInit(0));
    CUdevice device;
    int supported;
    DR(cuDeviceGet(&device,0));
    DR(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,device));
    REQUIRE(supported == 1);
    RT(cudaSetDevice(0));
    RT(cudaFree(nullptr));
    CUcontext context;
    DR(cuCtxGetCurrent(&context));
    Gate *host = nullptr,*mapped = nullptr;
    int mapping_fd = -1;
    size_t mapping_bytes = 0u;
    if (memfd) {
        long page = sysconf(_SC_PAGESIZE);
        REQUIRE(page > 0);
        mapping_bytes = ((sizeof(*host) + page - 1u) / page) * page;
        mapping_fd = memfd_create("sparkpipe-memop-probe",MFD_CLOEXEC);
        REQUIRE(mapping_fd >= 0);
        REQUIRE(ftruncate(mapping_fd,mapping_bytes) == 0);
        host = static_cast<Gate *>(mmap(nullptr,mapping_bytes,PROT_READ | PROT_WRITE,MAP_SHARED,mapping_fd,0));
        REQUIRE(host != MAP_FAILED);
        RT(cudaHostRegister(host,mapping_bytes,cudaHostRegisterPortable | cudaHostRegisterMapped));
    } else {
        RT(cudaHostAlloc(reinterpret_cast<void **>(&host),sizeof(*host),cudaHostAllocMapped));
    }
    std::printf("mapping=%s registration_bytes=%zu\n",memfd ? "memfd-shared-portable-mapped" : "cuda-host-mapped",
        memfd ? mapping_bytes : sizeof(*host));
    RT(cudaHostGetDevicePointer(reinterpret_cast<void **>(&mapped),host,0));
    std::memset(host,0,sizeof(*host));
    cudaStream_t stream;
    RT(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    RT(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    Enter<<<1,1,0,stream>>>(mapped);
    RT(cudaGetLastError());
    CUstreamCaptureStatus capture_status;
    cuuint64_t capture_id;
    CUgraph graph;
    const CUgraphNode *dependencies;
    size_t dependency_count;
    DR(cuStreamGetCaptureInfo(reinterpret_cast<CUstream>(stream),&capture_status,
        &capture_id,&graph,&dependencies,nullptr,&dependency_count));
    REQUIRE(capture_status == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    constexpr unsigned int count = 91u;
    CUgraphNode nodes[count];
    CUstreamBatchMemOpParams operation = {};
    operation.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    operation.waitValue.address = reinterpret_cast<CUdeviceptr>(mapped) + offsetof(Gate,ready);
    operation.waitValue.value64 = 1u;
    operation.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    CUDA_BATCH_MEM_OP_NODE_PARAMS params = {};
    params.ctx = context;
    params.count = 1u;
    params.paramArray = &operation;
    for (unsigned int i = 0; i < count; ++i) {
        DR(cuGraphAddBatchMemOpNode(&nodes[i],graph,dependencies,dependency_count,&params));
        dependencies = &nodes[i];
        dependency_count = 1u;
    }
    DR(cuStreamUpdateCaptureDependencies(reinterpret_cast<CUstream>(stream),&nodes[count-1u],nullptr,1u,CU_STREAM_SET_CAPTURE_DEPENDENCIES));
    Consume<<<1,1,0,stream>>>(mapped);
    RT(cudaGetLastError());
    cudaGraph_t captured;
    RT(cudaStreamEndCapture(stream,&captured));
    cudaGraphExec_t executable;
    RT(cudaGraphInstantiate(&executable,captured,0));
    auto started = std::chrono::steady_clock::now();
    for (unsigned int replay = 1; replay <= 1000u; ++replay) {
        operation.waitValue.value64 = replay;
        for (unsigned int i = 0; i < count; ++i)
            DR(cuGraphExecBatchMemOpNodeSetParams(reinterpret_cast<CUgraphExec>(executable),nodes[i],&params));
    }
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    std::printf("graph_nodes=%u updates=%u ns_per_replay=%.2f ns_per_node=%.2f gpu_launches=0\n",count,1000u,double(ns)/1000.0,double(ns)/(1000.0*count));
    RdmaLink link;
    if (rdma.receive && !link.Open(rdma,host)) {
        std::fprintf(stderr,"FAIL RDMA-open phase=%s errno=%d\n",link.phase,errno);
        return 1;
    }
    if (run || rdma.receive) {
        unsigned int trials = rdma.receive ? rdma.iterations * 3u : 3u;
        for (unsigned int trial = 0; trial < trials; ++trial) {
            unsigned long long generation = 2000u + trial;
            operation.waitValue.value64 = generation;
            for (unsigned int i = 0; i < count; ++i)
                DR(cuGraphExecBatchMemOpNodeSetParams(reinterpret_cast<CUgraphExec>(executable),nodes[i],&params));
            if (!rdma.receive)
                for (unsigned int i = 0; i < 512u; ++i) host->payload[i] = PayloadValue(trial,i);
            host->expected_entry = generation;
            host->entered = 0u;
            host->output = ~0ull;
            host->consumed = 0u;
            host->status = ~0ull;
            if (!rdma.receive) {
                host->error = 0u;
                __atomic_store_n(&host->ready,generation - 1u,__ATOMIC_RELEASE);
            }
            RT(cudaGraphLaunch(executable,stream));
            REQUIRE(WaitHostWord(&host->entered,generation,false));
            if (rdma.receive) {
                REQUIRE(link.Token(10u + trial * 3u,true));
                REQUIRE(link.Token(10u + trial * 3u,false));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            REQUIRE(cudaStreamQuery(stream) == cudaErrorNotReady);
            REQUIRE(host->output == ~0ull && host->consumed == 0u && host->status == ~0ull);
            const bool cancel = trial % 3u == 1u;
            if (rdma.receive) {
                REQUIRE(link.Token(11u + trial * 3u,true));
                REQUIRE(link.Token(11u + trial * 3u,false));
            } else {
                __atomic_store_n(&host->error,cancel ? 1u : 0u,__ATOMIC_RELEASE);
                __atomic_store_n(&host->ready,generation,__ATOMIC_RELEASE);
            }
            REQUIRE(WaitHostWord(&host->status,~0ull,true));
            RT(cudaStreamSynchronize(stream));
            REQUIRE(host->status == (cancel ? 1u : 0u));
            REQUIRE(host->consumed == (cancel ? 0u : 1u));
            REQUIRE(host->output == (cancel ? ~0ull : PayloadChecksum(trial)));
            if (rdma.receive) REQUIRE(link.Token(12u + trial * 3u,true));
        }
        std::printf("PASS delayed-ready stale-generation cancellation-before-release recovery; gpu_launches=%u payload_bytes=%zu network_visibility=%s\n",
            trials,sizeof(host->payload),rdma.receive ? "NIC-to-mapped-host-GPU" : "unqualified");
    }
    REQUIRE(link.Close());
    RT(cudaGraphExecDestroy(executable));
    RT(cudaGraphDestroy(captured));
    RT(cudaStreamDestroy(stream));
    if (memfd) {
        RT(cudaHostUnregister(host));
        REQUIRE(munmap(host,mapping_bytes) == 0);
        REQUIRE(close(mapping_fd) == 0);
    } else {
        RT(cudaFreeHost(host));
    }
    return 0;
}
