#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include "tests/fixtures/tp_mesh_hardware_fixture.h"
#include "sparkpipe/spark_tp_mesh_kernels.cuh"

#define REQUIRE(condition) do { if (!(condition)) { std::fprintf(stderr,"FAIL line=%d condition=%s\n",__LINE__,#condition); std::exit(1); } } while (0)
#define CUDA(call) do { cudaError_t status=(call); if (status != cudaSuccess) { std::fprintf(stderr,"FAIL line=%d CUDA=%s call=%s\n",__LINE__,cudaGetErrorString(status),#call); std::exit(1); } } while (0)

static uint64_t Now()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static uint64_t Load(const uint64_t *word) { return __atomic_load_n(word,__ATOMIC_ACQUIRE); }
static void Store(uint64_t *word,uint64_t value) { __atomic_store_n(word,value,__ATOMIC_RELEASE); }
static uint16_t Bf16(float value)
{
    uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));return static_cast<uint16_t>(bits>>16u);
}
static float Float(uint16_t value)
{
    uint32_t bits=static_cast<uint32_t>(value)<<16u;float result;std::memcpy(&result,&bits,sizeof(result));return result;
}

struct Rank
{
    cudaStream_t stream=nullptr;
    cudaEvent_t done=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    void *input=nullptr,*output=nullptr,*scratch=nullptr;
    SparkTpMeshRoundControl *control=nullptr;
};

struct Probe
{
    static constexpr uint64_t band_bytes=SPARK_WEIGHTD_MESH_SLOT_BYTES*SPARK_WEIGHTD_MESH_SLOTS_PER_BAND;
    static constexpr size_t tensor_bytes=SPARK_WEIGHTD_MESH_SLOT_BYTES+4096u;
    uint8_t *host=nullptr,*device=nullptr;
    Rank ranks[SPARK_WEIGHTD_MESH_RANKS_PER_BAND];
    uint32_t degree=0u,operation=0u,rows=0u,rounds=0u;
    uint64_t elements=0u,launch_count=0u,epoch=100u,cancel=0u,timeout=UINT64_C(2000000000);
    uint64_t shipped[16]={},pending[16]={},pending_at[16]={},enqueue_ns[16]={};
    std::atomic<bool> stop{false},hold{false};
    std::atomic<uint64_t> transfers{0u},delay_ns{0u};
    std::thread worker;
    std::vector<std::vector<uint8_t>> inputs;
    std::vector<uint8_t> expected;
    uint32_t completed_cases=0u;

    Probe()
    {
        int fd=memfd_create("spark-mesh-hardware-probe",MFD_CLOEXEC);
        REQUIRE(fd>=0 && ftruncate(fd,SPARK_WEIGHTD_MESH_REGION_BYTES)==0);
        host=static_cast<uint8_t *>(mmap(nullptr,SPARK_WEIGHTD_MESH_REGION_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0));
        REQUIRE(host != MAP_FAILED && close(fd)==0);
        CUDA(cudaHostRegister(host,SPARK_WEIGHTD_MESH_REGION_BYTES,cudaHostRegisterPortable|cudaHostRegisterMapped));
        CUDA(SparkGlm5NextMeshHardwarePrepare(host,reinterpret_cast<void **>(&device)));
        for (Rank &rank:ranks)
        {
            CUDA(cudaStreamCreateWithFlags(&rank.stream,cudaStreamNonBlocking));
            CUDA(cudaEventCreateWithFlags(&rank.done,cudaEventDisableTiming));
            CUDA(cudaMalloc(&rank.input,tensor_bytes));CUDA(cudaMalloc(&rank.output,tensor_bytes));
            CUDA(cudaMalloc(&rank.scratch,SPARK_WEIGHTD_MESH_SLOT_BYTES));
            CUDA(cudaMalloc(reinterpret_cast<void **>(&rank.control),sizeof(*rank.control)));
        }
        std::printf("CONFIG mapped_bytes=%llu ranks_max=16 slot_bytes=%u daemon=actual gate_bytes=%zu\n",
            static_cast<unsigned long long>(SPARK_WEIGHTD_MESH_REGION_BYTES),SPARK_WEIGHTD_MESH_SLOT_BYTES,sizeof(SparkWeightdMeshWaitRequest));
    }
    ~Probe()
    {
        EndWorker();
        for (Rank &rank:ranks)
        {
            CUDA(cudaStreamSynchronize(rank.stream));
            if (rank.executable) CUDA(cudaGraphExecDestroy(rank.executable));
            if (rank.graph) CUDA(cudaGraphDestroy(rank.graph));
            CUDA(cudaFree(rank.input));CUDA(cudaFree(rank.output));CUDA(cudaFree(rank.scratch));CUDA(cudaFree(rank.control));
            CUDA(cudaEventDestroy(rank.done));CUDA(cudaStreamDestroy(rank.stream));
        }
        CUDA(cudaHostUnregister(host));REQUIRE(munmap(host,SPARK_WEIGHTD_MESH_REGION_BYTES)==0);
    }
    uint64_t *Entry(uint32_t rank) { return reinterpret_cast<uint64_t *>(host+SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(rank,rank)); }
    SparkWeightdMeshWaitRequest *Gate(uint32_t rank) { return reinterpret_cast<SparkWeightdMeshWaitRequest *>(host+SPARK_WEIGHTD_MESH_WAIT_ENTRY(rank,rank)); }
    uint64_t *Cancel(uint32_t rank)
    {
        return reinterpret_cast<uint64_t *>(host+SPARK_WEIGHTD_MESH_DOORBELL_OFFSET+
            static_cast<uint64_t>(SPARK_WEIGHTD_MESH_DOORBELL_CELL_CANCEL+2u*rank)*SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES);
    }
    void EndWorker()
    {
        stop.store(true);if (worker.joinable()) worker.join();
    }
    void StartWorker()
    {
        REQUIRE(!worker.joinable());stop.store(false);
        worker=std::thread([this] {
            while (!stop.load())
            {
                uint64_t now=Now();
                for (uint32_t rank=0u;rank<degree;rank++)
                {
                    uint64_t *entry=Entry(rank),tag=Load(entry);
                    if (tag==0u || tag==shipped[rank]) continue;
                    if (pending[rank]==0u) { pending[rank]=tag;pending_at[rank]=now; }
                    REQUIRE(pending[rank]==tag);
                    if (hold.load() || now-pending_at[rank]<delay_ns.load()) continue;
                    uint64_t bytes=Load(entry+1),slot=Load(entry+2),mask=Load(entry+3);
                    REQUIRE(bytes>0u && bytes<=SPARK_WEIGHTD_MESH_SLOT_BYTES-16u);
                    REQUIRE(slot/SPARK_WEIGHTD_MESH_SLOTS_PER_RANK==rank && (mask&(UINT64_C(1)<<rank))==0u);
                    REQUIRE(mask!=0u && (mask&~((UINT64_C(1)<<degree)-1u))==0u);
                    uint8_t *source=host+rank*band_bytes+slot*SPARK_WEIGHTD_MESH_SLOT_BYTES;
                    REQUIRE(Load(reinterpret_cast<uint64_t *>(source+SPARK_WEIGHTD_MESH_SLOT_BYTES-8u))==tag);
                    for (uint32_t peer=0u;peer<degree;peer++)
                    {
                        if ((mask&(UINT64_C(1)<<peer))==0u) continue;
                        uint8_t *target=host+peer*band_bytes+slot*SPARK_WEIGHTD_MESH_SLOT_BYTES;
                        std::memcpy(target,source,bytes);
                        Store(reinterpret_cast<uint64_t *>(target+SPARK_WEIGHTD_MESH_SLOT_BYTES-8u),tag);
                    }
                    Store(reinterpret_cast<uint64_t *>(host+SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(rank,rank)),tag);
                    shipped[rank]=tag;pending[rank]=0u;transfers.fetch_add(1u);
                }
                SparkTestMeshWaitPoll(now);
                std::this_thread::yield();
            }
        });
    }
    void Setup(uint32_t next_degree,uint32_t next_operation,uint32_t next_rows,uint64_t next_elements,uint32_t next_rounds,bool preserve_graph=false)
    {
        EndWorker();degree=next_degree;operation=next_operation;rows=next_rows;elements=next_elements;rounds=next_rounds;epoch++;launch_count=0u;
        std::memset(host,0,SPARK_WEIGHTD_MESH_REGION_BYTES);std::memset(shipped,0,sizeof(shipped));std::memset(pending,0,sizeof(pending));
        hold.store(false);delay_ns.store(0u);transfers.store(0u);
        SparkTestMeshWaitInitialize(host,degree);
        for (uint32_t rank=0u;rank<degree;rank++)
        {
            if (!preserve_graph)
            {
                if (ranks[rank].executable) CUDA(cudaGraphExecDestroy(ranks[rank].executable));
                if (ranks[rank].graph) CUDA(cudaGraphDestroy(ranks[rank].graph));
                ranks[rank].executable=nullptr;ranks[rank].graph=nullptr;
            }
            SparkTpMeshRoundControl control={};control.epoch=epoch;control.cancel_expected=cancel;control.rounds_total=rounds;
            CUDA(cudaMemcpy(ranks[rank].control,&control,sizeof(control),cudaMemcpyHostToDevice));
            Store(Cancel(rank),cancel);
        }
    }
    void Data(uint32_t salt)
    {
        uint64_t local_elements=operation==0u ? elements/degree : elements;
        uint32_t width=operation==2u ? 8u : 2u;
        REQUIRE(elements*width<=tensor_bytes && local_elements*width<=tensor_bytes);
        inputs.assign(degree,std::vector<uint8_t>(local_elements*width));expected.assign(elements*width,0u);
        for (uint32_t rank=0u;rank<degree;rank++)
        {
            for (uint64_t i=0u;i<local_elements;i++)
            {
                if (operation==2u)
                {
                    uint64_t value=((static_cast<uint64_t>((rank+salt)%degree)+1u)<<60u) ^ (UINT64_C(0x100000001)*i+rank);
                    if ((i%7u)==0u && rank==degree-1u) value=UINT64_MAX-i;
                    std::memcpy(inputs[rank].data()+i*8u,&value,8u);
                    uint64_t old;std::memcpy(&old,expected.data()+i*8u,8u);
                    if (value>old) std::memcpy(expected.data()+i*8u,&value,8u);
                }
                else
                {
                    float value=rank==0u ? 1.0f+static_cast<float>((i+salt)%5u)/8.0f : static_cast<float>(1u+(rank>2u ? rank%3u : 0u))/256.0f;
                    if ((i%11u)==0u) { const float cancellation[]={256.0f,1.0f,-256.0f,1.0f};value=rank<4u ? cancellation[rank] : 0.0f; }
                    if ((i+salt)%2u) value=-value;
                    uint16_t bits=operation==0u ? static_cast<uint16_t>((rank+1u)*997u+i+salt) : Bf16(value);
                    std::memcpy(inputs[rank].data()+i*2u,&bits,2u);
                    if (operation==0u) std::memcpy(expected.data()+(rank*local_elements+i)*2u,&bits,2u);
                }
            }
            CUDA(cudaMemcpyAsync(ranks[rank].input,inputs[rank].data(),inputs[rank].size(),cudaMemcpyHostToDevice,ranks[rank].stream));
            CUDA(cudaMemsetAsync(ranks[rank].output,0xcdu,expected.size(),ranks[rank].stream));
            CUDA(cudaMemsetAsync(reinterpret_cast<uint8_t *>(ranks[rank].control)+offsetof(SparkTpMeshRoundControl,rounds_done),0,sizeof(uint64_t),ranks[rank].stream));
        }
        if (operation==1u)
            for (uint64_t i=0u;i<elements;i++)
            {
                double sum=0.0;
                for (uint32_t rank=0u;rank<degree;rank++) { uint16_t bits;std::memcpy(&bits,inputs[rank].data()+i*2u,2u);sum+=Float(bits); }
                uint16_t bits=Bf16(static_cast<float>(sum));std::memcpy(expected.data()+i*2u,&bits,2u);
            }
    }
    void Enqueue(uint32_t rank)
    {
        CUDA(SparkGlm5NextLaunchMeshHardware(ranks[rank].stream,device+rank*band_bytes,
            SPARK_WEIGHTD_MESH_SLOT_BYTES,SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            device+SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(rank,rank),device+SPARK_WEIGHTD_MESH_WAIT_ENTRY(rank,rank),
            ranks[rank].control,rank,degree,ranks[rank].input,ranks[rank].output,ranks[rank].scratch,
            elements,operation,rounds,rows,timeout));
    }
    double Capture()
    {
        uint64_t begin=Now();
        for (uint32_t rank=0u;rank<degree;rank++)
        {
            CUDA(cudaStreamSynchronize(ranks[rank].stream));
            CUDA(cudaStreamBeginCapture(ranks[rank].stream,cudaStreamCaptureModeThreadLocal));
            Enqueue(rank);CUDA(cudaStreamEndCapture(ranks[rank].stream,&ranks[rank].graph));
            CUDA(cudaGraphInstantiate(&ranks[rank].executable,ranks[rank].graph,nullptr,nullptr,0u));
        }
        return static_cast<double>(Now()-begin)/1e6;
    }
    void Launch(bool graph)
    {
        launch_count++;
        for (uint32_t rank=0u;rank<degree;rank++)
        {
            uint64_t begin=Now();
            if (graph) CUDA(cudaGraphLaunch(ranks[rank].executable,ranks[rank].stream));else Enqueue(rank);
            CUDA(cudaEventRecord(ranks[rank].done,ranks[rank].stream));enqueue_ns[rank]=Now()-begin;
        }
    }
    bool Done()
    {
        bool result=true;
        for (uint32_t rank=0u;rank<degree;rank++)
        {
            cudaError_t status=cudaEventQuery(ranks[rank].done);
            REQUIRE(status==cudaSuccess || status==cudaErrorNotReady);result &= status==cudaSuccess;
        }
        return result;
    }
    double Wait(uint64_t begin)
    {
        while (!Done())
        {
            REQUIRE(Now()-begin<UINT64_C(10000000000));
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        return static_cast<double>(Now()-begin)/1e6;
    }
    void Verify(bool failed=false,bool timed_out=false)
    {
        for (uint32_t rank=0u;rank<degree;rank++)
        {
            SparkTpMeshRoundControl control={};std::vector<uint8_t> output(expected.size());
            CUDA(cudaMemcpy(&control,ranks[rank].control,sizeof(control),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(output.data(),ranks[rank].output,output.size(),cudaMemcpyDeviceToHost));
            if ((control.error_word!=0u)!=failed)
            {
                EndWorker();
                for (uint32_t peer=0u;peer<degree;peer++)
                {
                    SparkTpMeshRoundControl other={};CUDA(cudaMemcpy(&other,ranks[peer].control,sizeof(other),cudaMemcpyDeviceToHost));
                    std::fprintf(stderr,"PEER rank=%u seq=%llu error=%llx diag=%llx rounds=%llu entry=%llx bytes=%llu slot=%llu mask=%llx enqueue_ms=%.3f\n",peer,
                        (unsigned long long)other.seq,(unsigned long long)other.error_word,(unsigned long long)other.diag_word,(unsigned long long)other.rounds_done,
                        (unsigned long long)Load(Entry(peer)),(unsigned long long)Load(Entry(peer)+1),(unsigned long long)Load(Entry(peer)+2),(unsigned long long)Load(Entry(peer)+3),enqueue_ns[peer]/1e6);
                    for (uint32_t slot=0u;slot<degree*2u;slot++) std::fprintf(stderr,"TAIL rank=%u slot=%u value=%llx\n",peer,slot,
                        (unsigned long long)Load(reinterpret_cast<uint64_t *>(host+peer*band_bytes+(slot+1u)*SPARK_WEIGHTD_MESH_SLOT_BYTES-8u)));
                }
                SparkWeightdMeshWaitRequest *gate=Gate(rank);
                std::fprintf(stderr,"CONTROL rank=%u op=%u rows=%u seq=%llu round_seq=%llx error=%llx diag=%llx done=%llu expected_error=%u gate_id=%llu kind=%llu tag=%llx mask=%llx ready=%llu gate_error=%llx shipped=%llx transfers=%llu\n",rank,operation,rows,
                    (unsigned long long)control.seq,(unsigned long long)control.round_seq,(unsigned long long)control.error_word,(unsigned long long)control.diag_word,(unsigned long long)control.rounds_done,failed,
                    (unsigned long long)Load(&gate->request_id),(unsigned long long)Load(&gate->kind),(unsigned long long)Load(&gate->tag),(unsigned long long)Load(&gate->peer_mask),(unsigned long long)Load(&gate->ready),(unsigned long long)Load(&gate->error),
                    (unsigned long long)Load(reinterpret_cast<uint64_t *>(host+SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(rank,rank))),(unsigned long long)transfers.load());
            }
            REQUIRE((control.error_word!=0u)==failed);
            REQUIRE(control.rounds_done==(failed ? 0u : rounds));
            if (timed_out) REQUIRE((control.error_word>>32u)==epoch);
            if (!failed)
            {
                uint32_t width=operation==2u ? 8u : operation==1u ? 4u : 2u;
                uint64_t chunks=(elements-1u)/((SPARK_WEIGHTD_MESH_SLOT_BYTES-16u)/width)+1u;
                uint64_t advances=rows==1u ? 1u : chunks*2u*SparkTpMeshTreeLevels(degree);
                REQUIRE(control.seq==launch_count*rounds*advances);
            }
            if (failed) REQUIRE(std::all_of(output.begin(),output.end(),[](uint8_t byte){return byte==0xcdu;}));
            else if (output!=expected)
            {
                size_t index=0u;while (index<output.size() && output[index]==expected[index]) index++;
                std::fprintf(stderr,"MISMATCH degree=%u rows=%u op=%u rank=%u byte=%zu actual=%u expected=%u\n",degree,rows,operation,rank,index,output[index],expected[index]);
                REQUIRE(false);
            }
        }
    }
    void Case(uint32_t n,uint32_t op,uint32_t b,uint64_t count,bool graph)
    {
        Setup(n,op,b,count,3u);Data(1u);double construction=graph ? Capture() : 0.0;
        StartWorker();uint64_t begin=Now();Launch(graph);double elapsed=Wait(begin);Verify();
        if (graph)
            for (uint32_t replay=0u;replay<2u;replay++) { Data(replay+2u);begin=Now();Launch(true);elapsed=Wait(begin);Verify(); }
        EndWorker();completed_cases++;
        std::printf("PASS numerical tp=%u rows=%u operation=%u elements=%llu graph=%u construct_ms=%.3f last_ms=%.3f transfers=%llu\n",n,b,op,static_cast<unsigned long long>(count),graph,construction,elapsed,static_cast<unsigned long long>(transfers.load()));
    }
    void Faults()
    {
        Setup(4u,1u,2u,513u,2u);Data(9u);Capture();hold.store(true);
        for (uint32_t rank=0u;rank<degree;rank++)
        {
            Gate(rank)->ready=1u;
            for (uint32_t slot=0u;slot<SPARK_WEIGHTD_MESH_SLOTS_PER_BAND;slot++)
                Store(reinterpret_cast<uint64_t *>(host+rank*band_bytes+(slot+1u)*SPARK_WEIGHTD_MESH_SLOT_BYTES-8u),(epoch<<32u)|UINT32_MAX);
        }
        StartWorker();uint64_t begin=Now();Launch(true);
        bool waiting=false;
        while (!waiting)
        {
            REQUIRE(Now()-begin<UINT64_C(1000000000));
            for (uint32_t rank=0u;rank<degree;rank++)
                waiting |= Load(&Gate(rank)->request_id)!=0u && Load(&Gate(rank)->kind)==SPARK_WEIGHTD_MESH_WAIT_PEERS && Load(&Gate(rank)->ready)==0u;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));REQUIRE(!Done());
        REQUIRE(Load(&Gate(0u)->request_id)==1u && Load(&Gate(0u)->ready)==0u && Load(Entry(0u))==0u);
        hold.store(false);Wait(begin);Verify();
        EndWorker();completed_cases++;std::puts("PASS delayed/stale tails and stale ready block graph replay until exact completion");
        Setup(4u,1u,2u,513u,2u,true);Data(10u);hold.store(true);StartWorker();begin=Now();Launch(true);
        while (Load(&Gate(0u)->request_id)==0u) REQUIRE(Now()-begin<UINT64_C(1000000000));
        REQUIRE(!Done());cancel++;
        for (uint32_t rank=0u;rank<degree;rank++) Store(Cancel(rank),cancel);
        Wait(begin);Verify(true);EndWorker();
        Setup(4u,1u,2u,513u,2u,true);Data(11u);StartWorker();begin=Now();Launch(true);Wait(begin);Verify();EndWorker();
        completed_cases++;std::puts("PASS cancel all pending graph work, drain, reset chain and replay same executable");
        Setup(4u,1u,2u,513u,2u,true);Data(12u);hold.store(true);StartWorker();begin=Now();Launch(true);
        Wait(begin);Verify(true,true);EndWorker();
        Setup(4u,1u,2u,513u,2u,true);Data(13u);StartWorker();begin=Now();Launch(true);Wait(begin);Verify();EndWorker();
        completed_cases++;std::puts("PASS missing peer times out with output untouched, drains and recovers same executable");
    }
    void Timings()
    {
        Setup(16u,1u,1u,257u,91u);Data(20u);double construction=Capture();StartWorker();
        std::vector<double> samples,waits,sources,peers,copies,math;
        SparkTpMeshRoundControl previous[16]={};
        for (uint32_t i=0u;i<8u;i++)
        {
            Data(i+20u);uint64_t begin=Now();Launch(true);double elapsed=Wait(begin);Verify();
            uint64_t maximum_wait=0u,maximum_source=0u,maximum_peer=0u,maximum_copy=0u,maximum_math=0u;
            for (uint32_t rank=0u;rank<degree;rank++)
            {
                SparkTpMeshRoundControl control={};CUDA(cudaMemcpy(&control,ranks[rank].control,sizeof(control),cudaMemcpyDeviceToHost));
                maximum_wait=std::max(maximum_wait,control.source_wait_ns+control.peer_wait_ns-previous[rank].source_wait_ns-previous[rank].peer_wait_ns);
                maximum_source=std::max(maximum_source,control.source_wait_ns-previous[rank].source_wait_ns);
                maximum_peer=std::max(maximum_peer,control.peer_wait_ns-previous[rank].peer_wait_ns);
                maximum_copy=std::max(maximum_copy,control.copy_ns-previous[rank].copy_ns);
                maximum_math=std::max(maximum_math,control.combine_ns-previous[rank].combine_ns);previous[rank]=control;
            }
            if (i>=2u) { samples.push_back(elapsed);waits.push_back(maximum_wait/1e6);sources.push_back(maximum_source/1e6);peers.push_back(maximum_peer/1e6);copies.push_back(maximum_copy/1e6);math.push_back(maximum_math/1e6); }
        }
        EndWorker();std::sort(samples.begin(),samples.end());std::sort(waits.begin(),waits.end());std::sort(sources.begin(),sources.end());std::sort(peers.begin(),peers.end());std::sort(copies.begin(),copies.end());std::sort(math.begin(),math.end());
        std::printf("TIMING tp=16 rows=1 rounds=91 graph_construct_ms=%.3f warmups=2 samples=%zu min_ms=%.3f median_ms=%.3f max_ms=%.3f transport=cpu-copy actual_daemon_gate=1\n",construction,samples.size(),samples.front(),samples[samples.size()/2u],samples.back());
        std::printf("PHASE_TIMING median_max_rank_wait_ms=%.3f source_wait_ms=%.3f peer_wait_ms=%.3f copy_ms=%.3f combine_ms=%.3f samples=%zu\n",waits[waits.size()/2u],sources[sources.size()/2u],peers[peers.size()/2u],copies[copies.size()/2u],math[math.size()/2u],samples.size());
        completed_cases++;
    }
};

int main(int argc,char **argv)
{
    if (argc!=2 || std::strcmp(argv[1],"--run")!=0)
    {
        std::fprintf(stderr,"usage: %s --run\n",argv[0]);return 2;
    }
    REQUIRE(std::setvbuf(stdout,nullptr,_IOLBF,0)==0);
    CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));CUDA(cudaSetDevice(0));
    CUmoduleLoadingMode loading;REQUIRE(cuModuleGetLoadingMode(&loading)==CUDA_SUCCESS);
    std::printf("ENV CUDA_MODULE_LOADING=%s CUDA_MODULE_DATA_LOADING=%s CUDA_DEVICE_MAX_CONNECTIONS=%s\n",loading==CU_MODULE_LAZY_LOADING ? "LAZY" : "EAGER",
        std::getenv("CUDA_MODULE_DATA_LOADING") ? std::getenv("CUDA_MODULE_DATA_LOADING") : "unset",std::getenv("CUDA_DEVICE_MAX_CONNECTIONS") ? std::getenv("CUDA_DEVICE_MAX_CONNECTIONS") : "unset");
    Probe probe;
    for (uint32_t degree:{2u,3u,4u,8u,16u})
        for (uint32_t operation:{0u,1u,2u})
            for (uint32_t rows:{1u,2u})
            {
                uint64_t elements=operation==0u ? degree*257u : 513u;
                probe.Case(degree,operation,rows,elements,false);
                probe.Case(degree,operation,rows,elements,true);
            }
    for (uint32_t operation:{0u,1u,2u})
    {
        uint32_t width=operation==2u ? 8u : operation==1u ? 4u : 2u;
        uint64_t elements=(SPARK_WEIGHTD_MESH_SLOT_BYTES-16u)/width+17u;
        if (operation==0u) elements=(elements+15u)&~UINT64_C(15);
        probe.Case(16u,operation,2u,elements,false);probe.Case(16u,operation,2u,elements,true);
    }
    probe.Faults();probe.Timings();
    std::printf("PASS tp_mesh_hardware_probe cases=%u GPU_math=actual GPU_wait=actual daemon_gate=actual transport=cpu-copy\n",probe.completed_cases);
    return 0;
}
