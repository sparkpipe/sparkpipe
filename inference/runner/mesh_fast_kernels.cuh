#pragma once

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "sparkpipe/spark_weightd_mesh_hybrid.h"

#define SPARK_TP_MESH_FAST_STREAM_REDUCE_MAX_ELEMENTS 49152u
#define SPARK_TP_MESH_FAST_RSAG_STREAM_SPAN_MAX 49152u
#define SPARK_TP_MESH_FAST_RSAG_FUSED_MAX_ELEMENTS 65536u
#define SPARK_TP_MESH_FAST_GATHER_FUSED_MAX_LOCAL 16384u
#define SPARK_TP_MESH_FAST_A2A_FUSED_MAX_ELEMENTS 65536u
#define SPARK_TP_MESH_FAST_MAX_U64_FUSED_MAX_ELEMENTS 64u
#define SPARK_TP_MESH_FAST_2D_MIN_ELEMENTS 131072u
#define SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS 128u
#define SPARK_TP_MESH_FAST_WIDE_GATHER_MIN_ELEMENTS 4096u
#define SPARK_TP_MESH_FAST_WIDE_PUBLISH_BYTES 49152u
#define SPARK_TP_MESH_FAST_WIDE_STAGE_BYTES 40960u
#define SPARK_TP_MESH_FAST_SHIP_POLL_NS 200000ull
#define SPARK_TP_MESH_FAST_MBOX_WORDS 18u
#define SPARK_TP_MESH_FAST_MBOX_FAIL 16u
#define SPARK_TP_MESH_FAST_MBOX_DONE 17u
#define SPARK_TP_MESH_FAST_MBOX_OK_SALT UINT64_C(0x53504d424f584f4b)
#define SPARK_TP_MESH_FAST_MBOX_FAIL_SALT UINT64_C(0x53504d424f584641)
#define SPARK_TP_MESH_FAST_MBOX_STAGE_OK_SALT UINT64_C(0x53504d5354474f4b)
#define SPARK_TP_MESH_FAST_MBOX_STAGE_FAIL_SALT UINT64_C(0x53504d5354474641)

#define SPARK_TP_MESH_WIRE_CODEC_BF16 0u
#define SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32 1u
#define SPARK_TP_MESH_WIRE_CODEC_FP8_B32 2u
#define SPARK_TP_MESH_WIRE_CODEC_FP8_B16 3u
#define SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32_AG 4u
#define SPARK_TP_MESH_WIRE_CODEC_FP8_B32_AG 5u
#define SPARK_TP_MESH_WIRE_CODEC_FP8_B16_AG 6u
#define SPARK_TP_MESH_WIRE_MIN_ELEMENTS 524288u

static uint32_t SparkTpMeshWireCodecState = UINT32_MAX;
static thread_local uint32_t SparkTpMeshWireCodecAllowedTls = 1u;
static thread_local uint32_t SparkTpMeshTwoDAllowedTls = 1u;

static inline uint32_t SparkTpMeshParseWireCodecEnv(const char *val)
{
    if ( val == 0 || val[0] == '\0' || strcmp(val,"bf16") == 0 || strcmp(val,"0") == 0 )
        return SPARK_TP_MESH_WIRE_CODEC_BF16;
    if ( strcmp(val,"mxfp8") == 0 || strcmp(val,"mxfp8_b32") == 0 )
        return SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32;
    if ( strcmp(val,"fp8_b32") == 0 || strcmp(val,"mxfp8_bf16") == 0 )
        return SPARK_TP_MESH_WIRE_CODEC_FP8_B32;
    if ( strcmp(val,"fp8_b16") == 0 || strcmp(val,"fp8_f16") == 0 )
        return SPARK_TP_MESH_WIRE_CODEC_FP8_B16;
    if ( strcmp(val,"mxfp8_ag") == 0 || strcmp(val,"mxfp8_b32_ag") == 0 )
        return SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32_AG;
    if ( strcmp(val,"fp8_b32_ag") == 0 || strcmp(val,"mxfp8_bf16_ag") == 0 )
        return SPARK_TP_MESH_WIRE_CODEC_FP8_B32_AG;
    if ( strcmp(val,"fp8_b16_ag") == 0 || strcmp(val,"fp8_f16_ag") == 0 )
        return SPARK_TP_MESH_WIRE_CODEC_FP8_B16_AG;
    return SPARK_TP_MESH_WIRE_CODEC_BF16;
}

static inline uint32_t SparkTpMeshFastWireCodec(void)
{
    uint32_t cur = __atomic_load_n(&SparkTpMeshWireCodecState,__ATOMIC_ACQUIRE);
    if ( cur == UINT32_MAX )
    {
        cur = SparkTpMeshParseWireCodecEnv(getenv("SPARK_MESH_WIRE_CODEC"));
        __atomic_store_n(&SparkTpMeshWireCodecState,cur,__ATOMIC_RELEASE);
    }
    return cur;
}

static inline void SparkTpMeshFastSetWireCodec(uint32_t codec)
{
    __atomic_store_n(&SparkTpMeshWireCodecState,codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16_AG ? codec : SPARK_TP_MESH_WIRE_CODEC_BF16,__ATOMIC_RELEASE);
}

static inline uint32_t SparkTpMeshFastSetWireCodecAllowed(uint32_t allowed)
{ uint32_t prev = SparkTpMeshWireCodecAllowedTls; SparkTpMeshWireCodecAllowedTls = allowed != 0u ? 1u : 0u; return prev; }
static inline uint32_t SparkTpMeshFastWireCodecAllowed(void) { return SparkTpMeshWireCodecAllowedTls; }
static inline uint32_t SparkTpMeshFastSetTwoDAllowed(uint32_t allowed)
{ uint32_t prev = SparkTpMeshTwoDAllowedTls; SparkTpMeshTwoDAllowedTls = allowed != 0u ? 1u : 0u; return prev; }
static inline uint32_t SparkTpMeshFastTwoDAllowed(void) { return SparkTpMeshTwoDAllowedTls; }

static __device__ __forceinline__ uint64_t SparkTpFastTimerNs(void)
{ uint64_t ns; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory"); return ns; }
static __device__ __forceinline__ uint64_t SparkTpLdAcquireGpuU64(const volatile void *a)
{ unsigned long long v; asm volatile("ld.acquire.gpu.global.u64 %0,[%1];" : "=l"(v) : "l"(a) : "memory"); return (uint64_t)v; }
static __device__ __forceinline__ void SparkTpStReleaseGpuU64(volatile void *a,uint64_t v)
{ asm volatile("st.release.gpu.global.u64 [%0],%1;" :: "l"(a),"l"((unsigned long long)v) : "memory"); }
static __device__ __forceinline__ void SparkTpStReleaseSysU32(volatile void *a,uint32_t v)
{ asm volatile("st.release.sys.global.u32 [%0],%1;" :: "l"(a),"r"(v) : "memory"); }
static __device__ __forceinline__ void SparkTpStReleaseSysU64(volatile void *a,uint64_t v)
{ asm volatile("st.release.sys.global.u64 [%0],%1;" :: "l"(a),"l"((unsigned long long)v) : "memory"); }
static __device__ __forceinline__ uint16_t SparkTpLdCvU16(const volatile void *a)
{ uint32_t v; asm volatile("ld.global.cv.u16 %0,[%1];" : "=r"(v) : "l"(a) : "memory"); return (uint16_t)v; }
static __device__ __forceinline__ uint32_t SparkTpLdcvU32(const volatile void *a)
{ uint32_t v; asm volatile("ld.global.cv.u32 %0,[%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
static __device__ __forceinline__ void SparkTpStVolatileU32(volatile void *a,uint32_t v)
{ asm volatile("st.volatile.global.u32 [%0],%1;" :: "l"(a),"r"(v) : "memory"); }
static __device__ __forceinline__ void SparkTpStVolatileU64(volatile void *a,uint64_t v)
{ asm volatile("st.volatile.global.u64 [%0],%1;" :: "l"(a),"l"((unsigned long long)v) : "memory"); }
static __device__ __forceinline__ ulonglong2 SparkTpLdCvU64Pair(const volatile void *a)
{ ulonglong2 v; asm volatile("ld.global.cv.v2.u64 {%0,%1},[%2];" : "=l"(v.x),"=l"(v.y) : "l"(a) : "memory"); return v; }
static __device__ __forceinline__ void SparkTpStU128(void *a,uint4 v)
{ asm volatile("st.global.v4.u32 [%0],{%1,%2,%3,%4};" :: "l"(a),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w) : "memory"); }
static __device__ __forceinline__ uint32_t SparkTpBe32(uint32_t v) { return __byte_perm(v,0u,0x0123u); }
static __device__ __forceinline__ uint64_t SparkTpBe64(uint64_t v)
{ return ((uint64_t)SparkTpBe32((uint32_t)v) << 32u) | (uint64_t)SparkTpBe32((uint32_t)(v >> 32u)); }

struct SparkTpMeshHybridRegion { void *host,*device; };
struct SparkTpMeshHybridOverride { const void *gate_dev; SparkWeightdMeshHybridCtrl *ctrl_dev; uint8_t *shm_dev; uint32_t band; };
struct SparkTpMeshHybridHostState
{
    uint32_t disabled_init,disabled,host_page_tables,region_count,override_count,shm_registered;
    SparkTpMeshHybridRegion regions[4u];
    SparkTpMeshHybridOverride overrides[4u];
    uint64_t shm_generation,shm_failed_generation,shm_bytes;
    void *shm_host;
    uint8_t *shm_dev;
    char shm_path[SPARK_WEIGHTD_MESH_HYBRID_SHM_PATH_BYTES];
};

static SparkTpMeshHybridHostState SparkTpMeshFastHybridHost;

static inline uint32_t SparkTpMeshFastHybridDisabled(void)
{
    if ( __atomic_load_n(&SparkTpMeshFastHybridHost.disabled_init,__ATOMIC_ACQUIRE) == 0u )
    {
        const char *env = getenv("SPARK_MESH_HYBRID_DISABLE");
        uint32_t dis = env != 0 && env[0] != '\0' && strcmp(env,"0") != 0 ? 1u : 0u;
        __atomic_store_n(&SparkTpMeshFastHybridHost.disabled,dis,__ATOMIC_RELAXED);
        __atomic_store_n(&SparkTpMeshFastHybridHost.disabled_init,1u,__ATOMIC_RELEASE);
    }
    return __atomic_load_n(&SparkTpMeshFastHybridHost.disabled,__ATOMIC_ACQUIRE);
}

static inline void SparkTpMeshFastHybridRecordRegion(void *host,void *device)
{
    int dev = 0,pageable = 0,host_tables = 0;
    if ( SparkGpuGetDevice(&dev) == SPARK_GPU_SUCCESS &&
         SparkGpuDeviceGetAttribute(&pageable,SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,dev) == SPARK_GPU_SUCCESS &&
         SparkGpuDeviceGetAttribute(&host_tables,SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,dev) == SPARK_GPU_SUCCESS &&
         pageable == 1 && host_tables == 1 )
        SparkTpMeshFastHybridHost.host_page_tables = 1u;
    if ( host == 0 || device == 0 )
        return;
    for ( uint32_t i = 0u; i < SparkTpMeshFastHybridHost.region_count; ++i )
        if ( SparkTpMeshFastHybridHost.regions[i].device == device )
        {
            SparkTpMeshFastHybridHost.regions[i].host = host;
            return;
        }
    uint32_t slot = SparkTpMeshFastHybridHost.region_count < 4u ? SparkTpMeshFastHybridHost.region_count++ : 3u;
    SparkTpMeshFastHybridHost.regions[slot].host = host;
    SparkTpMeshFastHybridHost.regions[slot].device = device;
}

static inline void SparkTpMeshFastHybridBindOverride(
    const void *gate_dev,SparkWeightdMeshHybridCtrl *ctrl_dev,uint8_t *shm_dev,uint32_t band)
{
    if ( gate_dev == 0 ) return;
    for ( uint32_t i = 0u; i < SparkTpMeshFastHybridHost.override_count; ++i )
        if ( SparkTpMeshFastHybridHost.overrides[i].gate_dev == gate_dev )
        {
            SparkTpMeshFastHybridHost.overrides[i] = {gate_dev,ctrl_dev,shm_dev,band};
            return;
        }
    uint32_t slot = SparkTpMeshFastHybridHost.override_count < 4u ? SparkTpMeshFastHybridHost.override_count++ : 3u;
    SparkTpMeshFastHybridHost.overrides[slot] = {gate_dev,ctrl_dev,shm_dev,band};
}

static inline void SparkTpMeshFastHybridClearOverride(const void *gate_dev)
{
    for ( uint32_t i = 0u; i < SparkTpMeshFastHybridHost.override_count; ++i )
        if ( SparkTpMeshFastHybridHost.overrides[i].gate_dev == gate_dev )
        { SparkTpMeshFastHybridHost.overrides[i].ctrl_dev = 0; SparkTpMeshFastHybridHost.overrides[i].shm_dev = 0; return; }
}

static inline uint8_t *SparkTpMeshFastHybridMapShm(SparkGpuStream stream,const SparkWeightdMeshHybridCtrl *ctrl_host)
{
    SparkTpMeshHybridHostState *st = &SparkTpMeshFastHybridHost;
    SparkGpuCaptureStatus cap_status = SPARK_GPU_CAPTURE_STATUS_NONE;
    uint64_t gen = __atomic_load_n(&ctrl_host->shm_generation,__ATOMIC_ACQUIRE);
    uint64_t bytes = ctrl_host->shm_bytes;
    char path[SPARK_WEIGHTD_MESH_HYBRID_SHM_PATH_BYTES];
    if ( gen == 0u || bytes == 0u || bytes > (uint64_t)SPARK_WEIGHTD_MESH_HYBRID_SHM_BYTES * 4u ) return 0;
    memcpy(path,ctrl_host->shm_path,sizeof(path));
    path[sizeof(path) - 1u] = '\0';
    if ( path[0] == '\0' ) return 0;
    if ( st->shm_dev != 0 && st->shm_generation == gen && strcmp(st->shm_path,path) == 0 ) return st->shm_dev;
    if ( st->shm_failed_generation == gen ) return 0;
    (void)SparkGpuStreamIsCapturing(stream,&cap_status);
    if ( cap_status != SPARK_GPU_CAPTURE_STATUS_NONE && st->host_page_tables == 0u ) return 0;
    if ( st->shm_host != 0 )
    {
        if ( st->shm_registered != 0u && cap_status == SPARK_GPU_CAPTURE_STATUS_NONE ) (void)SparkGpuHostUnregister(st->shm_host);
        (void)munmap(st->shm_host,(size_t)st->shm_bytes);
        st->shm_host = 0; st->shm_dev = 0; st->shm_bytes = 0u; st->shm_registered = 0u; st->shm_generation = 0u; st->shm_path[0] = '\0';
    }
    int fd = open(path,O_RDWR | O_CLOEXEC);
    if ( fd < 0 ) { st->shm_failed_generation = gen; return 0; }
    void *mapped = mmap(0,(size_t)bytes,PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
    (void)close(fd);
    if ( mapped == MAP_FAILED ) { st->shm_failed_generation = gen; return 0; }
    void *dev_ptr = 0;
    uint32_t registered = 0u;
    if ( cap_status == SPARK_GPU_CAPTURE_STATUS_NONE )
    {
        if ( SparkGpuHostRegister(mapped,(size_t)bytes,SPARK_GPU_HOST_REGISTER_PORTABLE | SPARK_GPU_HOST_REGISTER_MAPPED) == SPARK_GPU_SUCCESS )
        {
            registered = 1u;
            if ( SparkGpuHostGetDevicePointer(&dev_ptr,mapped,0u) != SPARK_GPU_SUCCESS ) { (void)SparkGpuGetLastError(); dev_ptr = 0; }
        }
        else (void)SparkGpuGetLastError();
    }
    if ( dev_ptr == 0 && st->host_page_tables != 0u )
    {
        volatile uint8_t *pages = (volatile uint8_t *)mapped;
        for ( uint64_t off = 0u; off < bytes; off += 4096u ) (void)pages[off];
        dev_ptr = mapped;
    }
    if ( dev_ptr == 0 )
    { if ( registered != 0u ) (void)SparkGpuHostUnregister(mapped); (void)munmap(mapped,(size_t)bytes); return 0; }
    st->shm_host = mapped; st->shm_dev = (uint8_t *)dev_ptr; st->shm_bytes = bytes;
    st->shm_registered = registered; st->shm_generation = gen;
    memcpy(st->shm_path,path,sizeof(st->shm_path));
    return st->shm_dev;
}

static inline void SparkTpMeshFastHybridResolve(
    SparkGpuStream stream,const void *entry,const void *gate,uint64_t slot_bytes,
    SparkWeightdMeshHybridCtrl **ctrl_out,uint8_t **shm_out,uint32_t *band_out)
{
    *ctrl_out = 0; *shm_out = 0; *band_out = 0u;
    if ( SparkTpMeshFastHybridDisabled() != 0u || slot_bytes != SPARK_WEIGHTD_MESH_SLOT_BYTES ) return;
    for ( uint32_t i = 0u; i < SparkTpMeshFastHybridHost.override_count; ++i )
        if ( SparkTpMeshFastHybridHost.overrides[i].gate_dev == gate &&
             SparkTpMeshFastHybridHost.overrides[i].ctrl_dev != 0 && SparkTpMeshFastHybridHost.overrides[i].shm_dev != 0 )
        { *ctrl_out = SparkTpMeshFastHybridHost.overrides[i].ctrl_dev; *shm_out = SparkTpMeshFastHybridHost.overrides[i].shm_dev; *band_out = SparkTpMeshFastHybridHost.overrides[i].band; return; }
    const uintptr_t entry_addr = (uintptr_t)entry,gate_addr = (uintptr_t)gate;
    const uint64_t db_span = (uint64_t)SPARK_WEIGHTD_MESH_BANDS * SPARK_WEIGHTD_MESH_RANKS_PER_BAND * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES;
    const uint64_t wait_span = (uint64_t)SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES;
    for ( uint32_t i = 0u; i < SparkTpMeshFastHybridHost.region_count; ++i )
    {
        const uintptr_t dev_base = (uintptr_t)SparkTpMeshFastHybridHost.regions[i].device;
        const uintptr_t db_base = dev_base + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET,wait_base = dev_base + SPARK_WEIGHTD_MESH_WAIT_OFFSET;
        if ( entry_addr < db_base || entry_addr >= db_base + db_span || gate_addr < wait_base || gate_addr >= wait_base + wait_span ) continue;
        const uint64_t cell_idx = (entry_addr - db_base) / SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES;
        const uint8_t *host_base = (const uint8_t *)SparkTpMeshFastHybridHost.regions[i].host;
        const SparkWeightdMeshWaitRequest *gate_host = (const SparkWeightdMeshWaitRequest *)(host_base + SPARK_WEIGHTD_MESH_WAIT_OFFSET + cell_idx * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES);
        if ( (__atomic_load_n(&gate_host->capabilities,__ATOMIC_ACQUIRE) & SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ) == 0u ) return;
        const SparkWeightdMeshHybridCtrl *ctrl_host = (const SparkWeightdMeshHybridCtrl *)(host_base + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + SPARK_WEIGHTD_MESH_HYBRID_OFFSET);
        if ( ctrl_host->magic != SPARK_WEIGHTD_MESH_HYBRID_MAGIC || ctrl_host->version != SPARK_WEIGHTD_MESH_HYBRID_VERSION ) return;
        uint8_t *shm_dev = SparkTpMeshFastHybridMapShm(stream,ctrl_host);
        if ( shm_dev == 0 ) return;
        *ctrl_out = (SparkWeightdMeshHybridCtrl *)((uint8_t *)dev_base + SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + SPARK_WEIGHTD_MESH_HYBRID_OFFSET);
        *shm_out = shm_dev; *band_out = (uint32_t)(cell_idx / SPARK_WEIGHTD_MESH_RANKS_PER_BAND);
        return;
    }
}

struct SparkTpMeshHybridGrant
{
    uint32_t ok,gen,ds,num_bb,is_inline,peer_count,dbrec_in_shm,fast_cred,band;
    uint32_t ag_ok,ag_ds,ag_num_bb,ag_is_inline;
    uint32_t pi[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    uint32_t wrs[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
};

struct __align__(16) SparkTpMeshHybridGpuCache
{
    const SparkWeightdMeshHybridCtrl *ctrl;
    uint64_t last_hyb_seq;
    uint32_t gen,owner_band,owner_rank,degree,peer_count,all_peers_mask,min_max_inline,dbrec_in_shm,rem_sq,rem_wr;
    uint32_t pad[2u];
    uint32_t pi[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    uint32_t wrs[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    SparkWeightdMeshHybridPeer peers[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
};

static __device__ SparkTpMeshHybridGpuCache SparkTpMeshHybridDevCache[SPARK_WEIGHTD_MESH_BANDS];

#define SPARK_TP_MESH_FAST_2D_CAP_ELEMENTS (16u * ((((SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES - 8u) / sizeof(uint16_t)) / 4u) & ~UINT64_C(15)))

#define SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG (UINT64_C(1) << 63u)
#define SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG (UINT64_C(1) << 62u)
#define SPARK_TP_MESH_FAST_PRIOR_FLAGS_MASK (SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG | SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG)

static __device__ __forceinline__ void SparkTpMeshHybridAcquireWarp0(
    const SparkTpMeshRoundControl *control,SparkWeightdMeshHybridCtrl *ctrl,
    uint32_t hybrid_band,uint32_t degree,uint32_t rank,
    uint32_t peer_mask,uint32_t xfer_len,uint32_t allow_inline,uint32_t allow_dma,
    uint32_t skip_cpu_check,SparkTpMeshHybridGrant &s_hyb,
    uint32_t ag_xfer_len = 0u,uint32_t ag_allow_inline = 0u,uint32_t ag_allow_dma = 0u,
    uint64_t prev_rseq = UINT64_MAX)
{
    const uint32_t lane = threadIdx.x & 31u;
    if ( lane == 0u ) { s_hyb.ok = 0u; s_hyb.ag_ok = 0u; }
    if ( ctrl == 0 || peer_mask == 0u || xfer_len == 0u ) return;
    SparkTpMeshHybridGpuCache *cache = &SparkTpMeshHybridDevCache[hybrid_band & (SPARK_WEIGHTD_MESH_BANDS - 1u)];
    if ( lane == 0u ) { SparkTpStVolatileU64(&ctrl->gpu_busy,1u); __threadfence_system(); }
    __syncwarp();
    uint32_t pre_ok = 0u,pcount = 0u,num_bb = 0u,ds = 0u,is_inl = 0u,gen = 0u,dbr = 0u;
    uint32_t ag_bb = 0u,ag_ds = 0u,ag_inl = 0u,all_peers = 0u,fast_cred = 0u,same_gen = 0u;
    if ( lane == 0u )
    {
        const ulonglong2 mg = SparkTpLdCvU64Pair(&ctrl->cpu_mode);
        const uint64_t mode = mg.x; gen = (uint32_t)mg.y;
        const uint64_t rseq = skip_cpu_check == 0u && control != 0 ? (prev_rseq != UINT64_MAX ? prev_rseq : SparkTpLdcvU64(&control->round_seq)) : 0u;
        same_gen = mode == 1u && gen != 0u && cache->ctrl == ctrl && cache->gen == gen ? 1u : 0u;
        const uint32_t gen_hit = same_gen != 0u && (skip_cpu_check != 0u || (rseq != 0u && cache->last_hyb_seq == rseq)) ? 1u : 0u;
        if ( mode == 1u && gen_hit == 0u ) __threadfence_system();
        uint32_t cpu_pending = 0u;
        if ( skip_cpu_check == 0u && rseq != 0u && gen_hit == 0u &&
             SparkTpLdcvU64(&ctrl->trigger[rseq & 1u].seq) != rseq && SparkTpLdcvU64(&ctrl->cpu_issued_seq) < rseq )
        {
            const volatile uint64_t *shipped = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
            if ( shipped == 0 || SparkTpLdAcquireU64(shipped) != rseq ) cpu_pending = 1u;
        }
        const uint32_t ob = same_gen != 0u ? cache->owner_band : ctrl->owner_band,dg = same_gen != 0u ? cache->degree : ctrl->degree;
        const uint32_t ork = same_gen != 0u ? cache->owner_rank : ctrl->owner_rank,all_mask = same_gen != 0u ? cache->all_peers_mask : ctrl->all_peers_mask;
        const uint32_t max_inl = same_gen != 0u ? cache->min_max_inline : ctrl->min_max_inline;
        pcount = same_gen != 0u ? cache->peer_count : ctrl->peer_count;
        dbr = same_gen != 0u ? cache->dbrec_in_shm : ctrl->dbrec_in_shm;
        if ( cpu_pending == 0u && mode == 1u && gen != 0u && ob == hybrid_band &&
             dg == degree && ork == rank && pcount != 0u && pcount <= SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS && (peer_mask & ~all_mask) == 0u )
        {
            all_peers = peer_mask == all_mask ? 1u : 0u;
            is_inl = allow_inline != 0u && xfer_len <= max_inl ? 1u : 0u;
            if ( is_inl != 0u || allow_dma != 0u )
            {
                ds = is_inl != 0u ? ((36u + xfer_len + 15u) >> 4u) : 3u;
                num_bb = (ds + 3u) >> 2u;
                if ( num_bb != 0u && num_bb <= 15u )
                {
                    pre_ok = 1u;
                    if ( ag_xfer_len != 0u )
                    {
                        ag_inl = ag_allow_inline != 0u && ag_xfer_len <= max_inl ? 1u : 0u;
                        if ( ag_inl != 0u || ag_allow_dma != 0u )
                        { ag_ds = ag_inl != 0u ? ((36u + ag_xfer_len + 15u) >> 4u) : 3u; ag_bb = (ag_ds + 3u) >> 2u; if ( ag_bb > 15u ) ag_bb = 0u; }
                    }
                    if ( gen_hit != 0u && cache->rem_sq >= num_bb + ag_bb && cache->rem_wr >= (ag_bb != 0u ? 2u : 1u) )
                        fast_cred = 1u;
                    else if ( same_gen == 0u )
                    {
                        cache->owner_band = ob; cache->degree = dg; cache->owner_rank = ork; cache->peer_count = pcount;
                        cache->all_peers_mask = all_mask; cache->min_max_inline = max_inl; cache->dbrec_in_shm = dbr;
                    }
                }
            }
        }
    }
    const uint32_t pk = __shfl_sync(0xffffffffu,pre_ok | (pcount << 1u) | (num_bb << 6u) | (ag_bb << 10u) | (all_peers << 14u) | (fast_cred << 15u) | (same_gen << 16u),0);
    pre_ok = pk & 1u; pcount = (pk >> 1u) & 31u; num_bb = (pk >> 6u) & 15u; ag_bb = (pk >> 10u) & 15u;
    all_peers = (pk >> 14u) & 1u; fast_cred = (pk >> 15u) & 1u; same_gen = (pk >> 16u) & 1u;
    if ( pre_ok == 0u ) { if ( lane == 0u ) SparkTpStReleaseSysU64(&ctrl->gpu_busy,0u); return; }
    uint32_t ag_ok = 0u;
    if ( fast_cred != 0u )
    {
        if ( lane < pcount && (all_peers != 0u || (peer_mask & (1u << cache->peers[lane].target)) != 0u) )
        {
            const uint32_t pi = cache->pi[lane],wrs = cache->wrs[lane];
            s_hyb.pi[lane] = pi; s_hyb.wrs[lane] = wrs + 1u;
            cache->pi[lane] = pi + num_bb; cache->wrs[lane] = wrs + 1u;
        }
        if ( lane == 0u ) { ag_ok = ag_bb != 0u ? 1u : 0u; cache->rem_sq -= num_bb; cache->rem_wr -= 1u; }
    }
    else
    {
        uint32_t credit_ok = 1u,active_peer = 0u,pi = 0u,wrs = 0u,sq_av = UINT32_MAX,wr_av = UINT32_MAX;
        if ( lane < pcount )
        {
            pi = same_gen != 0u ? cache->pi[lane] : SparkTpLdcvU32(&ctrl->gpu_pi[lane]);
            wrs = same_gen != 0u ? cache->wrs[lane] : SparkTpLdcvU32(&ctrl->gpu_wrs[lane]);
            if ( all_peers != 0u || (peer_mask & (1u << (same_gen != 0u ? cache->peers[lane].target : ctrl->peers[lane].target))) != 0u )
            {
                active_peer = 1u;
                sq_av = SparkTpLdcvU32(&ctrl->cpu_sq_limit[lane]) - pi;
                wr_av = SparkTpLdcvU32(&ctrl->cpu_wr_limit[lane]) - wrs;
                if ( sq_av < num_bb || wr_av < 1u ) credit_ok = 0u;
            }
        }
        if ( !__all_sync(0xffffffffu,credit_ok != 0u) ) { if ( lane == 0u ) SparkTpStReleaseSysU64(&ctrl->gpu_busy,0u); return; }
        for ( uint32_t s = 16u; s != 0u; s >>= 1u )
        {
            const uint32_t osq = __shfl_xor_sync(0xffffffffu,sq_av,s),owr = __shfl_xor_sync(0xffffffffu,wr_av,s);
            sq_av = osq < sq_av ? osq : sq_av; wr_av = owr < wr_av ? owr : wr_av;
        }
        if ( lane < pcount )
        {
            if ( active_peer != 0u ) { s_hyb.pi[lane] = pi; s_hyb.wrs[lane] = wrs + 1u; }
            cache->pi[lane] = active_peer != 0u ? pi + num_bb : pi;
            cache->wrs[lane] = active_peer != 0u ? wrs + 1u : wrs;
        }
        if ( same_gen == 0u )
            for ( uint32_t w = lane; w < pcount * 4u; w += 32u ) ((uint4 *)cache->peers)[w] = ((const uint4 *)ctrl->peers)[w];
        if ( lane == 0u )
        {
            ag_ok = ag_bb != 0u && sq_av >= num_bb + ag_bb && wr_av >= 2u ? 1u : 0u;
            cache->ctrl = ctrl; cache->gen = gen;
            cache->rem_sq = all_peers != 0u ? (sq_av - num_bb) : 0u; cache->rem_wr = all_peers != 0u ? (wr_av - 1u) : 0u;
        }
    }
    if ( lane == 0u )
    {
        s_hyb.ok = 1u; s_hyb.gen = gen; s_hyb.ds = ds; s_hyb.num_bb = num_bb; s_hyb.is_inline = is_inl;
        s_hyb.peer_count = pcount; s_hyb.dbrec_in_shm = dbr; s_hyb.fast_cred = same_gen; s_hyb.band = hybrid_band;
        s_hyb.ag_ok = ag_ok; s_hyb.ag_ds = ag_ds; s_hyb.ag_num_bb = ag_bb; s_hyb.ag_is_inline = ag_inl;
    }
}

static __device__ __forceinline__ uint32_t SparkTpMeshHybridRevalidateWarp(
    SparkWeightdMeshHybridCtrl *ctrl,uint32_t peer_mask,const SparkWeightdMeshHybridPeer *peers,
    SparkTpMeshHybridGrant &s_hyb)
{
    const uint32_t lane = threadIdx.x & 31u;
    uint32_t ok = 0u;
    if ( lane == 0u )
    {
        SparkTpStVolatileU64(&ctrl->gpu_busy,1u);
        __threadfence_system();
        const ulonglong2 mg = SparkTpLdCvU64Pair(&ctrl->cpu_mode);
        ok = mg.x == 1u && (uint32_t)mg.y == s_hyb.gen ? 1u : 0u;
    }
    ok = __shfl_sync(0xffffffffu,ok,0);
    if ( ok == 0u )
    {
        if ( lane == 0u ) { s_hyb.ok = 0u; s_hyb.ag_ok = 0u; SparkTpStReleaseSysU64(&ctrl->gpu_busy,0u); }
        return 0u;
    }
    SparkTpMeshHybridGpuCache *cache = &SparkTpMeshHybridDevCache[s_hyb.band & (SPARK_WEIGHTD_MESH_BANDS - 1u)];
    const uint32_t rs_bb = s_hyb.num_bb,ag_bb = s_hyb.ag_num_bb;
    if ( lane < s_hyb.peer_count && (peer_mask & (1u << peers[lane].target)) != 0u )
    {
        const uint32_t pi = s_hyb.pi[lane] + rs_bb;
        s_hyb.pi[lane] = pi;
        cache->pi[lane] = pi + ag_bb; cache->wrs[lane] = s_hyb.wrs[lane] + 1u;
    }
    __syncwarp();
    if ( lane == 0u )
    {
        if ( cache->rem_sq >= ag_bb && cache->rem_wr >= 1u ) { cache->rem_sq -= ag_bb; cache->rem_wr -= 1u; }
        else { cache->rem_sq = 0u; cache->rem_wr = 0u; }
        s_hyb.ok = 1u; s_hyb.ds = s_hyb.ag_ds; s_hyb.num_bb = ag_bb; s_hyb.is_inline = s_hyb.ag_is_inline;
    }
    return 1u;
}

static __device__ __forceinline__ uint32_t SparkTpMeshHybridInlineWord(
    const uint16_t *src16,uint64_t src_elems,uint32_t n_u32,uint32_t inl_hdr,uint64_t tag,uint32_t k)
{
    if ( k == 0u ) return inl_hdr;
    if ( k <= n_u32 )
    {
        const uint64_t e = (uint64_t)(k - 1u) * 2u;
        const uint32_t lo = e < src_elems ? (uint32_t)src16[e] : 0u;
        const uint32_t hi = e + 1u < src_elems ? (uint32_t)src16[e + 1u] : 0u;
        return lo | (hi << 16u);
    }
    if ( k == n_u32 + 1u ) return (uint32_t)tag;
    if ( k == n_u32 + 2u ) return (uint32_t)(tag >> 32u);
    return 0u;
}

static __device__ __forceinline__ void SparkTpMeshHybridWriteInlineWqe(
    uint8_t *shm,const SparkWeightdMeshHybridPeer *hp,uint32_t pi,uint32_t ds,
    uint64_t slot_base,const uint16_t *src16,uint64_t src_elems,uint64_t bytes,uint64_t tag,uint32_t lane)
{
    uint4 *sq_vec = (uint4 *)(shm + hp->sq_shm_off);
    const uint32_t sq_idx = pi & hp->sq_mask;
    const uint32_t sq_vec_mask = hp->sq_depth * 4u - 1u;
    const uint32_t vec_base = sq_idx * 4u + 2u;
    const uint32_t xfer_len = (uint32_t)(bytes + 8u);
    const uint32_t n_u32 = (uint32_t)(bytes >> 2u);
    const uint32_t n_vec = (n_u32 + 6u) >> 2u;
    const uint32_t inl_hdr = SparkTpBe32(xfer_len | 0x80000000u);
    if ( lane == 0u )
    {
        const uint32_t w0 = SparkTpBe32(((pi & 0xffffu) << 8u) | 0x08u);
        const uint32_t w1 = SparkTpBe32(hp->qpn_shifted | ds);
        SparkTpStU128(&sq_vec[sq_idx * 4u],make_uint4(w0,w1,0x08000000u,0u));
    }
    else if ( lane == 1u )
    {
        const uint64_t raddr_be = SparkTpBe64(hp->remote_base + slot_base);
        SparkTpStU128(&sq_vec[sq_idx * 4u + 1u],make_uint4((uint32_t)raddr_be,(uint32_t)(raddr_be >> 32u),hp->rkey_be,0u));
    }
    if ( src_elems * 2u >= bytes && (((uintptr_t)src16 & 15u) == 0u) && (bytes & 7u) == 0u )
    {
        const uint4 *src4 = (const uint4 *)src16;
        const uint32_t full_q = n_u32 >> 2u,rem = n_u32 & 3u;
        const uint32_t tlo = (uint32_t)tag,thi = (uint32_t)(tag >> 32u);
        uint32_t carry = inl_hdr;
        for ( uint32_t base = 0u; base < n_vec; base += 32u )
        {
            const uint32_t v = base + lane;
            uint4 cur = make_uint4(0u,0u,0u,0u);
            if ( v < full_q )
                cur = src4[v];
            else if ( v == full_q )
                cur = rem != 0u ? make_uint4(((const uint32_t *)src16)[full_q * 4u],((const uint32_t *)src16)[full_q * 4u + 1u],tlo,thi) :
                    make_uint4(tlo,thi,0u,0u);
            const uint32_t prev = __shfl_up_sync(0xffffffffu,cur.w,1);
            const uint32_t w0 = lane == 0u ? carry : prev;
            carry = __shfl_sync(0xffffffffu,cur.w,31);
            if ( v < n_vec )
                SparkTpStU128(&sq_vec[(vec_base + v) & sq_vec_mask],make_uint4(w0,cur.x,cur.y,cur.z));
        }
        return;
    }
    for ( uint32_t v = lane; v < n_vec; v += 32u )
    {
        const uint32_t k = v * 4u;
        SparkTpStU128(&sq_vec[(vec_base + v) & sq_vec_mask],make_uint4(
            SparkTpMeshHybridInlineWord(src16,src_elems,n_u32,inl_hdr,tag,k),
            SparkTpMeshHybridInlineWord(src16,src_elems,n_u32,inl_hdr,tag,k + 1u),
            SparkTpMeshHybridInlineWord(src16,src_elems,n_u32,inl_hdr,tag,k + 2u),
            SparkTpMeshHybridInlineWord(src16,src_elems,n_u32,inl_hdr,tag,k + 3u)));
    }
}

static __device__ __forceinline__ void SparkTpMeshHybridWriteDmaWqe(
    uint8_t *shm,const SparkWeightdMeshHybridPeer *hp,uint32_t pi,
    uint64_t slot_base,uint64_t local_iova,uint32_t lkey_be,uint32_t xfer_len)
{
    uint4 *wqe = (uint4 *)(shm + hp->sq_shm_off + (uint64_t)(pi & hp->sq_mask) * 64u);
    const uint32_t w0 = SparkTpBe32(((pi & 0xffffu) << 8u) | 0x08u);
    const uint32_t w1 = SparkTpBe32(hp->qpn_shifted | 3u);
    const uint64_t raddr_be = SparkTpBe64(hp->remote_base + slot_base);
    const uint64_t iova_be = SparkTpBe64(local_iova);
    SparkTpStU128(wqe,make_uint4(w0,w1,0x08000000u,0u));
    SparkTpStU128(wqe + 1,make_uint4((uint32_t)raddr_be,(uint32_t)(raddr_be >> 32u),hp->rkey_be,0u));
    SparkTpStU128(wqe + 2,make_uint4(SparkTpBe32(xfer_len),lkey_be,(uint32_t)iova_be,(uint32_t)(iova_be >> 32u)));
}

static __device__ __forceinline__ void SparkTpMeshHybridWriteDmaWarps(
    const SparkWeightdMeshHybridPeer *peers,uint8_t *shm,const SparkTpMeshHybridGrant &s_hyb,
    uint32_t peer_mask,uint32_t is_peer,uint64_t slot_base,uint64_t base_iova,uint32_t xfer_len,
    uint32_t warp_id,uint32_t warps)
{
    for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
    {
        const SparkWeightdMeshHybridPeer *hp = &peers[i];
        if ( (peer_mask & (1u << hp->target)) == 0u ) continue;
        SparkTpMeshHybridWriteDmaWqe(shm,hp,s_hyb.pi[i],slot_base,
            is_peer != 0u ? (base_iova + hp->staging_peer_off) : base_iova,
            is_peer != 0u ? hp->staging_lkey_be : hp->recv_lkey_be,xfer_len);
    }
}

static __device__ __forceinline__ void SparkTpMeshHybridRingWarp0(
    SparkWeightdMeshHybridCtrl *ctrl,uint8_t *shm,const SparkWeightdMeshHybridPeer *peers,
    const SparkTpMeshHybridGrant &g,uint32_t peer_mask,uint64_t seq,uint64_t tag,uint64_t slot,uint64_t bytes)
{
    const uint32_t lane = threadIdx.x & 31u;
    if ( g.dbrec_in_shm != 0u )
    {
        if ( lane < g.peer_count && (peer_mask & (1u << peers[lane].target)) != 0u )
            SparkTpStReleaseSysU32(shm + peers[lane].dbrec_shm_off + 4u,SparkTpBe32((g.pi[lane] + g.num_bb) & 0xffffu));
        __syncwarp();
    }
    if ( lane == 0u )
    {
        SparkWeightdMeshHybridTrigger *trig = &ctrl->trigger[seq & 1u];
        trig->meta = SparkWeightdMeshHybridMetaPack(peer_mask,g.ds,g.num_bb,g.gen);
        trig->slot = slot; trig->bytes = bytes;
        SparkTpStReleaseSysU64(&trig->seq,tag);
        SparkTpStReleaseSysU64(&ctrl->gpu_busy,0u);
        SparkTpMeshHybridDevCache[g.band & (SPARK_WEIGHTD_MESH_BANDS - 1u)].last_hyb_seq = tag;
    }
}

static __device__ __forceinline__ uint32_t SparkTpMeshHybridTryRingDmaWarp0(
    const SparkTpMeshRoundControl *control,SparkWeightdMeshHybridCtrl *ctrl,uint8_t *shm,
    uint32_t hybrid_band,uint32_t degree,uint32_t rank,
    uint32_t peer_mask,uint32_t is_peer,uint64_t slot,uint64_t slot_bytes,
    uint64_t seq,uint64_t tag,uint64_t bytes,uint64_t half_off)
{
    __shared__ SparkTpMeshHybridGrant s_hyb;
    const uint32_t lane = threadIdx.x & 31u;
    if ( ctrl == 0 || shm == 0 || slot_bytes != SPARK_WEIGHTD_MESH_SLOT_BYTES )
        return 0u;
    SparkTpMeshHybridAcquireWarp0(control,ctrl,hybrid_band,degree,rank,peer_mask,(uint32_t)(bytes + 8u),0u,1u,0u,s_hyb);
    __syncwarp();
    if ( s_hyb.ok == 0u )
        return 0u;
    const uint64_t slot_base = ((uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * slot_bytes;
    const uint64_t staging_band_iova = ctrl->staging_iova_base +
        (uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES + half_off;
    const uint64_t recv_iova = ctrl->recv_iova_base + slot_base;
    if ( lane < s_hyb.peer_count )
    {
        const SparkWeightdMeshHybridPeer *hp = &ctrl->peers[lane];
        if ( (peer_mask & (1u << hp->target)) != 0u )
            SparkTpMeshHybridWriteDmaWqe(shm,hp,s_hyb.pi[lane],slot_base,
                is_peer != 0u ? (staging_band_iova + hp->staging_peer_off) : recv_iova,
                is_peer != 0u ? hp->staging_lkey_be : hp->recv_lkey_be,(uint32_t)(bytes + 8u));
    }
    __syncwarp();
    SparkTpMeshHybridRingWarp0(ctrl,shm,ctrl->peers,s_hyb,peer_mask,seq,tag,slot,bytes);
    return 1u;
}

static __device__ __forceinline__ uint64_t SparkTpMeshFastRouteCompact(uint64_t route,uint32_t compact)
{
    if ( compact == 0u ) return route;
    SparkWeightdMeshRoute r;
    r.word = route;
    r.fields.compact_tail = 1u;
    return r.word;
}

static __device__ __forceinline__ void SparkTpMeshFastDmaOrDoorbellWarp0(
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band,
    uint32_t degree,uint32_t rank,uint32_t peer_mask,uint32_t is_peer,
    uint64_t slot,uint64_t slot_bytes,uint64_t seq,uint64_t tag,uint64_t bytes,
    uint64_t route,uint64_t half_bytes,uint32_t cap,uint32_t compact,
    uint64_t prior_flags,uint32_t set_cursor)
{
    const uint32_t hyb_done = compact != 0u ? SparkTpMeshHybridTryRingDmaWarp0(
        control,hybrid_ctrl,hybrid_shm,hybrid_band,degree,rank,peer_mask,is_peer,
        slot,slot_bytes,seq,tag,bytes,half_bytes) : 0u;
    if ( threadIdx.x != 0u ) return;
    const uint64_t prev_rseq = SparkTpLdcvU64(&control->round_seq);
    control->round_seq_prior = prev_rseq | (hyb_done == 0u ? SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG : 0u) | prior_flags;
    control->round_seq = tag;
    control->seq = seq;
    if ( set_cursor != 0u ) control->slot_cursor = seq;
    if ( hyb_done == 0u )
    {
        volatile uint64_t *cell = cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(seq)) : entry;
        cell[1] = bytes; cell[2] = slot; cell[3] = SparkTpMeshFastRouteCompact(route,compact);
        __threadfence_system();
        SparkTpStReleaseSysU64(&cell[0],tag);
    }
}

static __device__ __forceinline__ void SparkTpMeshFastClearMbox(volatile uint64_t *mbox)
{
    if ( mbox == 0 ) return;
#pragma unroll
    for ( uint32_t i = 0u; i < SPARK_TP_MESH_FAST_MBOX_WORDS; ++i )
        mbox[i] = 0u;
}

static __device__ __forceinline__ void SparkTpMeshFastMboxFinish(
    SparkTpMeshRoundControl *control,volatile uint64_t *mbox,uint64_t total_blocks,uint32_t last)
{
    if ( threadIdx.x != 0u ||
         atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull != (unsigned long long)total_blocks )
        return;
    const uint64_t end_ns = SparkTpGlobalTimerNs(),start_ns = SparkTpLdcvU64(&control->math_started_ns);
    control->math_blocks_done = 0u; control->math_started_ns = 0u; control->math_finished_ns = 0u;
    SparkTpMeshFastClearMbox(mbox);
    if ( SparkTpLdcvU64(&control->error_word) == 0u )
    {
        if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
        control->slot_cursor = SparkTpLdcvU64(&control->seq);
        if ( last != 0u ) control->rounds_done++;
    }
    __threadfence();
    LmDependentRelease();
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastCompactCap(
    const volatile SparkWeightdMeshWaitRequest *gate)
{
    return gate != 0 && (gate->capabilities & SPARK_WEIGHTD_MESH_CAPABILITY_COMPACT_TAIL) != 0u ? 1u : 0u;
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastStagingAllowPrior(
    const SparkTpMeshRoundControl *control,uint32_t use_half)
{
    if ( use_half == 0u ) return 0u;
    return (SparkTpLdcvU64(&control->round_seq_prior) & SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG) == 0u ? 1u : 0u;
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastShippedAccepts(
    const SparkTpMeshRoundControl *control,uint64_t shipped,uint32_t allow_prior)
{
    const uint64_t rseq = SparkTpLdcvU64(&control->round_seq);
    if ( rseq == 0u || shipped == rseq )
        return 1u;
    if ( allow_prior != 0u )
    {
        const uint64_t prior = SparkTpLdcvU64(&control->round_seq_prior) & ~SPARK_TP_MESH_FAST_PRIOR_FLAGS_MASK;
        return prior == 0u || shipped == prior ? 1u : 0u;
    }
    return 0u;
}

static __device__ __forceinline__ void SparkTpMeshFastShipWait(
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,uint64_t timeout_ns,uint32_t allow_prior)
{
    __shared__ volatile uint64_t s_ship_start;
    if ( SparkTpLdcvU64(&control->round_seq) == 0u ) return;
    const volatile uint64_t *shipped = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    if ( shipped != 0 )
    {
        if ( SparkTpMeshFastShippedAccepts(control,SparkTpLdAcquireU64(shipped),allow_prior) != 0u )
            return;
        const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
        const uint64_t expected = control->cancel_expected;
        const uint64_t poll_limit = timeout_ns < SPARK_TP_MESH_FAST_SHIP_POLL_NS ? timeout_ns : SPARK_TP_MESH_FAST_SHIP_POLL_NS;
        s_ship_start = SparkTpFastTimerNs();
        for ( uint32_t spins = 0u;; )
        {
            if ( SparkTpMeshFastShippedAccepts(control,SparkTpLdAcquireU64(shipped),allow_prior) != 0u )
            {
                const uint64_t base = s_ship_start,now = SparkTpFastTimerNs();
                if ( now > base ) control->source_wait_ns += now - base;
                return;
            }
            if ( (++spins & 31u) == 0u )
            {
                const uint64_t base = s_ship_start,now = SparkTpFastTimerNs();
                if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected ) break;
                if ( now > base && now - base >= poll_limit ) break;
            }
        }
    }
    SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns);
    SparkTpMeshGateAwait(gate,control);
    SparkTpMeshGateGuard(gate,control);
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastEnsureShipped(
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,uint64_t timeout_ns,uint32_t allow_prior,uint64_t prev_rseq)
{
    const volatile uint64_t *shipped = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    if ( prev_rseq != 0u && (shipped == 0 || SparkTpMeshFastShippedAccepts(control,SparkTpLdcvU64(shipped),allow_prior) == 0u) )
    {
        SparkTpMeshFastShipWait(gate,control,timeout_ns,allow_prior);
        return SparkTpLdcvU64(&control->error_word) != 0u ? 1u : 0u;
    }
    return 0u;
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastShipWaitFollower(
    SparkTpMeshRoundControl *control,volatile uint64_t *mbox,
    uint64_t mbox_ok,uint64_t mbox_fail,uint64_t timeout_ns,uint32_t allow_prior)
{
    __shared__ volatile uint64_t s_follow_start;
    const volatile uint64_t *shipped = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    if ( SparkTpLdcvU64(&control->round_seq) == 0u ||
         (shipped != 0 && SparkTpMeshFastShippedAccepts(control,SparkTpLdAcquireU64(shipped),allow_prior) != 0u) )
        return 0u;
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const uint64_t expected = control->cancel_expected;
    const uint64_t limit = timeout_ns > UINT64_MAX / 2u ? UINT64_MAX : 2u * timeout_ns;
    s_follow_start = SparkTpFastTimerNs();
    for ( uint32_t spins = 0u;; )
    {
        if ( mbox != 0 )
        {
            const uint64_t v = SparkTpLdAcquireGpuU64(&mbox[0]);
            if ( v == mbox_ok ) return 0u;
            if ( v == mbox_fail ) return 1u;
        }
        if ( shipped != 0 && SparkTpMeshFastShippedAccepts(control,SparkTpLdAcquireU64(shipped),allow_prior) != 0u )
            return 0u;
        if ( (++spins & 15u) == 0u )
        {
            if ( SparkTpLdcvU64(&control->error_word) != 0u ) return 1u;
            if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected ) return 1u;
            const uint64_t base = s_follow_start,now = SparkTpFastTimerNs();
            if ( now > base && now - base > limit )
            {
                atomicCAS((unsigned long long *)&control->error_word,0ull,UINT64_MAX);
                return 1u;
            }
        }
        __nanosleep(50u);
    }
}

static __global__ void SparkTpMeshFastPublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const void *source,uint64_t bytes,uint64_t route,
    volatile SparkWeightdMeshWaitRequest *gate,uint64_t timeout_ns,uint32_t copy_self,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint64_t s_started,s_seq,s_tag,s_slot,s_route,s_tail_off;
    __shared__ uint32_t s_failed,s_cap,s_compact,s_peer_mask,s_try_inl;
    __shared__ SparkTpMeshHybridGrant s_hyb;
    const uint32_t tid = threadIdx.x,warp_id = tid >> 5u,lane = tid & 31u,warps = blockDim.x >> 5u;
    LmDependentWait();
    if ( tid == 0u )
    {
        s_started = SparkTpFastTimerNs();
        const uint64_t prev_seq = SparkTpLdcvU64(&control->seq),epoch = SparkTpLdcvU64(&control->epoch);
        uint32_t failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 1u ? 1u : 0u;
        const uint64_t sequence = prev_seq + 1u,tag = (epoch << 32u) | sequence;
        const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
        const uint64_t previous = SparkTpLdcvU64(&control->round_seq);
        const SparkTpMeshHybridGpuCache *hcache = &SparkTpMeshHybridDevCache[hybrid_band & (SPARK_WEIGHTD_MESH_BANDS - 1u)];
        const uint32_t cache_ok = hybrid_ctrl != 0 && hybrid_shm != 0 && slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES &&
            hcache->ctrl == hybrid_ctrl && hcache->gen != 0u ? 1u : 0u;
        const uint32_t cap = cache_ok != 0u && previous != 0u && hcache->last_hyb_seq == previous ? 1u : SparkTpMeshFastCompactCap(gate);
        SparkWeightdMeshRoute r;
        r.word = route;
        const uint32_t compact = cap != 0u && bytes != 0u && (bytes & 7u) == 0u && bytes <= slot_bytes - 16u &&
            r.fields.pair == 0u && r.fields.half == SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE && r.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_FULL ? 1u : 0u;
        const uint32_t max_inl = cache_ok != 0u ? hcache->min_max_inline : (hybrid_ctrl != 0 ? hybrid_ctrl->min_max_inline : 0u);
        const uint32_t try_inl = failed == 0u && compact != 0u && copy_self == 0u && hybrid_ctrl != 0 && hybrid_shm != 0 &&
            slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES && (bytes + 8u) <= max_inl ? 1u : 0u;
        s_seq = sequence; s_tag = tag; s_slot = slot; s_route = SparkTpMeshFastRouteCompact(route,compact);
        s_tail_off = compact != 0u ? bytes : slot_bytes - 8u;
        s_cap = cap; s_compact = compact; s_peer_mask = (uint32_t)r.fields.peer_mask; s_try_inl = try_inl;
        if ( failed == 0u && try_inl == 0u )
            failed = SparkTpMeshFastEnsureShipped(gate,control,timeout_ns,cap,previous);
        s_failed = failed; s_hyb.ok = 0u;
    }
    __syncthreads();
    if ( s_failed == 0u && s_compact != 0u && hybrid_ctrl != 0 && hybrid_shm != 0 &&
         slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES && tid < 32u )
    {
        SparkTpMeshHybridAcquireWarp0(control,hybrid_ctrl,hybrid_band,degree,rank,s_peer_mask,(uint32_t)(bytes + 8u),1u,s_try_inl == 0u ? 1u : 0u,0u,s_hyb);
        if ( tid == 0u && s_try_inl != 0u && s_hyb.ok == 0u &&
             SparkTpMeshFastEnsureShipped(gate,control,timeout_ns,s_cap,SparkTpLdcvU64(&control->round_seq)) != 0u )
            s_failed = 1u;
    }
    __syncthreads();
    const uint32_t failed = s_failed;
    const uint64_t slot = s_slot;
    const SparkWeightdMeshHybridPeer *hpeers = s_hyb.ok != 0u ?
        (s_hyb.fast_cred != 0u ? SparkTpMeshHybridDevCache[hybrid_band & (SPARK_WEIGHTD_MESH_BANDS - 1u)].peers : hybrid_ctrl->peers) : 0;
    volatile uint8_t *destination = band + slot * slot_bytes;
    volatile uint64_t *cell = s_cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(s_seq)) : entry;
    if ( failed == 0u )
    {
        const uint64_t slot_base = ((uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * slot_bytes;
        if ( s_hyb.ok != 0u && s_hyb.is_inline != 0u )
        {
            if ( copy_self != 0u )
            {
                if ( bytes != 0u )
                    SparkTpMeshCopyBytesVolatile(destination,source,bytes,tid,blockDim.x);
                if ( tid == 0u )
                    *(volatile uint64_t *)(destination + s_tail_off) = s_tag;
            }
            for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
            {
                const SparkWeightdMeshHybridPeer *hp = &hpeers[i];
                if ( (s_peer_mask & (1u << hp->target)) == 0u ) continue;
                SparkTpMeshHybridWriteInlineWqe(hybrid_shm,hp,s_hyb.pi[i],s_hyb.ds,slot_base,
                    (const uint16_t *)source,bytes >> 1u,bytes,s_tag,lane);
            }
        }
        else
        {
            if ( bytes != 0u )
                SparkTpMeshCopyBytesVolatile(destination,source,bytes,tid,blockDim.x);
            if ( tid == 0u )
            {
                *(volatile uint64_t *)(destination + s_tail_off) = s_tag;
                if ( s_hyb.ok == 0u ) { cell[1] = bytes; cell[2] = slot; cell[3] = s_route; }
            }
            if ( s_hyb.ok != 0u && lane == 0u )
                SparkTpMeshHybridWriteDmaWarps(hpeers,hybrid_shm,s_hyb,s_peer_mask,0u,slot_base,
                    hybrid_ctrl->recv_iova_base + slot_base,(uint32_t)(bytes + 8u),warp_id,warps);
        }
    }
    __syncthreads();
    if ( failed == 0u && tid < 32u )
    {
        if ( s_hyb.ok != 0u )
            SparkTpMeshHybridRingWarp0(hybrid_ctrl,hybrid_shm,hpeers,s_hyb,s_peer_mask,s_seq,s_tag,slot,bytes);
        else if ( tid == 0u )
            __threadfence_system();
    }
    if ( tid != 0u ) return;
    if ( failed != 0u && SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
    if ( failed == 0u && SparkTpLdcvU64(&control->error_word) == 0u )
    {
        control->round_seq_prior = SparkTpLdcvU64(&control->round_seq) | (s_hyb.ok != 0u ? 0u : SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG);
        control->round_seq = s_tag;
        control->seq = s_seq;
        if ( s_hyb.ok == 0u )
            SparkTpStReleaseSysU64(&cell[0],s_tag);
        const uint64_t now = SparkTpFastTimerNs();
        control->copy_ns += now - s_started;
        control->wait_started_ns = now;
    }
    else
        control->wait_started_ns = SparkTpFastTimerNs();
    __threadfence();
    LmDependentRelease();
}

static __global__ void SparkTpMeshFastPublishWideKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const void *source,uint64_t bytes,uint64_t route,
    volatile SparkWeightdMeshWaitRequest *gate,uint64_t timeout_ns,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint64_t s_seq,s_tag,s_slot;
    __shared__ uint32_t s_failed,s_last;
    LmDependentWait();
    if ( threadIdx.x == 0u )
    {
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpFastTimerNs() | 1ull);
        const uint64_t prev_seq = SparkTpLdcvU64(&control->seq),epoch = SparkTpLdcvU64(&control->epoch),sequence = prev_seq + 1u;
        s_seq = sequence; s_tag = (epoch << 32u) | sequence;
        s_slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
        s_failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 1u ? 1u : 0u;
    }
    __syncthreads();
    const uint32_t failed = s_failed;
    const uint64_t slot = s_slot;
    volatile uint8_t *destination = band + slot * slot_bytes;
    if ( failed == 0u && bytes != 0u )
        SparkTpMeshCopyBytesVolatile(destination,source,bytes,(uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || threadIdx.x >= 32u ) return;
    uint32_t go = 0u,compact = 0u,cap = 0u,peer_mask = 0u;
    if ( threadIdx.x == 0u )
    {
        control->publish_blocks_done = 0u;
        cap = SparkTpMeshFastCompactCap(gate);
        SparkTpMeshFastShipWait(gate,control,timeout_ns,cap);
        if ( failed != 0u && SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
        if ( failed == 0u && SparkTpLdcvU64(&control->error_word) == 0u )
        {
            SparkWeightdMeshRoute r;
            r.word = route;
            compact = cap != 0u && bytes != 0u && (bytes & 7u) == 0u && bytes <= slot_bytes - 16u &&
                r.fields.pair == 0u && r.fields.half == SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE && r.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_FULL ? 1u : 0u;
            peer_mask = (uint32_t)r.fields.peer_mask;
            *(volatile uint64_t *)(destination + (compact != 0u ? bytes : slot_bytes - 8u)) = s_tag;
            __threadfence_system();
            go = 1u;
        }
    }
    go = __shfl_sync(0xffffffffu,go,0); compact = __shfl_sync(0xffffffffu,compact,0); peer_mask = __shfl_sync(0xffffffffu,peer_mask,0);
    if ( go != 0u )
        SparkTpMeshFastDmaOrDoorbellWarp0(entry,control,hybrid_ctrl,hybrid_shm,hybrid_band,
            degree,rank,peer_mask,0u,slot,slot_bytes,s_seq,s_tag,bytes,route,0u,cap,compact,0u,0u);
    if ( threadIdx.x != 0u ) return;
    if ( go != 0u )
    {
        const uint64_t pub_start = SparkTpLdcvU64(&control->publish_started_ns),now = SparkTpFastTimerNs();
        if ( now > pub_start ) control->copy_ns += now - pub_start;
        control->wait_started_ns = now;
    }
    else
        control->wait_started_ns = SparkTpFastTimerNs();
    control->publish_started_ns = 0u;
    __threadfence();
    LmDependentRelease();
}

static SparkGpuError SparkTpMeshFastPublish(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const void *source,uint64_t bytes,uint64_t route,
    SparkWeightdMeshWaitRequest *gate,uint64_t peer_mask,uint64_t timeout_ns,uint32_t copy_self)
{
    (void)peer_mask;
    SparkWeightdMeshHybridCtrl *hybrid_ctrl = 0;
    uint8_t *hybrid_shm = 0;
    uint32_t hybrid_band = 0u;
    SparkTpMeshFastHybridResolve(stream,(const void *)entry,(const void *)gate,slot_bytes,&hybrid_ctrl,&hybrid_shm,&hybrid_band);
    if ( bytes <= SPARK_TP_MESH_FAST_WIDE_PUBLISH_BYTES )
        return LM_LAUNCH_DEPENDENT((SparkTpMeshFastPublishKernel),1u,SPARK_TP_MESH_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,entry,control,rank,degree,source,bytes,route,gate,timeout_ns,copy_self,
            hybrid_ctrl,hybrid_shm,hybrid_band);
    uint64_t blocks = (bytes + 8191u) / 8192u;
    const uint32_t cap = SparkTpMeshMultiprocessors() < 32u ? SparkTpMeshMultiprocessors() : 32u;
    blocks = blocks < 2u ? 2u : blocks < cap ? blocks : cap;
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastPublishWideKernel),(uint32_t)blocks,SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,control,rank,degree,source,bytes,route,gate,timeout_ns,
        hybrid_ctrl,hybrid_shm,hybrid_band);
}

static __device__ __forceinline__ void SparkTpMeshFastRunPrefetch(
    LmWeightPrefetchGroup *prefetch,uint32_t resume,uint32_t tid,const volatile uint32_t *stop)
{
    if ( prefetch == 0 || tid < 32u || tid >= 32u + SPARK_TP_MESH_PREFETCH_THREADS ) return;
    const uint32_t spans = prefetch->count;
    const uint64_t extent = LmWeightPrefetchExtent(prefetch,spans);
    const uint64_t stride = (uint64_t)SPARK_TP_MESH_PREFETCH_THREADS * LM_WEIGHT_PREFETCH_PIECE_BYTES;
    uint64_t position = resume != 0u ? prefetch->progress : 0u;
    for ( ; position < extent && *stop == 0u; position += stride )
        LmWeightPrefetchPiece(prefetch,spans,position + (uint64_t)(tid - 32u) * LM_WEIGHT_PREFETCH_PIECE_BYTES);
    if ( tid == 32u ) prefetch->progress = position < extent ? position : extent;
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastWarpPollBatch(
    const volatile uint64_t *tail,uint64_t tag,uint32_t &missing,
    const volatile uint64_t *cancel,uint64_t expected,uint64_t started,uint64_t timeout_ns,
    SparkTpMeshRoundControl *control,uint64_t &arr_landed,uint32_t &fault_out)
{
    __shared__ volatile uint64_t s_poll_started;
    const uint32_t tid = threadIdx.x;
    uint32_t newly = 0u,spins = 0u;
    uint64_t seen = 0u,diag = 0u,fault = 0u;
    if ( tid == 0u ) s_poll_started = started;
    for (;;)
    {
        uint32_t just = 0u;
        if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
        {
            missing = 0u; just = 1u;
            arr_landed = SparkTpFastTimerNs();
        }
        newly = __ballot_sync(0xffffffffu,just != 0u);
        if ( newly != 0u ) break;
        if ( (++spins & 31u) == 0u )
        {
            if ( tid == 0u )
            {
                const uint64_t base = s_poll_started,now = SparkTpFastTimerNs();
                if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                    fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
                else if ( now > base && now - base > timeout_ns )
                    fault = tag;
            }
            fault = __shfl_sync(0xffffffffu,fault,0);
            if ( fault != 0u )
            {
                if ( fault == tag )
                {
                    uint32_t first = (uint32_t)(__ffs(__ballot_sync(0xffffffffu,missing != 0u)) - 1);
                    diag = __shfl_sync(0xffffffffu,seen,first);
                }
                break;
            }
        }
    }
    if ( fault != 0u && tid == 0u && SparkTpLdcvU64(&control->error_word) == 0u )
    {
        control->error_word = fault; control->diag_word = diag;
        __threadfence();
    }
    fault_out = fault != 0u ? 1u : 0u;
    return newly;
}

static __global__ void SparkTpMeshFastPairGatherKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    const volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,uint64_t local_elements,uint32_t pair_only)
{
    __shared__ uint32_t wait_over,failed;
    __shared__ volatile uint64_t s_wait_start;
    const uint32_t tid = threadIdx.x;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    LmDependentWait();
    if ( tid == 0u ) { wait_over = 0u; failed = SparkTpLdcvU64(&control->error_word) != 0u ? 1u : 0u; }
    __syncthreads();
    if ( failed != 0u ) return;
    SparkTpMeshCopyHalves(output + (uint64_t)rank * local_elements,local,local_elements,tid,blockDim.x);
    if ( tid == 0u ) s_wait_start = SparkTpFastTimerNs();
    __syncthreads();
    const uint64_t tag = SparkTpLdcvU64(&control->round_seq),ring = (tag - 1u) & (slots_per_rank - 1u);
    if ( tid < 32u )
    {
        const uint64_t bytes = local_elements * sizeof(uint16_t);
        const uint64_t tail_off = pair_only != 0u && SparkTpMeshFastCompactCap(gate) != 0u &&
            bytes != 0u && (bytes & 7u) == 0u && bytes <= slot_bytes - 16u ? bytes : slot_bytes - 8u;
        const volatile uint64_t *tail = (const volatile uint64_t *)(band + ((uint64_t)tid * slots_per_rank + ring) * slot_bytes + tail_off);
        const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
        const uint64_t expected = control->cancel_expected;
        uint32_t missing = pair_only != 0u ? (tid == (rank ^ 1u) ? 1u : 0u) : (tid < degree && tid != rank ? 1u : 0u);
        uint32_t fault = 0u;
        arr_base = SparkTpLdcvU64(&control->wait_started_ns);
        while ( __any_sync(0xffffffffu,missing != 0u) && fault == 0u )
            (void)SparkTpMeshFastWarpPollBatch(tail,tag,missing,cancel,expected,s_wait_start,timeout_ns,control,arr_landed,fault);
        if ( tid == 0u ) { *(volatile uint32_t *)&wait_over = 1u; arr_end = SparkTpGlobalTimerNs(); if ( fault != 0u ) failed = 1u; }
    }
    else
        SparkTpMeshFastRunPrefetch(prefetch,resume,tid,&wait_over);
    __syncthreads();
    if ( failed != 0u ) return;
    const uint64_t started = tid == 0u ? SparkTpGlobalTimerNs() : 0u;
    const uint32_t partner = rank ^ 1u;
    SparkTpMeshCopyHalvesFromBand(output + (uint64_t)partner * local_elements,
        (const volatile uint16_t *)(band + ((uint64_t)partner * slots_per_rank + ring) * slot_bytes),
        local_elements,tid,blockDim.x);
    __syncthreads();
    if ( tid == 0u )
    {
        control->combine_ns += SparkTpGlobalTimerNs() - started;
        control->slot_cursor = SparkTpLdcvU64(&control->seq);
        control->rounds_done++;
        __threadfence();
        LmDependentRelease();
    }
    if ( tid < 32u ) SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static __device__ __forceinline__ void SparkTpMeshFastPeerStageSlice(
    uint8_t *band,uint64_t slot_bytes,uint64_t slot,
    uint16_t *staging,uint64_t staging_slot_elements,uint64_t staging_half_elems,
    const uint16_t *local,uint32_t rank,uint32_t peer,
    uint64_t total_count,uint64_t src_stride,uint64_t slice,uint64_t doorbell_bytes,uint64_t tag,
    uint32_t mode,uint32_t copy_self,uint32_t compact,uint64_t lane,uint64_t lanes)
{
    if ( peer == rank && copy_self == 0u ) return;
    const uint64_t first = mode == 1u ?
        ((uint64_t)peer * slice < total_count ? (uint64_t)peer * slice : total_count) :
        (uint64_t)peer * src_stride;
    const uint64_t length = mode == 1u ?
        (total_count - first < slice ? total_count - first : slice) : slice;
    uint16_t *dst = peer == rank ?
        (uint16_t *)(band + slot * slot_bytes) :
        (staging + (uint64_t)peer * staging_slot_elements + staging_half_elems);
    if ( length != 0u )
        SparkTpMeshCopyHalves(dst,local + first,length,lane,lanes);
    if ( compact != 0u && peer != rank && lane == 0u )
        *(volatile uint64_t *)((volatile uint8_t *)dst + doorbell_bytes) = tag;
}

static __device__ __forceinline__ void SparkTpMeshFastPeerStageAll(
    uint8_t *band,uint64_t slot_bytes,uint64_t slot,
    uint16_t *staging,uint64_t staging_slot_elements,uint64_t staging_half_elems,
    const uint16_t *local,uint32_t degree,uint32_t rank,
    uint64_t total_count,uint64_t src_stride,uint64_t slice,uint64_t doorbell_bytes,uint64_t tag,
    uint32_t mode,uint32_t copy_self,uint32_t compact,uint32_t tid,uint32_t warp_id,uint32_t lane,uint32_t warps)
{
    if ( degree <= 4u )
        for ( uint32_t p = 0u; p < degree; ++p )
            SparkTpMeshFastPeerStageSlice(band,slot_bytes,slot,staging,staging_slot_elements,staging_half_elems,
                local,rank,p,total_count,src_stride,slice,doorbell_bytes,tag,mode,copy_self,compact,tid,blockDim.x);
    else
        for ( uint32_t p = warp_id; p < degree; p += warps )
            SparkTpMeshFastPeerStageSlice(band,slot_bytes,slot,staging,staging_slot_elements,staging_half_elems,
                local,rank,p,total_count,src_stride,slice,doorbell_bytes,tag,mode,copy_self,compact,lane,32u);
}

static __global__ void SparkTpMeshFastPeerStageSmallKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *local,uint32_t degree,uint32_t rank,
    uint64_t total_count,uint64_t src_stride,uint64_t slice,
    uint64_t doorbell_bytes,uint64_t route,uint64_t timeout_ns,uint32_t mode,uint32_t copy_self,uint32_t allow_compact,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint64_t s_started,s_seq,s_tag,s_slot,s_half_elems;
    __shared__ uint32_t s_failed,s_cap,s_compact,s_use_half,s_peer_mask;
    __shared__ SparkTpMeshHybridGrant s_hyb;
    const uint32_t tid = threadIdx.x,warp_id = tid >> 5u,lane = tid & 31u,warps = blockDim.x >> 5u;
    LmDependentWait();
    if ( tid == 0u )
    {
        s_started = SparkTpFastTimerNs();
        const uint64_t prev_seq = SparkTpLdcvU64(&control->seq),epoch = SparkTpLdcvU64(&control->epoch);
        uint32_t failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 1u ? 1u : 0u;
        const uint64_t sequence = prev_seq + 1u;
        const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
        const uint32_t cap = SparkTpMeshFastCompactCap(gate);
        const uint32_t compact = allow_compact != 0u && cap != 0u && doorbell_bytes != 0u && (doorbell_bytes & 7u) == 0u &&
            doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES && doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
        const uint32_t use_half = compact != 0u && doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ? 1u : 0u;
        const uint32_t allow_prior = SparkTpMeshFastStagingAllowPrior(control,use_half);
        SparkWeightdMeshRoute r;
        r.word = route;
        s_seq = sequence; s_tag = (epoch << 32u) | sequence; s_slot = slot;
        s_cap = cap; s_compact = compact; s_use_half = use_half; s_peer_mask = (uint32_t)r.fields.peer_mask;
        s_half_elems = use_half != 0u ? (SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot) / sizeof(uint16_t)) : 0u;
        if ( failed == 0u )
        {
            SparkTpMeshFastShipWait(gate,control,timeout_ns,allow_prior);
            if ( SparkTpLdcvU64(&control->error_word) != 0u ) failed = 1u;
        }
        s_failed = failed; s_hyb.ok = 0u;
    }
    __syncthreads();
    const uint32_t failed = s_failed;
    const uint64_t slot = s_slot;
    const uint32_t compact = s_compact;
    const uint64_t half_elems = s_half_elems;
    if ( failed == 0u && compact != 0u && hybrid_ctrl != 0 && hybrid_shm != 0 &&
         slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES && tid < 32u )
        SparkTpMeshHybridAcquireWarp0(control,hybrid_ctrl,hybrid_band,degree,rank,s_peer_mask,(uint32_t)(doorbell_bytes + 8u),1u,1u,0u,s_hyb);
    __syncthreads();
    volatile uint64_t *cell = s_cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(s_seq)) : entry;
    if ( failed == 0u )
    {
        const uint64_t slot_base = ((uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * slot_bytes;
        if ( s_hyb.ok != 0u && s_hyb.is_inline != 0u )
        {
            if ( copy_self != 0u )
                SparkTpMeshFastPeerStageSlice(band,slot_bytes,slot,staging,staging_slot_elements,half_elems,
                    local,rank,rank,total_count,src_stride,slice,doorbell_bytes,s_tag,mode,1u,compact,tid,blockDim.x);
            for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
            {
                const SparkWeightdMeshHybridPeer *hp = &hybrid_ctrl->peers[i];
                if ( (s_peer_mask & (1u << hp->target)) == 0u ) continue;
                const uint64_t first = mode == 1u ?
                    ((uint64_t)hp->target * slice < total_count ? (uint64_t)hp->target * slice : total_count) :
                    (uint64_t)hp->target * src_stride;
                const uint64_t length = mode == 1u ?
                    (total_count - first < slice ? total_count - first : slice) : slice;
                SparkTpMeshHybridWriteInlineWqe(hybrid_shm,hp,s_hyb.pi[i],s_hyb.ds,slot_base,
                    local + first,length,doorbell_bytes,s_tag,lane);
            }
        }
        else
        {
            SparkTpMeshFastPeerStageAll(band,slot_bytes,slot,staging,staging_slot_elements,half_elems,
                local,degree,rank,total_count,src_stride,slice,doorbell_bytes,s_tag,mode,copy_self,compact,tid,warp_id,lane,warps);
            if ( tid == 0u )
            {
                *(volatile uint64_t *)(band + slot * slot_bytes + (compact != 0u ? doorbell_bytes : slot_bytes - 8u)) = s_tag;
                if ( s_hyb.ok == 0u )
                {
                    cell[1] = doorbell_bytes;
                    cell[2] = slot;
                    cell[3] = SparkTpMeshFastRouteCompact(route,compact);
                }
            }
            if ( s_hyb.ok != 0u && lane == 0u )
                SparkTpMeshHybridWriteDmaWarps(hybrid_ctrl->peers,hybrid_shm,s_hyb,s_peer_mask,1u,slot_base,
                    hybrid_ctrl->staging_iova_base + (uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES + half_elems * sizeof(uint16_t),
                    (uint32_t)(doorbell_bytes + 8u),warp_id,warps);
        }
    }
    __syncthreads();
    if ( failed == 0u && tid < 32u )
    {
        if ( s_hyb.ok != 0u )
            SparkTpMeshHybridRingWarp0(hybrid_ctrl,hybrid_shm,hybrid_ctrl->peers,s_hyb,s_peer_mask,s_seq,s_tag,slot,doorbell_bytes);
        else if ( tid == 0u )
            __threadfence_system();
    }
    if ( tid != 0u ) return;
    if ( failed != 0u && SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
    if ( failed == 0u && SparkTpLdcvU64(&control->error_word) == 0u )
    {
        const uint64_t stg_flag = s_use_half == 0u && !(s_hyb.ok != 0u && s_hyb.is_inline != 0u) ?
            SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG : 0u;
        control->round_seq_prior = SparkTpLdcvU64(&control->round_seq) |
            (s_hyb.ok != 0u ? 0u : SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG) | stg_flag;
        control->round_seq = s_tag;
        control->seq = s_seq;
        if ( s_hyb.ok == 0u )
            SparkTpStReleaseSysU64(&cell[0],s_tag);
        const uint64_t now = SparkTpFastTimerNs();
        control->copy_ns += now - s_started;
        control->wait_started_ns = now;
    }
    else
        control->wait_started_ns = SparkTpFastTimerNs();
    __threadfence();
    LmDependentRelease();
}

static __global__ __launch_bounds__(SPARK_TP_MESH_THREADS,2) void SparkTpMeshFastPeerStageWideKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *local,volatile uint64_t *mbox,uint32_t degree,uint32_t rank,
    uint64_t total_count,uint64_t src_stride,uint64_t slice,
    uint64_t doorbell_bytes,uint64_t route,uint64_t timeout_ns,uint32_t mode,uint32_t copy_self,uint32_t allow_compact,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint64_t s_seq,s_tag,s_slot,s_half_elems;
    __shared__ uint32_t s_failed,s_last,s_cap,s_compact,s_use_half;
    const uint32_t peer = blockIdx.y,stripe = blockIdx.x,tid = threadIdx.x;
    LmDependentWait();
    if ( tid == 0u )
    {
        const uint64_t prev_seq = SparkTpLdcvU64(&control->seq),epoch = SparkTpLdcvU64(&control->epoch),sequence = prev_seq + 1u;
        uint32_t failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 1u ? 1u : 0u;
        const uint64_t tag = (epoch << 32u) | sequence,slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
        const uint64_t mbox_ok = tag ^ SPARK_TP_MESH_FAST_MBOX_STAGE_OK_SALT,mbox_fail = tag ^ SPARK_TP_MESH_FAST_MBOX_STAGE_FAIL_SALT;
        const uint32_t cap = SparkTpMeshFastCompactCap(gate);
        const uint32_t compact = allow_compact != 0u && cap != 0u && doorbell_bytes != 0u && (doorbell_bytes & 7u) == 0u &&
            doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES && doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
        const uint32_t use_half = compact != 0u && doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ? 1u : 0u;
        const uint32_t allow_prior = SparkTpMeshFastStagingAllowPrior(control,use_half);
        s_seq = sequence; s_tag = tag; s_slot = slot; s_cap = cap; s_compact = compact; s_use_half = use_half;
        s_half_elems = use_half != 0u ? (SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot) / sizeof(uint16_t)) : 0u;
        if ( failed == 0u )
        {
            if ( peer == 0u && stripe == 0u )
            {
                control->publish_started_ns = SparkTpFastTimerNs() | 1ull;
                SparkTpMeshFastShipWait(gate,control,timeout_ns,allow_prior);
                if ( SparkTpLdcvU64(&control->error_word) != 0u ) failed = 1u;
                SparkTpStReleaseGpuU64(&mbox[0],failed == 0u ? mbox_ok : mbox_fail);
            }
            else
                failed = SparkTpMeshFastShipWaitFollower(control,mbox,mbox_ok,mbox_fail,timeout_ns,allow_prior);
        }
        s_failed = failed;
    }
    __syncthreads();
    const uint32_t failed = s_failed;
    const uint64_t slot = s_slot;
    const uint32_t compact = s_compact;
    if ( failed == 0u )
    {
        SparkTpMeshFastPeerStageSlice(band,slot_bytes,slot,staging,staging_slot_elements,s_half_elems,
            local,rank,peer,total_count,src_stride,slice,doorbell_bytes,s_tag,mode,copy_self,compact,
            (uint64_t)stripe * blockDim.x + tid,(uint64_t)blockDim.x * gridDim.x);
        __threadfence_system();
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull ==
            (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || tid >= 32u ) return;
    uint32_t go = 0u,peer_mask = 0u;
    if ( tid == 0u )
    {
        control->publish_blocks_done = 0u;
        mbox[0] = 0u;
        if ( failed != 0u && SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
        if ( failed == 0u && SparkTpLdcvU64(&control->error_word) == 0u )
        {
            SparkWeightdMeshRoute r;
            r.word = route;
            peer_mask = (uint32_t)r.fields.peer_mask;
            *(volatile uint64_t *)(band + slot * slot_bytes + (compact != 0u ? doorbell_bytes : slot_bytes - 8u)) = s_tag;
            __threadfence_system();
            go = 1u;
        }
    }
    go = __shfl_sync(0xffffffffu,go,0); peer_mask = __shfl_sync(0xffffffffu,peer_mask,0);
    if ( go != 0u )
        SparkTpMeshFastDmaOrDoorbellWarp0(entry,control,hybrid_ctrl,hybrid_shm,hybrid_band,
            degree,rank,peer_mask,1u,slot,slot_bytes,s_seq,s_tag,doorbell_bytes,route,
            s_half_elems * sizeof(uint16_t),s_cap,compact,s_use_half == 0u ? SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG : 0u,0u);
    if ( tid != 0u ) return;
    if ( go != 0u )
    {
        const uint64_t pub_start = SparkTpLdcvU64(&control->publish_started_ns),now = SparkTpFastTimerNs();
        if ( pub_start != 0u && now > pub_start ) control->copy_ns += now - pub_start;
        control->wait_started_ns = now;
    }
    else
        control->wait_started_ns = SparkTpFastTimerNs();
    control->publish_started_ns = 0u;
    __threadfence();
    LmDependentRelease();
}

static SparkGpuError SparkTpMeshFastLaunchPeerStage(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *local,volatile uint64_t *mbox,uint32_t degree,uint32_t rank,
    uint64_t total_count,uint64_t src_stride,uint64_t slice,
    uint64_t doorbell_bytes,uint64_t route,uint64_t timeout_ns,uint32_t mode,uint32_t copy_self,uint32_t allow_compact)
{
    SparkWeightdMeshHybridCtrl *hybrid_ctrl = 0;
    uint8_t *hybrid_shm = 0;
    uint32_t hybrid_band = 0u;
    SparkTpMeshFastHybridResolve(stream,(const void *)entry,(const void *)gate,slot_bytes,&hybrid_ctrl,&hybrid_shm,&hybrid_band);
    const uint64_t copy_peers = copy_self != 0u ? (uint64_t)degree : (uint64_t)(degree - 1u);
    const uint64_t total_bytes = copy_peers * slice * sizeof(uint16_t);
    const uint32_t sm_count = SparkTpMeshMultiprocessors();
    if ( total_bytes <= SPARK_TP_MESH_FAST_WIDE_STAGE_BYTES || sm_count < degree )
        return LM_LAUNCH_DEPENDENT((SparkTpMeshFastPeerStageSmallKernel),1u,SPARK_TP_MESH_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,entry,gate,control,staging,staging_slot_elements,
            local,degree,rank,total_count,src_stride,slice,doorbell_bytes,route,timeout_ns,mode,copy_self,allow_compact,
            hybrid_ctrl,hybrid_shm,hybrid_band);
    uint64_t stripes = (slice * sizeof(uint16_t) + 16383u) / 16384u;
    const uint32_t max_stripes = sm_count / degree;
    if ( stripes < 1u ) stripes = 1u;
    if ( stripes > (uint64_t)max_stripes ) stripes = (uint64_t)max_stripes;
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastPeerStageWideKernel),dim3((uint32_t)stripes,degree),SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,gate,control,staging,staging_slot_elements,
        local,mbox,degree,rank,total_count,src_stride,slice,doorbell_bytes,route,timeout_ns,mode,copy_self,allow_compact,
        hybrid_ctrl,hybrid_shm,hybrid_band);
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastMboxPollAll(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,uint64_t tail_off,
    SparkTpMeshRoundControl *control,volatile uint64_t *mbox,
    uint32_t rank,uint32_t degree,uint64_t tag,uint64_t ring,
    uint64_t mbox_ok,uint64_t mbox_fail,uint64_t timeout_ns,uint64_t started,
    volatile uint32_t *wait_over,uint64_t &arr_base,uint64_t &arr_end,uint64_t &arr_landed)
{
    __shared__ volatile uint64_t s_poll_started;
    const uint32_t tid = threadIdx.x;
    const volatile uint64_t *tail = (const volatile uint64_t *)(band +
        ((uint64_t)tid * slots_per_rank + ring) * slot_bytes + tail_off);
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const uint64_t expected = control->cancel_expected;
    uint32_t missing = tid < degree && tid != rank ? 1u : 0u;
    uint32_t spins = 0u;
    uint64_t seen = 0u,diag = 0u,fault = 0u;
    if ( tid == 0u ) s_poll_started = started;
    arr_base = SparkTpLdcvU64(&control->wait_started_ns);
    for (;;)
    {
        if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
        {
            missing = 0u;
            if ( mbox != 0 ) SparkTpStReleaseGpuU64(&mbox[tid],mbox_ok);
            arr_landed = SparkTpFastTimerNs();
        }
        if ( __all_sync(0xffffffffu,missing == 0u) ) break;
        if ( (++spins & 31u) == 0u )
        {
            if ( tid == 0u )
            {
                const uint64_t base = s_poll_started,now = SparkTpFastTimerNs();
                if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                    fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
                else if ( now > base && now - base > timeout_ns )
                    fault = tag;
            }
            fault = __shfl_sync(0xffffffffu,fault,0);
            if ( fault != 0u )
            {
                if ( fault == tag )
                {
                    uint32_t first = (uint32_t)(__ffs(__ballot_sync(0xffffffffu,missing != 0u)) - 1);
                    diag = __shfl_sync(0xffffffffu,seen,first);
                }
                break;
            }
        }
    }
    if ( tid == 0u )
    {
        if ( fault != 0u )
        {
            if ( atomicCAS((unsigned long long *)&control->error_word,0ull,fault) == 0ull )
            {
                control->diag_word = diag;
                __threadfence();
            }
            if ( mbox != 0 ) SparkTpStReleaseGpuU64(&mbox[SPARK_TP_MESH_FAST_MBOX_FAIL],mbox_fail);
        }
        else
        {
            arr_end = SparkTpGlobalTimerNs();
            control->math_started_ns = arr_end;
            __threadfence();
        }
        *wait_over = 1u;
    }
    return fault == 0u ? 1u : 0u;
}

static __device__ __forceinline__ uint32_t SparkTpMeshFastMboxWaitPeer(
    volatile uint64_t *mbox,uint32_t peer,uint64_t mbox_ok,uint64_t mbox_fail,
    const volatile uint64_t *tail,SparkTpMeshRoundControl *control,uint64_t tag,uint64_t timeout_ns)
{
    __shared__ volatile uint64_t s_mb_start;
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const uint64_t expected = control->cancel_expected;
    s_mb_start = SparkTpFastTimerNs();
    for ( uint32_t spins = 0u;; )
    {
        if ( SparkTpLdAcquireGpuU64(&mbox[peer]) == mbox_ok ) return 1u;
        if ( (++spins & 15u) == 0u )
        {
            if ( SparkTpLdAcquireGpuU64(&mbox[SPARK_TP_MESH_FAST_MBOX_FAIL]) == mbox_fail ) return 0u;
            if ( SparkTpLdcvU64(&control->error_word) != 0u ) return 0u;
            if ( (spins & 63u) == 0u )
            {
                if ( tail != 0 && SparkTpLdAcquireU64(tail) == tag ) return 1u;
                if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected )
                {
                    atomicCAS((unsigned long long *)&control->error_word,0ull,SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag);
                    return 0u;
                }
                const uint64_t base = s_mb_start,now = SparkTpFastTimerNs();
                if ( now > base && now - base > timeout_ns )
                {
                    atomicCAS((unsigned long long *)&control->error_word,0ull,tag);
                    return 0u;
                }
            }
        }
        __nanosleep(50u);
    }
}

template<uint32_t WIDE256> struct SparkTpMeshFastVecTraits { using Type = uint4; };
template<> struct SparkTpMeshFastVecTraits<1u> { using Type = SparkTpU256; };

static __device__ __forceinline__ uint32_t SparkTpMeshFastStreamFollowerWait(
    const volatile uint64_t *tail,volatile uint64_t *mbox,uint32_t &missing,
    uint64_t tag,uint64_t mbox_ok,uint64_t mbox_fail,
    const volatile uint64_t *cancel,uint64_t expected,
    const volatile uint64_t *s_wait_start,uint64_t timeout_ns,
    SparkTpMeshRoundControl *control,uint32_t &fault_out)
{
    const uint32_t tid = threadIdx.x;
    uint32_t newly = 0u,spins = 0u,fault = 0u;
    for (;;)
    {
        uint32_t just = 0u;
        const uint32_t check_tail = (spins & 63u) == 63u ? 1u : 0u;
        if ( missing != 0u &&
             (SparkTpLdAcquireGpuU64(&mbox[tid]) == mbox_ok || (check_tail != 0u && SparkTpLdAcquireU64(tail) == tag)) )
        {
            missing = 0u;
            just = 1u;
        }
        newly = __ballot_sync(0xffffffffu,just != 0u);
        if ( newly != 0u ) break;
        if ( (++spins & 15u) == 0u )
        {
            uint32_t fail_flag = 0u;
            if ( tid == 0u )
            {
                if ( SparkTpLdAcquireGpuU64(&mbox[SPARK_TP_MESH_FAST_MBOX_FAIL]) == mbox_fail ||
                     SparkTpLdcvU64(&control->error_word) != 0u )
                    fail_flag = 1u;
                else if ( (spins & 63u) == 0u )
                {
                    if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected )
                    {
                        atomicCAS((unsigned long long *)&control->error_word,0ull,SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag);
                        fail_flag = 1u;
                    }
                    else
                    {
                        const uint64_t wbase = *s_wait_start,now = SparkTpFastTimerNs();
                        if ( now > wbase && now - wbase > timeout_ns )
                        {
                            atomicCAS((unsigned long long *)&control->error_word,0ull,tag);
                            fail_flag = 1u;
                        }
                    }
                }
            }
            if ( __shfl_sync(0xffffffffu,fail_flag,0) != 0u ) { fault = 1u; break; }
            __nanosleep(50u);
        }
    }
    fault_out = fault;
    return newly;
}

template<uint32_t MAX_DEG,uint32_t WIDE256>
static __device__ __forceinline__ void SparkTpMeshFastStreamReduceBody(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,uint64_t tail_off,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,volatile uint16_t *next_dst,
    volatile uint64_t *mbox,uint32_t vecs,uint32_t compute_blocks,
    uint64_t &arr_base,uint64_t &arr_end_out,uint64_t &arr_landed,uint32_t &record_ok)
{
    using Vec = typename SparkTpMeshFastVecTraits<WIDE256>::Type;
    constexpr uint32_t BATCH = MAX_DEG <= 4u ? 4u : 8u;
    __shared__ Vec staged[MAX_DEG][SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS];
    __shared__ uint32_t s_newly;
    __shared__ uint32_t s_failed;
    __shared__ uint64_t s_arr_end;
    __shared__ volatile uint32_t s_wait_over;
    __shared__ volatile uint64_t s_wait_start;
    const uint32_t tid = threadIdx.x;
    const uint64_t tag = SparkTpLdcvU64(&control->round_seq);
    const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
    const uint64_t mbox_ok = tag ^ SPARK_TP_MESH_FAST_MBOX_OK_SALT;
    const uint64_t mbox_fail = tag ^ SPARK_TP_MESH_FAST_MBOX_FAIL_SALT;
    if ( blockIdx.x == compute_blocks )
    {
        if ( tid == 0u ) s_wait_over = 0u;
        __syncthreads();
        if ( tid == 0u )
        {
            (void)SparkTpMeshFastMboxWaitPeer(mbox,SPARK_TP_MESH_FAST_MBOX_DONE,mbox_ok,mbox_fail,0,control,tag,timeout_ns);
            s_wait_over = 1u;
        }
        else
            SparkTpMeshFastRunPrefetch(prefetch,resume,tid,&s_wait_over);
        return;
    }
    const uint64_t stride = slots_per_rank * slot_bytes;
    const uint8_t *base = band + ring * slot_bytes;
    const uint32_t q = blockIdx.x * SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS + tid;
    float sum[WIDE256 != 0u ? 16u : 8u] = {0.0f};
    uint32_t next_peer = 0u,loaded_mask = 1u << rank;
    if ( q < vecs )
    {
        Vec own_val;
        if constexpr ( WIDE256 != 0u )
            own_val = local != 0 ? SparkTpLdNcU256(local + (uint64_t)q * 16u) :
                SparkTpLdCvU256(base + (uint64_t)rank * stride + (uint64_t)q * 32u);
        else
            own_val = local != 0 ? ((const uint4 *)local)[q] :
                SparkTpLdCvU128(base + (uint64_t)rank * stride + (uint64_t)q * 16u);
        if ( rank == 0u )
        {
            if constexpr ( WIDE256 != 0u ) SparkTpAccumU256Bf16(sum,own_val);
            else SparkTpAccumU128Bf16(sum,own_val);
            next_peer = 1u;
        }
        else
            staged[rank][tid] = own_val;
    }
    if ( tid == 0u ) { s_newly = 0u; s_failed = 0u; s_arr_end = 0u; s_wait_start = SparkTpFastTimerNs(); }
    __syncthreads();
    const uint32_t is_master = blockIdx.x == 0u ? 1u : 0u;
    const volatile uint64_t *tail = tid < 32u ?
        (const volatile uint64_t *)(band + ((uint64_t)tid * slots_per_rank + ring) * slot_bytes + tail_off) : 0;
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const uint64_t expected = control->cancel_expected;
    arr_base = is_master != 0u ? SparkTpLdcvU64(&control->wait_started_ns) : 0u;
    uint32_t missing = tid < degree && tid != rank ? 1u : 0u;
    uint32_t remaining_mask = ((1u << degree) - 1u) & ~(1u << rank);
    while ( remaining_mask != 0u )
    {
        if ( tid < 32u )
        {
            uint32_t newly = 0u,spins = 0u;
            uint64_t seen = 0u,diag = 0u,fault = 0u;
            if ( is_master != 0u )
            {
                for (;;)
                {
                    uint32_t just = 0u;
                    if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
                    {
                        missing = 0u;
                        just = 1u;
                        SparkTpStReleaseGpuU64(&mbox[tid],mbox_ok);
                        arr_landed = SparkTpFastTimerNs();
                    }
                    newly = __ballot_sync(0xffffffffu,just != 0u);
                    if ( newly != 0u ) break;
                    if ( (++spins & 31u) == 0u )
                    {
                        if ( tid == 0u )
                        {
                            const uint64_t wbase = s_wait_start,now = SparkTpFastTimerNs();
                            if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                                fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
                            else if ( now > wbase && now - wbase > timeout_ns )
                                fault = tag;
                        }
                        fault = __shfl_sync(0xffffffffu,fault,0);
                        if ( fault != 0u )
                        {
                            if ( fault == tag )
                            {
                                uint32_t first = (uint32_t)(__ffs(__ballot_sync(0xffffffffu,missing != 0u)) - 1);
                                diag = __shfl_sync(0xffffffffu,seen,first);
                            }
                            break;
                        }
                    }
                }
                if ( tid == 0u )
                {
                    s_newly = newly;
                    if ( fault != 0u )
                    {
                        if ( atomicCAS((unsigned long long *)&control->error_word,0ull,fault) == 0ull )
                        {
                            control->diag_word = diag;
                            __threadfence();
                        }
                        SparkTpStReleaseGpuU64(&mbox[SPARK_TP_MESH_FAST_MBOX_FAIL],mbox_fail);
                        s_failed = 1u;
                    }
                    else if ( (remaining_mask & ~newly) == 0u )
                    {
                        SparkTpStReleaseGpuU64(&mbox[SPARK_TP_MESH_FAST_MBOX_DONE],mbox_ok);
                        s_arr_end = SparkTpGlobalTimerNs();
                        control->math_started_ns = s_arr_end;
                        __threadfence();
                    }
                }
            }
            else
            {
                uint32_t f_out = 0u;
                newly = SparkTpMeshFastStreamFollowerWait(tail,mbox,missing,tag,mbox_ok,mbox_fail,
                    cancel,expected,&s_wait_start,timeout_ns,control,f_out);
                if ( tid == 0u )
                {
                    s_newly = newly;
                    if ( f_out != 0u ) s_failed = 1u;
                }
            }
        }
        __syncthreads();
        if ( s_failed != 0u ) break;
        uint32_t mask = s_newly;
        remaining_mask &= ~mask;
        while ( mask != 0u )
        {
            uint32_t ps[BATCH],n = 0u;
            Vec v[BATCH];
            while ( mask != 0u && n < BATCH )
            {
                uint32_t p = (uint32_t)(__ffs(mask) - 1);
                mask &= mask - 1u;
                ps[n++] = p;
            }
            if ( q < vecs )
            {
#pragma unroll
                for ( uint32_t j = 0u; j < BATCH; ++j )
                    if ( j < n )
                    {
                        if constexpr ( WIDE256 != 0u )
                            v[j] = SparkTpLdCvU256(base + (uint64_t)ps[j] * stride + (uint64_t)q * 32u);
                        else
                            v[j] = SparkTpLdCvU128(base + (uint64_t)ps[j] * stride + (uint64_t)q * 16u);
                    }
#pragma unroll
                for ( uint32_t j = 0u; j < BATCH; ++j )
                    if ( j < n )
                    {
                        uint32_t p = ps[j];
                        if ( p == next_peer )
                        {
                            if constexpr ( WIDE256 != 0u ) SparkTpAccumU256Bf16(sum,v[j]);
                            else SparkTpAccumU128Bf16(sum,v[j]);
                            next_peer++;
                            while ( next_peer < degree && (loaded_mask & (1u << next_peer)) != 0u )
                            {
                                if constexpr ( WIDE256 != 0u ) SparkTpAccumU256Bf16(sum,staged[next_peer][tid]);
                                else SparkTpAccumU128Bf16(sum,staged[next_peer][tid]);
                                next_peer++;
                            }
                        }
                        else
                            staged[p][tid] = v[j];
                        loaded_mask |= 1u << p;
                    }
            }
        }
        __syncthreads();
    }
    if ( s_failed == 0u && q < vecs )
    {
        if constexpr ( WIDE256 != 0u )
        {
            SparkTpU256 packed = SparkTpPackU256Bf16(sum);
            SparkTpStU256(output + (uint64_t)q * 16u,packed);
            if ( next_dst != 0 ) SparkTpStVolatileU256(next_dst + (uint64_t)q * 16u,packed);
        }
        else
        {
            uint4 packed = SparkTpPackU128Bf16(sum);
            ((uint4 *)output)[q] = packed;
            if ( next_dst != 0 ) SparkTpStVolatileU128(next_dst + (uint64_t)q * 8u,packed);
        }
    }
    if ( s_failed == 0u && next_dst != 0 ) __threadfence_system();
    if ( is_master != 0u && s_failed == 0u )
    {
        arr_end_out = s_arr_end;
        record_ok = 1u;
    }
}

template<uint32_t MAX_DEG,uint32_t WIDE256>
static __global__ void SparkTpMeshFastStreamAllReduceKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    const volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,volatile uint64_t *mbox,uint32_t vecs,uint32_t compute_blocks,uint64_t compact_bytes)
{
    const uint32_t tid = threadIdx.x;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    uint32_t record_ok = 0u;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) == 0u )
    {
        const uint64_t tail_off = SparkTpMeshFastCompactCap(gate) != 0u && compact_bytes != 0u &&
            (compact_bytes & 7u) == 0u && compact_bytes <= slot_bytes - 16u ? compact_bytes : slot_bytes - 8u;
        SparkTpMeshFastStreamReduceBody<MAX_DEG,WIDE256>(band,slot_bytes,slots_per_rank,tail_off,control,
            rank,degree,timeout_ns,prefetch,resume,local,output,0,mbox,vecs,compute_blocks,
            arr_base,arr_end,arr_landed,record_ok);
    }
    __syncthreads();
    SparkTpMeshFastMboxFinish(control,mbox,(uint64_t)gridDim.x,1u);
    if ( record_ok != 0u && tid < 32u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

template<uint32_t MAX_DEG,uint32_t WIDE256>
static __global__ void SparkTpMeshFastRsagStreamReducePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local_slice,uint16_t *output,uint64_t out_offset,
    volatile uint64_t *mbox,uint32_t vecs,uint32_t compute_blocks,
    uint64_t in_bytes,uint64_t next_doorbell_bytes,uint64_t next_route,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint32_t s_last;
    const uint32_t tid = threadIdx.x;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    uint32_t record_ok = 0u;
    LmDependentWait();
    const uint64_t seq = SparkTpLdcvU64(&control->seq);
    const uint32_t next_overflow = seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t next_seq = seq + 1u;
    const uint64_t next_slot = (uint64_t)rank * slots_per_rank + ((next_seq - 1u) & (slots_per_rank - 1u));
    const uint32_t cap = SparkTpMeshFastCompactCap(gate);
    if ( SparkTpLdcvU64(&control->error_word) == 0u )
    {
        const uint64_t in_tail_off = cap != 0u && in_bytes != 0u && (in_bytes & 7u) == 0u &&
            in_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES && in_bytes <= slot_bytes - 16u ?
            in_bytes : slot_bytes - 8u;
        volatile uint16_t *next_dst = next_overflow == 0u ? (volatile uint16_t *)(band + next_slot * slot_bytes) : 0;
        SparkTpMeshFastStreamReduceBody<MAX_DEG,WIDE256>(band,slot_bytes,slots_per_rank,in_tail_off,control,
            rank,degree,timeout_ns,prefetch,resume,local_slice,output + out_offset,next_dst,mbox,vecs,compute_blocks,
            arr_base,arr_end,arr_landed,record_ok);
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull ==
            (unsigned long long)gridDim.x ? 1u : 0u;
    __syncthreads();
    if ( s_last != 0u && tid < 32u )
    {
        uint32_t go = 0u,out_compact = 0u,peer_mask = 0u;
        const uint64_t next_tag = (SparkTpLdcvU64(&control->epoch) << 32u) | next_seq;
        if ( tid == 0u )
        {
            const uint64_t end_ns = SparkTpGlobalTimerNs();
            const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
            control->publish_blocks_done = 0u;
            control->math_started_ns = 0u;
            control->math_finished_ns = 0u;
            control->math_blocks_done = 0u;
            SparkTpMeshFastClearMbox(mbox);
            if ( SparkTpLdcvU64(&control->error_word) == 0u )
            {
                if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
                if ( next_overflow != 0u )
                    control->error_word = UINT64_MAX;
                else
                {
                    out_compact = cap != 0u && next_doorbell_bytes != 0u &&
                        (next_doorbell_bytes & 7u) == 0u && next_doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
                    *(volatile uint64_t *)(band + next_slot * slot_bytes + (out_compact != 0u ? next_doorbell_bytes : slot_bytes - 8u)) = next_tag;
                    SparkTpMeshFastShipWait(gate,control,timeout_ns,cap);
                    if ( SparkTpLdcvU64(&control->error_word) == 0u )
                    {
                        SparkWeightdMeshRoute r;
                        r.word = next_route;
                        peer_mask = (uint32_t)r.fields.peer_mask;
                        __threadfence_system();
                        go = 1u;
                    }
                }
            }
        }
        go = __shfl_sync(0xffffffffu,go,0); out_compact = __shfl_sync(0xffffffffu,out_compact,0); peer_mask = __shfl_sync(0xffffffffu,peer_mask,0);
        if ( go != 0u )
            SparkTpMeshFastDmaOrDoorbellWarp0(entry,control,hybrid_ctrl,hybrid_shm,hybrid_band,
                degree,rank,peer_mask,0u,next_slot,slot_bytes,next_seq,next_tag,next_doorbell_bytes,next_route,0u,cap,out_compact,0u,1u);
        if ( tid == 0u )
        {
            if ( go != 0u ) control->wait_started_ns = SparkTpGlobalTimerNs();
            __threadfence();
            LmDependentRelease();
        }
    }
    if ( record_ok != 0u && tid < 32u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshFastRsagWideReducePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,
    void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t first,uint64_t span,
    uint64_t next_doorbell_bytes,uint64_t next_route,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint32_t s_last;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint64_t seq = SparkTpLdcvU64(&control->seq);
    const uint64_t ring = (SparkTpLdcvU64(&control->round_seq) - 1u) & (slots_per_rank - 1u);
    const uint32_t next_overflow = seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t next_seq = seq + 1u;
    const uint64_t next_slot = (uint64_t)rank * slots_per_rank + ((next_seq - 1u) & (slots_per_rank - 1u));
    volatile uint16_t *next_dst = next_overflow == 0u ? (volatile uint16_t *)(band + next_slot * slot_bytes) : 0;
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->math_started_ns,0ull,started);
    SparkTpMeshHardwareDirectBody(band,slot_bytes,slots_per_rank,ring,rank,degree,output,local_elements,begin,count,1u,first,span,first,UINT32_MAX,next_dst);
    if ( next_dst != 0 ) __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || threadIdx.x >= 32u ) return;
    uint32_t go = 0u,out_compact = 0u,cap = 0u,peer_mask = 0u;
    uint64_t next_tag = 0u;
    if ( threadIdx.x == 0u )
    {
        control->publish_blocks_done = 0u;
        const uint64_t end_ns = SparkTpGlobalTimerNs(),begin_ns = SparkTpLdcvU64(&control->math_started_ns);
        control->math_started_ns = 0u; control->math_finished_ns = 0u; control->math_blocks_done = 0u;
        if ( begin_ns != 0u && end_ns > begin_ns ) control->combine_ns += end_ns - begin_ns;
        if ( next_overflow != 0u )
        { if ( SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX; }
        else if ( SparkTpLdcvU64(&control->error_word) == 0u )
        {
            next_tag = (SparkTpLdcvU64(&control->epoch) << 32u) | next_seq;
            cap = SparkTpMeshFastCompactCap(gate);
            out_compact = cap != 0u && next_doorbell_bytes != 0u &&
                (next_doorbell_bytes & 7u) == 0u && next_doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
            *(volatile uint64_t *)(band + next_slot * slot_bytes + (out_compact != 0u ? next_doorbell_bytes : slot_bytes - 8u)) = next_tag;
            SparkTpMeshFastShipWait(gate,control,timeout_ns,cap);
            if ( SparkTpLdcvU64(&control->error_word) == 0u )
            {
                SparkWeightdMeshRoute r;
                r.word = next_route;
                peer_mask = (uint32_t)r.fields.peer_mask;
                __threadfence_system();
                go = 1u;
            }
        }
    }
    go = __shfl_sync(0xffffffffu,go,0); out_compact = __shfl_sync(0xffffffffu,out_compact,0);
    peer_mask = __shfl_sync(0xffffffffu,peer_mask,0); next_tag = __shfl_sync(0xffffffffu,next_tag,0);
    if ( go != 0u )
        SparkTpMeshFastDmaOrDoorbellWarp0(entry,control,hybrid_ctrl,hybrid_shm,hybrid_band,
            degree,rank,peer_mask,0u,next_slot,slot_bytes,next_seq,next_tag,next_doorbell_bytes,next_route,0u,cap,out_compact,0u,1u);
    if ( threadIdx.x != 0u ) return;
    if ( go != 0u ) control->wait_started_ns = SparkTpGlobalTimerNs();
    __threadfence();
    LmDependentRelease();
}

static __global__ __launch_bounds__(SPARK_TP_MESH_THREADS,2) void SparkTpMeshFastWideStreamGatherKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    const volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,volatile uint64_t *mbox,
    uint64_t begin,uint64_t total_count,
    uint64_t slice_elements,uint32_t is_slice_gather,uint32_t skip_self,uint32_t last)
{
    __shared__ uint32_t s_go;
    __shared__ volatile uint32_t s_wait_over;
    __shared__ volatile uint64_t s_wait_start;
    const uint32_t peer = blockIdx.y,stripe = blockIdx.x,tid = threadIdx.x;
    const uint64_t lane = (uint64_t)stripe * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    uint32_t ok = 0u;
    LmDependentWait();
    const uint32_t entry_failed = SparkTpLdcvU64(&control->error_word) != 0u ? 1u : 0u;
    if ( entry_failed == 0u )
    {
        const uint64_t tag = SparkTpLdcvU64(&control->round_seq);
        const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
        const uint64_t mbox_ok = tag ^ SPARK_TP_MESH_FAST_MBOX_OK_SALT;
        const uint64_t mbox_fail = tag ^ SPARK_TP_MESH_FAST_MBOX_FAIL_SALT;
        const uint32_t cap = SparkTpMeshFastCompactCap(gate);
        const uint64_t start = is_slice_gather != 0u ? ((uint64_t)peer * slice_elements < total_count ? (uint64_t)peer * slice_elements : total_count) : 0u;
        const uint64_t length = is_slice_gather != 0u ? (total_count - start < slice_elements ? total_count - start : slice_elements) : slice_elements;
        const uint64_t dst_off = is_slice_gather != 0u ? (begin + start) : (begin + (uint64_t)peer * slice_elements);
        if ( peer == rank )
        {
            if ( stripe == 0u )
            {
                if ( tid == 0u ) { s_wait_over = 0u; s_wait_start = SparkTpFastTimerNs(); }
                __syncthreads();
                if ( tid < 32u )
                {
                    uint64_t p_db;
                    if ( is_slice_gather != 0u )
                    {
                        const uint64_t p_start = (uint64_t)tid * slice_elements < total_count ? (uint64_t)tid * slice_elements : total_count;
                        const uint64_t p_len = total_count - p_start < slice_elements ? total_count - p_start : slice_elements;
                        const uint64_t p_gb = (p_len * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
                        p_db = p_gb != 0u ? p_gb : 8u;
                    }
                    else
                        p_db = (slice_elements * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
                    const uint64_t tail_off = cap != 0u && p_db != 0u &&
                        (p_db & 7u) == 0u && p_db <= slot_bytes - 16u ? p_db : slot_bytes - 8u;
                    ok = SparkTpMeshFastMboxPollAll(band,slot_bytes,slots_per_rank,tail_off,control,mbox,
                        rank,degree,tag,ring,mbox_ok,mbox_fail,timeout_ns,s_wait_start,
                        &s_wait_over,arr_base,arr_end,arr_landed);
                }
                else
                {
                    if ( skip_self == 0u && length != 0u )
                    {
                        const uint64_t lane_self = (uint64_t)(tid - 32u),lanes_self = lanes - 32u;
                        if ( local != 0 )
                            SparkTpMeshCopyHalves(output + dst_off,local,length,lane_self,lanes_self);
                        else
                            SparkTpMeshCopyHalvesFromBand(output + dst_off,
                                (const volatile uint16_t *)(band + ((uint64_t)rank * slots_per_rank + ring) * slot_bytes),
                                length,lane_self,lanes_self);
                    }
                    SparkTpMeshFastRunPrefetch(prefetch,resume,tid,&s_wait_over);
                }
            }
            else if ( skip_self == 0u && length != 0u )
            {
                const uint64_t lane_self = lane - 32u,lanes_self = lanes - 32u;
                if ( local != 0 )
                    SparkTpMeshCopyHalves(output + dst_off,local,length,lane_self,lanes_self);
                else
                    SparkTpMeshCopyHalvesFromBand(output + dst_off,
                        (const volatile uint16_t *)(band + ((uint64_t)rank * slots_per_rank + ring) * slot_bytes),
                        length,lane_self,lanes_self);
            }
        }
        else if ( length != 0u )
        {
            if ( tid == 0u )
            {
                const uint64_t p_gb = (length * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
                const uint64_t p_db = is_slice_gather != 0u ? (p_gb != 0u ? p_gb : 8u) : ((slice_elements * sizeof(uint16_t) + 7u) & ~UINT64_C(7));
                const uint64_t tail_off = cap != 0u && p_db != 0u &&
                    (p_db & 7u) == 0u && p_db <= slot_bytes - 16u ? p_db : slot_bytes - 8u;
                const volatile uint64_t *tail = (const volatile uint64_t *)(band +
                    ((uint64_t)peer * slots_per_rank + ring) * slot_bytes + tail_off);
                s_go = SparkTpMeshFastMboxWaitPeer(mbox,peer,mbox_ok,mbox_fail,tail,control,tag,timeout_ns);
            }
            __syncthreads();
            if ( s_go != 0u )
                SparkTpMeshCopyHalvesFromBand(output + dst_off,
                    (const volatile uint16_t *)(band + ((uint64_t)peer * slots_per_rank + ring) * slot_bytes),
                    length,lane,lanes);
        }
    }
    __syncthreads();
    SparkTpMeshFastMboxFinish(control,mbox,(uint64_t)gridDim.x * gridDim.y,last);
    if ( peer == rank && stripe == 0u && tid < 32u && ok != 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static inline uint32_t SparkTpMeshFastGatherStripes(uint64_t slice_elements,uint32_t degree)
{
    const uint32_t max_stripes = SparkTpMeshMultiprocessors() / degree;
    const uint32_t wanted = slice_elements > 65536u ? 4u : (slice_elements > 16384u ? 2u : 1u);
    if ( max_stripes < 1u ) return 1u;
    return wanted < max_stripes ? wanted : max_stripes;
}

static SparkGpuError SparkTpMeshFastLaunchWideStreamGather(SparkGpuStream stream,
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    const SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint16_t *local,uint16_t *output,volatile uint64_t *mbox,uint64_t begin,uint64_t total_count,
    uint64_t slice_elements,uint32_t is_slice_gather,uint32_t skip_self,uint32_t last,uint64_t timeout_ns)
{
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const dim3 grid(SparkTpMeshFastGatherStripes(slice_elements,degree),degree);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastWideStreamGatherKernel),grid,SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,gate,control,rank,degree,timeout_ns,prefetch,resume,
        local,output,mbox,begin,total_count,slice_elements,is_slice_gather,skip_self,last);
}

static inline uint32_t SparkTpMeshFastCanStreamReduce(
    uint64_t elements,uint32_t degree,uint32_t wide256,uint32_t has_prefetch)
{
    const uint32_t vecs = (uint32_t)(elements / (wide256 != 0u ? 16u : 8u));
    const uint32_t compute_blocks = (vecs + SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS - 1u) /
        SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS;
    return compute_blocks + has_prefetch <= SparkTpMeshMultiprocessors() ? 1u : 0u;
}

#define SPARK_TP_MESH_FAST_FUSED_THREADS 512u

static __device__ __forceinline__ void SparkTpMeshFastRsagPublishRs(
    uint8_t *band,uint64_t slot_bytes,volatile uint64_t *entry,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *local_begin,const uint16_t *rs_smem,uint32_t degree,uint32_t rank,
    uint64_t count,uint64_t slice,uint64_t in_bytes,uint64_t rs_route,
    uint64_t rs_seq,uint64_t rs_tag,uint64_t rs_slot,uint64_t half_elems,uint64_t in_tail_off,
    uint32_t cap,uint32_t compact,uint64_t started,uint64_t prev_rseq,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band,
    SparkWeightdMeshHybridPeer *s_peers,SparkTpMeshHybridGrant &s_hyb,
    volatile uint64_t &s_wait_start,uint64_t &s_arr_end,uint64_t &s_copy_delta)
{
    const uint32_t tid = threadIdx.x,warp_id = tid >> 5u,lane = tid & 31u,warps = blockDim.x >> 5u;
    SparkWeightdMeshRoute r;
    r.word = rs_route;
    const uint32_t peer_mask = (uint32_t)r.fields.peer_mask;
    const uint64_t slot_base = ((uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + rs_slot) * slot_bytes;
    if ( s_hyb.ok != 0u && s_hyb.fast_cred == 0u )
    {
        for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
            if ( lane < 4u )
                ((uint4 *)&s_peers[i])[lane] = ((const uint4 *)&hybrid_ctrl->peers[i])[lane];
        __syncwarp();
    }
    if ( s_hyb.ok != 0u && s_hyb.is_inline != 0u )
    {
        for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
        {
            const SparkWeightdMeshHybridPeer *hp = &s_peers[i];
            if ( (peer_mask & (1u << hp->target)) == 0u ) continue;
            const uint64_t p_first = (uint64_t)hp->target * slice < count ? (uint64_t)hp->target * slice : count;
            const uint64_t p_len = count - p_first < slice ? count - p_first : slice;
            SparkTpMeshHybridWriteInlineWqe(hybrid_shm,hp,s_hyb.pi[i],s_hyb.ds,slot_base,
                rs_smem != 0 ? (rs_smem + (uint64_t)hp->target * 1024u) : (local_begin + p_first),
                rs_smem != 0 ? slice : p_len,in_bytes,rs_tag,lane);
        }
    }
    else
    {
        SparkTpMeshFastPeerStageAll(band,slot_bytes,rs_slot,staging,staging_slot_elements,half_elems,
            local_begin,degree,rank,count,0u,slice,in_bytes,rs_tag,1u,0u,compact,tid,warp_id,lane,warps);
        if ( tid == 0u )
        {
            *(volatile uint64_t *)(band + rs_slot * slot_bytes + in_tail_off) = rs_tag;
            if ( s_hyb.ok == 0u )
            {
                volatile uint64_t *cell = cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(rs_seq)) : entry;
                cell[1] = in_bytes; cell[2] = rs_slot; cell[3] = SparkTpMeshFastRouteCompact(rs_route,compact);
            }
        }
        if ( s_hyb.ok != 0u && lane == 0u )
            SparkTpMeshHybridWriteDmaWarps(s_peers,hybrid_shm,s_hyb,peer_mask,1u,slot_base,
                hybrid_ctrl->staging_iova_base + (uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES + half_elems * sizeof(uint16_t),
                (uint32_t)(in_bytes + 8u),warp_id,warps);
    }
    __syncthreads();
    if ( tid < 32u )
    {
        if ( s_hyb.ok != 0u )
            SparkTpMeshHybridRingWarp0(hybrid_ctrl,hybrid_shm,s_peers,s_hyb,peer_mask,rs_seq,rs_tag,rs_slot,in_bytes);
        else if ( tid == 0u )
            __threadfence_system();
    }
    if ( tid == 0u )
    {
        control->round_seq_prior = prev_rseq | (s_hyb.ok != 0u ? 0u : SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG);
        control->round_seq = rs_tag;
        control->seq = rs_seq;
        if ( s_hyb.ok == 0u )
            SparkTpStReleaseSysU64(&(cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(rs_seq)) : entry)[0],rs_tag);
        const uint64_t now = SparkTpFastTimerNs();
        s_copy_delta = now > started ? now - started : 0u;
        control->wait_started_ns = now;
        s_wait_start = now;
        s_arr_end = 0u;
    }
}

static __device__ __forceinline__ void SparkTpMeshFastRsagPublishAg(
    uint8_t *band,uint64_t slot_bytes,volatile uint64_t *entry,
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint32_t degree,uint32_t rank,uint32_t vecs,const uint4 *smem_row,
    uint64_t rs_tag,uint64_t ag_seq,uint64_t ag_tag,uint64_t ag_slot,
    uint64_t ag_doorbell_bytes,uint64_t ag_route,uint64_t ag_tail_off,
    uint32_t cap,uint32_t ag_compact,uint64_t timeout_ns,uint32_t rs_was_hybrid,uint32_t ag_reval,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band,
    SparkWeightdMeshHybridPeer *s_peers,SparkTpMeshHybridGrant &s_hyb,
    uint32_t &s_failed,volatile uint64_t &s_wait_start,uint64_t &s_arr_end,uint64_t &s_comb_acc)
{
    const uint32_t tid = threadIdx.x,warp_id = tid >> 5u,lane = tid & 31u,warps = blockDim.x >> 5u;
    volatile uint16_t *ag_dst = (volatile uint16_t *)(band + ag_slot * slot_bytes);
    SparkWeightdMeshRoute r;
    r.word = ag_route;
    const uint32_t peer_mask = (uint32_t)r.fields.peer_mask;
    const uint32_t smem_inline_ok = vecs <= 128u && ag_doorbell_bytes == (uint64_t)vecs * 16u ? 1u : 0u;
    if ( ag_reval == 0u )
    {
        if ( ag_compact != 0u && hybrid_ctrl != 0 && hybrid_shm != 0 &&
             slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES )
        {
            if ( tid < 32u )
            {
                uint32_t ok = s_hyb.ok != 0u && s_hyb.ag_ok != 0u ?
                    SparkTpMeshHybridRevalidateWarp(hybrid_ctrl,peer_mask,s_peers,s_hyb) : 0u;
                if ( ok == 0u )
                    SparkTpMeshHybridAcquireWarp0(control,hybrid_ctrl,hybrid_band,degree,rank,peer_mask,
                        (uint32_t)(ag_doorbell_bytes + 8u),smem_inline_ok,1u,rs_was_hybrid,s_hyb);
            }
        }
        else if ( tid == 0u )
            s_hyb.ok = 0u;
        __syncthreads();
        if ( s_hyb.ok == 0u )
        {
            if ( tid == 0u && SparkTpMeshFastEnsureShipped(gate,control,timeout_ns,cap,1u) != 0u ) s_failed = 1u;
            __syncthreads();
            if ( s_failed != 0u ) return;
        }
        if ( s_hyb.ok != 0u && s_hyb.ag_ok == 0u )
        {
            for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
                if ( lane < 4u )
                    ((uint4 *)&s_peers[i])[lane] = ((const uint4 *)&hybrid_ctrl->peers[i])[lane];
            __syncwarp();
        }
    }
    const uint64_t slot_base = ((uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + ag_slot) * slot_bytes;
    if ( s_hyb.ok != 0u && s_hyb.is_inline != 0u )
    {
        const uint16_t *src16 = (const uint16_t *)smem_row;
        for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
        {
            const SparkWeightdMeshHybridPeer *hp = &s_peers[i];
            if ( (peer_mask & (1u << hp->target)) == 0u ) continue;
            SparkTpMeshHybridWriteInlineWqe(hybrid_shm,hp,s_hyb.pi[i],s_hyb.ds,slot_base,
                src16,(uint64_t)vecs * 8u,ag_doorbell_bytes,ag_tag,lane);
        }
    }
    else
    {
        if ( vecs <= 128u && tid < vecs )
            SparkTpStVolatileU128(ag_dst + (uint64_t)tid * 8u,smem_row[tid]);
        if ( tid == 0u )
        {
            *(volatile uint64_t *)(band + ag_slot * slot_bytes + ag_tail_off) = ag_tag;
            if ( s_hyb.ok == 0u )
            {
                volatile uint64_t *ag_cell = cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(ag_seq)) : entry;
                ag_cell[1] = ag_doorbell_bytes; ag_cell[2] = ag_slot; ag_cell[3] = SparkTpMeshFastRouteCompact(ag_route,ag_compact);
            }
        }
        if ( s_hyb.ok != 0u && lane == 0u )
            SparkTpMeshHybridWriteDmaWarps(s_peers,hybrid_shm,s_hyb,peer_mask,0u,slot_base,
                hybrid_ctrl->recv_iova_base + slot_base,(uint32_t)(ag_doorbell_bytes + 8u),warp_id,warps);
    }
    __syncthreads();
    if ( tid < 32u )
    {
        if ( s_hyb.ok != 0u )
            SparkTpMeshHybridRingWarp0(hybrid_ctrl,hybrid_shm,s_peers,s_hyb,peer_mask,ag_seq,ag_tag,ag_slot,ag_doorbell_bytes);
        else if ( tid == 0u )
            __threadfence_system();
    }
    if ( tid == 0u )
    {
        const uint64_t end_ns = SparkTpFastTimerNs();
        s_comb_acc = s_arr_end != 0u && end_ns > s_arr_end ? end_ns - s_arr_end : 0u;
        control->round_seq_prior = rs_tag | (s_hyb.ok != 0u ? 0u : SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG);
        control->round_seq = ag_tag;
        control->seq = ag_seq;
        control->slot_cursor = ag_seq;
        if ( s_hyb.ok == 0u )
            SparkTpStReleaseSysU64(&(cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(ag_seq)) : entry)[0],ag_tag);
        const uint64_t ag_now = SparkTpFastTimerNs();
        control->wait_started_ns = ag_now;
        s_wait_start = ag_now;
        s_arr_end = 0u;
    }
}

static __global__ __launch_bounds__(SPARK_TP_MESH_FAST_FUSED_THREADS,1u) void SparkTpMeshFastRsagFusedKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *local,uint16_t *output,uint32_t degree,uint32_t rank,
    uint64_t begin,uint64_t count,uint64_t slice,uint64_t first,uint64_t span,
    uint64_t in_bytes,uint64_t rs_route,uint64_t ag_doorbell_bytes,uint64_t ag_route,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,uint32_t last,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint4 staged[16u][128u];
    __shared__ uint64_t s_started,s_rs_seq,s_rs_tag,s_rs_slot,s_ag_seq,s_ag_tag,s_ag_slot,s_prev_rseq;
    __shared__ uint64_t s_half_elems,s_in_tail_off,s_ag_tail_off,s_arr_end,s_comb_acc,s_copy_delta,s_rs_arr_base,s_rs_arr_end;
    __shared__ uint64_t s_rs_landed[SPARK_TP_MESH_ROUND_CONTROL_PEERS],s_ag_landed[SPARK_TP_MESH_ROUND_CONTROL_PEERS];
    __shared__ volatile uint64_t s_wait_start;
    __shared__ volatile uint32_t s_wait_over;
    __shared__ uint32_t s_failed,s_cap,s_compact,s_ag_compact,s_newly,s_ag_reval;
    __shared__ SparkTpMeshHybridGrant s_hyb;
    __shared__ __align__(16) SparkWeightdMeshHybridPeer s_peers[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    const uint32_t tid = threadIdx.x,warp_id = tid >> 5u,lane = tid & 31u;
    LmDependentWait();
    if ( tid < 32u )
    {
        uint32_t failed = 0u,compact = 0u,ag_compact = 0u,use_half = 0u,try_inl_first = 0u;
        uint64_t rs_seq = 0u,rs_slot = 0u,prev_rseq = 0u;
        if ( tid == 0u )
        {
            s_started = SparkTpFastTimerNs();
            const uint64_t prev_seq = SparkTpLdcvU64(&control->seq),epoch = SparkTpLdcvU64(&control->epoch);
            prev_rseq = SparkTpLdcvU64(&control->round_seq);
            failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 2u ? 1u : 0u;
            const SparkTpMeshHybridGpuCache *hcache = &SparkTpMeshHybridDevCache[hybrid_band & (SPARK_WEIGHTD_MESH_BANDS - 1u)];
            const uint32_t cache_ok = hybrid_ctrl != 0 && hybrid_shm != 0 && slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES &&
                hcache->ctrl == hybrid_ctrl && hcache->gen != 0u ? 1u : 0u;
            const uint32_t cap = cache_ok != 0u && prev_rseq != 0u && hcache->last_hyb_seq == prev_rseq ? 1u : SparkTpMeshFastCompactCap(gate);
            rs_seq = prev_seq + 1u;
            const uint64_t ag_seq = prev_seq + 2u;
            rs_slot = (uint64_t)rank * slots_per_rank + ((rs_seq - 1u) & (slots_per_rank - 1u));
            const uint64_t ag_slot = (uint64_t)rank * slots_per_rank + ((ag_seq - 1u) & (slots_per_rank - 1u));
            compact = cap != 0u && in_bytes != 0u && (in_bytes & 7u) == 0u &&
                in_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES && in_bytes <= slot_bytes - 16u ? 1u : 0u;
            use_half = compact != 0u && in_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ? 1u : 0u;
            ag_compact = cap != 0u && ag_doorbell_bytes != 0u && (ag_doorbell_bytes & 7u) == 0u &&
                ag_doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
            s_rs_seq = rs_seq; s_rs_tag = (epoch << 32u) | rs_seq; s_rs_slot = rs_slot; s_prev_rseq = prev_rseq;
            s_ag_seq = ag_seq; s_ag_tag = (epoch << 32u) | ag_seq; s_ag_slot = ag_slot;
            s_cap = cap; s_compact = compact; s_ag_compact = ag_compact;
            s_half_elems = use_half != 0u ? (SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(rs_slot) / sizeof(uint16_t)) : 0u;
            s_in_tail_off = compact != 0u ? in_bytes : slot_bytes - 8u;
            s_ag_tail_off = ag_compact != 0u ? ag_doorbell_bytes : slot_bytes - 8u;
            const uint32_t max_inl = cache_ok != 0u ? hcache->min_max_inline : (hybrid_ctrl != 0 ? hybrid_ctrl->min_max_inline : 0u);
            try_inl_first = failed == 0u && compact != 0u && hybrid_ctrl != 0 && hybrid_shm != 0 &&
                slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES && (in_bytes + 8u) <= max_inl ? 1u : 0u;
            if ( failed == 0u && try_inl_first == 0u )
            {
                SparkTpMeshFastShipWait(gate,control,timeout_ns,SparkTpMeshFastStagingAllowPrior(control,use_half));
                if ( SparkTpLdcvU64(&control->error_word) != 0u ) failed = 1u;
            }
            s_hyb.ok = 0u; s_hyb.ag_ok = 0u; s_failed = failed;
        }
        failed = __shfl_sync(0xffffffffu,failed,0);
        compact = __shfl_sync(0xffffffffu,compact,0);
        ag_compact = __shfl_sync(0xffffffffu,ag_compact,0);
        try_inl_first = __shfl_sync(0xffffffffu,try_inl_first,0);
        if ( failed == 0u && compact != 0u && hybrid_ctrl != 0 && hybrid_shm != 0 &&
             slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES )
        {
            SparkWeightdMeshRoute r,ar;
            r.word = rs_route; ar.word = ag_route;
            const uint32_t ag_inl_ok = (span >> 3u) <= 128u && ag_doorbell_bytes == ((span >> 3u) << 4u) ? 1u : 0u;
            const uint32_t ag_pre = ag_compact != 0u && (uint32_t)ar.fields.peer_mask == (uint32_t)r.fields.peer_mask ?
                (uint32_t)(ag_doorbell_bytes + 8u) : 0u;
            SparkTpMeshHybridAcquireWarp0(control,hybrid_ctrl,hybrid_band,degree,rank,(uint32_t)r.fields.peer_mask,
                (uint32_t)(in_bytes + 8u),1u,try_inl_first == 0u ? 1u : 0u,0u,s_hyb,ag_pre,ag_inl_ok,1u,prev_rseq);
            if ( tid == 0u && try_inl_first != 0u && s_hyb.ok == 0u )
            {
                SparkTpMeshFastShipWait(gate,control,timeout_ns,SparkTpMeshFastStagingAllowPrior(control,use_half));
                if ( SparkTpLdcvU64(&control->error_word) != 0u ) s_failed = 1u;
            }
        }
    }
    else if ( hybrid_ctrl != 0 && hybrid_shm != 0 )
    {
        const uint32_t pi = warp_id - 1u;
        const SparkTpMeshHybridGpuCache *hcache = &SparkTpMeshHybridDevCache[hybrid_band & (SPARK_WEIGHTD_MESH_BANDS - 1u)];
        if ( pi < SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS - 1u && lane < 4u )
            ((uint4 *)&s_peers[pi])[lane] = ((const uint4 *)(hcache->ctrl == hybrid_ctrl && hcache->gen != 0u ? &hcache->peers[pi] : &hybrid_ctrl->peers[pi]))[lane];
        if ( (slice >> 3u) <= 128u && (slice & 7u) == 0u && count == (uint64_t)degree * slice &&
             (((uintptr_t)(local + begin) & 15u) == 0u) && pi + 1u < degree )
        {
            const uint32_t p = pi >= rank ? pi + 1u : pi,sv = (uint32_t)(slice >> 3u);
            const uint4 *src4 = (const uint4 *)(local + begin + (uint64_t)p * slice);
            for ( uint32_t v = lane; v < sv; v += 32u ) staged[p][v] = src4[v];
        }
    }
    __syncthreads();
    if ( s_failed != 0u )
    {
        if ( tid == 0u )
        {
            if ( SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
            control->wait_started_ns = SparkTpFastTimerNs();
            __threadfence();
            LmDependentRelease();
        }
        return;
    }
    const uint64_t rs_seq = s_rs_seq,rs_tag = s_rs_tag,rs_slot = s_rs_slot;
    const uint64_t ag_seq = s_ag_seq,ag_tag = s_ag_tag,ag_slot = s_ag_slot;
    const uint32_t cap = s_cap,compact = s_compact;
    const uint64_t rs_ring = (rs_seq - 1u) & (slots_per_rank - 1u);
    const uint64_t ag_ring = (ag_seq - 1u) & (slots_per_rank - 1u);
    const uint16_t *rs_smem = (slice >> 3u) <= 128u && (slice & 7u) == 0u && count == (uint64_t)degree * slice &&
        (((uintptr_t)(local + begin) & 15u) == 0u) ? (const uint16_t *)&staged[0u][0u] : 0;
    SparkTpMeshFastRsagPublishRs(band,slot_bytes,entry,control,staging,staging_slot_elements,
        local + begin,rs_smem,degree,rank,count,slice,in_bytes,rs_route,
        rs_seq,rs_tag,rs_slot,s_half_elems,s_in_tail_off,cap,compact,s_started,s_prev_rseq,
        hybrid_ctrl,hybrid_shm,hybrid_band,s_peers,s_hyb,s_wait_start,s_arr_end,s_copy_delta);
    const uint32_t rs_was_hybrid = s_hyb.ok;
    const uint64_t stride = slots_per_rank * slot_bytes;
    const uint8_t *rs_base = band + rs_ring * slot_bytes;
    const uint32_t vecs = (uint32_t)(span >> 3u),lim = vecs < 128u ? vecs : 128u;
    const uint16_t *own_local = local + begin + first;
    uint16_t *own_out = output + begin + first;
    volatile uint16_t *ag_dst = (volatile uint16_t *)(band + ag_slot * slot_bytes);
    if ( tid < 32u ) __syncwarp();
    const volatile uint64_t *cancel = tid < 32u ? (const volatile uint64_t *)(uintptr_t)control->cancel_cell : 0;
    const uint64_t expected = tid < 32u ? control->cancel_expected : 0u;
    const volatile uint64_t *rs_tail = tid < 32u ?
        (const volatile uint64_t *)(band + ((uint64_t)tid * slots_per_rank + rs_ring) * slot_bytes + s_in_tail_off) : 0;
    uint32_t missing = tid < degree && tid != rank ? 1u : 0u;
    uint32_t remaining_mask = ((1u << degree) - 1u) & ~(1u << rank);
    uint64_t rs_arr_base = s_wait_start,rs_arr_landed = UINT64_MAX;
    uint32_t pf_resume = resume,rs_own_done = 0u;
    const uint32_t aux_warp = rank == 0u ? 1u : rank;
    while ( remaining_mask != 0u )
    {
        if ( prefetch != 0 )
        {
            if ( tid == 0u ) s_wait_over = 0u;
            __syncthreads();
        }
        if ( tid < 32u )
        {
            uint32_t fault = 0u;
            uint32_t newly = SparkTpMeshFastWarpPollBatch(rs_tail,rs_tag,missing,cancel,expected,s_wait_start,timeout_ns,control,rs_arr_landed,fault);
            if ( tid == 0u )
            {
                s_newly = newly;
                if ( fault != 0u ) s_failed = 1u;
                else if ( (remaining_mask & ~newly) == 0u ) s_arr_end = SparkTpFastTimerNs();
                *(volatile uint32_t *)&s_wait_over = 1u;
            }
        }
        else
        {
            if ( warp_id == aux_warp && rs_own_done == 0u )
            {
                const uint4 *own4 = (const uint4 *)own_local;
                for ( uint32_t v = lane; v < lim; v += 32u ) staged[rank][v] = own4[v];
                rs_own_done = 1u;
            }
            if ( prefetch != 0 )
                SparkTpMeshFastRunPrefetch(prefetch,pf_resume,tid,&s_wait_over);
        }
        pf_resume = 1u;
        __syncthreads();
        if ( s_failed != 0u ) break;
        const uint32_t mask = s_newly;
        remaining_mask &= ~mask;
        if ( (mask & (1u << warp_id)) != 0u )
        {
            const uint8_t *src = rs_base + (uint64_t)warp_id * stride;
            if ( ((((uintptr_t)src | ((uint64_t)lim << 4u)) & 31u) == 0u) )
            {
                const uint32_t octs = lim >> 1u;
                for ( uint32_t v = lane; v < octs; v += 32u )
                {
                    const SparkTpU256 w = SparkTpLdCvU256(src + (uint64_t)v * 32u);
                    staged[warp_id][v * 2u] = w.lo; staged[warp_id][v * 2u + 1u] = w.hi;
                }
            }
            else
                for ( uint32_t v = lane; v < lim; v += 32u )
                    staged[warp_id][v] = SparkTpLdCvU128(src + (uint64_t)v * 16u);
        }
        if ( remaining_mask == 0u && vecs <= 128u && warp_id == rank )
        {
            uint32_t reval = 0u;
            if ( s_hyb.ok != 0u && s_hyb.ag_ok != 0u )
            {
                SparkWeightdMeshRoute agr;
                agr.word = ag_route;
                reval = SparkTpMeshHybridRevalidateWarp(hybrid_ctrl,(uint32_t)agr.fields.peer_mask,s_peers,s_hyb);
            }
            if ( lane == 0u ) s_ag_reval = reval;
        }
        if ( remaining_mask != 0u ) __syncthreads();
    }
    if ( s_failed != 0u )
    {
        if ( tid == 0u ) { __threadfence(); LmDependentRelease(); }
        return;
    }
    const uint64_t rs_arr_end_saved = s_arr_end;
    if ( tid < SPARK_TP_MESH_ROUND_CONTROL_PEERS ) s_rs_landed[tid] = rs_arr_landed;
    if ( tid == 0u ) { s_rs_arr_base = rs_arr_base; s_rs_arr_end = rs_arr_end_saved; }
    __syncthreads();
    if ( vecs <= 128u )
    {
        if ( tid < vecs )
        {
            float sum[8] = {0.0f};
            for ( uint32_t p = 0u; p < degree; ++p )
                SparkTpAccumU128Bf16(sum,staged[p][tid]);
            const uint4 packed = SparkTpPackU128Bf16(sum);
            ((uint4 *)own_out)[tid] = packed;
            staged[0u][tid] = packed;
        }
        __syncthreads();
    }
    else
    {
        if ( tid < 128u )
        {
            float sum[8] = {0.0f};
            for ( uint32_t p = 0u; p < degree; ++p )
                SparkTpAccumU128Bf16(sum,staged[p][tid]);
            const uint4 packed = SparkTpPackU128Bf16(sum);
            ((uint4 *)own_out)[tid] = packed;
            SparkTpStVolatileU128(ag_dst + (uint64_t)tid * 8u,packed);
        }
        for ( uint32_t q = 128u + tid; q < vecs; q += blockDim.x )
        {
            float tsum[8] = {0.0f};
            for ( uint32_t p = 0u; p < degree; p += 4u )
            {
                uint4 v[4];
                const uint32_t n = degree - p < 4u ? degree - p : 4u;
#pragma unroll
                for ( uint32_t j = 0u; j < 4u; ++j )
                    if ( j < n )
                        v[j] = (p + j) == rank ? ((const uint4 *)own_local)[q] :
                            SparkTpLdCvU128(rs_base + (uint64_t)(p + j) * stride + (uint64_t)q * 16u);
#pragma unroll
                for ( uint32_t j = 0u; j < 4u; ++j )
                    if ( j < n )
                        SparkTpAccumU128Bf16(tsum,v[j]);
            }
            const uint4 packed = SparkTpPackU128Bf16(tsum);
            ((uint4 *)own_out)[q] = packed;
            SparkTpStVolatileU128(ag_dst + (uint64_t)q * 8u,packed);
        }
    }
    SparkTpMeshFastRsagPublishAg(band,slot_bytes,entry,gate,control,degree,rank,vecs,&staged[0u][0u],
        rs_tag,ag_seq,ag_tag,ag_slot,ag_doorbell_bytes,ag_route,s_ag_tail_off,
        cap,s_ag_compact,timeout_ns,rs_was_hybrid,vecs <= 128u ? s_ag_reval : 0u,
        hybrid_ctrl,hybrid_shm,hybrid_band,s_peers,s_hyb,s_failed,s_wait_start,s_arr_end,s_comb_acc);
    if ( tid < 32u ) __syncwarp();
    if ( s_failed != 0u )
    {
        if ( tid == 0u ) { __threadfence(); LmDependentRelease(); }
        return;
    }
    const volatile uint64_t *ag_tail = 0;
    if ( tid < 32u )
    {
        const uint64_t p_start = (uint64_t)tid * slice < count ? (uint64_t)tid * slice : count;
        const uint64_t p_len = count - p_start < slice ? count - p_start : slice;
        const uint64_t p_gb = (p_len * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
        const uint64_t p_db = p_gb != 0u ? p_gb : 8u;
        const uint64_t p_tail_off = cap != 0u && p_db != 0u && (p_db & 7u) == 0u && p_db <= slot_bytes - 16u ? p_db : slot_bytes - 8u;
        ag_tail = (const volatile uint64_t *)(band + ((uint64_t)tid * slots_per_rank + ag_ring) * slot_bytes + p_tail_off);
    }
    missing = tid < degree && tid != rank ? 1u : 0u;
    remaining_mask = ((1u << degree) - 1u) & ~(1u << rank);
    uint64_t ag_arr_landed = UINT64_MAX;
    uint32_t rs_rec_done = 0u;
    while ( remaining_mask != 0u )
    {
        if ( prefetch != 0 )
        {
            if ( tid == 0u ) s_wait_over = 0u;
            __syncthreads();
        }
        if ( tid < 32u )
        {
            uint32_t fault = 0u;
            uint32_t newly = SparkTpMeshFastWarpPollBatch(ag_tail,ag_tag,missing,cancel,expected,s_wait_start,timeout_ns,control,ag_arr_landed,fault);
            if ( tid == 0u )
            {
                s_newly = newly;
                if ( fault != 0u ) s_failed = 1u;
                else if ( (remaining_mask & ~newly) == 0u ) s_arr_end = SparkTpFastTimerNs();
                *(volatile uint32_t *)&s_wait_over = 1u;
            }
            if ( (remaining_mask & ~newly) == 0u && tid < SPARK_TP_MESH_ROUND_CONTROL_PEERS )
                s_ag_landed[tid] = ag_arr_landed;
        }
        else
        {
            if ( warp_id == 1u && rs_rec_done == 0u )
            {
                SparkTpMeshArrivalRecord(control,lane,s_rs_arr_base,s_rs_arr_end,
                    lane < SPARK_TP_MESH_ROUND_CONTROL_PEERS ? s_rs_landed[lane] : UINT64_MAX);
                rs_rec_done = 1u;
            }
            if ( prefetch != 0 )
                SparkTpMeshFastRunPrefetch(prefetch,pf_resume,tid,&s_wait_over);
        }
        pf_resume = 1u;
        __syncthreads();
        if ( s_failed != 0u ) break;
        const uint32_t mask = s_newly;
        remaining_mask &= ~mask;
        const uint32_t arrived_n = __popc(mask);
        if ( arrived_n == 1u && slice > 512u )
        {
            if ( warp_id != rank )
            {
                const uint32_t p = (uint32_t)(__ffs(mask) - 1);
                const uint64_t p_start = (uint64_t)p * slice < count ? (uint64_t)p * slice : count;
                const uint64_t p_len = count - p_start < slice ? count - p_start : slice;
                if ( p_len != 0u )
                    SparkTpMeshCopyHalvesFromBand(output + begin + p_start,
                        (const volatile uint16_t *)(band + ((uint64_t)p * slots_per_rank + ag_ring) * slot_bytes),
                        p_len,((warp_id > rank ? warp_id - 1u : warp_id) << 5u) + lane,blockDim.x - 32u);
            }
        }
        else if ( (mask & (1u << warp_id)) != 0u )
        {
            const uint64_t p_start = (uint64_t)warp_id * slice < count ? (uint64_t)warp_id * slice : count;
            const uint64_t p_len = count - p_start < slice ? count - p_start : slice;
            if ( p_len != 0u )
                SparkTpMeshCopyHalvesFromBand(output + begin + p_start,
                    (const volatile uint16_t *)(band + ((uint64_t)warp_id * slots_per_rank + ag_ring) * slot_bytes),
                    p_len,lane,32u);
        }
        if ( remaining_mask == 0u && warp_id == rank )
        {
            SparkTpMeshArrivalRecord(control,lane,s_wait_start,s_arr_end,
                lane < SPARK_TP_MESH_ROUND_CONTROL_PEERS ? s_ag_landed[lane] : UINT64_MAX);
            if ( lane == 0u )
            {
                control->copy_ns += s_copy_delta;
                if ( last != 0u ) control->rounds_done++;
            }
        }
        if ( remaining_mask != 0u ) __syncthreads();
    }
    __syncthreads();
    if ( tid == 0u )
    {
        const uint64_t end_ns = SparkTpFastTimerNs();
        __threadfence();
        LmDependentRelease();
        if ( s_failed == 0u )
        {
            const uint64_t ag_comb = s_arr_end != 0u && end_ns > s_arr_end ? end_ns - s_arr_end : 0u;
            if ( s_comb_acc + ag_comb != 0u ) control->combine_ns += s_comb_acc + ag_comb;
        }
    }
}

static __device__ __forceinline__ void SparkTpMeshFastGatherPublish(
    uint8_t *band,uint64_t slot_bytes,volatile uint64_t *entry,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    uint32_t rank,uint32_t degree,const uint16_t *local,uint64_t local_elements,uint64_t bytes,
    uint64_t seq,uint64_t tag,uint64_t slot,uint64_t route,uint64_t tail_off,uint64_t half_elems,
    uint64_t prev_rseq,uint64_t started,uint32_t cap,uint32_t compact,uint32_t use_half,
    uint32_t pmask,uint32_t fused_op,SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,
    uint32_t hybrid_band,SparkWeightdMeshHybridPeer *s_peers,const SparkTpMeshHybridGrant &s_hyb,
    uint64_t &s_copy_delta,volatile uint64_t &s_wait_start,uint64_t &s_arr_end)
{
    const uint32_t tid = threadIdx.x,warp_id = tid >> 5u,lane = tid & 31u,warps = blockDim.x >> 5u;
    if ( s_hyb.ok != 0u && s_hyb.fast_cred == 0u )
    {
        for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
            if ( lane < 4u ) ((uint4 *)&s_peers[i])[lane] = ((const uint4 *)&hybrid_ctrl->peers[i])[lane];
        __syncwarp();
    }
    const uint64_t slot_base = ((uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + slot) * slot_bytes;
    if ( s_hyb.ok != 0u && s_hyb.is_inline != 0u )
    {
        const uint64_t pelems = fused_op == 3u ? (local_elements * 4u) : local_elements;
        for ( uint32_t i = warp_id; i < s_hyb.peer_count; i += warps )
            if ( (pmask & (1u << s_peers[i].target)) != 0u )
                SparkTpMeshHybridWriteInlineWqe(hybrid_shm,&s_peers[i],s_hyb.pi[i],s_hyb.ds,slot_base,
                    fused_op == 2u ? (local + (uint64_t)s_peers[i].target * local_elements) : local,pelems,bytes,tag,lane);
    }
    else
    {
        if ( fused_op == 2u )
            SparkTpMeshFastPeerStageAll(band,slot_bytes,slot,staging,staging_slot_elements,half_elems,
                local,degree,rank,(uint64_t)degree * local_elements,local_elements,local_elements,
                bytes,tag,0u,0u,compact,tid,warp_id,lane,warps);
        else if ( bytes != 0u )
            SparkTpMeshCopyBytesVolatile(band + slot * slot_bytes,local,bytes,tid,blockDim.x);
        if ( tid == 0u )
        {
            *(volatile uint64_t *)(band + slot * slot_bytes + tail_off) = tag;
            if ( s_hyb.ok == 0u ) { volatile uint64_t *c = cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(seq)) : entry; c[1] = bytes; c[2] = slot; c[3] = route; }
        }
        if ( s_hyb.ok != 0u && lane == 0u )
            SparkTpMeshHybridWriteDmaWarps(s_peers,hybrid_shm,s_hyb,pmask,fused_op == 2u ? 1u : 0u,slot_base,
                fused_op == 2u ? (hybrid_ctrl->staging_iova_base + (uint64_t)hybrid_band * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES + half_elems * sizeof(uint16_t)) : (hybrid_ctrl->recv_iova_base + slot_base),
                (uint32_t)(bytes + 8u),warp_id,warps);
    }
    __syncthreads();
    if ( tid < 32u ) { if ( s_hyb.ok != 0u ) SparkTpMeshHybridRingWarp0(hybrid_ctrl,hybrid_shm,s_peers,s_hyb,pmask,seq,tag,slot,bytes); else if ( tid == 0u ) __threadfence_system(); }
    if ( tid == 0u )
    {
        const uint64_t stg_flag = fused_op == 2u && use_half == 0u && !(s_hyb.ok != 0u && s_hyb.is_inline != 0u) ?
            SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG : 0u;
        control->round_seq_prior = prev_rseq | (s_hyb.ok != 0u ? 0u : SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG) | stg_flag;
        control->round_seq = tag; control->seq = seq;
        if ( s_hyb.ok == 0u ) SparkTpStReleaseSysU64(&(cap != 0u ? (entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(seq)) : entry)[0],tag);
        const uint64_t now = SparkTpFastTimerNs();
        s_copy_delta = now > started ? now - started : 0u; control->wait_started_ns = now; s_wait_start = now; s_arr_end = 0u;
    }
}

static __global__ __launch_bounds__(SPARK_TP_MESH_FAST_FUSED_THREADS,1u) void SparkTpMeshFastGatherFusedKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    uint32_t rank,uint32_t degree,const uint16_t *local,uint16_t *output,uint64_t local_elements,
    uint64_t route,uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,uint32_t fused_op,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint64_t s_started,s_seq,s_tag,s_slot,s_route,s_tail_off,s_half_elems,s_arr_end,s_copy_delta,s_prev_rseq;
    __shared__ uint64_t s_landed[SPARK_TP_MESH_ROUND_CONTROL_PEERS];
    __shared__ volatile uint64_t s_wait_start;
    __shared__ volatile uint32_t s_wait_over;
    __shared__ uint32_t s_failed,s_cap,s_compact,s_use_half,s_peer_mask,s_newly;
    __shared__ SparkTpMeshHybridGrant s_hyb;
    __shared__ __align__(16) SparkWeightdMeshHybridPeer s_peers[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    const uint32_t tid = threadIdx.x,warp_id = tid >> 5u,lane = tid & 31u;
    const uint64_t raw_bytes = fused_op == 3u ? (local_elements * sizeof(uint64_t)) : (local_elements * sizeof(uint16_t));
    const uint64_t bytes = fused_op == 2u ? ((raw_bytes + 7u) & ~UINT64_C(7)) : raw_bytes;
    LmDependentWait();
    uint64_t own_max[4u] = {0u,0u,0u,0u};
    if ( fused_op == 3u && lane == rank )
        for ( uint32_t k = 0u; k < 4u; ++k )
            if ( warp_id + k * 16u < local_elements ) own_max[k] = ((const uint64_t *)local)[warp_id + k * 16u];
    if ( tid < 32u )
    {
        uint32_t failed = 0u,compact = 0u,try_inl = 0u,pmask = 0u,allow_prior = 0u;
        uint64_t prev_rseq = 0u;
        if ( tid == 0u )
        {
            s_started = SparkTpFastTimerNs();
            const uint64_t prev_seq = SparkTpLdcvU64(&control->seq),epoch = SparkTpLdcvU64(&control->epoch);
            prev_rseq = SparkTpLdcvU64(&control->round_seq);
            failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 1u ? 1u : 0u;
            const SparkTpMeshHybridGpuCache *hc = &SparkTpMeshHybridDevCache[hybrid_band & (SPARK_WEIGHTD_MESH_BANDS - 1u)];
            const uint32_t cache_ok = hybrid_ctrl != 0 && hybrid_shm != 0 && slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES && hc->ctrl == hybrid_ctrl && hc->gen != 0u ? 1u : 0u;
            const uint32_t cap = cache_ok != 0u && prev_rseq != 0u && hc->last_hyb_seq == prev_rseq ? 1u : SparkTpMeshFastCompactCap(gate);
            const uint64_t seq = prev_seq + 1u,slot = (uint64_t)rank * slots_per_rank + ((seq - 1u) & (slots_per_rank - 1u));
            SparkWeightdMeshRoute r; r.word = route; pmask = (uint32_t)r.fields.peer_mask;
            compact = cap != 0u && bytes != 0u && (bytes & 7u) == 0u && bytes <= slot_bytes - 16u &&
                (fused_op == 2u ? (bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES) :
                    (r.fields.pair == 0u && r.fields.half == SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE && r.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_FULL)) ? 1u : 0u;
            const uint32_t use_half = fused_op == 2u && compact != 0u && bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ? 1u : 0u;
            allow_prior = fused_op == 2u ? SparkTpMeshFastStagingAllowPrior(control,use_half) : cap;
            const uint32_t max_inl = cache_ok != 0u ? hc->min_max_inline : (hybrid_ctrl != 0 ? hybrid_ctrl->min_max_inline : 0u);
            try_inl = failed == 0u && compact != 0u && hybrid_ctrl != 0 && hybrid_shm != 0 && slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES && (bytes + 8u) <= max_inl ? 1u : 0u;
            s_seq = seq; s_tag = (epoch << 32u) | seq; s_slot = slot; s_prev_rseq = prev_rseq;
            s_route = SparkTpMeshFastRouteCompact(route,compact); s_tail_off = compact != 0u ? bytes : slot_bytes - 8u;
            s_half_elems = use_half != 0u ? (SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot) / sizeof(uint16_t)) : 0u;
            s_cap = cap; s_compact = compact; s_use_half = use_half; s_peer_mask = pmask;
            if ( failed == 0u && try_inl == 0u ) failed = SparkTpMeshFastEnsureShipped(gate,control,timeout_ns,allow_prior,prev_rseq);
            s_hyb.ok = 0u; s_failed = failed;
        }
        failed = __shfl_sync(0xffffffffu,failed,0); compact = __shfl_sync(0xffffffffu,compact,0);
        try_inl = __shfl_sync(0xffffffffu,try_inl,0); pmask = __shfl_sync(0xffffffffu,pmask,0); prev_rseq = __shfl_sync(0xffffffffu,prev_rseq,0);
        if ( failed == 0u && compact != 0u && hybrid_ctrl != 0 && hybrid_shm != 0 && slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES )
        {
            SparkTpMeshHybridAcquireWarp0(control,hybrid_ctrl,hybrid_band,degree,rank,pmask,(uint32_t)(bytes + 8u),1u,try_inl == 0u ? 1u : 0u,0u,s_hyb,0u,0u,0u,prev_rseq);
            if ( tid == 0u && try_inl != 0u && s_hyb.ok == 0u && SparkTpMeshFastEnsureShipped(gate,control,timeout_ns,allow_prior,prev_rseq) != 0u ) s_failed = 1u;
        }
    }
    else if ( hybrid_ctrl != 0 && hybrid_shm != 0 )
    {
        const uint32_t pi = warp_id - 1u;
        const SparkTpMeshHybridGpuCache *hc = &SparkTpMeshHybridDevCache[hybrid_band & (SPARK_WEIGHTD_MESH_BANDS - 1u)];
        if ( pi < SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS - 1u && lane < 4u )
            ((uint4 *)&s_peers[pi])[lane] = ((const uint4 *)(hc->ctrl == hybrid_ctrl && hc->gen != 0u ? &hc->peers[pi] : &hybrid_ctrl->peers[pi]))[lane];
    }
    __syncthreads();
    if ( s_failed != 0u )
    {
        if ( tid == 0u ) { if ( SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX; control->wait_started_ns = SparkTpFastTimerNs(); __threadfence(); LmDependentRelease(); }
        return;
    }
    const uint64_t seq = s_seq,tag = s_tag,slot = s_slot,ring = (seq - 1u) & (slots_per_rank - 1u);
    const uint32_t pmask = s_peer_mask;
    SparkTpMeshFastGatherPublish(band,slot_bytes,entry,control,staging,staging_slot_elements,rank,degree,
        local,local_elements,bytes,seq,tag,slot,s_route,s_tail_off,s_half_elems,s_prev_rseq,s_started,
        s_cap,s_compact,s_use_half,pmask,fused_op,hybrid_ctrl,hybrid_shm,hybrid_band,s_peers,s_hyb,
        s_copy_delta,s_wait_start,s_arr_end);
    if ( tid < 32u ) __syncwarp();
    const volatile uint64_t *cancel = tid < 32u ? (const volatile uint64_t *)(uintptr_t)control->cancel_cell : 0;
    const uint64_t expected = tid < 32u ? control->cancel_expected : 0u;
    const volatile uint64_t *tail = tid < 32u ? (const volatile uint64_t *)(band + ((uint64_t)tid * slots_per_rank + ring) * slot_bytes + s_tail_off) : 0;
    uint32_t missing = fused_op == 1u ? (tid == (rank ^ 1u) ? 1u : 0u) : (tid < degree && tid != rank ? 1u : 0u);
    uint32_t remaining_mask = pmask,pf_resume = resume,own_done = fused_op == 3u ? 1u : 0u;
    uint64_t arr_landed = UINT64_MAX;
    while ( remaining_mask != 0u )
    {
        if ( prefetch != 0 ) { if ( tid == 0u ) s_wait_over = 0u; __syncthreads(); }
        if ( tid < 32u )
        {
            uint32_t fault = 0u,newly = SparkTpMeshFastWarpPollBatch(tail,tag,missing,cancel,expected,s_wait_start,timeout_ns,control,arr_landed,fault);
            if ( tid == 0u ) { s_newly = newly; if ( fault != 0u ) s_failed = 1u; else if ( (remaining_mask & ~newly) == 0u ) s_arr_end = SparkTpFastTimerNs(); *(volatile uint32_t *)&s_wait_over = 1u; }
            if ( (remaining_mask & ~newly) == 0u && tid < SPARK_TP_MESH_ROUND_CONTROL_PEERS ) s_landed[tid] = arr_landed;
        }
        else
        {
            if ( own_done == 0u )
            {
                SparkTpMeshCopyHalves(output + (uint64_t)rank * local_elements,
                    fused_op == 2u ? (local + (uint64_t)rank * local_elements) : local,local_elements,tid - 32u,blockDim.x - 32u);
                own_done = 1u;
            }
            if ( prefetch != 0 ) SparkTpMeshFastRunPrefetch(prefetch,pf_resume,tid,&s_wait_over);
        }
        pf_resume = 1u;
        __syncthreads();
        if ( s_failed != 0u ) break;
        const uint32_t mask = s_newly; remaining_mask &= ~mask;
        if ( fused_op < 3u )
        {
            if ( __popc(mask) == 1u && local_elements > 512u )
            {
                if ( warp_id != rank )
                {
                    const uint32_t p = (uint32_t)(__ffs(mask) - 1);
                    SparkTpMeshCopyHalvesFromBand(output + (uint64_t)p * local_elements,(const volatile uint16_t *)(band + ((uint64_t)p * slots_per_rank + ring) * slot_bytes),
                        local_elements,((warp_id > rank ? warp_id - 1u : warp_id) << 5u) + lane,blockDim.x - 32u);
                }
            }
            else if ( (mask & (1u << warp_id)) != 0u )
                SparkTpMeshCopyHalvesFromBand(output + (uint64_t)warp_id * local_elements,(const volatile uint16_t *)(band + ((uint64_t)warp_id * slots_per_rank + ring) * slot_bytes),
                    local_elements,lane,32u);
        }
        if ( remaining_mask == 0u && warp_id == rank )
        {
            SparkTpMeshArrivalRecord(control,lane,s_wait_start,s_arr_end,lane < SPARK_TP_MESH_ROUND_CONTROL_PEERS ? s_landed[lane] : UINT64_MAX);
            if ( lane == 0u ) { control->copy_ns += s_copy_delta; control->slot_cursor = seq; control->rounds_done++; }
        }
        if ( remaining_mask != 0u ) __syncthreads();
    }
    if ( s_failed == 0u && fused_op == 3u )
        for ( uint32_t k = 0u; k < 4u; ++k )
        {
            const uint32_t e = warp_id + k * 16u;
            if ( e >= local_elements ) break;
            uint64_t v = lane < degree ? (lane == rank ? own_max[k] :
                SparkTpLdcvU64(band + ((uint64_t)lane * slots_per_rank + ring) * slot_bytes + (uint64_t)e * 8u)) : 0u;
            for ( uint32_t s = 16u; s != 0u; s >>= 1u ) { const uint64_t o = __shfl_xor_sync(0xffffffffu,v,s); if ( o > v ) v = o; }
            if ( lane == 0u ) ((uint64_t *)output)[e] = v;
        }
    __syncthreads();
    if ( tid == 0u )
    {
        const uint64_t end_ns = SparkTpFastTimerNs();
        __threadfence();
        LmDependentRelease();
        if ( s_failed == 0u && s_arr_end != 0u && end_ns > s_arr_end ) control->combine_ns += end_ns - s_arr_end;
    }
}

static __device__ __forceinline__ uint32_t SparkTpMeshFast2dGroupPeer(uint32_t rank,uint32_t is_col,uint32_t k)
{
    return is_col != 0u ? ((k << 2u) | (rank & 3u)) : ((rank & ~3u) | k);
}

static __device__ __forceinline__ uint32_t SparkTpMeshFast2dWarpWait(
    const volatile uint64_t *tail,uint64_t cur_tag,uint32_t missing,uint32_t p_lane,
    volatile uint64_t *mbox,uint64_t mbox_ok,uint64_t mbox_fail,uint32_t signal_each,
    SparkTpMeshRoundControl *control,uint64_t started,uint64_t timeout_ns,
    uint64_t &arr_landed,uint64_t &arr_end_out)
{
    __shared__ volatile uint64_t s_poll_started;
    const uint32_t tid = threadIdx.x;
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const uint64_t expected = control->cancel_expected;
    uint32_t spins = 0u;
    uint64_t seen = 0u,diag = 0u,fault = 0u;
    if ( tid == 0u ) s_poll_started = started;
    for (;;)
    {
        if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == cur_tag )
        {
            missing = 0u;
            if ( signal_each != 0u && mbox != 0 ) SparkTpStReleaseGpuU64(&mbox[p_lane],mbox_ok);
            arr_landed = SparkTpFastTimerNs();
        }
        if ( __all_sync(0xffffffffu,missing == 0u) ) break;
        if ( (++spins & 31u) == 0u )
        {
            if ( tid == 0u )
            {
                const uint64_t base = s_poll_started,now = SparkTpFastTimerNs();
                if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                    fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | cur_tag;
                else if ( now > base && now - base > timeout_ns )
                    fault = cur_tag;
            }
            fault = __shfl_sync(0xffffffffu,fault,0);
            if ( fault != 0u )
            {
                if ( fault == cur_tag )
                {
                    uint32_t first = (uint32_t)(__ffs(__ballot_sync(0xffffffffu,missing != 0u)) - 1);
                    diag = __shfl_sync(0xffffffffu,seen,first);
                }
                break;
            }
        }
    }
    if ( tid == 0u )
    {
        if ( fault != 0u )
        {
            if ( atomicCAS((unsigned long long *)&control->error_word,0ull,fault) == 0ull )
            {
                control->diag_word = diag;
                __threadfence();
            }
            if ( mbox != 0 ) SparkTpStReleaseGpuU64(&mbox[signal_each != 0u ? SPARK_TP_MESH_FAST_MBOX_FAIL : 0u],mbox_fail);
        }
        else
        {
            const uint64_t aend = SparkTpFastTimerNs();
            arr_end_out = aend;
            control->math_started_ns = aend;
            if ( signal_each == 0u && mbox != 0 ) SparkTpStReleaseGpuU64(&mbox[0],mbox_ok);
            else __threadfence();
        }
    }
    return fault == 0u ? 1u : 0u;
}

static __device__ __forceinline__ uint32_t SparkTpMeshFast2dMboxWaitGroup(
    volatile uint64_t *mbox,uint64_t mbox_ok,uint64_t mbox_fail,
    const volatile uint64_t *tail,uint64_t cur_tag,uint32_t missing,
    uint32_t need_ship,uint32_t allow_prior,SparkTpMeshRoundControl *control,uint64_t timeout_ns)
{
    __shared__ volatile uint64_t s_grp_start;
    const uint32_t tid = threadIdx.x;
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const volatile uint64_t *shipped = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    const uint64_t expected = control->cancel_expected;
    uint32_t spins = 0u;
    if ( tid == 0u ) s_grp_start = SparkTpFastTimerNs();
    for (;;)
    {
        uint32_t state = 0u;
        if ( tid == 0u )
        {
            const uint64_t v = SparkTpLdAcquireGpuU64(&mbox[0]);
            if ( v == mbox_ok ) state = 1u;
            else if ( v == mbox_fail ) state = 2u;
        }
        state = __shfl_sync(0xffffffffu,state,0);
        if ( state != 0u ) return state == 1u ? 1u : 0u;
        if ( (++spins & 15u) == 0u )
        {
            uint32_t fault = 0u;
            if ( tid == 0u && SparkTpLdcvU64(&control->error_word) != 0u ) fault = 1u;
            if ( __shfl_sync(0xffffffffu,fault,0) != 0u ) return 0u;
            if ( (spins & 63u) == 0u )
            {
                if ( missing != 0u && SparkTpLdAcquireU64(tail) == cur_tag )
                    missing = 0u;
                if ( __all_sync(0xffffffffu,missing == 0u) )
                {
                    uint32_t ship_ok = 1u;
                    if ( need_ship != 0u && tid == 0u )
                        ship_ok = shipped != 0 && SparkTpMeshFastShippedAccepts(control,SparkTpLdAcquireU64(shipped),allow_prior) != 0u ? 1u : 0u;
                    if ( __shfl_sync(0xffffffffu,ship_ok,0) != 0u ) return 1u;
                }
                if ( tid == 0u )
                {
                    if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected )
                    {
                        atomicCAS((unsigned long long *)&control->error_word,0ull,SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | cur_tag);
                        fault = 1u;
                    }
                    else
                    {
                        const uint64_t base = s_grp_start,now = SparkTpFastTimerNs();
                        if ( now > base && now - base > timeout_ns )
                        {
                            atomicCAS((unsigned long long *)&control->error_word,0ull,cur_tag);
                            fault = 1u;
                        }
                    }
                }
                if ( __shfl_sync(0xffffffffu,fault,0) != 0u ) return 0u;
            }
        }
        __nanosleep(50u);
    }
}

static __global__ __launch_bounds__(SPARK_TP_MESH_THREADS,2) void SparkTpMeshFast2dStageKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *local,volatile uint64_t *mbox,uint32_t rank,
    uint64_t count,uint64_t col_slice,uint64_t col_bytes,uint64_t row_route,uint64_t timeout_ns,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint64_t s_seq,s_tag,s_slot,s_half_elems;
    __shared__ uint32_t s_failed,s_last,s_cap,s_compact,s_use_half;
    const uint32_t col = rank & 3u,row = rank >> 2u;
    const uint32_t target_c = blockIdx.y < col ? blockIdx.y : blockIdx.y + 1u;
    const uint32_t peer = (row << 2u) | target_c;
    const uint32_t stripe = blockIdx.x,tid = threadIdx.x;
    LmDependentWait();
    if ( tid == 0u )
    {
        const uint64_t prev_seq = SparkTpLdcvU64(&control->seq),epoch = SparkTpLdcvU64(&control->epoch);
        uint32_t failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 4u ? 1u : 0u;
        const uint64_t seq = prev_seq + 1u,tag = (epoch << 32u) | seq;
        const uint64_t slot = (uint64_t)rank * slots_per_rank + ((seq - 1u) & (slots_per_rank - 1u));
        const uint64_t mbox_ok = tag ^ SPARK_TP_MESH_FAST_MBOX_STAGE_OK_SALT,mbox_fail = tag ^ SPARK_TP_MESH_FAST_MBOX_STAGE_FAIL_SALT;
        const uint32_t cap = SparkTpMeshFastCompactCap(gate);
        const uint32_t compact = cap != 0u && col_bytes != 0u && (col_bytes & 7u) == 0u &&
            col_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES && col_bytes <= slot_bytes - 16u ? 1u : 0u;
        const uint32_t use_half = compact != 0u && col_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ? 1u : 0u;
        const uint32_t allow_prior = SparkTpMeshFastStagingAllowPrior(control,use_half);
        s_seq = seq; s_tag = tag; s_slot = slot; s_cap = cap; s_compact = compact; s_use_half = use_half;
        s_half_elems = use_half != 0u ? (SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(slot) / sizeof(uint16_t)) : 0u;
        if ( failed == 0u )
        {
            if ( blockIdx.y == 0u && stripe == 0u )
            {
                control->publish_started_ns = SparkTpFastTimerNs() | 1ull;
                SparkTpMeshFastShipWait(gate,control,timeout_ns,allow_prior);
                if ( SparkTpLdcvU64(&control->error_word) != 0u ) failed = 1u;
                SparkTpStReleaseGpuU64(&mbox[0],failed == 0u ? mbox_ok : mbox_fail);
            }
            else
                failed = SparkTpMeshFastShipWaitFollower(control,mbox,mbox_ok,mbox_fail,timeout_ns,allow_prior);
        }
        s_failed = failed;
    }
    __syncthreads();
    if ( s_failed == 0u )
    {
        const uint64_t first = (uint64_t)target_c * col_slice < count ? (uint64_t)target_c * col_slice : count;
        const uint64_t length = count - first < col_slice ? count - first : col_slice;
        uint16_t *dst = staging + (uint64_t)peer * staging_slot_elements + s_half_elems;
        const uint64_t lane = (uint64_t)stripe * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
        if ( length != 0u )
            SparkTpMeshCopyHalves(dst,local + first,length,lane,lanes);
        if ( s_compact != 0u && lane == 0u )
            *(volatile uint64_t *)((volatile uint8_t *)dst + col_bytes) = s_tag;
        __threadfence_system();
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull ==
            (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || tid >= 32u ) return;
    uint32_t go = 0u,peer_mask = 0u;
    if ( tid == 0u )
    {
        control->publish_blocks_done = 0u;
        mbox[0] = 0u;
        if ( s_failed != 0u && SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
        if ( s_failed == 0u && SparkTpLdcvU64(&control->error_word) == 0u )
        {
            SparkWeightdMeshRoute r;
            r.word = row_route;
            peer_mask = (uint32_t)r.fields.peer_mask;
            *(volatile uint64_t *)(band + s_slot * slot_bytes + (s_compact != 0u ? col_bytes : slot_bytes - 8u)) = s_tag;
            __threadfence_system();
            go = 1u;
        }
    }
    go = __shfl_sync(0xffffffffu,go,0); peer_mask = __shfl_sync(0xffffffffu,peer_mask,0);
    if ( go != 0u )
        SparkTpMeshFastDmaOrDoorbellWarp0(entry,control,hybrid_ctrl,hybrid_shm,hybrid_band,
            16u,rank,peer_mask,1u,s_slot,slot_bytes,s_seq,s_tag,col_bytes,row_route,
            s_half_elems * sizeof(uint16_t),s_cap,s_compact,s_use_half == 0u ? SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG : 0u,0u);
    if ( tid != 0u ) return;
    if ( go != 0u )
    {
        const uint64_t pub_start = SparkTpLdcvU64(&control->publish_started_ns),now = SparkTpFastTimerNs();
        if ( pub_start != 0u && now > pub_start ) control->copy_ns += now - pub_start;
        control->wait_started_ns = now;
    }
    else
        control->wait_started_ns = SparkTpFastTimerNs();
    control->publish_started_ns = 0u;
    __threadfence();
    LmDependentRelease();
}

static __global__ __launch_bounds__(SPARK_TP_MESH_THREADS,2) void SparkTpMeshFast2dReducePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *own_src,uint16_t *output,volatile uint64_t *mbox,
    uint32_t rank,uint32_t is_col,uint64_t out_offset,uint64_t span,uint64_t sub_slice,
    uint64_t in_bytes,uint64_t next_doorbell_bytes,uint64_t next_route,uint64_t timeout_ns,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint32_t s_go;
    __shared__ uint32_t s_last;
    __shared__ uint64_t s_arr_end;
    const uint32_t tid = threadIdx.x,row = rank >> 2u,col = rank & 3u,self_k = is_col != 0u ? row : col;
    uint64_t arr_base = 0u,arr_landed = UINT64_MAX;
    uint32_t ok = 0u;
    LmDependentWait();
    const uint32_t entry_err = SparkTpLdcvU64(&control->error_word) != 0u ? 1u : 0u;
    const uint64_t cur_tag = SparkTpLdcvU64(&control->round_seq);
    const uint64_t cur_ring = (cur_tag - 1u) & (slots_per_rank - 1u);
    const uint64_t next_seq = SparkTpLdcvU64(&control->seq) + 1u;
    const uint64_t next_tag = (SparkTpLdcvU64(&control->epoch) << 32u) | next_seq;
    const uint64_t next_slot = (uint64_t)rank * slots_per_rank + ((next_seq - 1u) & (slots_per_rank - 1u));
    const uint32_t cap = SparkTpMeshFastCompactCap(gate);
    const uint64_t stage_ok = next_tag ^ SPARK_TP_MESH_FAST_MBOX_STAGE_OK_SALT;
    const uint64_t stage_fail = next_tag ^ SPARK_TP_MESH_FAST_MBOX_STAGE_FAIL_SALT;
    if ( entry_err == 0u )
    {
        if ( tid < 32u )
        {
            const uint32_t peer = tid < 4u ? SparkTpMeshFast2dGroupPeer(rank,is_col,tid) : rank;
            const uint32_t in_compact = cap != 0u && in_bytes != 0u && (in_bytes & 7u) == 0u &&
                in_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES && in_bytes <= slot_bytes - 16u ? 1u : 0u;
            const uint32_t next_stg_half = is_col == 0u && cap != 0u && next_doorbell_bytes != 0u && (next_doorbell_bytes & 7u) == 0u &&
                next_doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES && next_doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
            const uint32_t in_use_half = in_compact != 0u && in_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES && next_stg_half != 0u ?
                SparkTpMeshFastStagingAllowPrior(control,cap) : 0u;
            const volatile uint64_t *tail = (const volatile uint64_t *)(band +
                ((uint64_t)peer * slots_per_rank + cur_ring) * slot_bytes + (in_compact != 0u ? in_bytes : slot_bytes - 8u));
            const uint32_t missing = tid < 4u && tid != self_k ? 1u : 0u;
            if ( blockIdx.x == 0u )
            {
                uint64_t aend = 0u;
                arr_base = SparkTpLdcvU64(&control->wait_started_ns);
                ok = SparkTpMeshFast2dWarpWait(tail,cur_tag,missing,peer,
                    0,0u,0u,0u,control,SparkTpFastTimerNs(),timeout_ns,arr_landed,aend);
                if ( tid == 0u )
                {
                    uint32_t go = ok;
                    if ( is_col == 0u && go != 0u )
                    {
                        SparkTpMeshFastShipWait(gate,control,timeout_ns,in_use_half);
                        if ( SparkTpLdcvU64(&control->error_word) != 0u ) go = 0u;
                    }
                    SparkTpStReleaseGpuU64(&mbox[0],go != 0u ? stage_ok : stage_fail);
                    s_arr_end = aend;
                    s_go = go;
                }
            }
            else
            {
                uint32_t go = SparkTpMeshFast2dMboxWaitGroup(mbox,stage_ok,stage_fail,tail,cur_tag,missing,
                    is_col == 0u ? 1u : 0u,in_use_half,control,timeout_ns);
                if ( tid == 0u ) s_go = go;
            }
        }
        __syncthreads();
    }
    else
    {
        if ( tid == 0u ) s_go = 0u;
        __syncthreads();
    }
    const uint32_t next_compact = cap != 0u && next_doorbell_bytes != 0u && (next_doorbell_bytes & 7u) == 0u &&
        (is_col != 0u || next_doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES) &&
        next_doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
    const uint64_t next_half_elems = is_col == 0u && next_compact != 0u &&
        next_doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES ?
        (SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(next_slot) / sizeof(uint16_t)) : 0u;
    if ( s_go != 0u )
    {
        const uint64_t stride = slots_per_rank * slot_bytes;
        const uint8_t *base = band + cur_ring * slot_bytes;
        const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
        const uint32_t p0 = SparkTpMeshFast2dGroupPeer(rank,is_col,0u);
        const uint32_t p1 = SparkTpMeshFast2dGroupPeer(rank,is_col,1u);
        const uint32_t p2 = SparkTpMeshFast2dGroupPeer(rank,is_col,2u);
        const uint32_t p3 = SparkTpMeshFast2dGroupPeer(rank,is_col,3u);
        const uint32_t peers4[4] = { p0, p1, p2, p3 };
        const uint32_t num_subs = is_col != 0u ? 1u : 4u;
        volatile uint16_t *band_dst = is_col != 0u ? (volatile uint16_t *)(band + next_slot * slot_bytes) : 0;
        const uint32_t sub_aligned32 = (sub_slice & 15u) == 0u && (slot_bytes & 31u) == 0u &&
            (((uintptr_t)band | (uintptr_t)own_src | (uintptr_t)output | (uintptr_t)staging) & 31u) == 0u ? 1u : 0u;
        for ( uint32_t r = 0u; r < num_subs; ++r )
        {
            const uint64_t sub_off = (uint64_t)r * sub_slice;
            const uint64_t r_len = span > sub_off ? (span - sub_off < sub_slice ? span - sub_off : sub_slice) : 0u;
            uint16_t *dst = is_col != 0u ? (output + out_offset) :
                (r == row ? (output + out_offset + sub_off) :
                    (staging + (uint64_t)((r << 2u) | col) * staging_slot_elements + next_half_elems));
            uint64_t done = 0u;
            if ( sub_aligned32 != 0u && (((uintptr_t)dst & 31u) == 0u) )
            {
                const uint64_t octs = r_len >> 4u;
                for ( uint64_t q = lane; q < octs; q += lanes )
                {
                    float sum[16] = {0.0f};
                    SparkTpU256 v[4];
#pragma unroll
                    for ( uint32_t k = 0u; k < 4u; ++k )
                        v[k] = k == self_k ? SparkTpLdNcU256(own_src + sub_off + q * 16u) :
                            SparkTpLdCvU256(base + (uint64_t)peers4[k] * stride + (sub_off + q * 16u) * sizeof(uint16_t));
#pragma unroll
                    for ( uint32_t k = 0u; k < 4u; ++k )
                        SparkTpAccumU256Bf16(sum,v[k]);
                    SparkTpU256 packed = SparkTpPackU256Bf16(sum);
                    SparkTpStU256(dst + q * 16u,packed);
                    if ( band_dst != 0 ) SparkTpStVolatileU256(band_dst + q * 16u,packed);
                }
                done = octs << 4u;
            }
            const uint64_t quads = (r_len - done) >> 3u;
            for ( uint64_t q = lane; q < quads; q += lanes )
            {
                float sum[8] = {0.0f};
                uint4 v[4];
#pragma unroll
                for ( uint32_t k = 0u; k < 4u; ++k )
                    v[k] = k == self_k ? ((const uint4 *)(own_src + sub_off + done))[q] :
                        SparkTpLdCvU128(base + (uint64_t)peers4[k] * stride + (sub_off + done + q * 8u) * sizeof(uint16_t));
#pragma unroll
                for ( uint32_t k = 0u; k < 4u; ++k )
                    SparkTpAccumU128Bf16(sum,v[k]);
                uint4 packed = SparkTpPackU128Bf16(sum);
                ((uint4 *)(dst + done))[q] = packed;
                if ( band_dst != 0 ) SparkTpStVolatileU128(band_dst + done + q * 8u,packed);
            }
            for ( uint64_t i = done + (quads << 3u) + lane; i < r_len; i += lanes )
            {
                float acc = 0.0f;
#pragma unroll
                for ( uint32_t k = 0u; k < 4u; ++k )
                {
                    uint16_t h = k == self_k ? own_src[sub_off + i] :
                        SparkTpLdCvU16(base + (uint64_t)peers4[k] * stride + (sub_off + i) * sizeof(uint16_t));
                    acc += __uint_as_float((uint32_t)h << 16u);
                }
                uint16_t h = SparkTpFloatToBf16(acc);
                dst[i] = h;
                if ( band_dst != 0 ) band_dst[i] = h;
            }
            if ( is_col == 0u && r != row && next_compact != 0u && lane == 0u )
                *(volatile uint64_t *)((volatile uint8_t *)dst + next_doorbell_bytes) = next_tag;
        }
        __threadfence_system();
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull ==
            (unsigned long long)gridDim.x ? 1u : 0u;
    __syncthreads();
    if ( s_last != 0u && tid < 32u )
    {
        uint32_t go = 0u,peer_mask = 0u;
        if ( tid == 0u )
        {
            const uint64_t end_ns = SparkTpFastTimerNs();
            const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
            control->publish_blocks_done = 0u;
            control->math_started_ns = 0u;
            mbox[0] = 0u;
            if ( s_go != 0u && SparkTpLdcvU64(&control->error_word) == 0u )
            {
                if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
                *(volatile uint64_t *)(band + next_slot * slot_bytes + (next_compact != 0u ? next_doorbell_bytes : slot_bytes - 8u)) = next_tag;
                if ( is_col != 0u ) SparkTpMeshFastShipWait(gate,control,timeout_ns,cap);
                if ( SparkTpLdcvU64(&control->error_word) == 0u )
                {
                    SparkWeightdMeshRoute r;
                    r.word = next_route;
                    peer_mask = (uint32_t)r.fields.peer_mask;
                    __threadfence_system();
                    go = 1u;
                }
            }
        }
        go = __shfl_sync(0xffffffffu,go,0); peer_mask = __shfl_sync(0xffffffffu,peer_mask,0);
        if ( go != 0u )
            SparkTpMeshFastDmaOrDoorbellWarp0(entry,control,hybrid_ctrl,hybrid_shm,hybrid_band,
                16u,rank,peer_mask,is_col == 0u ? 1u : 0u,next_slot,slot_bytes,next_seq,next_tag,next_doorbell_bytes,next_route,
                next_half_elems * sizeof(uint16_t),cap,next_compact,
                is_col == 0u && !(next_compact != 0u && next_doorbell_bytes + 8u <= SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) ?
                    SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG : 0u,1u);
        if ( tid == 0u )
        {
            if ( go != 0u ) control->wait_started_ns = SparkTpFastTimerNs();
            __threadfence();
            LmDependentRelease();
        }
    }
    if ( blockIdx.x == 0u && tid < 32u && ok != 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,s_arr_end,arr_landed);
}

static __global__ __launch_bounds__(SPARK_TP_MESH_THREADS,2) void SparkTpMeshFast2dGatherKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *output,volatile uint64_t *mbox,
    uint32_t rank,uint32_t is_row_ag,uint64_t begin,uint64_t count,
    uint64_t base_offset,uint64_t piece_slice,uint64_t next_doorbell_bytes,uint64_t next_route,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,uint32_t last,
    SparkWeightdMeshHybridCtrl *hybrid_ctrl,uint8_t *hybrid_shm,uint32_t hybrid_band)
{
    __shared__ uint32_t s_go;
    __shared__ uint32_t s_last;
    __shared__ volatile uint32_t s_wait_over;
    __shared__ volatile uint64_t s_wait_start;
    const uint32_t k = blockIdx.y,stripe = blockIdx.x,tid = threadIdx.x;
    const uint32_t self_k = is_row_ag != 0u ? (rank & 3u) : (rank >> 2u);
    const uint32_t peer = SparkTpMeshFast2dGroupPeer(rank,is_row_ag != 0u ? 0u : 1u,k);
    const uint64_t lane = (uint64_t)stripe * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    uint32_t ok = 0u;
    LmDependentWait();
    const uint32_t entry_err = SparkTpLdcvU64(&control->error_word) != 0u ? 1u : 0u;
    const uint64_t cur_tag = SparkTpLdcvU64(&control->round_seq);
    const uint64_t cur_ring = (cur_tag - 1u) & (slots_per_rank - 1u);
    const uint64_t next_seq = SparkTpLdcvU64(&control->seq) + 1u;
    const uint64_t next_tag = (SparkTpLdcvU64(&control->epoch) << 32u) | next_seq;
    const uint64_t next_slot = (uint64_t)rank * slots_per_rank + ((next_seq - 1u) & (slots_per_rank - 1u));
    const uint32_t cap = SparkTpMeshFastCompactCap(gate);
    if ( entry_err == 0u )
    {
        const uint64_t mbox_ok = cur_tag ^ SPARK_TP_MESH_FAST_MBOX_OK_SALT;
        const uint64_t mbox_fail = cur_tag ^ SPARK_TP_MESH_FAST_MBOX_FAIL_SALT;
        const uint64_t start = base_offset + (uint64_t)k * piece_slice < count ? base_offset + (uint64_t)k * piece_slice : count;
        const uint64_t length = count - start < piece_slice ? count - start : piece_slice;
        uint16_t *next_band_dst = is_row_ag == 0u ? (uint16_t *)(band + next_slot * slot_bytes + (uint64_t)k * piece_slice * sizeof(uint16_t)) : 0;
        if ( k == self_k )
        {
            if ( stripe == 0u )
            {
                if ( tid == 0u ) { s_wait_over = 0u; s_wait_start = SparkTpFastTimerNs(); }
                __syncthreads();
                if ( tid < 32u )
                {
                    const uint32_t p_lane = tid < 4u ? SparkTpMeshFast2dGroupPeer(rank,is_row_ag != 0u ? 0u : 1u,tid) : rank;
                    const uint64_t p_start = base_offset + (uint64_t)tid * piece_slice < count ? base_offset + (uint64_t)tid * piece_slice : count;
                    const uint64_t p_len = count - p_start < piece_slice ? count - p_start : piece_slice;
                    const uint64_t p_gb = (p_len * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
                    const uint64_t p_db = p_gb != 0u ? p_gb : 8u;
                    const uint64_t tail_off = cap != 0u && p_db != 0u && (p_db & 7u) == 0u && p_db <= slot_bytes - 16u ? p_db : slot_bytes - 8u;
                    const volatile uint64_t *tail = (const volatile uint64_t *)(band +
                        ((uint64_t)p_lane * slots_per_rank + cur_ring) * slot_bytes + tail_off);
                    arr_base = SparkTpLdcvU64(&control->wait_started_ns);
                    ok = SparkTpMeshFast2dWarpWait(tail,cur_tag,tid < 4u && tid != self_k ? 1u : 0u,p_lane,
                        mbox,mbox_ok,mbox_fail,1u,control,s_wait_start,timeout_ns,arr_landed,arr_end);
                    if ( tid == 0u ) s_wait_over = 1u;
                }
                else
                {
                    if ( is_row_ag == 0u && length != 0u )
                        SparkTpMeshCopyBytesVolatile((volatile uint8_t *)next_band_dst,output + begin + start,length * sizeof(uint16_t),(uint64_t)(tid - 32u),lanes - 32u);
                    if ( is_row_ag != 0u )
                        SparkTpMeshFastRunPrefetch(prefetch,resume,tid,&s_wait_over);
                }
            }
            else if ( is_row_ag == 0u && length != 0u )
                SparkTpMeshCopyBytesVolatile((volatile uint8_t *)next_band_dst,output + begin + start,length * sizeof(uint16_t),lane - 32u,lanes - 32u);
        }
        else if ( length != 0u )
        {
            if ( tid == 0u )
            {
                const uint64_t p_gb = (length * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
                const uint64_t p_db = p_gb != 0u ? p_gb : 8u;
                const uint64_t tail_off = cap != 0u && p_db != 0u && (p_db & 7u) == 0u && p_db <= slot_bytes - 16u ? p_db : slot_bytes - 8u;
                const volatile uint64_t *tail = (const volatile uint64_t *)(band +
                    ((uint64_t)peer * slots_per_rank + cur_ring) * slot_bytes + tail_off);
                s_go = SparkTpMeshFastMboxWaitPeer(mbox,peer,mbox_ok,mbox_fail,tail,control,cur_tag,timeout_ns);
            }
            __syncthreads();
            if ( s_go != 0u )
            {
                const volatile uint16_t *src = (const volatile uint16_t *)(band + ((uint64_t)peer * slots_per_rank + cur_ring) * slot_bytes);
                uint16_t *out_dst = output + begin + start;
                if ( next_band_dst != 0 )
                {
                    uint64_t done = 0u;
                    if ( SparkTpMeshAligned32(out_dst,src) != 0u && (((uintptr_t)next_band_dst & 31u) == 0u) )
                    {
                        const uint64_t octs = length >> 4u;
                        for ( uint64_t i = lane; i < octs; i += lanes )
                        {
                            SparkTpU256 v = SparkTpLdCvU256(src + i * 16u);
                            SparkTpStU256(out_dst + i * 16u,v);
                            SparkTpStVolatileU256(next_band_dst + i * 16u,v);
                        }
                        done = octs << 4u;
                    }
                    if ( SparkTpMeshAligned16(out_dst + done,src + done) != 0u && (((uintptr_t)(next_band_dst + done) & 15u) == 0u) )
                    {
                        const uint64_t quads = (length - done) >> 3u;
                        for ( uint64_t i = lane; i < quads; i += lanes )
                        {
                            uint4 v = SparkTpLdCvU128(src + done + i * 8u);
                            ((uint4 *)(out_dst + done))[i] = v;
                            SparkTpStVolatileU128(next_band_dst + done + i * 8u,v);
                        }
                        done += quads << 3u;
                    }
                    for ( uint64_t i = done + lane; i < length; i += lanes )
                    {
                        uint16_t v = src[i];
                        out_dst[i] = v;
                        next_band_dst[i] = v;
                    }
                }
                else
                    SparkTpMeshCopyHalvesFromBand(out_dst,src,length,lane,lanes);
            }
        }
        if ( is_row_ag == 0u ) __threadfence_system();
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull ==
            (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( s_last != 0u && tid < 32u )
    {
        uint32_t go = 0u,next_compact = 0u,peer_mask = 0u;
        if ( tid == 0u )
        {
            const uint64_t end_ns = SparkTpFastTimerNs();
            const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
            control->math_blocks_done = 0u;
            control->math_started_ns = 0u;
            control->math_finished_ns = 0u;
            SparkTpMeshFastClearMbox(mbox);
            if ( SparkTpLdcvU64(&control->error_word) == 0u )
            {
                if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
                if ( is_row_ag == 0u )
                {
                    next_compact = cap != 0u && next_doorbell_bytes != 0u &&
                        (next_doorbell_bytes & 7u) == 0u && next_doorbell_bytes <= slot_bytes - 16u ? 1u : 0u;
                    *(volatile uint64_t *)(band + next_slot * slot_bytes + (next_compact != 0u ? next_doorbell_bytes : slot_bytes - 8u)) = next_tag;
                    SparkTpMeshFastShipWait(gate,control,timeout_ns,cap);
                    if ( SparkTpLdcvU64(&control->error_word) == 0u )
                    {
                        SparkWeightdMeshRoute r;
                        r.word = next_route;
                        peer_mask = (uint32_t)r.fields.peer_mask;
                        __threadfence_system();
                        go = 1u;
                    }
                }
                else
                {
                    control->slot_cursor = SparkTpLdcvU64(&control->seq);
                    if ( last != 0u ) control->rounds_done++;
                }
            }
        }
        go = __shfl_sync(0xffffffffu,go,0); next_compact = __shfl_sync(0xffffffffu,next_compact,0); peer_mask = __shfl_sync(0xffffffffu,peer_mask,0);
        if ( go != 0u )
            SparkTpMeshFastDmaOrDoorbellWarp0(entry,control,hybrid_ctrl,hybrid_shm,hybrid_band,
                16u,rank,peer_mask,0u,next_slot,slot_bytes,next_seq,next_tag,next_doorbell_bytes,next_route,0u,cap,next_compact,0u,1u);
        if ( tid == 0u )
        {
            if ( go != 0u ) control->wait_started_ns = SparkTpFastTimerNs();
            __threadfence();
            LmDependentRelease();
        }
    }
    if ( k == self_k && stripe == 0u && tid < 32u && ok != 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static SparkGpuError SparkTpMeshFast2dRsagChunk(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,
    const uint8_t *local,void *output,volatile uint64_t *mbox,uint8_t *staging,
    uint64_t total_elements,uint64_t begin,uint64_t count,uint64_t timeout_ns)
{
    SparkWeightdMeshHybridCtrl *hybrid_ctrl = 0;
    uint8_t *hybrid_shm = 0;
    uint32_t hybrid_band = 0u;
    SparkTpMeshFastHybridResolve(stream,(const void *)entry,(const void *)request,slot_bytes,&hybrid_ctrl,&hybrid_shm,&hybrid_band);
    const uint32_t row = rank >> 2u,col = rank & 3u;
    const uint32_t row_mask = (0xfu << (row << 2u)) & ~(1u << rank);
    const uint32_t col_mask = (0x1111u << col) & ~(1u << rank);
    const uint64_t sub_slice = ((count + 15u) / 16u + 7u) & ~UINT64_C(7);
    const uint64_t col_slice = 4u * sub_slice;
    const uint64_t staging_slot_elements = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
    const uint64_t col_bytes = col_slice * sizeof(uint16_t);
    const uint64_t sub_bytes = sub_slice * sizeof(uint16_t);
    if ( staging == 0 || col_slice > staging_slot_elements || col_bytes > slot_bytes - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    const uint64_t col_first = (uint64_t)col * col_slice < count ? (uint64_t)col * col_slice : count;
    const uint64_t col_span = count - col_first < col_slice ? count - col_first : col_slice;
    const uint64_t sub_first = col_first + (uint64_t)row * sub_slice < count ? col_first + (uint64_t)row * sub_slice : count;
    const uint64_t sub_span = count - sub_first < sub_slice ? count - sub_first : sub_slice;
    const uint64_t sub_ag_gb = (sub_span * sizeof(uint16_t) + 7u) & ~UINT64_C(7),sub_ag_db = sub_ag_gb != 0u ? sub_ag_gb : 8u;
    const uint64_t col_ag_gb = (col_span * sizeof(uint16_t) + 7u) & ~UINT64_C(7),col_ag_db = col_ag_gb != 0u ? col_ag_gb : 8u;
    const uint32_t sm_count = SparkTpMeshMultiprocessors();
    SparkWeightdMeshRoute r_row_rs = {0},r_col_rs = {0},r_col_ag = {0},r_row_ag = {0};
    r_row_rs.fields.peer_mask = row_mask; r_row_rs.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER; r_row_rs.fields.slice_bytes = col_bytes;
    r_col_rs.fields.peer_mask = col_mask; r_col_rs.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER; r_col_rs.fields.slice_bytes = sub_bytes;
    r_col_ag.fields.peer_mask = col_mask; r_col_ag.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    r_row_ag.fields.peer_mask = row_mask; r_row_ag.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    uint32_t stage_stripes = (uint32_t)((col_bytes + 32767u) / 32768u);
    const uint32_t max_stage_stripes = sm_count / 3u;
    if ( stage_stripes < 1u ) stage_stripes = 1u;
    if ( stage_stripes > max_stage_stripes ) stage_stripes = max_stage_stripes > 0u ? max_stage_stripes : 1u;
    SparkGpuError status = LM_LAUNCH_DEPENDENT((SparkTpMeshFast2dStageKernel),dim3(stage_stripes,3u),SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,request,control,(uint16_t *)staging,staging_slot_elements,
        (const uint16_t *)local + begin,mbox,rank,count,col_slice,col_bytes,r_row_rs.word,timeout_ns,
        hybrid_ctrl,hybrid_shm,hybrid_band);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    uint32_t r1_blocks = (uint32_t)((col_span / 16u + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
    const uint32_t r_cap = sm_count < 8u ? sm_count : 8u;
    if ( r1_blocks < 1u ) r1_blocks = 1u;
    if ( r1_blocks > r_cap ) r1_blocks = r_cap;
    status = LM_LAUNCH_DEPENDENT((SparkTpMeshFast2dReducePublishKernel),r1_blocks,SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,request,control,(uint16_t *)staging,staging_slot_elements,
        (const uint16_t *)local + begin + col_first,(uint16_t *)output,mbox,rank,0u,begin + col_first,
        col_span,sub_slice,col_bytes,sub_bytes,r_col_rs.word,timeout_ns,hybrid_ctrl,hybrid_shm,hybrid_band);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    uint32_t r2_blocks = (uint32_t)((sub_span / 16u + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
    if ( r2_blocks < 1u ) r2_blocks = 1u;
    if ( r2_blocks > r_cap ) r2_blocks = r_cap;
    status = LM_LAUNCH_DEPENDENT((SparkTpMeshFast2dReducePublishKernel),r2_blocks,SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,request,control,(uint16_t *)staging,staging_slot_elements,
        (const uint16_t *)output + begin + sub_first,(uint16_t *)output,mbox,rank,1u,begin + sub_first,
        sub_span,sub_slice,sub_bytes,sub_ag_db,r_col_ag.word,timeout_ns,hybrid_ctrl,hybrid_shm,hybrid_band);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    const uint32_t g1_stripes = SparkTpMeshFastGatherStripes(sub_slice,4u);
    status = LM_LAUNCH_DEPENDENT((SparkTpMeshFast2dGatherKernel),dim3(g1_stripes,4u),SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,request,control,(uint16_t *)output,mbox,
        rank,0u,begin,count,col_first,sub_slice,col_ag_db,r_row_ag.word,timeout_ns,(LmWeightPrefetchGroup *)0,0u,0u,
        hybrid_ctrl,hybrid_shm,hybrid_band);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const uint32_t g2_stripes = SparkTpMeshFastGatherStripes(col_slice,4u);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFast2dGatherKernel),dim3(g2_stripes,4u),SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,request,control,(uint16_t *)output,mbox,
        rank,1u,begin,count,0u,col_slice,0u,0u,timeout_ns,prefetch,resume,begin + count == total_elements ? 1u : 0u,
        hybrid_ctrl,hybrid_shm,hybrid_band);
}

static SparkGpuError SparkTpMeshFastPeerRsagChunk(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,volatile uint64_t *mbox,uint8_t *staging,uint64_t local_elements,uint64_t begin,uint64_t count,
    uint64_t timeout_ns,uint32_t stage)
{
    SparkWeightdMeshRoute route = {0};
    const uint64_t slice = SparkTpMeshRsagSlice(count,degree);
    const uint64_t staging_slot_elements = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
    const uint64_t first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count;
    const uint64_t span = count - first < slice ? count - first : slice;
    uint64_t gather_bytes;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    if ( staging == 0 || slice > staging_slot_elements ) return SPARK_GPU_ERROR_INVALID_VALUE;
    const uint16_t *own_local = (const uint16_t *)local + begin + first;
    const uint16_t *out_slice = (const uint16_t *)output + begin + first;
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const uint64_t in_bytes = slice * sizeof(uint16_t);
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = in_bytes;
    SparkWeightdMeshHybridCtrl *hybrid_ctrl = 0;
    uint8_t *hybrid_shm = 0;
    uint32_t hybrid_band = 0u;
    SparkTpMeshFastHybridResolve(stream,(const void *)entry,(const void *)request,slot_bytes,&hybrid_ctrl,&hybrid_shm,&hybrid_band);
    if ( stage == SPARK_TP_MESH_STAGE_ALL && count <= SPARK_TP_MESH_FAST_RSAG_FUSED_MAX_ELEMENTS &&
         span != 0u && (span & 7u) == 0u &&
         (((uintptr_t)band | (uintptr_t)out_slice | (uintptr_t)own_local) & 15u) == 0u &&
         (slot_bytes & 15u) == 0u &&
         in_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES && in_bytes <= slot_bytes - 16u )
    {
        SparkWeightdMeshRoute ag_route = {0};
        ag_route.fields.peer_mask = route.fields.peer_mask;
        ag_route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
        gather_bytes = (span * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
        const uint64_t ag_db = gather_bytes != 0u ? gather_bytes : 8u;
        return LM_LAUNCH_DEPENDENT((SparkTpMeshFastRsagFusedKernel),1u,SPARK_TP_MESH_FAST_FUSED_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,entry,request,control,(uint16_t *)staging,staging_slot_elements,
            (const uint16_t *)local,(uint16_t *)output,degree,rank,begin,count,slice,first,span,
            in_bytes,route.word,ag_db,ag_route.word,timeout_ns,prefetch,resume,
            begin + count == local_elements ? 1u : 0u,hybrid_ctrl,hybrid_shm,hybrid_band);
    }
    const uint32_t wide256 = degree <= 4u && (span & 15u) == 0u &&
        (((uintptr_t)band | (uintptr_t)out_slice) & 31u) == 0u && (slot_bytes & 31u) == 0u &&
        (stage != SPARK_TP_MESH_STAGE_ALL || (((uintptr_t)own_local & 31u) == 0u)) ? 1u : 0u;
    const uint32_t prefetch_budget = stage != SPARK_TP_MESH_STAGE_ALL ? 1u : (prefetch != 0 ? 1u : 0u);
    const uint32_t stream_reduce_ok = span != 0u && span <= SPARK_TP_MESH_FAST_RSAG_STREAM_SPAN_MAX &&
        (span & 7u) == 0u &&
        (((uintptr_t)band | (uintptr_t)out_slice) & 15u) == 0u && (slot_bytes & 15u) == 0u &&
        (stage != SPARK_TP_MESH_STAGE_ALL || (((uintptr_t)own_local & 15u) == 0u)) &&
        SparkTpMeshFastCanStreamReduce(span,degree,wide256,prefetch_budget) != 0u ? 1u : 0u;
    if ( stage != SPARK_TP_MESH_STAGE_COMPLETE )
    {
        const uint32_t copy_self = stage == SPARK_TP_MESH_STAGE_ALL && stream_reduce_ok != 0u ? 0u : 1u;
        status = SparkTpMeshFastLaunchPeerStage(stream,band,slot_bytes,slots_per_rank,entry,request,control,
            (uint16_t *)staging,staging_slot_elements,(const uint16_t *)local + begin,mbox,degree,rank,
            count,0u,slice,in_bytes,route.word,timeout_ns,1u,copy_self,stream_reduce_ok);
        if ( status != SPARK_GPU_SUCCESS || stage == SPARK_TP_MESH_STAGE_PUBLISH ) return status;
    }
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    route.fields.slice_bytes = 0u;
    gather_bytes = (span * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
    const uint64_t doorbell_bytes = gather_bytes != 0u ? gather_bytes : 8u;
    if ( stream_reduce_ok != 0u )
    {
        const uint16_t *local_arg = stage == SPARK_TP_MESH_STAGE_ALL ? own_local : 0;
        const uint32_t vecs = (uint32_t)(span / (wide256 != 0u ? 16u : 8u));
        const uint32_t compute_blocks = (vecs + SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS - 1u) /
            SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS;
        const uint32_t grid = compute_blocks + (prefetch != 0 ? 1u : 0u);
        if ( degree <= 4u )
        {
            if ( wide256 != 0u )
                status = LM_LAUNCH_DEPENDENT((SparkTpMeshFastRsagStreamReducePublishKernel<4u,1u>),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
                    band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,timeout_ns,prefetch,resume,
                    local_arg,(uint16_t *)output,begin + first,mbox,vecs,compute_blocks,in_bytes,doorbell_bytes,route.word,
                    hybrid_ctrl,hybrid_shm,hybrid_band);
            else
                status = LM_LAUNCH_DEPENDENT((SparkTpMeshFastRsagStreamReducePublishKernel<4u,0u>),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
                    band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,timeout_ns,prefetch,resume,
                    local_arg,(uint16_t *)output,begin + first,mbox,vecs,compute_blocks,in_bytes,doorbell_bytes,route.word,
                    hybrid_ctrl,hybrid_shm,hybrid_band);
        }
        else
            status = LM_LAUNCH_DEPENDENT((SparkTpMeshFastRsagStreamReducePublishKernel<16u,0u>),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
                band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,timeout_ns,prefetch,resume,
                local_arg,(uint16_t *)output,begin + first,mbox,vecs,compute_blocks,in_bytes,doorbell_bytes,route.word,
                hybrid_ctrl,hybrid_shm,hybrid_band);
    }
    else
    {
        status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
        if ( status == SPARK_GPU_SUCCESS )
            status = LM_LAUNCH_DEPENDENT((SparkTpMeshFastRsagWideReducePublishKernel),SparkTpMeshDirectBlocks(span != 0u ? span : 1u,1u),
                SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,request,control,
                rank,degree,timeout_ns,output,local_elements,begin,count,first,span,doorbell_bytes,route.word,
                hybrid_ctrl,hybrid_shm,hybrid_band);
    }
    if ( status != SPARK_GPU_SUCCESS ) return status;
    return SparkTpMeshFastLaunchWideStreamGather(stream,band,slot_bytes,slots_per_rank,request,control,rank,degree,
        0,(uint16_t *)output,mbox,begin,count,slice,1u,1u,begin + count == local_elements ? 1u : 0u,timeout_ns);
}

static SparkGpuError SparkTpMeshFastLaunchStreamReduce(SparkGpuStream stream,
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    const SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint16_t *local,uint16_t *output,volatile uint64_t *mbox,uint64_t elements,uint64_t compact_bytes,uint64_t timeout_ns)
{
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const uint32_t wide256 = degree <= 4u && (elements & 15u) == 0u &&
        (((uintptr_t)band | (uintptr_t)output) & 31u) == 0u && (slot_bytes & 31u) == 0u &&
        (local == 0 || (((uintptr_t)local & 31u) == 0u)) ? 1u : 0u;
    const uint32_t vecs = (uint32_t)(elements / (wide256 != 0u ? 16u : 8u));
    const uint32_t compute_blocks = (vecs + SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS - 1u) /
        SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS;
    const uint32_t grid = compute_blocks + (prefetch != 0 ? 1u : 0u);
    if ( degree <= 4u )
    {
        if ( wide256 != 0u )
            return LM_LAUNCH_DEPENDENT((SparkTpMeshFastStreamAllReduceKernel<4u,1u>),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
                band,slot_bytes,slots_per_rank,gate,control,rank,degree,timeout_ns,prefetch,resume,local,output,mbox,vecs,compute_blocks,compact_bytes);
        return LM_LAUNCH_DEPENDENT((SparkTpMeshFastStreamAllReduceKernel<4u,0u>),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,gate,control,rank,degree,timeout_ns,prefetch,resume,local,output,mbox,vecs,compute_blocks,compact_bytes);
    }
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastStreamAllReduceKernel<16u,0u>),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,gate,control,rank,degree,timeout_ns,prefetch,resume,local,output,mbox,vecs,compute_blocks,compact_bytes);
}

static SparkGpuError SparkTpMeshFastLaunchGatherFused(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    uint32_t rank,uint32_t degree,const uint16_t *local,uint16_t *output,uint64_t local_elements,
    uint64_t route,uint64_t timeout_ns,uint32_t fused_op)
{
    uint32_t resume = 0u,hb = 0u;
    SparkWeightdMeshHybridCtrl *hc = 0; uint8_t *hs = 0;
    SparkTpMeshFastHybridResolve(stream,(const void *)entry,(const void *)gate,slot_bytes,&hc,&hs,&hb);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastGatherFusedKernel),1u,SPARK_TP_MESH_FAST_FUSED_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,gate,control,staging,staging_slot_elements,rank,degree,
        local,output,local_elements,route,timeout_ns,LmWeightPrefetchSlot(stream,&resume),resume,fused_op,hc,hs,hb);
}

static __device__ __forceinline__ uint8_t SparkTpLdCvU8(const volatile void *address)
{
    uint32_t v;
    asm volatile("ld.global.cv.u8 %0,[%1];" : "=r"(v) : "l"(address) : "memory");
    return (uint8_t)v;
}

static __device__ __forceinline__ uint16_t SparkTpFloatPairToE4m3(float lo,float hi)
{
    uint16_t enc;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(enc) : "f"(hi),"f"(lo));
    return enc;
}

static __device__ __forceinline__ float2 SparkTpE4m3PairToFloat2(uint16_t enc)
{
    uint32_t widened;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(widened) : "h"(enc));
    float2 out;
    out.x = __half2float(__ushort_as_half((uint16_t)(widened & 0xffffu)));
    out.y = __half2float(__ushort_as_half((uint16_t)(widened >> 16u)));
    return out;
}

static __device__ __forceinline__ uint8_t SparkTpFloatToUe8m0Ceil(float v)
{
    if ( !(v > 0.0f) ) return 0u;
    uint32_t bits = __float_as_uint(v);
    uint32_t exp = (bits >> 23u) & 0xffu;
    if ( (bits & 0x7fffffu) != 0u && exp < 254u ) exp += 1u;
    if ( exp == 0u ) exp = 1u;
    return (uint8_t)exp;
}

static __device__ __forceinline__ float SparkTpUe8m0ToFloat(uint8_t code)
{
    return code == 0u ? 0.0f : __uint_as_float((uint32_t)code << 23u);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __host__ __device__ __forceinline__ uint64_t SparkTpMeshWireBytes(uint64_t elements)
{
    const uint64_t raw = elements + (elements / BLOCK) * (uint64_t)SCALE_BYTES;
    return (raw + 7u) & ~UINT64_C(7);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static inline uint64_t SparkTpMeshWireHalfWidth(void)
{
    const uint64_t per256 = 256u + (256u / BLOCK) * (uint64_t)SCALE_BYTES;
    return (SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / per256) * 256u;
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __device__ __forceinline__ float SparkTpMeshWireLoadScale(const uint8_t *scale_base,uint64_t oct_idx)
{
    const uint64_t sidx = BLOCK == 32u ? (oct_idx >> 1u) : oct_idx;
    if constexpr ( SCALE_BYTES == 1u )
        return SparkTpUe8m0ToFloat(SparkTpLdCvU8(scale_base + sidx));
    else
        return __uint_as_float((uint32_t)SparkTpLdCvU16(scale_base + sidx * 2u) << 16u);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __device__ __forceinline__ void SparkTpMeshWireStoreScaleWarp(
    uint8_t *scale_base,uint64_t oct_idx,uint64_t octs,uint16_t sc_bits)
{
    if constexpr ( BLOCK == 32u && SCALE_BYTES == 1u )
    {
        const uint32_t s0 = (uint32_t)(sc_bits & 0xffu);
        const uint32_t s1 = __shfl_down_sync(0xffffffffu,s0,2);
        const uint32_t s2 = __shfl_down_sync(0xffffffffu,s0,4);
        const uint32_t s3 = __shfl_down_sync(0xffffffffu,s0,6);
        if ( (oct_idx & 7u) == 0u && oct_idx + 7u < octs )
            ((uint32_t *)scale_base)[oct_idx >> 3u] = s0 | (s1 << 8u) | (s2 << 16u) | (s3 << 24u);
        else if ( (oct_idx & 1u) == 0u && oct_idx < octs && (oct_idx & ~UINT64_C(7)) + 7u >= octs )
            scale_base[oct_idx >> 1u] = (uint8_t)s0;
    }
    else if constexpr ( BLOCK == 32u && SCALE_BYTES == 2u )
    {
        const uint32_t s0 = (uint32_t)sc_bits;
        const uint32_t s1 = __shfl_down_sync(0xffffffffu,s0,2);
        if ( (oct_idx & 3u) == 0u && oct_idx + 3u < octs )
            ((uint32_t *)scale_base)[oct_idx >> 2u] = s0 | (s1 << 16u);
        else if ( (oct_idx & 1u) == 0u && oct_idx < octs && (oct_idx & ~UINT64_C(3)) + 3u >= octs )
            ((uint16_t *)scale_base)[oct_idx >> 1u] = sc_bits;
    }
    else
    {
        const uint32_t s0 = (uint32_t)sc_bits;
        const uint32_t s1 = __shfl_down_sync(0xffffffffu,s0,1);
        if ( (oct_idx & 1u) == 0u && oct_idx + 1u < octs )
            ((uint32_t *)scale_base)[oct_idx >> 1u] = s0 | (s1 << 16u);
    }
}

static __device__ __forceinline__ float SparkTpU32Bf16PairAbsMax(uint32_t w,float cur)
{
    float a = fabsf(__uint_as_float(w << 16u));
    float b = fabsf(__uint_as_float(w & 0xffff0000u));
    return fmaxf(cur,fmaxf(a,b));
}

static __device__ __forceinline__ uint32_t SparkTpQuantQuadBf16ToE4m3(uint32_t w0,uint32_t w1,float inv)
{
    uint16_t lo = SparkTpFloatPairToE4m3(__uint_as_float(w0 << 16u) * inv,__uint_as_float(w0 & 0xffff0000u) * inv);
    uint16_t hi = SparkTpFloatPairToE4m3(__uint_as_float(w1 << 16u) * inv,__uint_as_float(w1 & 0xffff0000u) * inv);
    return (uint32_t)lo | ((uint32_t)hi << 16u);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __device__ __forceinline__ void SparkTpMeshWireQuantOct(
    SparkTpU256 raw,uint32_t valid,uint4 &packed_out,uint16_t &sc_bits_out,float &sc_val_out)
{
    float amax = 0.0f;
    if ( valid != 0u )
    {
        amax = SparkTpU32Bf16PairAbsMax(raw.lo.x,amax);
        amax = SparkTpU32Bf16PairAbsMax(raw.lo.y,amax);
        amax = SparkTpU32Bf16PairAbsMax(raw.lo.z,amax);
        amax = SparkTpU32Bf16PairAbsMax(raw.lo.w,amax);
        amax = SparkTpU32Bf16PairAbsMax(raw.hi.x,amax);
        amax = SparkTpU32Bf16PairAbsMax(raw.hi.y,amax);
        amax = SparkTpU32Bf16PairAbsMax(raw.hi.z,amax);
        amax = SparkTpU32Bf16PairAbsMax(raw.hi.w,amax);
    }
    if constexpr ( BLOCK == 32u )
        amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,1));
    uint16_t sc_bits = 0u;
    float sc = 0.0f;
    if ( valid != 0u && amax > 0.0f )
    {
        if constexpr ( SCALE_BYTES == 1u )
        {
            uint8_t code = SparkTpFloatToUe8m0Ceil(amax * (1.0f / 448.0f));
            sc_bits = (uint16_t)code;
            sc = SparkTpUe8m0ToFloat(code);
        }
        else
        {
            sc_bits = SparkTpFloatToBf16(amax * (1.0f / 448.0f));
            sc = __uint_as_float((uint32_t)sc_bits << 16u);
            if ( sc == 0.0f )
            {
                sc_bits = 0x0080u;
                sc = __uint_as_float(0x00800000u);
            }
        }
    }
    const float inv = sc > 0.0f ? (1.0f / sc) : 0.0f;
    packed_out.x = valid != 0u ? SparkTpQuantQuadBf16ToE4m3(raw.lo.x,raw.lo.y,inv) : 0u;
    packed_out.y = valid != 0u ? SparkTpQuantQuadBf16ToE4m3(raw.lo.z,raw.lo.w,inv) : 0u;
    packed_out.z = valid != 0u ? SparkTpQuantQuadBf16ToE4m3(raw.hi.x,raw.hi.y,inv) : 0u;
    packed_out.w = valid != 0u ? SparkTpQuantQuadBf16ToE4m3(raw.hi.z,raw.hi.w,inv) : 0u;
    sc_bits_out = sc_bits;
    sc_val_out = sc;
}

static __device__ __forceinline__ SparkTpU256 SparkTpMeshWireDequantOctToU256(uint4 packed,float sc)
{
    SparkTpU256 out;
    float2 a = SparkTpE4m3PairToFloat2((uint16_t)(packed.x & 0xffffu));
    float2 b = SparkTpE4m3PairToFloat2((uint16_t)(packed.x >> 16u));
    out.lo.x = LmPackBf16Pair(a.x * sc,a.y * sc);
    out.lo.y = LmPackBf16Pair(b.x * sc,b.y * sc);
    a = SparkTpE4m3PairToFloat2((uint16_t)(packed.y & 0xffffu));
    b = SparkTpE4m3PairToFloat2((uint16_t)(packed.y >> 16u));
    out.lo.z = LmPackBf16Pair(a.x * sc,a.y * sc);
    out.lo.w = LmPackBf16Pair(b.x * sc,b.y * sc);
    a = SparkTpE4m3PairToFloat2((uint16_t)(packed.z & 0xffffu));
    b = SparkTpE4m3PairToFloat2((uint16_t)(packed.z >> 16u));
    out.hi.x = LmPackBf16Pair(a.x * sc,a.y * sc);
    out.hi.y = LmPackBf16Pair(b.x * sc,b.y * sc);
    a = SparkTpE4m3PairToFloat2((uint16_t)(packed.w & 0xffffu));
    b = SparkTpE4m3PairToFloat2((uint16_t)(packed.w >> 16u));
    out.hi.z = LmPackBf16Pair(a.x * sc,a.y * sc);
    out.hi.w = LmPackBf16Pair(b.x * sc,b.y * sc);
    return out;
}

static __device__ __forceinline__ void SparkTpMeshWireAccumOct(float sum[16],uint4 packed,float sc)
{
    float2 a = SparkTpE4m3PairToFloat2((uint16_t)(packed.x & 0xffffu));
    float2 b = SparkTpE4m3PairToFloat2((uint16_t)(packed.x >> 16u));
    sum[0] = fmaf(a.x,sc,sum[0]); sum[1] = fmaf(a.y,sc,sum[1]);
    sum[2] = fmaf(b.x,sc,sum[2]); sum[3] = fmaf(b.y,sc,sum[3]);
    a = SparkTpE4m3PairToFloat2((uint16_t)(packed.y & 0xffffu));
    b = SparkTpE4m3PairToFloat2((uint16_t)(packed.y >> 16u));
    sum[4] = fmaf(a.x,sc,sum[4]); sum[5] = fmaf(a.y,sc,sum[5]);
    sum[6] = fmaf(b.x,sc,sum[6]); sum[7] = fmaf(b.y,sc,sum[7]);
    a = SparkTpE4m3PairToFloat2((uint16_t)(packed.z & 0xffffu));
    b = SparkTpE4m3PairToFloat2((uint16_t)(packed.z >> 16u));
    sum[8] = fmaf(a.x,sc,sum[8]); sum[9] = fmaf(a.y,sc,sum[9]);
    sum[10] = fmaf(b.x,sc,sum[10]); sum[11] = fmaf(b.y,sc,sum[11]);
    a = SparkTpE4m3PairToFloat2((uint16_t)(packed.w & 0xffffu));
    b = SparkTpE4m3PairToFloat2((uint16_t)(packed.w >> 16u));
    sum[12] = fmaf(a.x,sc,sum[12]); sum[13] = fmaf(a.y,sc,sum[13]);
    sum[14] = fmaf(b.x,sc,sum[14]); sum[15] = fmaf(b.y,sc,sum[15]);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __device__ __forceinline__ void SparkTpMeshWireQuantRange(
    uint8_t *dst_base,uint64_t scale_off,const uint16_t *src,uint64_t count,uint64_t lane,uint64_t lanes)
{
    const uint64_t octs = count >> 4u;
    const uint64_t loop_octs = (octs + 31u) & ~UINT64_C(31);
    uint8_t *scale_base = dst_base + scale_off;
    for ( uint64_t i = lane; i < loop_octs; i += lanes )
    {
        const uint32_t valid = i < octs ? 1u : 0u;
        SparkTpU256 raw = {};
        if ( valid != 0u ) raw = SparkTpLdNcU256(src + i * 16u);
        uint4 packed;
        uint16_t sc_bits;
        float sc_val;
        SparkTpMeshWireQuantOct<BLOCK,SCALE_BYTES>(raw,valid,packed,sc_bits,sc_val);
        if ( valid != 0u )
            ((uint4 *)dst_base)[i] = packed;
        SparkTpMeshWireStoreScaleWarp<BLOCK,SCALE_BYTES>(scale_base,i,octs,sc_bits);
    }
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __device__ __forceinline__ void SparkTpMeshWireDequantRange(
    uint16_t *dst,const uint8_t *src_base,uint64_t scale_off,uint64_t count,uint64_t lane,uint64_t lanes)
{
    const uint64_t octs = count >> 4u;
    const uint8_t *scale_base = src_base + scale_off;
    for ( uint64_t i = lane; i < octs; i += lanes )
    {
        const uint4 packed = SparkTpLdCvU128(src_base + i * 16u);
        const float sc = SparkTpMeshWireLoadScale<BLOCK,SCALE_BYTES>(scale_base,i);
        SparkTpStU256(dst + i * 16u,SparkTpMeshWireDequantOctToU256(packed,sc));
    }
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __device__ __forceinline__ SparkTpU256 SparkTpMeshWireSumOctValue(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,uint64_t ring,
    uint32_t degree,uint64_t scale_off,uint64_t oct_idx)
{
    float sum[16] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    const uint64_t stride = slots_per_rank * slot_bytes;
    const uint8_t *base = band + ring * slot_bytes;
    for ( uint32_t peer = 0u; peer < degree; peer += 4u )
    {
        uint4 v[4];
        float sc[4];
        const uint32_t n = degree - peer < 4u ? degree - peer : 4u;
#pragma unroll
        for ( uint32_t j = 0u; j < 4u; ++j )
            if ( j < n )
            {
                const uint8_t *pbase = base + (uint64_t)(peer + j) * stride;
                v[j] = SparkTpLdCvU128(pbase + oct_idx * 16u);
                sc[j] = SparkTpMeshWireLoadScale<BLOCK,SCALE_BYTES>(pbase + scale_off,oct_idx);
            }
#pragma unroll
        for ( uint32_t j = 0u; j < 4u; ++j )
            if ( j < n )
                SparkTpMeshWireAccumOct(sum,v[j],sc[j]);
    }
    return SparkTpPackU256Bf16(sum);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshRsagMxfp8StageKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkTpMeshRoundControl *control,
    uint8_t *staging,uint64_t staging_slot_bytes,const uint16_t *local,uint32_t rank,
    uint64_t total_count,uint64_t slice,uint64_t source_stride,uint64_t scale_off,uint64_t doorbell_bytes,uint64_t route,uint32_t is_scatter)
{
    __shared__ uint32_t last;
    LmDependentWait();
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs());
    const uint64_t prev_seq = SparkTpLdcvU64(&control->seq);
    const uint32_t failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = prev_seq + 1u;
    const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    if ( failed == 0u )
    {
        const uint64_t peer = blockIdx.y;
        const uint64_t first = is_scatter != 0u ? (peer * source_stride) : (peer * slice < total_count ? peer * slice : total_count);
        const uint64_t length = is_scatter != 0u ? slice : (total_count - first < slice ? total_count - first : slice);
        uint8_t *dst = peer == rank ? (band + slot * slot_bytes) : (staging + peer * staging_slot_bytes);
        if ( length != 0u )
            SparkTpMeshWireQuantRange<BLOCK,SCALE_BYTES>(dst,scale_off,local + first,length,
                (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    }
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( last == 0u || threadIdx.x != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( SparkTpLdcvU64(&control->error_word) == 0u && prev_seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
    if ( failed == 0u )
    {
        const uint64_t tag = (SparkTpLdcvU64(&control->epoch) << 32u) | sequence;
        entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route;
        control->round_seq_prior = SparkTpLdcvU64(&control->round_seq) | SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG | SPARK_TP_MESH_FAST_PRIOR_FULL_STAGING_FLAG; control->round_seq = tag; control->seq = sequence;
        __threadfence_system();
        *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - SparkTpLdcvU64(&control->publish_started_ns);
    }
    control->publish_started_ns = 0u;
    control->wait_started_ns = SparkTpGlobalTimerNs();
    __threadfence();
    LmDependentRelease();
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES,uint32_t PHASE1_QUANT>
static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshRsagMxfp8CombineKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,
    uint16_t *output,uint64_t out_offset,uint64_t slice_scale_off,uint64_t span,
    uint64_t next_doorbell_bytes,uint64_t next_route)
{
    __shared__ uint32_t s_last;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint64_t seq = SparkTpLdcvU64(&control->seq);
    const uint64_t ring = (SparkTpLdcvU64(&control->round_seq) - 1u) & (slots_per_rank - 1u);
    const uint32_t next_overflow = seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t next_seq = seq + 1u;
    const uint64_t next_slot = (uint64_t)rank * slots_per_rank + ((next_seq - 1u) & (slots_per_rank - 1u));
    uint8_t *next_dst = next_overflow == 0u ? (band + next_slot * slot_bytes) : 0;
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->math_started_ns,0ull,started);
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lanes = (uint64_t)blockDim.x * gridDim.x;
    const uint64_t octs = span >> 4u;
    const uint64_t loop_octs = (octs + 31u) & ~UINT64_C(31);
    uint16_t *out16 = output + out_offset;
    for ( uint64_t i = lane; i < loop_octs; i += lanes )
    {
        const uint32_t valid = i < octs ? 1u : 0u;
        SparkTpU256 sum_bf16 = {};
        if ( valid != 0u )
        {
            if constexpr ( PHASE1_QUANT != 0u )
                sum_bf16 = SparkTpMeshWireSumOctValue<BLOCK,SCALE_BYTES>(band,slot_bytes,slots_per_rank,ring,degree,slice_scale_off,i);
            else
                sum_bf16 = SparkTpMeshHardwareSumOctValue(band,slot_bytes,slots_per_rank,ring,degree,i * 16u);
        }
        uint4 packed;
        uint16_t sc_bits;
        float sc_val;
        SparkTpMeshWireQuantOct<BLOCK,SCALE_BYTES>(sum_bf16,valid,packed,sc_bits,sc_val);
        if ( valid != 0u )
        {
            SparkTpStU256(out16 + i * 16u,SparkTpMeshWireDequantOctToU256(packed,sc_val));
            if ( next_dst != 0 )
                ((uint4 *)next_dst)[i] = packed;
        }
        if ( next_dst != 0 )
            SparkTpMeshWireStoreScaleWarp<BLOCK,SCALE_BYTES>(next_dst + span,i,octs,sc_bits);
    }
    if ( next_dst != 0 ) __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || threadIdx.x != 0u ) return;
    control->publish_blocks_done = 0u;
    const uint64_t end_ns = SparkTpGlobalTimerNs();
    const uint64_t begin_ns = SparkTpLdcvU64(&control->math_started_ns);
    control->math_started_ns = 0u;
    control->math_finished_ns = 0u;
    control->math_blocks_done = 0u;
    if ( begin_ns != 0u && end_ns > begin_ns ) control->combine_ns += end_ns - begin_ns;
    if ( next_overflow != 0u )
    {
        if ( SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
    }
    else if ( SparkTpLdcvU64(&control->error_word) == 0u )
    {
        const uint64_t next_tag = (SparkTpLdcvU64(&control->epoch) << 32u) | next_seq;
        *(volatile uint64_t *)(band + next_slot * slot_bytes + slot_bytes - 8u) = next_tag;
        SparkTpMeshShipGate(gate,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
        if ( SparkTpLdcvU64(&control->error_word) == 0u )
        {
            entry[1] = next_doorbell_bytes;
            entry[2] = next_slot;
            entry[3] = next_route;
            __threadfence_system();
            control->round_seq_prior = SparkTpLdcvU64(&control->round_seq) | SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG;
            control->round_seq = next_tag;
            control->seq = next_seq;
            control->slot_cursor = next_seq;
            entry[0] = next_tag;
            control->wait_started_ns = SparkTpGlobalTimerNs();
        }
    }
    __threadfence();
    LmDependentRelease();
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshMxfp8GatherStageKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkTpMeshRoundControl *control,uint32_t rank,const uint16_t *local,uint64_t local_elements,
    uint64_t doorbell_bytes,uint64_t route)
{
    __shared__ uint32_t last;
    LmDependentWait();
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs());
    const uint64_t prev_seq = SparkTpLdcvU64(&control->seq);
    const uint32_t failed = SparkTpLdcvU64(&control->error_word) != 0u || prev_seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = prev_seq + 1u;
    const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    if ( failed == 0u )
        SparkTpMeshWireQuantRange<BLOCK,SCALE_BYTES>(band + slot * slot_bytes,local_elements,local,local_elements,
            (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x ? 1u : 0u;
    __syncthreads();
    if ( last == 0u || threadIdx.x != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( SparkTpLdcvU64(&control->error_word) == 0u && prev_seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
    if ( failed == 0u )
    {
        const uint64_t tag = (SparkTpLdcvU64(&control->epoch) << 32u) | sequence;
        entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route;
        control->round_seq_prior = SparkTpLdcvU64(&control->round_seq) | SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG; control->round_seq = tag; control->seq = sequence;
        __threadfence_system();
        *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - SparkTpLdcvU64(&control->publish_started_ns);
    }
    control->publish_started_ns = 0u;
    control->wait_started_ns = SparkTpGlobalTimerNs();
    __threadfence();
    LmDependentRelease();
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshMxfp8GatherExpandKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    uint16_t *output,uint64_t begin,uint64_t total_count,
    uint64_t slice_elements,uint32_t is_slice_gather,uint32_t skip_self,uint32_t last)
{
    __shared__ uint32_t s_go;
    __shared__ volatile uint64_t s_peer_start;
    const uint32_t peer = blockIdx.y,stripe = blockIdx.x,tid = threadIdx.x;
    const uint64_t lane = (uint64_t)stripe * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    uint32_t ok = 0u;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t tag = SparkTpLdcvU64(&control->round_seq);
    const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
    const uint64_t start = is_slice_gather != 0u ? ((uint64_t)peer * slice_elements < total_count ? (uint64_t)peer * slice_elements : total_count) : 0u;
    const uint64_t length = is_slice_gather != 0u ? (total_count - start < slice_elements ? total_count - start : slice_elements) : slice_elements;
    const uint64_t dst_off = is_slice_gather != 0u ? (begin + start) : (begin + (uint64_t)peer * slice_elements);
    if ( peer == rank )
    {
        if ( skip_self == 0u && length != 0u )
            SparkTpMeshWireDequantRange<BLOCK,SCALE_BYTES>(output + dst_off,
                band + ((uint64_t)rank * slots_per_rank + ring) * slot_bytes,length,length,lane,lanes);
        if ( stripe == 0u )
        {
            __syncthreads();
            ok = SparkTpMeshArrivalWait(band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,
                prefetch,resume,0u,0u,arr_base,arr_end,arr_landed);
            if ( tid == 0u ) { control->math_started_ns = arr_end; __threadfence(); }
        }
    }
    else if ( length != 0u )
    {
        if ( tid == 0u )
        {
            const volatile uint64_t *tail = (const volatile uint64_t *)(band +
                ((uint64_t)peer * slots_per_rank + ring) * slot_bytes + slot_bytes - 8u);
            const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
            const uint64_t expected = control->cancel_expected;
            const uint32_t nap = length >= 65536u ? 50u : 0u;
            uint32_t go = 0u;
            s_peer_start = SparkTpFastTimerNs();
            for ( uint32_t spins = 0u;; )
            {
                if ( SparkTpLdAcquireU64(tail) == tag ) { go = 1u; break; }
                if ( (++spins & 31u) == 0u )
                {
                    const uint64_t now = SparkTpFastTimerNs();
                    if ( SparkTpLdcvU64(&control->error_word) != 0u ) break;
                    if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected ) break;
                    if ( now > s_peer_start && now - s_peer_start > timeout_ns ) break;
                }
                if ( nap != 0u ) __nanosleep(nap);
            }
            s_go = go;
        }
        __syncthreads();
        if ( s_go != 0u )
            SparkTpMeshWireDequantRange<BLOCK,SCALE_BYTES>(output + dst_off,
                band + ((uint64_t)peer * slots_per_rank + ring) * slot_bytes,length,length,lane,lanes);
    }
    __syncthreads();
    if ( tid == 0u &&
         atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y )
    {
        const uint64_t end_ns = SparkTpGlobalTimerNs();
        const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
        control->math_blocks_done = 0u;
        control->math_started_ns = 0u;
        control->math_finished_ns = 0u;
        if ( SparkTpLdcvU64(&control->error_word) == 0u )
        {
            if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
            control->slot_cursor = SparkTpLdcvU64(&control->seq);
            if ( last != 0u ) control->rounds_done++;
        }
        __threadfence();
        LmDependentRelease();
    }
    if ( peer == rank && stripe == 0u && tid < 32u && ok != 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshMxfp8ScatterCombineKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t degree,uint16_t *output,uint64_t begin,uint64_t count,uint32_t last)
{
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint64_t ring = (SparkTpLdcvU64(&control->round_seq) - 1u) & (slots_per_rank - 1u);
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lanes = (uint64_t)blockDim.x * gridDim.x;
    const uint64_t octs = count >> 4u;
    uint16_t *out16 = output + begin;
    for ( uint64_t i = lane; i < octs; i += lanes )
        SparkTpStU256(out16 + i * 16u,SparkTpMeshWireSumOctValue<BLOCK,SCALE_BYTES>(band,slot_bytes,slots_per_rank,ring,degree,count,i));
    SparkTpMeshHardwareCombineSettle(control,last,started);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshMxfp8PieceStageKernel(SparkTpMeshPiece piece)
{
    __shared__ uint32_t s_last;
    SparkTpMeshRoundControl *control = piece.control;
    const uint32_t tid = threadIdx.x;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    if ( tid == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs() | 1ull);
    const uint64_t prev_seq = SparkTpLdcvU64(&control->seq);
    const uint32_t failed = prev_seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t mask = piece.slots_per_rank - 1u,sequence = prev_seq + 1u;
    const uint64_t slot = (uint64_t)piece.rank * piece.slots_per_rank + ((sequence - 1u) & mask);
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
    if ( failed == 0u )
    {
        uint8_t *own = piece.band + slot * piece.slot_bytes + SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence));
        if ( piece.publish == SPARK_TP_MESH_PIECE_PUBLISH_GATHER )
            SparkTpMeshWireQuantRange<BLOCK,SCALE_BYTES>(own,piece.source_count,piece.source,piece.source_count,lane,lanes);
        else
        {
            uint8_t *dst = blockIdx.y == piece.rank ? own :
                ((uint8_t *)piece.staging + (uint64_t)blockIdx.y * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES +
                    SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(sequence - 1u));
            SparkTpMeshWireQuantRange<BLOCK,SCALE_BYTES>(dst,piece.source_count,
                piece.source + (uint64_t)blockIdx.y * piece.source_stride,piece.source_count,lane,lanes);
        }
        __threadfence_system();
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || tid != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( failed != 0u && SparkTpLdcvU64(&control->error_word) == 0u ) control->error_word = UINT64_MAX;
    if ( failed == 0u && SparkTpLdcvU64(&control->error_word) == 0u )
    {
        const uint64_t pub_start = SparkTpLdcvU64(&control->publish_started_ns);
        const uint64_t pub_end = SparkTpGlobalTimerNs();
        if ( pub_end > pub_start ) control->copy_ns += pub_end - pub_start;
        if ( piece.ring_direct != 0u )
        {
            const uint64_t tag = (SparkTpLdcvU64(&control->epoch) << 32u) | sequence;
            const uint32_t half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence);
            volatile uint64_t *cell = piece.entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(sequence);
            SparkWeightdMeshRoute routed;
            routed.word = piece.route;
            routed.fields.half = half;
            cell[1] = piece.doorbell_bytes;
            cell[2] = slot;
            cell[3] = routed.word;
            control->round_seq_prior = SparkTpLdcvU64(&control->round_seq) | SPARK_TP_MESH_FAST_PRIOR_CPU_FLAG;
            control->round_seq = tag;
            control->seq = sequence;
            *(volatile uint64_t *)(piece.band + slot * piece.slot_bytes + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half)) = tag;
            __threadfence_system();
            cell[0] = tag;
            control->wait_started_ns = SparkTpGlobalTimerNs();
        }
    }
    control->publish_started_ns = 0u;
    __threadfence();
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshMxfp8PieceCombineKernel(SparkTpMeshPiece piece)
{
    SparkTpMeshRoundControl *control = piece.control;
    const uint64_t mask = piece.slots_per_rank - 1u;
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,lanes = (uint64_t)blockDim.x * gridDim.x;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint64_t tag = SparkTpLdcvU64(&control->round_seq) - piece.lag,ring = (tag - 1u) & mask;
    const uint8_t *halves = piece.band + SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag));
    if ( piece.combine == SPARK_TP_MESH_PIECE_COMBINE_GATHER )
        SparkTpMeshWireDequantRange<BLOCK,SCALE_BYTES>(
            piece.output + (uint64_t)blockIdx.y * piece.output_stride + piece.begin,
            halves + ((uint64_t)blockIdx.y * piece.slots_per_rank + ring) * piece.slot_bytes,
            piece.count,piece.count,lane,lanes);
    else
    {
        uint16_t *out16 = piece.output + piece.begin;
        const uint64_t octs = piece.count >> 4u;
        for ( uint64_t i = lane; i < octs; i += lanes )
            SparkTpStU256(out16 + i * 16u,SparkTpMeshWireSumOctValue<BLOCK,SCALE_BYTES>(
                halves,piece.slot_bytes,piece.slots_per_rank,ring,piece.degree,piece.count,i));
    }
    SparkTpMeshHardwareCombineSettle(control,piece.last,started);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static SparkGpuError SparkTpMeshFastMxfp8PipelineRound(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint8_t *staging,uint64_t elements,uint32_t publish,uint32_t combine,uint64_t timeout_ns)
{
    const uint64_t width = SparkTpMeshWireHalfWidth<BLOCK,SCALE_BYTES>();
    const uint64_t chunks = (elements + width - 1u) / width;
    SparkWeightdMeshRoute route = {0};
    SparkTpMeshPiece piece = {0};
    if ( slot_bytes != SPARK_WEIGHTD_MESH_SLOT_BYTES || slots_per_rank != SPARK_WEIGHTD_MESH_SLOTS_PER_RANK || chunks < 3u ||
         (publish == SPARK_TP_MESH_PIECE_PUBLISH_SCATTER && staging == 0) )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    if ( publish == SPARK_TP_MESH_PIECE_PUBLISH_SCATTER )
        route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    piece.band = band;
    piece.entry = entry;
    piece.gate = request;
    piece.control = control;
    piece.staging = (uint16_t *)staging;
    piece.output = (uint16_t *)output;
    piece.slot_bytes = slot_bytes;
    piece.slots_per_rank = slots_per_rank;
    piece.source_stride = elements;
    piece.output_stride = elements;
    piece.timeout_ns = timeout_ns;
    piece.rank = rank;
    piece.degree = degree;
    for ( uint64_t launch = 0u; launch < chunks + SPARK_WEIGHTD_MESH_PIPELINE_DEPTH; ++launch )
    {
        const uint64_t taken = SparkTpMeshPipelineTaken(launch,SPARK_WEIGHTD_MESH_PIPELINE_DEPTH);
        const uint32_t has_publish = launch < chunks ? 1u : 0u;
        const uint32_t has_combine = taken != SPARK_TP_MESH_PIPELINE_NONE ? 1u : 0u;
        piece.publish = has_publish != 0u ? publish : 0u;
        piece.combine = has_combine != 0u ? combine : 0u;
        if ( has_publish != 0u )
        {
            const uint64_t begin = launch * width;
            const uint64_t count = elements - begin < width ? elements - begin : width;
            const uint64_t wire_b = SparkTpMeshWireBytes<BLOCK,SCALE_BYTES>(count);
            const dim3 stage_grid = publish == SPARK_TP_MESH_PIECE_PUBLISH_GATHER ?
                dim3(SparkTpMeshDirectBlocks(count,1u),1u) :
                dim3(SparkTpMeshPieceBlocks(count,degree),degree);
            const uint32_t ship_mode = SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD |
                (launch != 0u ? SPARK_TP_MESH_PUBLISH_SHIP_PRIOR : 0u);
            route.fields.slice_bytes = publish == SPARK_TP_MESH_PIECE_PUBLISH_SCATTER ? wire_b : 0u;
            piece.source = (const uint16_t *)local + begin;
            piece.source_count = count;
            piece.route = route.word;
            piece.doorbell_bytes = wire_b;
            piece.ship_prior = launch != 0u ? 1u : 0u;
            piece.ring_direct = has_combine == 0u ? 1u : 0u;
            LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1u,1u,0u,stream,request,control,ship_mode,timeout_ns);
            if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS ) return SparkGpuPeekAtLastError();
            SparkGpuError status = LM_LAUNCH_DEPENDENT((SparkTpMeshMxfp8PieceStageKernel<BLOCK,SCALE_BYTES>),stage_grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,piece);
            if ( status != SPARK_GPU_SUCCESS ) return status;
        }
        if ( has_combine != 0u )
        {
            const uint32_t arrive_lag = (uint32_t)((has_publish != 0u ? launch : chunks) - 1u - taken);
            piece.begin = taken * width;
            piece.count = elements - piece.begin < width ? elements - piece.begin : width;
            piece.lag = (uint32_t)((has_publish != 0u ? launch : chunks - 1u) - taken);
            piece.last = taken + 1u == chunks ? 1u : 0u;
            const dim3 combine_grid = combine == SPARK_TP_MESH_PIECE_COMBINE_GATHER ?
                dim3(SparkTpMeshPieceBlocks(piece.count,degree),degree) :
                dim3(SparkTpMeshDirectBlocks(piece.count,1u),1u);
            LM_LAUNCH((SparkTpMeshHardwarePieceArriveRingKernel),1u,32u,0u,stream,band,slot_bytes,slots_per_rank,
                entry,control,rank,degree,timeout_ns,arrive_lag,has_publish,piece.doorbell_bytes,piece.route);
            if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS ) return SparkGpuPeekAtLastError();
            SparkGpuError status = LM_LAUNCH_DEPENDENT((SparkTpMeshMxfp8PieceCombineKernel<BLOCK,SCALE_BYTES>),combine_grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,piece);
            if ( status != SPARK_GPU_SUCCESS ) return status;
        }
    }
    route.fields.slice_bytes = 0u;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    LM_LAUNCH((SparkTpMeshHardwarePublishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,rank,local,
        SPARK_TP_MESH_PIPELINE_MARKER_BYTES,0u,SPARK_TP_MESH_PIPELINE_MARKER_BYTES,route.word,1u,1u,request,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_HALF,route.fields.peer_mask,timeout_ns);
    if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS ) return SparkGpuPeekAtLastError();
    return SparkTpMeshHardwareArrivalView(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,SPARK_TP_MESH_VIEW_HALF);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static SparkGpuError SparkTpMeshFastPeerRsagMxfp8Chunk(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint8_t *staging,uint64_t local_elements,uint64_t begin,uint64_t count,
    uint64_t timeout_ns,uint32_t phase1_quant)
{
    SparkWeightdMeshRoute route = {0};
    const uint64_t slice = ((count + degree - 1u) / degree + 31u) & ~UINT64_C(31);
    const uint64_t first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count;
    const uint64_t span = count - first < slice ? count - first : slice;
    const uint64_t phase1_bytes = phase1_quant != 0u ? SparkTpMeshWireBytes<BLOCK,SCALE_BYTES>(slice) : (slice * sizeof(uint16_t));
    SparkGpuError status = SPARK_GPU_SUCCESS;
    if ( staging == 0 || phase1_bytes > SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES ||
         phase1_bytes > slot_bytes - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = phase1_bytes;
    LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1u,1u,0u,stream,request,control,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
    status = SparkGpuPeekAtLastError();
    if ( status != SPARK_GPU_SUCCESS ) return status;
    if ( phase1_quant != 0u )
        status = LM_LAUNCH_DEPENDENT((SparkTpMeshRsagMxfp8StageKernel<BLOCK,SCALE_BYTES>),
            dim3(SparkTpMeshPieceBlocks(slice,degree),degree),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,entry,control,staging,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES,
            (const uint16_t *)local + begin,rank,count,slice,0u,slice,phase1_bytes,route.word,0u);
    else
        status = LM_LAUNCH_DEPENDENT((SparkTpMeshRsagStageKernel),
            dim3(SparkTpMeshCopyBlocks(slice,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,entry,control,(uint16_t *)staging,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t),
            (const uint16_t *)local + begin,rank,count,slice,phase1_bytes,route.word);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    route.fields.slice_bytes = 0u;
    const uint64_t next_bytes = span != 0u ? SparkTpMeshWireBytes<BLOCK,SCALE_BYTES>(span) : 8u;
    const uint32_t blocks = SparkTpMeshDirectBlocks(span != 0u ? span : 1u,1u);
    if ( phase1_quant != 0u )
        status = LM_LAUNCH_DEPENDENT((SparkTpMeshRsagMxfp8CombineKernel<BLOCK,SCALE_BYTES,1u>),
            blocks,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,request,control,
            rank,degree,timeout_ns,(uint16_t *)output,begin + first,slice,span,next_bytes,route.word);
    else
        status = LM_LAUNCH_DEPENDENT((SparkTpMeshRsagMxfp8CombineKernel<BLOCK,SCALE_BYTES,0u>),
            blocks,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,request,control,
            rank,degree,timeout_ns,(uint16_t *)output,begin + first,slice,span,next_bytes,route.word);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const dim3 grid(SparkTpMeshFastGatherStripes(slice,degree),degree);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshMxfp8GatherExpandKernel<BLOCK,SCALE_BYTES>),grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,
        (uint16_t *)output,begin,count,slice,1u,1u,begin + count == local_elements ? 1u : 0u);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static SparkGpuError SparkTpMeshFastGatherMxfp8Round(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint64_t elements,uint64_t timeout_ns)
{
    const uint64_t local_elements = elements / degree;
    const uint64_t single_cap = 2u * SparkTpMeshWireHalfWidth<BLOCK,SCALE_BYTES>();
    if ( local_elements > single_cap )
        return SparkTpMeshFastMxfp8PipelineRound<BLOCK,SCALE_BYTES>(stream,band,slot_bytes,slots_per_rank,entry,request,control,
            rank,degree,local,output,0,local_elements,SPARK_TP_MESH_PIECE_PUBLISH_GATHER,SPARK_TP_MESH_PIECE_COMBINE_GATHER,timeout_ns);
    const uint64_t wire_b = SparkTpMeshWireBytes<BLOCK,SCALE_BYTES>(local_elements);
    SparkWeightdMeshRoute route = {0};
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1u,1u,0u,stream,request,control,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
    SparkGpuError status = SparkGpuPeekAtLastError();
    if ( status != SPARK_GPU_SUCCESS ) return status;
    status = LM_LAUNCH_DEPENDENT((SparkTpMeshMxfp8GatherStageKernel<BLOCK,SCALE_BYTES>),
        SparkTpMeshDirectBlocks(local_elements,1u),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,control,rank,(const uint16_t *)local,local_elements,wire_b,route.word);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const dim3 grid(SparkTpMeshFastGatherStripes(local_elements,degree),degree);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshMxfp8GatherExpandKernel<BLOCK,SCALE_BYTES>),grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,
        (uint16_t *)output,0u,elements,local_elements,0u,0u,1u);
}

template<uint32_t BLOCK,uint32_t SCALE_BYTES>
static SparkGpuError SparkTpMeshFastScatterMxfp8Round(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint8_t *staging,uint64_t elements,uint64_t timeout_ns)
{
    const uint64_t segment = elements / degree;
    const uint64_t single_cap = 2u * SparkTpMeshWireHalfWidth<BLOCK,SCALE_BYTES>();
    if ( segment > single_cap )
        return SparkTpMeshFastMxfp8PipelineRound<BLOCK,SCALE_BYTES>(stream,band,slot_bytes,slots_per_rank,entry,request,control,
            rank,degree,local,output,staging,segment,SPARK_TP_MESH_PIECE_PUBLISH_SCATTER,SPARK_TP_MESH_PIECE_COMBINE_SUM,timeout_ns);
    const uint64_t wire_b = SparkTpMeshWireBytes<BLOCK,SCALE_BYTES>(segment);
    SparkWeightdMeshRoute route = {0};
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = wire_b;
    LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1u,1u,0u,stream,request,control,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
    SparkGpuError status = SparkGpuPeekAtLastError();
    if ( status != SPARK_GPU_SUCCESS ) return status;
    status = LM_LAUNCH_DEPENDENT((SparkTpMeshRsagMxfp8StageKernel<BLOCK,SCALE_BYTES>),
        dim3(SparkTpMeshPieceBlocks(segment,degree),degree),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,control,staging,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES,
        (const uint16_t *)local,rank,elements,segment,segment,segment,wire_b,route.word,1u);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    return LM_LAUNCH_DEPENDENT((SparkTpMeshMxfp8ScatterCombineKernel<BLOCK,SCALE_BYTES>),
        SparkTpMeshDirectBlocks(segment,1u),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,control,degree,(uint16_t *)output,0u,segment,1u);
}

static SparkGpuError SparkTpMeshFastDispatchMxfp8Rsag(SparkGpuStream stream,
    uint32_t codec,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint8_t *staging,uint64_t elements,uint64_t timeout_ns)
{
    const uint32_t p1q = codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16 ? 1u : 0u;
    const uint32_t kind = p1q != 0u ? codec : (codec - 3u);
    const uint64_t cap = SparkTpMeshDirectPeerCapacity(degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
    SparkGpuError status = SPARK_GPU_SUCCESS;
    for ( uint64_t begin = 0u; begin < elements && status == SPARK_GPU_SUCCESS; )
    {
        const uint64_t count = elements - begin < cap ? elements - begin : cap;
        if ( kind == SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32 )
            status = SparkTpMeshFastPeerRsagMxfp8Chunk<32u,1u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,elements,begin,count,timeout_ns,p1q);
        else if ( kind == SPARK_TP_MESH_WIRE_CODEC_FP8_B32 )
            status = SparkTpMeshFastPeerRsagMxfp8Chunk<32u,2u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,elements,begin,count,timeout_ns,p1q);
        else
            status = SparkTpMeshFastPeerRsagMxfp8Chunk<16u,2u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,elements,begin,count,timeout_ns,p1q);
        begin += count;
    }
    return status;
}

static SparkGpuError SparkTpMeshFastDispatchMxfp8Gather(SparkGpuStream stream,
    uint32_t codec,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint64_t elements,uint64_t timeout_ns)
{
    const uint32_t kind = codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16 ? codec : (codec - 3u);
    if ( kind == SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32 )
        return SparkTpMeshFastGatherMxfp8Round<32u,1u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,elements,timeout_ns);
    if ( kind == SPARK_TP_MESH_WIRE_CODEC_FP8_B32 )
        return SparkTpMeshFastGatherMxfp8Round<32u,2u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,elements,timeout_ns);
    return SparkTpMeshFastGatherMxfp8Round<16u,2u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,elements,timeout_ns);
}

static SparkGpuError SparkTpMeshFastDispatchMxfp8Scatter(SparkGpuStream stream,
    uint32_t codec,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint8_t *staging,uint64_t elements,uint64_t timeout_ns)
{
    if ( codec == SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32 )
        return SparkTpMeshFastScatterMxfp8Round<32u,1u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,elements,timeout_ns);
    if ( codec == SPARK_TP_MESH_WIRE_CODEC_FP8_B32 )
        return SparkTpMeshFastScatterMxfp8Round<32u,2u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,elements,timeout_ns);
    return SparkTpMeshFastScatterMxfp8Round<16u,2u>(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,elements,timeout_ns);
}

extern "C" SparkGpuError SparkTpMeshHardwarePrepareFast(void *host,void **device_out)
{
    const void *const kernels[] = {
        (const void *)SparkTpMeshFastPublishKernel,
        (const void *)SparkTpMeshFastPublishWideKernel,
        (const void *)SparkTpMeshFastPairGatherKernel,
        (const void *)SparkTpMeshFastPeerStageSmallKernel,
        (const void *)SparkTpMeshFastPeerStageWideKernel,
        (const void *)SparkTpMeshFastStreamAllReduceKernel<4u,0u>,
        (const void *)SparkTpMeshFastStreamAllReduceKernel<4u,1u>,
        (const void *)SparkTpMeshFastStreamAllReduceKernel<16u,0u>,
        (const void *)SparkTpMeshFastRsagStreamReducePublishKernel<4u,0u>,
        (const void *)SparkTpMeshFastRsagStreamReducePublishKernel<4u,1u>,
        (const void *)SparkTpMeshFastRsagStreamReducePublishKernel<16u,0u>,
        (const void *)SparkTpMeshFastRsagWideReducePublishKernel,
        (const void *)SparkTpMeshFastRsagFusedKernel,
        (const void *)SparkTpMeshFastGatherFusedKernel,
        (const void *)SparkTpMeshFast2dStageKernel,
        (const void *)SparkTpMeshFast2dReducePublishKernel,
        (const void *)SparkTpMeshFast2dGatherKernel,
        (const void *)SparkTpMeshFastWideStreamGatherKernel,
        (const void *)SparkTpMeshRsagMxfp8StageKernel<32u,1u>,
        (const void *)SparkTpMeshRsagMxfp8StageKernel<32u,2u>,
        (const void *)SparkTpMeshRsagMxfp8StageKernel<16u,2u>,
        (const void *)SparkTpMeshRsagMxfp8CombineKernel<32u,1u,1u>,
        (const void *)SparkTpMeshRsagMxfp8CombineKernel<32u,2u,1u>,
        (const void *)SparkTpMeshRsagMxfp8CombineKernel<16u,2u,1u>,
        (const void *)SparkTpMeshRsagMxfp8CombineKernel<32u,1u,0u>,
        (const void *)SparkTpMeshRsagMxfp8CombineKernel<32u,2u,0u>,
        (const void *)SparkTpMeshRsagMxfp8CombineKernel<16u,2u,0u>,
        (const void *)SparkTpMeshMxfp8GatherStageKernel<32u,1u>,
        (const void *)SparkTpMeshMxfp8GatherStageKernel<32u,2u>,
        (const void *)SparkTpMeshMxfp8GatherStageKernel<16u,2u>,
        (const void *)SparkTpMeshMxfp8GatherExpandKernel<32u,1u>,
        (const void *)SparkTpMeshMxfp8GatherExpandKernel<32u,2u>,
        (const void *)SparkTpMeshMxfp8GatherExpandKernel<16u,2u>,
        (const void *)SparkTpMeshMxfp8ScatterCombineKernel<32u,1u>,
        (const void *)SparkTpMeshMxfp8ScatterCombineKernel<32u,2u>,
        (const void *)SparkTpMeshMxfp8ScatterCombineKernel<16u,2u>,
        (const void *)SparkTpMeshMxfp8PieceStageKernel<32u,1u>,
        (const void *)SparkTpMeshMxfp8PieceStageKernel<32u,2u>,
        (const void *)SparkTpMeshMxfp8PieceStageKernel<16u,2u>,
        (const void *)SparkTpMeshMxfp8PieceCombineKernel<32u,1u>,
        (const void *)SparkTpMeshMxfp8PieceCombineKernel<32u,2u>,
        (const void *)SparkTpMeshMxfp8PieceCombineKernel<16u,2u>
    };
    SparkGpuFunctionAttributes attributes;
    for ( uint32_t i = 0u; i < sizeof(kernels) / sizeof(kernels[0]); i++ )
    {
        SparkGpuError status = SparkGpuFuncGetAttributes(&attributes,kernels[i]);
        if ( status != SPARK_GPU_SUCCESS ) return status;
    }
    SparkGpuError prep = SparkTpMeshHardwarePrepare(host,device_out);
    if ( prep == SPARK_GPU_SUCCESS && device_out != 0 )
        SparkTpMeshFastHybridRecordRegion(host,*device_out);
    return prep;
}

extern "C" SparkGpuError SparkTpLaunchMeshHardwareFast(SparkGpuStream stream,
    void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    void *gate,void *round_control,uint32_t rank,uint32_t degree,
    const void *local,void *output,void *scratch,uint64_t elements,
    uint32_t operation,uint32_t rounds,uint32_t logical_rows,uint32_t slice_routes,void *staging,uint64_t timeout_ns,uint32_t stage)
{
    if ( band == 0 || entry == 0 || gate == 0 || round_control == 0 || local == 0 ||
         output == 0 || scratch == 0 || elements == 0u || rounds == 0u || logical_rows == 0u ||
         degree < 2u || degree > 16u || rank >= degree || operation > SPARK_TP_MESH_OPERATION_REDUCE_SCATTER ||
         slot_bytes <= 16u || slots_per_rank != SPARK_WEIGHTD_MESH_SLOTS_PER_RANK || timeout_ns == 0u ||
         (operation == 0u && (elements % degree != 0u || local == output)) ||
         (operation == SPARK_TP_MESH_OPERATION_ALL_TO_ALL && (local == output || slice_routes == 0u)) ||
         (operation == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER && (local == output || (slice_routes & SPARK_TP_MESH_ROUTES_PEER) == 0u ||
            elements % (4u * degree) != 0u || stage != SPARK_TP_MESH_STAGE_ALL)) ||
         ((slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging == 0) )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    if ( stage > SPARK_TP_MESH_STAGE_COMPLETE )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    if ( stage == SPARK_TP_MESH_STAGE_COMPLETE &&
         SparkTpMeshDirectStaged(slot_bytes,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES,degree,elements,operation,rounds,logical_rows,slice_routes) == 0u )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    if ( rounds != 1u || logical_rows != 1u )
        return SparkTpLaunchMeshHardware(stream,band,slot_bytes,slots_per_rank,entry,gate,round_control,
            rank,degree,local,output,scratch,elements,operation,rounds,logical_rows,slice_routes,staging,timeout_ns,stage);
    SparkTpMeshRoundControl *control = (SparkTpMeshRoundControl *)round_control;
    SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)gate;
    volatile uint64_t *mbox = (volatile uint64_t *)scratch;
    const uint32_t sm_ok = SparkTpMeshMultiprocessors() >= degree ? 1u : 0u;
    const uint32_t wire_codec = stage == SPARK_TP_MESH_STAGE_ALL && SparkTpMeshFastWireCodecAllowed() != 0u ? SparkTpMeshFastWireCodec() : SPARK_TP_MESH_WIRE_CODEC_BF16;
    const uint32_t wire_aligned = wire_codec != SPARK_TP_MESH_WIRE_CODEC_BF16 && elements >= SPARK_TP_MESH_WIRE_MIN_ELEMENTS &&
        (((uintptr_t)local | (uintptr_t)output) & 31u) == 0u && (((uintptr_t)band & 15u) == 0u) && (slot_bytes & 15u) == 0u ? 1u : 0u;
    if ( stage == SPARK_TP_MESH_STAGE_ALL && operation == SPARK_TP_MESH_OPERATION_ALL_GATHER &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PAIR) != 0u && (slice_routes & SPARK_TP_MESH_ROUTES_PAIR_FIRST) == 0u )
    {
        const uint64_t local_elements = elements / degree;
        if ( local_elements <= SparkTpMeshDirectCapacity(slot_bytes,0u) )
        {
            const uint32_t pair_only = (slice_routes & SPARK_TP_MESH_ROUTES_PAIR_ONLY) != 0u && (degree & 1u) == 0u ? 1u : 0u;
            SparkWeightdMeshRoute route = {0};
            uint32_t resume = 0u;
            LmWeightPrefetchGroup *prefetch;
            SparkGpuError status;
            route.fields.peer_mask = pair_only != 0u ? (1u << (rank ^ 1u)) : (((1u << degree) - 1u) & ~(1u << rank));
            route.fields.pair = pair_only != 0u ? 0u : 1u;
            if ( pair_only != 0u && local_elements * sizeof(uint16_t) <= 65536u )
                return SparkTpMeshFastLaunchGatherFused(stream,(uint8_t *)band,slot_bytes,slots_per_rank,
                    (volatile uint64_t *)entry,request,control,0,0u,rank,degree,(const uint16_t *)local,(uint16_t *)output,
                    local_elements,route.word,timeout_ns,1u);
            prefetch = LmWeightPrefetchSlot(stream,&resume);
            status = SparkTpMeshFastPublish(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,rank,degree,
                local,local_elements * sizeof(uint16_t),route.word,request,route.fields.peer_mask,timeout_ns,0u);
            if ( status != SPARK_GPU_SUCCESS ) return status;
            return LM_LAUNCH_DEPENDENT((SparkTpMeshFastPairGatherKernel),1u,SPARK_TP_MESH_THREADS,0u,stream,
                (const uint8_t *)band,slot_bytes,slots_per_rank,request,control,rank,degree,timeout_ns,prefetch,resume,
                (const uint16_t *)local,(uint16_t *)output,local_elements,pair_only);
        }
    }
    if ( stage == SPARK_TP_MESH_STAGE_ALL && operation == 2u &&
         elements <= SPARK_TP_MESH_FAST_MAX_U64_FUSED_MAX_ELEMENTS &&
         elements * sizeof(uint64_t) <= slot_bytes - 16u )
    {
        SparkWeightdMeshRoute route = {0};
        route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
        return SparkTpMeshFastLaunchGatherFused(stream,(uint8_t *)band,slot_bytes,slots_per_rank,
            (volatile uint64_t *)entry,request,control,0,0u,rank,degree,(const uint16_t *)local,(uint16_t *)output,
            elements,route.word,timeout_ns,3u);
    }
    if ( stage == SPARK_TP_MESH_STAGE_ALL && operation == SPARK_TP_MESH_OPERATION_ALL_TO_ALL &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging != 0 && sm_ok != 0u )
    {
        const uint64_t slice = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
        if ( elements <= slice &&
             slice * sizeof(uint16_t) <= slot_bytes - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES &&
             SparkTpMeshAllToAllPipelined(elements,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) == 0u )
        {
            SparkWeightdMeshRoute route = {0};
            SparkGpuError status;
            const uint64_t slice_bytes = (elements * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
            route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
            route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
            route.fields.slice_bytes = slice_bytes;
            if ( elements <= SPARK_TP_MESH_FAST_A2A_FUSED_MAX_ELEMENTS )
                return SparkTpMeshFastLaunchGatherFused(stream,(uint8_t *)band,slot_bytes,slots_per_rank,
                    (volatile uint64_t *)entry,request,control,(uint16_t *)staging,slice,rank,degree,
                    (const uint16_t *)local,(uint16_t *)output,elements,route.word,timeout_ns,2u);
            status = SparkTpMeshFastLaunchPeerStage(stream,(uint8_t *)band,slot_bytes,slots_per_rank,
                (volatile uint64_t *)entry,request,control,(uint16_t *)staging,slice,
                (const uint16_t *)local,mbox,degree,rank,elements * degree,elements,elements,
                slice_bytes,route.word,timeout_ns,0u,0u,1u);
            if ( status != SPARK_GPU_SUCCESS ) return status;
            return SparkTpMeshFastLaunchWideStreamGather(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,request,control,
                rank,degree,(const uint16_t *)local + (uint64_t)rank * elements,(uint16_t *)output,mbox,
                0u,elements * degree,elements,0u,0u,1u,timeout_ns);
        }
    }
    if ( operation == 1u && degree == 16u && stage == SPARK_TP_MESH_STAGE_ALL &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging != 0 && sm_ok != 0u &&
         elements >= 1024u && elements < SPARK_TP_MESH_RSAG_MIN_ELEMENTS_WIDE && (elements & 127u) == 0u &&
         SparkTpMeshDirectPhasesPerChunk(elements,degree,1u,slice_routes) == 1u )
    {
        SparkWeightdMeshHybridCtrl *hyb_ctrl = 0;
        uint8_t *hyb_shm = 0;
        uint32_t hyb_band = 0u;
        SparkTpMeshFastHybridResolve(stream,(const void *)entry,gate,slot_bytes,&hyb_ctrl,&hyb_shm,&hyb_band);
        if ( hyb_ctrl != 0 && hyb_shm != 0 )
            return SparkTpMeshFastPeerRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,mbox,(uint8_t *)staging,elements,0u,elements,timeout_ns,stage);
    }
    if ( operation == 1u &&
         SparkTpMeshDirectPhasesPerChunk(elements,degree,1u,slice_routes) == 1u &&
         elements <= SPARK_TP_MESH_FAST_STREAM_REDUCE_MAX_ELEMENTS &&
         (elements & 7u) == 0u &&
         elements <= SparkTpMeshDirectCapacity(slot_bytes,1u) &&
         (((uintptr_t)band | (uintptr_t)output | (uintptr_t)local) & 15u) == 0u &&
         (slot_bytes & 15u) == 0u )
    {
        uint32_t resume_probe = 0u;
        const uint32_t has_prefetch = stage != SPARK_TP_MESH_STAGE_ALL ? 1u : (LmWeightPrefetchSlot(stream,&resume_probe) != 0 ? 1u : 0u);
        const uint32_t wide256 = degree <= 4u && (elements & 15u) == 0u &&
            (((uintptr_t)band | (uintptr_t)output) & 31u) == 0u && (slot_bytes & 31u) == 0u &&
            (stage != SPARK_TP_MESH_STAGE_ALL || (((uintptr_t)local & 31u) == 0u)) ? 1u : 0u;
        if ( SparkTpMeshFastCanStreamReduce(elements,degree,wide256,has_prefetch) != 0u )
        {
            if ( stage != SPARK_TP_MESH_STAGE_COMPLETE )
            {
                SparkWeightdMeshRoute route = {0};
                SparkGpuError status;
                route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
                status = SparkTpMeshFastPublish(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,rank,degree,
                    local,elements * sizeof(uint16_t),route.word,request,route.fields.peer_mask,timeout_ns,
                    stage == SPARK_TP_MESH_STAGE_ALL ? 0u : 1u);
                if ( status != SPARK_GPU_SUCCESS || stage == SPARK_TP_MESH_STAGE_PUBLISH ) return status;
            }
            return SparkTpMeshFastLaunchStreamReduce(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,request,control,
                rank,degree,stage == SPARK_TP_MESH_STAGE_ALL ? (const uint16_t *)local : 0,(uint16_t *)output,mbox,elements,elements * sizeof(uint16_t),timeout_ns);
        }
    }
    if ( operation == 1u &&
         SparkTpMeshDirectPhasesPerChunk(elements,degree,1u,slice_routes) == 2u &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging != 0 )
    {
        if ( wire_aligned != 0u && (elements & 31u) == 0u && (((uintptr_t)staging & 15u) == 0u) )
            return SparkTpMeshFastDispatchMxfp8Rsag(stream,wire_codec,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,timeout_ns);
        if ( sm_ok != 0u )
        {
            if ( SparkTpMeshFastTwoDAllowed() != 0u && degree == 16u && stage == SPARK_TP_MESH_STAGE_ALL &&
                 elements >= SPARK_TP_MESH_FAST_2D_MIN_ELEMENTS &&
                 (((uintptr_t)local | (uintptr_t)output | (uintptr_t)band | (uintptr_t)staging) & 15u) == 0u &&
                 (slot_bytes & 15u) == 0u )
            {
                const uint64_t cap2d = SPARK_TP_MESH_FAST_2D_CAP_ELEMENTS;
                SparkGpuError status = SPARK_GPU_SUCCESS;
                for ( uint64_t begin = 0u; begin < elements && status == SPARK_GPU_SUCCESS; )
                {
                    const uint64_t count = elements - begin < cap2d ? elements - begin : cap2d;
                    status = SparkTpMeshFast2dRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                        request,control,rank,(const uint8_t *)local,output,mbox,(uint8_t *)staging,elements,begin,count,timeout_ns);
                    begin += count;
                }
                return status;
            }
            const uint64_t capacity = SparkTpMeshDirectPeerCapacity(degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
            if ( stage != SPARK_TP_MESH_STAGE_ALL )
            {
                if ( elements <= capacity )
                    return SparkTpMeshFastPeerRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                        request,control,rank,degree,(const uint8_t *)local,output,mbox,(uint8_t *)staging,elements,0u,elements,timeout_ns,stage);
            }
            else
            {
                SparkGpuError status = SPARK_GPU_SUCCESS;
                for ( uint64_t begin = 0u; begin < elements && status == SPARK_GPU_SUCCESS; )
                {
                    const uint64_t count = elements - begin < capacity ? elements - begin : capacity;
                    status = SparkTpMeshFastPeerRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                        request,control,rank,degree,(const uint8_t *)local,output,mbox,(uint8_t *)staging,elements,begin,count,timeout_ns,stage);
                    begin += count;
                }
                return status;
            }
        }
    }
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER &&
         (slice_routes & (SPARK_TP_MESH_ROUTES_PAIR | SPARK_TP_MESH_ROUTES_PAIR_FIRST)) == 0u )
    {
        const uint64_t local_elements = elements / degree;
        if ( wire_aligned != 0u && (local_elements & 31u) == 0u &&
             slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES )
            return SparkTpMeshFastDispatchMxfp8Gather(stream,wire_codec,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,elements,timeout_ns);
        if ( sm_ok != 0u &&
             SparkTpMeshDirectPipelined(elements,degree,0u,slot_bytes) == 0u &&
             local_elements <= SparkTpMeshDirectCapacity(slot_bytes,0u) )
        {
            if ( stage == SPARK_TP_MESH_STAGE_ALL && local_elements <= SPARK_TP_MESH_FAST_GATHER_FUSED_MAX_LOCAL )
            {
                SparkWeightdMeshRoute route = {0};
                route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
                return SparkTpMeshFastLaunchGatherFused(stream,(uint8_t *)band,slot_bytes,slots_per_rank,
                    (volatile uint64_t *)entry,request,control,0,0u,rank,degree,(const uint16_t *)local,(uint16_t *)output,
                    local_elements,route.word,timeout_ns,0u);
            }
            if ( stage != SPARK_TP_MESH_STAGE_COMPLETE )
            {
                SparkWeightdMeshRoute route = {0};
                SparkGpuError status;
                route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
                status = SparkTpMeshFastPublish(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,rank,degree,
                    local,local_elements * sizeof(uint16_t),route.word,request,route.fields.peer_mask,timeout_ns,
                    stage == SPARK_TP_MESH_STAGE_ALL ? 0u : 1u);
                if ( status != SPARK_GPU_SUCCESS || stage == SPARK_TP_MESH_STAGE_PUBLISH ) return status;
            }
            return SparkTpMeshFastLaunchWideStreamGather(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,request,control,
                rank,degree,stage == SPARK_TP_MESH_STAGE_ALL ? (const uint16_t *)local : 0,(uint16_t *)output,mbox,
                0u,elements,local_elements,0u,0u,1u,timeout_ns);
        }
    }
    if ( operation == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PAIR_FIRST) == 0u &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging != 0 )
    {
        const uint64_t segment = elements / degree;
        if ( wire_aligned != 0u && wire_codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16 &&
             (((uintptr_t)staging & 15u) == 0u) && slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES &&
             (segment & 31u) == 0u )
            return SparkTpMeshFastDispatchMxfp8Scatter(stream,wire_codec,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,timeout_ns);
        const uint64_t staging_slot_elements = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
        const uint16_t *own_local = (const uint16_t *)local + (uint64_t)rank * segment;
        if ( segment <= SPARK_TP_MESH_FAST_STREAM_REDUCE_MAX_ELEMENTS &&
             (segment & 7u) == 0u &&
             segment <= staging_slot_elements &&
             segment * sizeof(uint16_t) <= slot_bytes - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES &&
             SparkTpMeshScatterPipelined(elements,degree,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) == 0u &&
             (((uintptr_t)band | (uintptr_t)output | (uintptr_t)own_local) & 15u) == 0u &&
             (slot_bytes & 15u) == 0u )
        {
            uint32_t resume_probe = 0u;
            const uint32_t has_prefetch = LmWeightPrefetchSlot(stream,&resume_probe) != 0 ? 1u : 0u;
            const uint32_t wide256 = degree <= 4u && (segment & 15u) == 0u &&
                (((uintptr_t)band | (uintptr_t)output | (uintptr_t)own_local) & 31u) == 0u &&
                (slot_bytes & 31u) == 0u ? 1u : 0u;
            if ( SparkTpMeshFastCanStreamReduce(segment,degree,wide256,has_prefetch) != 0u )
            {
                SparkWeightdMeshRoute route = {0};
                SparkGpuError status;
                const uint64_t slice_bytes = (segment * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
                const uint64_t compact_bytes = slice_bytes + 8u <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES ? slice_bytes : 0u;
                route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
                route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
                route.fields.slice_bytes = slice_bytes;
                status = SparkTpMeshFastLaunchPeerStage(stream,(uint8_t *)band,slot_bytes,slots_per_rank,
                    (volatile uint64_t *)entry,request,control,(uint16_t *)staging,staging_slot_elements,
                    (const uint16_t *)local,mbox,degree,rank,elements,segment,segment,
                    slice_bytes,route.word,timeout_ns,2u,0u,1u);
                if ( status != SPARK_GPU_SUCCESS ) return status;
                return SparkTpMeshFastLaunchStreamReduce(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,request,control,
                    rank,degree,own_local,(uint16_t *)output,mbox,segment,compact_bytes,timeout_ns);
            }
        }
    }
    return SparkTpLaunchMeshHardware(stream,band,slot_bytes,slots_per_rank,entry,gate,round_control,
        rank,degree,local,output,scratch,elements,operation,rounds,logical_rows,slice_routes,staging,timeout_ns,stage);
}

static const SparkTpMeshLaunchers SparkTpMeshFastLaunchers =
{
    SPARK_TP_MESH_LAUNCHERS_ABI_VERSION,
    (uint32_t)sizeof(SparkTpMeshLaunchers),
    reinterpret_cast<SparkTpMeshCopyDownFunction>(SparkTpLaunchMeshCopyDown),
    reinterpret_cast<SparkTpMeshPublishFunction>(SparkTpLaunchMeshPublish),
    reinterpret_cast<SparkTpMeshTreeFunction>(SparkTpLaunchMeshTree),
    reinterpret_cast<SparkTpMeshHardwarePrepareFunction>(SparkTpMeshHardwarePrepareFast),
    reinterpret_cast<SparkTpMeshHardwareFunction>(SparkTpLaunchMeshHardwareFast),
    reinterpret_cast<SparkTpMeshSeqPadFunction>(SparkTpLaunchMeshSeqPad),
    reinterpret_cast<SparkTpMeshGuardFunction>(SparkTpLaunchMeshGuard),
    reinterpret_cast<SparkTpMeshWaitFunction>(SparkTpLaunchMeshWait),
    reinterpret_cast<SparkTpMeshRoundLoopFunction>(SparkTpLaunchMeshRoundLoop)
};

static inline int SparkTpMeshFastDisabled(void)
{
    const char *val = getenv("SPARK_MESH_FAST_DISABLE");
    return val != 0 && val[0] != '\0' && strcmp(val,"0") != 0 ? 1 : 0;
}

static inline void SparkTpMeshFastLaunchersInstall(void)
{
    if ( SparkTpMeshFastDisabled() != 0 )
        return;
    if ( SparkTpDeviceCollectiveRegisterLaunchers != 0 )
        SparkTpDeviceCollectiveRegisterLaunchers(&SparkTpMeshFastLaunchers);
}

__attribute__((constructor)) static void SparkTpMeshFastLaunchersInit(void)
{
    SparkTpMeshFastLaunchersInstall();
}

#endif
