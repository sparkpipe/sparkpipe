#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include "inference/runner/mesh_fast_kernels.cuh"

static uint32_t g_checks = 0u;
static uint32_t g_failures = 0u;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if ( !(cond) ) { \
        g_failures++; \
        fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, msg); \
    } \
} while (0)

#define CUDA_OK(expr) do { \
    cudaError_t _e = (expr); \
    if ( _e != cudaSuccess ) { \
        fprintf(stderr, "CUDA-FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

static uint64_t NowNs(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint16_t PatternBf16(uint32_t peer, uint32_t idx, uint32_t salt)
{
    uint32_t h = (peer + 1u) * 2654435761u ^ (idx + 7u) * 2246822519u ^ (salt * 3266489917u);
    int32_t sign = (h & 1u) ? -1 : 1;
    float v = (float)sign * (0.015625f + (float)((h >> 8u) & 255u) * 0.00390625f);
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(bits >> 16u);
}

typedef struct ResponderState
{
    uint8_t *band;
    volatile uint64_t *doorbell;
    volatile uint64_t *shipped;
    volatile uint64_t *cancel;
    uint16_t *peer_payloads[16u];
    uint64_t payload_bytes;
    uint32_t rank;
    uint32_t degree;
    uint32_t mode;
    uint32_t order[16u];
    uint32_t delay_ns[16u];
    volatile uint32_t stop;
    volatile uint32_t reset_last;
    volatile uint64_t last_seen_tag;
    volatile uint64_t last_route_word;
    volatile uint64_t last_doorbell_bytes;
    volatile uint32_t cancel_after_ns;
    volatile uint32_t drop_peer;
} ResponderState;

static void *ResponderMain(void *arg)
{
    ResponderState *st = (ResponderState *)arg;
    uint64_t seen = 0u;
    while ( __atomic_load_n(&st->stop, __ATOMIC_ACQUIRE) == 0u )
    {
        if ( __atomic_load_n(&st->reset_last, __ATOMIC_ACQUIRE) != 0u )
        {
            seen = 0u;
            __atomic_store_n(&st->reset_last, 0u, __ATOMIC_RELEASE);
        }
        uint64_t tag = __atomic_load_n(&st->doorbell[0], __ATOMIC_ACQUIRE);
        if ( tag == 0u || tag == seen )
            continue;
        seen = tag;
        uint64_t bytes = st->doorbell[1];
        SparkWeightdMeshRoute route;
        route.word = st->doorbell[3];
        __atomic_store_n(&st->last_seen_tag, tag, __ATOMIC_RELEASE);
        __atomic_store_n(&st->last_route_word, route.word, __ATOMIC_RELEASE);
        __atomic_store_n(&st->last_doorbell_bytes, bytes, __ATOMIC_RELEASE);
        __atomic_store_n(st->shipped, tag, __ATOMIC_RELEASE);
        uint64_t ring = (tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u);
        uint64_t t0 = NowNs();
        uint32_t cancel_ns = __atomic_load_n(&st->cancel_after_ns, __ATOMIC_ACQUIRE);
        if ( cancel_ns != 0u )
        {
            while ( NowNs() - t0 < (uint64_t)cancel_ns ) {}
            __atomic_fetch_add(st->cancel, 1ull, __ATOMIC_RELEASE);
            continue;
        }
        uint32_t drop = __atomic_load_n(&st->drop_peer, __ATOMIC_ACQUIRE);
        uint32_t mask = (uint32_t)route.fields.peer_mask;
        for ( uint32_t k = 0u; k < st->degree; ++k )
        {
            uint32_t p = st->order[k];
            if ( p >= st->degree || p == st->rank )
                continue;
            if ( (mask & (1u << p)) == 0u )
                continue;
            if ( drop == p )
                continue;
            uint32_t target_ns = st->delay_ns[p];
            while ( target_ns != 0u && NowNs() - t0 < (uint64_t)target_ns ) {}
            uint8_t *dst = st->band + ((uint64_t)p * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring) * SPARK_WEIGHTD_MESH_SLOT_BYTES;
            if ( st->peer_payloads[p] != 0 && st->payload_bytes != 0u )
                memcpy(dst, st->peer_payloads[p], (size_t)st->payload_bytes);
            __atomic_store_n((volatile uint64_t *)(dst + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u), tag, __ATOMIC_RELEASE);
        }
    }
    return 0;
}

typedef struct MeshFixture
{
    uint8_t *band_host;
    uint8_t *band_dev;
    uint8_t *staging_host;
    uint8_t *staging_dev;
    uint64_t *doorbell_host;
    uint64_t *doorbell_dev;
    uint64_t *shipped_host;
    uint64_t *shipped_dev;
    uint64_t *cancel_host;
    uint64_t *cancel_dev;
    SparkWeightdMeshWaitRequest *gate_host;
    SparkWeightdMeshWaitRequest *gate_dev;
    SparkTpMeshRoundControl *control;
    void *scratch;
    uint16_t *local_dev;
    uint16_t *out_ref;
    uint16_t *out_fast;
    uint16_t *prefetch_buf;
    LmWeightPrefetchGroup *prefetch_dev;
    cudaStream_t stream;
    ResponderState responder;
    pthread_t thread;
    uint64_t epoch;
} MeshFixture;

static void FixtureInit(MeshFixture *f)
{
    memset(f, 0, sizeof(*f));
    const size_t band_bytes = 16u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    const size_t staging_bytes = 16u * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES;
    CUDA_OK(cudaHostAlloc(&f->band_host, band_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->band_dev, f->band_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->staging_host, staging_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->staging_dev, f->staging_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->doorbell_host, 64u, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->doorbell_dev, f->doorbell_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->shipped_host, 64u, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->shipped_dev, f->shipped_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->cancel_host, 64u, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->cancel_dev, f->cancel_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->gate_host, sizeof(*f->gate_host), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->gate_dev, f->gate_host, 0u));
    CUDA_OK(cudaMalloc(&f->control, sizeof(*f->control)));
    CUDA_OK(cudaMalloc(&f->scratch, 2u * SPARK_WEIGHTD_MESH_SLOT_BYTES));
    CUDA_OK(cudaMalloc(&f->local_dev, 256u * 1024u * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&f->out_ref, 256u * 1024u * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&f->out_fast, 256u * 1024u * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&f->prefetch_buf, 65536u));
    CUDA_OK(cudaMalloc(&f->prefetch_dev, sizeof(LmWeightPrefetchGroup)));
    CUDA_OK(cudaStreamCreateWithFlags(&f->stream, cudaStreamNonBlocking));
    memset(f->band_host, 0, band_bytes);
    memset(f->staging_host, 0, staging_bytes);
    memset(f->doorbell_host, 0, 64u);
    memset(f->shipped_host, 0, 64u);
    memset(f->cancel_host, 0, 64u);
    memset(f->gate_host, 0, sizeof(*f->gate_host));
    for ( uint32_t p = 0u; p < 16u; ++p )
    {
        f->responder.peer_payloads[p] = (uint16_t *)calloc(32768u, sizeof(uint16_t));
        f->responder.order[p] = p;
    }
    f->responder.band = f->band_host;
    f->responder.doorbell = f->doorbell_host;
    f->responder.shipped = f->shipped_host;
    f->responder.cancel = f->cancel_host;
    f->responder.drop_peer = UINT32_MAX;
    f->epoch = 10u;
    if ( pthread_create(&f->thread, 0, ResponderMain, &f->responder) != 0 )
    {
        fprintf(stderr, "pthread_create failed\n");
        exit(1);
    }
}

static void FixtureDestroy(MeshFixture *f)
{
    __atomic_store_n(&f->responder.stop, 1u, __ATOMIC_RELEASE);
    pthread_join(f->thread, 0);
    for ( uint32_t p = 0u; p < 16u; ++p )
        free(f->responder.peer_payloads[p]);
    cudaStreamDestroy(f->stream);
    cudaFree(f->prefetch_dev);
    cudaFree(f->prefetch_buf);
    cudaFree(f->out_fast);
    cudaFree(f->out_ref);
    cudaFree(f->local_dev);
    cudaFree(f->scratch);
    cudaFree(f->control);
    cudaFreeHost(f->gate_host);
    cudaFreeHost(f->cancel_host);
    cudaFreeHost(f->shipped_host);
    cudaFreeHost(f->doorbell_host);
    cudaFreeHost(f->staging_host);
    cudaFreeHost(f->band_host);
}

static void FixtureResetEpoch(MeshFixture *f, uint32_t rank, uint32_t degree, uint64_t payload_bytes)
{
    SparkTpMeshRoundControl host;
    memset(&host, 0, sizeof(host));
    f->epoch++;
    host.epoch = f->epoch;
    host.cancel_expected = *f->cancel_host;
    host.shipped_cell = (uint64_t)(uintptr_t)f->shipped_dev;
    host.cancel_cell = (uint64_t)(uintptr_t)f->cancel_dev;
    f->doorbell_host[0] = 0u;
    f->doorbell_host[1] = 0u;
    f->doorbell_host[2] = 0u;
    f->doorbell_host[3] = 0u;
    *f->shipped_host = 0u;
    f->responder.rank = rank;
    f->responder.degree = degree;
    f->responder.payload_bytes = payload_bytes;
    __atomic_store_n(&f->responder.cancel_after_ns, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.drop_peer, UINT32_MAX, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.reset_last, 1u, __ATOMIC_RELEASE);
    while ( __atomic_load_n(&f->responder.reset_last, __ATOMIC_ACQUIRE) != 0u ) {}
    CUDA_OK(cudaMemcpy(f->control, &host, sizeof(host), cudaMemcpyHostToDevice));
}

static void SetSkewOrder(MeshFixture *f, uint32_t rank, uint32_t use_skew)
{
    uint32_t order[16u] = { 15u, 2u, 11u, 0u, 9u, 5u, 14u, 3u, 8u, 1u, 13u, 6u, 10u, 4u, 12u, 7u };
    for ( uint32_t i = 0u; i < 16u; ++i )
        f->responder.order[i] = order[i];
    for ( uint32_t p = 0u; p < 16u; ++p )
    {
        if ( use_skew == 0u )
            f->responder.delay_ns[p] = 0u;
        else if ( p == (rank ^ 1u) )
            f->responder.delay_ns[p] = 8000u;
        else if ( ((p ^ rank) & 8u) == 0u )
            f->responder.delay_ns[p] = 8000u + (p & 7u) * 1000u;
        else
            f->responder.delay_ns[p] = 22000u + (p & 7u) * 2800u;
    }
    if ( use_skew != 0u )
    {
        for ( uint32_t i = 0u; i < 16u; ++i )
            for ( uint32_t j = i + 1u; j < 16u; ++j )
                if ( f->responder.delay_ns[f->responder.order[j]] < f->responder.delay_ns[f->responder.order[i]] )
                {
                    uint32_t tmp = f->responder.order[i];
                    f->responder.order[i] = f->responder.order[j];
                    f->responder.order[j] = tmp;
                }
    }
}

static size_t CaptureNodeCount(cudaGraph_t graph)
{
    size_t n = 0u;
    CUDA_OK(cudaGraphGetNodes(graph, 0, &n));
    return n;
}

static void TestAllReduceExact(MeshFixture *f, uint32_t rank, uint32_t elements, uint32_t inplace, uint32_t with_prefetch)
{
    const uint32_t degree = 16u;
    uint16_t *h_local = (uint16_t *)malloc(elements * sizeof(uint16_t));
    uint16_t *h_ref = (uint16_t *)malloc(elements * sizeof(uint16_t));
    uint16_t *h_fast = (uint16_t *)malloc(elements * sizeof(uint16_t));
    for ( uint32_t p = 0u; p < degree; ++p )
        for ( uint32_t i = 0u; i < elements; ++i )
            f->responder.peer_payloads[p][i] = PatternBf16(p, i, 100u + rank + elements);
    memcpy(h_local, f->responder.peer_payloads[rank], elements * sizeof(uint16_t));
    SetSkewOrder(f, rank, 1u);

    FixtureResetEpoch(f, rank, degree, (uint64_t)elements * sizeof(uint16_t));
    CUDA_OK(cudaMemcpy(f->local_dev, h_local, elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->out_ref, 0, elements * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardware(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_ref, f->scratch,
        elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_ref, f->out_ref, elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    FixtureResetEpoch(f, rank, degree, (uint64_t)elements * sizeof(uint16_t));
    void *fast_out = inplace != 0u ? (void *)f->local_dev : (void *)f->out_fast;
    LmWeightPrefetchPlan plan;
    LmWeightPrefetchGroup host_group;
    memset(&plan, 0, sizeof(plan));
    memset(&host_group, 0, sizeof(host_group));
    if ( with_prefetch != 0u )
    {
        plan.host = &host_group;
        plan.device = f->prefetch_dev;
        plan.capacity = 1u;
    }

    cudaGraph_t graph = 0;
    cudaGraphExec_t exec = 0;
    CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
    if ( with_prefetch != 0u )
        LmWeightPrefetchArm(&plan, f->stream);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, fast_out, f->scratch,
        elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    if ( with_prefetch != 0u )
        LmWeightPrefetchNote(f->stream, f->prefetch_buf, 65536u);
    CUDA_OK(cudaStreamEndCapture(f->stream, &graph));
    if ( with_prefetch != 0u )
        CUDA_OK(LmWeightPrefetchDisarm(&plan, 1u));
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));

    for ( uint32_t rep = 0u; rep < 8u; ++rep )
    {
        CUDA_OK(cudaMemcpy(f->local_dev, h_local, elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
        if ( inplace == 0u )
            CUDA_OK(cudaMemset(f->out_fast, 0, elements * sizeof(uint16_t)));
        CUDA_OK(cudaGraphLaunch(exec, f->stream));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_fast, fast_out, elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK(memcmp(h_ref, h_fast, elements * sizeof(uint16_t)) == 0, "allreduce fast is bitwise identical to reference");
    }
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 8u, "allreduce graph replays complete with zero errors");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    free(h_fast);
    free(h_ref);
    free(h_local);
}

static void TestAllGatherExact(MeshFixture *f, uint32_t rank, uint32_t local_elements)
{
    const uint32_t degree = 16u;
    const uint32_t total = local_elements * degree;
    uint16_t *h_local = (uint16_t *)malloc(local_elements * sizeof(uint16_t));
    uint16_t *h_ref = (uint16_t *)malloc(total * sizeof(uint16_t));
    uint16_t *h_fast = (uint16_t *)malloc(total * sizeof(uint16_t));
    for ( uint32_t p = 0u; p < degree; ++p )
        for ( uint32_t i = 0u; i < local_elements; ++i )
            f->responder.peer_payloads[p][i] = PatternBf16(p, i, 300u + rank + local_elements);
    memcpy(h_local, f->responder.peer_payloads[rank], local_elements * sizeof(uint16_t));
    SetSkewOrder(f, rank, 1u);

    FixtureResetEpoch(f, rank, degree, (uint64_t)local_elements * sizeof(uint16_t));
    CUDA_OK(cudaMemcpy(f->local_dev, h_local, local_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->out_ref, 0, total * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardware(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_ref, f->scratch,
        total, 0u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_ref, f->out_ref, total * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    FixtureResetEpoch(f, rank, degree, (uint64_t)local_elements * sizeof(uint16_t));
    cudaGraph_t graph = 0;
    cudaGraphExec_t exec = 0;
    CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
        total, 0u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamEndCapture(f->stream, &graph));
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));

    for ( uint32_t rep = 0u; rep < 8u; ++rep )
    {
        CUDA_OK(cudaMemset(f->out_fast, 0, total * sizeof(uint16_t)));
        CUDA_OK(cudaGraphLaunch(exec, f->stream));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_fast, f->out_fast, total * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK(memcmp(h_ref, h_fast, total * sizeof(uint16_t)) == 0, "allgather fast is bitwise identical to reference");
    }
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 8u, "allgather graph replays complete with zero errors");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    free(h_fast);
    free(h_ref);
    free(h_local);
}

static void TestPairGatherExact(MeshFixture *f, uint32_t rank, uint32_t local_elements)
{
    const uint32_t degree = 16u;
    const uint32_t total = local_elements * degree;
    const uint32_t partner = rank ^ 1u;
    uint16_t *h_local = (uint16_t *)malloc(local_elements * sizeof(uint16_t));
    uint16_t *h_ref = (uint16_t *)malloc(total * sizeof(uint16_t));
    uint16_t *h_fast = (uint16_t *)malloc(total * sizeof(uint16_t));
    for ( uint32_t p = 0u; p < degree; ++p )
        for ( uint32_t i = 0u; i < local_elements; ++i )
            f->responder.peer_payloads[p][i] = PatternBf16(p, i, 500u + rank);
    memcpy(h_local, f->responder.peer_payloads[rank], local_elements * sizeof(uint16_t));
    SetSkewOrder(f, rank, 1u);

    FixtureResetEpoch(f, rank, degree, (uint64_t)local_elements * sizeof(uint16_t));
    CUDA_OK(cudaMemcpy(f->local_dev, h_local, local_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->out_ref, 0, total * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardware(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_ref, f->scratch,
        total, 0u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER | SPARK_TP_MESH_ROUTES_PAIR, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_ref, f->out_ref, total * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    FixtureResetEpoch(f, rank, degree, (uint64_t)local_elements * sizeof(uint16_t));
    CUDA_OK(cudaMemset(f->out_fast, 0, total * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
        total, 0u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER | SPARK_TP_MESH_ROUTES_PAIR | SPARK_TP_MESH_ROUTES_PAIR_ONLY,
        f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_fast, f->out_fast, total * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SparkWeightdMeshRoute route;
    route.word = __atomic_load_n(&f->responder.last_route_word, __ATOMIC_ACQUIRE);
    CHECK(route.fields.peer_mask == (1u << partner), "pair-only pair gather masks only partner peer");
    CHECK(route.fields.pair == 1u, "pair-only pair gather keeps pair route bit");
    CHECK(memcmp(h_ref + (size_t)rank * local_elements, h_fast + (size_t)rank * local_elements, local_elements * sizeof(uint16_t)) == 0,
        "pair-only pair gather own slice exact");
    CHECK(memcmp(h_ref + (size_t)partner * local_elements, h_fast + (size_t)partner * local_elements, local_elements * sizeof(uint16_t)) == 0,
        "pair-only pair gather partner slice exact");

    FixtureResetEpoch(f, rank, degree, (uint64_t)local_elements * sizeof(uint16_t));
    CUDA_OK(cudaMemset(f->out_fast, 0, total * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
        total, 0u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER | SPARK_TP_MESH_ROUTES_PAIR,
        f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    route.word = __atomic_load_n(&f->responder.last_route_word, __ATOMIC_ACQUIRE);
    CHECK(route.fields.peer_mask == (((1u << degree) - 1u) & ~(1u << rank)), "full pair gather restores full peer mask");
    {
        SparkTpMeshRoundControl ctrl;
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "full pair gather completes with zero errors");
    }
    free(h_fast);
    free(h_ref);
    free(h_local);
}

static void TestAllToAllExact(MeshFixture *f, uint32_t rank, uint32_t per_peer)
{
    const uint32_t degree = 16u;
    const uint32_t total = per_peer * degree;
    uint16_t *h_local = (uint16_t *)malloc(total * sizeof(uint16_t));
    uint16_t *h_ref = (uint16_t *)malloc(total * sizeof(uint16_t));
    uint16_t *h_fast = (uint16_t *)malloc(total * sizeof(uint16_t));
    for ( uint32_t p = 0u; p < degree; ++p )
    {
        for ( uint32_t i = 0u; i < per_peer; ++i )
        {
            h_local[(size_t)p * per_peer + i] = PatternBf16(rank, p * 4096u + i, 700u);
            f->responder.peer_payloads[p][i] = PatternBf16(p, rank * 4096u + i, 700u);
        }
    }
    SetSkewOrder(f, rank, 1u);

    FixtureResetEpoch(f, rank, degree, (uint64_t)per_peer * sizeof(uint16_t));
    CUDA_OK(cudaMemcpy(f->local_dev, h_local, total * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->out_ref, 0, total * sizeof(uint16_t)));
    cudaGraph_t ref_graph = 0, fast_graph = 0;
    cudaGraphExec_t ref_exec = 0, fast_exec = 0;
    CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
    CUDA_OK(SparkTpLaunchMeshHardware(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_ref, f->scratch,
        per_peer, SPARK_TP_MESH_OPERATION_ALL_TO_ALL, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamEndCapture(f->stream, &ref_graph));
    CUDA_OK(cudaGraphInstantiate(&ref_exec, ref_graph, 0, 0, 0));
    size_t ref_nodes = CaptureNodeCount(ref_graph);
    CUDA_OK(cudaGraphLaunch(ref_exec, f->stream));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_ref, f->out_ref, total * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    FixtureResetEpoch(f, rank, degree, (uint64_t)per_peer * sizeof(uint16_t));
    memset(f->staging_host, 0, 16u * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
    CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
        per_peer, SPARK_TP_MESH_OPERATION_ALL_TO_ALL, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamEndCapture(f->stream, &fast_graph));
    CUDA_OK(cudaGraphInstantiate(&fast_exec, fast_graph, 0, 0, 0));
    size_t fast_nodes = CaptureNodeCount(fast_graph);
    CHECK(ref_nodes == 5u && fast_nodes == 2u, "all-to-all fast reduces graph nodes from 5 to 2");

    for ( uint32_t rep = 0u; rep < 8u; ++rep )
    {
        CUDA_OK(cudaMemset(f->out_fast, 0, total * sizeof(uint16_t)));
        CUDA_OK(cudaGraphLaunch(fast_exec, f->stream));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_fast, f->out_fast, total * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK(memcmp(h_ref, h_fast, total * sizeof(uint16_t)) == 0, "all-to-all fast output bitwise identical to reference");
    }
    for ( uint32_t p = 0u; p < degree; ++p )
    {
        if ( p == rank ) continue;
        const uint16_t *staged = (const uint16_t *)(f->staging_host + (size_t)p * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
        CHECK(memcmp(staged, h_local + (size_t)p * per_peer, per_peer * sizeof(uint16_t)) == 0,
            "all-to-all fast stages exact per-peer payload");
    }
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 8u, "all-to-all graph replays complete with zero errors");
    cudaGraphExecDestroy(fast_exec);
    cudaGraphDestroy(fast_graph);
    cudaGraphExecDestroy(ref_exec);
    cudaGraphDestroy(ref_graph);
    free(h_fast);
    free(h_ref);
    free(h_local);
}

static void TestCancelAndTimeout(MeshFixture *f)
{
    const uint32_t rank = 0u, degree = 16u;
    SparkTpMeshRoundControl ctrl;
    SetSkewOrder(f, rank, 0u);

    FixtureResetEpoch(f, rank, degree, 4096u * sizeof(uint16_t));
    __atomic_store_n(&f->responder.cancel_after_ns, 50000u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
        4096u, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK((ctrl.error_word & SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED) != 0u && ctrl.rounds_done == 0u,
        "streaming allreduce cancels cleanly");

    FixtureResetEpoch(f, rank, degree, 6144u * sizeof(uint16_t));
    __atomic_store_n(&f->responder.drop_peer, 7u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
        6144u, SPARK_TP_MESH_OPERATION_ALL_TO_ALL, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u, "fused all-to-all times out cleanly when a peer drops");

    FixtureResetEpoch(f, rank, degree, 2048u * sizeof(uint16_t));
    __atomic_store_n(&f->responder.drop_peer, 1u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
        32768u, 0u, 1u, 1u, SPARK_TP_MESH_ROUTES_PEER | SPARK_TP_MESH_ROUTES_PAIR | SPARK_TP_MESH_ROUTES_PAIR_ONLY,
        f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u, "pair-only pair gather times out cleanly when partner drops");
}

typedef struct BenchResult
{
    double wall_us;
    double copy_us;
    double wait_us;
    double combine_us;
    size_t nodes;
} BenchResult;

static BenchResult BenchOp(MeshFixture *f, uint32_t fast, uint32_t op, uint32_t elements, uint32_t routes, uint32_t skew, uint32_t iters)
{
    const uint32_t rank = 0u, degree = 16u;
    const uint64_t payload_bytes = (op == 0u ? (uint64_t)(elements / degree) : (uint64_t)elements) * sizeof(uint16_t);
    SetSkewOrder(f, rank, skew);
    FixtureResetEpoch(f, rank, degree, payload_bytes);
    cudaGraph_t graph = 0;
    cudaGraphExec_t exec = 0;
    CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
    if ( fast != 0u )
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_fast, f->scratch,
            elements, op, 1u, 1u, routes, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    else
        CUDA_OK(SparkTpLaunchMeshHardware(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_ref, f->scratch,
            elements, op, 1u, 1u, routes, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamEndCapture(f->stream, &graph));
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));
    BenchResult res;
    res.nodes = CaptureNodeCount(graph);
    for ( uint32_t i = 0u; i < 16u; ++i )
    {
        CUDA_OK(cudaGraphLaunch(exec, f->stream));
        CUDA_OK(cudaStreamSynchronize(f->stream));
    }
    FixtureResetEpoch(f, rank, degree, payload_bytes);
    uint64_t t0 = NowNs();
    for ( uint32_t i = 0u; i < iters; ++i )
        CUDA_OK(cudaGraphLaunch(exec, f->stream));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    uint64_t t1 = NowNs();
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    if ( ctrl.error_word != 0u || ctrl.rounds_done != iters )
        fprintf(stderr, "BENCH-ERR fast=%u op=%u elem=%u routes=0x%x skew=%u err=0x%llx diag=0x%llx done=%llu/%u seq=%llu\n",
            fast, op, elements, routes, skew, (unsigned long long)ctrl.error_word, (unsigned long long)ctrl.diag_word,
            (unsigned long long)ctrl.rounds_done, iters, (unsigned long long)ctrl.seq);
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == iters, "benchmark rounds finished cleanly");
    res.wall_us = (double)(t1 - t0) / 1000.0 / (double)iters;
    res.copy_us = (double)ctrl.copy_ns / 1000.0 / (double)iters;
    res.wait_us = (double)ctrl.peer_wait_ns / 1000.0 / (double)iters;
    res.combine_us = (double)ctrl.combine_ns / 1000.0 / (double)iters;
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return res;
}

static void RunBenchmarks(MeshFixture *f)
{
    const uint32_t iters = 200u;
    struct { const char *name; uint32_t op; uint32_t elements; uint32_t ref_routes; uint32_t fast_routes; } cases[] = {
        { "ALL_REDUCE_4096", 1u, 4096u, SPARK_TP_MESH_ROUTES_PEER, SPARK_TP_MESH_ROUTES_PEER },
        { "ALL_REDUCE_6144", 1u, 6144u, SPARK_TP_MESH_ROUTES_PEER, SPARK_TP_MESH_ROUTES_PEER },
        { "ALL_GATHER_6144", 0u, 6144u, SPARK_TP_MESH_ROUTES_PEER, SPARK_TP_MESH_ROUTES_PEER },
        { "ALL_GATHER_65536", 0u, 65536u, SPARK_TP_MESH_ROUTES_PEER, SPARK_TP_MESH_ROUTES_PEER },
        { "PAIR_GATHER_32768", 0u, 32768u, SPARK_TP_MESH_ROUTES_PEER | SPARK_TP_MESH_ROUTES_PAIR,
            SPARK_TP_MESH_ROUTES_PEER | SPARK_TP_MESH_ROUTES_PAIR | SPARK_TP_MESH_ROUTES_PAIR_ONLY },
        { "ALL_TO_ALL_6144", SPARK_TP_MESH_OPERATION_ALL_TO_ALL, 6144u, SPARK_TP_MESH_ROUTES_PEER, SPARK_TP_MESH_ROUTES_PEER },
    };
    for ( uint32_t skew = 0u; skew <= 1u; ++skew )
    {
        printf("\n=== TP16 B1 BENCHMARK (%s, iters=%u) ===\n", skew != 0u ? "16-Spark Skew 8..42us" : "Zero Skew", iters);
        printf("%-18s | %5s | %9s %9s %9s | %9s %9s %9s | %8s\n",
            "Op", "Nodes", "RefWall", "RefCopy", "RefComb", "FastWall", "FastCopy", "FastComb", "Delta_us");
        for ( uint32_t c = 0u; c < sizeof(cases) / sizeof(cases[0]); ++c )
        {
            BenchResult r = BenchOp(f, 0u, cases[c].op, cases[c].elements, cases[c].ref_routes, skew, iters);
            BenchResult q = BenchOp(f, 1u, cases[c].op, cases[c].elements, cases[c].fast_routes, skew, iters);
            printf("%-18s | %2zu->%2zu | %8.2fus %8.2fus %8.2fus | %8.2fus %8.2fus %8.2fus | %+7.2fus\n",
                cases[c].name, r.nodes, q.nodes,
                r.wall_us, r.copy_us, r.combine_us,
                q.wall_us, q.copy_us, q.combine_us,
                q.wall_us - r.wall_us);
        }
    }
}

int main(void)
{
    MeshFixture fixture;
    FixtureInit(&fixture);
    void *alias = 0;
    CHECK(SparkTpMeshHardwarePrepareFast(fixture.band_host, &alias) == SPARK_GPU_SUCCESS && alias != 0,
        "fast hardware prepare preloads all fast kernels");

    uint32_t test_ranks[] = { 0u, 7u, 14u, 15u };
    for ( uint32_t r = 0u; r < 4u; ++r )
    {
        TestAllReduceExact(&fixture, test_ranks[r], 4096u, 0u, 0u);
        TestAllReduceExact(&fixture, test_ranks[r], 6144u, 1u, 1u);
        TestAllGatherExact(&fixture, test_ranks[r], 384u);
        TestAllGatherExact(&fixture, test_ranks[r], 4096u);
        TestPairGatherExact(&fixture, test_ranks[r], 2048u);
        TestAllToAllExact(&fixture, test_ranks[r], 6144u);
    }
    TestAllToAllExact(&fixture, 0u, 8192u);
    TestCancelAndTimeout(&fixture);
    RunBenchmarks(&fixture);

    FixtureDestroy(&fixture);
    printf("tp_mesh_fast_b1_cuda: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures != 0u ? 1 : 0;
}
