#include <cuda_runtime.h>
#include <math.h>
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

static uint16_t FloatToBf16Rn(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t lsb = (bits >> 16u) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16u);
}

static float Bf16ToFloat(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16u;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static uint8_t HostFloatToUe8m0Ceil(float v)
{
    if ( !(v > 0.0f) ) return 0u;
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t exp = (bits >> 23u) & 0xffu;
    if ( (bits & 0x7fffffu) != 0u && exp < 254u ) exp += 1u;
    if ( exp == 0u ) exp = 1u;
    return (uint8_t)exp;
}

static float HostUe8m0ToFloat(uint8_t code)
{
    if ( code == 0u ) return 0.0f;
    uint32_t bits = (uint32_t)code << 23u;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float g_e4m3_lut[256];

static void InitE4m3Lut(void)
{
    for ( uint32_t code = 0u; code < 256u; ++code )
    {
        uint32_t sign = (code >> 7u) & 1u;
        uint32_t exp = (code >> 3u) & 0xfu;
        uint32_t man = code & 0x7u;
        float mag;
        if ( exp == 0u ) mag = (float)man * (1.0f / 512.0f);
        else if ( exp == 15u && man == 7u ) mag = 0.0f;
        else mag = ldexpf(1.0f + (float)man * 0.125f, (int)exp - 7);
        g_e4m3_lut[code] = sign != 0u ? -mag : mag;
    }
}

static inline float HostE4m3ToFloat(uint8_t code)
{
    return g_e4m3_lut[code];
}

static inline uint8_t HostFloatToE4m3Rn(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    uint8_t sign = (uint8_t)((bits >> 24u) & 0x80u);
    uint32_t abits = bits & 0x7fffffffu;
    if ( abits == 0u ) return sign;
    if ( abits >= 0x43e00000u ) return (uint8_t)(sign | 0x7eu);
    int32_t exp = (int32_t)(abits >> 23u) - 127;
    if ( exp < -6 )
    {
        float scaled = fabsf(x) * 512.0f;
        uint32_t q = (uint32_t)scaled;
        float frac = scaled - (float)q;
        if ( frac > 0.5f || (frac == 0.5f && (q & 1u) != 0u) ) q++;
        if ( q >= 8u ) return (uint8_t)(sign | 0x08u);
        return (uint8_t)(sign | (uint8_t)q);
    }
    uint32_t fman = abits & 0x7fffffu;
    uint32_t m3 = fman >> 20u;
    uint32_t rem = fman & 0xfffffu;
    if ( rem > 0x80000u || (rem == 0x80000u && (m3 & 1u) != 0u) ) m3++;
    int32_t e4 = exp + 7;
    if ( m3 == 8u ) { m3 = 0u; e4++; }
    if ( e4 > 15 || (e4 == 15 && m3 > 6u) ) return (uint8_t)(sign | 0x7eu);
    return (uint8_t)(sign | (uint8_t)((e4 << 3) | (int32_t)m3));
}

static void HostQuantRange(uint8_t *dst, uint64_t scale_off, const uint16_t *src, uint64_t count, uint32_t block, uint32_t scale_bytes)
{
    uint8_t *sc_dst = dst + scale_off;
    for ( uint64_t b = 0u; b < count / block; ++b )
    {
        float amax = 0.0f;
        for ( uint32_t i = 0u; i < block; ++i )
        {
            float av = fabsf(Bf16ToFloat(src[b * block + i]));
            if ( av > amax ) amax = av;
        }
        float sc = 0.0f;
        if ( amax > 0.0f )
        {
            if ( scale_bytes == 1u )
            {
                uint8_t code = HostFloatToUe8m0Ceil(amax * (1.0f / 448.0f));
                sc_dst[b] = code;
                sc = HostUe8m0ToFloat(code);
            }
            else
            {
                uint16_t code = FloatToBf16Rn(amax * (1.0f / 448.0f));
                sc = Bf16ToFloat(code);
                if ( sc == 0.0f ) { code = 0x0080u; sc = Bf16ToFloat(code); }
                ((uint16_t *)sc_dst)[b] = code;
            }
        }
        else
        {
            if ( scale_bytes == 1u ) sc_dst[b] = 0u;
            else ((uint16_t *)sc_dst)[b] = 0u;
        }
        float inv = sc > 0.0f ? (1.0f / sc) : 0.0f;
        for ( uint32_t i = 0u; i < block; ++i )
            dst[b * block + i] = HostFloatToE4m3Rn(Bf16ToFloat(src[b * block + i]) * inv);
    }
}

static void HostDequantRange(uint16_t *dst, const uint8_t *src, uint64_t scale_off, uint64_t count, uint32_t block, uint32_t scale_bytes)
{
    const uint8_t *sc_src = src + scale_off;
    for ( uint64_t b = 0u; b < count / block; ++b )
    {
        float sc = scale_bytes == 1u ? HostUe8m0ToFloat(sc_src[b]) : Bf16ToFloat(((const uint16_t *)sc_src)[b]);
        for ( uint32_t i = 0u; i < block; ++i )
            dst[b * block + i] = FloatToBf16Rn(HostE4m3ToFloat(src[b * block + i]) * sc);
    }
}

static uint32_t CodecBlock(uint32_t codec)
{
    uint32_t kind = codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16 ? codec : (codec - 3u);
    return kind == SPARK_TP_MESH_WIRE_CODEC_FP8_B16 ? 16u : 32u;
}

static uint32_t CodecScaleBytes(uint32_t codec)
{
    uint32_t kind = codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16 ? codec : (codec - 3u);
    return kind == SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32 ? 1u : 2u;
}

static uint32_t CodecPhase1Quant(uint32_t codec)
{
    return codec >= SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32 && codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16 ? 1u : 0u;
}

static uint64_t CodecHalfWidth(uint32_t codec)
{
    uint32_t block = CodecBlock(codec);
    uint32_t sb = CodecScaleBytes(codec);
    uint64_t per256 = 256u + (256u / block) * (uint64_t)sb;
    return (SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / per256) * 256u;
}

static uint16_t PatternBf16(uint32_t peer, uint64_t idx, uint32_t salt)
{
    uint32_t h = (peer + 1u) * 2654435761u ^ ((uint32_t)idx + 7u) * 2246822519u ^ (salt * 3266489917u);
    int32_t sign = (h & 1u) ? -1 : 1;
    float mult = ((idx & 255u) == 13u) ? 16.0f : 1.0f;
    float v = (float)sign * mult * (0.015625f + (float)((h >> 8u) & 255u) * 0.00390625f);
    return FloatToBf16Rn(v);
}

typedef struct ResponderState
{
    uint8_t *band;
    uint8_t *staging;
    volatile uint64_t *doorbell;
    volatile uint64_t *doorbell_alt;
    volatile uint64_t *shipped;
    volatile uint64_t *cancel;
    volatile SparkWeightdMeshWaitRequest *gate;
    uint16_t *peer_buf[16u];
    uint64_t peer_buf_elements;
    uint32_t rank;
    uint32_t degree;
    uint32_t copy_payload;
    uint32_t op_kind;
    uint32_t codec;
    uint64_t total_elements;
    uint32_t chunk_idx;
    uint32_t wire_gbps_x10;
    uint32_t base_lat_ns;
    uint32_t order[16u];
    volatile uint32_t stop;
    volatile uint32_t reset_req;
    volatile uint32_t cancel_after_ns;
    volatile uint32_t drop_peer;
    volatile uint64_t rounds_serviced;
    volatile uint64_t wire_bytes_total;
} ResponderState;

static void ResponderServiceGate(ResponderState *st, uint64_t *gate_seen)
{
    uint64_t req_id = __atomic_load_n(&st->gate->request_id, __ATOMIC_ACQUIRE);
    if ( req_id == *gate_seen || req_id == 0u ) return;
    uint64_t kind = st->gate->kind;
    uint64_t tag = st->gate->tag;
    if ( kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED && __atomic_load_n(st->shipped, __ATOMIC_ACQUIRE) < tag ) return;
    *gate_seen = req_id;
    st->gate->error = 0u;
    st->gate->diag = 0u;
    __atomic_store_n(&st->gate->ready, 1u, __ATOMIC_RELEASE);
}

static void ResponderDeliverTag(ResponderState *st, volatile uint64_t *cell, uint64_t tag)
{
    uint64_t bytes = cell[1];
    SparkWeightdMeshRoute route;
    route.word = cell[3];
    uint64_t ring = (tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u);
    uint32_t half = route.fields.half;
    uint64_t half_off = SPARK_WEIGHTD_MESH_HALF_OFFSET(half);
    uint64_t tail_off = half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ?
        SPARK_WEIGHTD_MESH_TAIL_OFFSET(half) : (SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u);
    uint64_t t0 = NowNs();
    uint32_t cancel_ns = __atomic_load_n(&st->cancel_after_ns, __ATOMIC_ACQUIRE);
    if ( cancel_ns != 0u )
    {
        __atomic_store_n(st->shipped, tag, __ATOMIC_RELEASE);
        while ( NowNs() - t0 < (uint64_t)cancel_ns ) {}
        __atomic_fetch_add(st->cancel, 1ull, __ATOMIC_RELEASE);
        return;
    }
    uint32_t drop = __atomic_load_n(&st->drop_peer, __ATOMIC_ACQUIRE);
    uint32_t mask = (uint32_t)route.fields.peer_mask;
    uint64_t wire_per_peer_ns = 0u;
    uint64_t xfer_bytes = route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER ? route.fields.slice_bytes : bytes;
    if ( st->wire_gbps_x10 != 0u )
        wire_per_peer_ns = (xfer_bytes * 10ull) / (uint64_t)st->wire_gbps_x10;
    uint32_t sent = 0u;
    const uint32_t block = st->codec != 0u ? CodecBlock(st->codec) : 32u;
    const uint32_t sb = st->codec != 0u ? CodecScaleBytes(st->codec) : 1u;
    const uint32_t p1q = st->codec != 0u ? CodecPhase1Quant(st->codec) : 0u;
    for ( uint32_t k = 0u; k < st->degree; ++k )
    {
        uint32_t p = st->order[k];
        if ( p >= st->degree || p == st->rank ) continue;
        if ( (mask & (1u << p)) == 0u ) continue;
        if ( drop == p ) continue;
        sent++;
        if ( st->wire_gbps_x10 != 0u )
        {
            uint64_t target_ns = (uint64_t)st->base_lat_ns + (uint64_t)sent * wire_per_peer_ns;
            while ( NowNs() - t0 < target_ns ) {}
        }
        uint8_t *dst = st->band + ((uint64_t)p * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring) * SPARK_WEIGHTD_MESH_SLOT_BYTES + half_off;
        if ( st->copy_payload != 0u && st->peer_buf[p] != 0 )
        {
            if ( st->op_kind == 1u )
            {
                if ( SparkTpMeshDirectPhasesPerChunk(st->total_elements, st->degree, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER) == 1u )
                {
                    memcpy(dst, st->peer_buf[p], (size_t)(st->total_elements * sizeof(uint16_t)));
                }
                else if ( route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER )
                {
                    uint64_t slice = st->codec != 0u ? (((st->total_elements + st->degree - 1u) / st->degree + 31u) & ~UINT64_C(31)) :
                        SparkTpMeshRsagSlice(st->total_elements, st->degree);
                    uint64_t first = (uint64_t)st->rank * slice;
                    uint64_t len = st->total_elements > first ? st->total_elements - first : 0u;
                    if ( len > slice ) len = slice;
                    if ( len != 0u )
                    {
                        if ( p1q != 0u ) HostQuantRange(dst, slice, st->peer_buf[p] + first, len, block, sb);
                        else memcpy(dst, st->peer_buf[p] + first, (size_t)(len * sizeof(uint16_t)));
                    }
                }
                else
                {
                    uint64_t slice = st->codec != 0u ? (((st->total_elements + st->degree - 1u) / st->degree + 31u) & ~UINT64_C(31)) :
                        SparkTpMeshRsagSlice(st->total_elements, st->degree);
                    uint64_t first = (uint64_t)p * slice;
                    uint64_t len = st->total_elements > first ? st->total_elements - first : 0u;
                    if ( len > slice ) len = slice;
                    if ( len != 0u )
                    {
                        uint16_t *sum_bf16 = (uint16_t *)malloc((size_t)len * sizeof(uint16_t));
                        if ( p1q != 0u )
                        {
                            uint8_t *qtmp = (uint8_t *)malloc((size_t)(slice + (slice / block) * sb));
                            float *acc = (float *)calloc((size_t)len, sizeof(float));
                            for ( uint32_t q = 0u; q < st->degree; ++q )
                            {
                                HostQuantRange(qtmp, slice, st->peer_buf[q] + first, len, block, sb);
                                const uint8_t *sc_src = qtmp + slice;
                                for ( uint64_t b = 0u; b < len / block; ++b )
                                {
                                    float sc = sb == 1u ? HostUe8m0ToFloat(sc_src[b]) : Bf16ToFloat(((const uint16_t *)sc_src)[b]);
                                    for ( uint32_t j = 0u; j < block; ++j )
                                        acc[b * block + j] = fmaf(HostE4m3ToFloat(qtmp[b * block + j]), sc, acc[b * block + j]);
                                }
                            }
                            for ( uint64_t i = 0u; i < len; ++i ) sum_bf16[i] = FloatToBf16Rn(acc[i]);
                            free(acc); free(qtmp);
                        }
                        else
                        {
                            for ( uint64_t i = 0u; i < len; ++i )
                            {
                                float acc = 0.0f;
                                for ( uint32_t q = 0u; q < st->degree; ++q )
                                    acc += Bf16ToFloat(st->peer_buf[q][first + i]);
                                sum_bf16[i] = FloatToBf16Rn(acc);
                            }
                        }
                        if ( st->codec != 0u ) HostQuantRange(dst, len, sum_bf16, len, block, sb);
                        else memcpy(dst, sum_bf16, (size_t)(len * sizeof(uint16_t)));
                        free(sum_bf16);
                    }
                }
            }
            else if ( st->op_kind == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER )
            {
                uint64_t segment = st->total_elements / st->degree;
                uint64_t width = half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ?
                    (p1q != 0u ? CodecHalfWidth(st->codec) : (SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t))) : segment;
                uint64_t begin = (uint64_t)st->chunk_idx * width;
                uint64_t count = begin < segment ? (segment - begin < width ? segment - begin : width) : 0u;
                if ( count != 0u )
                {
                    if ( p1q != 0u ) HostQuantRange(dst, count, st->peer_buf[p] + (uint64_t)st->rank * segment + begin, count, block, sb);
                    else memcpy(dst, st->peer_buf[p] + (uint64_t)st->rank * segment + begin, (size_t)(count * sizeof(uint16_t)));
                }
            }
            else if ( st->op_kind == SPARK_TP_MESH_OPERATION_ALL_GATHER )
            {
                uint64_t local_el = st->total_elements / st->degree;
                uint64_t width = half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ?
                    (st->codec != 0u ? CodecHalfWidth(st->codec) : (SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t))) : local_el;
                uint64_t begin = (uint64_t)st->chunk_idx * width;
                uint64_t count = begin < local_el ? (local_el - begin < width ? local_el - begin : width) : 0u;
                if ( count != 0u )
                {
                    if ( st->codec != 0u ) HostQuantRange(dst, count, st->peer_buf[p] + begin, count, block, sb);
                    else memcpy(dst, st->peer_buf[p] + begin, (size_t)(count * sizeof(uint16_t)));
                }
            }
        }
        __atomic_store_n((volatile uint64_t *)(st->band + ((uint64_t)p * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring) * SPARK_WEIGHTD_MESH_SLOT_BYTES + tail_off),
            tag, __ATOMIC_RELEASE);
    }
    __atomic_store_n(st->shipped, tag, __ATOMIC_RELEASE);
    st->chunk_idx++;
    __atomic_fetch_add(&st->wire_bytes_total, (uint64_t)sent * xfer_bytes, __ATOMIC_RELEASE);
    __atomic_fetch_add(&st->rounds_serviced, 1ull, __ATOMIC_RELEASE);
}

static void *ResponderMain(void *arg)
{
    ResponderState *st = (ResponderState *)arg;
    uint64_t seen = 0u, gate_seen = 0u;
    while ( __atomic_load_n(&st->stop, __ATOMIC_ACQUIRE) == 0u )
    {
        if ( __atomic_load_n(&st->reset_req, __ATOMIC_ACQUIRE) != 0u )
        {
            seen = 0u; gate_seen = 0u; st->chunk_idx = 0u;
            __atomic_store_n(&st->reset_req, 0u, __ATOMIC_RELEASE);
        }
        ResponderServiceGate(st, &gate_seen);
        uint64_t t_pri = __atomic_load_n(&st->doorbell[0], __ATOMIC_ACQUIRE);
        uint64_t t_alt = __atomic_load_n(&st->doorbell_alt[0], __ATOMIC_ACQUIRE);
        uint64_t want = seen == 0u ? 0u : seen + 1u;
        volatile uint64_t *picked = 0;
        uint64_t tag = 0u;
        if ( want == 0u )
        {
            if ( t_pri != 0u && (t_alt == 0u || t_pri < t_alt) ) { picked = st->doorbell; tag = t_pri; }
            else if ( t_alt != 0u ) { picked = st->doorbell_alt; tag = t_alt; }
        }
        else
        {
            if ( t_pri == want ) { picked = st->doorbell; tag = t_pri; }
            else if ( t_alt == want ) { picked = st->doorbell_alt; tag = t_alt; }
        }
        if ( picked == 0 ) continue;
        seen = tag;
        ResponderDeliverTag(st, picked, tag);
    }
    return 0;
}

typedef struct WireFixture
{
    uint8_t *band_host, *band_dev;
    uint8_t *staging_host, *staging_dev;
    uint8_t *doorbell_page_host, *doorbell_page_dev;
    uint64_t *doorbell_host, *doorbell_dev, *doorbell_alt_host;
    uint64_t *shipped_host, *shipped_dev;
    uint64_t *cancel_host, *cancel_dev;
    SparkWeightdMeshWaitRequest *gate_host, *gate_dev;
    SparkTpMeshRoundControl *control;
    void *scratch;
    uint16_t *local_dev, *out_dev, *share_dev;
    cudaStream_t stream;
    ResponderState responder;
    pthread_t thread;
    uint64_t epoch;
} WireFixture;

static void FixtureInit(WireFixture *f, uint64_t max_elements)
{
    memset(f, 0, sizeof(*f));
    const size_t band_bytes = 16u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    const size_t staging_bytes = 16u * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES;
    CUDA_OK(cudaHostAlloc(&f->band_host, band_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->band_dev, f->band_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->staging_host, staging_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->staging_dev, f->staging_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->doorbell_page_host, SPARK_WEIGHTD_MESH_DOORBELL_BYTES, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->doorbell_page_dev, f->doorbell_page_host, 0u));
    f->doorbell_host = (uint64_t *)f->doorbell_page_host;
    f->doorbell_dev = (uint64_t *)f->doorbell_page_dev;
    f->doorbell_alt_host = f->doorbell_host + SPARK_WEIGHTD_MESH_DOORBELL_ALT_WORDS;
    CUDA_OK(cudaHostAlloc(&f->shipped_host, 64u, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->shipped_dev, f->shipped_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->cancel_host, 64u, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->cancel_dev, f->cancel_host, 0u));
    CUDA_OK(cudaHostAlloc(&f->gate_host, sizeof(*f->gate_host), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&f->gate_dev, f->gate_host, 0u));
    CUDA_OK(cudaMalloc(&f->control, sizeof(*f->control)));
    CUDA_OK(cudaMalloc(&f->scratch, 2u * SPARK_WEIGHTD_MESH_SLOT_BYTES));
    CUDA_OK(cudaMalloc(&f->local_dev, (size_t)max_elements * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&f->out_dev, (size_t)max_elements * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&f->share_dev, (size_t)(max_elements / 16u) * sizeof(uint16_t)));
    CUDA_OK(cudaStreamCreateWithFlags(&f->stream, cudaStreamNonBlocking));
    memset(f->band_host, 0, band_bytes);
    memset(f->staging_host, 0, staging_bytes);
    memset(f->doorbell_page_host, 0, SPARK_WEIGHTD_MESH_DOORBELL_BYTES);
    memset(f->shipped_host, 0, 64u);
    memset(f->cancel_host, 0, 64u);
    memset(f->gate_host, 0, sizeof(*f->gate_host));
    uint32_t perm[16u] = { 13u, 4u, 15u, 1u, 9u, 6u, 12u, 0u, 8u, 3u, 14u, 7u, 10u, 2u, 11u, 5u };
    for ( uint32_t p = 0u; p < 16u; ++p ) f->responder.order[p] = perm[p];
    f->responder.band = f->band_host;
    f->responder.staging = f->staging_host;
    f->responder.doorbell = f->doorbell_host;
    f->responder.doorbell_alt = f->doorbell_alt_host;
    f->responder.shipped = f->shipped_host;
    f->responder.cancel = f->cancel_host;
    f->responder.gate = f->gate_host;
    f->responder.drop_peer = UINT32_MAX;
    f->epoch = 500u;
    void *prepared = 0;
    CUDA_OK(SparkTpMeshHardwarePrepareFast(f->band_host, &prepared));
    if ( pthread_create(&f->thread, 0, ResponderMain, &f->responder) != 0 ) exit(1);
}

static void FixtureDestroy(WireFixture *f)
{
    __atomic_store_n(&f->responder.stop, 1u, __ATOMIC_RELEASE);
    pthread_join(f->thread, 0);
    for ( uint32_t p = 0u; p < 16u; ++p ) free(f->responder.peer_buf[p]);
    cudaStreamDestroy(f->stream);
    cudaFree(f->share_dev); cudaFree(f->out_dev); cudaFree(f->local_dev);
    cudaFree(f->scratch); cudaFree(f->control);
    cudaFreeHost(f->gate_host); cudaFreeHost(f->cancel_host); cudaFreeHost(f->shipped_host);
    cudaFreeHost(f->doorbell_page_host); cudaFreeHost(f->staging_host); cudaFreeHost(f->band_host);
}

static void FixtureResetEpoch(WireFixture *f, uint32_t rank, uint32_t degree, uint32_t op_kind, uint32_t codec,
    uint64_t total_elements, uint32_t copy_payload, uint32_t wire_gbps_x10, uint32_t base_lat_ns)
{
    SparkTpMeshRoundControl host;
    memset(&host, 0, sizeof(host));
    f->epoch++;
    host.epoch = f->epoch;
    host.cancel_expected = *f->cancel_host;
    host.shipped_cell = (uint64_t)(uintptr_t)f->shipped_dev;
    host.cancel_cell = (uint64_t)(uintptr_t)f->cancel_dev;
    memset(f->doorbell_host, 0, 32u);
    memset(f->doorbell_alt_host, 0, 32u);
    *f->shipped_host = 0u;
    memset(f->gate_host, 0, sizeof(*f->gate_host));
    f->responder.rank = rank;
    f->responder.degree = degree;
    f->responder.op_kind = op_kind;
    f->responder.codec = codec;
    f->responder.total_elements = total_elements;
    f->responder.copy_payload = copy_payload;
    f->responder.wire_gbps_x10 = wire_gbps_x10;
    f->responder.base_lat_ns = base_lat_ns;
    __atomic_store_n(&f->responder.cancel_after_ns, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.drop_peer, UINT32_MAX, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.rounds_serviced, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.wire_bytes_total, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.reset_req, 1u, __ATOMIC_RELEASE);
    while ( __atomic_load_n(&f->responder.reset_req, __ATOMIC_ACQUIRE) != 0u ) {}
    CUDA_OK(cudaMemcpy(f->control, &host, sizeof(host), cudaMemcpyHostToDevice));
}

static void EnsurePeerBufs(WireFixture *f, uint64_t elems_per_peer, uint32_t salt)
{
    if ( f->responder.peer_buf_elements < elems_per_peer )
    {
        for ( uint32_t p = 0u; p < 16u; ++p )
        {
            free(f->responder.peer_buf[p]);
            f->responder.peer_buf[p] = (uint16_t *)malloc((size_t)elems_per_peer * sizeof(uint16_t));
        }
        f->responder.peer_buf_elements = elems_per_peer;
    }
    for ( uint32_t p = 0u; p < 16u; ++p )
        for ( uint64_t i = 0u; i < elems_per_peer; ++i )
            f->responder.peer_buf[p][i] = PatternBf16(p, i, salt);
}

static void ComputeHostRsagRef(WireFixture *f, uint64_t elements, uint32_t codec, uint16_t *h_ref)
{
    const uint32_t degree = 16u;
    const uint32_t block = CodecBlock(codec);
    const uint32_t sb = CodecScaleBytes(codec);
    const uint32_t p1q = CodecPhase1Quant(codec);
    uint8_t *qtmp = (uint8_t *)malloc((size_t)(elements + (elements / block) * sb));
    float *acc = (float *)calloc((size_t)elements, sizeof(float));
    for ( uint32_t p = 0u; p < degree; ++p )
    {
        if ( p1q != 0u )
        {
            HostQuantRange(qtmp, elements, f->responder.peer_buf[p], elements, block, sb);
            const uint8_t *sc_src = qtmp + elements;
            for ( uint64_t b = 0u; b < elements / block; ++b )
            {
                float sc = sb == 1u ? HostUe8m0ToFloat(sc_src[b]) : Bf16ToFloat(((const uint16_t *)sc_src)[b]);
                for ( uint32_t j = 0u; j < block; ++j )
                    acc[b * block + j] = fmaf(HostE4m3ToFloat(qtmp[b * block + j]), sc, acc[b * block + j]);
            }
        }
        else
        {
            for ( uint64_t i = 0u; i < elements; ++i )
                acc[i] += Bf16ToFloat(f->responder.peer_buf[p][i]);
        }
    }
    uint16_t *sum_bf16 = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    for ( uint64_t i = 0u; i < elements; ++i ) sum_bf16[i] = FloatToBf16Rn(acc[i]);
    HostQuantRange(qtmp, elements, sum_bf16, elements, block, sb);
    HostDequantRange(h_ref, qtmp, elements, elements, block, sb);
    free(sum_bf16); free(acc); free(qtmp);
}

static void TestRsagCodecExact(WireFixture *f, uint32_t codec, uint64_t elements, uint32_t inplace)
{
    const uint32_t degree = 16u;
    EnsurePeerBufs(f, elements, 7000u + codec * 31u + inplace * 13u);
    uint16_t *h_ref = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    uint16_t *h_out0 = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    uint16_t *h_out7 = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    uint16_t *h_out15 = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    ComputeHostRsagRef(f, elements, codec, h_ref);
    uint32_t ranks[3] = { 0u, 7u, 15u };
    uint16_t *outs[3] = { h_out0, h_out7, h_out15 };
    SparkTpMeshFastSetWireCodec(codec);
    SparkTpMeshFastSetWireCodecAllowed(1u);
    for ( uint32_t r = 0u; r < 3u; ++r )
    {
        FixtureResetEpoch(f, ranks[r], degree, 1u, codec, elements, 1u, 0u, 0u);
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[ranks[r]], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
        void *dst = inplace != 0u ? (void *)f->local_dev : (void *)f->out_dev;
        if ( inplace == 0u ) CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, ranks[r], degree, f->local_dev, dst, f->scratch,
            elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(outs[r], dst, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        SparkTpMeshRoundControl ctrl;
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "MXFP8 RSAG completed cleanly");
        CHECK(memcmp(h_ref, outs[r], (size_t)elements * sizeof(uint16_t)) == 0, "MXFP8 RSAG matches CPU ref bit-for-bit");
    }
    CHECK(memcmp(h_out0, h_out7, (size_t)elements * sizeof(uint16_t)) == 0 &&
          memcmp(h_out0, h_out15, (size_t)elements * sizeof(uint16_t)) == 0,
        "MXFP8 RSAG is bit-identical across TP ranks 0, 7, 15");
    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_BF16);
    free(h_out15); free(h_out7); free(h_out0); free(h_ref);
}

static void TestGatherCodecExact(WireFixture *f, uint32_t codec, uint64_t total_elements)
{
    const uint32_t degree = 16u;
    const uint64_t local_el = total_elements / degree;
    const uint32_t block = CodecBlock(codec);
    const uint32_t sb = CodecScaleBytes(codec);
    EnsurePeerBufs(f, local_el, 8000u + codec * 29u + (uint32_t)(total_elements & 0xffffu));
    uint16_t *h_ref = (uint16_t *)malloc((size_t)total_elements * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc((size_t)total_elements * sizeof(uint16_t));
    uint8_t *qtmp = (uint8_t *)malloc((size_t)(local_el + (local_el / block) * sb));
    for ( uint32_t p = 0u; p < degree; ++p )
    {
        HostQuantRange(qtmp, local_el, f->responder.peer_buf[p], local_el, block, sb);
        HostDequantRange(h_ref + (size_t)p * local_el, qtmp, local_el, local_el, block, sb);
    }
    SparkTpMeshFastSetWireCodec(codec);
    SparkTpMeshFastSetWireCodecAllowed(1u);
    uint32_t ranks[2] = { 0u, 11u };
    for ( uint32_t r = 0u; r < 2u; ++r )
    {
        FixtureResetEpoch(f, ranks[r], degree, SPARK_TP_MESH_OPERATION_ALL_GATHER, codec, total_elements, 1u, 0u, 0u);
        CUDA_OK(cudaMemcpy(f->share_dev, f->responder.peer_buf[ranks[r]], (size_t)local_el * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)total_elements * sizeof(uint16_t)));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, ranks[r], degree, f->share_dev, f->out_dev, f->scratch,
            total_elements, SPARK_TP_MESH_OPERATION_ALL_GATHER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_out, f->out_dev, (size_t)total_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        SparkTpMeshRoundControl ctrl;
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "MXFP8 AllGather completed cleanly");
        CHECK(memcmp(h_ref, h_out, (size_t)total_elements * sizeof(uint16_t)) == 0, "MXFP8 AllGather matches CPU ref bit-for-bit");
    }
    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_BF16);
    free(qtmp); free(h_out); free(h_ref);
}

static void TestScatterCodecExact(WireFixture *f, uint32_t codec, uint64_t total_elements)
{
    const uint32_t degree = 16u;
    const uint64_t segment = total_elements / degree;
    const uint32_t block = CodecBlock(codec);
    const uint32_t sb = CodecScaleBytes(codec);
    EnsurePeerBufs(f, total_elements, 9000u + codec * 19u + (uint32_t)(total_elements & 0xffffu));
    uint16_t *h_ref = (uint16_t *)malloc((size_t)segment * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc((size_t)segment * sizeof(uint16_t));
    uint8_t *qtmp = (uint8_t *)malloc((size_t)(segment + (segment / block) * sb));
    float *acc = (float *)malloc((size_t)segment * sizeof(float));
    SparkTpMeshFastSetWireCodec(codec);
    SparkTpMeshFastSetWireCodecAllowed(1u);
    uint32_t ranks[2] = { 0u, 9u };
    for ( uint32_t r = 0u; r < 2u; ++r )
    {
        uint32_t rank = ranks[r];
        memset(acc, 0, (size_t)segment * sizeof(float));
        for ( uint32_t p = 0u; p < degree; ++p )
        {
            HostQuantRange(qtmp, segment, f->responder.peer_buf[p] + (uint64_t)rank * segment, segment, block, sb);
            const uint8_t *sc_src = qtmp + segment;
            for ( uint64_t b = 0u; b < segment / block; ++b )
            {
                float sc = sb == 1u ? HostUe8m0ToFloat(sc_src[b]) : Bf16ToFloat(((const uint16_t *)sc_src)[b]);
                for ( uint32_t j = 0u; j < block; ++j )
                    acc[b * block + j] = fmaf(HostE4m3ToFloat(qtmp[b * block + j]), sc, acc[b * block + j]);
            }
        }
        for ( uint64_t i = 0u; i < segment; ++i ) h_ref[i] = FloatToBf16Rn(acc[i]);
        FixtureResetEpoch(f, rank, degree, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, codec, total_elements, 1u, 0u, 0u);
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[rank], (size_t)total_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(f->share_dev, 0, (size_t)segment * sizeof(uint16_t)));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->share_dev, f->scratch,
            total_elements, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_out, f->share_dev, (size_t)segment * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        SparkTpMeshRoundControl ctrl;
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "MXFP8 ReduceScatter completed cleanly");
        CHECK(memcmp(h_ref, h_out, (size_t)segment * sizeof(uint16_t)) == 0, "MXFP8 ReduceScatter matches CPU ref bit-for-bit");
    }
    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_BF16);
    free(acc); free(qtmp); free(h_out); free(h_ref);
}

static void TestGateGxRowInvarianceAndSafety(WireFixture *f)
{
    const uint32_t degree = 16u;
    const uint64_t hidden = 4096u;
    const uint64_t big_elems = 1024ull * hidden;
    const uint64_t warm_elems = 128ull * hidden;
    EnsurePeerBufs(f, big_elems, 4242u);
    uint16_t *h_big = (uint16_t *)malloc((size_t)big_elems * sizeof(uint16_t));
    uint16_t *h_warm = (uint16_t *)malloc((size_t)warm_elems * sizeof(uint16_t));
    uint32_t codecs[] = { SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32, SPARK_TP_MESH_WIRE_CODEC_FP8_B32, SPARK_TP_MESH_WIRE_CODEC_FP8_B16, SPARK_TP_MESH_WIRE_CODEC_FP8_B32_AG };
    for ( uint32_t ci = 0u; ci < 4u; ++ci )
    {
        uint32_t codec = codecs[ci];
        SparkTpMeshFastSetWireCodec(codec);
        SparkTpMeshFastSetWireCodecAllowed(1u);
        FixtureResetEpoch(f, 0u, degree, 1u, codec, big_elems, 1u, 0u, 0u);
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[0], (size_t)big_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->local_dev, f->out_dev, f->scratch,
            big_elems, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_big, f->out_dev, (size_t)big_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        FixtureResetEpoch(f, 0u, degree, 1u, codec, warm_elems, 1u, 0u, 0u);
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[0], (size_t)warm_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->local_dev, f->out_dev, f->scratch,
            warm_elems, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_warm, f->out_dev, (size_t)warm_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        CHECK(memcmp(h_big, h_warm, (size_t)warm_elems * sizeof(uint16_t)) == 0,
            "Gate GX row-invariance: rows=128 warm chunk matches rows=1024 cold chunk bit-for-bit");
    }

    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32);
    SparkTpMeshFastSetWireCodecAllowed(1u);
    uint64_t small_sizes[] = { 4096u, 8192u, 8u * 4096u, 64u * 4096u };
    for ( uint32_t s = 0u; s < 4u; ++s )
    {
        uint64_t el = small_sizes[s];
        uint16_t *bf16_ref = (uint16_t *)malloc((size_t)el * sizeof(uint16_t));
        uint16_t *bf16_out = (uint16_t *)malloc((size_t)el * sizeof(uint16_t));
        for ( uint64_t i = 0u; i < el; ++i )
        {
            float acc = 0.0f;
            for ( uint32_t p = 0u; p < degree; ++p ) acc += Bf16ToFloat(f->responder.peer_buf[p][i]);
            bf16_ref[i] = FloatToBf16Rn(acc);
        }
        FixtureResetEpoch(f, 0u, degree, 1u, 0u, el, 1u, 0u, 0u);
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[0], (size_t)el * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->local_dev, f->out_dev, f->scratch,
            el, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(bf16_out, f->out_dev, (size_t)el * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK(memcmp(bf16_ref, bf16_out, (size_t)el * sizeof(uint16_t)) == 0,
            "B1/B8/small batches stay 0-ULP bit-exact BF16 even when SPARK_MESH_WIRE_CODEC=mxfp8");
        free(bf16_out); free(bf16_ref);
    }

    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32);
    SparkTpMeshFastSetWireCodecAllowed(0u);
    {
        uint64_t el = 512ull * 4096u;
        uint16_t *bf16_ref = (uint16_t *)malloc((size_t)el * sizeof(uint16_t));
        uint16_t *bf16_out = (uint16_t *)malloc((size_t)el * sizeof(uint16_t));
        for ( uint64_t i = 0u; i < el; ++i )
        {
            float acc = 0.0f;
            for ( uint32_t p = 0u; p < degree; ++p ) acc += Bf16ToFloat(f->responder.peer_buf[p][i]);
            bf16_ref[i] = FloatToBf16Rn(acc);
        }
        FixtureResetEpoch(f, 0u, degree, 1u, 0u, el, 1u, 0u, 0u);
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[0], (size_t)el * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->local_dev, f->out_dev, f->scratch,
            el, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(bf16_out, f->out_dev, (size_t)el * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK(memcmp(bf16_ref, bf16_out, (size_t)el * sizeof(uint16_t)) == 0,
            "Verify/decode gate (allowed=0) keeps large payloads 0-ULP bit-exact BF16");
        free(bf16_out); free(bf16_ref);
    }

    SparkTpMeshFastSetWireCodecAllowed(1u);
    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32);
    {
        SparkTpMeshRoundControl ctrl;
        FixtureResetEpoch(f, 0u, degree, 1u, SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32, 512ull * 4096u, 0u, 0u, 0u);
        __atomic_store_n(&f->responder.cancel_after_ns, 50000u, __ATOMIC_RELEASE);
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->local_dev, f->out_dev, f->scratch,
            512ull * 4096u, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK((ctrl.error_word & SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED) != 0u && ctrl.rounds_done == 0u,
            "MXFP8 RSAG cancels cleanly with terminal error");

        FixtureResetEpoch(f, 0u, degree, SPARK_TP_MESH_OPERATION_ALL_GATHER, SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32, 4096ull * 8192u, 0u, 0u, 0u);
        __atomic_store_n(&f->responder.drop_peer, 9u, __ATOMIC_RELEASE);
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->share_dev, f->out_dev, f->scratch,
            4096ull * 8192u, SPARK_TP_MESH_OPERATION_ALL_GATHER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u, "MXFP8 pipelined AllGather times out cleanly on dropped peer");
    }
    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_BF16);
    free(h_warm); free(h_big);
}

typedef struct BenchMetrics
{
    double wall_us;
    double copy_us;
    double wait_us;
    double combine_us;
    double wire_mb;
} BenchMetrics;

static BenchMetrics BenchCase(WireFixture *f, uint32_t op, uint64_t elements, uint32_t codec, uint32_t wire_gbps_x10, uint32_t base_lat_ns, uint32_t iters)
{
    const uint32_t rank = 0u, degree = 16u;
    const void *src = op == SPARK_TP_MESH_OPERATION_ALL_GATHER ? (const void *)f->share_dev : (const void *)f->local_dev;
    void *dst = op == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER ? (void *)f->share_dev : (void *)f->out_dev;
    SparkTpMeshFastSetWireCodec(codec);
    SparkTpMeshFastSetWireCodecAllowed(1u);
    for ( uint32_t w = 0u; w < 2u; ++w )
    {
        FixtureResetEpoch(f, rank, degree, op, codec, elements, 0u, wire_gbps_x10, base_lat_ns);
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, rank, degree, src, dst, f->scratch,
            elements, op, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
    }
    FixtureResetEpoch(f, rank, degree, op, codec, elements, 0u, wire_gbps_x10, base_lat_ns);
    uint64_t t0 = NowNs();
    for ( uint32_t i = 0u; i < iters; ++i )
    {
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, rank, degree, src, dst, f->scratch,
            elements, op, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
    }
    CUDA_OK(cudaStreamSynchronize(f->stream));
    uint64_t t1 = NowNs();
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == iters, "MXFP8 wire bench completed cleanly");
    BenchMetrics m;
    m.wall_us = (double)(t1 - t0) / 1000.0 / (double)iters;
    m.copy_us = (double)ctrl.copy_ns / 1000.0 / (double)iters;
    m.wait_us = (double)ctrl.peer_wait_ns / 1000.0 / (double)iters;
    m.combine_us = (double)ctrl.combine_ns / 1000.0 / (double)iters;
    m.wire_mb = (double)__atomic_load_n(&f->responder.wire_bytes_total, __ATOMIC_ACQUIRE) / (1024.0 * 1024.0) / (double)iters;
    SparkTpMeshFastSetWireCodec(SPARK_TP_MESH_WIRE_CODEC_BF16);
    return m;
}

static void RunWireBenchmarks(WireFixture *f)
{
    struct { const char *name; uint32_t op; uint32_t rows; uint32_t hidden; uint32_t iters; } cases[] = {
        { "AR_B512_4096",  1u, 512u,  4096u, 10u },
        { "AR_B512_8192",  1u, 512u,  8192u, 8u },
        { "AR_B1024_8192", 1u, 1024u, 8192u, 6u },
        { "RS_B1024_8192", SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 1024u, 8192u, 8u },
        { "RS_B2048_6144", SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 2048u, 6144u, 6u },
        { "RS_B4096_8192", SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 4096u, 8192u, 4u },
        { "AG_B1024_8192", SPARK_TP_MESH_OPERATION_ALL_GATHER, 1024u, 8192u, 8u },
        { "AG_B2048_6144", SPARK_TP_MESH_OPERATION_ALL_GATHER, 2048u, 6144u, 6u },
        { "AG_B4096_8192", SPARK_TP_MESH_OPERATION_ALL_GATHER, 4096u, 8192u, 4u },
    };
    struct { const char *label; uint32_t gbps_x10; uint32_t lat_ns; } modes[] = {
        { "1-SWITCH 1x100g (11.5 GB/s + 8us)", 115u, 8000u },
        { "2-SWITCH 2x100g (23.0 GB/s + 8us)", 230u, 8000u },
    };
    for ( uint32_t m = 0u; m < 2u; ++m )
    {
        printf("\n=== E9 WIRE COMPRESSION BENCH (%s) ===\n", modes[m].label);
        printf("%-14s | %9s | %9s (%5s) | %9s (%5s) | %9s (%5s) | %9s (%5s)\n",
            "Case", "BF16_us", "MXFP8_us", "Spdup", "FP8B32_us", "Spdup", "FP8B16_us", "Spdup", "AGOnly_us", "Spdup");
        for ( uint32_t c = 0u; c < sizeof(cases) / sizeof(cases[0]); ++c )
        {
            uint64_t elems = (uint64_t)cases[c].rows * cases[c].hidden;
            BenchMetrics b0 = BenchCase(f, cases[c].op, elems, SPARK_TP_MESH_WIRE_CODEC_BF16, modes[m].gbps_x10, modes[m].lat_ns, cases[c].iters);
            BenchMetrics b1 = BenchCase(f, cases[c].op, elems, SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32, modes[m].gbps_x10, modes[m].lat_ns, cases[c].iters);
            BenchMetrics b2 = BenchCase(f, cases[c].op, elems, SPARK_TP_MESH_WIRE_CODEC_FP8_B32, modes[m].gbps_x10, modes[m].lat_ns, cases[c].iters);
            BenchMetrics b3 = BenchCase(f, cases[c].op, elems, SPARK_TP_MESH_WIRE_CODEC_FP8_B16, modes[m].gbps_x10, modes[m].lat_ns, cases[c].iters);
            BenchMetrics b4 = BenchCase(f, cases[c].op, elems, SPARK_TP_MESH_WIRE_CODEC_FP8_B32_AG, modes[m].gbps_x10, modes[m].lat_ns, cases[c].iters);
            printf("%-14s | %7.1fus | %7.1fus (%4.2fx) | %7.1fus (%4.2fx) | %7.1fus (%4.2fx) | %7.1fus (%4.2fx)\n",
                cases[c].name, b0.wall_us,
                b1.wall_us, b0.wall_us / b1.wall_us,
                b2.wall_us, b0.wall_us / b2.wall_us,
                b3.wall_us, b0.wall_us / b3.wall_us,
                b4.wall_us, b0.wall_us / b4.wall_us);
        }
    }
}

int main(void)
{
    InitE4m3Lut();
    WireFixture fixture;
    FixtureInit(&fixture, 4096ull * 8192ull);

    uint32_t rsag_codecs[] = {
        SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32,
        SPARK_TP_MESH_WIRE_CODEC_FP8_B32,
        SPARK_TP_MESH_WIRE_CODEC_FP8_B16,
        SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32_AG,
        SPARK_TP_MESH_WIRE_CODEC_FP8_B32_AG,
        SPARK_TP_MESH_WIRE_CODEC_FP8_B16_AG
    };
    for ( uint32_t i = 0u; i < 6u; ++i )
    {
        TestRsagCodecExact(&fixture, rsag_codecs[i], 128ull * 4096u, i & 1u);
        TestRsagCodecExact(&fixture, rsag_codecs[i], 512ull * 4096u, (i + 1u) & 1u);
    }
    TestRsagCodecExact(&fixture, SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32, 1024ull * 8192u, 0u);
    TestRsagCodecExact(&fixture, SPARK_TP_MESH_WIRE_CODEC_FP8_B32, 1024ull * 8192u, 1u);

    uint32_t ag_codecs[] = {
        SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32,
        SPARK_TP_MESH_WIRE_CODEC_FP8_B32,
        SPARK_TP_MESH_WIRE_CODEC_FP8_B16
    };
    for ( uint32_t i = 0u; i < 3u; ++i )
    {
        TestGatherCodecExact(&fixture, ag_codecs[i], 1024ull * 8192u);
        TestGatherCodecExact(&fixture, ag_codecs[i], 2048ull * 6144u);
        TestGatherCodecExact(&fixture, ag_codecs[i], 4096ull * 4096u);
        TestScatterCodecExact(&fixture, ag_codecs[i], 1024ull * 8192u);
        TestScatterCodecExact(&fixture, ag_codecs[i], 2048ull * 6144u);
        TestScatterCodecExact(&fixture, ag_codecs[i], 4096ull * 4096u);
    }
    TestGatherCodecExact(&fixture, SPARK_TP_MESH_WIRE_CODEC_MXFP8_B32, 4096ull * 8192u);
    TestScatterCodecExact(&fixture, SPARK_TP_MESH_WIRE_CODEC_FP8_B32, 4096ull * 8192u);

    TestGateGxRowInvarianceAndSafety(&fixture);
    RunWireBenchmarks(&fixture);

    FixtureDestroy(&fixture);
    printf("tp_mesh_mxfp8_wire_cuda: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures != 0u ? 1 : 0;
}
