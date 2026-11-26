#include <cuda_runtime.h>
#include <endian.h>
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

static uint16_t PatternBf16(uint32_t peer, uint64_t idx, uint32_t salt)
{
    uint32_t h = (peer + 1u) * 2654435761u ^ ((uint32_t)idx + 7u) * 2246822519u ^ (salt * 3266489917u);
    int32_t sign = (h & 1u) ? -1 : 1;
    float v = (float)sign * (0.015625f + (float)((h >> 8u) & 255u) * 0.00390625f);
    return FloatToBf16Rn(v);
}

typedef struct PeerSource
{
    const uint16_t *data;
    uint64_t total_elements;
    uint64_t slice_stride;
    uint64_t slice_offset;
} PeerSource;

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
    SparkWeightdMeshHybridCtrl *hybrid_ctrl;
    uint8_t *hybrid_shm;
    uint32_t hybrid_pi[16u];
    uint32_t hybrid_wrs[16u];
    volatile uint32_t ship_delay_ns;
    volatile uint32_t revoke_every_n;
    volatile uint64_t hybrid_dma_count;
    volatile uint64_t hybrid_wqe_errors;
    volatile uint64_t cpu_doorbell_count;
    volatile uint64_t revoke_count;
    volatile uint32_t verify_staging_tail;
    volatile uint64_t staging_clobber_errors;
    volatile uint64_t full_staging_checked;
} ResponderState;

static void ResponderServiceGate(ResponderState *st, uint64_t *gate_seen)
{
    uint64_t req_id = __atomic_load_n(&st->gate->request_id, __ATOMIC_ACQUIRE);
    if ( req_id == *gate_seen || req_id == 0u )
        return;
    uint64_t kind = st->gate->kind;
    uint64_t tag = st->gate->tag;
    if ( kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED &&
         __atomic_load_n(st->shipped, __ATOMIC_ACQUIRE) < tag )
        return;
    *gate_seen = req_id;
    st->gate->error = 0u;
    st->gate->diag = 0u;
    __atomic_store_n(&st->gate->ready, 1u, __ATOMIC_RELEASE);
}

static uint16_t Sum2d4x4Bf16(uint16_t *const peer_buf[16u], uint64_t idx)
{
    float col_acc = 0.0f;
    for ( uint32_t r = 0u; r < 4u; ++r )
    {
        float row_acc = 0.0f;
        for ( uint32_t c = 0u; c < 4u; ++c )
            row_acc += Bf16ToFloat(peer_buf[(r << 2u) | c][idx]);
        col_acc += Bf16ToFloat(FloatToBf16Rn(row_acc));
    }
    return FloatToBf16Rn(col_acc);
}

static void ResponderDeliverRoute(ResponderState *st, uint64_t tag, uint64_t bytes, SparkWeightdMeshRoute route)
{
    uint64_t ring = (tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u);
    uint32_t half = route.fields.half;
    uint64_t half_off = SPARK_WEIGHTD_MESH_HALF_OFFSET(half);
    uint64_t tail_off = route.fields.compact_tail != 0u ? bytes :
        (half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ?
            SPARK_WEIGHTD_MESH_TAIL_OFFSET(half) : (SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u));
    uint64_t t0 = NowNs();
    SparkWeightdMeshHybridCtrl *hctrl = __atomic_load_n(&st->hybrid_ctrl, __ATOMIC_ACQUIRE);
    if ( hctrl != 0 )
        __atomic_store_n(&hctrl->cpu_issued_seq, tag, __ATOMIC_RELEASE);
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
    uint32_t is_2d = st->op_kind == 1u && st->degree == 16u && __builtin_popcount(mask) == 3 ? 1u : 0u;
    uint64_t wire_per_peer_ns = 0u;
    uint64_t xfer_bytes = route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER ? route.fields.slice_bytes : bytes;
    if ( st->wire_gbps_x10 != 0u )
        wire_per_peer_ns = (xfer_bytes * 10ull) / (uint64_t)st->wire_gbps_x10;
    uint32_t sent = 0u;
    for ( uint32_t k = 0u; k < 16u; ++k )
    {
        uint32_t p = st->order[k];
        if ( p >= st->degree || p == st->rank )
            continue;
        if ( (mask & (1u << p)) == 0u )
            continue;
        if ( drop == p )
            continue;
        sent++;
        if ( st->wire_gbps_x10 != 0u )
        {
            uint64_t target_ns = (uint64_t)st->base_lat_ns + (uint64_t)sent * wire_per_peer_ns;
            while ( NowNs() - t0 < target_ns ) {}
        }
        uint8_t *dst = st->band + ((uint64_t)p * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring) * SPARK_WEIGHTD_MESH_SLOT_BYTES + half_off;
        uint64_t p_tail_off = tail_off;
        if ( is_2d != 0u )
        {
            const uint64_t cap2d = SPARK_TP_MESH_FAST_2D_CAP_ELEMENTS;
            const uint64_t chunks_2d = st->total_elements > 0u ? ((st->total_elements + cap2d - 1u) / cap2d) : 1u;
            const uint64_t cidx = (st->chunk_idx / 4u) % chunks_2d;
            const uint64_t cbegin = cidx * cap2d;
            const uint64_t ccount = st->total_elements > cbegin ? (st->total_elements - cbegin < cap2d ? st->total_elements - cbegin : cap2d) : 0u;
            const uint64_t sub_slice = ((ccount + 15u) / 16u + 7u) & ~UINT64_C(7);
            const uint64_t col_slice = 4u * sub_slice;
            const uint32_t row = st->rank >> 2u, col = st->rank & 3u;
            const uint32_t row_mask = (0xfu << (row << 2u)) & ~(1u << st->rank);
            const uint32_t is_row = mask == row_mask ? 1u : 0u;
            uint64_t first = 0u, len = 0u;
            if ( route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER )
            {
                if ( is_row != 0u )
                {
                    first = (uint64_t)col * col_slice;
                    len = ccount > first ? (ccount - first < col_slice ? ccount - first : col_slice) : 0u;
                    if ( st->copy_payload != 0u && st->peer_buf[p] != 0 && len != 0u )
                        memcpy(dst, st->peer_buf[p] + cbegin + first, (size_t)(len * sizeof(uint16_t)));
                }
                else
                {
                    const uint32_t p_row = p >> 2u;
                    first = (uint64_t)col * col_slice + (uint64_t)row * sub_slice;
                    len = ccount > first ? (ccount - first < sub_slice ? ccount - first : sub_slice) : 0u;
                    if ( st->copy_payload != 0u && st->peer_buf[p] != 0 && len != 0u )
                    {
                        uint16_t *out16 = (uint16_t *)dst;
                        for ( uint64_t i = 0u; i < len; ++i )
                        {
                            float acc = 0.0f;
                            for ( uint32_t c = 0u; c < 4u; ++c )
                                acc += Bf16ToFloat(st->peer_buf[(p_row << 2u) | c][cbegin + first + i]);
                            out16[i] = FloatToBf16Rn(acc);
                        }
                    }
                }
            }
            else
            {
                if ( is_row == 0u )
                {
                    const uint32_t p_row = p >> 2u;
                    first = (uint64_t)col * col_slice + (uint64_t)p_row * sub_slice;
                    len = ccount > first ? (ccount - first < sub_slice ? ccount - first : sub_slice) : 0u;
                }
                else
                {
                    const uint32_t p_col = p & 3u;
                    first = (uint64_t)p_col * col_slice;
                    len = ccount > first ? (ccount - first < col_slice ? ccount - first : col_slice) : 0u;
                }
                if ( route.fields.compact_tail != 0u )
                {
                    uint64_t gb = (len * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
                    p_tail_off = gb != 0u ? gb : 8u;
                }
                if ( st->copy_payload != 0u && st->peer_buf[p] != 0 && len != 0u )
                {
                    uint16_t *out16 = (uint16_t *)dst;
                    for ( uint64_t i = 0u; i < len; ++i )
                        out16[i] = Sum2d4x4Bf16(st->peer_buf, cbegin + first + i);
                }
            }
            __atomic_store_n((volatile uint64_t *)(st->band + ((uint64_t)p * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring) * SPARK_WEIGHTD_MESH_SLOT_BYTES + p_tail_off),
                tag, __ATOMIC_RELEASE);
            continue;
        }
        if ( st->copy_payload != 0u && st->peer_buf[p] != 0 )
        {
            if ( st->op_kind == 1u )
            {
                if ( route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER )
                {
                    uint64_t slice = route.fields.slice_bytes / sizeof(uint16_t);
                    uint64_t first = (uint64_t)st->rank * slice;
                    uint64_t len = st->total_elements > first ? st->total_elements - first : 0u;
                    if ( len > slice ) len = slice;
                    if ( len != 0u )
                        memcpy(dst, st->peer_buf[p] + first, (size_t)(len * sizeof(uint16_t)));
                }
                else if ( SparkTpMeshDirectPhasesPerChunk(st->total_elements, st->degree, 1u,
                              SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER) == 1u )
                {
                    memcpy(dst, st->peer_buf[p], (size_t)(st->total_elements * sizeof(uint16_t)));
                }
                else
                {
                    uint64_t slice = SparkTpMeshRsagSlice(st->total_elements, st->degree);
                    uint64_t first = (uint64_t)p * slice;
                    uint64_t len = st->total_elements > first ? st->total_elements - first : 0u;
                    if ( len > slice ) len = slice;
                    uint16_t *out16 = (uint16_t *)dst;
                    for ( uint64_t i = 0u; i < len; ++i )
                    {
                        float acc = 0.0f;
                        for ( uint32_t q = 0u; q < st->degree; ++q )
                            acc += Bf16ToFloat(st->peer_buf[q][first + i]);
                        out16[i] = FloatToBf16Rn(acc);
                    }
                }
            }
            else if ( st->op_kind == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER )
            {
                uint64_t segment = st->total_elements / st->degree;
                uint64_t width = half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ?
                    (SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t)) : segment;
                uint64_t begin = (uint64_t)st->chunk_idx * width;
                uint64_t count = begin < segment ? (segment - begin < width ? segment - begin : width) : 0u;
                if ( count != 0u )
                    memcpy(dst, st->peer_buf[p] + (uint64_t)st->rank * segment + begin, (size_t)(count * sizeof(uint16_t)));
            }
            else if ( st->op_kind == SPARK_TP_MESH_OPERATION_ALL_GATHER )
            {
                uint64_t local_el = st->total_elements / st->degree;
                uint64_t width = half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ?
                    (SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t)) : local_el;
                uint64_t begin = (uint64_t)st->chunk_idx * width;
                uint64_t count = begin < local_el ? (local_el - begin < width ? local_el - begin : width) : 0u;
                if ( count != 0u )
                    memcpy(dst, st->peer_buf[p] + begin, (size_t)(count * sizeof(uint16_t)));
            }
            else if ( st->op_kind == SPARK_TP_MESH_OPERATION_ALL_TO_ALL )
            {
                uint64_t per_peer = st->total_elements;
                uint64_t width = half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ?
                    (SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t)) : per_peer;
                uint64_t begin = (uint64_t)st->chunk_idx * width;
                uint64_t count = begin < per_peer ? (per_peer - begin < width ? per_peer - begin : width) : 0u;
                if ( count != 0u )
                    memcpy(dst, st->peer_buf[p] + (uint64_t)st->rank * per_peer + begin, (size_t)(count * sizeof(uint16_t)));
            }
        }
        if ( st->op_kind == 1u && route.fields.compact_tail != 0u && route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_FULL &&
             SparkTpMeshDirectPhasesPerChunk(st->total_elements, st->degree, 1u,
                 SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER) == 2u )
        {
            uint64_t slice = SparkTpMeshRsagSlice(st->total_elements, st->degree);
            uint64_t first = (uint64_t)p * slice;
            uint64_t len = st->total_elements > first ? st->total_elements - first : 0u;
            if ( len > slice ) len = slice;
            uint64_t gb = (len * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
            p_tail_off = gb != 0u ? gb : 8u;
        }
        __atomic_store_n((volatile uint64_t *)(st->band + ((uint64_t)p * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring) * SPARK_WEIGHTD_MESH_SLOT_BYTES + p_tail_off),
            tag, __ATOMIC_RELEASE);
    }
    uint32_t rev_n = __atomic_load_n(&st->revoke_every_n, __ATOMIC_ACQUIRE);
    if ( hctrl != 0 && rev_n != 0u && ((st->chunk_idx + 1u) % rev_n) == 0u )
    {
        __atomic_store_n(&hctrl->cpu_mode, 0u, __ATOMIC_SEQ_CST);
        while ( __atomic_load_n(&hctrl->gpu_busy, __ATOMIC_ACQUIRE) != 0u ) {}
        memset(hctrl->trigger, 0, sizeof(hctrl->trigger));
        for ( uint32_t pi_i = 0u; pi_i < hctrl->peer_count; ++pi_i )
        {
            hctrl->gpu_pi[pi_i] = st->hybrid_pi[pi_i];
            hctrl->gpu_wrs[pi_i] = st->hybrid_wrs[pi_i];
            hctrl->cpu_sq_limit[pi_i] = st->hybrid_pi[pi_i] + hctrl->peers[pi_i].sq_depth;
            hctrl->cpu_wr_limit[pi_i] = st->hybrid_wrs[pi_i] + 64u;
        }
        __atomic_store_n(&hctrl->cpu_issued_seq, tag, __ATOMIC_RELEASE);
        __atomic_store_n(&hctrl->grant_gen, hctrl->grant_gen + 1u, __ATOMIC_RELEASE);
        __atomic_store_n(&hctrl->cpu_mode, 1u, __ATOMIC_SEQ_CST);
        __atomic_fetch_add(&st->revoke_count, 1ull, __ATOMIC_RELEASE);
    }
    uint32_t sdelay = __atomic_load_n(&st->ship_delay_ns, __ATOMIC_ACQUIRE);
    if ( sdelay != 0u )
    {
        uint64_t d0 = NowNs();
        while ( NowNs() - d0 < (uint64_t)sdelay ) {}
    }
    if ( __atomic_load_n(&st->verify_staging_tail, __ATOMIC_ACQUIRE) != 0u &&
         route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER &&
         route.fields.compact_tail != 0u &&
         bytes + 8u > SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES )
    {
        __atomic_fetch_add(&st->full_staging_checked, 1ull, __ATOMIC_RELEASE);
        for ( uint32_t p = 0u; p < st->degree; ++p )
        {
            if ( p == st->rank || (mask & (1u << p)) == 0u )
                continue;
            uint64_t stg_tail = __atomic_load_n(
                (const volatile uint64_t *)(st->staging + (uint64_t)p * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES + bytes),
                __ATOMIC_ACQUIRE);
            if ( stg_tail != tag )
                __atomic_fetch_add(&st->staging_clobber_errors, 1ull, __ATOMIC_RELEASE);
        }
    }
    __atomic_store_n(st->shipped, tag, __ATOMIC_RELEASE);
    st->chunk_idx++;
    __atomic_fetch_add(&st->rounds_serviced, 1ull, __ATOMIC_RELEASE);
}

static void ResponderDeliverTag(ResponderState *st, volatile uint64_t *cell, uint64_t tag)
{
    uint64_t bytes = cell[1];
    SparkWeightdMeshRoute route;
    route.word = cell[3];
    __atomic_fetch_add(&st->cpu_doorbell_count, 1ull, __ATOMIC_RELEASE);
    ResponderDeliverRoute(st, tag, bytes, route);
}

static uint32_t ResponderServiceHybrid(ResponderState *st, uint64_t *seen)
{
    SparkWeightdMeshHybridCtrl *ctrl = __atomic_load_n(&st->hybrid_ctrl, __ATOMIC_ACQUIRE);
    if ( ctrl == 0 || st->hybrid_shm == 0 )
        return 0u;
    uint64_t s0 = __atomic_load_n(&ctrl->trigger[0].seq, __ATOMIC_ACQUIRE);
    uint64_t s1 = __atomic_load_n(&ctrl->trigger[1].seq, __ATOMIC_ACQUIRE);
    if ( s1 > *seen )
    {
        uint64_t r0 = __atomic_load_n(&ctrl->trigger[0].seq, __ATOMIC_ACQUIRE);
        if ( r0 > s0 ) s0 = r0;
    }
    uint64_t want = *seen == 0u ? 0u : (*seen + 1u);
    uint32_t slot_idx = UINT32_MAX;
    uint64_t tag = 0u;
    if ( want == 0u )
    {
        if ( s0 != 0u && (s1 == 0u || s0 < s1) ) { slot_idx = 0u; tag = s0; }
        else if ( s1 != 0u ) { slot_idx = 1u; tag = s1; }
    }
    else
    {
        if ( s0 == want ) { slot_idx = 0u; tag = s0; }
        else if ( s1 == want ) { slot_idx = 1u; tag = s1; }
    }
    if ( slot_idx == UINT32_MAX )
        return 0u;
    uint64_t meta = __atomic_load_n(&ctrl->trigger[slot_idx].meta, __ATOMIC_ACQUIRE);
    uint64_t slot = __atomic_load_n(&ctrl->trigger[slot_idx].slot, __ATOMIC_ACQUIRE);
    uint64_t bytes = __atomic_load_n(&ctrl->trigger[slot_idx].bytes, __ATOMIC_ACQUIRE);
    if ( __atomic_load_n(&ctrl->trigger[slot_idx].seq, __ATOMIC_ACQUIRE) != tag )
        return 0u;
    uint32_t peer_mask = SparkWeightdMeshHybridMetaPeerMask(meta);
    uint32_t ds = SparkWeightdMeshHybridMetaDs(meta);
    uint32_t num_bb = SparkWeightdMeshHybridMetaNumBb(meta);
    uint32_t is_peer = 0u, saw = 0u;
    uint64_t slot_base = slot * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    if ( ds != 3u || num_bb != 1u || bytes == 0u || bytes > SPARK_WEIGHTD_MESH_SLOT_BYTES - 16u )
        __atomic_fetch_add(&st->hybrid_wqe_errors, 1ull, __ATOMIC_RELEASE);
    for ( uint32_t i = 0u; i < ctrl->peer_count; ++i )
    {
        const SparkWeightdMeshHybridPeer *hp = &ctrl->peers[i];
        if ( (peer_mask & (1u << hp->target)) == 0u )
            continue;
        uint32_t pi = st->hybrid_pi[i];
        uint32_t sq_idx = pi & hp->sq_mask;
        const uint8_t *wqe = st->hybrid_shm + hp->sq_shm_off + (size_t)sq_idx * 64u;
        uint32_t w0, w1, rkey, len, lkey;
        uint64_t raddr;
        memcpy(&w0, wqe + 0u, 4u);
        memcpy(&w1, wqe + 4u, 4u);
        memcpy(&raddr, wqe + 16u, 8u);
        memcpy(&rkey, wqe + 24u, 4u);
        memcpy(&len, wqe + 32u, 4u);
        memcpy(&lkey, wqe + 36u, 4u);
        uint32_t exp_w0 = htobe32(((uint32_t)(uint16_t)pi << 8u) | 0x08u);
        uint32_t exp_w1 = htobe32(hp->qpn_shifted | ds);
        if ( w0 != exp_w0 || w1 != exp_w1 || raddr != htobe64(hp->remote_base + slot_base) ||
             rkey != hp->rkey_be || len != htobe32((uint32_t)(bytes + 8u)) ||
             (lkey != hp->recv_lkey_be && lkey != hp->staging_lkey_be) )
            __atomic_fetch_add(&st->hybrid_wqe_errors, 1ull, __ATOMIC_RELEASE);
        if ( saw == 0u )
        {
            is_peer = lkey == hp->staging_lkey_be ? 1u : 0u;
            saw = 1u;
        }
        st->hybrid_pi[i] = pi + num_bb;
        st->hybrid_wrs[i]++;
        __atomic_store_n(&ctrl->cpu_sq_limit[i], st->hybrid_pi[i] + hp->sq_depth, __ATOMIC_RELEASE);
        __atomic_store_n(&ctrl->cpu_wr_limit[i], st->hybrid_wrs[i] + 64u, __ATOMIC_RELEASE);
    }
    SparkWeightdMeshRoute route;
    route.word = 0u;
    route.fields.peer_mask = peer_mask;
    route.fields.mode = is_peer != 0u ? SPARK_WEIGHTD_MESH_ROUTE_PEER : SPARK_WEIGHTD_MESH_ROUTE_FULL;
    route.fields.compact_tail = 1u;
    __atomic_fetch_add(&st->hybrid_dma_count, 1ull, __ATOMIC_RELEASE);
    *seen = tag;
    ResponderDeliverRoute(st, tag, bytes, route);
    return 1u;
}

static void *ResponderMain(void *arg)
{
    ResponderState *st = (ResponderState *)arg;
    uint64_t seen = 0u;
    uint64_t gate_seen = 0u;
    while ( __atomic_load_n(&st->stop, __ATOMIC_ACQUIRE) == 0u )
    {
        if ( __atomic_load_n(&st->reset_req, __ATOMIC_ACQUIRE) != 0u )
        {
            seen = 0u;
            gate_seen = 0u;
            st->chunk_idx = 0u;
            __atomic_store_n(&st->reset_req, 0u, __ATOMIC_RELEASE);
        }
        ResponderServiceGate(st, &gate_seen);
        if ( ResponderServiceHybrid(st, &seen) != 0u )
            continue;
        uint64_t t_pri = __atomic_load_n(&st->doorbell[0], __ATOMIC_ACQUIRE);
        uint64_t t_alt = __atomic_load_n(&st->doorbell_alt[0], __ATOMIC_ACQUIRE);
        uint64_t want = seen == 0u ? 0u : seen + 1u;
        volatile uint64_t *picked = 0;
        uint64_t tag = 0u;
        if ( want == 0u )
        {
            if ( t_pri != 0u && (t_alt == 0u || t_pri < t_alt) )
            {
                picked = st->doorbell;
                tag = t_pri;
            }
            else if ( t_alt != 0u )
            {
                picked = st->doorbell_alt;
                tag = t_alt;
            }
        }
        else
        {
            if ( t_pri == want )
            {
                picked = st->doorbell;
                tag = t_pri;
            }
            else if ( t_alt == want )
            {
                picked = st->doorbell_alt;
                tag = t_alt;
            }
        }
        if ( picked == 0 )
            continue;
        seen = tag;
        ResponderDeliverTag(st, picked, tag);
    }
    return 0;
}

typedef struct LargeFixture
{
    uint8_t *band_host;
    uint8_t *band_dev;
    uint8_t *staging_host;
    uint8_t *staging_dev;
    uint8_t *doorbell_page_host;
    uint8_t *doorbell_page_dev;
    uint64_t *doorbell_host;
    uint64_t *doorbell_dev;
    uint64_t *doorbell_alt_host;
    uint64_t *shipped_host;
    uint64_t *shipped_dev;
    uint64_t *cancel_host;
    uint64_t *cancel_dev;
    SparkWeightdMeshWaitRequest *gate_host;
    SparkWeightdMeshWaitRequest *gate_dev;
    SparkTpMeshRoundControl *control;
    void *scratch;
    uint16_t *local_dev;
    uint16_t *out_dev;
    uint16_t *share_dev;
    uint64_t max_elements;
    cudaStream_t stream;
    ResponderState responder;
    pthread_t thread;
    uint64_t epoch;
} LargeFixture;

static void FixtureInit(LargeFixture *f, uint64_t max_elements)
{
    memset(f, 0, sizeof(*f));
    f->max_elements = max_elements;
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
    uint32_t perm[16u] = { 15u, 2u, 11u, 0u, 9u, 5u, 14u, 3u, 8u, 1u, 13u, 6u, 10u, 4u, 12u, 7u };
    for ( uint32_t p = 0u; p < 16u; ++p )
        f->responder.order[p] = perm[p];
    f->responder.band = f->band_host;
    f->responder.staging = f->staging_host;
    f->responder.doorbell = f->doorbell_host;
    f->responder.doorbell_alt = f->doorbell_alt_host;
    f->responder.shipped = f->shipped_host;
    f->responder.cancel = f->cancel_host;
    f->responder.gate = f->gate_host;
    f->responder.drop_peer = UINT32_MAX;
    f->epoch = 100u;
    if ( pthread_create(&f->thread, 0, ResponderMain, &f->responder) != 0 )
    {
        fprintf(stderr, "pthread_create failed\n");
        exit(1);
    }
}

static void FixtureDestroy(LargeFixture *f)
{
    __atomic_store_n(&f->responder.stop, 1u, __ATOMIC_RELEASE);
    pthread_join(f->thread, 0);
    for ( uint32_t p = 0u; p < 16u; ++p )
        free(f->responder.peer_buf[p]);
    cudaStreamDestroy(f->stream);
    cudaFree(f->share_dev);
    cudaFree(f->out_dev);
    cudaFree(f->local_dev);
    cudaFree(f->scratch);
    cudaFree(f->control);
    cudaFreeHost(f->gate_host);
    cudaFreeHost(f->cancel_host);
    cudaFreeHost(f->shipped_host);
    cudaFreeHost(f->doorbell_page_host);
    cudaFreeHost(f->staging_host);
    cudaFreeHost(f->band_host);
}

static void FixtureResetEpoch(LargeFixture *f, uint32_t rank, uint32_t degree, uint32_t op_kind,
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
    f->responder.total_elements = total_elements;
    f->responder.copy_payload = copy_payload;
    f->responder.wire_gbps_x10 = wire_gbps_x10;
    f->responder.base_lat_ns = base_lat_ns;
    __atomic_store_n(&f->responder.cancel_after_ns, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.drop_peer, UINT32_MAX, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.rounds_serviced, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.ship_delay_ns, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.revoke_every_n, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.hybrid_dma_count, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.hybrid_wqe_errors, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.cpu_doorbell_count, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.revoke_count, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.verify_staging_tail, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.staging_clobber_errors, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.full_staging_checked, 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&f->responder.reset_req, 1u, __ATOMIC_RELEASE);
    while ( __atomic_load_n(&f->responder.reset_req, __ATOMIC_ACQUIRE) != 0u ) {}
    CUDA_OK(cudaMemcpyAsync(f->control, &host, sizeof(host), cudaMemcpyHostToDevice, f->stream));
    CUDA_OK(cudaStreamSynchronize(f->stream));
}

static void EnsurePeerBufs(LargeFixture *f, uint64_t elems_per_peer, uint32_t salt)
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

static void TestRsagExact(LargeFixture *f, uint32_t rank, uint64_t elements, uint32_t inplace)
{
    const uint32_t degree = 16u;
    const uint32_t expect_2d = elements >= SPARK_TP_MESH_FAST_2D_MIN_ELEMENTS ? 1u : 0u;
    EnsurePeerBufs(f, elements, 1000u + rank + (uint32_t)(elements & 0xffffu) + inplace * 17u);
    uint16_t *h_ref = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    for ( uint64_t i = 0u; i < elements; ++i )
    {
        if ( expect_2d != 0u )
            h_ref[i] = Sum2d4x4Bf16(f->responder.peer_buf, i);
        else
        {
            float acc = 0.0f;
            for ( uint32_t p = 0u; p < degree; ++p )
                acc += Bf16ToFloat(f->responder.peer_buf[p][i]);
            h_ref[i] = FloatToBf16Rn(acc);
        }
    }
    FixtureResetEpoch(f, rank, degree, 1u, elements, 1u, 0u, 0u);
    if ( expect_2d != 0u )
        f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
    CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[rank], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    void *dst = inplace != 0u ? (void *)f->local_dev : (void *)f->out_dev;
    if ( inplace == 0u )
        CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, dst, f->scratch,
        elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_out, dst, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "RSAG large exact completed cleanly");
    CHECK(memcmp(h_ref, h_out, (size_t)elements * sizeof(uint16_t)) == 0, "RSAG large output is 0-ULP bitwise exact");
    if ( expect_2d != 0u )
    {
        const uint64_t cap2d = SPARK_TP_MESH_FAST_2D_CAP_ELEMENTS;
        const uint64_t chunks = (elements + cap2d - 1u) / cap2d;
        CHECK(__atomic_load_n(&f->responder.rounds_serviced, __ATOMIC_ACQUIRE) == 4u * chunks,
            "2D 4x4 RS+AG executes 4 3-peer hops per chunk");
    }
    free(h_out);
    free(h_ref);
}

static void TestLargeCancelAndTimeout(LargeFixture *f)
{
    const uint32_t rank = 0u, degree = 16u;
    SparkTpMeshRoundControl ctrl;
    FixtureResetEpoch(f, rank, degree, 1u, 64ull * 6144ull, 0u, 0u, 0u);
    __atomic_store_n(&f->responder.cancel_after_ns, 50000u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_dev, f->scratch,
        64ull * 6144ull, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK((ctrl.error_word & SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED) != 0u && ctrl.rounds_done == 0u,
        "large RSAG cancels cleanly");

    FixtureResetEpoch(f, rank, degree, SPARK_TP_MESH_OPERATION_ALL_GATHER, 2048ull * 6144ull, 0u, 0u, 0u);
    __atomic_store_n(&f->responder.drop_peer, 9u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->share_dev, f->out_dev, f->scratch,
        2048ull * 6144ull, SPARK_TP_MESH_OPERATION_ALL_GATHER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u, "pipelined AllGather times out cleanly on dropped peer");
}

static void TestReduceScatterExact(LargeFixture *f, uint32_t rank, uint64_t elements)
{
    const uint32_t degree = 16u;
    const uint64_t segment = elements / degree;
    EnsurePeerBufs(f, elements, 2000u + rank + (uint32_t)(elements & 0xffffu));
    uint16_t *h_ref = (uint16_t *)malloc((size_t)segment * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc((size_t)segment * sizeof(uint16_t));
    for ( uint64_t i = 0u; i < segment; ++i )
    {
        float acc = 0.0f;
        for ( uint32_t p = 0u; p < degree; ++p )
            acc += Bf16ToFloat(f->responder.peer_buf[p][(uint64_t)rank * segment + i]);
        h_ref[i] = FloatToBf16Rn(acc);
    }
    FixtureResetEpoch(f, rank, degree, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, elements, 1u, 0u, 0u);
    CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[rank], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->share_dev, 0, (size_t)segment * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->share_dev, f->scratch,
        elements, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_out, f->share_dev, (size_t)segment * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "ReduceScatter large exact completed cleanly");
    CHECK(memcmp(h_ref, h_out, (size_t)segment * sizeof(uint16_t)) == 0, "ReduceScatter large output is 0-ULP bitwise exact");
    free(h_out);
    free(h_ref);
}

static void TestAllGatherExactDeg(LargeFixture *f, uint32_t rank, uint32_t degree, uint64_t total_elements)
{
    const uint64_t local_el = total_elements / degree;
    EnsurePeerBufs(f, local_el, 3000u + rank + degree * 31u + (uint32_t)(total_elements & 0xffffu));
    uint16_t *h_ref = (uint16_t *)malloc((size_t)total_elements * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc((size_t)total_elements * sizeof(uint16_t));
    for ( uint32_t p = 0u; p < degree; ++p )
        memcpy(h_ref + (size_t)p * local_el, f->responder.peer_buf[p], (size_t)local_el * sizeof(uint16_t));
    FixtureResetEpoch(f, rank, degree, SPARK_TP_MESH_OPERATION_ALL_GATHER, total_elements, 1u, 0u, 0u);
    CUDA_OK(cudaMemcpy(f->share_dev, f->responder.peer_buf[rank], (size_t)local_el * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)total_elements * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->share_dev, f->out_dev, f->scratch,
        total_elements, SPARK_TP_MESH_OPERATION_ALL_GATHER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_out, f->out_dev, (size_t)total_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "AllGather large exact completed cleanly");
    CHECK(memcmp(h_ref, h_out, (size_t)total_elements * sizeof(uint16_t)) == 0, "AllGather large output is 0-ULP bitwise exact");
    free(h_out);
    free(h_ref);
}

static void TestAllGatherExact(LargeFixture *f, uint32_t rank, uint64_t total_elements)
{
    TestAllGatherExactDeg(f, rank, 16u, total_elements);
}

static void TestRsagExactDeg(LargeFixture *f, uint32_t rank, uint32_t degree, uint64_t elements, uint32_t inplace)
{
    EnsurePeerBufs(f, elements, 4000u + rank + degree * 37u + (uint32_t)(elements & 0xffffu) + inplace * 17u);
    uint16_t *h_ref = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
    for ( uint64_t i = 0u; i < elements; ++i )
    {
        float acc = 0.0f;
        for ( uint32_t p = 0u; p < degree; ++p )
            acc += Bf16ToFloat(f->responder.peer_buf[p][i]);
        h_ref[i] = FloatToBf16Rn(acc);
    }
    FixtureResetEpoch(f, rank, degree, 1u, elements, 1u, 0u, 0u);
    CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[rank], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    void *dst = inplace != 0u ? (void *)f->local_dev : (void *)f->out_dev;
    if ( inplace == 0u )
        CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, dst, f->scratch,
        elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_out, dst, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "e14-r2 AllReduce exact completed cleanly");
    CHECK(memcmp(h_ref, h_out, (size_t)elements * sizeof(uint16_t)) == 0, "e14-r2 AllReduce output is 0-ULP bitwise exact");
    free(h_out);
    free(h_ref);
}

static void TestReduceScatterExactDeg(LargeFixture *f, uint32_t rank, uint32_t degree, uint64_t elements)
{
    const uint64_t segment = elements / degree;
    EnsurePeerBufs(f, elements, 5000u + rank + degree * 41u + (uint32_t)(elements & 0xffffu));
    uint16_t *h_ref = (uint16_t *)malloc((size_t)segment * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc((size_t)segment * sizeof(uint16_t));
    for ( uint64_t i = 0u; i < segment; ++i )
    {
        float acc = 0.0f;
        for ( uint32_t p = 0u; p < degree; ++p )
            acc += Bf16ToFloat(f->responder.peer_buf[p][(uint64_t)rank * segment + i]);
        h_ref[i] = FloatToBf16Rn(acc);
    }
    FixtureResetEpoch(f, rank, degree, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, elements, 1u, 0u, 0u);
    CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[rank], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->share_dev, 0, (size_t)segment * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->share_dev, f->scratch,
        elements, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_out, f->share_dev, (size_t)segment * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SparkTpMeshRoundControl ctrl;
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 1u, "e14-r2 ReduceScatter exact completed cleanly");
    CHECK(memcmp(h_ref, h_out, (size_t)segment * sizeof(uint16_t)) == 0, "e14-r2 ReduceScatter output is 0-ULP bitwise exact");
    free(h_out);
    free(h_ref);
}

static void TestE14R2Degree4AndScatterFastPaths(LargeFixture *f)
{
    uint32_t deg4_ranks[] = { 0u, 3u };
    for ( uint32_t r = 0u; r < 2u; ++r )
    {
        TestRsagExactDeg(f, deg4_ranks[r], 4u, 5120u, 0u);
        TestRsagExactDeg(f, deg4_ranks[r], 4u, 20480u, 0u);
        TestRsagExactDeg(f, deg4_ranks[r], 4u, 40960u, 1u);
        TestRsagExactDeg(f, deg4_ranks[r], 4u, 81920u, 0u);
        TestRsagExactDeg(f, deg4_ranks[r], 4u, 163840u, 1u);
        TestReduceScatterExactDeg(f, deg4_ranks[r], 4u, 20480u);
        TestReduceScatterExactDeg(f, deg4_ranks[r], 4u, 40960u);
        TestReduceScatterExactDeg(f, deg4_ranks[r], 4u, 81920u);
        TestReduceScatterExactDeg(f, deg4_ranks[r], 4u, 163840u);
        TestAllGatherExactDeg(f, deg4_ranks[r], 4u, 20480u);
        TestAllGatherExactDeg(f, deg4_ranks[r], 4u, 40960u);
        TestAllGatherExactDeg(f, deg4_ranks[r], 4u, 81920u);
        TestAllGatherExactDeg(f, deg4_ranks[r], 4u, 327680u);
    }
    TestReduceScatterExactDeg(f, 0u, 16u, 20480u);
    TestReduceScatterExactDeg(f, 7u, 16u, 40960u);
    TestReduceScatterExactDeg(f, 15u, 16u, 163840u);

    SparkTpMeshRoundControl ctrl;
    FixtureResetEpoch(f, 0u, 4u, 1u, 40960u, 0u, 0u, 0u);
    __atomic_store_n(&f->responder.cancel_after_ns, 50000u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, 0u, 4u, f->local_dev, f->out_dev, f->scratch,
        40960u, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 2000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK((ctrl.error_word & SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED) != 0u && ctrl.rounds_done == 0u,
        "e14-r2 deg4 wide256 stream allreduce cancels cleanly");

    FixtureResetEpoch(f, 0u, 4u, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 81920u, 0u, 0u, 0u);
    __atomic_store_n(&f->responder.drop_peer, 2u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, 0u, 4u, f->local_dev, f->share_dev, f->scratch,
        81920u, SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u,
        "e14-r2 deg4 fast reduce-scatter times out cleanly on dropped peer");

    TestReduceScatterExactDeg(f, 0u, 4u, 81920u);
}

typedef struct BenchMetrics
{
    double wall_us;
    double copy_us;
    double wait_us;
    double combine_us;
} BenchMetrics;

static BenchMetrics BenchOne(LargeFixture *f, uint32_t op, uint64_t elements, uint32_t wire_gbps_x10, uint32_t base_lat_ns, uint32_t iters)
{
    const uint32_t rank = 0u, degree = 16u;
    const void *src = op == SPARK_TP_MESH_OPERATION_ALL_GATHER ? (const void *)f->share_dev : (const void *)f->local_dev;
    void *dst = op == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER ? (void *)f->share_dev : (void *)f->out_dev;
    for ( uint32_t w = 0u; w < 3u; ++w )
    {
        FixtureResetEpoch(f, rank, degree, op, elements, 0u, wire_gbps_x10, base_lat_ns);
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, rank, degree, src, dst, f->scratch,
            elements, op, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
    }
    FixtureResetEpoch(f, rank, degree, op, elements, 0u, wire_gbps_x10, base_lat_ns);
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
    if ( ctrl.error_word != 0u || ctrl.rounds_done != iters )
        fprintf(stderr, "BENCH-ERR op=%u elem=%llu wire=%u err=0x%llx done=%llu/%u\n",
            op, (unsigned long long)elements, wire_gbps_x10, (unsigned long long)ctrl.error_word,
            (unsigned long long)ctrl.rounds_done, iters);
    CHECK(ctrl.error_word == 0u && ctrl.rounds_done == iters, "large bench completed cleanly");
    BenchMetrics m;
    m.wall_us = (double)(t1 - t0) / 1000.0 / (double)iters;
    m.copy_us = (double)ctrl.copy_ns / 1000.0 / (double)iters;
    m.wait_us = (double)ctrl.peer_wait_ns / 1000.0 / (double)iters;
    m.combine_us = (double)ctrl.combine_ns / 1000.0 / (double)iters;
    return m;
}

static void RunLargeBenchmarks(LargeFixture *f)
{
    struct { const char *name; uint32_t op; uint32_t rows; uint32_t hidden; uint32_t iters; } cases[] = {
        { "AR_B1_4096",    1u, 1u,    4096u, 50u },
        { "AR_B8_4096",    1u, 8u,    4096u, 50u },
        { "AR_B8_8192",    1u, 8u,    8192u, 50u },
        { "AR_B64_4096",   1u, 64u,   4096u, 30u },
        { "AR_B64_6144",   1u, 64u,   6144u, 30u },
        { "AR_B64_8192",   1u, 64u,   8192u, 30u },
        { "AR_B512_4096",  1u, 512u,  4096u, 15u },
        { "AR_B512_8192",  1u, 512u,  8192u, 10u },
        { "AR_B1024_4096", 1u, 1024u, 4096u, 10u },
        { "AR_B1024_8192", 1u, 1024u, 8192u, 8u },
        { "RS_B1024_8192", SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 1024u, 8192u, 10u },
        { "RS_B2048_6144", SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 2048u, 6144u, 8u },
        { "RS_B4096_4096", SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 4096u, 4096u, 6u },
        { "RS_B4096_8192", SPARK_TP_MESH_OPERATION_REDUCE_SCATTER, 4096u, 8192u, 4u },
        { "AG_B64_8192",   SPARK_TP_MESH_OPERATION_ALL_GATHER, 64u,   8192u, 30u },
        { "AG_B1024_8192", SPARK_TP_MESH_OPERATION_ALL_GATHER, 1024u, 8192u, 10u },
        { "AG_B2048_6144", SPARK_TP_MESH_OPERATION_ALL_GATHER, 2048u, 6144u, 8u },
        { "AG_B4096_4096", SPARK_TP_MESH_OPERATION_ALL_GATHER, 4096u, 4096u, 6u },
        { "AG_B4096_8192", SPARK_TP_MESH_OPERATION_ALL_GATHER, 4096u, 8192u, 4u },
    };
    struct { const char *label; uint32_t gbps_x10; uint32_t lat_ns; } modes[] = {
        { "PURE GPU (0-wire)", 0u, 0u },
        { "1-SWITCH 1x100g (11.5 GB/s + 8us)", 115u, 8000u },
        { "2-SWITCH 2x100g (23.0 GB/s + 8us)", 230u, 8000u },
    };
    for ( uint32_t m = 0u; m < 3u; ++m )
    {
        printf("\n=== TP16 LARGE COLLECTIVE BENCH (%s) ===\n", modes[m].label);
        printf("%-15s | %7s | %10s | %10s | %10s | %10s\n", "Case", "Payload", "Wall_us", "Copy_us", "Wait_us", "Comb_us");
        for ( uint32_t c = 0u; c < sizeof(cases) / sizeof(cases[0]); ++c )
        {
            uint64_t elems = (uint64_t)cases[c].rows * cases[c].hidden;
            BenchMetrics r = BenchOne(f, cases[c].op, elems, modes[m].gbps_x10, modes[m].lat_ns, cases[c].iters);
            printf("%-15s | %5.1fMB | %8.1fus | %8.1fus | %8.1fus | %8.1fus\n",
                cases[c].name, (double)(elems * 2u) / (1024.0 * 1024.0),
                r.wall_us, r.copy_us, r.wait_us, r.combine_us);
        }
    }
}

static void Test2dCrossRankAndGateGx(LargeFixture *f)
{
    const uint32_t degree = 16u;
    const uint64_t el_small = 64ull * 4096ull;
    const uint64_t el_big = 256ull * 4096ull;
    EnsurePeerBufs(f, el_big, 7777u);
    uint16_t *h_out0 = (uint16_t *)malloc((size_t)el_small * sizeof(uint16_t));
    uint16_t *h_out7 = (uint16_t *)malloc((size_t)el_small * sizeof(uint16_t));
    uint16_t *h_out15 = (uint16_t *)malloc((size_t)el_small * sizeof(uint16_t));
    uint16_t *h_big = (uint16_t *)malloc((size_t)el_big * sizeof(uint16_t));
    uint32_t ranks[3] = { 0u, 7u, 15u };
    uint16_t *outs[3] = { h_out0, h_out7, h_out15 };
    for ( uint32_t r = 0u; r < 3u; ++r )
    {
        FixtureResetEpoch(f, ranks[r], degree, 1u, el_small, 1u, 0u, 0u);
        f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[ranks[r]], (size_t)el_small * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)el_small * sizeof(uint16_t)));
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, ranks[r], degree, f->local_dev, f->out_dev, f->scratch,
            el_small, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(outs[r], f->out_dev, (size_t)el_small * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }
    CHECK(memcmp(h_out0, h_out7, (size_t)el_small * sizeof(uint16_t)) == 0 &&
          memcmp(h_out0, h_out15, (size_t)el_small * sizeof(uint16_t)) == 0,
        "2D 4x4 RS+AG output is bit-identical across TP ranks 0, 7, 15");
    FixtureResetEpoch(f, 0u, degree, 1u, el_big, 1u, 0u, 0u);
    f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
    CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[0], (size_t)el_big * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)el_big * sizeof(uint16_t)));
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->local_dev, f->out_dev, f->scratch,
        el_big, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(h_big, f->out_dev, (size_t)el_big * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CHECK(memcmp(h_out0, h_big, (size_t)el_small * sizeof(uint16_t)) == 0,
        "2D 4x4 RS+AG Gate GX row-invariance: 64-row warm chunk matches 256-row cold chunk bit-for-bit");
    free(h_big);
    free(h_out15);
    free(h_out7);
    free(h_out0);
}

static void Test2dRows63DeadlockRegression(LargeFixture *f)
{
    const uint32_t degree = 16u;
    const uint64_t sizes[2] = { 63ull * 4096ull, 63ull * 6144ull };
    for ( uint32_t s = 0u; s < 2u; ++s )
    {
        const uint64_t elements = sizes[s];
        EnsurePeerBufs(f, elements, 8100u + s * 37u);
        uint16_t *h_ref = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
        uint16_t *h_out = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
        for ( uint64_t i = 0u; i < elements; ++i )
            h_ref[i] = Sum2d4x4Bf16(f->responder.peer_buf, i);
        for ( uint32_t r = 0u; r < degree; ++r )
        {
            FixtureResetEpoch(f, r, degree, 1u, elements, 1u, 0u, 0u);
            f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
            CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[r], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
            cudaGraph_t graph = 0;
            cudaGraphExec_t exec = 0;
            CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
            CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                f->doorbell_dev, f->gate_dev, f->control, r, degree, f->local_dev, f->out_dev, f->scratch,
                elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
            CUDA_OK(cudaStreamEndCapture(f->stream, &graph));
            CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));
            const uint32_t reps = (r == 0u || r == 6u || r == 7u || r == 15u) ? 16u : 4u;
            for ( uint32_t iter = 0u; iter < reps; ++iter )
                CUDA_OK(cudaGraphLaunch(exec, f->stream));
            CUDA_OK(cudaStreamSynchronize(f->stream));
            CUDA_OK(cudaMemcpy(h_out, f->out_dev, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            SparkTpMeshRoundControl ctrl;
            CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
            CHECK(ctrl.error_word == 0u && ctrl.rounds_done == reps &&
                  memcmp(h_ref, h_out, (size_t)elements * sizeof(uint16_t)) == 0,
                "2D 4x4 rows=63 CUDA graph replays match 0-ULP across all 16 TP ranks");
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        }
        free(h_out);
        free(h_ref);
    }

    const uint32_t fail_ranks[3] = { 6u, 7u, 15u };
    for ( uint32_t i = 0u; i < 3u; ++i )
    {
        const uint32_t r = fail_ranks[i];
        const uint32_t row_peer = (r & ~3u) | ((r + 1u) & 3u);
        const uint32_t col_peer = (((r >> 2u) + 1u) & 3u) << 2u | (r & 3u);
        SparkTpMeshRoundControl ctrl;
        FixtureResetEpoch(f, r, degree, 1u, 63ull * 6144ull, 0u, 0u, 0u);
        f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
        __atomic_store_n(&f->responder.drop_peer, row_peer, __ATOMIC_RELEASE);
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, r, degree, f->local_dev, f->out_dev, f->scratch,
            63ull * 6144ull, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u,
            "2D 4x4 rows=63 times out cleanly on dropped row peer without hanging any block");

        FixtureResetEpoch(f, r, degree, 1u, 63ull * 4096ull, 0u, 0u, 0u);
        f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
        __atomic_store_n(&f->responder.drop_peer, col_peer, __ATOMIC_RELEASE);
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, r, degree, f->local_dev, f->out_dev, f->scratch,
            63ull * 4096ull, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u,
            "2D 4x4 rows=63 times out cleanly on dropped col peer without hanging any block");
    }

    SparkTpMeshRoundControl ctrl;
    FixtureResetEpoch(f, 7u, degree, SPARK_TP_MESH_OPERATION_ALL_GATHER, 16ull * 65536ull, 0u, 0u, 0u);
    f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
    __atomic_store_n(&f->responder.drop_peer, 2u, __ATOMIC_RELEASE);
    CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
        f->doorbell_dev, f->gate_dev, f->control, 7u, degree, f->share_dev, f->out_dev, f->scratch,
        16ull * 65536ull, SPARK_TP_MESH_OPERATION_ALL_GATHER, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000ull, SPARK_TP_MESH_STAGE_ALL));
    CUDA_OK(cudaStreamSynchronize(f->stream));
    CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
    CHECK(ctrl.error_word != 0u && ctrl.rounds_done == 0u,
        "WideStreamGather non-zero rank times out cleanly on dropped peer without hanging");
}

static void SetupHybrid2dCtrl(LargeFixture *f, SparkWeightdMeshHybridCtrl *ctrl_host,
    uint8_t *shm_host, uint32_t rank, uint32_t degree)
{
    memset(ctrl_host, 0, sizeof(*ctrl_host));
    memset(shm_host, 0, 16u * 65536u);
    ctrl_host->magic = SPARK_WEIGHTD_MESH_HYBRID_MAGIC;
    ctrl_host->version = SPARK_WEIGHTD_MESH_HYBRID_VERSION;
    ctrl_host->flags = SPARK_WEIGHTD_MESH_HYBRID_FLAG_BF_UNIQUE;
    ctrl_host->shm_generation = 1u;
    ctrl_host->shm_bytes = 16u * 65536u;
    ctrl_host->recv_iova_base = (uint64_t)(uintptr_t)f->band_host;
    ctrl_host->staging_iova_base = (uint64_t)(uintptr_t)f->staging_host;
    ctrl_host->owner_lane = 0u;
    ctrl_host->owner_band = 0u;
    ctrl_host->owner_rank = rank;
    ctrl_host->degree = degree;
    ctrl_host->peer_count = degree - 1u;
    ctrl_host->all_peers_mask = ((1u << degree) - 1u) & ~(1u << rank);
    ctrl_host->min_max_inline = 828u;
    ctrl_host->dbrec_in_shm = 0u;
    ctrl_host->grant_gen = 1u;
    ctrl_host->cpu_issued_seq = 0u;
    for ( uint32_t i = 0u, target = 0u; target < degree; ++target )
    {
        if ( target == rank ) continue;
        SparkWeightdMeshHybridPeer *hp = &ctrl_host->peers[i];
        hp->sq_shm_off = i * 65536u;
        hp->dbrec_shm_off = SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE;
        hp->sq_depth = 1024u;
        hp->sq_mask = 1023u;
        hp->qpn_shifted = (0x100u + target) << 8u;
        hp->rkey_be = htobe32(0x22000000u + target);
        hp->recv_lkey_be = htobe32(0x33000000u);
        hp->staging_lkey_be = htobe32(0x44000000u);
        hp->remote_base = UINT64_C(0x700000000000) + (uint64_t)target * UINT64_C(0x10000000);
        hp->staging_peer_off = (uint64_t)target * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES;
        hp->max_inline = 828u;
        hp->peer = (uint8_t)i;
        hp->target = (uint8_t)target;
        ctrl_host->gpu_pi[i] = 0u;
        ctrl_host->gpu_wrs[i] = 0u;
        ctrl_host->cpu_sq_limit[i] = 1024u;
        ctrl_host->cpu_wr_limit[i] = 64u;
        f->responder.hybrid_pi[i] = 0u;
        f->responder.hybrid_wrs[i] = 0u;
        i++;
    }
    __atomic_store_n(&ctrl_host->cpu_mode, 1u, __ATOMIC_SEQ_CST);
}

static void Test2dHybridB64AndRevokeResume(LargeFixture *f)
{
    const uint32_t degree = 16u;
    const uint64_t shapes[3] = { 63ull * 4096ull, 43ull * 7168ull, 63ull * 6144ull };
    const uint32_t test_ranks[4] = { 0u, 7u, 9u, 15u };
    SparkWeightdMeshHybridCtrl *ctrl_host = 0, *ctrl_dev = 0;
    uint8_t *shm_host = 0, *shm_dev = 0;
    CUDA_OK(cudaHostAlloc(&ctrl_host, sizeof(*ctrl_host), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&ctrl_dev, ctrl_host, 0u));
    CUDA_OK(cudaHostAlloc(&shm_host, 16u * 65536u, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&shm_dev, shm_host, 0u));
    f->responder.hybrid_shm = shm_host;
    SparkTpMeshFastHybridBindOverride(f->gate_dev, ctrl_dev, shm_dev, 0u);

    for ( uint32_t s = 0u; s < 3u; ++s )
    {
        const uint64_t elements = shapes[s];
        const uint64_t cap2d = SPARK_TP_MESH_FAST_2D_CAP_ELEMENTS;
        const uint64_t chunks = (elements + cap2d - 1u) / cap2d;
        EnsurePeerBufs(f, elements, 9100u + s * 53u);
        uint16_t *h_ref = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
        uint16_t *h_out = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
        for ( uint64_t i = 0u; i < elements; ++i )
            h_ref[i] = Sum2d4x4Bf16(f->responder.peer_buf, i);
        for ( uint32_t ri = 0u; ri < 4u; ++ri )
        {
            const uint32_t rank = test_ranks[ri];
            __atomic_store_n(&f->responder.hybrid_ctrl, (SparkWeightdMeshHybridCtrl *)0, __ATOMIC_RELEASE);
            FixtureResetEpoch(f, rank, degree, 1u, elements, 1u, 0u, 0u);
            SetupHybrid2dCtrl(f, ctrl_host, shm_host, rank, degree);
            __atomic_store_n(&f->responder.hybrid_ctrl, ctrl_host, __ATOMIC_RELEASE);
            f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES | SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ;
            CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[rank], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
            cudaGraph_t graph = 0;
            cudaGraphExec_t exec = 0;
            CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
            CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_dev, f->scratch,
                elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
            CUDA_OK(cudaStreamEndCapture(f->stream, &graph));
            CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));
            const uint32_t reps = 8u;
            for ( uint32_t rep = 0u; rep < reps; ++rep )
                CUDA_OK(cudaGraphLaunch(exec, f->stream));
            CUDA_OK(cudaStreamSynchronize(f->stream));
            CUDA_OK(cudaMemcpy(h_out, f->out_dev, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            SparkTpMeshRoundControl ctrl;
            CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
            CHECK(ctrl.error_word == 0u && ctrl.rounds_done == reps &&
                  memcmp(h_ref, h_out, (size_t)elements * sizeof(uint16_t)) == 0 &&
                  __atomic_load_n(&f->responder.hybrid_dma_count, __ATOMIC_ACQUIRE) == 4u * chunks * reps &&
                  __atomic_load_n(&f->responder.hybrid_wqe_errors, __ATOMIC_ACQUIRE) == 0u &&
                  __atomic_load_n(&f->responder.cpu_doorbell_count, __ATOMIC_ACQUIRE) == 0u,
                "2D 4x4 B64 Hybrid MODE 0 emits verified 3-peer DMA WQEs across all hops with 0-ULP match");
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        }

        __atomic_store_n(&f->responder.hybrid_ctrl, (SparkWeightdMeshHybridCtrl *)0, __ATOMIC_RELEASE);
        FixtureResetEpoch(f, 9u, degree, 1u, elements, 1u, 0u, 0u);
        SetupHybrid2dCtrl(f, ctrl_host, shm_host, 9u, degree);
        __atomic_store_n(&f->responder.ship_delay_ns, 300000u, __ATOMIC_RELEASE);
        __atomic_store_n(&f->responder.revoke_every_n, 2u, __ATOMIC_RELEASE);
        __atomic_store_n(&f->responder.hybrid_ctrl, ctrl_host, __ATOMIC_RELEASE);
        f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES | SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ;
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[9], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
        for ( uint32_t rep = 0u; rep < 4u; ++rep )
            CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                f->doorbell_dev, f->gate_dev, f->control, 9u, degree, f->local_dev, f->out_dev, f->scratch,
                elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_out, f->out_dev, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        SparkTpMeshRoundControl ctrl;
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 4u &&
              memcmp(h_ref, h_out, (size_t)elements * sizeof(uint16_t)) == 0 &&
              __atomic_load_n(&f->responder.revoke_count, __ATOMIC_ACQUIRE) >= 4u &&
              __atomic_load_n(&f->responder.hybrid_dma_count, __ATOMIC_ACQUIRE) >= 2u * chunks * 4u &&
              __atomic_load_n(&f->responder.hybrid_wqe_errors, __ATOMIC_ACQUIRE) == 0u,
            "2D 4x4 B64 Hybrid resumes immediately after revoke+re-grant under 300us ship delay via cpu_issued_seq");

        __atomic_store_n(&f->responder.hybrid_ctrl, (SparkWeightdMeshHybridCtrl *)0, __ATOMIC_RELEASE);
        FixtureResetEpoch(f, 7u, degree, 1u, elements, 1u, 0u, 0u);
        SetupHybrid2dCtrl(f, ctrl_host, shm_host, 7u, degree);
        __atomic_store_n(&ctrl_host->cpu_mode, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&f->responder.hybrid_ctrl, ctrl_host, __ATOMIC_RELEASE);
        f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES | SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ;
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[7], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
        for ( uint32_t rep = 0u; rep < 2u; ++rep )
            CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                f->doorbell_dev, f->gate_dev, f->control, 7u, degree, f->local_dev, f->out_dev, f->scratch,
                elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        uint64_t cpu_before = __atomic_load_n(&f->responder.cpu_doorbell_count, __ATOMIC_ACQUIRE);
        __atomic_store_n(&ctrl_host->grant_gen, 2u, __ATOMIC_RELEASE);
        __atomic_store_n(&ctrl_host->cpu_mode, 1u, __ATOMIC_SEQ_CST);
        for ( uint32_t rep = 0u; rep < 4u; ++rep )
            CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                f->doorbell_dev, f->gate_dev, f->control, 7u, degree, f->local_dev, f->out_dev, f->scratch,
                elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_out, f->out_dev, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
        CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 6u &&
              memcmp(h_ref, h_out, (size_t)elements * sizeof(uint16_t)) == 0 &&
              cpu_before == 8u * chunks &&
              __atomic_load_n(&f->responder.cpu_doorbell_count, __ATOMIC_ACQUIRE) == cpu_before &&
              __atomic_load_n(&f->responder.hybrid_dma_count, __ATOMIC_ACQUIRE) == 16u * chunks &&
              __atomic_load_n(&f->responder.hybrid_wqe_errors, __ATOMIC_ACQUIRE) == 0u,
            "2D 4x4 B64 transitions cleanly from CPU doorbells to Hybrid MODE 0 within the same epoch");

        free(h_out);
        free(h_ref);
    }

    SparkTpMeshFastHybridClearOverride(f->gate_dev);
    __atomic_store_n(&f->responder.hybrid_ctrl, (SparkWeightdMeshHybridCtrl *)0, __ATOMIC_RELEASE);
    f->responder.hybrid_shm = 0;
    f->gate_host->capabilities = 0u;
    cudaFreeHost(shm_host);
    cudaFreeHost(ctrl_host);
}

static void Test2dDisabledMatches1dVerify(LargeFixture *f)
{
    const uint32_t degree = 16u;
    const uint64_t sizes[2] = { 2ull * 24ull * 2816ull, 64ull * 4096ull };
    for ( uint32_t s = 0u; s < 2u; ++s )
    {
        const uint64_t elements = sizes[s];
        EnsurePeerBufs(f, elements, 9100u + s * 41u);
        uint16_t *h_ref1d = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
        uint16_t *h_out = (uint16_t *)malloc((size_t)elements * sizeof(uint16_t));
        for ( uint64_t i = 0u; i < elements; ++i )
        {
            float acc = 0.0f;
            for ( uint32_t p = 0u; p < degree; ++p )
                acc += Bf16ToFloat(f->responder.peer_buf[p][i]);
            h_ref1d[i] = FloatToBf16Rn(acc);
        }
        FixtureResetEpoch(f, 0u, degree, 1u, elements, 1u, 0u, 0u);
        f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES;
        CUDA_OK(cudaMemcpy(f->local_dev, f->responder.peer_buf[0], (size_t)elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)elements * sizeof(uint16_t)));
        const uint32_t prev_2d = SparkTpMeshFastSetTwoDAllowed(0u);
        CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
            f->doorbell_dev, f->gate_dev, f->control, 0u, degree, f->local_dev, f->out_dev, f->scratch,
            elements, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
        (void)SparkTpMeshFastSetTwoDAllowed(prev_2d);
        CUDA_OK(cudaStreamSynchronize(f->stream));
        CUDA_OK(cudaMemcpy(h_out, f->out_dev, (size_t)elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK(memcmp(h_ref1d, h_out, (size_t)elements * sizeof(uint16_t)) == 0,
            "Verify gate (TwoDAllowed=0) keeps >=131072-element all-reduce 0-ULP bit-identical to 1D FP32 sum");
        free(h_out);
        free(h_ref1d);
    }
}

static void TestFullStagingFollowedBy2dHalfOverlapRegression(LargeFixture *f)
{
    const uint32_t degree = 16u;
    const uint32_t rank = 5u;
    const uint64_t a2a_sizes[2] = { 262144ull, 266240ull };
    SparkWeightdMeshHybridCtrl *ctrl_host = 0, *ctrl_dev = 0;
    uint8_t *shm_host = 0, *shm_dev = 0;
    CUDA_OK(cudaHostAlloc(&ctrl_host, sizeof(*ctrl_host), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&ctrl_dev, ctrl_host, 0u));
    CUDA_OK(cudaHostAlloc(&shm_host, 16u * 65536u, cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(&shm_dev, shm_host, 0u));
    f->responder.hybrid_shm = shm_host;
    SparkTpMeshFastHybridBindOverride(f->gate_dev, ctrl_dev, shm_dev, 0u);

    const uint64_t ar_sizes[2] = { 262144ull, 32768ull };
    for ( uint32_t c = 0u; c < 2u; ++c )
    {
        for ( uint32_t m = 0u; m < 2u; ++m )
        {
            const uint64_t a2a_elems = a2a_sizes[c];
            const uint64_t ar_elems = ar_sizes[m];
            __atomic_store_n(&f->responder.hybrid_ctrl, (SparkWeightdMeshHybridCtrl *)0, __ATOMIC_RELEASE);
            FixtureResetEpoch(f, rank, degree, 1u, ar_elems, 0u, 0u, 0u);
            SetupHybrid2dCtrl(f, ctrl_host, shm_host, rank, degree);
            __atomic_store_n(&f->responder.ship_delay_ns, 400000u, __ATOMIC_RELEASE);
            __atomic_store_n(&f->responder.verify_staging_tail, 1u, __ATOMIC_RELEASE);
            __atomic_store_n(&f->responder.hybrid_ctrl, ctrl_host, __ATOMIC_RELEASE);
            f->gate_host->capabilities = SPARK_WEIGHTD_MESH_CAPABILITIES | SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ;
            CUDA_OK(cudaMemset(f->local_dev, 0x3c, (size_t)(16ull * a2a_elems) * sizeof(uint16_t)));
            CUDA_OK(cudaMemset(f->out_dev, 0, (size_t)(16ull * a2a_elems) * sizeof(uint16_t)));
            cudaGraph_t graph = 0;
            cudaGraphExec_t exec = 0;
            CUDA_OK(cudaStreamBeginCapture(f->stream, cudaStreamCaptureModeThreadLocal));
            CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_dev, f->scratch,
                a2a_elems, SPARK_TP_MESH_OPERATION_ALL_TO_ALL, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
            CUDA_OK(SparkTpLaunchMeshHardwareFast(f->stream, f->band_dev, SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                f->doorbell_dev, f->gate_dev, f->control, rank, degree, f->local_dev, f->out_dev, f->scratch,
                ar_elems, 1u, 1u, 1u, SPARK_TP_MESH_ROUTES_SLICE | SPARK_TP_MESH_ROUTES_PEER, f->staging_dev, 5000000000ull, SPARK_TP_MESH_STAGE_ALL));
            CUDA_OK(cudaStreamEndCapture(f->stream, &graph));
            CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));
            for ( uint32_t rep = 0u; rep < 4u; ++rep )
                CUDA_OK(cudaGraphLaunch(exec, f->stream));
            CUDA_OK(cudaStreamSynchronize(f->stream));
            SparkTpMeshRoundControl ctrl;
            CUDA_OK(cudaMemcpy(&ctrl, f->control, sizeof(ctrl), cudaMemcpyDeviceToHost));
            CHECK(ctrl.error_word == 0u && ctrl.rounds_done == 8u &&
                  __atomic_load_n(&f->responder.full_staging_checked, __ATOMIC_ACQUIRE) == 4u &&
                  __atomic_load_n(&f->responder.staging_clobber_errors, __ATOMIC_ACQUIRE) == 0u &&
                  __atomic_load_n(&f->responder.hybrid_wqe_errors, __ATOMIC_ACQUIRE) == 0u,
                "full-staging AllToAll (>=524288B) followed by half-staging 2D/RSAG waits for ship without clobbering staging tail");
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        }
    }

    SparkTpMeshFastHybridClearOverride(f->gate_dev);
    __atomic_store_n(&f->responder.hybrid_ctrl, (SparkWeightdMeshHybridCtrl *)0, __ATOMIC_RELEASE);
    f->responder.hybrid_shm = 0;
    f->gate_host->capabilities = 0u;
    cudaFreeHost(shm_host);
    cudaFreeHost(ctrl_host);
}

int main(void)
{
    LargeFixture fixture;
    FixtureInit(&fixture, 4096ull * 8192ull);

    uint32_t ranks[] = { 0u, 7u, 15u };
    for ( uint32_t r = 0u; r < 3u; ++r )
    {
        TestRsagExact(&fixture, ranks[r], 8ull * 4096ull, 0u);
        TestRsagExact(&fixture, ranks[r], 64ull * 4096ull, 1u);
        TestRsagExact(&fixture, ranks[r], 64ull * 5120ull, 0u);
        TestRsagExact(&fixture, ranks[r], 64ull * 6144ull, 0u);
        TestRsagExact(&fixture, ranks[r], 64ull * 7168ull, 1u);
        TestRsagExact(&fixture, ranks[r], 64ull * 8192ull, 0u);
        TestRsagExact(&fixture, ranks[r], 65ull * 4096ull, 0u);
        TestRsagExact(&fixture, ranks[r], 512ull * 4096ull, 1u);
        TestReduceScatterExact(&fixture, ranks[r], 512ull * 4096ull);
        TestReduceScatterExact(&fixture, ranks[r], 2048ull * 6144ull);
        TestAllGatherExact(&fixture, ranks[r], 64ull * 8192ull);
        TestAllGatherExact(&fixture, ranks[r], 2048ull * 6144ull);
    }
    TestRsagExact(&fixture, 0u, 1024ull * 8192ull, 0u);
    TestReduceScatterExact(&fixture, 0u, 4096ull * 4096ull);
    TestAllGatherExact(&fixture, 0u, 4096ull * 4096ull);
    TestE14R2Degree4AndScatterFastPaths(&fixture);
    Test2dCrossRankAndGateGx(&fixture);
    Test2dRows63DeadlockRegression(&fixture);
    Test2dHybridB64AndRevokeResume(&fixture);
    Test2dDisabledMatches1dVerify(&fixture);
    TestFullStagingFollowedBy2dHalfOverlapRegression(&fixture);
    TestLargeCancelAndTimeout(&fixture);

    RunLargeBenchmarks(&fixture);

    FixtureDestroy(&fixture);
    printf("tp_mesh_prefill_collective_cuda: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures != 0u ? 1 : 0;
}
