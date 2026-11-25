#pragma once

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_TP_MESH_FAST_STREAM_REDUCE_MAX_ELEMENTS 8192u
#define SPARK_TP_MESH_FAST_RSAG_STREAM_SPAN_MAX 32768u
#define SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS 128u
#define SPARK_TP_MESH_FAST_WIDE_GATHER_MIN_ELEMENTS 4096u

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
{
    uint32_t prev = SparkTpMeshWireCodecAllowedTls;
    SparkTpMeshWireCodecAllowedTls = allowed != 0u ? 1u : 0u;
    return prev;
}

static inline uint32_t SparkTpMeshFastWireCodecAllowed(void)
{
    return SparkTpMeshWireCodecAllowedTls;
}

static __device__ __forceinline__ uint64_t SparkTpFastTimerNs(void)
{
    uint64_t ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory");
    return ns;
}

static __global__ void SparkTpMeshFastPublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,
    const void *source,uint64_t bytes,uint64_t route,
    volatile SparkWeightdMeshWaitRequest *gate,uint64_t timeout_ns)
{
    __shared__ uint32_t ready;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint32_t failed = control->error_word != 0u || control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = control->seq + 1u;
    const uint64_t tag = (control->epoch << 32u) | sequence;
    const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    volatile uint8_t *destination = band + slot * slot_bytes;
    volatile uint64_t *tail = (volatile uint64_t *)(destination + slot_bytes - 8u);
    const volatile uint64_t *shipped_cell = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    const uint64_t previous = control->round_seq;
    uint64_t shipped = 0u;
    if ( threadIdx.x == 0u && shipped_cell != 0 && previous != 0u )
        shipped = SparkTpLdcvU64(shipped_cell);
    if ( failed == 0u && bytes != 0u )
        SparkTpMeshCopyBytesVolatile(destination,source,bytes,threadIdx.x,blockDim.x);
    if ( threadIdx.x == 0u )
    {
        ready = shipped_cell != 0 && SparkTpMeshShippedAccepts(control,shipped,0u) != 0u ? 1u : 0u;
        if ( failed == 0u )
        {
            *tail = tag;
            if ( ready != 0u ) { entry[1] = bytes; entry[2] = slot; entry[3] = route; }
        }
    }
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x != 0u ) return;
    if ( ready == 0u && !(shipped_cell != 0 && SparkTpMeshShippedAccepts(control,SparkTpLdAcquireU64(shipped_cell),0u) != 0u) )
    {
        SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns);
        SparkTpMeshGateAwait(gate,control);
        SparkTpMeshGateGuard(gate,control);
    }
    if ( failed != 0u && control->error_word == 0u ) control->error_word = UINT64_MAX;
    if ( control->error_word == 0u )
    {
        if ( ready == 0u ) { entry[1] = bytes; entry[2] = slot; entry[3] = route; __threadfence_system(); }
        control->round_seq_prior = control->round_seq;
        control->round_seq = tag;
        control->seq = sequence;
        entry[0] = tag;
        const uint64_t now = SparkTpGlobalTimerNs();
        control->copy_ns += now - started;
        control->wait_started_ns = now;
    }
    else
        control->wait_started_ns = SparkTpGlobalTimerNs();
    __threadfence();
    LmDependentRelease();
}

static SparkGpuError SparkTpMeshFastPublish(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,
    const void *source,uint64_t bytes,uint64_t route,
    SparkWeightdMeshWaitRequest *gate,uint64_t peer_mask,uint64_t timeout_ns)
{
    if ( bytes < SPARK_TP_MESH_WIDE_PUBLISH_BYTES )
    {
        LM_LAUNCH((SparkTpMeshFastPublishKernel),1u,SPARK_TP_MESH_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,entry,control,rank,source,bytes,route,gate,timeout_ns);
        return SparkGpuPeekAtLastError();
    }
    return SparkTpMeshHardwarePublish(stream,band,slot_bytes,slots_per_rank,entry,control,rank,
        source,bytes,0u,bytes,route,gate,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_ARRIVAL,peer_mask,timeout_ns);
}

static __global__ void SparkTpMeshFastPairGatherKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,uint64_t local_elements,uint32_t pair_only)
{
    __shared__ uint32_t wait_over;
    __shared__ uint32_t failed;
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
    const uint64_t tag = control->round_seq;
    const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
    if ( tid < 32u )
    {
        const volatile uint64_t *tail = (const volatile uint64_t *)(band +
            ((uint64_t)tid * slots_per_rank + ring) * slot_bytes + slot_bytes - 8u);
        const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
        const uint64_t expected = control->cancel_expected;
        const uint64_t started = s_wait_start;
        uint32_t missing = pair_only != 0u ? (tid == (rank ^ 1u) ? 1u : 0u) :
            (tid < degree && tid != rank ? 1u : 0u);
        uint32_t spins = 0u;
        uint64_t seen = 0u,diag = 0u,fault = 0u;
        arr_base = control->wait_started_ns;
        for (;;)
        {
            if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
            {
                missing = 0u;
                arr_landed = SparkTpFastTimerNs();
            }
            if ( __all_sync(0xffffffffu,missing == 0u) ) break;
            if ( (++spins & 31u) == 0u )
            {
                if ( tid == 0u )
                {
                    const uint64_t now = SparkTpFastTimerNs();
                    if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                        fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
                    else if ( now > started && now - started > timeout_ns )
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
        if ( tid == 0u ) { *(volatile uint32_t *)&wait_over = 1u; arr_end = SparkTpGlobalTimerNs(); }
        if ( fault != 0u && tid == 0u )
        {
            if ( control->error_word == 0u ) { control->error_word = fault; control->diag_word = diag; __threadfence(); }
            failed = 1u;
        }
    }
    else if ( prefetch != 0 && tid < 32u + SPARK_TP_MESH_PREFETCH_THREADS )
    {
        const volatile uint32_t *stop = &wait_over;
        const uint32_t spans = prefetch->count;
        const uint64_t extent = LmWeightPrefetchExtent(prefetch,spans);
        const uint64_t stride = (uint64_t)SPARK_TP_MESH_PREFETCH_THREADS * LM_WEIGHT_PREFETCH_PIECE_BYTES;
        uint64_t position = resume != 0u ? prefetch->progress : 0u;
        for ( ; position < extent && *stop == 0u; position += stride )
            LmWeightPrefetchPiece(prefetch,spans,position + (uint64_t)(tid - 32u) * LM_WEIGHT_PREFETCH_PIECE_BYTES);
        if ( tid == 32u ) prefetch->progress = position < extent ? position : extent;
    }
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
        control->slot_cursor = control->seq;
        control->rounds_done++;
    }
    LmDependentRelease();
    if ( tid < 32u ) SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static __global__ void SparkTpMeshFastAllToAllStagePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,
    const uint16_t *local,uint32_t degree,uint32_t rank,
    uint64_t per_peer_elements,uint64_t count,uint64_t doorbell_bytes,uint64_t route,uint64_t timeout_ns)
{
    __shared__ uint32_t s_last;
    __shared__ volatile uint64_t s_wait_start;
    const uint32_t peer = blockIdx.x,tid = threadIdx.x;
    const uint64_t previous = control->round_seq;
    const volatile uint64_t *shipped_cell = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    if ( tid == 0u && previous != 0u && SparkTpLdcvU64(&control->error_word) == 0u )
    {
        if ( shipped_cell != 0 )
        {
            if ( SparkTpLdAcquireU64(shipped_cell) != previous )
            {
                const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
                const uint64_t expected = control->cancel_expected;
                s_wait_start = SparkTpFastTimerNs();
                for ( uint32_t spins = 0u;; )
                {
                    if ( SparkTpLdAcquireU64(shipped_cell) == previous )
                    {
                        if ( peer == rank )
                        {
                            const uint64_t now = SparkTpFastTimerNs();
                            if ( now > s_wait_start ) control->source_wait_ns += now - s_wait_start;
                        }
                        break;
                    }
                    if ( (++spins & 15u) == 0u )
                    {
                        const uint64_t now = SparkTpFastTimerNs();
                        if ( SparkTpLdcvU64(&control->error_word) != 0u ) break;
                        if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected )
                        {
                            atomicCAS((unsigned long long *)&control->error_word,0ull,
                                SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | previous);
                            break;
                        }
                        if ( now > s_wait_start && now - s_wait_start > timeout_ns )
                        {
                            atomicCAS((unsigned long long *)&control->error_word,0ull,previous);
                            break;
                        }
                    }
                    __nanosleep(50u);
                }
            }
        }
        else if ( peer == 0u )
        {
            SparkTpMeshShipGate(gate,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
            control->publish_started_ns = SparkTpGlobalTimerNs() | 1ull;
            __threadfence();
        }
        else
        {
            while ( SparkTpLdAcquireU64(&control->publish_started_ns) == 0u &&
                    SparkTpLdAcquireU64(&control->error_word) == 0u )
                __nanosleep(200u);
        }
    }
    __syncthreads();
    if ( tid == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs() | 1ull);
    const uint32_t failed = SparkTpLdcvU64(&control->error_word) != 0u || control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = control->seq + 1u;
    const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    if ( failed == 0u && peer != rank )
        SparkTpMeshCopyHalves(staging + (uint64_t)peer * staging_slot_elements,
            local + (uint64_t)peer * per_peer_elements,count,tid,blockDim.x);
    __threadfence_system();
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)degree ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || tid != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( control->error_word == 0u && control->seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
    if ( failed == 0u && control->error_word == 0u )
    {
        const uint64_t tag = (control->epoch << 32u) | sequence;
        entry[1] = doorbell_bytes;
        entry[2] = slot;
        entry[3] = route;
        control->round_seq_prior = control->round_seq;
        control->round_seq = tag;
        control->seq = sequence;
        *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        const uint64_t pub_start = SparkTpLdcvU64(&control->publish_started_ns);
        const uint64_t pub_end = SparkTpGlobalTimerNs();
        if ( pub_end > pub_start ) control->copy_ns += pub_end - pub_start;
    }
    control->publish_started_ns = 0u;
    control->wait_started_ns = SparkTpGlobalTimerNs();
    LmDependentRelease();
}

static __global__ void SparkTpMeshFastStreamRecvKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,uint64_t self_offset,uint64_t output_stride,uint64_t count)
{
    __shared__ uint32_t s_go;
    __shared__ volatile uint64_t s_peer_start;
    const uint32_t peer = blockIdx.x,tid = threadIdx.x;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    uint32_t ok = 0u;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t tag = control->round_seq;
    const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
    if ( peer == rank )
    {
        if ( local != 0 )
            SparkTpMeshCopyHalves(output + (uint64_t)rank * output_stride,local + self_offset,count,tid,blockDim.x);
        else
            SparkTpMeshCopyHalvesFromBand(output + (uint64_t)rank * output_stride,
                (const volatile uint16_t *)(band + ((uint64_t)rank * slots_per_rank + ring) * slot_bytes),
                count,tid,blockDim.x);
        __syncthreads();
        ok = SparkTpMeshArrivalWait(band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,
            prefetch,resume,0u,0u,arr_base,arr_end,arr_landed);
        if ( tid == 0u ) { control->math_started_ns = arr_end; __threadfence(); }
    }
    else
    {
        if ( tid == 0u )
        {
            const volatile uint64_t *tail = (const volatile uint64_t *)(band +
                ((uint64_t)peer * slots_per_rank + ring) * slot_bytes + slot_bytes - 8u);
            const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
            const uint64_t expected = control->cancel_expected;
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
            }
            s_go = go;
        }
        __syncthreads();
        if ( s_go != 0u )
            SparkTpMeshCopyHalvesFromBand(output + (uint64_t)peer * output_stride,
                (const volatile uint16_t *)(band + ((uint64_t)peer * slots_per_rank + ring) * slot_bytes),
                count,tid,blockDim.x);
    }
    __syncthreads();
    if ( tid == 0u &&
         atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == (unsigned long long)degree )
    {
        const uint64_t end_ns = SparkTpGlobalTimerNs();
        const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
        control->math_blocks_done = 0u;
        control->math_started_ns = 0u;
        control->math_finished_ns = 0u;
        if ( control->error_word == 0u )
        {
            if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
            control->slot_cursor = control->seq;
            control->rounds_done++;
        }
    }
    LmDependentRelease();
    if ( peer == rank && tid < 32u && ok != 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static __global__ void SparkTpMeshFastStreamAllReduceKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,uint32_t quads,uint32_t compute_blocks)
{
    __shared__ uint4 staged[16u][SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS];
    __shared__ uint32_t s_newly;
    __shared__ uint32_t s_failed;
    __shared__ uint64_t s_arr_end;
    __shared__ volatile uint64_t s_wait_start;
    const uint32_t tid = threadIdx.x;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    if ( blockIdx.x == compute_blocks )
    {
        uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
        uint32_t ok = SparkTpMeshArrivalWait(band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,
            prefetch,resume,0u,0u,arr_base,arr_end,arr_landed);
        if ( tid == 0u ) { control->math_started_ns = arr_end; __threadfence(); }
        __syncthreads();
        if ( tid == 0u &&
             atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x )
        {
            const uint64_t end_ns = SparkTpGlobalTimerNs();
            const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
            control->math_blocks_done = 0u;
            control->math_started_ns = 0u;
            control->math_finished_ns = 0u;
            if ( control->error_word == 0u )
            {
                if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
                control->slot_cursor = control->seq;
                control->rounds_done++;
            }
        }
        LmDependentRelease();
        if ( tid < 32u && ok != 0u )
            SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
        return;
    }
    const uint64_t tag = control->round_seq;
    const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
    const uint64_t stride = slots_per_rank * slot_bytes;
    const uint8_t *base = band + ring * slot_bytes;
    const uint32_t q = blockIdx.x * SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS + tid;
    float sum[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    uint32_t next_peer = 0u,loaded_mask = 1u << rank;
    if ( q < quads )
    {
        uint4 own_val = local != 0 ? ((const uint4 *)local)[q] :
            SparkTpLdCvU128(base + (uint64_t)rank * stride + (uint64_t)q * 16u);
        if ( rank == 0u ) { SparkTpAccumU128Bf16(sum,own_val); next_peer = 1u; }
        else staged[rank][tid] = own_val;
    }
    if ( tid == 0u ) { s_newly = 0u; s_failed = 0u; s_arr_end = 0u; s_wait_start = SparkTpFastTimerNs(); }
    __syncthreads();
    const volatile uint64_t *tail = tid < 32u ?
        (const volatile uint64_t *)(band + ((uint64_t)tid * slots_per_rank + ring) * slot_bytes + slot_bytes - 8u) : 0;
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const uint64_t expected = control->cancel_expected;
    const uint64_t arr_base = control->wait_started_ns;
    uint64_t arr_landed = UINT64_MAX;
    uint32_t missing = tid < degree && tid != rank ? 1u : 0u;
    uint32_t remaining_mask = ((1u << degree) - 1u) & ~(1u << rank);
    while ( remaining_mask != 0u )
    {
        if ( tid < 32u )
        {
            uint32_t newly = 0u,spins = 0u;
            uint64_t seen = 0u,diag = 0u,fault = 0u;
            const uint32_t record_arr = prefetch == 0 && blockIdx.x == 0u ? 1u : 0u;
            for (;;)
            {
                uint32_t just = 0u;
                if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
                {
                    missing = 0u;
                    just = 1u;
                    if ( record_arr != 0u ) arr_landed = SparkTpFastTimerNs();
                }
                newly = __ballot_sync(0xffffffffu,just != 0u);
                if ( newly != 0u ) break;
                if ( (++spins & 31u) == 0u )
                {
                    if ( tid == 0u )
                    {
                        const uint64_t base = s_wait_start,now = SparkTpFastTimerNs();
                        if ( SparkTpLdcvU64(&control->error_word) != 0u ) fault = UINT64_MAX;
                        else if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                            fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
                        else if ( now > base && now - base > timeout_ns )
                            fault = tag;
                    }
                    fault = __shfl_sync(0xffffffffu,fault,0);
                    if ( fault != 0u )
                    {
                        if ( fault == tag )
                        {
                            uint32_t first = __ffs(__ballot_sync(0xffffffffu,missing != 0u)) - 1u;
                            seen = __shfl_sync(0xffffffffu,seen,first);
                            diag = seen;
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
                    if ( fault != UINT64_MAX && control->error_word == 0u )
                    {
                        control->error_word = fault;
                        control->diag_word = diag;
                        __threadfence();
                    }
                    s_failed = 1u;
                }
                else if ( (remaining_mask & ~newly) == 0u )
                    s_arr_end = SparkTpGlobalTimerNs();
            }
        }
        __syncthreads();
        if ( s_failed != 0u ) break;
        uint32_t mask = s_newly;
        remaining_mask &= ~mask;
        while ( mask != 0u )
        {
            uint32_t ps[8u],n = 0u;
            uint4 v[8u];
            while ( mask != 0u && n < 8u )
            {
                uint32_t p = (uint32_t)(__ffs(mask) - 1);
                mask &= mask - 1u;
                ps[n++] = p;
            }
            if ( q < quads )
            {
#pragma unroll
                for ( uint32_t j = 0u; j < 8u; ++j )
                    if ( j < n )
                        v[j] = SparkTpLdCvU128(base + (uint64_t)ps[j] * stride + (uint64_t)q * 16u);
#pragma unroll
                for ( uint32_t j = 0u; j < 8u; ++j )
                    if ( j < n )
                    {
                        uint32_t p = ps[j];
                        if ( p == next_peer )
                        {
                            SparkTpAccumU128Bf16(sum,v[j]);
                            next_peer++;
                            while ( next_peer < degree && (loaded_mask & (1u << next_peer)) != 0u )
                            {
                                SparkTpAccumU128Bf16(sum,staged[next_peer][tid]);
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
    if ( s_failed == 0u && q < quads )
        ((uint4 *)output)[q] = SparkTpPackU128Bf16(sum);
    if ( prefetch == 0 && blockIdx.x == 0u && tid == 0u )
    {
        control->math_started_ns = s_arr_end;
        __threadfence();
    }
    __syncthreads();
    if ( tid == 0u &&
         atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x )
    {
        const uint64_t end_ns = SparkTpGlobalTimerNs();
        const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
        control->math_blocks_done = 0u;
        control->math_started_ns = 0u;
        control->math_finished_ns = 0u;
        if ( control->error_word == 0u )
        {
            if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
            control->slot_cursor = control->seq;
            control->rounds_done++;
        }
    }
    LmDependentRelease();
    if ( prefetch == 0 && blockIdx.x == 0u && tid < 32u && s_failed == 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,s_arr_end,arr_landed);
}

static __global__ void SparkTpMeshFastRsagStreamReducePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    uint16_t *output,uint64_t out_offset,uint32_t quads,uint32_t compute_blocks,
    uint64_t next_doorbell_bytes,uint64_t next_route)
{
    __shared__ uint4 staged[16u][SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS];
    __shared__ uint32_t s_newly;
    __shared__ uint32_t s_failed;
    __shared__ uint32_t s_last;
    __shared__ uint64_t s_arr_end;
    __shared__ volatile uint64_t s_wait_start;
    const uint32_t tid = threadIdx.x;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    if ( blockIdx.x == compute_blocks )
    {
        uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
        uint32_t ok = SparkTpMeshArrivalWait(band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,
            prefetch,resume,0u,0u,arr_base,arr_end,arr_landed);
        if ( tid < 32u && ok != 0u )
            SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
        return;
    }
    const uint64_t seq = control->seq;
    const uint64_t tag = control->round_seq;
    const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
    const uint32_t next_overflow = seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t next_seq = seq + 1u;
    const uint64_t next_slot = (uint64_t)rank * slots_per_rank + ((next_seq - 1u) & (slots_per_rank - 1u));
    volatile uint16_t *next_dst = next_overflow == 0u ? (volatile uint16_t *)(band + next_slot * slot_bytes) : 0;
    const uint64_t stride = slots_per_rank * slot_bytes;
    const uint8_t *base = band + ring * slot_bytes;
    const uint32_t q = blockIdx.x * SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS + tid;
    float sum[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    uint32_t next_peer = 0u,loaded_mask = 1u << rank;
    if ( q < quads )
    {
        uint4 own_val = SparkTpLdCvU128(base + (uint64_t)rank * stride + (uint64_t)q * 16u);
        if ( rank == 0u ) { SparkTpAccumU128Bf16(sum,own_val); next_peer = 1u; }
        else staged[rank][tid] = own_val;
    }
    if ( tid == 0u ) { s_newly = 0u; s_failed = 0u; s_last = 0u; s_arr_end = 0u; s_wait_start = SparkTpFastTimerNs(); }
    __syncthreads();
    const volatile uint64_t *tail = tid < 32u ?
        (const volatile uint64_t *)(band + ((uint64_t)tid * slots_per_rank + ring) * slot_bytes + slot_bytes - 8u) : 0;
    const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
    const uint64_t expected = control->cancel_expected;
    const uint64_t arr_base = control->wait_started_ns;
    uint64_t arr_landed = UINT64_MAX;
    uint32_t missing = tid < degree && tid != rank ? 1u : 0u;
    uint32_t remaining_mask = ((1u << degree) - 1u) & ~(1u << rank);
    while ( remaining_mask != 0u )
    {
        if ( tid < 32u )
        {
            uint32_t newly = 0u,spins = 0u;
            uint64_t seen = 0u,diag = 0u,fault = 0u;
            const uint32_t record_arr = prefetch == 0 && blockIdx.x == 0u ? 1u : 0u;
            for (;;)
            {
                uint32_t just = 0u;
                if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
                {
                    missing = 0u;
                    just = 1u;
                    if ( record_arr != 0u ) arr_landed = SparkTpFastTimerNs();
                }
                newly = __ballot_sync(0xffffffffu,just != 0u);
                if ( newly != 0u ) break;
                if ( (++spins & 31u) == 0u )
                {
                    if ( tid == 0u )
                    {
                        const uint64_t base = s_wait_start,now = SparkTpFastTimerNs();
                        if ( SparkTpLdcvU64(&control->error_word) != 0u ) fault = UINT64_MAX;
                        else if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                            fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
                        else if ( now > base && now - base > timeout_ns )
                            fault = tag;
                    }
                    fault = __shfl_sync(0xffffffffu,fault,0);
                    if ( fault != 0u )
                    {
                        if ( fault == tag )
                        {
                            uint32_t first = __ffs(__ballot_sync(0xffffffffu,missing != 0u)) - 1u;
                            seen = __shfl_sync(0xffffffffu,seen,first);
                            diag = seen;
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
                    if ( fault != UINT64_MAX && control->error_word == 0u )
                    {
                        control->error_word = fault;
                        control->diag_word = diag;
                        __threadfence();
                    }
                    s_failed = 1u;
                }
                else if ( (remaining_mask & ~newly) == 0u )
                    s_arr_end = SparkTpGlobalTimerNs();
            }
        }
        __syncthreads();
        if ( s_failed != 0u ) break;
        uint32_t mask = s_newly;
        remaining_mask &= ~mask;
        while ( mask != 0u )
        {
            uint32_t ps[8u],n = 0u;
            uint4 v[8u];
            while ( mask != 0u && n < 8u )
            {
                uint32_t p = (uint32_t)(__ffs(mask) - 1);
                mask &= mask - 1u;
                ps[n++] = p;
            }
            if ( q < quads )
            {
#pragma unroll
                for ( uint32_t j = 0u; j < 8u; ++j )
                    if ( j < n )
                        v[j] = SparkTpLdCvU128(base + (uint64_t)ps[j] * stride + (uint64_t)q * 16u);
#pragma unroll
                for ( uint32_t j = 0u; j < 8u; ++j )
                    if ( j < n )
                    {
                        uint32_t p = ps[j];
                        if ( p == next_peer )
                        {
                            SparkTpAccumU128Bf16(sum,v[j]);
                            next_peer++;
                            while ( next_peer < degree && (loaded_mask & (1u << next_peer)) != 0u )
                            {
                                SparkTpAccumU128Bf16(sum,staged[next_peer][tid]);
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
    if ( s_failed == 0u && q < quads )
    {
        uint4 packed = SparkTpPackU128Bf16(sum);
        ((uint4 *)(output + out_offset))[q] = packed;
        if ( next_dst != 0 ) SparkTpStVolatileU128(next_dst + (uint64_t)q * 8u,packed);
    }
    if ( next_dst != 0 ) __threadfence_system();
    if ( prefetch == 0 && blockIdx.x == 0u && tid == 0u )
    {
        control->math_started_ns = s_arr_end;
        __threadfence();
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)compute_blocks ? 1u : 0u;
    __syncthreads();
    if ( s_last != 0u && tid == 0u )
    {
        const uint64_t end_ns = SparkTpGlobalTimerNs();
        const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
        control->publish_blocks_done = 0u;
        control->math_started_ns = 0u;
        control->math_finished_ns = 0u;
        control->math_blocks_done = 0u;
        if ( s_failed == 0u && control->error_word == 0u )
        {
            if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
            if ( next_overflow != 0u )
                control->error_word = UINT64_MAX;
            else
            {
                const uint64_t next_tag = (control->epoch << 32u) | next_seq;
                *(volatile uint64_t *)(band + next_slot * slot_bytes + slot_bytes - 8u) = next_tag;
                SparkTpMeshShipGate(gate,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
                if ( control->error_word == 0u )
                {
                    entry[1] = next_doorbell_bytes;
                    entry[2] = next_slot;
                    entry[3] = next_route;
                    __threadfence_system();
                    control->round_seq_prior = control->round_seq;
                    control->round_seq = next_tag;
                    control->seq = next_seq;
                    control->slot_cursor = next_seq;
                    entry[0] = next_tag;
                    control->wait_started_ns = SparkTpGlobalTimerNs();
                }
            }
        }
        __threadfence();
        LmDependentRelease();
    }
    if ( prefetch == 0 && blockIdx.x == 0u && tid < 32u && s_failed == 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,s_arr_end,arr_landed);
}

static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshFastRsagWideReducePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,volatile SparkWeightdMeshWaitRequest *gate,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,
    void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t first,uint64_t span,
    uint64_t next_doorbell_bytes,uint64_t next_route)
{
    __shared__ uint32_t s_last;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint64_t seq = control->seq;
    const uint64_t ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
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
        if ( control->error_word == 0u ) control->error_word = UINT64_MAX;
    }
    else if ( control->error_word == 0u )
    {
        const uint64_t next_tag = (control->epoch << 32u) | next_seq;
        *(volatile uint64_t *)(band + next_slot * slot_bytes + slot_bytes - 8u) = next_tag;
        SparkTpMeshShipGate(gate,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
        if ( control->error_word == 0u )
        {
            entry[1] = next_doorbell_bytes;
            entry[2] = next_slot;
            entry[3] = next_route;
            __threadfence_system();
            control->round_seq_prior = control->round_seq;
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

static __global__ void SparkTpMeshFastWideStreamGatherKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint64_t timeout_ns,LmWeightPrefetchGroup *prefetch,uint32_t resume,
    const uint16_t *local,uint16_t *output,uint64_t begin,uint64_t total_count,
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
    const uint64_t tag = control->round_seq;
    const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
    const uint64_t start = is_slice_gather != 0u ? ((uint64_t)peer * slice_elements < total_count ? (uint64_t)peer * slice_elements : total_count) : 0u;
    const uint64_t length = is_slice_gather != 0u ? (total_count - start < slice_elements ? total_count - start : slice_elements) : slice_elements;
    const uint64_t dst_off = is_slice_gather != 0u ? (begin + start) : (begin + (uint64_t)peer * slice_elements);
    if ( peer == rank )
    {
        if ( skip_self == 0u && length != 0u )
        {
            if ( local != 0 )
                SparkTpMeshCopyHalves(output + dst_off,local,length,lane,lanes);
            else
                SparkTpMeshCopyHalvesFromBand(output + dst_off,
                    (const volatile uint16_t *)(band + ((uint64_t)rank * slots_per_rank + ring) * slot_bytes),
                    length,lane,lanes);
        }
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
            SparkTpMeshCopyHalvesFromBand(output + dst_off,
                (const volatile uint16_t *)(band + ((uint64_t)peer * slots_per_rank + ring) * slot_bytes),
                length,lane,lanes);
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
            control->slot_cursor = control->seq;
            if ( last != 0u ) control->rounds_done++;
        }
    }
    LmDependentRelease();
    if ( peer == rank && stripe == 0u && tid < 32u && ok != 0u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static inline uint32_t SparkTpMeshFastGatherStripes(uint64_t slice_elements)
{
    if ( slice_elements > 65536u ) return 4u;
    if ( slice_elements > 16384u ) return 2u;
    return 1u;
}

static SparkGpuError SparkTpMeshFastLaunchWideStreamGather(SparkGpuStream stream,
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint16_t *local,uint16_t *output,uint64_t begin,uint64_t total_count,
    uint64_t slice_elements,uint32_t is_slice_gather,uint32_t skip_self,uint32_t last,uint64_t timeout_ns)
{
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const dim3 grid(SparkTpMeshFastGatherStripes(slice_elements),degree);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastWideStreamGatherKernel),grid,SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,
        local,output,begin,total_count,slice_elements,is_slice_gather,skip_self,last);
}

static SparkGpuError SparkTpMeshFastPeerRsagChunk(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint8_t *staging,uint64_t local_elements,uint64_t begin,uint64_t count,
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
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = slice * sizeof(uint16_t);
    if ( stage != SPARK_TP_MESH_STAGE_COMPLETE )
    {
        if ( count <= SPARK_TP_MESH_FUSED_STAGE_ELEMENTS )
        {
            LM_LAUNCH((SparkTpMeshRsagStagePublishKernel),1u,degree * 32u,0u,stream,band,slot_bytes,slots_per_rank,entry,request,control,
                (uint16_t *)staging,staging_slot_elements,(const uint16_t *)local + begin,rank,count,slice,route.fields.slice_bytes,route.word,timeout_ns);
            status = SparkGpuPeekAtLastError();
        }
        else
        {
            const uint64_t stage_bytes = route.fields.slice_bytes;
            LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1u,1u,0u,stream,request,control,
                SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
            status = SparkGpuPeekAtLastError();
            if ( status == SPARK_GPU_SUCCESS )
                status = LM_LAUNCH_DEPENDENT((SparkTpMeshRsagStageKernel),dim3(SparkTpMeshCopyBlocks(slice,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,
                    band,slot_bytes,slots_per_rank,entry,control,(uint16_t *)staging,staging_slot_elements,
                    (const uint16_t *)local + begin,rank,count,slice,stage_bytes,route.word);
        }
        if ( status != SPARK_GPU_SUCCESS || stage == SPARK_TP_MESH_STAGE_PUBLISH ) return status;
    }
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    route.fields.slice_bytes = 0u;
    gather_bytes = (span * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
    const uint64_t doorbell_bytes = gather_bytes != 0u ? gather_bytes : 8u;
    if ( span != 0u && span <= SPARK_TP_MESH_FAST_RSAG_STREAM_SPAN_MAX && (span & 7u) == 0u &&
         ((uintptr_t)band & 15u) == 0u && (slot_bytes & 15u) == 0u &&
         ((uintptr_t)((uint16_t *)output + begin + first) & 15u) == 0u )
    {
        uint32_t resume = 0u;
        LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
        const uint32_t quads = (uint32_t)(span / 8u);
        const uint32_t compute_blocks = (quads + SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS - 1u) /
            SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS;
        const uint32_t grid = compute_blocks + (prefetch != 0 ? 1u : 0u);
        status = LM_LAUNCH_DEPENDENT((SparkTpMeshFastRsagStreamReducePublishKernel),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
            band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,timeout_ns,prefetch,resume,
            (uint16_t *)output,begin + first,quads,compute_blocks,doorbell_bytes,route.word);
    }
    else
    {
        status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
        if ( status == SPARK_GPU_SUCCESS )
            status = LM_LAUNCH_DEPENDENT((SparkTpMeshFastRsagWideReducePublishKernel),SparkTpMeshDirectBlocks(span != 0u ? span : 1u,1u),
                SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,request,control,
                rank,degree,timeout_ns,output,local_elements,begin,count,first,span,doorbell_bytes,route.word);
    }
    if ( status != SPARK_GPU_SUCCESS ) return status;
    return SparkTpMeshFastLaunchWideStreamGather(stream,band,slot_bytes,slots_per_rank,control,rank,degree,
        0,(uint16_t *)output,begin,count,slice,1u,1u,begin + count == local_elements ? 1u : 0u,timeout_ns);
}

static SparkGpuError SparkTpMeshFastLaunchStreamReduce(SparkGpuStream stream,
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint16_t *local,uint16_t *output,uint64_t elements,uint64_t timeout_ns)
{
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    const uint32_t quads = (uint32_t)(elements / 8u);
    const uint32_t compute_blocks = (quads + SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS - 1u) /
        SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS;
    const uint32_t grid = compute_blocks + (prefetch != 0 ? 1u : 0u);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastStreamAllReduceKernel),grid,SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,local,output,quads,compute_blocks);
}

static SparkGpuError SparkTpMeshFastLaunchStreamRecv(SparkGpuStream stream,
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint16_t *local,uint16_t *output,uint64_t self_offset,uint64_t output_stride,uint64_t count,uint64_t timeout_ns)
{
    uint32_t resume = 0u;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    return LM_LAUNCH_DEPENDENT((SparkTpMeshFastStreamRecvKernel),degree,SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,local,output,self_offset,output_stride,count);
}

static __device__ __forceinline__ uint8_t SparkTpLdCvU8(const volatile void *address)
{
    uint32_t v;
    asm volatile("ld.global.cv.u8 %0,[%1];" : "=r"(v) : "l"(address) : "memory");
    return (uint8_t)v;
}

static __device__ __forceinline__ uint16_t SparkTpLdCvU16(const volatile void *address)
{
    uint32_t v;
    asm volatile("ld.global.cv.u16 %0,[%1];" : "=r"(v) : "l"(address) : "memory");
    return (uint16_t)v;
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
    const uint32_t failed = control->error_word != 0u || control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = control->seq + 1u;
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
    if ( control->error_word == 0u && control->seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
    if ( failed == 0u )
    {
        const uint64_t tag = (control->epoch << 32u) | sequence;
        entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route;
        control->round_seq_prior = control->round_seq; control->round_seq = tag; control->seq = sequence;
        __threadfence_system();
        *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - control->publish_started_ns;
    }
    control->publish_started_ns = 0u;
    control->wait_started_ns = SparkTpGlobalTimerNs();
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
    const uint64_t seq = control->seq;
    const uint64_t ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
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
        if ( control->error_word == 0u ) control->error_word = UINT64_MAX;
    }
    else if ( control->error_word == 0u )
    {
        const uint64_t next_tag = (control->epoch << 32u) | next_seq;
        *(volatile uint64_t *)(band + next_slot * slot_bytes + slot_bytes - 8u) = next_tag;
        SparkTpMeshShipGate(gate,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
        if ( control->error_word == 0u )
        {
            entry[1] = next_doorbell_bytes;
            entry[2] = next_slot;
            entry[3] = next_route;
            __threadfence_system();
            control->round_seq_prior = control->round_seq;
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
    const uint32_t failed = control->error_word != 0u || control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = control->seq + 1u;
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
    if ( control->error_word == 0u && control->seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
    if ( failed == 0u )
    {
        const uint64_t tag = (control->epoch << 32u) | sequence;
        entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route;
        control->round_seq_prior = control->round_seq; control->round_seq = tag; control->seq = sequence;
        __threadfence_system();
        *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - control->publish_started_ns;
    }
    control->publish_started_ns = 0u;
    control->wait_started_ns = SparkTpGlobalTimerNs();
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
    const uint64_t tag = control->round_seq;
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
            control->slot_cursor = control->seq;
            if ( last != 0u ) control->rounds_done++;
        }
    }
    LmDependentRelease();
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
    const uint64_t ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
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
    const uint32_t failed = control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t mask = piece.slots_per_rank - 1u,sequence = control->seq + 1u;
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
    if ( failed != 0u && control->error_word == 0u ) control->error_word = UINT64_MAX;
    if ( failed == 0u && control->error_word == 0u )
    {
        const uint64_t pub_start = SparkTpLdcvU64(&control->publish_started_ns);
        const uint64_t pub_end = SparkTpGlobalTimerNs();
        if ( pub_end > pub_start ) control->copy_ns += pub_end - pub_start;
        if ( piece.ring_direct != 0u )
        {
            const uint64_t tag = (control->epoch << 32u) | sequence;
            const uint32_t half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence);
            volatile uint64_t *cell = piece.entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(sequence);
            SparkWeightdMeshRoute routed;
            routed.word = piece.route;
            routed.fields.half = half;
            cell[1] = piece.doorbell_bytes;
            cell[2] = slot;
            cell[3] = routed.word;
            control->round_seq_prior = control->round_seq;
            control->round_seq = tag;
            control->seq = sequence;
            *(volatile uint64_t *)(piece.band + slot * piece.slot_bytes + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half)) = tag;
            __threadfence_system();
            cell[0] = tag;
            control->wait_started_ns = SparkTpGlobalTimerNs();
        }
    }
    control->publish_started_ns = 0u;
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
    const uint64_t tag = control->round_seq - piece.lag,ring = (tag - 1u) & mask;
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
    const dim3 grid(SparkTpMeshFastGatherStripes(slice),degree);
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
    const dim3 grid(SparkTpMeshFastGatherStripes(local_elements),degree);
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
        (const void *)SparkTpMeshFastPairGatherKernel,
        (const void *)SparkTpMeshFastAllToAllStagePublishKernel,
        (const void *)SparkTpMeshFastStreamRecvKernel,
        (const void *)SparkTpMeshFastStreamAllReduceKernel,
        (const void *)SparkTpMeshFastRsagStreamReducePublishKernel,
        (const void *)SparkTpMeshFastRsagWideReducePublishKernel,
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
    return SparkTpMeshHardwarePrepare(host,device_out);
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
    if ( stage == SPARK_TP_MESH_STAGE_PUBLISH || rounds != 1u || logical_rows != 1u )
        return SparkTpLaunchMeshHardware(stream,band,slot_bytes,slots_per_rank,entry,gate,round_control,
            rank,degree,local,output,scratch,elements,operation,rounds,logical_rows,slice_routes,staging,timeout_ns,stage);
    SparkTpMeshRoundControl *control = (SparkTpMeshRoundControl *)round_control;
    SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)gate;
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
            route.fields.pair = 1u;
            status = SparkTpMeshFastPublish(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,rank,
                local,local_elements * sizeof(uint16_t),route.word,request,route.fields.peer_mask,timeout_ns);
            if ( status != SPARK_GPU_SUCCESS ) return status;
            prefetch = LmWeightPrefetchSlot(stream,&resume);
            return LM_LAUNCH_DEPENDENT((SparkTpMeshFastPairGatherKernel),1u,SPARK_TP_MESH_THREADS,0u,stream,
                (const uint8_t *)band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,
                (const uint16_t *)local,(uint16_t *)output,local_elements,pair_only);
        }
    }
    if ( stage == SPARK_TP_MESH_STAGE_ALL && operation == SPARK_TP_MESH_OPERATION_ALL_TO_ALL &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging != 0 )
    {
        const uint64_t slice = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
        if ( elements <= slice &&
             slice * sizeof(uint16_t) <= slot_bytes - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES &&
             SparkTpMeshAllToAllPipelined(elements,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) == 0u )
        {
            SparkWeightdMeshRoute route = {0};
            SparkGpuError status;
            route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
            route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
            route.fields.slice_bytes = ((elements * sizeof(uint16_t) + 7u) & ~UINT64_C(7));
            LM_LAUNCH((SparkTpMeshFastAllToAllStagePublishKernel),degree,SPARK_TP_MESH_THREADS,0u,stream,
                (uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,request,control,
                (uint16_t *)staging,slice,(const uint16_t *)local,degree,rank,elements,elements,
                route.fields.slice_bytes,route.word,timeout_ns);
            status = SparkGpuPeekAtLastError();
            if ( status != SPARK_GPU_SUCCESS ) return status;
            return SparkTpMeshFastLaunchStreamRecv(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,control,
                rank,degree,(const uint16_t *)local,(uint16_t *)output,(uint64_t)rank * elements,elements,elements,timeout_ns);
        }
    }
    if ( operation == 1u &&
         SparkTpMeshDirectPhasesPerChunk(elements,degree,1u,slice_routes) == 1u &&
         elements <= SPARK_TP_MESH_FAST_STREAM_REDUCE_MAX_ELEMENTS &&
         (elements & 7u) == 0u &&
         elements <= SparkTpMeshDirectCapacity(slot_bytes,1u) &&
         (((uintptr_t)band | (uintptr_t)output | (uintptr_t)local) & 15u) == 0u &&
         (slot_bytes & 15u) == 0u )
    {
        if ( stage == SPARK_TP_MESH_STAGE_ALL )
        {
            SparkWeightdMeshRoute route = {0};
            SparkGpuError status;
            route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
            status = SparkTpMeshFastPublish(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,rank,
                local,elements * sizeof(uint16_t),route.word,request,route.fields.peer_mask,timeout_ns);
            if ( status != SPARK_GPU_SUCCESS ) return status;
        }
        return SparkTpMeshFastLaunchStreamReduce(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,control,
            rank,degree,stage == SPARK_TP_MESH_STAGE_ALL ? (const uint16_t *)local : 0,(uint16_t *)output,elements,timeout_ns);
    }
    if ( operation == 1u &&
         SparkTpMeshDirectPhasesPerChunk(elements,degree,1u,slice_routes) == 2u &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging != 0 )
    {
        if ( wire_aligned != 0u && (elements & 31u) == 0u && (((uintptr_t)staging & 15u) == 0u) )
            return SparkTpMeshFastDispatchMxfp8Rsag(stream,wire_codec,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,timeout_ns);
        const uint64_t capacity = SparkTpMeshDirectPeerCapacity(degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
        if ( stage == SPARK_TP_MESH_STAGE_COMPLETE )
            return SparkTpMeshFastPeerRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,0u,elements,timeout_ns,stage);
        SparkGpuError status = SPARK_GPU_SUCCESS;
        for ( uint64_t begin = 0u; begin < elements && status == SPARK_GPU_SUCCESS; )
        {
            const uint64_t count = elements - begin < capacity ? elements - begin : capacity;
            status = SparkTpMeshFastPeerRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,begin,count,timeout_ns,stage);
            begin += count;
        }
        return status;
    }
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER &&
         (slice_routes & (SPARK_TP_MESH_ROUTES_PAIR | SPARK_TP_MESH_ROUTES_PAIR_FIRST)) == 0u )
    {
        const uint64_t local_elements = elements / degree;
        if ( wire_aligned != 0u && (local_elements & 31u) == 0u &&
             slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES )
            return SparkTpMeshFastDispatchMxfp8Gather(stream,wire_codec,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
                request,control,rank,degree,(const uint8_t *)local,output,elements,timeout_ns);
        if ( SparkTpMeshDirectPipelined(elements,degree,0u,slot_bytes) == 0u &&
             local_elements <= SparkTpMeshDirectCapacity(slot_bytes,0u) )
        {
            if ( stage == SPARK_TP_MESH_STAGE_ALL )
            {
                SparkWeightdMeshRoute route = {0};
                SparkGpuError status;
                route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
                status = SparkTpMeshFastPublish(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,rank,
                    local,local_elements * sizeof(uint16_t),route.word,request,route.fields.peer_mask,timeout_ns);
                if ( status != SPARK_GPU_SUCCESS ) return status;
            }
            if ( local_elements > SPARK_TP_MESH_FAST_WIDE_GATHER_MIN_ELEMENTS )
                return SparkTpMeshFastLaunchWideStreamGather(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,control,
                    rank,degree,stage == SPARK_TP_MESH_STAGE_ALL ? (const uint16_t *)local : 0,(uint16_t *)output,
                    0u,elements,local_elements,0u,0u,1u,timeout_ns);
            return SparkTpMeshFastLaunchStreamRecv(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,control,
                rank,degree,stage == SPARK_TP_MESH_STAGE_ALL ? (const uint16_t *)local : 0,(uint16_t *)output,0u,local_elements,local_elements,timeout_ns);
        }
    }
    if ( operation == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER &&
         wire_aligned != 0u && wire_codec <= SPARK_TP_MESH_WIRE_CODEC_FP8_B16 &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PAIR_FIRST) == 0u &&
         (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging != 0 &&
         (((uintptr_t)staging & 15u) == 0u) && slot_bytes == SPARK_WEIGHTD_MESH_SLOT_BYTES &&
         ((elements / degree) & 31u) == 0u )
        return SparkTpMeshFastDispatchMxfp8Scatter(stream,wire_codec,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
            request,control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,timeout_ns);
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
