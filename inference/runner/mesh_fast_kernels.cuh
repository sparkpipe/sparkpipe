#pragma once

#if defined(__CUDACC__)

#define SPARK_TP_MESH_FAST_STREAM_REDUCE_MAX_ELEMENTS 8192u
#define SPARK_TP_MESH_FAST_STREAM_REDUCE_THREADS 128u

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
        control->copy_ns += SparkTpGlobalTimerNs() - started;
    }
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
                for (;;)
                {
                    const uint64_t now = SparkTpFastTimerNs();
                    if ( SparkTpLdAcquireU64(shipped_cell) == previous )
                    {
                        if ( peer == rank && now > s_wait_start ) control->source_wait_ns += now - s_wait_start;
                        break;
                    }
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
                    __nanosleep(200u);
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
            for (;;)
            {
                const uint64_t now = SparkTpFastTimerNs();
                if ( SparkTpLdAcquireU64(tail) == tag ) { go = 1u; break; }
                if ( SparkTpLdcvU64(&control->error_word) != 0u ) break;
                if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected ) break;
                if ( now > s_peer_start && now - s_peer_start > timeout_ns ) break;
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
    const uint64_t started = s_wait_start;
    const uint64_t arr_base = control->wait_started_ns;
    uint64_t arr_landed = UINT64_MAX;
    uint32_t missing = tid < degree && tid != rank ? 1u : 0u;
    uint32_t remaining_mask = ((1u << degree) - 1u) & ~(1u << rank);
    while ( remaining_mask != 0u )
    {
        if ( tid < 32u )
        {
            uint32_t newly = 0u;
            uint64_t seen = 0u,diag = 0u,fault = 0u;
            for (;;)
            {
                uint32_t just = 0u;
                if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
                {
                    missing = 0u;
                    just = 1u;
                    arr_landed = SparkTpFastTimerNs();
                }
                newly = __ballot_sync(0xffffffffu,just != 0u);
                if ( newly != 0u ) break;
                if ( tid == 0u )
                {
                    const uint64_t now = SparkTpFastTimerNs();
                    if ( SparkTpLdcvU64(&control->error_word) != 0u ) fault = UINT64_MAX;
                    else if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected )
                        fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
                    else if ( now > started && now - started > timeout_ns )
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

extern "C" SparkGpuError SparkTpMeshHardwarePrepareFast(void *host,void **device_out)
{
    const void *const kernels[] = {
        (const void *)SparkTpMeshFastPublishKernel,
        (const void *)SparkTpMeshFastPairGatherKernel,
        (const void *)SparkTpMeshFastAllToAllStagePublishKernel,
        (const void *)SparkTpMeshFastStreamRecvKernel,
        (const void *)SparkTpMeshFastStreamAllReduceKernel
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
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER &&
         (slice_routes & (SPARK_TP_MESH_ROUTES_PAIR | SPARK_TP_MESH_ROUTES_PAIR_FIRST)) == 0u &&
         SparkTpMeshDirectPipelined(elements,degree,0u,slot_bytes) == 0u )
    {
        const uint64_t local_elements = elements / degree;
        if ( local_elements <= SparkTpMeshDirectCapacity(slot_bytes,0u) )
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
            return SparkTpMeshFastLaunchStreamRecv(stream,(const uint8_t *)band,slot_bytes,slots_per_rank,control,
                rank,degree,stage == SPARK_TP_MESH_STAGE_ALL ? (const uint16_t *)local : 0,(uint16_t *)output,0u,local_elements,local_elements,timeout_ns);
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

static inline void SparkTpMeshFastLaunchersInstall(void)
{
    if ( SparkTpDeviceCollectiveRegisterLaunchers != 0 )
        SparkTpDeviceCollectiveRegisterLaunchers(&SparkTpMeshFastLaunchers);
}

__attribute__((constructor)) static void SparkTpMeshFastLaunchersInit(void)
{
    SparkTpMeshFastLaunchersInstall();
}

#endif
