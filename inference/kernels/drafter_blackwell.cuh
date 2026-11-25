#pragma once

#include <stdint.h>
#include <string.h>

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_status.h"

#define SPARK_DRAFTER_GRAPH_SLOTS 16u
#define SPARK_DRAFTER_TOPK_CHUNKS 16u
#define SPARK_DRAFTER_TOPK_BUCKETS 2048u
#define SPARK_DRAFTER_TOPK_LSB 32u
#define SPARK_DRAFTER_TOPK_THREADS 256u
#define SPARK_DRAFTER_TOPK_WARPS (SPARK_DRAFTER_TOPK_THREADS / 32u)

typedef struct SparkDrafterGraphSlot
{
	uint64_t key;
	uint64_t last_used;
	SparkGpuGraphExec exec;
	uint32_t valid;
	uint32_t warmed;
} SparkDrafterGraphSlot;

typedef struct SparkDrafterGraphCache
{
	uint64_t clock;
	uint32_t disabled;
	uint32_t reserved;
	SparkDrafterGraphSlot slots[SPARK_DRAFTER_GRAPH_SLOTS];
} SparkDrafterGraphCache;

static inline void SparkDrafterGraphCacheDestroy(SparkDrafterGraphCache *cache)
{
	if ( cache == 0 )
		return;
	for ( uint32_t index = 0u; index < SPARK_DRAFTER_GRAPH_SLOTS; index++ )
	{
		if ( cache->slots[index].valid != 0u && cache->slots[index].exec != 0 )
			(void)SparkGpuGraphExecDestroy(cache->slots[index].exec);
		memset(&cache->slots[index], 0, sizeof(cache->slots[index]));
	}
}

template<class ReplayFn>
static inline SparkStatus SparkDrafterGraphRun(SparkDrafterGraphCache *cache, SparkGpuStream stream, uint64_t key, ReplayFn &&replay)
{
	if ( cache == 0 || cache->disabled != 0u )
		return replay();
	SparkDrafterGraphSlot *slot = 0;
	for ( uint32_t index = 0u; index < SPARK_DRAFTER_GRAPH_SLOTS; index++ )
	{
		if ( (cache->slots[index].valid != 0u || cache->slots[index].warmed != 0u) && cache->slots[index].key == key )
		{
			slot = &cache->slots[index];
			break;
		}
	}
	if ( slot != 0 && slot->valid != 0u )
	{
		slot->last_used = ++cache->clock;
		return SparkGpuGraphLaunch(slot->exec, stream) == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}
	if ( slot == 0 )
	{
		for ( uint32_t index = 0u; index < SPARK_DRAFTER_GRAPH_SLOTS; index++ )
			if ( cache->slots[index].valid == 0u && cache->slots[index].warmed == 0u )
			{
				slot = &cache->slots[index];
				break;
			}
		if ( slot == 0 )
		{
			slot = &cache->slots[0];
			for ( uint32_t index = 1u; index < SPARK_DRAFTER_GRAPH_SLOTS; index++ )
				if ( cache->slots[index].last_used < slot->last_used )
					slot = &cache->slots[index];
			if ( slot->valid != 0u && slot->exec != 0 )
				(void)SparkGpuGraphExecDestroy(slot->exec);
			memset(slot, 0, sizeof(*slot));
		}
		slot->key = key;
		slot->warmed = 1u;
		slot->last_used = ++cache->clock;
		return replay();
	}
	slot->last_used = ++cache->clock;
	if ( SparkGpuStreamBeginCapture(stream, SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL) != SPARK_GPU_SUCCESS )
	{
		cache->disabled = 1u;
		return replay();
	}
	SparkStatus status = replay();
	SparkGpuGraph graph = 0;
	if ( SparkGpuStreamEndCapture(stream, &graph) != SPARK_GPU_SUCCESS || status != SPARK_STATUS_OK || graph == 0 )
	{
		if ( graph != 0 )
			(void)SparkGpuGraphDestroy(graph);
		cache->disabled = 1u;
		return replay();
	}
	SparkGpuGraphExec exec = 0;
	if ( SparkGpuGraphInstantiate(&exec, graph, 0ull) != SPARK_GPU_SUCCESS || exec == 0 )
	{
		(void)SparkGpuGraphDestroy(graph);
		cache->disabled = 1u;
		return replay();
	}
	(void)SparkGpuGraphDestroy(graph);
	if ( SparkGpuGraphUpload(exec, stream) != SPARK_GPU_SUCCESS || SparkGpuGraphLaunch(exec, stream) != SPARK_GPU_SUCCESS )
	{
		(void)SparkGpuGraphExecDestroy(exec);
		cache->disabled = 1u;
		return replay();
	}
	slot->exec = exec;
	slot->valid = 1u;
	return SPARK_STATUS_OK;
}

static __device__ __forceinline__ void SparkDrafterStage16(uint32_t destination, const void *source, uint32_t bytes)
{
#if defined(__CUDA_ARCH__)
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
#else
	(void)destination;
	(void)source;
	(void)bytes;
#endif
}

static __device__ __forceinline__ void SparkDrafterCommitGroup(void)
{
#if defined(__CUDA_ARCH__)
	asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template<uint32_t GROUPS>
static __device__ __forceinline__ void SparkDrafterWaitGroup(void)
{
#if defined(__CUDA_ARCH__)
	if constexpr ( GROUPS == 0u )
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	else
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}

static __device__ __forceinline__ uint32_t SparkDrafterBf16Key16(uint16_t raw)
{
	return (uint32_t)raw ^ ((raw & 0x8000u) != 0u ? 0xffffu : 0x8000u);
}

typedef struct SparkDrafterTopkBf16Workspace
{
	uint32_t *hist;
	uint32_t *chunk_gt;
	uint32_t *chunk_lsb;
	uint32_t max_rows;
} SparkDrafterTopkBf16Workspace;

static inline SparkStatus SparkDrafterTopkBf16Alloc(SparkDrafterTopkBf16Workspace *ws, uint32_t max_rows)
{
	memset(ws, 0, sizeof(*ws));
	ws->max_rows = max_rows;
	if ( SparkGpuMalloc((void **)&ws->hist, (size_t)max_rows * SPARK_DRAFTER_TOPK_BUCKETS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&ws->chunk_gt, (size_t)max_rows * SPARK_DRAFTER_TOPK_CHUNKS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&ws->chunk_lsb, (size_t)max_rows * SPARK_DRAFTER_TOPK_CHUNKS * SPARK_DRAFTER_TOPK_LSB * sizeof(uint32_t)) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static inline void SparkDrafterTopkBf16Free(SparkDrafterTopkBf16Workspace *ws)
{
	if ( ws == 0 )
		return;
	(void)SparkGpuFree(ws->hist);
	(void)SparkGpuFree(ws->chunk_gt);
	(void)SparkGpuFree(ws->chunk_lsb);
	memset(ws, 0, sizeof(*ws));
}

static __device__ __forceinline__ uint32_t SparkDrafterTopkFindMsb(const uint32_t *row_hist, uint32_t k, uint32_t *sh_slice, uint32_t *gt_above_out)
{
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	uint32_t slice_sum = 0u;
#pragma unroll
	for ( uint32_t j = 0u; j < 8u; j++ )
		slice_sum += row_hist[warp * 256u + j * 32u + lane];
#pragma unroll
	for ( uint32_t mask = 16u; mask > 0u; mask >>= 1u )
		slice_sum += __shfl_xor_sync(0xffffffffu, slice_sum, mask);
	if ( lane == 0u )
		sh_slice[warp] = slice_sum;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		uint32_t running = 0u, w = SPARK_DRAFTER_TOPK_WARPS - 1u;
		while ( w > 0u && running + sh_slice[w] < k )
		{
			running += sh_slice[w];
			w--;
		}
		uint32_t bucket = w * 256u + 255u;
		while ( bucket > w * 256u && running + row_hist[bucket] < k )
		{
			running += row_hist[bucket];
			bucket--;
		}
		sh_slice[0] = bucket;
		sh_slice[1] = running;
	}
	__syncthreads();
	if ( gt_above_out != 0 )
		*gt_above_out = sh_slice[1];
	return sh_slice[0];
}

__global__ __launch_bounds__(SPARK_DRAFTER_TOPK_THREADS, 2)
void SparkDrafterTopkBf16Pass1Kernel(const uint16_t *__restrict__ logits, uint32_t vocab, uint32_t *__restrict__ hist)
{
	__shared__ uint32_t sh[SPARK_DRAFTER_TOPK_BUCKETS];
	const uint32_t row = blockIdx.y, chunk = blockIdx.x;
	const uint32_t span = ((vocab + SPARK_DRAFTER_TOPK_CHUNKS - 1u) / SPARK_DRAFTER_TOPK_CHUNKS + 7u) & ~7u;
	const uint32_t begin = chunk * span;
	const uint32_t end = begin + span < vocab ? begin + span : vocab;
	const uint16_t *src = logits + (uint64_t)row * vocab;
#pragma unroll
	for ( uint32_t j = 0u; j < SPARK_DRAFTER_TOPK_BUCKETS / SPARK_DRAFTER_TOPK_THREADS; j++ )
		sh[j * SPARK_DRAFTER_TOPK_THREADS + threadIdx.x] = 0u;
	__syncthreads();
	if ( begin < end )
	{
		if ( (vocab & 7u) == 0u )
		{
			const uint4 *vsrc = (const uint4 *)src;
			const uint32_t vec_begin = begin >> 3u, vec_end = end >> 3u;
			for ( uint32_t v = vec_begin + threadIdx.x; v < vec_end; v += SPARK_DRAFTER_TOPK_THREADS )
			{
				const uint4 p = __ldg(vsrc + v);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.x & 0xffffu)) >> 5u], 1u);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.x >> 16u)) >> 5u], 1u);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.y & 0xffffu)) >> 5u], 1u);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.y >> 16u)) >> 5u], 1u);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.z & 0xffffu)) >> 5u], 1u);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.z >> 16u)) >> 5u], 1u);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.w & 0xffffu)) >> 5u], 1u);
				atomicAdd(&sh[SparkDrafterBf16Key16((uint16_t)(p.w >> 16u)) >> 5u], 1u);
			}
		}
		else
		{
			for ( uint32_t i = begin + threadIdx.x; i < end; i += SPARK_DRAFTER_TOPK_THREADS )
				atomicAdd(&sh[SparkDrafterBf16Key16(__ldg(src + i)) >> 5u], 1u);
		}
	}
	__syncthreads();
	uint32_t *row_hist = hist + (uint64_t)row * SPARK_DRAFTER_TOPK_BUCKETS;
#pragma unroll
	for ( uint32_t j = 0u; j < SPARK_DRAFTER_TOPK_BUCKETS / SPARK_DRAFTER_TOPK_THREADS; j++ )
	{
		const uint32_t b = j * SPARK_DRAFTER_TOPK_THREADS + threadIdx.x;
		const uint32_t count = sh[b];
		if ( count != 0u )
			atomicAdd(&row_hist[b], count);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(SPARK_DRAFTER_TOPK_THREADS, 2)
void SparkDrafterTopkBf16Pass2Kernel(const uint16_t *__restrict__ logits, uint32_t vocab, uint32_t k, const uint32_t *__restrict__ hist,
	uint32_t *__restrict__ chunk_gt, uint32_t *__restrict__ chunk_lsb)
{
	LmDependentWait();
	__shared__ uint32_t sh_slice[SPARK_DRAFTER_TOPK_WARPS];
	__shared__ uint32_t sh_lsb[SPARK_DRAFTER_TOPK_LSB];
	const uint32_t row = blockIdx.y, chunk = blockIdx.x;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t span = ((vocab + SPARK_DRAFTER_TOPK_CHUNKS - 1u) / SPARK_DRAFTER_TOPK_CHUNKS + 7u) & ~7u;
	const uint32_t begin = chunk * span;
	const uint32_t end = begin + span < vocab ? begin + span : vocab;
	const uint16_t *src = logits + (uint64_t)row * vocab;
	const uint32_t msb_t = SparkDrafterTopkFindMsb(hist + (uint64_t)row * SPARK_DRAFTER_TOPK_BUCKETS, k, sh_slice, (uint32_t *)0);
	uint32_t my_gt = 0u;
	if ( threadIdx.x < SPARK_DRAFTER_TOPK_LSB )
		sh_lsb[threadIdx.x] = 0u;
	__syncthreads();
	if ( begin < end )
	{
		if ( (vocab & 7u) == 0u )
		{
			const uint4 *vsrc = (const uint4 *)src;
			const uint32_t vec_begin = begin >> 3u, vec_end = end >> 3u;
			for ( uint32_t v = vec_begin + threadIdx.x; v < vec_end; v += SPARK_DRAFTER_TOPK_THREADS )
			{
				const uint4 p = __ldg(vsrc + v);
				const uint16_t h[8] = {
					(uint16_t)(p.x & 0xffffu), (uint16_t)(p.x >> 16u), (uint16_t)(p.y & 0xffffu), (uint16_t)(p.y >> 16u),
					(uint16_t)(p.z & 0xffffu), (uint16_t)(p.z >> 16u), (uint16_t)(p.w & 0xffffu), (uint16_t)(p.w >> 16u),
				};
#pragma unroll
				for ( uint32_t j = 0u; j < 8u; j++ )
				{
					const uint32_t key = SparkDrafterBf16Key16(h[j]);
					const uint32_t msb = key >> 5u;
					if ( msb > msb_t )
						my_gt++;
					else if ( msb == msb_t )
						atomicAdd(&sh_lsb[key & 31u], 1u);
				}
			}
		}
		else
		{
			for ( uint32_t i = begin + threadIdx.x; i < end; i += SPARK_DRAFTER_TOPK_THREADS )
			{
				const uint32_t key = SparkDrafterBf16Key16(__ldg(src + i));
				const uint32_t msb = key >> 5u;
				if ( msb > msb_t )
					my_gt++;
				else if ( msb == msb_t )
					atomicAdd(&sh_lsb[key & 31u], 1u);
			}
		}
	}
#pragma unroll
	for ( uint32_t mask = 16u; mask > 0u; mask >>= 1u )
		my_gt += __shfl_xor_sync(0xffffffffu, my_gt, mask);
	if ( lane == 0u )
		sh_slice[warp] = my_gt;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		uint32_t total_gt = 0u;
#pragma unroll
		for ( uint32_t w = 0u; w < SPARK_DRAFTER_TOPK_WARPS; w++ )
			total_gt += sh_slice[w];
		chunk_gt[(uint64_t)row * SPARK_DRAFTER_TOPK_CHUNKS + chunk] = total_gt;
	}
	if ( threadIdx.x < SPARK_DRAFTER_TOPK_LSB )
		chunk_lsb[((uint64_t)row * SPARK_DRAFTER_TOPK_CHUNKS + chunk) * SPARK_DRAFTER_TOPK_LSB + threadIdx.x] = sh_lsb[threadIdx.x];
	LmDependentRelease();
}

__global__ __launch_bounds__(SPARK_DRAFTER_TOPK_THREADS, 2)
void SparkDrafterTopkBf16Pass3Kernel(const uint16_t *__restrict__ logits, uint32_t vocab, uint32_t k, const uint32_t *__restrict__ hist,
	const uint32_t *__restrict__ chunk_gt, const uint32_t *__restrict__ chunk_lsb, uint32_t *__restrict__ out)
{
	LmDependentWait();
	__shared__ uint32_t sh_slice[SPARK_DRAFTER_TOPK_WARPS];
	__shared__ uint32_t sh_total_lsb[SPARK_DRAFTER_TOPK_LSB];
	__shared__ uint32_t sh_scan_gt[SPARK_DRAFTER_TOPK_WARPS];
	__shared__ uint32_t sh_scan_eq[SPARK_DRAFTER_TOPK_WARPS];
	__shared__ uint32_t s_threshold;
	__shared__ uint32_t s_emitted;
	__shared__ uint32_t s_eq_rem;
	__shared__ uint32_t s_stop;
	const uint32_t row = blockIdx.y, chunk = blockIdx.x;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t span = ((vocab + SPARK_DRAFTER_TOPK_CHUNKS - 1u) / SPARK_DRAFTER_TOPK_CHUNKS + 7u) & ~7u;
	const uint32_t begin = chunk * span;
	const uint32_t end = begin + span < vocab ? begin + span : vocab;
	uint32_t gt_above_msb = 0u;
	const uint32_t msb_t = SparkDrafterTopkFindMsb(hist + (uint64_t)row * SPARK_DRAFTER_TOPK_BUCKETS, k, sh_slice, &gt_above_msb);
	const uint32_t *row_cgt = chunk_gt + (uint64_t)row * SPARK_DRAFTER_TOPK_CHUNKS;
	const uint32_t *row_clsb = chunk_lsb + (uint64_t)row * SPARK_DRAFTER_TOPK_CHUNKS * SPARK_DRAFTER_TOPK_LSB;
	if ( threadIdx.x < SPARK_DRAFTER_TOPK_LSB )
	{
		uint32_t sum = 0u;
#pragma unroll
		for ( uint32_t c = 0u; c < SPARK_DRAFTER_TOPK_CHUNKS; c++ )
			sum += row_clsb[c * SPARK_DRAFTER_TOPK_LSB + threadIdx.x];
		sh_total_lsb[threadIdx.x] = sum;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		const uint32_t need = k > gt_above_msb ? k - gt_above_msb : 0u;
		uint32_t running = 0u, lsb_t = SPARK_DRAFTER_TOPK_LSB - 1u;
		while ( lsb_t > 0u && running + sh_total_lsb[lsb_t] < need )
		{
			running += sh_total_lsb[lsb_t];
			lsb_t--;
		}
		const uint32_t need_eq_total = need > running ? need - running : 0u;
		uint32_t gt_before = 0u, eq_before = 0u, my_gt = 0u, my_eq = 0u;
		for ( uint32_t c = 0u; c <= chunk; c++ )
		{
			uint32_t c_gt = row_cgt[c];
			for ( uint32_t l = lsb_t + 1u; l < SPARK_DRAFTER_TOPK_LSB; l++ )
				c_gt += row_clsb[c * SPARK_DRAFTER_TOPK_LSB + l];
			const uint32_t c_eq = row_clsb[c * SPARK_DRAFTER_TOPK_LSB + lsb_t];
			if ( c < chunk )
			{
				gt_before += c_gt;
				eq_before += c_eq;
			}
			else
			{
				my_gt = c_gt;
				my_eq = c_eq;
			}
		}
		const uint32_t eq_quota = need_eq_total > eq_before ? need_eq_total - eq_before : 0u;
		const uint32_t my_eq_take = my_eq < eq_quota ? my_eq : eq_quota;
		const uint32_t out_start = gt_before + (eq_before < need_eq_total ? eq_before : need_eq_total);
		s_threshold = (msb_t << 5u) | lsb_t;
		s_emitted = out_start;
		s_eq_rem = my_eq_take;
		s_stop = out_start + my_gt + my_eq_take;
	}
	__syncthreads();
	const uint32_t threshold = s_threshold, stop_out = s_stop;
	if ( s_emitted >= stop_out || begin >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *src = logits + (uint64_t)row * vocab;
	uint32_t *row_out = out + (uint64_t)row * k;
	if ( (vocab & 7u) == 0u )
	{
		const uint4 *vsrc = (const uint4 *)src;
		const uint32_t vec_begin = begin >> 3u, vec_end = end >> 3u;
		for ( uint32_t vbase = vec_begin; vbase < vec_end && s_emitted < stop_out; vbase += SPARK_DRAFTER_TOPK_THREADS )
		{
			const uint32_t v = vbase + threadIdx.x;
			uint32_t gt_bits = 0u, eq_bits = 0u;
			if ( v < vec_end )
			{
				const uint4 p = __ldg(vsrc + v);
				const uint16_t h[8] = {
					(uint16_t)(p.x & 0xffffu), (uint16_t)(p.x >> 16u), (uint16_t)(p.y & 0xffffu), (uint16_t)(p.y >> 16u),
					(uint16_t)(p.z & 0xffffu), (uint16_t)(p.z >> 16u), (uint16_t)(p.w & 0xffffu), (uint16_t)(p.w >> 16u),
				};
#pragma unroll
				for ( uint32_t j = 0u; j < 8u; j++ )
				{
					const uint32_t key = SparkDrafterBf16Key16(h[j]);
					if ( key > threshold )
						gt_bits |= 1u << j;
					else if ( key == threshold )
						eq_bits |= 1u << j;
				}
			}
			if ( __syncthreads_or((gt_bits | eq_bits) != 0u) )
			{
				const uint32_t my_g = (uint32_t)__popc(gt_bits), my_e = (uint32_t)__popc(eq_bits);
				uint32_t wg = my_g, we = my_e;
#pragma unroll
				for ( uint32_t off = 1u; off < 32u; off <<= 1u )
				{
					const uint32_t pg = __shfl_up_sync(0xffffffffu, wg, off);
					const uint32_t pe = __shfl_up_sync(0xffffffffu, we, off);
					if ( lane >= off )
					{
						wg += pg;
						we += pe;
					}
				}
				if ( lane == 31u )
				{
					sh_scan_gt[warp] = wg;
					sh_scan_eq[warp] = we;
				}
				__syncthreads();
				uint32_t g_before = wg - my_g, e_before = we - my_e, g_total = 0u, e_total = 0u;
#pragma unroll
				for ( uint32_t w = 0u; w < SPARK_DRAFTER_TOPK_WARPS; w++ )
				{
					if ( w < warp )
					{
						g_before += sh_scan_gt[w];
						e_before += sh_scan_eq[w];
					}
					g_total += sh_scan_gt[w];
					e_total += sh_scan_eq[w];
				}
				const uint32_t rem = s_eq_rem;
				if ( (gt_bits | eq_bits) != 0u )
				{
					uint32_t pos = s_emitted + g_before + (e_before < rem ? e_before : rem);
					uint32_t e_used = e_before;
					const uint32_t idx_base = v << 3u;
#pragma unroll
					for ( uint32_t j = 0u; j < 8u; j++ )
					{
						if ( ((gt_bits >> j) & 1u) != 0u )
							row_out[pos++] = idx_base + j;
						else if ( ((eq_bits >> j) & 1u) != 0u && e_used < rem )
						{
							row_out[pos++] = idx_base + j;
							e_used++;
						}
					}
				}
				__syncthreads();
				if ( threadIdx.x == 0u )
				{
					const uint32_t e_take = e_total < rem ? e_total : rem;
					s_emitted += g_total + e_take;
					s_eq_rem = rem - e_take;
				}
				__syncthreads();
			}
		}
	}
	else
	{
		for ( uint32_t base = begin; base < end && s_emitted < stop_out; base += SPARK_DRAFTER_TOPK_THREADS )
		{
			const uint32_t idx = base + threadIdx.x;
			uint32_t gt = 0u, eq = 0u;
			if ( idx < end )
			{
				const uint32_t key = SparkDrafterBf16Key16(__ldg(src + idx));
				gt = key > threshold ? 1u : 0u;
				eq = key == threshold ? 1u : 0u;
			}
			if ( __syncthreads_or((gt | eq) != 0u) )
			{
				const uint32_t below = lane != 0u ? (0xffffffffu >> (32u - lane)) : 0u;
				const uint32_t gm = __ballot_sync(0xffffffffu, gt != 0u), em = __ballot_sync(0xffffffffu, eq != 0u);
				if ( lane == 0u )
				{
					sh_scan_gt[warp] = (uint32_t)__popc(gm);
					sh_scan_eq[warp] = (uint32_t)__popc(em);
				}
				__syncthreads();
				uint32_t g_before = (uint32_t)__popc(gm & below), e_before = (uint32_t)__popc(em & below), g_total = 0u, e_total = 0u;
#pragma unroll
				for ( uint32_t w = 0u; w < SPARK_DRAFTER_TOPK_WARPS; w++ )
				{
					if ( w < warp )
					{
						g_before += sh_scan_gt[w];
						e_before += sh_scan_eq[w];
					}
					g_total += sh_scan_gt[w];
					e_total += sh_scan_eq[w];
				}
				const uint32_t rem = s_eq_rem;
				if ( gt != 0u || (eq != 0u && e_before < rem) )
					row_out[s_emitted + g_before + (e_before < rem ? e_before : rem)] = idx;
				__syncthreads();
				if ( threadIdx.x == 0u )
				{
					const uint32_t e_take = e_total < rem ? e_total : rem;
					s_emitted += g_total + e_take;
					s_eq_rem = rem - e_take;
				}
				__syncthreads();
			}
		}
	}
	LmDependentRelease();
}

static inline SparkStatus SparkDrafterTopkBf16Launch(const uint16_t *logits, uint32_t rows, uint32_t vocab, uint32_t k,
	const SparkDrafterTopkBf16Workspace *ws, uint32_t *out, SparkGpuStream stream)
{
	if ( rows == 0u || rows > ws->max_rows || k == 0u || k > vocab )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( SparkGpuMemsetAsync(ws->hist, 0, (size_t)rows * SPARK_DRAFTER_TOPK_BUCKETS * sizeof(uint32_t), stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	LM_LAUNCH((SparkDrafterTopkBf16Pass1Kernel), dim3(SPARK_DRAFTER_TOPK_CHUNKS, rows), SPARK_DRAFTER_TOPK_THREADS, 0u, stream,
		logits, vocab, ws->hist);
	if ( LM_LAUNCH_DEPENDENT((SparkDrafterTopkBf16Pass2Kernel), dim3(SPARK_DRAFTER_TOPK_CHUNKS, rows), SPARK_DRAFTER_TOPK_THREADS, 0u, stream,
		logits, vocab, k, ws->hist, ws->chunk_gt, ws->chunk_lsb) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( LM_LAUNCH_DEPENDENT((SparkDrafterTopkBf16Pass3Kernel), dim3(SPARK_DRAFTER_TOPK_CHUNKS, rows), SPARK_DRAFTER_TOPK_THREADS, 0u, stream,
		logits, vocab, k, ws->hist, ws->chunk_gt, ws->chunk_lsb, out) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}
