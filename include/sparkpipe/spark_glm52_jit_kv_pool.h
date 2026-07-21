#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

// Mooncake-style JIT KV fragment pool for the expert-queue batch plane.
// Host-side residency and transfer scheduling only; DMA is performed by the
// caller against the plans this pool emits. Deterministic: identical call
// sequences produce identical plans. Fragments are 64-token latent-KV blocks
// for the rank's own pipeline layers; the pool tiers them DRAM <-> NVMe and
// schedules stage-ins against each request wave's known arrival time, so the
// batch plane's long transit becomes prefetch lead and NVMe latency is hidden
// behind bandwidth budgeting.

#define SPARK_GLM52_JIT_KV_POOL_ABI_VERSION 1u
#define SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS 262144u
#define SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS 4096u
#define SPARK_GLM52_JIT_KV_POOL_FRAGMENT_TOKENS 64u

#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_FREE 0u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME 1u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_IN 2u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM 3u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_OUT 4u

typedef struct SparkGlm52JitKvFragment
{
	uint64_t sequence_id;
	uint64_t next_need_ns;
	uint32_t state;
	uint32_t fragment_index_in_sequence;
	uint32_t heap_position;
} SparkGlm52JitKvFragment;

typedef struct SparkGlm52JitKvTransfer
{
	uint32_t fragment_id;
	uint32_t direction_in;
	uint64_t start_ns;
	uint64_t done_ns;
} SparkGlm52JitKvTransfer;

typedef struct SparkGlm52JitKvPoolConfiguration
{
	uint32_t abi_version;
	uint32_t fragment_capacity;
	uint32_t dram_fragment_capacity;
	uint64_t fragment_bytes;
	uint64_t nvme_bytes_per_second;
} SparkGlm52JitKvPoolConfiguration;

typedef struct SparkGlm52JitKvPool
{
	uint32_t abi_version;
	uint32_t fragment_capacity;
	uint32_t dram_fragment_capacity;
	uint32_t dram_resident_count;
	uint32_t staging_in_count;
	uint32_t staging_out_count;
	uint64_t fragment_bytes;
	uint64_t nvme_bytes_per_second;
	uint64_t nvme_busy_until_ns;
	uint64_t stage_in_count;
	uint64_t stage_out_count;
	uint64_t hit_count;
	uint64_t miss_count;
	uint64_t late_count;
	uint64_t overflow_drain_count;
	uint32_t eviction_heap_count;
	uint32_t transfer_head;
	uint32_t transfer_count;
	SparkGlm52JitKvFragment fragments[SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS];
	SparkGlm52JitKvTransfer transfers[SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS];
	uint32_t eviction_heap[SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS];
} SparkGlm52JitKvPool;

SparkStatus SparkGlm52JitKvPoolInitialize(SparkGlm52JitKvPool *pool,const SparkGlm52JitKvPoolConfiguration *configuration);
SparkStatus SparkGlm52JitKvPoolAdmitFragment(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint64_t sequence_id,uint32_t fragment_index_in_sequence,uint32_t initial_state);
SparkStatus SparkGlm52JitKvPoolRequireByEta(SparkGlm52JitKvPool *pool,uint64_t now_ns,const uint32_t *fragment_ids,uint32_t fragment_count,uint64_t need_ns);
SparkStatus SparkGlm52JitKvPoolTick(SparkGlm52JitKvPool *pool,uint64_t now_ns);
uint32_t SparkGlm52JitKvPoolFragmentIsResident(const SparkGlm52JitKvPool *pool,uint32_t fragment_id);
