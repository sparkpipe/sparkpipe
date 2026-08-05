#pragma once

// The staged tile pipeline. One implementation, shared by every GEMM.
//
// The old tree had this written four times with four different schedules: two
// BF16 bodies selected by a policy macro, an FP8 body in a header excluded from
// every build, and a fourth inside the glm52 decode stage. Each had its own
// prologue, its own phase arithmetic and its own idea of when a stage was free.
// Three of the four never ran.
//
// SCHEDULE. Stage s holds K tile s, s+STAGES, s+2*STAGES and so on. Before
// consuming tile t the loop issues tile t+STAGES-1, so a transfer is always in
// flight across the wait. The CTA-wide __syncthreads every stage already needs -
// all warps read the same staged tile - is also what establishes that the stage
// being refilled was consumed, so there is no second "empty" barrier. One
// mbarrier per stage, nothing else.
//
// DEPTH. Two stages, a lookahead of one. Little's Law against 218 GB/s
// effective and 400-600 ns latency wants about 2.3 KB in flight per SM; a single
// tile per CTA clears that twenty times over. Depth past that satisfies a
// requirement already met and costs occupancy - six stages of a 16-row NVFP4
// tile is 110 KB and admits one CTA per SM, where two stages is 37 KB and
// admits three. The latency figure has never been measured in this repo, which
// is the one thing that could move this number.
//
// The stage arithmetic below - and the per-stage wait parity the GEMM carries
// across tiles, since a barrier's phase outlives one tile - is checked
// exhaustively by tests/test_mma_fragment_mapping.c across stages 2..6,
// k_tiles 1..48 and persistent multi-tile schedules: every tile consumed after
// it was produced, no stage refilled before consumption, wait parity matching
// the per-stage completion count.

#include "inference/kernels/activation.cuh"
#include "inference/kernels/layout.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/tma.cuh"
#include <stdint.h>

// The static __shared__ limit is 48 KB per block, not the 128 KB of L1/shared
// the SM has. ptxas enforces it as "uses too much shared data (0xc000 max)" and
// nothing before nvcc could see it - every geometry in this file was checked
// against 131072 and was wrong by 2.7x.
//
// Exceeding it needs DYNAMIC shared memory plus a runtime opt-in through
// cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize). That is
// what CUTLASS and Marlin do and what the launcher must do here; the constant
// below is the static ceiling a __shared__ declaration must respect.

#define LM_PIPELINE_STAGES 2u
#define LM_PIPELINE_LOOKAHEAD (LM_PIPELINE_STAGES - 1u)

// Ring position of a K tile.
static __device__ __forceinline__ uint32_t LmPipelineStage(uint32_t k_tile, uint32_t stages)
{
	return(k_tile % stages);
}

// An mbarrier flips phase on each completion, so the wait parity for a stage
// is the number of times that stage has already completed, kept modulo two.
// The barriers outlive one output tile, so that count is cumulative across the
// persistent tile loop and the GEMM carries it as one bit per stage rather
// than deriving it from k.

// The tile to issue while tile k_tile is being consumed.
static __device__ __forceinline__ uint32_t LmPipelineAhead(uint32_t k_tile, uint32_t stages)
{
	return(k_tile + stages - 1u);
}

template<uint32_t STAGES>
static __device__ void LmPipelineInitialise(uint64_t *barrier, uint32_t arrive_count)
{
	uint32_t stage;
	if ( threadIdx.x == 0u )
		for (stage = 0u; stage < STAGES; ++stage)
			LmMbarrierInit(&barrier[stage],arrive_count);
	LmMbarrierInitFence();
	__syncthreads();
}

template<uint32_t STAGES>
static __device__ void LmPipelineRelease(uint64_t *barrier)
{
	uint32_t stage;
	__syncthreads();
	if ( threadIdx.x == 0u )
		for (stage = 0u; stage < STAGES; ++stage)
			LmMbarrierInvalidate(&barrier[stage]);
}

// Geometry of one staged tile pair, in bytes. Separate from the kernel so the
// host can compute the same numbers when it sizes shared memory and chooses a
// tile height, and so a mismatch between the two is a test failure rather than
// a launch failure.
//
// Element width is in BITS because E2M1 is four, and a tile extent that assumes
// one byte per element is wrong by a factor of two in a way that still runs.
//
// constexpr so a kernel's geometry is a compile-time contract. A tile that does
// not fit shared memory, or a K extent too narrow to swizzle, should be a
// static_assert at the instantiation rather than a launch failure on the ring -
// the sparkdev should not be debugging a configuration the compiler could have
// rejected.


// A row pitch no span divides cannot be permuted at all, which is a real
// rejection rather than a hypothetical: the compiler produced it for INT7 at a
// 128-element tile during this file's own development. What each stored width
// can do:
//
//     bits   TILE_K   pitch   span   shared M16 x2   fits 48 KB static
//       16      128     256   128B          73,760          no, needs dynamic
//        8      128     128   128B          36,896          yes
//        7      256     224    32B          64,544          no, needs dynamic
//        6      128      96    32B          27,680          yes
//        4      128      64    64B          18,464          yes

// The span this tile will actually use. 128 where the pitch allows it, 32 for
// the sub-byte widths that no larger span divides.

// -- staging -----------------------------------------------------------------
//
// One elected thread issues both boxes and declares their total. The expected
// byte count must equal the sum of everything issued into the stage or the
// barrier never flips, so it is derived here from the same LmTileBytes the host
// sizes shared memory with rather than passed in.

struct LmTileGeometry
{
	uint32_t rows;
	uint32_t depth;
	uint32_t element_bits;
};

static __device__ __forceinline__ void LmPipelineProduce(
    const LmTileGeometry *a,
    const LmTileGeometry *b,
    const void *tensor_map_a,
    const void *tensor_map_b,
    void *stage_a,
    void *stage_b,
    uint64_t *barrier,
    uint32_t row_base,
    uint32_t neuron_base,
    uint32_t k_tile,
    uint32_t group_index,
    bool grouped)
{
	// THE K COORDINATE IS BYTES, PER OPERAND. Every descriptor is a UINT8
	// tensor, so the x coordinate advances in bytes - and a BF16 activation
	// against a 4-bit weight advances four times as far per K tile. One shared
	// byte offset was correct only while both operands were the same width,
	// which is the assumption the weight-only path retires. The caller passes
	// the K TILE INDEX and each operand prices its own stride.
	uint32_t k_byte_a = k_tile * LmTileBytes(1u,a->depth,a->element_bits);
	uint32_t k_byte_b = k_tile * LmTileBytes(1u,b->depth,b->element_bits);
	if ( threadIdx.x != 0u )
		return;
	LmMbarrierArriveExpect(barrier,
		LmTileBytes(a->rows,a->depth,a->element_bits)
		+ LmTileBytes(b->rows,b->depth,b->element_bits));
	LmTmaLoad2d(stage_a,tensor_map_a,barrier,(int32_t)k_byte_a,(int32_t)row_base);
	// Weights are expert-major, so one rank-3 descriptor covers every expert and
	// the third coordinate selects. A dense GEMM is the same tensor with one
	// group, which is why there is no separate dense staging path.
	if ( grouped )
		LmTmaLoad3d(stage_b,tensor_map_b,barrier,(int32_t)k_byte_b,(int32_t)neuron_base,(int32_t)group_index);
	else
		LmTmaLoad2d(stage_b,tensor_map_b,barrier,(int32_t)k_byte_b,(int32_t)neuron_base);
}

template<class FormatA,uint32_t TILE_ROWS,uint32_t TILE_K,uint32_t ACTIVATION_CODEC>
static __device__ __forceinline__ void LmPipelineProduceManualA(
	const LmTileGeometry *a,
	const LmTileGeometry *b,
	const void *tensor_map_b,
	const uint8_t *activation_bytes,
	const uint32_t *source_row_map,
	void *stage_a,
	void *stage_b,
	uint64_t *barrier,
	uint32_t row_base,
	uint32_t row_limit,
	uint32_t source_row_count,
	uint32_t input_dimension,
	uint32_t neuron_base,
	uint32_t k_tile,
	uint32_t group_index,
	bool grouped)
{
	const uint32_t row_pitch = LmTileBytes(1u,a->depth,a->element_bits);
	const uint32_t chunk_count = row_pitch / LM_SWIZZLE_CHUNK_BYTES;
	uint32_t index,local_row,chunk,packed_row,source_row,destination_offset;
	uint64_t source_offset;
	const uint4 *source;
	uint4 *destination;
	static_assert(FormatA::kStoredBits % 8u == 0u,
		"indirect activation staging requires byte-addressable rows");
	static_assert(LmTileBytes(1u,FormatA::kTileK,FormatA::kStoredBits) % LM_SWIZZLE_CHUNK_BYTES == 0u,
		"indirect activation staging requires complete 16-byte chunks");
	if ( threadIdx.x == 0u )
	{
		LmMbarrierArriveExpect(barrier,LmTileBytes(b->rows,b->depth,b->element_bits));
		if ( grouped )
			LmTmaLoad3d(stage_b,tensor_map_b,barrier,
				(int32_t)(k_tile * LmTileBytes(1u,b->depth,b->element_bits)),
				(int32_t)neuron_base,(int32_t)group_index);
		else
			LmTmaLoad2d(stage_b,tensor_map_b,barrier,
				(int32_t)(k_tile * LmTileBytes(1u,b->depth,b->element_bits)),
				(int32_t)neuron_base);
	}
	if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
	{
		static_assert(FormatA::kStoredBits == 16u,"FP8 QDQ source must be BF16");
		LmActivationStageFp8Qdq<TILE_ROWS,TILE_K,FormatA::kTmaSwizzle,ACTIVATION_CODEC>(
			activation_bytes,source_row_map,source_row_count,row_base,row_limit,
			k_tile * TILE_K,input_dimension,stage_a,0u,blockDim.x / 32u);
	}
	else
		for (index=threadIdx.x; index<a->rows * chunk_count; index+=blockDim.x)
		{
			local_row = index / chunk_count;
			chunk = index % chunk_count;
			packed_row = row_base + local_row < row_limit ? row_base + local_row : row_base;
			source_row = source_row_map != 0 ? source_row_map[packed_row] : packed_row;
			if ( source_row >= source_row_count )
				asm volatile("trap;\n");
			source_offset = ((source_row * input_dimension * FormatA::kStoredBits) / 8u)
				+ (k_tile * row_pitch) + (chunk * LM_SWIZZLE_CHUNK_BYTES);
			destination_offset = FormatA::kTmaSwizzle
				? LmSwizzledOffset(local_row,chunk * LM_SWIZZLE_CHUNK_BYTES,row_pitch,LmSwizzleSpanFor(row_pitch))
				: (local_row * row_pitch) + (chunk * LM_SWIZZLE_CHUNK_BYTES);
			source = (const uint4 *)(activation_bytes + source_offset);
			destination = (uint4 *)((uint8_t *)stage_a + destination_offset);
			*destination = *source;
		}
}

// -- grouped tile scheduling -------------------------------------------------
//
// Tile index to group, by binary search over a prefix the route build wrote.
// O(log G); a scan would be O(G) per tile and O(G * tiles) overall, which is the
// shape this codebase forbids.
//
// The prefix is also where the true tile count lives. A host-side estimate
// cannot know it - it is a function of per-expert row counts, which only exist
// on device - and bounding the grid loop on an estimate computed from average
// rows launches tiles for groups that received none. Each of those still streams
// a full weight tile before its stores are rejected, which at low batch is tens
// of times the necessary DRAM traffic on the term that is 96 percent of all of
// it.
static __device__ __forceinline__ uint32_t LmGroupOfTile(const uint32_t *group_tile_prefix, uint32_t group_count, uint32_t tile_index)
{
	uint32_t low = 0u,high = group_count,middle;
	while ( low + 1u < high )
	{
		middle = (low + high) >> 1u;
		if ( group_tile_prefix[middle] <= tile_index )
			low = middle;
		else
			high = middle;
	}
	return(low);
}

static __device__ __forceinline__ uint32_t LmTotalTiles(const uint32_t *group_tile_prefix, uint32_t group_count)
{
	return(group_tile_prefix[group_count]);
}
