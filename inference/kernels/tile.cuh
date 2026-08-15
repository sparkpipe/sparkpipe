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
#include "inference/kernels/route.cuh"
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
// The weight half is always one elected thread issuing one box; the activation
// half is one box on the dense path and a per-row-indexed set of bulk chunk
// copies on the indirect path. Both declare the same byte total through
// LmPipelineProduceWeight, because the expected count must equal the sum of
// everything issued into the stage or the barrier never flips.

struct LmTileGeometry
{
	uint32_t rows;
	uint32_t depth;
	uint32_t element_bits;
};

// Thread 0's half of a produce, shared by both A-staging paths: declare the
// stage's total bytes and issue the weight box. The expected byte count must
// equal the sum of everything issued into the stage or the barrier never
// flips, so it is derived here from the same LmTileBytes the host sizes shared
// memory with rather than passed in - and it is the SAME total on the indirect
// path, where the A bytes arrive as per-chunk bulk copies instead of one box.
//
// Weights are expert-major, so one rank-3 descriptor covers every expert and
// the third coordinate selects. A dense GEMM is the same tensor with one
// group, which is why there is no separate dense staging path. The weight side
// never has a row indirection - experts own their weight slices outright - so
// this half stays a tensor box on every path.
static __device__ __forceinline__ void LmPipelineProduceWeight(
    const LmTileGeometry *a,
    const LmTileGeometry *b,
    const void *tensor_map_b,
    void *stage_b,
    uint64_t *barrier,
    uint32_t k_byte_b,
    uint32_t neuron_base,
    uint32_t group_index,
    bool grouped)
{
	if ( threadIdx.x != 0u )
		return;
	LmMbarrierArriveExpect(barrier,
		LmTileBytes(a->rows,a->depth,a->element_bits)
		+ LmTileBytes(b->rows,b->depth,b->element_bits));
	if ( grouped )
		LmTmaLoad3d(stage_b,tensor_map_b,barrier,(int32_t)k_byte_b,(int32_t)neuron_base,(int32_t)group_index);
	else
		LmTmaLoad2d(stage_b,tensor_map_b,barrier,(int32_t)k_byte_b,(int32_t)neuron_base);
}

// INTERLEAVED WEIGHT STAGING - pack V2 mxfp4_ws_interleaved_v1 (K3).
//
// The B operand is a cell grid, not a [neuron, k] plane: per expert and per
// 128-element pack k-tile, each 16-neuron cell occupies 17 rows of 64 bytes -
// rows 0..15 payload (one neuron's k-tile, 64 bytes = 128 4-bit elements),
// row 16 the cell's sixteen 4-byte E8M0 group scales. One rank-3 UINT8 tensor
// map describes the whole expert operand ([64, rows_per_expert, experts], box
// [64, 17 * (TILE_N/16), 1], 64B swizzle - the contract in layer.cuh), and one
// bulk copy per pack k-tile stages the cells for the whole tile: the box's y
// coordinate is (k_tile * cells_total + neuron_base/16) * 17, x is 0, z the
// expert. The staged layout IS the cell layout: payload row r of cell c lands
// at stage byte (c*17 + r) * 64 and the cell's scales at row c*17 + 16, which
// LmGemmConsume's interleaved arm reads with the same arithmetic.
static __device__ __forceinline__ void LmPipelineProduceWeightInterleaved(
    const LmTileGeometry *a,
    const LmTileGeometry *b,
    const void *tensor_map_b,
    void *stage_b,
    uint64_t *barrier,
    uint32_t k_tile,
    uint32_t cells_total,
    uint32_t neuron_base,
    uint32_t group_index)
{
	if ( threadIdx.x != 0u )
		return;
	LmMbarrierArriveExpect(barrier,
		LmTileBytes(a->rows,a->depth,a->element_bits)
		+ LmTileBytes(b->rows,b->depth,b->element_bits));
	LmTmaLoad3d(stage_b,tensor_map_b,barrier,0,
		(int32_t)(((k_tile * cells_total) + (neuron_base / 16u)) * 17u),
		(int32_t)group_index);
}

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
    bool grouped,
    bool interleaved_b,
    uint32_t cells_total)
{
	// THE K COORDINATE IS BYTES, PER OPERAND. Every descriptor is a UINT8
	// tensor, so the x coordinate advances in bytes - and a BF16 activation
	// against a 4-bit weight advances four times as far per K tile. One shared
	// byte offset was correct only while both operands were the same width,
	// which is the assumption the weight-only path retires. The caller passes
	// the K TILE INDEX and each operand prices its own stride.
	uint32_t k_byte_a = k_tile * LmTileBytes(1u,a->depth,a->element_bits);
	uint32_t k_byte_b = k_tile * LmTileBytes(1u,b->depth,b->element_bits);
	if ( interleaved_b )
		LmPipelineProduceWeightInterleaved(a,b,tensor_map_b,stage_b,barrier,
			k_tile,cells_total,neuron_base,group_index);
	else
		LmPipelineProduceWeight(a,b,tensor_map_b,stage_b,barrier,k_byte_b,neuron_base,group_index,grouped);
	if ( threadIdx.x != 0u )
		return;
	if ( interleaved_b )
	{
		// A TILE_K=128 BF16 row is 256 bytes and no hardware swizzle spans
		// it, and a TMA box stages its rows at the BOX's width - so the two
		// 128-byte half-row boxes cannot land side by side in one 256-byte
		// row. The stage is therefore TWO BLOCKS of [TILE_M rows x 128
		// bytes]: block 0 holds k 0..63 of every row, block 1 k 64..127,
		// and the consume addresses (block, row, k & 63) with pitch 128.
		// The K coordinate advances WITH THE K TILE: the x coordinates are
		// k_byte_a and k_byte_a + 128, the destination bases are stage_a
		// and stage_a + TILE_M * 128. Both blocks' per-row swizzle selector
		// is r % 8 because block 1 starts TILE_M (a multiple of 8) sectors
		// in - which is the same selector the consume derives, and is why
		// the two 128-byte boxes stage without a third copy path.
		const uint32_t a_block_bytes =
			LmTileBytes(a->rows,64u,a->element_bits);
		LmTmaLoad2d(stage_a,tensor_map_a,barrier,(int32_t)k_byte_a,(int32_t)row_base);
		LmTmaLoad2d((uint8_t *)stage_a + a_block_bytes,
			tensor_map_a,barrier,(int32_t)(k_byte_a + 128u),(int32_t)row_base);
		return;
	}
	LmTmaLoad2d(stage_a,tensor_map_a,barrier,(int32_t)k_byte_a,(int32_t)row_base);
}

// INDIRECT A STAGING - the MoE gather deletion, route.cuh's consumer contract.
//
// A TMA box is affine: the rows it lands are a fixed coordinate window, so a
// per-row indirection can never be a box. When LmGemmArguments carries a
// source_row_map, packed A row p is staged from row
// LmRouteSourceRow(source_row_map, p) of the UN-gathered source tensor
// instead, one cp.async.bulk per 16-byte swizzle chunk - the chunk is the
// largest span the row's swizzle permutation leaves contiguous, so it is the
// granularity the staged layout forces. Every chunk is complete_tx-accounted
// against the same barrier and thread 0 declares the same a_bytes + b_bytes
// total through the same helper, so the barrier protocol - one barrier per
// stage, one ArriveExpect, phase-parity waits, the produce-ahead schedule - is
// IDENTICAL to the TMA path above. Only the copy engine changed, which is the
// roadmap's open hardware question (D9: bulk-16B vs per-thread cp.async
// occupancy), and the variant exists so that question is measured rather than
// guessed.
//
// Threads issue chunks round-robin, consecutive threads on consecutive chunks
// of one source row, so the global side coalesces. A chunk may complete before
// thread 0's expect lands; the mbarrier transaction count is signed and the
// phase completes only when the arrival AND a zero count meet, so issue order
// between the expect and the copies does not matter.
//
// RAGGED TAIL, per the contract: the group's last tile covers
// [row_base, row_base + TILE_M) but valid indices end at row_limit, and the
// bytes past row_limit are the NEXT group's indices - in range and wrong.
// Tail rows clamp to row_base, a live row; their stores are already dropped by
// the GEMM's row_limit check, so the duplicate load is dead traffic, never
// wrong output. The trap is the other half of that guard: a mapped row past
// source_row_count means the route build wrote past its own arrays, and a
// wild bulk copy faults or, worse, does not.
template<class FormatA, bool INTERLEAVED_B = false>
static __device__ __forceinline__ void LmPipelineProduceIndirectA(
	const LmTileGeometry *a,
	const LmTileGeometry *b,
	const void *tensor_map_b,
	const uint8_t *activation_bytes,
	const uint32_t *source_row_map,
	uint32_t source_row_count,
	uint32_t input_dimension,
	void *stage_a,
	void *stage_b,
	uint64_t *barrier,
	uint32_t row_base,
	uint32_t row_limit,
	uint32_t neuron_base,
	uint32_t k_tile,
	uint32_t group_index,
	bool grouped,
	uint32_t cells_total)
{
	const uint32_t row_pitch = LmTileBytes(1u,a->depth,a->element_bits);
	const uint32_t chunk_count = row_pitch / LM_SWIZZLE_CHUNK_BYTES;
	uint32_t index,local_row,chunk,packed_row,source_row,destination_offset;
	uint64_t source_offset;
	static_assert(FormatA::kStoredBits % 8u == 0u,
		"indirect activation staging requires byte-addressable rows");
	static_assert(LmTileBytes(1u,FormatA::kTileK,FormatA::kStoredBits) % LM_SWIZZLE_CHUNK_BYTES == 0u,
		"indirect activation staging requires complete 16-byte chunks");
	if constexpr ( INTERLEAVED_B )
		LmPipelineProduceWeightInterleaved(a,b,tensor_map_b,stage_b,barrier,
			k_tile,cells_total,neuron_base,group_index);
	else
		LmPipelineProduceWeight(a,b,tensor_map_b,stage_b,barrier,
			k_tile * LmTileBytes(1u,b->depth,b->element_bits),neuron_base,group_index,grouped);
	for (index=threadIdx.x; index<a->rows * chunk_count; index+=blockDim.x)
	{
		local_row = index / chunk_count;
		chunk = index % chunk_count;
		packed_row = row_base + local_row < row_limit ? row_base + local_row : row_base;
		source_row = LmRouteSourceRow(source_row_map,packed_row);
		if ( source_row >= source_row_count )
			asm volatile("trap;\n");
		source_offset = ((uint64_t)source_row * ((input_dimension * FormatA::kStoredBits) / 8u))
			+ (k_tile * row_pitch) + (chunk * LM_SWIZZLE_CHUNK_BYTES);
		if constexpr ( INTERLEAVED_B )
		{
			// THE TWO-BLOCK A STAGE (see LmPipelineProduce's interleaved
			// arm): chunks 0..7 are k 0..63 and land in block 0, chunks
			// 8..15 are k 64..127 in block 1, each block [rows x 128]
			// with the r % 8 per-row selector the consume derives from
			// pitch 128.
			const uint32_t a_block_bytes =
				LmTileBytes(a->rows,64u,a->element_bits);
			destination_offset = ((chunk / 8u) * a_block_bytes)
				+ LmSwizzledOffset(local_row,
					(chunk % 8u) * LM_SWIZZLE_CHUNK_BYTES,128u,128u);
		}
		else
		{
			destination_offset = FormatA::kTmaSwizzle
				? LmSwizzledOffset(local_row,chunk * LM_SWIZZLE_CHUNK_BYTES,row_pitch,LmSwizzleSpanFor(row_pitch))
				: (local_row * row_pitch) + (chunk * LM_SWIZZLE_CHUNK_BYTES);
		}
		LmTmaLoadBulk1d(
			(uint8_t *)stage_a + destination_offset,
			activation_bytes + source_offset,
			barrier,
			LM_SWIZZLE_CHUNK_BYTES);
	}
}

// ACTIVATION-CODEC STAGING. The A rows are staged BF16, quantised and
// dequantised into the codec's codes in shared by LmActivationStageFp8Qdq,
// optionally through the same route map the indirect path reads. The copies
// are plain shared stores, so the barrier declares and tracks ONLY the weight
// box and the caller owes a __syncthreads after the wait - the codec-free
// indirect path is LmPipelineProduceIndirectA above, whose bulk copies are
// complete_tx-accounted and need no such sync.
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
	static_assert(ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE,
		"the codec-free indirect path is LmPipelineProduceIndirectA");
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
