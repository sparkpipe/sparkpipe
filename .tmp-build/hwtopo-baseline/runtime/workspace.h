// Workspace layout for the first-party NVFP4 routed-MoE pipeline.
//
// Seven stages run between a normalised hidden state and a routed output:
// router top-k, packed-route build, activation quantise, w1 GEMM, SiLU-mul
// requantise, w2 GEMM, finalise. Each needs scratch, the sizes are all
// functions of (tokens, top_k, experts, hidden, intermediate), and getting one
// extent wrong overlaps two buffers - which corrupts data with no allocation
// error anywhere.
//
// The arithmetic lives here, separate from any CUDA, so it can be checked on a
// host. tests/test_group_gemm_workspace.c asserts every region is aligned, that
// regions do not overlap, and that the total is what the sum of the parts says.
//
// NVFP4 EXTENTS. Payloads are 4-bit, so element counts halve into bytes; scales
// are UE4M3, one byte per NVFP4_GROUP_SIZE elements. Both conversions are the
// kind that look right at a glance and are wrong by a factor of two.

#ifndef SPARK_LM_GROUP_GEMM_WORKSPACE_H
#define SPARK_LM_GROUP_GEMM_WORKSPACE_H

#include <stdint.h>

#define LM_WS_ALIGNMENT 256u
#define LM_WS_NVFP4_GROUP 16u
#define LM_WS_REGION_COUNT 9u
#define LM_WS_NVFP4_TILE_K 256u
#define SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL 4u
// 128 KB of L1/shared per SM on GB10, per docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md.
#define LM_WS_SHARED_LIMIT 131072u

#define LM_WS_REGION_PACKED_HIDDEN 0u
#define LM_WS_REGION_PACKED_HIDDEN_SCALE 1u
#define LM_WS_REGION_ROUTE_ROWS 2u
#define LM_WS_REGION_ROUTE_INDPTR 3u
#define LM_WS_REGION_GROUP_TILE_PREFIX 4u
#define LM_WS_REGION_GATE_UP_BF16 5u
#define LM_WS_REGION_INTERMEDIATE 6u
#define LM_WS_REGION_INTERMEDIATE_SCALE 7u
#define LM_WS_REGION_ROUTE_OUTPUT_BF16 8u

#define LM_WS_OK 0
#define LM_WS_ERR_NULL (-31)
#define LM_WS_ERR_SHAPE (-32)
#define LM_WS_ERR_GROUP (-33)
#define LM_WS_ERR_OVERFLOW (-34)
#define LM_WS_ERR_SHARED (-35)

// Regions that exist only because the pipeline is not fused. B12x materialises
// none of them - it declares FUSED_EXPERTS, IN_KERNEL_INPUT_QUANT and
// ZERO_DEVICE_MEMCPY, so its entire workspace is top_k route ids, top_k weights
// and one output buffer: about 96 KB, independent of batch. The unfused path
// needs 196 MB at B1024. That is a factor of two thousand, and it is the real
// cost of not fusing - not the 0.9 percent of bandwidth the round trips consume.
//
// The two largest are route_output at 96 MB and gate_up at 64 MB. Folding the
// finalize reduction into the second GEMM's epilogue removes the first; keeping
// the SiLU-mul in registers between the two GEMMs removes the second. Both are
// real work, and this constant is what justifies doing it.
#define LM_WS_UNFUSED_REGION_MASK ( \
	(1u << LM_WS_REGION_PACKED_HIDDEN) | \
	(1u << LM_WS_REGION_PACKED_HIDDEN_SCALE) | \
	(1u << LM_WS_REGION_GATE_UP_BF16) | \
	(1u << LM_WS_REGION_INTERMEDIATE) | \
	(1u << LM_WS_REGION_INTERMEDIATE_SCALE) | \
	(1u << LM_WS_REGION_ROUTE_OUTPUT_BF16))

typedef struct LmWorkspaceshape
{
	uint32_t tokens,top_k,expert_count,hidden_dimension,intermediate_dimension,tile_m,tile_n;
}
LmWorkspaceshape_t;

typedef struct LmWorkspacelayout
{
	uint64_t offset[LM_WS_REGION_COUNT];
	uint64_t bytes[LM_WS_REGION_COUNT];
	uint64_t total_bytes;
	uint64_t packed_rows;
	uint64_t total_tiles;
	uint64_t shared_bytes;
	uint32_t tile_m;
	uint32_t stages;
	uint32_t ctas_per_sm;
}
LmWorkspacelayout_t;

static uint64_t LmWorkspacealign_up(uint64_t value)
{
	return((value + (uint64_t)LM_WS_ALIGNMENT - 1u)
		& ~((uint64_t)LM_WS_ALIGNMENT - 1u));
}

// Rows in the packed layout: one per (token, route). Every downstream extent is
// a multiple of this, so it exists once.
static uint64_t LmWorkspacepacked_rows(const LmWorkspaceshape_t *shape)
{
	return((uint64_t)shape->tokens * (uint64_t)shape->top_k);
}

// Bytes this path needs at its maximum supported batch. Callers size a
// dedicated allocation from this rather than borrowing the B12x plan's
// workspace, which is sized for a fused kernel and is three orders of magnitude
// too small. Getting that wrong is not a corruption - the launcher compares and
// fails closed - but it is a hard stop at run time for a number that is known
// at plan time.
static uint64_t LmWorkspacebytes_for_max_batch(uint32_t max_tokens, uint32_t top_k, uint32_t expert_count, uint32_t hidden_dimension, uint32_t intermediate_dimension);

// Choose TILE_M for a token bucket.
//
// Rows per expert is tokens*top_k/experts, so it grows with the batch while
// TILE_M is a compile-time tile height. A fixed TILE_M=16 is exact through the
// point where rows reach 16 and then splits every expert into two M tiles - and
// because each M tile re-reads the expert's weight tile, that split doubles the
// weight stream, which is 96 percent of all traffic on this path. At GLM 5.2's
// 256 experts and top-8 that lands at B1024: rows/expert 32, two tiles, 2x the
// bytes. Selecting TILE_M per bucket is what keeps it at one.
//
// Rounding UP to the next tile height wastes MMA throughput on padded rows,
// which is free on a bandwidth-bound path. Rounding down would cost bandwidth,
// which is not. The asymmetry is why this only ever grows the tile.
static uint32_t LmWorkspaceselect_tile_m(uint64_t rows_per_expert)
{
	if ( rows_per_expert <= 16u )
		return(16u);
	if ( rows_per_expert <= 32u )
		return(32u);
	// 64 is the ceiling. B1024 is the supported maximum, which at 256 experts
	// and top-8 is 32 mean rows per expert and 64 at the 2x peak headroom, so a
	// 128-row tile is unreachable. Instantiating one would be dead code that
	// still costs compile time and a dispatch arm.
	return(64u);
}

// Shared memory one CTA needs. NVFP4 payloads are 4-bit, so both tile buffers
// halve; the barriers are two 8-byte mbarriers per stage.
static uint64_t LmWorkspaceshared_bytes(uint32_t tile_m, uint32_t tile_n, uint32_t tile_k, uint32_t stages, uint32_t element_bits)
{
	uint64_t per_stage;
	per_stage = ((uint64_t)tile_m + (uint64_t)tile_n) * (uint64_t)tile_k
		* (uint64_t)element_bits / 8u;
	return((uint64_t)stages * (per_stage + 16u));
}

// Deepest pipeline that fits, given the tile height already chosen.
//
// This is not a tuning knob dressed up as a constraint: at TILE_M=128 with the
// four stages that were hardcoded, an NVFP4 tile needs 131,136 bytes against a
// 131,072 limit and the launch fails outright. Selecting stages jointly with the
// tile height also turns a constraint into a gain at the small end - a 16-row
// tile leaves room for six stages where a 128-row tile leaves room for three,
// and the weight stream is what the extra depth covers.
//
// Depth beyond what Little's Law needs buys nothing. At 218 GB/s effective and
// an estimated 400-600 ns latency the requirement is about 2.7 KB in flight per
// SM, and even two stages of an 18 KB tile clear that by an order of magnitude.
// The cap therefore exists to stop shared memory being spent for no reason, not
// because more would help.
static uint32_t LmWorkspaceselect_stages(uint32_t tile_m, uint32_t tile_n, uint32_t tile_k, uint32_t element_bits, uint64_t shared_limit)
{
	// Two stages, which is a lookahead of one: one tile in flight while the
	// other is consumed. Deeper was tried and is not what the shared memory
	// should buy.
	//
	// Little's Law against 218 GB/s effective and 400-600 ns latency wants about
	// 2.3 KB in flight per SM. A single 24 KB tile per CTA already clears that
	// by twenty times, so stages three and beyond satisfy a requirement that was
	// met at two. What they cost is occupancy: six stages of a 16-row tile is
	// 110 KB and admits one CTA per SM, where two stages is 37 KB and admits
	// three. Depth past the requirement buys nothing; the CTAs might.
	//
	// This is a trade, not a free win - more CTAs do not create bandwidth on a
	// bandwidth-bound kernel, and the gain is only in tolerating tail effects
	// and uneven expert loads. It is the measurement the memory-latency
	// microbenchmark should settle.
	if ( LmWorkspaceshared_bytes(tile_m,tile_n,tile_k,2u,element_bits) <= shared_limit )
		return(2u);
	return(0u);
}

// CTAs that fit per SM at the selected geometry. Reported so a profile has
// something to compare against rather than being inferred after the fact.
static uint32_t LmWorkspacectas_per_sm(uint64_t shared_bytes, uint64_t shared_limit)
{
	if ( shared_bytes == 0u )
		return(0u);
	return((uint32_t)(shared_limit / shared_bytes));
}

// Rows the busiest expert is expected to hold. Routing is not uniform, so the
// mean understates the tile height a heavily loaded expert needs; the max-loaded
// group is what sets step time under a grouped launch. A 2x headroom factor is a
// heuristic and is the one number here that wants a measured route distribution
// behind it - the route-log collection already planned is what would supply it.
static uint64_t LmWorkspacepeak_rows_per_expert(const LmWorkspaceshape_t *shape)
{
	uint64_t mean_rows;
	mean_rows = (LmWorkspacepacked_rows(shape) + (uint64_t)shape->expert_count - 1u)
		/ (uint64_t)shape->expert_count;
	return(mean_rows * 2u);
}

// Grid tiles for the grouped GEMM: each expert contributes ceil(rows/TILE_M)
// M tiles times the N tile count. At decode most experts hold fewer rows than
// TILE_M, so this is dominated by the expert count, not the token count.
static uint64_t LmWorkspacetotal_tiles_for_tile_m(const LmWorkspaceshape_t *shape, uint32_t tile_m, uint32_t output_dimension)
{
	uint64_t rows_per_expert,m_tiles,n_tiles;
	rows_per_expert = (LmWorkspacepacked_rows(shape) + shape->expert_count - 1u)
		/ (uint64_t)shape->expert_count;
	m_tiles = (rows_per_expert + (uint64_t)tile_m - 1u) / (uint64_t)tile_m;
	if ( m_tiles == 0u )
		m_tiles = 1u;
	n_tiles = ((uint64_t)output_dimension + (uint64_t)shape->tile_n - 1u) / (uint64_t)shape->tile_n;
	return((uint64_t)shape->expert_count * m_tiles * n_tiles);
}

static int32_t LmWorkspacelayout_build(const LmWorkspaceshape_t *shape, LmWorkspacelayout_t *layout)
{
	uint64_t rows,hidden_bytes,hidden_scales,intermediate_bytes,intermediate_scales,cursor;
	uint32_t region,effective;
	if ( shape == 0 || layout == 0 )
		return(LM_WS_ERR_NULL);
	if ( shape->tokens == 0 || shape->top_k == 0 || shape->expert_count == 0
		|| shape->hidden_dimension == 0 || shape->intermediate_dimension == 0
		|| shape->tile_n == 0 )
		return(LM_WS_ERR_SHAPE);
	// A 4-bit payload needs an even element count, and a UE4M3 scale needs the
	// group to divide the row, or the last group is partial and the GEMM reads
	// a scale that was never written.
	if ( (shape->hidden_dimension % LM_WS_NVFP4_GROUP) != 0u
		|| (shape->intermediate_dimension % LM_WS_NVFP4_GROUP) != 0u )
		return(LM_WS_ERR_GROUP);
	// tile_m == 0 means "choose for this bucket", which is what the launcher
	// passes. An explicit value is honoured so a sweep can pin it.
	effective = shape->tile_m != 0u
		? shape->tile_m
		: LmWorkspaceselect_tile_m(LmWorkspacepeak_rows_per_expert(shape));
	layout->tile_m = effective;
	layout->stages = LmWorkspaceselect_stages(effective,shape->tile_n,
		LM_WS_NVFP4_TILE_K,SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL,
		LM_WS_SHARED_LIMIT);
	if ( layout->stages == 0u )
		return(LM_WS_ERR_SHARED);
	layout->shared_bytes = LmWorkspaceshared_bytes(effective,shape->tile_n,
		LM_WS_NVFP4_TILE_K,layout->stages,
		SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL);
	layout->ctas_per_sm = LmWorkspacectas_per_sm(layout->shared_bytes,
		LM_WS_SHARED_LIMIT);
	rows = LmWorkspacepacked_rows(shape);
	if ( rows == 0u || rows > 0xffffffffu )
		return(LM_WS_ERR_OVERFLOW);
	hidden_bytes = rows * ((uint64_t)shape->hidden_dimension / 2u);
	hidden_scales = rows * ((uint64_t)shape->hidden_dimension / LM_WS_NVFP4_GROUP);
	intermediate_bytes = rows * ((uint64_t)shape->intermediate_dimension / 2u);
	intermediate_scales = rows * ((uint64_t)shape->intermediate_dimension / LM_WS_NVFP4_GROUP);
	layout->bytes[LM_WS_REGION_PACKED_HIDDEN] = hidden_bytes;
	layout->bytes[LM_WS_REGION_PACKED_HIDDEN_SCALE] = hidden_scales;
	layout->bytes[LM_WS_REGION_ROUTE_ROWS] = rows * sizeof(uint32_t);
	layout->bytes[LM_WS_REGION_ROUTE_INDPTR] = ((uint64_t)shape->expert_count + 1u) * sizeof(uint32_t);
	layout->bytes[LM_WS_REGION_GROUP_TILE_PREFIX] = ((uint64_t)shape->expert_count + 1u) * sizeof(uint32_t);
	// w1 emits gate and up together, bf16, hence the factor of two on each.
	layout->bytes[LM_WS_REGION_GATE_UP_BF16] = rows * (uint64_t)shape->intermediate_dimension * 2u * 2u;
	layout->bytes[LM_WS_REGION_INTERMEDIATE] = intermediate_bytes;
	layout->bytes[LM_WS_REGION_INTERMEDIATE_SCALE] = intermediate_scales;
	layout->bytes[LM_WS_REGION_ROUTE_OUTPUT_BF16] = rows * (uint64_t)shape->hidden_dimension * 2u;
	cursor = 0u;
	for (region = 0u; region < LM_WS_REGION_COUNT; ++region)
	{
		layout->offset[region] = cursor;
		cursor = LmWorkspacealign_up(cursor + layout->bytes[region]);
	}
	layout->total_bytes = cursor;
	layout->packed_rows = rows;
	layout->total_tiles = LmWorkspacetotal_tiles_for_tile_m(shape,effective,
		shape->intermediate_dimension * 2u);
	return(LM_WS_OK);
}

static uint64_t LmWorkspacebytes_for_max_batch(uint32_t max_tokens, uint32_t top_k, uint32_t expert_count, uint32_t hidden_dimension, uint32_t intermediate_dimension)
{
	LmWorkspaceshape_t shape;
	LmWorkspacelayout_t layout;
	uint32_t index;
	for (index = 0u; index < sizeof(shape); ++index)
		((uint8_t *)&shape)[index] = 0u;
	shape.tokens = max_tokens;
	shape.top_k = top_k;
	shape.expert_count = expert_count;
	shape.hidden_dimension = hidden_dimension;
	shape.intermediate_dimension = intermediate_dimension;
	shape.tile_n = 128u;
	if ( LmWorkspacelayout_build(&shape,&layout) != LM_WS_OK )
		return(0u);
	return(layout.total_bytes);
}

#endif
