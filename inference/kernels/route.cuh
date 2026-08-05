#pragma once

// The route build: from the router's per-token expert choices to the packed,
// expert-major order the grouped GEMM streams.
//
// Nothing in the tree produced these arrays - every harness filled them by
// hand, which is how a driver came to be impossible: the top-k output lives on
// the device, and a host cannot pack what it cannot see without a sync on the
// hot path. This is the producer, and it is one kernel because the whole job
// is a counting sort over 896 buckets: count, prefix, scatter, with the
// counters in shared memory and the prefix serial - 896 additions is not a
// problem that needs a scan.
//
// THE ORDER WITHIN AN EXPERT IS WHATEVER THE ATOMICS SAY. The GEMM does not
// care - every packed row carries its source token - and demanding a stable
// order would buy determinism the finalize never reads.

// ROUTE ROW INDIRECTION CONSUMER CONTRACT
//
// The grouped weight-only GEMM reads activation rows through
// route_source_token directly. Expert weights retain their TMA pipeline while
// the CTA cooperatively stages each indexed BF16 activation row into the same
// swizzled shared tile consumed by MMA. No packed activation tensor is written,
// and no host sees the route.
//
// The producer side of the indirect form is ALREADY COMPLETE - this kernel
// writes everything an indirect consumer needs, which is why the contract
// can be stated exactly. The indirect consumer uses the arrays as follows:
//
//   - Packed row p of expert group g spans
//     p in [group_row_offset[g], group_row_offset[g + 1]), contiguous and
//     expert-major. Output rows stay packed - only the A READ is indirect.
//   - The A row for packed row p is route_source_token[p] of the
//     UN-gathered activation tensor. The A tensor map describes that source
//     tensor (rows x hidden), never a packed copy.
//   - route_source_token[p] < rows for every p < rows * top_k, by
//     construction (it is index / top_k over the route array).
//   - Within one group the source rows are DISTINCT if and only if the
//     router emits distinct experts per token; the top-k contract
//     guarantees that, and this kernel neither checks nor needs it.
//   - Order within a group is atomic-arrival order. An indirect consumer
//     must not assume sorted, unique-across-groups, or stable order.
//   - RAGGED TAIL. A group's last tile covers
//     [row_base, row_base + TILE_M) while valid indices end at row_limit =
//     group_row_offset[g + 1). The consumer reads indices only for
//     p < row_limit and CLAMPS the tail to row_base; the stores for those rows
//     are already dropped by the GEMM's row_limit check, so a duplicate load
//     is dead traffic, never wrong output. Reading an index past
//     row_limit instead of clamping is a wild TMA gather - the array's next
//     bytes are another group's indices, in-range but wrong, and nothing
//     faults.
//   - SCALE ROWS FOLLOW THE SOURCE. An indirect A-read changes which
//     activation row a fragment came from, so scale_a must be indexed by
//     route_source_token[p], not by p. The BF16-activation MoE paths carry
//     LmScaleTensorNone here and are unaffected; a quantized-activation
//     consumer that forgets this applies another token's scale and nothing
//     faults.
//   - LIFETIME. The next step's route build rewrites these arrays on the
//     same stream, so any consumer on that stream is ordered for free. A
//     consumer on another stream, or a CUDA graph that captured the build
//     and the GEMM together, needs the buffers to stay at the addresses the
//     graph recorded - the stage's replay contract guarantees that.
//
// The mapping itself is LmRouteSourceRow, defined just below the includes.


#include "inference/kernels/mma.cuh"
#include "runtime/launch.h"
#include <stdint.h>

// The indirection mapping from the contract above: packed row to source
// activation row. A function rather than an open-coded index so the consumer
// and this producer cannot drift on what the array means.
static __device__ __forceinline__ uint32_t LmRouteSourceRow(const uint32_t *__restrict__ route_source_token, uint32_t packed_row)
{
	return(route_source_token[packed_row]);
}

// route_expert:      [routes]  the router's choice per (token, k)
// group_row_offset:  [experts + 1]  exclusive prefix of per-expert row counts
// route_packed_row:  [routes]  where (token, k) landed in the packed order
// route_source_token:[routes]  which token a packed row came from
// The two tile tables ride along: the layer knows the tile heights its two
// expert GEMMs will select (the planner's choice is a pure function of the
// worst-case rows, host-computable), so the same thread that writes the row
// offsets prices both tables and the launcher makes no prefix launch at all.
// Null tables skip the pricing, which is any caller that is not a MoE layer.
template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
void LmRouteBuildKernel(const uint32_t *__restrict__ route_expert, uint32_t routes, uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t count[EXPERTS];
	uint32_t index,expert,packed;
	for (index = threadIdx.x; index < EXPERTS; index += THREADS)
		count[index] = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < routes; index += THREADS)
		atomicAdd(&count[route_expert[index]],1u);
	__syncthreads();
	// Serial exclusive prefix, one thread. After this, count[] holds each
	// expert's running cursor - the same array serves both jobs.
	if ( threadIdx.x == 0u )
	{
		uint32_t total = 0u,held;
		for (index = 0u; index < EXPERTS; ++index)
		{
			held = count[index];
			group_row_offset[index] = total;
			count[index] = total;
			total += held;
		}
		group_row_offset[EXPERTS] = total;
		if ( tile_prefix_up != 0 && tile_prefix_down != 0 )
		{
			uint32_t up = 0u,down = 0u,rows,row_tiles;
			for (index = 0u; index < EXPERTS; ++index)
			{
				rows = group_row_offset[index + 1u] - group_row_offset[index];
				row_tiles = (rows + tile_m - 1u) / tile_m;
				tile_prefix_up[index] = up;
				tile_prefix_down[index] = down;
				up += row_tiles * neuron_tiles_up;
				down += row_tiles * neuron_tiles_down;
			}
			tile_prefix_up[EXPERTS] = up;
			tile_prefix_down[EXPERTS] = down;
		}
	}
	__syncthreads();
	for (index = threadIdx.x; index < routes; index += THREADS)
	{
		expert = route_expert[index];
		packed = atomicAdd(&count[expert],1u);
		route_packed_row[index] = packed;
		route_source_token[packed] = index / top_k;
	}
}


// Build the expert-major route once for a complete logical batch. The caller
// passes both the token count and packed-row count so a stale replay/chunk count
// cannot silently price the grouped GEMM from routes instead of tokens.
template<uint32_t THREADS, uint32_t EXPERTS>
static int32_t LmRouteBuild(
	const uint32_t *route_expert,
	uint32_t rows,
	uint32_t packed_rows,
	uint32_t top_k,
	uint32_t *group_row_offset,
	uint32_t *route_packed_row,
	uint32_t *route_source_token,
	uint32_t output_dimension_up,
	uint32_t output_dimension_down,
	uint32_t tile_n,
	uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down,
	cudaStream_t stream)
{
	uint32_t expected_packed_rows;
	uint32_t tile_m;
	uint32_t neuron_tiles_up;
	uint32_t neuron_tiles_down;

	if ( route_expert == 0 || rows == 0u || top_k == 0u ||
		rows > UINT32_MAX / top_k || group_row_offset == 0 ||
		route_packed_row == 0 || route_source_token == 0 ||
		output_dimension_up == 0u || output_dimension_down == 0u ||
		tile_n == 0u || tile_prefix_up == 0 || tile_prefix_down == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	expected_packed_rows = rows * top_k;
	if ( packed_rows != expected_packed_rows )
		return(LM_LAUNCH_ERR_SHAPE);
	tile_m = LmLaunchGroupedTileM(rows,top_k,EXPERTS);
	neuron_tiles_up = (output_dimension_up + tile_n - 1u) / tile_n;
	neuron_tiles_down = (output_dimension_down + tile_n - 1u) / tile_n;
	LM_LAUNCH((LmRouteBuildKernel<THREADS,EXPERTS>), 1u, THREADS, 0, stream,
		route_expert,packed_rows,top_k,group_row_offset,route_packed_row,
		route_source_token,tile_m,neuron_tiles_up,tile_prefix_up,
		neuron_tiles_down,tile_prefix_down);
	return(cudaPeekAtLastError() == cudaSuccess
		? LM_LAUNCH_OK
		: LM_LAUNCH_ERR_LAUNCH);
}
