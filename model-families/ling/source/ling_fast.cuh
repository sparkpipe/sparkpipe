#pragma once

#include <atomic>

static __device__ __forceinline__ uint32_t LingSwizzle10(uint32_t row, uint32_t chunk)
{
	return chunk < 8u ? LmStreamSwizzle8(row, chunk) : (8u + ((chunk - 8u) ^ (row & 1u)));
}

static __global__ __launch_bounds__(32u, 1)
void LingBuildTilePrefixKernel(const uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ prefix,
	uint32_t groups, uint32_t tile_m, uint32_t n_tiles)
{
	const uint32_t lane = threadIdx.x, per = (groups + 31u) / 32u, first = lane * per;
	uint32_t own = 0u;
	LmDependentWait();
	for ( uint32_t g = first; g < first + per && g < groups; ++g )
	{
		const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
		own += ((r + tile_m - 1u) / tile_m) * n_tiles;
	}
	uint32_t run = own;
	#pragma unroll
	for ( uint32_t s = 1u; s < 32u; s <<= 1u )
	{
		const uint32_t other = __shfl_up_sync(0xffffffffu, run, s);
		if ( lane >= s )
			run += other;
	}
	uint32_t off = run - own;
	for ( uint32_t g = first; g < first + per && g < groups; ++g )
	{
		prefix[g] = off;
		const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
		off += ((r + tile_m - 1u) / tile_m) * n_tiles;
	}
	if ( lane == 31u )
		prefix[groups] = run;
	LmDependentRelease();
}

static __device__ __forceinline__ void LingGroupedTileLookup(const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ prefix, uint32_t groups, uint32_t tile, uint32_t tile_m, uint32_t n_tiles,
	uint32_t &grp, uint32_t &row_base, uint32_t &row_limit, uint32_t &nt)
{
	uint32_t lo = 0u, hi = groups;
	while ( lo + 1u < hi )
	{
		const uint32_t mid = (lo + hi) >> 1u;
		if ( __ldg(prefix + mid) <= tile )
			lo = mid;
		else
			hi = mid;
	}
	grp = lo;
	const uint32_t first = __ldg(group_row_offset + lo);
	row_limit = __ldg(group_row_offset + lo + 1u);
	const uint32_t within = tile - __ldg(prefix + lo);
	row_base = first + (within / n_tiles) * tile_m;
	nt = within % n_tiles;
}

template<uint32_t EXPERTS, uint32_t GROUPS, uint32_t TOPK_GROUPS, uint32_t TOPK>
static __device__ __forceinline__ void LingGroupTopkWarpFast(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t row,
	uint32_t expert_first, uint32_t local_experts, float scale, uint32_t *__restrict__ fault,
	uint32_t *__restrict__ out_exp, float *__restrict__ out_wt, uint32_t lane)
{
	constexpr uint32_t PER_LANE = (EXPERTS + 31u) / 32u;
	constexpr uint32_t PER_GROUP = EXPERTS / GROUPS;
	float sc[PER_LANE];
	float sel_v[GROUPS > 1u ? PER_LANE : 1u];
	unsigned long long key[PER_LANE];
	const uint64_t row_base = (uint64_t)row * EXPERTS;
	#pragma unroll
	for ( uint32_t j = 0u; j < PER_LANE; ++j )
	{
		const uint32_t e = lane + j * 32u;
		if ( e < EXPERTS )
		{
			const float s = SparkMathSigmoid(__ldg(logits + row_base + e));
			const float sel = s + (bias != 0 ? __ldg(bias + e) : 0.0f);
			sc[j] = s;
			if constexpr ( GROUPS > 1u )
				sel_v[j] = sel;
			key[j] = ((unsigned long long)LmTopkRouteKey(sel + 0.0f) << 32u) | (0xffffffffu - e);
		}
		else
		{
			sc[j] = 0.0f;
			if constexpr ( GROUPS > 1u )
				sel_v[j] = -INFINITY;
			key[j] = 0ull;
		}
	}
	if constexpr ( GROUPS > 1u )
	{
		float gscore[GROUPS];
		if constexpr ( (PER_GROUP & 31u) == 0u )
		{
			constexpr uint32_t CPG = PER_GROUP / 32u;
			#pragma unroll
			for ( uint32_t g = 0u; g < GROUPS; ++g )
			{
				float m1 = -INFINITY, m2 = -INFINITY;
				#pragma unroll
				for ( uint32_t c = 0u; c < CPG; ++c )
				{
					const float v = sel_v[g * CPG + c];
					if ( v > m1 ) { m2 = m1; m1 = v; }
					else if ( v > m2 ) { m2 = v; }
				}
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					const float om1 = __shfl_down_sync(0xffffffffu, m1, offset);
					const float om2 = __shfl_down_sync(0xffffffffu, m2, offset);
					if ( om1 > m1 ) { m2 = fmaxf(m1, om2); m1 = om1; }
					else { m2 = fmaxf(m2, om1); }
				}
				gscore[g] = m1 + m2;
			}
		}
		else
		{
			#pragma unroll
			for ( uint32_t g = 0u; g < GROUPS; ++g )
			{
				float m1 = -INFINITY, m2 = -INFINITY;
				#pragma unroll
				for ( uint32_t j = 0u; j < PER_LANE; ++j )
				{
					const uint32_t e = lane + j * 32u;
					if ( e < EXPERTS && (e / PER_GROUP) == g )
					{
						const float v = sel_v[j];
						if ( v > m1 ) { m2 = m1; m1 = v; }
						else if ( v > m2 ) { m2 = v; }
					}
				}
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					const float om1 = __shfl_down_sync(0xffffffffu, m1, offset);
					const float om2 = __shfl_down_sync(0xffffffffu, m2, offset);
					if ( om1 > m1 ) { m2 = fmaxf(m1, om2); m1 = om1; }
					else { m2 = fmaxf(m2, om1); }
				}
				gscore[g] = m1 + m2;
			}
		}
		uint32_t open_mask = 0u;
		if ( lane == 0u )
		{
			#pragma unroll
			for ( uint32_t pick = 0u; pick < TOPK_GROUPS; ++pick )
			{
				uint32_t best = GROUPS;
				float best_v = -INFINITY;
				#pragma unroll
				for ( uint32_t g = 0u; g < GROUPS; ++g )
				{
					if ( ((open_mask >> g) & 1u) == 0u && (best == GROUPS || gscore[g] > best_v) )
					{
						best = g;
						best_v = gscore[g];
					}
				}
				open_mask |= 1u << best;
			}
		}
		open_mask = __shfl_sync(0xffffffffu, open_mask, 0u);
		if constexpr ( (PER_GROUP & 31u) == 0u )
		{
			constexpr uint32_t CPG = PER_GROUP / 32u;
			#pragma unroll
			for ( uint32_t j = 0u; j < PER_LANE; ++j )
				if ( ((open_mask >> (j / CPG)) & 1u) == 0u )
					key[j] = 0ull;
		}
		else
		{
			#pragma unroll
			for ( uint32_t j = 0u; j < PER_LANE; ++j )
			{
				const uint32_t e = lane + j * 32u;
				if ( e < EXPERTS && ((open_mask >> (e / PER_GROUP)) & 1u) == 0u )
					key[j] = 0ull;
			}
		}
	}
	uint32_t picked_e[TOPK];
	float picked_w[TOPK];
	float total = 0.0f;
	#pragma unroll
	for ( uint32_t pick = 0u; pick < TOPK; ++pick )
	{
		unsigned long long best = 0ull;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			best = key[j] > best ? key[j] : best;
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		{
			const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
			best = other > best ? other : best;
		}
		const uint32_t e = 0xffffffffu - (uint32_t)best;
		const uint32_t owner_lane = e & 31u;
		float picked_s = 0.0f;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			if ( lane + j * 32u == e )
			{
				picked_s = sc[j];
				key[j] = 0ull;
			}
		picked_s = __shfl_sync(0xffffffffu, picked_s, (int)owner_lane);
		total += picked_s;
		if ( lane == 0u )
		{
			picked_e[pick] = e;
			picked_w[pick] = picked_s;
			LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
		}
	}
	if ( lane == 0u )
	{
		const float denom = total + 1e-20f;
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
		{
			const uint32_t ge = picked_e[pick];
			if ( ge >= expert_first && ge < expert_first + local_experts )
			{
				out_exp[pick] = ge - expert_first;
				out_wt[pick] = picked_w[pick] / denom * scale;
			}
			else
			{
				out_exp[pick] = local_experts;
				out_wt[pick] = 0.0f;
			}
		}
	}
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 1)
static void LingGroupRouteRow1FastKernel(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ packed_expert,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	float *__restrict__ packed_weight, uint32_t *__restrict__ active_experts)
{
	constexpr uint32_t EXPERTS = G::kExperts, GROUPS = G::kGroups, TOPK_GROUPS = G::kTopkGroups, TOPK = G::kTopK;
	constexpr uint32_t LOCAL_E = G::kLocalExperts, SLOTS = LOCAL_E + 1u;
	constexpr uint32_t WARPS = LG_THREADS / 32u;
	__shared__ float s_gscore[GROUPS > 1u ? GROUPS : 1u];
	__shared__ unsigned long long s_cand_k[WARPS * TOPK];
	__shared__ float s_cand_s[WARPS * TOPK];
	__shared__ uint32_t s_exp[TOPK];
	__shared__ float s_wt[TOPK];
	const uint32_t tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
	LmDependentWait();
	if constexpr ( GROUPS == WARPS && (EXPERTS / GROUPS) == 64u && TOPK_GROUPS * TOPK <= 32u )
	{
		float sc[2], sel_v[2];
		unsigned long long key[2];
		#pragma unroll
		for ( uint32_t c = 0u; c < 2u; ++c )
		{
			const uint32_t e = warp * 64u + c * 32u + lane;
			const float s = SparkMathSigmoid(__ldg(logits + e));
			const float sel = s + (bias != 0 ? __ldg(bias + e) : 0.0f);
			sc[c] = s;
			sel_v[c] = sel;
			key[c] = ((unsigned long long)LmTopkRouteKey(sel + 0.0f) << 32u) | (0xffffffffu - e);
		}
		float m1 = sel_v[0] > sel_v[1] ? sel_v[0] : sel_v[1];
		float m2 = sel_v[0] > sel_v[1] ? sel_v[1] : sel_v[0];
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		{
			const float om1 = __shfl_down_sync(0xffffffffu, m1, offset);
			const float om2 = __shfl_down_sync(0xffffffffu, m2, offset);
			if ( om1 > m1 ) { m2 = fmaxf(m1, om2); m1 = om1; }
			else { m2 = fmaxf(m2, om1); }
		}
		if ( lane == 0u )
			s_gscore[warp] = m1 + m2;
		__syncthreads();
		uint32_t open_mask = 0u;
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK_GROUPS; ++pick )
		{
			uint32_t best = GROUPS;
			float best_v = -INFINITY;
			#pragma unroll
			for ( uint32_t g = 0u; g < GROUPS; ++g )
			{
				if ( ((open_mask >> g) & 1u) == 0u && (best == GROUPS || s_gscore[g] > best_v) )
				{
					best = g;
					best_v = s_gscore[g];
				}
			}
			open_mask |= 1u << best;
		}
		if ( ((open_mask >> warp) & 1u) != 0u )
		{
			const uint32_t slot = (uint32_t)__popc(open_mask & ((1u << warp) - 1u));
			#pragma unroll
			for ( uint32_t pick = 0u; pick < TOPK; ++pick )
			{
				unsigned long long best = key[0] > key[1] ? key[0] : key[1];
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
					best = other > best ? other : best;
				}
				const uint32_t e = 0xffffffffu - (uint32_t)best;
				float picked_s = 0.0f;
				#pragma unroll
				for ( uint32_t c = 0u; c < 2u; ++c )
					if ( warp * 64u + c * 32u + lane == e )
					{
						picked_s = sc[c];
						key[c] = 0ull;
					}
				picked_s = __shfl_sync(0xffffffffu, picked_s, (int)(e & 31u));
				if ( lane == 0u )
				{
					s_cand_k[slot * TOPK + pick] = best;
					s_cand_s[slot * TOPK + pick] = picked_s;
				}
			}
		}
		__syncthreads();
		if ( warp == 0u )
		{
			constexpr uint32_t CANDS = TOPK_GROUPS * TOPK;
			unsigned long long ck = lane < CANDS ? s_cand_k[lane] : 0ull;
			float cs = lane < CANDS ? s_cand_s[lane] : 0.0f;
			uint32_t pe[TOPK];
			float pw[TOPK], total = 0.0f;
			#pragma unroll
			for ( uint32_t pick = 0u; pick < TOPK; ++pick )
			{
				unsigned long long best = ck;
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
					best = other > best ? other : best;
				}
				const uint32_t hit = __ballot_sync(0xffffffffu, ck == best && ck != 0ull);
				const uint32_t owner = (uint32_t)(__ffs(hit) - 1);
				const float ps = __shfl_sync(0xffffffffu, cs, (int)owner);
				if ( lane == owner )
					ck = 0ull;
				total += ps;
				if ( lane == 0u )
				{
					pe[pick] = 0xffffffffu - (uint32_t)best;
					pw[pick] = ps;
					LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
				}
			}
			if ( lane == 0u )
			{
				const float denom = total + 1e-20f;
				#pragma unroll
				for ( uint32_t pick = 0u; pick < TOPK; ++pick )
				{
					const uint32_t ge = pe[pick];
					if ( ge >= expert_first && ge < expert_first + LOCAL_E )
					{
						s_exp[pick] = ge - expert_first;
						s_wt[pick] = pw[pick] / denom * G::kRouteScale;
					}
					else
					{
						s_exp[pick] = LOCAL_E;
						s_wt[pick] = 0.0f;
					}
				}
			}
		}
	}
	else if constexpr ( GROUPS == 1u && (EXPERTS % 64u) == 0u && (EXPERTS / 64u) <= WARPS )
	{
		constexpr uint32_t ACTIVE_W = EXPERTS / 64u;
		if ( warp < ACTIVE_W )
		{
			float sc[2];
			unsigned long long key[2];
			#pragma unroll
			for ( uint32_t c = 0u; c < 2u; ++c )
			{
				const uint32_t e = warp * 64u + c * 32u + lane;
				const float s = SparkMathSigmoid(__ldg(logits + e));
				const float sel = s + (bias != 0 ? __ldg(bias + e) : 0.0f);
				sc[c] = s;
				key[c] = ((unsigned long long)LmTopkRouteKey(sel + 0.0f) << 32u) | (0xffffffffu - e);
			}
			#pragma unroll
			for ( uint32_t pick = 0u; pick < TOPK; ++pick )
			{
				unsigned long long best = key[0] > key[1] ? key[0] : key[1];
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
					best = other > best ? other : best;
				}
				const uint32_t e = 0xffffffffu - (uint32_t)best;
				float picked_s = 0.0f;
				#pragma unroll
				for ( uint32_t c = 0u; c < 2u; ++c )
					if ( warp * 64u + c * 32u + lane == e )
					{
						picked_s = sc[c];
						key[c] = 0ull;
					}
				picked_s = __shfl_sync(0xffffffffu, picked_s, (int)(e & 31u));
				if ( lane == 0u )
				{
					s_cand_k[warp * TOPK + pick] = best;
					s_cand_s[warp * TOPK + pick] = picked_s;
				}
			}
		}
		__syncthreads();
		if ( warp == 0u )
		{
			constexpr uint32_t CANDS = ACTIVE_W * TOPK;
			constexpr uint32_t C_PER = (CANDS + 31u) / 32u;
			unsigned long long ck[C_PER];
			float cs[C_PER];
			#pragma unroll
			for ( uint32_t j = 0u; j < C_PER; ++j )
			{
				const uint32_t idx = lane + j * 32u;
				ck[j] = idx < CANDS ? s_cand_k[idx] : 0ull;
				cs[j] = idx < CANDS ? s_cand_s[idx] : 0.0f;
			}
			uint32_t pe[TOPK];
			float pw[TOPK], total = 0.0f;
			#pragma unroll
			for ( uint32_t pick = 0u; pick < TOPK; ++pick )
			{
				unsigned long long best = 0ull;
				#pragma unroll
				for ( uint32_t j = 0u; j < C_PER; ++j )
					best = ck[j] > best ? ck[j] : best;
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
					best = other > best ? other : best;
				}
				float ps = 0.0f;
				uint32_t hit = 0u;
				#pragma unroll
				for ( uint32_t j = 0u; j < C_PER; ++j )
					if ( ck[j] == best && ck[j] != 0ull )
					{
						ps = cs[j];
						ck[j] = 0ull;
						hit = 1u;
					}
				const uint32_t owner = (uint32_t)(__ffs(__ballot_sync(0xffffffffu, hit != 0u)) - 1);
				ps = __shfl_sync(0xffffffffu, ps, (int)owner);
				total += ps;
				if ( lane == 0u )
				{
					pe[pick] = 0xffffffffu - (uint32_t)best;
					pw[pick] = ps;
					LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
				}
			}
			if ( lane == 0u )
			{
				const float denom = total + 1e-20f;
				#pragma unroll
				for ( uint32_t pick = 0u; pick < TOPK; ++pick )
				{
					const uint32_t ge = pe[pick];
					if ( ge >= expert_first && ge < expert_first + LOCAL_E )
					{
						s_exp[pick] = ge - expert_first;
						s_wt[pick] = pw[pick] / denom * G::kRouteScale;
					}
					else
					{
						s_exp[pick] = LOCAL_E;
						s_wt[pick] = 0.0f;
					}
				}
			}
		}
	}
	else
	{
		if ( warp == 0u )
			LingGroupTopkWarpFast<EXPERTS, GROUPS, TOPK_GROUPS, TOPK>(logits, bias, 0u, expert_first, LOCAL_E, G::kRouteScale,
				fault, s_exp, s_wt, lane);
	}
	__syncthreads();
	for ( uint32_t e = tid; e <= SLOTS; e += LG_THREADS )
	{
		uint32_t before = 0u;
		#pragma unroll
		for ( uint32_t j = 0u; j < TOPK; ++j )
			if ( s_exp[j] < e )
				before++;
		group_row_offset[e] = before;
	}
	if ( tid < TOPK )
	{
		const uint32_t mine = s_exp[tid];
		uint32_t packed = 0u;
		#pragma unroll
		for ( uint32_t j = 0u; j < TOPK; ++j )
			if ( s_exp[j] < mine || (s_exp[j] == mine && j < tid) )
				packed++;
		route_expert[tid] = mine;
		route_weight[tid] = s_wt[tid];
		route_packed_row[tid] = packed;
		route_source_token[packed] = 0u;
		packed_weight[packed] = s_wt[tid];
		packed_expert[packed] = mine;
		if ( mine < LOCAL_E )
			active_experts[1u + packed] = mine;
	}
	if ( tid == 0u )
	{
		uint32_t act = 0u;
		#pragma unroll
		for ( uint32_t j = 0u; j < TOPK; ++j )
			if ( s_exp[j] < LOCAL_E )
				act++;
		active_experts[0u] = act;
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 1)
static void LingGroupRouteSmallFastKernel(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ packed_expert,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	float *__restrict__ packed_weight, uint32_t *__restrict__ active_experts)
{
	constexpr uint32_t LOCAL_E = G::kLocalExperts;
	constexpr uint32_t SLOTS = LOCAL_E + 1u;
	__shared__ uint32_t s_exp[32];
	__shared__ float s_wt[32];
	__shared__ uint32_t s_cnt[SLOTS];
	__shared__ uint32_t s_off[SLOTS + 1u];
	const uint32_t tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, routes = rows * G::kTopK;
	LmDependentWait();
	if ( warp < rows )
		LingGroupTopkWarpFast<G::kExperts, G::kGroups, G::kTopkGroups, G::kTopK>(logits, bias, warp, expert_first, LOCAL_E, G::kRouteScale,
			fault, s_exp + warp * G::kTopK, s_wt + warp * G::kTopK, lane);
	for ( uint32_t e = tid; e < SLOTS; e += LG_THREADS )
		s_cnt[e] = 0u;
	__syncthreads();
	if ( tid < routes )
	{
		const uint32_t e = s_exp[tid];
		route_expert[tid] = e;
		route_weight[tid] = s_wt[tid];
		atomicAdd(&s_cnt[e], 1u);
	}
	__syncthreads();
	if ( tid == 0u )
	{
		uint32_t acc = 0u, act = 0u;
		for ( uint32_t e = 0u; e < SLOTS; ++e )
		{
			s_off[e] = acc;
			const uint32_t c = s_cnt[e];
			if ( c != 0u && e < LOCAL_E )
				active_experts[1u + act++] = e;
			acc += c;
		}
		s_off[SLOTS] = acc;
		active_experts[0u] = act;
	}
	__syncthreads();
	for ( uint32_t e = tid; e <= SLOTS; e += LG_THREADS )
		group_row_offset[e] = s_off[e];
	if ( tid < routes )
	{
		const uint32_t mine = s_exp[tid];
		uint32_t rank = 0u;
		for ( uint32_t j = 0u; j < tid; ++j )
			if ( s_exp[j] == mine )
				rank++;
		const uint32_t packed = s_off[mine] + rank;
		route_packed_row[tid] = packed;
		route_source_token[packed] = tid / G::kTopK;
		packed_weight[packed] = s_wt[tid];
		packed_expert[packed] = mine;
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingGroupTopkFastKernel(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	constexpr uint32_t WARPS = LG_THREADS / LM_WARP_LANES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.x * WARPS + warp;
	LmDependentWait();
	if ( row < rows )
		LingGroupTopkWarpFast<G::kExperts, G::kGroups, G::kTopkGroups, G::kTopK>(logits, bias, row, expert_first, G::kLocalExperts, G::kRouteScale,
			fault, route_expert + (uint64_t)row * G::kTopK, route_weight + (uint64_t)row * G::kTopK, lane);
	LmDependentRelease();
}

template<uint32_t LOCAL_E>
__global__ __launch_bounds__(LG_THREADS, 1)
static void LingRouteBuildFastKernel(const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint32_t routes,
	uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	float *__restrict__ packed_weight, uint32_t *__restrict__ packed_expert, uint32_t *__restrict__ active_experts)
{
	constexpr uint32_t SLOTS = LOCAL_E + 1u;
	__shared__ uint16_t w_hist[8][SLOTS];
	__shared__ uint32_t warp_tot[8], act_tot[8], chunk_base, act_base;
	const uint32_t tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
	LmDependentWait();
	for ( uint32_t i = tid; i < 8u * SLOTS; i += LG_THREADS )
		((uint16_t *)w_hist)[i] = 0u;
	if ( tid == 0u )
	{
		chunk_base = 0u;
		act_base = 0u;
	}
	__syncthreads();
	const uint32_t slice = (routes + 7u) >> 3u;
	const uint32_t w_beg = warp * slice;
	const uint32_t w_end = w_beg + slice < routes ? w_beg + slice : routes;
	for ( uint32_t p = w_beg; p < w_end; p += 32u )
	{
		const uint32_t i = p + lane;
		const bool active = i < w_end;
		const uint32_t e = active ? __ldg(route_expert + i) : 0xffffffffu;
		const uint32_t amask = __ballot_sync(0xffffffffu, active);
		if ( active )
		{
			const uint32_t m = __match_any_sync(amask, e);
			if ( lane == (uint32_t)(__ffs(m) - 1) )
				w_hist[warp][e] = (uint16_t)(w_hist[warp][e] + (uint16_t)__popc(m));
		}
	}
	__syncthreads();
	for ( uint32_t base_e = 0u; base_e < SLOTS; base_e += LG_THREADS )
	{
		const uint32_t e = base_e + tid;
		const bool valid = e < SLOTS;
		uint32_t c = 0u;
		uint16_t w_off[8];
		if ( valid )
		{
			#pragma unroll
			for ( uint32_t w = 0u; w < 8u; ++w )
			{
				w_off[w] = (uint16_t)c;
				c += w_hist[w][e];
			}
		}
		uint32_t scan = c;
		#pragma unroll
		for ( uint32_t step = 1u; step < 32u; step <<= 1u )
		{
			const uint32_t up = __shfl_up_sync(0xffffffffu, scan, step);
			if ( lane >= step )
				scan += up;
		}
		const uint32_t is_act = (e < LOCAL_E && c > 0u) ? 1u : 0u;
		const uint32_t act_mask = __ballot_sync(0xffffffffu, is_act != 0u);
		const uint32_t act_rank = (uint32_t)__popc(act_mask & ((1u << lane) - 1u));
		if ( lane == 31u )
		{
			warp_tot[warp] = scan;
			act_tot[warp] = (uint32_t)__popc(act_mask);
		}
		__syncthreads();
		uint32_t w_pre = 0u, a_pre = 0u, c_add = 0u, a_add = 0u;
		#pragma unroll
		for ( uint32_t w = 0u; w < 8u; ++w )
		{
			if ( w < warp ) { w_pre += warp_tot[w]; a_pre += act_tot[w]; }
			c_add += warp_tot[w];
			a_add += act_tot[w];
		}
		const uint32_t ex_off = chunk_base + w_pre + (scan - c);
		if ( valid )
		{
			group_row_offset[e] = ex_off;
			#pragma unroll
			for ( uint32_t w = 0u; w < 8u; ++w )
				w_hist[w][e] = (uint16_t)(ex_off + w_off[w]);
			if ( is_act != 0u )
				active_experts[1u + act_base + a_pre + act_rank] = e;
		}
		__syncthreads();
		if ( tid == 0u )
		{
			chunk_base += c_add;
			act_base += a_add;
		}
		__syncthreads();
	}
	if ( tid == 0u )
	{
		group_row_offset[SLOTS] = chunk_base;
		active_experts[0u] = act_base;
	}
	__syncthreads();
	for ( uint32_t p = w_beg; p < w_end; p += 32u )
	{
		const uint32_t i = p + lane;
		const bool active = i < w_end;
		const uint32_t e = active ? __ldg(route_expert + i) : 0xffffffffu;
		const uint32_t amask = __ballot_sync(0xffffffffu, active);
		if ( active )
		{
			const uint32_t m = __match_any_sync(amask, e);
			const uint32_t rank = (uint32_t)__popc(m & ((1u << lane) - 1u));
			const uint32_t packed = (uint32_t)w_hist[warp][e] + rank;
			if ( lane == (uint32_t)(__ffs(m) - 1) )
				w_hist[warp][e] = (uint16_t)(w_hist[warp][e] + (uint16_t)__popc(m));
			route_packed_row[i] = packed;
			route_source_token[packed] = i / top_k;
			packed_weight[packed] = __ldg(route_weight + i);
			packed_expert[packed] = e;
		}
	}
	LmDependentRelease();
}

template<uint32_t RANK, uint32_t OUT_DIM>
__global__ __launch_bounds__(64u, 4)
static void LingKdaLowRankPairB1FastKernel(const uint16_t *__restrict__ w_f, const uint16_t *__restrict__ in_f, uint16_t *__restrict__ out_f,
	const uint16_t *__restrict__ w_g, const uint16_t *__restrict__ in_g, uint16_t *__restrict__ out_g)
{
	constexpr uint32_t CHUNKS = RANK >> 3u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = 64u / LANES;
	const uint32_t which = blockIdx.y;
	const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
	const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
	const uint4 *w_base = (const uint4 *)(which == 0u ? w_f : w_g);
	const uint4 *a_base = (const uint4 *)(which == 0u ? in_f : in_g);
	uint16_t *dst = which == 0u ? out_f : out_g;
	LmDependentWait();
	if ( col0 < OUT_DIM )
	{
		const bool has1 = col0 + 1u < OUT_DIM;
		const uint4 *w0 = w_base + (uint64_t)col0 * CHUNKS;
		const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
		float acc0 = 0.0f, acc1 = 0.0f;
		#pragma unroll
		for ( uint32_t c = sub; c < CHUNKS; c += LANES )
		{
			const uint4 va = __ldg(a_base + c);
			acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w0 + c), va, 0.0f), acc0);
			acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w1 + c), va, 0.0f), acc1);
		}
		#pragma unroll
		for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
		{
			acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
			acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
		}
		if ( sub == 0u )
		{
			dst[col0] = LmFloatToBf16(acc0);
			if ( has1 )
				dst[col0 + 1u] = LmFloatToBf16(acc1);
		}
	}
	LmDependentRelease();
}

template<class G, class Format, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void LingMoeExactW1W3SiluKernel(const void *__restrict__ w1, const float *__restrict__ s1, const void *__restrict__ w3,
	const float *__restrict__ s3, const uint16_t *__restrict__ activation, const float *__restrict__ packed_weight,
	const uint32_t *__restrict__ active_experts, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ output, float limit)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t K = G::kHidden;
	constexpr uint32_t N = G::kMoeInter;
	constexpr uint32_t CHUNKS = K / ELEMS;
	constexpr uint32_t SC_STRIDE = K / LG_FP8_BLOCK;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = THREADS / LANES;
	constexpr uint32_t MAX_R = 8u;
	constexpr uint32_t VECS_PER_CHUNK = IS_FP8 ? 2u : 1u;
	constexpr uint32_t ACT_STEP_VECS = LANES * VECS_PER_CHUNK;
	__shared__ __align__(16) uint4 s_act[MAX_R][ACT_STEP_VECS];
	__shared__ uint32_t s_src[MAX_R];
	const uint32_t act_idx = blockIdx.y;
	LmDependentWait();
	if ( act_idx < __ldg(active_experts) )
	{
		const uint32_t exp = __ldg(active_experts + 1u + act_idx);
		const uint32_t r_begin = __ldg(group_row_offset + exp);
		const uint32_t r_end = __ldg(group_row_offset + exp + 1u);
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col = blockIdx.x * GROUPS + group;
		const bool valid_col = col < N;
		const uint64_t row_base = (uint64_t)exp * N + (valid_col ? col : 0u);
		const uint4 *w1_ptr = (const uint4 *)w1 + row_base * CHUNKS;
		const uint4 *w3_ptr = (const uint4 *)w3 + row_base * CHUNKS;
		const float *s1_ptr = IS_FP8 ? s1 + row_base * SC_STRIDE : (const float *)0;
		const float *s3_ptr = IS_FP8 ? s3 + row_base * SC_STRIDE : (const float *)0;
		for ( uint32_t r0 = r_begin; r0 < r_end; r0 += MAX_R )
		{
			const uint32_t count = r_end - r0 < MAX_R ? r_end - r0 : MAX_R;
			if ( threadIdx.x < count )
				s_src[threadIdx.x] = __ldg(route_source_token + r0 + threadIdx.x);
			__syncthreads();
			if ( count == 1u )
			{
				if ( valid_col )
				{
					const uint16_t *a_row = activation + (uint64_t)s_src[0] * K;
					float g0 = 0.0f, u0 = 0.0f;
					for ( uint32_t c = sub; c < CHUNKS; c += LANES )
					{
						const typename LmSkinnyFormat<Format>::Activation a = LmSkinnyFormat<Format>::Load(a_row + c * ELEMS);
						const uint4 vw1 = __ldcs(w1_ptr + c);
						const uint4 vw3 = __ldcs(w3_ptr + c);
						const uint32_t sg = c >> 3u;
						const float f1 = IS_FP8 ? __ldg(s1_ptr + sg) : 1.0f;
						const float f3 = IS_FP8 ? __ldg(s3_ptr + sg) : 1.0f;
						g0 = fmaf(f1, LmSkinnyFormat<Format>::Chunk(vw1, a), g0);
						u0 = fmaf(f3, LmSkinnyFormat<Format>::Chunk(vw3, a), u0);
					}
					#pragma unroll
					for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
					{
						g0 += __shfl_down_sync(0xffffffffu, g0, off);
						u0 += __shfl_down_sync(0xffffffffu, u0, off);
					}
					if ( sub == 0u )
					{
						const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
							!G::kExpertWeightAfterDown ? __ldg(packed_weight + r0) : 1.0f, limit };
						output[(uint64_t)r0 * N + col] = LmFloatToBf16(op(SparkMathRoundBf16(g0), SparkMathRoundBf16(u0)));
					}
				}
			}
			else if ( count <= 4u )
			{
				float g[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
				float u[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
				for ( uint32_t c_base = 0u; c_base < CHUNKS; c_base += LANES )
				{
					const uint32_t rem_c = CHUNKS - c_base < LANES ? CHUNKS - c_base : LANES;
					const uint32_t rem_v = rem_c * VECS_PER_CHUNK;
					for ( uint32_t idx = threadIdx.x; idx < count * rem_v; idx += THREADS )
					{
						const uint32_t r = idx / rem_v, v = idx % rem_v;
						s_act[r][v] = __ldg((const uint4 *)(activation + (uint64_t)s_src[r] * K) + c_base * VECS_PER_CHUNK + v);
					}
					__syncthreads();
					const uint32_t c = c_base + sub;
					if ( valid_col && c < CHUNKS )
					{
						const uint4 vw1 = __ldcs(w1_ptr + c);
						const uint4 vw3 = __ldcs(w3_ptr + c);
						if constexpr ( IS_FP8 )
						{
							const uint32_t sg = c >> 3u;
							const float f1 = __ldg(s1_ptr + sg);
							const float f3 = __ldg(s3_ptr + sg);
							const typename LmSkinnyFormat<LmFp8>::Decoded d1 = LmSkinnyFormat<LmFp8>::Decode(vw1);
							#pragma unroll
							for ( uint32_t r = 0u; r < 4u; ++r )
								if ( r < count )
								{
									LmSkinnyWide a;
									a.low = s_act[r][2u * sub];
									a.high = s_act[r][2u * sub + 1u];
									g[r] = fmaf(f1, LmSkinnyFormat<LmFp8>::ChunkDecoded(d1, a), g[r]);
								}
							const typename LmSkinnyFormat<LmFp8>::Decoded d3 = LmSkinnyFormat<LmFp8>::Decode(vw3);
							#pragma unroll
							for ( uint32_t r = 0u; r < 4u; ++r )
								if ( r < count )
								{
									LmSkinnyWide a;
									a.low = s_act[r][2u * sub];
									a.high = s_act[r][2u * sub + 1u];
									u[r] = fmaf(f3, LmSkinnyFormat<LmFp8>::ChunkDecoded(d3, a), u[r]);
								}
						}
						else
						{
							float w1f[8], w3f[8];
							LmSkinnyUnpackBf16x8(vw1, w1f);
							LmSkinnyUnpackBf16x8(vw3, w3f);
							#pragma unroll
							for ( uint32_t r = 0u; r < 4u; ++r )
								if ( r < count )
								{
									float af[8];
									LmSkinnyUnpackBf16x8(s_act[r][sub], af);
									g[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w1f, af), g[r]);
									u[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w3f, af), u[r]);
								}
						}
					}
					if ( c_base + LANES < CHUNKS )
						__syncthreads();
				}
				if ( valid_col )
				{
					#pragma unroll
					for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
						#pragma unroll
						for ( uint32_t r = 0u; r < 4u; ++r )
						{
							g[r] += __shfl_down_sync(0xffffffffu, g[r], off);
							u[r] += __shfl_down_sync(0xffffffffu, u[r], off);
						}
					if ( sub == 0u )
						#pragma unroll
						for ( uint32_t r = 0u; r < 4u; ++r )
							if ( r < count )
							{
								const uint32_t pr = r0 + r;
								const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
									!G::kExpertWeightAfterDown ? __ldg(packed_weight + pr) : 1.0f, limit };
								output[(uint64_t)pr * N + col] = LmFloatToBf16(op(SparkMathRoundBf16(g[r]), SparkMathRoundBf16(u[r])));
							}
				}
			}
			else
			{
				float g[MAX_R], u[MAX_R];
				#pragma unroll
				for ( uint32_t r = 0u; r < MAX_R; ++r )
					g[r] = u[r] = 0.0f;
				for ( uint32_t c_base = 0u; c_base < CHUNKS; c_base += LANES )
				{
					const uint32_t rem_c = CHUNKS - c_base < LANES ? CHUNKS - c_base : LANES;
					const uint32_t rem_v = rem_c * VECS_PER_CHUNK;
					for ( uint32_t idx = threadIdx.x; idx < count * rem_v; idx += THREADS )
					{
						const uint32_t r = idx / rem_v, v = idx % rem_v;
						s_act[r][v] = __ldg((const uint4 *)(activation + (uint64_t)s_src[r] * K) + c_base * VECS_PER_CHUNK + v);
					}
					__syncthreads();
					const uint32_t c = c_base + sub;
					if ( valid_col && c < CHUNKS )
					{
						const uint4 vw1 = __ldcs(w1_ptr + c);
						const uint4 vw3 = __ldcs(w3_ptr + c);
						if constexpr ( IS_FP8 )
						{
							const uint32_t sg = c >> 3u;
							const float f1 = __ldg(s1_ptr + sg);
							const float f3 = __ldg(s3_ptr + sg);
							const typename LmSkinnyFormat<LmFp8>::Decoded d1 = LmSkinnyFormat<LmFp8>::Decode(vw1);
							#pragma unroll
							for ( uint32_t r = 0u; r < MAX_R; ++r )
								if ( r < count )
								{
									LmSkinnyWide a;
									a.low = s_act[r][2u * sub];
									a.high = s_act[r][2u * sub + 1u];
									g[r] = fmaf(f1, LmSkinnyFormat<LmFp8>::ChunkDecoded(d1, a), g[r]);
								}
							const typename LmSkinnyFormat<LmFp8>::Decoded d3 = LmSkinnyFormat<LmFp8>::Decode(vw3);
							#pragma unroll
							for ( uint32_t r = 0u; r < MAX_R; ++r )
								if ( r < count )
								{
									LmSkinnyWide a;
									a.low = s_act[r][2u * sub];
									a.high = s_act[r][2u * sub + 1u];
									u[r] = fmaf(f3, LmSkinnyFormat<LmFp8>::ChunkDecoded(d3, a), u[r]);
								}
						}
						else
						{
							float w1f[8], w3f[8];
							LmSkinnyUnpackBf16x8(vw1, w1f);
							LmSkinnyUnpackBf16x8(vw3, w3f);
							#pragma unroll
							for ( uint32_t r = 0u; r < MAX_R; ++r )
								if ( r < count )
								{
									float af[8];
									LmSkinnyUnpackBf16x8(s_act[r][sub], af);
									g[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w1f, af), g[r]);
									u[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w3f, af), u[r]);
								}
						}
					}
					if ( c_base + LANES < CHUNKS )
						__syncthreads();
				}
				if ( valid_col )
				{
					#pragma unroll
					for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
						#pragma unroll
						for ( uint32_t r = 0u; r < MAX_R; ++r )
						{
							g[r] += __shfl_down_sync(0xffffffffu, g[r], off);
							u[r] += __shfl_down_sync(0xffffffffu, u[r], off);
						}
					if ( sub == 0u )
						#pragma unroll
						for ( uint32_t r = 0u; r < MAX_R; ++r )
							if ( r < count )
							{
								const uint32_t pr = r0 + r;
								const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
									!G::kExpertWeightAfterDown ? __ldg(packed_weight + pr) : 1.0f, limit };
								output[(uint64_t)pr * N + col] = LmFloatToBf16(op(SparkMathRoundBf16(g[r]), SparkMathRoundBf16(u[r])));
							}
				}
			}
			if ( r0 + MAX_R < r_end )
				__syncthreads();
		}
	}
	LmDependentRelease();
}

template<class G, class Format>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMoeExactW2Kernel(const void *__restrict__ w2, const float *__restrict__ s2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ active_experts, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t N = G::kHidden;
	constexpr uint32_t CHUNKS = K / ELEMS;
	constexpr uint32_t SC_STRIDE = K / LG_FP8_BLOCK;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = LG_THREADS / LANES;
	constexpr uint32_t MAX_R = 8u;
	constexpr uint32_t ACT_VECS = K >> 3u;
	constexpr bool HOIST_W = !IS_FP8 && CHUNKS <= 2u * LANES;
	__shared__ __align__(16) uint4 s_act[MAX_R][ACT_VECS];
	const uint32_t act_idx = blockIdx.y;
	LmDependentWait();
	if ( act_idx < __ldg(active_experts) )
	{
		const uint32_t exp = __ldg(active_experts + 1u + act_idx);
		const uint32_t r_begin = __ldg(group_row_offset + exp);
		const uint32_t r_end = __ldg(group_row_offset + exp + 1u);
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col = blockIdx.x * GROUPS + group;
		const bool valid_col = col < N;
		const uint64_t row_base = (uint64_t)exp * N + (valid_col ? col : 0u);
		const uint4 *w_ptr = (const uint4 *)w2 + row_base * CHUNKS;
		const float *s_ptr = IS_FP8 ? s2 + row_base * SC_STRIDE : (const float *)0;
		if constexpr ( HOIST_W )
		{
			uint4 hw0 = valid_col && sub < CHUNKS ? __ldcs(w_ptr + sub) : make_uint4(0u, 0u, 0u, 0u);
			uint4 hw1 = valid_col && sub + LANES < CHUNKS ? __ldcs(w_ptr + sub + LANES) : make_uint4(0u, 0u, 0u, 0u);
			for ( uint32_t r0 = r_begin; r0 < r_end; r0 += MAX_R )
			{
				const uint32_t count = r_end - r0 < MAX_R ? r_end - r0 : MAX_R;
				const uint32_t total_v = count * ACT_VECS;
				for ( uint32_t idx = threadIdx.x; idx < total_v; idx += LG_THREADS )
					((uint4 *)s_act)[idx] = __ldg((const uint4 *)(expert_act + (uint64_t)r0 * K) + idx);
				__syncthreads();
				if ( valid_col )
				{
					#pragma unroll
					for ( uint32_t r = 0u; r < MAX_R; ++r )
						if ( r < count )
						{
							float acc = 0.0f;
							if ( sub < CHUNKS )
								acc = fmaf(1.0f, LmSkinnyBf16Chunk(hw0, s_act[r][sub], 0.0f), acc);
							if ( sub + LANES < CHUNKS )
								acc = fmaf(1.0f, LmSkinnyBf16Chunk(hw1, s_act[r][sub + LANES], 0.0f), acc);
							#pragma unroll
							for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
								acc += __shfl_down_sync(0xffffffffu, acc, off);
							if ( sub == 0u )
								output[(uint64_t)(r0 + r) * N + col] = LmFloatToBf16(acc);
						}
				}
				if ( r0 + MAX_R < r_end )
					__syncthreads();
			}
		}
		else
		{
			for ( uint32_t r0 = r_begin; r0 < r_end; r0 += MAX_R )
			{
				const uint32_t count = r_end - r0 < MAX_R ? r_end - r0 : MAX_R;
				const uint32_t total_v = count * ACT_VECS;
				for ( uint32_t idx = threadIdx.x; idx < total_v; idx += LG_THREADS )
					((uint4 *)s_act)[idx] = __ldg((const uint4 *)(expert_act + (uint64_t)r0 * K) + idx);
				__syncthreads();
				if ( valid_col )
				{
					if ( count == 1u )
					{
						float acc = 0.0f;
						for ( uint32_t c = sub; c < CHUNKS; c += LANES )
						{
							const uint4 vw = __ldcs(w_ptr + c);
							if constexpr ( IS_FP8 )
							{
								LmSkinnyWide a;
								a.low = s_act[0][2u * c];
								a.high = s_act[0][2u * c + 1u];
								acc = fmaf(__ldg(s_ptr + (c >> 3u)), LmSkinnyFormat<LmFp8>::Chunk(vw, a), acc);
							}
							else
								acc = fmaf(1.0f, LmSkinnyBf16Chunk(vw, s_act[0][c], 0.0f), acc);
						}
						#pragma unroll
						for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
							acc += __shfl_down_sync(0xffffffffu, acc, off);
						if ( sub == 0u )
							output[(uint64_t)r0 * N + col] = LmFloatToBf16(acc);
					}
					else if ( count <= 4u )
					{
						float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
						for ( uint32_t c = sub; c < CHUNKS; c += LANES )
						{
							const uint4 vw = __ldcs(w_ptr + c);
							if constexpr ( IS_FP8 )
							{
								const float sc = __ldg(s_ptr + (c >> 3u));
								const typename LmSkinnyFormat<LmFp8>::Decoded dec = LmSkinnyFormat<LmFp8>::Decode(vw);
								#pragma unroll
								for ( uint32_t r = 0u; r < 4u; ++r )
									if ( r < count )
									{
										LmSkinnyWide a;
										a.low = s_act[r][2u * c];
										a.high = s_act[r][2u * c + 1u];
										acc[r] = fmaf(sc, LmSkinnyFormat<LmFp8>::ChunkDecoded(dec, a), acc[r]);
									}
							}
							else
							{
								float wf[8];
								LmSkinnyUnpackBf16x8(vw, wf);
								#pragma unroll
								for ( uint32_t r = 0u; r < 4u; ++r )
									if ( r < count )
									{
										float af[8];
										LmSkinnyUnpackBf16x8(s_act[r][c], af);
										acc[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf, af), acc[r]);
									}
							}
						}
						#pragma unroll
						for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
							#pragma unroll
							for ( uint32_t r = 0u; r < 4u; ++r )
								acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
						if ( sub == 0u )
							#pragma unroll
							for ( uint32_t r = 0u; r < 4u; ++r )
								if ( r < count )
									output[(uint64_t)(r0 + r) * N + col] = LmFloatToBf16(acc[r]);
					}
					else
					{
						float acc[MAX_R];
						#pragma unroll
						for ( uint32_t r = 0u; r < MAX_R; ++r )
							acc[r] = 0.0f;
						for ( uint32_t c = sub; c < CHUNKS; c += LANES )
						{
							const uint4 vw = __ldcs(w_ptr + c);
							if constexpr ( IS_FP8 )
							{
								const float sc = __ldg(s_ptr + (c >> 3u));
								const typename LmSkinnyFormat<LmFp8>::Decoded dec = LmSkinnyFormat<LmFp8>::Decode(vw);
								#pragma unroll
								for ( uint32_t r = 0u; r < MAX_R; ++r )
									if ( r < count )
									{
										LmSkinnyWide a;
										a.low = s_act[r][2u * c];
										a.high = s_act[r][2u * c + 1u];
										acc[r] = fmaf(sc, LmSkinnyFormat<LmFp8>::ChunkDecoded(dec, a), acc[r]);
									}
							}
							else
							{
								float wf[8];
								LmSkinnyUnpackBf16x8(vw, wf);
								#pragma unroll
								for ( uint32_t r = 0u; r < MAX_R; ++r )
									if ( r < count )
									{
										float af[8];
										LmSkinnyUnpackBf16x8(s_act[r][c], af);
										acc[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf, af), acc[r]);
									}
							}
						}
						#pragma unroll
						for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
							#pragma unroll
							for ( uint32_t r = 0u; r < MAX_R; ++r )
								acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
						if ( sub == 0u )
							#pragma unroll
							for ( uint32_t r = 0u; r < MAX_R; ++r )
								if ( r < count )
									output[(uint64_t)(r0 + r) * N + col] = LmFloatToBf16(acc[r]);
					}
				}
				if ( r0 + MAX_R < r_end )
					__syncthreads();
			}
		}
	}
	LmDependentRelease();
}

template<class G, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void LingMoeMmaW1W3SiluFp8Kernel(const uint8_t *__restrict__ w1, const float *__restrict__ s1,
	const uint8_t *__restrict__ w3, const float *__restrict__ s3, const uint16_t *__restrict__ act,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ tile_prefix, const uint32_t *__restrict__ route_source_token,
	uint16_t *__restrict__ inter_out, float limit)
{
	constexpr uint32_t K = G::kHidden, N = G::kMoeInter, TILE_N = 64u, TILE_K = 128u;
	constexpr uint32_t N_TILES = N / TILE_N, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u, B_ROW_BYTES = TILE_K, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * A_ROW_CHUNKS) / 256u, B_PER_T = (TILE_N * B_ROW_CHUNKS) / 256u;
	constexpr uint32_t A_STEP = 256u / A_ROW_CHUNKS, B_STEP = 256u / B_ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t a_row0 = threadIdx.x / A_ROW_CHUNKS, a_col0 = threadIdx.x % A_ROW_CHUNKS;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G::kLocalExperts);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		LingGroupedTileLookup(group_row_offset, tile_prefix, G::kLocalExperts, tile, TILE_M, N_TILES, grp, row_base, row_limit, nt);
		const uint32_t col_base = nt * TILE_N;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t packed = row_base + a_row0 + q * A_STEP;
			const uint32_t valid = packed < row_limit ? 1u : 0u;
			const uint32_t sr = valid != 0u ? __ldg(route_source_token + packed) : 0u;
			a_src[q] = act + (uint64_t)sr * K + a_col0 * 8u;
			a_mask |= valid << q;
		}
		const uint64_t exp_base = (uint64_t)grp * N + col_base;
		const uint8_t *bg_base = w1 + (exp_base + b_row0) * K + b_col0 * 16u;
		const uint8_t *bu_base = w3 + (exp_base + b_row0) * K + b_col0 * 16u;
		const uint64_t b_ptr_step = (uint64_t)B_STEP * K;
		auto issue_k = [&](uint32_t s_idx)
		{
			const uint32_t stage = shared_base + s_idx * STAGE_BYTES;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t r = a_row0 + q * A_STEP;
				const uint32_t valid = (a_mask >> q) & 1u;
				LmStreamCopy(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, a_col0) * 16u, a_src[q], valid << 4u);
				a_src[q] += TILE_K;
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = b_row0 + q * B_STEP;
				const uint32_t sw = LmStreamWeight<LmFp8>::template Offset<B_ROW_BYTES>(r, b_col0);
				LmStreamCopy(stage + A_BYTES + sw, bg_base + q * b_ptr_step, 16u);
				LmStreamCopy(stage + A_BYTES + B_BYTES + sw, bu_base + q * b_ptr_step, 16u);
			}
			bg_base += TILE_K;
			bu_base += TILE_K;
		};
		float tg[MF][NF][4], tu[MF][NF][4], pg[MF][NF][4], pu[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					tg[i][j][e] = tu[i][j][e] = pg[i][j][e] = pu[i][j][e] = 0.0f;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( s < K_TILES )
				issue_k(s);
			LmStreamCommit();
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			const uint32_t ahead = kt + STAGES - 1u;
			if ( ahead < K_TILES )
				issue_k(ahead % STAGES);
			LmStreamCommit();
			float sg_s[NF][2], su_s[NF][2];
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				const uint32_t col = warp_n * WARP_N + j * 8u + quad * 2u;
				sg_s[j][0] = __ldg(s1 + (exp_base + col) * K_TILES + kt);
				sg_s[j][1] = __ldg(s1 + (exp_base + col + 1u) * K_TILES + kt);
				su_s[j][0] = __ldg(s3 + (exp_base + col) * K_TILES + kt);
				su_s[j][1] = __ldg(s3 + (exp_base + col + 1u) * K_TILES + kt);
			}
			const uint32_t stage_a = shared_base + (kt % STAGES) * STAGE_BYTES;
			const uint32_t stage_bg = stage_a + A_BYTES;
			const uint32_t stage_bu = stage_bg + B_BYTES;
			#pragma unroll
			for ( uint32_t step = 0u; step < 4u; ++step )
			{
				uint32_t bg[NF][4], bu[NF][4];
				if constexpr ( NF == 1u )
				{
					LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_bg, warp_n * WARP_N, step, lane, bg[0]);
					LmStreamWeight<LmFp8>::template Fragment<B_ROW_BYTES>(stage_bu, warp_n * WARP_N, step, lane, bu[0]);
				}
				else
				{
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; j += 2u )
					{
						LmStreamWeight<LmFp8>::template FragmentPair<B_ROW_BYTES>(stage_bg, warp_n * WARP_N + j * 8u, step, lane, bg[j], bg[j + 1u]);
						LmStreamWeight<LmFp8>::template FragmentPair<B_ROW_BYTES>(stage_bu, warp_n * WARP_N + j * 8u, step, lane, bu[j], bu[j + 1u]);
					}
				}
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
				{
					const uint32_t row = ldm_a_row + i * 16u;
					uint32_t af[4];
					LmStreamLdMatrix4(af, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						uint32_t bg0[2] = { bg[j][0], bg[j][1] }, bu0[2] = { bu[j][0], bu[j][1] };
						LmMmaBf16(pg[i][j], af, bg0);
						LmMmaBf16(pu[i][j], af, bu0);
					}
					LmStreamLdMatrix4(af, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						uint32_t bg1[2] = { bg[j][2], bg[j][3] }, bu1[2] = { bu[j][2], bu[j][3] };
						LmMmaBf16(pg[i][j], af, bg1);
						LmMmaBf16(pu[i][j], af, bu1);
					}
				}
			}
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
					{
						tg[i][j][e] = fmaf(pg[i][j][e], sg_s[j][e & 1u], tg[i][j][e]);
						tu[i][j][e] = fmaf(pu[i][j][e], su_s[j][e & 1u], tu[i][j][e]);
						pg[i][j][e] = 0.0f;
						pu[i][j][e] = 0.0f;
					}
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < row_limit )
					{
						const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
							!G::kExpertWeightAfterDown ? __ldg(packed_weight + r) : 1.0f, limit };
						const uint16_t lo = LmFloatToBf16(op(SparkMathRoundBf16(tg[i][j][e]), SparkMathRoundBf16(tu[i][j][e])));
						const uint16_t hi = LmFloatToBf16(op(SparkMathRoundBf16(tg[i][j][e + 1u]), SparkMathRoundBf16(tu[i][j][e + 1u])));
						*(uint32_t *)(inter_out + (uint64_t)r * N + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<class G, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void LingMoeMmaW1W3SiluBf16Kernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3,
	const uint16_t *__restrict__ act, const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ tile_prefix, const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out, float limit)
{
	constexpr uint32_t K = G::kHidden, N = G::kMoeInter, TILE_N = 64u, TILE_K = 64u;
	constexpr uint32_t N_TILES = N / TILE_N, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u;
	constexpr uint32_t ROW_STEP = 256u / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G::kLocalExperts);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		LingGroupedTileLookup(group_row_offset, tile_prefix, G::kLocalExperts, tile, TILE_M, N_TILES, grp, row_base, row_limit, nt);
		const uint32_t col_base = nt * TILE_N;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			const uint32_t packed = row_base + r;
			const uint32_t valid = chunk < A_CHUNKS && packed < row_limit ? 1u : 0u;
			const uint32_t sr = valid != 0u ? __ldg(route_source_token + packed) : 0u;
			a_src[q] = act + (uint64_t)sr * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint64_t exp_base = (uint64_t)grp * N + col_base;
		const uint16_t *bg_base = w1 + (exp_base + row0) * K + col0 * 8u;
		const uint16_t *bu_base = w3 + (exp_base + row0) * K + col0 * 8u;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		auto issue_k = [&](uint32_t s_idx)
		{
			const uint32_t stage = shared_base + s_idx * STAGE_BYTES;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * 256u;
				if ( chunk < A_CHUNKS )
				{
					const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
					const uint32_t valid = (a_mask >> q) & 1u;
					LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], valid << 4u);
					a_src[q] += TILE_K;
				}
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u;
				LmStreamCopy(stage + A_BYTES + sw, bg_base + q * b_step, 16u);
				LmStreamCopy(stage + A_BYTES + B_BYTES + sw, bu_base + q * b_step, 16u);
			}
			bg_base += TILE_K;
			bu_base += TILE_K;
		};
		uint32_t a_reg[2][MF][4], bg_reg[2][NF][2], bu_reg[2][NF][2];
		float tg[MF][NF][4], tu[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					tg[i][j][e] = tu[i][j][e] = 0.0f;
		auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
		{
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
				const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			}
			if constexpr ( NF == 1u )
			{
				const uint32_t r = warp_n * WARP_N + (lane & 7u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
				LmStreamLdMatrix2(bg_reg[buf][0], stage_s + A_BYTES + off);
				LmStreamLdMatrix2(bu_reg[buf][0], stage_s + A_BYTES + B_BYTES + off);
			}
			else
			{
				#pragma unroll
				for ( uint32_t jj = 0u; jj < NF; jj += 2u )
				{
					const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
					const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
					const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
					uint32_t tmp[4];
					LmStreamLdMatrix4(tmp, stage_s + A_BYTES + off);
					bg_reg[buf][jj][0] = tmp[0]; bg_reg[buf][jj][1] = tmp[1];
					bg_reg[buf][jj + 1u][0] = tmp[2]; bg_reg[buf][jj + 1u][1] = tmp[3];
					LmStreamLdMatrix4(tmp, stage_s + A_BYTES + B_BYTES + off);
					bu_reg[buf][jj][0] = tmp[0]; bu_reg[buf][jj][1] = tmp[1];
					bu_reg[buf][jj + 1u][0] = tmp[2]; bu_reg[buf][jj + 1u][1] = tmp[3];
				}
			}
		};
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( s < K_TILES )
				issue_k(s);
			LmStreamCommit();
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			const uint32_t stage_s = shared_base + (kt % STAGES) * STAGE_BYTES;
			const uint32_t ahead = kt + STAGES - 1u;
			if ( ahead < K_TILES )
				issue_k(ahead % STAGES);
			LmStreamCommit();
			load_frag(0u, 0u, stage_s);
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 4u; ++ks )
			{
				const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
				if ( ks + 1u < 4u )
					load_frag(nxt, ks + 1u, stage_s);
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						LmMmaBf16(tg[i][j], a_reg[cur][i], bg_reg[cur][j]);
						LmMmaBf16(tu[i][j], a_reg[cur][i], bu_reg[cur][j]);
					}
			}
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < row_limit )
					{
						const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
							!G::kExpertWeightAfterDown ? __ldg(packed_weight + r) : 1.0f, limit };
						const uint16_t lo = LmFloatToBf16(op(SparkMathRoundBf16(tg[i][j][e]), SparkMathRoundBf16(tu[i][j][e])));
						const uint16_t hi = LmFloatToBf16(op(SparkMathRoundBf16(tg[i][j][e + 1u]), SparkMathRoundBf16(tu[i][j][e + 1u])));
						*(uint32_t *)(inter_out + (uint64_t)r * N + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<class G, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(WARPS_M * 5u * 32u, 1)
static void LingMoeMmaW1W3SiluN80Kernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3,
	const uint16_t *__restrict__ act, const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ tile_prefix, const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ inter_out, float limit)
{
	constexpr uint32_t K = G::kHidden, N = 80u, TILE_K = 64u, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS_N = 5u, THREADS = WARPS_M * WARPS_N * 32u;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = 16u, MF = WARP_M / 16u, NF = 2u;
	constexpr uint32_t ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = N * ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + THREADS - 1u) / THREADS;
	constexpr uint32_t B_CHUNKS = N * ROW_CHUNKS, B_PER_T = B_CHUNKS / THREADS;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G::kLocalExperts);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		LingGroupedTileLookup(group_row_offset, tile_prefix, G::kLocalExperts, tile, TILE_M, 1u, grp, row_base, row_limit, nt);
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * THREADS;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			const uint32_t packed = row_base + r;
			const uint32_t valid = chunk < A_CHUNKS && packed < row_limit ? 1u : 0u;
			const uint32_t sr = valid != 0u ? __ldg(route_source_token + packed) : 0u;
			a_src[q] = act + (uint64_t)sr * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint64_t exp_base = (uint64_t)grp * N;
		const uint16_t *bg_base = w1 + exp_base * K;
		const uint16_t *bu_base = w3 + exp_base * K;
		uint32_t k_off = 0u;
		auto issue_k = [&](uint32_t s_idx)
		{
			const uint32_t stage = shared_base + s_idx * STAGE_BYTES;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * THREADS;
				if ( chunk < A_CHUNKS )
				{
					const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
					const uint32_t valid = (a_mask >> q) & 1u;
					LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], valid << 4u);
					a_src[q] += TILE_K;
				}
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * THREADS;
				const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
				const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
				const uint64_t boff = (uint64_t)r * K + k_off + c * 8u;
				LmStreamCopy(stage + A_BYTES + sw, bg_base + boff, 16u);
				LmStreamCopy(stage + A_BYTES + B_BYTES + sw, bu_base + boff, 16u);
			}
			k_off += TILE_K;
		};
		uint32_t a_reg[2][MF][4], bg_reg[2][NF][2], bu_reg[2][NF][2];
		float tg[MF][NF][4], tu[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					tg[i][j][e] = tu[i][j][e] = 0.0f;
		auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
		{
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
				const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			}
			const uint32_t r = warp_n * WARP_N + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
			const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
			const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
			uint32_t tmp[4];
			LmStreamLdMatrix4(tmp, stage_s + A_BYTES + off);
			bg_reg[buf][0][0] = tmp[0]; bg_reg[buf][0][1] = tmp[1];
			bg_reg[buf][1][0] = tmp[2]; bg_reg[buf][1][1] = tmp[3];
			LmStreamLdMatrix4(tmp, stage_s + A_BYTES + B_BYTES + off);
			bu_reg[buf][0][0] = tmp[0]; bu_reg[buf][0][1] = tmp[1];
			bu_reg[buf][1][0] = tmp[2]; bu_reg[buf][1][1] = tmp[3];
		};
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( s < K_TILES )
				issue_k(s);
			LmStreamCommit();
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			const uint32_t stage_s = shared_base + (kt % STAGES) * STAGE_BYTES;
			const uint32_t ahead = kt + STAGES - 1u;
			if ( ahead < K_TILES )
				issue_k(ahead % STAGES);
			LmStreamCommit();
			load_frag(0u, 0u, stage_s);
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 4u; ++ks )
			{
				const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
				if ( ks + 1u < 4u )
					load_frag(nxt, ks + 1u, stage_s);
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						LmMmaBf16(tg[i][j], a_reg[cur][i], bg_reg[cur][j]);
						LmMmaBf16(tu[i][j], a_reg[cur][i], bu_reg[cur][j]);
					}
			}
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < row_limit )
					{
						const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
							!G::kExpertWeightAfterDown ? __ldg(packed_weight + r) : 1.0f, limit };
						const uint16_t lo = LmFloatToBf16(op(SparkMathRoundBf16(tg[i][j][e]), SparkMathRoundBf16(tu[i][j][e])));
						const uint16_t hi = LmFloatToBf16(op(SparkMathRoundBf16(tg[i][j][e + 1u]), SparkMathRoundBf16(tu[i][j][e + 1u])));
						*(uint32_t *)(inter_out + (uint64_t)r * N + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<class G, uint32_t TILE_M, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void LingMoeMmaW2K80SliceKernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ inter_in,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ tile_prefix,
	uint16_t *__restrict__ out_bf16, uint32_t col_begin, uint32_t slice_cols)
{
	constexpr uint32_t K = 80u, N = G::kHidden, TILE_N = 64u, ROW_CHUNKS = 10u, ROW_BYTES = 160u;
	constexpr uint32_t WARPS = 8u, WARPS_N = WARPS / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u;
	constexpr uint32_t B_CHUNKS = TILE_N * ROW_CHUNKS, B_PER_T = (B_CHUNKS + 255u) / 256u;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t n_tiles = slice_cols / TILE_N;
	LmDependentWait();
	const uint32_t total_tiles = __ldg(tile_prefix + G::kLocalExperts);
	if ( blockIdx.x >= total_tiles )
	{
		LmDependentRelease();
		return;
	}
	uint32_t rb0 = 0u, rl0 = 0u, nt0 = 0u, rb1 = 0u, rl1 = 0u, nt1 = 0u;
	auto issue_tile = [&](uint32_t t_idx, uint32_t s_idx)
	{
		uint32_t grp_o, rb_o, rl_o, nt_o;
		LingGroupedTileLookup(group_row_offset, tile_prefix, G::kLocalExperts, t_idx, TILE_M, n_tiles, grp_o, rb_o, rl_o, nt_o);
		if ( s_idx == 0u )
		{
			rb0 = rb_o; rl0 = rl_o; nt0 = nt_o;
		}
		else
		{
			rb1 = rb_o; rl1 = rl_o; nt1 = nt_o;
		}
		const uint32_t stage = shared_base + s_idx * STAGE_BYTES;
		const uint32_t col_base = col_begin + nt_o * TILE_N;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			if ( chunk < A_CHUNKS )
			{
				const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
				const uint32_t packed = rb_o + r, valid = packed < rl_o ? 1u : 0u;
				LmStreamCopy(stage + r * ROW_BYTES + LingSwizzle10(r, c) * 16u,
					valid != 0u ? inter_in + (uint64_t)packed * K + c * 8u : inter_in, valid << 4u);
			}
		}
		const uint16_t *b_base = w2 + ((uint64_t)grp_o * N + col_base) * K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			if ( chunk < B_CHUNKS )
			{
				const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
				LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LingSwizzle10(r, c) * 16u, b_base + (uint64_t)r * K + c * 8u, 16u);
			}
		}
	};
	uint32_t a_reg[2][MF][4], b_reg[2][NF][2];
	auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
	{
		#pragma unroll
		for ( uint32_t ii = 0u; ii < MF; ++ii )
		{
			const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
			const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
			LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LingSwizzle10(r, c) * 16u);
		}
		if constexpr ( NF == 1u )
		{
			const uint32_t r = warp_n * WARP_N + (lane & 7u);
			const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
			LmStreamLdMatrix2(b_reg[buf][0], stage_s + A_BYTES + r * ROW_BYTES + LingSwizzle10(r, c) * 16u);
		}
		else
		{
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LingSwizzle10(r, c) * 16u);
				b_reg[buf][jj][0] = tmp[0]; b_reg[buf][jj][1] = tmp[1];
				b_reg[buf][jj + 1u][0] = tmp[2]; b_reg[buf][jj + 1u][1] = tmp[3];
			}
		}
	};
	const uint32_t n_iters = (total_tiles - 1u - blockIdx.x) / gridDim.x + 1u;
	#pragma unroll
	for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
	{
		if ( s < n_iters )
			issue_tile(blockIdx.x + s * gridDim.x, s);
		LmStreamCommit();
	}
	for ( uint32_t iter = 0u; iter < n_iters; ++iter )
	{
		LmStreamWait<STAGES - 2u>();
		__syncthreads();
		const uint32_t cur_s = iter % STAGES;
		const uint32_t stage_s = shared_base + cur_s * STAGE_BYTES;
		const uint32_t ahead = iter + STAGES - 1u;
		if ( ahead < n_iters )
			issue_tile(blockIdx.x + ahead * gridDim.x, ahead % STAGES);
		LmStreamCommit();
		float acc[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
		load_frag(0u, 0u, stage_s);
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 5u; ++ks )
		{
			const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
			if ( ks + 1u < 5u )
				load_frag(nxt, ks + 1u, stage_s);
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
					LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
		}
		const uint32_t cur_rb = cur_s == 0u ? rb0 : rb1;
		const uint32_t cur_rl = cur_s == 0u ? rl0 : rl1;
		const uint32_t out_col_base = (cur_s == 0u ? nt0 : nt1) * TILE_N;
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = cur_rb + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = out_col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < cur_rl )
						*(uint32_t *)(out_bf16 + (uint64_t)r * slice_cols + c) =
							(uint32_t)LmFloatToBf16(acc[i][j][e]) | ((uint32_t)LmFloatToBf16(acc[i][j][e + 1u]) << 16u);
				}
	}
	LmStreamWait<0u>();
	LmDependentRelease();
}

template<bool WEIGHTED, uint32_t TOPK>
__global__ __launch_bounds__(128u, 4)
static void LingMoeFinalizeWarpSliceKernel(const uint16_t *__restrict__ packed_slice,
	const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert,
	const float *__restrict__ route_weight, const uint16_t *__restrict__ shared,
	uint16_t *__restrict__ output, uint32_t tokens, uint32_t local_experts,
	uint32_t col_offset, uint32_t slice_cols, uint32_t total_cols)
{
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t token = blockIdx.x * 4u + warp;
	if ( token < tokens )
	{
		const uint32_t my_exp = lane < TOPK ? __ldg(route_expert + (uint64_t)token * TOPK + lane) : local_experts;
		const uint32_t my_pr = (lane < TOPK && my_exp < local_experts) ? __ldg(route_packed_row + (uint64_t)token * TOPK + lane) : 0u;
		const float my_rw = (lane < TOPK && my_exp < local_experts) ? __ldg(route_weight + (uint64_t)token * TOPK + lane) : 0.0f;
		const uint32_t active_mask = __ballot_sync(0xffffffffu, lane < TOPK && my_exp < local_experts);
		for ( uint32_t elem = lane * 8u; elem < slice_cols; elem += 32u * 8u )
		{
			float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			uint32_t mask = active_mask;
			while ( mask != 0u )
			{
				const uint32_t r = (uint32_t)(__ffs(mask) - 1);
				mask &= mask - 1u;
				const uint32_t pr = __shfl_sync(0xffffffffu, my_pr, r);
				const float rw = __shfl_sync(0xffffffffu, my_rw, r);
				const uint4 v = *(const uint4 *)(packed_slice + (uint64_t)pr * slice_cols + elem);
				if constexpr ( WEIGHTED )
					LmFmaBf16x8(acc, v, rw);
				else
					LmAddBf16x8(acc, v);
			}
			LmAddBf16x8(acc, __ldcs((const uint4 *)(shared + (uint64_t)token * total_cols + col_offset + elem)));
			*(uint4 *)(output + (uint64_t)token * total_cols + col_offset + elem) = LmPackUint4FloatToBf16(acc);
		}
	}
	LmDependentRelease();
}

template<class G, uint32_t TAG, class Kernel>
static int32_t LingMoeMmaSetup(Kernel kernel, uint32_t threads, uint32_t shared_bytes, uint32_t sms, uint32_t *blocks_out)
{
	static std::atomic<uint64_t> granted(0u);
	static std::atomic<int32_t> resident(0);
	int device = 0, per_sm = 1;
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= LM_STREAM_MAX_DEVICES )
		SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( shared_bytes > 49152u &&
			SparkGpuFuncSetAttribute((const void *)kernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared_bytes) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		if ( SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, (int)threads, (size_t)shared_bytes) != SPARK_GPU_SUCCESS || per_sm < 1 )
			per_sm = 1;
		resident.store(per_sm);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	*blocks_out = sms * (uint32_t)resident.load();
	return LM_LAUNCH_OK;
}
