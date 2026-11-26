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

template<uint32_t RANK, uint32_t OUT_DIM, uint32_t RG>
__global__ __launch_bounds__(64u, 4)
static void LingKdaLowRankPairSkinnyKernel(const uint16_t *__restrict__ w_f, const uint16_t *__restrict__ in_f, uint16_t *__restrict__ out_f,
	const uint16_t *__restrict__ w_g, const uint16_t *__restrict__ in_g, uint16_t *__restrict__ out_g, uint32_t rows)
{
	constexpr uint32_t CHUNKS = RANK >> 3u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = 64u / LANES;
	__shared__ __align__(16) uint4 s_a[RG][CHUNKS];
	const uint32_t base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t which = blockIdx.z;
	const uint4 *w_base = (const uint4 *)(which == 0u ? w_f : w_g);
	const uint4 *a_base = (const uint4 *)(which == 0u ? in_f : in_g) + (uint64_t)base_row * CHUNKS;
	uint16_t *dst = which == 0u ? out_f : out_g;
	const uint32_t total_v = count * CHUNKS;
	for ( uint32_t idx = threadIdx.x; idx < total_v; idx += 64u )
		((uint4 *)s_a)[idx] = __ldg(a_base + idx);
	__syncthreads();
	const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
	const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
	const bool valid_c = col0 < OUT_DIM;
	const bool has1 = col0 + 1u < OUT_DIM;
	const uint32_t safe_c = valid_c ? col0 : 0u;
	const bool reuse = gridDim.y > 1u;
	const uint4 *w0 = w_base + (uint64_t)safe_c * CHUNKS;
	const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
	if constexpr ( RG == 1u )
	{
		float acc0 = 0.0f, acc1 = 0.0f;
		#pragma unroll
		for ( uint32_t c = sub; c < CHUNKS; c += LANES )
		{
			const uint4 va = s_a[0][c];
			const uint4 vw0 = reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
			const uint4 vw1 = reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
			acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(vw0, va, 0.0f), acc0);
			acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1, va, 0.0f), acc1);
		}
		#pragma unroll
		for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
		{
			acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
			acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
		}
		if ( sub == 0u && valid_c )
		{
			if ( has1 )
				*(uint32_t *)(dst + (uint64_t)base_row * OUT_DIM + col0) = LmPackBf16Pair(acc0, acc1);
			else
				dst[(uint64_t)base_row * OUT_DIM + col0] = LmFloatToBf16(acc0);
		}
	}
	else
	{
		float acc0[RG], acc1[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			acc0[r] = acc1[r] = 0.0f;
		#pragma unroll
		for ( uint32_t c = sub; c < CHUNKS; c += LANES )
		{
			float w0f[8], w1f[8];
			LmSkinnyUnpackBf16x8(reuse ? __ldg(w0 + c) : __ldcs(w0 + c), w0f);
			LmSkinnyUnpackBf16x8(reuse ? __ldg(w1 + c) : __ldcs(w1 + c), w1f);
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < count )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[r][c], af);
					acc0[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w0f, af), acc0[r]);
					acc1[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w1f, af), acc1[r]);
				}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			if ( r < count )
			{
				#pragma unroll
				for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
				{
					acc0[r] += __shfl_down_sync(0xffffffffu, acc0[r], offset);
					acc1[r] += __shfl_down_sync(0xffffffffu, acc1[r], offset);
				}
				if ( sub == 0u && valid_c )
				{
					if ( has1 )
						*(uint32_t *)(dst + (uint64_t)(base_row + r) * OUT_DIM + col0) = LmPackBf16Pair(acc0[r], acc1[r]);
					else
						dst[(uint64_t)(base_row + r) * OUT_DIM + col0] = LmFloatToBf16(acc0[r]);
				}
			}
	}
	LmDependentRelease();
}

template<uint32_t RANK, uint32_t OUT_DIM>
static int32_t LingKdaLowRankPairLaunch(const uint16_t *w_f, const uint16_t *in_f, uint16_t *out_f,
	const uint16_t *w_g, const uint16_t *in_g, uint16_t *out_g, uint32_t rows, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, w_f, (uint64_t)OUT_DIM * RANK * sizeof(uint16_t));
	LmWeightPrefetchNote(stream, w_g, (uint64_t)OUT_DIM * RANK * sizeof(uint16_t));
	constexpr uint32_t CHUNKS = RANK >> 3u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t COLS = (64u / LANES) * 2u;
	const uint32_t gx = (OUT_DIM + COLS - 1u) / COLS;
	if ( rows == 1u )
		(void)LM_LAUNCH_DEPENDENT((LingKdaLowRankPairSkinnyKernel<RANK, OUT_DIM, 1u>), dim3(gx, 1u, 2u), 64u, 0, stream,
			w_f, in_f, out_f, w_g, in_g, out_g, rows);
	else
		(void)LM_LAUNCH_DEPENDENT((LingKdaLowRankPairSkinnyKernel<RANK, OUT_DIM, 4u>), dim3(gx, (rows + 3u) / 4u, 2u), 64u, 0, stream,
			w_f, in_f, out_f, w_g, in_g, out_g, rows);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<class Format, uint32_t K, uint32_t N, uint32_t RG>
static __device__ __forceinline__ void LingSkinnySmemOutBody(const uint16_t (*s_in)[K], const void *__restrict__ out_w,
	const float *__restrict__ out_s, uint16_t *__restrict__ output, uint32_t base_row, uint32_t count)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t CHUNKS = K / ELEMS;
	constexpr uint32_t SC_STRIDE = K / LG_FP8_BLOCK;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = LG_THREADS / LANES;
	const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
	const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
	const bool valid_c = col0 < N, has1 = col0 + 1u < N;
	const uint32_t safe_c = valid_c ? col0 : 0u;
	const bool reuse = gridDim.y > 1u;
	const uint4 *w0 = (const uint4 *)out_w + (uint64_t)safe_c * CHUNKS;
	const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
	const float *sc0 = IS_FP8 ? out_s + (uint64_t)safe_c * SC_STRIDE : (const float *)0;
	const float *sc1 = IS_FP8 ? (has1 ? sc0 + SC_STRIDE : sc0) : (const float *)0;
	if constexpr ( RG == 1u )
	{
		float acc0 = 0.0f, acc1 = 0.0f;
		const uint4 *s_in4 = (const uint4 *)s_in[0];
		for ( uint32_t c = sub; c < CHUNKS; c += LANES )
		{
			const uint4 vw0 = reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
			const uint4 vw1 = reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
			if constexpr ( IS_FP8 )
			{
				LmSkinnyWide a;
				a.low = s_in4[2u * c];
				a.high = s_in4[2u * c + 1u];
				const uint32_t sg = c >> 3u;
				acc0 = fmaf(__ldg(sc0 + sg), LmSkinnyFormat<LmFp8>::Chunk(vw0, a), acc0);
				acc1 = fmaf(__ldg(sc1 + sg), LmSkinnyFormat<LmFp8>::Chunk(vw1, a), acc1);
			}
			else
			{
				const uint4 va = s_in4[c];
				acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(vw0, va, 0.0f), acc0);
				acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1, va, 0.0f), acc1);
			}
		}
		#pragma unroll
		for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
		{
			acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
			acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
		}
		if ( sub == 0u && valid_c )
		{
			if ( has1 )
				*(uint32_t *)(output + (uint64_t)base_row * N + col0) = LmPackBf16Pair(acc0, acc1);
			else
				output[(uint64_t)base_row * N + col0] = LmFloatToBf16(acc0);
		}
	}
	else
	{
		float acc0[RG], acc1[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			acc0[r] = acc1[r] = 0.0f;
		for ( uint32_t c = sub; c < CHUNKS; c += LANES )
		{
			const uint4 vw0 = reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
			const uint4 vw1 = reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
			if constexpr ( IS_FP8 )
			{
				const uint32_t sg = c >> 3u;
				const float f0 = __ldg(sc0 + sg), f1 = __ldg(sc1 + sg);
				const typename LmSkinnyFormat<LmFp8>::Decoded d0 = LmSkinnyFormat<LmFp8>::Decode(vw0);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < count )
					{
						const uint4 *sr = (const uint4 *)s_in[r];
						LmSkinnyWide a;
						a.low = sr[2u * c];
						a.high = sr[2u * c + 1u];
						acc0[r] = fmaf(f0, LmSkinnyFormat<LmFp8>::ChunkDecoded(d0, a), acc0[r]);
					}
				const typename LmSkinnyFormat<LmFp8>::Decoded d1 = LmSkinnyFormat<LmFp8>::Decode(vw1);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < count )
					{
						const uint4 *sr = (const uint4 *)s_in[r];
						LmSkinnyWide a;
						a.low = sr[2u * c];
						a.high = sr[2u * c + 1u];
						acc1[r] = fmaf(f1, LmSkinnyFormat<LmFp8>::ChunkDecoded(d1, a), acc1[r]);
					}
			}
			else
			{
				float w0f[8], w1f[8];
				LmSkinnyUnpackBf16x8(vw0, w0f);
				LmSkinnyUnpackBf16x8(vw1, w1f);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < count )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(((const uint4 *)s_in[r])[c], af);
						acc0[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w0f, af), acc0[r]);
						acc1[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w1f, af), acc1[r]);
					}
			}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			if ( r < count )
			{
				#pragma unroll
				for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
				{
					acc0[r] += __shfl_down_sync(0xffffffffu, acc0[r], off);
					acc1[r] += __shfl_down_sync(0xffffffffu, acc1[r], off);
				}
				if ( sub == 0u && valid_c )
				{
					if ( has1 )
						*(uint32_t *)(output + (uint64_t)(base_row + r) * N + col0) = LmPackBf16Pair(acc0[r], acc1[r]);
					else
						output[(uint64_t)(base_row + r) * N + col0] = LmFloatToBf16(acc0[r]);
				}
			}
	}
}

template<class G, class Format, uint32_t RG>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingKdaNormGateOutKernel(const uint16_t *__restrict__ mixed, const uint16_t *__restrict__ gate,
	const uint16_t *__restrict__ norm_weight, const void *__restrict__ out_w, const float *__restrict__ out_s,
	uint16_t *__restrict__ output, uint32_t rows)
{
	using S = LingShape<G>;
	constexpr uint32_t HEADS = S::kKdaRankHeads, HD = G::kKdaDim, K = S::kKdaWidth, N = G::kHidden;
	static_assert(HD == 128u && K <= 1024u, "fused KDA head norm + out fits shared memory");
	__shared__ __align__(16) uint16_t s_in[RG][K];
	__shared__ float s_wsum[8u];
	const uint32_t base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t half_id = threadIdx.x >> 7u, within = threadIdx.x & 127u;
	const uint32_t warp_in_half = within >> 5u, lane = threadIdx.x & 31u;
	const float nw = LmBf16ToFloat(__ldg(norm_weight + within));
	for ( uint32_t r = 0u; r < count; ++r )
	{
		const uint32_t row = base_row + r;
		for ( uint32_t h0 = 0u; h0 < HEADS; h0 += 2u )
		{
			const uint32_t h = h0 + half_id;
			const uint64_t idx = ((uint64_t)row * HEADS + h) * HD + within;
			const float v = h < HEADS ? LmBf16ToFloat(__ldg(mixed + idx)) : 0.0f;
			float sq = v * v;
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				sq += __shfl_down_sync(0xffffffffu, sq, off);
			if ( lane == 0u )
				s_wsum[half_id * 4u + warp_in_half] = sq;
			__syncthreads();
			if ( h < HEADS )
			{
				const float sum = s_wsum[half_id * 4u] + s_wsum[half_id * 4u + 1u] + s_wsum[half_id * 4u + 2u] + s_wsum[half_id * 4u + 3u];
				const float scale = SparkMathRmsScale(sum, (float)HD, G::kNormEpsilon);
				const float g = LmBf16ToFloat(__ldg(gate + idx));
				s_in[r][h * HD + within] = LmFloatToBf16(G::kKdaNormRoundOnce ? SparkMathRmsNormGateOnce(v, scale, nw, SparkMathSigmoidFast(g))
					: SparkMathRoundBf16(SparkMathRoundBf16(v * scale) * nw) * SparkMathSigmoidFast(g));
			}
			if ( h0 + 2u < HEADS || r + 1u < count )
				__syncthreads();
		}
	}
	__syncthreads();
	LingSkinnySmemOutBody<Format, K, N, RG>(s_in, out_w, out_s, output, base_row, count);
	LmDependentRelease();
}

template<class G, class Format>
static SparkStatus LingLaunchKdaNormGateOut(const uint16_t *mixed, const uint16_t *gate, const uint16_t *norm_w,
	const void *out_w, const float *out_s, uint16_t *output, uint32_t rows, SparkGpuStream stream)
{
	using S = LingShape<G>;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t CHUNKS = S::kKdaWidth / ELEMS;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t COLS = (LG_THREADS / LANES) * 2u;
	const uint32_t gx = (G::kHidden + COLS - 1u) / COLS;
	LmWeightPrefetchNote(stream, out_w, (uint64_t)G::kHidden * S::kKdaWidth * (LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32 ? 1u : 2u));
	if ( rows == 1u )
		(void)LM_LAUNCH_DEPENDENT((LingKdaNormGateOutKernel<G, Format, 1u>), dim3(gx, 1u), LG_THREADS, 0, stream,
			mixed, gate, norm_w, out_w, out_s, output, rows);
	else
		(void)LM_LAUNCH_DEPENDENT((LingKdaNormGateOutKernel<G, Format, 4u>), dim3(gx, (rows + 3u) / 4u), LG_THREADS, 0, stream,
			mixed, gate, norm_w, out_w, out_s, output, rows);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

template<class G, class Format, uint32_t RG>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMlaGateOutKernel(const uint16_t *__restrict__ value, const uint16_t *__restrict__ gate,
	const void *__restrict__ out_w, const float *__restrict__ out_s, uint16_t *__restrict__ output, uint32_t rows)
{
	using S = LingShape<G>;
	constexpr uint32_t HEADS = S::kMlaRankHeads, VD = G::kValueDim, K = HEADS * VD, N = G::kHidden;
	static_assert(K <= 1024u, "fused MLA gate + out fits shared memory");
	__shared__ __align__(16) uint16_t s_in[RG][K];
	const uint32_t base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	for ( uint32_t r = 0u; r < count; ++r )
	{
		const uint32_t row = base_row + r;
		for ( uint32_t i = threadIdx.x; i < K; i += LG_THREADS )
		{
			const uint32_t h = i / VD;
			const float g = SparkMathRoundBf16(SparkMathSigmoid(LmBf16ToFloat(__ldg(gate + (uint64_t)row * HEADS + h))));
			s_in[r][i] = LmFloatToBf16(LmBf16ToFloat(__ldg(value + (uint64_t)row * K + i)) * g);
		}
	}
	__syncthreads();
	LingSkinnySmemOutBody<Format, K, N, RG>(s_in, out_w, out_s, output, base_row, count);
	LmDependentRelease();
}

template<class G, class Format>
static SparkStatus LingLaunchMlaGateOut(const uint16_t *value, const uint16_t *gate, const void *out_w,
	const float *out_s, uint16_t *output, uint32_t rows, SparkGpuStream stream)
{
	using S = LingShape<G>;
	constexpr uint32_t K = S::kMlaRankHeads * G::kValueDim;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t CHUNKS = K / ELEMS;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t COLS = (LG_THREADS / LANES) * 2u;
	const uint32_t gx = (G::kHidden + COLS - 1u) / COLS;
	LmWeightPrefetchNote(stream, out_w, (uint64_t)G::kHidden * K * (LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32 ? 1u : 2u));
	if ( rows == 1u )
		(void)LM_LAUNCH_DEPENDENT((LingMlaGateOutKernel<G, Format, 1u>), dim3(gx, 1u), LG_THREADS, 0, stream,
			value, gate, out_w, out_s, output, rows);
	else
		(void)LM_LAUNCH_DEPENDENT((LingMlaGateOutKernel<G, Format, 4u>), dim3(gx, (rows + 3u) / 4u), LG_THREADS, 0, stream,
			value, gate, out_w, out_s, output, rows);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

template<class G, uint32_t RG>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingGqaGateOutBf16Kernel(const uint16_t *__restrict__ merged, const uint16_t *__restrict__ gate,
	const uint16_t *__restrict__ out_w, uint16_t *__restrict__ output, uint32_t rows)
{
	using S = LingShape<G>;
	constexpr uint32_t K = S::kGqaRankWidth, N = G::kHidden;
	static_assert(K <= 1024u, "fused GQA gate + out fits shared memory");
	__shared__ __align__(16) uint16_t s_in[RG][K];
	const uint32_t base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	for ( uint32_t r = 0u; r < count; ++r )
	{
		const uint64_t rbase = (uint64_t)(base_row + r) * K;
		for ( uint32_t i = threadIdx.x; i < K; i += LG_THREADS )
		{
			const float v = LmBf16ToFloat(__ldg(merged + rbase + i));
			const float g = SparkMathRoundBf16(SparkMathSigmoid(LmBf16ToFloat(__ldg(gate + rbase + i))));
			s_in[r][i] = LmFloatToBf16(v * g);
		}
	}
	__syncthreads();
	LingSkinnySmemOutBody<LmBf16Format, K, N, RG>(s_in, out_w, (const float *)0, output, base_row, count);
	LmDependentRelease();
}

template<class G>
static SparkStatus LingLaunchGqaGateOutBf16(const uint16_t *merged, const uint16_t *gate, const uint16_t *out_w,
	uint16_t *output, uint32_t rows, SparkGpuStream stream)
{
	using S = LingShape<G>;
	constexpr uint32_t CHUNKS = S::kGqaRankWidth >> 3u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t COLS = (LG_THREADS / LANES) * 2u;
	const uint32_t gx = (G::kHidden + COLS - 1u) / COLS;
	LmWeightPrefetchNote(stream, out_w, (uint64_t)G::kHidden * S::kGqaRankWidth * sizeof(uint16_t));
	if ( rows == 1u )
		(void)LM_LAUNCH_DEPENDENT((LingGqaGateOutBf16Kernel<G, 1u>), dim3(gx, 1u), LG_THREADS, 0, stream,
			merged, gate, out_w, output, rows);
	else
		(void)LM_LAUNCH_DEPENDENT((LingGqaGateOutBf16Kernel<G, 4u>), dim3(gx, (rows + 3u) / 4u), LG_THREADS, 0, stream,
			merged, gate, out_w, output, rows);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingGqaPrepStoreLocalKernel(const uint16_t *__restrict__ gathered, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ sequence_of_row, LmKvShardView kv_view, uint16_t *__restrict__ q_local,
	uint16_t *__restrict__ kv_cur, uint32_t rows, uint32_t q_first, uint32_t rank_heads, uint32_t kv_first, uint32_t local_kvh)
{
	using S = LingShape<G>;
	constexpr uint32_t LOCAL_V = S::kGqaRankFused >> 3u;
	constexpr uint32_t Q_V = S::kGqaWidth >> 3u;
	constexpr uint32_t HD_V = G::kGqaHeadDim >> 3u;
	constexpr uint32_t KVH = G::kGqaKvHeads;
	constexpr uint32_t K_V = KVH * HD_V;
	constexpr uint32_t SLOT_V = S::kSlot >> 3u;
	const uint32_t local_q_v = rank_heads * HD_V;
	const uint32_t total_v = local_q_v + SLOT_V;
	const uint32_t row = blockIdx.y;
	const uint32_t idx = blockIdx.x * LG_THREADS + threadIdx.x;
	LmDependentWait();
	if ( row < rows && idx < total_v )
	{
		const uint4 *g4 = (const uint4 *)gathered;
		if ( idx < local_q_v )
		{
			const uint32_t v = q_first * HD_V + idx;
			const uint32_t rank = v / LOCAL_V, elem = v % LOCAL_V;
			((uint4 *)q_local)[(uint64_t)row * local_q_v + idx] = __ldg(g4 + (((uint64_t)rank * rows) + row) * LOCAL_V + elem);
		}
		else
		{
			const uint32_t kv_idx = idx - local_q_v;
			const uint32_t is_v = kv_idx >= K_V ? 1u : 0u;
			const uint32_t head = (is_v != 0u ? kv_idx - K_V : kv_idx) / HD_V;
			const uint32_t vec = (is_v != 0u ? kv_idx - K_V : kv_idx) % HD_V;
			const bool need = head >= kv_first && head < kv_first + local_kvh;
			const uint32_t pos = __ldg(positions + row);
			const bool owns = SparkKvShardOwns(kv_view.shard, pos) != 0u;
			if ( owns || need )
			{
				const uint32_t v = Q_V + kv_idx;
				const uint32_t rank = v / LOCAL_V, elem = v % LOCAL_V;
				const uint4 val = __ldg(g4 + (((uint64_t)rank * rows) + row) * LOCAL_V + elem);
				if ( owns )
				{
					uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, __ldg(sequence_of_row + row), pos, row, LM_KV_ACCESS_WRITE);
					if ( slot != 0 )
						((uint4 *)slot)[kv_idx] = val;
				}
				if ( need )
					((uint4 *)kv_cur)[((uint64_t)(head - kv_first) * rows + row) * (2u * HD_V) + is_v * HD_V + vec] = val;
			}
		}
	}
	LmDependentRelease();
}

template<class Kv, uint32_t KVH, uint32_t HD>
__global__ __launch_bounds__(128u, 4)
static void LingGqaPriorKvDirectKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row,
	uint32_t old_ctx, uint32_t kv_first, uint32_t local_kvh, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t HD_V = HD >> 3u, SLOT_V = 2u * HD_V, K_V = KVH * HD_V;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t k_idx = blockIdx.x * 4u + warp, lh = blockIdx.y;
	LmDependentWait();
	if ( k_idx < old_ctx && lh < local_kvh )
	{
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, __ldg(sequence_of_row), k_idx, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
		{
			const uint32_t kh = kv_first + lh;
			const uint4 *ks = (const uint4 *)slot + kh * HD_V;
			const uint4 *vs = (const uint4 *)slot + K_V + kh * HD_V;
			uint4 *kd = (uint4 *)kv_prior + ((uint64_t)lh * old_ctx + k_idx) * SLOT_V;
			uint4 *vd = kd + HD_V;
			for ( uint32_t c = lane; c < HD_V; c += 32u )
				kd[c] = __ldg(ks + c);
			for ( uint32_t c = lane; c < HD_V; c += 32u )
				vd[c] = __ldg(vs + c);
		}
	}
	LmDependentRelease();
}

template<class Kv, uint32_t KVH, uint32_t HD>
__global__ __launch_bounds__(128u, 4)
static void LingGqaPriorKvPackKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row,
	uint32_t old_ctx, uint32_t max_peer_keys, uint32_t rank_heads, uint32_t group, uint32_t local_kvh,
	uint64_t a2a_stride_bytes, uint8_t *__restrict__ send_bytes)
{
	constexpr uint32_t HD_V = HD >> 3u, SLOT_V = 2u * HD_V, K_V = KVH * HD_V;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t idx = blockIdx.x * 4u + warp, peer = blockIdx.y;
	const uint32_t pos = SparkKvShardLocalPosition(kv_view.shard, idx);
	LmDependentWait();
	if ( idx < max_peer_keys && peer < kv_view.shard.degree && pos < old_ctx )
	{
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, __ldg(sequence_of_row), pos, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
		{
			const uint32_t peer_kv_first = (peer * rank_heads) / group;
			uint4 *peer_dst = (uint4 *)(send_bytes + (uint64_t)peer * a2a_stride_bytes);
			for ( uint32_t lh = 0u; lh < local_kvh; ++lh )
			{
				const uint32_t kh = peer_kv_first + lh;
				const uint4 *ks = (const uint4 *)slot + kh * HD_V;
				const uint4 *vs = (const uint4 *)slot + K_V + kh * HD_V;
				uint4 *kd = peer_dst + ((uint64_t)lh * max_peer_keys + idx) * SLOT_V;
				uint4 *vd = kd + HD_V;
				for ( uint32_t c = lane; c < HD_V; c += 32u )
					kd[c] = __ldg(ks + c);
				for ( uint32_t c = lane; c < HD_V; c += 32u )
					vd[c] = __ldg(vs + c);
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t HD>
__global__ __launch_bounds__(128u, 4)
static void LingGqaPriorKvUnpackKernel(SparkKvShard shard, const uint8_t *__restrict__ recv_bytes,
	uint64_t a2a_stride_bytes, uint32_t old_ctx, uint32_t max_peer_keys, uint32_t local_kvh, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t SLOT_V = (2u * HD) >> 3u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t idx = blockIdx.x * 4u + warp, src_rank = blockIdx.y;
	SparkKvShard src_shard = shard;
	src_shard.rank = src_rank;
	const uint32_t pos = SparkKvShardLocalPosition(src_shard, idx);
	LmDependentWait();
	if ( idx < max_peer_keys && src_rank < shard.degree && pos < old_ctx )
	{
		const uint4 *src_base = (const uint4 *)(recv_bytes + (uint64_t)src_rank * a2a_stride_bytes);
		for ( uint32_t lh = 0u; lh < local_kvh; ++lh )
		{
			const uint4 *s4 = src_base + ((uint64_t)lh * max_peer_keys + idx) * SLOT_V;
			uint4 *d4 = (uint4 *)kv_prior + ((uint64_t)lh * old_ctx + pos) * SLOT_V;
			for ( uint32_t c = lane; c < SLOT_V; c += 32u )
				d4[c] = __ldg(s4 + c);
		}
	}
	LmDependentRelease();
}

template<uint32_t HD, bool CHUNKED>
__global__ __launch_bounds__(64u, 2)
static void LingGqaLocalPrefillAttnKernel(const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ kv_prior,
	const uint16_t *__restrict__ kv_cur, const uint32_t *__restrict__ positions, const uint16_t *__restrict__ gate_bf16,
	uint32_t rows, uint32_t rank_heads, uint32_t q_first, uint32_t group, uint32_t kv_first, uint32_t prior_keys,
	float qk_scale, uint16_t *__restrict__ merged_out, float *__restrict__ partials)
{
	constexpr uint32_t Q_TILE = 16u, KV_TILE = 64u, TILES_PER_CHUNK = 8u, THREADS = 64u;
	constexpr uint32_t K_STEPS = HD / 16u, HALF_K_FRAGS = (KV_TILE / 2u) / 8u, HALF_V_FRAGS = (HD / 2u) / 8u, P_STEPS = KV_TILE / 16u;
	constexpr uint32_t K_CHUNKS = HD / 8u, SLOT_ELEMS = 2u * HD, K_ROW_BYTES = HD * 2u, P_ROW_BYTES = KV_TILE * 2u;
	static_assert(HD == 128u, "Ling local GQA prefill MMA requires head_dim 128");
	__shared__ __align__(128) uint8_t k_smem[KV_TILE * K_ROW_BYTES];
	__shared__ __align__(128) uint8_t v_smem[KV_TILE * K_ROW_BYTES];
	__shared__ __align__(128) uint8_t p_smem[Q_TILE * P_ROW_BYTES];
	__shared__ float s_peak[2][Q_TILE];
	__shared__ float s_sum[2][Q_TILE];
	const uint32_t k_base = (uint32_t)__cvta_generic_to_shared(k_smem);
	const uint32_t v_base = (uint32_t)__cvta_generic_to_shared(v_smem);
	const uint32_t p_base = (uint32_t)__cvta_generic_to_shared(p_smem);
	uint32_t *p_u32 = (uint32_t *)p_smem;
	const uint32_t local_h = blockIdx.x, q_block = blockIdx.y;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u, q_row_lo = lane >> 2u, q_row_hi = q_row_lo + 8u, quad = lane & 3u;
	const uint32_t row_first = q_block * Q_TILE;
	const uint32_t row_end = row_first + Q_TILE < rows ? row_first + Q_TILE : rows;
	const uint32_t lkvh = ((q_first + local_h) / group) - kv_first;
	const uint32_t total_kv = prior_keys + rows;
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	LmDependentWait();
	if ( row_first >= rows )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t max_pos = __ldg(positions + row_end - 1u);
	const uint32_t max_tile_end = (max_pos + KV_TILE) / KV_TILE;
	const uint32_t tile_begin = CHUNKED ? blockIdx.z * TILES_PER_CHUNK : 0u;
	const uint32_t tile_end = CHUNKED ? (tile_begin + TILES_PER_CHUNK < max_tile_end ? tile_begin + TILES_PER_CHUNK : max_tile_end) : max_tile_end;
	if constexpr ( CHUNKED )
	{
		if ( tile_begin >= tile_end )
		{
			LmDependentRelease();
			return;
		}
	}
	for ( uint32_t idx = threadIdx.x; idx < Q_TILE * K_CHUNKS; idx += THREADS )
	{
		const uint32_t r = idx / K_CHUNKS, c = idx % K_CHUNKS;
		const bool vr = row_first + r < row_end;
		const uint16_t *src = vr ? query_bf16 + ((uint64_t)(row_first + r) * rank_heads + local_h) * HD + c * 8u : query_bf16;
		LmPrefillCopy(k_base + r * K_ROW_BYTES + LmPrefillSwizzle(r, c) * 16u, src, vr ? 16u : 0u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	uint32_t q_frag[K_STEPS][4];
	{
		const uint32_t a_row = (matrix & 1u) * 8u + within;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < K_STEPS; ++ks )
			LmPrefillLoadX4(k_base + a_row * K_ROW_BYTES + LmPrefillSwizzle(a_row, 2u * ks + (matrix >> 1u)) * 16u, q_frag[ks]);
	}
	__syncthreads();
	const uint32_t row_lo = row_first + q_row_lo, row_hi = row_first + q_row_hi;
	const bool valid_row[2] = { row_lo < row_end, row_hi < row_end };
	const uint32_t rpos[2] = { valid_row[0] ? __ldg(positions + row_lo) : 0u, valid_row[1] ? __ldg(positions + row_hi) : 0u };
	auto copy_kv = [&](uint32_t tile, uint32_t dst_base, uint32_t elem_off)
	{
		for ( uint32_t idx = threadIdx.x; idx < KV_TILE * K_CHUNKS; idx += THREADS )
		{
			const uint32_t s = idx / K_CHUNKS, c = idx % K_CHUNKS;
			const uint32_t k_idx = tile * KV_TILE + s;
			const bool valid = k_idx < total_kv;
			const uint16_t *src = valid ?
				(k_idx < prior_keys ?
					kv_prior + ((uint64_t)lkvh * prior_keys + k_idx) * SLOT_ELEMS + elem_off + c * 8u :
					kv_cur + ((uint64_t)lkvh * rows + (k_idx - prior_keys)) * SLOT_ELEMS + elem_off + c * 8u) :
				query_bf16;
			LmPrefillCopy(dst_base + s * K_ROW_BYTES + LmPrefillSwizzle(s, c) * 16u, src, valid ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	float output[HALF_V_FRAGS][4];
	#pragma unroll
	for ( uint32_t f = 0u; f < HALF_V_FRAGS; ++f )
		output[f][0] = output[f][1] = output[f][2] = output[f][3] = 0.0f;
	float running_max[2] = { -INFINITY, -INFINITY };
	float running_sum[2] = { 0.0f, 0.0f };
	if ( tile_begin < tile_end )
		copy_kv(tile_begin, k_base, 0u);
	for ( uint32_t tile = tile_begin; tile < tile_end; ++tile )
	{
		copy_kv(tile, v_base, HD);
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		__syncthreads();
		float score[HALF_K_FRAGS][4];
		#pragma unroll
		for ( uint32_t f = 0u; f < HALF_K_FRAGS; ++f )
			score[f][0] = score[f][1] = score[f][2] = score[f][3] = 0.0f;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < K_STEPS; ++ks )
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < HALF_K_FRAGS; f += 2u )
			{
				uint32_t b[4], first[2], second[2];
				const uint32_t b_row = (warp * HALF_K_FRAGS + f + (matrix >> 1u)) * 8u + within;
				const uint32_t b_chunk = 2u * ks + (matrix & 1u);
				LmPrefillLoadX4(k_base + b_row * K_ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(score[f], q_frag[ks], first);
				LmMmaBf16(score[f + 1u], q_frag[ks], second);
			}
		}
		const uint32_t tile_pos0 = tile * KV_TILE + warp * 32u;
		const uint32_t tile_pos1 = tile_pos0 + 31u;
		const bool whole = __all_sync(0xffffffffu, valid_row[0] && valid_row[1] && tile_pos1 <= rpos[0] && tile_pos1 <= rpos[1]);
		float peak[2] = { -INFINITY, -INFINITY };
		if ( whole )
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < HALF_K_FRAGS; ++f )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					peak[e >> 1u] = fmaxf(peak[e >> 1u], score[f][e]);
		}
		else
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < HALF_K_FRAGS; ++f )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
				{
					const uint32_t key_pos = tile_pos0 + f * 8u + 2u * quad + (e & 1u);
					const uint32_t side = e >> 1u;
					const bool vis = valid_row[side] && key_pos <= rpos[side];
					score[f][e] = vis ? score[f][e] : -INFINITY;
					peak[side] = fmaxf(peak[side], score[f][e]);
				}
		}
		#pragma unroll
		for ( uint32_t side = 0u; side < 2u; ++side )
		{
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 1u));
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 2u));
		}
		if ( quad == 0u )
		{
			s_peak[warp][q_row_lo] = peak[0];
			s_peak[warp][q_row_hi] = peak[1];
		}
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		if ( tile + 1u < tile_end )
			copy_kv(tile + 1u, k_base, 0u);
		const float tile_peak[2] = {
			fmaxf(s_peak[0][q_row_lo], s_peak[1][q_row_lo]),
			fmaxf(s_peak[0][q_row_hi], s_peak[1][q_row_hi])
		};
		float rescale[2], shift[2];
		#pragma unroll
		for ( uint32_t side = 0u; side < 2u; ++side )
		{
			const float next_max = fmaxf(running_max[side], tile_peak[side]);
			rescale[side] = running_max[side] == -INFINITY ? 0.0f : SparkMathExp2Fast((running_max[side] - next_max) * scale_log2);
			running_max[side] = next_max;
			shift[side] = next_max == -INFINITY ? 0.0f : next_max * scale_log2;
		}
		if ( rescale[0] != 1.0f || rescale[1] != 1.0f )
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < HALF_V_FRAGS; ++f )
			{
				output[f][0] *= rescale[0];
				output[f][1] *= rescale[0];
				output[f][2] *= rescale[1];
				output[f][3] *= rescale[1];
			}
		}
		float lsum[2] = { 0.0f, 0.0f };
		#pragma unroll
		for ( uint32_t f = 0u; f < HALF_K_FRAGS; ++f )
		{
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
			{
				const uint32_t side = e >> 1u;
				const float wt = (running_max[side] == -INFINITY || score[f][e] == -INFINITY) ? 0.0f :
					SparkMathExp2Fast(fmaf(score[f][e], scale_log2, -shift[side]));
				lsum[side] += wt;
				score[f][e] = wt;
			}
			const uint32_t chunk = warp * HALF_K_FRAGS + f;
			p_u32[(q_row_lo * 8u + LmPrefillSwizzle(q_row_lo, chunk)) * 4u + quad] = LmPackBf16Pair(score[f][0], score[f][1]);
			p_u32[(q_row_hi * 8u + LmPrefillSwizzle(q_row_hi, chunk)) * 4u + quad] = LmPackBf16Pair(score[f][2], score[f][3]);
		}
		#pragma unroll
		for ( uint32_t side = 0u; side < 2u; ++side )
		{
			lsum[side] += __shfl_xor_sync(0xffffffffu, lsum[side], 1u);
			lsum[side] += __shfl_xor_sync(0xffffffffu, lsum[side], 2u);
		}
		if ( quad == 0u )
		{
			s_sum[warp][q_row_lo] = lsum[0];
			s_sum[warp][q_row_hi] = lsum[1];
		}
		__syncthreads();
		running_sum[0] = fmaf(running_sum[0], rescale[0], s_sum[0][q_row_lo] + s_sum[1][q_row_lo]);
		running_sum[1] = fmaf(running_sum[1], rescale[1], s_sum[0][q_row_hi] + s_sum[1][q_row_hi]);
		#pragma unroll
		for ( uint32_t step = 0u; step < P_STEPS; ++step )
		{
			uint32_t weights[4];
			const uint32_t a_row = (matrix & 1u) * 8u + within;
			const uint32_t a_chunk = 2u * step + (matrix >> 1u);
			LmPrefillLoadX4(p_base + a_row * P_ROW_BYTES + LmPrefillSwizzle(a_row, a_chunk) * 16u, weights);
			#pragma unroll
			for ( uint32_t f = 0u; f < HALF_V_FRAGS; f += 2u )
			{
				uint32_t b[4], first[2], second[2];
				const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
				const uint32_t b_chunk = warp * HALF_V_FRAGS + f + (matrix >> 1u);
				LmPrefillLoadX4Trans(v_base + b_row * K_ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(output[f], weights, first);
				LmMmaBf16(output[f + 1u], weights, second);
			}
		}
		if ( tile + 1u < tile_end )
			__syncthreads();
	}
	if constexpr ( CHUNKED )
	{
		const uint64_t chunk_off = (uint64_t)blockIdx.z * rows * rank_heads * (HD + 2u);
		if ( valid_row[0] )
		{
			float *dst = partials + chunk_off + ((uint64_t)row_lo * rank_heads + local_h) * (HD + 2u);
			if ( warp == 0u && quad == 0u )
			{
				dst[0] = running_max[0];
				dst[1] = running_sum[0];
			}
			#pragma unroll
			for ( uint32_t f = 0u; f < HALF_V_FRAGS; ++f )
			{
				const uint32_t col = (warp * HALF_V_FRAGS + f) * 8u + quad * 2u;
				*(float2 *)(dst + 2u + col) = make_float2(output[f][0], output[f][1]);
			}
		}
		if ( valid_row[1] )
		{
			float *dst = partials + chunk_off + ((uint64_t)row_hi * rank_heads + local_h) * (HD + 2u);
			if ( warp == 0u && quad == 0u )
			{
				dst[0] = running_max[1];
				dst[1] = running_sum[1];
			}
			#pragma unroll
			for ( uint32_t f = 0u; f < HALF_V_FRAGS; ++f )
			{
				const uint32_t col = (warp * HALF_V_FRAGS + f) * 8u + quad * 2u;
				*(float2 *)(dst + 2u + col) = make_float2(output[f][2], output[f][3]);
			}
		}
	}
	else
	{
		const float inv_scale[2] = {
			valid_row[0] && running_sum[0] > 0.0f ? (1.0f / running_sum[0]) : 0.0f,
			valid_row[1] && running_sum[1] > 0.0f ? (1.0f / running_sum[1]) : 0.0f
		};
		#pragma unroll
		for ( uint32_t f = 0u; f < HALF_V_FRAGS; ++f )
		{
			const uint32_t col = (warp * HALF_V_FRAGS + f) * 8u + quad * 2u;
			if ( valid_row[0] )
			{
				const uint64_t idx = ((uint64_t)row_lo * rank_heads + local_h) * HD + col;
				const uint32_t gp = *(const uint32_t *)(gate_bf16 + idx);
				const float o0 = SparkMathRoundBf16(output[f][0] * inv_scale[0]) * SparkMathRoundBf16(SparkMathSigmoid(__uint_as_float(gp << 16u)));
				const float o1 = SparkMathRoundBf16(output[f][1] * inv_scale[0]) * SparkMathRoundBf16(SparkMathSigmoid(__uint_as_float(gp & 0xffff0000u)));
				*(uint32_t *)(merged_out + idx) = LmPackBf16Pair(o0, o1);
			}
			if ( valid_row[1] )
			{
				const uint64_t idx = ((uint64_t)row_hi * rank_heads + local_h) * HD + col;
				const uint32_t gp = *(const uint32_t *)(gate_bf16 + idx);
				const float o0 = SparkMathRoundBf16(output[f][2] * inv_scale[1]) * SparkMathRoundBf16(SparkMathSigmoid(__uint_as_float(gp << 16u)));
				const float o1 = SparkMathRoundBf16(output[f][3] * inv_scale[1]) * SparkMathRoundBf16(SparkMathSigmoid(__uint_as_float(gp & 0xffff0000u)));
				*(uint32_t *)(merged_out + idx) = LmPackBf16Pair(o0, o1);
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t HD>
__global__ __launch_bounds__(HD, 4)
static void LingGqaLocalPrefillMergeGateKernel(const float *__restrict__ partials, const uint32_t *__restrict__ positions,
	const uint16_t *__restrict__ gate_bf16, uint32_t rows, uint32_t rank_heads, uint32_t num_chunks, float qk_scale,
	uint16_t *__restrict__ merged_out)
{
	const uint32_t local_h = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
	const uint64_t rec = (uint64_t)(HD + 2u);
	const uint64_t chunk_stride = (uint64_t)rows * rank_heads * rec;
	const uint64_t out_idx = ((uint64_t)row * rank_heads + local_h) * HD + d;
	LmDependentWait();
	if ( row < rows && local_h < rank_heads )
	{
		const uint32_t rpos = __ldg(positions + row);
		uint32_t c_end = (rpos / 512u) + 1u;
		if ( c_end > num_chunks )
			c_end = num_chunks;
		const float *p0 = partials + ((uint64_t)row * rank_heads + local_h) * rec;
		const float g = SparkMathRoundBf16(SparkMathSigmoid(LmBf16ToFloat(__ldg(gate_bf16 + out_idx))));
		if ( c_end == 1u )
		{
			const float sum = __ldg(p0 + 1u);
			const float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
			merged_out[out_idx] = LmFloatToBf16(SparkMathRoundBf16(__ldg(p0 + 2u + d) * inv) * g);
		}
		else
		{
			const float scale_log2 = qk_scale * 1.4426950408889634f;
			float gmax = -INFINITY;
			for ( uint32_t c = 0u; c < c_end; ++c )
				gmax = fmaxf(gmax, __ldg(p0 + (uint64_t)c * chunk_stride));
			float acc_sum = 0.0f, acc_val = 0.0f;
			if ( gmax > -INFINITY )
			{
				for ( uint32_t c = 0u; c < c_end; ++c )
				{
					const float *pc = p0 + (uint64_t)c * chunk_stride;
					const float cm = __ldg(pc);
					if ( cm > -INFINITY )
					{
						const float sc = SparkMathExp2Fast((cm - gmax) * scale_log2);
						acc_sum = fmaf(__ldg(pc + 1u), sc, acc_sum);
						acc_val = fmaf(__ldg(pc + 2u + d), sc, acc_val);
					}
				}
			}
			const float inv = acc_sum > 0.0f ? (1.0f / acc_sum) : 0.0f;
			merged_out[out_idx] = LmFloatToBf16(SparkMathRoundBf16(acc_val * inv) * g);
		}
	}
	LmDependentRelease();
}

template<class G, class Format, uint32_t THREADS, uint32_t LANES, uint32_t ACT_STEP_VECS, uint32_t R_CAP>
static __device__ __forceinline__ void LingMoeExactW1W3GroupPass(const uint4 *w1_0, const uint4 *w3_0,
	const float *s1_0, const float *s3_0, const uint16_t *activation, const float *packed_weight,
	const uint32_t *s_src, uint4 (*s_act)[ACT_STEP_VECS], uint16_t *output, uint32_t r0, uint32_t count, uint32_t sub,
	uint32_t col0, bool valid_col, bool has1, float limit)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t K = G::kHidden, N = G::kMoeInter, CHUNKS = K / ELEMS, SC_STRIDE = K / LG_FP8_BLOCK;
	constexpr uint32_t VECS_PER_CHUNK = IS_FP8 ? 2u : 1u;
	const uint32_t w_step = has1 ? CHUNKS : 0u;
	const uint32_t s_step = has1 ? SC_STRIDE : 0u;
	float g0[R_CAP], u0[R_CAP], g1[R_CAP], u1[R_CAP];
	#pragma unroll
	for ( uint32_t r = 0u; r < R_CAP; ++r )
		g0[r] = u0[r] = g1[r] = u1[r] = 0.0f;
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
		if ( c < CHUNKS )
		{
			if constexpr ( IS_FP8 )
			{
				const uint32_t sg = c >> 3u;
				auto accum_fp8 = [&](const uint4 *wp, const float *sp, float *acc)
				{
					const float sc = __ldg(sp + sg);
					const typename LmSkinnyFormat<LmFp8>::Decoded d = LmSkinnyFormat<LmFp8>::Decode(__ldcs(wp + c));
					#pragma unroll
					for ( uint32_t r = 0u; r < R_CAP; ++r )
						if ( r < count )
						{
							LmSkinnyWide a{ s_act[r][2u * sub], s_act[r][2u * sub + 1u] };
							acc[r] = fmaf(sc, LmSkinnyFormat<LmFp8>::ChunkDecoded(d, a), acc[r]);
						}
				};
				accum_fp8(w1_0, s1_0, g0);
				accum_fp8(w3_0, s3_0, u0);
				accum_fp8(w1_0 + w_step, s1_0 + s_step, g1);
				accum_fp8(w3_0 + w_step, s3_0 + s_step, u1);
			}
			else
			{
				float w1f0[8], w3f0[8], w1f1[8], w3f1[8];
				LmSkinnyUnpackBf16x8(__ldcs(w1_0 + c), w1f0);
				LmSkinnyUnpackBf16x8(__ldcs(w3_0 + c), w3f0);
				LmSkinnyUnpackBf16x8(__ldcs(w1_0 + w_step + c), w1f1);
				LmSkinnyUnpackBf16x8(__ldcs(w3_0 + w_step + c), w3f1);
				#pragma unroll
				for ( uint32_t r = 0u; r < R_CAP; ++r )
					if ( r < count )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(s_act[r][sub], af);
						g0[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w1f0, af), g0[r]);
						u0[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w3f0, af), u0[r]);
						g1[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w1f1, af), g1[r]);
						u1[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w3f1, af), u1[r]);
					}
			}
		}
		if ( c_base + LANES < CHUNKS )
			__syncthreads();
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < R_CAP; ++r )
		if ( r < count )
		{
			#pragma unroll
			for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
			{
				g0[r] += __shfl_down_sync(0xffffffffu, g0[r], off);
				u0[r] += __shfl_down_sync(0xffffffffu, u0[r], off);
				g1[r] += __shfl_down_sync(0xffffffffu, g1[r], off);
				u1[r] += __shfl_down_sync(0xffffffffu, u1[r], off);
			}
			if ( sub == 0u && valid_col )
			{
				const uint32_t pr = r0 + r;
				const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
					!G::kExpertWeightAfterDown ? __ldg(packed_weight + pr) : 1.0f, limit };
				const float v0 = op(SparkMathRoundBf16(g0[r]), SparkMathRoundBf16(u0[r]));
				if ( has1 )
				{
					const float v1 = op(SparkMathRoundBf16(g1[r]), SparkMathRoundBf16(u1[r]));
					*(uint32_t *)(output + (uint64_t)pr * N + col0) = LmPackBf16Pair(v0, v1);
				}
				else
					output[(uint64_t)pr * N + col0] = LmFloatToBf16(v0);
			}
		}
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
	constexpr uint32_t K = G::kHidden, N = G::kMoeInter, CHUNKS = K / ELEMS, SC_STRIDE = K / LG_FP8_BLOCK;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = THREADS / LANES, MAX_R = IS_FP8 ? 4u : 8u;
	constexpr uint32_t VECS_PER_CHUNK = IS_FP8 ? 2u : 1u, ACT_STEP_VECS = LANES * VECS_PER_CHUNK;
	__shared__ __align__(16) uint4 s_act[MAX_R][ACT_STEP_VECS];
	__shared__ uint32_t s_src[MAX_R];
	const uint32_t act_idx = blockIdx.y;
	LmDependentWait();
	if ( act_idx < __ldg(active_experts) )
	{
		const uint32_t exp = __ldg(active_experts + 1u + act_idx);
		const uint32_t r_begin = __ldg(group_row_offset + exp), r_end = __ldg(group_row_offset + exp + 1u);
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
		const bool valid_col = col0 < N, has1 = col0 + 1u < N;
		const uint64_t row_base = (uint64_t)exp * N + (valid_col ? col0 : 0u);
		const uint4 *w1_0 = (const uint4 *)w1 + row_base * CHUNKS, *w3_0 = (const uint4 *)w3 + row_base * CHUNKS;
		const float *s1_0 = IS_FP8 ? s1 + row_base * SC_STRIDE : (const float *)0;
		const float *s3_0 = IS_FP8 ? s3 + row_base * SC_STRIDE : (const float *)0;
		const uint32_t w_step = has1 ? CHUNKS : 0u, s_step = has1 ? SC_STRIDE : 0u;
		for ( uint32_t r0 = r_begin; r0 < r_end; r0 += MAX_R )
		{
			const uint32_t count = r_end - r0 < MAX_R ? r_end - r0 : MAX_R;
			if ( threadIdx.x < count )
				s_src[threadIdx.x] = __ldg(route_source_token + r0 + threadIdx.x);
			__syncthreads();
			if ( count == 1u )
			{
				const uint16_t *a_row = activation + (uint64_t)s_src[0] * K;
				float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const typename LmSkinnyFormat<Format>::Activation a = LmSkinnyFormat<Format>::Load(a_row + c * ELEMS);
					const uint32_t sg = c >> 3u;
					g0 = fmaf(IS_FP8 ? __ldg(s1_0 + sg) : 1.0f, LmSkinnyFormat<Format>::Chunk(__ldcs(w1_0 + c), a), g0);
					u0 = fmaf(IS_FP8 ? __ldg(s3_0 + sg) : 1.0f, LmSkinnyFormat<Format>::Chunk(__ldcs(w3_0 + c), a), u0);
					g1 = fmaf(IS_FP8 ? __ldg(s1_0 + s_step + sg) : 1.0f, LmSkinnyFormat<Format>::Chunk(__ldcs(w1_0 + w_step + c), a), g1);
					u1 = fmaf(IS_FP8 ? __ldg(s3_0 + s_step + sg) : 1.0f, LmSkinnyFormat<Format>::Chunk(__ldcs(w3_0 + w_step + c), a), u1);
				}
				#pragma unroll
				for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
				{
					g0 += __shfl_down_sync(0xffffffffu, g0, off);
					u0 += __shfl_down_sync(0xffffffffu, u0, off);
					g1 += __shfl_down_sync(0xffffffffu, g1, off);
					u1 += __shfl_down_sync(0xffffffffu, u1, off);
				}
				if ( sub == 0u && valid_col )
				{
					const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
						!G::kExpertWeightAfterDown ? __ldg(packed_weight + r0) : 1.0f, limit };
					const float v0 = op(SparkMathRoundBf16(g0), SparkMathRoundBf16(u0));
					if ( has1 )
					{
						const float v1 = op(SparkMathRoundBf16(g1), SparkMathRoundBf16(u1));
						*(uint32_t *)(output + (uint64_t)r0 * N + col0) = LmPackBf16Pair(v0, v1);
					}
					else
						output[(uint64_t)r0 * N + col0] = LmFloatToBf16(v0);
				}
			}
			else if constexpr ( IS_FP8 )
				LingMoeExactW1W3GroupPass<G, Format, THREADS, LANES, ACT_STEP_VECS, 4u>(w1_0, w3_0, s1_0, s3_0,
					activation, packed_weight, s_src, s_act, output, r0, count, sub, col0, valid_col, has1, limit);
			else if ( count <= 4u )
				LingMoeExactW1W3GroupPass<G, Format, THREADS, LANES, ACT_STEP_VECS, 4u>(w1_0, w3_0, s1_0, s3_0,
					activation, packed_weight, s_src, s_act, output, r0, count, sub, col0, valid_col, has1, limit);
			else
				LingMoeExactW1W3GroupPass<G, Format, THREADS, LANES, ACT_STEP_VECS, MAX_R>(w1_0, w3_0, s1_0, s3_0,
					activation, packed_weight, s_src, s_act, output, r0, count, sub, col0, valid_col, has1, limit);
			if ( r0 + MAX_R < r_end )
				__syncthreads();
		}
	}
	LmDependentRelease();
}

template<class G, class Format, uint32_t LANES, uint32_t ACT_VECS, uint32_t R_CAP>
static __device__ __forceinline__ void LingMoeExactW2GroupPass(const uint4 *w0, const float *s0,
	const uint4 (*s_act)[ACT_VECS], uint16_t *output, uint32_t r0, uint32_t count, uint32_t sub, uint32_t col0, bool valid_col, bool has1)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t K = G::kMoeInter, N = G::kHidden, CHUNKS = K / ELEMS, SC_STRIDE = K / LG_FP8_BLOCK;
	const uint32_t w_step = has1 ? CHUNKS : 0u, s_step = has1 ? SC_STRIDE : 0u;
	float a0[R_CAP], a1[R_CAP];
	#pragma unroll
	for ( uint32_t r = 0u; r < R_CAP; ++r )
		a0[r] = a1[r] = 0.0f;
	for ( uint32_t c = sub; c < CHUNKS; c += LANES )
	{
		if constexpr ( IS_FP8 )
		{
			const uint32_t sg = c >> 3u;
			auto accum_fp8 = [&](const uint4 *wp, const float *sp, float *acc)
			{
				const float sc = __ldg(sp + sg);
				const typename LmSkinnyFormat<LmFp8>::Decoded d = LmSkinnyFormat<LmFp8>::Decode(__ldcs(wp + c));
				#pragma unroll
				for ( uint32_t r = 0u; r < R_CAP; ++r )
					if ( r < count )
					{
						LmSkinnyWide a{ s_act[r][2u * c], s_act[r][2u * c + 1u] };
						acc[r] = fmaf(sc, LmSkinnyFormat<LmFp8>::ChunkDecoded(d, a), acc[r]);
					}
			};
			accum_fp8(w0, s0, a0);
			accum_fp8(w0 + w_step, s0 + s_step, a1);
		}
		else
		{
			float w0f[8], w1f[8];
			LmSkinnyUnpackBf16x8(__ldcs(w0 + c), w0f);
			LmSkinnyUnpackBf16x8(__ldcs(w0 + w_step + c), w1f);
			#pragma unroll
			for ( uint32_t r = 0u; r < R_CAP; ++r )
				if ( r < count )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_act[r][c], af);
					a0[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w0f, af), a0[r]);
					a1[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(w1f, af), a1[r]);
				}
		}
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < R_CAP; ++r )
		if ( r < count )
		{
			#pragma unroll
			for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
			{
				a0[r] += __shfl_down_sync(0xffffffffu, a0[r], off);
				a1[r] += __shfl_down_sync(0xffffffffu, a1[r], off);
			}
			if ( sub == 0u && valid_col )
			{
				if ( has1 )
					*(uint32_t *)(output + (uint64_t)(r0 + r) * N + col0) = LmPackBf16Pair(a0[r], a1[r]);
				else
					output[(uint64_t)(r0 + r) * N + col0] = LmFloatToBf16(a0[r]);
			}
		}
}

template<class G, class Format>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMoeExactW2Kernel(const void *__restrict__ w2, const float *__restrict__ s2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ active_experts, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output)
{
	constexpr bool IS_FP8 = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t K = G::kMoeInter, N = G::kHidden, CHUNKS = K / ELEMS, SC_STRIDE = K / LG_FP8_BLOCK;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = LG_THREADS / LANES, MAX_R = IS_FP8 ? 4u : 8u, ACT_VECS = K >> 3u;
	constexpr bool HOIST_W = !IS_FP8 && CHUNKS <= 2u * LANES;
	__shared__ __align__(16) uint4 s_act[MAX_R][ACT_VECS];
	const uint32_t act_idx = blockIdx.y;
	LmDependentWait();
	if ( act_idx < __ldg(active_experts) )
	{
		const uint32_t exp = __ldg(active_experts + 1u + act_idx);
		const uint32_t r_begin = __ldg(group_row_offset + exp), r_end = __ldg(group_row_offset + exp + 1u);
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
		const bool valid_col = col0 < N, has1 = col0 + 1u < N;
		const uint64_t row_base = (uint64_t)exp * N + (valid_col ? col0 : 0u);
		const uint4 *w0 = (const uint4 *)w2 + row_base * CHUNKS;
		const float *s0 = IS_FP8 ? s2 + row_base * SC_STRIDE : (const float *)0;
		const uint32_t w_step = has1 ? CHUNKS : 0u, s_step = has1 ? SC_STRIDE : 0u;
		if constexpr ( HOIST_W )
		{
			const uint4 hw0_0 = sub < CHUNKS ? __ldcs(w0 + sub) : make_uint4(0u, 0u, 0u, 0u);
			const uint4 hw1_0 = sub + LANES < CHUNKS ? __ldcs(w0 + sub + LANES) : make_uint4(0u, 0u, 0u, 0u);
			const uint4 hw0_1 = sub < CHUNKS ? __ldcs(w0 + w_step + sub) : make_uint4(0u, 0u, 0u, 0u);
			const uint4 hw1_1 = sub + LANES < CHUNKS ? __ldcs(w0 + w_step + sub + LANES) : make_uint4(0u, 0u, 0u, 0u);
			for ( uint32_t r0 = r_begin; r0 < r_end; r0 += MAX_R )
			{
				const uint32_t count = r_end - r0 < MAX_R ? r_end - r0 : MAX_R;
				const uint32_t total_v = count * ACT_VECS;
				for ( uint32_t idx = threadIdx.x; idx < total_v; idx += LG_THREADS )
					((uint4 *)s_act)[idx] = __ldg((const uint4 *)(expert_act + (uint64_t)r0 * K) + idx);
				__syncthreads();
				#pragma unroll
				for ( uint32_t r = 0u; r < MAX_R; ++r )
					if ( r < count )
					{
						float acc0 = 0.0f, acc1 = 0.0f;
						if ( sub < CHUNKS )
						{
							const uint4 va = s_act[r][sub];
							acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(hw0_0, va, 0.0f), acc0);
							acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(hw0_1, va, 0.0f), acc1);
						}
						if ( sub + LANES < CHUNKS )
						{
							const uint4 va = s_act[r][sub + LANES];
							acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(hw1_0, va, 0.0f), acc0);
							acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(hw1_1, va, 0.0f), acc1);
						}
						#pragma unroll
						for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
						{
							acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
							acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
						}
						if ( sub == 0u && valid_col )
						{
							if ( has1 )
								*(uint32_t *)(output + (uint64_t)(r0 + r) * N + col0) = LmPackBf16Pair(acc0, acc1);
							else
								output[(uint64_t)(r0 + r) * N + col0] = LmFloatToBf16(acc0);
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
				if ( count == 1u )
				{
					float acc0 = 0.0f, acc1 = 0.0f;
					for ( uint32_t c = sub; c < CHUNKS; c += LANES )
					{
						if constexpr ( IS_FP8 )
						{
							LmSkinnyWide a{ s_act[0][2u * c], s_act[0][2u * c + 1u] };
							const uint32_t sg = c >> 3u;
							acc0 = fmaf(__ldg(s0 + sg), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w0 + c), a), acc0);
							acc1 = fmaf(__ldg(s0 + s_step + sg), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w0 + w_step + c), a), acc1);
						}
						else
						{
							const uint4 va = s_act[0][c];
							acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w0 + c), va, 0.0f), acc0);
							acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w0 + w_step + c), va, 0.0f), acc1);
						}
					}
					#pragma unroll
					for ( uint32_t off = LANES >> 1u; off > 0u; off >>= 1u )
					{
						acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
						acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
					}
					if ( sub == 0u && valid_col )
					{
						if ( has1 )
							*(uint32_t *)(output + (uint64_t)r0 * N + col0) = LmPackBf16Pair(acc0, acc1);
						else
							output[(uint64_t)r0 * N + col0] = LmFloatToBf16(acc0);
					}
				}
				else if constexpr ( IS_FP8 )
					LingMoeExactW2GroupPass<G, Format, LANES, ACT_VECS, 4u>(w0, s0, s_act, output, r0, count, sub, col0, valid_col, has1);
				else if ( count <= 4u )
					LingMoeExactW2GroupPass<G, Format, LANES, ACT_VECS, 4u>(w0, s0, s_act, output, r0, count, sub, col0, valid_col, has1);
				else
					LingMoeExactW2GroupPass<G, Format, LANES, ACT_VECS, MAX_R>(w0, s0, s_act, output, r0, count, sub, col0, valid_col, has1);
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
