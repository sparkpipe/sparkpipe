#include "inference/kernels/metal/common.metal"

#define SPARK_CONV_TAPS_MAX 8u
#define SPARK_DELTA_COLUMNS 16u
#define SPARK_DELTA_COLS_PER_SIMD 4u

static inline float2 SparkGroupSum2(float2 value, threadgroup float2 *scratch, uint lane, uint simd, uint simds)
{
	value = simd_sum(value);
	if ( lane == 0u )
		scratch[simd] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float2 total = lane < simds ? scratch[lane] : float2(0.0f);
	total = simd_sum(total);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return total;
}

kernel void spark_causal_conv_silu(constant SparkConvParams &p [[buffer(0)]], device ushort *window [[buffer(1)]],
	device const uint *state_index [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const ushort *input [[buffer(4)]],
	device const ushort *weight [[buffer(5)]], device ushort *output [[buffer(6)]], device const uint *parent_rows [[buffer(7)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint channel = group.x * threads + tid, sequence = group.y, taps_count = p.kernel_taps;
	const uint dilation = p.dilation != 0u ? p.dilation : 1u, span = SPARK_CONV_SPAN(taps_count, dilation);
	if ( channel >= p.channels || sequence >= p.sequences || taps_count > SPARK_CONV_TAPS_MAX || span > SPARK_CONV_SPAN_MAX )
		return;
	device ushort *slot = window + ((ulong)state_index[sequence] * p.channels + channel) * span;
	const uint begin = row_begin[sequence], end = row_begin[sequence + 1u];
	if ( p.tree == 0u && dilation == 1u && taps_count == 4u )
	{
		float4 t = ((ulong)slot & 7ul) == 0ul ? SparkUnpack4(*(device const uint2 *)slot) :
			float4(SparkMathBf16ToFloat(slot[0]), SparkMathBf16ToFloat(slot[1]), SparkMathBf16ToFloat(slot[2]), SparkMathBf16ToFloat(slot[3]));
		float4 w;
		if ( p.weight_f32 != 0u )
		{
			device const float *wf = (device const float *)weight + (ulong)channel * 4u;
			w = ((ulong)wf & 15ul) == 0ul ? *(device const float4 *)wf : float4(wf[0], wf[1], wf[2], wf[3]);
		}
		else
		{
			device const ushort *wb = weight + (ulong)channel * 4u;
			w = ((ulong)wb & 7ul) == 0ul ? SparkUnpack4(*(device const uint2 *)wb) :
				float4(SparkMathBf16ToFloat(wb[0]), SparkMathBf16ToFloat(wb[1]), SparkMathBf16ToFloat(wb[2]), SparkMathBf16ToFloat(wb[3]));
		}
		const bool round_sum = p.weight_f32 == 0u && p.swish_once == 0u;
		uint row = begin;
		for ( ; row + 3u < end; row += 4u )
		{
			const ulong r0 = (ulong)row * p.row_stride + channel;
			const ulong r1 = (ulong)(row + 1u) * p.row_stride + channel;
			const ulong r2 = (ulong)(row + 2u) * p.row_stride + channel;
			const ulong r3 = (ulong)(row + 3u) * p.row_stride + channel;
			const float n0 = SparkMathBf16ToFloat(input[r0]);
			const float n1 = SparkMathBf16ToFloat(input[r1]);
			const float n2 = SparkMathBf16ToFloat(input[r2]);
			const float n3 = SparkMathBf16ToFloat(input[r3]);
			float s0 = ((t.y * w.x + t.z * w.y) + t.w * w.z) + n0 * w.w;
			float s1 = ((t.z * w.x + t.w * w.y) + n0 * w.z) + n1 * w.w;
			float s2 = ((t.w * w.x + n0 * w.y) + n1 * w.z) + n2 * w.w;
			float s3 = ((n0 * w.x + n1 * w.y) + n2 * w.z) + n3 * w.w;
			if ( round_sum )
			{
				s0 = SparkMathRoundBf16(s0);
				s1 = SparkMathRoundBf16(s1);
				s2 = SparkMathRoundBf16(s2);
				s3 = SparkMathRoundBf16(s3);
			}
			output[r0] = SparkMathFloatToBf16(SparkMathSwishFast(s0));
			output[r1] = SparkMathFloatToBf16(SparkMathSwishFast(s1));
			output[r2] = SparkMathFloatToBf16(SparkMathSwishFast(s2));
			output[r3] = SparkMathFloatToBf16(SparkMathSwishFast(s3));
			t = float4(n0, n1, n2, n3);
		}
		for ( ; row < end; row++ )
		{
			const float next = SparkMathBf16ToFloat(input[(ulong)row * p.row_stride + channel]);
			t = float4(t.y, t.z, t.w, next);
			float total = ((t.x * w.x + t.y * w.y) + t.z * w.z) + t.w * w.w;
			if ( round_sum )
				total = SparkMathRoundBf16(total);
			output[(ulong)row * p.row_stride + channel] = SparkMathFloatToBf16(SparkMathSwishFast(total));
		}
		if ( p.commit != 0u )
		{
			if ( ((ulong)slot & 7ul) == 0ul )
				*(device uint2 *)slot = SparkPack4(t);
			else
			{
				slot[0] = SparkMathFloatToBf16(t.x);
				slot[1] = SparkMathFloatToBf16(t.y);
				slot[2] = SparkMathFloatToBf16(t.z);
				slot[3] = SparkMathFloatToBf16(t.w);
			}
		}
		return;
	}
	float taps[SPARK_CONV_SPAN_MAX];
	float weights[SPARK_CONV_TAPS_MAX];
	for ( uint entry = 0u; entry < span; entry++ )
		taps[entry] = SparkMathBf16ToFloat(slot[entry]);
	for ( uint tap = 0u; tap < taps_count; tap++ )
		weights[tap] = p.weight_f32 != 0u ? ((device const float *)weight)[channel * taps_count + tap] : SparkMathBf16ToFloat(weight[channel * taps_count + tap]);
	for ( uint row = begin; row < end; row++ )
	{
		float total = 0.0f;
		if ( p.tree != 0u )
		{
			uint ancestor = row, missing = span;
			for ( uint t = 0u; t < span; t++ )
			{
				if ( ancestor != 0xffffffffu )
				{
					taps[span - 1u - t] = SparkMathBf16ToFloat(input[(ulong)ancestor * p.row_stride + channel]);
					ancestor = parent_rows[ancestor];
				}
				else
				{
					missing = missing == span ? t : missing;
					taps[span - 1u - t] = SparkMathBf16ToFloat(slot[span - t + missing - 1u]);
				}
			}
		}
		else
		{
			for ( uint entry = 0u; entry + 1u < span; entry++ )
				taps[entry] = taps[entry + 1u];
			taps[span - 1u] = SparkMathBf16ToFloat(input[(ulong)row * p.row_stride + channel]);
		}
		for ( uint tap = 0u; tap < taps_count; tap++ )
			total += taps[tap * dilation] * weights[tap];
		if ( p.weight_f32 == 0u && p.swish_once == 0u )
			total = SparkMathRoundBf16(total);
		output[(ulong)row * p.row_stride + channel] = SparkMathFloatToBf16(SparkMathSwishFast(total));
	}
	if ( p.commit != 0u )
		for ( uint entry = 0u; entry < span; entry++ )
			slot[entry] = SparkMathFloatToBf16(taps[entry]);
}

static inline float SparkDeltaPrenormed(constant SparkDeltaParams &p, float value, float inverse)
{
	return p.prenorm_epsilon != 0.0f ? SparkMathRoundBf16(value * inverse) : value;
}

static inline void SparkDeltaNormalize(constant SparkDeltaParams &p, device const ushort *mix, device float *query, device float *key, uint head,
	uint row, uint tid, uint threads, threadgroup float2 *scratch, uint lane, uint simd, uint simds)
{
	const uint D = p.key_dimension;
	device const ushort *row_mix = mix + (ulong)row * p.mix_stride;
	device float *q_out = query + ((ulong)row * p.key_heads + head) * D;
	device float *k_out = key + ((ulong)row * p.key_heads + head) * D;
	if ( D <= threads )
	{
		float q = tid < D ? SparkMathBf16ToFloat(row_mix[p.query_offset + head * D + tid]) : 0.0f;
		float k = tid < D ? SparkMathBf16ToFloat(row_mix[p.key_offset + head * D + tid]) : 0.0f;
		if ( p.prenorm_epsilon != 0.0f )
		{
			const float2 s0 = SparkGroupSum2(float2(q * q, k * k), scratch, lane, simd, simds);
			const float q_first = rsqrt(s0.x + p.prenorm_epsilon), k_first = rsqrt(s0.y + p.prenorm_epsilon);
			if ( tid < D )
			{
				q = SparkMathRoundBf16(q * q_first);
				k = SparkMathRoundBf16(k * k_first);
			}
		}
		const float2 s1 = SparkGroupSum2(float2(q * q, k * k), scratch, lane, simd, simds);
		const float q_scale = rsqrt(s1.x + p.epsilon) * rsqrt(float(D)), k_scale = rsqrt(s1.y + p.epsilon);
		if ( tid < D )
		{
			q_out[tid] = q * q_scale;
			k_out[tid] = k * k_scale;
		}
		return;
	}
	float2 sums = float2(0.0f);
	float q_first = 1.0f, k_first = 1.0f;
	if ( p.prenorm_epsilon != 0.0f )
	{
		for ( uint i = tid; i < D; i += threads )
		{
			const float q = SparkMathBf16ToFloat(row_mix[p.query_offset + head * D + i]);
			const float k = SparkMathBf16ToFloat(row_mix[p.key_offset + head * D + i]);
			sums += float2(q * q, k * k);
		}
		sums = SparkGroupSum2(sums, scratch, lane, simd, simds);
		q_first = rsqrt(sums.x + p.prenorm_epsilon);
		k_first = rsqrt(sums.y + p.prenorm_epsilon);
		sums = float2(0.0f);
	}
	for ( uint i = tid; i < D; i += threads )
	{
		const float q = SparkDeltaPrenormed(p, SparkMathBf16ToFloat(row_mix[p.query_offset + head * D + i]), q_first);
		const float k = SparkDeltaPrenormed(p, SparkMathBf16ToFloat(row_mix[p.key_offset + head * D + i]), k_first);
		sums += float2(q * q, k * k);
	}
	sums = SparkGroupSum2(sums, scratch, lane, simd, simds);
	const float q_scale = rsqrt(sums.x + p.epsilon) * rsqrt(float(D)), k_scale = rsqrt(sums.y + p.epsilon);
	for ( uint i = tid; i < D; i += threads )
	{
		q_out[i] = SparkDeltaPrenormed(p, SparkMathBf16ToFloat(row_mix[p.query_offset + head * D + i]), q_first) * q_scale;
		k_out[i] = SparkDeltaPrenormed(p, SparkMathBf16ToFloat(row_mix[p.key_offset + head * D + i]), k_first) * k_scale;
	}
}

kernel void spark_gated_delta_prepare(constant SparkDeltaParams &p [[buffer(0)]], device const ushort *mix [[buffer(1)]],
	device const ushort *gates [[buffer(2)]], device const float *log_scale [[buffer(3)]], device const float *bias [[buffer(4)]],
	device float *query [[buffer(5)]], device float *key [[buffer(6)]], device float *decay [[buffer(7)]], device float *beta [[buffer(8)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float2 scratch[32];
	const uint head = group.x, row = group.y, per_key = p.value_heads / p.key_heads;
	SparkDeltaNormalize(p, mix, query, key, head, row, tid, group_size.x, scratch, lane, simd, simds);
	if ( tid < per_key )
	{
		const uint value_head = head * per_key + tid;
		device const ushort *row_gates = gates + (ulong)row * p.gate_stride;
		const float shifted = SparkMathBf16ToFloat(row_gates[p.decay_offset + value_head]) + bias[value_head];
		decay[(ulong)row * p.value_heads + value_head] = SparkMathGatedDeltaDecay(log_scale[value_head], shifted);
		beta[(ulong)row * p.value_heads + value_head] = SparkMathSigmoid(SparkMathBf16ToFloat(row_gates[p.beta_offset + value_head]));
	}
}

kernel void spark_channel_delta_prepare(constant SparkDeltaParams &p [[buffer(0)]], device const ushort *mix [[buffer(1)]],
	device const ushort *decay_logits [[buffer(2)]], device const ushort *beta_logits [[buffer(3)]], device const float *log_scale [[buffer(4)]],
	device const float *bias [[buffer(5)]], device float *query [[buffer(6)]], device float *key [[buffer(7)]], device float *decay [[buffer(8)]],
	device float *beta [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
	uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float2 scratch[32];
	const uint head = group.x, row = group.y, D = p.key_dimension;
	SparkDeltaNormalize(p, mix, query, key, head, row, tid, group_size.x, scratch, lane, simd, simds);
	const float ls = log_scale[head];
	for ( uint i = tid; i < D; i += group_size.x )
		decay[((ulong)row * p.key_heads + head) * D + i] = SparkMathBoundedDecay(SparkMathBf16ToFloat(decay_logits[(ulong)row * p.gate_stride + head * D + i]),
			bias[head * D + i], ls, p.minimum_log_decay);
	if ( tid == 0u )
	{
		const float strength = SparkMathSigmoidFast(SparkMathBf16ToFloat(beta_logits[(ulong)row * p.beta_stride + head]));
		const float rounded = SparkMathRoundBf16(strength);
		beta[(ulong)row * p.value_heads + head] = p.beta_mode == SPARK_DELTA_BETA_DOUBLED ? 2.0f * rounded :
			p.beta_mode == SPARK_DELTA_BETA_ROUNDED ? rounded : strength;
	}
}

static inline void SparkDeltaLoadState(device const float *base, uint col0, uint V, uint tid, uint lane, uint simd,
	threadgroup float st[128][17], thread float4 (&cell)[SPARK_DELTA_COLS_PER_SIMD])
{
	const uint cg = (tid & 3u) * 4u, r0 = tid >> 2u;
	const bool in_bounds = col0 + cg + 4u <= V;
	for ( uint r = r0; r < 128u; r += 32u )
	{
		float4 v = in_bounds ? *(device const float4 *)(base + (ulong)r * V + col0 + cg) : float4(0.0f);
		st[r][cg + 0u] = v.x;
		st[r][cg + 1u] = v.y;
		st[r][cg + 2u] = v.z;
		st[r][cg + 3u] = v.w;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const uint c0 = simd * SPARK_DELTA_COLS_PER_SIMD, k0 = lane * 4u;
	for ( uint c = 0u; c < SPARK_DELTA_COLS_PER_SIMD; c++ )
		cell[c] = float4(st[k0 + 0u][c0 + c], st[k0 + 1u][c0 + c], st[k0 + 2u][c0 + c], st[k0 + 3u][c0 + c]);
}

static inline void SparkDeltaStoreState(device float *base, uint col0, uint V, uint tid, uint lane, uint simd,
	threadgroup float st[128][17], thread const float4 (&cell)[SPARK_DELTA_COLS_PER_SIMD])
{
	const uint c0 = simd * SPARK_DELTA_COLS_PER_SIMD, k0 = lane * 4u;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint c = 0u; c < SPARK_DELTA_COLS_PER_SIMD; c++ )
	{
		st[k0 + 0u][c0 + c] = cell[c].x;
		st[k0 + 1u][c0 + c] = cell[c].y;
		st[k0 + 2u][c0 + c] = cell[c].z;
		st[k0 + 3u][c0 + c] = cell[c].w;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const uint cg = (tid & 3u) * 4u, r0 = tid >> 2u;
	if ( col0 + cg + 4u <= V )
		for ( uint r = r0; r < 128u; r += 32u )
			*(device float4 *)(base + (ulong)r * V + col0 + cg) = float4(st[r][cg + 0u], st[r][cg + 1u], st[r][cg + 2u], st[r][cg + 3u]);
}

kernel void spark_gated_delta_rule(constant SparkDeltaParams &p [[buffer(0)]], device float *state [[buffer(1)]],
	device const uint *state_index [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const float *query [[buffer(4)]],
	device const float *key [[buffer(5)]], device const float *decay [[buffer(6)]], device const float *beta [[buffer(7)]],
	device const ushort *mix [[buffer(8)]], device ushort *output [[buffer(9)]], device const uint *state_loads [[buffer(10)]],
	device const uint *state_saves [[buffer(11)]], device float *snapshots [[buffer(12)]],
	uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float st[128][17];
	const uint col0 = group.x * SPARK_DELTA_COLUMNS, head = group.y, sequence = group.z;
	const uint K = p.key_dimension, V = p.value_dimension, key_head = head / (p.value_heads / p.key_heads);
	if ( col0 >= V || sequence >= p.sequences || K != 128u )
		return;
	const uint simd_col = col0 + simd * SPARK_DELTA_COLS_PER_SIMD;
	const bool simd_valid = simd_col < V;
	device float *s_base = state + (((ulong)state_index[sequence] * p.value_heads + head) * K) * V;
	float4 cell[SPARK_DELTA_COLS_PER_SIMD];
	SparkDeltaLoadState(s_base, col0, V, tid, lane, simd, st, cell);
	const uint begin = row_begin[sequence], end = row_begin[sequence + 1u];
	for ( uint row = begin; row < end; row++ )
	{
		if ( p.tree != 0u && state_loads[row] != 0u )
		{
			device const float *snap = snapshots + (((ulong)(state_loads[row] - 1u) * p.value_heads + head) * K) * V;
			threadgroup_barrier(mem_flags::mem_threadgroup);
			SparkDeltaLoadState(snap, col0, V, tid, lane, simd, st, cell);
		}
		if ( simd_valid )
		{
			const ulong vector = ((ulong)row * p.key_heads + key_head) * K + lane * 4u;
			const float4 k = *(device const float4 *)(key + vector);
			const float4 q = *(device const float4 *)(query + vector);
			const float strength = beta[(ulong)row * p.value_heads + head];
			device const ushort *v_ptr = mix + (ulong)row * p.mix_stride + p.value_offset + head * V + simd_col;
			const float4 vals = ((ulong)v_ptr & 7ul) == 0ul ? SparkUnpack4(*(device const uint2 *)v_ptr) :
				float4(SparkMathBf16ToFloat(v_ptr[0]), SparkMathBf16ToFloat(v_ptr[1]), SparkMathBf16ToFloat(v_ptr[2]), SparkMathBf16ToFloat(v_ptr[3]));
			if ( p.channel_decay != 0u )
			{
				const float4 dc = *(device const float4 *)(decay + ((ulong)row * p.value_heads + head) * K + lane * 4u);
				cell[0] *= dc;
				cell[1] *= dc;
				cell[2] *= dc;
				cell[3] *= dc;
			}
			else
			{
				const float dc = decay[(ulong)row * p.value_heads + head];
				cell[0] *= dc;
				cell[1] *= dc;
				cell[2] *= dc;
				cell[3] *= dc;
			}
			const float p0 = simd_sum(dot(cell[0], k));
			const float p1 = simd_sum(dot(cell[1], k));
			const float p2 = simd_sum(dot(cell[2], k));
			const float p3 = simd_sum(dot(cell[3], k));
			cell[0] = fma(float4(strength * (vals.x - p0)), k, cell[0]);
			cell[1] = fma(float4(strength * (vals.y - p1)), k, cell[1]);
			cell[2] = fma(float4(strength * (vals.z - p2)), k, cell[2]);
			cell[3] = fma(float4(strength * (vals.w - p3)), k, cell[3]);
			const float r0 = simd_sum(dot(cell[0], q));
			const float r1 = simd_sum(dot(cell[1], q));
			const float r2 = simd_sum(dot(cell[2], q));
			const float r3 = simd_sum(dot(cell[3], q));
			if ( lane == 0u )
			{
				device ushort *dst = output + (ulong)row * p.output_stride + head * V + simd_col;
				if ( ((ulong)dst & 7ul) == 0ul )
					*(device uint2 *)dst = SparkPack4(float4(r0, r1, r2, r3));
				else
				{
					dst[0] = SparkMathFloatToBf16(r0);
					dst[1] = SparkMathFloatToBf16(r1);
					dst[2] = SparkMathFloatToBf16(r2);
					dst[3] = SparkMathFloatToBf16(r3);
				}
			}
		}
		if ( p.tree != 0u && state_saves[row] != 0u )
			for ( uint branch = 0u; branch < SPARK_DELTA_SNAPSHOT_MAX; branch++ )
				if ( ((state_saves[row] >> branch) & 1u) != 0u )
				{
					device float *snap = snapshots + (((ulong)branch * p.value_heads + head) * K) * V;
					SparkDeltaStoreState(snap, col0, V, tid, lane, simd, st, cell);
				}
	}
	if ( p.commit != 0u )
		SparkDeltaStoreState(s_base, col0, V, tid, lane, simd, st, cell);
}
