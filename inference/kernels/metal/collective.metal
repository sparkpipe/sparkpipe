#include "inference/kernels/metal/common.metal"

kernel void spark_rank_sum_bf16(constant SparkRankParams &p [[buffer(0)]], device const ushort *lanes [[buffer(1)]], device ushort *full [[buffer(2)]],
	uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]])
{
	if ( (((ulong)lanes | (ulong)full) & 15ul) == 0ul && (p.elements & 7ul) == 0ul && (threads & 7u) == 0u )
	{
		const uint vec_threads = threads / 8u;
		if ( tid >= vec_threads )
			return;
		const ulong v = (ulong)group * vec_threads + tid, stride = p.elements / 8ul;
		if ( v >= stride )
			return;
		float4 sl = 0.0f, sh = 0.0f;
		for ( uint rank = 0u; rank < p.ranks; rank++ )
		{
			float4 xl, xh;
			SparkUnpack8(((device const uint4 *)lanes)[(ulong)rank * stride + v], xl, xh);
			sl += xl;
			sh += xh;
		}
		((device uint4 *)full)[v] = SparkPack8(sl, sh);
		return;
	}
	const ulong index = (ulong)group * threads + tid;
	if ( index >= p.elements )
		return;
	float total = 0.0f;
	for ( uint rank = 0u; rank < p.ranks; rank++ )
		total += SparkMathBf16ToFloat(lanes[(ulong)rank * p.elements + index]);
	full[index] = SparkMathFloatToBf16(total);
}

kernel void spark_rank_sum_f32(constant SparkRankParams &p [[buffer(0)]], device const float *lanes [[buffer(1)]], device ushort *full [[buffer(2)]],
	uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]])
{
	if ( ((ulong)lanes & 15ul) == 0ul && ((ulong)full & 7ul) == 0ul && (p.elements & 3ul) == 0ul && (threads & 3u) == 0u )
	{
		const uint vec_threads = threads / 4u;
		if ( tid >= vec_threads )
			return;
		const ulong v = (ulong)group * vec_threads + tid, stride = p.elements / 4ul;
		if ( v >= stride )
			return;
		float4 s = 0.0f;
		for ( uint rank = 0u; rank < p.ranks; rank++ )
			s += ((device const float4 *)lanes)[(ulong)rank * stride + v];
		((device uint2 *)full)[v] = SparkPack4(s);
		return;
	}
	const ulong index = (ulong)group * threads + tid;
	if ( index >= p.elements )
		return;
	float total = 0.0f;
	for ( uint rank = 0u; rank < p.ranks; rank++ )
		total += lanes[(ulong)rank * p.elements + index];
	full[index] = SparkMathFloatToBf16(total);
}

kernel void spark_rank_max_u64(constant SparkRankParams &p [[buffer(0)]], device const ulong *lanes [[buffer(1)]], device ulong *full [[buffer(2)]],
	uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]])
{
	if ( (((ulong)lanes | (ulong)full) & 15ul) == 0ul && (p.elements & 1ul) == 0ul && (threads & 1u) == 0u )
	{
		const uint vec_threads = threads / 2u;
		if ( tid >= vec_threads )
			return;
		const ulong v = (ulong)group * vec_threads + tid, stride = p.elements / 2ul;
		if ( v >= stride )
			return;
		ulong2 best = ((device const ulong2 *)lanes)[v];
		for ( uint rank = 1u; rank < p.ranks; rank++ )
			best = max(best, ((device const ulong2 *)lanes)[(ulong)rank * stride + v]);
		((device ulong2 *)full)[v] = best;
		return;
	}
	const ulong index = (ulong)group * threads + tid;
	if ( index >= p.elements )
		return;
	ulong best = lanes[index];
	for ( uint rank = 1u; rank < p.ranks; rank++ )
		best = max(best, lanes[(ulong)rank * p.elements + index]);
	full[index] = best;
}

kernel void spark_rank_rows(constant SparkRankParams &p [[buffer(0)]], device const ushort *gathered [[buffer(1)]], device ushort *rows [[buffer(2)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( (((ulong)gathered | (ulong)rows) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= wvec * p.ranks || row >= p.rows )
			return;
		const uint rank = v / wvec, offset = v % wvec;
		((device uint4 *)rows)[(ulong)row * wvec * p.ranks + v] = ((device const uint4 *)gathered)[((ulong)rank * p.rows + row) * wvec + offset];
		return;
	}
	const uint column = group.x * group_size.x + tid;
	if ( column >= p.width * p.ranks || row >= p.rows )
		return;
	const uint rank = column / p.width, offset = column % p.width;
	rows[(ulong)row * p.width * p.ranks + column] = gathered[((ulong)rank * p.rows + row) * p.width + offset];
}
