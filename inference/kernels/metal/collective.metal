#include "inference/kernels/metal/common.metal"

kernel void spark_rank_sum_bf16(constant SparkRankParams &p [[buffer(0)]], device const ushort *lanes [[buffer(1)]], device ushort *full [[buffer(2)]],
	uint index [[thread_position_in_grid]])
{
	if ( index >= p.elements )
		return;
	float total = 0.0f;
	for ( uint rank = 0u; rank < p.ranks; rank++ )
		total += SparkMathBf16ToFloat(lanes[(ulong)rank * p.elements + index]);
	full[index] = SparkMathFloatToBf16(total);
}

kernel void spark_rank_max_u64(constant SparkRankParams &p [[buffer(0)]], device const ulong *lanes [[buffer(1)]], device ulong *full [[buffer(2)]],
	uint index [[thread_position_in_grid]])
{
	if ( index >= p.elements )
		return;
	ulong best = lanes[index];
	for ( uint rank = 1u; rank < p.ranks; rank++ )
		best = max(best, lanes[(ulong)rank * p.elements + index]);
	full[index] = best;
}

kernel void spark_rank_rows(constant SparkRankParams &p [[buffer(0)]], device const ushort *gathered [[buffer(1)]], device ushort *rows [[buffer(2)]],
	uint2 index [[thread_position_in_grid]])
{
	const uint column = index.x, row = index.y;
	if ( column >= p.width * p.ranks || row >= p.rows )
		return;
	const uint rank = column / p.width, offset = column % p.width;
	rows[(ulong)row * p.width * p.ranks + column] = gathered[((ulong)rank * p.rows + row) * p.width + offset];
}
