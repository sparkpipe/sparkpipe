#include <metal_stdlib>
#include "inference/kernels/kernel_math.cuh"

using namespace metal;

static inline ushort SparkTestReferenceBf16(uint bits)
{
	if ( (bits & 0x7fffffffu) > 0x7f800000u )
		return(0x7fffu);
	return(ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static inline void SparkTestNote(device atomic_uint *counts, device atomic_uint *firsts, uint slot, uint bits)
{
	atomic_fetch_add_explicit(&counts[slot], 1u, memory_order_relaxed);
	atomic_fetch_min_explicit(&firsts[slot], bits, memory_order_relaxed);
}

kernel void spark_test_bf16_sweep(constant uint &unused [[buffer(0)]], device atomic_uint *counts [[buffer(1)]], device atomic_uint *firsts [[buffer(2)]],
	uint gid [[thread_position_in_grid]])
{
	for ( uint j = 0u; j < 256u; j++ )
	{
		const uint bits = gid * 256u + j;
		const float value = as_type<float>(bits);
		const ushort expected = SparkTestReferenceBf16(bits), shared = SparkMathFloatToBf16(value);
		const ushort hardware = as_type<ushort>(bfloat(value));
		const bool nan = (bits & 0x7fffffffu) > 0x7f800000u;
		if ( shared != expected )
			SparkTestNote(counts,firsts,0u,bits);
		if ( hardware != expected )
			SparkTestNote(counts,firsts,nan ? 2u : 1u,bits);
		if ( (nan ? ushort(0x7fffu) : hardware) != expected )
			SparkTestNote(counts,firsts,3u,bits);
	}
}
