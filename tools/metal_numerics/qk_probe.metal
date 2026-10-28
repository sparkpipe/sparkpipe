#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <uint ROWS, uint D>
static inline void ProbeScores(device bfloat *q, device bfloat *k, device float *out)
{
	constexpr auto desc = matmul2d_descriptor(ROWS, 64, D, false, true, false);
	matmul2d<desc, execution_simdgroups<4>> op;
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> queries(q, dextents<int32_t, 2>(D, ROWS), array<int32_t, 2>{ 1, int(D) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> keys(k, dextents<int32_t, 2>(D, 64), array<int32_t, 2>{ 1, int(D) });
	auto scores = op.template get_destination_cooperative_tensor<decltype(queries), decltype(keys), float>();
	op.run(queries, keys, scores);
	for ( uint16_t i = 0; i < scores.get_capacity(); ++i )
		if ( scores.is_valid_element(i) )
		{
			const auto index = scores.get_multidimensional_index(i);
			out[index[1] * 64 + index[0]] = scores[i];
		}
}

kernel void probe_qk512(device bfloat *q [[buffer(1)]], device bfloat *k [[buffer(2)]], device float *out [[buffer(3)]])
{
	ProbeScores<16, 512>(q, k, out);
}

kernel void probe_qk256(device bfloat *q [[buffer(1)]], device bfloat *k [[buffer(2)]], device float *out [[buffer(3)]])
{
	ProbeScores<32, 256>(q, k, out);
}

template <uint ROWS, uint D, uint PARTS>
static inline void ProbeScoresSplit(device bfloat *q, device bfloat *k, device float *out)
{
	constexpr uint CHUNK = D / PARTS;
	constexpr auto desc = matmul2d_descriptor(ROWS, 64, CHUNK, false, true, false);
	matmul2d<desc, execution_simdgroups<4>> op;
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> first_queries(q, dextents<int32_t, 2>(CHUNK, ROWS), array<int32_t, 2>{ 1, int(D) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> first_keys(k, dextents<int32_t, 2>(CHUNK, 64), array<int32_t, 2>{ 1, int(D) });
	auto total = op.template get_destination_cooperative_tensor<decltype(first_queries), decltype(first_keys), float>();
	auto partial = op.template get_destination_cooperative_tensor<decltype(first_queries), decltype(first_keys), float>();
	float sums[PARTS][64];
	for ( uint part = 0; part < PARTS; part++ )
	{
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> queries(q + part * CHUNK, dextents<int32_t, 2>(CHUNK, ROWS), array<int32_t, 2>{ 1, int(D) });
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> keys(k + part * CHUNK, dextents<int32_t, 2>(CHUNK, 64), array<int32_t, 2>{ 1, int(D) });
		op.run(queries, keys, partial);
		for ( uint16_t i = 0; i < partial.get_capacity() && i < 64; ++i )
			sums[part][i] = partial.is_valid_element(i) ? partial[i] : 0.0f;
	}
	for ( uint width = PARTS / 2; width > 0; width /= 2 )
		for ( uint part = 0; part < width; part++ )
			for ( uint16_t i = 0; i < 64; ++i )
				sums[part][i] += sums[part + width][i];
	for ( uint16_t i = 0; i < total.get_capacity() && i < 64; ++i )
		if ( total.is_valid_element(i) )
		{
			const auto index = total.get_multidimensional_index(i);
			out[index[1] * 64 + index[0]] = sums[0][i];
		}
}

kernel void probe_qk512_split(device bfloat *q [[buffer(1)]], device bfloat *k [[buffer(2)]], device float *out [[buffer(3)]])
{
	ProbeScoresSplit<16, 512, 4>(q, k, out);
}

kernel void probe_qk256_split(device bfloat *q [[buffer(1)]], device bfloat *k [[buffer(2)]], device float *out [[buffer(3)]])
{
	ProbeScoresSplit<32, 256, 4>(q, k, out);
}
