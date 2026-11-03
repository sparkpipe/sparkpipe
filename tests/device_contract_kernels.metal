#include <metal_stdlib>

using namespace metal;

struct ContractParams
{
	uint count;
	uint seed;
};

kernel void contract_write(constant ContractParams &p [[buffer(0)]], device uint *out [[buffer(1)]], uint index [[thread_position_in_grid]])
{
	if ( index < p.count )
		out[index] = index * 2654435761u + p.seed;
}

kernel void contract_read(constant ContractParams &p [[buffer(0)]], device const uint *in [[buffer(1)]], device uint *out [[buffer(2)]],
	uint index [[thread_position_in_grid]])
{
	if ( index < p.count )
		out[index] = in[index] + 1u;
}

kernel void contract_shared(constant ContractParams &p [[buffer(0)]], device uint *out [[buffer(1)]], threadgroup uint *scratch [[threadgroup(0)]],
	uint local [[thread_position_in_threadgroup]], uint group [[threadgroup_position_in_grid]], uint width [[threads_per_threadgroup]])
{
	scratch[local] = local + p.seed;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	out[group * width + local] = scratch[width - 1u - local];
}
