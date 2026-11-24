#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256) SparkDeviceProgramCopy(uint8_t *destination, const uint8_t *source, uint64_t bytes)
{
	const uint64_t thread = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,stride = (uint64_t)gridDim.x * blockDim.x;
	const uint64_t misalign = (16u - ((uintptr_t)destination & 15u)) & 15u,lead = misalign < bytes ? misalign : bytes;
	uint64_t index;
	if ( (((uintptr_t)destination ^ (uintptr_t)source) & 15u) == 0u )
	{
		const uint64_t body = (bytes - lead) / 16u;
		uint4 *to = (uint4 *)(void *)(destination + lead);
		const uint4 *from = (const uint4 *)(const void *)(source + lead);
		for ( index = thread; index < lead; index += stride )
			destination[index] = source[index];
		for ( index = thread; index < body; index += stride )
			to[index] = from[index];
		for ( index = lead + body * 16u + thread; index < bytes; index += stride )
			destination[index] = source[index];
		return;
	}
	for ( index = thread; index < bytes; index += stride )
		destination[index] = source[index];
}

extern "C" __global__ void __launch_bounds__(256) SparkDeviceProgramFill(uint8_t *destination, uint32_t value, uint64_t bytes)
{
	const uint64_t thread = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,stride = (uint64_t)gridDim.x * blockDim.x;
	const uint64_t misalign = (16u - ((uintptr_t)destination & 15u)) & 15u,lead = misalign < bytes ? misalign : bytes;
	const uint64_t body = (bytes - lead) / 16u;
	const uint32_t word = (value & 0xffu) * 0x01010101u;
	const uint4 pattern = make_uint4(word,word,word,word);
	uint4 *to = (uint4 *)(void *)(destination + lead);
	uint64_t index;
	for ( index = thread; index < lead; index += stride )
		destination[index] = (uint8_t)value;
	for ( index = thread; index < body; index += stride )
		to[index] = pattern;
	for ( index = lead + body * 16u + thread; index < bytes; index += stride )
		destination[index] = (uint8_t)value;
}
