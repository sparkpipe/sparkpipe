#pragma once

#include <stdint.h>

#define LM_SPLITMIX64_INCREMENT 0x9e3779b97f4a7c15ull
#define LM_SPLITMIX64_MULTIPLIER_A 0xbf58476d1ce4e5b9ull
#define LM_SPLITMIX64_MULTIPLIER_B 0x94d049bb133111ebull

static __host__ __device__ __forceinline__ uint64_t LmSplitMix64(uint64_t value)
{
	value += LM_SPLITMIX64_INCREMENT;
	value = (value ^ (value >> 30u)) * LM_SPLITMIX64_MULTIPLIER_A;
	value = (value ^ (value >> 27u)) * LM_SPLITMIX64_MULTIPLIER_B;
	return(value ^ (value >> 31u));
}
