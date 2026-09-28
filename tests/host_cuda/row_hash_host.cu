#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/row_hash.cuh"

#define HOST_ROWS 9u
#define HOST_WORDS 1283u
#define HOST_STRIDE_WORDS 1300u

static uint16_t host_data[HOST_ROWS * HOST_STRIDE_WORDS],host_moved[(HOST_ROWS + 4u) * (HOST_STRIDE_WORDS + 6u)];
static uint32_t host_state = 991u;

static uint32_t HostRandom(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return(host_state >> 8u);
}

static uint64_t HostSerial(const uint16_t *row,uint32_t words,uint32_t site)
{
	uint64_t sum = 0u;
	uint32_t index;
	for (index=words; index>0u; index--)
		sum += LmRowHashWord(site,index - 1u,row[index - 1u]);
	return(sum);
}

static uint64_t HostOne(const uint16_t *row,uint32_t words,uint32_t site)
{
	uint64_t hash;
	assert(LmRowHashLaunch(row,(uint64_t)words * 2u,words,1u,site,&hash,0) == LM_LAUNCH_OK);
	return(hash);
}

static void HostOrderAndPlacement(void)
{
	uint64_t hashes[HOST_ROWS],moved[HOST_ROWS + 4u];
	uint32_t row,index;
	for (index=0u; index<HOST_ROWS * HOST_STRIDE_WORDS; index++)
		host_data[index] = (uint16_t)HostRandom();
	assert(LmRowHashLaunch(host_data,HOST_STRIDE_WORDS * 2u,HOST_WORDS,HOST_ROWS,7u,hashes,0) == LM_LAUNCH_OK);
	for (row=0u; row<HOST_ROWS; row++)
	{
		assert(hashes[row] == HostSerial(host_data + row * HOST_STRIDE_WORDS,HOST_WORDS,7u));
		memcpy(host_moved + (HOST_ROWS + 3u - row) * (HOST_STRIDE_WORDS + 6u),host_data + row * HOST_STRIDE_WORDS,HOST_WORDS * 2u);
	}
	assert(LmRowHashLaunch(host_moved,(HOST_STRIDE_WORDS + 6u) * 2u,HOST_WORDS,HOST_ROWS + 4u,7u,moved,0) == LM_LAUNCH_OK);
	for (row=0u; row<HOST_ROWS; row++)
		assert(moved[HOST_ROWS + 3u - row] == hashes[row]);
	for (row=1u; row<HOST_ROWS; row++)
		assert(hashes[row] != hashes[0]);
	puts("row hash: 256 host threads and a reversed serial sum agree; a row keeps its hash at another row index and row stride");
}

static void HostSensitivity(void)
{
	uint16_t row[HOST_WORDS];
	uint64_t base;
	uint32_t bit,index,swap;
	float zero_f32[64],negative_f32[64];
	for (index=0u; index<HOST_WORDS; index++)
		row[index] = (uint16_t)HostRandom();
	base = HostOne(row,HOST_WORDS,3u);
	for (index=0u; index<HOST_WORDS; index+=HOST_WORDS - 1u)
		for (bit=0u; bit<16u; bit++)
		{
			row[index] ^= (uint16_t)(1u << bit);
			assert(HostOne(row,HOST_WORDS,3u) != base);
			row[index] ^= (uint16_t)(1u << bit);
		}
	assert(HostOne(row,HOST_WORDS,3u) == base);
	assert(HostOne(row,HOST_WORDS,4u) != base);
	row[10] = 0x1234u;
	row[11] = 0x4321u;
	base = HostOne(row,HOST_WORDS,3u);
	swap = row[10];
	row[10] = row[11];
	row[11] = (uint16_t)swap;
	assert(HostOne(row,HOST_WORDS,3u) != base);
	for (index=0u; index<64u; index++)
	{
		zero_f32[index] = 0.0f;
		negative_f32[index] = index == 17u ? -0.0f : 0.0f;
	}
	assert(HostOne((const uint16_t *)zero_f32,128u,5u) != HostOne((const uint16_t *)negative_f32,128u,5u));
	puts("row hash: every single-bit flip in the first and last word, a swap of two words, a different site and +0 against -0 change the hash");
}

static void HostRefusals(void)
{
	uint64_t hash;
	assert(LmRowHashLaunch((const uint8_t *)host_data + 1,2u,1u,1u,0u,&hash,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmRowHashLaunch(host_data,3u,1u,2u,0u,&hash,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmRowHashLaunch(host_data,4u,4u,2u,0u,&hash,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmRowHashLaunch(host_data,8u,4u,2u,LM_ROW_HASH_SITE_LIMIT,&hash,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmRowHashLaunch(host_data,8u,4u,0u,0u,&hash,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmRowHashLaunch(host_data,8u,0u,1u,0u,&hash,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmRowHashLaunch(0,8u,4u,1u,0u,&hash,0) == LM_LAUNCH_ERR_SHAPE);
	puts("row hash refusals: odd address, odd stride, overlapping rows, site out of range, zero rows, zero words and a null buffer");
}

int main(void)
{
	HostOrderAndPlacement();
	HostSensitivity();
	HostRefusals();
	puts("PASS row hash on host threads: per-row 64-bit hashes are order-free, placement-independent and change on any single-word change");
	return(0);
}
