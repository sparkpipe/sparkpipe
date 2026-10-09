#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/linear_attn.cuh"

#define DELTA_THREADS 256u
#define DELTA_KEY 128u
#define DELTA_VALUE 128u
#define DELTA_COLUMNS 32u

static int delta_failures;

#define DELTA_CUDA(call) do { cudaError_t delta_error = (call); if ( delta_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(delta_error)); exit(1); } } while (0)

static uint64_t DeltaMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static float DeltaUniform(uint64_t index, uint64_t salt)
{
	return (float)((int32_t)((DeltaMix(index ^ salt) >> 40u) & 0xfffu) - 2048) / 2048.0f;
}

static uint16_t DeltaBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

template<class T>
static T *DeltaDevice(const std::vector<T> &host)
{
	T *device;
	DELTA_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	DELTA_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static void DeltaCase(const char *label, const std::vector<uint32_t> &counts, uint32_t heads, uint32_t commit, uint32_t reverse)
{
	const uint32_t sequences = (uint32_t)counts.size();
	const uint64_t slot_bytes = (uint64_t)heads * DELTA_KEY * DELTA_VALUE * sizeof(float);
	std::vector<uint32_t> begin(sequences + 1u,0u),slots(sequences),indices;
	uint32_t rows,sequence,index;
	for ( sequence = 0u; sequence < sequences; sequence++ )
		begin[sequence + 1u] = begin[sequence] + counts[sequence];
	rows = begin[sequences];
	indices.resize(rows);
	for ( index = 0u; index < rows; index++ )
		indices[index] = reverse != 0u ? rows - 1u - index : index;
	for ( sequence = 0u; sequence < sequences; sequence++ )
		slots[sequence] = (sequences - 1u - sequence) * 2u + 1u;
	const uint32_t slot_count = sequences * 2u + 1u;
	std::vector<uint16_t> query((uint64_t)rows * heads * DELTA_KEY),key(query.size()),value((uint64_t)rows * heads * DELTA_VALUE);
	std::vector<float> forget((uint64_t)rows * heads * DELTA_KEY),beta((uint64_t)rows * heads),state((uint64_t)slot_count * slot_bytes / sizeof(float));
	for ( uint64_t i = 0u; i < query.size(); i++ )
	{
		query[i] = DeltaBf16(DeltaUniform(i,11u));
		key[i] = DeltaBf16(DeltaUniform(i,23u));
	}
	for ( uint64_t i = 0u; i < value.size(); i++ )
		value[i] = DeltaBf16(DeltaUniform(i,37u));
	for ( uint64_t i = 0u; i < forget.size(); i++ )
		forget[i] = 0.75f + 0.25f * (0.5f + 0.5f * DeltaUniform(i,41u));
	for ( uint64_t i = 0u; i < beta.size(); i++ )
		beta[i] = 0.5f + 0.5f * DeltaUniform(i,53u);
	for ( uint64_t i = 0u; i < state.size(); i++ )
		state[i] = 0.25f * DeltaUniform(i,67u);
	uint32_t *begin_device = DeltaDevice(begin),*slots_device = DeltaDevice(slots),*indices_device = DeltaDevice(indices);
	uint16_t *query_device = DeltaDevice(query),*key_device = DeltaDevice(key),*value_device = DeltaDevice(value);
	float *forget_device = DeltaDevice(forget),*beta_device = DeltaDevice(beta);
	float *state_old = DeltaDevice(state),*state_new = DeltaDevice(state);
	uint16_t *out_old,*out_new;
	const uint64_t out_elements = (uint64_t)rows * heads * DELTA_VALUE;
	DELTA_CUDA(cudaMalloc((void **)&out_old,out_elements * 2u));
	DELTA_CUDA(cudaMalloc((void **)&out_new,out_elements * 2u));
	DELTA_CUDA(cudaMemset(out_old,0,out_elements * 2u));
	DELTA_CUDA(cudaMemset(out_new,0xff,out_elements * 2u));
	LmDeltaRuleKernel<DELTA_THREADS,DELTA_KEY,DELTA_VALUE,float,DELTA_COLUMNS><<<dim3(sequences,heads,DELTA_VALUE / DELTA_COLUMNS),DELTA_THREADS,DELTA_KEY * DELTA_COLUMNS * sizeof(float)>>>(
		(uint8_t *)state_old,(uint32_t)slot_bytes,slots_device,begin_device,0,query_device,key_device,value_device,forget_device,beta_device,out_old,heads,1u,sequences,commit,indices_device);
	DELTA_CUDA(cudaGetLastError());
	LmDeltaRuleColumnKernel<DELTA_THREADS,DELTA_KEY,DELTA_VALUE,float><<<dim3(sequences,heads,DELTA_VALUE / LM_WARP_LANES),LM_DELTA_COLUMN_THREADS>>>(
		(uint8_t *)state_new,(uint32_t)slot_bytes,slots_device,begin_device,0,query_device,key_device,value_device,forget_device,beta_device,out_new,heads,1u,sequences,commit,indices_device);
	DELTA_CUDA(cudaGetLastError());
	DELTA_CUDA(cudaDeviceSynchronize());
	std::vector<uint16_t> old_host(out_elements),new_host(out_elements);
	std::vector<float> state_old_host(state.size()),state_new_host(state.size());
	DELTA_CUDA(cudaMemcpy(old_host.data(),out_old,out_elements * 2u,cudaMemcpyDeviceToHost));
	DELTA_CUDA(cudaMemcpy(new_host.data(),out_new,out_elements * 2u,cudaMemcpyDeviceToHost));
	DELTA_CUDA(cudaMemcpy(state_old_host.data(),state_old,state.size() * sizeof(float),cudaMemcpyDeviceToHost));
	DELTA_CUDA(cudaMemcpy(state_new_host.data(),state_new,state.size() * sizeof(float),cudaMemcpyDeviceToHost));
	uint64_t output_mismatches = 0u,state_mismatches = 0u,state_changed = 0u;
	for ( uint64_t i = 0u; i < out_elements; i++ )
		output_mismatches += old_host[i] != new_host[i];
	for ( uint64_t i = 0u; i < state.size(); i++ )
	{
		state_mismatches += memcmp(&state_old_host[i],&state_new_host[i],sizeof(float)) != 0;
		state_changed += memcmp(&state_old_host[i],&state[i],sizeof(float)) != 0;
	}
	printf("%s rows=%u sequences=%u heads=%u commit=%u: output mismatches %llu of %llu, state mismatches %llu, states changed %llu\n",
		label,rows,sequences,heads,commit,(unsigned long long)output_mismatches,(unsigned long long)out_elements,
		(unsigned long long)state_mismatches,(unsigned long long)state_changed);
	if ( output_mismatches != 0u || state_mismatches != 0u )
	{
		printf("FAIL %s: the column kernel differs from the block kernel\n",label);
		delta_failures++;
	}
	if ( (commit != 0u) != (state_changed != 0u) )
	{
		printf("FAIL %s: commit=%u but %llu state words changed\n",label,commit,(unsigned long long)state_changed);
		delta_failures++;
	}
	cudaFree(begin_device); cudaFree(slots_device); cudaFree(indices_device); cudaFree(query_device); cudaFree(key_device);
	cudaFree(value_device); cudaFree(forget_device); cudaFree(beta_device); cudaFree(state_old); cudaFree(state_new);
	cudaFree(out_old); cudaFree(out_new);
}

int main(void)
{
	DeltaCase("decode-row",{1u},6u,1u,0u);
	DeltaCase("short-prefill",{17u},6u,1u,0u);
	DeltaCase("wide-prefill",{1024u},6u,1u,0u);
	DeltaCase("eight-sequences",{1u,1u,1u,1u,1u,1u,1u,1u},6u,1u,0u);
	DeltaCase("mixed-lengths-reordered",{300u,5u,129u},12u,1u,1u);
	DeltaCase("no-commit",{64u,33u},6u,0u,0u);
	if ( delta_failures != 0 )
	{
		printf("FAIL delta rule column kernel: %d case(s)\n",delta_failures);
		return 1;
	}
	printf("PASS delta rule column kernel equals the block kernel bit for bit\n");
	return 0;
}
