#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/linear_attn.cuh"

#define SCAN_THREADS 256u
#define SCAN_KEY 128u
#define SCAN_VALUE 128u

static int scan_failures;

#define SCAN_CUDA(call) do { cudaError_t scan_error = (call); if ( scan_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(scan_error)); exit(1); } } while (0)

static uint64_t ScanMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static float ScanUniform(uint64_t index, uint64_t salt)
{
	return (float)((int32_t)((ScanMix(index ^ (salt << 48u)) >> 40u) & 0xfffu) - 2048) / 2048.0f;
}

static uint16_t ScanBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static float ScanFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return result;
}

template<class T>
static T *ScanDevice(const std::vector<T> &host)
{
	T *device;
	SCAN_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	SCAN_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static void ScanRun(std::vector<float> &state, const std::vector<uint32_t> &begin, uint32_t heads, const uint16_t *query, const uint16_t *key,
	const uint16_t *value, const float *forget, const float *beta, uint16_t *output)
{
	const uint32_t sequences = (uint32_t)begin.size() - 1u;
	const uint64_t slot_bytes = (uint64_t)heads * SCAN_KEY * SCAN_VALUE * sizeof(float);
	std::vector<uint32_t> slots(sequences);
	for ( uint32_t sequence = 0u; sequence < sequences; sequence++ )
		slots[sequence] = sequence;
	uint32_t *begin_device = ScanDevice(begin),*slots_device = ScanDevice(slots);
	float *state_device = ScanDevice(state);
	LmDeltaRuleColumnKernel<SCAN_THREADS,SCAN_KEY,SCAN_VALUE,float><<<dim3(sequences,heads,SCAN_VALUE / LM_WARP_LANES),LM_DELTA_COLUMN_THREADS>>>(
		(uint8_t *)state_device,(uint32_t)slot_bytes,slots_device,begin_device,0,query,key,value,forget,beta,output,heads,1u,sequences,1u,0);
	SCAN_CUDA(cudaGetLastError());
	SCAN_CUDA(cudaDeviceSynchronize());
	SCAN_CUDA(cudaMemcpy(state.data(),state_device,state.size() * sizeof(float),cudaMemcpyDeviceToHost));
	cudaFree(begin_device); cudaFree(slots_device); cudaFree(state_device);
}

static void ScanCase(const char *label, uint32_t rows, uint32_t segments, uint32_t heads, float beta_scale, float forget_floor)
{
	const uint64_t head_state = (uint64_t)SCAN_KEY * SCAN_VALUE,elements = (uint64_t)rows * heads * SCAN_KEY;
	std::vector<uint16_t> query(elements),key(elements),value((uint64_t)rows * heads * SCAN_VALUE),zero(value.size(),0u);
	std::vector<float> forget(elements),beta((uint64_t)rows * heads);
	std::vector<uint32_t> whole = {0u,rows},split(segments + 1u);
	uint32_t segment,head,row,column,inner;
	for ( uint64_t i = 0u; i < elements; i++ )
	{
		query[i] = ScanBf16(ScanUniform(i,11u) * 0.088f);
		forget[i] = forget_floor + (1.0f - forget_floor) * (0.5f + 0.5f * ScanUniform(i,41u));
	}
	for ( row = 0u; row < rows; row++ )
		for ( head = 0u; head < heads; head++ )
		{
			double norm = 0.0;
			for ( column = 0u; column < SCAN_KEY; column++ )
				norm += (double)ScanUniform(((uint64_t)row * heads + head) * SCAN_KEY + column,23u) * ScanUniform(((uint64_t)row * heads + head) * SCAN_KEY + column,23u);
			for ( column = 0u; column < SCAN_KEY; column++ )
				key[((uint64_t)row * heads + head) * SCAN_KEY + column] = ScanBf16((float)(ScanUniform(((uint64_t)row * heads + head) * SCAN_KEY + column,23u) / sqrt(norm + 1e-6)));
		}
	for ( uint64_t i = 0u; i < value.size(); i++ )
		value[i] = ScanBf16(ScanUniform(i,37u));
	for ( uint64_t i = 0u; i < beta.size(); i++ )
		beta[i] = beta_scale * (0.5f + 0.5f * ScanUniform(i,53u));
	for ( segment = 0u; segment <= segments; segment++ )
		split[segment] = (uint32_t)((uint64_t)rows * segment / segments);
	uint16_t *query_device = ScanDevice(query),*key_device = ScanDevice(key),*value_device = ScanDevice(value),*zero_device = ScanDevice(zero);
	float *forget_device = ScanDevice(forget),*beta_device = ScanDevice(beta);
	uint16_t *reference_device,*segmented_device,*scratch_device;
	const uint64_t out_elements = (uint64_t)rows * heads * SCAN_VALUE;
	SCAN_CUDA(cudaMalloc((void **)&reference_device,out_elements * 2u));
	SCAN_CUDA(cudaMalloc((void **)&segmented_device,out_elements * 2u));
	SCAN_CUDA(cudaMalloc((void **)&scratch_device,out_elements * 2u));

	std::vector<float> sequential((uint64_t)heads * head_state,0.0f);
	ScanRun(sequential,whole,heads,query_device,key_device,value_device,forget_device,beta_device,reference_device);

	std::vector<float> ends((uint64_t)segments * heads * head_state,0.0f),transforms(ends.size(),0.0f),incoming(ends.size(),0.0f);
	ScanRun(ends,split,heads,query_device,key_device,value_device,forget_device,beta_device,scratch_device);
	for ( segment = 0u; segment < segments; segment++ )
		for ( head = 0u; head < heads; head++ )
			for ( row = 0u; row < SCAN_KEY; row++ )
				transforms[((uint64_t)segment * heads + head) * head_state + (uint64_t)row * SCAN_KEY + row] = 1.0f;
	ScanRun(transforms,split,heads,query_device,key_device,zero_device,forget_device,beta_device,scratch_device);
	for ( segment = 1u; segment < segments; segment++ )
		for ( head = 0u; head < heads; head++ )
		{
			const float *m = &transforms[((uint64_t)(segment - 1u) * heads + head) * head_state];
			const float *previous = &incoming[((uint64_t)(segment - 1u) * heads + head) * head_state];
			const float *u = &ends[((uint64_t)(segment - 1u) * heads + head) * head_state];
			float *next = &incoming[((uint64_t)segment * heads + head) * head_state];
			for ( row = 0u; row < SCAN_KEY; row++ )
				for ( column = 0u; column < SCAN_VALUE; column++ )
				{
					double total = u[(uint64_t)row * SCAN_VALUE + column];
					for ( inner = 0u; inner < SCAN_KEY; inner++ )
						total += (double)m[(uint64_t)row * SCAN_KEY + inner] * previous[(uint64_t)inner * SCAN_VALUE + column];
					next[(uint64_t)row * SCAN_VALUE + column] = (float)total;
				}
		}
	std::vector<float> finals = incoming;
	ScanRun(finals,split,heads,query_device,key_device,value_device,forget_device,beta_device,segmented_device);

	std::vector<uint16_t> reference(out_elements),segmented(out_elements);
	SCAN_CUDA(cudaMemcpy(reference.data(),reference_device,out_elements * 2u,cudaMemcpyDeviceToHost));
	SCAN_CUDA(cudaMemcpy(segmented.data(),segmented_device,out_elements * 2u,cudaMemcpyDeviceToHost));
	uint64_t differ = 0u,beyond = 0u;
	double worst = 0.0,scale = 0.0,state_worst = 0.0,state_scale = 0.0;
	for ( uint64_t i = 0u; i < out_elements; i++ )
	{
		const double a = ScanFloat(reference[i]),b = ScanFloat(segmented[i]),gap = a > b ? a - b : b - a;
		differ += reference[i] != segmented[i];
		scale = fabs(a) > scale ? fabs(a) : scale;
		worst = gap > worst ? gap : worst;
	}
	for ( uint64_t i = 0u; i < out_elements; i++ )
	{
		const double a = ScanFloat(reference[i]),b = ScanFloat(segmented[i]),gap = a > b ? a - b : b - a;
		const double magnitude = fabs(a) > fabs(b) ? fabs(a) : fabs(b),ulp = magnitude > 0.0 ? ldexp(1.0,ilogb(magnitude) - 7) : 0.0;
		beyond += gap > ulp && gap > 1e-5 * scale;
	}
	for ( uint64_t i = 0u; i < sequential.size(); i++ )
	{
		const double expected = sequential[i],gap = fabs((double)finals[(uint64_t)(segments - 1u) * heads * head_state + i] - expected);
		state_worst = gap > state_worst ? gap : state_worst;
		state_scale = fabs(expected) > state_scale ? fabs(expected) : state_scale;
	}
	printf("%s rows=%u segments=%u heads=%u beta<=%.1f forget>=%.2f: outputs differing %llu of %llu (beyond one bf16 ulp and 1e-5 of the output scale: %llu), worst %.3e of %.3e; final state worst %.3e of %.3e\n",
		label,rows,segments,heads,beta_scale,forget_floor,(unsigned long long)differ,(unsigned long long)out_elements,(unsigned long long)beyond,worst,scale,state_worst,state_scale);
	if ( beyond != 0u || state_worst > 1e-5 * (state_scale > 1.0 ? state_scale : 1.0) || differ * 1000u > out_elements )
	{
		printf("FAIL %s: the segmented scan departs from the sequential recurrence\n",label);
		scan_failures++;
	}
	cudaFree(query_device); cudaFree(key_device); cudaFree(value_device); cudaFree(zero_device); cudaFree(forget_device); cudaFree(beta_device);
	cudaFree(reference_device); cudaFree(segmented_device); cudaFree(scratch_device);
}

int main(void)
{
	ScanCase("two-segments",512u,2u,4u,1.0f,0.75f);
	ScanCase("thirty-two-segments",4096u,32u,4u,2.0f,0.75f);
	ScanCase("slow-decay",8192u,32u,2u,2.0f,0.98f);
	ScanCase("uneven-segments",3001u,7u,3u,2.0f,0.9f);
	if ( scan_failures != 0 )
	{
		printf("FAIL segmented delta-rule scan: %d case(s)\n",scan_failures);
		return 1;
	}
	printf("PASS segmented delta-rule scan: zero-start chunks, chunk transforms from an identity state with zero values, and the prefixed incoming states reproduce the sequential recurrence\n");
	return 0;
}
