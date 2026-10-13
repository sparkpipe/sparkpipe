#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/nvfp4.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int8.cuh"
#include "inference/kernels/norm.cuh"

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

template<class Format> struct ProbeDecoder;

template<> struct ProbeDecoder<LmNvfp4>
{
	static constexpr const char *kName = "nvfp4";
	static __device__ float Decode(uint32_t code) { return(LmE2m1PairToFloat((uint8_t)code).x); }
};

template<> struct ProbeDecoder<LmFp8>
{
	static constexpr const char *kName = "fp8";
	static __device__ float Decode(uint32_t code) { return(LmE4m3ToFloat((uint8_t)code)); }
};

template<> struct ProbeDecoder<LmInt8>
{
	static constexpr const char *kName = "int8";
	static __device__ float Decode(uint32_t code) { return((float)(int8_t)(uint8_t)code); }
};

template<class Format>
__global__ void ProbeNearestKernel(const uint16_t *__restrict__ input_bf16, const uint32_t *__restrict__ source_row_map, const uint8_t *__restrict__ codes, const uint8_t *__restrict__ scales, uint32_t rows, uint32_t dimension, unsigned int *violations, unsigned long long *first_violation)
{
	const uint32_t groups = dimension / Format::kScaleGroup;
	const uint32_t mask = (1u << Format::kStoredBits) - 1u;
	uint64_t element = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row,column,code,candidate;
	uint64_t bit;
	float value,scale,got,best = INFINITY;
	if ( element >= (uint64_t)rows * dimension )
		return;
	row = (uint32_t)(element / dimension);
	column = (uint32_t)(element % dimension);
	value = LmBf16ToFloat(input_bf16[((uint64_t)(source_row_map != 0 ? source_row_map[row] : row) * dimension) + column]);
	scale = LmUe4m3ToFloat(scales[((uint64_t)row * groups) + (column / Format::kScaleGroup)]);
	bit = element * Format::kStoredBits;
	code = ((uint32_t)codes[bit >> 3u] >> (uint32_t)(bit & 7u)) & mask;
	got = fabsf((ProbeDecoder<Format>::Decode(code) * scale) - value);
	for (candidate = 0u; candidate <= mask; ++candidate)
	{
		float decoded = ProbeDecoder<Format>::Decode(candidate);
		if ( !isnan(decoded) )
			best = fminf(best,fabsf((decoded * scale) - value));
	}
	if ( !(got <= best + (fabsf(value) * 1.0e-6f) + 1.0e-30f) )
	{
		atomicAdd(violations,1u);
		atomicMin(first_violation,(unsigned long long)element);
	}
}

static float ProbeBf16ToFloat(uint16_t bits)
{
	uint32_t widened = (uint32_t)bits << 16u;
	float value;
	memcpy(&value,&widened,sizeof(value));
	return(value);
}

static uint16_t ProbeFloatToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static float ProbeUe4m3ToFloat(uint8_t code)
{
	uint32_t biased = (uint32_t)(code & 0x7fu) >> 3u,mantissa = (uint32_t)code & 7u;
	if ( biased == 0u )
		return(ldexpf((float)mantissa,-9));
	return(ldexpf(1.0f + ((float)mantissa / 8.0f),(int32_t)biased - 7));
}

static uint64_t ProbeMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return(z ^ (z >> 31u));
}

static std::vector<uint16_t> ProbeInput(uint32_t rows,uint32_t dimension,uint32_t group_size)
{
	std::vector<uint16_t> input((uint64_t)rows * dimension);
	uint32_t row,column;
	for (row = 0u; row < rows; ++row)
		for (column = 0u; column < dimension; ++column)
		{
			uint32_t group = column / group_size;
			uint64_t salt = ((uint64_t)row << 32u) | column;
			double unit = ((double)(ProbeMix(salt) >> 11u) / 9007199254740992.0) * 2.0 - 1.0;
			double magnitude = ldexp(1.0 + ((double)(ProbeMix(((uint64_t)row << 16u) ^ group ^ 0x5151u) & 1023u) / 1024.0),
				(int32_t)((row * 7u + group * 3u) % 27u) - 14);
			if ( row == 0u && group == 0u )
				magnitude = 0.0;
			if ( row == 0u && group == 1u )
				magnitude = 1.0e-4;
			if ( row == 0u && group == 2u )
				magnitude = 1048576.0;
			input[((uint64_t)row * dimension) + column] = ProbeFloatToBf16((float)(magnitude * unit));
		}
	return(input);
}

template<class Format>
static void ProbeFormat(uint32_t rows,uint32_t dimension,uint32_t mapped)
{
	const uint32_t groups = dimension / Format::kScaleGroup;
	const uint64_t elements = (uint64_t)rows * dimension;
	const uint64_t code_bytes = (elements * Format::kStoredBits) / 8u;
	std::vector<uint16_t> input = ProbeInput(rows,dimension,Format::kScaleGroup);
	std::vector<uint32_t> row_map(rows);
	std::vector<uint8_t> scales((uint64_t)rows * groups);
	std::vector<uint8_t> codes(code_bytes);
	uint16_t *device_input;
	uint32_t *device_map = 0;
	uint8_t *device_codes,*device_scales;
	unsigned int *device_violations,violations = 0u;
	unsigned long long *device_first,first = ~0ull;
	uint32_t row,group,index,tiny_zero = 0u,coarse = 0u;
	char label[64];
	snprintf(label,sizeof(label),"%s rows%u dim%u mapped%u",ProbeDecoder<Format>::kName,rows,dimension,mapped);
	for (row = 0u; row < rows; ++row)
		row_map[row] = mapped != 0u ? rows - 1u - row : row;
	PROBE_CUDA(cudaMalloc((void **)&device_input,elements * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_codes,code_bytes));
	PROBE_CUDA(cudaMalloc((void **)&device_scales,scales.size()));
	PROBE_CUDA(cudaMalloc((void **)&device_violations,sizeof(violations)));
	PROBE_CUDA(cudaMalloc((void **)&device_first,sizeof(first)));
	PROBE_CUDA(cudaMemcpy(device_input,input.data(),elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
	if ( mapped != 0u )
	{
		PROBE_CUDA(cudaMalloc((void **)&device_map,rows * sizeof(uint32_t)));
		PROBE_CUDA(cudaMemcpy(device_map,row_map.data(),rows * sizeof(uint32_t),cudaMemcpyHostToDevice));
	}
	PROBE_CUDA(cudaMemset(device_codes,0xa5,code_bytes));
	PROBE_CUDA(cudaMemset(device_scales,0xa5,scales.size()));
	PROBE_CUDA(cudaMemcpy(device_violations,&violations,sizeof(violations),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_first,&first,sizeof(first),cudaMemcpyHostToDevice));
	LmQuantiseRowsKernel<Format,256u><<<dim3(rows,groups),256u,(Format::kScaleGroup + 8u) * sizeof(float)>>>(
		device_input,device_map,device_codes,device_scales,rows,dimension);
	PROBE_CUDA(cudaGetLastError());
	ProbeNearestKernel<Format><<<(uint32_t)((elements + 255u) / 256u),256u>>>(
		device_input,device_map,device_codes,device_scales,rows,dimension,device_violations,device_first);
	PROBE_CUDA(cudaGetLastError());
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(&violations,device_violations,sizeof(violations),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(&first,device_first,sizeof(first),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(scales.data(),device_scales,scales.size(),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(codes.data(),device_codes,code_bytes,cudaMemcpyDeviceToHost));
	if ( violations != 0u )
	{
		printf("FAIL %s: %u of %llu codes are not the nearest grid point against the stored scale (first element %llu)\n",
			label,violations,(unsigned long long)elements,first);
		probe_failures++;
	}
	for (row = 0u; row < rows; ++row)
		for (group = 0u; group < groups; ++group)
		{
			const uint64_t source = (uint64_t)row_map[row] * dimension + (uint64_t)group * Format::kScaleGroup;
			const uint8_t code = scales[((uint64_t)row * groups) + group];
			float absmax = 0.0f;
			for (index = 0u; index < Format::kScaleGroup; ++index)
				absmax = fmaxf(absmax,fabsf(ProbeBf16ToFloat(input[source + index])));
			if ( (code & 0x7fu) == 0x7fu || (code & 0x80u) != 0u )
			{
				printf("FAIL %s: row %u group %u stored the invalid scale code 0x%02x\n",label,row,group,code);
				probe_failures++;
			}
			else if ( absmax > 0.0f && code == 0u )
				tiny_zero++;
			else if ( code >= 0x08u && code < 0x7eu && absmax > 0.0f )
			{
				const float scale = ProbeUe4m3ToFloat(code);
				const float top = absmax / scale;
				if ( top < Format::kMax * 0.875f || top > Format::kMax * 1.125f )
					coarse++;
			}
		}
	if ( tiny_zero != 0u )
	{
		printf("FAIL %s: %u nonzero groups stored the zero scale and dequantise to all zeros\n",label,tiny_zero);
		probe_failures++;
	}
	if ( coarse != 0u )
	{
		printf("FAIL %s: %u groups have a stored scale more than one eighth away from absmax/kMax\n",label,coarse);
		probe_failures++;
	}
	if ( violations == 0u && tiny_zero == 0u && coarse == 0u )
		printf("ok %s\n",label);
	if ( device_map != 0 )
		PROBE_CUDA(cudaFree(device_map));
	PROBE_CUDA(cudaFree(device_input));
	PROBE_CUDA(cudaFree(device_codes));
	PROBE_CUDA(cudaFree(device_scales));
	PROBE_CUDA(cudaFree(device_violations));
	PROBE_CUDA(cudaFree(device_first));
}

int main(void)
{
	uint32_t mapped;
	for (mapped = 0u; mapped < 2u; ++mapped)
	{
		ProbeFormat<LmNvfp4>(5u,1024u,mapped);
		ProbeFormat<LmFp8>(5u,1024u,mapped);
		ProbeFormat<LmInt8>(5u,1024u,mapped);
	}
	if ( probe_failures != 0 )
	{
		printf("FAIL quantise_rows_cuda: %d failures\n",probe_failures);
		return(1);
	}
	printf("PASS quantise_rows_cuda\n");
	return(0);
}
