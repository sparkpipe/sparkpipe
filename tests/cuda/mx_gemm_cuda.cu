#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/kernels/mx_gemm.cuh"

#define PROBE_ULP_LIMIT 1.0
#define PROBE_ACCUMULATION_EPSILON 3.814697265625e-06
#define PROBE_SAMPLES 2048u

static int probe_failures;
static uint32_t probe_multiprocessors;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)
#define PROBE_OK(call) do { int32_t probe_status = (call); if ( probe_status != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n",#call,probe_status); exit(1); } } while (0)

static uint64_t ProbeHash(uint64_t value)
{
	value = value * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
	value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
	return(value ^ (value >> 31u));
}

static double ProbeE4m3(uint8_t code)
{
	uint32_t exponent = (code >> 3u) & 15u,mantissa = code & 7u;
	double value = exponent == 0u ? ldexp((double)mantissa,-9) : ldexp(1.0 + mantissa / 8.0,(int)exponent - 7);
	return((code & 0x80u) != 0u ? -value : value);
}

static float ProbeBf16Float(uint16_t code)
{
	uint32_t bits = (uint32_t)code << 16u;
	float value;
	memcpy(&value,&bits,4u);
	return(value);
}

static uint16_t ProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return((uint16_t)(bits >> 16u));
}

static double ProbeUlp(double value)
{
	int exponent;
	if ( value == 0.0 )
		return(ldexp(1.0,-133));
	frexp(fabs(value),&exponent);
	return(ldexp(1.0,exponent - 8));
}

static double ProbeRoundE4m3(double value)
{
	double magnitude = fabs(value),quantum,rounded;
	int exponent;
	if ( magnitude == 0.0 )
		return(0.0);
	frexp(magnitude,&exponent);
	quantum = exponent - 1 < -6 ? ldexp(1.0,-9) : ldexp(1.0,exponent - 4);
	rounded = nearbyint(magnitude / quantum) * quantum;
	if ( rounded > 448.0 )
		rounded = 448.0;
	return(value < 0.0 ? -rounded : rounded);
}

static float ProbeQdqScale(float amax)
{
	float scaled = (amax > 1e-4f ? amax : 1e-4f) * (1.0f / 448.0f),scale;
	uint32_t bits;
	int exponent;
	memcpy(&bits,&scaled,4u);
	exponent = (int)((bits >> 23u) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u ? 1 : 0);
	bits = (uint32_t)(exponent + 127) << 23u;
	memcpy(&scale,&bits,4u);
	return(scale);
}

static void ProbeQuantize(uint64_t elements, uint32_t seed, uint32_t presquashed, cudaStream_t stream)
{
	std::vector<uint16_t> source(elements);
	std::vector<uint8_t> codes(elements),scales(elements / 32u);
	uint16_t *device_source;
	uint8_t *device_codes,*device_scales;
	uint64_t index,block,mismatches = 0u;
	for (index=0u; index<elements; index++)
	{
		const uint64_t hash = ProbeHash(index ^ ((uint64_t)seed << 40u));
		const int exponent = (int)(hash % 40u) - 30;
		float value = (float)ldexp(1.0 + (double)((hash >> 8u) & 0xffffu) / 65536.0,exponent);
		if ( ((hash >> 30u) & 1u) != 0u )
			value = -value;
		if ( (index / 32u) % 17u == 3u )
			value = 0.0f;
		if ( (index / 32u) % 23u == 5u )
			value *= 1e-9f;
		source[index] = ProbeBf16(value);
	}
	if ( presquashed != 0u )
		for (block=0u; block<elements / 32u; block++)
		{
			float amax = 0.0f,scale;
			for (index=block * 32u; index<block * 32u + 32u; index++)
				amax = fabsf(ProbeBf16Float(source[index])) > amax ? fabsf(ProbeBf16Float(source[index])) : amax;
			scale = ProbeQdqScale(amax);
			for (index=block * 32u; index<block * 32u + 32u; index++)
				source[index] = ProbeBf16((float)(ProbeRoundE4m3((double)(ProbeBf16Float(source[index]) / scale)) * scale));
		}
	PROBE_CUDA(cudaMalloc(&device_source,elements * 2u));
	PROBE_CUDA(cudaMalloc(&device_codes,elements));
	PROBE_CUDA(cudaMalloc(&device_scales,elements / 32u));
	PROBE_CUDA(cudaMemcpy(device_source,source.data(),elements * 2u,cudaMemcpyHostToDevice));
	PROBE_OK(LmMxQuantize(device_source,device_codes,device_scales,elements,stream));
	PROBE_CUDA(cudaStreamSynchronize(stream));
	PROBE_CUDA(cudaMemcpy(codes.data(),device_codes,elements,cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(scales.data(),device_scales,elements / 32u,cudaMemcpyDeviceToHost));
	for (block=0u; block<elements / 32u; block++)
	{
		float amax = 0.0f,scale;
		for (index=block * 32u; index<block * 32u + 32u; index++)
			amax = fabsf(ProbeBf16Float(source[index])) > amax ? fabsf(ProbeBf16Float(source[index])) : amax;
		scale = ProbeQdqScale(amax);
		for (index=block * 32u; index<block * 32u + 32u; index++)
		{
			double clamped = (double)(ProbeBf16Float(source[index]) / scale);
			const double expected = ProbeRoundE4m3(clamped > 448.0 ? 448.0 : clamped < -448.0 ? -448.0 : clamped) * scale;
			const double produced = ProbeE4m3(codes[index]) * ldexp(1.0,(int)scales[block] - 127);
			if ( expected != produced || (presquashed != 0u && produced != (double)ProbeBf16Float(source[index])) )
				mismatches++;
		}
	}
	if ( elements % 256u == 0u )
	{
		const uint32_t rows = (uint32_t)(elements / 256u);
		std::vector<uint8_t> strided_codes((size_t)rows * 128u), strided_scales((size_t)rows * 4u);
		PROBE_OK(LmMxQuantizeRows(device_source,256u,device_codes,device_scales,rows,128u,stream));
		PROBE_CUDA(cudaStreamSynchronize(stream));
		PROBE_CUDA(cudaMemcpy(strided_codes.data(),device_codes,strided_codes.size(),cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(strided_scales.data(),device_scales,strided_scales.size(),cudaMemcpyDeviceToHost));
		for (block=0u; block<rows; block++)
			for (index=0u; index<128u; index++)
				if ( strided_codes[block * 128u + index] != codes[block * 256u + index] ||
					strided_scales[block * 4u + index / 32u] != scales[block * 8u + index / 32u] )
					mismatches++;
	}
	PROBE_CUDA(cudaFree(device_source));
	PROBE_CUDA(cudaFree(device_codes));
	PROBE_CUDA(cudaFree(device_scales));
	if ( mismatches != 0u )
		probe_failures++;
	printf("%s quantize %llu elements%s: %llu of them differ from the fp8 qdq rounding%s\n",mismatches == 0u ? "PASS" : "FAIL",(unsigned long long)elements,
		presquashed != 0u ? " already on the fp8 grid" : "",(unsigned long long)mismatches,presquashed != 0u ? " or from their own value" : "");
}

typedef int32_t (*ProbeLaunch)(const LmMxGemmArguments *, uint32_t, cudaStream_t);

static void ProbeGemm(uint32_t rows, uint32_t input, uint32_t output, uint32_t seed, cudaStream_t stream)
{
	static const ProbeLaunch launches[] = { LmMxGemmLaunchTile<128u,128u,2u>, LmMxGemmLaunchTile<64u,128u,2u>, LmMxGemmLaunchTile<32u,128u,2u> };
	const uint64_t a_count = (uint64_t)rows * input,w_count = (uint64_t)output * input,out_count = (uint64_t)rows * output;
	std::vector<uint8_t> a(a_count),a_scale(a_count / 32u),w(w_count),w_scale(w_count / 32u);
	std::vector<uint16_t> result(out_count),other(out_count);
	uint8_t *device_a,*device_a_scale,*device_w,*device_w_scale;
	uint16_t *device_out;
	uint64_t index,differ = 0u;
	double worst = 0.0;
	uint32_t sample,variant,failed = probe_failures;
	LmMxGemmArguments args;
	for (index=0u; index<a_count; index++)
	{
		uint8_t code = (uint8_t)ProbeHash(index * 3u + seed);
		a[index] = (code & 0x7fu) == 0x7fu ? (uint8_t)(code ^ 1u) : code;
	}
	for (index=0u; index<w_count; index++)
	{
		uint8_t code = (uint8_t)ProbeHash(index * 5u + seed + 77u);
		w[index] = (code & 0x7fu) == 0x7fu ? (uint8_t)(code ^ 1u) : code;
	}
	for (index=0u; index<a_count / 32u; index++)
		a_scale[index] = (uint8_t)(118u + ProbeHash(index + seed * 7u) % 9u);
	for (index=0u; index<w_count / 32u; index++)
		w_scale[index] = (uint8_t)(114u + ProbeHash(index + seed * 11u) % 9u);
	PROBE_CUDA(cudaMalloc(&device_a,a_count));
	PROBE_CUDA(cudaMalloc(&device_a_scale,a_count / 32u));
	PROBE_CUDA(cudaMalloc(&device_w,w_count));
	PROBE_CUDA(cudaMalloc(&device_w_scale,w_count / 32u));
	PROBE_CUDA(cudaMalloc(&device_out,out_count * 2u));
	PROBE_CUDA(cudaMemcpy(device_a,a.data(),a_count,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_a_scale,a_scale.data(),a_count / 32u,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_w,w.data(),w_count,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_w_scale,w_scale.data(),w_count / 32u,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemsetAsync(device_out,0xff,out_count * 2u,stream));
	PROBE_OK(LmMxGemmDense(device_a,device_a_scale,device_w,device_w_scale,device_out,rows,input,output,0u,0u,probe_multiprocessors,stream));
	PROBE_CUDA(cudaStreamSynchronize(stream));
	PROBE_CUDA(cudaMemcpy(result.data(),device_out,out_count * 2u,cudaMemcpyDeviceToHost));
	for (sample=0u; sample<(out_count < PROBE_SAMPLES ? (uint32_t)out_count : PROBE_SAMPLES); sample++)
	{
		const uint64_t at = out_count <= PROBE_SAMPLES ? sample : ProbeHash(sample + seed) % out_count;
		const uint64_t row = at / output,neuron = at % output;
		double reference = 0.0,magnitude = 0.0,error;
		for (index=0u; index<input; index++)
		{
			const double product = ProbeE4m3(a[row * input + index]) * ldexp(1.0,(int)a_scale[(row * input + index) / 32u] - 127) *
				ProbeE4m3(w[neuron * input + index]) * ldexp(1.0,(int)w_scale[(neuron * input + index) / 32u] - 127);
			reference += product;
			magnitude += fabs(product);
		}
		error = fabs((double)ProbeBf16Float(result[at]) - reference) / (ProbeUlp(reference) + magnitude * PROBE_ACCUMULATION_EPSILON);
		worst = error > worst || !(error == error) ? error : worst;
	}
	if ( worst > PROBE_ULP_LIMIT || !(worst == worst) )
		probe_failures++;
	memset(&args,0,sizeof(args));
	args.activation = device_a;
	args.activation_scale = device_a_scale;
	args.weight = device_w;
	args.weight_scale = device_w_scale;
	args.output_bf16 = device_out;
	args.rows = rows;
	args.input_dimension = input;
	args.output_dimension = output;
	args.output_row_stride = output;
	for (variant=0u; variant<sizeof(launches) / sizeof(launches[0]); variant++)
	{
		PROBE_CUDA(cudaMemsetAsync(device_out,0xff,out_count * 2u,stream));
		PROBE_OK(launches[variant](&args,probe_multiprocessors,stream));
		PROBE_CUDA(cudaStreamSynchronize(stream));
		PROBE_CUDA(cudaMemcpy(other.data(),device_out,out_count * 2u,cudaMemcpyDeviceToHost));
		for (index=0u; index<out_count; index++)
			differ += other[index] != result[index] ? 1u : 0u;
	}
	if ( differ != 0u )
		probe_failures++;
	PROBE_CUDA(cudaFree(device_a));
	PROBE_CUDA(cudaFree(device_a_scale));
	PROBE_CUDA(cudaFree(device_w));
	PROBE_CUDA(cudaFree(device_w_scale));
	PROBE_CUDA(cudaFree(device_out));
	printf("%s mx gemm rows%u K%u N%u: max %.2f bf16 ulp (plus fp32 accumulation allowance) vs the fp64 reference; %llu outputs differ across tile shapes\n",
		failed == probe_failures ? "PASS" : "FAIL",rows,input,output,worst,(unsigned long long)differ);
}

int main(void)
{
	static const uint32_t shape[][3] = {
		{ 17u, 128u, 2u }, { 64u, 256u, 130u }, { 100u, 1280u, 512u }, { 1000u, 5120u, 576u }, { 4096u, 5120u, 1280u }, { 2048u, 2048u, 5120u },
		{ 777u, 6144u, 2000u } };
	cudaStream_t stream;
	uint32_t index;
	int value;
	PROBE_CUDA(cudaDeviceGetAttribute(&value,cudaDevAttrMultiProcessorCount,0));
	probe_multiprocessors = (uint32_t)value;
	PROBE_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	ProbeQuantize(1u << 20u,1u,0u,stream);
	ProbeQuantize(1u << 20u,2u,1u,stream);
	ProbeQuantize(4128u,3u,0u,stream);
	for (index=0u; index<sizeof(shape) / sizeof(shape[0]); index++)
		ProbeGemm(shape[index][0],shape[index][1],shape[index][2],index + 1u,stream);
	PROBE_CUDA(cudaStreamDestroy(stream));
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS mx gemm: the quantizer applies the fp8 qdq rounding and keeps values already on the fp8 grid; the native MXFP8 GEMM stays within %.1f bf16 ulp of the fp64 reference and is bit-identical across tile shapes\n",PROBE_ULP_LIMIT);
	return 0;
}
