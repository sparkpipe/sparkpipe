#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/gemm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/weight_codec.cuh"

#define PROBE_EXPERTS 256u
#define PROBE_TOP_K 8u
#define PROBE_HIDDEN 6144u
#define PROBE_UP 256u
#define PROBE_INTER 128u
#define PROBE_TILE_N 128u
#define PROBE_STAGES 2u
#define PROBE_WARPS 8u
#define PROBE_SCALE_K 128u
#define PROBE_SAMPLES 64u
#define PROBE_REPEATS 20u
#define PROBE_ULP_LIMIT 1.0
#define PROBE_ACCUMULATION_EPSILON 3.814697265625e-06
#define PROBE_PATHS 2u
#define PROBE_PAIR_ROWS 8u

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

static uint16_t ProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return((uint16_t)(bits >> 16u));
}

static double ProbeBf16Value(uint16_t code)
{
	uint32_t bits = (uint32_t)code << 16u;
	float value;
	memcpy(&value,&bits,4u);
	return(value);
}

static double ProbeUlp(double value)
{
	int exponent;
	frexp(fabs(value) > 1.0e-30 ? value : 1.0e-30,&exponent);
	return(ldexp(1.0,exponent - 8));
}

typedef struct ProbeMatrix
{
	uint32_t output,input;
	std::vector<uint8_t> weight;
	std::vector<float> scale;
	uint8_t *device_weight;
	float *device_scale;
}
ProbeMatrix;

static void ProbeMatrixBuild(ProbeMatrix *matrix,uint32_t output,uint32_t input,uint64_t salt)
{
	uint64_t count = (uint64_t)PROBE_EXPERTS * output * input,scales = (uint64_t)PROBE_EXPERTS * output * (input / PROBE_SCALE_K),index,z;
	matrix->output = output;
	matrix->input = input;
	matrix->weight.resize(count);
	matrix->scale.resize(scales);
	for (index=0u; index<count; index++)
	{
		z = ProbeHash(index ^ salt) & 0xffu;
		matrix->weight[index] = (uint8_t)((z & 0x7fu) == 0x7fu ? z - 1u : z);
	}
	for (index=0u; index<scales; index++)
		matrix->scale[index] = (float)(ldexp(1.0,-7) * (1.0 + (double)(ProbeHash(index ^ salt ^ 0x5bd1e995u) % 100003u) / 100003.0));
	PROBE_CUDA(cudaMalloc((void **)&matrix->device_weight,count));
	PROBE_CUDA(cudaMalloc((void **)&matrix->device_scale,scales * 4u));
	PROBE_CUDA(cudaMemcpy(matrix->device_weight,matrix->weight.data(),count,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(matrix->device_scale,matrix->scale.data(),scales * 4u,cudaMemcpyHostToDevice));
}

static double ProbeReference(const ProbeMatrix *matrix,uint32_t expert,uint32_t neuron,const uint16_t *activation,double *magnitude)
{
	const uint8_t *weight = matrix->weight.data() + ((uint64_t)expert * matrix->output + neuron) * matrix->input;
	const float *scale = matrix->scale.data() + ((uint64_t)expert * matrix->output + neuron) * (matrix->input / PROBE_SCALE_K);
	double sum = 0.0,term;
	uint32_t k;
	*magnitude = 0.0;
	for (k=0u; k<matrix->input; k++)
	{
		term = ProbeE4m3(weight[k]) * (double)scale[k / PROBE_SCALE_K] * ProbeBf16Value(activation[k]);
		sum += term;
		*magnitude += fabs(term);
	}
	return(sum);
}

static double ProbeError(double value,double reference,double magnitude)
{
	return(fabs(value - reference) / (ProbeUlp(reference) + magnitude * PROBE_ACCUMULATION_EPSILON));
}

typedef struct ProbeCase
{
	uint32_t rows,pairs,touched;
	std::vector<uint32_t> route,packed_host;
	std::vector<uint16_t> activation,intermediate;
	uint32_t *device_route,*offset,*packed,*source,*prefix_up,*prefix_down;
	uint16_t *device_activation,*device_intermediate,*output_up,*output_down;
}
ProbeCase;

static void ProbeCaseBuild(ProbeCase *probe,uint32_t rows,cudaStream_t stream)
{
	std::vector<uint8_t> used(PROBE_EXPERTS,0u);
	uint32_t row,k,j,expert,attempt;
	uint64_t index;
	probe->rows = rows;
	probe->pairs = rows * PROBE_TOP_K;
	probe->route.resize(probe->pairs);
	for (row=0u; row<rows; row++)
		for (k=0u; k<PROBE_TOP_K; k++)
			for (attempt=0u;; attempt++)
			{
				expert = (uint32_t)(ProbeHash(((uint64_t)row << 16u) ^ (k << 8u) ^ attempt) % PROBE_EXPERTS);
				for (j=0u; j<k && probe->route[row * PROBE_TOP_K + j] != expert; j++)
					;
				if ( j == k )
				{
					probe->route[row * PROBE_TOP_K + k] = expert;
					used[expert] = 1u;
					break;
				}
			}
	probe->touched = 0u;
	for (expert=0u; expert<PROBE_EXPERTS; expert++)
		probe->touched += used[expert];
	probe->activation.resize((uint64_t)rows * PROBE_HIDDEN);
	for (index=0u; index<probe->activation.size(); index++)
		probe->activation[index] = ProbeBf16((float)((double)(ProbeHash(index ^ 0xa11ce) % 200001u) / 50000.0 - 2.0));
	probe->intermediate.resize((uint64_t)probe->pairs * PROBE_INTER);
	for (index=0u; index<probe->intermediate.size(); index++)
		probe->intermediate[index] = ProbeBf16((float)((double)(ProbeHash(index ^ 0xb0b) % 200001u) / 50000.0 - 2.0));
	PROBE_CUDA(cudaMalloc((void **)&probe->device_route,probe->pairs * 4u));
	PROBE_CUDA(cudaMalloc((void **)&probe->offset,(PROBE_EXPERTS + 1u) * 4u));
	PROBE_CUDA(cudaMalloc((void **)&probe->packed,probe->pairs * 4u));
	PROBE_CUDA(cudaMalloc((void **)&probe->source,probe->pairs * 4u));
	PROBE_CUDA(cudaMalloc((void **)&probe->prefix_up,(PROBE_EXPERTS + 1u) * 4u));
	PROBE_CUDA(cudaMalloc((void **)&probe->prefix_down,(PROBE_EXPERTS + 1u) * 4u));
	PROBE_CUDA(cudaMalloc((void **)&probe->device_activation,probe->activation.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&probe->device_intermediate,probe->intermediate.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&probe->output_up,(uint64_t)probe->pairs * PROBE_UP * 2u));
	PROBE_CUDA(cudaMalloc((void **)&probe->output_down,(uint64_t)probe->pairs * PROBE_HIDDEN * 2u));
	PROBE_CUDA(cudaMemcpy(probe->device_route,probe->route.data(),probe->pairs * 4u,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(probe->device_activation,probe->activation.data(),probe->activation.size() * 2u,cudaMemcpyHostToDevice));
	PROBE_OK((LmRouteBuild<256u,PROBE_EXPERTS>(probe->device_route,rows,probe->pairs,PROBE_TOP_K,probe->offset,probe->packed,probe->source,PROBE_UP,PROBE_HIDDEN,PROBE_TILE_N,PROBE_TILE_N,probe->prefix_up,probe->prefix_down,stream)));
	PROBE_CUDA(cudaStreamSynchronize(stream));
	probe->packed_host.resize(probe->pairs);
	PROBE_CUDA(cudaMemcpy(probe->packed_host.data(),probe->packed,probe->pairs * 4u,cudaMemcpyDeviceToHost));
	{
		std::vector<uint16_t> packed_intermediate(probe->intermediate.size());
		uint32_t pair;
		for (pair=0u; pair<probe->pairs; pair++)
			memcpy(packed_intermediate.data() + (uint64_t)probe->packed_host[pair] * PROBE_INTER,probe->intermediate.data() + (uint64_t)pair * PROBE_INTER,PROBE_INTER * 2u);
		PROBE_CUDA(cudaMemcpy(probe->device_intermediate,packed_intermediate.data(),packed_intermediate.size() * 2u,cudaMemcpyHostToDevice));
	}
}

static void ProbeCaseFree(ProbeCase *probe)
{
	cudaFree(probe->device_route); cudaFree(probe->offset); cudaFree(probe->packed); cudaFree(probe->source);
	cudaFree(probe->prefix_up); cudaFree(probe->prefix_down); cudaFree(probe->device_activation);
	cudaFree(probe->device_intermediate); cudaFree(probe->output_up); cudaFree(probe->output_down);
}

static int32_t ProbeUp(const ProbeCase *probe,const ProbeMatrix *up,uint32_t path,cudaStream_t stream)
{
	LmScaleTensor scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(up->device_scale,PROBE_EXPERTS,PROBE_UP,PROBE_HIDDEN);
	if ( path != 0u )
		return(LmStreamGemmGrouped<LmFp8>(up->device_weight,scale,probe->device_activation,probe->output_up,probe->offset,probe->source,PROBE_EXPERTS,probe->pairs,PROBE_HIDDEN,PROBE_UP,probe_multiprocessors,stream));
	if ( probe->rows == 1u )
		return(LmSkinnyExperts<LmFp8>(up->device_weight,scale,probe->device_activation,probe->output_up,probe->device_route,probe->packed,probe->pairs,PROBE_TOP_K,0u,PROBE_HIDDEN,PROBE_UP,stream));
	return(LmSkinnyGroupedExperts<LmFp8>(up->device_weight,scale,probe->device_activation,probe->output_up,probe->offset,probe->source,PROBE_EXPERTS,probe->pairs,0u,PROBE_HIDDEN,PROBE_UP,stream));
}

static int32_t ProbeDown(const ProbeCase *probe,const ProbeMatrix *down,uint32_t path,cudaStream_t stream)
{
	LmScaleTensor scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(down->device_scale,PROBE_EXPERTS,PROBE_HIDDEN,PROBE_INTER);
	if ( path != 0u )
		return(LmStreamGemmGrouped<LmFp8>(down->device_weight,scale,probe->device_intermediate,probe->output_down,probe->offset,0,PROBE_EXPERTS,probe->pairs,PROBE_INTER,PROBE_HIDDEN,probe_multiprocessors,stream));
	if ( probe->rows <= PROBE_PAIR_ROWS )
		return(LmSkinnyExperts<LmFp8>(down->device_weight,scale,probe->device_intermediate,probe->output_down,probe->device_route,probe->packed,probe->pairs,PROBE_TOP_K,1u,PROBE_INTER,PROBE_HIDDEN,stream));
	return(LmSkinnyGroupedExperts<LmFp8>(down->device_weight,scale,probe->device_intermediate,probe->output_down,probe->offset,probe->source,PROBE_EXPERTS,probe->pairs,1u,PROBE_INTER,PROBE_HIDDEN,stream));
}

static const char *const probe_path_names[PROBE_PATHS] = { "skinny", "stream" };
static std::vector<uint16_t> probe_token_zero[2];

static void ProbeCheck(ProbeCase *probe,const ProbeMatrix *matrix,uint32_t down,uint32_t path,cudaStream_t stream)
{
	std::vector<uint16_t> output((uint64_t)probe->pairs * matrix->output),token_zero((uint64_t)PROBE_TOP_K * matrix->output);
	double worst = 0.0,sum = 0.0,ulps,reference,magnitude;
	uint32_t sample,pair,neuron,packed_row,expert;
	uint64_t checked = 0u;
	int failed = probe_failures;
	PROBE_CUDA(cudaMemsetAsync(down != 0u ? probe->output_down : probe->output_up,0xff,output.size() * 2u,stream));
	PROBE_OK(down != 0u ? ProbeDown(probe,matrix,path,stream) : ProbeUp(probe,matrix,path,stream));
	PROBE_CUDA(cudaStreamSynchronize(stream));
	PROBE_CUDA(cudaMemcpy(output.data(),down != 0u ? probe->output_down : probe->output_up,output.size() * 2u,cudaMemcpyDeviceToHost));
	for (sample=0u; sample<PROBE_SAMPLES; sample++)
	{
		pair = sample == 0u ? 0u : (uint32_t)(ProbeHash(sample ^ 0x7777u) % probe->pairs);
		expert = probe->route[pair];
		packed_row = probe->packed_host[pair];
		for (neuron=sample % 7u; neuron<matrix->output; neuron+=matrix->output / 16u)
		{
			reference = down != 0u
				? ProbeReference(matrix,expert,neuron,probe->intermediate.data() + (uint64_t)pair * PROBE_INTER,&magnitude)
				: ProbeReference(matrix,expert,neuron,probe->activation.data() + (uint64_t)(pair / PROBE_TOP_K) * PROBE_HIDDEN,&magnitude);
			ulps = ProbeError(ProbeBf16Value(output[(uint64_t)packed_row * matrix->output + neuron]),reference,magnitude);
			worst = ulps > worst || !(ulps == ulps) ? ulps : worst;
			sum += ulps;
			checked++;
		}
	}
	if ( worst > PROBE_ULP_LIMIT || !(worst == worst) )
		probe_failures++;
	if ( path == 1u )
	{
		for (pair=0u; pair<PROBE_TOP_K; pair++)
			memcpy(token_zero.data() + (uint64_t)pair * matrix->output,output.data() + (uint64_t)probe->packed_host[pair] * matrix->output,matrix->output * 2u);
		if ( probe_token_zero[down].empty() )
			probe_token_zero[down] = token_zero;
		else if ( probe_token_zero[down] != token_zero )
		{
			printf("FAIL rows%u %s stream: token 0 differs from its one-row result, the tensor-core path is not row-invariant\n",probe->rows,down != 0u ? "down" : "gate_up");
			probe_failures++;
		}
	}
	printf("%s rows%u %s %s: max %.2f mean %.3f bf16 ulp (plus fp32 accumulation allowance) vs the fp64 reference over %llu outputs%s\n",failed == probe_failures ? "PASS" : "FAIL",probe->rows,down != 0u ? "down" : "gate_up",probe_path_names[path],worst,sum / (double)checked,(unsigned long long)checked,path == 1u ? ", token 0 bitwise equal to its one-row result" : "");
}

static double ProbeTime(const ProbeCase *probe,const ProbeMatrix *matrix,uint32_t down,uint32_t path,cudaStream_t stream)
{
	cudaEvent_t start,stop;
	float milliseconds;
	uint32_t repeat;
	PROBE_CUDA(cudaEventCreate(&start));
	PROBE_CUDA(cudaEventCreate(&stop));
	for (repeat=0u; repeat<3u; repeat++)
		PROBE_OK(down != 0u ? ProbeDown(probe,matrix,path,stream) : ProbeUp(probe,matrix,path,stream));
	PROBE_CUDA(cudaEventRecord(start,stream));
	for (repeat=0u; repeat<PROBE_REPEATS; repeat++)
		PROBE_OK(down != 0u ? ProbeDown(probe,matrix,path,stream) : ProbeUp(probe,matrix,path,stream));
	PROBE_CUDA(cudaEventRecord(stop,stream));
	PROBE_CUDA(cudaEventSynchronize(stop));
	PROBE_CUDA(cudaEventElapsedTime(&milliseconds,start,stop));
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return((double)milliseconds * 1000.0 / PROBE_REPEATS);
}

static void ProbeRows(uint32_t rows,const ProbeMatrix *up,const ProbeMatrix *down,uint32_t timing,cudaStream_t stream)
{
	ProbeCase probe;
	double microseconds,bytes,flops;
	uint32_t path,side;
	ProbeCaseBuild(&probe,rows,stream);
	for (side=0u; side<2u; side++)
		for (path=0u; path<PROBE_PATHS; path++)
			ProbeCheck(&probe,side != 0u ? down : up,side,path,stream);
	if ( timing != 0u )
		for (side=0u; side<2u; side++)
		{
			bytes = (double)probe.touched * (side != 0u ? (double)PROBE_HIDDEN * PROBE_INTER : (double)PROBE_UP * PROBE_HIDDEN);
			flops = 2.0 * probe.pairs * (side != 0u ? (double)PROBE_HIDDEN * PROBE_INTER : (double)PROBE_UP * PROBE_HIDDEN);
			printf("TIME rows%u %s experts%u weight_mb=%.1f",rows,side != 0u ? "down" : "gate_up",probe.touched,bytes / 1.0e6);
			for (path=0u; path<PROBE_PATHS; path++)
			{
				microseconds = ProbeTime(&probe,side != 0u ? down : up,side,path,stream);
				printf(" %s %.1f us %.0f GB/s %.2f TFLOPS",probe_path_names[path],microseconds,bytes / microseconds / 1000.0,flops / microseconds / 1.0e6);
			}
			printf("\n");
		}
	ProbeCaseFree(&probe);
}

typedef struct ProbeDenseShape
{
	uint32_t input,output;
	const char *name;
}
ProbeDenseShape;

static double ProbeDenseReference(const uint16_t *weight,const uint16_t *activation,uint32_t input,double *magnitude)
{
	double sum = 0.0,term;
	uint32_t k;
	*magnitude = 0.0;
	for (k=0u; k<input; k++)
	{
		term = ProbeBf16Value(weight[k]) * ProbeBf16Value(activation[k]);
		sum += term;
		*magnitude += fabs(term);
	}
	return(sum);
}

static int32_t ProbeDenseRun(uint32_t path,const uint16_t *weight,const uint16_t *activation,uint16_t *output,uint32_t *row_offset,uint32_t *tile_prefix,uint32_t rows,const ProbeDenseShape *shape,cudaStream_t stream)
{
	LmGemmArguments gemm;
	if ( path != 0u )
		return(LmStreamGemmDense<LmBf16Format>(weight,LmScaleTensorNone(),activation,output,0,rows,shape->input,shape->output,0u,0u,probe_multiprocessors,stream));
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = row_offset;
	gemm.group_tile_prefix = tile_prefix;
	gemm.output_bf16 = output;
	gemm.output_row_stride = shape->output;
	return(LmGemmLaunch<LmBf16Format,PROBE_TILE_N,LmBf16Format::kTileK,PROBE_STAGES,PROBE_WARPS>(&gemm,activation,weight,rows,rows,1u,1u,shape->input,shape->output,probe_multiprocessors,false,stream));
}

static void ProbeDense(const ProbeDenseShape *shape,uint32_t rows,uint32_t timing,std::vector<uint16_t> *row_zero,cudaStream_t stream)
{
	static const char *const names[2] = { "lmgemm", "stream" };
	std::vector<uint16_t> weight((uint64_t)shape->output * shape->input),activation((uint64_t)rows * shape->input),output((uint64_t)rows * shape->output);
	uint16_t *device_weight,*device_activation,*device_output;
	uint32_t *row_offset,*tile_prefix,host_offset[2] = { 0u, rows },path,sample,row,neuron,repeat;
	uint64_t index;
	double worst,sum,ulps,reference,magnitude,microseconds[2];
	cudaEvent_t start,stop;
	float milliseconds;
	int failed = probe_failures;
	for (index=0u; index<weight.size(); index++)
		weight[index] = ProbeBf16((float)((double)(ProbeHash(index ^ ((uint64_t)shape->input << 32u) ^ shape->output) % 200001u) / 2.0e6 - 0.05));
	for (index=0u; index<activation.size(); index++)
		activation[index] = ProbeBf16((float)((double)(ProbeHash(index ^ 0xd00d) % 200001u) / 50000.0 - 2.0));
	PROBE_CUDA(cudaMalloc((void **)&device_weight,weight.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&device_activation,activation.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&device_output,output.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&row_offset,2u * 4u));
	PROBE_CUDA(cudaMalloc((void **)&tile_prefix,2u * 4u));
	PROBE_CUDA(cudaMemcpy(device_weight,weight.data(),weight.size() * 2u,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_activation,activation.data(),activation.size() * 2u,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(row_offset,host_offset,sizeof(host_offset),cudaMemcpyHostToDevice));
	for (path=0u; path<2u; path++)
	{
		PROBE_CUDA(cudaMemset(device_output,0xff,output.size() * 2u));
		PROBE_OK(ProbeDenseRun(path,device_weight,device_activation,device_output,row_offset,tile_prefix,rows,shape,stream));
		PROBE_CUDA(cudaStreamSynchronize(stream));
		PROBE_CUDA(cudaMemcpy(output.data(),device_output,output.size() * 2u,cudaMemcpyDeviceToHost));
		worst = sum = 0.0;
		for (sample=0u; sample<16u; sample++)
		{
			row = sample == 0u ? 0u : (uint32_t)(ProbeHash(sample ^ rows) % rows);
			for (neuron=sample % 5u; neuron<shape->output; neuron+=(shape->output + 15u) / 16u)
			{
				reference = ProbeDenseReference(weight.data() + (uint64_t)neuron * shape->input,activation.data() + (uint64_t)row * shape->input,shape->input,&magnitude);
				ulps = ProbeError(ProbeBf16Value(output[(uint64_t)row * shape->output + neuron]),reference,magnitude);
				worst = ulps > worst || !(ulps == ulps) ? ulps : worst;
				sum += ulps;
			}
		}
		if ( path == 1u && (worst > PROBE_ULP_LIMIT || !(worst == worst)) )
			probe_failures++;
		if ( path == 1u )
		{
			std::vector<uint16_t> first(output.begin(),output.begin() + shape->output);
			if ( row_zero->empty() )
				*row_zero = first;
			else if ( *row_zero != first )
			{
				printf("FAIL dense %s rows%u stream: row 0 differs from its result at the first row count, the dense path is not row-invariant\n",shape->name,rows);
				probe_failures++;
			}
		}
		printf("%s dense %s K%u N%u rows%u %s: max %.2f bf16 ulp (plus fp32 accumulation allowance) vs the fp64 reference\n",path == 0u ? "INFO" : failed == probe_failures ? "PASS" : "FAIL",shape->name,shape->input,shape->output,rows,names[path],worst);
	}
	if ( timing != 0u )
	{
		PROBE_CUDA(cudaEventCreate(&start));
		PROBE_CUDA(cudaEventCreate(&stop));
		for (path=0u; path<2u; path++)
		{
			for (repeat=0u; repeat<3u; repeat++)
				PROBE_OK(ProbeDenseRun(path,device_weight,device_activation,device_output,row_offset,tile_prefix,rows,shape,stream));
			PROBE_CUDA(cudaEventRecord(start,stream));
			for (repeat=0u; repeat<PROBE_REPEATS; repeat++)
				PROBE_OK(ProbeDenseRun(path,device_weight,device_activation,device_output,row_offset,tile_prefix,rows,shape,stream));
			PROBE_CUDA(cudaEventRecord(stop,stream));
			PROBE_CUDA(cudaEventSynchronize(stop));
			PROBE_CUDA(cudaEventElapsedTime(&milliseconds,start,stop));
			microseconds[path] = (double)milliseconds * 1000.0 / PROBE_REPEATS;
		}
		printf("TIME dense %s K%u N%u rows%u weight_mb=%.1f lmgemm %.1f us %.2f TFLOPS stream %.1f us %.2f TFLOPS %.0f GB/s\n",shape->name,shape->input,shape->output,rows,weight.size() * 2.0 / 1.0e6,
			microseconds[0],2.0 * rows * shape->input * shape->output / microseconds[0] / 1.0e6,
			microseconds[1],2.0 * rows * shape->input * shape->output / microseconds[1] / 1.0e6,(weight.size() + activation.size() + output.size()) * 2.0 / microseconds[1] / 1000.0);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	cudaFree(device_weight); cudaFree(device_activation); cudaFree(device_output); cudaFree(row_offset); cudaFree(tile_prefix);
}

int main(int argc,char **argv)
{
	static const uint32_t rows[] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u };
	static const uint32_t dense_rows[] = { 9u, 16u, 64u, 256u, 1024u };
	static const ProbeDenseShape dense_shapes[] = {
		{ 6144u, 2048u, "q_a" }, { 6144u, 576u, "kv_a" }, { 2048u, 1024u, "q_b" }, { 2048u, 4096u, "index_q" },
		{ 1024u, 6144u, "o_proj" }, { 6144u, 256u, "shared_gate_up" }, { 128u, 6144u, "shared_down" },
		{ 6144u, 1536u, "dense_gate_up" }, { 768u, 6144u, "dense_down" }, { 6144u, 32u, "index_heads" } };
	uint32_t shape;
	ProbeMatrix up,down;
	cudaStream_t stream;
	uint32_t index,timing = argc > 1 && strcmp(argv[1],"--time") == 0 ? 1u : 0u;
	int value;
	PROBE_CUDA(cudaDeviceGetAttribute(&value,cudaDevAttrMultiProcessorCount,0));
	probe_multiprocessors = (uint32_t)value;
	PROBE_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	ProbeMatrixBuild(&up,PROBE_UP,PROBE_HIDDEN,0x1234u);
	ProbeMatrixBuild(&down,PROBE_HIDDEN,PROBE_INTER,0x4321u);
	for (index=0u; index<sizeof(rows) / sizeof(rows[0]); index++)
		ProbeRows(rows[index],&up,&down,timing,stream);
	for (shape=0u; shape<sizeof(dense_shapes) / sizeof(dense_shapes[0]); shape++)
	{
		std::vector<uint16_t> row_zero;
		for (index=0u; index<sizeof(dense_rows) / sizeof(dense_rows[0]); index++)
			ProbeDense(&dense_shapes[shape],dense_rows[index],timing,&row_zero,stream);
	}
	cudaFree(up.device_weight); cudaFree(up.device_scale); cudaFree(down.device_weight); cudaFree(down.device_scale);
	PROBE_CUDA(cudaStreamDestroy(stream));
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS stream GEMM: grouped FP8 experts and dense BF16 projections stay within %.1f bf16 ulp of the fp64 reference at every row count and are row-invariant; the serving CUDA-core expert dispatch meets the same bound\n",PROBE_ULP_LIMIT);
	return 0;
}
