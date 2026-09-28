#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/kernels/skinny.cuh"

#define PROBE_EXPERTS 32u
#define PROBE_TOP_K 8u
#define PROBE_HIDDEN 4096u
#define PROBE_INTER 2048u
#define PROBE_REPEATS 50u

static const double probe_e2m1[16] = {0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0};
static uint32_t probe_state = 121u;

static uint32_t ProbeRandom(void)
{
	probe_state = probe_state * 1664525u + 1013904223u;
	return(probe_state >> 8u);
}

static uint16_t ProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static float ProbeFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out,&bits,sizeof(out));
	return(out);
}

static int ProbeCheck(cudaError_t status, const char *what)
{
	if ( status != cudaSuccess )
	{
		fprintf(stderr,"FAIL %s: %s\n",what,cudaGetErrorString(status));
		exit(1);
	}
	return(0);
}

static void ProbeCase(uint32_t input, uint32_t output)
{
	std::vector<uint8_t> payload((size_t)PROBE_EXPERTS * output * input / 2u), scale((size_t)PROBE_EXPERTS * output * input / 32u);
	std::vector<uint16_t> activation(input), result((size_t)PROBE_TOP_K * output);
	std::vector<uint32_t> expert(PROBE_TOP_K), packed(PROBE_TOP_K);
	uint8_t *device_payload, *device_scale;
	uint16_t *device_activation, *device_output;
	uint32_t *device_expert, *device_packed;
	cudaEvent_t start, stop;
	float milliseconds;
	double total, error, worst = 0.0, peak = 0.0, bytes;
	uint32_t index, pair, row, column;
	uint8_t byte;
	for (index=0u; index<payload.size(); index++)
		payload[index] = (uint8_t)ProbeRandom();
	for (index=0u; index<scale.size(); index++)
		scale[index] = (uint8_t)(118u + ProbeRandom() % 12u);
	for (index=0u; index<input; index++)
		activation[index] = ProbeBf16(((float)(ProbeRandom() % 2001u) - 1000.0f) / 1000.0f);
	for (pair=0u; pair<PROBE_TOP_K; pair++)
	{
		expert[pair] = (pair * 7u + 3u) % PROBE_EXPERTS;
		packed[pair] = pair;
	}
	ProbeCheck(cudaMalloc(&device_payload,payload.size()),"malloc payload");
	ProbeCheck(cudaMalloc(&device_scale,scale.size()),"malloc scale");
	ProbeCheck(cudaMalloc(&device_activation,activation.size() * 2u),"malloc activation");
	ProbeCheck(cudaMalloc(&device_output,result.size() * 2u),"malloc output");
	ProbeCheck(cudaMalloc(&device_expert,PROBE_TOP_K * 4u),"malloc expert");
	ProbeCheck(cudaMalloc(&device_packed,PROBE_TOP_K * 4u),"malloc packed");
	ProbeCheck(cudaMemcpy(device_payload,payload.data(),payload.size(),cudaMemcpyHostToDevice),"copy payload");
	ProbeCheck(cudaMemcpy(device_scale,scale.data(),scale.size(),cudaMemcpyHostToDevice),"copy scale");
	ProbeCheck(cudaMemcpy(device_activation,activation.data(),activation.size() * 2u,cudaMemcpyHostToDevice),"copy activation");
	ProbeCheck(cudaMemcpy(device_expert,expert.data(),PROBE_TOP_K * 4u,cudaMemcpyHostToDevice),"copy expert");
	ProbeCheck(cudaMemcpy(device_packed,packed.data(),PROBE_TOP_K * 4u,cudaMemcpyHostToDevice),"copy packed");
	LmScaleTensor scales = LmScaleTensorBlockUe8m0(device_scale,PROBE_EXPERTS,output,input,1u,32u);
	if ( LmSkinnyExperts<LmMxfp4>(device_payload,scales,device_activation,device_output,device_expert,device_packed,PROBE_TOP_K,PROBE_TOP_K,0u,input,output,0) != LM_LAUNCH_OK )
	{
		fprintf(stderr,"FAIL mxfp4 skinny launch refused input=%u output=%u\n",input,output);
		exit(1);
	}
	ProbeCheck(cudaDeviceSynchronize(),"skinny run");
	ProbeCheck(cudaMemcpy(result.data(),device_output,result.size() * 2u,cudaMemcpyDeviceToHost),"copy result");
	for (pair=0u; pair<PROBE_TOP_K; pair++)
		for (row=0u; row<output; row++)
		{
			total = 0.0;
			for (column=0u; column<input; column++)
			{
				byte = payload[((size_t)expert[pair] * output + row) * (input / 2u) + column / 2u];
				total += probe_e2m1[(column & 1u) != 0u ? byte >> 4u : byte & 15u] * ldexp(1.0,(int)scale[((size_t)expert[pair] * output + row) * (input / 32u) + column / 32u] - 127) * ProbeFloat(activation[column]);
			}
			error = fabs(total - ProbeFloat(result[(size_t)pair * output + row]));
			worst = error > worst ? error : worst;
			peak = fabs(total) > peak ? fabs(total) : peak;
		}
	if ( worst > peak * 8.0e-3 )
	{
		fprintf(stderr,"FAIL mxfp4 skinny on device vs f64 reference input=%u output=%u worst=%.6f peak=%.3f\n",input,output,worst,peak);
		exit(1);
	}
	ProbeCheck(cudaEventCreate(&start),"event");
	ProbeCheck(cudaEventCreate(&stop),"event");
	ProbeCheck(cudaEventRecord(start,0),"record");
	for (index=0u; index<PROBE_REPEATS; index++)
		LmSkinnyExperts<LmMxfp4>(device_payload,scales,device_activation,device_output,device_expert,device_packed,PROBE_TOP_K,PROBE_TOP_K,0u,input,output,0);
	ProbeCheck(cudaEventRecord(stop,0),"record");
	ProbeCheck(cudaEventSynchronize(stop),"sync");
	ProbeCheck(cudaEventElapsedTime(&milliseconds,start,stop),"elapsed");
	bytes = (double)PROBE_TOP_K * output * input * (0.5 + 1.0 / 32.0);
	printf("mxfp4 skinny device input=%u output=%u top_k=%u worst_abs=%.6f peak=%.3f time_us=%.1f weight_GBps=%.1f\n",input,output,PROBE_TOP_K,worst,peak,1000.0 * milliseconds / PROBE_REPEATS,bytes * PROBE_REPEATS / (milliseconds * 1.0e6));
	cudaFree(device_payload);
	cudaFree(device_scale);
	cudaFree(device_activation);
	cudaFree(device_output);
	cudaFree(device_expert);
	cudaFree(device_packed);
}

int main(void)
{
	ProbeCase(PROBE_HIDDEN,PROBE_INTER);
	ProbeCase(PROBE_INTER,PROBE_HIDDEN);
	puts("PASS skinny mxfp4 on device: 4096x2048 expert shapes match an f64 dequantised reference");
	return(0);
}
