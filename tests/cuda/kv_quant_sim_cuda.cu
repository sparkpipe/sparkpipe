#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/kv_quant_sim.cuh"

#define PROBE_HEADS 4u
#define PROBE_KEY 128u
#define PROBE_VALUE 128u
#define PROBE_THREADS 128u
#define PROBE_COLUMNS 32u
#define PROBE_STEPS 16u
#define PROBE_ELEMENTS (PROBE_HEADS * PROBE_KEY * PROBE_VALUE)

static void ProbeCheck(cudaError_t status, const char *what)
{
	if ( status != cudaSuccess )
	{
		fprintf(stderr,"FAIL %s: %s\n",what,cudaGetErrorString(status));
		exit(1);
	}
}

static __global__ void ProbeHardwareCodec(uint32_t *mismatch, uint32_t *compared)
{
	uint32_t bits = (blockIdx.x * blockDim.x) + threadIdx.x,local = 0u,count = 0u;
	float value,sim,hardware;
	if ( bits > 0xffffu )
		return;
	value = LmBf16ToFloat((uint16_t)bits);
	if ( !(fabsf(value) <= LM_E4M3_MAX) )
		return;
	sim = LmKvQuantSimRound(value,SPARK_KV_QUANT_SIM_FP8_E4M3);
	hardware = LmE4m3ToFloat(LmFloatToE4m3(value));
	count++;
	if ( __float_as_uint(sim) != __float_as_uint(hardware) )
		local++;
	if ( fabsf(value) <= LM_E2M1_MAX )
	{
		sim = LmKvQuantSimRound(value,SPARK_KV_QUANT_SIM_MXFP4);
		hardware = LmE2m1PairToFloat(LmFloatPairToE2m1(value,0.0f)).x;
		count++;
		if ( __float_as_uint(sim) != __float_as_uint(hardware) )
			local++;
	}
	atomicAdd(mismatch,local);
	atomicAdd(compared,count);
}

static int RunHardware(void)
{
	uint32_t *device,host[2];
	ProbeCheck(cudaMalloc(&device,sizeof(host)),"malloc");
	ProbeCheck(cudaMemset(device,0,sizeof(host)),"memset");
	ProbeHardwareCodec<<<256,256>>>(device,device + 1);
	ProbeCheck(cudaDeviceSynchronize(),"hardware codec");
	ProbeCheck(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost),"copy");
	printf("hardware_codec_mismatch %u\nhardware_codec_compared %u\n",host[0],host[1]);
	return(0);
}

static int RunRows(void)
{
	uint32_t header[5];
	uint64_t elements;
	int32_t status;
	uint16_t *device = 0;
	std::vector<uint16_t> rows;
	if ( fread(header,sizeof(uint32_t),5,stdin) != 5 )
		return(1);
	elements = (uint64_t)header[0] * header[2];
	rows.resize(elements);
	if ( fread(rows.data(),sizeof(uint16_t),elements,stdin) != elements )
		return(3);
	if ( elements != 0u )
	{
		ProbeCheck(cudaMalloc(&device,elements * 2u),"malloc rows");
		ProbeCheck(cudaMemcpy(device,rows.data(),elements * 2u,cudaMemcpyHostToDevice),"copy rows");
	}
	status = LmKvQuantSimRowsLaunch(device,header[2],header[0],header[1],header[3],header[4],0);
	ProbeCheck(cudaDeviceSynchronize(),"rows kernel");
	if ( elements != 0u )
		ProbeCheck(cudaMemcpy(rows.data(),device,elements * 2u,cudaMemcpyDeviceToHost),"copy back");
	if ( fwrite(&status,sizeof(status),1,stdout) != 1 )
		return(4);
	return(fwrite(rows.data(),sizeof(uint16_t),elements,stdout) == elements ? 0 : 5);
}

static uint16_t ProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static float ProbeRandom(uint32_t *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return((float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f);
}

template<class State>
static void ProbeDelta(uint8_t *pool, const uint32_t *begin, const uint32_t *count, const uint16_t *query, const uint16_t *key, const uint16_t *value, const float *forget, const float *gate, uint16_t *out)
{
	static const uint32_t index[1] = { 0u };
	uint32_t *device_index;
	ProbeCheck(cudaMalloc(&device_index,4u),"malloc index");
	ProbeCheck(cudaMemcpy(device_index,index,4u,cudaMemcpyHostToDevice),"copy index");
	ProbeCheck(cudaFuncSetAttribute((const void *)LmDeltaRuleKernel<PROBE_THREADS,PROBE_KEY,PROBE_VALUE,State,PROBE_COLUMNS>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(PROBE_KEY * PROBE_COLUMNS * 4u)),"attribute");
	LmDeltaRuleKernel<PROBE_THREADS,PROBE_KEY,PROBE_VALUE,State,PROBE_COLUMNS><<<dim3(1u,PROBE_HEADS,PROBE_VALUE / PROBE_COLUMNS),PROBE_THREADS,PROBE_KEY * PROBE_COLUMNS * 4u>>>(
		pool,PROBE_ELEMENTS * 4u,device_index,begin,count,query,key,value,forget,gate,out,PROBE_HEADS,1u,1u,1u,0);
	ProbeCheck(cudaDeviceSynchronize(),"delta rule");
	cudaFree(device_index);
}

template<class T>
static T *ProbeUpload(const std::vector<T> &host)
{
	T *device;
	ProbeCheck(cudaMalloc(&device,host.size() * sizeof(T)),"malloc");
	ProbeCheck(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice),"upload");
	return(device);
}

template<class State>
static void ProbeWaveAndSteps(const std::vector<float> &initial, const std::vector<uint16_t> &query, const std::vector<uint16_t> &key, const std::vector<uint16_t> &value, const std::vector<float> &forget, const std::vector<float> &gate, std::vector<float> *state_wave, std::vector<float> *state_steps, std::vector<uint16_t> *out_wave, std::vector<uint16_t> *out_steps)
{
	uint32_t step,wave_begin[2] = { 0u, PROBE_STEPS },wave_count[1] = { PROBE_STEPS },one_begin[2] = { 0u, 1u },one_count[1] = { 1u };
	uint8_t *pool_wave = (uint8_t *)ProbeUpload(initial),*pool_steps = (uint8_t *)ProbeUpload(initial);
	uint16_t *q = ProbeUpload(query),*k = ProbeUpload(key),*v = ProbeUpload(value),*o;
	float *f = ProbeUpload(forget),*g = ProbeUpload(gate);
	uint32_t *wb = ProbeUpload(std::vector<uint32_t>(wave_begin,wave_begin + 2)),*wc = ProbeUpload(std::vector<uint32_t>(wave_count,wave_count + 1));
	uint32_t *ob = ProbeUpload(std::vector<uint32_t>(one_begin,one_begin + 2)),*oc = ProbeUpload(std::vector<uint32_t>(one_count,one_count + 1));
	uint64_t keys = PROBE_HEADS * PROBE_KEY,values = PROBE_HEADS * PROBE_VALUE;
	ProbeCheck(cudaMalloc(&o,PROBE_STEPS * values * 2u),"malloc out");
	ProbeDelta<State>(pool_wave,wb,wc,q,k,v,f,g,o);
	out_wave->resize(PROBE_STEPS * values);
	ProbeCheck(cudaMemcpy(out_wave->data(),o,PROBE_STEPS * values * 2u,cudaMemcpyDeviceToHost),"out wave");
	for (step = 0u; step < PROBE_STEPS; step++)
		ProbeDelta<State>(pool_steps,ob,oc,q + step * keys,k + step * keys,v + step * values,f + step * keys,g + step * PROBE_HEADS,o + step * values);
	out_steps->resize(PROBE_STEPS * values);
	ProbeCheck(cudaMemcpy(out_steps->data(),o,PROBE_STEPS * values * 2u,cudaMemcpyDeviceToHost),"out steps");
	state_wave->resize(PROBE_ELEMENTS);
	state_steps->resize(PROBE_ELEMENTS);
	ProbeCheck(cudaMemcpy(state_wave->data(),pool_wave,PROBE_ELEMENTS * 4u,cudaMemcpyDeviceToHost),"state wave");
	ProbeCheck(cudaMemcpy(state_steps->data(),pool_steps,PROBE_ELEMENTS * 4u,cudaMemcpyDeviceToHost),"state steps");
	cudaFree(pool_wave); cudaFree(pool_steps); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(f); cudaFree(g); cudaFree(o);
	cudaFree(wb); cudaFree(wc); cudaFree(ob); cudaFree(oc);
}

static uint32_t ProbeDiffer(const void *a, const void *b, size_t elements, size_t width)
{
	uint32_t count = 0u;
	size_t index;
	for (index = 0u; index < elements; index++)
		if ( memcmp((const uint8_t *)a + index * width,(const uint8_t *)b + index * width,width) != 0 )
			count++;
	return(count);
}

static int RunState(void)
{
	uint32_t seed = 1293u,index,off_grid = 0u;
	std::vector<float> initial(PROBE_ELEMENTS),forget(PROBE_STEPS * PROBE_HEADS * PROBE_KEY),gate(PROBE_STEPS * PROBE_HEADS);
	std::vector<uint16_t> query(PROBE_STEPS * PROBE_HEADS * PROBE_KEY),key(query.size()),value(PROBE_STEPS * PROBE_HEADS * PROBE_VALUE);
	std::vector<float> grid_wave,grid_steps,fp32_wave,fp32_steps;
	std::vector<uint16_t> grid_out_wave,grid_out_steps,fp32_out_wave,fp32_out_steps;
	for (index = 0u; index < PROBE_ELEMENTS; index++)
	{
		uint32_t bits = (uint32_t)ProbeBf16(0.05f * ProbeRandom(&seed)) << 16u;
		memcpy(&initial[index],&bits,4u);
	}
	for (index = 0u; index < query.size(); index++)
	{
		query[index] = ProbeBf16(ProbeRandom(&seed));
		key[index] = ProbeBf16(ProbeRandom(&seed));
		forget[index] = 0.9f + 0.09f * ProbeRandom(&seed);
	}
	for (index = 0u; index < value.size(); index++)
		value[index] = ProbeBf16(2.0f * ProbeRandom(&seed));
	for (index = 0u; index < gate.size(); index++)
		gate[index] = 0.5f + 0.4f * ProbeRandom(&seed);
	ProbeWaveAndSteps<LmKvStateBf16Grid>(initial,query,key,value,forget,gate,&grid_wave,&grid_steps,&grid_out_wave,&grid_out_steps);
	ProbeWaveAndSteps<float>(initial,query,key,value,forget,gate,&fp32_wave,&fp32_steps,&fp32_out_wave,&fp32_out_steps);
	for (index = 0u; index < PROBE_ELEMENTS; index++)
	{
		uint32_t bits;
		memcpy(&bits,&grid_wave[index],4u);
		if ( (bits & 0xffffu) != 0u )
			off_grid++;
	}
	printf("grid_state_wave_vs_steps %u\n",ProbeDiffer(grid_wave.data(),grid_steps.data(),PROBE_ELEMENTS,4u));
	printf("grid_out_wave_vs_steps %u\n",ProbeDiffer(grid_out_wave.data(),grid_out_steps.data(),grid_out_wave.size(),2u));
	printf("grid_state_off_bf16_grid %u\n",off_grid);
	printf("fp32_state_wave_vs_steps %u\n",ProbeDiffer(fp32_wave.data(),fp32_steps.data(),PROBE_ELEMENTS,4u));
	printf("grid_vs_fp32_state_differs %u\n",ProbeDiffer(grid_wave.data(),fp32_wave.data(),PROBE_ELEMENTS,4u));
	return(0);
}

int main(int argc, char **argv)
{
	if ( argc == 2 && strcmp(argv[1],"rows") == 0 )
		return(RunRows());
	if ( argc == 2 && strcmp(argv[1],"hardware") == 0 )
		return(RunHardware());
	if ( argc == 2 && strcmp(argv[1],"state") == 0 )
		return(RunState());
	fprintf(stderr,"usage: %s rows|hardware|state\n",argv[0]);
	return(9);
}
