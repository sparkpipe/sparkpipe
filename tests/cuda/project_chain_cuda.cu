#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/project_chain.cuh"

#define PROBE_HEADS 4u
#define PROBE_REPEATS 200u

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

__global__ void ProbeFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	uint64_t index,z;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z ^= z >> 31u;
		values[index] = (uint16_t)(0x3c00u + (z & 0x3ffu) + ((z >> 20u) & 1u) * 0x8000u - ((z >> 12u) & 7u) * 0x80u);
	}
}

static double ProbeBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out,&bits,sizeof(out));
	return (double)out;
}

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET>
static void ProbeReference(const char *label, const uint16_t *input_device, const uint16_t *weight_device, uint32_t rows, const std::vector<uint16_t> &tile,
	const std::vector<uint16_t> &chain)
{
	std::vector<uint16_t> input((uint64_t)rows * PROBE_HEADS * INPUT_HEAD_DIM),weight((uint64_t)PROBE_HEADS * OUT_DIM * IN_DIM);
	uint64_t worse = 0u,checked = 0u,apart = 0u;
	double worst = 0.0;
	PROBE_CUDA(cudaMemcpy(input.data(),input_device,input.size() * 2u,cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(weight.data(),weight_device,weight.size() * 2u,cudaMemcpyDeviceToHost));
	for (uint32_t row=0u; row<rows; row++)
		for (uint32_t head=0u; head<PROBE_HEADS; head++)
			for (uint32_t out=0u; out<OUT_DIM; out++)
			{
				const uint64_t at = ((uint64_t)row * PROBE_HEADS + head) * OUT_DIM + out;
				double exact = 0.0,magnitude = 0.0,error,bound;
				for (uint32_t k=0u; k<IN_DIM; k++)
				{
					const double product = ProbeBf16(input[((uint64_t)row * PROBE_HEADS + head) * INPUT_HEAD_DIM + INPUT_OFFSET + k]) *
						ProbeBf16(weight[((uint64_t)head * OUT_DIM + out) * IN_DIM + k]);
					exact += product;
					magnitude += product < 0.0 ? -product : product;
				}
				error = ProbeBf16(chain[at]) - exact;
				error = error < 0.0 ? -error : error;
				bound = (exact < 0.0 ? -exact : exact) / 128.0 + magnitude * 1e-6;
				worse += error > bound ? 1u : 0u;
				worst = error / (bound > 0.0 ? bound : 1.0) > worst ? error / (bound > 0.0 ? bound : 1.0) : worst;
				apart += (tile[at] > chain[at] ? tile[at] - chain[at] : chain[at] - tile[at]) > 1u ? 1u : 0u;
				checked++;
			}
	if ( worse != 0u || apart * 1000u > checked )
	{
		printf("FAIL %s: %llu of %llu outputs miss the exact product by more than bf16 rounding (worst %.2f of the bound), %llu more than one bf16 step from the tile kernel\n",
			label,(unsigned long long)worse,(unsigned long long)checked,worst,(unsigned long long)apart);
		probe_failures++;
	}
}

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET>
static void ProbeShape(uint32_t rows)
{
	const uint64_t input_count = (uint64_t)rows * PROBE_HEADS * INPUT_HEAD_DIM,weight_count = (uint64_t)PROBE_HEADS * OUT_DIM * IN_DIM,output_count = (uint64_t)rows * PROBE_HEADS * OUT_DIM;
	std::vector<uint16_t> tile(output_count),chain(output_count);
	uint16_t *input,*weight,*output;
	cudaEvent_t begin,end;
	float tile_ms,chain_ms;
	uint32_t index;
	char label[96];
	snprintf(label,sizeof(label),"in%u out%u head%u offset%u rows%u",IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET,rows);
	PROBE_CUDA(cudaMalloc((void **)&input,input_count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,weight_count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&output,output_count * 2u));
	ProbeFillKernel<<<256,256>>>(input,input_count,rows * 7919u);
	ProbeFillKernel<<<256,256>>>(weight,weight_count,1ull << 50u);
	PROBE_CUDA(cudaEventCreate(&begin));
	PROBE_CUDA(cudaEventCreate(&end));
	PROBE_CUDA(cudaMemset(output,0xff,output_count * 2u));
	PROBE_CUDA((LmPerHeadProjectRowsLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaMemcpy(tile.data(),output,output_count * 2u,cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemset(output,0xff,output_count * 2u));
	PROBE_CUDA((LmPerHeadProjectChainLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaMemcpy(chain.data(),output,output_count * 2u,cudaMemcpyDeviceToHost));
	if ( rows <= LM_PROJECT_CHAIN_ROWS && memcmp(tile.data(),chain.data(),output_count * 2u) != 0 )
	{
		for (index=0u; index<output_count && tile[index] == chain[index]; index++)
			;
		printf("FAIL %s: chain output differs from the tile kernel at %u (%04x vs %04x)\n",label,index,chain[index],tile[index]);
		probe_failures++;
	}
	if ( rows > LM_PROJECT_CHAIN_ROWS )
		ProbeReference<IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(label,input,weight,rows,tile,chain);
	PROBE_CUDA(cudaEventRecord(begin));
	for (index=0u; index<PROBE_REPEATS; index++)
		PROBE_CUDA((LmPerHeadProjectRowsLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaEventRecord(end));
	PROBE_CUDA(cudaEventSynchronize(end));
	PROBE_CUDA(cudaEventElapsedTime(&tile_ms,begin,end));
	PROBE_CUDA(cudaEventRecord(begin));
	for (index=0u; index<PROBE_REPEATS; index++)
		PROBE_CUDA((LmPerHeadProjectChainLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaEventRecord(end));
	PROBE_CUDA(cudaEventSynchronize(end));
	PROBE_CUDA(cudaEventElapsedTime(&chain_ms,begin,end));
	printf("%s %s: %s, us tile %.1f chain %.1f\n",probe_failures == 0 ? "PASS" : "FAIL",label,rows <= LM_PROJECT_CHAIN_ROWS ? "bit-equal to the tile kernel" :
		"tensor cores within bf16 rounding of the exact product",tile_ms * 1000.0f / PROBE_REPEATS,chain_ms * 1000.0f / PROBE_REPEATS);
	cudaFree(input); cudaFree(weight); cudaFree(output);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
}

int main(void)
{
	uint32_t rows;
	for (rows=1u; rows<=8u; rows++)
	{
		ProbeShape<512u,256u,512u,0u>(rows);
		ProbeShape<192u,512u,256u,0u>(rows);
		ProbeShape<64u,128u,256u,192u>(rows);
	}
	ProbeShape<512u,256u,512u,0u>(16u);
	ProbeShape<192u,512u,256u,0u>(13u);
	ProbeShape<64u,128u,256u,192u>(9u);
	ProbeShape<128u,512u,320u,0u>(100u);
	ProbeShape<512u,128u,512u,0u>(64u);
	ProbeShape<192u,512u,256u,0u>(2048u);
	ProbeShape<512u,256u,512u,0u>(2048u);
	ProbeShape<128u,512u,320u,0u>(4096u);
	ProbeShape<512u,128u,512u,0u>(4096u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS per-head chain projection on the device: rows 1..8 are bit-equal to the tile kernel; above 8 rows the tensor-core kernel is within bf16 rounding of the exact product\n");
	return 0;
}
