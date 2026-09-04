#include <cstdio>
#include <cmath>
#include <cstring>

#include "spark_qwen38_max_resident_decode_stage_cuda.cu"

static int failures = 0;

#define CHECK(cond, name) do { if (!(cond)) { std::printf("FAIL %s\n", name); failures++; } } while (0)

static float HostBf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out,&bits,sizeof(out));
	return(out);
}

static float CpuSoftmaxWeight(const float *logits, uint32_t count, uint32_t expert)
{
	float maximum = -INFINITY,total = 0.0f;
	uint32_t index;
	for (index = 0u; index < count; index++)
		maximum = fmaxf(maximum,logits[index]);
	for (index = 0u; index < count; index++)
		total += expf(logits[index] - maximum);
	return(expf(logits[expert] - maximum) / total);
}

static void TestRouterSoftmax(void)
{
	const float logits[8] = {2.0f,-3.0f,5.0f,1.0f,-1.0f,0.5f,4.0f,-2.0f};
	float *device_logits = 0,*device_weights = 0;
	uint32_t *device_indices = 0;
	uint32_t indices[3] = {0u,0u,0u};
	float weights[3] = {0.0f,0.0f,0.0f};
	uint32_t rank;
	float expected,renorm;
	cudaMalloc(&device_logits,sizeof(logits));
	cudaMalloc(&device_weights,3u * sizeof(float));
	cudaMalloc(&device_indices,3u * sizeof(uint32_t));
	cudaMemcpy(device_logits,logits,sizeof(logits),cudaMemcpyHostToDevice);
	CHECK(SparkQwen38MaxLaunchGateSelect(0,device_logits,0,1u,8u,3u,1.0f,device_indices,device_weights) == cudaSuccess,"gate_select_launch");
	cudaMemcpy(indices,device_indices,sizeof(indices),cudaMemcpyDeviceToHost);
	cudaMemcpy(weights,device_weights,sizeof(weights),cudaMemcpyDeviceToHost);
	CHECK(indices[0] == 2u && indices[1] == 6u && indices[2] == 0u,"gate_select_topk_indices");
	renorm = 0.0f;
	for (rank = 0u; rank < 3u; rank++)
		renorm += CpuSoftmaxWeight(logits,8u,indices[rank]);
	for (rank = 0u; rank < 3u; rank++)
	{
		expected = CpuSoftmaxWeight(logits,8u,indices[rank]) / renorm;
		CHECK(fabsf(weights[rank] - expected) < 1.0e-4f,"gate_select_renormalized_weight");
		CHECK(weights[rank] > 0.0f,"gate_select_positive_weights");
	}
	cudaFree(device_logits);
	cudaFree(device_weights);
	cudaFree(device_indices);
}

static void TestSharedGate(void)
{
	const uint16_t input[4] = {0x3f80u,0x4000u,0x4040u,0x4080u};
	const uint16_t weight[4] = {0xbf00u,0x3f00u,0x3f80u,0xbf80u};
	uint16_t accum[4] = {0x3f80u,0x4000u,0x4040u,0x4080u};
	uint16_t *device_input = 0,*device_weight = 0,*device_accum = 0;
	uint16_t result[4] = {0u,0u,0u,0u};
	float logit = 0.0f,gate;
	uint32_t index;
	for (index = 0u; index < 4u; index++)
		logit += HostBf16ToFloat(input[index]) * HostBf16ToFloat(weight[index]);
	gate = 1.0f / (1.0f + expf(-logit));
	cudaMalloc(&device_input,sizeof(input));
	cudaMalloc(&device_weight,sizeof(weight));
	cudaMalloc(&device_accum,sizeof(accum));
	cudaMemcpy(device_input,input,sizeof(input),cudaMemcpyHostToDevice);
	cudaMemcpy(device_weight,weight,sizeof(weight),cudaMemcpyHostToDevice);
	cudaMemcpy(device_accum,accum,sizeof(accum),cudaMemcpyHostToDevice);
	CHECK(SparkQwen38MaxLaunchSharedGate(0,device_accum,device_weight,device_input,1u,4u) == cudaSuccess,"shared_gate_launch");
	cudaMemcpy(result,device_accum,sizeof(result),cudaMemcpyDeviceToHost);
	for (index = 0u; index < 4u; index++)
		CHECK(fabsf(HostBf16ToFloat(result[index]) - (gate * HostBf16ToFloat(accum[index]))) < 1.0e-2f,"shared_gate_scalar_product");
	cudaFree(device_input);
	cudaFree(device_weight);
	cudaFree(device_accum);
}

static void TestPairReduceOverwrite(void)
{
	const uint16_t slot[8] = {0x3f80u,0x4000u,0x4040u,0x4080u, 0x4100u,0x4180u,0x4200u,0x4280u};
	const uint32_t inverse[2] = {0u,1u};
	const float weights[2] = {0.25f,0.75f};
	uint16_t seed[4] = {0x47c7u,0x47c7u,0x47c7u,0x47c7u};
	uint16_t *device_slot = 0,*device_seed = 0;
	uint32_t *device_inverse = 0;
	float *device_weights = 0;
	uint16_t result[4] = {0u,0u,0u,0u};
	uint32_t index;
	cudaMalloc(&device_slot,sizeof(slot));
	cudaMalloc(&device_seed,sizeof(seed));
	cudaMalloc(&device_inverse,sizeof(inverse));
	cudaMalloc(&device_weights,sizeof(weights));
	cudaMemcpy(device_slot,slot,sizeof(slot),cudaMemcpyHostToDevice);
	cudaMemcpy(device_seed,seed,sizeof(seed),cudaMemcpyHostToDevice);
	cudaMemcpy(device_inverse,inverse,sizeof(inverse),cudaMemcpyHostToDevice);
	cudaMemcpy(device_weights,weights,sizeof(weights),cudaMemcpyHostToDevice);
	CHECK(SparkQwen38MaxLaunchMoePairReduceOverwrite(0,device_slot,device_inverse,device_weights,device_seed,1u,4u) == cudaSuccess,"pair_reduce_overwrite_launch");
	cudaMemcpy(result,device_seed,sizeof(result),cudaMemcpyDeviceToHost);
	for (index = 0u; index < 4u; index++)
	{
		float expected = (0.25f * HostBf16ToFloat(slot[index])) + (0.75f * HostBf16ToFloat(slot[4u + index]));
		CHECK(fabsf(HostBf16ToFloat(result[index]) - expected) < 1.0e-3f,"pair_reduce_overwrite_value");
	}
	cudaFree(device_slot);
	cudaFree(device_seed);
	cudaFree(device_inverse);
	cudaFree(device_weights);
}

int main(void)
{
	TestRouterSoftmax();
	TestSharedGate();
	TestPairReduceOverwrite();
	if ( failures == 0 )
		std::printf("test_qwen38_math_kernels PASS\n");
	else
		std::printf("test_qwen38_math_kernels FAIL %d\n",failures);
	return(failures == 0 ? 0 : 1);
}
