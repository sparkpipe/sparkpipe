#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/index_select.cuh"

#define PROBE_KEEP 2048u

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

static int probe_failures;

__global__ void ProbeFillScores(float *scores, uint64_t count, uint32_t salt, uint32_t levels)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		float value = ((float)(x & 0xffffffu) / 16777216.0f) * 6.0f - 1.0f;
		if ( levels != 0u )
			value = floorf(value * (float)levels) / (float)levels;
		scores[index] = value > 0.0f || levels == 0u ? value : 0.0f;
	}
}

struct ProbeCase
{
	const char *name;
	uint32_t rows;
	uint32_t keys;
	uint32_t levels;
	SparkKvShard shard;
};

static std::vector<uint64_t> ProbeRowSet(const std::vector<uint2> &candidates, uint32_t row)
{
	std::vector<uint64_t> set(PROBE_KEEP);
	for (uint32_t entry = 0u; entry < PROBE_KEEP; ++entry)
	{
		const uint2 candidate = candidates[(size_t)row * PROBE_KEEP + entry];
		set[entry] = ((uint64_t)candidate.y << 32u) | candidate.x;
	}
	std::sort(set.begin(),set.end());
	return set;
}

static void ProbeRun(const ProbeCase &probe)
{
	const uint32_t rows = probe.rows,keys = probe.keys,degree = probe.shard.degree;
	const uint32_t context = keys * degree;
	std::vector<uint32_t> positions(rows);
	std::vector<uint2> reference((size_t)rows * PROBE_KEEP),candidate((size_t)rows * PROBE_KEEP);
	float *scores,best_reference = 1e30f,best_candidate = 1e30f,elapsed;
	uint2 *reference_candidates,*candidate_candidates;
	uint32_t *sequence,*context_length,*position;
	LmFrameError *error,error_host;
	cudaEvent_t start,stop;
	for (uint32_t row = 0u; row < rows; ++row)
	{
		positions[row] = row % 4u == 0u ? context - 1u : (uint32_t)(((uint64_t)row * 2654435761u) % context);
		if ( row % 9u == 1u )
			positions[row] = PROBE_KEEP * degree - 1u + row % 3u * degree * probe.shard.grain;
	}
	PROBE_CUDA(cudaMalloc((void **)&scores,(size_t)rows * keys * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&reference_candidates,(size_t)rows * PROBE_KEEP * sizeof(uint2)));
	PROBE_CUDA(cudaMalloc((void **)&candidate_candidates,(size_t)rows * PROBE_KEEP * sizeof(uint2)));
	PROBE_CUDA(cudaMalloc((void **)&sequence,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&context_length,sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&position,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&error,sizeof(LmFrameError)));
	PROBE_CUDA(cudaMemset(sequence,0,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMemcpy(context_length,&context,sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(position,positions.data(),rows * sizeof(uint32_t),cudaMemcpyHostToDevice));
	LmFrameErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemset(reference_candidates,0xa5,(size_t)rows * PROBE_KEEP * sizeof(uint2)));
	PROBE_CUDA(cudaMemset(candidate_candidates,0x5a,(size_t)rows * PROBE_KEEP * sizeof(uint2)));
	ProbeFillScores<<<1024,256>>>(scores,(uint64_t)rows * keys,keys + probe.levels + degree,probe.levels);
	PROBE_CUDA(cudaGetLastError());
	PROBE_CUDA(cudaEventCreate(&start));
	PROBE_CUDA(cudaEventCreate(&stop));
	for (uint32_t pass = 0u; pass < 3u; ++pass)
	{
		PROBE_CUDA(cudaEventRecord(start));
		LmIndexSelectLocalKernel<LM_INDEX_SELECT_THREADS><<<rows,LM_INDEX_SELECT_THREADS>>>(probe.shard,scores,keys,sequence,context_length,position,PROBE_KEEP,
			PROBE_KEEP,reference_candidates,error);
		PROBE_CUDA(cudaPeekAtLastError());
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
		best_reference = elapsed < best_reference ? elapsed : best_reference;
		PROBE_CUDA(cudaEventRecord(start));
		PROBE_CUDA(LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(probe.shard,scores,keys,sequence,context_length,position,PROBE_KEEP,PROBE_KEEP,rows,
			candidate_candidates,error,0));
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
		best_candidate = elapsed < best_candidate ? elapsed : best_candidate;
	}
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(reference.data(),reference_candidates,reference.size() * sizeof(uint2),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(candidate.data(),candidate_candidates,candidate.size() * sizeof(uint2),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(&error_host,error,sizeof(error_host),cudaMemcpyDeviceToHost));
	uint32_t differing = 0u,first = rows,compared = 0u;
	for (uint32_t row = 0u; row < rows; ++row)
	{
		const uint32_t bound = positions[row] + 1u < context ? positions[row] + 1u : context;
		if ( bound <= PROBE_KEEP )
			continue;
		compared++;
		if ( ProbeRowSet(reference,row) != ProbeRowSet(candidate,row) )
		{
			differing++;
			first = first < row ? first : row;
		}
	}
	if ( differing != 0u || compared == 0u || error_host.error_code != LM_FRAME_ERROR_NONE )
	{
		printf("FAIL %s keys=%u degree=%u: %u of %u rows select differently (first row %u), error %u\n",probe.name,keys,degree,differing,compared,first,
			(unsigned)error_host.error_code);
		probe_failures++;
	}
	else
		printf("PASS %s keys=%u degree=%u: %u rows select the same candidates; radix %.3f ms, dispatched %.3f ms, %.2fx\n",probe.name,keys,degree,compared,
			best_reference,best_candidate,best_reference / best_candidate);
	PROBE_CUDA(cudaEventDestroy(start));
	PROBE_CUDA(cudaEventDestroy(stop));
	PROBE_CUDA(cudaFree(scores));
	PROBE_CUDA(cudaFree(reference_candidates));
	PROBE_CUDA(cudaFree(candidate_candidates));
	PROBE_CUDA(cudaFree(sequence));
	PROBE_CUDA(cudaFree(context_length));
	PROBE_CUDA(cudaFree(position));
	PROBE_CUDA(cudaFree(error));
}

int main(void)
{
	static const ProbeCase cases[] = {
		{"distinct scores", 128u, 65536u, 0u, {1u, 0u, 1u}},
		{"distinct scores", 128u, 262144u, 0u, {1u, 0u, 1u}},
		{"distinct scores", 128u, 1048576u, 0u, {1u, 0u, 1u}},
		{"distinct scores, rows of every length", 128u, 20000u, 0u, {1u, 0u, 1u}},
		{"scores on a 1/256 grid, hundreds tied at the threshold", 128u, 1048576u, 256u, {1u, 0u, 1u}},
		{"scores on a 1/4096 grid", 128u, 262144u, 4096u, {1u, 0u, 1u}},
		{"scores on a 1/2 grid, the tied bucket overflows the list", 32u, 262144u, 2u, {1u, 0u, 1u}},
		{"the list overflows, a second read takes the threshold bucket", 16u, 4194304u, 0u, {1u, 0u, 1u}},
		{"one rank of four, 64-position grains", 128u, 262144u, 0u, {4u, 1u, 64u}},
		{"short rows take the radix kernel", 128u, 12000u, 0u, {1u, 0u, 1u}},
	};
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index)
		ProbeRun(cases[index]);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d index select cases\n",probe_failures);
		return 1;
	}
	printf("PASS the sampled index select keeps the radix select's candidates\n");
	return 0;
}
