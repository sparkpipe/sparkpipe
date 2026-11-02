#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/topk.cuh"

#define PROBE_EXPERTS 256u
#define PROBE_GROUPS 8u
#define PROBE_TOP_GROUPS 4u
#define PROBE_K 8u
#define PROBE_ROWS 64u

static int probe_failures;

static void ProbeCheck(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n", what);
	probe_failures++;
}

static std::vector<uint32_t> ProbeReference(const float *logits)
{
	const uint32_t per_group = PROBE_EXPERTS / PROBE_GROUPS;
	std::vector<double> score(PROBE_EXPERTS);
	std::vector<std::pair<double,uint32_t>> groups;
	std::vector<uint32_t> candidates, chosen;
	for ( uint32_t index = 0u; index < PROBE_EXPERTS; index++ )
		score[index] = 1.0 / (1.0 + exp(-(double)logits[index]));
	for ( uint32_t group = 0u; group < PROBE_GROUPS; group++ )
	{
		std::vector<double> members(score.begin() + group * per_group, score.begin() + (group + 1u) * per_group);
		std::sort(members.begin(), members.end(), [](double a, double b) { return a > b; });
		groups.push_back({members[0] + members[1], group});
	}
	std::sort(groups.begin(), groups.end(), [](const std::pair<double,uint32_t> &a, const std::pair<double,uint32_t> &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
	for ( uint32_t keep = 0u; keep < PROBE_TOP_GROUPS; keep++ )
		for ( uint32_t member = 0u; member < per_group; member++ )
			candidates.push_back(groups[keep].second * per_group + member);
	std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
	chosen.assign(candidates.begin(), candidates.begin() + PROBE_K);
	return chosen;
}

int main(void)
{
	std::vector<float> logits((size_t)PROBE_ROWS * PROBE_EXPERTS);
	uint32_t state = 20261027u, tied_rows = 0u;
	for ( uint32_t row = 0u; row < PROBE_ROWS; row++ )
		for ( uint32_t index = 0u; index < PROBE_EXPERTS; index++ )
		{
			const uint32_t group = index / (PROBE_EXPERTS / PROBE_GROUPS);
			state = state * 1664525u + 1013904223u;
			const float base = (float)((group * 5u + row) % PROBE_GROUPS) * 0.5f - 2.0f;
			const float level = row % 2u == 0u ? (float)((state >> 28u) % 3u) * 0.25f : (float)(state >> 8u) / 16777216.0f;
			logits[(size_t)row * PROBE_EXPERTS + index] = base + level;
		}
	float *d_logits, *d_values;
	uint32_t *d_indices;
	cudaMalloc((void **)&d_logits, logits.size() * sizeof(float));
	cudaMalloc((void **)&d_values, (size_t)PROBE_ROWS * PROBE_K * sizeof(float));
	cudaMalloc((void **)&d_indices, (size_t)PROBE_ROWS * PROBE_K * sizeof(uint32_t));
	cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);
	uint32_t *d_fault = 0;
	cudaMalloc((void **)&d_fault, sizeof(uint32_t));
	cudaMemset(d_fault, 0, sizeof(uint32_t));
	LmTopkSmallKernel<256u,PROBE_K,true,PROBE_GROUPS,PROBE_TOP_GROUPS,LM_TOPK_SCORE_SIGMOID><<<PROBE_ROWS,256u,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t)>>>(
		d_logits, PROBE_EXPERTS, d_indices, d_values, (const float *)0, (const uint16_t *)0, 1.0f, d_fault);
	ProbeCheck(cudaDeviceSynchronize() == cudaSuccess, "the small top-k kernel runs");
	std::vector<uint32_t> got((size_t)PROBE_ROWS * PROBE_K);
	std::vector<float> values((size_t)PROBE_ROWS * PROBE_K);
	cudaMemcpy(got.data(), d_indices, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(values.data(), d_values, values.size() * sizeof(float), cudaMemcpyDeviceToHost);
	for ( uint32_t row = 0u; row < PROBE_ROWS; row++ )
	{
		const float *line = logits.data() + (size_t)row * PROBE_EXPERTS;
		const std::vector<uint32_t> want = ProbeReference(line);
		double total = 0.0;
		char what[160];
		bool tie = false;
		for ( uint32_t k = 0u; k + 1u < PROBE_K; k++ )
			tie = tie || line[want[k]] == line[want[k + 1u]];
		for ( uint32_t k = PROBE_K; k < PROBE_EXPERTS && !tie; k++ )
			tie = line[want[PROBE_K - 1u]] == line[k] && k > want[PROBE_K - 1u];
		tied_rows += tie ? 1u : 0u;
		snprintf(what, sizeof(what), "row %u selects the reference experts in order (score descending, lower index first on ties)", row);
		ProbeCheck(std::equal(want.begin(), want.end(), got.begin() + (size_t)row * PROBE_K), what);
		for ( uint32_t k = 0u; k < PROBE_K; k++ )
			total += values[(size_t)row * PROBE_K + k];
		snprintf(what, sizeof(what), "row %u renormalised weights sum to one", row);
		ProbeCheck(fabs(total - 1.0) < 1e-5, what);
	}
	ProbeCheck(tied_rows >= PROBE_ROWS / 4u, "the inputs carry ties at the selection boundary");
	cudaFree(d_logits); cudaFree(d_values); cudaFree(d_indices);
	if ( probe_failures != 0 )
	{
		printf("FAIL topk small cuda (%d)\n", probe_failures);
		return 1;
	}
	printf("PASS topk small cuda: %u rows of %u grouped experts select the reference top %u in order, %u rows with ties resolved to the lower expert index\n",
		PROBE_ROWS, PROBE_EXPERTS, PROBE_K, tied_rows);
	return 0;
}
