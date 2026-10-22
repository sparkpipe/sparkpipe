#include "modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROBE_RANKS 16u
#define PROBE_RANK 5u
#define PROBE_SLICE (K3_ROUTED_EXPERT_HIDDEN / PROBE_RANKS)
#define PROBE_MAX_ROWS 37u

static uint32_t probe_rows,probe_scatters,probe_gathers,probe_other,probe_fails,probe_layout_errors;
static int probe_failures;

static float ProbeBf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result, &bits, sizeof(result));
	return result;
}

static uint16_t ProbeFloatToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static uint16_t ProbeContribution(uint32_t rank, uint32_t row, uint32_t column)
{
	uint32_t mix = (rank * 2654435761u) ^ (row * 40503u) ^ (column * 2246822519u);
	mix ^= mix >> 15u;
	mix *= 2246822519u;
	mix ^= mix >> 13u;
	return ProbeFloatToBf16(((float)(mix % 2001u) / 1000.0f - 1.0f) * (float)(1u + row % 7u));
}

static uint16_t ProbeReduced(uint32_t row, uint32_t column, const uint16_t *own)
{
	float total = 0.0f;
	uint32_t rank;
	for (rank = 0u; rank < PROBE_RANKS; ++rank)
		total += ProbeBf16ToFloat(rank == PROBE_RANK && own != 0 ? *own : ProbeContribution(rank, row, column));
	return ProbeFloatToBf16(total);
}

static SparkStatus ProbeRound(void *context, void *stream, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements, const void *local, void *full)
{
	static uint16_t columns[PROBE_MAX_ROWS * K3_ROUTED_EXPERT_HIDDEN], reduced[PROBE_MAX_ROWS * PROBE_SLICE];
	static uint32_t squares[PROBE_RANKS * PROBE_MAX_ROWS];
	uint32_t row, column, rank;
	(void)context;
	if ( cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess || stage != K3_RUNNER_STAGE_ALL || rows != probe_rows || local == full )
	{
		probe_other++;
		return SPARK_STATUS_OK;
	}
	if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16 && row_elements == K3_ROUTED_EXPERT_HIDDEN )
	{
		probe_scatters++;
		cudaMemcpy(columns, local, (size_t)rows * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		for (rank = 0u; rank < PROBE_RANKS; ++rank)
			for (row = 0u; row < rows; ++row)
				for (column = 0u; column < PROBE_SLICE; ++column)
					if ( columns[((uint64_t)rank * rows + row) * PROBE_SLICE + column] != ProbeContribution(PROBE_RANK, row, rank * PROBE_SLICE + column) )
						probe_layout_errors++;
		for (row = 0u; row < rows; ++row)
			for (column = 0u; column < PROBE_SLICE; ++column)
				reduced[row * PROBE_SLICE + column] = ProbeReduced(row, PROBE_RANK * PROBE_SLICE + column,
					&columns[((uint64_t)PROBE_RANK * rows + row) * PROBE_SLICE + column]);
		cudaMemcpy(full, reduced, (size_t)rows * PROBE_SLICE * sizeof(uint16_t), cudaMemcpyHostToDevice);
		return SPARK_STATUS_OK;
	}
	if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && row_elements == 2u )
	{
		probe_gathers++;
		cudaMemcpy(squares + PROBE_RANK * rows, local, (size_t)rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for (rank = 0u; rank < PROBE_RANKS; ++rank)
		{
			if ( rank == PROBE_RANK )
				continue;
			for (row = 0u; row < rows; ++row)
			{
				float total = 0.0f, value;
				for (column = 0u; column < PROBE_SLICE; ++column)
				{
					value = ProbeBf16ToFloat(ProbeReduced(row, rank * PROBE_SLICE + column, 0));
					total += value * value;
				}
				memcpy(&squares[rank * rows + row], &total, sizeof(total));
			}
		}
		cudaMemcpy(full, squares, (size_t)PROBE_RANKS * rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		return SPARK_STATUS_OK;
	}
	probe_other++;
	return SPARK_STATUS_OK;
}

static void ProbeFail(void *context, SparkStatus status)
{
	(void)context; (void)status;
	probe_fails++;
}

static void ProbeCheck(int condition, const char *message)
{
	if ( condition == 0 )
	{
		printf("FAIL %s (rows %u scatters %u gathers %u other %u fails %u layout %u)\n", message, probe_rows, probe_scatters,
			probe_gathers, probe_other, probe_fails, probe_layout_errors);
		probe_failures++;
	}
}

static void *ProbeDevice(uint64_t bytes)
{
	void *pointer = 0;
	if ( cudaMalloc(&pointer, bytes) != cudaSuccess || cudaMemset(pointer, 0, bytes) != cudaSuccess )
	{
		printf("FAIL cuda allocation of %llu bytes\n", (unsigned long long)bytes);
		exit(1);
	}
	return pointer;
}

static void ProbeRun(SparkK3ModelState *state, K3LayerBuffers *buffers, const uint16_t *weight_device, uint32_t rows)
{
	static uint16_t contribution[PROBE_MAX_ROWS * K3_ROUTED_EXPERT_HIDDEN], expected[PROBE_MAX_ROWS * PROBE_SLICE],
		produced[PROBE_MAX_ROWS * PROBE_SLICE];
	uint16_t *full_device = (uint16_t *)ProbeDevice((uint64_t)rows * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t));
	uint16_t *expected_device = (uint16_t *)ProbeDevice((uint64_t)rows * PROBE_SLICE * sizeof(uint16_t));
	uint32_t row, column, exact = 0u, worst = 0u;
	char message[160];
	for (row = 0u; row < rows; ++row)
		for (column = 0u; column < K3_ROUTED_EXPERT_HIDDEN; ++column)
			contribution[row * K3_ROUTED_EXPERT_HIDDEN + column] = ProbeContribution(PROBE_RANK, row, column);
	cudaMemcpy(buffers->latent_bf16, contribution, sizeof(uint16_t) * rows * K3_ROUTED_EXPERT_HIDDEN, cudaMemcpyHostToDevice);
	for (row = 0u; row < rows; ++row)
		for (column = 0u; column < K3_ROUTED_EXPERT_HIDDEN; ++column)
			contribution[row * K3_ROUTED_EXPERT_HIDDEN + column] = ProbeReduced(row, column, 0);
	cudaMemcpy(full_device, contribution, sizeof(uint16_t) * rows * K3_ROUTED_EXPERT_HIDDEN, cudaMemcpyHostToDevice);
	LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t><<<rows, K3_LAYER_THREADS, (K3_ROUTED_EXPERT_HIDDEN + 8u) * sizeof(float)>>>(
		full_device, 0, weight_device, 0, full_device, K3_ROUTED_EXPERT_HIDDEN, K3_ROUTED_EXPERT_HIDDEN, K3_RMS_EPSILON);
	K3LatentSliceKernel<K3_LAYER_THREADS><<<dim3((PROBE_SLICE + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS, rows), K3_LAYER_THREADS>>>(
		full_device, expected_device, PROBE_SLICE, PROBE_RANK * PROBE_SLICE, K3_ROUTED_EXPERT_HIDDEN);
	cudaMemcpy(expected, expected_device, sizeof(uint16_t) * rows * PROBE_SLICE, cudaMemcpyDeviceToHost);
	probe_rows = rows;
	probe_scatters = probe_gathers = probe_other = probe_fails = probe_layout_errors = 0u;
	state->rows = rows;
	K3RunnerLayerCollective(&state->lanes[0], 0, 3u, K3_COLLECTIVE_LATENT_SLICE);
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		printf("FAIL device error after the latent slice round\n");
		exit(1);
	}
	cudaMemcpy(produced, buffers->latent_bf16, sizeof(uint16_t) * rows * PROBE_SLICE, cudaMemcpyDeviceToHost);
	for (row = 0u; row < rows * PROBE_SLICE; ++row)
	{
		uint32_t distance = produced[row] > expected[row] ? produced[row] - expected[row] : expected[row] - produced[row];
		exact += distance == 0u;
		worst = distance > worst ? distance : worst;
	}
	snprintf(message, sizeof(message), "%u rows: one column reduce-scatter and one row-statistics gather, nothing else", rows);
	ProbeCheck(probe_scatters == 1u && probe_gathers == 1u && probe_other == 0u && probe_fails == 0u, message);
	snprintf(message, sizeof(message), "%u rows: every rank's columns reach that rank's reduce-scatter segment", rows);
	ProbeCheck(probe_layout_errors == 0u, message);
	snprintf(message, sizeof(message), "%u rows: the normalised slice matches the all-reduce path within one bf16 step (%u of %u exact, worst %u)",
		rows, exact, rows * PROBE_SLICE, worst);
	ProbeCheck(worst <= 1u && exact * 100u >= rows * PROBE_SLICE * 99u, message);
	cudaFree(full_device);
	cudaFree(expected_device);
}

int main(void)
{
	static SparkStageRunnerServices services;
	static uint16_t weight[K3_ROUTED_EXPERT_HIDDEN];
	SparkK3ModelState *state = (SparkK3ModelState *)calloc(1u, sizeof(SparkK3ModelState));
	K3LayerBuffers *buffers = (K3LayerBuffers *)calloc(1u, sizeof(K3LayerBuffers));
	uint16_t *weight_device;
	uint32_t column;
	if ( state == 0 || buffers == 0 )
		return 1;
	for (column = 0u; column < K3_ROUTED_EXPERT_HIDDEN; ++column)
		weight[column] = ProbeFloatToBf16(0.5f + (float)(column % 13u) / 16.0f);
	weight_device = (uint16_t *)ProbeDevice(sizeof(weight));
	cudaMemcpy(weight_device, weight, sizeof(weight), cudaMemcpyHostToDevice);
	services.round = ProbeRound;
	services.fail = ProbeFail;
	state->services = &services;
	state->has_collective = 1u;
	state->tp_degree = PROBE_RANKS;
	state->tp_rank = PROBE_RANK;
	state->fused_rows = PROBE_MAX_ROWS;
	state->fused_device = (uint16_t *)ProbeDevice((uint64_t)PROBE_MAX_ROWS * K3_HIDDEN * sizeof(uint16_t));
	buffers->tp_sharded = 1u;
	buffers->tp_rank = PROBE_RANK;
	buffers->routed_down_rows = PROBE_SLICE;
	buffers->expert_w1_output = 2u * (K3_EXPERT_INTERMEDIATE / PROBE_RANKS);
	buffers->routed_norm_weight = weight_device;
	buffers->latent_bf16 = (uint16_t *)ProbeDevice((uint64_t)PROBE_MAX_ROWS * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t));
	state->dispatch.buffers = buffers;
	state->lanes[0].state = state;

	ProbeRun(state, buffers, weight_device, 3u);
	ProbeRun(state, buffers, weight_device, PROBE_MAX_ROWS);

	probe_scatters = probe_gathers = probe_other = probe_fails = 0u;
	state->tp_degree = PROBE_RANKS / 2u;
	K3RunnerLayerCollective(&state->lanes[0], 0, 3u, K3_COLLECTIVE_LATENT_SLICE);
	ProbeCheck(probe_fails == 1u && probe_scatters == 0u && probe_gathers == 0u,
		"a latent width that does not split into one slice per rank fails loudly before any round");
	state->tp_degree = PROBE_RANKS;
	probe_fails = 0u;
	state->rows = 2u * PROBE_MAX_ROWS + 1u;
	K3RunnerLayerCollective(&state->lanes[0], 0, 3u, K3_COLLECTIVE_LATENT_SLICE);
	ProbeCheck(probe_fails == 1u && probe_scatters == 0u, "rows beyond the column staging fail loudly before any round");
	if ( probe_failures != 0 )
	{
		printf("FAIL k3_latent_slice_cuda: %d failures\n", probe_failures);
		return 1;
	}
	printf("PASS k3_latent_slice_cuda: the routed latent reduce-scatters by rank columns, the norm statistics travel as exact fp32 sums, and the slice matches the all-reduce path\n");
	return 0;
}
