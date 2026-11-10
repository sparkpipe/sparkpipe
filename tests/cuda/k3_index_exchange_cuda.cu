#include "modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define XR_RANKS 16u
#define XR_CONTEXT 40000u
#define XR_LAYER 3u

struct XrRecord
{
	uint32_t calls;
	uint32_t operation;
	uint32_t rows;
	uint32_t row_elements;
	const void *local;
	void *full;
};

struct XrRank
{
	SparkStageRunnerServices services;
	SparkK3ModelState *state;
	K3LayerBuffers *b;
	SparkK3Dispatch dispatch;
	K3SliceState slice;
	XrRecord record;
	uint32_t *sequence_of_row;
	uint32_t *context_length;
	uint32_t *positions;
};

static XrRank xr[XR_RANKS];
static uint32_t xr_fails;
static int xr_failures;

static void XrCheck(int condition, const char *message)
{
	if ( condition == 0 )
	{
		printf("FAIL %s\n", message);
		xr_failures++;
	}
}

static void *XrDevice(uint64_t bytes)
{
	void *pointer = 0;
	if ( cudaMalloc(&pointer, bytes) != cudaSuccess || cudaMemset(pointer, 0, bytes) != cudaSuccess )
	{
		printf("FAIL cuda allocation of %llu bytes\n", (unsigned long long)bytes);
		exit(1);
	}
	return pointer;
}

static SparkStatus XrRound(void *context, void *stream, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements, const void *local, void *full)
{
	XrRecord *record = (XrRecord *)context;
	(void)stream;
	XrCheck(stage == K3_RUNNER_STAGE_ALL, "an index round is one whole round");
	record->calls++;
	record->operation = operation;
	record->rows = rows;
	record->row_elements = row_elements;
	record->local = local;
	record->full = full;
	return SPARK_STATUS_OK;
}

static void XrFail(void *context, SparkStatus status)
{
	(void)context; (void)status;
	xr_fails++;
}

static uint64_t XrSpan(const XrRank &r, const void *pointer, uint64_t bytes)
{
	const uint8_t *p = (const uint8_t *)pointer, *base = r.dispatch.index_scratch, *end = base + r.dispatch.index_scratch_bytes;
	const uint8_t *starts[] = {(const uint8_t *)r.b->index_send, (const uint8_t *)r.b->index_received, (const uint8_t *)r.b->index_bitmaps,
		(const uint8_t *)r.b->index_selected, (const uint8_t *)r.b->index_error, (const uint8_t *)r.b->index_selected_own, end};
	const uint32_t spans[][2] = {{0u, 1u}, {1u, 2u}, {3u, 4u}, {5u, 6u}};
	for ( uint32_t i = 0u; i < sizeof(spans) / sizeof(spans[0]); i++ )
		if ( p >= starts[spans[i][0]] && p + bytes <= starts[spans[i][1]] )
			return 1u;
	return 0u;
}

static uint32_t XrExchange(uint32_t phase)
{
	uint32_t calls = 0u;
	for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
	{
		memset(&xr[rank].record, 0, sizeof(xr[rank].record));
		K3RunnerLayerCollective(&xr[rank].state->lanes[0], 0, XR_LAYER, phase);
		calls += xr[rank].record.calls;
	}
	XrCheck(calls == 0u || calls == XR_RANKS, "every rank takes part in the same rounds");
	if ( calls == 0u )
		return 0u;
	const XrRecord &first = xr[0].record;
	const uint64_t block = (uint64_t)first.rows * first.row_elements * sizeof(uint16_t);
	for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
	{
		const XrRecord &r = xr[rank].record;
		XrCheck(r.calls == 1u && r.operation == first.operation && r.rows == first.rows && r.row_elements == first.row_elements,
			"every rank posts the same round shape");
		XrCheck(r.row_elements != 0u && r.row_elements <= K3_HIDDEN, "an index round row fits the collective row width");
		XrCheck(first.operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER || first.operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL,
			"index rounds are gathers or exchanges");
		XrCheck(first.operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL || r.local != r.full, "an exchange sends and lands in different buffers");
		XrCheck(XrSpan(xr[rank], r.local, first.operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? XR_RANKS * block : block) != 0u,
			"a round reads inside its send buffer");
		XrCheck(XrSpan(xr[rank], r.full, XR_RANKS * block) != 0u, "a round lands inside its receive buffer");
	}
	if ( xr_failures != 0 )
		return 0u;
	for ( uint32_t destination = 0u; destination < XR_RANKS; destination++ )
		for ( uint32_t source = 0u; source < XR_RANKS; source++ )
		{
			const uint8_t *from = (const uint8_t *)xr[source].record.local +
				(first.operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? destination * block : 0u);
			if ( cudaMemcpy((uint8_t *)xr[destination].record.full + source * block, from, block, cudaMemcpyDeviceToDevice) != cudaSuccess )
			{
				printf("FAIL emulated round copy\n");
				exit(1);
			}
		}
	return first.operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? 2u : 1u;
}

__global__ void XrScoresKernel(float *scores, uint32_t stride, uint32_t count, uint32_t first, uint32_t rank)
{
	for ( uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < (uint64_t)count * stride; index += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint32_t row = (uint32_t)(index / stride) + first, entry = (uint32_t)(index % stride);
		uint32_t x = (row * 2654435761u) ^ (entry * 2246822519u) ^ (rank * 3266489917u);
		x ^= x >> 15;
		x *= 2246822519u;
		x ^= x >> 13;
		scores[index] = x % 97u == 0u ? -INFINITY : (float)(x % 64u) * 0.125f - 4.0f;
	}
}

static void XrSetup(uint32_t attention_rows, uint32_t rows)
{
	std::vector<uint32_t> sequence(rows, 0u), positions(rows);
	const uint32_t context = XR_CONTEXT;
	for ( uint32_t row = 0u; row < rows; row++ )
		positions[row] = context - rows + row;
	for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
	{
		XrRank &r = xr[rank];
		memset(&r.dispatch, 0, sizeof(r.dispatch));
		memset(&r.slice, 0, sizeof(r.slice));
		r.services.context = &r.record;
		r.services.round = XrRound;
		r.services.fail = XrFail;
		r.state = (SparkK3ModelState *)calloc(1u, sizeof(SparkK3ModelState));
		r.b = (K3LayerBuffers *)calloc(1u, sizeof(K3LayerBuffers));
		if ( r.state == 0 || r.b == 0 )
			exit(1);
		r.state->services = &r.services;
		r.state->has_collective = 1u;
		r.state->rows = rows;
		r.state->dispatch.buffers = r.b;
		r.state->lanes[0].state = r.state;
		r.b->tp_sharded = 1u;
		r.b->kv_shard = SparkK3KvShardContext(rank, XR_RANKS);
		r.dispatch.buffers_host = r.b;
		r.dispatch.tp_degree = XR_RANKS;
		r.dispatch.mla_count = 1u;
		r.dispatch.attention_rows = attention_rows;
		r.dispatch.slice_state = &r.slice;
		if ( SparkK3DispatchEnableIndex(&r.dispatch) != SPARK_K3_DISPATCH_OK )
		{
			printf("FAIL enable the index on rank %u\n", rank);
			exit(1);
		}
		r.sequence_of_row = (uint32_t *)XrDevice((uint64_t)rows * sizeof(uint32_t));
		r.positions = (uint32_t *)XrDevice((uint64_t)rows * sizeof(uint32_t));
		r.context_length = (uint32_t *)XrDevice(sizeof(uint32_t));
		cudaMemcpy(r.sequence_of_row, sequence.data(), (size_t)rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(r.positions, positions.data(), (size_t)rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(r.context_length, &context, sizeof(uint32_t), cudaMemcpyHostToDevice);
		r.b->sequence_of_row = r.sequence_of_row;
		r.b->positions = r.positions;
		r.b->context_length = r.context_length;
		K3LayerIndexPlan(r.b, context);
	}
}

static void XrTeardown(void)
{
	for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
	{
		XrRank &r = xr[rank];
		cudaFree(r.dispatch.index_scratch);
		cudaFree(r.dispatch.index_access_error);
		delete[] r.dispatch.index_cache;
		cudaFree(r.sequence_of_row);
		cudaFree(r.positions);
		cudaFree(r.context_length);
		free(r.b);
		free(r.state);
	}
}

static void XrCase(uint32_t attention_rows, uint32_t rows, uint32_t expect_exchange)
{
	XrSetup(attention_rows, rows);
	K3LayerBuffers *b0 = xr[0].b;
	const uint32_t keep = b0->index_keep, stride = b0->index_score_stride, words = LmIndexSelectBitmapWords(XR_CONTEXT);
	uint2 *reference_received = (uint2 *)XrDevice((uint64_t)XR_RANKS * b0->index_chunk_rows * keep * sizeof(uint2));
	uint32_t *reference_bitmaps = (uint32_t *)XrDevice((uint64_t)b0->index_chunk_rows * 2u * words * sizeof(uint32_t));
	uint32_t *reference_selected = (uint32_t *)XrDevice((uint64_t)rows * K3_INDEX_TOPK * sizeof(uint32_t));
	const uint32_t chunks = K3LayerIndexChunks(b0, rows);
	uint32_t exchanges = 0u, gathers = 0u, selected_rounds = 0u, mismatched = 0u;
	char message[256];
	XrCheck(b0->index_active != 0u && keep == K3_INDEX_TOPK && stride > keep, "the case runs the indexed path with more local keys than it keeps");
	for ( uint32_t chunk = 0u; chunk < chunks && xr_failures == 0; chunk++ )
	{
		for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
			xr[rank].b->index_chunk = chunk;
		const uint32_t first = K3LayerIndexChunkFirst(b0), count = K3LayerIndexChunkCount(b0, rows);
		for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
		{
			K3LayerBuffers *b = xr[rank].b;
			XrScoresKernel<<<256, 256>>>(b->index_scores, stride, count, first, rank);
			XrCheck(LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(b->kv_shard, b->index_scores, stride, b->sequence_of_row + first, b->context_length,
				b->positions + first, keep, K3_INDEX_TOPK, count, b->index_send, b->index_error, 0) == cudaSuccess, "local select launches");
			cudaMemcpy(reference_received + (uint64_t)rank * count * keep, b->index_send, (size_t)count * keep * sizeof(uint2), cudaMemcpyDeviceToDevice);
		}
		XrCheck(LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(reference_received, (uint64_t)count * keep, b0->kv_shard, keep, b0->sequence_of_row + first,
			b0->context_length, b0->positions + first, K3_INDEX_TOPK, K3_INDEX_TOPK, count, words, reference_bitmaps,
			reference_selected + (uint64_t)first * K3_INDEX_TOPK, b0->index_error, 0) == cudaSuccess, "reference merge launches");
		const uint32_t kind = XrExchange(K3_COLLECTIVE_MLA_INDEX);
		exchanges += kind == 2u ? 1u : 0u;
		gathers += kind == 1u ? 1u : 0u;
		for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
			XrCheck(K3LayerMlaIndexMerge(xr[rank].b, rows, 0) == LM_LAUNCH_OK, "the merge accepts its rows");
		selected_rounds += XrExchange(K3_COLLECTIVE_MLA_SELECTED) != 0u ? 1u : 0u;
	}
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		printf("FAIL device error in the case of %u rows\n", rows);
		exit(1);
	}
	std::vector<uint32_t> want((size_t)rows * K3_INDEX_TOPK), got((size_t)rows * K3_INDEX_TOPK);
	cudaMemcpy(want.data(), reference_selected, want.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	uint32_t distinct = 0u, filled = 0u;
	for ( uint32_t row = 0u; row < rows; row++ )
	{
		distinct += row > 0u && memcmp(&want[(size_t)row * K3_INDEX_TOPK], &want[(size_t)(row - 1u) * K3_INDEX_TOPK], K3_INDEX_TOPK * sizeof(uint32_t)) != 0 ? 1u : 0u;
		filled += want[(size_t)row * K3_INDEX_TOPK + K3_INDEX_TOPK - 1u] != LM_INDEX_SELECT_NONE ? 1u : 0u;
	}
	for ( uint32_t rank = 0u; rank < XR_RANKS; rank++ )
	{
		cudaMemcpy(got.data(), xr[rank].b->index_selected, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		mismatched += memcmp(got.data(), want.data(), got.size() * sizeof(uint32_t)) != 0 ? 1u : 0u;
	}
	snprintf(message, sizeof(message), "%u rows: every rank holds the gathered path's selection (%u of %u ranks differ)", rows, mismatched, XR_RANKS);
	XrCheck(mismatched == 0u, message);
	snprintf(message, sizeof(message), "%u rows: rows select different keys and fill all %u slots (%u distinct, %u filled)", rows, K3_INDEX_TOPK, distinct, filled);
	XrCheck((rows < 2u || distinct == rows - 1u) && filled == rows, message);
	snprintf(message, sizeof(message), "%u rows: %u chunks run %u exchanges, %u gathers and %u selected rounds", rows, chunks, exchanges, gathers, selected_rounds);
	XrCheck(expect_exchange != 0u ? exchanges == chunks && gathers == 0u && selected_rounds == chunks : exchanges == 0u && gathers == chunks && selected_rounds == 0u,
		message);
	printf("CASE %u rows (attention rows %u): %u chunks, %u exchanges, %u gathers, %u selected rounds, %u ranks differ\n", rows, attention_rows, chunks, exchanges,
		gathers, selected_rounds, mismatched);
	cudaFree(reference_received);
	cudaFree(reference_bitmaps);
	cudaFree(reference_selected);
	XrTeardown();
}

int main(void)
{
	XrCase(600u, 600u, 1u);
	XrCase(600u, 17u, 1u);
	XrCase(600u, 16u, 1u);
	XrCase(600u, 5u, 0u);
	XrCase(100u, 100u, 1u);
	XrCase(100u, 1u, 0u);
	XrCheck(xr_fails == 0u, "no round failed");
	if ( xr_failures != 0 )
	{
		printf("FAIL k3_index_exchange_cuda: %d failures\n", xr_failures);
		return 1;
	}
	printf("PASS k3_index_exchange_cuda: chunks with a row per rank exchange candidates by row owner, merge only their own rows and gather the selection; "
		"every rank ends with the gathered path's selection, ties included\n");
	return 0;
}
